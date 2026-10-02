# mojolearn benchmark board

Generated 2026-10-01T01:35:25Z from `board.json` (schema `mojolearn-bench-board/1`).

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
| mojolearn | 0.8.31 (wheel mojolearn-0.8.31-py3-none-macosx_11_0_arm64.whl, sha256 7f69892187dd96abb816d4b1835565b434a88f9f0fff403078811256d3c2879b) |
| script commit | 26cdf0767744c104ac7486f650bd2774e2c90cfc |
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

Races: 1 planned, 241 done, 5 failed, 0 pending. Cells: 807 (HOST-MEMORY 2, MODE-MISMATCH 22, REFUSED 111, ok 672).

Inference cells: 504 (REFUSED 78, ok 426).

## Quality at a glance

Per lane and dataset: our FAST value, our IDENTICAL value, our CPU IDENTICAL value, and each opponent's.

| family | lane | dataset | metric | ours FAST | ours IDENTICAL | ours CPU | opponents |
|---|---|---|---|---|---|---|---|
| algos | adaboost-clf | taxi | accuracy (higher is better) | 0.765230 | 0.765230 | - | sklearn-cpu 0.765360 |
| algos | adaboost-clf | taxi | logloss (lower is better) | 0.543227 | 0.543227 | - | sklearn-cpu 0.540565 |
| algos | adaboost-reg | istella | r2 (higher is better) | 0.238879 | 0.234086 | - | sklearn-cpu 0.167622 |
| algos | adaboost-reg | istella | rmse (lower is better) | 0.728764 | 0.731056 | - | sklearn-cpu 0.762115 |
| algos | adaboost-reg | taxi | r2 (higher is better) | 0.216396 | -0.417994 | - | sklearn-cpu 0.563927 |
| algos | adaboost-reg | taxi | rmse (lower is better) | 14.098896 | 18.965923 | - | sklearn-cpu 10.517589 |
| algos | adafactor | synthetic | rel_fro_vs_torch_eager_fp32 | 1.45e-08 | 4.324e-05 | - | torch-eager-fp32 -; torch-compile-fp32 0.000000 |
| algos | adagrad | synthetic | rel_fro_vs_torch_eager_fp32 | 3.297e-09 | 3.297e-09 | - | torch-eager-fp32 -; torch-compile-fp32 0.000000 |
| algos | adam | synthetic | rel_fro_vs_torch_eager_fp32 | 3.34e-08 | 3.339e-08 | - | torch-eager-fp32 -; torch-compile-fp32 3.576e-08 |
| algos | adamax | synthetic | rel_fro_vs_torch_eager_fp32 | 4.267e-09 | 4.267e-09 | - | torch-eager-fp32 -; torch-compile-fp32 7.477e-09 |
| algos | adamw | synthetic | rel_fro_vs_torch_eager_fp32 | 3.34e-08 | 3.339e-08 | - | torch-eager-fp32 -; torch-compile-fp32 3.577e-08 |
| algos | additive-chi2 | istella | kernel_rel_error (lower is better) | 0.087730 | 0.087730 | - | sklearn-cpu 0.087730 |
| algos | additive-chi2 | taxi | kernel_rel_error (lower is better) | 0.093892 | 0.093892 | - | sklearn-cpu 0.093892 |
| algos | affinity-prop | istella | n_clusters | 342 | 342 | - | sklearn-cpu 342 |
| algos | affinity-prop | istella | silhouette (higher is better) | 0.089763 | 0.089763 | - | sklearn-cpu 0.089763 |
| algos | affinity-prop | istella | ari_vs_ours (1 is our partition exactly) | 1.000000 | - | - | sklearn-cpu 1.000000 |
| algos | affinity-prop | taxi | n_clusters | 272 | 272 | - | sklearn-cpu 272 |
| algos | affinity-prop | taxi | silhouette (higher is better) | 0.184644 | 0.184644 | - | sklearn-cpu 0.184644 |
| algos | affinity-prop | taxi | ari_vs_ours (1 is our partition exactly) | 1.000000 | - | - | sklearn-cpu 1.000000 |
| algos | als | taxi-zones | recall_at_10 (higher is better) | - | - | - | implicit-cpu 0.056786 |
| algos | auto-theta | synthetic | forecast_rmse (lower is better) | 1.438072 | 1.438912 | - | statsforecast-cpu 1.437804 |
| algos | auto-theta | taxi-hourly | forecast_rmse (lower is better) | 49.313360 | 49.054628 | - | statsforecast-cpu 49.273467 |
| algos | autoarima | synthetic | forecast_rmse (lower is better) | - | - | - | statsforecast-cpu 17.546480 |
| algos | autoarima | taxi-hourly | forecast_rmse (lower is better) | - | - | - | statsforecast-cpu 68.211648 |
| algos | bagging-clf | istella | accuracy (higher is better) | 0.941670 | 0.941670 | - | sklearn-cpu 0.941440 |
| algos | bagging-clf | istella | logloss (lower is better) | 0.149399 | 0.149399 | - | sklearn-cpu 0.152069 |
| algos | bagging-clf | taxi | accuracy (higher is better) | 0.767570 | 0.767570 | - | sklearn-cpu 0.767650 |
| algos | bagging-clf | taxi | logloss (lower is better) | 0.530450 | 0.530450 | - | sklearn-cpu 0.530078 |
| algos | bagging-reg | istella | r2 (higher is better) | 0.522033 | 0.522033 | - | sklearn-cpu 0.518475 |
| algos | bagging-reg | istella | rmse (lower is better) | 0.577510 | 0.577510 | - | sklearn-cpu 0.579656 |
| algos | bagging-reg | taxi | r2 (higher is better) | 0.918767 | 0.918767 | - | sklearn-cpu 0.938805 |
| algos | bagging-reg | taxi | rmse (lower is better) | 4.539457 | 4.539457 | - | sklearn-cpu 3.939993 |
| algos | bayesian-gmm | taxi | mean_log_likelihood (higher is better) | 4.895806 | 4.895581 | - | sklearn-cpu 6.178321 |
| algos | bernoulli-nb | istella | accuracy (higher is better) | 0.794050 | 0.794050 | - | sklearn-cpu 0.794050 |
| algos | bernoulli-nb | istella | logloss (lower is better) | 5.350576 | 5.350625 | - | sklearn-cpu 4.278741 |
| algos | bernoulli-nb | taxi | accuracy (higher is better) | 0.755560 | 0.755560 | - | sklearn-cpu 0.755560 |
| algos | bernoulli-nb | taxi | logloss (lower is better) | 0.557803 | 0.557803 | - | sklearn-cpu 0.557802 |
| algos | binarizer | istella | max_abs_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | binarizer | istella | rel_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | binarizer | taxi | max_abs_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | binarizer | taxi | rel_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | bootstrap | istella | ci_endpoint_diff_over_width_vs_scipy | 0.027369 | 0.027369 | - | scipy-cpu - |
| algos | bootstrap | istella | ci_high | 0.294850 | 0.294850 | - | scipy-cpu 0.295500 |
| algos | bootstrap | istella | ci_low | 0.271250 | 0.271250 | - | scipy-cpu 0.271750 |
| algos | bootstrap | istella | standard_error | 0.005987 | 0.005987 | - | scipy-cpu 0.006044 |
| algos | bootstrap | taxi | ci_endpoint_diff_over_width_vs_scipy | 0.005017 | 0.005021 | - | scipy-cpu - |
| algos | bootstrap | taxi | ci_high | 18.717377 | 18.717379 | - | scipy-cpu 18.715050 |
| algos | bootstrap | taxi | ci_low | 18.249411 | 18.249411 | - | scipy-cpu 18.251265 |
| algos | bootstrap | taxi | standard_error | 0.117837 | 0.117837 | - | scipy-cpu 0.117562 |
| algos | categorical-nb | istella | accuracy (higher is better) | 0.838850 | 0.838850 | - | sklearn-cpu 0.838850 |
| algos | categorical-nb | istella | logloss (lower is better) | 0.412625 | 0.412625 | - | sklearn-cpu 0.412625 |
| algos | categorical-nb | taxi | accuracy (higher is better) | 0.765850 | 0.765850 | - | sklearn-cpu 0.765850 |
| algos | categorical-nb | taxi | logloss (lower is better) | 0.538866 | 0.538866 | - | sklearn-cpu 0.538866 |
| algos | cca | istella | mean_canonical_corr | 0.998053 | 0.998053 | - | sklearn-cpu 0.999568 |
| algos | cca | taxi | mean_canonical_corr | 0.576863 | 0.576863 | - | sklearn-cpu 0.576863 |
| algos | cholesky | synthetic | relative_residual | 1.659e-07 | 2.9e-07 | - | numpy-cpu 3.928e-08; torch-gpu 5.449e-07 |
| algos | classical-mds | istella | trustworthiness_k15 (higher is better, 1 at most) | 0.830548 | - | - | sklearn-cpu 0.830548 |
| algos | classical-mds | taxi | trustworthiness_k15 (higher is better, 1 at most) | 0.765649 | - | - | sklearn-cpu 0.765649 |
| algos | complement-nb | istella | accuracy (higher is better) | 0.849360 | 0.849190 | - | sklearn-cpu 0.849350 |
| algos | complement-nb | istella | logloss (lower is better) | 3.762528 | 3.766564 | - | sklearn-cpu 3.174763 |
| algos | complement-nb | taxi | accuracy (higher is better) | 0.678030 | 0.677000 | - | sklearn-cpu 0.678030 |
| algos | complement-nb | taxi | logloss (lower is better) | 0.715492 | 0.715601 | - | sklearn-cpu 0.715493 |
| algos | complement-nb | text | accuracy (higher is better) | 0.983067 | 0.983067 | - | sklearn-cpu 0.983067 |
| algos | complement-nb | text | logloss (lower is better) | 0.559491 | 0.559491 | - | sklearn-cpu 0.557285 |
| algos | connected-components | istella | ari_vs_networkx | 1.000000 | 1.000000 | - | networkx-cpu - |
| algos | connected-components | istella | n_components | 81 | 81 | - | networkx-cpu 81 |
| algos | connected-components | taxi | ari_vs_networkx | 1.000000 | 1.000000 | - | networkx-cpu - |
| algos | connected-components | taxi | n_components | 588 | 588 | - | networkx-cpu 588 |
| algos | cross-val-score | istella | max_fold_score_diff_vs_sklearn | 0.015390 | 0.017412 | - | sklearn-cpu - |
| algos | cross-val-score | istella | mean_r2 | 0.331418 | 0.333971 | - | sklearn-cpu 0.319779 |
| algos | cross-val-score | taxi | max_fold_score_diff_vs_sklearn | 4.172e-06 | 5.007e-06 | - | sklearn-cpu - |
| algos | cross-val-score | taxi | mean_r2 | 0.937955 | 0.937955 | - | sklearn-cpu 0.937956 |
| algos | croston-optimized | synthetic | forecast_rmse (lower is better) | 1.675539 | 1.675538 | - | statsforecast-cpu 1.675539 |
| algos | croston-optimized | taxi-hourly | forecast_rmse (lower is better) | 1.398258 | 1.398253 | - | statsforecast-cpu 1.398240 |
| algos | croston-sba | synthetic | forecast_rmse (lower is better) | 1.674465 | 1.674465 | - | statsforecast-cpu 1.674465 |
| algos | croston-sba | taxi-hourly | forecast_rmse (lower is better) | 1.386677 | 1.386677 | - | statsforecast-cpu 1.386677 |
| algos | croston | synthetic | forecast_rmse (lower is better) | 1.674840 | 1.674840 | - | statsforecast-cpu 1.674840 |
| algos | croston | taxi-hourly | forecast_rmse (lower is better) | 1.390261 | 1.390261 | - | statsforecast-cpu 1.390261 |
| algos | damped-ets | synthetic | forecast_rmse (lower is better) | 13.931185 | 13.931153 | - | statsmodels-cpu 26.588704; statsforecast-cpu 13.945986 |
| algos | damped-ets | taxi-hourly | forecast_rmse (lower is better) | 96.690681 | 96.690449 | - | statsmodels-cpu 196.928662; statsforecast-cpu 96.685568 |
| algos | dart-reg | istella | r2 (higher is better) | 0.550733 | 0.550733 | - | lightgbm-cpu 0.564663; xgboost-cpu - |
| algos | dart-reg | istella | rmse (lower is better) | 0.559903 | 0.559903 | - | lightgbm-cpu 0.551154; xgboost-cpu - |
| algos | dart-reg | taxi | r2 (higher is better) | 0.925497 | 0.925497 | - | lightgbm-cpu 0.926249; xgboost-cpu 0.926049 |
| algos | dart-reg | taxi | rmse (lower is better) | 4.347340 | 4.347340 | - | lightgbm-cpu 4.325358; xgboost-cpu 4.331203 |
| algos | dart | istella | accuracy (higher is better) | 0.948720 | 0.948720 | - | lightgbm-cpu 0.951860; xgboost-cpu - |
| algos | dart | istella | logloss (lower is better) | 0.134126 | 0.134126 | - | lightgbm-cpu 0.123740; xgboost-cpu - |
| algos | dart | taxi | accuracy (higher is better) | 0.768340 | 0.768340 | - | lightgbm-cpu 0.768190; xgboost-cpu 0.768000 |
| algos | dart | taxi | logloss (lower is better) | 0.529060 | 0.529060 | - | lightgbm-cpu 0.529108; xgboost-cpu 0.529540 |
| algos | decision-tree-clf | istella | accuracy (higher is better) | 0.935000 | 0.935000 | - | sklearn-cpu 0.934410 |
| algos | decision-tree-clf | istella | logloss (lower is better) | 0.758303 | 0.758303 | - | sklearn-cpu 0.791706 |
| algos | decision-tree-clf | taxi | accuracy (higher is better) | 0.756300 | 0.756300 | - | sklearn-cpu 0.756560 |
| algos | decision-tree-clf | taxi | logloss (lower is better) | 1.202243 | 1.202243 | - | sklearn-cpu 1.149550 |
| algos | decision-tree-reg | istella | r2 (higher is better) | 0.379438 | 0.379438 | - | sklearn-cpu 0.373177 |
| algos | decision-tree-reg | istella | rmse (lower is better) | 0.658041 | 0.658041 | - | sklearn-cpu 0.661352 |
| algos | decision-tree-reg | taxi | r2 (higher is better) | 0.862608 | 0.862608 | - | sklearn-cpu 0.891507 |
| algos | decision-tree-reg | taxi | rmse (lower is better) | 5.903617 | 5.903617 | - | sklearn-cpu 5.246104 |
| algos | dict-learning | istella | component_sparsity | 0.086364 | 0.086364 | - | sklearn-cpu 0.086364 |
| algos | dict-learning | istella | relative_reconstruction_error (lower is better) | 0.652121 | 0.652121 | - | sklearn-cpu 0.652121 |
| algos | dict-learning | taxi | component_sparsity | 0.000000 | 0.000000 | - | sklearn-cpu 0.000000 |
| algos | dict-learning | taxi | relative_reconstruction_error (lower is better) | 0.459173 | 0.459169 | - | sklearn-cpu 0.460601 |
| algos | dynamic-optimized-theta | synthetic | forecast_rmse (lower is better) | 1.435976 | 1.436278 | - | statsforecast-cpu 1.436045 |
| algos | dynamic-optimized-theta | taxi-hourly | forecast_rmse (lower is better) | 49.083557 | 49.086628 | - | statsforecast-cpu 49.314797 |
| algos | dynamic-theta | synthetic | forecast_rmse (lower is better) | 1.437165 | 1.437256 | - | statsforecast-cpu 1.437262 |
| algos | dynamic-theta | taxi-hourly | forecast_rmse (lower is better) | 49.100462 | 49.101249 | - | statsforecast-cpu 49.269843 |
| algos | eigh | synthetic | max_eigenvalue_error | - | - | - | numpy-cpu 3.49e-08; torch-gpu - |
| algos | eigh | synthetic | relative_residual | - | - | - | numpy-cpu 2.824e-08; torch-gpu - |
| algos | elliptic-envelope | istella | fraction_flagged | - | - | - | sklearn-cpu 0.091570 |
| algos | elliptic-envelope | istella | jaccard_vs_sklearn | - | - | - | sklearn-cpu 1.000000 |
| algos | elliptic-envelope | taxi | fraction_flagged | 0.102370 | - | - | sklearn-cpu 0.102470 |
| algos | elliptic-envelope | taxi | jaccard_vs_sklearn | 0.963574 | - | - | sklearn-cpu 1.000000 |
| algos | factor-analysis | istella | mean_log_likelihood (higher is better) | - | - | - | sklearn-cpu 98.122830 |
| algos | factor-analysis | taxi | mean_log_likelihood (higher is better) | -14.823632 | -14.823632 | - | sklearn-cpu -14.823723 |
| algos | fastica | istella | mean_abs_excess_kurtosis | 356.288699 | 356.288855 | - | sklearn-cpu 922.531077 |
| algos | fastica | taxi | mean_abs_excess_kurtosis | 13.209463 | 13.204824 | - | sklearn-cpu 13.764740 |
| algos | garch | synthetic | mean_llf (higher is better) | -1938.224617 | -1938.223490 | - | arch-cpu -1938.221004 |
| algos | garch | taxi-hourly | mean_llf (higher is better) | -1132.954899 | -1132.712756 | - | arch-cpu -1129.806866 |
| algos | gaussian-nb | istella | accuracy (higher is better) | 0.876570 | 0.876530 | - | sklearn-cpu 0.876530 |
| algos | gaussian-nb | istella | logloss (lower is better) | 3.574405 | 3.574225 | - | sklearn-cpu 3.417392 |
| algos | gaussian-nb | taxi | accuracy (higher is better) | 0.719820 | 0.719900 | - | sklearn-cpu 0.719900 |
| algos | gaussian-nb | taxi | logloss (lower is better) | 1.132247 | 1.133898 | - | sklearn-cpu 1.133898 |
| algos | gaussian-rp | istella | mean_abs_distortion | 0.680693 | 0.680693 | - | sklearn-cpu 0.177966 |
| algos | gaussian-rp | taxi | mean_abs_distortion | 0.345752 | 0.345752 | - | sklearn-cpu 0.339791 |
| algos | gru-clf | synthetic | accuracy (higher is better) | 0.971842 | 0.971842 | - | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | gru-clf | synthetic | logloss (lower is better) | 0.065835 | 0.065835 | - | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | gru-clf | taxi-hourly | accuracy (higher is better) | 0.865668 | 0.865668 | - | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | gru-clf | taxi-hourly | logloss (lower is better) | 0.305841 | 0.305841 | - | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | gru-reg | synthetic | r2 (higher is better) | 0.981946 | 0.981946 | - | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | gru-reg | synthetic | rmse (lower is better) | 0.155672 | 0.155672 | - | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | gru-reg | taxi-hourly | r2 (higher is better) | 0.748219 | 0.748219 | - | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | gru-reg | taxi-hourly | rmse (lower is better) | 0.544182 | 0.544182 | - | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | incremental-pca | istella | explained_variance_fraction | 0.999994 | 0.999994 | - | sklearn-cpu 1.000000 |
| algos | incremental-pca | taxi | explained_variance_fraction | 0.999995 | 0.999995 | - | sklearn-cpu 0.999995 |
| algos | isomap | istella | trustworthiness_k15 (higher is better, 1 at most) | - | - | - | sklearn-cpu 0.853294 |
| algos | isomap | taxi | trustworthiness_k15 (higher is better, 1 at most) | - | - | - | sklearn-cpu 0.771827 |
| algos | iterative-imputer | istella | masked_rmse | 799040.699151 | 799013.020330 | - | sklearn-cpu 802428.936225 |
| algos | iterative-imputer | istella | max_abs_diff_vs_sklearn | 9.974e+06 | 1.011e+07 | - | sklearn-cpu - |
| algos | iterative-imputer | taxi | masked_rmse | 4.693971 | 4.693848 | - | sklearn-cpu 4.693973 |
| algos | iterative-imputer | taxi | max_abs_diff_vs_sklearn | 0.0003719 | 0.044070 | - | sklearn-cpu - |
| algos | kbins | istella | max_abs_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | kbins | istella | rel_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | kbins | taxi | max_abs_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | kbins | taxi | rel_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | kernel-pca | istella | subspace_cos_vs_sklearn | 1.000000 | 1.000000 | - | sklearn-cpu 1.000000 |
| algos | kernel-pca | taxi | subspace_cos_vs_sklearn | 1.000000 | 1.000000 | - | sklearn-cpu 1.000000 |
| algos | knn-imputer | istella | masked_rmse | nan | nan | - | sklearn-cpu - |
| algos | knn-imputer | taxi | masked_rmse | 6.151696 | 6.151696 | - | sklearn-cpu 5.256719 |
| algos | knn-imputer | taxi | max_abs_diff_vs_sklearn | 29.000000 | 29.000000 | - | sklearn-cpu - |
| algos | label-binarizer | istella | max_abs_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | label-binarizer | istella | rel_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | label-binarizer | taxi | max_abs_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | label-binarizer | taxi | rel_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | label-encoder | istella | max_abs_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | label-encoder | istella | rel_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | label-encoder | taxi | max_abs_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | label-encoder | taxi | rel_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | label-propagation | istella | accuracy (higher is better) | 0.125400 | 0.125400 | - | sklearn-cpu 0.905500 |
| algos | label-propagation | taxi | accuracy (higher is better) | 0.699200 | 0.701600 | - | sklearn-cpu 0.701600 |
| algos | label-spreading | istella | accuracy (higher is better) | 0.156000 | 0.156950 | - | sklearn-cpu 0.904450 |
| algos | label-spreading | taxi | accuracy (higher is better) | 0.676400 | 0.676400 | - | sklearn-cpu 0.676400 |
| algos | layernorm | synthetic | max_rel_diff_vs_torch_eager_fp32 | 0.089719 | 0.089719 | - | torch-eager-fp32 -; torch-compile-fp32 0.066421; torch-eager-bf16 0.000000; torch-compile-bf16 0.066421 |
| algos | layernorm | synthetic | rel_fro_vs_torch_eager_fp32 | 2.264e-07 | 2.264e-07 | - | torch-eager-fp32 -; torch-compile-fp32 2.25e-07; torch-eager-bf16 0.000000; torch-compile-bf16 2.25e-07 |
| algos | lda-clf | istella | accuracy (higher is better) | 0.909010 | 0.911660 | - | sklearn-cpu 0.901130 |
| algos | lda-clf | istella | logloss (lower is better) | 0.264392 | 0.247483 | - | sklearn-cpu 0.449237 |
| algos | lda-clf | taxi | accuracy (higher is better) | 0.762580 | 0.762580 | - | sklearn-cpu 0.762530 |
| algos | lda-clf | taxi | logloss (lower is better) | 0.539743 | 0.539749 | - | sklearn-cpu 0.539767 |
| algos | lda | taxi-zones | perplexity | 45.221811 | 45.221819 | - | sklearn-cpu 44.920445 |
| algos | lda | text | perplexity | - | - | - | sklearn-cpu 266.926844 |
| algos | lle | istella | trustworthiness_k15 (higher is better, 1 at most) | - | - | - | sklearn-cpu 0.856249 |
| algos | lle | taxi | trustworthiness_k15 (higher is better, 1 at most) | - | - | - | sklearn-cpu 0.770758 |
| algos | lof | istella | fraction_flagged | 0.010275 | 0.010805 | - | sklearn-cpu 0.033610 |
| algos | lof | istella | jaccard_vs_sklearn | 0.054168 | 0.054112 | - | sklearn-cpu 1.000000 |
| algos | lof | taxi | fraction_flagged | 0.008960 | 0.008960 | - | sklearn-cpu 0.008960 |
| algos | lof | taxi | jaccard_vs_sklearn | 1.000000 | 1.000000 | - | sklearn-cpu 1.000000 |
| algos | louvain | istella | modularity | 0.909755 | 0.909755 | - | networkx-cpu 0.908460 |
| algos | louvain | istella | n_communities | 39 | 39 | - | networkx-cpu 40 |
| algos | louvain | taxi | modularity | 0.941172 | 0.941172 | - | networkx-cpu 0.940781 |
| algos | louvain | taxi | n_communities | 58 | 58 | - | networkx-cpu 56 |
| algos | lstm-clf | synthetic | accuracy (higher is better) | 0.968696 | 0.968696 | - | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | lstm-clf | synthetic | logloss (lower is better) | 0.072441 | 0.072441 | - | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | lstm-clf | taxi-hourly | accuracy (higher is better) | 0.868218 | 0.868218 | - | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | lstm-clf | taxi-hourly | logloss (lower is better) | 0.299901 | 0.299901 | - | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | lstm-reg | synthetic | r2 (higher is better) | 0.981013 | 0.981013 | - | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | lstm-reg | synthetic | rmse (lower is better) | 0.159641 | 0.159641 | - | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | lstm-reg | taxi-hourly | r2 (higher is better) | 0.751679 | 0.751679 | - | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | lstm-reg | taxi-hourly | rmse (lower is better) | 0.540429 | 0.540429 | - | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | lstsq | istella | relative_residual | 0.849956 | 0.849957 | - | numpy-cpu 0.873341; torch-gpu - |
| algos | lstsq | taxi | relative_residual | 0.756366 | 0.756366 | - | numpy-cpu 0.756366; torch-gpu - |
| algos | lu-factor | synthetic | relative_residual | 3.249e-06 | 3.249e-06 | - | scipy-cpu 3.246e-06; torch-gpu 8.214e-07 |
| algos | lu-solve | synthetic | relative_residual | 3.249e-06 | 3.249e-06 | - | numpy-cpu 3.259e-08; torch-gpu 8.214e-07 |
| algos | maxabs-scaler | istella | max_abs_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | maxabs-scaler | istella | rel_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | maxabs-scaler | taxi | max_abs_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | maxabs-scaler | taxi | rel_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | mb-dict-learning | istella | component_sparsity | 0.086364 | 0.086364 | - | sklearn-cpu 0.086364 |
| algos | mb-dict-learning | istella | relative_reconstruction_error (lower is better) | 0.648339 | 0.648338 | - | sklearn-cpu 0.644014 |
| algos | mb-dict-learning | taxi | component_sparsity | 0.000000 | 0.000000 | - | sklearn-cpu 0.000000 |
| algos | mb-dict-learning | taxi | relative_reconstruction_error (lower is better) | 0.496685 | 0.496685 | - | sklearn-cpu 0.477049 |
| algos | mb-sparse-pca | istella | component_sparsity | 0.130682 | 0.130682 | - | sklearn-cpu 0.130682 |
| algos | mb-sparse-pca | istella | relative_reconstruction_error (lower is better) | 0.705389 | 0.705389 | - | sklearn-cpu 0.705389 |
| algos | mb-sparse-pca | taxi | component_sparsity | 0.022727 | 0.022727 | - | sklearn-cpu 0.022727 |
| algos | mb-sparse-pca | taxi | relative_reconstruction_error (lower is better) | 0.275935 | 0.275935 | - | sklearn-cpu 0.275933 |
| algos | mds | istella | trustworthiness_k15 (higher is better, 1 at most) | - | - | - | sklearn-cpu 0.580239 |
| algos | mds | taxi | trustworthiness_k15 (higher is better, 1 at most) | - | - | - | sklearn-cpu 0.604142 |
| algos | min-cov-det | istella | n_features | - | - | - | sklearn-cpu 220 |
| algos | min-cov-det | taxi | n_features | 11 | 11 | - | sklearn-cpu 11 |
| algos | min-cov-det | taxi | rel_diff_vs_sklearn | 0.205603 | 0.205603 | - | sklearn-cpu - |
| algos | minmax-scaler | istella | max_abs_diff_vs_sklearn | 1.192e-07 | 0.000000 | - | sklearn-cpu - |
| algos | minmax-scaler | istella | rel_diff_vs_sklearn | 1.694e-08 | 0.000000 | - | sklearn-cpu - |
| algos | minmax-scaler | taxi | max_abs_diff_vs_sklearn | 5.96e-08 | 0.000000 | - | sklearn-cpu - |
| algos | minmax-scaler | taxi | rel_diff_vs_sklearn | 2.63e-08 | 0.000000 | - | sklearn-cpu - |
| algos | mlp-clf | istella | accuracy (higher is better) | 0.944410 | 0.944270 | - | sklearn-cpu 0.943760 |
| algos | mlp-clf | istella | logloss (lower is better) | 0.136065 | 0.136955 | - | sklearn-cpu 0.136831 |
| algos | mlp-clf | taxi | accuracy (higher is better) | 0.767780 | 0.767810 | - | sklearn-cpu 0.767830 |
| algos | mlp-clf | taxi | logloss (lower is better) | 0.530482 | 0.530506 | - | sklearn-cpu 0.530444 |
| algos | mlp-reg | istella | r2 (higher is better) | 0.527265 | 0.526405 | - | sklearn-cpu 0.524815 |
| algos | mlp-reg | istella | rmse (lower is better) | 0.574340 | 0.574862 | - | sklearn-cpu 0.575826 |
| algos | mlp-reg | taxi | r2 (higher is better) | 0.931976 | 0.931981 | - | sklearn-cpu 0.929613 |
| algos | mlp-reg | taxi | rmse (lower is better) | 4.154000 | 4.153868 | - | sklearn-cpu 4.225537 |
| algos | moe | synthetic | max_rel_diff_vs_torch_eager_fp32 | 0.084128 | 0.070736 | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 22208.605483; torch-compile-bf16 22208.605483 |
| algos | moe | synthetic | rel_fro_vs_torch_eager_fp32 | 3.334e-07 | 3.02e-07 | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 0.055222; torch-compile-bf16 0.055222 |
| algos | multilabel-binarizer | istella | max_abs_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | multilabel-binarizer | istella | rel_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | multilabel-binarizer | taxi | max_abs_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | multilabel-binarizer | taxi | rel_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | multinomial-nb | istella | accuracy (higher is better) | 0.853620 | 0.853560 | - | sklearn-cpu 0.853620 |
| algos | multinomial-nb | istella | logloss (lower is better) | 3.628572 | 3.630944 | - | sklearn-cpu 3.087499 |
| algos | multinomial-nb | taxi | accuracy (higher is better) | 0.723160 | 0.723260 | - | sklearn-cpu 0.723160 |
| algos | multinomial-nb | taxi | logloss (lower is better) | 0.590725 | 0.589979 | - | sklearn-cpu 0.590725 |
| algos | multinomial-nb | text | accuracy (higher is better) | 0.983067 | 0.983067 | - | sklearn-cpu 0.983067 |
| algos | multinomial-nb | text | logloss (lower is better) | 0.559529 | 0.559529 | - | sklearn-cpu 0.557319 |
| algos | nadam | synthetic | rel_fro_vs_torch_eager_fp32 | 2.939e-08 | 2.939e-08 | - | torch-eager-fp32 -; torch-compile-fp32 4.102e-08 |
| algos | nearest-centroid | istella | accuracy (higher is better) | 0.852610 | 0.852610 | - | sklearn-cpu 0.852610 |
| algos | nearest-centroid | istella | logloss (lower is better) | 4.299222 | 4.299221 | - | sklearn-cpu 4.117692 |
| algos | nearest-centroid | taxi | accuracy (higher is better) | 0.666750 | 0.666750 | - | sklearn-cpu 0.666750 |
| algos | nearest-centroid | taxi | logloss (lower is better) | 0.782162 | 0.782162 | - | sklearn-cpu 0.781690 |
| algos | nmf | istella | relative_reconstruction_error (lower is better) | 0.325174 | 0.325174 | - | sklearn-cpu 0.325399 |
| algos | nmf | taxi | relative_reconstruction_error (lower is better) | 0.091156 | 0.091156 | - | sklearn-cpu 0.091155 |
| algos | normalizer | istella | max_abs_diff_vs_sklearn | 3.576e-07 | 3.576e-07 | - | sklearn-cpu - |
| algos | normalizer | istella | rel_diff_vs_sklearn | 5.674e-08 | 5.811e-08 | - | sklearn-cpu - |
| algos | normalizer | taxi | max_abs_diff_vs_sklearn | 1.192e-07 | 1.192e-07 | - | sklearn-cpu - |
| algos | normalizer | taxi | rel_diff_vs_sklearn | 3.283e-08 | 3.114e-08 | - | sklearn-cpu - |
| algos | ocsvm | istella | fraction_flagged | 0.078300 | 0.078300 | - | sklearn-cpu 0.078300 |
| algos | ocsvm | istella | jaccard_vs_sklearn | 1.000000 | 1.000000 | - | sklearn-cpu 1.000000 |
| algos | ocsvm | taxi | fraction_flagged | 0.136100 | 0.136100 | - | sklearn-cpu 0.136100 |
| algos | ocsvm | taxi | jaccard_vs_sklearn | 1.000000 | 1.000000 | - | sklearn-cpu 1.000000 |
| algos | onehot | istella | max_abs_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | onehot | istella | rel_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | onehot | taxi | max_abs_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | onehot | taxi | rel_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | optics | istella | n_clusters | 20 | 20 | - | sklearn-cpu 20 |
| algos | optics | istella | silhouette (higher is better) | -0.287356 | -0.287356 | - | sklearn-cpu -0.285806 |
| algos | optics | istella | ari_vs_ours (1 is our partition exactly) | 1.000000 | - | - | sklearn-cpu 0.984603 |
| algos | optics | taxi | n_clusters | 127 | 127 | - | sklearn-cpu - |
| algos | optics | taxi | silhouette (higher is better) | -0.353359 | -0.353359 | - | sklearn-cpu - |
| algos | optics | taxi | ari_vs_ours (1 is our partition exactly) | 1.000000 | - | - | sklearn-cpu - |
| algos | optimized-theta | synthetic | forecast_rmse (lower is better) | 1.439709 | 1.438855 | - | statsforecast-cpu 1.437815 |
| algos | optimized-theta | taxi-hourly | forecast_rmse (lower is better) | 49.152331 | 49.150860 | - | statsforecast-cpu 49.356608 |
| algos | ordinal | istella | max_abs_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | ordinal | istella | rel_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | ordinal | taxi | max_abs_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | ordinal | taxi | rel_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | pagerank | istella | l1_vs_networkx | 6.429e-08 | 5.89e-08 | - | networkx-cpu - |
| algos | pagerank | istella | sum | 1.000000 | 1.000000 | - | networkx-cpu 1.000000 |
| algos | pagerank | taxi | l1_vs_networkx | 8.411e-08 | 6.917e-08 | - | networkx-cpu - |
| algos | pagerank | taxi | sum | 1.000000 | 1.000000 | - | networkx-cpu 1.000000 |
| algos | permutation-test | istella | pvalue | - | - | - | scipy-cpu 0.184400 |
| algos | permutation-test | istella | statistic | - | - | - | scipy-cpu 0.011200 |
| algos | permutation-test | taxi | pvalue | - | - | - | scipy-cpu 0.001000 |
| algos | permutation-test | taxi | statistic | - | - | - | scipy-cpu -0.576708 |
| algos | pls-canonical | istella | mean_canonical_corr | 0.875340 | 0.875340 | - | sklearn-cpu 0.875341 |
| algos | pls-canonical | taxi | mean_canonical_corr | 0.559206 | 0.559206 | - | sklearn-cpu 0.559206 |
| algos | pls | istella | r2 (higher is better) | 0.289870 | 0.289870 | - | sklearn-cpu 0.289870 |
| algos | pls | istella | rmse (lower is better) | 0.703930 | 0.703930 | - | sklearn-cpu 0.703930 |
| algos | pls | taxi | r2 (higher is better) | 0.905216 | 0.905216 | - | sklearn-cpu 0.905216 |
| algos | pls | taxi | rmse (lower is better) | 4.903469 | 4.903469 | - | sklearn-cpu 4.903467 |
| algos | poly-count-sketch | istella | kernel_rel_error (lower is better) | 0.040849 | 0.040849 | - | sklearn-cpu 0.040849 |
| algos | poly-count-sketch | taxi | kernel_rel_error (lower is better) | 0.096596 | 0.096596 | - | sklearn-cpu 0.096596 |
| algos | poly-features | istella | max_abs_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | poly-features | istella | rel_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | poly-features | taxi | max_abs_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | poly-features | taxi | rel_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | power-transformer | taxi | max_abs_diff_vs_sklearn | 0.062769 | 0.191760 | - | sklearn-cpu - |
| algos | power-transformer | taxi | rel_diff_vs_sklearn | 0.0005406 | 0.025551 | - | sklearn-cpu - |
| algos | prophet | synthetic | forecast_rmse (lower is better) | 1.015060 | 1.015150 | - | prophet-cpu 1.015319 |
| algos | prophet | taxi-hourly | forecast_rmse (lower is better) | 32.028614 | 32.049281 | - | prophet-cpu 32.035281 |
| algos | qda | istella | accuracy (higher is better) | 0.866090 | 0.885080 | - | sklearn-cpu 0.880530 |
| algos | qda | istella | logloss (lower is better) | 4.052899 | 3.969196 | - | sklearn-cpu 3.476976 |
| algos | qda | taxi | accuracy (higher is better) | 0.727020 | 0.727020 | - | sklearn-cpu 0.727220 |
| algos | qda | taxi | logloss (lower is better) | 1.061003 | 1.061003 | - | sklearn-cpu 1.059265 |
| algos | qr | istella | relative_gram_difference | 0.602233 | 0.037513 | - | numpy-cpu 2.472e-08; torch-gpu - |
| algos | qr | taxi | relative_gram_difference | 0.001996 | 0.001996 | - | numpy-cpu 3.024e-08; torch-gpu - |
| algos | quantile-transformer | istella | max_abs_diff_vs_sklearn | 5.96e-08 | 5.96e-08 | - | sklearn-cpu - |
| algos | quantile-transformer | istella | rel_diff_vs_sklearn | 2.764e-08 | 2.764e-08 | - | sklearn-cpu - |
| algos | quantile-transformer | taxi | max_abs_diff_vs_sklearn | 5.96e-08 | 5.96e-08 | - | sklearn-cpu - |
| algos | quantile-transformer | taxi | rel_diff_vs_sklearn | 2.295e-08 | 2.294e-08 | - | sklearn-cpu - |
| algos | radius-neighbors | istella | neighbors_total | - | - | - | sklearn-cpu 1220718 |
| algos | radius-neighbors | taxi | count_agreement_vs_sklearn | 1.000000 | 1.000000 | - | sklearn-cpu - |
| algos | radius-neighbors | taxi | neighbors_total | 31 | 31 | - | sklearn-cpu 31 |
| algos | randomized-svd | istella | relative_reconstruction_error (lower is better) | 0.0002359 | 0.0002359 | - | sklearn-cpu 0.0002359; torch-gpu - |
| algos | randomized-svd | taxi | relative_reconstruction_error (lower is better) | 0.027197 | 0.027197 | - | sklearn-cpu 0.027197; torch-gpu - |
| algos | resample | istella | max_mean_shift_over_std | 0.003203 | 0.003203 | - | sklearn-cpu 0.002552 |
| algos | resample | taxi | max_mean_shift_over_std | 0.002917 | 0.002917 | - | sklearn-cpu 0.002257 |
| algos | rfe | istella | jaccard_vs_sklearn | 0.833333 | 0.818182 | - | sklearn-cpu 1.000000 |
| algos | rfe | istella | n_selected | 110 | 110 | - | sklearn-cpu 110 |
| algos | rfe | taxi | jaccard_vs_sklearn | 1.000000 | 1.000000 | - | sklearn-cpu 1.000000 |
| algos | rfe | taxi | n_selected | 5 | 5 | - | sklearn-cpu 5 |
| algos | ridge-cv | istella | r2 (higher is better) | - | - | - | sklearn-cpu 0.328683 |
| algos | ridge-cv | istella | rmse (lower is better) | - | - | - | sklearn-cpu 0.684423 |
| algos | ridge-cv | taxi | r2 (higher is better) | - | - | - | sklearn-cpu 0.908988 |
| algos | ridge-cv | taxi | rmse (lower is better) | - | - | - | sklearn-cpu 4.804917 |
| algos | rmsprop | synthetic | rel_fro_vs_torch_eager_fp32 | 3.772e-08 | 3.772e-08 | - | torch-eager-fp32 -; torch-compile-fp32 0.000000 |
| algos | rnn-clf | synthetic | accuracy (higher is better) | 0.953559 | 0.953559 | - | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | rnn-clf | synthetic | logloss (lower is better) | 0.103698 | 0.103698 | - | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | rnn-clf | taxi-hourly | accuracy (higher is better) | 0.868056 | 0.868056 | - | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | rnn-clf | taxi-hourly | logloss (lower is better) | 0.304864 | 0.304864 | - | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | rnn-reg | synthetic | r2 (higher is better) | 0.977348 | 0.977348 | - | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | rnn-reg | synthetic | rmse (lower is better) | 0.174374 | 0.174374 | - | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | rnn-reg | taxi-hourly | r2 (higher is better) | 0.738796 | 0.738796 | - | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | rnn-reg | taxi-hourly | rmse (lower is better) | 0.554271 | 0.554271 | - | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | robust-scaler | istella | max_abs_diff_vs_sklearn | 0.0001221 | 0.0001221 | - | sklearn-cpu - |
| algos | robust-scaler | istella | rel_diff_vs_sklearn | 7.339e-09 | 7.339e-09 | - | sklearn-cpu - |
| algos | robust-scaler | taxi | max_abs_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | robust-scaler | taxi | rel_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | select-chi2 | istella | jaccard_vs_sklearn | 1.000000 | 0.981982 | - | sklearn-cpu 1.000000 |
| algos | select-chi2 | istella | n_selected | 110 | 110 | - | sklearn-cpu 110 |
| algos | select-chi2 | taxi | jaccard_vs_sklearn | 1.000000 | 1.000000 | - | sklearn-cpu 1.000000 |
| algos | select-chi2 | taxi | n_selected | 5 | 5 | - | sklearn-cpu 5 |
| algos | select-f-classif | istella | jaccard_vs_sklearn | 1.000000 | 1.000000 | - | sklearn-cpu 1.000000 |
| algos | select-f-classif | istella | n_selected | 110 | 110 | - | sklearn-cpu 110 |
| algos | select-f-classif | taxi | jaccard_vs_sklearn | 1.000000 | 1.000000 | - | sklearn-cpu 1.000000 |
| algos | select-f-classif | taxi | n_selected | 5 | 5 | - | sklearn-cpu 5 |
| algos | select-f-regression | istella | jaccard_vs_sklearn | 1.000000 | 1.000000 | - | sklearn-cpu 1.000000 |
| algos | select-f-regression | istella | n_selected | 110 | 110 | - | sklearn-cpu 110 |
| algos | select-f-regression | taxi | jaccard_vs_sklearn | 1.000000 | 1.000000 | - | sklearn-cpu 1.000000 |
| algos | select-f-regression | taxi | n_selected | 5 | 5 | - | sklearn-cpu 5 |
| algos | select-mutual-info-reg | istella | jaccard_vs_sklearn | 0.286550 | 0.896552 | - | sklearn-cpu 1.000000 |
| algos | select-mutual-info-reg | istella | n_selected | 110 | 110 | - | sklearn-cpu 110 |
| algos | select-mutual-info-reg | taxi | jaccard_vs_sklearn | 1.000000 | 1.000000 | - | sklearn-cpu 1.000000 |
| algos | select-mutual-info-reg | taxi | n_selected | 5 | 5 | - | sklearn-cpu 5 |
| algos | select-mutual-info | istella | jaccard_vs_sklearn | 0.929825 | 0.929825 | - | sklearn-cpu 1.000000 |
| algos | select-mutual-info | istella | n_selected | 110 | 110 | - | sklearn-cpu 110 |
| algos | select-mutual-info | taxi | jaccard_vs_sklearn | 0.428571 | 0.428571 | - | sklearn-cpu 1.000000 |
| algos | select-mutual-info | taxi | n_selected | 5 | 5 | - | sklearn-cpu 5 |
| algos | select-r-regression | istella | jaccard_vs_sklearn | 1.000000 | 1.000000 | - | sklearn-cpu 1.000000 |
| algos | select-r-regression | istella | n_selected | 110 | 110 | - | sklearn-cpu 110 |
| algos | select-r-regression | taxi | jaccard_vs_sklearn | 1.000000 | 1.000000 | - | sklearn-cpu 1.000000 |
| algos | select-r-regression | taxi | n_selected | 5 | 5 | - | sklearn-cpu 5 |
| algos | sgd | synthetic | rel_fro_vs_torch_eager_fp32 | 6.095e-10 | 6.095e-10 | - | torch-eager-fp32 -; torch-compile-fp32 0.000000 |
| algos | simple-imputer | istella | masked_rmse | - | 346849.129968 | - | sklearn-cpu 346849.129968 |
| algos | simple-imputer | istella | max_abs_diff_vs_sklearn | - | 0.000000 | - | sklearn-cpu - |
| algos | simple-imputer | taxi | masked_rmse | 5.985180 | 5.985180 | - | sklearn-cpu 5.985180 |
| algos | simple-imputer | taxi | max_abs_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | skewed-chi2 | istella | kernel_rel_error (lower is better) | 0.671898 | 0.671898 | - | sklearn-cpu 0.671899 |
| algos | skewed-chi2 | taxi | kernel_rel_error (lower is better) | 0.037749 | 0.037749 | - | sklearn-cpu 0.037749 |
| algos | sparse-coder | istella | max_abs_diff_vs_sklearn | 6.104e-05 | 6.104e-05 | - | sklearn-cpu - |
| algos | sparse-coder | istella | rel_diff_vs_sklearn | 1.11e-07 | 1.11e-07 | - | sklearn-cpu - |
| algos | sparse-coder | taxi | max_abs_diff_vs_sklearn | 1.049e-05 | 1.049e-05 | - | sklearn-cpu - |
| algos | sparse-coder | taxi | rel_diff_vs_sklearn | 9.976e-08 | 9.976e-08 | - | sklearn-cpu - |
| algos | sparse-pca | istella | component_sparsity | - | - | - | sklearn-cpu 0.305682 |
| algos | sparse-pca | istella | relative_reconstruction_error (lower is better) | - | - | - | sklearn-cpu 0.750210 |
| algos | sparse-pca | taxi | component_sparsity | 0.488636 | 0.488636 | - | sklearn-cpu 0.488636 |
| algos | sparse-pca | taxi | relative_reconstruction_error (lower is better) | 0.277226 | 0.277226 | - | sklearn-cpu 0.277226 |
| algos | sparse-rp | istella | mean_abs_distortion | 1.883381 | 1.883381 | - | sklearn-cpu 0.474347 |
| algos | sparse-rp | taxi | mean_abs_distortion | 0.147163 | 0.147163 | - | sklearn-cpu 0.381016 |
| algos | spline | istella | max_abs_diff_vs_sklearn | 1.788e-07 | 1.788e-07 | - | sklearn-cpu - |
| algos | spline | istella | rel_diff_vs_sklearn | 6.424e-08 | 6.226e-08 | - | sklearn-cpu - |
| algos | spline | taxi | max_abs_diff_vs_sklearn | 1.192e-07 | 1.788e-07 | - | sklearn-cpu - |
| algos | spline | taxi | rel_diff_vs_sklearn | 6.181e-08 | 5.888e-08 | - | sklearn-cpu - |
| algos | standard-scaler | istella | max_abs_diff_vs_sklearn | 0.002426 | 0.002426 | - | sklearn-cpu - |
| algos | standard-scaler | istella | rel_diff_vs_sklearn | 3.591e-06 | 3.59e-06 | - | sklearn-cpu - |
| algos | standard-scaler | taxi | max_abs_diff_vs_sklearn | 0.000103 | 0.000103 | - | sklearn-cpu - |
| algos | standard-scaler | taxi | rel_diff_vs_sklearn | 3.84e-06 | 3.846e-06 | - | sklearn-cpu - |
| algos | stl | synthetic | rel_diff_vs_statsmodels | 6.535e-07 | 6.508e-07 | - | statsmodels-cpu - |
| algos | stl | synthetic | residual_std | 0.783175 | 0.783175 | - | statsmodels-cpu 0.783175 |
| algos | stl | taxi-hourly | rel_diff_vs_statsmodels | 1.258e-06 | 1.212e-06 | - | statsmodels-cpu - |
| algos | stl | taxi-hourly | residual_std | 18.312476 | 18.312475 | - | statsmodels-cpu 18.312475 |
| algos | svd | taxi | max_rel_singular_value_error | 6.332e-07 | 6.332e-07 | - | numpy-cpu 4.308e-08; torch-gpu 2.225e-06 |
| algos | svd | taxi | relative_reconstruction_error_100k_rows | 0.000333 | 0.000333 | - | numpy-cpu 4.314e-08; torch-gpu 1.351e-05 |
| algos | svgp | istella | r2 (higher is better) | -0.106016 | -0.106016 | - | gpytorch-gpu -; gpytorch-cpu -0.106040 |
| algos | svgp | istella | rmse (lower is better) | 0.878373 | 0.878373 | - | gpytorch-gpu -; gpytorch-cpu 0.878383 |
| algos | svgp | taxi | r2 (higher is better) | - | - | - | gpytorch-gpu -; gpytorch-cpu -0.209325 |
| algos | svgp | taxi | rmse (lower is better) | - | - | - | gpytorch-gpu -; gpytorch-cpu 17.829528 |
| algos | target-encoder | istella | max_abs_diff_vs_sklearn | 1.074e-08 | 1.074e-08 | - | sklearn-cpu - |
| algos | target-encoder | istella | rel_diff_vs_sklearn | 2.361e-08 | 2.361e-08 | - | sklearn-cpu - |
| algos | target-encoder | taxi | max_abs_diff_vs_sklearn | 2.965e-08 | 2.965e-08 | - | sklearn-cpu - |
| algos | target-encoder | taxi | rel_diff_vs_sklearn | 1.581e-08 | 1.581e-08 | - | sklearn-cpu - |
| algos | theta | synthetic | forecast_rmse (lower is better) | 1.436606 | 1.436610 | - | statsforecast-cpu 1.436557; statsmodels-cpu 1.434862 |
| algos | theta | taxi-hourly | forecast_rmse (lower is better) | 49.281168 | 49.020604 | - | statsforecast-cpu 49.253901; statsmodels-cpu 49.311757 |
| algos | var | synthetic | forecast_rmse (lower is better) | 1.140787 | 1.140864 | - | statsmodels-cpu 1.144945 |
| algos | var | taxi-hourly | forecast_rmse (lower is better) | 33.167955 | 33.167951 | - | statsmodels-cpu 33.167986 |
| algos | variance-threshold | istella | max_abs_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | variance-threshold | istella | rel_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | variance-threshold | taxi | max_abs_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | variance-threshold | taxi | rel_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| classical | hdbscan | taxi | n_clusters | 160 | 161 | - | sklearn-cpu 160 |
| classical | hdbscan | taxi | noise_fraction | 0.142220 | 0.140550 | - | sklearn-cpu 0.142240 |
| classical | hdbscan | taxi | rows | 100000 | 100000 | - | sklearn-cpu 100000 |
| classical | hdbscan | taxi | ari_vs_ours (1 is our partition exactly) | 0.991364 | - | - | sklearn-cpu 0.991251 |
| classical | hdbscan | taxi | noise_agreement_vs_ours | 0.998310 | - | - | sklearn-cpu 0.998290 |
| classical2 | gmm | taxi | bic (lower is better) | - | -3.67e+06 | - | sklearn-cpu - |
| classical2 | gmm | taxi | mean_log_likelihood (higher is better) | - | 12.861940 | - | sklearn-cpu - |
| classical2 | gmm | taxi | n_iter | - | 32 | - | sklearn-cpu - |
| classical2 | nystroem | istella | kernel_rel_error (lower is better) | 0.033430 | 0.033430 | - | sklearn-cpu 0.038958 |
| classical2 | nystroem | taxi | kernel_rel_error (lower is better) | 0.045618 | 0.045611 | - | sklearn-cpu 0.044370 |

## Inference at a glance

Batch prediction, each arm with its own fitted model from the same race; medians in ms. `FAST = IDENTICAL bits` compares our two tiers' predictions on the same rows; `CPU = IDENTICAL bits` compares our CPU tier's with our GPU IDENTICAL arm's.

| family | lane | dataset | batch | rows | ours FAST ms | ours IDENTICAL ms | FAST = IDENTICAL bits | ours CPU ms | CPU = IDENTICAL bits | opponents |
|---|---|---|---|---|---|---|---|---|---|---|
| algos | adaboost-clf | istella | Xq | - | - | - | - | - | - | sklearn-cpu - ms (IDENTICAL/arm -) |
| algos | adaboost-clf | taxi | Xq | - | 261.3 | 233.7 | - | - | - | sklearn-cpu 213.3 ms (IDENTICAL/arm 1.096) |
| algos | adaboost-reg | istella | Xq | - | 190.9 | 233.3 | - | - | - | sklearn-cpu 123.6 ms (IDENTICAL/arm 1.887) |
| algos | adaboost-reg | taxi | Xq | - | 84.8 | 84.9 | - | - | - | sklearn-cpu 50.1 ms (IDENTICAL/arm 1.694) |
| algos | additive-chi2 | istella | Xq | - | 2.3 | 2.3 | - | - | - | sklearn-cpu 3.3 ms (IDENTICAL/arm 0.714) |
| algos | additive-chi2 | taxi | Xq | - | 0.9 | 0.9 | - | - | - | sklearn-cpu 0.3 ms (IDENTICAL/arm 2.810) |
| algos | affinity-prop | istella | Xq | - | 504.0 | 508.8 | - | - | - | sklearn-cpu 33.0 ms (IDENTICAL/arm 15.437) |
| algos | affinity-prop | taxi | Xq | - | 16.2 | 23.7 | - | - | - | sklearn-cpu 9.1 ms (IDENTICAL/arm 2.603) |
| algos | bagging-clf | istella | Xq | - | 167.8 | 172.6 | - | - | - | sklearn-cpu 1121.5 ms (IDENTICAL/arm 0.154) |
| algos | bagging-clf | taxi | Xq | - | 41.2 | 44.3 | - | - | - | sklearn-cpu 591.2 ms (IDENTICAL/arm 0.075) |
| algos | bagging-reg | istella | Xq | - | 126.3 | 128.4 | - | - | - | sklearn-cpu 720.1 ms (IDENTICAL/arm 0.178) |
| algos | bagging-reg | taxi | Xq | - | 26.0 | 26.2 | - | - | - | sklearn-cpu 350.1 ms (IDENTICAL/arm 0.075) |
| algos | bayesian-gmm | istella | Xq | - | - | - | - | - | - | sklearn-cpu - ms (IDENTICAL/arm -) |
| algos | bayesian-gmm | taxi | Xq | - | 17.1 | 17.5 | - | - | - | sklearn-cpu 9.1 ms (IDENTICAL/arm 1.926) |
| algos | bernoulli-nb | istella | Xq | - | 139.9 | 140.6 | - | - | - | sklearn-cpu 239.8 ms (IDENTICAL/arm 0.586) |
| algos | bernoulli-nb | taxi | Xq | - | 10.7 | 8.0 | - | - | - | sklearn-cpu 23.8 ms (IDENTICAL/arm 0.337) |
| algos | binarizer | istella | Xq | - | 63.1 | 62.9 | - | - | - | sklearn-cpu 90.7 ms (IDENTICAL/arm 0.694) |
| algos | binarizer | taxi | Xq | - | 3.5 | 3.6 | - | - | - | sklearn-cpu 6.1 ms (IDENTICAL/arm 0.589) |
| algos | categorical-nb | istella | Xq | - | 8.7 | 26.4 | - | - | - | sklearn-cpu 21.1 ms (IDENTICAL/arm 1.251) |
| algos | categorical-nb | taxi | Xq | - | 6.5 | 23.8 | - | - | - | sklearn-cpu 16.5 ms (IDENTICAL/arm 1.443) |
| algos | cca | istella | Xq | - | 79.5 | 81.4 | - | - | - | sklearn-cpu 74.3 ms (IDENTICAL/arm 1.095) |
| algos | cca | taxi | Xq | - | 10.7 | 11.5 | - | - | - | sklearn-cpu 8.4 ms (IDENTICAL/arm 1.361) |
| algos | complement-nb | istella | Xq | - | 41.8 | 60.4 | - | - | - | sklearn-cpu 61.2 ms (IDENTICAL/arm 0.987) |
| algos | complement-nb | taxi | Xq | - | 6.5 | 6.6 | - | - | - | sklearn-cpu 11.8 ms (IDENTICAL/arm 0.561) |
| algos | complement-nb | text | Xq | - | 98.1 | 88.7 | - | - | - | sklearn-cpu 96.0 ms (IDENTICAL/arm 0.923) |
| algos | dart-reg | istella | Xq | - | 309.9 | 214.7 | - | - | - | lightgbm-cpu 128.8 ms (IDENTICAL/arm 1.667); xgboost-cpu - ms (IDENTICAL/arm -) |
| algos | dart-reg | taxi | Xq | - | 178.7 | 194.2 | - | - | - | lightgbm-cpu 120.2 ms (IDENTICAL/arm 1.617); xgboost-cpu 132.5 ms (IDENTICAL/arm 1.465) |
| algos | dart | istella | Xq | - | 440.1 | 437.5 | - | - | - | lightgbm-cpu 124.3 ms (IDENTICAL/arm 3.520); xgboost-cpu - ms (IDENTICAL/arm -) |
| algos | dart | taxi | Xq | - | 386.6 | 340.4 | - | - | - | lightgbm-cpu 123.8 ms (IDENTICAL/arm 2.750); xgboost-cpu 133.1 ms (IDENTICAL/arm 2.557) |
| algos | decision-tree-clf | istella | Xq | - | 16.8 | 17.3 | - | - | - | sklearn-cpu 22.1 ms (IDENTICAL/arm 0.782) |
| algos | decision-tree-clf | taxi | Xq | - | 5.2 | 4.6 | - | - | - | sklearn-cpu 12.8 ms (IDENTICAL/arm 0.360) |
| algos | decision-tree-reg | istella | Xq | - | 12.5 | 12.7 | - | - | - | sklearn-cpu 10.8 ms (IDENTICAL/arm 1.176) |
| algos | decision-tree-reg | taxi | Xq | - | 3.1 | 3.0 | - | - | - | sklearn-cpu 9.4 ms (IDENTICAL/arm 0.316) |
| algos | dict-learning | istella | Xq | - | 183.3 | 205.9 | - | - | - | sklearn-cpu 55.8 ms (IDENTICAL/arm 3.691) |
| algos | dict-learning | taxi | Xq | - | 90.3 | 123.9 | - | - | - | sklearn-cpu 51.5 ms (IDENTICAL/arm 2.406) |
| algos | elliptic-envelope | istella | Xq | - | - | - | - | - | - | sklearn-cpu 3923.2 ms (IDENTICAL/arm -) |
| algos | elliptic-envelope | taxi | Xq | - | 9.2 | - | - | - | - | sklearn-cpu 6.2 ms (IDENTICAL/arm -) |
| algos | factor-analysis | istella | Xq | - | - | - | - | - | - | sklearn-cpu 37.1 ms (IDENTICAL/arm -) |
| algos | factor-analysis | taxi | Xq | - | 6.6 | 6.7 | - | - | - | sklearn-cpu 4.1 ms (IDENTICAL/arm 1.619) |
| algos | fastica | istella | Xq | - | 43.4 | 42.8 | - | - | - | sklearn-cpu 18.0 ms (IDENTICAL/arm 2.380) |
| algos | fastica | taxi | Xq | - | 4.3 | 4.6 | - | - | - | sklearn-cpu 2.6 ms (IDENTICAL/arm 1.803) |
| algos | gaussian-nb | istella | Xq | - | 44.5 | 61.5 | - | - | - | sklearn-cpu 217.0 ms (IDENTICAL/arm 0.283) |
| algos | gaussian-nb | taxi | Xq | - | 6.7 | 6.3 | - | - | - | sklearn-cpu 15.2 ms (IDENTICAL/arm 0.414) |
| algos | gaussian-rp | istella | Xq | - | 36.7 | 37.8 | - | - | - | sklearn-cpu 6.5 ms (IDENTICAL/arm 5.836) |
| algos | gaussian-rp | taxi | Xq | - | 6.8 | 6.6 | - | - | - | sklearn-cpu 1.4 ms (IDENTICAL/arm 4.590) |
| algos | gru-clf | synthetic | Xq | - | 244.8 | 245.9 | - | - | - | torch-eager-fp32 - ms (IDENTICAL/arm -); torch-compile-fp32 - ms (IDENTICAL/arm -); torch-eager-bf16 - ms (IDENTICAL/arm -); torch-compile-bf16 - ms (IDENTICAL/arm -) |
| algos | gru-clf | taxi-hourly | Xq | - | 246.1 | 245.8 | - | - | - | torch-eager-fp32 - ms (IDENTICAL/arm -); torch-compile-fp32 - ms (IDENTICAL/arm -); torch-eager-bf16 - ms (IDENTICAL/arm -); torch-compile-bf16 - ms (IDENTICAL/arm -) |
| algos | gru-reg | synthetic | Xq | - | 122.6 | 122.3 | - | - | - | torch-eager-fp32 - ms (IDENTICAL/arm -); torch-compile-fp32 - ms (IDENTICAL/arm -); torch-eager-bf16 - ms (IDENTICAL/arm -); torch-compile-bf16 - ms (IDENTICAL/arm -) |
| algos | gru-reg | taxi-hourly | Xq | - | 122.5 | 122.8 | - | - | - | torch-eager-fp32 - ms (IDENTICAL/arm -); torch-compile-fp32 - ms (IDENTICAL/arm -); torch-eager-bf16 - ms (IDENTICAL/arm -); torch-compile-bf16 - ms (IDENTICAL/arm -) |
| algos | incremental-pca | istella | Xq | - | 37.8 | 38.3 | - | - | - | sklearn-cpu 25.1 ms (IDENTICAL/arm 1.525) |
| algos | incremental-pca | taxi | Xq | - | 4.5 | 4.6 | - | - | - | sklearn-cpu 2.7 ms (IDENTICAL/arm 1.692) |
| algos | iterative-imputer | istella | Xq | - | 14.8 | 11.9 | - | - | - | sklearn-cpu 107.3 ms (IDENTICAL/arm 0.111) |
| algos | iterative-imputer | taxi | Xq | - | 6.5 | 6.7 | - | - | - | sklearn-cpu 19.1 ms (IDENTICAL/arm 0.350) |
| algos | kbins | istella | Xq | - | 118.1 | 76.6 | - | - | - | sklearn-cpu 186.4 ms (IDENTICAL/arm 0.411) |
| algos | kbins | taxi | Xq | - | 3.8 | 3.5 | - | - | - | sklearn-cpu 8.3 ms (IDENTICAL/arm 0.417) |
| algos | kernel-pca | istella | Xq | - | 1373.5 | 1375.7 | - | - | - | sklearn-cpu 278.3 ms (IDENTICAL/arm 4.944) |
| algos | kernel-pca | taxi | Xq | - | 98.1 | 99.8 | - | - | - | sklearn-cpu 320.3 ms (IDENTICAL/arm 0.312) |
| algos | knn-imputer | istella | Xq | - | 57340.9 | 360.4 | - | - | - | sklearn-cpu - ms (IDENTICAL/arm -) |
| algos | knn-imputer | taxi | Xq | - | 1914.0 | 2085.4 | - | - | - | sklearn-cpu 32272.8 ms (IDENTICAL/arm 0.065) |
| algos | label-binarizer | istella | Xq | - | 20.7 | 20.0 | - | - | - | sklearn-cpu 3.1 ms (IDENTICAL/arm 6.516) |
| algos | label-binarizer | taxi | Xq | - | 92.6 | 100.1 | - | - | - | sklearn-cpu 12.1 ms (IDENTICAL/arm 8.258) |
| algos | label-encoder | istella | Xq | - | 18.1 | 17.5 | - | - | - | sklearn-cpu 1.2 ms (IDENTICAL/arm 14.197) |
| algos | label-encoder | taxi | Xq | - | 18.3 | 19.2 | - | - | - | sklearn-cpu 1.5 ms (IDENTICAL/arm 13.079) |
| algos | label-propagation | istella | Xq | - | 151570.6 | 169605.8 | - | - | - | sklearn-cpu 2994.6 ms (IDENTICAL/arm 56.637) |
| algos | label-propagation | taxi | Xq | - | 2196.5 | 2215.8 | - | - | - | sklearn-cpu 675.9 ms (IDENTICAL/arm 3.278) |
| algos | label-spreading | istella | Xq | - | 151625.1 | 169556.3 | - | - | - | sklearn-cpu 3039.3 ms (IDENTICAL/arm 55.787) |
| algos | label-spreading | taxi | Xq | - | 2180.1 | 2217.3 | - | - | - | sklearn-cpu 667.7 ms (IDENTICAL/arm 3.321) |
| algos | layernorm | synthetic | Xq | - | 40.9 | 39.9 | - | - | - | torch-eager-fp32 1.5 ms (IDENTICAL/arm 26.615); torch-compile-fp32 203.7 ms (IDENTICAL/arm 0.196); torch-eager-bf16 1.1 ms (IDENTICAL/arm 34.820); torch-compile-bf16 203.8 ms (IDENTICAL/arm 0.196) |
| algos | lda-clf | istella | Xq | - | 50.8 | 50.9 | - | - | - | sklearn-cpu 12.6 ms (IDENTICAL/arm 4.039) |
| algos | lda-clf | taxi | Xq | - | 6.2 | 6.4 | - | - | - | sklearn-cpu 1.7 ms (IDENTICAL/arm 3.856) |
| algos | lda | taxi-zones | Xq | - | 397.2 | 413.3 | - | - | - | sklearn-cpu 471.0 ms (IDENTICAL/arm 0.877) |
| algos | lda | text | Xq | - | - | - | - | - | - | sklearn-cpu 735.4 ms (IDENTICAL/arm -) |
| algos | lstm-clf | synthetic | Xq | - | 319.5 | 319.2 | - | - | - | torch-eager-fp32 - ms (IDENTICAL/arm -); torch-compile-fp32 - ms (IDENTICAL/arm -); torch-eager-bf16 - ms (IDENTICAL/arm -); torch-compile-bf16 - ms (IDENTICAL/arm -) |
| algos | lstm-clf | taxi-hourly | Xq | - | 320.3 | 318.3 | - | - | - | torch-eager-fp32 - ms (IDENTICAL/arm -); torch-compile-fp32 - ms (IDENTICAL/arm -); torch-eager-bf16 - ms (IDENTICAL/arm -); torch-compile-bf16 - ms (IDENTICAL/arm -) |
| algos | lstm-reg | synthetic | Xq | - | 158.3 | 166.6 | - | - | - | torch-eager-fp32 - ms (IDENTICAL/arm -); torch-compile-fp32 - ms (IDENTICAL/arm -); torch-eager-bf16 - ms (IDENTICAL/arm -); torch-compile-bf16 - ms (IDENTICAL/arm -) |
| algos | lstm-reg | taxi-hourly | Xq | - | 158.6 | 158.4 | - | - | - | torch-eager-fp32 - ms (IDENTICAL/arm -); torch-compile-fp32 - ms (IDENTICAL/arm -); torch-eager-bf16 - ms (IDENTICAL/arm -); torch-compile-bf16 - ms (IDENTICAL/arm -) |
| algos | maxabs-scaler | istella | Xq | - | 75.9 | 84.5 | - | - | - | sklearn-cpu 12.7 ms (IDENTICAL/arm 6.630) |
| algos | maxabs-scaler | taxi | Xq | - | 3.2 | 6.1 | - | - | - | sklearn-cpu 1.4 ms (IDENTICAL/arm 4.413) |
| algos | mb-dict-learning | istella | Xq | - | 201.1 | 189.9 | - | - | - | sklearn-cpu 82.2 ms (IDENTICAL/arm 2.311) |
| algos | mb-dict-learning | taxi | Xq | - | 101.3 | 58.6 | - | - | - | sklearn-cpu 17.8 ms (IDENTICAL/arm 3.290) |
| algos | mb-sparse-pca | istella | Xq | - | 20.1 | 28.1 | - | - | - | sklearn-cpu 18.5 ms (IDENTICAL/arm 1.519) |
| algos | mb-sparse-pca | taxi | Xq | - | 17.3 | 12.9 | - | - | - | sklearn-cpu 2.5 ms (IDENTICAL/arm 5.144) |
| algos | minmax-scaler | istella | Xq | - | 95.3 | 95.6 | - | - | - | sklearn-cpu 16.3 ms (IDENTICAL/arm 5.866) |
| algos | minmax-scaler | taxi | Xq | - | 4.4 | 4.5 | - | - | - | sklearn-cpu 2.0 ms (IDENTICAL/arm 2.222) |
| algos | mlp-clf | istella | Xq | - | 452.1 | 482.1 | - | - | - | sklearn-cpu 115.2 ms (IDENTICAL/arm 4.183) |
| algos | mlp-clf | taxi | Xq | - | 246.4 | 261.7 | - | - | - | sklearn-cpu 107.0 ms (IDENTICAL/arm 2.446) |
| algos | mlp-reg | istella | Xq | - | 229.1 | 248.3 | - | - | - | sklearn-cpu 63.7 ms (IDENTICAL/arm 3.898) |
| algos | mlp-reg | taxi | Xq | - | 124.0 | 130.9 | - | - | - | sklearn-cpu 58.9 ms (IDENTICAL/arm 2.222) |
| algos | moe | synthetic | Xq | - | 8658.2 | 8628.6 | - | - | - | torch-eager-fp32 73.5 ms (IDENTICAL/arm 117.398); torch-compile-fp32 70.6 ms (IDENTICAL/arm 122.209); torch-eager-bf16 113.1 ms (IDENTICAL/arm 76.267); torch-compile-bf16 110.7 ms (IDENTICAL/arm 77.921) |
| algos | multilabel-binarizer | istella | Xq | - | 40.0 | 41.5 | - | - | - | sklearn-cpu 20.3 ms (IDENTICAL/arm 2.046) |
| algos | multilabel-binarizer | taxi | Xq | - | 49.4 | 49.8 | - | - | - | sklearn-cpu 16.9 ms (IDENTICAL/arm 2.941) |
| algos | multinomial-nb | istella | Xq | - | 36.5 | 52.9 | - | - | - | sklearn-cpu 60.9 ms (IDENTICAL/arm 0.868) |
| algos | multinomial-nb | taxi | Xq | - | 7.2 | 5.5 | - | - | - | sklearn-cpu 12.2 ms (IDENTICAL/arm 0.450) |
| algos | multinomial-nb | text | Xq | - | 96.0 | 74.1 | - | - | - | sklearn-cpu 94.6 ms (IDENTICAL/arm 0.783) |
| algos | nearest-centroid | istella | Xq | - | 62.2 | 63.9 | - | - | - | sklearn-cpu 186.2 ms (IDENTICAL/arm 0.343) |
| algos | nearest-centroid | taxi | Xq | - | 32.6 | 32.6 | - | - | - | sklearn-cpu 12.8 ms (IDENTICAL/arm 2.546) |
| algos | nmf | istella | Xq | - | 398.1 | 412.1 | - | - | - | sklearn-cpu 178.7 ms (IDENTICAL/arm 2.306) |
| algos | nmf | taxi | Xq | - | 67.8 | 69.5 | - | - | - | sklearn-cpu 72.5 ms (IDENTICAL/arm 0.958) |
| algos | normalizer | istella | Xq | - | 72.9 | 73.3 | - | - | - | sklearn-cpu 18.8 ms (IDENTICAL/arm 3.887) |
| algos | normalizer | taxi | Xq | - | 3.4 | 3.6 | - | - | - | sklearn-cpu 1.9 ms (IDENTICAL/arm 1.936) |
| algos | ocsvm | istella | Xq | - | 144.0 | 144.0 | - | - | - | sklearn-cpu 893.0 ms (IDENTICAL/arm 0.161) |
| algos | ocsvm | taxi | Xq | - | 14.6 | 14.4 | - | - | - | sklearn-cpu 266.4 ms (IDENTICAL/arm 0.054) |
| algos | onehot | istella | Xq | - | 32.4 | 32.4 | - | - | - | sklearn-cpu 30.9 ms (IDENTICAL/arm 1.051) |
| algos | onehot | taxi | Xq | - | 113.5 | 113.9 | - | - | - | sklearn-cpu 33.9 ms (IDENTICAL/arm 3.363) |
| algos | optics | istella | Xq | - | 0.0 | 0.0 | - | - | - | sklearn-cpu 0.0 ms (IDENTICAL/arm 0.445) |
| algos | optics | taxi | Xq | - | 0.0 | 0.0 | - | - | - | sklearn-cpu - ms (IDENTICAL/arm -) |
| algos | ordinal | istella | Xq | - | 7.6 | 7.6 | - | - | - | sklearn-cpu 21.2 ms (IDENTICAL/arm 0.358) |
| algos | ordinal | taxi | Xq | - | 7.3 | 7.0 | - | - | - | sklearn-cpu 12.2 ms (IDENTICAL/arm 0.575) |
| algos | pls-canonical | istella | Xq | - | 79.5 | 78.3 | - | - | - | sklearn-cpu 74.5 ms (IDENTICAL/arm 1.052) |
| algos | pls-canonical | taxi | Xq | - | 10.5 | 11.0 | - | - | - | sklearn-cpu 8.5 ms (IDENTICAL/arm 1.294) |
| algos | pls | istella | Xq | - | 41.4 | 41.9 | - | - | - | sklearn-cpu 49.3 ms (IDENTICAL/arm 0.851) |
| algos | pls | taxi | Xq | - | 4.2 | 4.3 | - | - | - | sklearn-cpu 2.8 ms (IDENTICAL/arm 1.531) |
| algos | poly-count-sketch | istella | Xq | - | 3.6 | 3.7 | - | - | - | sklearn-cpu 6.2 ms (IDENTICAL/arm 0.590) |
| algos | poly-count-sketch | taxi | Xq | - | 3.2 | 3.2 | - | - | - | sklearn-cpu 2.9 ms (IDENTICAL/arm 1.106) |
| algos | poly-features | istella | Xq | - | 50.7 | 49.9 | - | - | - | sklearn-cpu 43.0 ms (IDENTICAL/arm 1.161) |
| algos | poly-features | taxi | Xq | - | 27.2 | 27.2 | - | - | - | sklearn-cpu 22.5 ms (IDENTICAL/arm 1.212) |
| algos | power-transformer | istella | Xq | - | 79.7 | 94.2 | - | - | - | sklearn-cpu - ms (IDENTICAL/arm -) |
| algos | power-transformer | taxi | Xq | - | 6.0 | 5.8 | - | - | - | sklearn-cpu 13.1 ms (IDENTICAL/arm 0.442) |
| algos | qda | istella | Xq | - | 733.9 | 737.3 | - | - | - | sklearn-cpu 138.7 ms (IDENTICAL/arm 5.317) |
| algos | qda | taxi | Xq | - | 7.2 | 8.1 | - | - | - | sklearn-cpu 15.6 ms (IDENTICAL/arm 0.518) |
| algos | quantile-transformer | istella | Xq | - | 164.3 | 152.1 | - | - | - | sklearn-cpu 964.4 ms (IDENTICAL/arm 0.158) |
| algos | quantile-transformer | taxi | Xq | - | 8.8 | 8.3 | - | - | - | sklearn-cpu 43.3 ms (IDENTICAL/arm 0.191) |
| algos | radius-neighbors | istella | Xq | - | - | - | - | - | - | sklearn-cpu 2736.3 ms (IDENTICAL/arm -) |
| algos | radius-neighbors | taxi | Xq | - | 302.1 | 308.7 | - | - | - | sklearn-cpu 101.5 ms (IDENTICAL/arm 3.040) |
| algos | ridge-cv | istella | Xq | - | - | - | - | - | - | sklearn-cpu 5.8 ms (IDENTICAL/arm -) |
| algos | ridge-cv | taxi | Xq | - | - | - | - | - | - | sklearn-cpu 0.6 ms (IDENTICAL/arm -) |
| algos | rnn-clf | synthetic | Xq | - | 93.3 | 99.4 | - | - | - | torch-eager-fp32 - ms (IDENTICAL/arm -); torch-compile-fp32 - ms (IDENTICAL/arm -); torch-eager-bf16 - ms (IDENTICAL/arm -); torch-compile-bf16 - ms (IDENTICAL/arm -) |
| algos | rnn-clf | taxi-hourly | Xq | - | 93.9 | 99.0 | - | - | - | torch-eager-fp32 - ms (IDENTICAL/arm -); torch-compile-fp32 - ms (IDENTICAL/arm -); torch-eager-bf16 - ms (IDENTICAL/arm -); torch-compile-bf16 - ms (IDENTICAL/arm -) |
| algos | rnn-reg | synthetic | Xq | - | 46.9 | 49.3 | - | - | - | torch-eager-fp32 - ms (IDENTICAL/arm -); torch-compile-fp32 - ms (IDENTICAL/arm -); torch-eager-bf16 - ms (IDENTICAL/arm -); torch-compile-bf16 - ms (IDENTICAL/arm -) |
| algos | rnn-reg | taxi-hourly | Xq | - | 46.6 | 49.0 | - | - | - | torch-eager-fp32 - ms (IDENTICAL/arm -); torch-compile-fp32 - ms (IDENTICAL/arm -); torch-eager-bf16 - ms (IDENTICAL/arm -); torch-compile-bf16 - ms (IDENTICAL/arm -) |
| algos | robust-scaler | istella | Xq | - | 82.4 | 79.1 | - | - | - | sklearn-cpu 26.1 ms (IDENTICAL/arm 3.032) |
| algos | robust-scaler | taxi | Xq | - | 4.3 | 5.4 | - | - | - | sklearn-cpu 2.2 ms (IDENTICAL/arm 2.437) |
| algos | simple-imputer | istella | Xq | - | 61.4 | 116.3 | - | - | - | sklearn-cpu 119.4 ms (IDENTICAL/arm 0.974) |
| algos | simple-imputer | taxi | Xq | - | 3.3 | 3.8 | - | - | - | sklearn-cpu 5.2 ms (IDENTICAL/arm 0.732) |
| algos | skewed-chi2 | istella | Xq | - | 5.3 | 5.5 | - | - | - | sklearn-cpu 1.4 ms (IDENTICAL/arm 3.912) |
| algos | skewed-chi2 | taxi | Xq | - | 1.7 | 1.8 | - | - | - | sklearn-cpu 0.7 ms (IDENTICAL/arm 2.433) |
| algos | sparse-coder | istella | Xq | - | 319.6 | 313.1 | - | - | - | sklearn-cpu 4161.6 ms (IDENTICAL/arm 0.075) |
| algos | sparse-coder | taxi | Xq | - | 224.8 | 223.9 | - | - | - | sklearn-cpu 4190.8 ms (IDENTICAL/arm 0.053) |
| algos | sparse-pca | istella | Xq | - | - | - | - | - | - | sklearn-cpu 10.1 ms (IDENTICAL/arm -) |
| algos | sparse-pca | taxi | Xq | - | 7.6 | 7.9 | - | - | - | sklearn-cpu 3.2 ms (IDENTICAL/arm 2.487) |
| algos | sparse-rp | istella | Xq | - | 36.5 | 36.3 | - | - | - | sklearn-cpu 38.1 ms (IDENTICAL/arm 0.952) |
| algos | sparse-rp | taxi | Xq | - | 6.7 | 6.7 | - | - | - | sklearn-cpu 1.6 ms (IDENTICAL/arm 4.036) |
| algos | spline | istella | Xq | - | 28.1 | 37.4 | - | - | - | sklearn-cpu 104.5 ms (IDENTICAL/arm 0.358) |
| algos | spline | taxi | Xq | - | 21.3 | 38.4 | - | - | - | sklearn-cpu 67.9 ms (IDENTICAL/arm 0.566) |
| algos | standard-scaler | istella | Xq | - | 95.9 | 93.5 | - | - | - | sklearn-cpu 36.8 ms (IDENTICAL/arm 2.542) |
| algos | standard-scaler | taxi | Xq | - | 4.2 | 4.7 | - | - | - | sklearn-cpu 2.2 ms (IDENTICAL/arm 2.116) |
| algos | svgp | istella | Xq | - | 501.4 | 497.0 | - | - | - | gpytorch-gpu - ms (IDENTICAL/arm -); gpytorch-cpu 146.7 ms (IDENTICAL/arm 3.388) |
| algos | svgp | taxi | Xq | - | - | - | - | - | - | gpytorch-gpu - ms (IDENTICAL/arm -); gpytorch-cpu 106.2 ms (IDENTICAL/arm -) |
| algos | target-encoder | istella | Xq | - | 8.3 | 8.2 | - | - | - | sklearn-cpu 22.2 ms (IDENTICAL/arm 0.369) |
| algos | target-encoder | taxi | Xq | - | 7.4 | 7.2 | - | - | - | sklearn-cpu 13.1 ms (IDENTICAL/arm 0.549) |
| algos | variance-threshold | istella | Xq | - | 60.2 | 88.5 | - | - | - | sklearn-cpu 20.6 ms (IDENTICAL/arm 4.296) |
| algos | variance-threshold | taxi | Xq | - | 3.7 | 3.4 | - | - | - | sklearn-cpu 0.9 ms (IDENTICAL/arm 3.950) |

## Classical

### hdbscan / taxi (rows full, shape 1000000x11)

race: done, driver rc 0, log `logs/classical.hdbscan.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 13817.8 | 13817.8..13817.8 | 1 | - | - | - | 341.2 | - | n_clusters=161, noise_fraction=0.140550, rows=100000 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 2502.6 | 2502.6..2502.6 | 1 | - | - | - | 228.5 | - | ari_vs_ours=0.991364, n_clusters=160, noise_agreement_vs_ours=0.998310, noise_fraction=0.142220, rows=100000 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 40853.4 | 40853.4..40853.4 | 1 | 0.338 | 0.061 | - | 250.0 | - | ari_vs_ours=0.991251, n_clusters=160, noise_agreement_vs_ours=0.998290, noise_fraction=0.142240, rows=100000 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: min_samples=10 (scikit-learn 11: the same core distance, the 10th neighbour besides the point), min_cluster_size=100, metric='euclidean', cluster_selection_method='eom', cluster_selection_epsilon=0.0, alpha=1.0, allow_single_cluster=False. Rows: the dbscan block's first 100,000 rows. Timed: fit.

mismatch: seed: no arm has a seed argument (deterministic)

mismatch: max_cluster_size: ours and cuML 0, scikit-learn None (both mean no limit)

mismatch: min_samples: ours and cuML 10, scikit-learn 11. The SAME k-th neighbour: cuML's runner.h:68-80 (ours transcribes it) runs the k-NN at min_samples + 1 including the point itself; scikit-learn's kneighbors(X, min_samples) counts the point itself (its HDBSCAN Notes say so). tools/bench_board_params.py maps both to one canonical value

mismatch: scikit-learn algorithm='auto', leaf_size=40, n_jobs=-1: its own; cuML build_algo at its default

config: cuML benchmark (RAPIDS), HDBSCAN (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn/HDBSCAN (get_params) |
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

## Classical, wave 2

### gmm / taxi (rows full, shape X 100000x11; Xq 20000x11)

race: done, driver rc 0, log `logs/classical2.gmm.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 984.4 | 984.4..984.4 | 1 | - | - | - | 230.5 | - | bic=-3.67e+06, mean_log_likelihood=12.861940, n_iter=32 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
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

### nystroem / istella (rows full, shape X 100000x220; Xcheck 1000x220)

race: done, driver rc 0, log `logs/classical2.nystroem.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1221.3 | 1221.3..1221.3 | 1 | - | - | - | 1697.4 | - | kernel_rel_error=0.033430 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1189.6 | 1189.6..1189.6 | 1 | - | - | - | 1695.9 | - | kernel_rel_error=0.033430 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 284.8 | 284.8..284.8 | 1 | 4.289 | 4.177 | - | 1351.3 | - | kernel_rel_error=0.038958 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

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

race: done, driver rc 0, log `logs/classical2.nystroem.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1394.2 | 1394.2..1394.2 | 1 | - | - | - | 582.1 | - | kernel_rel_error=0.045611 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1293.7 | 1293.7..1293.7 | 1 | - | - | - | 582.7 | - | kernel_rel_error=0.045618 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 235.4 | 235.4..235.4 | 1 | 5.922 | 5.495 | - | 392.3 | - | kernel_rel_error=0.044370 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

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

## Algorithm expansion

### adaboost-clf / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: failed, driver rc 1, log `logs/algos.adaboost-clf.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | HOST-MEMORY(killed at 26.1 GB: the driver's process tree held 31.2 GB, over 90% of the box's 34.4 GB) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | HOST-MEMORY(killed at 27.6 GB: the driver's process tree held 31.8 GB, over 90% of the box's 34.4 GB) |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(timeout: null) (measured this run) |

settings: {'algorithm': 'SAMME', 'estimator': 'DecisionTreeClassifier(max_depth=3, random_state=7)', 'learning_rate': 1.0, 'n_estimators': 50, 'random_state': 7}. Rows: None. Timed: None.

mismatch: ours' DecisionTree* is its forest builder with one tree: it splits on n_bins=128 quantile bins per feature (ours only; not a scikit-learn parameter), scikit-learn on exact thresholds

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| algorithm | "SAMME" | "SAMME" | "SAMME" |
| learning_rate | 1.0 | 1.0 | 1.0 |
| n_estimators | 50 | 50 | 50 |
| seed | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(timeout: null) |
| mojolearn FAST | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(timeout: null) |
| sklearn-cpu | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(timeout: null) |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### adaboost-clf / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.adaboost-clf.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3904.9 | 3904.9..3904.9 | 1 | - | - | - | 3246.2 | - | accuracy=0.765230, logloss=0.543227 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 3964.4 | 3964.4..3964.4 | 1 | - | - | - | 3460.3 | - | accuracy=0.765230, logloss=0.543227 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 33098.5 | 33098.5..33098.5 | 1 | 0.118 | 0.120 | - | 220.1 | - | accuracy=0.765360, logloss=0.540565 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'algorithm': 'SAMME', 'estimator': 'DecisionTreeClassifier(max_depth=3, random_state=7)', 'learning_rate': 1.0, 'n_estimators': 50, 'random_state': 7}. Rows: None. Timed: None.

mismatch: ours' DecisionTree* is its forest builder with one tree: it splits on n_bins=128 quantile bins per feature (ours only; not a scikit-learn parameter), scikit-learn on exact thresholds

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| algorithm | "SAMME" | "SAMME" | "SAMME" |
| learning_rate | 1.0 | 1.0 | 1.0 |
| n_estimators | 50 | 50 | 50 |
| seed | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 233.7 | 233.7..233.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 261.3 | 261.3..261.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 213.3 | 213.3..213.3 | 1 | 1.096 | 1.225 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### adaboost-reg / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.adaboost-reg.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 10264.9 | 10264.9..10264.9 | 1 | - | - | - | 18366.2 | - | finite=True, r2=0.234086, rmse=0.731056 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 8225.7 | 8225.7..8225.7 | 1 | - | - | - | 15781.2 | - | finite=True, r2=0.238879, rmse=0.728764 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 212240.9 | 212240.9..212240.9 | 1 | 0.048 | 0.039 | - | 1965.6 | - | finite=True, r2=0.167622, rmse=0.762115 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'estimator': 'DecisionTreeRegressor(max_depth=3, random_state=7)', 'learning_rate': 1.0, 'loss': 'linear', 'n_estimators': 50, 'random_state': 7}. Rows: None. Timed: None.

mismatch: ours' DecisionTree* is its forest builder with one tree: it splits on n_bins=128 quantile bins per feature (ours only; not a scikit-learn parameter), scikit-learn on exact thresholds

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| learning_rate | 1.0 | 1.0 | 1.0 |
| loss | "linear" | "linear" | "linear" |
| n_estimators | 50 | 50 | 50 |
| seed | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 233.3 | 233.3..233.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 190.9 | 190.9..190.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 123.6 | 123.6..123.6 | 1 | 1.887 | 1.544 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### adaboost-reg / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.adaboost-reg.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 4324.3 | 4324.3..4324.3 | 1 | - | - | - | 2466.8 | - | finite=True, r2=-0.417994, rmse=18.965923 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 3734.7 | 3734.7..3734.7 | 1 | - | - | - | 2418.5 | - | finite=True, r2=0.216396, rmse=14.098896 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 15168.6 | 15168.6..15168.6 | 1 | 0.285 | 0.246 | - | 254.1 | - | finite=True, r2=0.563927, rmse=10.517589 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'estimator': 'DecisionTreeRegressor(max_depth=3, random_state=7)', 'learning_rate': 1.0, 'loss': 'linear', 'n_estimators': 50, 'random_state': 7}. Rows: None. Timed: None.

mismatch: ours' DecisionTree* is its forest builder with one tree: it splits on n_bins=128 quantile bins per feature (ours only; not a scikit-learn parameter), scikit-learn on exact thresholds

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| learning_rate | 1.0 | 1.0 | 1.0 |
| loss | "linear" | "linear" | "linear" |
| n_estimators | 50 | 50 | 50 |
| seed | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 84.9 | 84.9..84.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 84.8 | 84.8..84.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 50.1 | 50.1..50.1 | 1 | 1.694 | 1.691 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### adafactor / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.adafactor.synthetic.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 2811.0 | 2811.0..2811.0 | 1 | - | - | - | 1754.6 | - | rel_fro_vs_torch_eager_fp32=4.324e-05 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 651.3 | 651.3..651.3 | 1 | - | - | - | 1755.4 | - | rel_fro_vs_torch_eager_fp32=1.45e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 498.3 | 498.3..498.3 | 1 | 5.642 | 1.307 | - | 1398.7 | 1032.4 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 548.9 | 548.9..548.9 | 1 | 5.121 | 1.187 | - | 1482.0 | 1032.4 | rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-eager-fp32, torch-compile-fp32: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

settings: {'beta2_decay': -0.8, 'd': 1.0, 'eps': [None, 0.001], 'lr': 0.001, 'maximize': False, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | torch-compile-fp32 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | torch (optimizer.defaults) | torch (optimizer.defaults) |
| eps | [null, 0.001] | [null, 0.001] | [null, 0.001] | [null, 0.001] |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| weight_decay | 0.0 | 0.0 | 0.0 | 0.0 |

### adagrad / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.adagrad.synthetic.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 395.3 | 395.3..395.3 | 1 | - | - | - | 1511.1 | - | rel_fro_vs_torch_eager_fp32=3.297e-09 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 383.0 | 383.0..383.0 | 1 | - | - | - | 1512.1 | - | rel_fro_vs_torch_eager_fp32=3.297e-09 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 53.1 | 53.1..53.1 | 1 | 7.440 | 7.208 | - | 1405.5 | 1032.4 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 142.1 | 142.1..142.1 | 1 | 2.782 | 2.696 | - | 1494.3 | 1032.4 | rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-eager-fp32, torch-compile-fp32: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

settings: {'eps': 1e-10, 'initial_accumulator_value': 0.0, 'lr': 0.001, 'lr_decay': 0.0, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | torch-compile-fp32 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | torch (optimizer.defaults) | torch (optimizer.defaults) |
| eps | 1e-10 | 1e-10 | 1e-10 | 1e-10 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| weight_decay | 0.0 | 0.0 | 0.0 | 0.0 |

### adam / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.adam.synthetic.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1219.0 | 1219.0..1219.0 | 1 | - | - | - | 1345.7 | - | rel_fro_vs_torch_eager_fp32=3.339e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested identical, read back unknown) |
| mojolearn FAST | mojolearn | gpu | fast | 1201.7 | 1201.7..1201.7 | 1 | - | - | - | 1344.2 | - | rel_fro_vs_torch_eager_fp32=3.34e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested fast, read back unknown) |
| torch-eager-fp32 | torch | gpu | opponent | 79.0 | 79.0..79.0 | 1 | - | - | - | 1402.7 | 1032.4 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 187.5 | 187.5..187.5 | 1 | - | - | - | 1492.4 | 1024.4 | rel_fro_vs_torch_eager_fp32=3.576e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-eager-fp32, torch-compile-fp32: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

settings: {'betas': [0.9, 0.999], 'eps': 1e-08, 'lr': 0.001, 'maximize': False, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | torch-compile-fp32 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | torch (optimizer.defaults) | torch (optimizer.defaults) |
| amsgrad | - | - | false | false |
| betas | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| eps | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| weight_decay | 0.0 | 0.0 | 0.0 | 0.0 |

### adamax / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.adamax.synthetic.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 519.4 | 519.4..519.4 | 1 | - | - | - | 1656.5 | - | rel_fro_vs_torch_eager_fp32=4.267e-09 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 524.8 | 524.8..524.8 | 1 | - | - | - | 1644.2 | - | rel_fro_vs_torch_eager_fp32=4.267e-09 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 72.8 | 72.8..72.8 | 1 | 7.133 | 7.206 | - | 1404.3 | 1032.4 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 203.1 | 203.1..203.1 | 1 | 2.557 | 2.583 | - | 1489.5 | 1024.4 | rel_fro_vs_torch_eager_fp32=7.477e-09 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-eager-fp32, torch-compile-fp32: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

settings: {'betas': [0.9, 0.999], 'eps': 1e-08, 'lr': 0.001, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | torch-compile-fp32 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | torch (optimizer.defaults) | torch (optimizer.defaults) |
| betas | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| eps | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| weight_decay | 0.0 | 0.0 | 0.0 | 0.0 |

### adamw / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.adamw.synthetic.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1214.4 | 1214.4..1214.4 | 1 | - | - | - | 1344.9 | - | rel_fro_vs_torch_eager_fp32=3.339e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested identical, read back unknown) |
| mojolearn FAST | mojolearn | gpu | fast | 1188.7 | 1188.7..1188.7 | 1 | - | - | - | 1345.7 | - | rel_fro_vs_torch_eager_fp32=3.34e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested fast, read back unknown) |
| torch-eager-fp32 | torch | gpu | opponent | 84.8 | 84.8..84.8 | 1 | - | - | - | 1404.4 | 1032.4 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 172.3 | 172.3..172.3 | 1 | - | - | - | 1490.7 | 1024.4 | rel_fro_vs_torch_eager_fp32=3.577e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-eager-fp32, torch-compile-fp32: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

settings: {'betas': [0.9, 0.999], 'eps': 1e-08, 'lr': 0.001, 'maximize': False, 'weight_decay': 0.01}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | torch-compile-fp32 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | torch (optimizer.defaults) | torch (optimizer.defaults) |
| amsgrad | - | - | false | false |
| betas | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| eps | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| weight_decay | 0.01 | 0.01 | 0.01 | 0.01 |

### additive-chi2 / istella (rows full, shape X 100000x220; Xq 1000x220; y 100000; yq 1000)

race: done, driver rc 0, log `logs/algos.additive-chi2.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 25.8 | 25.8..25.8 | 1 | - | - | - | 1007.4 | - | kernel_rel_error=0.087730 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 25.6 | 25.6..25.6 | 1 | - | - | - | 1008.6 | - | kernel_rel_error=0.087730 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 5.5 | 5.5..5.5 | 1 | 4.725 | 4.684 | - | 1085.1 | - | kernel_rel_error=0.087730 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 2.3 | 2.3..2.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 2.3 | 2.3..2.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 3.3 | 3.3..3.3 | 1 | 0.714 | 0.692 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### additive-chi2 / taxi (rows full, shape X 100000x11; Xq 1000x11; y 100000; yq 1000)

race: done, driver rc 0, log `logs/algos.additive-chi2.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1.4 | 1.4..1.4 | 1 | - | - | - | 110.9 | - | kernel_rel_error=0.093892 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1.4 | 1.4..1.4 | 1 | - | - | - | 112.5 | - | kernel_rel_error=0.093892 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 0.5 | 0.5..0.5 | 1 | 2.582 | 2.591 | - | 196.4 | - | kernel_rel_error=0.093892 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 0.9 | 0.9..0.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 0.9 | 0.9..0.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 0.3 | 0.3..0.3 | 1 | 2.810 | 2.784 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### affinity-prop / istella (rows full, shape X 5000x220; Xq 100000x220; y 5000; yq 100000)

race: done, driver rc 0, log `logs/algos.affinity-prop.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 2742.6 | 2742.6..2742.6 | 1 | - | - | - | 2530.7 | - | n_clusters=342, silhouette=0.089763 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1518.5 | 1518.5..1518.5 | 1 | - | - | - | 2534.1 | - | ari_vs_ours=1.000000, n_clusters=342, silhouette=0.089763 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 7619.4 | 7619.4..7619.4 | 1 | 0.360 | 0.199 | - | 1313.5 | - | ari_vs_ours=1.000000, n_clusters=342, silhouette=0.089763 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 508.8 | 508.8..508.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 504.0 | 504.0..504.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 33.0 | 33.0..33.0 | 1 | 15.437 | 15.290 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq) or fit labels(Xq)

inference call, ours-fast: predict(Xq) or fit labels(Xq)

inference call, sklearn-cpu: predict(Xq) or fit labels(Xq)

### affinity-prop / taxi (rows full, shape X 5000x11; Xq 100000x11; y 5000; yq 100000)

race: done, driver rc 0, log `logs/algos.affinity-prop.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1208.9 | 1208.9..1208.9 | 1 | - | - | - | 1652.8 | - | n_clusters=272, silhouette=0.184644 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1226.9 | 1226.9..1226.9 | 1 | - | - | - | 1654.2 | - | ari_vs_ours=1.000000, n_clusters=272, silhouette=0.184644 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 6413.2 | 6413.2..6413.2 | 1 | 0.188 | 0.191 | - | 404.8 | - | ari_vs_ours=1.000000, n_clusters=272, silhouette=0.184644 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 23.7 | 23.7..23.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 16.2 | 16.2..16.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 9.1 | 9.1..9.1 | 1 | 2.603 | 1.774 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq) or fit labels(Xq)

inference call, ours-fast: predict(Xq) or fit labels(Xq)

inference call, sklearn-cpu: predict(Xq) or fit labels(Xq)

### als / taxi-zones (rows full, shape X 129352x261; Xq 14373x261)

race: done, driver rc 0, log `logs/algos.als.taxi-zones.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(timeout: null) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(timeout: null) |
| implicit-cpu | implicit | cpu | opponent | 37466.9 | 37466.9..37466.9 | 1 | - | - | - | 290.0 | - | recall_at_10=0.056786 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host not sampled; GPU not sampled

memory, implicit-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'calculate_training_loss': False, 'cg_steps': 3, 'factors': 64, 'iterations': 15, 'random_state': 7, 'regularization': 0.01, 'use_cg': False}. Rows: None. Timed: None.

mismatch: implicit-gpu has only the conjugate-gradient solver (use_cg ignored there); ours and implicit-cpu solve each least-squares step exactly (use_cg=False)

mismatch: each library draws its own initial factors from random_state=7

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | implicit-cpu | ours | ours-fast |
|---|---||---|---||---|---|
| library (source) | implicit (declared) | mojolearn (get_params) | mojolearn (get_params) |
| alpha | 1.0 | 1.0 | 1.0 |
| n_estimators | 15 | 15 | 15 |
| seed | 7 | 7 | 7 |

### auto-theta / synthetic (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc 0, log `logs/algos.auto-theta.synthetic.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1430.3 | 1430.3..1430.3 | 1 | - | - | - | 65.9 | - | forecast_rmse=1.438912 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1303.7 | 1303.7..1303.7 | 1 | - | - | - | 67.2 | - | forecast_rmse=1.438072 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsforecast-cpu | statsforecast | cpu | opponent | 5428.4 | 5428.4..5428.4 | 1 | 0.263 | 0.240 | - | 183.9 | - | forecast_rmse=1.437804 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, statsforecast-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'decomposition_type': 'multiplicative', 'model': None, 'season_length': 24}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | statsforecast-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (attributes) | mojolearn (attributes) | statsforecast (declared) |
| alpha | null | null | - |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

accepted difference: ours-fast alpha: None on both ours and ours-fast: the same documented setting in both signatures

### auto-theta / taxi-hourly (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc 0, log `logs/algos.auto-theta.taxi-hourly.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 5492.8 | 5492.8..5492.8 | 1 | - | - | - | 68.0 | - | forecast_rmse=49.054628 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 4884.5 | 4884.5..4884.5 | 1 | - | - | - | 68.2 | - | forecast_rmse=49.313360 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsforecast-cpu | statsforecast | cpu | opponent | 9275.9 | 9275.9..9275.9 | 1 | 0.592 | 0.527 | - | 185.0 | - | forecast_rmse=49.273467 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, statsforecast-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'decomposition_type': 'multiplicative', 'model': None, 'season_length': 24}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | statsforecast-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (attributes) | mojolearn (attributes) | statsforecast (declared) |
| alpha | null | null | - |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

accepted difference: ours-fast alpha: None on both ours and ours-fast: the same documented setting in both signatures

### autoarima / synthetic (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc 0, log `logs/algos.autoarima.synthetic.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "NotImplementedError(\"AutoARIMA: seasonal_test='seas' (statsmodels' STL in the reference) is not implemented; pass D as one integer\")", "event": "error", "stage": "round 0"}) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "NotImplementedError(\"AutoARIMA: seasonal_test='seas' (statsmodels' STL in the reference) is not implemented; pass D as one integer\")", "event": "error", "stage": "round 0"}) |
| statsforecast-cpu | statsforecast | cpu | opponent | 5986.0 | 5986.0..5986.0 | 1 | - | - | - | 184.1 | - | forecast_rmse=17.546480 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host not sampled; GPU not sampled

memory, statsforecast-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'allow_intercept': True, 'ic': 'aicc', 'max_d': 1, 'max_p': 3, 'max_q': 3, 'seasonal': False, 'stepwise': False}. Rows: None. Timed: None.

mismatch: the likelihood optimizer and its stopping rule are each library's own

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | statsforecast-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | statsforecast (declared) |
| seasonal | false | false | false |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### autoarima / taxi-hourly (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc 0, log `logs/algos.autoarima.taxi-hourly.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "NotImplementedError(\"AutoARIMA: seasonal_test='seas' (statsmodels' STL in the reference) is not implemented; pass D as one integer\")", "event": "error", "stage": "round 0"}) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "NotImplementedError(\"AutoARIMA: seasonal_test='seas' (statsmodels' STL in the reference) is not implemented; pass D as one integer\")", "event": "error", "stage": "round 0"}) |
| statsforecast-cpu | statsforecast | cpu | opponent | 6280.0 | 6280.0..6280.0 | 1 | - | - | - | 190.0 | - | forecast_rmse=68.211648 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host not sampled; GPU not sampled

memory, statsforecast-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'allow_intercept': True, 'ic': 'aicc', 'max_d': 1, 'max_p': 3, 'max_q': 3, 'seasonal': False, 'stepwise': False}. Rows: None. Timed: None.

mismatch: the likelihood optimizer and its stopping rule are each library's own

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | statsforecast-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | statsforecast (declared) |
| seasonal | false | false | false |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### bagging-clf / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.bagging-clf.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 8919.3 | 8919.3..8919.3 | 1 | - | - | - | 5695.8 | - | accuracy=0.941670, logloss=0.149399 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 7062.2 | 7062.2..7062.2 | 1 | - | - | - | 5694.9 | - | accuracy=0.941670, logloss=0.149399 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 47461.8 | 47461.8..47461.8 | 1 | 0.188 | 0.149 | - | 1106.8 | - | accuracy=0.941440, logloss=0.152069 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'bootstrap': True, 'bootstrap_features': False, 'estimator': 'DecisionTreeClassifier(max_depth=12, random_state=7)', 'max_features': 1.0, 'max_samples': 1.0, 'n_estimators': 10, 'oob_score': False, 'random_state': 7}. Rows: None. Timed: None.

mismatch: ours' DecisionTree* is its forest builder with one tree: it splits on n_bins=128 quantile bins per feature (ours only; not a scikit-learn parameter), scikit-learn on exact thresholds

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| bootstrap | true | true | true |
| max_features | 1.0 | 1.0 | 1.0 |
| max_samples | 1.0 | 1.0 | 1.0 |
| n_estimators | 10 | 10 | 10 |
| seed | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 172.6 | 172.6..172.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 167.8 | 167.8..167.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 1121.5 | 1121.5..1121.5 | 1 | 0.154 | 0.150 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### bagging-clf / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.bagging-clf.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 897.7 | 897.7..897.7 | 1 | - | - | - | 490.5 | - | accuracy=0.767570, logloss=0.530450 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 885.6 | 885.6..885.6 | 1 | - | - | - | 485.0 | - | accuracy=0.767570, logloss=0.530450 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 2571.2 | 2571.2..2571.2 | 1 | 0.349 | 0.344 | - | 251.3 | - | accuracy=0.767650, logloss=0.530078 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'bootstrap': True, 'bootstrap_features': False, 'estimator': 'DecisionTreeClassifier(max_depth=12, random_state=7)', 'max_features': 1.0, 'max_samples': 1.0, 'n_estimators': 10, 'oob_score': False, 'random_state': 7}. Rows: None. Timed: None.

mismatch: ours' DecisionTree* is its forest builder with one tree: it splits on n_bins=128 quantile bins per feature (ours only; not a scikit-learn parameter), scikit-learn on exact thresholds

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| bootstrap | true | true | true |
| max_features | 1.0 | 1.0 | 1.0 |
| max_samples | 1.0 | 1.0 | 1.0 |
| n_estimators | 10 | 10 | 10 |
| seed | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 44.3 | 44.3..44.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 41.2 | 41.2..41.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 591.2 | 591.2..591.2 | 1 | 0.075 | 0.070 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### bagging-reg / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.bagging-reg.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 8582.8 | 8582.8..8582.8 | 1 | - | - | - | 5692.4 | - | finite=True, r2=0.522033, rmse=0.577510 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 6945.0 | 6945.0..6945.0 | 1 | - | - | - | 5691.7 | - | finite=True, r2=0.522033, rmse=0.577510 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 46121.8 | 46121.8..46121.8 | 1 | 0.186 | 0.151 | - | 1114.9 | - | finite=True, r2=0.518475, rmse=0.579656 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'bootstrap': True, 'bootstrap_features': False, 'estimator': 'DecisionTreeRegressor(max_depth=12, random_state=7)', 'max_features': 1.0, 'max_samples': 1.0, 'n_estimators': 10, 'oob_score': False, 'random_state': 7}. Rows: None. Timed: None.

mismatch: ours' DecisionTree* is its forest builder with one tree: it splits on n_bins=128 quantile bins per feature (ours only; not a scikit-learn parameter), scikit-learn on exact thresholds

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| bootstrap | true | true | true |
| max_features | 1.0 | 1.0 | 1.0 |
| max_samples | 1.0 | 1.0 | 1.0 |
| n_estimators | 10 | 10 | 10 |
| seed | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 128.4 | 128.4..128.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 126.3 | 126.3..126.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 720.1 | 720.1..720.1 | 1 | 0.178 | 0.175 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### bagging-reg / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.bagging-reg.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 862.2 | 862.2..862.2 | 1 | - | - | - | 482.7 | - | finite=True, r2=0.918767, rmse=4.539457 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 799.5 | 799.5..799.5 | 1 | - | - | - | 486.7 | - | finite=True, r2=0.918767, rmse=4.539457 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 2569.6 | 2569.6..2569.6 | 1 | 0.336 | 0.311 | - | 235.4 | - | finite=True, r2=0.938805, rmse=3.939993 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'bootstrap': True, 'bootstrap_features': False, 'estimator': 'DecisionTreeRegressor(max_depth=12, random_state=7)', 'max_features': 1.0, 'max_samples': 1.0, 'n_estimators': 10, 'oob_score': False, 'random_state': 7}. Rows: None. Timed: None.

mismatch: ours' DecisionTree* is its forest builder with one tree: it splits on n_bins=128 quantile bins per feature (ours only; not a scikit-learn parameter), scikit-learn on exact thresholds

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| bootstrap | true | true | true |
| max_features | 1.0 | 1.0 | 1.0 |
| max_samples | 1.0 | 1.0 | 1.0 |
| n_estimators | 10 | 10 | 10 |
| seed | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 26.2 | 26.2..26.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 26.0 | 26.0..26.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 350.1 | 350.1..350.1 | 1 | 0.075 | 0.074 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### bayesian-gmm / istella (rows full, shape X 100000x200; Xq 20000x200; y 100000; yq 20000)

race: failed, driver rc 1, log `logs/algos.bayesian-gmm.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "Exception('Fitting the mixture model failed because some components have ill-defined empirical covariance (for instance caused by singleton or collapsed samples). Try to decrease the number) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "Exception('Fitting the mixture model failed because some components have ill-defined empirical covariance (for instance caused by singleton or collapsed samples). Try to decrease the number) |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(error: {"error": "ValueError('Fitting the mixture model failed because some components have ill-defined empirical covariance (for instance caused by singleton or collapsed samples). Try to decrease the numbe) (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(error: {"error": "Exception('Fitting the mixture model failed because some components have ill-defined empirical covariance (for instance caused by singleton or collapsed samples). Try to decrease the number) |
| mojolearn FAST | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(error: {"error": "Exception('Fitting the mixture model failed because some components have ill-defined empirical covariance (for instance caused by singleton or collapsed samples). Try to decrease the number) |
| sklearn-cpu | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(error: {"error": "ValueError('Fitting the mixture model failed because some components have ill-defined empirical covariance (for instance caused by singleton or collapsed samples). Try to decrease the numbe) |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### bayesian-gmm / taxi (rows full, shape X 100000x11; Xq 20000x11; y 100000; yq 20000)

race: done, driver rc 0, log `logs/algos.bayesian-gmm.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3307.5 | 3307.5..3307.5 | 1 | - | - | - | 187.7 | - | mean_log_likelihood=4.895581 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1933.3 | 1933.3..1933.3 | 1 | - | - | - | 166.0 | - | mean_log_likelihood=4.895806 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 3889.0 | 3889.0..3889.0 | 1 | 0.850 | 0.497 | - | 251.6 | - | mean_log_likelihood=6.178321 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 17.5 | 17.5..17.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 17.1 | 17.1..17.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 9.1 | 9.1..9.1 | 1 | 1.926 | 1.888 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### bernoulli-nb / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.bernoulli-nb.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 944.8 | 944.8..944.8 | 1 | - | - | - | 4372.4 | - | accuracy=0.794050, logloss=5.350625 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 777.2 | 777.2..777.2 | 1 | - | - | - | 4377.6 | - | accuracy=0.794050, logloss=5.350576 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1137.3 | 1137.3..1137.3 | 1 | 0.831 | 0.683 | - | 3628.2 | - | accuracy=0.794050, logloss=4.278741 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'binarize': 0.0, 'fit_prior': True, 'force_alpha': True}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), BernoulliNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 1.0 | 1.0 | 1.0 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 140.6 | 140.6..140.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 139.9 | 139.9..139.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 239.8 | 239.8..239.8 | 1 | 0.586 | 0.583 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### bernoulli-nb / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.bernoulli-nb.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 262.4 | 262.4..262.4 | 1 | - | - | - | 301.3 | - | accuracy=0.755560, logloss=0.557803 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 51.3 | 51.3..51.3 | 1 | - | - | - | 305.4 | - | accuracy=0.755560, logloss=0.557803 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 111.4 | 111.4..111.4 | 1 | 2.357 | 0.461 | - | 240.7 | - | accuracy=0.755560, logloss=0.557802 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'binarize': 0.0, 'fit_prior': True, 'force_alpha': True}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), BernoulliNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 1.0 | 1.0 | 1.0 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 8.0 | 8.0..8.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 10.7 | 10.7..10.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 23.8 | 23.8..23.8 | 1 | 0.337 | 0.448 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### binarizer / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.binarizer.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 0.1 | 0.1..0.1 | 1 | - | - | - | 1661.6 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x220, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 0.1 | 0.1..0.1 | 1 | - | - | - | 1662.4 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x220, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 34.9 | 34.9..34.9 | 1 | 0.003 | 0.003 | - | 1342.4 | - | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'threshold': 0.0}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), Binarizer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 62.9 | 62.9..62.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 63.1 | 63.1..63.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 90.7 | 90.7..90.7 | 1 | 0.694 | 0.695 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### binarizer / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.binarizer.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 0.1 | 0.1..0.1 | 1 | - | - | - | 144.0 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x11, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 0.1 | 0.1..0.1 | 1 | - | - | - | 148.5 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x11, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 2.2 | 2.2..2.2 | 1 | 0.049 | 0.049 | - | 209.7 | - | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'threshold': 0.0}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), Binarizer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 3.6 | 3.6..3.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 3.5 | 3.5..3.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 6.1 | 6.1..6.1 | 1 | 0.589 | 0.584 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### bootstrap / istella (rows full, shape X 20000x220; Xq 100000x220; y 20000; yq 100000)

race: done, driver rc 0, log `logs/algos.bootstrap.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 51.9 | 51.9..51.9 | 1 | - | - | - | 1027.9 | - | ci_endpoint_diff_over_width_vs_scipy=0.027369, ci_high=0.294850, ci_low=0.271250, standard_error=0.005987 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested identical, read back unknown) |
| mojolearn FAST | mojolearn | gpu | fast | 35.1 | 35.1..35.1 | 1 | - | - | - | 1029.4 | - | ci_endpoint_diff_over_width_vs_scipy=0.027369, ci_high=0.294850, ci_low=0.271250, standard_error=0.005987 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested fast, read back unknown) |
| scipy-cpu | scipy | cpu | opponent | 699.1 | 699.1..699.1 | 1 | - | - | - | 1109.5 | - | ci_high=0.295500, ci_low=0.271750, standard_error=0.006044 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, scipy-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alternative': 'two-sided', 'confidence_level': 0.95, 'method': 'percentile', 'n_resamples': 9999, 'random_state': 7, 'statistic': 'mean'}. Rows: None. Timed: None.

mismatch: each library draws its resamples from its own generator seeded 7 (ours: the Philox position map; scipy: numpy default_rng(7)), so the intervals agree to Monte Carlo error, not bit for bit

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | scipy-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | scipy (declared) |
| seed | 7 | 7 | 7 |

### bootstrap / taxi (rows full, shape X 20000x11; Xq 100000x11; y 20000; yq 100000)

race: done, driver rc 0, log `logs/algos.bootstrap.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 41.7 | 41.7..41.7 | 1 | - | - | - | 132.0 | - | ci_endpoint_diff_over_width_vs_scipy=0.005021, ci_high=18.717379, ci_low=18.249411, standard_error=0.117837 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested identical, read back unknown) |
| mojolearn FAST | mojolearn | gpu | fast | 38.7 | 38.7..38.7 | 1 | - | - | - | 134.7 | - | ci_endpoint_diff_over_width_vs_scipy=0.005017, ci_high=18.717377, ci_low=18.249411, standard_error=0.117837 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested fast, read back unknown) |
| scipy-cpu | scipy | cpu | opponent | 700.6 | 700.6..700.6 | 1 | - | - | - | 215.9 | - | ci_high=18.715050, ci_low=18.251265, standard_error=0.117562 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, scipy-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alternative': 'two-sided', 'confidence_level': 0.95, 'method': 'percentile', 'n_resamples': 9999, 'random_state': 7, 'statistic': 'mean'}. Rows: None. Timed: None.

mismatch: each library draws its resamples from its own generator seeded 7 (ours: the Philox position map; scipy: numpy default_rng(7)), so the intervals agree to Monte Carlo error, not bit for bit

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | scipy-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | scipy (declared) |
| seed | 7 | 7 | 7 |

### categorical-nb / istella (rows full, shape X 1000000x8; Xq 100000x8; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.categorical-nb.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 325.2 | 325.2..325.2 | 1 | - | - | - | 254.7 | - | accuracy=0.838850, logloss=0.412625 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 135.1 | 135.1..135.1 | 1 | - | - | - | 259.9 | - | accuracy=0.838850, logloss=0.412625 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 121.4 | 121.4..121.4 | 1 | 2.678 | 1.113 | - | 215.8 | - | accuracy=0.838850, logloss=0.412625 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'fit_prior': True, 'force_alpha': True}. Rows: None. Timed: None.

mismatch: min_categories = every code seen in X or Xq on scikit-learn; ours refuses the option (option parity) and cuML has none

config: cuML benchmark (RAPIDS), CategoricalNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 1.0 | 1.0 | 1.0 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 26.4 | 26.4..26.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 8.7 | 8.7..8.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 21.1 | 21.1..21.1 | 1 | 1.251 | 0.411 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### categorical-nb / taxi (rows full, shape X 1000000x5; Xq 100000x5; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.categorical-nb.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 352.1 | 352.1..352.1 | 1 | - | - | - | 199.2 | - | accuracy=0.765850, logloss=0.538866 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 170.4 | 170.4..170.4 | 1 | - | - | - | 202.7 | - | accuracy=0.765850, logloss=0.538866 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 96.0 | 96.0..96.0 | 1 | 3.666 | 1.775 | - | 228.7 | - | accuracy=0.765850, logloss=0.538866 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'fit_prior': True, 'force_alpha': True}. Rows: None. Timed: None.

mismatch: min_categories = every code seen in X or Xq on scikit-learn; ours refuses the option (option parity) and cuML has none

config: cuML benchmark (RAPIDS), CategoricalNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 1.0 | 1.0 | 1.0 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 23.8 | 23.8..23.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 6.5 | 6.5..6.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 16.5 | 16.5..16.5 | 1 | 1.443 | 0.396 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### cca / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.cca.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 126157.8 | 126157.8..126157.8 | 1 | - | - | - | 5161.5 | - | mean_canonical_corr=0.998053 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 125215.7 | 125215.7..125215.7 | 1 | - | - | - | 5162.6 | - | mean_canonical_corr=0.998053 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 104922.9 | 104922.9..104922.9 | 1 | 1.202 | 1.193 | - | 6171.4 | - | mean_canonical_corr=0.999568 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'max_iter': 500, 'n_components': 2, 'scale': True, 'tol': 1e-06}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| max_iter | 500 | 500 | 500 |
| n_components | 2 | 2 | 2 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| tol | 1e-06 | 1e-06 | 1e-06 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 81.4 | 81.4..81.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 79.5 | 79.5..79.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 74.3 | 74.3..74.3 | 1 | 1.095 | 1.070 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### cca / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.cca.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 374.0 | 374.0..374.0 | 1 | - | - | - | 509.4 | - | mean_canonical_corr=0.576863 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 370.2 | 370.2..370.2 | 1 | - | - | - | 512.6 | - | mean_canonical_corr=0.576863 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 604.7 | 604.7..604.7 | 1 | 0.618 | 0.612 | - | 265.5 | - | mean_canonical_corr=0.576863 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'max_iter': 500, 'n_components': 2, 'scale': True, 'tol': 1e-06}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| max_iter | 500 | 500 | 500 |
| n_components | 2 | 2 | 2 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| tol | 1e-06 | 1e-06 | 1e-06 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 11.5 | 11.5..11.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 10.7 | 10.7..10.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 8.4 | 8.4..8.4 | 1 | 1.361 | 1.266 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### cholesky / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.cholesky.synthetic.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1382.1 | 1382.1..1382.1 | 1 | - | - | - | 1894.0 | - | relative_residual=2.9e-07 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1088.5 | 1088.5..1088.5 | 1 | - | - | - | 1880.2 | - | relative_residual=1.659e-07 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| numpy-cpu | numpy | cpu | opponent | 4074.6 | 4074.6..4074.6 | 1 | 0.339 | 0.267 | - | 810.4 | - | relative_residual=3.928e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-gpu | torch | gpu | opponent | 332.4 | 332.4..332.4 | 1 | 4.158 | 3.274 | - | 1201.4 | 1032.4 | relative_residual=5.449e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, numpy-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

memory, torch-gpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

settings: {'jitter': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | numpy-cpu | ours | ours-fast | torch-gpu |
|---|---||---|---||---|---||---|---|
| library (source) | numpy (declared) | mojolearn (declared) | mojolearn (declared) | torch (declared) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### classical-mds / istella (rows full, shape X 5000x220; Xq 2000x220)

race: done, driver rc 0, log `logs/algos.classical-mds.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(timeout: null) |
| mojolearn FAST | mojolearn | gpu | fast | 709.9 | 709.9..709.9 | 1 | - | - | - | 909.8 | - | trustworthiness_k15=0.830548 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 5735.2 | 5735.2..5735.2 | 1 | - | 0.124 | - | 202.8 | - | trustworthiness_k15=0.830548 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'n_components': 2}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | __main__ (attributes) |
| metric | "euclidean" | "euclidean" | - |
| n_components | 2 | 2 | 2 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### classical-mds / taxi (rows full, shape X 5000x11; Xq 2000x11)

race: done, driver rc 0, log `logs/algos.classical-mds.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(timeout: null) |
| mojolearn FAST | mojolearn | gpu | fast | 257.2 | 257.2..257.2 | 1 | - | - | - | 876.2 | - | trustworthiness_k15=0.765649 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 5279.4 | 5279.4..5279.4 | 1 | - | 0.049 | - | 185.3 | - | trustworthiness_k15=0.765649 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'n_components': 2}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | __main__ (attributes) |
| metric | "euclidean" | "euclidean" | - |
| n_components | 2 | 2 | 2 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### complement-nb / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.complement-nb.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 385.0 | 385.0..385.0 | 1 | - | - | - | 3619.4 | - | accuracy=0.849190, logloss=3.766564 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 245.6 | 245.6..245.6 | 1 | - | - | - | 3619.3 | - | accuracy=0.849360, logloss=3.762528 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 274.9 | 274.9..274.9 | 1 | 1.400 | 0.893 | - | 3706.6 | - | accuracy=0.849350, logloss=3.174763 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'fit_prior': True, 'force_alpha': True, 'norm': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), ComplementNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 1.0 | 1.0 | 1.0 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 60.4 | 60.4..60.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 41.8 | 41.8..41.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 61.2 | 61.2..61.2 | 1 | 0.987 | 0.682 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### complement-nb / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.complement-nb.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 216.8 | 216.8..216.8 | 1 | - | - | - | 262.5 | - | accuracy=0.677000, logloss=0.715601 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 26.1 | 26.1..26.1 | 1 | - | - | - | 267.0 | - | accuracy=0.678030, logloss=0.715492 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 54.3 | 54.3..54.3 | 1 | 3.991 | 0.480 | - | 277.6 | - | accuracy=0.678030, logloss=0.715493 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'fit_prior': True, 'force_alpha': True, 'norm': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), ComplementNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 1.0 | 1.0 | 1.0 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 6.6 | 6.6..6.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 6.5 | 6.5..6.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 11.8 | 11.8..11.8 | 1 | 0.561 | 0.549 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### complement-nb / text (rows full, shape X 86626x4096; Xq 9626x4096; y 86626; yq 9626)

race: done, driver rc 0, log `logs/algos.complement-nb.text.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 286.3 | 286.3..286.3 | 1 | - | - | - | 4279.8 | - | accuracy=0.983067, logloss=0.559491 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 486.0 | 486.0..486.0 | 1 | - | - | - | 4280.5 | - | accuracy=0.983067, logloss=0.559491 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 500.6 | 500.6..500.6 | 1 | 0.572 | 0.971 | - | 4360.1 | - | accuracy=0.983067, logloss=0.557285 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'fit_prior': True, 'force_alpha': True, 'norm': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), ComplementNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 1.0 | 1.0 | 1.0 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 88.7 | 88.7..88.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 98.1 | 98.1..98.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 96.0 | 96.0..96.0 | 1 | 0.923 | 1.022 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### connected-components / istella (rows full, shape X 20000x220; indices 304830; indices2 62848; indptr 20001; indptr2 20001; y 20000)

race: done, driver rc 0, log `logs/algos.connected-components.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3495.9 | 3495.9..3495.9 | 1 | - | - | - | 3137.4 | - | ari_vs_networkx=1.000000, n_components=81 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 3514.1 | 3514.1..3514.1 | 1 | - | - | - | 3957.3 | - | ari_vs_networkx=1.000000, n_components=81 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| networkx-cpu | networkx | cpu | opponent | 11.0 | 11.0..11.0 | 1 | 317.173 | 318.828 | - | 103.5 | - | n_components=81 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, networkx-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

mismatch: ours takes a dense adjacency matrix (its class's contract), built from the CSR graph before the clock; networkx and cuGraph take the graph itself

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | networkx-cpu | ours | ours-fast |
|---|---||---|---||---|---|
| library (source) | networkx (declared) | mojolearn (declared) | mojolearn (declared) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### connected-components / taxi (rows full, shape X 20000x11; indices 253708; indices2 54528; indptr 20001; indptr2 20001; y 20000)

race: done, driver rc 0, log `logs/algos.connected-components.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3005.6 | 3005.6..3005.6 | 1 | - | - | - | 3122.5 | - | ari_vs_networkx=1.000000, n_components=588 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 3000.2 | 3000.2..3000.2 | 1 | - | - | - | 4650.7 | - | ari_vs_networkx=1.000000, n_components=588 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| networkx-cpu | networkx | cpu | opponent | 10.9 | 10.9..10.9 | 1 | 274.891 | 274.401 | - | 92.2 | - | n_components=588 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, networkx-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

mismatch: ours takes a dense adjacency matrix (its class's contract), built from the CSR graph before the clock; networkx and cuGraph take the graph itself

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | networkx-cpu | ours | ours-fast |
|---|---||---|---||---|---|
| library (source) | networkx (declared) | mojolearn (declared) | mojolearn (declared) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### cross-val-score / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.cross-val-score.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 12433.9 | 12433.9..12433.9 | 1 | - | - | - | 4406.5 | - | max_fold_score_diff_vs_sklearn=0.017412, mean_r2=0.333971 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested identical, read back unknown) |
| mojolearn FAST | mojolearn | gpu | fast | 6894.1 | 6894.1..6894.1 | 1 | - | - | - | 4403.8 | - | max_fold_score_diff_vs_sklearn=0.015390, mean_r2=0.331418 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested fast, read back unknown) |
| sklearn-cpu | scikit-learn | cpu | opponent | 9905.6 | 9905.6..9905.6 | 1 | - | - | - | 1140.3 | - | mean_r2=0.319779 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'cv': 5, 'estimator': 'LinearRegression()', 'scoring': 'r2'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | sklearn (declared) |
| cv | 5 | 5 | 5 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### cross-val-score / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.cross-val-score.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 292.2 | 292.2..292.2 | 1 | - | - | - | 291.7 | - | max_fold_score_diff_vs_sklearn=5.007e-06, mean_r2=0.937955 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested identical, read back unknown) |
| mojolearn FAST | mojolearn | gpu | fast | 285.3 | 285.3..285.3 | 1 | - | - | - | 287.8 | - | max_fold_score_diff_vs_sklearn=4.172e-06, mean_r2=0.937955 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested fast, read back unknown) |
| sklearn-cpu | scikit-learn | cpu | opponent | 1221.8 | 1221.8..1221.8 | 1 | - | - | - | 265.1 | - | mean_r2=0.937956 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'cv': 5, 'estimator': 'LinearRegression()', 'scoring': 'r2'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | sklearn (declared) |
| cv | 5 | 5 | 5 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### croston-optimized / synthetic (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc 0, log `logs/algos.croston-optimized.synthetic.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 20.7 | 20.7..20.7 | 1 | - | - | - | 64.1 | - | forecast_rmse=1.675538 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 20.5 | 20.5..20.5 | 1 | - | - | - | 65.6 | - | forecast_rmse=1.675539 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsforecast-cpu | statsforecast | cpu | opponent | 63.5 | 63.5..63.5 | 1 | 0.327 | 0.324 | - | 185.0 | - | forecast_rmse=1.675539 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, statsforecast-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

mismatch: CrostonOptimized has no tuning parameter and no seed on either side

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | statsforecast-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (attributes) | mojolearn (attributes) | statsforecast (declared) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### croston-optimized / taxi-hourly (rows full, shape Yfit 48x1392; Yhold 48x48)

race: done, driver rc 0, log `logs/algos.croston-optimized.taxi-hourly.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 21.0 | 21.0..21.0 | 1 | - | - | - | 64.0 | - | forecast_rmse=1.398253 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 26.6 | 26.6..26.6 | 1 | - | - | - | 64.6 | - | forecast_rmse=1.398258 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsforecast-cpu | statsforecast | cpu | opponent | 35.9 | 35.9..35.9 | 1 | 0.586 | 0.740 | - | 185.7 | - | forecast_rmse=1.398240 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, statsforecast-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

mismatch: CrostonOptimized has no tuning parameter and no seed on either side

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | statsforecast-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (attributes) | mojolearn (attributes) | statsforecast (declared) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### croston-sba / synthetic (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc 0, log `logs/algos.croston-sba.synthetic.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3.8 | 3.8..3.8 | 1 | - | - | - | 65.3 | - | forecast_rmse=1.674465 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 4.1 | 4.1..4.1 | 1 | - | - | - | 64.8 | - | forecast_rmse=1.674465 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsforecast-cpu | statsforecast | cpu | opponent | 81.5 | 81.5..81.5 | 1 | 0.047 | 0.050 | - | 187.3 | - | forecast_rmse=1.674465 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, statsforecast-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

mismatch: CrostonSBA has no tuning parameter and no seed on either side

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | statsforecast-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (attributes) | mojolearn (attributes) | statsforecast (declared) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### croston-sba / taxi-hourly (rows full, shape Yfit 48x1392; Yhold 48x48)

race: done, driver rc 0, log `logs/algos.croston-sba.taxi-hourly.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3.6 | 3.6..3.6 | 1 | - | - | - | 63.2 | - | forecast_rmse=1.386677 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 3.9 | 3.9..3.9 | 1 | - | - | - | 65.6 | - | forecast_rmse=1.386677 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsforecast-cpu | statsforecast | cpu | opponent | 31.0 | 31.0..31.0 | 1 | 0.116 | 0.125 | - | 185.3 | - | forecast_rmse=1.386677 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, statsforecast-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

mismatch: CrostonSBA has no tuning parameter and no seed on either side

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | statsforecast-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (attributes) | mojolearn (attributes) | statsforecast (declared) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### croston / synthetic (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc 0, log `logs/algos.croston.synthetic.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3.7 | 3.7..3.7 | 1 | - | - | - | 65.1 | - | forecast_rmse=1.674840 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 3.8 | 3.8..3.8 | 1 | - | - | - | 63.8 | - | forecast_rmse=1.674840 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsforecast-cpu | statsforecast | cpu | opponent | 29.8 | 29.8..29.8 | 1 | 0.125 | 0.127 | - | 186.0 | - | forecast_rmse=1.674840 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, statsforecast-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

mismatch: CrostonClassic has no tuning parameter and no seed on either side (smoothing 0.1, fixed)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | statsforecast-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (attributes) | mojolearn (attributes) | statsforecast (declared) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### croston / taxi-hourly (rows full, shape Yfit 48x1392; Yhold 48x48)

race: done, driver rc 0, log `logs/algos.croston.taxi-hourly.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 4.1 | 4.1..4.1 | 1 | - | - | - | 64.4 | - | forecast_rmse=1.390261 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 3.9 | 3.9..3.9 | 1 | - | - | - | 64.8 | - | forecast_rmse=1.390261 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsforecast-cpu | statsforecast | cpu | opponent | 68.1 | 68.1..68.1 | 1 | 0.060 | 0.057 | - | 186.0 | - | forecast_rmse=1.390261 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, statsforecast-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

mismatch: CrostonClassic has no tuning parameter and no seed on either side (smoothing 0.1, fixed)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | statsforecast-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (attributes) | mojolearn (attributes) | statsforecast (declared) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### damped-ets / synthetic (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc 0, log `logs/algos.damped-ets.synthetic.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1229.2 | 1229.2..1229.2 | 1 | - | - | - | 67.1 | - | forecast_rmse=13.931153 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 823.0 | 823.0..823.0 | 1 | - | - | - | 66.2 | - | forecast_rmse=13.931185 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsmodels-cpu | statsmodels | cpu | opponent | 304.7 | 304.7..304.7 | 1 | 4.034 | 2.701 | - | 50.4 | - | forecast_rmse=26.588704 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| statsforecast-cpu | statsforecast | cpu | opponent | 197.2 | 197.2..197.2 | 1 | 6.233 | 4.173 | - | 184.8 | - | forecast_rmse=13.945986 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, statsmodels-cpu, statsforecast-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'damped': True, 'model': 'AAN', 'season_length': 1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | statsforecast-cpu | statsmodels-cpu |
|---|---||---|---||---|---||---|---|
| library (source) | mojolearn (attributes) | mojolearn (attributes) | statsforecast (declared) | statsmodels (declared) |
| alpha | null | null | - | - |
| damped_trend | - | - | - | true |
| gamma | null | null | - | - |
| initialization_method | - | - | - | "estimated" |
| seasonal | - | - | - | null |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| trend | - | - | - | "additive" |

accepted difference: ours-fast alpha: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: ours-fast gamma: None on both ours and ours-fast: the same documented setting in both signatures

### damped-ets / taxi-hourly (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc 0, log `logs/algos.damped-ets.taxi-hourly.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1356.4 | 1356.4..1356.4 | 1 | - | - | - | 69.0 | - | forecast_rmse=96.690449 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 945.1 | 945.1..945.1 | 1 | - | - | - | 68.7 | - | forecast_rmse=96.690681 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsmodels-cpu | statsmodels | cpu | opponent | 279.0 | 279.0..279.0 | 1 | 4.862 | 3.388 | - | 48.4 | - | forecast_rmse=196.928662 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| statsforecast-cpu | statsforecast | cpu | opponent | 170.2 | 170.2..170.2 | 1 | 7.969 | 5.553 | - | 186.3 | - | forecast_rmse=96.685568 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, statsmodels-cpu, statsforecast-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'damped': True, 'model': 'AAN', 'season_length': 1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | statsforecast-cpu | statsmodels-cpu |
|---|---||---|---||---|---||---|---|
| library (source) | mojolearn (attributes) | mojolearn (attributes) | statsforecast (declared) | statsmodels (declared) |
| alpha | null | null | - | - |
| damped_trend | - | - | - | true |
| gamma | null | null | - | - |
| initialization_method | - | - | - | "estimated" |
| seasonal | - | - | - | null |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| trend | - | - | - | "additive" |

accepted difference: ours-fast alpha: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: ours-fast gamma: None on both ours and ours-fast: the same documented setting in both signatures

### dart-reg / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.dart-reg.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 100860.2 | 100860.2..100860.2 | 1 | - | - | - | 4657.6 | - | finite=True, r2=0.550733, rmse=0.559903 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 99842.7 | 99842.7..99842.7 | 1 | - | - | - | 4639.3 | - | finite=True, r2=0.550733, rmse=0.559903 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| lightgbm-cpu | lightgbm | cpu | opponent | 33640.2 | 33640.2..33640.2 | 1 | 2.998 | 2.968 | - | 1602.2 | - | finite=True, r2=0.564663, rmse=0.551154 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| xgboost-cpu | xgboost | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(timeout: null) (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, lightgbm-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

memory, xgboost-cpu: host not sampled; GPU not sampled

settings: {'bagging_seed': 7, 'colsample_bytree': 1.0, 'drop_rate': 0.1, 'drop_seed': 7, 'feature_fraction_seed': 7, 'learning_rate': 0.1, 'max_bin': 255, 'max_delta_step': 0.0, 'max_depth': 8, 'max_drop': 50, 'min_child_samples': 20, 'n_estimators': 200, 'num_leaves': 255, 'random_state': 7, 'reg_alpha': 0.0, 'reg_lambda': 0.0, 'skip_drop': 0.5, 'subsample': 1.0, 'subsample_freq': 0, 'uniform_drop': False, 'xgboost_dart_mode': False}. Rows: None. Timed: None.

mismatch: LightGBM boosting='dart' grows leaf-wise (num_leaves=255, max_depth=8) as ours; XGBoost booster='dart' tree_method='hist' grows depth-wise (max_depth=8, no leaf cap)

mismatch: XGBoost has no max_drop (ours and LightGBM 50), no min_child_samples (ours and LightGBM 20 rows; XGBoost min_child_weight=1, a hessian sum), no uniform_drop / xgboost_dart_mode (XGBoost sample_type='uniform', normalize_type='tree') and no drop_seed: its drops come from random_state=7; each library's drop RNG is its own

mismatch: max_bin 255 on every arm (XGBoost's default is 256, set to 255 here)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | lightgbm-cpu | ours | ours-fast | xgboost-cpu |
|---|---||---|---||---|---||---|---|
| library (source) | lightgbm (get_params) | mojolearn (get_params) | mojolearn (get_params) | xgboost (get_params) |
| boosting_type | "dart" | - | - | "dart" |
| class_weight | null | - | - | - |
| feature_fraction | 1.0 | 1.0 | 1.0 | 1.0 |
| feature_fraction_bynode | - | - | - | null |
| grow_policy | - | - | - | null |
| learning_rate | 0.1 | 0.1 | 0.1 | 0.1 |
| max_bin | 255 | 255 | 255 | 255 |
| max_depth | 8 | 8 | 8 | 8 |
| max_leaves | 255 | 255 | 255 | null |
| min_child_weight | 0.001 | - | - | 1.0 |
| min_samples_leaf | 20 | 20 | 20 | - |
| min_split_gain | 0.0 | - | - | null |
| n_estimators | 200 | 200 | 200 | 200 |
| reg_alpha | 0.0 | 0.0 | 0.0 | 0.0 |
| reg_lambda | 0.0 | 0.0 | 0.0 | 0.0 |
| scale_pos_weight | - | - | - | 1.0 |
| seed | 7 | 7 | 7 | 7 |
| subsample | 1.0 | 1.0 | 1.0 | 1.0 |

accepted difference: xgboost-cpu max_leaves: XGBoost DART grows depth-wise (max_depth 8, no leaf cap); ours and LightGBM leaf-wise with num_leaves 255

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 214.7 | 214.7..214.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 309.9 | 309.9..309.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| lightgbm-cpu | Xq | - | 128.8 | 128.8..128.8 | 1 | 1.667 | 2.406 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| xgboost-cpu | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(timeout: null) |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, lightgbm-cpu: predict(Xq)(Xq)

inference call, xgboost-cpu: predict(Xq)(Xq)

### dart-reg / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.dart-reg.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 12379.9 | 12379.9..12379.9 | 1 | - | - | - | 981.4 | - | finite=True, r2=0.925497, rmse=4.347340 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 12384.9 | 12384.9..12384.9 | 1 | - | - | - | 988.8 | - | finite=True, r2=0.925497, rmse=4.347340 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| lightgbm-cpu | lightgbm | cpu | opponent | 24715.3 | 24715.3..24715.3 | 1 | 0.501 | 0.501 | - | 226.5 | - | finite=True, r2=0.926249, rmse=4.325358 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| xgboost-cpu | xgboost | cpu | opponent | 104076.7 | 104076.7..104076.7 | 1 | 0.119 | 0.119 | - | 254.0 | - | finite=True, r2=0.926049, rmse=4.331203 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, lightgbm-cpu, xgboost-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'bagging_seed': 7, 'colsample_bytree': 1.0, 'drop_rate': 0.1, 'drop_seed': 7, 'feature_fraction_seed': 7, 'learning_rate': 0.1, 'max_bin': 255, 'max_delta_step': 0.0, 'max_depth': 8, 'max_drop': 50, 'min_child_samples': 20, 'n_estimators': 200, 'num_leaves': 255, 'random_state': 7, 'reg_alpha': 0.0, 'reg_lambda': 0.0, 'skip_drop': 0.5, 'subsample': 1.0, 'subsample_freq': 0, 'uniform_drop': False, 'xgboost_dart_mode': False}. Rows: None. Timed: None.

mismatch: LightGBM boosting='dart' grows leaf-wise (num_leaves=255, max_depth=8) as ours; XGBoost booster='dart' tree_method='hist' grows depth-wise (max_depth=8, no leaf cap)

mismatch: XGBoost has no max_drop (ours and LightGBM 50), no min_child_samples (ours and LightGBM 20 rows; XGBoost min_child_weight=1, a hessian sum), no uniform_drop / xgboost_dart_mode (XGBoost sample_type='uniform', normalize_type='tree') and no drop_seed: its drops come from random_state=7; each library's drop RNG is its own

mismatch: max_bin 255 on every arm (XGBoost's default is 256, set to 255 here)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | lightgbm-cpu | ours | ours-fast | xgboost-cpu |
|---|---||---|---||---|---||---|---|
| library (source) | lightgbm (get_params) | mojolearn (get_params) | mojolearn (get_params) | xgboost (get_params) |
| boosting_type | "dart" | - | - | "dart" |
| class_weight | null | - | - | - |
| feature_fraction | 1.0 | 1.0 | 1.0 | 1.0 |
| feature_fraction_bynode | - | - | - | null |
| grow_policy | - | - | - | null |
| learning_rate | 0.1 | 0.1 | 0.1 | 0.1 |
| max_bin | 255 | 255 | 255 | 255 |
| max_depth | 8 | 8 | 8 | 8 |
| max_leaves | 255 | 255 | 255 | null |
| min_child_weight | 0.001 | - | - | 1.0 |
| min_samples_leaf | 20 | 20 | 20 | - |
| min_split_gain | 0.0 | - | - | null |
| n_estimators | 200 | 200 | 200 | 200 |
| reg_alpha | 0.0 | 0.0 | 0.0 | 0.0 |
| reg_lambda | 0.0 | 0.0 | 0.0 | 0.0 |
| scale_pos_weight | - | - | - | 1.0 |
| seed | 7 | 7 | 7 | 7 |
| subsample | 1.0 | 1.0 | 1.0 | 1.0 |

accepted difference: xgboost-cpu max_leaves: XGBoost DART grows depth-wise (max_depth 8, no leaf cap); ours and LightGBM leaf-wise with num_leaves 255

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 194.2 | 194.2..194.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 178.7 | 178.7..178.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| lightgbm-cpu | Xq | - | 120.2 | 120.2..120.2 | 1 | 1.617 | 1.488 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| xgboost-cpu | Xq | - | 132.5 | 132.5..132.5 | 1 | 1.465 | 1.349 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, lightgbm-cpu: predict(Xq)(Xq)

inference call, xgboost-cpu: predict(Xq)(Xq)

### dart / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.dart.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 101188.8 | 101188.8..101188.8 | 1 | - | - | - | 4406.6 | - | accuracy=0.948720, logloss=0.134126 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 99950.1 | 99950.1..99950.1 | 1 | - | - | - | 4647.5 | - | accuracy=0.948720, logloss=0.134126 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| lightgbm-cpu | lightgbm | cpu | opponent | 39154.3 | 39154.3..39154.3 | 1 | 2.584 | 2.553 | - | 1613.4 | - | accuracy=0.951860, logloss=0.123740 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| xgboost-cpu | xgboost | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(timeout: null) (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, lightgbm-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

memory, xgboost-cpu: host not sampled; GPU not sampled

settings: {'bagging_seed': 7, 'colsample_bytree': 1.0, 'drop_rate': 0.1, 'drop_seed': 7, 'feature_fraction_seed': 7, 'learning_rate': 0.1, 'max_bin': 255, 'max_delta_step': 0.0, 'max_depth': 8, 'max_drop': 50, 'min_child_samples': 20, 'n_estimators': 200, 'num_leaves': 255, 'random_state': 7, 'reg_alpha': 0.0, 'reg_lambda': 0.0, 'skip_drop': 0.5, 'subsample': 1.0, 'subsample_freq': 0, 'uniform_drop': False, 'xgboost_dart_mode': False}. Rows: None. Timed: None.

mismatch: LightGBM boosting='dart' grows leaf-wise (num_leaves=255, max_depth=8) as ours; XGBoost booster='dart' tree_method='hist' grows depth-wise (max_depth=8, no leaf cap)

mismatch: XGBoost has no max_drop (ours and LightGBM 50), no min_child_samples (ours and LightGBM 20 rows; XGBoost min_child_weight=1, a hessian sum), no uniform_drop / xgboost_dart_mode (XGBoost sample_type='uniform', normalize_type='tree') and no drop_seed: its drops come from random_state=7; each library's drop RNG is its own

mismatch: max_bin 255 on every arm (XGBoost's default is 256, set to 255 here)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | lightgbm-cpu | ours | ours-fast | xgboost-cpu |
|---|---||---|---||---|---||---|---|
| library (source) | lightgbm (get_params) | mojolearn (get_params) | mojolearn (get_params) | xgboost (get_params) |
| boosting_type | "dart" | - | - | "dart" |
| class_weight | null | - | - | - |
| feature_fraction | 1.0 | 1.0 | 1.0 | 1.0 |
| feature_fraction_bynode | - | - | - | null |
| grow_policy | - | - | - | null |
| learning_rate | 0.1 | 0.1 | 0.1 | 0.1 |
| max_bin | 255 | 255 | 255 | 255 |
| max_depth | 8 | 8 | 8 | 8 |
| max_leaves | 255 | 255 | 255 | null |
| min_child_weight | 0.001 | - | - | 1.0 |
| min_samples_leaf | 20 | 20 | 20 | - |
| min_split_gain | 0.0 | - | - | null |
| n_estimators | 200 | 200 | 200 | 200 |
| reg_alpha | 0.0 | 0.0 | 0.0 | 0.0 |
| reg_lambda | 0.0 | 0.0 | 0.0 | 0.0 |
| scale_pos_weight | - | - | - | 1.0 |
| seed | 7 | 7 | 7 | 7 |
| subsample | 1.0 | 1.0 | 1.0 | 1.0 |

accepted difference: xgboost-cpu max_leaves: XGBoost DART grows depth-wise (max_depth 8, no leaf cap); ours and LightGBM leaf-wise with num_leaves 255

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 437.5 | 437.5..437.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 440.1 | 440.1..440.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| lightgbm-cpu | Xq | - | 124.3 | 124.3..124.3 | 1 | 3.520 | 3.542 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| xgboost-cpu | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(timeout: null) |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, lightgbm-cpu: predict(Xq)(Xq)

inference call, xgboost-cpu: predict(Xq)(Xq)

### dart / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.dart.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 13258.7 | 13258.7..13258.7 | 1 | - | - | - | 1118.3 | - | accuracy=0.768340, logloss=0.529060 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 13264.5 | 13264.5..13264.5 | 1 | - | - | - | 990.7 | - | accuracy=0.768340, logloss=0.529060 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| lightgbm-cpu | lightgbm | cpu | opponent | 23149.5 | 23149.5..23149.5 | 1 | 0.573 | 0.573 | - | 262.7 | - | accuracy=0.768190, logloss=0.529108 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| xgboost-cpu | xgboost | cpu | opponent | 105541.5 | 105541.5..105541.5 | 1 | 0.126 | 0.126 | - | 263.4 | - | accuracy=0.768000, logloss=0.529540 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, lightgbm-cpu, xgboost-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'bagging_seed': 7, 'colsample_bytree': 1.0, 'drop_rate': 0.1, 'drop_seed': 7, 'feature_fraction_seed': 7, 'learning_rate': 0.1, 'max_bin': 255, 'max_delta_step': 0.0, 'max_depth': 8, 'max_drop': 50, 'min_child_samples': 20, 'n_estimators': 200, 'num_leaves': 255, 'random_state': 7, 'reg_alpha': 0.0, 'reg_lambda': 0.0, 'skip_drop': 0.5, 'subsample': 1.0, 'subsample_freq': 0, 'uniform_drop': False, 'xgboost_dart_mode': False}. Rows: None. Timed: None.

mismatch: LightGBM boosting='dart' grows leaf-wise (num_leaves=255, max_depth=8) as ours; XGBoost booster='dart' tree_method='hist' grows depth-wise (max_depth=8, no leaf cap)

mismatch: XGBoost has no max_drop (ours and LightGBM 50), no min_child_samples (ours and LightGBM 20 rows; XGBoost min_child_weight=1, a hessian sum), no uniform_drop / xgboost_dart_mode (XGBoost sample_type='uniform', normalize_type='tree') and no drop_seed: its drops come from random_state=7; each library's drop RNG is its own

mismatch: max_bin 255 on every arm (XGBoost's default is 256, set to 255 here)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | lightgbm-cpu | ours | ours-fast | xgboost-cpu |
|---|---||---|---||---|---||---|---|
| library (source) | lightgbm (get_params) | mojolearn (get_params) | mojolearn (get_params) | xgboost (get_params) |
| boosting_type | "dart" | - | - | "dart" |
| class_weight | null | - | - | - |
| feature_fraction | 1.0 | 1.0 | 1.0 | 1.0 |
| feature_fraction_bynode | - | - | - | null |
| grow_policy | - | - | - | null |
| learning_rate | 0.1 | 0.1 | 0.1 | 0.1 |
| max_bin | 255 | 255 | 255 | 255 |
| max_depth | 8 | 8 | 8 | 8 |
| max_leaves | 255 | 255 | 255 | null |
| min_child_weight | 0.001 | - | - | 1.0 |
| min_samples_leaf | 20 | 20 | 20 | - |
| min_split_gain | 0.0 | - | - | null |
| n_estimators | 200 | 200 | 200 | 200 |
| reg_alpha | 0.0 | 0.0 | 0.0 | 0.0 |
| reg_lambda | 0.0 | 0.0 | 0.0 | 0.0 |
| scale_pos_weight | - | - | - | 1.0 |
| seed | 7 | 7 | 7 | 7 |
| subsample | 1.0 | 1.0 | 1.0 | 1.0 |

accepted difference: xgboost-cpu max_leaves: XGBoost DART grows depth-wise (max_depth 8, no leaf cap); ours and LightGBM leaf-wise with num_leaves 255

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 340.4 | 340.4..340.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 386.6 | 386.6..386.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| lightgbm-cpu | Xq | - | 123.8 | 123.8..123.8 | 1 | 2.750 | 3.123 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| xgboost-cpu | Xq | - | 133.1 | 133.1..133.1 | 1 | 2.557 | 2.904 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, lightgbm-cpu: predict(Xq)(Xq)

inference call, xgboost-cpu: predict(Xq)(Xq)

### decision-tree-clf / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.decision-tree-clf.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1326.3 | 1326.3..1326.3 | 1 | - | - | - | 3863.1 | - | accuracy=0.935000, logloss=0.758303 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 866.2 | 866.2..866.2 | 1 | - | - | - | 3871.7 | - | accuracy=0.935000, logloss=0.758303 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 62864.7 | 62864.7..62864.7 | 1 | 0.021 | 0.014 | - | 1116.0 | - | accuracy=0.934410, logloss=0.791706 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'ccp_alpha': 0.0, 'criterion': 'gini', 'max_depth': 16, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'random_state': 7, 'splitter': 'best'}. Rows: None. Timed: None.

mismatch: cuml-gpu is cuML's forest with one tree, no bootstrap, every feature (no GPU single-tree class exists), n_bins=128 as ours

mismatch: ours' DecisionTree* is its forest builder with one tree: it splits on n_bins=128 quantile bins per feature (ours only; not a scikit-learn parameter), scikit-learn on exact thresholds

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| class_weight | null | null | null |
| criterion | "gini" | "gini" | "gini" |
| max_bin | 128 | 128 | - |
| max_depth | 16 | 16 | 16 |
| max_features | null | null | null |
| max_leaves | null | null | null |
| min_samples_leaf | 1 | 1 | 1 |
| min_split_gain | 0.0 | 0.0 | 0.0 |
| seed | 7 | 7 | 7 |

accepted difference: ours-fast class_weight: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: ours-fast max_features: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: ours-fast max_leaves: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: sklearn-cpu class_weight: None on both ours and sklearn-cpu: the same documented setting in both signatures

accepted difference: sklearn-cpu max_features: None on both ours and sklearn-cpu: the same documented setting in both signatures

accepted difference: sklearn-cpu max_leaves: None on both ours and sklearn-cpu: the same documented setting in both signatures

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 17.3 | 17.3..17.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 16.8 | 16.8..16.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 22.1 | 22.1..22.1 | 1 | 0.782 | 0.761 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### decision-tree-clf / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.decision-tree-clf.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 129.0 | 129.0..129.0 | 1 | - | - | - | 360.3 | - | accuracy=0.756300, logloss=1.202243 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 119.5 | 119.5..119.5 | 1 | - | - | - | 367.0 | - | accuracy=0.756300, logloss=1.202243 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 2583.3 | 2583.3..2583.3 | 1 | 0.050 | 0.046 | - | 233.1 | - | accuracy=0.756560, logloss=1.149550 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'ccp_alpha': 0.0, 'criterion': 'gini', 'max_depth': 16, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'random_state': 7, 'splitter': 'best'}. Rows: None. Timed: None.

mismatch: cuml-gpu is cuML's forest with one tree, no bootstrap, every feature (no GPU single-tree class exists), n_bins=128 as ours

mismatch: ours' DecisionTree* is its forest builder with one tree: it splits on n_bins=128 quantile bins per feature (ours only; not a scikit-learn parameter), scikit-learn on exact thresholds

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| class_weight | null | null | null |
| criterion | "gini" | "gini" | "gini" |
| max_bin | 128 | 128 | - |
| max_depth | 16 | 16 | 16 |
| max_features | null | null | null |
| max_leaves | null | null | null |
| min_samples_leaf | 1 | 1 | 1 |
| min_split_gain | 0.0 | 0.0 | 0.0 |
| seed | 7 | 7 | 7 |

accepted difference: ours-fast class_weight: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: ours-fast max_features: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: ours-fast max_leaves: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: sklearn-cpu class_weight: None on both ours and sklearn-cpu: the same documented setting in both signatures

accepted difference: sklearn-cpu max_features: None on both ours and sklearn-cpu: the same documented setting in both signatures

accepted difference: sklearn-cpu max_leaves: None on both ours and sklearn-cpu: the same documented setting in both signatures

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 4.6 | 4.6..4.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 5.2 | 5.2..5.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 12.8 | 12.8..12.8 | 1 | 0.360 | 0.406 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### decision-tree-reg / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.decision-tree-reg.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1215.6 | 1215.6..1215.6 | 1 | - | - | - | 3860.5 | - | finite=True, r2=0.379438, rmse=0.658041 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 827.2 | 827.2..827.2 | 1 | - | - | - | 3865.5 | - | finite=True, r2=0.379438, rmse=0.658041 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 60985.3 | 60985.3..60985.3 | 1 | 0.020 | 0.014 | - | 1096.3 | - | finite=True, r2=0.373177, rmse=0.661352 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'ccp_alpha': 0.0, 'criterion': 'squared_error', 'max_depth': 16, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'random_state': 7, 'splitter': 'best'}. Rows: None. Timed: None.

mismatch: cuml-gpu is cuML's forest with one tree (see decision-tree-clf)

mismatch: ours' DecisionTree* is its forest builder with one tree: it splits on n_bins=128 quantile bins per feature (ours only; not a scikit-learn parameter), scikit-learn on exact thresholds

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| criterion | "squared_error" | "squared_error" | "squared_error" |
| max_bin | 128 | 128 | - |
| max_depth | 16 | 16 | 16 |
| max_features | null | null | null |
| max_leaves | null | null | null |
| min_samples_leaf | 1 | 1 | 1 |
| min_split_gain | 0.0 | 0.0 | 0.0 |
| seed | 7 | 7 | 7 |

accepted difference: ours-fast max_features: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: ours-fast max_leaves: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: sklearn-cpu max_features: None on both ours and sklearn-cpu: the same documented setting in both signatures

accepted difference: sklearn-cpu max_leaves: None on both ours and sklearn-cpu: the same documented setting in both signatures

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 12.7 | 12.7..12.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 12.5 | 12.5..12.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 10.8 | 10.8..10.8 | 1 | 1.176 | 1.156 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### decision-tree-reg / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.decision-tree-reg.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 121.1 | 121.1..121.1 | 1 | - | - | - | 358.8 | - | finite=True, r2=0.862608, rmse=5.903617 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 109.4 | 109.4..109.4 | 1 | - | - | - | 364.9 | - | finite=True, r2=0.862608, rmse=5.903617 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 2577.2 | 2577.2..2577.2 | 1 | 0.047 | 0.042 | - | 209.4 | - | finite=True, r2=0.891507, rmse=5.246104 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'ccp_alpha': 0.0, 'criterion': 'squared_error', 'max_depth': 16, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'random_state': 7, 'splitter': 'best'}. Rows: None. Timed: None.

mismatch: cuml-gpu is cuML's forest with one tree (see decision-tree-clf)

mismatch: ours' DecisionTree* is its forest builder with one tree: it splits on n_bins=128 quantile bins per feature (ours only; not a scikit-learn parameter), scikit-learn on exact thresholds

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| criterion | "squared_error" | "squared_error" | "squared_error" |
| max_bin | 128 | 128 | - |
| max_depth | 16 | 16 | 16 |
| max_features | null | null | null |
| max_leaves | null | null | null |
| min_samples_leaf | 1 | 1 | 1 |
| min_split_gain | 0.0 | 0.0 | 0.0 |
| seed | 7 | 7 | 7 |

accepted difference: ours-fast max_features: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: ours-fast max_leaves: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: sklearn-cpu max_features: None on both ours and sklearn-cpu: the same documented setting in both signatures

accepted difference: sklearn-cpu max_leaves: None on both ours and sklearn-cpu: the same documented setting in both signatures

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 3.0 | 3.0..3.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 3.1 | 3.1..3.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 9.4 | 9.4..9.4 | 1 | 0.316 | 0.328 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### dict-learning / istella (rows full, shape X 20000x220; Xq 5000x220; y 20000; yq 5000)

race: done, driver rc 0, log `logs/algos.dict-learning.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 32099.6 | 32099.6..32099.6 | 1 | - | - | - | 1193.0 | - | component_sparsity=0.086364, relative_reconstruction_error=0.652121 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 31172.2 | 31172.2..31172.2 | 1 | - | - | - | 1194.4 | - | component_sparsity=0.086364, relative_reconstruction_error=0.652121 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 20922.7 | 20922.7..20922.7 | 1 | 1.534 | 1.490 | - | 1099.0 | - | component_sparsity=0.086364, relative_reconstruction_error=0.652121 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'fit_algorithm': 'cd', 'max_iter': 100, 'n_components': 16, 'positive_code': False, 'positive_dict': False, 'random_state': 7, 'split_sign': False, 'tol': 1e-08, 'transform_algorithm': 'lasso_cd', 'transform_max_iter': 1000}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 1.0 | 1.0 | 1.0 |
| max_iter | 100 | 100 | 100 |
| n_components | 16 | 16 | 16 |
| seed | 7 | 7 | 7 |
| tol | 1e-08 | 1e-08 | 1e-08 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 205.9 | 205.9..205.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 183.3 | 183.3..183.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 55.8 | 55.8..55.8 | 1 | 3.691 | 3.286 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### dict-learning / taxi (rows full, shape X 20000x11; Xq 5000x11; y 20000; yq 5000)

race: done, driver rc 0, log `logs/algos.dict-learning.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 16370.3 | 16370.3..16370.3 | 1 | - | - | - | 148.9 | - | component_sparsity=0.000000, relative_reconstruction_error=0.459169 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 15912.8 | 15912.8..15912.8 | 1 | - | - | - | 142.3 | - | component_sparsity=0.000000, relative_reconstruction_error=0.459173 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 15030.3 | 15030.3..15030.3 | 1 | 1.089 | 1.059 | - | 203.8 | - | component_sparsity=0.000000, relative_reconstruction_error=0.460601 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'fit_algorithm': 'cd', 'max_iter': 100, 'n_components': 16, 'positive_code': False, 'positive_dict': False, 'random_state': 7, 'split_sign': False, 'tol': 1e-08, 'transform_algorithm': 'lasso_cd', 'transform_max_iter': 1000}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 1.0 | 1.0 | 1.0 |
| max_iter | 100 | 100 | 100 |
| n_components | 16 | 16 | 16 |
| seed | 7 | 7 | 7 |
| tol | 1e-08 | 1e-08 | 1e-08 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 123.9 | 123.9..123.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 90.3 | 90.3..90.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 51.5 | 51.5..51.5 | 1 | 2.406 | 1.754 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### dynamic-optimized-theta / synthetic (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc 0, log `logs/algos.dynamic-optimized-theta.synthetic.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 711.3 | 711.3..711.3 | 1 | - | - | - | 65.8 | - | forecast_rmse=1.436278 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 634.7 | 634.7..634.7 | 1 | - | - | - | 66.7 | - | forecast_rmse=1.435976 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsforecast-cpu | statsforecast | cpu | opponent | 2341.8 | 2341.8..2341.8 | 1 | 0.304 | 0.271 | - | 186.6 | - | forecast_rmse=1.436045 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, statsforecast-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'decomposition_type': 'multiplicative', 'season_length': 24}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | statsforecast-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (attributes) | mojolearn (attributes) | statsforecast (declared) |
| alpha | null | null | - |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

accepted difference: ours-fast alpha: None on both ours and ours-fast: the same documented setting in both signatures

### dynamic-optimized-theta / taxi-hourly (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc 0, log `logs/algos.dynamic-optimized-theta.taxi-hourly.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3672.0 | 3672.0..3672.0 | 1 | - | - | - | 67.5 | - | forecast_rmse=49.086628 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 931.0 | 931.0..931.0 | 1 | - | - | - | 69.8 | - | forecast_rmse=49.083557 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsforecast-cpu | statsforecast | cpu | opponent | 2949.5 | 2949.5..2949.5 | 1 | 1.245 | 0.316 | - | 186.5 | - | forecast_rmse=49.314797 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, statsforecast-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'decomposition_type': 'multiplicative', 'season_length': 24}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | statsforecast-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (attributes) | mojolearn (attributes) | statsforecast (declared) |
| alpha | null | null | - |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

accepted difference: ours-fast alpha: None on both ours and ours-fast: the same documented setting in both signatures

### dynamic-theta / synthetic (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc 0, log `logs/algos.dynamic-theta.synthetic.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 176.0 | 176.0..176.0 | 1 | - | - | - | 69.7 | - | forecast_rmse=1.437256 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 160.0 | 160.0..160.0 | 1 | - | - | - | 68.2 | - | forecast_rmse=1.437165 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsforecast-cpu | statsforecast | cpu | opponent | 721.3 | 721.3..721.3 | 1 | 0.244 | 0.222 | - | 188.2 | - | forecast_rmse=1.437262 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, statsforecast-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'decomposition_type': 'multiplicative', 'season_length': 24}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | statsforecast-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (attributes) | mojolearn (attributes) | statsforecast (declared) |
| alpha | null | null | - |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

accepted difference: ours-fast alpha: None on both ours and ours-fast: the same documented setting in both signatures

### dynamic-theta / taxi-hourly (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc 0, log `logs/algos.dynamic-theta.taxi-hourly.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 586.9 | 586.9..586.9 | 1 | - | - | - | 67.6 | - | forecast_rmse=49.101249 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 515.7 | 515.7..515.7 | 1 | - | - | - | 67.3 | - | forecast_rmse=49.100462 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsforecast-cpu | statsforecast | cpu | opponent | 1736.2 | 1736.2..1736.2 | 1 | 0.338 | 0.297 | - | 186.6 | - | forecast_rmse=49.269843 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, statsforecast-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'decomposition_type': 'multiplicative', 'season_length': 24}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | statsforecast-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (attributes) | mojolearn (attributes) | statsforecast (declared) |
| alpha | null | null | - |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

accepted difference: ours-fast alpha: None on both ours and ours-fast: the same documented setting in both signatures

### eigh / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.eigh.synthetic.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(timeout: null) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(timeout: null) |
| numpy-cpu | numpy | cpu | opponent | 9863.2 | 9863.2..9863.2 | 1 | - | - | - | 257.9 | - | max_eigenvalue_error=3.49e-08, relative_residual=2.824e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-gpu | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "NotImplementedError(\"The operator 'aten::_linalg_eigh.eigenvalues' is not currently implemented for the MPS device. If you want this op to be considered for addition please comment on http) (measured this run) |

memory, ours, ours-fast, torch-gpu: host not sampled; GPU not sampled

memory, numpy-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'UPLO': 'L'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | numpy-cpu | ours | ours-fast | torch-gpu |
|---|---||---|---||---|---||---|---|
| library (source) | numpy (declared) | mojolearn (declared) | mojolearn (declared) | torch (declared) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### elliptic-envelope / istella (rows full, shape X 100000x220; Xq 100000x220; y 100000; yq 100000)

race: done, driver rc 0, log `logs/algos.elliptic-envelope.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(timeout: null) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "Exception(\"eigh: the Jacobi eigensolver did not converge in 15 sweeps at n = 220. An unconverged decomposition is not returned as if it were one; see DEVIATION 590. The remedy is more swee) |
| sklearn-cpu | scikit-learn | cpu | opponent | 53337.0 | 53337.0..53337.0 | 1 | - | - | - | 4533.5 | - | fraction_flagged=0.091570, jaccard_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host not sampled; GPU not sampled

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'contamination': 0.1, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| contamination | 0.1 | 0.1 | 0.1 |
| seed | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(timeout: null) |
| mojolearn FAST | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(error: {"error": "Exception(\"eigh: the Jacobi eigensolver did not converge in 15 sweeps at n = 220. An unconverged decomposition is not returned as if it were one; see DEVIATION 590. The remedy is more swee) |
| sklearn-cpu | Xq | - | 3923.2 | 3923.2..3923.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### elliptic-envelope / taxi (rows full, shape X 100000x11; Xq 100000x11; y 100000; yq 100000)

race: done, driver rc 0, log `logs/algos.elliptic-envelope.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(timeout: null) |
| mojolearn FAST | mojolearn | gpu | fast | 2230.1 | 2230.1..2230.1 | 1 | - | - | - | 261.4 | - | fraction_flagged=0.102370, jaccard_vs_sklearn=0.963574 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1526.2 | 1526.2..1526.2 | 1 | - | 1.461 | - | 278.0 | - | fraction_flagged=0.102470, jaccard_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'contamination': 0.1, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| contamination | 0.1 | 0.1 | 0.1 |
| seed | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(timeout: null) |
| mojolearn FAST | Xq | - | 9.2 | 9.2..9.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 6.2 | 6.2..6.2 | 1 | - | 1.483 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### factor-analysis / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.factor-analysis.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "Exception(\"the one-sided Jacobi SVD did not converge in 60 sweeps at n_cols = 220: the last sweep still performed 1 rotations against a tolerance of 9.536743e-07. The remedy is more sweeps) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "Exception(\"the one-sided Jacobi SVD did not converge in 60 sweeps at n_cols = 220: the last sweep still performed 1 rotations against a tolerance of 9.536743e-07. The remedy is more sweeps) |
| sklearn-cpu | scikit-learn | cpu | opponent | 51596.5 | 51596.5..51596.5 | 1 | - | - | - | 4448.4 | - | mean_log_likelihood=98.122830 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host not sampled; GPU not sampled

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'iterated_power': 3, 'max_iter': 1000, 'n_components': 8, 'random_state': 7, 'svd_method': 'randomized', 'tol': 0.01}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| max_iter | 1000 | 1000 | 1000 |
| n_components | 8 | 8 | 8 |
| seed | 7 | 7 | 7 |
| tol | 0.01 | 0.01 | 0.01 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(error: {"error": "Exception(\"the one-sided Jacobi SVD did not converge in 60 sweeps at n_cols = 220: the last sweep still performed 1 rotations against a tolerance of 9.536743e-07. The remedy is more sweeps) |
| mojolearn FAST | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(error: {"error": "Exception(\"the one-sided Jacobi SVD did not converge in 60 sweeps at n_cols = 220: the last sweep still performed 1 rotations against a tolerance of 9.536743e-07. The remedy is more sweeps) |
| sklearn-cpu | Xq | - | 37.1 | 37.1..37.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### factor-analysis / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.factor-analysis.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 225.2 | 225.2..225.2 | 1 | - | - | - | 473.7 | - | mean_log_likelihood=-14.823632 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 219.7 | 219.7..219.7 | 1 | - | - | - | 472.1 | - | mean_log_likelihood=-14.823632 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 189608.4 | 189608.4..189608.4 | 1 | 0.001 | 0.001 | - | 228.6 | - | mean_log_likelihood=-14.823723 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'iterated_power': 3, 'max_iter': 1000, 'n_components': 8, 'random_state': 7, 'svd_method': 'randomized', 'tol': 0.01}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| max_iter | 1000 | 1000 | 1000 |
| n_components | 8 | 8 | 8 |
| seed | 7 | 7 | 7 |
| tol | 0.01 | 0.01 | 0.01 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 6.7 | 6.7..6.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 6.6 | 6.6..6.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 4.1 | 4.1..4.1 | 1 | 1.619 | 1.585 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### fastica / istella (rows full, shape X 900000x220; Xq 100000x220)

race: done, driver rc 0, log `logs/algos.fastica.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3002.9 | 3002.9..3002.9 | 1 | - | - | - | 4046.7 | - | mean_abs_excess_kurtosis=356.288855 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 2738.3 | 2738.3..2738.3 | 1 | - | - | - | 4045.4 | - | mean_abs_excess_kurtosis=356.288699 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 20138.2 | 20138.2..20138.2 | 1 | 0.149 | 0.136 | - | 4033.3 | - | mean_abs_excess_kurtosis=922.531077 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'algorithm': 'parallel', 'fun': 'logcosh', 'max_iter': 200, 'n_components': 8, 'random_state': 7, 'tol': 0.0001, 'whiten': 'unit-variance', 'whiten_solver': 'svd'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

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

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 42.8 | 42.8..42.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 43.4 | 43.4..43.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 18.0 | 18.0..18.0 | 1 | 2.380 | 2.413 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### fastica / taxi (rows full, shape X 900000x11; Xq 100000x11)

race: done, driver rc 0, log `logs/algos.fastica.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 206.7 | 206.7..206.7 | 1 | - | - | - | 466.3 | - | mean_abs_excess_kurtosis=13.204824 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 192.4 | 192.4..192.4 | 1 | - | - | - | 469.2 | - | mean_abs_excess_kurtosis=13.209463 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 630.3 | 630.3..630.3 | 1 | 0.328 | 0.305 | - | 221.2 | - | mean_abs_excess_kurtosis=13.764740 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'algorithm': 'parallel', 'fun': 'logcosh', 'max_iter': 200, 'n_components': 8, 'random_state': 7, 'tol': 0.0001, 'whiten': 'unit-variance', 'whiten_solver': 'svd'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

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

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 4.6 | 4.6..4.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 4.3 | 4.3..4.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 2.6 | 2.6..2.6 | 1 | 1.803 | 1.667 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### garch / synthetic (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc 0, log `logs/algos.garch.synthetic.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3913.4 | 3913.4..3913.4 | 1 | - | - | - | 68.8 | - | mean_llf=-1938.223490 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 2116.3 | 2116.3..2116.3 | 1 | - | - | - | 67.3 | - | mean_llf=-1938.224617 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| arch-cpu | arch | cpu | opponent | 111.4 | 111.4..111.4 | 1 | 35.127 | 18.996 | - | 180.1 | - | mean_llf=-1938.221004 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, arch-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'dist': 'normal', 'mean': 'Constant', 'o': 0, 'p': 1, 'power': 2.0, 'q': 1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | arch-cpu | ours | ours-fast |
|---|---||---|---||---|---|
| library (source) | arch (declared) | mojolearn (attributes) | mojolearn (attributes) |
| p | 1 | 1 | 1 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### garch / taxi-hourly (rows full, shape Yfit 64x1391; Yhold 64x48)

race: done, driver rc 0, log `logs/algos.garch.taxi-hourly.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 6227.2 | 6227.2..6227.2 | 1 | - | - | - | 67.6 | - | mean_llf=-1132.712756 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 3837.5 | 3837.5..3837.5 | 1 | - | - | - | 67.2 | - | mean_llf=-1132.954899 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| arch-cpu | arch | cpu | opponent | 143.1 | 143.1..143.1 | 1 | 43.504 | 26.809 | - | 182.2 | - | mean_llf=-1129.806866 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, arch-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'dist': 'normal', 'mean': 'Constant', 'o': 0, 'p': 1, 'power': 2.0, 'q': 1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | arch-cpu | ours | ours-fast |
|---|---||---|---||---|---|
| library (source) | arch (declared) | mojolearn (attributes) | mojolearn (attributes) |
| p | 1 | 1 | 1 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### gaussian-nb / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.gaussian-nb.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 476.6 | 476.6..476.6 | 1 | - | - | - | 2695.0 | - | accuracy=0.876530, logloss=3.574225 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 327.7 | 327.7..327.7 | 1 | - | - | - | 2694.7 | - | accuracy=0.876570, logloss=3.574405 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 447.8 | 447.8..447.8 | 1 | 1.064 | 0.732 | - | 2564.8 | - | accuracy=0.876530, logloss=3.417392 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'var_smoothing': 1e-09}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), GaussianNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 61.5 | 61.5..61.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 44.5 | 44.5..44.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 217.0 | 217.0..217.0 | 1 | 0.283 | 0.205 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### gaussian-nb / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.gaussian-nb.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 312.5 | 312.5..312.5 | 1 | - | - | - | 221.7 | - | accuracy=0.719900, logloss=1.133898 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 30.8 | 30.8..30.8 | 1 | - | - | - | 221.7 | - | accuracy=0.719820, logloss=1.132247 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 98.3 | 98.3..98.3 | 1 | 3.179 | 0.314 | - | 224.9 | - | accuracy=0.719900, logloss=1.133898 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'var_smoothing': 1e-09}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), GaussianNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 6.3 | 6.3..6.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 6.7 | 6.7..6.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 15.2 | 15.2..15.2 | 1 | 0.414 | 0.440 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### gaussian-rp / istella (rows full, shape X 900000x220; Xq 100000x220)

race: done, driver rc 0, log `logs/algos.gaussian-rp.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 205.4 | 205.4..205.4 | 1 | - | - | - | 3984.7 | - | mean_abs_distortion=0.680693 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 210.4 | 210.4..210.4 | 1 | - | - | - | 3982.9 | - | mean_abs_distortion=0.680693 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 32.8 | 32.8..32.8 | 1 | 6.267 | 6.418 | - | 999.2 | - | mean_abs_distortion=0.177966 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'compute_inverse_components': False, 'eps': 0.1, 'n_components': 10, 'random_state': 7}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), GaussianRandomProjection (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| eps | 0.1 | 0.1 | 0.1 |
| n_components | 10 | 10 | 10 |
| seed | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 37.8 | 37.8..37.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 36.7 | 36.7..36.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 6.5 | 6.5..6.5 | 1 | 5.836 | 5.661 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### gaussian-rp / taxi (rows full, shape X 900000x11; Xq 100000x11)

race: done, driver rc 0, log `logs/algos.gaussian-rp.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 12.9 | 12.9..12.9 | 1 | - | - | - | 273.3 | - | mean_abs_distortion=0.345752 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 12.8 | 12.8..12.8 | 1 | - | - | - | 274.2 | - | mean_abs_distortion=0.345752 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 2.2 | 2.2..2.2 | 1 | 5.775 | 5.759 | - | 201.3 | - | mean_abs_distortion=0.339791 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'compute_inverse_components': False, 'eps': 0.1, 'n_components': 10, 'random_state': 7}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), GaussianRandomProjection (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| eps | 0.1 | 0.1 | 0.1 |
| n_components | 10 | 10 | 10 |
| seed | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 6.6 | 6.6..6.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 6.8 | 6.8..6.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 1.4 | 1.4..1.4 | 1 | 4.590 | 4.717 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### gru-clf / synthetic (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc 0, log `logs/algos.gru-clf.synthetic.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3302.8 | 3302.8..3302.8 | 1 | - | - | - | 402.2 | - | accuracy=0.971842, logloss=0.065835 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 2882.5 | 2882.5..2882.5 | 1 | - | - | - | 393.1 | - | accuracy=0.971842, logloss=0.065835 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host not sampled; GPU not sampled

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (attributes) | mojolearn (attributes) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| batch_size | 256 | 256 | 256 | 256 | 256 | 256 |
| betas | - | - | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| eps | - | - | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| hidden_size | 64 | 64 | 64 | 64 | 64 | 64 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| num_layers | 1 | 1 | 1 | 1 | 1 | 1 |
| seed | 7 | 7 | 7 | 7 | 7 | 7 |
| shuffle | true | true | true | true | true | true |
| weight_decay | - | - | 0.0 | 0.0 | 0.0 | 0.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 245.9 | 245.9..245.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 244.8 | 244.8..244.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | - | - | 0 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) |
| torch-compile-fp32 | Xq | - | - | - | 0 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) |
| torch-eager-bf16 | Xq | - | - | - | 0 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) |
| torch-compile-bf16 | Xq | - | - | - | 0 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, torch-eager-fp32: predict(Xq)(Xq)

inference call, torch-compile-fp32: predict(Xq)(Xq)

inference call, torch-eager-bf16: predict(Xq)(Xq)

inference call, torch-compile-bf16: predict(Xq)(Xq)

### gru-clf / taxi-hourly (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc 0, log `logs/algos.gru-clf.taxi-hourly.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3305.2 | 3305.2..3305.2 | 1 | - | - | - | 400.5 | - | accuracy=0.865668, logloss=0.305841 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 2875.9 | 2875.9..2875.9 | 1 | - | - | - | 398.6 | - | accuracy=0.865668, logloss=0.305841 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host not sampled; GPU not sampled

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (attributes) | mojolearn (attributes) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| batch_size | 256 | 256 | 256 | 256 | 256 | 256 |
| betas | - | - | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| eps | - | - | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| hidden_size | 64 | 64 | 64 | 64 | 64 | 64 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| num_layers | 1 | 1 | 1 | 1 | 1 | 1 |
| seed | 7 | 7 | 7 | 7 | 7 | 7 |
| shuffle | true | true | true | true | true | true |
| weight_decay | - | - | 0.0 | 0.0 | 0.0 | 0.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 245.8 | 245.8..245.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 246.1 | 246.1..246.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | - | - | 0 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) |
| torch-compile-fp32 | Xq | - | - | - | 0 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) |
| torch-eager-bf16 | Xq | - | - | - | 0 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) |
| torch-compile-bf16 | Xq | - | - | - | 0 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, torch-eager-fp32: predict(Xq)(Xq)

inference call, torch-compile-fp32: predict(Xq)(Xq)

inference call, torch-eager-bf16: predict(Xq)(Xq)

inference call, torch-compile-bf16: predict(Xq)(Xq)

### gru-reg / synthetic (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc 0, log `logs/algos.gru-reg.synthetic.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3318.0 | 3318.0..3318.0 | 1 | - | - | - | 381.4 | - | finite=True, r2=0.981946, rmse=0.155672 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 2873.4 | 2873.4..2873.4 | 1 | - | - | - | 389.7 | - | finite=True, r2=0.981946, rmse=0.155672 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host not sampled; GPU not sampled

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (attributes) | mojolearn (attributes) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| batch_size | 256 | 256 | 256 | 256 | 256 | 256 |
| betas | - | - | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| eps | - | - | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| hidden_size | 64 | 64 | 64 | 64 | 64 | 64 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| num_layers | 1 | 1 | 1 | 1 | 1 | 1 |
| seed | 7 | 7 | 7 | 7 | 7 | 7 |
| shuffle | true | true | true | true | true | true |
| weight_decay | - | - | 0.0 | 0.0 | 0.0 | 0.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 122.3 | 122.3..122.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 122.6 | 122.6..122.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | - | - | 0 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) |
| torch-compile-fp32 | Xq | - | - | - | 0 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) |
| torch-eager-bf16 | Xq | - | - | - | 0 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) |
| torch-compile-bf16 | Xq | - | - | - | 0 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, torch-eager-fp32: predict(Xq)(Xq)

inference call, torch-compile-fp32: predict(Xq)(Xq)

inference call, torch-eager-bf16: predict(Xq)(Xq)

inference call, torch-compile-bf16: predict(Xq)(Xq)

### gru-reg / taxi-hourly (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc 0, log `logs/algos.gru-reg.taxi-hourly.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3299.5 | 3299.5..3299.5 | 1 | - | - | - | 393.2 | - | finite=True, r2=0.748219, rmse=0.544182 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 2873.3 | 2873.3..2873.3 | 1 | - | - | - | 391.7 | - | finite=True, r2=0.748219, rmse=0.544182 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host not sampled; GPU not sampled

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (attributes) | mojolearn (attributes) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| batch_size | 256 | 256 | 256 | 256 | 256 | 256 |
| betas | - | - | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| eps | - | - | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| hidden_size | 64 | 64 | 64 | 64 | 64 | 64 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| num_layers | 1 | 1 | 1 | 1 | 1 | 1 |
| seed | 7 | 7 | 7 | 7 | 7 | 7 |
| shuffle | true | true | true | true | true | true |
| weight_decay | - | - | 0.0 | 0.0 | 0.0 | 0.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 122.8 | 122.8..122.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 122.5 | 122.5..122.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | - | - | 0 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) |
| torch-compile-fp32 | Xq | - | - | - | 0 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) |
| torch-eager-bf16 | Xq | - | - | - | 0 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) |
| torch-compile-bf16 | Xq | - | - | - | 0 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, torch-eager-fp32: predict(Xq)(Xq)

inference call, torch-compile-fp32: predict(Xq)(Xq)

inference call, torch-eager-bf16: predict(Xq)(Xq)

inference call, torch-compile-bf16: predict(Xq)(Xq)

### incremental-pca / istella (rows full, shape X 900000x220; Xq 100000x220)

race: done, driver rc 0, log `logs/algos.incremental-pca.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 4423.6 | 4423.6..4423.6 | 1 | - | - | - | 4116.5 | - | explained_variance_fraction=0.999994 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 3827.9 | 3827.9..3827.9 | 1 | - | - | - | 4112.3 | - | explained_variance_fraction=0.999994 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 6944.6 | 6944.6..6944.6 | 1 | 0.637 | 0.551 | - | 1777.9 | - | explained_variance_fraction=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'batch_size': 65536, 'n_components': 10, 'whiten': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), IncrementalPCA (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| batch_size | 65536 | 65536 | 65536 |
| n_components | 10 | 10 | 10 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| whiten | false | false | false |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 38.3 | 38.3..38.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 37.8 | 37.8..37.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 25.1 | 25.1..25.1 | 1 | 1.525 | 1.508 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### incremental-pca / taxi (rows full, shape X 900000x11; Xq 100000x11)

race: done, driver rc 0, log `logs/algos.incremental-pca.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 175.5 | 175.5..175.5 | 1 | - | - | - | 310.6 | - | explained_variance_fraction=0.999995 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 160.5 | 160.5..160.5 | 1 | - | - | - | 308.2 | - | explained_variance_fraction=0.999995 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 158.5 | 158.5..158.5 | 1 | 1.108 | 1.013 | - | 212.0 | - | explained_variance_fraction=0.999995 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'batch_size': 65536, 'n_components': 10, 'whiten': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), IncrementalPCA (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| batch_size | 65536 | 65536 | 65536 |
| n_components | 10 | 10 | 10 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| whiten | false | false | false |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 4.6 | 4.6..4.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 4.5 | 4.5..4.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 2.7 | 2.7..2.7 | 1 | 1.692 | 1.652 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### isomap / istella (rows full, shape X 10000x220; Xq 2000x220)

race: done, driver rc 0, log `logs/algos.isomap.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(timeout: null) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(timeout: null) |
| sklearn-cpu | scikit-learn | cpu | opponent | 15206.0 | 15206.0..15206.0 | 1 | - | - | - | 1707.2 | - | trustworthiness_k15=0.853294 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host not sampled; GPU not sampled

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'eigen_solver': 'auto', 'metric': 'minkowski', 'n_components': 2, 'n_neighbors': 10, 'neighbors_algorithm': 'auto', 'p': 2, 'path_method': 'auto', 'tol': 0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| max_iter | - | - | null |
| metric | "minkowski" | "minkowski" | "minkowski" |
| n_components | 2 | 2 | 2 |
| n_neighbors | 10 | 10 | 10 |
| p | 2 | 2 | 2 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| tol | - | - | 0 |

### isomap / taxi (rows full, shape X 10000x11; Xq 2000x11)

race: done, driver rc 0, log `logs/algos.isomap.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(timeout: null) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(timeout: null) |
| sklearn-cpu | scikit-learn | cpu | opponent | 15590.0 | 15590.0..15590.0 | 1 | - | - | - | 1682.8 | - | trustworthiness_k15=0.771827 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host not sampled; GPU not sampled

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'eigen_solver': 'auto', 'metric': 'minkowski', 'n_components': 2, 'n_neighbors': 10, 'neighbors_algorithm': 'auto', 'p': 2, 'path_method': 'auto', 'tol': 0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| max_iter | - | - | null |
| metric | "minkowski" | "minkowski" | "minkowski" |
| n_components | 2 | 2 | 2 |
| n_neighbors | 10 | 10 | 10 |
| p | 2 | 2 | 2 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| tol | - | - | 0 |

### iterative-imputer / istella (rows full, shape X 100000x32; X_true 100000x32; Xq 20000x32; Xq_true 20000x32; y 100000; yq 20000)

race: done, driver rc 0, log `logs/algos.iterative-imputer.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 11466.0 | 11466.0..11466.0 | 1 | - | - | - | 1183.3 | - | masked_rmse=799013.020330, max_abs_diff_vs_sklearn=1.011e+07 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 9425.9 | 9425.9..9425.9 | 1 | - | - | - | 1189.5 | - | masked_rmse=799040.699151, max_abs_diff_vs_sklearn=9.974e+06 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 12295.4 | 12295.4..12295.4 | 1 | 0.933 | 0.767 | - | 1126.3 | - | masked_rmse=802428.936225 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'add_indicator': False, 'imputation_order': 'ascending', 'initial_strategy': 'mean', 'keep_empty_features': False, 'max_iter': 10, 'max_value': inf, 'min_value': -inf, 'random_state': 7, 'sample_posterior': False, 'skip_complete': False, 'tol': 0.001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| max_iter | 10 | 10 | 10 |
| seed | 7 | 7 | 7 |
| tol | 0.001 | 0.001 | 0.001 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 11.9 | 11.9..11.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 14.8 | 14.8..14.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 107.3 | 107.3..107.3 | 1 | 0.111 | 0.138 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### iterative-imputer / taxi (rows full, shape X 100000x11; X_true 100000x11; Xq 20000x11; Xq_true 20000x11; y 100000; yq 20000)

race: done, driver rc 0, log `logs/algos.iterative-imputer.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 2331.1 | 2331.1..2331.1 | 1 | - | - | - | 209.7 | - | masked_rmse=4.693848, max_abs_diff_vs_sklearn=0.044070 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 380.6 | 380.6..380.6 | 1 | - | - | - | 216.7 | - | masked_rmse=4.693971, max_abs_diff_vs_sklearn=0.0003719 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1264.9 | 1264.9..1264.9 | 1 | 1.843 | 0.301 | - | 218.8 | - | masked_rmse=4.693973 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'add_indicator': False, 'imputation_order': 'ascending', 'initial_strategy': 'mean', 'keep_empty_features': False, 'max_iter': 10, 'max_value': inf, 'min_value': -inf, 'random_state': 7, 'sample_posterior': False, 'skip_complete': False, 'tol': 0.001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| max_iter | 10 | 10 | 10 |
| seed | 7 | 7 | 7 |
| tol | 0.001 | 0.001 | 0.001 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 6.7 | 6.7..6.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 6.5 | 6.5..6.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 19.1 | 19.1..19.1 | 1 | 0.350 | 0.338 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### kbins / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.kbins.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 5779.7 | 5779.7..5779.7 | 1 | - | - | - | 4744.7 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x220, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 925.7 | 925.7..925.7 | 1 | - | - | - | 5640.9 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x220, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 4820.5 | 4820.5..4820.5 | 1 | 1.199 | 0.192 | - | 1329.4 | - | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'encode': 'ordinal', 'n_bins': 16, 'quantile_method': 'linear', 'random_state': 7, 'strategy': 'quantile', 'subsample': None}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), KBinsDiscretizer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| max_bin | 16 | 16 | - |
| seed | 7 | 7 | 7 |
| subsample | null | null | null |

accepted difference: ours-fast subsample: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: sklearn-cpu subsample: None on both ours and sklearn-cpu: the same documented setting in both signatures

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 76.6 | 76.6..76.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 118.1 | 118.1..118.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 186.4 | 186.4..186.4 | 1 | 0.411 | 0.633 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### kbins / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.kbins.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 383.8 | 383.8..383.8 | 1 | - | - | - | 321.2 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x11, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 41.7 | 41.7..41.7 | 1 | - | - | - | 353.4 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x11, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 132.1 | 132.1..132.1 | 1 | 2.905 | 0.316 | - | 208.2 | - | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'encode': 'ordinal', 'n_bins': 16, 'quantile_method': 'linear', 'random_state': 7, 'strategy': 'quantile', 'subsample': None}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), KBinsDiscretizer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| max_bin | 16 | 16 | - |
| seed | 7 | 7 | 7 |
| subsample | null | null | null |

accepted difference: ours-fast subsample: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: sklearn-cpu subsample: None on both ours and sklearn-cpu: the same documented setting in both signatures

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 3.5 | 3.5..3.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 3.8 | 3.8..3.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 8.3 | 8.3..8.3 | 1 | 0.417 | 0.463 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### kernel-pca / istella (rows full, shape X 10000x220; Xq 10000x220; y 10000; yq 10000)

race: done, driver rc 0, log `logs/algos.kernel-pca.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 2215.5 | 2215.5..2215.5 | 1 | - | - | - | 2377.2 | - | subspace_cos_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 2202.2 | 2202.2..2202.2 | 1 | - | - | - | 2378.5 | - | subspace_cos_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1045.7 | 1045.7..1045.7 | 1 | 2.119 | 2.106 | - | 1127.8 | - | subspace_cos_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'coef0': 1, 'degree': 3, 'eigen_solver': 'auto', 'fit_inverse_transform': False, 'iterated_power': 'auto', 'kernel': 'rbf', 'n_components': 8, 'random_state': 7, 'remove_zero_eig': False, 'tol': 0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

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

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 1375.7 | 1375.7..1375.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 1373.5 | 1373.5..1373.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 278.3 | 278.3..278.3 | 1 | 4.944 | 4.936 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### kernel-pca / taxi (rows full, shape X 10000x11; Xq 10000x11; y 10000; yq 10000)

race: done, driver rc 0, log `logs/algos.kernel-pca.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 948.2 | 948.2..948.2 | 1 | - | - | - | 1464.6 | - | subspace_cos_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 932.7 | 932.7..932.7 | 1 | - | - | - | 1468.1 | - | subspace_cos_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 966.6 | 966.6..966.6 | 1 | 0.981 | 0.965 | - | 253.1 | - | subspace_cos_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'coef0': 1, 'degree': 3, 'eigen_solver': 'auto', 'fit_inverse_transform': False, 'iterated_power': 'auto', 'kernel': 'rbf', 'n_components': 8, 'random_state': 7, 'remove_zero_eig': False, 'tol': 0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

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

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 99.8 | 99.8..99.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 98.1 | 98.1..98.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 320.3 | 320.3..320.3 | 1 | 0.312 | 0.306 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### knn-imputer / istella (rows full, shape X 100000x220; X_true 100000x220; Xq 20000x220; Xq_true 20000x220; y 100000; yq 20000)

race: done, driver rc 0, log `logs/algos.knn-imputer.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 992.1 | 992.1..992.1 | 1 | - | - | - | 2209.9 | - | masked_rmse=nan | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 970.8 | 970.8..970.8 | 1 | - | - | - | 2216.4 | - | masked_rmse=nan | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(timeout: null) (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host not sampled; GPU not sampled

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
| mojolearn IDENTICAL | Xq | - | 360.4 | 360.4..360.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 57340.9 | 57340.9..57340.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(timeout: null) |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### knn-imputer / taxi (rows full, shape X 100000x11; X_true 100000x11; Xq 20000x11; Xq_true 20000x11; y 100000; yq 20000)

race: done, driver rc 0, log `logs/algos.knn-imputer.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 73.3 | 73.3..73.3 | 1 | - | - | - | 203.3 | - | masked_rmse=6.151696, max_abs_diff_vs_sklearn=29.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 71.7 | 71.7..71.7 | 1 | - | - | - | 202.6 | - | masked_rmse=6.151696, max_abs_diff_vs_sklearn=29.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 2.2 | 2.2..2.2 | 1 | 33.404 | 32.690 | - | 1245.1 | - | masked_rmse=5.256719 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 2085.4 | 2085.4..2085.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 1914.0 | 1914.0..1914.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 32272.8 | 32272.8..32272.8 | 1 | 0.065 | 0.059 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### label-binarizer / istella (rows full, shape X 1000000x8; Xq 100000x8; lab 1000000; labq 100000; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.label-binarizer.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 432.4 | 432.4..432.4 | 1 | - | - | - | 491.0 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x16, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 424.5 | 424.5..424.5 | 1 | - | - | - | 474.8 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x16, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 44.6 | 44.6..44.6 | 1 | 9.684 | 9.508 | - | 472.5 | - | output_shape=100000x16 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'neg_label': 0, 'pos_label': 1}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), LabelBinarizer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 20.0 | 20.0..20.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 20.7 | 20.7..20.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 3.1 | 3.1..3.1 | 1 | 6.516 | 6.717 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### label-binarizer / taxi (rows full, shape X 1000000x5; Xq 100000x5; lab 1000000; labq 100000; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.label-binarizer.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1415.1 | 1415.1..1415.1 | 1 | - | - | - | 5382.6 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x259, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1399.4 | 1399.4..1399.4 | 1 | - | - | - | 5368.5 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x259, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 137.0 | 137.0..137.0 | 1 | 10.333 | 10.218 | - | 6506.7 | - | output_shape=100000x259 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'neg_label': 0, 'pos_label': 1}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), LabelBinarizer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 100.1 | 100.1..100.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 92.6 | 92.6..92.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 12.1 | 12.1..12.1 | 1 | 8.258 | 7.646 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### label-encoder / istella (rows full, shape X 1000000x8; Xq 100000x8; lab 1000000; labq 100000; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.label-encoder.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 383.6 | 383.6..383.6 | 1 | - | - | - | 205.7 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 374.9 | 374.9..374.9 | 1 | - | - | - | 208.7 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 28.4 | 28.4..28.4 | 1 | 13.511 | 13.203 | - | 216.8 | - | output_shape=100000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), LabelEncoder (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 17.5 | 17.5..17.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 18.1 | 18.1..18.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 1.2 | 1.2..1.2 | 1 | 14.197 | 14.733 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### label-encoder / taxi (rows full, shape X 1000000x5; Xq 100000x5; lab 1000000; labq 100000; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.label-encoder.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 398.1 | 398.1..398.1 | 1 | - | - | - | 191.3 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 380.4 | 380.4..380.4 | 1 | - | - | - | 189.0 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 34.9 | 34.9..34.9 | 1 | 11.402 | 10.894 | - | 200.8 | - | output_shape=100000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), LabelEncoder (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 19.2 | 19.2..19.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 18.3 | 18.3..18.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 1.5 | 1.5..1.5 | 1 | 13.079 | 12.485 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### label-propagation / istella (rows full, shape X 200000x220; Xq 20000x220; y 200000; y_semi 200000; yq 20000)

race: done, driver rc 0, log `logs/algos.label-propagation.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 51833.0 | 51833.0..51833.0 | 1 | - | - | - | 1602.0 | - | accuracy=0.125400 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 50910.6 | 50910.6..50910.6 | 1 | - | - | - | 1615.9 | - | accuracy=0.125400 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 34050.2 | 34050.2..34050.2 | 1 | 1.522 | 1.495 | - | 1314.2 | - | accuracy=0.905500 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

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
| mojolearn IDENTICAL | Xq | - | 169605.8 | 169605.8..169605.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 151570.6 | 151570.6..151570.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 2994.6 | 2994.6..2994.6 | 1 | 56.637 | 50.614 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### label-propagation / taxi (rows full, shape X 200000x11; Xq 20000x11; y 200000; y_semi 200000; yq 20000)

race: done, driver rc 0, log `logs/algos.label-propagation.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 26757.1 | 26757.1..26757.1 | 1 | - | - | - | 222.3 | - | accuracy=0.701600 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 18661.8 | 18661.8..18661.8 | 1 | - | - | - | 222.8 | - | accuracy=0.699200 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 9571.0 | 9571.0..9571.0 | 1 | 2.796 | 1.950 | - | 288.7 | - | accuracy=0.701600 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

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
| mojolearn IDENTICAL | Xq | - | 2215.8 | 2215.8..2215.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 2196.5 | 2196.5..2196.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 675.9 | 675.9..675.9 | 1 | 3.278 | 3.250 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### label-spreading / istella (rows full, shape X 200000x220; Xq 20000x220; y 200000; y_semi 200000; yq 20000)

race: done, driver rc 0, log `logs/algos.label-spreading.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 56529.2 | 56529.2..56529.2 | 1 | - | - | - | 1627.8 | - | accuracy=0.156950 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 56234.3 | 56234.3..56234.3 | 1 | - | - | - | 1611.2 | - | accuracy=0.156000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 28827.8 | 28827.8..28827.8 | 1 | 1.961 | 1.951 | - | 1307.5 | - | accuracy=0.904450 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

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
| mojolearn IDENTICAL | Xq | - | 169556.3 | 169556.3..169556.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 151625.1 | 151625.1..151625.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 3039.3 | 3039.3..3039.3 | 1 | 55.787 | 49.888 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### label-spreading / taxi (rows full, shape X 200000x11; Xq 20000x11; y 200000; y_semi 200000; yq 20000)

race: done, driver rc 0, log `logs/algos.label-spreading.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 19369.3 | 19369.3..19369.3 | 1 | - | - | - | 230.1 | - | accuracy=0.676400 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 19205.9 | 19205.9..19205.9 | 1 | - | - | - | 232.7 | - | accuracy=0.676400 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 4693.7 | 4693.7..4693.7 | 1 | 4.127 | 4.092 | - | 303.5 | - | accuracy=0.676400 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

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
| mojolearn IDENTICAL | Xq | - | 2217.3 | 2217.3..2217.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 2180.1 | 2180.1..2180.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 667.7 | 667.7..667.7 | 1 | 3.321 | 3.265 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### lamb / synthetic (rows full, shape -)

race: failed, driver rc 1, log `logs/algos.lamb.synthetic.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "Exception('lamb_step: offsets must rise strictly from 0, below 2^24')", "event": "error", "stage": "round 0"}) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "Exception('lamb_step: offsets must rise strictly from 0, below 2^24')", "event": "error", "stage": "round 0"}) |

settings: {'betas': [0.9, 0.999], 'eps': 1e-06, 'lr': 0.001, 'weight_decay': 0.01}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast |
|---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) |
| betas | [0.9, 0.999] | [0.9, 0.999] |
| eps | 1e-06 | 1e-06 |
| learning_rate | 0.001 | 0.001 |
| seed | "none (deterministic)" | "none (deterministic)" |
| weight_decay | 0.01 | 0.01 |

### layernorm / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.layernorm.synthetic.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 101.0 | 101.0..101.0 | 1 | - | - | - | 891.4 | - | max_rel_diff_vs_torch_eager_fp32=0.089719, rel_fro_vs_torch_eager_fp32=2.264e-07 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 99.8 | 99.8..99.8 | 1 | - | - | - | 887.2 | - | max_rel_diff_vs_torch_eager_fp32=0.089719, rel_fro_vs_torch_eager_fp32=2.264e-07 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 17.9 | 17.9..17.9 | 1 | 5.635 | 5.565 | - | 1367.0 | 1160.4 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 222.8 | 222.8..222.8 | 1 | 0.453 | 0.448 | - | 1367.5 | 1032.4 | max_rel_diff_vs_torch_eager_fp32=0.066421, rel_fro_vs_torch_eager_fp32=2.25e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 15.8 | 15.8..15.8 | 1 | 6.392 | 6.312 | - | 1365.2 | 1160.4 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 213.8 | 213.8..213.8 | 1 | 0.472 | 0.467 | - | 1358.6 | 1032.4 | max_rel_diff_vs_torch_eager_fp32=0.066421, rel_fro_vs_torch_eager_fp32=2.25e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

settings: {'bias': True, 'elementwise_affine': True, 'eps': 1e-05, 'normalized_shape': 1024}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| eps | 1e-05 | 1e-05 | 1e-05 | 1e-05 | 1e-05 | 1e-05 |
| seed | "none (deterministic)" | "none (deterministic)" | 7 | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 39.9 | 39.9..39.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 40.9 | 40.9..40.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | 1.5 | 1.5..1.5 | 1 | 26.615 | 27.309 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 203.7 | 203.7..203.7 | 1 | 0.196 | 0.201 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 1.1 | 1.1..1.1 | 1 | 34.820 | 35.728 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 203.8 | 203.8..203.8 | 1 | 0.196 | 0.201 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, ours-fast: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### lda-clf / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.lda-clf.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 32811.4 | 32811.4..32811.4 | 1 | - | - | - | 4379.0 | - | accuracy=0.911660, logloss=0.247483 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 29716.4 | 29716.4..29716.4 | 1 | - | - | - | 4376.8 | - | accuracy=0.909010, logloss=0.264392 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 5492.4 | 5492.4..5492.4 | 1 | 5.974 | 5.410 | - | 5271.9 | - | accuracy=0.901130, logloss=0.449237 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'solver': 'svd', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| n_components | null | null | null |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| solver | "svd" | "svd" | "svd" |
| tol | 0.0001 | 0.0001 | 0.0001 |

accepted difference: ours-fast n_components: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: sklearn-cpu n_components: None on both ours and sklearn-cpu: the same documented setting in both signatures

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 50.9 | 50.9..50.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 50.8 | 50.8..50.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 12.6 | 12.6..12.6 | 1 | 4.039 | 4.033 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### lda-clf / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.lda-clf.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 311.4 | 311.4..311.4 | 1 | - | - | - | 305.7 | - | accuracy=0.762580, logloss=0.539749 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 102.6 | 102.6..102.6 | 1 | - | - | - | 308.0 | - | accuracy=0.762580, logloss=0.539743 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 247.2 | 247.2..247.2 | 1 | 1.260 | 0.415 | - | 246.9 | - | accuracy=0.762530, logloss=0.539767 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'solver': 'svd', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| n_components | null | null | null |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| solver | "svd" | "svd" | "svd" |
| tol | 0.0001 | 0.0001 | 0.0001 |

accepted difference: ours-fast n_components: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: sklearn-cpu n_components: None on both ours and sklearn-cpu: the same documented setting in both signatures

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 6.4 | 6.4..6.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 6.2 | 6.2..6.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 1.7 | 1.7..1.7 | 1 | 3.856 | 3.724 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### lda / taxi-zones (rows full, shape X 129352x261; Xq 14373x261)

race: done, driver rc 0, log `logs/algos.lda.taxi-zones.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 54645.7 | 54645.7..54645.7 | 1 | - | - | - | 5967.0 | - | perplexity=45.221819 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 52512.6 | 52512.6..52512.6 | 1 | - | - | - | 5962.3 | - | perplexity=45.221811 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 37471.0 | 37471.0..37471.0 | 1 | 1.458 | 1.401 | - | 332.6 | - | perplexity=44.920445 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'batch_size': 128, 'learning_decay': 0.7, 'learning_method': 'batch', 'learning_offset': 10.0, 'max_doc_update_iter': 100, 'max_iter': 20, 'mean_change_tol': 0.001, 'n_components': 16, 'perp_tol': 0.1, 'random_state': 7, 'total_samples': 1000000.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| batch_size | 128 | 128 | 128 |
| max_iter | 20 | 20 | 20 |
| n_components | 16 | 16 | 16 |
| seed | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 413.3 | 413.3..413.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 397.2 | 397.2..397.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 471.0 | 471.0..471.0 | 1 | 0.877 | 0.843 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### lda / text (rows full, shape X 86626x4096; Xq 9626x4096; y 86626; yq 9626)

race: done, driver rc 0, log `logs/algos.lda.text.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(timeout: null) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(timeout: null) |
| sklearn-cpu | scikit-learn | cpu | opponent | 119418.7 | 119418.7..119418.7 | 1 | - | - | - | 1678.7 | - | perplexity=266.926844 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host not sampled; GPU not sampled

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'batch_size': 128, 'learning_decay': 0.7, 'learning_method': 'batch', 'learning_offset': 10.0, 'max_doc_update_iter': 100, 'max_iter': 20, 'mean_change_tol': 0.001, 'n_components': 16, 'perp_tol': 0.1, 'random_state': 7, 'total_samples': 1000000.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| batch_size | 128 | 128 | 128 |
| max_iter | 20 | 20 | 20 |
| n_components | 16 | 16 | 16 |
| seed | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(timeout: null) |
| mojolearn FAST | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(timeout: null) |
| sklearn-cpu | Xq | - | 735.4 | 735.4..735.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### lion / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.lion.synthetic.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 379.8 | 379.8..379.8 | 1 | - | - | - | 1514.4 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 380.2 | 380.2..380.2 | 1 | - | - | - | 1510.6 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

settings: {'betas': [0.9, 0.99], 'lr': 0.001, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast |
|---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) |
| betas | [0.9, 0.99] | [0.9, 0.99] |
| learning_rate | 0.001 | 0.001 |
| seed | "none (deterministic)" | "none (deterministic)" |
| weight_decay | 0.0 | 0.0 |

### lle / istella (rows full, shape X 10000x220; Xq 2000x220)

race: done, driver rc 0, log `logs/algos.lle.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(timeout: null) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(timeout: null) |
| sklearn-cpu | scikit-learn | cpu | opponent | 3064.1 | 3064.1..3064.1 | 1 | - | - | - | 251.3 | - | trustworthiness_k15=0.856249 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host not sampled; GPU not sampled

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'eigen_solver': 'auto', 'hessian_tol': 0.0001, 'max_iter': 100, 'method': 'standard', 'modified_tol': 1e-12, 'n_components': 2, 'n_neighbors': 10, 'neighbors_algorithm': 'auto', 'random_state': 7, 'reg': 0.001, 'tol': 1e-06}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| max_iter | - | - | 100 |
| n_components | 2 | 2 | 2 |
| n_neighbors | 10 | 10 | 10 |
| seed | 7 | 7 | 7 |
| tol | - | - | 1e-06 |

### lle / taxi (rows full, shape X 10000x11; Xq 2000x11)

race: done, driver rc 0, log `logs/algos.lle.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(timeout: null) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(timeout: null) |
| sklearn-cpu | scikit-learn | cpu | opponent | 1219.2 | 1219.2..1219.2 | 1 | - | - | - | 172.6 | - | trustworthiness_k15=0.770758 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host not sampled; GPU not sampled

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'eigen_solver': 'auto', 'hessian_tol': 0.0001, 'max_iter': 100, 'method': 'standard', 'modified_tol': 1e-12, 'n_components': 2, 'n_neighbors': 10, 'neighbors_algorithm': 'auto', 'random_state': 7, 'reg': 0.001, 'tol': 1e-06}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| max_iter | - | - | 100 |
| n_components | 2 | 2 | 2 |
| n_neighbors | 10 | 10 | 10 |
| seed | 7 | 7 | 7 |
| tol | - | - | 1e-06 |

### lof / istella (rows full, shape X 200000x220; Xq 100000x220; y 200000; yq 100000)

race: done, driver rc 0, log `logs/algos.lof.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 31275.4 | 31275.4..31275.4 | 1 | - | - | - | 1622.0 | - | fraction_flagged=0.010805, jaccard_vs_sklearn=0.054112 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 31177.7 | 31177.7..31177.7 | 1 | - | - | - | 1621.1 | - | fraction_flagged=0.010275, jaccard_vs_sklearn=0.054168 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 28696.8 | 28696.8..28696.8 | 1 | 1.090 | 1.086 | - | 1315.0 | - | fraction_flagged=0.033610, jaccard_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.lof.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 19747.6 | 19747.6..19747.6 | 1 | - | - | - | 264.9 | - | fraction_flagged=0.008960, jaccard_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 19513.1 | 19513.1..19513.1 | 1 | - | - | - | 267.4 | - | fraction_flagged=0.008960, jaccard_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 9500.7 | 9500.7..9500.7 | 1 | 2.079 | 2.054 | - | 279.1 | - | fraction_flagged=0.008960, jaccard_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

### louvain / istella (rows full, shape X 20000x220; indices 304830; indices2 62848; indptr 20001; indptr2 20001; y 20000)

race: done, driver rc 0, log `logs/algos.louvain.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 67826.8 | 67826.8..67826.8 | 1 | - | - | - | 23152.9 | - | modularity=0.909755, n_communities=39 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 46047.3 | 46047.3..46047.3 | 1 | - | - | - | 23155.2 | - | modularity=0.909755, n_communities=39 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| networkx-cpu | networkx | cpu | opponent | 1560.7 | 1560.7..1560.7 | 1 | 43.459 | 29.504 | - | 274.6 | - | modularity=0.908460, n_communities=40 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, networkx-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'resolution': 1.0, 'seed': 7}. Rows: None. Timed: None.

mismatch: networkx louvain_communities(seed=7) and cuGraph louvain (max_level=100) are order-dependent; ours pins the vertex sweep (lowest id first)

mismatch: ours takes a dense adjacency matrix (its class's contract), built from the CSR graph before the clock; networkx and cuGraph take the graph itself

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | networkx-cpu | ours | ours-fast |
|---|---||---|---||---|---|
| library (source) | networkx (declared) | mojolearn (declared) | mojolearn (declared) |
| seed | 7 | 7 | 7 |

### louvain / taxi (rows full, shape X 20000x11; indices 253708; indices2 54528; indptr 20001; indptr2 20001; y 20000)

race: done, driver rc 0, log `logs/algos.louvain.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 68352.6 | 68352.6..68352.6 | 1 | - | - | - | 23092.3 | - | modularity=0.941172, n_communities=58 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 46146.7 | 46146.7..46146.7 | 1 | - | - | - | 23090.3 | - | modularity=0.941172, n_communities=58 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| networkx-cpu | networkx | cpu | opponent | 919.5 | 919.5..919.5 | 1 | 74.339 | 50.188 | - | 190.9 | - | modularity=0.940781, n_communities=56 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, networkx-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'resolution': 1.0, 'seed': 7}. Rows: None. Timed: None.

mismatch: networkx louvain_communities(seed=7) and cuGraph louvain (max_level=100) are order-dependent; ours pins the vertex sweep (lowest id first)

mismatch: ours takes a dense adjacency matrix (its class's contract), built from the CSR graph before the clock; networkx and cuGraph take the graph itself

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | networkx-cpu | ours | ours-fast |
|---|---||---|---||---|---|
| library (source) | networkx (declared) | mojolearn (declared) | mojolearn (declared) |
| seed | 7 | 7 | 7 |

### lstm-clf / synthetic (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc 0, log `logs/algos.lstm-clf.synthetic.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3894.4 | 3894.4..3894.4 | 1 | - | - | - | 465.0 | - | accuracy=0.968696, logloss=0.072441 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 3546.5 | 3546.5..3546.5 | 1 | - | - | - | 469.3 | - | accuracy=0.968696, logloss=0.072441 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host not sampled; GPU not sampled

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (attributes) | mojolearn (attributes) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| batch_size | 256 | 256 | 256 | 256 | 256 | 256 |
| betas | - | - | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| eps | - | - | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| hidden_size | 64 | 64 | 64 | 64 | 64 | 64 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| num_layers | 1 | 1 | 1 | 1 | 1 | 1 |
| seed | 7 | 7 | 7 | 7 | 7 | 7 |
| shuffle | true | true | true | true | true | true |
| weight_decay | - | - | 0.0 | 0.0 | 0.0 | 0.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 319.2 | 319.2..319.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 319.5 | 319.5..319.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | - | - | 0 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) |
| torch-compile-fp32 | Xq | - | - | - | 0 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) |
| torch-eager-bf16 | Xq | - | - | - | 0 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) |
| torch-compile-bf16 | Xq | - | - | - | 0 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, torch-eager-fp32: predict(Xq)(Xq)

inference call, torch-compile-fp32: predict(Xq)(Xq)

inference call, torch-eager-bf16: predict(Xq)(Xq)

inference call, torch-compile-bf16: predict(Xq)(Xq)

### lstm-clf / taxi-hourly (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc 0, log `logs/algos.lstm-clf.taxi-hourly.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3893.4 | 3893.4..3893.4 | 1 | - | - | - | 467.5 | - | accuracy=0.868218, logloss=0.299901 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 3539.4 | 3539.4..3539.4 | 1 | - | - | - | 472.6 | - | accuracy=0.868218, logloss=0.299901 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host not sampled; GPU not sampled

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (attributes) | mojolearn (attributes) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| batch_size | 256 | 256 | 256 | 256 | 256 | 256 |
| betas | - | - | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| eps | - | - | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| hidden_size | 64 | 64 | 64 | 64 | 64 | 64 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| num_layers | 1 | 1 | 1 | 1 | 1 | 1 |
| seed | 7 | 7 | 7 | 7 | 7 | 7 |
| shuffle | true | true | true | true | true | true |
| weight_decay | - | - | 0.0 | 0.0 | 0.0 | 0.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 318.3 | 318.3..318.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 320.3 | 320.3..320.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | - | - | 0 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) |
| torch-compile-fp32 | Xq | - | - | - | 0 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) |
| torch-eager-bf16 | Xq | - | - | - | 0 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) |
| torch-compile-bf16 | Xq | - | - | - | 0 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, torch-eager-fp32: predict(Xq)(Xq)

inference call, torch-compile-fp32: predict(Xq)(Xq)

inference call, torch-eager-bf16: predict(Xq)(Xq)

inference call, torch-compile-bf16: predict(Xq)(Xq)

### lstm-reg / synthetic (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc 0, log `logs/algos.lstm-reg.synthetic.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3890.8 | 3890.8..3890.8 | 1 | - | - | - | 462.3 | - | finite=True, r2=0.981013, rmse=0.159641 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 3532.0 | 3532.0..3532.0 | 1 | - | - | - | 454.7 | - | finite=True, r2=0.981013, rmse=0.159641 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host not sampled; GPU not sampled

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (attributes) | mojolearn (attributes) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| batch_size | 256 | 256 | 256 | 256 | 256 | 256 |
| betas | - | - | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| eps | - | - | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| hidden_size | 64 | 64 | 64 | 64 | 64 | 64 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| num_layers | 1 | 1 | 1 | 1 | 1 | 1 |
| seed | 7 | 7 | 7 | 7 | 7 | 7 |
| shuffle | true | true | true | true | true | true |
| weight_decay | - | - | 0.0 | 0.0 | 0.0 | 0.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 166.6 | 166.6..166.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 158.3 | 158.3..158.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | - | - | 0 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) |
| torch-compile-fp32 | Xq | - | - | - | 0 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) |
| torch-eager-bf16 | Xq | - | - | - | 0 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) |
| torch-compile-bf16 | Xq | - | - | - | 0 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, torch-eager-fp32: predict(Xq)(Xq)

inference call, torch-compile-fp32: predict(Xq)(Xq)

inference call, torch-eager-bf16: predict(Xq)(Xq)

inference call, torch-compile-bf16: predict(Xq)(Xq)

### lstm-reg / taxi-hourly (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc 0, log `logs/algos.lstm-reg.taxi-hourly.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3888.7 | 3888.7..3888.7 | 1 | - | - | - | 460.0 | - | finite=True, r2=0.751679, rmse=0.540429 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 3547.0 | 3547.0..3547.0 | 1 | - | - | - | 462.4 | - | finite=True, r2=0.751679, rmse=0.540429 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host not sampled; GPU not sampled

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (attributes) | mojolearn (attributes) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| batch_size | 256 | 256 | 256 | 256 | 256 | 256 |
| betas | - | - | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| eps | - | - | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| hidden_size | 64 | 64 | 64 | 64 | 64 | 64 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| num_layers | 1 | 1 | 1 | 1 | 1 | 1 |
| seed | 7 | 7 | 7 | 7 | 7 | 7 |
| shuffle | true | true | true | true | true | true |
| weight_decay | - | - | 0.0 | 0.0 | 0.0 | 0.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 158.4 | 158.4..158.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 158.6 | 158.6..158.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | - | - | 0 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) |
| torch-compile-fp32 | Xq | - | - | - | 0 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) |
| torch-eager-bf16 | Xq | - | - | - | 0 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) |
| torch-compile-bf16 | Xq | - | - | - | 0 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, torch-eager-fp32: predict(Xq)(Xq)

inference call, torch-compile-fp32: predict(Xq)(Xq)

inference call, torch-eager-bf16: predict(Xq)(Xq)

inference call, torch-compile-bf16: predict(Xq)(Xq)

### lstsq / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.lstsq.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 49526.7 | 49526.7..49526.7 | 1 | - | - | - | 4074.1 | - | relative_residual=0.849957 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 48535.4 | 48535.4..48535.4 | 1 | - | - | - | 4070.0 | - | relative_residual=0.849956 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| numpy-cpu | numpy | cpu | opponent | 4837.0 | 4837.0..4837.0 | 1 | 10.239 | 10.034 | - | 4344.3 | - | relative_residual=0.873341 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-gpu | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "NotImplementedError(\"The operator 'aten::linalg_lstsq.out' is not currently implemented for the MPS device. If you want this op to be considered for addition please comment on https://gith) (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, numpy-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

memory, torch-gpu: host not sampled; GPU not sampled

settings: {}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | numpy-cpu | ours | ours-fast | torch-gpu |
|---|---||---|---||---|---||---|---|
| library (source) | numpy (declared) | mojolearn (declared) | mojolearn (declared) | torch (declared) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### lstsq / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.lstsq.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 125.2 | 125.2..125.2 | 1 | - | - | - | 361.7 | - | relative_residual=0.756366 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 117.2 | 117.2..117.2 | 1 | - | - | - | 362.1 | - | relative_residual=0.756366 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| numpy-cpu | numpy | cpu | opponent | 125.5 | 125.5..125.5 | 1 | 0.997 | 0.934 | - | 89.5 | - | relative_residual=0.756366 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-gpu | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "NotImplementedError(\"The operator 'aten::linalg_lstsq.out' is not currently implemented for the MPS device. If you want this op to be considered for addition please comment on https://gith) (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, numpy-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

memory, torch-gpu: host not sampled; GPU not sampled

settings: {}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | numpy-cpu | ours | ours-fast | torch-gpu |
|---|---||---|---||---|---||---|---|
| library (source) | numpy (declared) | mojolearn (declared) | mojolearn (declared) | torch (declared) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### lu-factor / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.lu-factor.synthetic.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 56541.2 | 56541.2..56541.2 | 1 | - | - | - | 1903.6 | - | relative_residual=3.249e-06 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 56273.8 | 56273.8..56273.8 | 1 | - | - | - | 2165.3 | - | relative_residual=3.249e-06 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| scipy-cpu | scipy | cpu | opponent | 892.8 | 892.8..892.8 | 1 | 63.327 | 63.027 | - | 331.9 | - | relative_residual=3.246e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-gpu | torch | gpu | opponent | 2941.0 | 2941.0..2941.0 | 1 | 19.225 | 19.134 | - | 1223.6 | 1040.4 | relative_residual=8.214e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, scipy-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

memory, torch-gpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

settings: {}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | scipy-cpu | torch-gpu |
|---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | scipy (declared) | torch (declared) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### lu-solve / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.lu-solve.synthetic.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 56598.2 | 56598.2..56598.2 | 1 | - | - | - | 2156.3 | - | relative_residual=3.249e-06 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 56296.3 | 56296.3..56296.3 | 1 | - | - | - | 1907.6 | - | relative_residual=3.249e-06 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| numpy-cpu | numpy | cpu | opponent | 2453.3 | 2453.3..2453.3 | 1 | 23.071 | 22.947 | - | 826.5 | - | relative_residual=3.259e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-gpu | torch | gpu | opponent | 479.2 | 479.2..479.2 | 1 | 118.114 | 117.484 | - | 1215.7 | 1032.4 | relative_residual=8.214e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, numpy-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

memory, torch-gpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

settings: {}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | numpy-cpu | ours | ours-fast | torch-gpu |
|---|---||---|---||---|---||---|---|
| library (source) | numpy (declared) | mojolearn (declared) | mojolearn (declared) | torch (declared) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### maxabs-scaler / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.maxabs-scaler.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 358.0 | 358.0..358.0 | 1 | - | - | - | 3005.7 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x220, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 233.3 | 233.3..233.3 | 1 | - | - | - | 3004.2 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x220, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 126.1 | 126.1..126.1 | 1 | 2.838 | 1.850 | - | 2161.1 | - | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), MaxAbsScaler (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 84.5 | 84.5..84.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 75.9 | 75.9..75.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 12.7 | 12.7..12.7 | 1 | 6.630 | 5.953 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### maxabs-scaler / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.maxabs-scaler.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 181.8 | 181.8..181.8 | 1 | - | - | - | 214.5 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x11, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 17.3 | 17.3..17.3 | 1 | - | - | - | 215.0 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x11, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 16.6 | 16.6..16.6 | 1 | 10.952 | 1.044 | - | 211.8 | - | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), MaxAbsScaler (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 6.1 | 6.1..6.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 3.2 | 3.2..3.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 1.4 | 1.4..1.4 | 1 | 4.413 | 2.316 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### mb-dict-learning / istella (rows full, shape X 100000x220; Xq 5000x220; y 100000; yq 5000)

race: done, driver rc 0, log `logs/algos.mb-dict-learning.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 45467.0 | 45467.0..45467.0 | 1 | - | - | - | 1698.3 | - | component_sparsity=0.086364, relative_reconstruction_error=0.648338 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 43457.3 | 43457.3..43457.3 | 1 | - | - | - | 1703.1 | - | component_sparsity=0.086364, relative_reconstruction_error=0.648339 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 5780.4 | 5780.4..5780.4 | 1 | 7.866 | 7.518 | - | 1283.6 | - | component_sparsity=0.086364, relative_reconstruction_error=0.644014 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'batch_size': 256, 'fit_algorithm': 'lars', 'max_iter': 10, 'max_no_improvement': 10, 'n_components': 16, 'positive_code': False, 'positive_dict': False, 'random_state': 7, 'shuffle': True, 'split_sign': False, 'tol': 0.001, 'transform_algorithm': 'lasso_cd', 'transform_max_iter': 1000}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

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

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 189.9 | 189.9..189.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 201.1 | 201.1..201.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 82.2 | 82.2..82.2 | 1 | 2.311 | 2.447 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### mb-dict-learning / taxi (rows full, shape X 100000x11; Xq 5000x11; y 100000; yq 5000)

race: done, driver rc 0, log `logs/algos.mb-dict-learning.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 8544.9 | 8544.9..8544.9 | 1 | - | - | - | 169.8 | - | component_sparsity=0.000000, relative_reconstruction_error=0.496685 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 8017.3 | 8017.3..8017.3 | 1 | - | - | - | 171.2 | - | component_sparsity=0.000000, relative_reconstruction_error=0.496685 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 4452.2 | 4452.2..4452.2 | 1 | 1.919 | 1.801 | - | 249.8 | - | component_sparsity=0.000000, relative_reconstruction_error=0.477049 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'batch_size': 256, 'fit_algorithm': 'lars', 'max_iter': 10, 'max_no_improvement': 10, 'n_components': 16, 'positive_code': False, 'positive_dict': False, 'random_state': 7, 'shuffle': True, 'split_sign': False, 'tol': 0.001, 'transform_algorithm': 'lasso_cd', 'transform_max_iter': 1000}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

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

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 58.6 | 58.6..58.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 101.3 | 101.3..101.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 17.8 | 17.8..17.8 | 1 | 3.290 | 5.690 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### mb-sparse-pca / istella (rows full, shape X 100000x220; Xq 20000x220; y 100000; yq 20000)

race: done, driver rc 0, log `logs/algos.mb-sparse-pca.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 24628.8 | 24628.8..24628.8 | 1 | - | - | - | 2387.1 | - | component_sparsity=0.130682, relative_reconstruction_error=0.705389 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 22222.4 | 22222.4..22222.4 | 1 | - | - | - | 2399.7 | - | component_sparsity=0.130682, relative_reconstruction_error=0.705389 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 2391.6 | 2391.6..2391.6 | 1 | 10.298 | 9.292 | - | 1457.3 | - | component_sparsity=0.130682, relative_reconstruction_error=0.705389 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'batch_size': 1024, 'max_iter': 10, 'max_no_improvement': 10, 'method': 'lars', 'n_components': 8, 'random_state': 7, 'ridge_alpha': 0.01, 'shuffle': True, 'tol': 0.001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

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

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 28.1 | 28.1..28.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 20.1 | 20.1..20.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 18.5 | 18.5..18.5 | 1 | 1.519 | 1.087 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### mb-sparse-pca / taxi (rows full, shape X 100000x11; Xq 20000x11; y 100000; yq 20000)

race: done, driver rc 0, log `logs/algos.mb-sparse-pca.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 500.3 | 500.3..500.3 | 1 | - | - | - | 213.0 | - | component_sparsity=0.022727, relative_reconstruction_error=0.275935 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 497.8 | 497.8..497.8 | 1 | - | - | - | 209.0 | - | component_sparsity=0.022727, relative_reconstruction_error=0.275935 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1377.6 | 1377.6..1377.6 | 1 | 0.363 | 0.361 | - | 251.7 | - | component_sparsity=0.022727, relative_reconstruction_error=0.275933 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'batch_size': 1024, 'max_iter': 10, 'max_no_improvement': 10, 'method': 'lars', 'n_components': 8, 'random_state': 7, 'ridge_alpha': 0.01, 'shuffle': True, 'tol': 0.001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

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

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 12.9 | 12.9..12.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 17.3 | 17.3..17.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 2.5 | 2.5..2.5 | 1 | 5.144 | 6.901 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### mds / istella (rows full, shape X 5000x220; Xq 2000x220)

race: done, driver rc 0, log `logs/algos.mds.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(timeout: null) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(timeout: null) |
| sklearn-cpu | scikit-learn | cpu | opponent | 3148.0 | 3148.0..3148.0 | 1 | - | - | - | 387.6 | - | trustworthiness_k15=0.580239 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host not sampled; GPU not sampled

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'eps': 0.001, 'max_iter': 300, 'n_components': 2, 'n_init': 1, 'normalized_stress': 'auto', 'random_state': 7}. Rows: None. Timed: None.

mismatch: metric MDS on Euclidean distances on both, spelled differently: ours metric='euclidean', metric_mds=True, init='random' (scikit-learn 1.9's names); the pinned scikit-learn 1.7.2 metric=True, dissimilarity='euclidean' and a random start from random_state; each draws its own start

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

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

accepted difference: sklearn-cpu metric: the same metric MDS on Euclidean distances, spelled differently: ours metric='euclidean' (scikit-learn 1.9's name), the pinned scikit-learn 1.7.2 metric=True with dissimilarity='euclidean'

### mds / taxi (rows full, shape X 5000x11; Xq 2000x11)

race: done, driver rc 0, log `logs/algos.mds.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(timeout: null) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(timeout: null) |
| sklearn-cpu | scikit-learn | cpu | opponent | 3113.4 | 3113.4..3113.4 | 1 | - | - | - | 348.4 | - | trustworthiness_k15=0.604142 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host not sampled; GPU not sampled

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'eps': 0.001, 'max_iter': 300, 'n_components': 2, 'n_init': 1, 'normalized_stress': 'auto', 'random_state': 7}. Rows: None. Timed: None.

mismatch: metric MDS on Euclidean distances on both, spelled differently: ours metric='euclidean', metric_mds=True, init='random' (scikit-learn 1.9's names); the pinned scikit-learn 1.7.2 metric=True, dissimilarity='euclidean' and a random start from random_state; each draws its own start

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

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

accepted difference: sklearn-cpu metric: the same metric MDS on Euclidean distances, spelled differently: ours metric='euclidean' (scikit-learn 1.9's name), the pinned scikit-learn 1.7.2 metric=True with dissimilarity='euclidean'

### min-cov-det / istella (rows full, shape X 100000x220; Xq 100000x220; y 100000; yq 100000)

race: done, driver rc 0, log `logs/algos.min-cov-det.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(timeout: null) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "Exception(\"eigh: the Jacobi eigensolver did not converge in 15 sweeps at n = 220. An unconverged decomposition is not returned as if it were one; see DEVIATION 590. The remedy is more swee) |
| sklearn-cpu | scikit-learn | cpu | opponent | 52780.3 | 52780.3..52780.3 | 1 | - | - | - | 4814.6 | - | n_features=220 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host not sampled; GPU not sampled

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | 7 | 7 | 7 |

### min-cov-det / taxi (rows full, shape X 100000x11; Xq 100000x11; y 100000; yq 100000)

race: done, driver rc 0, log `logs/algos.min-cov-det.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 4728.9 | 4728.9..4728.9 | 1 | - | - | - | 256.0 | - | n_features=11, rel_diff_vs_sklearn=0.205603 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 2225.6 | 2225.6..2225.6 | 1 | - | - | - | 257.5 | - | n_features=11, rel_diff_vs_sklearn=0.205603 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1533.5 | 1533.5..1533.5 | 1 | 3.084 | 1.451 | - | 286.2 | - | n_features=11 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | 7 | 7 | 7 |

### minmax-scaler / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.minmax-scaler.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 599.3 | 599.3..599.3 | 1 | - | - | - | 3944.4 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x220, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 595.9 | 595.9..595.9 | 1 | - | - | - | 3944.2 | - | max_abs_diff_vs_sklearn=1.192e-07, output_shape=100000x220, rel_diff_vs_sklearn=1.694e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 83.8 | 83.8..83.8 | 1 | 7.150 | 7.110 | - | 1325.8 | - | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'clip': False, 'feature_range': [0, 1]}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 95.6 | 95.6..95.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 95.3 | 95.3..95.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 16.3 | 16.3..16.3 | 1 | 5.866 | 5.853 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### minmax-scaler / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.minmax-scaler.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 34.8 | 34.8..34.8 | 1 | - | - | - | 260.6 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x11, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 49.2 | 49.2..49.2 | 1 | - | - | - | 266.3 | - | max_abs_diff_vs_sklearn=5.96e-08, output_shape=100000x11, rel_diff_vs_sklearn=2.63e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 21.0 | 21.0..21.0 | 1 | 1.657 | 2.342 | - | 207.2 | - | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'clip': False, 'feature_range': [0, 1]}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 4.5 | 4.5..4.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 4.4 | 4.4..4.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 2.0 | 2.0..2.0 | 1 | 2.222 | 2.170 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### mlp-clf / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.mlp-clf.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 33958.8 | 33958.8..33958.8 | 1 | - | - | - | 2757.5 | - | accuracy=0.944270, logloss=0.136955 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 32336.5 | 32336.5..32336.5 | 1 | - | - | - | 2767.0 | - | accuracy=0.944410, logloss=0.136065 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 17913.9 | 17913.9..17913.9 | 1 | 1.896 | 1.805 | - | 1097.1 | - | accuracy=0.943760, logloss=0.136831 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'activation': 'relu', 'alpha': 0.0001, 'batch_size': 4096, 'beta_1': 0.9, 'beta_2': 0.999, 'early_stopping': False, 'epsilon': 1e-08, 'hidden_layer_sizes': [256, 256], 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'max_fun': 15000, 'max_iter': 5, 'momentum': 0.9, 'n_iter_no_change': 1000, 'nesterovs_momentum': True, 'power_t': 0.5, 'random_state': 7, 'shuffle': True, 'solver': 'adam', 'tol': 0.0, 'validation_fraction': 0.1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (attributes) | mojolearn (attributes) | sklearn (get_params) |
| alpha | 0.0001 | 0.0001 | 0.0001 |
| batch_size | 4096 | 4096 | 4096 |
| epsilon | 1e-08 | 1e-08 | 1e-08 |
| learning_rate | "constant" | "constant" | "constant" |
| max_iter | 5 | 5 | 5 |
| momentum | 0.9 | 0.9 | 0.9 |
| seed | 7 | 7 | 7 |
| shuffle | true | true | true |
| solver | "adam" | "adam" | "adam" |
| tol | 0.0 | 0.0 | 0.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 482.1 | 482.1..482.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 452.1 | 452.1..452.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 115.2 | 115.2..115.2 | 1 | 4.183 | 3.923 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### mlp-clf / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.mlp-clf.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 24994.7 | 24994.7..24994.7 | 1 | - | - | - | 274.5 | - | accuracy=0.767810, logloss=0.530506 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 23853.4 | 23853.4..23853.4 | 1 | - | - | - | 288.1 | - | accuracy=0.767780, logloss=0.530482 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 14471.3 | 14471.3..14471.3 | 1 | 1.727 | 1.648 | - | 227.3 | - | accuracy=0.767830, logloss=0.530444 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'activation': 'relu', 'alpha': 0.0001, 'batch_size': 4096, 'beta_1': 0.9, 'beta_2': 0.999, 'early_stopping': False, 'epsilon': 1e-08, 'hidden_layer_sizes': [256, 256], 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'max_fun': 15000, 'max_iter': 5, 'momentum': 0.9, 'n_iter_no_change': 1000, 'nesterovs_momentum': True, 'power_t': 0.5, 'random_state': 7, 'shuffle': True, 'solver': 'adam', 'tol': 0.0, 'validation_fraction': 0.1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (attributes) | mojolearn (attributes) | sklearn (get_params) |
| alpha | 0.0001 | 0.0001 | 0.0001 |
| batch_size | 4096 | 4096 | 4096 |
| epsilon | 1e-08 | 1e-08 | 1e-08 |
| learning_rate | "constant" | "constant" | "constant" |
| max_iter | 5 | 5 | 5 |
| momentum | 0.9 | 0.9 | 0.9 |
| seed | 7 | 7 | 7 |
| shuffle | true | true | true |
| solver | "adam" | "adam" | "adam" |
| tol | 0.0 | 0.0 | 0.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 261.7 | 261.7..261.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 246.4 | 246.4..246.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 107.0 | 107.0..107.0 | 1 | 2.446 | 2.303 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### mlp-reg / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.mlp-reg.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 33911.4 | 33911.4..33911.4 | 1 | - | - | - | 2764.8 | - | finite=True, r2=0.526405, rmse=0.574862 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 32310.6 | 32310.6..32310.6 | 1 | - | - | - | 2766.5 | - | finite=True, r2=0.527265, rmse=0.574340 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 17106.0 | 17106.0..17106.0 | 1 | 1.982 | 1.889 | - | 1078.1 | - | finite=True, r2=0.524815, rmse=0.575826 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'activation': 'relu', 'alpha': 0.0001, 'batch_size': 4096, 'beta_1': 0.9, 'beta_2': 0.999, 'early_stopping': False, 'epsilon': 1e-08, 'hidden_layer_sizes': [256, 256], 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'max_fun': 15000, 'max_iter': 5, 'momentum': 0.9, 'n_iter_no_change': 1000, 'nesterovs_momentum': True, 'power_t': 0.5, 'random_state': 7, 'shuffle': True, 'solver': 'adam', 'tol': 0.0, 'validation_fraction': 0.1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (attributes) | mojolearn (attributes) | sklearn (get_params) |
| alpha | 0.0001 | 0.0001 | 0.0001 |
| batch_size | 4096 | 4096 | 4096 |
| epsilon | 1e-08 | 1e-08 | 1e-08 |
| learning_rate | "constant" | "constant" | "constant" |
| loss | - | - | "squared_error" |
| max_iter | 5 | 5 | 5 |
| momentum | 0.9 | 0.9 | 0.9 |
| seed | 7 | 7 | 7 |
| shuffle | true | true | true |
| solver | "adam" | "adam" | "adam" |
| tol | 0.0 | 0.0 | 0.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 248.3 | 248.3..248.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 229.1 | 229.1..229.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 63.7 | 63.7..63.7 | 1 | 3.898 | 3.597 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### mlp-reg / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.mlp-reg.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 24948.8 | 24948.8..24948.8 | 1 | - | - | - | 283.2 | - | finite=True, r2=0.931981, rmse=4.153868 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 23837.5 | 23837.5..23837.5 | 1 | - | - | - | 284.6 | - | finite=True, r2=0.931976, rmse=4.154000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 12450.5 | 12450.5..12450.5 | 1 | 2.004 | 1.915 | - | 201.3 | - | finite=True, r2=0.929613, rmse=4.225537 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'activation': 'relu', 'alpha': 0.0001, 'batch_size': 4096, 'beta_1': 0.9, 'beta_2': 0.999, 'early_stopping': False, 'epsilon': 1e-08, 'hidden_layer_sizes': [256, 256], 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'max_fun': 15000, 'max_iter': 5, 'momentum': 0.9, 'n_iter_no_change': 1000, 'nesterovs_momentum': True, 'power_t': 0.5, 'random_state': 7, 'shuffle': True, 'solver': 'adam', 'tol': 0.0, 'validation_fraction': 0.1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (attributes) | mojolearn (attributes) | sklearn (get_params) |
| alpha | 0.0001 | 0.0001 | 0.0001 |
| batch_size | 4096 | 4096 | 4096 |
| epsilon | 1e-08 | 1e-08 | 1e-08 |
| learning_rate | "constant" | "constant" | "constant" |
| loss | - | - | "squared_error" |
| max_iter | 5 | 5 | 5 |
| momentum | 0.9 | 0.9 | 0.9 |
| seed | 7 | 7 | 7 |
| shuffle | true | true | true |
| solver | "adam" | "adam" | "adam" |
| tol | 0.0 | 0.0 | 0.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 130.9 | 130.9..130.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 124.0 | 124.0..124.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 58.9 | 58.9..58.9 | 1 | 2.222 | 2.105 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### moe / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.moe.synthetic.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 8614.9 | 8614.9..8614.9 | 1 | - | - | - | 1228.7 | - | max_rel_diff_vs_torch_eager_fp32=0.070736, rel_fro_vs_torch_eager_fp32=3.02e-07 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 8621.2 | 8621.2..8621.2 | 1 | - | - | - | 1225.0 | - | max_rel_diff_vs_torch_eager_fp32=0.084128, rel_fro_vs_torch_eager_fp32=3.334e-07 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 86.4 | 86.4..86.4 | 1 | 99.724 | 99.797 | - | 1256.5 | 1050.6 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 84.2 | 84.2..84.2 | 1 | 102.319 | 102.394 | - | 1433.9 | 1050.6 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 126.1 | 126.1..126.1 | 1 | 68.324 | 68.374 | - | 1250.4 | 1050.6 | max_rel_diff_vs_torch_eager_fp32=22208.605483, rel_fro_vs_torch_eager_fp32=0.055222 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 124.3 | 124.3..124.3 | 1 | 69.312 | 69.363 | - | 1431.6 | 1050.6 | max_rel_diff_vs_torch_eager_fp32=22208.605483, rel_fro_vs_torch_eager_fp32=0.055222 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

settings: {'hidden_size': 1024, 'intermediate_size': 2816, 'norm_topk_prob': True, 'num_experts': 8, 'top_k': 2}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| hidden_size | 1024 | 1024 | 1024 | 1024 | 1024 | 1024 |
| seed | 7 | 7 | 7 | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 8628.6 | 8628.6..8628.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 8658.2 | 8658.2..8658.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | 73.5 | 73.5..73.5 | 1 | 117.398 | 117.801 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 70.6 | 70.6..70.6 | 1 | 122.209 | 122.628 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 113.1 | 113.1..113.1 | 1 | 76.267 | 76.529 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 110.7 | 110.7..110.7 | 1 | 77.921 | 78.189 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, ours-fast: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### multilabel-binarizer / istella (rows full, shape X 200000x8; Xq 20000x8; lab 200000; labq 20000; y 200000; yq 20000)

race: done, driver rc 0, log `logs/algos.multilabel-binarizer.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1014.8 | 1014.8..1014.8 | 1 | - | - | - | 935.9 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=20000x119, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 937.6 | 937.6..937.6 | 1 | - | - | - | 906.7 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=20000x119, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 182.2 | 182.2..182.2 | 1 | 5.568 | 5.145 | - | 782.8 | - | output_shape=20000x119 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 41.5 | 41.5..41.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 40.0 | 40.0..40.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 20.3 | 20.3..20.3 | 1 | 2.046 | 1.974 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### multilabel-binarizer / taxi (rows full, shape X 200000x5; Xq 20000x5; lab 200000; labq 20000; y 200000; yq 20000)

race: done, driver rc 0, log `logs/algos.multilabel-binarizer.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 871.8 | 871.8..871.8 | 1 | - | - | - | 2373.0 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=20000x489, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 852.3 | 852.3..852.3 | 1 | - | - | - | 2366.8 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=20000x489, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 162.5 | 162.5..162.5 | 1 | 5.365 | 5.245 | - | 2739.7 | - | output_shape=20000x489 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 49.8 | 49.8..49.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 49.4 | 49.4..49.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 16.9 | 16.9..16.9 | 1 | 2.941 | 2.918 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### multinomial-nb / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.multinomial-nb.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 379.5 | 379.5..379.5 | 1 | - | - | - | 3616.3 | - | accuracy=0.853560, logloss=3.630944 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 249.1 | 249.1..249.1 | 1 | - | - | - | 3617.6 | - | accuracy=0.853620, logloss=3.628572 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 270.9 | 270.9..270.9 | 1 | 1.401 | 0.920 | - | 3708.3 | - | accuracy=0.853620, logloss=3.087499 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'fit_prior': True, 'force_alpha': True}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), MultinomialNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 1.0 | 1.0 | 1.0 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 52.9 | 52.9..52.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 36.5 | 36.5..36.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 60.9 | 60.9..60.9 | 1 | 0.868 | 0.600 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### multinomial-nb / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.multinomial-nb.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 214.3 | 214.3..214.3 | 1 | - | - | - | 267.9 | - | accuracy=0.723260, logloss=0.589979 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 25.8 | 25.8..25.8 | 1 | - | - | - | 268.6 | - | accuracy=0.723160, logloss=0.590725 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 53.0 | 53.0..53.0 | 1 | 4.040 | 0.487 | - | 296.7 | - | accuracy=0.723160, logloss=0.590725 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'fit_prior': True, 'force_alpha': True}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), MultinomialNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 1.0 | 1.0 | 1.0 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 5.5 | 5.5..5.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 7.2 | 7.2..7.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 12.2 | 12.2..12.2 | 1 | 0.450 | 0.590 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### multinomial-nb / text (rows full, shape X 86626x4096; Xq 9626x4096; y 86626; yq 9626)

race: done, driver rc 0, log `logs/algos.multinomial-nb.text.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 287.5 | 287.5..287.5 | 1 | - | - | - | 4283.0 | - | accuracy=0.983067, logloss=0.559529 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 481.1 | 481.1..481.1 | 1 | - | - | - | 4281.9 | - | accuracy=0.983067, logloss=0.559529 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 494.4 | 494.4..494.4 | 1 | 0.581 | 0.973 | - | 4361.0 | - | accuracy=0.983067, logloss=0.557319 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'fit_prior': True, 'force_alpha': True}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), MultinomialNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 1.0 | 1.0 | 1.0 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 74.1 | 74.1..74.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 96.0 | 96.0..96.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 94.6 | 94.6..94.6 | 1 | 0.783 | 1.014 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### nadam / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.nadam.synthetic.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 527.0 | 527.0..527.0 | 1 | - | - | - | 1645.3 | - | rel_fro_vs_torch_eager_fp32=2.939e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 513.7 | 513.7..513.7 | 1 | - | - | - | 1644.6 | - | rel_fro_vs_torch_eager_fp32=2.939e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 93.2 | 93.2..93.2 | 1 | 5.652 | 5.509 | - | 1407.0 | 1032.4 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 218.9 | 218.9..218.9 | 1 | 2.407 | 2.346 | - | 1493.7 | 1024.4 | rel_fro_vs_torch_eager_fp32=4.102e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-eager-fp32, torch-compile-fp32: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

settings: {'betas': [0.9, 0.999], 'decoupled_weight_decay': False, 'eps': 1e-08, 'lr': 0.001, 'momentum_decay': 0.004, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | torch-compile-fp32 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | torch (optimizer.defaults) | torch (optimizer.defaults) |
| betas | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| eps | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| weight_decay | 0.0 | 0.0 | 0.0 | 0.0 |

### nearest-centroid / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.nearest-centroid.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 6428.8 | 6428.8..6428.8 | 1 | - | - | - | 1934.6 | - | accuracy=0.852610, logloss=4.299221 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 2727.2 | 2727.2..2727.2 | 1 | - | - | - | 1934.6 | - | accuracy=0.852610, logloss=4.299222 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 709.0 | 709.0..709.0 | 1 | 9.068 | 3.847 | - | 4438.2 | - | accuracy=0.852610, logloss=4.117692 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 63.9 | 63.9..63.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 62.2 | 62.2..62.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 186.2 | 186.2..186.2 | 1 | 0.343 | 0.334 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### nearest-centroid / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.nearest-centroid.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 537.9 | 537.9..537.9 | 1 | - | - | - | 170.8 | - | accuracy=0.666750, logloss=0.782162 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 434.9 | 434.9..434.9 | 1 | - | - | - | 171.6 | - | accuracy=0.666750, logloss=0.782162 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 128.5 | 128.5..128.5 | 1 | 4.185 | 3.383 | - | 226.3 | - | accuracy=0.666750, logloss=0.781690 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 32.6 | 32.6..32.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 32.6 | 32.6..32.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 12.8 | 12.8..12.8 | 1 | 2.546 | 2.546 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### nmf / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.nmf.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 57845.6 | 57845.6..57845.6 | 1 | - | - | - | 5260.3 | - | relative_reconstruction_error=0.325174 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 56368.6 | 56368.6..56368.6 | 1 | - | - | - | 5262.3 | - | relative_reconstruction_error=0.325174 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 11062.4 | 11062.4..11062.4 | 1 | 5.229 | 5.096 | - | 3713.9 | - | relative_reconstruction_error=0.325399 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha_H': 'same', 'alpha_W': 0.0, 'beta_loss': 'frobenius', 'init': 'nndsvda', 'l1_ratio': 0.0, 'max_iter': 200, 'n_components': 8, 'random_state': 7, 'shuffle': False, 'solver': 'mu', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

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

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 412.1 | 412.1..412.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 398.1 | 398.1..398.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 178.7 | 178.7..178.7 | 1 | 2.306 | 2.228 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### nmf / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.nmf.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 2143.4 | 2143.4..2143.4 | 1 | - | - | - | 688.2 | - | relative_reconstruction_error=0.091156 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 2095.1 | 2095.1..2095.1 | 1 | - | - | - | 686.8 | - | relative_reconstruction_error=0.091156 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 3943.7 | 3943.7..3943.7 | 1 | 0.543 | 0.531 | - | 344.8 | - | relative_reconstruction_error=0.091155 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha_H': 'same', 'alpha_W': 0.0, 'beta_loss': 'frobenius', 'init': 'nndsvda', 'l1_ratio': 0.0, 'max_iter': 200, 'n_components': 8, 'random_state': 7, 'shuffle': False, 'solver': 'mu', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

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

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 69.5 | 69.5..69.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 67.8 | 67.8..67.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 72.5 | 72.5..72.5 | 1 | 0.958 | 0.935 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### normalizer / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.normalizer.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 0.1 | 0.1..0.1 | 1 | - | - | - | 1663.8 | - | max_abs_diff_vs_sklearn=3.576e-07, output_shape=100000x220, rel_diff_vs_sklearn=5.811e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 0.1 | 0.1..0.1 | 1 | - | - | - | 1661.5 | - | max_abs_diff_vs_sklearn=3.576e-07, output_shape=100000x220, rel_diff_vs_sklearn=5.674e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 35.5 | 35.5..35.5 | 1 | 0.004 | 0.004 | - | 1321.8 | - | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'norm': 'l2'}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), Normalizer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 73.3 | 73.3..73.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 72.9 | 72.9..72.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 18.8 | 18.8..18.8 | 1 | 3.887 | 3.866 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### normalizer / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.normalizer.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 0.1 | 0.1..0.1 | 1 | - | - | - | 152.0 | - | max_abs_diff_vs_sklearn=1.192e-07, output_shape=100000x11, rel_diff_vs_sklearn=3.114e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 0.1 | 0.1..0.1 | 1 | - | - | - | 152.5 | - | max_abs_diff_vs_sklearn=1.192e-07, output_shape=100000x11, rel_diff_vs_sklearn=3.283e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 2.1 | 2.1..2.1 | 1 | 0.057 | 0.044 | - | 209.6 | - | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'norm': 'l2'}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), Normalizer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 3.6 | 3.6..3.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 3.4 | 3.4..3.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 1.9 | 1.9..1.9 | 1 | 1.936 | 1.814 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### ocsvm / istella (rows full, shape X 10000x220; Xq 10000x220; y 10000; yq 10000)

race: done, driver rc 0, log `logs/algos.ocsvm.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1611.9 | 1611.9..1611.9 | 1 | - | - | - | 1855.4 | - | fraction_flagged=0.078300, jaccard_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1608.1 | 1608.1..1608.1 | 1 | - | - | - | 1859.7 | - | fraction_flagged=0.078300, jaccard_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 874.3 | 874.3..874.3 | 1 | 1.844 | 1.839 | - | 1140.5 | - | fraction_flagged=0.078300, jaccard_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 144.0 | 144.0..144.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 144.0 | 144.0..144.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 893.0 | 893.0..893.0 | 1 | 0.161 | 0.161 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### ocsvm / taxi (rows full, shape X 10000x11; Xq 10000x11; y 10000; yq 10000)

race: done, driver rc 0, log `logs/algos.ocsvm.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 337.4 | 337.4..337.4 | 1 | - | - | - | 944.9 | - | fraction_flagged=0.136100, jaccard_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 342.8 | 342.8..342.8 | 1 | - | - | - | 888.7 | - | fraction_flagged=0.136100, jaccard_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 240.0 | 240.0..240.0 | 1 | 1.406 | 1.428 | - | 227.4 | - | fraction_flagged=0.136100, jaccard_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 14.4 | 14.4..14.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 14.6 | 14.6..14.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 266.4 | 266.4..266.4 | 1 | 0.054 | 0.055 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### onehot / istella (rows full, shape X 1000000x8; Xq 100000x8; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.onehot.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 299.0 | 299.0..299.0 | 1 | - | - | - | 450.7 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x119, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 110.3 | 110.3..110.3 | 1 | - | - | - | 468.6 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x119, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 46.5 | 46.5..46.5 | 1 | 6.436 | 2.373 | - | 468.4 | - | output_shape=100000x119 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'categories': 'auto', 'handle_unknown': 'ignore', 'sparse_output': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), OneHotEncoder (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 32.4 | 32.4..32.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 32.4 | 32.4..32.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 30.9 | 30.9..30.9 | 1 | 1.051 | 1.052 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### onehot / taxi (rows full, shape X 1000000x5; Xq 100000x5; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.onehot.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 211.8 | 211.8..211.8 | 1 | - | - | - | 1090.9 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x508, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 114.2 | 114.2..114.2 | 1 | - | - | - | 1078.2 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x508, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 24.1 | 24.1..24.1 | 1 | 8.797 | 4.743 | - | 1341.1 | - | output_shape=100000x508 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'categories': 'auto', 'handle_unknown': 'ignore', 'sparse_output': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), OneHotEncoder (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 113.9 | 113.9..113.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 113.5 | 113.5..113.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 33.9 | 33.9..33.9 | 1 | 3.363 | 3.352 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### optics / istella (rows full, shape X 10000x220; Xq 100000x220; y 10000; yq 100000)

race: done, driver rc 0, log `logs/algos.optics.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1851.0 | 1851.0..1851.0 | 1 | - | - | - | 1784.8 | - | n_clusters=20, silhouette=-0.287356 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1919.5 | 1919.5..1919.5 | 1 | - | - | - | 1787.8 | - | ari_vs_ours=1.000000, n_clusters=20, silhouette=-0.287356 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 17384.3 | 17384.3..17384.3 | 1 | 0.106 | 0.110 | - | 1132.2 | - | ari_vs_ours=0.984603, n_clusters=20, silhouette=-0.285806 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| sklearn-cpu | Xq | - | 0.0 | 0.0..0.0 | 1 | 0.445 | 0.664 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq) or fit labels(Xq)

inference call, ours-fast: predict(Xq) or fit labels(Xq)

inference call, sklearn-cpu: predict(Xq) or fit labels(Xq)

### optics / taxi (rows full, shape X 10000x11; Xq 100000x11; y 10000; yq 100000)

race: done, driver rc 0, log `logs/algos.optics.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 547.1 | 547.1..547.1 | 1 | - | - | - | 880.8 | - | n_clusters=127, silhouette=-0.353359 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 667.6 | 667.6..667.6 | 1 | - | - | - | 885.4 | - | ari_vs_ours=1.000000, n_clusters=127, silhouette=-0.353359 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(timeout: null) (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host not sampled; GPU not sampled

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
| sklearn-cpu | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(timeout: null) |

inference call, ours: predict(Xq) or fit labels(Xq)

inference call, ours-fast: predict(Xq) or fit labels(Xq)

inference call, sklearn-cpu: predict(Xq) or fit labels(Xq)

### optimized-theta / synthetic (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc 0, log `logs/algos.optimized-theta.synthetic.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 501.7 | 501.7..501.7 | 1 | - | - | - | 67.8 | - | forecast_rmse=1.438855 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 466.3 | 466.3..466.3 | 1 | - | - | - | 67.4 | - | forecast_rmse=1.439709 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsforecast-cpu | statsforecast | cpu | opponent | 1667.8 | 1667.8..1667.8 | 1 | 0.301 | 0.280 | - | 184.7 | - | forecast_rmse=1.437815 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, statsforecast-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'decomposition_type': 'multiplicative', 'season_length': 24}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | statsforecast-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (attributes) | mojolearn (attributes) | statsforecast (declared) |
| alpha | null | null | - |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

accepted difference: ours-fast alpha: None on both ours and ours-fast: the same documented setting in both signatures

### optimized-theta / taxi-hourly (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc 0, log `logs/algos.optimized-theta.taxi-hourly.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 867.7 | 867.7..867.7 | 1 | - | - | - | 68.5 | - | forecast_rmse=49.150860 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 837.5 | 837.5..837.5 | 1 | - | - | - | 69.6 | - | forecast_rmse=49.152331 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsforecast-cpu | statsforecast | cpu | opponent | 2738.3 | 2738.3..2738.3 | 1 | 0.317 | 0.306 | - | 184.5 | - | forecast_rmse=49.356608 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, statsforecast-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'decomposition_type': 'multiplicative', 'season_length': 24}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | statsforecast-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (attributes) | mojolearn (attributes) | statsforecast (declared) |
| alpha | null | null | - |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

accepted difference: ours-fast alpha: None on both ours and ours-fast: the same documented setting in both signatures

### ordinal / istella (rows full, shape X 1000000x8; Xq 100000x8; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.ordinal.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 283.8 | 283.8..283.8 | 1 | - | - | - | 314.4 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x8, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 110.6 | 110.6..110.6 | 1 | - | - | - | 335.5 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x8, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 46.3 | 46.3..46.3 | 1 | 6.125 | 2.387 | - | 198.4 | - | output_shape=100000x8 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'categories': 'auto', 'handle_unknown': 'use_encoded_value', 'unknown_value': -1}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), OrdinalEncoder (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 7.6 | 7.6..7.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 7.6 | 7.6..7.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 21.2 | 21.2..21.2 | 1 | 0.358 | 0.359 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### ordinal / taxi (rows full, shape X 1000000x5; Xq 100000x5; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.ordinal.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 199.7 | 199.7..199.7 | 1 | - | - | - | 231.5 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x5, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 96.8 | 96.8..96.8 | 1 | - | - | - | 237.9 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x5, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 23.6 | 23.6..23.6 | 1 | 8.459 | 4.103 | - | 183.9 | - | output_shape=100000x5 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'categories': 'auto', 'handle_unknown': 'use_encoded_value', 'unknown_value': -1}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), OrdinalEncoder (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 7.0 | 7.0..7.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 7.3 | 7.3..7.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 12.2 | 12.2..12.2 | 1 | 0.575 | 0.599 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### pagerank / istella (rows full, shape X 20000x220; indices 304830; indices2 62848; indptr 20001; indptr2 20001; y 20000)

race: done, driver rc 0, log `logs/algos.pagerank.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1108.7 | 1108.7..1108.7 | 1 | - | - | - | 7780.6 | - | l1_vs_networkx=5.89e-08, sum=1.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1105.7 | 1105.7..1105.7 | 1 | - | - | - | 7683.9 | - | l1_vs_networkx=6.429e-08, sum=1.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| networkx-cpu | networkx | cpu | opponent | 97.4 | 97.4..97.4 | 1 | 11.385 | 11.355 | - | 200.0 | - | sum=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, networkx-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 0.85, 'max_iter': 100, 'tol': 1e-06}. Rows: None. Timed: None.

mismatch: ours takes a dense adjacency matrix (its class's contract), built from the CSR graph before the clock; networkx and cuGraph take the graph itself

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | networkx-cpu | ours | ours-fast |
|---|---||---|---||---|---|
| library (source) | networkx (declared) | mojolearn (declared) | mojolearn (declared) |
| alpha | 0.85 | 0.85 | 0.85 |
| max_iter | 100 | 100 | 100 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| tol | 1e-06 | 1e-06 | 1e-06 |

### pagerank / taxi (rows full, shape X 20000x11; indices 253708; indices2 54528; indptr 20001; indptr2 20001; y 20000)

race: done, driver rc 0, log `logs/algos.pagerank.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1103.1 | 1103.1..1103.1 | 1 | - | - | - | 7764.1 | - | l1_vs_networkx=6.917e-08, sum=1.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1148.4 | 1148.4..1148.4 | 1 | - | - | - | 7498.4 | - | l1_vs_networkx=8.411e-08, sum=1.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| networkx-cpu | networkx | cpu | opponent | 77.5 | 77.5..77.5 | 1 | 14.237 | 14.823 | - | 145.3 | - | sum=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, networkx-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 0.85, 'max_iter': 100, 'tol': 1e-06}. Rows: None. Timed: None.

mismatch: ours takes a dense adjacency matrix (its class's contract), built from the CSR graph before the clock; networkx and cuGraph take the graph itself

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | networkx-cpu | ours | ours-fast |
|---|---||---|---||---|---|
| library (source) | networkx (declared) | mojolearn (declared) | mojolearn (declared) |
| alpha | 0.85 | 0.85 | 0.85 |
| max_iter | 100 | 100 | 100 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| tol | 1e-06 | 1e-06 | 1e-06 |

### permutation-test / istella (rows full, shape X 20000x220; Xq 20000x220; y 20000; yq 20000)

race: done, driver rc 0, log `logs/algos.permutation-test.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested identical, read back unknown) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested fast, read back unknown) |
| scipy-cpu | scipy | cpu | opponent | 3957.1 | 3957.1..3957.1 | 1 | - | - | - | 1221.7 | - | pvalue=0.184400, statistic=0.011200 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host not sampled; GPU not sampled

memory, scipy-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alternative': 'two-sided', 'n_resamples': 9999, 'permutation_type': 'independent', 'random_state': 7, 'statistic': 'diff_means'}. Rows: None. Timed: None.

mismatch: each library draws its permutations from its own generator seeded 7, so the p-values agree to Monte Carlo error, not bit for bit

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | scipy-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | scipy (declared) |
| seed | 7 | 7 | 7 |

### permutation-test / taxi (rows full, shape X 20000x11; Xq 20000x11; y 20000; yq 20000)

race: done, driver rc 0, log `logs/algos.permutation-test.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested identical, read back unknown) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested fast, read back unknown) |
| scipy-cpu | scipy | cpu | opponent | 3932.7 | 3932.7..3932.7 | 1 | - | - | - | 313.1 | - | pvalue=0.001000, statistic=-0.576708 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host not sampled; GPU not sampled

memory, scipy-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alternative': 'two-sided', 'n_resamples': 9999, 'permutation_type': 'independent', 'random_state': 7, 'statistic': 'diff_means'}. Rows: None. Timed: None.

mismatch: each library draws its permutations from its own generator seeded 7, so the p-values agree to Monte Carlo error, not bit for bit

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | scipy-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | scipy (declared) |
| seed | 7 | 7 | 7 |

### pls-canonical / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.pls-canonical.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 4392.5 | 4392.5..4392.5 | 1 | - | - | - | 4290.9 | - | mean_canonical_corr=0.875340 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 4401.6 | 4401.6..4401.6 | 1 | - | - | - | 4264.2 | - | mean_canonical_corr=0.875340 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 5405.3 | 5405.3..5405.3 | 1 | 0.813 | 0.814 | - | 3654.4 | - | mean_canonical_corr=0.875341 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'algorithm': 'nipals', 'max_iter': 500, 'n_components': 2, 'scale': True, 'tol': 1e-06}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| algorithm | "nipals" | "nipals" | "nipals" |
| max_iter | 500 | 500 | 500 |
| n_components | 2 | 2 | 2 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| tol | 1e-06 | 1e-06 | 1e-06 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 78.3 | 78.3..78.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 79.5 | 79.5..79.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 74.5 | 74.5..74.5 | 1 | 1.052 | 1.067 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### pls-canonical / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.pls-canonical.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 239.6 | 239.6..239.6 | 1 | - | - | - | 446.4 | - | mean_canonical_corr=0.559206 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 235.7 | 235.7..235.7 | 1 | - | - | - | 445.1 | - | mean_canonical_corr=0.559206 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 352.6 | 352.6..352.6 | 1 | 0.680 | 0.668 | - | 278.8 | - | mean_canonical_corr=0.559206 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'algorithm': 'nipals', 'max_iter': 500, 'n_components': 2, 'scale': True, 'tol': 1e-06}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| algorithm | "nipals" | "nipals" | "nipals" |
| max_iter | 500 | 500 | 500 |
| n_components | 2 | 2 | 2 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| tol | 1e-06 | 1e-06 | 1e-06 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 11.0 | 11.0..11.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 10.5 | 10.5..10.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 8.5 | 8.5..8.5 | 1 | 1.294 | 1.235 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### pls / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.pls.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1268.7 | 1268.7..1268.7 | 1 | - | - | - | 5225.8 | - | finite=True, r2=0.289870, rmse=0.703930 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1265.6 | 1265.6..1265.6 | 1 | - | - | - | 5226.3 | - | finite=True, r2=0.289870, rmse=0.703930 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 2738.0 | 2738.0..2738.0 | 1 | 0.463 | 0.462 | - | 4563.0 | - | finite=True, r2=0.289870, rmse=0.703930 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'max_iter': 500, 'n_components': 4, 'scale': True, 'tol': 1e-06}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| max_iter | 500 | 500 | 500 |
| n_components | 4 | 4 | 4 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| tol | 1e-06 | 1e-06 | 1e-06 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 41.9 | 41.9..41.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 41.4 | 41.4..41.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 49.3 | 49.3..49.3 | 1 | 0.851 | 0.841 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### pls / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.pls.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 244.0 | 244.0..244.0 | 1 | - | - | - | 578.0 | - | finite=True, r2=0.905216, rmse=4.903469 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 239.9 | 239.9..239.9 | 1 | - | - | - | 580.6 | - | finite=True, r2=0.905216, rmse=4.903469 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 330.2 | 330.2..330.2 | 1 | 0.739 | 0.727 | - | 333.4 | - | finite=True, r2=0.905216, rmse=4.903467 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'max_iter': 500, 'n_components': 4, 'scale': True, 'tol': 1e-06}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| max_iter | 500 | 500 | 500 |
| n_components | 4 | 4 | 4 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| tol | 1e-06 | 1e-06 | 1e-06 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 4.3 | 4.3..4.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 4.2 | 4.2..4.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 2.8 | 2.8..2.8 | 1 | 1.531 | 1.495 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### poly-count-sketch / istella (rows full, shape X 100000x220; Xq 1000x220; y 100000; yq 1000)

race: done, driver rc 0, log `logs/algos.poly-count-sketch.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 0.4 | 0.4..0.4 | 1 | - | - | - | 1088.9 | - | kernel_rel_error=0.040849 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 0.4 | 0.4..0.4 | 1 | - | - | - | 1091.7 | - | kernel_rel_error=0.040849 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 4.2 | 4.2..4.2 | 1 | 0.105 | 0.100 | - | 1167.0 | - | kernel_rel_error=0.040849 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 3.7 | 3.7..3.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 3.6 | 3.6..3.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 6.2 | 6.2..6.2 | 1 | 0.590 | 0.583 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### poly-count-sketch / taxi (rows full, shape X 100000x11; Xq 1000x11; y 100000; yq 1000)

race: done, driver rc 0, log `logs/algos.poly-count-sketch.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 0.3 | 0.3..0.3 | 1 | - | - | - | 126.4 | - | kernel_rel_error=0.096596 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 0.3 | 0.3..0.3 | 1 | - | - | - | 128.5 | - | kernel_rel_error=0.096596 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 0.8 | 0.8..0.8 | 1 | 0.432 | 0.394 | - | 220.8 | - | kernel_rel_error=0.096596 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 3.2 | 3.2..3.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 3.2 | 3.2..3.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 2.9 | 2.9..2.9 | 1 | 1.106 | 1.078 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### poly-features / istella (rows full, shape X 1000000x16; Xq 100000x16; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.poly-features.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 0.5 | 0.5..0.5 | 1 | - | - | - | 1367.1 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x152, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 0.3 | 0.3..0.3 | 1 | - | - | - | 1375.8 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x152, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 3.0 | 3.0..3.0 | 1 | 0.178 | 0.106 | - | 1323.8 | - | output_shape=100000x152 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'degree': 2, 'include_bias': False, 'interaction_only': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), PolynomialFeatures (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| degree | 2 | 2 | 2 |
| order | "C" | "C" | "C" |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 49.9 | 49.9..49.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 50.7 | 50.7..50.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 43.0 | 43.0..43.0 | 1 | 1.161 | 1.179 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### poly-features / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.poly-features.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 0.3 | 0.3..0.3 | 1 | - | - | - | 302.1 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x77, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 0.2 | 0.2..0.2 | 1 | - | - | - | 305.3 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x77, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 2.2 | 2.2..2.2 | 1 | 0.115 | 0.111 | - | 312.3 | - | output_shape=100000x77 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'degree': 2, 'include_bias': False, 'interaction_only': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), PolynomialFeatures (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| degree | 2 | 2 | 2 |
| order | "C" | "C" | "C" |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 27.2 | 27.2..27.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 27.2 | 27.2..27.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 22.5 | 22.5..22.5 | 1 | 1.212 | 1.209 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### power-transformer / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.power-transformer.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 8805.9 | 8805.9..8805.9 | 1 | - | - | - | 6386.0 | - | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 8293.0 | 8293.0..8293.0 | 1 | - | - | - | 8060.8 | - | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(timeout: null) (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host not sampled; GPU not sampled

settings: {'method': 'yeo-johnson', 'standardize': True}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), PowerTransformer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 94.2 | 94.2..94.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 79.7 | 79.7..79.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(timeout: null) |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### power-transformer / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.power-transformer.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 4219.4 | 4219.4..4219.4 | 1 | - | - | - | 659.9 | - | max_abs_diff_vs_sklearn=0.191760, output_shape=100000x11, rel_diff_vs_sklearn=0.025551 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 820.8 | 820.8..820.8 | 1 | - | - | - | 498.4 | - | max_abs_diff_vs_sklearn=0.062769, output_shape=100000x11, rel_diff_vs_sklearn=0.0005406 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 18588.5 | 18588.5..18588.5 | 1 | 0.227 | 0.044 | - | 232.8 | - | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'method': 'yeo-johnson', 'standardize': True}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), PowerTransformer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 5.8 | 5.8..5.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 6.0 | 6.0..6.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 13.1 | 13.1..13.1 | 1 | 0.442 | 0.458 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### prophet / synthetic (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc 0, log `logs/algos.prophet.synthetic.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 22296.0 | 22296.0..22296.0 | 1 | - | - | - | 66.9 | - | forecast_rmse=1.015150 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 22545.6 | 22545.6..22545.6 | 1 | - | - | - | 68.1 | - | forecast_rmse=1.015060 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| prophet-cpu | prophet | cpu | opponent | 941.7 | 941.7..941.7 | 1 | 23.676 | 23.941 | - | 119.8 | - | forecast_rmse=1.015319 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, prophet-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'changepoint_prior_scale': 0.05, 'changepoint_range': 0.8, 'daily_seasonality': True, 'growth': 'linear', 'holidays_prior_scale': 10.0, 'max_iter': 10000, 'n_changepoints': 25, 'seasonality_mode': 'additive', 'seasonality_prior_scale': 10.0, 'weekly_seasonality': True, 'yearly_seasonality': False}. Rows: None. Timed: None.

mismatch: prophet fits by Stan's L-BFGS (MAP); ours by its own L-BFGS; parity is at a tolerance, the forecast RMSE is the comparable number

mismatch: prophet uncertainty_samples=0: ours computes no intervals, so prophet's 1,000 sampled intervals (prophet only, numpy's unseeded RNG) are switched off

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | prophet-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (attributes) | mojolearn (attributes) | prophet (declared) |
| max_iter | 10000 | 10000 | 10000 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### prophet / taxi-hourly (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc 0, log `logs/algos.prophet.taxi-hourly.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 26657.3 | 26657.3..26657.3 | 1 | - | - | - | 65.0 | - | forecast_rmse=32.049281 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 23991.7 | 23991.7..23991.7 | 1 | - | - | - | 66.6 | - | forecast_rmse=32.028614 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| prophet-cpu | prophet | cpu | opponent | 839.4 | 839.4..839.4 | 1 | 31.757 | 28.581 | - | 119.7 | - | forecast_rmse=32.035281 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, prophet-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'changepoint_prior_scale': 0.05, 'changepoint_range': 0.8, 'daily_seasonality': True, 'growth': 'linear', 'holidays_prior_scale': 10.0, 'max_iter': 10000, 'n_changepoints': 25, 'seasonality_mode': 'additive', 'seasonality_prior_scale': 10.0, 'weekly_seasonality': True, 'yearly_seasonality': False}. Rows: None. Timed: None.

mismatch: prophet fits by Stan's L-BFGS (MAP); ours by its own L-BFGS; parity is at a tolerance, the forecast RMSE is the comparable number

mismatch: prophet uncertainty_samples=0: ours computes no intervals, so prophet's 1,000 sampled intervals (prophet only, numpy's unseeded RNG) are switched off

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | prophet-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (attributes) | mojolearn (attributes) | prophet (declared) |
| max_iter | 10000 | 10000 | 10000 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### qda / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.qda.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 22768.3 | 22768.3..22768.3 | 1 | - | - | - | 2695.1 | - | accuracy=0.885080, logloss=3.969196 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 20951.7 | 20951.7..20951.7 | 1 | - | - | - | 2695.7 | - | accuracy=0.866090, logloss=4.052899 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 8895.9 | 8895.9..8895.9 | 1 | 2.559 | 2.355 | - | 8525.3 | - | accuracy=0.880530, logloss=3.476976 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'reg_param': 0.001, 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| solver | "svd" | "svd" | - |
| tol | 0.0001 | 0.0001 | 0.0001 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 737.3 | 737.3..737.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 733.9 | 733.9..733.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 138.7 | 138.7..138.7 | 1 | 5.317 | 5.293 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### qda / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.qda.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 277.8 | 277.8..277.8 | 1 | - | - | - | 216.0 | - | accuracy=0.727020, logloss=1.061003 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 233.0 | 233.0..233.0 | 1 | - | - | - | 218.5 | - | accuracy=0.727020, logloss=1.061003 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 203.5 | 203.5..203.5 | 1 | 1.365 | 1.145 | - | 253.9 | - | accuracy=0.727220, logloss=1.059265 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'reg_param': 0.001, 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| solver | "svd" | "svd" | - |
| tol | 0.0001 | 0.0001 | 0.0001 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 8.1 | 8.1..8.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 7.2 | 7.2..7.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 15.6 | 15.6..15.6 | 1 | 0.518 | 0.460 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### qr / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.qr.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 143641.6 | 143641.6..143641.6 | 1 | - | - | - | 6888.3 | - | relative_gram_difference=0.037513 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 143062.7 | 143062.7..143062.7 | 1 | - | - | - | 6371.8 | - | relative_gram_difference=0.602233 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| numpy-cpu | numpy | cpu | opponent | 10610.0 | 10610.0..10610.0 | 1 | 13.538 | 13.484 | - | 8526.8 | - | relative_gram_difference=2.472e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-gpu | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "RuntimeError('Invalid buffer size: 3725.29 GiB')", "event": "error", "stage": "round 0"}) (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, numpy-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

memory, torch-gpu: host not sampled; GPU not sampled

settings: {'mode': 'reduced'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | numpy-cpu | ours | ours-fast | torch-gpu |
|---|---||---|---||---|---||---|---|
| library (source) | numpy (declared) | mojolearn (declared) | mojolearn (declared) | torch (declared) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### qr / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.qr.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3723.8 | 3723.8..3723.8 | 1 | - | - | - | 422.2 | - | relative_gram_difference=0.001996 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 3726.7 | 3726.7..3726.7 | 1 | - | - | - | 422.9 | - | relative_gram_difference=0.001996 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| numpy-cpu | numpy | cpu | opponent | 197.3 | 197.3..197.3 | 1 | 18.872 | 18.887 | - | 173.5 | - | relative_gram_difference=3.024e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-gpu | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "RuntimeError('Invalid buffer size: 3725.29 GiB')", "event": "error", "stage": "round 0"}) (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, numpy-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

memory, torch-gpu: host not sampled; GPU not sampled

settings: {'mode': 'reduced'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | numpy-cpu | ours | ours-fast | torch-gpu |
|---|---||---|---||---|---||---|---|
| library (source) | numpy (declared) | mojolearn (declared) | mojolearn (declared) | torch (declared) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### quantile-transformer / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.quantile-transformer.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 6393.5 | 6393.5..6393.5 | 1 | - | - | - | 4753.4 | - | max_abs_diff_vs_sklearn=5.96e-08, output_shape=100000x220, rel_diff_vs_sklearn=2.764e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1206.0 | 1206.0..1206.0 | 1 | - | - | - | 5657.2 | - | max_abs_diff_vs_sklearn=5.96e-08, output_shape=100000x220, rel_diff_vs_sklearn=2.764e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 5674.6 | 5674.6..5674.6 | 1 | 1.127 | 0.213 | - | 1330.5 | - | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'ignore_implicit_zeros': False, 'n_quantiles': 1000, 'output_distribution': 'uniform', 'random_state': 7, 'subsample': None}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), QuantileTransformer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | 7 | 7 | 7 |
| subsample | null | null | null |

accepted difference: ours-fast subsample: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: sklearn-cpu subsample: None on both ours and sklearn-cpu: the same documented setting in both signatures

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 152.1 | 152.1..152.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 164.3 | 164.3..164.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 964.4 | 964.4..964.4 | 1 | 0.158 | 0.170 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### quantile-transformer / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.quantile-transformer.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 487.7 | 487.7..487.7 | 1 | - | - | - | 322.0 | - | max_abs_diff_vs_sklearn=5.96e-08, output_shape=100000x11, rel_diff_vs_sklearn=2.294e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 68.7 | 68.7..68.7 | 1 | - | - | - | 355.7 | - | max_abs_diff_vs_sklearn=5.96e-08, output_shape=100000x11, rel_diff_vs_sklearn=2.295e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 208.0 | 208.0..208.0 | 1 | 2.345 | 0.330 | - | 208.3 | - | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'ignore_implicit_zeros': False, 'n_quantiles': 1000, 'output_distribution': 'uniform', 'random_state': 7, 'subsample': None}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), QuantileTransformer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | 7 | 7 | 7 |
| subsample | null | null | null |

accepted difference: ours-fast subsample: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: sklearn-cpu subsample: None on both ours and sklearn-cpu: the same documented setting in both signatures

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 8.3 | 8.3..8.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 8.8 | 8.8..8.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 43.3 | 43.3..43.3 | 1 | 0.191 | 0.202 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### radius-neighbors / istella (rows full, shape X 200000x220; Xq 20000x220; y 200000; yq 20000)

race: done, driver rc 0, log `logs/algos.radius-neighbors.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "RuntimeError('mojolearn RadiusNeighbors: the counting pass found 1220718 neighbours and the filling pass found 11938, on the same arrays. The two passes rebuild the index independently, so ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "Exception('radius_neighbors_fill: the search found 1220718 edges and the caller allocated for 5384. The two calls saw different data. Re-run radius_neighbors_count against the arrays this c) |
| sklearn-cpu | scikit-learn | cpu | opponent | 8.2 | 8.2..8.2 | 1 | - | - | - | 1315.3 | - | neighbors_total=1220718 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host not sampled; GPU not sampled

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
| mojolearn IDENTICAL | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(error: {"error": "RuntimeError('mojolearn RadiusNeighbors: the counting pass found 1220718 neighbours and the filling pass found 11938, on the same arrays. The two passes rebuild the index independently, so ) |
| mojolearn FAST | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(error: {"error": "Exception('radius_neighbors_fill: the search found 1220718 edges and the caller allocated for 5384. The two calls saw different data. Re-run radius_neighbors_count against the arrays this c) |
| sklearn-cpu | Xq | - | 2736.3 | 2736.3..2736.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: radius_neighbors(Xq)(Xq)

inference call, ours-fast: radius_neighbors(Xq)(Xq)

inference call, sklearn-cpu: radius_neighbors(Xq)(Xq)

### radius-neighbors / taxi (rows full, shape X 200000x11; Xq 20000x11; y 200000; yq 20000)

race: done, driver rc 0, log `logs/algos.radius-neighbors.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 0.2 | 0.2..0.2 | 1 | - | - | - | 167.7 | - | count_agreement_vs_sklearn=1.000000, neighbors_total=31 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 0.2 | 0.2..0.2 | 1 | - | - | - | 166.6 | - | count_agreement_vs_sklearn=1.000000, neighbors_total=31 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 55.1 | 55.1..55.1 | 1 | 0.003 | 0.003 | - | 258.5 | - | neighbors_total=31 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 308.7 | 308.7..308.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 302.1 | 302.1..302.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 101.5 | 101.5..101.5 | 1 | 3.040 | 2.975 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: radius_neighbors(Xq)(Xq)

inference call, ours-fast: radius_neighbors(Xq)(Xq)

inference call, sklearn-cpu: radius_neighbors(Xq)(Xq)

### randomized-svd / istella (rows full, shape X 900000x220; Xq 100000x220)

race: done, driver rc 0, log `logs/algos.randomized-svd.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 2987.0 | 2987.0..2987.0 | 1 | - | - | - | 3921.3 | - | relative_reconstruction_error=0.0002359 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 2927.3 | 2927.3..2927.3 | 1 | - | - | - | 3922.4 | - | relative_reconstruction_error=0.0002359 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 623.1 | 623.1..623.1 | 1 | 4.793 | 4.698 | - | 1031.8 | - | relative_reconstruction_error=0.0002359 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-gpu | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "RuntimeError('Invalid buffer size: 3017.49 GiB')", "event": "error", "stage": "round 0"}) (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

memory, torch-gpu: host not sampled; GPU not sampled

settings: {'n_components': 8, 'n_iter': 4, 'n_oversamples': 10, 'random_state': 7}. Rows: None. Timed: None.

mismatch: torch-gpu is torch.svd_lowrank(q=18, niter=4), its randomized range finder

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu | torch-gpu |
|---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | sklearn (declared) | torch (declared) |
| n_components | 8 | 8 | 8 | 8 |
| n_iter | 4 | 4 | 4 | 4 |
| seed | 7 | 7 | 7 | 7 |

### randomized-svd / taxi (rows full, shape X 900000x11; Xq 100000x11)

race: done, driver rc 0, log `logs/algos.randomized-svd.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 696.4 | 696.4..696.4 | 1 | - | - | - | 572.8 | - | relative_reconstruction_error=0.027197 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 698.2 | 698.2..698.2 | 1 | - | - | - | 574.5 | - | relative_reconstruction_error=0.027197 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 250.1 | 250.1..250.1 | 1 | 2.785 | 2.792 | - | 234.5 | - | relative_reconstruction_error=0.027197 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-gpu | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "RuntimeError('Invalid buffer size: 3017.49 GiB')", "event": "error", "stage": "round 0"}) (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

memory, torch-gpu: host not sampled; GPU not sampled

settings: {'n_components': 8, 'n_iter': 4, 'n_oversamples': 10, 'random_state': 7}. Rows: None. Timed: None.

mismatch: torch-gpu is torch.svd_lowrank(q=18, niter=4), its randomized range finder

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu | torch-gpu |
|---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | sklearn (declared) | torch (declared) |
| n_components | 8 | 8 | 8 | 8 |
| n_iter | 4 | 4 | 4 | 4 |
| seed | 7 | 7 | 7 | 7 |

### resample / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.resample.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 450.5 | 450.5..450.5 | 1 | - | - | - | 4360.1 | - | max_mean_shift_over_std=0.003203 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested identical, read back unknown) |
| mojolearn FAST | mojolearn | gpu | fast | 455.5 | 455.5..455.5 | 1 | - | - | - | 4361.6 | - | max_mean_shift_over_std=0.003203 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested fast, read back unknown) |
| sklearn-cpu | scikit-learn | cpu | opponent | 395.5 | 395.5..395.5 | 1 | - | - | - | 3591.1 | - | max_mean_shift_over_std=0.002552 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'n_samples': None, 'random_state': 7, 'replace': True}. Rows: None. Timed: None.

mismatch: the drawn rows are each library's own (ours the Philox position map, scikit-learn numpy RandomState(7)); quality is the resampled column means against the population's

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | sklearn (declared) |
| seed | 7 | 7 | 7 |

### resample / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.resample.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 73.4 | 73.4..73.4 | 1 | - | - | - | 129.3 | - | max_mean_shift_over_std=0.002917 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested identical, read back unknown) |
| mojolearn FAST | mojolearn | gpu | fast | 73.2 | 73.2..73.2 | 1 | - | - | - | 132.6 | - | max_mean_shift_over_std=0.002917 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested fast, read back unknown) |
| sklearn-cpu | scikit-learn | cpu | opponent | 64.0 | 64.0..64.0 | 1 | - | - | - | 195.2 | - | max_mean_shift_over_std=0.002257 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'n_samples': None, 'random_state': 7, 'replace': True}. Rows: None. Timed: None.

mismatch: the drawn rows are each library's own (ours the Philox position map, scikit-learn numpy RandomState(7)); quality is the resampled column means against the population's

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | sklearn (declared) |
| seed | 7 | 7 | 7 |

### rfe / istella (rows full, shape X 200000x220; Xq 100000x220; y 200000; yq 100000)

race: done, driver rc 0, log `logs/algos.rfe.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 7418.8 | 7418.8..7418.8 | 1 | - | - | - | 2005.3 | - | jaccard_vs_sklearn=0.818182, n_selected=110 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 6144.5 | 6144.5..6144.5 | 1 | - | - | - | 2011.3 | - | jaccard_vs_sklearn=0.833333, n_selected=110 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 27817.5 | 27817.5..27817.5 | 1 | 0.267 | 0.221 | - | 1243.8 | - | jaccard_vs_sklearn=1.000000, n_selected=110 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'estimator': 'LogisticRegression(max_iter=200)', 'n_features_to_select': 'half', 'step': 0.1}. Rows: None. Timed: None.

mismatch: nested LogisticRegression(max_iter=200): ours is L-BFGS (cuML's quasi-Newton, memory 5, solver='qn'), scikit-learn solver='lbfgs' (scipy, memory 10); C=1.0 and tol=1e-4 are both libraries' defaults; each stops by its own criterion

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### rfe / taxi (rows full, shape X 200000x11; Xq 100000x11; y 200000; yq 100000)

race: done, driver rc 0, log `logs/algos.rfe.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 186.0 | 186.0..186.0 | 1 | - | - | - | 170.7 | - | jaccard_vs_sklearn=1.000000, n_selected=5 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 175.0 | 175.0..175.0 | 1 | - | - | - | 174.0 | - | jaccard_vs_sklearn=1.000000, n_selected=5 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 395.6 | 395.6..395.6 | 1 | 0.470 | 0.442 | - | 235.2 | - | jaccard_vs_sklearn=1.000000, n_selected=5 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'estimator': 'LogisticRegression(max_iter=200)', 'n_features_to_select': 'half', 'step': 0.1}. Rows: None. Timed: None.

mismatch: nested LogisticRegression(max_iter=200): ours is L-BFGS (cuML's quasi-Newton, memory 5, solver='qn'), scikit-learn solver='lbfgs' (scipy, memory 10); C=1.0 and tol=1e-4 are both libraries' defaults; each stops by its own criterion

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### ridge-cv / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: failed, driver rc 1, log `logs/algos.ridge-cv.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "ValueError('mojolearn RidgeCV: only cv=None, scoring=None, alpha_per_target=False are implemented')", "event": "error", "stage": "round 0"}) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "ValueError('mojolearn RidgeCV: only cv=None, scoring=None, alpha_per_target=False are implemented')", "event": "error", "stage": "round 0"}) |
| sklearn-cpu | scikit-learn | cpu | opponent | 191217.8 | 191217.8..191217.8 | 1 | - | - | - | 8630.0 | - | finite=True, r2=0.328683, rmse=0.684423 | - | LIKE-FOR-LIKE-SPAN | - | ok (stored (measured 2026-09-30T12:06:20Z on ip-172-31-45-205.ec2.internal, cpu (Apple M2 Pro))) |

memory, ours, ours-fast: host not sampled; GPU not sampled

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha_per_target': False, 'alphas': [0.001, 0.01, 0.1, 1.0, 10.0], 'cv': 5, 'fit_intercept': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast |
|---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) |
| cv | 5 | 5 |
| fit_intercept | true | true |
| seed | "none (deterministic)" | "none (deterministic)" |

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

race: failed, driver rc 1, log `logs/algos.ridge-cv.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "ValueError('mojolearn RidgeCV: only cv=None, scoring=None, alpha_per_target=False are implemented')", "event": "error", "stage": "round 0"}) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "ValueError('mojolearn RidgeCV: only cv=None, scoring=None, alpha_per_target=False are implemented')", "event": "error", "stage": "round 0"}) |
| sklearn-cpu | scikit-learn | cpu | opponent | 1258.8 | 1258.8..1258.8 | 1 | - | - | - | 258.1 | - | finite=True, r2=0.908988, rmse=4.804917 | - | LIKE-FOR-LIKE-SPAN | - | ok (stored (measured 2026-09-30T11:59:52Z on ip-172-31-45-205.ec2.internal, cpu (Apple M2 Pro))) |

memory, ours, ours-fast: host not sampled; GPU not sampled

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha_per_target': False, 'alphas': [0.001, 0.01, 0.1, 1.0, 10.0], 'cv': 5, 'fit_intercept': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast |
|---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) |
| cv | 5 | 5 |
| fit_intercept | true | true |
| seed | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(error: {"error": "ValueError('mojolearn RidgeCV: only cv=None, scoring=None, alpha_per_target=False are implemented')", "event": "error", "stage": "round 0"}) |
| mojolearn FAST | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(error: {"error": "ValueError('mojolearn RidgeCV: only cv=None, scoring=None, alpha_per_target=False are implemented')", "event": "error", "stage": "round 0"}) |
| sklearn-cpu | Xq | - | 0.6 | 0.6..0.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### rmsprop / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.rmsprop.synthetic.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 401.1 | 401.1..401.1 | 1 | - | - | - | 1511.5 | - | rel_fro_vs_torch_eager_fp32=3.772e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 384.3 | 384.3..384.3 | 1 | - | - | - | 1529.6 | - | rel_fro_vs_torch_eager_fp32=3.772e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 60.9 | 60.9..60.9 | 1 | 6.582 | 6.307 | - | 1407.6 | 1032.4 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 164.8 | 164.8..164.8 | 1 | 2.434 | 2.333 | - | 1488.9 | 1024.4 | rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-eager-fp32, torch-compile-fp32: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

settings: {'alpha': 0.99, 'centered': False, 'eps': 1e-08, 'lr': 0.001, 'momentum': 0.0, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | torch-compile-fp32 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | torch (optimizer.defaults) | torch (optimizer.defaults) |
| alpha | 0.99 | 0.99 | 0.99 | 0.99 |
| eps | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 |
| momentum | 0.0 | 0.0 | 0.0 | 0.0 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| weight_decay | 0.0 | 0.0 | 0.0 | 0.0 |

### rnn-clf / synthetic (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc 0, log `logs/algos.rnn-clf.synthetic.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1822.0 | 1822.0..1822.0 | 1 | - | - | - | 246.0 | - | accuracy=0.953559, logloss=0.103698 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1642.5 | 1642.5..1642.5 | 1 | - | - | - | 250.3 | - | accuracy=0.953559, logloss=0.103698 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host not sampled; GPU not sampled

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'nonlinearity': 'tanh', 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (attributes) | mojolearn (attributes) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| batch_size | 256 | 256 | 256 | 256 | 256 | 256 |
| betas | - | - | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| eps | - | - | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| hidden_size | 64 | 64 | 64 | 64 | 64 | 64 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| num_layers | 1 | 1 | 1 | 1 | 1 | 1 |
| seed | 7 | 7 | 7 | 7 | 7 | 7 |
| shuffle | true | true | true | true | true | true |
| weight_decay | - | - | 0.0 | 0.0 | 0.0 | 0.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 99.4 | 99.4..99.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 93.3 | 93.3..93.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | - | - | 0 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) |
| torch-compile-fp32 | Xq | - | - | - | 0 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) |
| torch-eager-bf16 | Xq | - | - | - | 0 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) |
| torch-compile-bf16 | Xq | - | - | - | 0 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, torch-eager-fp32: predict(Xq)(Xq)

inference call, torch-compile-fp32: predict(Xq)(Xq)

inference call, torch-eager-bf16: predict(Xq)(Xq)

inference call, torch-compile-bf16: predict(Xq)(Xq)

### rnn-clf / taxi-hourly (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc 0, log `logs/algos.rnn-clf.taxi-hourly.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1823.8 | 1823.8..1823.8 | 1 | - | - | - | 244.7 | - | accuracy=0.868056, logloss=0.304864 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1646.7 | 1646.7..1646.7 | 1 | - | - | - | 254.9 | - | accuracy=0.868056, logloss=0.304864 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host not sampled; GPU not sampled

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'nonlinearity': 'tanh', 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (attributes) | mojolearn (attributes) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| batch_size | 256 | 256 | 256 | 256 | 256 | 256 |
| betas | - | - | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| eps | - | - | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| hidden_size | 64 | 64 | 64 | 64 | 64 | 64 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| num_layers | 1 | 1 | 1 | 1 | 1 | 1 |
| seed | 7 | 7 | 7 | 7 | 7 | 7 |
| shuffle | true | true | true | true | true | true |
| weight_decay | - | - | 0.0 | 0.0 | 0.0 | 0.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 99.0 | 99.0..99.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 93.9 | 93.9..93.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | - | - | 0 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) |
| torch-compile-fp32 | Xq | - | - | - | 0 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) |
| torch-eager-bf16 | Xq | - | - | - | 0 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) |
| torch-compile-bf16 | Xq | - | - | - | 0 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, torch-eager-fp32: predict(Xq)(Xq)

inference call, torch-compile-fp32: predict(Xq)(Xq)

inference call, torch-eager-bf16: predict(Xq)(Xq)

inference call, torch-compile-bf16: predict(Xq)(Xq)

### rnn-reg / synthetic (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc 0, log `logs/algos.rnn-reg.synthetic.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1816.1 | 1816.1..1816.1 | 1 | - | - | - | 249.4 | - | finite=True, r2=0.977348, rmse=0.174374 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1675.1 | 1675.1..1675.1 | 1 | - | - | - | 234.6 | - | finite=True, r2=0.977348, rmse=0.174374 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host not sampled; GPU not sampled

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'nonlinearity': 'tanh', 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (attributes) | mojolearn (attributes) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| batch_size | 256 | 256 | 256 | 256 | 256 | 256 |
| betas | - | - | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| eps | - | - | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| hidden_size | 64 | 64 | 64 | 64 | 64 | 64 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| num_layers | 1 | 1 | 1 | 1 | 1 | 1 |
| seed | 7 | 7 | 7 | 7 | 7 | 7 |
| shuffle | true | true | true | true | true | true |
| weight_decay | - | - | 0.0 | 0.0 | 0.0 | 0.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 49.3 | 49.3..49.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 46.9 | 46.9..46.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | - | - | 0 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) |
| torch-compile-fp32 | Xq | - | - | - | 0 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) |
| torch-eager-bf16 | Xq | - | - | - | 0 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) |
| torch-compile-bf16 | Xq | - | - | - | 0 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, torch-eager-fp32: predict(Xq)(Xq)

inference call, torch-compile-fp32: predict(Xq)(Xq)

inference call, torch-eager-bf16: predict(Xq)(Xq)

inference call, torch-compile-bf16: predict(Xq)(Xq)

### rnn-reg / taxi-hourly (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc 0, log `logs/algos.rnn-reg.taxi-hourly.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1820.1 | 1820.1..1820.1 | 1 | - | - | - | 244.3 | - | finite=True, r2=0.738796, rmse=0.554271 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1645.2 | 1645.2..1645.2 | 1 | - | - | - | 245.0 | - | finite=True, r2=0.738796, rmse=0.554271 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host not sampled; GPU not sampled

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'nonlinearity': 'tanh', 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (attributes) | mojolearn (attributes) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| batch_size | 256 | 256 | 256 | 256 | 256 | 256 |
| betas | - | - | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| eps | - | - | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| hidden_size | 64 | 64 | 64 | 64 | 64 | 64 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| num_layers | 1 | 1 | 1 | 1 | 1 | 1 |
| seed | 7 | 7 | 7 | 7 | 7 | 7 |
| shuffle | true | true | true | true | true | true |
| weight_decay | - | - | 0.0 | 0.0 | 0.0 | 0.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 49.0 | 49.0..49.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 46.6 | 46.6..46.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | - | - | 0 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) |
| torch-compile-fp32 | Xq | - | - | - | 0 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) |
| torch-eager-bf16 | Xq | - | - | - | 0 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) |
| torch-compile-bf16 | Xq | - | - | - | 0 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, torch-eager-fp32: predict(Xq)(Xq)

inference call, torch-compile-fp32: predict(Xq)(Xq)

inference call, torch-eager-bf16: predict(Xq)(Xq)

inference call, torch-compile-bf16: predict(Xq)(Xq)

### robust-scaler / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.robust-scaler.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 6370.1 | 6370.1..6370.1 | 1 | - | - | - | 4750.0 | - | max_abs_diff_vs_sklearn=0.0001221, output_shape=100000x220, rel_diff_vs_sklearn=7.339e-09 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1184.9 | 1184.9..1184.9 | 1 | - | - | - | 5640.5 | - | max_abs_diff_vs_sklearn=0.0001221, output_shape=100000x220, rel_diff_vs_sklearn=7.339e-09 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 5004.3 | 5004.3..5004.3 | 1 | 1.273 | 0.237 | - | 1324.2 | - | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'quantile_range': [25.0, 75.0], 'unit_variance': False, 'with_centering': True, 'with_scaling': True}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), RobustScaler (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 79.1 | 79.1..79.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 82.4 | 82.4..82.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 26.1 | 26.1..26.1 | 1 | 3.032 | 3.156 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### robust-scaler / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.robust-scaler.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 491.8 | 491.8..491.8 | 1 | - | - | - | 320.2 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x11, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 67.2 | 67.2..67.2 | 1 | - | - | - | 354.3 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x11, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 134.3 | 134.3..134.3 | 1 | 3.662 | 0.501 | - | 210.5 | - | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'quantile_range': [25.0, 75.0], 'unit_variance': False, 'with_centering': True, 'with_scaling': True}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), RobustScaler (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 5.4 | 5.4..5.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 4.3 | 4.3..4.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 2.2 | 2.2..2.2 | 1 | 2.437 | 1.928 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### select-chi2 / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.select-chi2.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 388.4 | 388.4..388.4 | 1 | - | - | - | 3615.2 | - | jaccard_vs_sklearn=0.981982, n_selected=110 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 290.3 | 290.3..290.3 | 1 | - | - | - | 3614.0 | - | jaccard_vs_sklearn=1.000000, n_selected=110 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 407.0 | 407.0..407.0 | 1 | 0.954 | 0.713 | - | 3708.7 | - | jaccard_vs_sklearn=1.000000, n_selected=110 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'k': 'half'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### select-chi2 / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.select-chi2.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 220.8 | 220.8..220.8 | 1 | - | - | - | 259.3 | - | jaccard_vs_sklearn=1.000000, n_selected=5 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 24.7 | 24.7..24.7 | 1 | - | - | - | 310.6 | - | jaccard_vs_sklearn=1.000000, n_selected=5 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 76.1 | 76.1..76.1 | 1 | 2.903 | 0.325 | - | 265.9 | - | jaccard_vs_sklearn=1.000000, n_selected=5 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'k': 'half'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### select-f-classif / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.select-f-classif.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 389.6 | 389.6..389.6 | 1 | - | - | - | 2689.0 | - | jaccard_vs_sklearn=1.000000, n_selected=110 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 342.6 | 342.6..342.6 | 1 | - | - | - | 2686.8 | - | jaccard_vs_sklearn=1.000000, n_selected=110 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 388.8 | 388.8..388.8 | 1 | 1.002 | 0.881 | - | 3307.3 | - | jaccard_vs_sklearn=1.000000, n_selected=110 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'k': 'half'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### select-f-classif / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.select-f-classif.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 232.4 | 232.4..232.4 | 1 | - | - | - | 217.1 | - | jaccard_vs_sklearn=1.000000, n_selected=5 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 122.6 | 122.6..122.6 | 1 | - | - | - | 259.4 | - | jaccard_vs_sklearn=1.000000, n_selected=5 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 53.9 | 53.9..53.9 | 1 | 4.313 | 2.275 | - | 226.1 | - | jaccard_vs_sklearn=1.000000, n_selected=5 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'k': 'half'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### select-f-regression / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.select-f-regression.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 310.0 | 310.0..310.0 | 1 | - | - | - | 2679.0 | - | jaccard_vs_sklearn=1.000000, n_selected=110 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 288.5 | 288.5..288.5 | 1 | - | - | - | 2678.8 | - | jaccard_vs_sklearn=1.000000, n_selected=110 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 422.0 | 422.0..422.0 | 1 | 0.735 | 0.684 | - | 2764.6 | - | jaccard_vs_sklearn=1.000000, n_selected=110 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'k': 'half'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### select-f-regression / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.select-f-regression.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 153.5 | 153.5..153.5 | 1 | - | - | - | 203.9 | - | jaccard_vs_sklearn=1.000000, n_selected=5 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 127.4 | 127.4..127.4 | 1 | - | - | - | 250.0 | - | jaccard_vs_sklearn=1.000000, n_selected=5 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 43.3 | 43.3..43.3 | 1 | 3.543 | 2.939 | - | 196.6 | - | jaccard_vs_sklearn=1.000000, n_selected=5 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'k': 'half'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### select-mutual-info-reg / istella (rows full, shape X 100000x220; Xq 100000x220; y 100000; yq 100000)

race: done, driver rc 0, log `logs/algos.select-mutual-info-reg.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 37468.7 | 37468.7..37468.7 | 1 | - | - | - | 1747.7 | - | jaccard_vs_sklearn=0.896552, n_selected=110 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 340.2 | 340.2..340.2 | 1 | - | - | - | 1753.5 | - | jaccard_vs_sklearn=0.286550, n_selected=110 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 60325.4 | 60325.4..60325.4 | 1 | 0.621 | 0.006 | - | 1179.7 | - | jaccard_vs_sklearn=1.000000, n_selected=110 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'k': 'half'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | "none (no argument; draws random numbers, see exceptions)" | "none (no argument; draws random numbers, see exceptions)" | "none (no argument; draws random numbers, see exceptions)" |

accepted difference: ours seed: SelectKBest takes no seed argument on either side; the noise seed 7 is bound into score_func=mutual_info_regression(random_state=7) on both arms

accepted difference: ours-fast seed: SelectKBest takes no seed argument on either side; the noise seed 7 is bound into score_func=mutual_info_regression(random_state=7) on both arms

accepted difference: sklearn-cpu seed: SelectKBest takes no seed argument on either side; the noise seed 7 is bound into score_func=mutual_info_regression(random_state=7) on both arms

### select-mutual-info-reg / taxi (rows full, shape X 100000x11; Xq 100000x11; y 100000; yq 100000)

race: done, driver rc 0, log `logs/algos.select-mutual-info-reg.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 7547.8 | 7547.8..7547.8 | 1 | - | - | - | 155.1 | - | jaccard_vs_sklearn=1.000000, n_selected=5 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 9980.5 | 9980.5..9980.5 | 1 | - | - | - | 154.8 | - | jaccard_vs_sklearn=1.000000, n_selected=5 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 3426.1 | 3426.1..3426.1 | 1 | 2.203 | 2.913 | - | 237.0 | - | jaccard_vs_sklearn=1.000000, n_selected=5 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'k': 'half'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | "none (no argument; draws random numbers, see exceptions)" | "none (no argument; draws random numbers, see exceptions)" | "none (no argument; draws random numbers, see exceptions)" |

accepted difference: ours seed: SelectKBest takes no seed argument on either side; the noise seed 7 is bound into score_func=mutual_info_regression(random_state=7) on both arms

accepted difference: ours-fast seed: SelectKBest takes no seed argument on either side; the noise seed 7 is bound into score_func=mutual_info_regression(random_state=7) on both arms

accepted difference: sklearn-cpu seed: SelectKBest takes no seed argument on either side; the noise seed 7 is bound into score_func=mutual_info_regression(random_state=7) on both arms

### select-mutual-info / istella (rows full, shape X 100000x220; Xq 100000x220; y 100000; yq 100000)

race: done, driver rc 0, log `logs/algos.select-mutual-info.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 2479.4 | 2479.4..2479.4 | 1 | - | - | - | 2706.2 | - | jaccard_vs_sklearn=0.929825, n_selected=110 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 2399.7 | 2399.7..2399.7 | 1 | - | - | - | 2704.0 | - | jaccard_vs_sklearn=0.929825, n_selected=110 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 45059.1 | 45059.1..45059.1 | 1 | 0.055 | 0.053 | - | 1193.3 | - | jaccard_vs_sklearn=1.000000, n_selected=110 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'k': 'half'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | "none (no argument; draws random numbers, see exceptions)" | "none (no argument; draws random numbers, see exceptions)" | "none (no argument; draws random numbers, see exceptions)" |

accepted difference: ours seed: SelectKBest takes no seed argument on either side; the noise seed 7 is bound into score_func=mutual_info_classif(random_state=7) on both arms

accepted difference: ours-fast seed: SelectKBest takes no seed argument on either side; the noise seed 7 is bound into score_func=mutual_info_classif(random_state=7) on both arms

accepted difference: sklearn-cpu seed: SelectKBest takes no seed argument on either side; the noise seed 7 is bound into score_func=mutual_info_classif(random_state=7) on both arms

### select-mutual-info / taxi (rows full, shape X 100000x11; Xq 100000x11; y 100000; yq 100000)

race: done, driver rc 0, log `logs/algos.select-mutual-info.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 339.5 | 339.5..339.5 | 1 | - | - | - | 222.1 | - | jaccard_vs_sklearn=0.428571, n_selected=5 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 289.7 | 289.7..289.7 | 1 | - | - | - | 224.6 | - | jaccard_vs_sklearn=0.428571, n_selected=5 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 2199.9 | 2199.9..2199.9 | 1 | 0.154 | 0.132 | - | 240.3 | - | jaccard_vs_sklearn=1.000000, n_selected=5 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'k': 'half'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | "none (no argument; draws random numbers, see exceptions)" | "none (no argument; draws random numbers, see exceptions)" | "none (no argument; draws random numbers, see exceptions)" |

accepted difference: ours seed: SelectKBest takes no seed argument on either side; the noise seed 7 is bound into score_func=mutual_info_classif(random_state=7) on both arms

accepted difference: ours-fast seed: SelectKBest takes no seed argument on either side; the noise seed 7 is bound into score_func=mutual_info_classif(random_state=7) on both arms

accepted difference: sklearn-cpu seed: SelectKBest takes no seed argument on either side; the noise seed 7 is bound into score_func=mutual_info_classif(random_state=7) on both arms

### select-r-regression / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.select-r-regression.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 315.1 | 315.1..315.1 | 1 | - | - | - | 2675.0 | - | jaccard_vs_sklearn=1.000000, n_selected=110 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 299.1 | 299.1..299.1 | 1 | - | - | - | 2679.6 | - | jaccard_vs_sklearn=1.000000, n_selected=110 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 385.4 | 385.4..385.4 | 1 | 0.818 | 0.776 | - | 2760.0 | - | jaccard_vs_sklearn=1.000000, n_selected=110 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'k': 'half'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### select-r-regression / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.select-r-regression.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 151.9 | 151.9..151.9 | 1 | - | - | - | 205.8 | - | jaccard_vs_sklearn=1.000000, n_selected=5 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 126.9 | 126.9..126.9 | 1 | - | - | - | 253.8 | - | jaccard_vs_sklearn=1.000000, n_selected=5 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 42.6 | 42.6..42.6 | 1 | 3.564 | 2.979 | - | 204.8 | - | jaccard_vs_sklearn=1.000000, n_selected=5 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'k': 'half'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### sgd / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.sgd.synthetic.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1219.2 | 1219.2..1219.2 | 1 | - | - | - | 1345.0 | - | rel_fro_vs_torch_eager_fp32=6.095e-10 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested identical, read back unknown) |
| mojolearn FAST | mojolearn | gpu | fast | 1178.1 | 1178.1..1178.1 | 1 | - | - | - | 1346.5 | - | rel_fro_vs_torch_eager_fp32=6.095e-10 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested fast, read back unknown) |
| torch-eager-fp32 | torch | gpu | opponent | 40.3 | 40.3..40.3 | 1 | - | - | - | 1388.9 | 1024.4 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 215.5 | 215.5..215.5 | 1 | - | - | - | 1484.9 | 1024.4 | rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-eager-fp32, torch-compile-fp32: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

settings: {'dampening': 0.0, 'lr': 0.001, 'maximize': False, 'momentum': 0.9, 'nesterov': False, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | torch-compile-fp32 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | torch (optimizer.defaults) | torch (optimizer.defaults) |
| dampening | 0.0 | 0.0 | 0.0 | 0.0 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 |
| momentum | 0.9 | 0.9 | 0.9 | 0.9 |
| nesterov | false | false | false | false |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| weight_decay | 0.0 | 0.0 | 0.0 | 0.0 |

### simple-imputer / istella (rows full, shape X 1000000x220; X_true 1000000x220; Xq 100000x220; Xq_true 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.simple-imputer.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 5790.9 | 5790.9..5790.9 | 1 | - | - | - | 6368.2 | - | masked_rmse=346849.129968, max_abs_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1049.5 | 1049.5..1049.5 | 1 | - | - | - | 7257.1 | - | error=IndexError('boolean index did not match indexed array along axis 1; size of axis is 68 but size of corresponding boolean axis is 220') | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 13448.4 | 13448.4..13448.4 | 1 | 0.431 | 0.078 | - | 7274.9 | - | masked_rmse=346849.129968 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'add_indicator': False, 'keep_empty_features': False, 'strategy': 'median'}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), SimpleImputer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 116.3 | 116.3..116.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 61.4 | 61.4..61.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 119.4 | 119.4..119.4 | 1 | 0.974 | 0.514 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### simple-imputer / taxi (rows full, shape X 1000000x11; X_true 1000000x11; Xq 100000x11; Xq_true 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.simple-imputer.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 396.0 | 396.0..396.0 | 1 | - | - | - | 423.1 | - | masked_rmse=5.985180, max_abs_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 50.4 | 50.4..50.4 | 1 | - | - | - | 459.1 | - | masked_rmse=5.985180, max_abs_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 460.7 | 460.7..460.7 | 1 | 0.860 | 0.109 | - | 325.1 | - | masked_rmse=5.985180 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'add_indicator': False, 'keep_empty_features': False, 'strategy': 'median'}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), SimpleImputer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 3.8 | 3.8..3.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 3.3 | 3.3..3.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 5.2 | 5.2..5.2 | 1 | 0.732 | 0.629 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### skewed-chi2 / istella (rows full, shape X 100000x220; Xq 1000x220; y 100000; yq 1000)

race: done, driver rc 0, log `logs/algos.skewed-chi2.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 8.3 | 8.3..8.3 | 1 | - | - | - | 1007.5 | - | kernel_rel_error=0.671898 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 8.1 | 8.1..8.1 | 1 | - | - | - | 1012.1 | - | kernel_rel_error=0.671898 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 5.2 | 5.2..5.2 | 1 | 1.607 | 1.567 | - | 1083.4 | - | kernel_rel_error=0.671899 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 5.5 | 5.5..5.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 5.3 | 5.3..5.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 1.4 | 1.4..1.4 | 1 | 3.912 | 3.822 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### skewed-chi2 / taxi (rows full, shape X 100000x11; Xq 1000x11; y 100000; yq 1000)

race: done, driver rc 0, log `logs/algos.skewed-chi2.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1.3 | 1.3..1.3 | 1 | - | - | - | 115.4 | - | kernel_rel_error=0.037749 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1.3 | 1.3..1.3 | 1 | - | - | - | 120.1 | - | kernel_rel_error=0.037749 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 0.7 | 0.7..0.7 | 1 | 1.908 | 1.896 | - | 201.8 | - | kernel_rel_error=0.037749 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 1.8 | 1.8..1.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 1.7 | 1.7..1.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 0.7 | 0.7..0.7 | 1 | 2.433 | 2.384 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### sparse-coder / istella (rows full, shape X 5000x220; Xq 100000x220; y 5000; yq 100000)

race: done, driver rc 0, log `logs/algos.sparse-coder.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3.5 | 3.5..3.5 | 1 | - | - | - | 4801.9 | - | max_abs_diff_vs_sklearn=6.104e-05, output_shape=100000x64, rel_diff_vs_sklearn=1.11e-07 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 3.5 | 3.5..3.5 | 1 | - | - | - | 4803.3 | - | max_abs_diff_vs_sklearn=6.104e-05, output_shape=100000x64, rel_diff_vs_sklearn=1.11e-07 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 0.0 | 0.0..0.0 | 1 | 98.807 | 98.901 | - | 1244.0 | - | output_shape=100000x64 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'positive_code': False, 'split_sign': False, 'transform_algorithm': 'omp', 'transform_alpha': None, 'transform_max_iter': 1000, 'transform_n_nonzero_coefs': 4}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 313.1 | 313.1..313.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 319.6 | 319.6..319.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 4161.6 | 4161.6..4161.6 | 1 | 0.075 | 0.077 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### sparse-coder / taxi (rows full, shape X 5000x11; Xq 100000x11; y 5000; yq 100000)

race: done, driver rc 0, log `logs/algos.sparse-coder.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1.5 | 1.5..1.5 | 1 | - | - | - | 3665.3 | - | max_abs_diff_vs_sklearn=1.049e-05, output_shape=100000x64, rel_diff_vs_sklearn=9.976e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1.6 | 1.6..1.6 | 1 | - | - | - | 3664.9 | - | max_abs_diff_vs_sklearn=1.049e-05, output_shape=100000x64, rel_diff_vs_sklearn=9.976e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 0.0 | 0.0..0.0 | 1 | 44.847 | 47.224 | - | 301.2 | - | output_shape=100000x64 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'positive_code': False, 'split_sign': False, 'transform_algorithm': 'omp', 'transform_alpha': None, 'transform_max_iter': 1000, 'transform_n_nonzero_coefs': 4}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 223.9 | 223.9..223.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 224.8 | 224.8..224.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 4190.8 | 4190.8..4190.8 | 1 | 0.053 | 0.054 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### sparse-pca / istella (rows full, shape X 20000x220; Xq 20000x220; y 20000; yq 20000)

race: done, driver rc 0, log `logs/algos.sparse-pca.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "Exception(\"the one-sided Jacobi SVD did not converge in 60 sweeps at n_cols = 220: the last sweep still performed 17 rotations against a tolerance of 9.536743e-07. The remedy is more sweep) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "Exception(\"the one-sided Jacobi SVD did not converge in 60 sweeps at n_cols = 220: the last sweep still performed 14 rotations against a tolerance of 9.536743e-07. The remedy is more sweep) |
| sklearn-cpu | scikit-learn | cpu | opponent | 30803.9 | 30803.9..30803.9 | 1 | - | - | - | 1126.5 | - | component_sparsity=0.305682, relative_reconstruction_error=0.750210 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host not sampled; GPU not sampled

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'max_iter': 100, 'method': 'lars', 'n_components': 8, 'random_state': 7, 'ridge_alpha': 0.01, 'tol': 1e-06}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 1.0 | 1.0 | 1.0 |
| max_iter | 100 | 100 | 100 |
| n_components | 8 | 8 | 8 |
| seed | 7 | 7 | 7 |
| tol | 1e-06 | 1e-06 | 1e-06 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(error: {"error": "Exception(\"the one-sided Jacobi SVD did not converge in 60 sweeps at n_cols = 220: the last sweep still performed 17 rotations against a tolerance of 9.536743e-07. The remedy is more sweep) |
| mojolearn FAST | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(error: {"error": "Exception(\"the one-sided Jacobi SVD did not converge in 60 sweeps at n_cols = 220: the last sweep still performed 14 rotations against a tolerance of 9.536743e-07. The remedy is more sweep) |
| sklearn-cpu | Xq | - | 10.1 | 10.1..10.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### sparse-pca / taxi (rows full, shape X 20000x11; Xq 20000x11; y 20000; yq 20000)

race: done, driver rc 0, log `logs/algos.sparse-pca.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 5274.1 | 5274.1..5274.1 | 1 | - | - | - | 137.2 | - | component_sparsity=0.488636, relative_reconstruction_error=0.277226 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 5383.3 | 5383.3..5383.3 | 1 | - | - | - | 133.8 | - | component_sparsity=0.488636, relative_reconstruction_error=0.277226 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 8156.5 | 8156.5..8156.5 | 1 | 0.647 | 0.660 | - | 201.3 | - | component_sparsity=0.488636, relative_reconstruction_error=0.277226 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'max_iter': 100, 'method': 'lars', 'n_components': 8, 'random_state': 7, 'ridge_alpha': 0.01, 'tol': 1e-06}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 1.0 | 1.0 | 1.0 |
| max_iter | 100 | 100 | 100 |
| n_components | 8 | 8 | 8 |
| seed | 7 | 7 | 7 |
| tol | 1e-06 | 1e-06 | 1e-06 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 7.9 | 7.9..7.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 7.6 | 7.6..7.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 3.2 | 3.2..3.2 | 1 | 2.487 | 2.406 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### sparse-rp / istella (rows full, shape X 900000x220; Xq 100000x220)

race: done, driver rc 0, log `logs/algos.sparse-rp.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 201.4 | 201.4..201.4 | 1 | - | - | - | 3985.7 | - | mean_abs_distortion=1.883381 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 211.3 | 211.3..211.3 | 1 | - | - | - | 3986.2 | - | mean_abs_distortion=1.883381 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 33.6 | 33.6..33.6 | 1 | 5.993 | 6.289 | - | 999.7 | - | mean_abs_distortion=0.474347 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'compute_inverse_components': False, 'dense_output': False, 'density': 'auto', 'eps': 0.1, 'n_components': 10, 'random_state': 7}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), SparseRandomProjection (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| eps | 0.1 | 0.1 | 0.1 |
| n_components | 10 | 10 | 10 |
| seed | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 36.3 | 36.3..36.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 36.5 | 36.5..36.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 38.1 | 38.1..38.1 | 1 | 0.952 | 0.960 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### sparse-rp / taxi (rows full, shape X 900000x11; Xq 100000x11)

race: done, driver rc 0, log `logs/algos.sparse-rp.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 13.8 | 13.8..13.8 | 1 | - | - | - | 274.7 | - | mean_abs_distortion=0.147163 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 13.9 | 13.9..13.9 | 1 | - | - | - | 271.9 | - | mean_abs_distortion=0.147163 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 2.4 | 2.4..2.4 | 1 | 5.678 | 5.739 | - | 201.9 | - | mean_abs_distortion=0.381016 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'compute_inverse_components': False, 'dense_output': False, 'density': 'auto', 'eps': 0.1, 'n_components': 10, 'random_state': 7}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), SparseRandomProjection (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| eps | 0.1 | 0.1 | 0.1 |
| n_components | 10 | 10 | 10 |
| seed | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 6.7 | 6.7..6.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 6.7 | 6.7..6.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 1.6 | 1.6..1.6 | 1 | 4.036 | 4.041 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### spline / istella (rows full, shape X 1000000x16; Xq 100000x16; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.spline.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 175.8 | 175.8..175.8 | 1 | - | - | - | 1434.0 | - | max_abs_diff_vs_sklearn=1.788e-07, output_shape=100000x112, rel_diff_vs_sklearn=6.226e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 57.3 | 57.3..57.3 | 1 | - | - | - | 1437.6 | - | max_abs_diff_vs_sklearn=1.788e-07, output_shape=100000x112, rel_diff_vs_sklearn=6.424e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 20.9 | 20.9..20.9 | 1 | 8.407 | 2.739 | - | 1275.4 | - | output_shape=100000x112 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'degree': 3, 'extrapolation': 'constant', 'include_bias': True, 'knots': 'uniform', 'n_knots': 5}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| degree | 3 | 3 | 3 |
| order | "C" | "C" | "C" |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 37.4 | 37.4..37.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 28.1 | 28.1..28.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 104.5 | 104.5..104.5 | 1 | 0.358 | 0.269 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### spline / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.spline.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 170.3 | 170.3..170.3 | 1 | - | - | - | 400.5 | - | max_abs_diff_vs_sklearn=1.788e-07, output_shape=100000x77, rel_diff_vs_sklearn=5.888e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 40.2 | 40.2..40.2 | 1 | - | - | - | 400.6 | - | max_abs_diff_vs_sklearn=1.192e-07, output_shape=100000x77, rel_diff_vs_sklearn=6.181e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 20.7 | 20.7..20.7 | 1 | 8.226 | 1.940 | - | 317.6 | - | output_shape=100000x77 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'degree': 3, 'extrapolation': 'constant', 'include_bias': True, 'knots': 'uniform', 'n_knots': 5}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| degree | 3 | 3 | 3 |
| order | "C" | "C" | "C" |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 38.4 | 38.4..38.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 21.3 | 21.3..21.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 67.9 | 67.9..67.9 | 1 | 0.566 | 0.314 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### standard-scaler / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.standard-scaler.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 542.6 | 542.6..542.6 | 1 | - | - | - | 3937.5 | - | max_abs_diff_vs_sklearn=0.002426, output_shape=100000x220, rel_diff_vs_sklearn=3.59e-06 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 535.9 | 535.9..535.9 | 1 | - | - | - | 3936.7 | - | max_abs_diff_vs_sklearn=0.002426, output_shape=100000x220, rel_diff_vs_sklearn=3.591e-06 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 567.4 | 567.4..567.4 | 1 | 0.956 | 0.944 | - | 3047.7 | - | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'with_mean': True, 'with_std': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 93.5 | 93.5..93.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 95.9 | 95.9..95.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 36.8 | 36.8..36.8 | 1 | 2.542 | 2.609 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### standard-scaler / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.standard-scaler.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 34.5 | 34.5..34.5 | 1 | - | - | - | 266.1 | - | max_abs_diff_vs_sklearn=0.000103, output_shape=100000x11, rel_diff_vs_sklearn=3.846e-06 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 41.0 | 41.0..41.0 | 1 | - | - | - | 265.1 | - | max_abs_diff_vs_sklearn=0.000103, output_shape=100000x11, rel_diff_vs_sklearn=3.84e-06 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 58.0 | 58.0..58.0 | 1 | 0.594 | 0.706 | - | 227.0 | - | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'with_mean': True, 'with_std': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 4.7 | 4.7..4.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 4.2 | 4.2..4.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 2.2 | 2.2..2.2 | 1 | 2.116 | 1.898 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### stl / synthetic (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc 0, log `logs/algos.stl.synthetic.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 683.2 | 683.2..683.2 | 1 | - | - | - | 74.7 | - | rel_diff_vs_statsmodels=6.508e-07, residual_std=0.783175 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 599.0 | 599.0..599.0 | 1 | - | - | - | 73.4 | - | rel_diff_vs_statsmodels=6.535e-07, residual_std=0.783175 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsmodels-cpu | statsmodels | cpu | opponent | 181.8 | 181.8..181.8 | 1 | 3.759 | 3.295 | - | 52.2 | - | residual_std=0.783175 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, statsmodels-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'low_pass': None, 'low_pass_deg': 1, 'low_pass_jump': 1, 'period': 24, 'robust': False, 'seasonal': 7, 'seasonal_deg': 1, 'seasonal_jump': 1, 'trend': None, 'trend_deg': 1, 'trend_jump': 1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | statsmodels-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | statsmodels (declared) |
| seasonal | 7 | 7 | 7 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| trend | null | null | null |

accepted difference: ours-fast trend: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: statsmodels-cpu trend: None on both ours and statsmodels-cpu: the same documented setting in both signatures

### stl / taxi-hourly (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc 0, log `logs/algos.stl.taxi-hourly.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 682.9 | 682.9..682.9 | 1 | - | - | - | 74.4 | - | rel_diff_vs_statsmodels=1.212e-06, residual_std=18.312475 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 599.8 | 599.8..599.8 | 1 | - | - | - | 73.1 | - | rel_diff_vs_statsmodels=1.258e-06, residual_std=18.312476 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsmodels-cpu | statsmodels | cpu | opponent | 212.3 | 212.3..212.3 | 1 | 3.217 | 2.826 | - | 52.2 | - | residual_std=18.312475 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, statsmodels-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'low_pass': None, 'low_pass_deg': 1, 'low_pass_jump': 1, 'period': 24, 'robust': False, 'seasonal': 7, 'seasonal_deg': 1, 'seasonal_jump': 1, 'trend': None, 'trend_deg': 1, 'trend_jump': 1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | statsmodels-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | statsmodels (declared) |
| seasonal | 7 | 7 | 7 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| trend | null | null | null |

accepted difference: ours-fast trend: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: statsmodels-cpu trend: None on both ours and statsmodels-cpu: the same documented setting in both signatures

### svd / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.svd.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 4212.8 | 4212.8..4212.8 | 1 | - | - | - | 617.7 | - | max_rel_singular_value_error=6.332e-07, relative_reconstruction_error_100k_rows=0.000333 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 4204.7 | 4204.7..4204.7 | 1 | - | - | - | 631.9 | - | max_rel_singular_value_error=6.332e-07, relative_reconstruction_error_100k_rows=0.000333 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| numpy-cpu | numpy | cpu | opponent | 121.5 | 121.5..121.5 | 1 | 34.679 | 34.612 | - | 140.7 | - | max_rel_singular_value_error=4.308e-08, relative_reconstruction_error_100k_rows=4.314e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-gpu | torch | gpu | opponent | 63.8 | 63.8..63.8 | 1 | 66.040 | 65.914 | - | 1254.7 | 1032.4 | max_rel_singular_value_error=2.225e-06, relative_reconstruction_error_100k_rows=1.351e-05 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, numpy-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

memory, torch-gpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

settings: {'full_matrices': False}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | numpy-cpu | ours | ours-fast | torch-gpu |
|---|---||---|---||---|---||---|---|
| library (source) | numpy (declared) | mojolearn (declared) | mojolearn (declared) | torch (declared) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### svgp / istella (rows full, shape X 100000x220; Xq 20000x220; y 100000; yq 20000)

race: done, driver rc 0, log `logs/algos.svgp.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 4660.4 | 4660.4..4660.4 | 1 | - | - | - | 1324.4 | - | finite=True, r2=-0.106016, rmse=0.878373 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 2126.6 | 2126.6..2126.6 | 1 | - | - | - | 1325.3 | - | finite=True, r2=-0.106016, rmse=0.878373 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| gpytorch-gpu | gpytorch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) (measured this run) |
| gpytorch-cpu | gpytorch | cpu | opponent | 474.3 | 474.3..474.3 | 1 | 9.825 | 4.483 | - | 1445.0 | - | finite=True, r2=-0.106040, rmse=0.878383 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, gpytorch-gpu: host not sampled; GPU not sampled

memory, gpytorch-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'jitter': 1e-06, 'kernel_variance': 1.0, 'lengthscale': 1.0, 'n_inducing': 512, 'noise_variance': 1.0}. Rows: None. Timed: None.

mismatch: no seed on any arm: nothing is drawn (fixed inducing points, closed form)

mismatch: jitter: ours 1e-6 on K_uu; gpytorch adds its own Cholesky jitter (1e-6 in float32) only when a factorization fails

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | gpytorch-cpu | gpytorch-gpu | ours | ours-fast |
|---|---||---|---||---|---||---|---|
| library (source) | gpytorch (declared) | gpytorch (declared) | mojolearn (declared) | mojolearn (declared) |
| seed | 7 | 7 | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 497.0 | 497.0..497.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 501.4 | 501.4..501.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| gpytorch-gpu | Xq | - | - | - | 0 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) |
| gpytorch-cpu | Xq | - | 146.7 | 146.7..146.7 | 1 | 3.388 | 3.419 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, gpytorch-gpu: predict(Xq)(Xq)

inference call, gpytorch-cpu: predict(Xq)(Xq)

### svgp / taxi (rows full, shape X 100000x11; Xq 20000x11; y 100000; yq 20000)

race: done, driver rc 0, log `logs/algos.svgp.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "ValueError('SVGP: the inducing system is not positive definite; raise jitter or noise_variance')", "event": "error", "stage": "round 0"}) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "ValueError('SVGP: the inducing system is not positive definite; raise jitter or noise_variance')", "event": "error", "stage": "round 0"}) |
| gpytorch-gpu | gpytorch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) (measured this run) |
| gpytorch-cpu | gpytorch | cpu | opponent | 393.4 | 393.4..393.4 | 1 | - | - | - | 466.2 | - | finite=True, r2=-0.209325, rmse=17.829528 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, ours-fast, gpytorch-gpu: host not sampled; GPU not sampled

memory, gpytorch-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'jitter': 1e-06, 'kernel_variance': 1.0, 'lengthscale': 1.0, 'n_inducing': 512, 'noise_variance': 1.0}. Rows: None. Timed: None.

mismatch: no seed on any arm: nothing is drawn (fixed inducing points, closed form)

mismatch: jitter: ours 1e-6 on K_uu; gpytorch adds its own Cholesky jitter (1e-6 in float32) only when a factorization fails

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | gpytorch-cpu | gpytorch-gpu | ours | ours-fast |
|---|---||---|---||---|---||---|---|
| library (source) | gpytorch (declared) | gpytorch (declared) | mojolearn (declared) | mojolearn (declared) |
| seed | 7 | 7 | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(error: {"error": "ValueError('SVGP: the inducing system is not positive definite; raise jitter or noise_variance')", "event": "error", "stage": "round 0"}) |
| mojolearn FAST | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(error: {"error": "ValueError('SVGP: the inducing system is not positive definite; raise jitter or noise_variance')", "event": "error", "stage": "round 0"}) |
| gpytorch-gpu | Xq | - | - | - | 0 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) |
| gpytorch-cpu | Xq | - | 106.2 | 106.2..106.2 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, gpytorch-gpu: predict(Xq)(Xq)

inference call, gpytorch-cpu: predict(Xq)(Xq)

### target-encoder / istella (rows full, shape X 1000000x8; Xq 100000x8; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.target-encoder.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 719.8 | 719.8..719.8 | 1 | - | - | - | 501.1 | - | max_abs_diff_vs_sklearn=1.074e-08, output_shape=100000x8, rel_diff_vs_sklearn=2.361e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 539.0 | 539.0..539.0 | 1 | - | - | - | 517.7 | - | max_abs_diff_vs_sklearn=1.074e-08, output_shape=100000x8, rel_diff_vs_sklearn=2.361e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 280.0 | 280.0..280.0 | 1 | 2.571 | 1.925 | - | 195.8 | - | output_shape=100000x8 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'categories': 'auto', 'cv': 4, 'random_state': 42, 'shuffle': True, 'smooth': 0.0, 'target_type': 'binary'}. Rows: None. Timed: None.

mismatch: fold assignment: cuML 'interleaved' (row i in fold i mod 4, the cuML benchmark's cuml_args); scikit-learn and ours a KFold shuffled by seed 42 (its cpu_args)

config: cuML benchmark (RAPIDS), TargetEncoder (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 42): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| cv | 4 | 4 | 4 |
| seed | 42 | 42 | 42 |
| shuffle | true | true | true |
| smooth | 0.0 | 0.0 | 0.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 8.2 | 8.2..8.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 8.3 | 8.3..8.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 22.2 | 22.2..22.2 | 1 | 0.369 | 0.376 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### target-encoder / taxi (rows full, shape X 1000000x5; Xq 100000x5; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.target-encoder.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 676.2 | 676.2..676.2 | 1 | - | - | - | 363.2 | - | max_abs_diff_vs_sklearn=2.965e-08, output_shape=100000x5, rel_diff_vs_sklearn=1.581e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 540.5 | 540.5..540.5 | 1 | - | - | - | 379.2 | - | max_abs_diff_vs_sklearn=2.965e-08, output_shape=100000x5, rel_diff_vs_sklearn=1.581e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 179.6 | 179.6..179.6 | 1 | 3.766 | 3.010 | - | 203.5 | - | output_shape=100000x5 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'categories': 'auto', 'cv': 4, 'random_state': 42, 'shuffle': True, 'smooth': 0.0, 'target_type': 'binary'}. Rows: None. Timed: None.

mismatch: fold assignment: cuML 'interleaved' (row i in fold i mod 4, the cuML benchmark's cuml_args); scikit-learn and ours a KFold shuffled by seed 42 (its cpu_args)

config: cuML benchmark (RAPIDS), TargetEncoder (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 42): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| cv | 4 | 4 | 4 |
| seed | 42 | 42 | 42 |
| shuffle | true | true | true |
| smooth | 0.0 | 0.0 | 0.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 7.2 | 7.2..7.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 7.4 | 7.4..7.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 13.1 | 13.1..13.1 | 1 | 0.549 | 0.564 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### theta / synthetic (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc 0, log `logs/algos.theta.synthetic.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 167.7 | 167.7..167.7 | 1 | - | - | - | 67.1 | - | forecast_rmse=1.436610 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 159.0 | 159.0..159.0 | 1 | - | - | - | 68.0 | - | forecast_rmse=1.436606 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsforecast-cpu | statsforecast | cpu | opponent | 634.8 | 634.8..634.8 | 1 | 0.264 | 0.250 | - | 184.9 | - | forecast_rmse=1.436557 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| statsmodels-cpu | statsmodels | cpu | opponent | 149.2 | 149.2..149.2 | 1 | 1.124 | 1.066 | - | 50.3 | - | forecast_rmse=1.434862 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, statsforecast-cpu, statsmodels-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'decomposition_type': 'multiplicative', 'season_length': 24}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | statsforecast-cpu | statsmodels-cpu |
|---|---||---|---||---|---||---|---|
| library (source) | mojolearn (attributes) | mojolearn (attributes) | statsforecast (declared) | statsmodels (declared) |
| alpha | null | null | - | - |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

accepted difference: ours-fast alpha: None on both ours and ours-fast: the same documented setting in both signatures

### theta / taxi-hourly (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc 0, log `logs/algos.theta.taxi-hourly.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 477.9 | 477.9..477.9 | 1 | - | - | - | 69.8 | - | forecast_rmse=49.020604 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 2747.0 | 2747.0..2747.0 | 1 | - | - | - | 68.8 | - | forecast_rmse=49.281168 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsforecast-cpu | statsforecast | cpu | opponent | 1848.1 | 1848.1..1848.1 | 1 | 0.259 | 1.486 | - | 187.5 | - | forecast_rmse=49.253901 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| statsmodels-cpu | statsmodels | cpu | opponent | 198.1 | 198.1..198.1 | 1 | 2.413 | 13.869 | - | 51.3 | - | forecast_rmse=49.311757 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, statsforecast-cpu, statsmodels-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'decomposition_type': 'multiplicative', 'season_length': 24}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | statsforecast-cpu | statsmodels-cpu |
|---|---||---|---||---|---||---|---|
| library (source) | mojolearn (attributes) | mojolearn (attributes) | statsforecast (declared) | statsmodels (declared) |
| alpha | null | null | - | - |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

accepted difference: ours-fast alpha: None on both ours and ours-fast: the same documented setting in both signatures

### var / synthetic (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc 0, log `logs/algos.var.synthetic.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 24.0 | 24.0..24.0 | 1 | - | - | - | 67.9 | - | forecast_rmse=1.140864 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 18.2 | 18.2..18.2 | 1 | - | - | - | 65.6 | - | forecast_rmse=1.140787 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsmodels-cpu | statsmodels | cpu | opponent | 3.2 | 3.2..3.2 | 1 | 7.455 | 5.640 | - | 149.0 | - | forecast_rmse=1.144945 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, statsmodels-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'ic': None, 'maxlags': 2, 'method': 'ols', 'trend': 'c'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | statsmodels-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | statsmodels (declared) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| trend | "c" | "c" | "c" |

### var / taxi-hourly (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc 0, log `logs/algos.var.taxi-hourly.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 26.2 | 26.2..26.2 | 1 | - | - | - | 66.0 | - | forecast_rmse=33.167951 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 23.1 | 23.1..23.1 | 1 | - | - | - | 66.8 | - | forecast_rmse=33.167955 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsmodels-cpu | statsmodels | cpu | opponent | 3.1 | 3.1..3.1 | 1 | 8.316 | 7.336 | - | 146.6 | - | forecast_rmse=33.167986 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, statsmodels-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'ic': None, 'maxlags': 2, 'method': 'ols', 'trend': 'c'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | statsmodels-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | statsmodels (declared) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| trend | "c" | "c" | "c" |

### variance-threshold / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.variance-threshold.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 280.7 | 280.7..280.7 | 1 | - | - | - | 2918.5 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x142, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 199.6 | 199.6..199.6 | 1 | - | - | - | 2917.2 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x142, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 845.4 | 845.4..845.4 | 1 | 0.332 | 0.236 | - | 4700.2 | - | output_shape=100000x142 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'threshold': 0.01}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 88.5 | 88.5..88.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 60.2 | 60.2..60.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 20.6 | 20.6..20.6 | 1 | 4.296 | 2.921 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### variance-threshold / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.variance-threshold.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 117.8 | 117.8..117.8 | 1 | - | - | - | 208.7 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x10, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 12.6 | 12.6..12.6 | 1 | - | - | - | 216.3 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x10, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 67.6 | 67.6..67.6 | 1 | 1.744 | 0.186 | - | 231.4 | - | output_shape=100000x10 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'threshold': 0.01}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 3.4 | 3.4..3.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 3.7 | 3.7..3.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 0.9 | 0.9..0.9 | 1 | 3.950 | 4.337 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

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

