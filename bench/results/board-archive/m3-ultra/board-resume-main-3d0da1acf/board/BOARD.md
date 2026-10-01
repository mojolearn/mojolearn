# mojolearn benchmark board

Generated 2026-10-01T01:27:49Z from `board.json` (schema `mojolearn-bench-board/1`).

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

Races: 1 planned, 296 done, 6 failed, 0 pending. Cells: 1032 (MODE-MISMATCH 34, REFUSED 124, ok 874).

Inference cells: 647 (REFUSED 92, ok 555).

## Quality at a glance

Per lane and dataset: our FAST value, our IDENTICAL value, our CPU IDENTICAL value, and each opponent's.

| family | lane | dataset | metric | ours FAST | ours IDENTICAL | ours CPU | opponents |
|---|---|---|---|---|---|---|---|
| algos | adaboost-clf | istella | accuracy (higher is better) | 0.934990 | 0.934990 | - | sklearn-cpu - |
| algos | adaboost-clf | istella | logloss (lower is better) | 0.438712 | 0.438712 | - | sklearn-cpu - |
| algos | adaboost-clf | taxi | accuracy (higher is better) | 0.765230 | 0.765230 | - | sklearn-cpu 0.765360 |
| algos | adaboost-clf | taxi | logloss (lower is better) | 0.543227 | 0.543227 | - | sklearn-cpu 0.540565 |
| algos | adaboost-reg | istella | r2 (higher is better) | 0.238879 | 0.234086 | - | sklearn-cpu 0.167622 |
| algos | adaboost-reg | istella | rmse (lower is better) | 0.728764 | 0.731056 | - | sklearn-cpu 0.762115 |
| algos | adaboost-reg | taxi | r2 (higher is better) | 0.216396 | -0.417994 | - | sklearn-cpu 0.563927 |
| algos | adaboost-reg | taxi | rmse (lower is better) | 14.098896 | 18.965923 | - | sklearn-cpu 10.517589 |
| algos | adafactor | synthetic | rel_fro_vs_torch_eager_fp32 | 1.45e-08 | 4.324e-05 | - | torch-eager-fp32 -; torch-compile-fp32 0.000000 |
| algos | adagrad | synthetic | rel_fro_vs_torch_eager_fp32 | 3.297e-09 | 3.297e-09 | - | torch-eager-fp32 -; torch-compile-fp32 0.000000 |
| algos | adam | synthetic | rel_fro_vs_torch_eager_fp32 | 3.34e-08 | 3.339e-08 | - | torch-eager-fp32 -; torch-compile-fp32 3.91e-08 |
| algos | adamax | synthetic | rel_fro_vs_torch_eager_fp32 | 4.267e-09 | 4.267e-09 | - | torch-eager-fp32 -; torch-compile-fp32 5.258e-09 |
| algos | adamw | synthetic | rel_fro_vs_torch_eager_fp32 | 3.34e-08 | 3.339e-08 | - | torch-eager-fp32 -; torch-compile-fp32 3.91e-08 |
| algos | als | taxi-zones | recall_at_10 (higher is better) | - | - | - | implicit-cpu 0.056786 |
| algos | auto-theta | synthetic | forecast_rmse (lower is better) | 1.438072 | 1.438912 | - | statsforecast-cpu 1.437804 |
| algos | auto-theta | taxi-hourly | forecast_rmse (lower is better) | 49.313360 | 49.054628 | - | statsforecast-cpu 49.273467 |
| algos | autoarima | synthetic | forecast_rmse (lower is better) | - | - | - | statsforecast-cpu 17.546480 |
| algos | autoarima | taxi-hourly | forecast_rmse (lower is better) | - | - | - | statsforecast-cpu 68.211648 |
| algos | avgpool1d | synthetic | max_rel_diff_vs_torch_eager_fp32 | 0.000000 | 0.000000 | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | avgpool1d | synthetic | rel_fro_vs_torch_eager_fp32 | 0.000000 | 0.000000 | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | avgpool2d | synthetic | max_rel_diff_vs_torch_eager_fp32 | 0.000000 | 0.000000 | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | avgpool2d | synthetic | rel_fro_vs_torch_eager_fp32 | 0.000000 | 0.000000 | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | bagging-clf | istella | accuracy (higher is better) | 0.941670 | 0.941670 | - | sklearn-cpu 0.941440 |
| algos | bagging-clf | istella | logloss (lower is better) | 0.149399 | 0.149399 | - | sklearn-cpu 0.152069 |
| algos | bagging-clf | taxi | accuracy (higher is better) | 0.767570 | 0.767570 | - | sklearn-cpu 0.767650 |
| algos | bagging-clf | taxi | logloss (lower is better) | 0.530450 | 0.530450 | - | sklearn-cpu 0.530078 |
| algos | bagging-reg | istella | r2 (higher is better) | 0.522033 | 0.522033 | - | sklearn-cpu 0.518475 |
| algos | bagging-reg | istella | rmse (lower is better) | 0.577510 | 0.577510 | - | sklearn-cpu 0.579656 |
| algos | bagging-reg | taxi | r2 (higher is better) | 0.918767 | 0.918767 | - | sklearn-cpu 0.938805 |
| algos | bagging-reg | taxi | rmse (lower is better) | 4.539457 | 4.539457 | - | sklearn-cpu 3.939993 |
| algos | batchnorm1d | synthetic | max_rel_diff_vs_torch_eager_fp32 | 0.063434 | 0.063434 | - | torch-eager-fp32 -; torch-compile-fp32 0.012127; torch-eager-bf16 0.000000; torch-compile-bf16 0.012127 |
| algos | batchnorm1d | synthetic | rel_fro_vs_torch_eager_fp32 | 4.096e-06 | 4.095e-06 | - | torch-eager-fp32 -; torch-compile-fp32 3.022e-07; torch-eager-bf16 0.000000; torch-compile-bf16 3.022e-07 |
| algos | batchnorm2d | synthetic | max_rel_diff_vs_torch_eager_fp32 | 0.034926 | 0.034926 | - | torch-eager-fp32 -; torch-compile-fp32 0.002228; torch-eager-bf16 0.000000; torch-compile-bf16 0.002228 |
| algos | batchnorm2d | synthetic | rel_fro_vs_torch_eager_fp32 | 2.344e-05 | 2.344e-05 | - | torch-eager-fp32 -; torch-compile-fp32 1.886e-07; torch-eager-bf16 0.000000; torch-compile-bf16 1.886e-07 |
| algos | bernoulli-nb | istella | accuracy (higher is better) | 0.794050 | 0.794050 | - | sklearn-cpu 0.794050 |
| algos | bernoulli-nb | istella | logloss (lower is better) | 5.350586 | 5.350625 | - | sklearn-cpu 4.278741 |
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
| algos | bpe-encode | enwik8 | documents_equal_to_ours | 1.000000 | 1.000000 | - | hf-tokenizers-cpu 1.000000 |
| algos | bpe-encode | enwik8 | tokens | 1457323 | 1457323 | - | hf-tokenizers-cpu 1457323 |
| algos | bpe-train | enwik8 | jaccard_vs_ours | 1.000000 | 1.000000 | - | hf-tokenizers-cpu 0.999512 |
| algos | bpe-train | enwik8 | n_tokens | 4096 | 4096 | - | hf-tokenizers-cpu 4096 |
| algos | cagra | istella | recall_at_10 (higher is better) | - | - | - | faiss-cpu 0.999375 |
| algos | cagra | taxi | recall_at_10 (higher is better) | - | - | - | faiss-cpu 0.927700 |
| algos | calibrated | istella | accuracy (higher is better) | 0.885090 | 0.885090 | - | sklearn-cpu 0.885090 |
| algos | calibrated | istella | logloss (lower is better) | 0.289832 | 0.289787 | - | sklearn-cpu 0.289787 |
| algos | calibrated | taxi | accuracy (higher is better) | 0.755330 | 0.755330 | - | sklearn-cpu 0.755330 |
| algos | calibrated | taxi | logloss (lower is better) | 0.550718 | 0.550727 | - | sklearn-cpu 0.550727 |
| algos | categorical-nb | istella | accuracy (higher is better) | 0.838850 | 0.838850 | - | sklearn-cpu 0.838850 |
| algos | categorical-nb | istella | logloss (lower is better) | 0.412625 | 0.412625 | - | sklearn-cpu 0.412625 |
| algos | categorical-nb | taxi | accuracy (higher is better) | 0.765850 | 0.765850 | - | sklearn-cpu 0.765850 |
| algos | categorical-nb | taxi | logloss (lower is better) | 0.538866 | 0.538866 | - | sklearn-cpu 0.538866 |
| algos | cca | istella | mean_canonical_corr | 0.998053 | 0.998053 | - | sklearn-cpu 0.999568 |
| algos | cca | taxi | mean_canonical_corr | 0.576863 | 0.576863 | - | sklearn-cpu 0.576863 |
| algos | cholesky | synthetic | relative_residual | 1.659e-07 | 2.9e-07 | - | numpy-cpu 3.928e-08; torch-gpu 5.449e-07 |
| algos | classical-mds | istella | trustworthiness_k15 (higher is better, 1 at most) | 0.830548 | - | - | sklearn-cpu 0.830548 |
| algos | classical-mds | taxi | trustworthiness_k15 (higher is better, 1 at most) | 0.765649 | - | - | sklearn-cpu 0.765649 |
| algos | clip-grad-norm | synthetic | norm | 1.000000 | 1.000000 | - | torch-eager-fp32 1.000000; torch-compile-fp32 1.000000 |
| algos | clip-grad-norm | synthetic | norm_rel_diff_vs_ours | 0.000000 | 0.000000 | - | torch-eager-fp32 0.000000; torch-compile-fp32 0.000000 |
| algos | cnn-clf | synthetic | accuracy (higher is better) | 1.000000 | 1.000000 | - | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | complement-nb | istella | accuracy (higher is better) | 0.849360 | 0.849190 | - | sklearn-cpu 0.849350 |
| algos | complement-nb | istella | logloss (lower is better) | 3.762531 | 3.766564 | - | sklearn-cpu 3.174763 |
| algos | complement-nb | taxi | accuracy (higher is better) | 0.678030 | 0.677000 | - | sklearn-cpu 0.678030 |
| algos | complement-nb | taxi | logloss (lower is better) | 0.715492 | 0.715601 | - | sklearn-cpu 0.715493 |
| algos | complement-nb | text | accuracy (higher is better) | 0.983067 | 0.983067 | - | sklearn-cpu 0.983067 |
| algos | complement-nb | text | logloss (lower is better) | 0.559491 | 0.559491 | - | sklearn-cpu 0.557285 |
| algos | connected-components | istella | ari_vs_networkx | 1.000000 | 1.000000 | - | networkx-cpu - |
| algos | connected-components | istella | n_components | 81 | 81 | - | networkx-cpu 81 |
| algos | connected-components | taxi | ari_vs_networkx | 1.000000 | 1.000000 | - | networkx-cpu - |
| algos | connected-components | taxi | n_components | 588 | 588 | - | networkx-cpu 588 |
| algos | conv1d | synthetic | max_rel_diff_vs_torch_eager_fp32 | 0.391155 | 0.391155 | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 3479.164094; torch-compile-bf16 3418.128937 |
| algos | conv1d | synthetic | rel_fro_vs_torch_eager_fp32 | 3.947e-07 | 3.947e-07 | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 0.002870; torch-compile-bf16 0.003296 |
| algos | conv2d | synthetic | max_rel_diff_vs_torch_eager_fp32 | 0.461936 | 0.461936 | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 3463.592380; torch-compile-bf16 3418.426961 |
| algos | conv2d | synthetic | rel_fro_vs_torch_eager_fp32 | 3.303e-07 | 3.303e-07 | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 0.002935; torch-compile-bf16 0.003382 |
| algos | cross-entropy | synthetic | loss_rel_err_vs_fp64 | 4.564e-08 | 4.564e-08 | - | torch-eager-fp32 4.564e-08; torch-compile-fp32 4.564e-08 |
| algos | cross-entropy | synthetic | grad_max_rel_diff_vs_ours | 2.794e-09 | - | - | torch-eager-fp32 5.588e-09; torch-compile-fp32 5.588e-09 |
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
| algos | dict-learning | taxi | relative_reconstruction_error (lower is better) | 0.459169 | 0.459169 | - | sklearn-cpu 0.460601 |
| algos | dynamic-optimized-theta | synthetic | forecast_rmse (lower is better) | 1.435976 | 1.436278 | - | statsforecast-cpu 1.436045 |
| algos | dynamic-optimized-theta | taxi-hourly | forecast_rmse (lower is better) | 49.083557 | 49.086628 | - | statsforecast-cpu 49.314797 |
| algos | dynamic-theta | synthetic | forecast_rmse (lower is better) | 1.437165 | 1.437256 | - | statsforecast-cpu 1.437262 |
| algos | dynamic-theta | taxi-hourly | forecast_rmse (lower is better) | 49.100462 | 49.101249 | - | statsforecast-cpu 49.269843 |
| algos | eigh | synthetic | max_eigenvalue_error | - | - | - | numpy-cpu 3.49e-08; torch-gpu - |
| algos | eigh | synthetic | relative_residual | - | - | - | numpy-cpu 2.824e-08; torch-gpu - |
| algos | elliptic-envelope | istella | fraction_flagged | - | - | - | sklearn-cpu 0.091570 |
| algos | elliptic-envelope | istella | jaccard_vs_sklearn | - | - | - | sklearn-cpu 1.000000 |
| algos | elliptic-envelope | taxi | fraction_flagged | 0.102370 | 0.102370 | - | sklearn-cpu 0.102470 |
| algos | elliptic-envelope | taxi | jaccard_vs_sklearn | 0.963574 | 0.963574 | - | sklearn-cpu 1.000000 |
| algos | embedding | synthetic | max_rel_diff_vs_torch_eager_fp32 | 0.000000 | 0.000000 | - | torch-eager-fp32 -; torch-compile-fp32 0.000000 |
| algos | embedding | synthetic | rel_fro_vs_torch_eager_fp32 | 0.000000 | 0.000000 | - | torch-eager-fp32 -; torch-compile-fp32 0.000000 |
| algos | factor-analysis | istella | mean_log_likelihood (higher is better) | - | - | - | sklearn-cpu 98.122830 |
| algos | factor-analysis | taxi | mean_log_likelihood (higher is better) | -14.823632 | -14.823632 | - | sklearn-cpu -14.823723 |
| algos | fastica | istella | mean_abs_excess_kurtosis | 356.288827 | 356.288855 | - | sklearn-cpu 922.531077 |
| algos | fastica | taxi | mean_abs_excess_kurtosis | 13.205464 | 13.204824 | - | sklearn-cpu 13.764740 |
| algos | garch | synthetic | mean_llf (higher is better) | -1938.224617 | -1938.223490 | - | arch-cpu -1938.221004 |
| algos | garch | taxi-hourly | mean_llf (higher is better) | -1132.954899 | -1132.712756 | - | arch-cpu -1129.806866 |
| algos | gaussian-nb | istella | accuracy (higher is better) | 0.876570 | 0.876530 | - | sklearn-cpu 0.876530 |
| algos | gaussian-nb | istella | logloss (lower is better) | 3.574408 | 3.574225 | - | sklearn-cpu 3.417392 |
| algos | gaussian-nb | taxi | accuracy (higher is better) | 0.719820 | 0.719900 | - | sklearn-cpu 0.719900 |
| algos | gaussian-nb | taxi | logloss (lower is better) | 1.132247 | 1.133898 | - | sklearn-cpu 1.133898 |
| algos | gaussian-rp | istella | mean_abs_distortion | 0.680693 | 0.680693 | - | sklearn-cpu 0.177966 |
| algos | gaussian-rp | taxi | mean_abs_distortion | 0.345752 | 0.345752 | - | sklearn-cpu 0.339791 |
| algos | gcn | istella | max_rel_diff_vs_torch_eager_fp32 | - | - | - | torch-eager-fp32 -; torch-compile-fp32 0.017593; torch-eager-bf16 2744.908399; torch-compile-bf16 2744.915191 |
| algos | gcn | istella | rel_fro_vs_torch_eager_fp32 | - | - | - | torch-eager-fp32 -; torch-compile-fp32 1.004e-07; torch-eager-bf16 0.002187; torch-compile-bf16 0.002187 |
| algos | gcn | taxi | max_rel_diff_vs_torch_eager_fp32 | - | - | - | torch-eager-fp32 -; torch-compile-fp32 0.003725; torch-eager-bf16 1509.509981; torch-compile-bf16 1509.509981 |
| algos | gcn | taxi | rel_fro_vs_torch_eager_fp32 | - | - | - | torch-eager-fp32 -; torch-compile-fp32 9.327e-08; torch-eager-bf16 0.002330; torch-compile-bf16 0.002330 |
| algos | global-avgpool | synthetic | max_rel_diff_vs_torch_eager_fp32 | 0.015661 | 0.015661 | - | torch-eager-fp32 -; torch-compile-fp32 0.007299; torch-eager-bf16 0.000000; torch-compile-bf16 0.007299 |
| algos | global-avgpool | synthetic | rel_fro_vs_torch_eager_fp32 | 1.493e-07 | 1.493e-07 | - | torch-eager-fp32 -; torch-compile-fp32 9.321e-08; torch-eager-bf16 0.000000; torch-compile-bf16 9.321e-08 |
| algos | global-maxpool | synthetic | max_rel_diff_vs_torch_eager_fp32 | 0.000000 | 0.000000 | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | global-maxpool | synthetic | rel_fro_vs_torch_eager_fp32 | 0.000000 | 0.000000 | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | graphsage | istella | max_rel_diff_vs_torch_eager_fp32 | - | - | - | torch-eager-fp32 -; torch-compile-fp32 0.178814; torch-eager-bf16 3907.114267; torch-compile-bf16 3678.172827 |
| algos | graphsage | istella | rel_fro_vs_torch_eager_fp32 | - | - | - | torch-eager-fp32 -; torch-compile-fp32 7.998e-08; torch-eager-bf16 0.003366; torch-compile-bf16 0.003054 |
| algos | graphsage | taxi | max_rel_diff_vs_torch_eager_fp32 | - | - | - | torch-eager-fp32 -; torch-compile-fp32 0.119209; torch-eager-bf16 5460.333333; torch-compile-bf16 4608.154297 |
| algos | graphsage | taxi | rel_fro_vs_torch_eager_fp32 | - | - | - | torch-eager-fp32 -; torch-compile-fp32 5.006e-08; torch-eager-bf16 0.003557; torch-compile-bf16 0.003285 |
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
| algos | isomap | istella | trustworthiness_k15 (higher is better, 1 at most) | 0.853298 | - | - | sklearn-cpu 0.853294 |
| algos | isomap | taxi | trustworthiness_k15 (higher is better, 1 at most) | 0.771828 | - | - | sklearn-cpu 0.771828 |
| algos | iterative-imputer | istella | masked_rmse | 799040.699151 | 799013.020330 | - | sklearn-cpu 802428.936225 |
| algos | iterative-imputer | istella | max_abs_diff_vs_sklearn | 9.974e+06 | 1.011e+07 | - | sklearn-cpu - |
| algos | iterative-imputer | taxi | masked_rmse | 4.693971 | 4.693848 | - | sklearn-cpu 4.693973 |
| algos | iterative-imputer | taxi | max_abs_diff_vs_sklearn | 0.0003719 | 0.044070 | - | sklearn-cpu - |
| algos | ivf-filter | istella | recall_at_10 (higher is better) | 0.652700 | 0.609250 | - | faiss-cpu 0.841950 |
| algos | ivf-filter | taxi | recall_at_10 (higher is better) | 0.982950 | 0.980075 | - | faiss-cpu 0.982050 |
| algos | ivf-pq | istella | recall_at_10 (higher is better) | 0.599475 | 0.550825 | - | faiss-cpu 0.802975 |
| algos | ivf-pq | taxi | recall_at_10 (higher is better) | 0.976450 | 0.973225 | - | faiss-cpu 0.980075 |
| algos | ivf-rabitq | istella | recall_at_10 (higher is better) | 0.132550 | 0.125125 | - | faiss-cpu 0.050450 |
| algos | ivf-rabitq | taxi | recall_at_10 (higher is better) | 0.115775 | 0.110475 | - | faiss-cpu 0.126275 |
| algos | ivf-refine | istella | recall_at_10 (higher is better) | 0.862250 | 0.809175 | - | faiss-cpu 0.993425 |
| algos | ivf-refine | taxi | recall_at_10 (higher is better) | 0.999675 | 0.999675 | - | faiss-cpu 0.999225 |
| algos | ivf-sq | istella | recall_at_10 (higher is better) | 0.628550 | 0.728025 | - | faiss-cpu 0.591300 |
| algos | ivf-sq | taxi | recall_at_10 (higher is better) | 0.902025 | 0.934975 | - | faiss-cpu 0.857025 |
| algos | jl-min-dim | synthetic | equal_fraction_vs_sklearn | 1.000000 | 1.000000 | - | sklearn-cpu 1.000000 |
| algos | kbins | istella | max_abs_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | kbins | istella | rel_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | kbins | taxi | max_abs_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | kbins | taxi | rel_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | kernel-pca | istella | subspace_cos_vs_sklearn | 1.000000 | 1.000000 | - | sklearn-cpu 1.000000 |
| algos | kernel-pca | taxi | subspace_cos_vs_sklearn | 1.000000 | 1.000000 | - | sklearn-cpu 1.000000 |
| algos | kernel-shap | istella | rel_error_vs_exact | 4.179e-09 | 4.179e-09 | - | shap-cpu 1.405e-14 |
| algos | kernel-shap | taxi | rel_error_vs_exact | 1.731e-08 | 1.731e-08 | - | shap-cpu 6.394e-14 |
| algos | knn-imputer | istella | masked_rmse | 323953.237332 | 323953.237332 | - | sklearn-cpu - |
| algos | kpss | synthetic | flag_agreement_vs_statsmodels | 1.000000 | 1.000000 | - | statsmodels-cpu 1.000000 |
| algos | kpss | synthetic | stat_max_rel_diff_vs_statsmodels | 4.062e-05 | 4.318e-05 | - | statsmodels-cpu 0.000000 |
| algos | kpss | synthetic | stationary_fraction | 0.031250 | 0.031250 | - | statsmodels-cpu 0.031250 |
| algos | kpss | taxi-hourly | flag_agreement_vs_statsmodels | 1.000000 | 1.000000 | - | statsmodels-cpu 1.000000 |
| algos | kpss | taxi-hourly | stat_max_rel_diff_vs_statsmodels | 3.946e-05 | 3.946e-05 | - | statsmodels-cpu 0.000000 |
| algos | kpss | taxi-hourly | stationary_fraction | 0.687500 | 0.687500 | - | statsmodels-cpu 0.687500 |
| algos | label-binarizer | istella | max_abs_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | label-binarizer | istella | rel_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | label-binarizer | taxi | max_abs_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | label-binarizer | taxi | rel_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | label-encoder | istella | max_abs_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | label-encoder | istella | rel_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | label-encoder | taxi | max_abs_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | label-encoder | taxi | rel_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | label-propagation | istella | accuracy (higher is better) | - | - | - | sklearn-cpu 0.905500 |
| algos | label-propagation | taxi | accuracy (higher is better) | 0.701600 | 0.701600 | - | sklearn-cpu 0.701600 |
| algos | label-spreading | istella | accuracy (higher is better) | - | - | - | sklearn-cpu 0.904450 |
| algos | label-spreading | taxi | accuracy (higher is better) | 0.676400 | 0.676400 | - | sklearn-cpu 0.676400 |
| algos | layernorm | synthetic | max_rel_diff_vs_torch_eager_fp32 | 0.089719 | 0.089719 | - | torch-eager-fp32 -; torch-compile-fp32 0.066421; torch-eager-bf16 0.000000; torch-compile-bf16 0.066421 |
| algos | layernorm | synthetic | rel_fro_vs_torch_eager_fp32 | 2.264e-07 | 2.264e-07 | - | torch-eager-fp32 -; torch-compile-fp32 2.25e-07; torch-eager-bf16 0.000000; torch-compile-bf16 2.25e-07 |
| algos | lda-clf | istella | accuracy (higher is better) | 0.909010 | 0.911660 | - | sklearn-cpu 0.901130 |
| algos | lda-clf | istella | logloss (lower is better) | 0.264395 | 0.247483 | - | sklearn-cpu 0.449237 |
| algos | lda-clf | taxi | accuracy (higher is better) | 0.762580 | 0.762580 | - | sklearn-cpu 0.762530 |
| algos | lda-clf | taxi | logloss (lower is better) | 0.539743 | 0.539749 | - | sklearn-cpu 0.539767 |
| algos | lda | taxi-zones | perplexity | 45.221810 | 45.221819 | - | sklearn-cpu 44.897929 |
| algos | lda | text | perplexity | - | - | - | sklearn-cpu 266.944425 |
| algos | lle | istella | trustworthiness_k15 (higher is better, 1 at most) | 0.872427 | 0.872427 | - | sklearn-cpu 0.849140 |
| algos | lle | taxi | trustworthiness_k15 (higher is better, 1 at most) | 0.839814 | 0.839814 | - | sklearn-cpu 0.770758 |
| algos | logreg-cv | istella | accuracy (higher is better) | - | - | - | sklearn-cpu 0.924630 |
| algos | logreg-cv | istella | logloss (lower is better) | - | - | - | sklearn-cpu 0.181351 |
| algos | logreg-cv | taxi | accuracy (higher is better) | - | - | - | sklearn-cpu 0.763300 |
| algos | logreg-cv | taxi | logloss (lower is better) | - | - | - | sklearn-cpu 0.538988 |
| algos | louvain | istella | modularity | 0.909755 | 0.909755 | - | networkx-cpu 0.908460 |
| algos | louvain | istella | n_communities | 39 | 39 | - | networkx-cpu 40 |
| algos | louvain | taxi | modularity | 0.941172 | 0.941172 | - | networkx-cpu 0.940781 |
| algos | louvain | taxi | n_communities | 58 | 58 | - | networkx-cpu 56 |
| algos | lr-constant | synthetic | max_rel_diff_vs_ours (0 is our output exactly) | 0.000000 | 0.000000 | - | torch-cpu 1.038e-07 |
| algos | lr-exponential | synthetic | max_rel_diff_vs_ours (0 is our output exactly) | 0.000000 | 0.000000 | - | torch-cpu 5.933e-08 |
| algos | lr-onecycle | synthetic | max_rel_diff_vs_ours (0 is our output exactly) | 0.000000 | 0.000000 | - | torch-cpu 5.951e-08 |
| algos | lr-step | synthetic | max_rel_diff_vs_ours (0 is our output exactly) | 0.000000 | 0.000000 | - | torch-cpu 1.49e-08 |
| algos | lr-warmup-linear | synthetic | max_rel_diff_vs_ours (0 is our output exactly) | 0.000000 | 0.000000 | - | torch-cpu 0.001000 |
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
| algos | lu-factor | synthetic | relative_residual | 3.249e-06 | 3.249e-06 | - | scipy-cpu 3.246e-06; torch-gpu 8.234e-07 |
| algos | lu-solve | synthetic | relative_residual | 3.249e-06 | 3.249e-06 | - | numpy-cpu 3.259e-08; torch-gpu 8.234e-07 |
| algos | maxabs-scaler | istella | max_abs_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | maxabs-scaler | istella | rel_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | maxabs-scaler | taxi | max_abs_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | maxabs-scaler | taxi | rel_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | maxpool1d | synthetic | max_rel_diff_vs_torch_eager_fp32 | 0.000000 | 0.000000 | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | maxpool1d | synthetic | rel_fro_vs_torch_eager_fp32 | 0.000000 | 0.000000 | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | maxpool2d | synthetic | max_rel_diff_vs_torch_eager_fp32 | 0.000000 | 0.000000 | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | maxpool2d | synthetic | rel_fro_vs_torch_eager_fp32 | 0.000000 | 0.000000 | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | mb-dict-learning | istella | component_sparsity | 0.086364 | 0.086364 | - | sklearn-cpu 0.086364 |
| algos | mb-dict-learning | istella | relative_reconstruction_error (lower is better) | 0.648339 | 0.648338 | - | sklearn-cpu 0.644013 |
| algos | mb-dict-learning | taxi | component_sparsity | 0.000000 | 0.000000 | - | sklearn-cpu 0.000000 |
| algos | mb-dict-learning | taxi | relative_reconstruction_error (lower is better) | 0.496684 | 0.496685 | - | sklearn-cpu 0.477049 |
| algos | mb-sparse-pca | istella | component_sparsity | 0.130682 | 0.130682 | - | sklearn-cpu 0.130682 |
| algos | mb-sparse-pca | istella | relative_reconstruction_error (lower is better) | 0.705389 | 0.705389 | - | sklearn-cpu 0.705389 |
| algos | mb-sparse-pca | taxi | component_sparsity | 0.022727 | 0.022727 | - | sklearn-cpu 0.022727 |
| algos | mb-sparse-pca | taxi | relative_reconstruction_error (lower is better) | 0.275935 | 0.275935 | - | sklearn-cpu 0.275933 |
| algos | mds | istella | trustworthiness_k15 (higher is better, 1 at most) | 0.586415 | 0.586415 | - | sklearn-cpu 0.580239 |
| algos | mds | taxi | trustworthiness_k15 (higher is better, 1 at most) | 0.606536 | 0.606536 | - | sklearn-cpu 0.604142 |
| algos | min-cov-det | istella | n_features | - | - | - | sklearn-cpu 220 |
| algos | min-cov-det | taxi | n_features | 11 | 11 | - | sklearn-cpu 11 |
| algos | min-cov-det | taxi | rel_diff_vs_sklearn | 0.205603 | 0.205603 | - | sklearn-cpu - |
| algos | minmax-scaler | istella | max_abs_diff_vs_sklearn | 1.192e-07 | 0.000000 | - | sklearn-cpu - |
| algos | minmax-scaler | istella | rel_diff_vs_sklearn | 1.694e-08 | 0.000000 | - | sklearn-cpu - |
| algos | minmax-scaler | taxi | max_abs_diff_vs_sklearn | 5.96e-08 | 0.000000 | - | sklearn-cpu - |
| algos | minmax-scaler | taxi | rel_diff_vs_sklearn | 2.63e-08 | 0.000000 | - | sklearn-cpu - |
| algos | mlp-clf | istella | accuracy (higher is better) | 0.944560 | 0.944270 | - | sklearn-cpu 0.943760 |
| algos | mlp-clf | istella | logloss (lower is better) | 0.136389 | 0.136955 | - | sklearn-cpu 0.136831 |
| algos | mlp-clf | taxi | accuracy (higher is better) | 0.767750 | 0.767810 | - | sklearn-cpu 0.767830 |
| algos | mlp-clf | taxi | logloss (lower is better) | 0.530490 | 0.530506 | - | sklearn-cpu 0.530444 |
| algos | mlp-reg | istella | r2 (higher is better) | 0.527265 | 0.526405 | - | sklearn-cpu 0.524815 |
| algos | mlp-reg | istella | rmse (lower is better) | 0.574340 | 0.574862 | - | sklearn-cpu 0.575826 |
| algos | mlp-reg | taxi | r2 (higher is better) | 0.931976 | 0.931981 | - | sklearn-cpu 0.929613 |
| algos | mlp-reg | taxi | rmse (lower is better) | 4.154000 | 4.153868 | - | sklearn-cpu 4.225537 |
| algos | moe | synthetic | max_rel_diff_vs_torch_eager_fp32 | 0.062871 | 0.059674 | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 22341.757794; torch-compile-bf16 22341.757794 |
| algos | moe | synthetic | rel_fro_vs_torch_eager_fp32 | 2.845e-07 | 3.083e-07 | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 0.055222; torch-compile-bf16 0.055222 |
| algos | multilabel-binarizer | istella | max_abs_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | multilabel-binarizer | istella | rel_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | multilabel-binarizer | taxi | max_abs_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | multilabel-binarizer | taxi | rel_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | multinomial-nb | istella | accuracy (higher is better) | 0.853620 | 0.853560 | - | sklearn-cpu 0.853620 |
| algos | multinomial-nb | istella | logloss (lower is better) | 3.628575 | 3.630944 | - | sklearn-cpu 3.087499 |
| algos | multinomial-nb | taxi | accuracy (higher is better) | 0.723160 | 0.723260 | - | sklearn-cpu 0.723160 |
| algos | multinomial-nb | taxi | logloss (lower is better) | 0.590725 | 0.589979 | - | sklearn-cpu 0.590725 |
| algos | multinomial-nb | text | accuracy (higher is better) | 0.983067 | 0.983067 | - | sklearn-cpu 0.983067 |
| algos | multinomial-nb | text | logloss (lower is better) | 0.559529 | 0.559529 | - | sklearn-cpu 0.557319 |
| algos | multioutput-clf | istella | accuracy (higher is better) | 0.959175 | 0.959110 | - | sklearn-cpu 0.959225 |
| algos | multioutput-clf | taxi | accuracy (higher is better) | 0.863560 | 0.863560 | - | sklearn-cpu 0.863550 |
| algos | multioutput-reg | istella | r2 (higher is better) | 0.439854 | 0.455327 | - | sklearn-cpu 0.455345 |
| algos | multioutput-reg | taxi | r2 (higher is better) | 0.604241 | 0.604240 | - | sklearn-cpu 0.604316 |
| algos | nadam | synthetic | rel_fro_vs_torch_eager_fp32 | 2.939e-08 | 2.939e-08 | - | torch-eager-fp32 -; torch-compile-fp32 4.44e-08 |
| algos | nmf | istella | relative_reconstruction_error (lower is better) | 0.325174 | 0.325174 | - | sklearn-cpu 0.325402 |
| algos | nmf | taxi | relative_reconstruction_error (lower is better) | 0.091156 | 0.091156 | - | sklearn-cpu 0.091155 |
| algos | normalizer | istella | max_abs_diff_vs_sklearn | 3.576e-07 | 3.576e-07 | - | sklearn-cpu - |
| algos | normalizer | istella | rel_diff_vs_sklearn | 5.674e-08 | 5.811e-08 | - | sklearn-cpu - |
| algos | normalizer | taxi | max_abs_diff_vs_sklearn | 1.192e-07 | 1.192e-07 | - | sklearn-cpu - |
| algos | normalizer | taxi | rel_diff_vs_sklearn | 3.283e-08 | 3.114e-08 | - | sklearn-cpu - |
| algos | onehot | istella | max_abs_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | onehot | istella | rel_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | onehot | taxi | max_abs_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | onehot | taxi | rel_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | optimized-theta | synthetic | forecast_rmse (lower is better) | 1.439709 | 1.438855 | - | statsforecast-cpu 1.437815 |
| algos | optimized-theta | taxi-hourly | forecast_rmse (lower is better) | 49.152331 | 49.150860 | - | statsforecast-cpu 49.356608 |
| algos | ordinal | istella | max_abs_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | ordinal | istella | rel_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | ordinal | taxi | max_abs_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | ordinal | taxi | rel_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | ovr | istella | accuracy (higher is better) | 0.892700 | 0.892670 | - | sklearn-cpu 0.892730 |
| algos | ovr | taxi | accuracy (higher is better) | 0.478940 | 0.478940 | - | sklearn-cpu 0.478890 |
| algos | pagerank | istella | l1_vs_networkx | 6.429e-08 | 5.89e-08 | - | networkx-cpu - |
| algos | pagerank | istella | sum | 1.000000 | 1.000000 | - | networkx-cpu 1.000000 |
| algos | pagerank | taxi | l1_vs_networkx | 8.411e-08 | 6.917e-08 | - | networkx-cpu - |
| algos | pagerank | taxi | sum | 1.000000 | 1.000000 | - | networkx-cpu 1.000000 |
| algos | permutation-shap | istella | rel_error_vs_exact | 5.339e-09 | 5.339e-09 | - | shap-cpu 3.692e-10 |
| algos | permutation-shap | taxi | rel_error_vs_exact | 2.15e-08 | 2.15e-08 | - | shap-cpu 1.279e-15 |
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
| algos | poly-features | istella | max_abs_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | poly-features | istella | rel_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | poly-features | taxi | max_abs_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | poly-features | taxi | rel_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | power-transformer | istella | max_abs_diff_vs_sklearn | 5.248894 | 353.551971 | - | sklearn-cpu - |
| algos | power-transformer | istella | rel_diff_vs_sklearn | 0.108987 | 1.000000 | - | sklearn-cpu - |
| algos | power-transformer | taxi | max_abs_diff_vs_sklearn | 0.062769 | 0.191760 | - | sklearn-cpu - |
| algos | power-transformer | taxi | rel_diff_vs_sklearn | 0.0005187 | 0.025551 | - | sklearn-cpu - |
| algos | prophet | synthetic | forecast_rmse (lower is better) | 1.015207 | 1.015150 | - | prophet-cpu 1.015319 |
| algos | prophet | taxi-hourly | forecast_rmse (lower is better) | 32.033729 | 32.049281 | - | prophet-cpu 32.035281 |
| algos | qda | istella | accuracy (higher is better) | 0.866090 | 0.866090 | - | sklearn-cpu 0.880530 |
| algos | qda | istella | logloss (lower is better) | 4.052902 | 4.052873 | - | sklearn-cpu 3.476976 |
| algos | qda | taxi | accuracy (higher is better) | 0.727020 | 0.727020 | - | sklearn-cpu 0.727220 |
| algos | qda | taxi | logloss (lower is better) | 1.061003 | 1.061003 | - | sklearn-cpu 1.059265 |
| algos | qr | istella | relative_gram_difference | 0.0009027 | 0.0009027 | - | numpy-cpu 2.472e-08; torch-gpu - |
| algos | qr | taxi | relative_gram_difference | 0.001996 | 0.001996 | - | numpy-cpu 3.024e-08; torch-gpu - |
| algos | quantile-transformer | istella | max_abs_diff_vs_sklearn | 5.96e-08 | 5.96e-08 | - | sklearn-cpu - |
| algos | quantile-transformer | istella | rel_diff_vs_sklearn | 2.764e-08 | 2.764e-08 | - | sklearn-cpu - |
| algos | quantile-transformer | taxi | max_abs_diff_vs_sklearn | 5.96e-08 | 5.96e-08 | - | sklearn-cpu - |
| algos | quantile-transformer | taxi | rel_diff_vs_sklearn | 2.295e-08 | 2.294e-08 | - | sklearn-cpu - |
| algos | random-trees-embedding | istella | nonzeros_per_row | 10.000000 | 10.000000 | - | sklearn-cpu 10.000000 |
| algos | random-trees-embedding | istella | output_columns | 209 | 209 | - | sklearn-cpu 251 |
| algos | random-trees-embedding | taxi | nonzeros_per_row | 10.000000 | 10.000000 | - | sklearn-cpu 10.000000 |
| algos | random-trees-embedding | taxi | output_columns | 292 | 292 | - | sklearn-cpu 244 |
| algos | randomized-svd | istella | relative_reconstruction_error (lower is better) | 0.0002359 | 0.0002359 | - | sklearn-cpu 0.000236; torch-gpu - |
| algos | randomized-svd | taxi | relative_reconstruction_error (lower is better) | 0.027197 | 0.027197 | - | sklearn-cpu 0.027197; torch-gpu - |
| algos | resample | istella | max_mean_shift_over_std | 0.003203 | 0.003203 | - | sklearn-cpu 0.002552 |
| algos | resample | taxi | max_mean_shift_over_std | 0.002917 | 0.002917 | - | sklearn-cpu 0.002257 |
| algos | resnet-block | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | - | torch-eager-fp32 -; torch-compile-fp32 0.357628; torch-eager-bf16 18498.390913; torch-compile-bf16 18498.390913 |
| algos | resnet-block | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | - | torch-eager-fp32 -; torch-compile-fp32 1.857e-07; torch-eager-bf16 0.003425; torch-compile-bf16 0.003425 |
| algos | rfe | istella | jaccard_vs_sklearn | 0.880342 | 0.818182 | - | sklearn-cpu 1.000000 |
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
| algos | select-d | synthetic | d_agreement_vs_statsmodels | 1.000000 | 1.000000 | - | statsmodels-cpu 1.000000 |
| algos | select-d | taxi-hourly | d_agreement_vs_statsmodels | 1.000000 | 1.000000 | - | statsmodels-cpu 1.000000 |
| algos | select-f-classif | istella | jaccard_vs_sklearn | 1.000000 | 1.000000 | - | sklearn-cpu 1.000000 |
| algos | select-f-classif | istella | n_selected | 110 | 110 | - | sklearn-cpu 110 |
| algos | select-f-classif | taxi | jaccard_vs_sklearn | 1.000000 | 1.000000 | - | sklearn-cpu 1.000000 |
| algos | select-f-classif | taxi | n_selected | 5 | 5 | - | sklearn-cpu 5 |
| algos | select-f-regression | istella | jaccard_vs_sklearn | 1.000000 | 1.000000 | - | sklearn-cpu 1.000000 |
| algos | select-f-regression | istella | n_selected | 110 | 110 | - | sklearn-cpu 110 |
| algos | select-f-regression | taxi | jaccard_vs_sklearn | 1.000000 | 1.000000 | - | sklearn-cpu 1.000000 |
| algos | select-f-regression | taxi | n_selected | 5 | 5 | - | sklearn-cpu 5 |
| algos | select-mutual-info-reg | istella | jaccard_vs_sklearn | 0.929825 | 0.929825 | - | sklearn-cpu 1.000000 |
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
| algos | simple-imputer | istella | masked_rmse | 346849.129968 | 346849.129968 | - | sklearn-cpu 346849.129968 |
| algos | simple-imputer | istella | max_abs_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | simple-imputer | taxi | masked_rmse | 5.985180 | 5.985180 | - | sklearn-cpu 5.985180 |
| algos | simple-imputer | taxi | max_abs_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
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
| algos | stacking-clf | istella | accuracy (higher is better) | 0.929970 | 0.929960 | - | sklearn-cpu 0.930040 |
| algos | stacking-clf | istella | logloss (lower is better) | 0.193693 | 0.193706 | - | sklearn-cpu 0.193931 |
| algos | stacking-clf | taxi | accuracy (higher is better) | 0.767920 | 0.767920 | - | sklearn-cpu 0.755330 |
| algos | stacking-clf | taxi | logloss (lower is better) | 0.536364 | 0.536361 | - | sklearn-cpu 0.547756 |
| algos | stacking-reg | istella | r2 (higher is better) | - | - | - | sklearn-cpu 0.447644 |
| algos | stacking-reg | istella | rmse (lower is better) | - | - | - | sklearn-cpu 0.620826 |
| algos | stacking-reg | taxi | r2 (higher is better) | - | - | - | sklearn-cpu 0.932532 |
| algos | stacking-reg | taxi | rmse (lower is better) | - | - | - | sklearn-cpu 4.137015 |
| algos | standard-scaler | istella | max_abs_diff_vs_sklearn | 0.002426 | 0.002426 | - | sklearn-cpu - |
| algos | standard-scaler | istella | rel_diff_vs_sklearn | 3.591e-06 | 3.59e-06 | - | sklearn-cpu - |
| algos | standard-scaler | taxi | max_abs_diff_vs_sklearn | 0.000103 | 0.000103 | - | sklearn-cpu - |
| algos | standard-scaler | taxi | rel_diff_vs_sklearn | 3.84e-06 | 3.846e-06 | - | sklearn-cpu - |
| algos | stl | synthetic | rel_diff_vs_statsmodels | 6.535e-07 | 6.508e-07 | - | statsmodels-cpu - |
| algos | stl | synthetic | residual_std | 0.783175 | 0.783175 | - | statsmodels-cpu 0.783175 |
| algos | stl | taxi-hourly | rel_diff_vs_statsmodels | 1.258e-06 | 1.212e-06 | - | statsmodels-cpu - |
| algos | stl | taxi-hourly | residual_std | 18.312476 | 18.312475 | - | statsmodels-cpu 18.312475 |
| algos | svd | istella | max_rel_singular_value_error | 28579.906372 | 28579.906372 | - | numpy-cpu 37.354529; torch-gpu 4.945e+06 |
| algos | svd | istella | relative_reconstruction_error_100k_rows | 0.0005441 | 0.0005441 | - | numpy-cpu 4.1e-08; torch-gpu 0.0005905 |
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
| algos | tree-shap | istella | max_additivity_error | 1.099e-06 | 1.099e-06 | - | shap-cpu 1.862e-06; xgboost-cpu 1.862e-06; lightgbm-cpu 4.441e-15 |
| algos | tree-shap | taxi | max_additivity_error | 3.25e-05 | 3.25e-05 | - | shap-cpu 0.0001199; xgboost-cpu 0.0001199; lightgbm-cpu 5.684e-13 |
| algos | tsne | istella | trustworthiness_k15 (higher is better, 1 at most) | 0.992039 | 0.991881 | - | sklearn-cpu 0.991970 |
| algos | tsne | taxi | trustworthiness_k15 (higher is better, 1 at most) | 0.998869 | 0.998927 | - | sklearn-cpu 0.998860 |
| algos | var | synthetic | forecast_rmse (lower is better) | 1.140787 | 1.140864 | - | statsmodels-cpu 1.144945 |
| algos | var | taxi-hourly | forecast_rmse (lower is better) | 33.167955 | 33.167951 | - | statsmodels-cpu 33.167986 |
| algos | variance-threshold | istella | max_abs_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | variance-threshold | istella | rel_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | variance-threshold | taxi | max_abs_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | variance-threshold | taxi | rel_diff_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu - |
| algos | voting-clf | istella | accuracy (higher is better) | 0.918360 | 0.918320 | - | sklearn-cpu 0.918480 |
| algos | voting-clf | istella | logloss (lower is better) | 0.187737 | 0.187682 | - | sklearn-cpu 0.187541 |
| algos | voting-clf | taxi | accuracy (higher is better) | 0.742290 | 0.742320 | - | sklearn-cpu 0.742350 |
| algos | voting-clf | taxi | logloss (lower is better) | 0.554795 | 0.554799 | - | sklearn-cpu 0.554650 |
| algos | voting-reg | istella | r2 (higher is better) | - | - | - | sklearn-cpu 0.406702 |
| algos | voting-reg | istella | rmse (lower is better) | - | - | - | sklearn-cpu 0.643423 |
| algos | voting-reg | taxi | r2 (higher is better) | - | - | - | sklearn-cpu 0.924635 |
| algos | voting-reg | taxi | rmse (lower is better) | - | - | - | sklearn-cpu 4.372421 |
| classical | hdbscan | istella | n_clusters | 47 | 47 | - | sklearn-cpu - |
| classical | hdbscan | istella | noise_fraction | 0.253810 | 0.253810 | - | sklearn-cpu - |
| classical | hdbscan | istella | rows | 100000 | 100000 | - | sklearn-cpu - |
| classical | hdbscan | istella | ari_vs_ours (1 is our partition exactly) | 1.000000 | - | - | sklearn-cpu - |
| classical | hdbscan | istella | noise_agreement_vs_ours | 1.000000 | - | - | sklearn-cpu - |
| classical | hdbscan | taxi | n_clusters | 160 | 161 | - | sklearn-cpu 160 |
| classical | hdbscan | taxi | noise_fraction | 0.142220 | 0.140550 | - | sklearn-cpu 0.142240 |
| classical | hdbscan | taxi | rows | 100000 | 100000 | - | sklearn-cpu 100000 |
| classical | hdbscan | taxi | ari_vs_ours (1 is our partition exactly) | 0.991364 | - | - | sklearn-cpu 0.991251 |
| classical | hdbscan | taxi | noise_agreement_vs_ours | 0.998310 | - | - | sklearn-cpu 0.998290 |
| classical2 | gmm | taxi | bic (lower is better) | - | -3.67e+06 | - | sklearn-cpu - |
| classical2 | gmm | taxi | mean_log_likelihood (higher is better) | - | 12.861940 | - | sklearn-cpu - |
| classical2 | gmm | taxi | n_iter | - | 32 | - | sklearn-cpu - |
| classical2 | nystroem | istella | kernel_rel_error (lower is better) | 0.033431 | 0.033430 | - | sklearn-cpu 0.038958 |
| classical2 | nystroem | taxi | kernel_rel_error (lower is better) | 0.045612 | 0.045611 | - | sklearn-cpu 0.044370 |

## Inference at a glance

Batch prediction, each arm with its own fitted model from the same race; medians in ms. `FAST = IDENTICAL bits` compares our two tiers' predictions on the same rows; `CPU = IDENTICAL bits` compares our CPU tier's with our GPU IDENTICAL arm's.

| family | lane | dataset | batch | rows | ours FAST ms | ours IDENTICAL ms | FAST = IDENTICAL bits | ours CPU ms | CPU = IDENTICAL bits | opponents |
|---|---|---|---|---|---|---|---|---|---|---|
| algos | adaboost-clf | istella | Xq | - | 1247.3 | 922.5 | - | - | - | sklearn-cpu - ms (IDENTICAL/arm -) |
| algos | adaboost-clf | taxi | Xq | - | 754.4 | 371.0 | - | - | - | sklearn-cpu 163.4 ms (IDENTICAL/arm 2.271) |
| algos | adaboost-reg | istella | Xq | - | 121.7 | 158.3 | - | - | - | sklearn-cpu 87.3 ms (IDENTICAL/arm 1.814) |
| algos | adaboost-reg | taxi | Xq | - | 66.0 | 66.4 | - | - | - | sklearn-cpu 38.2 ms (IDENTICAL/arm 1.740) |
| algos | avgpool1d | synthetic | Xq | - | 15.4 | 15.8 | - | - | - | torch-eager-fp32 1.7 ms (IDENTICAL/arm 9.427); torch-compile-fp32 1.0 ms (IDENTICAL/arm 15.194); torch-eager-bf16 1.0 ms (IDENTICAL/arm 16.416); torch-compile-bf16 1.2 ms (IDENTICAL/arm 13.651) |
| algos | avgpool2d | synthetic | Xq | - | 28.8 | 27.6 | - | - | - | torch-eager-fp32 0.8 ms (IDENTICAL/arm 34.565); torch-compile-fp32 0.8 ms (IDENTICAL/arm 35.645); torch-eager-bf16 0.8 ms (IDENTICAL/arm 35.275); torch-compile-bf16 0.8 ms (IDENTICAL/arm 36.027) |
| algos | bagging-clf | istella | Xq | - | 115.6 | 114.4 | - | - | - | sklearn-cpu 940.7 ms (IDENTICAL/arm 0.122) |
| algos | bagging-clf | taxi | Xq | - | 33.5 | 30.0 | - | - | - | sklearn-cpu 558.6 ms (IDENTICAL/arm 0.054) |
| algos | bagging-reg | istella | Xq | - | 82.1 | 81.4 | - | - | - | sklearn-cpu 495.4 ms (IDENTICAL/arm 0.164) |
| algos | bagging-reg | taxi | Xq | - | 15.7 | 15.0 | - | - | - | sklearn-cpu 212.2 ms (IDENTICAL/arm 0.071) |
| algos | batchnorm1d | synthetic | Xq | - | 32.9 | 34.1 | - | - | - | torch-eager-fp32 0.9 ms (IDENTICAL/arm 37.042); torch-compile-fp32 2.6 ms (IDENTICAL/arm 13.128); torch-eager-bf16 1.4 ms (IDENTICAL/arm 24.596); torch-compile-bf16 3.0 ms (IDENTICAL/arm 11.290) |
| algos | batchnorm2d | synthetic | Xq | - | 28.0 | 26.5 | - | - | - | torch-eager-fp32 1.4 ms (IDENTICAL/arm 18.512); torch-compile-fp32 2.6 ms (IDENTICAL/arm 10.016); torch-eager-bf16 0.9 ms (IDENTICAL/arm 29.952); torch-compile-bf16 2.6 ms (IDENTICAL/arm 10.090) |
| algos | bernoulli-nb | istella | Xq | - | 116.9 | 118.6 | - | - | - | sklearn-cpu 185.5 ms (IDENTICAL/arm 0.639) |
| algos | bernoulli-nb | taxi | Xq | - | 10.3 | 7.7 | - | - | - | sklearn-cpu 20.4 ms (IDENTICAL/arm 0.379) |
| algos | binarizer | istella | Xq | - | 55.6 | 57.1 | - | - | - | sklearn-cpu 70.2 ms (IDENTICAL/arm 0.814) |
| algos | binarizer | taxi | Xq | - | 3.5 | 3.4 | - | - | - | sklearn-cpu 5.1 ms (IDENTICAL/arm 0.668) |
| algos | cagra | istella | Xq | - | - | - | - | - | - | faiss-cpu 9.8 ms (IDENTICAL/arm -) |
| algos | cagra | taxi | Xq | - | - | - | - | - | - | faiss-cpu 3.9 ms (IDENTICAL/arm -) |
| algos | calibrated | istella | Xq | - | 159.8 | 163.8 | - | - | - | sklearn-cpu 988.9 ms (IDENTICAL/arm 0.166) |
| algos | calibrated | taxi | Xq | - | 65.0 | 60.7 | - | - | - | sklearn-cpu 81.5 ms (IDENTICAL/arm 0.745) |
| algos | categorical-nb | istella | Xq | - | 6.7 | 26.3 | - | - | - | sklearn-cpu 15.6 ms (IDENTICAL/arm 1.684) |
| algos | categorical-nb | taxi | Xq | - | 6.4 | 25.9 | - | - | - | sklearn-cpu 12.3 ms (IDENTICAL/arm 2.107) |
| algos | cca | istella | Xq | - | 43.7 | 43.8 | - | - | - | sklearn-cpu 57.5 ms (IDENTICAL/arm 0.762) |
| algos | cca | taxi | Xq | - | 5.8 | 11.0 | - | - | - | sklearn-cpu 7.2 ms (IDENTICAL/arm 1.529) |
| algos | cnn-clf | synthetic | Xq | - | 25.3 | 43.5 | - | - | - | torch-eager-fp32 - ms (IDENTICAL/arm -); torch-compile-fp32 - ms (IDENTICAL/arm -); torch-eager-bf16 - ms (IDENTICAL/arm -); torch-compile-bf16 - ms (IDENTICAL/arm -) |
| algos | complement-nb | istella | Xq | - | 35.3 | 33.5 | - | - | - | sklearn-cpu 47.1 ms (IDENTICAL/arm 0.712) |
| algos | complement-nb | taxi | Xq | - | 5.1 | 4.9 | - | - | - | sklearn-cpu 9.6 ms (IDENTICAL/arm 0.508) |
| algos | complement-nb | text | Xq | - | 43.4 | 51.3 | - | - | - | sklearn-cpu 66.7 ms (IDENTICAL/arm 0.769) |
| algos | conv1d | synthetic | Xq | - | 64.1 | 66.4 | - | - | - | torch-eager-fp32 2.5 ms (IDENTICAL/arm 26.632); torch-compile-fp32 3.0 ms (IDENTICAL/arm 21.930); torch-eager-bf16 2.2 ms (IDENTICAL/arm 30.417); torch-compile-bf16 3.2 ms (IDENTICAL/arm 20.916) |
| algos | conv2d | synthetic | Xq | - | 26.3 | 26.7 | - | - | - | torch-eager-fp32 1.5 ms (IDENTICAL/arm 17.555); torch-compile-fp32 2.3 ms (IDENTICAL/arm 11.430); torch-eager-bf16 1.5 ms (IDENTICAL/arm 18.203); torch-compile-bf16 1.7 ms (IDENTICAL/arm 15.708) |
| algos | dart-reg | istella | Xq | - | 130.3 | 114.5 | - | - | - | lightgbm-cpu 64.1 ms (IDENTICAL/arm 1.787); xgboost-cpu - ms (IDENTICAL/arm -) |
| algos | dart-reg | taxi | Xq | - | 144.3 | 178.1 | - | - | - | lightgbm-cpu 57.3 ms (IDENTICAL/arm 3.107); xgboost-cpu 114.4 ms (IDENTICAL/arm 1.557) |
| algos | dart | istella | Xq | - | 246.6 | 228.0 | - | - | - | lightgbm-cpu 57.5 ms (IDENTICAL/arm 3.969); xgboost-cpu - ms (IDENTICAL/arm -) |
| algos | dart | taxi | Xq | - | 278.3 | 272.4 | - | - | - | lightgbm-cpu 51.5 ms (IDENTICAL/arm 5.289); xgboost-cpu 113.4 ms (IDENTICAL/arm 2.403) |
| algos | decision-tree-clf | istella | Xq | - | 11.8 | 11.6 | - | - | - | sklearn-cpu 18.0 ms (IDENTICAL/arm 0.646) |
| algos | decision-tree-clf | taxi | Xq | - | 3.4 | 3.6 | - | - | - | sklearn-cpu 11.1 ms (IDENTICAL/arm 0.327) |
| algos | decision-tree-reg | istella | Xq | - | 9.0 | 8.9 | - | - | - | sklearn-cpu 8.8 ms (IDENTICAL/arm 1.018) |
| algos | decision-tree-reg | taxi | Xq | - | 2.0 | 2.5 | - | - | - | sklearn-cpu 9.3 ms (IDENTICAL/arm 0.267) |
| algos | dict-learning | istella | Xq | - | 96.7 | 139.6 | - | - | - | sklearn-cpu 44.0 ms (IDENTICAL/arm 3.174) |
| algos | dict-learning | taxi | Xq | - | 88.7 | 89.4 | - | - | - | sklearn-cpu 40.2 ms (IDENTICAL/arm 2.225) |
| algos | dropout2d | synthetic | Xq | - | 52.2 | 52.4 | - | - | - | torch-eager-fp32 1.4 ms (IDENTICAL/arm 36.343); torch-compile-fp32 0.6 ms (IDENTICAL/arm 92.050) |
| algos | elliptic-envelope | istella | Xq | - | - | - | - | - | - | sklearn-cpu 3250.4 ms (IDENTICAL/arm -) |
| algos | elliptic-envelope | taxi | Xq | - | 5.8 | 5.7 | - | - | - | sklearn-cpu 5.0 ms (IDENTICAL/arm 1.134) |
| algos | embedding | synthetic | Xq | - | 337.7 | 337.5 | - | - | - | torch-eager-fp32 1.3 ms (IDENTICAL/arm 255.835); torch-compile-fp32 1.4 ms (IDENTICAL/arm 235.616) |
| algos | factor-analysis | istella | Xq | - | - | - | - | - | - | sklearn-cpu 30.7 ms (IDENTICAL/arm -) |
| algos | factor-analysis | taxi | Xq | - | 4.6 | 4.6 | - | - | - | sklearn-cpu 3.7 ms (IDENTICAL/arm 1.263) |
| algos | fastica | istella | Xq | - | 25.1 | 32.8 | - | - | - | sklearn-cpu 11.0 ms (IDENTICAL/arm 2.978) |
| algos | fastica | taxi | Xq | - | 3.4 | 3.4 | - | - | - | sklearn-cpu 2.1 ms (IDENTICAL/arm 1.614) |
| algos | gaussian-nb | istella | Xq | - | 31.0 | 31.3 | - | - | - | sklearn-cpu 183.8 ms (IDENTICAL/arm 0.170) |
| algos | gaussian-nb | taxi | Xq | - | 4.7 | 4.8 | - | - | - | sklearn-cpu 12.2 ms (IDENTICAL/arm 0.390) |
| algos | gaussian-rp | istella | Xq | - | 22.6 | 23.4 | - | - | - | sklearn-cpu 4.0 ms (IDENTICAL/arm 5.895) |
| algos | gaussian-rp | taxi | Xq | - | 3.1 | 3.1 | - | - | - | sklearn-cpu 1.3 ms (IDENTICAL/arm 2.422) |
| algos | gcn | istella | Xq | - | 98.1 | 96.9 | - | - | - | torch-eager-fp32 31.4 ms (IDENTICAL/arm 3.081); torch-compile-fp32 4.9 ms (IDENTICAL/arm 19.574); torch-eager-bf16 31.8 ms (IDENTICAL/arm 3.050); torch-compile-bf16 4.8 ms (IDENTICAL/arm 20.092) |
| algos | gcn | taxi | Xq | - | 88.5 | 88.7 | - | - | - | torch-eager-fp32 26.0 ms (IDENTICAL/arm 3.406); torch-compile-fp32 3.9 ms (IDENTICAL/arm 22.562); torch-eager-bf16 26.2 ms (IDENTICAL/arm 3.383); torch-compile-bf16 3.7 ms (IDENTICAL/arm 24.043) |
| algos | global-avgpool | synthetic | Xq | - | 0.5 | 0.5 | - | - | - | torch-eager-fp32 1.0 ms (IDENTICAL/arm 0.513); torch-compile-fp32 0.4 ms (IDENTICAL/arm 1.173); torch-eager-bf16 1.0 ms (IDENTICAL/arm 0.519); torch-compile-bf16 0.4 ms (IDENTICAL/arm 1.298) |
| algos | global-maxpool | synthetic | Xq | - | 0.6 | 0.5 | - | - | - | torch-eager-fp32 0.3 ms (IDENTICAL/arm 1.558); torch-compile-fp32 0.4 ms (IDENTICAL/arm 1.388); torch-eager-bf16 0.3 ms (IDENTICAL/arm 1.763); torch-compile-bf16 0.4 ms (IDENTICAL/arm 1.336) |
| algos | graphsage | istella | Xq | - | 140.3 | 141.7 | - | - | - | torch-eager-fp32 44.4 ms (IDENTICAL/arm 3.191); torch-compile-fp32 8.5 ms (IDENTICAL/arm 16.631); torch-eager-bf16 44.2 ms (IDENTICAL/arm 3.205); torch-compile-bf16 8.0 ms (IDENTICAL/arm 17.764) |
| algos | graphsage | taxi | Xq | - | 85.6 | 88.4 | - | - | - | torch-eager-fp32 5.1 ms (IDENTICAL/arm 17.403); torch-compile-fp32 5.2 ms (IDENTICAL/arm 16.966); torch-eager-bf16 5.8 ms (IDENTICAL/arm 15.218); torch-compile-bf16 3.5 ms (IDENTICAL/arm 25.371) |
| algos | gru-clf | synthetic | Xq | - | 82.5 | 88.0 | - | - | - | torch-eager-fp32 - ms (IDENTICAL/arm -); torch-compile-fp32 - ms (IDENTICAL/arm -); torch-eager-bf16 - ms (IDENTICAL/arm -); torch-compile-bf16 - ms (IDENTICAL/arm -) |
| algos | gru-clf | taxi-hourly | Xq | - | 83.0 | 88.8 | - | - | - | torch-eager-fp32 - ms (IDENTICAL/arm -); torch-compile-fp32 - ms (IDENTICAL/arm -); torch-eager-bf16 - ms (IDENTICAL/arm -); torch-compile-bf16 - ms (IDENTICAL/arm -) |
| algos | gru-reg | synthetic | Xq | - | 41.0 | 43.3 | - | - | - | torch-eager-fp32 - ms (IDENTICAL/arm -); torch-compile-fp32 - ms (IDENTICAL/arm -); torch-eager-bf16 - ms (IDENTICAL/arm -); torch-compile-bf16 - ms (IDENTICAL/arm -) |
| algos | gru-reg | taxi-hourly | Xq | - | 41.0 | 43.6 | - | - | - | torch-eager-fp32 - ms (IDENTICAL/arm -); torch-compile-fp32 - ms (IDENTICAL/arm -); torch-eager-bf16 - ms (IDENTICAL/arm -); torch-compile-bf16 - ms (IDENTICAL/arm -) |
| algos | incremental-pca | istella | Xq | - | 21.9 | 24.9 | - | - | - | sklearn-cpu 8.2 ms (IDENTICAL/arm 3.027) |
| algos | incremental-pca | taxi | Xq | - | 3.5 | 3.8 | - | - | - | sklearn-cpu 2.5 ms (IDENTICAL/arm 1.509) |
| algos | iterative-imputer | istella | Xq | - | 10.5 | 10.2 | - | - | - | sklearn-cpu 68.8 ms (IDENTICAL/arm 0.148) |
| algos | iterative-imputer | taxi | Xq | - | 5.7 | 5.6 | - | - | - | sklearn-cpu 13.1 ms (IDENTICAL/arm 0.429) |
| algos | ivf-filter | istella | Xq | - | 1148.3 | 1278.0 | - | - | - | faiss-cpu 88.7 ms (IDENTICAL/arm 14.409) |
| algos | ivf-filter | taxi | Xq | - | 59.9 | 60.9 | - | - | - | faiss-cpu 48.2 ms (IDENTICAL/arm 1.263) |
| algos | ivf-pq | istella | Xq | - | 1177.3 | 1281.0 | - | - | - | faiss-cpu 61.7 ms (IDENTICAL/arm 20.776) |
| algos | ivf-pq | taxi | Xq | - | 53.1 | 53.3 | - | - | - | faiss-cpu 39.9 ms (IDENTICAL/arm 1.337) |
| algos | ivf-rabitq | istella | Xq | - | 390.9 | 415.5 | - | - | - | faiss-cpu 60.8 ms (IDENTICAL/arm 6.836) |
| algos | ivf-rabitq | taxi | Xq | - | 37.2 | 38.3 | - | - | - | faiss-cpu 42.8 ms (IDENTICAL/arm 0.894) |
| algos | ivf-refine | istella | Xq | - | 1281.8 | 1402.4 | - | - | - | faiss-cpu 64.1 ms (IDENTICAL/arm 21.895) |
| algos | ivf-refine | taxi | Xq | - | 80.8 | 81.4 | - | - | - | faiss-cpu 40.4 ms (IDENTICAL/arm 2.016) |
| algos | ivf-sq | istella | Xq | - | 777.2 | 830.8 | - | - | - | faiss-cpu 358.7 ms (IDENTICAL/arm 2.316) |
| algos | ivf-sq | taxi | Xq | - | 38.3 | 39.3 | - | - | - | faiss-cpu 42.8 ms (IDENTICAL/arm 0.918) |
| algos | kbins | istella | Xq | - | 62.0 | 64.6 | - | - | - | sklearn-cpu 154.6 ms (IDENTICAL/arm 0.418) |
| algos | kbins | taxi | Xq | - | 3.1 | 3.1 | - | - | - | sklearn-cpu 7.1 ms (IDENTICAL/arm 0.443) |
| algos | kernel-pca | istella | Xq | - | 348.0 | 316.8 | - | - | - | sklearn-cpu 167.5 ms (IDENTICAL/arm 1.892) |
| algos | kernel-pca | taxi | Xq | - | 156.8 | 143.5 | - | - | - | sklearn-cpu 166.5 ms (IDENTICAL/arm 0.862) |
| algos | knn-imputer | istella | Xq | - | 121065.9 | 120540.9 | - | - | - | sklearn-cpu - ms (IDENTICAL/arm -) |
| algos | label-binarizer | istella | Xq | - | 17.3 | 16.9 | - | - | - | sklearn-cpu 2.3 ms (IDENTICAL/arm 7.224) |
| algos | label-binarizer | taxi | Xq | - | 73.4 | 72.0 | - | - | - | sklearn-cpu 4.0 ms (IDENTICAL/arm 18.142) |
| algos | label-encoder | istella | Xq | - | 15.4 | 15.4 | - | - | - | sklearn-cpu 1.1 ms (IDENTICAL/arm 14.629) |
| algos | label-encoder | taxi | Xq | - | 16.3 | 16.0 | - | - | - | sklearn-cpu 1.5 ms (IDENTICAL/arm 10.886) |
| algos | label-propagation | istella | Xq | - | - | - | - | - | - | sklearn-cpu 1268.4 ms (IDENTICAL/arm -) |
| algos | label-propagation | taxi | Xq | - | 333.1 | 358.7 | - | - | - | sklearn-cpu 1004.6 ms (IDENTICAL/arm -) |
| algos | label-spreading | istella | Xq | - | - | - | - | - | - | sklearn-cpu 1277.1 ms (IDENTICAL/arm -) |
| algos | label-spreading | taxi | Xq | - | 877.0 | 1193.1 | - | - | - | sklearn-cpu 959.8 ms (IDENTICAL/arm -) |
| algos | layernorm | synthetic | Xq | - | 23.3 | 25.8 | - | - | - | torch-eager-fp32 0.5 ms (IDENTICAL/arm 52.772); torch-compile-fp32 48.2 ms (IDENTICAL/arm 0.536); torch-eager-bf16 0.5 ms (IDENTICAL/arm 56.963); torch-compile-bf16 48.2 ms (IDENTICAL/arm 0.536) |
| algos | lda-clf | istella | Xq | - | 39.5 | 39.4 | - | - | - | sklearn-cpu 8.3 ms (IDENTICAL/arm 4.722) |
| algos | lda-clf | taxi | Xq | - | 4.8 | 4.9 | - | - | - | sklearn-cpu 1.7 ms (IDENTICAL/arm 2.873) |
| algos | lda | taxi-zones | Xq | - | 642.6 | 619.2 | - | - | - | sklearn-cpu 255.3 ms (IDENTICAL/arm 2.426) |
| algos | lda | text | Xq | - | - | - | - | - | - | sklearn-cpu 631.7 ms (IDENTICAL/arm -) |
| algos | logreg-cv | istella | Xq | - | - | - | - | - | - | sklearn-cpu 24.3 ms (IDENTICAL/arm -) |
| algos | logreg-cv | taxi | Xq | - | - | - | - | - | - | sklearn-cpu 2.7 ms (IDENTICAL/arm -) |
| algos | lstm-clf | synthetic | Xq | - | 104.9 | 112.1 | - | - | - | torch-eager-fp32 - ms (IDENTICAL/arm -); torch-compile-fp32 - ms (IDENTICAL/arm -); torch-eager-bf16 - ms (IDENTICAL/arm -); torch-compile-bf16 - ms (IDENTICAL/arm -) |
| algos | lstm-clf | taxi-hourly | Xq | - | 108.9 | 111.5 | - | - | - | torch-eager-fp32 - ms (IDENTICAL/arm -); torch-compile-fp32 - ms (IDENTICAL/arm -); torch-eager-bf16 - ms (IDENTICAL/arm -); torch-compile-bf16 - ms (IDENTICAL/arm -) |
| algos | lstm-reg | synthetic | Xq | - | 51.7 | 55.3 | - | - | - | torch-eager-fp32 - ms (IDENTICAL/arm -); torch-compile-fp32 - ms (IDENTICAL/arm -); torch-eager-bf16 - ms (IDENTICAL/arm -); torch-compile-bf16 - ms (IDENTICAL/arm -) |
| algos | lstm-reg | taxi-hourly | Xq | - | 52.4 | 55.1 | - | - | - | torch-eager-fp32 - ms (IDENTICAL/arm -); torch-compile-fp32 - ms (IDENTICAL/arm -); torch-eager-bf16 - ms (IDENTICAL/arm -); torch-compile-bf16 - ms (IDENTICAL/arm -) |
| algos | maxabs-scaler | istella | Xq | - | 59.3 | 69.0 | - | - | - | sklearn-cpu 6.9 ms (IDENTICAL/arm 10.050) |
| algos | maxabs-scaler | taxi | Xq | - | 2.6 | 3.5 | - | - | - | sklearn-cpu 1.1 ms (IDENTICAL/arm 3.253) |
| algos | maxpool1d | synthetic | Xq | - | 28.2 | 28.4 | - | - | - | torch-eager-fp32 1.3 ms (IDENTICAL/arm 21.899); torch-compile-fp32 1.1 ms (IDENTICAL/arm 26.519); torch-eager-bf16 0.7 ms (IDENTICAL/arm 39.648); torch-compile-bf16 1.1 ms (IDENTICAL/arm 24.780) |
| algos | maxpool2d | synthetic | Xq | - | 50.2 | 50.7 | - | - | - | torch-eager-fp32 1.4 ms (IDENTICAL/arm 36.254); torch-compile-fp32 1.3 ms (IDENTICAL/arm 38.987); torch-eager-bf16 0.7 ms (IDENTICAL/arm 69.014); torch-compile-bf16 0.7 ms (IDENTICAL/arm 71.104) |
| algos | mb-dict-learning | istella | Xq | - | 143.3 | 50.0 | - | - | - | sklearn-cpu 32.9 ms (IDENTICAL/arm 1.523) |
| algos | mb-dict-learning | taxi | Xq | - | 35.1 | 32.5 | - | - | - | sklearn-cpu 50.0 ms (IDENTICAL/arm 0.650) |
| algos | mb-sparse-pca | istella | Xq | - | 10.6 | 10.7 | - | - | - | sklearn-cpu 8.5 ms (IDENTICAL/arm 1.263) |
| algos | mb-sparse-pca | taxi | Xq | - | 9.8 | 9.1 | - | - | - | sklearn-cpu 2.4 ms (IDENTICAL/arm 3.750) |
| algos | minmax-scaler | istella | Xq | - | 72.7 | 70.4 | - | - | - | sklearn-cpu 9.6 ms (IDENTICAL/arm 7.348) |
| algos | minmax-scaler | taxi | Xq | - | 4.1 | 4.0 | - | - | - | sklearn-cpu 1.7 ms (IDENTICAL/arm 2.305) |
| algos | mlp-clf | istella | Xq | - | 169.9 | 177.7 | - | - | - | sklearn-cpu 75.2 ms (IDENTICAL/arm 2.364) |
| algos | mlp-clf | taxi | Xq | - | 83.3 | 86.0 | - | - | - | sklearn-cpu 67.3 ms (IDENTICAL/arm 1.277) |
| algos | mlp-reg | istella | Xq | - | 87.9 | 90.2 | - | - | - | sklearn-cpu 38.5 ms (IDENTICAL/arm 2.341) |
| algos | mlp-reg | taxi | Xq | - | 42.1 | 44.1 | - | - | - | sklearn-cpu 35.1 ms (IDENTICAL/arm 1.254) |
| algos | moe | synthetic | Xq | - | 963.8 | 964.9 | - | - | - | torch-eager-fp32 28.5 ms (IDENTICAL/arm 33.905); torch-compile-fp32 28.1 ms (IDENTICAL/arm 34.322); torch-eager-bf16 27.0 ms (IDENTICAL/arm 35.776); torch-compile-bf16 27.5 ms (IDENTICAL/arm 35.135) |
| algos | multilabel-binarizer | istella | Xq | - | 36.4 | 36.3 | - | - | - | sklearn-cpu 16.0 ms (IDENTICAL/arm 2.260) |
| algos | multilabel-binarizer | taxi | Xq | - | 36.1 | 34.3 | - | - | - | sklearn-cpu 11.5 ms (IDENTICAL/arm 2.989) |
| algos | multinomial-nb | istella | Xq | - | 31.0 | 30.2 | - | - | - | sklearn-cpu 44.7 ms (IDENTICAL/arm 0.675) |
| algos | multinomial-nb | taxi | Xq | - | 6.8 | 4.6 | - | - | - | sklearn-cpu 10.1 ms (IDENTICAL/arm 0.461) |
| algos | multinomial-nb | text | Xq | - | 51.5 | 50.9 | - | - | - | sklearn-cpu 65.5 ms (IDENTICAL/arm 0.777) |
| algos | multioutput-clf | istella | Xq | - | 44.1 | 42.2 | - | - | - | sklearn-cpu 22.8 ms (IDENTICAL/arm 1.847) |
| algos | multioutput-clf | taxi | Xq | - | 23.3 | 21.3 | - | - | - | sklearn-cpu 3.4 ms (IDENTICAL/arm 6.332) |
| algos | multioutput-reg | istella | Xq | - | 38.9 | 21.7 | - | - | - | sklearn-cpu 7.5 ms (IDENTICAL/arm 2.899) |
| algos | multioutput-reg | taxi | Xq | - | 5.9 | 6.3 | - | - | - | sklearn-cpu 0.6 ms (IDENTICAL/arm 10.065) |
| algos | nmf | istella | Xq | - | 175.5 | 172.9 | - | - | - | sklearn-cpu 95.4 ms (IDENTICAL/arm 1.814) |
| algos | nmf | taxi | Xq | - | 30.8 | 32.3 | - | - | - | sklearn-cpu 59.4 ms (IDENTICAL/arm 0.543) |
| algos | normalizer | istella | Xq | - | 54.6 | 55.9 | - | - | - | sklearn-cpu 11.6 ms (IDENTICAL/arm 4.805) |
| algos | normalizer | taxi | Xq | - | 3.4 | 3.9 | - | - | - | sklearn-cpu 1.6 ms (IDENTICAL/arm 2.528) |
| algos | onehot | istella | Xq | - | 22.7 | 21.6 | - | - | - | sklearn-cpu 25.1 ms (IDENTICAL/arm 0.863) |
| algos | onehot | taxi | Xq | - | 109.6 | 110.8 | - | - | - | sklearn-cpu 30.0 ms (IDENTICAL/arm 3.690) |
| algos | ordinal | istella | Xq | - | 7.9 | 8.0 | - | - | - | sklearn-cpu 18.1 ms (IDENTICAL/arm 0.439) |
| algos | ordinal | taxi | Xq | - | 7.1 | 7.2 | - | - | - | sklearn-cpu 10.4 ms (IDENTICAL/arm 0.695) |
| algos | ovr | istella | Xq | - | 111.5 | 110.8 | - | - | - | sklearn-cpu 87.2 ms (IDENTICAL/arm 1.271) |
| algos | ovr | taxi | Xq | - | 33.2 | 34.4 | - | - | - | sklearn-cpu 9.8 ms (IDENTICAL/arm 3.501) |
| algos | pls-canonical | istella | Xq | - | 39.6 | 39.8 | - | - | - | sklearn-cpu 57.3 ms (IDENTICAL/arm 0.695) |
| algos | pls-canonical | taxi | Xq | - | 7.5 | 5.8 | - | - | - | sklearn-cpu 7.2 ms (IDENTICAL/arm 0.806) |
| algos | pls | istella | Xq | - | 22.3 | 22.0 | - | - | - | sklearn-cpu 33.4 ms (IDENTICAL/arm 0.657) |
| algos | pls | taxi | Xq | - | 2.2 | 2.2 | - | - | - | sklearn-cpu 2.4 ms (IDENTICAL/arm 0.935) |
| algos | poly-features | istella | Xq | - | 29.1 | 28.2 | - | - | - | sklearn-cpu 30.5 ms (IDENTICAL/arm 0.925) |
| algos | poly-features | taxi | Xq | - | 12.5 | 12.2 | - | - | - | sklearn-cpu 19.4 ms (IDENTICAL/arm 0.627) |
| algos | power-transformer | istella | Xq | - | 60.2 | 36.9 | - | - | - | sklearn-cpu 236.4 ms (IDENTICAL/arm 0.156) |
| algos | power-transformer | taxi | Xq | - | 4.1 | 4.2 | - | - | - | sklearn-cpu 10.5 ms (IDENTICAL/arm 0.394) |
| algos | qda | istella | Xq | - | 176.2 | 191.6 | - | - | - | sklearn-cpu 109.3 ms (IDENTICAL/arm 1.753) |
| algos | qda | taxi | Xq | - | 5.1 | 5.3 | - | - | - | sklearn-cpu 13.0 ms (IDENTICAL/arm 0.403) |
| algos | quantile-transformer | istella | Xq | - | 84.5 | 78.9 | - | - | - | sklearn-cpu 812.5 ms (IDENTICAL/arm 0.097) |
| algos | quantile-transformer | taxi | Xq | - | 3.9 | 4.5 | - | - | - | sklearn-cpu 35.5 ms (IDENTICAL/arm 0.126) |
| algos | random-trees-embedding | istella | Xq | - | 17.9 | 17.9 | - | - | - | sklearn-cpu 73.6 ms (IDENTICAL/arm 0.244) |
| algos | random-trees-embedding | taxi | Xq | - | 41.4 | 49.6 | - | - | - | sklearn-cpu 120.9 ms (IDENTICAL/arm 0.410) |
| algos | resnet-block | synthetic | Xq | - | 181.9 | 185.6 | - | - | - | torch-eager-fp32 3.8 ms (IDENTICAL/arm 48.794); torch-compile-fp32 33.5 ms (IDENTICAL/arm 5.547); torch-eager-bf16 3.3 ms (IDENTICAL/arm 55.551); torch-compile-bf16 32.8 ms (IDENTICAL/arm 5.654) |
| algos | ridge-cv | istella | Xq | - | - | - | - | - | - | sklearn-cpu 3.9 ms (IDENTICAL/arm -) |
| algos | ridge-cv | taxi | Xq | - | - | - | - | - | - | sklearn-cpu 0.3 ms (IDENTICAL/arm -) |
| algos | rnn-clf | synthetic | Xq | - | 38.4 | 40.6 | - | - | - | torch-eager-fp32 - ms (IDENTICAL/arm -); torch-compile-fp32 - ms (IDENTICAL/arm -); torch-eager-bf16 - ms (IDENTICAL/arm -); torch-compile-bf16 - ms (IDENTICAL/arm -) |
| algos | rnn-clf | taxi-hourly | Xq | - | 38.8 | 43.5 | - | - | - | torch-eager-fp32 - ms (IDENTICAL/arm -); torch-compile-fp32 - ms (IDENTICAL/arm -); torch-eager-bf16 - ms (IDENTICAL/arm -); torch-compile-bf16 - ms (IDENTICAL/arm -) |
| algos | rnn-reg | synthetic | Xq | - | 19.9 | 20.0 | - | - | - | torch-eager-fp32 - ms (IDENTICAL/arm -); torch-compile-fp32 - ms (IDENTICAL/arm -); torch-eager-bf16 - ms (IDENTICAL/arm -); torch-compile-bf16 - ms (IDENTICAL/arm -) |
| algos | rnn-reg | taxi-hourly | Xq | - | 21.4 | 20.1 | - | - | - | torch-eager-fp32 - ms (IDENTICAL/arm -); torch-compile-fp32 - ms (IDENTICAL/arm -); torch-eager-bf16 - ms (IDENTICAL/arm -); torch-compile-bf16 - ms (IDENTICAL/arm -) |
| algos | robust-scaler | istella | Xq | - | 82.2 | 64.4 | - | - | - | sklearn-cpu 19.7 ms (IDENTICAL/arm 3.261) |
| algos | robust-scaler | taxi | Xq | - | 2.9 | 4.4 | - | - | - | sklearn-cpu 1.9 ms (IDENTICAL/arm 2.360) |
| algos | simple-imputer | istella | Xq | - | 62.6 | 60.9 | - | - | - | sklearn-cpu 94.2 ms (IDENTICAL/arm 0.646) |
| algos | simple-imputer | taxi | Xq | - | 2.8 | 3.2 | - | - | - | sklearn-cpu 4.3 ms (IDENTICAL/arm 0.740) |
| algos | sparse-coder | istella | Xq | - | 186.9 | 185.0 | - | - | - | sklearn-cpu 3469.1 ms (IDENTICAL/arm 0.053) |
| algos | sparse-coder | taxi | Xq | - | 164.7 | 164.2 | - | - | - | sklearn-cpu 3443.9 ms (IDENTICAL/arm 0.048) |
| algos | sparse-pca | istella | Xq | - | - | - | - | - | - | sklearn-cpu 5.6 ms (IDENTICAL/arm -) |
| algos | sparse-pca | taxi | Xq | - | 5.0 | 5.2 | - | - | - | sklearn-cpu 1.8 ms (IDENTICAL/arm 2.869) |
| algos | sparse-rp | istella | Xq | - | 21.5 | 22.3 | - | - | - | sklearn-cpu 31.7 ms (IDENTICAL/arm 0.703) |
| algos | sparse-rp | taxi | Xq | - | 3.1 | 3.0 | - | - | - | sklearn-cpu 1.4 ms (IDENTICAL/arm 2.123) |
| algos | spline | istella | Xq | - | 16.4 | 27.4 | - | - | - | sklearn-cpu 80.6 ms (IDENTICAL/arm 0.340) |
| algos | spline | taxi | Xq | - | 10.6 | 20.4 | - | - | - | sklearn-cpu 54.2 ms (IDENTICAL/arm 0.376) |
| algos | stacking-clf | istella | Xq | - | 67.7 | 57.7 | - | - | - | sklearn-cpu 191.6 ms (IDENTICAL/arm 0.301) |
| algos | stacking-clf | taxi | Xq | - | 28.4 | 33.1 | - | - | - | sklearn-cpu 23.3 ms (IDENTICAL/arm 1.423) |
| algos | stacking-reg | istella | Xq | - | - | - | - | - | - | sklearn-cpu 10.1 ms (IDENTICAL/arm -) |
| algos | stacking-reg | taxi | Xq | - | - | - | - | - | - | sklearn-cpu 4.3 ms (IDENTICAL/arm -) |
| algos | standard-scaler | istella | Xq | - | 69.3 | 68.8 | - | - | - | sklearn-cpu 30.1 ms (IDENTICAL/arm 2.284) |
| algos | standard-scaler | taxi | Xq | - | 4.0 | 4.0 | - | - | - | sklearn-cpu 2.0 ms (IDENTICAL/arm 1.961) |
| algos | svgp | istella | Xq | - | 50.4 | 57.2 | - | - | - | gpytorch-gpu - ms (IDENTICAL/arm -); gpytorch-cpu 96.1 ms (IDENTICAL/arm 0.595) |
| algos | svgp | taxi | Xq | - | - | - | - | - | - | gpytorch-gpu - ms (IDENTICAL/arm -); gpytorch-cpu 64.9 ms (IDENTICAL/arm -) |
| algos | target-encoder | istella | Xq | - | 8.4 | 8.1 | - | - | - | sklearn-cpu 19.6 ms (IDENTICAL/arm 0.415) |
| algos | target-encoder | taxi | Xq | - | 7.6 | 7.4 | - | - | - | sklearn-cpu 11.3 ms (IDENTICAL/arm 0.655) |
| algos | variance-threshold | istella | Xq | - | 38.8 | 37.6 | - | - | - | sklearn-cpu 19.0 ms (IDENTICAL/arm 1.979) |
| algos | variance-threshold | taxi | Xq | - | 3.1 | 2.7 | - | - | - | sklearn-cpu 0.7 ms (IDENTICAL/arm 3.734) |
| algos | voting-clf | istella | Xq | - | 54.9 | 54.3 | - | - | - | sklearn-cpu 231.6 ms (IDENTICAL/arm 0.234) |
| algos | voting-clf | taxi | Xq | - | 11.0 | 10.9 | - | - | - | sklearn-cpu 26.6 ms (IDENTICAL/arm 0.410) |
| algos | voting-reg | istella | Xq | - | - | - | - | - | - | sklearn-cpu 14.3 ms (IDENTICAL/arm -) |
| algos | voting-reg | taxi | Xq | - | - | - | - | - | - | sklearn-cpu 4.6 ms (IDENTICAL/arm -) |

## Classical

### hdbscan / istella (rows full, shape 1000000x220)

race: done, driver rc 0, log `logs/classical.hdbscan.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 50979.4 | 50979.4..50979.4 | 1 | - | - | - | 1934.7 | - | n_clusters=47, noise_fraction=0.253810, rows=100000 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 56130.7 | 56130.7..56130.7 | 1 | - | - | - | 1908.1 | - | ari_vs_ours=1.000000, n_clusters=47, noise_agreement_vs_ours=1.000000, noise_fraction=0.253810, rows=100000 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(timeout: null) (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host not sampled; GPU not sampled

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

### hdbscan / taxi (rows full, shape 1000000x11)

race: done, driver rc 0, log `logs/classical.hdbscan.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 4883.7 | 4883.7..4883.7 | 1 | - | - | - | 631.5 | - | n_clusters=161, noise_fraction=0.140550, rows=100000 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 954.0 | 954.0..954.0 | 1 | - | - | - | 520.5 | - | ari_vs_ours=0.991364, n_clusters=160, noise_agreement_vs_ours=0.998310, noise_fraction=0.142220, rows=100000 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 30810.1 | 30810.1..30810.1 | 1 | 0.159 | 0.031 | - | 276.0 | - | ari_vs_ours=0.991251, n_clusters=160, noise_agreement_vs_ours=0.998290, noise_fraction=0.142240, rows=100000 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/classical2.gmm.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 626.5 | 626.5..626.5 | 1 | - | - | - | 491.2 | - | bic=-3.67e+06, mean_log_likelihood=12.861940, n_iter=32 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
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

race: done, driver rc 0, log `logs/classical2.nystroem.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 578.1 | 578.1..578.1 | 1 | - | - | - | 2004.6 | - | kernel_rel_error=0.033430 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 479.6 | 479.6..479.6 | 1 | - | - | - | 1988.7 | - | kernel_rel_error=0.033431 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 509.4 | 509.4..509.4 | 1 | 1.135 | 0.941 | - | 1461.3 | - | kernel_rel_error=0.038958 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/classical2.nystroem.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 631.1 | 631.1..631.1 | 1 | - | - | - | 1031.2 | - | kernel_rel_error=0.045611 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 494.2 | 494.2..494.2 | 1 | - | - | - | 869.5 | - | kernel_rel_error=0.045612 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 246.1 | 246.1..246.1 | 1 | 2.564 | 2.008 | - | 495.8 | - | kernel_rel_error=0.044370 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.adaboost-clf.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 13446.0 | 13446.0..13446.0 | 1 | - | - | - | 51143.7 | - | accuracy=0.934990, logloss=0.438712 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 13876.5 | 13876.5..13876.5 | 1 | - | - | - | 51370.2 | - | accuracy=0.934990, logloss=0.438712 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(timeout: null) (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host not sampled; GPU not sampled

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
| mojolearn IDENTICAL | Xq | - | 922.5 | 922.5..922.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 1247.3 | 1247.3..1247.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(timeout: null) |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### adaboost-clf / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.adaboost-clf.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 2973.5 | 2973.5..2973.5 | 1 | - | - | - | 3570.1 | - | accuracy=0.765230, logloss=0.543227 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 2990.7 | 2990.7..2990.7 | 1 | - | - | - | 3768.6 | - | accuracy=0.765230, logloss=0.543227 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 27303.6 | 27303.6..27303.6 | 1 | 0.109 | 0.110 | - | 263.5 | - | accuracy=0.765360, logloss=0.540565 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 371.0 | 371.0..371.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 754.4 | 754.4..754.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 163.4 | 163.4..163.4 | 1 | 2.271 | 4.618 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### adaboost-reg / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.adaboost-reg.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 6775.5 | 6775.5..6775.5 | 1 | - | - | - | 18643.0 | - | finite=True, r2=0.234086, rmse=0.731056 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 5457.5 | 5457.5..5457.5 | 1 | - | - | - | 16065.4 | - | finite=True, r2=0.238879, rmse=0.728764 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 183984.7 | 183984.7..183984.7 | 1 | 0.037 | 0.030 | - | 2035.5 | - | finite=True, r2=0.167622, rmse=0.762115 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 158.3 | 158.3..158.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 121.7 | 121.7..121.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 87.3 | 87.3..87.3 | 1 | 1.814 | 1.395 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### adaboost-reg / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.adaboost-reg.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3061.5 | 3061.5..3061.5 | 1 | - | - | - | 2775.3 | - | finite=True, r2=-0.417994, rmse=18.965923 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 2582.7 | 2582.7..2582.7 | 1 | - | - | - | 2742.9 | - | finite=True, r2=0.216396, rmse=14.098896 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 12196.7 | 12196.7..12196.7 | 1 | 0.251 | 0.212 | - | 358.0 | - | finite=True, r2=0.563927, rmse=10.517589 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 66.4 | 66.4..66.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 66.0 | 66.0..66.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 38.2 | 38.2..38.2 | 1 | 1.740 | 1.729 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### adafactor / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.adafactor.synthetic.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3258.3 | 3258.3..3258.3 | 1 | - | - | - | 2279.9 | - | rel_fro_vs_torch_eager_fp32=4.324e-05 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 448.7 | 448.7..448.7 | 1 | - | - | - | 2275.6 | - | rel_fro_vs_torch_eager_fp32=1.45e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 343.5 | 343.5..343.5 | 1 | 9.487 | 1.306 | - | 1911.1 | 1032.5 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 389.8 | 389.8..389.8 | 1 | 8.358 | 1.151 | - | 1968.8 | 1032.5 | rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.adagrad.synthetic.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 303.9 | 303.9..303.9 | 1 | - | - | - | 2117.2 | - | rel_fro_vs_torch_eager_fp32=3.297e-09 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 297.8 | 297.8..297.8 | 1 | - | - | - | 2116.8 | - | rel_fro_vs_torch_eager_fp32=3.297e-09 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 14.2 | 14.2..14.2 | 1 | 21.425 | 20.996 | - | 1919.8 | 1032.5 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 109.9 | 109.9..109.9 | 1 | 2.764 | 2.709 | - | 1980.6 | 1032.5 | rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.adam.synthetic.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1083.8 | 1083.8..1083.8 | 1 | - | - | - | 1925.4 | - | rel_fro_vs_torch_eager_fp32=3.339e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested identical, read back unknown) |
| mojolearn FAST | mojolearn | gpu | fast | 1089.3 | 1089.3..1089.3 | 1 | - | - | - | 1924.3 | - | rel_fro_vs_torch_eager_fp32=3.34e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested fast, read back unknown) |
| torch-eager-fp32 | torch | gpu | opponent | 29.3 | 29.3..29.3 | 1 | - | - | - | 1919.5 | 1032.5 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 115.6 | 115.6..115.6 | 1 | - | - | - | 1978.4 | 1024.5 | rel_fro_vs_torch_eager_fp32=3.91e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.adamax.synthetic.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 413.6 | 413.6..413.6 | 1 | - | - | - | 2202.7 | - | rel_fro_vs_torch_eager_fp32=4.267e-09 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 416.2 | 416.2..416.2 | 1 | - | - | - | 2248.2 | - | rel_fro_vs_torch_eager_fp32=4.267e-09 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 27.7 | 27.7..27.7 | 1 | 14.925 | 15.018 | - | 1919.9 | 1032.5 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 112.6 | 112.6..112.6 | 1 | 3.673 | 3.696 | - | 1978.0 | 1024.5 | rel_fro_vs_torch_eager_fp32=5.258e-09 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.adamw.synthetic.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1084.8 | 1084.8..1084.8 | 1 | - | - | - | 1925.5 | - | rel_fro_vs_torch_eager_fp32=3.339e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested identical, read back unknown) |
| mojolearn FAST | mojolearn | gpu | fast | 1082.7 | 1082.7..1082.7 | 1 | - | - | - | 1923.3 | - | rel_fro_vs_torch_eager_fp32=3.34e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested fast, read back unknown) |
| torch-eager-fp32 | torch | gpu | opponent | 31.1 | 31.1..31.1 | 1 | - | - | - | 1920.0 | 1032.5 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 124.6 | 124.6..124.6 | 1 | - | - | - | 1974.7 | 1024.5 | rel_fro_vs_torch_eager_fp32=3.91e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

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

### als / taxi-zones (rows full, shape X 129352x261; Xq 14373x261)

race: done, driver rc 0, log `logs/algos.als.taxi-zones.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(timeout: null) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(timeout: null) |
| implicit-cpu | implicit | cpu | opponent | 30318.4 | 30318.4..30318.4 | 1 | - | - | - | 465.5 | - | recall_at_10=0.056786 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.auto-theta.synthetic.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1237.8 | 1237.8..1237.8 | 1 | - | - | - | 343.4 | - | forecast_rmse=1.438912 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 972.2 | 972.2..972.2 | 1 | - | - | - | 343.2 | - | forecast_rmse=1.438072 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsforecast-cpu | statsforecast | cpu | opponent | 1952.6 | 1952.6..1952.6 | 1 | 0.634 | 0.498 | - | 189.5 | - | forecast_rmse=1.437804 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.auto-theta.taxi-hourly.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 4757.8 | 4757.8..4757.8 | 1 | - | - | - | 392.0 | - | forecast_rmse=49.054628 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 3449.5 | 3449.5..3449.5 | 1 | - | - | - | 380.3 | - | forecast_rmse=49.313360 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsforecast-cpu | statsforecast | cpu | opponent | 4032.5 | 4032.5..4032.5 | 1 | 1.180 | 0.855 | - | 189.2 | - | forecast_rmse=49.273467 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.autoarima.synthetic.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "NotImplementedError(\"AutoARIMA: seasonal_test='seas' (statsmodels' STL in the reference) is not implemented; pass D as one integer\")", "event": "error", "stage": "round 0"}) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "NotImplementedError(\"AutoARIMA: seasonal_test='seas' (statsmodels' STL in the reference) is not implemented; pass D as one integer\")", "event": "error", "stage": "round 0"}) |
| statsforecast-cpu | statsforecast | cpu | opponent | 2880.0 | 2880.0..2880.0 | 1 | - | - | - | 188.8 | - | forecast_rmse=17.546480 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.autoarima.taxi-hourly.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "NotImplementedError(\"AutoARIMA: seasonal_test='seas' (statsmodels' STL in the reference) is not implemented; pass D as one integer\")", "event": "error", "stage": "round 0"}) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "NotImplementedError(\"AutoARIMA: seasonal_test='seas' (statsmodels' STL in the reference) is not implemented; pass D as one integer\")", "event": "error", "stage": "round 0"}) |
| statsforecast-cpu | statsforecast | cpu | opponent | 2844.1 | 2844.1..2844.1 | 1 | - | - | - | 196.7 | - | forecast_rmse=68.211648 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

### avgpool1d / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.avgpool1d.synthetic.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 52.2 | 52.2..52.2 | 1 | - | - | - | 826.9 | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 51.7 | 51.7..51.7 | 1 | - | - | - | 827.8 | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 10.2 | 10.2..10.2 | 1 | 5.113 | 5.062 | - | 1581.3 | 1024.5 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 1.7 | 1.7..1.7 | 1 | 30.688 | 30.378 | - | 1679.5 | 1024.5 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 2.6 | 2.6..2.6 | 1 | 20.422 | 20.215 | - | 1579.9 | 1024.5 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 2.1 | 2.1..2.1 | 1 | 25.257 | 25.002 | - | 1675.3 | 1024.5 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

settings: {'ceil_mode': False, 'count_include_pad': True, 'kernel_size': 2, 'padding': 0, 'stride': 2}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | "none (deterministic)" | 7 | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 15.8 | 15.8..15.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 15.4 | 15.4..15.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | 1.7 | 1.7..1.7 | 1 | 9.427 | 9.169 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 1.0 | 1.0..1.0 | 1 | 15.194 | 14.777 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 1.0 | 1.0..1.0 | 1 | 16.416 | 15.966 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 1.2 | 1.2..1.2 | 1 | 13.651 | 13.277 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, ours-fast: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### avgpool2d / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.avgpool2d.synthetic.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 151.0 | 151.0..151.0 | 1 | - | - | - | 1423.2 | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 147.1 | 147.1..147.1 | 1 | - | - | - | 1424.1 | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 12.4 | 12.4..12.4 | 1 | 12.174 | 11.859 | - | 1730.9 | 1024.5 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 11.6 | 11.6..11.6 | 1 | 13.066 | 12.728 | - | 1813.0 | 1024.5 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 3.4 | 3.4..3.4 | 1 | 44.710 | 43.553 | - | 1731.7 | 1024.5 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 2.5 | 2.5..2.5 | 1 | 59.452 | 57.912 | - | 1809.1 | 1024.5 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

settings: {'ceil_mode': False, 'count_include_pad': True, 'divisor_override': None, 'kernel_size': 2, 'padding': 0, 'stride': 2}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | "none (deterministic)" | 7 | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 27.6 | 27.6..27.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 28.8 | 28.8..28.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | 0.8 | 0.8..0.8 | 1 | 34.565 | 36.047 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 0.8 | 0.8..0.8 | 1 | 35.645 | 37.173 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 0.8 | 0.8..0.8 | 1 | 35.275 | 36.787 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 0.8 | 0.8..0.8 | 1 | 36.027 | 37.571 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, ours-fast: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### bagging-clf / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.bagging-clf.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 4185.2 | 4185.2..4185.2 | 1 | - | - | - | 6152.3 | - | accuracy=0.941670, logloss=0.149399 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 4560.5 | 4560.5..4560.5 | 1 | - | - | - | 6661.0 | - | accuracy=0.941670, logloss=0.149399 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 37698.8 | 37698.8..37698.8 | 1 | 0.111 | 0.121 | - | 1139.1 | - | accuracy=0.941440, logloss=0.152069 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 114.4 | 114.4..114.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 115.6 | 115.6..115.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 940.7 | 940.7..940.7 | 1 | 0.122 | 0.123 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### bagging-clf / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.bagging-clf.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 577.0 | 577.0..577.0 | 1 | - | - | - | 821.4 | - | accuracy=0.767570, logloss=0.530450 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 574.0 | 574.0..574.0 | 1 | - | - | - | 853.3 | - | accuracy=0.767570, logloss=0.530450 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 2206.4 | 2206.4..2206.4 | 1 | 0.262 | 0.260 | - | 260.9 | - | accuracy=0.767650, logloss=0.530078 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 30.0 | 30.0..30.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 33.5 | 33.5..33.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 558.6 | 558.6..558.6 | 1 | 0.054 | 0.060 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### bagging-reg / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.bagging-reg.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 4106.2 | 4106.2..4106.2 | 1 | - | - | - | 5955.6 | - | finite=True, r2=0.522033, rmse=0.577510 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 4537.1 | 4537.1..4537.1 | 1 | - | - | - | 5968.2 | - | finite=True, r2=0.522033, rmse=0.577510 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 36039.3 | 36039.3..36039.3 | 1 | 0.114 | 0.126 | - | 1130.4 | - | finite=True, r2=0.518475, rmse=0.579656 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 81.4 | 81.4..81.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 82.1 | 82.1..82.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 495.4 | 495.4..495.4 | 1 | 0.164 | 0.166 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### bagging-reg / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.bagging-reg.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 509.6 | 509.6..509.6 | 1 | - | - | - | 833.6 | - | finite=True, r2=0.918767, rmse=4.539457 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 530.8 | 530.8..530.8 | 1 | - | - | - | 827.4 | - | finite=True, r2=0.918767, rmse=4.539457 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 2201.7 | 2201.7..2201.7 | 1 | 0.231 | 0.241 | - | 258.3 | - | finite=True, r2=0.938805, rmse=3.939993 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 15.0 | 15.0..15.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 15.7 | 15.7..15.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 212.2 | 212.2..212.2 | 1 | 0.071 | 0.074 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### batchnorm1d / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.batchnorm1d.synthetic.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 81.0 | 81.0..81.0 | 1 | - | - | - | 1021.8 | - | max_rel_diff_vs_torch_eager_fp32=0.063434, rel_fro_vs_torch_eager_fp32=4.095e-06 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 74.7 | 74.7..74.7 | 1 | - | - | - | 1021.6 | - | max_rel_diff_vs_torch_eager_fp32=0.063434, rel_fro_vs_torch_eager_fp32=4.096e-06 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 11.7 | 11.7..11.7 | 1 | 6.906 | 6.372 | - | 1626.1 | 1098.7 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 12.3 | 12.3..12.3 | 1 | 6.564 | 6.056 | - | 1696.2 | 1032.5 | max_rel_diff_vs_torch_eager_fp32=0.012127, rel_fro_vs_torch_eager_fp32=3.022e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 2.4 | 2.4..2.4 | 1 | 33.952 | 31.327 | - | 1630.8 | 1098.7 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 4.3 | 4.3..4.3 | 1 | 18.728 | 17.280 | - | 1687.7 | 1032.5 | max_rel_diff_vs_torch_eager_fp32=0.012127, rel_fro_vs_torch_eager_fp32=3.022e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

settings: {'affine': True, 'eps': 1e-05, 'momentum': 0.1, 'num_features': 256, 'track_running_stats': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| eps | 1e-05 | 1e-05 | 1e-05 | 1e-05 | 1e-05 | 1e-05 |
| momentum | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| seed | "none (deterministic)" | "none (deterministic)" | 7 | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 34.1 | 34.1..34.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 32.9 | 32.9..32.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | 0.9 | 0.9..0.9 | 1 | 37.042 | 35.742 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 2.6 | 2.6..2.6 | 1 | 13.128 | 12.667 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 1.4 | 1.4..1.4 | 1 | 24.596 | 23.733 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 3.0 | 3.0..3.0 | 1 | 11.290 | 10.894 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, ours-fast: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### batchnorm2d / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.batchnorm2d.synthetic.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 71.0 | 71.0..71.0 | 1 | - | - | - | 887.0 | - | max_rel_diff_vs_torch_eager_fp32=0.034926, rel_fro_vs_torch_eager_fp32=2.344e-05 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 63.3 | 63.3..63.3 | 1 | - | - | - | 886.6 | - | max_rel_diff_vs_torch_eager_fp32=0.034926, rel_fro_vs_torch_eager_fp32=2.344e-05 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 11.3 | 11.3..11.3 | 1 | 6.296 | 5.612 | - | 1628.6 | 1084.7 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 12.6 | 12.6..12.6 | 1 | 5.623 | 5.012 | - | 1678.1 | 1032.5 | max_rel_diff_vs_torch_eager_fp32=0.002228, rel_fro_vs_torch_eager_fp32=1.886e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 2.3 | 2.3..2.3 | 1 | 30.527 | 27.210 | - | 1631.7 | 1084.7 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 4.2 | 4.2..4.2 | 1 | 16.878 | 15.044 | - | 1669.3 | 1032.5 | max_rel_diff_vs_torch_eager_fp32=0.002228, rel_fro_vs_torch_eager_fp32=1.886e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

settings: {'affine': True, 'eps': 1e-05, 'momentum': 0.1, 'num_features': 64, 'track_running_stats': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| eps | 1e-05 | 1e-05 | 1e-05 | 1e-05 | 1e-05 | 1e-05 |
| momentum | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| seed | "none (deterministic)" | "none (deterministic)" | 7 | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 26.5 | 26.5..26.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 28.0 | 28.0..28.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | 1.4 | 1.4..1.4 | 1 | 18.512 | 19.568 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 2.6 | 2.6..2.6 | 1 | 10.016 | 10.587 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 0.9 | 0.9..0.9 | 1 | 29.952 | 31.661 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 2.6 | 2.6..2.6 | 1 | 10.090 | 10.666 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, ours-fast: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### bernoulli-nb / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.bernoulli-nb.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 779.7 | 779.7..779.7 | 1 | - | - | - | 4636.9 | - | accuracy=0.794050, logloss=5.350625 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 614.8 | 614.8..614.8 | 1 | - | - | - | 4636.3 | - | accuracy=0.794050, logloss=5.350586 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 868.5 | 868.5..868.5 | 1 | 0.898 | 0.708 | - | 3637.2 | - | accuracy=0.794050, logloss=4.278741 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 118.6 | 118.6..118.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 116.9 | 116.9..116.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 185.5 | 185.5..185.5 | 1 | 0.639 | 0.630 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### bernoulli-nb / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.bernoulli-nb.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 217.6 | 217.6..217.6 | 1 | - | - | - | 565.7 | - | accuracy=0.755560, logloss=0.557803 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 37.9 | 37.9..37.9 | 1 | - | - | - | 566.2 | - | accuracy=0.755560, logloss=0.557803 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 92.5 | 92.5..92.5 | 1 | 2.352 | 0.410 | - | 399.1 | - | accuracy=0.755560, logloss=0.557802 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 7.7 | 7.7..7.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 10.3 | 10.3..10.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 20.4 | 20.4..20.4 | 1 | 0.379 | 0.505 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### binarizer / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.binarizer.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 0.1 | 0.1..0.1 | 1 | - | - | - | 2006.9 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x220, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 0.1 | 0.1..0.1 | 1 | - | - | - | 2005.1 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x220, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 25.3 | 25.3..25.3 | 1 | 0.005 | 0.005 | - | 1451.5 | - | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 57.1 | 57.1..57.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 55.6 | 55.6..55.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 70.2 | 70.2..70.2 | 1 | 0.814 | 0.792 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### binarizer / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.binarizer.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 0.1 | 0.1..0.1 | 1 | - | - | - | 406.6 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x11, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 0.1 | 0.1..0.1 | 1 | - | - | - | 411.3 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x11, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1.6 | 1.6..1.6 | 1 | 0.069 | 0.052 | - | 213.8 | - | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 3.4 | 3.4..3.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 3.5 | 3.5..3.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 5.1 | 5.1..5.1 | 1 | 0.668 | 0.681 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### bootstrap / istella (rows full, shape X 20000x220; Xq 100000x220; y 20000; yq 100000)

race: done, driver rc 0, log `logs/algos.bootstrap.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 15.4 | 15.4..15.4 | 1 | - | - | - | 1291.3 | - | ci_endpoint_diff_over_width_vs_scipy=0.027369, ci_high=0.294850, ci_low=0.271250, standard_error=0.005987 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested identical, read back unknown) |
| mojolearn FAST | mojolearn | gpu | fast | 14.2 | 14.2..14.2 | 1 | - | - | - | 1293.0 | - | ci_endpoint_diff_over_width_vs_scipy=0.027369, ci_high=0.294850, ci_low=0.271250, standard_error=0.005987 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested fast, read back unknown) |
| scipy-cpu | scipy | cpu | opponent | 525.2 | 525.2..525.2 | 1 | - | - | - | 1168.4 | - | ci_high=0.295500, ci_low=0.271750, standard_error=0.006044 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.bootstrap.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 16.1 | 16.1..16.1 | 1 | - | - | - | 399.5 | - | ci_endpoint_diff_over_width_vs_scipy=0.005021, ci_high=18.717379, ci_low=18.249411, standard_error=0.117837 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested identical, read back unknown) |
| mojolearn FAST | mojolearn | gpu | fast | 16.3 | 16.3..16.3 | 1 | - | - | - | 396.4 | - | ci_endpoint_diff_over_width_vs_scipy=0.005017, ci_high=18.717377, ci_low=18.249411, standard_error=0.117837 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested fast, read back unknown) |
| scipy-cpu | scipy | cpu | opponent | 525.3 | 525.3..525.3 | 1 | - | - | - | 273.2 | - | ci_high=18.715050, ci_low=18.251265, standard_error=0.117562 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

### bpe-encode / enwik8 (rows full, shape -)

race: done, driver rc 0, log `logs/algos.bpe-encode.enwik8.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 56.4 | 56.4..56.4 | 1 | - | - | - | 263.9 | - | documents_equal_to_ours=1.000000, tokens=1457323 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 49.6 | 49.6..49.6 | 1 | - | - | - | 266.3 | - | documents_equal_to_ours=1.000000, tokens=1457323 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| hf-tokenizers-cpu | tokenizers | cpu | opponent | 133.9 | 133.9..133.9 | 1 | 0.421 | 0.370 | - | 583.3 | - | documents_equal_to_ours=1.000000, tokens=1457323 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, hf-tokenizers-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'min_frequency': 2, 'vocab_size': 4096}. Rows: None. Timed: None.

mismatch: bpe-train: each library breaks count ties by its own rule

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | hf-tokenizers-cpu | ours | ours-fast |
|---|---||---|---||---|---|
| library (source) | tokenizers (declared) | mojolearn (declared) | mojolearn (declared) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### bpe-train / enwik8 (rows full, shape -)

race: done, driver rc 0, log `logs/algos.bpe-train.enwik8.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 47.5 | 47.5..47.5 | 1 | - | - | - | 103.1 | - | jaccard_vs_ours=1.000000, n_tokens=4096 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 47.6 | 47.6..47.6 | 1 | - | - | - | 96.5 | - | jaccard_vs_ours=1.000000, n_tokens=4096 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| hf-tokenizers-cpu | tokenizers | cpu | opponent | 1224.2 | 1224.2..1224.2 | 1 | 0.039 | 0.039 | - | 216.1 | - | jaccard_vs_ours=0.999512, n_tokens=4096 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, hf-tokenizers-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'min_frequency': 2, 'vocab_size': 4096}. Rows: None. Timed: None.

mismatch: bpe-train: each library breaks count ties by its own rule

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | hf-tokenizers-cpu | ours | ours-fast |
|---|---||---|---||---|---|
| library (source) | tokenizers (declared) | mojolearn (declared) | mojolearn (declared) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### cagra / istella (rows full, shape index 400000x220; queries 4000x220)

race: done, driver rc 0, log `logs/algos.cagra.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "TypeError(\"CagraIndex.__init__() got an unexpected keyword argument 'random_state'\")", "event": "error", "stage": "round 0"}) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "TypeError(\"CagraIndex.__init__() got an unexpected keyword argument 'random_state'\")", "event": "error", "stage": "round 0"}) |
| faiss-cpu | faiss | cpu | opponent | 1633.5 | 1633.5..1633.5 | 1 | - | - | - | 1295.0 | - | recall_at_10=0.999375 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host not sampled; GPU not sampled

memory, faiss-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'graph_degree': 32, 'intermediate_graph_degree': 64, 'itopk_size': 64, 'n_neighbors': 10, 'random_state': 7}. Rows: None. Timed: None.

mismatch: faiss-cpu is HNSW (IndexHNSWFlat M=32, efConstruction=128, efSearch=64), the CPU graph index; cuvs-gpu is CAGRA itself

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | faiss-cpu | ours | ours-fast |
|---|---||---|---||---|---|
| library (source) | faiss (declared) | mojolearn (declared) | mojolearn (declared) |
| n_neighbors | 10 | 10 | 10 |
| seed | "none (no argument; draws random numbers, see exceptions)" | 7 | 7 |

accepted difference: faiss-cpu seed: faiss IndexHNSWFlat takes no seed argument (its level draw uses faiss' fixed internal seed)

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(error: {"error": "TypeError(\"CagraIndex.__init__() got an unexpected keyword argument 'random_state'\")", "event": "error", "stage": "round 0"}) |
| mojolearn FAST | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(error: {"error": "TypeError(\"CagraIndex.__init__() got an unexpected keyword argument 'random_state'\")", "event": "error", "stage": "round 0"}) |
| faiss-cpu | Xq | - | 9.8 | 9.8..9.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: search(queries)(Xq)

inference call, ours-fast: search(queries)(Xq)

inference call, faiss-cpu: search(queries)(Xq)

### cagra / taxi (rows full, shape index 400000x11; queries 4000x11)

race: done, driver rc 0, log `logs/algos.cagra.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "TypeError(\"CagraIndex.__init__() got an unexpected keyword argument 'random_state'\")", "event": "error", "stage": "round 0"}) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "TypeError(\"CagraIndex.__init__() got an unexpected keyword argument 'random_state'\")", "event": "error", "stage": "round 0"}) |
| faiss-cpu | faiss | cpu | opponent | 1015.6 | 1015.6..1015.6 | 1 | - | - | - | 338.1 | - | recall_at_10=0.927700 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host not sampled; GPU not sampled

memory, faiss-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'graph_degree': 32, 'intermediate_graph_degree': 64, 'itopk_size': 64, 'n_neighbors': 10, 'random_state': 7}. Rows: None. Timed: None.

mismatch: faiss-cpu is HNSW (IndexHNSWFlat M=32, efConstruction=128, efSearch=64), the CPU graph index; cuvs-gpu is CAGRA itself

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | faiss-cpu | ours | ours-fast |
|---|---||---|---||---|---|
| library (source) | faiss (declared) | mojolearn (declared) | mojolearn (declared) |
| n_neighbors | 10 | 10 | 10 |
| seed | "none (no argument; draws random numbers, see exceptions)" | 7 | 7 |

accepted difference: faiss-cpu seed: faiss IndexHNSWFlat takes no seed argument (its level draw uses faiss' fixed internal seed)

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(error: {"error": "TypeError(\"CagraIndex.__init__() got an unexpected keyword argument 'random_state'\")", "event": "error", "stage": "round 0"}) |
| mojolearn FAST | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(error: {"error": "TypeError(\"CagraIndex.__init__() got an unexpected keyword argument 'random_state'\")", "event": "error", "stage": "round 0"}) |
| faiss-cpu | Xq | - | 3.9 | 3.9..3.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: search(queries)(Xq)

inference call, ours-fast: search(queries)(Xq)

inference call, faiss-cpu: search(queries)(Xq)

### calibrated / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.calibrated.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3165.4 | 3165.4..3165.4 | 1 | - | - | - | 3657.5 | - | accuracy=0.885090, logloss=0.289787 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 2186.3 | 2186.3..2186.3 | 1 | - | - | - | 3663.7 | - | accuracy=0.885090, logloss=0.289832 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 2784.4 | 2784.4..2784.4 | 1 | 1.137 | 0.785 | - | 3092.6 | - | accuracy=0.885090, logloss=0.289787 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'cv': 5, 'ensemble': True, 'estimator': 'GaussianNB()', 'method': 'sigmoid'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| cv | 5 | 5 | 5 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 163.8 | 163.8..163.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 159.8 | 159.8..159.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 988.9 | 988.9..988.9 | 1 | 0.166 | 0.162 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### calibrated / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.calibrated.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 2616.9 | 2616.9..2616.9 | 1 | - | - | - | 559.8 | - | accuracy=0.755330, logloss=0.550727 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1358.0 | 1358.0..1358.0 | 1 | - | - | - | 543.2 | - | accuracy=0.755330, logloss=0.550718 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 555.6 | 555.6..555.6 | 1 | 4.710 | 2.444 | - | 334.5 | - | accuracy=0.755330, logloss=0.550727 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'cv': 5, 'ensemble': True, 'estimator': 'GaussianNB()', 'method': 'sigmoid'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| cv | 5 | 5 | 5 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 60.7 | 60.7..60.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 65.0 | 65.0..65.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 81.5 | 81.5..81.5 | 1 | 0.745 | 0.797 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### categorical-nb / istella (rows full, shape X 1000000x8; Xq 100000x8; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.categorical-nb.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 339.2 | 339.2..339.2 | 1 | - | - | - | 517.8 | - | accuracy=0.838850, logloss=0.412625 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 145.7 | 145.7..145.7 | 1 | - | - | - | 518.6 | - | accuracy=0.838850, logloss=0.412625 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 84.4 | 84.4..84.4 | 1 | 4.021 | 1.727 | - | 299.0 | - | accuracy=0.838850, logloss=0.412625 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 26.3 | 26.3..26.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 6.7 | 6.7..6.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 15.6 | 15.6..15.6 | 1 | 1.684 | 0.427 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### categorical-nb / taxi (rows full, shape X 1000000x5; Xq 100000x5; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.categorical-nb.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 323.6 | 323.6..323.6 | 1 | - | - | - | 458.4 | - | accuracy=0.765850, logloss=0.538866 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 142.6 | 142.6..142.6 | 1 | - | - | - | 458.0 | - | accuracy=0.765850, logloss=0.538866 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 78.3 | 78.3..78.3 | 1 | 4.136 | 1.823 | - | 265.2 | - | accuracy=0.765850, logloss=0.538866 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 25.9 | 25.9..25.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 6.4 | 6.4..6.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 12.3 | 12.3..12.3 | 1 | 2.107 | 0.523 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### cca / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.cca.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 39774.9 | 39774.9..39774.9 | 1 | - | - | - | 5570.6 | - | mean_canonical_corr=0.998053 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 38792.7 | 38792.7..38792.7 | 1 | - | - | - | 5586.4 | - | mean_canonical_corr=0.998053 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 73060.8 | 73060.8..73060.8 | 1 | 0.544 | 0.531 | - | 6367.6 | - | mean_canonical_corr=0.999568 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 43.8 | 43.8..43.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 43.7 | 43.7..43.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 57.5 | 57.5..57.5 | 1 | 0.762 | 0.761 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### cca / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.cca.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 220.2 | 220.2..220.2 | 1 | - | - | - | 942.3 | - | mean_canonical_corr=0.576863 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 212.8 | 212.8..212.8 | 1 | - | - | - | 938.9 | - | mean_canonical_corr=0.576863 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 463.5 | 463.5..463.5 | 1 | 0.475 | 0.459 | - | 432.8 | - | mean_canonical_corr=0.576863 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 11.0 | 11.0..11.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 5.8 | 5.8..5.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 7.2 | 7.2..7.2 | 1 | 1.529 | 0.810 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### cholesky / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.cholesky.synthetic.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 971.3 | 971.3..971.3 | 1 | - | - | - | 2662.4 | - | relative_residual=2.9e-07 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 958.2 | 958.2..958.2 | 1 | - | - | - | 2647.5 | - | relative_residual=1.659e-07 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| numpy-cpu | numpy | cpu | opponent | 1112.5 | 1112.5..1112.5 | 1 | 0.873 | 0.861 | - | 1324.0 | - | relative_residual=3.928e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-gpu | torch | gpu | opponent | 110.3 | 110.3..110.3 | 1 | 8.810 | 8.691 | - | 1715.8 | 1032.5 | relative_residual=5.449e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.classical-mds.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(timeout: null) |
| mojolearn FAST | mojolearn | gpu | fast | 205.1 | 205.1..205.1 | 1 | - | - | - | 1296.1 | - | trustworthiness_k15=0.830548 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 2623.3 | 2623.3..2623.3 | 1 | - | 0.078 | - | 466.4 | - | trustworthiness_k15=0.830548 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.classical-mds.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(timeout: null) |
| mojolearn FAST | mojolearn | gpu | fast | 184.5 | 184.5..184.5 | 1 | - | - | - | 1262.4 | - | trustworthiness_k15=0.765649 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 2535.5 | 2535.5..2535.5 | 1 | - | 0.073 | - | 463.3 | - | trustworthiness_k15=0.765649 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

### clip-grad-norm / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.clip-grad-norm.synthetic.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 40.0 | 40.0..40.0 | 1 | - | - | - | 532.4 | - | norm=1.000000, norm_rel_diff_vs_ours=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested identical, read back unknown) |
| mojolearn FAST | mojolearn | gpu | fast | 264.6 | 264.6..264.6 | 1 | - | - | - | 531.1 | - | norm=1.000000, norm_rel_diff_vs_ours=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested fast, read back unknown) |
| torch-eager-fp32 | torch | gpu | opponent | 17.1 | 17.1..17.1 | 1 | - | - | - | 684.7 | 136.5 | norm=1.000000, norm_rel_diff_vs_ours=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 17.0 | 17.0..17.0 | 1 | - | - | - | 834.9 | 136.5 | norm=1.000000, norm_rel_diff_vs_ours=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-eager-fp32, torch-compile-fp32: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

settings: {'error_if_nonfinite': True, 'max_norm': 1.0, 'norm_type': 2.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | torch-compile-fp32 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | "none (deterministic)" | 7 | 7 |

### cnn-clf / synthetic (rows full, shape X 20000x1x28x28; Xq 5000x1x28x28; y 20000; yq 5000)

race: done, driver rc 0, log `logs/algos.cnn-clf.synthetic.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1740.1 | 1740.1..1740.1 | 1 | - | - | - | 1022.7 | - | accuracy=1.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1113.6 | 1113.6..1113.6 | 1 | - | - | - | 1019.2 | - | accuracy=1.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host not sampled; GPU not sampled

settings: {'batch_size': 128, 'conv_channels': [8, 16], 'dampening': 0.0, 'input_shape': [1, 28, 28], 'kernel_size': 3, 'learning_rate': 0.01, 'max_iter': 2, 'momentum': 0.9, 'nesterov': False, 'optimizer': 'sgd', 'pool_size': 2, 'random_state': 7, 'shuffle': True, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (attributes) | mojolearn (attributes) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| batch_size | 128 | 128 | 128 | 128 | 128 | 128 |
| betas | [0.9, 0.999] | [0.9, 0.999] | - | - | - | - |
| dampening | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| eps | 1e-08 | 1e-08 | - | - | - | - |
| learning_rate | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| max_iter | 2 | 2 | 2 | 2 | 2 | 2 |
| momentum | 0.9 | 0.9 | 0.9 | 0.9 | 0.9 | 0.9 |
| nesterov | false | false | false | false | false | false |
| seed | 7 | 7 | 7 | 7 | 7 | 7 |
| shuffle | true | true | true | true | true | true |
| weight_decay | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 43.5 | 43.5..43.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 25.3 | 25.3..25.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
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

### complement-nb / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.complement-nb.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 314.4 | 314.4..314.4 | 1 | - | - | - | 3960.9 | - | accuracy=0.849190, logloss=3.766564 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 155.4 | 155.4..155.4 | 1 | - | - | - | 3964.2 | - | accuracy=0.849360, logloss=3.762531 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 186.2 | 186.2..186.2 | 1 | 1.688 | 0.834 | - | 3799.3 | - | accuracy=0.849350, logloss=3.174763 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 33.5 | 33.5..33.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 35.3 | 35.3..35.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 47.1 | 47.1..47.1 | 1 | 0.712 | 0.750 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### complement-nb / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.complement-nb.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 210.2 | 210.2..210.2 | 1 | - | - | - | 524.3 | - | accuracy=0.677000, logloss=0.715601 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 21.9 | 21.9..21.9 | 1 | - | - | - | 526.3 | - | accuracy=0.678030, logloss=0.715492 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 40.9 | 40.9..40.9 | 1 | 5.144 | 0.535 | - | 378.5 | - | accuracy=0.678030, logloss=0.715493 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 4.9 | 4.9..4.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 5.1 | 5.1..5.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 9.6 | 9.6..9.6 | 1 | 0.508 | 0.530 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### complement-nb / text (rows full, shape X 86626x4096; Xq 9626x4096; y 86626; yq 9626)

race: done, driver rc 0, log `logs/algos.complement-nb.text.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 207.3 | 207.3..207.3 | 1 | - | - | - | 4540.1 | - | accuracy=0.983067, logloss=0.559491 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 247.8 | 247.8..247.8 | 1 | - | - | - | 4540.5 | - | accuracy=0.983067, logloss=0.559491 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 264.8 | 264.8..264.8 | 1 | 0.783 | 0.936 | - | 4367.1 | - | accuracy=0.983067, logloss=0.557285 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 51.3 | 51.3..51.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 43.4 | 43.4..43.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 66.7 | 66.7..66.7 | 1 | 0.769 | 0.651 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### connected-components / istella (rows full, shape X 20000x220; indices 304830; indices2 62848; indptr 20001; indptr2 20001; y 20000)

race: done, driver rc 0, log `logs/algos.connected-components.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1573.0 | 1573.0..1573.0 | 1 | - | - | - | 3396.1 | - | ari_vs_networkx=1.000000, n_components=81 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1569.2 | 1569.2..1569.2 | 1 | - | - | - | 4922.5 | - | ari_vs_networkx=1.000000, n_components=81 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| networkx-cpu | networkx | cpu | opponent | 7.0 | 7.0..7.0 | 1 | 225.876 | 225.326 | - | 107.7 | - | n_components=81 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.connected-components.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1369.2 | 1369.2..1369.2 | 1 | - | - | - | 3380.8 | - | ari_vs_networkx=1.000000, n_components=588 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1366.0 | 1366.0..1366.0 | 1 | - | - | - | 4001.8 | - | ari_vs_networkx=1.000000, n_components=588 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| networkx-cpu | networkx | cpu | opponent | 7.3 | 7.3..7.3 | 1 | 186.450 | 186.016 | - | 93.1 | - | n_components=588 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

### conv1d / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.conv1d.synthetic.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 231.1 | 231.1..231.1 | 1 | - | - | - | 2868.6 | - | max_rel_diff_vs_torch_eager_fp32=0.391155, rel_fro_vs_torch_eager_fp32=3.947e-07 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 235.4 | 235.4..235.4 | 1 | - | - | - | 2865.0 | - | max_rel_diff_vs_torch_eager_fp32=0.391155, rel_fro_vs_torch_eager_fp32=3.947e-07 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 19.0 | 19.0..19.0 | 1 | 12.148 | 12.377 | - | 1635.2 | 1040.7 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 17.9 | 17.9..17.9 | 1 | 12.904 | 13.146 | - | 1772.5 | 1040.5 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 21.7 | 21.7..21.7 | 1 | 10.624 | 10.824 | - | 1640.7 | 1040.7 | max_rel_diff_vs_torch_eager_fp32=3479.164094, rel_fro_vs_torch_eager_fp32=0.002870 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 8.0 | 8.0..8.0 | 1 | 28.971 | 29.516 | - | 1778.7 | 1040.7 | max_rel_diff_vs_torch_eager_fp32=3418.128937, rel_fro_vs_torch_eager_fp32=0.003296 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

settings: {'bias': True, 'dilation': 1, 'groups': 1, 'in_channels': 128, 'kernel_size': 3, 'out_channels': 128, 'padding': 1, 'padding_mode': 'zeros', 'stride': 1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | 7 | 7 | 7 | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 66.4 | 66.4..66.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 64.1 | 64.1..64.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | 2.5 | 2.5..2.5 | 1 | 26.632 | 25.689 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 3.0 | 3.0..3.0 | 1 | 21.930 | 21.154 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 2.2 | 2.2..2.2 | 1 | 30.417 | 29.341 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 3.2 | 3.2..3.2 | 1 | 20.916 | 20.176 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, ours-fast: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### conv2d / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.conv2d.synthetic.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 122.8 | 122.8..122.8 | 1 | - | - | - | 2061.0 | - | max_rel_diff_vs_torch_eager_fp32=0.461936, rel_fro_vs_torch_eager_fp32=3.303e-07 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 116.9 | 116.9..116.9 | 1 | - | - | - | 2061.9 | - | max_rel_diff_vs_torch_eager_fp32=0.461936, rel_fro_vs_torch_eager_fp32=3.303e-07 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 14.2 | 14.2..14.2 | 1 | 8.632 | 8.214 | - | 1551.0 | 1036.5 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 13.9 | 13.9..13.9 | 1 | 8.827 | 8.400 | - | 1688.0 | 1036.5 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 13.4 | 13.4..13.4 | 1 | 9.179 | 8.735 | - | 1553.4 | 1036.7 | max_rel_diff_vs_torch_eager_fp32=3463.592380, rel_fro_vs_torch_eager_fp32=0.002935 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 5.4 | 5.4..5.4 | 1 | 22.654 | 21.559 | - | 1692.7 | 1036.7 | max_rel_diff_vs_torch_eager_fp32=3418.426961, rel_fro_vs_torch_eager_fp32=0.003382 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

settings: {'bias': True, 'dilation': 1, 'groups': 1, 'in_channels': 64, 'kernel_size': 3, 'out_channels': 64, 'padding': 1, 'padding_mode': 'zeros', 'stride': 1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | 7 | 7 | 7 | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 26.7 | 26.7..26.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 26.3 | 26.3..26.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | 1.5 | 1.5..1.5 | 1 | 17.555 | 17.268 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 2.3 | 2.3..2.3 | 1 | 11.430 | 11.242 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 1.5 | 1.5..1.5 | 1 | 18.203 | 17.904 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 1.7 | 1.7..1.7 | 1 | 15.708 | 15.451 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, ours-fast: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### cross-entropy / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.cross-entropy.synthetic.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 269.9 | 269.9..269.9 | 1 | - | - | - | 2644.3 | - | loss_rel_err_vs_fp64=4.564e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested identical, read back unknown) |
| mojolearn FAST | mojolearn | gpu | fast | 266.4 | 266.4..266.4 | 1 | - | - | - | 2641.3 | - | grad_max_rel_diff_vs_ours=2.794e-09, loss_rel_err_vs_fp64=4.564e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested fast, read back unknown) |
| torch-eager-fp32 | torch | gpu | opponent | 5.9 | 5.9..5.9 | 1 | - | - | - | 2119.0 | 1290.7 | grad_max_rel_diff_vs_ours=5.588e-09, loss_rel_err_vs_fp64=4.564e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 3.6 | 3.6..3.6 | 1 | - | - | - | 1956.0 | 1032.5 | grad_max_rel_diff_vs_ours=5.588e-09, loss_rel_err_vs_fp64=4.564e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-eager-fp32, torch-compile-fp32: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

settings: {'ignore_index': -100, 'label_smoothing': 0.0, 'reduction': 'mean'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | torch-compile-fp32 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | "none (deterministic)" | 7 | 7 |

### cross-val-score / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.cross-val-score.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3985.1 | 3985.1..3985.1 | 1 | - | - | - | 4840.6 | - | max_fold_score_diff_vs_sklearn=0.017412, mean_r2=0.333971 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested identical, read back unknown) |
| mojolearn FAST | mojolearn | gpu | fast | 3276.2 | 3276.2..3276.2 | 1 | - | - | - | 4846.0 | - | max_fold_score_diff_vs_sklearn=0.015390, mean_r2=0.331418 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested fast, read back unknown) |
| sklearn-cpu | scikit-learn | cpu | opponent | 7050.5 | 7050.5..7050.5 | 1 | - | - | - | 1156.9 | - | mean_r2=0.319779 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.cross-val-score.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 175.5 | 175.5..175.5 | 1 | - | - | - | 624.0 | - | max_fold_score_diff_vs_sklearn=5.007e-06, mean_r2=0.937955 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested identical, read back unknown) |
| mojolearn FAST | mojolearn | gpu | fast | 182.3 | 182.3..182.3 | 1 | - | - | - | 619.5 | - | max_fold_score_diff_vs_sklearn=4.172e-06, mean_r2=0.937955 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested fast, read back unknown) |
| sklearn-cpu | scikit-learn | cpu | opponent | 1282.3 | 1282.3..1282.3 | 1 | - | - | - | 269.8 | - | mean_r2=0.937956 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.croston-optimized.synthetic.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 14.0 | 14.0..14.0 | 1 | - | - | - | 324.5 | - | forecast_rmse=1.675538 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 13.1 | 13.1..13.1 | 1 | - | - | - | 324.7 | - | forecast_rmse=1.675539 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsforecast-cpu | statsforecast | cpu | opponent | 62.3 | 62.3..62.3 | 1 | 0.225 | 0.210 | - | 190.3 | - | forecast_rmse=1.675539 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.croston-optimized.taxi-hourly.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 14.1 | 14.1..14.1 | 1 | - | - | - | 323.8 | - | forecast_rmse=1.398253 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 16.8 | 16.8..16.8 | 1 | - | - | - | 323.9 | - | forecast_rmse=1.398258 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsforecast-cpu | statsforecast | cpu | opponent | 60.5 | 60.5..60.5 | 1 | 0.233 | 0.278 | - | 188.1 | - | forecast_rmse=1.398240 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.croston-sba.synthetic.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 5.5 | 5.5..5.5 | 1 | - | - | - | 325.0 | - | forecast_rmse=1.674465 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 5.1 | 5.1..5.1 | 1 | - | - | - | 323.4 | - | forecast_rmse=1.674465 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsforecast-cpu | statsforecast | cpu | opponent | 34.5 | 34.5..34.5 | 1 | 0.160 | 0.149 | - | 188.3 | - | forecast_rmse=1.674465 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.croston-sba.taxi-hourly.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 6.8 | 6.8..6.8 | 1 | - | - | - | 322.8 | - | forecast_rmse=1.386677 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 6.3 | 6.3..6.3 | 1 | - | - | - | 324.2 | - | forecast_rmse=1.386677 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsforecast-cpu | statsforecast | cpu | opponent | 47.9 | 47.9..47.9 | 1 | 0.142 | 0.132 | - | 189.3 | - | forecast_rmse=1.386677 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.croston.synthetic.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 6.3 | 6.3..6.3 | 1 | - | - | - | 325.5 | - | forecast_rmse=1.674840 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 7.0 | 7.0..7.0 | 1 | - | - | - | 324.2 | - | forecast_rmse=1.674840 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsforecast-cpu | statsforecast | cpu | opponent | 91.7 | 91.7..91.7 | 1 | 0.069 | 0.076 | - | 189.2 | - | forecast_rmse=1.674840 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.croston.taxi-hourly.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 6.1 | 6.1..6.1 | 1 | - | - | - | 324.3 | - | forecast_rmse=1.390261 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 5.0 | 5.0..5.0 | 1 | - | - | - | 323.4 | - | forecast_rmse=1.390261 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsforecast-cpu | statsforecast | cpu | opponent | 54.2 | 54.2..54.2 | 1 | 0.112 | 0.093 | - | 187.2 | - | forecast_rmse=1.390261 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.damped-ets.synthetic.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1031.5 | 1031.5..1031.5 | 1 | - | - | - | 406.6 | - | forecast_rmse=13.931153 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 541.4 | 541.4..541.4 | 1 | - | - | - | 422.8 | - | forecast_rmse=13.931185 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsmodels-cpu | statsmodels | cpu | opponent | 179.1 | 179.1..179.1 | 1 | 5.758 | 3.022 | - | 51.5 | - | forecast_rmse=26.588704 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| statsforecast-cpu | statsforecast | cpu | opponent | 167.2 | 167.2..167.2 | 1 | 6.168 | 3.237 | - | 190.4 | - | forecast_rmse=13.945986 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.damped-ets.taxi-hourly.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1140.3 | 1140.3..1140.3 | 1 | - | - | - | 405.5 | - | forecast_rmse=96.690449 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 624.4 | 624.4..624.4 | 1 | - | - | - | 422.4 | - | forecast_rmse=96.690681 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsmodels-cpu | statsmodels | cpu | opponent | 227.6 | 227.6..227.6 | 1 | 5.011 | 2.744 | - | 51.6 | - | forecast_rmse=196.928662 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| statsforecast-cpu | statsforecast | cpu | opponent | 185.0 | 185.0..185.0 | 1 | 6.163 | 3.374 | - | 190.6 | - | forecast_rmse=96.685568 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.dart-reg.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 57419.4 | 57419.4..57419.4 | 1 | - | - | - | 4886.0 | - | finite=True, r2=0.550733, rmse=0.559903 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 66171.5 | 66171.5..66171.5 | 1 | - | - | - | 4931.2 | - | finite=True, r2=0.550733, rmse=0.559903 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| lightgbm-cpu | lightgbm | cpu | opponent | 32389.1 | 32389.1..32389.1 | 1 | 1.773 | 2.043 | - | 1854.3 | - | finite=True, r2=0.564663, rmse=0.551154 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
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
| mojolearn IDENTICAL | Xq | - | 114.5 | 114.5..114.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 130.3 | 130.3..130.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| lightgbm-cpu | Xq | - | 64.1 | 64.1..64.1 | 1 | 1.787 | 2.035 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| xgboost-cpu | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(timeout: null) |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, lightgbm-cpu: predict(Xq)(Xq)

inference call, xgboost-cpu: predict(Xq)(Xq)

### dart-reg / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.dart-reg.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 9727.1 | 9727.1..9727.1 | 1 | - | - | - | 1223.0 | - | finite=True, r2=0.925497, rmse=4.347340 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 9743.6 | 9743.6..9743.6 | 1 | - | - | - | 1448.8 | - | finite=True, r2=0.925497, rmse=4.347340 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| lightgbm-cpu | lightgbm | cpu | opponent | 27649.9 | 27649.9..27649.9 | 1 | 0.352 | 0.352 | - | 308.9 | - | finite=True, r2=0.926249, rmse=4.325358 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| xgboost-cpu | xgboost | cpu | opponent | 49059.8 | 49059.8..49059.8 | 1 | 0.198 | 0.199 | - | 270.8 | - | finite=True, r2=0.926049, rmse=4.331203 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 178.1 | 178.1..178.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 144.3 | 144.3..144.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| lightgbm-cpu | Xq | - | 57.3 | 57.3..57.3 | 1 | 3.107 | 2.518 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| xgboost-cpu | Xq | - | 114.4 | 114.4..114.4 | 1 | 1.557 | 1.262 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, lightgbm-cpu: predict(Xq)(Xq)

inference call, xgboost-cpu: predict(Xq)(Xq)

### dart / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.dart.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 57616.4 | 57616.4..57616.4 | 1 | - | - | - | 4895.1 | - | accuracy=0.948720, logloss=0.134126 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 66613.2 | 66613.2..66613.2 | 1 | - | - | - | 4885.5 | - | accuracy=0.948720, logloss=0.134126 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| lightgbm-cpu | lightgbm | cpu | opponent | 35424.9 | 35424.9..35424.9 | 1 | 1.626 | 1.880 | - | 1914.2 | - | accuracy=0.951860, logloss=0.123740 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
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
| mojolearn IDENTICAL | Xq | - | 228.0 | 228.0..228.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 246.6 | 246.6..246.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| lightgbm-cpu | Xq | - | 57.5 | 57.5..57.5 | 1 | 3.969 | 4.292 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| xgboost-cpu | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(timeout: null) |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, lightgbm-cpu: predict(Xq)(Xq)

inference call, xgboost-cpu: predict(Xq)(Xq)

### dart / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.dart.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 10372.8 | 10372.8..10372.8 | 1 | - | - | - | 1186.9 | - | accuracy=0.768340, logloss=0.529060 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 10416.9 | 10416.9..10416.9 | 1 | - | - | - | 1321.6 | - | accuracy=0.768340, logloss=0.529060 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| lightgbm-cpu | lightgbm | cpu | opponent | 26497.9 | 26497.9..26497.9 | 1 | 0.391 | 0.393 | - | 275.8 | - | accuracy=0.768190, logloss=0.529108 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| xgboost-cpu | xgboost | cpu | opponent | 49274.5 | 49274.5..49274.5 | 1 | 0.211 | 0.211 | - | 278.5 | - | accuracy=0.768000, logloss=0.529540 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 272.4 | 272.4..272.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 278.3 | 278.3..278.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| lightgbm-cpu | Xq | - | 51.5 | 51.5..51.5 | 1 | 5.289 | 5.404 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| xgboost-cpu | Xq | - | 113.4 | 113.4..113.4 | 1 | 2.403 | 2.455 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, lightgbm-cpu: predict(Xq)(Xq)

inference call, xgboost-cpu: predict(Xq)(Xq)

### decision-tree-clf / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.decision-tree-clf.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 466.3 | 466.3..466.3 | 1 | - | - | - | 4121.8 | - | accuracy=0.935000, logloss=0.758303 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 486.6 | 486.6..486.6 | 1 | - | - | - | 4116.4 | - | accuracy=0.935000, logloss=0.758303 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 54799.9 | 54799.9..54799.9 | 1 | 0.009 | 0.009 | - | 1117.9 | - | accuracy=0.934410, logloss=0.791706 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 11.6 | 11.6..11.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 11.8 | 11.8..11.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 18.0 | 18.0..18.0 | 1 | 0.646 | 0.653 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### decision-tree-clf / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.decision-tree-clf.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 81.8 | 81.8..81.8 | 1 | - | - | - | 619.2 | - | accuracy=0.756300, logloss=1.202243 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 80.2 | 80.2..80.2 | 1 | - | - | - | 624.0 | - | accuracy=0.756300, logloss=1.202243 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 2218.4 | 2218.4..2218.4 | 1 | 0.037 | 0.036 | - | 234.3 | - | accuracy=0.756560, logloss=1.149550 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 3.6 | 3.6..3.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 3.4 | 3.4..3.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 11.1 | 11.1..11.1 | 1 | 0.327 | 0.310 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### decision-tree-reg / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.decision-tree-reg.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 453.8 | 453.8..453.8 | 1 | - | - | - | 4120.3 | - | finite=True, r2=0.379438, rmse=0.658041 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 483.1 | 483.1..483.1 | 1 | - | - | - | 4116.3 | - | finite=True, r2=0.379438, rmse=0.658041 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 52136.1 | 52136.1..52136.1 | 1 | 0.009 | 0.009 | - | 1102.4 | - | finite=True, r2=0.373177, rmse=0.661352 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 8.9 | 8.9..8.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 9.0 | 9.0..9.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 8.8 | 8.8..8.8 | 1 | 1.018 | 1.020 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### decision-tree-reg / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.decision-tree-reg.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 72.3 | 72.3..72.3 | 1 | - | - | - | 621.6 | - | finite=True, r2=0.862608, rmse=5.903617 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 73.6 | 73.6..73.6 | 1 | - | - | - | 621.5 | - | finite=True, r2=0.862608, rmse=5.903617 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 2116.9 | 2116.9..2116.9 | 1 | 0.034 | 0.035 | - | 229.8 | - | finite=True, r2=0.891507, rmse=5.246104 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 2.5 | 2.5..2.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 2.0 | 2.0..2.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 9.3 | 9.3..9.3 | 1 | 0.267 | 0.218 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### dict-learning / istella (rows full, shape X 20000x220; Xq 5000x220; y 20000; yq 5000)

race: done, driver rc 0, log `logs/algos.dict-learning.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 19308.2 | 19308.2..19308.2 | 1 | - | - | - | 1487.5 | - | component_sparsity=0.086364, relative_reconstruction_error=0.652121 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 17583.2 | 17583.2..17583.2 | 1 | - | - | - | 1492.7 | - | component_sparsity=0.086364, relative_reconstruction_error=0.652121 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 16441.5 | 16441.5..16441.5 | 1 | 1.174 | 1.069 | - | 1203.3 | - | component_sparsity=0.086364, relative_reconstruction_error=0.652121 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 139.6 | 139.6..139.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 96.7 | 96.7..96.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 44.0 | 44.0..44.0 | 1 | 3.174 | 2.199 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### dict-learning / taxi (rows full, shape X 20000x11; Xq 5000x11; y 20000; yq 5000)

race: done, driver rc 0, log `logs/algos.dict-learning.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 10976.1 | 10976.1..10976.1 | 1 | - | - | - | 448.0 | - | component_sparsity=0.000000, relative_reconstruction_error=0.459169 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 10277.2 | 10277.2..10277.2 | 1 | - | - | - | 452.6 | - | component_sparsity=0.000000, relative_reconstruction_error=0.459169 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 11560.8 | 11560.8..11560.8 | 1 | 0.949 | 0.889 | - | 217.3 | - | component_sparsity=0.000000, relative_reconstruction_error=0.460601 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 89.4 | 89.4..89.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 88.7 | 88.7..88.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 40.2 | 40.2..40.2 | 1 | 2.225 | 2.207 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### dropout2d / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.dropout2d.synthetic.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 81.1 | 81.1..81.1 | 1 | - | - | - | 1177.0 | - | error=KeyError('y') | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 81.1 | 81.1..81.1 | 1 | - | - | - | 1174.6 | - | error=KeyError('y') | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 2.4 | 2.4..2.4 | 1 | 34.025 | 34.036 | - | 1539.0 | 1032.5 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 2.2 | 2.2..2.2 | 1 | 36.351 | 36.364 | - | 1640.0 | 1032.5 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-eager-fp32, torch-compile-fp32: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

settings: {'p': 0.1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | torch-compile-fp32 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | torch (declared) | torch (declared) |
| p | 0.1 | 0.1 | 0.1 | 0.1 |
| seed | 7 | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 52.4 | 52.4..52.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 52.2 | 52.2..52.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | 1.4 | 1.4..1.4 | 1 | 36.343 | 36.190 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 0.6 | 0.6..0.6 | 1 | 92.050 | 91.660 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, ours-fast: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

### dynamic-optimized-theta / synthetic (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc 0, log `logs/algos.dynamic-optimized-theta.synthetic.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 638.8 | 638.8..638.8 | 1 | - | - | - | 343.4 | - | forecast_rmse=1.436278 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 496.5 | 496.5..496.5 | 1 | - | - | - | 342.9 | - | forecast_rmse=1.435976 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsforecast-cpu | statsforecast | cpu | opponent | 959.3 | 959.3..959.3 | 1 | 0.666 | 0.518 | - | 190.6 | - | forecast_rmse=1.436045 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.dynamic-optimized-theta.taxi-hourly.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3183.6 | 3183.6..3183.6 | 1 | - | - | - | 380.1 | - | forecast_rmse=49.086628 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 735.2 | 735.2..735.2 | 1 | - | - | - | 343.8 | - | forecast_rmse=49.083557 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsforecast-cpu | statsforecast | cpu | opponent | 1351.8 | 1351.8..1351.8 | 1 | 2.355 | 0.544 | - | 189.1 | - | forecast_rmse=49.314797 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.dynamic-theta.synthetic.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 142.2 | 142.2..142.2 | 1 | - | - | - | 344.3 | - | forecast_rmse=1.437256 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 127.3 | 127.3..127.3 | 1 | - | - | - | 343.6 | - | forecast_rmse=1.437165 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsforecast-cpu | statsforecast | cpu | opponent | 333.4 | 333.4..333.4 | 1 | 0.426 | 0.382 | - | 190.3 | - | forecast_rmse=1.437262 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.dynamic-theta.taxi-hourly.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 520.8 | 520.8..520.8 | 1 | - | - | - | 342.9 | - | forecast_rmse=49.101249 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 398.8 | 398.8..398.8 | 1 | - | - | - | 343.5 | - | forecast_rmse=49.100462 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsforecast-cpu | statsforecast | cpu | opponent | 768.3 | 768.3..768.3 | 1 | 0.678 | 0.519 | - | 189.3 | - | forecast_rmse=49.269843 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.eigh.synthetic.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(timeout: null) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(timeout: null) |
| numpy-cpu | numpy | cpu | opponent | 4813.1 | 4813.1..4813.1 | 1 | - | - | - | 549.2 | - | max_eigenvalue_error=3.49e-08, relative_residual=2.824e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
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

race: done, driver rc 0, log `logs/algos.elliptic-envelope.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(timeout: null) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "Exception(\"eigh: the Jacobi eigensolver did not converge in 15 sweeps at n = 220. An unconverged decomposition is not returned as if it were one; see DEVIATION 590. The remedy is more swee) |
| sklearn-cpu | scikit-learn | cpu | opponent | 43758.6 | 43758.6..43758.6 | 1 | - | - | - | 4183.8 | - | fraction_flagged=0.091570, jaccard_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| sklearn-cpu | Xq | - | 3250.4 | 3250.4..3250.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### elliptic-envelope / taxi (rows full, shape X 100000x11; Xq 100000x11; y 100000; yq 100000)

race: done, driver rc 0, log `logs/algos.elliptic-envelope.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 4010.2 | 4010.2..4010.2 | 1 | - | - | - | 538.8 | - | fraction_flagged=0.102370, jaccard_vs_sklearn=0.963574 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1783.8 | 1783.8..1783.8 | 1 | - | - | - | 535.5 | - | fraction_flagged=0.102370, jaccard_vs_sklearn=0.963574 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1423.3 | 1423.3..1423.3 | 1 | 2.817 | 1.253 | - | 319.2 | - | fraction_flagged=0.102470, jaccard_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

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
| mojolearn IDENTICAL | Xq | - | 5.7 | 5.7..5.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 5.8 | 5.8..5.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 5.0 | 5.0..5.0 | 1 | 1.134 | 1.152 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### embedding / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.embedding.synthetic.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 794.7 | 794.7..794.7 | 1 | - | - | - | 2056.1 | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested identical, read back unknown) |
| mojolearn FAST | mojolearn | gpu | fast | 795.4 | 795.4..795.4 | 1 | - | - | - | 2054.9 | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested fast, read back unknown) |
| torch-eager-fp32 | torch | gpu | opponent | 14.3 | 14.3..14.3 | 1 | - | - | - | 1753.5 | 1032.5 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 11.1 | 11.1..11.1 | 1 | - | - | - | 1853.1 | 1032.5 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-eager-fp32, torch-compile-fp32: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

settings: {'embedding_dim': 1024, 'max_norm': None, 'norm_type': 2.0, 'num_embeddings': 32768, 'padding_idx': None, 'scale_grad_by_freq': False, 'sparse': False}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | torch-compile-fp32 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | "none (deterministic)" | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 337.5 | 337.5..337.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 337.7 | 337.7..337.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | 1.3 | 1.3..1.3 | 1 | 255.835 | 256.014 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 1.4 | 1.4..1.4 | 1 | 235.616 | 235.781 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, ours-fast: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

### factor-analysis / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.factor-analysis.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "Exception(\"the one-sided Jacobi SVD did not converge in 60 sweeps at n_cols = 220: the last sweep still performed 1 rotations against a tolerance of 9.536743e-07. The remedy is more sweeps) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "Exception(\"the one-sided Jacobi SVD did not converge in 60 sweeps at n_cols = 220: the last sweep still performed 1 rotations against a tolerance of 9.536743e-07. The remedy is more sweeps) |
| sklearn-cpu | scikit-learn | cpu | opponent | 35135.4 | 35135.4..35135.4 | 1 | - | - | - | 4594.5 | - | mean_log_likelihood=98.122830 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| sklearn-cpu | Xq | - | 30.7 | 30.7..30.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### factor-analysis / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.factor-analysis.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 181.5 | 181.5..181.5 | 1 | - | - | - | 846.4 | - | mean_log_likelihood=-14.823632 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 177.9 | 177.9..177.9 | 1 | - | - | - | 846.1 | - | mean_log_likelihood=-14.823632 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 144606.7 | 144606.7..144606.7 | 1 | 0.001 | 0.001 | - | 233.3 | - | mean_log_likelihood=-14.823723 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 4.6 | 4.6..4.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 4.6 | 4.6..4.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 3.7 | 3.7..3.7 | 1 | 1.263 | 1.249 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### fastica / istella (rows full, shape X 900000x220; Xq 100000x220)

race: done, driver rc 0, log `logs/algos.fastica.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1575.8 | 1575.8..1575.8 | 1 | - | - | - | 4393.9 | - | mean_abs_excess_kurtosis=356.288855 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1333.6 | 1333.6..1333.6 | 1 | - | - | - | 4391.8 | - | mean_abs_excess_kurtosis=356.288827 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 16795.7 | 16795.7..16795.7 | 1 | 0.094 | 0.079 | - | 4160.3 | - | mean_abs_excess_kurtosis=922.531077 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 32.8 | 32.8..32.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 25.1 | 25.1..25.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 11.0 | 11.0..11.0 | 1 | 2.978 | 2.284 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### fastica / taxi (rows full, shape X 900000x11; Xq 100000x11)

race: done, driver rc 0, log `logs/algos.fastica.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 123.7 | 123.7..123.7 | 1 | - | - | - | 798.6 | - | mean_abs_excess_kurtosis=13.204824 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 102.6 | 102.6..102.6 | 1 | - | - | - | 799.9 | - | mean_abs_excess_kurtosis=13.205464 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 419.3 | 419.3..419.3 | 1 | 0.295 | 0.245 | - | 379.4 | - | mean_abs_excess_kurtosis=13.764740 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 3.4 | 3.4..3.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 3.4 | 3.4..3.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 2.1 | 2.1..2.1 | 1 | 1.614 | 1.605 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### garch / synthetic (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc 0, log `logs/algos.garch.synthetic.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3032.0 | 3032.0..3032.0 | 1 | - | - | - | 379.3 | - | mean_llf=-1938.223490 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1539.5 | 1539.5..1539.5 | 1 | - | - | - | 358.4 | - | mean_llf=-1938.224617 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| arch-cpu | arch | cpu | opponent | 105.5 | 105.5..105.5 | 1 | 28.735 | 14.591 | - | 182.2 | - | mean_llf=-1938.221004 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.garch.taxi-hourly.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 4847.6 | 4847.6..4847.6 | 1 | - | - | - | 402.9 | - | mean_llf=-1132.712756 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 2779.8 | 2779.8..2779.8 | 1 | - | - | - | 359.3 | - | mean_llf=-1132.954899 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| arch-cpu | arch | cpu | opponent | 124.7 | 124.7..124.7 | 1 | 38.871 | 22.290 | - | 191.0 | - | mean_llf=-1129.806866 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.gaussian-nb.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 405.0 | 405.0..405.0 | 1 | - | - | - | 2954.6 | - | accuracy=0.876530, logloss=3.574225 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 182.6 | 182.6..182.6 | 1 | - | - | - | 2958.4 | - | accuracy=0.876570, logloss=3.574408 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 360.9 | 360.9..360.9 | 1 | 1.122 | 0.506 | - | 2758.7 | - | accuracy=0.876530, logloss=3.417392 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 31.3 | 31.3..31.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 31.0 | 31.0..31.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 183.8 | 183.8..183.8 | 1 | 0.170 | 0.169 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### gaussian-nb / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.gaussian-nb.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 295.5 | 295.5..295.5 | 1 | - | - | - | 481.3 | - | accuracy=0.719900, logloss=1.133898 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 25.6 | 25.6..25.6 | 1 | - | - | - | 484.0 | - | accuracy=0.719820, logloss=1.132247 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 76.2 | 76.2..76.2 | 1 | 3.878 | 0.336 | - | 299.7 | - | accuracy=0.719900, logloss=1.133898 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 4.8 | 4.8..4.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 4.7 | 4.7..4.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 12.2 | 12.2..12.2 | 1 | 0.390 | 0.383 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### gaussian-rp / istella (rows full, shape X 900000x220; Xq 100000x220)

race: done, driver rc 0, log `logs/algos.gaussian-rp.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 135.8 | 135.8..135.8 | 1 | - | - | - | 4326.4 | - | mean_abs_distortion=0.680693 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 140.1 | 140.1..140.1 | 1 | - | - | - | 4327.3 | - | mean_abs_distortion=0.680693 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 24.3 | 24.3..24.3 | 1 | 5.583 | 5.761 | - | 997.5 | - | mean_abs_distortion=0.177966 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 23.4 | 23.4..23.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 22.6 | 22.6..22.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 4.0 | 4.0..4.0 | 1 | 5.895 | 5.711 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### gaussian-rp / taxi (rows full, shape X 900000x11; Xq 100000x11)

race: done, driver rc 0, log `logs/algos.gaussian-rp.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 5.8 | 5.8..5.8 | 1 | - | - | - | 571.0 | - | mean_abs_distortion=0.345752 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 5.8 | 5.8..5.8 | 1 | - | - | - | 570.4 | - | mean_abs_distortion=0.345752 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1.6 | 1.6..1.6 | 1 | 3.589 | 3.542 | - | 202.9 | - | mean_abs_distortion=0.339791 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 3.1 | 3.1..3.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 3.1 | 3.1..3.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 1.3 | 1.3..1.3 | 1 | 2.422 | 2.450 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### gcn / istella (rows full, shape X 100000x220; indices 1521510; indptr 100001; y 100000)

race: done, driver rc 0, log `logs/algos.gcn.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 189.6 | 189.6..189.6 | 1 | - | - | - | 1753.2 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 190.8 | 190.8..190.8 | 1 | - | - | - | 1769.4 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 110.4 | 110.4..110.4 | 1 | 1.718 | 1.729 | - | 4214.3 | 3429.1 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 18.8 | 18.8..18.8 | 1 | 10.113 | 10.177 | - | 1927.2 | 1044.7 | max_rel_diff_vs_torch_eager_fp32=0.017593, rel_fro_vs_torch_eager_fp32=1.004e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 102.6 | 102.6..102.6 | 1 | 1.847 | 1.859 | - | 4311.2 | 3523.1 | max_rel_diff_vs_torch_eager_fp32=2744.908399, rel_fro_vs_torch_eager_fp32=0.002187 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 26.9 | 26.9..26.9 | 1 | 7.056 | 7.101 | - | 2827.8 | 1938.7 | max_rel_diff_vs_torch_eager_fp32=2744.915191, rel_fro_vs_torch_eager_fp32=0.002187 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

settings: {'add_self_loops': True, 'bias': True, 'improved': False, 'normalize': True, 'out_channels': 128}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| normalize | true | true | true | true | true | true |
| seed | 7 | 7 | 7 | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 96.9 | 96.9..96.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 98.1 | 98.1..98.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | 31.4 | 31.4..31.4 | 1 | 3.081 | 3.121 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 4.9 | 4.9..4.9 | 1 | 19.574 | 19.825 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 31.8 | 31.8..31.8 | 1 | 3.050 | 3.089 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 4.8 | 4.8..4.8 | 1 | 20.092 | 20.350 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, ours-fast: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### gcn / taxi (rows full, shape X 100000x11; indices 1258298; indptr 100001; y 100000)

race: done, driver rc 0, log `logs/algos.gcn.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 139.7 | 139.7..139.7 | 1 | - | - | - | 1540.5 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 140.4 | 140.4..140.4 | 1 | - | - | - | 1392.2 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 90.2 | 90.2..90.2 | 1 | 1.550 | 1.557 | - | 3142.6 | 2439.1 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 17.8 | 17.8..17.8 | 1 | 7.870 | 7.906 | - | 1901.4 | 1102.7 | max_rel_diff_vs_torch_eager_fp32=0.003725, rel_fro_vs_torch_eager_fp32=9.327e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 87.8 | 87.8..87.8 | 1 | 1.592 | 1.599 | - | 3905.9 | 3201.1 | max_rel_diff_vs_torch_eager_fp32=1509.509981, rel_fro_vs_torch_eager_fp32=0.002330 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 22.6 | 22.6..22.6 | 1 | 6.171 | 6.199 | - | 2680.2 | 1872.7 | max_rel_diff_vs_torch_eager_fp32=1509.509981, rel_fro_vs_torch_eager_fp32=0.002330 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

settings: {'add_self_loops': True, 'bias': True, 'improved': False, 'normalize': True, 'out_channels': 128}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| normalize | true | true | true | true | true | true |
| seed | 7 | 7 | 7 | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 88.7 | 88.7..88.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 88.5 | 88.5..88.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | 26.0 | 26.0..26.0 | 1 | 3.406 | 3.399 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 3.9 | 3.9..3.9 | 1 | 22.562 | 22.512 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 26.2 | 26.2..26.2 | 1 | 3.383 | 3.375 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 3.7 | 3.7..3.7 | 1 | 24.043 | 23.989 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, ours-fast: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### global-avgpool / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.global-avgpool.synthetic.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 5.4 | 5.4..5.4 | 1 | - | - | - | 467.6 | - | max_rel_diff_vs_torch_eager_fp32=0.015661, rel_fro_vs_torch_eager_fp32=1.493e-07 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 3.7 | 3.7..3.7 | 1 | - | - | - | 464.0 | - | max_rel_diff_vs_torch_eager_fp32=0.015661, rel_fro_vs_torch_eager_fp32=1.493e-07 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 2.0 | 2.0..2.0 | 1 | 2.652 | 1.825 | - | 500.2 | 40.5 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 1.2 | 1.2..1.2 | 1 | 4.466 | 3.073 | - | 632.3 | 40.5 | max_rel_diff_vs_torch_eager_fp32=0.007299, rel_fro_vs_torch_eager_fp32=9.321e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 1.4 | 1.4..1.4 | 1 | 3.894 | 2.680 | - | 501.4 | 40.5 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 1.2 | 1.2..1.2 | 1 | 4.657 | 3.205 | - | 628.9 | 40.5 | max_rel_diff_vs_torch_eager_fp32=0.007299, rel_fro_vs_torch_eager_fp32=9.321e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

settings: {'output_size': 1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | "none (deterministic)" | 7 | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 0.5 | 0.5..0.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 0.5 | 0.5..0.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | 1.0 | 1.0..1.0 | 1 | 0.513 | 0.535 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 0.4 | 0.4..0.4 | 1 | 1.173 | 1.223 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 1.0 | 1.0..1.0 | 1 | 0.519 | 0.541 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 0.4 | 0.4..0.4 | 1 | 1.298 | 1.352 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, ours-fast: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### global-maxpool / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.global-maxpool.synthetic.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 5.6 | 5.6..5.6 | 1 | - | - | - | 471.6 | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 5.7 | 5.7..5.7 | 1 | - | - | - | 465.0 | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 3.9 | 3.9..3.9 | 1 | 1.427 | 1.453 | - | 508.3 | 42.5 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 1.9 | 1.9..1.9 | 1 | 2.953 | 3.006 | - | 637.2 | 42.5 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 1.3 | 1.3..1.3 | 1 | 4.282 | 4.360 | - | 507.6 | 42.5 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 2.1 | 2.1..2.1 | 1 | 2.676 | 2.725 | - | 635.5 | 42.5 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

settings: {'output_size': 1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | "none (deterministic)" | 7 | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 0.5 | 0.5..0.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 0.6 | 0.6..0.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | 0.3 | 0.3..0.3 | 1 | 1.558 | 1.727 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 0.4 | 0.4..0.4 | 1 | 1.388 | 1.538 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 0.3 | 0.3..0.3 | 1 | 1.763 | 1.954 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 0.4 | 0.4..0.4 | 1 | 1.336 | 1.481 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, ours-fast: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### graphsage / istella (rows full, shape X 100000x220; indices 1521510; indptr 100001; y 100000)

race: done, driver rc 0, log `logs/algos.graphsage.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 352.4 | 352.4..352.4 | 1 | - | - | - | 1854.0 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 356.0 | 356.0..356.0 | 1 | - | - | - | 1867.8 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 66.6 | 66.6..66.6 | 1 | 5.295 | 5.348 | - | 3114.6 | 2326.7 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 19.9 | 19.9..19.9 | 1 | 17.728 | 17.905 | - | 1978.5 | 1094.7 | max_rel_diff_vs_torch_eager_fp32=0.178814, rel_fro_vs_torch_eager_fp32=7.998e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 62.7 | 62.7..62.7 | 1 | 5.618 | 5.674 | - | 3115.5 | 2326.7 | max_rel_diff_vs_torch_eager_fp32=3907.114267, rel_fro_vs_torch_eager_fp32=0.003366 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 10.2 | 10.2..10.2 | 1 | 34.631 | 34.977 | - | 1957.1 | 1064.7 | max_rel_diff_vs_torch_eager_fp32=3678.172827, rel_fro_vs_torch_eager_fp32=0.003054 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

settings: {'aggr': 'mean', 'bias': True, 'normalize': False, 'out_channels': 128, 'project': False, 'root_weight': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| normalize | false | false | false | false | false | false |
| seed | 7 | 7 | 7 | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 141.7 | 141.7..141.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 140.3 | 140.3..140.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | 44.4 | 44.4..44.4 | 1 | 3.191 | 3.158 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 8.5 | 8.5..8.5 | 1 | 16.631 | 16.461 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 44.2 | 44.2..44.2 | 1 | 3.205 | 3.172 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 8.0 | 8.0..8.0 | 1 | 17.764 | 17.582 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, ours-fast: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### graphsage / taxi (rows full, shape X 100000x11; indices 1258298; indptr 100001; y 100000)

race: done, driver rc 0, log `logs/algos.graphsage.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 125.3 | 125.3..125.3 | 1 | - | - | - | 1487.9 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 120.5 | 120.5..120.5 | 1 | - | - | - | 1350.0 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 17.6 | 17.6..17.6 | 1 | 7.112 | 6.842 | - | 1765.3 | 1070.7 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 17.7 | 17.7..17.7 | 1 | 7.084 | 6.815 | - | 1911.6 | 1120.7 | max_rel_diff_vs_torch_eager_fp32=0.119209, rel_fro_vs_torch_eager_fp32=5.006e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 16.7 | 16.7..16.7 | 1 | 7.503 | 7.218 | - | 1767.9 | 1068.7 | max_rel_diff_vs_torch_eager_fp32=5460.333333, rel_fro_vs_torch_eager_fp32=0.003557 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 6.2 | 6.2..6.2 | 1 | 20.371 | 19.598 | - | 1893.7 | 1094.7 | max_rel_diff_vs_torch_eager_fp32=4608.154297, rel_fro_vs_torch_eager_fp32=0.003285 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

settings: {'aggr': 'mean', 'bias': True, 'normalize': False, 'out_channels': 128, 'project': False, 'root_weight': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| normalize | false | false | false | false | false | false |
| seed | 7 | 7 | 7 | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 88.4 | 88.4..88.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 85.6 | 85.6..85.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | 5.1 | 5.1..5.1 | 1 | 17.403 | 16.839 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 5.2 | 5.2..5.2 | 1 | 16.966 | 16.416 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 5.8 | 5.8..5.8 | 1 | 15.218 | 14.725 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 3.5 | 3.5..3.5 | 1 | 25.371 | 24.548 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, ours-fast: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### gru-clf / synthetic (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc 0, log `logs/algos.gru-clf.synthetic.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1737.5 | 1737.5..1737.5 | 1 | - | - | - | 675.8 | - | accuracy=0.971842, logloss=0.065835 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1662.0 | 1662.0..1662.0 | 1 | - | - | - | 684.5 | - | accuracy=0.971842, logloss=0.065835 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
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
| mojolearn IDENTICAL | Xq | - | 88.0 | 88.0..88.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 82.5 | 82.5..82.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
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

race: done, driver rc 0, log `logs/algos.gru-clf.taxi-hourly.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1725.8 | 1725.8..1725.8 | 1 | - | - | - | 669.8 | - | accuracy=0.865668, logloss=0.305841 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1672.6 | 1672.6..1672.6 | 1 | - | - | - | 677.7 | - | accuracy=0.865668, logloss=0.305841 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
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
| mojolearn IDENTICAL | Xq | - | 88.8 | 88.8..88.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 83.0 | 83.0..83.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
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

race: done, driver rc 0, log `logs/algos.gru-reg.synthetic.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1729.4 | 1729.4..1729.4 | 1 | - | - | - | 671.8 | - | finite=True, r2=0.981946, rmse=0.155672 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1658.8 | 1658.8..1658.8 | 1 | - | - | - | 667.7 | - | finite=True, r2=0.981946, rmse=0.155672 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
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
| mojolearn IDENTICAL | Xq | - | 43.3 | 43.3..43.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 41.0 | 41.0..41.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
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

race: done, driver rc 0, log `logs/algos.gru-reg.taxi-hourly.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1722.2 | 1722.2..1722.2 | 1 | - | - | - | 672.3 | - | finite=True, r2=0.748219, rmse=0.544182 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1652.5 | 1652.5..1652.5 | 1 | - | - | - | 673.8 | - | finite=True, r2=0.748219, rmse=0.544182 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
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
| mojolearn IDENTICAL | Xq | - | 43.6 | 43.6..43.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 41.0 | 41.0..41.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
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

race: done, driver rc 0, log `logs/algos.incremental-pca.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 2431.8 | 2431.8..2431.8 | 1 | - | - | - | 4611.4 | - | explained_variance_fraction=0.999994 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 2027.4 | 2027.4..2027.4 | 1 | - | - | - | 4612.9 | - | explained_variance_fraction=0.999994 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 5745.4 | 5745.4..5745.4 | 1 | 0.423 | 0.353 | - | 1803.1 | - | explained_variance_fraction=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 24.9 | 24.9..24.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 21.9 | 21.9..21.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 8.2 | 8.2..8.2 | 1 | 3.027 | 2.659 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### incremental-pca / taxi (rows full, shape X 900000x11; Xq 100000x11)

race: done, driver rc 0, log `logs/algos.incremental-pca.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 151.1 | 151.1..151.1 | 1 | - | - | - | 652.8 | - | explained_variance_fraction=0.999995 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 132.9 | 132.9..132.9 | 1 | - | - | - | 648.3 | - | explained_variance_fraction=0.999995 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 137.9 | 137.9..137.9 | 1 | 1.096 | 0.964 | - | 262.4 | - | explained_variance_fraction=0.999995 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 3.8 | 3.8..3.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 3.5 | 3.5..3.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 2.5 | 2.5..2.5 | 1 | 1.509 | 1.379 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### isomap / istella (rows full, shape X 10000x220; Xq 2000x220)

race: done, driver rc 0, log `logs/algos.isomap.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(timeout: null) |
| mojolearn FAST | mojolearn | gpu | fast | 11066.5 | 11066.5..11066.5 | 1 | - | - | - | 4691.0 | - | trustworthiness_k15=0.853298 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 13251.7 | 13251.7..13251.7 | 1 | - | 0.835 | - | 1744.5 | - | trustworthiness_k15=0.853294 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

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

race: done, driver rc 0, log `logs/algos.isomap.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(timeout: null) |
| mojolearn FAST | mojolearn | gpu | fast | 9049.8 | 9049.8..9049.8 | 1 | - | - | - | 4612.3 | - | trustworthiness_k15=0.771828 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 13664.7 | 13664.7..13664.7 | 1 | - | 0.662 | - | 1693.3 | - | trustworthiness_k15=0.771828 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

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

race: done, driver rc 0, log `logs/algos.iterative-imputer.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 7819.7 | 7819.7..7819.7 | 1 | - | - | - | 1611.0 | - | masked_rmse=799013.020330, max_abs_diff_vs_sklearn=1.011e+07 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 4618.8 | 4618.8..4618.8 | 1 | - | - | - | 1609.0 | - | masked_rmse=799040.699151, max_abs_diff_vs_sklearn=9.974e+06 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 10351.4 | 10351.4..10351.4 | 1 | 0.755 | 0.446 | - | 1378.0 | - | masked_rmse=802428.936225 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 10.2 | 10.2..10.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 10.5 | 10.5..10.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 68.8 | 68.8..68.8 | 1 | 0.148 | 0.153 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### iterative-imputer / taxi (rows full, shape X 100000x11; X_true 100000x11; Xq 20000x11; Xq_true 20000x11; y 100000; yq 20000)

race: done, driver rc 0, log `logs/algos.iterative-imputer.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1769.8 | 1769.8..1769.8 | 1 | - | - | - | 465.1 | - | masked_rmse=4.693848, max_abs_diff_vs_sklearn=0.044070 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 232.3 | 232.3..232.3 | 1 | - | - | - | 475.9 | - | masked_rmse=4.693971, max_abs_diff_vs_sklearn=0.0003719 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1049.1 | 1049.1..1049.1 | 1 | 1.687 | 0.221 | - | 238.9 | - | masked_rmse=4.693973 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 5.6 | 5.6..5.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 5.7 | 5.7..5.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 13.1 | 13.1..13.1 | 1 | 0.429 | 0.433 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### ivf-filter / istella (rows full, shape index 400000x220; queries 4000x220)

race: done, driver rc 0, log `logs/algos.ivf-filter.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 10442.6 | 10442.6..10442.6 | 1 | - | - | - | 2781.0 | - | recall_at_10=0.609250 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 6181.6 | 6181.6..6181.6 | 1 | - | - | - | 3195.5 | - | recall_at_10=0.652700 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| faiss-cpu | faiss | cpu | opponent | 7234.3 | 7234.3..7234.3 | 1 | 1.443 | 0.854 | - | 845.4 | - | recall_at_10=0.841950 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, faiss-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'kmeans_n_iters': 20, 'n_lists': 1024, 'n_neighbors': 10, 'n_probes': 32, 'pq_bits': 8, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | faiss-cpu | ours | ours-fast |
|---|---||---|---||---|---|
| library (source) | faiss (declared) | mojolearn (declared) | mojolearn (declared) |
| n_neighbors | 10 | 10 | 10 |
| nlist | 1024 | 1024 | 1024 |
| nprobe | 32 | 32 | 32 |
| seed | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 1278.0 | 1278.0..1278.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 1148.3 | 1148.3..1148.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| faiss-cpu | Xq | - | 88.7 | 88.7..88.7 | 1 | 14.409 | 12.947 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: search(queries)(Xq)

inference call, ours-fast: search(queries)(Xq)

inference call, faiss-cpu: search(queries)(Xq)

### ivf-filter / taxi (rows full, shape index 400000x11; queries 4000x11)

race: done, driver rc 0, log `logs/algos.ivf-filter.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1577.2 | 1577.2..1577.2 | 1 | - | - | - | 789.5 | - | recall_at_10=0.980075 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1270.8 | 1270.8..1270.8 | 1 | - | - | - | 804.6 | - | recall_at_10=0.982950 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| faiss-cpu | faiss | cpu | opponent | 4367.8 | 4367.8..4367.8 | 1 | 0.361 | 0.291 | - | 175.6 | - | recall_at_10=0.982050 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, faiss-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'kmeans_n_iters': 20, 'n_lists': 1024, 'n_neighbors': 10, 'n_probes': 32, 'pq_bits': 8, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | faiss-cpu | ours | ours-fast |
|---|---||---|---||---|---|
| library (source) | faiss (declared) | mojolearn (declared) | mojolearn (declared) |
| n_neighbors | 10 | 10 | 10 |
| nlist | 1024 | 1024 | 1024 |
| nprobe | 32 | 32 | 32 |
| seed | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 60.9 | 60.9..60.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 59.9 | 59.9..59.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| faiss-cpu | Xq | - | 48.2 | 48.2..48.2 | 1 | 1.263 | 1.242 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: search(queries)(Xq)

inference call, ours-fast: search(queries)(Xq)

inference call, faiss-cpu: search(queries)(Xq)

### ivf-pq / istella (rows full, shape index 400000x220; queries 4000x220)

race: done, driver rc 0, log `logs/algos.ivf-pq.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 10218.8 | 10218.8..10218.8 | 1 | - | - | - | 2792.1 | - | recall_at_10=0.550825 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 6151.6 | 6151.6..6151.6 | 1 | - | - | - | 3192.3 | - | recall_at_10=0.599475 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| faiss-cpu | faiss | cpu | opponent | 7007.9 | 7007.9..7007.9 | 1 | 1.458 | 0.878 | - | 840.1 | - | recall_at_10=0.802975 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, faiss-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'kmeans_n_iters': 20, 'n_lists': 1024, 'n_neighbors': 10, 'n_probes': 32, 'pq_bits': 8, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | faiss-cpu | ours | ours-fast |
|---|---||---|---||---|---|
| library (source) | faiss (declared) | mojolearn (declared) | mojolearn (declared) |
| n_neighbors | 10 | 10 | 10 |
| nlist | 1024 | 1024 | 1024 |
| nprobe | 32 | 32 | 32 |
| seed | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 1281.0 | 1281.0..1281.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 1177.3 | 1177.3..1177.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| faiss-cpu | Xq | - | 61.7 | 61.7..61.7 | 1 | 20.776 | 19.093 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: search(queries)(Xq)

inference call, ours-fast: search(queries)(Xq)

inference call, faiss-cpu: search(queries)(Xq)

### ivf-pq / taxi (rows full, shape index 400000x11; queries 4000x11)

race: done, driver rc 0, log `logs/algos.ivf-pq.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1616.5 | 1616.5..1616.5 | 1 | - | - | - | 844.3 | - | recall_at_10=0.973225 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1263.9 | 1263.9..1263.9 | 1 | - | - | - | 865.8 | - | recall_at_10=0.976450 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| faiss-cpu | faiss | cpu | opponent | 4367.0 | 4367.0..4367.0 | 1 | 0.370 | 0.289 | - | 187.1 | - | recall_at_10=0.980075 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, faiss-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'kmeans_n_iters': 20, 'n_lists': 1024, 'n_neighbors': 10, 'n_probes': 32, 'pq_bits': 8, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | faiss-cpu | ours | ours-fast |
|---|---||---|---||---|---|
| library (source) | faiss (declared) | mojolearn (declared) | mojolearn (declared) |
| n_neighbors | 10 | 10 | 10 |
| nlist | 1024 | 1024 | 1024 |
| nprobe | 32 | 32 | 32 |
| seed | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 53.3 | 53.3..53.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 53.1 | 53.1..53.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| faiss-cpu | Xq | - | 39.9 | 39.9..39.9 | 1 | 1.337 | 1.331 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: search(queries)(Xq)

inference call, ours-fast: search(queries)(Xq)

inference call, faiss-cpu: search(queries)(Xq)

### ivf-rabitq / istella (rows full, shape index 400000x220; queries 4000x220)

race: done, driver rc 0, log `logs/algos.ivf-rabitq.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 4271.2 | 4271.2..4271.2 | 1 | - | - | - | 2190.0 | - | recall_at_10=0.125125 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1937.8 | 1937.8..1937.8 | 1 | - | - | - | 2636.9 | - | recall_at_10=0.132550 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| faiss-cpu | faiss | cpu | opponent | 5474.0 | 5474.0..5474.0 | 1 | 0.780 | 0.354 | - | 695.3 | - | recall_at_10=0.050450 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, faiss-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'kmeans_n_iters': 20, 'n_lists': 1024, 'n_neighbors': 10, 'n_probes': 32, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | faiss-cpu | ours | ours-fast |
|---|---||---|---||---|---|
| library (source) | faiss (declared) | mojolearn (declared) | mojolearn (declared) |
| n_neighbors | 10 | 10 | 10 |
| nlist | 1024 | 1024 | 1024 |
| nprobe | 32 | 32 | 32 |
| seed | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 415.5 | 415.5..415.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 390.9 | 390.9..390.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| faiss-cpu | Xq | - | 60.8 | 60.8..60.8 | 1 | 6.836 | 6.432 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: search(queries)(Xq)

inference call, ours-fast: search(queries)(Xq)

inference call, faiss-cpu: search(queries)(Xq)

### ivf-rabitq / taxi (rows full, shape index 400000x11; queries 4000x11)

race: done, driver rc 0, log `logs/algos.ivf-rabitq.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 554.6 | 554.6..554.6 | 1 | - | - | - | 711.4 | - | recall_at_10=0.110475 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 465.9 | 465.9..465.9 | 1 | - | - | - | 812.3 | - | recall_at_10=0.115775 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| faiss-cpu | faiss | cpu | opponent | 3984.1 | 3984.1..3984.1 | 1 | 0.139 | 0.117 | - | 150.6 | - | recall_at_10=0.126275 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, faiss-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'kmeans_n_iters': 20, 'n_lists': 1024, 'n_neighbors': 10, 'n_probes': 32, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | faiss-cpu | ours | ours-fast |
|---|---||---|---||---|---|
| library (source) | faiss (declared) | mojolearn (declared) | mojolearn (declared) |
| n_neighbors | 10 | 10 | 10 |
| nlist | 1024 | 1024 | 1024 |
| nprobe | 32 | 32 | 32 |
| seed | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 38.3 | 38.3..38.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 37.2 | 37.2..37.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| faiss-cpu | Xq | - | 42.8 | 42.8..42.8 | 1 | 0.894 | 0.869 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: search(queries)(Xq)

inference call, ours-fast: search(queries)(Xq)

inference call, faiss-cpu: search(queries)(Xq)

### ivf-refine / istella (rows full, shape index 400000x220; queries 4000x220)

race: done, driver rc 0, log `logs/algos.ivf-refine.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 10215.3 | 10215.3..10215.3 | 1 | - | - | - | 2794.2 | - | recall_at_10=0.809175 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 6341.3 | 6341.3..6341.3 | 1 | - | - | - | 3194.2 | - | recall_at_10=0.862250 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| faiss-cpu | faiss | cpu | opponent | 7321.6 | 7321.6..7321.6 | 1 | 1.395 | 0.866 | - | 1506.4 | - | recall_at_10=0.993425 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, faiss-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'kmeans_n_iters': 20, 'n_lists': 1024, 'n_neighbors': 10, 'n_probes': 32, 'pq_bits': 8, 'random_state': 7, 'refine_ratio': 4}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | faiss-cpu | ours | ours-fast |
|---|---||---|---||---|---|
| library (source) | faiss (declared) | mojolearn (declared) | mojolearn (declared) |
| n_neighbors | 10 | 10 | 10 |
| nlist | 1024 | 1024 | 1024 |
| nprobe | 32 | 32 | 32 |
| seed | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 1402.4 | 1402.4..1402.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 1281.8 | 1281.8..1281.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| faiss-cpu | Xq | - | 64.1 | 64.1..64.1 | 1 | 21.895 | 20.012 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: search(queries)(Xq)

inference call, ours-fast: search(queries)(Xq)

inference call, faiss-cpu: search(queries)(Xq)

### ivf-refine / taxi (rows full, shape index 400000x11; queries 4000x11)

race: done, driver rc 0, log `logs/algos.ivf-refine.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1634.1 | 1634.1..1634.1 | 1 | - | - | - | 855.0 | - | recall_at_10=0.999675 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1296.2 | 1296.2..1296.2 | 1 | - | - | - | 872.1 | - | recall_at_10=0.999675 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| faiss-cpu | faiss | cpu | opponent | 4362.6 | 4362.6..4362.6 | 1 | 0.375 | 0.297 | - | 212.0 | - | recall_at_10=0.999225 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, faiss-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'kmeans_n_iters': 20, 'n_lists': 1024, 'n_neighbors': 10, 'n_probes': 32, 'pq_bits': 8, 'random_state': 7, 'refine_ratio': 4}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | faiss-cpu | ours | ours-fast |
|---|---||---|---||---|---|
| library (source) | faiss (declared) | mojolearn (declared) | mojolearn (declared) |
| n_neighbors | 10 | 10 | 10 |
| nlist | 1024 | 1024 | 1024 |
| nprobe | 32 | 32 | 32 |
| seed | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 81.4 | 81.4..81.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 80.8 | 80.8..80.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| faiss-cpu | Xq | - | 40.4 | 40.4..40.4 | 1 | 2.016 | 2.000 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: search(queries)(Xq)

inference call, ours-fast: search(queries)(Xq)

inference call, faiss-cpu: search(queries)(Xq)

### ivf-sq / istella (rows full, shape index 400000x220; queries 4000x220)

race: done, driver rc 0, log `logs/algos.ivf-sq.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 4474.8 | 4474.8..4474.8 | 1 | - | - | - | 4739.1 | - | recall_at_10=0.728025 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 2141.1 | 2141.1..2141.1 | 1 | - | - | - | 4972.1 | - | recall_at_10=0.628550 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| faiss-cpu | faiss | cpu | opponent | 5134.6 | 5134.6..5134.6 | 1 | 0.871 | 0.417 | - | 867.0 | - | recall_at_10=0.591300 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, faiss-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'kmeans_n_iters': 20, 'n_lists': 1024, 'n_neighbors': 10, 'n_probes': 32, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | faiss-cpu | ours | ours-fast |
|---|---||---|---||---|---|
| library (source) | faiss (declared) | mojolearn (declared) | mojolearn (declared) |
| n_neighbors | 10 | 10 | 10 |
| nlist | 1024 | 1024 | 1024 |
| nprobe | 32 | 32 | 32 |
| seed | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 830.8 | 830.8..830.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 777.2 | 777.2..777.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| faiss-cpu | Xq | - | 358.7 | 358.7..358.7 | 1 | 2.316 | 2.167 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: search(queries)(Xq)

inference call, ours-fast: search(queries)(Xq)

inference call, faiss-cpu: search(queries)(Xq)

### ivf-sq / taxi (rows full, shape index 400000x11; queries 4000x11)

race: done, driver rc 0, log `logs/algos.ivf-sq.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 556.7 | 556.7..556.7 | 1 | - | - | - | 775.1 | - | recall_at_10=0.934975 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 470.0 | 470.0..470.0 | 1 | - | - | - | 854.8 | - | recall_at_10=0.902025 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| faiss-cpu | faiss | cpu | opponent | 3782.8 | 3782.8..3782.8 | 1 | 0.147 | 0.124 | - | 139.0 | - | recall_at_10=0.857025 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, faiss-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'kmeans_n_iters': 20, 'n_lists': 1024, 'n_neighbors': 10, 'n_probes': 32, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | faiss-cpu | ours | ours-fast |
|---|---||---|---||---|---|
| library (source) | faiss (declared) | mojolearn (declared) | mojolearn (declared) |
| n_neighbors | 10 | 10 | 10 |
| nlist | 1024 | 1024 | 1024 |
| nprobe | 32 | 32 | 32 |
| seed | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 39.3 | 39.3..39.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 38.3 | 38.3..38.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| faiss-cpu | Xq | - | 42.8 | 42.8..42.8 | 1 | 0.918 | 0.896 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: search(queries)(Xq)

inference call, ours-fast: search(queries)(Xq)

inference call, faiss-cpu: search(queries)(Xq)

### jl-min-dim / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.jl-min-dim.synthetic.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 5.5 | 5.5..5.5 | 1 | - | - | - | 62.3 | - | equal_fraction_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 5.4 | 5.4..5.4 | 1 | - | - | - | 63.1 | - | equal_fraction_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 169.9 | 169.9..169.9 | 1 | 0.032 | 0.032 | - | 147.6 | - | equal_fraction_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | sklearn (declared) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### kbins / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.kbins.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1415.9 | 1415.9..1415.9 | 1 | - | - | - | 5091.7 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x220, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 275.0 | 275.0..275.0 | 1 | - | - | - | 5982.6 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x220, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 3823.4 | 3823.4..3823.4 | 1 | 0.370 | 0.072 | - | 1415.6 | - | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 64.6 | 64.6..64.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 62.0 | 62.0..62.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 154.6 | 154.6..154.6 | 1 | 0.418 | 0.401 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### kbins / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.kbins.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 185.0 | 185.0..185.0 | 1 | - | - | - | 577.0 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x11, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 24.8 | 24.8..24.8 | 1 | - | - | - | 612.6 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x11, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 108.4 | 108.4..108.4 | 1 | 1.707 | 0.229 | - | 213.2 | - | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 3.1 | 3.1..3.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 3.1 | 3.1..3.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 7.1 | 7.1..7.1 | 1 | 0.443 | 0.433 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### kernel-pca / istella (rows full, shape X 10000x220; Xq 10000x220; y 10000; yq 10000)

race: done, driver rc 0, log `logs/algos.kernel-pca.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1020.9 | 1020.9..1020.9 | 1 | - | - | - | 2636.9 | - | subspace_cos_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1003.4 | 1003.4..1003.4 | 1 | - | - | - | 2660.3 | - | subspace_cos_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 559.7 | 559.7..559.7 | 1 | 1.824 | 1.793 | - | 1309.2 | - | subspace_cos_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 316.8 | 316.8..316.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 348.0 | 348.0..348.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 167.5 | 167.5..167.5 | 1 | 1.892 | 2.078 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### kernel-pca / taxi (rows full, shape X 10000x11; Xq 10000x11; y 10000; yq 10000)

race: done, driver rc 0, log `logs/algos.kernel-pca.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 999.0 | 999.0..999.0 | 1 | - | - | - | 1728.6 | - | subspace_cos_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 952.7 | 952.7..952.7 | 1 | - | - | - | 1773.8 | - | subspace_cos_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 460.5 | 460.5..460.5 | 1 | 2.169 | 2.069 | - | 397.6 | - | subspace_cos_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 143.5 | 143.5..143.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 156.8 | 156.8..156.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 166.5 | 166.5..166.5 | 1 | 0.862 | 0.941 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### kernel-shap / istella (rows full, shape X 100000x220; Xq 100x220; y 100000; yq 100)

race: done, driver rc 0, log `logs/algos.kernel-shap.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 12538.6 | 12538.6..12538.6 | 1 | - | - | - | 1374.1 | - | rel_error_vs_exact=4.179e-09 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 12407.5 | 12407.5..12407.5 | 1 | - | - | - | 1367.8 | - | rel_error_vs_exact=4.179e-09 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| shap-cpu | shap | cpu | opponent | 7695.5 | 7695.5..7695.5 | 1 | 1.629 | 1.612 | - | 1597.7 | - | rel_error_vs_exact=1.405e-14 | - | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, shap-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'l1_reg': False, 'link': 'identity', 'n_background': 100, 'nsamples': 2048}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | shap-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | shap (declared) |
| seed | 7 | 7 | 7 |

### kernel-shap / taxi (rows full, shape X 100000x11; Xq 100x11; y 100000; yq 100)

race: done, driver rc 0, log `logs/algos.kernel-shap.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 364.9 | 364.9..364.9 | 1 | - | - | - | 181.8 | - | rel_error_vs_exact=1.731e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 361.6 | 361.6..361.6 | 1 | - | - | - | 180.6 | - | rel_error_vs_exact=1.731e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| shap-cpu | shap | cpu | opponent | 920.5 | 920.5..920.5 | 1 | 0.396 | 0.393 | - | 320.2 | - | rel_error_vs_exact=6.394e-14 | - | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, shap-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'l1_reg': False, 'link': 'identity', 'n_background': 100, 'nsamples': 2048}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | shap-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | shap (declared) |
| seed | 7 | 7 | 7 |

### knn-imputer / istella (rows full, shape X 100000x220; X_true 100000x220; Xq 20000x220; Xq_true 20000x220; y 100000; yq 20000)

race: done, driver rc 0, log `logs/algos.knn-imputer.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 836.5 | 836.5..836.5 | 1 | - | - | - | 2621.7 | - | masked_rmse=323953.237332 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 820.8 | 820.8..820.8 | 1 | - | - | - | 2625.5 | - | masked_rmse=323953.237332 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
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
| mojolearn IDENTICAL | Xq | - | 120540.9 | 120540.9..120540.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 121065.9 | 121065.9..121065.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(timeout: null) |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### kpss / synthetic (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc 0, log `logs/algos.kpss.synthetic.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 4.6 | 4.6..4.6 | 1 | - | - | - | 328.4 | - | flag_agreement_vs_statsmodels=1.000000, stat_max_rel_diff_vs_statsmodels=4.318e-05, stationary_fraction=0.031250 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 3.3 | 3.3..3.3 | 1 | - | - | - | 328.3 | - | flag_agreement_vs_statsmodels=1.000000, stat_max_rel_diff_vs_statsmodels=4.062e-05, stationary_fraction=0.031250 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsmodels-cpu | statsmodels | cpu | opponent | 2.6 | 2.6..2.6 | 1 | 1.739 | 1.255 | - | 162.3 | - | flag_agreement_vs_statsmodels=1.000000, stat_max_rel_diff_vs_statsmodels=0.000000, stationary_fraction=0.031250 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, statsmodels-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'D': 0, 'd': 0, 'pval_threshold': 0.05, 's': 0}. Rows: None. Timed: None.

mismatch: statsmodels interpolates the p-value in its table and computes in float64; ours decides against cuML's table in float32

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | statsmodels-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | statsmodels (declared) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### kpss / taxi-hourly (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc 0, log `logs/algos.kpss.taxi-hourly.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3.4 | 3.4..3.4 | 1 | - | - | - | 326.3 | - | flag_agreement_vs_statsmodels=1.000000, stat_max_rel_diff_vs_statsmodels=3.946e-05, stationary_fraction=0.687500 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 4.2 | 4.2..4.2 | 1 | - | - | - | 328.9 | - | flag_agreement_vs_statsmodels=1.000000, stat_max_rel_diff_vs_statsmodels=3.946e-05, stationary_fraction=0.687500 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsmodels-cpu | statsmodels | cpu | opponent | 2.6 | 2.6..2.6 | 1 | 1.295 | 1.627 | - | 161.9 | - | flag_agreement_vs_statsmodels=1.000000, stat_max_rel_diff_vs_statsmodels=0.000000, stationary_fraction=0.687500 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, statsmodels-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'D': 0, 'd': 0, 'pval_threshold': 0.05, 's': 0}. Rows: None. Timed: None.

mismatch: statsmodels interpolates the p-value in its table and computes in float64; ours decides against cuML's table in float32

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | statsmodels-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | statsmodels (declared) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### label-binarizer / istella (rows full, shape X 1000000x8; Xq 100000x8; lab 1000000; labq 100000; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.label-binarizer.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 325.7 | 325.7..325.7 | 1 | - | - | - | 807.9 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x16, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 324.7 | 324.7..324.7 | 1 | - | - | - | 789.9 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x16, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 31.7 | 31.7..31.7 | 1 | 10.285 | 10.254 | - | 612.6 | - | output_shape=100000x16 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 16.9 | 16.9..16.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 17.3 | 17.3..17.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 2.3 | 2.3..2.3 | 1 | 7.224 | 7.373 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### label-binarizer / taxi (rows full, shape X 1000000x5; Xq 100000x5; lab 1000000; labq 100000; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.label-binarizer.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1365.0 | 1365.0..1365.0 | 1 | - | - | - | 5735.2 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x259, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1366.0 | 1366.0..1366.0 | 1 | - | - | - | 5729.7 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x259, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 117.5 | 117.5..117.5 | 1 | 11.613 | 11.622 | - | 6713.7 | - | output_shape=100000x259 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 72.0 | 72.0..72.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 73.4 | 73.4..73.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 4.0 | 4.0..4.0 | 1 | 18.142 | 18.504 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### label-encoder / istella (rows full, shape X 1000000x8; Xq 100000x8; lab 1000000; labq 100000; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.label-encoder.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 293.4 | 293.4..293.4 | 1 | - | - | - | 489.8 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 302.7 | 302.7..302.7 | 1 | - | - | - | 482.5 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 22.5 | 22.5..22.5 | 1 | 13.014 | 13.428 | - | 240.6 | - | output_shape=100000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 15.4 | 15.4..15.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 15.4 | 15.4..15.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 1.1 | 1.1..1.1 | 1 | 14.629 | 14.622 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### label-encoder / taxi (rows full, shape X 1000000x5; Xq 100000x5; lab 1000000; labq 100000; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.label-encoder.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 300.6 | 300.6..300.6 | 1 | - | - | - | 459.7 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 307.0 | 307.0..307.0 | 1 | - | - | - | 461.2 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 29.0 | 29.0..29.0 | 1 | 10.348 | 10.568 | - | 226.6 | - | output_shape=100000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 16.0 | 16.0..16.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 16.3 | 16.3..16.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 1.5 | 1.5..1.5 | 1 | 10.886 | 11.095 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### label-propagation / istella (rows full, shape X 200000x220; Xq 20000x220; y 200000; y_semi 200000; yq 20000)

race: failed, driver rc 1, log `logs/algos.label-propagation.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(timeout: null) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(timeout: null) |
| sklearn-cpu | scikit-learn | cpu | opponent | 14551.5 | 14551.5..14551.5 | 1 | - | - | - | 1368.5 | - | accuracy=0.905500 | - | LIKE-FOR-LIKE-SPAN | - | ok (stored (measured 2026-09-30T16:39:22Z on ip-172-31-43-215.ec2.internal, cpu (Apple M3 Ultra))) |

memory, ours, ours-fast: host not sampled; GPU not sampled

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'gamma': 20, 'kernel': 'knn', 'max_iter': 1000, 'n_neighbors': 7, 'tol': 0.001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast |
|---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) |
| gamma | 20 | 20 |
| kernel | "knn" | "knn" |
| max_iter | 1000 | 1000 |
| n_neighbors | 7 | 7 |
| seed | "none (deterministic)" | "none (deterministic)" |
| tol | 0.001 | 0.001 |

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

race: done, driver rc 0, log `logs/algos.label-propagation.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 8294.7 | 8294.7..8294.7 | 1 | - | - | - | 534.0 | - | accuracy=0.701600 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 5156.6 | 5156.6..5156.6 | 1 | - | - | - | 501.3 | - | accuracy=0.701600 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 10366.0 | 10366.0..10366.0 | 1 | 0.800 | 0.497 | - | 328.6 | - | accuracy=0.701600 | - | LIKE-FOR-LIKE-SPAN | - | ok (stored (measured 2026-09-30T16:25:59Z on ip-172-31-43-215.ec2.internal, cpu (Apple M3 Ultra))) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'gamma': 20, 'kernel': 'knn', 'max_iter': 1000, 'n_neighbors': 7, 'tol': 0.001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast |
|---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) |
| gamma | 20 | 20 |
| kernel | "knn" | "knn" |
| max_iter | 1000 | 1000 |
| n_neighbors | 7 | 7 |
| seed | "none (deterministic)" | "none (deterministic)" |
| tol | 0.001 | 0.001 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 358.7 | 358.7..358.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 333.1 | 333.1..333.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 1004.6 | 1004.6..1004.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### label-spreading / istella (rows full, shape X 200000x220; Xq 20000x220; y 200000; y_semi 200000; yq 20000)

race: failed, driver rc 1, log `logs/algos.label-spreading.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(timeout: null) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(timeout: null) |
| sklearn-cpu | scikit-learn | cpu | opponent | 10273.3 | 10273.3..10273.3 | 1 | - | - | - | 1395.3 | - | accuracy=0.904450 | - | LIKE-FOR-LIKE-SPAN | - | ok (stored (measured 2026-09-30T16:53:38Z on ip-172-31-43-215.ec2.internal, cpu (Apple M3 Ultra))) |

memory, ours, ours-fast: host not sampled; GPU not sampled

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 0.2, 'gamma': 20, 'kernel': 'knn', 'max_iter': 30, 'n_neighbors': 7, 'tol': 0.001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast |
|---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) |
| alpha | 0.2 | 0.2 |
| gamma | 20 | 20 |
| kernel | "knn" | "knn" |
| max_iter | 30 | 30 |
| n_neighbors | 7 | 7 |
| seed | "none (deterministic)" | "none (deterministic)" |
| tol | 0.001 | 0.001 |

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

race: done, driver rc 0, log `logs/algos.label-spreading.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 7720.1 | 7720.1..7720.1 | 1 | - | - | - | 549.1 | - | accuracy=0.676400 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 9853.7 | 9853.7..9853.7 | 1 | - | - | - | 496.4 | - | accuracy=0.676400 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 6647.9 | 6647.9..6647.9 | 1 | 1.161 | 1.482 | - | 360.0 | - | accuracy=0.676400 | - | LIKE-FOR-LIKE-SPAN | - | ok (stored (measured 2026-09-30T16:40:20Z on ip-172-31-43-215.ec2.internal, cpu (Apple M3 Ultra))) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 0.2, 'gamma': 20, 'kernel': 'knn', 'max_iter': 30, 'n_neighbors': 7, 'tol': 0.001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast |
|---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) |
| alpha | 0.2 | 0.2 |
| gamma | 20 | 20 |
| kernel | "knn" | "knn" |
| max_iter | 30 | 30 |
| n_neighbors | 7 | 7 |
| seed | "none (deterministic)" | "none (deterministic)" |
| tol | 0.001 | 0.001 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 1193.1 | 1193.1..1193.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 877.0 | 877.0..877.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 959.8 | 959.8..959.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### lamb / synthetic (rows full, shape -)

race: failed, driver rc 1, log `logs/algos.lamb.synthetic.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

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

race: done, driver rc 0, log `logs/algos.layernorm.synthetic.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 66.7 | 66.7..66.7 | 1 | - | - | - | 1423.5 | - | max_rel_diff_vs_torch_eager_fp32=0.089719, rel_fro_vs_torch_eager_fp32=2.264e-07 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 68.3 | 68.3..68.3 | 1 | - | - | - | 1408.0 | - | max_rel_diff_vs_torch_eager_fp32=0.089719, rel_fro_vs_torch_eager_fp32=2.264e-07 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 15.1 | 15.1..15.1 | 1 | 4.429 | 4.536 | - | 1721.3 | 1160.8 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 58.0 | 58.0..58.0 | 1 | 1.150 | 1.178 | - | 1696.2 | 1032.5 | max_rel_diff_vs_torch_eager_fp32=0.066421, rel_fro_vs_torch_eager_fp32=2.25e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 2.8 | 2.8..2.8 | 1 | 24.179 | 24.759 | - | 1722.2 | 1160.8 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 50.5 | 50.5..50.5 | 1 | 1.320 | 1.352 | - | 1688.5 | 1032.5 | max_rel_diff_vs_torch_eager_fp32=0.066421, rel_fro_vs_torch_eager_fp32=2.25e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 25.8 | 25.8..25.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 23.3 | 23.3..23.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | 0.5 | 0.5..0.5 | 1 | 52.772 | 47.647 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 48.2 | 48.2..48.2 | 1 | 0.536 | 0.484 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 0.5 | 0.5..0.5 | 1 | 56.963 | 51.430 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 48.2 | 48.2..48.2 | 1 | 0.536 | 0.484 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, ours-fast: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### lda-clf / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.lda-clf.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 22219.4 | 22219.4..22219.4 | 1 | - | - | - | 4642.5 | - | accuracy=0.911660, logloss=0.247483 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 19239.8 | 19239.8..19239.8 | 1 | - | - | - | 4645.7 | - | accuracy=0.909010, logloss=0.264395 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 3739.1 | 3739.1..3739.1 | 1 | 5.942 | 5.146 | - | 5478.0 | - | accuracy=0.901130, logloss=0.449237 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 39.4 | 39.4..39.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 39.5 | 39.5..39.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 8.3 | 8.3..8.3 | 1 | 4.722 | 4.729 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### lda-clf / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.lda-clf.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 295.8 | 295.8..295.8 | 1 | - | - | - | 570.5 | - | accuracy=0.762580, logloss=0.539749 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 88.5 | 88.5..88.5 | 1 | - | - | - | 566.2 | - | accuracy=0.762580, logloss=0.539743 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 185.5 | 185.5..185.5 | 1 | 1.594 | 0.477 | - | 453.5 | - | accuracy=0.762530, logloss=0.539767 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 4.9 | 4.9..4.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 4.8 | 4.8..4.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 1.7 | 1.7..1.7 | 1 | 2.873 | 2.851 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### lda / taxi-zones (rows full, shape X 129352x261; Xq 14373x261)

race: done, driver rc 0, log `logs/algos.lda.taxi-zones.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 41933.6 | 41933.6..41933.6 | 1 | - | - | - | 6365.4 | - | perplexity=45.221819 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 38694.7 | 38694.7..38694.7 | 1 | - | - | - | 6359.6 | - | perplexity=45.221810 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 20926.2 | 20926.2..20926.2 | 1 | 2.004 | 1.849 | - | 330.3 | - | perplexity=44.897929 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 619.2 | 619.2..619.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 642.6 | 642.6..642.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 255.3 | 255.3..255.3 | 1 | 2.426 | 2.517 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### lda / text (rows full, shape X 86626x4096; Xq 9626x4096; y 86626; yq 9626)

race: done, driver rc 0, log `logs/algos.lda.text.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(timeout: null) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(timeout: null) |
| sklearn-cpu | scikit-learn | cpu | opponent | 72657.3 | 72657.3..72657.3 | 1 | - | - | - | 1695.9 | - | perplexity=266.944425 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| sklearn-cpu | Xq | - | 631.7 | 631.7..631.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### lion / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.lion.synthetic.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 338.1 | 338.1..338.1 | 1 | - | - | - | 1961.7 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 300.1 | 300.1..300.1 | 1 | - | - | - | 1991.3 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |

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

race: done, driver rc 0, log `logs/algos.lle.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 35252.0 | 35252.0..35252.0 | 1 | - | - | - | 4084.7 | - | trustworthiness_k15=0.872427 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 34710.3 | 34710.3..34710.3 | 1 | - | - | - | 4087.7 | - | trustworthiness_k15=0.872427 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 2852.0 | 2852.0..2852.0 | 1 | 12.361 | 12.171 | - | 385.7 | - | trustworthiness_k15=0.849140 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

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

race: done, driver rc 0, log `logs/algos.lle.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 35197.1 | 35197.1..35197.1 | 1 | - | - | - | 4030.4 | - | trustworthiness_k15=0.839814 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 34658.5 | 34658.5..34658.5 | 1 | - | - | - | 4049.6 | - | trustworthiness_k15=0.839814 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1246.8 | 1246.8..1246.8 | 1 | 28.230 | 27.798 | - | 232.6 | - | trustworthiness_k15=0.770758 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

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

### logreg-cv / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: failed, driver rc 1, log `logs/algos.logreg-cv.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(timeout: null) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(timeout: null) |
| sklearn-cpu | scikit-learn | cpu | opponent | 72840.3 | 72840.3..72840.3 | 1 | - | - | - | 2981.6 | - | accuracy=0.924630, logloss=0.181351 | - | LIKE-FOR-LIKE-SPAN | - | ok (stored (measured 2026-09-30T11:26:40Z on ip-172-31-43-215.ec2.internal, cpu (Apple M3 Ultra))) |

memory, ours, ours-fast: host not sampled; GPU not sampled

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'Cs': [0.1, 1.0, 10.0], 'cv': 5, 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1.0, 'max_iter': 1000, 'penalty': 'l2', 'random_state': 7, 'refit': True, 'solver': 'lbfgs', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast |
|---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) |
| class_weight | null | null |
| cv | 5 | 5 |
| fit_intercept | true | true |
| max_iter | 1000 | 1000 |
| penalty | "l2" | "l2" |
| seed | 7 | 7 |
| solver | "lbfgs" | "lbfgs" |
| tol | 0.0001 | 0.0001 |

accepted difference: ours-fast class_weight: None on both ours and ours-fast: the same documented setting in both signatures

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(timeout: null) |
| mojolearn FAST | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(timeout: null) |
| sklearn-cpu | Xq | - | 24.3 | 24.3..24.3 | 1 | - | 2.354 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### logreg-cv / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.logreg-cv.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(timeout: null) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(timeout: null) |
| sklearn-cpu | scikit-learn | cpu | opponent | 1758.0 | 1758.0..1758.0 | 1 | - | - | - | 339.1 | - | accuracy=0.763300, logloss=0.538988 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host not sampled; GPU not sampled

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
| mojolearn FAST | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(timeout: null) |
| sklearn-cpu | Xq | - | 2.7 | 2.7..2.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### louvain / istella (rows full, shape X 20000x220; indices 304830; indices2 62848; indptr 20001; indptr2 20001; y 20000)

race: done, driver rc 0, log `logs/algos.louvain.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 59614.6 | 59614.6..59614.6 | 1 | - | - | - | 22917.5 | - | modularity=0.909755, n_communities=39 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 41320.8 | 41320.8..41320.8 | 1 | - | - | - | 22914.6 | - | modularity=0.909755, n_communities=39 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| networkx-cpu | networkx | cpu | opponent | 1376.1 | 1376.1..1376.1 | 1 | 43.320 | 30.026 | - | 276.4 | - | modularity=0.908460, n_communities=40 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.louvain.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 58664.2 | 58664.2..58664.2 | 1 | - | - | - | 23025.0 | - | modularity=0.941172, n_communities=58 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 39876.6 | 39876.6..39876.6 | 1 | - | - | - | 22773.3 | - | modularity=0.941172, n_communities=58 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| networkx-cpu | networkx | cpu | opponent | 780.8 | 780.8..780.8 | 1 | 75.131 | 51.069 | - | 194.3 | - | modularity=0.940781, n_communities=56 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

### lr-constant / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.lr-constant.synthetic.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 552.4 | 552.4..552.4 | 1 | - | - | - | 67.0 | - | max_rel_diff_vs_ours=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested identical, read back unknown) |
| mojolearn FAST | mojolearn | gpu | fast | 563.2 | 563.2..563.2 | 1 | - | - | - | 66.6 | - | max_rel_diff_vs_ours=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested fast, read back unknown) |
| torch-cpu | torch | cpu | opponent | 87.4 | 87.4..87.4 | 1 | - | - | - | 331.7 | - | max_rel_diff_vs_ours=1.038e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'peak_lr': 0.001, 'warmup_steps': 1000}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | torch-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | torch (declared) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### lr-exponential / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.lr-exponential.synthetic.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 227.8 | 227.8..227.8 | 1 | - | - | - | 67.4 | - | max_rel_diff_vs_ours=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 225.0 | 225.0..225.0 | 1 | - | - | - | 65.5 | - | max_rel_diff_vs_ours=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-cpu | torch | cpu | opponent | 76.7 | 76.7..76.7 | 1 | 2.968 | 2.931 | - | 335.5 | - | max_rel_diff_vs_ours=5.933e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'base_lr': 0.1, 'gamma': 0.9999}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | torch-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | torch (declared) |
| gamma | 0.9999 | 0.9999 | 0.9999 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### lr-onecycle / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.lr-onecycle.synthetic.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1374.8 | 1374.8..1374.8 | 1 | - | - | - | 65.7 | - | max_rel_diff_vs_ours=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1364.8 | 1364.8..1364.8 | 1 | - | - | - | 64.8 | - | max_rel_diff_vs_ours=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-cpu | torch | cpu | opponent | 110.8 | 110.8..110.8 | 1 | 12.410 | 12.319 | - | 333.5 | - | max_rel_diff_vs_ours=5.951e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'anneal_strategy': 'cos', 'div_factor': 25.0, 'final_div_factor': 10000.0, 'max_lr': 0.1, 'pct_start': 0.3, 'three_phase': False, 'total_steps': 100000}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | torch-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | torch (declared) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### lr-step / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.lr-step.synthetic.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 12.0 | 12.0..12.0 | 1 | - | - | - | 64.2 | - | max_rel_diff_vs_ours=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 11.8 | 11.8..11.8 | 1 | - | - | - | 63.4 | - | max_rel_diff_vs_ours=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-cpu | torch | cpu | opponent | 85.1 | 85.1..85.1 | 1 | 0.141 | 0.139 | - | 331.6 | - | max_rel_diff_vs_ours=1.49e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'base_lr': 0.1, 'gamma': 0.5, 'step_size': 1000}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | torch-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | torch (declared) |
| gamma | 0.5 | 0.5 | 0.5 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### lr-warmup-cosine / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.lr-warmup-cosine.synthetic.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested identical, read back unknown) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested fast, read back unknown) |
| torch-cpu | torch | cpu | opponent | 113.6 | 113.6..113.6 | 1 | - | - | - | 332.1 | - | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host not sampled; GPU not sampled

memory, torch-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'min_lr': 1e-05, 'peak_lr': 0.001, 'total_steps': 100000, 'warmup_steps': 1000}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | torch-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | torch (declared) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### lr-warmup-linear / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.lr-warmup-linear.synthetic.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 775.8 | 775.8..775.8 | 1 | - | - | - | 66.4 | - | max_rel_diff_vs_ours=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested identical, read back unknown) |
| mojolearn FAST | mojolearn | gpu | fast | 785.7 | 785.7..785.7 | 1 | - | - | - | 65.0 | - | max_rel_diff_vs_ours=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested fast, read back unknown) |
| torch-cpu | torch | cpu | opponent | 101.5 | 101.5..101.5 | 1 | - | - | - | 329.5 | - | max_rel_diff_vs_ours=0.001000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'min_lr': 1e-05, 'peak_lr': 0.001, 'total_steps': 100000, 'warmup_steps': 1000}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | torch-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | torch (declared) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### lstm-clf / synthetic (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc 0, log `logs/algos.lstm-clf.synthetic.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 2003.9 | 2003.9..2003.9 | 1 | - | - | - | 743.9 | - | accuracy=0.968696, logloss=0.072441 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1896.7 | 1896.7..1896.7 | 1 | - | - | - | 742.2 | - | accuracy=0.968696, logloss=0.072441 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
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
| mojolearn IDENTICAL | Xq | - | 112.1 | 112.1..112.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 104.9 | 104.9..104.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
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

race: done, driver rc 0, log `logs/algos.lstm-clf.taxi-hourly.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 2032.2 | 2032.2..2032.2 | 1 | - | - | - | 748.2 | - | accuracy=0.868218, logloss=0.299901 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1938.3 | 1938.3..1938.3 | 1 | - | - | - | 748.7 | - | accuracy=0.868218, logloss=0.299901 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
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
| mojolearn IDENTICAL | Xq | - | 111.5 | 111.5..111.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 108.9 | 108.9..108.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
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

race: done, driver rc 0, log `logs/algos.lstm-reg.synthetic.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1986.0 | 1986.0..1986.0 | 1 | - | - | - | 745.0 | - | finite=True, r2=0.981013, rmse=0.159641 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1890.1 | 1890.1..1890.1 | 1 | - | - | - | 742.8 | - | finite=True, r2=0.981013, rmse=0.159641 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
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
| mojolearn IDENTICAL | Xq | - | 55.3 | 55.3..55.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 51.7 | 51.7..51.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
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

race: done, driver rc 0, log `logs/algos.lstm-reg.taxi-hourly.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1995.3 | 1995.3..1995.3 | 1 | - | - | - | 743.1 | - | finite=True, r2=0.751679, rmse=0.540429 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1890.9 | 1890.9..1890.9 | 1 | - | - | - | 742.3 | - | finite=True, r2=0.751679, rmse=0.540429 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
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
| mojolearn IDENTICAL | Xq | - | 55.1 | 55.1..55.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 52.4 | 52.4..52.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
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

race: done, driver rc 0, log `logs/algos.lstsq.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 25627.5 | 25627.5..25627.5 | 1 | - | - | - | 4388.7 | - | relative_residual=0.849957 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 24586.0 | 24586.0..24586.0 | 1 | - | - | - | 4367.1 | - | relative_residual=0.849956 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| numpy-cpu | numpy | cpu | opponent | 3086.4 | 3086.4..3086.4 | 1 | 8.303 | 7.966 | - | 4363.4 | - | relative_residual=0.873341 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
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

race: done, driver rc 0, log `logs/algos.lstsq.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 67.7 | 67.7..67.7 | 1 | - | - | - | 698.0 | - | relative_residual=0.756366 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 69.4 | 69.4..69.4 | 1 | - | - | - | 698.1 | - | relative_residual=0.756366 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| numpy-cpu | numpy | cpu | opponent | 52.2 | 52.2..52.2 | 1 | 1.295 | 1.328 | - | 273.1 | - | relative_residual=0.756366 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
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

race: done, driver rc 0, log `logs/algos.lu-factor.synthetic.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 24997.3 | 24997.3..24997.3 | 1 | - | - | - | 2433.0 | - | relative_residual=3.249e-06 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 21655.9 | 21655.9..21655.9 | 1 | - | - | - | 2170.3 | - | relative_residual=3.249e-06 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| scipy-cpu | scipy | cpu | opponent | 428.0 | 428.0..428.0 | 1 | 58.404 | 50.597 | - | 404.0 | - | relative_residual=3.246e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-gpu | torch | gpu | opponent | 3045.5 | 3045.5..3045.5 | 1 | 8.208 | 7.111 | - | 1755.8 | 1040.8 | relative_residual=8.234e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.lu-solve.synthetic.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 22122.1 | 22122.1..22122.1 | 1 | - | - | - | 2442.5 | - | relative_residual=3.249e-06 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 21692.7 | 21692.7..21692.7 | 1 | - | - | - | 2293.1 | - | relative_residual=3.249e-06 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| numpy-cpu | numpy | cpu | opponent | 1023.4 | 1023.4..1023.4 | 1 | 21.617 | 21.197 | - | 843.0 | - | relative_residual=3.259e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-gpu | torch | gpu | opponent | 505.5 | 505.5..505.5 | 1 | 43.764 | 42.915 | - | 1750.2 | 1032.5 | relative_residual=8.234e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.maxabs-scaler.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 226.1 | 226.1..226.1 | 1 | - | - | - | 3349.3 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x220, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 134.7 | 134.7..134.7 | 1 | - | - | - | 3349.1 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x220, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 99.7 | 99.7..99.7 | 1 | 2.269 | 1.352 | - | 2249.8 | - | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 69.0 | 69.0..69.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 59.3 | 59.3..59.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 6.9 | 6.9..6.9 | 1 | 10.050 | 8.641 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### maxabs-scaler / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.maxabs-scaler.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 119.3 | 119.3..119.3 | 1 | - | - | - | 480.6 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x11, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 11.6 | 11.6..11.6 | 1 | - | - | - | 481.4 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x11, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 12.4 | 12.4..12.4 | 1 | 9.640 | 0.936 | - | 254.5 | - | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 3.5 | 3.5..3.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 2.6 | 2.6..2.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 1.1 | 1.1..1.1 | 1 | 3.253 | 2.414 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### maxpool1d / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.maxpool1d.synthetic.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 62.9 | 62.9..62.9 | 1 | - | - | - | 953.6 | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 63.5 | 63.5..63.5 | 1 | - | - | - | 953.9 | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 10.9 | 10.9..10.9 | 1 | 5.744 | 5.801 | - | 1613.7 | 1056.5 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 2.1 | 2.1..2.1 | 1 | 30.125 | 30.424 | - | 1680.7 | 1024.5 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 3.2 | 3.2..3.2 | 1 | 19.932 | 20.130 | - | 1615.3 | 1056.5 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 2.0 | 2.0..2.0 | 1 | 30.851 | 31.157 | - | 1677.5 | 1024.5 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

settings: {'ceil_mode': False, 'dilation': 1, 'kernel_size': 2, 'padding': 0, 'stride': 2}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | "none (deterministic)" | 7 | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 28.4 | 28.4..28.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 28.2 | 28.2..28.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | 1.3 | 1.3..1.3 | 1 | 21.899 | 21.686 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 1.1 | 1.1..1.1 | 1 | 26.519 | 26.261 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 0.7 | 0.7..0.7 | 1 | 39.648 | 39.262 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 1.1 | 1.1..1.1 | 1 | 24.780 | 24.539 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, ours-fast: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### maxpool2d / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.maxpool2d.synthetic.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 153.7 | 153.7..153.7 | 1 | - | - | - | 1620.4 | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 155.7 | 155.7..155.7 | 1 | - | - | - | 1620.6 | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 13.1 | 13.1..13.1 | 1 | 11.760 | 11.907 | - | 1785.6 | 1074.5 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 12.5 | 12.5..12.5 | 1 | 12.324 | 12.478 | - | 1869.9 | 1074.5 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 4.2 | 4.2..4.2 | 1 | 36.745 | 37.203 | - | 1785.3 | 1074.5 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 4.6 | 4.6..4.6 | 1 | 33.439 | 33.857 | - | 1868.7 | 1074.5 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

settings: {'ceil_mode': False, 'dilation': 1, 'kernel_size': 2, 'padding': 0, 'stride': 2}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | "none (deterministic)" | 7 | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 50.7 | 50.7..50.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 50.2 | 50.2..50.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | 1.4 | 1.4..1.4 | 1 | 36.254 | 35.918 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 1.3 | 1.3..1.3 | 1 | 38.987 | 38.625 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 0.7 | 0.7..0.7 | 1 | 69.014 | 68.374 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 0.7 | 0.7..0.7 | 1 | 71.104 | 70.445 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, ours-fast: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### mb-dict-learning / istella (rows full, shape X 100000x220; Xq 5000x220; y 100000; yq 5000)

race: done, driver rc 0, log `logs/algos.mb-dict-learning.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 16175.3 | 16175.3..16175.3 | 1 | - | - | - | 2211.9 | - | component_sparsity=0.086364, relative_reconstruction_error=0.648338 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 15249.0 | 15249.0..15249.0 | 1 | - | - | - | 2143.3 | - | component_sparsity=0.086364, relative_reconstruction_error=0.648339 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 5030.3 | 5030.3..5030.3 | 1 | 3.216 | 3.031 | - | 1314.0 | - | component_sparsity=0.086364, relative_reconstruction_error=0.644013 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 50.0 | 50.0..50.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 143.3 | 143.3..143.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 32.9 | 32.9..32.9 | 1 | 1.523 | 4.362 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### mb-dict-learning / taxi (rows full, shape X 100000x11; Xq 5000x11; y 100000; yq 5000)

race: done, driver rc 0, log `logs/algos.mb-dict-learning.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3752.9 | 3752.9..3752.9 | 1 | - | - | - | 466.3 | - | component_sparsity=0.000000, relative_reconstruction_error=0.496685 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 3589.8 | 3589.8..3589.8 | 1 | - | - | - | 467.4 | - | component_sparsity=0.000000, relative_reconstruction_error=0.496684 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 5292.6 | 5292.6..5292.6 | 1 | 0.709 | 0.678 | - | 254.4 | - | component_sparsity=0.000000, relative_reconstruction_error=0.477049 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 32.5 | 32.5..32.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 35.1 | 35.1..35.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 50.0 | 50.0..50.0 | 1 | 0.650 | 0.702 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### mb-sparse-pca / istella (rows full, shape X 100000x220; Xq 20000x220; y 100000; yq 20000)

race: done, driver rc 0, log `logs/algos.mb-sparse-pca.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 9817.4 | 9817.4..9817.4 | 1 | - | - | - | 2894.1 | - | component_sparsity=0.130682, relative_reconstruction_error=0.705389 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 9041.0 | 9041.0..9041.0 | 1 | - | - | - | 2905.0 | - | component_sparsity=0.130682, relative_reconstruction_error=0.705389 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 2005.9 | 2005.9..2005.9 | 1 | 4.894 | 4.507 | - | 1589.7 | - | component_sparsity=0.130682, relative_reconstruction_error=0.705389 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 10.7 | 10.7..10.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 10.6 | 10.6..10.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 8.5 | 8.5..8.5 | 1 | 1.263 | 1.252 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### mb-sparse-pca / taxi (rows full, shape X 100000x11; Xq 20000x11; y 100000; yq 20000)

race: done, driver rc 0, log `logs/algos.mb-sparse-pca.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 231.4 | 231.4..231.4 | 1 | - | - | - | 539.3 | - | component_sparsity=0.022727, relative_reconstruction_error=0.275935 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 238.3 | 238.3..238.3 | 1 | - | - | - | 528.0 | - | component_sparsity=0.022727, relative_reconstruction_error=0.275935 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 2223.3 | 2223.3..2223.3 | 1 | 0.104 | 0.107 | - | 266.9 | - | component_sparsity=0.022727, relative_reconstruction_error=0.275933 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 9.1 | 9.1..9.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 9.8 | 9.8..9.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 2.4 | 2.4..2.4 | 1 | 3.750 | 4.023 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### mds / istella (rows full, shape X 5000x220; Xq 2000x220)

race: done, driver rc 0, log `logs/algos.mds.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 608.3 | 608.3..608.3 | 1 | - | - | - | 1640.8 | - | trustworthiness_k15=0.586415 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 602.0 | 602.0..602.0 | 1 | - | - | - | 1640.5 | - | trustworthiness_k15=0.586415 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 2666.3 | 2666.3..2666.3 | 1 | 0.228 | 0.226 | - | 417.2 | - | trustworthiness_k15=0.580239 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

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

race: done, driver rc 0, log `logs/algos.mds.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 522.1 | 522.1..522.1 | 1 | - | - | - | 1607.5 | - | trustworthiness_k15=0.606536 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 522.3 | 522.3..522.3 | 1 | - | - | - | 1607.1 | - | trustworthiness_k15=0.606536 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 2631.5 | 2631.5..2631.5 | 1 | 0.198 | 0.198 | - | 392.9 | - | trustworthiness_k15=0.604142 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

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

race: done, driver rc 0, log `logs/algos.min-cov-det.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(timeout: null) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "Exception(\"eigh: the Jacobi eigensolver did not converge in 15 sweeps at n = 220. An unconverged decomposition is not returned as if it were one; see DEVIATION 590. The remedy is more swee) |
| sklearn-cpu | scikit-learn | cpu | opponent | 43631.7 | 43631.7..43631.7 | 1 | - | - | - | 4279.1 | - | n_features=220 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.min-cov-det.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3985.2 | 3985.2..3985.2 | 1 | - | - | - | 531.3 | - | n_features=11, rel_diff_vs_sklearn=0.205603 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1760.6 | 1760.6..1760.6 | 1 | - | - | - | 545.8 | - | n_features=11, rel_diff_vs_sklearn=0.205603 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1413.4 | 1413.4..1413.4 | 1 | 2.819 | 1.246 | - | 305.1 | - | n_features=11 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.minmax-scaler.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 346.8 | 346.8..346.8 | 1 | - | - | - | 4285.1 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x220, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 357.1 | 357.1..357.1 | 1 | - | - | - | 4287.1 | - | max_abs_diff_vs_sklearn=1.192e-07, output_shape=100000x220, rel_diff_vs_sklearn=1.694e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 63.0 | 63.0..63.0 | 1 | 5.502 | 5.665 | - | 1409.5 | - | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 70.4 | 70.4..70.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 72.7 | 72.7..72.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 9.6 | 9.6..9.6 | 1 | 7.348 | 7.581 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### minmax-scaler / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.minmax-scaler.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 20.8 | 20.8..20.8 | 1 | - | - | - | 524.4 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x11, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 20.8 | 20.8..20.8 | 1 | - | - | - | 524.9 | - | max_abs_diff_vs_sklearn=5.96e-08, output_shape=100000x11, rel_diff_vs_sklearn=2.63e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 17.6 | 17.6..17.6 | 1 | 1.180 | 1.185 | - | 207.5 | - | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 4.0 | 4.0..4.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 4.1 | 4.1..4.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 1.7 | 1.7..1.7 | 1 | 2.305 | 2.371 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### mlp-clf / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.mlp-clf.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 11697.4 | 11697.4..11697.4 | 1 | - | - | - | 3051.0 | - | accuracy=0.944270, logloss=0.136955 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 10933.7 | 10933.7..10933.7 | 1 | - | - | - | 3054.8 | - | accuracy=0.944560, logloss=0.136389 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 14327.0 | 14327.0..14327.0 | 1 | 0.816 | 0.763 | - | 1304.4 | - | accuracy=0.943760, logloss=0.136831 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 177.7 | 177.7..177.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 169.9 | 169.9..169.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 75.2 | 75.2..75.2 | 1 | 2.364 | 2.260 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### mlp-clf / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.mlp-clf.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 8752.4 | 8752.4..8752.4 | 1 | - | - | - | 569.1 | - | accuracy=0.767810, logloss=0.530506 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 8090.2 | 8090.2..8090.2 | 1 | - | - | - | 566.3 | - | accuracy=0.767750, logloss=0.530490 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 11431.0 | 11431.0..11431.0 | 1 | 0.766 | 0.708 | - | 428.5 | - | accuracy=0.767830, logloss=0.530444 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 86.0 | 86.0..86.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 83.3 | 83.3..83.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 67.3 | 67.3..67.3 | 1 | 1.277 | 1.237 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### mlp-reg / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.mlp-reg.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 11674.8 | 11674.8..11674.8 | 1 | - | - | - | 3044.5 | - | finite=True, r2=0.526405, rmse=0.574862 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 10911.1 | 10911.1..10911.1 | 1 | - | - | - | 3045.4 | - | finite=True, r2=0.527265, rmse=0.574340 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 13650.9 | 13650.9..13650.9 | 1 | 0.855 | 0.799 | - | 1301.4 | - | finite=True, r2=0.524815, rmse=0.575826 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 90.2 | 90.2..90.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 87.9 | 87.9..87.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 38.5 | 38.5..38.5 | 1 | 2.341 | 2.282 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### mlp-reg / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.mlp-reg.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 8729.3 | 8729.3..8729.3 | 1 | - | - | - | 562.2 | - | finite=True, r2=0.931981, rmse=4.153868 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 8067.6 | 8067.6..8067.6 | 1 | - | - | - | 559.9 | - | finite=True, r2=0.931976, rmse=4.154000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 9958.9 | 9958.9..9958.9 | 1 | 0.877 | 0.810 | - | 410.8 | - | finite=True, r2=0.929613, rmse=4.225537 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 44.1 | 44.1..44.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 42.1 | 42.1..42.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 35.1 | 35.1..35.1 | 1 | 1.254 | 1.199 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### moe / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.moe.synthetic.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 968.0 | 968.0..968.0 | 1 | - | - | - | 1907.8 | - | max_rel_diff_vs_torch_eager_fp32=0.059674, rel_fro_vs_torch_eager_fp32=3.083e-07 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 959.3 | 959.3..959.3 | 1 | - | - | - | 1893.7 | - | max_rel_diff_vs_torch_eager_fp32=0.062871, rel_fro_vs_torch_eager_fp32=2.845e-07 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 39.4 | 39.4..39.4 | 1 | 24.600 | 24.378 | - | 1777.9 | 1050.8 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 39.9 | 39.9..39.9 | 1 | 24.273 | 24.054 | - | 1961.4 | 1050.8 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 37.7 | 37.7..37.7 | 1 | 25.674 | 25.442 | - | 1778.6 | 1050.8 | max_rel_diff_vs_torch_eager_fp32=22341.757794, rel_fro_vs_torch_eager_fp32=0.055222 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 38.4 | 38.4..38.4 | 1 | 25.217 | 24.990 | - | 1964.8 | 1050.8 | max_rel_diff_vs_torch_eager_fp32=22341.757794, rel_fro_vs_torch_eager_fp32=0.055222 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 964.9 | 964.9..964.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 963.8 | 963.8..963.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | 28.5 | 28.5..28.5 | 1 | 33.905 | 33.868 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 28.1 | 28.1..28.1 | 1 | 34.322 | 34.285 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 27.0 | 27.0..27.0 | 1 | 35.776 | 35.738 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 27.5 | 27.5..27.5 | 1 | 35.135 | 35.097 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, ours-fast: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### multilabel-binarizer / istella (rows full, shape X 200000x8; Xq 20000x8; lab 200000; labq 20000; y 200000; yq 20000)

race: done, driver rc 0, log `logs/algos.multilabel-binarizer.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 759.8 | 759.8..759.8 | 1 | - | - | - | 1290.8 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=20000x119, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 756.7 | 756.7..756.7 | 1 | - | - | - | 1275.4 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=20000x119, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 143.5 | 143.5..143.5 | 1 | 5.294 | 5.273 | - | 989.2 | - | output_shape=20000x119 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 36.3 | 36.3..36.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 36.4 | 36.4..36.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 16.0 | 16.0..16.0 | 1 | 2.260 | 2.268 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### multilabel-binarizer / taxi (rows full, shape X 200000x5; Xq 20000x5; lab 200000; labq 20000; y 200000; yq 20000)

race: done, driver rc 0, log `logs/algos.multilabel-binarizer.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 716.9 | 716.9..716.9 | 1 | - | - | - | 2669.5 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=20000x489, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 715.3 | 715.3..715.3 | 1 | - | - | - | 2657.5 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=20000x489, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 129.4 | 129.4..129.4 | 1 | 5.538 | 5.526 | - | 2823.9 | - | output_shape=20000x489 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 34.3 | 34.3..34.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 36.1 | 36.1..36.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 11.5 | 11.5..11.5 | 1 | 2.989 | 3.151 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### multinomial-nb / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.multinomial-nb.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 312.4 | 312.4..312.4 | 1 | - | - | - | 3981.2 | - | accuracy=0.853560, logloss=3.630944 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 148.9 | 148.9..148.9 | 1 | - | - | - | 3964.1 | - | accuracy=0.853620, logloss=3.628575 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 183.4 | 183.4..183.4 | 1 | 1.704 | 0.812 | - | 3803.2 | - | accuracy=0.853620, logloss=3.087499 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 30.2 | 30.2..30.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 31.0 | 31.0..31.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 44.7 | 44.7..44.7 | 1 | 0.675 | 0.693 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### multinomial-nb / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.multinomial-nb.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 209.0 | 209.0..209.0 | 1 | - | - | - | 530.6 | - | accuracy=0.723260, logloss=0.589979 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 21.6 | 21.6..21.6 | 1 | - | - | - | 525.5 | - | accuracy=0.723160, logloss=0.590725 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 40.8 | 40.8..40.8 | 1 | 5.129 | 0.531 | - | 382.0 | - | accuracy=0.723160, logloss=0.590725 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 4.6 | 4.6..4.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 6.8 | 6.8..6.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 10.1 | 10.1..10.1 | 1 | 0.461 | 0.671 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### multinomial-nb / text (rows full, shape X 86626x4096; Xq 9626x4096; y 86626; yq 9626)

race: done, driver rc 0, log `logs/algos.multinomial-nb.text.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 202.2 | 202.2..202.2 | 1 | - | - | - | 4542.0 | - | accuracy=0.983067, logloss=0.559529 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 248.7 | 248.7..248.7 | 1 | - | - | - | 4542.3 | - | accuracy=0.983067, logloss=0.559529 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 265.7 | 265.7..265.7 | 1 | 0.761 | 0.936 | - | 4370.6 | - | accuracy=0.983067, logloss=0.557319 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 50.9 | 50.9..50.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 51.5 | 51.5..51.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 65.5 | 65.5..65.5 | 1 | 0.777 | 0.786 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### multioutput-clf / istella (rows full, shape X 1000000x219; Xq 100000x219; Y 1000000x2; Yq 100000x2; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.multioutput-clf.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 2160.5 | 2160.5..2160.5 | 1 | - | - | - | 3494.5 | - | accuracy=0.959110 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 2182.5 | 2182.5..2182.5 | 1 | - | - | - | 3485.8 | - | accuracy=0.959175 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 22339.7 | 22339.7..22339.7 | 1 | 0.097 | 0.098 | - | 3866.3 | - | accuracy=0.959225 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'estimator': 'LogisticRegression(max_iter=200)'}. Rows: None. Timed: None.

mismatch: nested LogisticRegression(max_iter=200): ours is L-BFGS (cuML's quasi-Newton, memory 5, solver='qn'), scikit-learn solver='lbfgs' (scipy, memory 10); C=1.0 and tol=1e-4 are both libraries' defaults; each stops by its own criterion

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 42.2 | 42.2..42.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 44.1 | 44.1..44.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 22.8 | 22.8..22.8 | 1 | 1.847 | 1.933 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### multioutput-clf / taxi (rows full, shape X 1000000x10; Xq 100000x10; Y 1000000x2; Yq 100000x2; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.multioutput-clf.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 227.9 | 227.9..227.9 | 1 | - | - | - | 701.6 | - | accuracy=0.863560 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 213.2 | 213.2..213.2 | 1 | - | - | - | 648.8 | - | accuracy=0.863560 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 686.0 | 686.0..686.0 | 1 | 0.332 | 0.311 | - | 340.2 | - | accuracy=0.863550 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'estimator': 'LogisticRegression(max_iter=200)'}. Rows: None. Timed: None.

mismatch: nested LogisticRegression(max_iter=200): ours is L-BFGS (cuML's quasi-Newton, memory 5, solver='qn'), scikit-learn solver='lbfgs' (scipy, memory 10); C=1.0 and tol=1e-4 are both libraries' defaults; each stops by its own criterion

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 21.3 | 21.3..21.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 23.3 | 23.3..23.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 3.4 | 3.4..3.4 | 1 | 6.332 | 6.948 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### multioutput-reg / istella (rows full, shape X 1000000x219; Xq 100000x219; Y 1000000x2; Yq 100000x2; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.multioutput-reg.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1720.2 | 1720.2..1720.2 | 1 | - | - | - | 6368.8 | - | r2=0.455327 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1777.8 | 1777.8..1777.8 | 1 | - | - | - | 6368.8 | - | r2=0.439854 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 13401.5 | 13401.5..13401.5 | 1 | 0.128 | 0.133 | - | 9554.7 | - | r2=0.455345 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'estimator': 'Ridge(alpha=1.0)'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 21.7 | 21.7..21.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 38.9 | 38.9..38.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 7.5 | 7.5..7.5 | 1 | 2.899 | 5.199 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### multioutput-reg / taxi (rows full, shape X 1000000x10; Xq 100000x10; Y 1000000x2; Yq 100000x2; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.multioutput-reg.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 100.1 | 100.1..100.1 | 1 | - | - | - | 711.5 | - | r2=0.604240 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 102.9 | 102.9..102.9 | 1 | - | - | - | 706.0 | - | r2=0.604241 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 53.7 | 53.7..53.7 | 1 | 1.865 | 1.917 | - | 291.8 | - | r2=0.604316 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'estimator': 'Ridge(alpha=1.0)'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 6.3 | 6.3..6.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 5.9 | 5.9..5.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 0.6 | 0.6..0.6 | 1 | 10.065 | 9.386 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### nadam / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.nadam.synthetic.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 411.3 | 411.3..411.3 | 1 | - | - | - | 2200.6 | - | rel_fro_vs_torch_eager_fp32=2.939e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 417.9 | 417.9..417.9 | 1 | - | - | - | 2201.8 | - | rel_fro_vs_torch_eager_fp32=2.939e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 32.9 | 32.9..32.9 | 1 | 12.509 | 12.712 | - | 1919.3 | 1032.5 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 121.9 | 121.9..121.9 | 1 | 3.373 | 3.427 | - | 1975.8 | 1024.5 | rel_fro_vs_torch_eager_fp32=4.44e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

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

### nmf / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.nmf.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 29313.7 | 29313.7..29313.7 | 1 | - | - | - | 5892.1 | - | relative_reconstruction_error=0.325174 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 28223.4 | 28223.4..28223.4 | 1 | - | - | - | 5871.5 | - | relative_reconstruction_error=0.325174 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 7059.1 | 7059.1..7059.1 | 1 | 4.153 | 3.998 | - | 3711.0 | - | relative_reconstruction_error=0.325402 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 172.9 | 172.9..172.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 175.5 | 175.5..175.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 95.4 | 95.4..95.4 | 1 | 1.814 | 1.840 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### nmf / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.nmf.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 959.1 | 959.1..959.1 | 1 | - | - | - | 1186.9 | - | relative_reconstruction_error=0.091156 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 931.4 | 931.4..931.4 | 1 | - | - | - | 1165.0 | - | relative_reconstruction_error=0.091156 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 3419.4 | 3419.4..3419.4 | 1 | 0.280 | 0.272 | - | 487.8 | - | relative_reconstruction_error=0.091155 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 32.3 | 32.3..32.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 30.8 | 30.8..30.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 59.4 | 59.4..59.4 | 1 | 0.543 | 0.518 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### normalizer / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.normalizer.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 0.1 | 0.1..0.1 | 1 | - | - | - | 2006.4 | - | max_abs_diff_vs_sklearn=3.576e-07, output_shape=100000x220, rel_diff_vs_sklearn=5.811e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 0.1 | 0.1..0.1 | 1 | - | - | - | 2006.2 | - | max_abs_diff_vs_sklearn=3.576e-07, output_shape=100000x220, rel_diff_vs_sklearn=5.674e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 25.3 | 25.3..25.3 | 1 | 0.004 | 0.004 | - | 1410.4 | - | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 55.9 | 55.9..55.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 54.6 | 54.6..54.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 11.6 | 11.6..11.6 | 1 | 4.805 | 4.693 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### normalizer / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.normalizer.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 0.1 | 0.1..0.1 | 1 | - | - | - | 409.0 | - | max_abs_diff_vs_sklearn=1.192e-07, output_shape=100000x11, rel_diff_vs_sklearn=3.114e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 0.1 | 0.1..0.1 | 1 | - | - | - | 410.5 | - | max_abs_diff_vs_sklearn=1.192e-07, output_shape=100000x11, rel_diff_vs_sklearn=3.283e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1.6 | 1.6..1.6 | 1 | 0.063 | 0.058 | - | 212.5 | - | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 3.9 | 3.9..3.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 3.4 | 3.4..3.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 1.6 | 1.6..1.6 | 1 | 2.528 | 2.190 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### onehot / istella (rows full, shape X 1000000x8; Xq 100000x8; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.onehot.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 109.9 | 109.9..109.9 | 1 | - | - | - | 751.6 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x119, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 81.3 | 81.3..81.3 | 1 | - | - | - | 772.3 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x119, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 36.7 | 36.7..36.7 | 1 | 2.994 | 2.214 | - | 468.2 | - | output_shape=100000x119 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 21.6 | 21.6..21.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 22.7 | 22.7..22.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 25.1 | 25.1..25.1 | 1 | 0.863 | 0.907 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### onehot / taxi (rows full, shape X 1000000x5; Xq 100000x5; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.onehot.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 97.7 | 97.7..97.7 | 1 | - | - | - | 1542.0 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x508, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 74.1 | 74.1..74.1 | 1 | - | - | - | 1530.3 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x508, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 19.0 | 19.0..19.0 | 1 | 5.140 | 3.899 | - | 1339.8 | - | output_shape=100000x508 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 110.8 | 110.8..110.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 109.6 | 109.6..109.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 30.0 | 30.0..30.0 | 1 | 3.690 | 3.650 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### optimized-theta / synthetic (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc 0, log `logs/algos.optimized-theta.synthetic.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 405.6 | 405.6..405.6 | 1 | - | - | - | 345.1 | - | forecast_rmse=1.438855 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 320.7 | 320.7..320.7 | 1 | - | - | - | 344.9 | - | forecast_rmse=1.439709 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsforecast-cpu | statsforecast | cpu | opponent | 748.5 | 748.5..748.5 | 1 | 0.542 | 0.428 | - | 190.7 | - | forecast_rmse=1.437815 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.optimized-theta.taxi-hourly.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 721.7 | 721.7..721.7 | 1 | - | - | - | 343.4 | - | forecast_rmse=49.150860 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 586.8 | 586.8..586.8 | 1 | - | - | - | 343.2 | - | forecast_rmse=49.152331 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsforecast-cpu | statsforecast | cpu | opponent | 1290.5 | 1290.5..1290.5 | 1 | 0.559 | 0.455 | - | 189.6 | - | forecast_rmse=49.356608 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.ordinal.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 109.8 | 109.8..109.8 | 1 | - | - | - | 575.6 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x8, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 78.9 | 78.9..78.9 | 1 | - | - | - | 592.6 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x8, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 36.9 | 36.9..36.9 | 1 | 2.980 | 2.139 | - | 207.1 | - | output_shape=100000x8 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 8.0 | 8.0..8.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 7.9 | 7.9..7.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 18.1 | 18.1..18.1 | 1 | 0.439 | 0.436 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### ordinal / taxi (rows full, shape X 1000000x5; Xq 100000x5; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.ordinal.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 82.5 | 82.5..82.5 | 1 | - | - | - | 489.5 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x5, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 67.3 | 67.3..67.3 | 1 | - | - | - | 497.6 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x5, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 18.8 | 18.8..18.8 | 1 | 4.394 | 3.586 | - | 185.7 | - | output_shape=100000x5 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 7.2 | 7.2..7.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 7.1 | 7.1..7.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 10.4 | 10.4..10.4 | 1 | 0.695 | 0.681 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### ovr / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.ovr.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 5056.7 | 5056.7..5056.7 | 1 | - | - | - | 2991.8 | - | accuracy=0.892670 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 5272.7 | 5272.7..5272.7 | 1 | - | - | - | 3022.2 | - | accuracy=0.892700 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 75801.5 | 75801.5..75801.5 | 1 | 0.067 | 0.070 | - | 2968.8 | - | accuracy=0.892730 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'estimator': 'LogisticRegression(max_iter=200)'}. Rows: None. Timed: None.

mismatch: nested LogisticRegression(max_iter=200): ours is L-BFGS (cuML's quasi-Newton, memory 5, solver='qn'), scikit-learn solver='lbfgs' (scipy, memory 10); C=1.0 and tol=1e-4 are both libraries' defaults; each stops by its own criterion

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 110.8 | 110.8..110.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 111.5 | 111.5..111.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 87.2 | 87.2..87.2 | 1 | 1.271 | 1.279 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### ovr / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.ovr.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 386.2 | 386.2..386.2 | 1 | - | - | - | 530.6 | - | accuracy=0.478940 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 471.6 | 471.6..471.6 | 1 | - | - | - | 510.3 | - | accuracy=0.478940 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1461.0 | 1461.0..1461.0 | 1 | 0.264 | 0.323 | - | 337.0 | - | accuracy=0.478890 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'estimator': 'LogisticRegression(max_iter=200)'}. Rows: None. Timed: None.

mismatch: nested LogisticRegression(max_iter=200): ours is L-BFGS (cuML's quasi-Newton, memory 5, solver='qn'), scikit-learn solver='lbfgs' (scipy, memory 10); C=1.0 and tol=1e-4 are both libraries' defaults; each stops by its own criterion

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 34.4 | 34.4..34.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 33.2 | 33.2..33.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 9.8 | 9.8..9.8 | 1 | 3.501 | 3.384 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### pagerank / istella (rows full, shape X 20000x220; indices 304830; indices2 62848; indptr 20001; indptr2 20001; y 20000)

race: done, driver rc 0, log `logs/algos.pagerank.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 652.3 | 652.3..652.3 | 1 | - | - | - | 8042.1 | - | l1_vs_networkx=5.89e-08, sum=1.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 643.2 | 643.2..643.2 | 1 | - | - | - | 8043.4 | - | l1_vs_networkx=6.429e-08, sum=1.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| networkx-cpu | networkx | cpu | opponent | 82.0 | 82.0..82.0 | 1 | 7.955 | 7.844 | - | 202.4 | - | sum=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.pagerank.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 644.9 | 644.9..644.9 | 1 | - | - | - | 7701.9 | - | l1_vs_networkx=6.917e-08, sum=1.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 707.0 | 707.0..707.0 | 1 | - | - | - | 7601.8 | - | l1_vs_networkx=8.411e-08, sum=1.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| networkx-cpu | networkx | cpu | opponent | 64.9 | 64.9..64.9 | 1 | 9.940 | 10.896 | - | 170.0 | - | sum=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

### permutation-shap / istella (rows full, shape X 100000x220; Xq 100x220; y 100000; yq 100)

race: done, driver rc 0, log `logs/algos.permutation-shap.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 14549.2 | 14549.2..14549.2 | 1 | - | - | - | 2014.8 | - | rel_error_vs_exact=5.339e-09 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 14681.8 | 14681.8..14681.8 | 1 | - | - | - | 2017.6 | - | rel_error_vs_exact=5.339e-09 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| shap-cpu | shap | cpu | opponent | 12334.7 | 12334.7..12334.7 | 1 | 1.180 | 1.190 | - | 1727.8 | - | rel_error_vs_exact=3.692e-10 | - | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, shap-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'n_background': 100, 'npermutations': 10}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | shap-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | shap (declared) |
| seed | 7 | 7 | 7 |

### permutation-shap / taxi (rows full, shape X 100000x11; Xq 100x11; y 100000; yq 100)

race: done, driver rc 0, log `logs/algos.permutation-shap.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 52.8 | 52.8..52.8 | 1 | - | - | - | 151.8 | - | rel_error_vs_exact=2.15e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 53.2 | 53.2..53.2 | 1 | - | - | - | 151.3 | - | rel_error_vs_exact=2.15e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| shap-cpu | shap | cpu | opponent | 83.0 | 83.0..83.0 | 1 | 0.636 | 0.641 | - | 353.3 | - | rel_error_vs_exact=1.279e-15 | - | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, shap-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'n_background': 100, 'npermutations': 10}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | shap-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | shap (declared) |
| seed | 7 | 7 | 7 |

### permutation-test / istella (rows full, shape X 20000x220; Xq 20000x220; y 20000; yq 20000)

race: done, driver rc 0, log `logs/algos.permutation-test.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested identical, read back unknown) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested fast, read back unknown) |
| scipy-cpu | scipy | cpu | opponent | 3041.5 | 3041.5..3041.5 | 1 | - | - | - | 1387.0 | - | pvalue=0.184400, statistic=0.011200 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.permutation-test.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested identical, read back unknown) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested fast, read back unknown) |
| scipy-cpu | scipy | cpu | opponent | 3055.1 | 3055.1..3055.1 | 1 | - | - | - | 477.4 | - | pvalue=0.001000, statistic=-0.576708 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.pls-canonical.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3312.8 | 3312.8..3312.8 | 1 | - | - | - | 4843.1 | - | mean_canonical_corr=0.875340 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 3410.7 | 3410.7..3410.7 | 1 | - | - | - | 4905.0 | - | mean_canonical_corr=0.875340 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 4064.7 | 4064.7..4064.7 | 1 | 0.815 | 0.839 | - | 3844.9 | - | mean_canonical_corr=0.875341 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 39.8 | 39.8..39.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 39.6 | 39.6..39.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 57.3 | 57.3..57.3 | 1 | 0.695 | 0.691 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### pls-canonical / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.pls-canonical.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 127.9 | 127.9..127.9 | 1 | - | - | - | 852.5 | - | mean_canonical_corr=0.559206 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 124.5 | 124.5..124.5 | 1 | - | - | - | 850.4 | - | mean_canonical_corr=0.559206 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 268.7 | 268.7..268.7 | 1 | 0.476 | 0.463 | - | 439.0 | - | mean_canonical_corr=0.559206 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 5.8 | 5.8..5.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 7.5 | 7.5..7.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 7.2 | 7.2..7.2 | 1 | 0.806 | 1.033 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### pls / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.pls.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 569.1 | 569.1..569.1 | 1 | - | - | - | 5566.2 | - | finite=True, r2=0.289870, rmse=0.703930 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 566.0 | 566.0..566.0 | 1 | - | - | - | 5573.5 | - | finite=True, r2=0.289870, rmse=0.703930 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 2017.9 | 2017.9..2017.9 | 1 | 0.282 | 0.280 | - | 4813.5 | - | finite=True, r2=0.289870, rmse=0.703930 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 22.0 | 22.0..22.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 22.3 | 22.3..22.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 33.4 | 33.4..33.4 | 1 | 0.657 | 0.667 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### pls / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.pls.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 121.8 | 121.8..121.8 | 1 | - | - | - | 919.2 | - | finite=True, r2=0.905216, rmse=4.903469 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 121.1 | 121.1..121.1 | 1 | - | - | - | 918.7 | - | finite=True, r2=0.905216, rmse=4.903469 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 242.5 | 242.5..242.5 | 1 | 0.502 | 0.499 | - | 509.4 | - | finite=True, r2=0.905216, rmse=4.903467 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 2.2 | 2.2..2.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 2.2 | 2.2..2.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 2.4 | 2.4..2.4 | 1 | 0.935 | 0.913 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### poly-features / istella (rows full, shape X 1000000x16; Xq 100000x16; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.poly-features.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 0.5 | 0.5..0.5 | 1 | - | - | - | 1684.2 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x152, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 0.3 | 0.3..0.3 | 1 | - | - | - | 1681.8 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x152, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 2.1 | 2.1..2.1 | 1 | 0.219 | 0.143 | - | 1378.7 | - | output_shape=100000x152 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 28.2 | 28.2..28.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 29.1 | 29.1..29.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 30.5 | 30.5..30.5 | 1 | 0.925 | 0.954 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### poly-features / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.poly-features.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 0.2 | 0.2..0.2 | 1 | - | - | - | 560.2 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x77, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 0.2 | 0.2..0.2 | 1 | - | - | - | 563.7 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x77, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1.6 | 1.6..1.6 | 1 | 0.135 | 0.147 | - | 312.7 | - | output_shape=100000x77 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 12.2 | 12.2..12.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 12.5 | 12.5..12.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 19.4 | 19.4..19.4 | 1 | 0.627 | 0.645 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### power-transformer / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.power-transformer.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 366.2 | 366.2..366.2 | 1 | - | - | - | 4788.1 | - | max_abs_diff_vs_sklearn=353.551971, output_shape=100000x220, rel_diff_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 2925.8 | 2925.8..2925.8 | 1 | - | - | - | 8153.8 | - | max_abs_diff_vs_sklearn=5.248894, output_shape=100000x220, rel_diff_vs_sklearn=0.108987 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 276766.7 | 276766.7..276766.7 | 1 | 0.001 | 0.011 | - | 4092.4 | - | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 36.9 | 36.9..36.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 60.2 | 60.2..60.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 236.4 | 236.4..236.4 | 1 | 0.156 | 0.255 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### power-transformer / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.power-transformer.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 2726.2 | 2726.2..2726.2 | 1 | - | - | - | 1080.5 | - | max_abs_diff_vs_sklearn=0.191760, output_shape=100000x11, rel_diff_vs_sklearn=0.025551 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 481.8 | 481.8..481.8 | 1 | - | - | - | 754.2 | - | max_abs_diff_vs_sklearn=0.062769, output_shape=100000x11, rel_diff_vs_sklearn=0.0005187 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 13227.7 | 13227.7..13227.7 | 1 | 0.206 | 0.036 | - | 523.4 | - | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 4.2 | 4.2..4.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 4.1 | 4.1..4.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 10.5 | 10.5..10.5 | 1 | 0.394 | 0.388 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### prophet / synthetic (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc 0, log `logs/algos.prophet.synthetic.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 16646.0 | 16646.0..16646.0 | 1 | - | - | - | 413.8 | - | forecast_rmse=1.015150 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 14986.1 | 14986.1..14986.1 | 1 | - | - | - | 386.5 | - | forecast_rmse=1.015207 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| prophet-cpu | prophet | cpu | opponent | 452.3 | 452.3..452.3 | 1 | 36.805 | 33.135 | - | 122.9 | - | forecast_rmse=1.015319 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.prophet.taxi-hourly.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 19960.5 | 19960.5..19960.5 | 1 | - | - | - | 342.3 | - | forecast_rmse=32.049281 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 16747.6 | 16747.6..16747.6 | 1 | - | - | - | 373.0 | - | forecast_rmse=32.033729 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| prophet-cpu | prophet | cpu | opponent | 548.8 | 548.8..548.8 | 1 | 36.371 | 30.516 | - | 123.1 | - | forecast_rmse=32.035281 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.qda.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 19071.7 | 19071.7..19071.7 | 1 | - | - | - | 2957.5 | - | accuracy=0.866090, logloss=4.052873 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 16299.6 | 16299.6..16299.6 | 1 | - | - | - | 2962.3 | - | accuracy=0.866090, logloss=4.052902 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 6397.0 | 6397.0..6397.0 | 1 | 2.981 | 2.548 | - | 8748.0 | - | accuracy=0.880530, logloss=3.476976 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 191.6 | 191.6..191.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 176.2 | 176.2..176.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 109.3 | 109.3..109.3 | 1 | 1.753 | 1.612 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### qda / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.qda.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 239.4 | 239.4..239.4 | 1 | - | - | - | 482.4 | - | accuracy=0.727020, logloss=1.061003 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 188.2 | 188.2..188.2 | 1 | - | - | - | 482.1 | - | accuracy=0.727020, logloss=1.061003 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 140.6 | 140.6..140.6 | 1 | 1.703 | 1.338 | - | 479.5 | - | accuracy=0.727220, logloss=1.059265 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 5.3 | 5.3..5.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 5.1 | 5.1..5.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 13.0 | 13.0..13.0 | 1 | 0.403 | 0.395 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### qn-reg / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: failed, driver rc 3, log `logs/algos.qn-reg.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (1): sklearn-cpu: tol is 0.0001 (tol) on ours and 1e-06 (tol) on sklearn-cpu") |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (1): sklearn-cpu: tol is 0.0001 (tol) on ours and 1e-06 (tol) on sklearn-cpu") |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): sklearn-cpu: tol is 0.0001 (tol) on ours and 1e-06 (tol) on sklearn-cpu") (measured this run) |

settings: {'fit_intercept': True, 'l1_strength': 0.0, 'l2_strength': 0.0, 'lbfgs_memory': 5, 'linesearch_max_iter': 50, 'loss': 'squared_error', 'max_iter': 1000, 'penalty_normalized': True, 'tol': 0.0001}. Rows: None. Timed: None.

mismatch: scikit-learn LinearRegression solves the same least-squares problem in closed form (scipy lstsq); it has no max_iter, tol or L-BFGS settings

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | sklearn (get_params) |
| fit_intercept | true | true | true |
| loss | "squared_error" | "squared_error" | - |
| max_iter | 1000 | 1000 | - |
| positive | - | - | false |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| tol | 0.0001 | 0.0001 | 1e-06 |

REFUSED: sklearn-cpu: tol is 0.0001 (tol) on ours and 1e-06 (tol) on sklearn-cpu

### qn-reg / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: failed, driver rc 3, log `logs/algos.qn-reg.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (1): sklearn-cpu: tol is 0.0001 (tol) on ours and 1e-06 (tol) on sklearn-cpu") |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (1): sklearn-cpu: tol is 0.0001 (tol) on ours and 1e-06 (tol) on sklearn-cpu") |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): sklearn-cpu: tol is 0.0001 (tol) on ours and 1e-06 (tol) on sklearn-cpu") (measured this run) |

settings: {'fit_intercept': True, 'l1_strength': 0.0, 'l2_strength': 0.0, 'lbfgs_memory': 5, 'linesearch_max_iter': 50, 'loss': 'squared_error', 'max_iter': 1000, 'penalty_normalized': True, 'tol': 0.0001}. Rows: None. Timed: None.

mismatch: scikit-learn LinearRegression solves the same least-squares problem in closed form (scipy lstsq); it has no max_iter, tol or L-BFGS settings

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | sklearn (get_params) |
| fit_intercept | true | true | true |
| loss | "squared_error" | "squared_error" | - |
| max_iter | 1000 | 1000 | - |
| positive | - | - | false |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| tol | 0.0001 | 0.0001 | 1e-06 |

REFUSED: sklearn-cpu: tol is 0.0001 (tol) on ours and 1e-06 (tol) on sklearn-cpu

### qr / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.qr.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 71660.4 | 71660.4..71660.4 | 1 | - | - | - | 7102.0 | - | relative_gram_difference=0.0009027 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 71611.4 | 71611.4..71611.4 | 1 | - | - | - | 7105.3 | - | relative_gram_difference=0.0009027 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| numpy-cpu | numpy | cpu | opponent | 8839.8 | 8839.8..8839.8 | 1 | 8.107 | 8.101 | - | 8548.5 | - | relative_gram_difference=2.472e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
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

race: done, driver rc 0, log `logs/algos.qr.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3258.9 | 3258.9..3258.9 | 1 | - | - | - | 763.5 | - | relative_gram_difference=0.001996 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 3225.8 | 3225.8..3225.8 | 1 | - | - | - | 765.0 | - | relative_gram_difference=0.001996 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| numpy-cpu | numpy | cpu | opponent | 114.4 | 114.4..114.4 | 1 | 28.483 | 28.193 | - | 425.7 | - | relative_gram_difference=3.024e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
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

race: done, driver rc 0, log `logs/algos.quantile-transformer.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1504.4 | 1504.4..1504.4 | 1 | - | - | - | 5109.1 | - | max_abs_diff_vs_sklearn=5.96e-08, output_shape=100000x220, rel_diff_vs_sklearn=2.764e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 400.0 | 400.0..400.0 | 1 | - | - | - | 5996.7 | - | max_abs_diff_vs_sklearn=5.96e-08, output_shape=100000x220, rel_diff_vs_sklearn=2.764e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 5049.2 | 5049.2..5049.2 | 1 | 0.298 | 0.079 | - | 1416.8 | - | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 78.9 | 78.9..78.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 84.5 | 84.5..84.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 812.5 | 812.5..812.5 | 1 | 0.097 | 0.104 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### quantile-transformer / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.quantile-transformer.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 182.6 | 182.6..182.6 | 1 | - | - | - | 581.4 | - | max_abs_diff_vs_sklearn=5.96e-08, output_shape=100000x11, rel_diff_vs_sklearn=2.294e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 34.5 | 34.5..34.5 | 1 | - | - | - | 615.0 | - | max_abs_diff_vs_sklearn=5.96e-08, output_shape=100000x11, rel_diff_vs_sklearn=2.295e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 173.9 | 173.9..173.9 | 1 | 1.050 | 0.198 | - | 212.9 | - | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 4.5 | 4.5..4.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 3.9 | 3.9..3.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 35.5 | 35.5..35.5 | 1 | 0.126 | 0.111 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### random-trees-embedding / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.random-trees-embedding.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 255.6 | 255.6..255.6 | 1 | - | - | - | 3962.8 | - | nonzeros_per_row=10.000000, output_columns=209 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 260.3 | 260.3..260.3 | 1 | - | - | - | 3677.8 | - | nonzeros_per_row=10.000000, output_columns=209 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 624.7 | 624.7..624.7 | 1 | 0.409 | 0.417 | - | 3575.1 | - | nonzeros_per_row=10.000000, output_columns=251 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'max_depth': 5, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 10, 'random_state': 7, 'sparse_output': False}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| max_depth | 5 | 5 | 5 |
| max_leaves | null | null | null |
| min_samples_leaf | 1 | 1 | 1 |
| min_split_gain | 0.0 | 0.0 | 0.0 |
| n_estimators | 10 | 10 | 10 |
| seed | 7 | 7 | 7 |

accepted difference: ours-fast max_leaves: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: sklearn-cpu max_leaves: None on both ours and sklearn-cpu: the same documented setting in both signatures

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 17.9 | 17.9..17.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 17.9 | 17.9..17.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 73.6 | 73.6..73.6 | 1 | 0.244 | 0.243 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### random-trees-embedding / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.random-trees-embedding.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 55.8 | 55.8..55.8 | 1 | - | - | - | 1083.5 | - | nonzeros_per_row=10.000000, output_columns=292 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 52.8 | 52.8..52.8 | 1 | - | - | - | 1139.1 | - | nonzeros_per_row=10.000000, output_columns=292 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 441.5 | 441.5..441.5 | 1 | 0.126 | 0.120 | - | 2632.8 | - | nonzeros_per_row=10.000000, output_columns=244 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'max_depth': 5, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 10, 'random_state': 7, 'sparse_output': False}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| max_depth | 5 | 5 | 5 |
| max_leaves | null | null | null |
| min_samples_leaf | 1 | 1 | 1 |
| min_split_gain | 0.0 | 0.0 | 0.0 |
| n_estimators | 10 | 10 | 10 |
| seed | 7 | 7 | 7 |

accepted difference: ours-fast max_leaves: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: sklearn-cpu max_leaves: None on both ours and sklearn-cpu: the same documented setting in both signatures

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 49.6 | 49.6..49.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 41.4 | 41.4..41.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 120.9 | 120.9..120.9 | 1 | 0.410 | 0.343 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### randomized-svd / istella (rows full, shape X 900000x220; Xq 100000x220)

race: done, driver rc 0, log `logs/algos.randomized-svd.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1229.7 | 1229.7..1229.7 | 1 | - | - | - | 4183.0 | - | relative_reconstruction_error=0.0002359 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1166.7 | 1166.7..1166.7 | 1 | - | - | - | 4213.4 | - | relative_reconstruction_error=0.0002359 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 368.9 | 368.9..368.9 | 1 | 3.333 | 3.162 | - | 1284.8 | - | relative_reconstruction_error=0.000236 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
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

race: done, driver rc 0, log `logs/algos.randomized-svd.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 374.8 | 374.8..374.8 | 1 | - | - | - | 940.8 | - | relative_reconstruction_error=0.027197 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 370.3 | 370.3..370.3 | 1 | - | - | - | 938.4 | - | relative_reconstruction_error=0.027197 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 162.8 | 162.8..162.8 | 1 | 2.303 | 2.275 | - | 461.9 | - | relative_reconstruction_error=0.027197 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
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

race: done, driver rc 0, log `logs/algos.resample.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 390.6 | 390.6..390.6 | 1 | - | - | - | 4627.0 | - | max_mean_shift_over_std=0.003203 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested identical, read back unknown) |
| mojolearn FAST | mojolearn | gpu | fast | 387.4 | 387.4..387.4 | 1 | - | - | - | 4618.6 | - | max_mean_shift_over_std=0.003203 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested fast, read back unknown) |
| sklearn-cpu | scikit-learn | cpu | opponent | 335.8 | 335.8..335.8 | 1 | - | - | - | 3596.5 | - | max_mean_shift_over_std=0.002552 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.resample.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 60.4 | 60.4..60.4 | 1 | - | - | - | 566.0 | - | max_mean_shift_over_std=0.002917 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested identical, read back unknown) |
| mojolearn FAST | mojolearn | gpu | fast | 60.5 | 60.5..60.5 | 1 | - | - | - | 562.9 | - | max_mean_shift_over_std=0.002917 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested fast, read back unknown) |
| sklearn-cpu | scikit-learn | cpu | opponent | 51.9 | 51.9..51.9 | 1 | - | - | - | 331.2 | - | max_mean_shift_over_std=0.002257 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

### resnet-block / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.resnet-block.synthetic.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 450.9 | 450.9..450.9 | 1 | - | - | - | 2453.2 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 409.7 | 409.7..409.7 | 1 | - | - | - | 2455.5 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 24.1 | 24.1..24.1 | 1 | 18.738 | 17.026 | - | 1636.7 | 1088.8 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 20.2 | 20.2..20.2 | 1 | 22.310 | 20.271 | - | 1705.6 | 1036.7 | max_rel_diff_vs_torch_eager_fp32=0.357628, rel_fro_vs_torch_eager_fp32=1.857e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 18.2 | 18.2..18.2 | 1 | 24.749 | 22.488 | - | 1635.8 | 1088.8 | max_rel_diff_vs_torch_eager_fp32=18498.390913, rel_fro_vs_torch_eager_fp32=0.003425 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 10.5 | 10.5..10.5 | 1 | 43.123 | 39.182 | - | 1705.3 | 1036.7 | max_rel_diff_vs_torch_eager_fp32=18498.390913, rel_fro_vs_torch_eager_fp32=0.003425 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

settings: {'inplanes': 64, 'planes': 64}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | 7 | 7 | 7 | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 185.6 | 185.6..185.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 181.9 | 181.9..181.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | 3.8 | 3.8..3.8 | 1 | 48.794 | 47.826 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 33.5 | 33.5..33.5 | 1 | 5.547 | 5.437 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 3.3 | 3.3..3.3 | 1 | 55.551 | 54.449 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 32.8 | 32.8..32.8 | 1 | 5.654 | 5.542 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, ours-fast: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### rfe / istella (rows full, shape X 200000x220; Xq 100000x220; y 200000; yq 100000)

race: done, driver rc 0, log `logs/algos.rfe.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 2187.7 | 2187.7..2187.7 | 1 | - | - | - | 2276.8 | - | jaccard_vs_sklearn=0.818182, n_selected=110 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 2250.9 | 2250.9..2250.9 | 1 | - | - | - | 2277.7 | - | jaccard_vs_sklearn=0.880342, n_selected=110 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 16177.3 | 16177.3..16177.3 | 1 | 0.135 | 0.139 | - | 1422.8 | - | jaccard_vs_sklearn=1.000000, n_selected=110 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.rfe.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 139.9 | 139.9..139.9 | 1 | - | - | - | 436.9 | - | jaccard_vs_sklearn=1.000000, n_selected=5 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 134.1 | 134.1..134.1 | 1 | - | - | - | 437.6 | - | jaccard_vs_sklearn=1.000000, n_selected=5 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 312.9 | 312.9..312.9 | 1 | 0.447 | 0.429 | - | 269.7 | - | jaccard_vs_sklearn=1.000000, n_selected=5 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.ridge-cv.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "ValueError('mojolearn RidgeCV: only cv=None, scoring=None, alpha_per_target=False are implemented')", "event": "error", "stage": "round 0"}) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "ValueError('mojolearn RidgeCV: only cv=None, scoring=None, alpha_per_target=False are implemented')", "event": "error", "stage": "round 0"}) |
| sklearn-cpu | scikit-learn | cpu | opponent | 134463.1 | 134463.1..134463.1 | 1 | - | - | - | 8806.0 | - | finite=True, r2=0.328683, rmse=0.684423 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.ridge-cv.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "ValueError('mojolearn RidgeCV: only cv=None, scoring=None, alpha_per_target=False are implemented')", "event": "error", "stage": "round 0"}) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "ValueError('mojolearn RidgeCV: only cv=None, scoring=None, alpha_per_target=False are implemented')", "event": "error", "stage": "round 0"}) |
| sklearn-cpu | scikit-learn | cpu | opponent | 946.4 | 946.4..946.4 | 1 | - | - | - | 378.7 | - | finite=True, r2=0.908988, rmse=4.804917 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| sklearn-cpu | Xq | - | 0.3 | 0.3..0.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### rmsprop / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.rmsprop.synthetic.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 299.6 | 299.6..299.6 | 1 | - | - | - | 2075.5 | - | rel_fro_vs_torch_eager_fp32=3.772e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 304.4 | 304.4..304.4 | 1 | - | - | - | 2054.4 | - | rel_fro_vs_torch_eager_fp32=3.772e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 25.2 | 25.2..25.2 | 1 | 11.904 | 12.093 | - | 1918.8 | 1032.5 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 109.0 | 109.0..109.0 | 1 | 2.749 | 2.793 | - | 1980.5 | 1024.5 | rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.rnn-clf.synthetic.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1517.4 | 1517.4..1517.4 | 1 | - | - | - | 539.2 | - | accuracy=0.953559, logloss=0.103698 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1504.0 | 1504.0..1504.0 | 1 | - | - | - | 528.6 | - | accuracy=0.953559, logloss=0.103698 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
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
| mojolearn IDENTICAL | Xq | - | 40.6 | 40.6..40.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 38.4 | 38.4..38.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
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

race: done, driver rc 0, log `logs/algos.rnn-clf.taxi-hourly.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1531.6 | 1531.6..1531.6 | 1 | - | - | - | 534.4 | - | accuracy=0.868056, logloss=0.304864 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1514.6 | 1514.6..1514.6 | 1 | - | - | - | 539.0 | - | accuracy=0.868056, logloss=0.304864 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
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
| mojolearn IDENTICAL | Xq | - | 43.5 | 43.5..43.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 38.8 | 38.8..38.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
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

race: done, driver rc 0, log `logs/algos.rnn-reg.synthetic.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1520.0 | 1520.0..1520.0 | 1 | - | - | - | 531.7 | - | finite=True, r2=0.977348, rmse=0.174374 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1518.7 | 1518.7..1518.7 | 1 | - | - | - | 533.7 | - | finite=True, r2=0.977348, rmse=0.174374 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
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
| mojolearn IDENTICAL | Xq | - | 20.0 | 20.0..20.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 19.9 | 19.9..19.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
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

race: done, driver rc 0, log `logs/algos.rnn-reg.taxi-hourly.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1531.8 | 1531.8..1531.8 | 1 | - | - | - | 532.2 | - | finite=True, r2=0.738796, rmse=0.554271 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1531.3 | 1531.3..1531.3 | 1 | - | - | - | 532.8 | - | finite=True, r2=0.738796, rmse=0.554271 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
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
| mojolearn IDENTICAL | Xq | - | 20.1 | 20.1..20.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 21.4 | 21.4..21.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
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

race: done, driver rc 0, log `logs/algos.robust-scaler.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1490.5 | 1490.5..1490.5 | 1 | - | - | - | 5090.9 | - | max_abs_diff_vs_sklearn=0.0001221, output_shape=100000x220, rel_diff_vs_sklearn=7.339e-09 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 361.2 | 361.2..361.2 | 1 | - | - | - | 5981.1 | - | max_abs_diff_vs_sklearn=0.0001221, output_shape=100000x220, rel_diff_vs_sklearn=7.339e-09 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 4588.1 | 4588.1..4588.1 | 1 | 0.325 | 0.079 | - | 1413.1 | - | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 64.4 | 64.4..64.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 82.2 | 82.2..82.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 19.7 | 19.7..19.7 | 1 | 3.261 | 4.163 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### robust-scaler / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.robust-scaler.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 181.2 | 181.2..181.2 | 1 | - | - | - | 579.0 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x11, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 33.8 | 33.8..33.8 | 1 | - | - | - | 611.2 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x11, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 109.2 | 109.2..109.2 | 1 | 1.660 | 0.310 | - | 210.8 | - | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 4.4 | 4.4..4.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 2.9 | 2.9..2.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 1.9 | 1.9..1.9 | 1 | 2.360 | 1.542 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### select-chi2 / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.select-chi2.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 317.8 | 317.8..317.8 | 1 | - | - | - | 3959.6 | - | jaccard_vs_sklearn=0.981982, n_selected=110 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 152.0 | 152.0..152.0 | 1 | - | - | - | 3959.6 | - | jaccard_vs_sklearn=1.000000, n_selected=110 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 283.0 | 283.0..283.0 | 1 | 1.123 | 0.537 | - | 3797.9 | - | jaccard_vs_sklearn=1.000000, n_selected=110 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.select-chi2.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 208.2 | 208.2..208.2 | 1 | - | - | - | 521.8 | - | jaccard_vs_sklearn=1.000000, n_selected=5 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 21.0 | 21.0..21.0 | 1 | - | - | - | 565.4 | - | jaccard_vs_sklearn=1.000000, n_selected=5 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 58.5 | 58.5..58.5 | 1 | 3.556 | 0.358 | - | 391.2 | - | jaccard_vs_sklearn=1.000000, n_selected=5 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'k': 'half'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### select-d / synthetic (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc 0, log `logs/algos.select-d.synthetic.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 7.5 | 7.5..7.5 | 1 | - | - | - | 328.3 | - | d_agreement_vs_statsmodels=1.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 6.7 | 6.7..6.7 | 1 | - | - | - | 330.1 | - | d_agreement_vs_statsmodels=1.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsmodels-cpu | statsmodels | cpu | opponent | 5.1 | 5.1..5.1 | 1 | 1.491 | 1.318 | - | 162.0 | - | d_agreement_vs_statsmodels=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, statsmodels-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'D': 0, 'pval_threshold': 0.05, 's': 0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | statsmodels-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | statsmodels (declared) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### select-d / taxi-hourly (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc 0, log `logs/algos.select-d.taxi-hourly.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 6.3 | 6.3..6.3 | 1 | - | - | - | 330.3 | - | d_agreement_vs_statsmodels=1.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 7.1 | 7.1..7.1 | 1 | - | - | - | 332.2 | - | d_agreement_vs_statsmodels=1.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsmodels-cpu | statsmodels | cpu | opponent | 3.4 | 3.4..3.4 | 1 | 1.853 | 2.070 | - | 163.8 | - | d_agreement_vs_statsmodels=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, statsmodels-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'D': 0, 'pval_threshold': 0.05, 's': 0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | statsmodels-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | statsmodels (declared) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### select-f-classif / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.select-f-classif.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 318.7 | 318.7..318.7 | 1 | - | - | - | 2949.5 | - | jaccard_vs_sklearn=1.000000, n_selected=110 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 229.2 | 229.2..229.2 | 1 | - | - | - | 2951.0 | - | jaccard_vs_sklearn=1.000000, n_selected=110 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 294.4 | 294.4..294.4 | 1 | 1.083 | 0.779 | - | 3500.3 | - | jaccard_vs_sklearn=1.000000, n_selected=110 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.select-f-classif.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 214.5 | 214.5..214.5 | 1 | - | - | - | 475.7 | - | jaccard_vs_sklearn=1.000000, n_selected=5 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 127.1 | 127.1..127.1 | 1 | - | - | - | 522.2 | - | jaccard_vs_sklearn=1.000000, n_selected=5 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 41.1 | 41.1..41.1 | 1 | 5.221 | 3.093 | - | 327.9 | - | jaccard_vs_sklearn=1.000000, n_selected=5 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.select-f-regression.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 250.5 | 250.5..250.5 | 1 | - | - | - | 2937.5 | - | jaccard_vs_sklearn=1.000000, n_selected=110 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 208.0 | 208.0..208.0 | 1 | - | - | - | 2936.7 | - | jaccard_vs_sklearn=1.000000, n_selected=110 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 267.9 | 267.9..267.9 | 1 | 0.935 | 0.776 | - | 2764.7 | - | jaccard_vs_sklearn=1.000000, n_selected=110 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.select-f-regression.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 147.6 | 147.6..147.6 | 1 | - | - | - | 463.9 | - | jaccard_vs_sklearn=1.000000, n_selected=5 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 97.8 | 97.8..97.8 | 1 | - | - | - | 462.9 | - | jaccard_vs_sklearn=1.000000, n_selected=5 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 31.3 | 31.3..31.3 | 1 | 4.713 | 3.123 | - | 291.6 | - | jaccard_vs_sklearn=1.000000, n_selected=5 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.select-mutual-info-reg.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 69577.5 | 69577.5..69577.5 | 1 | - | - | - | 2026.4 | - | jaccard_vs_sklearn=0.929825, n_selected=110 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 46503.2 | 46503.2..46503.2 | 1 | - | - | - | 2026.1 | - | jaccard_vs_sklearn=0.929825, n_selected=110 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 50021.4 | 50021.4..50021.4 | 1 | 1.391 | 0.930 | - | 1228.5 | - | jaccard_vs_sklearn=1.000000, n_selected=110 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.select-mutual-info-reg.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 2791.9 | 2791.9..2791.9 | 1 | - | - | - | 429.9 | - | jaccard_vs_sklearn=1.000000, n_selected=5 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 2325.4 | 2325.4..2325.4 | 1 | - | - | - | 429.0 | - | jaccard_vs_sklearn=1.000000, n_selected=5 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 2714.9 | 2714.9..2714.9 | 1 | 1.028 | 0.857 | - | 255.3 | - | jaccard_vs_sklearn=1.000000, n_selected=5 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.select-mutual-info.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 805.9 | 805.9..805.9 | 1 | - | - | - | 3133.8 | - | jaccard_vs_sklearn=0.929825, n_selected=110 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 793.1 | 793.1..793.1 | 1 | - | - | - | 3137.3 | - | jaccard_vs_sklearn=0.929825, n_selected=110 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 36427.3 | 36427.3..36427.3 | 1 | 0.022 | 0.022 | - | 1220.6 | - | jaccard_vs_sklearn=1.000000, n_selected=110 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.select-mutual-info.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 225.2 | 225.2..225.2 | 1 | - | - | - | 639.9 | - | jaccard_vs_sklearn=0.428571, n_selected=5 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 209.8 | 209.8..209.8 | 1 | - | - | - | 640.9 | - | jaccard_vs_sklearn=0.428571, n_selected=5 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1786.3 | 1786.3..1786.3 | 1 | 0.126 | 0.117 | - | 248.6 | - | jaccard_vs_sklearn=1.000000, n_selected=5 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.select-r-regression.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 258.7 | 258.7..258.7 | 1 | - | - | - | 2965.1 | - | jaccard_vs_sklearn=1.000000, n_selected=110 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 202.7 | 202.7..202.7 | 1 | - | - | - | 2938.3 | - | jaccard_vs_sklearn=1.000000, n_selected=110 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 268.4 | 268.4..268.4 | 1 | 0.964 | 0.755 | - | 2764.9 | - | jaccard_vs_sklearn=1.000000, n_selected=110 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.select-r-regression.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 145.0 | 145.0..145.0 | 1 | - | - | - | 465.8 | - | jaccard_vs_sklearn=1.000000, n_selected=5 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 97.2 | 97.2..97.2 | 1 | - | - | - | 509.8 | - | jaccard_vs_sklearn=1.000000, n_selected=5 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 31.8 | 31.8..31.8 | 1 | 4.562 | 3.056 | - | 290.4 | - | jaccard_vs_sklearn=1.000000, n_selected=5 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.sgd.synthetic.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1066.8 | 1066.8..1066.8 | 1 | - | - | - | 1922.3 | - | rel_fro_vs_torch_eager_fp32=6.095e-10 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested identical, read back unknown) |
| mojolearn FAST | mojolearn | gpu | fast | 1061.8 | 1061.8..1061.8 | 1 | - | - | - | 1922.0 | - | rel_fro_vs_torch_eager_fp32=6.095e-10 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested fast, read back unknown) |
| torch-eager-fp32 | torch | gpu | opponent | 17.9 | 17.9..17.9 | 1 | - | - | - | 1900.9 | 1024.5 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 127.0 | 127.0..127.0 | 1 | - | - | - | 1965.3 | 1024.5 | rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.simple-imputer.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1412.0 | 1412.0..1412.0 | 1 | - | - | - | 6875.5 | - | masked_rmse=346849.129968, max_abs_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 298.0 | 298.0..298.0 | 1 | - | - | - | 7767.1 | - | masked_rmse=346849.129968, max_abs_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 12578.4 | 12578.4..12578.4 | 1 | 0.112 | 0.024 | - | 7665.2 | - | masked_rmse=346849.129968 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 60.9 | 60.9..60.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 62.6 | 62.6..62.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 94.2 | 94.2..94.2 | 1 | 0.646 | 0.664 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### simple-imputer / taxi (rows full, shape X 1000000x11; X_true 1000000x11; Xq 100000x11; Xq_true 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.simple-imputer.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 190.2 | 190.2..190.2 | 1 | - | - | - | 811.8 | - | masked_rmse=5.985180, max_abs_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 27.1 | 27.1..27.1 | 1 | - | - | - | 841.2 | - | masked_rmse=5.985180, max_abs_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 370.2 | 370.2..370.2 | 1 | 0.514 | 0.073 | - | 544.9 | - | masked_rmse=5.985180 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 3.2 | 3.2..3.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 2.8 | 2.8..2.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 4.3 | 4.3..4.3 | 1 | 0.740 | 0.656 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### sparse-coder / istella (rows full, shape X 5000x220; Xq 100000x220; y 5000; yq 100000)

race: done, driver rc 0, log `logs/algos.sparse-coder.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 6.7 | 6.7..6.7 | 1 | - | - | - | 5254.9 | - | max_abs_diff_vs_sklearn=6.104e-05, output_shape=100000x64, rel_diff_vs_sklearn=1.11e-07 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 6.1 | 6.1..6.1 | 1 | - | - | - | 5252.5 | - | max_abs_diff_vs_sklearn=6.104e-05, output_shape=100000x64, rel_diff_vs_sklearn=1.11e-07 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 0.0 | 0.0..0.0 | 1 | 228.000 | 208.822 | - | 1318.6 | - | output_shape=100000x64 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 185.0 | 185.0..185.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 186.9 | 186.9..186.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 3469.1 | 3469.1..3469.1 | 1 | 0.053 | 0.054 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### sparse-coder / taxi (rows full, shape X 5000x11; Xq 100000x11; y 5000; yq 100000)

race: done, driver rc 0, log `logs/algos.sparse-coder.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 5.0 | 5.0..5.0 | 1 | - | - | - | 3923.5 | - | max_abs_diff_vs_sklearn=1.049e-05, output_shape=100000x64, rel_diff_vs_sklearn=9.976e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 4.0 | 4.0..4.0 | 1 | - | - | - | 3923.9 | - | max_abs_diff_vs_sklearn=1.049e-05, output_shape=100000x64, rel_diff_vs_sklearn=9.976e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 0.0 | 0.0..0.0 | 1 | 117.529 | 93.271 | - | 350.0 | - | output_shape=100000x64 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 164.2 | 164.2..164.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 164.7 | 164.7..164.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 3443.9 | 3443.9..3443.9 | 1 | 0.048 | 0.048 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### sparse-pca / istella (rows full, shape X 20000x220; Xq 20000x220; y 20000; yq 20000)

race: done, driver rc 0, log `logs/algos.sparse-pca.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "Exception(\"the one-sided Jacobi SVD did not converge in 60 sweeps at n_cols = 220: the last sweep still performed 17 rotations against a tolerance of 9.536743e-07. The remedy is more sweep) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "Exception(\"the one-sided Jacobi SVD did not converge in 60 sweeps at n_cols = 220: the last sweep still performed 14 rotations against a tolerance of 9.536743e-07. The remedy is more sweep) |
| sklearn-cpu | scikit-learn | cpu | opponent | 9903.9 | 9903.9..9903.9 | 1 | - | - | - | 1267.2 | - | component_sparsity=0.305682, relative_reconstruction_error=0.750210 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| sklearn-cpu | Xq | - | 5.6 | 5.6..5.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### sparse-pca / taxi (rows full, shape X 20000x11; Xq 20000x11; y 20000; yq 20000)

race: done, driver rc 0, log `logs/algos.sparse-pca.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1996.0 | 1996.0..1996.0 | 1 | - | - | - | 434.0 | - | component_sparsity=0.488636, relative_reconstruction_error=0.277226 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 2004.6 | 2004.6..2004.6 | 1 | - | - | - | 430.5 | - | component_sparsity=0.488636, relative_reconstruction_error=0.277226 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 11923.2 | 11923.2..11923.2 | 1 | 0.167 | 0.168 | - | 221.4 | - | component_sparsity=0.488636, relative_reconstruction_error=0.277226 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 5.2 | 5.2..5.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 5.0 | 5.0..5.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 1.8 | 1.8..1.8 | 1 | 2.869 | 2.778 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### sparse-rp / istella (rows full, shape X 900000x220; Xq 100000x220)

race: done, driver rc 0, log `logs/algos.sparse-rp.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 135.4 | 135.4..135.4 | 1 | - | - | - | 4329.0 | - | mean_abs_distortion=1.883381 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 134.7 | 134.7..134.7 | 1 | - | - | - | 4328.4 | - | mean_abs_distortion=1.883381 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 25.3 | 25.3..25.3 | 1 | 5.357 | 5.329 | - | 1085.3 | - | mean_abs_distortion=0.474347 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 22.3 | 22.3..22.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 21.5 | 21.5..21.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 31.7 | 31.7..31.7 | 1 | 0.703 | 0.676 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### sparse-rp / taxi (rows full, shape X 900000x11; Xq 100000x11)

race: done, driver rc 0, log `logs/algos.sparse-rp.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 6.4 | 6.4..6.4 | 1 | - | - | - | 569.8 | - | mean_abs_distortion=0.147163 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 6.1 | 6.1..6.1 | 1 | - | - | - | 569.7 | - | mean_abs_distortion=0.147163 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1.8 | 1.8..1.8 | 1 | 3.555 | 3.399 | - | 201.9 | - | mean_abs_distortion=0.381016 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 3.0 | 3.0..3.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 3.1 | 3.1..3.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 1.4 | 1.4..1.4 | 1 | 2.123 | 2.204 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### spline / istella (rows full, shape X 1000000x16; Xq 100000x16; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.spline.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 166.4 | 166.4..166.4 | 1 | - | - | - | 1771.6 | - | max_abs_diff_vs_sklearn=1.788e-07, output_shape=100000x112, rel_diff_vs_sklearn=6.226e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 46.2 | 46.2..46.2 | 1 | - | - | - | 1771.7 | - | max_abs_diff_vs_sklearn=1.788e-07, output_shape=100000x112, rel_diff_vs_sklearn=6.424e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 17.6 | 17.6..17.6 | 1 | 9.461 | 2.627 | - | 1311.4 | - | output_shape=100000x112 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 27.4 | 27.4..27.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 16.4 | 16.4..16.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 80.6 | 80.6..80.6 | 1 | 0.340 | 0.203 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### spline / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.spline.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 149.5 | 149.5..149.5 | 1 | - | - | - | 693.4 | - | max_abs_diff_vs_sklearn=1.788e-07, output_shape=100000x77, rel_diff_vs_sklearn=5.888e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 29.3 | 29.3..29.3 | 1 | - | - | - | 692.5 | - | max_abs_diff_vs_sklearn=1.192e-07, output_shape=100000x77, rel_diff_vs_sklearn=6.181e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 17.6 | 17.6..17.6 | 1 | 8.473 | 1.661 | - | 319.7 | - | output_shape=100000x77 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 20.4 | 20.4..20.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 10.6 | 10.6..10.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 54.2 | 54.2..54.2 | 1 | 0.376 | 0.196 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### stacking-clf / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.stacking-clf.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 4700.3 | 4700.3..4700.3 | 1 | - | - | - | 4633.9 | - | accuracy=0.929960, logloss=0.193706 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 3755.5 | 3755.5..3755.5 | 1 | - | - | - | 4625.8 | - | accuracy=0.929970, logloss=0.193693 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 149799.2 | 149799.2..149799.2 | 1 | 0.031 | 0.025 | - | 3196.2 | - | accuracy=0.930040, logloss=0.193931 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'cv': 5, 'estimators': [['nb', 'GaussianNB()'], ['dt', 'DecisionTreeClassifier(max_depth=8, random_state=7)']], 'final_estimator': 'LogisticRegression(max_iter=200)', 'passthrough': False, 'stack_method': 'auto'}. Rows: None. Timed: None.

mismatch: nested LogisticRegression(max_iter=200): ours is L-BFGS (cuML's quasi-Newton, memory 5, solver='qn'), scikit-learn solver='lbfgs' (scipy, memory 10); C=1.0 and tol=1e-4 are both libraries' defaults; each stops by its own criterion

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| cv | 5 | 5 | 5 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 57.7 | 57.7..57.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 67.7 | 67.7..67.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 191.6 | 191.6..191.6 | 1 | 0.301 | 0.353 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### stacking-clf / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.stacking-clf.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 2395.2 | 2395.2..2395.2 | 1 | - | - | - | 714.0 | - | accuracy=0.767920, logloss=0.536361 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1045.7 | 1045.7..1045.7 | 1 | - | - | - | 712.1 | - | accuracy=0.767920, logloss=0.536364 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 7095.2 | 7095.2..7095.2 | 1 | 0.338 | 0.147 | - | 423.2 | - | accuracy=0.755330, logloss=0.547756 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'cv': 5, 'estimators': [['nb', 'GaussianNB()'], ['dt', 'DecisionTreeClassifier(max_depth=8, random_state=7)']], 'final_estimator': 'LogisticRegression(max_iter=200)', 'passthrough': False, 'stack_method': 'auto'}. Rows: None. Timed: None.

mismatch: nested LogisticRegression(max_iter=200): ours is L-BFGS (cuML's quasi-Newton, memory 5, solver='qn'), scikit-learn solver='lbfgs' (scipy, memory 10); C=1.0 and tol=1e-4 are both libraries' defaults; each stops by its own criterion

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| cv | 5 | 5 | 5 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 33.1 | 33.1..33.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 28.4 | 28.4..28.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 23.3 | 23.3..23.3 | 1 | 1.423 | 1.221 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### stacking-reg / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.stacking-reg.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | UNKNOWN(no binding path) | REFUSED(not_ready: {"error": "NotImplementedError(\"mojolearn ElasticNet: random_state is refused. It selects nothing here -- the only randomness in cdFit is the coordinate shuffle, and selection='random' is refused by ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | UNKNOWN(no binding path) | REFUSED(not_ready: {"error": "NotImplementedError(\"mojolearn ElasticNet: random_state is refused. It selects nothing here -- the only randomness in cdFit is the coordinate shuffle, and selection='random' is refused by ) |
| sklearn-cpu | scikit-learn | cpu | opponent | 153420.6 | 153420.6..153420.6 | 1 | - | - | - | 3356.6 | - | finite=True, r2=0.447644, rmse=0.620826 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host not sampled; GPU not sampled

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'cv': 5, 'estimators': [['lasso', 'Lasso(alpha=0.01, max_iter=1000, random_state=7, tol=0.001)'], ['dt', 'DecisionTreeRegressor(max_depth=8, random_state=7)']], 'final_estimator': 'Ridge(alpha=1.0)', 'passthrough': False}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `sklearn-cpu`, seed 7): MATCHED

| parameter | sklearn-cpu |
|---|---|
| library (source) | sklearn (get_params) |
| cv | 5 |
| seed | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(not_ready: {"error": "NotImplementedError(\"mojolearn ElasticNet: random_state is refused. It selects nothing here -- the only randomness in cdFit is the coordinate shuffle, and selection='random' is refused by ) |
| mojolearn FAST | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(not_ready: {"error": "NotImplementedError(\"mojolearn ElasticNet: random_state is refused. It selects nothing here -- the only randomness in cdFit is the coordinate shuffle, and selection='random' is refused by ) |
| sklearn-cpu | Xq | - | 10.1 | 10.1..10.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### stacking-reg / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.stacking-reg.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | UNKNOWN(no binding path) | REFUSED(not_ready: {"error": "NotImplementedError(\"mojolearn ElasticNet: random_state is refused. It selects nothing here -- the only randomness in cdFit is the coordinate shuffle, and selection='random' is refused by ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | UNKNOWN(no binding path) | REFUSED(not_ready: {"error": "NotImplementedError(\"mojolearn ElasticNet: random_state is refused. It selects nothing here -- the only randomness in cdFit is the coordinate shuffle, and selection='random' is refused by ) |
| sklearn-cpu | scikit-learn | cpu | opponent | 6007.5 | 6007.5..6007.5 | 1 | - | - | - | 416.7 | - | finite=True, r2=0.932532, rmse=4.137015 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host not sampled; GPU not sampled

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'cv': 5, 'estimators': [['lasso', 'Lasso(alpha=0.01, max_iter=1000, random_state=7, tol=0.001)'], ['dt', 'DecisionTreeRegressor(max_depth=8, random_state=7)']], 'final_estimator': 'Ridge(alpha=1.0)', 'passthrough': False}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `sklearn-cpu`, seed 7): MATCHED

| parameter | sklearn-cpu |
|---|---|
| library (source) | sklearn (get_params) |
| cv | 5 |
| seed | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(not_ready: {"error": "NotImplementedError(\"mojolearn ElasticNet: random_state is refused. It selects nothing here -- the only randomness in cdFit is the coordinate shuffle, and selection='random' is refused by ) |
| mojolearn FAST | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(not_ready: {"error": "NotImplementedError(\"mojolearn ElasticNet: random_state is refused. It selects nothing here -- the only randomness in cdFit is the coordinate shuffle, and selection='random' is refused by ) |
| sklearn-cpu | Xq | - | 4.3 | 4.3..4.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### standard-scaler / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.standard-scaler.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 334.0 | 334.0..334.0 | 1 | - | - | - | 4282.1 | - | max_abs_diff_vs_sklearn=0.002426, output_shape=100000x220, rel_diff_vs_sklearn=3.59e-06 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 338.3 | 338.3..338.3 | 1 | - | - | - | 4289.1 | - | max_abs_diff_vs_sklearn=0.002426, output_shape=100000x220, rel_diff_vs_sklearn=3.591e-06 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 478.3 | 478.3..478.3 | 1 | 0.698 | 0.707 | - | 3343.1 | - | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 68.8 | 68.8..68.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 69.3 | 69.3..69.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 30.1 | 30.1..30.1 | 1 | 2.284 | 2.300 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### standard-scaler / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.standard-scaler.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 22.6 | 22.6..22.6 | 1 | - | - | - | 523.6 | - | max_abs_diff_vs_sklearn=0.000103, output_shape=100000x11, rel_diff_vs_sklearn=3.846e-06 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 21.8 | 21.8..21.8 | 1 | - | - | - | 525.2 | - | max_abs_diff_vs_sklearn=0.000103, output_shape=100000x11, rel_diff_vs_sklearn=3.84e-06 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 45.4 | 45.4..45.4 | 1 | 0.498 | 0.479 | - | 302.9 | - | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 4.0 | 4.0..4.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 4.0 | 4.0..4.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 2.0 | 2.0..2.0 | 1 | 1.961 | 1.981 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### stl / synthetic (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc 0, log `logs/algos.stl.synthetic.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 418.1 | 418.1..418.1 | 1 | - | - | - | 351.1 | - | rel_diff_vs_statsmodels=6.508e-07, residual_std=0.783175 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 367.3 | 367.3..367.3 | 1 | - | - | - | 350.1 | - | rel_diff_vs_statsmodels=6.535e-07, residual_std=0.783175 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsmodels-cpu | statsmodels | cpu | opponent | 104.9 | 104.9..104.9 | 1 | 3.987 | 3.502 | - | 55.0 | - | residual_std=0.783175 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.stl.taxi-hourly.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 417.8 | 417.8..417.8 | 1 | - | - | - | 349.6 | - | rel_diff_vs_statsmodels=1.212e-06, residual_std=18.312475 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 371.0 | 371.0..371.0 | 1 | - | - | - | 350.0 | - | rel_diff_vs_statsmodels=1.258e-06, residual_std=18.312476 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsmodels-cpu | statsmodels | cpu | opponent | 118.8 | 118.8..118.8 | 1 | 3.518 | 3.124 | - | 55.0 | - | residual_std=18.312475 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

### svd / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.svd.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 137381.4 | 137381.4..137381.4 | 1 | - | - | - | 9776.3 | - | max_rel_singular_value_error=28579.906372, relative_reconstruction_error_100k_rows=0.0005441 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 137112.0 | 137112.0..137112.0 | 1 | - | - | - | 9747.2 | - | max_rel_singular_value_error=28579.906372, relative_reconstruction_error_100k_rows=0.0005441 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| numpy-cpu | numpy | cpu | opponent | 6113.8 | 6113.8..6113.8 | 1 | 22.471 | 22.427 | - | 8720.1 | - | max_rel_singular_value_error=37.354529, relative_reconstruction_error_100k_rows=4.1e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-gpu | torch | gpu | opponent | 2473.2 | 2473.2..2473.2 | 1 | 55.548 | 55.439 | - | 5744.3 | 2528.5 | max_rel_singular_value_error=4.945e+06, relative_reconstruction_error_100k_rows=0.0005905 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

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

### svd / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.svd.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 4443.1 | 4443.1..4443.1 | 1 | - | - | - | 1095.0 | - | max_rel_singular_value_error=6.332e-07, relative_reconstruction_error_100k_rows=0.000333 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 4417.9 | 4417.9..4417.9 | 1 | - | - | - | 1099.6 | - | max_rel_singular_value_error=6.332e-07, relative_reconstruction_error_100k_rows=0.000333 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| numpy-cpu | numpy | cpu | opponent | 79.7 | 79.7..79.7 | 1 | 55.745 | 55.429 | - | 393.3 | - | max_rel_singular_value_error=4.308e-08, relative_reconstruction_error_100k_rows=4.314e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-gpu | torch | gpu | opponent | 51.8 | 51.8..51.8 | 1 | 85.814 | 85.329 | - | 1721.8 | 1032.5 | max_rel_singular_value_error=2.225e-06, relative_reconstruction_error_100k_rows=1.351e-05 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.svgp.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3143.8 | 3143.8..3143.8 | 1 | - | - | - | 1587.7 | - | finite=True, r2=-0.106016, rmse=0.878373 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 929.8 | 929.8..929.8 | 1 | - | - | - | 1584.9 | - | finite=True, r2=-0.106016, rmse=0.878373 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| gpytorch-gpu | gpytorch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) (measured this run) |
| gpytorch-cpu | gpytorch | cpu | opponent | 307.3 | 307.3..307.3 | 1 | 10.230 | 3.026 | - | 1626.8 | - | finite=True, r2=-0.106040, rmse=0.878383 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 57.2 | 57.2..57.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 50.4 | 50.4..50.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| gpytorch-gpu | Xq | - | - | - | 0 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) |
| gpytorch-cpu | Xq | - | 96.1 | 96.1..96.1 | 1 | 0.595 | 0.524 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, gpytorch-gpu: predict(Xq)(Xq)

inference call, gpytorch-cpu: predict(Xq)(Xq)

### svgp / taxi (rows full, shape X 100000x11; Xq 20000x11; y 100000; yq 20000)

race: done, driver rc 0, log `logs/algos.svgp.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "ValueError('SVGP: the inducing system is not positive definite; raise jitter or noise_variance')", "event": "error", "stage": "round 0"}) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "ValueError('SVGP: the inducing system is not positive definite; raise jitter or noise_variance')", "event": "error", "stage": "round 0"}) |
| gpytorch-gpu | gpytorch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "TypeError(\"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.\")", "event": "error", "stage": "round 0"}) (measured this run) |
| gpytorch-cpu | gpytorch | cpu | opponent | 247.8 | 247.8..247.8 | 1 | - | - | - | 873.4 | - | finite=True, r2=-0.209325, rmse=17.829528 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

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
| gpytorch-cpu | Xq | - | 64.9 | 64.9..64.9 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, gpytorch-gpu: predict(Xq)(Xq)

inference call, gpytorch-cpu: predict(Xq)(Xq)

### target-encoder / istella (rows full, shape X 1000000x8; Xq 100000x8; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.target-encoder.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 529.6 | 529.6..529.6 | 1 | - | - | - | 753.0 | - | max_abs_diff_vs_sklearn=1.074e-08, output_shape=100000x8, rel_diff_vs_sklearn=2.361e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 460.2 | 460.2..460.2 | 1 | - | - | - | 780.3 | - | max_abs_diff_vs_sklearn=1.074e-08, output_shape=100000x8, rel_diff_vs_sklearn=2.361e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 228.8 | 228.8..228.8 | 1 | 2.315 | 2.011 | - | 279.7 | - | output_shape=100000x8 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 8.1 | 8.1..8.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 8.4 | 8.4..8.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 19.6 | 19.6..19.6 | 1 | 0.415 | 0.428 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### target-encoder / taxi (rows full, shape X 1000000x5; Xq 100000x5; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.target-encoder.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 523.1 | 523.1..523.1 | 1 | - | - | - | 587.3 | - | max_abs_diff_vs_sklearn=2.965e-08, output_shape=100000x5, rel_diff_vs_sklearn=1.581e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 459.5 | 459.5..459.5 | 1 | - | - | - | 638.9 | - | max_abs_diff_vs_sklearn=2.965e-08, output_shape=100000x5, rel_diff_vs_sklearn=1.581e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 147.3 | 147.3..147.3 | 1 | 3.550 | 3.119 | - | 241.7 | - | output_shape=100000x5 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 7.4 | 7.4..7.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 7.6 | 7.6..7.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 11.3 | 11.3..11.3 | 1 | 0.655 | 0.667 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### theta / synthetic (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc 0, log `logs/algos.theta.synthetic.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 134.6 | 134.6..134.6 | 1 | - | - | - | 344.4 | - | forecast_rmse=1.436610 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 110.5 | 110.5..110.5 | 1 | - | - | - | 343.1 | - | forecast_rmse=1.436606 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsforecast-cpu | statsforecast | cpu | opponent | 304.1 | 304.1..304.1 | 1 | 0.443 | 0.363 | - | 190.4 | - | forecast_rmse=1.436557 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| statsmodels-cpu | statsmodels | cpu | opponent | 103.7 | 103.7..103.7 | 1 | 1.298 | 1.066 | - | 52.1 | - | forecast_rmse=1.434862 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.theta.taxi-hourly.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 396.3 | 396.3..396.3 | 1 | - | - | - | 343.3 | - | forecast_rmse=49.020604 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1752.3 | 1752.3..1752.3 | 1 | - | - | - | 342.8 | - | forecast_rmse=49.281168 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsforecast-cpu | statsforecast | cpu | opponent | 960.7 | 960.7..960.7 | 1 | 0.412 | 1.824 | - | 191.2 | - | forecast_rmse=49.253901 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| statsmodels-cpu | statsmodels | cpu | opponent | 171.3 | 171.3..171.3 | 1 | 2.314 | 10.232 | - | 52.5 | - | forecast_rmse=49.311757 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

### tree-shap / istella (rows full, shape X 100000x220; Xq 10000x220; y 100000; yq 10000)

race: done, driver rc 0, log `logs/algos.tree-shap.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 10265.8 | 10265.8..10265.8 | 1 | - | - | - | 1533.5 | - | max_additivity_error=1.099e-06 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 9966.5 | 9966.5..9966.5 | 1 | - | - | - | 1540.6 | - | max_additivity_error=1.099e-06 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| shap-cpu | shap | cpu | opponent | 226.2 | 226.2..226.2 | 1 | 45.386 | 44.063 | - | 1377.1 | - | max_additivity_error=1.862e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| xgboost-cpu | xgboost | cpu | opponent | 221.4 | 221.4..221.4 | 1 | 46.357 | 45.006 | - | 1302.1 | - | max_additivity_error=1.862e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| lightgbm-cpu | lightgbm | cpu | opponent | 148.3 | 148.3..148.3 | 1 | 69.212 | 67.195 | - | 1292.3 | - | max_additivity_error=4.441e-15 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, shap-cpu, xgboost-cpu, lightgbm-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 100}. Rows: None. Timed: None.

mismatch: ours explains its RandomForestRegressor (TreeExplainer takes RF, ExtraTrees, DecisionTree and DART models), the opponents their GBDT of the same size

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | lightgbm-cpu | ours | ours-fast | shap-cpu | xgboost-cpu |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | lightgbm (declared) | mojolearn (declared) | mojolearn (declared) | shap (declared) | xgboost (declared) |
| learning_rate | 0.1 | - | - | 0.1 | 0.1 |
| max_depth | 6 | 6 | 6 | 6 | 6 |
| n_estimators | 100 | 100 | 100 | 100 | 100 |
| seed | 7 | 7 | 7 | 7 | 7 |

### tree-shap / taxi (rows full, shape X 100000x11; Xq 10000x11; y 100000; yq 10000)

race: done, driver rc 0, log `logs/algos.tree-shap.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 4174.6 | 4174.6..4174.6 | 1 | - | - | - | 407.8 | - | max_additivity_error=3.25e-05 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 4127.1 | 4127.1..4127.1 | 1 | - | - | - | 405.8 | - | max_additivity_error=3.25e-05 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| shap-cpu | shap | cpu | opponent | 147.5 | 147.5..147.5 | 1 | 28.298 | 27.976 | - | 293.0 | - | max_additivity_error=0.0001199 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| xgboost-cpu | xgboost | cpu | opponent | 142.0 | 142.0..142.0 | 1 | 29.402 | 29.068 | - | 229.1 | - | max_additivity_error=0.0001199 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| lightgbm-cpu | lightgbm | cpu | opponent | 72.7 | 72.7..72.7 | 1 | 57.390 | 56.737 | - | 225.3 | - | max_additivity_error=5.684e-13 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, shap-cpu, xgboost-cpu, lightgbm-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 100}. Rows: None. Timed: None.

mismatch: ours explains its RandomForestRegressor (TreeExplainer takes RF, ExtraTrees, DecisionTree and DART models), the opponents their GBDT of the same size

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | lightgbm-cpu | ours | ours-fast | shap-cpu | xgboost-cpu |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | lightgbm (declared) | mojolearn (declared) | mojolearn (declared) | shap (declared) | xgboost (declared) |
| learning_rate | 0.1 | - | - | 0.1 | 0.1 |
| max_depth | 6 | 6 | 6 | 6 | 6 |
| n_estimators | 100 | 100 | 100 | 100 | 100 |
| seed | 7 | 7 | 7 | 7 | 7 |

### tsne / istella (rows full, shape X 20000x220; Xq 2000x220)

race: done, driver rc 0, log `logs/algos.tsne.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 4921.3 | 4921.3..4921.3 | 1 | - | - | - | 506.3 | - | trustworthiness_k15=0.991881 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 4107.0 | 4107.0..4107.0 | 1 | - | - | - | 506.2 | - | trustworthiness_k15=0.992039 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 19428.9 | 19428.9..19428.9 | 1 | 0.253 | 0.211 | - | 297.1 | - | trustworthiness_k15=0.991970 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'early_exaggeration': 12.0, 'init': 'seeded', 'learning_rate': 'auto', 'max_iter': 1000, 'n_components': 2, 'perplexity': 30.0, 'random_state': 7}. Rows: None. Timed: None.

mismatch: gradients: ours exact repulsion over k-NN affinities (no Barnes-Hut atomics under IDENTICAL); scikit-learn Barnes-Hut (angle=0.5, scikit-learn only); cuML FFT

mismatch: init: ours and scikit-learn start from the SAME array, ours' 'random' rule (default_rng(7).uniform(-5e-5, 5e-5, (n, 2)) float32); cuML takes only 'random' and draws its own start

mismatch: scikit-learn's early stop is switched off (n_iter_without_progress=1000, min_grad_norm=0.0): ours runs exactly max_iter steps

config: cuML benchmark (RAPIDS), TSNE (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| early_exaggeration | 12.0 | 12.0 | 12.0 |
| init | "array([[ 1.2509547e-05, 3.9721381e-05],\n [ 2.7568569e-05, -2.7479280e-05],\n [-1.9983372e-05, 3.7355345e-05],\n ...,\n [ 1.3436503e-05, -5.4599764e-06],\n [-1.2881169e-05, 6.7022388e-06],\n [-2.8746643e-05, -4.1470521e-05]], shape=(20000, 2), dtype=float32)" | "array([[ 1.2509547e-05, 3.9721381e-05],\n [ 2.7568569e-05, -2.7479280e-05],\n [-1.9983372e-05, 3.7355345e-05],\n ...,\n [ 1.3436503e-05, -5.4599764e-06],\n [-1.2881169e-05, 6.7022388e-06],\n [-2.8746643e-05, -4.1470521e-05]], shape=(20000, 2), dtype=float32)" | "array([[ 1.2509547e-05, 3.9721381e-05],\n [ 2.7568569e-05, -2.7479280e-05],\n [-1.9983372e-05, 3.7355345e-05],\n ...,\n [ 1.3436503e-05, -5.4599764e-06],\n [-1.2881169e-05, 6.7022388e-06],\n [-2.8746643e-05, -4.1470521e-05]], shape=(20000, 2), dtype=float32)" |
| learning_rate | "auto" | "auto" | "auto" |
| max_iter | 1000 | 1000 | 1000 |
| metric | - | - | "euclidean" |
| n_components | 2 | 2 | 2 |
| perplexity | 30.0 | 30.0 | 30.0 |
| seed | 7 | 7 | 7 |

### tsne / taxi (rows full, shape X 20000x11; Xq 2000x11)

race: done, driver rc 0, log `logs/algos.tsne.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 4040.5 | 4040.5..4040.5 | 1 | - | - | - | 452.2 | - | trustworthiness_k15=0.998927 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 3296.5 | 3296.5..3296.5 | 1 | - | - | - | 451.2 | - | trustworthiness_k15=0.998869 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 17507.0 | 17507.0..17507.0 | 1 | 0.231 | 0.188 | - | 273.9 | - | trustworthiness_k15=0.998860 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'early_exaggeration': 12.0, 'init': 'seeded', 'learning_rate': 'auto', 'max_iter': 1000, 'n_components': 2, 'perplexity': 30.0, 'random_state': 7}. Rows: None. Timed: None.

mismatch: gradients: ours exact repulsion over k-NN affinities (no Barnes-Hut atomics under IDENTICAL); scikit-learn Barnes-Hut (angle=0.5, scikit-learn only); cuML FFT

mismatch: init: ours and scikit-learn start from the SAME array, ours' 'random' rule (default_rng(7).uniform(-5e-5, 5e-5, (n, 2)) float32); cuML takes only 'random' and draws its own start

mismatch: scikit-learn's early stop is switched off (n_iter_without_progress=1000, min_grad_norm=0.0): ours runs exactly max_iter steps

config: cuML benchmark (RAPIDS), TSNE (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| early_exaggeration | 12.0 | 12.0 | 12.0 |
| init | "array([[ 1.2509547e-05, 3.9721381e-05],\n [ 2.7568569e-05, -2.7479280e-05],\n [-1.9983372e-05, 3.7355345e-05],\n ...,\n [ 1.3436503e-05, -5.4599764e-06],\n [-1.2881169e-05, 6.7022388e-06],\n [-2.8746643e-05, -4.1470521e-05]], shape=(20000, 2), dtype=float32)" | "array([[ 1.2509547e-05, 3.9721381e-05],\n [ 2.7568569e-05, -2.7479280e-05],\n [-1.9983372e-05, 3.7355345e-05],\n ...,\n [ 1.3436503e-05, -5.4599764e-06],\n [-1.2881169e-05, 6.7022388e-06],\n [-2.8746643e-05, -4.1470521e-05]], shape=(20000, 2), dtype=float32)" | "array([[ 1.2509547e-05, 3.9721381e-05],\n [ 2.7568569e-05, -2.7479280e-05],\n [-1.9983372e-05, 3.7355345e-05],\n ...,\n [ 1.3436503e-05, -5.4599764e-06],\n [-1.2881169e-05, 6.7022388e-06],\n [-2.8746643e-05, -4.1470521e-05]], shape=(20000, 2), dtype=float32)" |
| learning_rate | "auto" | "auto" | "auto" |
| max_iter | 1000 | 1000 | 1000 |
| metric | - | - | "euclidean" |
| n_components | 2 | 2 | 2 |
| perplexity | 30.0 | 30.0 | 30.0 |
| seed | 7 | 7 | 7 |

### var / synthetic (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc 0, log `logs/algos.var.synthetic.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 11.2 | 11.2..11.2 | 1 | - | - | - | 327.6 | - | forecast_rmse=1.140864 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 12.7 | 12.7..12.7 | 1 | - | - | - | 326.7 | - | forecast_rmse=1.140787 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsmodels-cpu | statsmodels | cpu | opponent | 2.6 | 2.6..2.6 | 1 | 4.230 | 4.808 | - | 152.8 | - | forecast_rmse=1.144945 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.var.taxi-hourly.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 14.0 | 14.0..14.0 | 1 | - | - | - | 325.0 | - | forecast_rmse=33.167951 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 16.7 | 16.7..16.7 | 1 | - | - | - | 325.3 | - | forecast_rmse=33.167955 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsmodels-cpu | statsmodels | cpu | opponent | 2.8 | 2.8..2.8 | 1 | 5.054 | 6.035 | - | 149.5 | - | forecast_rmse=33.167986 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.variance-threshold.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 235.8 | 235.8..235.8 | 1 | - | - | - | 3231.3 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x142, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 127.8 | 127.8..127.8 | 1 | - | - | - | 3231.2 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x142, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 687.8 | 687.8..687.8 | 1 | 0.343 | 0.186 | - | 4906.8 | - | output_shape=100000x142 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 37.6 | 37.6..37.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 38.8 | 38.8..38.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 19.0 | 19.0..19.0 | 1 | 1.979 | 2.043 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### variance-threshold / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.variance-threshold.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 122.5 | 122.5..122.5 | 1 | - | - | - | 468.1 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x10, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 10.2 | 10.2..10.2 | 1 | - | - | - | 466.3 | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x10, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 50.6 | 50.6..50.6 | 1 | 2.422 | 0.201 | - | 395.4 | - | output_shape=100000x10 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 2.7 | 2.7..2.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 3.1 | 3.1..3.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 0.7 | 0.7..0.7 | 1 | 3.734 | 4.250 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### voting-clf / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.voting-clf.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1665.1 | 1665.1..1665.1 | 1 | - | - | - | 4755.0 | - | accuracy=0.918320, logloss=0.187682 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1518.3 | 1518.3..1518.3 | 1 | - | - | - | 4746.9 | - | accuracy=0.918360, logloss=0.187737 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 47440.0 | 47440.0..47440.0 | 1 | 0.035 | 0.032 | - | 2976.4 | - | accuracy=0.918480, logloss=0.187541 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'estimators': [['lr', 'LogisticRegression(max_iter=200)'], ['nb', 'GaussianNB()'], ['dt', 'DecisionTreeClassifier(max_depth=8, random_state=7)']], 'flatten_transform': True, 'voting': 'soft'}. Rows: None. Timed: None.

mismatch: nested LogisticRegression(max_iter=200): ours is L-BFGS (cuML's quasi-Newton, memory 5, solver='qn'), scikit-learn solver='lbfgs' (scipy, memory 10); C=1.0 and tol=1e-4 are both libraries' defaults; each stops by its own criterion

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| weights | null | null | null |

accepted difference: ours-fast weights: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: sklearn-cpu weights: None on both ours and sklearn-cpu: the same documented setting in both signatures

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 54.3 | 54.3..54.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 54.9 | 54.9..54.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 231.6 | 231.6..231.6 | 1 | 0.234 | 0.237 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### voting-clf / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.voting-clf.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 474.9 | 474.9..474.9 | 1 | - | - | - | 613.1 | - | accuracy=0.742320, logloss=0.554799 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 191.2 | 191.2..191.2 | 1 | - | - | - | 614.2 | - | accuracy=0.742290, logloss=0.554795 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1597.0 | 1597.0..1597.0 | 1 | 0.297 | 0.120 | - | 402.9 | - | accuracy=0.742350, logloss=0.554650 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'estimators': [['lr', 'LogisticRegression(max_iter=200)'], ['nb', 'GaussianNB()'], ['dt', 'DecisionTreeClassifier(max_depth=8, random_state=7)']], 'flatten_transform': True, 'voting': 'soft'}. Rows: None. Timed: None.

mismatch: nested LogisticRegression(max_iter=200): ours is L-BFGS (cuML's quasi-Newton, memory 5, solver='qn'), scikit-learn solver='lbfgs' (scipy, memory 10); C=1.0 and tol=1e-4 are both libraries' defaults; each stops by its own criterion

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| weights | null | null | null |

accepted difference: ours-fast weights: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: sklearn-cpu weights: None on both ours and sklearn-cpu: the same documented setting in both signatures

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 10.9 | 10.9..10.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 11.0 | 11.0..11.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 26.6 | 26.6..26.6 | 1 | 0.410 | 0.415 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### voting-reg / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.voting-reg.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | UNKNOWN(no binding path) | REFUSED(not_ready: {"error": "NotImplementedError(\"mojolearn ElasticNet: random_state is refused. It selects nothing here -- the only randomness in cdFit is the coordinate shuffle, and selection='random' is refused by ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | UNKNOWN(no binding path) | REFUSED(not_ready: {"error": "NotImplementedError(\"mojolearn ElasticNet: random_state is refused. It selects nothing here -- the only randomness in cdFit is the coordinate shuffle, and selection='random' is refused by ) |
| sklearn-cpu | scikit-learn | cpu | opponent | 37764.6 | 37764.6..37764.6 | 1 | - | - | - | 8656.1 | - | finite=True, r2=0.406702, rmse=0.643423 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host not sampled; GPU not sampled

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'estimators': [['ridge', 'Ridge(alpha=1.0)'], ['lasso', 'Lasso(alpha=0.01, max_iter=1000, random_state=7, tol=0.001)'], ['dt', 'DecisionTreeRegressor(max_depth=8, random_state=7)']]}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `sklearn-cpu`, seed 7): MATCHED

| parameter | sklearn-cpu |
|---|---|
| library (source) | sklearn (get_params) |
| seed | "none (deterministic)" |
| weights | null |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(not_ready: {"error": "NotImplementedError(\"mojolearn ElasticNet: random_state is refused. It selects nothing here -- the only randomness in cdFit is the coordinate shuffle, and selection='random' is refused by ) |
| mojolearn FAST | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(not_ready: {"error": "NotImplementedError(\"mojolearn ElasticNet: random_state is refused. It selects nothing here -- the only randomness in cdFit is the coordinate shuffle, and selection='random' is refused by ) |
| sklearn-cpu | Xq | - | 14.3 | 14.3..14.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### voting-reg / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.voting-reg.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | UNKNOWN(no binding path) | REFUSED(not_ready: {"error": "NotImplementedError(\"mojolearn ElasticNet: random_state is refused. It selects nothing here -- the only randomness in cdFit is the coordinate shuffle, and selection='random' is refused by ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | UNKNOWN(no binding path) | REFUSED(not_ready: {"error": "NotImplementedError(\"mojolearn ElasticNet: random_state is refused. It selects nothing here -- the only randomness in cdFit is the coordinate shuffle, and selection='random' is refused by ) |
| sklearn-cpu | scikit-learn | cpu | opponent | 1272.1 | 1272.1..1272.1 | 1 | - | - | - | 297.8 | - | finite=True, r2=0.924635, rmse=4.372421 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host not sampled; GPU not sampled

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'estimators': [['ridge', 'Ridge(alpha=1.0)'], ['lasso', 'Lasso(alpha=0.01, max_iter=1000, random_state=7, tol=0.001)'], ['dt', 'DecisionTreeRegressor(max_depth=8, random_state=7)']]}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `sklearn-cpu`, seed 7): MATCHED

| parameter | sklearn-cpu |
|---|---|
| library (source) | sklearn (get_params) |
| seed | "none (deterministic)" |
| weights | null |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(not_ready: {"error": "NotImplementedError(\"mojolearn ElasticNet: random_state is refused. It selects nothing here -- the only randomness in cdFit is the coordinate shuffle, and selection='random' is refused by ) |
| mojolearn FAST | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(not_ready: {"error": "NotImplementedError(\"mojolearn ElasticNet: random_state is refused. It selects nothing here -- the only randomness in cdFit is the coordinate shuffle, and selection='random' is refused by ) |
| sklearn-cpu | Xq | - | 4.6 | 4.6..4.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

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

