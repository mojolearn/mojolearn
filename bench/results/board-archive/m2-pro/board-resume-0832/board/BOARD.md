# mojolearn benchmark board

Generated 2026-10-01T05:54:20Z from `board.json` (schema `mojolearn-bench-board/1`).

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
| mojolearn | 0.8.32 (wheel mojolearn-0.8.32-py3-none-macosx_11_0_arm64.whl, sha256 9cd5a1384f53777aec6259d590fbdffb4cefc6478cb78cb28fe0f94c745a5ce0) |
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

Races: 1 planned, 73 done, 2 failed, 0 pending. Cells: 281 (MODE-MISMATCH 12, REFUSED 25, ok 244).

Inference cells: 200 (REFUSED 19, ok 181).

## Quality at a glance

Per lane and dataset: our FAST value, our IDENTICAL value, our CPU IDENTICAL value, and each opponent's.

| family | lane | dataset | metric | ours FAST | ours IDENTICAL | ours CPU | opponents |
|---|---|---|---|---|---|---|---|
| algos | avgpool1d | synthetic | max_rel_diff_vs_torch_eager_fp32 | 0.000000 | 0.000000 | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | avgpool1d | synthetic | rel_fro_vs_torch_eager_fp32 | 0.000000 | 0.000000 | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | avgpool2d | synthetic | max_rel_diff_vs_torch_eager_fp32 | 0.000000 | 0.000000 | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | avgpool2d | synthetic | rel_fro_vs_torch_eager_fp32 | 0.000000 | 0.000000 | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | batchnorm1d | synthetic | max_rel_diff_vs_torch_eager_fp32 | 0.063434 | 0.063434 | - | torch-eager-fp32 -; torch-compile-fp32 0.012127; torch-eager-bf16 0.000000; torch-compile-bf16 0.012127 |
| algos | batchnorm1d | synthetic | rel_fro_vs_torch_eager_fp32 | 4.096e-06 | 4.095e-06 | - | torch-eager-fp32 -; torch-compile-fp32 3.022e-07; torch-eager-bf16 0.000000; torch-compile-bf16 3.022e-07 |
| algos | batchnorm2d | synthetic | max_rel_diff_vs_torch_eager_fp32 | 0.034926 | 0.034926 | - | torch-eager-fp32 -; torch-compile-fp32 0.005340; torch-eager-bf16 0.000000; torch-compile-bf16 0.005340 |
| algos | batchnorm2d | synthetic | rel_fro_vs_torch_eager_fp32 | 2.344e-05 | 2.344e-05 | - | torch-eager-fp32 -; torch-compile-fp32 2.914e-07; torch-eager-bf16 0.000000; torch-compile-bf16 2.914e-07 |
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
| algos | clip-grad-norm | synthetic | norm | 1.000000 | 1.000000 | - | torch-eager-fp32 1.000000; torch-compile-fp32 1.000000 |
| algos | clip-grad-norm | synthetic | norm_rel_diff_vs_ours | 0.000000 | 0.000000 | - | torch-eager-fp32 0.000000; torch-compile-fp32 0.000000 |
| algos | cnn-clf | synthetic | accuracy (higher is better) | 1.000000 | 1.000000 | - | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | conv1d | synthetic | max_rel_diff_vs_torch_eager_fp32 | 0.391155 | 0.391155 | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 3479.164094; torch-compile-bf16 3418.128937 |
| algos | conv1d | synthetic | rel_fro_vs_torch_eager_fp32 | 3.947e-07 | 3.947e-07 | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 0.002870; torch-compile-bf16 0.003296 |
| algos | conv2d | synthetic | max_rel_diff_vs_torch_eager_fp32 | 0.461936 | 0.461936 | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 3463.592380; torch-compile-bf16 3418.426961 |
| algos | conv2d | synthetic | rel_fro_vs_torch_eager_fp32 | 3.303e-07 | 3.303e-07 | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 0.002935; torch-compile-bf16 0.003382 |
| algos | cross-entropy | synthetic | loss_rel_err_vs_fp64 | 4.564e-08 | 4.564e-08 | - | torch-eager-fp32 4.564e-08; torch-compile-fp32 4.564e-08 |
| algos | cross-entropy | synthetic | grad_max_rel_diff_vs_ours | 1.863e-09 | - | - | torch-eager-fp32 3.725e-09; torch-compile-fp32 5.588e-09 |
| algos | embedding | synthetic | max_rel_diff_vs_torch_eager_fp32 | 0.000000 | 0.000000 | - | torch-eager-fp32 -; torch-compile-fp32 0.000000 |
| algos | embedding | synthetic | rel_fro_vs_torch_eager_fp32 | 0.000000 | 0.000000 | - | torch-eager-fp32 -; torch-compile-fp32 0.000000 |
| algos | gcn | istella | max_rel_diff_vs_torch_eager_fp32 | - | - | - | torch-eager-fp32 -; torch-compile-fp32 0.014120; torch-eager-bf16 2744.908399; torch-compile-bf16 2744.917456 |
| algos | gcn | istella | rel_fro_vs_torch_eager_fp32 | - | - | - | torch-eager-fp32 -; torch-compile-fp32 2.389e-07; torch-eager-bf16 0.002187; torch-compile-bf16 0.002187 |
| algos | gcn | taxi | max_rel_diff_vs_torch_eager_fp32 | - | - | - | torch-eager-fp32 -; torch-compile-fp32 0.005122; torch-eager-bf16 1509.511378; torch-compile-bf16 1509.509399 |
| algos | gcn | taxi | rel_fro_vs_torch_eager_fp32 | - | - | - | torch-eager-fp32 -; torch-compile-fp32 2.408e-07; torch-eager-bf16 0.002330; torch-compile-bf16 0.002330 |
| algos | global-avgpool | synthetic | max_rel_diff_vs_torch_eager_fp32 | 0.015661 | 0.015661 | - | torch-eager-fp32 -; torch-compile-fp32 0.007299; torch-eager-bf16 0.000000; torch-compile-bf16 0.007299 |
| algos | global-avgpool | synthetic | rel_fro_vs_torch_eager_fp32 | 1.493e-07 | 1.493e-07 | - | torch-eager-fp32 -; torch-compile-fp32 9.321e-08; torch-eager-bf16 0.000000; torch-compile-bf16 9.321e-08 |
| algos | global-maxpool | synthetic | max_rel_diff_vs_torch_eager_fp32 | 0.000000 | 0.000000 | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | global-maxpool | synthetic | rel_fro_vs_torch_eager_fp32 | 0.000000 | 0.000000 | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | graphsage | istella | max_rel_diff_vs_torch_eager_fp32 | - | - | - | torch-eager-fp32 -; torch-compile-fp32 0.089407; torch-eager-bf16 3907.114267; torch-compile-bf16 3678.187728 |
| algos | graphsage | istella | rel_fro_vs_torch_eager_fp32 | - | - | - | torch-eager-fp32 -; torch-compile-fp32 7.979e-08; torch-eager-bf16 0.003366; torch-compile-bf16 0.003054 |
| algos | graphsage | taxi | max_rel_diff_vs_torch_eager_fp32 | - | - | - | torch-eager-fp32 -; torch-compile-fp32 0.153846; torch-eager-bf16 5460.333333; torch-compile-bf16 4608.154297 |
| algos | graphsage | taxi | rel_fro_vs_torch_eager_fp32 | - | - | - | torch-eager-fp32 -; torch-compile-fp32 4.956e-08; torch-eager-bf16 0.003557; torch-compile-bf16 0.003285 |
| algos | ivf-filter | istella | recall_at_10 (higher is better) | 0.589100 | 0.609250 | - | faiss-cpu 0.841950 |
| algos | ivf-filter | taxi | recall_at_10 (higher is better) | 0.982950 | 0.980075 | - | faiss-cpu 0.982050 |
| algos | ivf-pq | istella | recall_at_10 (higher is better) | 0.599475 | 0.565025 | - | faiss-cpu 0.802975 |
| algos | ivf-pq | taxi | recall_at_10 (higher is better) | 0.976450 | 0.973225 | - | faiss-cpu 0.980075 |
| algos | ivf-rabitq | istella | recall_at_10 (higher is better) | 0.132550 | 0.125125 | - | faiss-cpu 0.050450 |
| algos | ivf-rabitq | taxi | recall_at_10 (higher is better) | 0.115775 | 0.110475 | - | faiss-cpu 0.126275 |
| algos | ivf-refine | istella | recall_at_10 (higher is better) | 0.862250 | 0.809175 | - | faiss-cpu 0.993425 |
| algos | ivf-refine | taxi | recall_at_10 (higher is better) | 0.999675 | 0.999675 | - | faiss-cpu 0.999225 |
| algos | ivf-sq | istella | recall_at_10 (higher is better) | 0.631000 | 0.728025 | - | faiss-cpu 0.591300 |
| algos | ivf-sq | taxi | recall_at_10 (higher is better) | 0.902025 | 0.934975 | - | faiss-cpu 0.857025 |
| algos | jl-min-dim | synthetic | equal_fraction_vs_sklearn | 1.000000 | 1.000000 | - | sklearn-cpu 1.000000 |
| algos | kernel-shap | istella | rel_error_vs_exact | 4.179e-09 | 4.179e-09 | - | shap-cpu 1.405e-14 |
| algos | kernel-shap | taxi | rel_error_vs_exact | 1.731e-08 | 1.731e-08 | - | shap-cpu 6.394e-14 |
| algos | kpss | synthetic | flag_agreement_vs_statsmodels | 1.000000 | 1.000000 | - | statsmodels-cpu 1.000000 |
| algos | kpss | synthetic | stat_max_rel_diff_vs_statsmodels | 4.062e-05 | 4.318e-05 | - | statsmodels-cpu 0.000000 |
| algos | kpss | synthetic | stationary_fraction | 0.031250 | 0.031250 | - | statsmodels-cpu 0.031250 |
| algos | kpss | taxi-hourly | flag_agreement_vs_statsmodels | 1.000000 | 1.000000 | - | statsmodels-cpu 1.000000 |
| algos | kpss | taxi-hourly | stat_max_rel_diff_vs_statsmodels | 3.946e-05 | 3.946e-05 | - | statsmodels-cpu 0.000000 |
| algos | kpss | taxi-hourly | stationary_fraction | 0.687500 | 0.687500 | - | statsmodels-cpu 0.687500 |
| algos | logreg-cv | istella | accuracy (higher is better) | - | 0.885090 | - | sklearn-cpu 0.924630 |
| algos | logreg-cv | istella | logloss (lower is better) | - | 0.693147 | - | sklearn-cpu 0.181351 |
| algos | logreg-cv | taxi | accuracy (higher is better) | - | - | - | sklearn-cpu 0.763300 |
| algos | logreg-cv | taxi | logloss (lower is better) | - | - | - | sklearn-cpu 0.538988 |
| algos | lr-constant | synthetic | max_rel_diff_vs_ours (0 is our output exactly) | 0.000000 | 0.000000 | - | torch-cpu 1.038e-07 |
| algos | lr-exponential | synthetic | max_rel_diff_vs_ours (0 is our output exactly) | 0.000000 | 0.000000 | - | torch-cpu 5.933e-08 |
| algos | lr-onecycle | synthetic | max_rel_diff_vs_ours (0 is our output exactly) | 0.000000 | 0.000000 | - | torch-cpu 5.951e-08 |
| algos | lr-step | synthetic | max_rel_diff_vs_ours (0 is our output exactly) | 0.000000 | 0.000000 | - | torch-cpu 1.49e-08 |
| algos | lr-warmup-cosine | synthetic | max_rel_diff_vs_ours (0 is our output exactly) | 0.000000 | 0.000000 | - | torch-cpu 0.0001571 |
| algos | lr-warmup-linear | synthetic | max_rel_diff_vs_ours (0 is our output exactly) | 0.000000 | 0.000000 | - | torch-cpu 0.001000 |
| algos | maxpool1d | synthetic | max_rel_diff_vs_torch_eager_fp32 | 0.000000 | 0.000000 | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | maxpool1d | synthetic | rel_fro_vs_torch_eager_fp32 | 0.000000 | 0.000000 | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | maxpool2d | synthetic | max_rel_diff_vs_torch_eager_fp32 | 0.000000 | 0.000000 | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | maxpool2d | synthetic | rel_fro_vs_torch_eager_fp32 | 0.000000 | 0.000000 | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | multioutput-clf | istella | accuracy (higher is better) | 0.959115 | 0.959110 | - | sklearn-cpu 0.959225 |
| algos | multioutput-clf | taxi | accuracy (higher is better) | 0.863560 | 0.863560 | - | sklearn-cpu 0.863550 |
| algos | multioutput-reg | istella | r2 (higher is better) | 0.439854 | 0.455327 | - | sklearn-cpu 0.455345 |
| algos | multioutput-reg | taxi | r2 (higher is better) | 0.604241 | 0.604240 | - | sklearn-cpu 0.604316 |
| algos | ovr | istella | accuracy (higher is better) | 0.892760 | 0.892670 | - | sklearn-cpu 0.892730 |
| algos | ovr | taxi | accuracy (higher is better) | 0.478970 | 0.478940 | - | sklearn-cpu 0.478890 |
| algos | permutation-shap | istella | rel_error_vs_exact | 5.339e-09 | 5.339e-09 | - | shap-cpu 3.692e-10 |
| algos | permutation-shap | taxi | rel_error_vs_exact | 2.15e-08 | 2.15e-08 | - | shap-cpu 1.279e-15 |
| algos | random-trees-embedding | istella | nonzeros_per_row | 10.000000 | 10.000000 | - | sklearn-cpu 10.000000 |
| algos | random-trees-embedding | istella | output_columns | 209 | 209 | - | sklearn-cpu 251 |
| algos | random-trees-embedding | taxi | nonzeros_per_row | 10.000000 | 10.000000 | - | sklearn-cpu 10.000000 |
| algos | random-trees-embedding | taxi | output_columns | 292 | 292 | - | sklearn-cpu 244 |
| algos | resnet-block | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | - | torch-eager-fp32 -; torch-compile-fp32 0.380952; torch-eager-bf16 18498.390913; torch-compile-bf16 18498.569727 |
| algos | resnet-block | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | - | torch-eager-fp32 -; torch-compile-fp32 1.952e-07; torch-eager-bf16 0.003425; torch-compile-bf16 0.003425 |
| algos | select-d | synthetic | d_agreement_vs_statsmodels | 1.000000 | 1.000000 | - | statsmodels-cpu 1.000000 |
| algos | select-d | taxi-hourly | d_agreement_vs_statsmodels | 1.000000 | 1.000000 | - | statsmodels-cpu 1.000000 |
| algos | stacking-clf | istella | accuracy (higher is better) | 0.929970 | 0.929960 | - | sklearn-cpu 0.930040 |
| algos | stacking-clf | istella | logloss (lower is better) | 0.193693 | 0.193706 | - | sklearn-cpu 0.193931 |
| algos | stacking-clf | taxi | accuracy (higher is better) | 0.767920 | 0.767920 | - | sklearn-cpu 0.755330 |
| algos | stacking-clf | taxi | logloss (lower is better) | 0.536364 | 0.536361 | - | sklearn-cpu 0.547756 |
| algos | stacking-reg | istella | r2 (higher is better) | - | - | - | sklearn-cpu 0.447644 |
| algos | stacking-reg | istella | rmse (lower is better) | - | - | - | sklearn-cpu 0.620826 |
| algos | stacking-reg | taxi | r2 (higher is better) | - | - | - | sklearn-cpu 0.932532 |
| algos | stacking-reg | taxi | rmse (lower is better) | - | - | - | sklearn-cpu 4.137015 |
| algos | tree-shap | istella | max_additivity_error | 1.099e-06 | 1.099e-06 | - | shap-cpu 1.862e-06; xgboost-cpu 1.862e-06; lightgbm-cpu 4.441e-15 |
| algos | tree-shap | taxi | max_additivity_error | 3.25e-05 | 3.25e-05 | - | shap-cpu 0.0001199; xgboost-cpu 0.0001199; lightgbm-cpu 5.684e-13 |
| algos | tsne | istella | trustworthiness_k15 (higher is better, 1 at most) | 0.992164 | 0.991881 | - | sklearn-cpu 0.992041 |
| algos | tsne | taxi | trustworthiness_k15 (higher is better, 1 at most) | 0.998912 | 0.998927 | - | sklearn-cpu 0.998860 |
| algos | voting-clf | istella | accuracy (higher is better) | 0.918350 | 0.918320 | - | sklearn-cpu 0.918480 |
| algos | voting-clf | istella | logloss (lower is better) | 0.187755 | 0.187682 | - | sklearn-cpu 0.187541 |
| algos | voting-clf | taxi | accuracy (higher is better) | 0.742290 | 0.742320 | - | sklearn-cpu 0.742350 |
| algos | voting-clf | taxi | logloss (lower is better) | 0.554802 | 0.554799 | - | sklearn-cpu 0.554650 |
| algos | voting-reg | istella | r2 (higher is better) | - | - | - | sklearn-cpu 0.406702 |
| algos | voting-reg | istella | rmse (lower is better) | - | - | - | sklearn-cpu 0.643423 |
| algos | voting-reg | taxi | r2 (higher is better) | - | - | - | sklearn-cpu 0.924635 |
| algos | voting-reg | taxi | rmse (lower is better) | - | - | - | sklearn-cpu 4.372421 |

## Inference at a glance

Batch prediction, each arm with its own fitted model from the same race; medians in ms. `FAST = IDENTICAL bits` compares our two tiers' predictions on the same rows; `CPU = IDENTICAL bits` compares our CPU tier's with our GPU IDENTICAL arm's.

| family | lane | dataset | batch | rows | ours FAST ms | ours IDENTICAL ms | FAST = IDENTICAL bits | ours CPU ms | CPU = IDENTICAL bits | opponents |
|---|---|---|---|---|---|---|---|---|---|---|
| algos | avgpool1d | synthetic | Xq | - | 26.4 | 20.2 | - | - | - | torch-eager-fp32 2.6 ms (IDENTICAL/arm 7.748); torch-compile-fp32 1.2 ms (IDENTICAL/arm 17.152); torch-eager-bf16 2.6 ms (IDENTICAL/arm 7.683); torch-compile-bf16 1.2 ms (IDENTICAL/arm 17.280) |
| algos | avgpool2d | synthetic | Xq | - | 36.7 | 37.8 | - | - | - | torch-eager-fp32 2.2 ms (IDENTICAL/arm 17.451); torch-compile-fp32 2.1 ms (IDENTICAL/arm 18.382); torch-eager-bf16 2.0 ms (IDENTICAL/arm 18.961); torch-compile-bf16 2.1 ms (IDENTICAL/arm 18.371) |
| algos | batchnorm1d | synthetic | Xq | - | 49.1 | 55.3 | - | - | - | torch-eager-fp32 2.0 ms (IDENTICAL/arm 27.900); torch-compile-fp32 6.2 ms (IDENTICAL/arm 8.956); torch-eager-bf16 2.2 ms (IDENTICAL/arm 25.074); torch-compile-bf16 12.1 ms (IDENTICAL/arm 4.563) |
| algos | batchnorm2d | synthetic | Xq | - | 37.8 | 36.9 | - | - | - | torch-eager-fp32 1.9 ms (IDENTICAL/arm 18.991); torch-compile-fp32 5.5 ms (IDENTICAL/arm 6.770); torch-eager-bf16 1.9 ms (IDENTICAL/arm 19.514); torch-compile-bf16 4.8 ms (IDENTICAL/arm 7.729) |
| algos | cagra | istella | Xq | - | - | - | - | - | - | faiss-cpu 20.7 ms (IDENTICAL/arm -) |
| algos | cagra | taxi | Xq | - | - | - | - | - | - | faiss-cpu 8.6 ms (IDENTICAL/arm -) |
| algos | calibrated | istella | Xq | - | 219.5 | 225.1 | - | - | - | sklearn-cpu 1060.1 ms (IDENTICAL/arm 0.212) |
| algos | calibrated | taxi | Xq | - | 71.2 | 68.4 | - | - | - | sklearn-cpu 121.1 ms (IDENTICAL/arm 0.565) |
| algos | cnn-clf | synthetic | Xq | - | 79.0 | 93.8 | - | - | - | torch-eager-fp32 - ms (IDENTICAL/arm -); torch-compile-fp32 - ms (IDENTICAL/arm -); torch-eager-bf16 - ms (IDENTICAL/arm -); torch-compile-bf16 - ms (IDENTICAL/arm -) |
| algos | conv1d | synthetic | Xq | - | 87.6 | 92.5 | - | - | - | torch-eager-fp32 7.6 ms (IDENTICAL/arm 12.246); torch-compile-fp32 8.6 ms (IDENTICAL/arm 10.795); torch-eager-bf16 11.0 ms (IDENTICAL/arm 8.416); torch-compile-bf16 11.4 ms (IDENTICAL/arm 8.143) |
| algos | conv2d | synthetic | Xq | - | 43.8 | 46.2 | - | - | - | torch-eager-fp32 3.1 ms (IDENTICAL/arm 15.098); torch-compile-fp32 4.9 ms (IDENTICAL/arm 9.386); torch-eager-bf16 3.4 ms (IDENTICAL/arm 13.571); torch-compile-bf16 4.8 ms (IDENTICAL/arm 9.720) |
| algos | dropout2d | synthetic | Xq | - | 68.5 | 68.5 | - | - | - | torch-eager-fp32 5.2 ms (IDENTICAL/arm 13.207); torch-compile-fp32 0.9 ms (IDENTICAL/arm 80.015) |
| algos | embedding | synthetic | Xq | - | 372.5 | 363.7 | - | - | - | torch-eager-fp32 2.3 ms (IDENTICAL/arm 156.987); torch-compile-fp32 2.6 ms (IDENTICAL/arm 139.787) |
| algos | gcn | istella | Xq | - | 140.7 | 130.5 | - | - | - | torch-eager-fp32 148.0 ms (IDENTICAL/arm 0.882); torch-compile-fp32 15.3 ms (IDENTICAL/arm 8.538); torch-eager-bf16 146.8 ms (IDENTICAL/arm 0.889); torch-compile-bf16 15.8 ms (IDENTICAL/arm 8.276) |
| algos | gcn | taxi | Xq | - | 123.3 | 112.9 | - | - | - | torch-eager-fp32 122.1 ms (IDENTICAL/arm 0.925); torch-compile-fp32 12.3 ms (IDENTICAL/arm 9.153); torch-eager-bf16 120.2 ms (IDENTICAL/arm 0.940); torch-compile-bf16 11.9 ms (IDENTICAL/arm 9.510) |
| algos | global-avgpool | synthetic | Xq | - | 1.1 | 1.1 | - | - | - | torch-eager-fp32 8.4 ms (IDENTICAL/arm 0.133); torch-compile-fp32 0.8 ms (IDENTICAL/arm 1.317); torch-eager-bf16 3.1 ms (IDENTICAL/arm 0.358); torch-compile-bf16 0.7 ms (IDENTICAL/arm 1.484) |
| algos | global-maxpool | synthetic | Xq | - | 1.1 | 1.1 | - | - | - | torch-eager-fp32 0.6 ms (IDENTICAL/arm 1.911); torch-compile-fp32 0.6 ms (IDENTICAL/arm 1.959); torch-eager-bf16 0.6 ms (IDENTICAL/arm 1.853); torch-compile-bf16 0.6 ms (IDENTICAL/arm 1.944) |
| algos | graphsage | istella | Xq | - | 201.4 | 187.8 | - | - | - | torch-eager-fp32 206.2 ms (IDENTICAL/arm 0.910); torch-compile-fp32 26.9 ms (IDENTICAL/arm 6.979); torch-eager-bf16 207.7 ms (IDENTICAL/arm 0.904); torch-compile-bf16 28.5 ms (IDENTICAL/arm 6.595) |
| algos | graphsage | taxi | Xq | - | 114.5 | 124.5 | - | - | - | torch-eager-fp32 12.0 ms (IDENTICAL/arm 10.377); torch-compile-fp32 7.5 ms (IDENTICAL/arm 16.646); torch-eager-bf16 11.0 ms (IDENTICAL/arm 11.360); torch-compile-bf16 8.0 ms (IDENTICAL/arm 15.590) |
| algos | ivf-filter | istella | Xq | - | 6248.5 | 4465.9 | - | - | - | faiss-cpu 175.4 ms (IDENTICAL/arm 25.456) |
| algos | ivf-filter | taxi | Xq | - | 212.3 | 216.7 | - | - | - | faiss-cpu 72.7 ms (IDENTICAL/arm 2.982) |
| algos | ivf-pq | istella | Xq | - | 4341.3 | 3754.2 | - | - | - | faiss-cpu 150.4 ms (IDENTICAL/arm 24.968) |
| algos | ivf-pq | taxi | Xq | - | 213.0 | 216.7 | - | - | - | faiss-cpu 44.6 ms (IDENTICAL/arm 4.856) |
| algos | ivf-rabitq | istella | Xq | - | 968.2 | 1060.1 | - | - | - | faiss-cpu 213.8 ms (IDENTICAL/arm 4.958) |
| algos | ivf-rabitq | taxi | Xq | - | 112.5 | 116.7 | - | - | - | faiss-cpu 67.4 ms (IDENTICAL/arm 1.731) |
| algos | ivf-refine | istella | Xq | - | 4748.1 | 5105.2 | - | - | - | faiss-cpu 144.7 ms (IDENTICAL/arm 35.287) |
| algos | ivf-refine | taxi | Xq | - | 475.8 | 484.1 | - | - | - | faiss-cpu 40.5 ms (IDENTICAL/arm 11.963) |
| algos | ivf-sq | istella | Xq | - | 5267.9 | 5563.9 | - | - | - | faiss-cpu 953.8 ms (IDENTICAL/arm 5.833) |
| algos | ivf-sq | taxi | Xq | - | 168.2 | 170.9 | - | - | - | faiss-cpu 64.2 ms (IDENTICAL/arm 2.662) |
| algos | logreg-cv | istella | Xq | - | - | 66.6 | - | - | - | sklearn-cpu 49.9 ms (IDENTICAL/arm 1.334) |
| algos | logreg-cv | taxi | Xq | - | - | - | - | - | - | sklearn-cpu 3.6 ms (IDENTICAL/arm -) |
| algos | maxpool1d | synthetic | Xq | - | 36.6 | 35.0 | - | - | - | torch-eager-fp32 3.4 ms (IDENTICAL/arm 10.373); torch-compile-fp32 1.2 ms (IDENTICAL/arm 30.116); torch-eager-bf16 1.9 ms (IDENTICAL/arm 18.628); torch-compile-bf16 1.2 ms (IDENTICAL/arm 29.593) |
| algos | maxpool2d | synthetic | Xq | - | 61.9 | 62.1 | - | - | - | torch-eager-fp32 2.6 ms (IDENTICAL/arm 23.457); torch-compile-fp32 2.1 ms (IDENTICAL/arm 29.486); torch-eager-bf16 2.6 ms (IDENTICAL/arm 23.501); torch-compile-bf16 2.0 ms (IDENTICAL/arm 30.622) |
| algos | multioutput-clf | istella | Xq | - | 52.0 | 61.7 | - | - | - | sklearn-cpu 49.9 ms (IDENTICAL/arm 1.238) |
| algos | multioutput-clf | taxi | Xq | - | 26.7 | 25.3 | - | - | - | sklearn-cpu 4.0 ms (IDENTICAL/arm 6.404) |
| algos | multioutput-reg | istella | Xq | - | 33.6 | 37.8 | - | - | - | sklearn-cpu 11.8 ms (IDENTICAL/arm 3.189) |
| algos | multioutput-reg | taxi | Xq | - | 7.2 | 7.2 | - | - | - | sklearn-cpu 0.9 ms (IDENTICAL/arm 7.569) |
| algos | ovr | istella | Xq | - | 143.4 | 160.7 | - | - | - | sklearn-cpu 222.4 ms (IDENTICAL/arm 0.723) |
| algos | ovr | taxi | Xq | - | 38.4 | 38.7 | - | - | - | sklearn-cpu 13.0 ms (IDENTICAL/arm 2.974) |
| algos | random-trees-embedding | istella | Xq | - | 27.9 | 26.5 | - | - | - | sklearn-cpu 126.2 ms (IDENTICAL/arm 0.210) |
| algos | random-trees-embedding | taxi | Xq | - | 28.5 | 49.1 | - | - | - | sklearn-cpu 64.0 ms (IDENTICAL/arm 0.768) |
| algos | resnet-block | synthetic | Xq | - | 232.9 | 242.5 | - | - | - | torch-eager-fp32 9.2 ms (IDENTICAL/arm 26.411); torch-compile-fp32 294.0 ms (IDENTICAL/arm 0.825); torch-eager-bf16 10.6 ms (IDENTICAL/arm 22.933); torch-compile-bf16 293.0 ms (IDENTICAL/arm 0.828) |
| algos | stacking-clf | istella | Xq | - | 77.6 | 76.8 | - | - | - | sklearn-cpu 231.1 ms (IDENTICAL/arm 0.332) |
| algos | stacking-clf | taxi | Xq | - | 33.4 | 32.3 | - | - | - | sklearn-cpu 31.8 ms (IDENTICAL/arm 1.016) |
| algos | stacking-reg | istella | Xq | - | - | - | - | - | - | sklearn-cpu 14.4 ms (IDENTICAL/arm -) |
| algos | stacking-reg | taxi | Xq | - | - | - | - | - | - | sklearn-cpu 5.1 ms (IDENTICAL/arm -) |
| algos | voting-clf | istella | Xq | - | 80.5 | 76.3 | - | - | - | sklearn-cpu 282.7 ms (IDENTICAL/arm 0.270) |
| algos | voting-clf | taxi | Xq | - | 15.3 | 15.1 | - | - | - | sklearn-cpu 33.1 ms (IDENTICAL/arm 0.456) |
| algos | voting-reg | istella | Xq | - | - | - | - | - | - | sklearn-cpu 20.1 ms (IDENTICAL/arm -) |
| algos | voting-reg | taxi | Xq | - | - | - | - | - | - | sklearn-cpu 5.6 ms (IDENTICAL/arm -) |

## Algorithm expansion

### avgpool1d / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.avgpool1d.synthetic.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 85.9 | 85.9..85.9 | 1 | - | - | - | 501.3 | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 79.7 | 79.7..79.7 | 1 | - | - | - | 505.6 | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 14.2 | 14.2..14.2 | 1 | 6.039 | 5.606 | - | 1258.6 | 1024.4 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 3.4 | 3.4..3.4 | 1 | 24.959 | 23.170 | - | 1351.3 | 1024.4 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 6.8 | 6.8..6.8 | 1 | 12.587 | 11.685 | - | 1259.0 | 1024.4 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 2.9 | 2.9..2.9 | 1 | 29.239 | 27.143 | - | 1352.9 | 1024.4 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 20.2 | 20.2..20.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 26.4 | 26.4..26.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | 2.6 | 2.6..2.6 | 1 | 7.748 | 10.143 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 1.2 | 1.2..1.2 | 1 | 17.152 | 22.455 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 2.6 | 2.6..2.6 | 1 | 7.683 | 10.059 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 1.2 | 1.2..1.2 | 1 | 17.280 | 22.622 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, ours-fast: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### avgpool2d / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.avgpool2d.synthetic.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 225.7 | 225.7..225.7 | 1 | - | - | - | 918.3 | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 226.0 | 226.0..226.0 | 1 | - | - | - | 918.1 | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 14.0 | 14.0..14.0 | 1 | 16.087 | 16.106 | - | 1229.8 | 1024.4 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 12.7 | 12.7..12.7 | 1 | 17.725 | 17.747 | - | 1356.8 | 1024.4 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 7.3 | 7.3..7.3 | 1 | 30.866 | 30.903 | - | 1226.7 | 1024.4 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 4.4 | 4.4..4.4 | 1 | 50.965 | 51.026 | - | 1349.1 | 1024.4 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 37.8 | 37.8..37.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 36.7 | 36.7..36.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | 2.2 | 2.2..2.2 | 1 | 17.451 | 16.971 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 2.1 | 2.1..2.1 | 1 | 18.382 | 17.877 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 2.0 | 2.0..2.0 | 1 | 18.961 | 18.440 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 2.1 | 2.1..2.1 | 1 | 18.371 | 17.866 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, ours-fast: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### batchnorm1d / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.batchnorm1d.synthetic.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 112.0 | 112.0..112.0 | 1 | - | - | - | 632.0 | - | max_rel_diff_vs_torch_eager_fp32=0.063434, rel_fro_vs_torch_eager_fp32=4.095e-06 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 102.2 | 102.2..102.2 | 1 | - | - | - | 633.6 | - | max_rel_diff_vs_torch_eager_fp32=0.063434, rel_fro_vs_torch_eager_fp32=4.096e-06 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 14.9 | 14.9..14.9 | 1 | 7.533 | 6.872 | - | 1303.3 | 1098.4 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 23.1 | 23.1..23.1 | 1 | 4.860 | 4.434 | - | 1365.0 | 1032.4 | max_rel_diff_vs_torch_eager_fp32=0.012127, rel_fro_vs_torch_eager_fp32=3.022e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 4.9 | 4.9..4.9 | 1 | 22.654 | 20.665 | - | 1308.0 | 1098.4 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 17.9 | 17.9..17.9 | 1 | 6.272 | 5.721 | - | 1361.4 | 1032.4 | max_rel_diff_vs_torch_eager_fp32=0.012127, rel_fro_vs_torch_eager_fp32=3.022e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 55.3 | 55.3..55.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 49.1 | 49.1..49.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | 2.0 | 2.0..2.0 | 1 | 27.900 | 24.768 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 6.2 | 6.2..6.2 | 1 | 8.956 | 7.950 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 2.2 | 2.2..2.2 | 1 | 25.074 | 22.259 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 12.1 | 12.1..12.1 | 1 | 4.563 | 4.051 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, ours-fast: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### batchnorm2d / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.batchnorm2d.synthetic.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 92.6 | 92.6..92.6 | 1 | - | - | - | 530.5 | - | max_rel_diff_vs_torch_eager_fp32=0.034926, rel_fro_vs_torch_eager_fp32=2.344e-05 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 93.7 | 93.7..93.7 | 1 | - | - | - | 531.2 | - | max_rel_diff_vs_torch_eager_fp32=0.034926, rel_fro_vs_torch_eager_fp32=2.344e-05 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 13.6 | 13.6..13.6 | 1 | 6.813 | 6.891 | - | 1291.9 | 1084.4 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 19.8 | 19.8..19.8 | 1 | 4.684 | 4.738 | - | 1384.7 | 1032.4 | max_rel_diff_vs_torch_eager_fp32=0.005340, rel_fro_vs_torch_eager_fp32=2.914e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 4.1 | 4.1..4.1 | 1 | 22.575 | 22.833 | - | 1292.5 | 1084.4 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 13.4 | 13.4..13.4 | 1 | 6.911 | 6.990 | - | 1378.5 | 1032.4 | max_rel_diff_vs_torch_eager_fp32=0.005340, rel_fro_vs_torch_eager_fp32=2.914e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 36.9 | 36.9..36.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 37.8 | 37.8..37.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | 1.9 | 1.9..1.9 | 1 | 18.991 | 19.468 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 5.5 | 5.5..5.5 | 1 | 6.770 | 6.940 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 1.9 | 1.9..1.9 | 1 | 19.514 | 20.004 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 4.8 | 4.8..4.8 | 1 | 7.729 | 7.923 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, ours-fast: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### bpe-encode / enwik8 (rows full, shape -)

race: done, driver rc 0, log `logs/algos.bpe-encode.enwik8.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 94.6 | 94.6..94.6 | 1 | - | - | - | 254.1 | - | documents_equal_to_ours=1.000000, tokens=1457323 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 87.2 | 87.2..87.2 | 1 | - | - | - | 242.4 | - | documents_equal_to_ours=1.000000, tokens=1457323 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| hf-tokenizers-cpu | tokenizers | cpu | opponent | 210.3 | 210.3..210.3 | 1 | 0.450 | 0.415 | - | 446.1 | - | documents_equal_to_ours=1.000000, tokens=1457323 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.bpe-train.enwik8.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 54.6 | 54.6..54.6 | 1 | - | - | - | 89.8 | - | jaccard_vs_ours=1.000000, n_tokens=4096 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 53.3 | 53.3..53.3 | 1 | - | - | - | 90.7 | - | jaccard_vs_ours=1.000000, n_tokens=4096 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| hf-tokenizers-cpu | tokenizers | cpu | opponent | 257.8 | 257.8..257.8 | 1 | 0.212 | 0.207 | - | 187.6 | - | jaccard_vs_ours=0.999512, n_tokens=4096 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.cagra.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "TypeError(\"CagraIndex.__init__() got an unexpected keyword argument 'random_state'\")", "event": "error", "stage": "round 0"}) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "TypeError(\"CagraIndex.__init__() got an unexpected keyword argument 'random_state'\")", "event": "error", "stage": "round 0"}) |
| faiss-cpu | faiss | cpu | opponent | 4264.7 | 4264.7..4264.7 | 1 | - | - | - | 1272.6 | - | recall_at_10=0.999375 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| faiss-cpu | Xq | - | 20.7 | 20.7..20.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: search(queries)(Xq)

inference call, ours-fast: search(queries)(Xq)

inference call, faiss-cpu: search(queries)(Xq)

### cagra / taxi (rows full, shape index 400000x11; queries 4000x11)

race: done, driver rc 0, log `logs/algos.cagra.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "TypeError(\"CagraIndex.__init__() got an unexpected keyword argument 'random_state'\")", "event": "error", "stage": "round 0"}) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "TypeError(\"CagraIndex.__init__() got an unexpected keyword argument 'random_state'\")", "event": "error", "stage": "round 0"}) |
| faiss-cpu | faiss | cpu | opponent | 2238.9 | 2238.9..2238.9 | 1 | - | - | - | 313.5 | - | recall_at_10=0.927700 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| faiss-cpu | Xq | - | 8.6 | 8.6..8.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: search(queries)(Xq)

inference call, ours-fast: search(queries)(Xq)

inference call, faiss-cpu: search(queries)(Xq)

### calibrated / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.calibrated.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 4268.9 | 4268.9..4268.9 | 1 | - | - | - | 3056.1 | - | accuracy=0.885090, logloss=0.289787 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 2630.7 | 2630.7..2630.7 | 1 | - | - | - | 3224.3 | - | accuracy=0.885090, logloss=0.289832 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 3683.1 | 3683.1..3683.1 | 1 | 1.159 | 0.714 | - | 2938.5 | - | accuracy=0.885090, logloss=0.289787 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 225.1 | 225.1..225.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 219.5 | 219.5..219.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 1060.1 | 1060.1..1060.1 | 1 | 0.212 | 0.207 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### calibrated / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.calibrated.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 2929.4 | 2929.4..2929.4 | 1 | - | - | - | 260.9 | - | accuracy=0.755330, logloss=0.550727 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1331.6 | 1331.6..1331.6 | 1 | - | - | - | 256.9 | - | accuracy=0.755330, logloss=0.550718 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 738.9 | 738.9..738.9 | 1 | 3.965 | 1.802 | - | 227.6 | - | accuracy=0.755330, logloss=0.550727 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 68.4 | 68.4..68.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 71.2 | 71.2..71.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 121.1 | 121.1..121.1 | 1 | 0.565 | 0.588 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### clip-grad-norm / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.clip-grad-norm.synthetic.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 49.2 | 49.2..49.2 | 1 | - | - | - | 208.9 | - | norm=1.000000, norm_rel_diff_vs_ours=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested identical, read back unknown) |
| mojolearn FAST | mojolearn | gpu | fast | 246.2 | 246.2..246.2 | 1 | - | - | - | 206.4 | - | norm=1.000000, norm_rel_diff_vs_ours=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested fast, read back unknown) |
| torch-eager-fp32 | torch | gpu | opponent | 22.4 | 22.4..22.4 | 1 | - | - | - | 421.0 | 136.4 | norm=1.000000, norm_rel_diff_vs_ours=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 22.6 | 22.6..22.6 | 1 | - | - | - | 536.2 | 136.4 | norm=1.000000, norm_rel_diff_vs_ours=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.cnn-clf.synthetic.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1749.8 | 1749.8..1749.8 | 1 | - | - | - | 487.3 | - | accuracy=1.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 2154.2 | 2154.2..2154.2 | 1 | - | - | - | 487.8 | - | accuracy=1.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
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
| mojolearn IDENTICAL | Xq | - | 93.8 | 93.8..93.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 79.0 | 79.0..79.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
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

### conv1d / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.conv1d.synthetic.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 343.7 | 343.7..343.7 | 1 | - | - | - | 2304.3 | - | max_rel_diff_vs_torch_eager_fp32=0.391155, rel_fro_vs_torch_eager_fp32=3.947e-07 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 347.4 | 347.4..347.4 | 1 | - | - | - | 2303.7 | - | max_rel_diff_vs_torch_eager_fp32=0.391155, rel_fro_vs_torch_eager_fp32=3.947e-07 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 38.5 | 38.5..38.5 | 1 | 8.926 | 9.022 | - | 1250.4 | 1040.4 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 33.5 | 33.5..33.5 | 1 | 10.270 | 10.380 | - | 1382.4 | 1040.4 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 52.6 | 52.6..52.6 | 1 | 6.532 | 6.602 | - | 1253.5 | 1040.7 | max_rel_diff_vs_torch_eager_fp32=3479.164094, rel_fro_vs_torch_eager_fp32=0.002870 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 35.9 | 35.9..35.9 | 1 | 9.574 | 9.676 | - | 1383.8 | 1040.7 | max_rel_diff_vs_torch_eager_fp32=3418.128937, rel_fro_vs_torch_eager_fp32=0.003296 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 92.5 | 92.5..92.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 87.6 | 87.6..87.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | 7.6 | 7.6..7.6 | 1 | 12.246 | 11.604 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 8.6 | 8.6..8.6 | 1 | 10.795 | 10.229 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 11.0 | 11.0..11.0 | 1 | 8.416 | 7.975 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 11.4 | 11.4..11.4 | 1 | 8.143 | 7.716 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, ours-fast: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### conv2d / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.conv2d.synthetic.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 265.4 | 265.4..265.4 | 1 | - | - | - | 1654.5 | - | max_rel_diff_vs_torch_eager_fp32=0.461936, rel_fro_vs_torch_eager_fp32=3.303e-07 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 208.4 | 208.4..208.4 | 1 | - | - | - | 1655.3 | - | max_rel_diff_vs_torch_eager_fp32=0.461936, rel_fro_vs_torch_eager_fp32=3.303e-07 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 22.8 | 22.8..22.8 | 1 | 11.642 | 9.139 | - | 1243.4 | 1036.4 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 21.5 | 21.5..21.5 | 1 | 12.336 | 9.684 | - | 1393.0 | 1036.4 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 26.1 | 26.1..26.1 | 1 | 10.180 | 7.991 | - | 1244.0 | 1036.7 | max_rel_diff_vs_torch_eager_fp32=3463.592380, rel_fro_vs_torch_eager_fp32=0.002935 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 14.5 | 14.5..14.5 | 1 | 18.346 | 14.402 | - | 1394.9 | 1036.7 | max_rel_diff_vs_torch_eager_fp32=3418.426961, rel_fro_vs_torch_eager_fp32=0.003382 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 46.2 | 46.2..46.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 43.8 | 43.8..43.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | 3.1 | 3.1..3.1 | 1 | 15.098 | 14.323 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 4.9 | 4.9..4.9 | 1 | 9.386 | 8.903 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 3.4 | 3.4..3.4 | 1 | 13.571 | 12.874 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 4.8 | 4.8..4.8 | 1 | 9.720 | 9.221 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, ours-fast: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### cross-entropy / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.cross-entropy.synthetic.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 358.5 | 358.5..358.5 | 1 | - | - | - | 2136.5 | - | loss_rel_err_vs_fp64=4.564e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested identical, read back unknown) |
| mojolearn FAST | mojolearn | gpu | fast | 367.5 | 367.5..367.5 | 1 | - | - | - | 2135.4 | - | grad_max_rel_diff_vs_ours=1.863e-09, loss_rel_err_vs_fp64=4.564e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested fast, read back unknown) |
| torch-eager-fp32 | torch | gpu | opponent | 17.6 | 17.6..17.6 | 1 | - | - | - | 1571.9 | 1290.4 | grad_max_rel_diff_vs_ours=3.725e-09, loss_rel_err_vs_fp64=4.564e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 8.3 | 8.3..8.3 | 1 | - | - | - | 1471.0 | 1032.4 | grad_max_rel_diff_vs_ours=5.588e-09, loss_rel_err_vs_fp64=4.564e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-eager-fp32, torch-compile-fp32: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

settings: {'ignore_index': -100, 'label_smoothing': 0.0, 'reduction': 'mean'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | torch-compile-fp32 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | "none (deterministic)" | 7 | 7 |

### dropout2d / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.dropout2d.synthetic.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 92.1 | 92.1..92.1 | 1 | - | - | - | 770.0 | - | error=KeyError('y') | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 97.8 | 97.8..97.8 | 1 | - | - | - | 767.9 | - | error=KeyError('y') | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 10.4 | 10.4..10.4 | 1 | 8.879 | 9.432 | - | 1232.0 | 1032.4 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 2.6 | 2.6..2.6 | 1 | 35.510 | 37.720 | - | 1328.8 | 1032.4 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 68.5 | 68.5..68.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 68.5 | 68.5..68.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | 5.2 | 5.2..5.2 | 1 | 13.207 | 13.198 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 0.9 | 0.9..0.9 | 1 | 80.015 | 79.962 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, ours-fast: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

### embedding / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.embedding.synthetic.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 871.8 | 871.8..871.8 | 1 | - | - | - | 1550.5 | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested identical, read back unknown) |
| mojolearn FAST | mojolearn | gpu | fast | 877.2 | 877.2..877.2 | 1 | - | - | - | 1555.3 | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested fast, read back unknown) |
| torch-eager-fp32 | torch | gpu | opponent | 23.4 | 23.4..23.4 | 1 | - | - | - | 1237.4 | 1032.4 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 19.6 | 19.6..19.6 | 1 | - | - | - | 1364.6 | 1032.4 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 363.7 | 363.7..363.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 372.5 | 372.5..372.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | 2.3 | 2.3..2.3 | 1 | 156.987 | 160.783 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 2.6 | 2.6..2.6 | 1 | 139.787 | 143.167 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, ours-fast: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

### gcn / istella (rows full, shape X 100000x220; indices 1521510; indptr 100001; y 100000)

race: done, driver rc 0, log `logs/algos.gcn.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 254.4 | 254.4..254.4 | 1 | - | - | - | 1146.5 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 248.9 | 248.9..248.9 | 1 | - | - | - | 1141.0 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 393.0 | 393.0..393.0 | 1 | 0.647 | 0.633 | - | 3874.3 | 3429.0 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 44.9 | 44.9..44.9 | 1 | 5.672 | 5.550 | - | 1550.7 | 1044.6 | max_rel_diff_vs_torch_eager_fp32=0.014120, rel_fro_vs_torch_eager_fp32=2.389e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 379.1 | 379.1..379.1 | 1 | 0.671 | 0.657 | - | 3963.5 | 3523.0 | max_rel_diff_vs_torch_eager_fp32=2744.908399, rel_fro_vs_torch_eager_fp32=0.002187 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 112.2 | 112.2..112.2 | 1 | 2.269 | 2.220 | - | 2446.6 | 1938.6 | max_rel_diff_vs_torch_eager_fp32=2744.917456, rel_fro_vs_torch_eager_fp32=0.002187 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 130.5 | 130.5..130.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 140.7 | 140.7..140.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | 148.0 | 148.0..148.0 | 1 | 0.882 | 0.951 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 15.3 | 15.3..15.3 | 1 | 8.538 | 9.204 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 146.8 | 146.8..146.8 | 1 | 0.889 | 0.958 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 15.8 | 15.8..15.8 | 1 | 8.276 | 8.921 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, ours-fast: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### gcn / taxi (rows full, shape X 100000x11; indices 1258298; indptr 100001; y 100000)

race: done, driver rc 0, log `logs/algos.gcn.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 179.1 | 179.1..179.1 | 1 | - | - | - | 842.8 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 182.2 | 182.2..182.2 | 1 | - | - | - | 851.3 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 339.1 | 339.1..339.1 | 1 | 0.528 | 0.537 | - | 2811.1 | 2439.0 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 36.5 | 36.5..36.5 | 1 | 4.912 | 4.995 | - | 1533.2 | 1102.6 | max_rel_diff_vs_torch_eager_fp32=0.005122, rel_fro_vs_torch_eager_fp32=2.408e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 314.1 | 314.1..314.1 | 1 | 0.570 | 0.580 | - | 3572.1 | 3201.0 | max_rel_diff_vs_torch_eager_fp32=1509.511378, rel_fro_vs_torch_eager_fp32=0.002330 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 94.2 | 94.2..94.2 | 1 | 1.901 | 1.933 | - | 2310.2 | 1872.6 | max_rel_diff_vs_torch_eager_fp32=1509.509399, rel_fro_vs_torch_eager_fp32=0.002330 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 112.9 | 112.9..112.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 123.3 | 123.3..123.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | 122.1 | 122.1..122.1 | 1 | 0.925 | 1.010 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 12.3 | 12.3..12.3 | 1 | 9.153 | 9.990 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 120.2 | 120.2..120.2 | 1 | 0.940 | 1.026 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 11.9 | 11.9..11.9 | 1 | 9.510 | 10.380 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, ours-fast: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### global-avgpool / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.global-avgpool.synthetic.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 11.7 | 11.7..11.7 | 1 | - | - | - | 206.8 | - | max_rel_diff_vs_torch_eager_fp32=0.015661, rel_fro_vs_torch_eager_fp32=1.493e-07 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 12.3 | 12.3..12.3 | 1 | - | - | - | 202.5 | - | max_rel_diff_vs_torch_eager_fp32=0.015661, rel_fro_vs_torch_eager_fp32=1.493e-07 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 11.1 | 11.1..11.1 | 1 | 1.059 | 1.107 | - | 242.4 | 40.4 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 1.7 | 1.7..1.7 | 1 | 7.098 | 7.421 | - | 371.5 | 40.4 | max_rel_diff_vs_torch_eager_fp32=0.007299, rel_fro_vs_torch_eager_fp32=9.321e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 3.7 | 3.7..3.7 | 1 | 3.192 | 3.337 | - | 243.3 | 40.4 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 1.6 | 1.6..1.6 | 1 | 7.318 | 7.652 | - | 368.2 | 40.4 | max_rel_diff_vs_torch_eager_fp32=0.007299, rel_fro_vs_torch_eager_fp32=9.321e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 1.1 | 1.1..1.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 1.1 | 1.1..1.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | 8.4 | 8.4..8.4 | 1 | 0.133 | 0.125 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 0.8 | 0.8..0.8 | 1 | 1.317 | 1.245 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 3.1 | 3.1..3.1 | 1 | 0.358 | 0.338 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 0.7 | 0.7..0.7 | 1 | 1.484 | 1.402 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, ours-fast: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### global-maxpool / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.global-maxpool.synthetic.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 4.5 | 4.5..4.5 | 1 | - | - | - | 210.9 | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 4.3 | 4.3..4.3 | 1 | - | - | - | 208.0 | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 6.6 | 6.6..6.6 | 1 | 0.682 | 0.656 | - | 248.1 | 42.4 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 3.9 | 3.9..3.9 | 1 | 1.138 | 1.095 | - | 371.6 | 42.4 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 5.6 | 5.6..5.6 | 1 | 0.794 | 0.764 | - | 247.3 | 42.4 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 3.7 | 3.7..3.7 | 1 | 1.208 | 1.162 | - | 374.0 | 42.4 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 1.1 | 1.1..1.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 1.1 | 1.1..1.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | 0.6 | 0.6..0.6 | 1 | 1.911 | 1.870 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 0.6 | 0.6..0.6 | 1 | 1.959 | 1.916 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 0.6 | 0.6..0.6 | 1 | 1.853 | 1.814 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 0.6 | 0.6..0.6 | 1 | 1.944 | 1.902 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, ours-fast: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### graphsage / istella (rows full, shape X 100000x220; indices 1521510; indptr 100001; y 100000)

race: done, driver rc 0, log `logs/algos.graphsage.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 439.6 | 439.6..439.6 | 1 | - | - | - | 1281.1 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 442.1 | 442.1..442.1 | 1 | - | - | - | 1279.0 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 240.6 | 240.6..240.6 | 1 | 1.827 | 1.838 | - | 2766.3 | 2326.6 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 45.5 | 45.5..45.5 | 1 | 9.665 | 9.722 | - | 1597.5 | 1094.6 | max_rel_diff_vs_torch_eager_fp32=0.089407, rel_fro_vs_torch_eager_fp32=7.979e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 234.3 | 234.3..234.3 | 1 | 1.876 | 1.887 | - | 2767.0 | 2326.6 | max_rel_diff_vs_torch_eager_fp32=3907.114267, rel_fro_vs_torch_eager_fp32=0.003366 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 46.1 | 46.1..46.1 | 1 | 9.540 | 9.595 | - | 1573.9 | 1064.6 | max_rel_diff_vs_torch_eager_fp32=3678.187728, rel_fro_vs_torch_eager_fp32=0.003054 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 187.8 | 187.8..187.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 201.4 | 201.4..201.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | 206.2 | 206.2..206.2 | 1 | 0.910 | 0.976 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 26.9 | 26.9..26.9 | 1 | 6.979 | 7.485 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 207.7 | 207.7..207.7 | 1 | 0.904 | 0.970 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 28.5 | 28.5..28.5 | 1 | 6.595 | 7.074 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, ours-fast: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### graphsage / taxi (rows full, shape X 100000x11; indices 1258298; indptr 100001; y 100000)

race: done, driver rc 0, log `logs/algos.graphsage.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 172.4 | 172.4..172.4 | 1 | - | - | - | 863.1 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 154.1 | 154.1..154.1 | 1 | - | - | - | 847.5 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 43.2 | 43.2..43.2 | 1 | 3.987 | 3.565 | - | 1432.1 | 1070.6 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 25.2 | 25.2..25.2 | 1 | 6.852 | 6.127 | - | 1552.0 | 1120.6 | max_rel_diff_vs_torch_eager_fp32=0.153846, rel_fro_vs_torch_eager_fp32=4.956e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 28.1 | 28.1..28.1 | 1 | 6.123 | 5.475 | - | 1432.4 | 1068.6 | max_rel_diff_vs_torch_eager_fp32=5460.333333, rel_fro_vs_torch_eager_fp32=0.003557 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 13.3 | 13.3..13.3 | 1 | 12.931 | 11.563 | - | 1525.1 | 1094.6 | max_rel_diff_vs_torch_eager_fp32=4608.154297, rel_fro_vs_torch_eager_fp32=0.003285 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 124.5 | 124.5..124.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 114.5 | 114.5..114.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | 12.0 | 12.0..12.0 | 1 | 10.377 | 9.540 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 7.5 | 7.5..7.5 | 1 | 16.646 | 15.303 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 11.0 | 11.0..11.0 | 1 | 11.360 | 10.444 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 8.0 | 8.0..8.0 | 1 | 15.590 | 14.332 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, ours-fast: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### ivf-filter / istella (rows full, shape index 400000x220; queries 4000x220)

race: done, driver rc 0, log `logs/algos.ivf-filter.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 25598.3 | 25598.3..25598.3 | 1 | - | - | - | 2531.5 | - | recall_at_10=0.609250 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 18038.5 | 18038.5..18038.5 | 1 | - | - | - | 2917.5 | - | recall_at_10=0.589100 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| faiss-cpu | faiss | cpu | opponent | 15163.1 | 15163.1..15163.1 | 1 | 1.688 | 1.190 | - | 573.7 | - | recall_at_10=0.841950 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 4465.9 | 4465.9..4465.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 6248.5 | 6248.5..6248.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| faiss-cpu | Xq | - | 175.4 | 175.4..175.4 | 1 | 25.456 | 35.617 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: search(queries)(Xq)

inference call, ours-fast: search(queries)(Xq)

inference call, faiss-cpu: search(queries)(Xq)

### ivf-filter / taxi (rows full, shape index 400000x11; queries 4000x11)

race: done, driver rc 0, log `logs/algos.ivf-filter.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3612.1 | 3612.1..3612.1 | 1 | - | - | - | 520.1 | - | recall_at_10=0.980075 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1757.9 | 1757.9..1757.9 | 1 | - | - | - | 536.0 | - | recall_at_10=0.982950 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| faiss-cpu | faiss | cpu | opponent | 8668.2 | 8668.2..8668.2 | 1 | 0.417 | 0.203 | - | 150.8 | - | recall_at_10=0.982050 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 216.7 | 216.7..216.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 212.3 | 212.3..212.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| faiss-cpu | Xq | - | 72.7 | 72.7..72.7 | 1 | 2.982 | 2.922 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: search(queries)(Xq)

inference call, ours-fast: search(queries)(Xq)

inference call, faiss-cpu: search(queries)(Xq)

### ivf-pq / istella (rows full, shape index 400000x220; queries 4000x220)

race: done, driver rc 0, log `logs/algos.ivf-pq.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 26141.2 | 26141.2..26141.2 | 1 | - | - | - | 2519.5 | - | recall_at_10=0.565025 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 18753.7 | 18753.7..18753.7 | 1 | - | - | - | 2916.6 | - | recall_at_10=0.599475 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| faiss-cpu | faiss | cpu | opponent | 15152.0 | 15152.0..15152.0 | 1 | 1.725 | 1.238 | - | 572.9 | - | recall_at_10=0.802975 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 3754.2 | 3754.2..3754.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 4341.3 | 4341.3..4341.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| faiss-cpu | Xq | - | 150.4 | 150.4..150.4 | 1 | 24.968 | 28.873 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: search(queries)(Xq)

inference call, ours-fast: search(queries)(Xq)

inference call, faiss-cpu: search(queries)(Xq)

### ivf-pq / taxi (rows full, shape index 400000x11; queries 4000x11)

race: done, driver rc 0, log `logs/algos.ivf-pq.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3648.9 | 3648.9..3648.9 | 1 | - | - | - | 520.3 | - | recall_at_10=0.973225 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1754.9 | 1754.9..1754.9 | 1 | - | - | - | 534.0 | - | recall_at_10=0.976450 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| faiss-cpu | faiss | cpu | opponent | 8843.4 | 8843.4..8843.4 | 1 | 0.413 | 0.198 | - | 152.7 | - | recall_at_10=0.980075 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 216.7 | 216.7..216.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 213.0 | 213.0..213.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| faiss-cpu | Xq | - | 44.6 | 44.6..44.6 | 1 | 4.856 | 4.774 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: search(queries)(Xq)

inference call, ours-fast: search(queries)(Xq)

inference call, faiss-cpu: search(queries)(Xq)

### ivf-rabitq / istella (rows full, shape index 400000x220; queries 4000x220)

race: done, driver rc 0, log `logs/algos.ivf-rabitq.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 16299.8 | 16299.8..16299.8 | 1 | - | - | - | 1659.3 | - | recall_at_10=0.125125 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 14501.4 | 14501.4..14501.4 | 1 | - | - | - | 2365.5 | - | recall_at_10=0.132550 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| faiss-cpu | faiss | cpu | opponent | 12874.1 | 12874.1..12874.1 | 1 | 1.266 | 1.126 | - | 453.7 | - | recall_at_10=0.050450 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 1060.1 | 1060.1..1060.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 968.2 | 968.2..968.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| faiss-cpu | Xq | - | 213.8 | 213.8..213.8 | 1 | 4.958 | 4.528 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: search(queries)(Xq)

inference call, ours-fast: search(queries)(Xq)

inference call, faiss-cpu: search(queries)(Xq)

### ivf-rabitq / taxi (rows full, shape index 400000x11; queries 4000x11)

race: done, driver rc 0, log `logs/algos.ivf-rabitq.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1838.5 | 1838.5..1838.5 | 1 | - | - | - | 448.8 | - | recall_at_10=0.110475 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 927.6 | 927.6..927.6 | 1 | - | - | - | 465.5 | - | recall_at_10=0.115775 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| faiss-cpu | faiss | cpu | opponent | 8106.0 | 8106.0..8106.0 | 1 | 0.227 | 0.114 | - | 115.2 | - | recall_at_10=0.126275 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 116.7 | 116.7..116.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 112.5 | 112.5..112.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| faiss-cpu | Xq | - | 67.4 | 67.4..67.4 | 1 | 1.731 | 1.669 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: search(queries)(Xq)

inference call, ours-fast: search(queries)(Xq)

inference call, faiss-cpu: search(queries)(Xq)

### ivf-refine / istella (rows full, shape index 400000x220; queries 4000x220)

race: done, driver rc 0, log `logs/algos.ivf-refine.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 25638.2 | 25638.2..25638.2 | 1 | - | - | - | 2527.4 | - | recall_at_10=0.809175 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 18785.2 | 18785.2..18785.2 | 1 | - | - | - | 2917.0 | - | recall_at_10=0.862250 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| faiss-cpu | faiss | cpu | opponent | 15153.2 | 15153.2..15153.2 | 1 | 1.692 | 1.240 | - | 1238.4 | - | recall_at_10=0.993425 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 5105.2 | 5105.2..5105.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 4748.1 | 4748.1..4748.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| faiss-cpu | Xq | - | 144.7 | 144.7..144.7 | 1 | 35.287 | 32.819 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: search(queries)(Xq)

inference call, ours-fast: search(queries)(Xq)

inference call, faiss-cpu: search(queries)(Xq)

### ivf-refine / taxi (rows full, shape index 400000x11; queries 4000x11)

race: done, driver rc 0, log `logs/algos.ivf-refine.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3662.4 | 3662.4..3662.4 | 1 | - | - | - | 524.5 | - | recall_at_10=0.999675 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1820.2 | 1820.2..1820.2 | 1 | - | - | - | 540.3 | - | recall_at_10=0.999675 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| faiss-cpu | faiss | cpu | opponent | 8766.2 | 8766.2..8766.2 | 1 | 0.418 | 0.208 | - | 182.8 | - | recall_at_10=0.999225 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 484.1 | 484.1..484.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 475.8 | 475.8..475.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| faiss-cpu | Xq | - | 40.5 | 40.5..40.5 | 1 | 11.963 | 11.759 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: search(queries)(Xq)

inference call, ours-fast: search(queries)(Xq)

inference call, faiss-cpu: search(queries)(Xq)

### ivf-sq / istella (rows full, shape index 400000x220; queries 4000x220)

race: done, driver rc 0, log `logs/algos.ivf-sq.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 16449.8 | 16449.8..16449.8 | 1 | - | - | - | 4474.4 | - | recall_at_10=0.728025 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 14392.1 | 14392.1..14392.1 | 1 | - | - | - | 4198.9 | - | recall_at_10=0.631000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| faiss-cpu | faiss | cpu | opponent | 11827.2 | 11827.2..11827.2 | 1 | 1.391 | 1.217 | - | 623.3 | - | recall_at_10=0.591300 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 5563.9 | 5563.9..5563.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 5267.9 | 5267.9..5267.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| faiss-cpu | Xq | - | 953.8 | 953.8..953.8 | 1 | 5.833 | 5.523 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: search(queries)(Xq)

inference call, ours-fast: search(queries)(Xq)

inference call, faiss-cpu: search(queries)(Xq)

### ivf-sq / taxi (rows full, shape index 400000x11; queries 4000x11)

race: done, driver rc 0, log `logs/algos.ivf-sq.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1729.3 | 1729.3..1729.3 | 1 | - | - | - | 509.0 | - | recall_at_10=0.934975 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 929.4 | 929.4..929.4 | 1 | - | - | - | 538.4 | - | recall_at_10=0.902025 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| faiss-cpu | faiss | cpu | opponent | 7806.0 | 7806.0..7806.0 | 1 | 0.222 | 0.119 | - | 117.9 | - | recall_at_10=0.857025 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 170.9 | 170.9..170.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 168.2 | 168.2..168.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| faiss-cpu | Xq | - | 64.2 | 64.2..64.2 | 1 | 2.662 | 2.619 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: search(queries)(Xq)

inference call, ours-fast: search(queries)(Xq)

inference call, faiss-cpu: search(queries)(Xq)

### jl-min-dim / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.jl-min-dim.synthetic.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 7.0 | 7.0..7.0 | 1 | - | - | - | 62.7 | - | equal_fraction_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 6.6 | 6.6..6.6 | 1 | - | - | - | 63.5 | - | equal_fraction_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 201.0 | 201.0..201.0 | 1 | 0.035 | 0.033 | - | 147.7 | - | equal_fraction_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | sklearn (declared) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### kernel-shap / istella (rows full, shape X 100000x220; Xq 100x220; y 100000; yq 100)

race: done, driver rc 0, log `logs/algos.kernel-shap.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 16104.2 | 16104.2..16104.2 | 1 | - | - | - | 1138.3 | - | rel_error_vs_exact=4.179e-09 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 16140.6 | 16140.6..16140.6 | 1 | - | - | - | 1156.3 | - | rel_error_vs_exact=4.179e-09 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| shap-cpu | shap | cpu | opponent | 9884.3 | 9884.3..9884.3 | 1 | 1.629 | 1.633 | - | 1404.5 | - | rel_error_vs_exact=1.405e-14 | - | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.kernel-shap.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 756.1 | 756.1..756.1 | 1 | - | - | - | 146.9 | - | rel_error_vs_exact=1.731e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 617.0 | 617.0..617.0 | 1 | - | - | - | 144.4 | - | rel_error_vs_exact=1.731e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| shap-cpu | shap | cpu | opponent | 1253.4 | 1253.4..1253.4 | 1 | 0.603 | 0.492 | - | 284.6 | - | rel_error_vs_exact=6.394e-14 | - | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, shap-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'l1_reg': False, 'link': 'identity', 'n_background': 100, 'nsamples': 2048}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | shap-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | shap (declared) |
| seed | 7 | 7 | 7 |

### kpss / synthetic (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc 0, log `logs/algos.kpss.synthetic.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 4.2 | 4.2..4.2 | 1 | - | - | - | 72.1 | - | flag_agreement_vs_statsmodels=1.000000, stat_max_rel_diff_vs_statsmodels=4.318e-05, stationary_fraction=0.031250 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 4.1 | 4.1..4.1 | 1 | - | - | - | 75.2 | - | flag_agreement_vs_statsmodels=1.000000, stat_max_rel_diff_vs_statsmodels=4.062e-05, stationary_fraction=0.031250 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsmodels-cpu | statsmodels | cpu | opponent | 3.1 | 3.1..3.1 | 1 | 1.347 | 1.311 | - | 162.8 | - | flag_agreement_vs_statsmodels=1.000000, stat_max_rel_diff_vs_statsmodels=0.000000, stationary_fraction=0.031250 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.kpss.taxi-hourly.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 4.2 | 4.2..4.2 | 1 | - | - | - | 69.5 | - | flag_agreement_vs_statsmodels=1.000000, stat_max_rel_diff_vs_statsmodels=3.946e-05, stationary_fraction=0.687500 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 4.2 | 4.2..4.2 | 1 | - | - | - | 69.8 | - | flag_agreement_vs_statsmodels=1.000000, stat_max_rel_diff_vs_statsmodels=3.946e-05, stationary_fraction=0.687500 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsmodels-cpu | statsmodels | cpu | opponent | 3.2 | 3.2..3.2 | 1 | 1.337 | 1.325 | - | 160.5 | - | flag_agreement_vs_statsmodels=1.000000, stat_max_rel_diff_vs_statsmodels=0.000000, stationary_fraction=0.687500 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

### logreg-cv / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.logreg-cv.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 688.1 | 688.1..688.1 | 1 | - | - | - | 1984.6 | - | accuracy=0.885090, logloss=0.693147 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(timeout: null) |
| sklearn-cpu | scikit-learn | cpu | opponent | 95382.2 | 95382.2..95382.2 | 1 | 0.007 | - | - | 2798.7 | - | accuracy=0.924630, logloss=0.181351 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, ours-fast: host not sampled; GPU not sampled

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
| mojolearn IDENTICAL | Xq | - | 66.6 | 66.6..66.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(timeout: null) |
| sklearn-cpu | Xq | - | 49.9 | 49.9..49.9 | 1 | 1.334 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### logreg-cv / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.logreg-cv.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(timeout: null) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(timeout: null) |
| sklearn-cpu | scikit-learn | cpu | opponent | 1934.7 | 1934.7..1934.7 | 1 | - | - | - | 332.3 | - | accuracy=0.763300, logloss=0.538988 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| sklearn-cpu | Xq | - | 3.6 | 3.6..3.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### lr-constant / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.lr-constant.synthetic.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 676.2 | 676.2..676.2 | 1 | - | - | - | 63.3 | - | max_rel_diff_vs_ours=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested identical, read back unknown) |
| mojolearn FAST | mojolearn | gpu | fast | 681.3 | 681.3..681.3 | 1 | - | - | - | 62.3 | - | max_rel_diff_vs_ours=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested fast, read back unknown) |
| torch-cpu | torch | cpu | opponent | 104.9 | 104.9..104.9 | 1 | - | - | - | 327.6 | - | max_rel_diff_vs_ours=1.038e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.lr-exponential.synthetic.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 271.0 | 271.0..271.0 | 1 | - | - | - | 61.3 | - | max_rel_diff_vs_ours=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 266.3 | 266.3..266.3 | 1 | - | - | - | 65.3 | - | max_rel_diff_vs_ours=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-cpu | torch | cpu | opponent | 96.7 | 96.7..96.7 | 1 | 2.804 | 2.755 | - | 324.8 | - | max_rel_diff_vs_ours=5.933e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.lr-onecycle.synthetic.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1618.7 | 1618.7..1618.7 | 1 | - | - | - | 63.2 | - | max_rel_diff_vs_ours=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1621.5 | 1621.5..1621.5 | 1 | - | - | - | 64.3 | - | max_rel_diff_vs_ours=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-cpu | torch | cpu | opponent | 136.1 | 136.1..136.1 | 1 | 11.897 | 11.917 | - | 326.1 | - | max_rel_diff_vs_ours=5.951e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.lr-step.synthetic.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 15.5 | 15.5..15.5 | 1 | - | - | - | 59.6 | - | max_rel_diff_vs_ours=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 16.4 | 16.4..16.4 | 1 | - | - | - | 60.3 | - | max_rel_diff_vs_ours=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-cpu | torch | cpu | opponent | 122.3 | 122.3..122.3 | 1 | 0.127 | 0.135 | - | 330.8 | - | max_rel_diff_vs_ours=1.49e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.lr-warmup-cosine.synthetic.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 630.0 | 630.0..630.0 | 1 | - | - | - | 64.1 | - | max_rel_diff_vs_ours=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested identical, read back unknown) |
| mojolearn FAST | mojolearn | gpu | fast | 630.5 | 630.5..630.5 | 1 | - | - | - | 63.5 | - | max_rel_diff_vs_ours=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested fast, read back unknown) |
| torch-cpu | torch | cpu | opponent | 137.4 | 137.4..137.4 | 1 | - | - | - | 327.3 | - | max_rel_diff_vs_ours=0.0001571 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'min_lr': 1e-05, 'peak_lr': 0.001, 'total_steps': 100000, 'warmup_steps': 1000}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | torch-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | torch (declared) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### lr-warmup-linear / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.lr-warmup-linear.synthetic.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 937.0 | 937.0..937.0 | 1 | - | - | - | 62.1 | - | max_rel_diff_vs_ours=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested identical, read back unknown) |
| mojolearn FAST | mojolearn | gpu | fast | 945.8 | 945.8..945.8 | 1 | - | - | - | 65.5 | - | max_rel_diff_vs_ours=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested fast, read back unknown) |
| torch-cpu | torch | cpu | opponent | 119.7 | 119.7..119.7 | 1 | - | - | - | 326.3 | - | max_rel_diff_vs_ours=0.001000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'min_lr': 1e-05, 'peak_lr': 0.001, 'total_steps': 100000, 'warmup_steps': 1000}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | torch-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | torch (declared) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### maxpool1d / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.maxpool1d.synthetic.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 73.3 | 73.3..73.3 | 1 | - | - | - | 630.8 | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 78.6 | 78.6..78.6 | 1 | - | - | - | 630.9 | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 20.9 | 20.9..20.9 | 1 | 3.513 | 3.768 | - | 1293.6 | 1056.4 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 16.2 | 16.2..16.2 | 1 | 4.514 | 4.842 | - | 1359.7 | 1024.4 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 9.3 | 9.3..9.3 | 1 | 7.915 | 8.489 | - | 1293.9 | 1056.4 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 5.7 | 5.7..5.7 | 1 | 12.745 | 13.669 | - | 1352.5 | 1024.4 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 35.0 | 35.0..35.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 36.6 | 36.6..36.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | 3.4 | 3.4..3.4 | 1 | 10.373 | 10.858 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 1.2 | 1.2..1.2 | 1 | 30.116 | 31.523 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 1.9 | 1.9..1.9 | 1 | 18.628 | 19.499 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 1.2 | 1.2..1.2 | 1 | 29.593 | 30.977 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, ours-fast: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### maxpool2d / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.maxpool2d.synthetic.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 172.4 | 172.4..172.4 | 1 | - | - | - | 1067.1 | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 184.1 | 184.1..184.1 | 1 | - | - | - | 1066.4 | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 28.1 | 28.1..28.1 | 1 | 6.144 | 6.560 | - | 1280.3 | 1074.4 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 19.5 | 19.5..19.5 | 1 | 8.834 | 9.431 | - | 1408.2 | 1074.4 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 11.7 | 11.7..11.7 | 1 | 14.683 | 15.676 | - | 1277.7 | 1074.4 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 11.5 | 11.5..11.5 | 1 | 15.022 | 16.038 | - | 1402.9 | 1074.4 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 62.1 | 62.1..62.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 61.9 | 61.9..61.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | 2.6 | 2.6..2.6 | 1 | 23.457 | 23.361 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 2.1 | 2.1..2.1 | 1 | 29.486 | 29.365 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 2.6 | 2.6..2.6 | 1 | 23.501 | 23.404 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 2.0 | 2.0..2.0 | 1 | 30.622 | 30.495 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, ours-fast: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### multioutput-clf / istella (rows full, shape X 1000000x219; Xq 100000x219; Y 1000000x2; Yq 100000x2; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.multioutput-clf.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 13528.3 | 13528.3..13528.3 | 1 | - | - | - | 3782.5 | - | accuracy=0.959110 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 10544.6 | 10544.6..10544.6 | 1 | - | - | - | 3772.8 | - | accuracy=0.959115 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 40953.3 | 40953.3..40953.3 | 1 | 0.330 | 0.257 | - | 3687.6 | - | accuracy=0.959225 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 61.7 | 61.7..61.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 52.0 | 52.0..52.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 49.9 | 49.9..49.9 | 1 | 1.238 | 1.042 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### multioutput-clf / taxi (rows full, shape X 1000000x10; Xq 100000x10; Y 1000000x2; Yq 100000x2; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.multioutput-clf.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 364.4 | 364.4..364.4 | 1 | - | - | - | 425.7 | - | accuracy=0.863560 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 364.8 | 364.8..364.8 | 1 | - | - | - | 423.5 | - | accuracy=0.863560 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 819.9 | 819.9..819.9 | 1 | 0.444 | 0.445 | - | 261.9 | - | accuracy=0.863550 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 25.3 | 25.3..25.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 26.7 | 26.7..26.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 4.0 | 4.0..4.0 | 1 | 6.404 | 6.752 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### multioutput-reg / istella (rows full, shape X 1000000x219; Xq 100000x219; Y 1000000x2; Yq 100000x2; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.multioutput-reg.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 5854.2 | 5854.2..5854.2 | 1 | - | - | - | 6111.9 | - | r2=0.455327 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 4435.2 | 4435.2..4435.2 | 1 | - | - | - | 6112.5 | - | r2=0.439854 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 18240.0 | 18240.0..18240.0 | 1 | 0.321 | 0.243 | - | 9527.6 | - | r2=0.455345 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 37.8 | 37.8..37.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 33.6 | 33.6..33.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 11.8 | 11.8..11.8 | 1 | 3.189 | 2.841 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### multioutput-reg / taxi (rows full, shape X 1000000x10; Xq 100000x10; Y 1000000x2; Yq 100000x2; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.multioutput-reg.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 170.1 | 170.1..170.1 | 1 | - | - | - | 439.1 | - | r2=0.604240 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 160.0 | 160.0..160.0 | 1 | - | - | - | 436.7 | - | r2=0.604241 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 71.2 | 71.2..71.2 | 1 | 2.391 | 2.248 | - | 245.9 | - | r2=0.604316 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 7.2 | 7.2..7.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 7.2 | 7.2..7.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 0.9 | 0.9..0.9 | 1 | 7.569 | 7.597 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### ovr / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.ovr.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 33770.8 | 33770.8..33770.8 | 1 | - | - | - | 2753.6 | - | accuracy=0.892670 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 26140.6 | 26140.6..26140.6 | 1 | - | - | - | 2753.5 | - | accuracy=0.892760 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 137076.5 | 137076.5..137076.5 | 1 | 0.246 | 0.191 | - | 2775.1 | - | accuracy=0.892730 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 160.7 | 160.7..160.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 143.4 | 143.4..143.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 222.4 | 222.4..222.4 | 1 | 0.723 | 0.645 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### ovr / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.ovr.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 729.5 | 729.5..729.5 | 1 | - | - | - | 267.7 | - | accuracy=0.478940 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 537.8 | 537.8..537.8 | 1 | - | - | - | 246.8 | - | accuracy=0.478970 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1980.4 | 1980.4..1980.4 | 1 | 0.368 | 0.272 | - | 225.7 | - | accuracy=0.478890 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 38.7 | 38.7..38.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 38.4 | 38.4..38.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 13.0 | 13.0..13.0 | 1 | 2.974 | 2.951 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### permutation-shap / istella (rows full, shape X 100000x220; Xq 100x220; y 100000; yq 100)

race: done, driver rc 0, log `logs/algos.permutation-shap.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 19790.8 | 19790.8..19790.8 | 1 | - | - | - | 1843.2 | - | rel_error_vs_exact=5.339e-09 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 19638.3 | 19638.3..19638.3 | 1 | - | - | - | 1839.0 | - | rel_error_vs_exact=5.339e-09 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| shap-cpu | shap | cpu | opponent | 17306.3 | 17306.3..17306.3 | 1 | 1.144 | 1.135 | - | 1552.8 | - | rel_error_vs_exact=3.692e-10 | - | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.permutation-shap.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 61.4 | 61.4..61.4 | 1 | - | - | - | 143.3 | - | rel_error_vs_exact=2.15e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 62.9 | 62.9..62.9 | 1 | - | - | - | 147.9 | - | rel_error_vs_exact=2.15e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| shap-cpu | shap | cpu | opponent | 99.6 | 99.6..99.6 | 1 | 0.617 | 0.631 | - | 347.4 | - | rel_error_vs_exact=1.279e-15 | - | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, shap-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'n_background': 100, 'npermutations': 10}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | shap-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | shap (declared) |
| seed | 7 | 7 | 7 |

### qn-reg / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: failed, driver rc 3, log `logs/algos.qn-reg.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

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

race: failed, driver rc 3, log `logs/algos.qn-reg.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

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

### random-trees-embedding / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.random-trees-embedding.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 375.1 | 375.1..375.1 | 1 | - | - | - | 3164.2 | - | nonzeros_per_row=10.000000, output_columns=209 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 377.4 | 377.4..377.4 | 1 | - | - | - | 3155.5 | - | nonzeros_per_row=10.000000, output_columns=209 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 690.8 | 690.8..690.8 | 1 | 0.543 | 0.546 | - | 3384.3 | - | nonzeros_per_row=10.000000, output_columns=251 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 26.5 | 26.5..26.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 27.9 | 27.9..27.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 126.2 | 126.2..126.2 | 1 | 0.210 | 0.221 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### random-trees-embedding / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.random-trees-embedding.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 71.8 | 71.8..71.8 | 1 | - | - | - | 867.6 | - | nonzeros_per_row=10.000000, output_columns=292 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 74.5 | 74.5..74.5 | 1 | - | - | - | 862.3 | - | nonzeros_per_row=10.000000, output_columns=292 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 497.2 | 497.2..497.2 | 1 | 0.144 | 0.150 | - | 2442.3 | - | nonzeros_per_row=10.000000, output_columns=244 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 49.1 | 49.1..49.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 28.5 | 28.5..28.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 64.0 | 64.0..64.0 | 1 | 0.768 | 0.445 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### resnet-block / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.resnet-block.synthetic.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 745.1 | 745.1..745.1 | 1 | - | - | - | 1658.0 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 610.5 | 610.5..610.5 | 1 | - | - | - | 1659.6 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 39.5 | 39.5..39.5 | 1 | 18.866 | 15.457 | - | 1296.9 | 1082.7 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 42.2 | 42.2..42.2 | 1 | 17.670 | 14.477 | - | 1403.1 | 1036.7 | max_rel_diff_vs_torch_eager_fp32=0.380952, rel_fro_vs_torch_eager_fp32=1.952e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 45.3 | 45.3..45.3 | 1 | 16.454 | 13.481 | - | 1296.7 | 1088.7 | max_rel_diff_vs_torch_eager_fp32=18498.390913, rel_fro_vs_torch_eager_fp32=0.003425 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 51.4 | 51.4..51.4 | 1 | 14.492 | 11.873 | - | 1404.4 | 1036.7 | max_rel_diff_vs_torch_eager_fp32=18498.569727, rel_fro_vs_torch_eager_fp32=0.003425 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 242.5 | 242.5..242.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 232.9 | 232.9..232.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | 9.2 | 9.2..9.2 | 1 | 26.411 | 25.360 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 294.0 | 294.0..294.0 | 1 | 0.825 | 0.792 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 10.6 | 10.6..10.6 | 1 | 22.933 | 22.021 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 293.0 | 293.0..293.0 | 1 | 0.828 | 0.795 | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, ours-fast: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### select-d / synthetic (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc 0, log `logs/algos.select-d.synthetic.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 7.7 | 7.7..7.7 | 1 | - | - | - | 72.9 | - | d_agreement_vs_statsmodels=1.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 6.5 | 6.5..6.5 | 1 | - | - | - | 73.9 | - | d_agreement_vs_statsmodels=1.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsmodels-cpu | statsmodels | cpu | opponent | 6.2 | 6.2..6.2 | 1 | 1.244 | 1.051 | - | 162.0 | - | d_agreement_vs_statsmodels=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.select-d.taxi-hourly.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 7.7 | 7.7..7.7 | 1 | - | - | - | 67.7 | - | d_agreement_vs_statsmodels=1.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 7.6 | 7.6..7.6 | 1 | - | - | - | 70.8 | - | d_agreement_vs_statsmodels=1.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsmodels-cpu | statsmodels | cpu | opponent | 4.1 | 4.1..4.1 | 1 | 1.872 | 1.853 | - | 161.4 | - | d_agreement_vs_statsmodels=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, statsmodels-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'D': 0, 'pval_threshold': 0.05, 's': 0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | statsmodels-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | statsmodels (declared) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### stacking-clf / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.stacking-clf.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 6829.8 | 6829.8..6829.8 | 1 | - | - | - | 4189.8 | - | accuracy=0.929960, logloss=0.193706 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 6048.3 | 6048.3..6048.3 | 1 | - | - | - | 4201.7 | - | accuracy=0.929970, logloss=0.193693 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 173229.3 | 173229.3..173229.3 | 1 | 0.039 | 0.035 | - | 2983.1 | - | accuracy=0.930040, logloss=0.193931 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 76.8 | 76.8..76.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 77.6 | 77.6..77.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 231.1 | 231.1..231.1 | 1 | 0.332 | 0.336 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### stacking-clf / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.stacking-clf.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 2561.8 | 2561.8..2561.8 | 1 | - | - | - | 402.4 | - | accuracy=0.767920, logloss=0.536361 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1259.0 | 1259.0..1259.0 | 1 | - | - | - | 392.1 | - | accuracy=0.767920, logloss=0.536364 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 8635.0 | 8635.0..8635.0 | 1 | 0.297 | 0.146 | - | 256.1 | - | accuracy=0.755330, logloss=0.547756 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 32.3 | 32.3..32.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 33.4 | 33.4..33.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 31.8 | 31.8..31.8 | 1 | 1.016 | 1.050 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### stacking-reg / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.stacking-reg.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | UNKNOWN(no binding path) | REFUSED(not_ready: {"error": "NotImplementedError(\"mojolearn ElasticNet: random_state is refused. It selects nothing here -- the only randomness in cdFit is the coordinate shuffle, and selection='random' is refused by ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | UNKNOWN(no binding path) | REFUSED(not_ready: {"error": "NotImplementedError(\"mojolearn ElasticNet: random_state is refused. It selects nothing here -- the only randomness in cdFit is the coordinate shuffle, and selection='random' is refused by ) |
| sklearn-cpu | scikit-learn | cpu | opponent | 179536.9 | 179536.9..179536.9 | 1 | - | - | - | 3165.7 | - | finite=True, r2=0.447644, rmse=0.620826 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| sklearn-cpu | Xq | - | 14.4 | 14.4..14.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### stacking-reg / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.stacking-reg.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | UNKNOWN(no binding path) | REFUSED(not_ready: {"error": "NotImplementedError(\"mojolearn ElasticNet: random_state is refused. It selects nothing here -- the only randomness in cdFit is the coordinate shuffle, and selection='random' is refused by ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | UNKNOWN(no binding path) | REFUSED(not_ready: {"error": "NotImplementedError(\"mojolearn ElasticNet: random_state is refused. It selects nothing here -- the only randomness in cdFit is the coordinate shuffle, and selection='random' is refused by ) |
| sklearn-cpu | scikit-learn | cpu | opponent | 7552.9 | 7552.9..7552.9 | 1 | - | - | - | 273.7 | - | finite=True, r2=0.932532, rmse=4.137015 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| sklearn-cpu | Xq | - | 5.1 | 5.1..5.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### tree-shap / istella (rows full, shape X 100000x220; Xq 10000x220; y 100000; yq 10000)

race: done, driver rc 0, log `logs/algos.tree-shap.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 11775.4 | 11775.4..11775.4 | 1 | - | - | - | 1258.5 | - | max_additivity_error=1.099e-06 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 11477.7 | 11477.7..11477.7 | 1 | - | - | - | 1263.5 | - | max_additivity_error=1.099e-06 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| shap-cpu | shap | cpu | opponent | 628.0 | 628.0..628.0 | 1 | 18.751 | 18.277 | - | 1284.6 | - | max_additivity_error=1.862e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| xgboost-cpu | xgboost | cpu | opponent | 612.6 | 612.6..612.6 | 1 | 19.222 | 18.736 | - | 1231.2 | - | max_additivity_error=1.862e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| lightgbm-cpu | lightgbm | cpu | opponent | 428.3 | 428.3..428.3 | 1 | 27.492 | 26.797 | - | 1242.8 | - | max_additivity_error=4.441e-15 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.tree-shap.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 5028.9 | 5028.9..5028.9 | 1 | - | - | - | 144.3 | - | max_additivity_error=3.25e-05 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 4994.5 | 4994.5..4994.5 | 1 | - | - | - | 143.9 | - | max_additivity_error=3.25e-05 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| shap-cpu | shap | cpu | opponent | 359.6 | 359.6..359.6 | 1 | 13.984 | 13.888 | - | 270.5 | - | max_additivity_error=0.0001199 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| xgboost-cpu | xgboost | cpu | opponent | 355.7 | 355.7..355.7 | 1 | 14.138 | 14.041 | - | 202.6 | - | max_additivity_error=0.0001199 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| lightgbm-cpu | lightgbm | cpu | opponent | 201.8 | 201.8..201.8 | 1 | 24.922 | 24.751 | - | 207.9 | - | max_additivity_error=5.684e-13 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.tsne.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 16187.3 | 16187.3..16187.3 | 1 | - | - | - | 250.3 | - | trustworthiness_k15=0.991881 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 13640.0 | 13640.0..13640.0 | 1 | - | - | - | 286.3 | - | trustworthiness_k15=0.992164 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 28689.0 | 28689.0..28689.0 | 1 | 0.564 | 0.475 | - | 187.2 | - | trustworthiness_k15=0.992041 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/algos.tsne.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 10410.0 | 10410.0..10410.0 | 1 | - | - | - | 193.9 | - | trustworthiness_k15=0.998927 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 7940.1 | 7940.1..7940.1 | 1 | - | - | - | 225.8 | - | trustworthiness_k15=0.998912 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 24469.6 | 24469.6..24469.6 | 1 | 0.425 | 0.324 | - | 197.6 | - | trustworthiness_k15=0.998860 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

### voting-clf / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.voting-clf.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 7721.0 | 7721.0..7721.0 | 1 | - | - | - | 4047.1 | - | accuracy=0.918320, logloss=0.187682 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 5957.6 | 5957.6..5957.6 | 1 | - | - | - | 4436.9 | - | accuracy=0.918350, logloss=0.187755 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 65746.4 | 65746.4..65746.4 | 1 | 0.117 | 0.091 | - | 2776.3 | - | accuracy=0.918480, logloss=0.187541 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 76.3 | 76.3..76.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 80.5 | 80.5..80.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 282.7 | 282.7..282.7 | 1 | 0.270 | 0.285 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### voting-clf / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.voting-clf.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 680.1 | 680.1..680.1 | 1 | - | - | - | 344.0 | - | accuracy=0.742320, logloss=0.554799 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 488.8 | 488.8..488.8 | 1 | - | - | - | 348.4 | - | accuracy=0.742290, logloss=0.554802 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1936.4 | 1936.4..1936.4 | 1 | 0.351 | 0.252 | - | 230.6 | - | accuracy=0.742350, logloss=0.554650 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 15.1 | 15.1..15.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 15.3 | 15.3..15.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 33.1 | 33.1..33.1 | 1 | 0.456 | 0.460 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### voting-reg / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.voting-reg.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | UNKNOWN(no binding path) | REFUSED(not_ready: {"error": "NotImplementedError(\"mojolearn ElasticNet: random_state is refused. It selects nothing here -- the only randomness in cdFit is the coordinate shuffle, and selection='random' is refused by ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | UNKNOWN(no binding path) | REFUSED(not_ready: {"error": "NotImplementedError(\"mojolearn ElasticNet: random_state is refused. It selects nothing here -- the only randomness in cdFit is the coordinate shuffle, and selection='random' is refused by ) |
| sklearn-cpu | scikit-learn | cpu | opponent | 45059.8 | 45059.8..45059.8 | 1 | - | - | - | 8637.7 | - | finite=True, r2=0.406702, rmse=0.643423 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| sklearn-cpu | Xq | - | 20.1 | 20.1..20.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### voting-reg / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.voting-reg.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | UNKNOWN(no binding path) | REFUSED(not_ready: {"error": "NotImplementedError(\"mojolearn ElasticNet: random_state is refused. It selects nothing here -- the only randomness in cdFit is the coordinate shuffle, and selection='random' is refused by ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | UNKNOWN(no binding path) | REFUSED(not_ready: {"error": "NotImplementedError(\"mojolearn ElasticNet: random_state is refused. It selects nothing here -- the only randomness in cdFit is the coordinate shuffle, and selection='random' is refused by ) |
| sklearn-cpu | scikit-learn | cpu | opponent | 1519.8 | 1519.8..1519.8 | 1 | - | - | - | 217.3 | - | finite=True, r2=0.924635, rmse=4.372421 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| sklearn-cpu | Xq | - | 5.6 | 5.6..5.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

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

