# mojolearn benchmark board

Generated 2026-09-30T06:35:24Z from `board.json` (schema `mojolearn-bench-board/1`).

## Box

| field | value |
|---|---|
| vendor / API | amd / hip |
| GPU | AMD Instinct Mi325X VF |
| GPU driver | 6.12.12 |
| CPU | AMD EPYC 9575F 64-Core Processor (20 logical cores) |
| memory bytes | 168790056960 |
| OS | Ubuntu 24.04.2 LTS |
| Python | 3.12.3 CPython |
| mojolearn | 0.8.25 (wheel mojolearn-0.8.25-py3-none-manylinux_2_35_x86_64.whl, sha256 bafc6196927055d47f9113bf90ed378fdcf895ce215aceafa24e8787cd974641) |
| script commit | 74ce083bc773b6cbb29d15b49d774f3fa79554af |
| patch sync | - |
| modes | identical |
| rounds | 1 timed after 1 warm-up, arms interleaved round by round |
| seed | 7 |
| opponent versions | catboost 1.2.10, xgboost 3.2.0, lightgbm 4.7.0, scikit-learn 1.7.2, torch 2.13.0+rocm7.1, numba 0.67.0, statsmodels 0.15.0, faiss-cpu 1.15.1, numpy 2.5.3 |

## How to read this board

- Times are wall milliseconds of the public fit call (trees) or the lane's timed call (classical), median of the timed rounds; min..max beside it.
- `ours IDENTICAL / arm` is our IDENTICAL median divided by that opponent's median; `ours FAST / arm` likewise. Below 1.0 our median time is the lower one, above 1.0 the higher one. A ratio is shown only when both arms completed every round in this run, and only against an opponent: our two modes are never divided by each other here.
- Quality comes from the drivers: FSPEED-ACC for trees (held-out rows), one float64 NumPy function per lane for classical.
- Comparability: trees carry FSPEED-FIT-VERDICT (total leaves within 10% across arms is COMPARABLE); classical carry the clock span (SPAN-ASYMMETRIC names an arm whose clock excludes an upload or a fit that ours includes).
- Classical, wave 2 (`classical2`, tools/bench_board_more.py): the same worker protocol as classical; every lane's parameters, rows, timed span and each unavoidable mismatch with its reason are in the cells' `settings.lane_config`. Quality is one float64 NumPy function per lane over each arm's saved outputs.
- Neural: our IDENTICAL arm only (the neural surface builds no other tier, on any vendor) against torch at every fast setting it supports on this box, one arm each, the setting in the arm name: `torch-eager-fp32` (TF32 off), `torch-compile-fp32` (torch.compile, inductor), `torch-eager-tf32` / `torch-compile-tf32` (NVIDIA CUDA only), `torch-eager-bf16` / `torch-compile-bf16` (bf16 autocast mixed precision). TF32 and bf16 arms are ANOTHER PRECISION than ours; their quality columns show how far. The `*-infer` lanes are the CPU *Inference classes and race `torch-cpu-*` arms. An arm torch cannot run on this box is REFUSED by name in its cell. Every clock is host in, host out, synchronized. Every arm starts from the same parameters and reads the same inputs, so losses and outputs are comparable; `max_abs_diff_vs_ours` / `max_rel_diff_vs_ours` are the arm's output against ours. On NVIDIA the Mamba forward lanes add mamba_ssm's own fused kernels (the deployment path, our weights loaded): `mamba-ssm-fp32` (TF32 off, Triton fp32 dots IEEE) and `mamba-ssm-tf32` (TF32 on, Triton's default tf32 dots; ANOTHER PRECISION).
- `installed_wheel` confirms our binding loaded from site-packages, not the repo tree.
- Our CPU tier (`mojolearn CPU IDENTICAL`, arm `ours-cpu`): the same public estimator in a worker started under MOJOLEARN_VENDOR=cpu, the wheel's CPU switch (no GPU set loads; the host bindings answer, IDENTICAL only), read back as vendor cpu or refused by name. It races in the same rounds as every arm; `ours CPU / arm` is its median over each opponent's. `bits_equal_vs_ours_identical` compares its output with our GPU IDENTICAL arm's, bit for bit. Our CPU and GPU times are never divided by each other here.
- Memory: `peak host MB` and `peak GPU MB` are the highest per-round peaks over the timed rounds, read outside the clock; each arm's method is listed under its table (host: the resettable peak RSS on Linux, the peak physical footprint on macOS, which holds Metal buffers too; GPU: torch's own counter for torch arms, the driver's per-process figure for the rest, none on Apple).
- Inference: after a race's fit rounds each arm predicts with its own fitted model (no fit retimed), same rows, same output kind, one warm-up then the timed rounds interleaved. Trees: batch `test` (the held-out split) and `large` (1,000,000 training rows, capped at the training rows), host rows in and host predictions out on every arm; each arm's call is printed under its table. Classical: kmeans predict, pca transform, ols predict and svc predict on the eval rows, with the fit's clock span. Ratios are per batch, ours over each opponent.

## Coverage

Races: 90 planned, 90 done, 0 failed, 0 pending. Cells: 356 (MODE-MISMATCH 8, REFUSED 22, ok 326).

Inference cells: 182 (REFUSED 20, ok 162).

## Quality at a glance

Per lane and dataset: our FAST value, our IDENTICAL value, our CPU IDENTICAL value, and each opponent's.

| family | lane | dataset | metric | ours FAST | ours IDENTICAL | ours CPU | opponents |
|---|---|---|---|---|---|---|---|
| algos | adafactor | synthetic | rel_fro_vs_torch_eager_fp32 | - | 4.324e-05 | - | torch-eager-fp32 -; torch-compile-fp32 0.000000 |
| algos | adagrad | synthetic | rel_fro_vs_torch_eager_fp32 | - | 3.207e-09 | - | torch-eager-fp32 -; torch-compile-fp32 0.000000 |
| algos | adam | synthetic | rel_fro_vs_torch_eager_fp32 | - | 3.34e-08 | - | torch-eager-fp32 -; torch-compile-fp32 5.116e-08 |
| algos | adamax | synthetic | rel_fro_vs_torch_eager_fp32 | - | 4.144e-09 | - | torch-eager-fp32 -; torch-compile-fp32 6.17e-09 |
| algos | adamw | synthetic | rel_fro_vs_torch_eager_fp32 | - | 3.34e-08 | - | torch-eager-fp32 -; torch-compile-fp32 2.272e-07 |
| algos | avgpool1d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | avgpool1d | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | avgpool2d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | avgpool2d | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | batchnorm1d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | - | torch-eager-fp32 -; torch-compile-fp32 0.002793; torch-eager-bf16 0.000000; torch-compile-bf16 0.002793 |
| algos | batchnorm1d | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | - | torch-eager-fp32 -; torch-compile-fp32 5.248e-08; torch-eager-bf16 0.000000; torch-compile-bf16 5.248e-08 |
| algos | batchnorm2d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | - | torch-eager-fp32 -; torch-compile-fp32 0.001471; torch-eager-bf16 0.000000; torch-compile-bf16 0.001471 |
| algos | batchnorm2d | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | - | torch-eager-fp32 -; torch-compile-fp32 6.868e-08; torch-eager-bf16 0.000000; torch-compile-bf16 6.868e-08 |
| algos | cholesky | synthetic | relative_residual | - | 2.9e-07 | - | numpy-cpu 3.928e-08; torch-gpu 2.519e-07 |
| algos | clip-grad-norm | synthetic | norm | - | 1.000000 | - | torch-eager-fp32 1.000000; torch-compile-fp32 1.000000 |
| algos | clip-grad-norm | synthetic | norm_rel_diff_vs_ours | - | 0.000000 | - | torch-eager-fp32 0.000000; torch-compile-fp32 0.000000 |
| algos | cnn-clf | synthetic | accuracy (higher is better) | - | 1.000000 | - | torch-eager-fp32 1.000000; torch-compile-fp32 1.000000; torch-eager-bf16 1.000000; torch-compile-bf16 1.000000 |
| algos | conv1d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 3418.035805; torch-compile-bf16 3418.035805 |
| algos | conv1d | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 0.004377; torch-compile-bf16 0.004377 |
| algos | conv2d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 3418.486565; torch-compile-bf16 3418.486565 |
| algos | conv2d | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 0.003382; torch-compile-bf16 0.003382 |
| algos | cross-entropy | synthetic | loss_rel_err_vs_fp64 | - | 4.564e-08 | - | torch-eager-fp32 4.564e-08; torch-compile-fp32 4.564e-08 |
| algos | cross-entropy | synthetic | grad_max_rel_diff_vs_ours | - | - | - | torch-eager-fp32 4.191e-09; torch-compile-fp32 7.451e-09 |
| algos | eigh | synthetic | max_eigenvalue_error | - | - | - | numpy-cpu 3.49e-08; torch-gpu 1.044e-06 |
| algos | eigh | synthetic | relative_residual | - | - | - | numpy-cpu 2.824e-08; torch-gpu 9.614e-07 |
| algos | embedding | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000 |
| algos | embedding | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000 |
| algos | gcn | istella | max_rel_diff_vs_torch_eager_fp32 | - | - | - | torch-eager-fp32 -; torch-compile-fp32 0.007740; torch-eager-bf16 2866.979199; torch-compile-bf16 2866.972319 |
| algos | gcn | istella | rel_fro_vs_torch_eager_fp32 | - | - | - | torch-eager-fp32 -; torch-compile-fp32 9.044e-08; torch-eager-bf16 0.002187; torch-compile-bf16 0.002187 |
| algos | gcn | taxi | max_rel_diff_vs_torch_eager_fp32 | - | - | - | torch-eager-fp32 -; torch-compile-fp32 0.002794; torch-eager-bf16 1509.504160; torch-compile-bf16 1509.503229 |
| algos | gcn | taxi | rel_fro_vs_torch_eager_fp32 | - | - | - | torch-eager-fp32 -; torch-compile-fp32 7.648e-08; torch-eager-bf16 0.002330; torch-compile-bf16 0.002330 |
| algos | global-avgpool | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | - | torch-eager-fp32 -; torch-compile-fp32 0.007299; torch-eager-bf16 0.000000; torch-compile-bf16 0.007299 |
| algos | global-avgpool | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | - | torch-eager-fp32 -; torch-compile-fp32 1.01e-07; torch-eager-bf16 0.000000; torch-compile-bf16 1.01e-07 |
| algos | global-maxpool | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | global-maxpool | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | graphsage | istella | max_rel_diff_vs_torch_eager_fp32 | - | - | - | torch-eager-fp32 -; torch-compile-fp32 0.108108; torch-eager-bf16 3907.203674; torch-compile-bf16 3678.128123 |
| algos | graphsage | istella | rel_fro_vs_torch_eager_fp32 | - | - | - | torch-eager-fp32 -; torch-compile-fp32 8.065e-08; torch-eager-bf16 0.003366; torch-compile-bf16 0.003054 |
| algos | graphsage | taxi | max_rel_diff_vs_torch_eager_fp32 | - | - | - | torch-eager-fp32 -; torch-compile-fp32 0.074506; torch-eager-bf16 5040.230769; torch-compile-bf16 4607.677460 |
| algos | graphsage | taxi | rel_fro_vs_torch_eager_fp32 | - | - | - | torch-eager-fp32 -; torch-compile-fp32 4.566e-08; torch-eager-bf16 0.003557; torch-compile-bf16 0.003285 |
| algos | gru-clf | synthetic | accuracy (higher is better) | - | 0.971842 | - | torch-eager-fp32 0.971842; torch-compile-fp32 0.971842; torch-eager-bf16 0.971680; torch-compile-bf16 0.971680 |
| algos | gru-clf | synthetic | logloss (lower is better) | - | 0.065835 | - | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | gru-clf | taxi-hourly | accuracy (higher is better) | - | 0.865668 | - | torch-eager-fp32 0.865668; torch-compile-fp32 0.865668; torch-eager-bf16 0.865777; torch-compile-bf16 0.865777 |
| algos | gru-clf | taxi-hourly | logloss (lower is better) | - | 0.305841 | - | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | gru-reg | synthetic | r2 (higher is better) | - | 0.981946 | - | torch-eager-fp32 0.981946; torch-compile-fp32 0.981946; torch-eager-bf16 0.981981; torch-compile-bf16 0.981981 |
| algos | gru-reg | synthetic | rmse (lower is better) | - | 0.155672 | - | torch-eager-fp32 0.155672; torch-compile-fp32 0.155672; torch-eager-bf16 0.155521; torch-compile-bf16 0.155521 |
| algos | gru-reg | taxi-hourly | r2 (higher is better) | - | 0.748219 | - | torch-eager-fp32 0.748219; torch-compile-fp32 0.748219; torch-eager-bf16 0.748253; torch-compile-bf16 0.748253 |
| algos | gru-reg | taxi-hourly | rmse (lower is better) | - | 0.544182 | - | torch-eager-fp32 0.544182; torch-compile-fp32 0.544182; torch-eager-bf16 0.544145; torch-compile-bf16 0.544145 |
| algos | layernorm | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | - | torch-eager-fp32 -; torch-compile-fp32 0.007530; torch-eager-bf16 0.000000; torch-compile-bf16 0.007530 |
| algos | layernorm | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | - | torch-eager-fp32 -; torch-compile-fp32 5.326e-08; torch-eager-bf16 0.000000; torch-compile-bf16 5.326e-08 |
| algos | lr-constant | synthetic | max_rel_diff_vs_ours (0 is our output exactly) | - | 0.000000 | - | torch-cpu 1.038e-07 |
| algos | lr-exponential | synthetic | max_rel_diff_vs_ours (0 is our output exactly) | - | 0.000000 | - | torch-cpu 5.933e-08 |
| algos | lr-onecycle | synthetic | max_rel_diff_vs_ours (0 is our output exactly) | - | 0.000000 | - | torch-cpu 5.951e-08 |
| algos | lr-step | synthetic | max_rel_diff_vs_ours (0 is our output exactly) | - | 0.000000 | - | torch-cpu 1.49e-08 |
| algos | lr-warmup-cosine | synthetic | max_rel_diff_vs_ours (0 is our output exactly) | - | 0.000000 | - | torch-cpu 0.0001571 |
| algos | lr-warmup-linear | synthetic | max_rel_diff_vs_ours (0 is our output exactly) | - | 0.000000 | - | torch-cpu 0.001000 |
| algos | lstm-clf | synthetic | accuracy (higher is better) | - | 0.968696 | - | torch-eager-fp32 0.968696; torch-compile-fp32 0.968696; torch-eager-bf16 0.968913; torch-compile-bf16 0.968913 |
| algos | lstm-clf | synthetic | logloss (lower is better) | - | 0.072441 | - | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | lstm-clf | taxi-hourly | accuracy (higher is better) | - | 0.868218 | - | torch-eager-fp32 0.868218; torch-compile-fp32 0.868218; torch-eager-bf16 0.868056; torch-compile-bf16 0.868056 |
| algos | lstm-clf | taxi-hourly | logloss (lower is better) | - | 0.299901 | - | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | lstm-reg | synthetic | r2 (higher is better) | - | 0.981013 | - | torch-eager-fp32 0.981013; torch-compile-fp32 0.981013; torch-eager-bf16 0.981004; torch-compile-bf16 0.981004 |
| algos | lstm-reg | synthetic | rmse (lower is better) | - | 0.159641 | - | torch-eager-fp32 0.159642; torch-compile-fp32 0.159642; torch-eager-bf16 0.159679; torch-compile-bf16 0.159679 |
| algos | lstm-reg | taxi-hourly | r2 (higher is better) | - | 0.751679 | - | torch-eager-fp32 0.751679; torch-compile-fp32 0.751679; torch-eager-bf16 0.751591; torch-compile-bf16 0.751591 |
| algos | lstm-reg | taxi-hourly | rmse (lower is better) | - | 0.540429 | - | torch-eager-fp32 0.540429; torch-compile-fp32 0.540429; torch-eager-bf16 0.540526; torch-compile-bf16 0.540526 |
| algos | lstsq | istella | relative_residual | - | 0.849957 | - | numpy-cpu 0.876106; torch-gpu nan |
| algos | lstsq | taxi | relative_residual | - | 0.756366 | - | numpy-cpu 0.756366; torch-gpu 0.756366 |
| algos | lu-factor | synthetic | relative_residual | - | 3.249e-06 | - | scipy-cpu 3.439e-07; torch-gpu 4.003e-07 |
| algos | lu-solve | synthetic | relative_residual | - | 3.249e-06 | - | numpy-cpu 3.259e-08; torch-gpu 4.003e-07 |
| algos | maxpool1d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | maxpool1d | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | maxpool2d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | maxpool2d | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | moe | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | - | torch-eager-fp32 -; torch-compile-fp32 0.017192; torch-eager-bf16 22778.779951; torch-compile-bf16 22796.112469 |
| algos | moe | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | - | torch-eager-fp32 -; torch-compile-fp32 1.088e-07; torch-eager-bf16 0.055222; torch-compile-bf16 0.055199 |
| algos | nadam | synthetic | rel_fro_vs_torch_eager_fp32 | - | 2.945e-08 | - | torch-eager-fp32 -; torch-compile-fp32 1.464e-07 |
| algos | qr | istella | relative_gram_difference | - | 0.0009027 | - | numpy-cpu 2.462e-08; torch-gpu 0.0001695 |
| algos | qr | taxi | relative_gram_difference | - | 0.001996 | - | numpy-cpu 3.024e-08; torch-gpu 9.54e-07 |
| algos | randomized-svd | istella | relative_reconstruction_error (lower is better) | - | 0.0002359 | - | sklearn-cpu 0.0002359; torch-gpu 0.0002359 |
| algos | randomized-svd | taxi | relative_reconstruction_error (lower is better) | - | 0.027197 | - | sklearn-cpu 0.027197; torch-gpu 0.027197 |
| algos | resnet-block | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | - | torch-eager-fp32 -; torch-compile-fp32 0.357628; torch-eager-bf16 24577.140808; torch-compile-bf16 18499.135971 |
| algos | resnet-block | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | - | torch-eager-fp32 -; torch-compile-fp32 1.809e-07; torch-eager-bf16 0.004174; torch-compile-bf16 0.003425 |
| algos | rmsprop | synthetic | rel_fro_vs_torch_eager_fp32 | - | 3.829e-08 | - | torch-eager-fp32 -; torch-compile-fp32 0.000000 |
| algos | rnn-clf | synthetic | accuracy (higher is better) | - | 0.953559 | - | torch-eager-fp32 0.953559; torch-compile-fp32 0.953559; torch-eager-bf16 0.953396; torch-compile-bf16 0.953396 |
| algos | rnn-clf | synthetic | logloss (lower is better) | - | 0.103698 | - | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | rnn-clf | taxi-hourly | accuracy (higher is better) | - | 0.868056 | - | torch-eager-fp32 0.868056; torch-compile-fp32 0.868056; torch-eager-bf16 0.868056; torch-compile-bf16 0.868056 |
| algos | rnn-clf | taxi-hourly | logloss (lower is better) | - | 0.304864 | - | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | rnn-reg | synthetic | r2 (higher is better) | - | 0.977348 | - | torch-eager-fp32 0.977348; torch-compile-fp32 0.977348; torch-eager-bf16 0.977383; torch-compile-bf16 0.977383 |
| algos | rnn-reg | synthetic | rmse (lower is better) | - | 0.174374 | - | torch-eager-fp32 0.174374; torch-compile-fp32 0.174374; torch-eager-bf16 0.174238; torch-compile-bf16 0.174238 |
| algos | rnn-reg | taxi-hourly | r2 (higher is better) | - | 0.738796 | - | torch-eager-fp32 0.738796; torch-compile-fp32 0.738796; torch-eager-bf16 0.738886; torch-compile-bf16 0.738886 |
| algos | rnn-reg | taxi-hourly | rmse (lower is better) | - | 0.554271 | - | torch-eager-fp32 0.554271; torch-compile-fp32 0.554271; torch-eager-bf16 0.554176; torch-compile-bf16 0.554176 |
| algos | sgd | synthetic | rel_fro_vs_torch_eager_fp32 | - | 0.000000 | - | torch-eager-fp32 -; torch-compile-fp32 6.772e-10 |
| algos | svd | istella | max_rel_singular_value_error | - | 73.628098 | - | numpy-cpu 1.000000; torch-gpu 2.707e+08 |
| algos | svd | istella | relative_reconstruction_error_100k_rows | - | 0.0005441 | - | numpy-cpu 4.1e-08; torch-gpu 0.026534 |
| algos | svd | taxi | max_rel_singular_value_error | - | 6.332e-07 | - | numpy-cpu 4.308e-08; torch-gpu 4.409e-05 |
| algos | svd | taxi | relative_reconstruction_error_100k_rows | - | 0.000333 | - | numpy-cpu 4.314e-08; torch-gpu 0.003043 |
| algos | svgp | istella | r2 (higher is better) | - | -0.106016 | - | gpytorch-gpu -0.106040; gpytorch-cpu -0.106040 |
| algos | svgp | istella | rmse (lower is better) | - | 0.878373 | - | gpytorch-gpu 0.878383; gpytorch-cpu 0.878383 |
| algos | svgp | taxi | r2 (higher is better) | - | - | - | gpytorch-gpu -0.209526; gpytorch-cpu -0.209532 |
| algos | svgp | taxi | rmse (lower is better) | - | - | - | gpytorch-gpu 17.831014; gpytorch-cpu 17.831057 |
| classical | kmeans | istella | inertia (lower is better) | - | 6.051e+17 | - | torch-gpu 5.991e+17; sklearn-cpu 6.049e+17 |
| classical | kmeans | istella | inertia_over_ours | - | 1.000000 | - | torch-gpu 0.990156; sklearn-cpu 0.999759 |
| classical | kmeans | istella | n_iter | - | 33 | - | torch-gpu 57; sklearn-cpu 24 |
| classical | kmeans | taxi | inertia (lower is better) | - | 3.093e+08 | - | torch-gpu 3.129e+08; sklearn-cpu 3.093e+08 |
| classical | kmeans | taxi | inertia_over_ours | - | 1.000000 | - | torch-gpu 1.011535; sklearn-cpu 0.999937 |
| classical | kmeans | taxi | n_iter | - | 91 | - | torch-gpu 80; sklearn-cpu 58 |
| classical | knn | istella | recall_at_k (higher is better) | - | 0.976250 | - | torch-gpu 0.979840; sklearn-cpu 1.000000 |
| classical | knn | istella | rows_with_repeated_ids | - | 0 | - | torch-gpu 0; sklearn-cpu 0 |
| classical | knn | taxi | recall_at_k (higher is better) | - | 0.999754 | - | torch-gpu 0.999738; sklearn-cpu 1.000000 |
| classical | knn | taxi | rows_with_repeated_ids | - | 0 | - | torch-gpu 0; sklearn-cpu 0 |
| classical | ols | istella | r2 (higher is better) | - | 0.331944 | - | torch-gpu nan; torch-gpu-eigh 0.151604; sklearn-cpu 0.001881 |
| classical | ols | istella | rmse (lower is better) | - | 0.682027 | - | torch-gpu nan; torch-gpu-eigh 0.768590; sklearn-cpu 0.833655 |
| classical | ols | taxi | r2 (higher is better) | - | 0.908837 | - | torch-gpu 0.908837; torch-gpu-eigh 0.908827; sklearn-cpu 0.724850 |
| classical | ols | taxi | rmse (lower is better) | - | 4.696466 | - | torch-gpu 4.696472; torch-gpu-eigh 4.696715; sklearn-cpu 8.159187 |
| classical | pca | istella | explained_variance_ratio_sum (higher is better) | - | 1.000000 | - | torch-gpu 1.000000; sklearn-cpu 1.000000 |
| classical | pca | taxi | explained_variance_ratio_sum (higher is better) | - | 0.999996 | - | torch-gpu 0.999996; sklearn-cpu 0.999995 |
| neural | gemm-bf16 | gaussian | max_rel_err_vs_fp64 (lower is better) | - | 1.155e-07 | - | torch-eager-bf16 0.002759; torch-compile-bf16 0.002759 |
| neural | gemm-bf16 | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-bf16 0.999023; torch-compile-bf16 0.999023 |
| neural | gemm-bf16 | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-bf16 0.002759; torch-compile-bf16 0.002759 |
| neural | gemm-int8 | gaussian | max_rel_err_vs_fp64 (lower is better) | - | 0.000000 | - | - |
| neural | gemm | gaussian | max_rel_err_vs_fp64 (lower is better) | - | 2.399e-07 | - | torch-eager-fp32 2.802e-06; torch-compile-fp32 2.802e-06; torch-eager-bf16 0.003752; torch-compile-bf16 0.003752 |
| neural | gemm | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 0.001038; torch-compile-fp32 0.001038; torch-eager-bf16 1.358337; torch-compile-bf16 1.358337 |
| neural | gemm | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 2.866e-06; torch-compile-fp32 2.866e-06; torch-eager-bf16 0.003751; torch-compile-bf16 0.003751 |
| neural | lm-forward | bytes | mean_nll (lower is better) | - | 9.018733 | - | torch-eager-fp32 9.018733; torch-compile-fp32 9.018733; torch-eager-bf16 9.018647; torch-compile-bf16 9.018653 |
| neural | lm-forward | bytes | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 1.222e-06; torch-compile-fp32 1.192e-06; torch-eager-bf16 0.006313; torch-compile-bf16 0.006870 |
| neural | lm-forward | bytes | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 1.677e-06; torch-compile-fp32 1.637e-06; torch-eager-bf16 0.008666; torch-compile-bf16 0.009432 |
| neural | lm-host-train-step | bytes | loss_first_step (same init and batches on every arm) | - | 9.017858 | - | torch-cpu-eager-fp32 9.017857; torch-cpu-compile-fp32 9.017857; torch-cpu-eager-bf16 9.017747; torch-cpu-compile-bf16 9.017767 |
| neural | lm-host-train-step | bytes | loss_last_step (same init and batches on every arm) | - | 8.367768 | - | torch-cpu-eager-fp32 8.367767; torch-cpu-compile-fp32 8.367766; torch-cpu-eager-bf16 8.368348; torch-cpu-compile-bf16 8.367756 |
| neural | lm-host-train-step | bytes | steps | - | 2 | - | torch-cpu-eager-fp32 2; torch-cpu-compile-fp32 2; torch-cpu-eager-bf16 2; torch-cpu-compile-bf16 2 |
| neural | lm-host-train-step | bytes | loss_first_abs_diff_vs_ours (0 is our value exactly) | - | - | - | torch-cpu-eager-fp32 9.537e-07; torch-cpu-compile-fp32 9.537e-07; torch-cpu-eager-bf16 0.0001106; torch-cpu-compile-bf16 9.06e-05 |
| neural | lm-host-train-step | bytes | loss_last_abs_diff_vs_ours (0 is our value exactly) | - | - | - | torch-cpu-eager-fp32 9.537e-07; torch-cpu-compile-fp32 1.907e-06; torch-cpu-eager-bf16 0.0005798; torch-cpu-compile-bf16 1.24e-05 |
| neural | lm-infer | bytes | mean_nll (lower is better) | - | 9.017857 | - | torch-cpu-eager-fp32 9.017857; torch-cpu-compile-fp32 9.017857; torch-cpu-eager-bf16 9.017748; torch-cpu-compile-bf16 9.017761 |
| neural | lm-infer | bytes | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-eager-fp32 7.227e-07; torch-cpu-compile-fp32 6.706e-07; torch-cpu-eager-bf16 0.006180; torch-cpu-compile-bf16 0.006585 |
| neural | lm-infer | bytes | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-eager-fp32 9.921e-07; torch-cpu-compile-fp32 9.205e-07; torch-cpu-eager-bf16 0.008484; torch-cpu-compile-bf16 0.009040 |
| neural | lm-train-step | bytes | loss_first_step (same init and batches on every arm) | - | 9.018733 | - | torch-eager-fp32 9.018732; torch-compile-fp32 9.018734; torch-eager-bf16 9.018677; torch-compile-bf16 9.018653 |
| neural | lm-train-step | bytes | loss_last_step (same init and batches on every arm) | - | 8.422411 | - | torch-eager-fp32 8.422411; torch-compile-fp32 8.422412; torch-eager-bf16 8.420410; torch-compile-bf16 8.419378 |
| neural | lm-train-step | bytes | steps | - | 2 | - | torch-eager-fp32 2; torch-compile-fp32 2; torch-eager-bf16 2; torch-compile-bf16 2 |
| neural | lm-train-step | bytes | loss_first_abs_diff_vs_ours (0 is our value exactly) | - | - | - | torch-eager-fp32 9.537e-07; torch-compile-fp32 9.537e-07; torch-eager-bf16 5.627e-05; torch-compile-bf16 8.011e-05 |
| neural | lm-train-step | bytes | loss_last_abs_diff_vs_ours (0 is our value exactly) | - | - | - | torch-eager-fp32 0.000000; torch-compile-fp32 9.537e-07; torch-eager-bf16 0.002001; torch-compile-bf16 0.003033 |
| neural | mamba1-forward | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 2.384e-07; torch-eager-bf16 5.436e-05 |
| neural | mamba1-forward | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 1.189e-07; torch-eager-bf16 2.711e-05 |
| neural | mamba1-infer | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-eager-fp32 1.192e-07; torch-cpu-eager-bf16 5.15e-05 |
| neural | mamba1-infer | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-eager-fp32 5.948e-08; torch-cpu-eager-bf16 2.57e-05 |
| neural | mamba2-forward | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 1.073e-06; torch-compile-fp32 1.073e-06; torch-eager-bf16 0.007144; torch-compile-bf16 0.007144 |
| neural | mamba2-forward | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 3.648e-07; torch-compile-fp32 3.648e-07; torch-eager-bf16 0.002429; torch-compile-bf16 0.002429 |
| neural | mamba2-infer | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-eager-fp32 7.749e-07; torch-cpu-compile-fp32 7.749e-07; torch-cpu-eager-bf16 0.005920; torch-cpu-compile-bf16 0.005920 |
| neural | mamba2-infer | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-eager-fp32 2.751e-07; torch-cpu-compile-fp32 2.751e-07; torch-cpu-eager-bf16 0.002102; torch-cpu-compile-bf16 0.002102 |
| neural | mamba3-forward | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 3.874e-07; torch-compile-fp32 4.768e-07; torch-eager-bf16 0.002069; torch-compile-bf16 0.001982 |
| neural | mamba3-forward | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 1.695e-07; torch-compile-fp32 2.086e-07; torch-eager-bf16 0.0009051; torch-compile-bf16 0.0008672 |
| neural | mamba3-infer | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-eager-fp32 2.384e-07; torch-cpu-compile-fp32 2.98e-07; torch-cpu-eager-bf16 0.002148; torch-cpu-compile-bf16 0.002040 |
| neural | mamba3-infer | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-eager-fp32 1.06e-07; torch-cpu-compile-fp32 1.324e-07; torch-cpu-eager-bf16 0.0009547; torch-cpu-compile-bf16 0.0009065 |
| neural | mlp-infer | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-eager-fp32 1.192e-07; torch-cpu-compile-fp32 1.192e-07; torch-cpu-eager-bf16 0.004285; torch-cpu-compile-bf16 0.004285 |
| neural | mlp-infer | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-eager-fp32 1.1e-07; torch-cpu-compile-fp32 1.1e-07; torch-cpu-eager-bf16 0.003956; torch-cpu-compile-bf16 0.003956 |
| neural | mlp-train-step | gaussian | loss_first_step (same init and batches on every arm) | - | 1.160401 | - | torch-eager-fp32 1.160401; torch-compile-fp32 1.160401; torch-eager-bf16 1.160498; torch-compile-bf16 1.160498 |
| neural | mlp-train-step | gaussian | loss_last_step (same init and batches on every arm) | - | 1.123361 | - | torch-eager-fp32 1.123361; torch-compile-fp32 1.123361; torch-eager-bf16 1.123461; torch-compile-bf16 1.123461 |
| neural | mlp-train-step | gaussian | steps | - | 2 | - | torch-eager-fp32 2; torch-compile-fp32 2; torch-eager-bf16 2; torch-compile-bf16 2 |
| neural | mlp-train-step | gaussian | loss_first_abs_diff_vs_ours (0 is our value exactly) | - | - | - | torch-eager-fp32 2.384e-07; torch-compile-fp32 2.384e-07; torch-eager-bf16 9.656e-05; torch-compile-bf16 9.656e-05 |
| neural | mlp-train-step | gaussian | loss_last_abs_diff_vs_ours (0 is our value exactly) | - | - | - | torch-eager-fp32 2.384e-07; torch-compile-fp32 2.384e-07; torch-eager-bf16 0.0001005; torch-compile-bf16 0.0001006 |
| neural | samba-forward | bytes | mean_nll (lower is better) | - | 5.635910 | - | torch-eager-fp32 5.635910; torch-compile-fp32 5.635910; torch-eager-bf16 5.635950; torch-compile-bf16 5.635933 |
| neural | samba-forward | bytes | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 2.146e-06; torch-compile-fp32 2.168e-06; torch-eager-bf16 0.017180; torch-compile-bf16 0.016678 |
| neural | samba-forward | bytes | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 1.206e-06; torch-compile-fp32 1.219e-06; torch-eager-bf16 0.009659; torch-compile-bf16 0.009377 |
| neural | samba-infer | bytes | mean_nll (lower is better) | - | 5.635910 | - | torch-cpu-eager-fp32 5.635910; torch-cpu-compile-fp32 5.635910; torch-cpu-eager-bf16 5.635976; torch-cpu-compile-bf16 5.635914 |
| neural | samba-infer | bytes | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-eager-fp32 1.997e-06; torch-cpu-compile-fp32 1.848e-06; torch-cpu-eager-bf16 0.018279; torch-cpu-compile-bf16 0.016166 |
| neural | samba-infer | bytes | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-eager-fp32 1.123e-06; torch-cpu-compile-fp32 1.039e-06; torch-cpu-eager-bf16 0.010276; torch-cpu-compile-bf16 0.009088 |
| neural | samba-train-step | bytes | loss_first_step (same init and batches on every arm) | - | 5.635910 | - | torch-eager-fp32 5.635910; torch-compile-fp32 5.635909; torch-eager-bf16 5.635949; torch-compile-bf16 - |
| neural | samba-train-step | bytes | loss_last_step (same init and batches on every arm) | - | 4.833934 | - | torch-eager-fp32 4.833934; torch-compile-fp32 4.833934; torch-eager-bf16 4.834015; torch-compile-bf16 - |
| neural | samba-train-step | bytes | steps | - | 2 | - | torch-eager-fp32 2; torch-compile-fp32 2; torch-eager-bf16 2; torch-compile-bf16 - |
| neural | samba-train-step | bytes | loss_first_abs_diff_vs_ours (0 is our value exactly) | - | - | - | torch-eager-fp32 4.768e-07; torch-compile-fp32 9.537e-07; torch-eager-bf16 3.91e-05; torch-compile-bf16 - |
| neural | samba-train-step | bytes | loss_last_abs_diff_vs_ours (0 is our value exactly) | - | - | - | torch-eager-fp32 4.768e-07; torch-compile-fp32 4.768e-07; torch-eager-bf16 8.106e-05; torch-compile-bf16 - |
| neural | transformer-forward | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 4.768e-07; torch-compile-fp32 4.768e-07; torch-eager-bf16 0.001819; torch-compile-bf16 0.001819 |
| neural | transformer-forward | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 1.02e-07; torch-compile-fp32 1.02e-07; torch-eager-bf16 0.0003893; torch-compile-bf16 0.0003893 |
| neural | transformer-infer | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-eager-fp32 4.768e-07; torch-cpu-compile-fp32 9.537e-07; torch-cpu-eager-bf16 0.001941; torch-cpu-compile-bf16 0.001819 |
| neural | transformer-infer | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-eager-fp32 1.076e-07; torch-cpu-compile-fp32 2.153e-07; torch-cpu-eager-bf16 0.0004383; torch-cpu-compile-bf16 0.0004107 |

## Inference at a glance

Batch prediction, each arm with its own fitted model from the same race; medians in ms. `FAST = IDENTICAL bits` compares our two tiers' predictions on the same rows; `CPU = IDENTICAL bits` compares our CPU tier's with our GPU IDENTICAL arm's.

| family | lane | dataset | batch | rows | ours FAST ms | ours IDENTICAL ms | FAST = IDENTICAL bits | ours CPU ms | CPU = IDENTICAL bits | opponents |
|---|---|---|---|---|---|---|---|---|---|---|
| algos | avgpool1d | synthetic | Xq | - | - | - | - | - | - | torch-eager-fp32 0.1 ms (IDENTICAL/arm -); torch-compile-fp32 0.1 ms (IDENTICAL/arm -); torch-eager-bf16 0.1 ms (IDENTICAL/arm -); torch-compile-bf16 0.1 ms (IDENTICAL/arm -) |
| algos | avgpool2d | synthetic | Xq | - | - | - | - | - | - | torch-eager-fp32 0.1 ms (IDENTICAL/arm -); torch-compile-fp32 0.2 ms (IDENTICAL/arm -); torch-eager-bf16 0.1 ms (IDENTICAL/arm -); torch-compile-bf16 0.2 ms (IDENTICAL/arm -) |
| algos | batchnorm1d | synthetic | Xq | - | - | - | - | - | - | torch-eager-fp32 0.1 ms (IDENTICAL/arm -); torch-compile-fp32 0.2 ms (IDENTICAL/arm -); torch-eager-bf16 0.1 ms (IDENTICAL/arm -); torch-compile-bf16 0.2 ms (IDENTICAL/arm -) |
| algos | batchnorm2d | synthetic | Xq | - | - | - | - | - | - | torch-eager-fp32 0.2 ms (IDENTICAL/arm -); torch-compile-fp32 0.2 ms (IDENTICAL/arm -); torch-eager-bf16 0.2 ms (IDENTICAL/arm -); torch-compile-bf16 0.2 ms (IDENTICAL/arm -) |
| algos | cnn-clf | synthetic | Xq | - | - | 6.4 | - | - | - | torch-eager-fp32 0.9 ms (IDENTICAL/arm 6.845); torch-compile-fp32 0.8 ms (IDENTICAL/arm 7.585); torch-eager-bf16 1.0 ms (IDENTICAL/arm 6.432); torch-compile-bf16 0.8 ms (IDENTICAL/arm 7.955) |
| algos | conv1d | synthetic | Xq | - | - | - | - | - | - | torch-eager-fp32 0.7 ms (IDENTICAL/arm -); torch-compile-fp32 0.7 ms (IDENTICAL/arm -); torch-eager-bf16 0.4 ms (IDENTICAL/arm -); torch-compile-bf16 0.4 ms (IDENTICAL/arm -) |
| algos | conv2d | synthetic | Xq | - | - | - | - | - | - | torch-eager-fp32 0.3 ms (IDENTICAL/arm -); torch-compile-fp32 0.3 ms (IDENTICAL/arm -); torch-eager-bf16 0.3 ms (IDENTICAL/arm -); torch-compile-bf16 0.3 ms (IDENTICAL/arm -) |
| algos | dropout2d | synthetic | Xq | - | - | - | - | - | - | torch-eager-fp32 0.1 ms (IDENTICAL/arm -); torch-compile-fp32 0.2 ms (IDENTICAL/arm -) |
| algos | embedding | synthetic | Xq | - | - | - | - | - | - | torch-eager-fp32 0.1 ms (IDENTICAL/arm -); torch-compile-fp32 0.2 ms (IDENTICAL/arm -) |
| algos | gcn | istella | Xq | - | - | - | - | - | - | torch-eager-fp32 2.6 ms (IDENTICAL/arm -); torch-compile-fp32 1.9 ms (IDENTICAL/arm -); torch-eager-bf16 2.4 ms (IDENTICAL/arm -); torch-compile-bf16 1.8 ms (IDENTICAL/arm -) |
| algos | gcn | taxi | Xq | - | - | - | - | - | - | torch-eager-fp32 2.0 ms (IDENTICAL/arm -); torch-compile-fp32 1.6 ms (IDENTICAL/arm -); torch-eager-bf16 2.1 ms (IDENTICAL/arm -); torch-compile-bf16 1.6 ms (IDENTICAL/arm -) |
| algos | global-avgpool | synthetic | Xq | - | - | - | - | - | - | torch-eager-fp32 0.0 ms (IDENTICAL/arm -); torch-compile-fp32 0.1 ms (IDENTICAL/arm -); torch-eager-bf16 0.1 ms (IDENTICAL/arm -); torch-compile-bf16 0.1 ms (IDENTICAL/arm -) |
| algos | global-maxpool | synthetic | Xq | - | - | - | - | - | - | torch-eager-fp32 0.1 ms (IDENTICAL/arm -); torch-compile-fp32 0.1 ms (IDENTICAL/arm -); torch-eager-bf16 0.1 ms (IDENTICAL/arm -); torch-compile-bf16 0.1 ms (IDENTICAL/arm -) |
| algos | graphsage | istella | Xq | - | - | - | - | - | - | torch-eager-fp32 2.6 ms (IDENTICAL/arm -); torch-compile-fp32 2.6 ms (IDENTICAL/arm -); torch-eager-bf16 2.6 ms (IDENTICAL/arm -); torch-compile-bf16 2.6 ms (IDENTICAL/arm -) |
| algos | graphsage | taxi | Xq | - | - | - | - | - | - | torch-eager-fp32 0.4 ms (IDENTICAL/arm -); torch-compile-fp32 0.4 ms (IDENTICAL/arm -); torch-eager-bf16 0.4 ms (IDENTICAL/arm -); torch-compile-bf16 0.4 ms (IDENTICAL/arm -) |
| algos | gru-clf | synthetic | Xq | - | - | 12.9 | - | - | - | torch-eager-fp32 6.2 ms (IDENTICAL/arm 2.085); torch-compile-fp32 6.2 ms (IDENTICAL/arm 2.094); torch-eager-bf16 6.1 ms (IDENTICAL/arm 2.108); torch-compile-bf16 6.2 ms (IDENTICAL/arm 2.095) |
| algos | gru-clf | taxi-hourly | Xq | - | - | 12.9 | - | - | - | torch-eager-fp32 6.0 ms (IDENTICAL/arm 2.145); torch-compile-fp32 6.2 ms (IDENTICAL/arm 2.084); torch-eager-bf16 6.1 ms (IDENTICAL/arm 2.123); torch-compile-bf16 6.3 ms (IDENTICAL/arm 2.040) |
| algos | gru-reg | synthetic | Xq | - | - | 6.5 | - | - | - | torch-eager-fp32 6.1 ms (IDENTICAL/arm 1.058); torch-compile-fp32 6.2 ms (IDENTICAL/arm 1.046); torch-eager-bf16 6.1 ms (IDENTICAL/arm 1.066); torch-compile-bf16 6.0 ms (IDENTICAL/arm 1.071) |
| algos | gru-reg | taxi-hourly | Xq | - | - | 6.6 | - | - | - | torch-eager-fp32 6.2 ms (IDENTICAL/arm 1.066); torch-compile-fp32 6.0 ms (IDENTICAL/arm 1.086); torch-eager-bf16 6.1 ms (IDENTICAL/arm 1.079); torch-compile-bf16 6.1 ms (IDENTICAL/arm 1.082) |
| algos | layernorm | synthetic | Xq | - | - | - | - | - | - | torch-eager-fp32 0.1 ms (IDENTICAL/arm -); torch-compile-fp32 0.1 ms (IDENTICAL/arm -); torch-eager-bf16 0.1 ms (IDENTICAL/arm -); torch-compile-bf16 0.1 ms (IDENTICAL/arm -) |
| algos | lstm-clf | synthetic | Xq | - | - | 16.2 | - | - | - | torch-eager-fp32 3.3 ms (IDENTICAL/arm 4.946); torch-compile-fp32 3.2 ms (IDENTICAL/arm 5.090); torch-eager-bf16 3.0 ms (IDENTICAL/arm 5.439); torch-compile-bf16 3.1 ms (IDENTICAL/arm 5.182) |
| algos | lstm-clf | taxi-hourly | Xq | - | - | 16.0 | - | - | - | torch-eager-fp32 3.3 ms (IDENTICAL/arm 4.889); torch-compile-fp32 3.2 ms (IDENTICAL/arm 4.942); torch-eager-bf16 3.0 ms (IDENTICAL/arm 5.340); torch-compile-bf16 3.0 ms (IDENTICAL/arm 5.309) |
| algos | lstm-reg | synthetic | Xq | - | - | 8.2 | - | - | - | torch-eager-fp32 3.1 ms (IDENTICAL/arm 2.687); torch-compile-fp32 3.1 ms (IDENTICAL/arm 2.677); torch-eager-bf16 3.0 ms (IDENTICAL/arm 2.710); torch-compile-bf16 3.1 ms (IDENTICAL/arm 2.626) |
| algos | lstm-reg | taxi-hourly | Xq | - | - | 8.4 | - | - | - | torch-eager-fp32 3.2 ms (IDENTICAL/arm 2.588); torch-compile-fp32 3.2 ms (IDENTICAL/arm 2.586); torch-eager-bf16 3.1 ms (IDENTICAL/arm 2.734); torch-compile-bf16 3.1 ms (IDENTICAL/arm 2.674) |
| algos | maxpool1d | synthetic | Xq | - | - | - | - | - | - | torch-eager-fp32 0.1 ms (IDENTICAL/arm -); torch-compile-fp32 0.1 ms (IDENTICAL/arm -); torch-eager-bf16 0.1 ms (IDENTICAL/arm -); torch-compile-bf16 0.1 ms (IDENTICAL/arm -) |
| algos | maxpool2d | synthetic | Xq | - | - | - | - | - | - | torch-eager-fp32 0.1 ms (IDENTICAL/arm -); torch-compile-fp32 0.1 ms (IDENTICAL/arm -); torch-eager-bf16 0.1 ms (IDENTICAL/arm -); torch-compile-bf16 0.2 ms (IDENTICAL/arm -) |
| algos | moe | synthetic | Xq | - | - | - | - | - | - | torch-eager-fp32 3.9 ms (IDENTICAL/arm -); torch-compile-fp32 3.9 ms (IDENTICAL/arm -); torch-eager-bf16 2.1 ms (IDENTICAL/arm -); torch-compile-bf16 2.1 ms (IDENTICAL/arm -) |
| algos | resnet-block | synthetic | Xq | - | - | - | - | - | - | torch-eager-fp32 0.8 ms (IDENTICAL/arm -); torch-compile-fp32 0.6 ms (IDENTICAL/arm -); torch-eager-bf16 0.7 ms (IDENTICAL/arm -); torch-compile-bf16 0.5 ms (IDENTICAL/arm -) |
| algos | rnn-clf | synthetic | Xq | - | - | 7.1 | - | - | - | torch-eager-fp32 2.5 ms (IDENTICAL/arm 2.863); torch-compile-fp32 2.6 ms (IDENTICAL/arm 2.781); torch-eager-bf16 2.9 ms (IDENTICAL/arm 2.442); torch-compile-bf16 2.6 ms (IDENTICAL/arm 2.697) |
| algos | rnn-clf | taxi-hourly | Xq | - | - | 7.2 | - | - | - | torch-eager-fp32 2.6 ms (IDENTICAL/arm 2.801); torch-compile-fp32 2.3 ms (IDENTICAL/arm 3.122); torch-eager-bf16 2.9 ms (IDENTICAL/arm 2.476); torch-compile-bf16 3.1 ms (IDENTICAL/arm 2.291) |
| algos | rnn-reg | synthetic | Xq | - | - | 3.7 | - | - | - | torch-eager-fp32 2.7 ms (IDENTICAL/arm 1.371); torch-compile-fp32 2.7 ms (IDENTICAL/arm 1.355); torch-eager-bf16 3.0 ms (IDENTICAL/arm 1.239); torch-compile-bf16 3.1 ms (IDENTICAL/arm 1.191) |
| algos | rnn-reg | taxi-hourly | Xq | - | - | 3.6 | - | - | - | torch-eager-fp32 2.7 ms (IDENTICAL/arm 1.356); torch-compile-fp32 2.7 ms (IDENTICAL/arm 1.322); torch-eager-bf16 3.0 ms (IDENTICAL/arm 1.194); torch-compile-bf16 3.1 ms (IDENTICAL/arm 1.149) |
| algos | svgp | istella | Xq | - | - | 58.6 | - | - | - | gpytorch-gpu 3.2 ms (IDENTICAL/arm 18.284); gpytorch-cpu 240.6 ms (IDENTICAL/arm 0.244) |
| algos | svgp | taxi | Xq | - | - | - | - | - | - | gpytorch-gpu 2.6 ms (IDENTICAL/arm -); gpytorch-cpu 144.9 ms (IDENTICAL/arm -) |
| classical | kmeans | istella | Xq | 500000 | - | 36.7 | - | - | - | torch-gpu 0.4 ms (IDENTICAL/arm 92.607); sklearn-cpu 23.3 ms (IDENTICAL/arm 1.764) |
| classical | kmeans | taxi | Xq | 500000 | - | 3.7 | - | - | - | torch-gpu 0.3 ms (IDENTICAL/arm 14.171); sklearn-cpu 4.6 ms (IDENTICAL/arm 0.974) |
| classical | ols | istella | Xq | 500000 | - | 12.3 | - | - | - | torch-gpu 0.2 ms (IDENTICAL/arm 60.078); torch-gpu-eigh 0.3 ms (IDENTICAL/arm 45.406); sklearn-cpu 18.5 ms (IDENTICAL/arm 0.665) |
| classical | ols | taxi | Xq | 500000 | - | 1.5 | - | - | - | torch-gpu 0.1 ms (IDENTICAL/arm 14.523); torch-gpu-eigh 0.1 ms (IDENTICAL/arm 16.328); sklearn-cpu 1.8 ms (IDENTICAL/arm 0.898) |
| classical | pca | istella | Xq | 500000 | - | 15.9 | - | - | - | torch-gpu 0.6 ms (IDENTICAL/arm 25.414); sklearn-cpu 24.7 ms (IDENTICAL/arm 0.715) |
| classical | pca | taxi | Xq | 500000 | - | 7.5 | - | - | - | torch-gpu 0.3 ms (IDENTICAL/arm 28.084); sklearn-cpu 5.6 ms (IDENTICAL/arm 1.220) |

## Classical

### kmeans / istella (rows full, shape 2043304x220)

race: done, driver rc 0, log `logs/classical.kmeans.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 736.0 | 736.0..736.0 | 1 | - | - | - | 4186.8 | 676.5 | inertia=6.051e+17, inertia_over_ours=1.000000, n_iter=33 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-gpu | torch | gpu | opponent | 1882.4 | 1882.4..1882.4 | 1 | 0.391 | - | - | 5867.6 | 3536.8 | inertia=5.991e+17, inertia_over_ours=0.990156, n_iter=57 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| sklearn-cpu | scikit-learn | cpu | opponent | 5614.6 | 5614.6..5614.6 | 1 | 0.131 | - | - | 5794.7 | - | inertia=6.049e+17, inertia_over_ours=0.999759, n_iter=24 | yes | LIKE-FOR-LIKE-SPAN | - | ok (stored (measured 2026-09-29T16:34:26Z on mojolearn-steward-do-amd, cpu (AMD EPYC 9575F 64-Core Processor))) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: n_clusters=8, init='k-means++', max_iter=300, n_init=1 (the cuML benchmark's KMeans), oversampling_factor=0 on ours and cuML (its cuml_args; the classic sequential seeding), tol=1e-7, metric='euclidean', Lloyd; seed 7 (ours, scikit-learn and cuML random_state=7, torch a generator seeded 7). Rows: big block: 4,000,000 taxi rows or the Istella-S train split, raw. Timed: fit (the k-means++ seeding included on every arm).

mismatch: k-means++: each library draws its own start from its own generator seeded 7, so the starts differ; scikit-learn greedy k-means++ (2 + log k candidates per center), ours and cuML the sequential k-means++ of cuML (oversampling_factor=0), torch-gpu scikit-learn's greedy rule written out

mismatch: tol=1e-7 on ours, scikit-learn and cuML (ours and cuML refuse 0); each library applies it through its own convergence test; torch-gpu stops by scikit-learn's (center shift <= tol x mean feature variance). n_iter is in every quality cell

mismatch: algorithm: scikit-learn algorithm='lloyd'; ours and cuML have no such parameter (Lloyd); torch-gpu is written as Lloyd

config: cuML benchmark (RAPIDS), KMeans (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-gpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | torch (declared) |
| init | "k-means++" | "k-means++" |
| max_iter | 300 | 300 |
| metric | "euclidean" | "euclidean" |
| n_clusters | 8 | 8 |
| n_init | 1 | 1 |
| oversampling_factor | 0.0 | - |
| seed | 7 | 7 |
| tol | 1e-07 | 1e-07 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | 500000 | 36.7 | 36.7..36.7 | 1 | - | - | - | eval_inertia=1.417e+17, label_agreement_own_centers=1.000000 | yes | LIKE-FOR-LIKE-SPAN | ok |
| torch-gpu | Xq | 500000 | 0.4 | 0.4..0.4 | 1 | 92.607 | - | - | agreement_vs_ours=0.038610, bits_equal_vs_ours=False, eval_inertia=1.402e+17, label_agreement_own_centers=1.000000 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| sklearn-cpu | Xq | 500000 | 23.3 | 23.3..23.3 | 1 | 1.764 | - | - | agreement_vs_ours=0.031282, bits_equal_vs_ours=False, eval_inertia=1.418e+17, label_agreement_own_centers=1.000000 | yes | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: mojolearn KMeans.predict(Xq), host rows in, host result out

inference call, torch-gpu: torch chunked addmm(//c//^2, Xq, c.T, alpha -2).argmin over the fitted centers; Xq uploaded before the clock, which ends at the device synchronize

inference call, sklearn-cpu: sklearn KMeans.predict(Xq), host rows in, host result out

### kmeans / taxi (rows full, shape 4000000x11)

race: done, driver rc 0, log `logs/classical.kmeans.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 396.0 | 396.0..396.0 | 1 | - | - | - | 2263.0 | 676.5 | inertia=3.093e+08, inertia_over_ours=1.000000, n_iter=91 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-gpu | torch | gpu | opponent | 798.3 | 798.3..798.3 | 1 | 0.496 | - | - | 3940.1 | 506.6 | inertia=3.129e+08, inertia_over_ours=1.011535, n_iter=80 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| sklearn-cpu | scikit-learn | cpu | opponent | 2297.6 | 2297.6..2297.6 | 1 | 0.172 | - | - | 772.1 | - | inertia=3.093e+08, inertia_over_ours=0.999937, n_iter=58 | yes | LIKE-FOR-LIKE-SPAN | - | ok (stored (measured 2026-09-29T16:33:52Z on mojolearn-steward-do-amd, cpu (AMD EPYC 9575F 64-Core Processor))) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: n_clusters=8, init='k-means++', max_iter=300, n_init=1 (the cuML benchmark's KMeans), oversampling_factor=0 on ours and cuML (its cuml_args; the classic sequential seeding), tol=1e-7, metric='euclidean', Lloyd; seed 7 (ours, scikit-learn and cuML random_state=7, torch a generator seeded 7). Rows: big block: 4,000,000 taxi rows or the Istella-S train split, raw. Timed: fit (the k-means++ seeding included on every arm).

mismatch: k-means++: each library draws its own start from its own generator seeded 7, so the starts differ; scikit-learn greedy k-means++ (2 + log k candidates per center), ours and cuML the sequential k-means++ of cuML (oversampling_factor=0), torch-gpu scikit-learn's greedy rule written out

mismatch: tol=1e-7 on ours, scikit-learn and cuML (ours and cuML refuse 0); each library applies it through its own convergence test; torch-gpu stops by scikit-learn's (center shift <= tol x mean feature variance). n_iter is in every quality cell

mismatch: algorithm: scikit-learn algorithm='lloyd'; ours and cuML have no such parameter (Lloyd); torch-gpu is written as Lloyd

config: cuML benchmark (RAPIDS), KMeans (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-gpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | torch (declared) |
| init | "k-means++" | "k-means++" |
| max_iter | 300 | 300 |
| metric | "euclidean" | "euclidean" |
| n_clusters | 8 | 8 |
| n_init | 1 | 1 |
| oversampling_factor | 0.0 | - |
| seed | 7 | 7 |
| tol | 1e-07 | 1e-07 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | 500000 | 3.7 | 3.7..3.7 | 1 | - | - | - | eval_inertia=4.824e+07, label_agreement_own_centers=0.999998 | yes | LIKE-FOR-LIKE-SPAN | ok |
| torch-gpu | Xq | 500000 | 0.3 | 0.3..0.3 | 1 | 14.171 | - | - | agreement_vs_ours=0.540178, bits_equal_vs_ours=False, eval_inertia=4.926e+07, label_agreement_own_centers=1.000000 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| sklearn-cpu | Xq | 500000 | 4.6 | 4.6..4.6 | 1 | 0.974 | - | - | agreement_vs_ours=0.003656, bits_equal_vs_ours=False, eval_inertia=4.816e+07, label_agreement_own_centers=1.000000 | yes | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: mojolearn KMeans.predict(Xq), host rows in, host result out

inference call, torch-gpu: torch chunked addmm(//c//^2, Xq, c.T, alpha -2).argmin over the fitted centers; Xq uploaded before the clock, which ends at the device synchronize

inference call, sklearn-cpu: sklearn KMeans.predict(Xq), host rows in, host result out

### knn / istella (rows full, shape 400000x220)

race: done, driver rc 0, log `logs/classical.knn.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1183.8 | 1183.8..1183.8 | 1 | - | - | - | 2368.8 | 1594.4 | recall_at_k=0.976250, rows_with_repeated_ids=0 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-gpu | torch | gpu | opponent | 29.1 | 29.1..29.1 | 1 | 40.740 | - | - | 3225.3 | 3884.6 | recall_at_k=0.979840, rows_with_repeated_ids=0 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| sklearn-cpu | scikit-learn | cpu | opponent | 536.9 | 536.9..536.9 | 1 | 2.205 | - | - | 587.8 | - | recall_at_k=1.000000, rows_with_repeated_ids=0 | yes | LIKE-FOR-LIKE-SPAN | - | ok (stored (measured 2026-09-29T16:36:26Z on mojolearn-steward-do-amd, cpu (AMD EPYC 9575F 64-Core Processor))) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: n_neighbors=64 (the cuML benchmark's NearestNeighbors), metric='euclidean', algorithm='brute' (ours, scikit-learn, cuML); torch cdist p=2 plus topk. Rows: knn block: 400,000 index rows, 4,000 queries, raw. Timed: kneighbors; the fit (index) is before the clock on every arm.

mismatch: seed: no arm has a seed argument (exact search); torch-gpu torch.manual_seed(7)

mismatch: query_tile: ours only (tiling, results unchanged); n_jobs=-1: scikit-learn only

config: cuML benchmark (RAPIDS), NearestNeighbors (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-gpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | torch (declared) |
| algorithm | "brute" | "brute" |
| metric | "euclidean" | "euclidean" |
| n_neighbors | 64 | 64 |
| p | 2 | 2 |
| seed | "none (deterministic)" | 7 |

### knn / taxi (rows full, shape 400000x11)

race: done, driver rc 0, log `logs/classical.knn.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1282.8 | 1282.8..1282.8 | 1 | - | - | - | 2046.5 | 1276.4 | recall_at_k=0.999754, rows_with_repeated_ids=0 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-gpu | torch | gpu | opponent | 23.9 | 23.9..23.9 | 1 | 53.751 | - | - | 2903.6 | 3242.7 | recall_at_k=0.999738, rows_with_repeated_ids=0 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| sklearn-cpu | scikit-learn | cpu | opponent | 381.1 | 381.1..381.1 | 1 | 3.366 | - | - | 238.1 | - | recall_at_k=1.000000, rows_with_repeated_ids=0 | yes | LIKE-FOR-LIKE-SPAN | - | ok (stored (measured 2026-09-29T16:36:05Z on mojolearn-steward-do-amd, cpu (AMD EPYC 9575F 64-Core Processor))) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: n_neighbors=64 (the cuML benchmark's NearestNeighbors), metric='euclidean', algorithm='brute' (ours, scikit-learn, cuML); torch cdist p=2 plus topk. Rows: knn block: 400,000 index rows, 4,000 queries, raw. Timed: kneighbors; the fit (index) is before the clock on every arm.

mismatch: seed: no arm has a seed argument (exact search); torch-gpu torch.manual_seed(7)

mismatch: query_tile: ours only (tiling, results unchanged); n_jobs=-1: scikit-learn only

config: cuML benchmark (RAPIDS), NearestNeighbors (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-gpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | torch (declared) |
| algorithm | "brute" | "brute" |
| metric | "euclidean" | "euclidean" |
| n_neighbors | 64 | 64 |
| p | 2 | 2 |
| seed | "none (deterministic)" | 7 |

### ols / istella (rows full, shape 2043304x220)

race: done, driver rc 0, log `logs/classical.ols.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1246.2 | 1246.2..1246.2 | 1 | - | - | - | 5892.2 | 674.5 | finite=True, r2=0.331944, rmse=0.682027 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-gpu | torch | gpu | opponent | 6950.7 | 6950.7..6950.7 | 1 | 0.179 | - | - | 3897.5 | 5304.1 | finite=False, r2=nan, rmse=nan | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-gpu-eigh | torch | gpu | opponent | 31.1 | 31.1..31.1 | 1 | 40.108 | - | - | 4890.2 | 3650.2 | finite=True, r2=0.151604, rmse=0.768590 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| sklearn-cpu | scikit-learn | cpu | opponent | 3980.6 | 3980.6..3980.6 | 1 | 0.313 | - | - | 5793.0 | - | finite=True, r2=0.001881, rmse=0.833655 | yes | LIKE-FOR-LIKE-SPAN | - | ok (stored (measured 2026-09-29T16:35:47Z on mojolearn-steward-do-amd, cpu (AMD EPYC 9575F 64-Core Processor))) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, torch-gpu, torch-gpu-eigh: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: fit_intercept=True. Rows: big block, raw; R2 and RMSE on the 500,000 eval rows. Timed: fit.

mismatch: seed: no arm has a seed argument (closed-form fit); torch-gpu torch.manual_seed(7)

mismatch: solver: ours eig of the normal equations (no parameter), scikit-learn scipy lstsq gelsd (no parameter), cuML algorithm='eig', torch-gpu torch.linalg.lstsq (gels on CUDA), torch-gpu-eigh eigh with a pseudo-inverse cutoff

mismatch: scikit-learn positive=False and copy_X=True: parameters ours does not have

config: cuML benchmark (RAPIDS), LinearRegression (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-gpu | torch-gpu-eigh |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | torch (declared) | torch (declared) |
| fit_intercept | true | true | true |
| seed | "none (deterministic)" | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | 500000 | 12.3 | 12.3..12.3 | 1 | - | - | - | predict_max_rel_err_own_fp64=9.581e-07, r2_eval=0.331944, rmse_eval=0.682027 | yes | LIKE-FOR-LIKE-SPAN | ok |
| torch-gpu | Xq | 500000 | 0.2 | 0.2..0.2 | 1 | 60.078 | - | - | bits_equal_vs_ours=False, max_abs_diff_vs_ours=nan, predict_max_rel_err_own_fp64=nan, r2_eval=nan, rmse_eval=nan | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-gpu-eigh | Xq | 500000 | 0.3 | 0.3..0.3 | 1 | 45.406 | - | - | bits_equal_vs_ours=False, max_abs_diff_vs_ours=6.699507, predict_max_rel_err_own_fp64=3.579e-07, r2_eval=0.151604, rmse_eval=0.768590 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| sklearn-cpu | Xq | 500000 | 18.5 | 18.5..18.5 | 1 | 0.665 | - | - | bits_equal_vs_ours=False, max_abs_diff_vs_ours=7.010309, predict_max_rel_err_own_fp64=7.648e-08, r2_eval=0.001881, rmse_eval=0.833655 | yes | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: mojolearn LinearRegression.predict(Xq), host rows in, host result out

inference call, torch-gpu: torch Xq @ coef + intercept; Xq uploaded before the clock, which ends at the device synchronize

inference call, torch-gpu-eigh: torch Xq @ coef + intercept; Xq uploaded before the clock, which ends at the device synchronize

inference call, sklearn-cpu: sklearn LinearRegression.predict(Xq), host rows in, host result out

### ols / taxi (rows full, shape 4000000x11)

race: done, driver rc 0, log `logs/classical.ols.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 170.6 | 170.6..170.6 | 1 | - | - | - | 2414.8 | 674.4 | finite=True, r2=0.908837, rmse=4.696466 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-gpu | torch | gpu | opponent | 113.3 | 113.3..113.3 | 1 | 1.505 | - | - | 1892.4 | 695.4 | finite=True, r2=0.908837, rmse=4.696472 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-gpu-eigh | torch | gpu | opponent | 24.3 | 24.3..24.3 | 1 | 7.012 | - | - | 2871.0 | 572.0 | finite=True, r2=0.908827, rmse=4.696715 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| sklearn-cpu | scikit-learn | cpu | opponent | 304.7 | 304.7..304.7 | 1 | 0.560 | - | - | 782.5 | - | finite=True, r2=0.724850, rmse=8.159187 | yes | LIKE-FOR-LIKE-SPAN | - | ok (stored (measured 2026-09-29T16:35:14Z on mojolearn-steward-do-amd, cpu (AMD EPYC 9575F 64-Core Processor))) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, torch-gpu, torch-gpu-eigh: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: fit_intercept=True. Rows: big block, raw; R2 and RMSE on the 500,000 eval rows. Timed: fit.

mismatch: seed: no arm has a seed argument (closed-form fit); torch-gpu torch.manual_seed(7)

mismatch: solver: ours eig of the normal equations (no parameter), scikit-learn scipy lstsq gelsd (no parameter), cuML algorithm='eig', torch-gpu torch.linalg.lstsq (gels on CUDA), torch-gpu-eigh eigh with a pseudo-inverse cutoff

mismatch: scikit-learn positive=False and copy_X=True: parameters ours does not have

config: cuML benchmark (RAPIDS), LinearRegression (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-gpu | torch-gpu-eigh |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | torch (declared) | torch (declared) |
| fit_intercept | true | true | true |
| seed | "none (deterministic)" | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | 500000 | 1.5 | 1.5..1.5 | 1 | - | - | - | predict_max_rel_err_own_fp64=8.387e-08, r2_eval=0.908837, rmse_eval=4.696466 | yes | LIKE-FOR-LIKE-SPAN | ok |
| torch-gpu | Xq | 500000 | 0.1 | 0.1..0.1 | 1 | 14.523 | - | - | bits_equal_vs_ours=False, max_abs_diff_vs_ours=0.058048, predict_max_rel_err_own_fp64=9.855e-08, r2_eval=0.908837, rmse_eval=4.696472 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-gpu-eigh | Xq | 500000 | 0.1 | 0.1..0.1 | 1 | 16.328 | - | - | bits_equal_vs_ours=False, max_abs_diff_vs_ours=0.533283, predict_max_rel_err_own_fp64=7.82e-08, r2_eval=0.908827, rmse_eval=4.696715 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| sklearn-cpu | Xq | 500000 | 1.8 | 1.8..1.8 | 1 | 0.898 | - | - | bits_equal_vs_ours=False, max_abs_diff_vs_ours=197.960266, predict_max_rel_err_own_fp64=1.385e-07, r2_eval=0.724850, rmse_eval=8.159187 | yes | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: mojolearn LinearRegression.predict(Xq), host rows in, host result out

inference call, torch-gpu: torch Xq @ coef + intercept; Xq uploaded before the clock, which ends at the device synchronize

inference call, torch-gpu-eigh: torch Xq @ coef + intercept; Xq uploaded before the clock, which ends at the device synchronize

inference call, sklearn-cpu: sklearn LinearRegression.predict(Xq), host rows in, host result out

### pca / istella (rows full, shape 2043304x220)

race: done, driver rc 0, log `logs/classical.pca.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 287.7 | 287.7..287.7 | 1 | - | - | - | 4171.8 | 674.4 | explained_variance_ratio_sum=1.000000 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-gpu | torch | gpu | opponent | 9.3 | 9.3..9.3 | 1 | 30.965 | - | - | 4861.4 | 3506.7 | explained_variance_ratio_sum=1.000000 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| sklearn-cpu | scikit-learn | cpu | opponent | 621.7 | 621.7..621.7 | 1 | 0.463 | - | - | 2342.9 | - | explained_variance_ratio_sum=1.000000 | yes | LIKE-FOR-LIKE-SPAN | - | ok (stored (measured 2026-09-29T16:34:57Z on mojolearn-steward-do-amd, cpu (AMD EPYC 9575F 64-Core Processor))) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: n_components=10 (the cuML benchmark's PCA), whiten=False, random_state=7; ours and scikit-learn svd_solver='covariance_eigh'. Rows: big block, raw. Timed: fit.

mismatch: svd_solver: cuML has no 'covariance_eigh'; cuml-gpu runs svd_solver='full'

mismatch: torch-gpu: no estimator; covariance eigh written out, torch.manual_seed(7) (it draws nothing)

mismatch: random_state is read by none of the covariance solvers; set to 7 on every arm

config: cuML benchmark (RAPIDS), PCA (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-gpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | torch (declared) |
| n_components | 10 | 10 |
| seed | 7 | 7 |
| svd_solver | "covariance_eigh" | "covariance_eigh" |
| tol | 0.0 | - |
| whiten | false | false |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | 500000 | 15.9 | 15.9..15.9 | 1 | - | - | - | transform_max_rel_err_own_fp64=3.75e-07 | yes | LIKE-FOR-LIKE-SPAN | ok |
| torch-gpu | Xq | 500000 | 0.6 | 0.6..0.6 | 1 | 25.414 | - | - | bits_equal_vs_ours=False, max_abs_diff_vs_ours=1.678e+06, transform_max_rel_err_own_fp64=3.427e-07 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| sklearn-cpu | Xq | 500000 | 24.7 | 24.7..24.7 | 1 | 0.715 | - | - | bits_equal_vs_ours=False, max_abs_diff_vs_ours=64529.218750, transform_max_rel_err_own_fp64=3.857e-07 | yes | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: mojolearn PCA.transform(Xq), host rows in, host result out

inference call, torch-gpu: torch (Xq - mean) @ components.T; Xq uploaded before the clock, which ends at the device synchronize

inference call, sklearn-cpu: sklearn PCA.transform(Xq), host rows in, host result out

### pca / taxi (rows full, shape 4000000x11)

race: done, driver rc 0, log `logs/classical.pca.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 20.3 | 20.3..20.3 | 1 | - | - | - | 2233.1 | 674.3 | explained_variance_ratio_sum=0.999996 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-gpu | torch | gpu | opponent | 1.2 | 1.2..1.2 | 1 | 17.604 | - | - | 2908.1 | 412.0 | explained_variance_ratio_sum=0.999996 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| sklearn-cpu | scikit-learn | cpu | opponent | 82.2 | 82.2..82.2 | 1 | 0.247 | - | - | 401.6 | - | explained_variance_ratio_sum=0.999995 | yes | LIKE-FOR-LIKE-SPAN | - | ok (stored (measured 2026-09-29T16:34:39Z on mojolearn-steward-do-amd, cpu (AMD EPYC 9575F 64-Core Processor))) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: n_components=10 (the cuML benchmark's PCA), whiten=False, random_state=7; ours and scikit-learn svd_solver='covariance_eigh'. Rows: big block, raw. Timed: fit.

mismatch: svd_solver: cuML has no 'covariance_eigh'; cuml-gpu runs svd_solver='full'

mismatch: torch-gpu: no estimator; covariance eigh written out, torch.manual_seed(7) (it draws nothing)

mismatch: random_state is read by none of the covariance solvers; set to 7 on every arm

config: cuML benchmark (RAPIDS), PCA (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-gpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | torch (declared) |
| n_components | 10 | 10 |
| seed | 7 | 7 |
| svd_solver | "covariance_eigh" | "covariance_eigh" |
| tol | 0.0 | - |
| whiten | false | false |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | 500000 | 7.5 | 7.5..7.5 | 1 | - | - | - | transform_max_rel_err_own_fp64=1.085e-07 | yes | LIKE-FOR-LIKE-SPAN | ok |
| torch-gpu | Xq | 500000 | 0.3 | 0.3..0.3 | 1 | 28.084 | - | - | bits_equal_vs_ours=False, max_abs_diff_vs_ours=322.178131, transform_max_rel_err_own_fp64=1.245e-07 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| sklearn-cpu | Xq | 500000 | 5.6 | 5.6..5.6 | 1 | 1.220 | - | - | bits_equal_vs_ours=False, max_abs_diff_vs_ours=0.897790, transform_max_rel_err_own_fp64=1.299e-07 | yes | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: mojolearn PCA.transform(Xq), host rows in, host result out

inference call, torch-gpu: torch (Xq - mean) @ components.T; Xq uploaded before the clock, which ends at the device synchronize

inference call, sklearn-cpu: sklearn PCA.transform(Xq), host rows in, host result out

## Neural

### gemm-bf16 / gaussian (neural shape full: 4096x4096x4096)

race: done, driver rc 0, log `logs/neural.gemm-bf16.gaussian.shape-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 30.6 | 30.6..30.6 | 1 | - | - | - | 1035.6 | 3244.4 | max_rel_err_vs_fp64=1.155e-07 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-bf16 | torch | gpu | opponent | 13.6 | 13.6..13.6 | 1 | 2.251 | - | - | 2688.6 | 312.0 | max_abs_diff_vs_ours=0.999023, max_rel_diff_vs_ours=0.002759, max_rel_err_vs_fp64=0.002759 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 39.2 | 39.2..39.2 | 1 | 0.781 | - | - | 2883.9 | 312.0 | max_abs_diff_vs_ours=0.999023, max_rel_diff_vs_ours=0.002759, max_rel_err_vs_fp64=0.002759 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-eager-bf16 |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | 7 | 7 |

### gemm-int8 / gaussian (neural shape full: 4096x4096x4096)

race: done, driver rc 0, log `logs/neural.gemm-int8.gaussian.shape-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1799.0 | 1799.0..1799.0 | 1 | - | - | - | 936.4 | 674.3 | max_rel_err_vs_fp64=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours |
|---|---|
| library (source) | mojolearn (declared) |
| seed | "none (deterministic)" |

### gemm / gaussian (neural shape full: 4096x4096x4096)

race: done, driver rc 0, log `logs/neural.gemm.gaussian.shape-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 27.4 | 27.4..27.4 | 1 | - | - | - | 969.7 | 3244.3 | max_rel_err_vs_fp64=2.399e-07 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 12.0 | 12.0..12.0 | 1 | 2.293 | - | - | 1789.3 | 268.0 | max_abs_diff_vs_ours=0.001038, max_rel_diff_vs_ours=2.866e-06, max_rel_err_vs_fp64=2.802e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 14.5 | 14.5..14.5 | 1 | 1.896 | - | - | 2017.8 | 268.0 | max_abs_diff_vs_ours=0.001038, max_rel_diff_vs_ours=2.866e-06, max_rel_err_vs_fp64=2.802e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 12.9 | 12.9..12.9 | 1 | 2.121 | - | - | 2626.3 | 376.0 | max_abs_diff_vs_ours=1.358337, max_rel_diff_vs_ours=0.003751, max_rel_err_vs_fp64=0.003752 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 12.8 | 12.8..12.8 | 1 | 2.137 | - | - | 3071.1 | 376.0 | max_abs_diff_vs_ours=1.358337, max_rel_diff_vs_ours=0.003751, max_rel_err_vs_fp64=0.003752 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 |

### lm-forward / bytes (neural shape full: B1 L2048 DM384 H6 KV6 HD64 FF1024 layers8 V8192)

race: done, driver rc 0, log `logs/neural.lm-forward.bytes.shape-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 91.3 | 91.3..91.3 | 1 | - | - | - | 2352.1 | 7284.5 | mean_nll=9.018733 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 11.3 | 11.3..11.3 | 1 | 8.115 | - | - | 3072.9 | 241.0 | max_abs_diff_vs_ours=1.222e-06, max_rel_diff_vs_ours=1.677e-06, mean_nll=9.018733 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 11.0 | 11.0..11.0 | 1 | 8.281 | - | - | 3255.9 | 222.5 | max_abs_diff_vs_ours=1.192e-06, max_rel_diff_vs_ours=1.637e-06, mean_nll=9.018733 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 13.5 | 13.5..13.5 | 1 | 6.759 | - | - | 2797.4 | 327.5 | max_abs_diff_vs_ours=0.006313, max_rel_diff_vs_ours=0.008666, mean_nll=9.018647 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 13.3 | 13.3..13.3 | 1 | 6.855 | - | - | 3011.1 | 327.5 | max_abs_diff_vs_ours=0.006870, max_rel_diff_vs_ours=0.009432, mean_nll=9.018653 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 |

### lm-host-train-step / bytes (neural shape full: B1 L512 DM384 H6 KV6 HD64 FF1024 layers8 V8192)

race: done, driver rc 0, log `logs/neural.lm-host-train-step.bytes.shape-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | cpu | identical | 1549.8 | 1549.8..1549.8 | 1 | - | - | - | 2134.2 | - | loss_first_step=9.017858, loss_last_step=8.367768, steps=2 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-cpu-eager-fp32 | torch | cpu | opponent | 86.9 | 86.9..86.9 | 1 | 17.828 | - | - | 1593.2 | - | loss_first_abs_diff_vs_ours=9.537e-07, loss_first_step=9.017857, loss_last_abs_diff_vs_ours=9.537e-07, loss_last_step=8.367767, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-compile-fp32 | torch | cpu | opponent | 75.6 | 75.6..75.6 | 1 | 20.503 | - | - | 1636.2 | - | loss_first_abs_diff_vs_ours=9.537e-07, loss_first_step=9.017857, loss_last_abs_diff_vs_ours=1.907e-06, loss_last_step=8.367766, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-eager-bf16 | torch | cpu | opponent | 68.5 | 68.5..68.5 | 1 | 22.619 | - | - | 1538.1 | - | loss_first_abs_diff_vs_ours=0.0001106, loss_first_step=9.017747, loss_last_abs_diff_vs_ours=0.0005798, loss_last_step=8.368348, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-compile-bf16 | torch | cpu | opponent | 48.8 | 48.8..48.8 | 1 | 31.729 | - | - | 1641.1 | - | loss_first_abs_diff_vs_ours=9.06e-05, loss_first_step=9.017767, loss_last_abs_diff_vs_ours=1.24e-05, loss_last_step=8.367756, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, torch-cpu-eager-fp32, torch-cpu-compile-fp32, torch-cpu-eager-bf16, torch-cpu-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-cpu-compile-bf16 | torch-cpu-compile-fp32 | torch-cpu-eager-bf16 | torch-cpu-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| amsgrad | false | false | false | false | false |
| betas | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| eps | 1e-08 | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 |
| weight_decay | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |

### lm-infer / bytes (neural shape full: B1 L512 DM384 H6 KV6 HD64 FF1024 layers8 V8192)

race: done, driver rc 0, log `logs/neural.lm-infer.bytes.shape-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | cpu | identical | 778.5 | 778.5..778.5 | 1 | - | - | - | 460.3 | - | mean_nll=9.017857 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-cpu-eager-fp32 | torch | cpu | opponent | 25.7 | 25.7..25.7 | 1 | 30.266 | - | - | 1083.4 | - | max_abs_diff_vs_ours=7.227e-07, max_rel_diff_vs_ours=9.921e-07, mean_nll=9.017857 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-compile-fp32 | torch | cpu | opponent | 18.0 | 18.0..18.0 | 1 | 43.232 | - | - | 1323.9 | - | max_abs_diff_vs_ours=6.706e-07, max_rel_diff_vs_ours=9.205e-07, mean_nll=9.017857 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-eager-bf16 | torch | cpu | opponent | 17.2 | 17.2..17.2 | 1 | 45.135 | - | - | 1107.5 | - | max_abs_diff_vs_ours=0.006180, max_rel_diff_vs_ours=0.008484, mean_nll=9.017748 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-compile-bf16 | torch | cpu | opponent | 11.8 | 11.8..11.8 | 1 | 66.079 | - | - | 1331.1 | - | max_abs_diff_vs_ours=0.006585, max_rel_diff_vs_ours=0.009040, mean_nll=9.017761 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, torch-cpu-eager-fp32, torch-cpu-compile-fp32, torch-cpu-eager-bf16, torch-cpu-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-cpu-compile-bf16 | torch-cpu-compile-fp32 | torch-cpu-eager-bf16 | torch-cpu-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 |

### lm-train-step / bytes (neural shape full: B1 L2048 DM384 H6 KV6 HD64 FF1024 layers8 V8192)

race: done, driver rc 0, log `logs/neural.lm-train-step.bytes.shape-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 123.6 | 123.6..123.6 | 1 | - | - | - | 2125.8 | 6505.1 | loss_first_step=9.018733, loss_last_step=8.422411, steps=2 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 12.8 | 12.8..12.8 | 1 | 9.691 | - | - | 4803.5 | 1089.6 | loss_first_abs_diff_vs_ours=9.537e-07, loss_first_step=9.018732, loss_last_abs_diff_vs_ours=0.000000, loss_last_step=8.422411, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 10.3 | 10.3..10.3 | 1 | 12.028 | - | - | 4852.1 | 859.6 | loss_first_abs_diff_vs_ours=9.537e-07, loss_first_step=9.018734, loss_last_abs_diff_vs_ours=9.537e-07, loss_last_step=8.422412, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 13.4 | 13.4..13.4 | 1 | 9.209 | - | - | 3061.0 | 930.6 | loss_first_abs_diff_vs_ours=5.627e-05, loss_first_step=9.018677, loss_last_abs_diff_vs_ours=0.002001, loss_last_step=8.420410, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 45.7 | 45.7..45.7 | 1 | 2.703 | - | - | 3070.1 | 688.1 | loss_first_abs_diff_vs_ours=8.011e-05, loss_first_step=9.018653, loss_last_abs_diff_vs_ours=0.003033, loss_last_step=8.419378, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| amsgrad | false | false | false | false | false |
| betas | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| eps | 1e-08 | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 |
| weight_decay | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |

### mamba1-forward / gaussian (neural shape full: B1 L2048 DM384)

race: done, driver rc 0, log `logs/neural.mamba1-forward.gaussian.shape-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 18.9 | 18.9..18.9 | 1 | - | - | - | 731.2 | 3246.6 | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 87.7 | 87.7..87.7 | 1 | 0.216 | - | - | 3964.5 | 422.2 | max_abs_diff_vs_ours=2.384e-07, max_rel_diff_vs_ours=1.189e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 110.9 | 110.9..110.9 | 1 | 0.171 | - | - | 3654.8 | 442.5 | max_abs_diff_vs_ours=5.436e-05, max_rel_diff_vs_ours=2.711e-05 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-eager-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | 7 | 7 |
| ssm_d_conv | 4 | - | - |
| ssm_d_state | 16 | - | - |
| ssm_dt_rank | 24 | - | - |
| ssm_expand | 2 | - | - |

### mamba1-infer / gaussian (neural shape full: B1 L512 DM384)

race: done, driver rc 0, log `logs/neural.mamba1-infer.gaussian.shape-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | cpu | identical | 48.5 | 48.5..48.5 | 1 | - | - | - | 104.6 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-cpu-eager-fp32 | torch | cpu | opponent | 24.3 | 24.3..24.3 | 1 | 1.992 | - | - | 1124.2 | - | max_abs_diff_vs_ours=1.192e-07, max_rel_diff_vs_ours=5.948e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-eager-bf16 | torch | cpu | opponent | 38.6 | 38.6..38.6 | 1 | 1.255 | - | - | 1156.3 | - | max_abs_diff_vs_ours=5.15e-05, max_rel_diff_vs_ours=2.57e-05 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, torch-cpu-eager-fp32, torch-cpu-eager-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-cpu-eager-bf16 | torch-cpu-eager-fp32 |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | 7 | 7 |

### mamba2-forward / gaussian (neural shape full: B1 L2048 DM384)

race: done, driver rc 0, log `logs/neural.mamba2-forward.gaussian.shape-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 16.6 | 16.6..16.6 | 1 | - | - | - | 734.9 | 3248.6 | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 9.0 | 9.0..9.0 | 1 | 1.851 | - | - | 3394.8 | 3296.3 | max_abs_diff_vs_ours=1.073e-06, max_rel_diff_vs_ours=3.648e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 10.5 | 10.5..10.5 | 1 | 1.581 | - | - | 3709.4 | 3296.5 | max_abs_diff_vs_ours=1.073e-06, max_rel_diff_vs_ours=3.648e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 8.2 | 8.2..8.2 | 1 | 2.035 | - | - | 3174.5 | 1828.8 | max_abs_diff_vs_ours=0.007144, max_rel_diff_vs_ours=0.002429 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 47.0 | 47.0..47.0 | 1 | 0.354 | - | - | 3467.4 | 1828.8 | max_abs_diff_vs_ours=0.007144, max_rel_diff_vs_ours=0.002429 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 |
| ssm_chunk_size | 256 | - | - | - | - |
| ssm_d_conv | 4 | - | - | - | - |
| ssm_d_state | 128 | - | - | - | - |
| ssm_dt_limit | [0.0, Infinity] | - | - | - | - |
| ssm_expand | 2 | - | - | - | - |
| ssm_headdim | 64 | - | - | - | - |
| ssm_ngroups | 1 | - | - | - | - |

### mamba2-infer / gaussian (neural shape full: B1 L512 DM384)

race: done, driver rc 0, log `logs/neural.mamba2-infer.gaussian.shape-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | cpu | identical | 37.7 | 37.7..37.7 | 1 | - | - | - | 122.6 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-cpu-eager-fp32 | torch | cpu | opponent | 121.3 | 121.3..121.3 | 1 | 0.311 | - | - | 1836.8 | - | max_abs_diff_vs_ours=7.749e-07, max_rel_diff_vs_ours=2.751e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-compile-fp32 | torch | cpu | opponent | 118.7 | 118.7..118.7 | 1 | 0.317 | - | - | 1949.1 | - | max_abs_diff_vs_ours=7.749e-07, max_rel_diff_vs_ours=2.751e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-eager-bf16 | torch | cpu | opponent | 96.7 | 96.7..96.7 | 1 | 0.390 | - | - | 1455.9 | - | max_abs_diff_vs_ours=0.005920, max_rel_diff_vs_ours=0.002102 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-compile-bf16 | torch | cpu | opponent | 94.8 | 94.8..94.8 | 1 | 0.398 | - | - | 1558.5 | - | max_abs_diff_vs_ours=0.005920, max_rel_diff_vs_ours=0.002102 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, torch-cpu-eager-fp32, torch-cpu-compile-fp32, torch-cpu-eager-bf16, torch-cpu-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-cpu-compile-bf16 | torch-cpu-compile-fp32 | torch-cpu-eager-bf16 | torch-cpu-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 |

### mamba3-forward / gaussian (neural shape full: B1 L2048 DM384)

race: done, driver rc 0, log `logs/neural.mamba3-forward.gaussian.shape-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 11.4 | 11.4..11.4 | 1 | - | - | - | 2038.2 | 3248.7 | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 69.3 | 69.3..69.3 | 1 | 0.165 | - | - | 4951.0 | 287.5 | max_abs_diff_vs_ours=3.874e-07, max_rel_diff_vs_ours=1.695e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 6.5 | 6.5..6.5 | 1 | 1.759 | - | - | 8023.3 | 151.5 | max_abs_diff_vs_ours=4.768e-07, max_rel_diff_vs_ours=2.086e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 70.4 | 70.4..70.4 | 1 | 0.162 | - | - | 3308.0 | 347.4 | max_abs_diff_vs_ours=0.002069, max_rel_diff_vs_ours=0.0009051 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 7.1 | 7.1..7.1 | 1 | 1.600 | - | - | 6694.3 | 226.2 | max_abs_diff_vs_ours=0.001982, max_rel_diff_vs_ours=0.0008672 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 |
| ssm_chunk_size | 64 | - | - | - | - |
| ssm_d_state | 128 | - | - | - | - |
| ssm_expand | 2 | - | - | - | - |
| ssm_headdim | 64 | - | - | - | - |
| ssm_ngroups | 1 | - | - | - | - |
| ssm_rope_angles | 32 | - | - | - | - |

### mamba3-infer / gaussian (neural shape full: B1 L512 DM384)

race: done, driver rc 0, log `logs/neural.mamba3-infer.gaussian.shape-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | cpu | identical | 40.3 | 40.3..40.3 | 1 | - | - | - | 128.3 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-cpu-eager-fp32 | torch | cpu | opponent | 11.6 | 11.6..11.6 | 1 | 3.487 | - | - | 1097.6 | - | max_abs_diff_vs_ours=2.384e-07, max_rel_diff_vs_ours=1.06e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-compile-fp32 | torch | cpu | opponent | 4.1 | 4.1..4.1 | 1 | 9.849 | - | - | 1365.7 | - | max_abs_diff_vs_ours=2.98e-07, max_rel_diff_vs_ours=1.324e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-eager-bf16 | torch | cpu | opponent | 15.7 | 15.7..15.7 | 1 | 2.563 | - | - | 1101.4 | - | max_abs_diff_vs_ours=0.002148, max_rel_diff_vs_ours=0.0009547 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-compile-bf16 | torch | cpu | opponent | 3.6 | 3.6..3.6 | 1 | 11.346 | - | - | 1361.3 | - | max_abs_diff_vs_ours=0.002040, max_rel_diff_vs_ours=0.0009065 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, torch-cpu-eager-fp32, torch-cpu-compile-fp32, torch-cpu-eager-bf16, torch-cpu-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-cpu-compile-bf16 | torch-cpu-compile-fp32 | torch-cpu-eager-bf16 | torch-cpu-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 |

### mlp-infer / gaussian (neural shape full: rows256 8-16-3)

race: done, driver rc 0, log `logs/neural.mlp-infer.gaussian.shape-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | cpu | identical | 0.2 | 0.2..0.2 | 1 | - | - | - | 71.4 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-cpu-eager-fp32 | torch | cpu | opponent | 0.3 | 0.3..0.3 | 1 | 0.562 | - | - | 961.7 | - | max_abs_diff_vs_ours=1.192e-07, max_rel_diff_vs_ours=1.1e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-compile-fp32 | torch | cpu | opponent | 0.5 | 0.5..0.5 | 1 | 0.358 | - | - | 1140.4 | - | max_abs_diff_vs_ours=1.192e-07, max_rel_diff_vs_ours=1.1e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-eager-bf16 | torch | cpu | opponent | 0.5 | 0.5..0.5 | 1 | 0.389 | - | - | 978.8 | - | max_abs_diff_vs_ours=0.004285, max_rel_diff_vs_ours=0.003956 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-compile-bf16 | torch | cpu | opponent | 0.3 | 0.3..0.3 | 1 | 0.647 | - | - | 1148.7 | - | max_abs_diff_vs_ours=0.004285, max_rel_diff_vs_ours=0.003956 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, torch-cpu-eager-fp32, torch-cpu-compile-fp32, torch-cpu-eager-bf16, torch-cpu-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-cpu-compile-bf16 | torch-cpu-compile-fp32 | torch-cpu-eager-bf16 | torch-cpu-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 |

### mlp-train-step / gaussian (neural shape full: rows256 8-16-3)

race: done, driver rc 0, log `logs/neural.mlp-train-step.gaussian.shape-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 2.8 | 2.8..2.8 | 1 | - | - | - | 2031.5 | 676.5 | loss_first_step=1.160401, loss_last_step=1.123361, steps=2 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 1.7 | 1.7..1.7 | 1 | 1.616 | - | - | 4354.9 | 152.1 | loss_first_abs_diff_vs_ours=2.384e-07, loss_first_step=1.160401, loss_last_abs_diff_vs_ours=2.384e-07, loss_last_step=1.123361, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 1.8 | 1.8..1.8 | 1 | 1.587 | - | - | 4263.6 | 152.1 | loss_first_abs_diff_vs_ours=2.384e-07, loss_first_step=1.160401, loss_last_abs_diff_vs_ours=2.384e-07, loss_last_step=1.123361, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 1.5 | 1.5..1.5 | 1 | 1.925 | - | - | 2707.7 | 152.0 | loss_first_abs_diff_vs_ours=9.656e-05, loss_first_step=1.160498, loss_last_abs_diff_vs_ours=0.0001005, loss_last_step=1.123461, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 71.3 | 71.3..71.3 | 1 | 0.040 | - | - | 2828.7 | 152.0 | loss_first_abs_diff_vs_ours=9.656e-05, loss_first_step=1.160498, loss_last_abs_diff_vs_ours=0.0001006, loss_last_step=1.123461, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| amsgrad | false | false | false | false | false |
| betas | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| eps | 1e-08 | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 |
| weight_decay | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |

### samba-forward / bytes (neural shape full: B2 L512 DM384 V256 H6 FF1024 layers mamba3+attention+mamba3+attention)

race: done, driver rc 0, log `logs/neural.samba-forward.bytes.shape-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 26.7 | 26.7..26.7 | 1 | - | - | - | 2493.0 | 8881.7 | mean_nll=5.635910 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 40.5 | 40.5..40.5 | 1 | 0.659 | - | - | 3922.7 | 205.4 | max_abs_diff_vs_ours=2.146e-06, max_rel_diff_vs_ours=1.206e-06, mean_nll=5.635910 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 4.4 | 4.4..4.4 | 1 | 6.079 | - | - | 4862.8 | 132.8 | max_abs_diff_vs_ours=2.168e-06, max_rel_diff_vs_ours=1.219e-06, mean_nll=5.635910 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 40.3 | 40.3..40.3 | 1 | 0.662 | - | - | 3343.2 | 280.2 | max_abs_diff_vs_ours=0.017180, max_rel_diff_vs_ours=0.009659, mean_nll=5.635950 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 4.5 | 4.5..4.5 | 1 | 5.940 | - | - | 4269.0 | 209.3 | max_abs_diff_vs_ours=0.016678, max_rel_diff_vs_ours=0.009377, mean_nll=5.635933 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| dropout | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| eps | 1e-05 | 1e-05 | 1e-05 | 1e-05 | 1e-05 |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 |

### samba-infer / bytes (neural shape full: B2 L512 DM384 V256 H6 FF1024 layers mamba3+attention+mamba3+attention)

race: done, driver rc 0, log `logs/neural.samba-infer.bytes.shape-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | cpu | identical | 236.6 | 236.6..236.6 | 1 | - | - | - | 237.2 | - | mean_nll=5.635910 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-cpu-eager-fp32 | torch | cpu | opponent | 39.1 | 39.1..39.1 | 1 | 6.046 | - | - | 1143.1 | - | max_abs_diff_vs_ours=1.997e-06, max_rel_diff_vs_ours=1.123e-06, mean_nll=5.635910 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-compile-fp32 | torch | cpu | opponent | 21.7 | 21.7..21.7 | 1 | 10.887 | - | - | 1686.0 | - | max_abs_diff_vs_ours=1.848e-06, max_rel_diff_vs_ours=1.039e-06, mean_nll=5.635910 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-eager-bf16 | torch | cpu | opponent | 45.5 | 45.5..45.5 | 1 | 5.202 | - | - | 1162.8 | - | max_abs_diff_vs_ours=0.018279, max_rel_diff_vs_ours=0.010276, mean_nll=5.635976 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-compile-bf16 | torch | cpu | opponent | 14.4 | 14.4..14.4 | 1 | 16.444 | - | - | 1681.6 | - | max_abs_diff_vs_ours=0.016166, max_rel_diff_vs_ours=0.009088, mean_nll=5.635914 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, torch-cpu-eager-fp32, torch-cpu-compile-fp32, torch-cpu-eager-bf16, torch-cpu-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-cpu-compile-bf16 | torch-cpu-compile-fp32 | torch-cpu-eager-bf16 | torch-cpu-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| dropout | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| eps | 1e-05 | 1e-05 | 1e-05 | 1e-05 | 1e-05 |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 |

### samba-train-step / bytes (neural shape full: B2 L512 DM384 V256 H6 FF1024 layers mamba3+attention+mamba3+attention)

race: done, driver rc 0, log `logs/neural.samba-train-step.bytes.shape-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 239.6 | 239.6..239.6 | 1 | - | - | - | 2574.1 | 8962.7 | loss_first_step=5.635910, loss_last_step=4.833934, steps=2 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 116.1 | 116.1..116.1 | 1 | 2.063 | - | - | 5209.4 | 534.3 | loss_first_abs_diff_vs_ours=4.768e-07, loss_first_step=5.635910, loss_last_abs_diff_vs_ours=4.768e-07, loss_last_step=4.833934, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 30.4 | 30.4..30.4 | 1 | 7.883 | - | - | 7777.2 | 449.6 | loss_first_abs_diff_vs_ours=9.537e-07, loss_first_step=5.635909, loss_last_abs_diff_vs_ours=4.768e-07, loss_last_step=4.833934, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 113.9 | 113.9..113.9 | 1 | 2.105 | - | - | 3500.1 | 485.2 | loss_first_abs_diff_vs_ours=3.91e-05, loss_first_step=5.635949, loss_last_abs_diff_vs_ours=8.106e-05, loss_last_step=4.834015, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(error: {"error": "REFUSED: torch-compile-bf16 on cuda failed in round 0 (compile happens here): InductorError('RuntimeError: A compilation subprocess exited unexpectedly. This is likely due to a crash. To fa) (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, torch-compile-bf16: host not sampled; GPU not sampled

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| amsgrad | false | false | false | false | false |
| betas | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| dropout | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| eps | 1e-08 | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 |
| weight_decay | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |

### transformer-forward / gaussian (neural shape full: B1 L2048 DM384 H6 KV6 HD64 FF1024)

race: done, driver rc 0, log `logs/neural.transformer-forward.gaussian.shape-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 14.4 | 14.4..14.4 | 1 | - | - | - | 2044.2 | 3248.5 | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 1.9 | 1.9..1.9 | 1 | 7.480 | - | - | 2827.0 | 147.3 | max_abs_diff_vs_ours=4.768e-07, max_rel_diff_vs_ours=1.02e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 1.8 | 1.8..1.8 | 1 | 7.799 | - | - | 3015.3 | 119.3 | max_abs_diff_vs_ours=4.768e-07, max_rel_diff_vs_ours=1.02e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 1.5 | 1.5..1.5 | 1 | 9.346 | - | - | 2661.9 | 215.1 | max_abs_diff_vs_ours=0.001819, max_rel_diff_vs_ours=0.0003893 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 65.8 | 65.8..65.8 | 1 | 0.219 | - | - | 2808.3 | 178.3 | max_abs_diff_vs_ours=0.001819, max_rel_diff_vs_ours=0.0003893 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| eps | 1e-06 | 1e-06 | 1e-06 | 1e-06 | 1e-06 |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 |

### transformer-infer / gaussian (neural shape full: B1 L512 DM384 H6 KV6 HD64 FF1024)

race: done, driver rc 0, log `logs/neural.transformer-infer.gaussian.shape-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | cpu | identical | 26.5 | 26.5..26.5 | 1 | - | - | - | 143.3 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-cpu-eager-fp32 | torch | cpu | opponent | 3.5 | 3.5..3.5 | 1 | 7.496 | - | - | 1100.9 | - | max_abs_diff_vs_ours=4.768e-07, max_rel_diff_vs_ours=1.076e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-compile-fp32 | torch | cpu | opponent | 3.0 | 3.0..3.0 | 1 | 8.883 | - | - | 1161.2 | - | max_abs_diff_vs_ours=9.537e-07, max_rel_diff_vs_ours=2.153e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-eager-bf16 | torch | cpu | opponent | 3.1 | 3.1..3.1 | 1 | 8.461 | - | - | 1108.7 | - | max_abs_diff_vs_ours=0.001941, max_rel_diff_vs_ours=0.0004383 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-compile-bf16 | torch | cpu | opponent | 2.0 | 2.0..2.0 | 1 | 13.096 | - | - | 1171.8 | - | max_abs_diff_vs_ours=0.001819, max_rel_diff_vs_ours=0.0004107 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, torch-cpu-eager-fp32, torch-cpu-compile-fp32, torch-cpu-eager-bf16, torch-cpu-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-cpu-compile-bf16 | torch-cpu-compile-fp32 | torch-cpu-eager-bf16 | torch-cpu-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| eps | 1e-06 | 1e-06 | 1e-06 | 1e-06 | 1e-06 |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 |

## Algorithm expansion

### adafactor / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.adafactor.synthetic.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 10744.3 | 10744.3..10744.3 | 1 | - | - | - | 2936.0 | 674.3 | rel_fro_vs_torch_eager_fp32=4.324e-05 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 5.8 | 5.8..5.8 | 1 | 1839.934 | - | - | 2717.2 | 960.0 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 102.6 | 102.6..102.6 | 1 | 104.757 | - | - | 2807.9 | 1024.0 | rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'beta2_decay': -0.8, 'd': 1.0, 'eps': [None, 0.001], 'lr': 0.001, 'maximize': False, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-fp32 | torch-eager-fp32 |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (optimizer.defaults) | torch (optimizer.defaults) |
| eps | [null, 0.001] | [null, 0.001] | [null, 0.001] |
| learning_rate | 0.001 | 0.001 | 0.001 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| weight_decay | 0.0 | 0.0 | 0.0 |

### adagrad / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.adagrad.synthetic.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 125.7 | 125.7..125.7 | 1 | - | - | - | 2935.7 | 674.3 | rel_fro_vs_torch_eager_fp32=3.207e-09 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 4.5 | 4.5..4.5 | 1 | 27.977 | - | - | 2621.6 | 960.0 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 51.2 | 51.2..51.2 | 1 | 2.455 | - | - | 3014.9 | 832.0 | rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'eps': 1e-10, 'initial_accumulator_value': 0.0, 'lr': 0.001, 'lr_decay': 0.0, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-fp32 | torch-eager-fp32 |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (optimizer.defaults) | torch (optimizer.defaults) |
| eps | 1e-10 | 1e-10 | 1e-10 |
| learning_rate | 0.001 | 0.001 | 0.001 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| weight_decay | 0.0 | 0.0 | 0.0 |

### adam / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.adam.synthetic.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 149.9 | 149.9..149.9 | 1 | - | - | - | 3000.0 | 674.3 | rel_fro_vs_torch_eager_fp32=3.34e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested identical, read back unknown) |
| torch-eager-fp32 | torch | gpu | opponent | 6.1 | 6.1..6.1 | 1 | - | - | - | 2623.2 | 960.0 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 79.3 | 79.3..79.3 | 1 | - | - | - | 3135.3 | 896.0 | rel_fro_vs_torch_eager_fp32=5.116e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'betas': [0.9, 0.999], 'eps': 1e-08, 'lr': 0.001, 'maximize': False, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-fp32 | torch-eager-fp32 |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (optimizer.defaults) | torch (optimizer.defaults) |
| amsgrad | - | false | false |
| betas | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| eps | 1e-08 | 1e-08 | 1e-08 |
| learning_rate | 0.001 | 0.001 | 0.001 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| weight_decay | 0.0 | 0.0 | 0.0 |

### adamax / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.adamax.synthetic.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 176.3 | 176.3..176.3 | 1 | - | - | - | 2999.8 | 674.3 | rel_fro_vs_torch_eager_fp32=4.144e-09 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 5.7 | 5.7..5.7 | 1 | 30.839 | - | - | 2631.7 | 960.0 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 218.6 | 218.6..218.6 | 1 | 0.806 | - | - | 3145.7 | 896.0 | rel_fro_vs_torch_eager_fp32=6.17e-09 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'betas': [0.9, 0.999], 'eps': 1e-08, 'lr': 0.001, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-fp32 | torch-eager-fp32 |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (optimizer.defaults) | torch (optimizer.defaults) |
| betas | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| eps | 1e-08 | 1e-08 | 1e-08 |
| learning_rate | 0.001 | 0.001 | 0.001 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| weight_decay | 0.0 | 0.0 | 0.0 |

### adamw / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.adamw.synthetic.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 150.2 | 150.2..150.2 | 1 | - | - | - | 3000.1 | 674.3 | rel_fro_vs_torch_eager_fp32=3.34e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested identical, read back unknown) |
| torch-eager-fp32 | torch | gpu | opponent | 6.7 | 6.7..6.7 | 1 | - | - | - | 2623.2 | 960.0 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 84.5 | 84.5..84.5 | 1 | - | - | - | 3134.2 | 896.0 | rel_fro_vs_torch_eager_fp32=2.272e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'betas': [0.9, 0.999], 'eps': 1e-08, 'lr': 0.001, 'maximize': False, 'weight_decay': 0.01}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-fp32 | torch-eager-fp32 |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (optimizer.defaults) | torch (optimizer.defaults) |
| amsgrad | - | false | false |
| betas | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| eps | 1e-08 | 1e-08 | 1e-08 |
| learning_rate | 0.001 | 0.001 | 0.001 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| weight_decay | 0.01 | 0.01 | 0.01 |

### avgpool1d / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.avgpool1d.synthetic.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | UNKNOWN(no binding path) | REFUSED(not_ready: {"error": "ImportError('libamdhip64.so.7: cannot open shared object file: No such file or directory')", "event": "error", "stage": "ready"}) |
| torch-eager-fp32 | torch | gpu | opponent | 0.8 | 0.8..0.8 | 1 | - | - | - | 1599.6 | 224.0 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 1.0 | 1.0..1.0 | 1 | - | - | - | 2050.1 | 224.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 0.8 | 0.8..0.8 | 1 | - | - | - | 1599.5 | 224.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 1.1 | 1.1..1.1 | 1 | - | - | - | 1965.6 | 224.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'ceil_mode': False, 'count_include_pad': True, 'kernel_size': 2, 'padding': 0, 'stride': 2}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---|
| library (source) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | 7 | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(not_ready: {"error": "ImportError('libamdhip64.so.7: cannot open shared object file: No such file or directory')", "event": "error", "stage": "ready"}) |
| torch-eager-fp32 | Xq | - | 0.1 | 0.1..0.1 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 0.1 | 0.1..0.1 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 0.1 | 0.1..0.1 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 0.1 | 0.1..0.1 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### avgpool2d / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.avgpool2d.synthetic.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | UNKNOWN(no binding path) | REFUSED(not_ready: {"error": "ImportError('libamdhip64.so.7: cannot open shared object file: No such file or directory')", "event": "error", "stage": "ready"}) |
| torch-eager-fp32 | torch | gpu | opponent | 0.9 | 0.9..0.9 | 1 | - | - | - | 1790.3 | 541.0 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 1.3 | 1.3..1.3 | 1 | - | - | - | 2202.2 | 541.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 1.0 | 1.0..1.0 | 1 | - | - | - | 1790.0 | 541.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 1.3 | 1.3..1.3 | 1 | - | - | - | 2121.6 | 541.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'ceil_mode': False, 'count_include_pad': True, 'divisor_override': None, 'kernel_size': 2, 'padding': 0, 'stride': 2}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---|
| library (source) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | 7 | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(not_ready: {"error": "ImportError('libamdhip64.so.7: cannot open shared object file: No such file or directory')", "event": "error", "stage": "ready"}) |
| torch-eager-fp32 | Xq | - | 0.1 | 0.1..0.1 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 0.2 | 0.2..0.2 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 0.1 | 0.1..0.1 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 0.2 | 0.2..0.2 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### batchnorm1d / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.batchnorm1d.synthetic.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | UNKNOWN(no binding path) | REFUSED(not_ready: {"error": "ImportError('libamdhip64.so.7: cannot open shared object file: No such file or directory')", "event": "error", "stage": "ready"}) |
| torch-eager-fp32 | torch | gpu | opponent | 0.9 | 0.9..0.9 | 1 | - | - | - | 2260.1 | 320.0 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 1.5 | 1.5..1.5 | 1 | - | - | - | 2114.8 | 320.0 | max_rel_diff_vs_torch_eager_fp32=0.002793, rel_fro_vs_torch_eager_fp32=5.248e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 1.0 | 1.0..1.0 | 1 | - | - | - | 1892.7 | 320.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 1.4 | 1.4..1.4 | 1 | - | - | - | 2008.9 | 320.0 | max_rel_diff_vs_torch_eager_fp32=0.002793, rel_fro_vs_torch_eager_fp32=5.248e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'affine': True, 'eps': 1e-05, 'momentum': 0.1, 'num_features': 256, 'track_running_stats': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---|
| library (source) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| eps | 1e-05 | 1e-05 | 1e-05 | 1e-05 |
| momentum | 0.1 | 0.1 | 0.1 | 0.1 |
| seed | 7 | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(not_ready: {"error": "ImportError('libamdhip64.so.7: cannot open shared object file: No such file or directory')", "event": "error", "stage": "ready"}) |
| torch-eager-fp32 | Xq | - | 0.1 | 0.1..0.1 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 0.2 | 0.2..0.2 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 0.1 | 0.1..0.1 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 0.2 | 0.2..0.2 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### batchnorm2d / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.batchnorm2d.synthetic.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | UNKNOWN(no binding path) | REFUSED(not_ready: {"error": "ImportError('libamdhip64.so.7: cannot open shared object file: No such file or directory')", "event": "error", "stage": "ready"}) |
| torch-eager-fp32 | torch | gpu | opponent | 1.0 | 1.0..1.0 | 1 | - | - | - | 2193.7 | 250.0 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 1.4 | 1.4..1.4 | 1 | - | - | - | 2083.4 | 250.0 | max_rel_diff_vs_torch_eager_fp32=0.001471, rel_fro_vs_torch_eager_fp32=6.868e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 0.9 | 0.9..0.9 | 1 | - | - | - | 1827.8 | 250.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 1.4 | 1.4..1.4 | 1 | - | - | - | 1977.6 | 250.0 | max_rel_diff_vs_torch_eager_fp32=0.001471, rel_fro_vs_torch_eager_fp32=6.868e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'affine': True, 'eps': 1e-05, 'momentum': 0.1, 'num_features': 64, 'track_running_stats': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---|
| library (source) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| eps | 1e-05 | 1e-05 | 1e-05 | 1e-05 |
| momentum | 0.1 | 0.1 | 0.1 | 0.1 |
| seed | 7 | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(not_ready: {"error": "ImportError('libamdhip64.so.7: cannot open shared object file: No such file or directory')", "event": "error", "stage": "ready"}) |
| torch-eager-fp32 | Xq | - | 0.2 | 0.2..0.2 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 0.2 | 0.2..0.2 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 0.2 | 0.2..0.2 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 0.2 | 0.2..0.2 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### cholesky / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.cholesky.synthetic.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 557.4 | 557.4..557.4 | 1 | - | - | - | 3905.4 | 3244.4 | relative_residual=2.9e-07 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| numpy-cpu | numpy | cpu | opponent | 7138.9 | 7138.9..7138.9 | 1 | 0.078 | - | - | 2650.2 | - | relative_residual=3.928e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-gpu | torch | gpu | opponent | 32.0 | 32.0..32.0 | 1 | 17.432 | - | - | 2616.0 | 790.4 | relative_residual=2.519e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'jitter': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | numpy-cpu | ours | torch-gpu |
|---|---||---|---||---|---|
| library (source) | numpy (declared) | mojolearn (declared) | torch (declared) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### clip-grad-norm / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.clip-grad-norm.synthetic.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 25.7 | 25.7..25.7 | 1 | - | - | - | 2171.1 | 674.3 | norm=1.000000, norm_rel_diff_vs_ours=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested identical, read back unknown) |
| torch-eager-fp32 | torch | gpu | opponent | 0.6 | 0.6..0.6 | 1 | - | - | - | 1884.3 | 128.0 | norm=1.000000, norm_rel_diff_vs_ours=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 92.3 | 92.3..92.3 | 1 | - | - | - | 2068.0 | 128.0 | norm=1.000000, norm_rel_diff_vs_ours=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'error_if_nonfinite': True, 'max_norm': 1.0, 'norm_type': 2.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-fp32 | torch-eager-fp32 |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | 7 | 7 |

### cnn-clf / synthetic (rows full, shape X 20000x1x28x28; Xq 5000x1x28x28; y 20000; yq 5000)

race: done, driver rc 0, log `logs/algos.cnn-clf.synthetic.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 179.2 | 179.2..179.2 | 1 | - | - | - | 806.9 | 1690.5 | accuracy=1.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 178.2 | 178.2..178.2 | 1 | 1.006 | - | - | 5759.4 | 423.5 | accuracy=1.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 210.1 | 210.1..210.1 | 1 | 0.853 | - | - | 5130.7 | 350.0 | accuracy=1.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 258.5 | 258.5..258.5 | 1 | 0.693 | - | - | 4287.9 | 431.6 | accuracy=1.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 287.6 | 287.6..287.6 | 1 | 0.623 | - | - | 3433.3 | 431.6 | accuracy=1.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 128, 'conv_channels': [8, 16], 'dampening': 0.0, 'input_shape': [1, 28, 28], 'kernel_size': 3, 'learning_rate': 0.01, 'max_iter': 2, 'momentum': 0.9, 'nesterov': False, 'optimizer': 'sgd', 'pool_size': 2, 'random_state': 7, 'shuffle': True, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (attributes) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| batch_size | 128 | 128 | 128 | 128 | 128 |
| betas | [0.9, 0.999] | - | - | - | - |
| dampening | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| eps | 1e-08 | - | - | - | - |
| learning_rate | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| max_iter | 2 | 2 | 2 | 2 | 2 |
| momentum | 0.9 | 0.9 | 0.9 | 0.9 | 0.9 |
| nesterov | false | false | false | false | false |
| seed | 7 | 7 | 7 | 7 | 7 |
| shuffle | true | true | true | true | true |
| weight_decay | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 6.4 | 6.4..6.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | 0.9 | 0.9..0.9 | 1 | 6.845 | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 0.8 | 0.8..0.8 | 1 | 7.585 | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 1.0 | 1.0..1.0 | 1 | 6.432 | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 0.8 | 0.8..0.8 | 1 | 7.955 | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: predict(Xq)(Xq)

inference call, torch-eager-fp32: predict(Xq)(Xq)

inference call, torch-compile-fp32: predict(Xq)(Xq)

inference call, torch-eager-bf16: predict(Xq)(Xq)

inference call, torch-compile-bf16: predict(Xq)(Xq)

### conv1d / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.conv1d.synthetic.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | UNKNOWN(no binding path) | REFUSED(not_ready: {"error": "ImportError('libamdhip64.so.7: cannot open shared object file: No such file or directory')", "event": "error", "stage": "ready"}) |
| torch-eager-fp32 | torch | gpu | opponent | 1.9 | 1.9..1.9 | 1 | - | - | - | 2747.7 | 896.6 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 2.4 | 2.4..2.4 | 1 | - | - | - | 2437.2 | 896.6 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 1.4 | 1.4..1.4 | 1 | - | - | - | 2994.4 | 832.4 | max_rel_diff_vs_torch_eager_fp32=3418.035805, rel_fro_vs_torch_eager_fp32=0.004377 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 2.0 | 2.0..2.0 | 1 | - | - | - | 2478.6 | 832.4 | max_rel_diff_vs_torch_eager_fp32=3418.035805, rel_fro_vs_torch_eager_fp32=0.004377 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'bias': True, 'dilation': 1, 'groups': 1, 'in_channels': 128, 'kernel_size': 3, 'out_channels': 128, 'padding': 1, 'padding_mode': 'zeros', 'stride': 1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---|
| library (source) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | 7 | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(not_ready: {"error": "ImportError('libamdhip64.so.7: cannot open shared object file: No such file or directory')", "event": "error", "stage": "ready"}) |
| torch-eager-fp32 | Xq | - | 0.7 | 0.7..0.7 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 0.7 | 0.7..0.7 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 0.4 | 0.4..0.4 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 0.4 | 0.4..0.4 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### conv2d / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.conv2d.synthetic.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | UNKNOWN(no binding path) | REFUSED(not_ready: {"error": "ImportError('libamdhip64.so.7: cannot open shared object file: No such file or directory')", "event": "error", "stage": "ready"}) |
| torch-eager-fp32 | torch | gpu | opponent | 1.3 | 1.3..1.3 | 1 | - | - | - | 2396.5 | 347.4 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 1.8 | 1.8..1.8 | 1 | - | - | - | 2278.0 | 347.4 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 1.4 | 1.4..1.4 | 1 | - | - | - | 2428.6 | 320.8 | max_rel_diff_vs_torch_eager_fp32=3418.486565, rel_fro_vs_torch_eager_fp32=0.003382 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 1.6 | 1.6..1.6 | 1 | - | - | - | 2329.5 | 320.8 | max_rel_diff_vs_torch_eager_fp32=3418.486565, rel_fro_vs_torch_eager_fp32=0.003382 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'bias': True, 'dilation': 1, 'groups': 1, 'in_channels': 64, 'kernel_size': 3, 'out_channels': 64, 'padding': 1, 'padding_mode': 'zeros', 'stride': 1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---|
| library (source) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | 7 | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(not_ready: {"error": "ImportError('libamdhip64.so.7: cannot open shared object file: No such file or directory')", "event": "error", "stage": "ready"}) |
| torch-eager-fp32 | Xq | - | 0.3 | 0.3..0.3 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 0.3 | 0.3..0.3 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 0.3 | 0.3..0.3 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 0.3 | 0.3..0.3 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### cross-entropy / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.cross-entropy.synthetic.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 216.7 | 216.7..216.7 | 1 | - | - | - | 2897.2 | 674.4 | loss_rel_err_vs_fp64=4.564e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested identical, read back unknown) |
| torch-eager-fp32 | torch | gpu | opponent | 8.4 | 8.4..8.4 | 1 | - | - | - | 2195.6 | 1024.1 | grad_max_rel_diff_vs_ours=4.191e-09, loss_rel_err_vs_fp64=4.564e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 84.4 | 84.4..84.4 | 1 | - | - | - | 2403.6 | 512.1 | grad_max_rel_diff_vs_ours=7.451e-09, loss_rel_err_vs_fp64=4.564e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'ignore_index': -100, 'label_smoothing': 0.0, 'reduction': 'mean'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-fp32 | torch-eager-fp32 |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | 7 | 7 |

### dropout2d / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.dropout2d.synthetic.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | UNKNOWN(no binding path) | REFUSED(not_ready: {"error": "ImportError('libamdhip64.so.7: cannot open shared object file: No such file or directory')", "event": "error", "stage": "ready"}) |
| torch-eager-fp32 | torch | gpu | opponent | 0.9 | 0.9..0.9 | 1 | - | - | - | 1596.4 | 250.0 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 97.2 | 97.2..97.2 | 1 | - | - | - | 1833.4 | 250.0 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'p': 0.1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-fp32 | torch-eager-fp32 |
|---|---||---|---|
| library (source) | torch (declared) | torch (declared) |
| p | 0.1 | 0.1 |
| seed | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(not_ready: {"error": "ImportError('libamdhip64.so.7: cannot open shared object file: No such file or directory')", "event": "error", "stage": "ready"}) |
| torch-eager-fp32 | Xq | - | 0.1 | 0.1..0.1 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 0.2 | 0.2..0.2 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

### eigh / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.eigh.synthetic.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(timeout: null) |
| numpy-cpu | numpy | cpu | opponent | 2356.5 | 2356.5..2356.5 | 1 | - | - | - | 970.0 | - | max_eigenvalue_error=3.49e-08, relative_residual=2.824e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-gpu | torch | gpu | opponent | 117.5 | 117.5..117.5 | 1 | - | - | - | 2037.8 | 400.7 | max_eigenvalue_error=1.044e-06, relative_residual=9.614e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'UPLO': 'L'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | numpy-cpu | ours | torch-gpu |
|---|---||---|---||---|---|
| library (source) | numpy (declared) | mojolearn (declared) | torch (declared) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### embedding / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.embedding.synthetic.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | UNKNOWN(no binding path) | REFUSED(not_ready: {"error": "ImportError('libamdhip64.so.7: cannot open shared object file: No such file or directory')", "event": "error", "stage": "ready"}) |
| torch-eager-fp32 | torch | gpu | opponent | 1.1 | 1.1..1.1 | 1 | - | - | - | 2290.0 | 782.6 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 1.1 | 1.1..1.1 | 1 | - | - | - | 2342.8 | 640.2 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'embedding_dim': 1024, 'max_norm': None, 'norm_type': 2.0, 'num_embeddings': 32768, 'padding_idx': None, 'scale_grad_by_freq': False, 'sparse': False}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-fp32 | torch-eager-fp32 |
|---|---||---|---|
| library (source) | torch (declared) | torch (declared) |
| seed | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(not_ready: {"error": "ImportError('libamdhip64.so.7: cannot open shared object file: No such file or directory')", "event": "error", "stage": "ready"}) |
| torch-eager-fp32 | Xq | - | 0.1 | 0.1..0.1 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 0.2 | 0.2..0.2 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

### gcn / istella (rows full, shape X 100000x220; indices 1521510; indptr 100001; y 100000)

race: done, driver rc 0, log `logs/algos.gcn.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | UNKNOWN(no binding path) | REFUSED(not_ready: {"error": "ImportError('libamdhip64.so.7: cannot open shared object file: No such file or directory')", "event": "error", "stage": "ready"}) |
| torch-eager-fp32 | torch | gpu | opponent | 4.8 | 4.8..4.8 | 1 | - | - | - | 4551.2 | 2070.3 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 3.9 | 3.9..3.9 | 1 | - | - | - | 4591.6 | 535.2 | max_rel_diff_vs_torch_eager_fp32=0.007740, rel_fro_vs_torch_eager_fp32=9.044e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 4.9 | 4.9..4.9 | 1 | - | - | - | 3211.8 | 2459.5 | max_rel_diff_vs_torch_eager_fp32=2866.979199, rel_fro_vs_torch_eager_fp32=0.002187 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 3.1 | 3.1..3.1 | 1 | - | - | - | 3528.1 | 918.2 | max_rel_diff_vs_torch_eager_fp32=2866.972319, rel_fro_vs_torch_eager_fp32=0.002187 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'add_self_loops': True, 'bias': True, 'improved': False, 'normalize': True, 'out_channels': 128}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---|
| library (source) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| normalize | true | true | true | true |
| seed | 7 | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(not_ready: {"error": "ImportError('libamdhip64.so.7: cannot open shared object file: No such file or directory')", "event": "error", "stage": "ready"}) |
| torch-eager-fp32 | Xq | - | 2.6 | 2.6..2.6 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 1.9 | 1.9..1.9 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 2.4 | 2.4..2.4 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 1.8 | 1.8..1.8 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### gcn / taxi (rows full, shape X 100000x11; indices 1258298; indptr 100001; y 100000)

race: done, driver rc 0, log `logs/algos.gcn.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | UNKNOWN(no binding path) | REFUSED(not_ready: {"error": "ImportError('libamdhip64.so.7: cannot open shared object file: No such file or directory')", "event": "error", "stage": "ready"}) |
| torch-eager-fp32 | torch | gpu | opponent | 4.2 | 4.2..4.2 | 1 | - | - | - | 4465.7 | 1725.4 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 3.4 | 3.4..3.4 | 1 | - | - | - | 5152.1 | 446.3 | max_rel_diff_vs_torch_eager_fp32=0.002794, rel_fro_vs_torch_eager_fp32=7.648e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 4.1 | 4.1..4.1 | 1 | - | - | - | 3125.5 | 2010.7 | max_rel_diff_vs_torch_eager_fp32=1509.504160, rel_fro_vs_torch_eager_fp32=0.002330 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 2.8 | 2.8..2.8 | 1 | - | - | - | 3444.2 | 726.3 | max_rel_diff_vs_torch_eager_fp32=1509.503229, rel_fro_vs_torch_eager_fp32=0.002330 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'add_self_loops': True, 'bias': True, 'improved': False, 'normalize': True, 'out_channels': 128}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---|
| library (source) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| normalize | true | true | true | true |
| seed | 7 | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(not_ready: {"error": "ImportError('libamdhip64.so.7: cannot open shared object file: No such file or directory')", "event": "error", "stage": "ready"}) |
| torch-eager-fp32 | Xq | - | 2.0 | 2.0..2.0 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 1.6 | 1.6..1.6 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 2.1 | 2.1..2.1 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 1.6 | 1.6..1.6 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### global-avgpool / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.global-avgpool.synthetic.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | UNKNOWN(no binding path) | REFUSED(not_ready: {"error": "ImportError('libamdhip64.so.7: cannot open shared object file: No such file or directory')", "event": "error", "stage": "ready"}) |
| torch-eager-fp32 | torch | gpu | opponent | 0.7 | 0.7..0.7 | 1 | - | - | - | 1634.8 | 12.6 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 1.0 | 1.0..1.0 | 1 | - | - | - | 1965.6 | 12.6 | max_rel_diff_vs_torch_eager_fp32=0.007299, rel_fro_vs_torch_eager_fp32=1.01e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 0.6 | 0.6..0.6 | 1 | - | - | - | 1634.6 | 12.6 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 1.0 | 1.0..1.0 | 1 | - | - | - | 1883.3 | 12.6 | max_rel_diff_vs_torch_eager_fp32=0.007299, rel_fro_vs_torch_eager_fp32=1.01e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'output_size': 1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---|
| library (source) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | 7 | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(not_ready: {"error": "ImportError('libamdhip64.so.7: cannot open shared object file: No such file or directory')", "event": "error", "stage": "ready"}) |
| torch-eager-fp32 | Xq | - | 0.0 | 0.0..0.0 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 0.1 | 0.1..0.1 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 0.1 | 0.1..0.1 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 0.1 | 0.1..0.1 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### global-maxpool / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.global-maxpool.synthetic.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | UNKNOWN(no binding path) | REFUSED(not_ready: {"error": "ImportError('libamdhip64.so.7: cannot open shared object file: No such file or directory')", "event": "error", "stage": "ready"}) |
| torch-eager-fp32 | torch | gpu | opponent | 0.6 | 0.6..0.6 | 1 | - | - | - | 1610.5 | 12.9 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 1.1 | 1.1..1.1 | 1 | - | - | - | 1993.0 | 12.9 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 0.6 | 0.6..0.6 | 1 | - | - | - | 1610.4 | 12.9 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 0.8 | 0.8..0.8 | 1 | - | - | - | 1902.0 | 12.9 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'output_size': 1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---|
| library (source) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | 7 | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(not_ready: {"error": "ImportError('libamdhip64.so.7: cannot open shared object file: No such file or directory')", "event": "error", "stage": "ready"}) |
| torch-eager-fp32 | Xq | - | 0.1 | 0.1..0.1 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 0.1 | 0.1..0.1 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 0.1 | 0.1..0.1 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 0.1 | 0.1..0.1 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### graphsage / istella (rows full, shape X 100000x220; indices 1521510; indptr 100001; y 100000)

race: done, driver rc 0, log `logs/algos.graphsage.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | UNKNOWN(no binding path) | REFUSED(not_ready: {"error": "ImportError('libamdhip64.so.7: cannot open shared object file: No such file or directory')", "event": "error", "stage": "ready"}) |
| torch-eager-fp32 | torch | gpu | opponent | 2.7 | 2.7..2.7 | 1 | - | - | - | 4278.1 | 1803.1 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 3.1 | 3.1..3.1 | 1 | - | - | - | 4493.5 | 539.5 | max_rel_diff_vs_torch_eager_fp32=0.108108, rel_fro_vs_torch_eager_fp32=8.065e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 2.6 | 2.6..2.6 | 1 | - | - | - | 2936.5 | 1778.7 | max_rel_diff_vs_torch_eager_fp32=3907.203674, rel_fro_vs_torch_eager_fp32=0.003366 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 3.0 | 3.0..3.0 | 1 | - | - | - | 3382.6 | 466.3 | max_rel_diff_vs_torch_eager_fp32=3678.128123, rel_fro_vs_torch_eager_fp32=0.003054 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'aggr': 'mean', 'bias': True, 'normalize': False, 'out_channels': 128, 'project': False, 'root_weight': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---|
| library (source) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| normalize | false | false | false | false |
| seed | 7 | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(not_ready: {"error": "ImportError('libamdhip64.so.7: cannot open shared object file: No such file or directory')", "event": "error", "stage": "ready"}) |
| torch-eager-fp32 | Xq | - | 2.6 | 2.6..2.6 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 2.6 | 2.6..2.6 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 2.6 | 2.6..2.6 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 2.6 | 2.6..2.6 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### graphsage / taxi (rows full, shape X 100000x11; indices 1258298; indptr 100001; y 100000)

race: done, driver rc 0, log `logs/algos.graphsage.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | UNKNOWN(no binding path) | REFUSED(not_ready: {"error": "ImportError('libamdhip64.so.7: cannot open shared object file: No such file or directory')", "event": "error", "stage": "ready"}) |
| torch-eager-fp32 | torch | gpu | opponent | 1.5 | 1.5..1.5 | 1 | - | - | - | 4191.6 | 424.5 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 1.7 | 1.7..1.7 | 1 | - | - | - | 4852.5 | 375.7 | max_rel_diff_vs_torch_eager_fp32=0.074506, rel_fro_vs_torch_eager_fp32=4.566e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 1.5 | 1.5..1.5 | 1 | - | - | - | 2850.0 | 327.9 | max_rel_diff_vs_torch_eager_fp32=5040.230769, rel_fro_vs_torch_eager_fp32=0.003557 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 2.0 | 2.0..2.0 | 1 | - | - | - | 3299.8 | 302.5 | max_rel_diff_vs_torch_eager_fp32=4607.677460, rel_fro_vs_torch_eager_fp32=0.003285 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'aggr': 'mean', 'bias': True, 'normalize': False, 'out_channels': 128, 'project': False, 'root_weight': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---|
| library (source) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| normalize | false | false | false | false |
| seed | 7 | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(not_ready: {"error": "ImportError('libamdhip64.so.7: cannot open shared object file: No such file or directory')", "event": "error", "stage": "ready"}) |
| torch-eager-fp32 | Xq | - | 0.4 | 0.4..0.4 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 0.4 | 0.4..0.4 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 0.4 | 0.4..0.4 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 0.4 | 0.4..0.4 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### gru-clf / synthetic (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc 0, log `logs/algos.gru-clf.synthetic.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 918.6 | 918.6..918.6 | 1 | - | - | - | 2056.4 | 678.5 | accuracy=0.971842, logloss=0.065835 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 1740.4 | 1740.4..1740.4 | 1 | 0.528 | - | - | 4839.2 | 476.7 | accuracy=0.971842 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 1795.5 | 1795.5..1795.5 | 1 | 0.512 | - | - | 4886.9 | 476.7 | accuracy=0.971842 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 1839.7 | 1839.7..1839.7 | 1 | 0.499 | - | - | 3216.3 | 320.7 | accuracy=0.971680 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 1794.7 | 1794.7..1794.7 | 1 | 0.512 | - | - | 3265.1 | 320.7 | accuracy=0.971680 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (attributes) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| batch_size | 256 | 256 | 256 | 256 | 256 |
| betas | - | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| eps | - | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| hidden_size | 64 | 64 | 64 | 64 | 64 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| num_layers | 1 | 1 | 1 | 1 | 1 |
| seed | 7 | 7 | 7 | 7 | 7 |
| shuffle | true | true | true | true | true |
| weight_decay | - | 0.0 | 0.0 | 0.0 | 0.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 12.9 | 12.9..12.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | 6.2 | 6.2..6.2 | 1 | 2.085 | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 6.2 | 6.2..6.2 | 1 | 2.094 | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 6.1 | 6.1..6.1 | 1 | 2.108 | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 6.2 | 6.2..6.2 | 1 | 2.095 | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: predict(Xq)(Xq)

inference call, torch-eager-fp32: predict(Xq)(Xq)

inference call, torch-compile-fp32: predict(Xq)(Xq)

inference call, torch-eager-bf16: predict(Xq)(Xq)

inference call, torch-compile-bf16: predict(Xq)(Xq)

### gru-clf / taxi-hourly (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc 0, log `logs/algos.gru-clf.taxi-hourly.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 919.2 | 919.2..919.2 | 1 | - | - | - | 2056.8 | 678.5 | accuracy=0.865668, logloss=0.305841 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 1692.5 | 1692.5..1692.5 | 1 | 0.543 | - | - | 5229.5 | 476.7 | accuracy=0.865668 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 1798.4 | 1798.4..1798.4 | 1 | 0.511 | - | - | 4887.1 | 476.7 | accuracy=0.865668 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 1892.1 | 1892.1..1892.1 | 1 | 0.486 | - | - | 3667.0 | 320.7 | accuracy=0.865777 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 1898.9 | 1898.9..1898.9 | 1 | 0.484 | - | - | 3266.8 | 320.7 | accuracy=0.865777 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (attributes) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| batch_size | 256 | 256 | 256 | 256 | 256 |
| betas | - | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| eps | - | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| hidden_size | 64 | 64 | 64 | 64 | 64 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| num_layers | 1 | 1 | 1 | 1 | 1 |
| seed | 7 | 7 | 7 | 7 | 7 |
| shuffle | true | true | true | true | true |
| weight_decay | - | 0.0 | 0.0 | 0.0 | 0.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 12.9 | 12.9..12.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | 6.0 | 6.0..6.0 | 1 | 2.145 | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 6.2 | 6.2..6.2 | 1 | 2.084 | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 6.1 | 6.1..6.1 | 1 | 2.123 | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 6.3 | 6.3..6.3 | 1 | 2.040 | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: predict(Xq)(Xq)

inference call, torch-eager-fp32: predict(Xq)(Xq)

inference call, torch-compile-fp32: predict(Xq)(Xq)

inference call, torch-eager-bf16: predict(Xq)(Xq)

inference call, torch-compile-bf16: predict(Xq)(Xq)

### gru-reg / synthetic (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc 0, log `logs/algos.gru-reg.synthetic.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 916.1 | 916.1..916.1 | 1 | - | - | - | 2055.4 | 678.4 | finite=True, r2=0.981946, rmse=0.155672 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 1801.3 | 1801.3..1801.3 | 1 | 0.509 | - | - | 2921.5 | 476.4 | finite=True, r2=0.981946, rmse=0.155672 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 1751.3 | 1751.3..1751.3 | 1 | 0.523 | - | - | 2969.2 | 476.4 | finite=True, r2=0.981946, rmse=0.155672 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 1893.3 | 1893.3..1893.3 | 1 | 0.484 | - | - | 2650.5 | 320.4 | finite=True, r2=0.981981, rmse=0.155521 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 1727.4 | 1727.4..1727.4 | 1 | 0.530 | - | - | 2698.5 | 320.4 | finite=True, r2=0.981981, rmse=0.155521 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (attributes) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| batch_size | 256 | 256 | 256 | 256 | 256 |
| betas | - | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| eps | - | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| hidden_size | 64 | 64 | 64 | 64 | 64 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| num_layers | 1 | 1 | 1 | 1 | 1 |
| seed | 7 | 7 | 7 | 7 | 7 |
| shuffle | true | true | true | true | true |
| weight_decay | - | 0.0 | 0.0 | 0.0 | 0.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 6.5 | 6.5..6.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | 6.1 | 6.1..6.1 | 1 | 1.058 | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 6.2 | 6.2..6.2 | 1 | 1.046 | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 6.1 | 6.1..6.1 | 1 | 1.066 | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 6.0 | 6.0..6.0 | 1 | 1.071 | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: predict(Xq)(Xq)

inference call, torch-eager-fp32: predict(Xq)(Xq)

inference call, torch-compile-fp32: predict(Xq)(Xq)

inference call, torch-eager-bf16: predict(Xq)(Xq)

inference call, torch-compile-bf16: predict(Xq)(Xq)

### gru-reg / taxi-hourly (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc 0, log `logs/algos.gru-reg.taxi-hourly.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 917.0 | 917.0..917.0 | 1 | - | - | - | 2054.7 | 678.4 | finite=True, r2=0.748219, rmse=0.544182 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 1777.8 | 1777.8..1777.8 | 1 | 0.516 | - | - | 2921.5 | 476.4 | finite=True, r2=0.748219, rmse=0.544182 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 1724.0 | 1724.0..1724.0 | 1 | 0.532 | - | - | 2969.4 | 476.4 | finite=True, r2=0.748219, rmse=0.544182 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 1881.1 | 1881.1..1881.1 | 1 | 0.487 | - | - | 2650.5 | 320.4 | finite=True, r2=0.748253, rmse=0.544145 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 1727.7 | 1727.7..1727.7 | 1 | 0.531 | - | - | 2698.4 | 320.4 | finite=True, r2=0.748253, rmse=0.544145 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (attributes) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| batch_size | 256 | 256 | 256 | 256 | 256 |
| betas | - | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| eps | - | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| hidden_size | 64 | 64 | 64 | 64 | 64 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| num_layers | 1 | 1 | 1 | 1 | 1 |
| seed | 7 | 7 | 7 | 7 | 7 |
| shuffle | true | true | true | true | true |
| weight_decay | - | 0.0 | 0.0 | 0.0 | 0.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 6.6 | 6.6..6.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | 6.2 | 6.2..6.2 | 1 | 1.066 | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 6.0 | 6.0..6.0 | 1 | 1.086 | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 6.1 | 6.1..6.1 | 1 | 1.079 | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 6.1 | 6.1..6.1 | 1 | 1.082 | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: predict(Xq)(Xq)

inference call, torch-eager-fp32: predict(Xq)(Xq)

inference call, torch-compile-fp32: predict(Xq)(Xq)

inference call, torch-eager-bf16: predict(Xq)(Xq)

inference call, torch-compile-bf16: predict(Xq)(Xq)

### layernorm / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.layernorm.synthetic.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | UNKNOWN(no binding path) | REFUSED(not_ready: {"error": "ImportError('libamdhip64.so.7: cannot open shared object file: No such file or directory')", "event": "error", "stage": "ready"}) |
| torch-eager-fp32 | torch | gpu | opponent | 0.9 | 0.9..0.9 | 1 | - | - | - | 1640.7 | 320.1 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 1.4 | 1.4..1.4 | 1 | - | - | - | 2084.3 | 392.1 | max_rel_diff_vs_torch_eager_fp32=0.007530, rel_fro_vs_torch_eager_fp32=5.326e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 0.9 | 0.9..0.9 | 1 | - | - | - | 1640.3 | 320.1 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 1.2 | 1.2..1.2 | 1 | - | - | - | 2011.5 | 392.1 | max_rel_diff_vs_torch_eager_fp32=0.007530, rel_fro_vs_torch_eager_fp32=5.326e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'bias': True, 'elementwise_affine': True, 'eps': 1e-05, 'normalized_shape': 1024}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---|
| library (source) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| eps | 1e-05 | 1e-05 | 1e-05 | 1e-05 |
| seed | 7 | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(not_ready: {"error": "ImportError('libamdhip64.so.7: cannot open shared object file: No such file or directory')", "event": "error", "stage": "ready"}) |
| torch-eager-fp32 | Xq | - | 0.1 | 0.1..0.1 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 0.1 | 0.1..0.1 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 0.1 | 0.1..0.1 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 0.1 | 0.1..0.1 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### lr-constant / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.lr-constant.synthetic.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 632.5 | 632.5..632.5 | 1 | - | - | - | 86.7 | - | max_rel_diff_vs_ours=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested identical, read back unknown) |
| torch-cpu | torch | cpu | opponent | 102.8 | 102.8..102.8 | 1 | - | - | - | 1228.7 | - | max_rel_diff_vs_ours=1.038e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids lists no VRAM for this pid

memory, torch-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'peak_lr': 0.001, 'warmup_steps': 1000}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-cpu |
|---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) |
| seed | "none (deterministic)" | "none (deterministic)" |

### lr-exponential / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.lr-exponential.synthetic.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 227.0 | 227.0..227.0 | 1 | - | - | - | 86.6 | - | max_rel_diff_vs_ours=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-cpu | torch | cpu | opponent | 93.5 | 93.5..93.5 | 1 | 2.428 | - | - | 1228.8 | - | max_rel_diff_vs_ours=5.933e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids lists no VRAM for this pid

memory, torch-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'base_lr': 0.1, 'gamma': 0.9999}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-cpu |
|---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) |
| gamma | 0.9999 | 0.9999 |
| seed | "none (deterministic)" | "none (deterministic)" |

### lr-onecycle / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.lr-onecycle.synthetic.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1276.3 | 1276.3..1276.3 | 1 | - | - | - | 86.8 | - | max_rel_diff_vs_ours=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-cpu | torch | cpu | opponent | 135.7 | 135.7..135.7 | 1 | 9.409 | - | - | 1228.7 | - | max_rel_diff_vs_ours=5.951e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids lists no VRAM for this pid

memory, torch-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'anneal_strategy': 'cos', 'div_factor': 25.0, 'final_div_factor': 10000.0, 'max_lr': 0.1, 'pct_start': 0.3, 'three_phase': False, 'total_steps': 100000}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-cpu |
|---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) |
| seed | "none (deterministic)" | "none (deterministic)" |

### lr-step / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.lr-step.synthetic.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 11.9 | 11.9..11.9 | 1 | - | - | - | 83.9 | - | max_rel_diff_vs_ours=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-cpu | torch | cpu | opponent | 103.4 | 103.4..103.4 | 1 | 0.115 | - | - | 1228.9 | - | max_rel_diff_vs_ours=1.49e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids lists no VRAM for this pid

memory, torch-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'base_lr': 0.1, 'gamma': 0.5, 'step_size': 1000}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-cpu |
|---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) |
| gamma | 0.5 | 0.5 |
| seed | "none (deterministic)" | "none (deterministic)" |

### lr-warmup-cosine / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.lr-warmup-cosine.synthetic.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 332516.9 | 332516.9..332516.9 | 1 | - | - | - | 93.0 | - | max_rel_diff_vs_ours=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested identical, read back unknown) |
| torch-cpu | torch | cpu | opponent | 127.3 | 127.3..127.3 | 1 | - | - | - | 1228.7 | - | max_rel_diff_vs_ours=0.0001571 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids lists no VRAM for this pid

memory, torch-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'min_lr': 1e-05, 'peak_lr': 0.001, 'total_steps': 100000, 'warmup_steps': 1000}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-cpu |
|---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) |
| seed | "none (deterministic)" | "none (deterministic)" |

### lr-warmup-linear / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.lr-warmup-linear.synthetic.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 893.8 | 893.8..893.8 | 1 | - | - | - | 86.8 | - | max_rel_diff_vs_ours=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested identical, read back unknown) |
| torch-cpu | torch | cpu | opponent | 115.6 | 115.6..115.6 | 1 | - | - | - | 1228.7 | - | max_rel_diff_vs_ours=0.001000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids lists no VRAM for this pid

memory, torch-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'min_lr': 1e-05, 'peak_lr': 0.001, 'total_steps': 100000, 'warmup_steps': 1000}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-cpu |
|---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) |
| seed | "none (deterministic)" | "none (deterministic)" |

### lstm-clf / synthetic (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc 0, log `logs/algos.lstm-clf.synthetic.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1028.0 | 1028.0..1028.0 | 1 | - | - | - | 2056.4 | 678.5 | accuracy=0.968696, logloss=0.072441 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 901.1 | 901.1..901.1 | 1 | 1.141 | - | - | 4838.6 | 502.8 | accuracy=0.968696 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 863.4 | 863.4..863.4 | 1 | 1.191 | - | - | 4886.7 | 502.8 | accuracy=0.968696 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 987.5 | 987.5..987.5 | 1 | 1.041 | - | - | 3217.8 | 334.8 | accuracy=0.968913 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 936.9 | 936.9..936.9 | 1 | 1.097 | - | - | 3266.1 | 334.8 | accuracy=0.968913 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (attributes) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| batch_size | 256 | 256 | 256 | 256 | 256 |
| betas | - | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| eps | - | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| hidden_size | 64 | 64 | 64 | 64 | 64 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| num_layers | 1 | 1 | 1 | 1 | 1 |
| seed | 7 | 7 | 7 | 7 | 7 |
| shuffle | true | true | true | true | true |
| weight_decay | - | 0.0 | 0.0 | 0.0 | 0.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 16.2 | 16.2..16.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | 3.3 | 3.3..3.3 | 1 | 4.946 | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 3.2 | 3.2..3.2 | 1 | 5.090 | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 3.0 | 3.0..3.0 | 1 | 5.439 | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 3.1 | 3.1..3.1 | 1 | 5.182 | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: predict(Xq)(Xq)

inference call, torch-eager-fp32: predict(Xq)(Xq)

inference call, torch-compile-fp32: predict(Xq)(Xq)

inference call, torch-eager-bf16: predict(Xq)(Xq)

inference call, torch-compile-bf16: predict(Xq)(Xq)

### lstm-clf / taxi-hourly (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc 0, log `logs/algos.lstm-clf.taxi-hourly.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1027.4 | 1027.4..1027.4 | 1 | - | - | - | 2056.9 | 678.5 | accuracy=0.868218, logloss=0.299901 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 887.0 | 887.0..887.0 | 1 | 1.158 | - | - | 5225.7 | 502.8 | accuracy=0.868218 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 881.9 | 881.9..881.9 | 1 | 1.165 | - | - | 4886.4 | 502.8 | accuracy=0.868218 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 976.8 | 976.8..976.8 | 1 | 1.052 | - | - | 3685.3 | 334.8 | accuracy=0.868056 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 962.8 | 962.8..962.8 | 1 | 1.067 | - | - | 3266.3 | 334.8 | accuracy=0.868056 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (attributes) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| batch_size | 256 | 256 | 256 | 256 | 256 |
| betas | - | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| eps | - | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| hidden_size | 64 | 64 | 64 | 64 | 64 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| num_layers | 1 | 1 | 1 | 1 | 1 |
| seed | 7 | 7 | 7 | 7 | 7 |
| shuffle | true | true | true | true | true |
| weight_decay | - | 0.0 | 0.0 | 0.0 | 0.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 16.0 | 16.0..16.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | 3.3 | 3.3..3.3 | 1 | 4.889 | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 3.2 | 3.2..3.2 | 1 | 4.942 | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 3.0 | 3.0..3.0 | 1 | 5.340 | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 3.0 | 3.0..3.0 | 1 | 5.309 | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: predict(Xq)(Xq)

inference call, torch-eager-fp32: predict(Xq)(Xq)

inference call, torch-compile-fp32: predict(Xq)(Xq)

inference call, torch-eager-bf16: predict(Xq)(Xq)

inference call, torch-compile-bf16: predict(Xq)(Xq)

### lstm-reg / synthetic (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc 0, log `logs/algos.lstm-reg.synthetic.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1027.2 | 1027.2..1027.2 | 1 | - | - | - | 2055.1 | 678.4 | finite=True, r2=0.981013, rmse=0.159641 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 874.9 | 874.9..874.9 | 1 | 1.174 | - | - | 2920.6 | 502.5 | finite=True, r2=0.981013, rmse=0.159642 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 886.2 | 886.2..886.2 | 1 | 1.159 | - | - | 2968.4 | 502.5 | finite=True, r2=0.981013, rmse=0.159642 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 993.6 | 993.6..993.6 | 1 | 1.034 | - | - | 2649.7 | 334.5 | finite=True, r2=0.981004, rmse=0.159679 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 1007.1 | 1007.1..1007.1 | 1 | 1.020 | - | - | 2697.4 | 334.5 | finite=True, r2=0.981004, rmse=0.159679 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (attributes) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| batch_size | 256 | 256 | 256 | 256 | 256 |
| betas | - | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| eps | - | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| hidden_size | 64 | 64 | 64 | 64 | 64 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| num_layers | 1 | 1 | 1 | 1 | 1 |
| seed | 7 | 7 | 7 | 7 | 7 |
| shuffle | true | true | true | true | true |
| weight_decay | - | 0.0 | 0.0 | 0.0 | 0.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 8.2 | 8.2..8.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | 3.1 | 3.1..3.1 | 1 | 2.687 | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 3.1 | 3.1..3.1 | 1 | 2.677 | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 3.0 | 3.0..3.0 | 1 | 2.710 | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 3.1 | 3.1..3.1 | 1 | 2.626 | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: predict(Xq)(Xq)

inference call, torch-eager-fp32: predict(Xq)(Xq)

inference call, torch-compile-fp32: predict(Xq)(Xq)

inference call, torch-eager-bf16: predict(Xq)(Xq)

inference call, torch-compile-bf16: predict(Xq)(Xq)

### lstm-reg / taxi-hourly (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc 0, log `logs/algos.lstm-reg.taxi-hourly.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1026.7 | 1026.7..1026.7 | 1 | - | - | - | 2055.2 | 678.4 | finite=True, r2=0.751679, rmse=0.540429 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 849.2 | 849.2..849.2 | 1 | 1.209 | - | - | 2930.9 | 502.5 | finite=True, r2=0.751679, rmse=0.540429 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 911.8 | 911.8..911.8 | 1 | 1.126 | - | - | 2965.2 | 502.5 | finite=True, r2=0.751679, rmse=0.540429 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 948.1 | 948.1..948.1 | 1 | 1.083 | - | - | 2649.5 | 334.5 | finite=True, r2=0.751591, rmse=0.540526 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 954.6 | 954.6..954.6 | 1 | 1.076 | - | - | 2697.5 | 334.5 | finite=True, r2=0.751591, rmse=0.540526 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (attributes) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| batch_size | 256 | 256 | 256 | 256 | 256 |
| betas | - | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| eps | - | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| hidden_size | 64 | 64 | 64 | 64 | 64 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| num_layers | 1 | 1 | 1 | 1 | 1 |
| seed | 7 | 7 | 7 | 7 | 7 |
| shuffle | true | true | true | true | true |
| weight_decay | - | 0.0 | 0.0 | 0.0 | 0.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 8.4 | 8.4..8.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | 3.2 | 3.2..3.2 | 1 | 2.588 | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 3.2 | 3.2..3.2 | 1 | 2.586 | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 3.1 | 3.1..3.1 | 1 | 2.734 | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 3.1 | 3.1..3.1 | 1 | 2.674 | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: predict(Xq)(Xq)

inference call, torch-eager-fp32: predict(Xq)(Xq)

inference call, torch-compile-fp32: predict(Xq)(Xq)

inference call, torch-eager-bf16: predict(Xq)(Xq)

inference call, torch-compile-bf16: predict(Xq)(Xq)

### lstsq / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.lstsq.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 20562.5 | 20562.5..20562.5 | 1 | - | - | - | 3808.3 | 676.4 | relative_residual=0.849957 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| numpy-cpu | numpy | cpu | opponent | 2526.8 | 2526.8..2526.8 | 1 | 8.138 | - | - | 4351.2 | - | relative_residual=0.876106 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-gpu | torch | gpu | opponent | 3342.1 | 3342.1..3342.1 | 1 | 6.153 | - | - | 2660.7 | 1819.9 | relative_residual=nan | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | numpy-cpu | ours | torch-gpu |
|---|---||---|---||---|---|
| library (source) | numpy (declared) | mojolearn (declared) | torch (declared) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### lstsq / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.lstsq.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 51.9 | 51.9..51.9 | 1 | - | - | - | 2133.7 | 878.4 | relative_residual=0.756366 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| numpy-cpu | numpy | cpu | opponent | 91.5 | 91.5..91.5 | 1 | 0.568 | - | - | 284.0 | - | relative_residual=0.756366 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-gpu | torch | gpu | opponent | 27.5 | 27.5..27.5 | 1 | 1.889 | - | - | 1707.2 | 223.5 | relative_residual=0.756366 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | numpy-cpu | ours | torch-gpu |
|---|---||---|---||---|---|
| library (source) | numpy (declared) | mojolearn (declared) | torch (declared) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### lu-factor / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.lu-factor.synthetic.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 596834.2 | 596834.2..596834.2 | 1 | - | - | - | 2274.8 | 1194.4 | relative_residual=3.249e-06 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| scipy-cpu | scipy | cpu | opponent | 4553.9 | 4553.9..4553.9 | 1 | 131.059 | - | - | 628.0 | - | relative_residual=3.439e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-gpu | torch | gpu | opponent | 85.0 | 85.0..85.0 | 1 | 7018.998 | - | - | 2080.1 | 710.0 | relative_residual=4.003e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, scipy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | scipy-cpu | torch-gpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | scipy (declared) | torch (declared) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### lu-solve / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.lu-solve.synthetic.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 615982.2 | 615982.2..615982.2 | 1 | - | - | - | 2274.7 | 1194.4 | relative_residual=3.249e-06 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| numpy-cpu | numpy | cpu | opponent | 5413.8 | 5413.8..5413.8 | 1 | 113.781 | - | - | 1389.2 | - | relative_residual=3.259e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-gpu | torch | gpu | opponent | 82.7 | 82.7..82.7 | 1 | 7446.797 | - | - | 2104.8 | 712.0 | relative_residual=4.003e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | numpy-cpu | ours | torch-gpu |
|---|---||---|---||---|---|
| library (source) | numpy (declared) | mojolearn (declared) | torch (declared) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### maxpool1d / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.maxpool1d.synthetic.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | UNKNOWN(no binding path) | REFUSED(not_ready: {"error": "ImportError('libamdhip64.so.7: cannot open shared object file: No such file or directory')", "event": "error", "stage": "ready"}) |
| torch-eager-fp32 | torch | gpu | opponent | 0.8 | 0.8..0.8 | 1 | - | - | - | 1609.5 | 288.0 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 1.4 | 1.4..1.4 | 1 | - | - | - | 2138.9 | 232.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 0.8 | 0.8..0.8 | 1 | - | - | - | 1610.8 | 288.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 1.2 | 1.2..1.2 | 1 | - | - | - | 1976.3 | 232.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'ceil_mode': False, 'dilation': 1, 'kernel_size': 2, 'padding': 0, 'stride': 2}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---|
| library (source) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | 7 | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(not_ready: {"error": "ImportError('libamdhip64.so.7: cannot open shared object file: No such file or directory')", "event": "error", "stage": "ready"}) |
| torch-eager-fp32 | Xq | - | 0.1 | 0.1..0.1 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 0.1 | 0.1..0.1 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 0.1 | 0.1..0.1 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 0.1 | 0.1..0.1 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### maxpool2d / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.maxpool2d.synthetic.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | UNKNOWN(no binding path) | REFUSED(not_ready: {"error": "ImportError('libamdhip64.so.7: cannot open shared object file: No such file or directory')", "event": "error", "stage": "ready"}) |
| torch-eager-fp32 | torch | gpu | opponent | 0.7 | 0.7..0.7 | 1 | - | - | - | 1801.1 | 639.0 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 1.0 | 1.0..1.0 | 1 | - | - | - | 2484.0 | 553.5 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 1.0 | 1.0..1.0 | 1 | - | - | - | 1801.3 | 639.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 1.3 | 1.3..1.3 | 1 | - | - | - | 2127.4 | 553.5 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'ceil_mode': False, 'dilation': 1, 'kernel_size': 2, 'padding': 0, 'stride': 2}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---|
| library (source) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | 7 | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(not_ready: {"error": "ImportError('libamdhip64.so.7: cannot open shared object file: No such file or directory')", "event": "error", "stage": "ready"}) |
| torch-eager-fp32 | Xq | - | 0.1 | 0.1..0.1 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 0.1 | 0.1..0.1 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 0.1 | 0.1..0.1 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 0.2 | 0.2..0.2 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### moe / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.moe.synthetic.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | UNKNOWN(no binding path) | REFUSED(not_ready: {"error": "ImportError('libamdhip64.so.7: cannot open shared object file: No such file or directory')", "event": "error", "stage": "ready"}) |
| torch-eager-fp32 | torch | gpu | opponent | 4.5 | 4.5..4.5 | 1 | - | - | - | 3404.0 | 537.0 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 5.0 | 5.0..5.0 | 1 | - | - | - | 3676.5 | 514.6 | max_rel_diff_vs_torch_eager_fp32=0.017192, rel_fro_vs_torch_eager_fp32=1.088e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 2.7 | 2.7..2.7 | 1 | - | - | - | 2792.9 | 641.3 | max_rel_diff_vs_torch_eager_fp32=22778.779951, rel_fro_vs_torch_eager_fp32=0.055222 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 3.1 | 3.1..3.1 | 1 | - | - | - | 3061.3 | 501.4 | max_rel_diff_vs_torch_eager_fp32=22796.112469, rel_fro_vs_torch_eager_fp32=0.055199 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'hidden_size': 1024, 'intermediate_size': 2816, 'norm_topk_prob': True, 'num_experts': 8, 'top_k': 2}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---|
| library (source) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| hidden_size | 1024 | 1024 | 1024 | 1024 |
| seed | 7 | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(not_ready: {"error": "ImportError('libamdhip64.so.7: cannot open shared object file: No such file or directory')", "event": "error", "stage": "ready"}) |
| torch-eager-fp32 | Xq | - | 3.9 | 3.9..3.9 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 3.9 | 3.9..3.9 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 2.1 | 2.1..2.1 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 2.1 | 2.1..2.1 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### nadam / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.nadam.synthetic.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 177.6 | 177.6..177.6 | 1 | - | - | - | 2999.6 | 674.3 | rel_fro_vs_torch_eager_fp32=2.945e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 7.0 | 7.0..7.0 | 1 | 25.230 | - | - | 2624.0 | 960.0 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 91.7 | 91.7..91.7 | 1 | 1.937 | - | - | 3136.2 | 896.0 | rel_fro_vs_torch_eager_fp32=1.464e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'betas': [0.9, 0.999], 'decoupled_weight_decay': False, 'eps': 1e-08, 'lr': 0.001, 'momentum_decay': 0.004, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-fp32 | torch-eager-fp32 |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (optimizer.defaults) | torch (optimizer.defaults) |
| betas | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| eps | 1e-08 | 1e-08 | 1e-08 |
| learning_rate | 0.001 | 0.001 | 0.001 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| weight_decay | 0.0 | 0.0 | 0.0 |

### qr / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.qr.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 76870.3 | 76870.3..76870.3 | 1 | - | - | - | 5850.5 | 674.3 | relative_gram_difference=0.0009027 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| numpy-cpu | numpy | cpu | opponent | 4646.6 | 4646.6..4646.6 | 1 | 16.543 | - | - | 8532.1 | - | relative_gram_difference=2.462e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-gpu | torch | gpu | opponent | 2664.9 | 2664.9..2664.9 | 1 | 28.845 | - | - | 2740.9 | 2520.9 | relative_gram_difference=0.0001695 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'mode': 'reduced'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | numpy-cpu | ours | torch-gpu |
|---|---||---|---||---|---|
| library (source) | numpy (declared) | mojolearn (declared) | torch (declared) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### qr / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.qr.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3985.2 | 3985.2..3985.2 | 1 | - | - | - | 1071.1 | 674.3 | relative_gram_difference=0.001996 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| numpy-cpu | numpy | cpu | opponent | 177.3 | 177.3..177.3 | 1 | 22.477 | - | - | 478.4 | - | relative_gram_difference=3.024e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-gpu | torch | gpu | opponent | 27.0 | 27.0..27.0 | 1 | 147.858 | - | - | 1594.3 | 126.0 | relative_gram_difference=9.54e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'mode': 'reduced'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | numpy-cpu | ours | torch-gpu |
|---|---||---|---||---|---|
| library (source) | numpy (declared) | mojolearn (declared) | torch (declared) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### randomized-svd / istella (rows full, shape X 900000x220; Xq 100000x220)

race: done, driver rc 0, log `logs/algos.randomized-svd.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 840.8 | 840.8..840.8 | 1 | - | - | - | 3749.5 | 738.5 | relative_reconstruction_error=0.0002359 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 812.6 | 812.6..812.6 | 1 | 1.035 | - | - | 1710.8 | - | relative_reconstruction_error=0.0002359 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-gpu | torch | gpu | opponent | 206.8 | 206.8..206.8 | 1 | 4.065 | - | - | 4516.2 | 1018.1 | relative_reconstruction_error=0.0002359 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'n_components': 8, 'n_iter': 4, 'n_oversamples': 10, 'random_state': 7}. Rows: None. Timed: None.

mismatch: torch-gpu is torch.svd_lowrank(q=18, niter=4), its randomized range finder

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu | torch-gpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | sklearn (declared) | torch (declared) |
| n_components | 8 | 8 | 8 |
| n_iter | 4 | 4 | 4 |
| seed | 7 | 7 | 7 |

### randomized-svd / taxi (rows full, shape X 900000x11; Xq 100000x11)

race: done, driver rc 0, log `logs/algos.randomized-svd.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 249.7 | 249.7..249.7 | 1 | - | - | - | 2255.1 | 934.5 | relative_reconstruction_error=0.027197 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 375.4 | 375.4..375.4 | 1 | 0.665 | - | - | 476.9 | - | relative_reconstruction_error=0.027197 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-gpu | torch | gpu | opponent | 120.8 | 120.8..120.8 | 1 | 2.067 | - | - | 3715.1 | 228.0 | relative_reconstruction_error=0.027197 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'n_components': 8, 'n_iter': 4, 'n_oversamples': 10, 'random_state': 7}. Rows: None. Timed: None.

mismatch: torch-gpu is torch.svd_lowrank(q=18, niter=4), its randomized range finder

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu | torch-gpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | sklearn (declared) | torch (declared) |
| n_components | 8 | 8 | 8 |
| n_iter | 4 | 4 | 4 |
| seed | 7 | 7 | 7 |

### resnet-block / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.resnet-block.synthetic.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | UNKNOWN(no binding path) | REFUSED(not_ready: {"error": "ImportError('libamdhip64.so.7: cannot open shared object file: No such file or directory')", "event": "error", "stage": "ready"}) |
| torch-eager-fp32 | torch | gpu | opponent | 2.0 | 2.0..2.0 | 1 | - | - | - | 2322.0 | 547.6 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 2.1 | 2.1..2.1 | 1 | - | - | - | 2382.8 | 510.1 | max_rel_diff_vs_torch_eager_fp32=0.357628, rel_fro_vs_torch_eager_fp32=1.809e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 2.0 | 2.0..2.0 | 1 | - | - | - | 2472.7 | 471.0 | max_rel_diff_vs_torch_eager_fp32=24577.140808, rel_fro_vs_torch_eager_fp32=0.004174 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 2.1 | 2.1..2.1 | 1 | - | - | - | 2368.7 | 434.5 | max_rel_diff_vs_torch_eager_fp32=18499.135971, rel_fro_vs_torch_eager_fp32=0.003425 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'inplanes': 64, 'planes': 64}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---|
| library (source) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | 7 | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(not_ready: {"error": "ImportError('libamdhip64.so.7: cannot open shared object file: No such file or directory')", "event": "error", "stage": "ready"}) |
| torch-eager-fp32 | Xq | - | 0.8 | 0.8..0.8 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 0.6 | 0.6..0.6 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 0.7 | 0.7..0.7 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 0.5 | 0.5..0.5 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### rmsprop / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.rmsprop.synthetic.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 120.2 | 120.2..120.2 | 1 | - | - | - | 2935.6 | 674.3 | rel_fro_vs_torch_eager_fp32=3.829e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 4.9 | 4.9..4.9 | 1 | 24.573 | - | - | 2619.7 | 896.0 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 53.8 | 53.8..53.8 | 1 | 2.233 | - | - | 3015.5 | 832.0 | rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'alpha': 0.99, 'centered': False, 'eps': 1e-08, 'lr': 0.001, 'momentum': 0.0, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-fp32 | torch-eager-fp32 |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (optimizer.defaults) | torch (optimizer.defaults) |
| alpha | 0.99 | 0.99 | 0.99 |
| eps | 1e-08 | 1e-08 | 1e-08 |
| learning_rate | 0.001 | 0.001 | 0.001 |
| momentum | 0.0 | 0.0 | 0.0 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| weight_decay | 0.0 | 0.0 | 0.0 |

### rnn-clf / synthetic (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc 0, log `logs/algos.rnn-clf.synthetic.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 693.0 | 693.0..693.0 | 1 | - | - | - | 2056.3 | 678.5 | accuracy=0.953559, logloss=0.103698 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 837.8 | 837.8..837.8 | 1 | 0.827 | - | - | 4832.5 | 260.6 | accuracy=0.953559 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 797.3 | 797.3..797.3 | 1 | 0.869 | - | - | 4881.4 | 260.6 | accuracy=0.953559 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 936.2 | 936.2..936.2 | 1 | 0.740 | - | - | 3218.4 | 212.6 | accuracy=0.953396 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 804.1 | 804.1..804.1 | 1 | 0.862 | - | - | 3267.1 | 212.6 | accuracy=0.953396 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'nonlinearity': 'tanh', 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (attributes) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| batch_size | 256 | 256 | 256 | 256 | 256 |
| betas | - | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| eps | - | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| hidden_size | 64 | 64 | 64 | 64 | 64 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| num_layers | 1 | 1 | 1 | 1 | 1 |
| seed | 7 | 7 | 7 | 7 | 7 |
| shuffle | true | true | true | true | true |
| weight_decay | - | 0.0 | 0.0 | 0.0 | 0.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 7.1 | 7.1..7.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | 2.5 | 2.5..2.5 | 1 | 2.863 | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 2.6 | 2.6..2.6 | 1 | 2.781 | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 2.9 | 2.9..2.9 | 1 | 2.442 | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 2.6 | 2.6..2.6 | 1 | 2.697 | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: predict(Xq)(Xq)

inference call, torch-eager-fp32: predict(Xq)(Xq)

inference call, torch-compile-fp32: predict(Xq)(Xq)

inference call, torch-eager-bf16: predict(Xq)(Xq)

inference call, torch-compile-bf16: predict(Xq)(Xq)

### rnn-clf / taxi-hourly (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc 0, log `logs/algos.rnn-clf.taxi-hourly.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 693.6 | 693.6..693.6 | 1 | - | - | - | 2056.5 | 678.5 | accuracy=0.868056, logloss=0.304864 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 849.1 | 849.1..849.1 | 1 | 0.817 | - | - | 5176.5 | 260.6 | accuracy=0.868056 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 829.9 | 829.9..829.9 | 1 | 0.836 | - | - | 4881.0 | 260.6 | accuracy=0.868056 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 932.6 | 932.6..932.6 | 1 | 0.744 | - | - | 3568.2 | 212.6 | accuracy=0.868056 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 952.4 | 952.4..952.4 | 1 | 0.728 | - | - | 3267.2 | 212.6 | accuracy=0.868056 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'nonlinearity': 'tanh', 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (attributes) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| batch_size | 256 | 256 | 256 | 256 | 256 |
| betas | - | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| eps | - | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| hidden_size | 64 | 64 | 64 | 64 | 64 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| num_layers | 1 | 1 | 1 | 1 | 1 |
| seed | 7 | 7 | 7 | 7 | 7 |
| shuffle | true | true | true | true | true |
| weight_decay | - | 0.0 | 0.0 | 0.0 | 0.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 7.2 | 7.2..7.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | 2.6 | 2.6..2.6 | 1 | 2.801 | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 2.3 | 2.3..2.3 | 1 | 3.122 | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 2.9 | 2.9..2.9 | 1 | 2.476 | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 3.1 | 3.1..3.1 | 1 | 2.291 | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: predict(Xq)(Xq)

inference call, torch-eager-fp32: predict(Xq)(Xq)

inference call, torch-compile-fp32: predict(Xq)(Xq)

inference call, torch-eager-bf16: predict(Xq)(Xq)

inference call, torch-compile-bf16: predict(Xq)(Xq)

### rnn-reg / synthetic (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc 0, log `logs/algos.rnn-reg.synthetic.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 690.9 | 690.9..690.9 | 1 | - | - | - | 2055.1 | 678.4 | finite=True, r2=0.977348, rmse=0.174374 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 856.8 | 856.8..856.8 | 1 | 0.806 | - | - | 2934.7 | 260.3 | finite=True, r2=0.977348, rmse=0.174374 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 867.3 | 867.3..867.3 | 1 | 0.797 | - | - | 2967.2 | 260.3 | finite=True, r2=0.977348, rmse=0.174374 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 939.1 | 939.1..939.1 | 1 | 0.736 | - | - | 2642.9 | 212.3 | finite=True, r2=0.977383, rmse=0.174238 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 952.3 | 952.3..952.3 | 1 | 0.726 | - | - | 2691.2 | 212.3 | finite=True, r2=0.977383, rmse=0.174238 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'nonlinearity': 'tanh', 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (attributes) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| batch_size | 256 | 256 | 256 | 256 | 256 |
| betas | - | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| eps | - | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| hidden_size | 64 | 64 | 64 | 64 | 64 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| num_layers | 1 | 1 | 1 | 1 | 1 |
| seed | 7 | 7 | 7 | 7 | 7 |
| shuffle | true | true | true | true | true |
| weight_decay | - | 0.0 | 0.0 | 0.0 | 0.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 3.7 | 3.7..3.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | 2.7 | 2.7..2.7 | 1 | 1.371 | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 2.7 | 2.7..2.7 | 1 | 1.355 | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 3.0 | 3.0..3.0 | 1 | 1.239 | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 3.1 | 3.1..3.1 | 1 | 1.191 | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: predict(Xq)(Xq)

inference call, torch-eager-fp32: predict(Xq)(Xq)

inference call, torch-compile-fp32: predict(Xq)(Xq)

inference call, torch-eager-bf16: predict(Xq)(Xq)

inference call, torch-compile-bf16: predict(Xq)(Xq)

### rnn-reg / taxi-hourly (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc 0, log `logs/algos.rnn-reg.taxi-hourly.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 691.4 | 691.4..691.4 | 1 | - | - | - | 2055.1 | 678.4 | finite=True, r2=0.738796, rmse=0.554271 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 845.1 | 845.1..845.1 | 1 | 0.818 | - | - | 2933.1 | 260.3 | finite=True, r2=0.738796, rmse=0.554271 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 857.8 | 857.8..857.8 | 1 | 0.806 | - | - | 2966.9 | 260.3 | finite=True, r2=0.738796, rmse=0.554271 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 916.4 | 916.4..916.4 | 1 | 0.754 | - | - | 2643.1 | 212.3 | finite=True, r2=0.738886, rmse=0.554176 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 973.8 | 973.8..973.8 | 1 | 0.710 | - | - | 2691.9 | 212.3 | finite=True, r2=0.738886, rmse=0.554176 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'nonlinearity': 'tanh', 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (attributes) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| batch_size | 256 | 256 | 256 | 256 | 256 |
| betas | - | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| eps | - | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| hidden_size | 64 | 64 | 64 | 64 | 64 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| num_layers | 1 | 1 | 1 | 1 | 1 |
| seed | 7 | 7 | 7 | 7 | 7 |
| shuffle | true | true | true | true | true |
| weight_decay | - | 0.0 | 0.0 | 0.0 | 0.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 3.6 | 3.6..3.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | Xq | - | 2.7 | 2.7..2.7 | 1 | 1.356 | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | Xq | - | 2.7 | 2.7..2.7 | 1 | 1.322 | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-eager-bf16 | Xq | - | 3.0 | 3.0..3.0 | 1 | 1.194 | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-bf16 | Xq | - | 3.1 | 3.1..3.1 | 1 | 1.149 | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: predict(Xq)(Xq)

inference call, torch-eager-fp32: predict(Xq)(Xq)

inference call, torch-compile-fp32: predict(Xq)(Xq)

inference call, torch-eager-bf16: predict(Xq)(Xq)

inference call, torch-compile-bf16: predict(Xq)(Xq)

### sgd / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.sgd.synthetic.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 149.3 | 149.3..149.3 | 1 | - | - | - | 3000.1 | 674.3 | rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested identical, read back unknown) |
| torch-eager-fp32 | torch | gpu | opponent | 3.7 | 3.7..3.7 | 1 | - | - | - | 2599.9 | 832.0 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 103.8 | 103.8..103.8 | 1 | - | - | - | 3001.0 | 832.0 | rel_fro_vs_torch_eager_fp32=6.772e-10 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'dampening': 0.0, 'lr': 0.001, 'maximize': False, 'momentum': 0.9, 'nesterov': False, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-fp32 | torch-eager-fp32 |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (optimizer.defaults) | torch (optimizer.defaults) |
| dampening | 0.0 | 0.0 | 0.0 |
| learning_rate | 0.001 | 0.001 | 0.001 |
| momentum | 0.9 | 0.9 | 0.9 |
| nesterov | false | false | false |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| weight_decay | 0.0 | 0.0 | 0.0 |

### svd / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.svd.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 95827.9 | 95827.9..95827.9 | 1 | - | - | - | 7944.2 | 1698.4 | max_rel_singular_value_error=73.628098, relative_reconstruction_error_100k_rows=0.0005441 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| numpy-cpu | numpy | cpu | opponent | 3741.4 | 3741.4..3741.4 | 1 | 25.613 | - | - | 8708.9 | - | max_rel_singular_value_error=1.000000, relative_reconstruction_error_100k_rows=4.1e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-gpu | torch | gpu | opponent | 2939.0 | 2939.0..2939.0 | 1 | 32.605 | - | - | 2916.8 | 3360.7 | max_rel_singular_value_error=2.707e+08, relative_reconstruction_error_100k_rows=0.026534 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'full_matrices': False}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | numpy-cpu | ours | torch-gpu |
|---|---||---|---||---|---|
| library (source) | numpy (declared) | mojolearn (declared) | torch (declared) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### svd / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.svd.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 4142.6 | 4142.6..4142.6 | 1 | - | - | - | 2487.2 | 866.4 | max_rel_singular_value_error=6.332e-07, relative_reconstruction_error_100k_rows=0.000333 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| numpy-cpu | numpy | cpu | opponent | 130.5 | 130.5..130.5 | 1 | 31.739 | - | - | 485.8 | - | max_rel_singular_value_error=4.308e-08, relative_reconstruction_error_100k_rows=4.314e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-gpu | torch | gpu | opponent | 272.1 | 272.1..272.1 | 1 | 15.223 | - | - | 1865.6 | 168.0 | max_rel_singular_value_error=4.409e-05, relative_reconstruction_error_100k_rows=0.003043 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'full_matrices': False}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | numpy-cpu | ours | torch-gpu |
|---|---||---|---||---|---|
| library (source) | numpy (declared) | mojolearn (declared) | torch (declared) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### svgp / istella (rows full, shape X 100000x220; Xq 20000x220; y 100000; yq 20000)

race: done, driver rc 0, log `logs/algos.svgp.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 948.6 | 948.6..948.6 | 1 | - | - | - | 1761.3 | 676.3 | finite=True, r2=-0.106016, rmse=0.878373 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| gpytorch-gpu | gpytorch | gpu | opponent | 16.2 | 16.2..16.2 | 1 | 58.392 | - | - | 5943.4 | 1687.8 | finite=True, r2=-0.106040, rmse=0.878383 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| gpytorch-cpu | gpytorch | cpu | opponent | 672.0 | 672.0..672.0 | 1 | 1.412 | - | - | 3457.1 | - | finite=True, r2=-0.106040, rmse=0.878383 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, gpytorch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, gpytorch-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'jitter': 1e-06, 'kernel_variance': 1.0, 'lengthscale': 1.0, 'n_inducing': 512, 'noise_variance': 1.0}. Rows: None. Timed: None.

mismatch: no seed on any arm: nothing is drawn (fixed inducing points, closed form)

mismatch: jitter: ours 1e-6 on K_uu; gpytorch adds its own Cholesky jitter (1e-6 in float32) only when a factorization fails

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | gpytorch-cpu | gpytorch-gpu | ours |
|---|---||---|---||---|---|
| library (source) | gpytorch (declared) | gpytorch (declared) | mojolearn (declared) |
| seed | 7 | 7 | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 58.6 | 58.6..58.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| gpytorch-gpu | Xq | - | 3.2 | 3.2..3.2 | 1 | 18.284 | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| gpytorch-cpu | Xq | - | 240.6 | 240.6..240.6 | 1 | 0.244 | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: predict(Xq)(Xq)

inference call, gpytorch-gpu: predict(Xq)(Xq)

inference call, gpytorch-cpu: predict(Xq)(Xq)

### svgp / taxi (rows full, shape X 100000x11; Xq 20000x11; y 100000; yq 20000)

race: done, driver rc 0, log `logs/algos.svgp.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "ValueError('SVGP: the inducing system is not positive definite; raise jitter or noise_variance')", "event": "error", "stage": "round 0"}) |
| gpytorch-gpu | gpytorch | gpu | opponent | 15.4 | 15.4..15.4 | 1 | - | - | - | 4970.7 | 1590.9 | finite=True, r2=-0.209526, rmse=17.831014 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| gpytorch-cpu | gpytorch | cpu | opponent | 504.6 | 504.6..504.6 | 1 | - | - | - | 2461.6 | - | finite=True, r2=-0.209532, rmse=17.831057 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, gpytorch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, gpytorch-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'jitter': 1e-06, 'kernel_variance': 1.0, 'lengthscale': 1.0, 'n_inducing': 512, 'noise_variance': 1.0}. Rows: None. Timed: None.

mismatch: no seed on any arm: nothing is drawn (fixed inducing points, closed form)

mismatch: jitter: ours 1e-6 on K_uu; gpytorch adds its own Cholesky jitter (1e-6 in float32) only when a factorization fails

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | gpytorch-cpu | gpytorch-gpu | ours |
|---|---||---|---||---|---|
| library (source) | gpytorch (declared) | gpytorch (declared) | mojolearn (declared) |
| seed | 7 | 7 | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(error: {"error": "ValueError('SVGP: the inducing system is not positive definite; raise jitter or noise_variance')", "event": "error", "stage": "round 0"}) |
| gpytorch-gpu | Xq | - | 2.6 | 2.6..2.6 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| gpytorch-cpu | Xq | - | 144.9 | 144.9..144.9 | 1 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: predict(Xq)(Xq)

inference call, gpytorch-gpu: predict(Xq)(Xq)

inference call, gpytorch-cpu: predict(Xq)(Xq)

## Not covered by this board

- Classical, wave 2: RadiusNeighbors, the preprocessing scalers, HDBSCAN's prediction data, Cholesky and the parallel_* and Distributed* wrappers are public and not raced here; taxi-derived time series are not used (the ARIMA and ExponentialSmoothing lanes fit seeded synthetic series, as the repo's own ARIMA quality work does).
- Classical, wave 2, not planned on this vendor: cuML and cuVS: CUDA only; no ROCm build is pinned.
- Classical, wave 2, not planned on this vendor: faiss-gpu: the pinned FAISS GPU builds are CUDA; faiss-cpu is the arm on this box.
- Inference, trees: a single-row latency batch is not timed (the batches are the held-out split and 1,000,000 training rows); ONNX, Treelite and other export paths are not raced.
- Inference, classical: the classical2 family's predict calls (the linear models, GaussianMixture, SVR, KernelRidge and others in tools/bench_board_more.py) are timed as those lanes define their clocks, not as a separate inference cell; svc times predict, not decision_function.
- Our CPU tier: `--no-cpu-arm` was passed, so no `ours-cpu` arm ran.
- Our CPU tier, no ours-cpu arm: neural lm-host-train-step, lm-infer, mamba1-infer, mamba2-infer, mamba3-infer, mlp-infer, samba-infer, transformer-infer: its `ours` arm already IS the CPU path (the public *Inference class runs on the host binding).
- Our CPU tier: a GBDT configuration the host side does not restate refuses by name in its ours-cpu cell (python/mojolearn/host_surface.py NO_CPU_PATH lists them), and a FAST-only run (`--modes fast`) has no ours-cpu arm: the host bindings build IDENTICAL only.
- Memory: GPU memory on Apple has no per-process counter (Metal buffers are inside the host footprint); the trees driver runs every arm in one process, so its GPU figure is the process total; a figure taken at the round's end misses a buffer freed inside the round; inference cells carry memory only on the classical lanes.
- Neural: The Mamba torch-* arms are the repo's pure-PyTorch references (mamba/corpus/gen_corpus.py: mamba_ssm's selective_scan_ref for Mamba-1, the chunked SSD reference for Mamba-2, the SISO reference for Mamba-3), not deployment kernels. On NVIDIA the mamba-ssm-* arms are mamba_ssm's own fused CUDA/Triton kernels (the deployment path); on AMD and Apple the Mamba lanes race the references only (NOT_PLANNED says why).
- Neural: The blocks' backward (the Mamba and TransformerBlock VJPs), their decode `step`, ragged `lengths` and the carried-state forward are public and not raced; only a zero-state forward is.
- Neural: SmallMLPTrainer.predict_logits (the GPU forward of the 8-16-3 MLP) is not raced; MLPInference (its CPU forward) and the training step are.
- Neural: The GPT-3-small target shape is not on the board (the LM lanes use the smaller control shape so one shape runs on every box, a 16 GB Mac included).
- Neural, not planned on this vendor: torch-eager-tf32 / torch-compile-tf32: TF32 is an NVIDIA CUDA tensor-core matmul mode; torch on ROCm accepts the flag and changes nothing
- Neural, not planned on this vendor: torch-compile-* on mamba1-forward and mamba1-infer: the only torch Mamba-1 twin is the pure-PyTorch reference scan, a per-token Python loop that torch.compile would unroll L times; mamba1 races the eager arms only
- Neural, not planned on this vendor: gemm-bf16 races torch's bf16 settings only (bf16 operands, fp32 accumulate)
- Neural, not planned on this vendor: gemm-int8: torch._int_mm is a CUDA kernel; torch on ROCm has no int8 matmul, so ours races alone
- Neural, not planned on this vendor: mamba-ssm-* on mamba*-forward: mamba_ssm publishes no ROCm wheel, and its gfx942 source build (setup.py's HIP path, causal-conv1d's and the Mamba-3 Triton kernels on ROCm) has not been built and checked on the board's AMD box; the Mamba lanes race the torch reference arms there
- Neural, not planned on this vendor: mamba-ssm bf16: our Mamba blocks are float32, so mamba_ssm races in float32 (and its TF32 setting); torch's bf16 arms carry the lower precision

