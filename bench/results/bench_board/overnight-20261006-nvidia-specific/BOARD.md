# mojolearn benchmark board

Generated 2026-10-06T05:21:56Z from `board.json` (schema `mojolearn-bench-board/1`).

## Box

| field | value |
|---|---|
| vendor / API | nvidia / cuda |
| GPU | NVIDIA L40S |
| GPU driver | 580.159.03 |
| CPU | AMD EPYC 9554 64-Core Processor (128 logical cores) |
| memory bytes | 1081850585088 |
| OS | Ubuntu 22.04.5 LTS |
| Python | 3.11.10 CPython |
| mojolearn | None (wheel None, sha256 None) |
| script commit | - |
| patch sync | - |
| modes | identical |
| rounds | 1 timed after 1 warm-up, arms interleaved round by round |
| seed | 7 |
| opponent versions | catboost 1.2.10, xgboost 3.2.0, lightgbm 4.7.0, scikit-learn 1.7.2, cuml-cu12 26.8.0, cuvs-cu12 26.8.1, torch 2.13.0+cu129, numba 0.64.0, statsmodels 0.15.0, faiss-cpu 1.15.1, numpy 2.4.6 |

## How to read this board

- Times are wall milliseconds of the public fit call (trees) or the lane's timed call (classical), median of the timed rounds; min..max beside it.
- `ours IDENTICAL / arm` is our IDENTICAL median divided by that opponent's median; `ours FAST / arm` likewise. Below 1.0 our median time is the lower one, above 1.0 the higher one. A ratio is shown only when both arms completed every round in this run, and only against an opponent: our two modes are never divided by each other here.
- Quality comes from the drivers: FSPEED-ACC for trees (held-out rows), one float64 NumPy function per lane for classical.
- Comparability: trees carry FSPEED-FIT-VERDICT (total leaves within 10% across arms is COMPARABLE); classical carry the clock span (SPAN-ASYMMETRIC names an arm whose clock excludes an upload or a fit that ours includes).
- Classical, wave 2 (`classical2`, tools/bench_board_more.py): the same worker protocol as classical; every lane's parameters, rows, timed span and each unavoidable mismatch with its reason are in the cells' `settings.lane_config`. Quality is one float64 NumPy function per lane over each arm's saved outputs.
- Neural: our IDENTICAL arm only (the neural surface builds no other tier, on any vendor) against torch at every fast setting it supports on this box, one arm each, the setting in the arm name: `torch-eager-fp32` (TF32 off), `torch-compile-fp32` (torch.compile, inductor), `torch-eager-tf32` / `torch-compile-tf32` (NVIDIA CUDA only), `torch-eager-bf16` / `torch-compile-bf16` (bf16 autocast mixed precision). TF32 and bf16 arms are ANOTHER PRECISION than ours; their quality columns show how far. An arm torch cannot run on this box is REFUSED by name in its cell. Every clock is host in, host out, synchronized. Every arm starts from the same parameters and reads the same inputs, so losses and outputs are comparable; `max_abs_diff_vs_ours` / `max_rel_diff_vs_ours` are the arm's output against ours.
- `installed_wheel` confirms our binding loaded from site-packages, not the repo tree.
- Our CPU is never raced or reported: the board races only our GPU, against GPU opponents; a race keeps CPU opponents only when it has no GPU opponent (Andrew, Oct 2 2026). A cell of ours on the CPU in an old record is dropped before rendering.
- Memory: `peak host MB` and `peak GPU MB` are the highest per-round peaks over the timed rounds, read outside the clock; each arm's method is listed under its table (host: the resettable peak RSS on Linux, the peak physical footprint on macOS, which holds Metal buffers too; GPU: torch's own counter for torch arms, the driver's per-process figure for the rest, none on Apple).
- Inference: after a race's fit rounds each arm predicts with its own fitted model (no fit retimed), same rows, same output kind, one warm-up then the timed rounds interleaved. Trees: batch `test` (the held-out split) and `large` (1,000,000 training rows, capped at the training rows), host rows in and host predictions out on every arm; each arm's call is printed under its table. Classical: kmeans predict, pca transform, ols predict and svc predict on the eval rows, with the fit's clock span. Ratios are per batch, ours over each opponent.

## Coverage

Races: 114 planned, 25 done, 1 failed, 0 unsupported, 88 pending. Cells: 59 (REFUSED 1, ok 58).

Inference cells: 48 (ok 48).

## Quality at a glance

Per lane and dataset: our FAST value, our IDENTICAL value, and each opponent's.

| family | lane | dataset | metric | ours FAST | ours IDENTICAL | opponents |
|---|---|---|---|---|---|---|
| algos | adafactor | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000 |
| algos | adam | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 4.891e-08 |
| algos | adamw | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 2.203e-07 |
| algos | autoarima | taxi-hourly | forecast_rmse (lower is better) | - | - | cuml-gpu 106.817545 |
| algos | avgpool2d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-tf32 0.000000; torch-compile-tf32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | avgpool2d | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-tf32 0.000000; torch-compile-tf32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | batchnorm2d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.001863; torch-eager-tf32 0.000000; torch-compile-tf32 0.001863; torch-eager-bf16 0.000000; torch-compile-bf16 0.001863 |
| algos | batchnorm2d | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 4.327e-08; torch-eager-tf32 0.000000; torch-compile-tf32 4.327e-08; torch-eager-bf16 0.000000; torch-compile-bf16 4.327e-08 |
| algos | bernoulli-nb | taxi | accuracy (higher is better) | - | - | cuml-gpu 0.755560 |
| algos | bernoulli-nb | taxi | logloss (lower is better) | - | - | cuml-gpu 0.557803 |
| algos | cagra | taxi | recall_at_10 (higher is better) | - | - | cuvs-gpu 0.619300 |
| algos | categorical-nb | taxi | accuracy (higher is better) | - | - | cuml-gpu 0.765850 |
| algos | categorical-nb | taxi | logloss (lower is better) | - | - | cuml-gpu 0.538866 |
| algos | complement-nb | istella | accuracy (higher is better) | - | - | cuml-gpu 0.849350 |
| algos | complement-nb | istella | logloss (lower is better) | - | - | cuml-gpu 3.763060 |
| algos | complement-nb | text | accuracy (higher is better) | - | - | cuml-gpu 0.983067 |
| algos | complement-nb | text | logloss (lower is better) | - | - | cuml-gpu 0.559490 |
| algos | connected-components | taxi | n_components | - | - | cugraph-gpu 588 |
| algos | conv2d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.506768; torch-eager-tf32 382.931903; torch-compile-tf32 382.931903; torch-eager-bf16 3418.337554; torch-compile-bf16 3433.596343 |
| algos | conv2d | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 5.007e-07; torch-eager-tf32 0.0003021; torch-compile-tf32 0.0003021; torch-eager-bf16 0.003382; torch-compile-bf16 0.003380 |
| algos | dart-reg | istella | r2 (higher is better) | - | - | xgboost-gpu 0.577383 |
| algos | dart-reg | istella | rmse (lower is better) | - | - | xgboost-gpu 0.543042 |
| algos | dart | istella | accuracy (higher is better) | - | - | xgboost-gpu 0.954280 |
| algos | dart | istella | logloss (lower is better) | - | - | xgboost-gpu 0.115097 |
| algos | decision-tree-clf | istella | accuracy (higher is better) | - | - | cuml-gpu 0.935000 |
| algos | decision-tree-clf | istella | logloss (lower is better) | - | - | cuml-gpu 0.758303 |
| algos | decision-tree-reg | istella | r2 (higher is better) | - | - | cuml-gpu 0.379438 |
| algos | decision-tree-reg | istella | rmse (lower is better) | - | - | cuml-gpu 0.658041 |
| algos | embedding | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000 |
| algos | embedding | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000 |
| algos | gaussian-nb | taxi | accuracy (higher is better) | - | - | cuml-gpu 0.719810 |
| algos | gaussian-nb | taxi | logloss (lower is better) | - | - | cuml-gpu 1.132317 |
| algos | gaussian-rp | taxi | mean_abs_distortion | - | - | cuml-gpu 0.302259 |
| algos | gcn | taxi | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.002327; torch-eager-tf32 0.001863; torch-compile-tf32 0.002794; torch-eager-bf16 1509.509282; torch-compile-bf16 1509.508234 |
| algos | gcn | taxi | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 9.803e-08; torch-eager-tf32 7.223e-08; torch-compile-tf32 9.801e-08; torch-eager-bf16 0.002330; torch-compile-bf16 0.002330 |
| algos | global-maxpool | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-tf32 0.000000; torch-compile-tf32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | global-maxpool | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-tf32 0.000000; torch-compile-tf32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |

## Inference at a glance

Batch prediction, each arm with its own fitted model from the same race; medians in ms. `FAST = IDENTICAL bits` compares our two tiers' predictions on the same rows. Our CPU is never raced or reported.

| family | lane | dataset | batch | rows | ours FAST ms | ours IDENTICAL ms | FAST = IDENTICAL bits | opponents |
|---|---|---|---|---|---|---|---|---|
| algos | avgpool2d | synthetic | Xq | - | - | - | - | torch-eager-fp32 0.5 ms (IDENTICAL/arm -); torch-compile-fp32 0.6 ms (IDENTICAL/arm -); torch-eager-tf32 0.5 ms (IDENTICAL/arm -); torch-compile-tf32 0.6 ms (IDENTICAL/arm -); torch-eager-bf16 0.5 ms (IDENTICAL/arm -); torch-compile-bf16 0.6 ms (IDENTICAL/arm -) |
| algos | batchnorm2d | synthetic | Xq | - | - | - | - | torch-eager-fp32 0.2 ms (IDENTICAL/arm -); torch-compile-fp32 0.2 ms (IDENTICAL/arm -); torch-eager-tf32 0.2 ms (IDENTICAL/arm -); torch-compile-tf32 0.2 ms (IDENTICAL/arm -); torch-eager-bf16 0.2 ms (IDENTICAL/arm -); torch-compile-bf16 0.2 ms (IDENTICAL/arm -) |
| algos | bernoulli-nb | taxi | Xq | - | - | - | - | cuml-gpu 2.0 ms (IDENTICAL/arm -) |
| algos | binarizer | taxi | Xq | - | - | - | - | cuml-gpu 11.4 ms (IDENTICAL/arm -) |
| algos | cagra | taxi | Xq | - | - | - | - | cuvs-gpu 1275.0 ms (IDENTICAL/arm -) |
| algos | categorical-nb | taxi | Xq | - | - | - | - | cuml-gpu 6.5 ms (IDENTICAL/arm -) |
| algos | complement-nb | istella | Xq | - | - | - | - | cuml-gpu 2.2 ms (IDENTICAL/arm -) |
| algos | complement-nb | text | Xq | - | - | - | - | cuml-gpu 2.4 ms (IDENTICAL/arm -) |
| algos | conv2d | synthetic | Xq | - | - | - | - | torch-eager-fp32 0.5 ms (IDENTICAL/arm -); torch-compile-fp32 1.2 ms (IDENTICAL/arm -); torch-eager-tf32 0.4 ms (IDENTICAL/arm -); torch-compile-tf32 0.6 ms (IDENTICAL/arm -); torch-eager-bf16 0.3 ms (IDENTICAL/arm -); torch-compile-bf16 0.5 ms (IDENTICAL/arm -) |
| algos | dart-reg | istella | Xq | - | - | - | - | xgboost-gpu 146.3 ms (IDENTICAL/arm -) |
| algos | dart | istella | Xq | - | - | - | - | xgboost-gpu 157.1 ms (IDENTICAL/arm -) |
| algos | decision-tree-clf | istella | Xq | - | - | - | - | cuml-gpu 3.9 ms (IDENTICAL/arm -) |
| algos | decision-tree-reg | istella | Xq | - | - | - | - | cuml-gpu 0.7 ms (IDENTICAL/arm -) |
| algos | dropout2d | synthetic | Xq | - | - | - | - | torch-eager-fp32 0.2 ms (IDENTICAL/arm -); torch-compile-fp32 0.3 ms (IDENTICAL/arm -); torch-eager-tf32 0.2 ms (IDENTICAL/arm -); torch-compile-tf32 0.5 ms (IDENTICAL/arm -) |
| algos | embedding | synthetic | Xq | - | - | - | - | torch-eager-fp32 0.4 ms (IDENTICAL/arm -); torch-compile-fp32 0.4 ms (IDENTICAL/arm -) |
| algos | gaussian-nb | taxi | Xq | - | - | - | - | cuml-gpu 8.5 ms (IDENTICAL/arm -) |
| algos | gaussian-rp | taxi | Xq | - | - | - | - | cuml-gpu 3.9 ms (IDENTICAL/arm -) |
| algos | gcn | taxi | Xq | - | - | - | - | torch-eager-fp32 5.5 ms (IDENTICAL/arm -); torch-compile-fp32 1.5 ms (IDENTICAL/arm -); torch-eager-tf32 5.5 ms (IDENTICAL/arm -); torch-compile-tf32 1.5 ms (IDENTICAL/arm -); torch-eager-bf16 4.5 ms (IDENTICAL/arm -); torch-compile-bf16 1.4 ms (IDENTICAL/arm -) |
| algos | global-maxpool | synthetic | Xq | - | - | - | - | torch-eager-fp32 0.1 ms (IDENTICAL/arm -); torch-compile-fp32 0.2 ms (IDENTICAL/arm -); torch-eager-tf32 0.1 ms (IDENTICAL/arm -); torch-compile-tf32 0.1 ms (IDENTICAL/arm -); torch-eager-bf16 0.1 ms (IDENTICAL/arm -); torch-compile-bf16 0.1 ms (IDENTICAL/arm -) |

## Algorithm expansion

### adafactor / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.adafactor.synthetic.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 15.3 | 15.3..15.3 | 1 | - | - | 1035.9 | 960.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 169.7 | 169.7..169.7 | 1 | - | - | 1118.9 | 1024.0 | rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'beta2_decay': -0.8, 'd': 1.0, 'eps': [None, 0.001], 'lr': 0.001, 'maximize': False, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-fp32 | torch-eager-fp32 |
|---|---||---|---|
| library (source) | torch (optimizer.defaults) | torch (optimizer.defaults) |
| eps | [null, 0.001] | [null, 0.001] |
| learning_rate | 0.001 | 0.001 |
| seed | "none (deterministic)" | "none (deterministic)" |
| weight_decay | 0.0 | 0.0 |

### adam / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.adam.synthetic.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 16.8 | 16.8..16.8 | 1 | - | - | 970.9 | 960.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 141.0 | 141.0..141.0 | 1 | - | - | 1441.2 | 896.0 | rel_fro_vs_torch_eager_fp32=4.891e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'betas': [0.9, 0.999], 'eps': 1e-08, 'lr': 0.001, 'maximize': False, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-fp32 | torch-eager-fp32 |
|---|---||---|---|
| library (source) | torch (optimizer.defaults) | torch (optimizer.defaults) |
| amsgrad | false | false |
| betas | [0.9, 0.999] | [0.9, 0.999] |
| eps | 1e-08 | 1e-08 |
| learning_rate | 0.001 | 0.001 |
| seed | "none (deterministic)" | "none (deterministic)" |
| weight_decay | 0.0 | 0.0 |

### adamw / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.adamw.synthetic.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 18.0 | 18.0..18.0 | 1 | - | - | 970.7 | 960.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 147.8 | 147.8..147.8 | 1 | - | - | 1440.7 | 896.0 | rel_fro_vs_torch_eager_fp32=2.203e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'betas': [0.9, 0.999], 'eps': 1e-08, 'lr': 0.001, 'maximize': False, 'weight_decay': 0.01}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-fp32 | torch-eager-fp32 |
|---|---||---|---|
| library (source) | torch (optimizer.defaults) | torch (optimizer.defaults) |
| amsgrad | false | false |
| betas | [0.9, 0.999] | [0.9, 0.999] |
| eps | 1e-08 | 1e-08 |
| learning_rate | 0.001 | 0.001 |
| seed | "none (deterministic)" | "none (deterministic)" |
| weight_decay | 0.01 | 0.01 |

### als / text (rows full, shape X 86626x4096; Xq 9626x4096; y 86626; yq 9626)

race: failed, driver rc 1, log `logs/algos.als.text.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implicit-gpu | implicit | gpu | opponent | - | - | 0 | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(not_ready: {"error": "RuntimeError('REFUSED: the pinned implicit wheel was built without CUDA (implicit.gpu.HAS_CUDA is False)')", "event": "error", "stage": "ready"}) (measured this run) |

settings: {'alpha': 1.0, 'calculate_training_loss': False, 'cg_steps': 3, 'factors': 64, 'iterations': 15, 'random_state': 7, 'regularization': 0.01, 'use_cg': False}. Rows: None. Timed: None.

mismatch: implicit-gpu has only the conjugate-gradient solver (use_cg ignored there); ours and implicit-cpu solve each least-squares step exactly (use_cg=False)

mismatch: each library draws its own initial factors from random_state=7

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `None`, seed 7): MATCHED

| parameter |  |

| library (source) |  |

### autoarima / taxi-hourly (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc 0, log `logs/algos.autoarima.taxi-hourly.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 157302.2 | 157302.2..157302.2 | 1 | - | - | 868.2 | 816.0 | forecast_rmse=106.817545 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'allow_intercept': True, 'ic': 'aicc', 'max_d': 1, 'max_p': 3, 'max_q': 3, 'seasonal': False, 'stepwise': False}. Rows: None. Timed: None.

mismatch: the likelihood optimizer and its stopping rule are each library's own

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (declared) |
| seasonal | false |
| seed | "none (deterministic)" |

### avgpool2d / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.avgpool2d.synthetic.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 1.9 | 1.9..1.9 | 1 | - | - | 711.3 | 541.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 1.1 | 1.1..1.1 | 1 | - | - | 943.1 | 540.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | opponent | 2.3 | 2.3..2.3 | 1 | - | - | 711.5 | 541.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | opponent | 2.3 | 2.3..2.3 | 1 | - | - | 892.9 | 541.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 1.7 | 1.7..1.7 | 1 | - | - | 711.3 | 541.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 1.7 | 1.7..1.7 | 1 | - | - | 887.7 | 541.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'ceil_mode': False, 'count_include_pad': True, 'divisor_override': None, 'kernel_size': 2, 'padding': 0, 'stride': 2}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-bf16 | torch-compile-fp32 | torch-compile-tf32 | torch-eager-bf16 | torch-eager-fp32 | torch-eager-tf32 |
|---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | 7 | 7 | 7 | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | Xq | - | 0.5 | 0.5..0.5 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | Xq | - | 0.6 | 0.6..0.6 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-tf32 | Xq | - | 0.5 | 0.5..0.5 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-tf32 | Xq | - | 0.6 | 0.6..0.6 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | Xq | - | 0.5 | 0.5..0.5 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | Xq | - | 0.6 | 0.6..0.6 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-tf32: forward(x) (no autograd)(Xq)

inference call, torch-compile-tf32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### batchnorm2d / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.batchnorm2d.synthetic.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 1.3 | 1.3..1.3 | 1 | - | - | 761.8 | 250.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 1.4 | 1.4..1.4 | 1 | - | - | 965.4 | 246.0 | max_rel_diff_vs_torch_eager_fp32=0.001863, rel_fro_vs_torch_eager_fp32=4.327e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | opponent | 1.2 | 1.2..1.2 | 1 | - | - | 761.8 | 250.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | opponent | 1.6 | 1.6..1.6 | 1 | - | - | 903.3 | 250.0 | max_rel_diff_vs_torch_eager_fp32=0.001863, rel_fro_vs_torch_eager_fp32=4.327e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 1.4 | 1.4..1.4 | 1 | - | - | 761.7 | 250.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 1.9 | 1.9..1.9 | 1 | - | - | 894.8 | 250.0 | max_rel_diff_vs_torch_eager_fp32=0.001863, rel_fro_vs_torch_eager_fp32=4.327e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'affine': True, 'eps': 1e-05, 'momentum': 0.1, 'num_features': 64, 'track_running_stats': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-bf16 | torch-compile-fp32 | torch-compile-tf32 | torch-eager-bf16 | torch-eager-fp32 | torch-eager-tf32 |
|---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| eps | 1e-05 | 1e-05 | 1e-05 | 1e-05 | 1e-05 | 1e-05 |
| momentum | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| seed | 7 | 7 | 7 | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | Xq | - | 0.2 | 0.2..0.2 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | Xq | - | 0.2 | 0.2..0.2 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-tf32 | Xq | - | 0.2 | 0.2..0.2 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-tf32 | Xq | - | 0.2 | 0.2..0.2 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | Xq | - | 0.2 | 0.2..0.2 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | Xq | - | 0.2 | 0.2..0.2 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-tf32: forward(x) (no autograd)(Xq)

inference call, torch-compile-tf32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### bernoulli-nb / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.bernoulli-nb.taxi.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 12.9 | 12.9..12.9 | 1 | - | - | 1172.6 | 494.0 | accuracy=0.755560, logloss=0.557803 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'alpha': 1.0, 'binarize': 0.0, 'fit_prior': True, 'force_alpha': True}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), BernoulliNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (get_params) |
| alpha | 1.0 |
| seed | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | Xq | - | 2.0 | 2.0..2.0 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: predict(Xq)(Xq)

### binarizer / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.binarizer.taxi.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 2.4 | 2.4..2.4 | 1 | - | - | 892.9 | 486.0 | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'threshold': 0.0}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), Binarizer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (get_params) |
| seed | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | Xq | - | 11.4 | 11.4..11.4 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: transform(Xq)(Xq)

### cagra / taxi (rows full, shape index 400000x11; queries 4000x11)

race: done, driver rc 0, log `logs/algos.cagra.taxi.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuvs-gpu | cuvs | gpu | opponent | 460.0 | 460.0..460.0 | 1 | - | - | 1410.6 | 526.0 | recall_at_10=0.619300 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuvs-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'graph_degree': 32, 'intermediate_graph_degree': 64, 'itopk_size': 64, 'n_neighbors': 10, 'random_state': 7}. Rows: None. Timed: None.

mismatch: faiss-cpu is HNSW (IndexHNSWFlat M=32, efConstruction=128, efSearch=64), the CPU graph index; cuvs-gpu is CAGRA itself

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuvs-gpu`, seed 7): MATCHED

| parameter | cuvs-gpu |
|---|---|
| library (source) | cuvs (declared) |
| n_neighbors | 10 |
| seed | "none (no argument; draws random numbers, see exceptions)" |

accepted difference: cuvs-gpu seed: cuVS IndexParams take no seed and the index build samples rows; ours and faiss get 7

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cuvs-gpu | Xq | - | 1275.0 | 1275.0..1275.0 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuvs-gpu: search(queries)(Xq)

### categorical-nb / taxi (rows full, shape X 1000000x5; Xq 100000x5; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.categorical-nb.taxi.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 13.5 | 13.5..13.5 | 1 | - | - | 1109.7 | 460.0 | accuracy=0.765850, logloss=0.538866 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'alpha': 1.0, 'fit_prior': True, 'force_alpha': True}. Rows: None. Timed: None.

mismatch: min_categories = every code seen in X or Xq on ours and scikit-learn; cuML has no min_categories option

config: cuML benchmark (RAPIDS), CategoricalNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (get_params) |
| alpha | 1.0 |
| seed | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | Xq | - | 6.5 | 6.5..6.5 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: predict(Xq)(Xq)

### clip-grad-norm / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.clip-grad-norm.synthetic.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 0.6 | 0.6..0.6 | 1 | - | - | 991.9 | 128.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 0.5 | 0.5..0.5 | 1 | - | - | 1182.7 | 128.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'error_if_nonfinite': True, 'max_norm': 1.0, 'norm_type': 2.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-fp32 | torch-eager-fp32 |
|---|---||---|---|
| library (source) | torch (declared) | torch (declared) |
| seed | 7 | 7 |

### complement-nb / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.complement-nb.istella.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 56.5 | 56.5..56.5 | 1 | - | - | 3934.4 | 1372.0 | accuracy=0.849350, logloss=3.763060 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'alpha': 1.0, 'fit_prior': True, 'force_alpha': True, 'norm': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), ComplementNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (get_params) |
| alpha | 1.0 |
| seed | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | Xq | - | 2.2 | 2.2..2.2 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: predict(Xq)(Xq)

### complement-nb / text (rows full, shape X 86626x4096; Xq 9626x4096; y 86626; yq 9626)

race: done, driver rc 0, log `logs/algos.complement-nb.text.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 61.1 | 61.1..61.1 | 1 | - | - | 4710.5 | 1946.0 | accuracy=0.983067, logloss=0.559490 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'alpha': 1.0, 'fit_prior': True, 'force_alpha': True, 'norm': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), ComplementNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (get_params) |
| alpha | 1.0 |
| seed | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | Xq | - | 2.4 | 2.4..2.4 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: predict(Xq)(Xq)

### connected-components / taxi (rows full, shape X 20000x11; indices 253708; indices2 54528; indptr 20001; indptr2 20001; y 20000)

race: done, driver rc 0, log `logs/algos.connected-components.taxi.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cugraph-gpu | cugraph | gpu | opponent | 23.7 | 23.7..23.7 | 1 | - | - | 934.4 | 432.0 | n_components=588 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cugraph-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {}. Rows: None. Timed: None.

mismatch: ours takes a dense adjacency matrix (its class's contract), built from the CSR graph before the clock; networkx and cuGraph take the graph itself

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cugraph-gpu`, seed 7): MATCHED

| parameter | cugraph-gpu |
|---|---|
| library (source) | cugraph (declared) |
| seed | "none (deterministic)" |

### conv2d / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.conv2d.synthetic.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 2.3 | 2.3..2.3 | 1 | - | - | 910.5 | 250.5 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 2.5 | 2.5..2.5 | 1 | - | - | 1131.1 | 346.8 | max_rel_diff_vs_torch_eager_fp32=0.506768, rel_fro_vs_torch_eager_fp32=5.007e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | opponent | 1.8 | 1.8..1.8 | 1 | - | - | 927.7 | 354.2 | max_rel_diff_vs_torch_eager_fp32=382.931903, rel_fro_vs_torch_eager_fp32=0.0003021 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | opponent | 2.2 | 2.2..2.2 | 1 | - | - | 1081.7 | 354.2 | max_rel_diff_vs_torch_eager_fp32=382.931903, rel_fro_vs_torch_eager_fp32=0.0003021 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 1.3 | 1.3..1.3 | 1 | - | - | 971.3 | 273.7 | max_rel_diff_vs_torch_eager_fp32=3418.337554, rel_fro_vs_torch_eager_fp32=0.003382 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 1.7 | 1.7..1.7 | 1 | - | - | 1180.3 | 272.7 | max_rel_diff_vs_torch_eager_fp32=3433.596343, rel_fro_vs_torch_eager_fp32=0.003380 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'bias': True, 'dilation': 1, 'groups': 1, 'in_channels': 64, 'kernel_size': 3, 'out_channels': 64, 'padding': 1, 'padding_mode': 'zeros', 'stride': 1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-bf16 | torch-compile-fp32 | torch-compile-tf32 | torch-eager-bf16 | torch-eager-fp32 | torch-eager-tf32 |
|---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | 7 | 7 | 7 | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | Xq | - | 0.5 | 0.5..0.5 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | Xq | - | 1.2 | 1.2..1.2 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-tf32 | Xq | - | 0.4 | 0.4..0.4 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-tf32 | Xq | - | 0.6 | 0.6..0.6 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | Xq | - | 0.3 | 0.3..0.3 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | Xq | - | 0.5 | 0.5..0.5 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-tf32: forward(x) (no autograd)(Xq)

inference call, torch-compile-tf32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### dart-reg / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.dart-reg.istella.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| xgboost-gpu | xgboost | gpu | opponent | 10334.1 | 10334.1..10334.1 | 1 | - | - | 2005.6 | 504.0 | finite=True, r2=0.577383, rmse=0.543042 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, xgboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'bagging_seed': 7, 'colsample_bytree': 1.0, 'drop_rate': 0.1, 'drop_seed': 7, 'feature_fraction_seed': 7, 'learning_rate': 0.1, 'max_bin': 255, 'max_delta_step': 0.0, 'max_depth': 8, 'max_drop': 50, 'min_child_samples': 20, 'n_estimators': 200, 'num_leaves': 255, 'random_state': 7, 'reg_alpha': 0.0, 'reg_lambda': 0.0, 'skip_drop': 0.5, 'subsample': 1.0, 'subsample_freq': 0, 'uniform_drop': False, 'xgboost_dart_mode': False}. Rows: None. Timed: None.

mismatch: LightGBM boosting='dart' grows leaf-wise (num_leaves=255, max_depth=8) as ours; XGBoost booster='dart' tree_method='hist' grows depth-wise (max_depth=8, no leaf cap)

mismatch: XGBoost has no max_drop (ours and LightGBM 50), no min_child_samples (ours and LightGBM 20 rows; XGBoost min_child_weight=1, a hessian sum), no uniform_drop / xgboost_dart_mode (XGBoost sample_type='uniform', normalize_type='tree') and no drop_seed: its drops come from random_state=7; each library's drop RNG is its own

mismatch: max_bin 255 on every arm (XGBoost's default is 256, set to 255 here)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `xgboost-gpu`, seed 7): MATCHED

| parameter | xgboost-gpu |
|---|---|
| library (source) | xgboost (get_params) |
| boosting_type | "dart" |
| feature_fraction | 1.0 |
| feature_fraction_bynode | null |
| grow_policy | null |
| learning_rate | 0.1 |
| max_bin | 255 |
| max_depth | 8 |
| max_leaves | null |
| min_child_weight | 1.0 |
| min_split_gain | null |
| n_estimators | 200 |
| reg_alpha | 0.0 |
| reg_lambda | 0.0 |
| scale_pos_weight | 1.0 |
| seed | 7 |
| subsample | 1.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| xgboost-gpu | Xq | - | 146.3 | 146.3..146.3 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, xgboost-gpu: predict(Xq)(Xq)

### dart / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.dart.istella.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| xgboost-gpu | xgboost | gpu | opponent | 10004.8 | 10004.8..10004.8 | 1 | - | - | 2043.5 | 504.0 | accuracy=0.954280, logloss=0.115097 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, xgboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'bagging_seed': 7, 'colsample_bytree': 1.0, 'drop_rate': 0.1, 'drop_seed': 7, 'feature_fraction_seed': 7, 'learning_rate': 0.1, 'max_bin': 255, 'max_delta_step': 0.0, 'max_depth': 8, 'max_drop': 50, 'min_child_samples': 20, 'n_estimators': 200, 'num_leaves': 255, 'random_state': 7, 'reg_alpha': 0.0, 'reg_lambda': 0.0, 'skip_drop': 0.5, 'subsample': 1.0, 'subsample_freq': 0, 'uniform_drop': False, 'xgboost_dart_mode': False}. Rows: None. Timed: None.

mismatch: LightGBM boosting='dart' grows leaf-wise (num_leaves=255, max_depth=8) as ours; XGBoost booster='dart' tree_method='hist' grows depth-wise (max_depth=8, no leaf cap)

mismatch: XGBoost has no max_drop (ours and LightGBM 50), no min_child_samples (ours and LightGBM 20 rows; XGBoost min_child_weight=1, a hessian sum), no uniform_drop / xgboost_dart_mode (XGBoost sample_type='uniform', normalize_type='tree') and no drop_seed: its drops come from random_state=7; each library's drop RNG is its own

mismatch: max_bin 255 on every arm (XGBoost's default is 256, set to 255 here)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `xgboost-gpu`, seed 7): MATCHED

| parameter | xgboost-gpu |
|---|---|
| library (source) | xgboost (get_params) |
| boosting_type | "dart" |
| feature_fraction | 1.0 |
| feature_fraction_bynode | null |
| grow_policy | null |
| learning_rate | 0.1 |
| max_bin | 255 |
| max_depth | 8 |
| max_leaves | null |
| min_child_weight | 1.0 |
| min_split_gain | null |
| n_estimators | 200 |
| reg_alpha | 0.0 |
| reg_lambda | 0.0 |
| scale_pos_weight | 1.0 |
| seed | 7 |
| subsample | 1.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| xgboost-gpu | Xq | - | 157.1 | 157.1..157.1 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, xgboost-gpu: predict(Xq)(Xq)

### decision-tree-clf / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.decision-tree-clf.istella.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 516.3 | 516.3..516.3 | 1 | - | - | 2957.0 | 1368.0 | accuracy=0.935000, logloss=0.758303 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'ccp_alpha': 0.0, 'criterion': 'gini', 'max_depth': 16, 'max_features': 1.0, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'random_state': 7, 'splitter': 'best'}. Rows: None. Timed: None.

mismatch: cuml-gpu is cuML's forest with one tree, no bootstrap, every feature (no GPU single-tree class exists), n_bins=128 as ours

mismatch: ours' DecisionTree* is its forest builder with one tree: it splits on n_bins=128 quantile bins per feature (ours only; not a scikit-learn parameter), scikit-learn on exact thresholds

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (get_params) |
| bootstrap | false |
| class_weight | null |
| max_bin | 128 |
| max_depth | 16 |
| max_features | 1.0 |
| max_leaves | -1 |
| max_samples | 1.0 |
| min_samples_leaf | 1 |
| min_split_gain | 0.0 |
| n_estimators | 1 |
| seed | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | Xq | - | 3.9 | 3.9..3.9 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: predict(Xq)(Xq)

### decision-tree-reg / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.decision-tree-reg.istella.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 523.1 | 523.1..523.1 | 1 | - | - | 2864.0 | 1364.0 | finite=True, r2=0.379438, rmse=0.658041 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'ccp_alpha': 0.0, 'criterion': 'squared_error', 'max_depth': 16, 'max_features': 1.0, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'random_state': 7, 'splitter': 'best'}. Rows: None. Timed: None.

mismatch: cuml-gpu is cuML's forest with one tree (see decision-tree-clf)

mismatch: ours' DecisionTree* is its forest builder with one tree: it splits on n_bins quantile bins per feature (128; 256 in FAST, rf_dt_default_bins) (ours only; not a scikit-learn parameter), scikit-learn on exact thresholds

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (get_params) |
| bootstrap | false |
| max_bin | 128 |
| max_depth | 16 |
| max_features | 1.0 |
| max_leaves | -1 |
| max_samples | 1.0 |
| min_samples_leaf | 1 |
| min_split_gain | 0.0 |
| n_estimators | 1 |
| seed | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | Xq | - | 0.7 | 0.7..0.7 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: predict(Xq)(Xq)

### dropout2d / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.dropout2d.synthetic.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 1.0 | 1.0..1.0 | 1 | - | - | 697.8 | 250.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 1.2 | 1.2..1.2 | 1 | - | - | 916.4 | 247.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | opponent | 1.1 | 1.1..1.1 | 1 | - | - | 697.7 | 250.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | opponent | 1.3 | 1.3..1.3 | 1 | - | - | 852.0 | 250.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'p': 0.1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-fp32 | torch-compile-tf32 | torch-eager-fp32 | torch-eager-tf32 |
|---|---||---|---||---|---||---|---|
| library (source) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| p | 0.1 | 0.1 | 0.1 | 0.1 |
| seed | 7 | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | Xq | - | 0.2 | 0.2..0.2 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | Xq | - | 0.3 | 0.3..0.3 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-tf32 | Xq | - | 0.2 | 0.2..0.2 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-tf32 | Xq | - | 0.5 | 0.5..0.5 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-tf32: forward(x) (no autograd)(Xq)

inference call, torch-compile-tf32: forward(x) (no autograd)(Xq)

### embedding / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.embedding.synthetic.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 1.9 | 1.9..1.9 | 1 | - | - | 838.7 | 782.6 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 2.1 | 2.1..2.1 | 1 | - | - | 1167.7 | 640.2 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'embedding_dim': 1024, 'max_norm': None, 'norm_type': 2.0, 'num_embeddings': 32768, 'padding_idx': None, 'scale_grad_by_freq': False, 'sparse': False}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-fp32 | torch-eager-fp32 |
|---|---||---|---|
| library (source) | torch (declared) | torch (declared) |
| seed | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | Xq | - | 0.4 | 0.4..0.4 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | Xq | - | 0.4 | 0.4..0.4 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

### gaussian-nb / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.gaussian-nb.taxi.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 22.3 | 22.3..22.3 | 1 | - | - | 974.2 | 486.0 | accuracy=0.719810, logloss=1.132317 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'var_smoothing': 1e-09}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), GaussianNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (get_params) |
| seed | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | Xq | - | 8.5 | 8.5..8.5 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: predict(Xq)(Xq)

### gaussian-rp / taxi (rows full, shape X 900000x11; Xq 100000x11)

race: done, driver rc 0, log `logs/algos.gaussian-rp.taxi.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 2.3 | 2.3..2.3 | 1 | - | - | 894.4 | 438.0 | mean_abs_distortion=0.302259 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'compute_inverse_components': False, 'eps': 0.1, 'n_components': 10, 'random_state': 7}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), GaussianRandomProjection (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (get_params) |
| eps | 0.1 |
| n_components | 10 |
| seed | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | Xq | - | 3.9 | 3.9..3.9 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: transform(Xq)(Xq)

### gcn / taxi (rows full, shape X 100000x11; indices 1258298; indptr 100001; y 100000)

race: done, driver rc 0, log `logs/algos.gcn.taxi.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 12.6 | 12.6..12.6 | 1 | - | - | 1344.8 | 1589.7 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 3.3 | 3.3..3.3 | 1 | - | - | 1281.3 | 310.5 | max_rel_diff_vs_torch_eager_fp32=0.002327, rel_fro_vs_torch_eager_fp32=9.803e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | opponent | 10.9 | 10.9..10.9 | 1 | - | - | 1347.7 | 1589.7 | max_rel_diff_vs_torch_eager_fp32=0.001863, rel_fro_vs_torch_eager_fp32=7.223e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | opponent | 3.7 | 3.7..3.7 | 1 | - | - | 1242.7 | 310.5 | max_rel_diff_vs_torch_eager_fp32=0.002794, rel_fro_vs_torch_eager_fp32=9.801e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 15.9 | 15.9..15.9 | 1 | - | - | 1464.0 | 1874.9 | max_rel_diff_vs_torch_eager_fp32=1509.509282, rel_fro_vs_torch_eager_fp32=0.002330 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 9.3 | 9.3..9.3 | 1 | - | - | 1468.7 | 591.0 | max_rel_diff_vs_torch_eager_fp32=1509.508234, rel_fro_vs_torch_eager_fp32=0.002330 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'add_self_loops': True, 'bias': True, 'improved': False, 'normalize': True, 'out_channels': 128}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-bf16 | torch-compile-fp32 | torch-compile-tf32 | torch-eager-bf16 | torch-eager-fp32 | torch-eager-tf32 |
|---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| normalize | true | true | true | true | true | true |
| seed | 7 | 7 | 7 | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | Xq | - | 5.5 | 5.5..5.5 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | Xq | - | 1.5 | 1.5..1.5 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-tf32 | Xq | - | 5.5 | 5.5..5.5 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-tf32 | Xq | - | 1.5 | 1.5..1.5 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | Xq | - | 4.5 | 4.5..4.5 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | Xq | - | 1.4 | 1.4..1.4 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-tf32: forward(x) (no autograd)(Xq)

inference call, torch-compile-tf32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### global-maxpool / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.global-maxpool.synthetic.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 0.6 | 0.6..0.6 | 1 | - | - | 669.4 | 12.9 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 1.2 | 1.2..1.2 | 1 | - | - | 904.1 | 12.9 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | opponent | 0.6 | 0.6..0.6 | 1 | - | - | 669.8 | 12.9 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | opponent | 1.1 | 1.1..1.1 | 1 | - | - | 855.8 | 12.9 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 0.6 | 0.6..0.6 | 1 | - | - | 669.4 | 12.9 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 1.0 | 1.0..1.0 | 1 | - | - | 852.1 | 12.9 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'output_size': 1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-bf16 | torch-compile-fp32 | torch-compile-tf32 | torch-eager-bf16 | torch-eager-fp32 | torch-eager-tf32 |
|---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | 7 | 7 | 7 | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | Xq | - | 0.1 | 0.1..0.1 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | Xq | - | 0.2 | 0.2..0.2 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-tf32 | Xq | - | 0.1 | 0.1..0.1 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-tf32 | Xq | - | 0.1 | 0.1..0.1 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | Xq | - | 0.1 | 0.1..0.1 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | Xq | - | 0.1 | 0.1..0.1 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-tf32: forward(x) (no autograd)(Xq)

inference call, torch-compile-tf32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

## Not covered by this board

- Classical, wave 2: RadiusNeighbors, the preprocessing scalers, HDBSCAN's prediction data, Cholesky and the parallel_* and Distributed* wrappers are public and not raced here; taxi-derived time series are not used (the ARIMA and ExponentialSmoothing lanes fit seeded synthetic series, as the repo's own ARIMA quality work does).
- Classical, wave 2, not planned on this vendor: cuML GaussianMixture, GaussianProcessRegressor/Classifier, Nystroem, RBFSampler: cuML 26.8.0 has none; scikit-learn on the CPU is the arm.
- Classical, wave 2, not planned on this vendor: faiss-gpu: no pinned PyPI wheel for this image; cuVS ivf_flat from the pinned rapids set (cuvs-cu12==26.8.1) is the IVF-Flat GPU arm.
- Classical, wave 2, not planned on this vendor: umap-learn and faiss-cpu: not installed on NVIDIA; cuML UMAP and cuVS are the arms.
- Classical, wave 2, not planned on this vendor: cuML SpectralClustering/SpectralEmbedding: present only in newer cuML; if the pinned 26.8.0 lacks them the arm refuses by name and scikit-learn stands beside it.
- Inference, trees: a single-row latency batch is not timed (the batches are the held-out split and 1,000,000 training rows); ONNX, Treelite and other export paths are not raced.
- Inference, classical: the classical2 family's predict calls (the linear models, GaussianMixture, SVR, KernelRidge and others in tools/bench_board_more.py) are timed as those lanes define their clocks, not as a separate inference cell; svc times predict, not decision_function.
- Our CPU: never raced or reported; the board races only our GPU (Andrew, Oct 2 2026). The host column gives same-bits digests only (lq ID).
- Neural, not planned: lm-host-train-step, lm-infer, mamba1-infer, mamba2-infer, mamba3-infer, mlp-infer, samba-infer, transformer-infer: ours runs the CPU binding, and our CPU is never raced, in no numeric mode (the Apple FAST neural tier is the GPU lanes only).
- Memory: GPU memory on Apple has no per-process counter (Metal buffers are inside the host footprint); the trees driver runs every arm in one process, so its GPU figure is the process total; a figure taken at the round's end misses a buffer freed inside the round; inference cells carry memory only on the classical lanes.
- Neural: The Mamba opponents are the repo's pure-PyTorch references (mamba/corpus/gen_corpus.py: mamba_ssm's selective_scan_ref for Mamba-1, the chunked SSD reference for Mamba-2, the SISO reference for Mamba-3), not mamba-ssm's fused CUDA/Triton kernels, which the board does not install; a Mamba ratio here is against a reference implementation, not a deployment kernel.
- Neural: The blocks' backward (the Mamba and TransformerBlock VJPs), their decode `step`, ragged `lengths` and the carried-state forward are public and not raced; only a zero-state forward is.
- Neural: SmallMLPTrainer.predict_logits (the GPU forward of the 8-16-3 MLP) is not raced; MLPInference (its CPU forward) and the training step are.
- Neural: The GPT-3-small target shape is not on the board (the LM lanes use the smaller control shape so one shape runs on every box, a 16 GB Mac included).
- Neural, not planned on this vendor: torch-compile-* on mamba1-forward and mamba1-infer: the only torch Mamba-1 twin is the pure-PyTorch reference scan, a per-token Python loop that torch.compile would unroll L times; mamba1 races the eager arms only
- Neural, not planned on this vendor: gemm-bf16 races torch's bf16 settings only (bf16 operands, fp32 accumulate)

