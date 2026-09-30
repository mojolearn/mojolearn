# mojolearn benchmark board

Generated 2026-09-29T14:26:07Z from `board.json` (schema `mojolearn-bench-board/1`).

> SMOKE RUN: `--rows 2000` is below the 1,000,000-row tree floor or the classical lane shapes; `--neural-shape small` is a plumbing shape. These numbers are plumbing checks, not results.

## Box

| field | value |
|---|---|
| vendor / API | apple / metal |
| GPU | Apple M3 Ultra |
| GPU driver | macOS 26.7 |
| CPU | Apple M3 Ultra (28 logical cores) |
| memory bytes | 274877906944 |
| OS | macOS 26.7 |
| Python | 3.13.15 CPython |
| mojolearn | 0.8.25 (wheel mojolearn-0.8.25-py3-none-macosx_11_0_arm64.whl, sha256 ee1187c950c29e791c906cb3beb1c45206e4d8534fea897bc839db4505b38548) |
| script commit | 6426398eb762a63da26efa21432681be062d0e32 |
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

Races: 383 planned, 9 done, 0 failed, 374 pending. Cells: 27 (ok 27).

## Quality at a glance

Per lane and dataset: our FAST value, our IDENTICAL value, our CPU IDENTICAL value, and each opponent's.

| family | lane | dataset | metric | ours FAST | ours IDENTICAL | ours CPU | opponents |
|---|---|---|---|---|---|---|---|
| algos | gamma | istella | r2 (higher is better) | -211.241328 | -211.257868 | - | sklearn-cpu -630.882560 |
| algos | gamma | istella | rmse (lower is better) | 11.565023 | 11.565474 | - | sklearn-cpu 19.954907 |
| algos | gamma | taxi | r2 (higher is better) | -12.664545 | -12.664540 | - | sklearn-cpu -12.670271 |
| algos | gamma | taxi | rmse (lower is better) | 53.996321 | 53.996312 | - | sklearn-cpu 54.007635 |
| algos | poisson | istella | r2 (higher is better) | -4589.957432 | -4590.757864 | - | sklearn-cpu -4972.234454 |
| algos | poisson | istella | rmse (lower is better) | 53.787724 | 53.792412 | - | sklearn-cpu 55.982334 |
| algos | poisson | taxi | r2 (higher is better) | 0.459311 | 0.459311 | - | sklearn-cpu 0.459307 |
| algos | poisson | taxi | rmse (lower is better) | 10.740895 | 10.740895 | - | sklearn-cpu 10.740931 |
| algos | sgd-clf | istella | accuracy (higher is better) | 0.916500 | 0.916500 | - | sklearn-cpu 0.913500 |
| algos | sgd-clf | taxi | accuracy (higher is better) | 0.755000 | 0.755000 | - | sklearn-cpu 0.756500 |
| algos | sgd-reg | istella | r2 (higher is better) | -2.425e+24 | -2.425e+24 | - | sklearn-cpu -1.163e+24 |
| algos | sgd-reg | istella | rmse (lower is better) | 1.236e+12 | 1.236e+12 | - | sklearn-cpu 8.561e+11 |
| algos | sgd-reg | taxi | r2 (higher is better) | 0.896990 | 0.896990 | - | sklearn-cpu 0.913457 |
| algos | sgd-reg | taxi | rmse (lower is better) | 4.688205 | 4.688205 | - | sklearn-cpu 4.297175 |
| algos | tweedie | taxi | r2 (higher is better) | -0.713547 | -0.713546 | - | sklearn-cpu -0.713910 |
| algos | tweedie | taxi | rmse (lower is better) | 19.121176 | 19.121171 | - | sklearn-cpu 19.123200 |

## Algorithm expansion

### gamma / istella (rows 2000, shape X 2000x220; Xq 2000x220; y 2000; yq 2000)

race: done, driver rc 0, log `logs/algos.gamma.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 21462.0 | 21462.0..21462.0 | 1 | - | - | - | 529.6 | - | finite=True, r2=-211.257868, rmse=11.565474 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 18110.9 | 18110.9..18110.9 | 1 | - | - | - | 511.8 | - | finite=True, r2=-211.241328, rmse=11.565023 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 254.0 | 254.0..254.0 | 1 | 84.511 | 71.315 | - | 163.4 | - | finite=True, r2=-630.882560, rmse=19.954907 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'fit_intercept': True, 'max_iter': 100, 'solver': 'lbfgs', 'tol': 0.0001}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.0001 | 0.0001 | 0.0001 |
| fit_intercept | true | true | true |
| max_iter | 100 | 100 | 100 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| solver | "lbfgs" | "lbfgs" | "lbfgs" |
| tol | 0.0001 | 0.0001 | 0.0001 |

### gamma / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; yq 2000)

race: done, driver rc 0, log `logs/algos.gamma.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 6.9 | 6.9..6.9 | 1 | - | - | - | 472.8 | - | finite=True, r2=-12.664540, rmse=53.996312 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 5.5 | 5.5..5.5 | 1 | - | - | - | 469.3 | - | finite=True, r2=-12.664545, rmse=53.996321 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 12.7 | 12.7..12.7 | 1 | 0.547 | 0.433 | - | 149.6 | - | finite=True, r2=-12.670271, rmse=54.007635 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'fit_intercept': True, 'max_iter': 100, 'solver': 'lbfgs', 'tol': 0.0001}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.0001 | 0.0001 | 0.0001 |
| fit_intercept | true | true | true |
| max_iter | 100 | 100 | 100 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| solver | "lbfgs" | "lbfgs" | "lbfgs" |
| tol | 0.0001 | 0.0001 | 0.0001 |

### poisson / istella (rows 2000, shape X 2000x220; Xq 2000x220; y 2000; yq 2000)

race: done, driver rc 0, log `logs/algos.poisson.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1931.5 | 1931.5..1931.5 | 1 | - | - | - | 481.3 | - | finite=True, r2=-4590.757864, rmse=53.792412 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1618.0 | 1618.0..1618.0 | 1 | - | - | - | 477.2 | - | finite=True, r2=-4589.957432, rmse=53.787724 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 246.6 | 246.6..246.6 | 1 | 7.831 | 6.560 | - | 158.4 | - | finite=True, r2=-4972.234454, rmse=55.982334 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'fit_intercept': True, 'max_iter': 100, 'solver': 'lbfgs', 'tol': 0.0001}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.0001 | 0.0001 | 0.0001 |
| fit_intercept | true | true | true |
| max_iter | 100 | 100 | 100 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| solver | "lbfgs" | "lbfgs" | "lbfgs" |
| tol | 0.0001 | 0.0001 | 0.0001 |

### poisson / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; yq 2000)

race: done, driver rc 0, log `logs/algos.poisson.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 6.9 | 6.9..6.9 | 1 | - | - | - | 466.6 | - | finite=True, r2=0.459311, rmse=10.740895 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 5.7 | 5.7..5.7 | 1 | - | - | - | 468.3 | - | finite=True, r2=0.459311, rmse=10.740895 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 12.7 | 12.7..12.7 | 1 | 0.546 | 0.445 | - | 148.4 | - | finite=True, r2=0.459307, rmse=10.740931 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'fit_intercept': True, 'max_iter': 100, 'solver': 'lbfgs', 'tol': 0.0001}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.0001 | 0.0001 | 0.0001 |
| fit_intercept | true | true | true |
| max_iter | 100 | 100 | 100 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| solver | "lbfgs" | "lbfgs" | "lbfgs" |
| tol | 0.0001 | 0.0001 | 0.0001 |

### sgd-clf / istella (rows 2000, shape X 2000x220; Xq 2000x220; y 2000; yq 2000)

race: done, driver rc 0, log `logs/algos.sgd-clf.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1615.4 | 1615.4..1615.4 | 1 | - | - | - | 479.8 | - | accuracy=0.916500 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1444.9 | 1444.9..1444.9 | 1 | - | - | - | 476.9 | - | accuracy=0.916500 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 34.7 | 34.7..34.7 | 1 | 46.490 | 41.583 | - | 149.6 | - | accuracy=0.913500 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'average': False, 'early_stopping': False, 'epsilon': 0.1, 'eta0': 0.005, 'fit_intercept': True, 'l1_ratio': 0.15, 'learning_rate': 'constant', 'loss': 'hinge', 'max_iter': 100, 'n_iter_no_change': 5, 'penalty': 'l2', 'power_t': 0.5, 'random_state': 7, 'shuffle': True, 'tol': None}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

mismatch: cuML MBSGD is mini-batch SGD (batch_size 4096); scikit-learn and ours are per-sample SGD (the reference)

mismatch: cuML reads epochs=100, the others max_iter=100

config: cuML benchmark (RAPIDS), MBSGDClassifier (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.0001 | 0.0001 | 0.0001 |
| class_weight | null | null | null |
| epsilon | 0.1 | 0.1 | 0.1 |
| eta0 | 0.005 | 0.005 | 0.005 |
| fit_intercept | true | true | true |
| l1_ratio | 0.15 | 0.15 | 0.15 |
| learning_rate | "constant" | "constant" | "constant" |
| loss | "hinge" | "hinge" | "hinge" |
| max_iter | 100 | 100 | 100 |
| penalty | "l2" | "l2" | "l2" |
| seed | 7 | 7 | 7 |
| shuffle | true | true | true |
| tol | null | null | null |

accepted difference: ours-fast class_weight: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: ours-fast tol: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: sklearn-cpu class_weight: None on both ours and sklearn-cpu: the same documented setting in both signatures

accepted difference: sklearn-cpu tol: None on both ours and sklearn-cpu: the same documented setting in both signatures

### sgd-clf / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; yq 2000)

race: done, driver rc 0, log `logs/algos.sgd-clf.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 232.1 | 232.1..232.1 | 1 | - | - | - | 473.0 | - | accuracy=0.755000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 392.7 | 392.7..392.7 | 1 | - | - | - | 471.8 | - | accuracy=0.755000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 6.9 | 6.9..6.9 | 1 | 33.441 | 56.574 | - | 144.3 | - | accuracy=0.756500 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'average': False, 'early_stopping': False, 'epsilon': 0.1, 'eta0': 0.005, 'fit_intercept': True, 'l1_ratio': 0.15, 'learning_rate': 'constant', 'loss': 'hinge', 'max_iter': 100, 'n_iter_no_change': 5, 'penalty': 'l2', 'power_t': 0.5, 'random_state': 7, 'shuffle': True, 'tol': None}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

mismatch: cuML MBSGD is mini-batch SGD (batch_size 4096); scikit-learn and ours are per-sample SGD (the reference)

mismatch: cuML reads epochs=100, the others max_iter=100

config: cuML benchmark (RAPIDS), MBSGDClassifier (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.0001 | 0.0001 | 0.0001 |
| class_weight | null | null | null |
| epsilon | 0.1 | 0.1 | 0.1 |
| eta0 | 0.005 | 0.005 | 0.005 |
| fit_intercept | true | true | true |
| l1_ratio | 0.15 | 0.15 | 0.15 |
| learning_rate | "constant" | "constant" | "constant" |
| loss | "hinge" | "hinge" | "hinge" |
| max_iter | 100 | 100 | 100 |
| penalty | "l2" | "l2" | "l2" |
| seed | 7 | 7 | 7 |
| shuffle | true | true | true |
| tol | null | null | null |

accepted difference: ours-fast class_weight: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: ours-fast tol: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: sklearn-cpu class_weight: None on both ours and sklearn-cpu: the same documented setting in both signatures

accepted difference: sklearn-cpu tol: None on both ours and sklearn-cpu: the same documented setting in both signatures

### sgd-reg / istella (rows 2000, shape X 2000x220; Xq 2000x220; y 2000; yq 2000)

race: done, driver rc 0, log `logs/algos.sgd-reg.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1637.2 | 1637.2..1637.2 | 1 | - | - | - | 473.0 | - | finite=True, r2=-2.425e+24, rmse=1.236e+12 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1294.0 | 1294.0..1294.0 | 1 | - | - | - | 474.6 | - | finite=True, r2=-2.425e+24, rmse=1.236e+12 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 67.5 | 67.5..67.5 | 1 | 24.265 | 19.178 | - | 147.3 | - | finite=True, r2=-1.163e+24, rmse=8.561e+11 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'average': False, 'early_stopping': False, 'epsilon': 0.1, 'eta0': 0.005, 'fit_intercept': True, 'l1_ratio': 0.15, 'learning_rate': 'constant', 'loss': 'squared_error', 'max_iter': 100, 'n_iter_no_change': 5, 'penalty': 'l2', 'power_t': 0.25, 'random_state': 7, 'shuffle': True, 'tol': None}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

mismatch: cuML MBSGD is mini-batch SGD (batch_size 4096)

config: cuML benchmark (RAPIDS), MBSGDRegressor (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.0001 | 0.0001 | 0.0001 |
| epsilon | 0.1 | 0.1 | 0.1 |
| eta0 | 0.005 | 0.005 | 0.005 |
| fit_intercept | true | true | true |
| l1_ratio | 0.15 | 0.15 | 0.15 |
| learning_rate | "constant" | "constant" | "constant" |
| loss | "squared_error" | "squared_error" | "squared_error" |
| max_iter | 100 | 100 | 100 |
| penalty | "l2" | "l2" | "l2" |
| seed | 7 | 7 | 7 |
| shuffle | true | true | true |
| tol | null | null | null |

accepted difference: ours-fast tol: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: sklearn-cpu tol: None on both ours and sklearn-cpu: the same documented setting in both signatures

### sgd-reg / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; yq 2000)

race: done, driver rc 0, log `logs/algos.sgd-reg.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 225.3 | 225.3..225.3 | 1 | - | - | - | 471.1 | - | finite=True, r2=0.896990, rmse=4.688205 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 396.1 | 396.1..396.1 | 1 | - | - | - | 467.6 | - | finite=True, r2=0.896990, rmse=4.688205 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 5.8 | 5.8..5.8 | 1 | 38.689 | 68.010 | - | 144.3 | - | finite=True, r2=0.913457, rmse=4.297175 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'average': False, 'early_stopping': False, 'epsilon': 0.1, 'eta0': 0.005, 'fit_intercept': True, 'l1_ratio': 0.15, 'learning_rate': 'constant', 'loss': 'squared_error', 'max_iter': 100, 'n_iter_no_change': 5, 'penalty': 'l2', 'power_t': 0.25, 'random_state': 7, 'shuffle': True, 'tol': None}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

mismatch: cuML MBSGD is mini-batch SGD (batch_size 4096)

config: cuML benchmark (RAPIDS), MBSGDRegressor (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.0001 | 0.0001 | 0.0001 |
| epsilon | 0.1 | 0.1 | 0.1 |
| eta0 | 0.005 | 0.005 | 0.005 |
| fit_intercept | true | true | true |
| l1_ratio | 0.15 | 0.15 | 0.15 |
| learning_rate | "constant" | "constant" | "constant" |
| loss | "squared_error" | "squared_error" | "squared_error" |
| max_iter | 100 | 100 | 100 |
| penalty | "l2" | "l2" | "l2" |
| seed | 7 | 7 | 7 |
| shuffle | true | true | true |
| tol | null | null | null |

accepted difference: ours-fast tol: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: sklearn-cpu tol: None on both ours and sklearn-cpu: the same documented setting in both signatures

### tweedie / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; yq 2000)

race: done, driver rc 0, log `logs/algos.tweedie.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 6.1 | 6.1..6.1 | 1 | - | - | - | 472.7 | - | finite=True, r2=-0.713546, rmse=19.121171 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 5.2 | 5.2..5.2 | 1 | - | - | - | 469.3 | - | finite=True, r2=-0.713547, rmse=19.121176 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 10.0 | 10.0..10.0 | 1 | 0.605 | 0.517 | - | 148.0 | - | finite=True, r2=-0.713910, rmse=19.123200 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'fit_intercept': True, 'link': 'log', 'max_iter': 100, 'power': 1.5, 'solver': 'lbfgs', 'tol': 0.0001}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.0001 | 0.0001 | 0.0001 |
| fit_intercept | true | true | true |
| max_iter | 100 | 100 | 100 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| solver | "lbfgs" | "lbfgs" | "lbfgs" |
| tol | 0.0001 | 0.0001 | 0.0001 |

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

