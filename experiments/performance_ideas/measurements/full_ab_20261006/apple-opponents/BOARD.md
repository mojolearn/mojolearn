# mojolearn benchmark board

Generated 2026-10-06T12:23:38Z from `board.json` (schema `mojolearn-bench-board/1`).

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
| mojolearn | 0.8.34 (wheel None, sha256 None) |
| script commit | 0f779ed5d3f0a2ab2f418e054d3af950a766e5f1 |
| patch sync | - |
| modes | identical |
| rounds | 1 timed after 1 warm-up, arms interleaved round by round |
| seed | 7 |
| opponent versions | catboost 1.2.10, xgboost 3.2.0, lightgbm 4.7.0, scikit-learn 1.7.2, torch 2.13.0, umap-learn 0.5.12, pynndescent 0.6.0, numba 0.67.0, statsmodels 0.15.0, faiss-cpu 1.15.1, numpy 2.5.3 |

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

Races: 4 planned, 4 done, 0 failed, 0 unsupported, 0 pending. Cells: 4 (ok 4).

Inference cells: 3 (ok 3).

## Quality at a glance

Per lane and dataset: our FAST value, our IDENTICAL value, and each opponent's.

| family | lane | dataset | metric | ours FAST | ours IDENTICAL | opponents |
|---|---|---|---|---|---|---|
| algos | als | text | recall_at_10 (higher is better) | - | - | implicit-cpu 0.548196 |
| algos | gamma | istella | r2 (higher is better) | - | - | sklearn-cpu 0.310339 |
| algos | gamma | istella | rmse (lower is better) | - | - | sklearn-cpu 0.692968 |
| algos | tweedie | istella | r2 (higher is better) | - | - | sklearn-cpu -45804.275829 |
| algos | tweedie | istella | rmse (lower is better) | - | - | sklearn-cpu 178.588288 |
| algos | tweedie | taxi | r2 (higher is better) | - | - | sklearn-cpu -25.819594 |
| algos | tweedie | taxi | rmse (lower is better) | - | - | sklearn-cpu 80.554149 |

## Inference at a glance

Batch prediction, each arm with its own fitted model from the same race; medians in ms. `FAST = IDENTICAL bits` compares our two tiers' predictions on the same rows. Our CPU is never raced or reported.

| family | lane | dataset | batch | rows | ours FAST ms | ours IDENTICAL ms | FAST = IDENTICAL bits | opponents |
|---|---|---|---|---|---|---|---|---|
| algos | gamma | istella | Xq | - | - | - | - | sklearn-cpu 80.4 ms (IDENTICAL/arm -) |
| algos | tweedie | istella | Xq | - | - | - | - | sklearn-cpu 80.2 ms (IDENTICAL/arm -) |
| algos | tweedie | taxi | Xq | - | - | - | - | sklearn-cpu 14.7 ms (IDENTICAL/arm -) |

## Algorithm expansion

### als / text (rows full, shape X 86626x4096; Xq 9626x4096; y 86626; yq 9626)

race: done, driver rc 0, log `logs/algos.als.text.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implicit-cpu | implicit | cpu | opponent | 321340.4 | 321340.4..321340.4 | 1 | - | - | 2070.9 | - | recall_at_10=0.548196 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, implicit-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'calculate_training_loss': False, 'cg_steps': 3, 'factors': 64, 'iterations': 15, 'random_state': 7, 'regularization': 0.01, 'use_cg': False}. Rows: None. Timed: None.

mismatch: implicit-gpu has only the conjugate-gradient solver (use_cg ignored there); ours and implicit-cpu solve each least-squares step exactly (use_cg=False)

mismatch: each library draws its own initial factors from random_state=7

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `implicit-cpu`, seed 7): MATCHED

| parameter | implicit-cpu |
|---|---|
| library (source) | implicit (declared) |
| alpha | 1.0 |
| n_estimators | 15 |
| seed | 7 |

### gamma / istella (rows full, shape X 2043304x220; Xq 500000x220; y 2043304; yq 500000)

race: done, driver rc 0, log `logs/algos.gamma.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sklearn-cpu | scikit-learn | cpu | opponent | 252506.5 | 252506.5..252506.5 | 1 | - | - | 6142.1 | - | finite=True, r2=0.310339, rmse=0.692968 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'fit_intercept': True, 'max_iter': 100, 'solver': 'lbfgs', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `sklearn-cpu`, seed 7): MATCHED

| parameter | sklearn-cpu |
|---|---|
| library (source) | sklearn (get_params) |
| alpha | 0.0001 |
| fit_intercept | true |
| max_iter | 100 |
| seed | "none (deterministic)" |
| solver | "lbfgs" |
| tol | 0.0001 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sklearn-cpu | Xq | - | 80.4 | 80.4..80.4 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, sklearn-cpu: predict(Xq)(Xq)

### tweedie / istella (rows full, shape X 2043304x220; Xq 500000x220; y 2043304; yq 500000)

race: done, driver rc 0, log `logs/algos.tweedie.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sklearn-cpu | scikit-learn | cpu | opponent | 250154.3 | 250154.3..250154.3 | 1 | - | - | 6137.9 | - | finite=True, r2=-45804.275829, rmse=178.588288 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'fit_intercept': True, 'link': 'log', 'max_iter': 100, 'power': 1.5, 'solver': 'lbfgs', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `sklearn-cpu`, seed 7): MATCHED

| parameter | sklearn-cpu |
|---|---|
| library (source) | sklearn (get_params) |
| alpha | 0.0001 |
| fit_intercept | true |
| max_iter | 100 |
| seed | "none (deterministic)" |
| solver | "lbfgs" |
| tol | 0.0001 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sklearn-cpu | Xq | - | 80.2 | 80.2..80.2 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, sklearn-cpu: predict(Xq)(Xq)

### tweedie / taxi (rows full, shape X 5250086x11; Xq 500000x11; y 5250086; yq 500000)

race: done, driver rc 0, log `logs/algos.tweedie.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sklearn-cpu | scikit-learn | cpu | opponent | 4055.3 | 4055.3..4055.3 | 1 | - | - | 656.8 | - | finite=True, r2=-25.819594, rmse=80.554149 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'fit_intercept': True, 'link': 'log', 'max_iter': 100, 'power': 1.5, 'solver': 'lbfgs', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `sklearn-cpu`, seed 7): MATCHED

| parameter | sklearn-cpu |
|---|---|
| library (source) | sklearn (get_params) |
| alpha | 0.0001 |
| fit_intercept | true |
| max_iter | 100 |
| seed | "none (deterministic)" |
| solver | "lbfgs" |
| tol | 0.0001 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sklearn-cpu | Xq | - | 14.7 | 14.7..14.7 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, sklearn-cpu: predict(Xq)(Xq)

## Not covered by this board

- Classical, wave 2: RadiusNeighbors, the preprocessing scalers, HDBSCAN's prediction data, Cholesky and the parallel_* and Distributed* wrappers are public and not raced here; taxi-derived time series are not used (the ARIMA and ExponentialSmoothing lanes fit seeded synthetic series, as the repo's own ARIMA quality work does).
- Classical, wave 2, not planned on this vendor: cuML and cuVS: CUDA only; no Apple build exists.
- Classical, wave 2, not planned on this vendor: faiss-gpu: CUDA only; faiss-cpu is the arm on this box.
- Inference, trees: a single-row latency batch is not timed (the batches are the held-out split and 1,000,000 training rows); ONNX, Treelite and other export paths are not raced.
- Inference, classical: the classical2 family's predict calls (the linear models, GaussianMixture, SVR, KernelRidge and others in tools/bench_board_more.py) are timed as those lanes define their clocks, not as a separate inference cell; svc times predict, not decision_function.
- Our CPU: never raced or reported; the board races only our GPU (Andrew, Oct 2 2026). The host column gives same-bits digests only (lq ID).
- Neural, not planned: lm-host-train-step, lm-infer, mamba1-infer, mamba2-infer, mamba3-infer, mlp-infer, samba-infer, transformer-infer: ours runs the CPU binding, and our CPU is never raced, in no numeric mode (the Apple FAST neural tier is the GPU lanes only).
- Memory: GPU memory on Apple has no per-process counter (Metal buffers are inside the host footprint); the trees driver runs every arm in one process, so its GPU figure is the process total; a figure taken at the round's end misses a buffer freed inside the round; inference cells carry memory only on the classical lanes.
- Neural: The Mamba opponents are the repo's pure-PyTorch references (mamba/corpus/gen_corpus.py: mamba_ssm's selective_scan_ref for Mamba-1, the chunked SSD reference for Mamba-2, the SISO reference for Mamba-3), not mamba-ssm's fused CUDA/Triton kernels, which the board does not install; a Mamba ratio here is against a reference implementation, not a deployment kernel.
- Neural: The blocks' backward (the Mamba and TransformerBlock VJPs), their decode `step`, ragged `lengths` and the carried-state forward are public and not raced; only a zero-state forward is.
- Neural: SmallMLPTrainer.predict_logits (the GPU forward of the 8-16-3 MLP) is not raced; MLPInference (its CPU forward) and the training step are.
- Neural: The GPT-3-small target shape is not on the board (the LM lanes use the smaller control shape so one shape runs on every box, a 16 GB Mac included).
- Neural, not planned on this vendor: torch-eager-tf32 / torch-compile-tf32: TF32 is an NVIDIA CUDA tensor-core matmul mode; torch on MPS accepts the flag and changes nothing
- Neural, not planned on this vendor: torch-compile-* on mamba1-forward and mamba1-infer: the only torch Mamba-1 twin is the pure-PyTorch reference scan, a per-token Python loop that torch.compile would unroll L times; mamba1 races the eager arms only
- Neural, not planned on this vendor: gemm-bf16 races torch's bf16 settings only (bf16 operands, fp32 accumulate)
- Neural, not planned on this vendor: gemm-int8: torch._int_mm is a CUDA kernel; torch on MPS has no int8 matmul, so ours races alone

