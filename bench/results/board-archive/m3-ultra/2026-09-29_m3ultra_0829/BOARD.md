# mojolearn benchmark board

Generated 2026-09-30T00:44:46Z from `board.json` (schema `mojolearn-bench-board/1`).

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
| mojolearn | 0.8.29 (wheel mojolearn-0.8.29-py3-none-macosx_11_0_arm64.whl, sha256 d758f5b80e7084b42a1fbc6fa8dd6f2e75a948e17447c21bf9df0d151cc9244a) |
| script commit | 9cee9ef68a90029b6a158ba7da2fa1d4fa064ee5 |
| patch sync | - |
| modes | fast, identical |
| rounds | 1 timed after 1 warm-up, arms interleaved round by round |
| seed | 7 |
| opponent versions | catboost 1.2.10, xgboost 3.2.0, lightgbm 4.7.0, scikit-learn 1.7.2, torch 2.13.0, numba 0.67.0, statsmodels 0.15.0, faiss-cpu 1.15.1, numpy 2.5.3 |

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

Races: 1 planned, 3 done, 1 failed, 0 pending. Cells: 12 (HOST-MEMORY 1, REFUSED 1, ok 10).

Inference cells: 6 (ok 6).

## Quality at a glance

Per lane and dataset: our FAST value, our IDENTICAL value, our CPU IDENTICAL value, and each opponent's.

| family | lane | dataset | metric | ours FAST | ours IDENTICAL | ours CPU | opponents |
|---|---|---|---|---|---|---|---|
| algos | bayesian-gmm | istella | mean_log_likelihood (higher is better) | 175.561762 | 175.322029 | - | sklearn-cpu 201.770716 |
| algos | bayesian-gmm | taxi | mean_log_likelihood (higher is better) | 4.895586 | 4.895581 | - | sklearn-cpu 6.178321 |
| classical | dbscan | istella | n_clusters | 40131 | 40131 | - | sklearn-cpu 40131 |
| classical | dbscan | istella | noise_fraction | 0.219391 | 0.219391 | - | sklearn-cpu 0.219391 |
| classical | dbscan | istella | rows | 1000000 | 1000000 | - | sklearn-cpu 1000000 |
| classical | dbscan | istella | ari_vs_ours (1 is our partition exactly) | 1.000000 | - | - | sklearn-cpu 1.000000 |
| classical | dbscan | istella | noise_agreement_vs_ours | 1.000000 | - | - | sklearn-cpu 1.000000 |
| classical | dbscan | taxi | n_clusters | 36 | - | - | sklearn-cpu - |
| classical | dbscan | taxi | noise_fraction | 0.000174 | - | - | sklearn-cpu - |
| classical | dbscan | taxi | rows | 1000000 | - | - | sklearn-cpu - |

## Inference at a glance

Batch prediction, each arm with its own fitted model from the same race; medians in ms. `FAST = IDENTICAL bits` compares our two tiers' predictions on the same rows; `CPU = IDENTICAL bits` compares our CPU tier's with our GPU IDENTICAL arm's.

| family | lane | dataset | batch | rows | ours FAST ms | ours IDENTICAL ms | FAST = IDENTICAL bits | ours CPU ms | CPU = IDENTICAL bits | opponents |
|---|---|---|---|---|---|---|---|---|---|---|
| algos | bayesian-gmm | istella | Xq | - | 55.2 | 55.9 | - | - | - | sklearn-cpu 89.6 ms (IDENTICAL/arm 0.624) |
| algos | bayesian-gmm | taxi | Xq | - | 13.6 | 13.7 | - | - | - | sklearn-cpu 5.9 ms (IDENTICAL/arm 2.302) |

## Classical

### dbscan / istella (rows full, shape 1000000x220)

race: done, driver rc 0, log `logs/classical.dbscan.istella.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 255151.4 | 255151.4..255151.4 | 1 | - | - | - | 4154.2 | - | n_clusters=40131, noise_fraction=0.219391, rows=1000000 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 247363.7 | 247363.7..247363.7 | 1 | - | - | - | 4119.5 | - | ari_vs_ours=1.000000, n_clusters=40131, noise_agreement_vs_ours=1.000000, noise_fraction=0.219391, rows=1000000 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 248662.7 | 248662.7..248662.7 | 1 | 1.026 | 0.995 | - | 5648.5 | - | ari_vs_ours=1.000000, n_clusters=40131, noise_agreement_vs_ours=1.000000, noise_fraction=0.219391, rows=1000000 | yes | LIKE-FOR-LIKE-SPAN | - | ok (stored (measured 2026-09-29T18:37:38Z on ip-172-31-37-211.ec2.internal, cpu (Apple M3 Ultra))) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: eps=3, min_samples=2 (the cuML benchmark's DBSCAN) on every arm; metric='euclidean'. Rows: dbscan block: 1,000,000 rows, standardized. Timed: fit.

mismatch: seed: no arm has a seed argument (deterministic)

mismatch: algorithm (an exact neighbor search on every arm, results unchanged): ours 'rbc' (its default), scikit-learn 'brute' (the cuML benchmark's cpu_args; it has no 'rbc'), cuml-gpu 'brute', cuml-gpu-rbc 'rbc'

mismatch: leaf_size=30 and n_jobs=-1: scikit-learn only

config: cuML benchmark (RAPIDS), DBSCAN (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast |
|---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) |
| algorithm | "rbc" | "rbc" |
| eps | 3.0 | 3.0 |
| metric | "euclidean" | "euclidean" |
| min_samples | 2 | 2 |
| seed | "none (deterministic)" | "none (deterministic)" |

### dbscan / taxi (rows full, shape 1000000x11)

race: failed, driver rc 0, log `logs/classical.dbscan.taxi.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(timeout: null) |
| mojolearn FAST | mojolearn | gpu | fast | 758590.2 | 758590.2..758590.2 | 1 | - | - | - | 8888.6 | - | n_clusters=36, noise_fraction=0.000174, rows=1000000 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | HOST-MEMORY(killed at 230.2 GB: the driver's process tree held 247.5 GB, over 90% of the box's 274.9 GB) (measured this run) |

memory, ours, sklearn-cpu: host not sampled; GPU not sampled

memory, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

settings: eps=3, min_samples=2 (the cuML benchmark's DBSCAN) on every arm; metric='euclidean'. Rows: dbscan block: 1,000,000 rows, standardized. Timed: fit.

mismatch: seed: no arm has a seed argument (deterministic)

mismatch: algorithm (an exact neighbor search on every arm, results unchanged): ours 'rbc' (its default), scikit-learn 'brute' (the cuML benchmark's cpu_args; it has no 'rbc'), cuml-gpu 'brute', cuml-gpu-rbc 'rbc'

mismatch: leaf_size=30 and n_jobs=-1: scikit-learn only

config: cuML benchmark (RAPIDS), DBSCAN (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| algorithm | "rbc" | "rbc" | "brute" |
| eps | 3.0 | 3.0 | 3.0 |
| leaf_size | - | - | 30 |
| metric | "euclidean" | "euclidean" | "euclidean" |
| min_samples | 2 | 2 | 2 |
| p | - | - | null |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

accepted difference: sklearn-cpu algorithm: an exact eps search on every arm: ours 'rbc' (its default), scikit-learn has no 'rbc' and runs 'brute' (the cuML benchmark's cpu_args)

## Algorithm expansion

### bayesian-gmm / istella (rows full, shape X 100000x200; Xq 20000x200; y 100000; yq 20000)

race: done, driver rc 0, log `logs/algos.bayesian-gmm.istella.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 39953.5 | 39953.5..39953.5 | 1 | - | - | - | 2168.7 | - | mean_log_likelihood=175.322029 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 70783.5 | 70783.5..70783.5 | 1 | - | - | - | 1917.3 | - | mean_log_likelihood=175.561762 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 82775.7 | 82775.7..82775.7 | 1 | 0.483 | 0.855 | - | 1441.8 | - | mean_log_likelihood=201.770716 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| reg_covar | 0.003 | 0.003 | 0.003 |
| seed | 7 | 7 | 7 |
| tol | 0.001 | 0.001 | 0.001 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 55.9 | 55.9..55.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 55.2 | 55.2..55.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 89.6 | 89.6..89.6 | 1 | 0.624 | 0.616 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### bayesian-gmm / taxi (rows full, shape X 100000x11; Xq 20000x11; y 100000; yq 20000)

race: done, driver rc 0, log `logs/algos.bayesian-gmm.taxi.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 2630.8 | 2630.8..2630.8 | 1 | - | - | - | 449.4 | - | mean_log_likelihood=4.895581 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 444.9 | 444.9..444.9 | 1 | - | - | - | 424.3 | - | mean_log_likelihood=4.895586 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 4017.7 | 4017.7..4017.7 | 1 | 0.655 | 0.111 | - | 258.3 | - | mean_log_likelihood=6.178321 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 13.7 | 13.7..13.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 13.6 | 13.6..13.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 5.9 | 5.9..5.9 | 1 | 2.302 | 2.289 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

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
- Neural: The Mamba torch-* arms are the repo's pure-PyTorch references (mamba/corpus/gen_corpus.py: mamba_ssm's selective_scan_ref for Mamba-1, the chunked SSD reference for Mamba-2, the SISO reference for Mamba-3), not deployment kernels. On NVIDIA the mamba-ssm-* arms are mamba_ssm's own fused CUDA/Triton kernels (the deployment path); on AMD and Apple the Mamba lanes race the references only (NOT_PLANNED says why).
- Neural: The blocks' backward (the Mamba and TransformerBlock VJPs), their decode `step`, ragged `lengths` and the carried-state forward are public and not raced; only a zero-state forward is.
- Neural: SmallMLPTrainer.predict_logits (the GPU forward of the 8-16-3 MLP) is not raced; MLPInference (its CPU forward) and the training step are.
- Neural: The GPT-3-small target shape is not on the board (the LM lanes use the smaller control shape so one shape runs on every box, a 16 GB Mac included).
- Neural, not planned on this vendor: torch-eager-tf32 / torch-compile-tf32: TF32 is an NVIDIA CUDA tensor-core matmul mode; torch on MPS accepts the flag and changes nothing
- Neural, not planned on this vendor: torch-compile-* on mamba1-forward and mamba1-infer: the only torch Mamba-1 twin is the pure-PyTorch reference scan, a per-token Python loop that torch.compile would unroll L times; mamba1 races the eager arms only
- Neural, not planned on this vendor: gemm-bf16 races torch's bf16 settings only (bf16 operands, fp32 accumulate)
- Neural, not planned on this vendor: gemm-int8: torch._int_mm is a CUDA kernel; torch on MPS has no int8 matmul, so ours races alone
- Neural, not planned on this vendor: mamba-ssm-* on mamba*-forward: mamba_ssm's kernels are CUDA and Triton (no Metal build exists); the Mamba lanes race the torch reference arms
- Neural, not planned on this vendor: mamba-ssm bf16: our Mamba blocks are float32, so mamba_ssm races in float32 (and its TF32 setting); torch's bf16 arms carry the lower precision

