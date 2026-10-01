# mojolearn benchmark board

Generated 2026-09-29T23:22:49Z from `board.json` (schema `mojolearn-bench-board/1`).

## Box

| field | value |
|---|---|
| vendor / API | nvidia / cuda |
| GPU | NVIDIA L40S |
| GPU driver | 580.159.03 |
| CPU | AMD EPYC 9554 64-Core Processor (256 logical cores) |
| memory bytes | 1081799102464 |
| OS | Ubuntu 22.04.5 LTS |
| Python | 3.11.10 CPython |
| mojolearn | 0.8.25 (wheel mojolearn-0.8.25-py3-none-manylinux_2_35_x86_64.whl, sha256 bafc6196927055d47f9113bf90ed378fdcf895ce215aceafa24e8787cd974641) |
| script commit | 4f3e685348ddc41ec22ce00c53187c3528d929b7 |
| patch sync | synced commit 4f3e685348ddc41ec22ce00c53187c3528d929b7 over base f4e78b35406eda59f8417b3acaa27c2830dca16c, patch sha256 47576fc48f13c4cd6cc1e4f130be92eb24c82c4aaee319912fb27d47caafc719 |
| modes | identical |
| rounds | 1 timed after 1 warm-up, arms interleaved round by round |
| seed | 7 |
| opponent versions | catboost 1.2.10, xgboost 3.2.0, lightgbm 4.7.0, scikit-learn 1.7.2, cuml-cu12 26.8.0, cuvs-cu12 26.8.1, torch 2.13.0+cu129, numba 0.64.0, numpy 2.4.6 |

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

Races: 3 planned, 3 done, 0 failed, 0 pending. Cells: 24 (ok 24).

## Quality at a glance

Per lane and dataset: our FAST value, our IDENTICAL value, our CPU IDENTICAL value, and each opponent's.

| family | lane | dataset | metric | ours FAST | ours IDENTICAL | ours CPU | opponents |
|---|---|---|---|---|---|---|---|
| neural | mamba1-forward | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | mamba-ssm-fp32 1.192e-07; mamba-ssm-tf32 3.815e-06; torch-eager-bf16 5.15e-05; torch-eager-fp32 1.192e-07; torch-eager-tf32 3.815e-06 |
| neural | mamba1-forward | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | mamba-ssm-fp32 5.945e-08; mamba-ssm-tf32 1.902e-06; torch-eager-bf16 2.568e-05; torch-eager-fp32 5.945e-08; torch-eager-tf32 1.902e-06 |
| neural | mamba2-forward | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | mamba-ssm-fp32 1.907e-06; mamba-ssm-tf32 0.0005126; torch-compile-bf16 0.007544; torch-compile-fp32 2.146e-06; torch-compile-tf32 0.000509; torch-eager-bf16 0.007544; torch-eager-fp32 2.146e-06; torch-eager-tf32 0.000509 |
| neural | mamba2-forward | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | mamba-ssm-fp32 6.485e-07; mamba-ssm-tf32 0.0001743; torch-compile-bf16 0.002565; torch-compile-fp32 7.295e-07; torch-compile-tf32 0.0001731; torch-eager-bf16 0.002565; torch-eager-fp32 7.295e-07; torch-eager-tf32 0.0001731 |
| neural | mamba3-forward | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | mamba-ssm-fp32 0.001235; mamba-ssm-tf32 0.001376; torch-compile-bf16 0.001982; torch-compile-fp32 4.768e-07; torch-compile-tf32 0.0001798; torch-eager-bf16 0.002095; torch-eager-fp32 4.768e-07; torch-eager-tf32 0.0001827 |
| neural | mamba3-forward | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | mamba-ssm-fp32 0.0005401; mamba-ssm-tf32 0.0006018; torch-compile-bf16 0.0008672; torch-compile-fp32 2.086e-07; torch-compile-tf32 7.867e-05; torch-eager-bf16 0.0009164; torch-eager-fp32 2.086e-07; torch-eager-tf32 7.993e-05 |

## Neural

### mamba1-forward / gaussian (neural shape full: B1 L2048 DM384)

race: done, driver rc 0, log `logs/neural.mamba1-forward.gaussian.shape-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 9.9 | 9.9..9.9 | 1 | - | - | - | 198.0 | 908.0 | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mamba-ssm-fp32 | mamba-ssm | gpu | opponent | 3.5 | 3.5..3.5 | 1 | 2.832 | - | - | 1247.2 | 54.6 | max_abs_diff_vs_ours=1.192e-07, max_rel_diff_vs_ours=5.945e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| mamba-ssm-tf32 | mamba-ssm | gpu | opponent | 4.9 | 4.9..4.9 | 1 | 2.030 | - | - | 1225.3 | 54.6 | max_abs_diff_vs_ours=3.815e-06, max_rel_diff_vs_ours=1.902e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 195.9 | 195.9..195.9 | 1 | 0.051 | - | - | 1189.8 | 307.1 | max_abs_diff_vs_ours=5.15e-05, max_rel_diff_vs_ours=2.568e-05 | - | LIKE-FOR-LIKE-SPAN | - | ok (stored (measured 2026-09-29T17:33:43Z on 5785dd7f2fd2, gpu (NVIDIA L40S))) |
| torch-eager-fp32 | torch | gpu | opponent | 149.6 | 149.6..149.6 | 1 | 0.066 | - | - | 1062.5 | 354.3 | max_abs_diff_vs_ours=1.192e-07, max_rel_diff_vs_ours=5.945e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (stored (measured 2026-09-29T17:33:43Z on 5785dd7f2fd2, gpu (NVIDIA L40S))) |
| torch-eager-tf32 | torch | gpu | opponent | 153.5 | 153.5..153.5 | 1 | 0.065 | - | - | 1041.7 | 354.3 | max_abs_diff_vs_ours=3.815e-06, max_rel_diff_vs_ours=1.902e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (stored (measured 2026-09-29T17:33:43Z on 5785dd7f2fd2, gpu (NVIDIA L40S))) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, mamba-ssm-fp32, mamba-ssm-tf32, torch-eager-bf16, torch-eager-fp32, torch-eager-tf32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | mamba-ssm-fp32 | mamba-ssm-tf32 | ours |
|---|---||---|---||---|---|
| library (source) | mamba-ssm (declared) | mamba-ssm (declared) | mojolearn (declared) |
| seed | 7 | 7 | "none (deterministic)" |
| ssm_d_conv | 4 | 4 | 4 |
| ssm_d_state | 16 | 16 | 16 |
| ssm_dt_rank | 24 | 24 | 24 |
| ssm_expand | 2 | 2 | 2 |

### mamba2-forward / gaussian (neural shape full: B1 L2048 DM384)

race: done, driver rc 0, log `logs/neural.mamba2-forward.gaussian.shape-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 13.3 | 13.3..13.3 | 1 | - | - | - | 200.4 | 908.0 | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mamba-ssm-fp32 | mamba-ssm | gpu | opponent | 3.9 | 3.9..3.9 | 1 | 3.395 | - | - | 1460.8 | 58.6 | max_abs_diff_vs_ours=1.907e-06, max_rel_diff_vs_ours=6.485e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| mamba-ssm-tf32 | mamba-ssm | gpu | opponent | 2.9 | 2.9..2.9 | 1 | 4.578 | - | - | 1289.7 | 58.6 | max_abs_diff_vs_ours=0.0005126, max_rel_diff_vs_ours=0.0001743 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 21.7 | 21.7..21.7 | 1 | 0.614 | - | - | 1383.4 | 1692.8 | max_abs_diff_vs_ours=0.007544, max_rel_diff_vs_ours=0.002565 | - | LIKE-FOR-LIKE-SPAN | - | ok (stored (measured 2026-09-29T17:34:20Z on 5785dd7f2fd2, gpu (NVIDIA L40S))) |
| torch-compile-fp32 | torch | gpu | opponent | 26.1 | 26.1..26.1 | 1 | 0.511 | - | - | 1250.8 | 3228.4 | max_abs_diff_vs_ours=2.146e-06, max_rel_diff_vs_ours=7.295e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (stored (measured 2026-09-29T17:34:20Z on 5785dd7f2fd2, gpu (NVIDIA L40S))) |
| torch-compile-tf32 | torch | gpu | opponent | 26.5 | 26.5..26.5 | 1 | 0.504 | - | - | 1208.0 | 3228.4 | max_abs_diff_vs_ours=0.000509, max_rel_diff_vs_ours=0.0001731 | - | LIKE-FOR-LIKE-SPAN | - | ok (stored (measured 2026-09-29T17:34:20Z on 5785dd7f2fd2, gpu (NVIDIA L40S))) |
| torch-eager-bf16 | torch | gpu | opponent | 21.7 | 21.7..21.7 | 1 | 0.615 | - | - | 1326.0 | 1692.8 | max_abs_diff_vs_ours=0.007544, max_rel_diff_vs_ours=0.002565 | - | LIKE-FOR-LIKE-SPAN | - | ok (stored (measured 2026-09-29T17:34:20Z on 5785dd7f2fd2, gpu (NVIDIA L40S))) |
| torch-eager-fp32 | torch | gpu | opponent | 25.8 | 25.8..25.8 | 1 | 0.516 | - | - | 1192.2 | 3228.4 | max_abs_diff_vs_ours=2.146e-06, max_rel_diff_vs_ours=7.295e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (stored (measured 2026-09-29T17:34:20Z on 5785dd7f2fd2, gpu (NVIDIA L40S))) |
| torch-eager-tf32 | torch | gpu | opponent | 25.7 | 25.7..25.7 | 1 | 0.519 | - | - | 1181.2 | 3228.4 | max_abs_diff_vs_ours=0.000509, max_rel_diff_vs_ours=0.0001731 | - | LIKE-FOR-LIKE-SPAN | - | ok (stored (measured 2026-09-29T17:34:20Z on 5785dd7f2fd2, gpu (NVIDIA L40S))) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, mamba-ssm-fp32, mamba-ssm-tf32, torch-compile-bf16, torch-compile-fp32, torch-compile-tf32, torch-eager-bf16, torch-eager-fp32, torch-eager-tf32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | mamba-ssm-fp32 | mamba-ssm-tf32 | ours |
|---|---||---|---||---|---|
| library (source) | mamba-ssm (declared) | mamba-ssm (declared) | mojolearn (declared) |
| seed | 7 | 7 | "none (deterministic)" |
| ssm_chunk_size | 256 | 256 | 256 |
| ssm_d_conv | 4 | 4 | 4 |
| ssm_d_state | 128 | 128 | 128 |
| ssm_dt_limit | [0.0, Infinity] | [0.0, Infinity] | [0.0, Infinity] |
| ssm_expand | 2 | 2 | 2 |
| ssm_headdim | 64 | 64 | 64 |
| ssm_ngroups | 1 | 1 | 1 |

### mamba3-forward / gaussian (neural shape full: B1 L2048 DM384)

race: done, driver rc 0, log `logs/neural.mamba3-forward.gaussian.shape-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 5.7 | 5.7..5.7 | 1 | - | - | - | 1505.0 | 908.0 | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mamba-ssm-fp32 | mamba-ssm | gpu | opponent | 4.5 | 4.5..4.5 | 1 | 1.288 | - | - | 1414.9 | 70.0 | max_abs_diff_vs_ours=0.001235, max_rel_diff_vs_ours=0.0005401 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| mamba-ssm-tf32 | mamba-ssm | gpu | opponent | 4.1 | 4.1..4.1 | 1 | 1.400 | - | - | 1410.9 | 70.0 | max_abs_diff_vs_ours=0.001376, max_rel_diff_vs_ours=0.0006018 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 9.1 | 9.1..9.1 | 1 | 0.633 | - | - | 3809.6 | 90.5 | max_abs_diff_vs_ours=0.001982, max_rel_diff_vs_ours=0.0008672 | - | LIKE-FOR-LIKE-SPAN | - | ok (stored (measured 2026-09-29T17:51:09Z on 5785dd7f2fd2, gpu (NVIDIA L40S))) |
| torch-compile-fp32 | torch | gpu | opponent | 13.5 | 13.5..13.5 | 1 | 0.424 | - | - | 3465.6 | 83.6 | max_abs_diff_vs_ours=4.768e-07, max_rel_diff_vs_ours=2.086e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (stored (measured 2026-09-29T17:51:09Z on 5785dd7f2fd2, gpu (NVIDIA L40S))) |
| torch-compile-tf32 | torch | gpu | opponent | 9.3 | 9.3..9.3 | 1 | 0.614 | - | - | 3068.5 | 83.6 | max_abs_diff_vs_ours=0.0001798, max_rel_diff_vs_ours=7.867e-05 | - | LIKE-FOR-LIKE-SPAN | - | ok (stored (measured 2026-09-29T17:51:09Z on 5785dd7f2fd2, gpu (NVIDIA L40S))) |
| torch-eager-bf16 | torch | gpu | opponent | 83.8 | 83.8..83.8 | 1 | 0.068 | - | - | 1313.7 | 212.6 | max_abs_diff_vs_ours=0.002095, max_rel_diff_vs_ours=0.0009164 | - | LIKE-FOR-LIKE-SPAN | - | ok (stored (measured 2026-09-29T17:51:09Z on 5785dd7f2fd2, gpu (NVIDIA L40S))) |
| torch-eager-fp32 | torch | gpu | opponent | 83.7 | 83.7..83.7 | 1 | 0.069 | - | - | 1190.9 | 219.7 | max_abs_diff_vs_ours=4.768e-07, max_rel_diff_vs_ours=2.086e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (stored (measured 2026-09-29T17:51:09Z on 5785dd7f2fd2, gpu (NVIDIA L40S))) |
| torch-eager-tf32 | torch | gpu | opponent | 83.9 | 83.9..83.9 | 1 | 0.068 | - | - | 1186.6 | 219.7 | max_abs_diff_vs_ours=0.0001827, max_rel_diff_vs_ours=7.993e-05 | - | LIKE-FOR-LIKE-SPAN | - | ok (stored (measured 2026-09-29T17:51:09Z on 5785dd7f2fd2, gpu (NVIDIA L40S))) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, mamba-ssm-fp32, mamba-ssm-tf32, torch-compile-bf16, torch-compile-fp32, torch-compile-tf32, torch-eager-bf16, torch-eager-fp32, torch-eager-tf32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | mamba-ssm-fp32 | mamba-ssm-tf32 | ours |
|---|---||---|---||---|---|
| library (source) | mamba-ssm (declared) | mamba-ssm (declared) | mojolearn (declared) |
| seed | 7 | 7 | "none (deterministic)" |
| ssm_chunk_size | 64 | 64 | 64 |
| ssm_d_state | 128 | 128 | 128 |
| ssm_expand | 2 | 2 | 2 |
| ssm_headdim | 64 | 64 | 64 |
| ssm_ngroups | 1 | 1 | 1 |
| ssm_rope_angles | 32 | 32 | 32 |

## Not covered by this board

- Classical, wave 2: RadiusNeighbors, the preprocessing scalers, HDBSCAN's prediction data, Cholesky and the parallel_* and Distributed* wrappers are public and not raced here; taxi-derived time series are not used (the ARIMA and ExponentialSmoothing lanes fit seeded synthetic series, as the repo's own ARIMA quality work does).
- Classical, wave 2, not planned on this vendor: cuML GaussianMixture, GaussianProcessRegressor/Classifier, Nystroem, RBFSampler: cuML 26.8.0 has none; scikit-learn on the CPU is the arm.
- Classical, wave 2, not planned on this vendor: faiss-gpu: no pinned PyPI wheel for this image; cuVS ivf_flat from the pinned rapids set (cuvs-cu12==26.8.1) is the IVF-Flat GPU arm.
- Classical, wave 2, not planned on this vendor: umap-learn and faiss-cpu: not installed on NVIDIA; cuML UMAP and cuVS are the arms.
- Classical, wave 2, not planned on this vendor: cuML SpectralClustering/SpectralEmbedding: present only in newer cuML; if the pinned 26.8.0 lacks them the arm refuses by name and scikit-learn stands beside it.
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
- Neural, not planned on this vendor: torch-compile-* on mamba1-forward and mamba1-infer: the only torch Mamba-1 twin is the pure-PyTorch reference scan, a per-token Python loop that torch.compile would unroll L times; mamba1 races the eager arms only
- Neural, not planned on this vendor: gemm-bf16 races torch's bf16 settings only (bf16 operands, fp32 accumulate)
- Neural, not planned on this vendor: mamba-ssm bf16: our Mamba blocks are float32, so mamba_ssm races in float32 (and its TF32 setting); torch's bf16 arms carry the lower precision

