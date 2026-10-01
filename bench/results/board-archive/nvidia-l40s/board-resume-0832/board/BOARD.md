# mojolearn benchmark board

Generated 2026-10-01T03:35:01Z from `board.json` (schema `mojolearn-bench-board/1`).

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
| mojolearn | 0.8.32 (wheel mojolearn-0.8.32-py3-none-manylinux_2_35_x86_64.whl, sha256 0c7103babf28517fd6cfa50a5f62d38242e0d3fb477c5ce3a8714cf3f03e8239) |
| script commit | a19d159d7d3a7322d0159e6c3482de29c4464109 |
| patch sync | synced commit a19d159d7d3a7322d0159e6c3482de29c4464109 over base f4e78b35406eda59f8417b3acaa27c2830dca16c, patch sha256 cec3b66754e9e311b788e8168332d4930d202af37cac09ab3903a72bb26b39ee |
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
- Neural: our IDENTICAL arm only (the neural surface builds no other tier, on any vendor) against torch at every fast setting it supports on this box, one arm each, the setting in the arm name: `torch-eager-fp32` (TF32 off), `torch-compile-fp32` (torch.compile, inductor), `torch-eager-tf32` / `torch-compile-tf32` (NVIDIA CUDA only), `torch-eager-bf16` / `torch-compile-bf16` (bf16 autocast mixed precision). TF32 and bf16 arms are ANOTHER PRECISION than ours; their quality columns show how far. The `*-infer` lanes are the CPU *Inference classes and race `torch-cpu-*` arms. An arm torch cannot run on this box is REFUSED by name in its cell. Every clock is host in, host out, synchronized. Every arm starts from the same parameters and reads the same inputs, so losses and outputs are comparable; `max_abs_diff_vs_ours` / `max_rel_diff_vs_ours` are the arm's output against ours.
- `installed_wheel` confirms our binding loaded from site-packages, not the repo tree.
- Our CPU tier (`mojolearn CPU IDENTICAL`, arm `ours-cpu`): the same public estimator in a worker started under MOJOLEARN_VENDOR=cpu, the wheel's CPU switch (no GPU set loads; the host bindings answer, IDENTICAL only), read back as vendor cpu or refused by name. It races in the same rounds as every arm; `ours CPU / arm` is its median over each opponent's. `bits_equal_vs_ours_identical` compares its output with our GPU IDENTICAL arm's, bit for bit. Our CPU and GPU times are never divided by each other here.
- Memory: `peak host MB` and `peak GPU MB` are the highest per-round peaks over the timed rounds, read outside the clock; each arm's method is listed under its table (host: the resettable peak RSS on Linux, the peak physical footprint on macOS, which holds Metal buffers too; GPU: torch's own counter for torch arms, the driver's per-process figure for the rest, none on Apple).
- Inference: after a race's fit rounds each arm predicts with its own fitted model (no fit retimed), same rows, same output kind, one warm-up then the timed rounds interleaved. Trees: batch `test` (the held-out split) and `large` (1,000,000 training rows, capped at the training rows), host rows in and host predictions out on every arm; each arm's call is printed under its table. Classical: kmeans predict, pca transform, ols predict and svc predict on the eval rows, with the fit's clock span. Ratios are per batch, ours over each opponent.

## Coverage

Races: 1 planned, 15 done, 2 failed, 0 pending. Cells: 38 (MODE-MISMATCH 5, REFUSED 6, ok 27).

## Quality at a glance

Per lane and dataset: our FAST value, our IDENTICAL value, our CPU IDENTICAL value, and each opponent's.

| family | lane | dataset | metric | ours FAST | ours IDENTICAL | ours CPU | opponents |
|---|---|---|---|---|---|---|---|
| algos | bpe-encode | enwik8 | documents_equal_to_ours | - | 1.000000 | - | hf-tokenizers-cpu 1.000000 |
| algos | bpe-encode | enwik8 | tokens | - | 1457323 | - | hf-tokenizers-cpu 1457323 |
| algos | bpe-train | enwik8 | jaccard_vs_ours | - | 1.000000 | - | hf-tokenizers-cpu 0.999512 |
| algos | bpe-train | enwik8 | n_tokens | - | 4096 | - | hf-tokenizers-cpu 4096 |
| algos | clip-grad-norm | synthetic | norm | - | 1.000000 | - | torch-eager-fp32 1.000000; torch-compile-fp32 1.000000 |
| algos | clip-grad-norm | synthetic | norm_rel_diff_vs_ours | - | 0.000000 | - | torch-eager-fp32 1.192e-07; torch-compile-fp32 1.192e-07 |
| algos | cross-entropy | synthetic | loss_rel_err_vs_fp64 | - | 4.564e-08 | - | torch-eager-fp32 5.477e-08; torch-compile-fp32 4.564e-08 |
| algos | cross-entropy | synthetic | grad_max_rel_diff_vs_ours | - | - | - | torch-eager-fp32 4.657e-09; torch-compile-fp32 4.657e-09 |
| algos | jl-min-dim | synthetic | equal_fraction_vs_sklearn | - | 1.000000 | - | sklearn-cpu 1.000000 |
| algos | kpss | synthetic | flag_agreement_vs_statsmodels | - | 1.000000 | - | statsmodels-cpu 1.000000 |
| algos | kpss | synthetic | stat_max_rel_diff_vs_statsmodels | - | 4.318e-05 | - | statsmodels-cpu 0.000000 |
| algos | kpss | synthetic | stationary_fraction | - | 0.031250 | - | statsmodels-cpu 0.031250 |
| algos | kpss | taxi-hourly | flag_agreement_vs_statsmodels | - | 1.000000 | - | statsmodels-cpu 1.000000 |
| algos | kpss | taxi-hourly | stat_max_rel_diff_vs_statsmodels | - | 3.946e-05 | - | statsmodels-cpu 0.000000 |
| algos | kpss | taxi-hourly | stationary_fraction | - | 0.687500 | - | statsmodels-cpu 0.687500 |
| algos | lr-constant | synthetic | max_rel_diff_vs_ours (0 is our output exactly) | - | 0.000000 | - | torch-cpu 1.038e-07 |
| algos | lr-exponential | synthetic | max_rel_diff_vs_ours (0 is our output exactly) | - | 0.000000 | - | torch-cpu 5.933e-08 |
| algos | lr-onecycle | synthetic | max_rel_diff_vs_ours (0 is our output exactly) | - | 0.000000 | - | torch-cpu 5.951e-08 |
| algos | lr-step | synthetic | max_rel_diff_vs_ours (0 is our output exactly) | - | 0.000000 | - | torch-cpu 1.49e-08 |
| algos | lr-warmup-cosine | synthetic | max_rel_diff_vs_ours (0 is our output exactly) | - | 0.000000 | - | torch-cpu 0.0001571 |
| algos | lr-warmup-linear | synthetic | max_rel_diff_vs_ours (0 is our output exactly) | - | 0.000000 | - | torch-cpu 0.001000 |
| algos | select-d | synthetic | d_agreement_vs_statsmodels | - | 1.000000 | - | statsmodels-cpu 1.000000 |
| algos | select-d | taxi-hourly | d_agreement_vs_statsmodels | - | 1.000000 | - | statsmodels-cpu 1.000000 |

## Algorithm expansion

### bpe-encode / enwik8 (rows full, shape -)

race: done, driver rc 0, log `logs/algos.bpe-encode.enwik8.rows-full.log`, ran on f003873fb257

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 141.2 | 141.2..141.2 | 1 | - | - | - | 243.7 | - | documents_equal_to_ours=1.000000, tokens=1457323 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| hf-tokenizers-cpu | tokenizers | cpu | opponent | 236.9 | 236.9..236.9 | 1 | 0.596 | - | - | 590.6 | - | documents_equal_to_ours=1.000000, tokens=1457323 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi lists no compute app with this pid (a container's pid namespace hides it, or no context was opened)

memory, hf-tokenizers-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'min_frequency': 2, 'vocab_size': 4096}. Rows: None. Timed: None.

mismatch: bpe-train: each library breaks count ties by its own rule

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | hf-tokenizers-cpu | ours |
|---|---||---|---|
| library (source) | tokenizers (declared) | mojolearn (declared) |
| seed | "none (deterministic)" | "none (deterministic)" |

### bpe-train / enwik8 (rows full, shape -)

race: done, driver rc 0, log `logs/algos.bpe-train.enwik8.rows-full.log`, ran on f003873fb257

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 81.4 | 81.4..81.4 | 1 | - | - | - | 129.7 | - | jaccard_vs_ours=1.000000, n_tokens=4096 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| hf-tokenizers-cpu | tokenizers | cpu | opponent | 561.1 | 561.1..561.1 | 1 | 0.145 | - | - | 246.2 | - | jaccard_vs_ours=0.999512, n_tokens=4096 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi lists no compute app with this pid (a container's pid namespace hides it, or no context was opened)

memory, hf-tokenizers-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'min_frequency': 2, 'vocab_size': 4096}. Rows: None. Timed: None.

mismatch: bpe-train: each library breaks count ties by its own rule

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | hf-tokenizers-cpu | ours |
|---|---||---|---|
| library (source) | tokenizers (declared) | mojolearn (declared) |
| seed | "none (deterministic)" | "none (deterministic)" |

### clip-grad-norm / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.clip-grad-norm.synthetic.rows-full.log`, ran on f003873fb257

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 8.8 | 8.8..8.8 | 1 | - | - | - | 1569.2 | 682.0 | norm=1.000000, norm_rel_diff_vs_ours=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested identical, read back unknown) |
| torch-eager-fp32 | torch | gpu | opponent | 1.0 | 1.0..1.0 | 1 | - | - | - | 977.7 | 128.0 | norm=1.000000, norm_rel_diff_vs_ours=1.192e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 0.8 | 0.8..0.8 | 1 | - | - | - | 1168.8 | 128.0 | norm=1.000000, norm_rel_diff_vs_ours=1.192e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'error_if_nonfinite': True, 'max_norm': 1.0, 'norm_type': 2.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-fp32 | torch-eager-fp32 |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | 7 | 7 |

### cross-entropy / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.cross-entropy.synthetic.rows-full.log`, ran on f003873fb257

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 524.6 | 524.6..524.6 | 1 | - | - | - | 2355.2 | 1962.0 | loss_rel_err_vs_fp64=4.564e-08 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested identical, read back unknown) |
| torch-eager-fp32 | torch | gpu | opponent | 7.6 | 7.6..7.6 | 1 | - | - | - | 825.2 | 1024.1 | grad_max_rel_diff_vs_ours=4.657e-09, loss_rel_err_vs_fp64=5.477e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 9.5 | 9.5..9.5 | 1 | - | - | - | 1063.2 | 512.1 | grad_max_rel_diff_vs_ours=4.657e-09, loss_rel_err_vs_fp64=4.564e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'ignore_index': -100, 'label_smoothing': 0.0, 'reduction': 'mean'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-fp32 | torch-eager-fp32 |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | 7 | 7 |

### jl-min-dim / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.jl-min-dim.synthetic.rows-full.log`, ran on f003873fb257

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 8.2 | 8.2..8.2 | 1 | - | - | - | 89.3 | - | equal_fraction_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 275.9 | 275.9..275.9 | 1 | 0.030 | - | - | 201.0 | - | equal_fraction_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi lists no compute app with this pid (a container's pid namespace hides it, or no context was opened)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (declared) | sklearn (declared) |
| seed | "none (deterministic)" | "none (deterministic)" |

### kpss / synthetic (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc 0, log `logs/algos.kpss.synthetic.rows-full.log`, ran on f003873fb257

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1.5 | 1.5..1.5 | 1 | - | - | - | 1499.9 | 682.0 | flag_agreement_vs_statsmodels=1.000000, stat_max_rel_diff_vs_statsmodels=4.318e-05, stationary_fraction=0.031250 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsmodels-cpu | statsmodels | cpu | opponent | 3.4 | 3.4..3.4 | 1 | 0.436 | - | - | 213.0 | - | flag_agreement_vs_statsmodels=1.000000, stat_max_rel_diff_vs_statsmodels=0.000000, stationary_fraction=0.031250 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, statsmodels-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'D': 0, 'd': 0, 'pval_threshold': 0.05, 's': 0}. Rows: None. Timed: None.

mismatch: statsmodels interpolates the p-value in its table and computes in float64; ours decides against cuML's table in float32

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | statsmodels-cpu |
|---|---||---|---|
| library (source) | mojolearn (declared) | statsmodels (declared) |
| seed | "none (deterministic)" | "none (deterministic)" |

### kpss / taxi-hourly (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc 0, log `logs/algos.kpss.taxi-hourly.rows-full.log`, ran on f003873fb257

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1.6 | 1.6..1.6 | 1 | - | - | - | 1499.8 | 682.0 | flag_agreement_vs_statsmodels=1.000000, stat_max_rel_diff_vs_statsmodels=3.946e-05, stationary_fraction=0.687500 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsmodels-cpu | statsmodels | cpu | opponent | 3.5 | 3.5..3.5 | 1 | 0.461 | - | - | 213.6 | - | flag_agreement_vs_statsmodels=1.000000, stat_max_rel_diff_vs_statsmodels=0.000000, stationary_fraction=0.687500 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, statsmodels-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'D': 0, 'd': 0, 'pval_threshold': 0.05, 's': 0}. Rows: None. Timed: None.

mismatch: statsmodels interpolates the p-value in its table and computes in float64; ours decides against cuML's table in float32

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | statsmodels-cpu |
|---|---||---|---|
| library (source) | mojolearn (declared) | statsmodels (declared) |
| seed | "none (deterministic)" | "none (deterministic)" |

### lr-constant / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.lr-constant.synthetic.rows-full.log`, ran on f003873fb257

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 905.1 | 905.1..905.1 | 1 | - | - | - | 89.9 | - | max_rel_diff_vs_ours=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested identical, read back unknown) |
| torch-cpu | torch | cpu | opponent | 147.3 | 147.3..147.3 | 1 | - | - | - | 793.8 | - | max_rel_diff_vs_ours=1.038e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi lists no compute app with this pid (a container's pid namespace hides it, or no context was opened)

memory, torch-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'peak_lr': 0.001, 'warmup_steps': 1000}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-cpu |
|---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) |
| seed | "none (deterministic)" | "none (deterministic)" |

### lr-exponential / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.lr-exponential.synthetic.rows-full.log`, ran on f003873fb257

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 312.1 | 312.1..312.1 | 1 | - | - | - | 90.0 | - | max_rel_diff_vs_ours=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-cpu | torch | cpu | opponent | 138.1 | 138.1..138.1 | 1 | 2.260 | - | - | 794.1 | - | max_rel_diff_vs_ours=5.933e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi lists no compute app with this pid (a container's pid namespace hides it, or no context was opened)

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

race: done, driver rc 0, log `logs/algos.lr-onecycle.synthetic.rows-full.log`, ran on f003873fb257

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1855.8 | 1855.8..1855.8 | 1 | - | - | - | 89.8 | - | max_rel_diff_vs_ours=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-cpu | torch | cpu | opponent | 191.7 | 191.7..191.7 | 1 | 9.683 | - | - | 794.3 | - | max_rel_diff_vs_ours=5.951e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi lists no compute app with this pid (a container's pid namespace hides it, or no context was opened)

memory, torch-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'anneal_strategy': 'cos', 'div_factor': 25.0, 'final_div_factor': 10000.0, 'max_lr': 0.1, 'pct_start': 0.3, 'three_phase': False, 'total_steps': 100000}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-cpu |
|---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) |
| seed | "none (deterministic)" | "none (deterministic)" |

### lr-step / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.lr-step.synthetic.rows-full.log`, ran on f003873fb257

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 29.9 | 29.9..29.9 | 1 | - | - | - | 87.5 | - | max_rel_diff_vs_ours=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-cpu | torch | cpu | opponent | 144.8 | 144.8..144.8 | 1 | 0.206 | - | - | 793.6 | - | max_rel_diff_vs_ours=1.49e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi lists no compute app with this pid (a container's pid namespace hides it, or no context was opened)

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

race: done, driver rc 0, log `logs/algos.lr-warmup-cosine.synthetic.rows-full.log`, ran on f003873fb257

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 797.5 | 797.5..797.5 | 1 | - | - | - | 89.6 | - | max_rel_diff_vs_ours=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested identical, read back unknown) |
| torch-cpu | torch | cpu | opponent | 181.9 | 181.9..181.9 | 1 | - | - | - | 794.0 | - | max_rel_diff_vs_ours=0.0001571 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi lists no compute app with this pid (a container's pid namespace hides it, or no context was opened)

memory, torch-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'min_lr': 1e-05, 'peak_lr': 0.001, 'total_steps': 100000, 'warmup_steps': 1000}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-cpu |
|---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) |
| seed | "none (deterministic)" | "none (deterministic)" |

### lr-warmup-linear / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.lr-warmup-linear.synthetic.rows-full.log`, ran on f003873fb257

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1375.6 | 1375.6..1375.6 | 1 | - | - | - | 89.9 | - | max_rel_diff_vs_ours=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | MODE-MISMATCH(requested identical, read back unknown) |
| torch-cpu | torch | cpu | opponent | 166.2 | 166.2..166.2 | 1 | - | - | - | 794.8 | - | max_rel_diff_vs_ours=0.001000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi lists no compute app with this pid (a container's pid namespace hides it, or no context was opened)

memory, torch-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'min_lr': 1e-05, 'peak_lr': 0.001, 'total_steps': 100000, 'warmup_steps': 1000}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-cpu |
|---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) |
| seed | "none (deterministic)" | "none (deterministic)" |

### qn-reg / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: failed, driver rc 3, log `logs/algos.qn-reg.istella.rows-full.log`, ran on f003873fb257

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (2): sklearn-cpu: tol is 0.0001 (tol) on ours and 1e-06 (tol) on sklearn-cpu; cuml-gpu: loss is 'squared_error' (loss) on ours and 'l2' (loss) on cuml-gpu") |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (2): sklearn-cpu: tol is 0.0001 (tol) on ours and 1e-06 (tol) on sklearn-cpu; cuml-gpu: loss is 'squared_error' (loss) on ours and 'l2' (loss) on cuml-gpu") (measured this run) |
| cuml-gpu | cuml | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (2): sklearn-cpu: tol is 0.0001 (tol) on ours and 1e-06 (tol) on sklearn-cpu; cuml-gpu: loss is 'squared_error' (loss) on ours and 'l2' (loss) on cuml-gpu") (measured this run) |

settings: {'fit_intercept': True, 'l1_strength': 0.0, 'l2_strength': 0.0, 'lbfgs_memory': 5, 'linesearch_max_iter': 50, 'loss': 'squared_error', 'max_iter': 1000, 'penalty_normalized': True, 'tol': 0.0001}. Rows: None. Timed: None.

mismatch: scikit-learn LinearRegression solves the same least-squares problem in closed form (scipy lstsq); it has no max_iter, tol or L-BFGS settings

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | cuml-gpu | ours | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (declared) | sklearn (get_params) |
| fit_intercept | true | true | true |
| loss | "l2" | "squared_error" | - |
| max_iter | 1000 | 1000 | - |
| positive | - | - | false |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| tol | 0.0001 | 0.0001 | 1e-06 |

REFUSED: sklearn-cpu: tol is 0.0001 (tol) on ours and 1e-06 (tol) on sklearn-cpu

REFUSED: cuml-gpu: loss is 'squared_error' (loss) on ours and 'l2' (loss) on cuml-gpu

### qn-reg / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: failed, driver rc 3, log `logs/algos.qn-reg.taxi.rows-full.log`, ran on f003873fb257

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (2): sklearn-cpu: tol is 0.0001 (tol) on ours and 1e-06 (tol) on sklearn-cpu; cuml-gpu: loss is 'squared_error' (loss) on ours and 'l2' (loss) on cuml-gpu") |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (2): sklearn-cpu: tol is 0.0001 (tol) on ours and 1e-06 (tol) on sklearn-cpu; cuml-gpu: loss is 'squared_error' (loss) on ours and 'l2' (loss) on cuml-gpu") (measured this run) |
| cuml-gpu | cuml | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (2): sklearn-cpu: tol is 0.0001 (tol) on ours and 1e-06 (tol) on sklearn-cpu; cuml-gpu: loss is 'squared_error' (loss) on ours and 'l2' (loss) on cuml-gpu") (measured this run) |

settings: {'fit_intercept': True, 'l1_strength': 0.0, 'l2_strength': 0.0, 'lbfgs_memory': 5, 'linesearch_max_iter': 50, 'loss': 'squared_error', 'max_iter': 1000, 'penalty_normalized': True, 'tol': 0.0001}. Rows: None. Timed: None.

mismatch: scikit-learn LinearRegression solves the same least-squares problem in closed form (scipy lstsq); it has no max_iter, tol or L-BFGS settings

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | cuml-gpu | ours | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (declared) | sklearn (get_params) |
| fit_intercept | true | true | true |
| loss | "l2" | "squared_error" | - |
| max_iter | 1000 | 1000 | - |
| positive | - | - | false |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| tol | 0.0001 | 0.0001 | 1e-06 |

REFUSED: sklearn-cpu: tol is 0.0001 (tol) on ours and 1e-06 (tol) on sklearn-cpu

REFUSED: cuml-gpu: loss is 'squared_error' (loss) on ours and 'l2' (loss) on cuml-gpu

### select-d / synthetic (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc 0, log `logs/algos.select-d.synthetic.rows-full.log`, ran on f003873fb257

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1.8 | 1.8..1.8 | 1 | - | - | - | 1499.6 | 682.0 | d_agreement_vs_statsmodels=1.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsmodels-cpu | statsmodels | cpu | opponent | 6.4 | 6.4..6.4 | 1 | 0.289 | - | - | 213.3 | - | d_agreement_vs_statsmodels=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, statsmodels-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'D': 0, 'pval_threshold': 0.05, 's': 0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | statsmodels-cpu |
|---|---||---|---|
| library (source) | mojolearn (declared) | statsmodels (declared) |
| seed | "none (deterministic)" | "none (deterministic)" |

### select-d / taxi-hourly (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc 0, log `logs/algos.select-d.taxi-hourly.rows-full.log`, ran on f003873fb257

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 2.1 | 2.1..2.1 | 1 | - | - | - | 1500.3 | 682.0 | d_agreement_vs_statsmodels=1.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsmodels-cpu | statsmodels | cpu | opponent | 4.5 | 4.5..4.5 | 1 | 0.462 | - | - | 212.7 | - | d_agreement_vs_statsmodels=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, statsmodels-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'D': 0, 'pval_threshold': 0.05, 's': 0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | statsmodels-cpu |
|---|---||---|---|
| library (source) | mojolearn (declared) | statsmodels (declared) |
| seed | "none (deterministic)" | "none (deterministic)" |

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
- Neural: The Mamba opponents are the repo's pure-PyTorch references (mamba/corpus/gen_corpus.py: mamba_ssm's selective_scan_ref for Mamba-1, the chunked SSD reference for Mamba-2, the SISO reference for Mamba-3), not mamba-ssm's fused CUDA/Triton kernels, which the board does not install; a Mamba ratio here is against a reference implementation, not a deployment kernel.
- Neural: The blocks' backward (the Mamba and TransformerBlock VJPs), their decode `step`, ragged `lengths` and the carried-state forward are public and not raced; only a zero-state forward is.
- Neural: SmallMLPTrainer.predict_logits (the GPU forward of the 8-16-3 MLP) is not raced; MLPInference (its CPU forward) and the training step are.
- Neural: The GPT-3-small target shape is not on the board (the LM lanes use the smaller control shape so one shape runs on every box, a 16 GB Mac included).
- Neural, not planned on this vendor: torch-compile-* on mamba1-forward and mamba1-infer: the only torch Mamba-1 twin is the pure-PyTorch reference scan, a per-token Python loop that torch.compile would unroll L times; mamba1 races the eager arms only
- Neural, not planned on this vendor: gemm-bf16 races torch's bf16 settings only (bf16 operands, fp32 accumulate)

