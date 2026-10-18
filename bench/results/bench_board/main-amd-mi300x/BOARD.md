# mojolearn benchmark board

Generated 2026-10-18T06:49:16Z from `board.json` (schema `mojolearn-bench-board/1`).

> MAIN BOARD amd-mi300x, version label main@80ace6b7a. Unreleased: not reproducible by pip install; the release boards are the reference.

> Cells: 24 races; oldest cell main@8d5855b83 (2026-10-10T06:56:04Z), newest cell main@80ace6b7a (2026-10-16T05:02:09Z). Boxes: AMD Instinct MI300X (amd2, Hot Aisle).

> Rule: each lane x dataset shows the newest default-configuration race on main (highest commit date, then job number) whose status is ok. A newer ok cell replaces an older one whatever the two times are; a run that is not ok is never a numeric cell and never replaces an ok cell (FAILED table; an older ok cell stays, flagged with the newer failed run). 0 replaced or failed observations are in LEDGER.md. A/B and grid arms (MOJOLEARN_BUILD_DEFINES, MOJOLEARN_GRID_TAG grid runs) are never on this board.

> Ours: recorded scored runs from lq jobs or verified paired sessions; the status column names the cell's commit, box/job, commit date and the other vendor's digest at the same commit (identity: MATCH 12, n/a 12).

> Opponents: retained from each paired session when present; otherwise retained historical pairs or stored opponent boards (none found). Ingestion runs no measurements. `ours IDENTICAL / arm` divides the two stored medians, and the clock columns read a torch GPU arm kernel/kernel and every other arm whole/whole (AGENTS.md measurement item 6). Our kernel clock is `-` unless the cell recorded upload_ms_separate. Opponents withheld for changed lane settings: 0 races.

> Provenance: main@ identifies the source or harness freeze. Paired wheel runs separately record the installed library version and loaded binary hashes; they are not source builds of that harness commit.

> Neural lanes: the headline (its own table, and a line under each neural race) is ours IDENTICAL over torch's fastest bf16 arm, eager or compile, what customers run; the fp32 twin is the second column. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax. A cell of ours whose output hash equals a copied opponent's shows quality identical_to=<arm> (the same bits) where the own-host reference gave none.

## Identity

Same lane, dataset and commit on the other GPU vendor (identity = equal output digests on NVIDIA and AMD). Counts: MATCH 12, n/a 12.

DIFFER: none.

## FAILED

Runs on main whose status is not ok (error, refused, timeout, not_ready, NO-RECORD, NO-OURS-CELL). They are never a numeric cell and never replace an ok cell; an older ok cell stays on the board flagged with the failed run. 0 failed runs.


## Box

| field | value |
|---|---|
| vendor / API | amd / hip |
| GPU | AMD Instinct MI300X |
| GPU driver | - |
| CPU | None (None logical cores) |
| memory bytes | - |
| OS | - |
| Python | - |
| mojolearn | main@80ace6b7a (wheel mixed source/wheel provenance; see each cell's version and measurement record, sha256 -) |
| script commit | 80ace6b7aaa97f9e7f6a32646cf04b7548f29c6c |
| patch sync | - |
| modes | identical |
| rounds | 1 timed after 1 warm-up, arms interleaved round by round |
| seed | 7 |
| opponent versions |  |

## How to read this board

- Times are wall milliseconds of the public fit call (trees) or the lane's timed call (classical), median of the timed rounds; min..max beside it.
- `ours IDENTICAL / arm` is our IDENTICAL median divided by that opponent's median; `ours FAST / arm` likewise. Below 1.0 our median time is the lower one, above 1.0 the higher one. A ratio is shown only when both arms completed every round in this run, and only against an opponent: our two modes are never divided by each other here.
- Two clocks (AGENTS.md measurement item 6): `whole ms` is the operation including the host-to-device copy of its inputs, `kernel ms` the same with the inputs already on the device; `copy ms` comes only from a stored field, named beside it (`upload_ms_separate`: our separate upload probe, kernel = median - copy; `upload_ms_untimed`: an opponent's pre-clock upload, whole = median + copy; `cpu-arm`: no device copy exists). A clock the stored fields cannot give is `-`, never estimated. `ours IDENTICAL / arm (clock)` reads a torch GPU arm on kernel/kernel and every other arm on whole/whole; when that clock is missing on a side it falls back to the other common clock (labelled), and with no common clock it is the two stored medians labelled MIXED with each side's clock.
- Quality comes from the drivers: FSPEED-ACC for trees (held-out rows), one float64 NumPy function per lane for classical.
- Comparability: trees carry FSPEED-FIT-VERDICT (total leaves within 10% across arms is COMPARABLE); classical carry the clock span (SPAN-ASYMMETRIC names an arm whose clock excludes an upload or a fit that ours includes).
- Classical, wave 2 (`classical2`, tools/bench_board_more.py): the same worker protocol as classical; every lane's parameters, rows, timed span and each unavoidable mismatch with its reason are in the cells' `settings.lane_config`. Quality is one float64 NumPy function per lane over each arm's saved outputs.
- Neural: our IDENTICAL arm only (the neural surface builds no other tier, on any vendor) against torch at every fast setting it supports on this box, one arm each, the setting in the arm name: `torch-eager-fp32` (TF32 off), `torch-compile-fp32` (torch.compile, inductor), `torch-eager-tf32` / `torch-compile-tf32` (NVIDIA CUDA only), `torch-eager-bf16` / `torch-compile-bf16` (bf16 autocast mixed precision). TF32 and bf16 arms are ANOTHER PRECISION than ours; their quality columns show how far. An arm torch cannot run on this box is REFUSED by name in its cell. Every clock is host in, host out, synchronized. Every arm starts from the same parameters and reads the same inputs, so losses and outputs are comparable; `max_abs_diff_vs_ours` / `max_rel_diff_vs_ours` are the arm's output against ours. The torch models are library classes: transformers' LlamaForCausalLM / LlamaDecoderLayer for the LM and transformer lanes, mamba_ssm's Mamba3 blocks with transformers' attention layers for Samba; the Mamba lanes' opponent is mamba_ssm's own fused kernels (`mamba-ssm-fp32`, and `mamba-ssm-tf32` on NVIDIA), and their pure-PyTorch reference torch arms are REFUSED by name. A compile arm compiles in its untimed warm-up (`compile_warmup_ms`).
- `installed_wheel` confirms our binding loaded from site-packages, not the repo tree.
- Our CPU is never raced or reported: the board races only our GPU, against GPU opponents; a race keeps CPU opponents only when it has no GPU opponent (Andrew, Oct 2 2026). A cell of ours on the CPU in an old record is dropped before rendering.
- Memory: `peak host MB` and `peak GPU MB` are the highest per-round peaks over the timed rounds, read outside the clock; each arm's method is listed under its table (host: the resettable peak RSS on Linux, the peak physical footprint on macOS, which holds Metal buffers too; GPU: torch's own counter for torch arms, the driver's per-process figure for the rest, none on Apple).
- Inference: after a race's fit rounds each arm predicts with its own fitted model (no fit retimed), same rows, same output kind, one warm-up then the timed rounds interleaved. Trees: batch `test` (the held-out split) and `large` (1,000,000 training rows, capped at the training rows), host rows in and host predictions out on every arm; each arm's call is printed under its table. Classical: kmeans predict, pca transform, ols predict and svc predict on the eval rows, with the fit's clock span. Ratios are per batch, ours over each opponent.

## Coverage

Races: 24 planned, 24 done, 0 failed, 0 unsupported, 0 pending. Cells: 97 (REFUSED 18, not-scored 1, ok 75, recompiled-in-clock 3).

Inference cells: 24 (REFUSED 1, ok 23).

## Quality at a glance

Per lane and dataset: our FAST value, our IDENTICAL value, and each opponent's.

| family | lane | dataset | metric | ours FAST | ours IDENTICAL | opponents |
|---|---|---|---|---|---|---|
| algos | adafactor | synthetic | optimizer_update_error_vs_torch_fp32 | - | 3.386e-06 | torch-eager-fp32 0.000000; torch-compile-fp32 0.000000 |
| algos | adafactor | synthetic | rel_fro_vs_torch_eager_fp32 (0 is the eager FP32 torch output exactly) | - | 1.033e-08 | torch-eager-fp32 0.000000; torch-compile-fp32 0.000000 |
| algos | adafactor | synthetic | relative_error_vs_own_host | - | 0.000000 | torch-eager-fp32 -; torch-compile-fp32 - |
| algos | adagrad | synthetic | optimizer_update_error_vs_torch_fp32 | - | 1.874e-06 | torch-eager-fp32 0.000000; torch-compile-fp32 0.000000 |
| algos | adagrad | synthetic | rel_fro_vs_torch_eager_fp32 (0 is the eager FP32 torch output exactly) | - | 3.207e-09 | torch-eager-fp32 0.000000; torch-compile-fp32 0.000000 |
| algos | adagrad | synthetic | relative_error_vs_own_host | - | 0.000000 | torch-eager-fp32 -; torch-compile-fp32 - |
| algos | adam | synthetic | optimizer_update_error_vs_torch_fp32 | - | 8.844e-06 | torch-eager-fp32 0.000000; torch-compile-fp32 1.355e-05; torch-fused-fp32 6.002e-06 |
| algos | adam | synthetic | rel_fro_vs_torch_eager_fp32 (0 is the eager FP32 torch output exactly) | - | 3.34e-08 | torch-eager-fp32 0.000000; torch-compile-fp32 5.116e-08; torch-fused-fp32 2.267e-08 |
| algos | adam | synthetic | relative_error_vs_own_host | - | 0.000000 | torch-eager-fp32 -; torch-compile-fp32 -; torch-fused-fp32 - |
| algos | adamax | synthetic | optimizer_update_error_vs_torch_fp32 | - | 1.54e-06 | torch-eager-fp32 0.000000; torch-compile-fp32 2.292e-06 |
| algos | adamax | synthetic | rel_fro_vs_torch_eager_fp32 (0 is the eager FP32 torch output exactly) | - | 4.144e-09 | torch-eager-fp32 0.000000; torch-compile-fp32 6.17e-09 |
| algos | adamax | synthetic | relative_error_vs_own_host | - | 0.000000 | torch-eager-fp32 -; torch-compile-fp32 - |
| algos | adamw | synthetic | optimizer_update_error_vs_torch_fp32 | - | 8.841e-06 | torch-eager-fp32 0.000000; torch-compile-fp32 6.012e-05; torch-fused-fp32 6.574e-05 |
| algos | adamw | synthetic | rel_fro_vs_torch_eager_fp32 (0 is the eager FP32 torch output exactly) | - | 3.34e-08 | torch-eager-fp32 0.000000; torch-compile-fp32 2.272e-07; torch-fused-fp32 2.484e-07 |
| algos | adamw | synthetic | relative_error_vs_own_host | - | 0.000000 | torch-eager-fp32 -; torch-compile-fp32 -; torch-fused-fp32 - |
| algos | avgpool1d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | 0.000000 | torch-eager-fp32 0.000000; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | avgpool1d | synthetic | rel_fro_vs_torch_eager_fp32 (0 is the eager FP32 torch output exactly) | - | 0.000000 | torch-eager-fp32 0.000000; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | avgpool1d | synthetic | relative_error_vs_own_host | - | 0.000000 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | avgpool2d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | 0.000000 | torch-eager-fp32 0.000000; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | avgpool2d | synthetic | rel_fro_vs_torch_eager_fp32 (0 is the eager FP32 torch output exactly) | - | 0.000000 | torch-eager-fp32 0.000000; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | avgpool2d | synthetic | relative_error_vs_own_host | - | 0.000000 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | batchnorm1d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | 0.004642 | torch-eager-fp32 0.000000; torch-compile-fp32 0.002793; torch-eager-bf16 0.000000; torch-compile-bf16 0.002793 |
| algos | batchnorm1d | synthetic | rel_fro_vs_torch_eager_fp32 (0 is the eager FP32 torch output exactly) | - | 8.097e-08 | torch-eager-fp32 0.000000; torch-compile-fp32 5.248e-08; torch-eager-bf16 0.000000; torch-compile-bf16 5.248e-08 |
| algos | batchnorm1d | synthetic | relative_error_vs_own_host | - | 0.000000 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | batchnorm2d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | 0.001395 | torch-eager-fp32 0.000000; torch-compile-fp32 0.001471; torch-eager-bf16 0.000000; torch-compile-bf16 0.001471 |
| algos | batchnorm2d | synthetic | rel_fro_vs_torch_eager_fp32 (0 is the eager FP32 torch output exactly) | - | 1.249e-07 | torch-eager-fp32 0.000000; torch-compile-fp32 6.868e-08; torch-eager-bf16 0.000000; torch-compile-bf16 6.868e-08 |
| algos | batchnorm2d | synthetic | relative_error_vs_own_host | - | 0.000000 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | cagra | istella | recall_at_10 (higher is better) | - | 0.997250 | hipvs-gpu - |
| algos | cagra | taxi | recall_at_10 (higher is better) | - | 0.997925 | hipvs-gpu 0.628775 |
| algos | cholesky | synthetic | relative_residual | - | 2.9e-07 | torch-gpu 2.519e-07 |
| algos | incremental-pca | taxi | explained_variance_fraction | - | 0.999995 | - |
| neural | gemm-bf16 | gaussian | max_rel_err_vs_fp64 (lower is better) | - | 1.155e-07 | torch-eager-bf16 0.002759; torch-compile-bf16 0.002759 |
| neural | gemm-bf16 | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | torch-eager-bf16 0.999023; torch-compile-bf16 0.999023 |
| neural | gemm-bf16 | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | torch-eager-bf16 0.002759; torch-compile-bf16 0.002759 |
| neural | gemm | gaussian | max_rel_err_vs_fp64 (lower is better) | - | 2.399e-07 | torch-eager-fp32 2.802e-06; torch-compile-fp32 2.802e-06; torch-eager-bf16 0.003752; torch-compile-bf16 0.003752 |
| neural | gemm | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | torch-eager-fp32 0.001038; torch-compile-fp32 0.001038; torch-eager-bf16 1.358337; torch-compile-bf16 1.358337 |
| neural | gemm | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | torch-eager-fp32 2.866e-06; torch-compile-fp32 2.866e-06; torch-eager-bf16 0.003751; torch-compile-bf16 0.003751 |
| neural | lm-forward | bytes | mean_nll (lower is better) | - | 9.015586 | torch-eager-fp32 9.015586; torch-compile-fp32 9.015586; torch-eager-bf16 9.015493; torch-compile-bf16 9.015495 |
| neural | lm-forward | bytes | max_abs_diff_vs_ours (0 is our output exactly) | - | - | torch-eager-fp32 1.401e-06; torch-compile-fp32 1.371e-06; torch-eager-bf16 0.006313; torch-compile-bf16 0.006870 |
| neural | lm-forward | bytes | max_rel_diff_vs_ours (0 is our output exactly) | - | - | torch-eager-fp32 1.923e-06; torch-compile-fp32 1.882e-06; torch-eager-bf16 0.008666; torch-compile-bf16 0.009432 |
| neural | mamba2-forward | gaussian | host_reference_atol | - | 1e-06 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 -; mamba-ssm-fp32 - |
| neural | mamba2-forward | gaussian | host_reference_rtol | - | 1e-05 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 -; mamba-ssm-fp32 - |
| neural | mamba2-forward | gaussian | host_reference_tolerance_ratio | - | 0.000000 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 -; mamba-ssm-fp32 - |
| neural | mamba2-forward | gaussian | max_abs_diff_vs_host | - | 0.000000 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 -; mamba-ssm-fp32 - |
| neural | mamba2-forward | gaussian | max_rel_diff_vs_host | - | 0.000000 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 -; mamba-ssm-fp32 - |
| neural | mamba2-forward | gaussian | max_rel_diff_vs_torch_eager_fp32 | - | 0.195652 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 -; mamba-ssm-fp32 0.048930 |
| neural | mamba2-forward | gaussian | rel_fro_vs_torch_eager_fp32 (0 is the eager FP32 torch output exactly) | - | 1.372e-07 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 -; mamba-ssm-fp32 7.547e-08 |
| neural | mamba2-forward | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 -; mamba-ssm-fp32 1.073e-06 |
| neural | mamba2-forward | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 -; mamba-ssm-fp32 3.648e-07 |
| neural | mamba3-forward | gaussian | host_reference_atol | - | 1e-06 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 -; mamba-ssm-fp32 - |
| neural | mamba3-forward | gaussian | host_reference_rtol | - | 1e-05 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 -; mamba-ssm-fp32 - |
| neural | mamba3-forward | gaussian | host_reference_tolerance_ratio | - | 0.000000 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 -; mamba-ssm-fp32 - |
| neural | mamba3-forward | gaussian | max_abs_diff_vs_host | - | 0.000000 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 -; mamba-ssm-fp32 - |
| neural | mamba3-forward | gaussian | max_rel_diff_vs_host | - | 0.000000 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 -; mamba-ssm-fp32 - |
| neural | mamba3-forward | gaussian | max_rel_diff_vs_torch_eager_fp32 | - | 0.013978 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 -; mamba-ssm-fp32 - |
| neural | mamba3-forward | gaussian | rel_fro_vs_torch_eager_fp32 (0 is the eager FP32 torch output exactly) | - | 5.247e-08 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 -; mamba-ssm-fp32 - |
| neural | mlp-predict | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | torch-eager-fp32 0.000000; torch-compile-fp32 0.000000; torch-eager-bf16 0.004285; torch-compile-bf16 0.004285 |
| neural | mlp-predict | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | torch-eager-fp32 0.000000; torch-compile-fp32 0.000000; torch-eager-bf16 0.003956; torch-compile-bf16 0.003956 |
| neural | mlp-train-step | gaussian | loss_first_step (same init and batches on every arm) | - | 1.160401 | torch-eager-fp32 1.160401; torch-compile-fp32 1.160401; torch-eager-bf16 1.160498; torch-compile-bf16 1.160498 |
| neural | mlp-train-step | gaussian | loss_last_step (same init and batches on every arm) | - | 1.123361 | torch-eager-fp32 1.123361; torch-compile-fp32 1.123361; torch-eager-bf16 1.123461; torch-compile-bf16 1.123461 |
| neural | mlp-train-step | gaussian | steps | - | 2 | torch-eager-fp32 2; torch-compile-fp32 2; torch-eager-bf16 2; torch-compile-bf16 2 |
| neural | mlp-train-step | gaussian | loss_first_abs_diff_vs_ours (0 is our value exactly) | - | - | torch-eager-fp32 2.384e-07; torch-compile-fp32 2.384e-07; torch-eager-bf16 9.656e-05; torch-compile-bf16 9.656e-05 |
| neural | mlp-train-step | gaussian | loss_last_abs_diff_vs_ours (0 is our value exactly) | - | - | torch-eager-fp32 2.384e-07; torch-compile-fp32 2.384e-07; torch-eager-bf16 0.0001005; torch-compile-bf16 0.0001006 |
| neural | samba-forward | bytes | mean_nll (lower is better) | - | 5.640151 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| neural | samba-train-step | bytes | loss_first_step (same init and batches on every arm) | - | 5.640152 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| neural | samba-train-step | bytes | loss_last_step (same init and batches on every arm) | - | 4.822321 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| neural | samba-train-step | bytes | steps | - | 2 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| neural | transformer-decode | gaussian | host_reference_atol | - | 1e-06 | torch-eager-fp32 -; torch-eager-bf16 - |
| neural | transformer-decode | gaussian | host_reference_rtol | - | 1e-05 | torch-eager-fp32 -; torch-eager-bf16 - |
| neural | transformer-decode | gaussian | host_reference_tolerance_ratio | - | 0.000000 | torch-eager-fp32 -; torch-eager-bf16 - |
| neural | transformer-decode | gaussian | max_abs_diff_vs_host | - | 0.000000 | torch-eager-fp32 -; torch-eager-bf16 - |
| neural | transformer-decode | gaussian | max_rel_diff_vs_host | - | 0.000000 | torch-eager-fp32 -; torch-eager-bf16 - |
| neural | transformer-decode | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | torch-eager-fp32 4.768e-07; torch-eager-bf16 0.001819 |
| neural | transformer-decode | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | torch-eager-fp32 1.02e-07; torch-eager-bf16 0.0003893 |
| neural | transformer-forward | gaussian | host_reference_atol | - | 1e-06 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| neural | transformer-forward | gaussian | host_reference_rtol | - | 1e-05 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| neural | transformer-forward | gaussian | host_reference_tolerance_ratio | - | 0.000000 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| neural | transformer-forward | gaussian | max_abs_diff_vs_host | - | 0.000000 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| neural | transformer-forward | gaussian | max_rel_diff_vs_host | - | 0.000000 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| neural | transformer-forward | gaussian | max_rel_diff_vs_torch_eager_fp32 | - | 0.015169 | torch-eager-fp32 -; torch-compile-fp32 0.005081; torch-eager-bf16 76.084014; torch-compile-bf16 41.369141 |
| neural | transformer-forward | gaussian | rel_fro_vs_torch_eager_fp32 (0 is the eager FP32 torch output exactly) | - | 5.717e-08 | torch-eager-fp32 -; torch-compile-fp32 3.096e-08; torch-eager-bf16 0.0002764; torch-compile-bf16 0.0002629 |
| neural | transformer-forward | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | torch-eager-fp32 4.768e-07; torch-compile-fp32 4.768e-07; torch-eager-bf16 0.001819; torch-compile-bf16 0.001819 |
| neural | transformer-forward | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | torch-eager-fp32 1.02e-07; torch-compile-fp32 1.02e-07; torch-eager-bf16 0.0003893; torch-compile-bf16 0.0003893 |

## Inference at a glance

Batch prediction, each arm with its own fitted model from the same race; medians in ms. `FAST = IDENTICAL bits` compares our two tiers' predictions on the same rows. Our CPU is never raced or reported.

| family | lane | dataset | batch | rows | ours FAST ms | ours IDENTICAL ms | FAST = IDENTICAL bits | opponents |
|---|---|---|---|---|---|---|---|---|
| algos | avgpool1d | synthetic | Xq | - | - | 0.2 | - | torch-eager-fp32 0.1 ms (IDENTICAL/arm 2.397); torch-compile-fp32 0.6 ms (IDENTICAL/arm 0.410); torch-eager-bf16 0.1 ms (IDENTICAL/arm 2.064); torch-compile-bf16 0.2 ms (IDENTICAL/arm 1.076) |
| algos | avgpool2d | synthetic | Xq | - | - | 0.2 | - | torch-eager-fp32 0.2 ms (IDENTICAL/arm 0.950); torch-compile-fp32 0.3 ms (IDENTICAL/arm 0.813); torch-eager-bf16 0.2 ms (IDENTICAL/arm 0.912); torch-compile-bf16 0.6 ms (IDENTICAL/arm 0.389) |
| algos | batchnorm1d | synthetic | Xq | - | - | 0.4 | - | torch-eager-fp32 0.2 ms (IDENTICAL/arm 2.360); torch-compile-fp32 1.3 ms (IDENTICAL/arm 0.281); torch-eager-bf16 0.2 ms (IDENTICAL/arm 1.925); torch-compile-bf16 0.7 ms (IDENTICAL/arm 0.508) |
| algos | batchnorm2d | synthetic | Xq | - | - | 0.4 | - | torch-eager-fp32 0.5 ms (IDENTICAL/arm 0.768); torch-compile-fp32 1.0 ms (IDENTICAL/arm 0.377); torch-eager-bf16 0.6 ms (IDENTICAL/arm 0.695); torch-compile-bf16 0.3 ms (IDENTICAL/arm 1.507) |
| algos | cagra | istella | Xq | - | - | 237.9 | - | hipvs-gpu - ms (IDENTICAL/arm -) |
| algos | cagra | taxi | Xq | - | - | 154.2 | - | hipvs-gpu 2.6 ms (IDENTICAL/arm 59.233) |


## Neural headline: ours IDENTICAL against torch bf16

What customers run is torch in bf16. The headline divides our IDENTICAL median by torch's fastest bf16 arm (eager or compile, the lower stored median); the fp32 twin (torch's fastest fp32 arm) is the second column. Per-arm ratios stay in each race's table below.

| lane | dataset | ours IDENTICAL ms | torch bf16 ms (fastest arm) | ours / torch bf16 | torch fp32 twin ms (fastest arm) | ours / torch fp32 | note |
|---|---|---|---|---|---|---|---|
| adafactor | synthetic | 36.3 | - | - | 9.1 (torch-compile-fp32) | 3.975 | no torch bf16 arm on this lane |
| adagrad | synthetic | 6.5 | - | - | 3.6 (torch-compile-fp32) | 1.791 | no torch bf16 arm on this lane |
| adam | synthetic | 25.2 | - | - | 7.1 (torch-eager-fp32) | 3.538 | no torch bf16 arm on this lane |
| adamax | synthetic | 4.8 | - | - | 6.8 (torch-eager-fp32) | 0.702 | no torch bf16 arm on this lane |
| adamw | synthetic | 42.1 | - | - | 6.8 (torch-eager-fp32) | 6.181 | no torch bf16 arm on this lane |
| avgpool1d | synthetic | 1.6 | 1.0 (torch-eager-bf16) | 1.627 | 0.7 (torch-eager-fp32) | 2.210 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| avgpool2d | synthetic | 8.2 | 2.3 (torch-compile-bf16) | 3.549 | 1.3 (torch-eager-fp32) | 6.257 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| batchnorm1d | synthetic | 2.0 | 1.3 (torch-eager-bf16) | 1.554 | 1.4 (torch-eager-fp32) | 1.490 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| batchnorm2d | synthetic | 1.2 | 1.9 (torch-compile-bf16) | 0.657 | 8.4 (torch-eager-fp32) | 0.146 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| gemm-bf16 | gaussian | 97.4 | 20.4 (torch-compile-bf16) | 4.782 | - | - | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| gemm | gaussian | 120.2 | 23.1 (torch-eager-bf16) | 5.209 | 23.5 (torch-compile-fp32) | 5.119 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| lm-forward | bytes | 116.8 | 22.5 (torch-compile-bf16) | 5.195 | 23.6 (torch-eager-fp32) | 4.947 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| mamba2-forward | gaussian | 6.2 | - | - | - | - | no torch bf16 arm on this lane |
| mamba3-forward | gaussian | 9.5 | - | - | - | - | no torch bf16 arm on this lane |
| mlp-predict | gaussian | - | 0.9 (torch-compile-bf16) | - | 1.3 (torch-eager-fp32) | - | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| mlp-train-step | gaussian | 7.3 | 5.2 (torch-eager-bf16) | 1.400 | 5.0 (torch-eager-fp32) | 1.449 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| samba-forward | bytes | 8.8 | - | - | - | - | no torch bf16 arm on this lane |
| samba-train-step | bytes | 54.0 | - | - | - | - | no torch bf16 arm on this lane |
| transformer-decode | gaussian | 2509.8 | 1441.6 (torch-eager-bf16) | 1.741 | 1288.1 (torch-eager-fp32) | 1.948 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| transformer-forward | gaussian | 9.6 | 2.5 (torch-compile-bf16) | 3.850 | 2.1 (torch-compile-fp32) | 4.489 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |

## Neural

### gemm-bf16 / gaussian (neural shape full: 4096x4096x4096)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/measure-0852/ingest0849/inputs/amd-neural.gemm-bf16.gaussian.shape=full.json (amd2/paired-8fcc53ef0554731cc6640e646de1c7b57ee63d890319e020a469a8f3857da666)`, ran on enc1-gpuvm005

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 20.4 ms) = 4.782; fp32 twin: ours IDENTICAL / torch fp32: - (no completed torch fp32 arm). Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI300X (amd2, Hot Aisle) | 0.8.49 | identical | 97.4 | 97.4..97.4 | 1 | - | - | 97.4 | - | - (stored whole) | - | - | 2383.1 | 4447.8 | max_rel_err_vs_fp64=1.155e-07 | - | main board, one scored run | wheel | ok (main@328d8c59d amd2/paired-8fcc53ef0554731cc6640e646de1c7b57ee63d890319e020a469a8f3857da666 2026-10-16; identity vs the nvidia columns: n/a) |
| torch-eager-bf16 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | 39.3 | 39.3..39.3 | 1 | 2.479 | - | 39.3 | - | - (stored whole) | 2.479 (whole/whole) | - | 2877.8 | 312.0 | max_abs_diff_vs_ours=0.999023, max_rel_diff_vs_ours=0.002759, max_rel_err_vs_fp64=0.002759 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | 20.4 | 20.4..20.4 | 1 | 4.782 | - | 20.4 | - | - (stored whole) | 4.782 (whole/whole) | - | 2946.3 | 312.0 | max_abs_diff_vs_ours=0.999023, max_rel_diff_vs_ours=0.002759, max_rel_err_vs_fp64=0.002759 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

calls:
- `ours`: `mojolearn.matmul_bf16(operand_value_type='bfloat16', operation='matmul', seed=7, transpose_b=False, a=float32[4096x4096], b=float32[4096x4096])`
- `torch-eager-bf16`: `torch._VariableFunctionsClass.matmul(operand_value_type='bfloat16', operation='matmul', seed=7, transpose_b=False, a=float32[4096x4096], b=float32[4096x4096])`
- `torch-compile-bf16`: `torch._VariableFunctionsClass.matmul(operand_value_type='bfloat16', operation='matmul', seed=7, transpose_b=False, a=float32[4096x4096], b=float32[4096x4096])`

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/122355/vram_* bytes for this pid at the round's end (not a peak)

memory, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-eager-bf16 |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) |
| input_arrays | {"a": {"dtype": "float32", "sha256": "485a7287793a9d7d33d710ea2820ea9c7519d31100db4bafeee813baa532dd7d", "shape": [4096, 4096]}, "b": {"dtype": "float32", "sha256": "3f38db408e287e78e700728b77d904ea5344de168156ad74431c2148fc961104", "shape": [4096, 4096]}} | {"a": {"dtype": "float32", "sha256": "485a7287793a9d7d33d710ea2820ea9c7519d31100db4bafeee813baa532dd7d", "shape": [4096, 4096]}, "b": {"dtype": "float32", "sha256": "3f38db408e287e78e700728b77d904ea5344de168156ad74431c2148fc961104", "shape": [4096, 4096]}} | {"a": {"dtype": "float32", "sha256": "485a7287793a9d7d33d710ea2820ea9c7519d31100db4bafeee813baa532dd7d", "shape": [4096, 4096]}, "b": {"dtype": "float32", "sha256": "3f38db408e287e78e700728b77d904ea5344de168156ad74431c2148fc961104", "shape": [4096, 4096]}} |
| operand_value_type | "bfloat16" | "bfloat16" | "bfloat16" |
| operation | "matmul" | "matmul" | "matmul" |
| seed | 7 | 7 | 7 |
| transpose_b | false | false | false |

### gemm / gaussian (neural shape full: 4096x4096x4096)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/measure-0852/ingest0849/inputs/amd-neural.gemm.gaussian.shape=full.json (amd2/paired-a23c3977a6d57b16f8f2f10479d23aab608350c0333efb90572ccdab0e0461e3)`, ran on enc1-gpuvm005

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 23.1 ms) = 5.209; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 23.5 ms) = 5.119. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI300X (amd2, Hot Aisle) | 0.8.49 | identical | 120.2 | 120.2..120.2 | 1 | - | - | 120.2 | - | - (stored whole) | - | - | 2446.6 | 4447.7 | max_rel_err_vs_fp64=2.399e-07 | - | main board, one scored run | wheel | ok (main@328d8c59d amd2/paired-a23c3977a6d57b16f8f2f10479d23aab608350c0333efb90572ccdab0e0461e3 2026-10-16; identity vs the nvidia columns: n/a) |
| torch-eager-fp32 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | 42.3 | 42.3..42.3 | 1 | 2.838 | - | 42.3 | - | - (stored whole) | 2.838 (whole/whole) | - | 2011.8 | 268.0 | max_abs_diff_vs_ours=0.001038, max_rel_diff_vs_ours=2.866e-06, max_rel_err_vs_fp64=2.802e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | 23.5 | 23.5..23.5 | 1 | 5.119 | - | 23.5 | - | - (stored whole) | 5.119 (whole/whole) | - | 2068.0 | 268.0 | max_abs_diff_vs_ours=0.001038, max_rel_diff_vs_ours=2.866e-06, max_rel_err_vs_fp64=2.802e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | 23.1 | 23.1..23.1 | 1 | 5.209 | - | 23.1 | - | - (stored whole) | 5.209 (whole/whole) | - | 2814.4 | 376.0 | max_abs_diff_vs_ours=1.358337, max_rel_diff_vs_ours=0.003751, max_rel_err_vs_fp64=0.003752 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | 41.6 | 41.6..41.6 | 1 | 2.890 | - | 41.6 | - | - (stored whole) | 2.890 (whole/whole) | - | 3119.9 | 376.0 | max_abs_diff_vs_ours=1.358337, max_rel_diff_vs_ours=0.003751, max_rel_err_vs_fp64=0.003752 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

calls:
- `ours`: `mojolearn.matmul(operand_value_type='float32', operation='matmul', seed=7, transpose_b=False, a=float32[4096x4096], b=float32[4096x4096])`
- `torch-eager-fp32`: `torch._VariableFunctionsClass.matmul(operand_value_type='float32', operation='matmul', seed=7, transpose_b=False, a=float32[4096x4096], b=float32[4096x4096])`
- `torch-compile-fp32`: `torch._VariableFunctionsClass.matmul(operand_value_type='float32', operation='matmul', seed=7, transpose_b=False, a=float32[4096x4096], b=float32[4096x4096])`
- `torch-eager-bf16`: `torch._VariableFunctionsClass.matmul(operand_value_type='float32', operation='matmul', seed=7, transpose_b=False, a=float32[4096x4096], b=float32[4096x4096])`
- `torch-compile-bf16`: `torch._VariableFunctionsClass.matmul(operand_value_type='float32', operation='matmul', seed=7, transpose_b=False, a=float32[4096x4096], b=float32[4096x4096])`

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/123368/vram_* bytes for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| input_arrays | {"a": {"dtype": "float32", "sha256": "485a7287793a9d7d33d710ea2820ea9c7519d31100db4bafeee813baa532dd7d", "shape": [4096, 4096]}, "b": {"dtype": "float32", "sha256": "3f38db408e287e78e700728b77d904ea5344de168156ad74431c2148fc961104", "shape": [4096, 4096]}} | {"a": {"dtype": "float32", "sha256": "485a7287793a9d7d33d710ea2820ea9c7519d31100db4bafeee813baa532dd7d", "shape": [4096, 4096]}, "b": {"dtype": "float32", "sha256": "3f38db408e287e78e700728b77d904ea5344de168156ad74431c2148fc961104", "shape": [4096, 4096]}} | {"a": {"dtype": "float32", "sha256": "485a7287793a9d7d33d710ea2820ea9c7519d31100db4bafeee813baa532dd7d", "shape": [4096, 4096]}, "b": {"dtype": "float32", "sha256": "3f38db408e287e78e700728b77d904ea5344de168156ad74431c2148fc961104", "shape": [4096, 4096]}} | {"a": {"dtype": "float32", "sha256": "485a7287793a9d7d33d710ea2820ea9c7519d31100db4bafeee813baa532dd7d", "shape": [4096, 4096]}, "b": {"dtype": "float32", "sha256": "3f38db408e287e78e700728b77d904ea5344de168156ad74431c2148fc961104", "shape": [4096, 4096]}} | {"a": {"dtype": "float32", "sha256": "485a7287793a9d7d33d710ea2820ea9c7519d31100db4bafeee813baa532dd7d", "shape": [4096, 4096]}, "b": {"dtype": "float32", "sha256": "3f38db408e287e78e700728b77d904ea5344de168156ad74431c2148fc961104", "shape": [4096, 4096]}} |
| operand_value_type | "float32" | "float32" | "float32" | "float32" | "float32" |
| operation | "matmul" | "matmul" | "matmul" | "matmul" | "matmul" |
| seed | 7 | 7 | 7 | 7 | 7 |
| transpose_b | false | false | false | false | false |

### lm-forward / bytes (neural shape full: B1 L2048 DM384 H6 KV6 HD64 FF1024 layers8 V8192)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/measure-0852/ingest0849/inputs/amd-neural.lm-forward.bytes.shape=full.json (amd2/paired-26e4845764e6652ca3eb3729a9203d8cfa0502643ced44e53a763eff533ad39e)`, ran on enc1-gpuvm005

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 22.5 ms) = 5.195; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 23.6 ms) = 4.947. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI300X (amd2, Hot Aisle) | 0.8.49 | identical | 116.8 | 116.8..116.8 | 1 | - | - | 116.8 | - | - (stored whole) | - | - | 2741.3 | 8737.6 | mean_nll=9.015586 | - | main board, one scored run | wheel | ok (main@328d8c59d amd2/paired-26e4845764e6652ca3eb3729a9203d8cfa0502643ced44e53a763eff533ad39e 2026-10-16; identity vs the nvidia columns: n/a) |
| torch-eager-fp32 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | 23.6 | 23.6..23.6 | 1 | 4.947 | - | 23.6 | - | - (stored whole) | 4.947 (whole/whole) | - | 4248.7 | 221.5 | max_abs_diff_vs_ours=1.401e-06, max_rel_diff_vs_ours=1.923e-06, mean_nll=9.015586 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | 50.0 | 50.0..50.0 | 1 | 2.338 | - | 50.0 | - | - (stored whole) | 2.338 (whole/whole) | - | 3514.2 | 242.0 | max_abs_diff_vs_ours=1.371e-06, max_rel_diff_vs_ours=1.882e-06, mean_nll=9.015586 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | 62.6 | 62.6..62.6 | 1 | 1.866 | - | 62.6 | - | - (stored whole) | 1.866 (whole/whole) | - | 4012.3 | 326.5 | max_abs_diff_vs_ours=0.006313, max_rel_diff_vs_ours=0.008666, mean_nll=9.015493 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | 22.5 | 22.5..22.5 | 1 | 5.195 | - | 22.5 | - | - (stored whole) | 5.195 (whole/whole) | - | 3304.5 | 326.5 | max_abs_diff_vs_ours=0.006870, max_rel_diff_vs_ours=0.009432, mean_nll=9.015495 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

calls:
- `torch-eager-fp32`: `transformers.LlamaForCausalLM(config_readback='{"d_model": 384, "head_dim": 64, "intermedia..., seed=7)`
- `torch-compile-fp32`: `transformers.LlamaForCausalLM(config_readback='{"d_model": 384, "head_dim": 64, "intermedia..., seed=7)`
- `torch-eager-bf16`: `transformers.LlamaForCausalLM(config_readback='{"d_model": 384, "head_dim": 64, "intermedia..., seed=7)`
- `torch-compile-bf16`: `transformers.LlamaForCausalLM(config_readback='{"d_model": 384, "head_dim": 64, "intermedia..., seed=7)`

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/125022/vram_* bytes for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| config_readback | "{\"d_model\": 384, \"head_dim\": 64, \"intermediate\": 1024, \"layers\": 8, \"n_heads\": 6, \"n_kv_heads\": 6, \"param_dtype\": \"float32\", \"tied_head\": false, \"vocab\": 8192}" | "{\"d_model\": 384, \"head_dim\": 64, \"intermediate\": 1024, \"layers\": 8, \"n_heads\": 6, \"n_kv_heads\": 6, \"param_dtype\": \"float32\", \"tied_head\": false, \"vocab\": 8192}" | "{\"d_model\": 384, \"head_dim\": 64, \"intermediate\": 1024, \"layers\": 8, \"n_heads\": 6, \"n_kv_heads\": 6, \"param_dtype\": \"float32\", \"tied_head\": false, \"vocab\": 8192}" | "{\"d_model\": 384, \"head_dim\": 64, \"intermediate\": 1024, \"layers\": 8, \"n_heads\": 6, \"n_kv_heads\": 6, \"param_dtype\": \"float32\", \"tied_head\": false, \"vocab\": 8192}" | "{\"d_model\": 384, \"head_dim\": 64, \"intermediate\": 1024, \"layers\": 8, \"n_heads\": 6, \"n_kv_heads\": 6, \"param_dtype\": \"float32\", \"tied_head\": false, \"vocab\": 8192}" |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 |

### mamba2-forward / gaussian (neural shape full: B1 L2048 DM384)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/measure-0852/ingest0849/inputs/amd-neural.mamba2-forward.gaussian.shape=full.json (amd2/paired-9ed9770849454ea1c01359b7ba314c6c429448b09bb9e28d478e04a4f956f0f8)`, ran on enc1-gpuvm005

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32: - (no completed torch fp32 arm). Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI300X (amd2, Hot Aisle) | 0.8.49 | identical | 6.2 | 6.2..6.2 | 1 | - | - | 6.2 | - | - (stored whole) | - | - | 2207.0 | 2528.1 | host_reference=Mamba2BlockInference, host_reference_atol=1e-06, host_reference_finite=True, host_reference_passed=True, host_reference_rtol=1e-05, host_reference_tolerance_ratio=0.000000, max_abs_diff_vs_host=0.000000, max_rel_diff_vs_host=0.000000, max_rel_diff_vs_torch_eager_fp32=0.195652, rel_fro_vs_torch_eager_fp32=1.372e-07 | - | main board, one scored run | wheel | ok (main@80ace6b7a amd2/paired-9ed9770849454ea1c01359b7ba314c6c429448b09bb9e28d478e04a4f956f0f8 2026-10-16; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(refused: "not an opponent: the torch twin of mamba2-forward is a pure-PyTorch reference implementation (mamba/corpus/gen_corpus.py), not a library kernel; mamba_ssm's fused kernels are the opponent only where ) (measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(refused: "not an opponent: the torch twin of mamba2-forward is a pure-PyTorch reference implementation (mamba/corpus/gen_corpus.py), not a library kernel; mamba_ssm's fused kernels are the opponent only where ) (measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(refused: "not an opponent: the torch twin of mamba2-forward is a pure-PyTorch reference implementation (mamba/corpus/gen_corpus.py), not a library kernel; mamba_ssm's fused kernels are the opponent only where ) (measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(refused: "not an opponent: the torch twin of mamba2-forward is a pure-PyTorch reference implementation (mamba/corpus/gen_corpus.py), not a library kernel; mamba_ssm's fused kernels are the opponent only where ) (measured this run) |
| mamba-ssm-fp32 | mamba-ssm | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | mamba-ssm 2.3.2.post1+ge9594ce1c732 | opponent | 3.7 | 3.7..3.7 | 1 | 1.671 | - | 3.7 | - | - (stored whole) | 1.671 (whole/whole) | - | 3441.2 | 126.5 | max_abs_diff_vs_ours=1.073e-06, max_rel_diff_vs_ours=3.648e-07, max_rel_diff_vs_torch_eager_fp32=0.048930, rel_fro_vs_torch_eager_fp32=7.547e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

calls:
- `mamba-ssm-fp32`: `mamba_ssm.Block(seed=7, ssm_chunk_size=256, ssm_d_conv=4, ssm_d_state=128, ssm_dt_limit=[0, inf], ssm_expand=2, ssm_headdim=64, ssm_ngroups=1)`

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/132952/vram_* bytes for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host not sampled; GPU not sampled

memory, mamba-ssm-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | mamba-ssm-fp32 | ours |
|---|---||---|---|
| library (source) | mamba-ssm (declared) | mojolearn (declared) |
| seed | 7 | "none (deterministic)" |
| ssm_chunk_size | 256 | 256 |
| ssm_d_conv | 4 | 4 |
| ssm_d_state | 128 | 128 |
| ssm_dt_limit | [0.0, Infinity] | [0.0, Infinity] |
| ssm_expand | 2 | 2 |
| ssm_headdim | 64 | 64 |
| ssm_ngroups | 1 | 1 |

### mamba3-forward / gaussian (neural shape full: B1 L2048 DM384)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/measure-0852/ingest0849/inputs/amd-neural.mamba3-forward.gaussian.shape=full.json (amd2/paired-b2c2b9b5767795cb72b7561cebfd8df0449fa3fa7c0431db01709c23c9028f85)`, ran on enc1-gpuvm005

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32: - (no completed torch fp32 arm). Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI300X (amd2, Hot Aisle) | 0.8.49 | identical | 9.5 | 9.5..9.5 | 1 | - | - | 9.5 | - | - (stored whole) | - | - | 2209.7 | 2528.1 | host_reference=Mamba3BlockInference, host_reference_atol=1e-06, host_reference_finite=True, host_reference_passed=True, host_reference_rtol=1e-05, host_reference_tolerance_ratio=0.000000, max_abs_diff_vs_host=0.000000, max_rel_diff_vs_host=0.000000, max_rel_diff_vs_torch_eager_fp32=0.013978, rel_fro_vs_torch_eager_fp32=5.247e-08 | - | main board, one scored run | wheel | ok (main@80ace6b7a amd2/paired-b2c2b9b5767795cb72b7561cebfd8df0449fa3fa7c0431db01709c23c9028f85 2026-10-16; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(refused: "not an opponent: the torch twin of mamba3-forward is a pure-PyTorch reference implementation (mamba/corpus/gen_corpus.py), not a library kernel; mamba_ssm's fused kernels are the opponent only where ) (measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(refused: "not an opponent: the torch twin of mamba3-forward is a pure-PyTorch reference implementation (mamba/corpus/gen_corpus.py), not a library kernel; mamba_ssm's fused kernels are the opponent only where ) (measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(refused: "not an opponent: the torch twin of mamba3-forward is a pure-PyTorch reference implementation (mamba/corpus/gen_corpus.py), not a library kernel; mamba_ssm's fused kernels are the opponent only where ) (measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(refused: "not an opponent: the torch twin of mamba3-forward is a pure-PyTorch reference implementation (mamba/corpus/gen_corpus.py), not a library kernel; mamba_ssm's fused kernels are the opponent only where ) (measured this run) |
| mamba-ssm-fp32 | mamba-ssm | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | mamba-ssm 2.3.2.post1+ge9594ce1c732 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(error: {"error": "worker returned no protocol message: worker_exited", "event": "error", "read_failure": {"deadline_s": 1800, "elapsed_s": 5.356679921998875, "reason": "eof"}, "reason": "worker_exited", "ret) (measured this run) |

calls:
- `mamba-ssm-fp32`: `mamba_ssm.Block(seed=7, ssm_chunk_size=64, ssm_d_state=128, ssm_expand=2, ssm_headdim=64, ssm_ngroups=1, ssm_rope_angles=32)`

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/133984/vram_* bytes for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16, mamba-ssm-fp32: host not sampled; GPU not sampled

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | mamba-ssm-fp32 | ours |
|---|---||---|---|
| library (source) | mamba-ssm (declared) | mojolearn (declared) |
| seed | 7 | "none (deterministic)" |
| ssm_chunk_size | 64 | 64 |
| ssm_d_state | 128 | 128 |
| ssm_expand | 2 | 2 |
| ssm_headdim | 64 | 64 |
| ssm_ngroups | 1 | 1 |
| ssm_rope_angles | 32 | 32 |

### mlp-predict / gaussian (neural shape full: rows256 8-16-3)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/measure-0852/ingest0849/inputs/amd-neural.mlp-predict.gaussian.shape=full.json (amd2/paired-cd2a29d36ceebb7c1a093aa1fadca665be6056d67c04b3b7a41e70cb84727f6d)`, ran on enc1-gpuvm005

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 0.9 ms) = -; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 1.3 ms) = -. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI300X (amd2, Hot Aisle) | 0.8.49 | identical | 0.7 | 0.7..0.7 | 1 | - | - | 0.7 | - | - (stored whole) | - | - | 1987.1 | 1704.3 | - | - | main board, one scored run | wheel | not-scored: quality not scored (scorer failed): - (main@80ace6b7a amd2/paired-cd2a29d36ceebb7c1a093aa1fadca665be6056d67c04b3b7a41e70cb84727f6d 2026-10-16; identity vs the nvidia columns: n/a) |
| torch-eager-fp32 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | 1.3 | 1.3..1.3 | 1 | 0.586 | - | 1.3 | - | - (stored whole) | 0.586 (whole/whole) | - | 2537.1 | 76.0 | max_abs_diff_vs_ours=0.000000, max_rel_diff_vs_ours=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | 2.7 | 2.7..2.7 | 1 | 0.270 | - | 2.7 | - | - (stored whole) | 0.270 (whole/whole) | - | 3458.8 | 76.0 | max_abs_diff_vs_ours=0.000000, max_rel_diff_vs_ours=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | 3.1 | 3.1..3.1 | 1 | 0.235 | - | 3.1 | - | - (stored whole) | 0.235 (whole/whole) | - | 2334.8 | 152.0 | max_abs_diff_vs_ours=0.004285, max_rel_diff_vs_ours=0.003956 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | 0.9 | 0.9..0.9 | 1 | 0.800 | - | 0.9 | - | - (stored whole) | 0.800 (whole/whole) | - | 2786.1 | 152.0 | max_abs_diff_vs_ours=0.004285, max_rel_diff_vs_ours=0.003956 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

calls:
- `torch-eager-fp32`: `torch.Sequential(config_readback='{"activation": "relu", "bias": true, "d_in":..., seed=7)`
- `torch-compile-fp32`: `torch.Sequential(config_readback='{"activation": "relu", "bias": true, "d_in":..., seed=7)`
- `torch-eager-bf16`: `torch.Sequential(config_readback='{"activation": "relu", "bias": true, "d_in":..., seed=7)`
- `torch-compile-bf16`: `torch.Sequential(config_readback='{"activation": "relu", "bias": true, "d_in":..., seed=7)`

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/134952/vram_* bytes for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| config_readback | "{\"activation\": \"relu\", \"bias\": true, \"d_in\": 8, \"d_out\": 3, \"hidden\": 16}" | "{\"activation\": \"relu\", \"bias\": true, \"d_in\": 8, \"d_out\": 3, \"hidden\": 16}" | "{\"activation\": \"relu\", \"bias\": true, \"d_in\": 8, \"d_out\": 3, \"hidden\": 16}" | "{\"activation\": \"relu\", \"bias\": true, \"d_in\": 8, \"d_out\": 3, \"hidden\": 16}" | "{\"activation\": \"relu\", \"bias\": true, \"d_in\": 8, \"d_out\": 3, \"hidden\": 16}" |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 |

### mlp-train-step / gaussian (neural shape full: rows256 8-16-3)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/measure-0852/ingest0849/inputs/amd-neural.mlp-train-step.gaussian.shape=full.json (amd2/paired-cfb0d380ff919206894df01629f0936428c5afd8e6df25c847ea1c7d08a3143a)`, ran on enc1-gpuvm005

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 5.2 ms) = 1.400; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 5.0 ms) = 1.449. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI300X (amd2, Hot Aisle) | 0.8.49 | identical | 7.3 | 7.3..7.3 | 1 | - | - | 7.3 | - | - (stored whole) | - | - | 1988.8 | 1704.5 | loss_first_step=1.160401, loss_last_step=1.123361, steps=2 | - | main board, one scored run | wheel | ok (main@80ace6b7a amd2/paired-cfb0d380ff919206894df01629f0936428c5afd8e6df25c847ea1c7d08a3143a 2026-10-16; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | 5.0 | 5.0..5.0 | 1 | 1.449 | - | 5.0 | - | - (stored whole) | 1.449 (whole/whole) | - | 4418.1 | 152.1 | loss_first_abs_diff_vs_ours=2.384e-07, loss_first_step=1.160401, loss_last_abs_diff_vs_ours=2.384e-07, loss_last_step=1.123361, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | 5.8 | 5.8..5.8 | 1 | 1.250 | - | 5.8 | - | - (stored whole) | 1.250 (whole/whole) | - | 4375.3 | 152.1 | loss_first_abs_diff_vs_ours=2.384e-07, loss_first_step=1.160401, loss_last_abs_diff_vs_ours=2.384e-07, loss_last_step=1.123361, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | 5.2 | 5.2..5.2 | 1 | 1.400 | - | 5.2 | - | - (stored whole) | 1.400 (whole/whole) | - | 2768.3 | 152.0 | loss_first_abs_diff_vs_ours=9.656e-05, loss_first_step=1.160498, loss_last_abs_diff_vs_ours=0.0001005, loss_last_step=1.123461, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | 6.4 | 6.4..6.4 | 1 | 1.149 | - | 6.4 | - | - (stored whole) | 1.149 (whole/whole) | - | 2890.7 | 152.0 | loss_first_abs_diff_vs_ours=9.656e-05, loss_first_step=1.160498, loss_last_abs_diff_vs_ours=0.0001006, loss_last_step=1.123461, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

calls:
- `torch-eager-fp32`: `torch.Sequential(amsgrad=False, betas=[0.9, 0.999], capturable=False, config_readback='{"activation": "relu", "bias": true, "d_in":..., decoupled_weight_decay=True, differentiable=False, eps=1e-08, lr=0.001, maximize=False, seed=7, ...)`
- `torch-compile-fp32`: `torch.Sequential(amsgrad=False, betas=[0.9, 0.999], capturable=False, config_readback='{"activation": "relu", "bias": true, "d_in":..., decoupled_weight_decay=True, differentiable=False, eps=1e-08, lr=0.001, maximize=False, seed=7, ...)`
- `torch-eager-bf16`: `torch.Sequential(amsgrad=False, betas=[0.9, 0.999], capturable=False, config_readback='{"activation": "relu", "bias": true, "d_in":..., decoupled_weight_decay=True, differentiable=False, eps=1e-08, lr=0.001, maximize=False, seed=7, ...)`
- `torch-compile-bf16`: `torch.Sequential(amsgrad=False, betas=[0.9, 0.999], capturable=False, config_readback='{"activation": "relu", "bias": true, "d_in":..., decoupled_weight_decay=True, differentiable=False, eps=1e-08, lr=0.001, maximize=False, seed=7, ...)`

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/136144/vram_* bytes for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| amsgrad | false | false | false | false | false |
| betas | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| capturable | - | false | false | false | false |
| config_readback | "{\"activation\": \"relu\", \"bias\": true, \"d_in\": 8, \"d_out\": 3, \"hidden\": 16}" | "{\"activation\": \"relu\", \"bias\": true, \"d_in\": 8, \"d_out\": 3, \"hidden\": 16}" | "{\"activation\": \"relu\", \"bias\": true, \"d_in\": 8, \"d_out\": 3, \"hidden\": 16}" | "{\"activation\": \"relu\", \"bias\": true, \"d_in\": 8, \"d_out\": 3, \"hidden\": 16}" | "{\"activation\": \"relu\", \"bias\": true, \"d_in\": 8, \"d_out\": 3, \"hidden\": 16}" |
| decoupled_weight_decay | true | true | true | true | true |
| differentiable | - | false | false | false | false |
| eps | 1e-08 | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| foreach | - | null | null | null | null |
| fused | - | null | null | null | null |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| maximize | false | false | false | false | false |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 |
| weight_decay | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |

accepted difference: torch-eager-fp32 capturable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-eager-fp32 differentiable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-eager-fp32 foreach: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

accepted difference: torch-eager-fp32 fused: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

accepted difference: torch-compile-fp32 capturable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-compile-fp32 differentiable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-compile-fp32 foreach: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

accepted difference: torch-compile-fp32 fused: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

accepted difference: torch-eager-bf16 capturable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-eager-bf16 differentiable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-eager-bf16 foreach: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

accepted difference: torch-eager-bf16 fused: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

accepted difference: torch-compile-bf16 capturable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-compile-bf16 differentiable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-compile-bf16 foreach: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

accepted difference: torch-compile-bf16 fused: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

### samba-forward / bytes (neural shape full: B2 L512 DM384 V256 H6 FF1024 layers mamba3+attention+mamba3+attention)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/measure-0852/ingest0849/inputs/amd-neural.samba-forward.bytes.shape=full.json (amd2/paired-7a507cd2c9d3ab421de8264eea972db86dd4b790cd48541c4abf6c52b004e360)`, ran on enc1-gpuvm005

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32: - (no completed torch fp32 arm). Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI300X (amd2, Hot Aisle) | 0.8.49 | identical | 8.8 | 8.8..8.8 | 1 | - | - | 8.8 | - | - (stored whole) | - | - | 2401.6 | 2709.8 | mean_nll=5.640151 | - | main board, one scored run | wheel | ok (main@80ace6b7a amd2/paired-7a507cd2c9d3ab421de8264eea972db86dd4b790cd48541c4abf6c52b004e360 2026-10-16; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(error: {"error": "worker returned no protocol message: worker_exited", "event": "error", "read_failure": {"deadline_s": 1800, "elapsed_s": 5.4883972390016424, "reason": "eof"}, "reason": "worker_exited", "re) (measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(error: {"error": "REFUSED: torch-compile-fp32 on cuda failed in round 0 (compile happens here): InductorError('AssertionError: \\n\\nSet TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this esp) (measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(error: {"error": "worker returned no protocol message: worker_exited", "event": "error", "read_failure": {"deadline_s": 1800, "elapsed_s": 2.2070068520006316, "reason": "eof"}, "reason": "worker_exited", "re) (measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(error: {"error": "REFUSED: torch-compile-bf16 on cuda failed in round 0 (compile happens here): InductorError('AssertionError: \\n\\nSet TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this esp) (measured this run) |

calls:
- `torch-eager-fp32`: `mamba_ssm.Block(config_readback='{"d_model": 384, "head_dim": 64, "intermedia..., dropout=0, eps=1e-05, max_grad_norm='n/a', rope_theta=10000, seed=7)`
- `torch-compile-fp32`: `mamba_ssm.Block(config_readback='{"d_model": 384, "head_dim": 64, "intermedia..., dropout=0, eps=1e-05, max_grad_norm='n/a', rope_theta=10000, seed=7)`
- `torch-eager-bf16`: `mamba_ssm.Block(config_readback='{"d_model": 384, "head_dim": 64, "intermedia..., dropout=0, eps=1e-05, max_grad_norm='n/a', rope_theta=10000, seed=7)`
- `torch-compile-bf16`: `mamba_ssm.Block(config_readback='{"d_model": 384, "head_dim": 64, "intermedia..., dropout=0, eps=1e-05, max_grad_norm='n/a', rope_theta=10000, seed=7)`

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/137772/vram_* bytes for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host not sampled; GPU not sampled

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| config_readback | "{\"d_model\": 384, \"head_dim\": 64, \"intermediate\": 1024, \"layers\": [\"mamba3\", \"attention\", \"mamba3\", \"attention\"], \"n_heads\": 6, \"n_kv_heads\": 6, \"norm_eps\": \"9.999999747378752e-06\", \"tied_head\": true, \"vocab\": 256}" | "{\"d_model\": 384, \"head_dim\": 64, \"intermediate\": 1024, \"layers\": [\"mamba3\", \"attention\", \"mamba3\", \"attention\"], \"n_heads\": 6, \"n_kv_heads\": 6, \"norm_eps\": \"9.999999747378752e-06\", \"tied_head\": true, \"vocab\": 256}" | "{\"d_model\": 384, \"head_dim\": 64, \"intermediate\": 1024, \"layers\": [\"mamba3\", \"attention\", \"mamba3\", \"attention\"], \"n_heads\": 6, \"n_kv_heads\": 6, \"norm_eps\": \"9.999999747378752e-06\", \"tied_head\": true, \"vocab\": 256}" | "{\"d_model\": 384, \"head_dim\": 64, \"intermediate\": 1024, \"layers\": [\"mamba3\", \"attention\", \"mamba3\", \"attention\"], \"n_heads\": 6, \"n_kv_heads\": 6, \"norm_eps\": \"9.999999747378752e-06\", \"tied_head\": true, \"vocab\": 256}" | "{\"d_model\": 384, \"head_dim\": 64, \"intermediate\": 1024, \"layers\": [\"mamba3\", \"attention\", \"mamba3\", \"attention\"], \"n_heads\": 6, \"n_kv_heads\": 6, \"norm_eps\": \"9.999999747378752e-06\", \"tied_head\": true, \"vocab\": 256}" |
| dropout | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| eps | 1e-05 | 1e-05 | 1e-05 | 1e-05 | 1e-05 |
| max_grad_norm | "n/a" | "n/a" | "n/a" | "n/a" | "n/a" |
| rope_theta | 10000.0 | 10000.0 | 10000.0 | 10000.0 | 10000.0 |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 |

### samba-train-step / bytes (neural shape full: B2 L512 DM384 V256 H6 FF1024 layers mamba3+attention+mamba3+attention)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/measure-0852/ingest0849/inputs/amd-neural.samba-train-step.bytes.shape=full.json (amd2/paired-4fc1d4fa5626cacc1a299fc4fcf4d0675d57fbcdeebe2c30633f7db3a7347f05)`, ran on enc1-gpuvm005

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32: - (no completed torch fp32 arm). Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI300X (amd2, Hot Aisle) | 0.8.49 | identical | 54.0 | 54.0..54.0 | 1 | - | - | 54.0 | - | - (stored whole) | - | - | 2403.3 | 2710.8 | loss_first_step=5.640152, loss_last_step=4.822321, steps=2 | - | main board, one scored run | wheel | ok (main@80ace6b7a amd2/paired-4fc1d4fa5626cacc1a299fc4fcf4d0675d57fbcdeebe2c30633f7db3a7347f05 2026-10-16; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(error: {"error": "worker returned no protocol message: worker_exited", "event": "error", "read_failure": {"deadline_s": 1800, "elapsed_s": 5.3416759240026295, "reason": "eof"}, "reason": "worker_exited", "re) (measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(error: {"error": "REFUSED: torch-compile-fp32 on cuda failed in round 0 (compile happens here): InductorError('AssertionError: \\n\\nSet TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this esp) (measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(error: {"error": "worker returned no protocol message: worker_exited", "event": "error", "read_failure": {"deadline_s": 1800, "elapsed_s": 2.123334418000013, "reason": "eof"}, "reason": "worker_exited", "ret) (measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(error: {"error": "REFUSED: torch-compile-bf16 on cuda failed in round 0 (compile happens here): InductorError('AssertionError: \\n\\nSet TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this esp) (measured this run) |

calls:
- `torch-eager-fp32`: `mamba_ssm.Block(accumulation_steps=1, amsgrad=False, betas=[0.9, 0.999], capturable=False, config_readback='{"d_model": 384, "head_dim": 64, "intermedia..., decoupled_weight_decay=True, differentiable=False, dropout=0, eps=1e-08, lr=0.001, ...)`
- `torch-compile-fp32`: `mamba_ssm.Block(accumulation_steps=1, amsgrad=False, betas=[0.9, 0.999], capturable=False, config_readback='{"d_model": 384, "head_dim": 64, "intermedia..., decoupled_weight_decay=True, differentiable=False, dropout=0, eps=1e-08, lr=0.001, ...)`
- `torch-eager-bf16`: `mamba_ssm.Block(accumulation_steps=1, amsgrad=False, betas=[0.9, 0.999], capturable=False, config_readback='{"d_model": 384, "head_dim": 64, "intermedia..., decoupled_weight_decay=True, differentiable=False, dropout=0, eps=1e-08, lr=0.001, ...)`
- `torch-compile-bf16`: `mamba_ssm.Block(accumulation_steps=1, amsgrad=False, betas=[0.9, 0.999], capturable=False, config_readback='{"d_model": 384, "head_dim": 64, "intermedia..., decoupled_weight_decay=True, differentiable=False, dropout=0, eps=1e-08, lr=0.001, ...)`

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/139362/vram_* bytes for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host not sampled; GPU not sampled

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| accumulation_steps | 1 | 1 | 1 | 1 | 1 |
| amsgrad | false | false | false | false | false |
| betas | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| capturable | - | false | false | false | false |
| config_readback | "{\"d_model\": 384, \"head_dim\": 64, \"intermediate\": 1024, \"layers\": [\"mamba3\", \"attention\", \"mamba3\", \"attention\"], \"n_heads\": 6, \"n_kv_heads\": 6, \"norm_eps\": \"9.999999747378752e-06\", \"tied_head\": true, \"vocab\": 256}" | "{\"d_model\": 384, \"head_dim\": 64, \"intermediate\": 1024, \"layers\": [\"mamba3\", \"attention\", \"mamba3\", \"attention\"], \"n_heads\": 6, \"n_kv_heads\": 6, \"norm_eps\": \"9.999999747378752e-06\", \"tied_head\": true, \"vocab\": 256}" | "{\"d_model\": 384, \"head_dim\": 64, \"intermediate\": 1024, \"layers\": [\"mamba3\", \"attention\", \"mamba3\", \"attention\"], \"n_heads\": 6, \"n_kv_heads\": 6, \"norm_eps\": \"9.999999747378752e-06\", \"tied_head\": true, \"vocab\": 256}" | "{\"d_model\": 384, \"head_dim\": 64, \"intermediate\": 1024, \"layers\": [\"mamba3\", \"attention\", \"mamba3\", \"attention\"], \"n_heads\": 6, \"n_kv_heads\": 6, \"norm_eps\": \"9.999999747378752e-06\", \"tied_head\": true, \"vocab\": 256}" | "{\"d_model\": 384, \"head_dim\": 64, \"intermediate\": 1024, \"layers\": [\"mamba3\", \"attention\", \"mamba3\", \"attention\"], \"n_heads\": 6, \"n_kv_heads\": 6, \"norm_eps\": \"9.999999747378752e-06\", \"tied_head\": true, \"vocab\": 256}" |
| decoupled_weight_decay | true | true | true | true | true |
| differentiable | - | false | false | false | false |
| dropout | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| eps | 1e-08 | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| foreach | - | null | null | null | null |
| fused | - | null | null | null | null |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| max_grad_norm | "off" | "off" | "off" | "off" | "off" |
| maximize | false | false | false | false | false |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 |
| weight_decay | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |

accepted difference: torch-eager-fp32 capturable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-eager-fp32 differentiable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-eager-fp32 foreach: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

accepted difference: torch-eager-fp32 fused: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

accepted difference: torch-compile-fp32 capturable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-compile-fp32 differentiable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-compile-fp32 foreach: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

accepted difference: torch-compile-fp32 fused: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

accepted difference: torch-eager-bf16 capturable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-eager-bf16 differentiable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-eager-bf16 foreach: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

accepted difference: torch-eager-bf16 fused: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

accepted difference: torch-compile-bf16 capturable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-compile-bf16 differentiable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-compile-bf16 foreach: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

accepted difference: torch-compile-bf16 fused: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

### transformer-decode / gaussian (neural shape full: B1 L2048 DM384 H6 KV6 HD64 FF1024)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/measure-0852/ingest0849/inputs/amd-neural.transformer-decode.gaussian.shape=full.json (amd2/paired-c8b2effa040db92f41822768366e94fb671e4a856f18075f28256768c0d07b8e)`, ran on enc1-gpuvm005

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 1441.6 ms) = 1.741; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 1288.1 ms) = 1.948. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI300X (amd2, Hot Aisle) | 0.8.49 | identical | 2509.8 | 2509.8..2509.8 | 1 | - | - | 2509.8 | - | - (stored whole) | - | - | 2030.2 | 1700.5 | host_reference=TransformerBlockInference, host_reference_atol=1e-06, host_reference_finite=True, host_reference_passed=True, host_reference_rtol=1e-05, host_reference_tolerance_ratio=0.000000, max_abs_diff_vs_host=0.000000, max_rel_diff_vs_host=0.000000 | - | main board, one scored run | wheel | ok (main@80ace6b7a amd2/paired-c8b2effa040db92f41822768366e94fb671e4a856f18075f28256768c0d07b8e 2026-10-16; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | 1288.1 | 1288.1..1288.1 | 1 | 1.948 | - | 1288.1 | - | - (stored whole) | 1.948 (whole/whole) | - | 4014.4 | 99.3 | max_abs_diff_vs_ours=4.768e-07, max_rel_diff_vs_ours=1.02e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | 1441.6 | 1441.6..1441.6 | 1 | 1.741 | - | 1441.6 | - | - (stored whole) | 1.741 (whole/whole) | - | 3734.0 | 178.7 | max_abs_diff_vs_ours=0.001819, max_rel_diff_vs_ours=0.0003893 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

calls:
- `torch-eager-fp32`: `transformers.LlamaDecoderLayer(eps=1e-06, rope_theta=10000, seed=7)`
- `torch-eager-bf16`: `transformers.LlamaDecoderLayer(eps=1e-06, rope_theta=10000, seed=7)`

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/140741/vram_* bytes for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-eager-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) |
| eps | 1e-06 | 1e-06 | 1e-06 |
| rope_theta | 10000.0 | 10000.0 | 10000.0 |
| seed | "none (deterministic)" | 7 | 7 |

### transformer-forward / gaussian (neural shape full: B1 L2048 DM384 H6 KV6 HD64 FF1024)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/measure-0852/ingest0849/inputs/amd-neural.transformer-forward.gaussian.shape=full.json (amd2/paired-2668ab306f2c6f14124c6d8d9c5ff5b1fa1e4084d2b53a7701856fb3fc5adff4)`, ran on enc1-gpuvm005

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 2.5 ms) = 3.850; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 2.1 ms) = 4.489. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI300X (amd2, Hot Aisle) | 0.8.49 | identical | 9.6 | 9.6..9.6 | 1 | - | - | 9.6 | - | - (stored whole) | - | - | 2214.1 | 2528.0 | host_reference=TransformerBlockInference, host_reference_atol=1e-06, host_reference_finite=True, host_reference_passed=True, host_reference_rtol=1e-05, host_reference_tolerance_ratio=0.000000, max_abs_diff_vs_host=0.000000, max_rel_diff_vs_host=0.000000, max_rel_diff_vs_torch_eager_fp32=0.015169, rel_fro_vs_torch_eager_fp32=5.717e-08 | - | main board, one scored run | wheel | ok (main@80ace6b7a amd2/paired-2668ab306f2c6f14124c6d8d9c5ff5b1fa1e4084d2b53a7701856fb3fc5adff4 2026-10-16; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | 19.5 | 19.5..19.5 | 1 | 0.493 | - | 19.5 | - | - (stored whole) | 0.493 (whole/whole) | - | 3828.4 | 116.8 | max_abs_diff_vs_ours=4.768e-07, max_rel_diff_vs_ours=1.02e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | 2.1 | 2.1..2.1 | 1 | 4.489 | - | 2.1 | - | - (stored whole) | 4.489 (whole/whole) | - | 4029.2 | 109.8 | max_abs_diff_vs_ours=4.768e-07, max_rel_diff_vs_ours=1.02e-07, max_rel_diff_vs_torch_eager_fp32=0.005081, rel_fro_vs_torch_eager_fp32=3.096e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | 5.3 | 5.3..5.3 | 1 | 1.827 | - | 5.3 | - | - (stored whole) | 1.827 (whole/whole) | - | 3543.8 | 183.6 | max_abs_diff_vs_ours=0.001819, max_rel_diff_vs_ours=0.0003893, max_rel_diff_vs_torch_eager_fp32=76.084014, rel_fro_vs_torch_eager_fp32=0.0002764 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | 2.5 | 2.5..2.5 | 1 | 3.850 | - | 2.5 | - | - (stored whole) | 3.850 (whole/whole) | - | 3786.3 | 175.5 | max_abs_diff_vs_ours=0.001819, max_rel_diff_vs_ours=0.0003893, max_rel_diff_vs_torch_eager_fp32=41.369141, rel_fro_vs_torch_eager_fp32=0.0002629 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

calls:
- `torch-eager-fp32`: `transformers.LlamaDecoderLayer(eps=1e-06, rope_theta=10000, seed=7)`
- `torch-compile-fp32`: `transformers.LlamaDecoderLayer(eps=1e-06, rope_theta=10000, seed=7)`
- `torch-eager-bf16`: `transformers.LlamaDecoderLayer(eps=1e-06, rope_theta=10000, seed=7)`
- `torch-compile-bf16`: `transformers.LlamaDecoderLayer(eps=1e-06, rope_theta=10000, seed=7)`

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/142228/vram_* bytes for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| eps | 1e-06 | 1e-06 | 1e-06 | 1e-06 | 1e-06 |
| rope_theta | 10000.0 | 10000.0 | 10000.0 | 10000.0 | 10000.0 |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 |

## Algorithm expansion

### adafactor / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/measure-0852/ingest0849/inputs/amd-algos.adafactor.synthetic.rows=full.json (amd2/paired-536d9d65f7b4cf19a0c71e1c69b0ed243f40c3c573733ec00153c696ef8fafce)`, ran on enc1-gpuvm005

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 9.1 ms) = 3.975. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI300X (amd2, Hot Aisle) | 0.8.49 | identical | 36.3 | 36.3..36.3 | 1 | - | - | - | 36.3 | - (stored kernel) | - | - | 1729.1 | 1877.8 | optimizer_update_error_vs_torch_fp32=3.386e-06, rel_fro_vs_torch_eager_fp32=1.033e-08, relative_error_vs_own_host=0.000000, scorer=scored | - | main board, one scored run | wheel | ok (main@80ace6b7a amd2/paired-536d9d65f7b4cf19a0c71e1c69b0ed243f40c3c573733ec00153c696ef8fafce 2026-10-16; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | 11.8 | 11.8..11.8 | 1 | 3.080 | - | 193.5 | 11.8 | 181.77 (upload_ms_untimed) | 3.080 (kernel/kernel) | - | 2802.5 | 832.0 | optimizer_update_error_vs_torch_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | 9.1 | 9.1..9.1 | 1 | 3.975 | - | 193.2 | 9.1 | 184.05 (upload_ms_untimed) | 3.975 (kernel/kernel) | - | 2846.0 | 832.0 | optimizer_update_error_vs_torch_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

calls:
- `ours`: `mojolearn.Adafactor(beta2_decay=-0.8, d=1, eps=[None, 0.001], lr=0.001, maximize=False, weight_decay=0, p0=float32[16777216], grad0=float32[16777216], grad1=float32[16777216], grad2=float32[16777216], grad3=float32[16777216], ...)`
- `torch-eager-fp32`: `torch.Adafactor(beta2_decay=-0.8, d=1, eps=[None, 0.001], foreach=True, lr=0.001, maximize=False, weight_decay=0, p0=float32[16777216], grad0=float32[16777216], grad1=float32[16777216], grad2=float32[16777216], grad3=float32[16777216], ...)`
- `torch-compile-fp32`: `torch.Adafactor(beta2_decay=-0.8, d=1, eps=[None, 0.001], foreach=True, lr=0.001, maximize=False, weight_decay=0, p0=float32[16777216], grad0=float32[16777216], grad1=float32[16777216], grad2=float32[16777216], grad3=float32[16777216], ...)`

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/143554/vram_* bytes for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'beta2_decay': -0.8, 'd': 1.0, 'eps': [None, 0.001], 'lr': 0.001, 'maximize': False, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-fp32 | torch-eager-fp32 |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (optimizer.defaults) | torch (optimizer.defaults) |
| beta2_decay | -0.8 | -0.8 | -0.8 |
| d | 1.0 | 1.0 | 1.0 |
| eps | [null, 0.001] | [null, 0.001] | [null, 0.001] |
| foreach | - | true | true |
| input_arrays | [{"dtype": "float32", "name": "p0", "sha256": "485a7287793a9d7d33d710ea2820ea9c7519d31100db4bafeee813baa532dd7d", "shape": [16777216]}, {"dtype": "float32", "name": "grad0", "sha256": "e6c0159d56a2df748f38d3f3592321edc11067354f52e751e382384dca4178bc", "shape": [16777216]}, {"dtype": "float32", "name": "grad1", "sha256": "249bb61249ee25b0cd42e6bcaa012c27c565dde4ccd798448a60e935fd4ace7e", "shape": [16777216]}, {"dtype": "float32", "name": "grad2", "sha256": "4cf167fba649d3d82fdb6c7c41749a74e8fc8a9bd70491154f03bfc9272b2ed6", "shape": [16777216]}, {"dtype": "float32", "name": "grad3", "sha256": "594f963cfcef38863774b145fac84f56ae7f37190af1cf270e7cb23d3b86b587", "shape": [16777216]}, {"dtype": "float32", "name": "grad4", "sha256": "ee091b2ac84630d1d0328c337b12c5bb6bd6ba2b28f3350eab106bc848960f4e", "shape": [16777216]}, {"dtype": "float32", "name": "grad5", "sha256": "a042832a1a0884016cafe5aada2f1df6bf0b09422e9358536881a536fff70d02", "shape": [16777216]}, {"dtype": "float32", "name": "grad6", "sha256": "736aade529a9b43a7666f9d5a77bf9ef70c253bb5a26280143d941c915910763", "shape": [16777216]}, {"dtype": "float32", "name": "grad7", "sha256": "c4fdaf0915399dc3568c7fd4a49465f9e3c69a10d3e77b1fc15d926db244f302", "shape": [16777216]}, {"dtype": "float32", "name": "grad8", "sha256": "6b970f1bc919f22be37581526578d65a535e33dd886ff46cc64d95aca61fec12", "shape": [16777216]}, {"dtype": "float32", "name": "grad9", "sha256": "00adaa7225e175a85e28e59772e3afbb901e068e5ca8d7332adfa127f7fcbdd7", "shape": [16777216]}] | [{"dtype": "float32", "name": "p0", "sha256": "485a7287793a9d7d33d710ea2820ea9c7519d31100db4bafeee813baa532dd7d", "shape": [16777216]}, {"dtype": "float32", "name": "grad0", "sha256": "e6c0159d56a2df748f38d3f3592321edc11067354f52e751e382384dca4178bc", "shape": [16777216]}, {"dtype": "float32", "name": "grad1", "sha256": "249bb61249ee25b0cd42e6bcaa012c27c565dde4ccd798448a60e935fd4ace7e", "shape": [16777216]}, {"dtype": "float32", "name": "grad2", "sha256": "4cf167fba649d3d82fdb6c7c41749a74e8fc8a9bd70491154f03bfc9272b2ed6", "shape": [16777216]}, {"dtype": "float32", "name": "grad3", "sha256": "594f963cfcef38863774b145fac84f56ae7f37190af1cf270e7cb23d3b86b587", "shape": [16777216]}, {"dtype": "float32", "name": "grad4", "sha256": "ee091b2ac84630d1d0328c337b12c5bb6bd6ba2b28f3350eab106bc848960f4e", "shape": [16777216]}, {"dtype": "float32", "name": "grad5", "sha256": "a042832a1a0884016cafe5aada2f1df6bf0b09422e9358536881a536fff70d02", "shape": [16777216]}, {"dtype": "float32", "name": "grad6", "sha256": "736aade529a9b43a7666f9d5a77bf9ef70c253bb5a26280143d941c915910763", "shape": [16777216]}, {"dtype": "float32", "name": "grad7", "sha256": "c4fdaf0915399dc3568c7fd4a49465f9e3c69a10d3e77b1fc15d926db244f302", "shape": [16777216]}, {"dtype": "float32", "name": "grad8", "sha256": "6b970f1bc919f22be37581526578d65a535e33dd886ff46cc64d95aca61fec12", "shape": [16777216]}, {"dtype": "float32", "name": "grad9", "sha256": "00adaa7225e175a85e28e59772e3afbb901e068e5ca8d7332adfa127f7fcbdd7", "shape": [16777216]}] | [{"dtype": "float32", "name": "p0", "sha256": "485a7287793a9d7d33d710ea2820ea9c7519d31100db4bafeee813baa532dd7d", "shape": [16777216]}, {"dtype": "float32", "name": "grad0", "sha256": "e6c0159d56a2df748f38d3f3592321edc11067354f52e751e382384dca4178bc", "shape": [16777216]}, {"dtype": "float32", "name": "grad1", "sha256": "249bb61249ee25b0cd42e6bcaa012c27c565dde4ccd798448a60e935fd4ace7e", "shape": [16777216]}, {"dtype": "float32", "name": "grad2", "sha256": "4cf167fba649d3d82fdb6c7c41749a74e8fc8a9bd70491154f03bfc9272b2ed6", "shape": [16777216]}, {"dtype": "float32", "name": "grad3", "sha256": "594f963cfcef38863774b145fac84f56ae7f37190af1cf270e7cb23d3b86b587", "shape": [16777216]}, {"dtype": "float32", "name": "grad4", "sha256": "ee091b2ac84630d1d0328c337b12c5bb6bd6ba2b28f3350eab106bc848960f4e", "shape": [16777216]}, {"dtype": "float32", "name": "grad5", "sha256": "a042832a1a0884016cafe5aada2f1df6bf0b09422e9358536881a536fff70d02", "shape": [16777216]}, {"dtype": "float32", "name": "grad6", "sha256": "736aade529a9b43a7666f9d5a77bf9ef70c253bb5a26280143d941c915910763", "shape": [16777216]}, {"dtype": "float32", "name": "grad7", "sha256": "c4fdaf0915399dc3568c7fd4a49465f9e3c69a10d3e77b1fc15d926db244f302", "shape": [16777216]}, {"dtype": "float32", "name": "grad8", "sha256": "6b970f1bc919f22be37581526578d65a535e33dd886ff46cc64d95aca61fec12", "shape": [16777216]}, {"dtype": "float32", "name": "grad9", "sha256": "00adaa7225e175a85e28e59772e3afbb901e068e5ca8d7332adfa127f7fcbdd7", "shape": [16777216]}] |
| learning_rate | 0.001 | 0.001 | 0.001 |
| maximize | false | false | false |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| weight_decay | 0.0 | 0.0 | 0.0 |

accepted difference: torch-eager-fp32 foreach: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

accepted difference: torch-compile-fp32 foreach: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

### adagrad / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/measure-0852/ingest0849/inputs/amd-algos.adagrad.synthetic.rows=full.json (amd2/paired-36d5e26e897fd6ba5f2f52e695b3e1e92416e2553482e30e1134e92d13b1095a)`, ran on enc1-gpuvm005

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 3.6 ms) = 1.791. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI300X (amd2, Hot Aisle) | 0.8.49 | identical | 6.5 | 6.5..6.5 | 1 | - | - | - | 6.5 | - (stored kernel) | - | - | 1727.9 | 1877.7 | optimizer_update_error_vs_torch_fp32=1.874e-06, rel_fro_vs_torch_eager_fp32=3.207e-09, relative_error_vs_own_host=0.000000, scorer=scored | - | main board, one scored run | wheel | ok (main@80ace6b7a amd2/paired-36d5e26e897fd6ba5f2f52e695b3e1e92416e2553482e30e1134e92d13b1095a 2026-10-16; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | 5.8 | 5.8..5.8 | 1 | 1.118 | - | 207.2 | 5.8 | 201.40 (upload_ms_untimed) | 1.118 (kernel/kernel) | - | 2784.4 | 960.0 | optimizer_update_error_vs_torch_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | 3.6 | 3.6..3.6 | 1 | 1.791 | - | 207.8 | 3.6 | 204.15 (upload_ms_untimed) | 1.791 (kernel/kernel) | - | 3131.2 | 896.0 | optimizer_update_error_vs_torch_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

calls:
- `ours`: `mojolearn.Adagrad(eps=1e-10, initial_accumulator_value=0, lr=0.001, lr_decay=0, maximize=False, weight_decay=0, p0=float32[16777216], grad0=float32[16777216], grad1=float32[16777216], grad2=float32[16777216], grad3=float32[16777216], ...)`
- `torch-eager-fp32`: `torch.Adagrad(differentiable=False, eps=1e-10, foreach=True, initial_accumulator_value=0, lr=0.001, lr_decay=0, maximize=False, weight_decay=0, p0=float32[16777216], grad0=float32[16777216], grad1=float32[16777216], grad2=float32[16777216], ...)`
- `torch-compile-fp32`: `torch.Adagrad(differentiable=False, eps=1e-10, foreach=True, initial_accumulator_value=0, lr=0.001, lr_decay=0, maximize=False, weight_decay=0, p0=float32[16777216], grad0=float32[16777216], grad1=float32[16777216], grad2=float32[16777216], ...)`

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/144414/vram_* bytes for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'eps': 1e-10, 'initial_accumulator_value': 0.0, 'lr': 0.001, 'lr_decay': 0.0, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-fp32 | torch-eager-fp32 |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (optimizer.defaults) | torch (optimizer.defaults) |
| differentiable | - | false | false |
| eps | 1e-10 | 1e-10 | 1e-10 |
| foreach | - | true | true |
| fused | - | null | null |
| initial_accumulator_value | 0.0 | 0.0 | 0.0 |
| input_arrays | [{"dtype": "float32", "name": "p0", "sha256": "485a7287793a9d7d33d710ea2820ea9c7519d31100db4bafeee813baa532dd7d", "shape": [16777216]}, {"dtype": "float32", "name": "grad0", "sha256": "e6c0159d56a2df748f38d3f3592321edc11067354f52e751e382384dca4178bc", "shape": [16777216]}, {"dtype": "float32", "name": "grad1", "sha256": "249bb61249ee25b0cd42e6bcaa012c27c565dde4ccd798448a60e935fd4ace7e", "shape": [16777216]}, {"dtype": "float32", "name": "grad2", "sha256": "4cf167fba649d3d82fdb6c7c41749a74e8fc8a9bd70491154f03bfc9272b2ed6", "shape": [16777216]}, {"dtype": "float32", "name": "grad3", "sha256": "594f963cfcef38863774b145fac84f56ae7f37190af1cf270e7cb23d3b86b587", "shape": [16777216]}, {"dtype": "float32", "name": "grad4", "sha256": "ee091b2ac84630d1d0328c337b12c5bb6bd6ba2b28f3350eab106bc848960f4e", "shape": [16777216]}, {"dtype": "float32", "name": "grad5", "sha256": "a042832a1a0884016cafe5aada2f1df6bf0b09422e9358536881a536fff70d02", "shape": [16777216]}, {"dtype": "float32", "name": "grad6", "sha256": "736aade529a9b43a7666f9d5a77bf9ef70c253bb5a26280143d941c915910763", "shape": [16777216]}, {"dtype": "float32", "name": "grad7", "sha256": "c4fdaf0915399dc3568c7fd4a49465f9e3c69a10d3e77b1fc15d926db244f302", "shape": [16777216]}, {"dtype": "float32", "name": "grad8", "sha256": "6b970f1bc919f22be37581526578d65a535e33dd886ff46cc64d95aca61fec12", "shape": [16777216]}, {"dtype": "float32", "name": "grad9", "sha256": "00adaa7225e175a85e28e59772e3afbb901e068e5ca8d7332adfa127f7fcbdd7", "shape": [16777216]}] | [{"dtype": "float32", "name": "p0", "sha256": "485a7287793a9d7d33d710ea2820ea9c7519d31100db4bafeee813baa532dd7d", "shape": [16777216]}, {"dtype": "float32", "name": "grad0", "sha256": "e6c0159d56a2df748f38d3f3592321edc11067354f52e751e382384dca4178bc", "shape": [16777216]}, {"dtype": "float32", "name": "grad1", "sha256": "249bb61249ee25b0cd42e6bcaa012c27c565dde4ccd798448a60e935fd4ace7e", "shape": [16777216]}, {"dtype": "float32", "name": "grad2", "sha256": "4cf167fba649d3d82fdb6c7c41749a74e8fc8a9bd70491154f03bfc9272b2ed6", "shape": [16777216]}, {"dtype": "float32", "name": "grad3", "sha256": "594f963cfcef38863774b145fac84f56ae7f37190af1cf270e7cb23d3b86b587", "shape": [16777216]}, {"dtype": "float32", "name": "grad4", "sha256": "ee091b2ac84630d1d0328c337b12c5bb6bd6ba2b28f3350eab106bc848960f4e", "shape": [16777216]}, {"dtype": "float32", "name": "grad5", "sha256": "a042832a1a0884016cafe5aada2f1df6bf0b09422e9358536881a536fff70d02", "shape": [16777216]}, {"dtype": "float32", "name": "grad6", "sha256": "736aade529a9b43a7666f9d5a77bf9ef70c253bb5a26280143d941c915910763", "shape": [16777216]}, {"dtype": "float32", "name": "grad7", "sha256": "c4fdaf0915399dc3568c7fd4a49465f9e3c69a10d3e77b1fc15d926db244f302", "shape": [16777216]}, {"dtype": "float32", "name": "grad8", "sha256": "6b970f1bc919f22be37581526578d65a535e33dd886ff46cc64d95aca61fec12", "shape": [16777216]}, {"dtype": "float32", "name": "grad9", "sha256": "00adaa7225e175a85e28e59772e3afbb901e068e5ca8d7332adfa127f7fcbdd7", "shape": [16777216]}] | [{"dtype": "float32", "name": "p0", "sha256": "485a7287793a9d7d33d710ea2820ea9c7519d31100db4bafeee813baa532dd7d", "shape": [16777216]}, {"dtype": "float32", "name": "grad0", "sha256": "e6c0159d56a2df748f38d3f3592321edc11067354f52e751e382384dca4178bc", "shape": [16777216]}, {"dtype": "float32", "name": "grad1", "sha256": "249bb61249ee25b0cd42e6bcaa012c27c565dde4ccd798448a60e935fd4ace7e", "shape": [16777216]}, {"dtype": "float32", "name": "grad2", "sha256": "4cf167fba649d3d82fdb6c7c41749a74e8fc8a9bd70491154f03bfc9272b2ed6", "shape": [16777216]}, {"dtype": "float32", "name": "grad3", "sha256": "594f963cfcef38863774b145fac84f56ae7f37190af1cf270e7cb23d3b86b587", "shape": [16777216]}, {"dtype": "float32", "name": "grad4", "sha256": "ee091b2ac84630d1d0328c337b12c5bb6bd6ba2b28f3350eab106bc848960f4e", "shape": [16777216]}, {"dtype": "float32", "name": "grad5", "sha256": "a042832a1a0884016cafe5aada2f1df6bf0b09422e9358536881a536fff70d02", "shape": [16777216]}, {"dtype": "float32", "name": "grad6", "sha256": "736aade529a9b43a7666f9d5a77bf9ef70c253bb5a26280143d941c915910763", "shape": [16777216]}, {"dtype": "float32", "name": "grad7", "sha256": "c4fdaf0915399dc3568c7fd4a49465f9e3c69a10d3e77b1fc15d926db244f302", "shape": [16777216]}, {"dtype": "float32", "name": "grad8", "sha256": "6b970f1bc919f22be37581526578d65a535e33dd886ff46cc64d95aca61fec12", "shape": [16777216]}, {"dtype": "float32", "name": "grad9", "sha256": "00adaa7225e175a85e28e59772e3afbb901e068e5ca8d7332adfa127f7fcbdd7", "shape": [16777216]}] |
| learning_rate | 0.001 | 0.001 | 0.001 |
| lr_decay | 0.0 | 0.0 | 0.0 |
| maximize | false | false | false |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| weight_decay | 0.0 | 0.0 | 0.0 |

accepted difference: torch-eager-fp32 differentiable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-eager-fp32 foreach: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

accepted difference: torch-eager-fp32 fused: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

accepted difference: torch-compile-fp32 differentiable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-compile-fp32 foreach: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

accepted difference: torch-compile-fp32 fused: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

### adam / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/measure-0852/ingest0849/inputs/amd-algos.adam.synthetic.rows=full.json (amd2/paired-5f2f7488b2fbdf170d2f52790cd34e11fc02b2953f5ef9713657d01b0674f67a)`, ran on enc1-gpuvm005

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 7.1 ms) = 3.538. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI300X (amd2, Hot Aisle) | 0.8.49 | identical | 25.2 | 25.2..25.2 | 1 | - | - | - | 25.2 | - (stored kernel) | - | - | 3038.9 | 2069.7 | optimizer_update_error_vs_torch_fp32=8.844e-06, rel_fro_vs_torch_eager_fp32=3.34e-08, relative_error_vs_own_host=0.000000, scorer=scored | - | main board, one scored run | wheel | ok (main@80ace6b7a amd2/paired-5f2f7488b2fbdf170d2f52790cd34e11fc02b2953f5ef9713657d01b0674f67a 2026-10-16; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | 7.1 | 7.1..7.1 | 1 | 3.538 | - | 189.0 | 7.1 | 181.91 (upload_ms_untimed) | 3.538 (kernel/kernel) | - | 2784.6 | 896.0 | optimizer_update_error_vs_torch_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | 101.0 | 101.0..101.0 | 1 | 0.250 | - | 293.3 | 101.0 | 192.37 (upload_ms_untimed) | 0.250 (kernel/kernel) | - | 3226.6 | 1344.0 | optimizer_update_error_vs_torch_fp32=1.355e-05, rel_fro_vs_torch_eager_fp32=5.116e-08 | - | LIKE-FOR-LIKE-SPAN | - | recompiled-in-clock: 1 torch.compile compile(s) inside the timed run (measured this run) |
| torch-fused-fp32 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | 5.8 | 5.8..5.8 | 1 | 4.308 | - | 194.8 | 5.8 | 188.92 (upload_ms_untimed) | 4.308 (kernel/kernel) | - | 2715.7 | 896.0 | optimizer_update_error_vs_torch_fp32=6.002e-06, rel_fro_vs_torch_eager_fp32=2.267e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

calls:
- `ours`: `mojolearn.Adam(amsgrad=False, betas=[0.9, 0.999], decoupled_weight_decay=False, eps=1e-08, lr=0.001, maximize=False, weight_decay=0, p0=float32[16777216], grad0=float32[16777216], grad1=float32[16777216], grad2=float32[16777216], ...)`
- `torch-eager-fp32`: `torch.Adam(amsgrad=False, betas=[0.9, 0.999], capturable=False, decoupled_weight_decay=False, differentiable=False, eps=1e-08, foreach=True, lr=0.001, maximize=False, weight_decay=0, p0=float32[16777216], grad0=float32[16777216], ...)`
- `torch-compile-fp32`: `torch.Adam(amsgrad=False, betas=[0.9, 0.999], capturable=False, decoupled_weight_decay=False, differentiable=False, eps=1e-08, foreach=True, lr=0.001, maximize=False, weight_decay=0, p0=float32[16777216], grad0=float32[16777216], ...)`
- `torch-fused-fp32`: `torch.Adam(amsgrad=False, betas=[0.9, 0.999], capturable=False, decoupled_weight_decay=False, differentiable=False, eps=1e-08, fused=True, lr=0.001, maximize=False, weight_decay=0, p0=float32[16777216], grad0=float32[16777216], ...)`

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/145541/vram_* bytes for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-fused-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'betas': [0.9, 0.999], 'eps': 1e-08, 'lr': 0.001, 'maximize': False, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-fp32 | torch-eager-fp32 | torch-fused-fp32 |
|---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (optimizer.defaults) | torch (optimizer.defaults) | torch (optimizer.defaults) |
| amsgrad | false | false | false | false |
| betas | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| capturable | - | false | false | false |
| decoupled_weight_decay | false | false | false | false |
| differentiable | - | false | false | false |
| eps | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| foreach | - | true | true | null |
| fused | - | null | null | true |
| input_arrays | [{"dtype": "float32", "name": "p0", "sha256": "485a7287793a9d7d33d710ea2820ea9c7519d31100db4bafeee813baa532dd7d", "shape": [16777216]}, {"dtype": "float32", "name": "grad0", "sha256": "e6c0159d56a2df748f38d3f3592321edc11067354f52e751e382384dca4178bc", "shape": [16777216]}, {"dtype": "float32", "name": "grad1", "sha256": "249bb61249ee25b0cd42e6bcaa012c27c565dde4ccd798448a60e935fd4ace7e", "shape": [16777216]}, {"dtype": "float32", "name": "grad2", "sha256": "4cf167fba649d3d82fdb6c7c41749a74e8fc8a9bd70491154f03bfc9272b2ed6", "shape": [16777216]}, {"dtype": "float32", "name": "grad3", "sha256": "594f963cfcef38863774b145fac84f56ae7f37190af1cf270e7cb23d3b86b587", "shape": [16777216]}, {"dtype": "float32", "name": "grad4", "sha256": "ee091b2ac84630d1d0328c337b12c5bb6bd6ba2b28f3350eab106bc848960f4e", "shape": [16777216]}, {"dtype": "float32", "name": "grad5", "sha256": "a042832a1a0884016cafe5aada2f1df6bf0b09422e9358536881a536fff70d02", "shape": [16777216]}, {"dtype": "float32", "name": "grad6", "sha256": "736aade529a9b43a7666f9d5a77bf9ef70c253bb5a26280143d941c915910763", "shape": [16777216]}, {"dtype": "float32", "name": "grad7", "sha256": "c4fdaf0915399dc3568c7fd4a49465f9e3c69a10d3e77b1fc15d926db244f302", "shape": [16777216]}, {"dtype": "float32", "name": "grad8", "sha256": "6b970f1bc919f22be37581526578d65a535e33dd886ff46cc64d95aca61fec12", "shape": [16777216]}, {"dtype": "float32", "name": "grad9", "sha256": "00adaa7225e175a85e28e59772e3afbb901e068e5ca8d7332adfa127f7fcbdd7", "shape": [16777216]}] | [{"dtype": "float32", "name": "p0", "sha256": "485a7287793a9d7d33d710ea2820ea9c7519d31100db4bafeee813baa532dd7d", "shape": [16777216]}, {"dtype": "float32", "name": "grad0", "sha256": "e6c0159d56a2df748f38d3f3592321edc11067354f52e751e382384dca4178bc", "shape": [16777216]}, {"dtype": "float32", "name": "grad1", "sha256": "249bb61249ee25b0cd42e6bcaa012c27c565dde4ccd798448a60e935fd4ace7e", "shape": [16777216]}, {"dtype": "float32", "name": "grad2", "sha256": "4cf167fba649d3d82fdb6c7c41749a74e8fc8a9bd70491154f03bfc9272b2ed6", "shape": [16777216]}, {"dtype": "float32", "name": "grad3", "sha256": "594f963cfcef38863774b145fac84f56ae7f37190af1cf270e7cb23d3b86b587", "shape": [16777216]}, {"dtype": "float32", "name": "grad4", "sha256": "ee091b2ac84630d1d0328c337b12c5bb6bd6ba2b28f3350eab106bc848960f4e", "shape": [16777216]}, {"dtype": "float32", "name": "grad5", "sha256": "a042832a1a0884016cafe5aada2f1df6bf0b09422e9358536881a536fff70d02", "shape": [16777216]}, {"dtype": "float32", "name": "grad6", "sha256": "736aade529a9b43a7666f9d5a77bf9ef70c253bb5a26280143d941c915910763", "shape": [16777216]}, {"dtype": "float32", "name": "grad7", "sha256": "c4fdaf0915399dc3568c7fd4a49465f9e3c69a10d3e77b1fc15d926db244f302", "shape": [16777216]}, {"dtype": "float32", "name": "grad8", "sha256": "6b970f1bc919f22be37581526578d65a535e33dd886ff46cc64d95aca61fec12", "shape": [16777216]}, {"dtype": "float32", "name": "grad9", "sha256": "00adaa7225e175a85e28e59772e3afbb901e068e5ca8d7332adfa127f7fcbdd7", "shape": [16777216]}] | [{"dtype": "float32", "name": "p0", "sha256": "485a7287793a9d7d33d710ea2820ea9c7519d31100db4bafeee813baa532dd7d", "shape": [16777216]}, {"dtype": "float32", "name": "grad0", "sha256": "e6c0159d56a2df748f38d3f3592321edc11067354f52e751e382384dca4178bc", "shape": [16777216]}, {"dtype": "float32", "name": "grad1", "sha256": "249bb61249ee25b0cd42e6bcaa012c27c565dde4ccd798448a60e935fd4ace7e", "shape": [16777216]}, {"dtype": "float32", "name": "grad2", "sha256": "4cf167fba649d3d82fdb6c7c41749a74e8fc8a9bd70491154f03bfc9272b2ed6", "shape": [16777216]}, {"dtype": "float32", "name": "grad3", "sha256": "594f963cfcef38863774b145fac84f56ae7f37190af1cf270e7cb23d3b86b587", "shape": [16777216]}, {"dtype": "float32", "name": "grad4", "sha256": "ee091b2ac84630d1d0328c337b12c5bb6bd6ba2b28f3350eab106bc848960f4e", "shape": [16777216]}, {"dtype": "float32", "name": "grad5", "sha256": "a042832a1a0884016cafe5aada2f1df6bf0b09422e9358536881a536fff70d02", "shape": [16777216]}, {"dtype": "float32", "name": "grad6", "sha256": "736aade529a9b43a7666f9d5a77bf9ef70c253bb5a26280143d941c915910763", "shape": [16777216]}, {"dtype": "float32", "name": "grad7", "sha256": "c4fdaf0915399dc3568c7fd4a49465f9e3c69a10d3e77b1fc15d926db244f302", "shape": [16777216]}, {"dtype": "float32", "name": "grad8", "sha256": "6b970f1bc919f22be37581526578d65a535e33dd886ff46cc64d95aca61fec12", "shape": [16777216]}, {"dtype": "float32", "name": "grad9", "sha256": "00adaa7225e175a85e28e59772e3afbb901e068e5ca8d7332adfa127f7fcbdd7", "shape": [16777216]}] | [{"dtype": "float32", "name": "p0", "sha256": "485a7287793a9d7d33d710ea2820ea9c7519d31100db4bafeee813baa532dd7d", "shape": [16777216]}, {"dtype": "float32", "name": "grad0", "sha256": "e6c0159d56a2df748f38d3f3592321edc11067354f52e751e382384dca4178bc", "shape": [16777216]}, {"dtype": "float32", "name": "grad1", "sha256": "249bb61249ee25b0cd42e6bcaa012c27c565dde4ccd798448a60e935fd4ace7e", "shape": [16777216]}, {"dtype": "float32", "name": "grad2", "sha256": "4cf167fba649d3d82fdb6c7c41749a74e8fc8a9bd70491154f03bfc9272b2ed6", "shape": [16777216]}, {"dtype": "float32", "name": "grad3", "sha256": "594f963cfcef38863774b145fac84f56ae7f37190af1cf270e7cb23d3b86b587", "shape": [16777216]}, {"dtype": "float32", "name": "grad4", "sha256": "ee091b2ac84630d1d0328c337b12c5bb6bd6ba2b28f3350eab106bc848960f4e", "shape": [16777216]}, {"dtype": "float32", "name": "grad5", "sha256": "a042832a1a0884016cafe5aada2f1df6bf0b09422e9358536881a536fff70d02", "shape": [16777216]}, {"dtype": "float32", "name": "grad6", "sha256": "736aade529a9b43a7666f9d5a77bf9ef70c253bb5a26280143d941c915910763", "shape": [16777216]}, {"dtype": "float32", "name": "grad7", "sha256": "c4fdaf0915399dc3568c7fd4a49465f9e3c69a10d3e77b1fc15d926db244f302", "shape": [16777216]}, {"dtype": "float32", "name": "grad8", "sha256": "6b970f1bc919f22be37581526578d65a535e33dd886ff46cc64d95aca61fec12", "shape": [16777216]}, {"dtype": "float32", "name": "grad9", "sha256": "00adaa7225e175a85e28e59772e3afbb901e068e5ca8d7332adfa127f7fcbdd7", "shape": [16777216]}] |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 |
| maximize | false | false | false | false |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| weight_decay | 0.0 | 0.0 | 0.0 | 0.0 |

accepted difference: torch-eager-fp32 capturable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-eager-fp32 differentiable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-eager-fp32 foreach: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

accepted difference: torch-eager-fp32 fused: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

accepted difference: torch-compile-fp32 capturable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-compile-fp32 differentiable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-compile-fp32 foreach: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

accepted difference: torch-compile-fp32 fused: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

accepted difference: torch-fused-fp32 capturable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-fused-fp32 differentiable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-fused-fp32 foreach: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

accepted difference: torch-fused-fp32 fused: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

### adamax / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/measure-0852/ingest0849/inputs/amd-algos.adamax.synthetic.rows=full.json (amd2/paired-a035ce358d9ddc54fdf29c7727fa45ca8927c27a92e1d6791d795238cf934282)`, ran on enc1-gpuvm005

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 6.8 ms) = 0.702. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI300X (amd2, Hot Aisle) | 0.8.49 | identical | 4.8 | 4.8..4.8 | 1 | - | - | - | 4.8 | - (stored kernel) | - | - | 1727.8 | 1877.7 | optimizer_update_error_vs_torch_fp32=1.54e-06, rel_fro_vs_torch_eager_fp32=4.144e-09, relative_error_vs_own_host=0.000000, scorer=scored | - | main board, one scored run | wheel | ok (main@80ace6b7a amd2/paired-a035ce358d9ddc54fdf29c7727fa45ca8927c27a92e1d6791d795238cf934282 2026-10-16; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | 6.8 | 6.8..6.8 | 1 | 0.702 | - | 189.6 | 6.8 | 182.77 (upload_ms_untimed) | 0.702 (kernel/kernel) | - | 2784.6 | 896.0 | optimizer_update_error_vs_torch_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | 72.9 | 72.9..72.9 | 1 | 0.066 | - | 253.6 | 72.9 | 180.73 (upload_ms_untimed) | 0.066 (kernel/kernel) | - | 3261.7 | 1344.0 | optimizer_update_error_vs_torch_fp32=2.292e-06, rel_fro_vs_torch_eager_fp32=6.17e-09 | - | LIKE-FOR-LIKE-SPAN | - | recompiled-in-clock: 1 torch.compile compile(s) inside the timed run (measured this run) |

calls:
- `ours`: `mojolearn.Adamax(betas=[0.9, 0.999], eps=1e-08, lr=0.001, maximize=False, weight_decay=0, p0=float32[16777216], grad0=float32[16777216], grad1=float32[16777216], grad2=float32[16777216], grad3=float32[16777216], grad4=float32[16777216], ...)`
- `torch-eager-fp32`: `torch.Adamax(betas=[0.9, 0.999], capturable=False, differentiable=False, eps=1e-08, foreach=True, lr=0.001, maximize=False, weight_decay=0, p0=float32[16777216], grad0=float32[16777216], grad1=float32[16777216], grad2=float32[16777216], ...)`
- `torch-compile-fp32`: `torch.Adamax(betas=[0.9, 0.999], capturable=False, differentiable=False, eps=1e-08, foreach=True, lr=0.001, maximize=False, weight_decay=0, p0=float32[16777216], grad0=float32[16777216], grad1=float32[16777216], grad2=float32[16777216], ...)`

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/146578/vram_* bytes for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'betas': [0.9, 0.999], 'eps': 1e-08, 'lr': 0.001, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-fp32 | torch-eager-fp32 |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (optimizer.defaults) | torch (optimizer.defaults) |
| betas | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| capturable | - | false | false |
| differentiable | - | false | false |
| eps | 1e-08 | 1e-08 | 1e-08 |
| foreach | - | true | true |
| input_arrays | [{"dtype": "float32", "name": "p0", "sha256": "485a7287793a9d7d33d710ea2820ea9c7519d31100db4bafeee813baa532dd7d", "shape": [16777216]}, {"dtype": "float32", "name": "grad0", "sha256": "e6c0159d56a2df748f38d3f3592321edc11067354f52e751e382384dca4178bc", "shape": [16777216]}, {"dtype": "float32", "name": "grad1", "sha256": "249bb61249ee25b0cd42e6bcaa012c27c565dde4ccd798448a60e935fd4ace7e", "shape": [16777216]}, {"dtype": "float32", "name": "grad2", "sha256": "4cf167fba649d3d82fdb6c7c41749a74e8fc8a9bd70491154f03bfc9272b2ed6", "shape": [16777216]}, {"dtype": "float32", "name": "grad3", "sha256": "594f963cfcef38863774b145fac84f56ae7f37190af1cf270e7cb23d3b86b587", "shape": [16777216]}, {"dtype": "float32", "name": "grad4", "sha256": "ee091b2ac84630d1d0328c337b12c5bb6bd6ba2b28f3350eab106bc848960f4e", "shape": [16777216]}, {"dtype": "float32", "name": "grad5", "sha256": "a042832a1a0884016cafe5aada2f1df6bf0b09422e9358536881a536fff70d02", "shape": [16777216]}, {"dtype": "float32", "name": "grad6", "sha256": "736aade529a9b43a7666f9d5a77bf9ef70c253bb5a26280143d941c915910763", "shape": [16777216]}, {"dtype": "float32", "name": "grad7", "sha256": "c4fdaf0915399dc3568c7fd4a49465f9e3c69a10d3e77b1fc15d926db244f302", "shape": [16777216]}, {"dtype": "float32", "name": "grad8", "sha256": "6b970f1bc919f22be37581526578d65a535e33dd886ff46cc64d95aca61fec12", "shape": [16777216]}, {"dtype": "float32", "name": "grad9", "sha256": "00adaa7225e175a85e28e59772e3afbb901e068e5ca8d7332adfa127f7fcbdd7", "shape": [16777216]}] | [{"dtype": "float32", "name": "p0", "sha256": "485a7287793a9d7d33d710ea2820ea9c7519d31100db4bafeee813baa532dd7d", "shape": [16777216]}, {"dtype": "float32", "name": "grad0", "sha256": "e6c0159d56a2df748f38d3f3592321edc11067354f52e751e382384dca4178bc", "shape": [16777216]}, {"dtype": "float32", "name": "grad1", "sha256": "249bb61249ee25b0cd42e6bcaa012c27c565dde4ccd798448a60e935fd4ace7e", "shape": [16777216]}, {"dtype": "float32", "name": "grad2", "sha256": "4cf167fba649d3d82fdb6c7c41749a74e8fc8a9bd70491154f03bfc9272b2ed6", "shape": [16777216]}, {"dtype": "float32", "name": "grad3", "sha256": "594f963cfcef38863774b145fac84f56ae7f37190af1cf270e7cb23d3b86b587", "shape": [16777216]}, {"dtype": "float32", "name": "grad4", "sha256": "ee091b2ac84630d1d0328c337b12c5bb6bd6ba2b28f3350eab106bc848960f4e", "shape": [16777216]}, {"dtype": "float32", "name": "grad5", "sha256": "a042832a1a0884016cafe5aada2f1df6bf0b09422e9358536881a536fff70d02", "shape": [16777216]}, {"dtype": "float32", "name": "grad6", "sha256": "736aade529a9b43a7666f9d5a77bf9ef70c253bb5a26280143d941c915910763", "shape": [16777216]}, {"dtype": "float32", "name": "grad7", "sha256": "c4fdaf0915399dc3568c7fd4a49465f9e3c69a10d3e77b1fc15d926db244f302", "shape": [16777216]}, {"dtype": "float32", "name": "grad8", "sha256": "6b970f1bc919f22be37581526578d65a535e33dd886ff46cc64d95aca61fec12", "shape": [16777216]}, {"dtype": "float32", "name": "grad9", "sha256": "00adaa7225e175a85e28e59772e3afbb901e068e5ca8d7332adfa127f7fcbdd7", "shape": [16777216]}] | [{"dtype": "float32", "name": "p0", "sha256": "485a7287793a9d7d33d710ea2820ea9c7519d31100db4bafeee813baa532dd7d", "shape": [16777216]}, {"dtype": "float32", "name": "grad0", "sha256": "e6c0159d56a2df748f38d3f3592321edc11067354f52e751e382384dca4178bc", "shape": [16777216]}, {"dtype": "float32", "name": "grad1", "sha256": "249bb61249ee25b0cd42e6bcaa012c27c565dde4ccd798448a60e935fd4ace7e", "shape": [16777216]}, {"dtype": "float32", "name": "grad2", "sha256": "4cf167fba649d3d82fdb6c7c41749a74e8fc8a9bd70491154f03bfc9272b2ed6", "shape": [16777216]}, {"dtype": "float32", "name": "grad3", "sha256": "594f963cfcef38863774b145fac84f56ae7f37190af1cf270e7cb23d3b86b587", "shape": [16777216]}, {"dtype": "float32", "name": "grad4", "sha256": "ee091b2ac84630d1d0328c337b12c5bb6bd6ba2b28f3350eab106bc848960f4e", "shape": [16777216]}, {"dtype": "float32", "name": "grad5", "sha256": "a042832a1a0884016cafe5aada2f1df6bf0b09422e9358536881a536fff70d02", "shape": [16777216]}, {"dtype": "float32", "name": "grad6", "sha256": "736aade529a9b43a7666f9d5a77bf9ef70c253bb5a26280143d941c915910763", "shape": [16777216]}, {"dtype": "float32", "name": "grad7", "sha256": "c4fdaf0915399dc3568c7fd4a49465f9e3c69a10d3e77b1fc15d926db244f302", "shape": [16777216]}, {"dtype": "float32", "name": "grad8", "sha256": "6b970f1bc919f22be37581526578d65a535e33dd886ff46cc64d95aca61fec12", "shape": [16777216]}, {"dtype": "float32", "name": "grad9", "sha256": "00adaa7225e175a85e28e59772e3afbb901e068e5ca8d7332adfa127f7fcbdd7", "shape": [16777216]}] |
| learning_rate | 0.001 | 0.001 | 0.001 |
| maximize | false | false | false |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| weight_decay | 0.0 | 0.0 | 0.0 |

accepted difference: torch-eager-fp32 capturable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-eager-fp32 differentiable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-eager-fp32 foreach: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

accepted difference: torch-compile-fp32 capturable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-compile-fp32 differentiable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-compile-fp32 foreach: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

### adamw / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/measure-0852/ingest0849/inputs/amd-algos.adamw.synthetic.rows=full.json (amd2/paired-9b113f18580c422f955635ecb8a7eb122597ab4bf8e9b7608057578e66901dea)`, ran on enc1-gpuvm005

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 6.8 ms) = 6.181. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI300X (amd2, Hot Aisle) | 0.8.49 | identical | 42.1 | 42.1..42.1 | 1 | - | - | - | 42.1 | - (stored kernel) | - | - | 3038.7 | 2069.7 | optimizer_update_error_vs_torch_fp32=8.841e-06, rel_fro_vs_torch_eager_fp32=3.34e-08, relative_error_vs_own_host=0.000000, scorer=scored | - | main board, one scored run | wheel | ok (main@80ace6b7a amd2/paired-9b113f18580c422f955635ecb8a7eb122597ab4bf8e9b7608057578e66901dea 2026-10-16; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | 6.8 | 6.8..6.8 | 1 | 6.181 | - | 211.1 | 6.8 | 204.24 (upload_ms_untimed) | 6.181 (kernel/kernel) | - | 2784.5 | 896.0 | optimizer_update_error_vs_torch_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | 108.3 | 108.3..108.3 | 1 | 0.388 | - | 293.0 | 108.3 | 184.66 (upload_ms_untimed) | 0.388 (kernel/kernel) | - | 3227.2 | 1344.0 | optimizer_update_error_vs_torch_fp32=6.012e-05, rel_fro_vs_torch_eager_fp32=2.272e-07 | - | LIKE-FOR-LIKE-SPAN | - | recompiled-in-clock: 1 torch.compile compile(s) inside the timed run (measured this run) |
| torch-fused-fp32 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | 5.5 | 5.5..5.5 | 1 | 7.670 | - | 204.1 | 5.5 | 198.63 (upload_ms_untimed) | 7.670 (kernel/kernel) | - | 2715.7 | 896.0 | optimizer_update_error_vs_torch_fp32=6.574e-05, rel_fro_vs_torch_eager_fp32=2.484e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

calls:
- `ours`: `mojolearn.AdamW(amsgrad=False, betas=[0.9, 0.999], decoupled_weight_decay=True, eps=1e-08, lr=0.001, maximize=False, weight_decay=0.01, p0=float32[16777216], grad0=float32[16777216], grad1=float32[16777216], grad2=float32[16777216], ...)`
- `torch-eager-fp32`: `torch.AdamW(amsgrad=False, betas=[0.9, 0.999], capturable=False, decoupled_weight_decay=True, differentiable=False, eps=1e-08, foreach=True, lr=0.001, maximize=False, weight_decay=0.01, p0=float32[16777216], grad0=float32[16777216], ...)`
- `torch-compile-fp32`: `torch.AdamW(amsgrad=False, betas=[0.9, 0.999], capturable=False, decoupled_weight_decay=True, differentiable=False, eps=1e-08, foreach=True, lr=0.001, maximize=False, weight_decay=0.01, p0=float32[16777216], grad0=float32[16777216], ...)`
- `torch-fused-fp32`: `torch.AdamW(amsgrad=False, betas=[0.9, 0.999], capturable=False, decoupled_weight_decay=True, differentiable=False, eps=1e-08, fused=True, lr=0.001, maximize=False, weight_decay=0.01, p0=float32[16777216], grad0=float32[16777216], ...)`

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/147723/vram_* bytes for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-fused-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'betas': [0.9, 0.999], 'eps': 1e-08, 'lr': 0.001, 'maximize': False, 'weight_decay': 0.01}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-fp32 | torch-eager-fp32 | torch-fused-fp32 |
|---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (optimizer.defaults) | torch (optimizer.defaults) | torch (optimizer.defaults) |
| amsgrad | false | false | false | false |
| betas | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| capturable | - | false | false | false |
| decoupled_weight_decay | true | true | true | true |
| differentiable | - | false | false | false |
| eps | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| foreach | - | true | true | null |
| fused | - | null | null | true |
| input_arrays | [{"dtype": "float32", "name": "p0", "sha256": "485a7287793a9d7d33d710ea2820ea9c7519d31100db4bafeee813baa532dd7d", "shape": [16777216]}, {"dtype": "float32", "name": "grad0", "sha256": "e6c0159d56a2df748f38d3f3592321edc11067354f52e751e382384dca4178bc", "shape": [16777216]}, {"dtype": "float32", "name": "grad1", "sha256": "249bb61249ee25b0cd42e6bcaa012c27c565dde4ccd798448a60e935fd4ace7e", "shape": [16777216]}, {"dtype": "float32", "name": "grad2", "sha256": "4cf167fba649d3d82fdb6c7c41749a74e8fc8a9bd70491154f03bfc9272b2ed6", "shape": [16777216]}, {"dtype": "float32", "name": "grad3", "sha256": "594f963cfcef38863774b145fac84f56ae7f37190af1cf270e7cb23d3b86b587", "shape": [16777216]}, {"dtype": "float32", "name": "grad4", "sha256": "ee091b2ac84630d1d0328c337b12c5bb6bd6ba2b28f3350eab106bc848960f4e", "shape": [16777216]}, {"dtype": "float32", "name": "grad5", "sha256": "a042832a1a0884016cafe5aada2f1df6bf0b09422e9358536881a536fff70d02", "shape": [16777216]}, {"dtype": "float32", "name": "grad6", "sha256": "736aade529a9b43a7666f9d5a77bf9ef70c253bb5a26280143d941c915910763", "shape": [16777216]}, {"dtype": "float32", "name": "grad7", "sha256": "c4fdaf0915399dc3568c7fd4a49465f9e3c69a10d3e77b1fc15d926db244f302", "shape": [16777216]}, {"dtype": "float32", "name": "grad8", "sha256": "6b970f1bc919f22be37581526578d65a535e33dd886ff46cc64d95aca61fec12", "shape": [16777216]}, {"dtype": "float32", "name": "grad9", "sha256": "00adaa7225e175a85e28e59772e3afbb901e068e5ca8d7332adfa127f7fcbdd7", "shape": [16777216]}] | [{"dtype": "float32", "name": "p0", "sha256": "485a7287793a9d7d33d710ea2820ea9c7519d31100db4bafeee813baa532dd7d", "shape": [16777216]}, {"dtype": "float32", "name": "grad0", "sha256": "e6c0159d56a2df748f38d3f3592321edc11067354f52e751e382384dca4178bc", "shape": [16777216]}, {"dtype": "float32", "name": "grad1", "sha256": "249bb61249ee25b0cd42e6bcaa012c27c565dde4ccd798448a60e935fd4ace7e", "shape": [16777216]}, {"dtype": "float32", "name": "grad2", "sha256": "4cf167fba649d3d82fdb6c7c41749a74e8fc8a9bd70491154f03bfc9272b2ed6", "shape": [16777216]}, {"dtype": "float32", "name": "grad3", "sha256": "594f963cfcef38863774b145fac84f56ae7f37190af1cf270e7cb23d3b86b587", "shape": [16777216]}, {"dtype": "float32", "name": "grad4", "sha256": "ee091b2ac84630d1d0328c337b12c5bb6bd6ba2b28f3350eab106bc848960f4e", "shape": [16777216]}, {"dtype": "float32", "name": "grad5", "sha256": "a042832a1a0884016cafe5aada2f1df6bf0b09422e9358536881a536fff70d02", "shape": [16777216]}, {"dtype": "float32", "name": "grad6", "sha256": "736aade529a9b43a7666f9d5a77bf9ef70c253bb5a26280143d941c915910763", "shape": [16777216]}, {"dtype": "float32", "name": "grad7", "sha256": "c4fdaf0915399dc3568c7fd4a49465f9e3c69a10d3e77b1fc15d926db244f302", "shape": [16777216]}, {"dtype": "float32", "name": "grad8", "sha256": "6b970f1bc919f22be37581526578d65a535e33dd886ff46cc64d95aca61fec12", "shape": [16777216]}, {"dtype": "float32", "name": "grad9", "sha256": "00adaa7225e175a85e28e59772e3afbb901e068e5ca8d7332adfa127f7fcbdd7", "shape": [16777216]}] | [{"dtype": "float32", "name": "p0", "sha256": "485a7287793a9d7d33d710ea2820ea9c7519d31100db4bafeee813baa532dd7d", "shape": [16777216]}, {"dtype": "float32", "name": "grad0", "sha256": "e6c0159d56a2df748f38d3f3592321edc11067354f52e751e382384dca4178bc", "shape": [16777216]}, {"dtype": "float32", "name": "grad1", "sha256": "249bb61249ee25b0cd42e6bcaa012c27c565dde4ccd798448a60e935fd4ace7e", "shape": [16777216]}, {"dtype": "float32", "name": "grad2", "sha256": "4cf167fba649d3d82fdb6c7c41749a74e8fc8a9bd70491154f03bfc9272b2ed6", "shape": [16777216]}, {"dtype": "float32", "name": "grad3", "sha256": "594f963cfcef38863774b145fac84f56ae7f37190af1cf270e7cb23d3b86b587", "shape": [16777216]}, {"dtype": "float32", "name": "grad4", "sha256": "ee091b2ac84630d1d0328c337b12c5bb6bd6ba2b28f3350eab106bc848960f4e", "shape": [16777216]}, {"dtype": "float32", "name": "grad5", "sha256": "a042832a1a0884016cafe5aada2f1df6bf0b09422e9358536881a536fff70d02", "shape": [16777216]}, {"dtype": "float32", "name": "grad6", "sha256": "736aade529a9b43a7666f9d5a77bf9ef70c253bb5a26280143d941c915910763", "shape": [16777216]}, {"dtype": "float32", "name": "grad7", "sha256": "c4fdaf0915399dc3568c7fd4a49465f9e3c69a10d3e77b1fc15d926db244f302", "shape": [16777216]}, {"dtype": "float32", "name": "grad8", "sha256": "6b970f1bc919f22be37581526578d65a535e33dd886ff46cc64d95aca61fec12", "shape": [16777216]}, {"dtype": "float32", "name": "grad9", "sha256": "00adaa7225e175a85e28e59772e3afbb901e068e5ca8d7332adfa127f7fcbdd7", "shape": [16777216]}] | [{"dtype": "float32", "name": "p0", "sha256": "485a7287793a9d7d33d710ea2820ea9c7519d31100db4bafeee813baa532dd7d", "shape": [16777216]}, {"dtype": "float32", "name": "grad0", "sha256": "e6c0159d56a2df748f38d3f3592321edc11067354f52e751e382384dca4178bc", "shape": [16777216]}, {"dtype": "float32", "name": "grad1", "sha256": "249bb61249ee25b0cd42e6bcaa012c27c565dde4ccd798448a60e935fd4ace7e", "shape": [16777216]}, {"dtype": "float32", "name": "grad2", "sha256": "4cf167fba649d3d82fdb6c7c41749a74e8fc8a9bd70491154f03bfc9272b2ed6", "shape": [16777216]}, {"dtype": "float32", "name": "grad3", "sha256": "594f963cfcef38863774b145fac84f56ae7f37190af1cf270e7cb23d3b86b587", "shape": [16777216]}, {"dtype": "float32", "name": "grad4", "sha256": "ee091b2ac84630d1d0328c337b12c5bb6bd6ba2b28f3350eab106bc848960f4e", "shape": [16777216]}, {"dtype": "float32", "name": "grad5", "sha256": "a042832a1a0884016cafe5aada2f1df6bf0b09422e9358536881a536fff70d02", "shape": [16777216]}, {"dtype": "float32", "name": "grad6", "sha256": "736aade529a9b43a7666f9d5a77bf9ef70c253bb5a26280143d941c915910763", "shape": [16777216]}, {"dtype": "float32", "name": "grad7", "sha256": "c4fdaf0915399dc3568c7fd4a49465f9e3c69a10d3e77b1fc15d926db244f302", "shape": [16777216]}, {"dtype": "float32", "name": "grad8", "sha256": "6b970f1bc919f22be37581526578d65a535e33dd886ff46cc64d95aca61fec12", "shape": [16777216]}, {"dtype": "float32", "name": "grad9", "sha256": "00adaa7225e175a85e28e59772e3afbb901e068e5ca8d7332adfa127f7fcbdd7", "shape": [16777216]}] |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 |
| maximize | false | false | false | false |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| weight_decay | 0.01 | 0.01 | 0.01 | 0.01 |

accepted difference: torch-eager-fp32 capturable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-eager-fp32 differentiable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-eager-fp32 foreach: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

accepted difference: torch-eager-fp32 fused: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

accepted difference: torch-compile-fp32 capturable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-compile-fp32 differentiable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-compile-fp32 foreach: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

accepted difference: torch-compile-fp32 fused: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

accepted difference: torch-fused-fp32 capturable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-fused-fp32 differentiable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-fused-fp32 foreach: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

accepted difference: torch-fused-fp32 fused: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

### avgpool1d / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/measure-0852/ingest0849/inputs/amd-algos.avgpool1d.synthetic.rows=full.json (amd2/paired-ed7283bc5cfd8d41005c848ae84889f8a84799e699900a6dbf99b888a63b42a3)`, ran on enc1-gpuvm005

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 1.0 ms) = 1.627; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 0.7 ms) = 2.210. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI300X (amd2, Hot Aisle) | 0.8.49 | identical | 1.6 | 1.6..1.6 | 1 | - | - | - | 1.6 | - (stored kernel) | - | - | 2396.9 | 1877.7 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000, relative_error_vs_own_host=0.000000, scorer=scored | - | main board, one scored run | wheel | ok (main@80ace6b7a amd2/paired-ed7283bc5cfd8d41005c848ae84889f8a84799e699900a6dbf99b888a63b42a3 2026-10-16; identity vs the nvidia columns: n/a) |
| torch-eager-fp32 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | 0.7 | 0.7..0.7 | 1 | 2.210 | - | 173.6 | 0.7 | 172.89 (upload_ms_untimed) | 2.210 (kernel/kernel) | - | 1900.6 | 224.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | 2.2 | 2.2..2.2 | 1 | 0.736 | - | 174.1 | 2.2 | 171.88 (upload_ms_untimed) | 0.736 (kernel/kernel) | - | 2236.2 | 224.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | 1.0 | 1.0..1.0 | 1 | 1.627 | - | 174.4 | 1.0 | 173.34 (upload_ms_untimed) | 1.627 (kernel/kernel) | - | 1895.7 | 224.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | 1.5 | 1.5..1.5 | 1 | 1.072 | - | 173.4 | 1.5 | 171.87 (upload_ms_untimed) | 1.072 (kernel/kernel) | - | 2153.3 | 224.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

calls:
- `ours`: `mojolearn.AvgPool1d(ceil_mode=False, count_include_pad=True, kernel_size=2, padding=0, stride=2, x=float32[64x64x4096])`
- `torch-eager-fp32`: `torch.AvgPool1d(ceil_mode=False, count_include_pad=True, kernel_size=2, padding=0, seed=7, stride=2, x=float32[64x64x4096])`
- `torch-compile-fp32`: `torch.AvgPool1d(ceil_mode=False, count_include_pad=True, kernel_size=2, padding=0, seed=7, stride=2, x=float32[64x64x4096])`
- `torch-eager-bf16`: `torch.AvgPool1d(ceil_mode=False, count_include_pad=True, kernel_size=2, padding=0, seed=7, stride=2, x=float32[64x64x4096])`
- `torch-compile-bf16`: `torch.AvgPool1d(ceil_mode=False, count_include_pad=True, kernel_size=2, padding=0, seed=7, stride=2, x=float32[64x64x4096])`

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/148893/vram_* bytes for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'ceil_mode': False, 'count_include_pad': True, 'kernel_size': 2, 'padding': 0, 'stride': 2}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| ceil_mode | false | false | false | false | false |
| count_include_pad | true | true | true | true | true |
| input_arrays | [{"dtype": "float32", "name": "x", "sha256": "128bb6c8835123e5f00ffe3700a26223ba8b37f7280afaec092571b2861be93c", "shape": [64, 64, 4096]}] | [{"dtype": "float32", "name": "x", "sha256": "128bb6c8835123e5f00ffe3700a26223ba8b37f7280afaec092571b2861be93c", "shape": [64, 64, 4096]}] | [{"dtype": "float32", "name": "x", "sha256": "128bb6c8835123e5f00ffe3700a26223ba8b37f7280afaec092571b2861be93c", "shape": [64, 64, 4096]}] | [{"dtype": "float32", "name": "x", "sha256": "128bb6c8835123e5f00ffe3700a26223ba8b37f7280afaec092571b2861be93c", "shape": [64, 64, 4096]}] | [{"dtype": "float32", "name": "x", "sha256": "128bb6c8835123e5f00ffe3700a26223ba8b37f7280afaec092571b2861be93c", "shape": [64, 64, 4096]}] |
| kernel_size | 2 | 2 | 2 | 2 | 2 |
| padding | 0 | 0 | 0 | 0 | 0 |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 |
| stride | 2 | 2 | 2 | 2 | 2 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | hardware | version | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | AMD Instinct MI300X (amd2, Hot Aisle) | 0.8.49 | Xq | - | 0.2 | 0.2..0.2 | 1 | - | - | - | 0.2 | - (stored kernel) | - | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000, relative_error_vs_own_host=0.000000, scorer=scored | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | Xq | - | 0.1 | 0.1..0.1 | 1 | 2.397 | - | 173.0 | 0.1 | 172.89 (predict_upload_ms (the predict inputs' own copy)) | 2.397 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | Xq | - | 0.6 | 0.6..0.6 | 1 | 0.410 | - | 172.4 | 0.6 | 171.88 (predict_upload_ms (the predict inputs' own copy)) | 0.410 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | Xq | - | 0.1 | 0.1..0.1 | 1 | 2.064 | - | 173.5 | 0.1 | 173.34 (predict_upload_ms (the predict inputs' own copy)) | 2.064 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | Xq | - | 0.2 | 0.2..0.2 | 1 | 1.076 | - | 172.1 | 0.2 | 171.87 (predict_upload_ms (the predict inputs' own copy)) | 1.076 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: mojolearn.AvgPool1d(ceil_mode=False, count_include_pad=True, kernel_size=2, padding=0, stride=2, x=float32[64x64x4096]); forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: torch.AvgPool1d(ceil_mode=False, count_include_pad=True, kernel_size=2, padding=0, seed=7, stride=2, x=float32[64x64x4096]); forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: torch.AvgPool1d(ceil_mode=False, count_include_pad=True, kernel_size=2, padding=0, seed=7, stride=2, x=float32[64x64x4096]); forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: torch.AvgPool1d(ceil_mode=False, count_include_pad=True, kernel_size=2, padding=0, seed=7, stride=2, x=float32[64x64x4096]); forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: torch.AvgPool1d(ceil_mode=False, count_include_pad=True, kernel_size=2, padding=0, seed=7, stride=2, x=float32[64x64x4096]); forward(x) (no autograd)(Xq)

### avgpool2d / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/measure-0852/ingest0849/inputs/amd-algos.avgpool2d.synthetic.rows=full.json (amd2/paired-06cacc300f3f1c824ac07fc6e56c116d10a8822be31d1d51b22fe736e08a8481)`, ran on enc1-gpuvm005

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 2.3 ms) = 3.549; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 1.3 ms) = 6.257. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI300X (amd2, Hot Aisle) | 0.8.49 | identical | 8.2 | 8.2..8.2 | 1 | - | - | - | 8.2 | - (stored kernel) | - | - | 2596.4 | 1877.7 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000, relative_error_vs_own_host=0.000000, scorer=scored | - | main board, one scored run | wheel | ok (main@80ace6b7a amd2/paired-06cacc300f3f1c824ac07fc6e56c116d10a8822be31d1d51b22fe736e08a8481 2026-10-16; identity vs the nvidia columns: n/a) |
| torch-eager-fp32 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | 1.3 | 1.3..1.3 | 1 | 6.257 | - | 157.7 | 1.3 | 156.38 (upload_ms_untimed) | 6.257 (kernel/kernel) | - | 2238.9 | 541.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | 1.4 | 1.4..1.4 | 1 | 5.744 | - | 168.4 | 1.4 | 167.02 (upload_ms_untimed) | 5.744 (kernel/kernel) | - | 2556.9 | 541.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | 7.9 | 7.9..7.9 | 1 | 1.030 | - | 168.7 | 7.9 | 160.77 (upload_ms_untimed) | 1.030 (kernel/kernel) | - | 2239.2 | 541.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | 2.3 | 2.3..2.3 | 1 | 3.549 | - | 166.8 | 2.3 | 164.47 (upload_ms_untimed) | 3.549 (kernel/kernel) | - | 2471.8 | 541.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

calls:
- `ours`: `mojolearn.AvgPool2d(ceil_mode=False, count_include_pad=True, kernel_size=2, padding=0, stride=2, x=float32[64x64x112x112])`
- `torch-eager-fp32`: `torch.AvgPool2d(ceil_mode=False, count_include_pad=True, kernel_size=2, padding=0, seed=7, stride=2, x=float32[64x64x112x112])`
- `torch-compile-fp32`: `torch.AvgPool2d(ceil_mode=False, count_include_pad=True, kernel_size=2, padding=0, seed=7, stride=2, x=float32[64x64x112x112])`
- `torch-eager-bf16`: `torch.AvgPool2d(ceil_mode=False, count_include_pad=True, kernel_size=2, padding=0, seed=7, stride=2, x=float32[64x64x112x112])`
- `torch-compile-bf16`: `torch.AvgPool2d(ceil_mode=False, count_include_pad=True, kernel_size=2, padding=0, seed=7, stride=2, x=float32[64x64x112x112])`

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/150103/vram_* bytes for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'ceil_mode': False, 'count_include_pad': True, 'divisor_override': None, 'kernel_size': 2, 'padding': 0, 'stride': 2}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| ceil_mode | false | false | false | false | false |
| count_include_pad | true | true | true | true | true |
| divisor_override | null | null | null | null | null |
| input_arrays | [{"dtype": "float32", "name": "x", "sha256": "2df96f3879e0a6060dab5f7b7561b7291c5a106d1a8d4e116fc5e44e3023f819", "shape": [64, 64, 112, 112]}] | [{"dtype": "float32", "name": "x", "sha256": "2df96f3879e0a6060dab5f7b7561b7291c5a106d1a8d4e116fc5e44e3023f819", "shape": [64, 64, 112, 112]}] | [{"dtype": "float32", "name": "x", "sha256": "2df96f3879e0a6060dab5f7b7561b7291c5a106d1a8d4e116fc5e44e3023f819", "shape": [64, 64, 112, 112]}] | [{"dtype": "float32", "name": "x", "sha256": "2df96f3879e0a6060dab5f7b7561b7291c5a106d1a8d4e116fc5e44e3023f819", "shape": [64, 64, 112, 112]}] | [{"dtype": "float32", "name": "x", "sha256": "2df96f3879e0a6060dab5f7b7561b7291c5a106d1a8d4e116fc5e44e3023f819", "shape": [64, 64, 112, 112]}] |
| kernel_size | 2 | 2 | 2 | 2 | 2 |
| padding | 0 | 0 | 0 | 0 | 0 |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 |
| stride | 2 | 2 | 2 | 2 | 2 |

accepted difference: torch-eager-fp32 divisor_override: None selects the kernel-area divisor on both AvgPool2d implementations; all active arguments remain compared

accepted difference: torch-compile-fp32 divisor_override: None selects the kernel-area divisor on both AvgPool2d implementations; all active arguments remain compared

accepted difference: torch-eager-bf16 divisor_override: None selects the kernel-area divisor on both AvgPool2d implementations; all active arguments remain compared

accepted difference: torch-compile-bf16 divisor_override: None selects the kernel-area divisor on both AvgPool2d implementations; all active arguments remain compared

Inference (each arm predicts with its own model from the fit rounds above):

| arm | hardware | version | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | AMD Instinct MI300X (amd2, Hot Aisle) | 0.8.49 | Xq | - | 0.2 | 0.2..0.2 | 1 | - | - | - | 0.2 | - (stored kernel) | - | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000, relative_error_vs_own_host=0.000000, scorer=scored | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | Xq | - | 0.2 | 0.2..0.2 | 1 | 0.950 | - | 156.6 | 0.2 | 156.38 (predict_upload_ms (the predict inputs' own copy)) | 0.950 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | Xq | - | 0.3 | 0.3..0.3 | 1 | 0.813 | - | 167.3 | 0.3 | 167.02 (predict_upload_ms (the predict inputs' own copy)) | 0.813 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | Xq | - | 0.2 | 0.2..0.2 | 1 | 0.912 | - | 161.0 | 0.2 | 160.77 (predict_upload_ms (the predict inputs' own copy)) | 0.912 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | Xq | - | 0.6 | 0.6..0.6 | 1 | 0.389 | - | 165.0 | 0.6 | 164.47 (predict_upload_ms (the predict inputs' own copy)) | 0.389 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: mojolearn.AvgPool2d(ceil_mode=False, count_include_pad=True, kernel_size=2, padding=0, stride=2, x=float32[64x64x112x112]); forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: torch.AvgPool2d(ceil_mode=False, count_include_pad=True, kernel_size=2, padding=0, seed=7, stride=2, x=float32[64x64x112x112]); forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: torch.AvgPool2d(ceil_mode=False, count_include_pad=True, kernel_size=2, padding=0, seed=7, stride=2, x=float32[64x64x112x112]); forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: torch.AvgPool2d(ceil_mode=False, count_include_pad=True, kernel_size=2, padding=0, seed=7, stride=2, x=float32[64x64x112x112]); forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: torch.AvgPool2d(ceil_mode=False, count_include_pad=True, kernel_size=2, padding=0, seed=7, stride=2, x=float32[64x64x112x112]); forward(x) (no autograd)(Xq)

### batchnorm1d / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/measure-0852/ingest0849/inputs/amd-algos.batchnorm1d.synthetic.rows=full.json (amd2/paired-0304aaf18ea8ae47a0c2a3a845423a0735ecd1b6889e39572e2cd2504b638eec)`, ran on enc1-gpuvm005

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 1.3 ms) = 1.554; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 1.4 ms) = 1.490. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI300X (amd2, Hot Aisle) | 0.8.49 | identical | 2.0 | 2.0..2.0 | 1 | - | - | - | 2.0 | - (stored kernel) | - | - | 2524.5 | 1877.8 | max_rel_diff_vs_torch_eager_fp32=0.004642, rel_fro_vs_torch_eager_fp32=8.097e-08, relative_error_vs_own_host=0.000000, scorer=scored | - | main board, one scored run | wheel | ok (main@80ace6b7a amd2/paired-0304aaf18ea8ae47a0c2a3a845423a0735ecd1b6889e39572e2cd2504b638eec 2026-10-16; identity vs the nvidia columns: n/a) |
| torch-eager-fp32 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | 1.4 | 1.4..1.4 | 1 | 1.490 | - | 18.6 | 1.4 | 17.22 (upload_ms_untimed) | 1.490 (kernel/kernel) | - | 2630.9 | 320.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | 94.0 | 94.0..94.0 | 1 | 0.021 | - | 112.5 | 94.0 | 18.53 (upload_ms_untimed) | 0.021 (kernel/kernel) | - | 2367.1 | 320.0 | max_rel_diff_vs_torch_eager_fp32=0.002793, rel_fro_vs_torch_eager_fp32=5.248e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | 1.3 | 1.3..1.3 | 1 | 1.554 | - | 21.1 | 1.3 | 19.78 (upload_ms_untimed) | 1.554 (kernel/kernel) | - | 2261.0 | 320.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | 2.8 | 2.8..2.8 | 1 | 0.710 | - | 19.7 | 2.8 | 16.83 (upload_ms_untimed) | 0.710 (kernel/kernel) | - | 2260.5 | 320.0 | max_rel_diff_vs_torch_eager_fp32=0.002793, rel_fro_vs_torch_eager_fp32=5.248e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

calls:
- `ours`: `mojolearn.BatchNorm1d(affine=True, eps=1e-05, momentum=0.1, num_features=256, track_running_stats=True, x=float32[64x256x1024], bias=float32[256], num_batches_tracked=int64[1], running_mean=float32[256], running_var=float32[256], ...)`
- `torch-eager-fp32`: `torch.BatchNorm1d(affine=True, eps=1e-05, momentum=0.1, num_features=256, seed=7, track_running_stats=True, x=float32[64x256x1024], bias=float32[256], num_batches_tracked=int64[1], running_mean=float32[256], running_var=float32[256], ...)`
- `torch-compile-fp32`: `torch.BatchNorm1d(affine=True, eps=1e-05, momentum=0.1, num_features=256, seed=7, track_running_stats=True, x=float32[64x256x1024], bias=float32[256], num_batches_tracked=int64[1], running_mean=float32[256], running_var=float32[256], ...)`
- `torch-eager-bf16`: `torch.BatchNorm1d(affine=True, eps=1e-05, momentum=0.1, num_features=256, seed=7, track_running_stats=True, x=float32[64x256x1024], bias=float32[256], num_batches_tracked=int64[1], running_mean=float32[256], running_var=float32[256], ...)`
- `torch-compile-bf16`: `torch.BatchNorm1d(affine=True, eps=1e-05, momentum=0.1, num_features=256, seed=7, track_running_stats=True, x=float32[64x256x1024], bias=float32[256], num_batches_tracked=int64[1], running_mean=float32[256], running_var=float32[256], ...)`

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/151331/vram_* bytes for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'affine': True, 'eps': 1e-05, 'momentum': 0.1, 'num_features': 256, 'track_running_stats': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| affine | true | true | true | true | true |
| eps | 1e-05 | 1e-05 | 1e-05 | 1e-05 | 1e-05 |
| input_arrays | [{"dtype": "float32", "name": "x", "sha256": "128bb6c8835123e5f00ffe3700a26223ba8b37f7280afaec092571b2861be93c", "shape": [64, 256, 1024]}, {"dtype": "float32", "name": "bias", "sha256": "5f70bf18a086007016e948b04aed3b82103a36bea41755b6cddfaf10ace3c6ef", "shape": [256]}, {"dtype": "int64", "name": "num_batches_tracked", "sha256": "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc", "shape": [1]}, {"dtype": "float32", "name": "running_mean", "sha256": "5f70bf18a086007016e948b04aed3b82103a36bea41755b6cddfaf10ace3c6ef", "shape": [256]}, {"dtype": "float32", "name": "running_var", "sha256": "893a106828fbdb9521e1d868c985aab7ad2ae2f606edc55329265a5e7676006c", "shape": [256]}, {"dtype": "float32", "name": "weight", "sha256": "893a106828fbdb9521e1d868c985aab7ad2ae2f606edc55329265a5e7676006c", "shape": [256]}] | [{"dtype": "float32", "name": "x", "sha256": "128bb6c8835123e5f00ffe3700a26223ba8b37f7280afaec092571b2861be93c", "shape": [64, 256, 1024]}, {"dtype": "float32", "name": "bias", "sha256": "5f70bf18a086007016e948b04aed3b82103a36bea41755b6cddfaf10ace3c6ef", "shape": [256]}, {"dtype": "int64", "name": "num_batches_tracked", "sha256": "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc", "shape": [1]}, {"dtype": "float32", "name": "running_mean", "sha256": "5f70bf18a086007016e948b04aed3b82103a36bea41755b6cddfaf10ace3c6ef", "shape": [256]}, {"dtype": "float32", "name": "running_var", "sha256": "893a106828fbdb9521e1d868c985aab7ad2ae2f606edc55329265a5e7676006c", "shape": [256]}, {"dtype": "float32", "name": "weight", "sha256": "893a106828fbdb9521e1d868c985aab7ad2ae2f606edc55329265a5e7676006c", "shape": [256]}] | [{"dtype": "float32", "name": "x", "sha256": "128bb6c8835123e5f00ffe3700a26223ba8b37f7280afaec092571b2861be93c", "shape": [64, 256, 1024]}, {"dtype": "float32", "name": "bias", "sha256": "5f70bf18a086007016e948b04aed3b82103a36bea41755b6cddfaf10ace3c6ef", "shape": [256]}, {"dtype": "int64", "name": "num_batches_tracked", "sha256": "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc", "shape": [1]}, {"dtype": "float32", "name": "running_mean", "sha256": "5f70bf18a086007016e948b04aed3b82103a36bea41755b6cddfaf10ace3c6ef", "shape": [256]}, {"dtype": "float32", "name": "running_var", "sha256": "893a106828fbdb9521e1d868c985aab7ad2ae2f606edc55329265a5e7676006c", "shape": [256]}, {"dtype": "float32", "name": "weight", "sha256": "893a106828fbdb9521e1d868c985aab7ad2ae2f606edc55329265a5e7676006c", "shape": [256]}] | [{"dtype": "float32", "name": "x", "sha256": "128bb6c8835123e5f00ffe3700a26223ba8b37f7280afaec092571b2861be93c", "shape": [64, 256, 1024]}, {"dtype": "float32", "name": "bias", "sha256": "5f70bf18a086007016e948b04aed3b82103a36bea41755b6cddfaf10ace3c6ef", "shape": [256]}, {"dtype": "int64", "name": "num_batches_tracked", "sha256": "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc", "shape": [1]}, {"dtype": "float32", "name": "running_mean", "sha256": "5f70bf18a086007016e948b04aed3b82103a36bea41755b6cddfaf10ace3c6ef", "shape": [256]}, {"dtype": "float32", "name": "running_var", "sha256": "893a106828fbdb9521e1d868c985aab7ad2ae2f606edc55329265a5e7676006c", "shape": [256]}, {"dtype": "float32", "name": "weight", "sha256": "893a106828fbdb9521e1d868c985aab7ad2ae2f606edc55329265a5e7676006c", "shape": [256]}] | [{"dtype": "float32", "name": "x", "sha256": "128bb6c8835123e5f00ffe3700a26223ba8b37f7280afaec092571b2861be93c", "shape": [64, 256, 1024]}, {"dtype": "float32", "name": "bias", "sha256": "5f70bf18a086007016e948b04aed3b82103a36bea41755b6cddfaf10ace3c6ef", "shape": [256]}, {"dtype": "int64", "name": "num_batches_tracked", "sha256": "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc", "shape": [1]}, {"dtype": "float32", "name": "running_mean", "sha256": "5f70bf18a086007016e948b04aed3b82103a36bea41755b6cddfaf10ace3c6ef", "shape": [256]}, {"dtype": "float32", "name": "running_var", "sha256": "893a106828fbdb9521e1d868c985aab7ad2ae2f606edc55329265a5e7676006c", "shape": [256]}, {"dtype": "float32", "name": "weight", "sha256": "893a106828fbdb9521e1d868c985aab7ad2ae2f606edc55329265a5e7676006c", "shape": [256]}] |
| momentum | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| num_features | 256 | 256 | 256 | 256 | 256 |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 |
| track_running_stats | true | true | true | true | true |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | hardware | version | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | AMD Instinct MI300X (amd2, Hot Aisle) | 0.8.49 | Xq | - | 0.4 | 0.4..0.4 | 1 | - | - | - | 0.4 | - (stored kernel) | - | - | max_rel_diff_vs_torch_eager_fp32=0.004642, rel_fro_vs_torch_eager_fp32=8.097e-08, relative_error_vs_own_host=0.000000, scorer=scored | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | Xq | - | 0.2 | 0.2..0.2 | 1 | 2.360 | - | 17.4 | 0.2 | 17.22 (predict_upload_ms (the predict inputs' own copy)) | 2.360 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | Xq | - | 1.3 | 1.3..1.3 | 1 | 0.281 | - | 19.9 | 1.3 | 18.53 (predict_upload_ms (the predict inputs' own copy)) | 0.281 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.002793, rel_fro_vs_torch_eager_fp32=5.248e-08 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | Xq | - | 0.2 | 0.2..0.2 | 1 | 1.925 | - | 20.0 | 0.2 | 19.78 (predict_upload_ms (the predict inputs' own copy)) | 1.925 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | Xq | - | 0.7 | 0.7..0.7 | 1 | 0.508 | - | 17.6 | 0.7 | 16.83 (predict_upload_ms (the predict inputs' own copy)) | 0.508 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.002793, rel_fro_vs_torch_eager_fp32=5.248e-08 | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: mojolearn.BatchNorm1d(affine=True, eps=1e-05, momentum=0.1, num_features=256, track_running_stats=True, x=float32[64x256x1024], bias=float32[256], num_batches_tracked=int64[1], running_mean=float32[256], running_var=float32[256], ...); forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: torch.BatchNorm1d(affine=True, eps=1e-05, momentum=0.1, num_features=256, seed=7, track_running_stats=True, x=float32[64x256x1024], bias=float32[256], num_batches_tracked=int64[1], running_mean=float32[256], running_var=float32[256], ...); forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: torch.BatchNorm1d(affine=True, eps=1e-05, momentum=0.1, num_features=256, seed=7, track_running_stats=True, x=float32[64x256x1024], bias=float32[256], num_batches_tracked=int64[1], running_mean=float32[256], running_var=float32[256], ...); forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: torch.BatchNorm1d(affine=True, eps=1e-05, momentum=0.1, num_features=256, seed=7, track_running_stats=True, x=float32[64x256x1024], bias=float32[256], num_batches_tracked=int64[1], running_mean=float32[256], running_var=float32[256], ...); forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: torch.BatchNorm1d(affine=True, eps=1e-05, momentum=0.1, num_features=256, seed=7, track_running_stats=True, x=float32[64x256x1024], bias=float32[256], num_batches_tracked=int64[1], running_mean=float32[256], running_var=float32[256], ...); forward(x) (no autograd)(Xq)

### batchnorm2d / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/measure-0852/ingest0849/inputs/amd-algos.batchnorm2d.synthetic.rows=full.json (amd2/paired-c62f401e7edc2ada51051dbb2669d15bb1688f966c207ff502eca617a4ada33f)`, ran on enc1-gpuvm005

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 1.9 ms) = 0.657; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 8.4 ms) = 0.146. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI300X (amd2, Hot Aisle) | 0.8.49 | identical | 1.2 | 1.2..1.2 | 1 | - | - | - | 1.2 | - (stored kernel) | - | - | 2450.3 | 1877.8 | max_rel_diff_vs_torch_eager_fp32=0.001395, rel_fro_vs_torch_eager_fp32=1.249e-07, relative_error_vs_own_host=0.000000, scorer=scored | - | main board, one scored run | wheel | ok (main@80ace6b7a amd2/paired-c62f401e7edc2ada51051dbb2669d15bb1688f966c207ff502eca617a4ada33f 2026-10-16; identity vs the nvidia columns: n/a) |
| torch-eager-fp32 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | 8.4 | 8.4..8.4 | 1 | 0.146 | - | 32.1 | 8.4 | 23.63 (upload_ms_untimed) | 0.146 (kernel/kernel) | - | 2328.7 | 250.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | 12.0 | 12.0..12.0 | 1 | 0.103 | - | 32.4 | 12.0 | 20.42 (upload_ms_untimed) | 0.103 (kernel/kernel) | - | 2288.3 | 250.0 | max_rel_diff_vs_torch_eager_fp32=0.001471, rel_fro_vs_torch_eager_fp32=6.868e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | 37.1 | 37.1..37.1 | 1 | 0.033 | - | 57.7 | 37.1 | 20.67 (upload_ms_untimed) | 0.033 (kernel/kernel) | - | 1958.7 | 250.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | 1.9 | 1.9..1.9 | 1 | 0.657 | - | 23.3 | 1.9 | 21.38 (upload_ms_untimed) | 0.657 (kernel/kernel) | - | 2181.3 | 250.0 | max_rel_diff_vs_torch_eager_fp32=0.001471, rel_fro_vs_torch_eager_fp32=6.868e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

calls:
- `ours`: `mojolearn.BatchNorm2d(affine=True, eps=1e-05, momentum=0.1, num_features=64, track_running_stats=True, x=float32[64x64x56x56], bias=float32[64], num_batches_tracked=int64[1], running_mean=float32[64], running_var=float32[64], ...)`
- `torch-eager-fp32`: `torch.BatchNorm2d(affine=True, eps=1e-05, momentum=0.1, num_features=64, seed=7, track_running_stats=True, x=float32[64x64x56x56], bias=float32[64], num_batches_tracked=int64[1], running_mean=float32[64], running_var=float32[64], ...)`
- `torch-compile-fp32`: `torch.BatchNorm2d(affine=True, eps=1e-05, momentum=0.1, num_features=64, seed=7, track_running_stats=True, x=float32[64x64x56x56], bias=float32[64], num_batches_tracked=int64[1], running_mean=float32[64], running_var=float32[64], ...)`
- `torch-eager-bf16`: `torch.BatchNorm2d(affine=True, eps=1e-05, momentum=0.1, num_features=64, seed=7, track_running_stats=True, x=float32[64x64x56x56], bias=float32[64], num_batches_tracked=int64[1], running_mean=float32[64], running_var=float32[64], ...)`
- `torch-compile-bf16`: `torch.BatchNorm2d(affine=True, eps=1e-05, momentum=0.1, num_features=64, seed=7, track_running_stats=True, x=float32[64x64x56x56], bias=float32[64], num_batches_tracked=int64[1], running_mean=float32[64], running_var=float32[64], ...)`

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/152579/vram_* bytes for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'affine': True, 'eps': 1e-05, 'momentum': 0.1, 'num_features': 64, 'track_running_stats': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| affine | true | true | true | true | true |
| eps | 1e-05 | 1e-05 | 1e-05 | 1e-05 | 1e-05 |
| input_arrays | [{"dtype": "float32", "name": "x", "sha256": "cd7e0a7d8c0c594cca519b9cf49c8f086513282644cb8b6c06771dcb98504f2d", "shape": [64, 64, 56, 56]}, {"dtype": "float32", "name": "bias", "sha256": "5341e6b2646979a70e57653007a1f310169421ec9bdd9f1a5648f75ade005af1", "shape": [64]}, {"dtype": "int64", "name": "num_batches_tracked", "sha256": "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc", "shape": [1]}, {"dtype": "float32", "name": "running_mean", "sha256": "5341e6b2646979a70e57653007a1f310169421ec9bdd9f1a5648f75ade005af1", "shape": [64]}, {"dtype": "float32", "name": "running_var", "sha256": "2f20cd03c9cd392a406c56232b0ff93a15f6d6d7da79086bfa14f55d4a4031b0", "shape": [64]}, {"dtype": "float32", "name": "weight", "sha256": "2f20cd03c9cd392a406c56232b0ff93a15f6d6d7da79086bfa14f55d4a4031b0", "shape": [64]}] | [{"dtype": "float32", "name": "x", "sha256": "cd7e0a7d8c0c594cca519b9cf49c8f086513282644cb8b6c06771dcb98504f2d", "shape": [64, 64, 56, 56]}, {"dtype": "float32", "name": "bias", "sha256": "5341e6b2646979a70e57653007a1f310169421ec9bdd9f1a5648f75ade005af1", "shape": [64]}, {"dtype": "int64", "name": "num_batches_tracked", "sha256": "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc", "shape": [1]}, {"dtype": "float32", "name": "running_mean", "sha256": "5341e6b2646979a70e57653007a1f310169421ec9bdd9f1a5648f75ade005af1", "shape": [64]}, {"dtype": "float32", "name": "running_var", "sha256": "2f20cd03c9cd392a406c56232b0ff93a15f6d6d7da79086bfa14f55d4a4031b0", "shape": [64]}, {"dtype": "float32", "name": "weight", "sha256": "2f20cd03c9cd392a406c56232b0ff93a15f6d6d7da79086bfa14f55d4a4031b0", "shape": [64]}] | [{"dtype": "float32", "name": "x", "sha256": "cd7e0a7d8c0c594cca519b9cf49c8f086513282644cb8b6c06771dcb98504f2d", "shape": [64, 64, 56, 56]}, {"dtype": "float32", "name": "bias", "sha256": "5341e6b2646979a70e57653007a1f310169421ec9bdd9f1a5648f75ade005af1", "shape": [64]}, {"dtype": "int64", "name": "num_batches_tracked", "sha256": "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc", "shape": [1]}, {"dtype": "float32", "name": "running_mean", "sha256": "5341e6b2646979a70e57653007a1f310169421ec9bdd9f1a5648f75ade005af1", "shape": [64]}, {"dtype": "float32", "name": "running_var", "sha256": "2f20cd03c9cd392a406c56232b0ff93a15f6d6d7da79086bfa14f55d4a4031b0", "shape": [64]}, {"dtype": "float32", "name": "weight", "sha256": "2f20cd03c9cd392a406c56232b0ff93a15f6d6d7da79086bfa14f55d4a4031b0", "shape": [64]}] | [{"dtype": "float32", "name": "x", "sha256": "cd7e0a7d8c0c594cca519b9cf49c8f086513282644cb8b6c06771dcb98504f2d", "shape": [64, 64, 56, 56]}, {"dtype": "float32", "name": "bias", "sha256": "5341e6b2646979a70e57653007a1f310169421ec9bdd9f1a5648f75ade005af1", "shape": [64]}, {"dtype": "int64", "name": "num_batches_tracked", "sha256": "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc", "shape": [1]}, {"dtype": "float32", "name": "running_mean", "sha256": "5341e6b2646979a70e57653007a1f310169421ec9bdd9f1a5648f75ade005af1", "shape": [64]}, {"dtype": "float32", "name": "running_var", "sha256": "2f20cd03c9cd392a406c56232b0ff93a15f6d6d7da79086bfa14f55d4a4031b0", "shape": [64]}, {"dtype": "float32", "name": "weight", "sha256": "2f20cd03c9cd392a406c56232b0ff93a15f6d6d7da79086bfa14f55d4a4031b0", "shape": [64]}] | [{"dtype": "float32", "name": "x", "sha256": "cd7e0a7d8c0c594cca519b9cf49c8f086513282644cb8b6c06771dcb98504f2d", "shape": [64, 64, 56, 56]}, {"dtype": "float32", "name": "bias", "sha256": "5341e6b2646979a70e57653007a1f310169421ec9bdd9f1a5648f75ade005af1", "shape": [64]}, {"dtype": "int64", "name": "num_batches_tracked", "sha256": "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc", "shape": [1]}, {"dtype": "float32", "name": "running_mean", "sha256": "5341e6b2646979a70e57653007a1f310169421ec9bdd9f1a5648f75ade005af1", "shape": [64]}, {"dtype": "float32", "name": "running_var", "sha256": "2f20cd03c9cd392a406c56232b0ff93a15f6d6d7da79086bfa14f55d4a4031b0", "shape": [64]}, {"dtype": "float32", "name": "weight", "sha256": "2f20cd03c9cd392a406c56232b0ff93a15f6d6d7da79086bfa14f55d4a4031b0", "shape": [64]}] |
| momentum | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| num_features | 64 | 64 | 64 | 64 | 64 |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 |
| track_running_stats | true | true | true | true | true |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | hardware | version | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | AMD Instinct MI300X (amd2, Hot Aisle) | 0.8.49 | Xq | - | 0.4 | 0.4..0.4 | 1 | - | - | - | 0.4 | - (stored kernel) | - | - | max_rel_diff_vs_torch_eager_fp32=0.001395, rel_fro_vs_torch_eager_fp32=1.249e-07, relative_error_vs_own_host=0.000000, scorer=scored | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | Xq | - | 0.5 | 0.5..0.5 | 1 | 0.768 | - | 24.1 | 0.5 | 23.63 (predict_upload_ms (the predict inputs' own copy)) | 0.768 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | Xq | - | 1.0 | 1.0..1.0 | 1 | 0.377 | - | 21.5 | 1.0 | 20.42 (predict_upload_ms (the predict inputs' own copy)) | 0.377 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.001471, rel_fro_vs_torch_eager_fp32=6.868e-08 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | Xq | - | 0.6 | 0.6..0.6 | 1 | 0.695 | - | 21.2 | 0.6 | 20.67 (predict_upload_ms (the predict inputs' own copy)) | 0.695 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | Xq | - | 0.3 | 0.3..0.3 | 1 | 1.507 | - | 21.6 | 0.3 | 21.38 (predict_upload_ms (the predict inputs' own copy)) | 1.507 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.001471, rel_fro_vs_torch_eager_fp32=6.868e-08 | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: mojolearn.BatchNorm2d(affine=True, eps=1e-05, momentum=0.1, num_features=64, track_running_stats=True, x=float32[64x64x56x56], bias=float32[64], num_batches_tracked=int64[1], running_mean=float32[64], running_var=float32[64], ...); forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: torch.BatchNorm2d(affine=True, eps=1e-05, momentum=0.1, num_features=64, seed=7, track_running_stats=True, x=float32[64x64x56x56], bias=float32[64], num_batches_tracked=int64[1], running_mean=float32[64], running_var=float32[64], ...); forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: torch.BatchNorm2d(affine=True, eps=1e-05, momentum=0.1, num_features=64, seed=7, track_running_stats=True, x=float32[64x64x56x56], bias=float32[64], num_batches_tracked=int64[1], running_mean=float32[64], running_var=float32[64], ...); forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: torch.BatchNorm2d(affine=True, eps=1e-05, momentum=0.1, num_features=64, seed=7, track_running_stats=True, x=float32[64x64x56x56], bias=float32[64], num_batches_tracked=int64[1], running_mean=float32[64], running_var=float32[64], ...); forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: torch.BatchNorm2d(affine=True, eps=1e-05, momentum=0.1, num_features=64, seed=7, track_running_stats=True, x=float32[64x64x56x56], bias=float32[64], num_batches_tracked=int64[1], running_mean=float32[64], running_var=float32[64], ...); forward(x) (no autograd)(Xq)

### cagra / istella (rows full, shape index 400000x220; queries 4000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/measure-0852/ingest0849/inputs/amd-algos.cagra.istella.rows=full.json (amd2/paired-568fc5d15d2ea8e23507a933f7462b62df8c0a01f86c7c5c4e9e80e486b47bd9)`, ran on enc1-gpuvm005

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI300X (amd2, Hot Aisle) | 0.8.49 | identical | 16347.3 | 16347.3..16347.3 | 1 | - | - | 16347.3 | 16339.7 | 7.58 (upload_ms_separate) | - | - | 3546.7 | 2059.2 | recall_at_10=0.997250 | - | main board, one scored run | wheel | ok (main@80ace6b7a amd2/paired-568fc5d15d2ea8e23507a933f7462b62df8c0a01f86c7c5c4e9e80e486b47bd9 2026-10-16; identity vs the nvidia columns: n/a) |
| hipvs-gpu | hipvs | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | hipvs 01.00.00 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "CuvsException('RAFT failure at file=/src/hipVS/cpp/src/neighbors/detail/cagra/graph_core.cuh line=1466: Could not generate an intermediate CAGRA graph because the initial kNN graph contains) (measured this run) |

calls:
- `hipvs-gpu`: `cuvs.build(graph_degree=32, intermediate_graph_degree=64, itopk_size=64, n_neighbors=10)`

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/153619/vram_* bytes for this pid at the round's end (not a peak)

memory, hipvs-gpu: host not sampled; GPU not sampled

settings: {'graph_degree': 32, 'intermediate_graph_degree': 64, 'itopk_size': 64, 'n_neighbors': 10, 'random_state': 7}. Rows: None. Timed: None.

mismatch: cuvs-gpu (NVIDIA) and hipvs-gpu (AMD, hipVS) are CAGRA itself

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | hipvs-gpu | ours |
|---|---||---|---|
| library (source) | cuvs (declared) | mojolearn (declared) |
| graph_degree | 32 | 32 |
| intermediate_graph_degree | 64 | 64 |
| itopk_size | 64 | 64 |
| n_neighbors | 10 | 10 |
| seed | "none (no argument; draws random numbers, see exceptions)" | 7 |

accepted difference: hipvs-gpu seed: cuVS IndexParams take no seed and the index build samples rows; ours and faiss get 7

Inference (each arm predicts with its own model from the fit rounds above):

| arm | hardware | version | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | AMD Instinct MI300X (amd2, Hot Aisle) | 0.8.49 | Xq | - | 237.9 | 237.9..237.9 | 1 | - | - | 237.9 | 237.8 | 0.10 (predict_upload_ms) | - | - | recall_at_10=0.997250 | - | LIKE-FOR-LIKE-SPAN | ok |
| hipvs-gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | hipvs 01.00.00 | Xq | - | - | - | 0 | - | - | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(error: {"error": "CuvsException('RAFT failure at file=/src/hipVS/cpp/src/neighbors/detail/cagra/graph_core.cuh line=1466: Could not generate an intermediate CAGRA graph because the initial kNN graph contains) |

inference call, ours: search(queries)(Xq)

inference call, hipvs-gpu: cuvs.build(graph_degree=32, intermediate_graph_degree=64, itopk_size=64, n_neighbors=10); search(queries)(Xq)

### cagra / taxi (rows full, shape index 400000x11; queries 4000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/measure-0852/ingest0849/inputs/amd-algos.cagra.taxi.rows=full.json (amd2/paired-f71a189e64fec47d54b6101b6a14c146cf1d896936c4538b64e140f29bc96e4a)`, ran on enc1-gpuvm005

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI300X (amd2, Hot Aisle) | 0.8.49 | identical | 4766.8 | 4766.8..4766.8 | 1 | - | - | 4766.8 | 4766.3 | 0.44 (upload_ms_separate) | - | - | 2621.1 | 2059.2 | recall_at_10=0.997925 | - | main board, one scored run | wheel | ok (main@80ace6b7a amd2/paired-f71a189e64fec47d54b6101b6a14c146cf1d896936c4538b64e140f29bc96e4a 2026-10-16; identity vs the nvidia columns: n/a) |
| hipvs-gpu | hipvs | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | hipvs 01.00.00 | opponent | 1443.9 | 1443.9..1443.9 | 1 | 3.301 | - | 1493.2 | 1443.9 | 49.26 (upload_ms_untimed) | 3.192 (whole/whole) | - | 3020.5 | 2192.6 | recall_at_10=0.628775 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

calls:
- `hipvs-gpu`: `cuvs.build(graph_degree=32, intermediate_graph_degree=64, itopk_size=64, n_neighbors=10)`

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/154333/vram_* bytes for this pid at the round's end (not a peak)

memory, hipvs-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/154334/vram_* bytes for this pid at the round's end (not a peak)

settings: {'graph_degree': 32, 'intermediate_graph_degree': 64, 'itopk_size': 64, 'n_neighbors': 10, 'random_state': 7}. Rows: None. Timed: None.

mismatch: cuvs-gpu (NVIDIA) and hipvs-gpu (AMD, hipVS) are CAGRA itself

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | hipvs-gpu | ours |
|---|---||---|---|
| library (source) | cuvs (declared) | mojolearn (declared) |
| graph_degree | 32 | 32 |
| intermediate_graph_degree | 64 | 64 |
| itopk_size | 64 | 64 |
| n_neighbors | 10 | 10 |
| seed | "none (no argument; draws random numbers, see exceptions)" | 7 |

accepted difference: hipvs-gpu seed: cuVS IndexParams take no seed and the index build samples rows; ours and faiss get 7

Inference (each arm predicts with its own model from the fit rounds above):

| arm | hardware | version | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | AMD Instinct MI300X (amd2, Hot Aisle) | 0.8.49 | Xq | - | 154.2 | 154.2..154.2 | 1 | - | - | 154.2 | 154.1 | 0.05 (predict_upload_ms) | - | - | recall_at_10=0.997925 | - | LIKE-FOR-LIKE-SPAN | ok |
| hipvs-gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | hipvs 01.00.00 | Xq | - | 2.6 | 2.6..2.6 | 1 | 59.233 | - | 2.8 | 2.6 | 0.19 (predict_upload_ms (the predict inputs' own copy)) | 55.253 (whole/whole) | - | recall_at_10=0.628775 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: search(queries)(Xq)

inference call, hipvs-gpu: cuvs.build(graph_degree=32, intermediate_graph_degree=64, itopk_size=64, n_neighbors=10); search(queries)(Xq)

### cholesky / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/measure-0852/ingest0849/inputs/amd-algos.cholesky.synthetic.rows=full.json (amd2/paired-e42e16c32faed6503f9677a16f88034b9ef53bbd124175e50677c4c52bfbec90)`, ran on enc1-gpuvm005

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI300X (amd2, Hot Aisle) | 0.8.49 | identical | 348.0 | 348.0..348.0 | 1 | - | - | 348.0 | 341.9 | 6.10 (upload_ms_separate) | - | - | 4183.4 | 2229.2 | relative_residual=2.9e-07 | - | main board, one scored run | wheel | ok (main@80ace6b7a amd2/paired-e42e16c32faed6503f9677a16f88034b9ef53bbd124175e50677c4c52bfbec90 2026-10-16; identity vs the nvidia columns: n/a) |
| torch-gpu | torch | gpu | AMD Instinct MI300X VF (host enc1-gpuvm005) | torch 2.13.0+rocm7.1 | opponent | 32.9 | 32.9..32.9 | 1 | 10.568 | - | 196.0 | 32.9 | 163.09 (upload_ms_untimed) | 10.383 (kernel/kernel) | - | 3583.7 | 790.4 | relative_residual=2.519e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

calls:
- `ours`: `mojolearn.Cholesky(jitter=0, seed=7, arg0=float32[8192x8192])`
- `torch-gpu`: `torch.linalg_cholesky(jitter=0, seed=7, arg0=float32[8192x8192])`

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/154944/vram_* bytes for this pid at the round's end (not a peak)

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'jitter': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-gpu |
|---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) |
| input_arrays | [{"dtype": "float32", "sha256": "e1cc202332be3803baafedd32dff310e4af25316f9711e8522028b15962d0054", "shape": [8192, 8192]}] | [{"dtype": "float32", "sha256": "e1cc202332be3803baafedd32dff310e4af25316f9711e8522028b15962d0054", "shape": [8192, 8192]}] |
| jitter | 0.0 | 0.0 |
| seed | 7 | 7 |

### incremental-pca / taxi (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd2-results.txt (amd2/b0002)`, ran on AMD Instinct MI300X (amd2, Hot Aisle) job b0002

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI300X (amd2, Hot Aisle) | mojolearn 0.8.37 (source build, main@8d5855b83) | identical | 74.1 | 74.1..74.1 | 1 | - | - | 74.1 | - | - (stored whole) | - | - | - | - | explained_variance_fraction=0.999995 | - | main board, one scored run | - | ok (main@8d5855b83 amd2/b0002 2026-10-10; identity vs the nvidia columns: n/a) |

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

## Not covered by this board

- Classical, wave 2: RadiusNeighbors, the preprocessing scalers, HDBSCAN's prediction data, Cholesky and the parallel_* and Distributed* wrappers are public and not raced here; taxi-derived time series are not used (the ARIMA and ExponentialSmoothing lanes fit seeded synthetic series, as the repo's own ARIMA quality work does).
- Classical, wave 2, not planned on this vendor: cuML: CUDA only; no ROCm build exists. cuVS's ROCm port, hipVS (amd-hipvs), is the IVF-Flat GPU arm (hipvs-gpu); it has no ivf_sq, refine, UMAP, spectral or k-NN estimators, so those lanes keep their CPU opponents.
- Classical, wave 2, not planned on this vendor: faiss-gpu: the pinned FAISS GPU builds are CUDA; faiss-cpu is the arm on this box.
- Classical, wave 2, not planned on this vendor: scikit-learn on linearsvc: refused, different algorithm and model: liblinear (dual='auto' takes its primal trust-region Newton here) with a penalized intercept; ours and cuML L-BFGS on the primal with an unpenalized intercept (2026-10-13 audit); no opponent on this box.
- Classical, wave 2, not planned on this vendor: scikit-learn on linearsvr: refused, different algorithm and model: liblinear dual coordinate descent with a penalized intercept; ours and cuML L-BFGS on the primal with an unpenalized intercept (2026-10-13 audit); no opponent on this box.
- Classical, wave 2, not planned on this vendor: scikit-learn on tsvd: refused, different algorithm: ours 'covariance_eigh' (the eigendecomposition of X^T X); scikit-learn TruncatedSVD has only 'arpack' (Lanczos on X) and 'randomized', and no CPU library has ours' algorithm (2026-10-13 audit); no opponent on this box.
- Inference, trees: a single-row latency batch is not timed (the batches are the held-out split and 1,000,000 training rows); ONNX, Treelite and other export paths are not raced.
- Inference, classical: the classical2 family's predict calls (the linear models, GaussianMixture, SVR, KernelRidge and others in tools/bench_board_more.py) are timed as those lanes define their clocks, not as a separate inference cell; svc times predict, not decision_function.
- Our CPU: never raced or reported; the board races only our GPU (Andrew, Oct 2 2026). The host column gives same-bits digests only (lq ID).
- Neural, not planned: lm-host-train-step, lm-infer, mamba1-infer, mamba2-infer, mamba3-infer, mlp-infer, samba-infer, transformer-infer: ours runs the CPU binding, and our CPU is never raced, in no numeric mode (the Apple FAST neural tier is the GPU lanes only).
- Memory: GPU memory on Apple has no per-process counter (Metal buffers are inside the host footprint); the trees driver runs every arm in one process, so its GPU figure is the process total; a figure taken at the round's end misses a buffer freed inside the round; inference cells carry memory only on the classical lanes.
- Neural: The Mamba lanes' torch-* arms would be the repo's pure-PyTorch references (mamba/corpus/gen_corpus.py: mamba_ssm's selective_scan_ref for Mamba-1, the chunked SSD reference for Mamba-2, the SISO reference for Mamba-3), not mamba-ssm's fused CUDA/Triton kernels; a reference implementation is not an opponent, so they are refused by name. The Mamba opponent is mamba_ssm's own fused kernels (mamba-ssm-fp32 on NVIDIA and AMD, mamba-ssm-tf32 on NVIDIA), and the Samba torch arms run mamba_ssm's Mamba3 layers with transformers' LlamaDecoderLayer attention; where mamba_ssm is not built (Apple) those lanes have no opponent.
- Neural: The blocks' backward (the Mamba and TransformerBlock VJPs), ragged `lengths` and a prefill followed by decode are public and not raced; the zero-state forward (*-forward) and the zero-state token-by-token decode (*-decode) are.
- Neural: The byte LM has no incremental decode on any route: LanguageModelTrainer (GPU) and LanguageModelInference (CPU) expose full-sequence logits only (lm-forward on the GPU), no KV-cache state or step, so there is no lm-decode row.
- Neural: The *-infer and lm-host-train-step rows are the CPU host binding and are never raced; their GPU twins are the *-forward, *-decode and mlp-predict rows.
- Neural: The GPT-3-small target shape is not on the board (the LM lanes use the smaller control shape so one shape runs on every box, a 16 GB Mac included).
- Neural, not planned on this vendor: torch-eager-tf32 / torch-compile-tf32: TF32 is an NVIDIA CUDA tensor-core matmul mode; torch on ROCm accepts the flag and changes nothing
- Neural, not planned on this vendor: torch-compile-* on mamba1-forward and mamba1-infer: the only torch Mamba-1 twin is the pure-PyTorch reference scan, a per-token Python loop that torch.compile would unroll L times; mamba1 races the eager arms only
- Neural, not planned on this vendor: gemm-bf16 races torch's bf16 settings only (bf16 operands, fp32 accumulate)
- Neural, not planned on this vendor: mamba1-decode, mamba2-decode, mamba3-decode and samba-decode race ours alone: the repo's torch Mamba references are full-sequence scans with no carried-state decode step (a torch decode twin is not written yet)
- Neural, not planned on this vendor: torch-compile-* on transformer-decode: the twin is a per-token loop over a growing KV cache; transformer-decode races the eager arms only
- Neural, not planned on this vendor: gemm-int8: torch._int_mm is a CUDA kernel; torch on ROCm has no int8 matmul, so ours races alone
- Neural, not planned on this vendor: mamba-ssm-tf32: TF32 is an NVIDIA CUDA mode; mamba_ssm races in float32 (mamba-ssm-fp32, from its ROCm/HIP source build) on AMD
- Neural, not planned on this vendor: mamba-ssm bf16: our Mamba blocks are float32, so mamba_ssm races in float32 (and its TF32 setting on NVIDIA)

