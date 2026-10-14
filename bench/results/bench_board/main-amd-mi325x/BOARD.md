# mojolearn benchmark board

Generated 2026-10-14T06:49:28Z from `board.json` (schema `mojolearn-bench-board/1`).

> MAIN BOARD amd-mi325x, version label main@50be53955. Unreleased: not reproducible by pip install; the release boards are the reference.

> Cells: 230 races; oldest cell main@89485bc96 (2026-10-08T13:27:06Z), newest cell main@50be53955 (2026-10-14T05:52:40Z). Boxes: AMD Instinct MI325X (amd, DO).

> Rule: each lane x dataset shows the newest default-configuration race on main (highest commit date, then job number) whose status is ok. A newer ok cell replaces an older one whatever the two times are; a run that is not ok is never a numeric cell and never replaces an ok cell (FAILED table; an older ok cell stays, flagged with the newer failed run). 309 replaced or failed observations are in LEDGER.md. A/B and grid arms (MOJOLEARN_BUILD_DEFINES, MOJOLEARN_GRID_TAG grid runs) are never on this board.

> Ours: recorded scored runs from lq jobs or verified paired sessions; the status column names the cell's commit, box/job, commit date and the other vendor's digest at the same commit (identity: MATCH 87, n/a 143).

> Opponents: retained from each paired session when present; otherwise retained historical pairs or stored opponent boards (opponents-20261006, release-board-resume-r2). Ingestion runs no measurements. `ours IDENTICAL / arm` divides the two stored medians, and the clock columns read a torch GPU arm kernel/kernel and every other arm whole/whole (AGENTS.md measurement item 6). Our kernel clock is `-` unless the cell recorded upload_ms_separate. Opponents withheld for changed lane settings: 37 races.

> Provenance: main@ identifies the source or harness freeze. Paired wheel runs separately record the installed library version and loaded binary hashes; they are not source builds of that harness commit.

> Neural lanes: the headline (its own table, and a line under each neural race) is ours IDENTICAL over torch's fastest bf16 arm, eager or compile, what customers run; the fp32 twin is the second column. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax. A cell of ours whose output hash equals a copied opponent's shows quality identical_to=<arm> (the same bits) where the own-host reference gave none.

## Identity

Same lane, dataset and commit on the other GPU vendor (identity = equal output digests on NVIDIA and AMD). Counts: MATCH 87, n/a 143.

DIFFER: none.

## FAILED

Runs on main whose status is not ok (error, refused, timeout, not_ready, NO-RECORD, NO-OURS-CELL). They are never a numeric cell and never replace an ok cell; an older ok cell stays on the board flagged with the failed run. 6 failed runs.

| lane | dataset | commit | hardware | version | box/job | reason | ok cell on the board |
|---|---|---|---|---|---|---|---|
| gcn | istella | main@a8fd6d926 | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@a8fd6d926) | amd/a0006 | not_ready | main@6416999f7 amd/a1210 stays |
| gcn | taxi | main@a8fd6d926 | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@a8fd6d926) | amd/a0006 | not_ready | main@6416999f7 amd/a1210 stays |
| graphsage | istella | main@a8fd6d926 | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@a8fd6d926) | amd/a0006 | not_ready | main@6416999f7 amd/a1210 stays |
| graphsage | taxi | main@a8fd6d926 | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@a8fd6d926) | amd/a0006 | not_ready | main@6416999f7 amd/a1210 stays |
| moe | synthetic | main@a8fd6d926 | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@a8fd6d926) | amd/a0006 | not_ready | main@a1c96a5ba amd/a1682 stays |
| resnet-block | synthetic | main@a8fd6d926 | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@a8fd6d926) | amd/a0006 | not_ready | main@bac431c11 amd/a0004 stays |

## Box

| field | value |
|---|---|
| vendor / API | amd / hip |
| GPU | AMD Instinct MI325X |
| GPU driver | - |
| CPU | None (None logical cores) |
| memory bytes | - |
| OS | - |
| Python | - |
| mojolearn | main@50be53955 (wheel mixed source/wheel provenance; see each cell's version and measurement record, sha256 -) |
| script commit | 50be53955e44d38634eb7117dfc43a3dc427be01 |
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

Races: 230 planned, 230 done, 0 failed, 0 unsupported, 0 pending. Cells: 595 (HOST-MEMORY 1, REFUSED 12, ok 582).

Inference cells: 65 (REFUSED 2, ok 63).

## Quality at a glance

Per lane and dataset: our FAST value, our IDENTICAL value, and each opponent's.

| family | lane | dataset | metric | ours FAST | ours IDENTICAL | opponents |
|---|---|---|---|---|---|---|
| algos | adafactor | synthetic | optimizer_update_error_vs_torch_fp32 | - | 3.386e-06 | torch-eager-fp32 0.000000; torch-compile-fp32 0.000000 |
| algos | adafactor | synthetic | rel_fro_vs_torch_eager_fp32 | - | 1.033e-08 | torch-eager-fp32 0.000000; torch-compile-fp32 0.000000 |
| algos | adagrad | synthetic | optimizer_update_error_vs_torch_fp32 | - | 1.874e-06 | torch-eager-fp32 0.000000; torch-compile-fp32 0.000000 |
| algos | adagrad | synthetic | rel_fro_vs_torch_eager_fp32 | - | 3.207e-09 | torch-eager-fp32 0.000000; torch-compile-fp32 0.000000 |
| algos | adam | synthetic | optimizer_update_error_vs_torch_fp32 | - | 8.844e-06 | torch-eager-fp32 0.000000; torch-compile-fp32 1.355e-05; torch-fused-fp32 6.002e-06 |
| algos | adam | synthetic | rel_fro_vs_torch_eager_fp32 | - | 3.34e-08 | torch-eager-fp32 0.000000; torch-compile-fp32 5.116e-08; torch-fused-fp32 2.267e-08 |
| algos | adamax | synthetic | optimizer_update_error_vs_torch_fp32 | - | 1.54e-06 | torch-eager-fp32 0.000000; torch-compile-fp32 2.292e-06 |
| algos | adamax | synthetic | rel_fro_vs_torch_eager_fp32 | - | 4.144e-09 | torch-eager-fp32 0.000000; torch-compile-fp32 6.17e-09 |
| algos | adamw | synthetic | optimizer_update_error_vs_torch_fp32 | - | 8.841e-06 | torch-eager-fp32 0.000000; torch-compile-fp32 6.012e-05; torch-fused-fp32 6.574e-05 |
| algos | adamw | synthetic | rel_fro_vs_torch_eager_fp32 | - | 3.34e-08 | torch-eager-fp32 0.000000; torch-compile-fp32 2.272e-07; torch-fused-fp32 2.484e-07 |
| algos | ard | istella | r2 (higher is better) | - | -0.124856 | sklearn-cpu 0.327436 |
| algos | ard | istella | rmse (lower is better) | - | 0.885949 | sklearn-cpu 0.685058 |
| algos | avgpool1d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | 0.000000 | torch-eager-fp32 0.000000; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | avgpool1d | synthetic | rel_fro_vs_torch_eager_fp32 | - | 0.000000 | torch-eager-fp32 0.000000; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | avgpool2d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | 0.000000 | torch-eager-fp32 0.000000; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | avgpool2d | synthetic | rel_fro_vs_torch_eager_fp32 | - | 0.000000 | torch-eager-fp32 0.000000; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | batchnorm1d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | 0.004642 | torch-eager-fp32 0.000000; torch-compile-fp32 0.002793; torch-eager-bf16 0.000000; torch-compile-bf16 0.002793 |
| algos | batchnorm1d | synthetic | rel_fro_vs_torch_eager_fp32 | - | 8.097e-08 | torch-eager-fp32 0.000000; torch-compile-fp32 5.248e-08; torch-eager-bf16 0.000000; torch-compile-bf16 5.248e-08 |
| algos | batchnorm2d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | 0.001395 | torch-eager-fp32 0.000000; torch-compile-fp32 0.001471; torch-eager-bf16 0.000000; torch-compile-bf16 0.001471 |
| algos | batchnorm2d | synthetic | rel_fro_vs_torch_eager_fp32 | - | 1.249e-07 | torch-eager-fp32 0.000000; torch-compile-fp32 6.868e-08; torch-eager-bf16 0.000000; torch-compile-bf16 6.868e-08 |
| algos | bayesian-ridge | istella | r2 (higher is better) | - | 0.317909 | sklearn-cpu -8.145e+07 |
| algos | bayesian-ridge | istella | rmse (lower is better) | - | 0.689893 | sklearn-cpu 7538.692925 |
| algos | bernoulli-nb | istella | accuracy (higher is better) | - | 0.794050 | sklearn-cpu 0.794050 |
| algos | bernoulli-nb | istella | logloss (lower is better) | - | 5.350625 | sklearn-cpu 4.278741 |
| algos | bernoulli-nb | taxi | accuracy (higher is better) | - | 0.755560 | sklearn-cpu 0.755560 |
| algos | bernoulli-nb | taxi | logloss (lower is better) | - | 0.557803 | sklearn-cpu 0.557802 |
| algos | binarizer | istella | max_abs_diff_vs_sklearn | - | 0.000000 | sklearn-cpu - |
| algos | binarizer | istella | rel_diff_vs_sklearn | - | 0.000000 | sklearn-cpu - |
| algos | binarizer | taxi | max_abs_diff_vs_sklearn | - | 0.000000 | sklearn-cpu - |
| algos | binarizer | taxi | rel_diff_vs_sklearn | - | 0.000000 | sklearn-cpu - |
| algos | categorical-nb | istella | accuracy (higher is better) | - | 0.838850 | sklearn-cpu 0.838850 |
| algos | categorical-nb | istella | logloss (lower is better) | - | 0.412625 | sklearn-cpu 0.412625 |
| algos | categorical-nb | taxi | accuracy (higher is better) | - | 0.765850 | sklearn-cpu 0.765850 |
| algos | categorical-nb | taxi | logloss (lower is better) | - | 0.538866 | sklearn-cpu 0.538866 |
| algos | cholesky | synthetic | relative_residual | - | 2.9e-07 | torch-gpu 2.519e-07 |
| algos | cnn-clf | synthetic | accuracy (higher is better) | - | 1.000000 | torch-eager-fp32 1.000000; torch-compile-fp32 1.000000; torch-eager-bf16 1.000000; torch-compile-bf16 1.000000 |
| algos | complement-nb | istella | accuracy (higher is better) | - | 0.849360 | sklearn-cpu 0.849350 |
| algos | complement-nb | istella | logloss (lower is better) | - | 3.762524 | sklearn-cpu 3.174763 |
| algos | complement-nb | taxi | accuracy (higher is better) | - | 0.678020 | sklearn-cpu 0.678030 |
| algos | complement-nb | taxi | logloss (lower is better) | - | 0.715492 | sklearn-cpu 0.715493 |
| algos | complement-nb | text | accuracy (higher is better) | - | 0.983067 | sklearn-cpu 0.983067 |
| algos | complement-nb | text | logloss (lower is better) | - | 0.559491 | sklearn-cpu 0.557285 |
| algos | connected-components | istella | n_components | - | 81 | - |
| algos | connected-components | taxi | n_components | - | 588 | - |
| algos | conv1d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | 0.318264 | torch-eager-fp32 0.000000; torch-compile-fp32 0.000000; torch-eager-bf16 3418.035805; torch-compile-bf16 3418.035805 |
| algos | conv1d | synthetic | rel_fro_vs_torch_eager_fp32 | - | 3.948e-07 | torch-eager-fp32 0.000000; torch-compile-fp32 0.000000; torch-eager-bf16 0.004377; torch-compile-bf16 0.004377 |
| algos | conv2d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | 0.372529 | torch-eager-fp32 0.000000; torch-compile-fp32 0.000000; torch-eager-bf16 3418.486565; torch-compile-bf16 3418.486565 |
| algos | conv2d | synthetic | rel_fro_vs_torch_eager_fp32 | - | 3.307e-07 | torch-eager-fp32 0.000000; torch-compile-fp32 0.000000; torch-eager-bf16 0.003382; torch-compile-bf16 0.003382 |
| algos | damped-ets | synthetic | forecast_rmse (lower is better) | - | 13.931153 | statsmodels-cpu 26.588738; statsforecast-cpu 13.945986 |
| algos | damped-ets | taxi-hourly | forecast_rmse (lower is better) | - | 96.690449 | statsmodels-cpu 196.955273; statsforecast-cpu 96.685568 |
| algos | dropout2d | synthetic | dropout_contract_relative_error | - | 5.422e-08 | torch-eager-fp32 5.422e-08; torch-compile-fp32 5.421e-08 |
| algos | dropout2d | synthetic | dropout_probability_z | - | 0.708333 | torch-eager-fp32 1.489583; torch-compile-fp32 1.791667 |
| algos | dropout2d | synthetic | dropped_channel_fraction | - | 0.096680 | torch-eager-fp32 0.093018; torch-compile-fp32 0.108398 |
| algos | dropout2d | synthetic | expected_dropout_probability | - | 0.100000 | torch-eager-fp32 0.100000; torch-compile-fp32 0.100000 |
| algos | dropout2d | synthetic | identifiable_channels | - | 4096 | torch-eager-fp32 4096; torch-compile-fp32 4096 |
| algos | dynamic-optimized-theta | synthetic | forecast_rmse (lower is better) | - | 1.436278 | statsforecast-cpu 1.436045 |
| algos | dynamic-optimized-theta | taxi-hourly | forecast_rmse (lower is better) | - | 49.086628 | statsforecast-cpu 49.314797 |
| algos | eigh | synthetic | max_eigenvalue_error | - | 3.692e-07 | torch-gpu 1.044e-06 |
| algos | eigh | synthetic | relative_residual | - | 6.672e-07 | torch-gpu 9.614e-07 |
| algos | elliptic-envelope | istella | fraction_flagged | - | 0.092480 | sklearn-cpu - |
| algos | elliptic-envelope | taxi | fraction_flagged | - | 0.102090 | sklearn-cpu 0.102470 |
| algos | elliptic-envelope | taxi | jaccard_vs_sklearn | - | - | sklearn-cpu 1.000000 |
| algos | embedding | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | 0.000000 | torch-eager-fp32 0.000000; torch-compile-fp32 0.000000 |
| algos | embedding | synthetic | rel_fro_vs_torch_eager_fp32 | - | 0.000000 | torch-eager-fp32 0.000000; torch-compile-fp32 0.000000 |
| algos | enet-cv | istella | r2 (higher is better) | - | 0.326805 | sklearn-cpu 0.326805 |
| algos | enet-cv | istella | rmse (lower is better) | - | 0.685379 | sklearn-cpu 0.685379 |
| algos | enet-cv | taxi | r2 (higher is better) | - | 0.909004 | sklearn-cpu 0.909004 |
| algos | enet-cv | taxi | rmse (lower is better) | - | 4.804486 | sklearn-cpu 4.804486 |
| algos | gaussian-nb | istella | accuracy (higher is better) | - | 0.876570 | sklearn-cpu 0.876530 |
| algos | gaussian-nb | istella | logloss (lower is better) | - | 3.574420 | sklearn-cpu 3.417392 |
| algos | gaussian-nb | taxi | accuracy (higher is better) | - | 0.719820 | sklearn-cpu 0.719900 |
| algos | gaussian-nb | taxi | logloss (lower is better) | - | 1.132249 | sklearn-cpu 1.133898 |
| algos | gcn | istella | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.025557; torch-eager-bf16 2662.412698; torch-compile-bf16 2662.401917 |
| algos | gcn | istella | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 9.36e-08; torch-eager-bf16 0.002187; torch-compile-bf16 0.002187 |
| algos | gcn | taxi | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.001702; torch-eager-bf16 1509.509399; torch-compile-bf16 1509.509515 |
| algos | gcn | taxi | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 7.846e-08; torch-eager-bf16 0.002330; torch-compile-bf16 0.002330 |
| algos | global-avgpool | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | 0.015661 | torch-eager-fp32 0.000000; torch-compile-fp32 0.007299; torch-eager-bf16 0.000000; torch-compile-bf16 0.007299 |
| algos | global-avgpool | synthetic | rel_fro_vs_torch_eager_fp32 | - | 1.506e-07 | torch-eager-fp32 0.000000; torch-compile-fp32 1.01e-07; torch-eager-bf16 0.000000; torch-compile-bf16 1.01e-07 |
| algos | global-maxpool | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | 0.000000 | torch-eager-fp32 0.000000; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | global-maxpool | synthetic | rel_fro_vs_torch_eager_fp32 | - | 0.000000 | torch-eager-fp32 0.000000; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | graphsage | istella | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.140845; torch-eager-bf16 3907.144070; torch-compile-bf16 3678.232431 |
| algos | graphsage | istella | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 8.05e-08; torch-eager-bf16 0.003366; torch-compile-bf16 0.003054 |
| algos | graphsage | taxi | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.162162; torch-eager-bf16 4680.142857; torch-compile-bf16 4608.154297 |
| algos | graphsage | taxi | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 8.776e-08; torch-eager-bf16 0.003557; torch-compile-bf16 0.003285 |
| algos | gru-clf | synthetic | accuracy (higher is better) | - | 0.971625 | torch-eager-fp32 0.971842; torch-compile-fp32 0.971842; torch-eager-bf16 0.971788; torch-compile-bf16 0.971788 |
| algos | gru-clf | synthetic | logloss (lower is better) | - | 0.070054 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | gru-clf | taxi-hourly | accuracy (higher is better) | - | 0.865723 | torch-eager-fp32 0.865668; torch-compile-fp32 0.865668; torch-eager-bf16 0.865777; torch-compile-bf16 0.865777 |
| algos | gru-clf | taxi-hourly | logloss (lower is better) | - | 0.303537 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | gru-reg | synthetic | r2 (higher is better) | - | 0.982337 | torch-eager-fp32 0.981946; torch-compile-fp32 0.981946; torch-eager-bf16 0.981940; torch-compile-bf16 0.981940 |
| algos | gru-reg | synthetic | rmse (lower is better) | - | 0.153975 | torch-eager-fp32 0.155672; torch-compile-fp32 0.155672; torch-eager-bf16 0.155700; torch-compile-bf16 0.155700 |
| algos | gru-reg | taxi-hourly | r2 (higher is better) | - | 0.747875 | torch-eager-fp32 0.748219; torch-compile-fp32 0.748219; torch-eager-bf16 0.748254; torch-compile-bf16 0.748254 |
| algos | gru-reg | taxi-hourly | rmse (lower is better) | - | 0.544554 | torch-eager-fp32 0.544182; torch-compile-fp32 0.544182; torch-eager-bf16 0.544144; torch-compile-bf16 0.544144 |
| algos | ivf-pq | istella | recall_at_10 (higher is better) | - | 0.802100 | faiss-cpu 0.801550 |
| algos | ivf-pq | taxi | recall_at_10 (higher is better) | - | 0.981250 | faiss-cpu 0.979450 |
| algos | ivf-refine | istella | recall_at_10 (higher is better) | - | 0.994575 | faiss-cpu 0.993125 |
| algos | jl-min-dim | synthetic | equal_fraction_vs_sklearn | - | 1.000000 | sklearn-cpu 1.000000 |
| algos | kbins | istella | max_abs_diff_vs_sklearn | - | 0.000000 | sklearn-cpu - |
| algos | kbins | istella | rel_diff_vs_sklearn | - | 0.000000 | sklearn-cpu - |
| algos | kbins | taxi | max_abs_diff_vs_sklearn | - | 0.000000 | sklearn-cpu - |
| algos | kbins | taxi | rel_diff_vs_sklearn | - | 0.000000 | sklearn-cpu - |
| algos | kernel-pca | istella | rbf_feature_distance_relative_stress | - | 0.610112 | sklearn-cpu - |
| algos | kernel-pca | istella | subspace_cos_vs_sklearn | - | 1.000000 | sklearn-cpu 1.000000 |
| algos | kernel-pca | taxi | rbf_feature_distance_relative_stress | - | 0.545709 | sklearn-cpu - |
| algos | kernel-pca | taxi | subspace_cos_vs_sklearn | - | 1.000000 | sklearn-cpu 1.000000 |
| algos | kernel-shap | istella | rel_error_vs_exact | - | 8.307e-08 | shap-cpu 8.63e-15 |
| algos | kernel-shap | taxi | rel_error_vs_exact | - | 1.056e-07 | - |
| algos | knn-imputer | istella | masked_rmse | - | 323953.237332 | sklearn-cpu 986208.700423 |
| algos | knn-imputer | taxi | masked_rmse | - | 6.151696 | sklearn-cpu 5.263919 |
| algos | label-binarizer | istella | max_abs_diff_vs_sklearn | - | 0.000000 | sklearn-cpu - |
| algos | label-binarizer | istella | rel_diff_vs_sklearn | - | 0.000000 | sklearn-cpu - |
| algos | label-binarizer | taxi | max_abs_diff_vs_sklearn | - | 0.000000 | sklearn-cpu - |
| algos | label-binarizer | taxi | rel_diff_vs_sklearn | - | 0.000000 | sklearn-cpu - |
| algos | label-encoder | istella | max_abs_diff_vs_sklearn | - | 0.000000 | sklearn-cpu - |
| algos | label-encoder | istella | rel_diff_vs_sklearn | - | 0.000000 | sklearn-cpu - |
| algos | label-encoder | taxi | max_abs_diff_vs_sklearn | - | 0.000000 | sklearn-cpu - |
| algos | label-encoder | taxi | rel_diff_vs_sklearn | - | 0.000000 | sklearn-cpu - |
| algos | lamb | synthetic | relative_error_vs_own_host | - | 0.000000 | - |
| algos | lars | istella | r2 (higher is better) | - | 0.328699 | sklearn-cpu -1.051e+22 |
| algos | lars | istella | rmse (lower is better) | - | 0.684414 | sklearn-cpu 8.562e+10 |
| algos | lars | taxi | r2 (higher is better) | - | 0.908983 | sklearn-cpu 0.908983 |
| algos | lars | taxi | rmse (lower is better) | - | 4.805051 | sklearn-cpu 4.805057 |
| algos | lasso-cv | istella | r2 (higher is better) | - | 0.325506 | sklearn-cpu 0.325507 |
| algos | lasso-cv | istella | rmse (lower is better) | - | 0.686040 | sklearn-cpu 0.686040 |
| algos | lasso-cv | taxi | r2 (higher is better) | - | 0.909038 | sklearn-cpu 0.909038 |
| algos | lasso-cv | taxi | rmse (lower is better) | - | 4.803593 | sklearn-cpu 4.803593 |
| algos | lasso-lars | istella | r2 (higher is better) | - | 0.310834 | sklearn-cpu 0.310591 |
| algos | lasso-lars | istella | rmse (lower is better) | - | 0.693461 | sklearn-cpu 0.693584 |
| algos | lasso-lars | taxi | r2 (higher is better) | - | 0.908998 | sklearn-cpu 0.908998 |
| algos | lasso-lars | taxi | rmse (lower is better) | - | 4.804663 | sklearn-cpu 4.804669 |
| algos | layernorm | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | 0.011086 | torch-eager-fp32 0.000000; torch-compile-fp32 0.007530; torch-eager-bf16 0.000000; torch-compile-bf16 0.007530 |
| algos | layernorm | synthetic | rel_fro_vs_torch_eager_fp32 | - | 6.669e-08 | torch-eager-fp32 0.000000; torch-compile-fp32 5.326e-08; torch-eager-bf16 0.000000; torch-compile-bf16 5.326e-08 |
| algos | lda-clf | istella | accuracy (higher is better) | - | 0.912830 | sklearn-cpu 0.899520 |
| algos | lda-clf | istella | logloss (lower is better) | - | 0.235875 | sklearn-cpu 0.439049 |
| algos | lda-clf | taxi | accuracy (higher is better) | - | 0.762530 | sklearn-cpu 0.762530 |
| algos | lda-clf | taxi | logloss (lower is better) | - | 0.539763 | sklearn-cpu 0.539767 |
| algos | lion | synthetic | relative_error_vs_own_host | - | 0.000000 | - |
| algos | louvain | istella | modularity | - | 0.911187 | - |
| algos | louvain | istella | n_communities | - | 40 | - |
| algos | louvain | taxi | modularity | - | 0.941953 | - |
| algos | louvain | taxi | n_communities | - | 58 | - |
| algos | lstm-clf | synthetic | accuracy (higher is better) | - | 0.967068 | torch-eager-fp32 0.968696; torch-compile-fp32 0.968696; torch-eager-bf16 0.968913; torch-compile-bf16 0.968913 |
| algos | lstm-clf | synthetic | logloss (lower is better) | - | 0.079720 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | lstm-clf | taxi-hourly | accuracy (higher is better) | - | 0.870443 | torch-eager-fp32 0.868218; torch-compile-fp32 0.868218; torch-eager-bf16 0.868056; torch-compile-bf16 0.868056 |
| algos | lstm-clf | taxi-hourly | logloss (lower is better) | - | 0.297146 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | lstm-reg | synthetic | r2 (higher is better) | - | 0.979398 | torch-eager-fp32 0.981013; torch-compile-fp32 0.981013; torch-eager-bf16 0.981004; torch-compile-bf16 0.981004 |
| algos | lstm-reg | synthetic | rmse (lower is better) | - | 0.166295 | torch-eager-fp32 0.159641; torch-compile-fp32 0.159641; torch-eager-bf16 0.159679; torch-compile-bf16 0.159679 |
| algos | lstm-reg | taxi-hourly | r2 (higher is better) | - | 0.754306 | torch-eager-fp32 0.751679; torch-compile-fp32 0.751679; torch-eager-bf16 0.751591; torch-compile-bf16 0.751591 |
| algos | lstm-reg | taxi-hourly | rmse (lower is better) | - | 0.537563 | torch-eager-fp32 0.540429; torch-compile-fp32 0.540429; torch-eager-bf16 0.540526; torch-compile-bf16 0.540526 |
| algos | lstsq | istella | relative_residual | - | 0.849956 | torch-gpu - |
| algos | lstsq | taxi | relative_residual | - | 0.756366 | torch-gpu 0.756366 |
| algos | lu-factor | synthetic | relative_residual | - | 3.256e-06 | torch-gpu 4.003e-07 |
| algos | lu-solve | synthetic | relative_residual | - | 3.256e-06 | torch-gpu 4.003e-07 |
| algos | maxabs-scaler | istella | max_abs_diff_vs_sklearn | - | 0.000000 | sklearn-cpu - |
| algos | maxabs-scaler | istella | rel_diff_vs_sklearn | - | 0.000000 | sklearn-cpu - |
| algos | maxabs-scaler | taxi | max_abs_diff_vs_sklearn | - | 0.000000 | sklearn-cpu - |
| algos | maxabs-scaler | taxi | rel_diff_vs_sklearn | - | 0.000000 | sklearn-cpu - |
| algos | maxpool1d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | 0.000000 | torch-eager-fp32 0.000000; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | maxpool1d | synthetic | rel_fro_vs_torch_eager_fp32 | - | 0.000000 | torch-eager-fp32 0.000000; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | maxpool2d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | 0.000000 | torch-eager-fp32 0.000000; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | maxpool2d | synthetic | rel_fro_vs_torch_eager_fp32 | - | 0.000000 | torch-eager-fp32 0.000000; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | min-cov-det | istella | n_features | - | 220 | sklearn-cpu - |
| algos | min-cov-det | istella | rel_diff_vs_sklearn | - | 0.239361 | sklearn-cpu - |
| algos | min-cov-det | taxi | n_features | - | 11 | sklearn-cpu 11 |
| algos | min-cov-det | taxi | rel_diff_vs_sklearn | - | 0.238565 | sklearn-cpu - |
| algos | minibatch-kmeans | istella | n_clusters | - | 8 | sklearn-cpu 8 |
| algos | minibatch-kmeans | istella | silhouette (higher is better) | - | 0.116696 | sklearn-cpu 0.111849 |
| algos | minibatch-kmeans | istella | ari_vs_ours (1 is our partition exactly) | - | - | sklearn-cpu 0.622446 |
| algos | minibatch-kmeans | taxi | n_clusters | - | 8 | sklearn-cpu 8 |
| algos | minibatch-kmeans | taxi | silhouette (higher is better) | - | 0.138060 | sklearn-cpu 0.165473 |
| algos | minibatch-kmeans | taxi | ari_vs_ours (1 is our partition exactly) | - | - | sklearn-cpu 0.525151 |
| algos | minmax-scaler | istella | max_abs_diff_vs_sklearn | - | 0.000000 | sklearn-cpu - |
| algos | minmax-scaler | istella | rel_diff_vs_sklearn | - | 0.000000 | sklearn-cpu - |
| algos | minmax-scaler | taxi | max_abs_diff_vs_sklearn | - | 0.000000 | sklearn-cpu - |
| algos | minmax-scaler | taxi | rel_diff_vs_sklearn | - | 0.000000 | sklearn-cpu - |
| algos | moe | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.044703; torch-eager-bf16 22862.630675; torch-compile-bf16 22880.029448 |
| algos | moe | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 2.011e-07; torch-eager-bf16 0.055222; torch-compile-bf16 0.055199 |
| algos | multilabel-binarizer | istella | max_abs_diff_vs_sklearn | - | 0.000000 | sklearn-cpu - |
| algos | multilabel-binarizer | istella | rel_diff_vs_sklearn | - | 0.000000 | sklearn-cpu - |
| algos | multilabel-binarizer | taxi | max_abs_diff_vs_sklearn | - | 0.000000 | sklearn-cpu - |
| algos | multilabel-binarizer | taxi | rel_diff_vs_sklearn | - | 0.000000 | sklearn-cpu - |
| algos | multinomial-nb | istella | accuracy (higher is better) | - | 0.853620 | sklearn-cpu 0.853620 |
| algos | multinomial-nb | istella | logloss (lower is better) | - | 3.628565 | sklearn-cpu 3.087499 |
| algos | multinomial-nb | taxi | accuracy (higher is better) | - | 0.723160 | sklearn-cpu 0.723160 |
| algos | multinomial-nb | taxi | logloss (lower is better) | - | 0.590725 | sklearn-cpu 0.590725 |
| algos | multinomial-nb | text | accuracy (higher is better) | - | 0.983067 | sklearn-cpu 0.983067 |
| algos | multinomial-nb | text | logloss (lower is better) | - | 0.559529 | sklearn-cpu 0.557319 |
| algos | nadam | synthetic | optimizer_update_error_vs_torch_fp32 | - | 1.131e-05 | torch-eager-fp32 0.000000; torch-compile-fp32 5.621e-05 |
| algos | nadam | synthetic | rel_fro_vs_torch_eager_fp32 | - | 2.945e-08 | torch-eager-fp32 0.000000; torch-compile-fp32 1.464e-07 |
| algos | nearest-centroid | istella | accuracy (higher is better) | - | 0.852610 | sklearn-cpu 0.852610 |
| algos | nearest-centroid | istella | logloss (lower is better) | - | 4.299485 | sklearn-cpu 4.117692 |
| algos | nearest-centroid | taxi | accuracy (higher is better) | - | 0.666700 | sklearn-cpu 0.666750 |
| algos | nearest-centroid | taxi | logloss (lower is better) | - | 0.781905 | sklearn-cpu 0.781690 |
| algos | normalizer | istella | max_abs_diff_vs_sklearn | - | 3.576e-07 | sklearn-cpu - |
| algos | normalizer | istella | rel_diff_vs_sklearn | - | 5.647e-08 | sklearn-cpu - |
| algos | normalizer | taxi | max_abs_diff_vs_sklearn | - | 1.192e-07 | sklearn-cpu - |
| algos | normalizer | taxi | rel_diff_vs_sklearn | - | 3.043e-08 | sklearn-cpu - |
| algos | onehot | istella | max_abs_diff_vs_sklearn | - | 0.000000 | sklearn-cpu - |
| algos | onehot | istella | rel_diff_vs_sklearn | - | 0.000000 | sklearn-cpu - |
| algos | onehot | taxi | max_abs_diff_vs_sklearn | - | 0.000000 | sklearn-cpu - |
| algos | onehot | taxi | rel_diff_vs_sklearn | - | 0.000000 | sklearn-cpu - |
| algos | optimized-theta | synthetic | forecast_rmse (lower is better) | - | 1.438855 | statsforecast-cpu 1.437815 |
| algos | optimized-theta | taxi-hourly | forecast_rmse (lower is better) | - | 49.150860 | statsforecast-cpu 49.356608 |
| algos | ordinal | istella | max_abs_diff_vs_sklearn | - | 0.000000 | sklearn-cpu - |
| algos | ordinal | istella | rel_diff_vs_sklearn | - | 0.000000 | sklearn-cpu - |
| algos | ordinal | taxi | max_abs_diff_vs_sklearn | - | 0.000000 | sklearn-cpu - |
| algos | ordinal | taxi | rel_diff_vs_sklearn | - | 0.000000 | sklearn-cpu - |
| algos | pagerank | istella | sum | - | 1.000000 | networkx-cpu 1.000000 |
| algos | pagerank | taxi | sum | - | 1.000000 | networkx-cpu 1.000000 |
| algos | permutation-shap | istella | rel_error_vs_exact | - | 6.68e-08 | shap-cpu 3.692e-10 |
| algos | permutation-shap | taxi | rel_error_vs_exact | - | 1.054e-07 | shap-cpu 1.218e-15 |
| algos | poly-features | istella | max_abs_diff_vs_sklearn | - | 0.000000 | sklearn-cpu - |
| algos | poly-features | istella | rel_diff_vs_sklearn | - | 0.000000 | sklearn-cpu - |
| algos | poly-features | taxi | max_abs_diff_vs_sklearn | - | 0.000000 | sklearn-cpu - |
| algos | poly-features | taxi | rel_diff_vs_sklearn | - | 0.000000 | sklearn-cpu - |
| algos | power-transformer | istella | max_abs_diff_vs_sklearn | - | 5.575023 | sklearn-cpu - |
| algos | power-transformer | istella | rel_diff_vs_sklearn | - | 0.105054 | sklearn-cpu - |
| algos | power-transformer | taxi | max_abs_diff_vs_sklearn | - | 0.064570 | sklearn-cpu - |
| algos | power-transformer | taxi | rel_diff_vs_sklearn | - | 0.0008076 | sklearn-cpu - |
| algos | qr | istella | relative_gram_difference | - | 1.485e-07 | torch-gpu 0.0001695 |
| algos | qr | taxi | relative_gram_difference | - | 1.714e-07 | torch-gpu 9.54e-07 |
| algos | quantile-transformer | istella | max_abs_diff_vs_sklearn | - | 5.96e-08 | - |
| algos | quantile-transformer | istella | rel_diff_vs_sklearn | - | 2.764e-08 | - |
| algos | quantile-transformer | taxi | max_abs_diff_vs_sklearn | - | 5.96e-08 | - |
| algos | quantile-transformer | taxi | rel_diff_vs_sklearn | - | 2.294e-08 | - |
| algos | quantile | istella | r2 (higher is better) | - | -0.039998 | sklearn-cpu -0.044780 |
| algos | quantile | istella | rmse (lower is better) | - | 0.851877 | sklearn-cpu 0.853833 |
| algos | quantile | taxi | r2 (higher is better) | - | 0.899596 | sklearn-cpu 0.899678 |
| algos | quantile | taxi | rmse (lower is better) | - | 5.046749 | sklearn-cpu 5.044706 |
| algos | randomized-svd | istella | relative_reconstruction_error (lower is better) | - | 0.0002359 | torch-gpu 0.0002359 |
| algos | randomized-svd | taxi | relative_reconstruction_error (lower is better) | - | 0.027197 | torch-gpu 0.027197 |
| algos | resnet-block | synthetic | relative_error_vs_own_host | - | 0.000000 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | resnet-block | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.834465; torch-eager-bf16 28679.370880; torch-compile-bf16 18454.670906 |
| algos | resnet-block | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 1.645e-07; torch-eager-bf16 0.004232; torch-compile-bf16 0.003485 |
| algos | rfe | istella | jaccard_vs_exact | - | 0.774194 | sklearn-cpu - |
| algos | rfe | istella | jaccard_vs_sklearn | - | 0.848739 | sklearn-cpu 1.000000 |
| algos | rfe | istella | n_selected | - | 110 | sklearn-cpu 110 |
| algos | rfe | taxi | jaccard_vs_exact | - | 1.000000 | sklearn-cpu - |
| algos | rfe | taxi | jaccard_vs_sklearn | - | 1.000000 | sklearn-cpu 1.000000 |
| algos | rfe | taxi | n_selected | - | 5 | sklearn-cpu 5 |
| algos | ridge-cv | istella | r2 (higher is better) | - | 0.328684 | sklearn-cpu 0.328683 |
| algos | ridge-cv | istella | rmse (lower is better) | - | 0.684422 | sklearn-cpu 0.684423 |
| algos | ridge-cv | taxi | r2 (higher is better) | - | 0.908981 | sklearn-cpu 0.908983 |
| algos | ridge-cv | taxi | rmse (lower is better) | - | 4.805109 | sklearn-cpu 4.805057 |
| algos | rmsprop | synthetic | optimizer_update_error_vs_torch_fp32 | - | 2.228e-06 | torch-eager-fp32 0.000000; torch-compile-fp32 0.000000 |
| algos | rmsprop | synthetic | rel_fro_vs_torch_eager_fp32 | - | 3.829e-08 | torch-eager-fp32 0.000000; torch-compile-fp32 0.000000 |
| algos | rnn-clf | synthetic | accuracy (higher is better) | - | 0.967828 | torch-eager-fp32 0.953559; torch-compile-fp32 0.953559; torch-eager-bf16 0.953559; torch-compile-bf16 0.953559 |
| algos | rnn-clf | synthetic | logloss (lower is better) | - | 0.079029 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | rnn-clf | taxi-hourly | accuracy (higher is better) | - | 0.862684 | torch-eager-fp32 0.868056; torch-compile-fp32 0.868056; torch-eager-bf16 0.867947; torch-compile-bf16 0.867947 |
| algos | rnn-clf | taxi-hourly | logloss (lower is better) | - | 0.313008 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | rnn-reg | synthetic | r2 (higher is better) | - | 0.978602 | torch-eager-fp32 0.977348; torch-compile-fp32 0.977348; torch-eager-bf16 0.977371; torch-compile-bf16 0.977371 |
| algos | rnn-reg | synthetic | rmse (lower is better) | - | 0.169476 | torch-eager-fp32 0.174374; torch-compile-fp32 0.174374; torch-eager-bf16 0.174283; torch-compile-bf16 0.174283 |
| algos | rnn-reg | taxi-hourly | r2 (higher is better) | - | 0.743904 | torch-eager-fp32 0.738796; torch-compile-fp32 0.738796; torch-eager-bf16 0.739017; torch-compile-bf16 0.739017 |
| algos | rnn-reg | taxi-hourly | rmse (lower is better) | - | 0.548825 | torch-eager-fp32 0.554271; torch-compile-fp32 0.554271; torch-eager-bf16 0.554037; torch-compile-bf16 0.554037 |
| algos | robust-scaler | istella | max_abs_diff_vs_sklearn | - | 0.0001221 | sklearn-cpu - |
| algos | robust-scaler | istella | rel_diff_vs_sklearn | - | 7.339e-09 | sklearn-cpu - |
| algos | robust-scaler | taxi | max_abs_diff_vs_sklearn | - | 0.000000 | sklearn-cpu - |
| algos | robust-scaler | taxi | rel_diff_vs_sklearn | - | 0.000000 | sklearn-cpu - |
| algos | select-chi2 | istella | jaccard_vs_sklearn | - | 1.000000 | sklearn-cpu 1.000000 |
| algos | select-chi2 | istella | n_selected | - | 110 | sklearn-cpu 110 |
| algos | select-chi2 | taxi | jaccard_vs_sklearn | - | 1.000000 | sklearn-cpu 1.000000 |
| algos | select-chi2 | taxi | n_selected | - | 5 | sklearn-cpu 5 |
| algos | select-d | synthetic | relative_error_vs_own_host | - | 0.000000 | statsmodels-cpu - |
| algos | select-d | synthetic | d_agreement_vs_statsmodels | - | - | statsmodels-cpu 1.000000 |
| algos | select-d | taxi-hourly | relative_error_vs_own_host | - | 0.000000 | statsmodels-cpu - |
| algos | select-d | taxi-hourly | d_agreement_vs_statsmodels | - | - | statsmodels-cpu 1.000000 |
| algos | select-f-classif | istella | jaccard_vs_sklearn | - | 1.000000 | sklearn-cpu 1.000000 |
| algos | select-f-classif | istella | n_selected | - | 110 | sklearn-cpu 110 |
| algos | select-f-classif | taxi | jaccard_vs_sklearn | - | 1.000000 | sklearn-cpu 1.000000 |
| algos | select-f-classif | taxi | n_selected | - | 5 | sklearn-cpu 5 |
| algos | select-f-regression | istella | jaccard_vs_sklearn | - | 1.000000 | sklearn-cpu 1.000000 |
| algos | select-f-regression | istella | n_selected | - | 110 | sklearn-cpu 110 |
| algos | select-f-regression | taxi | jaccard_vs_sklearn | - | 1.000000 | sklearn-cpu 1.000000 |
| algos | select-f-regression | taxi | n_selected | - | 5 | sklearn-cpu 5 |
| algos | select-mutual-info-reg | istella | jaccard_vs_sklearn | - | 0.929825 | sklearn-cpu 1.000000 |
| algos | select-mutual-info-reg | istella | n_selected | - | 110 | sklearn-cpu 110 |
| algos | select-mutual-info-reg | taxi | jaccard_vs_sklearn | - | 1.000000 | sklearn-cpu 1.000000 |
| algos | select-mutual-info-reg | taxi | n_selected | - | 5 | sklearn-cpu 5 |
| algos | select-mutual-info | istella | jaccard_vs_sklearn | - | 0.929825 | sklearn-cpu 1.000000 |
| algos | select-mutual-info | istella | n_selected | - | 110 | sklearn-cpu 110 |
| algos | select-mutual-info | taxi | jaccard_vs_sklearn | - | 0.428571 | sklearn-cpu 1.000000 |
| algos | select-mutual-info | taxi | n_selected | - | 5 | sklearn-cpu 5 |
| algos | select-r-regression | istella | jaccard_vs_sklearn | - | 1.000000 | sklearn-cpu 1.000000 |
| algos | select-r-regression | istella | n_selected | - | 110 | sklearn-cpu 110 |
| algos | select-r-regression | taxi | jaccard_vs_sklearn | - | 1.000000 | sklearn-cpu 1.000000 |
| algos | select-r-regression | taxi | n_selected | - | 5 | sklearn-cpu 5 |
| algos | sgd-clf | istella | accuracy (higher is better) | - | 0.920330 | - |
| algos | sgd-clf | taxi | accuracy (higher is better) | - | 0.755330 | - |
| algos | sgd | synthetic | optimizer_update_error_vs_torch_fp32 | - | 0.000000 | torch-eager-fp32 0.000000; torch-compile-fp32 4.764e-06; torch-fused-fp32 4.764e-06 |
| algos | sgd | synthetic | rel_fro_vs_torch_eager_fp32 | - | 0.000000 | torch-eager-fp32 0.000000; torch-compile-fp32 6.772e-10; torch-fused-fp32 6.772e-10 |
| algos | sparse-coder | istella | max_abs_diff_vs_sklearn | - | 6.104e-05 | sklearn-cpu - |
| algos | sparse-coder | istella | rel_diff_vs_sklearn | - | 9.356e-08 | sklearn-cpu - |
| algos | sparse-coder | taxi | max_abs_diff_vs_sklearn | - | 4.292e-06 | sklearn-cpu - |
| algos | sparse-coder | taxi | rel_diff_vs_sklearn | - | 8.246e-08 | sklearn-cpu - |
| algos | spline | istella | max_abs_diff_vs_sklearn | - | 1.788e-07 | sklearn-cpu - |
| algos | spline | istella | rel_diff_vs_sklearn | - | 6.226e-08 | sklearn-cpu - |
| algos | spline | taxi | max_abs_diff_vs_sklearn | - | 1.788e-07 | sklearn-cpu - |
| algos | spline | taxi | rel_diff_vs_sklearn | - | 5.888e-08 | sklearn-cpu - |
| algos | standard-scaler | istella | max_abs_diff_vs_sklearn | - | 0.002426 | sklearn-cpu - |
| algos | standard-scaler | istella | rel_diff_vs_sklearn | - | 3.59e-06 | sklearn-cpu - |
| algos | standard-scaler | taxi | max_abs_diff_vs_sklearn | - | 0.000103 | sklearn-cpu - |
| algos | standard-scaler | taxi | rel_diff_vs_sklearn | - | 3.846e-06 | sklearn-cpu - |
| algos | svd | istella | max_rel_singular_value_error | - | 662.224259 | torch-gpu 2.707e+08 |
| algos | svd | istella | relative_reconstruction_error_100k_rows | - | 3.379e-05 | torch-gpu 0.026534 |
| algos | svd | taxi | max_rel_singular_value_error | - | 7.036e-07 | torch-gpu 4.409e-05 |
| algos | svd | taxi | relative_reconstruction_error_100k_rows | - | 1.18e-06 | torch-gpu 0.003043 |
| algos | svgp | istella | r2 (higher is better) | - | -0.106016 | gpytorch-gpu -0.106040; gpytorch-cpu -0.106040 |
| algos | svgp | istella | rmse (lower is better) | - | 0.878373 | gpytorch-gpu 0.878383; gpytorch-cpu 0.878383 |
| algos | svgp | taxi | r2 (higher is better) | - | -0.194982 | gpytorch-gpu -0.209454; gpytorch-cpu -0.209532 |
| algos | svgp | taxi | rmse (lower is better) | - | 17.723482 | gpytorch-gpu 17.830481; gpytorch-cpu 17.831057 |
| algos | target-encoder | istella | max_abs_diff_vs_sklearn | - | 1.074e-08 | sklearn-cpu - |
| algos | target-encoder | istella | rel_diff_vs_sklearn | - | 2.361e-08 | sklearn-cpu - |
| algos | target-encoder | taxi | max_abs_diff_vs_sklearn | - | 2.965e-08 | sklearn-cpu - |
| algos | target-encoder | taxi | rel_diff_vs_sklearn | - | 1.581e-08 | sklearn-cpu - |
| algos | theta | synthetic | forecast_rmse (lower is better) | - | 1.436610 | statsforecast-cpu 1.436557; statsmodels-cpu 1.434862 |
| algos | theta | taxi-hourly | forecast_rmse (lower is better) | - | 49.020604 | statsforecast-cpu 49.253901; statsmodels-cpu 49.311757 |
| algos | tree-shap | istella | max_additivity_error | - | 1.175e-06 | - |
| algos | tree-shap | taxi | max_additivity_error | - | 3.858e-05 | - |
| algos | tsne | istella | trustworthiness_k15 (higher is better, 1 at most) | - | 0.992124 | sklearn-cpu 0.992170 |
| algos | tsne | taxi | trustworthiness_k15 (higher is better, 1 at most) | - | 0.998921 | sklearn-cpu 0.998823 |
| algos | variance-threshold | istella | max_abs_diff_vs_sklearn | - | 0.000000 | sklearn-cpu - |
| algos | variance-threshold | istella | rel_diff_vs_sklearn | - | 0.000000 | sklearn-cpu - |
| algos | variance-threshold | taxi | max_abs_diff_vs_sklearn | - | 0.000000 | sklearn-cpu - |
| algos | variance-threshold | taxi | rel_diff_vs_sklearn | - | 0.000000 | sklearn-cpu - |
| classical | dbscan | istella | n_clusters | - | 40131 | sklearn-cpu 40131 |
| classical | dbscan | istella | noise_fraction | - | 0.219391 | sklearn-cpu 0.219391 |
| classical | dbscan | istella | rows | - | 1000000 | sklearn-cpu 1000000 |
| classical | dbscan | istella | ari_vs_ours (1 is our partition exactly) | - | - | sklearn-cpu 1.000000 |
| classical | dbscan | istella | noise_agreement_vs_ours | - | - | sklearn-cpu 1.000000 |
| classical | dbscan | taxi | n_clusters | - | 36 | sklearn-cpu - |
| classical | dbscan | taxi | noise_fraction | - | 0.000174 | sklearn-cpu - |
| classical | dbscan | taxi | rows | - | 1000000 | sklearn-cpu - |
| classical | hdbscan | istella | n_clusters | - | 47 | sklearn-cpu 52 |
| classical | hdbscan | istella | noise_fraction | - | 0.253810 | sklearn-cpu 0.252570 |
| classical | hdbscan | istella | rows | - | 100000 | sklearn-cpu 100000 |
| classical | hdbscan | taxi | n_clusters | - | 161 | sklearn-cpu 161 |
| classical | hdbscan | taxi | noise_fraction | - | 0.140550 | sklearn-cpu 0.134630 |
| classical | hdbscan | taxi | rows | - | 100000 | sklearn-cpu 100000 |
| classical | kde | istella | mean_log_likelihood (higher is better) | - | -222.270583 | sklearn-cpu -226.977407 |
| classical | kde | istella | rows_without_density | - | 0 | sklearn-cpu 0 |
| classical | kde | taxi | mean_log_likelihood (higher is better) | - | -14.826436 | sklearn-cpu -14.826437 |
| classical | kde | taxi | rows_without_density | - | 0 | sklearn-cpu 0 |
| classical | kmeans | istella | inertia (lower is better) | - | 6.051e+17 | hipvs-gpu 5.959e+17 |
| classical | kmeans | istella | inertia_over_ours | - | 1.000000 | hipvs-gpu 0.984774 |
| classical | kmeans | istella | n_iter | - | 33 | hipvs-gpu 301 |
| classical | kmeans | istella | n_iter_over_ours | - | 1.000000 | hipvs-gpu 9.121212 |
| classical | kmeans | taxi | inertia (lower is better) | - | 3.093e+08 | hipvs-gpu 3.06e+08 |
| classical | kmeans | taxi | inertia_over_ours | - | 1.000000 | hipvs-gpu 0.989329 |
| classical | kmeans | taxi | n_iter | - | 91 | hipvs-gpu 79 |
| classical | kmeans | taxi | n_iter_over_ours | - | 1.000000 | hipvs-gpu 0.868132 |
| classical | knn | istella | recall_at_k (higher is better) | - | 0.976250 | hipvs-gpu 0.976133 |
| classical | knn | istella | rows_with_repeated_ids | - | 0 | hipvs-gpu 0 |
| classical | knn | taxi | recall_at_k (higher is better) | - | 0.999754 | hipvs-gpu 0.999777 |
| classical | knn | taxi | rows_with_repeated_ids | - | 0 | hipvs-gpu 0 |
| classical | ols | istella | r2 (higher is better) | - | 0.332506 | torch-gpu nan; torch-gpu-eigh 0.151604; sklearn-cpu 0.001881 |
| classical | ols | istella | rmse (lower is better) | - | 0.681740 | torch-gpu nan; torch-gpu-eigh 0.768589; sklearn-cpu 0.833655 |
| classical | ols | taxi | r2 (higher is better) | - | 0.908837 | torch-gpu 0.908840; torch-gpu-eigh 0.908822; sklearn-cpu 0.724850 |
| classical | ols | taxi | rmse (lower is better) | - | 4.696477 | torch-gpu 4.696376; torch-gpu-eigh 4.696849; sklearn-cpu 8.159187 |
| classical | pca | istella | explained_variance_ratio_sum (higher is better) | - | 1.000000 | torch-gpu 1.000000; sklearn-cpu 1.000000 |
| classical | pca | taxi | explained_variance_ratio_sum (higher is better) | - | 0.999997 | torch-gpu 0.999997; sklearn-cpu 0.999995 |
| classical | svc | istella | accuracy (higher is better) | - | 0.922200 | sklearn-cpu 0.922200 |
| classical | svc | istella | n_support | - | 2400 | sklearn-cpu 2400 |
| classical | svc | taxi | accuracy (higher is better) | - | 0.767500 | sklearn-cpu 0.767500 |
| classical | svc | taxi | n_support | - | 5527 | sklearn-cpu 5675 |
| classical2 | elasticnet | istella | r2 (higher is better) | - | 0.260922 | - |
| classical2 | elasticnet | istella | rmse (lower is better) | - | 0.718134 | - |
| classical2 | elasticnet | taxi | r2 (higher is better) | - | 0.907378 | - |
| classical2 | elasticnet | taxi | rmse (lower is better) | - | 4.847225 | - |
| classical2 | gmm | istella | bic (lower is better) | - | -3.851e+07 | - |
| classical2 | gmm | istella | mean_log_likelihood (higher is better) | - | 200.794500 | - |
| classical2 | gmm | istella | n_iter | - | 24 | - |
| classical2 | gmm | taxi | bic (lower is better) | - | -3.668e+06 | - |
| classical2 | gmm | taxi | mean_log_likelihood (higher is better) | - | 12.807640 | - |
| classical2 | gmm | taxi | n_iter | - | 29 | - |
| classical2 | ivf | istella | recall_at_k (higher is better) | - | 0.999925 | faiss-cpu - |
| classical2 | ivf | istella | rows_with_repeated_ids | - | 0 | faiss-cpu - |
| classical2 | ivf | taxi | recall_at_k (higher is better) | - | 0.999650 | faiss-cpu - |
| classical2 | ivf | taxi | rows_with_repeated_ids | - | 0 | faiss-cpu - |
| classical2 | lasso | istella | r2 (higher is better) | - | 0.310837 | - |
| classical2 | lasso | istella | rmse (lower is better) | - | 0.693460 | - |
| classical2 | lasso | taxi | r2 (higher is better) | - | 0.908995 | - |
| classical2 | lasso | taxi | rmse (lower is better) | - | 4.804744 | - |
| classical2 | logreg | istella | accuracy (higher is better) | - | 0.924590 | - |
| classical2 | logreg | istella | logloss (lower is better) | - | 0.181249 | - |
| classical2 | logreg | istella | nonfinite_proba_rows | - | 0 | - |
| classical2 | logreg | taxi | accuracy (higher is better) | - | 0.763350 | - |
| classical2 | logreg | taxi | logloss (lower is better) | - | 0.538985 | - |
| classical2 | logreg | taxi | nonfinite_proba_rows | - | 0 | - |
| classical2 | ridge | istella | r2 (higher is better) | - | 0.328674 | sklearn-cpu 0.328676 |
| classical2 | ridge | istella | rmse (lower is better) | - | 0.684427 | sklearn-cpu 0.684426 |
| classical2 | ridge | taxi | r2 (higher is better) | - | 0.908983 | sklearn-cpu 0.908983 |
| classical2 | ridge | taxi | rmse (lower is better) | - | 4.805050 | sklearn-cpu 4.805056 |
| classical2 | tsvd | istella | explained_variance_ratio_sum (higher is better) | - | 1.000000 | sklearn-cpu 1.000000 |
| classical2 | tsvd | istella | relative_reconstruction_error (lower is better) | - | 0.0001314 | sklearn-cpu 0.000122 |
| classical2 | tsvd | taxi | explained_variance_ratio_sum (higher is better) | - | 0.999965 | sklearn-cpu 0.999965 |
| classical2 | tsvd | taxi | relative_reconstruction_error (lower is better) | - | 0.003257 | sklearn-cpu 0.003257 |
| neural | gemm-bf16 | gaussian | max_rel_err_vs_fp64 (lower is better) | - | 1.155e-07 | torch-eager-bf16 0.002759; torch-compile-bf16 0.002759 |
| neural | gemm-bf16 | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | torch-eager-bf16 0.999023; torch-compile-bf16 0.999023 |
| neural | gemm-bf16 | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | torch-eager-bf16 0.002759; torch-compile-bf16 0.002759 |
| neural | gemm | gaussian | max_rel_err_vs_fp64 (lower is better) | - | 2.399e-07 | torch-eager-fp32 2.802e-06; torch-compile-fp32 2.802e-06; torch-eager-bf16 0.003752; torch-compile-bf16 0.003752 |
| neural | gemm | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | torch-eager-fp32 0.001038; torch-compile-fp32 0.001038; torch-eager-bf16 1.358337; torch-compile-bf16 1.358337 |
| neural | gemm | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | torch-eager-fp32 2.866e-06; torch-compile-fp32 2.866e-06; torch-eager-bf16 0.003751; torch-compile-bf16 0.003751 |
| neural | lm-forward | bytes | mean_nll (lower is better) | - | 9.018733 | torch-eager-fp32 9.018733; torch-eager-bf16 9.018647; torch-compile-fp32 9.018733; torch-compile-bf16 9.018664 |
| neural | lm-train-step | bytes | loss_first_step (same init and batches on every arm) | - | 9.018733 | torch-eager-fp32 9.018732; torch-compile-fp32 9.018733; torch-eager-bf16 9.018646; torch-compile-bf16 9.018663 |
| neural | lm-train-step | bytes | loss_last_step (same init and batches on every arm) | - | 8.418446 | torch-eager-fp32 8.418447; torch-compile-fp32 8.418446; torch-eager-bf16 8.416107; torch-compile-bf16 8.415815 |
| neural | lm-train-step | bytes | steps | - | 2 | torch-eager-fp32 2; torch-compile-fp32 2; torch-eager-bf16 2; torch-compile-bf16 2 |
| neural | mlp-train-step | gaussian | loss_first_step (same init and batches on every arm) | - | 1.160401 | torch-eager-fp32 1.160401; torch-compile-fp32 1.160401; torch-eager-bf16 1.160498; torch-compile-bf16 1.160367 |
| neural | mlp-train-step | gaussian | loss_last_step (same init and batches on every arm) | - | 1.123361 | torch-eager-fp32 1.123361; torch-compile-fp32 1.123361; torch-eager-bf16 1.123461; torch-compile-bf16 1.123438 |
| neural | mlp-train-step | gaussian | steps | - | 2 | torch-eager-fp32 2; torch-compile-fp32 2; torch-eager-bf16 2; torch-compile-bf16 2 |
| neural | samba-forward | bytes | mean_nll (lower is better) | - | 5.635910 | torch-eager-fp32 5.635910; torch-compile-fp32 5.635910; torch-eager-bf16 5.635892; torch-compile-bf16 5.635953 |
| neural | samba-train-step | bytes | loss_first_step (same init and batches on every arm) | - | 5.635910 | torch-eager-fp32 5.635909; torch-compile-fp32 -; torch-eager-bf16 5.635892; torch-compile-bf16 - |
| neural | samba-train-step | bytes | loss_last_step (same init and batches on every arm) | - | 4.833934 | torch-eager-fp32 4.833934; torch-compile-fp32 -; torch-eager-bf16 4.834043; torch-compile-bf16 - |
| neural | samba-train-step | bytes | steps | - | 2 | torch-eager-fp32 2; torch-compile-fp32 -; torch-eager-bf16 2; torch-compile-bf16 - |
| trees | gbdt-categorical | taxi | auc (higher is better) | - | 0.631297 | xgboost-gpu -; catboost-cpu 0.628772; xgboost-cpu 0.631473; lightgbm-cpu 0.632665 |
| trees | gbdt-categorical | taxi | logloss (lower is better) | - | 0.528285 | xgboost-gpu -; catboost-cpu 0.528923; xgboost-cpu 0.528686; lightgbm-cpu 0.528094 |
| trees | gbdt-depthwise | istella | auc (higher is better) | - | 0.983304 | - |
| trees | gbdt-depthwise | istella | logloss (lower is better) | - | 0.156072 | - |
| trees | gbdt-depthwise | taxi | auc (higher is better) | - | 0.632205 | - |
| trees | gbdt-depthwise | taxi | logloss (lower is better) | - | 0.527838 | - |
| trees | gbdt-lossguide | istella | auc (higher is better) | - | 0.983586 | xgboost-gpu -; catboost-cpu 0.983135; xgboost-cpu 0.983622; lightgbm-cpu 0.983778 |
| trees | gbdt-lossguide | istella | logloss (lower is better) | - | 0.150066 | xgboost-gpu -; catboost-cpu 0.157692; xgboost-cpu 0.149263; lightgbm-cpu 0.149653 |
| trees | gbdt-lossguide | taxi | auc (higher is better) | - | 0.632096 | xgboost-gpu -; catboost-cpu 0.632578; xgboost-cpu 0.630968; lightgbm-cpu 0.632243 |
| trees | gbdt-lossguide | taxi | logloss (lower is better) | - | 0.528044 | xgboost-gpu -; catboost-cpu 0.527851; xgboost-cpu 0.528677; lightgbm-cpu 0.528067 |
| trees | gbdt-multiclass | istella | accuracy (higher is better) | - | 0.907556 | xgboost-gpu -; catboost-cpu 0.907768; xgboost-cpu 0.910140; lightgbm-cpu 0.910058 |
| trees | gbdt-multiclass | istella | mlogloss (lower is better) | - | 0.258413 | xgboost-gpu -; catboost-cpu 0.258286; xgboost-cpu 0.246803; lightgbm-cpu 0.245916 |
| trees | gbdt-multiclass | taxi | accuracy (higher is better) | - | 0.599380 | xgboost-gpu -; catboost-cpu 0.599150; xgboost-cpu 0.601128; lightgbm-cpu 0.601580 |
| trees | gbdt-multiclass | taxi | mlogloss (lower is better) | - | 1.012595 | xgboost-gpu -; catboost-cpu 1.012734; xgboost-cpu 1.005128; lightgbm-cpu 1.004282 |
| trees | gbdt-ordered | istella | auc (higher is better) | - | 0.979444 | catboost-cpu 0.979221 |
| trees | gbdt-ordered | istella | logloss (lower is better) | - | 0.190928 | catboost-cpu 0.192114 |
| trees | gbdt-ordered | taxi | auc (higher is better) | - | 0.629203 | catboost-cpu 0.628918 |
| trees | gbdt-ordered | taxi | logloss (lower is better) | - | 0.529007 | catboost-cpu 0.529083 |
| trees | gbdt-rank-pairlogit | istella | map (higher is better) | - | 0.854545 | xgboost-gpu -; catboost-cpu 0.846328; xgboost-cpu 0.872796 |
| trees | gbdt-rank-pairlogit | istella | ndcg10 (higher is better) | - | 0.719953 | xgboost-gpu -; catboost-cpu 0.713361; xgboost-cpu 0.738397 |
| trees | gbdt-rank-pairlogit | istella | ndcg5 (higher is better) | - | 0.650400 | xgboost-gpu -; catboost-cpu 0.643611; xgboost-cpu 0.670093 |
| trees | gbdt-symmetric | istella | auc (higher is better) | - | 0.980129 | catboost-cpu 0.979899 |
| trees | gbdt-symmetric | istella | logloss (lower is better) | - | 0.186686 | catboost-cpu 0.188093 |
| trees | gbdt-symmetric | taxi | auc (higher is better) | - | 0.630436 | catboost-cpu 0.630269 |
| trees | gbdt-symmetric | taxi | logloss (lower is better) | - | 0.528554 | catboost-cpu 0.528650 |

## Inference at a glance

Batch prediction, each arm with its own fitted model from the same race; medians in ms. `FAST = IDENTICAL bits` compares our two tiers' predictions on the same rows. Our CPU is never raced or reported.

| family | lane | dataset | batch | rows | ours FAST ms | ours IDENTICAL ms | FAST = IDENTICAL bits | opponents |
|---|---|---|---|---|---|---|---|---|
| algos | avgpool1d | synthetic | Xq | - | - | 0.1 | - | torch-eager-fp32 0.1 ms (IDENTICAL/arm 1.299); torch-compile-fp32 0.1 ms (IDENTICAL/arm 0.759); torch-eager-bf16 0.1 ms (IDENTICAL/arm 1.084); torch-compile-bf16 0.1 ms (IDENTICAL/arm 0.730) |
| algos | avgpool2d | synthetic | Xq | - | - | 0.2 | - | torch-eager-fp32 0.1 ms (IDENTICAL/arm 1.420); torch-compile-fp32 0.2 ms (IDENTICAL/arm 0.868); torch-eager-bf16 0.1 ms (IDENTICAL/arm 1.378); torch-compile-bf16 0.2 ms (IDENTICAL/arm 0.940) |
| algos | batchnorm1d | synthetic | Xq | - | - | 1.3 | - | torch-eager-fp32 0.1 ms (IDENTICAL/arm 10.152); torch-compile-fp32 0.2 ms (IDENTICAL/arm 7.351); torch-eager-bf16 0.1 ms (IDENTICAL/arm 9.751); torch-compile-bf16 0.2 ms (IDENTICAL/arm 7.249) |
| algos | batchnorm2d | synthetic | Xq | - | - | 1.3 | - | torch-eager-fp32 0.2 ms (IDENTICAL/arm 6.623); torch-compile-fp32 0.2 ms (IDENTICAL/arm 8.063); torch-eager-bf16 0.2 ms (IDENTICAL/arm 6.211); torch-compile-bf16 0.2 ms (IDENTICAL/arm 7.883) |
| algos | conv1d | synthetic | Xq | - | - | 1.7 | - | torch-eager-fp32 0.7 ms (IDENTICAL/arm 2.537); torch-compile-fp32 0.9 ms (IDENTICAL/arm 2.039); torch-eager-bf16 0.4 ms (IDENTICAL/arm 4.613); torch-compile-bf16 0.5 ms (IDENTICAL/arm 3.568) |
| algos | conv2d | synthetic | Xq | - | - | 1.1 | - | torch-eager-fp32 0.3 ms (IDENTICAL/arm 3.868); torch-compile-fp32 0.4 ms (IDENTICAL/arm 3.135); torch-eager-bf16 0.3 ms (IDENTICAL/arm 4.307); torch-compile-bf16 0.3 ms (IDENTICAL/arm 3.308) |
| algos | dropout2d | synthetic | Xq | - | - | 0.1 | - | torch-eager-fp32 0.1 ms (IDENTICAL/arm 0.827); torch-compile-fp32 0.1 ms (IDENTICAL/arm 0.660) |
| algos | embedding | synthetic | Xq | - | - | 0.3 | - | torch-eager-fp32 0.1 ms (IDENTICAL/arm 3.086); torch-compile-fp32 0.2 ms (IDENTICAL/arm 1.758) |
| algos | global-avgpool | synthetic | Xq | - | - | 0.1 | - | torch-eager-fp32 0.0 ms (IDENTICAL/arm 1.213); torch-compile-fp32 0.1 ms (IDENTICAL/arm 0.452); torch-eager-bf16 0.0 ms (IDENTICAL/arm 1.260); torch-compile-bf16 0.1 ms (IDENTICAL/arm 0.524) |
| algos | global-maxpool | synthetic | Xq | - | - | 0.1 | - | torch-eager-fp32 0.1 ms (IDENTICAL/arm 0.643); torch-compile-fp32 0.1 ms (IDENTICAL/arm 0.486); torch-eager-bf16 0.1 ms (IDENTICAL/arm 0.570); torch-compile-bf16 0.1 ms (IDENTICAL/arm 0.535) |
| algos | layernorm | synthetic | Xq | - | - | 0.1 | - | torch-eager-fp32 0.1 ms (IDENTICAL/arm 1.826); torch-compile-fp32 0.1 ms (IDENTICAL/arm 1.149); torch-eager-bf16 0.1 ms (IDENTICAL/arm 1.806); torch-compile-bf16 0.1 ms (IDENTICAL/arm 1.081) |
| algos | maxpool1d | synthetic | Xq | - | - | 0.1 | - | torch-eager-fp32 0.1 ms (IDENTICAL/arm 0.991); torch-compile-fp32 0.1 ms (IDENTICAL/arm 0.700); torch-eager-bf16 0.1 ms (IDENTICAL/arm 0.926); torch-compile-bf16 0.1 ms (IDENTICAL/arm 0.735) |
| algos | maxpool2d | synthetic | Xq | - | - | 0.2 | - | torch-eager-fp32 0.1 ms (IDENTICAL/arm 1.211); torch-compile-fp32 0.2 ms (IDENTICAL/arm 0.986); torch-eager-bf16 0.1 ms (IDENTICAL/arm 1.177); torch-compile-bf16 0.2 ms (IDENTICAL/arm 0.993) |
| classical | kmeans | istella | Xq | 500000 | - | 18.1 | - | hipvs-gpu - ms (IDENTICAL/arm -) |
| classical | kmeans | taxi | Xq | 500000 | - | 1.7 | - | hipvs-gpu - ms (IDENTICAL/arm -) |


## Neural headline: ours IDENTICAL against torch bf16

What customers run is torch in bf16. The headline divides our IDENTICAL median by torch's fastest bf16 arm (eager or compile, the lower stored median); the fp32 twin (torch's fastest fp32 arm) is the second column. Per-arm ratios stay in each race's table below.

| lane | dataset | ours IDENTICAL ms | torch bf16 ms (fastest arm) | ours / torch bf16 | torch fp32 twin ms (fastest arm) | ours / torch fp32 | note |
|---|---|---|---|---|---|---|---|
| adafactor | synthetic | 8.2 | - | - | 7.2 (torch-eager-fp32) | 1.130 | no torch bf16 arm on this lane |
| adagrad | synthetic | 5.2 | - | - | 2.6 (torch-compile-fp32) | 2.037 | no torch bf16 arm on this lane |
| adam | synthetic | 11.6 | - | - | 5.9 (torch-eager-fp32) | 1.953 | no torch bf16 arm on this lane |
| adamax | synthetic | 9.1 | - | - | 5.6 (torch-eager-fp32) | 1.626 | no torch bf16 arm on this lane |
| adamw | synthetic | 11.7 | - | - | 6.5 (torch-eager-fp32) | 1.788 | no torch bf16 arm on this lane |
| avgpool1d | synthetic | 0.4 | 0.7 (torch-eager-bf16) | 0.499 | 0.7 (torch-eager-fp32) | 0.517 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| avgpool2d | synthetic | 0.6 | 0.8 (torch-eager-bf16) | 0.750 | 0.9 (torch-eager-fp32) | 0.669 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| batchnorm1d | synthetic | 0.6 | 0.9 (torch-eager-bf16) | 0.655 | 1.0 (torch-eager-fp32) | 0.599 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| batchnorm2d | synthetic | 0.6 | 1.2 (torch-eager-bf16) | 0.524 | 1.0 (torch-eager-fp32) | 0.620 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| cnn-clf | synthetic | 315.0 | 251.7 (torch-eager-bf16) | 1.251 | 172.9 (torch-eager-fp32) | 1.822 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| conv1d | synthetic | 7.1 | 1.4 (torch-eager-bf16) | 5.128 | 1.9 (torch-eager-fp32) | 3.693 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| conv2d | synthetic | 6.4 | 1.5 (torch-eager-bf16) | 4.127 | 1.2 (torch-eager-fp32) | 5.487 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| dropout2d | synthetic | 0.3 | - | - | 0.7 (torch-eager-fp32) | 0.397 | no torch bf16 arm on this lane |
| embedding | synthetic | 1.6 | - | - | 1.0 (torch-eager-fp32) | 1.520 | no torch bf16 arm on this lane |
| gcn | istella | 3.0 | 3.3 (torch-compile-bf16) | 0.921 | 3.6 (torch-compile-fp32) | 0.834 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| gcn | taxi | 1.4 | 2.8 (torch-compile-bf16) | 0.491 | 3.2 (torch-compile-fp32) | 0.428 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| global-avgpool | synthetic | 0.2 | 0.5 (torch-eager-bf16) | 0.540 | 0.7 (torch-eager-fp32) | 0.373 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| global-maxpool | synthetic | 0.2 | 0.5 (torch-eager-bf16) | 0.516 | 0.6 (torch-eager-fp32) | 0.416 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| graphsage | istella | 5.9 | 3.2 (torch-compile-bf16) | 1.843 | 3.4 (torch-compile-fp32) | 1.756 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| graphsage | taxi | 1.6 | 1.4 (torch-compile-bf16) | 1.190 | 1.4 (torch-eager-fp32) | 1.157 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| gru-clf | synthetic | 514.9 | 1852.7 (torch-eager-bf16) | 0.278 | 1946.7 (torch-eager-fp32) | 0.265 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| gru-clf | taxi-hourly | 515.5 | 1899.7 (torch-compile-bf16) | 0.271 | 1792.8 (torch-eager-fp32) | 0.288 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| gru-reg | synthetic | 513.3 | 2078.2 (torch-eager-bf16) | 0.247 | 1993.7 (torch-eager-fp32) | 0.257 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| gru-reg | taxi-hourly | 513.4 | 1801.3 (torch-eager-bf16) | 0.285 | 2006.3 (torch-compile-fp32) | 0.256 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| lamb | synthetic | 13.4 | - | - | - | - | no torch bf16 arm on this lane |
| layernorm | synthetic | 0.6 | 0.8 (torch-eager-bf16) | 0.841 | 0.8 (torch-eager-fp32) | 0.810 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| lion | synthetic | 3.5 | - | - | - | - | no torch bf16 arm on this lane |
| lstm-clf | synthetic | 631.6 | 854.9 (torch-eager-bf16) | 0.739 | 894.9 (torch-compile-fp32) | 0.706 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| lstm-clf | taxi-hourly | 630.7 | 977.6 (torch-compile-bf16) | 0.645 | 815.7 (torch-compile-fp32) | 0.773 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| lstm-reg | synthetic | 628.6 | 1053.4 (torch-eager-bf16) | 0.597 | 923.1 (torch-eager-fp32) | 0.681 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| lstm-reg | taxi-hourly | 627.9 | 1017.5 (torch-eager-bf16) | 0.617 | 920.3 (torch-eager-fp32) | 0.682 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| maxpool1d | synthetic | 0.3 | 0.9 (torch-eager-bf16) | 0.368 | 0.8 (torch-eager-fp32) | 0.431 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| maxpool2d | synthetic | 0.5 | 1.1 (torch-compile-bf16) | 0.490 | 0.9 (torch-eager-fp32) | 0.613 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| moe | synthetic | 15.2 | 3.2 (torch-compile-bf16) | 4.751 | 5.6 (torch-eager-fp32) | 2.720 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| nadam | synthetic | 9.0 | - | - | 6.8 (torch-eager-fp32) | 1.317 | no torch bf16 arm on this lane |
| resnet-block | synthetic | 12.6 | 2.1 (torch-eager-bf16) | 6.106 | 2.2 (torch-eager-fp32) | 5.696 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| rmsprop | synthetic | 9.0 | - | - | 4.8 (torch-eager-fp32) | 1.872 | no torch bf16 arm on this lane |
| rnn-clf | synthetic | 305.9 | 950.8 (torch-eager-bf16) | 0.322 | 879.7 (torch-compile-fp32) | 0.348 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| rnn-clf | taxi-hourly | 306.6 | 975.1 (torch-eager-bf16) | 0.314 | 743.5 (torch-eager-fp32) | 0.412 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| rnn-reg | synthetic | 300.1 | 898.5 (torch-eager-bf16) | 0.334 | 891.9 (torch-eager-fp32) | 0.337 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| rnn-reg | taxi-hourly | 300.7 | 1010.8 (torch-eager-bf16) | 0.297 | 762.7 (torch-compile-fp32) | 0.394 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| sgd | synthetic | 10.5 | - | - | 3.7 (torch-eager-fp32) | 2.872 | no torch bf16 arm on this lane |
| gemm-bf16 | gaussian | 23.1 | 9.2 (torch-eager-bf16) | 2.513 | - | - | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| gemm | gaussian | 28.6 | 10.8 (torch-eager-bf16) | 2.646 | 11.7 (torch-compile-fp32) | 2.452 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| lm-forward | bytes | 33.6 | 13.3 (torch-eager-bf16) | 2.533 | 14.5 (torch-compile-fp32) | 2.318 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| lm-train-step | bytes | 43.0 | 8.2 (torch-compile-bf16) | 5.254 | 15.2 (torch-compile-fp32) | 2.830 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| mamba1-forward | gaussian | 3.1 | 105.1 (torch-eager-bf16) | 0.030 | 87.6 (torch-eager-fp32) | 0.036 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| mamba2-forward | gaussian | 4.4 | 11.5 (torch-compile-bf16) | 0.384 | 12.7 (torch-compile-fp32) | 0.347 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| mlp-train-step | gaussian | 3.4 | 1.5 (torch-compile-bf16) | 2.304 | 1.3 (torch-eager-fp32) | 2.578 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| samba-forward | bytes | 5.6 | 5.7 (torch-compile-bf16) | 0.994 | 5.9 (torch-compile-fp32) | 0.953 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| samba-train-step | bytes | 69.0 | 127.6 (torch-eager-bf16) | 0.541 | 135.2 (torch-eager-fp32) | 0.511 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| transformer-forward | gaussian | 3.2 | 1.9 (torch-eager-bf16) | 1.660 | 1.6 (torch-compile-fp32) | 2.045 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |

## Trees

### gbdt-categorical / taxi (rows full, shape taxicat-4110786x16)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-grid-logs.txt (amd/a1694)`, ran on AMD Instinct MI325X (amd, DO) job a1694

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 10482.7 | 10482.7..10482.7 | 1 | - | - | 10482.7 | - | - (stored whole) | - | - | - | - | auc=0.631297, logloss=0.528285 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1694 2026-10-11; identity vs the nvidia columns: n/a) |
| xgboost-gpu | xgboost | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | xgboost 3.2.0 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | UNKNOWN | - | REFUSED(asked for by --arms and not built here; built: xgboost-cpu) (copied from opponents-20261006; measured this run) |
| catboost-cpu | catboost | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | catboost 1.2.10 | opponent | 97857.4 | 97857.4..97857.4 | 1 | 0.107 | - | 97857.4 | 97857.4 | 0.00 (cpu-arm) | 0.107 (whole/whole) | - | 11918.4 | - | auc=0.628772, logloss=0.528923 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| xgboost-cpu | xgboost | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | xgboost 3.2.0 | opponent | 18924.8 | 18924.8..18924.8 | 1 | 0.554 | - | 18924.8 | 18924.8 | 0.00 (cpu-arm) | 0.554 (whole/whole) | - | 8838.5 | - | auc=0.631473, logloss=0.528686 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| lightgbm-cpu | lightgbm | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | lightgbm 4.7.0 | opponent | 10928.0 | 10928.0..10928.0 | 1 | 0.959 | - | 10928.0 | 10928.0 | 0.00 (cpu-arm) | 0.959 (whole/whole) | - | 8741.9 | - | auc=0.632665, logloss=0.528094 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours, xgboost-gpu: host not sampled; GPU not sampled

memory, catboost-cpu, xgboost-cpu, lightgbm-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (binary task) (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-depthwise / istella (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a0499)`, ran on AMD Instinct MI325X (amd, DO) job a0499

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 6295.4 | 6295.4..6295.4 | 1 | - | - | 6295.4 | - | - (stored whole) | - | - | - | - | auc=0.983304, logloss=0.156072 | - | main board, one scored run | - | ok (main@89485bc96 amd/a0499 2026-10-08; identity vs the nvidia columns: n/a) |

settings: None. Rows: None. Timed: None.

mismatch: opponents withheld: the lane settings at this tree's HEAD differ from the settings opponents-20261006 recorded for its opponent race (an opponent job must score them again)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-depthwise / taxi (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a0499)`, ran on AMD Instinct MI325X (amd, DO) job a0499

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 3631.3 | 3631.3..3631.3 | 1 | - | - | 3631.3 | - | - (stored whole) | - | - | - | - | auc=0.632205, logloss=0.527838 | - | main board, one scored run | - | ok (main@89485bc96 amd/a0499 2026-10-08; identity vs the nvidia columns: n/a) |

settings: None. Rows: None. Timed: None.

mismatch: opponents withheld: the lane settings at this tree's HEAD differ from the settings opponents-20261006 recorded for its opponent race (an opponent job must score them again)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-lossguide / istella (rows full, shape istella-2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-grid-logs.txt (amd/a1694)`, ran on AMD Instinct MI325X (amd, DO) job a1694

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 6445.8 | 6445.8..6445.8 | 1 | - | - | 6445.8 | - | - (stored whole) | - | - | - | - | auc=0.983586, logloss=0.150066 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1694 2026-10-11; identity vs the nvidia columns: n/a) |
| xgboost-gpu | xgboost | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | xgboost 3.2.0 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | UNKNOWN | - | REFUSED(asked for by --arms and not built here; built: xgboost-cpu) (copied from opponents-20261006; measured this run) |
| catboost-cpu | catboost | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | catboost 1.2.10 | opponent | 103179.7 | 103179.7..103179.7 | 1 | 0.062 | - | 103179.7 | 103179.7 | 0.00 (cpu-arm) | 0.062 (whole/whole) | - | 10528.0 | - | auc=0.983135, logloss=0.157692 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| xgboost-cpu | xgboost | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | xgboost 3.2.0 | opponent | 23613.1 | 23613.1..23613.1 | 1 | 0.273 | - | 23613.1 | 23613.1 | 0.00 (cpu-arm) | 0.273 (whole/whole) | - | 11234.1 | - | auc=0.983622, logloss=0.149263 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| lightgbm-cpu | lightgbm | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | lightgbm 4.7.0 | opponent | 19843.3 | 19843.3..19843.3 | 1 | 0.325 | - | 19843.3 | 19843.3 | 0.00 (cpu-arm) | 0.325 (whole/whole) | - | 11308.2 | - | auc=0.983778, logloss=0.149653 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours, xgboost-gpu: host not sampled; GPU not sampled

memory, catboost-cpu, xgboost-cpu, lightgbm-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-lossguide / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-grid-logs.txt (amd/a1694)`, ran on AMD Instinct MI325X (amd, DO) job a1694

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 3708.8 | 3708.8..3708.8 | 1 | - | - | 3708.8 | - | - (stored whole) | - | - | - | - | auc=0.632096, logloss=0.528044 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1694 2026-10-11; identity vs the nvidia columns: n/a) |
| xgboost-gpu | xgboost | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | xgboost 3.2.0 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | UNKNOWN | - | REFUSED(asked for by --arms and not built here; built: xgboost-cpu) (copied from opponents-20261006; measured this run) |
| catboost-cpu | catboost | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | catboost 1.2.10 | opponent | 77432.0 | 77432.0..77432.0 | 1 | 0.048 | - | 77432.0 | 77432.0 | 0.00 (cpu-arm) | 0.048 (whole/whole) | - | 6674.6 | - | auc=0.632578, logloss=0.527851 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| xgboost-cpu | xgboost | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | xgboost 3.2.0 | opponent | 10350.4 | 10350.4..10350.4 | 1 | 0.358 | - | 10350.4 | 10350.4 | 0.00 (cpu-arm) | 0.358 (whole/whole) | - | 6853.6 | - | auc=0.630968, logloss=0.528677 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| lightgbm-cpu | lightgbm | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | lightgbm 4.7.0 | opponent | 10099.2 | 10099.2..10099.2 | 1 | 0.367 | - | 10099.2 | 10099.2 | 0.00 (cpu-arm) | 0.367 (whole/whole) | - | 6933.8 | - | auc=0.632243, logloss=0.528067 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours, xgboost-gpu: host not sampled; GPU not sampled

memory, catboost-cpu, xgboost-cpu, lightgbm-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-multiclass / istella (rows full, shape istellamc-2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a1116)`, ran on AMD Instinct MI325X (amd, DO) job a1116

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@d886b53dc) | identical | 10051.4 | 10051.4..10051.4 | 1 | - | - | 10051.4 | - | - (stored whole) | - | - | - | - | accuracy=0.907556, mlogloss=0.258413 | - | main board, one scored run | - | ok (main@d886b53dc amd/a1116 2026-10-09; identity vs the nvidia columns: n/a) |
| xgboost-gpu | xgboost | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | xgboost 3.2.0 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | UNKNOWN | - | REFUSED(asked for by --arms and not built here; built: xgboost-cpu) (copied from opponents-20261006; measured this run) |
| catboost-cpu | catboost | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | catboost 1.2.10 | opponent | 561543.4 | 561543.4..561543.4 | 1 | 0.018 | - | 561543.4 | 561543.4 | 0.00 (cpu-arm) | 0.018 (whole/whole) | - | 11547.8 | - | accuracy=0.907768, mlogloss=0.258286 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| xgboost-cpu | xgboost | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | xgboost 3.2.0 | opponent | 76053.9 | 76053.9..76053.9 | 1 | 0.132 | - | 76053.9 | 76053.9 | 0.00 (cpu-arm) | 0.132 (whole/whole) | - | 12606.4 | - | accuracy=0.910140, mlogloss=0.246803 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| lightgbm-cpu | lightgbm | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | lightgbm 4.7.0 | opponent | 79078.6 | 79078.6..79078.6 | 1 | 0.127 | - | 79078.6 | 79078.6 | 0.00 (cpu-arm) | 0.127 (whole/whole) | - | 12549.6 | - | accuracy=0.910058, mlogloss=0.245916 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours, xgboost-gpu: host not sampled; GPU not sampled

memory, catboost-cpu, xgboost-cpu, lightgbm-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (MULTICLASS task) (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-multiclass / taxi (rows full, shape taximc-4110786x16)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a1116)`, ran on AMD Instinct MI325X (amd, DO) job a1116

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@d886b53dc) | identical | 6049.8 | 6049.8..6049.8 | 1 | - | - | 6049.8 | - | - (stored whole) | - | - | - | - | accuracy=0.599380, mlogloss=1.012595 | - | main board, one scored run | - | ok (main@d886b53dc amd/a1116 2026-10-09; identity vs the nvidia columns: n/a) |
| xgboost-gpu | xgboost | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | xgboost 3.2.0 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | UNKNOWN | - | REFUSED(asked for by --arms and not built here; built: xgboost-cpu) (copied from opponents-20261006; measured this run) |
| catboost-cpu | catboost | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | catboost 1.2.10 | opponent | 178349.9 | 178349.9..178349.9 | 1 | 0.034 | - | 178349.9 | 178349.9 | 0.00 (cpu-arm) | 0.034 (whole/whole) | - | 8021.3 | - | accuracy=0.599150, mlogloss=1.012734 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| xgboost-cpu | xgboost | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | xgboost 3.2.0 | opponent | 35861.4 | 35861.4..35861.4 | 1 | 0.169 | - | 35861.4 | 35861.4 | 0.00 (cpu-arm) | 0.169 (whole/whole) | - | 8463.4 | - | accuracy=0.601128, mlogloss=1.005128 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| lightgbm-cpu | lightgbm | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | lightgbm 4.7.0 | opponent | 41539.9 | 41539.9..41539.9 | 1 | 0.146 | - | 41539.9 | 41539.9 | 0.00 (cpu-arm) | 0.146 (whole/whole) | - | 8711.3 | - | accuracy=0.601580, mlogloss=1.004282 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours, xgboost-gpu: host not sampled; GPU not sampled

memory, catboost-cpu, xgboost-cpu, lightgbm-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (MULTICLASS task) (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-ordered / istella (rows full, shape istella-2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a1116)`, ran on AMD Instinct MI325X (amd, DO) job a1116

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@d886b53dc) | identical | 34452.1 | 34452.1..34452.1 | 1 | - | - | 34452.1 | - | - (stored whole) | - | - | - | - | auc=0.979444, logloss=0.190928 | - | main board, one scored run | - | ok (main@d886b53dc amd/a1116 2026-10-09; identity vs the nvidia columns: n/a) |
| catboost-cpu | catboost | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | catboost 1.2.10 | opponent | 273122.4 | 273122.4..273122.4 | 1 | 0.126 | - | 273122.4 | 273122.4 | 0.00 (cpu-arm) | 0.126 (whole/whole) | - | 11159.1 | - | auc=0.979221, logloss=0.192114 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, catboost-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

config: NVIDIA gbm-bench, cat shared_params, ntrees 500, boosting_type 'Ordered' (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-ordered / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a1116)`, ran on AMD Instinct MI325X (amd, DO) job a1116

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@d886b53dc) | identical | 19884.4 | 19884.4..19884.4 | 1 | - | - | 19884.4 | - | - (stored whole) | - | - | - | - | auc=0.629203, logloss=0.529007 | - | main board, one scored run | - | ok (main@d886b53dc amd/a1116 2026-10-09; identity vs the nvidia columns: n/a) |
| catboost-cpu | catboost | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | catboost 1.2.10 | opponent | 78566.1 | 78566.1..78566.1 | 1 | 0.253 | - | 78566.1 | 78566.1 | 0.00 (cpu-arm) | 0.253 (whole/whole) | - | 8753.0 | - | auc=0.628918, logloss=0.529083 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, catboost-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

config: NVIDIA gbm-bench, cat shared_params, ntrees 500, boosting_type 'Ordered' (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-rank-pairlogit / istella (rows full, shape istellarank-2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a1116)`, ran on AMD Instinct MI325X (amd, DO) job a1116

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@d886b53dc) | identical | 1541.0 | 1541.0..1541.0 | 1 | - | - | 1541.0 | - | - (stored whole) | - | - | - | - | map=0.854545, ndcg10=0.719953, ndcg5=0.650400 | - | main board, one scored run | - | ok (main@d886b53dc amd/a1116 2026-10-09; identity vs the nvidia columns: n/a) |
| xgboost-gpu | xgboost | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | xgboost 3.2.0 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | UNKNOWN | - | REFUSED(asked for by --arms and not built here; built: xgboost-cpu) (copied from opponents-20261006; measured this run) |
| catboost-cpu | catboost | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | catboost 1.2.10 | opponent | 31176.3 | 31176.3..31176.3 | 1 | 0.049 | - | 31176.3 | 31176.3 | 0.00 (cpu-arm) | 0.049 (whole/whole) | - | 12433.4 | - | map=0.846328, ndcg10=0.713361, ndcg5=0.643611 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| xgboost-cpu | xgboost | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | xgboost 3.2.0 | opponent | 6075.9 | 6075.9..6075.9 | 1 | 0.254 | - | 6075.9 | 6075.9 | 0.00 (cpu-arm) | 0.254 (whole/whole) | - | 13581.1 | - | map=0.872796, ndcg10=0.738397, ndcg5=0.670093 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours, xgboost-gpu: host not sampled; GPU not sampled

memory, catboost-cpu, xgboost-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-symmetric / istella (rows full, shape istella-2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a0499)`, ran on AMD Instinct MI325X (amd, DO) job a0499

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 5995.7 | 5995.7..5995.7 | 1 | - | - | 5995.7 | - | - (stored whole) | - | - | - | - | auc=0.980129, logloss=0.186686 | - | main board, one scored run | - | ok (main@89485bc96 amd/a0499 2026-10-08; identity vs the nvidia columns: n/a) |
| catboost-cpu | catboost | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | catboost 1.2.10 | opponent | 44398.5 | 44398.5..44398.5 | 1 | 0.135 | - | 44398.5 | 44398.5 | 0.00 (cpu-arm) | 0.135 (whole/whole) | - | 9400.6 | - | auc=0.979899, logloss=0.188093 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, catboost-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-symmetric / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a0499)`, ran on AMD Instinct MI325X (amd, DO) job a0499

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 3958.8 | 3958.8..3958.8 | 1 | - | - | 3958.8 | - | - (stored whole) | - | - | - | - | auc=0.630436, logloss=0.528554 | - | main board, one scored run | - | ok (main@89485bc96 amd/a0499 2026-10-08; identity vs the nvidia columns: n/a) |
| catboost-cpu | catboost | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | catboost 1.2.10 | opponent | 21286.9 | 21286.9..21286.9 | 1 | 0.186 | - | 21286.9 | 21286.9 | 0.00 (cpu-arm) | 0.186 (whole/whole) | - | 5984.4 | - | auc=0.630269, logloss=0.528650 | yes | COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, catboost-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

## Classical

### dbscan / istella (rows full, shape 1000000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a1558)`, ran on AMD Instinct MI325X (amd, DO) job a1558

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@ca25d9321) | identical | 54245.1 | 54245.1..54245.1 | 1 | - | - | 54245.1 | - | - (stored whole) | - | - | - | - | n_clusters=40131, noise_fraction=0.219391, rows=1000000 | - | main board, one scored run | - | ok (main@ca25d9321 amd/a1558 2026-10-10; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 263657.7 | 263657.7..263657.7 | 1 | 0.206 | - | 263657.7 | 263657.7 | 0.00 (cpu-arm) | 0.206 (whole/whole) | - | 5971.6 | - | ari_vs_ours=1.000000, n_clusters=40131, noise_agreement_vs_ours=1.000000, noise_fraction=0.219391, rows=1000000 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: eps=3, min_samples=2 (the cuML benchmark's DBSCAN) on every arm; metric='euclidean'. Rows: dbscan block: 1,000,000 rows, standardized. Timed: fit.

mismatch: seed: no arm has a seed argument (deterministic)

mismatch: algorithm (an exact neighbor search on every arm, results unchanged): ours 'rbc' (its default), scikit-learn 'brute' (the cuML benchmark's cpu_args; it has no 'rbc'), cuml-gpu 'brute', cuml-gpu-rbc 'rbc'

mismatch: leaf_size=30 and n_jobs=-1: scikit-learn only

config: cuML benchmark (RAPIDS), DBSCAN (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### dbscan / taxi (rows full, shape 1000000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a1558)`, ran on AMD Instinct MI325X (amd, DO) job a1558

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@ca25d9321) | identical | 7056.5 | 7056.5..7056.5 | 1 | - | - | 7056.5 | - | - (stored whole) | - | - | - | - | n_clusters=36, noise_fraction=0.000174, rows=1000000 | - | main board, one scored run | - | ok (main@ca25d9321 amd/a1558 2026-10-10; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | HOST-MEMORY(killed at 143.0 GB: the driver's process tree held 153.6 GB, over 90% of the box's 168.8 GB) (copied from release-board-resume-r2; measured this run) |

settings: eps=3, min_samples=2 (the cuML benchmark's DBSCAN) on every arm; metric='euclidean'. Rows: dbscan block: 1,000,000 rows, standardized. Timed: fit.

mismatch: seed: no arm has a seed argument (deterministic)

mismatch: algorithm (an exact neighbor search on every arm, results unchanged): ours 'rbc' (its default), scikit-learn 'brute' (the cuML benchmark's cpu_args; it has no 'rbc'), cuml-gpu 'brute', cuml-gpu-rbc 'rbc'

mismatch: leaf_size=30 and n_jobs=-1: scikit-learn only

config: cuML benchmark (RAPIDS), DBSCAN (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### hdbscan / istella (rows full, shape 1000000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-grid-logs.txt (amd/a1714)`, ran on AMD Instinct MI325X (amd, DO) job a1714

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 955.2 | 955.2..955.2 | 1 | - | - | 955.2 | - | - (stored whole) | - | - | - | - | n_clusters=47, noise_fraction=0.253810, rows=100000 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1714 2026-10-11; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 829354.4 | 829354.4..829354.4 | 1 | 0.001 | - | 829354.4 | 829354.4 | 0.00 (cpu-arm) | 0.001 (whole/whole) | - | 1462.1 | - | n_clusters=52, noise_fraction=0.252570, rows=100000 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: min_samples=10, min_cluster_size=100, metric='euclidean', cluster_selection_method='eom', cluster_selection_epsilon=0.0, alpha=1.0, allow_single_cluster=False. Rows: the dbscan block's first 100,000 rows. Timed: fit.

mismatch: seed: no arm has a seed argument (deterministic)

mismatch: max_cluster_size: ours and cuML 0, scikit-learn None (both mean no limit)

mismatch: scikit-learn algorithm='auto', leaf_size=40, n_jobs=-1: its own; cuML build_algo at its default

config: cuML benchmark (RAPIDS), HDBSCAN (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### hdbscan / taxi (rows full, shape 1000000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-grid-logs.txt (amd/a1714)`, ran on AMD Instinct MI325X (amd, DO) job a1714

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 323.4 | 323.4..323.4 | 1 | - | - | 323.4 | - | - (stored whole) | - | - | - | - | n_clusters=161, noise_fraction=0.140550, rows=100000 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1714 2026-10-11; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 38362.1 | 38362.1..38362.1 | 1 | 0.008 | - | 38362.1 | 38362.1 | 0.00 (cpu-arm) | 0.008 (whole/whole) | - | 334.0 | - | n_clusters=161, noise_fraction=0.134630, rows=100000 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: min_samples=10, min_cluster_size=100, metric='euclidean', cluster_selection_method='eom', cluster_selection_epsilon=0.0, alpha=1.0, allow_single_cluster=False. Rows: the dbscan block's first 100,000 rows. Timed: fit.

mismatch: seed: no arm has a seed argument (deterministic)

mismatch: max_cluster_size: ours and cuML 0, scikit-learn None (both mean no limit)

mismatch: scikit-learn algorithm='auto', leaf_size=40, n_jobs=-1: its own; cuML build_algo at its default

config: cuML benchmark (RAPIDS), HDBSCAN (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### kde / istella (rows full, shape 100000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-grid-logs.txt (amd/a1704)`, ran on AMD Instinct MI325X (amd, DO) job a1704

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 23.4 | 23.4..23.4 | 1 | - | - | 23.4 | - | - (stored whole) | - | - | - | - | mean_log_likelihood=-222.270583, rows_without_density=0 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1704 2026-10-11; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 57344.0 | 57344.0..57344.0 | 1 | 0.0004081 | - | 57344.0 | 57344.0 | 0.00 (cpu-arm) | 0.0004081 (whole/whole) | - | 466.0 | - | mean_log_likelihood=-226.977407, rows_without_density=0 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: bandwidth=1.0, kernel='gaussian' (the cuML benchmark's KernelDensity), metric='euclidean' on every arm; ours and scikit-learn atol=0, rtol=0, algorithm='auto', leaf_size=40, breadth_first=True. Rows: kde block: 100,000 fit rows, 2,000 queries, standardized. Timed: score_samples; the fit is before the clock on every arm.

mismatch: seed: no arm has a seed argument (exact density)

mismatch: cuML has no atol, rtol, algorithm, leaf_size or breadth_first (exact brute force)

config: cuML benchmark (RAPIDS), KernelDensity (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### kde / taxi (rows full, shape 100000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-grid-logs.txt (amd/a1704)`, ran on AMD Instinct MI325X (amd, DO) job a1704

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 12.0 | 12.0..12.0 | 1 | - | - | 12.0 | - | - (stored whole) | - | - | - | - | mean_log_likelihood=-14.826436, rows_without_density=0 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1704 2026-10-11; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 7083.5 | 7083.5..7083.5 | 1 | 0.002 | - | 7083.5 | 7083.5 | 0.00 (cpu-arm) | 0.002 (whole/whole) | - | 209.0 | - | mean_log_likelihood=-14.826437, rows_without_density=0 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: bandwidth=1.0, kernel='gaussian' (the cuML benchmark's KernelDensity), metric='euclidean' on every arm; ours and scikit-learn atol=0, rtol=0, algorithm='auto', leaf_size=40, breadth_first=True. Rows: kde block: 100,000 fit rows, 2,000 queries, standardized. Timed: score_samples; the fit is before the clock on every arm.

mismatch: seed: no arm has a seed argument (exact density)

mismatch: cuML has no atol, rtol, algorithm, leaf_size or breadth_first (exact brute force)

config: cuML benchmark (RAPIDS), KernelDensity (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### kmeans / istella (rows full, shape 2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/amd-wave8/campaign/measured/classical.kmeans.istella.rows=full/board.json (amd/paired-578dcfb7f8d25a48cd7af1536281e66d50bb6acb0bcbb9165bbdb58aa4b36ae2)`, ran on mojolearn-steward-do-amd

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | 0.8.44 | identical | 293.5 | 293.5..293.5 | 1 | - | - | 293.5 | 261.5 | 32.04 (upload_ms_separate) | - | - | 4537.7 | 234707.5 | inertia=6.051e+17, inertia_over_ours=1.000000, n_iter=33, n_iter_over_ours=1.000000 | yes | main board, one scored run | wheel | ok (main@50be53955 amd/paired-578dcfb7f8d25a48cd7af1536281e66d50bb6acb0bcbb9165bbdb58aa4b36ae2 2026-10-14; identity vs the nvidia columns: n/a) |
| hipvs-gpu | hipvs | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | hipvs 01.00.00 | opponent | 38671.7 | 38671.7..38671.7 | 1 | 0.008 | - | 38734.2 | 38671.7 | 62.52 (upload_ms_untimed) | 0.008 (whole/whole) | - | 3396.9 | 2884.2 | inertia=5.959e+17, inertia_over_ours=0.984774, n_iter=301, n_iter_over_ours=9.121212 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/7391/vram_* bytes for this pid at the round's end (not a peak)

memory, hipvs-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/7392/vram_* bytes for this pid at the round's end (not a peak)

settings: n_clusters=8, init='k-means++', max_iter=300, n_init=1 (the cuML benchmark's KMeans), oversampling_factor=0 on ours and cuML (its cuml_args; the classic sequential seeding), tol=1e-7 (an absolute squared center shift on every arm; scikit-learn's relative tol rescaled to it), metric='euclidean', Lloyd; seed 7 (ours, scikit-learn and cuML random_state=7). Rows: big block: 4,000,000 taxi rows or the Istella-S train split, raw. Timed: fit (the k-means++ seeding included on every arm).

mismatch: k-means++: each library draws its own start from its own generator seeded 7, so the starts differ and so may n_iter; scikit-learn greedy k-means++ (2 + log k candidates per center), ours and cuML the sequential k-means++ of cuML (oversampling_factor=0)

mismatch: stopping rule, the same on every arm: total squared center shift below 1e-7 (absolute) or max_iter=300. ours and cuML tol=1e-7 is that threshold (cuVS kmeans.cuh sqrdNormError < tol; inertia_check off); scikit-learn's tol is relative to mean(var(X)), so it gets 1e-7 / mean(var(X)) of the same rows (the same absolute threshold; it also stops when no label changes, a zero shift). n_iter and n_iter_over_ours are in every quality cell

mismatch: algorithm: scikit-learn algorithm='lloyd'; ours and cuML have no such parameter (Lloyd)

mismatch: torch-gpu: refused, not an opponent (the harness's own k-means from torch ops); hipvs-gpu (AMD): hipVS cuvs.cluster.kmeans, init_method='KMeansPlusPlus' with oversampling_factor=0, tol=1e-7 under cuVS's rule, squared L2, no seed argument

config: cuML benchmark (RAPIDS), KMeans (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | hipvs-gpu | ours |
|---|---||---|---|
| library (source) | cuvs (declared) | mojolearn (get_params) |
| hierarchical | false | - |
| hierarchical_n_iters | 20 | - |
| init | "KMeansPlusPlus" | "k-means++" |
| init_centroids | - | null |
| max_iter | 300 | 300 |
| metric | "sqeuclidean" | "euclidean" |
| n_clusters | 8 | 8 |
| n_init | 1 | 1 |
| oversampling_factor | 0.0 | 0.0 |
| seed | "none (no argument; draws random numbers, see exceptions)" | 7 |
| tol | 1e-07 | 1e-07 |

accepted difference: hipvs-gpu seed: hipVS (AMD's cuVS port) cuvs.cluster.kmeans KMeansParams takes no seed: its k-means++ draws from cuVS's own generator; ours, cuML and scikit-learn get 7

accepted difference: hipvs-gpu hierarchical: hipVS hierarchical=False selects ordinary KMeans; hierarchical_n_iters is inactive

accepted difference: hipvs-gpu hierarchical_n_iters: hipVS hierarchical=False selects ordinary KMeans; hierarchical_n_iters is inactive

accepted difference: hipvs-gpu init: same k-means++ initializer, spelled KMeansPlusPlus by hipVS KMeansParams; oversampling is compared separately

accepted difference: hipvs-gpu init_centroids: no supplied initial centroids; hipVS fit(params, X) uses the separately compared init_method

accepted difference: hipvs-gpu metric: both KMeans objectives minimize squared Euclidean distances; hipVS names the squared distance explicitly

Inference (each arm predicts with its own model from the fit rounds above):

| arm | hardware | version | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | AMD Instinct MI325X (amd, DO) | 0.8.44 | Xq | 500000 | 18.1 | 18.1..18.1 | 1 | - | - | 18.1 | - | - (stored whole) | - | - | eval_inertia=1.417e+17, label_agreement_own_centers=1.000000 | yes | LIKE-FOR-LIKE-SPAN | ok |
| hipvs-gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | hipvs 01.00.00 | Xq | 500000 | - | - | 0 | - | - | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(error: {"error": "RuntimeError('no inference call wired for HipvsKMeans')", "event": "infer_error", "stage": "infer 0"}) |

inference call, ours: mojolearn KMeans.predict(Xq), host rows in, host result out

inference call, hipvs-gpu: predict(Xq)

### kmeans / taxi (rows full, shape 4000000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/amd-wave8/campaign/measured/classical.kmeans.taxi.rows=full/board.json (amd/paired-3c492a6c51e41f72cec4bcfc63591ffa313aef5a9d8508f5b6830211b4af7cc7)`, ran on mojolearn-steward-do-amd

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | 0.8.44 | identical | 184.0 | 184.0..184.0 | 1 | - | - | 184.0 | 180.9 | 3.18 (upload_ms_separate) | - | - | 2623.0 | 236099.4 | inertia=3.093e+08, inertia_over_ours=1.000000, n_iter=91, n_iter_over_ours=1.000000 | yes | main board, one scored run | wheel | ok (main@50be53955 amd/paired-3c492a6c51e41f72cec4bcfc63591ffa313aef5a9d8508f5b6830211b4af7cc7 2026-10-14; identity vs the nvidia columns: n/a) |
| hipvs-gpu | hipvs | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | hipvs 01.00.00 | opponent | 14153.2 | 14153.2..14153.2 | 1 | 0.013 | - | 14191.0 | 14153.2 | 37.76 (upload_ms_untimed) | 0.013 (whole/whole) | - | 1481.2 | 1466.2 | inertia=3.06e+08, inertia_over_ours=0.989329, n_iter=79, n_iter_over_ours=0.868132 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/6658/vram_* bytes for this pid at the round's end (not a peak)

memory, hipvs-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/6659/vram_* bytes for this pid at the round's end (not a peak)

settings: n_clusters=8, init='k-means++', max_iter=300, n_init=1 (the cuML benchmark's KMeans), oversampling_factor=0 on ours and cuML (its cuml_args; the classic sequential seeding), tol=1e-7 (an absolute squared center shift on every arm; scikit-learn's relative tol rescaled to it), metric='euclidean', Lloyd; seed 7 (ours, scikit-learn and cuML random_state=7). Rows: big block: 4,000,000 taxi rows or the Istella-S train split, raw. Timed: fit (the k-means++ seeding included on every arm).

mismatch: k-means++: each library draws its own start from its own generator seeded 7, so the starts differ and so may n_iter; scikit-learn greedy k-means++ (2 + log k candidates per center), ours and cuML the sequential k-means++ of cuML (oversampling_factor=0)

mismatch: stopping rule, the same on every arm: total squared center shift below 1e-7 (absolute) or max_iter=300. ours and cuML tol=1e-7 is that threshold (cuVS kmeans.cuh sqrdNormError < tol; inertia_check off); scikit-learn's tol is relative to mean(var(X)), so it gets 1e-7 / mean(var(X)) of the same rows (the same absolute threshold; it also stops when no label changes, a zero shift). n_iter and n_iter_over_ours are in every quality cell

mismatch: algorithm: scikit-learn algorithm='lloyd'; ours and cuML have no such parameter (Lloyd)

mismatch: torch-gpu: refused, not an opponent (the harness's own k-means from torch ops); hipvs-gpu (AMD): hipVS cuvs.cluster.kmeans, init_method='KMeansPlusPlus' with oversampling_factor=0, tol=1e-7 under cuVS's rule, squared L2, no seed argument

config: cuML benchmark (RAPIDS), KMeans (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | hipvs-gpu | ours |
|---|---||---|---|
| library (source) | cuvs (declared) | mojolearn (get_params) |
| hierarchical | false | - |
| hierarchical_n_iters | 20 | - |
| init | "KMeansPlusPlus" | "k-means++" |
| init_centroids | - | null |
| max_iter | 300 | 300 |
| metric | "sqeuclidean" | "euclidean" |
| n_clusters | 8 | 8 |
| n_init | 1 | 1 |
| oversampling_factor | 0.0 | 0.0 |
| seed | "none (no argument; draws random numbers, see exceptions)" | 7 |
| tol | 1e-07 | 1e-07 |

accepted difference: hipvs-gpu seed: hipVS (AMD's cuVS port) cuvs.cluster.kmeans KMeansParams takes no seed: its k-means++ draws from cuVS's own generator; ours, cuML and scikit-learn get 7

accepted difference: hipvs-gpu hierarchical: hipVS hierarchical=False selects ordinary KMeans; hierarchical_n_iters is inactive

accepted difference: hipvs-gpu hierarchical_n_iters: hipVS hierarchical=False selects ordinary KMeans; hierarchical_n_iters is inactive

accepted difference: hipvs-gpu init: same k-means++ initializer, spelled KMeansPlusPlus by hipVS KMeansParams; oversampling is compared separately

accepted difference: hipvs-gpu init_centroids: no supplied initial centroids; hipVS fit(params, X) uses the separately compared init_method

accepted difference: hipvs-gpu metric: both KMeans objectives minimize squared Euclidean distances; hipVS names the squared distance explicitly

Inference (each arm predicts with its own model from the fit rounds above):

| arm | hardware | version | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | AMD Instinct MI325X (amd, DO) | 0.8.44 | Xq | 500000 | 1.7 | 1.7..1.7 | 1 | - | - | 1.7 | - | - (stored whole) | - | - | eval_inertia=4.824e+07, label_agreement_own_centers=0.999998 | yes | LIKE-FOR-LIKE-SPAN | ok |
| hipvs-gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | hipvs 01.00.00 | Xq | 500000 | - | - | 0 | - | - | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(error: {"error": "RuntimeError('no inference call wired for HipvsKMeans')", "event": "infer_error", "stage": "infer 0"}) |

inference call, ours: mojolearn KMeans.predict(Xq), host rows in, host result out

inference call, hipvs-gpu: predict(Xq)

### knn / istella (rows full, shape 400000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/amd-wave8/campaign/measured/classical.knn.istella.rows=full/board.json (amd/paired-850b839c06df9b27d80b6305556daf4a1e7ca7f1f28bcae0d06f695c233fc31b)`, ran on mojolearn-steward-do-amd

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | 0.8.44 | identical | 30.1 | 30.1..30.1 | 1 | - | - | 30.1 | 23.0 | 7.15 (upload_ms_separate) | - | - | 2718.5 | 236517.3 | recall_at_k=0.976250, rows_with_repeated_ids=0 | yes | main board, one scored run | wheel | ok (main@50be53955 amd/paired-850b839c06df9b27d80b6305556daf4a1e7ca7f1f28bcae0d06f695c233fc31b 2026-10-14; identity vs the nvidia columns: n/a) |
| hipvs-gpu | hipvs | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | hipvs 01.00.00 | opponent | 42.6 | 42.6..42.6 | 1 | 0.707 | - | 76.8 | 42.6 | 34.26 (upload_ms_untimed) | 0.392 (whole/whole) | - | 1478.7 | 1335.3 | recall_at_k=0.976133, rows_with_repeated_ids=0 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/8775/vram_* bytes for this pid at the round's end (not a peak)

memory, hipvs-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/8776/vram_* bytes for this pid at the round's end (not a peak)

settings: n_neighbors=64 (the cuML benchmark's NearestNeighbors), metric='euclidean', algorithm='brute' (ours, scikit-learn, cuML). Rows: knn block: 400,000 index rows, 4,000 queries, raw. Timed: kneighbors; the fit (index) is before the clock on every arm.

mismatch: seed: no arm has a seed argument (exact search)

mismatch: query_tile: ours only (tiling, results unchanged); n_jobs=-1: scikit-learn only

mismatch: torch-gpu: refused, not an opponent (the harness's own cdist + topk kNN); hipvs-gpu (AMD): hipVS cuvs.neighbors.brute_force, built before the clock, search timed

config: cuML benchmark (RAPIDS), NearestNeighbors (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | hipvs-gpu | ours |
|---|---||---|---|
| library (source) | cuvs (declared) | mojolearn (get_params) |
| algorithm | "brute" | "brute" |
| metric | "euclidean" | "euclidean" |
| n_neighbors | 64 | 64 |
| p | 2 | 2 |
| query_tile | - | 0 |
| seed | "none (deterministic)" | "none (deterministic)" |

accepted difference: hipvs-gpu query_tile: both brute-force searches cover every index/query pair; query_tile only partitions our complete query work

### knn / taxi (rows full, shape 400000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/amd-wave8/campaign/measured/classical.knn.taxi.rows=full/board.json (amd/paired-3a64fecd056673fd18f0bdc019f3f696ecaf14c9f7e6d000c87891232efa6b21)`, ran on mojolearn-steward-do-amd

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | 0.8.44 | identical | 13.4 | 13.4..13.4 | 1 | - | - | 13.4 | 13.0 | 0.38 (upload_ms_separate) | - | - | 2396.9 | 236781.3 | recall_at_k=0.999754, rows_with_repeated_ids=0 | yes | main board, one scored run | wheel | ok (main@50be53955 amd/paired-3a64fecd056673fd18f0bdc019f3f696ecaf14c9f7e6d000c87891232efa6b21 2026-10-14; identity vs the nvidia columns: n/a) |
| hipvs-gpu | hipvs | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | hipvs 01.00.00 | opponent | 15.7 | 15.7..15.7 | 1 | 0.855 | - | 42.3 | 15.7 | 26.68 (upload_ms_untimed) | 0.316 (whole/whole) | - | 1156.8 | 1043.3 | recall_at_k=0.999777, rows_with_repeated_ids=0 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/8163/vram_* bytes for this pid at the round's end (not a peak)

memory, hipvs-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/8164/vram_* bytes for this pid at the round's end (not a peak)

settings: n_neighbors=64 (the cuML benchmark's NearestNeighbors), metric='euclidean', algorithm='brute' (ours, scikit-learn, cuML). Rows: knn block: 400,000 index rows, 4,000 queries, raw. Timed: kneighbors; the fit (index) is before the clock on every arm.

mismatch: seed: no arm has a seed argument (exact search)

mismatch: query_tile: ours only (tiling, results unchanged); n_jobs=-1: scikit-learn only

mismatch: torch-gpu: refused, not an opponent (the harness's own cdist + topk kNN); hipvs-gpu (AMD): hipVS cuvs.neighbors.brute_force, built before the clock, search timed

config: cuML benchmark (RAPIDS), NearestNeighbors (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | hipvs-gpu | ours |
|---|---||---|---|
| library (source) | cuvs (declared) | mojolearn (get_params) |
| algorithm | "brute" | "brute" |
| metric | "euclidean" | "euclidean" |
| n_neighbors | 64 | 64 |
| p | 2 | 2 |
| query_tile | - | 0 |
| seed | "none (deterministic)" | "none (deterministic)" |

accepted difference: hipvs-gpu query_tile: both brute-force searches cover every index/query pair; query_tile only partitions our complete query work

### ols / istella (rows full, shape 2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-grid-logs.txt (amd/a1716)`, ran on AMD Instinct MI325X (amd, DO) job a1716

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 444.7 | 444.7..444.7 | 1 | - | - | 444.7 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.332506, rmse=0.681740 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1716 2026-10-11; identity vs the nvidia columns: n/a) |
| torch-gpu | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 2340.8 | 2340.8..2340.8 | 1 | 0.190 | - | 2523.3 | 2340.8 | 182.42 (upload_ms_untimed) | 0.176 (whole/whole (kernel not derivable)) | - | 5648.2 | 5303.6 | finite=False, r2=nan, rmse=nan | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-gpu-eigh | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 42.8 | 42.8..42.8 | 1 | 10.388 | - | 222.4 | 42.8 | 179.55 (upload_ms_untimed) | 2.000 (whole/whole (kernel not derivable)) | - | 5051.5 | 3649.4 | finite=True, r2=0.151604, rmse=0.768589 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 3980.6 | 3980.6..3980.6 | 1 | 0.112 | - | 3980.6 | 3980.6 | 0.00 (cpu-arm) | 0.112 (whole/whole) | - | 5793.0 | - | finite=True, r2=0.001881, rmse=0.833655 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; stored (measured 2026-09-29T16:35:47Z on mojolearn-steward-do-amd, cpu (AMD EPYC 9575F 64-Core Processor))) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu, torch-gpu-eigh: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: fit_intercept=True. Rows: big block, raw; R2 and RMSE on the 500,000 eval rows. Timed: fit.

mismatch: seed: no arm has a seed argument (closed-form fit); torch-gpu torch.manual_seed(7)

mismatch: solver: ours eig of the normal equations (no parameter), scikit-learn scipy lstsq gelsd (no parameter), cuML algorithm='eig', torch-gpu torch.linalg.lstsq (gels on CUDA), torch-gpu-eigh eigh with a pseudo-inverse cutoff

mismatch: scikit-learn positive=False and copy_X=True: parameters ours does not have

config: cuML benchmark (RAPIDS), LinearRegression (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ols / taxi (rows full, shape 4000000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-grid-logs.txt (amd/a1716)`, ran on AMD Instinct MI325X (amd, DO) job a1716

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 14.1 | 14.1..14.1 | 1 | - | - | 14.1 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.908837, rmse=4.696477 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1716 2026-10-11; identity vs the nvidia columns: n/a) |
| torch-gpu | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 100.7 | 100.7..100.7 | 1 | 0.140 | - | 243.7 | 100.7 | 143.01 (upload_ms_untimed) | 0.058 (whole/whole (kernel not derivable)) | - | 3535.9 | 695.3 | finite=True, r2=0.908840, rmse=4.696376 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-gpu-eigh | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 34.0 | 34.0..34.0 | 1 | 0.414 | - | 176.6 | 34.0 | 142.61 (upload_ms_untimed) | 0.080 (whole/whole (kernel not derivable)) | - | 3018.3 | 572.0 | finite=True, r2=0.908822, rmse=4.696849 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 304.7 | 304.7..304.7 | 1 | 0.046 | - | 304.7 | 304.7 | 0.00 (cpu-arm) | 0.046 (whole/whole) | - | 782.5 | - | finite=True, r2=0.724850, rmse=8.159187 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; stored (measured 2026-09-29T16:35:14Z on mojolearn-steward-do-amd, cpu (AMD EPYC 9575F 64-Core Processor))) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu, torch-gpu-eigh: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: fit_intercept=True. Rows: big block, raw; R2 and RMSE on the 500,000 eval rows. Timed: fit.

mismatch: seed: no arm has a seed argument (closed-form fit); torch-gpu torch.manual_seed(7)

mismatch: solver: ours eig of the normal equations (no parameter), scikit-learn scipy lstsq gelsd (no parameter), cuML algorithm='eig', torch-gpu torch.linalg.lstsq (gels on CUDA), torch-gpu-eigh eigh with a pseudo-inverse cutoff

mismatch: scikit-learn positive=False and copy_X=True: parameters ours does not have

config: cuML benchmark (RAPIDS), LinearRegression (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### pca / istella (rows full, shape 2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-grid-logs.txt (amd/a1676)`, ran on AMD Instinct MI325X (amd, DO) job a1676

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 51.8 | 51.8..51.8 | 1 | - | - | 51.8 | - | - (stored whole) | - | - | - | - | explained_variance_ratio_sum=1.000000 | - | main board, one scored run | - | ok (main@a1c96a5ba amd/a1676 2026-10-11; identity vs the nvidia columns: n/a) |
| torch-gpu | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 19.9 | 19.9..19.9 | 1 | 2.605 | - | 206.7 | 19.9 | 186.78 (upload_ms_untimed) | 0.251 (whole/whole (kernel not derivable)) | - | 5044.4 | 3505.8 | explained_variance_ratio_sum=1.000000 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 621.7 | 621.7..621.7 | 1 | 0.083 | - | 621.7 | 621.7 | 0.00 (cpu-arm) | 0.083 (whole/whole) | - | 2342.9 | - | explained_variance_ratio_sum=1.000000 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; stored (measured 2026-09-29T16:34:57Z on mojolearn-steward-do-amd, cpu (AMD EPYC 9575F 64-Core Processor))) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: n_components=10 (the cuML benchmark's PCA), whiten=False, random_state=7; ours and scikit-learn svd_solver='covariance_eigh'. Rows: big block, raw. Timed: fit.

mismatch: svd_solver: cuML has no 'covariance_eigh'; cuml-gpu runs svd_solver='full'

mismatch: torch-gpu: no estimator; covariance eigh written out, torch.manual_seed(7) (it draws nothing)

mismatch: random_state is read by none of the covariance solvers; set to 7 on every arm

config: cuML benchmark (RAPIDS), PCA (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### pca / taxi (rows full, shape 4000000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-grid-logs.txt (amd/a1676)`, ran on AMD Instinct MI325X (amd, DO) job a1676

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 7.6 | 7.6..7.6 | 1 | - | - | 7.6 | - | - (stored whole) | - | - | - | - | explained_variance_ratio_sum=0.999997 | - | main board, one scored run | - | ok (main@a1c96a5ba amd/a1676 2026-10-11; identity vs the nvidia columns: n/a) |
| torch-gpu | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 10.9 | 10.9..10.9 | 1 | 0.697 | - | 152.7 | 10.9 | 141.81 (upload_ms_untimed) | 0.050 (whole/whole (kernel not derivable)) | - | 3009.5 | 412.0 | explained_variance_ratio_sum=0.999997 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 82.2 | 82.2..82.2 | 1 | 0.092 | - | 82.2 | 82.2 | 0.00 (cpu-arm) | 0.092 (whole/whole) | - | 401.6 | - | explained_variance_ratio_sum=0.999995 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; stored (measured 2026-09-29T16:34:39Z on mojolearn-steward-do-amd, cpu (AMD EPYC 9575F 64-Core Processor))) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: n_components=10 (the cuML benchmark's PCA), whiten=False, random_state=7; ours and scikit-learn svd_solver='covariance_eigh'. Rows: big block, raw. Timed: fit.

mismatch: svd_solver: cuML has no 'covariance_eigh'; cuml-gpu runs svd_solver='full'

mismatch: torch-gpu: no estimator; covariance eigh written out, torch.manual_seed(7) (it draws nothing)

mismatch: random_state is read by none of the covariance solvers; set to 7 on every arm

config: cuML benchmark (RAPIDS), PCA (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### svc / istella (rows full, shape 10000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-grid-logs.txt (amd/a1708)`, ran on AMD Instinct MI325X (amd, DO) job a1708

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 51.7 | 51.7..51.7 | 1 | - | - | 51.7 | - | - (stored whole) | - | - | - | - | accuracy=0.922200, n_support=2400 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1708 2026-10-11; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 854.5 | 854.5..854.5 | 1 | 0.060 | - | 854.5 | 854.5 | 0.00 (cpu-arm) | 0.060 (whole/whole) | - | 335.4 | - | accuracy=0.922200, n_support=2400 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: C=1.0, kernel='rbf', gamma=1/d, tol=1e-3, degree=3, coef0=0.0, max_iter=-1, cache_size=2000 MB, class_weight=None. Rows: svc block: 10,000 fit rows, 10,000 eval rows, standardized. Timed: fit.

mismatch: seed: scikit-learn and cuML random_state=7 (read only with probability=True); ours refuses random_state without probability=True, so ours stays None

mismatch: cache_size=2000 on every arm; ours honors it only as the prediction buffer (DEVIATION 871), so its training is unaffected

mismatch: shrinking=True: scikit-learn only; nochange_steps=1000: ours and cuML only

config: cuML benchmark (RAPIDS), SVC-RBF (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### svc / taxi (rows full, shape 10000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-grid-logs.txt (amd/a1708)`, ran on AMD Instinct MI325X (amd, DO) job a1708

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 1511.4 | 1511.4..1511.4 | 1 | - | - | 1511.4 | - | - (stored whole) | - | - | - | - | accuracy=0.767500, n_support=5527 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1708 2026-10-11; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 1727.5 | 1727.5..1727.5 | 1 | 0.875 | - | 1727.5 | 1727.5 | 0.00 (cpu-arm) | 0.875 (whole/whole) | - | 455.3 | - | accuracy=0.767500, n_support=5675 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: C=1.0, kernel='rbf', gamma=1/d, tol=1e-3, degree=3, coef0=0.0, max_iter=-1, cache_size=2000 MB, class_weight=None. Rows: svc block: 10,000 fit rows, 10,000 eval rows, standardized. Timed: fit.

mismatch: seed: scikit-learn and cuML random_state=7 (read only with probability=True); ours refuses random_state without probability=True, so ours stays None

mismatch: cache_size=2000 on every arm; ours honors it only as the prediction buffer (DEVIATION 871), so its training is unaffected

mismatch: shrinking=True: scikit-learn only; nochange_steps=1000: ours and cuML only

config: cuML benchmark (RAPIDS), SVC-RBF (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

## Classical, wave 2

### elasticnet / istella (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a1161)`, ran on AMD Instinct MI325X (amd, DO) job a1161

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@0a7b206f1) | identical | 66.5 | 66.5..66.5 | 1 | - | - | 66.5 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.260922, rmse=0.718134 | - | main board, one scored run | - | ok (main@0a7b206f1 amd/a1161 2026-10-09; identity vs nvidia-l40s: MATCH) |

settings: alpha=0.1, l1_ratio=0.5 (the cuML benchmark's ElasticNet), fit_intercept=True, max_iter=1000, tol=1e-4, selection='cyclic', positive=False; ours solver='cd' and precompute=False (the only value it takes), scikit-learn precompute=True. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: precompute: ours is constructed precompute=False but its coordinate descent (IDENTICAL and FAST) forms X^T X and X^T y once and sweeps the Gram whenever solver/impl/cd_gram_rule.mojo's shape rule holds (tall, at most 256 columns: both datasets here); scikit-learn precompute=True is that algorithm. cuML is refused: its cd sweeps every row per coordinate and refuses precompute

mismatch: seed: ours refuses random_state (it selects nothing with selection='cyclic'); scikit-learn random_state=7

mismatch: tol: each library's own stopping rule reads it

mismatch: opponents withheld: the lane settings at this tree's HEAD differ from the settings release-board-resume-r2 recorded for its opponent race (an opponent job must score them again)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### elasticnet / taxi (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a1161)`, ran on AMD Instinct MI325X (amd, DO) job a1161

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@0a7b206f1) | identical | 5.0 | 5.0..5.0 | 1 | - | - | 5.0 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.907378, rmse=4.847225 | - | main board, one scored run | - | ok (main@0a7b206f1 amd/a1161 2026-10-09; identity vs nvidia-l40s: MATCH) |

settings: alpha=0.1, l1_ratio=0.5 (the cuML benchmark's ElasticNet), fit_intercept=True, max_iter=1000, tol=1e-4, selection='cyclic', positive=False; ours solver='cd' and precompute=False (the only value it takes), scikit-learn precompute=True. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: precompute: ours is constructed precompute=False but its coordinate descent (IDENTICAL and FAST) forms X^T X and X^T y once and sweeps the Gram whenever solver/impl/cd_gram_rule.mojo's shape rule holds (tall, at most 256 columns: both datasets here); scikit-learn precompute=True is that algorithm. cuML is refused: its cd sweeps every row per coordinate and refuses precompute

mismatch: seed: ours refuses random_state (it selects nothing with selection='cyclic'); scikit-learn random_state=7

mismatch: tol: each library's own stopping rule reads it

mismatch: opponents withheld: the lane settings at this tree's HEAD differ from the settings release-board-resume-r2 recorded for its opponent race (an opponent job must score them again)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gmm / istella (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/r43/ingest/amd-gridlogs.txt (amd/a1590)`, ran on AMD Instinct MI325X (amd, DO) job a1590

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.42 (source build, main@a6b5f76cb) | identical | 591.2 | 591.2..591.2 | 1 | - | - | 591.2 | - | - (stored whole) | - | - | - | - | bic=-3.851e+07, mean_log_likelihood=200.794500, n_iter=24 | - | main board, one scored run | - | ok (main@a6b5f76cb amd/a1590 2026-10-13; identity vs the nvidia columns: n/a) |

settings: n_components=8, covariance_type='full', tol=1e-3, reg_covar=1e-6 on taxi and 3e-3 on Istella-S (GMM_REG_COVAR), max_iter=100, init_params='kmeans', n_init=1, warm_start=False, random_state=7. Rows: 100000 fit and 20000 held-out stride rows of the reg block (standardized by the fit rows); an explicitly full_dataset_coverage recipe retains all fit/eval rows. Timed: fit.

mismatch: init_params='kmeans': each library seeds its own k-means (ours the identity-certified k-means, scikit-learn KMeans(n_init=1, k-means++))

mismatch: opponents withheld: the lane settings at this tree's HEAD differ from the settings release-board-resume-r2 recorded for its opponent race (an opponent job must score them again)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gmm / taxi (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/r43/ingest/amd-gridlogs.txt (amd/a1590)`, ran on AMD Instinct MI325X (amd, DO) job a1590

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.42 (source build, main@a6b5f76cb) | identical | 61.5 | 61.5..61.5 | 1 | - | - | 61.5 | - | - (stored whole) | - | - | - | - | bic=-3.668e+06, mean_log_likelihood=12.807640, n_iter=29 | - | main board, one scored run | - | ok (main@a6b5f76cb amd/a1590 2026-10-13; identity vs the nvidia columns: n/a) |

settings: n_components=8, covariance_type='full', tol=1e-3, reg_covar=1e-6 on taxi and 3e-3 on Istella-S (GMM_REG_COVAR), max_iter=100, init_params='kmeans', n_init=1, warm_start=False, random_state=7. Rows: 100000 fit and 20000 held-out stride rows of the reg block (standardized by the fit rows); an explicitly full_dataset_coverage recipe retains all fit/eval rows. Timed: fit.

mismatch: init_params='kmeans': each library seeds its own k-means (ours the identity-certified k-means, scikit-learn KMeans(n_init=1, k-means++))

mismatch: opponents withheld: the lane settings at this tree's HEAD differ from the settings release-board-resume-r2 recorded for its opponent race (an opponent job must score them again)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ivf / istella (rows full, shape index 400000x220; queries 4000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/amd_ingest/gridlogs.txt (amd/a0009)`, ran on AMD Instinct MI325X (amd, DO) job a0009

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@1658ba796) | identical | 196.0 | 196.0..196.0 | 1 | - | - | 196.0 | - | - (stored whole) | - | - | - | - | recall_at_k=0.999925, rows_with_repeated_ids=0 | - | main board, one scored run | - | ok (main@1658ba796 amd/a0009 2026-10-12; identity vs the nvidia columns: n/a) |
| faiss-cpu | faiss | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | faiss 1.15.1 | opponent | 3704.2 | 3704.2..3704.2 | 1 | 0.053 | - | 3704.2 | 3704.2 | 0.00 (cpu-arm) | 0.053 (whole/whole) | - | 1286.6 | - | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, faiss-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: IVF-Flat, n_lists=1024, n_probes=32, k=10, squared L2, k-means 20 iterations, seed 7. Rows: the classical knn lane's block (tools/knn_datasets.real_block): 400000 index rows, 4000 queries, raw. Timed: build + search of every query.

mismatch: quantizer training set: each library's own (FAISS subsamples to 256 rows per list; cuVS kmeans_trainset_fraction 0.5; ours its own)

mismatch: seed: ours random_state=7, faiss cp.seed=7; cuVS IndexParams takes none

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ivf / taxi (rows full, shape index 400000x11; queries 4000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/amd_ingest/gridlogs.txt (amd/a0009)`, ran on AMD Instinct MI325X (amd, DO) job a0009

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@1658ba796) | identical | 46.1 | 46.1..46.1 | 1 | - | - | 46.1 | - | - (stored whole) | - | - | - | - | recall_at_k=0.999650, rows_with_repeated_ids=0 | - | main board, one scored run | - | ok (main@1658ba796 amd/a0009 2026-10-12; identity vs the nvidia columns: n/a) |
| faiss-cpu | faiss | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | faiss 1.15.1 | opponent | 143.0 | 143.0..143.0 | 1 | 0.322 | - | 143.0 | 143.0 | 0.00 (cpu-arm) | 0.322 (whole/whole) | - | 139.0 | - | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, faiss-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: IVF-Flat, n_lists=1024, n_probes=32, k=10, squared L2, k-means 20 iterations, seed 7. Rows: the classical knn lane's block (tools/knn_datasets.real_block): 400000 index rows, 4000 queries, raw. Timed: build + search of every query.

mismatch: quantizer training set: each library's own (FAISS subsamples to 256 rows per list; cuVS kmeans_trainset_fraction 0.5; ours its own)

mismatch: seed: ours random_state=7, faiss cp.seed=7; cuVS IndexParams takes none

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lasso / istella (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a1199)`, ran on AMD Instinct MI325X (amd, DO) job a1199

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@8d5855b83) | identical | 83.3 | 83.3..83.3 | 1 | - | - | 83.3 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.310837, rmse=0.693460 | - | main board, one scored run | - | ok (main@8d5855b83 amd/a1199 2026-10-10; identity vs the nvidia columns: n/a) |

settings: alpha=0.01, fit_intercept=True, max_iter=1000, tol=1e-4, selection='cyclic', positive=False; ours solver='cd' and precompute=False (the only value it takes), scikit-learn precompute=True. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: precompute: ours is constructed precompute=False but its coordinate descent (IDENTICAL and FAST) forms X^T X and X^T y once and sweeps the Gram whenever solver/impl/cd_gram_rule.mojo's shape rule holds (tall, at most 256 columns: both datasets here); scikit-learn precompute=True is that algorithm. cuML is refused: its cd sweeps every row per coordinate and refuses precompute

mismatch: seed: ours refuses random_state (it selects nothing with selection='cyclic'); scikit-learn random_state=7

mismatch: tol: each library's own stopping rule reads it

mismatch: opponents withheld: the lane settings at this tree's HEAD differ from the settings release-board-resume-r2 recorded for its opponent race (an opponent job must score them again)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lasso / taxi (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a1199)`, ran on AMD Instinct MI325X (amd, DO) job a1199

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@8d5855b83) | identical | 5.0 | 5.0..5.0 | 1 | - | - | 5.0 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.908995, rmse=4.804744 | - | main board, one scored run | - | ok (main@8d5855b83 amd/a1199 2026-10-10; identity vs the nvidia columns: n/a) |

settings: alpha=0.01, fit_intercept=True, max_iter=1000, tol=1e-4, selection='cyclic', positive=False; ours solver='cd' and precompute=False (the only value it takes), scikit-learn precompute=True. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: precompute: ours is constructed precompute=False but its coordinate descent (IDENTICAL and FAST) forms X^T X and X^T y once and sweeps the Gram whenever solver/impl/cd_gram_rule.mojo's shape rule holds (tall, at most 256 columns: both datasets here); scikit-learn precompute=True is that algorithm. cuML is refused: its cd sweeps every row per coordinate and refuses precompute

mismatch: seed: ours refuses random_state (it selects nothing with selection='cyclic'); scikit-learn random_state=7

mismatch: tol: each library's own stopping rule reads it

mismatch: opponents withheld: the lane settings at this tree's HEAD differ from the settings release-board-resume-r2 recorded for its opponent race (an opponent job must score them again)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### logreg / istella (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a1161)`, ran on AMD Instinct MI325X (amd, DO) job a1161

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@0a7b206f1) | identical | 2121.1 | 2121.1..2121.1 | 1 | - | - | 2121.1 | - | - (stored whole) | - | - | - | - | accuracy=0.924590, logloss=0.181249, nonfinite_proba_rows=0 | - | main board, one scored run | - | ok (main@0a7b206f1 amd/a1161 2026-10-09; identity vs nvidia-l40s: MATCH) |

settings: penalty='l2', C=1.0, tol=1e-4, max_iter=1000, fit_intercept=True, class_weight=None; ours and cuML linesearch_max_iter=50, L-BFGS memory 5 (cuML lbfgs_memory=5; ours fixes 5); scikit-learn random_state=7. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: solver: ours 'qn' (L-BFGS, cuML's), scikit-learn 'lbfgs', cuML 'qn'; each library's own stopping rule reads tol

mismatch: seed: ours and cuML LogisticRegression have no seed argument

mismatch: l1_ratio: ours None, scikit-learn None or 0.0 (its l2 spelling from 1.8)

mismatch: opponents withheld: the lane settings at this tree's HEAD differ from the settings release-board-resume-r2 recorded for its opponent race (an opponent job must score them again)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### logreg / taxi (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a1161)`, ran on AMD Instinct MI325X (amd, DO) job a1161

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@0a7b206f1) | identical | 23.4 | 23.4..23.4 | 1 | - | - | 23.4 | - | - (stored whole) | - | - | - | - | accuracy=0.763350, logloss=0.538985, nonfinite_proba_rows=0 | - | main board, one scored run | - | ok (main@0a7b206f1 amd/a1161 2026-10-09; identity vs nvidia-l40s: MATCH) |

settings: penalty='l2', C=1.0, tol=1e-4, max_iter=1000, fit_intercept=True, class_weight=None; ours and cuML linesearch_max_iter=50, L-BFGS memory 5 (cuML lbfgs_memory=5; ours fixes 5); scikit-learn random_state=7. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: solver: ours 'qn' (L-BFGS, cuML's), scikit-learn 'lbfgs', cuML 'qn'; each library's own stopping rule reads tol

mismatch: seed: ours and cuML LogisticRegression have no seed argument

mismatch: l1_ratio: ours None, scikit-learn None or 0.0 (its l2 spelling from 1.8)

mismatch: opponents withheld: the lane settings at this tree's HEAD differ from the settings release-board-resume-r2 recorded for its opponent race (an opponent job must score them again)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ridge / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-grid-logs.txt (amd/a1716)`, ran on AMD Instinct MI325X (amd, DO) job a1716

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 419.7 | 419.7..419.7 | 1 | - | - | 419.7 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.328674, rmse=0.684427 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1716 2026-10-11; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 4288.8 | 4288.8..4288.8 | 1 | 0.098 | - | 4288.8 | 4288.8 | 0.00 (cpu-arm) | 0.098 (whole/whole) | - | 8688.9 | - | finite=True, r2=0.328676, rmse=0.684426 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: alpha=1.0, fit_intercept=True; scikit-learn positive=False, random_state=7. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: solver, named on every arm: ours and cuML 'eig' (eigendecomposition of the normal equations), scikit-learn 'cholesky' (it has no 'eig')

mismatch: seed: ours and cuML Ridge have no seed argument

config: cuML benchmark (RAPIDS), Ridge (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ridge / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-grid-logs.txt (amd/a1716)`, ran on AMD Instinct MI325X (amd, DO) job a1716

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 3.4 | 3.4..3.4 | 1 | - | - | 3.4 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.908983, rmse=4.805050 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1716 2026-10-11; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 33.2 | 33.2..33.2 | 1 | 0.103 | - | 33.2 | 33.2 | 0.00 (cpu-arm) | 0.103 (whole/whole) | - | 291.3 | - | finite=True, r2=0.908983, rmse=4.805056 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: alpha=1.0, fit_intercept=True; scikit-learn positive=False, random_state=7. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: solver, named on every arm: ours and cuML 'eig' (eigendecomposition of the normal equations), scikit-learn 'cholesky' (it has no 'eig')

mismatch: seed: ours and cuML Ridge have no seed argument

config: cuML benchmark (RAPIDS), Ridge (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### tsvd / istella (rows full, shape X 1000000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-grid-logs.txt (amd/a1676)`, ran on AMD Instinct MI325X (amd, DO) job a1676

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 29.6 | 29.6..29.6 | 1 | - | - | 29.6 | - | - (stored whole) | - | - | - | - | explained_variance_ratio_sum=1.000000, relative_reconstruction_error=0.0001314 | - | main board, one scored run | - | ok (main@a1c96a5ba amd/a1676 2026-10-11; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 689.8 | 689.8..689.8 | 1 | 0.043 | - | 689.8 | 689.8 | 0.00 (cpu-arm) | 0.043 (whole/whole) | - | 2422.2 | - | explained_variance_ratio_sum=1.000000, relative_reconstruction_error=0.000122 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: n_components=10 (the cuML benchmark's tSVD), tol=0.0, n_iter=5, n_oversamples=10, random_state=7. Rows: 1000000 stride rows of the train split, raw (sentinel cleaned, not scaled). Timed: fit.

mismatch: algorithm: ours 'covariance_eigh' (eigh of X^T X), scikit-learn 'arpack' (tol=0, exact to ARPACK's tolerance), cuML 'full'

config: cuML benchmark (RAPIDS), tSVD (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### tsvd / taxi (rows full, shape X 1000000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a1553)`, ran on AMD Instinct MI325X (amd, DO) job a1553

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@a8e0548a5) | identical | 3.9 | 3.9..3.9 | 1 | - | - | 3.9 | - | - (stored whole) | - | - | - | - | explained_variance_ratio_sum=0.999965, relative_reconstruction_error=0.003257 | - | main board, one scored run | - | ok (main@a8e0548a5 amd/a1553 2026-10-10; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 205.6 | 205.6..205.6 | 1 | 0.019 | - | 205.6 | 205.6 | 0.00 (cpu-arm) | 0.019 (whole/whole) | - | 460.6 | - | explained_variance_ratio_sum=0.999965, relative_reconstruction_error=0.003257 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: n_components=10 (the cuML benchmark's tSVD), tol=0.0, n_iter=5, n_oversamples=10, random_state=7. Rows: 1000000 stride rows of the train split, raw (sentinel cleaned, not scaled). Timed: fit.

mismatch: algorithm: ours 'covariance_eigh' (eigh of X^T X), scikit-learn 'arpack' (tol=0, exact to ARPACK's tolerance), cuML 'full'

config: cuML benchmark (RAPIDS), tSVD (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

## Neural

### gemm-bf16 / gaussian (neural shape full: 4096x4096x4096)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/full-gpu-repair-20261014/amd/repair/measured/neural.gemm-bf16.gaussian.shape=full/board.json (amd/paired-e05c3a1cfb08c926151e851a4030988a18137fd7630eedb0c69a9e63d2cabc33)`, ran on mojolearn-steward-do-amd

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 9.2 ms) = 2.513; fp32 twin: ours IDENTICAL / torch fp32: - (no completed torch fp32 arm). Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | 0.8.44 | identical | 23.1 | 23.1..23.1 | 1 | - | - | 23.1 | - | - (stored whole) | - | - | 2312.7 | 238239.5 | max_rel_err_vs_fp64=1.155e-07 | - | main board, one scored run | wheel | ok (main@78a5691bc amd/paired-e05c3a1cfb08c926151e851a4030988a18137fd7630eedb0c69a9e63d2cabc33 2026-10-14; identity vs the nvidia columns: n/a) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 9.2 | 9.2..9.2 | 1 | 2.513 | - | 9.2 | - | - (stored whole) | 2.513 (whole/whole (kernel not derivable)) | - | 2734.7 | 312.0 | max_abs_diff_vs_ours=0.999023, max_rel_diff_vs_ours=0.002759, max_rel_err_vs_fp64=0.002759 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 9.5 | 9.5..9.5 | 1 | 2.428 | - | 9.5 | - | - (stored whole) | 2.428 (whole/whole (kernel not derivable)) | - | 2941.5 | 312.0 | max_abs_diff_vs_ours=0.999023, max_rel_diff_vs_ours=0.002759, max_rel_err_vs_fp64=0.002759 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/89089/vram_* bytes for this pid at the round's end (not a peak)

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

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/measurement-readiness-20261013/amd/gemm/board.json (amd/paired-5a3c9037c17666d2bbaa35e9ed11d6e3216888944a408f6873cae86e5e99fcc1)`, ran on mojolearn-steward-do-amd

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 10.8 ms) = 2.646; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 11.7 ms) = 2.452. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | 0.8.44 | identical | 28.6 | 28.6..28.6 | 1 | - | - | 28.6 | - | - (stored whole) | - | - | 2283.4 | 3692.3 | max_rel_err_vs_fp64=2.399e-07 | - | main board, one scored run | wheel | ok (main@18679c3bb amd/paired-5a3c9037c17666d2bbaa35e9ed11d6e3216888944a408f6873cae86e5e99fcc1 2026-10-14; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 11.8 | 11.8..11.8 | 1 | 2.414 | - | 11.8 | - | - (stored whole) | 2.414 (whole/whole (kernel not derivable)) | - | 1906.3 | 268.0 | max_abs_diff_vs_ours=0.001038, max_rel_diff_vs_ours=2.866e-06, max_rel_err_vs_fp64=2.802e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 11.7 | 11.7..11.7 | 1 | 2.452 | - | 11.7 | - | - (stored whole) | 2.452 (whole/whole (kernel not derivable)) | - | 2066.8 | 268.0 | max_abs_diff_vs_ours=0.001038, max_rel_diff_vs_ours=2.866e-06, max_rel_err_vs_fp64=2.802e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 10.8 | 10.8..10.8 | 1 | 2.646 | - | 10.8 | - | - (stored whole) | 2.646 (whole/whole (kernel not derivable)) | - | 2685.9 | 376.0 | max_abs_diff_vs_ours=1.358337, max_rel_diff_vs_ours=0.003751, max_rel_err_vs_fp64=0.003752 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 11.7 | 11.7..11.7 | 1 | 2.440 | - | 11.7 | - | - (stored whole) | 2.440 (whole/whole (kernel not derivable)) | - | 3128.1 | 376.0 | max_abs_diff_vs_ours=1.358337, max_rel_diff_vs_ours=0.003751, max_rel_err_vs_fp64=0.003752 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/11196/vram_* bytes for this pid at the round's end (not a peak)

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

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-grid-logs.txt (amd/a1697)`, ran on AMD Instinct MI325X (amd, DO) job a1697

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 13.3 ms) = 2.533; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 14.5 ms) = 2.318. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 33.6 | 33.6..33.6 | 1 | - | - | 33.6 | - | - (stored whole) | - | - | - | - | mean_nll=9.018733 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1697 2026-10-11; identity vs the nvidia columns: n/a) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 17.6 | 17.6..17.6 | 1 | 1.911 | - | 17.6 | - | - (stored whole) | 1.911 (whole/whole (kernel not derivable)) | - | 3122.2 | 241.0 | mean_nll=9.018733 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 13.3 | 13.3..13.3 | 1 | 2.533 | - | 13.3 | - | - (stored whole) | 2.533 (whole/whole (kernel not derivable)) | - | 4342.2 | 240.5 | mean_nll=9.018647 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 14.5 | 14.5..14.5 | 1 | 2.318 | - | 14.5 | - | - (stored whole) | 2.318 (whole/whole (kernel not derivable)) | - | 3217.8 | 222.5 | mean_nll=9.018733 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 13.8 | 13.8..13.8 | 1 | 2.433 | - | 13.8 | - | - (stored whole) | 2.433 (whole/whole (kernel not derivable)) | - | 4448.6 | 194.5 | mean_nll=9.018664 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-eager-bf16, torch-compile-fp32, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lm-train-step / bytes (neural shape full: B1 L2048 DM384 H6 KV6 HD64 FF1024 layers8 V8192)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-grid-logs.txt (amd/a1697)`, ran on AMD Instinct MI325X (amd, DO) job a1697

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 8.2 ms) = 5.254; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 15.2 ms) = 2.830. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 43.0 | 43.0..43.0 | 1 | - | - | 43.0 | - | - (stored whole) | - | - | - | - | loss_first_step=9.018733, loss_last_step=8.418446, steps=2 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1697 2026-10-11; identity vs the nvidia columns: n/a) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 18.2 | 18.2..18.2 | 1 | 2.357 | - | 18.2 | - | - (stored whole) | 2.357 (whole/whole (kernel not derivable)) | - | 3311.5 | 931.1 | loss_first_step=9.018732, loss_last_step=8.418447, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 15.2 | 15.2..15.2 | 1 | 2.830 | - | 15.2 | - | - (stored whole) | 2.830 (whole/whole (kernel not derivable)) | - | 3359.7 | 780.1 | loss_first_step=9.018733, loss_last_step=8.418446, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 13.0 | 13.0..13.0 | 1 | 3.310 | - | 13.0 | - | - (stored whole) | 3.310 (whole/whole (kernel not derivable)) | - | 4897.8 | 774.1 | loss_first_step=9.018646, loss_last_step=8.416107, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 8.2 | 8.2..8.2 | 1 | 5.254 | - | 8.2 | - | - (stored whole) | 5.254 (whole/whole (kernel not derivable)) | - | 4959.4 | 602.6 | loss_first_step=9.018663, loss_last_step=8.415815, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### mamba1-forward / gaussian (neural shape full: B1 L2048 DM384)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-grid-logs.txt (amd/a1652)`, ran on AMD Instinct MI325X (amd, DO) job a1652

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 105.1 ms) = 0.030; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 87.6 ms) = 0.036. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@2afa942e3) | identical | 3.1 | 3.1..3.1 | 1 | - | - | 3.1 | - | - (stored whole) | - | - | - | - | - | - | main board, one scored run | - | ok (main@2afa942e3 amd/a1652 2026-10-11; identity vs the nvidia columns: n/a) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 87.6 | 87.6..87.6 | 1 | 0.036 | - | 87.6 | - | - (stored whole) | 0.036 (whole/whole (kernel not derivable)) | - | 3407.3 | 474.2 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 105.1 | 105.1..105.1 | 1 | 0.030 | - | 105.1 | - | - (stored whole) | 0.030 (whole/whole (kernel not derivable)) | - | 4676.6 | 418.5 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-eager-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### mamba2-forward / gaussian (neural shape full: B1 L2048 DM384)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-grid-logs.txt (amd/a1697)`, ran on AMD Instinct MI325X (amd, DO) job a1697

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 11.5 ms) = 0.384; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 12.7 ms) = 0.347. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 4.4 | 4.4..4.4 | 1 | - | - | 4.4 | - | - (stored whole) | - | - | - | - | - | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1697 2026-10-11; identity vs the nvidia columns: n/a) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 13.7 | 13.7..13.7 | 1 | 0.322 | - | 13.7 | - | - (stored whole) | 0.322 (whole/whole (kernel not derivable)) | - | 3088.4 | 3221.3 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 12.7 | 12.7..12.7 | 1 | 0.347 | - | 12.7 | - | - (stored whole) | 0.347 (whole/whole (kernel not derivable)) | - | 3085.8 | 3221.3 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 13.4 | 13.4..13.4 | 1 | 0.329 | - | 13.4 | - | - (stored whole) | 0.329 (whole/whole (kernel not derivable)) | - | 4353.3 | 1676.6 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 11.5 | 11.5..11.5 | 1 | 0.384 | - | 11.5 | - | - (stored whole) | 0.384 (whole/whole (kernel not derivable)) | - | 4340.2 | 1676.6 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### mlp-train-step / gaussian (neural shape full: rows256 8-16-3)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-grid-logs.txt (amd/a1697)`, ran on AMD Instinct MI325X (amd, DO) job a1697

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 1.5 ms) = 2.304; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 1.3 ms) = 2.578. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 3.4 | 3.4..3.4 | 1 | - | - | 3.4 | - | - (stored whole) | - | - | - | - | loss_first_step=1.160401, loss_last_step=1.123361, steps=2 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1697 2026-10-11; identity vs the nvidia columns: n/a) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1.3 | 1.3..1.3 | 1 | 2.578 | - | 1.3 | - | - (stored whole) | 2.578 (whole/whole (kernel not derivable)) | - | 2955.8 | 76.1 | loss_first_step=1.160401, loss_last_step=1.123361, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 2.1 | 2.1..2.1 | 1 | 1.597 | - | 2.1 | - | - (stored whole) | 1.597 (whole/whole (kernel not derivable)) | - | 3007.0 | 76.1 | loss_first_step=1.160401, loss_last_step=1.123361, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1.9 | 1.9..1.9 | 1 | 1.810 | - | 1.9 | - | - (stored whole) | 1.810 (whole/whole (kernel not derivable)) | - | 4617.8 | 76.0 | loss_first_step=1.160498, loss_last_step=1.123461, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1.5 | 1.5..1.5 | 1 | 2.304 | - | 1.5 | - | - (stored whole) | 2.304 (whole/whole (kernel not derivable)) | - | 4698.8 | 76.0 | loss_first_step=1.160367, loss_last_step=1.123438, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### samba-forward / bytes (neural shape full: B2 L512 DM384 V256 H6 FF1024 layers mamba3+attention+mamba3+attention)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a1189)`, ran on AMD Instinct MI325X (amd, DO) job a1189

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 5.7 ms) = 0.994; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 5.9 ms) = 0.953. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@9f83ea479) | identical | 5.6 | 5.6..5.6 | 1 | - | - | 5.6 | - | - (stored whole) | - | - | - | - | mean_nll=5.635910 | - | main board, one scored run | - | ok (main@9f83ea479 amd/a1189 2026-10-10; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 40.2 | 40.2..40.2 | 1 | 0.140 | - | 40.2 | - | - (stored whole) | 0.140 (whole/whole (kernel not derivable)) | - | 3041.1 | 203.9 | mean_nll=5.635910 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 5.9 | 5.9..5.9 | 1 | 0.953 | - | 5.9 | - | - (stored whole) | 0.953 (whole/whole (kernel not derivable)) | - | 3698.7 | 173.9 | mean_nll=5.635910 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 40.9 | 40.9..40.9 | 1 | 0.138 | - | 40.9 | - | - (stored whole) | 0.138 (whole/whole (kernel not derivable)) | - | 4831.9 | 201.7 | mean_nll=5.635892 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 5.7 | 5.7..5.7 | 1 | 0.994 | - | 5.7 | - | - (stored whole) | 0.994 (whole/whole (kernel not derivable)) | - | 5545.2 | 156.2 | mean_nll=5.635953 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### samba-train-step / bytes (neural shape full: B2 L512 DM384 V256 H6 FF1024 layers mamba3+attention+mamba3+attention)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-grid-logs.txt (amd/a1649)`, ran on AMD Instinct MI325X (amd, DO) job a1649

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 127.6 ms) = 0.541; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 135.2 ms) = 0.511. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@2afa942e3) | identical | 69.0 | 69.0..69.0 | 1 | - | - | 69.0 | - | - (stored whole) | - | - | - | - | loss_first_step=5.635910, loss_last_step=4.833934, steps=2 | - | main board, one scored run | - | ok (main@2afa942e3 amd/a1649 2026-10-11; identity vs the nvidia columns: n/a) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 135.2 | 135.2..135.2 | 1 | 0.511 | - | 135.2 | - | - (stored whole) | 0.511 (whole/whole (kernel not derivable)) | - | 3267.0 | 432.1 | loss_first_step=5.635909, loss_last_step=4.833934, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(timeout: null) (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 127.6 | 127.6..127.6 | 1 | 0.541 | - | 127.6 | - | - (stored whole) | 0.541 (whole/whole (kernel not derivable)) | - | 4934.6 | 384.6 | loss_first_step=5.635892, loss_last_step=4.834043, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(timeout: null) (copied from opponents-20261006; measured this run) |

memory, ours, torch-compile-fp32, torch-compile-bf16: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-eager-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### transformer-forward / gaussian (neural shape full: B1 L2048 DM384 H6 KV6 HD64 FF1024)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-grid-logs.txt (amd/a1697)`, ran on AMD Instinct MI325X (amd, DO) job a1697

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 1.9 ms) = 1.660; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 1.6 ms) = 2.045. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 3.2 | 3.2..3.2 | 1 | - | - | 3.2 | - | - (stored whole) | - | - | - | - | - | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1697 2026-10-11; identity vs the nvidia columns: n/a) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1.8 | 1.8..1.8 | 1 | 1.747 | - | 1.8 | - | - (stored whole) | 1.747 (whole/whole (kernel not derivable)) | - | 2873.1 | 147.3 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1.6 | 1.6..1.6 | 1 | 2.045 | - | 1.6 | - | - (stored whole) | 2.045 (whole/whole (kernel not derivable)) | - | 2982.1 | 122.3 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1.9 | 1.9..1.9 | 1 | 1.660 | - | 1.9 | - | - (stored whole) | 1.660 (whole/whole (kernel not derivable)) | - | 4144.3 | 139.1 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(error: {"error": "REFUSED: torch-compile-bf16 on cuda failed in round 0 (compile happens here): RuntimeError('self and mat2 must have the same dtype, but got Float and BFloat16')", "event": "error", "stage":) (copied from opponents-20261006; measured this run) |

memory, ours, torch-compile-bf16: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

## Algorithm expansion

### adafactor / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/amd/campaign/measured/algos.adafactor.synthetic.rows=full/board.json (amd/paired-f10cf2497d39cc7319d4de70c12bf716db9e88b0028ecc67a6aa49070a81732e)`, ran on mojolearn-steward-do-amd

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 7.2 ms) = 1.130. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | 0.8.44 | identical | 8.2 | 8.2..8.2 | 1 | - | - | - | 8.2 | - (stored kernel) | - | - | 1722.4 | 235846.9 | optimizer_update_error_vs_torch_fp32=3.386e-06, rel_fro_vs_torch_eager_fp32=1.033e-08 | - | main board, one scored run | wheel | ok (main@79fcd24a5 amd/paired-f10cf2497d39cc7319d4de70c12bf716db9e88b0028ecc67a6aa49070a81732e 2026-10-14; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 7.2 | 7.2..7.2 | 1 | 1.130 | - | - | 7.2 | - (stored kernel) | 1.130 (kernel/kernel) | - | 2785.7 | 832.0 | optimizer_update_error_vs_torch_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 104.5 | 104.5..104.5 | 1 | 0.078 | - | - | 104.5 | - (stored kernel) | 0.078 (kernel/kernel) | - | 2840.6 | 832.0 | optimizer_update_error_vs_torch_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/102984/vram_* bytes for this pid at the round's end (not a peak)

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

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/amd/campaign/measured/algos.adagrad.synthetic.rows=full/board.json (amd/paired-af94094fad922b4efc0abda02c1eb1d9d018067c8743a9f2fb66108edcd28323)`, ran on mojolearn-steward-do-amd

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 2.6 ms) = 2.037. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | 0.8.44 | identical | 5.2 | 5.2..5.2 | 1 | - | - | - | 5.2 | - (stored kernel) | - | - | 1528.4 | 235669.5 | optimizer_update_error_vs_torch_fp32=1.874e-06, rel_fro_vs_torch_eager_fp32=3.207e-09 | - | main board, one scored run | wheel | ok (main@79fcd24a5 amd/paired-af94094fad922b4efc0abda02c1eb1d9d018067c8743a9f2fb66108edcd28323 2026-10-14; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 4.4 | 4.4..4.4 | 1 | 1.197 | - | - | 4.4 | - (stored kernel) | 1.197 (kernel/kernel) | - | 2742.2 | 960.0 | optimizer_update_error_vs_torch_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 2.6 | 2.6..2.6 | 1 | 2.037 | - | - | 2.6 | - (stored kernel) | 2.037 (kernel/kernel) | - | 3089.0 | 896.0 | optimizer_update_error_vs_torch_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/67193/vram_* bytes for this pid at the round's end (not a peak)

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
| maximize | - | false | false |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| weight_decay | 0.0 | 0.0 | 0.0 |

accepted difference: torch-eager-fp32 differentiable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-eager-fp32 foreach: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

accepted difference: torch-eager-fp32 fused: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

accepted difference: torch-eager-fp32 maximize: both standalone optimizers minimize; our sequence optimizer rejects maximize=True

accepted difference: torch-compile-fp32 differentiable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-compile-fp32 foreach: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

accepted difference: torch-compile-fp32 fused: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

accepted difference: torch-compile-fp32 maximize: both standalone optimizers minimize; our sequence optimizer rejects maximize=True

### adam / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/amd/campaign/measured/algos.adam.synthetic.rows=full/board.json (amd/paired-b7d8836277782a06993d4691465ee382d5c783cad374eaa186780d08b0bc34e8)`, ran on mojolearn-steward-do-amd

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 5.9 ms) = 1.953. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | 0.8.44 | identical | 11.6 | 11.6..11.6 | 1 | - | - | - | 11.6 | - (stored kernel) | - | - | 3033.8 | 235846.9 | optimizer_update_error_vs_torch_fp32=8.844e-06, rel_fro_vs_torch_eager_fp32=3.34e-08 | - | main board, one scored run | wheel | ok (main@79fcd24a5 amd/paired-b7d8836277782a06993d4691465ee382d5c783cad374eaa186780d08b0bc34e8 2026-10-14; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 5.9 | 5.9..5.9 | 1 | 1.953 | - | - | 5.9 | - (stored kernel) | 1.953 (kernel/kernel) | - | 2782.5 | 896.0 | optimizer_update_error_vs_torch_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 53.6 | 53.6..53.6 | 1 | 0.216 | - | - | 53.6 | - (stored kernel) | 0.216 (kernel/kernel) | - | 3227.1 | 960.0 | optimizer_update_error_vs_torch_fp32=1.355e-05, rel_fro_vs_torch_eager_fp32=5.116e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-fused-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 4.0 | 4.0..4.0 | 1 | 2.894 | - | - | 4.0 | - (stored kernel) | 2.894 (kernel/kernel) | - | 2703.8 | 896.0 | optimizer_update_error_vs_torch_fp32=6.002e-06, rel_fro_vs_torch_eager_fp32=2.267e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/10718/vram_* bytes for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-fused-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'betas': [0.9, 0.999], 'eps': 1e-08, 'lr': 0.001, 'maximize': False, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-fp32 | torch-eager-fp32 | torch-fused-fp32 |
|---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (optimizer.defaults) | torch (optimizer.defaults) | torch (optimizer.defaults) |
| amsgrad | - | false | false | false |
| betas | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| capturable | - | false | false | false |
| decoupled_weight_decay | - | false | false | false |
| differentiable | - | false | false | false |
| eps | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| foreach | - | true | true | null |
| fused | - | null | null | true |
| input_arrays | [{"dtype": "float32", "name": "p0", "sha256": "485a7287793a9d7d33d710ea2820ea9c7519d31100db4bafeee813baa532dd7d", "shape": [16777216]}, {"dtype": "float32", "name": "grad0", "sha256": "e6c0159d56a2df748f38d3f3592321edc11067354f52e751e382384dca4178bc", "shape": [16777216]}, {"dtype": "float32", "name": "grad1", "sha256": "249bb61249ee25b0cd42e6bcaa012c27c565dde4ccd798448a60e935fd4ace7e", "shape": [16777216]}, {"dtype": "float32", "name": "grad2", "sha256": "4cf167fba649d3d82fdb6c7c41749a74e8fc8a9bd70491154f03bfc9272b2ed6", "shape": [16777216]}, {"dtype": "float32", "name": "grad3", "sha256": "594f963cfcef38863774b145fac84f56ae7f37190af1cf270e7cb23d3b86b587", "shape": [16777216]}, {"dtype": "float32", "name": "grad4", "sha256": "ee091b2ac84630d1d0328c337b12c5bb6bd6ba2b28f3350eab106bc848960f4e", "shape": [16777216]}, {"dtype": "float32", "name": "grad5", "sha256": "a042832a1a0884016cafe5aada2f1df6bf0b09422e9358536881a536fff70d02", "shape": [16777216]}, {"dtype": "float32", "name": "grad6", "sha256": "736aade529a9b43a7666f9d5a77bf9ef70c253bb5a26280143d941c915910763", "shape": [16777216]}, {"dtype": "float32", "name": "grad7", "sha256": "c4fdaf0915399dc3568c7fd4a49465f9e3c69a10d3e77b1fc15d926db244f302", "shape": [16777216]}, {"dtype": "float32", "name": "grad8", "sha256": "6b970f1bc919f22be37581526578d65a535e33dd886ff46cc64d95aca61fec12", "shape": [16777216]}, {"dtype": "float32", "name": "grad9", "sha256": "00adaa7225e175a85e28e59772e3afbb901e068e5ca8d7332adfa127f7fcbdd7", "shape": [16777216]}] | [{"dtype": "float32", "name": "p0", "sha256": "485a7287793a9d7d33d710ea2820ea9c7519d31100db4bafeee813baa532dd7d", "shape": [16777216]}, {"dtype": "float32", "name": "grad0", "sha256": "e6c0159d56a2df748f38d3f3592321edc11067354f52e751e382384dca4178bc", "shape": [16777216]}, {"dtype": "float32", "name": "grad1", "sha256": "249bb61249ee25b0cd42e6bcaa012c27c565dde4ccd798448a60e935fd4ace7e", "shape": [16777216]}, {"dtype": "float32", "name": "grad2", "sha256": "4cf167fba649d3d82fdb6c7c41749a74e8fc8a9bd70491154f03bfc9272b2ed6", "shape": [16777216]}, {"dtype": "float32", "name": "grad3", "sha256": "594f963cfcef38863774b145fac84f56ae7f37190af1cf270e7cb23d3b86b587", "shape": [16777216]}, {"dtype": "float32", "name": "grad4", "sha256": "ee091b2ac84630d1d0328c337b12c5bb6bd6ba2b28f3350eab106bc848960f4e", "shape": [16777216]}, {"dtype": "float32", "name": "grad5", "sha256": "a042832a1a0884016cafe5aada2f1df6bf0b09422e9358536881a536fff70d02", "shape": [16777216]}, {"dtype": "float32", "name": "grad6", "sha256": "736aade529a9b43a7666f9d5a77bf9ef70c253bb5a26280143d941c915910763", "shape": [16777216]}, {"dtype": "float32", "name": "grad7", "sha256": "c4fdaf0915399dc3568c7fd4a49465f9e3c69a10d3e77b1fc15d926db244f302", "shape": [16777216]}, {"dtype": "float32", "name": "grad8", "sha256": "6b970f1bc919f22be37581526578d65a535e33dd886ff46cc64d95aca61fec12", "shape": [16777216]}, {"dtype": "float32", "name": "grad9", "sha256": "00adaa7225e175a85e28e59772e3afbb901e068e5ca8d7332adfa127f7fcbdd7", "shape": [16777216]}] | [{"dtype": "float32", "name": "p0", "sha256": "485a7287793a9d7d33d710ea2820ea9c7519d31100db4bafeee813baa532dd7d", "shape": [16777216]}, {"dtype": "float32", "name": "grad0", "sha256": "e6c0159d56a2df748f38d3f3592321edc11067354f52e751e382384dca4178bc", "shape": [16777216]}, {"dtype": "float32", "name": "grad1", "sha256": "249bb61249ee25b0cd42e6bcaa012c27c565dde4ccd798448a60e935fd4ace7e", "shape": [16777216]}, {"dtype": "float32", "name": "grad2", "sha256": "4cf167fba649d3d82fdb6c7c41749a74e8fc8a9bd70491154f03bfc9272b2ed6", "shape": [16777216]}, {"dtype": "float32", "name": "grad3", "sha256": "594f963cfcef38863774b145fac84f56ae7f37190af1cf270e7cb23d3b86b587", "shape": [16777216]}, {"dtype": "float32", "name": "grad4", "sha256": "ee091b2ac84630d1d0328c337b12c5bb6bd6ba2b28f3350eab106bc848960f4e", "shape": [16777216]}, {"dtype": "float32", "name": "grad5", "sha256": "a042832a1a0884016cafe5aada2f1df6bf0b09422e9358536881a536fff70d02", "shape": [16777216]}, {"dtype": "float32", "name": "grad6", "sha256": "736aade529a9b43a7666f9d5a77bf9ef70c253bb5a26280143d941c915910763", "shape": [16777216]}, {"dtype": "float32", "name": "grad7", "sha256": "c4fdaf0915399dc3568c7fd4a49465f9e3c69a10d3e77b1fc15d926db244f302", "shape": [16777216]}, {"dtype": "float32", "name": "grad8", "sha256": "6b970f1bc919f22be37581526578d65a535e33dd886ff46cc64d95aca61fec12", "shape": [16777216]}, {"dtype": "float32", "name": "grad9", "sha256": "00adaa7225e175a85e28e59772e3afbb901e068e5ca8d7332adfa127f7fcbdd7", "shape": [16777216]}] | [{"dtype": "float32", "name": "p0", "sha256": "485a7287793a9d7d33d710ea2820ea9c7519d31100db4bafeee813baa532dd7d", "shape": [16777216]}, {"dtype": "float32", "name": "grad0", "sha256": "e6c0159d56a2df748f38d3f3592321edc11067354f52e751e382384dca4178bc", "shape": [16777216]}, {"dtype": "float32", "name": "grad1", "sha256": "249bb61249ee25b0cd42e6bcaa012c27c565dde4ccd798448a60e935fd4ace7e", "shape": [16777216]}, {"dtype": "float32", "name": "grad2", "sha256": "4cf167fba649d3d82fdb6c7c41749a74e8fc8a9bd70491154f03bfc9272b2ed6", "shape": [16777216]}, {"dtype": "float32", "name": "grad3", "sha256": "594f963cfcef38863774b145fac84f56ae7f37190af1cf270e7cb23d3b86b587", "shape": [16777216]}, {"dtype": "float32", "name": "grad4", "sha256": "ee091b2ac84630d1d0328c337b12c5bb6bd6ba2b28f3350eab106bc848960f4e", "shape": [16777216]}, {"dtype": "float32", "name": "grad5", "sha256": "a042832a1a0884016cafe5aada2f1df6bf0b09422e9358536881a536fff70d02", "shape": [16777216]}, {"dtype": "float32", "name": "grad6", "sha256": "736aade529a9b43a7666f9d5a77bf9ef70c253bb5a26280143d941c915910763", "shape": [16777216]}, {"dtype": "float32", "name": "grad7", "sha256": "c4fdaf0915399dc3568c7fd4a49465f9e3c69a10d3e77b1fc15d926db244f302", "shape": [16777216]}, {"dtype": "float32", "name": "grad8", "sha256": "6b970f1bc919f22be37581526578d65a535e33dd886ff46cc64d95aca61fec12", "shape": [16777216]}, {"dtype": "float32", "name": "grad9", "sha256": "00adaa7225e175a85e28e59772e3afbb901e068e5ca8d7332adfa127f7fcbdd7", "shape": [16777216]}] |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 |
| maximize | false | false | false | false |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| weight_decay | 0.0 | 0.0 | 0.0 | 0.0 |

accepted difference: torch-eager-fp32 amsgrad: ordinary Adam moments on both arms; AMSGrad is disabled and our API does not implement that variant

accepted difference: torch-eager-fp32 capturable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-eager-fp32 decoupled_weight_decay: Adam uses coupled decay and AdamW uses decoupled decay; our class fixes that choice (_training_impl.py _KIND)

accepted difference: torch-eager-fp32 differentiable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-eager-fp32 foreach: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

accepted difference: torch-eager-fp32 fused: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

accepted difference: torch-compile-fp32 amsgrad: ordinary Adam moments on both arms; AMSGrad is disabled and our API does not implement that variant

accepted difference: torch-compile-fp32 capturable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-compile-fp32 decoupled_weight_decay: Adam uses coupled decay and AdamW uses decoupled decay; our class fixes that choice (_training_impl.py _KIND)

accepted difference: torch-compile-fp32 differentiable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-compile-fp32 foreach: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

accepted difference: torch-compile-fp32 fused: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

accepted difference: torch-fused-fp32 amsgrad: ordinary Adam moments on both arms; AMSGrad is disabled and our API does not implement that variant

accepted difference: torch-fused-fp32 capturable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-fused-fp32 decoupled_weight_decay: Adam uses coupled decay and AdamW uses decoupled decay; our class fixes that choice (_training_impl.py _KIND)

accepted difference: torch-fused-fp32 differentiable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-fused-fp32 foreach: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

accepted difference: torch-fused-fp32 fused: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

### adamax / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/amd/campaign/measured/algos.adamax.synthetic.rows=full/board.json (amd/paired-bb0b9e92ff74e892d405f1ce25ae5c6a696f6a475ef4e3b370133d8c9eb61ca3)`, ran on mojolearn-steward-do-amd

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 5.6 ms) = 1.626. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | 0.8.44 | identical | 9.1 | 9.1..9.1 | 1 | - | - | - | 9.1 | - (stored kernel) | - | - | 1528.3 | 235669.5 | optimizer_update_error_vs_torch_fp32=1.54e-06, rel_fro_vs_torch_eager_fp32=4.144e-09 | - | main board, one scored run | wheel | ok (main@79fcd24a5 amd/paired-bb0b9e92ff74e892d405f1ce25ae5c6a696f6a475ef4e3b370133d8c9eb61ca3 2026-10-14; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 5.6 | 5.6..5.6 | 1 | 1.626 | - | - | 5.6 | - (stored kernel) | 1.626 (kernel/kernel) | - | 2752.0 | 896.0 | optimizer_update_error_vs_torch_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 47.9 | 47.9..47.9 | 1 | 0.190 | - | - | 47.9 | - (stored kernel) | 0.190 (kernel/kernel) | - | 3221.6 | 960.0 | optimizer_update_error_vs_torch_fp32=2.292e-06, rel_fro_vs_torch_eager_fp32=6.17e-09 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/101046/vram_* bytes for this pid at the round's end (not a peak)

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
| maximize | - | false | false |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| weight_decay | 0.0 | 0.0 | 0.0 |

accepted difference: torch-eager-fp32 capturable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-eager-fp32 differentiable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-eager-fp32 foreach: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

accepted difference: torch-eager-fp32 maximize: both standalone optimizers minimize; our sequence optimizer rejects maximize=True

accepted difference: torch-compile-fp32 capturable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-compile-fp32 differentiable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-compile-fp32 foreach: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

accepted difference: torch-compile-fp32 maximize: both standalone optimizers minimize; our sequence optimizer rejects maximize=True

### adamw / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/amd/campaign/measured/algos.adamw.synthetic.rows=full/board.json (amd/paired-a66aa7eff428a6de1c4a5f3040f82126bf28db7ef40c0925793a4bde8b639da5)`, ran on mojolearn-steward-do-amd

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 6.5 ms) = 1.788. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | 0.8.44 | identical | 11.7 | 11.7..11.7 | 1 | - | - | - | 11.7 | - (stored kernel) | - | - | 3032.6 | 235846.9 | optimizer_update_error_vs_torch_fp32=8.841e-06, rel_fro_vs_torch_eager_fp32=3.34e-08 | - | main board, one scored run | wheel | ok (main@79fcd24a5 amd/paired-a66aa7eff428a6de1c4a5f3040f82126bf28db7ef40c0925793a4bde8b639da5 2026-10-14; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 6.5 | 6.5..6.5 | 1 | 1.788 | - | - | 6.5 | - (stored kernel) | 1.788 (kernel/kernel) | - | 2782.5 | 896.0 | optimizer_update_error_vs_torch_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 54.6 | 54.6..54.6 | 1 | 0.214 | - | - | 54.6 | - (stored kernel) | 0.214 (kernel/kernel) | - | 3223.3 | 960.0 | optimizer_update_error_vs_torch_fp32=6.012e-05, rel_fro_vs_torch_eager_fp32=2.272e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-fused-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 3.5 | 3.5..3.5 | 1 | 3.373 | - | - | 3.5 | - (stored kernel) | 3.373 (kernel/kernel) | - | 2703.8 | 896.0 | optimizer_update_error_vs_torch_fp32=6.574e-05, rel_fro_vs_torch_eager_fp32=2.484e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/11538/vram_* bytes for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-fused-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'betas': [0.9, 0.999], 'eps': 1e-08, 'lr': 0.001, 'maximize': False, 'weight_decay': 0.01}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-fp32 | torch-eager-fp32 | torch-fused-fp32 |
|---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (optimizer.defaults) | torch (optimizer.defaults) | torch (optimizer.defaults) |
| amsgrad | - | false | false | false |
| betas | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| capturable | - | false | false | false |
| decoupled_weight_decay | - | true | true | true |
| differentiable | - | false | false | false |
| eps | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| foreach | - | true | true | null |
| fused | - | null | null | true |
| input_arrays | [{"dtype": "float32", "name": "p0", "sha256": "485a7287793a9d7d33d710ea2820ea9c7519d31100db4bafeee813baa532dd7d", "shape": [16777216]}, {"dtype": "float32", "name": "grad0", "sha256": "e6c0159d56a2df748f38d3f3592321edc11067354f52e751e382384dca4178bc", "shape": [16777216]}, {"dtype": "float32", "name": "grad1", "sha256": "249bb61249ee25b0cd42e6bcaa012c27c565dde4ccd798448a60e935fd4ace7e", "shape": [16777216]}, {"dtype": "float32", "name": "grad2", "sha256": "4cf167fba649d3d82fdb6c7c41749a74e8fc8a9bd70491154f03bfc9272b2ed6", "shape": [16777216]}, {"dtype": "float32", "name": "grad3", "sha256": "594f963cfcef38863774b145fac84f56ae7f37190af1cf270e7cb23d3b86b587", "shape": [16777216]}, {"dtype": "float32", "name": "grad4", "sha256": "ee091b2ac84630d1d0328c337b12c5bb6bd6ba2b28f3350eab106bc848960f4e", "shape": [16777216]}, {"dtype": "float32", "name": "grad5", "sha256": "a042832a1a0884016cafe5aada2f1df6bf0b09422e9358536881a536fff70d02", "shape": [16777216]}, {"dtype": "float32", "name": "grad6", "sha256": "736aade529a9b43a7666f9d5a77bf9ef70c253bb5a26280143d941c915910763", "shape": [16777216]}, {"dtype": "float32", "name": "grad7", "sha256": "c4fdaf0915399dc3568c7fd4a49465f9e3c69a10d3e77b1fc15d926db244f302", "shape": [16777216]}, {"dtype": "float32", "name": "grad8", "sha256": "6b970f1bc919f22be37581526578d65a535e33dd886ff46cc64d95aca61fec12", "shape": [16777216]}, {"dtype": "float32", "name": "grad9", "sha256": "00adaa7225e175a85e28e59772e3afbb901e068e5ca8d7332adfa127f7fcbdd7", "shape": [16777216]}] | [{"dtype": "float32", "name": "p0", "sha256": "485a7287793a9d7d33d710ea2820ea9c7519d31100db4bafeee813baa532dd7d", "shape": [16777216]}, {"dtype": "float32", "name": "grad0", "sha256": "e6c0159d56a2df748f38d3f3592321edc11067354f52e751e382384dca4178bc", "shape": [16777216]}, {"dtype": "float32", "name": "grad1", "sha256": "249bb61249ee25b0cd42e6bcaa012c27c565dde4ccd798448a60e935fd4ace7e", "shape": [16777216]}, {"dtype": "float32", "name": "grad2", "sha256": "4cf167fba649d3d82fdb6c7c41749a74e8fc8a9bd70491154f03bfc9272b2ed6", "shape": [16777216]}, {"dtype": "float32", "name": "grad3", "sha256": "594f963cfcef38863774b145fac84f56ae7f37190af1cf270e7cb23d3b86b587", "shape": [16777216]}, {"dtype": "float32", "name": "grad4", "sha256": "ee091b2ac84630d1d0328c337b12c5bb6bd6ba2b28f3350eab106bc848960f4e", "shape": [16777216]}, {"dtype": "float32", "name": "grad5", "sha256": "a042832a1a0884016cafe5aada2f1df6bf0b09422e9358536881a536fff70d02", "shape": [16777216]}, {"dtype": "float32", "name": "grad6", "sha256": "736aade529a9b43a7666f9d5a77bf9ef70c253bb5a26280143d941c915910763", "shape": [16777216]}, {"dtype": "float32", "name": "grad7", "sha256": "c4fdaf0915399dc3568c7fd4a49465f9e3c69a10d3e77b1fc15d926db244f302", "shape": [16777216]}, {"dtype": "float32", "name": "grad8", "sha256": "6b970f1bc919f22be37581526578d65a535e33dd886ff46cc64d95aca61fec12", "shape": [16777216]}, {"dtype": "float32", "name": "grad9", "sha256": "00adaa7225e175a85e28e59772e3afbb901e068e5ca8d7332adfa127f7fcbdd7", "shape": [16777216]}] | [{"dtype": "float32", "name": "p0", "sha256": "485a7287793a9d7d33d710ea2820ea9c7519d31100db4bafeee813baa532dd7d", "shape": [16777216]}, {"dtype": "float32", "name": "grad0", "sha256": "e6c0159d56a2df748f38d3f3592321edc11067354f52e751e382384dca4178bc", "shape": [16777216]}, {"dtype": "float32", "name": "grad1", "sha256": "249bb61249ee25b0cd42e6bcaa012c27c565dde4ccd798448a60e935fd4ace7e", "shape": [16777216]}, {"dtype": "float32", "name": "grad2", "sha256": "4cf167fba649d3d82fdb6c7c41749a74e8fc8a9bd70491154f03bfc9272b2ed6", "shape": [16777216]}, {"dtype": "float32", "name": "grad3", "sha256": "594f963cfcef38863774b145fac84f56ae7f37190af1cf270e7cb23d3b86b587", "shape": [16777216]}, {"dtype": "float32", "name": "grad4", "sha256": "ee091b2ac84630d1d0328c337b12c5bb6bd6ba2b28f3350eab106bc848960f4e", "shape": [16777216]}, {"dtype": "float32", "name": "grad5", "sha256": "a042832a1a0884016cafe5aada2f1df6bf0b09422e9358536881a536fff70d02", "shape": [16777216]}, {"dtype": "float32", "name": "grad6", "sha256": "736aade529a9b43a7666f9d5a77bf9ef70c253bb5a26280143d941c915910763", "shape": [16777216]}, {"dtype": "float32", "name": "grad7", "sha256": "c4fdaf0915399dc3568c7fd4a49465f9e3c69a10d3e77b1fc15d926db244f302", "shape": [16777216]}, {"dtype": "float32", "name": "grad8", "sha256": "6b970f1bc919f22be37581526578d65a535e33dd886ff46cc64d95aca61fec12", "shape": [16777216]}, {"dtype": "float32", "name": "grad9", "sha256": "00adaa7225e175a85e28e59772e3afbb901e068e5ca8d7332adfa127f7fcbdd7", "shape": [16777216]}] | [{"dtype": "float32", "name": "p0", "sha256": "485a7287793a9d7d33d710ea2820ea9c7519d31100db4bafeee813baa532dd7d", "shape": [16777216]}, {"dtype": "float32", "name": "grad0", "sha256": "e6c0159d56a2df748f38d3f3592321edc11067354f52e751e382384dca4178bc", "shape": [16777216]}, {"dtype": "float32", "name": "grad1", "sha256": "249bb61249ee25b0cd42e6bcaa012c27c565dde4ccd798448a60e935fd4ace7e", "shape": [16777216]}, {"dtype": "float32", "name": "grad2", "sha256": "4cf167fba649d3d82fdb6c7c41749a74e8fc8a9bd70491154f03bfc9272b2ed6", "shape": [16777216]}, {"dtype": "float32", "name": "grad3", "sha256": "594f963cfcef38863774b145fac84f56ae7f37190af1cf270e7cb23d3b86b587", "shape": [16777216]}, {"dtype": "float32", "name": "grad4", "sha256": "ee091b2ac84630d1d0328c337b12c5bb6bd6ba2b28f3350eab106bc848960f4e", "shape": [16777216]}, {"dtype": "float32", "name": "grad5", "sha256": "a042832a1a0884016cafe5aada2f1df6bf0b09422e9358536881a536fff70d02", "shape": [16777216]}, {"dtype": "float32", "name": "grad6", "sha256": "736aade529a9b43a7666f9d5a77bf9ef70c253bb5a26280143d941c915910763", "shape": [16777216]}, {"dtype": "float32", "name": "grad7", "sha256": "c4fdaf0915399dc3568c7fd4a49465f9e3c69a10d3e77b1fc15d926db244f302", "shape": [16777216]}, {"dtype": "float32", "name": "grad8", "sha256": "6b970f1bc919f22be37581526578d65a535e33dd886ff46cc64d95aca61fec12", "shape": [16777216]}, {"dtype": "float32", "name": "grad9", "sha256": "00adaa7225e175a85e28e59772e3afbb901e068e5ca8d7332adfa127f7fcbdd7", "shape": [16777216]}] |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 |
| maximize | false | false | false | false |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| weight_decay | 0.01 | 0.01 | 0.01 | 0.01 |

accepted difference: torch-eager-fp32 amsgrad: ordinary Adam moments on both arms; AMSGrad is disabled and our API does not implement that variant

accepted difference: torch-eager-fp32 capturable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-eager-fp32 decoupled_weight_decay: Adam uses coupled decay and AdamW uses decoupled decay; our class fixes that choice (_training_impl.py _KIND)

accepted difference: torch-eager-fp32 differentiable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-eager-fp32 foreach: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

accepted difference: torch-eager-fp32 fused: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

accepted difference: torch-compile-fp32 amsgrad: ordinary Adam moments on both arms; AMSGrad is disabled and our API does not implement that variant

accepted difference: torch-compile-fp32 capturable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-compile-fp32 decoupled_weight_decay: Adam uses coupled decay and AdamW uses decoupled decay; our class fixes that choice (_training_impl.py _KIND)

accepted difference: torch-compile-fp32 differentiable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-compile-fp32 foreach: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

accepted difference: torch-compile-fp32 fused: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

accepted difference: torch-fused-fp32 amsgrad: ordinary Adam moments on both arms; AMSGrad is disabled and our API does not implement that variant

accepted difference: torch-fused-fp32 capturable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-fused-fp32 decoupled_weight_decay: Adam uses coupled decay and AdamW uses decoupled decay; our class fixes that choice (_training_impl.py _KIND)

accepted difference: torch-fused-fp32 differentiable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-fused-fp32 foreach: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

accepted difference: torch-fused-fp32 fused: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

### ard / istella (rows full, shape X 100000x220; Xq 100000x220; y 100000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1661)`, ran on AMD Instinct MI325X (amd, DO) job a1661

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@6cd6bed7f) | identical | 1234.1 | 1234.1..1234.1 | 1 | - | - | 1234.1 | - | - (stored whole) | - | - | - | - | finite=True, r2=-0.124856, rmse=0.885949 | - | main board, one scored run | - | ok (main@6cd6bed7f amd/a1661 2026-10-11; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 14123.5 | 14123.5..14123.5 | 1 | 0.087 | - | 14123.5 | 14123.5 | 0.00 (cpu-arm) | 0.087 (whole/whole) | - | 1389.6 | - | finite=True, r2=0.327436, rmse=0.685058 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha_1': 1e-06, 'alpha_2': 1e-06, 'fit_intercept': True, 'lambda_1': 1e-06, 'lambda_2': 1e-06, 'max_iter': 300, 'threshold_lambda': 10000.0, 'tol': 0.001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### avgpool1d / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/amd/campaign/measured/algos.avgpool1d.synthetic.rows=full/board.json (amd/paired-c1a3592fe99e264d7771d8f70a4727d834f6334904fb3f679b6c0b6faba09881)`, ran on mojolearn-steward-do-amd

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 0.7 ms) = 0.499; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 0.7 ms) = 0.517. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | 0.8.44 | identical | 0.4 | 0.4..0.4 | 1 | - | - | - | 0.4 | - (stored kernel) | - | - | 2196.9 | 235669.5 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | main board, one scored run | wheel | ok (main@79fcd24a5 amd/paired-c1a3592fe99e264d7771d8f70a4727d834f6334904fb3f679b6c0b6faba09881 2026-10-14; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 0.7 | 0.7..0.7 | 1 | 0.517 | - | - | 0.7 | - (stored kernel) | 0.517 (kernel/kernel) | - | 1785.7 | 224.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.2 | 1.2..1.2 | 1 | 0.293 | - | - | 1.2 | - (stored kernel) | 0.293 (kernel/kernel) | - | 2237.5 | 224.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 0.7 | 0.7..0.7 | 1 | 0.499 | - | - | 0.7 | - (stored kernel) | 0.499 (kernel/kernel) | - | 1785.2 | 224.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.3 | 1.3..1.3 | 1 | 0.276 | - | - | 1.3 | - (stored kernel) | 0.276 (kernel/kernel) | - | 2152.1 | 224.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/103784/vram_* bytes for this pid at the round's end (not a peak)

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
| mojolearn IDENTICAL | AMD Instinct MI325X (amd, DO) | 0.8.44 | Xq | - | 0.1 | 0.1..0.1 | 1 | - | - | - | 0.1 | - (stored kernel) | - | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | Xq | - | 0.1 | 0.1..0.1 | 1 | 1.299 | - | - | 0.1 | - (stored kernel) | 1.299 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | Xq | - | 0.1 | 0.1..0.1 | 1 | 0.759 | - | - | 0.1 | - (stored kernel) | 0.759 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | Xq | - | 0.1 | 0.1..0.1 | 1 | 1.084 | - | - | 0.1 | - (stored kernel) | 1.084 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | Xq | - | 0.1 | 0.1..0.1 | 1 | 0.730 | - | - | 0.1 | - (stored kernel) | 0.730 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### avgpool2d / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/amd/campaign/measured/algos.avgpool2d.synthetic.rows=full/board.json (amd/paired-ddf54539edfbea7f894d4f108dee90ab74bc97f7b0fd79dfd2d47173dd3f73b2)`, ran on mojolearn-steward-do-amd

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 0.8 ms) = 0.750; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 0.9 ms) = 0.669. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | 0.8.44 | identical | 0.6 | 0.6..0.6 | 1 | - | - | - | 0.6 | - (stored kernel) | - | - | 2397.0 | 235669.5 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | main board, one scored run | wheel | ok (main@79fcd24a5 amd/paired-ddf54539edfbea7f894d4f108dee90ab74bc97f7b0fd79dfd2d47173dd3f73b2 2026-10-14; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 0.9 | 0.9..0.9 | 1 | 0.669 | - | - | 0.9 | - (stored kernel) | 0.669 (kernel/kernel) | - | 2099.4 | 541.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.2 | 1.2..1.2 | 1 | 0.504 | - | - | 1.2 | - (stored kernel) | 0.504 (kernel/kernel) | - | 2554.5 | 541.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 0.8 | 0.8..0.8 | 1 | 0.750 | - | - | 0.8 | - (stored kernel) | 0.750 (kernel/kernel) | - | 2099.3 | 541.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.1 | 1.1..1.1 | 1 | 0.517 | - | - | 1.1 | - (stored kernel) | 0.517 (kernel/kernel) | - | 2471.5 | 541.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/104881/vram_* bytes for this pid at the round's end (not a peak)

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
| mojolearn IDENTICAL | AMD Instinct MI325X (amd, DO) | 0.8.44 | Xq | - | 0.2 | 0.2..0.2 | 1 | - | - | - | 0.2 | - (stored kernel) | - | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | Xq | - | 0.1 | 0.1..0.1 | 1 | 1.420 | - | - | 0.1 | - (stored kernel) | 1.420 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | Xq | - | 0.2 | 0.2..0.2 | 1 | 0.868 | - | - | 0.2 | - (stored kernel) | 0.868 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | Xq | - | 0.1 | 0.1..0.1 | 1 | 1.378 | - | - | 0.1 | - (stored kernel) | 1.378 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | Xq | - | 0.2 | 0.2..0.2 | 1 | 0.940 | - | - | 0.2 | - (stored kernel) | 0.940 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### batchnorm1d / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/amd-wave4/campaign/measured/algos.batchnorm1d.synthetic.rows=full/board.json (amd/paired-92c32b752176a9b8bf2065554aa750c29dc1d4f2b63043868a88df19cfa6f5d4)`, ran on mojolearn-steward-do-amd

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 0.9 ms) = 0.655; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 1.0 ms) = 0.599. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | 0.8.44 | identical | 0.6 | 0.6..0.6 | 1 | - | - | - | 0.6 | - (stored kernel) | - | - | 2325.2 | 235669.6 | max_rel_diff_vs_torch_eager_fp32=0.004642, rel_fro_vs_torch_eager_fp32=8.097e-08 | - | main board, one scored run | wheel | ok (main@3c60869cf amd/paired-92c32b752176a9b8bf2065554aa750c29dc1d4f2b63043868a88df19cfa6f5d4 2026-10-14; identity vs the nvidia columns: n/a) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.0 | 1.0..1.0 | 1 | 0.599 | - | - | 1.0 | - (stored kernel) | 0.599 (kernel/kernel) | - | 2511.0 | 320.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.5 | 1.5..1.5 | 1 | 0.393 | - | - | 1.5 | - (stored kernel) | 0.393 (kernel/kernel) | - | 2367.4 | 320.0 | max_rel_diff_vs_torch_eager_fp32=0.002793, rel_fro_vs_torch_eager_fp32=5.248e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 0.9 | 0.9..0.9 | 1 | 0.655 | - | - | 0.9 | - (stored kernel) | 0.655 (kernel/kernel) | - | 2144.3 | 320.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.5 | 1.5..1.5 | 1 | 0.404 | - | - | 1.5 | - (stored kernel) | 0.404 (kernel/kernel) | - | 2258.9 | 320.0 | max_rel_diff_vs_torch_eager_fp32=0.002793, rel_fro_vs_torch_eager_fp32=5.248e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/160907/vram_* bytes for this pid at the round's end (not a peak)

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
| mojolearn IDENTICAL | AMD Instinct MI325X (amd, DO) | 0.8.44 | Xq | - | 1.3 | 1.3..1.3 | 1 | - | - | - | 1.3 | - (stored kernel) | - | - | max_rel_diff_vs_torch_eager_fp32=0.004642, rel_fro_vs_torch_eager_fp32=8.097e-08 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | Xq | - | 0.1 | 0.1..0.1 | 1 | 10.152 | - | - | 0.1 | - (stored kernel) | 10.152 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | Xq | - | 0.2 | 0.2..0.2 | 1 | 7.351 | - | - | 0.2 | - (stored kernel) | 7.351 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.002793, rel_fro_vs_torch_eager_fp32=5.248e-08 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | Xq | - | 0.1 | 0.1..0.1 | 1 | 9.751 | - | - | 0.1 | - (stored kernel) | 9.751 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | Xq | - | 0.2 | 0.2..0.2 | 1 | 7.249 | - | - | 0.2 | - (stored kernel) | 7.249 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.002793, rel_fro_vs_torch_eager_fp32=5.248e-08 | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### batchnorm2d / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/amd-wave4/campaign/measured/algos.batchnorm2d.synthetic.rows=full/board.json (amd/paired-290f77101012de7e3d43ccd22c144fb63e003a06a75d6d6dacf855f451300c5f)`, ran on mojolearn-steward-do-amd

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 1.2 ms) = 0.524; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 1.0 ms) = 0.620. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | 0.8.44 | identical | 0.6 | 0.6..0.6 | 1 | - | - | - | 0.6 | - (stored kernel) | - | - | 2250.4 | 235669.6 | max_rel_diff_vs_torch_eager_fp32=0.001395, rel_fro_vs_torch_eager_fp32=1.249e-07 | - | main board, one scored run | wheel | ok (main@3c60869cf amd/paired-290f77101012de7e3d43ccd22c144fb63e003a06a75d6d6dacf855f451300c5f 2026-10-14; identity vs the nvidia columns: n/a) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.0 | 1.0..1.0 | 1 | 0.620 | - | - | 1.0 | - (stored kernel) | 0.620 (kernel/kernel) | - | 2397.3 | 250.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.3 | 1.3..1.3 | 1 | 0.466 | - | - | 1.3 | - (stored kernel) | 0.466 (kernel/kernel) | - | 2286.2 | 250.0 | max_rel_diff_vs_torch_eager_fp32=0.001471, rel_fro_vs_torch_eager_fp32=6.868e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.2 | 1.2..1.2 | 1 | 0.524 | - | - | 1.2 | - (stored kernel) | 0.524 (kernel/kernel) | - | 2030.8 | 250.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.3 | 1.3..1.3 | 1 | 0.459 | - | - | 1.3 | - (stored kernel) | 0.459 (kernel/kernel) | - | 2183.2 | 250.0 | max_rel_diff_vs_torch_eager_fp32=0.001471, rel_fro_vs_torch_eager_fp32=6.868e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/162603/vram_* bytes for this pid at the round's end (not a peak)

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
| mojolearn IDENTICAL | AMD Instinct MI325X (amd, DO) | 0.8.44 | Xq | - | 1.3 | 1.3..1.3 | 1 | - | - | - | 1.3 | - (stored kernel) | - | - | max_rel_diff_vs_torch_eager_fp32=0.001395, rel_fro_vs_torch_eager_fp32=1.249e-07 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | Xq | - | 0.2 | 0.2..0.2 | 1 | 6.623 | - | - | 0.2 | - (stored kernel) | 6.623 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | Xq | - | 0.2 | 0.2..0.2 | 1 | 8.063 | - | - | 0.2 | - (stored kernel) | 8.063 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.001471, rel_fro_vs_torch_eager_fp32=6.868e-08 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | Xq | - | 0.2 | 0.2..0.2 | 1 | 6.211 | - | - | 0.2 | - (stored kernel) | 6.211 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | Xq | - | 0.2 | 0.2..0.2 | 1 | 7.883 | - | - | 0.2 | - (stored kernel) | 7.883 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.001471, rel_fro_vs_torch_eager_fp32=6.868e-08 | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### bayesian-ridge / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1661)`, ran on AMD Instinct MI325X (amd, DO) job a1661

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@6cd6bed7f) | identical | 1130.2 | 1130.2..1130.2 | 1 | - | - | 1130.2 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.317909, rmse=0.689893 | - | main board, one scored run | - | ok (main@6cd6bed7f amd/a1661 2026-10-11; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 3401.4 | 3401.4..3401.4 | 1 | 0.332 | - | 3401.4 | 3401.4 | 0.00 (cpu-arm) | 0.332 (whole/whole) | - | 4508.1 | - | finite=True, r2=-8.145e+07, rmse=7538.692925 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha_1': 1e-06, 'alpha_2': 1e-06, 'fit_intercept': True, 'lambda_1': 1e-06, 'lambda_2': 1e-06, 'max_iter': 300, 'tol': 0.001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### bernoulli-nb / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1210)`, ran on AMD Instinct MI325X (amd, DO) job a1210

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@6416999f7) | identical | 26.0 | 26.0..26.0 | 1 | - | - | 26.0 | - | - (stored whole) | - | - | - | - | accuracy=0.794050, logloss=5.350625 | - | main board, one scored run | - | ok (main@6416999f7 amd/a1210 2026-10-10; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 773.0 | 773.0..773.0 | 1 | 0.034 | - | 773.0 | 773.0 | 0.00 (cpu-arm) | 0.034 (whole/whole) | - | 3874.7 | - | accuracy=0.794050, logloss=4.278741 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'binarize': 0.0, 'fit_prior': True, 'force_alpha': True}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), BernoulliNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### bernoulli-nb / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1210)`, ran on AMD Instinct MI325X (amd, DO) job a1210

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@6416999f7) | identical | 10.8 | 10.8..10.8 | 1 | - | - | 10.8 | - | - (stored whole) | - | - | - | - | accuracy=0.755560, logloss=0.557803 | - | main board, one scored run | - | ok (main@6416999f7 amd/a1210 2026-10-10; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 77.3 | 77.3..77.3 | 1 | 0.139 | - | 77.3 | 77.3 | 0.00 (cpu-arm) | 0.139 (whole/whole) | - | 446.1 | - | accuracy=0.755560, logloss=0.557802 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'binarize': 0.0, 'fit_prior': True, 'force_alpha': True}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), BernoulliNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### binarizer / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/amd_ingest/results.txt (amd/a0005)`, ran on AMD Instinct MI325X (amd, DO) job a0005

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@a8fd6d926) | identical | 0.1 | 0.1..0.1 | 1 | - | - | 0.1 | - | - (stored whole) | - | - | - | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x220, rel_diff_vs_sklearn=0.000000 | - | main board, one scored run | - | ok (main@a8fd6d926 amd/a0005 2026-10-12; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 26.2 | 26.2..26.2 | 1 | 0.004 | - | 26.2 | 26.2 | 0.00 (cpu-arm) | 0.004 (whole/whole) | - | 1508.5 | - | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'threshold': 0.0}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), Binarizer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### binarizer / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/amd_ingest/results.txt (amd/a0005)`, ran on AMD Instinct MI325X (amd, DO) job a0005

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@a8fd6d926) | identical | 0.1 | 0.1..0.1 | 1 | - | - | 0.1 | - | - (stored whole) | - | - | - | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x11, rel_diff_vs_sklearn=0.000000 | - | main board, one scored run | - | ok (main@a8fd6d926 amd/a0005 2026-10-12; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 1.6 | 1.6..1.6 | 1 | 0.051 | - | 1.6 | 1.6 | 0.00 (cpu-arm) | 0.051 (whole/whole) | - | 273.0 | - | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'threshold': 0.0}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), Binarizer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### categorical-nb / istella (rows full, shape X 1000000x8; Xq 100000x8; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1210)`, ran on AMD Instinct MI325X (amd, DO) job a1210

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@6416999f7) | identical | 18.0 | 18.0..18.0 | 1 | - | - | 18.0 | - | - (stored whole) | - | - | - | - | accuracy=0.838850, logloss=0.412625 | - | main board, one scored run | - | ok (main@6416999f7 amd/a1210 2026-10-10; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 89.3 | 89.3..89.3 | 1 | 0.201 | - | 89.3 | 89.3 | 0.00 (cpu-arm) | 0.201 (whole/whole) | - | 360.0 | - | accuracy=0.838850, logloss=0.412625 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'fit_prior': True, 'force_alpha': True}. Rows: None. Timed: None.

mismatch: min_categories = every code seen in X or Xq on scikit-learn; ours refuses the option (option parity) and cuML has none

config: cuML benchmark (RAPIDS), CategoricalNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### categorical-nb / taxi (rows full, shape X 1000000x5; Xq 100000x5; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1210)`, ran on AMD Instinct MI325X (amd, DO) job a1210

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@6416999f7) | identical | 18.2 | 18.2..18.2 | 1 | - | - | 18.2 | - | - (stored whole) | - | - | - | - | accuracy=0.765850, logloss=0.538866 | - | main board, one scored run | - | ok (main@6416999f7 amd/a1210 2026-10-10; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 74.9 | 74.9..74.9 | 1 | 0.243 | - | 74.9 | 74.9 | 0.00 (cpu-arm) | 0.243 (whole/whole) | - | 323.7 | - | accuracy=0.765850, logloss=0.538866 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'fit_prior': True, 'force_alpha': True}. Rows: None. Timed: None.

mismatch: min_categories = every code seen in X or Xq on scikit-learn; ours refuses the option (option parity) and cuML has none

config: cuML benchmark (RAPIDS), CategoricalNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### cholesky / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/full-gpu-repair-20261014/recovered/algos.cholesky.synthetic.rows=full/board.json (amd/paired-7955aaac7353279f046f9d01201a096ad04e28a36b05dff4ada54d3febb86d33)`, ran on mojolearn-steward-do-amd

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | 0.8.44 | identical | 160.3 | 160.3..160.3 | 1 | - | - | 160.3 | 154.6 | 5.76 (upload_ms_separate) | - | - | 4176.6 | 236193.2 | relative_residual=2.9e-07 | - | main board, one scored run | wheel | ok (main@d425c1417 amd/paired-7955aaac7353279f046f9d01201a096ad04e28a36b05dff4ada54d3febb86d33 2026-10-14; identity vs the nvidia columns: n/a) |
| torch-gpu | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 32.2 | 32.2..32.2 | 1 | 4.977 | - | - | 32.2 | - (stored kernel) | 4.798 (kernel/kernel) | - | 3441.8 | 790.4 | relative_residual=2.519e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/96177/vram_* bytes for this pid at the round's end (not a peak)

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

### cnn-clf / synthetic (rows full, shape X 20000x1x28x28; Xq 5000x1x28x28; y 20000; yq 5000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1647)`, ran on AMD Instinct MI325X (amd, DO) job a1647

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 251.7 ms) = 1.251; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 172.9 ms) = 1.822. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@2afa942e3) | identical | 315.0 | 315.0..315.0 | 1 | - | - | 315.0 | - | - (stored whole) | - | - | - | - | accuracy=1.000000 | - | main board, one scored run | - | ok (main@2afa942e3 amd/a1647 2026-10-11; identity vs the nvidia columns: n/a) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 172.9 | 172.9..172.9 | 1 | 1.822 | - | - | 172.9 | - (stored kernel) | 1.822 (MIXED ours whole / arm kernel) | - | 3768.4 | 271.5 | accuracy=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 245.0 | 245.0..245.0 | 1 | 1.286 | - | - | 245.0 | - (stored kernel) | 1.286 (MIXED ours whole / arm kernel) | - | 3778.8 | 198.0 | accuracy=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 251.7 | 251.7..251.7 | 1 | 1.251 | - | - | 251.7 | - (stored kernel) | 1.251 (MIXED ours whole / arm kernel) | - | 5492.3 | 278.6 | accuracy=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 269.9 | 269.9..269.9 | 1 | 1.167 | - | - | 269.9 | - (stored kernel) | 1.167 (MIXED ours whole / arm kernel) | - | 5267.5 | 277.6 | accuracy=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 128, 'conv_channels': [8, 16], 'dampening': 0.0, 'input_shape': [1, 28, 28], 'kernel_size': 3, 'learning_rate': 0.01, 'max_iter': 2, 'momentum': 0.9, 'nesterov': False, 'optimizer': 'sgd', 'pool_size': 2, 'random_state': 7, 'shuffle': True, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### complement-nb / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1706)`, ran on AMD Instinct MI325X (amd, DO) job a1706

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 67.1 | 67.1..67.1 | 1 | - | - | 67.1 | - | - (stored whole) | - | - | - | - | accuracy=0.849360, logloss=3.762524 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1706 2026-10-11; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 196.8 | 196.8..196.8 | 1 | 0.341 | - | 196.8 | 196.8 | 0.00 (cpu-arm) | 0.341 (whole/whole) | - | 3958.9 | - | accuracy=0.849350, logloss=3.174763 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'fit_prior': True, 'force_alpha': True, 'norm': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), ComplementNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### complement-nb / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1706)`, ran on AMD Instinct MI325X (amd, DO) job a1706

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 29.3 | 29.3..29.3 | 1 | - | - | 29.3 | - | - (stored whole) | - | - | - | - | accuracy=0.678020, logloss=0.715492 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1706 2026-10-11; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 38.2 | 38.2..38.2 | 1 | 0.769 | - | 38.2 | 38.2 | 0.00 (cpu-arm) | 0.769 (whole/whole) | - | 450.3 | - | accuracy=0.678030, logloss=0.715493 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'fit_prior': True, 'force_alpha': True, 'norm': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), ComplementNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### complement-nb / text (rows full, shape X 86626x4096; Xq 9626x4096; y 86626; yq 9626)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0879)`, ran on AMD Instinct MI325X (amd, DO) job a0879

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@42d1e42c6) | identical | 21.0 | 21.0..21.0 | 1 | - | - | 21.0 | - | - (stored whole) | - | - | - | - | accuracy=0.983067, logloss=0.559491 | - | main board, one scored run | - | ok (main@42d1e42c6 amd/a0879 2026-10-08; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 269.7 | 269.7..269.7 | 1 | 0.078 | - | 269.7 | 269.7 | 0.00 (cpu-arm) | 0.078 (whole/whole) | - | 4455.8 | - | accuracy=0.983067, logloss=0.557285 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'fit_prior': True, 'force_alpha': True, 'norm': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), ComplementNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### connected-components / istella (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0870)`, ran on AMD Instinct MI325X (amd, DO) job a0870

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@4df610f3b) | identical | 0.7 | 0.7..0.7 | 1 | - | - | 0.7 | - | - (stored whole) | - | - | - | - | n_components=81 | - | main board, one scored run | - | ok (main@4df610f3b amd/a0870 2026-10-08; identity vs nvidia-l40s: MATCH) |

settings: {}. Rows: None. Timed: None.

mismatch: ours takes the CSR graph as an (indptr, indices, n) tuple and uploads it inside the clock (the 2026-10-08 CSR entries; before, a dense n x n adjacency built before the clock); networkx and cuGraph take the graph itself

mismatch: opponents withheld: the lane settings at this tree's HEAD differ from the settings release-board-resume-r2 recorded for its opponent race (an opponent job must score them again)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### connected-components / taxi (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0870)`, ran on AMD Instinct MI325X (amd, DO) job a0870

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@4df610f3b) | identical | 0.6 | 0.6..0.6 | 1 | - | - | 0.6 | - | - (stored whole) | - | - | - | - | n_components=588 | - | main board, one scored run | - | ok (main@4df610f3b amd/a0870 2026-10-08; identity vs nvidia-l40s: MATCH) |

settings: {}. Rows: None. Timed: None.

mismatch: ours takes the CSR graph as an (indptr, indices, n) tuple and uploads it inside the clock (the 2026-10-08 CSR entries; before, a dense n x n adjacency built before the clock); networkx and cuGraph take the graph itself

mismatch: opponents withheld: the lane settings at this tree's HEAD differ from the settings release-board-resume-r2 recorded for its opponent race (an opponent job must score them again)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### conv1d / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/amd-wave4/campaign/measured/algos.conv1d.synthetic.rows=full/board.json (amd/paired-ea5d7efc1138eebc85d5e289358420f00569a685aeb29734913c6c8caebbb381)`, ran on mojolearn-steward-do-amd

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 1.4 ms) = 5.128; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 1.9 ms) = 3.693. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | 0.8.44 | identical | 7.1 | 7.1..7.1 | 1 | - | - | - | 7.1 | - (stored kernel) | - | - | 3022.2 | 239248.5 | max_rel_diff_vs_torch_eager_fp32=0.318264, rel_fro_vs_torch_eager_fp32=3.948e-07 | - | main board, one scored run | wheel | ok (main@3c60869cf amd/paired-ea5d7efc1138eebc85d5e289358420f00569a685aeb29734913c6c8caebbb381 2026-10-14; identity vs the nvidia columns: n/a) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.9 | 1.9..1.9 | 1 | 3.693 | - | - | 1.9 | - (stored kernel) | 3.693 (kernel/kernel) | - | 3210.3 | 896.6 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 2.6 | 2.6..2.6 | 1 | 2.742 | - | - | 2.6 | - (stored kernel) | 2.742 (kernel/kernel) | - | 2879.6 | 896.6 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.4 | 1.4..1.4 | 1 | 5.128 | - | - | 1.4 | - (stored kernel) | 5.128 (kernel/kernel) | - | 3456.1 | 832.4 | max_rel_diff_vs_torch_eager_fp32=3418.035805, rel_fro_vs_torch_eager_fp32=0.004377 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 2.2 | 2.2..2.2 | 1 | 3.269 | - | - | 2.2 | - (stored kernel) | 3.269 (kernel/kernel) | - | 2922.4 | 832.4 | max_rel_diff_vs_torch_eager_fp32=3418.035805, rel_fro_vs_torch_eager_fp32=0.004377 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/154314/vram_* bytes for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'bias': True, 'dilation': 1, 'groups': 1, 'in_channels': 128, 'kernel_size': 3, 'out_channels': 128, 'padding': 1, 'padding_mode': 'zeros', 'stride': 1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| bias | true | true | true | true | true |
| dilation | 1 | 1 | 1 | 1 | 1 |
| groups | 1 | 1 | 1 | 1 | 1 |
| in_channels | 128 | 128 | 128 | 128 | 128 |
| input_arrays | [{"dtype": "float32", "name": "x", "sha256": "cec4f976c7feae09a100521e6e404c90584b25330e0bef803c3ae82efaa217e7", "shape": [64, 128, 4096]}, {"dtype": "float32", "name": "bias", "sha256": "a8e12a7ee1bfe5640579402eca2de99b8e6c38a3d243507e7ad1edadebc7eb84", "shape": [128]}, {"dtype": "float32", "name": "weight", "sha256": "cd1becee90e2a8fbfe515641fefd6bfbeb9dba42273fb21af1273164cb40fe15", "shape": [128, 128, 3]}] | [{"dtype": "float32", "name": "x", "sha256": "cec4f976c7feae09a100521e6e404c90584b25330e0bef803c3ae82efaa217e7", "shape": [64, 128, 4096]}, {"dtype": "float32", "name": "bias", "sha256": "a8e12a7ee1bfe5640579402eca2de99b8e6c38a3d243507e7ad1edadebc7eb84", "shape": [128]}, {"dtype": "float32", "name": "weight", "sha256": "cd1becee90e2a8fbfe515641fefd6bfbeb9dba42273fb21af1273164cb40fe15", "shape": [128, 128, 3]}] | [{"dtype": "float32", "name": "x", "sha256": "cec4f976c7feae09a100521e6e404c90584b25330e0bef803c3ae82efaa217e7", "shape": [64, 128, 4096]}, {"dtype": "float32", "name": "bias", "sha256": "a8e12a7ee1bfe5640579402eca2de99b8e6c38a3d243507e7ad1edadebc7eb84", "shape": [128]}, {"dtype": "float32", "name": "weight", "sha256": "cd1becee90e2a8fbfe515641fefd6bfbeb9dba42273fb21af1273164cb40fe15", "shape": [128, 128, 3]}] | [{"dtype": "float32", "name": "x", "sha256": "cec4f976c7feae09a100521e6e404c90584b25330e0bef803c3ae82efaa217e7", "shape": [64, 128, 4096]}, {"dtype": "float32", "name": "bias", "sha256": "a8e12a7ee1bfe5640579402eca2de99b8e6c38a3d243507e7ad1edadebc7eb84", "shape": [128]}, {"dtype": "float32", "name": "weight", "sha256": "cd1becee90e2a8fbfe515641fefd6bfbeb9dba42273fb21af1273164cb40fe15", "shape": [128, 128, 3]}] | [{"dtype": "float32", "name": "x", "sha256": "cec4f976c7feae09a100521e6e404c90584b25330e0bef803c3ae82efaa217e7", "shape": [64, 128, 4096]}, {"dtype": "float32", "name": "bias", "sha256": "a8e12a7ee1bfe5640579402eca2de99b8e6c38a3d243507e7ad1edadebc7eb84", "shape": [128]}, {"dtype": "float32", "name": "weight", "sha256": "cd1becee90e2a8fbfe515641fefd6bfbeb9dba42273fb21af1273164cb40fe15", "shape": [128, 128, 3]}] |
| kernel_size | 3 | 3 | 3 | 3 | 3 |
| out_channels | 128 | 128 | 128 | 128 | 128 |
| padding | 1 | 1 | 1 | 1 | 1 |
| padding_mode | "zeros" | "zeros" | "zeros" | "zeros" | "zeros" |
| seed | 7 | 7 | 7 | 7 | 7 |
| stride | 1 | 1 | 1 | 1 | 1 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | hardware | version | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | AMD Instinct MI325X (amd, DO) | 0.8.44 | Xq | - | 1.7 | 1.7..1.7 | 1 | - | - | - | 1.7 | - (stored kernel) | - | - | max_rel_diff_vs_torch_eager_fp32=0.318264, rel_fro_vs_torch_eager_fp32=3.948e-07 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | Xq | - | 0.7 | 0.7..0.7 | 1 | 2.537 | - | - | 0.7 | - (stored kernel) | 2.537 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | Xq | - | 0.9 | 0.9..0.9 | 1 | 2.039 | - | - | 0.9 | - (stored kernel) | 2.039 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | Xq | - | 0.4 | 0.4..0.4 | 1 | 4.613 | - | - | 0.4 | - (stored kernel) | 4.613 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=3418.035805, rel_fro_vs_torch_eager_fp32=0.004377 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | Xq | - | 0.5 | 0.5..0.5 | 1 | 3.568 | - | - | 0.5 | - (stored kernel) | 3.568 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=3418.035805, rel_fro_vs_torch_eager_fp32=0.004377 | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### conv2d / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/amd-wave4/campaign/measured/algos.conv2d.synthetic.rows=full/board.json (amd/paired-1887dd5be5a0351a8e0653c306ff4824c626d38a40e1b0625e2d1638709f7d65)`, ran on mojolearn-steward-do-amd

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 1.5 ms) = 4.127; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 1.2 ms) = 5.487. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | 0.8.44 | identical | 6.4 | 6.4..6.4 | 1 | - | - | - | 6.4 | - (stored kernel) | - | - | 2627.0 | 239248.5 | max_rel_diff_vs_torch_eager_fp32=0.372529, rel_fro_vs_torch_eager_fp32=3.307e-07 | - | main board, one scored run | wheel | ok (main@3c60869cf amd/paired-1887dd5be5a0351a8e0653c306ff4824c626d38a40e1b0625e2d1638709f7d65 2026-10-14; identity vs the nvidia columns: n/a) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.2 | 1.2..1.2 | 1 | 5.487 | - | - | 1.2 | - (stored kernel) | 5.487 (kernel/kernel) | - | 2601.7 | 347.4 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 2.0 | 2.0..2.0 | 1 | 3.169 | - | - | 2.0 | - (stored kernel) | 3.169 (kernel/kernel) | - | 2481.4 | 347.4 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.5 | 1.5..1.5 | 1 | 4.127 | - | - | 1.5 | - (stored kernel) | 4.127 (kernel/kernel) | - | 2634.0 | 320.8 | max_rel_diff_vs_torch_eager_fp32=3418.486565, rel_fro_vs_torch_eager_fp32=0.003382 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.8 | 1.8..1.8 | 1 | 3.612 | - | - | 1.8 | - (stored kernel) | 3.612 (kernel/kernel) | - | 2529.8 | 320.8 | max_rel_diff_vs_torch_eager_fp32=3418.486565, rel_fro_vs_torch_eager_fp32=0.003382 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/158566/vram_* bytes for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'bias': True, 'dilation': 1, 'groups': 1, 'in_channels': 64, 'kernel_size': 3, 'out_channels': 64, 'padding': 1, 'padding_mode': 'zeros', 'stride': 1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| bias | true | true | true | true | true |
| dilation | 1 | 1 | 1 | 1 | 1 |
| groups | 1 | 1 | 1 | 1 | 1 |
| in_channels | 64 | 64 | 64 | 64 | 64 |
| input_arrays | [{"dtype": "float32", "name": "x", "sha256": "cd7e0a7d8c0c594cca519b9cf49c8f086513282644cb8b6c06771dcb98504f2d", "shape": [64, 64, 56, 56]}, {"dtype": "float32", "name": "bias", "sha256": "a055c18d708f2d056981b307f2bfe3634d34ae7e4cd5b6a55330b3a6a96bd6ed", "shape": [64]}, {"dtype": "float32", "name": "weight", "sha256": "a5f32fbce52da86968c08ac2499c8f321445b4391778c6bcb53417a9eb9669bc", "shape": [64, 64, 3, 3]}] | [{"dtype": "float32", "name": "x", "sha256": "cd7e0a7d8c0c594cca519b9cf49c8f086513282644cb8b6c06771dcb98504f2d", "shape": [64, 64, 56, 56]}, {"dtype": "float32", "name": "bias", "sha256": "a055c18d708f2d056981b307f2bfe3634d34ae7e4cd5b6a55330b3a6a96bd6ed", "shape": [64]}, {"dtype": "float32", "name": "weight", "sha256": "a5f32fbce52da86968c08ac2499c8f321445b4391778c6bcb53417a9eb9669bc", "shape": [64, 64, 3, 3]}] | [{"dtype": "float32", "name": "x", "sha256": "cd7e0a7d8c0c594cca519b9cf49c8f086513282644cb8b6c06771dcb98504f2d", "shape": [64, 64, 56, 56]}, {"dtype": "float32", "name": "bias", "sha256": "a055c18d708f2d056981b307f2bfe3634d34ae7e4cd5b6a55330b3a6a96bd6ed", "shape": [64]}, {"dtype": "float32", "name": "weight", "sha256": "a5f32fbce52da86968c08ac2499c8f321445b4391778c6bcb53417a9eb9669bc", "shape": [64, 64, 3, 3]}] | [{"dtype": "float32", "name": "x", "sha256": "cd7e0a7d8c0c594cca519b9cf49c8f086513282644cb8b6c06771dcb98504f2d", "shape": [64, 64, 56, 56]}, {"dtype": "float32", "name": "bias", "sha256": "a055c18d708f2d056981b307f2bfe3634d34ae7e4cd5b6a55330b3a6a96bd6ed", "shape": [64]}, {"dtype": "float32", "name": "weight", "sha256": "a5f32fbce52da86968c08ac2499c8f321445b4391778c6bcb53417a9eb9669bc", "shape": [64, 64, 3, 3]}] | [{"dtype": "float32", "name": "x", "sha256": "cd7e0a7d8c0c594cca519b9cf49c8f086513282644cb8b6c06771dcb98504f2d", "shape": [64, 64, 56, 56]}, {"dtype": "float32", "name": "bias", "sha256": "a055c18d708f2d056981b307f2bfe3634d34ae7e4cd5b6a55330b3a6a96bd6ed", "shape": [64]}, {"dtype": "float32", "name": "weight", "sha256": "a5f32fbce52da86968c08ac2499c8f321445b4391778c6bcb53417a9eb9669bc", "shape": [64, 64, 3, 3]}] |
| kernel_size | 3 | 3 | 3 | 3 | 3 |
| out_channels | 64 | 64 | 64 | 64 | 64 |
| padding | 1 | 1 | 1 | 1 | 1 |
| padding_mode | "zeros" | "zeros" | "zeros" | "zeros" | "zeros" |
| seed | 7 | 7 | 7 | 7 | 7 |
| stride | 1 | 1 | 1 | 1 | 1 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | hardware | version | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | AMD Instinct MI325X (amd, DO) | 0.8.44 | Xq | - | 1.1 | 1.1..1.1 | 1 | - | - | - | 1.1 | - (stored kernel) | - | - | max_rel_diff_vs_torch_eager_fp32=0.372529, rel_fro_vs_torch_eager_fp32=3.307e-07 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | Xq | - | 0.3 | 0.3..0.3 | 1 | 3.868 | - | - | 0.3 | - (stored kernel) | 3.868 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | Xq | - | 0.4 | 0.4..0.4 | 1 | 3.135 | - | - | 0.4 | - (stored kernel) | 3.135 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | Xq | - | 0.3 | 0.3..0.3 | 1 | 4.307 | - | - | 0.3 | - (stored kernel) | 4.307 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=3418.486565, rel_fro_vs_torch_eager_fp32=0.003382 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | Xq | - | 0.3 | 0.3..0.3 | 1 | 3.308 | - | - | 0.3 | - (stored kernel) | 3.308 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=3418.486565, rel_fro_vs_torch_eager_fp32=0.003382 | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### damped-ets / synthetic (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1679)`, ran on AMD Instinct MI325X (amd, DO) job a1679

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 1680.4 | 1680.4..1680.4 | 1 | - | - | 1680.4 | - | - (stored whole) | - | - | - | - | forecast_rmse=13.931153 | - | main board, one scored run | - | ok (main@a1c96a5ba amd/a1679 2026-10-11; identity vs the nvidia columns: n/a) |
| statsmodels-cpu | statsmodels | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | statsmodels 0.15.0 | opponent | 94.5 | 94.5..94.5 | 1 | 17.790 | - | 94.5 | 94.5 | 0.00 (cpu-arm) | 17.790 (whole/whole) | - | 58.1 | - | forecast_rmse=26.588738 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| statsforecast-cpu | statsforecast | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | statsforecast 2.1.1 | opponent | 93.0 | 93.0..93.0 | 1 | 18.063 | - | 93.0 | 93.0 | 0.00 (cpu-arm) | 18.063 (whole/whole) | - | 283.5 | - | forecast_rmse=13.945986 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, statsmodels-cpu, statsforecast-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'damped': True, 'model': 'AAN', 'season_length': 1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### damped-ets / taxi-hourly (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1679)`, ran on AMD Instinct MI325X (amd, DO) job a1679

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 1772.4 | 1772.4..1772.4 | 1 | - | - | 1772.4 | - | - (stored whole) | - | - | - | - | forecast_rmse=96.690449 | - | main board, one scored run | - | ok (main@a1c96a5ba amd/a1679 2026-10-11; identity vs the nvidia columns: n/a) |
| statsmodels-cpu | statsmodels | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | statsmodels 0.15.0 | opponent | 94.0 | 94.0..94.0 | 1 | 18.847 | - | 94.0 | 94.0 | 0.00 (cpu-arm) | 18.847 (whole/whole) | - | 57.9 | - | forecast_rmse=196.955273 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| statsforecast-cpu | statsforecast | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | statsforecast 2.1.1 | opponent | 63.3 | 63.3..63.3 | 1 | 27.996 | - | 63.3 | 63.3 | 0.00 (cpu-arm) | 27.996 (whole/whole) | - | 283.6 | - | forecast_rmse=96.685568 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, statsmodels-cpu, statsforecast-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'damped': True, 'model': 'AAN', 'season_length': 1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### dropout2d / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/amd-wave6/campaign/measured/algos.dropout2d.synthetic.rows=full/board.json (amd/paired-83c79858ab104ed99cc6702e9e6e73017e0a5a67b5f99086ff8470099d7f0694)`, ran on mojolearn-steward-do-amd

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 0.7 ms) = 0.397. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | 0.8.44 | identical | 0.3 | 0.3..0.3 | 1 | - | - | - | 0.3 | - (stored kernel) | - | - | 2251.6 | 235669.5 | dropout_contract_relative_error=5.422e-08, dropout_probability_z=0.708333, dropped_channel_fraction=0.096680, expected_dropout_probability=0.100000, identifiable_channels=4096 | - | main board, one scored run | wheel | ok (main@0f3f674f2 amd/paired-83c79858ab104ed99cc6702e9e6e73017e0a5a67b5f99086ff8470099d7f0694 2026-10-14; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 0.7 | 0.7..0.7 | 1 | 0.397 | - | - | 0.7 | - (stored kernel) | 0.397 (kernel/kernel) | - | 1849.1 | 250.0 | dropout_contract_relative_error=5.422e-08, dropout_probability_z=1.489583, dropped_channel_fraction=0.093018, expected_dropout_probability=0.100000, identifiable_channels=4096 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 0.7 | 0.7..0.7 | 1 | 0.387 | - | - | 0.7 | - (stored kernel) | 0.387 (kernel/kernel) | - | 1977.2 | 250.0 | dropout_contract_relative_error=5.421e-08, dropout_probability_z=1.791667, dropped_channel_fraction=0.108398, expected_dropout_probability=0.100000, identifiable_channels=4096 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/190582/vram_* bytes for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'p': 0.1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-fp32 | torch-eager-fp32 |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) |
| input_arrays | [{"dtype": "float32", "name": "x", "sha256": "cd7e0a7d8c0c594cca519b9cf49c8f086513282644cb8b6c06771dcb98504f2d", "shape": [64, 64, 56, 56]}] | [{"dtype": "float32", "name": "x", "sha256": "cd7e0a7d8c0c594cca519b9cf49c8f086513282644cb8b6c06771dcb98504f2d", "shape": [64, 64, 56, 56]}] | [{"dtype": "float32", "name": "x", "sha256": "cd7e0a7d8c0c594cca519b9cf49c8f086513282644cb8b6c06771dcb98504f2d", "shape": [64, 64, 56, 56]}] |
| p | 0.1 | 0.1 | 0.1 |
| seed | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | hardware | version | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | AMD Instinct MI325X (amd, DO) | 0.8.44 | Xq | - | 0.1 | 0.1..0.1 | 1 | - | - | - | 0.1 | - (stored kernel) | - | - | dropout_contract_relative_error=5.422e-08, dropout_probability_z=0.708333, dropped_channel_fraction=0.096680, expected_dropout_probability=0.100000, identifiable_channels=4096 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | Xq | - | 0.1 | 0.1..0.1 | 1 | 0.827 | - | - | 0.1 | - (stored kernel) | 0.827 (kernel/kernel) | - | dropout_contract_relative_error=5.422e-08, dropout_probability_z=1.489583, dropped_channel_fraction=0.093018, expected_dropout_probability=0.100000, identifiable_channels=4096 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | Xq | - | 0.1 | 0.1..0.1 | 1 | 0.660 | - | - | 0.1 | - (stored kernel) | 0.660 (kernel/kernel) | - | dropout_contract_relative_error=5.421e-08, dropout_probability_z=1.791667, dropped_channel_fraction=0.108398, expected_dropout_probability=0.100000, identifiable_channels=4096 | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

### dynamic-optimized-theta / synthetic (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1679)`, ran on AMD Instinct MI325X (amd, DO) job a1679

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 765.3 | 765.3..765.3 | 1 | - | - | 765.3 | - | - (stored whole) | - | - | - | - | forecast_rmse=1.436278 | - | main board, one scored run | - | ok (main@a1c96a5ba amd/a1679 2026-10-11; identity vs the nvidia columns: n/a) |
| statsforecast-cpu | statsforecast | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | statsforecast 2.1.1 | opponent | 354.2 | 354.2..354.2 | 1 | 2.161 | - | 354.2 | 354.2 | 0.00 (cpu-arm) | 2.161 (whole/whole) | - | 283.6 | - | forecast_rmse=1.436045 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, statsforecast-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'decomposition_type': 'multiplicative', 'season_length': 24}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### dynamic-optimized-theta / taxi-hourly (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1679)`, ran on AMD Instinct MI325X (amd, DO) job a1679

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 4411.8 | 4411.8..4411.8 | 1 | - | - | 4411.8 | - | - (stored whole) | - | - | - | - | forecast_rmse=49.086628 | - | main board, one scored run | - | ok (main@a1c96a5ba amd/a1679 2026-10-11; identity vs the nvidia columns: n/a) |
| statsforecast-cpu | statsforecast | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | statsforecast 2.1.1 | opponent | 798.7 | 798.7..798.7 | 1 | 5.524 | - | 798.7 | 798.7 | 0.00 (cpu-arm) | 5.524 (whole/whole) | - | 283.8 | - | forecast_rmse=49.314797 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, statsforecast-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'decomposition_type': 'multiplicative', 'season_length': 24}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### eigh / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/full-gpu-repair-20261014/recovered/algos.eigh.synthetic.rows=full/board.json (amd/paired-eb54108940e0efa172c4f43e74f7e020ff4d30a3061bcbcd3a9afc2243469f31)`, ran on mojolearn-steward-do-amd

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | 0.8.44 | identical | 524.7 | 524.7..524.7 | 1 | - | - | 524.7 | 523.1 | 1.57 (upload_ms_separate) | - | - | 3025.7 | 238765.7 | max_eigenvalue_error=3.692e-07, relative_residual=6.672e-07 | - | main board, one scored run | wheel | ok (main@d425c1417 amd/paired-eb54108940e0efa172c4f43e74f7e020ff4d30a3061bcbcd3a9afc2243469f31 2026-10-14; identity vs the nvidia columns: n/a) |
| torch-gpu | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 117.8 | 117.8..117.8 | 1 | 4.454 | - | - | 117.8 | - (stored kernel) | 4.440 (kernel/kernel) | - | 2287.2 | 400.7 | max_eigenvalue_error=1.044e-06, relative_residual=9.614e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/97631/vram_* bytes for this pid at the round's end (not a peak)

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'UPLO': 'L'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-gpu |
|---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) |
| UPLO | "L" | "L" |
| input_arrays | [{"dtype": "float32", "sha256": "2c4b0bf4609c4d52caa9d6bc06c51bfea085fc0f03a7b31ab31b9d26214119cb", "shape": [4096, 4096]}] | [{"dtype": "float32", "sha256": "2c4b0bf4609c4d52caa9d6bc06c51bfea085fc0f03a7b31ab31b9d26214119cb", "shape": [4096, 4096]}] |
| seed | 7 | 7 |

### elliptic-envelope / istella (rows full, shape X 100000x220; Xq 100000x220; y 100000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/r43/ingest/amd-results.txt (amd/a1589)`, ran on AMD Instinct MI325X (amd, DO) job a1589

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.42 (source build, main@a6b5f76cb) | identical | 239074.1 | 239074.1..239074.1 | 1 | - | - | 239074.1 | - | - (stored whole) | - | - | - | - | fraction_flagged=0.092480 | - | main board, one scored run | - | ok (main@a6b5f76cb amd/a1589 2026-10-13; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(timeout: null) (copied from release-board-resume-r2; measured this run) |

settings: {'contamination': 0.1, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### elliptic-envelope / taxi (rows full, shape X 100000x11; Xq 100000x11; y 100000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/r43/ingest/amd-results.txt (amd/a1589)`, ran on AMD Instinct MI325X (amd, DO) job a1589

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.42 (source build, main@a6b5f76cb) | identical | 9759.6 | 9759.6..9759.6 | 1 | - | - | 9759.6 | - | - (stored whole) | - | - | - | - | fraction_flagged=0.102090 | - | main board, one scored run | - | ok (main@a6b5f76cb amd/a1589 2026-10-13; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 1240.4 | 1240.4..1240.4 | 1 | 7.868 | - | 1240.4 | 1240.4 | 0.00 (cpu-arm) | 7.868 (whole/whole) | - | 368.1 | - | fraction_flagged=0.102470, jaccard_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'contamination': 0.1, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### embedding / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/amd-wave5/campaign/measured/algos.embedding.synthetic.rows=full/board.json (amd/paired-d385c9af47829b9f2d20f8a0cad1cbe5b9baccb183d80fbde7aa704a954145c6)`, ran on mojolearn-steward-do-amd

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 1.0 ms) = 1.520. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | 0.8.44 | identical | 1.6 | 1.6..1.6 | 1 | - | - | 1.6 | 1.5 | 0.07 (upload_ms_separate) | - | - | 3011.3 | 235893.3 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | main board, one scored run | wheel | ok (main@d11b82091 amd/paired-d385c9af47829b9f2d20f8a0cad1cbe5b9baccb183d80fbde7aa704a954145c6 2026-10-14; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.0 | 1.0..1.0 | 1 | 1.520 | - | - | 1.0 | - (stored kernel) | 1.447 (kernel/kernel) | - | 2705.8 | 782.6 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.3 | 1.3..1.3 | 1 | 1.191 | - | - | 1.3 | - (stored kernel) | 1.133 (kernel/kernel) | - | 2786.4 | 640.2 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/179300/vram_* bytes for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'embedding_dim': 1024, 'max_norm': None, 'norm_type': 2.0, 'num_embeddings': 32768, 'padding_idx': None, 'scale_grad_by_freq': False, 'sparse': False}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-fp32 | torch-eager-fp32 |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) |
| embedding_dim | 1024 | 1024 | 1024 |
| input_arrays | [{"dtype": "int64", "name": "x", "sha256": "cd5552f6f397dec3b3b97176b3b1ffff7b294e48376d95667419773fa3959e1b", "shape": [64, 512]}, {"dtype": "float32", "name": "weight", "sha256": "cec4f976c7feae09a100521e6e404c90584b25330e0bef803c3ae82efaa217e7", "shape": [32768, 1024]}] | [{"dtype": "int64", "name": "x", "sha256": "cd5552f6f397dec3b3b97176b3b1ffff7b294e48376d95667419773fa3959e1b", "shape": [64, 512]}, {"dtype": "float32", "name": "weight", "sha256": "cec4f976c7feae09a100521e6e404c90584b25330e0bef803c3ae82efaa217e7", "shape": [32768, 1024]}] | [{"dtype": "int64", "name": "x", "sha256": "cd5552f6f397dec3b3b97176b3b1ffff7b294e48376d95667419773fa3959e1b", "shape": [64, 512]}, {"dtype": "float32", "name": "weight", "sha256": "cec4f976c7feae09a100521e6e404c90584b25330e0bef803c3ae82efaa217e7", "shape": [32768, 1024]}] |
| max_norm | null | null | null |
| norm_type | 2.0 | 2.0 | 2.0 |
| num_embeddings | 32768 | 32768 | 32768 |
| padding_idx | null | null | null |
| scale_grad_by_freq | false | false | false |
| seed | "none (deterministic)" | 7 | 7 |
| sparse | false | false | false |

accepted difference: torch-eager-fp32 max_norm: None disables embedding-vector renormalization on both arms; all active arguments remain compared

accepted difference: torch-eager-fp32 padding_idx: None means there is no padding token on either embedding; all active arguments remain compared

accepted difference: torch-compile-fp32 max_norm: None disables embedding-vector renormalization on both arms; all active arguments remain compared

accepted difference: torch-compile-fp32 padding_idx: None means there is no padding token on either embedding; all active arguments remain compared

Inference (each arm predicts with its own model from the fit rounds above):

| arm | hardware | version | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | AMD Instinct MI325X (amd, DO) | 0.8.44 | Xq | - | 0.3 | 0.3..0.3 | 1 | - | - | 0.3 | - | - (stored whole) | - | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | Xq | - | 0.1 | 0.1..0.1 | 1 | 3.086 | - | - | 0.1 | - (stored kernel) | 3.086 (MIXED ours whole / arm kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | Xq | - | 0.2 | 0.2..0.2 | 1 | 1.758 | - | - | 0.2 | - (stored kernel) | 1.758 (MIXED ours whole / arm kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

### enet-cv / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1645)`, ran on AMD Instinct MI325X (amd, DO) job a1645

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@2afa942e3) | identical | 854.8 | 854.8..854.8 | 1 | - | - | 854.8 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.326805, rmse=0.685379 | - | main board, one scored run | - | ok (main@2afa942e3 amd/a1645 2026-10-11; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 47198.6 | 47198.6..47198.6 | 1 | 0.018 | - | 47198.6 | 47198.6 | 0.00 (cpu-arm) | 0.018 (whole/whole) | - | 8285.3 | - | finite=True, r2=0.326805, rmse=0.685379 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alphas': [0.001, 0.01, 0.1, 1.0], 'cv': 5, 'eps': 0.001, 'fit_intercept': True, 'l1_ratio': 0.5, 'max_iter': 1000, 'positive': False, 'random_state': 7, 'selection': 'cyclic', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### enet-cv / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1645)`, ran on AMD Instinct MI325X (amd, DO) job a1645

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@2afa942e3) | identical | 222.8 | 222.8..222.8 | 1 | - | - | 222.8 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.909004, rmse=4.804486 | - | main board, one scored run | - | ok (main@2afa942e3 amd/a1645 2026-10-11; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 279.5 | 279.5..279.5 | 1 | 0.797 | - | 279.5 | 279.5 | 0.00 (cpu-arm) | 0.797 (whole/whole) | - | 673.5 | - | finite=True, r2=0.909004, rmse=4.804486 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alphas': [0.001, 0.01, 0.1, 1.0], 'cv': 5, 'eps': 0.001, 'fit_intercept': True, 'l1_ratio': 0.5, 'max_iter': 1000, 'positive': False, 'random_state': 7, 'selection': 'cyclic', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gaussian-nb / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1130)`, ran on AMD Instinct MI325X (amd, DO) job a1130

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@a47bd9fb2) | identical | 28.2 | 28.2..28.2 | 1 | - | - | 28.2 | - | - (stored whole) | - | - | - | - | accuracy=0.876570, logloss=3.574420 | - | main board, one scored run | - | ok (main@a47bd9fb2 amd/a1130 2026-10-09; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 420.3 | 420.3..420.3 | 1 | 0.067 | - | 420.3 | 420.3 | 0.00 (cpu-arm) | 0.067 (whole/whole) | - | 2633.4 | - | accuracy=0.876530, logloss=3.417392 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'var_smoothing': 1e-09}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), GaussianNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gaussian-nb / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1130)`, ran on AMD Instinct MI325X (amd, DO) job a1130

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@a47bd9fb2) | identical | 12.2 | 12.2..12.2 | 1 | - | - | 12.2 | - | - (stored whole) | - | - | - | - | accuracy=0.719820, logloss=1.132249 | - | main board, one scored run | - | ok (main@a47bd9fb2 amd/a1130 2026-10-09; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 82.1 | 82.1..82.1 | 1 | 0.149 | - | 82.1 | 82.1 | 0.00 (cpu-arm) | 0.149 (whole/whole) | - | 324.2 | - | accuracy=0.719900, logloss=1.133898 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'var_smoothing': 1e-09}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), GaussianNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gcn / istella (rows full, shape X 100000x220; indices 1521510; indptr 100001; y 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1210)`, ran on AMD Instinct MI325X (amd, DO) job a1210

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 3.3 ms) = 0.921; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 3.6 ms) = 0.834. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@6416999f7) | identical | 3.0 | 3.0..3.0 | 1 | - | - | 3.0 | - | - (stored whole) | - | - | - | - | error=Own host reference failed (bba_neural_fixtures.FixturesMissing: neural fixtures missing: run tools/neural_fixtures.py generate (tools/neural_fixtures_box.sh generate on a box, then stage) (no fixture directory: pass --neural-fixtures)); see /root/lq/out/a1210/work-gcn-istella-def/gcn-istella-host-quality/host.log | - | main board, one scored run | - | ok (main@6416999f7 amd/a1210 2026-10-10; identity vs the nvidia columns: n/a; newer run a8fd6d926/a0006 failed: not_ready) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 16.9 | 16.9..16.9 | 1 | 0.180 | - | - | 16.9 | - (stored kernel) | 0.180 (MIXED ours whole / arm kernel) | - | 3417.9 | 1918.3 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 3.6 | 3.6..3.6 | 1 | 0.834 | - | - | 3.6 | - (stored kernel) | 0.834 (MIXED ours whole / arm kernel) | - | 3477.3 | 383.2 | max_rel_diff_vs_torch_eager_fp32=0.025557, rel_fro_vs_torch_eager_fp32=9.36e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 10.7 | 10.7..10.7 | 1 | 0.285 | - | - | 10.7 | - (stored kernel) | 0.285 (MIXED ours whole / arm kernel) | - | 4528.1 | 2307.5 | max_rel_diff_vs_torch_eager_fp32=2662.412698, rel_fro_vs_torch_eager_fp32=0.002187 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 3.3 | 3.3..3.3 | 1 | 0.921 | - | - | 3.3 | - (stored kernel) | 0.921 (MIXED ours whole / arm kernel) | - | 5066.5 | 766.2 | max_rel_diff_vs_torch_eager_fp32=2662.401917, rel_fro_vs_torch_eager_fp32=0.002187 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'add_self_loops': True, 'bias': True, 'improved': False, 'normalize': True, 'out_channels': 128}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gcn / taxi (rows full, shape X 100000x11; indices 1258298; indptr 100001; y 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1210)`, ran on AMD Instinct MI325X (amd, DO) job a1210

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 2.8 ms) = 0.491; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 3.2 ms) = 0.428. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@6416999f7) | identical | 1.4 | 1.4..1.4 | 1 | - | - | 1.4 | - | - (stored whole) | - | - | - | - | error=Own host reference failed (bba_neural_fixtures.FixturesMissing: neural fixtures missing: run tools/neural_fixtures.py generate (tools/neural_fixtures_box.sh generate on a box, then stage) (no fixture directory: pass --neural-fixtures)); see /root/lq/out/a1210/work-gcn-taxi-def/gcn-taxi-host-quality/host.log | - | main board, one scored run | - | ok (main@6416999f7 amd/a1210 2026-10-10; identity vs the nvidia columns: n/a; newer run a8fd6d926/a0006 failed: not_ready) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 14.1 | 14.1..14.1 | 1 | 0.097 | - | - | 14.1 | - (stored kernel) | 0.097 (MIXED ours whole / arm kernel) | - | 3333.2 | 1573.4 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 3.2 | 3.2..3.2 | 1 | 0.428 | - | - | 3.2 | - (stored kernel) | 0.428 (MIXED ours whole / arm kernel) | - | 3461.6 | 295.5 | max_rel_diff_vs_torch_eager_fp32=0.001702, rel_fro_vs_torch_eager_fp32=7.846e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 9.0 | 9.0..9.0 | 1 | 0.152 | - | - | 9.0 | - (stored kernel) | 0.152 (MIXED ours whole / arm kernel) | - | 4436.3 | 1858.7 | max_rel_diff_vs_torch_eager_fp32=1509.509399, rel_fro_vs_torch_eager_fp32=0.002330 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 2.8 | 2.8..2.8 | 1 | 0.491 | - | - | 2.8 | - (stored kernel) | 0.491 (MIXED ours whole / arm kernel) | - | 4978.9 | 574.3 | max_rel_diff_vs_torch_eager_fp32=1509.509515, rel_fro_vs_torch_eager_fp32=0.002330 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'add_self_loops': True, 'bias': True, 'improved': False, 'normalize': True, 'out_channels': 128}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### global-avgpool / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/amd-wave5/campaign/measured/algos.global-avgpool.synthetic.rows=full/board.json (amd/paired-7f40c2d85a6a832dbf13528a9897fe3c94118d5086d1c26c9880b8d7404e5975)`, ran on mojolearn-steward-do-amd

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 0.5 ms) = 0.540; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 0.7 ms) = 0.373. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | 0.8.44 | identical | 0.2 | 0.2..0.2 | 1 | - | - | - | 0.2 | - (stored kernel) | - | - | 700.6 | 235669.5 | max_rel_diff_vs_torch_eager_fp32=0.015661, rel_fro_vs_torch_eager_fp32=1.506e-07 | - | main board, one scored run | wheel | ok (main@d11b82091 amd/paired-7f40c2d85a6a832dbf13528a9897fe3c94118d5086d1c26c9880b8d7404e5975 2026-10-14; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 0.7 | 0.7..0.7 | 1 | 0.373 | - | - | 0.7 | - (stored kernel) | 0.373 (kernel/kernel) | - | 1687.0 | 12.6 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 0.9 | 0.9..0.9 | 1 | 0.257 | - | - | 0.9 | - (stored kernel) | 0.257 (kernel/kernel) | - | 2051.8 | 12.6 | max_rel_diff_vs_torch_eager_fp32=0.007299, rel_fro_vs_torch_eager_fp32=1.01e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 0.5 | 0.5..0.5 | 1 | 0.540 | - | - | 0.5 | - (stored kernel) | 0.540 (kernel/kernel) | - | 1686.3 | 12.6 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 0.6 | 0.6..0.6 | 1 | 0.422 | - | - | 0.6 | - (stored kernel) | 0.422 (kernel/kernel) | - | 1980.5 | 12.6 | max_rel_diff_vs_torch_eager_fp32=0.007299, rel_fro_vs_torch_eager_fp32=1.01e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/181221/vram_* bytes for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'output_size': 1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| input_arrays | [{"dtype": "float32", "name": "x", "sha256": "8fe0278be594910f0b7581c036615fa6de591a3cb29304f63bff7d16046f6c97", "shape": [64, 512, 7, 7]}] | [{"dtype": "float32", "name": "x", "sha256": "8fe0278be594910f0b7581c036615fa6de591a3cb29304f63bff7d16046f6c97", "shape": [64, 512, 7, 7]}] | [{"dtype": "float32", "name": "x", "sha256": "8fe0278be594910f0b7581c036615fa6de591a3cb29304f63bff7d16046f6c97", "shape": [64, 512, 7, 7]}] | [{"dtype": "float32", "name": "x", "sha256": "8fe0278be594910f0b7581c036615fa6de591a3cb29304f63bff7d16046f6c97", "shape": [64, 512, 7, 7]}] | [{"dtype": "float32", "name": "x", "sha256": "8fe0278be594910f0b7581c036615fa6de591a3cb29304f63bff7d16046f6c97", "shape": [64, 512, 7, 7]}] |
| output_size | 1 | 1 | 1 | 1 | 1 |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | hardware | version | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | AMD Instinct MI325X (amd, DO) | 0.8.44 | Xq | - | 0.1 | 0.1..0.1 | 1 | - | - | - | 0.1 | - (stored kernel) | - | - | max_rel_diff_vs_torch_eager_fp32=0.015661, rel_fro_vs_torch_eager_fp32=1.506e-07 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | Xq | - | 0.0 | 0.0..0.0 | 1 | 1.213 | - | - | 0.0 | - (stored kernel) | 1.213 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | Xq | - | 0.1 | 0.1..0.1 | 1 | 0.452 | - | - | 0.1 | - (stored kernel) | 0.452 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.007299, rel_fro_vs_torch_eager_fp32=1.01e-07 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | Xq | - | 0.0 | 0.0..0.0 | 1 | 1.260 | - | - | 0.0 | - (stored kernel) | 1.260 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | Xq | - | 0.1 | 0.1..0.1 | 1 | 0.524 | - | - | 0.1 | - (stored kernel) | 0.524 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.007299, rel_fro_vs_torch_eager_fp32=1.01e-07 | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### global-maxpool / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/amd-wave5/campaign/measured/algos.global-maxpool.synthetic.rows=full/board.json (amd/paired-f721f4183d1237b97766ade5525d0022088cfbe17bd6c713aa41695fd4433667)`, ran on mojolearn-steward-do-amd

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 0.5 ms) = 0.516; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 0.6 ms) = 0.416. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | 0.8.44 | identical | 0.2 | 0.2..0.2 | 1 | - | - | - | 0.2 | - (stored kernel) | - | - | 700.5 | 235669.5 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | main board, one scored run | wheel | ok (main@d11b82091 amd/paired-f721f4183d1237b97766ade5525d0022088cfbe17bd6c713aa41695fd4433667 2026-10-14; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 0.6 | 0.6..0.6 | 1 | 0.416 | - | - | 0.6 | - (stored kernel) | 0.416 (kernel/kernel) | - | 1662.0 | 12.9 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 0.8 | 0.8..0.8 | 1 | 0.305 | - | - | 0.8 | - (stored kernel) | 0.305 (kernel/kernel) | - | 2077.1 | 12.9 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 0.5 | 0.5..0.5 | 1 | 0.516 | - | - | 0.5 | - (stored kernel) | 0.516 (kernel/kernel) | - | 1661.6 | 12.9 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 0.6 | 0.6..0.6 | 1 | 0.373 | - | - | 0.6 | - (stored kernel) | 0.373 (kernel/kernel) | - | 1998.7 | 12.9 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/182623/vram_* bytes for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'output_size': 1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| input_arrays | [{"dtype": "float32", "name": "x", "sha256": "8fe0278be594910f0b7581c036615fa6de591a3cb29304f63bff7d16046f6c97", "shape": [64, 512, 7, 7]}] | [{"dtype": "float32", "name": "x", "sha256": "8fe0278be594910f0b7581c036615fa6de591a3cb29304f63bff7d16046f6c97", "shape": [64, 512, 7, 7]}] | [{"dtype": "float32", "name": "x", "sha256": "8fe0278be594910f0b7581c036615fa6de591a3cb29304f63bff7d16046f6c97", "shape": [64, 512, 7, 7]}] | [{"dtype": "float32", "name": "x", "sha256": "8fe0278be594910f0b7581c036615fa6de591a3cb29304f63bff7d16046f6c97", "shape": [64, 512, 7, 7]}] | [{"dtype": "float32", "name": "x", "sha256": "8fe0278be594910f0b7581c036615fa6de591a3cb29304f63bff7d16046f6c97", "shape": [64, 512, 7, 7]}] |
| output_size | 1 | 1 | 1 | 1 | 1 |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | hardware | version | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | AMD Instinct MI325X (amd, DO) | 0.8.44 | Xq | - | 0.1 | 0.1..0.1 | 1 | - | - | - | 0.1 | - (stored kernel) | - | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | Xq | - | 0.1 | 0.1..0.1 | 1 | 0.643 | - | - | 0.1 | - (stored kernel) | 0.643 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | Xq | - | 0.1 | 0.1..0.1 | 1 | 0.486 | - | - | 0.1 | - (stored kernel) | 0.486 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | Xq | - | 0.1 | 0.1..0.1 | 1 | 0.570 | - | - | 0.1 | - (stored kernel) | 0.570 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | Xq | - | 0.1 | 0.1..0.1 | 1 | 0.535 | - | - | 0.1 | - (stored kernel) | 0.535 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### graphsage / istella (rows full, shape X 100000x220; indices 1521510; indptr 100001; y 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1210)`, ran on AMD Instinct MI325X (amd, DO) job a1210

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 3.2 ms) = 1.843; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 3.4 ms) = 1.756. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@6416999f7) | identical | 5.9 | 5.9..5.9 | 1 | - | - | 5.9 | - | - (stored whole) | - | - | - | - | error=Own host reference failed (bba_neural_fixtures.FixturesMissing: neural fixtures missing: run tools/neural_fixtures.py generate (tools/neural_fixtures_box.sh generate on a box, then stage) (no fixture directory: pass --neural-fixtures)); see /root/lq/out/a1210/work-graphsage-istella-def/graphsage-istella-host-quality/host.log | - | main board, one scored run | - | ok (main@6416999f7 amd/a1210 2026-10-10; identity vs the nvidia columns: n/a; newer run a8fd6d926/a0006 failed: not_ready) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 12.5 | 12.5..12.5 | 1 | 0.474 | - | - | 12.5 | - (stored kernel) | 0.474 (MIXED ours whole / arm kernel) | - | 3373.3 | 1651.2 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 3.4 | 3.4..3.4 | 1 | 1.756 | - | - | 3.4 | - (stored kernel) | 1.756 (MIXED ours whole / arm kernel) | - | 3458.5 | 463.5 | max_rel_diff_vs_torch_eager_fp32=0.140845, rel_fro_vs_torch_eager_fp32=8.05e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 12.6 | 12.6..12.6 | 1 | 0.469 | - | - | 12.6 | - (stored kernel) | 0.469 (MIXED ours whole / arm kernel) | - | 4489.7 | 1626.7 | max_rel_diff_vs_torch_eager_fp32=3907.144070, rel_fro_vs_torch_eager_fp32=0.003366 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 3.2 | 3.2..3.2 | 1 | 1.843 | - | - | 3.2 | - (stored kernel) | 1.843 (MIXED ours whole / arm kernel) | - | 5119.7 | 433.0 | max_rel_diff_vs_torch_eager_fp32=3678.232431, rel_fro_vs_torch_eager_fp32=0.003054 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'aggr': 'mean', 'bias': True, 'normalize': False, 'out_channels': 128, 'project': False, 'root_weight': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### graphsage / taxi (rows full, shape X 100000x11; indices 1258298; indptr 100001; y 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1210)`, ran on AMD Instinct MI325X (amd, DO) job a1210

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 1.4 ms) = 1.190; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 1.4 ms) = 1.157. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@6416999f7) | identical | 1.6 | 1.6..1.6 | 1 | - | - | 1.6 | - | - (stored whole) | - | - | - | - | error=Own host reference failed (bba_neural_fixtures.FixturesMissing: neural fixtures missing: run tools/neural_fixtures.py generate (tools/neural_fixtures_box.sh generate on a box, then stage) (no fixture directory: pass --neural-fixtures)); see /root/lq/out/a1210/work-graphsage-taxi-def/graphsage-taxi-host-quality/host.log | - | main board, one scored run | - | ok (main@6416999f7 amd/a1210 2026-10-10; identity vs the nvidia columns: n/a; newer run a8fd6d926/a0006 failed: not_ready) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1.4 | 1.4..1.4 | 1 | 1.157 | - | - | 1.4 | - (stored kernel) | 1.157 (MIXED ours whole / arm kernel) | - | 3308.4 | 299.7 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1.5 | 1.5..1.5 | 1 | 1.086 | - | - | 1.5 | - (stored kernel) | 1.086 (MIXED ours whole / arm kernel) | - | 3439.0 | 299.7 | max_rel_diff_vs_torch_eager_fp32=0.162162, rel_fro_vs_torch_eager_fp32=8.776e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1.5 | 1.5..1.5 | 1 | 1.124 | - | - | 1.5 | - (stored kernel) | 1.124 (MIXED ours whole / arm kernel) | - | 4422.1 | 227.1 | max_rel_diff_vs_torch_eager_fp32=4680.142857, rel_fro_vs_torch_eager_fp32=0.003557 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1.4 | 1.4..1.4 | 1 | 1.190 | - | - | 1.4 | - (stored kernel) | 1.190 (MIXED ours whole / arm kernel) | - | 5037.8 | 230.2 | max_rel_diff_vs_torch_eager_fp32=4608.154297, rel_fro_vs_torch_eager_fp32=0.003285 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'aggr': 'mean', 'bias': True, 'normalize': False, 'out_channels': 128, 'project': False, 'root_weight': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gru-clf / synthetic (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1211)`, ran on AMD Instinct MI325X (amd, DO) job a1211

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 1852.7 ms) = 0.278; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 1946.7 ms) = 0.265. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 514.9 | 514.9..514.9 | 1 | - | - | 514.9 | - | - (stored whole) | - | - | - | - | accuracy=0.971625, logloss=0.070054 | - | main board, one scored run | - | ok (main@1665b5626 amd/a1211 2026-10-10; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1946.7 | 1946.7..1946.7 | 1 | 0.265 | - | - | 1946.7 | - (stored kernel) | 0.265 (MIXED ours whole / arm kernel) | - | 3217.3 | 324.7 | accuracy=0.971842 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1997.8 | 1997.8..1997.8 | 1 | 0.258 | - | - | 1997.8 | - (stored kernel) | 0.258 (MIXED ours whole / arm kernel) | - | 3248.0 | 324.7 | accuracy=0.971842 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1852.7 | 1852.7..1852.7 | 1 | 0.278 | - | - | 1852.7 | - (stored kernel) | 0.278 (MIXED ours whole / arm kernel) | - | 4957.7 | 168.7 | accuracy=0.971788 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1973.8 | 1973.8..1973.8 | 1 | 0.261 | - | - | 1973.8 | - (stored kernel) | 0.261 (MIXED ours whole / arm kernel) | - | 5008.5 | 168.7 | accuracy=0.971788 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gru-clf / taxi-hourly (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1211)`, ran on AMD Instinct MI325X (amd, DO) job a1211

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 1899.7 ms) = 0.271; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 1792.8 ms) = 0.288. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 515.5 | 515.5..515.5 | 1 | - | - | 515.5 | - | - (stored whole) | - | - | - | - | accuracy=0.865723, logloss=0.303537 | - | main board, one scored run | - | ok (main@1665b5626 amd/a1211 2026-10-10; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1792.8 | 1792.8..1792.8 | 1 | 0.288 | - | - | 1792.8 | - (stored kernel) | 0.288 (MIXED ours whole / arm kernel) | - | 3610.7 | 324.7 | accuracy=0.865668 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 2027.5 | 2027.5..2027.5 | 1 | 0.254 | - | - | 2027.5 | - (stored kernel) | 0.254 (MIXED ours whole / arm kernel) | - | 3258.5 | 324.7 | accuracy=0.865668 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 2025.2 | 2025.2..2025.2 | 1 | 0.255 | - | - | 2025.2 | - (stored kernel) | 0.255 (MIXED ours whole / arm kernel) | - | 5294.2 | 168.7 | accuracy=0.865777 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1899.7 | 1899.7..1899.7 | 1 | 0.271 | - | - | 1899.7 | - (stored kernel) | 0.271 (MIXED ours whole / arm kernel) | - | 5008.7 | 168.7 | accuracy=0.865777 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gru-reg / synthetic (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1211)`, ran on AMD Instinct MI325X (amd, DO) job a1211

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 2078.2 ms) = 0.247; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 1993.7 ms) = 0.257. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 513.3 | 513.3..513.3 | 1 | - | - | 513.3 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.982337, rmse=0.153975 | - | main board, one scored run | - | ok (main@1665b5626 amd/a1211 2026-10-10; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1993.7 | 1993.7..1993.7 | 1 | 0.257 | - | - | 1993.7 | - (stored kernel) | 0.257 (MIXED ours whole / arm kernel) | - | 3044.7 | 324.4 | finite=True, r2=0.981946, rmse=0.155672 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1996.8 | 1996.8..1996.8 | 1 | 0.257 | - | - | 1996.8 | - (stored kernel) | 0.257 (MIXED ours whole / arm kernel) | - | 3072.8 | 324.4 | finite=True, r2=0.981946, rmse=0.155672 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 2078.2 | 2078.2..2078.2 | 1 | 0.247 | - | - | 2078.2 | - (stored kernel) | 0.247 (MIXED ours whole / arm kernel) | - | 3656.3 | 168.4 | finite=True, r2=0.981940, rmse=0.155700 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 2123.0 | 2123.0..2123.0 | 1 | 0.242 | - | - | 2123.0 | - (stored kernel) | 0.242 (MIXED ours whole / arm kernel) | - | 3708.5 | 168.4 | finite=True, r2=0.981940, rmse=0.155700 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gru-reg / taxi-hourly (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1211)`, ran on AMD Instinct MI325X (amd, DO) job a1211

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 1801.3 ms) = 0.285; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 2006.3 ms) = 0.256. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 513.4 | 513.4..513.4 | 1 | - | - | 513.4 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.747875, rmse=0.544554 | - | main board, one scored run | - | ok (main@1665b5626 amd/a1211 2026-10-10; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 2012.1 | 2012.1..2012.1 | 1 | 0.255 | - | - | 2012.1 | - (stored kernel) | 0.255 (MIXED ours whole / arm kernel) | - | 3044.6 | 324.4 | finite=True, r2=0.748219, rmse=0.544182 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 2006.3 | 2006.3..2006.3 | 1 | 0.256 | - | - | 2006.3 | - (stored kernel) | 0.256 (MIXED ours whole / arm kernel) | - | 3072.3 | 324.4 | finite=True, r2=0.748219, rmse=0.544182 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1801.3 | 1801.3..1801.3 | 1 | 0.285 | - | - | 1801.3 | - (stored kernel) | 0.285 (MIXED ours whole / arm kernel) | - | 3655.7 | 168.4 | finite=True, r2=0.748254, rmse=0.544144 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 2127.7 | 2127.7..2127.7 | 1 | 0.241 | - | - | 2127.7 | - (stored kernel) | 0.241 (MIXED ours whole / arm kernel) | - | 3708.4 | 168.4 | finite=True, r2=0.748254, rmse=0.544144 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ivf-pq / istella (rows full, shape index 400000x220; queries 4000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1710)`, ran on AMD Instinct MI325X (amd, DO) job a1710

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 541.5 | 541.5..541.5 | 1 | - | - | 541.5 | - | - (stored whole) | - | - | - | - | recall_at_10=0.802100 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1710 2026-10-11; identity vs the nvidia columns: n/a) |
| faiss-cpu | faiss | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | faiss 1.15.1 | opponent | 5118.6 | 5118.6..5118.6 | 1 | 0.106 | - | 5118.6 | 5118.6 | 0.00 (cpu-arm) | 0.106 (whole/whole) | - | 754.2 | - | recall_at_10=0.801550 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, faiss-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'kmeans_n_iters': 20, 'n_lists': 1024, 'n_neighbors': 10, 'n_probes': 32, 'pq_bits': 8, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ivf-pq / taxi (rows full, shape index 400000x11; queries 4000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1160)`, ran on AMD Instinct MI325X (amd, DO) job a1160

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@0a7b206f1) | identical | 84.6 | 84.6..84.6 | 1 | - | - | 84.6 | - | - (stored whole) | - | - | - | - | recall_at_10=0.981250 | - | main board, one scored run | - | ok (main@0a7b206f1 amd/a1160 2026-10-09; identity vs nvidia-l40s: MATCH) |
| faiss-cpu | faiss | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | faiss 1.15.1 | opponent | 914.0 | 914.0..914.0 | 1 | 0.093 | - | 914.0 | 914.0 | 0.00 (cpu-arm) | 0.093 (whole/whole) | - | 142.7 | - | recall_at_10=0.979450 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, faiss-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'kmeans_n_iters': 20, 'n_lists': 1024, 'n_neighbors': 10, 'n_probes': 32, 'pq_bits': 8, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ivf-refine / istella (rows full, shape index 400000x220; queries 4000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1710)`, ran on AMD Instinct MI325X (amd, DO) job a1710

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 543.0 | 543.0..543.0 | 1 | - | - | 543.0 | - | - (stored whole) | - | - | - | - | recall_at_10=0.994575 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1710 2026-10-11; identity vs the nvidia columns: n/a) |
| faiss-cpu | faiss | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | faiss 1.15.1 | opponent | 5684.0 | 5684.0..5684.0 | 1 | 0.096 | - | 5684.0 | 5684.0 | 0.00 (cpu-arm) | 0.096 (whole/whole) | - | 1313.8 | - | recall_at_10=0.993125 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, faiss-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'kmeans_n_iters': 20, 'n_lists': 1024, 'n_neighbors': 10, 'n_probes': 32, 'pq_bits': 8, 'random_state': 7, 'refine_ratio': 4}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### jl-min-dim / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/amd_ingest/results.txt (amd/a0005)`, ran on AMD Instinct MI325X (amd, DO) job a0005

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@a8fd6d926) | identical | 6.5 | 6.5..6.5 | 1 | - | - | 6.5 | - | - (stored whole) | - | - | - | - | equal_fraction_vs_sklearn=1.000000 | - | main board, one scored run | - | ok (main@a8fd6d926 amd/a0005 2026-10-12; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 164.9 | 164.9..164.9 | 1 | 0.039 | - | 164.9 | 164.9 | 0.00 (cpu-arm) | 0.039 (whole/whole) | - | 204.8 | - | equal_fraction_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### kbins / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/amd_ingest/results.txt (amd/a0005)`, ran on AMD Instinct MI325X (amd, DO) job a0005

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@a8fd6d926) | identical | 126.0 | 126.0..126.0 | 1 | - | - | 126.0 | - | - (stored whole) | - | - | - | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x220, rel_diff_vs_sklearn=0.000000 | - | main board, one scored run | - | ok (main@a8fd6d926 amd/a0005 2026-10-12; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 2318.5 | 2318.5..2318.5 | 1 | 0.054 | - | 2318.5 | 2318.5 | 0.00 (cpu-arm) | 0.054 (whole/whole) | - | 1470.6 | - | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'encode': 'ordinal', 'n_bins': 16, 'quantile_method': 'linear', 'random_state': 7, 'strategy': 'quantile', 'subsample': None}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), KBinsDiscretizer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### kbins / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/amd_ingest/results.txt (amd/a0005)`, ran on AMD Instinct MI325X (amd, DO) job a0005

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@a8fd6d926) | identical | 9.6 | 9.6..9.6 | 1 | - | - | 9.6 | - | - (stored whole) | - | - | - | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x11, rel_diff_vs_sklearn=0.000000 | - | main board, one scored run | - | ok (main@a8fd6d926 amd/a0005 2026-10-12; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 108.6 | 108.6..108.6 | 1 | 0.089 | - | 108.6 | 108.6 | 0.00 (cpu-arm) | 0.089 (whole/whole) | - | 272.7 | - | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'encode': 'ordinal', 'n_bins': 16, 'quantile_method': 'linear', 'random_state': 7, 'strategy': 'quantile', 'subsample': None}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), KBinsDiscretizer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### kernel-pca / istella (rows full, shape X 10000x220; Xq 10000x220; y 10000; yq 10000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/amd_ingest/results.txt (amd/a0005)`, ran on AMD Instinct MI325X (amd, DO) job a0005

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@a8fd6d926) | identical | 695.7 | 695.7..695.7 | 1 | - | - | 695.7 | - | - (stored whole) | - | - | - | - | rbf_feature_distance_relative_stress=0.610112, subspace_cos_vs_sklearn=1.000000 | - | main board, one scored run | - | ok (main@a8fd6d926 amd/a0005 2026-10-12; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 591.7 | 591.7..591.7 | 1 | 1.176 | - | 591.7 | 591.7 | 0.00 (cpu-arm) | 1.176 (whole/whole) | - | 2674.5 | - | subspace_cos_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'coef0': 1, 'degree': 3, 'eigen_solver': 'auto', 'fit_inverse_transform': False, 'iterated_power': 'auto', 'kernel': 'rbf', 'n_components': 8, 'random_state': 7, 'remove_zero_eig': False, 'tol': 0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### kernel-pca / taxi (rows full, shape X 10000x11; Xq 10000x11; y 10000; yq 10000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/amd_ingest/results.txt (amd/a0005)`, ran on AMD Instinct MI325X (amd, DO) job a0005

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@a8fd6d926) | identical | 556.6 | 556.6..556.6 | 1 | - | - | 556.6 | - | - (stored whole) | - | - | - | - | rbf_feature_distance_relative_stress=0.545709, subspace_cos_vs_sklearn=1.000000 | - | main board, one scored run | - | ok (main@a8fd6d926 amd/a0005 2026-10-12; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 506.8 | 506.8..506.8 | 1 | 1.098 | - | 506.8 | 506.8 | 0.00 (cpu-arm) | 1.098 (whole/whole) | - | 1671.0 | - | subspace_cos_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'coef0': 1, 'degree': 3, 'eigen_solver': 'auto', 'fit_inverse_transform': False, 'iterated_power': 'auto', 'kernel': 'rbf', 'n_components': 8, 'random_state': 7, 'remove_zero_eig': False, 'tol': 0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### kernel-shap / istella (rows full, shape X 100000x220; Xq 100x220; y 100000; yq 100)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1692)`, ran on AMD Instinct MI325X (amd, DO) job a1692

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 8243.6 | 8243.6..8243.6 | 1 | - | - | 8243.6 | - | - (stored whole) | - | - | - | - | rel_error_vs_exact=8.307e-08 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1692 2026-10-11; identity vs the nvidia columns: n/a) |
| shap-cpu | shap | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | shap 0.51.0 | opponent | 10339.5 | 10339.5..10339.5 | 1 | 0.797 | - | 10339.5 | 10339.5 | 0.00 (cpu-arm) | 0.797 (whole/whole) | - | 2196.3 | - | rel_error_vs_exact=8.63e-15 | - | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, shap-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'l1_reg': False, 'link': 'identity', 'n_background': 100, 'nsamples': 2048}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### kernel-shap / taxi (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0904)`, ran on AMD Instinct MI325X (amd, DO) job a0904

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@03b648834) | identical | 252.6 | 252.6..252.6 | 1 | - | - | 252.6 | - | - (stored whole) | - | - | - | - | rel_error_vs_exact=1.056e-07 | - | main board, one scored run | - | ok (main@03b648834 amd/a0904 2026-10-09; identity vs nvidia-l40s: MATCH) |

settings: {'n_background': 100, 'nsamples': 2048, 'link': 'identity', 'l1_reg': False, 'explained_model': 'one float32 ridge (alpha 1, centered) on every arm, coefficients solved once in a fixed order by the harness'}. Rows: None. Timed: None.

mismatch: opponents withheld: the lane settings at this tree's HEAD differ from the settings release-board-resume-r2 recorded for its opponent race (an opponent job must score them again)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### knn-imputer / istella (rows full, shape X 100000x220; X_true 100000x220; Xq 20000x220; Xq_true 20000x220; y 100000; yq 20000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1686)`, ran on AMD Instinct MI325X (amd, DO) job a1686

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 22.9 | 22.9..22.9 | 1 | - | - | 22.9 | - | - (stored whole) | - | - | - | - | masked_rmse=323953.237332 | - | main board, one scored run | - | ok (main@a1c96a5ba amd/a1686 2026-10-11; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 16.2 | 16.2..16.2 | 1 | 1.413 | - | 16.2 | 16.2 | 0.00 (cpu-arm) | 1.413 (whole/whole) | - | 3528.4 | - | masked_rmse=986208.700423 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'add_indicator': False, 'keep_empty_features': False, 'metric': 'nan_euclidean', 'n_neighbors': 5, 'weights': 'uniform'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### knn-imputer / taxi (rows full, shape X 100000x11; X_true 100000x11; Xq 20000x11; Xq_true 20000x11; y 100000; yq 20000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1160)`, ran on AMD Instinct MI325X (amd, DO) job a1160

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@0a7b206f1) | identical | 1.1 | 1.1..1.1 | 1 | - | - | 1.1 | - | - (stored whole) | - | - | - | - | masked_rmse=6.151696 | - | main board, one scored run | - | ok (main@0a7b206f1 amd/a1160 2026-10-09; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 1.9 | 1.9..1.9 | 1 | 0.579 | - | 1.9 | 1.9 | 0.00 (cpu-arm) | 0.579 (whole/whole) | - | 1966.8 | - | masked_rmse=5.263919 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'add_indicator': False, 'keep_empty_features': False, 'metric': 'nan_euclidean', 'n_neighbors': 5, 'weights': 'uniform'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### label-binarizer / istella (rows full, shape X 1000000x8; Xq 100000x8; lab 1000000; labq 100000; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/amd_ingest/results.txt (amd/a0005)`, ran on AMD Instinct MI325X (amd, DO) job a0005

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@a8fd6d926) | identical | 16.1 | 16.1..16.1 | 1 | - | - | 16.1 | - | - (stored whole) | - | - | - | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x16, rel_diff_vs_sklearn=0.000000 | - | main board, one scored run | - | ok (main@a8fd6d926 amd/a0005 2026-10-12; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 51.9 | 51.9..51.9 | 1 | 0.310 | - | 51.9 | 51.9 | 0.00 (cpu-arm) | 0.310 (whole/whole) | - | 680.8 | - | output_shape=100000x16 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'neg_label': 0, 'pos_label': 1}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), LabelBinarizer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### label-binarizer / taxi (rows full, shape X 1000000x5; Xq 100000x5; lab 1000000; labq 100000; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/amd_ingest/results.txt (amd/a0005)`, ran on AMD Instinct MI325X (amd, DO) job a0005

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@a8fd6d926) | identical | 223.0 | 223.0..223.0 | 1 | - | - | 223.0 | - | - (stored whole) | - | - | - | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x259, rel_diff_vs_sklearn=0.000000 | - | main board, one scored run | - | ok (main@a8fd6d926 amd/a0005 2026-10-12; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 139.4 | 139.4..139.4 | 1 | 1.600 | - | 139.4 | 139.4 | 0.00 (cpu-arm) | 1.600 (whole/whole) | - | 6601.1 | - | output_shape=100000x259 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'neg_label': 0, 'pos_label': 1}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), LabelBinarizer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### label-encoder / istella (rows full, shape X 1000000x8; Xq 100000x8; lab 1000000; labq 100000; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/amd_ingest/results.txt (amd/a0005)`, ran on AMD Instinct MI325X (amd, DO) job a0005

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@a8fd6d926) | identical | 2.9 | 2.9..2.9 | 1 | - | - | 2.9 | - | - (stored whole) | - | - | - | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000, rel_diff_vs_sklearn=0.000000 | - | main board, one scored run | - | ok (main@a8fd6d926 amd/a0005 2026-10-12; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 6.8 | 6.8..6.8 | 1 | 0.426 | - | 6.8 | 6.8 | 0.00 (cpu-arm) | 0.426 (whole/whole) | - | 313.1 | - | output_shape=100000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), LabelEncoder (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### label-encoder / taxi (rows full, shape X 1000000x5; Xq 100000x5; lab 1000000; labq 100000; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/amd_ingest/results.txt (amd/a0005)`, ran on AMD Instinct MI325X (amd, DO) job a0005

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@a8fd6d926) | identical | 2.7 | 2.7..2.7 | 1 | - | - | 2.7 | - | - (stored whole) | - | - | - | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000, rel_diff_vs_sklearn=0.000000 | - | main board, one scored run | - | ok (main@a8fd6d926 amd/a0005 2026-10-12; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 20.6 | 20.6..20.6 | 1 | 0.134 | - | 20.6 | 20.6 | 0.00 (cpu-arm) | 0.134 (whole/whole) | - | 300.5 | - | output_shape=100000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), LabelEncoder (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lamb / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/amd_ingest/results.txt (amd/a0006)`, ran on AMD Instinct MI325X (amd, DO) job a0006

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32: - (no completed torch fp32 arm). Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@a8fd6d926) | identical | 13.4 | 13.4..13.4 | 1 | - | - | 13.4 | - | - (stored whole) | - | - | - | - | relative_error_vs_own_host=0.000000 | - | main board, one scored run | - | ok (main@a8fd6d926 amd/a0006 2026-10-12; identity vs the nvidia columns: n/a) |

settings: {'betas': [0.9, 0.999], 'eps': 1e-06, 'lr': 0.001, 'weight_decay': 0.01}. Rows: None. Timed: None.

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lars / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1657)`, ran on AMD Instinct MI325X (amd, DO) job a1657

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@6cd6bed7f) | identical | 1157.1 | 1157.1..1157.1 | 1 | - | - | 1157.1 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.328699, rmse=0.684414 | - | main board, one scored run | - | ok (main@6cd6bed7f amd/a1657 2026-10-11; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 474.5 | 474.5..474.5 | 1 | 2.439 | - | 474.5 | 474.5 | 0.00 (cpu-arm) | 2.439 (whole/whole) | - | 1980.0 | - | finite=True, r2=-1.051e+22, rmse=8.562e+10 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'eps': 2.220446049250313e-16, 'fit_intercept': True, 'n_nonzero_coefs': 500, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lars / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1657)`, ran on AMD Instinct MI325X (amd, DO) job a1657

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@6cd6bed7f) | identical | 3.4 | 3.4..3.4 | 1 | - | - | 3.4 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.908983, rmse=4.805051 | - | main board, one scored run | - | ok (main@6cd6bed7f amd/a1657 2026-10-11; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 32.4 | 32.4..32.4 | 1 | 0.104 | - | 32.4 | 32.4 | 0.00 (cpu-arm) | 0.104 (whole/whole) | - | 303.2 | - | finite=True, r2=0.908983, rmse=4.805057 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'eps': 2.220446049250313e-16, 'fit_intercept': True, 'n_nonzero_coefs': 500, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lasso-cv / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1645)`, ran on AMD Instinct MI325X (amd, DO) job a1645

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@2afa942e3) | identical | 836.6 | 836.6..836.6 | 1 | - | - | 836.6 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.325506, rmse=0.686040 | - | main board, one scored run | - | ok (main@2afa942e3 amd/a1645 2026-10-11; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 42194.4 | 42194.4..42194.4 | 1 | 0.020 | - | 42194.4 | 42194.4 | 0.00 (cpu-arm) | 0.020 (whole/whole) | - | 8912.8 | - | finite=True, r2=0.325507, rmse=0.686040 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alphas': [0.001, 0.01, 0.1, 1.0], 'cv': 5, 'eps': 0.001, 'fit_intercept': True, 'max_iter': 1000, 'positive': False, 'random_state': 7, 'selection': 'cyclic', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lasso-cv / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1645)`, ran on AMD Instinct MI325X (amd, DO) job a1645

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@2afa942e3) | identical | 198.7 | 198.7..198.7 | 1 | - | - | 198.7 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.909038, rmse=4.803593 | - | main board, one scored run | - | ok (main@2afa942e3 amd/a1645 2026-10-11; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 277.2 | 277.2..277.2 | 1 | 0.717 | - | 277.2 | 277.2 | 0.00 (cpu-arm) | 0.717 (whole/whole) | - | 680.5 | - | finite=True, r2=0.909038, rmse=4.803593 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alphas': [0.001, 0.01, 0.1, 1.0], 'cv': 5, 'eps': 0.001, 'fit_intercept': True, 'max_iter': 1000, 'positive': False, 'random_state': 7, 'selection': 'cyclic', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lasso-lars / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1657)`, ran on AMD Instinct MI325X (amd, DO) job a1657

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@6cd6bed7f) | identical | 75.3 | 75.3..75.3 | 1 | - | - | 75.3 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.310834, rmse=0.693461 | - | main board, one scored run | - | ok (main@6cd6bed7f amd/a1657 2026-10-11; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 461.2 | 461.2..461.2 | 1 | 0.163 | - | 461.2 | 461.2 | 0.00 (cpu-arm) | 0.163 (whole/whole) | - | 1979.4 | - | finite=True, r2=0.310591, rmse=0.693584 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 0.01, 'eps': 2.220446049250313e-16, 'fit_intercept': True, 'max_iter': 500, 'positive': False, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lasso-lars / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1657)`, ran on AMD Instinct MI325X (amd, DO) job a1657

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@6cd6bed7f) | identical | 3.4 | 3.4..3.4 | 1 | - | - | 3.4 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.908998, rmse=4.804663 | - | main board, one scored run | - | ok (main@6cd6bed7f amd/a1657 2026-10-11; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 32.4 | 32.4..32.4 | 1 | 0.104 | - | 32.4 | 32.4 | 0.00 (cpu-arm) | 0.104 (whole/whole) | - | 303.3 | - | finite=True, r2=0.908998, rmse=4.804669 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 0.01, 'eps': 2.220446049250313e-16, 'fit_intercept': True, 'max_iter': 500, 'positive': False, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### layernorm / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/amd/campaign/measured/algos.layernorm.synthetic.rows=full/board.json (amd/paired-5939cde02360535239ede45137d834b8f9785d819a429934a6922df8c93ed92f)`, ran on mojolearn-steward-do-amd

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 0.8 ms) = 0.841; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 0.8 ms) = 0.810. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | 0.8.44 | identical | 0.6 | 0.6..0.6 | 1 | - | - | - | 0.6 | - (stored kernel) | - | - | 2518.4 | 235846.9 | max_rel_diff_vs_torch_eager_fp32=0.011086, rel_fro_vs_torch_eager_fp32=6.669e-08 | - | main board, one scored run | wheel | ok (main@79fcd24a5 amd/paired-5939cde02360535239ede45137d834b8f9785d819a429934a6922df8c93ed92f 2026-10-14; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 0.8 | 0.8..0.8 | 1 | 0.810 | - | - | 0.8 | - (stored kernel) | 0.810 (kernel/kernel) | - | 1890.1 | 320.1 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.4 | 1.4..1.4 | 1 | 0.459 | - | - | 1.4 | - (stored kernel) | 0.459 (kernel/kernel) | - | 2336.8 | 392.1 | max_rel_diff_vs_torch_eager_fp32=0.007530, rel_fro_vs_torch_eager_fp32=5.326e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 0.8 | 0.8..0.8 | 1 | 0.841 | - | - | 0.8 | - (stored kernel) | 0.841 (kernel/kernel) | - | 1890.4 | 320.1 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.4 | 1.4..1.4 | 1 | 0.458 | - | - | 1.4 | - (stored kernel) | 0.458 (kernel/kernel) | - | 2261.4 | 392.1 | max_rel_diff_vs_torch_eager_fp32=0.007530, rel_fro_vs_torch_eager_fp32=5.326e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/108584/vram_* bytes for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'bias': True, 'elementwise_affine': True, 'eps': 1e-05, 'normalized_shape': 1024}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| bias | true | true | true | true | true |
| elementwise_affine | true | true | true | true | true |
| eps | 1e-05 | 1e-05 | 1e-05 | 1e-05 | 1e-05 |
| input_arrays | [{"dtype": "float32", "name": "x", "sha256": "128bb6c8835123e5f00ffe3700a26223ba8b37f7280afaec092571b2861be93c", "shape": [16384, 1024]}, {"dtype": "float32", "name": "bias", "sha256": "ad7facb2586fc6e966c004d7d1d16b024f5805ff7cb47c7a85dabd8b48892ca7", "shape": [1024]}, {"dtype": "float32", "name": "weight", "sha256": "e9bac255f4adc7cb4ada9298e193a5ff66b434d15afabd458505325f29c398c7", "shape": [1024]}] | [{"dtype": "float32", "name": "x", "sha256": "128bb6c8835123e5f00ffe3700a26223ba8b37f7280afaec092571b2861be93c", "shape": [16384, 1024]}, {"dtype": "float32", "name": "bias", "sha256": "ad7facb2586fc6e966c004d7d1d16b024f5805ff7cb47c7a85dabd8b48892ca7", "shape": [1024]}, {"dtype": "float32", "name": "weight", "sha256": "e9bac255f4adc7cb4ada9298e193a5ff66b434d15afabd458505325f29c398c7", "shape": [1024]}] | [{"dtype": "float32", "name": "x", "sha256": "128bb6c8835123e5f00ffe3700a26223ba8b37f7280afaec092571b2861be93c", "shape": [16384, 1024]}, {"dtype": "float32", "name": "bias", "sha256": "ad7facb2586fc6e966c004d7d1d16b024f5805ff7cb47c7a85dabd8b48892ca7", "shape": [1024]}, {"dtype": "float32", "name": "weight", "sha256": "e9bac255f4adc7cb4ada9298e193a5ff66b434d15afabd458505325f29c398c7", "shape": [1024]}] | [{"dtype": "float32", "name": "x", "sha256": "128bb6c8835123e5f00ffe3700a26223ba8b37f7280afaec092571b2861be93c", "shape": [16384, 1024]}, {"dtype": "float32", "name": "bias", "sha256": "ad7facb2586fc6e966c004d7d1d16b024f5805ff7cb47c7a85dabd8b48892ca7", "shape": [1024]}, {"dtype": "float32", "name": "weight", "sha256": "e9bac255f4adc7cb4ada9298e193a5ff66b434d15afabd458505325f29c398c7", "shape": [1024]}] | [{"dtype": "float32", "name": "x", "sha256": "128bb6c8835123e5f00ffe3700a26223ba8b37f7280afaec092571b2861be93c", "shape": [16384, 1024]}, {"dtype": "float32", "name": "bias", "sha256": "ad7facb2586fc6e966c004d7d1d16b024f5805ff7cb47c7a85dabd8b48892ca7", "shape": [1024]}, {"dtype": "float32", "name": "weight", "sha256": "e9bac255f4adc7cb4ada9298e193a5ff66b434d15afabd458505325f29c398c7", "shape": [1024]}] |
| normalized_shape | 1024 | 1024 | 1024 | 1024 | 1024 |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | hardware | version | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | AMD Instinct MI325X (amd, DO) | 0.8.44 | Xq | - | 0.1 | 0.1..0.1 | 1 | - | - | - | 0.1 | - (stored kernel) | - | - | max_rel_diff_vs_torch_eager_fp32=0.011086, rel_fro_vs_torch_eager_fp32=6.669e-08 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | Xq | - | 0.1 | 0.1..0.1 | 1 | 1.826 | - | - | 0.1 | - (stored kernel) | 1.826 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | Xq | - | 0.1 | 0.1..0.1 | 1 | 1.149 | - | - | 0.1 | - (stored kernel) | 1.149 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.007530, rel_fro_vs_torch_eager_fp32=5.326e-08 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | Xq | - | 0.1 | 0.1..0.1 | 1 | 1.806 | - | - | 0.1 | - (stored kernel) | 1.806 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | Xq | - | 0.1 | 0.1..0.1 | 1 | 1.081 | - | - | 0.1 | - (stored kernel) | 1.081 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.007530, rel_fro_vs_torch_eager_fp32=5.326e-08 | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### lda-clf / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1066)`, ran on AMD Instinct MI325X (amd, DO) job a1066

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@8ed94710a) | identical | 315.8 | 315.8..315.8 | 1 | - | - | 315.8 | - | - (stored whole) | - | - | - | - | accuracy=0.912830, logloss=0.235875 | - | main board, one scored run | - | ok (main@8ed94710a amd/a1066 2026-10-09; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 3925.4 | 3925.4..3925.4 | 1 | 0.080 | - | 3925.4 | 3925.4 | 0.00 (cpu-arm) | 0.080 (whole/whole) | - | 5444.6 | - | accuracy=0.899520, logloss=0.439049 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'solver': 'svd', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lda-clf / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1066)`, ran on AMD Instinct MI325X (amd, DO) job a1066

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@8ed94710a) | identical | 17.4 | 17.4..17.4 | 1 | - | - | 17.4 | - | - (stored whole) | - | - | - | - | accuracy=0.762530, logloss=0.539763 | - | main board, one scored run | - | ok (main@8ed94710a amd/a1066 2026-10-09; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 230.9 | 230.9..230.9 | 1 | 0.075 | - | 230.9 | 230.9 | 0.00 (cpu-arm) | 0.075 (whole/whole) | - | 513.8 | - | accuracy=0.762530, logloss=0.539767 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'solver': 'svd', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lion / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/amd_ingest/results.txt (amd/a0006)`, ran on AMD Instinct MI325X (amd, DO) job a0006

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32: - (no completed torch fp32 arm). Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@a8fd6d926) | identical | 3.5 | 3.5..3.5 | 1 | - | - | 3.5 | - | - (stored whole) | - | - | - | - | relative_error_vs_own_host=0.000000 | - | main board, one scored run | - | ok (main@a8fd6d926 amd/a0006 2026-10-12; identity vs the nvidia columns: n/a) |

settings: {'betas': [0.9, 0.99], 'lr': 0.001, 'weight_decay': 0.0}. Rows: None. Timed: None.

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### louvain / istella (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0870)`, ran on AMD Instinct MI325X (amd, DO) job a0870

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@4df610f3b) | identical | 359.9 | 359.9..359.9 | 1 | - | - | 359.9 | - | - (stored whole) | - | - | - | - | modularity=0.911187, n_communities=40 | - | main board, one scored run | - | ok (main@4df610f3b amd/a0870 2026-10-08; identity vs nvidia-l40s: MATCH) |

settings: {'resolution': 1.0, 'threshold': 1e-07, 'seed': 7}. Rows: None. Timed: None.

mismatch: networkx louvain_communities(seed=7) and cuGraph louvain (max_level=100) are order-dependent; ours pins a colour-batched move order (a fixed hash colouring, ties to the smallest community id)

mismatch: ours takes the CSR graph as an (indptr, indices, n) tuple and uploads it inside the clock (the 2026-10-08 CSR entries; before, a dense n x n adjacency built before the clock); networkx and cuGraph take the graph itself

mismatch: opponents withheld: the lane settings at this tree's HEAD differ from the settings release-board-resume-r2 recorded for its opponent race (an opponent job must score them again)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### louvain / taxi (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0870)`, ran on AMD Instinct MI325X (amd, DO) job a0870

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@4df610f3b) | identical | 71.2 | 71.2..71.2 | 1 | - | - | 71.2 | - | - (stored whole) | - | - | - | - | modularity=0.941953, n_communities=58 | - | main board, one scored run | - | ok (main@4df610f3b amd/a0870 2026-10-08; identity vs nvidia-l40s: MATCH) |

settings: {'resolution': 1.0, 'threshold': 1e-07, 'seed': 7}. Rows: None. Timed: None.

mismatch: networkx louvain_communities(seed=7) and cuGraph louvain (max_level=100) are order-dependent; ours pins a colour-batched move order (a fixed hash colouring, ties to the smallest community id)

mismatch: ours takes the CSR graph as an (indptr, indices, n) tuple and uploads it inside the clock (the 2026-10-08 CSR entries; before, a dense n x n adjacency built before the clock); networkx and cuGraph take the graph itself

mismatch: opponents withheld: the lane settings at this tree's HEAD differ from the settings release-board-resume-r2 recorded for its opponent race (an opponent job must score them again)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lstm-clf / synthetic (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1211)`, ran on AMD Instinct MI325X (amd, DO) job a1211

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 854.9 ms) = 0.739; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 894.9 ms) = 0.706. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 631.6 | 631.6..631.6 | 1 | - | - | 631.6 | - | - (stored whole) | - | - | - | - | accuracy=0.967068, logloss=0.079720 | - | main board, one scored run | - | ok (main@1665b5626 amd/a1211 2026-10-10; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 903.2 | 903.2..903.2 | 1 | 0.699 | - | - | 903.2 | - (stored kernel) | 0.699 (MIXED ours whole / arm kernel) | - | 3208.6 | 350.8 | accuracy=0.968696 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 894.9 | 894.9..894.9 | 1 | 0.706 | - | - | 894.9 | - (stored kernel) | 0.706 (MIXED ours whole / arm kernel) | - | 3257.9 | 350.8 | accuracy=0.968696 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 854.9 | 854.9..854.9 | 1 | 0.739 | - | - | 854.9 | - (stored kernel) | 0.739 (MIXED ours whole / arm kernel) | - | 4967.1 | 182.8 | accuracy=0.968913 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1044.0 | 1044.0..1044.0 | 1 | 0.605 | - | - | 1044.0 | - (stored kernel) | 0.605 (MIXED ours whole / arm kernel) | - | 5000.0 | 182.8 | accuracy=0.968913 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lstm-clf / taxi-hourly (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1211)`, ran on AMD Instinct MI325X (amd, DO) job a1211

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 977.6 ms) = 0.645; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 815.7 ms) = 0.773. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 630.7 | 630.7..630.7 | 1 | - | - | 630.7 | - | - (stored whole) | - | - | - | - | accuracy=0.870443, logloss=0.297146 | - | main board, one scored run | - | ok (main@1665b5626 amd/a1211 2026-10-10; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 861.1 | 861.1..861.1 | 1 | 0.732 | - | - | 861.1 | - (stored kernel) | 0.732 (MIXED ours whole / arm kernel) | - | 3603.8 | 350.8 | accuracy=0.868218 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 815.7 | 815.7..815.7 | 1 | 0.773 | - | - | 815.7 | - (stored kernel) | 0.773 (MIXED ours whole / arm kernel) | - | 3258.3 | 350.8 | accuracy=0.868218 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1039.6 | 1039.6..1039.6 | 1 | 0.607 | - | - | 1039.6 | - (stored kernel) | 0.607 (MIXED ours whole / arm kernel) | - | 5292.2 | 182.8 | accuracy=0.868056 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 977.6 | 977.6..977.6 | 1 | 0.645 | - | - | 977.6 | - (stored kernel) | 0.645 (MIXED ours whole / arm kernel) | - | 5008.3 | 182.8 | accuracy=0.868056 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lstm-reg / synthetic (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1211)`, ran on AMD Instinct MI325X (amd, DO) job a1211

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 1053.4 ms) = 0.597; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 923.1 ms) = 0.681. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 628.6 | 628.6..628.6 | 1 | - | - | 628.6 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.979398, rmse=0.166295 | - | main board, one scored run | - | ok (main@1665b5626 amd/a1211 2026-10-10; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 923.1 | 923.1..923.1 | 1 | 0.681 | - | - | 923.1 | - (stored kernel) | 0.681 (MIXED ours whole / arm kernel) | - | 3036.9 | 350.5 | finite=True, r2=0.981013, rmse=0.159641 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 973.6 | 973.6..973.6 | 1 | 0.646 | - | - | 973.6 | - (stored kernel) | 0.646 (MIXED ours whole / arm kernel) | - | 3082.7 | 350.5 | finite=True, r2=0.981013, rmse=0.159641 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1053.4 | 1053.4..1053.4 | 1 | 0.597 | - | - | 1053.4 | - (stored kernel) | 0.597 (MIXED ours whole / arm kernel) | - | 3660.4 | 182.5 | finite=True, r2=0.981004, rmse=0.159679 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1065.8 | 1065.8..1065.8 | 1 | 0.590 | - | - | 1065.8 | - (stored kernel) | 0.590 (MIXED ours whole / arm kernel) | - | 3706.1 | 182.5 | finite=True, r2=0.981004, rmse=0.159679 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lstm-reg / taxi-hourly (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1211)`, ran on AMD Instinct MI325X (amd, DO) job a1211

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 1017.5 ms) = 0.617; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 920.3 ms) = 0.682. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 627.9 | 627.9..627.9 | 1 | - | - | 627.9 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.754306, rmse=0.537563 | - | main board, one scored run | - | ok (main@1665b5626 amd/a1211 2026-10-10; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 920.3 | 920.3..920.3 | 1 | 0.682 | - | - | 920.3 | - (stored kernel) | 0.682 (MIXED ours whole / arm kernel) | - | 3043.3 | 350.5 | finite=True, r2=0.751679, rmse=0.540429 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 976.5 | 976.5..976.5 | 1 | 0.643 | - | - | 976.5 | - (stored kernel) | 0.643 (MIXED ours whole / arm kernel) | - | 3084.4 | 350.5 | finite=True, r2=0.751679, rmse=0.540429 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1017.5 | 1017.5..1017.5 | 1 | 0.617 | - | - | 1017.5 | - (stored kernel) | 0.617 (MIXED ours whole / arm kernel) | - | 3667.0 | 182.5 | finite=True, r2=0.751591, rmse=0.540526 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1057.7 | 1057.7..1057.7 | 1 | 0.594 | - | - | 1057.7 | - (stored kernel) | 0.594 (MIXED ours whole / arm kernel) | - | 3707.8 | 182.5 | finite=True, r2=0.751591, rmse=0.540526 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lstsq / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/full-gpu-repair-20261014/amd/repair2/measured/algos.lstsq.istella.rows=full/board.json (amd/paired-67859f8b54a103e967715d76d15eaa572d9e01fcb17cab37e9d6b88d78d9091b)`, ran on mojolearn-steward-do-amd

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | 0.8.44 | identical | 321.6 | 321.6..321.6 | 1 | - | - | 321.6 | 305.9 | 15.76 (upload_ms_separate) | - | - | 3495.3 | 235661.1 | relative_residual=0.849956 | - | main board, one scored run | wheel | ok (main@d425c1417 amd/paired-67859f8b54a103e967715d76d15eaa572d9e01fcb17cab37e9d6b88d78d9091b 2026-10-14; identity vs the nvidia columns: n/a) |
| torch-gpu | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "RuntimeError(\"REFUSED: torch.linalg.lstsq (driver 'gels' on CUDA/ROCm: QR without pivoting, full rank assumed) returned a non-finite solution on this rank-deficient X; not a result\")", "e) (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/94228/vram_* bytes for this pid at the round's end (not a peak)

memory, torch-gpu: host not sampled; GPU not sampled

settings: {}. Rows: None. Timed: None.

mismatch: torch-gpu: torch.linalg.lstsq on CUDA/ROCm has only driver='gels' (QR without pivoting, full rank assumed); ours, scipy (gelsd) and CuPy are rank-revealing; a non-finite torch solution (rank-deficient X, e.g. Istella) refuses the arm

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-gpu |
|---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) |
| input_arrays | [{"dtype": "float32", "sha256": "14c1e3963ef015c9b24cea93d1380679b59623b22edf61cb634001e62b4142a8", "shape": [1000000, 220]}, {"dtype": "float32", "sha256": "909c00fe78632b5e033bba49666d5b2b8ebaac4615466126e525f3cf47c4f3b5", "shape": [1000000]}] | [{"dtype": "float32", "sha256": "14c1e3963ef015c9b24cea93d1380679b59623b22edf61cb634001e62b4142a8", "shape": [1000000, 220]}, {"dtype": "float32", "sha256": "909c00fe78632b5e033bba49666d5b2b8ebaac4615466126e525f3cf47c4f3b5", "shape": [1000000]}] |
| seed | 7 | 7 |

### lstsq / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/full-gpu-repair-20261014/amd/repair2/measured/algos.lstsq.taxi.rows=full/board.json (amd/paired-9d7c466693bba67fc8323d918c76429fdd964cf4b9fc939cf4ab661be4496343)`, ran on mojolearn-steward-do-amd

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | 0.8.44 | identical | 6.1 | 6.1..6.1 | 1 | - | - | 6.1 | 5.3 | 0.89 (upload_ms_separate) | - | - | 2423.0 | 236201.6 | relative_residual=0.756366 | - | main board, one scored run | wheel | ok (main@d425c1417 amd/paired-9d7c466693bba67fc8323d918c76429fdd964cf4b9fc939cf4ab661be4496343 2026-10-14; identity vs the nvidia columns: n/a) |
| torch-gpu | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 26.5 | 26.5..26.5 | 1 | 0.231 | - | - | 26.5 | - (stored kernel) | 0.198 (kernel/kernel) | - | 1766.5 | 223.5 | relative_residual=0.756366 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/93751/vram_* bytes for this pid at the round's end (not a peak)

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {}. Rows: None. Timed: None.

mismatch: torch-gpu: torch.linalg.lstsq on CUDA/ROCm has only driver='gels' (QR without pivoting, full rank assumed); ours, scipy (gelsd) and CuPy are rank-revealing; a non-finite torch solution (rank-deficient X, e.g. Istella) refuses the arm

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-gpu |
|---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) |
| input_arrays | [{"dtype": "float32", "sha256": "33fdb983b45d51732ce911c8887c0c9f9cf7a7416ae67ebfe983fa0b3a4c8b77", "shape": [1000000, 11]}, {"dtype": "float32", "sha256": "5cda40092ba17e30737e2496638e974415a8f261f8325676b16da6739af5e09a", "shape": [1000000]}] | [{"dtype": "float32", "sha256": "33fdb983b45d51732ce911c8887c0c9f9cf7a7416ae67ebfe983fa0b3a4c8b77", "shape": [1000000, 11]}, {"dtype": "float32", "sha256": "5cda40092ba17e30737e2496638e974415a8f261f8325676b16da6739af5e09a", "shape": [1000000]}] |
| seed | 7 | 7 |

### lu-factor / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/full-gpu-repair-20261014/recovered/algos.lu-factor.synthetic.rows=full/board.json (amd/paired-9ff9fc147260942d5ae2bbbc4d112ce44679fdd2705896d478e795591e59d340)`, ran on mojolearn-steward-do-amd

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | 0.8.44 | identical | 317.7 | 317.7..317.7 | 1 | - | - | 317.7 | 312.8 | 4.84 (upload_ms_separate) | - | - | 3096.2 | 236186.7 | relative_residual=3.256e-06 | - | main board, one scored run | wheel | ok (main@d425c1417 amd/paired-9ff9fc147260942d5ae2bbbc4d112ce44679fdd2705896d478e795591e59d340 2026-10-14; identity vs the nvidia columns: n/a) |
| torch-gpu | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 83.6 | 83.6..83.6 | 1 | 3.799 | - | - | 83.6 | - (stored kernel) | 3.741 (kernel/kernel) | - | 2145.8 | 710.0 | relative_residual=4.003e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/95688/vram_* bytes for this pid at the round's end (not a peak)

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-gpu |
|---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) |
| input_arrays | [{"dtype": "float32", "sha256": "e3500eebc0f84bfe008ee69884a10df5fab41e5bec2a57ae24d76cb8dbddeb20", "shape": [8192, 8192]}, {"dtype": "float32", "sha256": "56589d513ac0c940e300abfd9e69b92d6fe1b6ba3f08b861624290d33b5b50ff", "shape": [8192, 64]}] | [{"dtype": "float32", "sha256": "e3500eebc0f84bfe008ee69884a10df5fab41e5bec2a57ae24d76cb8dbddeb20", "shape": [8192, 8192]}, {"dtype": "float32", "sha256": "56589d513ac0c940e300abfd9e69b92d6fe1b6ba3f08b861624290d33b5b50ff", "shape": [8192, 64]}] |
| seed | 7 | 7 |

### lu-solve / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/full-gpu-repair-20261014/recovered/algos.lu-solve.synthetic.rows=full/board.json (amd/paired-532f6d268c817067b137f1555deaadf41035d6ff4c8195940a99731ee3a707d3)`, ran on mojolearn-steward-do-amd

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | 0.8.44 | identical | 310.7 | 310.7..310.7 | 1 | - | - | 310.7 | 305.9 | 4.84 (upload_ms_separate) | - | - | 3166.0 | 236009.2 | relative_residual=3.256e-06 | - | main board, one scored run | wheel | ok (main@d425c1417 amd/paired-532f6d268c817067b137f1555deaadf41035d6ff4c8195940a99731ee3a707d3 2026-10-14; identity vs the nvidia columns: n/a) |
| torch-gpu | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 82.9 | 82.9..82.9 | 1 | 3.746 | - | - | 82.9 | - (stored kernel) | 3.688 (kernel/kernel) | - | 2144.2 | 712.0 | relative_residual=4.003e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/93288/vram_* bytes for this pid at the round's end (not a peak)

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-gpu |
|---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) |
| input_arrays | [{"dtype": "float32", "sha256": "e3500eebc0f84bfe008ee69884a10df5fab41e5bec2a57ae24d76cb8dbddeb20", "shape": [8192, 8192]}, {"dtype": "float32", "sha256": "56589d513ac0c940e300abfd9e69b92d6fe1b6ba3f08b861624290d33b5b50ff", "shape": [8192, 64]}] | [{"dtype": "float32", "sha256": "e3500eebc0f84bfe008ee69884a10df5fab41e5bec2a57ae24d76cb8dbddeb20", "shape": [8192, 8192]}, {"dtype": "float32", "sha256": "56589d513ac0c940e300abfd9e69b92d6fe1b6ba3f08b861624290d33b5b50ff", "shape": [8192, 64]}] |
| seed | 7 | 7 |

### maxabs-scaler / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/amd_ingest/results.txt (amd/a0005)`, ran on AMD Instinct MI325X (amd, DO) job a0005

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@a8fd6d926) | identical | 17.9 | 17.9..17.9 | 1 | - | - | 17.9 | - | - (stored whole) | - | - | - | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x220, rel_diff_vs_sklearn=0.000000 | - | main board, one scored run | - | ok (main@a8fd6d926 amd/a0005 2026-10-12; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 123.0 | 123.0..123.0 | 1 | 0.146 | - | 123.0 | 123.0 | 0.00 (cpu-arm) | 0.146 (whole/whole) | - | 2222.0 | - | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), MaxAbsScaler (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### maxabs-scaler / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/amd_ingest/results.txt (amd/a0005)`, ran on AMD Instinct MI325X (amd, DO) job a0005

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@a8fd6d926) | identical | 2.6 | 2.6..2.6 | 1 | - | - | 2.6 | - | - (stored whole) | - | - | - | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x11, rel_diff_vs_sklearn=0.000000 | - | main board, one scored run | - | ok (main@a8fd6d926 amd/a0005 2026-10-12; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 27.7 | 27.7..27.7 | 1 | 0.093 | - | 27.7 | 27.7 | 0.00 (cpu-arm) | 0.093 (whole/whole) | - | 308.5 | - | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), MaxAbsScaler (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### maxpool1d / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/amd/campaign/measured/algos.maxpool1d.synthetic.rows=full/board.json (amd/paired-9b09a78002421099ee1e041691a221ccfd1ee58caec079a17f2c9dbdc800cfe0)`, ran on mojolearn-steward-do-amd

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 0.9 ms) = 0.368; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 0.8 ms) = 0.431. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | 0.8.44 | identical | 0.3 | 0.3..0.3 | 1 | - | - | - | 0.3 | - (stored kernel) | - | - | 2197.0 | 235669.5 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | main board, one scored run | wheel | ok (main@79fcd24a5 amd/paired-9b09a78002421099ee1e041691a221ccfd1ee58caec079a17f2c9dbdc800cfe0 2026-10-14; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 0.8 | 0.8..0.8 | 1 | 0.431 | - | - | 0.8 | - (stored kernel) | 0.431 (kernel/kernel) | - | 1800.5 | 288.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.5 | 1.5..1.5 | 1 | 0.238 | - | - | 1.5 | - (stored kernel) | 0.238 (kernel/kernel) | - | 2322.8 | 232.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 0.9 | 0.9..0.9 | 1 | 0.368 | - | - | 0.9 | - (stored kernel) | 0.368 (kernel/kernel) | - | 1796.8 | 288.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.1 | 1.1..1.1 | 1 | 0.330 | - | - | 1.1 | - (stored kernel) | 0.330 (kernel/kernel) | - | 2161.4 | 232.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/106310/vram_* bytes for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'ceil_mode': False, 'dilation': 1, 'kernel_size': 2, 'padding': 0, 'stride': 2}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| ceil_mode | false | false | false | false | false |
| dilation | 1 | 1 | 1 | 1 | 1 |
| input_arrays | [{"dtype": "float32", "name": "x", "sha256": "128bb6c8835123e5f00ffe3700a26223ba8b37f7280afaec092571b2861be93c", "shape": [64, 64, 4096]}] | [{"dtype": "float32", "name": "x", "sha256": "128bb6c8835123e5f00ffe3700a26223ba8b37f7280afaec092571b2861be93c", "shape": [64, 64, 4096]}] | [{"dtype": "float32", "name": "x", "sha256": "128bb6c8835123e5f00ffe3700a26223ba8b37f7280afaec092571b2861be93c", "shape": [64, 64, 4096]}] | [{"dtype": "float32", "name": "x", "sha256": "128bb6c8835123e5f00ffe3700a26223ba8b37f7280afaec092571b2861be93c", "shape": [64, 64, 4096]}] | [{"dtype": "float32", "name": "x", "sha256": "128bb6c8835123e5f00ffe3700a26223ba8b37f7280afaec092571b2861be93c", "shape": [64, 64, 4096]}] |
| kernel_size | 2 | 2 | 2 | 2 | 2 |
| padding | 0 | 0 | 0 | 0 | 0 |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 |
| stride | 2 | 2 | 2 | 2 | 2 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | hardware | version | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | AMD Instinct MI325X (amd, DO) | 0.8.44 | Xq | - | 0.1 | 0.1..0.1 | 1 | - | - | - | 0.1 | - (stored kernel) | - | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | Xq | - | 0.1 | 0.1..0.1 | 1 | 0.991 | - | - | 0.1 | - (stored kernel) | 0.991 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | Xq | - | 0.1 | 0.1..0.1 | 1 | 0.700 | - | - | 0.1 | - (stored kernel) | 0.700 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | Xq | - | 0.1 | 0.1..0.1 | 1 | 0.926 | - | - | 0.1 | - (stored kernel) | 0.926 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | Xq | - | 0.1 | 0.1..0.1 | 1 | 0.735 | - | - | 0.1 | - (stored kernel) | 0.735 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### maxpool2d / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/amd/campaign/measured/algos.maxpool2d.synthetic.rows=full/board.json (amd/paired-6e6eecc8c845fc493a7cd56ba91c16e19946b221260de30760196dfea866dadc)`, ran on mojolearn-steward-do-amd

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 1.1 ms) = 0.490; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 0.9 ms) = 0.613. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | 0.8.44 | identical | 0.5 | 0.5..0.5 | 1 | - | - | - | 0.5 | - (stored kernel) | - | - | 2396.7 | 235669.5 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | main board, one scored run | wheel | ok (main@79fcd24a5 amd/paired-6e6eecc8c845fc493a7cd56ba91c16e19946b221260de30760196dfea866dadc 2026-10-14; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 0.9 | 0.9..0.9 | 1 | 0.613 | - | - | 0.9 | - (stored kernel) | 0.613 (kernel/kernel) | - | 2110.6 | 639.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.2 | 1.2..1.2 | 1 | 0.448 | - | - | 1.2 | - (stored kernel) | 0.448 (kernel/kernel) | - | 2836.1 | 553.5 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.8 | 1.8..1.8 | 1 | 0.298 | - | - | 1.8 | - (stored kernel) | 0.298 (kernel/kernel) | - | 2111.0 | 639.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.1 | 1.1..1.1 | 1 | 0.490 | - | - | 1.1 | - (stored kernel) | 0.490 (kernel/kernel) | - | 2485.1 | 553.5 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/107450/vram_* bytes for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'ceil_mode': False, 'dilation': 1, 'kernel_size': 2, 'padding': 0, 'stride': 2}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| ceil_mode | false | false | false | false | false |
| dilation | 1 | 1 | 1 | 1 | 1 |
| input_arrays | [{"dtype": "float32", "name": "x", "sha256": "2df96f3879e0a6060dab5f7b7561b7291c5a106d1a8d4e116fc5e44e3023f819", "shape": [64, 64, 112, 112]}] | [{"dtype": "float32", "name": "x", "sha256": "2df96f3879e0a6060dab5f7b7561b7291c5a106d1a8d4e116fc5e44e3023f819", "shape": [64, 64, 112, 112]}] | [{"dtype": "float32", "name": "x", "sha256": "2df96f3879e0a6060dab5f7b7561b7291c5a106d1a8d4e116fc5e44e3023f819", "shape": [64, 64, 112, 112]}] | [{"dtype": "float32", "name": "x", "sha256": "2df96f3879e0a6060dab5f7b7561b7291c5a106d1a8d4e116fc5e44e3023f819", "shape": [64, 64, 112, 112]}] | [{"dtype": "float32", "name": "x", "sha256": "2df96f3879e0a6060dab5f7b7561b7291c5a106d1a8d4e116fc5e44e3023f819", "shape": [64, 64, 112, 112]}] |
| kernel_size | 2 | 2 | 2 | 2 | 2 |
| padding | 0 | 0 | 0 | 0 | 0 |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 |
| stride | 2 | 2 | 2 | 2 | 2 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | hardware | version | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | AMD Instinct MI325X (amd, DO) | 0.8.44 | Xq | - | 0.2 | 0.2..0.2 | 1 | - | - | - | 0.2 | - (stored kernel) | - | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | Xq | - | 0.1 | 0.1..0.1 | 1 | 1.211 | - | - | 0.1 | - (stored kernel) | 1.211 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | Xq | - | 0.2 | 0.2..0.2 | 1 | 0.986 | - | - | 0.2 | - (stored kernel) | 0.986 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | Xq | - | 0.1 | 0.1..0.1 | 1 | 1.177 | - | - | 0.1 | - (stored kernel) | 1.177 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | Xq | - | 0.2 | 0.2..0.2 | 1 | 0.993 | - | - | 0.2 | - (stored kernel) | 0.993 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### min-cov-det / istella (rows full, shape X 100000x220; Xq 100000x220; y 100000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/r43/ingest/amd-results.txt (amd/a1589)`, ran on AMD Instinct MI325X (amd, DO) job a1589

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.42 (source build, main@a6b5f76cb) | identical | 243389.1 | 243389.1..243389.1 | 1 | - | - | 243389.1 | - | - (stored whole) | - | - | - | - | n_features=220, rel_diff_vs_sklearn=0.239361 | - | main board, one scored run | - | ok (main@a6b5f76cb amd/a1589 2026-10-13; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(timeout: null) (copied from release-board-resume-r2; measured this run) |

settings: {'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### min-cov-det / taxi (rows full, shape X 100000x11; Xq 100000x11; y 100000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/r43/ingest/amd-results.txt (amd/a1589)`, ran on AMD Instinct MI325X (amd, DO) job a1589

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.42 (source build, main@a6b5f76cb) | identical | 9851.8 | 9851.8..9851.8 | 1 | - | - | 9851.8 | - | - (stored whole) | - | - | - | - | n_features=11, rel_diff_vs_sklearn=0.238565 | - | main board, one scored run | - | ok (main@a6b5f76cb amd/a1589 2026-10-13; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 1251.0 | 1251.0..1251.0 | 1 | 7.875 | - | 1251.0 | 1251.0 | 0.00 (cpu-arm) | 7.875 (whole/whole) | - | 366.5 | - | n_features=11 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### minibatch-kmeans / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1688)`, ran on AMD Instinct MI325X (amd, DO) job a1688

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 344.1 | 344.1..344.1 | 1 | - | - | 344.1 | - | - (stored whole) | - | - | - | - | n_clusters=8, silhouette=0.116696 | - | main board, one scored run | - | ok (main@a1c96a5ba amd/a1688 2026-10-11; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 200.4 | 200.4..200.4 | 1 | 1.717 | - | 200.4 | 200.4 | 0.00 (cpu-arm) | 1.717 (whole/whole) | - | 1178.2 | - | ari_vs_ours=0.622446, n_clusters=8, silhouette=0.111849 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'batch_size': 4096, 'init': 'k-means++', 'max_iter': 100, 'max_no_improvement': 10, 'n_clusters': 8, 'n_init': 1, 'random_state': 7, 'reassignment_ratio': 0.01, 'tol': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### minibatch-kmeans / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1688)`, ran on AMD Instinct MI325X (amd, DO) job a1688

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 48.2 | 48.2..48.2 | 1 | - | - | 48.2 | - | - (stored whole) | - | - | - | - | n_clusters=8, silhouette=0.138060 | - | main board, one scored run | - | ok (main@a1c96a5ba amd/a1688 2026-10-11; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 26.5 | 26.5..26.5 | 1 | 1.815 | - | 26.5 | 26.5 | 0.00 (cpu-arm) | 1.815 (whole/whole) | - | 280.6 | - | ari_vs_ours=0.525151, n_clusters=8, silhouette=0.165473 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'batch_size': 4096, 'init': 'k-means++', 'max_iter': 100, 'max_no_improvement': 10, 'n_clusters': 8, 'n_init': 1, 'random_state': 7, 'reassignment_ratio': 0.01, 'tol': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### minmax-scaler / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/amd_ingest/results.txt (amd/a0010)`, ran on AMD Instinct MI325X (amd, DO) job a0010

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@1658ba796) | identical | 21.9 | 21.9..21.9 | 1 | - | - | 21.9 | - | - (stored whole) | - | - | - | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x220, rel_diff_vs_sklearn=0.000000 | - | main board, one scored run | - | ok (main@1658ba796 amd/a0010 2026-10-12; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 100.9 | 100.9..100.9 | 1 | 0.217 | - | 100.9 | 100.9 | 0.00 (cpu-arm) | 0.217 (whole/whole) | - | 1466.8 | - | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'clip': False, 'feature_range': [0, 1]}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### minmax-scaler / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/amd_ingest/results.txt (amd/a0010)`, ran on AMD Instinct MI325X (amd, DO) job a0010

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@1658ba796) | identical | 3.2 | 3.2..3.2 | 1 | - | - | 3.2 | - | - (stored whole) | - | - | - | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x11, rel_diff_vs_sklearn=0.000000 | - | main board, one scored run | - | ok (main@1658ba796 amd/a0010 2026-10-12; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 47.3 | 47.3..47.3 | 1 | 0.067 | - | 47.3 | 47.3 | 0.00 (cpu-arm) | 0.067 (whole/whole) | - | 271.0 | - | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'clip': False, 'feature_range': [0, 1]}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### moe / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1682)`, ran on AMD Instinct MI325X (amd, DO) job a1682

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 3.2 ms) = 4.751; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 5.6 ms) = 2.720. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 15.2 | 15.2..15.2 | 1 | - | - | 15.2 | - | - (stored whole) | - | - | - | - | error=Own host reference failed (bba_neural_fixtures.FixturesMissing: neural fixtures missing: run tools/neural_fixtures.py generate (tools/neural_fixtures_box.sh generate on a box, then stage) (no fixture directory: pass --neural-fixtures)); see /root/lq/out/a1682/work-moe-synthetic-def/moe-synthetic-host-quality/host.log | - | main board, one scored run | - | ok (main@a1c96a5ba amd/a1682 2026-10-11; identity vs the nvidia columns: n/a; newer run a8fd6d926/a0006 failed: not_ready) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 5.6 | 5.6..5.6 | 1 | 2.720 | - | - | 5.6 | - (stored kernel) | 2.720 (MIXED ours whole / arm kernel) | - | 3220.1 | 514.6 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 6.8 | 6.8..6.8 | 1 | 2.231 | - | - | 6.8 | - (stored kernel) | 2.231 (MIXED ours whole / arm kernel) | - | 3395.8 | 514.6 | max_rel_diff_vs_torch_eager_fp32=0.044703, rel_fro_vs_torch_eager_fp32=2.011e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 3.6 | 3.6..3.6 | 1 | 4.270 | - | - | 3.6 | - (stored kernel) | 4.270 (MIXED ours whole / arm kernel) | - | 3597.0 | 623.0 | max_rel_diff_vs_torch_eager_fp32=22862.630675, rel_fro_vs_torch_eager_fp32=0.055222 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 3.2 | 3.2..3.2 | 1 | 4.751 | - | - | 3.2 | - (stored kernel) | 4.751 (MIXED ours whole / arm kernel) | - | 3723.1 | 501.4 | max_rel_diff_vs_torch_eager_fp32=22880.029448, rel_fro_vs_torch_eager_fp32=0.055199 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'hidden_size': 1024, 'intermediate_size': 2816, 'norm_topk_prob': True, 'num_experts': 8, 'top_k': 2}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### multilabel-binarizer / istella (rows full, shape X 200000x8; Xq 20000x8; lab 200000; labq 20000; y 200000; yq 20000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/amd_ingest/results.txt (amd/a0005)`, ran on AMD Instinct MI325X (amd, DO) job a0005

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@a8fd6d926) | identical | 104.9 | 104.9..104.9 | 1 | - | - | 104.9 | - | - (stored whole) | - | - | - | - | max_abs_diff_vs_sklearn=0.000000, output_shape=20000x119, rel_diff_vs_sklearn=0.000000 | - | main board, one scored run | - | ok (main@a8fd6d926 amd/a0005 2026-10-12; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 161.2 | 161.2..161.2 | 1 | 0.651 | - | 161.2 | 161.2 | 0.00 (cpu-arm) | 0.651 (whole/whole) | - | 1031.1 | - | output_shape=20000x119 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### multilabel-binarizer / taxi (rows full, shape X 200000x5; Xq 20000x5; lab 200000; labq 20000; y 200000; yq 20000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/amd_ingest/results.txt (amd/a0005)`, ran on AMD Instinct MI325X (amd, DO) job a0005

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@a8fd6d926) | identical | 138.6 | 138.6..138.6 | 1 | - | - | 138.6 | - | - (stored whole) | - | - | - | - | max_abs_diff_vs_sklearn=0.000000, output_shape=20000x489, rel_diff_vs_sklearn=0.000000 | - | main board, one scored run | - | ok (main@a8fd6d926 amd/a0005 2026-10-12; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 134.7 | 134.7..134.7 | 1 | 1.029 | - | 134.7 | 134.7 | 0.00 (cpu-arm) | 1.029 (whole/whole) | - | 2815.8 | - | output_shape=20000x489 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### multinomial-nb / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1706)`, ran on AMD Instinct MI325X (amd, DO) job a1706

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 67.5 | 67.5..67.5 | 1 | - | - | 67.5 | - | - (stored whole) | - | - | - | - | accuracy=0.853620, logloss=3.628565 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1706 2026-10-11; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 209.1 | 209.1..209.1 | 1 | 0.323 | - | 209.1 | 209.1 | 0.00 (cpu-arm) | 0.323 (whole/whole) | - | 3958.7 | - | accuracy=0.853620, logloss=3.087499 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'fit_prior': True, 'force_alpha': True}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), MultinomialNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### multinomial-nb / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1706)`, ran on AMD Instinct MI325X (amd, DO) job a1706

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 27.9 | 27.9..27.9 | 1 | - | - | 27.9 | - | - (stored whole) | - | - | - | - | accuracy=0.723160, logloss=0.590725 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1706 2026-10-11; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 38.6 | 38.6..38.6 | 1 | 0.723 | - | 38.6 | 38.6 | 0.00 (cpu-arm) | 0.723 (whole/whole) | - | 450.2 | - | accuracy=0.723160, logloss=0.590725 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'fit_prior': True, 'force_alpha': True}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), MultinomialNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### multinomial-nb / text (rows full, shape X 86626x4096; Xq 9626x4096; y 86626; yq 9626)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0879)`, ran on AMD Instinct MI325X (amd, DO) job a0879

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@42d1e42c6) | identical | 17.7 | 17.7..17.7 | 1 | - | - | 17.7 | - | - (stored whole) | - | - | - | - | accuracy=0.983067, logloss=0.559529 | - | main board, one scored run | - | ok (main@42d1e42c6 amd/a0879 2026-10-08; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 271.3 | 271.3..271.3 | 1 | 0.065 | - | 271.3 | 271.3 | 0.00 (cpu-arm) | 0.065 (whole/whole) | - | 4455.8 | - | accuracy=0.983067, logloss=0.557319 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'fit_prior': True, 'force_alpha': True}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), MultinomialNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### nadam / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/amd/campaign/measured/algos.nadam.synthetic.rows=full/board.json (amd/paired-e74a9ffcd6adea505e07521941126ea7d6feb89813c58275d059aea2d6d4dd47)`, ran on mojolearn-steward-do-amd

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 6.8 ms) = 1.317. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | 0.8.44 | identical | 9.0 | 9.0..9.0 | 1 | - | - | - | 9.0 | - (stored kernel) | - | - | 1528.0 | 235669.5 | optimizer_update_error_vs_torch_fp32=1.131e-05, rel_fro_vs_torch_eager_fp32=2.945e-08 | - | main board, one scored run | wheel | ok (main@79fcd24a5 amd/paired-e74a9ffcd6adea505e07521941126ea7d6feb89813c58275d059aea2d6d4dd47 2026-10-14; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 6.8 | 6.8..6.8 | 1 | 1.317 | - | - | 6.8 | - (stored kernel) | 1.317 (kernel/kernel) | - | 2782.8 | 896.0 | optimizer_update_error_vs_torch_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 64.4 | 64.4..64.4 | 1 | 0.139 | - | - | 64.4 | - (stored kernel) | 0.139 (kernel/kernel) | - | 3226.3 | 960.0 | optimizer_update_error_vs_torch_fp32=5.621e-05, rel_fro_vs_torch_eager_fp32=1.464e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/102251/vram_* bytes for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'betas': [0.9, 0.999], 'decoupled_weight_decay': False, 'eps': 1e-08, 'lr': 0.001, 'momentum_decay': 0.004, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-fp32 | torch-eager-fp32 |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (optimizer.defaults) | torch (optimizer.defaults) |
| betas | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| capturable | - | false | false |
| decoupled_weight_decay | false | false | false |
| differentiable | - | false | false |
| eps | 1e-08 | 1e-08 | 1e-08 |
| foreach | - | true | true |
| input_arrays | [{"dtype": "float32", "name": "p0", "sha256": "485a7287793a9d7d33d710ea2820ea9c7519d31100db4bafeee813baa532dd7d", "shape": [16777216]}, {"dtype": "float32", "name": "grad0", "sha256": "e6c0159d56a2df748f38d3f3592321edc11067354f52e751e382384dca4178bc", "shape": [16777216]}, {"dtype": "float32", "name": "grad1", "sha256": "249bb61249ee25b0cd42e6bcaa012c27c565dde4ccd798448a60e935fd4ace7e", "shape": [16777216]}, {"dtype": "float32", "name": "grad2", "sha256": "4cf167fba649d3d82fdb6c7c41749a74e8fc8a9bd70491154f03bfc9272b2ed6", "shape": [16777216]}, {"dtype": "float32", "name": "grad3", "sha256": "594f963cfcef38863774b145fac84f56ae7f37190af1cf270e7cb23d3b86b587", "shape": [16777216]}, {"dtype": "float32", "name": "grad4", "sha256": "ee091b2ac84630d1d0328c337b12c5bb6bd6ba2b28f3350eab106bc848960f4e", "shape": [16777216]}, {"dtype": "float32", "name": "grad5", "sha256": "a042832a1a0884016cafe5aada2f1df6bf0b09422e9358536881a536fff70d02", "shape": [16777216]}, {"dtype": "float32", "name": "grad6", "sha256": "736aade529a9b43a7666f9d5a77bf9ef70c253bb5a26280143d941c915910763", "shape": [16777216]}, {"dtype": "float32", "name": "grad7", "sha256": "c4fdaf0915399dc3568c7fd4a49465f9e3c69a10d3e77b1fc15d926db244f302", "shape": [16777216]}, {"dtype": "float32", "name": "grad8", "sha256": "6b970f1bc919f22be37581526578d65a535e33dd886ff46cc64d95aca61fec12", "shape": [16777216]}, {"dtype": "float32", "name": "grad9", "sha256": "00adaa7225e175a85e28e59772e3afbb901e068e5ca8d7332adfa127f7fcbdd7", "shape": [16777216]}] | [{"dtype": "float32", "name": "p0", "sha256": "485a7287793a9d7d33d710ea2820ea9c7519d31100db4bafeee813baa532dd7d", "shape": [16777216]}, {"dtype": "float32", "name": "grad0", "sha256": "e6c0159d56a2df748f38d3f3592321edc11067354f52e751e382384dca4178bc", "shape": [16777216]}, {"dtype": "float32", "name": "grad1", "sha256": "249bb61249ee25b0cd42e6bcaa012c27c565dde4ccd798448a60e935fd4ace7e", "shape": [16777216]}, {"dtype": "float32", "name": "grad2", "sha256": "4cf167fba649d3d82fdb6c7c41749a74e8fc8a9bd70491154f03bfc9272b2ed6", "shape": [16777216]}, {"dtype": "float32", "name": "grad3", "sha256": "594f963cfcef38863774b145fac84f56ae7f37190af1cf270e7cb23d3b86b587", "shape": [16777216]}, {"dtype": "float32", "name": "grad4", "sha256": "ee091b2ac84630d1d0328c337b12c5bb6bd6ba2b28f3350eab106bc848960f4e", "shape": [16777216]}, {"dtype": "float32", "name": "grad5", "sha256": "a042832a1a0884016cafe5aada2f1df6bf0b09422e9358536881a536fff70d02", "shape": [16777216]}, {"dtype": "float32", "name": "grad6", "sha256": "736aade529a9b43a7666f9d5a77bf9ef70c253bb5a26280143d941c915910763", "shape": [16777216]}, {"dtype": "float32", "name": "grad7", "sha256": "c4fdaf0915399dc3568c7fd4a49465f9e3c69a10d3e77b1fc15d926db244f302", "shape": [16777216]}, {"dtype": "float32", "name": "grad8", "sha256": "6b970f1bc919f22be37581526578d65a535e33dd886ff46cc64d95aca61fec12", "shape": [16777216]}, {"dtype": "float32", "name": "grad9", "sha256": "00adaa7225e175a85e28e59772e3afbb901e068e5ca8d7332adfa127f7fcbdd7", "shape": [16777216]}] | [{"dtype": "float32", "name": "p0", "sha256": "485a7287793a9d7d33d710ea2820ea9c7519d31100db4bafeee813baa532dd7d", "shape": [16777216]}, {"dtype": "float32", "name": "grad0", "sha256": "e6c0159d56a2df748f38d3f3592321edc11067354f52e751e382384dca4178bc", "shape": [16777216]}, {"dtype": "float32", "name": "grad1", "sha256": "249bb61249ee25b0cd42e6bcaa012c27c565dde4ccd798448a60e935fd4ace7e", "shape": [16777216]}, {"dtype": "float32", "name": "grad2", "sha256": "4cf167fba649d3d82fdb6c7c41749a74e8fc8a9bd70491154f03bfc9272b2ed6", "shape": [16777216]}, {"dtype": "float32", "name": "grad3", "sha256": "594f963cfcef38863774b145fac84f56ae7f37190af1cf270e7cb23d3b86b587", "shape": [16777216]}, {"dtype": "float32", "name": "grad4", "sha256": "ee091b2ac84630d1d0328c337b12c5bb6bd6ba2b28f3350eab106bc848960f4e", "shape": [16777216]}, {"dtype": "float32", "name": "grad5", "sha256": "a042832a1a0884016cafe5aada2f1df6bf0b09422e9358536881a536fff70d02", "shape": [16777216]}, {"dtype": "float32", "name": "grad6", "sha256": "736aade529a9b43a7666f9d5a77bf9ef70c253bb5a26280143d941c915910763", "shape": [16777216]}, {"dtype": "float32", "name": "grad7", "sha256": "c4fdaf0915399dc3568c7fd4a49465f9e3c69a10d3e77b1fc15d926db244f302", "shape": [16777216]}, {"dtype": "float32", "name": "grad8", "sha256": "6b970f1bc919f22be37581526578d65a535e33dd886ff46cc64d95aca61fec12", "shape": [16777216]}, {"dtype": "float32", "name": "grad9", "sha256": "00adaa7225e175a85e28e59772e3afbb901e068e5ca8d7332adfa127f7fcbdd7", "shape": [16777216]}] |
| learning_rate | 0.001 | 0.001 | 0.001 |
| maximize | - | false | false |
| momentum_decay | 0.004 | 0.004 | 0.004 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| weight_decay | 0.0 | 0.0 | 0.0 |

accepted difference: torch-eager-fp32 capturable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-eager-fp32 differentiable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-eager-fp32 foreach: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

accepted difference: torch-eager-fp32 maximize: both standalone optimizers minimize; our sequence optimizer rejects maximize=True

accepted difference: torch-compile-fp32 capturable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-compile-fp32 differentiable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-compile-fp32 foreach: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

accepted difference: torch-compile-fp32 maximize: both standalone optimizers minimize; our sequence optimizer rejects maximize=True

### nearest-centroid / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1684)`, ran on AMD Instinct MI325X (amd, DO) job a1684

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 38.1 | 38.1..38.1 | 1 | - | - | 38.1 | - | - (stored whole) | - | - | - | - | accuracy=0.852610, logloss=4.299485 | - | main board, one scored run | - | ok (main@a1c96a5ba amd/a1684 2026-10-11; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 669.1 | 669.1..669.1 | 1 | 0.057 | - | 669.1 | 669.1 | 0.00 (cpu-arm) | 0.057 (whole/whole) | - | 4526.8 | - | accuracy=0.852610, logloss=4.117692 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'metric': 'euclidean', 'priors': 'uniform'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### nearest-centroid / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1684)`, ran on AMD Instinct MI325X (amd, DO) job a1684

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 21.4 | 21.4..21.4 | 1 | - | - | 21.4 | - | - (stored whole) | - | - | - | - | accuracy=0.666700, logloss=0.781905 | - | main board, one scored run | - | ok (main@a1c96a5ba amd/a1684 2026-10-11; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 116.3 | 116.3..116.3 | 1 | 0.184 | - | 116.3 | 116.3 | 0.00 (cpu-arm) | 0.184 (whole/whole) | - | 460.1 | - | accuracy=0.666750, logloss=0.781690 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'metric': 'euclidean', 'priors': 'uniform'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### normalizer / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/amd_ingest/results.txt (amd/a0005)`, ran on AMD Instinct MI325X (amd, DO) job a0005

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@a8fd6d926) | identical | 0.1 | 0.1..0.1 | 1 | - | - | 0.1 | - | - (stored whole) | - | - | - | - | max_abs_diff_vs_sklearn=3.576e-07, output_shape=100000x220, rel_diff_vs_sklearn=5.647e-08 | - | main board, one scored run | - | ok (main@a8fd6d926 amd/a0005 2026-10-12; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 26.5 | 26.5..26.5 | 1 | 0.004 | - | 26.5 | 26.5 | 0.00 (cpu-arm) | 0.004 (whole/whole) | - | 1467.5 | - | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'norm': 'l2'}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), Normalizer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### normalizer / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/amd_ingest/results.txt (amd/a0005)`, ran on AMD Instinct MI325X (amd, DO) job a0005

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@a8fd6d926) | identical | 0.1 | 0.1..0.1 | 1 | - | - | 0.1 | - | - (stored whole) | - | - | - | - | max_abs_diff_vs_sklearn=1.192e-07, output_shape=100000x11, rel_diff_vs_sklearn=3.043e-08 | - | main board, one scored run | - | ok (main@a8fd6d926 amd/a0005 2026-10-12; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 1.6 | 1.6..1.6 | 1 | 0.060 | - | 1.6 | 1.6 | 0.00 (cpu-arm) | 0.060 (whole/whole) | - | 271.6 | - | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'norm': 'l2'}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), Normalizer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### onehot / istella (rows full, shape X 1000000x8; Xq 100000x8; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/amd_ingest/results.txt (amd/a0005)`, ran on AMD Instinct MI325X (amd, DO) job a0005

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@a8fd6d926) | identical | 14.2 | 14.2..14.2 | 1 | - | - | 14.2 | - | - (stored whole) | - | - | - | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x119, rel_diff_vs_sklearn=0.000000 | - | main board, one scored run | - | ok (main@a8fd6d926 amd/a0005 2026-10-12; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 13.3 | 13.3..13.3 | 1 | 1.069 | - | 13.3 | 13.3 | 0.00 (cpu-arm) | 1.069 (whole/whole) | - | 539.3 | - | output_shape=100000x119 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'categories': 'auto', 'handle_unknown': 'ignore', 'sparse_output': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), OneHotEncoder (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### onehot / taxi (rows full, shape X 1000000x5; Xq 100000x5; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/amd_ingest/results.txt (amd/a0005)`, ran on AMD Instinct MI325X (amd, DO) job a0005

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@a8fd6d926) | identical | 11.7 | 11.7..11.7 | 1 | - | - | 11.7 | - | - (stored whole) | - | - | - | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x508, rel_diff_vs_sklearn=0.000000 | - | main board, one scored run | - | ok (main@a8fd6d926 amd/a0005 2026-10-12; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 8.0 | 8.0..8.0 | 1 | 1.454 | - | 8.0 | 8.0 | 0.00 (cpu-arm) | 1.454 (whole/whole) | - | 1407.9 | - | output_shape=100000x508 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'categories': 'auto', 'handle_unknown': 'ignore', 'sparse_output': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), OneHotEncoder (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### optimized-theta / synthetic (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1679)`, ran on AMD Instinct MI325X (amd, DO) job a1679

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 407.2 | 407.2..407.2 | 1 | - | - | 407.2 | - | - (stored whole) | - | - | - | - | forecast_rmse=1.438855 | - | main board, one scored run | - | ok (main@a1c96a5ba amd/a1679 2026-10-11; identity vs the nvidia columns: n/a) |
| statsforecast-cpu | statsforecast | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | statsforecast 2.1.1 | opponent | 303.9 | 303.9..303.9 | 1 | 1.340 | - | 303.9 | 303.9 | 0.00 (cpu-arm) | 1.340 (whole/whole) | - | 283.6 | - | forecast_rmse=1.437815 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, statsforecast-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'decomposition_type': 'multiplicative', 'season_length': 24}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### optimized-theta / taxi-hourly (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1679)`, ran on AMD Instinct MI325X (amd, DO) job a1679

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 735.3 | 735.3..735.3 | 1 | - | - | 735.3 | - | - (stored whole) | - | - | - | - | forecast_rmse=49.150860 | - | main board, one scored run | - | ok (main@a1c96a5ba amd/a1679 2026-10-11; identity vs the nvidia columns: n/a) |
| statsforecast-cpu | statsforecast | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | statsforecast 2.1.1 | opponent | 770.4 | 770.4..770.4 | 1 | 0.954 | - | 770.4 | 770.4 | 0.00 (cpu-arm) | 0.954 (whole/whole) | - | 283.5 | - | forecast_rmse=49.356608 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, statsforecast-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'decomposition_type': 'multiplicative', 'season_length': 24}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ordinal / istella (rows full, shape X 1000000x8; Xq 100000x8; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/amd_ingest/results.txt (amd/a0005)`, ran on AMD Instinct MI325X (amd, DO) job a0005

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@a8fd6d926) | identical | 12.8 | 12.8..12.8 | 1 | - | - | 12.8 | - | - (stored whole) | - | - | - | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x8, rel_diff_vs_sklearn=0.000000 | - | main board, one scored run | - | ok (main@a8fd6d926 amd/a0005 2026-10-12; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 13.6 | 13.6..13.6 | 1 | 0.943 | - | 13.6 | 13.6 | 0.00 (cpu-arm) | 0.943 (whole/whole) | - | 267.7 | - | output_shape=100000x8 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'categories': 'auto', 'handle_unknown': 'use_encoded_value', 'unknown_value': -1}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), OrdinalEncoder (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ordinal / taxi (rows full, shape X 1000000x5; Xq 100000x5; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/amd_ingest/results.txt (amd/a0005)`, ran on AMD Instinct MI325X (amd, DO) job a0005

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@a8fd6d926) | identical | 11.6 | 11.6..11.6 | 1 | - | - | 11.6 | - | - (stored whole) | - | - | - | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x5, rel_diff_vs_sklearn=0.000000 | - | main board, one scored run | - | ok (main@a8fd6d926 amd/a0005 2026-10-12; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 7.4 | 7.4..7.4 | 1 | 1.557 | - | 7.4 | 7.4 | 0.00 (cpu-arm) | 1.557 (whole/whole) | - | 246.0 | - | output_shape=100000x5 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'categories': 'auto', 'handle_unknown': 'use_encoded_value', 'unknown_value': -1}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), OrdinalEncoder (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### pagerank / istella (rows full, shape X 20000x220; indices 304830; indices2 62848; indptr 20001; indptr2 20001; y 20000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0870)`, ran on AMD Instinct MI325X (amd, DO) job a0870

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@4df610f3b) | identical | 2.4 | 2.4..2.4 | 1 | - | - | 2.4 | - | - (stored whole) | - | - | - | - | sum=1.000000 | - | main board, one scored run | - | ok (main@4df610f3b amd/a0870 2026-10-08; identity vs nvidia-l40s: MATCH) |
| networkx-cpu | networkx | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | networkx 3.6.1 | opponent | 115.0 | 115.0..115.0 | 1 | 0.021 | - | 115.0 | 115.0 | 0.00 (cpu-arm) | 0.021 (whole/whole) | - | 213.2 | - | sum=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, networkx-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 0.85, 'max_iter': 100, 'tol': 1e-06}. Rows: None. Timed: None.

mismatch: ours takes a dense adjacency matrix (its class's contract), built from the CSR graph before the clock; networkx and cuGraph take the graph itself

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### pagerank / taxi (rows full, shape X 20000x11; indices 253708; indices2 54528; indptr 20001; indptr2 20001; y 20000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0870)`, ran on AMD Instinct MI325X (amd, DO) job a0870

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@4df610f3b) | identical | 2.3 | 2.3..2.3 | 1 | - | - | 2.3 | - | - (stored whole) | - | - | - | - | sum=1.000000 | - | main board, one scored run | - | ok (main@4df610f3b amd/a0870 2026-10-08; identity vs nvidia-l40s: MATCH) |
| networkx-cpu | networkx | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | networkx 3.6.1 | opponent | 88.5 | 88.5..88.5 | 1 | 0.026 | - | 88.5 | 88.5 | 0.00 (cpu-arm) | 0.026 (whole/whole) | - | 178.4 | - | sum=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, networkx-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 0.85, 'max_iter': 100, 'tol': 1e-06}. Rows: None. Timed: None.

mismatch: ours takes a dense adjacency matrix (its class's contract), built from the CSR graph before the clock; networkx and cuGraph take the graph itself

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### permutation-shap / istella (rows full, shape X 100000x220; Xq 100x220; y 100000; yq 100)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1690)`, ran on AMD Instinct MI325X (amd, DO) job a1690

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 1975.6 | 1975.6..1975.6 | 1 | - | - | 1975.6 | - | - (stored whole) | - | - | - | - | rel_error_vs_exact=6.68e-08 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1690 2026-10-11; identity vs the nvidia columns: n/a) |
| shap-cpu | shap | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | shap 0.51.0 | opponent | 22453.9 | 22453.9..22453.9 | 1 | 0.088 | - | 22453.9 | 22453.9 | 0.00 (cpu-arm) | 0.088 (whole/whole) | - | 1842.9 | - | rel_error_vs_exact=3.692e-10 | - | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, shap-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'n_background': 100, 'npermutations': 10}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### permutation-shap / taxi (rows full, shape X 100000x11; Xq 100x11; y 100000; yq 100)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1690)`, ran on AMD Instinct MI325X (amd, DO) job a1690

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 34.4 | 34.4..34.4 | 1 | - | - | 34.4 | - | - (stored whole) | - | - | - | - | rel_error_vs_exact=1.054e-07 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1690 2026-10-11; identity vs the nvidia columns: n/a) |
| shap-cpu | shap | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | shap 0.51.0 | opponent | 104.1 | 104.1..104.1 | 1 | 0.330 | - | 104.1 | 104.1 | 0.00 (cpu-arm) | 0.330 (whole/whole) | - | 478.9 | - | rel_error_vs_exact=1.218e-15 | - | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, shap-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'n_background': 100, 'npermutations': 10}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### poly-features / istella (rows full, shape X 1000000x16; Xq 100000x16; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/amd_ingest/results.txt (amd/a0005)`, ran on AMD Instinct MI325X (amd, DO) job a0005

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@a8fd6d926) | identical | 0.3 | 0.3..0.3 | 1 | - | - | 0.3 | - | - (stored whole) | - | - | - | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x152, rel_diff_vs_sklearn=0.000000 | - | main board, one scored run | - | ok (main@a8fd6d926 amd/a0005 2026-10-12; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 2.2 | 2.2..2.2 | 1 | 0.147 | - | 2.2 | 2.2 | 0.00 (cpu-arm) | 0.147 (whole/whole) | - | 1436.6 | - | output_shape=100000x152 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'degree': 2, 'include_bias': False, 'interaction_only': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), PolynomialFeatures (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### poly-features / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/amd_ingest/results.txt (amd/a0005)`, ran on AMD Instinct MI325X (amd, DO) job a0005

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@a8fd6d926) | identical | 0.2 | 0.2..0.2 | 1 | - | - | 0.2 | - | - (stored whole) | - | - | - | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x77, rel_diff_vs_sklearn=0.000000 | - | main board, one scored run | - | ok (main@a8fd6d926 amd/a0005 2026-10-12; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 1.6 | 1.6..1.6 | 1 | 0.130 | - | 1.6 | 1.6 | 0.00 (cpu-arm) | 0.130 (whole/whole) | - | 375.8 | - | output_shape=100000x77 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'degree': 2, 'include_bias': False, 'interaction_only': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), PolynomialFeatures (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### power-transformer / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/amd_ingest/results.txt (amd/a0005)`, ran on AMD Instinct MI325X (amd, DO) job a0005

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@a8fd6d926) | identical | 1297.3 | 1297.3..1297.3 | 1 | - | - | 1297.3 | - | - (stored whole) | - | - | - | - | max_abs_diff_vs_sklearn=5.575023, output_shape=100000x220, rel_diff_vs_sklearn=0.105054 | - | main board, one scored run | - | ok (main@a8fd6d926 amd/a0005 2026-10-12; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 248761.3 | 248761.3..248761.3 | 1 | 0.005 | - | 248761.3 | 248761.3 | 0.00 (cpu-arm) | 0.005 (whole/whole) | - | 4138.5 | - | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'method': 'yeo-johnson', 'standardize': True}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), PowerTransformer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### power-transformer / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/amd_ingest/results.txt (amd/a0005)`, ran on AMD Instinct MI325X (amd, DO) job a0005

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@a8fd6d926) | identical | 76.7 | 76.7..76.7 | 1 | - | - | 76.7 | - | - (stored whole) | - | - | - | - | max_abs_diff_vs_sklearn=0.064570, output_shape=100000x11, rel_diff_vs_sklearn=0.0008076 | - | main board, one scored run | - | ok (main@a8fd6d926 amd/a0005 2026-10-12; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 11808.3 | 11808.3..11808.3 | 1 | 0.006 | - | 11808.3 | 11808.3 | 0.00 (cpu-arm) | 0.006 (whole/whole) | - | 523.2 | - | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'method': 'yeo-johnson', 'standardize': True}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), PowerTransformer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### qr / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/full-gpu-repair-20261014/amd/repair2/measured/algos.qr.istella.rows=full/board.json (amd/paired-b16350aa8a6cc8060e72869750d6b4982aaaea729ad1d85415ab954eb27709dc)`, ran on mojolearn-steward-do-amd

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | 0.8.44 | identical | 658.6 | 658.6..658.6 | 1 | - | - | 658.6 | 642.9 | 15.68 (upload_ms_separate) | - | - | 4987.9 | 235497.2 | relative_gram_difference=1.485e-07 | - | main board, one scored run | wheel | ok (main@d425c1417 amd/paired-b16350aa8a6cc8060e72869750d6b4982aaaea729ad1d85415ab954eb27709dc 2026-10-14; identity vs the nvidia columns: n/a) |
| torch-gpu | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 2664.5 | 2664.5..2664.5 | 1 | 0.247 | - | - | 2664.5 | - (stored kernel) | 0.241 (kernel/kernel) | - | 2798.6 | 2520.9 | relative_gram_difference=0.0001695 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/97098/vram_* bytes for this pid at the round's end (not a peak)

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'mode': 'reduced'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-gpu |
|---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) |
| input_arrays | [{"dtype": "float32", "sha256": "14c1e3963ef015c9b24cea93d1380679b59623b22edf61cb634001e62b4142a8", "shape": [1000000, 220]}] | [{"dtype": "float32", "sha256": "14c1e3963ef015c9b24cea93d1380679b59623b22edf61cb634001e62b4142a8", "shape": [1000000, 220]}] |
| mode | "reduced" | "reduced" |
| seed | 7 | 7 |

### qr / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/full-gpu-repair-20261014/amd/repair2/measured/algos.qr.taxi.rows=full/board.json (amd/paired-df9aa6dace7616fe768ae9de54e60cb84e80b877cb820fe42c0c8dd73a03978e)`, ran on mojolearn-steward-do-amd

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | 0.8.44 | identical | 14.6 | 14.6..14.6 | 1 | - | - | 14.6 | 13.8 | 0.81 (upload_ms_separate) | - | - | 2515.3 | 236215.2 | relative_gram_difference=1.714e-07 | - | main board, one scored run | wheel | ok (main@d425c1417 amd/paired-df9aa6dace7616fe768ae9de54e60cb84e80b877cb820fe42c0c8dd73a03978e 2026-10-14; identity vs the nvidia columns: n/a) |
| torch-gpu | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 26.0 | 26.0..26.0 | 1 | 0.561 | - | - | 26.0 | - (stored kernel) | 0.530 (kernel/kernel) | - | 1646.0 | 126.0 | relative_gram_difference=9.54e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/96668/vram_* bytes for this pid at the round's end (not a peak)

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'mode': 'reduced'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-gpu |
|---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) |
| input_arrays | [{"dtype": "float32", "sha256": "33fdb983b45d51732ce911c8887c0c9f9cf7a7416ae67ebfe983fa0b3a4c8b77", "shape": [1000000, 11]}] | [{"dtype": "float32", "sha256": "33fdb983b45d51732ce911c8887c0c9f9cf7a7416ae67ebfe983fa0b3a4c8b77", "shape": [1000000, 11]}] |
| mode | "reduced" | "reduced" |
| seed | 7 | 7 |

### quantile-transformer / istella (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/amd_ingest/results.txt (amd/a0005)`, ran on AMD Instinct MI325X (amd, DO) job a0005

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@a8fd6d926) | identical | 125.8 | 125.8..125.8 | 1 | - | - | 125.8 | - | - (stored whole) | - | - | - | - | max_abs_diff_vs_sklearn=5.96e-08, output_shape=100000x220, rel_diff_vs_sklearn=2.764e-08 | - | main board, one scored run | - | ok (main@a8fd6d926 amd/a0005 2026-10-12; identity vs the nvidia columns: n/a) |

settings: {'ignore_implicit_zeros': False, 'n_quantiles': 1000, 'output_distribution': 'uniform', 'random_state': 7, 'subsample': 1000000000}. Rows: None. Timed: None.

mismatch: opponents withheld: the lane settings at this tree's HEAD differ from the settings release-board-resume-r2 recorded for its opponent race (an opponent job must score them again)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### quantile-transformer / taxi (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/amd_ingest/results.txt (amd/a0005)`, ran on AMD Instinct MI325X (amd, DO) job a0005

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@a8fd6d926) | identical | 9.1 | 9.1..9.1 | 1 | - | - | 9.1 | - | - (stored whole) | - | - | - | - | max_abs_diff_vs_sklearn=5.96e-08, output_shape=100000x11, rel_diff_vs_sklearn=2.294e-08 | - | main board, one scored run | - | ok (main@a8fd6d926 amd/a0005 2026-10-12; identity vs the nvidia columns: n/a) |

settings: {'ignore_implicit_zeros': False, 'n_quantiles': 1000, 'output_distribution': 'uniform', 'random_state': 7, 'subsample': 1000000000}. Rows: None. Timed: None.

mismatch: opponents withheld: the lane settings at this tree's HEAD differ from the settings release-board-resume-r2 recorded for its opponent race (an opponent job must score them again)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### quantile / istella (rows full, shape X 100000x220; Xq 100000x220; y 100000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0535)`, ran on AMD Instinct MI325X (amd, DO) job a0535

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@de2b2b739) | identical | 862.1 | 862.1..862.1 | 1 | - | - | 862.1 | - | - (stored whole) | - | - | - | - | finite=True, r2=-0.039998, rmse=0.851877 | - | main board, one scored run | - | ok (main@de2b2b739 amd/a0535 2026-10-08; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 878876.6 | 878876.6..878876.6 | 1 | 0.0009809 | - | 878876.6 | 878876.6 | 0.00 (cpu-arm) | 0.0009809 (whole/whole) | - | 8006.1 | - | finite=True, r2=-0.044780, rmse=0.853833 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'fit_intercept': True, 'quantile': 0.5, 'solver': 'highs'}. Rows: None. Timed: None.

mismatch: solver: 'highs' passed to both; ours accepts the name and always runs ADMM (max_iter=5000, tol=1e-4, ours only), scikit-learn solves the HiGHS LP

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### quantile / taxi (rows full, shape X 100000x11; Xq 100000x11; y 100000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0535)`, ran on AMD Instinct MI325X (amd, DO) job a0535

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@de2b2b739) | identical | 499.3 | 499.3..499.3 | 1 | - | - | 499.3 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.899596, rmse=5.046749 | - | main board, one scored run | - | ok (main@de2b2b739 amd/a0535 2026-10-08; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 225040.2 | 225040.2..225040.2 | 1 | 0.002 | - | 225040.2 | 225040.2 | 0.00 (cpu-arm) | 0.002 (whole/whole) | - | 928.9 | - | finite=True, r2=0.899678, rmse=5.044706 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'fit_intercept': True, 'quantile': 0.5, 'solver': 'highs'}. Rows: None. Timed: None.

mismatch: solver: 'highs' passed to both; ours accepts the name and always runs ADMM (max_iter=5000, tol=1e-4, ours only), scikit-learn solves the HiGHS LP

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### randomized-svd / istella (rows full, shape X 900000x220; Xq 100000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/full-gpu-repair-20261014/amd/repair2/measured/algos.randomized-svd.istella.rows=full/board.json (amd/paired-c26b3dba59844b3c0a67c83a7585fcb5b049f5155c7b9674f90725a19744e7be)`, ran on mojolearn-steward-do-amd

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | 0.8.44 | identical | 119.6 | 119.6..119.6 | 1 | - | - | 119.6 | 105.5 | 14.10 (upload_ms_separate) | - | - | 3515.3 | 235755.1 | relative_reconstruction_error=0.0002359 | - | main board, one scored run | wheel | ok (main@d425c1417 amd/paired-c26b3dba59844b3c0a67c83a7585fcb5b049f5155c7b9674f90725a19744e7be 2026-10-14; identity vs the nvidia columns: n/a) |
| torch-gpu | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 204.3 | 204.3..204.3 | 1 | 0.586 | - | - | 204.3 | - (stored kernel) | 0.517 (kernel/kernel) | - | 4573.9 | 1018.1 | relative_reconstruction_error=0.0002359 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/95211/vram_* bytes for this pid at the round's end (not a peak)

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'n_components': 8, 'n_iter': 4, 'n_oversamples': 10, 'power_iteration_normalizer': 'QR', 'random_state': 7}. Rows: None. Timed: None.

mismatch: torch-gpu is torch.svd_lowrank(q = n_components + n_oversamples, clipped to min(X.shape), niter=4), its randomized range finder with a QR step every half iteration, as ours and scikit-learn's power_iteration_normalizer='QR'

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-gpu |
|---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) |
| input_arrays | [{"dtype": "float32", "sha256": "5403ce8f5a28e98afde7196e066c67e47d5dc39ca38a2907c58baceae6789701", "shape": [900000, 220]}] | [{"dtype": "float32", "sha256": "5403ce8f5a28e98afde7196e066c67e47d5dc39ca38a2907c58baceae6789701", "shape": [900000, 220]}] |
| n_components | 8 | 8 |
| n_iter | 4 | 4 |
| n_oversamples | 10 | 10 |
| power_iteration_normalizer | "QR" | "QR" |
| seed | 7 | 7 |

### randomized-svd / taxi (rows full, shape X 900000x11; Xq 100000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/full-gpu-repair-20261014/amd/repair2/measured/algos.randomized-svd.taxi.rows=full/board.json (amd/paired-7fbd2f00aec152e90f440a7da9fa5206a7170c08dbb8f1aeb453130b8c89ec16)`, ran on mojolearn-steward-do-amd

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | 0.8.44 | identical | 43.2 | 43.2..43.2 | 1 | - | - | 43.2 | 42.4 | 0.74 (upload_ms_separate) | - | - | 2718.2 | 236401.1 | relative_reconstruction_error=0.027197 | - | main board, one scored run | wheel | ok (main@d425c1417 amd/paired-7fbd2f00aec152e90f440a7da9fa5206a7170c08dbb8f1aeb453130b8c89ec16 2026-10-14; identity vs the nvidia columns: n/a) |
| torch-gpu | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 118.9 | 118.9..118.9 | 1 | 0.363 | - | - | 118.9 | - (stored kernel) | 0.357 (kernel/kernel) | - | 3772.6 | 228.0 | relative_reconstruction_error=0.027197 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/94669/vram_* bytes for this pid at the round's end (not a peak)

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'n_components': 8, 'n_iter': 4, 'n_oversamples': 10, 'power_iteration_normalizer': 'QR', 'random_state': 7}. Rows: None. Timed: None.

mismatch: torch-gpu is torch.svd_lowrank(q = n_components + n_oversamples, clipped to min(X.shape), niter=4), its randomized range finder with a QR step every half iteration, as ours and scikit-learn's power_iteration_normalizer='QR'

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-gpu |
|---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) |
| input_arrays | [{"dtype": "float32", "sha256": "50644520ed293dfc0da84cfd01db8311abacbbf5fa89525a2cf58d3cbf7ed24a", "shape": [900000, 11]}] | [{"dtype": "float32", "sha256": "50644520ed293dfc0da84cfd01db8311abacbbf5fa89525a2cf58d3cbf7ed24a", "shape": [900000, 11]}] |
| n_components | 8 | 8 |
| n_iter | 4 | 4 |
| n_oversamples | 10 | 10 |
| power_iteration_normalizer | "QR" | "QR" |
| seed | 7 | 7 |

### resnet-block / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/amd_ingest/results.txt (amd/a0004)`, ran on AMD Instinct MI325X (amd, DO) job a0004

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 2.1 ms) = 6.106; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 2.2 ms) = 5.696. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@bac431c11) | identical | 12.6 | 12.6..12.6 | 1 | - | - | 12.6 | - | - (stored whole) | - | - | - | - | relative_error_vs_own_host=0.000000 | - | main board, one scored run | - | ok (main@bac431c11 amd/a0004 2026-10-12; identity vs the nvidia columns: n/a; newer run a8fd6d926/a0006 failed: not_ready) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 2.2 | 2.2..2.2 | 1 | 5.696 | - | - | 2.2 | - (stored kernel) | 5.696 (MIXED ours whole / arm kernel) | - | 2892.9 | 543.6 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 2.2 | 2.2..2.2 | 1 | 5.656 | - | - | 2.2 | - (stored kernel) | 5.656 (MIXED ours whole / arm kernel) | - | 3084.9 | 505.1 | max_rel_diff_vs_torch_eager_fp32=0.834465, rel_fro_vs_torch_eager_fp32=1.645e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 2.1 | 2.1..2.1 | 1 | 6.106 | - | - | 2.1 | - (stored kernel) | 6.106 (MIXED ours whole / arm kernel) | - | 3181.3 | 468.0 | max_rel_diff_vs_torch_eager_fp32=28679.370880, rel_fro_vs_torch_eager_fp32=0.004232 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 2.4 | 2.4..2.4 | 1 | 5.245 | - | - | 2.4 | - (stored kernel) | 5.245 (MIXED ours whole / arm kernel) | - | 3029.2 | 430.5 | max_rel_diff_vs_torch_eager_fp32=18454.670906, rel_fro_vs_torch_eager_fp32=0.003485 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'inplanes': 64, 'planes': 64}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### rfe / istella (rows full, shape X 200000x220; Xq 100000x220; y 200000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/rfe-istella-20261012/ingest/rfe_exact_amd.txt (amd/a0010)`, ran on AMD Instinct MI325X (amd, DO) job a0010

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@1658ba796) | identical | 887.0 | 887.0..887.0 | 1 | - | - | 887.0 | - | - (stored whole) | - | - | - | - | jaccard_vs_exact=0.774194, jaccard_vs_sklearn=0.848739, n_selected=110 | - | main board, one scored run | - | ok (main@1658ba796 amd/a0010 2026-10-12; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 26346.6 | 26346.6..26346.6 | 1 | 0.034 | - | 26346.6 | 26346.6 | 0.00 (cpu-arm) | 0.034 (whole/whole) | - | 1812.8 | - | jaccard_vs_sklearn=1.000000, n_selected=110 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'estimator': 'LogisticRegression(max_iter=200)', 'n_features_to_select': 'half', 'step': 0.1}. Rows: None. Timed: None.

mismatch: nested LogisticRegression(max_iter=200): ours is L-BFGS (cuML's quasi-Newton, memory 5, solver='qn'), scikit-learn solver='lbfgs' (scipy, memory 10); C=1.0 and tol=1e-4 are both libraries' defaults; each stops by its own criterion

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### rfe / taxi (rows full, shape X 200000x11; Xq 100000x11; y 200000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/rfe-istella-20261012/ingest/rfe_exact_amd.txt (amd/a0010)`, ran on AMD Instinct MI325X (amd, DO) job a0010

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@1658ba796) | identical | 103.4 | 103.4..103.4 | 1 | - | - | 103.4 | - | - (stored whole) | - | - | - | - | jaccard_vs_exact=1.000000, jaccard_vs_sklearn=1.000000, n_selected=5 | - | main board, one scored run | - | ok (main@1658ba796 amd/a0010 2026-10-12; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 1156.0 | 1156.0..1156.0 | 1 | 0.089 | - | 1156.0 | 1156.0 | 0.00 (cpu-arm) | 0.089 (whole/whole) | - | 298.5 | - | jaccard_vs_sklearn=1.000000, n_selected=5 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'estimator': 'LogisticRegression(max_iter=200)', 'n_features_to_select': 'half', 'step': 0.1}. Rows: None. Timed: None.

mismatch: nested LogisticRegression(max_iter=200): ours is L-BFGS (cuML's quasi-Newton, memory 5, solver='qn'), scikit-learn solver='lbfgs' (scipy, memory 10); C=1.0 and tol=1e-4 are both libraries' defaults; each stops by its own criterion

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ridge-cv / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1661)`, ran on AMD Instinct MI325X (amd, DO) job a1661

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@6cd6bed7f) | identical | 62879.7 | 62879.7..62879.7 | 1 | - | - | 62879.7 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.328684, rmse=0.684422 | - | main board, one scored run | - | ok (main@6cd6bed7f amd/a1661 2026-10-11; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 78354.8 | 78354.8..78354.8 | 1 | 0.802 | - | 78354.8 | 78354.8 | 0.00 (cpu-arm) | 0.802 (whole/whole) | - | 8705.1 | - | finite=True, r2=0.328683, rmse=0.684423 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha_per_target': False, 'alphas': [0.001, 0.01, 0.1, 1.0, 10.0], 'cv': 5, 'fit_intercept': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ridge-cv / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1068)`, ran on AMD Instinct MI325X (amd, DO) job a1068

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@8ed94710a) | identical | 288.9 | 288.9..288.9 | 1 | - | - | 288.9 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.908981, rmse=4.805109 | - | main board, one scored run | - | ok (main@8ed94710a amd/a1068 2026-10-09; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 994.6 | 994.6..994.6 | 1 | 0.290 | - | 994.6 | 994.6 | 0.00 (cpu-arm) | 0.290 (whole/whole) | - | 380.7 | - | finite=True, r2=0.908983, rmse=4.805057 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha_per_target': False, 'alphas': [0.001, 0.01, 0.1, 1.0, 10.0], 'cv': 5, 'fit_intercept': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### rmsprop / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/amd/campaign/measured/algos.rmsprop.synthetic.rows=full/board.json (amd/paired-421be61b8e830b32967486d6092277b4b14eae1e3675c88e34bac9620bdfb82f)`, ran on mojolearn-steward-do-amd

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 4.8 ms) = 1.872. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | 0.8.44 | identical | 9.0 | 9.0..9.0 | 1 | - | - | - | 9.0 | - (stored kernel) | - | - | 1528.4 | 235669.5 | optimizer_update_error_vs_torch_fp32=2.228e-06, rel_fro_vs_torch_eager_fp32=3.829e-08 | - | main board, one scored run | wheel | ok (main@79fcd24a5 amd/paired-421be61b8e830b32967486d6092277b4b14eae1e3675c88e34bac9620bdfb82f 2026-10-14; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 4.8 | 4.8..4.8 | 1 | 1.872 | - | - | 4.8 | - (stored kernel) | 1.872 (kernel/kernel) | - | 2740.3 | 832.0 | optimizer_update_error_vs_torch_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 40.3 | 40.3..40.3 | 1 | 0.223 | - | - | 40.3 | - (stored kernel) | 0.223 (kernel/kernel) | - | 3087.8 | 832.0 | optimizer_update_error_vs_torch_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/48175/vram_* bytes for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'alpha': 0.99, 'centered': False, 'eps': 1e-08, 'lr': 0.001, 'momentum': 0.0, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-fp32 | torch-eager-fp32 |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (optimizer.defaults) | torch (optimizer.defaults) |
| alpha | 0.99 | 0.99 | 0.99 |
| capturable | - | false | false |
| centered | false | false | false |
| differentiable | - | false | false |
| eps | 1e-08 | 1e-08 | 1e-08 |
| foreach | - | true | true |
| input_arrays | [{"dtype": "float32", "name": "p0", "sha256": "485a7287793a9d7d33d710ea2820ea9c7519d31100db4bafeee813baa532dd7d", "shape": [16777216]}, {"dtype": "float32", "name": "grad0", "sha256": "e6c0159d56a2df748f38d3f3592321edc11067354f52e751e382384dca4178bc", "shape": [16777216]}, {"dtype": "float32", "name": "grad1", "sha256": "249bb61249ee25b0cd42e6bcaa012c27c565dde4ccd798448a60e935fd4ace7e", "shape": [16777216]}, {"dtype": "float32", "name": "grad2", "sha256": "4cf167fba649d3d82fdb6c7c41749a74e8fc8a9bd70491154f03bfc9272b2ed6", "shape": [16777216]}, {"dtype": "float32", "name": "grad3", "sha256": "594f963cfcef38863774b145fac84f56ae7f37190af1cf270e7cb23d3b86b587", "shape": [16777216]}, {"dtype": "float32", "name": "grad4", "sha256": "ee091b2ac84630d1d0328c337b12c5bb6bd6ba2b28f3350eab106bc848960f4e", "shape": [16777216]}, {"dtype": "float32", "name": "grad5", "sha256": "a042832a1a0884016cafe5aada2f1df6bf0b09422e9358536881a536fff70d02", "shape": [16777216]}, {"dtype": "float32", "name": "grad6", "sha256": "736aade529a9b43a7666f9d5a77bf9ef70c253bb5a26280143d941c915910763", "shape": [16777216]}, {"dtype": "float32", "name": "grad7", "sha256": "c4fdaf0915399dc3568c7fd4a49465f9e3c69a10d3e77b1fc15d926db244f302", "shape": [16777216]}, {"dtype": "float32", "name": "grad8", "sha256": "6b970f1bc919f22be37581526578d65a535e33dd886ff46cc64d95aca61fec12", "shape": [16777216]}, {"dtype": "float32", "name": "grad9", "sha256": "00adaa7225e175a85e28e59772e3afbb901e068e5ca8d7332adfa127f7fcbdd7", "shape": [16777216]}] | [{"dtype": "float32", "name": "p0", "sha256": "485a7287793a9d7d33d710ea2820ea9c7519d31100db4bafeee813baa532dd7d", "shape": [16777216]}, {"dtype": "float32", "name": "grad0", "sha256": "e6c0159d56a2df748f38d3f3592321edc11067354f52e751e382384dca4178bc", "shape": [16777216]}, {"dtype": "float32", "name": "grad1", "sha256": "249bb61249ee25b0cd42e6bcaa012c27c565dde4ccd798448a60e935fd4ace7e", "shape": [16777216]}, {"dtype": "float32", "name": "grad2", "sha256": "4cf167fba649d3d82fdb6c7c41749a74e8fc8a9bd70491154f03bfc9272b2ed6", "shape": [16777216]}, {"dtype": "float32", "name": "grad3", "sha256": "594f963cfcef38863774b145fac84f56ae7f37190af1cf270e7cb23d3b86b587", "shape": [16777216]}, {"dtype": "float32", "name": "grad4", "sha256": "ee091b2ac84630d1d0328c337b12c5bb6bd6ba2b28f3350eab106bc848960f4e", "shape": [16777216]}, {"dtype": "float32", "name": "grad5", "sha256": "a042832a1a0884016cafe5aada2f1df6bf0b09422e9358536881a536fff70d02", "shape": [16777216]}, {"dtype": "float32", "name": "grad6", "sha256": "736aade529a9b43a7666f9d5a77bf9ef70c253bb5a26280143d941c915910763", "shape": [16777216]}, {"dtype": "float32", "name": "grad7", "sha256": "c4fdaf0915399dc3568c7fd4a49465f9e3c69a10d3e77b1fc15d926db244f302", "shape": [16777216]}, {"dtype": "float32", "name": "grad8", "sha256": "6b970f1bc919f22be37581526578d65a535e33dd886ff46cc64d95aca61fec12", "shape": [16777216]}, {"dtype": "float32", "name": "grad9", "sha256": "00adaa7225e175a85e28e59772e3afbb901e068e5ca8d7332adfa127f7fcbdd7", "shape": [16777216]}] | [{"dtype": "float32", "name": "p0", "sha256": "485a7287793a9d7d33d710ea2820ea9c7519d31100db4bafeee813baa532dd7d", "shape": [16777216]}, {"dtype": "float32", "name": "grad0", "sha256": "e6c0159d56a2df748f38d3f3592321edc11067354f52e751e382384dca4178bc", "shape": [16777216]}, {"dtype": "float32", "name": "grad1", "sha256": "249bb61249ee25b0cd42e6bcaa012c27c565dde4ccd798448a60e935fd4ace7e", "shape": [16777216]}, {"dtype": "float32", "name": "grad2", "sha256": "4cf167fba649d3d82fdb6c7c41749a74e8fc8a9bd70491154f03bfc9272b2ed6", "shape": [16777216]}, {"dtype": "float32", "name": "grad3", "sha256": "594f963cfcef38863774b145fac84f56ae7f37190af1cf270e7cb23d3b86b587", "shape": [16777216]}, {"dtype": "float32", "name": "grad4", "sha256": "ee091b2ac84630d1d0328c337b12c5bb6bd6ba2b28f3350eab106bc848960f4e", "shape": [16777216]}, {"dtype": "float32", "name": "grad5", "sha256": "a042832a1a0884016cafe5aada2f1df6bf0b09422e9358536881a536fff70d02", "shape": [16777216]}, {"dtype": "float32", "name": "grad6", "sha256": "736aade529a9b43a7666f9d5a77bf9ef70c253bb5a26280143d941c915910763", "shape": [16777216]}, {"dtype": "float32", "name": "grad7", "sha256": "c4fdaf0915399dc3568c7fd4a49465f9e3c69a10d3e77b1fc15d926db244f302", "shape": [16777216]}, {"dtype": "float32", "name": "grad8", "sha256": "6b970f1bc919f22be37581526578d65a535e33dd886ff46cc64d95aca61fec12", "shape": [16777216]}, {"dtype": "float32", "name": "grad9", "sha256": "00adaa7225e175a85e28e59772e3afbb901e068e5ca8d7332adfa127f7fcbdd7", "shape": [16777216]}] |
| learning_rate | 0.001 | 0.001 | 0.001 |
| maximize | - | false | false |
| momentum | 0.0 | 0.0 | 0.0 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| weight_decay | 0.0 | 0.0 | 0.0 |

accepted difference: torch-eager-fp32 capturable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-eager-fp32 differentiable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-eager-fp32 foreach: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

accepted difference: torch-eager-fp32 maximize: both standalone optimizers minimize; our sequence optimizer rejects maximize=True

accepted difference: torch-compile-fp32 capturable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-compile-fp32 differentiable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-compile-fp32 foreach: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

accepted difference: torch-compile-fp32 maximize: both standalone optimizers minimize; our sequence optimizer rejects maximize=True

### rnn-clf / synthetic (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1211)`, ran on AMD Instinct MI325X (amd, DO) job a1211

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 950.8 ms) = 0.322; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 879.7 ms) = 0.348. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 305.9 | 305.9..305.9 | 1 | - | - | 305.9 | - | - (stored whole) | - | - | - | - | accuracy=0.967828, logloss=0.079029 | - | main board, one scored run | - | ok (main@1665b5626 amd/a1211 2026-10-10; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 889.2 | 889.2..889.2 | 1 | 0.344 | - | - | 889.2 | - (stored kernel) | 0.344 (MIXED ours whole / arm kernel) | - | 3215.3 | 110.3 | accuracy=0.953559 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 879.7 | 879.7..879.7 | 1 | 0.348 | - | - | 879.7 | - (stored kernel) | 0.348 (MIXED ours whole / arm kernel) | - | 3255.8 | 110.3 | accuracy=0.953559 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 950.8 | 950.8..950.8 | 1 | 0.322 | - | - | 950.8 | - (stored kernel) | 0.322 (MIXED ours whole / arm kernel) | - | 4965.1 | 98.7 | accuracy=0.953559 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 995.6 | 995.6..995.6 | 1 | 0.307 | - | - | 995.6 | - (stored kernel) | 0.307 (MIXED ours whole / arm kernel) | - | 5005.9 | 98.7 | accuracy=0.953559 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'nonlinearity': 'tanh', 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### rnn-clf / taxi-hourly (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1211)`, ran on AMD Instinct MI325X (amd, DO) job a1211

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 975.1 ms) = 0.314; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 743.5 ms) = 0.412. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 306.6 | 306.6..306.6 | 1 | - | - | 306.6 | - | - (stored whole) | - | - | - | - | accuracy=0.862684, logloss=0.313008 | - | main board, one scored run | - | ok (main@1665b5626 amd/a1211 2026-10-10; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 743.5 | 743.5..743.5 | 1 | 0.412 | - | - | 743.5 | - (stored kernel) | 0.412 (MIXED ours whole / arm kernel) | - | 3478.2 | 110.3 | accuracy=0.868056 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 885.8 | 885.8..885.8 | 1 | 0.346 | - | - | 885.8 | - (stored kernel) | 0.346 (MIXED ours whole / arm kernel) | - | 3256.1 | 110.3 | accuracy=0.868056 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 975.1 | 975.1..975.1 | 1 | 0.314 | - | - | 975.1 | - (stored kernel) | 0.314 (MIXED ours whole / arm kernel) | - | 5234.9 | 98.7 | accuracy=0.867947 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1002.0 | 1002.0..1002.0 | 1 | 0.306 | - | - | 1002.0 | - (stored kernel) | 0.306 (MIXED ours whole / arm kernel) | - | 5006.2 | 98.7 | accuracy=0.867947 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'nonlinearity': 'tanh', 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### rnn-reg / synthetic (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1211)`, ran on AMD Instinct MI325X (amd, DO) job a1211

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 898.5 ms) = 0.334; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 891.9 ms) = 0.337. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 300.1 | 300.1..300.1 | 1 | - | - | 300.1 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.978602, rmse=0.169476 | - | main board, one scored run | - | ok (main@1665b5626 amd/a1211 2026-10-10; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 891.9 | 891.9..891.9 | 1 | 0.337 | - | - | 891.9 | - (stored kernel) | 0.337 (MIXED ours whole / arm kernel) | - | 3040.0 | 109.9 | finite=True, r2=0.977348, rmse=0.174374 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 901.6 | 901.6..901.6 | 1 | 0.333 | - | - | 901.6 | - (stored kernel) | 0.333 (MIXED ours whole / arm kernel) | - | 3082.4 | 109.9 | finite=True, r2=0.977348, rmse=0.174374 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 898.5 | 898.5..898.5 | 1 | 0.334 | - | - | 898.5 | - (stored kernel) | 0.334 (MIXED ours whole / arm kernel) | - | 3663.2 | 98.4 | finite=True, r2=0.977371, rmse=0.174283 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 953.5 | 953.5..953.5 | 1 | 0.315 | - | - | 953.5 | - (stored kernel) | 0.315 (MIXED ours whole / arm kernel) | - | 3705.5 | 98.4 | finite=True, r2=0.977371, rmse=0.174283 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'nonlinearity': 'tanh', 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### rnn-reg / taxi-hourly (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1211)`, ran on AMD Instinct MI325X (amd, DO) job a1211

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 1010.8 ms) = 0.297; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 762.7 ms) = 0.394. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 300.7 | 300.7..300.7 | 1 | - | - | 300.7 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.743904, rmse=0.548825 | - | main board, one scored run | - | ok (main@1665b5626 amd/a1211 2026-10-10; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 863.9 | 863.9..863.9 | 1 | 0.348 | - | - | 863.9 | - (stored kernel) | 0.348 (MIXED ours whole / arm kernel) | - | 3041.2 | 109.9 | finite=True, r2=0.738796, rmse=0.554271 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 762.7 | 762.7..762.7 | 1 | 0.394 | - | - | 762.7 | - (stored kernel) | 0.394 (MIXED ours whole / arm kernel) | - | 3082.6 | 109.9 | finite=True, r2=0.738796, rmse=0.554271 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1010.8 | 1010.8..1010.8 | 1 | 0.297 | - | - | 1010.8 | - (stored kernel) | 0.297 (MIXED ours whole / arm kernel) | - | 3664.6 | 98.4 | finite=True, r2=0.739017, rmse=0.554037 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1033.9 | 1033.9..1033.9 | 1 | 0.291 | - | - | 1033.9 | - (stored kernel) | 0.291 (MIXED ours whole / arm kernel) | - | 3705.5 | 98.4 | finite=True, r2=0.739017, rmse=0.554037 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'nonlinearity': 'tanh', 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### robust-scaler / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/amd_ingest/results.txt (amd/a0005)`, ran on AMD Instinct MI325X (amd, DO) job a0005

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@a8fd6d926) | identical | 125.0 | 125.0..125.0 | 1 | - | - | 125.0 | - | - (stored whole) | - | - | - | - | max_abs_diff_vs_sklearn=0.0001221, output_shape=100000x220, rel_diff_vs_sklearn=7.339e-09 | - | main board, one scored run | - | ok (main@a8fd6d926 amd/a0005 2026-10-12; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 2630.1 | 2630.1..2630.1 | 1 | 0.048 | - | 2630.1 | 2630.1 | 0.00 (cpu-arm) | 0.048 (whole/whole) | - | 1467.2 | - | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'quantile_range': [25.0, 75.0], 'unit_variance': False, 'with_centering': True, 'with_scaling': True}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), RobustScaler (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### robust-scaler / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/amd_ingest/results.txt (amd/a0005)`, ran on AMD Instinct MI325X (amd, DO) job a0005

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@a8fd6d926) | identical | 8.6 | 8.6..8.6 | 1 | - | - | 8.6 | - | - (stored whole) | - | - | - | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x11, rel_diff_vs_sklearn=0.000000 | - | main board, one scored run | - | ok (main@a8fd6d926 amd/a0005 2026-10-12; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 104.3 | 104.3..104.3 | 1 | 0.083 | - | 104.3 | 104.3 | 0.00 (cpu-arm) | 0.083 (whole/whole) | - | 274.4 | - | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'quantile_range': [25.0, 75.0], 'unit_variance': False, 'with_centering': True, 'with_scaling': True}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), RobustScaler (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### select-chi2 / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/amd_ingest/results.txt (amd/a0010)`, ran on AMD Instinct MI325X (amd, DO) job a0010

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@1658ba796) | identical | 26.4 | 26.4..26.4 | 1 | - | - | 26.4 | - | - (stored whole) | - | - | - | - | jaccard_vs_sklearn=1.000000, n_selected=110 | - | main board, one scored run | - | ok (main@1658ba796 amd/a0010 2026-10-12; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 282.9 | 282.9..282.9 | 1 | 0.093 | - | 282.9 | 282.9 | 0.00 (cpu-arm) | 0.093 (whole/whole) | - | 3765.1 | - | jaccard_vs_sklearn=1.000000, n_selected=110 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'k': 'half'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### select-chi2 / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/amd_ingest/results.txt (amd/a0010)`, ran on AMD Instinct MI325X (amd, DO) job a0010

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@1658ba796) | identical | 10.6 | 10.6..10.6 | 1 | - | - | 10.6 | - | - (stored whole) | - | - | - | - | jaccard_vs_sklearn=1.000000, n_selected=5 | - | main board, one scored run | - | ok (main@1658ba796 amd/a0010 2026-10-12; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 59.3 | 59.3..59.3 | 1 | 0.178 | - | 59.3 | 59.3 | 0.00 (cpu-arm) | 0.178 (whole/whole) | - | 439.0 | - | jaccard_vs_sklearn=1.000000, n_selected=5 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'k': 'half'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### select-d / synthetic (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/amd_ingest/results.txt (amd/a0012)`, ran on AMD Instinct MI325X (amd, DO) job a0012

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.42 (source build, main@4e9a2bf60) | identical | 1.0 | 1.0..1.0 | 1 | - | - | 1.0 | - | - (stored whole) | - | - | - | - | relative_error_vs_own_host=0.000000 | - | main board, one scored run | - | ok (main@4e9a2bf60 amd/a0012 2026-10-12; identity vs the nvidia columns: n/a) |
| statsmodels-cpu | statsmodels | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | statsmodels 0.15.0 | opponent | 4.0 | 4.0..4.0 | 1 | 0.240 | - | 4.0 | 4.0 | 0.00 (cpu-arm) | 0.240 (whole/whole) | - | 217.9 | - | d_agreement_vs_statsmodels=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, statsmodels-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'D': 0, 'pval_threshold': 0.05, 's': 0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### select-d / taxi-hourly (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/amd_ingest/results.txt (amd/a0012)`, ran on AMD Instinct MI325X (amd, DO) job a0012

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.42 (source build, main@4e9a2bf60) | identical | 0.9 | 0.9..0.9 | 1 | - | - | 0.9 | - | - (stored whole) | - | - | - | - | relative_error_vs_own_host=0.000000 | - | main board, one scored run | - | ok (main@4e9a2bf60 amd/a0012 2026-10-12; identity vs the nvidia columns: n/a) |
| statsmodels-cpu | statsmodels | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | statsmodels 0.15.0 | opponent | 2.7 | 2.7..2.7 | 1 | 0.339 | - | 2.7 | 2.7 | 0.00 (cpu-arm) | 0.339 (whole/whole) | - | 217.6 | - | d_agreement_vs_statsmodels=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, statsmodels-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'D': 0, 'pval_threshold': 0.05, 's': 0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### select-f-classif / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/amd_ingest/results.txt (amd/a0010)`, ran on AMD Instinct MI325X (amd, DO) job a0010

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@1658ba796) | identical | 27.4 | 27.4..27.4 | 1 | - | - | 27.4 | - | - (stored whole) | - | - | - | - | jaccard_vs_sklearn=1.000000, n_selected=110 | - | main board, one scored run | - | ok (main@1658ba796 amd/a0010 2026-10-12; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 343.0 | 343.0..343.0 | 1 | 0.080 | - | 343.0 | 343.0 | 0.00 (cpu-arm) | 0.080 (whole/whole) | - | 3562.9 | - | jaccard_vs_sklearn=1.000000, n_selected=110 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'k': 'half'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### select-f-classif / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/amd_ingest/results.txt (amd/a0010)`, ran on AMD Instinct MI325X (amd, DO) job a0010

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@1658ba796) | identical | 11.7 | 11.7..11.7 | 1 | - | - | 11.7 | - | - (stored whole) | - | - | - | - | jaccard_vs_sklearn=1.000000, n_selected=5 | - | main board, one scored run | - | ok (main@1658ba796 amd/a0010 2026-10-12; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 49.7 | 49.7..49.7 | 1 | 0.235 | - | 49.7 | 49.7 | 0.00 (cpu-arm) | 0.235 (whole/whole) | - | 383.4 | - | jaccard_vs_sklearn=1.000000, n_selected=5 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'k': 'half'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### select-f-regression / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/amd_ingest/results.txt (amd/a0005)`, ran on AMD Instinct MI325X (amd, DO) job a0005

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@a8fd6d926) | identical | 24.6 | 24.6..24.6 | 1 | - | - | 24.6 | - | - (stored whole) | - | - | - | - | jaccard_vs_sklearn=1.000000, n_selected=110 | - | main board, one scored run | - | ok (main@a8fd6d926 amd/a0005 2026-10-12; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 311.2 | 311.2..311.2 | 1 | 0.079 | - | 311.2 | 311.2 | 0.00 (cpu-arm) | 0.079 (whole/whole) | - | 2821.9 | - | jaccard_vs_sklearn=1.000000, n_selected=110 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'k': 'half'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### select-f-regression / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/amd_ingest/results.txt (amd/a0005)`, ran on AMD Instinct MI325X (amd, DO) job a0005

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@a8fd6d926) | identical | 8.2 | 8.2..8.2 | 1 | - | - | 8.2 | - | - (stored whole) | - | - | - | - | jaccard_vs_sklearn=1.000000, n_selected=5 | - | main board, one scored run | - | ok (main@a8fd6d926 amd/a0005 2026-10-12; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 34.7 | 34.7..34.7 | 1 | 0.235 | - | 34.7 | 34.7 | 0.00 (cpu-arm) | 0.235 (whole/whole) | - | 350.9 | - | jaccard_vs_sklearn=1.000000, n_selected=5 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'k': 'half'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### select-mutual-info-reg / istella (rows full, shape X 100000x220; Xq 100000x220; y 100000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/amd_ingest/results.txt (amd/a0005)`, ran on AMD Instinct MI325X (amd, DO) job a0005

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@a8fd6d926) | identical | 9394.1 | 9394.1..9394.1 | 1 | - | - | 9394.1 | - | - (stored whole) | - | - | - | - | jaccard_vs_sklearn=0.929825, n_selected=110 | - | main board, one scored run | - | ok (main@a8fd6d926 amd/a0005 2026-10-12; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 63140.1 | 63140.1..63140.1 | 1 | 0.149 | - | 63140.1 | 63140.1 | 0.00 (cpu-arm) | 0.149 (whole/whole) | - | 1891.0 | - | jaccard_vs_sklearn=1.000000, n_selected=110 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'k': 'half'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### select-mutual-info-reg / taxi (rows full, shape X 100000x11; Xq 100000x11; y 100000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/amd_ingest/results.txt (amd/a0005)`, ran on AMD Instinct MI325X (amd, DO) job a0005

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@a8fd6d926) | identical | 499.6 | 499.6..499.6 | 1 | - | - | 499.6 | - | - (stored whole) | - | - | - | - | jaccard_vs_sklearn=1.000000, n_selected=5 | - | main board, one scored run | - | ok (main@a8fd6d926 amd/a0005 2026-10-12; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 3513.7 | 3513.7..3513.7 | 1 | 0.142 | - | 3513.7 | 3513.7 | 0.00 (cpu-arm) | 0.142 (whole/whole) | - | 293.7 | - | jaccard_vs_sklearn=1.000000, n_selected=5 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'k': 'half'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### select-mutual-info / istella (rows full, shape X 100000x220; Xq 100000x220; y 100000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/amd_ingest/results.txt (amd/a0010)`, ran on AMD Instinct MI325X (amd, DO) job a0010

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@1658ba796) | identical | 458.3 | 458.3..458.3 | 1 | - | - | 458.3 | - | - (stored whole) | - | - | - | - | jaccard_vs_sklearn=0.929825, n_selected=110 | - | main board, one scored run | - | ok (main@1658ba796 amd/a0010 2026-10-12; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 41171.4 | 41171.4..41171.4 | 1 | 0.011 | - | 41171.4 | 41171.4 | 0.00 (cpu-arm) | 0.011 (whole/whole) | - | 1892.1 | - | jaccard_vs_sklearn=1.000000, n_selected=110 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'k': 'half'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### select-mutual-info / taxi (rows full, shape X 100000x11; Xq 100000x11; y 100000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/amd_ingest/results.txt (amd/a0010)`, ran on AMD Instinct MI325X (amd, DO) job a0010

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@1658ba796) | identical | 190.2 | 190.2..190.2 | 1 | - | - | 190.2 | - | - (stored whole) | - | - | - | - | jaccard_vs_sklearn=0.428571, n_selected=5 | - | main board, one scored run | - | ok (main@1658ba796 amd/a0010 2026-10-12; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 2040.1 | 2040.1..2040.1 | 1 | 0.093 | - | 2040.1 | 2040.1 | 0.00 (cpu-arm) | 0.093 (whole/whole) | - | 296.4 | - | jaccard_vs_sklearn=1.000000, n_selected=5 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'k': 'half'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### select-r-regression / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/amd_ingest/results.txt (amd/a0005)`, ran on AMD Instinct MI325X (amd, DO) job a0005

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@a8fd6d926) | identical | 22.3 | 22.3..22.3 | 1 | - | - | 22.3 | - | - (stored whole) | - | - | - | - | jaccard_vs_sklearn=1.000000, n_selected=110 | - | main board, one scored run | - | ok (main@a8fd6d926 amd/a0005 2026-10-12; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 307.4 | 307.4..307.4 | 1 | 0.073 | - | 307.4 | 307.4 | 0.00 (cpu-arm) | 0.073 (whole/whole) | - | 2821.2 | - | jaccard_vs_sklearn=1.000000, n_selected=110 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'k': 'half'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### select-r-regression / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/amd_ingest/results.txt (amd/a0005)`, ran on AMD Instinct MI325X (amd, DO) job a0005

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@a8fd6d926) | identical | 8.9 | 8.9..8.9 | 1 | - | - | 8.9 | - | - (stored whole) | - | - | - | - | jaccard_vs_sklearn=1.000000, n_selected=5 | - | main board, one scored run | - | ok (main@a8fd6d926 amd/a0005 2026-10-12; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 30.0 | 30.0..30.0 | 1 | 0.297 | - | 30.0 | 30.0 | 0.00 (cpu-arm) | 0.297 (whole/whole) | - | 350.1 | - | jaccard_vs_sklearn=1.000000, n_selected=5 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'k': 'half'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### sgd-clf / istella (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1065)`, ran on AMD Instinct MI325X (amd, DO) job a1065

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@8ed94710a) | identical | 3961.9 | 3961.9..3961.9 | 1 | - | - | 3961.9 | - | - (stored whole) | - | - | - | - | accuracy=0.920330 | - | main board, one scored run | - | ok (main@8ed94710a amd/a1065 2026-10-09; identity vs the nvidia columns: n/a) |

settings: {'l1_ratio': 0.15, 'fit_intercept': True, 'epsilon': 0.1, 'learning_rate': 'constant', 'eta0': 0.005, 'power_t': 0.5, 'early_stopping': False, 'n_iter_no_change': 5, 'average': False, 'penalty': 'l2', 'alpha': 0.0001, 'max_iter': 100, 'tol': None, 'shuffle': True, 'random_state': 7, 'batch_size': 4096, 'warm_start': False, 'loss': 'hinge', 'class_weight': None}. Rows: None. Timed: None.

mismatch: minibatch SGD of 4096 rows a step on ours and cuML MBSGD (batch_size pinned on both); scikit-learn's SGD is per-sample and is refused, not raced

mismatch: cuML reads epochs=100 and tol=0.0 (no early stop); ours max_iter=100 and tol=None

mismatch: intercept: cuML MBSGD centers X and y before the epochs (GLM preProcessData) and recovers the intercept; ours learns the intercept in the SGD steps

mismatch: opponents withheld: the lane settings at this tree's HEAD differ from the settings release-board-resume-r2 recorded for its opponent race (an opponent job must score them again)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### sgd-clf / taxi (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1065)`, ran on AMD Instinct MI325X (amd, DO) job a1065

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@8ed94710a) | identical | 2320.4 | 2320.4..2320.4 | 1 | - | - | 2320.4 | - | - (stored whole) | - | - | - | - | accuracy=0.755330 | - | main board, one scored run | - | ok (main@8ed94710a amd/a1065 2026-10-09; identity vs the nvidia columns: n/a) |

settings: {'l1_ratio': 0.15, 'fit_intercept': True, 'epsilon': 0.1, 'learning_rate': 'constant', 'eta0': 0.005, 'power_t': 0.5, 'early_stopping': False, 'n_iter_no_change': 5, 'average': False, 'penalty': 'l2', 'alpha': 0.0001, 'max_iter': 100, 'tol': None, 'shuffle': True, 'random_state': 7, 'batch_size': 4096, 'warm_start': False, 'loss': 'hinge', 'class_weight': None}. Rows: None. Timed: None.

mismatch: minibatch SGD of 4096 rows a step on ours and cuML MBSGD (batch_size pinned on both); scikit-learn's SGD is per-sample and is refused, not raced

mismatch: cuML reads epochs=100 and tol=0.0 (no early stop); ours max_iter=100 and tol=None

mismatch: intercept: cuML MBSGD centers X and y before the epochs (GLM preProcessData) and recovers the intercept; ours learns the intercept in the SGD steps

mismatch: opponents withheld: the lane settings at this tree's HEAD differ from the settings release-board-resume-r2 recorded for its opponent race (an opponent job must score them again)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### sgd / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/amd/campaign/measured/algos.sgd.synthetic.rows=full/board.json (amd/paired-3489211c0634fa16636de0de07b0cf61b71728bb4524f9425c18e7cbed6ef81c)`, ran on mojolearn-steward-do-amd

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 3.7 ms) = 2.872. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | 0.8.44 | identical | 10.5 | 10.5..10.5 | 1 | - | - | - | 10.5 | - (stored kernel) | - | - | 3032.5 | 235846.9 | optimizer_update_error_vs_torch_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | main board, one scored run | wheel | ok (main@79fcd24a5 amd/paired-3489211c0634fa16636de0de07b0cf61b71728bb4524f9425c18e7cbed6ef81c 2026-10-14; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 3.7 | 3.7..3.7 | 1 | 2.872 | - | - | 3.7 | - (stored kernel) | 2.872 (kernel/kernel) | - | 2711.9 | 832.0 | optimizer_update_error_vs_torch_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 70.7 | 70.7..70.7 | 1 | 0.148 | - | - | 70.7 | - (stored kernel) | 0.148 (kernel/kernel) | - | 3091.3 | 832.0 | optimizer_update_error_vs_torch_fp32=4.764e-06, rel_fro_vs_torch_eager_fp32=6.772e-10 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-fused-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 2.9 | 2.9..2.9 | 1 | 3.573 | - | - | 2.9 | - (stored kernel) | 3.573 (kernel/kernel) | - | 2699.1 | 832.0 | optimizer_update_error_vs_torch_fp32=4.764e-06, rel_fro_vs_torch_eager_fp32=6.772e-10 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/21024/vram_* bytes for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-fused-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'dampening': 0.0, 'lr': 0.001, 'maximize': False, 'momentum': 0.9, 'nesterov': False, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-fp32 | torch-eager-fp32 | torch-fused-fp32 |
|---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (optimizer.defaults) | torch (optimizer.defaults) | torch (optimizer.defaults) |
| dampening | 0.0 | 0.0 | 0.0 | 0.0 |
| differentiable | - | false | false | false |
| foreach | - | true | true | null |
| fused | - | null | null | true |
| input_arrays | [{"dtype": "float32", "name": "p0", "sha256": "485a7287793a9d7d33d710ea2820ea9c7519d31100db4bafeee813baa532dd7d", "shape": [16777216]}, {"dtype": "float32", "name": "grad0", "sha256": "e6c0159d56a2df748f38d3f3592321edc11067354f52e751e382384dca4178bc", "shape": [16777216]}, {"dtype": "float32", "name": "grad1", "sha256": "249bb61249ee25b0cd42e6bcaa012c27c565dde4ccd798448a60e935fd4ace7e", "shape": [16777216]}, {"dtype": "float32", "name": "grad2", "sha256": "4cf167fba649d3d82fdb6c7c41749a74e8fc8a9bd70491154f03bfc9272b2ed6", "shape": [16777216]}, {"dtype": "float32", "name": "grad3", "sha256": "594f963cfcef38863774b145fac84f56ae7f37190af1cf270e7cb23d3b86b587", "shape": [16777216]}, {"dtype": "float32", "name": "grad4", "sha256": "ee091b2ac84630d1d0328c337b12c5bb6bd6ba2b28f3350eab106bc848960f4e", "shape": [16777216]}, {"dtype": "float32", "name": "grad5", "sha256": "a042832a1a0884016cafe5aada2f1df6bf0b09422e9358536881a536fff70d02", "shape": [16777216]}, {"dtype": "float32", "name": "grad6", "sha256": "736aade529a9b43a7666f9d5a77bf9ef70c253bb5a26280143d941c915910763", "shape": [16777216]}, {"dtype": "float32", "name": "grad7", "sha256": "c4fdaf0915399dc3568c7fd4a49465f9e3c69a10d3e77b1fc15d926db244f302", "shape": [16777216]}, {"dtype": "float32", "name": "grad8", "sha256": "6b970f1bc919f22be37581526578d65a535e33dd886ff46cc64d95aca61fec12", "shape": [16777216]}, {"dtype": "float32", "name": "grad9", "sha256": "00adaa7225e175a85e28e59772e3afbb901e068e5ca8d7332adfa127f7fcbdd7", "shape": [16777216]}] | [{"dtype": "float32", "name": "p0", "sha256": "485a7287793a9d7d33d710ea2820ea9c7519d31100db4bafeee813baa532dd7d", "shape": [16777216]}, {"dtype": "float32", "name": "grad0", "sha256": "e6c0159d56a2df748f38d3f3592321edc11067354f52e751e382384dca4178bc", "shape": [16777216]}, {"dtype": "float32", "name": "grad1", "sha256": "249bb61249ee25b0cd42e6bcaa012c27c565dde4ccd798448a60e935fd4ace7e", "shape": [16777216]}, {"dtype": "float32", "name": "grad2", "sha256": "4cf167fba649d3d82fdb6c7c41749a74e8fc8a9bd70491154f03bfc9272b2ed6", "shape": [16777216]}, {"dtype": "float32", "name": "grad3", "sha256": "594f963cfcef38863774b145fac84f56ae7f37190af1cf270e7cb23d3b86b587", "shape": [16777216]}, {"dtype": "float32", "name": "grad4", "sha256": "ee091b2ac84630d1d0328c337b12c5bb6bd6ba2b28f3350eab106bc848960f4e", "shape": [16777216]}, {"dtype": "float32", "name": "grad5", "sha256": "a042832a1a0884016cafe5aada2f1df6bf0b09422e9358536881a536fff70d02", "shape": [16777216]}, {"dtype": "float32", "name": "grad6", "sha256": "736aade529a9b43a7666f9d5a77bf9ef70c253bb5a26280143d941c915910763", "shape": [16777216]}, {"dtype": "float32", "name": "grad7", "sha256": "c4fdaf0915399dc3568c7fd4a49465f9e3c69a10d3e77b1fc15d926db244f302", "shape": [16777216]}, {"dtype": "float32", "name": "grad8", "sha256": "6b970f1bc919f22be37581526578d65a535e33dd886ff46cc64d95aca61fec12", "shape": [16777216]}, {"dtype": "float32", "name": "grad9", "sha256": "00adaa7225e175a85e28e59772e3afbb901e068e5ca8d7332adfa127f7fcbdd7", "shape": [16777216]}] | [{"dtype": "float32", "name": "p0", "sha256": "485a7287793a9d7d33d710ea2820ea9c7519d31100db4bafeee813baa532dd7d", "shape": [16777216]}, {"dtype": "float32", "name": "grad0", "sha256": "e6c0159d56a2df748f38d3f3592321edc11067354f52e751e382384dca4178bc", "shape": [16777216]}, {"dtype": "float32", "name": "grad1", "sha256": "249bb61249ee25b0cd42e6bcaa012c27c565dde4ccd798448a60e935fd4ace7e", "shape": [16777216]}, {"dtype": "float32", "name": "grad2", "sha256": "4cf167fba649d3d82fdb6c7c41749a74e8fc8a9bd70491154f03bfc9272b2ed6", "shape": [16777216]}, {"dtype": "float32", "name": "grad3", "sha256": "594f963cfcef38863774b145fac84f56ae7f37190af1cf270e7cb23d3b86b587", "shape": [16777216]}, {"dtype": "float32", "name": "grad4", "sha256": "ee091b2ac84630d1d0328c337b12c5bb6bd6ba2b28f3350eab106bc848960f4e", "shape": [16777216]}, {"dtype": "float32", "name": "grad5", "sha256": "a042832a1a0884016cafe5aada2f1df6bf0b09422e9358536881a536fff70d02", "shape": [16777216]}, {"dtype": "float32", "name": "grad6", "sha256": "736aade529a9b43a7666f9d5a77bf9ef70c253bb5a26280143d941c915910763", "shape": [16777216]}, {"dtype": "float32", "name": "grad7", "sha256": "c4fdaf0915399dc3568c7fd4a49465f9e3c69a10d3e77b1fc15d926db244f302", "shape": [16777216]}, {"dtype": "float32", "name": "grad8", "sha256": "6b970f1bc919f22be37581526578d65a535e33dd886ff46cc64d95aca61fec12", "shape": [16777216]}, {"dtype": "float32", "name": "grad9", "sha256": "00adaa7225e175a85e28e59772e3afbb901e068e5ca8d7332adfa127f7fcbdd7", "shape": [16777216]}] | [{"dtype": "float32", "name": "p0", "sha256": "485a7287793a9d7d33d710ea2820ea9c7519d31100db4bafeee813baa532dd7d", "shape": [16777216]}, {"dtype": "float32", "name": "grad0", "sha256": "e6c0159d56a2df748f38d3f3592321edc11067354f52e751e382384dca4178bc", "shape": [16777216]}, {"dtype": "float32", "name": "grad1", "sha256": "249bb61249ee25b0cd42e6bcaa012c27c565dde4ccd798448a60e935fd4ace7e", "shape": [16777216]}, {"dtype": "float32", "name": "grad2", "sha256": "4cf167fba649d3d82fdb6c7c41749a74e8fc8a9bd70491154f03bfc9272b2ed6", "shape": [16777216]}, {"dtype": "float32", "name": "grad3", "sha256": "594f963cfcef38863774b145fac84f56ae7f37190af1cf270e7cb23d3b86b587", "shape": [16777216]}, {"dtype": "float32", "name": "grad4", "sha256": "ee091b2ac84630d1d0328c337b12c5bb6bd6ba2b28f3350eab106bc848960f4e", "shape": [16777216]}, {"dtype": "float32", "name": "grad5", "sha256": "a042832a1a0884016cafe5aada2f1df6bf0b09422e9358536881a536fff70d02", "shape": [16777216]}, {"dtype": "float32", "name": "grad6", "sha256": "736aade529a9b43a7666f9d5a77bf9ef70c253bb5a26280143d941c915910763", "shape": [16777216]}, {"dtype": "float32", "name": "grad7", "sha256": "c4fdaf0915399dc3568c7fd4a49465f9e3c69a10d3e77b1fc15d926db244f302", "shape": [16777216]}, {"dtype": "float32", "name": "grad8", "sha256": "6b970f1bc919f22be37581526578d65a535e33dd886ff46cc64d95aca61fec12", "shape": [16777216]}, {"dtype": "float32", "name": "grad9", "sha256": "00adaa7225e175a85e28e59772e3afbb901e068e5ca8d7332adfa127f7fcbdd7", "shape": [16777216]}] |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 |
| maximize | false | false | false | false |
| momentum | 0.9 | 0.9 | 0.9 | 0.9 |
| nesterov | false | false | false | false |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| weight_decay | 0.0 | 0.0 | 0.0 | 0.0 |

accepted difference: torch-eager-fp32 differentiable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-eager-fp32 foreach: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

accepted difference: torch-eager-fp32 fused: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

accepted difference: torch-compile-fp32 differentiable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-compile-fp32 foreach: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

accepted difference: torch-compile-fp32 fused: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

accepted difference: torch-fused-fp32 differentiable: no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode

accepted difference: torch-fused-fp32 foreach: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

accepted difference: torch-fused-fp32 fused: torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded

### sparse-coder / istella (rows full, shape X 5000x220; Xq 100000x220; y 5000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/amd_ingest/results.txt (amd/a0005)`, ran on AMD Instinct MI325X (amd, DO) job a0005

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@a8fd6d926) | identical | 0.6 | 0.6..0.6 | 1 | - | - | 0.6 | - | - (stored whole) | - | - | - | - | max_abs_diff_vs_sklearn=6.104e-05, output_shape=100000x64, rel_diff_vs_sklearn=9.356e-08 | - | main board, one scored run | - | ok (main@a8fd6d926 amd/a0005 2026-10-12; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 0.0 | 0.0..0.0 | 1 | 49.993 | - | 0.0 | 0.0 | 0.00 (cpu-arm) | 49.993 (whole/whole) | - | 1351.4 | - | output_shape=100000x64 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'positive_code': False, 'split_sign': False, 'transform_algorithm': 'omp', 'transform_alpha': None, 'transform_max_iter': 1000, 'transform_n_nonzero_coefs': 4}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### sparse-coder / taxi (rows full, shape X 5000x11; Xq 100000x11; y 5000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/amd_ingest/results.txt (amd/a0005)`, ran on AMD Instinct MI325X (amd, DO) job a0005

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@a8fd6d926) | identical | 0.5 | 0.5..0.5 | 1 | - | - | 0.5 | - | - (stored whole) | - | - | - | - | max_abs_diff_vs_sklearn=4.292e-06, output_shape=100000x64, rel_diff_vs_sklearn=8.246e-08 | - | main board, one scored run | - | ok (main@a8fd6d926 amd/a0005 2026-10-12; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 0.0 | 0.0..0.0 | 1 | 47.262 | - | 0.0 | 0.0 | 0.00 (cpu-arm) | 47.262 (whole/whole) | - | 384.8 | - | output_shape=100000x64 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'positive_code': False, 'split_sign': False, 'transform_algorithm': 'omp', 'transform_alpha': None, 'transform_max_iter': 1000, 'transform_n_nonzero_coefs': 4}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### spline / istella (rows full, shape X 1000000x16; Xq 100000x16; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/amd_ingest/results.txt (amd/a0005)`, ran on AMD Instinct MI325X (amd, DO) job a0005

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@a8fd6d926) | identical | 17.1 | 17.1..17.1 | 1 | - | - | 17.1 | - | - (stored whole) | - | - | - | - | max_abs_diff_vs_sklearn=1.788e-07, output_shape=100000x112, rel_diff_vs_sklearn=6.226e-08 | - | main board, one scored run | - | ok (main@a8fd6d926 amd/a0005 2026-10-12; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 43.3 | 43.3..43.3 | 1 | 0.395 | - | 43.3 | 43.3 | 0.00 (cpu-arm) | 0.395 (whole/whole) | - | 1375.6 | - | output_shape=100000x112 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'degree': 3, 'extrapolation': 'constant', 'include_bias': True, 'knots': 'uniform', 'n_knots': 5}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### spline / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/amd_ingest/results.txt (amd/a0005)`, ran on AMD Instinct MI325X (amd, DO) job a0005

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@a8fd6d926) | identical | 12.2 | 12.2..12.2 | 1 | - | - | 12.2 | - | - (stored whole) | - | - | - | - | max_abs_diff_vs_sklearn=1.788e-07, output_shape=100000x77, rel_diff_vs_sklearn=5.888e-08 | - | main board, one scored run | - | ok (main@a8fd6d926 amd/a0005 2026-10-12; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 41.5 | 41.5..41.5 | 1 | 0.294 | - | 41.5 | 41.5 | 0.00 (cpu-arm) | 0.294 (whole/whole) | - | 378.2 | - | output_shape=100000x77 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'degree': 3, 'extrapolation': 'constant', 'include_bias': True, 'knots': 'uniform', 'n_knots': 5}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### standard-scaler / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/amd_ingest/results.txt (amd/a0010)`, ran on AMD Instinct MI325X (amd, DO) job a0010

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@1658ba796) | identical | 24.6 | 24.6..24.6 | 1 | - | - | 24.6 | - | - (stored whole) | - | - | - | - | max_abs_diff_vs_sklearn=0.002426, output_shape=100000x220, rel_diff_vs_sklearn=3.59e-06 | - | main board, one scored run | - | ok (main@1658ba796 amd/a0010 2026-10-12; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 480.0 | 480.0..480.0 | 1 | 0.051 | - | 480.0 | 480.0 | 0.00 (cpu-arm) | 0.051 (whole/whole) | - | 3271.2 | - | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'with_mean': True, 'with_std': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### standard-scaler / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/amd_ingest/results.txt (amd/a0010)`, ran on AMD Instinct MI325X (amd, DO) job a0010

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@1658ba796) | identical | 4.0 | 4.0..4.0 | 1 | - | - | 4.0 | - | - (stored whole) | - | - | - | - | max_abs_diff_vs_sklearn=0.000103, output_shape=100000x11, rel_diff_vs_sklearn=3.846e-06 | - | main board, one scored run | - | ok (main@1658ba796 amd/a0010 2026-10-12; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 60.6 | 60.6..60.6 | 1 | 0.067 | - | 60.6 | 60.6 | 0.00 (cpu-arm) | 0.067 (whole/whole) | - | 365.6 | - | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'with_mean': True, 'with_std': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### svd / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/full-gpu-repair-20261014/amd/repair2/measured/algos.svd.istella.rows=full/board.json (amd/paired-8dda3f4099c39b1878274e19001fc0031af72c2abe92d5d3a41a2e031efd7d07)`, ran on mojolearn-steward-do-amd

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | 0.8.44 | identical | 710.2 | 710.2..710.2 | 1 | - | - | 710.2 | 693.4 | 16.80 (upload_ms_separate) | - | - | 6838.8 | 235497.7 | max_rel_singular_value_error=662.224259, relative_reconstruction_error_100k_rows=3.379e-05 | - | main board, one scored run | wheel | ok (main@d425c1417 amd/paired-8dda3f4099c39b1878274e19001fc0031af72c2abe92d5d3a41a2e031efd7d07 2026-10-14; identity vs the nvidia columns: n/a) |
| torch-gpu | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 2936.6 | 2936.6..2936.6 | 1 | 0.242 | - | - | 2936.6 | - (stored kernel) | 0.236 (kernel/kernel) | - | 5491.8 | 3360.7 | max_rel_singular_value_error=2.707e+08, relative_reconstruction_error_100k_rows=0.026534 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/98734/vram_* bytes for this pid at the round's end (not a peak)

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'full_matrices': False}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-gpu |
|---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) |
| full_matrices | false | false |
| input_arrays | [{"dtype": "float32", "sha256": "14c1e3963ef015c9b24cea93d1380679b59623b22edf61cb634001e62b4142a8", "shape": [1000000, 220]}] | [{"dtype": "float32", "sha256": "14c1e3963ef015c9b24cea93d1380679b59623b22edf61cb634001e62b4142a8", "shape": [1000000, 220]}] |
| seed | 7 | 7 |

### svd / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/full-gpu-repair-20261014/amd/repair2/measured/algos.svd.taxi.rows=full/board.json (amd/paired-8128e7ff29d9e15332fab30e5cb6258398a00290ffef9f5eee1800e7e002f18d)`, ran on mojolearn-steward-do-amd

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | 0.8.44 | identical | 16.4 | 16.4..16.4 | 1 | - | - | 16.4 | 15.6 | 0.85 (upload_ms_separate) | - | - | 2611.1 | 236215.6 | max_rel_singular_value_error=7.036e-07, relative_reconstruction_error_100k_rows=1.18e-06 | - | main board, one scored run | wheel | ok (main@d425c1417 amd/paired-8128e7ff29d9e15332fab30e5cb6258398a00290ffef9f5eee1800e7e002f18d 2026-10-14; identity vs the nvidia columns: n/a) |
| torch-gpu | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 272.1 | 272.1..272.1 | 1 | 0.060 | - | - | 272.1 | - (stored kernel) | 0.057 (kernel/kernel) | - | 2049.3 | 168.0 | max_rel_singular_value_error=4.409e-05, relative_reconstruction_error_100k_rows=0.003043 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU KFD /sys/class/kfd/kfd/proc/98162/vram_* bytes for this pid at the round's end (not a peak)

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'full_matrices': False}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-gpu |
|---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) |
| full_matrices | false | false |
| input_arrays | [{"dtype": "float32", "sha256": "33fdb983b45d51732ce911c8887c0c9f9cf7a7416ae67ebfe983fa0b3a4c8b77", "shape": [1000000, 11]}] | [{"dtype": "float32", "sha256": "33fdb983b45d51732ce911c8887c0c9f9cf7a7416ae67ebfe983fa0b3a4c8b77", "shape": [1000000, 11]}] |
| seed | 7 | 7 |

### svgp / istella (rows full, shape X 100000x220; Xq 20000x220; y 100000; yq 20000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1655)`, ran on AMD Instinct MI325X (amd, DO) job a1655

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c69768593) | identical | 620.3 | 620.3..620.3 | 1 | - | - | 620.3 | - | - (stored whole) | - | - | - | - | finite=True, r2=-0.106016, rmse=0.878373 | - | main board, one scored run | - | ok (main@c69768593 amd/a1655 2026-10-11; identity vs the nvidia columns: n/a) |
| gpytorch-gpu | gpytorch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | gpytorch 1.15.2 | opponent | 48.7 | 48.7..48.7 | 1 | 12.727 | - | - | 48.7 | - (stored kernel) | 12.727 (MIXED ours whole / arm kernel) | - | 4591.1 | 1611.8 | finite=True, r2=-0.106040, rmse=0.878383 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| gpytorch-cpu | gpytorch | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | gpytorch 1.15.2 | opponent | 672.0 | 672.0..672.0 | 1 | 0.923 | - | 672.0 | 672.0 | 0.00 (cpu-arm) | 0.923 (whole/whole) | - | 3457.1 | - | finite=True, r2=-0.106040, rmse=0.878383 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, gpytorch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, gpytorch-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'jitter': 1e-06, 'kernel_variance': 1.0, 'lengthscale': 1.0, 'n_inducing': 512, 'noise_variance': 1.0}. Rows: None. Timed: None.

mismatch: no seed on any arm: nothing is drawn (fixed inducing points, closed form)

mismatch: jitter: ours 1e-6 on K_uu; gpytorch adds its own Cholesky jitter (1e-6 in float32) only when a factorization fails

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### svgp / taxi (rows full, shape X 100000x11; Xq 20000x11; y 100000; yq 20000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1655)`, ran on AMD Instinct MI325X (amd, DO) job a1655

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c69768593) | identical | 715.9 | 715.9..715.9 | 1 | - | - | 715.9 | - | - (stored whole) | - | - | - | - | finite=True, r2=-0.194982, rmse=17.723482 | - | main board, one scored run | - | ok (main@c69768593 amd/a1655 2026-10-11; identity vs the nvidia columns: n/a) |
| gpytorch-gpu | gpytorch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | gpytorch 1.15.2 | opponent | 46.8 | 46.8..46.8 | 1 | 15.300 | - | - | 46.8 | - (stored kernel) | 15.300 (MIXED ours whole / arm kernel) | - | 3618.6 | 1514.9 | finite=True, r2=-0.209454, rmse=17.830481 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| gpytorch-cpu | gpytorch | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | gpytorch 1.15.2 | opponent | 504.6 | 504.6..504.6 | 1 | 1.419 | - | 504.6 | 504.6 | 0.00 (cpu-arm) | 1.419 (whole/whole) | - | 2461.6 | - | finite=True, r2=-0.209532, rmse=17.831057 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, gpytorch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, gpytorch-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'jitter': 1e-06, 'kernel_variance': 1.0, 'lengthscale': 1.0, 'n_inducing': 512, 'noise_variance': 1.0}. Rows: None. Timed: None.

mismatch: no seed on any arm: nothing is drawn (fixed inducing points, closed form)

mismatch: jitter: ours 1e-6 on K_uu; gpytorch adds its own Cholesky jitter (1e-6 in float32) only when a factorization fails

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### target-encoder / istella (rows full, shape X 1000000x8; Xq 100000x8; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/amd_ingest/results.txt (amd/a0010)`, ran on AMD Instinct MI325X (amd, DO) job a0010

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@1658ba796) | identical | 78.9 | 78.9..78.9 | 1 | - | - | 78.9 | - | - (stored whole) | - | - | - | - | max_abs_diff_vs_sklearn=1.074e-08, output_shape=100000x8, rel_diff_vs_sklearn=2.361e-08 | - | main board, one scored run | - | ok (main@1658ba796 amd/a0010 2026-10-12; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 202.8 | 202.8..202.8 | 1 | 0.389 | - | 202.8 | 202.8 | 0.00 (cpu-arm) | 0.389 (whole/whole) | - | 348.0 | - | output_shape=100000x8 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'categories': 'auto', 'cv': 4, 'random_state': 42, 'shuffle': True, 'smooth': 0.0, 'target_type': 'binary'}. Rows: None. Timed: None.

mismatch: fold assignment: cuML 'interleaved' (row i in fold i mod 4, the cuML benchmark's cuml_args); scikit-learn and ours a KFold shuffled by seed 42 (its cpu_args)

config: cuML benchmark (RAPIDS), TargetEncoder (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### target-encoder / taxi (rows full, shape X 1000000x5; Xq 100000x5; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/amd_ingest/results.txt (amd/a0010)`, ran on AMD Instinct MI325X (amd, DO) job a0010

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@1658ba796) | identical | 71.9 | 71.9..71.9 | 1 | - | - | 71.9 | - | - (stored whole) | - | - | - | - | max_abs_diff_vs_sklearn=2.965e-08, output_shape=100000x5, rel_diff_vs_sklearn=1.581e-08 | - | main board, one scored run | - | ok (main@1658ba796 amd/a0010 2026-10-12; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 116.5 | 116.5..116.5 | 1 | 0.617 | - | 116.5 | 116.5 | 0.00 (cpu-arm) | 0.617 (whole/whole) | - | 307.5 | - | output_shape=100000x5 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'categories': 'auto', 'cv': 4, 'random_state': 42, 'shuffle': True, 'smooth': 0.0, 'target_type': 'binary'}. Rows: None. Timed: None.

mismatch: fold assignment: cuML 'interleaved' (row i in fold i mod 4, the cuML benchmark's cuml_args); scikit-learn and ours a KFold shuffled by seed 42 (its cpu_args)

config: cuML benchmark (RAPIDS), TargetEncoder (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### theta / synthetic (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1679)`, ran on AMD Instinct MI325X (amd, DO) job a1679

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 122.3 | 122.3..122.3 | 1 | - | - | 122.3 | - | - (stored whole) | - | - | - | - | forecast_rmse=1.436610 | - | main board, one scored run | - | ok (main@a1c96a5ba amd/a1679 2026-10-11; identity vs the nvidia columns: n/a) |
| statsforecast-cpu | statsforecast | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | statsforecast 2.1.1 | opponent | 104.8 | 104.8..104.8 | 1 | 1.167 | - | 104.8 | 104.8 | 0.00 (cpu-arm) | 1.167 (whole/whole) | - | 283.6 | - | forecast_rmse=1.436557 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| statsmodels-cpu | statsmodels | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | statsmodels 0.15.0 | opponent | 38.6 | 38.6..38.6 | 1 | 3.169 | - | 38.6 | 38.6 | 0.00 (cpu-arm) | 3.169 (whole/whole) | - | 57.6 | - | forecast_rmse=1.434862 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, statsforecast-cpu, statsmodels-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'decomposition_type': 'multiplicative', 'season_length': 24}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### theta / taxi-hourly (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1679)`, ran on AMD Instinct MI325X (amd, DO) job a1679

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 390.7 | 390.7..390.7 | 1 | - | - | 390.7 | - | - (stored whole) | - | - | - | - | forecast_rmse=49.020604 | - | main board, one scored run | - | ok (main@a1c96a5ba amd/a1679 2026-10-11; identity vs the nvidia columns: n/a) |
| statsforecast-cpu | statsforecast | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | statsforecast 2.1.1 | opponent | 548.4 | 548.4..548.4 | 1 | 0.712 | - | 548.4 | 548.4 | 0.00 (cpu-arm) | 0.712 (whole/whole) | - | 283.4 | - | forecast_rmse=49.253901 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| statsmodels-cpu | statsmodels | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | statsmodels 0.15.0 | opponent | 52.3 | 52.3..52.3 | 1 | 7.471 | - | 52.3 | 52.3 | 0.00 (cpu-arm) | 7.471 (whole/whole) | - | 58.1 | - | forecast_rmse=49.311757 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, statsforecast-cpu, statsmodels-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'decomposition_type': 'multiplicative', 'season_length': 24}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### tree-shap / istella (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0877)`, ran on AMD Instinct MI325X (amd, DO) job a0877

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@42d1e42c6) | identical | 8.5 | 8.5..8.5 | 1 | - | - | 8.5 | - | - (stored whole) | - | - | - | - | max_additivity_error=1.175e-06 | - | main board, one scored run | - | ok (main@42d1e42c6 amd/a0877 2026-10-08; identity vs nvidia-l40s: MATCH) |

settings: {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'explained_model': "ours' RandomForestRegressor, covers = 100 background rows"}. Rows: None. Timed: None.

mismatch: opponents withheld: the lane settings at this tree's HEAD differ from the settings release-board-resume-r2 recorded for its opponent race (an opponent job must score them again)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### tree-shap / taxi (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0877)`, ran on AMD Instinct MI325X (amd, DO) job a0877

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@42d1e42c6) | identical | 2.8 | 2.8..2.8 | 1 | - | - | 2.8 | - | - (stored whole) | - | - | - | - | max_additivity_error=3.858e-05 | - | main board, one scored run | - | ok (main@42d1e42c6 amd/a0877 2026-10-08; identity vs nvidia-l40s: MATCH) |

settings: {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'explained_model': "ours' RandomForestRegressor, covers = 100 background rows"}. Rows: None. Timed: None.

mismatch: opponents withheld: the lane settings at this tree's HEAD differ from the settings release-board-resume-r2 recorded for its opponent race (an opponent job must score them again)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### tsne / istella (rows full, shape X 20000x220; Xq 2000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1710)`, ran on AMD Instinct MI325X (amd, DO) job a1710

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 1216.2 | 1216.2..1216.2 | 1 | - | - | 1216.2 | - | - (stored whole) | - | - | - | - | trustworthiness_k15=0.992124 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1710 2026-10-11; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 17896.2 | 17896.2..17896.2 | 1 | 0.068 | - | 17896.2 | 17896.2 | 0.00 (cpu-arm) | 0.068 (whole/whole) | - | 370.6 | - | trustworthiness_k15=0.992170 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'early_exaggeration': 12.0, 'init': 'seeded', 'learning_rate': 'auto', 'max_iter': 1000, 'n_components': 2, 'perplexity': 30.0, 'random_state': 7}. Rows: None. Timed: None.

mismatch: gradients: ours exact repulsion over k-NN affinities (no Barnes-Hut atomics under IDENTICAL); scikit-learn Barnes-Hut (angle=0.5, scikit-learn only); cuML FFT

mismatch: init: ours and scikit-learn start from the SAME array, ours' 'random' rule (default_rng(7).uniform(-5e-5, 5e-5, (n, 2)) float32); cuML takes only 'random' and draws its own start

mismatch: scikit-learn's early stop is switched off (n_iter_without_progress=1000, min_grad_norm=0.0): ours runs exactly max_iter steps

config: cuML benchmark (RAPIDS), TSNE (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### tsne / taxi (rows full, shape X 20000x11; Xq 2000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1711)`, ran on AMD Instinct MI325X (amd, DO) job a1711

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 1047.3 | 1047.3..1047.3 | 1 | - | - | 1047.3 | - | - (stored whole) | - | - | - | - | trustworthiness_k15=0.998921 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1711 2026-10-11; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 15382.8 | 15382.8..15382.8 | 1 | 0.068 | - | 15382.8 | 15382.8 | 0.00 (cpu-arm) | 0.068 (whole/whole) | - | 379.6 | - | trustworthiness_k15=0.998823 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'early_exaggeration': 12.0, 'init': 'seeded', 'learning_rate': 'auto', 'max_iter': 1000, 'n_components': 2, 'perplexity': 30.0, 'random_state': 7}. Rows: None. Timed: None.

mismatch: gradients: ours exact repulsion over k-NN affinities (no Barnes-Hut atomics under IDENTICAL); scikit-learn Barnes-Hut (angle=0.5, scikit-learn only); cuML FFT

mismatch: init: ours and scikit-learn start from the SAME array, ours' 'random' rule (default_rng(7).uniform(-5e-5, 5e-5, (n, 2)) float32); cuML takes only 'random' and draws its own start

mismatch: scikit-learn's early stop is switched off (n_iter_without_progress=1000, min_grad_norm=0.0): ours runs exactly max_iter steps

config: cuML benchmark (RAPIDS), TSNE (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### variance-threshold / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/amd_ingest/results.txt (amd/a0005)`, ran on AMD Instinct MI325X (amd, DO) job a0005

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@a8fd6d926) | identical | 18.8 | 18.8..18.8 | 1 | - | - | 18.8 | - | - (stored whole) | - | - | - | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x142, rel_diff_vs_sklearn=0.000000 | - | main board, one scored run | - | ok (main@a8fd6d926 amd/a0005 2026-10-12; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 707.2 | 707.2..707.2 | 1 | 0.027 | - | 707.2 | 707.2 | 0.00 (cpu-arm) | 0.027 (whole/whole) | - | 5070.5 | - | output_shape=100000x142 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'threshold': 0.01}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### variance-threshold / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/amd_ingest/results.txt (amd/a0005)`, ran on AMD Instinct MI325X (amd, DO) job a0005

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.41 (source build, main@a8fd6d926) | identical | 2.6 | 2.6..2.6 | 1 | - | - | 2.6 | - | - (stored whole) | - | - | - | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x10, rel_diff_vs_sklearn=0.000000 | - | main board, one scored run | - | ok (main@a8fd6d926 amd/a0005 2026-10-12; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 63.1 | 63.1..63.1 | 1 | 0.041 | - | 63.1 | 63.1 | 0.00 (cpu-arm) | 0.041 (whole/whole) | - | 458.5 | - | output_shape=100000x10 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'threshold': 0.01}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

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

