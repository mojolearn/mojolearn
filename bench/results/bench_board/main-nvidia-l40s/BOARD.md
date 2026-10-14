# mojolearn benchmark board

Generated 2026-10-14T07:44:23Z from `board.json` (schema `mojolearn-bench-board/1`).

> MAIN BOARD nvidia-l40s, version label main@dbc9b18c1. Unreleased: not reproducible by pip install; the release boards are the reference.

> Cells: 262 races; oldest cell main@89485bc96 (2026-10-08T13:27:06Z), newest cell main@dbc9b18c1 (2026-10-14T07:18:30Z). Boxes: NVIDIA L40S (nv, RunPod), NVIDIA L40S (nv2, RunPod).

> Rule: each lane x dataset shows the newest default-configuration race on main (highest commit date, then job number) whose status is ok. A newer ok cell replaces an older one whatever the two times are; a run that is not ok is never a numeric cell and never replaces an ok cell (FAILED table; an older ok cell stays, flagged with the newer failed run). 355 replaced or failed observations are in LEDGER.md. A/B and grid arms (MOJOLEARN_BUILD_DEFINES, MOJOLEARN_GRID_TAG grid runs) are never on this board.

> Ours: recorded scored runs from lq jobs or verified paired sessions; the status column names the cell's commit, box/job, commit date and the other vendor's digest at the same commit (identity: MATCH 85, n/a 177).

> Opponents: retained from each paired session when present; otherwise retained historical pairs or stored opponent boards (opponents-default-20261006, opponents-specific-20261006, release-board-resume-r2). Ingestion runs no measurements. `ours IDENTICAL / arm` divides the two stored medians, and the clock columns read a torch GPU arm kernel/kernel and every other arm whole/whole (AGENTS.md measurement item 6). Our kernel clock is `-` unless the cell recorded upload_ms_separate. Opponents withheld for changed lane settings: 43 races.

> Provenance: main@ identifies the source or harness freeze. Paired wheel runs separately record the installed library version and loaded binary hashes; they are not source builds of that harness commit.

> Neural lanes: the headline (its own table, and a line under each neural race) is ours IDENTICAL over torch's fastest bf16 arm, eager or compile, what customers run; the fp32 twin is the second column. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax. A cell of ours whose output hash equals a copied opponent's shows quality identical_to=<arm> (the same bits) where the own-host reference gave none.

## Identity

Same lane, dataset and commit on the other GPU vendor (identity = equal output digests on NVIDIA and AMD). Counts: MATCH 85, n/a 177.

DIFFER: none.

## FAILED

Runs on main whose status is not ok (error, refused, timeout, not_ready, NO-RECORD, NO-OURS-CELL). They are never a numeric cell and never replace an ok cell; an older ok cell stays on the board flagged with the failed run. 0 failed runs.


## Box

| field | value |
|---|---|
| vendor / API | nvidia / cuda |
| GPU | NVIDIA L40S |
| GPU driver | - |
| CPU | None (None logical cores) |
| memory bytes | - |
| OS | - |
| Python | - |
| mojolearn | main@dbc9b18c1 (wheel mixed source/wheel provenance; see each cell's version and measurement record, sha256 -) |
| script commit | dbc9b18c156d7ebcfbf94bdd7224f3b319eac6ab |
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

Races: 262 planned, 262 done, 0 failed, 0 unsupported, 0 pending. Cells: 819 (REFUSED 25, ok 794).

Inference cells: 137 (REFUSED 1, ok 136).

## Quality at a glance

Per lane and dataset: our FAST value, our IDENTICAL value, and each opponent's.

| family | lane | dataset | metric | ours FAST | ours IDENTICAL | opponents |
|---|---|---|---|---|---|---|
| algos | adafactor | synthetic | optimizer_update_error_vs_torch_fp32 | - | 3.444e-06 | torch-eager-fp32 0.000000; torch-compile-fp32 0.000000 |
| algos | adafactor | synthetic | rel_fro_vs_torch_eager_fp32 | - | 1.051e-08 | torch-eager-fp32 0.000000; torch-compile-fp32 0.000000 |
| algos | adagrad | synthetic | optimizer_update_error_vs_torch_fp32 | - | 1.874e-06 | torch-eager-fp32 0.000000; torch-compile-fp32 0.000000 |
| algos | adagrad | synthetic | rel_fro_vs_torch_eager_fp32 | - | 3.207e-09 | torch-eager-fp32 0.000000; torch-compile-fp32 0.000000 |
| algos | adam | synthetic | optimizer_update_error_vs_torch_fp32 | - | 8.844e-06 | torch-eager-fp32 0.000000; torch-compile-fp32 1.295e-05; torch-fused-fp32 5.957e-06 |
| algos | adam | synthetic | rel_fro_vs_torch_eager_fp32 | - | 3.34e-08 | torch-eager-fp32 0.000000; torch-compile-fp32 4.891e-08; torch-fused-fp32 2.25e-08 |
| algos | adamax | synthetic | optimizer_update_error_vs_torch_fp32 | - | 1.54e-06 | torch-eager-fp32 0.000000; torch-compile-fp32 2.376e-06 |
| algos | adamax | synthetic | rel_fro_vs_torch_eager_fp32 | - | 4.144e-09 | torch-eager-fp32 0.000000; torch-compile-fp32 6.395e-09 |
| algos | adamw | synthetic | optimizer_update_error_vs_torch_fp32 | - | 8.841e-06 | torch-eager-fp32 0.000000; torch-compile-fp32 5.831e-05; torch-fused-fp32 6.574e-05 |
| algos | adamw | synthetic | rel_fro_vs_torch_eager_fp32 | - | 3.34e-08 | torch-eager-fp32 0.000000; torch-compile-fp32 2.203e-07; torch-fused-fp32 2.484e-07 |
| algos | ard | istella | r2 (higher is better) | - | -0.124856 | sklearn-cpu 0.327402 |
| algos | ard | istella | rmse (lower is better) | - | 0.885949 | sklearn-cpu 0.685075 |
| algos | autoarima | synthetic | forecast_rmse (lower is better) | - | 3.260829 | cuml-gpu 32.552990; statsforecast-cpu 17.543262 |
| algos | autoarima | taxi-hourly | forecast_rmse (lower is better) | - | 73.388358 | cuml-gpu 106.817545; statsforecast-cpu 68.213954 |
| algos | avgpool1d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | 0.000000 | torch-eager-fp32 0.000000; torch-compile-fp32 0.000000; torch-eager-tf32 0.000000; torch-compile-tf32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | avgpool1d | synthetic | rel_fro_vs_torch_eager_fp32 | - | 0.000000 | torch-eager-fp32 0.000000; torch-compile-fp32 0.000000; torch-eager-tf32 0.000000; torch-compile-tf32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | avgpool2d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | 0.000000 | torch-eager-fp32 0.000000; torch-compile-fp32 0.000000; torch-eager-tf32 0.000000; torch-compile-tf32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | avgpool2d | synthetic | rel_fro_vs_torch_eager_fp32 | - | 0.000000 | torch-eager-fp32 0.000000; torch-compile-fp32 0.000000; torch-eager-tf32 0.000000; torch-compile-tf32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | batchnorm1d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | 0.002205 | torch-eager-fp32 0.000000; torch-compile-fp32 0.002792; torch-eager-tf32 0.000000; torch-compile-tf32 0.002792; torch-eager-bf16 0.000000; torch-compile-bf16 0.002792 |
| algos | batchnorm1d | synthetic | rel_fro_vs_torch_eager_fp32 | - | 8.002e-08 | torch-eager-fp32 0.000000; torch-compile-fp32 5.109e-08; torch-eager-tf32 0.000000; torch-compile-tf32 5.109e-08; torch-eager-bf16 0.000000; torch-compile-bf16 5.109e-08 |
| algos | batchnorm2d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | 0.001584 | torch-eager-fp32 0.000000; torch-compile-fp32 0.001863; torch-eager-tf32 0.000000; torch-compile-tf32 0.001863; torch-eager-bf16 0.000000; torch-compile-bf16 0.001863 |
| algos | batchnorm2d | synthetic | rel_fro_vs_torch_eager_fp32 | - | 1.314e-07 | torch-eager-fp32 0.000000; torch-compile-fp32 4.327e-08; torch-eager-tf32 0.000000; torch-compile-tf32 4.327e-08; torch-eager-bf16 0.000000; torch-compile-bf16 4.327e-08 |
| algos | bayesian-ridge | istella | r2 (higher is better) | - | 0.317909 | sklearn-cpu -395441.818968 |
| algos | bayesian-ridge | istella | rmse (lower is better) | - | 0.689893 | sklearn-cpu 525.293799 |
| algos | bernoulli-nb | istella | accuracy (higher is better) | - | 0.794050 | cuml-gpu 0.794050; sklearn-cpu 0.794050 |
| algos | bernoulli-nb | istella | logloss (lower is better) | - | 5.350625 | cuml-gpu 5.350631; sklearn-cpu 4.278741 |
| algos | bernoulli-nb | taxi | accuracy (higher is better) | - | 0.755560 | cuml-gpu 0.755560; sklearn-cpu 0.755560 |
| algos | bernoulli-nb | taxi | logloss (lower is better) | - | 0.557803 | cuml-gpu 0.557803; sklearn-cpu 0.557802 |
| algos | binarizer | istella | max_abs_diff_vs_sklearn | - | 0.000000 | cuml-gpu 0.000000 |
| algos | binarizer | istella | rel_diff_vs_sklearn | - | 0.000000 | cuml-gpu 0.000000 |
| algos | binarizer | taxi | max_abs_diff_vs_sklearn | - | 0.000000 | cuml-gpu 0.000000 |
| algos | binarizer | taxi | rel_diff_vs_sklearn | - | 0.000000 | cuml-gpu 0.000000 |
| algos | categorical-nb | istella | accuracy (higher is better) | - | 0.838850 | cuml-gpu 0.838850; sklearn-cpu 0.838850 |
| algos | categorical-nb | istella | logloss (lower is better) | - | 0.412625 | cuml-gpu 0.412625; sklearn-cpu 0.412625 |
| algos | categorical-nb | taxi | accuracy (higher is better) | - | 0.765850 | cuml-gpu 0.765850; sklearn-cpu 0.765850 |
| algos | categorical-nb | taxi | logloss (lower is better) | - | 0.538866 | cuml-gpu 0.538866; sklearn-cpu 0.538866 |
| algos | cholesky | synthetic | relative_residual | - | 2.9e-07 | torch-gpu 1.509e-07; cupy-gpu 1.365e-07 |
| algos | cnn-clf | synthetic | accuracy (higher is better) | - | 1.000000 | torch-eager-fp32 1.000000; torch-compile-fp32 1.000000; torch-eager-tf32 1.000000; torch-compile-tf32 1.000000; torch-eager-bf16 1.000000; torch-compile-bf16 1.000000 |
| algos | complement-nb | istella | accuracy (higher is better) | - | 0.849360 | cuml-gpu 0.849350; sklearn-cpu 0.849350 |
| algos | complement-nb | istella | logloss (lower is better) | - | 3.762524 | cuml-gpu 3.763060; sklearn-cpu 3.174763 |
| algos | complement-nb | taxi | accuracy (higher is better) | - | 0.678020 | cuml-gpu 0.678060; sklearn-cpu 0.678030 |
| algos | complement-nb | taxi | logloss (lower is better) | - | 0.715492 | cuml-gpu 0.715531; sklearn-cpu 0.715493 |
| algos | complement-nb | text | accuracy (higher is better) | - | 0.983067 | cuml-gpu 0.983067; sklearn-cpu 0.983067 |
| algos | complement-nb | text | logloss (lower is better) | - | 0.559491 | cuml-gpu 0.559490; sklearn-cpu 0.557285 |
| algos | connected-components | istella | n_components | - | 81 | - |
| algos | connected-components | taxi | n_components | - | 588 | - |
| algos | conv1d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | 0.268221 | torch-eager-fp32 0.000000; torch-compile-fp32 0.000000; torch-eager-tf32 0.000000; torch-compile-tf32 0.000000; torch-eager-bf16 3417.849541; torch-compile-bf16 3524.661064 |
| algos | conv1d | synthetic | rel_fro_vs_torch_eager_fp32 | - | 3.613e-07 | torch-eager-fp32 0.000000; torch-compile-fp32 0.000000; torch-eager-tf32 0.000000; torch-compile-tf32 0.000000; torch-eager-bf16 0.003296; torch-compile-bf16 0.003294 |
| algos | conv2d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | 0.248197 | torch-eager-fp32 0.000000; torch-compile-fp32 0.506768; torch-eager-tf32 382.931903; torch-compile-tf32 382.931903; torch-eager-bf16 3418.337554; torch-compile-bf16 3433.596343 |
| algos | conv2d | synthetic | rel_fro_vs_torch_eager_fp32 | - | 4.453e-07 | torch-eager-fp32 0.000000; torch-compile-fp32 5.007e-07; torch-eager-tf32 0.0003021; torch-compile-tf32 0.0003021; torch-eager-bf16 0.003382; torch-compile-bf16 0.003380 |
| algos | damped-ets | synthetic | forecast_rmse (lower is better) | - | 13.931153 | statsmodels-cpu 26.588738; statsforecast-cpu 13.945986 |
| algos | damped-ets | taxi-hourly | forecast_rmse (lower is better) | - | 96.690449 | statsmodels-cpu 196.955273; statsforecast-cpu 96.685568 |
| algos | dropout2d | synthetic | dropout_contract_relative_error | - | 5.422e-08 | torch-eager-fp32 5.422e-08; torch-compile-fp32 5.421e-08; torch-eager-tf32 5.422e-08; torch-compile-tf32 5.421e-08 |
| algos | dropout2d | synthetic | dropout_probability_z | - | 0.708333 | torch-eager-fp32 1.489583; torch-compile-fp32 1.791667; torch-eager-tf32 1.489583; torch-compile-tf32 1.791667 |
| algos | dropout2d | synthetic | dropped_channel_fraction | - | 0.096680 | torch-eager-fp32 0.093018; torch-compile-fp32 0.108398; torch-eager-tf32 0.093018; torch-compile-tf32 0.108398 |
| algos | dropout2d | synthetic | expected_dropout_probability | - | 0.100000 | torch-eager-fp32 0.100000; torch-compile-fp32 0.100000; torch-eager-tf32 0.100000; torch-compile-tf32 0.100000 |
| algos | dropout2d | synthetic | identifiable_channels | - | 4096 | torch-eager-fp32 4096; torch-compile-fp32 4096; torch-eager-tf32 4096; torch-compile-tf32 4096 |
| algos | dynamic-optimized-theta | synthetic | forecast_rmse (lower is better) | - | 1.436278 | statsforecast-cpu 1.436045 |
| algos | dynamic-optimized-theta | taxi-hourly | forecast_rmse (lower is better) | - | 49.086628 | statsforecast-cpu 49.314797 |
| algos | eigh | synthetic | max_eigenvalue_error | - | 3.692e-07 | torch-gpu 1.044e-06; cupy-gpu 1.044e-06; numpy-cpu 3.49e-08 |
| algos | eigh | synthetic | relative_residual | - | 6.672e-07 | torch-gpu 1.016e-06; cupy-gpu 1.016e-06; numpy-cpu 2.824e-08 |
| algos | elliptic-envelope | istella | fraction_flagged | - | 0.092480 | sklearn-cpu - |
| algos | elliptic-envelope | istella | jaccard_vs_sklearn | - | 0.062240 | sklearn-cpu - |
| algos | elliptic-envelope | taxi | fraction_flagged | - | 0.102090 | sklearn-cpu 0.102470 |
| algos | elliptic-envelope | taxi | jaccard_vs_sklearn | - | 0.925814 | sklearn-cpu 1.000000 |
| algos | embedding | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | 0.000000 | torch-eager-fp32 0.000000; torch-compile-fp32 0.000000 |
| algos | embedding | synthetic | rel_fro_vs_torch_eager_fp32 | - | 0.000000 | torch-eager-fp32 0.000000; torch-compile-fp32 0.000000 |
| algos | enet-cv | istella | r2 (higher is better) | - | 0.326805 | sklearn-cpu 0.326805 |
| algos | enet-cv | istella | rmse (lower is better) | - | 0.685379 | sklearn-cpu 0.685379 |
| algos | enet-cv | taxi | r2 (higher is better) | - | 0.909004 | sklearn-cpu 0.909004 |
| algos | enet-cv | taxi | rmse (lower is better) | - | 4.804486 | sklearn-cpu 4.804486 |
| algos | gaussian-nb | istella | accuracy (higher is better) | - | 0.876570 | cuml-gpu 0.876570; sklearn-cpu 0.876530 |
| algos | gaussian-nb | istella | logloss (lower is better) | - | 3.574420 | cuml-gpu 3.416741; sklearn-cpu 3.417392 |
| algos | gaussian-nb | taxi | accuracy (higher is better) | - | 0.719820 | cuml-gpu 0.719810; sklearn-cpu 0.719900 |
| algos | gaussian-nb | taxi | logloss (lower is better) | - | 1.132249 | cuml-gpu 1.132317; sklearn-cpu 1.133898 |
| algos | gaussian-rp | istella | mean_abs_distortion | - | 0.680693 | cuml-gpu 0.443920; sklearn-cpu 0.177966 |
| algos | gaussian-rp | taxi | mean_abs_distortion | - | 0.345752 | cuml-gpu 0.302259; sklearn-cpu 0.339791 |
| algos | gcn | istella | relative_error_vs_own_host | - | 0.000000 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-tf32 -; torch-compile-tf32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | gcn | istella | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.009646; torch-eager-tf32 281.122146; torch-compile-tf32 281.117866; torch-eager-bf16 2718.059111; torch-compile-bf16 2718.055850 |
| algos | gcn | istella | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 1.041e-07; torch-eager-tf32 0.0002661; torch-compile-tf32 0.0002661; torch-eager-bf16 0.002187; torch-compile-bf16 0.002187 |
| algos | gcn | taxi | relative_error_vs_own_host | - | 0.000000 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-tf32 -; torch-compile-tf32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | gcn | taxi | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.002327; torch-eager-tf32 0.001863; torch-compile-tf32 0.002794; torch-eager-bf16 1509.509282; torch-compile-bf16 1509.508234 |
| algos | gcn | taxi | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 9.803e-08; torch-eager-tf32 7.223e-08; torch-compile-tf32 9.801e-08; torch-eager-bf16 0.002330; torch-compile-bf16 0.002330 |
| algos | global-avgpool | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | 0.030258 | torch-eager-fp32 0.000000; torch-compile-fp32 0.009731; torch-eager-tf32 0.000000; torch-compile-tf32 0.009731; torch-eager-bf16 0.000000; torch-compile-bf16 0.009731 |
| algos | global-avgpool | synthetic | rel_fro_vs_torch_eager_fp32 | - | 1.512e-07 | torch-eager-fp32 0.000000; torch-compile-fp32 9.64e-08; torch-eager-tf32 0.000000; torch-compile-tf32 9.64e-08; torch-eager-bf16 0.000000; torch-compile-bf16 9.64e-08 |
| algos | global-maxpool | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | 0.000000 | torch-eager-fp32 0.000000; torch-compile-fp32 0.000000; torch-eager-tf32 0.000000; torch-compile-tf32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | global-maxpool | synthetic | rel_fro_vs_torch_eager_fp32 | - | 0.000000 | torch-eager-fp32 0.000000; torch-compile-fp32 0.000000; torch-eager-tf32 0.000000; torch-compile-tf32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | graphsage | istella | relative_error_vs_own_host | - | 0.000000 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-tf32 -; torch-compile-tf32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | graphsage | istella | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.134110; torch-eager-tf32 430.934131; torch-compile-tf32 430.934131; torch-eager-bf16 3907.114267; torch-compile-bf16 3678.210080 |
| algos | graphsage | istella | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 9.7e-08; torch-eager-tf32 0.0002969; torch-compile-tf32 0.0002969; torch-eager-bf16 0.003366; torch-compile-bf16 0.003054 |
| algos | graphsage | taxi | relative_error_vs_own_host | - | 0.000000 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-tf32 -; torch-compile-tf32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | graphsage | taxi | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.238419; torch-eager-tf32 0.029851; torch-compile-tf32 1475.334167; torch-eager-bf16 5460.333333; torch-compile-bf16 4608.154297 |
| algos | graphsage | taxi | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 6.954e-08; torch-eager-tf32 4.218e-08; torch-compile-tf32 0.0002145; torch-eager-bf16 0.003557; torch-compile-bf16 0.003285 |
| algos | gru-clf | synthetic | accuracy (higher is better) | - | 0.971625 | torch-eager-fp32 0.971842; torch-compile-fp32 0.971842; torch-eager-tf32 0.971842; torch-compile-tf32 0.971842; torch-eager-bf16 0.971951; torch-compile-bf16 0.971951 |
| algos | gru-clf | synthetic | logloss (lower is better) | - | 0.070054 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-tf32 -; torch-compile-tf32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | gru-clf | taxi-hourly | accuracy (higher is better) | - | 0.865723 | torch-eager-fp32 0.865668; torch-compile-fp32 0.865668; torch-eager-tf32 0.865668; torch-compile-tf32 0.865668; torch-eager-bf16 0.865668; torch-compile-bf16 0.865668 |
| algos | gru-clf | taxi-hourly | logloss (lower is better) | - | 0.303537 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-tf32 -; torch-compile-tf32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | gru-reg | synthetic | r2 (higher is better) | - | 0.982337 | torch-eager-fp32 0.981946; torch-compile-fp32 0.981946; torch-eager-tf32 0.981946; torch-compile-tf32 0.981946; torch-eager-bf16 0.981898; torch-compile-bf16 0.981898 |
| algos | gru-reg | synthetic | rmse (lower is better) | - | 0.153975 | torch-eager-fp32 0.155672; torch-compile-fp32 0.155672; torch-eager-tf32 0.155672; torch-compile-tf32 0.155672; torch-eager-bf16 0.155878; torch-compile-bf16 0.155878 |
| algos | gru-reg | taxi-hourly | r2 (higher is better) | - | 0.747875 | torch-eager-fp32 0.748219; torch-compile-fp32 0.748219; torch-eager-tf32 0.748219; torch-compile-tf32 0.748219; torch-eager-bf16 0.748325; torch-compile-bf16 0.748325 |
| algos | gru-reg | taxi-hourly | rmse (lower is better) | - | 0.544554 | torch-eager-fp32 0.544182; torch-compile-fp32 0.544182; torch-eager-tf32 0.544182; torch-compile-tf32 0.544182; torch-eager-bf16 0.544067; torch-compile-bf16 0.544067 |
| algos | incremental-pca | istella | explained_variance_fraction | - | 1.000000 | cuml-gpu 1.000000; sklearn-cpu 1.000000 |
| algos | incremental-pca | taxi | explained_variance_fraction | - | 0.999995 | cuml-gpu 0.999995; sklearn-cpu 0.999995 |
| algos | ivf-pq | istella | recall_at_10 (higher is better) | - | 0.802100 | cuvs-gpu 0.791300; faiss-cpu - |
| algos | ivf-pq | taxi | recall_at_10 (higher is better) | - | 0.981250 | cuvs-gpu 0.976875; faiss-cpu 0.979450 |
| algos | ivf-refine | istella | recall_at_10 (higher is better) | - | 0.994575 | cuvs-gpu 0.993625; faiss-cpu - |
| algos | ivf-refine | taxi | recall_at_10 (higher is better) | - | 0.999675 | cuvs-gpu 0.999050; faiss-cpu 0.999375 |
| algos | ivf-sq | taxi | recall_at_10 (higher is better) | - | 0.934975 | cuvs-gpu 0.772500; faiss-cpu 0.857050 |
| algos | jl-min-dim | synthetic | equal_fraction_vs_sklearn | - | 1.000000 | - |
| algos | kbins | istella | max_abs_diff_vs_sklearn | - | 0.000000 | cuml-gpu -; sklearn-cpu - |
| algos | kbins | istella | rel_diff_vs_sklearn | - | 0.000000 | cuml-gpu -; sklearn-cpu - |
| algos | kbins | taxi | max_abs_diff_vs_sklearn | - | 0.000000 | cuml-gpu -; sklearn-cpu - |
| algos | kbins | taxi | rel_diff_vs_sklearn | - | 0.000000 | cuml-gpu -; sklearn-cpu - |
| algos | kernel-pca | istella | rbf_feature_distance_relative_stress | - | 0.610112 | sklearn-cpu - |
| algos | kernel-pca | istella | subspace_cos_vs_sklearn | - | 1.000000 | sklearn-cpu 1.000000 |
| algos | kernel-pca | taxi | rbf_feature_distance_relative_stress | - | 0.545709 | sklearn-cpu - |
| algos | kernel-pca | taxi | subspace_cos_vs_sklearn | - | 1.000000 | sklearn-cpu 1.000000 |
| algos | kernel-shap | istella | rel_error_vs_exact | - | 8.307e-08 | cuml-gpu 0.032177; shap-cpu 8.574e-15 |
| algos | kernel-shap | taxi | rel_error_vs_exact | - | 1.056e-07 | - |
| algos | knn-imputer | istella | masked_rmse | - | 323953.237332 | sklearn-cpu 986208.700423 |
| algos | knn-imputer | taxi | masked_rmse | - | 6.151696 | sklearn-cpu 5.263919 |
| algos | label-binarizer | istella | max_abs_diff_vs_sklearn | - | 0.000000 | cuml-gpu 0.000000 |
| algos | label-binarizer | istella | rel_diff_vs_sklearn | - | 0.000000 | cuml-gpu 0.000000 |
| algos | label-binarizer | taxi | max_abs_diff_vs_sklearn | - | 0.000000 | cuml-gpu 0.000000 |
| algos | label-binarizer | taxi | rel_diff_vs_sklearn | - | 0.000000 | cuml-gpu 0.000000 |
| algos | label-encoder | istella | max_abs_diff_vs_sklearn | - | 0.000000 | cuml-gpu -; sklearn-cpu - |
| algos | label-encoder | istella | rel_diff_vs_sklearn | - | 0.000000 | cuml-gpu -; sklearn-cpu - |
| algos | label-encoder | taxi | max_abs_diff_vs_sklearn | - | 0.000000 | cuml-gpu -; sklearn-cpu - |
| algos | label-encoder | taxi | rel_diff_vs_sklearn | - | 0.000000 | cuml-gpu -; sklearn-cpu - |
| algos | lamb | synthetic | relative_error_vs_own_host | - | 0.000000 | - |
| algos | lars | istella | r2 (higher is better) | - | 0.328699 | cuml-gpu 0.328088; sklearn-cpu -4.245e+13 |
| algos | lars | istella | rmse (lower is better) | - | 0.684414 | cuml-gpu 0.684726; sklearn-cpu 5.442e+06 |
| algos | lars | taxi | r2 (higher is better) | - | 0.908983 | cuml-gpu 0.908983; sklearn-cpu 0.908983 |
| algos | lars | taxi | rmse (lower is better) | - | 4.805051 | cuml-gpu 4.805052; sklearn-cpu 4.805055 |
| algos | lasso-cv | istella | r2 (higher is better) | - | 0.325506 | sklearn-cpu 0.325507 |
| algos | lasso-cv | istella | rmse (lower is better) | - | 0.686040 | sklearn-cpu 0.686040 |
| algos | lasso-cv | taxi | r2 (higher is better) | - | 0.909038 | sklearn-cpu 0.909038 |
| algos | lasso-cv | taxi | rmse (lower is better) | - | 4.803593 | sklearn-cpu 4.803593 |
| algos | lasso-lars | istella | r2 (higher is better) | - | 0.310834 | sklearn-cpu 0.310816 |
| algos | lasso-lars | istella | rmse (lower is better) | - | 0.693461 | sklearn-cpu 0.693470 |
| algos | lasso-lars | taxi | r2 (higher is better) | - | 0.908998 | sklearn-cpu 0.908998 |
| algos | lasso-lars | taxi | rmse (lower is better) | - | 4.804663 | sklearn-cpu 4.804667 |
| algos | layernorm | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | 0.011127 | torch-eager-fp32 0.000000; torch-compile-fp32 0.006419; torch-eager-tf32 0.000000; torch-compile-tf32 0.006419; torch-eager-bf16 0.000000; torch-compile-bf16 0.006419 |
| algos | layernorm | synthetic | rel_fro_vs_torch_eager_fp32 | - | 7.007e-08 | torch-eager-fp32 0.000000; torch-compile-fp32 5.316e-08; torch-eager-tf32 0.000000; torch-compile-tf32 5.316e-08; torch-eager-bf16 0.000000; torch-compile-bf16 5.316e-08 |
| algos | lda-clf | istella | accuracy (higher is better) | - | 0.912830 | sklearn-cpu 0.899510 |
| algos | lda-clf | istella | logloss (lower is better) | - | 0.235875 | sklearn-cpu 0.438978 |
| algos | lda-clf | taxi | accuracy (higher is better) | - | 0.762530 | sklearn-cpu 0.762530 |
| algos | lda-clf | taxi | logloss (lower is better) | - | 0.539763 | sklearn-cpu 0.539767 |
| algos | lion | synthetic | relative_error_vs_own_host | - | 0.000000 | - |
| algos | louvain | istella | modularity | - | 0.911187 | - |
| algos | louvain | istella | n_communities | - | 40 | - |
| algos | louvain | taxi | modularity | - | 0.941953 | - |
| algos | louvain | taxi | n_communities | - | 58 | - |
| algos | lstm-clf | synthetic | accuracy (higher is better) | - | 0.967068 | torch-eager-fp32 0.968696; torch-compile-fp32 0.968696; torch-eager-tf32 0.968696; torch-compile-tf32 0.968696; torch-eager-bf16 0.968913; torch-compile-bf16 0.968913 |
| algos | lstm-clf | synthetic | logloss (lower is better) | - | 0.079720 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-tf32 -; torch-compile-tf32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | lstm-clf | taxi-hourly | accuracy (higher is better) | - | 0.870443 | torch-eager-fp32 0.868218; torch-compile-fp32 0.868218; torch-eager-tf32 0.868164; torch-compile-tf32 0.868164; torch-eager-bf16 0.868327; torch-compile-bf16 0.868327 |
| algos | lstm-clf | taxi-hourly | logloss (lower is better) | - | 0.297146 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-tf32 -; torch-compile-tf32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | lstm-reg | synthetic | r2 (higher is better) | - | 0.979398 | torch-eager-fp32 0.981013; torch-compile-fp32 0.981013; torch-eager-tf32 0.981013; torch-compile-tf32 0.981013; torch-eager-bf16 0.980998; torch-compile-bf16 0.980998 |
| algos | lstm-reg | synthetic | rmse (lower is better) | - | 0.166295 | torch-eager-fp32 0.159641; torch-compile-fp32 0.159641; torch-eager-tf32 0.159642; torch-compile-tf32 0.159642; torch-eager-bf16 0.159706; torch-compile-bf16 0.159706 |
| algos | lstm-reg | taxi-hourly | r2 (higher is better) | - | 0.754306 | torch-eager-fp32 0.751679; torch-compile-fp32 0.751679; torch-eager-tf32 0.751680; torch-compile-tf32 0.751680; torch-eager-bf16 0.751712; torch-compile-bf16 0.751712 |
| algos | lstm-reg | taxi-hourly | rmse (lower is better) | - | 0.537563 | torch-eager-fp32 0.540430; torch-compile-fp32 0.540430; torch-eager-tf32 0.540429; torch-compile-tf32 0.540429; torch-eager-bf16 0.540394; torch-compile-bf16 0.540394 |
| algos | lstsq | istella | relative_residual | - | 0.849956 | torch-gpu nan; cupy-gpu 0.849957; numpy-cpu 0.876581 |
| algos | lstsq | taxi | relative_residual | - | 0.756366 | torch-gpu 0.756366; cupy-gpu 0.756366; numpy-cpu 0.756366 |
| algos | lu-factor | synthetic | relative_residual | - | 3.256e-06 | torch-gpu 3.386e-07; cupy-gpu 3.386e-07 |
| algos | lu-solve | synthetic | relative_residual | - | 3.256e-06 | torch-gpu 3.386e-07; cupy-gpu 3.386e-07 |
| algos | maxabs-scaler | istella | max_abs_diff_vs_sklearn | - | 0.000000 | cuml-gpu 0.000000 |
| algos | maxabs-scaler | istella | rel_diff_vs_sklearn | - | 0.000000 | cuml-gpu 0.000000 |
| algos | maxabs-scaler | taxi | max_abs_diff_vs_sklearn | - | 0.000000 | cuml-gpu 0.000000 |
| algos | maxabs-scaler | taxi | rel_diff_vs_sklearn | - | 0.000000 | cuml-gpu 0.000000 |
| algos | maxpool1d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | 0.000000 | torch-eager-fp32 0.000000; torch-compile-fp32 0.000000; torch-eager-tf32 0.000000; torch-compile-tf32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | maxpool1d | synthetic | rel_fro_vs_torch_eager_fp32 | - | 0.000000 | torch-eager-fp32 0.000000; torch-compile-fp32 0.000000; torch-eager-tf32 0.000000; torch-compile-tf32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | maxpool2d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | 0.000000 | torch-eager-fp32 0.000000; torch-compile-fp32 0.000000; torch-eager-tf32 0.000000; torch-compile-tf32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | maxpool2d | synthetic | rel_fro_vs_torch_eager_fp32 | - | 0.000000 | torch-eager-fp32 0.000000; torch-compile-fp32 0.000000; torch-eager-tf32 0.000000; torch-compile-tf32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
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
| algos | minmax-scaler | istella | max_abs_diff_vs_sklearn | - | 0.000000 | cuml-gpu 0.000000 |
| algos | minmax-scaler | istella | rel_diff_vs_sklearn | - | 0.000000 | cuml-gpu 0.000000 |
| algos | minmax-scaler | taxi | max_abs_diff_vs_sklearn | - | 0.000000 | cuml-gpu 0.000000 |
| algos | minmax-scaler | taxi | rel_diff_vs_sklearn | - | 0.000000 | cuml-gpu 0.000000 |
| algos | moe | synthetic | relative_error_vs_own_host | - | 0.000000 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-tf32 -; torch-compile-tf32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | moe | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.026193; torch-eager-tf32 1004.691291; torch-compile-tf32 1004.691291; torch-eager-bf16 22834.612745; torch-compile-bf16 22851.987745 |
| algos | moe | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 1.548e-07; torch-eager-tf32 0.017534; torch-compile-tf32 0.017534; torch-eager-bf16 0.055222; torch-compile-bf16 0.055199 |
| algos | multilabel-binarizer | istella | max_abs_diff_vs_sklearn | - | 0.000000 | sklearn-cpu - |
| algos | multilabel-binarizer | istella | rel_diff_vs_sklearn | - | 0.000000 | sklearn-cpu - |
| algos | multilabel-binarizer | taxi | max_abs_diff_vs_sklearn | - | 0.000000 | sklearn-cpu - |
| algos | multilabel-binarizer | taxi | rel_diff_vs_sklearn | - | 0.000000 | sklearn-cpu - |
| algos | multinomial-nb | istella | accuracy (higher is better) | - | 0.853620 | cuml-gpu 0.853620; sklearn-cpu 0.853620 |
| algos | multinomial-nb | istella | logloss (lower is better) | - | 3.628565 | cuml-gpu 3.628599; sklearn-cpu 3.087499 |
| algos | multinomial-nb | taxi | accuracy (higher is better) | - | 0.723160 | cuml-gpu 0.723160; sklearn-cpu 0.723160 |
| algos | multinomial-nb | taxi | logloss (lower is better) | - | 0.590725 | cuml-gpu 0.590750; sklearn-cpu 0.590725 |
| algos | multinomial-nb | text | accuracy (higher is better) | - | 0.983067 | cuml-gpu 0.983067; sklearn-cpu 0.983067 |
| algos | multinomial-nb | text | logloss (lower is better) | - | 0.559529 | cuml-gpu 0.559524; sklearn-cpu 0.557319 |
| algos | nadam | synthetic | optimizer_update_error_vs_torch_fp32 | - | 1.131e-05 | torch-eager-fp32 0.000000; torch-compile-fp32 5.752e-05 |
| algos | nadam | synthetic | rel_fro_vs_torch_eager_fp32 | - | 2.945e-08 | torch-eager-fp32 0.000000; torch-compile-fp32 1.498e-07 |
| algos | nearest-centroid | istella | accuracy (higher is better) | - | 0.852610 | sklearn-cpu 0.852610 |
| algos | nearest-centroid | istella | logloss (lower is better) | - | 4.299485 | sklearn-cpu 4.117692 |
| algos | nearest-centroid | taxi | accuracy (higher is better) | - | 0.666700 | sklearn-cpu 0.666750 |
| algos | nearest-centroid | taxi | logloss (lower is better) | - | 0.781905 | sklearn-cpu 0.781690 |
| algos | normalizer | istella | max_abs_diff_vs_sklearn | - | 3.576e-07 | cuml-gpu 1.788e-07 |
| algos | normalizer | istella | rel_diff_vs_sklearn | - | 5.647e-08 | cuml-gpu 4.428e-08 |
| algos | normalizer | taxi | max_abs_diff_vs_sklearn | - | 1.192e-07 | cuml-gpu 1.788e-07 |
| algos | normalizer | taxi | rel_diff_vs_sklearn | - | 3.043e-08 | cuml-gpu 2.246e-08 |
| algos | onehot | istella | max_abs_diff_vs_sklearn | - | 0.000000 | cuml-gpu -; sklearn-cpu - |
| algos | onehot | istella | rel_diff_vs_sklearn | - | 0.000000 | cuml-gpu -; sklearn-cpu - |
| algos | onehot | taxi | max_abs_diff_vs_sklearn | - | 0.000000 | cuml-gpu -; sklearn-cpu - |
| algos | onehot | taxi | rel_diff_vs_sklearn | - | 0.000000 | cuml-gpu -; sklearn-cpu - |
| algos | optimized-theta | synthetic | forecast_rmse (lower is better) | - | 1.438855 | statsforecast-cpu 1.437815 |
| algos | optimized-theta | taxi-hourly | forecast_rmse (lower is better) | - | 49.150860 | statsforecast-cpu 49.356608 |
| algos | ordinal | istella | max_abs_diff_vs_sklearn | - | 0.000000 | sklearn-cpu - |
| algos | ordinal | istella | rel_diff_vs_sklearn | - | 0.000000 | sklearn-cpu - |
| algos | ordinal | taxi | max_abs_diff_vs_sklearn | - | 0.000000 | sklearn-cpu - |
| algos | ordinal | taxi | rel_diff_vs_sklearn | - | 0.000000 | sklearn-cpu - |
| algos | pagerank | istella | sum | - | 1.000000 | cugraph-gpu 1.000000; networkx-cpu 1.000000 |
| algos | pagerank | taxi | sum | - | 1.000000 | cugraph-gpu 1.000000; networkx-cpu 1.000000 |
| algos | permutation-shap | istella | rel_error_vs_exact | - | 6.68e-08 | cuml-gpu 1.483e-07; shap-cpu 3.692e-10 |
| algos | permutation-shap | taxi | rel_error_vs_exact | - | 1.054e-07 | cuml-gpu 1.701e-07; shap-cpu 1.144e-15 |
| algos | poly-features | istella | max_abs_diff_vs_sklearn | - | 0.000000 | cuml-gpu 0.000000 |
| algos | poly-features | istella | rel_diff_vs_sklearn | - | 0.000000 | cuml-gpu 0.000000 |
| algos | poly-features | taxi | max_abs_diff_vs_sklearn | - | 0.000000 | cuml-gpu 0.000000 |
| algos | poly-features | taxi | rel_diff_vs_sklearn | - | 0.000000 | cuml-gpu 0.000000 |
| algos | power-transformer | istella | max_abs_diff_vs_sklearn | - | 5.575023 | cuml-gpu - |
| algos | power-transformer | istella | rel_diff_vs_sklearn | - | 0.105054 | cuml-gpu - |
| algos | power-transformer | taxi | max_abs_diff_vs_sklearn | - | 0.064570 | cuml-gpu 0.063349 |
| algos | power-transformer | taxi | rel_diff_vs_sklearn | - | 0.0008076 | cuml-gpu 0.0003244 |
| algos | qr | istella | relative_gram_difference | - | 1.485e-07 | torch-gpu 3.787e-07; cupy-gpu 3.787e-07; numpy-cpu 2.459e-08 |
| algos | qr | taxi | relative_gram_difference | - | 1.714e-07 | torch-gpu 5.971e-06; cupy-gpu 5.971e-06; numpy-cpu 3.024e-08 |
| algos | quantile-transformer | istella | max_abs_diff_vs_sklearn | - | 5.96e-08 | cuml-gpu 0.000000 |
| algos | quantile-transformer | istella | rel_diff_vs_sklearn | - | 2.764e-08 | cuml-gpu 0.000000 |
| algos | quantile-transformer | taxi | max_abs_diff_vs_sklearn | - | 5.96e-08 | cuml-gpu 0.000000 |
| algos | quantile-transformer | taxi | rel_diff_vs_sklearn | - | 2.294e-08 | cuml-gpu 0.000000 |
| algos | quantile | istella | r2 (higher is better) | - | -0.039998 | sklearn-cpu -0.044780 |
| algos | quantile | istella | rmse (lower is better) | - | 0.851877 | sklearn-cpu 0.853833 |
| algos | quantile | taxi | r2 (higher is better) | - | 0.899596 | sklearn-cpu 0.899678 |
| algos | quantile | taxi | rmse (lower is better) | - | 5.046749 | sklearn-cpu 5.044706 |
| algos | randomized-svd | istella | relative_reconstruction_error (lower is better) | - | 0.0002359 | torch-gpu 0.0002359; sklearn-cpu 0.0002359 |
| algos | randomized-svd | taxi | relative_reconstruction_error (lower is better) | - | 0.027197 | torch-gpu 0.027197; sklearn-cpu 0.027197 |
| algos | resnet-block | synthetic | relative_error_vs_own_host | - | 0.000000 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-tf32 -; torch-compile-tf32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | resnet-block | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.818182; torch-eager-tf32 1532.793045; torch-compile-tf32 1532.793045; torch-eager-bf16 17838.627100; torch-compile-bf16 18497.318029 |
| algos | resnet-block | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 5.381e-07; torch-eager-tf32 0.0003478; torch-compile-tf32 0.0003478; torch-eager-bf16 0.003621; torch-compile-bf16 0.003425 |
| algos | rfe | istella | jaccard_vs_exact | - | 0.774194 | sklearn-cpu - |
| algos | rfe | istella | jaccard_vs_sklearn | - | 0.880342 | sklearn-cpu 1.000000 |
| algos | rfe | istella | n_selected | - | 110 | sklearn-cpu 110 |
| algos | rfe | taxi | jaccard_vs_exact | - | 1.000000 | sklearn-cpu - |
| algos | rfe | taxi | jaccard_vs_sklearn | - | 1.000000 | sklearn-cpu 1.000000 |
| algos | rfe | taxi | n_selected | - | 5 | sklearn-cpu 5 |
| algos | ridge-cv | istella | r2 (higher is better) | - | 0.328684 | sklearn-cpu 0.328683 |
| algos | ridge-cv | istella | rmse (lower is better) | - | 0.684422 | sklearn-cpu 0.684423 |
| algos | ridge-cv | taxi | r2 (higher is better) | - | 0.908981 | sklearn-cpu 0.908983 |
| algos | ridge-cv | taxi | rmse (lower is better) | - | 4.805109 | sklearn-cpu 4.805055 |
| algos | rmsprop | synthetic | optimizer_update_error_vs_torch_fp32 | - | 2.228e-06 | torch-eager-fp32 0.000000; torch-compile-fp32 0.000000 |
| algos | rmsprop | synthetic | rel_fro_vs_torch_eager_fp32 | - | 3.829e-08 | torch-eager-fp32 0.000000; torch-compile-fp32 0.000000 |
| algos | rnn-clf | synthetic | accuracy (higher is better) | - | 0.967828 | torch-eager-fp32 0.953559; torch-compile-fp32 0.953559; torch-eager-tf32 0.953505; torch-compile-tf32 0.953505; torch-eager-bf16 0.953559; torch-compile-bf16 0.953559 |
| algos | rnn-clf | synthetic | logloss (lower is better) | - | 0.079029 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-tf32 -; torch-compile-tf32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | rnn-clf | taxi-hourly | accuracy (higher is better) | - | 0.862684 | torch-eager-fp32 0.868056; torch-compile-fp32 0.868056; torch-eager-tf32 0.868056; torch-compile-tf32 0.868056; torch-eager-bf16 0.868001; torch-compile-bf16 0.868001 |
| algos | rnn-clf | taxi-hourly | logloss (lower is better) | - | 0.313008 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-tf32 -; torch-compile-tf32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | rnn-reg | synthetic | r2 (higher is better) | - | 0.978602 | torch-eager-fp32 0.977347; torch-compile-fp32 0.977347; torch-eager-tf32 0.977347; torch-compile-tf32 0.977347; torch-eager-bf16 0.977345; torch-compile-bf16 0.977345 |
| algos | rnn-reg | synthetic | rmse (lower is better) | - | 0.169476 | torch-eager-fp32 0.174374; torch-compile-fp32 0.174374; torch-eager-tf32 0.174374; torch-compile-tf32 0.174374; torch-eager-bf16 0.174385; torch-compile-bf16 0.174385 |
| algos | rnn-reg | taxi-hourly | r2 (higher is better) | - | 0.743904 | torch-eager-fp32 0.738796; torch-compile-fp32 0.738796; torch-eager-tf32 0.738800; torch-compile-tf32 0.738800; torch-eager-bf16 0.738906; torch-compile-bf16 0.738906 |
| algos | rnn-reg | taxi-hourly | rmse (lower is better) | - | 0.548825 | torch-eager-fp32 0.554271; torch-compile-fp32 0.554271; torch-eager-tf32 0.554267; torch-compile-tf32 0.554267; torch-eager-bf16 0.554155; torch-compile-bf16 0.554155 |
| algos | robust-scaler | istella | max_abs_diff_vs_sklearn | - | 0.0001221 | cuml-gpu -; sklearn-cpu - |
| algos | robust-scaler | istella | rel_diff_vs_sklearn | - | 7.339e-09 | cuml-gpu -; sklearn-cpu - |
| algos | robust-scaler | taxi | max_abs_diff_vs_sklearn | - | 0.000000 | cuml-gpu -; sklearn-cpu - |
| algos | robust-scaler | taxi | rel_diff_vs_sklearn | - | 0.000000 | cuml-gpu -; sklearn-cpu - |
| algos | select-chi2 | istella | jaccard_vs_sklearn | - | 1.000000 | sklearn-cpu 1.000000 |
| algos | select-chi2 | istella | n_selected | - | 110 | sklearn-cpu 110 |
| algos | select-chi2 | taxi | jaccard_vs_sklearn | - | 1.000000 | sklearn-cpu 1.000000 |
| algos | select-chi2 | taxi | n_selected | - | 5 | sklearn-cpu 5 |
| algos | select-d | synthetic | relative_error_vs_own_host | - | 0.000000 | - |
| algos | select-d | taxi-hourly | relative_error_vs_own_host | - | 0.000000 | - |
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
| algos | sgd-reg | istella | r2 (higher is better) | - | 0.327829 | - |
| algos | sgd-reg | istella | rmse (lower is better) | - | 0.684858 | - |
| algos | sgd-reg | taxi | r2 (higher is better) | - | 0.908969 | - |
| algos | sgd-reg | taxi | rmse (lower is better) | - | 4.805411 | - |
| algos | sgd | synthetic | optimizer_update_error_vs_torch_fp32 | - | 0.000000 | torch-eager-fp32 0.000000; torch-compile-fp32 4.764e-06; torch-fused-fp32 0.000000 |
| algos | sgd | synthetic | rel_fro_vs_torch_eager_fp32 | - | 0.000000 | torch-eager-fp32 0.000000; torch-compile-fp32 6.772e-10; torch-fused-fp32 0.000000 |
| algos | simple-imputer | istella | masked_rmse | - | 346849.129968 | cuml-gpu 346849.129968; sklearn-cpu 346849.129968 |
| algos | simple-imputer | taxi | masked_rmse | - | 5.985180 | cuml-gpu 5.985180; sklearn-cpu 5.985180 |
| algos | sparse-coder | istella | max_abs_diff_vs_sklearn | - | 6.104e-05 | sklearn-cpu - |
| algos | sparse-coder | istella | rel_diff_vs_sklearn | - | 9.356e-08 | sklearn-cpu - |
| algos | sparse-coder | taxi | max_abs_diff_vs_sklearn | - | 4.292e-06 | sklearn-cpu - |
| algos | sparse-coder | taxi | rel_diff_vs_sklearn | - | 8.246e-08 | sklearn-cpu - |
| algos | sparse-rp | istella | mean_abs_distortion | - | 1.883381 | cuml-gpu 0.876709; sklearn-cpu 0.474347 |
| algos | sparse-rp | taxi | mean_abs_distortion | - | 0.147163 | cuml-gpu 0.266849; sklearn-cpu 0.381016 |
| algos | spline | istella | max_abs_diff_vs_sklearn | - | 1.788e-07 | sklearn-cpu - |
| algos | spline | istella | rel_diff_vs_sklearn | - | 6.226e-08 | sklearn-cpu - |
| algos | spline | taxi | max_abs_diff_vs_sklearn | - | 1.788e-07 | sklearn-cpu - |
| algos | spline | taxi | rel_diff_vs_sklearn | - | 5.888e-08 | sklearn-cpu - |
| algos | standard-scaler | istella | max_abs_diff_vs_sklearn | - | 0.002426 | cuml-gpu 0.022049 |
| algos | standard-scaler | istella | rel_diff_vs_sklearn | - | 3.602e-06 | cuml-gpu 0.0001523 |
| algos | standard-scaler | taxi | max_abs_diff_vs_sklearn | - | 0.000103 | cuml-gpu 0.0003331 |
| algos | standard-scaler | taxi | rel_diff_vs_sklearn | - | 3.854e-06 | cuml-gpu 3.216e-05 |
| algos | svd | istella | max_rel_singular_value_error | - | 662.224271 | torch-gpu 8.068917; cupy-gpu 10270.720886; numpy-cpu 1.000000 |
| algos | svd | istella | relative_reconstruction_error_100k_rows | - | 3.379e-05 | torch-gpu 2.861e-05; cupy-gpu 2.385e-06; numpy-cpu 4.1e-08 |
| algos | svd | taxi | max_rel_singular_value_error | - | 7.036e-07 | torch-gpu 2.595e-06; cupy-gpu 3.763e-06; numpy-cpu 4.308e-08 |
| algos | svd | taxi | relative_reconstruction_error_100k_rows | - | 1.18e-06 | torch-gpu 7.244e-06; cupy-gpu 6.886e-06; numpy-cpu 4.314e-08 |
| algos | svgp | istella | r2 (higher is better) | - | -0.106016 | gpytorch-gpu -0.106040; gpytorch-cpu -0.106040 |
| algos | svgp | istella | rmse (lower is better) | - | 0.878373 | gpytorch-gpu 0.878383; gpytorch-cpu 0.878383 |
| algos | svgp | taxi | r2 (higher is better) | - | -0.194982 | gpytorch-gpu -0.209527; gpytorch-cpu -0.209546 |
| algos | svgp | taxi | rmse (lower is better) | - | 17.723482 | gpytorch-gpu 17.831018; gpytorch-cpu 17.831162 |
| algos | target-encoder | istella | max_abs_diff_vs_sklearn | - | 1.074e-08 | cuml-gpu -; sklearn-cpu - |
| algos | target-encoder | istella | rel_diff_vs_sklearn | - | 2.361e-08 | cuml-gpu -; sklearn-cpu - |
| algos | target-encoder | taxi | max_abs_diff_vs_sklearn | - | 2.965e-08 | cuml-gpu -; sklearn-cpu - |
| algos | target-encoder | taxi | rel_diff_vs_sklearn | - | 1.581e-08 | cuml-gpu -; sklearn-cpu - |
| algos | theta | synthetic | forecast_rmse (lower is better) | - | 1.436610 | statsforecast-cpu 1.436557; statsmodels-cpu 1.434862 |
| algos | theta | taxi-hourly | forecast_rmse (lower is better) | - | 49.020604 | statsforecast-cpu 49.253901; statsmodels-cpu 49.311757 |
| algos | tree-shap | istella | max_additivity_error | - | 1.175e-06 | - |
| algos | tree-shap | taxi | max_additivity_error | - | 3.858e-05 | - |
| algos | tsne | istella | trustworthiness_k15 (higher is better, 1 at most) | - | 0.992124 | cuml-gpu 0.990033; sklearn-cpu 0.992182 |
| algos | tsne | taxi | trustworthiness_k15 (higher is better, 1 at most) | - | 0.998921 | cuml-gpu 0.998353; sklearn-cpu 0.998823 |
| algos | variance-threshold | istella | max_abs_diff_vs_sklearn | - | 0.000000 | sklearn-cpu - |
| algos | variance-threshold | istella | rel_diff_vs_sklearn | - | 0.000000 | sklearn-cpu - |
| algos | variance-threshold | taxi | max_abs_diff_vs_sklearn | - | 0.000000 | sklearn-cpu - |
| algos | variance-threshold | taxi | rel_diff_vs_sklearn | - | 0.000000 | sklearn-cpu - |
| classical | dbscan | istella | n_clusters | - | 40131 | cuml-gpu 40131 |
| classical | dbscan | istella | noise_fraction | - | 0.219391 | cuml-gpu 0.219391 |
| classical | dbscan | istella | rows | - | 1000000 | cuml-gpu 1000000 |
| classical | dbscan | istella | ari_vs_ours (1 is our partition exactly) | - | - | cuml-gpu 1.000000 |
| classical | dbscan | istella | noise_agreement_vs_ours | - | - | cuml-gpu 1.000000 |
| classical | dbscan | taxi | n_clusters | - | 36 | cuml-gpu 36 |
| classical | dbscan | taxi | noise_fraction | - | 0.000174 | cuml-gpu 0.000174 |
| classical | dbscan | taxi | rows | - | 1000000 | cuml-gpu 1000000 |
| classical | dbscan | taxi | ari_vs_ours (1 is our partition exactly) | - | - | cuml-gpu 1.000000 |
| classical | dbscan | taxi | noise_agreement_vs_ours | - | - | cuml-gpu 1.000000 |
| classical | hdbscan | istella | n_clusters | - | 47 | cuml-gpu 53 |
| classical | hdbscan | istella | noise_fraction | - | 0.253810 | cuml-gpu 0.256280 |
| classical | hdbscan | istella | rows | - | 100000 | cuml-gpu 100000 |
| classical | hdbscan | istella | silhouette (higher is better) | - | -0.003856 | cuml-gpu -0.018228 |
| classical | hdbscan | istella | ari_vs_ours (1 is our partition exactly) | - | - | cuml-gpu 0.857120 |
| classical | hdbscan | istella | noise_agreement_vs_ours | - | - | cuml-gpu 0.940610 |
| classical | hdbscan | taxi | n_clusters | - | 161 | cuml-gpu 159 |
| classical | hdbscan | taxi | noise_fraction | - | 0.140550 | cuml-gpu 0.130970 |
| classical | hdbscan | taxi | rows | - | 100000 | cuml-gpu 100000 |
| classical | hdbscan | taxi | silhouette (higher is better) | - | 0.001505 | cuml-gpu 0.011960 |
| classical | hdbscan | taxi | ari_vs_ours (1 is our partition exactly) | - | - | cuml-gpu 0.917885 |
| classical | hdbscan | taxi | noise_agreement_vs_ours | - | - | cuml-gpu 0.984420 |
| classical | kde | istella | mean_log_likelihood (higher is better) | - | -222.270583 | cuml-gpu -222.270582 |
| classical | kde | istella | rows_without_density | - | 0 | cuml-gpu 0 |
| classical | kde | taxi | mean_log_likelihood (higher is better) | - | -14.826436 | cuml-gpu -14.826437 |
| classical | kde | taxi | rows_without_density | - | 0 | cuml-gpu 0 |
| classical | kmeans | istella | inertia (lower is better) | - | 6.051e+17 | cuml-gpu 6.111e+17 |
| classical | kmeans | istella | inertia_over_ours | - | 1.000000 | cuml-gpu 1.009912 |
| classical | kmeans | istella | n_iter | - | 33 | cuml-gpu 21 |
| classical | kmeans | istella | n_iter_over_ours | - | 1.000000 | cuml-gpu 0.636364 |
| classical | kmeans | taxi | inertia (lower is better) | - | 3.093e+08 | cuml-gpu 3.06e+08 |
| classical | kmeans | taxi | inertia_over_ours | - | 1.000000 | cuml-gpu 0.989330 |
| classical | kmeans | taxi | n_iter | - | 91 | cuml-gpu 33 |
| classical | kmeans | taxi | n_iter_over_ours | - | 1.000000 | cuml-gpu 0.362637 |
| classical | knn | istella | recall_at_k (higher is better) | - | 0.976250 | cuml-gpu 0.976402 |
| classical | knn | istella | rows_with_repeated_ids | - | 0 | cuml-gpu 0 |
| classical | knn | taxi | recall_at_k (higher is better) | - | 0.999754 | cuml-gpu 0.999742 |
| classical | knn | taxi | rows_with_repeated_ids | - | 0 | cuml-gpu 0 |
| classical | ols | istella | r2 (higher is better) | - | 0.332506 | cuml-gpu -11031.855105; torch-gpu nan; torch-gpu-eigh 0.151604 |
| classical | ols | istella | rmse (lower is better) | - | 0.681740 | cuml-gpu 87.647429; torch-gpu nan; torch-gpu-eigh 0.768590 |
| classical | ols | taxi | r2 (higher is better) | - | 0.908837 | cuml-gpu 0.908836; torch-gpu 0.908836; torch-gpu-eigh 0.908836 |
| classical | ols | taxi | rmse (lower is better) | - | 4.696477 | cuml-gpu 4.696488; torch-gpu 4.696480; torch-gpu-eigh 4.696490 |
| classical | pca | istella | explained_variance_ratio_sum (higher is better) | - | 1.000000 | cuml-gpu 1.000000 |
| classical | pca | taxi | explained_variance_ratio_sum (higher is better) | - | 0.999997 | cuml-gpu 0.999996 |
| classical | svc | istella | accuracy (higher is better) | - | 0.922200 | cuml-gpu 0.922200 |
| classical | svc | istella | n_support | - | 2400 | cuml-gpu 2401 |
| classical | svc | taxi | accuracy (higher is better) | - | 0.767500 | cuml-gpu 0.767500 |
| classical | svc | taxi | n_support | - | 5527 | cuml-gpu 5541 |
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
| classical2 | ivf | istella | recall_at_k (higher is better) | - | 0.999925 | cuvs-gpu 0.999975 |
| classical2 | ivf | istella | rows_with_repeated_ids | - | 0 | cuvs-gpu 0 |
| classical2 | ivf | taxi | recall_at_k (higher is better) | - | 0.999650 | cuvs-gpu 0.999450 |
| classical2 | ivf | taxi | rows_with_repeated_ids | - | 0 | cuvs-gpu 0 |
| classical2 | kernel-ridge | istella | r2 (higher is better) | - | 0.385427 | cuml-gpu 0.385427 |
| classical2 | kernel-ridge | istella | rmse (lower is better) | - | 0.646407 | cuml-gpu 0.646407 |
| classical2 | kernel-ridge | taxi | r2 (higher is better) | - | 0.726543 | cuml-gpu 0.726543 |
| classical2 | kernel-ridge | taxi | rmse (lower is better) | - | 8.330373 | cuml-gpu 8.330375 |
| classical2 | knn-clf | istella | accuracy (higher is better) | - | 0.926250 | cuml-gpu 0.926250 |
| classical2 | knn-clf | taxi | accuracy (higher is better) | - | 0.741750 | cuml-gpu 0.741750 |
| classical2 | knn-reg | istella | r2 (higher is better) | - | 0.418145 | cuml-gpu 0.418145 |
| classical2 | knn-reg | istella | rmse (lower is better) | - | 0.625388 | cuml-gpu 0.625388 |
| classical2 | knn-reg | taxi | r2 (higher is better) | - | 0.937323 | cuml-gpu 0.937323 |
| classical2 | knn-reg | taxi | rmse (lower is better) | - | 3.842028 | cuml-gpu 3.842028 |
| classical2 | lasso | istella | r2 (higher is better) | - | 0.310837 | - |
| classical2 | lasso | istella | rmse (lower is better) | - | 0.693460 | - |
| classical2 | lasso | taxi | r2 (higher is better) | - | 0.908995 | - |
| classical2 | lasso | taxi | rmse (lower is better) | - | 4.804744 | - |
| classical2 | linearsvc | istella | accuracy (higher is better) | - | 0.923470 | cuml-gpu 0.922600 |
| classical2 | linearsvc | taxi | accuracy (higher is better) | - | 0.763330 | cuml-gpu 0.762990 |
| classical2 | linearsvr | istella | r2 (higher is better) | - | -0.106761 | cuml-gpu -0.106761 |
| classical2 | linearsvr | istella | rmse (lower is better) | - | 0.878794 | cuml-gpu 0.878795 |
| classical2 | linearsvr | taxi | r2 (higher is better) | - | 0.899814 | cuml-gpu 0.899818 |
| classical2 | linearsvr | taxi | rmse (lower is better) | - | 5.041273 | cuml-gpu 5.041176 |
| classical2 | logreg | istella | accuracy (higher is better) | - | 0.924590 | - |
| classical2 | logreg | istella | logloss (lower is better) | - | 0.181249 | - |
| classical2 | logreg | istella | nonfinite_proba_rows | - | 0 | - |
| classical2 | logreg | taxi | accuracy (higher is better) | - | 0.763350 | - |
| classical2 | logreg | taxi | logloss (lower is better) | - | 0.538985 | - |
| classical2 | logreg | taxi | nonfinite_proba_rows | - | 0 | - |
| classical2 | ridge | istella | r2 (higher is better) | - | 0.328674 | cuml-gpu -0.251259 |
| classical2 | ridge | istella | rmse (lower is better) | - | 0.684427 | cuml-gpu 0.934403 |
| classical2 | ridge | taxi | r2 (higher is better) | - | 0.908983 | cuml-gpu 0.908983 |
| classical2 | ridge | taxi | rmse (lower is better) | - | 4.805050 | cuml-gpu 4.805051 |
| classical2 | svr | istella | r2 (higher is better) | - | 0.318258 | cuml-gpu 0.318232 |
| classical2 | svr | istella | rmse (lower is better) | - | 0.680816 | cuml-gpu 0.680829 |
| classical2 | svr | taxi | r2 (higher is better) | - | 0.767551 | cuml-gpu 0.767551 |
| classical2 | svr | taxi | rmse (lower is better) | - | 7.680395 | cuml-gpu 7.680405 |
| classical2 | tsvd | istella | explained_variance_ratio_sum (higher is better) | - | 1.000000 | cuml-gpu 1.000000 |
| classical2 | tsvd | istella | relative_reconstruction_error (lower is better) | - | 0.0001314 | cuml-gpu 0.0001472 |
| classical2 | tsvd | taxi | explained_variance_ratio_sum (higher is better) | - | 0.999965 | cuml-gpu 0.999964 |
| classical2 | tsvd | taxi | relative_reconstruction_error (lower is better) | - | 0.003257 | cuml-gpu 0.003257 |
| neural | gemm | gaussian | max_rel_err_vs_fp64 (lower is better) | - | 2.399e-07 | torch-eager-fp32 1.401e-06; torch-eager-tf32 0.0002784; torch-compile-fp32 1.401e-06; torch-compile-tf32 0.0002784; torch-eager-bf16 0.003752; torch-compile-bf16 0.003752 |
| neural | gemm | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | torch-eager-fp32 0.0005341; torch-eager-tf32 0.100822; torch-compile-fp32 0.0005341; torch-compile-tf32 0.100822; torch-eager-bf16 1.358337; torch-compile-bf16 1.358337 |
| neural | gemm | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | torch-eager-fp32 1.475e-06; torch-eager-tf32 0.0002785; torch-compile-fp32 1.475e-06; torch-compile-tf32 0.0002785; torch-eager-bf16 0.003751; torch-compile-bf16 0.003751 |
| neural | lm-forward | bytes | mean_nll (lower is better) | - | 9.018733 | torch-eager-fp32 9.018733; torch-eager-tf32 9.018733; torch-compile-fp32 9.018733; torch-compile-tf32 9.018732; torch-eager-bf16 9.018664; torch-compile-bf16 9.018669 |
| neural | lm-train-step | bytes | loss_first_step (same init and batches on every arm) | - | 9.018733 | torch-eager-fp32 9.018733; torch-eager-tf32 9.018732; torch-compile-fp32 9.018734; torch-compile-tf32 9.018732; torch-eager-bf16 9.018402; torch-compile-bf16 9.018669 |
| neural | lm-train-step | bytes | loss_last_step (same init and batches on every arm) | - | 8.418446 | torch-eager-fp32 8.418449; torch-eager-tf32 8.418548; torch-compile-fp32 8.418448; torch-compile-tf32 8.418557; torch-eager-bf16 8.417328; torch-compile-bf16 8.417006 |
| neural | lm-train-step | bytes | steps | - | 2 | torch-eager-fp32 2; torch-eager-tf32 2; torch-compile-fp32 2; torch-compile-tf32 2; torch-eager-bf16 2; torch-compile-bf16 2 |
| neural | mamba1-forward | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | torch-eager-fp32 -; torch-eager-tf32 -; torch-eager-bf16 -; mamba-ssm-fp32 1.192e-07; mamba-ssm-tf32 3.815e-06 |
| neural | mamba1-forward | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | torch-eager-fp32 -; torch-eager-tf32 -; torch-eager-bf16 -; mamba-ssm-fp32 5.945e-08; mamba-ssm-tf32 1.902e-06 |
| neural | mamba2-forward | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | torch-eager-fp32 -; torch-eager-tf32 -; torch-compile-fp32 -; torch-compile-tf32 -; torch-eager-bf16 -; torch-compile-bf16 -; mamba-ssm-fp32 1.907e-06; mamba-ssm-tf32 0.0005126 |
| neural | mamba2-forward | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | torch-eager-fp32 -; torch-eager-tf32 -; torch-compile-fp32 -; torch-compile-tf32 -; torch-eager-bf16 -; torch-compile-bf16 -; mamba-ssm-fp32 6.485e-07; mamba-ssm-tf32 0.0001743 |
| neural | mamba3-forward | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | torch-eager-fp32 -; torch-eager-tf32 -; torch-compile-fp32 -; torch-compile-tf32 -; torch-eager-bf16 -; torch-compile-bf16 -; mamba-ssm-fp32 0.001235; mamba-ssm-tf32 0.001376 |
| neural | mamba3-forward | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | torch-eager-fp32 -; torch-eager-tf32 -; torch-compile-fp32 -; torch-compile-tf32 -; torch-eager-bf16 -; torch-compile-bf16 -; mamba-ssm-fp32 0.0005401; mamba-ssm-tf32 0.0006018 |
| neural | mlp-train-step | gaussian | loss_first_step (same init and batches on every arm) | - | 1.160401 | torch-eager-fp32 1.160401; torch-eager-tf32 1.160392; torch-compile-fp32 1.160401; torch-compile-tf32 1.160392; torch-eager-bf16 1.160498; torch-compile-bf16 1.160498 |
| neural | mlp-train-step | gaussian | loss_last_step (same init and batches on every arm) | - | 1.123361 | torch-eager-fp32 1.123361; torch-eager-tf32 1.123355; torch-compile-fp32 1.123361; torch-compile-tf32 1.123355; torch-eager-bf16 1.123461; torch-compile-bf16 1.123462 |
| neural | mlp-train-step | gaussian | steps | - | 2 | torch-eager-fp32 2; torch-eager-tf32 2; torch-compile-fp32 2; torch-compile-tf32 2; torch-eager-bf16 2; torch-compile-bf16 2 |
| neural | samba-forward | bytes | mean_nll (lower is better) | - | 5.635910 | torch-eager-fp32 5.635910; torch-eager-tf32 5.635948; torch-compile-fp32 5.635910; torch-compile-tf32 5.635950; torch-eager-bf16 5.635952; torch-compile-bf16 5.635985 |
| neural | samba-train-step | bytes | loss_first_step (same init and batches on every arm) | - | 5.635910 | torch-eager-fp32 5.635910; torch-eager-tf32 5.635948; torch-compile-fp32 5.635910; torch-compile-tf32 5.635947; torch-eager-bf16 5.635952; torch-compile-bf16 - |
| neural | samba-train-step | bytes | loss_last_step (same init and batches on every arm) | - | 4.833934 | torch-eager-fp32 4.833934; torch-eager-tf32 4.833591; torch-compile-fp32 4.833934; torch-compile-tf32 4.833560; torch-eager-bf16 4.833967; torch-compile-bf16 - |
| neural | samba-train-step | bytes | steps | - | 2 | torch-eager-fp32 2; torch-eager-tf32 2; torch-compile-fp32 2; torch-compile-tf32 2; torch-eager-bf16 2; torch-compile-bf16 - |
| trees | et | istella | auc (higher is better) | - | 0.937987 | sklearn-et-cpu 0.937659 |
| trees | et | istella | logloss (lower is better) | - | 0.189989 | sklearn-et-cpu 0.190177 |
| trees | et | taxi | auc (higher is better) | - | 0.618907 | sklearn-et-cpu 0.619262 |
| trees | et | taxi | logloss (lower is better) | - | 0.526142 | sklearn-et-cpu 0.525951 |
| trees | gbdt-categorical | taxi | auc (higher is better) | - | 0.631297 | catboost-gpu 0.630387; xgboost-gpu 0.631978; catboost-cpu -; xgboost-cpu -; lightgbm-cpu -; lightgbm-cuda 0.500000 |
| trees | gbdt-categorical | taxi | logloss (lower is better) | - | 0.528285 | catboost-gpu 0.528535; xgboost-gpu 0.528548; catboost-cpu -; xgboost-cpu -; lightgbm-cpu -; lightgbm-cuda 0.554692 |
| trees | gbdt-depthwise | istella | auc (higher is better) | - | 0.983304 | - |
| trees | gbdt-depthwise | istella | logloss (lower is better) | - | 0.156072 | - |
| trees | gbdt-depthwise | taxi | auc (higher is better) | - | 0.632205 | - |
| trees | gbdt-depthwise | taxi | logloss (lower is better) | - | 0.527838 | - |
| trees | gbdt-lossguide | istella | auc (higher is better) | - | 0.983586 | catboost-gpu 0.983668; xgboost-gpu 0.983622; catboost-cpu -; xgboost-cpu -; lightgbm-cuda 0.500000; lightgbm-cpu - |
| trees | gbdt-lossguide | istella | logloss (lower is better) | - | 0.150066 | catboost-gpu 0.149188; xgboost-gpu 0.149263; catboost-cpu -; xgboost-cpu -; lightgbm-cuda 0.356515; lightgbm-cpu - |
| trees | gbdt-lossguide | taxi | auc (higher is better) | - | 0.632096 | catboost-gpu 0.631766; xgboost-gpu 0.630969; catboost-cpu -; xgboost-cpu -; lightgbm-cpu -; lightgbm-cuda 0.500000 |
| trees | gbdt-lossguide | taxi | logloss (lower is better) | - | 0.528044 | catboost-gpu 0.528065; xgboost-gpu 0.528678; catboost-cpu -; xgboost-cpu -; lightgbm-cpu -; lightgbm-cuda 0.554692 |
| trees | gbdt-multiclass | istella | accuracy (higher is better) | - | 0.907556 | catboost-gpu 0.907780; xgboost-gpu 0.910140; catboost-cpu -; xgboost-cpu -; lightgbm-cuda 0.911402; lightgbm-cpu - |
| trees | gbdt-multiclass | istella | mlogloss (lower is better) | - | 0.258413 | catboost-gpu 0.258149; xgboost-gpu 0.246803; catboost-cpu -; xgboost-cpu -; lightgbm-cuda 0.243459; lightgbm-cpu - |
| trees | gbdt-multiclass | taxi | accuracy (higher is better) | - | 0.599380 | catboost-gpu 0.599270; xgboost-gpu 0.601200; catboost-cpu -; xgboost-cpu -; lightgbm-cpu -; lightgbm-cuda 0.601896 |
| trees | gbdt-multiclass | taxi | mlogloss (lower is better) | - | 1.012595 | catboost-gpu 1.012704; xgboost-gpu 1.005204; catboost-cpu -; xgboost-cpu -; lightgbm-cpu -; lightgbm-cuda 1.003173 |
| trees | gbdt-ordered | istella | auc (higher is better) | - | 0.979444 | - |
| trees | gbdt-ordered | istella | logloss (lower is better) | - | 0.190928 | - |
| trees | gbdt-ordered | taxi | auc (higher is better) | - | 0.629203 | - |
| trees | gbdt-ordered | taxi | logloss (lower is better) | - | 0.529007 | - |
| trees | gbdt-rank-pairlogit | istella | map (higher is better) | - | 0.854545 | catboost-gpu 0.853929; xgboost-gpu 0.872796; catboost-cpu -; xgboost-cpu - |
| trees | gbdt-rank-pairlogit | istella | ndcg10 (higher is better) | - | 0.719953 | catboost-gpu 0.719621; xgboost-gpu 0.738397; catboost-cpu -; xgboost-cpu - |
| trees | gbdt-rank-pairlogit | istella | ndcg5 (higher is better) | - | 0.650400 | catboost-gpu 0.650025; xgboost-gpu 0.670093; catboost-cpu -; xgboost-cpu - |
| trees | gbdt-symmetric | istella | auc (higher is better) | - | 0.980129 | catboost-gpu 0.980018; catboost-cpu - |
| trees | gbdt-symmetric | istella | logloss (lower is better) | - | 0.186686 | catboost-gpu 0.187292; catboost-cpu - |
| trees | gbdt-symmetric | taxi | auc (higher is better) | - | 0.630436 | catboost-gpu 0.630310; catboost-cpu - |
| trees | gbdt-symmetric | taxi | logloss (lower is better) | - | 0.528554 | catboost-gpu 0.528616; catboost-cpu - |
| trees | iforest | istella | auc (higher is better) | - | 0.829336 | cuml-iforest-gpu 0.830358 |
| trees | iforest | taxi | auc (higher is better) | - | 0.547466 | cuml-iforest-gpu 0.551846 |
| trees | rf | istella | auc (higher is better) | - | 0.945385 | cuml-rf-gpu 0.945348 |
| trees | rf | istella | logloss (lower is better) | - | 0.182017 | cuml-rf-gpu 0.182069 |
| trees | rf | taxi | auc (higher is better) | - | 0.617838 | cuml-rf-gpu 0.617836 |
| trees | rf | taxi | logloss (lower is better) | - | 0.525953 | cuml-rf-gpu 0.525954 |

## Inference at a glance

Batch prediction, each arm with its own fitted model from the same race; medians in ms. `FAST = IDENTICAL bits` compares our two tiers' predictions on the same rows. Our CPU is never raced or reported.

| family | lane | dataset | batch | rows | ours FAST ms | ours IDENTICAL ms | FAST = IDENTICAL bits | opponents |
|---|---|---|---|---|---|---|---|---|
| algos | avgpool1d | synthetic | Xq | - | - | 0.2 | - | torch-eager-fp32 0.2 ms (IDENTICAL/arm 1.002); torch-compile-fp32 0.2 ms (IDENTICAL/arm 0.772); torch-eager-tf32 0.2 ms (IDENTICAL/arm 1.004); torch-compile-tf32 0.3 ms (IDENTICAL/arm 0.722); torch-eager-bf16 0.2 ms (IDENTICAL/arm 0.911); torch-compile-bf16 0.3 ms (IDENTICAL/arm 0.653) |
| algos | avgpool2d | synthetic | Xq | - | - | 0.5 | - | torch-eager-fp32 0.4 ms (IDENTICAL/arm 1.041); torch-compile-fp32 0.6 ms (IDENTICAL/arm 0.810); torch-eager-tf32 0.5 ms (IDENTICAL/arm 0.957); torch-compile-tf32 0.6 ms (IDENTICAL/arm 0.817); torch-eager-bf16 0.5 ms (IDENTICAL/arm 0.993); torch-compile-bf16 0.6 ms (IDENTICAL/arm 0.813) |
| algos | batchnorm1d | synthetic | Xq | - | - | 0.4 | - | torch-eager-fp32 0.3 ms (IDENTICAL/arm 1.321); torch-compile-fp32 0.4 ms (IDENTICAL/arm 1.065); torch-eager-tf32 0.3 ms (IDENTICAL/arm 1.256); torch-compile-tf32 0.4 ms (IDENTICAL/arm 0.991); torch-eager-bf16 0.3 ms (IDENTICAL/arm 1.253); torch-compile-bf16 0.4 ms (IDENTICAL/arm 0.940) |
| algos | batchnorm2d | synthetic | Xq | - | - | 0.3 | - | torch-eager-fp32 0.2 ms (IDENTICAL/arm 1.255); torch-compile-fp32 0.2 ms (IDENTICAL/arm 1.098); torch-eager-tf32 0.2 ms (IDENTICAL/arm 1.233); torch-compile-tf32 0.3 ms (IDENTICAL/arm 0.916); torch-eager-bf16 0.2 ms (IDENTICAL/arm 1.096); torch-compile-bf16 0.3 ms (IDENTICAL/arm 0.990) |
| algos | binarizer | istella | Xq | - | - | 76.4 | - | cuml-gpu 8.2 ms (IDENTICAL/arm 9.337) |
| algos | binarizer | taxi | Xq | - | - | 4.4 | - | cuml-gpu 2.1 ms (IDENTICAL/arm 2.063) |
| algos | conv1d | synthetic | Xq | - | - | 2.7 | - | torch-eager-fp32 1.2 ms (IDENTICAL/arm 2.372); torch-compile-fp32 1.2 ms (IDENTICAL/arm 2.253); torch-eager-tf32 1.1 ms (IDENTICAL/arm 2.396); torch-compile-tf32 1.3 ms (IDENTICAL/arm 2.189); torch-eager-bf16 1.1 ms (IDENTICAL/arm 2.467); torch-compile-bf16 1.1 ms (IDENTICAL/arm 2.410) |
| algos | conv2d | synthetic | Xq | - | - | 1.6 | - | torch-eager-fp32 0.5 ms (IDENTICAL/arm 3.074); torch-compile-fp32 1.3 ms (IDENTICAL/arm 1.292); torch-eager-tf32 0.4 ms (IDENTICAL/arm 4.522); torch-compile-tf32 0.6 ms (IDENTICAL/arm 2.726); torch-eager-bf16 0.4 ms (IDENTICAL/arm 4.538); torch-compile-bf16 0.3 ms (IDENTICAL/arm 4.703) |
| algos | dropout2d | synthetic | Xq | - | - | 0.3 | - | torch-eager-fp32 0.2 ms (IDENTICAL/arm 1.154); torch-compile-fp32 0.3 ms (IDENTICAL/arm 0.846); torch-eager-tf32 0.2 ms (IDENTICAL/arm 1.175); torch-compile-tf32 0.3 ms (IDENTICAL/arm 0.803) |
| algos | embedding | synthetic | Xq | - | - | 0.6 | - | torch-eager-fp32 0.4 ms (IDENTICAL/arm 1.424); torch-compile-fp32 0.4 ms (IDENTICAL/arm 1.276) |
| algos | global-avgpool | synthetic | Xq | - | - | 0.0 | - | torch-eager-fp32 0.0 ms (IDENTICAL/arm 1.004); torch-compile-fp32 0.1 ms (IDENTICAL/arm 0.371); torch-eager-tf32 0.1 ms (IDENTICAL/arm 0.819); torch-compile-tf32 0.1 ms (IDENTICAL/arm 0.350); torch-eager-bf16 0.1 ms (IDENTICAL/arm 0.755); torch-compile-bf16 0.1 ms (IDENTICAL/arm 0.361) |
| algos | global-maxpool | synthetic | Xq | - | - | 0.1 | - | torch-eager-fp32 0.1 ms (IDENTICAL/arm 1.085); torch-compile-fp32 0.3 ms (IDENTICAL/arm 0.339); torch-eager-tf32 0.1 ms (IDENTICAL/arm 1.002); torch-compile-tf32 0.2 ms (IDENTICAL/arm 0.541); torch-eager-bf16 0.1 ms (IDENTICAL/arm 0.878); torch-compile-bf16 0.1 ms (IDENTICAL/arm 0.716) |
| algos | label-binarizer | istella | Xq | - | - | 6.1 | - | cuml-gpu 6.1 ms (IDENTICAL/arm 0.992) |
| algos | label-binarizer | taxi | Xq | - | - | 83.5 | - | cuml-gpu 30.2 ms (IDENTICAL/arm 2.768) |
| algos | layernorm | synthetic | Xq | - | - | 0.2 | - | torch-eager-fp32 0.2 ms (IDENTICAL/arm 0.963); torch-compile-fp32 0.3 ms (IDENTICAL/arm 0.590); torch-eager-tf32 0.2 ms (IDENTICAL/arm 0.928); torch-compile-tf32 0.3 ms (IDENTICAL/arm 0.593); torch-eager-bf16 0.2 ms (IDENTICAL/arm 0.843); torch-compile-bf16 0.3 ms (IDENTICAL/arm 0.594) |
| algos | maxabs-scaler | istella | Xq | - | - | 75.1 | - | cuml-gpu 2.9 ms (IDENTICAL/arm 26.021) |
| algos | maxabs-scaler | taxi | Xq | - | - | 4.2 | - | cuml-gpu 0.8 ms (IDENTICAL/arm 5.018) |
| algos | maxpool1d | synthetic | Xq | - | - | 0.2 | - | torch-eager-fp32 0.3 ms (IDENTICAL/arm 0.928); torch-compile-fp32 0.2 ms (IDENTICAL/arm 1.060); torch-eager-tf32 0.3 ms (IDENTICAL/arm 0.919); torch-compile-tf32 0.3 ms (IDENTICAL/arm 0.945); torch-eager-bf16 0.3 ms (IDENTICAL/arm 0.883); torch-compile-bf16 0.3 ms (IDENTICAL/arm 0.818) |
| algos | maxpool2d | synthetic | Xq | - | - | 0.5 | - | torch-eager-fp32 0.6 ms (IDENTICAL/arm 0.905); torch-compile-fp32 0.5 ms (IDENTICAL/arm 1.004); torch-eager-tf32 0.6 ms (IDENTICAL/arm 0.894); torch-compile-tf32 0.5 ms (IDENTICAL/arm 1.007); torch-eager-bf16 0.6 ms (IDENTICAL/arm 0.883); torch-compile-bf16 0.5 ms (IDENTICAL/arm 0.991) |
| algos | minmax-scaler | istella | Xq | - | - | 57.2 | - | cuml-gpu 3.1 ms (IDENTICAL/arm 18.673) |
| algos | minmax-scaler | taxi | Xq | - | - | 3.1 | - | cuml-gpu 0.7 ms (IDENTICAL/arm 4.268) |
| algos | normalizer | istella | Xq | - | - | 73.0 | - | cuml-gpu 1.7 ms (IDENTICAL/arm 43.962) |
| algos | normalizer | taxi | Xq | - | - | 4.5 | - | cuml-gpu 4.2 ms (IDENTICAL/arm 1.090) |
| algos | poly-features | istella | Xq | - | - | 49.9 | - | cuml-gpu 14.8 ms (IDENTICAL/arm 3.381) |
| algos | poly-features | taxi | Xq | - | - | 24.9 | - | cuml-gpu 4.1 ms (IDENTICAL/arm 6.104) |
| algos | power-transformer | istella | Xq | - | - | 75.9 | - | cuml-gpu - ms (IDENTICAL/arm -) |
| algos | power-transformer | taxi | Xq | - | - | 4.4 | - | cuml-gpu 12.5 ms (IDENTICAL/arm 0.351) |
| algos | quantile-transformer | istella | Xq | - | - | 81.8 | - | cuml-gpu 261.5 ms (IDENTICAL/arm 0.313) |
| algos | quantile-transformer | taxi | Xq | - | - | 4.1 | - | cuml-gpu 27.5 ms (IDENTICAL/arm 0.148) |
| algos | standard-scaler | istella | Xq | - | - | 52.8 | - | cuml-gpu 3.6 ms (IDENTICAL/arm 14.720) |
| algos | standard-scaler | taxi | Xq | - | - | 3.2 | - | cuml-gpu 0.6 ms (IDENTICAL/arm 4.918) |
| classical | kmeans | istella | Xq | 500000 | - | 38.4 | - | cuml-gpu 5.2 ms (IDENTICAL/arm 7.339) |
| classical | kmeans | taxi | Xq | 500000 | - | 3.1 | - | cuml-gpu 3.0 ms (IDENTICAL/arm 1.018) |
| classical | pca | istella | Xq | 500000 | - | 34.8 | - | cuml-gpu 10.3 ms (IDENTICAL/arm 3.380) |
| classical | pca | taxi | Xq | 500000 | - | 18.3 | - | cuml-gpu 2.8 ms (IDENTICAL/arm 6.444) |
| trees | iforest | istella | test | 500000 | - | 57.3 | - | cuml-iforest-gpu 77.0 ms (IDENTICAL/arm 0.744) |
| trees | iforest | istella | large | 1000000 | - | 121.6 | - | cuml-iforest-gpu 156.8 ms (IDENTICAL/arm 0.775) |
| trees | iforest | taxi | test | 500000 | - | 2.7 | - | cuml-iforest-gpu 5.6 ms (IDENTICAL/arm 0.481) |
| trees | iforest | taxi | large | 1000000 | - | 4.8 | - | cuml-iforest-gpu 9.4 ms (IDENTICAL/arm 0.511) |


## Neural headline: ours IDENTICAL against torch bf16

What customers run is torch in bf16. The headline divides our IDENTICAL median by torch's fastest bf16 arm (eager or compile, the lower stored median); the fp32 twin (torch's fastest fp32 arm) is the second column. Per-arm ratios stay in each race's table below.

| lane | dataset | ours IDENTICAL ms | torch bf16 ms (fastest arm) | ours / torch bf16 | torch fp32 twin ms (fastest arm) | ours / torch fp32 | note |
|---|---|---|---|---|---|---|---|
| adafactor | synthetic | 13.8 | - | - | 16.6 (torch-eager-fp32) | 0.829 | no torch bf16 arm on this lane |
| adagrad | synthetic | 8.7 | - | - | 9.1 (torch-compile-fp32) | 0.956 | no torch bf16 arm on this lane |
| adam | synthetic | 23.9 | - | - | 16.7 (torch-eager-fp32) | 1.431 | no torch bf16 arm on this lane |
| adamax | synthetic | 11.1 | - | - | 18.0 (torch-eager-fp32) | 0.619 | no torch bf16 arm on this lane |
| adamw | synthetic | 23.9 | - | - | 17.9 (torch-eager-fp32) | 1.334 | no torch bf16 arm on this lane |
| avgpool1d | synthetic | 1.1 | 1.2 (torch-eager-bf16) | 0.877 | 0.9 (torch-compile-fp32) | 1.136 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| avgpool2d | synthetic | 1.6 | 1.8 (torch-eager-bf16) | 0.891 | 1.2 (torch-compile-fp32) | 1.328 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| batchnorm1d | synthetic | 1.6 | 1.7 (torch-eager-bf16) | 0.928 | 1.3 (torch-compile-fp32) | 1.212 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| batchnorm2d | synthetic | 1.6 | 1.6 (torch-eager-bf16) | 0.968 | 1.1 (torch-compile-fp32) | 1.437 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| cnn-clf | synthetic | 200.4 | 483.3 (torch-eager-bf16) | 0.415 | 458.6 (torch-eager-fp32) | 0.437 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| conv1d | synthetic | 9.9 | 3.2 (torch-eager-bf16) | 3.119 | 3.9 (torch-compile-fp32) | 2.511 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| conv2d | synthetic | 6.3 | 1.8 (torch-eager-bf16) | 3.465 | 2.2 (torch-eager-fp32) | 2.916 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| dropout2d | synthetic | 1.3 | - | - | 1.2 (torch-eager-fp32) | 1.064 | no torch bf16 arm on this lane |
| embedding | synthetic | 2.5 | - | - | 2.2 (torch-compile-fp32) | 1.125 | no torch bf16 arm on this lane |
| gcn | istella | 2.9 | 8.5 (torch-compile-bf16) | 0.342 | 4.5 (torch-compile-fp32) | 0.654 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| gcn | taxi | 1.7 | 9.3 (torch-compile-bf16) | 0.183 | 3.3 (torch-compile-fp32) | 0.512 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| global-avgpool | synthetic | 0.3 | 0.5 (torch-eager-bf16) | 0.575 | 0.5 (torch-eager-fp32) | 0.565 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| global-maxpool | synthetic | 0.4 | 0.6 (torch-eager-bf16) | 0.605 | 0.6 (torch-eager-fp32) | 0.660 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| graphsage | istella | 6.2 | 3.3 (torch-compile-bf16) | 1.860 | 4.2 (torch-compile-fp32) | 1.459 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| graphsage | taxi | 1.8 | 1.7 (torch-eager-bf16) | 1.063 | 1.6 (torch-eager-fp32) | 1.117 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| gru-clf | synthetic | 447.1 | 1311.6 (torch-compile-bf16) | 0.341 | 1103.7 (torch-eager-fp32) | 0.405 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| gru-clf | taxi-hourly | 447.7 | 789.0 (torch-compile-bf16) | 0.567 | 591.3 (torch-eager-fp32) | 0.757 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| gru-reg | synthetic | 446.3 | 1333.4 (torch-compile-bf16) | 0.335 | 1133.5 (torch-eager-fp32) | 0.394 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| gru-reg | taxi-hourly | 447.0 | 696.1 (torch-eager-bf16) | 0.642 | 650.5 (torch-eager-fp32) | 0.687 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| lamb | synthetic | 21.7 | - | - | - | - | no torch bf16 arm on this lane |
| layernorm | synthetic | 1.2 | 2.0 (torch-compile-bf16) | 0.591 | 1.5 (torch-eager-fp32) | 0.781 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| lion | synthetic | 8.5 | - | - | - | - | no torch bf16 arm on this lane |
| lstm-clf | synthetic | 546.6 | 743.9 (torch-eager-bf16) | 0.735 | 516.0 (torch-eager-fp32) | 1.059 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| lstm-clf | taxi-hourly | 546.5 | 1735.3 (torch-compile-bf16) | 0.315 | 1246.5 (torch-eager-fp32) | 0.438 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| lstm-reg | synthetic | 545.6 | 857.8 (torch-compile-bf16) | 0.636 | 564.5 (torch-compile-fp32) | 0.966 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| lstm-reg | taxi-hourly | 545.8 | 1539.5 (torch-compile-bf16) | 0.355 | 1243.7 (torch-compile-fp32) | 0.439 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| maxpool1d | synthetic | 1.2 | 1.5 (torch-eager-bf16) | 0.832 | 1.3 (torch-compile-fp32) | 0.959 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| maxpool2d | synthetic | 1.2 | 1.9 (torch-eager-bf16) | 0.637 | 1.7 (torch-eager-fp32) | 0.695 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| moe | synthetic | 14.5 | 4.3 (torch-eager-bf16) | 3.330 | 11.9 (torch-compile-fp32) | 1.213 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| nadam | synthetic | 10.9 | - | - | 21.1 (torch-eager-fp32) | 0.513 | no torch bf16 arm on this lane |
| resnet-block | synthetic | 15.0 | 2.8 (torch-eager-bf16) | 5.279 | 5.3 (torch-compile-fp32) | 2.850 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| rmsprop | synthetic | 8.7 | - | - | 13.1 (torch-eager-fp32) | 0.664 | no torch bf16 arm on this lane |
| rnn-clf | synthetic | 244.0 | 873.9 (torch-compile-bf16) | 0.279 | 904.6 (torch-eager-fp32) | 0.270 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| rnn-clf | taxi-hourly | 243.9 | 642.8 (torch-compile-bf16) | 0.379 | 507.7 (torch-eager-fp32) | 0.480 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| rnn-reg | synthetic | 242.4 | 1442.1 (torch-compile-bf16) | 0.168 | 902.4 (torch-compile-fp32) | 0.269 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| rnn-reg | taxi-hourly | 242.4 | 612.7 (torch-eager-bf16) | 0.396 | 664.7 (torch-eager-fp32) | 0.365 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| sgd | synthetic | 20.9 | - | - | 9.5 (torch-eager-fp32) | 2.201 | no torch bf16 arm on this lane |
| gemm | gaussian | 53.2 | 35.5 (torch-eager-bf16) | 1.500 | 37.6 (torch-eager-fp32) | 1.416 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| lm-forward | bytes | 55.4 | 42.5 (torch-compile-bf16) | 1.305 | 45.8 (torch-compile-fp32) | 1.209 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| lm-train-step | bytes | 37.5 | 10.8 (torch-compile-bf16) | 3.485 | 21.6 (torch-compile-fp32) | 1.740 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| mamba1-forward | gaussian | 5.4 | 193.2 (torch-eager-bf16) | 0.028 | 150.9 (torch-eager-fp32) | 0.036 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| mamba2-forward | gaussian | 7.8 | 21.2 (torch-eager-bf16) | 0.371 | 25.9 (torch-eager-fp32) | 0.302 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| mamba3-forward | gaussian | 4.1 | 16.3 (torch-compile-bf16) | 0.251 | 12.4 (torch-compile-fp32) | 0.331 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| mlp-train-step | gaussian | 2.0 | 2.0 (torch-eager-bf16) | 1.021 | 1.5 (torch-eager-fp32) | 1.365 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| samba-forward | bytes | 5.0 | 5.5 (torch-compile-bf16) | 0.910 | 12.1 (torch-compile-fp32) | 0.413 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| samba-train-step | bytes | 42.4 | 165.9 (torch-eager-bf16) | 0.255 | 46.3 (torch-compile-fp32) | 0.915 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| transformer-forward | gaussian | 7.6 | 3.4 (torch-eager-bf16) | 2.207 | 3.6 (torch-eager-fp32) | 2.108 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |

## Trees

### et / istella (rows full, shape istella-2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v1008)`, ran on NVIDIA L40S (nv2, RunPod) job v1008

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@444f0b96e) | identical | 4917.3 | 4917.3..4917.3 | 1 | - | - | 4917.3 | - | - (stored whole) | - | - | - | - | auc=0.937987, logloss=0.189989 | - | main board, one scored run | - | ok (main@444f0b96e nv2/v1008 2026-10-09; identity vs the amd columns: n/a) |
| sklearn-et-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 16539.7 | 16539.7..16539.7 | 1 | 0.297 | - | 16539.7 | 16539.7 | 0.00 (cpu-arm) | 0.297 (whole/whole) | - | 24890.4 | - | auc=0.937659, logloss=0.190177 | yes | COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-et-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### et / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v1008)`, ran on NVIDIA L40S (nv2, RunPod) job v1008

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@444f0b96e) | identical | 4359.3 | 4359.3..4359.3 | 1 | - | - | 4359.3 | - | - (stored whole) | - | - | - | - | auc=0.618907, logloss=0.526142 | - | main board, one scored run | - | ok (main@444f0b96e nv2/v1008 2026-10-09; identity vs the amd columns: n/a) |
| sklearn-et-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 12501.5 | 12501.5..12501.5 | 1 | 0.349 | - | 12501.5 | 12501.5 | 0.00 (cpu-arm) | 0.349 (whole/whole) | - | 13627.4 | - | auc=0.619262, logloss=0.525951 | yes | COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-et-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-categorical / taxi (rows full, shape taxicat-4110786x16)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-grid-logs.txt (nv2/v0112)`, ran on NVIDIA L40S (nv2, RunPod) job v0112

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 7305.0 | 7305.0..7305.0 | 1 | - | - | 7305.0 | - | - (stored whole) | - | - | - | - | auc=0.631297, logloss=0.528285 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0112 2026-10-11; identity vs the amd columns: n/a) |
| catboost-gpu | catboost | gpu | NVIDIA L40S (host 24a11adce16e) | catboost 1.2.10 | opponent | 26444.4 | 26444.4..26444.4 | 1 | 0.276 | - | 26444.4 | - | - (stored whole) | 0.276 (whole/whole) | - | 4571.4 | 498.0 | auc=0.630387, logloss=0.528535 | yes | COMPARABLE | - | ok (copied from opponents-specific-20261006; measured this run) |
| xgboost-gpu | xgboost | gpu | NVIDIA L40S (host 24a11adce16e) | xgboost 3.2.0 | opponent | 13662.3 | 13662.3..13662.3 | 1 | 0.535 | - | 13662.3 | - | - (stored whole) | 0.535 (whole/whole) | - | 1886.0 | 498.0 | auc=0.631978, logloss=0.528548 | yes | COMPARABLE | - | ok (copied from opponents-specific-20261006; measured this run) |
| catboost-cpu | catboost | cpu | CPU AMD EPYC 9554 64-Core Processor (host 24a11adce16e, NVIDIA L40S box) | catboost 1.2.10 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | COMPARABLE | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (copied from opponents-specific-20261006; measured this run) |
| xgboost-cpu | xgboost | cpu | CPU AMD EPYC 9554 64-Core Processor (host 24a11adce16e, NVIDIA L40S box) | xgboost 3.2.0 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | COMPARABLE | - | REFUSED(GPU-PATH-ONLY: xgboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against ) (copied from opponents-specific-20261006; measured this run) |
| lightgbm-cpu | lightgbm | cpu | CPU AMD EPYC 9554 64-Core Processor (host 24a11adce16e, NVIDIA L40S box) | lightgbm 4.7.0 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | COMPARABLE | - | REFUSED(GPU-PATH-ONLY: lightgbm-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (copied from opponents-specific-20261006; measured this run) |
| lightgbm-cuda | lightgbm | gpu | NVIDIA L40S (host 24a11adce16e) | lightgbm 4.7.0 | opponent | 1636.8 | 1636.8..1636.8 | 1 | 4.463 | - | 1636.8 | - | - (stored whole) | 4.463 (whole/whole) | - | 1321.9 | 546.0 | auc=0.500000, logloss=0.554692 | yes | UNKNOWN | - | ok (copied from opponents-specific-20261006; measured this run) |

memory, ours, catboost-cpu, xgboost-cpu, lightgbm-cpu: host not sampled; GPU not sampled

memory, catboost-gpu, xgboost-gpu, lightgbm-cuda: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (binary task) (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-depthwise / istella (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v1007)`, ran on NVIDIA L40S (nv2, RunPod) job v1007

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@444f0b96e) | identical | 7232.6 | 7232.6..7232.6 | 1 | - | - | 7232.6 | - | - (stored whole) | - | - | - | - | auc=0.983304, logloss=0.156072 | - | main board, one scored run | - | ok (main@444f0b96e nv2/v1007 2026-10-09; identity vs the amd columns: n/a) |

settings: None. Rows: None. Timed: None.

mismatch: opponents withheld: the lane settings at this tree's HEAD differ from the settings opponents-default-20261006 recorded for its opponent race (an opponent job must score them again)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-depthwise / taxi (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v1007)`, ran on NVIDIA L40S (nv2, RunPod) job v1007

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@444f0b96e) | identical | 3692.4 | 3692.4..3692.4 | 1 | - | - | 3692.4 | - | - (stored whole) | - | - | - | - | auc=0.632205, logloss=0.527838 | - | main board, one scored run | - | ok (main@444f0b96e nv2/v1007 2026-10-09; identity vs the amd columns: n/a) |

settings: None. Rows: None. Timed: None.

mismatch: opponents withheld: the lane settings at this tree's HEAD differ from the settings opponents-specific-20261006 recorded for its opponent race (an opponent job must score them again)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-lossguide / istella (rows full, shape istella-2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-grid-logs.txt (nv2/v0112)`, ran on NVIDIA L40S (nv2, RunPod) job v0112

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 7437.9 | 7437.9..7437.9 | 1 | - | - | 7437.9 | - | - (stored whole) | - | - | - | - | auc=0.983586, logloss=0.150066 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0112 2026-10-11; identity vs the amd columns: n/a) |
| catboost-gpu | catboost | gpu | NVIDIA L40S (host cc560ebdaf91) | catboost 1.2.10 | opponent | 23630.6 | 23630.6..23630.6 | 1 | 0.315 | - | 23630.6 | - | - (stored whole) | 0.315 (whole/whole) | - | 5379.7 | 496.0 | auc=0.983668, logloss=0.149188 | yes | COMPARABLE | - | ok (copied from opponents-default-20261006; measured this run) |
| xgboost-gpu | xgboost | gpu | NVIDIA L40S (host cc560ebdaf91) | xgboost 3.2.0 | opponent | 13126.2 | 13126.2..13126.2 | 1 | 0.567 | - | 13126.2 | - | - (stored whole) | 0.567 (whole/whole) | - | 6475.4 | 496.0 | auc=0.983622, logloss=0.149263 | yes | COMPARABLE | - | ok (copied from opponents-default-20261006; measured this run) |
| catboost-cpu | catboost | cpu | CPU AMD EPYC 9554 64-Core Processor (host cc560ebdaf91, NVIDIA L40S box) | catboost 1.2.10 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | COMPARABLE | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (copied from opponents-default-20261006; measured this run) |
| xgboost-cpu | xgboost | cpu | CPU AMD EPYC 9554 64-Core Processor (host cc560ebdaf91, NVIDIA L40S box) | xgboost 3.2.0 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | COMPARABLE | - | REFUSED(GPU-PATH-ONLY: xgboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against ) (copied from opponents-default-20261006; measured this run) |
| lightgbm-cuda | lightgbm | gpu | NVIDIA L40S (host cc560ebdaf91) | lightgbm 4.7.0 | opponent | 4935.5 | 4935.5..4935.5 | 1 | 1.507 | - | 4935.5 | - | - (stored whole) | 1.507 (whole/whole) | - | 4205.8 | 544.0 | auc=0.500000, logloss=0.356515 | yes | UNKNOWN | - | ok (copied from opponents-default-20261006; measured this run) |
| lightgbm-cpu | lightgbm | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | lightgbm 4.7.0 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | COMPARABLE | - | REFUSED(GPU-PATH-ONLY: lightgbm-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (copied from release-board-resume-r2; measured this run) |

memory, ours, catboost-cpu, xgboost-cpu, lightgbm-cpu: host not sampled; GPU not sampled

memory, catboost-gpu, xgboost-gpu, lightgbm-cuda: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-lossguide / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-grid-logs.txt (nv2/v0112)`, ran on NVIDIA L40S (nv2, RunPod) job v0112

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 3743.9 | 3743.9..3743.9 | 1 | - | - | 3743.9 | - | - (stored whole) | - | - | - | - | auc=0.632096, logloss=0.528044 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0112 2026-10-11; identity vs the amd columns: n/a) |
| catboost-gpu | catboost | gpu | NVIDIA L40S (host 24a11adce16e) | catboost 1.2.10 | opponent | 15380.3 | 15380.3..15380.3 | 1 | 0.243 | - | 15380.3 | - | - (stored whole) | 0.243 (whole/whole) | - | 1670.4 | 496.0 | auc=0.631766, logloss=0.528065 | yes | NOT-COMPARABLE | - | ok (copied from opponents-specific-20261006; measured this run) |
| xgboost-gpu | xgboost | gpu | NVIDIA L40S (host 24a11adce16e) | xgboost 3.2.0 | opponent | 6565.0 | 6565.0..6565.0 | 1 | 0.570 | - | 6565.0 | - | - (stored whole) | 0.570 (whole/whole) | - | 1720.3 | 496.0 | auc=0.630969, logloss=0.528678 | yes | NOT-COMPARABLE | - | ok (copied from opponents-specific-20261006; measured this run) |
| catboost-cpu | catboost | cpu | CPU AMD EPYC 9554 64-Core Processor (host 24a11adce16e, NVIDIA L40S box) | catboost 1.2.10 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (copied from opponents-specific-20261006; measured this run) |
| xgboost-cpu | xgboost | cpu | CPU AMD EPYC 9554 64-Core Processor (host 24a11adce16e, NVIDIA L40S box) | xgboost 3.2.0 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: xgboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against ) (copied from opponents-specific-20261006; measured this run) |
| lightgbm-cpu | lightgbm | cpu | CPU AMD EPYC 9554 64-Core Processor (host 24a11adce16e, NVIDIA L40S box) | lightgbm 4.7.0 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: lightgbm-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (copied from opponents-specific-20261006; measured this run) |
| lightgbm-cuda | lightgbm | gpu | NVIDIA L40S (host 24a11adce16e) | lightgbm 4.7.0 | opponent | 1722.8 | 1722.8..1722.8 | 1 | 2.173 | - | 1722.8 | - | - (stored whole) | 2.173 (whole/whole) | - | 1314.0 | 544.0 | auc=0.500000, logloss=0.554692 | yes | UNKNOWN | - | ok (copied from opponents-specific-20261006; measured this run) |

memory, ours, catboost-cpu, xgboost-cpu, lightgbm-cpu: host not sampled; GPU not sampled

memory, catboost-gpu, xgboost-gpu, lightgbm-cuda: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-multiclass / istella (rows full, shape istellamc-2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-grid-logs.txt (nv/n0653)`, ran on NVIDIA L40S (nv, RunPod) job n0653

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@9499046fa) | identical | 14667.8 | 14667.8..14667.8 | 1 | - | - | 14667.8 | - | - (stored whole) | - | - | - | - | accuracy=0.907556, mlogloss=0.258413 | - | main board, one scored run | - | ok (main@9499046fa nv/n0653 2026-10-09; identity vs the amd columns: n/a) |
| catboost-gpu | catboost | gpu | NVIDIA L40S (host cc560ebdaf91) | catboost 1.2.10 | opponent | 15195.6 | 15195.6..15195.6 | 1 | 0.965 | - | 15195.6 | - | - (stored whole) | 0.965 (whole/whole) | - | 5364.3 | 558.0 | accuracy=0.907780, mlogloss=0.258149 | yes | NOT-COMPARABLE | - | ok (copied from opponents-default-20261006; measured this run) |
| xgboost-gpu | xgboost | gpu | NVIDIA L40S (host cc560ebdaf91) | xgboost 3.2.0 | opponent | 33556.0 | 33556.0..33556.0 | 1 | 0.437 | - | 33556.0 | - | - (stored whole) | 0.437 (whole/whole) | - | 6693.0 | 558.0 | accuracy=0.910140, mlogloss=0.246803 | yes | NOT-COMPARABLE | - | ok (copied from opponents-default-20261006; measured this run) |
| catboost-cpu | catboost | cpu | CPU AMD EPYC 9554 64-Core Processor (host cc560ebdaf91, NVIDIA L40S box) | catboost 1.2.10 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (copied from opponents-default-20261006; measured this run) |
| xgboost-cpu | xgboost | cpu | CPU AMD EPYC 9554 64-Core Processor (host cc560ebdaf91, NVIDIA L40S box) | xgboost 3.2.0 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: xgboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against ) (copied from opponents-default-20261006; measured this run) |
| lightgbm-cuda | lightgbm | gpu | NVIDIA L40S (host cc560ebdaf91) | lightgbm 4.7.0 | opponent | 170658.0 | 170658.0..170658.0 | 1 | 0.086 | - | 170658.0 | - | - (stored whole) | 0.086 (whole/whole) | - | 4394.5 | 936.0 | accuracy=0.911402, mlogloss=0.243459 | yes | UNKNOWN | - | ok (copied from opponents-default-20261006; measured this run) |
| lightgbm-cpu | lightgbm | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | lightgbm 4.7.0 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: lightgbm-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (copied from release-board-resume-r2; measured this run) |

memory, ours, catboost-cpu, xgboost-cpu, lightgbm-cpu: host not sampled; GPU not sampled

memory, catboost-gpu, xgboost-gpu, lightgbm-cuda: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (MULTICLASS task) (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-multiclass / taxi (rows full, shape taximc-4110786x16)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-grid-logs.txt (nv/n0653)`, ran on NVIDIA L40S (nv, RunPod) job n0653

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@9499046fa) | identical | 7044.5 | 7044.5..7044.5 | 1 | - | - | 7044.5 | - | - (stored whole) | - | - | - | - | accuracy=0.599380, mlogloss=1.012595 | - | main board, one scored run | - | ok (main@9499046fa nv/n0653 2026-10-09; identity vs the amd columns: n/a) |
| catboost-gpu | catboost | gpu | NVIDIA L40S (host 24a11adce16e) | catboost 1.2.10 | opponent | 10221.5 | 10221.5..10221.5 | 1 | 0.689 | - | 10221.5 | - | - (stored whole) | 0.689 (whole/whole) | - | 1717.7 | 610.0 | accuracy=0.599270, mlogloss=1.012704 | yes | NOT-COMPARABLE | - | ok (copied from opponents-specific-20261006; measured this run) |
| xgboost-gpu | xgboost | gpu | NVIDIA L40S (host 24a11adce16e) | xgboost 3.2.0 | opponent | 12711.7 | 12711.7..12711.7 | 1 | 0.554 | - | 12711.7 | - | - (stored whole) | 0.554 (whole/whole) | - | 1903.0 | 610.0 | accuracy=0.601200, mlogloss=1.005204 | yes | NOT-COMPARABLE | - | ok (copied from opponents-specific-20261006; measured this run) |
| catboost-cpu | catboost | cpu | CPU AMD EPYC 9554 64-Core Processor (host 24a11adce16e, NVIDIA L40S box) | catboost 1.2.10 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (copied from opponents-specific-20261006; measured this run) |
| xgboost-cpu | xgboost | cpu | CPU AMD EPYC 9554 64-Core Processor (host 24a11adce16e, NVIDIA L40S box) | xgboost 3.2.0 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: xgboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against ) (copied from opponents-specific-20261006; measured this run) |
| lightgbm-cpu | lightgbm | cpu | CPU AMD EPYC 9554 64-Core Processor (host 24a11adce16e, NVIDIA L40S box) | lightgbm 4.7.0 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: lightgbm-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (copied from opponents-specific-20261006; measured this run) |
| lightgbm-cuda | lightgbm | gpu | NVIDIA L40S (host 24a11adce16e) | lightgbm 4.7.0 | opponent | 76949.9 | 76949.9..76949.9 | 1 | 0.092 | - | 76949.9 | - | - (stored whole) | 0.092 (whole/whole) | - | 1457.7 | 856.0 | accuracy=0.601896, mlogloss=1.003173 | yes | UNKNOWN | - | ok (copied from opponents-specific-20261006; measured this run) |

memory, ours, catboost-cpu, xgboost-cpu, lightgbm-cpu: host not sampled; GPU not sampled

memory, catboost-gpu, xgboost-gpu, lightgbm-cuda: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (MULTICLASS task) (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-ordered / istella (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-grid-logs.txt (nv/n0653)`, ran on NVIDIA L40S (nv, RunPod) job n0653

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@9499046fa) | identical | 35370.0 | 35370.0..35370.0 | 1 | - | - | 35370.0 | - | - (stored whole) | - | - | - | - | auc=0.979444, logloss=0.190928 | - | main board, one scored run | - | ok (main@9499046fa nv/n0653 2026-10-09; identity vs the amd columns: n/a) |

settings: None. Rows: None. Timed: None.

mismatch: Ordered boosting on ours and CatBoost only: XGBoost and LightGBM have no ordered (permutation) boosting, so the lane races CatBoost alone, on the symmetric grower Ordered requires (catboost_options.cpp:757-759)

mismatch: each library draws its fold permutations from its own generator seeded 7; the streams are not shared, so the fits differ by construction

mismatch: approx_on_full_history: CatBoost CPU only (default False, refused on their GPU); ours restates their GPU Ordered boosting, which has no full-history arm, so both arms run the False behaviour

mismatch: opponents withheld: the lane settings at this tree's HEAD differ from the settings release-board-resume-r2 recorded for its opponent race (an opponent job must score them again)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-ordered / taxi (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-grid-logs.txt (nv/n0653)`, ran on NVIDIA L40S (nv, RunPod) job n0653

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@9499046fa) | identical | 19986.4 | 19986.4..19986.4 | 1 | - | - | 19986.4 | - | - (stored whole) | - | - | - | - | auc=0.629203, logloss=0.529007 | - | main board, one scored run | - | ok (main@9499046fa nv/n0653 2026-10-09; identity vs the amd columns: n/a) |

settings: None. Rows: None. Timed: None.

mismatch: Ordered boosting on ours and CatBoost only: XGBoost and LightGBM have no ordered (permutation) boosting, so the lane races CatBoost alone, on the symmetric grower Ordered requires (catboost_options.cpp:757-759)

mismatch: each library draws its fold permutations from its own generator seeded 7; the streams are not shared, so the fits differ by construction

mismatch: approx_on_full_history: CatBoost CPU only (default False, refused on their GPU); ours restates their GPU Ordered boosting, which has no full-history arm, so both arms run the False behaviour

mismatch: opponents withheld: the lane settings at this tree's HEAD differ from the settings release-board-resume-r2 recorded for its opponent race (an opponent job must score them again)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-rank-pairlogit / istella (rows full, shape istellarank-2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-grid-logs.txt (nv/n0653)`, ran on NVIDIA L40S (nv, RunPod) job n0653

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@9499046fa) | identical | 2768.8 | 2768.8..2768.8 | 1 | - | - | 2768.8 | - | - (stored whole) | - | - | - | - | map=0.854545, ndcg10=0.719953, ndcg5=0.650400 | - | main board, one scored run | - | ok (main@9499046fa nv/n0653 2026-10-09; identity vs the amd columns: n/a) |
| catboost-gpu | catboost | gpu | NVIDIA L40S (host cc560ebdaf91) | catboost 1.2.10 | opponent | 4241.3 | 4241.3..4241.3 | 1 | 0.653 | - | 4241.3 | - | - (stored whole) | 0.653 (whole/whole) | - | 6504.8 | 556.0 | map=0.853929, ndcg10=0.719621, ndcg5=0.650025 | yes | COMPARABLE | - | ok (copied from opponents-default-20261006; measured this run) |
| xgboost-gpu | xgboost | gpu | NVIDIA L40S (host cc560ebdaf91) | xgboost 3.2.0 | opponent | 5040.9 | 5040.9..5040.9 | 1 | 0.549 | - | 5040.9 | - | - (stored whole) | 0.549 (whole/whole) | - | 7300.1 | 556.0 | map=0.872796, ndcg10=0.738397, ndcg5=0.670093 | yes | COMPARABLE | - | ok (copied from opponents-default-20261006; measured this run) |
| catboost-cpu | catboost | cpu | CPU AMD EPYC 9554 64-Core Processor (host cc560ebdaf91, NVIDIA L40S box) | catboost 1.2.10 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | COMPARABLE | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (copied from opponents-default-20261006; measured this run) |
| xgboost-cpu | xgboost | cpu | CPU AMD EPYC 9554 64-Core Processor (host cc560ebdaf91, NVIDIA L40S box) | xgboost 3.2.0 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | COMPARABLE | - | REFUSED(GPU-PATH-ONLY: xgboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against ) (copied from opponents-default-20261006; measured this run) |

memory, ours, catboost-cpu, xgboost-cpu: host not sampled; GPU not sampled

memory, catboost-gpu, xgboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-symmetric / istella (rows full, shape istella-2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v1007)`, ran on NVIDIA L40S (nv2, RunPod) job v1007

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@444f0b96e) | identical | 6137.4 | 6137.4..6137.4 | 1 | - | - | 6137.4 | - | - (stored whole) | - | - | - | - | auc=0.980129, logloss=0.186686 | - | main board, one scored run | - | ok (main@444f0b96e nv2/v1007 2026-10-09; identity vs the amd columns: n/a) |
| catboost-gpu | catboost | gpu | NVIDIA L40S (host cc560ebdaf91) | catboost 1.2.10 | opponent | 8181.4 | 8181.4..8181.4 | 1 | 0.750 | - | 8181.4 | - | - (stored whole) | 0.750 (whole/whole) | - | 4878.1 | 428.0 | auc=0.980018, logloss=0.187292 | yes | UNKNOWN | - | ok (copied from opponents-default-20261006; measured this run) |
| catboost-cpu | catboost | cpu | CPU AMD EPYC 9554 64-Core Processor (host cc560ebdaf91, NVIDIA L40S box) | catboost 1.2.10 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | UNKNOWN | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (copied from opponents-default-20261006; measured this run) |

memory, ours, catboost-cpu: host not sampled; GPU not sampled

memory, catboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-symmetric / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v1007)`, ran on NVIDIA L40S (nv2, RunPod) job v1007

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@444f0b96e) | identical | 3503.2 | 3503.2..3503.2 | 1 | - | - | 3503.2 | - | - (stored whole) | - | - | - | - | auc=0.630436, logloss=0.528554 | - | main board, one scored run | - | ok (main@444f0b96e nv2/v1007 2026-10-09; identity vs the amd columns: n/a) |
| catboost-gpu | catboost | gpu | NVIDIA L40S (host 24a11adce16e) | catboost 1.2.10 | opponent | 5982.0 | 5982.0..5982.0 | 1 | 0.586 | - | 5982.0 | - | - (stored whole) | 0.586 (whole/whole) | - | 1127.4 | 428.0 | auc=0.630310, logloss=0.528616 | yes | UNKNOWN | - | ok (copied from opponents-specific-20261006; measured this run) |
| catboost-cpu | catboost | cpu | CPU AMD EPYC 9554 64-Core Processor (host 24a11adce16e, NVIDIA L40S box) | catboost 1.2.10 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | UNKNOWN | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (copied from opponents-specific-20261006; measured this run) |

memory, ours, catboost-cpu: host not sampled; GPU not sampled

memory, catboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### iforest / istella (rows full, shape istella-2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/full-gpu-repair-20261014/nvidia/measured/trees.iforest.istella.rows=full/board.json (nv2/paired-83ace7870a3360eea1242722a4088b24f8781fe86919d10c2376880967febf90)`, ran on b6c1fd6fb058

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 196.7 | 196.7..196.7 | 1 | - | - | 196.7 | - | - (stored whole) | - | - | 6265.1 | 4014.0 | auc=0.829336 | yes | main board, one scored run | wheel | ok (main@fcdfa0fe2 nv2/paired-83ace7870a3360eea1242722a4088b24f8781fe86919d10c2376880967febf90 2026-10-14; identity vs the amd columns: n/a) |
| cuml-iforest-gpu | cuml | gpu | NVIDIA L40S (host b6c1fd6fb058) | cuml 26.08.00 | opponent | 2305.2 | 2305.2..2305.2 | 1 | 0.085 | - | 2305.2 | - | - (stored whole) | 0.085 (whole/whole) | - | 6285.1 | 4014.0 | auc=0.830358 | yes | UNKNOWN | - | ok (measured this run) |

memory, ours, cuml-iforest-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-iforest-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| bootstrap | false | false |
| contamination | "auto" | "auto" |
| max_depth | null | null |
| max_features | 1.0 | 1.0 |
| max_samples | 256 | 256 |
| n_estimators | 100 | 100 |
| seed | 7 | 7 |
| warm_start | false | false |

accepted difference: cuml-iforest-gpu max_depth: ours None is the auto depth ceil(log2(max_samples)) = 8; sklearn has no max_depth parameter and fixes the same value

Inference (each arm predicts with its own model from the fit rounds above):

| arm | hardware | version | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | NVIDIA L40S (nv2, RunPod) | 0.8.44 | test | 500000 | 57.3 | 57.3..57.3 | 1 | - | - | 57.3 | - | - (stored whole) | - | - | auc=0.829336, auc_matches_fit=True | yes | UNKNOWN | ok |
| cuml-iforest-gpu | NVIDIA L40S (host b6c1fd6fb058) | cuml 26.08.00 | test | 500000 | 77.0 | 77.0..77.0 | 1 | 0.744 | - | 77.0 | - | - (stored whole) | 0.744 (whole/whole) | - | auc=0.830358, auc_matches_fit=True | yes | UNKNOWN | ok |
| mojolearn IDENTICAL | NVIDIA L40S (nv2, RunPod) | 0.8.44 | large | 1000000 | 121.6 | 121.6..121.6 | 1 | - | - | 121.6 | - | - (stored whole) | - | - | - | yes | UNKNOWN | ok |
| cuml-iforest-gpu | NVIDIA L40S (host b6c1fd6fb058) | cuml 26.08.00 | large | 1000000 | 156.8 | 156.8..156.8 | 1 | 0.775 | - | 156.8 | - | - (stored whole) | 0.775 (whole/whole) | - | - | yes | UNKNOWN | ok |

inference call, ours: mojolearn IsolationForest.score_samples(X) (the forest is rebuilt inside every scoring call, DEVIATION 874, so this clock includes a forest build)

inference call, cuml-iforest-gpu: cuml IsolationForest.score_samples(host X)

### iforest / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/full-gpu-repair-20261014/nvidia/measured/trees.iforest.taxi.rows=full/board.json (nv2/paired-c0fd96307852be0b01180e533be5e061bea825b0b5ba6e3faed3ff54d95a0f74)`, ran on b6c1fd6fb058

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 148.7 | 148.7..148.7 | 1 | - | - | 148.7 | - | - (stored whole) | - | - | 2634.6 | 942.0 | auc=0.547466 | yes | main board, one scored run | wheel | ok (main@fcdfa0fe2 nv2/paired-c0fd96307852be0b01180e533be5e061bea825b0b5ba6e3faed3ff54d95a0f74 2026-10-14; identity vs the amd columns: n/a) |
| cuml-iforest-gpu | cuml | gpu | NVIDIA L40S (host b6c1fd6fb058) | cuml 26.08.00 | opponent | 115.1 | 115.1..115.1 | 1 | 1.292 | - | 115.1 | - | - (stored whole) | 1.292 (whole/whole) | - | 2655.6 | 942.0 | auc=0.551846 | yes | UNKNOWN | - | ok (measured this run) |

memory, ours, cuml-iforest-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-iforest-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| bootstrap | false | false |
| contamination | "auto" | "auto" |
| max_depth | null | null |
| max_features | 1.0 | 1.0 |
| max_samples | 256 | 256 |
| n_estimators | 100 | 100 |
| seed | 7 | 7 |
| warm_start | false | false |

accepted difference: cuml-iforest-gpu max_depth: ours None is the auto depth ceil(log2(max_samples)) = 8; sklearn has no max_depth parameter and fixes the same value

Inference (each arm predicts with its own model from the fit rounds above):

| arm | hardware | version | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | NVIDIA L40S (nv2, RunPod) | 0.8.44 | test | 500000 | 2.7 | 2.7..2.7 | 1 | - | - | 2.7 | - | - (stored whole) | - | - | auc=0.547466, auc_matches_fit=True | yes | UNKNOWN | ok |
| cuml-iforest-gpu | NVIDIA L40S (host b6c1fd6fb058) | cuml 26.08.00 | test | 500000 | 5.6 | 5.6..5.6 | 1 | 0.481 | - | 5.6 | - | - (stored whole) | 0.481 (whole/whole) | - | auc=0.551846, auc_matches_fit=True | yes | UNKNOWN | ok |
| mojolearn IDENTICAL | NVIDIA L40S (nv2, RunPod) | 0.8.44 | large | 1000000 | 4.8 | 4.8..4.8 | 1 | - | - | 4.8 | - | - (stored whole) | - | - | - | yes | UNKNOWN | ok |
| cuml-iforest-gpu | NVIDIA L40S (host b6c1fd6fb058) | cuml 26.08.00 | large | 1000000 | 9.4 | 9.4..9.4 | 1 | 0.511 | - | 9.4 | - | - (stored whole) | 0.511 (whole/whole) | - | - | yes | UNKNOWN | ok |

inference call, ours: mojolearn IsolationForest.score_samples(X) (the forest is rebuilt inside every scoring call, DEVIATION 874, so this clock includes a forest build)

inference call, cuml-iforest-gpu: cuml IsolationForest.score_samples(host X)

### rf / istella (rows full, shape istella-2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v1008)`, ran on NVIDIA L40S (nv2, RunPod) job v1008

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@444f0b96e) | identical | 30726.6 | 30726.6..30726.6 | 1 | - | - | 30726.6 | - | - (stored whole) | - | - | - | - | auc=0.945385, logloss=0.182017 | - | main board, one scored run | - | ok (main@444f0b96e nv2/v1008 2026-10-09; identity vs the amd columns: n/a) |
| cuml-rf-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 24023.2 | 24023.2..24023.2 | 1 | 1.279 | - | 24023.2 | - | - (stored whole) | 1.279 (whole/whole) | - | 5222.5 | 444.0 | auc=0.945348, logloss=0.182069 | yes | UNKNOWN | - | ok (copied from opponents-default-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-rf-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

config: NVIDIA gbm-bench, skrf/cumlrf: max_depth 8, n_estimators 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### rf / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v1008)`, ran on NVIDIA L40S (nv2, RunPod) job v1008

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@444f0b96e) | identical | 9400.8 | 9400.8..9400.8 | 1 | - | - | 9400.8 | - | - (stored whole) | - | - | - | - | auc=0.617838, logloss=0.525953 | - | main board, one scored run | - | ok (main@444f0b96e nv2/v1008 2026-10-09; identity vs the amd columns: n/a) |
| cuml-rf-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 20651.0 | 20651.0..20651.0 | 1 | 0.455 | - | 20651.0 | - | - (stored whole) | 0.455 (whole/whole) | - | 1601.6 | 444.0 | auc=0.617836, logloss=0.525954 | yes | UNKNOWN | - | ok (copied from opponents-specific-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-rf-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

config: NVIDIA gbm-bench, skrf/cumlrf: max_depth 8, n_estimators 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

## Classical

### dbscan / istella (rows full, shape 1000000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/nvidia-wave8/campaign/measured/classical.dbscan.istella.rows=full/board.json (nv2/paired-f639cdf6bf5a6bc70391e70b26a0dda9180b3a22234025684f6dbca7fd4d66c2)`, ran on d2fe567d3b88

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 601944.2 | 601944.2..601944.2 | 1 | - | - | 601944.2 | 601907.3 | 36.82 (upload_ms_separate) | - | - | 5641.3 | 3498.0 | n_clusters=40131, noise_fraction=0.219391, rows=1000000 | yes | main board, one scored run | wheel | ok (main@50be53955 nv2/paired-f639cdf6bf5a6bc70391e70b26a0dda9180b3a22234025684f6dbca7fd4d66c2 2026-10-14; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host d2fe567d3b88) | cuml 26.08.00 | opponent | 62967.2 | 62967.2..62967.2 | 1 | 9.560 | - | 63588.8 | 62967.2 | 621.60 (upload_ms_untimed) | 9.466 (whole/whole) | - | 2649.1 | 1932.0 | ari_vs_ours=1.000000, n_clusters=40131, noise_agreement_vs_ours=1.000000, noise_fraction=0.219391, rows=1000000 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: eps=3, min_samples=2 (the cuML benchmark's DBSCAN) on every arm; metric='euclidean'. Rows: dbscan block: 1,000,000 rows, standardized. Timed: fit.

mismatch: seed: no arm has a seed argument (deterministic)

mismatch: algorithm='brute' on every raced arm (ours, cuml-gpu, scikit-learn: the cuML benchmark's cpu_args); ours' default 'rbc' is not raced, and cuml-gpu-rbc (overflows at 1,000,000 rows) is not on the roster

mismatch: leaf_size=30 and n_jobs=-1: scikit-learn only

config: cuML benchmark (RAPIDS), DBSCAN (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| algorithm | "brute" | "brute" |
| calc_core_sample_indices | true | - |
| eps | 3.0 | 3.0 |
| max_iterations | - | null |
| max_mbytes_per_batch | null | null |
| metric | "euclidean" | "euclidean" |
| min_samples | 2 | 2 |
| prediction_data | - | false |
| seed | "none (deterministic)" | "none (deterministic)" |

accepted difference: cuml-gpu calc_core_sample_indices: our DBSCAN always produces core_sample_indices_ and components_; cuML is explicitly set to retain both

accepted difference: cuml-gpu max_iterations: both propagate labels to the fixed point; cuML exposes no iteration cap, and our None requests completion

accepted difference: cuml-gpu max_mbytes_per_batch: both libraries choose distance-batch memory automatically; neither setting subsamples the input

accepted difference: cuml-gpu prediction_data: no additional prediction cache is requested; cuML DBSCAN exposes no prediction-data option

### dbscan / taxi (rows full, shape 1000000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/nvidia-wave8/campaign/measured/classical.dbscan.taxi.rows=full/board.json (nv2/paired-4cb0d8fec346b3017270f76c4ec43b4576ae05ad9fe23131be60311b6c5a7ac4)`, ran on d2fe567d3b88

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 943355.8 | 943355.8..943355.8 | 1 | - | - | 943355.8 | 943353.9 | 1.89 (upload_ms_separate) | - | - | 1783.5 | 7338.0 | n_clusters=36, noise_fraction=0.000174, rows=1000000 | yes | main board, one scored run | wheel | ok (main@50be53955 nv2/paired-4cb0d8fec346b3017270f76c4ec43b4576ae05ad9fe23131be60311b6c5a7ac4 2026-10-14; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host d2fe567d3b88) | cuml 26.08.00 | opponent | 440513.6 | 440513.6..440513.6 | 1 | 2.141 | - | 440554.7 | 440513.6 | 41.03 (upload_ms_untimed) | 2.141 (whole/whole) | - | 893.1 | 520.0 | ari_vs_ours=1.000000, n_clusters=36, noise_agreement_vs_ours=1.000000, noise_fraction=0.000174, rows=1000000 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: eps=3, min_samples=2 (the cuML benchmark's DBSCAN) on every arm; metric='euclidean'. Rows: dbscan block: 1,000,000 rows, standardized. Timed: fit.

mismatch: seed: no arm has a seed argument (deterministic)

mismatch: algorithm='brute' on every raced arm (ours, cuml-gpu, scikit-learn: the cuML benchmark's cpu_args); ours' default 'rbc' is not raced, and cuml-gpu-rbc (overflows at 1,000,000 rows) is not on the roster

mismatch: leaf_size=30 and n_jobs=-1: scikit-learn only

config: cuML benchmark (RAPIDS), DBSCAN (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| algorithm | "brute" | "brute" |
| calc_core_sample_indices | true | - |
| eps | 3.0 | 3.0 |
| max_iterations | - | null |
| max_mbytes_per_batch | null | null |
| metric | "euclidean" | "euclidean" |
| min_samples | 2 | 2 |
| prediction_data | - | false |
| seed | "none (deterministic)" | "none (deterministic)" |

accepted difference: cuml-gpu calc_core_sample_indices: our DBSCAN always produces core_sample_indices_ and components_; cuML is explicitly set to retain both

accepted difference: cuml-gpu max_iterations: both propagate labels to the fixed point; cuML exposes no iteration cap, and our None requests completion

accepted difference: cuml-gpu max_mbytes_per_batch: both libraries choose distance-batch memory automatically; neither setting subsamples the input

accepted difference: cuml-gpu prediction_data: no additional prediction cache is requested; cuML DBSCAN exposes no prediction-data option

### hdbscan / istella (rows full, shape 100000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/nvidia-wave8/campaign/measured/classical.hdbscan.istella.rows=full/board.json (nv2/paired-44d04846d956a42f366cbe28267320efdaa0d4dbf6f0efcdc0c7eed23d4dc7b2)`, ran on d2fe567d3b88

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 1166.5 | 1166.5..1166.5 | 1 | - | - | 1166.5 | 1162.8 | 3.63 (upload_ms_separate) | - | - | 2438.4 | 1194.0 | n_clusters=47, noise_fraction=0.253810, rows=100000, silhouette=-0.003856, silhouette_library=sklearn.metrics.silhouette_score, silhouette_library_version=1.7.2, silhouette_scope=all consumed rows; noise is a label group; no sampling | yes | main board, one scored run | wheel | ok (main@50be53955 nv2/paired-44d04846d956a42f366cbe28267320efdaa0d4dbf6f0efcdc0c7eed23d4dc7b2 2026-10-14; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host d2fe567d3b88) | cuml 26.08.00 | opponent | 805.7 | 805.7..805.7 | 1 | 1.448 | - | 889.6 | 805.7 | 83.86 (upload_ms_untimed) | 1.311 (whole/whole) | - | 1865.3 | 526.0 | ari_vs_ours=0.857120, n_clusters=53, noise_agreement_vs_ours=0.940610, noise_fraction=0.256280, rows=100000, silhouette=-0.018228, silhouette_library=sklearn.metrics.silhouette_score, silhouette_library_version=1.7.2, silhouette_scope=all consumed rows; noise is a label group; no sampling | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: min_samples=10 (scikit-learn 11: the same core distance, the 10th neighbour besides the point), min_cluster_size=100, metric='euclidean', cluster_selection_method='eom', cluster_selection_epsilon=0.0, alpha=1.0, allow_single_cluster=False. Rows: the dbscan block's first 100,000 rows. Timed: fit.

mismatch: seed: no arm has a seed argument (deterministic)

mismatch: max_cluster_size: ours and cuML 0, scikit-learn None (both mean no limit)

mismatch: min_samples: ours and cuML 10, scikit-learn 11. The SAME k-th neighbour: cuML's runner.h:68-80 (ours transcribes it) runs the k-NN at min_samples + 1 including the point itself; scikit-learn's kneighbors(X, min_samples) counts the point itself (its HDBSCAN Notes say so). tools/bench_board_params.py maps both to one canonical value

mismatch: k-NN graph: exact brute force on every arm; cuML build_algo='brute_force' pinned (as ours); scikit-learn algorithm='brute' (FAIRNESS FIX 2026-10-13: it was 'auto', a KD/ball tree), leaf_size=40 unused, n_jobs=-1

config: cuML benchmark (RAPIDS), HDBSCAN (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| allow_single_cluster | false | false |
| alpha | 1.0 | 1.0 |
| build_algo | "brute_force" | - |
| build_kwds | null | - |
| cluster_selection_epsilon | 0.0 | 0.0 |
| cluster_selection_method | "eom" | "eom" |
| device_ids | null | - |
| gen_min_span_tree | false | - |
| max_cluster_size | 0 | 0 |
| metric | "euclidean" | "euclidean" |
| min_cluster_size | 100 | 100 |
| min_samples | 10 | 10 |
| p | null | - |
| prediction_data | false | false |
| seed | "none (deterministic)" | "none (deterministic)" |

accepted difference: cuml-gpu build_algo: our HDBSCAN fixes exhaustive neighbors; cuML explicitly selects the same exact brute-force graph construction

accepted difference: cuml-gpu build_kwds: cuML's exact graph uses the default unpartitioned single-device path; no approximate or multi-device overrides are requested

accepted difference: cuml-gpu device_ids: cuML's exact graph uses the default unpartitioned single-device path; no approximate or multi-device overrides are requested

accepted difference: cuml-gpu gen_min_span_tree: both construct the internal MST for clustering; neither requests an additional exported minimum-spanning-tree object

accepted difference: cuml-gpu p: p only parameterizes Minkowski distance; both arms explicitly use Euclidean distance

### hdbscan / taxi (rows full, shape 100000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/nvidia-wave8/campaign/measured/classical.hdbscan.taxi.rows=full/board.json (nv2/paired-2b487df5a61c13ee70661a7f7e7a89376cc4ccff20b465e01743068fe9a2f4ad)`, ran on d2fe567d3b88

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 407.7 | 407.7..407.7 | 1 | - | - | 407.7 | 407.4 | 0.27 (upload_ms_separate) | - | - | 1550.0 | 682.0 | n_clusters=161, noise_fraction=0.140550, rows=100000, silhouette=0.001505, silhouette_library=sklearn.metrics.silhouette_score, silhouette_library_version=1.7.2, silhouette_scope=all consumed rows; noise is a label group; no sampling | yes | main board, one scored run | wheel | ok (main@50be53955 nv2/paired-2b487df5a61c13ee70661a7f7e7a89376cc4ccff20b465e01743068fe9a2f4ad 2026-10-14; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host d2fe567d3b88) | cuml 26.08.00 | opponent | 358.4 | 358.4..358.4 | 1 | 1.138 | - | 366.1 | 358.4 | 7.69 (upload_ms_untimed) | 1.114 (whole/whole) | - | 954.1 | 448.0 | ari_vs_ours=0.917885, n_clusters=159, noise_agreement_vs_ours=0.984420, noise_fraction=0.130970, rows=100000, silhouette=0.011960, silhouette_library=sklearn.metrics.silhouette_score, silhouette_library_version=1.7.2, silhouette_scope=all consumed rows; noise is a label group; no sampling | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: min_samples=10 (scikit-learn 11: the same core distance, the 10th neighbour besides the point), min_cluster_size=100, metric='euclidean', cluster_selection_method='eom', cluster_selection_epsilon=0.0, alpha=1.0, allow_single_cluster=False. Rows: the dbscan block's first 100,000 rows. Timed: fit.

mismatch: seed: no arm has a seed argument (deterministic)

mismatch: max_cluster_size: ours and cuML 0, scikit-learn None (both mean no limit)

mismatch: min_samples: ours and cuML 10, scikit-learn 11. The SAME k-th neighbour: cuML's runner.h:68-80 (ours transcribes it) runs the k-NN at min_samples + 1 including the point itself; scikit-learn's kneighbors(X, min_samples) counts the point itself (its HDBSCAN Notes say so). tools/bench_board_params.py maps both to one canonical value

mismatch: k-NN graph: exact brute force on every arm; cuML build_algo='brute_force' pinned (as ours); scikit-learn algorithm='brute' (FAIRNESS FIX 2026-10-13: it was 'auto', a KD/ball tree), leaf_size=40 unused, n_jobs=-1

config: cuML benchmark (RAPIDS), HDBSCAN (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| allow_single_cluster | false | false |
| alpha | 1.0 | 1.0 |
| build_algo | "brute_force" | - |
| build_kwds | null | - |
| cluster_selection_epsilon | 0.0 | 0.0 |
| cluster_selection_method | "eom" | "eom" |
| device_ids | null | - |
| gen_min_span_tree | false | - |
| max_cluster_size | 0 | 0 |
| metric | "euclidean" | "euclidean" |
| min_cluster_size | 100 | 100 |
| min_samples | 10 | 10 |
| p | null | - |
| prediction_data | false | false |
| seed | "none (deterministic)" | "none (deterministic)" |

accepted difference: cuml-gpu build_algo: our HDBSCAN fixes exhaustive neighbors; cuML explicitly selects the same exact brute-force graph construction

accepted difference: cuml-gpu build_kwds: cuML's exact graph uses the default unpartitioned single-device path; no approximate or multi-device overrides are requested

accepted difference: cuml-gpu device_ids: cuML's exact graph uses the default unpartitioned single-device path; no approximate or multi-device overrides are requested

accepted difference: cuml-gpu gen_min_span_tree: both construct the internal MST for clustering; neither requests an additional exported minimum-spanning-tree object

accepted difference: cuml-gpu p: p only parameterizes Minkowski distance; both arms explicitly use Euclidean distance

### kde / istella (rows full, shape 100000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/nvidia-wave9/campaign/measured/classical.kde.istella.rows=full/board.json (nv2/paired-74874a6f69764526d0f26e0d552b482006b953eb6802a3a01065f9c1fdec872b)`, ran on d2fe567d3b88

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 33.9 | 33.9..33.9 | 1 | - | - | 33.9 | 30.1 | 3.78 (upload_ms_separate) | - | - | 1579.7 | 682.0 | mean_log_likelihood=-222.270583, rows_without_density=0 | yes | main board, one scored run | wheel | ok (main@2b9a0ea93 nv2/paired-74874a6f69764526d0f26e0d552b482006b953eb6802a3a01065f9c1fdec872b 2026-10-14; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host d2fe567d3b88) | cuml 26.08.00 | opponent | 6.5 | 6.5..6.5 | 1 | 5.180 | - | 78.7 | 6.5 | 72.11 (upload_ms_untimed) | 0.431 (whole/whole) | - | 1001.9 | 514.0 | mean_log_likelihood=-222.270582, rows_without_density=0 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: bandwidth=1.0, kernel='gaussian' (the cuML benchmark's KernelDensity), metric='euclidean' on every arm; ours and scikit-learn atol=0, rtol=0, algorithm='auto', leaf_size=40, breadth_first=True. Rows: kde block: 100,000 fit rows, 2,000 queries, standardized. Timed: score_samples; the fit is before the clock on every arm.

mismatch: seed: no arm has a seed argument (exact density)

mismatch: cuML has no atol, rtol, algorithm, leaf_size or breadth_first (exact brute force)

config: cuML benchmark (RAPIDS), KernelDensity (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| algorithm | - | "auto" |
| atol | - | 0.0 |
| bandwidth | 1.0 | 1.0 |
| breadth_first | - | true |
| kernel | "gaussian" | "gaussian" |
| leaf_size | - | 40 |
| metric | "euclidean" | "euclidean" |
| metric_params | null | null |
| rtol | - | 0.0 |
| seed | "none (deterministic)" | "none (deterministic)" |

accepted difference: cuml-gpu algorithm: both implementations use exact brute-force KDE; ours refuses nondefault tree controls and cuML exposes no tree (density.py; kernel_density.pyx)

accepted difference: cuml-gpu atol: both implementations use exact brute-force KDE; ours refuses nondefault tree controls and cuML exposes no tree (density.py; kernel_density.pyx)

accepted difference: cuml-gpu breadth_first: both implementations use exact brute-force KDE; ours refuses nondefault tree controls and cuML exposes no tree (density.py; kernel_density.pyx)

accepted difference: cuml-gpu leaf_size: both implementations use exact brute-force KDE; ours refuses nondefault tree controls and cuML exposes no tree (density.py; kernel_density.pyx)

accepted difference: cuml-gpu metric_params: None means no metric keyword overrides, p=2 on both implementations (density.py; kernel_density.pyx)

accepted difference: cuml-gpu rtol: both implementations use exact brute-force KDE; ours refuses nondefault tree controls and cuML exposes no tree (density.py; kernel_density.pyx)

### kde / taxi (rows full, shape 100000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/nvidia-wave9/campaign/measured/classical.kde.taxi.rows=full/board.json (nv2/paired-dd01d8d9c31ecfbd85294b1ca8cdc4220444b69efd5aee3d3009af8c2786509a)`, ran on d2fe567d3b88

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 5.7 | 5.7..5.7 | 1 | - | - | 5.7 | 5.4 | 0.29 (upload_ms_separate) | - | - | 1499.0 | 682.0 | mean_log_likelihood=-14.826436, rows_without_density=0 | yes | main board, one scored run | wheel | ok (main@2b9a0ea93 nv2/paired-dd01d8d9c31ecfbd85294b1ca8cdc4220444b69efd5aee3d3009af8c2786509a 2026-10-14; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host d2fe567d3b88) | cuml 26.08.00 | opponent | 2.0 | 2.0..2.0 | 1 | 2.930 | - | 10.6 | 2.0 | 8.69 (upload_ms_untimed) | 0.538 (whole/whole) | - | 800.7 | 436.0 | mean_log_likelihood=-14.826437, rows_without_density=0 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: bandwidth=1.0, kernel='gaussian' (the cuML benchmark's KernelDensity), metric='euclidean' on every arm; ours and scikit-learn atol=0, rtol=0, algorithm='auto', leaf_size=40, breadth_first=True. Rows: kde block: 100,000 fit rows, 2,000 queries, standardized. Timed: score_samples; the fit is before the clock on every arm.

mismatch: seed: no arm has a seed argument (exact density)

mismatch: cuML has no atol, rtol, algorithm, leaf_size or breadth_first (exact brute force)

config: cuML benchmark (RAPIDS), KernelDensity (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| algorithm | - | "auto" |
| atol | - | 0.0 |
| bandwidth | 1.0 | 1.0 |
| breadth_first | - | true |
| kernel | "gaussian" | "gaussian" |
| leaf_size | - | 40 |
| metric | "euclidean" | "euclidean" |
| metric_params | null | null |
| rtol | - | 0.0 |
| seed | "none (deterministic)" | "none (deterministic)" |

accepted difference: cuml-gpu algorithm: both implementations use exact brute-force KDE; ours refuses nondefault tree controls and cuML exposes no tree (density.py; kernel_density.pyx)

accepted difference: cuml-gpu atol: both implementations use exact brute-force KDE; ours refuses nondefault tree controls and cuML exposes no tree (density.py; kernel_density.pyx)

accepted difference: cuml-gpu breadth_first: both implementations use exact brute-force KDE; ours refuses nondefault tree controls and cuML exposes no tree (density.py; kernel_density.pyx)

accepted difference: cuml-gpu leaf_size: both implementations use exact brute-force KDE; ours refuses nondefault tree controls and cuML exposes no tree (density.py; kernel_density.pyx)

accepted difference: cuml-gpu metric_params: None means no metric keyword overrides, p=2 on both implementations (density.py; kernel_density.pyx)

accepted difference: cuml-gpu rtol: both implementations use exact brute-force KDE; ours refuses nondefault tree controls and cuML exposes no tree (density.py; kernel_density.pyx)

### kmeans / istella (rows full, shape 2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/nvidia-wave9/campaign/measured/classical.kmeans.istella.rows=full/board.json (nv2/paired-bd16817332a31b839f5ed9b31d7d0fbac7e80ca784627db1c6f1fbebb04ea32b)`, ran on d2fe567d3b88

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 463.7 | 463.7..463.7 | 1 | - | - | 463.7 | 377.3 | 86.38 (upload_ms_separate) | - | - | 3658.0 | 2474.0 | inertia=6.051e+17, inertia_over_ours=1.000000, n_iter=33, n_iter_over_ours=1.000000 | yes | main board, one scored run | wheel | ok (main@2b9a0ea93 nv2/paired-bd16817332a31b839f5ed9b31d7d0fbac7e80ca784627db1c6f1fbebb04ea32b 2026-10-14; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host d2fe567d3b88) | cuml 26.08.00 | opponent | 545.1 | 545.1..545.1 | 1 | 0.851 | - | 1738.7 | 545.1 | 1193.60 (upload_ms_untimed) | 0.267 (whole/whole) | - | 4969.7 | 2154.0 | inertia=6.111e+17, inertia_over_ours=1.009912, n_iter=21, n_iter_over_ours=0.636364 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: n_clusters=8, init='k-means++', max_iter=300, n_init=1 (the cuML benchmark's KMeans), oversampling_factor=0 on ours and cuML (its cuml_args; the classic sequential seeding), tol=1e-7 (an absolute squared center shift on every arm; scikit-learn's relative tol rescaled to it), metric='euclidean', Lloyd; seed 7 (ours, scikit-learn and cuML random_state=7). Rows: big block: 4,000,000 taxi rows or the Istella-S train split, raw. Timed: fit (the k-means++ seeding included on every arm).

mismatch: k-means++: each library draws its own start from its own generator seeded 7, so the starts differ and so may n_iter; scikit-learn greedy k-means++ (2 + log k candidates per center), ours and cuML the sequential k-means++ of cuML (oversampling_factor=0)

mismatch: stopping rule, the same on every arm: total squared center shift below 1e-7 (absolute) or max_iter=300. ours and cuML tol=1e-7 is that threshold (cuVS kmeans.cuh sqrdNormError < tol; inertia_check off); scikit-learn's tol is relative to mean(var(X)), so it gets 1e-7 / mean(var(X)) of the same rows (the same absolute threshold; it also stops when no label changes, a zero shift). n_iter and n_iter_over_ours are in every quality cell

mismatch: algorithm: scikit-learn algorithm='lloyd'; ours and cuML have no such parameter (Lloyd)

mismatch: torch-gpu: refused, not an opponent (the harness's own k-means from torch ops); hipvs-gpu (AMD): hipVS cuvs.cluster.kmeans, init_method='KMeansPlusPlus' with oversampling_factor=0, tol=1e-7 under cuVS's rule, squared L2, no seed argument

config: cuML benchmark (RAPIDS), KMeans (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| device_buffer_samples | 0 | - |
| init | "k-means++" | "k-means++" |
| init_centroids | - | null |
| init_size | 0 | - |
| max_iter | 300 | 300 |
| max_samples_per_batch | 32768 | - |
| metric | - | "euclidean" |
| n_clusters | 8 | 8 |
| n_init | 1 | 1 |
| oversampling_factor | 0.0 | 0.0 |
| seed | 7 | 7 |
| tol | 1e-07 | 1e-07 |

accepted difference: cuml-gpu device_buffer_samples: cuML device_buffer_samples=0 uses the full in-memory dataset; init_size only applies to its out-of-core path (kmeans.pyx)

accepted difference: cuml-gpu init_centroids: no supplied centroids; both use the separately compared init method (cluster.py; kmeans.pyx)

accepted difference: cuml-gpu init_size: cuML device_buffer_samples=0 uses the full in-memory dataset; init_size only applies to its out-of-core path (kmeans.pyx)

accepted difference: cuml-gpu max_samples_per_batch: cuML distance-computation batch size partitions all rows without subsampling; library implementation tiling is retained, not a model setting (kmeans.pyx)

accepted difference: cuml-gpu metric: cuML KMeans fixes DistanceType.L2Expanded, matching our squared Euclidean objective (kmeans.pyx set_params)

Inference (each arm predicts with its own model from the fit rounds above):

| arm | hardware | version | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | NVIDIA L40S (nv2, RunPod) | 0.8.44 | Xq | 500000 | 38.4 | 38.4..38.4 | 1 | - | - | 38.4 | - | - (stored whole) | - | - | eval_inertia=1.417e+17, label_agreement_own_centers=1.000000 | yes | LIKE-FOR-LIKE-SPAN | ok |
| cuml-gpu | NVIDIA L40S (host d2fe567d3b88) | cuml 26.08.00 | Xq | 500000 | 5.2 | 5.2..5.2 | 1 | 7.339 | - | 327.5 | 5.2 | 322.29 (upload_ms_untimed (the predict rows' own upload)) | 0.117 (whole/whole) | - | agreement_vs_ours=0.766428, bits_equal_vs_ours=False, eval_inertia=1.442e+17, label_agreement_own_centers=1.000000 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: mojolearn KMeans.predict(Xq), host rows in, host result out

inference call, cuml-gpu: cuml KMeans.predict(Xq on the device, output_type cupy); Xq uploaded before the clock, which ends at the device synchronize

### kmeans / taxi (rows full, shape 4000000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/nvidia-wave9/campaign/measured/classical.kmeans.taxi.rows=full/board.json (nv2/paired-34357c4a00f228184bf4b07300b8850bb469acf7d66949e2daa89b1a4a188951)`, ran on d2fe567d3b88

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 185.7 | 185.7..185.7 | 1 | - | - | 185.7 | 178.3 | 7.42 (upload_ms_separate) | - | - | 1741.4 | 1194.0 | inertia=3.093e+08, inertia_over_ours=1.000000, n_iter=91, n_iter_over_ours=1.000000 | yes | main board, one scored run | wheel | ok (main@2b9a0ea93 nv2/paired-34357c4a00f228184bf4b07300b8850bb469acf7d66949e2daa89b1a4a188951 2026-10-14; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host d2fe567d3b88) | cuml 26.08.00 | opponent | 334.2 | 334.2..334.2 | 1 | 0.556 | - | 499.0 | 334.2 | 164.85 (upload_ms_untimed) | 0.372 (whole/whole) | - | 1254.5 | 614.0 | inertia=3.06e+08, inertia_over_ours=0.989330, n_iter=33, n_iter_over_ours=0.362637 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: n_clusters=8, init='k-means++', max_iter=300, n_init=1 (the cuML benchmark's KMeans), oversampling_factor=0 on ours and cuML (its cuml_args; the classic sequential seeding), tol=1e-7 (an absolute squared center shift on every arm; scikit-learn's relative tol rescaled to it), metric='euclidean', Lloyd; seed 7 (ours, scikit-learn and cuML random_state=7). Rows: big block: 4,000,000 taxi rows or the Istella-S train split, raw. Timed: fit (the k-means++ seeding included on every arm).

mismatch: k-means++: each library draws its own start from its own generator seeded 7, so the starts differ and so may n_iter; scikit-learn greedy k-means++ (2 + log k candidates per center), ours and cuML the sequential k-means++ of cuML (oversampling_factor=0)

mismatch: stopping rule, the same on every arm: total squared center shift below 1e-7 (absolute) or max_iter=300. ours and cuML tol=1e-7 is that threshold (cuVS kmeans.cuh sqrdNormError < tol; inertia_check off); scikit-learn's tol is relative to mean(var(X)), so it gets 1e-7 / mean(var(X)) of the same rows (the same absolute threshold; it also stops when no label changes, a zero shift). n_iter and n_iter_over_ours are in every quality cell

mismatch: algorithm: scikit-learn algorithm='lloyd'; ours and cuML have no such parameter (Lloyd)

mismatch: torch-gpu: refused, not an opponent (the harness's own k-means from torch ops); hipvs-gpu (AMD): hipVS cuvs.cluster.kmeans, init_method='KMeansPlusPlus' with oversampling_factor=0, tol=1e-7 under cuVS's rule, squared L2, no seed argument

config: cuML benchmark (RAPIDS), KMeans (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| device_buffer_samples | 0 | - |
| init | "k-means++" | "k-means++" |
| init_centroids | - | null |
| init_size | 0 | - |
| max_iter | 300 | 300 |
| max_samples_per_batch | 32768 | - |
| metric | - | "euclidean" |
| n_clusters | 8 | 8 |
| n_init | 1 | 1 |
| oversampling_factor | 0.0 | 0.0 |
| seed | 7 | 7 |
| tol | 1e-07 | 1e-07 |

accepted difference: cuml-gpu device_buffer_samples: cuML device_buffer_samples=0 uses the full in-memory dataset; init_size only applies to its out-of-core path (kmeans.pyx)

accepted difference: cuml-gpu init_centroids: no supplied centroids; both use the separately compared init method (cluster.py; kmeans.pyx)

accepted difference: cuml-gpu init_size: cuML device_buffer_samples=0 uses the full in-memory dataset; init_size only applies to its out-of-core path (kmeans.pyx)

accepted difference: cuml-gpu max_samples_per_batch: cuML distance-computation batch size partitions all rows without subsampling; library implementation tiling is retained, not a model setting (kmeans.pyx)

accepted difference: cuml-gpu metric: cuML KMeans fixes DistanceType.L2Expanded, matching our squared Euclidean objective (kmeans.pyx set_params)

Inference (each arm predicts with its own model from the fit rounds above):

| arm | hardware | version | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | NVIDIA L40S (nv2, RunPod) | 0.8.44 | Xq | 500000 | 3.1 | 3.1..3.1 | 1 | - | - | 3.1 | - | - (stored whole) | - | - | eval_inertia=4.824e+07, label_agreement_own_centers=0.999998 | yes | LIKE-FOR-LIKE-SPAN | ok |
| cuml-gpu | NVIDIA L40S (host d2fe567d3b88) | cuml 26.08.00 | Xq | 500000 | 3.0 | 3.0..3.0 | 1 | 1.018 | - | 22.5 | 3.0 | 19.54 (upload_ms_untimed (the predict rows' own upload)) | 0.136 (whole/whole) | - | agreement_vs_ours=0.006326, bits_equal_vs_ours=False, eval_inertia=4.572e+07, label_agreement_own_centers=1.000000 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: mojolearn KMeans.predict(Xq), host rows in, host result out

inference call, cuml-gpu: cuml KMeans.predict(Xq on the device, output_type cupy); Xq uploaded before the clock, which ends at the device synchronize

### knn / istella (rows full, shape 400000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/nvidia-wave9/campaign/measured/classical.knn.istella.rows=full/board.json (nv2/paired-c363d75925cba8aeb5d6a2e1661c4f5397f5d6ef90f522c691556f30296c7f09)`, ran on d2fe567d3b88

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 44.8 | 44.8..44.8 | 1 | - | - | 44.8 | 29.5 | 15.34 (upload_ms_separate) | - | - | 1842.7 | 2474.0 | recall_at_k=0.976250, rows_with_repeated_ids=0 | yes | main board, one scored run | wheel | ok (main@2b9a0ea93 nv2/paired-c363d75925cba8aeb5d6a2e1661c4f5397f5d6ef90f522c691556f30296c7f09 2026-10-14; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host d2fe567d3b88) | cuml 26.08.00 | opponent | 71.1 | 71.1..71.1 | 1 | 0.630 | - | 389.4 | 71.1 | 318.30 (upload_ms_untimed) | 0.115 (whole/whole) | - | 1626.3 | 772.0 | recall_at_k=0.976402, rows_with_repeated_ids=0 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: n_neighbors=64 (the cuML benchmark's NearestNeighbors), metric='euclidean', algorithm='brute' (ours, scikit-learn, cuML). Rows: knn block: 400,000 index rows, 4,000 queries, raw. Timed: kneighbors; the fit (index) is before the clock on every arm.

mismatch: seed: no arm has a seed argument (exact search)

mismatch: query_tile: ours only (tiling, results unchanged); n_jobs=-1: scikit-learn only

mismatch: torch-gpu: refused, not an opponent (the harness's own cdist + topk kNN); hipvs-gpu (AMD): hipVS cuvs.neighbors.brute_force, built before the clock, search timed

config: cuML benchmark (RAPIDS), NearestNeighbors (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| algo_params | null | - |
| algorithm | "brute" | "brute" |
| metric | "euclidean" | "euclidean" |
| metric_params | null | - |
| n_neighbors | 64 | 64 |
| p | 2 | 2 |
| query_tile | - | 0 |
| radius | 1.0 | - |
| seed | "none (deterministic)" | "none (deterministic)" |

accepted difference: cuml-gpu algo_params: no algorithm or distance keyword overrides on the explicit brute-force kneighbors call (nearest_neighbors.pyx)

accepted difference: cuml-gpu metric_params: no algorithm or distance keyword overrides on the explicit brute-force kneighbors call (nearest_neighbors.pyx)

accepted difference: cuml-gpu query_tile: our query_tile partitions all queries without changing k or the exact distance metric; internal tiling is logged (neighbors.py)

accepted difference: cuml-gpu radius: kneighbors uses the compared n_neighbors, never the radius-neighbors query option (nearest_neighbors.pyx)

### knn / taxi (rows full, shape 400000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/nvidia-wave9/campaign/measured/classical.knn.taxi.rows=full/board.json (nv2/paired-845e25f655e4e77d56614380b3c0abdd32501cfa9a01dded0fd361d22056b6b5)`, ran on d2fe567d3b88

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 24.7 | 24.7..24.7 | 1 | - | - | 24.7 | 23.8 | 0.84 (upload_ms_separate) | - | - | 1520.9 | 1706.0 | recall_at_k=0.999754, rows_with_repeated_ids=0 | yes | main board, one scored run | wheel | ok (main@2b9a0ea93 nv2/paired-845e25f655e4e77d56614380b3c0abdd32501cfa9a01dded0fd361d22056b6b5 2026-10-14; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host d2fe567d3b88) | cuml 26.08.00 | opponent | 39.5 | 39.5..39.5 | 1 | 0.624 | - | 63.9 | 39.5 | 24.35 (upload_ms_untimed) | 0.386 (whole/whole) | - | 822.4 | 452.0 | recall_at_k=0.999742, rows_with_repeated_ids=0 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: n_neighbors=64 (the cuML benchmark's NearestNeighbors), metric='euclidean', algorithm='brute' (ours, scikit-learn, cuML). Rows: knn block: 400,000 index rows, 4,000 queries, raw. Timed: kneighbors; the fit (index) is before the clock on every arm.

mismatch: seed: no arm has a seed argument (exact search)

mismatch: query_tile: ours only (tiling, results unchanged); n_jobs=-1: scikit-learn only

mismatch: torch-gpu: refused, not an opponent (the harness's own cdist + topk kNN); hipvs-gpu (AMD): hipVS cuvs.neighbors.brute_force, built before the clock, search timed

config: cuML benchmark (RAPIDS), NearestNeighbors (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| algo_params | null | - |
| algorithm | "brute" | "brute" |
| metric | "euclidean" | "euclidean" |
| metric_params | null | - |
| n_neighbors | 64 | 64 |
| p | 2 | 2 |
| query_tile | - | 0 |
| radius | 1.0 | - |
| seed | "none (deterministic)" | "none (deterministic)" |

accepted difference: cuml-gpu algo_params: no algorithm or distance keyword overrides on the explicit brute-force kneighbors call (nearest_neighbors.pyx)

accepted difference: cuml-gpu metric_params: no algorithm or distance keyword overrides on the explicit brute-force kneighbors call (nearest_neighbors.pyx)

accepted difference: cuml-gpu query_tile: our query_tile partitions all queries without changing k or the exact distance metric; internal tiling is logged (neighbors.py)

accepted difference: cuml-gpu radius: kneighbors uses the compared n_neighbors, never the radius-neighbors query option (nearest_neighbors.pyx)

### ols / istella (rows full, shape 2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-grid-logs.txt (nv2/v0134)`, ran on NVIDIA L40S (nv2, RunPod) job v0134

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 504.6 | 504.6..504.6 | 1 | - | - | 504.6 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.332506, rmse=0.681740 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0134 2026-10-11; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 78.5 | 78.5..78.5 | 1 | 6.425 | - | 1328.2 | 78.5 | 1249.71 (upload_ms_untimed) | 0.380 (whole/whole) | - | 5075.9 | 2154.0 | finite=True, r2=-11031.855105, rmse=87.647429 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-gpu | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 580.8 | 580.8..580.8 | 1 | 0.869 | - | 733.1 | 580.8 | 152.30 (upload_ms_untimed) | 0.688 (whole/whole (kernel not derivable)) | - | 3013.5 | 8894.4 | finite=False, r2=nan, rmse=nan | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-gpu-eigh | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 19.8 | 19.8..19.8 | 1 | 25.432 | - | 159.6 | 19.8 | 139.80 (upload_ms_untimed) | 3.161 (whole/whole (kernel not derivable)) | - | 3044.8 | 3455.2 | finite=True, r2=0.151604, rmse=0.768590 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, torch-gpu, torch-gpu-eigh: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: fit_intercept=True. Rows: big block, raw; R2 and RMSE on the 500,000 eval rows. Timed: fit.

mismatch: seed: no arm has a seed argument (closed-form fit); torch-gpu torch.manual_seed(7)

mismatch: solver: ours eig of the normal equations (no parameter), scikit-learn scipy lstsq gelsd (no parameter), cuML algorithm='eig', torch-gpu torch.linalg.lstsq (gels on CUDA), torch-gpu-eigh eigh with a pseudo-inverse cutoff

mismatch: scikit-learn positive=False and copy_X=True: parameters ours does not have

config: cuML benchmark (RAPIDS), LinearRegression (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ols / taxi (rows full, shape 4000000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-grid-logs.txt (nv2/v0134)`, ran on NVIDIA L40S (nv2, RunPod) job v0134

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 14.9 | 14.9..14.9 | 1 | - | - | 14.9 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.908837, rmse=4.696477 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0134 2026-10-11; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 22.1 | 22.1..22.1 | 1 | 0.678 | - | 196.3 | 22.1 | 174.26 (upload_ms_untimed) | 0.076 (whole/whole) | - | 1356.3 | 614.0 | finite=True, r2=0.908836, rmse=4.696488 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-gpu | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 39.0 | 39.0..39.0 | 1 | 0.383 | - | 53.6 | 39.0 | 14.66 (upload_ms_untimed) | 0.278 (whole/whole (kernel not derivable)) | - | 1081.6 | 4649.6 | finite=True, r2=0.908836, rmse=4.696480 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-gpu-eigh | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 2.0 | 2.0..2.0 | 1 | 7.648 | - | 16.2 | 2.0 | 14.27 (upload_ms_untimed) | 0.921 (whole/whole (kernel not derivable)) | - | 1098.7 | 376.4 | finite=True, r2=0.908836, rmse=4.696490 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, torch-gpu, torch-gpu-eigh: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: fit_intercept=True. Rows: big block, raw; R2 and RMSE on the 500,000 eval rows. Timed: fit.

mismatch: seed: no arm has a seed argument (closed-form fit); torch-gpu torch.manual_seed(7)

mismatch: solver: ours eig of the normal equations (no parameter), scikit-learn scipy lstsq gelsd (no parameter), cuML algorithm='eig', torch-gpu torch.linalg.lstsq (gels on CUDA), torch-gpu-eigh eigh with a pseudo-inverse cutoff

mismatch: scikit-learn positive=False and copy_X=True: parameters ours does not have

config: cuML benchmark (RAPIDS), LinearRegression (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### pca / istella (rows full, shape 2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/nvidia-wave9/campaign/measured/classical.pca.istella.rows=full/board.json (nv2/paired-1f349fba0c806514a7a72729720af0ab8c10f4f9c8e06afaca1dbdaa82cff05b)`, ran on d2fe567d3b88

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 104.6 | 104.6..104.6 | 1 | - | - | 104.6 | 26.7 | 77.82 (upload_ms_separate) | - | - | 3640.8 | 2474.0 | explained_variance_ratio_sum=1.000000 | yes | main board, one scored run | wheel | ok (main@2b9a0ea93 nv2/paired-1f349fba0c806514a7a72729720af0ab8c10f4f9c8e06afaca1dbdaa82cff05b 2026-10-14; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host d2fe567d3b88) | cuml 26.08.00 | opponent | 73.9 | 73.9..73.9 | 1 | 1.414 | - | 1268.1 | 73.9 | 1194.16 (upload_ms_untimed) | 0.082 (whole/whole) | - | 5059.4 | 2190.0 | explained_variance_ratio_sum=1.000000 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: n_components=10 (the cuML benchmark's PCA), whiten=False, random_state=7; ours and scikit-learn svd_solver='covariance_eigh'. Rows: big block, raw. Timed: fit.

mismatch: svd_solver: cuML has no 'covariance_eigh'; cuml-gpu runs svd_solver='full' (cuML's COV_EIG_DQ: the eigendecomposition of the covariance, ours' algorithm)

mismatch: torch-gpu: refused (fairness audit 2026-10-13): torch has no PCA estimator of this algorithm; the old arm was a covariance eigh our harness wrote from torch ops, not an opponent (torch.pca_lowrank is a randomized method, a different algorithm)

mismatch: random_state is read by none of the covariance solvers; set to 7 on every arm

config: cuML benchmark (RAPIDS), PCA (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| copy | true | true |
| iterated_power | 15 | "auto" |
| n_components | 10 | 10 |
| n_oversamples | - | 10 |
| power_iteration_normalizer | - | "auto" |
| seed | "none (deterministic)" | 7 |
| svd_solver | "covariance_eigh" | "covariance_eigh" |
| tol | 1e-07 | 0.0 |
| whiten | false | false |

accepted difference: cuml-gpu iterated_power: covariance eigendecomposition on both arms: randomized/Jacobi iteration controls are inactive (cuML pca.pyx; mojolearn decomposition.py)

accepted difference: cuml-gpu n_oversamples: covariance eigendecomposition on both arms: randomized/Jacobi iteration controls are inactive (cuML pca.pyx; mojolearn decomposition.py)

accepted difference: cuml-gpu power_iteration_normalizer: covariance eigendecomposition on both arms: randomized/Jacobi iteration controls are inactive (cuML pca.pyx; mojolearn decomposition.py)

accepted difference: cuml-gpu tol: cuML PCA's tol is read only by its 'jacobi' solver (cuml/decomposition/pca.pyx: 'Used if algorithm = "jacobi"'); the raced 'full' solver, COV_EIG_DQ, the covariance eigendecomposition of ours' covariance_eigh, has no tolerance. cuML's own scikit-learn interop maps scikit-learn's tol 0.0 to its 1e-7

Inference (each arm predicts with its own model from the fit rounds above):

| arm | hardware | version | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | NVIDIA L40S (nv2, RunPod) | 0.8.44 | Xq | 500000 | 34.8 | 34.8..34.8 | 1 | - | - | 34.8 | - | - (stored whole) | - | - | transform_max_rel_err_own_fp64=3.69e-07 | yes | LIKE-FOR-LIKE-SPAN | ok |
| cuml-gpu | NVIDIA L40S (host d2fe567d3b88) | cuml 26.08.00 | Xq | 500000 | 10.3 | 10.3..10.3 | 1 | 3.380 | - | 281.5 | 10.3 | 271.25 (upload_ms_untimed (the predict rows' own upload)) | 0.124 (whole/whole) | - | bits_equal_vs_ours=False, max_abs_diff_vs_ours=118908.242188, transform_max_rel_err_own_fp64=3.694e-07 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: mojolearn PCA.transform(Xq), host rows in, host result out

inference call, cuml-gpu: cuml PCA.transform(Xq on the device, output_type cupy); Xq uploaded before the clock, which ends at the device synchronize

### pca / taxi (rows full, shape 4000000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/nvidia-wave9/campaign/measured/classical.pca.taxi.rows=full/board.json (nv2/paired-2e3591f4c1a55a93dab37866200e52e883e6247f78886dd10771d8ec705510f6)`, ran on d2fe567d3b88

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 9.7 | 9.7..9.7 | 1 | - | - | 9.7 | 1.9 | 7.82 (upload_ms_separate) | - | - | 1700.6 | 938.0 | explained_variance_ratio_sum=0.999997 | yes | main board, one scored run | wheel | ok (main@2b9a0ea93 nv2/paired-2e3591f4c1a55a93dab37866200e52e883e6247f78886dd10771d8ec705510f6 2026-10-14; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host d2fe567d3b88) | cuml 26.08.00 | opponent | 19.2 | 19.2..19.2 | 1 | 0.508 | - | 182.1 | 19.2 | 162.91 (upload_ms_untimed) | 0.053 (whole/whole) | - | 1343.5 | 642.0 | explained_variance_ratio_sum=0.999996 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: n_components=10 (the cuML benchmark's PCA), whiten=False, random_state=7; ours and scikit-learn svd_solver='covariance_eigh'. Rows: big block, raw. Timed: fit.

mismatch: svd_solver: cuML has no 'covariance_eigh'; cuml-gpu runs svd_solver='full' (cuML's COV_EIG_DQ: the eigendecomposition of the covariance, ours' algorithm)

mismatch: torch-gpu: refused (fairness audit 2026-10-13): torch has no PCA estimator of this algorithm; the old arm was a covariance eigh our harness wrote from torch ops, not an opponent (torch.pca_lowrank is a randomized method, a different algorithm)

mismatch: random_state is read by none of the covariance solvers; set to 7 on every arm

config: cuML benchmark (RAPIDS), PCA (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| copy | true | true |
| iterated_power | 15 | "auto" |
| n_components | 10 | 10 |
| n_oversamples | - | 10 |
| power_iteration_normalizer | - | "auto" |
| seed | "none (deterministic)" | 7 |
| svd_solver | "covariance_eigh" | "covariance_eigh" |
| tol | 1e-07 | 0.0 |
| whiten | false | false |

accepted difference: cuml-gpu iterated_power: covariance eigendecomposition on both arms: randomized/Jacobi iteration controls are inactive (cuML pca.pyx; mojolearn decomposition.py)

accepted difference: cuml-gpu n_oversamples: covariance eigendecomposition on both arms: randomized/Jacobi iteration controls are inactive (cuML pca.pyx; mojolearn decomposition.py)

accepted difference: cuml-gpu power_iteration_normalizer: covariance eigendecomposition on both arms: randomized/Jacobi iteration controls are inactive (cuML pca.pyx; mojolearn decomposition.py)

accepted difference: cuml-gpu tol: cuML PCA's tol is read only by its 'jacobi' solver (cuml/decomposition/pca.pyx: 'Used if algorithm = "jacobi"'); the raced 'full' solver, COV_EIG_DQ, the covariance eigendecomposition of ours' covariance_eigh, has no tolerance. cuML's own scikit-learn interop maps scikit-learn's tol 0.0 to its 1e-7

Inference (each arm predicts with its own model from the fit rounds above):

| arm | hardware | version | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | NVIDIA L40S (nv2, RunPod) | 0.8.44 | Xq | 500000 | 18.3 | 18.3..18.3 | 1 | - | - | 18.3 | - | - (stored whole) | - | - | transform_max_rel_err_own_fp64=9.857e-08 | yes | LIKE-FOR-LIKE-SPAN | ok |
| cuml-gpu | NVIDIA L40S (host d2fe567d3b88) | cuml 26.08.00 | Xq | 500000 | 2.8 | 2.8..2.8 | 1 | 6.444 | - | 22.2 | 2.8 | 19.34 (upload_ms_untimed (the predict rows' own upload)) | 0.824 (whole/whole) | - | bits_equal_vs_ours=False, max_abs_diff_vs_ours=0.005283, transform_max_rel_err_own_fp64=1.067e-07 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: mojolearn PCA.transform(Xq), host rows in, host result out

inference call, cuml-gpu: cuml PCA.transform(Xq on the device, output_type cupy); Xq uploaded before the clock, which ends at the device synchronize

### svc / istella (rows full, shape 10000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-grid-logs.txt (nv2/v0126)`, ran on NVIDIA L40S (nv2, RunPod) job v0126

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 44.3 | 44.3..44.3 | 1 | - | - | 44.3 | - | - (stored whole) | - | - | - | - | accuracy=0.922200, n_support=2400 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0126 2026-10-11; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 23.0 | 23.0..23.0 | 1 | 1.923 | - | 40.1 | 23.0 | 17.11 (upload_ms_untimed) | 1.103 (whole/whole) | - | 1008.8 | 464.0 | accuracy=0.922200, n_support=2401 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: C=1.0, kernel='rbf', gamma=1/d, tol=1e-3, degree=3, coef0=0.0, max_iter=-1, cache_size=2000 MB, class_weight=None. Rows: svc block: 10,000 fit rows, 10,000 eval rows, standardized. Timed: fit.

mismatch: seed: scikit-learn and cuML random_state=7 (read only with probability=True); ours refuses random_state without probability=True, so ours stays None

mismatch: cache_size=2000 on every arm; ours honors it only as the prediction buffer (DEVIATION 871), so its training is unaffected

mismatch: shrinking=True: scikit-learn only; nochange_steps=1000: ours and cuML only

config: cuML benchmark (RAPIDS), SVC-RBF (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### svc / taxi (rows full, shape 10000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-grid-logs.txt (nv2/v0126)`, ran on NVIDIA L40S (nv2, RunPod) job v0126

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 1205.5 | 1205.5..1205.5 | 1 | - | - | 1205.5 | - | - (stored whole) | - | - | - | - | accuracy=0.767500, n_support=5527 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0126 2026-10-11; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 379.4 | 379.4..379.4 | 1 | 3.178 | - | 387.8 | 379.4 | 8.39 (upload_ms_untimed) | 3.109 (whole/whole) | - | 985.0 | 440.0 | accuracy=0.767500, n_support=5541 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: C=1.0, kernel='rbf', gamma=1/d, tol=1e-3, degree=3, coef0=0.0, max_iter=-1, cache_size=2000 MB, class_weight=None. Rows: svc block: 10,000 fit rows, 10,000 eval rows, standardized. Timed: fit.

mismatch: seed: scikit-learn and cuML random_state=7 (read only with probability=True); ours refuses random_state without probability=True, so ours stays None

mismatch: cache_size=2000 on every arm; ours honors it only as the prediction buffer (DEVIATION 871), so its training is unaffected

mismatch: shrinking=True: scikit-learn only; nochange_steps=1000: ours and cuML only

config: cuML benchmark (RAPIDS), SVC-RBF (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

## Classical, wave 2

### elasticnet / istella (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v1021)`, ran on NVIDIA L40S (nv2, RunPod) job v1021

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@0a7b206f1) | identical | 96.9 | 96.9..96.9 | 1 | - | - | 96.9 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.260922, rmse=0.718134 | - | main board, one scored run | - | ok (main@0a7b206f1 nv2/v1021 2026-10-09; identity vs amd-mi325x: MATCH) |

settings: alpha=0.1, l1_ratio=0.5 (the cuML benchmark's ElasticNet), fit_intercept=True, max_iter=1000, tol=1e-4, selection='cyclic', positive=False; ours solver='cd' and precompute=False (the only value it takes), scikit-learn precompute=True. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: precompute: ours is constructed precompute=False but its coordinate descent (IDENTICAL and FAST) forms X^T X and X^T y once and sweeps the Gram whenever solver/impl/cd_gram_rule.mojo's shape rule holds (tall, at most 256 columns: both datasets here); scikit-learn precompute=True is that algorithm. cuML is refused: its cd sweeps every row per coordinate and refuses precompute

mismatch: seed: ours refuses random_state (it selects nothing with selection='cyclic'); scikit-learn random_state=7

mismatch: tol: each library's own stopping rule reads it

mismatch: opponents withheld: the lane settings at this tree's HEAD differ from the settings opponents-default-20261006 recorded for its opponent race (an opponent job must score them again)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### elasticnet / taxi (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v1021)`, ran on NVIDIA L40S (nv2, RunPod) job v1021

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@0a7b206f1) | identical | 5.6 | 5.6..5.6 | 1 | - | - | 5.6 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.907378, rmse=4.847225 | - | main board, one scored run | - | ok (main@0a7b206f1 nv2/v1021 2026-10-09; identity vs amd-mi325x: MATCH) |

settings: alpha=0.1, l1_ratio=0.5 (the cuML benchmark's ElasticNet), fit_intercept=True, max_iter=1000, tol=1e-4, selection='cyclic', positive=False; ours solver='cd' and precompute=False (the only value it takes), scikit-learn precompute=True. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: precompute: ours is constructed precompute=False but its coordinate descent (IDENTICAL and FAST) forms X^T X and X^T y once and sweeps the Gram whenever solver/impl/cd_gram_rule.mojo's shape rule holds (tall, at most 256 columns: both datasets here); scikit-learn precompute=True is that algorithm. cuML is refused: its cd sweeps every row per coordinate and refuses precompute

mismatch: seed: ours refuses random_state (it selects nothing with selection='cyclic'); scikit-learn random_state=7

mismatch: tol: each library's own stopping rule reads it

mismatch: opponents withheld: the lane settings at this tree's HEAD differ from the settings opponents-specific-20261006 recorded for its opponent race (an opponent job must score them again)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gmm / istella (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/r43/ingest/nv-gridlogs.txt (nv/n0002)`, ran on NVIDIA L40S (nv, RunPod) job n0002

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.42 (source build, main@4c75b4144) | identical | 572.4 | 572.4..572.4 | 1 | - | - | 572.4 | - | - (stored whole) | - | - | - | - | bic=-3.851e+07, mean_log_likelihood=200.794500, n_iter=24 | - | main board, one scored run | - | ok (main@4c75b4144 nv/n0002 2026-10-13; identity vs the amd columns: n/a) |

settings: n_components=8, covariance_type='full', tol=1e-3, reg_covar=1e-6 on taxi and 3e-3 on Istella-S (GMM_REG_COVAR), max_iter=100, init_params='kmeans', n_init=1, warm_start=False, random_state=7. Rows: 100000 fit and 20000 held-out stride rows of the reg block (standardized by the fit rows); an explicitly full_dataset_coverage recipe retains all fit/eval rows. Timed: fit.

mismatch: init_params='kmeans': each library seeds its own k-means (ours the identity-certified k-means, scikit-learn KMeans(n_init=1, k-means++))

mismatch: opponents withheld: the lane settings at this tree's HEAD differ from the settings release-board-resume-r2 recorded for its opponent race (an opponent job must score them again)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gmm / taxi (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/r43/ingest/nv-gridlogs.txt (nv/n0002)`, ran on NVIDIA L40S (nv, RunPod) job n0002

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.42 (source build, main@4c75b4144) | identical | 36.5 | 36.5..36.5 | 1 | - | - | 36.5 | - | - (stored whole) | - | - | - | - | bic=-3.668e+06, mean_log_likelihood=12.807640, n_iter=29 | - | main board, one scored run | - | ok (main@4c75b4144 nv/n0002 2026-10-13; identity vs the amd columns: n/a) |

settings: n_components=8, covariance_type='full', tol=1e-3, reg_covar=1e-6 on taxi and 3e-3 on Istella-S (GMM_REG_COVAR), max_iter=100, init_params='kmeans', n_init=1, warm_start=False, random_state=7. Rows: 100000 fit and 20000 held-out stride rows of the reg block (standardized by the fit rows); an explicitly full_dataset_coverage recipe retains all fit/eval rows. Timed: fit.

mismatch: init_params='kmeans': each library seeds its own k-means (ours the identity-certified k-means, scikit-learn KMeans(n_init=1, k-means++))

mismatch: opponents withheld: the lane settings at this tree's HEAD differ from the settings release-board-resume-r2 recorded for its opponent race (an opponent job must score them again)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ivf / istella (rows full, shape index 400000x220; queries 4000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/nv_ingest/gridlogs_nvc3.txt (nv/n0002)`, ran on NVIDIA L40S (nv, RunPod) job n0002

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.42 (source build, main@12412eca4) | identical | 311.0 | 311.0..311.0 | 1 | - | - | 311.0 | - | - (stored whole) | - | - | - | - | recall_at_k=0.999925, rows_with_repeated_ids=0 | - | main board, one scored run | - | ok (main@12412eca4 nv/n0002 2026-10-12; identity vs the amd columns: n/a) |
| cuvs-gpu | cuvs | gpu | NVIDIA L40S (host 24a11adce16e) | cuvs 26.08.01 | opponent | 301.6 | 301.6..301.6 | 1 | 1.031 | - | 635.6 | 301.6 | 333.95 (upload_ms_untimed) | 0.489 (whole/whole) | - | 1730.7 | 770.0 | recall_at_k=0.999975, rows_with_repeated_ids=0 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuvs-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: IVF-Flat, n_lists=1024, n_probes=32, k=10, squared L2, k-means 20 iterations, seed 7. Rows: the classical knn lane's block (tools/knn_datasets.real_block): 400000 index rows, 4000 queries, raw. Timed: build + search of every query.

mismatch: quantizer training set: each library's own (FAISS subsamples to 256 rows per list; cuVS kmeans_trainset_fraction 0.5; ours its own)

mismatch: seed: ours random_state=7, faiss cp.seed=7; cuVS IndexParams takes none

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ivf / taxi (rows full, shape index 400000x11; queries 4000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/nv_ingest/gridlogs_nvc3.txt (nv/n0002)`, ran on NVIDIA L40S (nv, RunPod) job n0002

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.42 (source build, main@12412eca4) | identical | 53.0 | 53.0..53.0 | 1 | - | - | 53.0 | - | - (stored whole) | - | - | - | - | recall_at_k=0.999650, rows_with_repeated_ids=0 | - | main board, one scored run | - | ok (main@12412eca4 nv/n0002 2026-10-12; identity vs the amd columns: n/a) |
| cuvs-gpu | cuvs | gpu | NVIDIA L40S (host cc560ebdaf91) | cuvs 26.08.01 | opponent | 297.5 | 297.5..297.5 | 1 | 0.178 | - | 323.7 | 297.5 | 26.15 (upload_ms_untimed) | 0.164 (whole/whole) | - | 933.3 | 450.0 | recall_at_k=0.999450, rows_with_repeated_ids=0 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuvs-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: IVF-Flat, n_lists=1024, n_probes=32, k=10, squared L2, k-means 20 iterations, seed 7. Rows: the classical knn lane's block (tools/knn_datasets.real_block): 400000 index rows, 4000 queries, raw. Timed: build + search of every query.

mismatch: quantizer training set: each library's own (FAISS subsamples to 256 rows per list; cuVS kmeans_trainset_fraction 0.5; ours its own)

mismatch: seed: ours random_state=7, faiss cp.seed=7; cuVS IndexParams takes none

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### kernel-ridge / istella (rows full, shape X 10000x220; Xq 10000x220; y 10000; yq 10000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/nvidia-wave2/campaign/measured/classical2.kernel-ridge.istella.rows=full/board.json (nv2/paired-6a8ab60301dcb48f51cf921cf1c34dba973c4b4dd04249f8766240eb7c538eb3)`, ran on d2fe567d3b88

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 536.5 | 536.5..536.5 | 1 | - | - | 536.5 | - | - (stored whole) | - | - | 2436.4 | 1450.0 | finite=True, r2=0.385427, rmse=0.646407 | - | main board, one scored run | wheel | ok (main@aa8e9e72b nv2/paired-6a8ab60301dcb48f51cf921cf1c34dba973c4b4dd04249f8766240eb7c538eb3 2026-10-14; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host d2fe567d3b88) | cuml 26.08.00 | opponent | 323.1 | 323.1..323.1 | 1 | 1.661 | - | 338.0 | 323.1 | 14.91 (upload_ms_untimed) | 1.587 (whole/whole) | - | 1805.4 | 494.0 | finite=True, r2=0.385427, rmse=0.646407 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: alpha=1.0, kernel='rbf', gamma=1/d, degree=3, coef0=1.0. Rows: 10000 fit and 10000 held-out stride rows of the reg block (standardized by the fit rows). Timed: fit.

mismatch: seed: no arm has a seed argument (closed-form fit)

config: cuML benchmark (RAPIDS), KernelRidge (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| alpha | 1.0 | 1.0 |
| coef0 | 1.0 | 1.0 |
| degree | 3 | 3 |
| gamma | 0.004545454545454545 | 0.004545454545454545 |
| kernel | "rbf" | "rbf" |
| kernel_params | null | - |
| seed | "none (deterministic)" | "none (deterministic)" |

accepted difference: cuml-gpu kernel_params: cuML's named kernel uses gamma/degree/coef0; kernel_params=None supplies no callable-kernel overrides

### kernel-ridge / taxi (rows full, shape X 10000x11; Xq 10000x11; y 10000; yq 10000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/nvidia-wave2/campaign/measured/classical2.kernel-ridge.taxi.rows=full/board.json (nv2/paired-756ebc1d1e6b38301e0b53aa75e20d17d8dfb610adc3f251ce24403772a9ada8)`, ran on d2fe567d3b88

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 533.0 | 533.0..533.0 | 1 | - | - | 533.0 | - | - (stored whole) | - | - | 1543.2 | 1194.0 | finite=True, r2=0.726543, rmse=8.330373 | - | main board, one scored run | wheel | ok (main@aa8e9e72b nv2/paired-756ebc1d1e6b38301e0b53aa75e20d17d8dfb610adc3f251ce24403772a9ada8 2026-10-14; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host d2fe567d3b88) | cuml 26.08.00 | opponent | 323.8 | 323.8..323.8 | 1 | 1.646 | - | 328.3 | 323.8 | 4.41 (upload_ms_untimed) | 1.624 (whole/whole) | - | 1046.7 | 474.0 | finite=True, r2=0.726543, rmse=8.330375 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: alpha=1.0, kernel='rbf', gamma=1/d, degree=3, coef0=1.0. Rows: 10000 fit and 10000 held-out stride rows of the reg block (standardized by the fit rows). Timed: fit.

mismatch: seed: no arm has a seed argument (closed-form fit)

config: cuML benchmark (RAPIDS), KernelRidge (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| alpha | 1.0 | 1.0 |
| coef0 | 1.0 | 1.0 |
| degree | 3 | 3 |
| gamma | 0.09090909090909091 | 0.09090909090909091 |
| kernel | "rbf" | "rbf" |
| kernel_params | null | - |
| seed | "none (deterministic)" | "none (deterministic)" |

accepted difference: cuml-gpu kernel_params: cuML's named kernel uses gamma/degree/coef0; kernel_params=None supplies no callable-kernel overrides

### knn-clf / istella (rows full, shape X 200000x220; Xq 4000x220; y 200000; yq 4000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/nvidia-wave2/campaign/measured/classical2.knn-clf.istella.rows=full/board.json (nv2/paired-cf50f09718629492777771994db95f70823212d3c032407e5192c279e380bb2b)`, ran on d2fe567d3b88

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 32.7 | 32.7..32.7 | 1 | - | - | 32.7 | - | - (stored whole) | - | - | 2598.3 | 938.0 | accuracy=0.926250 | - | main board, one scored run | wheel | ok (main@aa8e9e72b nv2/paired-cf50f09718629492777771994db95f70823212d3c032407e5192c279e380bb2b 2026-10-14; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host d2fe567d3b88) | cuml 26.08.00 | opponent | 39.5 | 39.5..39.5 | 1 | 0.829 | - | 203.8 | 39.5 | 164.29 (upload_ms_untimed) | 0.161 (whole/whole) | - | 2055.7 | 602.0 | accuracy=0.926250 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: n_neighbors=10, weights='uniform', metric='euclidean', algorithm='brute'. Rows: 200000 fit rows, 4000 queries (stride subsets of the cls block, standardized by the fit rows). Timed: fit + predict of the queries.

mismatch: seed: no arm has a seed argument (exact search)

config: cuML benchmark (RAPIDS), KNeighborsClassifier (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| algo_params | null | - |
| algorithm | "brute" | "brute" |
| metric | "euclidean" | "euclidean" |
| metric_params | null | - |
| n_neighbors | 10 | 10 |
| p | 2 | 2 |
| query_tile | - | 0 |
| seed | "none (deterministic)" | "none (deterministic)" |
| weights | "uniform" | "uniform" |

accepted difference: cuml-gpu algo_params: no algorithm or distance keyword overrides on the explicit brute-force kneighbors call (nearest_neighbors.pyx)

accepted difference: cuml-gpu metric_params: no algorithm or distance keyword overrides on the explicit brute-force kneighbors call (nearest_neighbors.pyx)

accepted difference: cuml-gpu query_tile: our query_tile partitions all queries without changing k or the exact distance metric; internal tiling is logged (neighbors.py)

### knn-clf / taxi (rows full, shape X 200000x11; Xq 4000x11; y 200000; yq 4000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/nvidia-wave2/campaign/measured/classical2.knn-clf.taxi.rows=full/board.json (nv2/paired-63f6480feb9828c074c78d0e398d9574b102f148ca27fa8629f0d6a48af3f598)`, ran on d2fe567d3b88

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 13.7 | 13.7..13.7 | 1 | - | - | 13.7 | - | - (stored whole) | - | - | 1558.1 | 682.0 | accuracy=0.741750 | - | main board, one scored run | wheel | ok (main@aa8e9e72b nv2/paired-63f6480feb9828c074c78d0e398d9574b102f148ca27fa8629f0d6a48af3f598 2026-10-14; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host d2fe567d3b88) | cuml 26.08.00 | opponent | 8.4 | 8.4..8.4 | 1 | 1.629 | - | 22.4 | 8.4 | 14.00 (upload_ms_untimed) | 0.613 (whole/whole) | - | 977.8 | 440.0 | accuracy=0.741750 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: n_neighbors=10, weights='uniform', metric='euclidean', algorithm='brute'. Rows: 200000 fit rows, 4000 queries (stride subsets of the cls block, standardized by the fit rows). Timed: fit + predict of the queries.

mismatch: seed: no arm has a seed argument (exact search)

config: cuML benchmark (RAPIDS), KNeighborsClassifier (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| algo_params | null | - |
| algorithm | "brute" | "brute" |
| metric | "euclidean" | "euclidean" |
| metric_params | null | - |
| n_neighbors | 10 | 10 |
| p | 2 | 2 |
| query_tile | - | 0 |
| seed | "none (deterministic)" | "none (deterministic)" |
| weights | "uniform" | "uniform" |

accepted difference: cuml-gpu algo_params: no algorithm or distance keyword overrides on the explicit brute-force kneighbors call (nearest_neighbors.pyx)

accepted difference: cuml-gpu metric_params: no algorithm or distance keyword overrides on the explicit brute-force kneighbors call (nearest_neighbors.pyx)

accepted difference: cuml-gpu query_tile: our query_tile partitions all queries without changing k or the exact distance metric; internal tiling is logged (neighbors.py)

### knn-reg / istella (rows full, shape X 200000x220; Xq 4000x220; y 200000; yq 4000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/nvidia-wave2/campaign/measured/classical2.knn-reg.istella.rows=full/board.json (nv2/paired-52b44e64247f12cafc679b335c740f36f4650cab72b30c5d57e1795672a831c5)`, ran on d2fe567d3b88

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 30.0 | 30.0..30.0 | 1 | - | - | 30.0 | - | - (stored whole) | - | - | 2591.9 | 938.0 | finite=True, r2=0.418145, rmse=0.625388 | - | main board, one scored run | wheel | ok (main@aa8e9e72b nv2/paired-52b44e64247f12cafc679b335c740f36f4650cab72b30c5d57e1795672a831c5 2026-10-14; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host d2fe567d3b88) | cuml 26.08.00 | opponent | 38.4 | 38.4..38.4 | 1 | 0.783 | - | 197.2 | 38.4 | 158.79 (upload_ms_untimed) | 0.152 (whole/whole) | - | 1959.7 | 602.0 | finite=True, r2=0.418145, rmse=0.625388 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: n_neighbors=10, weights='uniform', metric='euclidean', algorithm='brute'. Rows: 200000 fit rows, 4000 queries (stride subsets of the reg block, standardized by the fit rows). Timed: fit + predict of the queries.

mismatch: seed: no arm has a seed argument (exact search)

config: cuML benchmark (RAPIDS), KNeighborsRegressor (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| algo_params | null | - |
| algorithm | "brute" | "brute" |
| metric | "euclidean" | "euclidean" |
| metric_params | null | - |
| n_neighbors | 10 | 10 |
| p | 2 | 2 |
| query_tile | - | 0 |
| seed | "none (deterministic)" | "none (deterministic)" |
| weights | "uniform" | "uniform" |

accepted difference: cuml-gpu algo_params: no algorithm or distance keyword overrides on the explicit brute-force kneighbors call (nearest_neighbors.pyx)

accepted difference: cuml-gpu metric_params: no algorithm or distance keyword overrides on the explicit brute-force kneighbors call (nearest_neighbors.pyx)

accepted difference: cuml-gpu query_tile: our query_tile partitions all queries without changing k or the exact distance metric; internal tiling is logged (neighbors.py)

### knn-reg / taxi (rows full, shape X 200000x11; Xq 4000x11; y 200000; yq 4000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/nvidia-wave2/campaign/measured/classical2.knn-reg.taxi.rows=full/board.json (nv2/paired-58a1b0c0274a143e1c260e9c766d83b55f92f8ad4201a36c3b27be128567dbb6)`, ran on d2fe567d3b88

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 12.4 | 12.4..12.4 | 1 | - | - | 12.4 | - | - (stored whole) | - | - | 1552.0 | 682.0 | finite=True, r2=0.937323, rmse=3.842028 | - | main board, one scored run | wheel | ok (main@aa8e9e72b nv2/paired-58a1b0c0274a143e1c260e9c766d83b55f92f8ad4201a36c3b27be128567dbb6 2026-10-14; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host d2fe567d3b88) | cuml 26.08.00 | opponent | 9.6 | 9.6..9.6 | 1 | 1.289 | - | 23.2 | 9.6 | 13.58 (upload_ms_untimed) | 0.533 (whole/whole) | - | 837.9 | 440.0 | finite=True, r2=0.937323, rmse=3.842028 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: n_neighbors=10, weights='uniform', metric='euclidean', algorithm='brute'. Rows: 200000 fit rows, 4000 queries (stride subsets of the reg block, standardized by the fit rows). Timed: fit + predict of the queries.

mismatch: seed: no arm has a seed argument (exact search)

config: cuML benchmark (RAPIDS), KNeighborsRegressor (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| algo_params | null | - |
| algorithm | "brute" | "brute" |
| metric | "euclidean" | "euclidean" |
| metric_params | null | - |
| n_neighbors | 10 | 10 |
| p | 2 | 2 |
| query_tile | - | 0 |
| seed | "none (deterministic)" | "none (deterministic)" |
| weights | "uniform" | "uniform" |

accepted difference: cuml-gpu algo_params: no algorithm or distance keyword overrides on the explicit brute-force kneighbors call (nearest_neighbors.pyx)

accepted difference: cuml-gpu metric_params: no algorithm or distance keyword overrides on the explicit brute-force kneighbors call (nearest_neighbors.pyx)

accepted difference: cuml-gpu query_tile: our query_tile partitions all queries without changing k or the exact distance metric; internal tiling is logged (neighbors.py)

### lasso / istella (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v1079)`, ran on NVIDIA L40S (nv2, RunPod) job v1079

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@ca8ea1f8d) | identical | 105.8 | 105.8..105.8 | 1 | - | - | 105.8 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.310837, rmse=0.693460 | - | main board, one scored run | - | ok (main@ca8ea1f8d nv2/v1079 2026-10-10; identity vs the amd columns: n/a) |

settings: alpha=0.01, fit_intercept=True, max_iter=1000, tol=1e-4, selection='cyclic', positive=False; ours solver='cd' and precompute=False (the only value it takes), scikit-learn precompute=True. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: precompute: ours is constructed precompute=False but its coordinate descent (IDENTICAL and FAST) forms X^T X and X^T y once and sweeps the Gram whenever solver/impl/cd_gram_rule.mojo's shape rule holds (tall, at most 256 columns: both datasets here); scikit-learn precompute=True is that algorithm. cuML is refused: its cd sweeps every row per coordinate and refuses precompute

mismatch: seed: ours refuses random_state (it selects nothing with selection='cyclic'); scikit-learn random_state=7

mismatch: tol: each library's own stopping rule reads it

mismatch: opponents withheld: the lane settings at this tree's HEAD differ from the settings opponents-specific-20261006 recorded for its opponent race (an opponent job must score them again)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lasso / taxi (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v1079)`, ran on NVIDIA L40S (nv2, RunPod) job v1079

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@ca8ea1f8d) | identical | 5.9 | 5.9..5.9 | 1 | - | - | 5.9 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.908995, rmse=4.804744 | - | main board, one scored run | - | ok (main@ca8ea1f8d nv2/v1079 2026-10-10; identity vs the amd columns: n/a) |

settings: alpha=0.01, fit_intercept=True, max_iter=1000, tol=1e-4, selection='cyclic', positive=False; ours solver='cd' and precompute=False (the only value it takes), scikit-learn precompute=True. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: precompute: ours is constructed precompute=False but its coordinate descent (IDENTICAL and FAST) forms X^T X and X^T y once and sweeps the Gram whenever solver/impl/cd_gram_rule.mojo's shape rule holds (tall, at most 256 columns: both datasets here); scikit-learn precompute=True is that algorithm. cuML is refused: its cd sweeps every row per coordinate and refuses precompute

mismatch: seed: ours refuses random_state (it selects nothing with selection='cyclic'); scikit-learn random_state=7

mismatch: tol: each library's own stopping rule reads it

mismatch: opponents withheld: the lane settings at this tree's HEAD differ from the settings opponents-default-20261006 recorded for its opponent race (an opponent job must score them again)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### linearsvc / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/nvidia-wave2/campaign/measured/classical2.linearsvc.istella.rows=full/board.json (nv2/paired-684b80a4cfdf4f4f4fc4bd68e745642225118d0cbca43efdd9f3f96681fa50ef)`, ran on d2fe567d3b88

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 433.8 | 433.8..433.8 | 1 | - | - | 433.8 | - | - (stored whole) | - | - | 2469.9 | 2218.0 | accuracy=0.923470 | - | main board, one scored run | wheel | ok (main@aa8e9e72b nv2/paired-684b80a4cfdf4f4f4fc4bd68e745642225118d0cbca43efdd9f3f96681fa50ef 2026-10-14; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host d2fe567d3b88) | cuml 26.08.00 | opponent | 271.3 | 271.3..271.3 | 1 | 1.599 | - | 924.6 | 271.3 | 653.28 (upload_ms_untimed) | 0.469 (whole/whole) | - | 2997.2 | 1366.0 | accuracy=0.922600 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: penalty='l2', loss='squared_hinge', C=1.0, tol=1e-4, max_iter=1000, fit_intercept=True, class_weight=None; ours and cuML penalized_intercept=False; scikit-learn intercept_scaling=1.0, random_state=7. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: solver: ours and cuML L-BFGS on the primal with an unpenalized intercept; scikit-learn liblinear (dual='auto'), which penalizes the intercept

mismatch: seed: ours and cuML LinearSVC have no seed argument

config: cuML benchmark (RAPIDS), LinearSVC (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| C | 1.0 | 1.0 |
| class_weight | null | null |
| fit_intercept | true | true |
| lbfgs_memory | 5 | 5 |
| linesearch_max_iter | 100 | 100 |
| loss | "squared_hinge" | "squared_hinge" |
| max_iter | 1000 | 1000 |
| multi_class | "ovr" | "ovr" |
| n_streams | 1 | - |
| penalized_intercept | false | false |
| penalty | "l2" | "l2" |
| seed | "none (deterministic)" | "none (deterministic)" |
| tol | 0.0001 | 0.0001 |

accepted difference: cuml-gpu class_weight: None (unweighted) set explicitly on every arm

accepted difference: cuml-gpu n_streams: cuML n_streams schedules fitting in parallel streams; loss, penalty, multiclass strategy and stopping controls are compared separately

### linearsvc / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/nvidia-wave2/campaign/measured/classical2.linearsvc.taxi.rows=full/board.json (nv2/paired-9152af55eb9a00ada6345f6733b74bee925f0a052934a3713b54f3ee2b963234)`, ran on d2fe567d3b88

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 64.6 | 64.6..64.6 | 1 | - | - | 64.6 | - | - (stored whole) | - | - | 1592.9 | 682.0 | accuracy=0.763330 | - | main board, one scored run | wheel | ok (main@aa8e9e72b nv2/paired-9152af55eb9a00ada6345f6733b74bee925f0a052934a3713b54f3ee2b963234 2026-10-14; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host d2fe567d3b88) | cuml 26.08.00 | opponent | 8.2 | 8.2..8.2 | 1 | 7.838 | - | 56.6 | 8.2 | 48.37 (upload_ms_untimed) | 1.141 (whole/whole) | - | 1127.4 | 490.0 | accuracy=0.762990 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: penalty='l2', loss='squared_hinge', C=1.0, tol=1e-4, max_iter=1000, fit_intercept=True, class_weight=None; ours and cuML penalized_intercept=False; scikit-learn intercept_scaling=1.0, random_state=7. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: solver: ours and cuML L-BFGS on the primal with an unpenalized intercept; scikit-learn liblinear (dual='auto'), which penalizes the intercept

mismatch: seed: ours and cuML LinearSVC have no seed argument

config: cuML benchmark (RAPIDS), LinearSVC (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| C | 1.0 | 1.0 |
| class_weight | null | null |
| fit_intercept | true | true |
| lbfgs_memory | 5 | 5 |
| linesearch_max_iter | 100 | 100 |
| loss | "squared_hinge" | "squared_hinge" |
| max_iter | 1000 | 1000 |
| multi_class | "ovr" | "ovr" |
| n_streams | 1 | - |
| penalized_intercept | false | false |
| penalty | "l2" | "l2" |
| seed | "none (deterministic)" | "none (deterministic)" |
| tol | 0.0001 | 0.0001 |

accepted difference: cuml-gpu class_weight: None (unweighted) set explicitly on every arm

accepted difference: cuml-gpu n_streams: cuML n_streams schedules fitting in parallel streams; loss, penalty, multiclass strategy and stopping controls are compared separately

### linearsvr / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/measurement-readiness-20261013/nvidia/fitted-receipts/linearsvr-istella/board.json (nv/paired-3e83b678b03aebb52c79fbdba694f32799385651f751008ef001ee09ef763cba)`, ran on cb7901bedc82

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | 0.8.44 | identical | 95.7 | 95.7..95.7 | 1 | - | - | 95.7 | - | - (stored whole) | - | - | 2415.6 | 1450.0 | finite=True, r2=-0.106761, rmse=0.878794 | - | main board, one scored run | wheel | ok (main@da2a48fc3 nv/paired-3e83b678b03aebb52c79fbdba694f32799385651f751008ef001ee09ef763cba 2026-10-14; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cb7901bedc82) | cuml 26.08.00 | opponent | 90.2 | 90.2..90.2 | 1 | 1.061 | - | 809.3 | 90.2 | 719.12 (upload_ms_untimed) | 0.118 (whole/whole) | - | 2890.8 | 1374.0 | finite=True, r2=-0.106761, rmse=0.878795 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: penalty='l2', loss='epsilon_insensitive', epsilon=0.0, C=1.0, tol=1e-4, max_iter=1000, fit_intercept=True. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: penalty='l2' set on ours and cuML (their default is 'l1'); scikit-learn has only l2

mismatch: solver: ours and cuML L-BFGS on the primal; scikit-learn liblinear dual coordinate descent (dual=True, the only form for this loss)

mismatch: intercept: ours and cuML penalized_intercept=False; scikit-learn intercept_scaling=1.0 (liblinear penalizes it)

mismatch: seed: ours and cuML LinearSVR have no seed argument; scikit-learn 7

config: cuML benchmark (RAPIDS), LinearSVR (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| C | 1.0 | 1.0 |
| epsilon | 0.0 | 0.0 |
| fit_intercept | true | true |
| lbfgs_memory | 5 | 5 |
| linesearch_max_iter | 100 | 100 |
| loss | "epsilon_insensitive" | "epsilon_insensitive" |
| max_iter | 1000 | 1000 |
| penalized_intercept | false | false |
| penalty | "l2" | "l2" |
| seed | "none (deterministic)" | "none (deterministic)" |
| tol | 0.0001 | 0.0001 |

### linearsvr / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/measurement-readiness-20261013/nvidia/fitted-receipts/linearsvr-taxi/board.json (nv/paired-75f4f193e27d2f86536459384001eba1569f211ce2ab15e696ff37050160ebb0)`, ran on cb7901bedc82

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | 0.8.44 | identical | 35.2 | 35.2..35.2 | 1 | - | - | 35.2 | - | - (stored whole) | - | - | 1540.1 | 682.0 | finite=True, r2=0.899814, rmse=5.041273 | - | main board, one scored run | wheel | ok (main@da2a48fc3 nv/paired-75f4f193e27d2f86536459384001eba1569f211ce2ab15e696ff37050160ebb0 2026-10-14; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cb7901bedc82) | cuml 26.08.00 | opponent | 35.3 | 35.3..35.3 | 1 | 0.995 | - | 96.1 | 35.3 | 60.73 (upload_ms_untimed) | 0.366 (whole/whole) | - | 950.9 | 498.0 | finite=True, r2=0.899818, rmse=5.041176 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: penalty='l2', loss='epsilon_insensitive', epsilon=0.0, C=1.0, tol=1e-4, max_iter=1000, fit_intercept=True. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: penalty='l2' set on ours and cuML (their default is 'l1'); scikit-learn has only l2

mismatch: solver: ours and cuML L-BFGS on the primal; scikit-learn liblinear dual coordinate descent (dual=True, the only form for this loss)

mismatch: intercept: ours and cuML penalized_intercept=False; scikit-learn intercept_scaling=1.0 (liblinear penalizes it)

mismatch: seed: ours and cuML LinearSVR have no seed argument; scikit-learn 7

config: cuML benchmark (RAPIDS), LinearSVR (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| C | 1.0 | 1.0 |
| epsilon | 0.0 | 0.0 |
| fit_intercept | true | true |
| lbfgs_memory | 5 | 5 |
| linesearch_max_iter | 100 | 100 |
| loss | "epsilon_insensitive" | "epsilon_insensitive" |
| max_iter | 1000 | 1000 |
| penalized_intercept | false | false |
| penalty | "l2" | "l2" |
| seed | "none (deterministic)" | "none (deterministic)" |
| tol | 0.0001 | 0.0001 |

### logreg / istella (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v1021)`, ran on NVIDIA L40S (nv2, RunPod) job v1021

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@0a7b206f1) | identical | 1797.6 | 1797.6..1797.6 | 1 | - | - | 1797.6 | - | - (stored whole) | - | - | - | - | accuracy=0.924590, logloss=0.181249, nonfinite_proba_rows=0 | - | main board, one scored run | - | ok (main@0a7b206f1 nv2/v1021 2026-10-09; identity vs amd-mi325x: MATCH) |

settings: penalty='l2', C=1.0, tol=1e-4, max_iter=1000, fit_intercept=True, class_weight=None; ours and cuML linesearch_max_iter=50, L-BFGS memory 5 (cuML lbfgs_memory=5; ours fixes 5); scikit-learn random_state=7. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: solver: ours 'qn' (L-BFGS, cuML's), scikit-learn 'lbfgs', cuML 'qn'; each library's own stopping rule reads tol

mismatch: seed: ours and cuML LogisticRegression have no seed argument

mismatch: l1_ratio: ours None, scikit-learn None or 0.0 (its l2 spelling from 1.8)

mismatch: opponents withheld: the lane settings at this tree's HEAD differ from the settings opponents-specific-20261006 recorded for its opponent race (an opponent job must score them again)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### logreg / taxi (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v1021)`, ran on NVIDIA L40S (nv2, RunPod) job v1021

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@0a7b206f1) | identical | 20.4 | 20.4..20.4 | 1 | - | - | 20.4 | - | - (stored whole) | - | - | - | - | accuracy=0.763350, logloss=0.538985, nonfinite_proba_rows=0 | - | main board, one scored run | - | ok (main@0a7b206f1 nv2/v1021 2026-10-09; identity vs amd-mi325x: MATCH) |

settings: penalty='l2', C=1.0, tol=1e-4, max_iter=1000, fit_intercept=True, class_weight=None; ours and cuML linesearch_max_iter=50, L-BFGS memory 5 (cuML lbfgs_memory=5; ours fixes 5); scikit-learn random_state=7. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: solver: ours 'qn' (L-BFGS, cuML's), scikit-learn 'lbfgs', cuML 'qn'; each library's own stopping rule reads tol

mismatch: seed: ours and cuML LogisticRegression have no seed argument

mismatch: l1_ratio: ours None, scikit-learn None or 0.0 (its l2 spelling from 1.8)

mismatch: opponents withheld: the lane settings at this tree's HEAD differ from the settings opponents-default-20261006 recorded for its opponent race (an opponent job must score them again)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ridge / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-grid-logs.txt (nv2/v0134)`, ran on NVIDIA L40S (nv2, RunPod) job v0134

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 526.0 | 526.0..526.0 | 1 | - | - | 526.0 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.328674, rmse=0.684427 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0134 2026-10-11; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 48.2 | 48.2..48.2 | 1 | 10.905 | - | 754.2 | 48.2 | 705.96 (upload_ms_untimed) | 0.697 (whole/whole) | - | 2969.7 | 1410.0 | finite=True, r2=-0.251259, rmse=0.934403 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: alpha=1.0, fit_intercept=True; scikit-learn positive=False, random_state=7. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: solver, named on every arm: ours and cuML 'eig' (eigendecomposition of the normal equations), scikit-learn 'cholesky' (it has no 'eig')

mismatch: seed: ours and cuML Ridge have no seed argument

config: cuML benchmark (RAPIDS), Ridge (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ridge / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-grid-logs.txt (nv2/v0134)`, ran on NVIDIA L40S (nv2, RunPod) job v0134

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 4.6 | 4.6..4.6 | 1 | - | - | 4.6 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.908983, rmse=4.805050 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0134 2026-10-11; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 15.2 | 15.2..15.2 | 1 | 0.305 | - | 71.7 | 15.2 | 56.48 (upload_ms_untimed) | 0.065 (whole/whole) | - | 1000.4 | 534.0 | finite=True, r2=0.908983, rmse=4.805051 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: alpha=1.0, fit_intercept=True; scikit-learn positive=False, random_state=7. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: solver, named on every arm: ours and cuML 'eig' (eigendecomposition of the normal equations), scikit-learn 'cholesky' (it has no 'eig')

mismatch: seed: ours and cuML Ridge have no seed argument

config: cuML benchmark (RAPIDS), Ridge (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### svr / istella (rows full, shape X 10000x220; Xq 10000x220; y 10000; yq 10000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/measurement-readiness-20261013/nvidia/fitted-receipts/svr-istella/board.json (nv/paired-fda4176c7e740df5ff5ae80130eb0a489690f57d22a1cbf89b5e4f88b74ae796)`, ran on cb7901bedc82

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | 0.8.44 | identical | 63.3 | 63.3..63.3 | 1 | - | - | 63.3 | - | - (stored whole) | - | - | 2465.1 | 908.0 | finite=True, r2=0.318258, rmse=0.680816 | - | main board, one scored run | wheel | ok (main@da2a48fc3 nv/paired-fda4176c7e740df5ff5ae80130eb0a489690f57d22a1cbf89b5e4f88b74ae796 2026-10-14; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cb7901bedc82) | cuml 26.08.00 | opponent | 44.8 | 44.8..44.8 | 1 | 1.413 | - | 78.9 | 44.8 | 34.07 (upload_ms_untimed) | 0.803 (whole/whole) | - | 1786.9 | 462.0 | finite=True, r2=0.318232, rmse=0.680829 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: kernel='rbf', gamma=1/d, C=1.0, epsilon=0.1, tol=1e-3, degree=3, coef0=0.0, max_iter=-1, cache_size=2000 MB. Rows: 10000 fit and 10000 held-out stride rows of the reg block (standardized by the fit rows). Timed: fit.

mismatch: cache_size=2000 on every arm; ours honors it at predict only (DEVIATION 871); libsvm is single-threaded; shrinking=True is scikit-learn's only

mismatch: seed: no arm has a seed argument

config: cuML benchmark (RAPIDS), SVR-RBF (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| C | 1.0 | 1.0 |
| cache_size | 2000.0 | 2000.0 |
| coef0 | 0.0 | 0.0 |
| degree | 3 | 3 |
| epsilon | 0.1 | 0.1 |
| gamma | 0.004545454545454545 | 0.004545454545454545 |
| kernel | "rbf" | "rbf" |
| max_iter | -1 | -1 |
| nochange_steps | 1000 | 1000 |
| seed | "none (deterministic)" | "none (deterministic)" |
| tol | 0.001 | 0.001 |

### svr / taxi (rows full, shape X 10000x11; Xq 10000x11; y 10000; yq 10000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/measurement-readiness-20261013/nvidia/fitted-receipts/svr-taxi/board.json (nv/paired-7d82312095685a1f10d81cbcb08abd4bf163737d507cbf7c7290c4a057de1126)`, ran on cb7901bedc82

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | 0.8.44 | identical | 65.0 | 65.0..65.0 | 1 | - | - | 65.0 | - | - (stored whole) | - | - | 1542.1 | 938.0 | finite=True, r2=0.767551, rmse=7.680395 | - | main board, one scored run | wheel | ok (main@da2a48fc3 nv/paired-7d82312095685a1f10d81cbcb08abd4bf163737d507cbf7c7290c4a057de1126 2026-10-14; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cb7901bedc82) | cuml 26.08.00 | opponent | 44.3 | 44.3..44.3 | 1 | 1.467 | - | 48.7 | 44.3 | 4.39 (upload_ms_untimed) | 1.335 (whole/whole) | - | 889.1 | 442.0 | finite=True, r2=0.767551, rmse=7.680405 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: kernel='rbf', gamma=1/d, C=1.0, epsilon=0.1, tol=1e-3, degree=3, coef0=0.0, max_iter=-1, cache_size=2000 MB. Rows: 10000 fit and 10000 held-out stride rows of the reg block (standardized by the fit rows). Timed: fit.

mismatch: cache_size=2000 on every arm; ours honors it at predict only (DEVIATION 871); libsvm is single-threaded; shrinking=True is scikit-learn's only

mismatch: seed: no arm has a seed argument

config: cuML benchmark (RAPIDS), SVR-RBF (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| C | 1.0 | 1.0 |
| cache_size | 2000.0 | 2000.0 |
| coef0 | 0.0 | 0.0 |
| degree | 3 | 3 |
| epsilon | 0.1 | 0.1 |
| gamma | 0.09090909090909091 | 0.09090909090909091 |
| kernel | "rbf" | "rbf" |
| max_iter | -1 | -1 |
| nochange_steps | 1000 | 1000 |
| seed | "none (deterministic)" | "none (deterministic)" |
| tol | 0.001 | 0.001 |

### tsvd / istella (rows full, shape X 1000000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-grid-logs.txt (nv2/v0094)`, ran on NVIDIA L40S (nv2, RunPod) job v0094

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 53.6 | 53.6..53.6 | 1 | - | - | 53.6 | - | - (stored whole) | - | - | - | - | explained_variance_ratio_sum=1.000000, relative_reconstruction_error=0.0001314 | - | main board, one scored run | - | ok (main@a1c96a5ba nv2/v0094 2026-10-11; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 35.3 | 35.3..35.3 | 1 | 1.519 | - | 699.3 | 35.3 | 664.00 (upload_ms_untimed) | 0.077 (whole/whole) | - | 2749.0 | 1314.0 | explained_variance_ratio_sum=1.000000, relative_reconstruction_error=0.0001472 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: n_components=10 (the cuML benchmark's tSVD), tol=0.0, n_iter=5, n_oversamples=10, random_state=7. Rows: 1000000 stride rows of the train split, raw (sentinel cleaned, not scaled). Timed: fit.

mismatch: algorithm: ours 'covariance_eigh' (eigh of X^T X), scikit-learn 'arpack' (tol=0, exact to ARPACK's tolerance), cuML 'full'

config: cuML benchmark (RAPIDS), tSVD (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### tsvd / taxi (rows full, shape X 1000000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v1226)`, ran on NVIDIA L40S (nv2, RunPod) job v1226

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@a8e0548a5) | identical | 2.9 | 2.9..2.9 | 1 | - | - | 2.9 | - | - (stored whole) | - | - | - | - | explained_variance_ratio_sum=0.999965, relative_reconstruction_error=0.003257 | - | main board, one scored run | - | ok (main@a8e0548a5 nv2/v1226 2026-10-10; identity vs amd-mi325x: MATCH) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 5.9 | 5.9..5.9 | 1 | 0.492 | - | 54.4 | 5.9 | 48.44 (upload_ms_untimed) | 0.054 (whole/whole) | - | 982.5 | 516.0 | explained_variance_ratio_sum=0.999964, relative_reconstruction_error=0.003257 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: n_components=10 (the cuML benchmark's tSVD), tol=0.0, n_iter=5, n_oversamples=10, random_state=7. Rows: 1000000 stride rows of the train split, raw (sentinel cleaned, not scaled). Timed: fit.

mismatch: algorithm: ours 'covariance_eigh' (eigh of X^T X), scikit-learn 'arpack' (tol=0, exact to ARPACK's tolerance), cuML 'full'

config: cuML benchmark (RAPIDS), tSVD (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

## Neural

### gemm / gaussian (neural shape full: 4096x4096x4096)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/measurement-readiness-20261013/nvidia/gemm/board.json (nv/paired-5253dbab563a2378f64b75895112d48ef6775fcce51209b399727d4608084b55)`, ran on cb7901bedc82

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 35.5 ms) = 1.500; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 37.6 ms) = 1.416. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | 0.8.44 | identical | 53.2 | 53.2..53.2 | 1 | - | - | 53.2 | - | - (stored whole) | - | - | 1743.4 | 908.0 | max_rel_err_vs_fp64=2.399e-07 | - | main board, one scored run | wheel | ok (main@18679c3bb nv/paired-5253dbab563a2378f64b75895112d48ef6775fcce51209b399727d4608084b55 2026-10-14; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host cb7901bedc82) | torch 2.13.0+cu129 | opponent | 37.6 | 37.6..37.6 | 1 | 1.416 | - | 37.6 | - | - (stored whole) | 1.416 (whole/whole) | - | 937.5 | 200.1 | max_abs_diff_vs_ours=0.0005341, max_rel_diff_vs_ours=1.475e-06, max_rel_err_vs_fp64=1.401e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host cb7901bedc82) | torch 2.13.0+cu129 | opponent | 36.6 | 36.6..36.6 | 1 | 1.453 | - | 36.6 | - | - (stored whole) | 1.453 (whole/whole) | - | 937.7 | 200.1 | max_abs_diff_vs_ours=0.100822, max_rel_diff_vs_ours=0.0002785, max_rel_err_vs_fp64=0.0002784 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host cb7901bedc82) | torch 2.13.0+cu129 | opponent | 39.2 | 39.2..39.2 | 1 | 1.357 | - | 39.2 | - | - (stored whole) | 1.357 (whole/whole) | - | 1145.9 | 200.1 | max_abs_diff_vs_ours=0.0005341, max_rel_diff_vs_ours=1.475e-06, max_rel_err_vs_fp64=1.401e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host cb7901bedc82) | torch 2.13.0+cu129 | opponent | 46.6 | 46.6..46.6 | 1 | 1.142 | - | 46.6 | - | - (stored whole) | 1.142 (whole/whole) | - | 1144.3 | 200.1 | max_abs_diff_vs_ours=0.100822, max_rel_diff_vs_ours=0.0002785, max_rel_err_vs_fp64=0.0002784 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host cb7901bedc82) | torch 2.13.0+cu129 | opponent | 35.5 | 35.5..35.5 | 1 | 1.500 | - | 35.5 | - | - (stored whole) | 1.500 (whole/whole) | - | 1174.8 | 240.2 | max_abs_diff_vs_ours=1.358337, max_rel_diff_vs_ours=0.003751, max_rel_err_vs_fp64=0.003752 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host cb7901bedc82) | torch 2.13.0+cu129 | opponent | 36.3 | 36.3..36.3 | 1 | 1.467 | - | 36.3 | - | - (stored whole) | 1.467 (whole/whole) | - | 1419.3 | 240.2 | max_abs_diff_vs_ours=1.358337, max_rel_diff_vs_ours=0.003751, max_rel_err_vs_fp64=0.003752 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, torch-eager-fp32, torch-eager-tf32, torch-compile-fp32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-compile-tf32 | torch-eager-bf16 | torch-eager-fp32 | torch-eager-tf32 |
|---|---||---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| input_arrays | {"a": {"dtype": "float32", "sha256": "485a7287793a9d7d33d710ea2820ea9c7519d31100db4bafeee813baa532dd7d", "shape": [4096, 4096]}, "b": {"dtype": "float32", "sha256": "3f38db408e287e78e700728b77d904ea5344de168156ad74431c2148fc961104", "shape": [4096, 4096]}} | {"a": {"dtype": "float32", "sha256": "485a7287793a9d7d33d710ea2820ea9c7519d31100db4bafeee813baa532dd7d", "shape": [4096, 4096]}, "b": {"dtype": "float32", "sha256": "3f38db408e287e78e700728b77d904ea5344de168156ad74431c2148fc961104", "shape": [4096, 4096]}} | {"a": {"dtype": "float32", "sha256": "485a7287793a9d7d33d710ea2820ea9c7519d31100db4bafeee813baa532dd7d", "shape": [4096, 4096]}, "b": {"dtype": "float32", "sha256": "3f38db408e287e78e700728b77d904ea5344de168156ad74431c2148fc961104", "shape": [4096, 4096]}} | {"a": {"dtype": "float32", "sha256": "485a7287793a9d7d33d710ea2820ea9c7519d31100db4bafeee813baa532dd7d", "shape": [4096, 4096]}, "b": {"dtype": "float32", "sha256": "3f38db408e287e78e700728b77d904ea5344de168156ad74431c2148fc961104", "shape": [4096, 4096]}} | {"a": {"dtype": "float32", "sha256": "485a7287793a9d7d33d710ea2820ea9c7519d31100db4bafeee813baa532dd7d", "shape": [4096, 4096]}, "b": {"dtype": "float32", "sha256": "3f38db408e287e78e700728b77d904ea5344de168156ad74431c2148fc961104", "shape": [4096, 4096]}} | {"a": {"dtype": "float32", "sha256": "485a7287793a9d7d33d710ea2820ea9c7519d31100db4bafeee813baa532dd7d", "shape": [4096, 4096]}, "b": {"dtype": "float32", "sha256": "3f38db408e287e78e700728b77d904ea5344de168156ad74431c2148fc961104", "shape": [4096, 4096]}} | {"a": {"dtype": "float32", "sha256": "485a7287793a9d7d33d710ea2820ea9c7519d31100db4bafeee813baa532dd7d", "shape": [4096, 4096]}, "b": {"dtype": "float32", "sha256": "3f38db408e287e78e700728b77d904ea5344de168156ad74431c2148fc961104", "shape": [4096, 4096]}} |
| operand_value_type | "float32" | "float32" | "float32" | "float32" | "float32" | "float32" | "float32" |
| operation | "matmul" | "matmul" | "matmul" | "matmul" | "matmul" | "matmul" | "matmul" |
| seed | 7 | 7 | 7 | 7 | 7 | 7 | 7 |
| transpose_b | false | false | false | false | false | false | false |

### lm-forward / bytes (neural shape full: B1 L2048 DM384 H6 KV6 HD64 FF1024 layers8 V8192)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-grid-logs.txt (nv2/v0139)`, ran on NVIDIA L40S (nv2, RunPod) job v0139

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 42.5 ms) = 1.305; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 45.8 ms) = 1.209. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ff77698) | identical | 55.4 | 55.4..55.4 | 1 | - | - | 55.4 | - | - (stored whole) | - | - | - | - | mean_nll=9.018733 | - | main board, one scored run | - | ok (main@c8ff77698 nv2/v0139 2026-10-11; identity vs the amd columns: n/a) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 46.5 | 46.5..46.5 | 1 | 1.191 | - | 46.5 | - | - (stored whole) | 1.191 (whole/whole) | - | 1107.8 | 173.2 | mean_nll=9.018733 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 45.2 | 45.2..45.2 | 1 | 1.227 | - | 45.2 | - | - (stored whole) | 1.227 (whole/whole) | - | 1105.0 | 173.2 | mean_nll=9.018733 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 45.8 | 45.8..45.8 | 1 | 1.209 | - | 45.8 | - | - (stored whole) | 1.209 (whole/whole) | - | 1206.6 | 154.7 | mean_nll=9.018733 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 44.1 | 44.1..44.1 | 1 | 1.257 | - | 44.1 | - | - (stored whole) | 1.257 (whole/whole) | - | 1208.3 | 154.7 | mean_nll=9.018732 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 44.3 | 44.3..44.3 | 1 | 1.252 | - | 44.3 | - | - (stored whole) | 1.252 (whole/whole) | - | 1233.3 | 192.5 | mean_nll=9.018664 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 42.5 | 42.5..42.5 | 1 | 1.305 | - | 42.5 | - | - (stored whole) | 1.305 (whole/whole) | - | 1378.4 | 192.5 | mean_nll=9.018669 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-eager-tf32, torch-compile-fp32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lm-train-step / bytes (neural shape full: B1 L2048 DM384 H6 KV6 HD64 FF1024 layers8 V8192)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-grid-logs.txt (nv2/v0139)`, ran on NVIDIA L40S (nv2, RunPod) job v0139

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 10.8 ms) = 3.485; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 21.6 ms) = 1.740. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ff77698) | identical | 37.5 | 37.5..37.5 | 1 | - | - | 37.5 | - | - (stored whole) | - | - | - | - | loss_first_step=9.018733, loss_last_step=8.418446, steps=2 | - | main board, one scored run | - | ok (main@c8ff77698 nv2/v0139 2026-10-11; identity vs the amd columns: n/a) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 21.9 | 21.9..21.9 | 1 | 1.714 | - | 21.9 | - | - (stored whole) | 1.714 (whole/whole) | - | 1257.2 | 959.2 | loss_first_step=9.018733, loss_last_step=8.418449, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 17.9 | 17.9..17.9 | 1 | 2.102 | - | 17.9 | - | - (stored whole) | 2.102 (whole/whole) | - | 1252.4 | 959.2 | loss_first_step=9.018732, loss_last_step=8.418548, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 21.6 | 21.6..21.6 | 1 | 1.740 | - | 21.6 | - | - (stored whole) | 1.740 (whole/whole) | - | 1224.5 | 720.7 | loss_first_step=9.018734, loss_last_step=8.418448, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 22.1 | 22.1..22.1 | 1 | 1.698 | - | 22.1 | - | - (stored whole) | 1.698 (whole/whole) | - | 1219.0 | 727.2 | loss_first_step=9.018732, loss_last_step=8.418557, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 24.4 | 24.4..24.4 | 1 | 1.538 | - | 24.4 | - | - (stored whole) | 1.538 (whole/whole) | - | 1388.2 | 796.6 | loss_first_step=9.018402, loss_last_step=8.417328, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 10.8 | 10.8..10.8 | 1 | 3.485 | - | 10.8 | - | - (stored whole) | 3.485 (whole/whole) | - | 1394.1 | 546.6 | loss_first_step=9.018669, loss_last_step=8.417006, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-eager-tf32, torch-compile-fp32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### mamba1-forward / gaussian (neural shape full: B1 L2048 DM384)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-grid-logs.txt (nv2/v0070)`, ran on NVIDIA L40S (nv2, RunPod) job v0070

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 193.2 ms) = 0.028; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 150.9 ms) = 0.036. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@2afa942e3) | identical | 5.4 | 5.4..5.4 | 1 | - | - | 5.4 | - | - (stored whole) | - | - | - | - | - | - | main board, one scored run | - | ok (main@2afa942e3 nv2/v0070 2026-10-11; identity vs the amd columns: n/a) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 150.9 | 150.9..150.9 | 1 | 0.036 | - | 150.9 | - | - (stored whole) | 0.036 (whole/whole) | - | 1062.2 | 354.3 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 152.3 | 152.3..152.3 | 1 | 0.035 | - | 152.3 | - | - (stored whole) | 0.035 (whole/whole) | - | 1041.6 | 354.3 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 193.2 | 193.2..193.2 | 1 | 0.028 | - | 193.2 | - | - (stored whole) | 0.028 (whole/whole) | - | 1189.9 | 307.1 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| mamba-ssm-fp32 | mamba-ssm | gpu | NVIDIA L40S (host f003873fb257) | mamba-ssm 2.3.2.post1+ge9594ce1c732 | opponent | 3.5 | 3.5..3.5 | 1 | 1.541 | - | 3.5 | - | - (stored whole) | 1.541 (whole/whole) | - | 1247.2 | 54.6 | max_abs_diff_vs_ours=1.192e-07, max_rel_diff_vs_ours=5.945e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| mamba-ssm-tf32 | mamba-ssm | gpu | NVIDIA L40S (host f003873fb257) | mamba-ssm 2.3.2.post1+ge9594ce1c732 | opponent | 4.9 | 4.9..4.9 | 1 | 1.105 | - | 4.9 | - | - (stored whole) | 1.105 (whole/whole) | - | 1225.3 | 54.6 | max_abs_diff_vs_ours=3.815e-06, max_rel_diff_vs_ours=1.902e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-eager-tf32, torch-eager-bf16, mamba-ssm-fp32, mamba-ssm-tf32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### mamba2-forward / gaussian (neural shape full: B1 L2048 DM384)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-grid-logs.txt (nv2/v0139)`, ran on NVIDIA L40S (nv2, RunPod) job v0139

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 21.2 ms) = 0.371; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 25.9 ms) = 0.302. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ff77698) | identical | 7.8 | 7.8..7.8 | 1 | - | - | 7.8 | - | - (stored whole) | - | - | - | - | - | - | main board, one scored run | - | ok (main@c8ff77698 nv2/v0139 2026-10-11; identity vs the amd columns: n/a) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 25.9 | 25.9..25.9 | 1 | 0.302 | - | 25.9 | - | - (stored whole) | 0.302 (whole/whole) | - | 1191.4 | 3228.4 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 25.4 | 25.4..25.4 | 1 | 0.308 | - | 25.4 | - | - (stored whole) | 0.308 (whole/whole) | - | 1186.5 | 3228.4 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 28.5 | 28.5..28.5 | 1 | 0.275 | - | 28.5 | - | - (stored whole) | 0.275 (whole/whole) | - | 1256.0 | 3228.4 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 28.2 | 28.2..28.2 | 1 | 0.278 | - | 28.2 | - | - (stored whole) | 0.278 (whole/whole) | - | 1208.1 | 3228.4 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 21.2 | 21.2..21.2 | 1 | 0.371 | - | 21.2 | - | - (stored whole) | 0.371 (whole/whole) | - | 1326.1 | 1692.8 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 25.7 | 25.7..25.7 | 1 | 0.305 | - | 25.7 | - | - (stored whole) | 0.305 (whole/whole) | - | 1382.1 | 1692.8 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| mamba-ssm-fp32 | mamba-ssm | gpu | NVIDIA L40S (host f003873fb257) | mamba-ssm 2.3.2.post1+ge9594ce1c732 | opponent | 3.9 | 3.9..3.9 | 1 | 1.997 | - | 3.9 | - | - (stored whole) | 1.997 (whole/whole) | - | 1460.8 | 58.6 | max_abs_diff_vs_ours=1.907e-06, max_rel_diff_vs_ours=6.485e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| mamba-ssm-tf32 | mamba-ssm | gpu | NVIDIA L40S (host f003873fb257) | mamba-ssm 2.3.2.post1+ge9594ce1c732 | opponent | 2.9 | 2.9..2.9 | 1 | 2.692 | - | 2.9 | - | - (stored whole) | 2.692 (whole/whole) | - | 1289.7 | 58.6 | max_abs_diff_vs_ours=0.0005126, max_rel_diff_vs_ours=0.0001743 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-eager-tf32, torch-compile-fp32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16, mamba-ssm-fp32, mamba-ssm-tf32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### mamba3-forward / gaussian (neural shape full: B1 L2048 DM384)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v0999)`, ran on NVIDIA L40S (nv2, RunPod) job v0999

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 16.3 ms) = 0.251; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 12.4 ms) = 0.331. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@444f0b96e) | identical | 4.1 | 4.1..4.1 | 1 | - | - | 4.1 | - | - (stored whole) | - | - | - | - | - | - | main board, one scored run | - | ok (main@444f0b96e nv2/v0999 2026-10-09; identity vs the amd columns: n/a) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 84.0 | 84.0..84.0 | 1 | 0.049 | - | 84.0 | - | - (stored whole) | 0.049 (whole/whole) | - | 1191.2 | 219.7 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 83.4 | 83.4..83.4 | 1 | 0.049 | - | 83.4 | - | - (stored whole) | 0.049 (whole/whole) | - | 1186.5 | 219.7 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 12.4 | 12.4..12.4 | 1 | 0.331 | - | 12.4 | - | - (stored whole) | 0.331 (whole/whole) | - | 3464.6 | 83.6 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 16.5 | 16.5..16.5 | 1 | 0.248 | - | 16.5 | - | - (stored whole) | 0.248 (whole/whole) | - | 3072.8 | 83.6 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 83.9 | 83.9..83.9 | 1 | 0.049 | - | 83.9 | - | - (stored whole) | 0.049 (whole/whole) | - | 1313.3 | 212.6 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 16.3 | 16.3..16.3 | 1 | 0.251 | - | 16.3 | - | - (stored whole) | 0.251 (whole/whole) | - | 3813.6 | 90.5 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| mamba-ssm-fp32 | mamba-ssm | gpu | NVIDIA L40S (host f003873fb257) | mamba-ssm 2.3.2.post1+ge9594ce1c732 | opponent | 4.5 | 4.5..4.5 | 1 | 0.919 | - | 4.5 | - | - (stored whole) | 0.919 (whole/whole) | - | 1414.9 | 70.0 | max_abs_diff_vs_ours=0.001235, max_rel_diff_vs_ours=0.0005401 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| mamba-ssm-tf32 | mamba-ssm | gpu | NVIDIA L40S (host f003873fb257) | mamba-ssm 2.3.2.post1+ge9594ce1c732 | opponent | 4.1 | 4.1..4.1 | 1 | 0.999 | - | 4.1 | - | - (stored whole) | 0.999 (whole/whole) | - | 1410.9 | 70.0 | max_abs_diff_vs_ours=0.001376, max_rel_diff_vs_ours=0.0006018 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-eager-tf32, torch-compile-fp32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16, mamba-ssm-fp32, mamba-ssm-tf32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### mlp-train-step / gaussian (neural shape full: rows256 8-16-3)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-grid-logs.txt (nv2/v0139)`, ran on NVIDIA L40S (nv2, RunPod) job v0139

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 2.0 ms) = 1.021; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 1.5 ms) = 1.365. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ff77698) | identical | 2.0 | 2.0..2.0 | 1 | - | - | 2.0 | - | - (stored whole) | - | - | - | - | loss_first_step=1.160401, loss_last_step=1.123361, steps=2 | - | main board, one scored run | - | ok (main@c8ff77698 nv2/v0139 2026-10-11; identity vs the amd columns: n/a) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 1.5 | 1.5..1.5 | 1 | 1.365 | - | 1.5 | - | - (stored whole) | 1.365 (whole/whole) | - | 1015.5 | 16.3 | loss_first_step=1.160401, loss_last_step=1.123361, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 1.9 | 1.9..1.9 | 1 | 1.065 | - | 1.9 | - | - (stored whole) | 1.065 (whole/whole) | - | 1016.6 | 16.3 | loss_first_step=1.160392, loss_last_step=1.123355, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 2.0 | 2.0..2.0 | 1 | 1.017 | - | 2.0 | - | - (stored whole) | 1.017 (whole/whole) | - | 1104.5 | 16.3 | loss_first_step=1.160401, loss_last_step=1.123361, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 1.7 | 1.7..1.7 | 1 | 1.158 | - | 1.7 | - | - (stored whole) | 1.158 (whole/whole) | - | 1049.0 | 16.3 | loss_first_step=1.160392, loss_last_step=1.123355, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 2.0 | 2.0..2.0 | 1 | 1.021 | - | 2.0 | - | - (stored whole) | 1.021 (whole/whole) | - | 1229.1 | 16.3 | loss_first_step=1.160498, loss_last_step=1.123461, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 2.4 | 2.4..2.4 | 1 | 0.849 | - | 2.4 | - | - (stored whole) | 0.849 (whole/whole) | - | 1295.1 | 16.3 | loss_first_step=1.160498, loss_last_step=1.123462, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-eager-tf32, torch-compile-fp32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### samba-forward / bytes (neural shape full: B2 L512 DM384 V256 H6 FF1024 layers mamba3+attention+mamba3+attention)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v1049)`, ran on NVIDIA L40S (nv2, RunPod) job v1049

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 5.5 ms) = 0.910; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 12.1 ms) = 0.413. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@9f83ea479) | identical | 5.0 | 5.0..5.0 | 1 | - | - | 5.0 | - | - (stored whole) | - | - | - | - | mean_nll=5.635910 | - | main board, one scored run | - | ok (main@9f83ea479 nv2/v1049 2026-10-10; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 47.0 | 47.0..47.0 | 1 | 0.106 | - | 47.0 | - | - (stored whole) | 0.106 (whole/whole) | - | 1266.2 | 137.4 | mean_nll=5.635910 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 46.6 | 46.6..46.6 | 1 | 0.107 | - | 46.6 | - | - (stored whole) | 0.107 (whole/whole) | - | 1245.0 | 137.4 | mean_nll=5.635948 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 12.1 | 12.1..12.1 | 1 | 0.413 | - | 12.1 | - | - (stored whole) | 0.413 (whole/whole) | - | 1870.8 | 64.2 | mean_nll=5.635910 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 4.9 | 4.9..4.9 | 1 | 1.010 | - | 4.9 | - | - (stored whole) | 1.010 (whole/whole) | - | 1726.6 | 64.2 | mean_nll=5.635950 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 47.7 | 47.7..47.7 | 1 | 0.105 | - | 47.7 | - | - (stored whole) | 0.105 (whole/whole) | - | 1371.7 | 142.8 | mean_nll=5.635952 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 5.5 | 5.5..5.5 | 1 | 0.910 | - | 5.5 | - | - (stored whole) | 0.910 (whole/whole) | - | 2114.4 | 74.1 | mean_nll=5.635985 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-eager-tf32, torch-compile-fp32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### samba-train-step / bytes (neural shape full: B2 L512 DM384 V256 H6 FF1024 layers mamba3+attention+mamba3+attention)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-grid-logs.txt (nv2/v0067)`, ran on NVIDIA L40S (nv2, RunPod) job v0067

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 165.9 ms) = 0.255; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 46.3 ms) = 0.915. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@2afa942e3) | identical | 42.4 | 42.4..42.4 | 1 | - | - | 42.4 | - | - (stored whole) | - | - | - | - | loss_first_step=5.635910, loss_last_step=4.833934, steps=2 | - | main board, one scored run | - | ok (main@2afa942e3 nv2/v0067 2026-10-11; identity vs the amd columns: n/a) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 165.2 | 165.2..165.2 | 1 | 0.256 | - | 165.2 | - | - (stored whole) | 0.256 (whole/whole) | - | 1406.8 | 382.4 | loss_first_step=5.635910, loss_last_step=4.833934, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 147.5 | 147.5..147.5 | 1 | 0.287 | - | 147.5 | - | - (stored whole) | 0.287 (whole/whole) | - | 1400.1 | 382.4 | loss_first_step=5.635948, loss_last_step=4.833591, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 46.3 | 46.3..46.3 | 1 | 0.915 | - | 46.3 | - | - (stored whole) | 0.915 (whole/whole) | - | 2931.2 | 311.2 | loss_first_step=5.635910, loss_last_step=4.833934, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 43.2 | 43.2..43.2 | 1 | 0.982 | - | 43.2 | - | - (stored whole) | 0.982 (whole/whole) | - | 2375.7 | 316.8 | loss_first_step=5.635947, loss_last_step=4.833560, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 165.9 | 165.9..165.9 | 1 | 0.255 | - | 165.9 | - | - (stored whole) | 0.255 (whole/whole) | - | 1525.4 | 330.6 | loss_first_step=5.635952, loss_last_step=4.833967, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(timeout: null) (copied from opponents-specific-20261006; measured this run) |

memory, ours, torch-compile-bf16: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-eager-tf32, torch-compile-fp32, torch-compile-tf32, torch-eager-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### transformer-forward / gaussian (neural shape full: B1 L2048 DM384 H6 KV6 HD64 FF1024)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-grid-logs.txt (nv2/v0139)`, ran on NVIDIA L40S (nv2, RunPod) job v0139

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 3.4 ms) = 2.207; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 3.6 ms) = 2.108. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ff77698) | identical | 7.6 | 7.6..7.6 | 1 | - | - | 7.6 | - | - (stored whole) | - | - | - | - | - | - | main board, one scored run | - | ok (main@c8ff77698 nv2/v0139 2026-10-11; identity vs the amd columns: n/a) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 3.6 | 3.6..3.6 | 1 | 2.108 | - | 3.6 | - | - (stored whole) | 2.108 (whole/whole) | - | 1099.1 | 78.4 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 3.6 | 3.6..3.6 | 1 | 2.109 | - | 3.6 | - | - (stored whole) | 2.109 (whole/whole) | - | 1094.7 | 78.4 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 4.5 | 4.5..4.5 | 1 | 1.671 | - | 4.5 | - | - (stored whole) | 1.671 (whole/whole) | - | 993.5 | 46.9 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 4.3 | 4.3..4.3 | 1 | 1.769 | - | 4.3 | - | - (stored whole) | 1.769 (whole/whole) | - | 950.5 | 46.9 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 3.4 | 3.4..3.4 | 1 | 2.207 | - | 3.4 | - | - (stored whole) | 2.207 (whole/whole) | - | 1291.8 | 80.1 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 3.8 | 3.8..3.8 | 1 | 2.015 | - | 3.8 | - | - (stored whole) | 2.015 (whole/whole) | - | 1230.9 | 41.8 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-eager-tf32, torch-compile-fp32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

## Algorithm expansion

### adafactor / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/nvidia/campaign/measured/algos.adafactor.synthetic.rows=full/board.json (nv2/paired-e04405e5f8fcc9e4b47325d2c7bea05296c3d7b5ca9e2e10e290b5c4644f15bd)`, ran on d2fe567d3b88

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 16.6 ms) = 0.829. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 13.8 | 13.8..13.8 | 1 | - | - | - | 13.8 | - (stored kernel) | - | - | 1027.2 | 1450.0 | optimizer_update_error_vs_torch_fp32=3.444e-06, rel_fro_vs_torch_eager_fp32=1.051e-08 | - | main board, one scored run | wheel | ok (main@79fcd24a5 nv2/paired-e04405e5f8fcc9e4b47325d2c7bea05296c3d7b5ca9e2e10e290b5c4644f15bd 2026-10-14; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 16.6 | 16.6..16.6 | 1 | 0.829 | - | - | 16.6 | - (stored kernel) | 0.829 (kernel/kernel) | - | 1105.3 | 832.0 | optimizer_update_error_vs_torch_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 281.7 | 281.7..281.7 | 1 | 0.049 | - | - | 281.7 | - (stored kernel) | 0.049 (kernel/kernel) | - | 1158.9 | 832.0 | optimizer_update_error_vs_torch_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

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

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/nvidia/campaign/measured/algos.adagrad.synthetic.rows=full/board.json (nv2/paired-5fe579b0411a138733083f59a97fce679407262c5f6282e50c7c3deee4b3c737)`, ran on d2fe567d3b88

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 9.1 ms) = 0.956. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 8.7 | 8.7..8.7 | 1 | - | - | - | 8.7 | - (stored kernel) | - | - | 1025.3 | 1450.0 | optimizer_update_error_vs_torch_fp32=1.874e-06, rel_fro_vs_torch_eager_fp32=3.207e-09 | - | main board, one scored run | wheel | ok (main@79fcd24a5 nv2/paired-5fe579b0411a138733083f59a97fce679407262c5f6282e50c7c3deee4b3c737 2026-10-14; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 14.9 | 14.9..14.9 | 1 | 0.585 | - | - | 14.9 | - (stored kernel) | 0.585 (kernel/kernel) | - | 1034.3 | 960.0 | optimizer_update_error_vs_torch_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 9.1 | 9.1..9.1 | 1 | 0.956 | - | - | 9.1 | - (stored kernel) | 0.956 (kernel/kernel) | - | 1332.8 | 896.0 | optimizer_update_error_vs_torch_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

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

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/nvidia/campaign/measured/algos.adam.synthetic.rows=full/board.json (nv2/paired-e4b3d944a394b8532919b0be38365674898311bfd748fbadfd6ecee83d31ef7a)`, ran on d2fe567d3b88

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 16.7 ms) = 1.431. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 23.9 | 23.9..23.9 | 1 | - | - | - | 23.9 | - (stored kernel) | - | - | 2337.7 | 1706.0 | optimizer_update_error_vs_torch_fp32=8.844e-06, rel_fro_vs_torch_eager_fp32=3.34e-08 | - | main board, one scored run | wheel | ok (main@79fcd24a5 nv2/paired-e4b3d944a394b8532919b0be38365674898311bfd748fbadfd6ecee83d31ef7a 2026-10-14; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 16.7 | 16.7..16.7 | 1 | 1.431 | - | - | 16.7 | - (stored kernel) | 1.431 (kernel/kernel) | - | 1031.2 | 896.0 | optimizer_update_error_vs_torch_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 93.8 | 93.8..93.8 | 1 | 0.254 | - | - | 93.8 | - (stored kernel) | 0.254 (kernel/kernel) | - | 1477.1 | 960.0 | optimizer_update_error_vs_torch_fp32=1.295e-05, rel_fro_vs_torch_eager_fp32=4.891e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-fused-fp32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 11.3 | 11.3..11.3 | 1 | 2.114 | - | - | 11.3 | - (stored kernel) | 2.114 (kernel/kernel) | - | 980.1 | 896.0 | optimizer_update_error_vs_torch_fp32=5.957e-06, rel_fro_vs_torch_eager_fp32=2.25e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

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

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/nvidia/campaign/measured/algos.adamax.synthetic.rows=full/board.json (nv2/paired-725e0ddd7564f0a4f2e5cf2af894bb5a072bfc3d29eef4eb4d5c7f6f4f2ae743)`, ran on d2fe567d3b88

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 18.0 ms) = 0.619. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 11.1 | 11.1..11.1 | 1 | - | - | - | 11.1 | - (stored kernel) | - | - | 1025.9 | 1450.0 | optimizer_update_error_vs_torch_fp32=1.54e-06, rel_fro_vs_torch_eager_fp32=4.144e-09 | - | main board, one scored run | wheel | ok (main@79fcd24a5 nv2/paired-725e0ddd7564f0a4f2e5cf2af894bb5a072bfc3d29eef4eb4d5c7f6f4f2ae743 2026-10-14; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 18.0 | 18.0..18.0 | 1 | 0.619 | - | - | 18.0 | - (stored kernel) | 0.619 (kernel/kernel) | - | 1046.5 | 896.0 | optimizer_update_error_vs_torch_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 81.1 | 81.1..81.1 | 1 | 0.137 | - | - | 81.1 | - (stored kernel) | 0.137 (kernel/kernel) | - | 1476.9 | 960.0 | optimizer_update_error_vs_torch_fp32=2.376e-06, rel_fro_vs_torch_eager_fp32=6.395e-09 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

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

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/nvidia/campaign/measured/algos.adamw.synthetic.rows=full/board.json (nv2/paired-cd0fac0863b7eff0b97cd53e95474fb02e8e987b2362a6008b67ba19706f6d25)`, ran on d2fe567d3b88

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 17.9 ms) = 1.334. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 23.9 | 23.9..23.9 | 1 | - | - | - | 23.9 | - (stored kernel) | - | - | 2337.7 | 1706.0 | optimizer_update_error_vs_torch_fp32=8.841e-06, rel_fro_vs_torch_eager_fp32=3.34e-08 | - | main board, one scored run | wheel | ok (main@79fcd24a5 nv2/paired-cd0fac0863b7eff0b97cd53e95474fb02e8e987b2362a6008b67ba19706f6d25 2026-10-14; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 17.9 | 17.9..17.9 | 1 | 1.334 | - | - | 17.9 | - (stored kernel) | 1.334 (kernel/kernel) | - | 1034.5 | 896.0 | optimizer_update_error_vs_torch_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 94.9 | 94.9..94.9 | 1 | 0.252 | - | - | 94.9 | - (stored kernel) | 0.252 (kernel/kernel) | - | 1476.5 | 960.0 | optimizer_update_error_vs_torch_fp32=5.831e-05, rel_fro_vs_torch_eager_fp32=2.203e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-fused-fp32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 12.4 | 12.4..12.4 | 1 | 1.930 | - | - | 12.4 | - (stored kernel) | 1.930 (kernel/kernel) | - | 979.3 | 896.0 | optimizer_update_error_vs_torch_fp32=6.574e-05, rel_fro_vs_torch_eager_fp32=2.484e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

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

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0079)`, ran on NVIDIA L40S (nv2, RunPod) job v0079

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 317.7 | 317.7..317.7 | 1 | - | - | 317.7 | - | - (stored whole) | - | - | - | - | finite=True, r2=-0.124856, rmse=0.885949 | - | main board, one scored run | - | ok (main@a1c96a5ba nv2/v0079 2026-10-11; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 7400.8 | 7400.8..7400.8 | 1 | 0.043 | - | 7400.8 | 7400.8 | 0.00 (cpu-arm) | 0.043 (whole/whole) | - | 1381.8 | - | finite=True, r2=0.327402, rmse=0.685075 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha_1': 1e-06, 'alpha_2': 1e-06, 'fit_intercept': True, 'lambda_1': 1e-06, 'lambda_2': 1e-06, 'max_iter': 300, 'threshold_lambda': 10000.0, 'tol': 0.001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### autoarima / synthetic (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0031)`, ran on NVIDIA L40S (nv2, RunPod) job v0031

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@52d438ade) | identical | 12843.5 | 12843.5..12843.5 | 1 | - | - | 12843.5 | - | - (stored whole) | - | - | - | - | forecast_rmse=3.260829 | - | main board, one scored run | - | ok (main@52d438ade nv2/v0031 2026-10-11; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 822483.3 | 822483.3..822483.3 | 1 | 0.016 | - | 822487.8 | 822483.3 | 4.57 (upload_ms_untimed) | 0.016 (whole/whole) | - | 860.9 | 810.0 | forecast_rmse=32.552990 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| statsforecast-cpu | statsforecast | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | statsforecast 2.1.1 | opponent | 7452.1 | 7452.1..7452.1 | 1 | 1.723 | - | 7452.1 | 7452.1 | 0.00 (cpu-arm) | 1.723 (whole/whole) | - | 467.2 | - | forecast_rmse=17.543262 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, statsforecast-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'allow_intercept': True, 'ic': 'aicc', 'max_d': 1, 'max_p': 3, 'max_q': 3, 'seasonal': False, 'stepwise': False}. Rows: None. Timed: None.

mismatch: the likelihood optimizer and its stopping rule are each library's own

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### autoarima / taxi-hourly (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0031)`, ran on NVIDIA L40S (nv2, RunPod) job v0031

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@52d438ade) | identical | 14474.8 | 14474.8..14474.8 | 1 | - | - | 14474.8 | - | - (stored whole) | - | - | - | - | forecast_rmse=73.388358 | - | main board, one scored run | - | ok (main@52d438ade nv2/v0031 2026-10-11; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 157302.2 | 157302.2..157302.2 | 1 | 0.092 | - | 157306.6 | 157302.2 | 4.47 (upload_ms_untimed) | 0.092 (whole/whole) | - | 868.2 | 816.0 | forecast_rmse=106.817545 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| statsforecast-cpu | statsforecast | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | statsforecast 2.1.1 | opponent | 6148.5 | 6148.5..6148.5 | 1 | 2.354 | - | 6148.5 | 6148.5 | 0.00 (cpu-arm) | 2.354 (whole/whole) | - | 466.7 | - | forecast_rmse=68.213954 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, statsforecast-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'allow_intercept': True, 'ic': 'aicc', 'max_d': 1, 'max_p': 3, 'max_q': 3, 'seasonal': False, 'stepwise': False}. Rows: None. Timed: None.

mismatch: the likelihood optimizer and its stopping rule are each library's own

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### avgpool1d / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/nvidia/campaign/measured/algos.avgpool1d.synthetic.rows=full/board.json (nv2/paired-b60f39af305538a3ae74b3fc8be9ab4ff8f7799140e8d11ae45b2ce70a6ab972)`, ran on d2fe567d3b88

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 1.2 ms) = 0.877; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 0.9 ms) = 1.136. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 1.1 | 1.1..1.1 | 1 | - | - | - | 1.1 | - (stored kernel) | - | - | 1693.8 | 938.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | main board, one scored run | wheel | ok (main@79fcd24a5 nv2/paired-b60f39af305538a3ae74b3fc8be9ab4ff8f7799140e8d11ae45b2ce70a6ab972 2026-10-14; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 1.4 | 1.4..1.4 | 1 | 0.754 | - | - | 1.4 | - (stored kernel) | 0.754 (kernel/kernel) | - | 830.2 | 224.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 0.9 | 0.9..0.9 | 1 | 1.136 | - | - | 0.9 | - (stored kernel) | 1.136 (kernel/kernel) | - | 1056.9 | 224.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 1.1 | 1.1..1.1 | 1 | 0.950 | - | - | 1.1 | - (stored kernel) | 0.950 (kernel/kernel) | - | 828.4 | 224.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 1.5 | 1.5..1.5 | 1 | 0.713 | - | - | 1.5 | - (stored kernel) | 0.713 (kernel/kernel) | - | 1007.5 | 224.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 1.2 | 1.2..1.2 | 1 | 0.877 | - | - | 1.2 | - (stored kernel) | 0.877 (kernel/kernel) | - | 829.4 | 224.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 2.0 | 2.0..2.0 | 1 | 0.519 | - | - | 2.0 | - (stored kernel) | 0.519 (kernel/kernel) | - | 1004.1 | 224.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'ceil_mode': False, 'count_include_pad': True, 'kernel_size': 2, 'padding': 0, 'stride': 2}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-compile-tf32 | torch-eager-bf16 | torch-eager-fp32 | torch-eager-tf32 |
|---|---||---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| ceil_mode | false | false | false | false | false | false | false |
| count_include_pad | true | true | true | true | true | true | true |
| input_arrays | [{"dtype": "float32", "name": "x", "sha256": "128bb6c8835123e5f00ffe3700a26223ba8b37f7280afaec092571b2861be93c", "shape": [64, 64, 4096]}] | [{"dtype": "float32", "name": "x", "sha256": "128bb6c8835123e5f00ffe3700a26223ba8b37f7280afaec092571b2861be93c", "shape": [64, 64, 4096]}] | [{"dtype": "float32", "name": "x", "sha256": "128bb6c8835123e5f00ffe3700a26223ba8b37f7280afaec092571b2861be93c", "shape": [64, 64, 4096]}] | [{"dtype": "float32", "name": "x", "sha256": "128bb6c8835123e5f00ffe3700a26223ba8b37f7280afaec092571b2861be93c", "shape": [64, 64, 4096]}] | [{"dtype": "float32", "name": "x", "sha256": "128bb6c8835123e5f00ffe3700a26223ba8b37f7280afaec092571b2861be93c", "shape": [64, 64, 4096]}] | [{"dtype": "float32", "name": "x", "sha256": "128bb6c8835123e5f00ffe3700a26223ba8b37f7280afaec092571b2861be93c", "shape": [64, 64, 4096]}] | [{"dtype": "float32", "name": "x", "sha256": "128bb6c8835123e5f00ffe3700a26223ba8b37f7280afaec092571b2861be93c", "shape": [64, 64, 4096]}] |
| kernel_size | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| padding | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 | 7 | 7 |
| stride | 2 | 2 | 2 | 2 | 2 | 2 | 2 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | hardware | version | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | NVIDIA L40S (nv2, RunPod) | 0.8.44 | Xq | - | 0.2 | 0.2..0.2 | 1 | - | - | - | 0.2 | - (stored kernel) | - | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.2 | 0.2..0.2 | 1 | 1.002 | - | - | 0.2 | - (stored kernel) | 1.002 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.2 | 0.2..0.2 | 1 | 0.772 | - | - | 0.2 | - (stored kernel) | 0.772 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-tf32 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.2 | 0.2..0.2 | 1 | 1.004 | - | - | 0.2 | - (stored kernel) | 1.004 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-tf32 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.3 | 0.3..0.3 | 1 | 0.722 | - | - | 0.3 | - (stored kernel) | 0.722 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.2 | 0.2..0.2 | 1 | 0.911 | - | - | 0.2 | - (stored kernel) | 0.911 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.3 | 0.3..0.3 | 1 | 0.653 | - | - | 0.3 | - (stored kernel) | 0.653 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-tf32: forward(x) (no autograd)(Xq)

inference call, torch-compile-tf32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### avgpool2d / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/nvidia/campaign/measured/algos.avgpool2d.synthetic.rows=full/board.json (nv2/paired-9c458807517b437a6b86c2a9e89add414e431c532004f8c0725fc16bea6159fa)`, ran on d2fe567d3b88

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 1.8 ms) = 0.891; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 1.2 ms) = 1.328. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 1.6 | 1.6..1.6 | 1 | - | - | - | 1.6 | - (stored kernel) | - | - | 1894.7 | 1194.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | main board, one scored run | wheel | ok (main@79fcd24a5 nv2/paired-9c458807517b437a6b86c2a9e89add414e431c532004f8c0725fc16bea6159fa 2026-10-14; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 1.6 | 1.6..1.6 | 1 | 1.005 | - | - | 1.6 | - (stored kernel) | 1.005 (kernel/kernel) | - | 1013.1 | 541.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 1.2 | 1.2..1.2 | 1 | 1.328 | - | - | 1.2 | - (stored kernel) | 1.328 (kernel/kernel) | - | 1240.9 | 540.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 250.2 | 250.2..250.2 | 1 | 0.006 | - | - | 250.2 | - (stored kernel) | 0.006 (kernel/kernel) | - | 1008.4 | 541.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 4.6 | 4.6..4.6 | 1 | 0.346 | - | - | 4.6 | - (stored kernel) | 0.346 (kernel/kernel) | - | 1193.8 | 541.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 1.8 | 1.8..1.8 | 1 | 0.891 | - | - | 1.8 | - (stored kernel) | 0.891 (kernel/kernel) | - | 1012.8 | 541.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 2.0 | 2.0..2.0 | 1 | 0.800 | - | - | 2.0 | - (stored kernel) | 0.800 (kernel/kernel) | - | 1184.8 | 541.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'ceil_mode': False, 'count_include_pad': True, 'divisor_override': None, 'kernel_size': 2, 'padding': 0, 'stride': 2}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-compile-tf32 | torch-eager-bf16 | torch-eager-fp32 | torch-eager-tf32 |
|---|---||---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| ceil_mode | false | false | false | false | false | false | false |
| count_include_pad | true | true | true | true | true | true | true |
| divisor_override | null | null | null | null | null | null | null |
| input_arrays | [{"dtype": "float32", "name": "x", "sha256": "2df96f3879e0a6060dab5f7b7561b7291c5a106d1a8d4e116fc5e44e3023f819", "shape": [64, 64, 112, 112]}] | [{"dtype": "float32", "name": "x", "sha256": "2df96f3879e0a6060dab5f7b7561b7291c5a106d1a8d4e116fc5e44e3023f819", "shape": [64, 64, 112, 112]}] | [{"dtype": "float32", "name": "x", "sha256": "2df96f3879e0a6060dab5f7b7561b7291c5a106d1a8d4e116fc5e44e3023f819", "shape": [64, 64, 112, 112]}] | [{"dtype": "float32", "name": "x", "sha256": "2df96f3879e0a6060dab5f7b7561b7291c5a106d1a8d4e116fc5e44e3023f819", "shape": [64, 64, 112, 112]}] | [{"dtype": "float32", "name": "x", "sha256": "2df96f3879e0a6060dab5f7b7561b7291c5a106d1a8d4e116fc5e44e3023f819", "shape": [64, 64, 112, 112]}] | [{"dtype": "float32", "name": "x", "sha256": "2df96f3879e0a6060dab5f7b7561b7291c5a106d1a8d4e116fc5e44e3023f819", "shape": [64, 64, 112, 112]}] | [{"dtype": "float32", "name": "x", "sha256": "2df96f3879e0a6060dab5f7b7561b7291c5a106d1a8d4e116fc5e44e3023f819", "shape": [64, 64, 112, 112]}] |
| kernel_size | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| padding | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 | 7 | 7 |
| stride | 2 | 2 | 2 | 2 | 2 | 2 | 2 |

accepted difference: torch-eager-fp32 divisor_override: None selects the kernel-area divisor on both AvgPool2d implementations; all active arguments remain compared

accepted difference: torch-compile-fp32 divisor_override: None selects the kernel-area divisor on both AvgPool2d implementations; all active arguments remain compared

accepted difference: torch-eager-tf32 divisor_override: None selects the kernel-area divisor on both AvgPool2d implementations; all active arguments remain compared

accepted difference: torch-compile-tf32 divisor_override: None selects the kernel-area divisor on both AvgPool2d implementations; all active arguments remain compared

accepted difference: torch-eager-bf16 divisor_override: None selects the kernel-area divisor on both AvgPool2d implementations; all active arguments remain compared

accepted difference: torch-compile-bf16 divisor_override: None selects the kernel-area divisor on both AvgPool2d implementations; all active arguments remain compared

Inference (each arm predicts with its own model from the fit rounds above):

| arm | hardware | version | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | NVIDIA L40S (nv2, RunPod) | 0.8.44 | Xq | - | 0.5 | 0.5..0.5 | 1 | - | - | - | 0.5 | - (stored kernel) | - | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.4 | 0.4..0.4 | 1 | 1.041 | - | - | 0.4 | - (stored kernel) | 1.041 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.6 | 0.6..0.6 | 1 | 0.810 | - | - | 0.6 | - (stored kernel) | 0.810 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-tf32 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.5 | 0.5..0.5 | 1 | 0.957 | - | - | 0.5 | - (stored kernel) | 0.957 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-tf32 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.6 | 0.6..0.6 | 1 | 0.817 | - | - | 0.6 | - (stored kernel) | 0.817 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.5 | 0.5..0.5 | 1 | 0.993 | - | - | 0.5 | - (stored kernel) | 0.993 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.6 | 0.6..0.6 | 1 | 0.813 | - | - | 0.6 | - (stored kernel) | 0.813 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-tf32: forward(x) (no autograd)(Xq)

inference call, torch-compile-tf32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### batchnorm1d / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/nvidia-wave5/campaign/measured/algos.batchnorm1d.synthetic.rows=full/board.json (nv2/paired-b334e834c14ef661f4c0930df16dbf266750a5cc32f6e43a4aba239f3a102f27)`, ran on d2fe567d3b88

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 1.7 ms) = 0.928; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 1.3 ms) = 1.212. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 1.6 | 1.6..1.6 | 1 | - | - | - | 1.6 | - (stored kernel) | - | - | 1825.0 | 938.0 | max_rel_diff_vs_torch_eager_fp32=0.002205, rel_fro_vs_torch_eager_fp32=8.002e-08 | - | main board, one scored run | wheel | ok (main@d11b82091 nv2/paired-b334e834c14ef661f4c0930df16dbf266750a5cc32f6e43a4aba239f3a102f27 2026-10-14; identity vs the amd columns: n/a) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 1.8 | 1.8..1.8 | 1 | 0.898 | - | - | 1.8 | - (stored kernel) | 0.898 (kernel/kernel) | - | 971.5 | 320.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 1.3 | 1.3..1.3 | 1 | 1.212 | - | - | 1.3 | - (stored kernel) | 1.212 (kernel/kernel) | - | 1173.0 | 320.0 | max_rel_diff_vs_torch_eager_fp32=0.002792, rel_fro_vs_torch_eager_fp32=5.109e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 1.8 | 1.8..1.8 | 1 | 0.870 | - | - | 1.8 | - (stored kernel) | 0.870 (kernel/kernel) | - | 975.2 | 320.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 1.5 | 1.5..1.5 | 1 | 1.088 | - | - | 1.5 | - (stored kernel) | 1.088 (kernel/kernel) | - | 1115.4 | 320.0 | max_rel_diff_vs_torch_eager_fp32=0.002792, rel_fro_vs_torch_eager_fp32=5.109e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 1.7 | 1.7..1.7 | 1 | 0.928 | - | - | 1.7 | - (stored kernel) | 0.928 (kernel/kernel) | - | 977.2 | 320.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 4.4 | 4.4..4.4 | 1 | 0.364 | - | - | 4.4 | - (stored kernel) | 0.364 (kernel/kernel) | - | 1109.2 | 320.0 | max_rel_diff_vs_torch_eager_fp32=0.002792, rel_fro_vs_torch_eager_fp32=5.109e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'affine': True, 'eps': 1e-05, 'momentum': 0.1, 'num_features': 256, 'track_running_stats': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-compile-tf32 | torch-eager-bf16 | torch-eager-fp32 | torch-eager-tf32 |
|---|---||---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| affine | true | true | true | true | true | true | true |
| eps | 1e-05 | 1e-05 | 1e-05 | 1e-05 | 1e-05 | 1e-05 | 1e-05 |
| input_arrays | [{"dtype": "float32", "name": "x", "sha256": "128bb6c8835123e5f00ffe3700a26223ba8b37f7280afaec092571b2861be93c", "shape": [64, 256, 1024]}, {"dtype": "float32", "name": "bias", "sha256": "5f70bf18a086007016e948b04aed3b82103a36bea41755b6cddfaf10ace3c6ef", "shape": [256]}, {"dtype": "int64", "name": "num_batches_tracked", "sha256": "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc", "shape": [1]}, {"dtype": "float32", "name": "running_mean", "sha256": "5f70bf18a086007016e948b04aed3b82103a36bea41755b6cddfaf10ace3c6ef", "shape": [256]}, {"dtype": "float32", "name": "running_var", "sha256": "893a106828fbdb9521e1d868c985aab7ad2ae2f606edc55329265a5e7676006c", "shape": [256]}, {"dtype": "float32", "name": "weight", "sha256": "893a106828fbdb9521e1d868c985aab7ad2ae2f606edc55329265a5e7676006c", "shape": [256]}] | [{"dtype": "float32", "name": "x", "sha256": "128bb6c8835123e5f00ffe3700a26223ba8b37f7280afaec092571b2861be93c", "shape": [64, 256, 1024]}, {"dtype": "float32", "name": "bias", "sha256": "5f70bf18a086007016e948b04aed3b82103a36bea41755b6cddfaf10ace3c6ef", "shape": [256]}, {"dtype": "int64", "name": "num_batches_tracked", "sha256": "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc", "shape": [1]}, {"dtype": "float32", "name": "running_mean", "sha256": "5f70bf18a086007016e948b04aed3b82103a36bea41755b6cddfaf10ace3c6ef", "shape": [256]}, {"dtype": "float32", "name": "running_var", "sha256": "893a106828fbdb9521e1d868c985aab7ad2ae2f606edc55329265a5e7676006c", "shape": [256]}, {"dtype": "float32", "name": "weight", "sha256": "893a106828fbdb9521e1d868c985aab7ad2ae2f606edc55329265a5e7676006c", "shape": [256]}] | [{"dtype": "float32", "name": "x", "sha256": "128bb6c8835123e5f00ffe3700a26223ba8b37f7280afaec092571b2861be93c", "shape": [64, 256, 1024]}, {"dtype": "float32", "name": "bias", "sha256": "5f70bf18a086007016e948b04aed3b82103a36bea41755b6cddfaf10ace3c6ef", "shape": [256]}, {"dtype": "int64", "name": "num_batches_tracked", "sha256": "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc", "shape": [1]}, {"dtype": "float32", "name": "running_mean", "sha256": "5f70bf18a086007016e948b04aed3b82103a36bea41755b6cddfaf10ace3c6ef", "shape": [256]}, {"dtype": "float32", "name": "running_var", "sha256": "893a106828fbdb9521e1d868c985aab7ad2ae2f606edc55329265a5e7676006c", "shape": [256]}, {"dtype": "float32", "name": "weight", "sha256": "893a106828fbdb9521e1d868c985aab7ad2ae2f606edc55329265a5e7676006c", "shape": [256]}] | [{"dtype": "float32", "name": "x", "sha256": "128bb6c8835123e5f00ffe3700a26223ba8b37f7280afaec092571b2861be93c", "shape": [64, 256, 1024]}, {"dtype": "float32", "name": "bias", "sha256": "5f70bf18a086007016e948b04aed3b82103a36bea41755b6cddfaf10ace3c6ef", "shape": [256]}, {"dtype": "int64", "name": "num_batches_tracked", "sha256": "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc", "shape": [1]}, {"dtype": "float32", "name": "running_mean", "sha256": "5f70bf18a086007016e948b04aed3b82103a36bea41755b6cddfaf10ace3c6ef", "shape": [256]}, {"dtype": "float32", "name": "running_var", "sha256": "893a106828fbdb9521e1d868c985aab7ad2ae2f606edc55329265a5e7676006c", "shape": [256]}, {"dtype": "float32", "name": "weight", "sha256": "893a106828fbdb9521e1d868c985aab7ad2ae2f606edc55329265a5e7676006c", "shape": [256]}] | [{"dtype": "float32", "name": "x", "sha256": "128bb6c8835123e5f00ffe3700a26223ba8b37f7280afaec092571b2861be93c", "shape": [64, 256, 1024]}, {"dtype": "float32", "name": "bias", "sha256": "5f70bf18a086007016e948b04aed3b82103a36bea41755b6cddfaf10ace3c6ef", "shape": [256]}, {"dtype": "int64", "name": "num_batches_tracked", "sha256": "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc", "shape": [1]}, {"dtype": "float32", "name": "running_mean", "sha256": "5f70bf18a086007016e948b04aed3b82103a36bea41755b6cddfaf10ace3c6ef", "shape": [256]}, {"dtype": "float32", "name": "running_var", "sha256": "893a106828fbdb9521e1d868c985aab7ad2ae2f606edc55329265a5e7676006c", "shape": [256]}, {"dtype": "float32", "name": "weight", "sha256": "893a106828fbdb9521e1d868c985aab7ad2ae2f606edc55329265a5e7676006c", "shape": [256]}] | [{"dtype": "float32", "name": "x", "sha256": "128bb6c8835123e5f00ffe3700a26223ba8b37f7280afaec092571b2861be93c", "shape": [64, 256, 1024]}, {"dtype": "float32", "name": "bias", "sha256": "5f70bf18a086007016e948b04aed3b82103a36bea41755b6cddfaf10ace3c6ef", "shape": [256]}, {"dtype": "int64", "name": "num_batches_tracked", "sha256": "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc", "shape": [1]}, {"dtype": "float32", "name": "running_mean", "sha256": "5f70bf18a086007016e948b04aed3b82103a36bea41755b6cddfaf10ace3c6ef", "shape": [256]}, {"dtype": "float32", "name": "running_var", "sha256": "893a106828fbdb9521e1d868c985aab7ad2ae2f606edc55329265a5e7676006c", "shape": [256]}, {"dtype": "float32", "name": "weight", "sha256": "893a106828fbdb9521e1d868c985aab7ad2ae2f606edc55329265a5e7676006c", "shape": [256]}] | [{"dtype": "float32", "name": "x", "sha256": "128bb6c8835123e5f00ffe3700a26223ba8b37f7280afaec092571b2861be93c", "shape": [64, 256, 1024]}, {"dtype": "float32", "name": "bias", "sha256": "5f70bf18a086007016e948b04aed3b82103a36bea41755b6cddfaf10ace3c6ef", "shape": [256]}, {"dtype": "int64", "name": "num_batches_tracked", "sha256": "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc", "shape": [1]}, {"dtype": "float32", "name": "running_mean", "sha256": "5f70bf18a086007016e948b04aed3b82103a36bea41755b6cddfaf10ace3c6ef", "shape": [256]}, {"dtype": "float32", "name": "running_var", "sha256": "893a106828fbdb9521e1d868c985aab7ad2ae2f606edc55329265a5e7676006c", "shape": [256]}, {"dtype": "float32", "name": "weight", "sha256": "893a106828fbdb9521e1d868c985aab7ad2ae2f606edc55329265a5e7676006c", "shape": [256]}] |
| momentum | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| num_features | 256 | 256 | 256 | 256 | 256 | 256 | 256 |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 | 7 | 7 |
| track_running_stats | true | true | true | true | true | true | true |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | hardware | version | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | NVIDIA L40S (nv2, RunPod) | 0.8.44 | Xq | - | 0.4 | 0.4..0.4 | 1 | - | - | - | 0.4 | - (stored kernel) | - | - | max_rel_diff_vs_torch_eager_fp32=0.002205, rel_fro_vs_torch_eager_fp32=8.002e-08 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.3 | 0.3..0.3 | 1 | 1.321 | - | - | 0.3 | - (stored kernel) | 1.321 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.4 | 0.4..0.4 | 1 | 1.065 | - | - | 0.4 | - (stored kernel) | 1.065 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.002792, rel_fro_vs_torch_eager_fp32=5.109e-08 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-tf32 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.3 | 0.3..0.3 | 1 | 1.256 | - | - | 0.3 | - (stored kernel) | 1.256 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-tf32 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.4 | 0.4..0.4 | 1 | 0.991 | - | - | 0.4 | - (stored kernel) | 0.991 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.002792, rel_fro_vs_torch_eager_fp32=5.109e-08 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.3 | 0.3..0.3 | 1 | 1.253 | - | - | 0.3 | - (stored kernel) | 1.253 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.4 | 0.4..0.4 | 1 | 0.940 | - | - | 0.4 | - (stored kernel) | 0.940 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.002792, rel_fro_vs_torch_eager_fp32=5.109e-08 | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-tf32: forward(x) (no autograd)(Xq)

inference call, torch-compile-tf32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### batchnorm2d / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/nvidia-wave5/campaign/measured/algos.batchnorm2d.synthetic.rows=full/board.json (nv2/paired-70ab80420ebc83b9c9c63544d4328ea8cc8b42742652f4baec3e64cbc1d07ae5)`, ran on d2fe567d3b88

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 1.6 ms) = 0.968; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 1.1 ms) = 1.437. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 1.6 | 1.6..1.6 | 1 | - | - | - | 1.6 | - (stored kernel) | - | - | 1747.9 | 682.0 | max_rel_diff_vs_torch_eager_fp32=0.001584, rel_fro_vs_torch_eager_fp32=1.314e-07 | - | main board, one scored run | wheel | ok (main@d11b82091 nv2/paired-70ab80420ebc83b9c9c63544d4328ea8cc8b42742652f4baec3e64cbc1d07ae5 2026-10-14; identity vs the amd columns: n/a) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 1.3 | 1.3..1.3 | 1 | 1.251 | - | - | 1.3 | - (stored kernel) | 1.251 (kernel/kernel) | - | 916.0 | 250.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 1.1 | 1.1..1.1 | 1 | 1.437 | - | - | 1.1 | - (stored kernel) | 1.437 (kernel/kernel) | - | 1118.5 | 246.0 | max_rel_diff_vs_torch_eager_fp32=0.001863, rel_fro_vs_torch_eager_fp32=4.327e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 1.3 | 1.3..1.3 | 1 | 1.218 | - | - | 1.3 | - (stored kernel) | 1.218 (kernel/kernel) | - | 916.7 | 250.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 2.3 | 2.3..2.3 | 1 | 0.689 | - | - | 2.3 | - (stored kernel) | 0.689 (kernel/kernel) | - | 1053.9 | 250.0 | max_rel_diff_vs_torch_eager_fp32=0.001863, rel_fro_vs_torch_eager_fp32=4.327e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 1.6 | 1.6..1.6 | 1 | 0.968 | - | - | 1.6 | - (stored kernel) | 0.968 (kernel/kernel) | - | 913.8 | 250.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 2.3 | 2.3..2.3 | 1 | 0.699 | - | - | 2.3 | - (stored kernel) | 0.699 (kernel/kernel) | - | 1043.9 | 250.0 | max_rel_diff_vs_torch_eager_fp32=0.001863, rel_fro_vs_torch_eager_fp32=4.327e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'affine': True, 'eps': 1e-05, 'momentum': 0.1, 'num_features': 64, 'track_running_stats': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-compile-tf32 | torch-eager-bf16 | torch-eager-fp32 | torch-eager-tf32 |
|---|---||---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| affine | true | true | true | true | true | true | true |
| eps | 1e-05 | 1e-05 | 1e-05 | 1e-05 | 1e-05 | 1e-05 | 1e-05 |
| input_arrays | [{"dtype": "float32", "name": "x", "sha256": "cd7e0a7d8c0c594cca519b9cf49c8f086513282644cb8b6c06771dcb98504f2d", "shape": [64, 64, 56, 56]}, {"dtype": "float32", "name": "bias", "sha256": "5341e6b2646979a70e57653007a1f310169421ec9bdd9f1a5648f75ade005af1", "shape": [64]}, {"dtype": "int64", "name": "num_batches_tracked", "sha256": "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc", "shape": [1]}, {"dtype": "float32", "name": "running_mean", "sha256": "5341e6b2646979a70e57653007a1f310169421ec9bdd9f1a5648f75ade005af1", "shape": [64]}, {"dtype": "float32", "name": "running_var", "sha256": "2f20cd03c9cd392a406c56232b0ff93a15f6d6d7da79086bfa14f55d4a4031b0", "shape": [64]}, {"dtype": "float32", "name": "weight", "sha256": "2f20cd03c9cd392a406c56232b0ff93a15f6d6d7da79086bfa14f55d4a4031b0", "shape": [64]}] | [{"dtype": "float32", "name": "x", "sha256": "cd7e0a7d8c0c594cca519b9cf49c8f086513282644cb8b6c06771dcb98504f2d", "shape": [64, 64, 56, 56]}, {"dtype": "float32", "name": "bias", "sha256": "5341e6b2646979a70e57653007a1f310169421ec9bdd9f1a5648f75ade005af1", "shape": [64]}, {"dtype": "int64", "name": "num_batches_tracked", "sha256": "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc", "shape": [1]}, {"dtype": "float32", "name": "running_mean", "sha256": "5341e6b2646979a70e57653007a1f310169421ec9bdd9f1a5648f75ade005af1", "shape": [64]}, {"dtype": "float32", "name": "running_var", "sha256": "2f20cd03c9cd392a406c56232b0ff93a15f6d6d7da79086bfa14f55d4a4031b0", "shape": [64]}, {"dtype": "float32", "name": "weight", "sha256": "2f20cd03c9cd392a406c56232b0ff93a15f6d6d7da79086bfa14f55d4a4031b0", "shape": [64]}] | [{"dtype": "float32", "name": "x", "sha256": "cd7e0a7d8c0c594cca519b9cf49c8f086513282644cb8b6c06771dcb98504f2d", "shape": [64, 64, 56, 56]}, {"dtype": "float32", "name": "bias", "sha256": "5341e6b2646979a70e57653007a1f310169421ec9bdd9f1a5648f75ade005af1", "shape": [64]}, {"dtype": "int64", "name": "num_batches_tracked", "sha256": "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc", "shape": [1]}, {"dtype": "float32", "name": "running_mean", "sha256": "5341e6b2646979a70e57653007a1f310169421ec9bdd9f1a5648f75ade005af1", "shape": [64]}, {"dtype": "float32", "name": "running_var", "sha256": "2f20cd03c9cd392a406c56232b0ff93a15f6d6d7da79086bfa14f55d4a4031b0", "shape": [64]}, {"dtype": "float32", "name": "weight", "sha256": "2f20cd03c9cd392a406c56232b0ff93a15f6d6d7da79086bfa14f55d4a4031b0", "shape": [64]}] | [{"dtype": "float32", "name": "x", "sha256": "cd7e0a7d8c0c594cca519b9cf49c8f086513282644cb8b6c06771dcb98504f2d", "shape": [64, 64, 56, 56]}, {"dtype": "float32", "name": "bias", "sha256": "5341e6b2646979a70e57653007a1f310169421ec9bdd9f1a5648f75ade005af1", "shape": [64]}, {"dtype": "int64", "name": "num_batches_tracked", "sha256": "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc", "shape": [1]}, {"dtype": "float32", "name": "running_mean", "sha256": "5341e6b2646979a70e57653007a1f310169421ec9bdd9f1a5648f75ade005af1", "shape": [64]}, {"dtype": "float32", "name": "running_var", "sha256": "2f20cd03c9cd392a406c56232b0ff93a15f6d6d7da79086bfa14f55d4a4031b0", "shape": [64]}, {"dtype": "float32", "name": "weight", "sha256": "2f20cd03c9cd392a406c56232b0ff93a15f6d6d7da79086bfa14f55d4a4031b0", "shape": [64]}] | [{"dtype": "float32", "name": "x", "sha256": "cd7e0a7d8c0c594cca519b9cf49c8f086513282644cb8b6c06771dcb98504f2d", "shape": [64, 64, 56, 56]}, {"dtype": "float32", "name": "bias", "sha256": "5341e6b2646979a70e57653007a1f310169421ec9bdd9f1a5648f75ade005af1", "shape": [64]}, {"dtype": "int64", "name": "num_batches_tracked", "sha256": "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc", "shape": [1]}, {"dtype": "float32", "name": "running_mean", "sha256": "5341e6b2646979a70e57653007a1f310169421ec9bdd9f1a5648f75ade005af1", "shape": [64]}, {"dtype": "float32", "name": "running_var", "sha256": "2f20cd03c9cd392a406c56232b0ff93a15f6d6d7da79086bfa14f55d4a4031b0", "shape": [64]}, {"dtype": "float32", "name": "weight", "sha256": "2f20cd03c9cd392a406c56232b0ff93a15f6d6d7da79086bfa14f55d4a4031b0", "shape": [64]}] | [{"dtype": "float32", "name": "x", "sha256": "cd7e0a7d8c0c594cca519b9cf49c8f086513282644cb8b6c06771dcb98504f2d", "shape": [64, 64, 56, 56]}, {"dtype": "float32", "name": "bias", "sha256": "5341e6b2646979a70e57653007a1f310169421ec9bdd9f1a5648f75ade005af1", "shape": [64]}, {"dtype": "int64", "name": "num_batches_tracked", "sha256": "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc", "shape": [1]}, {"dtype": "float32", "name": "running_mean", "sha256": "5341e6b2646979a70e57653007a1f310169421ec9bdd9f1a5648f75ade005af1", "shape": [64]}, {"dtype": "float32", "name": "running_var", "sha256": "2f20cd03c9cd392a406c56232b0ff93a15f6d6d7da79086bfa14f55d4a4031b0", "shape": [64]}, {"dtype": "float32", "name": "weight", "sha256": "2f20cd03c9cd392a406c56232b0ff93a15f6d6d7da79086bfa14f55d4a4031b0", "shape": [64]}] | [{"dtype": "float32", "name": "x", "sha256": "cd7e0a7d8c0c594cca519b9cf49c8f086513282644cb8b6c06771dcb98504f2d", "shape": [64, 64, 56, 56]}, {"dtype": "float32", "name": "bias", "sha256": "5341e6b2646979a70e57653007a1f310169421ec9bdd9f1a5648f75ade005af1", "shape": [64]}, {"dtype": "int64", "name": "num_batches_tracked", "sha256": "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc", "shape": [1]}, {"dtype": "float32", "name": "running_mean", "sha256": "5341e6b2646979a70e57653007a1f310169421ec9bdd9f1a5648f75ade005af1", "shape": [64]}, {"dtype": "float32", "name": "running_var", "sha256": "2f20cd03c9cd392a406c56232b0ff93a15f6d6d7da79086bfa14f55d4a4031b0", "shape": [64]}, {"dtype": "float32", "name": "weight", "sha256": "2f20cd03c9cd392a406c56232b0ff93a15f6d6d7da79086bfa14f55d4a4031b0", "shape": [64]}] |
| momentum | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| num_features | 64 | 64 | 64 | 64 | 64 | 64 | 64 |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 | 7 | 7 |
| track_running_stats | true | true | true | true | true | true | true |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | hardware | version | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | NVIDIA L40S (nv2, RunPod) | 0.8.44 | Xq | - | 0.3 | 0.3..0.3 | 1 | - | - | - | 0.3 | - (stored kernel) | - | - | max_rel_diff_vs_torch_eager_fp32=0.001584, rel_fro_vs_torch_eager_fp32=1.314e-07 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.2 | 0.2..0.2 | 1 | 1.255 | - | - | 0.2 | - (stored kernel) | 1.255 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.2 | 0.2..0.2 | 1 | 1.098 | - | - | 0.2 | - (stored kernel) | 1.098 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.001863, rel_fro_vs_torch_eager_fp32=4.327e-08 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-tf32 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.2 | 0.2..0.2 | 1 | 1.233 | - | - | 0.2 | - (stored kernel) | 1.233 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-tf32 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.3 | 0.3..0.3 | 1 | 0.916 | - | - | 0.3 | - (stored kernel) | 0.916 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.001863, rel_fro_vs_torch_eager_fp32=4.327e-08 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.2 | 0.2..0.2 | 1 | 1.096 | - | - | 0.2 | - (stored kernel) | 1.096 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.3 | 0.3..0.3 | 1 | 0.990 | - | - | 0.3 | - (stored kernel) | 0.990 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.001863, rel_fro_vs_torch_eager_fp32=4.327e-08 | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-tf32: forward(x) (no autograd)(Xq)

inference call, torch-compile-tf32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### bayesian-ridge / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0079)`, ran on NVIDIA L40S (nv2, RunPod) job v0079

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 611.4 | 611.4..611.4 | 1 | - | - | 611.4 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.317909, rmse=0.689893 | - | main board, one scored run | - | ok (main@a1c96a5ba nv2/v0079 2026-10-11; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 8161.0 | 8161.0..8161.0 | 1 | 0.075 | - | 8161.0 | 8161.0 | 0.00 (cpu-arm) | 0.075 (whole/whole) | - | 3653.3 | - | finite=True, r2=-395441.818968, rmse=525.293799 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha_1': 1e-06, 'alpha_2': 1e-06, 'fit_intercept': True, 'lambda_1': 1e-06, 'lambda_2': 1e-06, 'max_iter': 300, 'tol': 0.001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### bernoulli-nb / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1090)`, ran on NVIDIA L40S (nv2, RunPod) job v1090

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 52.3 | 52.3..52.3 | 1 | - | - | 52.3 | - | - (stored whole) | - | - | - | - | accuracy=0.794050, logloss=5.350625 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1090 2026-10-10; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 64.3 | 64.3..64.3 | 1 | 0.813 | - | 766.1 | 64.3 | 701.76 (upload_ms_untimed) | 0.068 (whole/whole) | - | 3129.9 | 1370.0 | accuracy=0.794050, logloss=5.350631 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 1027.1 | 1027.1..1027.1 | 1 | 0.051 | - | 1027.1 | 1027.1 | 0.00 (cpu-arm) | 0.051 (whole/whole) | - | 3823.4 | - | accuracy=0.794050, logloss=4.278741 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'binarize': 0.0, 'fit_prior': True, 'force_alpha': True}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), BernoulliNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### bernoulli-nb / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1090)`, ran on NVIDIA L40S (nv2, RunPod) job v1090

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 13.5 | 13.5..13.5 | 1 | - | - | 13.5 | - | - (stored whole) | - | - | - | - | accuracy=0.755560, logloss=0.557803 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1090 2026-10-10; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 12.9 | 12.9..12.9 | 1 | 1.047 | - | 70.7 | 12.9 | 57.84 (upload_ms_untimed) | 0.191 (whole/whole) | - | 1172.6 | 494.0 | accuracy=0.755560, logloss=0.557803 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 121.5 | 121.5..121.5 | 1 | 0.111 | - | 121.5 | 121.5 | 0.00 (cpu-arm) | 0.111 (whole/whole) | - | 431.8 | - | accuracy=0.755560, logloss=0.557802 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'binarize': 0.0, 'fit_prior': True, 'force_alpha': True}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), BernoulliNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### binarizer / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/nvidia-wave2/campaign/measured/algos.binarizer.istella.rows=full/board.json (nv2/paired-603f66102ae1d3de379e5b5704cb17e2bda911358ead73c1d74b6fca794273ef)`, ran on d2fe567d3b88

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 0.2 | 0.2..0.2 | 1 | - | - | 0.2 | - | - (stored whole) | - | - | 1622.5 | 1706.0 | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x220, rel_diff_vs_sklearn=0.000000 | - | main board, one scored run | wheel | ok (main@aa8e9e72b nv2/paired-603f66102ae1d3de379e5b5704cb17e2bda911358ead73c1d74b6fca794273ef 2026-10-14; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host d2fe567d3b88) | cuml 26.08.00 | opponent | 2.4 | 2.4..2.4 | 1 | 0.086 | - | 676.6 | 2.4 | 674.17 (upload_ms_untimed) | 0.0003069 (whole/whole) | - | 3224.8 | 1440.0 | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x220, rel_diff_vs_sklearn=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'threshold': 0.0}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), Binarizer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params + constructor attributes threshold,copy) | mojolearn (get_params) |
| copy | true | true |
| seed | "none (deterministic)" | "none (deterministic)" |
| threshold | 0.0 | 0.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | hardware | version | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | NVIDIA L40S (nv2, RunPod) | 0.8.44 | Xq | - | 76.4 | 76.4..76.4 | 1 | - | - | 76.4 | - | - (stored whole) | - | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x220, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| cuml-gpu | NVIDIA L40S (host d2fe567d3b88) | cuml 26.08.00 | Xq | - | 8.2 | 8.2..8.2 | 1 | 9.337 | - | - | 8.2 | - (stored kernel) | 9.337 (MIXED ours whole / arm kernel) | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x220, rel_diff_vs_sklearn=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: transform(Xq)(Xq)

inference call, cuml-gpu: transform(Xq)(Xq)

### binarizer / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/nvidia-wave2/campaign/measured/algos.binarizer.taxi.rows=full/board.json (nv2/paired-5cc93a8b6016ca88157c94ac4bf3748b987a46a0728f836d3a89aac22b06841c)`, ran on d2fe567d3b88

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 0.2 | 0.2..0.2 | 1 | - | - | 0.2 | - | - (stored whole) | - | - | 269.6 | 682.0 | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x11, rel_diff_vs_sklearn=0.000000 | - | main board, one scored run | wheel | ok (main@aa8e9e72b nv2/paired-5cc93a8b6016ca88157c94ac4bf3748b987a46a0728f836d3a89aac22b06841c 2026-10-14; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host d2fe567d3b88) | cuml 26.08.00 | opponent | 0.9 | 0.9..0.9 | 1 | 0.199 | - | 48.8 | 0.9 | 47.96 (upload_ms_untimed) | 0.004 (whole/whole) | - | 899.2 | 486.0 | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x11, rel_diff_vs_sklearn=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'threshold': 0.0}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), Binarizer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params + constructor attributes threshold,copy) | mojolearn (get_params) |
| copy | true | true |
| seed | "none (deterministic)" | "none (deterministic)" |
| threshold | 0.0 | 0.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | hardware | version | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | NVIDIA L40S (nv2, RunPod) | 0.8.44 | Xq | - | 4.4 | 4.4..4.4 | 1 | - | - | 4.4 | - | - (stored whole) | - | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x11, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| cuml-gpu | NVIDIA L40S (host d2fe567d3b88) | cuml 26.08.00 | Xq | - | 2.1 | 2.1..2.1 | 1 | 2.063 | - | - | 2.1 | - (stored kernel) | 2.063 (MIXED ours whole / arm kernel) | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x11, rel_diff_vs_sklearn=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: transform(Xq)(Xq)

inference call, cuml-gpu: transform(Xq)(Xq)

### categorical-nb / istella (rows full, shape X 1000000x8; Xq 100000x8; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1090)`, ran on NVIDIA L40S (nv2, RunPod) job v1090

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 21.0 | 21.0..21.0 | 1 | - | - | 21.0 | - | - (stored whole) | - | - | - | - | accuracy=0.838850, logloss=0.412625 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1090 2026-10-10; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 76.1 | 76.1..76.1 | 1 | 0.276 | - | 104.2 | 76.1 | 28.07 (upload_ms_untimed) | 0.201 (whole/whole) | - | 1007.4 | 476.0 | accuracy=0.838850, logloss=0.412625 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 137.8 | 137.8..137.8 | 1 | 0.152 | - | 137.8 | 137.8 | 0.00 (cpu-arm) | 0.152 (whole/whole) | - | 352.7 | - | accuracy=0.838850, logloss=0.412625 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'fit_prior': True, 'force_alpha': True}. Rows: None. Timed: None.

mismatch: min_categories = every code seen in X or Xq on ours and scikit-learn; cuML has no min_categories option

config: cuML benchmark (RAPIDS), CategoricalNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### categorical-nb / taxi (rows full, shape X 1000000x5; Xq 100000x5; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1090)`, ran on NVIDIA L40S (nv2, RunPod) job v1090

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 17.1 | 17.1..17.1 | 1 | - | - | 17.1 | - | - (stored whole) | - | - | - | - | accuracy=0.765850, logloss=0.538866 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1090 2026-10-10; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 13.5 | 13.5..13.5 | 1 | 1.267 | - | 43.1 | 13.5 | 29.63 (upload_ms_untimed) | 0.396 (whole/whole) | - | 1109.7 | 460.0 | accuracy=0.765850, logloss=0.538866 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 117.0 | 117.0..117.0 | 1 | 0.146 | - | 117.0 | 117.0 | 0.00 (cpu-arm) | 0.146 (whole/whole) | - | 315.0 | - | accuracy=0.765850, logloss=0.538866 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'fit_prior': True, 'force_alpha': True}. Rows: None. Timed: None.

mismatch: min_categories = every code seen in X or Xq on ours and scikit-learn; cuML has no min_categories option

config: cuML benchmark (RAPIDS), CategoricalNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### cholesky / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/nvidia-wave10/campaign/measured/algos.cholesky.synthetic.rows=full/board.json (nv2/paired-f950308e25a2847f5c8f2e0413ffeda089275464076b2ac1df969a0e7d186e09)`, ran on d2fe567d3b88

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 196.4 | 196.4..196.4 | 1 | - | - | 196.4 | 185.2 | 11.24 (upload_ms_separate) | - | - | 3298.7 | 1194.0 | relative_residual=2.9e-07 | - | main board, one scored run | wheel | ok (main@dbc9b18c1 nv2/paired-f950308e25a2847f5c8f2e0413ffeda089275464076b2ac1df969a0e7d186e09 2026-10-14; identity vs the amd columns: n/a) |
| torch-gpu | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 12.0 | 12.0..12.0 | 1 | 16.328 | - | - | 12.0 | - (stored kernel) | 15.394 (kernel/kernel) | - | 1975.3 | 768.3 | relative_residual=1.509e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| cupy-gpu | cupy | gpu | NVIDIA L40S (host d2fe567d3b88) | cupy 14.2.0 | opponent | 14.8 | 14.8..14.8 | 1 | 13.278 | - | - | 14.8 | - (stored kernel) | 12.518 (kernel/kernel (whole not derivable)) | - | 2055.5 | 1234.0 | relative_residual=1.365e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cupy-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'jitter': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cupy-gpu | ours | torch-gpu |
|---|---||---|---||---|---|
| library (source) | cupy (declared) | mojolearn (declared) | torch (declared) |
| input_arrays | [{"dtype": "float32", "sha256": "e1cc202332be3803baafedd32dff310e4af25316f9711e8522028b15962d0054", "shape": [8192, 8192]}] | [{"dtype": "float32", "sha256": "e1cc202332be3803baafedd32dff310e4af25316f9711e8522028b15962d0054", "shape": [8192, 8192]}] | [{"dtype": "float32", "sha256": "e1cc202332be3803baafedd32dff310e4af25316f9711e8522028b15962d0054", "shape": [8192, 8192]}] |
| jitter | 0.0 | 0.0 | 0.0 |
| seed | 7 | 7 | 7 |

### cnn-clf / synthetic (rows full, shape X 20000x1x28x28; Xq 5000x1x28x28; y 20000; yq 5000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0065)`, ran on NVIDIA L40S (nv2, RunPod) job v0065

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 483.3 ms) = 0.415; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 458.6 ms) = 0.437. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@2afa942e3) | identical | 200.4 | 200.4..200.4 | 1 | - | - | 200.4 | - | - (stored whole) | - | - | - | - | accuracy=1.000000 | - | main board, one scored run | - | ok (main@2afa942e3 nv2/v0065 2026-10-11; identity vs the amd columns: n/a) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 458.6 | 458.6..458.6 | 1 | 0.437 | - | - | 458.6 | - (stored kernel) | 0.437 (MIXED ours whole / arm kernel) | - | 1317.2 | 287.8 | accuracy=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 569.3 | 569.3..569.3 | 1 | 0.352 | - | - | 569.3 | - (stored kernel) | 0.352 (MIXED ours whole / arm kernel) | - | 1531.3 | 214.3 | accuracy=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 475.9 | 475.9..475.9 | 1 | 0.421 | - | - | 475.9 | - (stored kernel) | 0.421 (MIXED ours whole / arm kernel) | - | 1318.3 | 287.8 | accuracy=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 464.8 | 464.8..464.8 | 1 | 0.431 | - | - | 464.8 | - (stored kernel) | 0.431 (MIXED ours whole / arm kernel) | - | 1358.0 | 214.3 | accuracy=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 483.3 | 483.3..483.3 | 1 | 0.415 | - | - | 483.3 | - (stored kernel) | 0.415 (MIXED ours whole / arm kernel) | - | 1552.4 | 204.7 | accuracy=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 683.3 | 683.3..683.3 | 1 | 0.293 | - | - | 683.3 | - (stored kernel) | 0.293 (MIXED ours whole / arm kernel) | - | 1618.5 | 154.0 | accuracy=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 128, 'conv_channels': [8, 16], 'dampening': 0.0, 'input_shape': [1, 28, 28], 'kernel_size': 3, 'learning_rate': 0.01, 'max_iter': 2, 'momentum': 0.9, 'nesterov': False, 'optimizer': 'sgd', 'pool_size': 2, 'random_state': 7, 'shuffle': True, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### complement-nb / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0124)`, ran on NVIDIA L40S (nv2, RunPod) job v0124

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 116.0 | 116.0..116.0 | 1 | - | - | 116.0 | - | - (stored whole) | - | - | - | - | accuracy=0.849360, logloss=3.762524 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0124 2026-10-11; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 56.5 | 56.5..56.5 | 1 | 2.055 | - | 785.5 | 56.5 | 729.03 (upload_ms_untimed) | 0.148 (whole/whole) | - | 3934.4 | 1372.0 | accuracy=0.849350, logloss=3.763060 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 278.9 | 278.9..278.9 | 1 | 0.416 | - | 278.9 | 278.9 | 0.00 (cpu-arm) | 0.416 (whole/whole) | - | 3912.0 | - | accuracy=0.849350, logloss=3.174763 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'fit_prior': True, 'force_alpha': True, 'norm': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), ComplementNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### complement-nb / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0124)`, ran on NVIDIA L40S (nv2, RunPod) job v0124

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 29.3 | 29.3..29.3 | 1 | - | - | 29.3 | - | - (stored whole) | - | - | - | - | accuracy=0.678020, logloss=0.715492 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0124 2026-10-11; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 12.3 | 12.3..12.3 | 1 | 2.393 | - | 66.8 | 12.3 | 54.56 (upload_ms_untimed) | 0.439 (whole/whole) | - | 1100.8 | 496.0 | accuracy=0.678060, logloss=0.715531 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 70.7 | 70.7..70.7 | 1 | 0.415 | - | 70.7 | 70.7 | 0.00 (cpu-arm) | 0.415 (whole/whole) | - | 434.5 | - | accuracy=0.678030, logloss=0.715493 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'fit_prior': True, 'force_alpha': True, 'norm': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), ComplementNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### complement-nb / text (rows full, shape X 86626x4096; Xq 9626x4096; y 86626; yq 9626)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0532)`, ran on NVIDIA L40S (nv, RunPod) job n0532

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@42d1e42c6) | identical | 161.9 | 161.9..161.9 | 1 | - | - | 161.9 | - | - (stored whole) | - | - | - | - | accuracy=0.983067, logloss=0.559491 | - | main board, one scored run | - | ok (main@42d1e42c6 nv/n0532 2026-10-08; identity vs amd-mi325x: MATCH) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 61.1 | 61.1..61.1 | 1 | 2.651 | - | 1467.5 | 61.1 | 1406.40 (upload_ms_untimed) | 0.110 (whole/whole) | - | 4710.5 | 1946.0 | accuracy=0.983067, logloss=0.559490 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 410.6 | 410.6..410.6 | 1 | 0.394 | - | 410.6 | 410.6 | 0.00 (cpu-arm) | 0.394 (whole/whole) | - | 4414.9 | - | accuracy=0.983067, logloss=0.557285 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'fit_prior': True, 'force_alpha': True, 'norm': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), ComplementNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### connected-components / istella (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0523)`, ran on NVIDIA L40S (nv, RunPod) job n0523

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@4df610f3b) | identical | 0.8 | 0.8..0.8 | 1 | - | - | 0.8 | - | - (stored whole) | - | - | - | - | n_components=81 | - | main board, one scored run | - | ok (main@4df610f3b nv/n0523 2026-10-08; identity vs amd-mi325x: MATCH) |

settings: {}. Rows: None. Timed: None.

mismatch: ours takes the CSR graph as an (indptr, indices, n) tuple and uploads it inside the clock (the 2026-10-08 CSR entries; before, a dense n x n adjacency built before the clock); networkx and cuGraph take the graph itself

mismatch: opponents withheld: the lane settings at this tree's HEAD differ from the settings opponents-default-20261006 recorded for its opponent race (an opponent job must score them again)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### connected-components / taxi (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0523)`, ran on NVIDIA L40S (nv, RunPod) job n0523

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@4df610f3b) | identical | 1.6 | 1.6..1.6 | 1 | - | - | 1.6 | - | - (stored whole) | - | - | - | - | n_components=588 | - | main board, one scored run | - | ok (main@4df610f3b nv/n0523 2026-10-08; identity vs amd-mi325x: MATCH) |

settings: {}. Rows: None. Timed: None.

mismatch: ours takes the CSR graph as an (indptr, indices, n) tuple and uploads it inside the clock (the 2026-10-08 CSR entries; before, a dense n x n adjacency built before the clock); networkx and cuGraph take the graph itself

mismatch: opponents withheld: the lane settings at this tree's HEAD differ from the settings opponents-specific-20261006 recorded for its opponent race (an opponent job must score them again)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### conv1d / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/nvidia-wave5/campaign/measured/algos.conv1d.synthetic.rows=full/board.json (nv2/paired-bae6c90886c366a7aa0d150fa240f2938db6bb763ac4d2880196b521eeb9f5a2)`, ran on d2fe567d3b88

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 3.2 ms) = 3.119; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 3.9 ms) = 2.511. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 9.9 | 9.9..9.9 | 1 | - | - | - | 9.9 | - (stored kernel) | - | - | 2145.5 | 1962.0 | max_rel_diff_vs_torch_eager_fp32=0.268221, rel_fro_vs_torch_eager_fp32=3.613e-07 | - | main board, one scored run | wheel | ok (main@d11b82091 nv2/paired-bae6c90886c366a7aa0d150fa240f2938db6bb763ac4d2880196b521eeb9f5a2 2026-10-14; identity vs the amd columns: n/a) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 4.6 | 4.6..4.6 | 1 | 2.142 | - | - | 4.6 | - (stored kernel) | 2.142 (kernel/kernel) | - | 1373.0 | 640.4 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 3.9 | 3.9..3.9 | 1 | 2.511 | - | - | 3.9 | - (stored kernel) | 2.511 (kernel/kernel) | - | 1561.7 | 640.4 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 3.6 | 3.6..3.6 | 1 | 2.728 | - | - | 3.6 | - (stored kernel) | 2.728 (kernel/kernel) | - | 1387.0 | 900.4 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 3.8 | 3.8..3.8 | 1 | 2.573 | - | - | 3.8 | - (stored kernel) | 2.573 (kernel/kernel) | - | 1528.7 | 900.4 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 3.2 | 3.2..3.2 | 1 | 3.119 | - | - | 3.2 | - (stored kernel) | 3.119 (kernel/kernel) | - | 1359.4 | 706.5 | max_rel_diff_vs_torch_eager_fp32=3417.849541, rel_fro_vs_torch_eager_fp32=0.003296 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 3.8 | 3.8..3.8 | 1 | 2.613 | - | - | 3.8 | - (stored kernel) | 2.613 (kernel/kernel) | - | 1560.1 | 706.5 | max_rel_diff_vs_torch_eager_fp32=3524.661064, rel_fro_vs_torch_eager_fp32=0.003294 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'bias': True, 'dilation': 1, 'groups': 1, 'in_channels': 128, 'kernel_size': 3, 'out_channels': 128, 'padding': 1, 'padding_mode': 'zeros', 'stride': 1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-compile-tf32 | torch-eager-bf16 | torch-eager-fp32 | torch-eager-tf32 |
|---|---||---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| bias | true | true | true | true | true | true | true |
| dilation | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| groups | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| in_channels | 128 | 128 | 128 | 128 | 128 | 128 | 128 |
| input_arrays | [{"dtype": "float32", "name": "x", "sha256": "cec4f976c7feae09a100521e6e404c90584b25330e0bef803c3ae82efaa217e7", "shape": [64, 128, 4096]}, {"dtype": "float32", "name": "bias", "sha256": "a8e12a7ee1bfe5640579402eca2de99b8e6c38a3d243507e7ad1edadebc7eb84", "shape": [128]}, {"dtype": "float32", "name": "weight", "sha256": "cd1becee90e2a8fbfe515641fefd6bfbeb9dba42273fb21af1273164cb40fe15", "shape": [128, 128, 3]}] | [{"dtype": "float32", "name": "x", "sha256": "cec4f976c7feae09a100521e6e404c90584b25330e0bef803c3ae82efaa217e7", "shape": [64, 128, 4096]}, {"dtype": "float32", "name": "bias", "sha256": "a8e12a7ee1bfe5640579402eca2de99b8e6c38a3d243507e7ad1edadebc7eb84", "shape": [128]}, {"dtype": "float32", "name": "weight", "sha256": "cd1becee90e2a8fbfe515641fefd6bfbeb9dba42273fb21af1273164cb40fe15", "shape": [128, 128, 3]}] | [{"dtype": "float32", "name": "x", "sha256": "cec4f976c7feae09a100521e6e404c90584b25330e0bef803c3ae82efaa217e7", "shape": [64, 128, 4096]}, {"dtype": "float32", "name": "bias", "sha256": "a8e12a7ee1bfe5640579402eca2de99b8e6c38a3d243507e7ad1edadebc7eb84", "shape": [128]}, {"dtype": "float32", "name": "weight", "sha256": "cd1becee90e2a8fbfe515641fefd6bfbeb9dba42273fb21af1273164cb40fe15", "shape": [128, 128, 3]}] | [{"dtype": "float32", "name": "x", "sha256": "cec4f976c7feae09a100521e6e404c90584b25330e0bef803c3ae82efaa217e7", "shape": [64, 128, 4096]}, {"dtype": "float32", "name": "bias", "sha256": "a8e12a7ee1bfe5640579402eca2de99b8e6c38a3d243507e7ad1edadebc7eb84", "shape": [128]}, {"dtype": "float32", "name": "weight", "sha256": "cd1becee90e2a8fbfe515641fefd6bfbeb9dba42273fb21af1273164cb40fe15", "shape": [128, 128, 3]}] | [{"dtype": "float32", "name": "x", "sha256": "cec4f976c7feae09a100521e6e404c90584b25330e0bef803c3ae82efaa217e7", "shape": [64, 128, 4096]}, {"dtype": "float32", "name": "bias", "sha256": "a8e12a7ee1bfe5640579402eca2de99b8e6c38a3d243507e7ad1edadebc7eb84", "shape": [128]}, {"dtype": "float32", "name": "weight", "sha256": "cd1becee90e2a8fbfe515641fefd6bfbeb9dba42273fb21af1273164cb40fe15", "shape": [128, 128, 3]}] | [{"dtype": "float32", "name": "x", "sha256": "cec4f976c7feae09a100521e6e404c90584b25330e0bef803c3ae82efaa217e7", "shape": [64, 128, 4096]}, {"dtype": "float32", "name": "bias", "sha256": "a8e12a7ee1bfe5640579402eca2de99b8e6c38a3d243507e7ad1edadebc7eb84", "shape": [128]}, {"dtype": "float32", "name": "weight", "sha256": "cd1becee90e2a8fbfe515641fefd6bfbeb9dba42273fb21af1273164cb40fe15", "shape": [128, 128, 3]}] | [{"dtype": "float32", "name": "x", "sha256": "cec4f976c7feae09a100521e6e404c90584b25330e0bef803c3ae82efaa217e7", "shape": [64, 128, 4096]}, {"dtype": "float32", "name": "bias", "sha256": "a8e12a7ee1bfe5640579402eca2de99b8e6c38a3d243507e7ad1edadebc7eb84", "shape": [128]}, {"dtype": "float32", "name": "weight", "sha256": "cd1becee90e2a8fbfe515641fefd6bfbeb9dba42273fb21af1273164cb40fe15", "shape": [128, 128, 3]}] |
| kernel_size | 3 | 3 | 3 | 3 | 3 | 3 | 3 |
| out_channels | 128 | 128 | 128 | 128 | 128 | 128 | 128 |
| padding | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| padding_mode | "zeros" | "zeros" | "zeros" | "zeros" | "zeros" | "zeros" | "zeros" |
| seed | 7 | 7 | 7 | 7 | 7 | 7 | 7 |
| stride | 1 | 1 | 1 | 1 | 1 | 1 | 1 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | hardware | version | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | NVIDIA L40S (nv2, RunPod) | 0.8.44 | Xq | - | 2.7 | 2.7..2.7 | 1 | - | - | - | 2.7 | - (stored kernel) | - | - | max_rel_diff_vs_torch_eager_fp32=0.268221, rel_fro_vs_torch_eager_fp32=3.613e-07 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 1.2 | 1.2..1.2 | 1 | 2.372 | - | - | 1.2 | - (stored kernel) | 2.372 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 1.2 | 1.2..1.2 | 1 | 2.253 | - | - | 1.2 | - (stored kernel) | 2.253 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-tf32 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 1.1 | 1.1..1.1 | 1 | 2.396 | - | - | 1.1 | - (stored kernel) | 2.396 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-tf32 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 1.3 | 1.3..1.3 | 1 | 2.189 | - | - | 1.3 | - (stored kernel) | 2.189 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 1.1 | 1.1..1.1 | 1 | 2.467 | - | - | 1.1 | - (stored kernel) | 2.467 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=3417.849541, rel_fro_vs_torch_eager_fp32=0.003296 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 1.1 | 1.1..1.1 | 1 | 2.410 | - | - | 1.1 | - (stored kernel) | 2.410 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=3524.661064, rel_fro_vs_torch_eager_fp32=0.003294 | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-tf32: forward(x) (no autograd)(Xq)

inference call, torch-compile-tf32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### conv2d / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/nvidia-wave5/campaign/measured/algos.conv2d.synthetic.rows=full/board.json (nv2/paired-62c8f15cc9a300656258cd5d0e0b6e0daaa445440e5a1eea5799fa39d56d4307)`, ran on d2fe567d3b88

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 1.8 ms) = 3.465; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 2.2 ms) = 2.916. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 6.3 | 6.3..6.3 | 1 | - | - | - | 6.3 | - (stored kernel) | - | - | 1751.2 | 1450.0 | max_rel_diff_vs_torch_eager_fp32=0.248197, rel_fro_vs_torch_eager_fp32=4.453e-07 | - | main board, one scored run | wheel | ok (main@d11b82091 nv2/paired-62c8f15cc9a300656258cd5d0e0b6e0daaa445440e5a1eea5799fa39d56d4307 2026-10-14; identity vs the amd columns: n/a) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 2.2 | 2.2..2.2 | 1 | 2.916 | - | - | 2.2 | - (stored kernel) | 2.916 (kernel/kernel) | - | 1062.0 | 250.5 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 2.4 | 2.4..2.4 | 1 | 2.576 | - | - | 2.4 | - (stored kernel) | 2.576 (kernel/kernel) | - | 1277.9 | 346.8 | max_rel_diff_vs_torch_eager_fp32=0.506768, rel_fro_vs_torch_eager_fp32=5.007e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 1.8 | 1.8..1.8 | 1 | 3.562 | - | - | 1.8 | - (stored kernel) | 3.562 (kernel/kernel) | - | 1077.8 | 354.2 | max_rel_diff_vs_torch_eager_fp32=382.931903, rel_fro_vs_torch_eager_fp32=0.0003021 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 2.7 | 2.7..2.7 | 1 | 2.300 | - | - | 2.7 | - (stored kernel) | 2.300 (kernel/kernel) | - | 1228.9 | 354.2 | max_rel_diff_vs_torch_eager_fp32=382.931903, rel_fro_vs_torch_eager_fp32=0.0003021 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 1.8 | 1.8..1.8 | 1 | 3.465 | - | - | 1.8 | - (stored kernel) | 3.465 (kernel/kernel) | - | 1120.7 | 273.7 | max_rel_diff_vs_torch_eager_fp32=3418.337554, rel_fro_vs_torch_eager_fp32=0.003382 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 2.1 | 2.1..2.1 | 1 | 2.984 | - | - | 2.1 | - (stored kernel) | 2.984 (kernel/kernel) | - | 1329.9 | 272.7 | max_rel_diff_vs_torch_eager_fp32=3433.596343, rel_fro_vs_torch_eager_fp32=0.003380 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'bias': True, 'dilation': 1, 'groups': 1, 'in_channels': 64, 'kernel_size': 3, 'out_channels': 64, 'padding': 1, 'padding_mode': 'zeros', 'stride': 1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-compile-tf32 | torch-eager-bf16 | torch-eager-fp32 | torch-eager-tf32 |
|---|---||---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| bias | true | true | true | true | true | true | true |
| dilation | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| groups | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| in_channels | 64 | 64 | 64 | 64 | 64 | 64 | 64 |
| input_arrays | [{"dtype": "float32", "name": "x", "sha256": "cd7e0a7d8c0c594cca519b9cf49c8f086513282644cb8b6c06771dcb98504f2d", "shape": [64, 64, 56, 56]}, {"dtype": "float32", "name": "bias", "sha256": "a055c18d708f2d056981b307f2bfe3634d34ae7e4cd5b6a55330b3a6a96bd6ed", "shape": [64]}, {"dtype": "float32", "name": "weight", "sha256": "a5f32fbce52da86968c08ac2499c8f321445b4391778c6bcb53417a9eb9669bc", "shape": [64, 64, 3, 3]}] | [{"dtype": "float32", "name": "x", "sha256": "cd7e0a7d8c0c594cca519b9cf49c8f086513282644cb8b6c06771dcb98504f2d", "shape": [64, 64, 56, 56]}, {"dtype": "float32", "name": "bias", "sha256": "a055c18d708f2d056981b307f2bfe3634d34ae7e4cd5b6a55330b3a6a96bd6ed", "shape": [64]}, {"dtype": "float32", "name": "weight", "sha256": "a5f32fbce52da86968c08ac2499c8f321445b4391778c6bcb53417a9eb9669bc", "shape": [64, 64, 3, 3]}] | [{"dtype": "float32", "name": "x", "sha256": "cd7e0a7d8c0c594cca519b9cf49c8f086513282644cb8b6c06771dcb98504f2d", "shape": [64, 64, 56, 56]}, {"dtype": "float32", "name": "bias", "sha256": "a055c18d708f2d056981b307f2bfe3634d34ae7e4cd5b6a55330b3a6a96bd6ed", "shape": [64]}, {"dtype": "float32", "name": "weight", "sha256": "a5f32fbce52da86968c08ac2499c8f321445b4391778c6bcb53417a9eb9669bc", "shape": [64, 64, 3, 3]}] | [{"dtype": "float32", "name": "x", "sha256": "cd7e0a7d8c0c594cca519b9cf49c8f086513282644cb8b6c06771dcb98504f2d", "shape": [64, 64, 56, 56]}, {"dtype": "float32", "name": "bias", "sha256": "a055c18d708f2d056981b307f2bfe3634d34ae7e4cd5b6a55330b3a6a96bd6ed", "shape": [64]}, {"dtype": "float32", "name": "weight", "sha256": "a5f32fbce52da86968c08ac2499c8f321445b4391778c6bcb53417a9eb9669bc", "shape": [64, 64, 3, 3]}] | [{"dtype": "float32", "name": "x", "sha256": "cd7e0a7d8c0c594cca519b9cf49c8f086513282644cb8b6c06771dcb98504f2d", "shape": [64, 64, 56, 56]}, {"dtype": "float32", "name": "bias", "sha256": "a055c18d708f2d056981b307f2bfe3634d34ae7e4cd5b6a55330b3a6a96bd6ed", "shape": [64]}, {"dtype": "float32", "name": "weight", "sha256": "a5f32fbce52da86968c08ac2499c8f321445b4391778c6bcb53417a9eb9669bc", "shape": [64, 64, 3, 3]}] | [{"dtype": "float32", "name": "x", "sha256": "cd7e0a7d8c0c594cca519b9cf49c8f086513282644cb8b6c06771dcb98504f2d", "shape": [64, 64, 56, 56]}, {"dtype": "float32", "name": "bias", "sha256": "a055c18d708f2d056981b307f2bfe3634d34ae7e4cd5b6a55330b3a6a96bd6ed", "shape": [64]}, {"dtype": "float32", "name": "weight", "sha256": "a5f32fbce52da86968c08ac2499c8f321445b4391778c6bcb53417a9eb9669bc", "shape": [64, 64, 3, 3]}] | [{"dtype": "float32", "name": "x", "sha256": "cd7e0a7d8c0c594cca519b9cf49c8f086513282644cb8b6c06771dcb98504f2d", "shape": [64, 64, 56, 56]}, {"dtype": "float32", "name": "bias", "sha256": "a055c18d708f2d056981b307f2bfe3634d34ae7e4cd5b6a55330b3a6a96bd6ed", "shape": [64]}, {"dtype": "float32", "name": "weight", "sha256": "a5f32fbce52da86968c08ac2499c8f321445b4391778c6bcb53417a9eb9669bc", "shape": [64, 64, 3, 3]}] |
| kernel_size | 3 | 3 | 3 | 3 | 3 | 3 | 3 |
| out_channels | 64 | 64 | 64 | 64 | 64 | 64 | 64 |
| padding | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| padding_mode | "zeros" | "zeros" | "zeros" | "zeros" | "zeros" | "zeros" | "zeros" |
| seed | 7 | 7 | 7 | 7 | 7 | 7 | 7 |
| stride | 1 | 1 | 1 | 1 | 1 | 1 | 1 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | hardware | version | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | NVIDIA L40S (nv2, RunPod) | 0.8.44 | Xq | - | 1.6 | 1.6..1.6 | 1 | - | - | - | 1.6 | - (stored kernel) | - | - | max_rel_diff_vs_torch_eager_fp32=0.248197, rel_fro_vs_torch_eager_fp32=4.453e-07 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.5 | 0.5..0.5 | 1 | 3.074 | - | - | 0.5 | - (stored kernel) | 3.074 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 1.3 | 1.3..1.3 | 1 | 1.292 | - | - | 1.3 | - (stored kernel) | 1.292 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.506768, rel_fro_vs_torch_eager_fp32=5.007e-07 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-tf32 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.4 | 0.4..0.4 | 1 | 4.522 | - | - | 0.4 | - (stored kernel) | 4.522 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=382.931903, rel_fro_vs_torch_eager_fp32=0.0003021 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-tf32 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.6 | 0.6..0.6 | 1 | 2.726 | - | - | 0.6 | - (stored kernel) | 2.726 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=382.931903, rel_fro_vs_torch_eager_fp32=0.0003021 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.4 | 0.4..0.4 | 1 | 4.538 | - | - | 0.4 | - (stored kernel) | 4.538 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=3418.337554, rel_fro_vs_torch_eager_fp32=0.003382 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.3 | 0.3..0.3 | 1 | 4.703 | - | - | 0.3 | - (stored kernel) | 4.703 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=3433.596343, rel_fro_vs_torch_eager_fp32=0.003380 | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-tf32: forward(x) (no autograd)(Xq)

inference call, torch-compile-tf32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### damped-ets / synthetic (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0097)`, ran on NVIDIA L40S (nv2, RunPod) job v0097

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 675.3 | 675.3..675.3 | 1 | - | - | 675.3 | - | - (stored whole) | - | - | - | - | forecast_rmse=13.931153 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0097 2026-10-11; identity vs the amd columns: n/a) |
| statsmodels-cpu | statsmodels | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | statsmodels 0.15.0 | opponent | 173.6 | 173.6..173.6 | 1 | 3.889 | - | 173.6 | 173.6 | 0.00 (cpu-arm) | 3.889 (whole/whole) | - | 58.6 | - | forecast_rmse=26.588738 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| statsforecast-cpu | statsforecast | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | statsforecast 2.1.1 | opponent | 164.3 | 164.3..164.3 | 1 | 4.109 | - | 164.3 | 164.3 | 0.00 (cpu-arm) | 4.109 (whole/whole) | - | 468.0 | - | forecast_rmse=13.945986 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, statsmodels-cpu, statsforecast-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'damped': True, 'model': 'AAN', 'season_length': 1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### damped-ets / taxi-hourly (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0097)`, ran on NVIDIA L40S (nv2, RunPod) job v0097

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 743.2 | 743.2..743.2 | 1 | - | - | 743.2 | - | - (stored whole) | - | - | - | - | forecast_rmse=96.690449 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0097 2026-10-11; identity vs the amd columns: n/a) |
| statsmodels-cpu | statsmodels | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | statsmodels 0.15.0 | opponent | 173.0 | 173.0..173.0 | 1 | 4.297 | - | 173.0 | 173.0 | 0.00 (cpu-arm) | 4.297 (whole/whole) | - | 58.2 | - | forecast_rmse=196.955273 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| statsforecast-cpu | statsforecast | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | statsforecast 2.1.1 | opponent | 133.9 | 133.9..133.9 | 1 | 5.551 | - | 133.9 | 133.9 | 0.00 (cpu-arm) | 5.551 (whole/whole) | - | 468.2 | - | forecast_rmse=96.685568 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, statsmodels-cpu, statsforecast-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'damped': True, 'model': 'AAN', 'season_length': 1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### dropout2d / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/nvidia-wave6/campaign/measured/algos.dropout2d.synthetic.rows=full/board.json (nv2/paired-455bb276f248951866a3598362da9b6e3c9eff23bb2fa0beda5ac3309e53a939)`, ran on d2fe567d3b88

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 1.2 ms) = 1.064. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 1.3 | 1.3..1.3 | 1 | - | - | - | 1.3 | - (stored kernel) | - | - | 1751.5 | 938.0 | dropout_contract_relative_error=5.422e-08, dropout_probability_z=0.708333, dropped_channel_fraction=0.096680, expected_dropout_probability=0.100000, identifiable_channels=4096 | - | main board, one scored run | wheel | ok (main@0f3f674f2 nv2/paired-455bb276f248951866a3598362da9b6e3c9eff23bb2fa0beda5ac3309e53a939 2026-10-14; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 1.2 | 1.2..1.2 | 1 | 1.064 | - | - | 1.2 | - (stored kernel) | 1.064 (kernel/kernel) | - | 898.8 | 250.0 | dropout_contract_relative_error=5.422e-08, dropout_probability_z=1.489583, dropped_channel_fraction=0.093018, expected_dropout_probability=0.100000, identifiable_channels=4096 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 4.9 | 4.9..4.9 | 1 | 0.267 | - | - | 4.9 | - (stored kernel) | 0.267 (kernel/kernel) | - | 1036.2 | 250.0 | dropout_contract_relative_error=5.421e-08, dropout_probability_z=1.791667, dropped_channel_fraction=0.108398, expected_dropout_probability=0.100000, identifiable_channels=4096 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 1.6 | 1.6..1.6 | 1 | 0.835 | - | - | 1.6 | - (stored kernel) | 0.835 (kernel/kernel) | - | 900.0 | 250.0 | dropout_contract_relative_error=5.422e-08, dropout_probability_z=1.489583, dropped_channel_fraction=0.093018, expected_dropout_probability=0.100000, identifiable_channels=4096 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 2.1 | 2.1..2.1 | 1 | 0.606 | - | - | 2.1 | - (stored kernel) | 0.606 (kernel/kernel) | - | 1035.1 | 250.0 | dropout_contract_relative_error=5.421e-08, dropout_probability_z=1.791667, dropped_channel_fraction=0.108398, expected_dropout_probability=0.100000, identifiable_channels=4096 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'p': 0.1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-fp32 | torch-compile-tf32 | torch-eager-fp32 | torch-eager-tf32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| input_arrays | [{"dtype": "float32", "name": "x", "sha256": "cd7e0a7d8c0c594cca519b9cf49c8f086513282644cb8b6c06771dcb98504f2d", "shape": [64, 64, 56, 56]}] | [{"dtype": "float32", "name": "x", "sha256": "cd7e0a7d8c0c594cca519b9cf49c8f086513282644cb8b6c06771dcb98504f2d", "shape": [64, 64, 56, 56]}] | [{"dtype": "float32", "name": "x", "sha256": "cd7e0a7d8c0c594cca519b9cf49c8f086513282644cb8b6c06771dcb98504f2d", "shape": [64, 64, 56, 56]}] | [{"dtype": "float32", "name": "x", "sha256": "cd7e0a7d8c0c594cca519b9cf49c8f086513282644cb8b6c06771dcb98504f2d", "shape": [64, 64, 56, 56]}] | [{"dtype": "float32", "name": "x", "sha256": "cd7e0a7d8c0c594cca519b9cf49c8f086513282644cb8b6c06771dcb98504f2d", "shape": [64, 64, 56, 56]}] |
| p | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| seed | 7 | 7 | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | hardware | version | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | NVIDIA L40S (nv2, RunPod) | 0.8.44 | Xq | - | 0.3 | 0.3..0.3 | 1 | - | - | - | 0.3 | - (stored kernel) | - | - | dropout_contract_relative_error=5.422e-08, dropout_probability_z=0.708333, dropped_channel_fraction=0.096680, expected_dropout_probability=0.100000, identifiable_channels=4096 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.2 | 0.2..0.2 | 1 | 1.154 | - | - | 0.2 | - (stored kernel) | 1.154 (kernel/kernel) | - | dropout_contract_relative_error=5.422e-08, dropout_probability_z=1.489583, dropped_channel_fraction=0.093018, expected_dropout_probability=0.100000, identifiable_channels=4096 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.3 | 0.3..0.3 | 1 | 0.846 | - | - | 0.3 | - (stored kernel) | 0.846 (kernel/kernel) | - | dropout_contract_relative_error=5.421e-08, dropout_probability_z=1.791667, dropped_channel_fraction=0.108398, expected_dropout_probability=0.100000, identifiable_channels=4096 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-tf32 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.2 | 0.2..0.2 | 1 | 1.175 | - | - | 0.2 | - (stored kernel) | 1.175 (kernel/kernel) | - | dropout_contract_relative_error=5.422e-08, dropout_probability_z=1.489583, dropped_channel_fraction=0.093018, expected_dropout_probability=0.100000, identifiable_channels=4096 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-tf32 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.3 | 0.3..0.3 | 1 | 0.803 | - | - | 0.3 | - (stored kernel) | 0.803 (kernel/kernel) | - | dropout_contract_relative_error=5.421e-08, dropout_probability_z=1.791667, dropped_channel_fraction=0.108398, expected_dropout_probability=0.100000, identifiable_channels=4096 | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-tf32: forward(x) (no autograd)(Xq)

inference call, torch-compile-tf32: forward(x) (no autograd)(Xq)

### dynamic-optimized-theta / synthetic (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0097)`, ran on NVIDIA L40S (nv2, RunPod) job v0097

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 337.4 | 337.4..337.4 | 1 | - | - | 337.4 | - | - (stored whole) | - | - | - | - | forecast_rmse=1.436278 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0097 2026-10-11; identity vs the amd columns: n/a) |
| statsforecast-cpu | statsforecast | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | statsforecast 2.1.1 | opponent | 1030.6 | 1030.6..1030.6 | 1 | 0.327 | - | 1030.6 | 1030.6 | 0.00 (cpu-arm) | 0.327 (whole/whole) | - | 467.8 | - | forecast_rmse=1.436045 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, statsforecast-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'decomposition_type': 'multiplicative', 'season_length': 24}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### dynamic-optimized-theta / taxi-hourly (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0097)`, ran on NVIDIA L40S (nv2, RunPod) job v0097

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 1789.9 | 1789.9..1789.9 | 1 | - | - | 1789.9 | - | - (stored whole) | - | - | - | - | forecast_rmse=49.086628 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0097 2026-10-11; identity vs the amd columns: n/a) |
| statsforecast-cpu | statsforecast | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | statsforecast 2.1.1 | opponent | 1829.8 | 1829.8..1829.8 | 1 | 0.978 | - | 1829.8 | 1829.8 | 0.00 (cpu-arm) | 0.978 (whole/whole) | - | 467.5 | - | forecast_rmse=49.314797 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, statsforecast-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'decomposition_type': 'multiplicative', 'season_length': 24}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### eigh / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0051)`, ran on NVIDIA L40S (nv2, RunPod) job v0051

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@2afa942e3) | identical | 403.3 | 403.3..403.3 | 1 | - | - | 403.3 | - | - (stored whole) | - | - | - | - | max_eigenvalue_error=3.692e-07, relative_residual=6.672e-07 | - | main board, one scored run | - | ok (main@2afa942e3 nv2/v0051 2026-10-11; identity vs the amd columns: n/a) |
| torch-gpu | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 81.2 | 81.2..81.2 | 1 | 4.967 | - | - | 81.2 | - (stored kernel) | 4.967 (MIXED ours whole / arm kernel) | - | 937.9 | 450.1 | max_eigenvalue_error=1.044e-06, relative_residual=1.016e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| cupy-gpu | cupy | gpu | NVIDIA L40S (host cc560ebdaf91) | cupy 14.2.0 | opponent | 80.4 | 80.4..80.4 | 1 | 5.013 | - | - | 80.4 | - (stored kernel) | 5.013 (MIXED ours whole / arm kernel) | - | 753.9 | 1114.0 | max_eigenvalue_error=1.044e-06, relative_residual=1.016e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| numpy-cpu | numpy | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | numpy 2.4.6 | opponent | 14605.9 | 14605.9..14605.9 | 1 | 0.028 | - | 14605.9 | 14605.9 | 0.00 (cpu-arm) | 0.028 (whole/whole) | - | 945.8 | - | max_eigenvalue_error=3.49e-08, relative_residual=2.824e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, cupy-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'UPLO': 'L'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### elliptic-envelope / istella (rows full, shape X 100000x220; Xq 100000x220; y 100000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/r43/ingest/nv-results.txt (nv/n0003)`, ran on NVIDIA L40S (nv, RunPod) job n0003

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.42 (source build, main@4c75b4144) | identical | 168744.1 | 168744.1..168744.1 | 1 | - | - | 168744.1 | - | - (stored whole) | - | - | - | - | fraction_flagged=0.092480, jaccard_vs_sklearn=0.062240 | - | main board, one scored run | - | ok (main@4c75b4144 nv/n0003 2026-10-13; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(timeout: null) (copied from release-board-resume-r2; measured this run) |

settings: {'contamination': 0.1, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### elliptic-envelope / taxi (rows full, shape X 100000x11; Xq 100000x11; y 100000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/r43/ingest/nv-results.txt (nv/n0003)`, ran on NVIDIA L40S (nv, RunPod) job n0003

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.42 (source build, main@4c75b4144) | identical | 3360.4 | 3360.4..3360.4 | 1 | - | - | 3360.4 | - | - (stored whole) | - | - | - | - | fraction_flagged=0.102090, jaccard_vs_sklearn=0.925814 | - | main board, one scored run | - | ok (main@4c75b4144 nv/n0003 2026-10-13; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 2902.9 | 2902.9..2902.9 | 1 | 1.158 | - | 2902.9 | 2902.9 | 0.00 (cpu-arm) | 1.158 (whole/whole) | - | 348.9 | - | fraction_flagged=0.102470, jaccard_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'contamination': 0.1, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### embedding / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/nvidia-wave5/campaign/measured/algos.embedding.synthetic.rows=full/board.json (nv2/paired-bdd40e3c4e91a64b4d5cfb30b592b743a59d021bbb95302e4d97ce67a500253f)`, ran on d2fe567d3b88

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 2.2 ms) = 1.125. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 2.5 | 2.5..2.5 | 1 | - | - | 2.5 | 2.3 | 0.19 (upload_ms_separate) | - | - | 2170.7 | 1450.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | main board, one scored run | wheel | ok (main@d11b82091 nv2/paired-bdd40e3c4e91a64b4d5cfb30b592b743a59d021bbb95302e4d97ce67a500253f 2026-10-14; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 6.1 | 6.1..6.1 | 1 | 0.402 | - | - | 6.1 | - (stored kernel) | 0.372 (kernel/kernel) | - | 1215.4 | 782.6 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 2.2 | 2.2..2.2 | 1 | 1.125 | - | - | 2.2 | - (stored kernel) | 1.041 (kernel/kernel) | - | 1548.6 | 640.2 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

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
| mojolearn IDENTICAL | NVIDIA L40S (nv2, RunPod) | 0.8.44 | Xq | - | 0.6 | 0.6..0.6 | 1 | - | - | 0.6 | - | - (stored whole) | - | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.4 | 0.4..0.4 | 1 | 1.424 | - | - | 0.4 | - (stored kernel) | 1.424 (MIXED ours whole / arm kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-compile-fp32 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.4 | 0.4..0.4 | 1 | 1.276 | - | - | 0.4 | - (stored kernel) | 1.276 (MIXED ours whole / arm kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

### enet-cv / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0063)`, ran on NVIDIA L40S (nv2, RunPod) job v0063

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@2afa942e3) | identical | 464.6 | 464.6..464.6 | 1 | - | - | 464.6 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.326805, rmse=0.685379 | - | main board, one scored run | - | ok (main@2afa942e3 nv2/v0063 2026-10-11; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 112446.5 | 112446.5..112446.5 | 1 | 0.004 | - | 112446.5 | 112446.5 | 0.00 (cpu-arm) | 0.004 (whole/whole) | - | 8909.5 | - | finite=True, r2=0.326805, rmse=0.685379 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alphas': [0.001, 0.01, 0.1, 1.0], 'cv': 5, 'eps': 0.001, 'fit_intercept': True, 'l1_ratio': 0.5, 'max_iter': 1000, 'positive': False, 'random_state': 7, 'selection': 'cyclic', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### enet-cv / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0063)`, ran on NVIDIA L40S (nv2, RunPod) job v0063

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@2afa942e3) | identical | 19.4 | 19.4..19.4 | 1 | - | - | 19.4 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.909004, rmse=4.804486 | - | main board, one scored run | - | ok (main@2afa942e3 nv2/v0063 2026-10-11; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 507.5 | 507.5..507.5 | 1 | 0.038 | - | 507.5 | 507.5 | 0.00 (cpu-arm) | 0.038 (whole/whole) | - | 645.0 | - | finite=True, r2=0.909004, rmse=4.804486 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alphas': [0.001, 0.01, 0.1, 1.0], 'cv': 5, 'eps': 0.001, 'fit_intercept': True, 'l1_ratio': 0.5, 'max_iter': 1000, 'positive': False, 'random_state': 7, 'selection': 'cyclic', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gaussian-nb / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0668)`, ran on NVIDIA L40S (nv, RunPod) job n0668

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@6b882cf81) | identical | 92.1 | 92.1..92.1 | 1 | - | - | 92.1 | - | - (stored whole) | - | - | - | - | accuracy=0.876570, logloss=3.574420 | - | main board, one scored run | - | ok (main@6b882cf81 nv/n0668 2026-10-09; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 125.7 | 125.7..125.7 | 1 | 0.733 | - | 828.2 | 125.7 | 702.51 (upload_ms_untimed) | 0.111 (whole/whole) | - | 2931.8 | 1362.0 | accuracy=0.876570, logloss=3.416741 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 595.4 | 595.4..595.4 | 1 | 0.155 | - | 595.4 | 595.4 | 0.00 (cpu-arm) | 0.155 (whole/whole) | - | 2625.7 | - | accuracy=0.876530, logloss=3.417392 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'var_smoothing': 1e-09}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), GaussianNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gaussian-nb / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0668)`, ran on NVIDIA L40S (nv, RunPod) job n0668

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@6b882cf81) | identical | 20.0 | 20.0..20.0 | 1 | - | - | 20.0 | - | - (stored whole) | - | - | - | - | accuracy=0.719820, logloss=1.132249 | - | main board, one scored run | - | ok (main@6b882cf81 nv/n0668 2026-10-09; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 22.3 | 22.3..22.3 | 1 | 0.897 | - | 79.7 | 22.3 | 57.40 (upload_ms_untimed) | 0.251 (whole/whole) | - | 974.2 | 486.0 | accuracy=0.719810, logloss=1.132317 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 137.0 | 137.0..137.0 | 1 | 0.146 | - | 137.0 | 137.0 | 0.00 (cpu-arm) | 0.146 (whole/whole) | - | 314.3 | - | accuracy=0.719900, logloss=1.133898 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'var_smoothing': 1e-09}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), GaussianNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gaussian-rp / istella (rows full, shape X 900000x220; Xq 100000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on NVIDIA L40S (nv, RunPod) job n0314

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 54.3 | 54.3..54.3 | 1 | - | - | 54.3 | - | - (stored whole) | - | - | - | - | mean_abs_distortion=0.680693 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 31.4 | 31.4..31.4 | 1 | 1.729 | - | 31.4 | - | - (stored whole) | 1.729 (whole/whole) | - | 1813.0 | 438.0 | mean_abs_distortion=0.443920 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 32.3 | 32.3..32.3 | 1 | 1.684 | - | 32.3 | 32.3 | 0.00 (cpu-arm) | 1.684 (whole/whole) | - | 1134.6 | - | mean_abs_distortion=0.177966 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'compute_inverse_components': False, 'eps': 0.1, 'n_components': 10, 'random_state': 7}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), GaussianRandomProjection (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gaussian-rp / taxi (rows full, shape X 900000x11; Xq 100000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on NVIDIA L40S (nv, RunPod) job n0314

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 3.4 | 3.4..3.4 | 1 | - | - | 3.4 | - | - (stored whole) | - | - | - | - | mean_abs_distortion=0.345752 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 2.3 | 2.3..2.3 | 1 | 1.498 | - | 2.3 | - | - (stored whole) | 1.498 (whole/whole) | - | 894.4 | 438.0 | mean_abs_distortion=0.302259 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 3.6 | 3.6..3.6 | 1 | 0.957 | - | 3.6 | 3.6 | 0.00 (cpu-arm) | 0.957 (whole/whole) | - | 257.1 | - | mean_abs_distortion=0.339791 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'compute_inverse_components': False, 'eps': 0.1, 'n_components': 10, 'random_state': 7}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), GaussianRandomProjection (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gcn / istella (rows full, shape X 100000x220; indices 1521510; indptr 100001; y 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/nv_ingest/results_nvc3.txt (nv/n0006)`, ran on NVIDIA L40S (nv, RunPod) job n0006

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 8.5 ms) = 0.342; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 4.5 ms) = 0.654. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.42 (source build, main@8899093fc) | identical | 2.9 | 2.9..2.9 | 1 | - | - | 2.9 | - | - (stored whole) | - | - | - | - | relative_error_vs_own_host=0.000000 | - | main board, one scored run | - | ok (main@8899093fc nv/n0006 2026-10-13; identity vs the amd columns: n/a) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 13.7 | 13.7..13.7 | 1 | 0.213 | - | - | 13.7 | - (stored kernel) | 0.213 (MIXED ours whole / arm kernel) | - | 1426.9 | 1934.6 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 4.5 | 4.5..4.5 | 1 | 0.654 | - | - | 4.5 | - (stored kernel) | 0.654 (MIXED ours whole / arm kernel) | - | 1348.5 | 399.4 | max_rel_diff_vs_torch_eager_fp32=0.009646, rel_fro_vs_torch_eager_fp32=1.041e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 13.7 | 13.7..13.7 | 1 | 0.213 | - | - | 13.7 | - (stored kernel) | 0.213 (MIXED ours whole / arm kernel) | - | 1424.0 | 1934.6 | max_rel_diff_vs_torch_eager_fp32=281.122146, rel_fro_vs_torch_eager_fp32=0.0002661 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 8.2 | 8.2..8.2 | 1 | 0.356 | - | - | 8.2 | - (stored kernel) | 0.356 (MIXED ours whole / arm kernel) | - | 1315.1 | 399.4 | max_rel_diff_vs_torch_eager_fp32=281.117866, rel_fro_vs_torch_eager_fp32=0.0002661 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 13.8 | 13.8..13.8 | 1 | 0.212 | - | - | 13.8 | - (stored kernel) | 0.212 (MIXED ours whole / arm kernel) | - | 1548.3 | 2323.8 | max_rel_diff_vs_torch_eager_fp32=2718.059111, rel_fro_vs_torch_eager_fp32=0.002187 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 8.5 | 8.5..8.5 | 1 | 0.342 | - | - | 8.5 | - (stored kernel) | 0.342 (MIXED ours whole / arm kernel) | - | 1571.5 | 782.4 | max_rel_diff_vs_torch_eager_fp32=2718.055850, rel_fro_vs_torch_eager_fp32=0.002187 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'add_self_loops': True, 'bias': True, 'improved': False, 'normalize': True, 'out_channels': 128}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gcn / taxi (rows full, shape X 100000x11; indices 1258298; indptr 100001; y 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/nv_ingest/results_nvc3.txt (nv/n0006)`, ran on NVIDIA L40S (nv, RunPod) job n0006

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 9.3 ms) = 0.183; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 3.3 ms) = 0.512. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.42 (source build, main@8899093fc) | identical | 1.7 | 1.7..1.7 | 1 | - | - | 1.7 | - | - (stored whole) | - | - | - | - | relative_error_vs_own_host=0.000000 | - | main board, one scored run | - | ok (main@8899093fc nv/n0006 2026-10-13; identity vs the amd columns: n/a) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 12.6 | 12.6..12.6 | 1 | 0.135 | - | - | 12.6 | - (stored kernel) | 0.135 (MIXED ours whole / arm kernel) | - | 1344.8 | 1589.7 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 3.3 | 3.3..3.3 | 1 | 0.512 | - | - | 3.3 | - (stored kernel) | 0.512 (MIXED ours whole / arm kernel) | - | 1281.3 | 310.5 | max_rel_diff_vs_torch_eager_fp32=0.002327, rel_fro_vs_torch_eager_fp32=9.803e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 10.9 | 10.9..10.9 | 1 | 0.155 | - | - | 10.9 | - (stored kernel) | 0.155 (MIXED ours whole / arm kernel) | - | 1347.7 | 1589.7 | max_rel_diff_vs_torch_eager_fp32=0.001863, rel_fro_vs_torch_eager_fp32=7.223e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 3.7 | 3.7..3.7 | 1 | 0.456 | - | - | 3.7 | - (stored kernel) | 0.456 (MIXED ours whole / arm kernel) | - | 1242.7 | 310.5 | max_rel_diff_vs_torch_eager_fp32=0.002794, rel_fro_vs_torch_eager_fp32=9.801e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 15.9 | 15.9..15.9 | 1 | 0.107 | - | - | 15.9 | - (stored kernel) | 0.107 (MIXED ours whole / arm kernel) | - | 1464.0 | 1874.9 | max_rel_diff_vs_torch_eager_fp32=1509.509282, rel_fro_vs_torch_eager_fp32=0.002330 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 9.3 | 9.3..9.3 | 1 | 0.183 | - | - | 9.3 | - (stored kernel) | 0.183 (MIXED ours whole / arm kernel) | - | 1468.7 | 591.0 | max_rel_diff_vs_torch_eager_fp32=1509.508234, rel_fro_vs_torch_eager_fp32=0.002330 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'add_self_loops': True, 'bias': True, 'improved': False, 'normalize': True, 'out_channels': 128}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### global-avgpool / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/nvidia-wave5/campaign/measured/algos.global-avgpool.synthetic.rows=full/board.json (nv2/paired-632e4f96f147f2844d710d6fb85796bdb791c543034ead7fe8e50b017a6f08e9)`, ran on d2fe567d3b88

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 0.5 ms) = 0.575; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 0.5 ms) = 0.565. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 0.3 | 0.3..0.3 | 1 | - | - | - | 0.3 | - (stored kernel) | - | - | 203.4 | 682.0 | max_rel_diff_vs_torch_eager_fp32=0.030258, rel_fro_vs_torch_eager_fp32=1.512e-07 | - | main board, one scored run | wheel | ok (main@d11b82091 nv2/paired-632e4f96f147f2844d710d6fb85796bdb791c543034ead7fe8e50b017a6f08e9 2026-10-14; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 0.5 | 0.5..0.5 | 1 | 0.565 | - | - | 0.5 | - (stored kernel) | 0.565 (kernel/kernel) | - | 706.6 | 12.6 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 0.9 | 0.9..0.9 | 1 | 0.313 | - | - | 0.9 | - (stored kernel) | 0.313 (kernel/kernel) | - | 908.9 | 12.6 | max_rel_diff_vs_torch_eager_fp32=0.009731, rel_fro_vs_torch_eager_fp32=9.64e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 0.5 | 0.5..0.5 | 1 | 0.532 | - | - | 0.5 | - (stored kernel) | 0.532 (kernel/kernel) | - | 706.6 | 12.6 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 0.8 | 0.8..0.8 | 1 | 0.359 | - | - | 0.8 | - (stored kernel) | 0.359 (kernel/kernel) | - | 857.2 | 12.6 | max_rel_diff_vs_torch_eager_fp32=0.009731, rel_fro_vs_torch_eager_fp32=9.64e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 0.5 | 0.5..0.5 | 1 | 0.575 | - | - | 0.5 | - (stored kernel) | 0.575 (kernel/kernel) | - | 706.7 | 12.6 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 0.6 | 0.6..0.6 | 1 | 0.488 | - | - | 0.6 | - (stored kernel) | 0.488 (kernel/kernel) | - | 852.8 | 12.6 | max_rel_diff_vs_torch_eager_fp32=0.009731, rel_fro_vs_torch_eager_fp32=9.64e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'output_size': 1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-compile-tf32 | torch-eager-bf16 | torch-eager-fp32 | torch-eager-tf32 |
|---|---||---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| input_arrays | [{"dtype": "float32", "name": "x", "sha256": "8fe0278be594910f0b7581c036615fa6de591a3cb29304f63bff7d16046f6c97", "shape": [64, 512, 7, 7]}] | [{"dtype": "float32", "name": "x", "sha256": "8fe0278be594910f0b7581c036615fa6de591a3cb29304f63bff7d16046f6c97", "shape": [64, 512, 7, 7]}] | [{"dtype": "float32", "name": "x", "sha256": "8fe0278be594910f0b7581c036615fa6de591a3cb29304f63bff7d16046f6c97", "shape": [64, 512, 7, 7]}] | [{"dtype": "float32", "name": "x", "sha256": "8fe0278be594910f0b7581c036615fa6de591a3cb29304f63bff7d16046f6c97", "shape": [64, 512, 7, 7]}] | [{"dtype": "float32", "name": "x", "sha256": "8fe0278be594910f0b7581c036615fa6de591a3cb29304f63bff7d16046f6c97", "shape": [64, 512, 7, 7]}] | [{"dtype": "float32", "name": "x", "sha256": "8fe0278be594910f0b7581c036615fa6de591a3cb29304f63bff7d16046f6c97", "shape": [64, 512, 7, 7]}] | [{"dtype": "float32", "name": "x", "sha256": "8fe0278be594910f0b7581c036615fa6de591a3cb29304f63bff7d16046f6c97", "shape": [64, 512, 7, 7]}] |
| output_size | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | hardware | version | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | NVIDIA L40S (nv2, RunPod) | 0.8.44 | Xq | - | 0.0 | 0.0..0.0 | 1 | - | - | - | 0.0 | - (stored kernel) | - | - | max_rel_diff_vs_torch_eager_fp32=0.030258, rel_fro_vs_torch_eager_fp32=1.512e-07 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.0 | 0.0..0.0 | 1 | 1.004 | - | - | 0.0 | - (stored kernel) | 1.004 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.1 | 0.1..0.1 | 1 | 0.371 | - | - | 0.1 | - (stored kernel) | 0.371 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.009731, rel_fro_vs_torch_eager_fp32=9.64e-08 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-tf32 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.1 | 0.1..0.1 | 1 | 0.819 | - | - | 0.1 | - (stored kernel) | 0.819 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-tf32 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.1 | 0.1..0.1 | 1 | 0.350 | - | - | 0.1 | - (stored kernel) | 0.350 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.009731, rel_fro_vs_torch_eager_fp32=9.64e-08 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.1 | 0.1..0.1 | 1 | 0.755 | - | - | 0.1 | - (stored kernel) | 0.755 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.1 | 0.1..0.1 | 1 | 0.361 | - | - | 0.1 | - (stored kernel) | 0.361 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.009731, rel_fro_vs_torch_eager_fp32=9.64e-08 | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-tf32: forward(x) (no autograd)(Xq)

inference call, torch-compile-tf32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### global-maxpool / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/nvidia-wave5/campaign/measured/algos.global-maxpool.synthetic.rows=full/board.json (nv2/paired-81f7e55aaf889874211dee38ad616b0ba6473a2407d5447e983d34dc58d5ca33)`, ran on d2fe567d3b88

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 0.6 ms) = 0.605; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 0.6 ms) = 0.660. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 0.4 | 0.4..0.4 | 1 | - | - | - | 0.4 | - (stored kernel) | - | - | 203.5 | 682.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | main board, one scored run | wheel | ok (main@d11b82091 nv2/paired-81f7e55aaf889874211dee38ad616b0ba6473a2407d5447e983d34dc58d5ca33 2026-10-14; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 0.6 | 0.6..0.6 | 1 | 0.660 | - | - | 0.6 | - (stored kernel) | 0.660 (kernel/kernel) | - | 682.7 | 12.9 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 1.8 | 1.8..1.8 | 1 | 0.220 | - | - | 1.8 | - (stored kernel) | 0.220 (kernel/kernel) | - | 918.9 | 12.9 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 0.6 | 0.6..0.6 | 1 | 0.639 | - | - | 0.6 | - (stored kernel) | 0.639 (kernel/kernel) | - | 683.4 | 12.9 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 1.7 | 1.7..1.7 | 1 | 0.233 | - | - | 1.7 | - (stored kernel) | 0.233 (kernel/kernel) | - | 869.7 | 12.9 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 0.6 | 0.6..0.6 | 1 | 0.605 | - | - | 0.6 | - (stored kernel) | 0.605 (kernel/kernel) | - | 683.6 | 12.9 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 1.1 | 1.1..1.1 | 1 | 0.370 | - | - | 1.1 | - (stored kernel) | 0.370 (kernel/kernel) | - | 866.1 | 12.9 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'output_size': 1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-compile-tf32 | torch-eager-bf16 | torch-eager-fp32 | torch-eager-tf32 |
|---|---||---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| input_arrays | [{"dtype": "float32", "name": "x", "sha256": "8fe0278be594910f0b7581c036615fa6de591a3cb29304f63bff7d16046f6c97", "shape": [64, 512, 7, 7]}] | [{"dtype": "float32", "name": "x", "sha256": "8fe0278be594910f0b7581c036615fa6de591a3cb29304f63bff7d16046f6c97", "shape": [64, 512, 7, 7]}] | [{"dtype": "float32", "name": "x", "sha256": "8fe0278be594910f0b7581c036615fa6de591a3cb29304f63bff7d16046f6c97", "shape": [64, 512, 7, 7]}] | [{"dtype": "float32", "name": "x", "sha256": "8fe0278be594910f0b7581c036615fa6de591a3cb29304f63bff7d16046f6c97", "shape": [64, 512, 7, 7]}] | [{"dtype": "float32", "name": "x", "sha256": "8fe0278be594910f0b7581c036615fa6de591a3cb29304f63bff7d16046f6c97", "shape": [64, 512, 7, 7]}] | [{"dtype": "float32", "name": "x", "sha256": "8fe0278be594910f0b7581c036615fa6de591a3cb29304f63bff7d16046f6c97", "shape": [64, 512, 7, 7]}] | [{"dtype": "float32", "name": "x", "sha256": "8fe0278be594910f0b7581c036615fa6de591a3cb29304f63bff7d16046f6c97", "shape": [64, 512, 7, 7]}] |
| output_size | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | hardware | version | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | NVIDIA L40S (nv2, RunPod) | 0.8.44 | Xq | - | 0.1 | 0.1..0.1 | 1 | - | - | - | 0.1 | - (stored kernel) | - | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.1 | 0.1..0.1 | 1 | 1.085 | - | - | 0.1 | - (stored kernel) | 1.085 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.3 | 0.3..0.3 | 1 | 0.339 | - | - | 0.3 | - (stored kernel) | 0.339 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-tf32 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.1 | 0.1..0.1 | 1 | 1.002 | - | - | 0.1 | - (stored kernel) | 1.002 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-tf32 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.2 | 0.2..0.2 | 1 | 0.541 | - | - | 0.2 | - (stored kernel) | 0.541 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.1 | 0.1..0.1 | 1 | 0.878 | - | - | 0.1 | - (stored kernel) | 0.878 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.1 | 0.1..0.1 | 1 | 0.716 | - | - | 0.1 | - (stored kernel) | 0.716 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-tf32: forward(x) (no autograd)(Xq)

inference call, torch-compile-tf32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### graphsage / istella (rows full, shape X 100000x220; indices 1521510; indptr 100001; y 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/nv_ingest/results_nvc3.txt (nv/n0006)`, ran on NVIDIA L40S (nv, RunPod) job n0006

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 3.3 ms) = 1.860; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 4.2 ms) = 1.459. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.42 (source build, main@8899093fc) | identical | 6.2 | 6.2..6.2 | 1 | - | - | 6.2 | - | - (stored whole) | - | - | - | - | relative_error_vs_own_host=0.000000 | - | main board, one scored run | - | ok (main@8899093fc nv/n0006 2026-10-13; identity vs the amd columns: n/a) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 8.6 | 8.6..8.6 | 1 | 0.720 | - | - | 8.6 | - (stored kernel) | 0.720 (MIXED ours whole / arm kernel) | - | 1263.5 | 1667.4 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 4.2 | 4.2..4.2 | 1 | 1.459 | - | - | 4.2 | - (stored kernel) | 1.459 (MIXED ours whole / arm kernel) | - | 1310.3 | 403.7 | max_rel_diff_vs_torch_eager_fp32=0.134110, rel_fro_vs_torch_eager_fp32=9.7e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 8.5 | 8.5..8.5 | 1 | 0.729 | - | - | 8.5 | - (stored kernel) | 0.729 (MIXED ours whole / arm kernel) | - | 1259.5 | 1667.4 | max_rel_diff_vs_torch_eager_fp32=430.934131, rel_fro_vs_torch_eager_fp32=0.0002969 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 3.8 | 3.8..3.8 | 1 | 1.633 | - | - | 3.8 | - (stored kernel) | 1.633 (MIXED ours whole / arm kernel) | - | 1257.6 | 403.8 | max_rel_diff_vs_torch_eager_fp32=430.934131, rel_fro_vs_torch_eager_fp32=0.0002969 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 8.1 | 8.1..8.1 | 1 | 0.761 | - | - | 8.1 | - (stored kernel) | 0.761 (MIXED ours whole / arm kernel) | - | 1399.9 | 1642.9 | max_rel_diff_vs_torch_eager_fp32=3907.114267, rel_fro_vs_torch_eager_fp32=0.003366 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 3.3 | 3.3..3.3 | 1 | 1.860 | - | - | 3.3 | - (stored kernel) | 1.860 (MIXED ours whole / arm kernel) | - | 1447.6 | 330.6 | max_rel_diff_vs_torch_eager_fp32=3678.210080, rel_fro_vs_torch_eager_fp32=0.003054 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'aggr': 'mean', 'bias': True, 'normalize': False, 'out_channels': 128, 'project': False, 'root_weight': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### graphsage / taxi (rows full, shape X 100000x11; indices 1258298; indptr 100001; y 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/nv_ingest/results_nvc3.txt (nv/n0006)`, ran on NVIDIA L40S (nv, RunPod) job n0006

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 1.7 ms) = 1.063; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 1.6 ms) = 1.117. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.42 (source build, main@8899093fc) | identical | 1.8 | 1.8..1.8 | 1 | - | - | 1.8 | - | - (stored whole) | - | - | - | - | relative_error_vs_own_host=0.000000 | - | main board, one scored run | - | ok (main@8899093fc nv/n0006 2026-10-13; identity vs the amd columns: n/a) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 1.6 | 1.6..1.6 | 1 | 1.117 | - | - | 1.6 | - (stored kernel) | 1.117 (MIXED ours whole / arm kernel) | - | 1181.3 | 289.1 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 1.9 | 1.9..1.9 | 1 | 0.951 | - | - | 1.9 | - (stored kernel) | 0.951 (MIXED ours whole / arm kernel) | - | 1246.5 | 240.0 | max_rel_diff_vs_torch_eager_fp32=0.238419, rel_fro_vs_torch_eager_fp32=6.954e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 2.3 | 2.3..2.3 | 1 | 0.795 | - | - | 2.3 | - (stored kernel) | 0.795 (MIXED ours whole / arm kernel) | - | 1182.5 | 289.1 | max_rel_diff_vs_torch_eager_fp32=0.029851, rel_fro_vs_torch_eager_fp32=4.218e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 2.0 | 2.0..2.0 | 1 | 0.900 | - | - | 2.0 | - (stored kernel) | 0.900 (MIXED ours whole / arm kernel) | - | 1240.6 | 240.0 | max_rel_diff_vs_torch_eager_fp32=1475.334167, rel_fro_vs_torch_eager_fp32=0.0002145 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 1.7 | 1.7..1.7 | 1 | 1.063 | - | - | 1.7 | - (stored kernel) | 1.063 (MIXED ours whole / arm kernel) | - | 1316.0 | 262.6 | max_rel_diff_vs_torch_eager_fp32=5460.333333, rel_fro_vs_torch_eager_fp32=0.003557 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 1.7 | 1.7..1.7 | 1 | 1.052 | - | - | 1.7 | - (stored kernel) | 1.052 (MIXED ours whole / arm kernel) | - | 1364.9 | 167.2 | max_rel_diff_vs_torch_eager_fp32=4608.154297, rel_fro_vs_torch_eager_fp32=0.003285 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'aggr': 'mean', 'bias': True, 'normalize': False, 'out_channels': 128, 'project': False, 'root_weight': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gru-clf / synthetic (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1091)`, ran on NVIDIA L40S (nv2, RunPod) job v1091

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 1311.6 ms) = 0.341; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 1103.7 ms) = 0.405. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 447.1 | 447.1..447.1 | 1 | - | - | 447.1 | - | - (stored whole) | - | - | - | - | accuracy=0.971625, logloss=0.070054 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1091 2026-10-10; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1103.7 | 1103.7..1103.7 | 1 | 0.405 | - | - | 1103.7 | - (stored kernel) | 0.405 (MIXED ours whole / arm kernel) | - | 1122.3 | 643.7 | accuracy=0.971842 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1190.3 | 1190.3..1190.3 | 1 | 0.376 | - | - | 1190.3 | - (stored kernel) | 0.376 (MIXED ours whole / arm kernel) | - | 1174.3 | 643.7 | accuracy=0.971842 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1063.3 | 1063.3..1063.3 | 1 | 0.420 | - | - | 1063.3 | - (stored kernel) | 0.420 (MIXED ours whole / arm kernel) | - | 1123.7 | 643.7 | accuracy=0.971842 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1248.7 | 1248.7..1248.7 | 1 | 0.358 | - | - | 1248.7 | - (stored kernel) | 0.358 (MIXED ours whole / arm kernel) | - | 1174.6 | 643.7 | accuracy=0.971842 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1548.3 | 1548.3..1548.3 | 1 | 0.289 | - | - | 1548.3 | - (stored kernel) | 0.289 (MIXED ours whole / arm kernel) | - | 1487.8 | 340.9 | accuracy=0.971951 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1311.6 | 1311.6..1311.6 | 1 | 0.341 | - | - | 1311.6 | - (stored kernel) | 0.341 (MIXED ours whole / arm kernel) | - | 1539.4 | 340.9 | accuracy=0.971951 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gru-clf / taxi-hourly (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1091)`, ran on NVIDIA L40S (nv2, RunPod) job v1091

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 789.0 ms) = 0.567; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 591.3 ms) = 0.757. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 447.7 | 447.7..447.7 | 1 | - | - | 447.7 | - | - (stored whole) | - | - | - | - | accuracy=0.865723, logloss=0.303537 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1091 2026-10-10; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 591.3 | 591.3..591.3 | 1 | 0.757 | - | - | 591.3 | - (stored kernel) | 0.757 (MIXED ours whole / arm kernel) | - | 1122.2 | 643.7 | accuracy=0.865668 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 662.7 | 662.7..662.7 | 1 | 0.676 | - | - | 662.7 | - (stored kernel) | 0.676 (MIXED ours whole / arm kernel) | - | 1173.5 | 643.7 | accuracy=0.865668 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 641.7 | 641.7..641.7 | 1 | 0.698 | - | - | 641.7 | - (stored kernel) | 0.698 (MIXED ours whole / arm kernel) | - | 1123.2 | 643.7 | accuracy=0.865668 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 733.4 | 733.4..733.4 | 1 | 0.611 | - | - | 733.4 | - (stored kernel) | 0.611 (MIXED ours whole / arm kernel) | - | 1174.8 | 643.7 | accuracy=0.865668 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 838.4 | 838.4..838.4 | 1 | 0.534 | - | - | 838.4 | - (stored kernel) | 0.534 (MIXED ours whole / arm kernel) | - | 1487.9 | 340.9 | accuracy=0.865668 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 789.0 | 789.0..789.0 | 1 | 0.567 | - | - | 789.0 | - (stored kernel) | 0.567 (MIXED ours whole / arm kernel) | - | 1539.0 | 340.9 | accuracy=0.865668 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gru-reg / synthetic (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1091)`, ran on NVIDIA L40S (nv2, RunPod) job v1091

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 1333.4 ms) = 0.335; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 1133.5 ms) = 0.394. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 446.3 | 446.3..446.3 | 1 | - | - | 446.3 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.982337, rmse=0.153975 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1091 2026-10-10; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1133.5 | 1133.5..1133.5 | 1 | 0.394 | - | - | 1133.5 | - (stored kernel) | 0.394 (MIXED ours whole / arm kernel) | - | 1136.9 | 643.4 | finite=True, r2=0.981946, rmse=0.155672 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1143.5 | 1143.5..1143.5 | 1 | 0.390 | - | - | 1143.5 | - (stored kernel) | 0.390 (MIXED ours whole / arm kernel) | - | 1188.0 | 643.4 | finite=True, r2=0.981946, rmse=0.155672 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1020.0 | 1020.0..1020.0 | 1 | 0.438 | - | - | 1020.0 | - (stored kernel) | 0.438 (MIXED ours whole / arm kernel) | - | 1137.7 | 643.4 | finite=True, r2=0.981946, rmse=0.155672 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1145.8 | 1145.8..1145.8 | 1 | 0.390 | - | - | 1145.8 | - (stored kernel) | 0.390 (MIXED ours whole / arm kernel) | - | 1190.4 | 643.4 | finite=True, r2=0.981946, rmse=0.155672 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1528.3 | 1528.3..1528.3 | 1 | 0.292 | - | - | 1528.3 | - (stored kernel) | 0.292 (MIXED ours whole / arm kernel) | - | 1534.4 | 340.6 | finite=True, r2=0.981898, rmse=0.155878 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1333.4 | 1333.4..1333.4 | 1 | 0.335 | - | - | 1333.4 | - (stored kernel) | 0.335 (MIXED ours whole / arm kernel) | - | 1585.9 | 340.6 | finite=True, r2=0.981898, rmse=0.155878 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gru-reg / taxi-hourly (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1091)`, ran on NVIDIA L40S (nv2, RunPod) job v1091

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 696.1 ms) = 0.642; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 650.5 ms) = 0.687. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 447.0 | 447.0..447.0 | 1 | - | - | 447.0 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.747875, rmse=0.544554 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1091 2026-10-10; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 650.5 | 650.5..650.5 | 1 | 0.687 | - | - | 650.5 | - (stored kernel) | 0.687 (MIXED ours whole / arm kernel) | - | 1137.2 | 643.4 | finite=True, r2=0.748219, rmse=0.544182 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 859.6 | 859.6..859.6 | 1 | 0.520 | - | - | 859.6 | - (stored kernel) | 0.520 (MIXED ours whole / arm kernel) | - | 1188.0 | 643.4 | finite=True, r2=0.748219, rmse=0.544182 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 545.3 | 545.3..545.3 | 1 | 0.820 | - | - | 545.3 | - (stored kernel) | 0.820 (MIXED ours whole / arm kernel) | - | 1138.4 | 643.4 | finite=True, r2=0.748219, rmse=0.544182 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 721.3 | 721.3..721.3 | 1 | 0.620 | - | - | 721.3 | - (stored kernel) | 0.620 (MIXED ours whole / arm kernel) | - | 1190.0 | 643.4 | finite=True, r2=0.748219, rmse=0.544182 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 696.1 | 696.1..696.1 | 1 | 0.642 | - | - | 696.1 | - (stored kernel) | 0.642 (MIXED ours whole / arm kernel) | - | 1534.1 | 340.6 | finite=True, r2=0.748325, rmse=0.544067 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 762.0 | 762.0..762.0 | 1 | 0.587 | - | - | 762.0 | - (stored kernel) | 0.587 (MIXED ours whole / arm kernel) | - | 1585.0 | 340.6 | finite=True, r2=0.748325, rmse=0.544067 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### incremental-pca / istella (rows full, shape X 900000x220; Xq 100000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on NVIDIA L40S (nv, RunPod) job n0314

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 864.1 | 864.1..864.1 | 1 | - | - | 864.1 | - | - (stored whole) | - | - | - | - | explained_variance_fraction=1.000000 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 1515.2 | 1515.2..1515.2 | 1 | 0.570 | - | 2206.1 | 1515.2 | 690.93 (upload_ms_untimed) | 0.392 (whole/whole) | - | 3076.8 | 1314.0 | explained_variance_fraction=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 17717.5 | 17717.5..17717.5 | 1 | 0.049 | - | 17717.5 | 17717.5 | 0.00 (cpu-arm) | 0.049 (whole/whole) | - | 2303.9 | - | explained_variance_fraction=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'batch_size': 65536, 'n_components': 10, 'whiten': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), IncrementalPCA (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### incremental-pca / taxi (rows full, shape X 900000x11; Xq 100000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on NVIDIA L40S (nv, RunPod) job n0314

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 107.6 | 107.6..107.6 | 1 | - | - | 107.6 | - | - (stored whole) | - | - | - | - | explained_variance_fraction=0.999995 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 178.2 | 178.2..178.2 | 1 | 0.604 | - | 231.5 | 178.2 | 53.32 (upload_ms_untimed) | 0.465 (whole/whole) | - | 1176.1 | 518.0 | explained_variance_fraction=0.999995 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 669.5 | 669.5..669.5 | 1 | 0.161 | - | 669.5 | 669.5 | 0.00 (cpu-arm) | 0.161 (whole/whole) | - | 307.8 | - | explained_variance_fraction=0.999995 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'batch_size': 65536, 'n_components': 10, 'whiten': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), IncrementalPCA (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ivf-pq / istella (rows full, shape index 400000x220; queries 4000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0128)`, ran on NVIDIA L40S (nv2, RunPod) job v0128

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 756.9 | 756.9..756.9 | 1 | - | - | 756.9 | - | - (stored whole) | - | - | - | - | recall_at_10=0.802100 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0128 2026-10-11; identity vs the amd columns: n/a) |
| cuvs-gpu | cuvs | gpu | NVIDIA L40S (host cc560ebdaf91) | cuvs 26.08.01 | opponent | 1905.2 | 1905.2..1905.2 | 1 | 0.397 | - | 2248.5 | 1905.2 | 343.31 (upload_ms_untimed) | 0.337 (whole/whole) | - | 1793.1 | 822.0 | recall_at_10=0.791300 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| faiss-cpu | faiss | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | faiss 1.15.1 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(timeout: null) (copied from release-board-resume-r2; measured this run) |

memory, ours, faiss-cpu: host not sampled; GPU not sampled

memory, cuvs-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'kmeans_n_iters': 20, 'n_lists': 1024, 'n_neighbors': 10, 'n_probes': 32, 'pq_bits': 8, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ivf-pq / taxi (rows full, shape index 400000x11; queries 4000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1020)`, ran on NVIDIA L40S (nv2, RunPod) job v1020

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@0a7b206f1) | identical | 100.1 | 100.1..100.1 | 1 | - | - | 100.1 | - | - (stored whole) | - | - | - | - | recall_at_10=0.981250 | - | main board, one scored run | - | ok (main@0a7b206f1 nv2/v1020 2026-10-09; identity vs amd-mi325x: MATCH) |
| cuvs-gpu | cuvs | gpu | NVIDIA L40S (host 24a11adce16e) | cuvs 26.08.01 | opponent | 471.5 | 471.5..471.5 | 1 | 0.212 | - | 502.2 | 471.5 | 30.72 (upload_ms_untimed) | 0.199 (whole/whole) | - | 995.4 | 466.0 | recall_at_10=0.976875 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| faiss-cpu | faiss | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | faiss 1.15.1 | opponent | 67512.8 | 67512.8..67512.8 | 1 | 0.001 | - | 67512.8 | 67512.8 | 0.00 (cpu-arm) | 0.001 (whole/whole) | - | 135.3 | - | recall_at_10=0.979450 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuvs-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, faiss-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'kmeans_n_iters': 20, 'n_lists': 1024, 'n_neighbors': 10, 'n_probes': 32, 'pq_bits': 8, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ivf-refine / istella (rows full, shape index 400000x220; queries 4000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0128)`, ran on NVIDIA L40S (nv2, RunPod) job v0128

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 753.5 | 753.5..753.5 | 1 | - | - | 753.5 | - | - (stored whole) | - | - | - | - | recall_at_10=0.994575 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0128 2026-10-11; identity vs the amd columns: n/a) |
| cuvs-gpu | cuvs | gpu | NVIDIA L40S (host cc560ebdaf91) | cuvs 26.08.01 | opponent | 1255.0 | 1255.0..1255.0 | 1 | 0.600 | - | 1596.3 | 1255.0 | 341.25 (upload_ms_untimed) | 0.472 (whole/whole) | - | 1815.7 | 822.0 | recall_at_10=0.993625 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| faiss-cpu | faiss | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | faiss 1.15.1 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(timeout: null) (copied from release-board-resume-r2; measured this run) |

memory, ours, faiss-cpu: host not sampled; GPU not sampled

memory, cuvs-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'kmeans_n_iters': 20, 'n_lists': 1024, 'n_neighbors': 10, 'n_probes': 32, 'pq_bits': 8, 'random_state': 7, 'refine_ratio': 4}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ivf-refine / taxi (rows full, shape index 400000x11; queries 4000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on NVIDIA L40S (nv, RunPod) job n0314

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 349.6 | 349.6..349.6 | 1 | - | - | 349.6 | - | - (stored whole) | - | - | - | - | recall_at_10=0.999675 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs the amd columns: n/a) |
| cuvs-gpu | cuvs | gpu | NVIDIA L40S (host 24a11adce16e) | cuvs 26.08.01 | opponent | 381.6 | 381.6..381.6 | 1 | 0.916 | - | 408.0 | 381.6 | 26.37 (upload_ms_untimed) | 0.857 (whole/whole) | - | 1026.7 | 468.0 | recall_at_10=0.999050 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| faiss-cpu | faiss | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | faiss 1.15.1 | opponent | 70811.7 | 70811.7..70811.7 | 1 | 0.005 | - | 70811.7 | 70811.7 | 0.00 (cpu-arm) | 0.005 (whole/whole) | - | 153.2 | - | recall_at_10=0.999375 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuvs-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, faiss-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'kmeans_n_iters': 20, 'n_lists': 1024, 'n_neighbors': 10, 'n_probes': 32, 'pq_bits': 8, 'random_state': 7, 'refine_ratio': 4}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ivf-sq / taxi (rows full, shape index 400000x11; queries 4000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0318)`, ran on NVIDIA L40S (nv, RunPod) job n0318

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@de2b2b739) | identical | 155.5 | 155.5..155.5 | 1 | - | - | 155.5 | - | - (stored whole) | - | - | - | - | recall_at_10=0.934975 | - | main board, one scored run | - | ok (main@de2b2b739 nv/n0318 2026-10-08; identity vs the amd columns: n/a) |
| cuvs-gpu | cuvs | gpu | NVIDIA L40S (host 24a11adce16e) | cuvs 26.08.01 | opponent | 138.0 | 138.0..138.0 | 1 | 1.126 | - | 163.5 | 138.0 | 25.50 (upload_ms_untimed) | 0.951 (whole/whole) | - | 960.6 | 464.0 | recall_at_10=0.772500 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| faiss-cpu | faiss | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | faiss 1.15.1 | opponent | 4836.7 | 4836.7..4836.7 | 1 | 0.032 | - | 4836.7 | 4836.7 | 0.00 (cpu-arm) | 0.032 (whole/whole) | - | 124.2 | - | recall_at_10=0.857050 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuvs-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, faiss-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'kmeans_n_iters': 20, 'n_lists': 1024, 'n_neighbors': 10, 'n_probes': 32, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### jl-min-dim / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/nv_ingest/results_nvc2.txt (nv/n0001)`, ran on NVIDIA L40S (nv, RunPod) job n0001

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.42 (source build, main@9c7ad7413) | identical | 9.1 | 9.1..9.1 | 1 | - | - | 9.1 | - | - (stored whole) | - | - | - | - | equal_fraction_vs_sklearn=1.000000 | - | main board, one scored run | - | ok (main@9c7ad7413 nv/n0001 2026-10-12; identity vs the amd columns: n/a) |

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### kbins / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/nv_ingest/results_nvc2.txt (nv/n0001)`, ran on NVIDIA L40S (nv, RunPod) job n0001

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.42 (source build, main@9c7ad7413) | identical | 276.1 | 276.1..276.1 | 1 | - | - | 276.1 | - | - (stored whole) | - | - | - | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x220, rel_diff_vs_sklearn=0.000000 | - | main board, one scored run | - | ok (main@9c7ad7413 nv/n0001 2026-10-12; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 739.0 | 739.0..739.0 | 1 | 0.374 | - | 1447.6 | 739.0 | 708.65 (upload_ms_untimed) | 0.191 (whole/whole) | - | 3068.2 | 1442.0 | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 5641.8 | 5641.8..5641.8 | 1 | 0.049 | - | 5641.8 | 5641.8 | 0.00 (cpu-arm) | 0.049 (whole/whole) | - | 1459.8 | - | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'encode': 'ordinal', 'n_bins': 16, 'quantile_method': 'linear', 'random_state': 7, 'strategy': 'quantile', 'subsample': None}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), KBinsDiscretizer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### kbins / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/nv_ingest/results_nvc2.txt (nv/n0001)`, ran on NVIDIA L40S (nv, RunPod) job n0001

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.42 (source build, main@9c7ad7413) | identical | 7.9 | 7.9..7.9 | 1 | - | - | 7.9 | - | - (stored whole) | - | - | - | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x11, rel_diff_vs_sklearn=0.000000 | - | main board, one scored run | - | ok (main@9c7ad7413 nv/n0001 2026-10-12; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 26.6 | 26.6..26.6 | 1 | 0.296 | - | 85.0 | 26.6 | 58.41 (upload_ms_untimed) | 0.093 (whole/whole) | - | 952.0 | 488.0 | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 173.1 | 173.1..173.1 | 1 | 0.045 | - | 173.1 | 173.1 | 0.00 (cpu-arm) | 0.045 (whole/whole) | - | 263.6 | - | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'encode': 'ordinal', 'n_bins': 16, 'quantile_method': 'linear', 'random_state': 7, 'strategy': 'quantile', 'subsample': None}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), KBinsDiscretizer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### kernel-pca / istella (rows full, shape X 10000x220; Xq 10000x220; y 10000; yq 10000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/nv_ingest/results_nvc2.txt (nv/n0001)`, ran on NVIDIA L40S (nv, RunPod) job n0001

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.42 (source build, main@9c7ad7413) | identical | 1145.9 | 1145.9..1145.9 | 1 | - | - | 1145.9 | - | - (stored whole) | - | - | - | - | rbf_feature_distance_relative_stress=0.610112, subspace_cos_vs_sklearn=1.000000 | - | main board, one scored run | - | ok (main@9c7ad7413 nv/n0001 2026-10-12; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 1513.3 | 1513.3..1513.3 | 1 | 0.757 | - | 1513.3 | 1513.3 | 0.00 (cpu-arm) | 0.757 (whole/whole) | - | 2714.0 | - | subspace_cos_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'coef0': 1, 'degree': 3, 'eigen_solver': 'auto', 'fit_inverse_transform': False, 'iterated_power': 'auto', 'kernel': 'rbf', 'n_components': 8, 'random_state': 7, 'remove_zero_eig': False, 'tol': 0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### kernel-pca / taxi (rows full, shape X 10000x11; Xq 10000x11; y 10000; yq 10000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/nv_ingest/results_nvc2.txt (nv/n0001)`, ran on NVIDIA L40S (nv, RunPod) job n0001

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.42 (source build, main@9c7ad7413) | identical | 1088.5 | 1088.5..1088.5 | 1 | - | - | 1088.5 | - | - (stored whole) | - | - | - | - | rbf_feature_distance_relative_stress=0.545709, subspace_cos_vs_sklearn=1.000000 | - | main board, one scored run | - | ok (main@9c7ad7413 nv/n0001 2026-10-12; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 1058.7 | 1058.7..1058.7 | 1 | 1.028 | - | 1058.7 | 1058.7 | 0.00 (cpu-arm) | 1.028 (whole/whole) | - | 1733.8 | - | subspace_cos_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'coef0': 1, 'degree': 3, 'eigen_solver': 'auto', 'fit_inverse_transform': False, 'iterated_power': 'auto', 'kernel': 'rbf', 'n_components': 8, 'random_state': 7, 'remove_zero_eig': False, 'tol': 0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### kernel-shap / istella (rows full, shape X 100000x220; Xq 100x220; y 100000; yq 100)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0110)`, ran on NVIDIA L40S (nv2, RunPod) job v0110

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 3928.3 | 3928.3..3928.3 | 1 | - | - | 3928.3 | - | - (stored whole) | - | - | - | - | rel_error_vs_exact=8.307e-08 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0110 2026-10-11; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 15329.7 | 15329.7..15329.7 | 1 | 0.256 | - | 15334.1 | 15329.7 | 4.41 (upload_ms_untimed) | 0.256 (whole/whole) | - | 2042.9 | 438.0 | rel_error_vs_exact=0.032177 | - | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (copied from opponents-default-20261006; measured this run) |
| shap-cpu | shap | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | shap 0.51.0 | opponent | 25374.0 | 25374.0..25374.0 | 1 | 0.155 | - | 25374.0 | 25374.0 | 0.00 (cpu-arm) | 0.155 (whole/whole) | - | 2183.9 | - | rel_error_vs_exact=8.574e-15 | - | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, shap-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'l1_reg': False, 'link': 'identity', 'n_background': 100, 'nsamples': 2048}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### kernel-shap / taxi (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0583)`, ran on NVIDIA L40S (nv, RunPod) job n0583

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@03b648834) | identical | 319.5 | 319.5..319.5 | 1 | - | - | 319.5 | - | - (stored whole) | - | - | - | - | rel_error_vs_exact=1.056e-07 | - | main board, one scored run | - | ok (main@03b648834 nv/n0583 2026-10-09; identity vs amd-mi325x: MATCH) |

settings: {'n_background': 100, 'nsamples': 2048, 'link': 'identity', 'l1_reg': False, 'explained_model': 'one float32 ridge (alpha 1, centered) on every arm, coefficients solved once in a fixed order by the harness'}. Rows: None. Timed: None.

mismatch: opponents withheld: the lane settings at this tree's HEAD differ from the settings opponents-specific-20261006 recorded for its opponent race (an opponent job must score them again)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### knn-imputer / istella (rows full, shape X 100000x220; X_true 100000x220; Xq 20000x220; Xq_true 20000x220; y 100000; yq 20000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0104)`, ran on NVIDIA L40S (nv2, RunPod) job v0104

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 51.7 | 51.7..51.7 | 1 | - | - | 51.7 | - | - (stored whole) | - | - | - | - | masked_rmse=323953.237332 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0104 2026-10-11; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 23.2 | 23.2..23.2 | 1 | 2.230 | - | 23.2 | 23.2 | 0.00 (cpu-arm) | 2.230 (whole/whole) | - | 3499.1 | - | masked_rmse=986208.700423 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'add_indicator': False, 'keep_empty_features': False, 'metric': 'nan_euclidean', 'n_neighbors': 5, 'weights': 'uniform'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### knn-imputer / taxi (rows full, shape X 100000x11; X_true 100000x11; Xq 20000x11; Xq_true 20000x11; y 100000; yq 20000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1020)`, ran on NVIDIA L40S (nv2, RunPod) job v1020

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@0a7b206f1) | identical | 2.7 | 2.7..2.7 | 1 | - | - | 2.7 | - | - (stored whole) | - | - | - | - | masked_rmse=6.151696 | - | main board, one scored run | - | ok (main@0a7b206f1 nv2/v1020 2026-10-09; identity vs amd-mi325x: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 6.0 | 6.0..6.0 | 1 | 0.452 | - | 6.0 | 6.0 | 0.00 (cpu-arm) | 0.452 (whole/whole) | - | 1952.7 | - | masked_rmse=5.263919 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'add_indicator': False, 'keep_empty_features': False, 'metric': 'nan_euclidean', 'n_neighbors': 5, 'weights': 'uniform'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### label-binarizer / istella (rows full, shape X 1000000x8; Xq 100000x8; lab 1000000; labq 100000; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/recovered-inference/algos.label-binarizer.istella.rows=full/board.json (nv2/paired-1b51a2d3e6be7337c7c515be39930da414aaeffab0ff1542f03ef0be032afb5f)`, ran on b6c1fd6fb058

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 53.7 | 53.7..53.7 | 1 | - | - | 53.7 | 53.3 | 0.48 (upload_ms_separate) | - | - | 644.8 | 682.0 | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x16, rel_diff_vs_sklearn=0.000000 | - | main board, one scored run | wheel | ok (main@fcdfa0fe2 nv2/paired-1b51a2d3e6be7337c7c515be39930da414aaeffab0ff1542f03ef0be032afb5f 2026-10-14; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host b6c1fd6fb058) | cuml 26.08.00 | opponent | 18.3 | 18.3..18.3 | 1 | 2.940 | - | 44.4 | 18.3 | 26.16 (upload_ms_untimed) | 1.209 (whole/whole) | - | 1498.4 | 558.0 | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x16, rel_diff_vs_sklearn=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'neg_label': 0, 'pos_label': 1}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), LabelBinarizer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| neg_label | 0 | 0 |
| pos_label | 1 | 1 |
| seed | "none (deterministic)" | "none (deterministic)" |
| sparse_output | false | false |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | hardware | version | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | NVIDIA L40S (nv2, RunPod) | 0.8.44 | Xq | - | 6.1 | 6.1..6.1 | 1 | - | - | 6.1 | - | - (stored whole) | - | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x16, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| cuml-gpu | NVIDIA L40S (host b6c1fd6fb058) | cuml 26.08.00 | Xq | - | 6.1 | 6.1..6.1 | 1 | 0.992 | - | - | 6.1 | - (stored kernel) | 0.992 (MIXED ours whole / arm kernel) | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x16, rel_diff_vs_sklearn=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: transform(Xq)(Xq)

inference call, cuml-gpu: transform(Xq)(Xq)

### label-binarizer / taxi (rows full, shape X 1000000x5; Xq 100000x5; lab 1000000; labq 100000; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/recovered-inference/algos.label-binarizer.taxi.rows=full/board.json (nv2/paired-d7f9ff05915c39e48f79456c6b007b339f4c4c2e179ee3c122ffc1347dcd8d83)`, ran on b6c1fd6fb058

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 781.0 | 781.0..781.0 | 1 | - | - | 781.0 | 780.1 | 0.85 (upload_ms_separate) | - | - | 6652.7 | 1706.0 | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x259, rel_diff_vs_sklearn=0.000000 | - | main board, one scored run | wheel | ok (main@fcdfa0fe2 nv2/paired-d7f9ff05915c39e48f79456c6b007b339f4c4c2e179ee3c122ffc1347dcd8d83 2026-10-14; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host b6c1fd6fb058) | cuml 26.08.00 | opponent | 54.5 | 54.5..54.5 | 1 | 14.337 | - | 91.5 | 54.5 | 37.01 (upload_ms_untimed) | 8.537 (whole/whole) | - | 6627.0 | 1560.0 | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x259, rel_diff_vs_sklearn=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'neg_label': 0, 'pos_label': 1}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), LabelBinarizer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| neg_label | 0 | 0 |
| pos_label | 1 | 1 |
| seed | "none (deterministic)" | "none (deterministic)" |
| sparse_output | false | false |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | hardware | version | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | NVIDIA L40S (nv2, RunPod) | 0.8.44 | Xq | - | 83.5 | 83.5..83.5 | 1 | - | - | 83.5 | - | - (stored whole) | - | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x259, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| cuml-gpu | NVIDIA L40S (host b6c1fd6fb058) | cuml 26.08.00 | Xq | - | 30.2 | 30.2..30.2 | 1 | 2.768 | - | - | 30.2 | - (stored kernel) | 2.768 (MIXED ours whole / arm kernel) | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x259, rel_diff_vs_sklearn=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: transform(Xq)(Xq)

inference call, cuml-gpu: transform(Xq)(Xq)

### label-encoder / istella (rows full, shape X 1000000x8; Xq 100000x8; lab 1000000; labq 100000; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/nv_ingest/results_nvc2.txt (nv/n0001)`, ran on NVIDIA L40S (nv, RunPod) job n0001

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.42 (source build, main@9c7ad7413) | identical | 7.1 | 7.1..7.1 | 1 | - | - | 7.1 | - | - (stored whole) | - | - | - | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000, rel_diff_vs_sklearn=0.000000 | - | main board, one scored run | - | ok (main@9c7ad7413 nv/n0001 2026-10-12; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 7.9 | 7.9..7.9 | 1 | 0.891 | - | 37.3 | 7.9 | 29.43 (upload_ms_untimed) | 0.189 (whole/whole) | - | 1022.1 | 490.0 | output_shape=100000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 28.5 | 28.5..28.5 | 1 | 0.248 | - | 28.5 | 28.5 | 0.00 (cpu-arm) | 0.248 (whole/whole) | - | 308.0 | - | output_shape=100000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), LabelEncoder (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### label-encoder / taxi (rows full, shape X 1000000x5; Xq 100000x5; lab 1000000; labq 100000; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/nv_ingest/results_nvc2.txt (nv/n0001)`, ran on NVIDIA L40S (nv, RunPod) job n0001

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.42 (source build, main@9c7ad7413) | identical | 6.8 | 6.8..6.8 | 1 | - | - | 6.8 | - | - (stored whole) | - | - | - | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000, rel_diff_vs_sklearn=0.000000 | - | main board, one scored run | - | ok (main@9c7ad7413 nv/n0001 2026-10-12; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 38.9 | 38.9..38.9 | 1 | 0.175 | - | 68.5 | 38.9 | 29.58 (upload_ms_untimed) | 0.099 (whole/whole) | - | 1025.1 | 472.0 | output_shape=100000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 110.3 | 110.3..110.3 | 1 | 0.062 | - | 110.3 | 110.3 | 0.00 (cpu-arm) | 0.062 (whole/whole) | - | 295.4 | - | output_shape=100000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), LabelEncoder (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lamb / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/nv_ingest/results_nvc3.txt (nv/n0001)`, ran on NVIDIA L40S (nv, RunPod) job n0001

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32: - (no completed torch fp32 arm). Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.42 (source build, main@12412eca4) | identical | 21.7 | 21.7..21.7 | 1 | - | - | 21.7 | - | - (stored whole) | - | - | - | - | relative_error_vs_own_host=0.000000 | - | main board, one scored run | - | ok (main@12412eca4 nv/n0001 2026-10-12; identity vs the amd columns: n/a) |

settings: {'betas': [0.9, 0.999], 'eps': 1e-06, 'lr': 0.001, 'weight_decay': 0.01}. Rows: None. Timed: None.

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lars / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0075)`, ran on NVIDIA L40S (nv2, RunPod) job v0075

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@6cd6bed7f) | identical | 337.9 | 337.9..337.9 | 1 | - | - | 337.9 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.328699, rmse=0.684414 | - | main board, one scored run | - | ok (main@6cd6bed7f nv2/v0075 2026-10-11; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 43.4 | 43.4..43.4 | 1 | 7.778 | - | 745.1 | 43.4 | 701.69 (upload_ms_untimed) | 0.454 (whole/whole) | - | 3008.6 | 1374.0 | finite=True, r2=0.328088, rmse=0.684726 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 922.6 | 922.6..922.6 | 1 | 0.366 | - | 922.6 | 922.6 | 0.00 (cpu-arm) | 0.366 (whole/whole) | - | 1975.3 | - | finite=True, r2=-4.245e+13, rmse=5.442e+06 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'eps': 2.220446049250313e-16, 'fit_intercept': True, 'n_nonzero_coefs': 500, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lars / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0075)`, ran on NVIDIA L40S (nv2, RunPod) job v0075

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@6cd6bed7f) | identical | 3.1 | 3.1..3.1 | 1 | - | - | 3.1 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.908983, rmse=4.805051 | - | main board, one scored run | - | ok (main@6cd6bed7f nv2/v0075 2026-10-11; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 4.5 | 4.5..4.5 | 1 | 0.676 | - | 63.0 | 4.5 | 58.47 (upload_ms_untimed) | 0.049 (whole/whole) | - | 1046.3 | 498.0 | finite=True, r2=0.908983, rmse=4.805052 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 50.8 | 50.8..50.8 | 1 | 0.060 | - | 50.8 | 50.8 | 0.00 (cpu-arm) | 0.060 (whole/whole) | - | 292.2 | - | finite=True, r2=0.908983, rmse=4.805055 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'eps': 2.220446049250313e-16, 'fit_intercept': True, 'n_nonzero_coefs': 500, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lasso-cv / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0063)`, ran on NVIDIA L40S (nv2, RunPod) job v0063

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@2afa942e3) | identical | 428.9 | 428.9..428.9 | 1 | - | - | 428.9 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.325506, rmse=0.686040 | - | main board, one scored run | - | ok (main@2afa942e3 nv2/v0063 2026-10-11; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 97412.0 | 97412.0..97412.0 | 1 | 0.004 | - | 97412.0 | 97412.0 | 0.00 (cpu-arm) | 0.004 (whole/whole) | - | 8911.4 | - | finite=True, r2=0.325507, rmse=0.686040 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alphas': [0.001, 0.01, 0.1, 1.0], 'cv': 5, 'eps': 0.001, 'fit_intercept': True, 'max_iter': 1000, 'positive': False, 'random_state': 7, 'selection': 'cyclic', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lasso-cv / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0063)`, ran on NVIDIA L40S (nv2, RunPod) job v0063

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@2afa942e3) | identical | 18.0 | 18.0..18.0 | 1 | - | - | 18.0 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.909038, rmse=4.803593 | - | main board, one scored run | - | ok (main@2afa942e3 nv2/v0063 2026-10-11; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 313.1 | 313.1..313.1 | 1 | 0.058 | - | 313.1 | 313.1 | 0.00 (cpu-arm) | 0.058 (whole/whole) | - | 610.4 | - | finite=True, r2=0.909038, rmse=4.803593 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alphas': [0.001, 0.01, 0.1, 1.0], 'cv': 5, 'eps': 0.001, 'fit_intercept': True, 'max_iter': 1000, 'positive': False, 'random_state': 7, 'selection': 'cyclic', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lasso-lars / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0075)`, ran on NVIDIA L40S (nv2, RunPod) job v0075

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@6cd6bed7f) | identical | 90.5 | 90.5..90.5 | 1 | - | - | 90.5 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.310834, rmse=0.693461 | - | main board, one scored run | - | ok (main@6cd6bed7f nv2/v0075 2026-10-11; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 904.7 | 904.7..904.7 | 1 | 0.100 | - | 904.7 | 904.7 | 0.00 (cpu-arm) | 0.100 (whole/whole) | - | 1969.5 | - | finite=True, r2=0.310816, rmse=0.693470 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 0.01, 'eps': 2.220446049250313e-16, 'fit_intercept': True, 'max_iter': 500, 'positive': False, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lasso-lars / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0075)`, ran on NVIDIA L40S (nv2, RunPod) job v0075

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@6cd6bed7f) | identical | 3.2 | 3.2..3.2 | 1 | - | - | 3.2 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.908998, rmse=4.804663 | - | main board, one scored run | - | ok (main@6cd6bed7f nv2/v0075 2026-10-11; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 62.7 | 62.7..62.7 | 1 | 0.051 | - | 62.7 | 62.7 | 0.00 (cpu-arm) | 0.051 (whole/whole) | - | 296.3 | - | finite=True, r2=0.908998, rmse=4.804667 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 0.01, 'eps': 2.220446049250313e-16, 'fit_intercept': True, 'max_iter': 500, 'positive': False, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### layernorm / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/nvidia/campaign/measured/algos.layernorm.synthetic.rows=full/board.json (nv2/paired-677ad4132a2408301e2286269856efa208b29be8b92092a281b95c418c51ad9d)`, ran on d2fe567d3b88

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 2.0 ms) = 0.591; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 1.5 ms) = 0.781. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 1.2 | 1.2..1.2 | 1 | - | - | - | 1.2 | - (stored kernel) | - | - | 1821.5 | 938.0 | max_rel_diff_vs_torch_eager_fp32=0.011127, rel_fro_vs_torch_eager_fp32=7.007e-08 | - | main board, one scored run | wheel | ok (main@79fcd24a5 nv2/paired-677ad4132a2408301e2286269856efa208b29be8b92092a281b95c418c51ad9d 2026-10-14; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 1.5 | 1.5..1.5 | 1 | 0.781 | - | - | 1.5 | - (stored kernel) | 0.781 (kernel/kernel) | - | 930.0 | 320.1 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 1.6 | 1.6..1.6 | 1 | 0.728 | - | - | 1.6 | - (stored kernel) | 0.728 (kernel/kernel) | - | 1181.9 | 336.1 | max_rel_diff_vs_torch_eager_fp32=0.006419, rel_fro_vs_torch_eager_fp32=5.316e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 1.6 | 1.6..1.6 | 1 | 0.750 | - | - | 1.6 | - (stored kernel) | 0.750 (kernel/kernel) | - | 928.6 | 320.1 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 1.6 | 1.6..1.6 | 1 | 0.721 | - | - | 1.6 | - (stored kernel) | 0.721 (kernel/kernel) | - | 1127.1 | 336.1 | max_rel_diff_vs_torch_eager_fp32=0.006419, rel_fro_vs_torch_eager_fp32=5.316e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 2.8 | 2.8..2.8 | 1 | 0.426 | - | - | 2.8 | - (stored kernel) | 0.426 (kernel/kernel) | - | 932.6 | 320.1 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 2.0 | 2.0..2.0 | 1 | 0.591 | - | - | 2.0 | - (stored kernel) | 0.591 (kernel/kernel) | - | 1120.8 | 336.1 | max_rel_diff_vs_torch_eager_fp32=0.006419, rel_fro_vs_torch_eager_fp32=5.316e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'bias': True, 'elementwise_affine': True, 'eps': 1e-05, 'normalized_shape': 1024}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-compile-tf32 | torch-eager-bf16 | torch-eager-fp32 | torch-eager-tf32 |
|---|---||---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| bias | true | true | true | true | true | true | true |
| elementwise_affine | true | true | true | true | true | true | true |
| eps | 1e-05 | 1e-05 | 1e-05 | 1e-05 | 1e-05 | 1e-05 | 1e-05 |
| input_arrays | [{"dtype": "float32", "name": "x", "sha256": "128bb6c8835123e5f00ffe3700a26223ba8b37f7280afaec092571b2861be93c", "shape": [16384, 1024]}, {"dtype": "float32", "name": "bias", "sha256": "ad7facb2586fc6e966c004d7d1d16b024f5805ff7cb47c7a85dabd8b48892ca7", "shape": [1024]}, {"dtype": "float32", "name": "weight", "sha256": "e9bac255f4adc7cb4ada9298e193a5ff66b434d15afabd458505325f29c398c7", "shape": [1024]}] | [{"dtype": "float32", "name": "x", "sha256": "128bb6c8835123e5f00ffe3700a26223ba8b37f7280afaec092571b2861be93c", "shape": [16384, 1024]}, {"dtype": "float32", "name": "bias", "sha256": "ad7facb2586fc6e966c004d7d1d16b024f5805ff7cb47c7a85dabd8b48892ca7", "shape": [1024]}, {"dtype": "float32", "name": "weight", "sha256": "e9bac255f4adc7cb4ada9298e193a5ff66b434d15afabd458505325f29c398c7", "shape": [1024]}] | [{"dtype": "float32", "name": "x", "sha256": "128bb6c8835123e5f00ffe3700a26223ba8b37f7280afaec092571b2861be93c", "shape": [16384, 1024]}, {"dtype": "float32", "name": "bias", "sha256": "ad7facb2586fc6e966c004d7d1d16b024f5805ff7cb47c7a85dabd8b48892ca7", "shape": [1024]}, {"dtype": "float32", "name": "weight", "sha256": "e9bac255f4adc7cb4ada9298e193a5ff66b434d15afabd458505325f29c398c7", "shape": [1024]}] | [{"dtype": "float32", "name": "x", "sha256": "128bb6c8835123e5f00ffe3700a26223ba8b37f7280afaec092571b2861be93c", "shape": [16384, 1024]}, {"dtype": "float32", "name": "bias", "sha256": "ad7facb2586fc6e966c004d7d1d16b024f5805ff7cb47c7a85dabd8b48892ca7", "shape": [1024]}, {"dtype": "float32", "name": "weight", "sha256": "e9bac255f4adc7cb4ada9298e193a5ff66b434d15afabd458505325f29c398c7", "shape": [1024]}] | [{"dtype": "float32", "name": "x", "sha256": "128bb6c8835123e5f00ffe3700a26223ba8b37f7280afaec092571b2861be93c", "shape": [16384, 1024]}, {"dtype": "float32", "name": "bias", "sha256": "ad7facb2586fc6e966c004d7d1d16b024f5805ff7cb47c7a85dabd8b48892ca7", "shape": [1024]}, {"dtype": "float32", "name": "weight", "sha256": "e9bac255f4adc7cb4ada9298e193a5ff66b434d15afabd458505325f29c398c7", "shape": [1024]}] | [{"dtype": "float32", "name": "x", "sha256": "128bb6c8835123e5f00ffe3700a26223ba8b37f7280afaec092571b2861be93c", "shape": [16384, 1024]}, {"dtype": "float32", "name": "bias", "sha256": "ad7facb2586fc6e966c004d7d1d16b024f5805ff7cb47c7a85dabd8b48892ca7", "shape": [1024]}, {"dtype": "float32", "name": "weight", "sha256": "e9bac255f4adc7cb4ada9298e193a5ff66b434d15afabd458505325f29c398c7", "shape": [1024]}] | [{"dtype": "float32", "name": "x", "sha256": "128bb6c8835123e5f00ffe3700a26223ba8b37f7280afaec092571b2861be93c", "shape": [16384, 1024]}, {"dtype": "float32", "name": "bias", "sha256": "ad7facb2586fc6e966c004d7d1d16b024f5805ff7cb47c7a85dabd8b48892ca7", "shape": [1024]}, {"dtype": "float32", "name": "weight", "sha256": "e9bac255f4adc7cb4ada9298e193a5ff66b434d15afabd458505325f29c398c7", "shape": [1024]}] |
| normalized_shape | 1024 | 1024 | 1024 | 1024 | 1024 | 1024 | 1024 |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | hardware | version | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | NVIDIA L40S (nv2, RunPod) | 0.8.44 | Xq | - | 0.2 | 0.2..0.2 | 1 | - | - | - | 0.2 | - (stored kernel) | - | - | max_rel_diff_vs_torch_eager_fp32=0.011127, rel_fro_vs_torch_eager_fp32=7.007e-08 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.2 | 0.2..0.2 | 1 | 0.963 | - | - | 0.2 | - (stored kernel) | 0.963 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.3 | 0.3..0.3 | 1 | 0.590 | - | - | 0.3 | - (stored kernel) | 0.590 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.006419, rel_fro_vs_torch_eager_fp32=5.316e-08 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-tf32 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.2 | 0.2..0.2 | 1 | 0.928 | - | - | 0.2 | - (stored kernel) | 0.928 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-tf32 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.3 | 0.3..0.3 | 1 | 0.593 | - | - | 0.3 | - (stored kernel) | 0.593 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.006419, rel_fro_vs_torch_eager_fp32=5.316e-08 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.2 | 0.2..0.2 | 1 | 0.843 | - | - | 0.2 | - (stored kernel) | 0.843 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.3 | 0.3..0.3 | 1 | 0.594 | - | - | 0.3 | - (stored kernel) | 0.594 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.006419, rel_fro_vs_torch_eager_fp32=5.316e-08 | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-tf32: forward(x) (no autograd)(Xq)

inference call, torch-compile-tf32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### lda-clf / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0607)`, ran on NVIDIA L40S (nv, RunPod) job n0607

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@d3c0fd72b) | identical | 309.9 | 309.9..309.9 | 1 | - | - | 309.9 | - | - (stored whole) | - | - | - | - | accuracy=0.912830, logloss=0.235875 | - | main board, one scored run | - | ok (main@d3c0fd72b nv/n0607 2026-10-09; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 11501.6 | 11501.6..11501.6 | 1 | 0.027 | - | 11501.6 | 11501.6 | 0.00 (cpu-arm) | 0.027 (whole/whole) | - | 4591.9 | - | accuracy=0.899510, logloss=0.438978 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'solver': 'svd', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lda-clf / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0607)`, ran on NVIDIA L40S (nv, RunPod) job n0607

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@d3c0fd72b) | identical | 27.9 | 27.9..27.9 | 1 | - | - | 27.9 | - | - (stored whole) | - | - | - | - | accuracy=0.762530, logloss=0.539763 | - | main board, one scored run | - | ok (main@d3c0fd72b nv/n0607 2026-10-09; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 398.4 | 398.4..398.4 | 1 | 0.070 | - | 398.4 | 398.4 | 0.00 (cpu-arm) | 0.070 (whole/whole) | - | 462.6 | - | accuracy=0.762530, logloss=0.539767 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'solver': 'svd', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lion / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/nv_ingest/results_nvc3.txt (nv/n0001)`, ran on NVIDIA L40S (nv, RunPod) job n0001

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32: - (no completed torch fp32 arm). Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.42 (source build, main@12412eca4) | identical | 8.5 | 8.5..8.5 | 1 | - | - | 8.5 | - | - (stored whole) | - | - | - | - | relative_error_vs_own_host=0.000000 | - | main board, one scored run | - | ok (main@12412eca4 nv/n0001 2026-10-12; identity vs the amd columns: n/a) |

settings: {'betas': [0.9, 0.99], 'lr': 0.001, 'weight_decay': 0.0}. Rows: None. Timed: None.

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### louvain / istella (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0523)`, ran on NVIDIA L40S (nv, RunPod) job n0523

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@4df610f3b) | identical | 178.8 | 178.8..178.8 | 1 | - | - | 178.8 | - | - (stored whole) | - | - | - | - | modularity=0.911187, n_communities=40 | - | main board, one scored run | - | ok (main@4df610f3b nv/n0523 2026-10-08; identity vs amd-mi325x: MATCH) |

settings: {'resolution': 1.0, 'threshold': 1e-07, 'seed': 7}. Rows: None. Timed: None.

mismatch: networkx louvain_communities(seed=7) and cuGraph louvain (max_level=100) are order-dependent; ours pins a colour-batched move order (a fixed hash colouring, ties to the smallest community id)

mismatch: ours takes the CSR graph as an (indptr, indices, n) tuple and uploads it inside the clock (the 2026-10-08 CSR entries; before, a dense n x n adjacency built before the clock); networkx and cuGraph take the graph itself

mismatch: opponents withheld: the lane settings at this tree's HEAD differ from the settings opponents-specific-20261006 recorded for its opponent race (an opponent job must score them again)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### louvain / taxi (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0523)`, ran on NVIDIA L40S (nv, RunPod) job n0523

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@4df610f3b) | identical | 38.2 | 38.2..38.2 | 1 | - | - | 38.2 | - | - (stored whole) | - | - | - | - | modularity=0.941953, n_communities=58 | - | main board, one scored run | - | ok (main@4df610f3b nv/n0523 2026-10-08; identity vs amd-mi325x: MATCH) |

settings: {'resolution': 1.0, 'threshold': 1e-07, 'seed': 7}. Rows: None. Timed: None.

mismatch: networkx louvain_communities(seed=7) and cuGraph louvain (max_level=100) are order-dependent; ours pins a colour-batched move order (a fixed hash colouring, ties to the smallest community id)

mismatch: ours takes the CSR graph as an (indptr, indices, n) tuple and uploads it inside the clock (the 2026-10-08 CSR entries; before, a dense n x n adjacency built before the clock); networkx and cuGraph take the graph itself

mismatch: opponents withheld: the lane settings at this tree's HEAD differ from the settings opponents-default-20261006 recorded for its opponent race (an opponent job must score them again)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lstm-clf / synthetic (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1091)`, ran on NVIDIA L40S (nv2, RunPod) job v1091

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 743.9 ms) = 0.735; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 516.0 ms) = 1.059. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 546.6 | 546.6..546.6 | 1 | - | - | 546.6 | - | - (stored whole) | - | - | - | - | accuracy=0.967068, logloss=0.079720 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1091 2026-10-10; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 516.0 | 516.0..516.0 | 1 | 1.059 | - | - | 516.0 | - (stored kernel) | 1.059 (MIXED ours whole / arm kernel) | - | 1122.6 | 695.4 | accuracy=0.968696 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 931.7 | 931.7..931.7 | 1 | 0.587 | - | - | 931.7 | - (stored kernel) | 0.587 (MIXED ours whole / arm kernel) | - | 1173.6 | 695.4 | accuracy=0.968696 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 532.1 | 532.1..532.1 | 1 | 1.027 | - | - | 532.1 | - (stored kernel) | 1.027 (MIXED ours whole / arm kernel) | - | 1124.1 | 695.4 | accuracy=0.968696 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 675.9 | 675.9..675.9 | 1 | 0.809 | - | - | 675.9 | - (stored kernel) | 0.809 (MIXED ours whole / arm kernel) | - | 1175.8 | 695.4 | accuracy=0.968696 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 743.9 | 743.9..743.9 | 1 | 0.735 | - | - | 743.9 | - (stored kernel) | 0.735 (MIXED ours whole / arm kernel) | - | 1490.9 | 367.2 | accuracy=0.968913 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 838.2 | 838.2..838.2 | 1 | 0.652 | - | - | 838.2 | - (stored kernel) | 0.652 (MIXED ours whole / arm kernel) | - | 1542.0 | 367.2 | accuracy=0.968913 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lstm-clf / taxi-hourly (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1091)`, ran on NVIDIA L40S (nv2, RunPod) job v1091

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 1735.3 ms) = 0.315; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 1246.5 ms) = 0.438. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 546.5 | 546.5..546.5 | 1 | - | - | 546.5 | - | - (stored whole) | - | - | - | - | accuracy=0.870443, logloss=0.297146 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1091 2026-10-10; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1246.5 | 1246.5..1246.5 | 1 | 0.438 | - | - | 1246.5 | - (stored kernel) | 0.438 (MIXED ours whole / arm kernel) | - | 1122.8 | 695.4 | accuracy=0.868218 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1257.4 | 1257.4..1257.4 | 1 | 0.435 | - | - | 1257.4 | - (stored kernel) | 0.435 (MIXED ours whole / arm kernel) | - | 1173.7 | 695.4 | accuracy=0.868218 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1230.4 | 1230.4..1230.4 | 1 | 0.444 | - | - | 1230.4 | - (stored kernel) | 0.444 (MIXED ours whole / arm kernel) | - | 1124.0 | 695.4 | accuracy=0.868164 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1277.4 | 1277.4..1277.4 | 1 | 0.428 | - | - | 1277.4 | - (stored kernel) | 0.428 (MIXED ours whole / arm kernel) | - | 1175.5 | 695.4 | accuracy=0.868164 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1805.5 | 1805.5..1805.5 | 1 | 0.303 | - | - | 1805.5 | - (stored kernel) | 0.303 (MIXED ours whole / arm kernel) | - | 1490.6 | 367.2 | accuracy=0.868327 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1735.3 | 1735.3..1735.3 | 1 | 0.315 | - | - | 1735.3 | - (stored kernel) | 0.315 (MIXED ours whole / arm kernel) | - | 1542.3 | 367.2 | accuracy=0.868327 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lstm-reg / synthetic (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1091)`, ran on NVIDIA L40S (nv2, RunPod) job v1091

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 857.8 ms) = 0.636; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 564.5 ms) = 0.966. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 545.6 | 545.6..545.6 | 1 | - | - | 545.6 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.979398, rmse=0.166295 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1091 2026-10-10; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 826.4 | 826.4..826.4 | 1 | 0.660 | - | - | 826.4 | - (stored kernel) | 0.660 (MIXED ours whole / arm kernel) | - | 1131.8 | 695.1 | finite=True, r2=0.981013, rmse=0.159641 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 564.5 | 564.5..564.5 | 1 | 0.966 | - | - | 564.5 | - (stored kernel) | 0.966 (MIXED ours whole / arm kernel) | - | 1183.3 | 695.1 | finite=True, r2=0.981013, rmse=0.159641 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 801.6 | 801.6..801.6 | 1 | 0.681 | - | - | 801.6 | - (stored kernel) | 0.681 (MIXED ours whole / arm kernel) | - | 1133.8 | 695.1 | finite=True, r2=0.981013, rmse=0.159642 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 797.0 | 797.0..797.0 | 1 | 0.684 | - | - | 797.0 | - (stored kernel) | 0.684 (MIXED ours whole / arm kernel) | - | 1184.9 | 695.1 | finite=True, r2=0.981013, rmse=0.159642 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 978.7 | 978.7..978.7 | 1 | 0.557 | - | - | 978.7 | - (stored kernel) | 0.557 (MIXED ours whole / arm kernel) | - | 1536.5 | 366.9 | finite=True, r2=0.980998, rmse=0.159706 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 857.8 | 857.8..857.8 | 1 | 0.636 | - | - | 857.8 | - (stored kernel) | 0.636 (MIXED ours whole / arm kernel) | - | 1587.9 | 366.9 | finite=True, r2=0.980998, rmse=0.159706 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lstm-reg / taxi-hourly (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1091)`, ran on NVIDIA L40S (nv2, RunPod) job v1091

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 1539.5 ms) = 0.355; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 1243.7 ms) = 0.439. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 545.8 | 545.8..545.8 | 1 | - | - | 545.8 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.754306, rmse=0.537563 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1091 2026-10-10; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1330.9 | 1330.9..1330.9 | 1 | 0.410 | - | - | 1330.9 | - (stored kernel) | 0.410 (MIXED ours whole / arm kernel) | - | 1131.8 | 695.1 | finite=True, r2=0.751679, rmse=0.540430 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1243.7 | 1243.7..1243.7 | 1 | 0.439 | - | - | 1243.7 | - (stored kernel) | 0.439 (MIXED ours whole / arm kernel) | - | 1183.5 | 695.1 | finite=True, r2=0.751679, rmse=0.540430 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1253.8 | 1253.8..1253.8 | 1 | 0.435 | - | - | 1253.8 | - (stored kernel) | 0.435 (MIXED ours whole / arm kernel) | - | 1133.7 | 695.1 | finite=True, r2=0.751680, rmse=0.540429 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1285.2 | 1285.2..1285.2 | 1 | 0.425 | - | - | 1285.2 | - (stored kernel) | 0.425 (MIXED ours whole / arm kernel) | - | 1185.1 | 695.1 | finite=True, r2=0.751680, rmse=0.540429 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1711.2 | 1711.2..1711.2 | 1 | 0.319 | - | - | 1711.2 | - (stored kernel) | 0.319 (MIXED ours whole / arm kernel) | - | 1536.4 | 366.9 | finite=True, r2=0.751712, rmse=0.540394 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1539.5 | 1539.5..1539.5 | 1 | 0.355 | - | - | 1539.5 | - (stored kernel) | 0.355 (MIXED ours whole / arm kernel) | - | 1587.7 | 366.9 | finite=True, r2=0.751712, rmse=0.540394 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lstsq / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0088)`, ran on NVIDIA L40S (nv2, RunPod) job v0088

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 311.7 | 311.7..311.7 | 1 | - | - | 311.7 | - | - (stored whole) | - | - | - | - | relative_residual=0.849956 | - | main board, one scored run | - | ok (main@a1c96a5ba nv2/v0088 2026-10-11; identity vs the amd columns: n/a) |
| torch-gpu | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 220.0 | 220.0..220.0 | 1 | 1.417 | - | - | 220.0 | - (stored kernel) | 1.417 (MIXED ours whole / arm kernel) | - | 1783.9 | 3516.2 | relative_residual=nan | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| cupy-gpu | cupy | gpu | NVIDIA L40S (host 24a11adce16e) | cupy 14.2.0 | opponent | 244.2 | 244.2..244.2 | 1 | 1.276 | - | - | 244.2 | - (stored kernel) | 1.276 (MIXED ours whole / arm kernel) | - | 2740.5 | 9158.0 | relative_residual=0.849957 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| numpy-cpu | numpy | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | numpy 2.4.6 | opponent | 7725.5 | 7725.5..7725.5 | 1 | 0.040 | - | 7725.5 | 7725.5 | 0.00 (cpu-arm) | 0.040 (whole/whole) | - | 4348.1 | - | relative_residual=0.876581 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, cupy-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lstsq / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0088)`, ran on NVIDIA L40S (nv2, RunPod) job v0088

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 7.5 | 7.5..7.5 | 1 | - | - | 7.5 | - | - (stored whole) | - | - | - | - | relative_residual=0.756366 | - | main board, one scored run | - | ok (main@a1c96a5ba nv2/v0088 2026-10-11; identity vs the amd columns: n/a) |
| torch-gpu | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 7.1 | 7.1..7.1 | 1 | 1.059 | - | - | 7.1 | - (stored kernel) | 1.059 (MIXED ours whole / arm kernel) | - | 913.0 | 1122.8 | relative_residual=0.756366 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| cupy-gpu | cupy | gpu | NVIDIA L40S (host cc560ebdaf91) | cupy 14.2.0 | opponent | 9.9 | 9.9..9.9 | 1 | 0.756 | - | - | 9.9 | - (stored kernel) | 0.756 (MIXED ours whole / arm kernel) | - | 727.9 | 2778.0 | relative_residual=0.756366 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| numpy-cpu | numpy | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | numpy 2.4.6 | opponent | 183.1 | 183.1..183.1 | 1 | 0.041 | - | 183.1 | 183.1 | 0.00 (cpu-arm) | 0.041 (whole/whole) | - | 279.6 | - | relative_residual=0.756366 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, cupy-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lu-factor / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/nvidia-wave10/campaign/measured/algos.lu-factor.synthetic.rows=full/board.json (nv2/paired-999d7580f4ce2aa62e4125b77d215a646a1ddfa5193e526fcc3241f23120a304)`, ran on d2fe567d3b88

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 358.8 | 358.8..358.8 | 1 | - | - | 358.8 | 341.0 | 17.83 (upload_ms_separate) | - | - | 2030.7 | 1450.0 | relative_residual=3.256e-06 | - | main board, one scored run | wheel | ok (main@dbc9b18c1 nv2/paired-999d7580f4ce2aa62e4125b77d215a646a1ddfa5193e526fcc3241f23120a304 2026-10-14; identity vs the amd columns: n/a) |
| torch-gpu | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 35.8 | 35.8..35.8 | 1 | 10.020 | - | - | 35.8 | - (stored kernel) | 9.522 (kernel/kernel) | - | 830.7 | 528.3 | relative_residual=3.386e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| cupy-gpu | cupy | gpu | NVIDIA L40S (host d2fe567d3b88) | cupy 14.2.0 | opponent | 39.3 | 39.3..39.3 | 1 | 9.137 | - | - | 39.3 | - (stored kernel) | 8.683 (kernel/kernel (whole not derivable)) | - | 825.9 | 1108.0 | relative_residual=3.386e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cupy-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cupy-gpu | ours | torch-gpu |
|---|---||---|---||---|---|
| library (source) | cupy (declared) | mojolearn (declared) | torch (declared) |
| input_arrays | [{"dtype": "float32", "sha256": "e3500eebc0f84bfe008ee69884a10df5fab41e5bec2a57ae24d76cb8dbddeb20", "shape": [8192, 8192]}, {"dtype": "float32", "sha256": "56589d513ac0c940e300abfd9e69b92d6fe1b6ba3f08b861624290d33b5b50ff", "shape": [8192, 64]}] | [{"dtype": "float32", "sha256": "e3500eebc0f84bfe008ee69884a10df5fab41e5bec2a57ae24d76cb8dbddeb20", "shape": [8192, 8192]}, {"dtype": "float32", "sha256": "56589d513ac0c940e300abfd9e69b92d6fe1b6ba3f08b861624290d33b5b50ff", "shape": [8192, 64]}] | [{"dtype": "float32", "sha256": "e3500eebc0f84bfe008ee69884a10df5fab41e5bec2a57ae24d76cb8dbddeb20", "shape": [8192, 8192]}, {"dtype": "float32", "sha256": "56589d513ac0c940e300abfd9e69b92d6fe1b6ba3f08b861624290d33b5b50ff", "shape": [8192, 64]}] |
| seed | 7 | 7 | 7 |

### lu-solve / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/nvidia-wave10/campaign/measured/algos.lu-solve.synthetic.rows=full/board.json (nv2/paired-9e91de6c6f96f2f9ea2a91b3b4bb551a294b649577b8073849dad95b781e3c69)`, ran on d2fe567d3b88

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 350.2 | 350.2..350.2 | 1 | - | - | 350.2 | 338.9 | 11.26 (upload_ms_separate) | - | - | 2296.8 | 1194.0 | relative_residual=3.256e-06 | - | main board, one scored run | wheel | ok (main@dbc9b18c1 nv2/paired-9e91de6c6f96f2f9ea2a91b3b4bb551a294b649577b8073849dad95b781e3c69 2026-10-14; identity vs the amd columns: n/a) |
| torch-gpu | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 35.8 | 35.8..35.8 | 1 | 9.789 | - | - | 35.8 | - (stored kernel) | 9.474 (kernel/kernel) | - | 826.9 | 528.3 | relative_residual=3.386e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| cupy-gpu | cupy | gpu | NVIDIA L40S (host d2fe567d3b88) | cupy 14.2.0 | opponent | 35.6 | 35.6..35.6 | 1 | 9.839 | - | - | 35.6 | - (stored kernel) | 9.523 (kernel/kernel (whole not derivable)) | - | 782.2 | 984.0 | relative_residual=3.386e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cupy-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cupy-gpu | ours | torch-gpu |
|---|---||---|---||---|---|
| library (source) | cupy (declared) | mojolearn (declared) | torch (declared) |
| input_arrays | [{"dtype": "float32", "sha256": "e3500eebc0f84bfe008ee69884a10df5fab41e5bec2a57ae24d76cb8dbddeb20", "shape": [8192, 8192]}, {"dtype": "float32", "sha256": "56589d513ac0c940e300abfd9e69b92d6fe1b6ba3f08b861624290d33b5b50ff", "shape": [8192, 64]}] | [{"dtype": "float32", "sha256": "e3500eebc0f84bfe008ee69884a10df5fab41e5bec2a57ae24d76cb8dbddeb20", "shape": [8192, 8192]}, {"dtype": "float32", "sha256": "56589d513ac0c940e300abfd9e69b92d6fe1b6ba3f08b861624290d33b5b50ff", "shape": [8192, 64]}] | [{"dtype": "float32", "sha256": "e3500eebc0f84bfe008ee69884a10df5fab41e5bec2a57ae24d76cb8dbddeb20", "shape": [8192, 8192]}, {"dtype": "float32", "sha256": "56589d513ac0c940e300abfd9e69b92d6fe1b6ba3f08b861624290d33b5b50ff", "shape": [8192, 64]}] |
| seed | 7 | 7 | 7 |

### maxabs-scaler / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/recovered-inference/algos.maxabs-scaler.istella.rows=full/board.json (nv2/paired-e317f203df3e0a85e9c9081eaa9fd961dd2cc4eb856b4e34b89818e910eea367)`, ran on b6c1fd6fb058

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 37.4 | 37.4..37.4 | 1 | - | - | 37.4 | 1.1 | 36.29 (upload_ms_separate) | - | - | 1623.2 | 1706.0 | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x220, rel_diff_vs_sklearn=0.000000 | - | main board, one scored run | wheel | ok (main@fcdfa0fe2 nv2/paired-e317f203df3e0a85e9c9081eaa9fd961dd2cc4eb856b4e34b89818e910eea367 2026-10-14; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host b6c1fd6fb058) | cuml 26.08.00 | opponent | 23.6 | 23.6..23.6 | 1 | 1.581 | - | 653.9 | 23.6 | 630.30 (upload_ms_untimed) | 0.057 (whole/whole) | - | 3338.0 | 1442.0 | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x220, rel_diff_vs_sklearn=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), MaxAbsScaler (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| copy | true | true |
| seed | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | hardware | version | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | NVIDIA L40S (nv2, RunPod) | 0.8.44 | Xq | - | 75.1 | 75.1..75.1 | 1 | - | - | 75.1 | - | - (stored whole) | - | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x220, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| cuml-gpu | NVIDIA L40S (host b6c1fd6fb058) | cuml 26.08.00 | Xq | - | 2.9 | 2.9..2.9 | 1 | 26.021 | - | - | 2.9 | - (stored kernel) | 26.021 (MIXED ours whole / arm kernel) | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x220, rel_diff_vs_sklearn=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: transform(Xq)(Xq)

inference call, cuml-gpu: transform(Xq)(Xq)

### maxabs-scaler / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/recovered-inference/algos.maxabs-scaler.taxi.rows=full/board.json (nv2/paired-3829de1bb56115682e661b9d40ac146e2c633d98d71b5e9b9e34e340edd40a6a)`, ran on b6c1fd6fb058

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 2.7 | 2.7..2.7 | 1 | - | - | 2.7 | 0.7 | 2.00 (upload_ms_separate) | - | - | 266.5 | 682.0 | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x11, rel_diff_vs_sklearn=0.000000 | - | main board, one scored run | wheel | ok (main@fcdfa0fe2 nv2/paired-3829de1bb56115682e661b9d40ac146e2c633d98d71b5e9b9e34e340edd40a6a 2026-10-14; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host b6c1fd6fb058) | cuml 26.08.00 | opponent | 3.4 | 3.4..3.4 | 1 | 0.790 | - | 61.7 | 3.4 | 58.37 (upload_ms_untimed) | 0.043 (whole/whole) | - | 900.7 | 488.0 | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x11, rel_diff_vs_sklearn=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), MaxAbsScaler (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| copy | true | true |
| seed | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | hardware | version | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | NVIDIA L40S (nv2, RunPod) | 0.8.44 | Xq | - | 4.2 | 4.2..4.2 | 1 | - | - | 4.2 | - | - (stored whole) | - | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x11, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| cuml-gpu | NVIDIA L40S (host b6c1fd6fb058) | cuml 26.08.00 | Xq | - | 0.8 | 0.8..0.8 | 1 | 5.018 | - | - | 0.8 | - (stored kernel) | 5.018 (MIXED ours whole / arm kernel) | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x11, rel_diff_vs_sklearn=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: transform(Xq)(Xq)

inference call, cuml-gpu: transform(Xq)(Xq)

### maxpool1d / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/nvidia/campaign/measured/algos.maxpool1d.synthetic.rows=full/board.json (nv2/paired-62442a7ef7fa417de1f2a6d1cf3237134573b6038789e9dc55e483a9538a76d9)`, ran on d2fe567d3b88

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 1.5 ms) = 0.832; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 1.3 ms) = 0.959. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 1.2 | 1.2..1.2 | 1 | - | - | - | 1.2 | - (stored kernel) | - | - | 1694.3 | 938.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | main board, one scored run | wheel | ok (main@79fcd24a5 nv2/paired-62442a7ef7fa417de1f2a6d1cf3237134573b6038789e9dc55e483a9538a76d9 2026-10-14; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 1.3 | 1.3..1.3 | 1 | 0.951 | - | - | 1.3 | - (stored kernel) | 0.951 (kernel/kernel) | - | 836.2 | 288.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 1.3 | 1.3..1.3 | 1 | 0.959 | - | - | 1.3 | - (stored kernel) | 0.959 (kernel/kernel) | - | 1076.4 | 232.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 1.2 | 1.2..1.2 | 1 | 0.993 | - | - | 1.2 | - (stored kernel) | 0.993 (kernel/kernel) | - | 838.3 | 288.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 1.6 | 1.6..1.6 | 1 | 0.746 | - | - | 1.6 | - (stored kernel) | 0.746 (kernel/kernel) | - | 1026.9 | 232.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 1.5 | 1.5..1.5 | 1 | 0.832 | - | - | 1.5 | - (stored kernel) | 0.832 (kernel/kernel) | - | 839.6 | 288.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 8.4 | 8.4..8.4 | 1 | 0.145 | - | - | 8.4 | - (stored kernel) | 0.145 (kernel/kernel) | - | 1021.4 | 232.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'ceil_mode': False, 'dilation': 1, 'kernel_size': 2, 'padding': 0, 'stride': 2}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-compile-tf32 | torch-eager-bf16 | torch-eager-fp32 | torch-eager-tf32 |
|---|---||---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| ceil_mode | false | false | false | false | false | false | false |
| dilation | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| input_arrays | [{"dtype": "float32", "name": "x", "sha256": "128bb6c8835123e5f00ffe3700a26223ba8b37f7280afaec092571b2861be93c", "shape": [64, 64, 4096]}] | [{"dtype": "float32", "name": "x", "sha256": "128bb6c8835123e5f00ffe3700a26223ba8b37f7280afaec092571b2861be93c", "shape": [64, 64, 4096]}] | [{"dtype": "float32", "name": "x", "sha256": "128bb6c8835123e5f00ffe3700a26223ba8b37f7280afaec092571b2861be93c", "shape": [64, 64, 4096]}] | [{"dtype": "float32", "name": "x", "sha256": "128bb6c8835123e5f00ffe3700a26223ba8b37f7280afaec092571b2861be93c", "shape": [64, 64, 4096]}] | [{"dtype": "float32", "name": "x", "sha256": "128bb6c8835123e5f00ffe3700a26223ba8b37f7280afaec092571b2861be93c", "shape": [64, 64, 4096]}] | [{"dtype": "float32", "name": "x", "sha256": "128bb6c8835123e5f00ffe3700a26223ba8b37f7280afaec092571b2861be93c", "shape": [64, 64, 4096]}] | [{"dtype": "float32", "name": "x", "sha256": "128bb6c8835123e5f00ffe3700a26223ba8b37f7280afaec092571b2861be93c", "shape": [64, 64, 4096]}] |
| kernel_size | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| padding | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 | 7 | 7 |
| stride | 2 | 2 | 2 | 2 | 2 | 2 | 2 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | hardware | version | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | NVIDIA L40S (nv2, RunPod) | 0.8.44 | Xq | - | 0.2 | 0.2..0.2 | 1 | - | - | - | 0.2 | - (stored kernel) | - | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.3 | 0.3..0.3 | 1 | 0.928 | - | - | 0.3 | - (stored kernel) | 0.928 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.2 | 0.2..0.2 | 1 | 1.060 | - | - | 0.2 | - (stored kernel) | 1.060 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-tf32 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.3 | 0.3..0.3 | 1 | 0.919 | - | - | 0.3 | - (stored kernel) | 0.919 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-tf32 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.3 | 0.3..0.3 | 1 | 0.945 | - | - | 0.3 | - (stored kernel) | 0.945 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.3 | 0.3..0.3 | 1 | 0.883 | - | - | 0.3 | - (stored kernel) | 0.883 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.3 | 0.3..0.3 | 1 | 0.818 | - | - | 0.3 | - (stored kernel) | 0.818 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-tf32: forward(x) (no autograd)(Xq)

inference call, torch-compile-tf32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### maxpool2d / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/nvidia/campaign/measured/algos.maxpool2d.synthetic.rows=full/board.json (nv2/paired-94acc187bf9b7e7439564948a2c805e9937ea95813b9d7c10ca42035fd37ec69)`, ran on d2fe567d3b88

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 1.9 ms) = 0.637; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 1.7 ms) = 0.695. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 1.2 | 1.2..1.2 | 1 | - | - | - | 1.2 | - (stored kernel) | - | - | 1894.2 | 1194.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | main board, one scored run | wheel | ok (main@79fcd24a5 nv2/paired-94acc187bf9b7e7439564948a2c805e9937ea95813b9d7c10ca42035fd37ec69 2026-10-14; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 1.7 | 1.7..1.7 | 1 | 0.695 | - | - | 1.7 | - (stored kernel) | 0.695 (kernel/kernel) | - | 1019.5 | 639.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 2.0 | 2.0..2.0 | 1 | 0.618 | - | - | 2.0 | - (stored kernel) | 0.618 (kernel/kernel) | - | 1520.8 | 552.5 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 2.0 | 2.0..2.0 | 1 | 0.612 | - | - | 2.0 | - (stored kernel) | 0.612 (kernel/kernel) | - | 1020.7 | 639.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 4.2 | 4.2..4.2 | 1 | 0.287 | - | - | 4.2 | - (stored kernel) | 0.287 (kernel/kernel) | - | 1212.4 | 553.5 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 1.9 | 1.9..1.9 | 1 | 0.637 | - | - | 1.9 | - (stored kernel) | 0.637 (kernel/kernel) | - | 1018.1 | 639.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 2.5 | 2.5..2.5 | 1 | 0.477 | - | - | 2.5 | - (stored kernel) | 0.477 (kernel/kernel) | - | 1204.8 | 553.5 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'ceil_mode': False, 'dilation': 1, 'kernel_size': 2, 'padding': 0, 'stride': 2}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-compile-tf32 | torch-eager-bf16 | torch-eager-fp32 | torch-eager-tf32 |
|---|---||---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| ceil_mode | false | false | false | false | false | false | false |
| dilation | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| input_arrays | [{"dtype": "float32", "name": "x", "sha256": "2df96f3879e0a6060dab5f7b7561b7291c5a106d1a8d4e116fc5e44e3023f819", "shape": [64, 64, 112, 112]}] | [{"dtype": "float32", "name": "x", "sha256": "2df96f3879e0a6060dab5f7b7561b7291c5a106d1a8d4e116fc5e44e3023f819", "shape": [64, 64, 112, 112]}] | [{"dtype": "float32", "name": "x", "sha256": "2df96f3879e0a6060dab5f7b7561b7291c5a106d1a8d4e116fc5e44e3023f819", "shape": [64, 64, 112, 112]}] | [{"dtype": "float32", "name": "x", "sha256": "2df96f3879e0a6060dab5f7b7561b7291c5a106d1a8d4e116fc5e44e3023f819", "shape": [64, 64, 112, 112]}] | [{"dtype": "float32", "name": "x", "sha256": "2df96f3879e0a6060dab5f7b7561b7291c5a106d1a8d4e116fc5e44e3023f819", "shape": [64, 64, 112, 112]}] | [{"dtype": "float32", "name": "x", "sha256": "2df96f3879e0a6060dab5f7b7561b7291c5a106d1a8d4e116fc5e44e3023f819", "shape": [64, 64, 112, 112]}] | [{"dtype": "float32", "name": "x", "sha256": "2df96f3879e0a6060dab5f7b7561b7291c5a106d1a8d4e116fc5e44e3023f819", "shape": [64, 64, 112, 112]}] |
| kernel_size | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| padding | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 | 7 | 7 |
| stride | 2 | 2 | 2 | 2 | 2 | 2 | 2 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | hardware | version | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | NVIDIA L40S (nv2, RunPod) | 0.8.44 | Xq | - | 0.5 | 0.5..0.5 | 1 | - | - | - | 0.5 | - (stored kernel) | - | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-fp32 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.6 | 0.6..0.6 | 1 | 0.905 | - | - | 0.6 | - (stored kernel) | 0.905 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.5 | 0.5..0.5 | 1 | 1.004 | - | - | 0.5 | - (stored kernel) | 1.004 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-tf32 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.6 | 0.6..0.6 | 1 | 0.894 | - | - | 0.6 | - (stored kernel) | 0.894 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-tf32 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.5 | 0.5..0.5 | 1 | 1.007 | - | - | 0.5 | - (stored kernel) | 1.007 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.6 | 0.6..0.6 | 1 | 0.883 | - | - | 0.6 | - (stored kernel) | 0.883 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | Xq | - | 0.5 | 0.5..0.5 | 1 | 0.991 | - | - | 0.5 | - (stored kernel) | 0.991 (kernel/kernel) | - | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: forward(x) (no autograd)(Xq)

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-tf32: forward(x) (no autograd)(Xq)

inference call, torch-compile-tf32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### min-cov-det / istella (rows full, shape X 100000x220; Xq 100000x220; y 100000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/r43/ingest/nv-results.txt (nv/n0003)`, ran on NVIDIA L40S (nv, RunPod) job n0003

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.42 (source build, main@4c75b4144) | identical | 168823.2 | 168823.2..168823.2 | 1 | - | - | 168823.2 | - | - (stored whole) | - | - | - | - | n_features=220, rel_diff_vs_sklearn=0.239361 | - | main board, one scored run | - | ok (main@4c75b4144 nv/n0003 2026-10-13; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(timeout: null) (copied from release-board-resume-r2; measured this run) |

settings: {'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### min-cov-det / taxi (rows full, shape X 100000x11; Xq 100000x11; y 100000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/r43/ingest/nv-results.txt (nv/n0003)`, ran on NVIDIA L40S (nv, RunPod) job n0003

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.42 (source build, main@4c75b4144) | identical | 3420.2 | 3420.2..3420.2 | 1 | - | - | 3420.2 | - | - (stored whole) | - | - | - | - | n_features=11, rel_diff_vs_sklearn=0.238565 | - | main board, one scored run | - | ok (main@4c75b4144 nv/n0003 2026-10-13; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 3011.1 | 3011.1..3011.1 | 1 | 1.136 | - | 3011.1 | 3011.1 | 0.00 (cpu-arm) | 1.136 (whole/whole) | - | 354.8 | - | n_features=11 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### minibatch-kmeans / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0106)`, ran on NVIDIA L40S (nv2, RunPod) job v0106

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 518.6 | 518.6..518.6 | 1 | - | - | 518.6 | - | - (stored whole) | - | - | - | - | n_clusters=8, silhouette=0.116696 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0106 2026-10-11; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 242.5 | 242.5..242.5 | 1 | 2.138 | - | 242.5 | 242.5 | 0.00 (cpu-arm) | 2.138 (whole/whole) | - | 1178.0 | - | ari_vs_ours=0.622446, n_clusters=8, silhouette=0.111849 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'batch_size': 4096, 'init': 'k-means++', 'max_iter': 100, 'max_no_improvement': 10, 'n_clusters': 8, 'n_init': 1, 'random_state': 7, 'reassignment_ratio': 0.01, 'tol': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### minibatch-kmeans / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0106)`, ran on NVIDIA L40S (nv2, RunPod) job v0106

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 48.7 | 48.7..48.7 | 1 | - | - | 48.7 | - | - (stored whole) | - | - | - | - | n_clusters=8, silhouette=0.138060 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0106 2026-10-11; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 50.8 | 50.8..50.8 | 1 | 0.958 | - | 50.8 | 50.8 | 0.00 (cpu-arm) | 0.958 (whole/whole) | - | 276.0 | - | ari_vs_ours=0.525151, n_clusters=8, silhouette=0.165473 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'batch_size': 4096, 'init': 'k-means++', 'max_iter': 100, 'max_no_improvement': 10, 'n_clusters': 8, 'n_init': 1, 'random_state': 7, 'reassignment_ratio': 0.01, 'tol': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### minmax-scaler / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/recovered-inference/algos.minmax-scaler.istella.rows=full/board.json (nv2/paired-23044e523bd487b9a527ef2bd29e53b8aca717a74b50d075ffc7bbd453801864)`, ran on b6c1fd6fb058

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 40.9 | 40.9..40.9 | 1 | - | - | 40.9 | 4.5 | 36.42 (upload_ms_separate) | - | - | 2932.6 | 1450.0 | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x220, rel_diff_vs_sklearn=0.000000 | - | main board, one scored run | wheel | ok (main@fcdfa0fe2 nv2/paired-23044e523bd487b9a527ef2bd29e53b8aca717a74b50d075ffc7bbd453801864 2026-10-14; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host b6c1fd6fb058) | cuml 26.08.00 | opponent | 40.4 | 40.4..40.4 | 1 | 1.012 | - | 718.7 | 40.4 | 678.23 (upload_ms_untimed) | 0.057 (whole/whole) | - | 3227.2 | 1442.0 | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x220, rel_diff_vs_sklearn=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'clip': False, 'feature_range': [0, 1]}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params + fixed unclipped MinMaxScaler behavior) | mojolearn (get_params) |
| clip | false | false |
| copy | true | true |
| feature_range | [0, 1] | [0, 1] |
| seed | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | hardware | version | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | NVIDIA L40S (nv2, RunPod) | 0.8.44 | Xq | - | 57.2 | 57.2..57.2 | 1 | - | - | 57.2 | - | - (stored whole) | - | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x220, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| cuml-gpu | NVIDIA L40S (host b6c1fd6fb058) | cuml 26.08.00 | Xq | - | 3.1 | 3.1..3.1 | 1 | 18.673 | - | - | 3.1 | - (stored kernel) | 18.673 (MIXED ours whole / arm kernel) | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x220, rel_diff_vs_sklearn=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: transform(Xq)(Xq)

inference call, cuml-gpu: transform(Xq)(Xq)

### minmax-scaler / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/recovered-inference/algos.minmax-scaler.taxi.rows=full/board.json (nv2/paired-8266692af051cb856d24a4e8a4e1132b80272a77ec297d90e370bdcb0297ec04)`, ran on b6c1fd6fb058

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 3.3 | 3.3..3.3 | 1 | - | - | 3.3 | 0.6 | 2.69 (upload_ms_separate) | - | - | 1577.9 | 682.0 | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x11, rel_diff_vs_sklearn=0.000000 | - | main board, one scored run | wheel | ok (main@fcdfa0fe2 nv2/paired-8266692af051cb856d24a4e8a4e1132b80272a77ec297d90e370bdcb0297ec04 2026-10-14; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host b6c1fd6fb058) | cuml 26.08.00 | opponent | 5.4 | 5.4..5.4 | 1 | 0.604 | - | 58.7 | 5.4 | 53.32 (upload_ms_untimed) | 0.055 (whole/whole) | - | 899.3 | 488.0 | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x11, rel_diff_vs_sklearn=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'clip': False, 'feature_range': [0, 1]}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params + fixed unclipped MinMaxScaler behavior) | mojolearn (get_params) |
| clip | false | false |
| copy | true | true |
| feature_range | [0, 1] | [0, 1] |
| seed | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | hardware | version | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | NVIDIA L40S (nv2, RunPod) | 0.8.44 | Xq | - | 3.1 | 3.1..3.1 | 1 | - | - | 3.1 | - | - (stored whole) | - | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x11, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| cuml-gpu | NVIDIA L40S (host b6c1fd6fb058) | cuml 26.08.00 | Xq | - | 0.7 | 0.7..0.7 | 1 | 4.268 | - | - | 0.7 | - (stored kernel) | 4.268 (MIXED ours whole / arm kernel) | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x11, rel_diff_vs_sklearn=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: transform(Xq)(Xq)

inference call, cuml-gpu: transform(Xq)(Xq)

### moe / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/nv_ingest/results_nvc3.txt (nv/n0006)`, ran on NVIDIA L40S (nv, RunPod) job n0006

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 4.3 ms) = 3.330; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 11.9 ms) = 1.213. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.42 (source build, main@8899093fc) | identical | 14.5 | 14.5..14.5 | 1 | - | - | 14.5 | - | - (stored whole) | - | - | - | - | relative_error_vs_own_host=0.000000 | - | main board, one scored run | - | ok (main@8899093fc nv/n0006 2026-10-13; identity vs the amd columns: n/a) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 12.0 | 12.0..12.0 | 1 | 1.202 | - | - | 12.0 | - (stored kernel) | 1.202 (MIXED ours whole / arm kernel) | - | 1008.6 | 469.1 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 11.9 | 11.9..11.9 | 1 | 1.213 | - | - | 11.9 | - (stored kernel) | 1.213 (MIXED ours whole / arm kernel) | - | 1273.6 | 446.7 | max_rel_diff_vs_torch_eager_fp32=0.026193, rel_fro_vs_torch_eager_fp32=1.548e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 9.8 | 9.8..9.8 | 1 | 1.471 | - | - | 9.8 | - (stored kernel) | 1.471 (MIXED ours whole / arm kernel) | - | 1004.4 | 469.1 | max_rel_diff_vs_torch_eager_fp32=1004.691291, rel_fro_vs_torch_eager_fp32=0.017534 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 5.1 | 5.1..5.1 | 1 | 2.841 | - | - | 5.1 | - (stored kernel) | 2.841 (MIXED ours whole / arm kernel) | - | 1220.6 | 478.7 | max_rel_diff_vs_torch_eager_fp32=1004.691291, rel_fro_vs_torch_eager_fp32=0.017534 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 4.3 | 4.3..4.3 | 1 | 3.330 | - | - | 4.3 | - (stored kernel) | 3.330 (MIXED ours whole / arm kernel) | - | 1129.6 | 574.3 | max_rel_diff_vs_torch_eager_fp32=22834.612745, rel_fro_vs_torch_eager_fp32=0.055222 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 4.4 | 4.4..4.4 | 1 | 3.280 | - | - | 4.4 | - (stored kernel) | 3.280 (MIXED ours whole / arm kernel) | - | 1397.4 | 433.6 | max_rel_diff_vs_torch_eager_fp32=22851.987745, rel_fro_vs_torch_eager_fp32=0.055199 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'hidden_size': 1024, 'intermediate_size': 2816, 'norm_topk_prob': True, 'num_experts': 8, 'top_k': 2}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### multilabel-binarizer / istella (rows full, shape X 200000x8; Xq 20000x8; lab 200000; labq 20000; y 200000; yq 20000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/nv_ingest/results_nvc2.txt (nv/n0001)`, ran on NVIDIA L40S (nv, RunPod) job n0001

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.42 (source build, main@9c7ad7413) | identical | 264.2 | 264.2..264.2 | 1 | - | - | 264.2 | - | - (stored whole) | - | - | - | - | max_abs_diff_vs_sklearn=0.000000, output_shape=20000x119, rel_diff_vs_sklearn=0.000000 | - | main board, one scored run | - | ok (main@9c7ad7413 nv/n0001 2026-10-12; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 215.1 | 215.1..215.1 | 1 | 1.228 | - | 215.1 | 215.1 | 0.00 (cpu-arm) | 1.228 (whole/whole) | - | 1023.3 | - | output_shape=20000x119 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### multilabel-binarizer / taxi (rows full, shape X 200000x5; Xq 20000x5; lab 200000; labq 20000; y 200000; yq 20000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/nv_ingest/results_nvc2.txt (nv/n0001)`, ran on NVIDIA L40S (nv, RunPod) job n0001

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.42 (source build, main@9c7ad7413) | identical | 444.7 | 444.7..444.7 | 1 | - | - | 444.7 | - | - (stored whole) | - | - | - | - | max_abs_diff_vs_sklearn=0.000000, output_shape=20000x489, rel_diff_vs_sklearn=0.000000 | - | main board, one scored run | - | ok (main@9c7ad7413 nv/n0001 2026-10-12; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 166.7 | 166.7..166.7 | 1 | 2.668 | - | 166.7 | 166.7 | 0.00 (cpu-arm) | 2.668 (whole/whole) | - | 2808.4 | - | output_shape=20000x489 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### multinomial-nb / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0124)`, ran on NVIDIA L40S (nv2, RunPod) job v0124

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 112.8 | 112.8..112.8 | 1 | - | - | 112.8 | - | - (stored whole) | - | - | - | - | accuracy=0.853620, logloss=3.628565 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0124 2026-10-11; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 56.7 | 56.7..56.7 | 1 | 1.989 | - | 769.6 | 56.7 | 712.86 (upload_ms_untimed) | 0.147 (whole/whole) | - | 3910.6 | 1372.0 | accuracy=0.853620, logloss=3.628599 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 271.1 | 271.1..271.1 | 1 | 0.416 | - | 271.1 | 271.1 | 0.00 (cpu-arm) | 0.416 (whole/whole) | - | 3913.9 | - | accuracy=0.853620, logloss=3.087499 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'fit_prior': True, 'force_alpha': True}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), MultinomialNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### multinomial-nb / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0124)`, ran on NVIDIA L40S (nv2, RunPod) job v0124

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 30.2 | 30.2..30.2 | 1 | - | - | 30.2 | - | - (stored whole) | - | - | - | - | accuracy=0.723160, logloss=0.590725 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0124 2026-10-11; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 72.6 | 72.6..72.6 | 1 | 0.416 | - | 127.1 | 72.6 | 54.55 (upload_ms_untimed) | 0.238 (whole/whole) | - | 1077.0 | 496.0 | accuracy=0.723160, logloss=0.590750 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 69.9 | 69.9..69.9 | 1 | 0.432 | - | 69.9 | 69.9 | 0.00 (cpu-arm) | 0.432 (whole/whole) | - | 436.4 | - | accuracy=0.723160, logloss=0.590725 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'fit_prior': True, 'force_alpha': True}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), MultinomialNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### multinomial-nb / text (rows full, shape X 86626x4096; Xq 9626x4096; y 86626; yq 9626)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0532)`, ran on NVIDIA L40S (nv, RunPod) job n0532

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@42d1e42c6) | identical | 56.2 | 56.2..56.2 | 1 | - | - | 56.2 | - | - (stored whole) | - | - | - | - | accuracy=0.983067, logloss=0.559529 | - | main board, one scored run | - | ok (main@42d1e42c6 nv/n0532 2026-10-08; identity vs amd-mi325x: MATCH) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 50.0 | 50.0..50.0 | 1 | 1.123 | - | 1390.0 | 50.0 | 1339.98 (upload_ms_untimed) | 0.040 (whole/whole) | - | 4711.5 | 1946.0 | accuracy=0.983067, logloss=0.559524 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 454.4 | 454.4..454.4 | 1 | 0.124 | - | 454.4 | 454.4 | 0.00 (cpu-arm) | 0.124 (whole/whole) | - | 4414.9 | - | accuracy=0.983067, logloss=0.557319 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'fit_prior': True, 'force_alpha': True}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), MultinomialNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### nadam / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/nvidia/campaign/measured/algos.nadam.synthetic.rows=full/board.json (nv2/paired-634a0f039e431c86e628269246b6a949d47b330a596c5cb4cb5da10b834b004e)`, ran on d2fe567d3b88

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 21.1 ms) = 0.513. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 10.9 | 10.9..10.9 | 1 | - | - | - | 10.9 | - (stored kernel) | - | - | 1023.9 | 1450.0 | optimizer_update_error_vs_torch_fp32=1.131e-05, rel_fro_vs_torch_eager_fp32=2.945e-08 | - | main board, one scored run | wheel | ok (main@79fcd24a5 nv2/paired-634a0f039e431c86e628269246b6a949d47b330a596c5cb4cb5da10b834b004e 2026-10-14; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 21.1 | 21.1..21.1 | 1 | 0.513 | - | - | 21.1 | - (stored kernel) | 0.513 (kernel/kernel) | - | 1034.9 | 896.0 | optimizer_update_error_vs_torch_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 109.9 | 109.9..109.9 | 1 | 0.099 | - | - | 109.9 | - (stored kernel) | 0.099 (kernel/kernel) | - | 1478.2 | 960.0 | optimizer_update_error_vs_torch_fp32=5.752e-05, rel_fro_vs_torch_eager_fp32=1.498e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

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

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0102)`, ran on NVIDIA L40S (nv2, RunPod) job v0102

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 46.0 | 46.0..46.0 | 1 | - | - | 46.0 | - | - (stored whole) | - | - | - | - | accuracy=0.852610, logloss=4.299485 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0102 2026-10-11; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 1024.1 | 1024.1..1024.1 | 1 | 0.045 | - | 1024.1 | 1024.1 | 0.00 (cpu-arm) | 0.045 (whole/whole) | - | 4523.4 | - | accuracy=0.852610, logloss=4.117692 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'metric': 'euclidean', 'priors': 'uniform'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### nearest-centroid / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0102)`, ran on NVIDIA L40S (nv2, RunPod) job v0102

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 9.2 | 9.2..9.2 | 1 | - | - | 9.2 | - | - (stored whole) | - | - | - | - | accuracy=0.666700, logloss=0.781905 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0102 2026-10-11; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 260.8 | 260.8..260.8 | 1 | 0.035 | - | 260.8 | 260.8 | 0.00 (cpu-arm) | 0.035 (whole/whole) | - | 457.8 | - | accuracy=0.666750, logloss=0.781690 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'metric': 'euclidean', 'priors': 'uniform'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### normalizer / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/nvidia-wave2/campaign/measured/algos.normalizer.istella.rows=full/board.json (nv2/paired-48558cd4f0e060d7976bc41f2685104e65090151ab2e556e6557d018a92134c9)`, ran on d2fe567d3b88

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 0.1 | 0.1..0.1 | 1 | - | - | 0.1 | - | - (stored whole) | - | - | 1624.0 | 1706.0 | max_abs_diff_vs_sklearn=3.576e-07, output_shape=100000x220, rel_diff_vs_sklearn=5.647e-08 | - | main board, one scored run | wheel | ok (main@aa8e9e72b nv2/paired-48558cd4f0e060d7976bc41f2685104e65090151ab2e556e6557d018a92134c9 2026-10-14; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host d2fe567d3b88) | cuml 26.08.00 | opponent | 2.2 | 2.2..2.2 | 1 | 0.048 | - | 676.9 | 2.2 | 674.65 (upload_ms_untimed) | 0.0001585 (whole/whole) | - | 3301.3 | 1450.0 | max_abs_diff_vs_sklearn=1.788e-07, output_shape=100000x220, rel_diff_vs_sklearn=4.428e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'norm': 'l2'}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), Normalizer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params + constructor attributes norm,copy) | mojolearn (get_params) |
| copy | true | true |
| norm | "l2" | "l2" |
| seed | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | hardware | version | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | NVIDIA L40S (nv2, RunPod) | 0.8.44 | Xq | - | 73.0 | 73.0..73.0 | 1 | - | - | 73.0 | - | - (stored whole) | - | - | max_abs_diff_vs_sklearn=3.576e-07, output_shape=100000x220, rel_diff_vs_sklearn=5.647e-08 | - | LIKE-FOR-LIKE-SPAN | ok |
| cuml-gpu | NVIDIA L40S (host d2fe567d3b88) | cuml 26.08.00 | Xq | - | 1.7 | 1.7..1.7 | 1 | 43.962 | - | - | 1.7 | - (stored kernel) | 43.962 (MIXED ours whole / arm kernel) | - | max_abs_diff_vs_sklearn=1.788e-07, output_shape=100000x220, rel_diff_vs_sklearn=4.428e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: transform(Xq)(Xq)

inference call, cuml-gpu: transform(Xq)(Xq)

### normalizer / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/nvidia-wave2/campaign/measured/algos.normalizer.taxi.rows=full/board.json (nv2/paired-c9556c3f252aa1194a64c96376c0fb74bd48032229cbb52a97a38711cf64b01e)`, ran on d2fe567d3b88

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 0.2 | 0.2..0.2 | 1 | - | - | 0.2 | - | - (stored whole) | - | - | 267.1 | 682.0 | max_abs_diff_vs_sklearn=1.192e-07, output_shape=100000x11, rel_diff_vs_sklearn=3.043e-08 | - | main board, one scored run | wheel | ok (main@aa8e9e72b nv2/paired-c9556c3f252aa1194a64c96376c0fb74bd48032229cbb52a97a38711cf64b01e 2026-10-14; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host d2fe567d3b88) | cuml 26.08.00 | opponent | 0.6 | 0.6..0.6 | 1 | 0.307 | - | 55.8 | 0.6 | 55.20 (upload_ms_untimed) | 0.003 (whole/whole) | - | 1053.3 | 496.0 | max_abs_diff_vs_sklearn=1.788e-07, output_shape=100000x11, rel_diff_vs_sklearn=2.246e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'norm': 'l2'}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), Normalizer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params + constructor attributes norm,copy) | mojolearn (get_params) |
| copy | true | true |
| norm | "l2" | "l2" |
| seed | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | hardware | version | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | NVIDIA L40S (nv2, RunPod) | 0.8.44 | Xq | - | 4.5 | 4.5..4.5 | 1 | - | - | 4.5 | - | - (stored whole) | - | - | max_abs_diff_vs_sklearn=1.192e-07, output_shape=100000x11, rel_diff_vs_sklearn=3.043e-08 | - | LIKE-FOR-LIKE-SPAN | ok |
| cuml-gpu | NVIDIA L40S (host d2fe567d3b88) | cuml 26.08.00 | Xq | - | 4.2 | 4.2..4.2 | 1 | 1.090 | - | - | 4.2 | - (stored kernel) | 1.090 (MIXED ours whole / arm kernel) | - | max_abs_diff_vs_sklearn=1.788e-07, output_shape=100000x11, rel_diff_vs_sklearn=2.246e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: transform(Xq)(Xq)

inference call, cuml-gpu: transform(Xq)(Xq)

### onehot / istella (rows full, shape X 1000000x8; Xq 100000x8; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/nv_ingest/results_nvc2.txt (nv/n0001)`, ran on NVIDIA L40S (nv, RunPod) job n0001

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.42 (source build, main@9c7ad7413) | identical | 34.5 | 34.5..34.5 | 1 | - | - | 34.5 | - | - (stored whole) | - | - | - | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x119, rel_diff_vs_sklearn=0.000000 | - | main board, one scored run | - | ok (main@9c7ad7413 nv/n0001 2026-10-12; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 52.0 | 52.0..52.0 | 1 | 0.664 | - | 82.4 | 52.0 | 30.41 (upload_ms_untimed) | 0.419 (whole/whole) | - | 1251.6 | 524.0 | output_shape=100000x119 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 93.8 | 93.8..93.8 | 1 | 0.368 | - | 93.8 | 93.8 | 0.00 (cpu-arm) | 0.368 (whole/whole) | - | 531.2 | - | output_shape=100000x119 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'categories': 'auto', 'handle_unknown': 'ignore', 'sparse_output': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), OneHotEncoder (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### onehot / taxi (rows full, shape X 1000000x5; Xq 100000x5; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/nv_ingest/results_nvc2.txt (nv/n0001)`, ran on NVIDIA L40S (nv, RunPod) job n0001

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.42 (source build, main@9c7ad7413) | identical | 23.6 | 23.6..23.6 | 1 | - | - | 23.6 | - | - (stored whole) | - | - | - | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x508, rel_diff_vs_sklearn=0.000000 | - | main board, one scored run | - | ok (main@9c7ad7413 nv/n0001 2026-10-12; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 173.9 | 173.9..173.9 | 1 | 0.136 | - | 202.4 | 173.9 | 28.46 (upload_ms_untimed) | 0.116 (whole/whole) | - | 1557.8 | 656.0 | output_shape=100000x508 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 48.4 | 48.4..48.4 | 1 | 0.487 | - | 48.4 | 48.4 | 0.00 (cpu-arm) | 0.487 (whole/whole) | - | 1399.8 | - | output_shape=100000x508 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'categories': 'auto', 'handle_unknown': 'ignore', 'sparse_output': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), OneHotEncoder (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### optimized-theta / synthetic (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0097)`, ran on NVIDIA L40S (nv2, RunPod) job v0097

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 175.1 | 175.1..175.1 | 1 | - | - | 175.1 | - | - (stored whole) | - | - | - | - | forecast_rmse=1.438855 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0097 2026-10-11; identity vs the amd columns: n/a) |
| statsforecast-cpu | statsforecast | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | statsforecast 2.1.1 | opponent | 750.7 | 750.7..750.7 | 1 | 0.233 | - | 750.7 | 750.7 | 0.00 (cpu-arm) | 0.233 (whole/whole) | - | 467.7 | - | forecast_rmse=1.437815 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, statsforecast-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'decomposition_type': 'multiplicative', 'season_length': 24}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### optimized-theta / taxi-hourly (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0097)`, ran on NVIDIA L40S (nv2, RunPod) job v0097

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 309.6 | 309.6..309.6 | 1 | - | - | 309.6 | - | - (stored whole) | - | - | - | - | forecast_rmse=49.150860 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0097 2026-10-11; identity vs the amd columns: n/a) |
| statsforecast-cpu | statsforecast | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | statsforecast 2.1.1 | opponent | 1772.3 | 1772.3..1772.3 | 1 | 0.175 | - | 1772.3 | 1772.3 | 0.00 (cpu-arm) | 0.175 (whole/whole) | - | 467.1 | - | forecast_rmse=49.356608 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, statsforecast-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'decomposition_type': 'multiplicative', 'season_length': 24}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ordinal / istella (rows full, shape X 1000000x8; Xq 100000x8; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/nv_ingest/results_nvc2.txt (nv/n0001)`, ran on NVIDIA L40S (nv, RunPod) job n0001

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.42 (source build, main@9c7ad7413) | identical | 33.2 | 33.2..33.2 | 1 | - | - | 33.2 | - | - (stored whole) | - | - | - | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x8, rel_diff_vs_sklearn=0.000000 | - | main board, one scored run | - | ok (main@9c7ad7413 nv/n0001 2026-10-12; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 100.3 | 100.3..100.3 | 1 | 0.331 | - | 100.3 | 100.3 | 0.00 (cpu-arm) | 0.331 (whole/whole) | - | 259.8 | - | output_shape=100000x8 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'categories': 'auto', 'handle_unknown': 'use_encoded_value', 'unknown_value': -1}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), OrdinalEncoder (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ordinal / taxi (rows full, shape X 1000000x5; Xq 100000x5; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/nv_ingest/results_nvc2.txt (nv/n0001)`, ran on NVIDIA L40S (nv, RunPod) job n0001

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.42 (source build, main@9c7ad7413) | identical | 19.7 | 19.7..19.7 | 1 | - | - | 19.7 | - | - (stored whole) | - | - | - | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x5, rel_diff_vs_sklearn=0.000000 | - | main board, one scored run | - | ok (main@9c7ad7413 nv/n0001 2026-10-12; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 48.5 | 48.5..48.5 | 1 | 0.405 | - | 48.5 | 48.5 | 0.00 (cpu-arm) | 0.405 (whole/whole) | - | 240.9 | - | output_shape=100000x5 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'categories': 'auto', 'handle_unknown': 'use_encoded_value', 'unknown_value': -1}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), OrdinalEncoder (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### pagerank / istella (rows full, shape X 20000x220; indices 304830; indices2 62848; indptr 20001; indptr2 20001; y 20000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0523)`, ran on NVIDIA L40S (nv, RunPod) job n0523

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@4df610f3b) | identical | 1.4 | 1.4..1.4 | 1 | - | - | 1.4 | - | - (stored whole) | - | - | - | - | sum=1.000000 | - | main board, one scored run | - | ok (main@4df610f3b nv/n0523 2026-10-08; identity vs amd-mi325x: MATCH) |
| cugraph-gpu | cugraph | gpu | NVIDIA L40S (host cc560ebdaf91) | cugraph 26.08.00 | opponent | 5.0 | 5.0..5.0 | 1 | 0.284 | - | 5.0 | 5.0 | 0.01 (upload_ms_untimed) | 0.283 (whole/whole) | - | 1009.8 | 440.0 | sum=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| networkx-cpu | networkx | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | networkx 3.6.1 | opponent | 274.0 | 274.0..274.0 | 1 | 0.005 | - | 274.0 | 274.0 | 0.00 (cpu-arm) | 0.005 (whole/whole) | - | 211.2 | - | sum=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cugraph-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, networkx-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 0.85, 'max_iter': 100, 'tol': 1e-06}. Rows: None. Timed: None.

mismatch: ours takes a dense adjacency matrix (its class's contract), built from the CSR graph before the clock; networkx and cuGraph take the graph itself

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### pagerank / taxi (rows full, shape X 20000x11; indices 253708; indices2 54528; indptr 20001; indptr2 20001; y 20000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0523)`, ran on NVIDIA L40S (nv, RunPod) job n0523

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@4df610f3b) | identical | 1.3 | 1.3..1.3 | 1 | - | - | 1.3 | - | - (stored whole) | - | - | - | - | sum=1.000000 | - | main board, one scored run | - | ok (main@4df610f3b nv/n0523 2026-10-08; identity vs amd-mi325x: MATCH) |
| cugraph-gpu | cugraph | gpu | NVIDIA L40S (host 24a11adce16e) | cugraph 26.08.00 | opponent | 5.0 | 5.0..5.0 | 1 | 0.257 | - | 5.0 | 5.0 | 0.02 (upload_ms_untimed) | 0.256 (whole/whole) | - | 1023.2 | 438.0 | sum=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| networkx-cpu | networkx | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | networkx 3.6.1 | opponent | 189.4 | 189.4..189.4 | 1 | 0.007 | - | 189.4 | 189.4 | 0.00 (cpu-arm) | 0.007 (whole/whole) | - | 175.9 | - | sum=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cugraph-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, networkx-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 0.85, 'max_iter': 100, 'tol': 1e-06}. Rows: None. Timed: None.

mismatch: ours takes a dense adjacency matrix (its class's contract), built from the CSR graph before the clock; networkx and cuGraph take the graph itself

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### permutation-shap / istella (rows full, shape X 100000x220; Xq 100x220; y 100000; yq 100)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0108)`, ran on NVIDIA L40S (nv2, RunPod) job v0108

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 3872.5 | 3872.5..3872.5 | 1 | - | - | 3872.5 | - | - (stored whole) | - | - | - | - | rel_error_vs_exact=6.68e-08 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0108 2026-10-11; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 2320.3 | 2320.3..2320.3 | 1 | 1.669 | - | 2324.4 | 2320.3 | 4.11 (upload_ms_untimed) | 1.666 (whole/whole) | - | 1915.3 | 438.0 | rel_error_vs_exact=1.483e-07 | - | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (copied from opponents-default-20261006; measured this run) |
| shap-cpu | shap | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | shap 0.51.0 | opponent | 90631.6 | 90631.6..90631.6 | 1 | 0.043 | - | 90631.6 | 90631.6 | 0.00 (cpu-arm) | 0.043 (whole/whole) | - | 1725.8 | - | rel_error_vs_exact=3.692e-10 | - | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, shap-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'n_background': 100, 'npermutations': 10}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### permutation-shap / taxi (rows full, shape X 100000x11; Xq 100x11; y 100000; yq 100)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0108)`, ran on NVIDIA L40S (nv2, RunPod) job v0108

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 66.3 | 66.3..66.3 | 1 | - | - | 66.3 | - | - (stored whole) | - | - | - | - | rel_error_vs_exact=1.054e-07 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0108 2026-10-11; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 319.6 | 319.6..319.6 | 1 | 0.207 | - | 323.9 | 319.6 | 4.28 (upload_ms_untimed) | 0.205 (whole/whole) | - | 956.8 | 438.0 | rel_error_vs_exact=1.701e-07 | - | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (copied from opponents-specific-20261006; measured this run) |
| shap-cpu | shap | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | shap 0.51.0 | opponent | 133.5 | 133.5..133.5 | 1 | 0.497 | - | 133.5 | 133.5 | 0.00 (cpu-arm) | 0.497 (whole/whole) | - | 476.7 | - | rel_error_vs_exact=1.144e-15 | - | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, shap-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'n_background': 100, 'npermutations': 10}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### poly-features / istella (rows full, shape X 1000000x16; Xq 100000x16; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/recovered-inference/algos.poly-features.istella.rows=full/board.json (nv2/paired-b0f4afc8403f3cfad35b5e935bda8edb67542cef64b7ef984d53d72d1293e698)`, ran on b6c1fd6fb058

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 0.6 | 0.6..0.6 | 1 | - | - | 0.6 | - | - (stored whole) | - | - | 1533.4 | 682.0 | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x152, rel_diff_vs_sklearn=0.000000 | - | main board, one scored run | wheel | ok (main@fcdfa0fe2 nv2/paired-b0f4afc8403f3cfad35b5e935bda8edb67542cef64b7ef984d53d72d1293e698 2026-10-14; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host b6c1fd6fb058) | cuml 26.08.00 | opponent | 2.7 | 2.7..2.7 | 1 | 0.208 | - | 53.1 | 2.7 | 50.35 (upload_ms_untimed) | 0.011 (whole/whole) | - | 2084.6 | 560.0 | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x152, rel_diff_vs_sklearn=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'degree': 2, 'include_bias': False, 'interaction_only': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), PolynomialFeatures (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| degree | 2 | 2 |
| include_bias | false | false |
| interaction_only | false | false |
| order | "C" | "C" |
| seed | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | hardware | version | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | NVIDIA L40S (nv2, RunPod) | 0.8.44 | Xq | - | 49.9 | 49.9..49.9 | 1 | - | - | 49.9 | - | - (stored whole) | - | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x152, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| cuml-gpu | NVIDIA L40S (host b6c1fd6fb058) | cuml 26.08.00 | Xq | - | 14.8 | 14.8..14.8 | 1 | 3.381 | - | - | 14.8 | - (stored kernel) | 3.381 (MIXED ours whole / arm kernel) | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x152, rel_diff_vs_sklearn=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: transform(Xq)(Xq)

inference call, cuml-gpu: transform(Xq)(Xq)

### poly-features / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/recovered-inference/algos.poly-features.taxi.rows=full/board.json (nv2/paired-01e2a37227a2f2c23e22271e6854da9e56837831ba971ed24ba0632de7d324cf)`, ran on b6c1fd6fb058

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 0.2 | 0.2..0.2 | 1 | - | - | 0.2 | - | - (stored whole) | - | - | 419.2 | 682.0 | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x77, rel_diff_vs_sklearn=0.000000 | - | main board, one scored run | wheel | ok (main@fcdfa0fe2 nv2/paired-01e2a37227a2f2c23e22271e6854da9e56837831ba971ed24ba0632de7d324cf 2026-10-14; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host b6c1fd6fb058) | cuml 26.08.00 | opponent | 1.4 | 1.4..1.4 | 1 | 0.177 | - | 50.6 | 1.4 | 49.16 (upload_ms_untimed) | 0.005 (whole/whole) | - | 1025.4 | 510.0 | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x77, rel_diff_vs_sklearn=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'degree': 2, 'include_bias': False, 'interaction_only': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), PolynomialFeatures (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| degree | 2 | 2 |
| include_bias | false | false |
| interaction_only | false | false |
| order | "C" | "C" |
| seed | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | hardware | version | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | NVIDIA L40S (nv2, RunPod) | 0.8.44 | Xq | - | 24.9 | 24.9..24.9 | 1 | - | - | 24.9 | - | - (stored whole) | - | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x77, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | ok |
| cuml-gpu | NVIDIA L40S (host b6c1fd6fb058) | cuml 26.08.00 | Xq | - | 4.1 | 4.1..4.1 | 1 | 6.104 | - | - | 4.1 | - (stored kernel) | 6.104 (MIXED ours whole / arm kernel) | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x77, rel_diff_vs_sklearn=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: transform(Xq)(Xq)

inference call, cuml-gpu: transform(Xq)(Xq)

### power-transformer / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/recovered-inference/algos.power-transformer.istella.rows=full/board.json (nv2/paired-5b25cf619e0d356e160d67f974b36101886747d77ef7cb63fc60a02a4d30fe06)`, ran on b6c1fd6fb058

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 3354.1 | 3354.1..3354.1 | 1 | - | - | 3354.1 | 3316.8 | 37.37 (upload_ms_separate) | - | - | 3889.1 | 5034.0 | max_abs_diff_vs_sklearn=5.575023, output_shape=100000x220, rel_diff_vs_sklearn=0.105054 | - | main board, one scored run | wheel | ok (main@fcdfa0fe2 nv2/paired-5b25cf619e0d356e160d67f974b36101886747d77ef7cb63fc60a02a4d30fe06 2026-10-14; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host b6c1fd6fb058) | cuml 26.08.00 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "BracketError('The algorithm terminated without finding a valid bracket. Consider trying different initial points.')", "event": "error", "stage": "round 0"}) (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, cuml-gpu: host not sampled; GPU not sampled

settings: {'method': 'yeo-johnson', 'standardize': True}. Rows: None. Timed: None.

mismatch: lambda search: ours golden-section, 48 steps in [-8, 8], float32 (x_prep/transform.mojo PT_ITERS); scikit-learn and cuML scipy.optimize.brent (bracket (-2, 2), tol 1.48e-8); the lambdas and outputs differ (rel 0.1 on Istella)

config: cuML benchmark (RAPIDS), PowerTransformer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| copy | true | true |
| method | "yeo-johnson" | "yeo-johnson" |
| seed | "none (deterministic)" | "none (deterministic)" |
| standardize | true | true |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | hardware | version | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | NVIDIA L40S (nv2, RunPod) | 0.8.44 | Xq | - | 75.9 | 75.9..75.9 | 1 | - | - | 75.9 | - | - (stored whole) | - | - | max_abs_diff_vs_sklearn=5.575023, output_shape=100000x220, rel_diff_vs_sklearn=0.105054 | - | LIKE-FOR-LIKE-SPAN | ok |
| cuml-gpu | NVIDIA L40S (host b6c1fd6fb058) | cuml 26.08.00 | Xq | - | - | - | 0 | - | - | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(error: {"error": "BracketError('The algorithm terminated without finding a valid bracket. Consider trying different initial points.')", "event": "error", "stage": "round 0"}) |

inference call, ours: transform(Xq)(Xq)

inference call, cuml-gpu: transform(Xq)(Xq)

### power-transformer / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/recovered-inference/algos.power-transformer.taxi.rows=full/board.json (nv2/paired-51002b072a16127171b467c5b158217fefaa2cd66105b29f5e728fbf46886ccc)`, ran on b6c1fd6fb058

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 103.4 | 103.4..103.4 | 1 | - | - | 103.4 | 101.5 | 1.85 (upload_ms_separate) | - | - | 297.4 | 1194.0 | max_abs_diff_vs_sklearn=0.064570, output_shape=100000x11, rel_diff_vs_sklearn=0.0008076 | - | main board, one scored run | wheel | ok (main@fcdfa0fe2 nv2/paired-51002b072a16127171b467c5b158217fefaa2cd66105b29f5e728fbf46886ccc 2026-10-14; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host b6c1fd6fb058) | cuml 26.08.00 | opponent | 2808.5 | 2808.5..2808.5 | 1 | 0.037 | - | 2857.8 | 2808.5 | 49.29 (upload_ms_untimed) | 0.036 (whole/whole) | - | 949.5 | 488.0 | max_abs_diff_vs_sklearn=0.063349, output_shape=100000x11, rel_diff_vs_sklearn=0.0003244 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'method': 'yeo-johnson', 'standardize': True}. Rows: None. Timed: None.

mismatch: lambda search: ours golden-section, 48 steps in [-8, 8], float32 (x_prep/transform.mojo PT_ITERS); scikit-learn and cuML scipy.optimize.brent (bracket (-2, 2), tol 1.48e-8); the lambdas and outputs differ (rel 0.1 on Istella)

config: cuML benchmark (RAPIDS), PowerTransformer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| copy | true | true |
| method | "yeo-johnson" | "yeo-johnson" |
| seed | "none (deterministic)" | "none (deterministic)" |
| standardize | true | true |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | hardware | version | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | NVIDIA L40S (nv2, RunPod) | 0.8.44 | Xq | - | 4.4 | 4.4..4.4 | 1 | - | - | 4.4 | - | - (stored whole) | - | - | max_abs_diff_vs_sklearn=0.064570, output_shape=100000x11, rel_diff_vs_sklearn=0.0008076 | - | LIKE-FOR-LIKE-SPAN | ok |
| cuml-gpu | NVIDIA L40S (host b6c1fd6fb058) | cuml 26.08.00 | Xq | - | 12.5 | 12.5..12.5 | 1 | 0.351 | - | - | 12.5 | - (stored kernel) | 0.351 (MIXED ours whole / arm kernel) | - | max_abs_diff_vs_sklearn=0.063349, output_shape=100000x11, rel_diff_vs_sklearn=0.0003244 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: transform(Xq)(Xq)

inference call, cuml-gpu: transform(Xq)(Xq)

### qr / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0088)`, ran on NVIDIA L40S (nv2, RunPod) job v0088

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 880.3 | 880.3..880.3 | 1 | - | - | 880.3 | - | - (stored whole) | - | - | - | - | relative_gram_difference=1.485e-07 | - | main board, one scored run | - | ok (main@a1c96a5ba nv2/v0088 2026-10-11; identity vs the amd columns: n/a) |
| torch-gpu | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 227.1 | 227.1..227.1 | 1 | 3.876 | - | - | 227.1 | - (stored kernel) | 3.876 (MIXED ours whole / arm kernel) | - | 1703.4 | 3008.9 | relative_gram_difference=3.787e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| cupy-gpu | cupy | gpu | NVIDIA L40S (host cc560ebdaf91) | cupy 14.2.0 | opponent | 230.5 | 230.5..230.5 | 1 | 3.818 | - | - | 230.5 | - (stored kernel) | 3.818 (MIXED ours whole / arm kernel) | - | 2486.4 | 5160.0 | relative_gram_difference=3.787e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| numpy-cpu | numpy | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | numpy 2.4.6 | opponent | 17211.6 | 17211.6..17211.6 | 1 | 0.051 | - | 17211.6 | 17211.6 | 0.00 (cpu-arm) | 0.051 (whole/whole) | - | 8528.1 | - | relative_gram_difference=2.459e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, cupy-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'mode': 'reduced'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### qr / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0088)`, ran on NVIDIA L40S (nv2, RunPod) job v0088

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 31.5 | 31.5..31.5 | 1 | - | - | 31.5 | - | - (stored whole) | - | - | - | - | relative_gram_difference=1.714e-07 | - | main board, one scored run | - | ok (main@a1c96a5ba nv2/v0088 2026-10-11; identity vs the amd columns: n/a) |
| torch-gpu | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 7.0 | 7.0..7.0 | 1 | 4.501 | - | - | 7.0 | - (stored kernel) | 4.501 (MIXED ours whole / arm kernel) | - | 831.1 | 168.1 | relative_gram_difference=5.971e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| cupy-gpu | cupy | gpu | NVIDIA L40S (host 24a11adce16e) | cupy 14.2.0 | opponent | 6.8 | 6.8..6.8 | 1 | 4.593 | - | - | 6.8 | - (stored kernel) | 4.593 (MIXED ours whole / arm kernel) | - | 653.7 | 726.0 | relative_gram_difference=5.971e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| numpy-cpu | numpy | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | numpy 2.4.6 | opponent | 444.9 | 444.9..444.9 | 1 | 0.071 | - | 444.9 | 444.9 | 0.00 (cpu-arm) | 0.071 (whole/whole) | - | 475.9 | - | relative_gram_difference=3.024e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, cupy-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'mode': 'reduced'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### quantile-transformer / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/recovered-inference/algos.quantile-transformer.istella.rows=full/board.json (nv2/paired-b24aca1675f65a5b15fba36fd0cc5fddd76658f8fc32cefdb271e1b5ab670e3a)`, ran on b6c1fd6fb058

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 278.4 | 278.4..278.4 | 1 | - | - | 278.4 | 241.7 | 36.66 (upload_ms_separate) | - | - | 1629.4 | 5546.0 | max_abs_diff_vs_sklearn=5.96e-08, output_shape=100000x220, rel_diff_vs_sklearn=2.764e-08 | - | main board, one scored run | wheel | ok (main@fcdfa0fe2 nv2/paired-b24aca1675f65a5b15fba36fd0cc5fddd76658f8fc32cefdb271e1b5ab670e3a 2026-10-14; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host b6c1fd6fb058) | cuml 26.08.00 | opponent | 5967.6 | 5967.6..5967.6 | 1 | 0.047 | - | 6642.3 | 5967.6 | 674.73 (upload_ms_untimed) | 0.042 (whole/whole) | - | 3819.3 | 1444.0 | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x220, rel_diff_vs_sklearn=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'ignore_implicit_zeros': False, 'n_quantiles': 1000, 'output_distribution': 'uniform', 'random_state': 7, 'subsample': 1000000000}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), QuantileTransformer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| copy | true | true |
| ignore_implicit_zeros | false | false |
| n_quantiles | 1000 | 1000 |
| output_distribution | "uniform" | "uniform" |
| seed | 7 | 7 |
| subsample | 1000000000 | 1000000000 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | hardware | version | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | NVIDIA L40S (nv2, RunPod) | 0.8.44 | Xq | - | 81.8 | 81.8..81.8 | 1 | - | - | 81.8 | - | - (stored whole) | - | - | max_abs_diff_vs_sklearn=5.96e-08, output_shape=100000x220, rel_diff_vs_sklearn=2.764e-08 | - | LIKE-FOR-LIKE-SPAN | ok |
| cuml-gpu | NVIDIA L40S (host b6c1fd6fb058) | cuml 26.08.00 | Xq | - | 261.5 | 261.5..261.5 | 1 | 0.313 | - | - | 261.5 | - (stored kernel) | 0.313 (MIXED ours whole / arm kernel) | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x220, rel_diff_vs_sklearn=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: transform(Xq)(Xq)

inference call, cuml-gpu: transform(Xq)(Xq)

### quantile-transformer / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/recovered-inference/algos.quantile-transformer.taxi.rows=full/board.json (nv2/paired-fc1327efa1399e793f84d34ae28ae0a84be82adbe1885e604f0f68130bbb3006)`, ran on b6c1fd6fb058

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 10.0 | 10.0..10.0 | 1 | - | - | 10.0 | 7.9 | 2.08 (upload_ms_separate) | - | - | 269.4 | 938.0 | max_abs_diff_vs_sklearn=5.96e-08, output_shape=100000x11, rel_diff_vs_sklearn=2.294e-08 | - | main board, one scored run | wheel | ok (main@fcdfa0fe2 nv2/paired-fc1327efa1399e793f84d34ae28ae0a84be82adbe1885e604f0f68130bbb3006 2026-10-14; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host b6c1fd6fb058) | cuml 26.08.00 | opponent | 232.2 | 232.2..232.2 | 1 | 0.043 | - | 284.2 | 232.2 | 51.99 (upload_ms_untimed) | 0.035 (whole/whole) | - | 935.7 | 488.0 | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x11, rel_diff_vs_sklearn=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'ignore_implicit_zeros': False, 'n_quantiles': 1000, 'output_distribution': 'uniform', 'random_state': 7, 'subsample': 1000000000}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), QuantileTransformer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| copy | true | true |
| ignore_implicit_zeros | false | false |
| n_quantiles | 1000 | 1000 |
| output_distribution | "uniform" | "uniform" |
| seed | 7 | 7 |
| subsample | 1000000000 | 1000000000 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | hardware | version | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | NVIDIA L40S (nv2, RunPod) | 0.8.44 | Xq | - | 4.1 | 4.1..4.1 | 1 | - | - | 4.1 | - | - (stored whole) | - | - | max_abs_diff_vs_sklearn=5.96e-08, output_shape=100000x11, rel_diff_vs_sklearn=2.294e-08 | - | LIKE-FOR-LIKE-SPAN | ok |
| cuml-gpu | NVIDIA L40S (host b6c1fd6fb058) | cuml 26.08.00 | Xq | - | 27.5 | 27.5..27.5 | 1 | 0.148 | - | - | 27.5 | - (stored kernel) | 0.148 (MIXED ours whole / arm kernel) | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x11, rel_diff_vs_sklearn=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: transform(Xq)(Xq)

inference call, cuml-gpu: transform(Xq)(Xq)

### quantile / istella (rows full, shape X 100000x220; Xq 100000x220; y 100000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0328)`, ran on NVIDIA L40S (nv, RunPod) job n0328

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@de2b2b739) | identical | 358.7 | 358.7..358.7 | 1 | - | - | 358.7 | - | - (stored whole) | - | - | - | - | finite=True, r2=-0.039998, rmse=0.851877 | - | main board, one scored run | - | ok (main@de2b2b739 nv/n0328 2026-10-08; identity vs amd-mi325x: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 1.404e+06 | 1.404e+06..1.404e+06 | 1 | 0.0002554 | - | 1.404e+06 | 1.404e+06 | 0.00 (cpu-arm) | 0.0002554 (whole/whole) | - | 7995.7 | - | finite=True, r2=-0.044780, rmse=0.853833 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'fit_intercept': True, 'quantile': 0.5, 'solver': 'highs'}. Rows: None. Timed: None.

mismatch: solver: 'highs' passed to both; ours accepts the name and always runs ADMM (max_iter=5000, tol=1e-4, ours only), scikit-learn solves the HiGHS LP

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### quantile / taxi (rows full, shape X 100000x11; Xq 100000x11; y 100000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0328)`, ran on NVIDIA L40S (nv, RunPod) job n0328

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@de2b2b739) | identical | 234.9 | 234.9..234.9 | 1 | - | - | 234.9 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.899596, rmse=5.046749 | - | main board, one scored run | - | ok (main@de2b2b739 nv/n0328 2026-10-08; identity vs amd-mi325x: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 386298.7 | 386298.7..386298.7 | 1 | 0.0006081 | - | 386298.7 | 386298.7 | 0.00 (cpu-arm) | 0.0006081 (whole/whole) | - | 926.3 | - | finite=True, r2=0.899678, rmse=5.044706 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'fit_intercept': True, 'quantile': 0.5, 'solver': 'highs'}. Rows: None. Timed: None.

mismatch: solver: 'highs' passed to both; ours accepts the name and always runs ADMM (max_iter=5000, tol=1e-4, ours only), scikit-learn solves the HiGHS LP

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### randomized-svd / istella (rows full, shape X 900000x220; Xq 100000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0088)`, ran on NVIDIA L40S (nv2, RunPod) job v0088

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 163.7 | 163.7..163.7 | 1 | - | - | 163.7 | - | - (stored whole) | - | - | - | - | relative_reconstruction_error=0.0002359 | - | main board, one scored run | - | ok (main@a1c96a5ba nv2/v0088 2026-10-11; identity vs the amd columns: n/a) |
| torch-gpu | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 65.7 | 65.7..65.7 | 1 | 2.493 | - | - | 65.7 | - (stored kernel) | 2.493 (MIXED ours whole / arm kernel) | - | 1637.5 | 1012.2 | relative_reconstruction_error=0.0002359 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 1589.1 | 1589.1..1589.1 | 1 | 0.103 | - | 1589.1 | 1589.1 | 0.00 (cpu-arm) | 0.103 (whole/whole) | - | 2079.3 | - | relative_reconstruction_error=0.0002359 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'n_components': 8, 'n_iter': 4, 'n_oversamples': 10, 'random_state': 7}. Rows: None. Timed: None.

mismatch: torch-gpu is torch.svd_lowrank(q=18, niter=4), its randomized range finder

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### randomized-svd / taxi (rows full, shape X 900000x11; Xq 100000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0088)`, ran on NVIDIA L40S (nv2, RunPod) job v0088

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 57.7 | 57.7..57.7 | 1 | - | - | 57.7 | - | - (stored whole) | - | - | - | - | relative_reconstruction_error=0.027197 | - | main board, one scored run | - | ok (main@a1c96a5ba nv2/v0088 2026-10-11; identity vs the amd columns: n/a) |
| torch-gpu | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 32.0 | 32.0..32.0 | 1 | 1.804 | - | - | 32.0 | - (stored kernel) | 1.804 (MIXED ours whole / arm kernel) | - | 838.9 | 198.1 | relative_reconstruction_error=0.027197 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 1017.5 | 1017.5..1017.5 | 1 | 0.057 | - | 1017.5 | 1017.5 | 0.00 (cpu-arm) | 0.057 (whole/whole) | - | 477.8 | - | relative_reconstruction_error=0.027197 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'n_components': 8, 'n_iter': 4, 'n_oversamples': 10, 'random_state': 7}. Rows: None. Timed: None.

mismatch: torch-gpu is torch.svd_lowrank(q=18, niter=4), its randomized range finder

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### resnet-block / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/nv_ingest/results_nvc3.txt (nv/n0006)`, ran on NVIDIA L40S (nv, RunPod) job n0006

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 2.8 ms) = 5.279; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 5.3 ms) = 2.850. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.42 (source build, main@8899093fc) | identical | 15.0 | 15.0..15.0 | 1 | - | - | 15.0 | - | - (stored whole) | - | - | - | - | relative_error_vs_own_host=0.000000 | - | main board, one scored run | - | ok (main@8899093fc nv/n0006 2026-10-13; identity vs the amd columns: n/a) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 5.3 | 5.3..5.3 | 1 | 2.849 | - | - | 5.3 | - (stored kernel) | 2.849 (MIXED ours whole / arm kernel) | - | 942.5 | 450.7 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 5.3 | 5.3..5.3 | 1 | 2.850 | - | - | 5.3 | - (stored kernel) | 2.850 (MIXED ours whole / arm kernel) | - | 1146.2 | 409.2 | max_rel_diff_vs_torch_eager_fp32=0.818182, rel_fro_vs_torch_eager_fp32=5.381e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 4.4 | 4.4..4.4 | 1 | 3.415 | - | - | 4.4 | - (stored kernel) | 3.415 (MIXED ours whole / arm kernel) | - | 959.7 | 554.3 | max_rel_diff_vs_torch_eager_fp32=1532.793045, rel_fro_vs_torch_eager_fp32=0.0003478 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 4.3 | 4.3..4.3 | 1 | 3.480 | - | - | 4.3 | - (stored kernel) | 3.480 (MIXED ours whole / arm kernel) | - | 1109.0 | 516.8 | max_rel_diff_vs_torch_eager_fp32=1532.793045, rel_fro_vs_torch_eager_fp32=0.0003478 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 2.8 | 2.8..2.8 | 1 | 5.279 | - | - | 2.8 | - (stored kernel) | 5.279 (MIXED ours whole / arm kernel) | - | 1002.7 | 424.9 | max_rel_diff_vs_torch_eager_fp32=17838.627100, rel_fro_vs_torch_eager_fp32=0.003621 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 3.4 | 3.4..3.4 | 1 | 4.440 | - | - | 3.4 | - (stored kernel) | 4.440 (MIXED ours whole / arm kernel) | - | 1173.4 | 383.4 | max_rel_diff_vs_torch_eager_fp32=18497.318029, rel_fro_vs_torch_eager_fp32=0.003425 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'inplanes': 64, 'planes': 64}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### rfe / istella (rows full, shape X 200000x220; Xq 100000x220; y 200000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/rfe-istella-20261012/ingest/rfe_exact_nv.txt (nv/n0005)`, ran on NVIDIA L40S (nv, RunPod) job n0005

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.42 (source build, main@8899093fc) | identical | 1362.9 | 1362.9..1362.9 | 1 | - | - | 1362.9 | - | - (stored whole) | - | - | - | - | jaccard_vs_exact=0.774194, jaccard_vs_sklearn=0.880342, n_selected=110 | - | main board, one scored run | - | ok (main@8899093fc nv/n0005 2026-10-13; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 91711.7 | 91711.7..91711.7 | 1 | 0.015 | - | 91711.7 | 91711.7 | 0.00 (cpu-arm) | 0.015 (whole/whole) | - | 1799.2 | - | jaccard_vs_sklearn=1.000000, n_selected=110 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'estimator': 'LogisticRegression(max_iter=200)', 'n_features_to_select': 'half', 'step': 0.1}. Rows: None. Timed: None.

mismatch: nested LogisticRegression(max_iter=200): ours is L-BFGS (cuML's quasi-Newton, memory 5, solver='qn'), scikit-learn solver='lbfgs' (scipy, memory 10); C=1.0 and tol=1e-4 are both libraries' defaults; each stops by its own criterion

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### rfe / taxi (rows full, shape X 200000x11; Xq 100000x11; y 200000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/rfe-istella-20261012/ingest/rfe_exact_nv.txt (nv/n0003)`, ran on NVIDIA L40S (nv, RunPod) job n0003

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.42 (source build, main@12412eca4) | identical | 63.8 | 63.8..63.8 | 1 | - | - | 63.8 | - | - (stored whole) | - | - | - | - | jaccard_vs_exact=1.000000, jaccard_vs_sklearn=1.000000, n_selected=5 | - | main board, one scored run | - | ok (main@12412eca4 nv/n0003 2026-10-12; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 2890.6 | 2890.6..2890.6 | 1 | 0.022 | - | 2890.6 | 2890.6 | 0.00 (cpu-arm) | 0.022 (whole/whole) | - | 294.7 | - | jaccard_vs_sklearn=1.000000, n_selected=5 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'estimator': 'LogisticRegression(max_iter=200)', 'n_features_to_select': 'half', 'step': 0.1}. Rows: None. Timed: None.

mismatch: nested LogisticRegression(max_iter=200): ours is L-BFGS (cuML's quasi-Newton, memory 5, solver='qn'), scikit-learn solver='lbfgs' (scipy, memory 10); C=1.0 and tol=1e-4 are both libraries' defaults; each stops by its own criterion

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ridge-cv / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0079)`, ran on NVIDIA L40S (nv2, RunPod) job v0079

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 23140.7 | 23140.7..23140.7 | 1 | - | - | 23140.7 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.328684, rmse=0.684422 | - | main board, one scored run | - | ok (main@a1c96a5ba nv2/v0079 2026-10-11; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 188507.2 | 188507.2..188507.2 | 1 | 0.123 | - | 188507.2 | 188507.2 | 0.00 (cpu-arm) | 0.123 (whole/whole) | - | 8535.6 | - | finite=True, r2=0.328683, rmse=0.684423 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha_per_target': False, 'alphas': [0.001, 0.01, 0.1, 1.0, 10.0], 'cv': 5, 'fit_intercept': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ridge-cv / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0029)`, ran on NVIDIA L40S (nv2, RunPod) job v0029

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@52d438ade) | identical | 154.7 | 154.7..154.7 | 1 | - | - | 154.7 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.908981, rmse=4.805109 | - | main board, one scored run | - | ok (main@52d438ade nv2/v0029 2026-10-11; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 2252.8 | 2252.8..2252.8 | 1 | 0.069 | - | 2252.8 | 2252.8 | 0.00 (cpu-arm) | 0.069 (whole/whole) | - | 378.4 | - | finite=True, r2=0.908983, rmse=4.805055 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha_per_target': False, 'alphas': [0.001, 0.01, 0.1, 1.0, 10.0], 'cv': 5, 'fit_intercept': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### rmsprop / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/nvidia/campaign/measured/algos.rmsprop.synthetic.rows=full/board.json (nv2/paired-af7e0c3d99652bd189943e26f91ab8db7ad89829c771c269af68c6ec831482f3)`, ran on d2fe567d3b88

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 13.1 ms) = 0.664. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 8.7 | 8.7..8.7 | 1 | - | - | - | 8.7 | - (stored kernel) | - | - | 1027.3 | 1450.0 | optimizer_update_error_vs_torch_fp32=2.228e-06, rel_fro_vs_torch_eager_fp32=3.829e-08 | - | main board, one scored run | wheel | ok (main@79fcd24a5 nv2/paired-af7e0c3d99652bd189943e26f91ab8db7ad89829c771c269af68c6ec831482f3 2026-10-14; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 13.1 | 13.1..13.1 | 1 | 0.664 | - | - | 13.1 | - (stored kernel) | 0.664 (kernel/kernel) | - | 1023.9 | 832.0 | optimizer_update_error_vs_torch_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 64.5 | 64.5..64.5 | 1 | 0.135 | - | - | 64.5 | - (stored kernel) | 0.135 (kernel/kernel) | - | 1341.4 | 832.0 | optimizer_update_error_vs_torch_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

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

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1091)`, ran on NVIDIA L40S (nv2, RunPod) job v1091

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 873.9 ms) = 0.279; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 904.6 ms) = 0.270. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 244.0 | 244.0..244.0 | 1 | - | - | 244.0 | - | - (stored whole) | - | - | - | - | accuracy=0.967828, logloss=0.079029 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1091 2026-10-10; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 904.6 | 904.6..904.6 | 1 | 0.270 | - | - | 904.6 | - (stored kernel) | 0.270 (MIXED ours whole / arm kernel) | - | 1122.2 | 404.2 | accuracy=0.953559 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1154.2 | 1154.2..1154.2 | 1 | 0.211 | - | - | 1154.2 | - (stored kernel) | 0.211 (MIXED ours whole / arm kernel) | - | 1173.6 | 404.2 | accuracy=0.953559 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 876.8 | 876.8..876.8 | 1 | 0.278 | - | - | 876.8 | - (stored kernel) | 0.278 (MIXED ours whole / arm kernel) | - | 1123.6 | 404.2 | accuracy=0.953505 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1181.2 | 1181.2..1181.2 | 1 | 0.207 | - | - | 1181.2 | - (stored kernel) | 0.207 (MIXED ours whole / arm kernel) | - | 1174.6 | 404.2 | accuracy=0.953505 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1223.4 | 1223.4..1223.4 | 1 | 0.199 | - | - | 1223.4 | - (stored kernel) | 0.199 (MIXED ours whole / arm kernel) | - | 1459.3 | 220.9 | accuracy=0.953559 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 873.9 | 873.9..873.9 | 1 | 0.279 | - | - | 873.9 | - (stored kernel) | 0.279 (MIXED ours whole / arm kernel) | - | 1510.4 | 220.9 | accuracy=0.953559 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'nonlinearity': 'tanh', 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### rnn-clf / taxi-hourly (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1091)`, ran on NVIDIA L40S (nv2, RunPod) job v1091

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 642.8 ms) = 0.379; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 507.7 ms) = 0.480. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 243.9 | 243.9..243.9 | 1 | - | - | 243.9 | - | - (stored whole) | - | - | - | - | accuracy=0.862684, logloss=0.313008 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1091 2026-10-10; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 507.7 | 507.7..507.7 | 1 | 0.480 | - | - | 507.7 | - (stored kernel) | 0.480 (MIXED ours whole / arm kernel) | - | 1122.3 | 404.2 | accuracy=0.868056 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 667.6 | 667.6..667.6 | 1 | 0.365 | - | - | 667.6 | - (stored kernel) | 0.365 (MIXED ours whole / arm kernel) | - | 1174.0 | 404.2 | accuracy=0.868056 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 518.2 | 518.2..518.2 | 1 | 0.471 | - | - | 518.2 | - (stored kernel) | 0.471 (MIXED ours whole / arm kernel) | - | 1123.5 | 404.2 | accuracy=0.868056 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 597.0 | 597.0..597.0 | 1 | 0.409 | - | - | 597.0 | - (stored kernel) | 0.409 (MIXED ours whole / arm kernel) | - | 1174.7 | 404.2 | accuracy=0.868056 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 786.6 | 786.6..786.6 | 1 | 0.310 | - | - | 786.6 | - (stored kernel) | 0.310 (MIXED ours whole / arm kernel) | - | 1459.1 | 220.9 | accuracy=0.868001 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 642.8 | 642.8..642.8 | 1 | 0.379 | - | - | 642.8 | - (stored kernel) | 0.379 (MIXED ours whole / arm kernel) | - | 1511.1 | 220.9 | accuracy=0.868001 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'nonlinearity': 'tanh', 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### rnn-reg / synthetic (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1091)`, ran on NVIDIA L40S (nv2, RunPod) job v1091

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 1442.1 ms) = 0.168; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 902.4 ms) = 0.269. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 242.4 | 242.4..242.4 | 1 | - | - | 242.4 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.978602, rmse=0.169476 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1091 2026-10-10; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1095.6 | 1095.6..1095.6 | 1 | 0.221 | - | - | 1095.6 | - (stored kernel) | 0.221 (MIXED ours whole / arm kernel) | - | 1138.3 | 403.9 | finite=True, r2=0.977347, rmse=0.174374 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 902.4 | 902.4..902.4 | 1 | 0.269 | - | - | 902.4 | - (stored kernel) | 0.269 (MIXED ours whole / arm kernel) | - | 1188.9 | 403.9 | finite=True, r2=0.977347, rmse=0.174374 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1136.5 | 1136.5..1136.5 | 1 | 0.213 | - | - | 1136.5 | - (stored kernel) | 0.213 (MIXED ours whole / arm kernel) | - | 1133.1 | 403.9 | finite=True, r2=0.977347, rmse=0.174374 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1161.7 | 1161.7..1161.7 | 1 | 0.209 | - | - | 1161.7 | - (stored kernel) | 0.209 (MIXED ours whole / arm kernel) | - | 1184.7 | 403.9 | finite=True, r2=0.977347, rmse=0.174374 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1644.0 | 1644.0..1644.0 | 1 | 0.147 | - | - | 1644.0 | - (stored kernel) | 0.147 (MIXED ours whole / arm kernel) | - | 1505.8 | 220.5 | finite=True, r2=0.977345, rmse=0.174385 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1442.1 | 1442.1..1442.1 | 1 | 0.168 | - | - | 1442.1 | - (stored kernel) | 0.168 (MIXED ours whole / arm kernel) | - | 1556.6 | 220.5 | finite=True, r2=0.977345, rmse=0.174385 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'nonlinearity': 'tanh', 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### rnn-reg / taxi-hourly (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1091)`, ran on NVIDIA L40S (nv2, RunPod) job v1091

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 612.7 ms) = 0.396; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 664.7 ms) = 0.365. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 242.4 | 242.4..242.4 | 1 | - | - | 242.4 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.743904, rmse=0.548825 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1091 2026-10-10; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 664.7 | 664.7..664.7 | 1 | 0.365 | - | - | 664.7 | - (stored kernel) | 0.365 (MIXED ours whole / arm kernel) | - | 1138.1 | 403.9 | finite=True, r2=0.738796, rmse=0.554271 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 753.0 | 753.0..753.0 | 1 | 0.322 | - | - | 753.0 | - (stored kernel) | 0.322 (MIXED ours whole / arm kernel) | - | 1189.1 | 403.9 | finite=True, r2=0.738796, rmse=0.554271 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 724.2 | 724.2..724.2 | 1 | 0.335 | - | - | 724.2 | - (stored kernel) | 0.335 (MIXED ours whole / arm kernel) | - | 1133.0 | 403.9 | finite=True, r2=0.738800, rmse=0.554267 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 633.1 | 633.1..633.1 | 1 | 0.383 | - | - | 633.1 | - (stored kernel) | 0.383 (MIXED ours whole / arm kernel) | - | 1184.7 | 403.9 | finite=True, r2=0.738800, rmse=0.554267 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 612.7 | 612.7..612.7 | 1 | 0.396 | - | - | 612.7 | - (stored kernel) | 0.396 (MIXED ours whole / arm kernel) | - | 1505.1 | 220.5 | finite=True, r2=0.738906, rmse=0.554155 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 877.8 | 877.8..877.8 | 1 | 0.276 | - | - | 877.8 | - (stored kernel) | 0.276 (MIXED ours whole / arm kernel) | - | 1556.6 | 220.5 | finite=True, r2=0.738906, rmse=0.554155 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'nonlinearity': 'tanh', 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### robust-scaler / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/nv_ingest/results_nvc2.txt (nv/n0001)`, ran on NVIDIA L40S (nv, RunPod) job n0001

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.42 (source build, main@9c7ad7413) | identical | 273.3 | 273.3..273.3 | 1 | - | - | 273.3 | - | - (stored whole) | - | - | - | - | max_abs_diff_vs_sklearn=0.0001221, output_shape=100000x220, rel_diff_vs_sklearn=7.339e-09 | - | main board, one scored run | - | ok (main@9c7ad7413 nv/n0001 2026-10-12; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 828.4 | 828.4..828.4 | 1 | 0.330 | - | 1526.7 | 828.4 | 698.31 (upload_ms_untimed) | 0.179 (whole/whole) | - | 3064.3 | 1442.0 | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 6394.7 | 6394.7..6394.7 | 1 | 0.043 | - | 6394.7 | 6394.7 | 0.00 (cpu-arm) | 0.043 (whole/whole) | - | 1455.4 | - | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'quantile_range': [25.0, 75.0], 'unit_variance': False, 'with_centering': True, 'with_scaling': True}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), RobustScaler (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### robust-scaler / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/nv_ingest/results_nvc2.txt (nv/n0001)`, ran on NVIDIA L40S (nv, RunPod) job n0001

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.42 (source build, main@9c7ad7413) | identical | 8.9 | 8.9..8.9 | 1 | - | - | 8.9 | - | - (stored whole) | - | - | - | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x11, rel_diff_vs_sklearn=0.000000 | - | main board, one scored run | - | ok (main@9c7ad7413 nv/n0001 2026-10-12; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 34.3 | 34.3..34.3 | 1 | 0.259 | - | 91.2 | 34.3 | 56.92 (upload_ms_untimed) | 0.097 (whole/whole) | - | 947.6 | 488.0 | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 184.7 | 184.7..184.7 | 1 | 0.048 | - | 184.7 | 184.7 | 0.00 (cpu-arm) | 0.048 (whole/whole) | - | 265.1 | - | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'quantile_range': [25.0, 75.0], 'unit_variance': False, 'with_centering': True, 'with_scaling': True}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), RobustScaler (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### select-chi2 / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/nv_ingest/results_nvc3.txt (nv/n0003)`, ran on NVIDIA L40S (nv, RunPod) job n0003

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.42 (source build, main@12412eca4) | identical | 58.0 | 58.0..58.0 | 1 | - | - | 58.0 | - | - (stored whole) | - | - | - | - | jaccard_vs_sklearn=1.000000, n_selected=110 | - | main board, one scored run | - | ok (main@12412eca4 nv/n0003 2026-10-12; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 371.2 | 371.2..371.2 | 1 | 0.156 | - | 371.2 | 371.2 | 0.00 (cpu-arm) | 0.156 (whole/whole) | - | 3758.2 | - | jaccard_vs_sklearn=1.000000, n_selected=110 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'k': 'half'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### select-chi2 / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/nv_ingest/results_nvc3.txt (nv/n0003)`, ran on NVIDIA L40S (nv, RunPod) job n0003

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.42 (source build, main@12412eca4) | identical | 13.3 | 13.3..13.3 | 1 | - | - | 13.3 | - | - (stored whole) | - | - | - | - | jaccard_vs_sklearn=1.000000, n_selected=5 | - | main board, one scored run | - | ok (main@12412eca4 nv/n0003 2026-10-12; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 107.0 | 107.0..107.0 | 1 | 0.124 | - | 107.0 | 107.0 | 0.00 (cpu-arm) | 0.124 (whole/whole) | - | 409.7 | - | jaccard_vs_sklearn=1.000000, n_selected=5 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'k': 'half'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### select-d / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/nv_ingest/results_nvc3.txt (nv/n0004)`, ran on NVIDIA L40S (nv, RunPod) job n0004

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.42 (source build, main@e129be096) | identical | 1.6 | 1.6..1.6 | 1 | - | - | 1.6 | - | - (stored whole) | - | - | - | - | relative_error_vs_own_host=0.000000 | - | main board, one scored run | - | ok (main@e129be096 nv/n0004 2026-10-12; identity vs the amd columns: n/a) |

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### select-d / taxi-hourly (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/nv_ingest/results_nvc3.txt (nv/n0004)`, ran on NVIDIA L40S (nv, RunPod) job n0004

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.42 (source build, main@e129be096) | identical | 1.6 | 1.6..1.6 | 1 | - | - | 1.6 | - | - (stored whole) | - | - | - | - | relative_error_vs_own_host=0.000000 | - | main board, one scored run | - | ok (main@e129be096 nv/n0004 2026-10-12; identity vs the amd columns: n/a) |

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### select-f-classif / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/nv_ingest/results_nvc3.txt (nv/n0003)`, ran on NVIDIA L40S (nv, RunPod) job n0003

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.42 (source build, main@12412eca4) | identical | 50.9 | 50.9..50.9 | 1 | - | - | 50.9 | - | - (stored whole) | - | - | - | - | jaccard_vs_sklearn=1.000000, n_selected=110 | - | main board, one scored run | - | ok (main@12412eca4 nv/n0003 2026-10-12; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 444.3 | 444.3..444.3 | 1 | 0.115 | - | 444.3 | 444.3 | 0.00 (cpu-arm) | 0.115 (whole/whole) | - | 3555.7 | - | jaccard_vs_sklearn=1.000000, n_selected=110 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'k': 'half'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### select-f-classif / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/nv_ingest/results_nvc3.txt (nv/n0003)`, ran on NVIDIA L40S (nv, RunPod) job n0003

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.42 (source build, main@12412eca4) | identical | 13.8 | 13.8..13.8 | 1 | - | - | 13.8 | - | - (stored whole) | - | - | - | - | jaccard_vs_sklearn=1.000000, n_selected=5 | - | main board, one scored run | - | ok (main@12412eca4 nv/n0003 2026-10-12; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 77.8 | 77.8..77.8 | 1 | 0.178 | - | 77.8 | 77.8 | 0.00 (cpu-arm) | 0.178 (whole/whole) | - | 377.4 | - | jaccard_vs_sklearn=1.000000, n_selected=5 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'k': 'half'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### select-f-regression / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/nv_ingest/results_nvc2.txt (nv/n0001)`, ran on NVIDIA L40S (nv, RunPod) job n0001

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.42 (source build, main@9c7ad7413) | identical | 51.4 | 51.4..51.4 | 1 | - | - | 51.4 | - | - (stored whole) | - | - | - | - | jaccard_vs_sklearn=1.000000, n_selected=110 | - | main board, one scored run | - | ok (main@9c7ad7413 nv/n0001 2026-10-12; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 491.1 | 491.1..491.1 | 1 | 0.105 | - | 491.1 | 491.1 | 0.00 (cpu-arm) | 0.105 (whole/whole) | - | 2814.4 | - | jaccard_vs_sklearn=1.000000, n_selected=110 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'k': 'half'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### select-f-regression / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/nv_ingest/results_nvc2.txt (nv/n0001)`, ran on NVIDIA L40S (nv, RunPod) job n0001

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.42 (source build, main@9c7ad7413) | identical | 7.1 | 7.1..7.1 | 1 | - | - | 7.1 | - | - (stored whole) | - | - | - | - | jaccard_vs_sklearn=1.000000, n_selected=5 | - | main board, one scored run | - | ok (main@9c7ad7413 nv/n0001 2026-10-12; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 106.0 | 106.0..106.0 | 1 | 0.067 | - | 106.0 | 106.0 | 0.00 (cpu-arm) | 0.067 (whole/whole) | - | 341.0 | - | jaccard_vs_sklearn=1.000000, n_selected=5 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'k': 'half'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### select-mutual-info-reg / istella (rows full, shape X 100000x220; Xq 100000x220; y 100000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/nv_ingest/results_nvc2.txt (nv/n0001)`, ran on NVIDIA L40S (nv, RunPod) job n0001

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.42 (source build, main@9c7ad7413) | identical | 15515.9 | 15515.9..15515.9 | 1 | - | - | 15515.9 | - | - (stored whole) | - | - | - | - | jaccard_vs_sklearn=0.929825, n_selected=110 | - | main board, one scored run | - | ok (main@9c7ad7413 nv/n0001 2026-10-12; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 96232.9 | 96232.9..96232.9 | 1 | 0.161 | - | 96232.9 | 96232.9 | 0.00 (cpu-arm) | 0.161 (whole/whole) | - | 1877.6 | - | jaccard_vs_sklearn=1.000000, n_selected=110 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'k': 'half'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### select-mutual-info-reg / taxi (rows full, shape X 100000x11; Xq 100000x11; y 100000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/nv_ingest/results_nvc2.txt (nv/n0001)`, ran on NVIDIA L40S (nv, RunPod) job n0001

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.42 (source build, main@9c7ad7413) | identical | 692.6 | 692.6..692.6 | 1 | - | - | 692.6 | - | - (stored whole) | - | - | - | - | jaccard_vs_sklearn=1.000000, n_selected=5 | - | main board, one scored run | - | ok (main@9c7ad7413 nv/n0001 2026-10-12; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 5427.4 | 5427.4..5427.4 | 1 | 0.128 | - | 5427.4 | 5427.4 | 0.00 (cpu-arm) | 0.128 (whole/whole) | - | 284.3 | - | jaccard_vs_sklearn=1.000000, n_selected=5 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'k': 'half'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### select-mutual-info / istella (rows full, shape X 100000x220; Xq 100000x220; y 100000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/nv_ingest/results_nvc3.txt (nv/n0003)`, ran on NVIDIA L40S (nv, RunPod) job n0003

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.42 (source build, main@12412eca4) | identical | 510.1 | 510.1..510.1 | 1 | - | - | 510.1 | - | - (stored whole) | - | - | - | - | jaccard_vs_sklearn=0.929825, n_selected=110 | - | main board, one scored run | - | ok (main@12412eca4 nv/n0003 2026-10-12; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 63131.4 | 63131.4..63131.4 | 1 | 0.008 | - | 63131.4 | 63131.4 | 0.00 (cpu-arm) | 0.008 (whole/whole) | - | 1887.0 | - | jaccard_vs_sklearn=1.000000, n_selected=110 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'k': 'half'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### select-mutual-info / taxi (rows full, shape X 100000x11; Xq 100000x11; y 100000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/nv_ingest/results_nvc3.txt (nv/n0003)`, ran on NVIDIA L40S (nv, RunPod) job n0003

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.42 (source build, main@12412eca4) | identical | 77.1 | 77.1..77.1 | 1 | - | - | 77.1 | - | - (stored whole) | - | - | - | - | jaccard_vs_sklearn=0.428571, n_selected=5 | - | main board, one scored run | - | ok (main@12412eca4 nv/n0003 2026-10-12; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 3200.7 | 3200.7..3200.7 | 1 | 0.024 | - | 3200.7 | 3200.7 | 0.00 (cpu-arm) | 0.024 (whole/whole) | - | 289.6 | - | jaccard_vs_sklearn=1.000000, n_selected=5 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'k': 'half'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### select-r-regression / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/nv_ingest/results_nvc2.txt (nv/n0001)`, ran on NVIDIA L40S (nv, RunPod) job n0001

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.42 (source build, main@9c7ad7413) | identical | 51.8 | 51.8..51.8 | 1 | - | - | 51.8 | - | - (stored whole) | - | - | - | - | jaccard_vs_sklearn=1.000000, n_selected=110 | - | main board, one scored run | - | ok (main@9c7ad7413 nv/n0001 2026-10-12; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 590.7 | 590.7..590.7 | 1 | 0.088 | - | 590.7 | 590.7 | 0.00 (cpu-arm) | 0.088 (whole/whole) | - | 2813.6 | - | jaccard_vs_sklearn=1.000000, n_selected=110 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'k': 'half'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### select-r-regression / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/nv_ingest/results_nvc2.txt (nv/n0001)`, ran on NVIDIA L40S (nv, RunPod) job n0001

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.42 (source build, main@9c7ad7413) | identical | 7.2 | 7.2..7.2 | 1 | - | - | 7.2 | - | - (stored whole) | - | - | - | - | jaccard_vs_sklearn=1.000000, n_selected=5 | - | main board, one scored run | - | ok (main@9c7ad7413 nv/n0001 2026-10-12; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 138.8 | 138.8..138.8 | 1 | 0.052 | - | 138.8 | 138.8 | 0.00 (cpu-arm) | 0.052 (whole/whole) | - | 342.8 | - | jaccard_vs_sklearn=1.000000, n_selected=5 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'k': 'half'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### sgd-clf / istella (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0606)`, ran on NVIDIA L40S (nv, RunPod) job n0606

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@d3c0fd72b) | identical | 1588.5 | 1588.5..1588.5 | 1 | - | - | 1588.5 | - | - (stored whole) | - | - | - | - | accuracy=0.920330 | - | main board, one scored run | - | ok (main@d3c0fd72b nv/n0606 2026-10-09; identity vs the amd columns: n/a) |

settings: {'l1_ratio': 0.15, 'fit_intercept': True, 'epsilon': 0.1, 'learning_rate': 'constant', 'eta0': 0.005, 'power_t': 0.5, 'early_stopping': False, 'n_iter_no_change': 5, 'average': False, 'penalty': 'l2', 'alpha': 0.0001, 'max_iter': 100, 'tol': None, 'shuffle': True, 'random_state': 7, 'batch_size': 4096, 'warm_start': False, 'loss': 'hinge', 'class_weight': None}. Rows: None. Timed: None.

mismatch: minibatch SGD of 4096 rows a step on ours and cuML MBSGD (batch_size pinned on both); scikit-learn's SGD is per-sample and is refused, not raced

mismatch: cuML reads epochs=100 and tol=0.0 (no early stop); ours max_iter=100 and tol=None

mismatch: intercept: cuML MBSGD centers X and y before the epochs (GLM preProcessData) and recovers the intercept; ours learns the intercept in the SGD steps

mismatch: opponents withheld: the lane settings at this tree's HEAD differ from the settings opponents-default-20261006 recorded for its opponent race (an opponent job must score them again)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### sgd-clf / taxi (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0606)`, ran on NVIDIA L40S (nv, RunPod) job n0606

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@d3c0fd72b) | identical | 1036.6 | 1036.6..1036.6 | 1 | - | - | 1036.6 | - | - (stored whole) | - | - | - | - | accuracy=0.755330 | - | main board, one scored run | - | ok (main@d3c0fd72b nv/n0606 2026-10-09; identity vs the amd columns: n/a) |

settings: {'l1_ratio': 0.15, 'fit_intercept': True, 'epsilon': 0.1, 'learning_rate': 'constant', 'eta0': 0.005, 'power_t': 0.5, 'early_stopping': False, 'n_iter_no_change': 5, 'average': False, 'penalty': 'l2', 'alpha': 0.0001, 'max_iter': 100, 'tol': None, 'shuffle': True, 'random_state': 7, 'batch_size': 4096, 'warm_start': False, 'loss': 'hinge', 'class_weight': None}. Rows: None. Timed: None.

mismatch: minibatch SGD of 4096 rows a step on ours and cuML MBSGD (batch_size pinned on both); scikit-learn's SGD is per-sample and is refused, not raced

mismatch: cuML reads epochs=100 and tol=0.0 (no early stop); ours max_iter=100 and tol=None

mismatch: intercept: cuML MBSGD centers X and y before the epochs (GLM preProcessData) and recovers the intercept; ours learns the intercept in the SGD steps

mismatch: opponents withheld: the lane settings at this tree's HEAD differ from the settings opponents-specific-20261006 recorded for its opponent race (an opponent job must score them again)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### sgd-reg / istella (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on NVIDIA L40S (nv, RunPod) job n0314

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 1561.5 | 1561.5..1561.5 | 1 | - | - | 1561.5 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.327829, rmse=0.684858 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs the amd columns: n/a) |

settings: {'l1_ratio': 0.15, 'fit_intercept': True, 'epsilon': 0.1, 'learning_rate': 'constant', 'eta0': 0.005, 'power_t': 0.25, 'early_stopping': False, 'n_iter_no_change': 5, 'average': False, 'penalty': 'l2', 'alpha': 0.0001, 'max_iter': 100, 'tol': None, 'shuffle': True, 'random_state': 7, 'batch_size': 4096, 'warm_start': False, 'loss': 'squared_error'}. Rows: None. Timed: None.

mismatch: minibatch SGD of 4096 rows a step on ours and cuML MBSGD (batch_size pinned on both); scikit-learn's SGD is per-sample and is refused, not raced

mismatch: cuML reads epochs=100 and tol=0.0 (no early stop); ours max_iter=100 and tol=None

mismatch: intercept: cuML MBSGD centers X and y before the epochs (GLM preProcessData) and recovers the intercept; ours learns the intercept in the SGD steps

mismatch: opponents withheld: the lane settings at this tree's HEAD differ from the settings opponents-default-20261006 recorded for its opponent race (an opponent job must score them again)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### sgd-reg / taxi (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on NVIDIA L40S (nv, RunPod) job n0314

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 1024.3 | 1024.3..1024.3 | 1 | - | - | 1024.3 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.908969, rmse=4.805411 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs the amd columns: n/a) |

settings: {'l1_ratio': 0.15, 'fit_intercept': True, 'epsilon': 0.1, 'learning_rate': 'constant', 'eta0': 0.005, 'power_t': 0.25, 'early_stopping': False, 'n_iter_no_change': 5, 'average': False, 'penalty': 'l2', 'alpha': 0.0001, 'max_iter': 100, 'tol': None, 'shuffle': True, 'random_state': 7, 'batch_size': 4096, 'warm_start': False, 'loss': 'squared_error'}. Rows: None. Timed: None.

mismatch: minibatch SGD of 4096 rows a step on ours and cuML MBSGD (batch_size pinned on both); scikit-learn's SGD is per-sample and is refused, not raced

mismatch: cuML reads epochs=100 and tol=0.0 (no early stop); ours max_iter=100 and tol=None

mismatch: intercept: cuML MBSGD centers X and y before the epochs (GLM preProcessData) and recovers the intercept; ours learns the intercept in the SGD steps

mismatch: opponents withheld: the lane settings at this tree's HEAD differ from the settings opponents-specific-20261006 recorded for its opponent race (an opponent job must score them again)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### sgd / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/nvidia/campaign/measured/algos.sgd.synthetic.rows=full/board.json (nv2/paired-5270971d6e48fb01e58bb8e8f4b44819be66304eee5b380eaab96ced31088392)`, ran on d2fe567d3b88

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 9.5 ms) = 2.201. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 20.9 | 20.9..20.9 | 1 | - | - | - | 20.9 | - (stored kernel) | - | - | 2337.0 | 1706.0 | optimizer_update_error_vs_torch_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | main board, one scored run | wheel | ok (main@79fcd24a5 nv2/paired-5270971d6e48fb01e58bb8e8f4b44819be66304eee5b380eaab96ced31088392 2026-10-14; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 9.5 | 9.5..9.5 | 1 | 2.201 | - | - | 9.5 | - (stored kernel) | 2.201 (kernel/kernel) | - | 988.7 | 832.0 | optimizer_update_error_vs_torch_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 111.7 | 111.7..111.7 | 1 | 0.187 | - | - | 111.7 | - (stored kernel) | 0.187 (kernel/kernel) | - | 1207.0 | 832.0 | optimizer_update_error_vs_torch_fp32=4.764e-06, rel_fro_vs_torch_eager_fp32=6.772e-10 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-fused-fp32 | torch | gpu | NVIDIA L40S (host d2fe567d3b88) | torch 2.13.0+cu129 | opponent | 8.8 | 8.8..8.8 | 1 | 2.371 | - | - | 8.8 | - (stored kernel) | 2.371 (kernel/kernel) | - | 963.1 | 832.0 | optimizer_update_error_vs_torch_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

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

### simple-imputer / istella (rows full, shape X 1000000x220; X_true 1000000x220; Xq 100000x220; Xq_true 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on NVIDIA L40S (nv, RunPod) job n0314

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 240.2 | 240.2..240.2 | 1 | - | - | 240.2 | - | - (stored whole) | - | - | - | - | masked_rmse=346849.129968 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 250.6 | 250.6..250.6 | 1 | 0.959 | - | 981.6 | 250.6 | 731.04 (upload_ms_untimed) | 0.245 (whole/whole) | - | 4914.9 | 1442.0 | masked_rmse=346849.129968 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 36684.7 | 36684.7..36684.7 | 1 | 0.007 | - | 36684.7 | 36684.7 | 0.00 (cpu-arm) | 0.007 (whole/whole) | - | 8060.1 | - | masked_rmse=346849.129968 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'add_indicator': False, 'keep_empty_features': False, 'strategy': 'median'}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), SimpleImputer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### simple-imputer / taxi (rows full, shape X 1000000x11; X_true 1000000x11; Xq 100000x11; Xq_true 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on NVIDIA L40S (nv, RunPod) job n0314

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 9.8 | 9.8..9.8 | 1 | - | - | 9.8 | - | - (stored whole) | - | - | - | - | masked_rmse=5.985180 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 22.4 | 22.4..22.4 | 1 | 0.440 | - | 84.8 | 22.4 | 62.39 (upload_ms_untimed) | 0.116 (whole/whole) | - | 1043.3 | 488.0 | masked_rmse=5.985180 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 1480.4 | 1480.4..1480.4 | 1 | 0.007 | - | 1480.4 | 1480.4 | 0.00 (cpu-arm) | 0.007 (whole/whole) | - | 597.8 | - | masked_rmse=5.985180 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'add_indicator': False, 'keep_empty_features': False, 'strategy': 'median'}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), SimpleImputer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### sparse-coder / istella (rows full, shape X 5000x220; Xq 100000x220; y 5000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/nv_ingest/results_nvc2.txt (nv/n0001)`, ran on NVIDIA L40S (nv, RunPod) job n0001

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.42 (source build, main@9c7ad7413) | identical | 0.5 | 0.5..0.5 | 1 | - | - | 0.5 | - | - (stored whole) | - | - | - | - | max_abs_diff_vs_sklearn=6.104e-05, output_shape=100000x64, rel_diff_vs_sklearn=9.356e-08 | - | main board, one scored run | - | ok (main@9c7ad7413 nv/n0001 2026-10-12; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 0.0 | 0.0..0.0 | 1 | 13.199 | - | 0.0 | 0.0 | 0.00 (cpu-arm) | 13.199 (whole/whole) | - | 1339.3 | - | output_shape=100000x64 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'positive_code': False, 'split_sign': False, 'transform_algorithm': 'omp', 'transform_alpha': None, 'transform_max_iter': 1000, 'transform_n_nonzero_coefs': 4}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### sparse-coder / taxi (rows full, shape X 5000x11; Xq 100000x11; y 5000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/nv_ingest/results_nvc2.txt (nv/n0001)`, ran on NVIDIA L40S (nv, RunPod) job n0001

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.42 (source build, main@9c7ad7413) | identical | 0.5 | 0.5..0.5 | 1 | - | - | 0.5 | - | - (stored whole) | - | - | - | - | max_abs_diff_vs_sklearn=4.292e-06, output_shape=100000x64, rel_diff_vs_sklearn=8.246e-08 | - | main board, one scored run | - | ok (main@9c7ad7413 nv/n0001 2026-10-12; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 0.0 | 0.0..0.0 | 1 | 20.196 | - | 0.0 | 0.0 | 0.00 (cpu-arm) | 20.196 (whole/whole) | - | 375.6 | - | output_shape=100000x64 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'positive_code': False, 'split_sign': False, 'transform_algorithm': 'omp', 'transform_alpha': None, 'transform_max_iter': 1000, 'transform_n_nonzero_coefs': 4}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### sparse-rp / istella (rows full, shape X 900000x220; Xq 100000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on NVIDIA L40S (nv, RunPod) job n0314

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 54.2 | 54.2..54.2 | 1 | - | - | 54.2 | - | - (stored whole) | - | - | - | - | mean_abs_distortion=1.883381 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 108.4 | 108.4..108.4 | 1 | 0.500 | - | 108.4 | - | - (stored whole) | 0.500 (whole/whole) | - | 1993.6 | 430.0 | mean_abs_distortion=0.876709 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 32.5 | 32.5..32.5 | 1 | 1.671 | - | 32.5 | 32.5 | 0.00 (cpu-arm) | 1.671 (whole/whole) | - | 1134.7 | - | mean_abs_distortion=0.474347 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'compute_inverse_components': False, 'dense_output': False, 'density': 'auto', 'eps': 0.1, 'n_components': 10, 'random_state': 7}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), SparseRandomProjection (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### sparse-rp / taxi (rows full, shape X 900000x11; Xq 100000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on NVIDIA L40S (nv, RunPod) job n0314

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 4.7 | 4.7..4.7 | 1 | - | - | 4.7 | - | - (stored whole) | - | - | - | - | mean_abs_distortion=0.147163 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 5.9 | 5.9..5.9 | 1 | 0.794 | - | 5.9 | - | - (stored whole) | 0.794 (whole/whole) | - | 932.9 | 430.0 | mean_abs_distortion=0.266849 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 3.9 | 3.9..3.9 | 1 | 1.186 | - | 3.9 | 3.9 | 0.00 (cpu-arm) | 1.186 (whole/whole) | - | 257.7 | - | mean_abs_distortion=0.381016 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'compute_inverse_components': False, 'dense_output': False, 'density': 'auto', 'eps': 0.1, 'n_components': 10, 'random_state': 7}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), SparseRandomProjection (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### spline / istella (rows full, shape X 1000000x16; Xq 100000x16; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/nv_ingest/results_nvc2.txt (nv/n0001)`, ran on NVIDIA L40S (nv, RunPod) job n0001

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.42 (source build, main@9c7ad7413) | identical | 51.7 | 51.7..51.7 | 1 | - | - | 51.7 | - | - (stored whole) | - | - | - | - | max_abs_diff_vs_sklearn=1.788e-07, output_shape=100000x112, rel_diff_vs_sklearn=6.226e-08 | - | main board, one scored run | - | ok (main@9c7ad7413 nv/n0001 2026-10-12; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 57.9 | 57.9..57.9 | 1 | 0.892 | - | 57.9 | 57.9 | 0.00 (cpu-arm) | 0.892 (whole/whole) | - | 1367.0 | - | output_shape=100000x112 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'degree': 3, 'extrapolation': 'constant', 'include_bias': True, 'knots': 'uniform', 'n_knots': 5}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### spline / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/nv_ingest/results_nvc2.txt (nv/n0001)`, ran on NVIDIA L40S (nv, RunPod) job n0001

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.42 (source build, main@9c7ad7413) | identical | 36.2 | 36.2..36.2 | 1 | - | - | 36.2 | - | - (stored whole) | - | - | - | - | max_abs_diff_vs_sklearn=1.788e-07, output_shape=100000x77, rel_diff_vs_sklearn=5.888e-08 | - | main board, one scored run | - | ok (main@9c7ad7413 nv/n0001 2026-10-12; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 59.9 | 59.9..59.9 | 1 | 0.604 | - | 59.9 | 59.9 | 0.00 (cpu-arm) | 0.604 (whole/whole) | - | 370.1 | - | output_shape=100000x77 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'degree': 3, 'extrapolation': 'constant', 'include_bias': True, 'knots': 'uniform', 'n_knots': 5}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### standard-scaler / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/recovered-inference/algos.standard-scaler.istella.rows=full/board.json (nv2/paired-51bb62d9b9553a593f8ed05931b2e2ef977b9523ab65a2c976996c3a108c8ce5)`, ran on b6c1fd6fb058

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 45.6 | 45.6..45.6 | 1 | - | - | 45.6 | 9.3 | 36.39 (upload_ms_separate) | - | - | 2935.2 | 1450.0 | max_abs_diff_vs_sklearn=0.002426, output_shape=100000x220, rel_diff_vs_sklearn=3.602e-06 | - | main board, one scored run | wheel | ok (main@fcdfa0fe2 nv2/paired-51bb62d9b9553a593f8ed05931b2e2ef977b9523ab65a2c976996c3a108c8ce5 2026-10-14; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host b6c1fd6fb058) | cuml 26.08.00 | opponent | 415.0 | 415.0..415.0 | 1 | 0.110 | - | 1050.7 | 415.0 | 635.72 (upload_ms_untimed) | 0.043 (whole/whole) | - | 3294.7 | 1442.0 | max_abs_diff_vs_sklearn=0.022049, output_shape=100000x220, rel_diff_vs_sklearn=0.0001523 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'with_mean': True, 'with_std': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| copy | true | true |
| seed | "none (deterministic)" | "none (deterministic)" |
| with_mean | true | true |
| with_std | true | true |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | hardware | version | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | NVIDIA L40S (nv2, RunPod) | 0.8.44 | Xq | - | 52.8 | 52.8..52.8 | 1 | - | - | 52.8 | - | - (stored whole) | - | - | max_abs_diff_vs_sklearn=0.002426, output_shape=100000x220, rel_diff_vs_sklearn=3.602e-06 | - | LIKE-FOR-LIKE-SPAN | ok |
| cuml-gpu | NVIDIA L40S (host b6c1fd6fb058) | cuml 26.08.00 | Xq | - | 3.6 | 3.6..3.6 | 1 | 14.720 | - | - | 3.6 | - (stored kernel) | 14.720 (MIXED ours whole / arm kernel) | - | max_abs_diff_vs_sklearn=0.022049, output_shape=100000x220, rel_diff_vs_sklearn=0.0001523 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: transform(Xq)(Xq)

inference call, cuml-gpu: transform(Xq)(Xq)

### standard-scaler / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014/recovered-inference/algos.standard-scaler.taxi.rows=full/board.json (nv2/paired-b021dcec1d3416ffdb5a93fcd40da3066f37f39065b166431c3dd5cc687ac7a5)`, ran on b6c1fd6fb058

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | 0.8.44 | identical | 3.3 | 3.3..3.3 | 1 | - | - | 3.3 | 1.3 | 2.03 (upload_ms_separate) | - | - | 1579.5 | 682.0 | max_abs_diff_vs_sklearn=0.000103, output_shape=100000x11, rel_diff_vs_sklearn=3.854e-06 | - | main board, one scored run | wheel | ok (main@fcdfa0fe2 nv2/paired-b021dcec1d3416ffdb5a93fcd40da3066f37f39065b166431c3dd5cc687ac7a5 2026-10-14; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host b6c1fd6fb058) | cuml 26.08.00 | opponent | 54.3 | 54.3..54.3 | 1 | 0.061 | - | 107.3 | 54.3 | 52.99 (upload_ms_untimed) | 0.031 (whole/whole) | - | 906.0 | 488.0 | max_abs_diff_vs_sklearn=0.0003331, output_shape=100000x11, rel_diff_vs_sklearn=3.216e-05 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'with_mean': True, 'with_std': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| copy | true | true |
| seed | "none (deterministic)" | "none (deterministic)" |
| with_mean | true | true |
| with_std | true | true |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | hardware | version | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | NVIDIA L40S (nv2, RunPod) | 0.8.44 | Xq | - | 3.2 | 3.2..3.2 | 1 | - | - | 3.2 | - | - (stored whole) | - | - | max_abs_diff_vs_sklearn=0.000103, output_shape=100000x11, rel_diff_vs_sklearn=3.854e-06 | - | LIKE-FOR-LIKE-SPAN | ok |
| cuml-gpu | NVIDIA L40S (host b6c1fd6fb058) | cuml 26.08.00 | Xq | - | 0.6 | 0.6..0.6 | 1 | 4.918 | - | - | 0.6 | - (stored kernel) | 4.918 (MIXED ours whole / arm kernel) | - | max_abs_diff_vs_sklearn=0.0003331, output_shape=100000x11, rel_diff_vs_sklearn=3.216e-05 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: transform(Xq)(Xq)

inference call, cuml-gpu: transform(Xq)(Xq)

### svd / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0088)`, ran on NVIDIA L40S (nv2, RunPod) job v0088

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 958.4 | 958.4..958.4 | 1 | - | - | 958.4 | - | - (stored whole) | - | - | - | - | max_rel_singular_value_error=662.224271, relative_reconstruction_error_100k_rows=3.379e-05 | - | main board, one scored run | - | ok (main@a1c96a5ba nv2/v0088 2026-10-11; identity vs the amd columns: n/a) |
| torch-gpu | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 266.0 | 266.0..266.0 | 1 | 3.603 | - | - | 266.0 | - (stored kernel) | 3.603 (MIXED ours whole / arm kernel) | - | 1883.3 | 5176.3 | max_rel_singular_value_error=8.068917, relative_reconstruction_error_100k_rows=2.861e-05 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| cupy-gpu | cupy | gpu | NVIDIA L40S (host 24a11adce16e) | cupy 14.2.0 | opponent | 242.8 | 242.8..242.8 | 1 | 3.947 | - | - | 242.8 | - (stored kernel) | 3.947 (MIXED ours whole / arm kernel) | - | 2666.2 | 9146.0 | max_rel_singular_value_error=10270.720886, relative_reconstruction_error_100k_rows=2.385e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| numpy-cpu | numpy | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | numpy 2.4.6 | opponent | 13448.4 | 13448.4..13448.4 | 1 | 0.071 | - | 13448.4 | 13448.4 | 0.00 (cpu-arm) | 0.071 (whole/whole) | - | 8679.1 | - | max_rel_singular_value_error=1.000000, relative_reconstruction_error_100k_rows=4.1e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, cupy-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'full_matrices': False}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### svd / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0088)`, ran on NVIDIA L40S (nv2, RunPod) job v0088

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 29.6 | 29.6..29.6 | 1 | - | - | 29.6 | - | - (stored whole) | - | - | - | - | max_rel_singular_value_error=7.036e-07, relative_reconstruction_error_100k_rows=1.18e-06 | - | main board, one scored run | - | ok (main@a1c96a5ba nv2/v0088 2026-10-11; identity vs the amd columns: n/a) |
| torch-gpu | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 8.6 | 8.6..8.6 | 1 | 3.440 | - | - | 8.6 | - (stored kernel) | 3.440 (MIXED ours whole / arm kernel) | - | 850.4 | 1186.8 | max_rel_singular_value_error=2.595e-06, relative_reconstruction_error_100k_rows=7.244e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| cupy-gpu | cupy | gpu | NVIDIA L40S (host cc560ebdaf91) | cupy 14.2.0 | opponent | 8.3 | 8.3..8.3 | 1 | 3.573 | - | - | 8.3 | - (stored kernel) | 3.573 (MIXED ours whole / arm kernel) | - | 658.5 | 2766.0 | max_rel_singular_value_error=3.763e-06, relative_reconstruction_error_100k_rows=6.886e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| numpy-cpu | numpy | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | numpy 2.4.6 | opponent | 319.0 | 319.0..319.0 | 1 | 0.093 | - | 319.0 | 319.0 | 0.00 (cpu-arm) | 0.093 (whole/whole) | - | 450.8 | - | max_rel_singular_value_error=4.308e-08, relative_reconstruction_error_100k_rows=4.314e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, cupy-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'full_matrices': False}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### svgp / istella (rows full, shape X 100000x220; Xq 20000x220; y 100000; yq 20000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0073)`, ran on NVIDIA L40S (nv2, RunPod) job v0073

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c69768593) | identical | 207.0 | 207.0..207.0 | 1 | - | - | 207.0 | - | - (stored whole) | - | - | - | - | finite=True, r2=-0.106016, rmse=0.878373 | - | main board, one scored run | - | ok (main@c69768593 nv2/v0073 2026-10-11; identity vs the amd columns: n/a) |
| gpytorch-gpu | gpytorch | gpu | NVIDIA L40S (host 24a11adce16e) | gpytorch 1.15.2 | opponent | 36.2 | 36.2..36.2 | 1 | 5.710 | - | - | 36.2 | - (stored kernel) | 5.710 (MIXED ours whole / arm kernel) | - | 2157.6 | 1492.0 | finite=True, r2=-0.106040, rmse=0.878383 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| gpytorch-cpu | gpytorch | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | gpytorch 1.15.2 | opponent | 14121.1 | 14121.1..14121.1 | 1 | 0.015 | - | 14121.1 | 14121.1 | 0.00 (cpu-arm) | 0.015 (whole/whole) | - | 3161.9 | - | finite=True, r2=-0.106040, rmse=0.878383 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, gpytorch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, gpytorch-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'jitter': 1e-06, 'kernel_variance': 1.0, 'lengthscale': 1.0, 'n_inducing': 512, 'noise_variance': 1.0}. Rows: None. Timed: None.

mismatch: no seed on any arm: nothing is drawn (fixed inducing points, closed form)

mismatch: jitter: ours 1e-6 on K_uu; gpytorch adds its own Cholesky jitter (1e-6 in float32) only when a factorization fails

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### svgp / taxi (rows full, shape X 100000x11; Xq 20000x11; y 100000; yq 20000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0073)`, ran on NVIDIA L40S (nv2, RunPod) job v0073

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c69768593) | identical | 332.8 | 332.8..332.8 | 1 | - | - | 332.8 | - | - (stored whole) | - | - | - | - | finite=True, r2=-0.194982, rmse=17.723482 | - | main board, one scored run | - | ok (main@c69768593 nv2/v0073 2026-10-11; identity vs the amd columns: n/a) |
| gpytorch-gpu | gpytorch | gpu | NVIDIA L40S (host cc560ebdaf91) | gpytorch 1.15.2 | opponent | 29.1 | 29.1..29.1 | 1 | 11.445 | - | - | 29.1 | - (stored kernel) | 11.445 (MIXED ours whole / arm kernel) | - | 1184.3 | 1395.0 | finite=True, r2=-0.209527, rmse=17.831018 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| gpytorch-cpu | gpytorch | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | gpytorch 1.15.2 | opponent | 10845.8 | 10845.8..10845.8 | 1 | 0.031 | - | 10845.8 | 10845.8 | 0.00 (cpu-arm) | 0.031 (whole/whole) | - | 2183.1 | - | finite=True, r2=-0.209546, rmse=17.831162 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, gpytorch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, gpytorch-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'jitter': 1e-06, 'kernel_variance': 1.0, 'lengthscale': 1.0, 'n_inducing': 512, 'noise_variance': 1.0}. Rows: None. Timed: None.

mismatch: no seed on any arm: nothing is drawn (fixed inducing points, closed form)

mismatch: jitter: ours 1e-6 on K_uu; gpytorch adds its own Cholesky jitter (1e-6 in float32) only when a factorization fails

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### target-encoder / istella (rows full, shape X 1000000x8; Xq 100000x8; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/nv_ingest/results_nvc3.txt (nv/n0003)`, ran on NVIDIA L40S (nv, RunPod) job n0003

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.42 (source build, main@12412eca4) | identical | 126.4 | 126.4..126.4 | 1 | - | - | 126.4 | - | - (stored whole) | - | - | - | - | max_abs_diff_vs_sklearn=1.074e-08, output_shape=100000x8, rel_diff_vs_sklearn=2.361e-08 | - | main board, one scored run | - | ok (main@12412eca4 nv/n0003 2026-10-12; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 530.1 | 530.1..530.1 | 1 | 0.239 | - | 560.7 | 530.1 | 30.57 (upload_ms_untimed) | 0.226 (whole/whole) | - | 1169.3 | 684.0 | output_shape=100000x8 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 542.6 | 542.6..542.6 | 1 | 0.233 | - | 542.6 | 542.6 | 0.00 (cpu-arm) | 0.233 (whole/whole) | - | 340.2 | - | output_shape=100000x8 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'categories': 'auto', 'cv': 4, 'random_state': 42, 'shuffle': True, 'smooth': 0.0, 'target_type': 'binary'}. Rows: None. Timed: None.

mismatch: fold assignment: cuML 'interleaved' (row i in fold i mod 4, the cuML benchmark's cuml_args); scikit-learn and ours a KFold shuffled by seed 42 (its cpu_args)

config: cuML benchmark (RAPIDS), TargetEncoder (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### target-encoder / taxi (rows full, shape X 1000000x5; Xq 100000x5; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/nv_ingest/results_nvc3.txt (nv/n0003)`, ran on NVIDIA L40S (nv, RunPod) job n0003

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.42 (source build, main@12412eca4) | identical | 98.4 | 98.4..98.4 | 1 | - | - | 98.4 | - | - (stored whole) | - | - | - | - | max_abs_diff_vs_sklearn=2.965e-08, output_shape=100000x5, rel_diff_vs_sklearn=1.581e-08 | - | main board, one scored run | - | ok (main@12412eca4 nv/n0003 2026-10-12; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 1170.7 | 1170.7..1170.7 | 1 | 0.084 | - | 1205.0 | 1170.7 | 34.30 (upload_ms_untimed) | 0.082 (whole/whole) | - | 1331.0 | 614.0 | output_shape=100000x5 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 227.5 | 227.5..227.5 | 1 | 0.432 | - | 227.5 | 227.5 | 0.00 (cpu-arm) | 0.432 (whole/whole) | - | 296.9 | - | output_shape=100000x5 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'categories': 'auto', 'cv': 4, 'random_state': 42, 'shuffle': True, 'smooth': 0.0, 'target_type': 'binary'}. Rows: None. Timed: None.

mismatch: fold assignment: cuML 'interleaved' (row i in fold i mod 4, the cuML benchmark's cuml_args); scikit-learn and ours a KFold shuffled by seed 42 (its cpu_args)

config: cuML benchmark (RAPIDS), TargetEncoder (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### theta / synthetic (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0097)`, ran on NVIDIA L40S (nv2, RunPod) job v0097

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 55.7 | 55.7..55.7 | 1 | - | - | 55.7 | - | - (stored whole) | - | - | - | - | forecast_rmse=1.436610 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0097 2026-10-11; identity vs the amd columns: n/a) |
| statsforecast-cpu | statsforecast | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | statsforecast 2.1.1 | opponent | 284.0 | 284.0..284.0 | 1 | 0.196 | - | 284.0 | 284.0 | 0.00 (cpu-arm) | 0.196 (whole/whole) | - | 467.4 | - | forecast_rmse=1.436557 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| statsmodels-cpu | statsmodels | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | statsmodels 0.15.0 | opponent | 82.5 | 82.5..82.5 | 1 | 0.676 | - | 82.5 | 82.5 | 0.00 (cpu-arm) | 0.676 (whole/whole) | - | 58.3 | - | forecast_rmse=1.434862 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, statsforecast-cpu, statsmodels-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'decomposition_type': 'multiplicative', 'season_length': 24}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### theta / taxi-hourly (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0097)`, ran on NVIDIA L40S (nv2, RunPod) job v0097

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 170.7 | 170.7..170.7 | 1 | - | - | 170.7 | - | - (stored whole) | - | - | - | - | forecast_rmse=49.020604 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0097 2026-10-11; identity vs the amd columns: n/a) |
| statsforecast-cpu | statsforecast | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | statsforecast 2.1.1 | opponent | 1274.4 | 1274.4..1274.4 | 1 | 0.134 | - | 1274.4 | 1274.4 | 0.00 (cpu-arm) | 0.134 (whole/whole) | - | 467.3 | - | forecast_rmse=49.253901 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| statsmodels-cpu | statsmodels | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | statsmodels 0.15.0 | opponent | 102.3 | 102.3..102.3 | 1 | 1.669 | - | 102.3 | 102.3 | 0.00 (cpu-arm) | 1.669 (whole/whole) | - | 58.1 | - | forecast_rmse=49.311757 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, statsforecast-cpu, statsmodels-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'decomposition_type': 'multiplicative', 'season_length': 24}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### tree-shap / istella (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0530)`, ran on NVIDIA L40S (nv, RunPod) job n0530

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@42d1e42c6) | identical | 17.4 | 17.4..17.4 | 1 | - | - | 17.4 | - | - (stored whole) | - | - | - | - | max_additivity_error=1.175e-06 | - | main board, one scored run | - | ok (main@42d1e42c6 nv/n0530 2026-10-08; identity vs amd-mi325x: MATCH) |

settings: {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'explained_model': "ours' RandomForestRegressor, covers = 100 background rows"}. Rows: None. Timed: None.

mismatch: opponents withheld: the lane settings at this tree's HEAD differ from the settings opponents-specific-20261006 recorded for its opponent race (an opponent job must score them again)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### tree-shap / taxi (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0530)`, ran on NVIDIA L40S (nv, RunPod) job n0530

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@42d1e42c6) | identical | 2.2 | 2.2..2.2 | 1 | - | - | 2.2 | - | - (stored whole) | - | - | - | - | max_additivity_error=3.858e-05 | - | main board, one scored run | - | ok (main@42d1e42c6 nv/n0530 2026-10-08; identity vs amd-mi325x: MATCH) |

settings: {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'explained_model': "ours' RandomForestRegressor, covers = 100 background rows"}. Rows: None. Timed: None.

mismatch: opponents withheld: the lane settings at this tree's HEAD differ from the settings opponents-default-20261006 recorded for its opponent race (an opponent job must score them again)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### tsne / istella (rows full, shape X 20000x220; Xq 2000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0128)`, ran on NVIDIA L40S (nv2, RunPod) job v0128

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 1949.7 | 1949.7..1949.7 | 1 | - | - | 1949.7 | - | - (stored whole) | - | - | - | - | trustworthiness_k15=0.992124 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0128 2026-10-11; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 467.7 | 467.7..467.7 | 1 | 4.169 | - | 492.2 | 467.7 | 24.52 (upload_ms_untimed) | 3.961 (whole/whole) | - | 966.6 | 464.0 | trustworthiness_k15=0.990033 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 43202.0 | 43202.0..43202.0 | 1 | 0.045 | - | 43202.0 | 43202.0 | 0.00 (cpu-arm) | 0.045 (whole/whole) | - | 330.8 | - | trustworthiness_k15=0.992182 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'early_exaggeration': 12.0, 'init': 'seeded', 'learning_rate': 'auto', 'max_iter': 1000, 'n_components': 2, 'perplexity': 30.0, 'random_state': 7}. Rows: None. Timed: None.

mismatch: gradients: ours exact repulsion over k-NN affinities (no Barnes-Hut atomics under IDENTICAL); scikit-learn Barnes-Hut (angle=0.5, scikit-learn only); cuML FFT

mismatch: init: ours and scikit-learn start from the SAME array, ours' 'random' rule (default_rng(7).uniform(-5e-5, 5e-5, (n, 2)) float32); cuML takes only 'random' and draws its own start

mismatch: scikit-learn's early stop is switched off (n_iter_without_progress=1000, min_grad_norm=0.0): ours runs exactly max_iter steps

config: cuML benchmark (RAPIDS), TSNE (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### tsne / taxi (rows full, shape X 20000x11; Xq 2000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0129)`, ran on NVIDIA L40S (nv2, RunPod) job v0129

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 1070.1 | 1070.1..1070.1 | 1 | - | - | 1070.1 | - | - (stored whole) | - | - | - | - | trustworthiness_k15=0.998921 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0129 2026-10-11; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 396.9 | 396.9..396.9 | 1 | 2.696 | - | 401.6 | 396.9 | 4.72 (upload_ms_untimed) | 2.664 (whole/whole) | - | 918.1 | 448.0 | trustworthiness_k15=0.998353 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 38518.3 | 38518.3..38518.3 | 1 | 0.028 | - | 38518.3 | 38518.3 | 0.00 (cpu-arm) | 0.028 (whole/whole) | - | 364.1 | - | trustworthiness_k15=0.998823 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'early_exaggeration': 12.0, 'init': 'seeded', 'learning_rate': 'auto', 'max_iter': 1000, 'n_components': 2, 'perplexity': 30.0, 'random_state': 7}. Rows: None. Timed: None.

mismatch: gradients: ours exact repulsion over k-NN affinities (no Barnes-Hut atomics under IDENTICAL); scikit-learn Barnes-Hut (angle=0.5, scikit-learn only); cuML FFT

mismatch: init: ours and scikit-learn start from the SAME array, ours' 'random' rule (default_rng(7).uniform(-5e-5, 5e-5, (n, 2)) float32); cuML takes only 'random' and draws its own start

mismatch: scikit-learn's early stop is switched off (n_iter_without_progress=1000, min_grad_norm=0.0): ours runs exactly max_iter steps

config: cuML benchmark (RAPIDS), TSNE (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### variance-threshold / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/nv_ingest/results_nvc2.txt (nv/n0001)`, ran on NVIDIA L40S (nv, RunPod) job n0001

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.42 (source build, main@9c7ad7413) | identical | 49.0 | 49.0..49.0 | 1 | - | - | 49.0 | - | - (stored whole) | - | - | - | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x142, rel_diff_vs_sklearn=0.000000 | - | main board, one scored run | - | ok (main@9c7ad7413 nv/n0001 2026-10-12; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 896.5 | 896.5..896.5 | 1 | 0.055 | - | 896.5 | 896.5 | 0.00 (cpu-arm) | 0.055 (whole/whole) | - | 5064.4 | - | output_shape=100000x142 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'threshold': 0.01}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### variance-threshold / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/nv_ingest/results_nvc2.txt (nv/n0001)`, ran on NVIDIA L40S (nv, RunPod) job n0001

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.42 (source build, main@9c7ad7413) | identical | 4.2 | 4.2..4.2 | 1 | - | - | 4.2 | - | - (stored whole) | - | - | - | - | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x10, rel_diff_vs_sklearn=0.000000 | - | main board, one scored run | - | ok (main@9c7ad7413 nv/n0001 2026-10-12; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 90.7 | 90.7..90.7 | 1 | 0.046 | - | 90.7 | 90.7 | 0.00 (cpu-arm) | 0.046 (whole/whole) | - | 450.1 | - | output_shape=100000x10 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'threshold': 0.01}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

## Not covered by this board

- Classical, wave 2: RadiusNeighbors, the preprocessing scalers, HDBSCAN's prediction data, Cholesky and the parallel_* and Distributed* wrappers are public and not raced here; taxi-derived time series are not used (the ARIMA and ExponentialSmoothing lanes fit seeded synthetic series, as the repo's own ARIMA quality work does).
- Classical, wave 2, not planned on this vendor: cuML GaussianMixture, GaussianProcessRegressor/Classifier, Nystroem, RBFSampler: cuML 26.8.0 has none; scikit-learn on the CPU is the arm.
- Classical, wave 2, not planned on this vendor: faiss-gpu: no pinned PyPI wheel for this image; cuVS ivf_flat from the pinned rapids set (cuvs-cu12==26.8.1) is the IVF-Flat GPU arm.
- Classical, wave 2, not planned on this vendor: umap-learn and faiss-cpu: not installed on NVIDIA; cuML UMAP and cuVS are the arms.
- Classical, wave 2, not planned on this vendor: cuML SpectralClustering/SpectralEmbedding: present only in newer cuML; if the pinned 26.8.0 lacks them the arm refuses by name and scikit-learn stands beside it.
- Classical, wave 2, not planned on this vendor: cuML on elasticnet: refused, different algorithm: ours' coordinate descent forms X^T X once and sweeps the Gram; cuML's cd sweeps every row per coordinate and refuses precompute (UnsupportedOnGPU) (2026-10-13 audit); scikit-learn on the CPU is the arm.
- Classical, wave 2, not planned on this vendor: cuML on lasso: refused, different algorithm: ours' coordinate descent forms X^T X once and sweeps the Gram; cuML's cd sweeps every row per coordinate and refuses precompute (UnsupportedOnGPU) (2026-10-13 audit); scikit-learn on the CPU is the arm.
- Classical, wave 2, not planned on this vendor: cuML on linearsvc: refused, different algorithm and model: liblinear (dual='auto' takes its primal trust-region Newton here) with a penalized intercept; ours and cuML L-BFGS on the primal with an unpenalized intercept (2026-10-13 audit); scikit-learn on the CPU is the arm.
- Classical, wave 2, not planned on this vendor: cuML on linearsvr: refused, different algorithm and model: liblinear dual coordinate descent with a penalized intercept; ours and cuML L-BFGS on the primal with an unpenalized intercept (2026-10-13 audit); scikit-learn on the CPU is the arm.
- Classical, wave 2, not planned on this vendor: cuML on ridge: refused, different algorithm: ours solves the centered normal equations by Cholesky (eigendecomposition only when its trust gate rejects the factor); cuML Ridge has no Cholesky solver ('eig' also forms U = A V) (2026-10-13 audit); scikit-learn on the CPU is the arm.
- Classical, wave 2, not planned on this vendor: cuML on tsvd: refused, different algorithm: ours 'covariance_eigh' (the eigendecomposition of X^T X); scikit-learn TruncatedSVD has only 'arpack' (Lanczos on X) and 'randomized', and no CPU library has ours' algorithm (2026-10-13 audit); scikit-learn on the CPU is the arm.
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
- Neural, not planned on this vendor: torch-compile-* on mamba1-forward and mamba1-infer: the only torch Mamba-1 twin is the pure-PyTorch reference scan, a per-token Python loop that torch.compile would unroll L times; mamba1 races the eager arms only
- Neural, not planned on this vendor: gemm-bf16 races torch's bf16 settings only (bf16 operands, fp32 accumulate)
- Neural, not planned on this vendor: mamba1-decode, mamba2-decode, mamba3-decode and samba-decode race ours alone: the repo's torch Mamba references are full-sequence scans with no carried-state decode step (a torch decode twin is not written yet)
- Neural, not planned on this vendor: torch-compile-* on transformer-decode: the twin is a per-token loop over a growing KV cache; transformer-decode races the eager arms only
- Neural, not planned on this vendor: mamba-ssm bf16: our Mamba blocks are float32, so mamba_ssm races in float32 (and its TF32 setting on NVIDIA)

