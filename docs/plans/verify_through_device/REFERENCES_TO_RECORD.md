# Verify through `device=`: references to record (vd-verify, lane vdv-ledger, 2026-11-24)

The orchestrator records these once, at the release's one verification, from the built wheels of the frozen
source. No value is written here or anywhere by a lane, and none is ever taken from a failing side
(AGENTS.md part 4; `docs/plans/VERIFY_THROUGH_DEVICE.md` item 5). The reference table
(`python/mojolearn/verify_reference/table.json`) keeps one reference per lane, fixture and part across every column.
So the release records every lane on every column anyway, and reuse stays opt-in and unused. This list names what
must come out of that record non-OWED, and what is new.

Static: lane names and part classes are read from source (`@lane(...)`, `_fit(...)`, `_batch_decl(...)`,
`_one_sided(...)`) on `lane/vd-verify`; nothing was run. vdv-sabotage's lanes are left out (vd-verify adds them).

## How each column is recorded (commands from vdv-harness-RESULT.md and vdv-tools-RESULT.md)

Fixtures: all nine (`tools/identity_break.py` `FIXTURES`: base, ties, hashed, wide, denormal, denormal_ftz, dupes,
odd, negative), the default size, `--repeats 1`, every part (train, infer, model, batch, and the property parts
batchgrad / batchscale / ragged / stepfull, which the harness collects by default; their n/a words come from the
registries).

| column | where | command |
|---|---|---|
| CPU (recorded first) | GitHub CPU runner, built checkout of the frozen commit | `bash tools/record_identity_column.sh cpu /abs/out` (harness `--side cpu`: every public call with `device="cpu"`, no `MOJOLEARN_VENDOR=cpu` process). Shards: `MOJOLEARN_NUMERIC_MODE=identical python tools/identity_break.py --side cpu --jobs N --json cpu.json`, with N = the allocation's cores (`_allocated_cores()`) |
| NVIDIA, plus a CPU record from the same process | the L40S (RunPod), built checkout | `WITH_CPU=1 bash tools/record_identity_column.sh nvidia-l40s-sm_89 /abs/out` (harness `identity_break.py --side both --require-backend cuda --vendor <label> --json <gpu> --cpu-json <cpu>`); NVIDIA only: `bash tools/record_identity_column.sh nvidia-l40s-sm_89 /abs/out` (`--side gpu`) |
| AMD, Apple (the release's one verification) | MI300X; M2 Pro | the same recorder with `amd-mi300x-gfx942` / `apple-<chip>`. Part 8 of AGENTS.md (NVIDIA only) covers measurement, not the release verification |
| admission | the Mac, in the branch that carries the new table | `bash tools/admit_identity_columns.sh <records dir...>` (checks every record with `tools/identity_columns.py check`, builds the table, runs the self-test's `ols/base` on both sides) |

The CPU record from a GPU box (`WITH_CPU=1`) is new. Its admission was written for CPU-only records (vdv-tools doubt).
Taking the CPU column from the GitHub runner keeps the AGENTS.md rule ("recorded first, on GitHub runners").

## A. The 70 lanes whose reference was OWED (inventory, gate-6 audit `reference_evidence`, status OWED)

Source: `~/mojolearn-evidence/w3-integrate-6-gate/gate6-3cda/gate6-3cda8a05b3/audit.json` (70 OWED, 512
RETAINED; every OWED lane reads "no numeric reference part retained for this lane"). 66 exist at HEAD. The four
`par-*` lanes (`par-causal-lm`, `par-mlp`, `par-samba`, `par-samba-clip`) were deleted on 2026-10-31 (TOMBSTONES
`MOJOLEARN_PAR_VERIFY`); `par-` is in `RECORD_EXCLUDED_PREFIXES` and `RETIRED_LANE_PREFIXES`, so nothing is recorded
for them. Their single-device replacements are the `psd-*` lanes in section D.

| lane | file | parts | batch part | CPU column | NVIDIA column |
|---|---|---|---|---|---|
| `forest-shard` | coverage.py | train, stepfull | n/a:coverage-probe | record | record |
| `gbdt-cat-bootstrap` | identity_break.py | train, stepfull | batch protocol | record | record |
| `gbdt-cat-noise` | identity_break.py | train, stepfull | batch protocol | record | record |
| `gbdt-cat-nonsym` | identity_break.py | train, infer, model, stepfull | batch protocol | record | record |
| `gbdt-cat-ordered` | identity_break.py | train, stepfull | batch protocol | record | record |
| `gbdt-cat-rest` | identity_break.py | train, infer, model, stepfull | batch protocol | record | record |
| `gbdt-cat-rmse` | identity_break.py | train, stepfull | batch protocol | record | record |
| `gbdt-eval-gaps` | identity_break.py | train, infer, model, stepfull | batch protocol | record | record |
| `gbdt-multi-walk` | identity_break.py | train, infer, model, stepfull | batch protocol | record | record |
| `gbdt-nonsym-rest` | identity_break.py | train, infer, model, stepfull | batch protocol | record | record |
| `gbdt-ordered-weights` | identity_break.py | train, infer, model, stepfull | batch protocol | record | record |
| `gbdt-pointwise-arms` | identity_break.py | train, infer, model, stepfull | batch protocol | record | record |
| `gbdt-pointwise-multiclass` | identity_break.py | train, infer, model, stepfull | batch protocol | record | record |
| `gbdt-pointwise-rest` | identity_break.py | train, infer, model, stepfull | batch protocol | record | record |
| `gbdt-rank-bootstrap` | identity_break.py | train, stepfull | batch protocol | record | record |
| `gbdt-rank-leaves` | identity_break.py | train, stepfull | batch protocol | record | record |
| `gbdt-rank-pairs` | identity_break.py | train, stepfull | batch protocol | record | record |
| `gbdt-rank-pointwise-arms` | identity_break.py | train, stepfull | batch protocol | record | record |
| `gbdt-rest-gaps` | identity_break.py | train, infer, model, stepfull | batch protocol | record | record |
| `gbdt-shape-gaps` | identity_break.py | train, infer, model, stepfull | batch protocol | record | record |
| `gbdt-simple-leaves` | identity_break.py | train, infer, model, stepfull | batch protocol | record | record |
| `gbdt-symmetric-rest` | identity_break.py | train, infer, model, stepfull | batch protocol | record | record |
| `gbdt-weighted-nonsym` | identity_break.py | train, stepfull | batch protocol | record | record |
| `gbdt-weighted-rank` | identity_break.py | train, infer, model, stepfull | batch protocol | record | record |
| `gbdt-weighted-rest` | identity_break.py | train, stepfull | batch protocol | record | record |
| `gbdt-weighted-special` | identity_break.py | train, infer, model, stepfull | batch protocol | record | record |
| `gbdt-weighted-symmetric` | identity_break.py | train, stepfull | batch protocol | record | record |
| `ivf-distributed` | coverage.py | train, stepfull | n/a:coverage-probe | record | record |
| `kernel-ridge-wide` | coverage.py | train, stepfull | n/a:coverage-probe | record | record |
| `neighbors-reference-shard` | coverage.py | train, stepfull | n/a:coverage-probe | record | record |
| `par-causal-lm` | gone (retired `par-*`) | - | - | nothing to record | nothing to record |
| `par-mlp` | gone (retired `par-*`) | - | - | nothing to record | nothing to record |
| `par-samba` | gone (retired `par-*`) | - | - | nothing to record | nothing to record |
| `par-samba-clip` | gone (retired `par-*`) | - | - | nothing to record | nothing to record |
| `samba-decode-session` | coverage.py | train, stepfull | n/a:fixed-session-batch (vdv-fin) | record | record |
| `sequence-attention-cross` | nn_attention.py | train, stepfull | n/a:training-step | record | record |
| `sequence-attention-encoder` | nn_attention.py | train, stepfull | n/a:training-step | record | record |
| `sequence-blocks-bigru-lengths` | nn_recurrent.py | train, stepfull | n/a:training-step | record | record |
| `sequence-blocks-bilstm` | nn_recurrent.py | train, stepfull | n/a:training-step | record | record |
| `sequence-blocks-cnn1d` | sequence.py | train, stepfull | n/a:training-step | record | record |
| `sequence-blocks-gru-lengths` | sequence.py | train, stepfull | n/a:training-step | record | record |
| `sequence-blocks-inceptiontime` | sequence.py | train, stepfull | n/a:training-step | record | record |
| `sequence-blocks-lstm-lengths` | sequence.py | train, stepfull | n/a:training-step | record | record |
| `sequence-blocks-resnet1d` | sequence.py | train, stepfull | n/a:training-step | record | record |
| `sequence-composed-activations` | nn_activations.py | train, stepfull | n/a:training-step | record | record |
| `sequence-composed-conv1d` | sequence.py | train, stepfull | n/a:training-step | record | record |
| `sequence-composed-conv3d` | nn_conv3d.py | train, stepfull | n/a:training-step | record | record |
| `sequence-composed-log-softmax-ctc` | nn_losses_2.py | train, stepfull | n/a:training-step | record | record |
| `sequence-composed-losses` | nn_losses.py | train, stepfull | n/a:training-step | record | record |
| `sequence-composed-mamba1` | sequence.py | train, stepfull | n/a:training-step | record | record |
| `sequence-composed-mamba2` | sequence.py | train, stepfull | n/a:training-step | record | record |
| `sequence-composed-mamba3` | sequence.py | train, stepfull | n/a:training-step | record | record |
| `sequence-composed-pool3d` | nn_conv3d.py | train, stepfull | n/a:training-step | record | record |
| `sequence-composed-recurrent` | sequence.py | train, stepfull | n/a:training-step | record | record |
| `sequence-composed-transformer` | sequence.py | train, stepfull | n/a:training-step | record | record |
| `sequence-nnconv-pads` | nn_conv.py | train, stepfull | n/a:training-step | record | record |
| `sequence-nnconv-pools` | nn_conv.py | train, stepfull | n/a:training-step | record | record |
| `sequence-nnconv-shuffle` | nn_conv.py | train, stepfull | n/a:training-step | record | record |
| `sequence-nnconv-transpose` | nn_conv.py | train, stepfull | n/a:training-step | record | record |
| `sequence-nnconv-upsample` | nn_conv.py | train, stepfull | n/a:training-step | record | record |
| `sequence-norms-instance` | nn_norms.py | train, stepfull | n/a:training-step | record | record |
| `sequence-norms-resnet1d-gn` | nn_norms.py | train, stepfull | n/a:training-step | record | record |
| `sequence-optim-adagrad` | nn_optim.py | train, stepfull | n/a:training-trajectory (vdv-fin) | record | record |
| `sequence-optim-lamb` | nn_optim.py | train, stepfull | n/a:training-trajectory (vdv-fin) | record | record |
| `sequence-optim-rmsprop` | nn_optim.py | train, stepfull | n/a:training-trajectory (vdv-fin) | record | record |
| `trees-gbdt-importances` | trees.py | train, infer, model, stepfull | batch protocol | record | record |
| `x-lasso-short` | coverage.py | train, stepfull | n/a:coverage-probe | record | record |
| `x-ols-square` | coverage.py | train, stepfull | n/a:coverage-probe | record | record |
| `x-prep-rfe-no-round` | prep.py | train, stepfull | n/a:fit-probe (vdv-fin) | record | record |
| `x-ridge-rank-deficient` | coverage.py | train, stepfull | n/a:coverage-probe | record | record |

## B. The `ols/base` cell (w3-fix-5-RESULT.md)

- Cause: 0d1b8c4dd8 changed the archive format word (`mojolearn-linear-1` -> `-2`, `core/native_archive_write.mojo:198`),
  so the saved-model `model` part of `ols` moved. `ols/base` reads 1 DIVERGENT part on the CPU build;
  `test_verify_all.py::test_all_checks_batch_reference_without_opt_in` and
  `::test_the_printed_verdict_carries_its_own_scope_end_to_end` stay red until it is re-recorded.
- Lane `ols` (identity_break.py), all nine fixtures, parts train, infer, model, batch protocol, stepfull. Both columns:
  recorded by the full CPU and NVIDIA records above (one reference spans every column), then admitted.
- Confirm the part first, on the CPU build: `python -m mojolearn verify --lanes ols --fixtures base --no-models --json`
  (w3-fix-5 inferred the `model` part and did not confirm it).

## C. The public endpoint reference (neural, training), `verify --self-test` (w3-fix-5-RESULT.md)

- Stale since the nn-* fixture lanes and r5-proving d83b80fb89 (named in f36d60dc61). The self-test reads OWED on
  every vendor until it is regenerated.
- Both columns are captured by `tools/verify_api_endpoints.py --side cpu|gpu|both` (vdv-fin, 2026-11-24): one file
  per side, each side in the same process through the device view (`capture(side=)`, as
  `_verify_api_runner.verify` runs them). A CPU-only installation records the CPU side only. `admit()` /
  `tools/verify_api_reference.py --cpu` admit the CPU column: it must reproduce the NVIDIA bytes on every endpoint
  it executes; the device-only endpoints it records `not applicable` are not missing from it. With the CPU column
  admitted, the CPU side of `verify --self-test` is judged (MATCH / FAILED) instead of UNTESTED.
- CPU column, recorded first, on the GitHub CPU runner from the built checkout of the frozen commit:
  `MOJOLEARN_NUMERIC_MODE=identical python tools/verify_api_endpoints.py --side cpu --out api-cpu.json`
  (keep `api-cpu.json` as the run's artifact).
- NVIDIA column plus a CPU capture from the same process, on the L40S from the built wheel:
  `MOJOLEARN_NUMERIC_MODE=identical python tools/verify_api_endpoints.py --side both --out api-cuda.json --cpu-out api-cpu-l40s.json`.
- Admission, on the L40S (the package must import; the GitHub runner's `api-cpu.json` copied beside the captures):
  `python tools/verify_api_reference.py --cpu api-cpu.json --cuda api-cuda.json --out python/mojolearn/_verification_api_reference.py`,
  then commit the generated file. `api-cpu-l40s.json` is the cross-check of the CPU column on a GPU box: admit it
  as `--cpu` in a second, uncommitted call (`--out /abs/check.py`); it must admit with `cpu` validated. AMD / Apple:
  add `--hip` / `--metal` GPU-side captures (`--side gpu --out api-hip.json`) to the same `verify_api_reference.py`
  call at the release.

## D. Lanes changed in place by vd-verify (their new bytes come with the full record)

- `sequence-device-tensor`, `sequence-adafactor-device`, `sequence-lamb-device` (coverage.py, vdv-fragfix): the CPU
  side now passes host arrays instead of uploading to the GPU.
- `public-numeric-metadata` (now one-sided, no GPU side), `saved-model-host-infer` (GPU column becomes n/a:one-sided),
  `training-device-cross-entropy`, `optim-resident-io`, `sequence-device-release` (CPU column n/a:one-sided): the CPU
  record now carries `n/a:one-sided` cells instead of leaving them out (vdv-tools).
- `trees-stacking-clf` (`lacking` part, revision key `lacking-fold-20261124-1`, w3-fix-5): its input rows changed, and
  the revision has not been recorded in any column.
- Lanes rewritten to reach the product through the device view (vdv-fragfix): `ordered-gradient-sum`,
  `mamba1-decode-session`, `transformer-decode-session`, `cross-vendor-fold`, `safetensors-widen-f16`,
  `neural-trainer-unfused`, `forest-shard`, `neighbors-reference-shard`, `ivf-distributed`. Five of them stay
  refused until vd-verify decides (UNMAPPED.md section 3).
- The tape light reference and all witnesses (vdv-harness: case sources changed to `ops = _ops()`).
- The recipe and check records (vdv-check): the formats gain `gpu_side` / `sides` and lose `process_wall_s`. No
  reference table exists for them.

## E. Every new lane (381), by fragment

All of them are new in `lane/vd-verify` against `059785de37`, all nine fixtures. "record" means the column takes a
value at the release. "n/a:one-sided" means the lane declares that side missing (`ONE_SIDED`). The 12 two-sided `psd-*`
lanes can take their CPU column on a GPU box now that vdv-gaps routes a `device='cpu'` member to host workers. A
CPU-only install also gives it. `psd-causal-lm` and `psd-causal-lm-text` are one-sided (no CPU side).


#### `tools/identity_lanes/checks_generic.py` (9)

| lane | file | parts | batch part | CPU column | NVIDIA column |
|---|---|---|---|---|---|
| `check-function` | checks_generic.py | train, stepfull | n/a:verification-check | record | record |
| `check-generic-cluster` | checks_generic.py | train, stepfull | n/a:verification-check | record | record |
| `check-generic-decomp` | checks_generic.py | train, stepfull | n/a:verification-check | record | record |
| `check-generic-linear` | checks_generic.py | train, stepfull | n/a:verification-check | record | record |
| `check-generic-neighbors` | checks_generic.py | train, stepfull | n/a:verification-check | record | record |
| `check-generic-other` | checks_generic.py | train, stepfull | n/a:verification-check | record | record |
| `check-generic-prep` | checks_generic.py | train, stepfull | n/a:verification-check | record | record |
| `check-generic-sequence` | checks_generic.py | train, stepfull | n/a:verification-check | record | record |
| `check-generic-trees` | checks_generic.py | train, stepfull | n/a:verification-check | record | record |

#### `tools/identity_lanes/functions.py` (16)

| lane | file | parts | batch part | CPU column | NVIDIA column |
|---|---|---|---|---|---|
| `fn-chi2` | functions.py | train, stepfull | n/a:function | record | record |
| `fn-connected-components` | functions.py | train, stepfull | n/a:function | record | record |
| `fn-f-classif` | functions.py | train, stepfull | n/a:function | record | record |
| `fn-f-regression` | functions.py | train, stepfull | n/a:function | record | record |
| `fn-kpss-test` | functions.py | train, stepfull | n/a:function | record | record |
| `fn-mutual-info-classif` | functions.py | train, stepfull | n/a:function | record | record |
| `fn-mutual-info-regression` | functions.py | train, stepfull | n/a:function | record | record |
| `fn-r-regression` | functions.py | train, stepfull | n/a:function | record | record |
| `fn-refine` | functions.py | train, stepfull | n/a:function | record | record |
| `fn-resample-bootstrap` | functions.py | train, stepfull | n/a:function | record | record |
| `fn-resample-indices` | functions.py | train, stepfull | n/a:function | record | record |
| `fn-resample-monte-carlo` | functions.py | train, stepfull | n/a:function | record | record |
| `fn-resample-permutation-test` | functions.py | train, stepfull | n/a:function | record | record |
| `fn-resample-resample` | functions.py | train, stepfull | n/a:function | record | record |
| `fn-select-d` | functions.py | train, stepfull | n/a:function | record | record |
| `fn-spectral-embedding` | functions.py | train, stepfull | n/a:function | record | record |

#### `tools/identity_lanes/functions_host.py` (3)

| lane | file | parts | batch part | CPU column | NVIDIA column |
|---|---|---|---|---|---|
| `fn-catboost-auto-learning-rate` | functions_host.py | train, stepfull | n/a:function | record | n/a:one-sided |
| `fn-host-predict` | functions_host.py | train, stepfull | n/a:function | record | n/a:one-sided |
| `fn-jl-min-dim` | functions_host.py | train, stepfull | n/a:function | record | n/a:one-sided |

#### `tools/identity_lanes/functions_linalg.py` (20)

| lane | file | parts | batch part | CPU column | NVIDIA column |
|---|---|---|---|---|---|
| `fn-linalg-dequantize-int15` | functions_linalg.py | train, stepfull | n/a:function | record | record |
| `fn-linalg-dequantize-int8` | functions_linalg.py | train, stepfull | n/a:function | record | record |
| `fn-linalg-eigh` | functions_linalg.py | train, stepfull | n/a:function | record | record |
| `fn-linalg-from-bf16` | functions_linalg.py | train, stepfull | n/a:function | record | record |
| `fn-linalg-matmul` | functions_linalg.py | train, stepfull | n/a:function | record | record |
| `fn-linalg-matmul-bf16` | functions_linalg.py | train, stepfull | n/a:function | record | record |
| `fn-linalg-matmul-int15` | functions_linalg.py | train, stepfull | n/a:function | record | record |
| `fn-linalg-matmul-int8` | functions_linalg.py | train, stepfull | n/a:function | record | record |
| `fn-linalg-qr` | functions_linalg.py | train, stepfull | n/a:function | record | record |
| `fn-linalg-quantize-int15` | functions_linalg.py | train, stepfull | n/a:function | record | record |
| `fn-linalg-quantize-int8` | functions_linalg.py | train, stepfull | n/a:function | record | record |
| `fn-linalg-svd` | functions_linalg.py | train, stepfull | n/a:function | record | record |
| `fn-linalg-svdvals` | functions_linalg.py | train, stepfull | n/a:function | record | record |
| `fn-linalg-to-bf16` | functions_linalg.py | train, stepfull | n/a:function | record | record |
| `fn-lstsq` | functions_linalg.py | train, stepfull | n/a:function | record | record |
| `fn-lu-factor` | functions_linalg.py | train, stepfull | n/a:function | record | record |
| `fn-lu-solve` | functions_linalg.py | train, stepfull | n/a:function | record | record |
| `fn-randomized-svd` | functions_linalg.py | train, stepfull | n/a:function | record | record |
| `fn-solve` | functions_linalg.py | train, stepfull | n/a:function | record | record |
| `fn-sparse-encode` | functions_linalg.py | train, stepfull | n/a:function | record | record |

#### `tools/identity_lanes/functions_metrics.py` (69)

| lane | file | parts | batch part | CPU column | NVIDIA column |
|---|---|---|---|---|---|
| `fn-metrics-accuracy-score` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-adjusted-mutual-info-score` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-adjusted-rand-score` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-auc` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-average-precision-score` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-balanced-accuracy-score` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-brier-score-loss` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-calinski-harabasz-score` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-class-likelihood-ratios` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-classification-report` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-cohen-kappa-score` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-completeness-score` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-confusion-matrix` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-contingency-matrix` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-coverage-error` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-d2-absolute-error-score` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-d2-brier-score` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-d2-log-loss-score` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-d2-pinball-score` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-d2-tweedie-score` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-davies-bouldin-score` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-dcg-score` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-det-curve` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-entropy` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-explained-variance-score` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-f1-score` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-fbeta-score` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-fowlkes-mallows-score` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-hamming-loss` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-hinge-loss` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-homogeneity-completeness-v-measure` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-homogeneity-score` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-jaccard-score` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-kl-divergence` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-label-ranking-average-precision-score` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-label-ranking-loss` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-log-loss` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-matthews-corrcoef` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-max-error` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-mean-absolute-error` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-mean-absolute-percentage-error` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-mean-gamma-deviance` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-mean-pinball-loss` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-mean-poisson-deviance` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-mean-squared-error` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-mean-squared-log-error` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-mean-tweedie-deviance` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-median-absolute-error` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-multilabel-confusion-matrix` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-mutual-info-score` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-ndcg-score` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-normalized-mutual-info-score` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-pair-confusion-matrix` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-precision-recall-curve` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-precision-recall-fscore-support` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-precision-score` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-r2-score` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-rand-score` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-recall-score` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-roc-auc-score` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-roc-curve` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-root-mean-squared-error` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-root-mean-squared-log-error` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-silhouette-samples` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-silhouette-score` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-top-k-accuracy-score` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-trustworthiness` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-v-measure-score` | functions_metrics.py | train, stepfull | n/a:function | record | record |
| `fn-metrics-zero-one-loss` | functions_metrics.py | train, stepfull | n/a:function | record | record |

#### `tools/identity_lanes/functions_msel.py` (11)

| lane | file | parts | batch part | CPU column | NVIDIA column |
|---|---|---|---|---|---|
| `fn-msel-check-cv` | functions_msel.py | train, stepfull | n/a:function | record | record |
| `fn-msel-cross-val-predict` | functions_msel.py | train, stepfull | n/a:function | record | record |
| `fn-msel-cross-val-score` | functions_msel.py | train, stepfull | n/a:function | record | record |
| `fn-msel-cross-validate` | functions_msel.py | train, stepfull | n/a:function | record | record |
| `fn-msel-default-folds` | functions_msel.py | train, stepfull | n/a:function | record | record |
| `fn-msel-learning-curve` | functions_msel.py | train, stepfull | n/a:function | record | record |
| `fn-msel-permutation-test-score` | functions_msel.py | train, stepfull | n/a:function | record | record |
| `fn-msel-scorers` | functions_msel.py | train, stepfull | n/a:function | record | record |
| `fn-msel-split-descriptor` | functions_msel.py | train, stepfull | n/a:function | record | record |
| `fn-msel-train-test-split` | functions_msel.py | train, stepfull | n/a:function | record | record |
| `fn-msel-validation-curve` | functions_msel.py | train, stepfull | n/a:function | record | record |

#### `tools/identity_lanes/functions_training.py` (15)

| lane | file | parts | batch part | CPU column | NVIDIA column |
|---|---|---|---|---|---|
| `fn-layer-norm-backward` | functions_training.py | train, stepfull | n/a:function | record | record |
| `fn-layer-norm-forward` | functions_training.py | train, stepfull | n/a:function | record | record |
| `fn-training-accumulate-grads` | functions_training.py | train, stepfull | n/a:function | record | record |
| `fn-training-chunked-lm-head-loss` | functions_training.py | train, stepfull | n/a:function | record | record |
| `fn-training-clip-grad-norm` | functions_training.py | train, stepfull | n/a:function | record | record |
| `fn-training-cross-entropy` | functions_training.py | train, stepfull | n/a:function | record | record |
| `fn-training-embedding-backward` | functions_training.py | train, stepfull | n/a:function | record | record |
| `fn-training-embedding-forward` | functions_training.py | train, stepfull | n/a:function | record | record |
| `fn-training-linear-backward` | functions_training.py | train, stepfull | n/a:function | record | record |
| `fn-training-linear-forward` | functions_training.py | train, stepfull | n/a:function | record | record |
| `fn-training-mlp-inference-sessions` | functions_training.py | train, stepfull | n/a:function | record | record |
| `fn-training-residual-dropout` | functions_training.py | train, stepfull | n/a:function | record | record |
| `fn-training-residual-dropout-backward` | functions_training.py | train, stepfull | n/a:function | record | record |
| `fn-training-rms-norm-backward` | functions_training.py | train, stepfull | n/a:function | record | record |
| `fn-training-rms-norm-forward` | functions_training.py | train, stepfull | n/a:function | record | record |

#### `tools/identity_lanes/generic.py` (32)

| lane | file | parts | batch part | CPU column | NVIDIA column |
|---|---|---|---|---|---|
| `generic-attrs-cluster` | generic.py | train, stepfull | n/a:generic-mechanism | record | record |
| `generic-attrs-decomp` | generic.py | train, stepfull | n/a:generic-mechanism | record | record |
| `generic-attrs-linear` | generic.py | train, stepfull | n/a:generic-mechanism | record | record |
| `generic-attrs-neighbors` | generic.py | train, stepfull | n/a:generic-mechanism | record | record |
| `generic-attrs-other` | generic.py | train, stepfull | n/a:generic-mechanism | record | record |
| `generic-attrs-prep` | generic.py | train, stepfull | n/a:generic-mechanism | record | record |
| `generic-attrs-sequence` | generic.py | train, stepfull | n/a:generic-mechanism | record | record |
| `generic-attrs-trees` | generic.py | train, stepfull | n/a:generic-mechanism | record | record |
| `generic-digest-cluster` | generic.py | train, stepfull | n/a:generic-mechanism | record | record |
| `generic-digest-decomp` | generic.py | train, stepfull | n/a:generic-mechanism | record | record |
| `generic-digest-linear` | generic.py | train, stepfull | n/a:generic-mechanism | record | record |
| `generic-digest-neighbors` | generic.py | train, stepfull | n/a:generic-mechanism | record | record |
| `generic-digest-other` | generic.py | train, stepfull | n/a:generic-mechanism | record | record |
| `generic-digest-prep` | generic.py | train, stepfull | n/a:generic-mechanism | record | record |
| `generic-digest-sequence` | generic.py | train, stepfull | n/a:generic-mechanism | record | record |
| `generic-digest-trees` | generic.py | train, stepfull | n/a:generic-mechanism | record | record |
| `generic-params-cluster` | generic.py | train, stepfull | n/a:generic-mechanism | record | record |
| `generic-params-decomp` | generic.py | train, stepfull | n/a:generic-mechanism | record | record |
| `generic-params-linear` | generic.py | train, stepfull | n/a:generic-mechanism | record | record |
| `generic-params-neighbors` | generic.py | train, stepfull | n/a:generic-mechanism | record | record |
| `generic-params-other` | generic.py | train, stepfull | n/a:generic-mechanism | record | record |
| `generic-params-prep` | generic.py | train, stepfull | n/a:generic-mechanism | record | record |
| `generic-params-sequence` | generic.py | train, stepfull | n/a:generic-mechanism | record | record |
| `generic-params-trees` | generic.py | train, stepfull | n/a:generic-mechanism | record | record |
| `generic-pickle-cluster` | generic.py | train, stepfull | n/a:generic-mechanism | record | record |
| `generic-pickle-decomp` | generic.py | train, stepfull | n/a:generic-mechanism | record | record |
| `generic-pickle-linear` | generic.py | train, stepfull | n/a:generic-mechanism | record | record |
| `generic-pickle-neighbors` | generic.py | train, stepfull | n/a:generic-mechanism | record | record |
| `generic-pickle-other` | generic.py | train, stepfull | n/a:generic-mechanism | record | record |
| `generic-pickle-prep` | generic.py | train, stepfull | n/a:generic-mechanism | record | record |
| `generic-pickle-sequence` | generic.py | train, stepfull | n/a:generic-mechanism | record | record |
| `generic-pickle-trees` | generic.py | train, stepfull | n/a:generic-mechanism | record | record |

#### `tools/identity_lanes/m_predict.py` (49)

| lane | file | parts | batch part | CPU column | NVIDIA column |
|---|---|---|---|---|---|
| `agglomerative-fit-predict` | m_predict.py | train, stepfull | n/a:fit-endpoint (vdv-fin) | record | record |
| `cholesky-profile-jitter` | m_predict.py | train, stepfull | n/a:scalar-reduction (vdv-fin) | record | record |
| `dbscan-fit-predict` | m_predict.py | train, stepfull | n/a:fit-endpoint (vdv-fin) | record | record |
| `et-clf-methods` | m_predict.py | train, infer, model, stepfull | batch protocol (vdv-fin) | record | record |
| `gmm-fit-predict` | m_predict.py | train, stepfull | n/a:fit-endpoint (vdv-fin) | record | record |
| `gmm-methods` | m_predict.py | train, infer, model, stepfull | n/a:scalar-reduction (vdv-fin) | record | record |
| `gpc-methods` | m_predict.py | train, infer, model, stepfull | batch protocol (vdv-fin) | record | record |
| `hdbscan-fit-predict` | m_predict.py | train, stepfull | n/a:fit-endpoint (vdv-fin) | record | record |
| `holtwinters-methods` | m_predict.py | train, infer, model, stepfull | n/a:whole-series (vdv-fin) | record | record |
| `iforest-decision-function` | m_predict.py | train, infer, model, stepfull | batch protocol (vdv-fin) | record | record |
| `iforest-fit-predict` | m_predict.py | train, stepfull | n/a:fit-endpoint (vdv-fin) | record | record |
| `m-autoarima-predict` | m_predict.py | train, stepfull | n/a:whole-series (vdv-fin) | record | record |
| `m-dt-reg-methods` | m_predict.py | train, infer, model, stepfull | batch protocol (vdv-fin) | record | record |
| `m-et-reg-methods` | m_predict.py | train, infer, model, stepfull | batch protocol (vdv-fin) | record | record |
| `m-gbdt-readbacks` | m_predict.py | train, infer, model, stepfull | batch protocol (vdv-fin) | record | record |
| `m-gridsearch-methods` | m_predict.py | train, infer, model, stepfull | batch protocol (vdv-fin) | record | record |
| `m-isomap-reconstruction-error` | m_predict.py | train, stepfull | n/a:scalar-reduction (vdv-fin) | record | record |
| `m-kmeans-fit-predict` | m_predict.py | train, stepfull | n/a:fit-endpoint (vdv-fin) | record | record |
| `m-prophet-add-seasonality` | m_predict.py | train, stepfull | n/a:whole-series (vdv-fin) | record | record |
| `m-rf-methods` | m_predict.py | train, infer, model, stepfull | batch protocol (vdv-fin) | record | record |
| `m-robust-cov-methods` | m_predict.py | train, infer, model, stepfull | batch protocol (vdv-fin) | record | record |
| `public-theta-variants-forecast` | m_predict.py | train, stepfull | n/a:forecast-fit (vdv-fin) | record | record |
| `sequence-croston-forecast` | m_predict.py | train, stepfull | n/a:forecast-fit (vdv-fin) | record | record |
| `sequence-ets-forecast` | m_predict.py | train, stepfull | n/a:forecast-fit (vdv-fin) | record | record |
| `sequence-theta-forecast` | m_predict.py | train, stepfull | n/a:forecast-fit (vdv-fin) | record | record |
| `spectral-fit-predict` | m_predict.py | train, stepfull | n/a:fit-endpoint (vdv-fin) | record | record |
| `trees-adaboost-clf-methods` | m_predict.py | train, infer, model, stepfull | batch protocol (vdv-fin) | record | record |
| `trees-dt-clf-methods` | m_predict.py | train, infer, model, stepfull | batch protocol (vdv-fin) | record | record |
| `trees-voting-reg-methods` | m_predict.py | train, infer, model, stepfull | batch protocol (vdv-fin) | record | record |
| `x-cluster-affinity-propagation-fit-predict` | m_predict.py | train, stepfull | n/a:fit-endpoint (vdv-fin) | record | record |
| `x-cluster-bgmm-fit-predict` | m_predict.py | train, stepfull | n/a:fit-endpoint (vdv-fin) | record | record |
| `x-cluster-meanshift-fit-predict` | m_predict.py | train, stepfull | n/a:fit-endpoint (vdv-fin) | record | record |
| `x-cluster-optics-fit-predict` | m_predict.py | train, stepfull | n/a:fit-endpoint (vdv-fin) | record | record |
| `x-decomp-lda-perplexity` | m_predict.py | train, infer, model, stepfull | n/a:scalar-reduction (vdv-fin) | record | record |
| `x-decomp-robust-cov-fit-predict` | m_predict.py | train, stepfull | n/a:fit-endpoint (vdv-fin) | record | record |
| `x-isotonic-methods` | m_predict.py | train, infer, model, stepfull | batch protocol (vdv-fin) | record | record |
| `x-neighbors-lof-decision-function` | m_predict.py | train, infer, model, stepfull | batch protocol (vdv-fin) | record | record |
| `x-neighbors-louvain-fit-predict` | m_predict.py | train, stepfull | n/a:fit-endpoint (vdv-fin) | record | record |
| `x-neighbors-ocsvm-fit-predict` | m_predict.py | train, stepfull | n/a:fit-endpoint (vdv-fin) | record | record |
| `x-neighbors-ocsvm-score-samples` | m_predict.py | train, infer, model, stepfull | batch protocol (vdv-fin) | record | record |
| `x-prep-bernoulli-nb-methods` | m_predict.py | train, infer, model, stepfull | batch protocol (vdv-fin) | record | record |
| `x-prep-categorical-nb-methods` | m_predict.py | train, infer, model, stepfull | batch protocol (vdv-fin) | record | record |
| `x-prep-complement-nb-methods` | m_predict.py | train, infer, model, stepfull | batch protocol (vdv-fin) | record | record |
| `x-prep-gaussian-nb-methods` | m_predict.py | train, infer, model, stepfull | batch protocol (vdv-fin) | record | record |
| `x-prep-lda-methods` | m_predict.py | train, infer, model, stepfull | batch protocol (vdv-fin) | record | record |
| `x-prep-multinomial-nb-methods` | m_predict.py | train, infer, model, stepfull | batch protocol (vdv-fin) | record | record |
| `x-prep-qda-methods` | m_predict.py | train, infer, model, stepfull | batch protocol (vdv-fin) | record | record |
| `x-prep-score-edges-rfe-methods` | m_predict.py | train, infer, model, stepfull | batch protocol (vdv-fin) | record | record |
| `x-sgd-ocsvm-score-samples` | m_predict.py | train, infer, model, stepfull | batch protocol (vdv-fin) | record | record |

#### `tools/identity_lanes/m_score.py` (50)

| lane | file | parts | batch part | CPU column | NVIDIA column |
|---|---|---|---|---|---|
| `elasticnet-score` | m_score.py | train, stepfull | n/a:scalar-reduction (vdv-fin) | record | record |
| `gp-score` | m_score.py | train, stepfull | n/a:scalar-reduction (vdv-fin) | record | record |
| `kde-score` | m_score.py | train, stepfull | n/a:scalar-reduction (vdv-fin) | record | record |
| `kernel-ridge-score` | m_score.py | train, stepfull | n/a:scalar-reduction (vdv-fin) | record | record |
| `knn-clf-score` | m_score.py | train, stepfull | n/a:scalar-reduction (vdv-fin) | record | record |
| `lasso-score` | m_score.py | train, stepfull | n/a:scalar-reduction (vdv-fin) | record | record |
| `linear-svc-score` | m_score.py | train, stepfull | n/a:scalar-reduction (vdv-fin) | record | record |
| `linear-svr-score` | m_score.py | train, stepfull | n/a:scalar-reduction (vdv-fin) | record | record |
| `m-dart-score` | m_score.py | train, infer, model, stepfull | n/a:scalar-reduction (vdv-fin) | record | record |
| `m-huber-score` | m_score.py | train, stepfull | n/a:scalar-reduction (vdv-fin) | record | record |
| `m-kmeans-score` | m_score.py | train, stepfull | n/a:scalar-reduction (vdv-fin) | record | record |
| `m-knn-reg-score` | m_score.py | train, stepfull | n/a:scalar-reduction (vdv-fin) | record | record |
| `m-linear-ridge-score` | m_score.py | train, infer, model, stepfull | n/a:scalar-reduction (vdv-fin) | record | record |
| `m-multioutput-clf-score` | m_score.py | train, stepfull | n/a:scalar-reduction (vdv-fin) | record | record |
| `m-multioutput-ridge-score` | m_score.py | train, stepfull | n/a:scalar-reduction (vdv-fin) | record | record |
| `m-svr-gamma-scale-score` | m_score.py | train, stepfull | n/a:scalar-reduction (vdv-fin) | record | record |
| `qn-absolute-score` | m_score.py | train, stepfull | n/a:scalar-reduction (vdv-fin) | record | record |
| `trees-adaboost-reg-score` | m_score.py | train, stepfull | n/a:scalar-reduction (vdv-fin) | record | record |
| `trees-bagging-clf-score` | m_score.py | train, stepfull | n/a:scalar-reduction (vdv-fin) | record | record |
| `trees-bagging-reg-score` | m_score.py | train, stepfull | n/a:scalar-reduction (vdv-fin) | record | record |
| `trees-calibrated-score` | m_score.py | train, stepfull | n/a:scalar-reduction (vdv-fin) | record | record |
| `trees-onevsrest-score` | m_score.py | train, stepfull | n/a:scalar-reduction (vdv-fin) | record | record |
| `trees-stacking-clf-score` | m_score.py | train, stepfull | n/a:scalar-reduction (vdv-fin) | record | record |
| `x-ard-score` | m_score.py | train, stepfull | n/a:scalar-reduction (vdv-fin) | record | record |
| `x-bayes-ridge-score` | m_score.py | train, stepfull | n/a:scalar-reduction (vdv-fin) | record | record |
| `x-cluster-bgmm-score` | m_score.py | train, stepfull | n/a:scalar-reduction (vdv-fin) | record | record |
| `x-cluster-bisecting-kmeans-score` | m_score.py | train, stepfull | n/a:scalar-reduction (vdv-fin) | record | record |
| `x-cluster-minibatch-kmeans-score` | m_score.py | train, stepfull | n/a:scalar-reduction (vdv-fin) | record | record |
| `x-decomp-factor-analysis-score` | m_score.py | train, stepfull | n/a:scalar-reduction (vdv-fin) | record | record |
| `x-enet-cv-score` | m_score.py | train, stepfull | n/a:scalar-reduction (vdv-fin) | record | record |
| `x-glm-gamma-score` | m_score.py | train, stepfull | n/a:scalar-reduction (vdv-fin) | record | record |
| `x-glm-poisson-score` | m_score.py | train, stepfull | n/a:scalar-reduction (vdv-fin) | record | record |
| `x-glm-tweedie-score` | m_score.py | train, stepfull | n/a:scalar-reduction (vdv-fin) | record | record |
| `x-lars-score` | m_score.py | train, stepfull | n/a:scalar-reduction (vdv-fin) | record | record |
| `x-lasso-cv-score` | m_score.py | train, stepfull | n/a:scalar-reduction (vdv-fin) | record | record |
| `x-lasso-lars-score` | m_score.py | train, stepfull | n/a:scalar-reduction (vdv-fin) | record | record |
| `x-logistic-cv-score` | m_score.py | train, stepfull | n/a:scalar-reduction (vdv-fin) | record | record |
| `x-neighbors-label-propagation-score` | m_score.py | train, stepfull | n/a:scalar-reduction (vdv-fin) | record | record |
| `x-neighbors-label-spreading-score` | m_score.py | train, stepfull | n/a:scalar-reduction (vdv-fin) | record | record |
| `x-neighbors-nearest-centroid-score` | m_score.py | train, stepfull | n/a:scalar-reduction (vdv-fin) | record | record |
| `x-neighbors-svc-multiclass-score` | m_score.py | train, stepfull | n/a:scalar-reduction (vdv-fin) | record | record |
| `x-neighbors-svgp-score` | m_score.py | train, stepfull | n/a:scalar-reduction (vdv-fin) | record | record |
| `x-pa-clf-score` | m_score.py | train, stepfull | n/a:scalar-reduction (vdv-fin) | record | record |
| `x-pa-reg-score` | m_score.py | train, stepfull | n/a:scalar-reduction (vdv-fin) | record | record |
| `x-perceptron-score` | m_score.py | train, stepfull | n/a:scalar-reduction (vdv-fin) | record | record |
| `x-quantile-score` | m_score.py | train, stepfull | n/a:scalar-reduction (vdv-fin) | record | record |
| `x-ridge-clf-score` | m_score.py | train, stepfull | n/a:scalar-reduction (vdv-fin) | record | record |
| `x-ridge-cv-score` | m_score.py | train, stepfull | n/a:scalar-reduction (vdv-fin) | record | record |
| `x-sgd-clf-score` | m_score.py | train, stepfull | n/a:scalar-reduction (vdv-fin) | record | record |
| `x-sgd-reg-score` | m_score.py | train, stepfull | n/a:scalar-reduction (vdv-fin) | record | record |

#### `tools/identity_lanes/m_transform.py` (38)

| lane | file | parts | batch part | CPU column | NVIDIA column |
|---|---|---|---|---|---|
| `kernel-ridge-laplacian-fit-transform` | m_transform.py | train, stepfull | n/a:fit-endpoint (vdv-fin) | record | record |
| `m-stacking-reg-calls` | m_transform.py | train, infer, model, stepfull | batch protocol (vdv-fin) | record | record |
| `m-voting-clf-calls` | m_transform.py | train, infer, model, stepfull | batch protocol (vdv-fin) | record | record |
| `rbf-sampler-fit-transform` | m_transform.py | train, stepfull | n/a:fit-endpoint (vdv-fin) | record | record |
| `spectral-embedding-fit-transform` | m_transform.py | train, stepfull | n/a:fit-endpoint (vdv-fin) | record | record |
| `trees-random-embedding-fit-transform` | m_transform.py | train, stepfull | n/a:fit-endpoint (vdv-fin) | record | record |
| `trees-stacking-clf-fit-transform` | m_transform.py | train, stepfull | n/a:fit-endpoint (vdv-fin) | record | record |
| `trees-voting-reg-fit-transform` | m_transform.py | train, stepfull | n/a:fit-endpoint (vdv-fin) | record | record |
| `tsvd-inverse-fit-transform` | m_transform.py | train, stepfull | n/a:fit-endpoint (vdv-fin) | record | record |
| `umap-fit-transform` | m_transform.py | train, stepfull | n/a:fit-endpoint (vdv-fin) | record | record |
| `x-cluster-bisecting-kmeans-fit-calls` | m_transform.py | train, stepfull | n/a:fit-endpoint (vdv-fin) | record | record |
| `x-cluster-minibatch-kmeans-fit-calls` | m_transform.py | train, stepfull | n/a:fit-endpoint (vdv-fin) | record | record |
| `x-decomp-dict-learning-inverse` | m_transform.py | train, stepfull | n/a:fit-endpoint (vdv-fin) | record | record |
| `x-decomp-eigh-wide-kpca-fit` | m_transform.py | train, infer, model, stepfull | batch protocol (vdv-fin) | record | record |
| `x-decomp-ipca-fit-calls` | m_transform.py | train, stepfull | n/a:fit-endpoint (vdv-fin) | record | record |
| `x-decomp-lda-fit-calls` | m_transform.py | train, stepfull | n/a:fit-endpoint (vdv-fin) | record | record |
| `x-decomp-pls-fit-calls` | m_transform.py | train, stepfull | n/a:fit-endpoint (vdv-fin) | record | record |
| `x-decomp-sparse-pca-fit-calls` | m_transform.py | train, stepfull | n/a:fit-endpoint (vdv-fin) | record | record |
| `x-decomp-srp-fit-calls` | m_transform.py | train, stepfull | n/a:fit-endpoint (vdv-fin) | record | record |
| `x-isotonic-fit-transform` | m_transform.py | train, stepfull | n/a:fit-endpoint (vdv-fin) | record | record |
| `x-neighbors-additive-chi2-fit-transform` | m_transform.py | train, stepfull | n/a:fit-endpoint (vdv-fin) | record | record |
| `x-neighbors-knn-imputer-fit-transform` | m_transform.py | train, stepfull | n/a:fit-endpoint (vdv-fin) | record | record |
| `x-neighbors-poly-sketch-fit-transform` | m_transform.py | train, stepfull | n/a:fit-endpoint (vdv-fin) | record | record |
| `x-neighbors-skewed-chi2-fit-transform` | m_transform.py | train, stepfull | n/a:fit-endpoint (vdv-fin) | record | record |
| `x-prep-binarizer-fit-transform` | m_transform.py | train, stepfull | n/a:fit-endpoint (vdv-fin) | record | record |
| `x-prep-encoder-categories-fit-transform` | m_transform.py | train, stepfull | n/a:fit-endpoint (vdv-fin) | record | record |
| `x-prep-inverse-transforms-fit-transform` | m_transform.py | train, stepfull | n/a:fit-endpoint (vdv-fin) | record | record |
| `x-prep-iterative-imputer-fit` | m_transform.py | train, infer, model, stepfull | batch protocol (vdv-fin) | record | record |
| `x-prep-label-encoder-inverse` | m_transform.py | train, infer, model, stepfull | batch protocol (vdv-fin) | record | record |
| `x-prep-maxabs-scaler-fit-transform` | m_transform.py | train, stepfull | n/a:fit-endpoint (vdv-fin) | record | record |
| `x-prep-mi-discrete-fit-transform` | m_transform.py | train, stepfull | n/a:fit-endpoint (vdv-fin) | record | record |
| `x-prep-multilabel-binarizer-fit-calls` | m_transform.py | train, stepfull | n/a:fit-endpoint (vdv-fin) | record | record |
| `x-prep-normalizer-fit-transform` | m_transform.py | train, stepfull | n/a:fit-endpoint (vdv-fin) | record | record |
| `x-prep-robust-scaler-fit-transform` | m_transform.py | train, stepfull | n/a:fit-endpoint (vdv-fin) | record | record |
| `x-prep-score-edges-rfe-fit-transform` | m_transform.py | train, stepfull | n/a:fit-endpoint (vdv-fin) | record | record |
| `x-prep-simple-imputer-fit-transform` | m_transform.py | train, stepfull | n/a:fit-endpoint (vdv-fin) | record | record |
| `x-prep-spline-options-fit-transform` | m_transform.py | train, stepfull | n/a:fit-endpoint (vdv-fin) | record | record |
| `x-prep-variance-threshold-fit-transform` | m_transform.py | train, stepfull | n/a:fit-endpoint (vdv-fin) | record | record |

#### `tools/identity_lanes/nn_endpoints.py` (14)

| lane | file | parts | batch part | CPU column | NVIDIA column |
|---|---|---|---|---|---|
| `nn-ep-cnn-fits` | nn_endpoints.py | train, stepfull | n/a:endpoint-round-trips | record | record |
| `nn-ep-cnn-layers` | nn_endpoints.py | train, stepfull | n/a:endpoint-round-trips | record | record |
| `nn-ep-embedding` | nn_endpoints.py | train, stepfull | n/a:endpoint-round-trips | record | record |
| `nn-ep-embedding-device` | nn_endpoints.py | train, stepfull | n/a:endpoint-round-trips | n/a:one-sided | record |
| `nn-ep-layernorm` | nn_endpoints.py | train, stepfull | n/a:endpoint-round-trips | record | record |
| `nn-ep-mamba-blocks` | nn_endpoints.py | train, stepfull | n/a:endpoint-round-trips | record | record |
| `nn-ep-moe` | nn_endpoints.py | train, stepfull | n/a:endpoint-round-trips | record | record |
| `nn-ep-rnn-state` | nn_endpoints.py | train, stepfull | n/a:endpoint-round-trips | record | record |
| `nn-ep-samba-inference` | nn_endpoints.py | train, stepfull | n/a:endpoint-round-trips | record | n/a:one-sided |
| `nn-ep-samba-stack` | nn_endpoints.py | train, stepfull | n/a:endpoint-round-trips | record | record |
| `nn-ep-sequential-conv` | nn_endpoints.py | train, stepfull | n/a:endpoint-round-trips | record | record |
| `nn-ep-sequential-dense` | nn_endpoints.py | train, stepfull | n/a:endpoint-round-trips | record | record |
| `nn-ep-sequential-lm` | nn_endpoints.py | train, stepfull | n/a:endpoint-round-trips | record | record |
| `nn-ep-transformer-block` | nn_endpoints.py | train, stepfull | n/a:endpoint-round-trips | record | record |

#### `tools/identity_lanes/nn_programs.py` (6)

| lane | file | parts | batch part | CPU column | NVIDIA column |
|---|---|---|---|---|---|
| `nn-pg-array` | nn_programs.py | train, stepfull | n/a:program-endpoints | record | n/a:one-sided |
| `nn-pg-bpe-decode` | nn_programs.py | train, stepfull | n/a:program-endpoints | record | record |
| `nn-pg-causal-lm` | nn_programs.py | train, stepfull | n/a:program-endpoints | record | record |
| `nn-pg-imported-program` | nn_programs.py | train, stepfull | n/a:program-endpoints | record | record |
| `nn-pg-program-decode` | nn_programs.py | train, stepfull | n/a:program-endpoints | record | record |
| `nn-pg-session-blocks` | nn_programs.py | train, stepfull | n/a:program-endpoints | record | record |

#### `tools/identity_lanes/nn_training.py` (7)

| lane | file | parts | batch part | CPU column | NVIDIA column |
|---|---|---|---|---|---|
| `nn-tr-byte-lm` | nn_training.py | train, stepfull | n/a:training-endpoints | record | record |
| `nn-tr-generator` | nn_training.py | train, stepfull | n/a:training-endpoints | record | record |
| `nn-tr-optim-resident` | nn_training.py | train, stepfull | n/a:training-endpoints | n/a:one-sided | record |
| `nn-tr-optim-state` | nn_training.py | train, stepfull | n/a:training-endpoints | record | record |
| `nn-tr-schedules` | nn_training.py | train, stepfull | n/a:training-endpoints | record | record |
| `nn-tr-seqopt-state` | nn_training.py | train, stepfull | n/a:training-endpoints | record | record |
| `nn-tr-small-mlp` | nn_training.py | train, stepfull | n/a:training-endpoints | record | record |

#### `tools/identity_lanes/parallel_single.py` (42)

| lane | file | parts | batch part | CPU column | NVIDIA column |
|---|---|---|---|---|---|
| `psd-arima` | parallel_single.py | train, infer, model, stepfull | n/a:ordered-shard-driver | record | record |
| `psd-boosting` | parallel_single.py | train, infer, model, stepfull | n/a:ordered-shard-driver | n/a:one-sided | record |
| `psd-boosting-reg` | parallel_single.py | train, infer, model, stepfull | n/a:ordered-shard-driver | n/a:one-sided | record |
| `psd-byte-lm` | parallel_single.py | train, stepfull | n/a:ordered-shard-driver | n/a:one-sided | record |
| `psd-byte-lm-corpus` | parallel_single.py | train, stepfull | n/a:ordered-shard-driver | n/a:one-sided | record |
| `psd-byte-lm-model-pool` | parallel_single.py | train, stepfull | n/a:ordered-shard-driver | n/a:one-sided | record |
| `psd-byte-lm-offload` | parallel_single.py | train, stepfull | n/a:ordered-shard-driver | n/a:one-sided | record |
| `psd-byte-lm-worker` | parallel_single.py | train, stepfull | n/a:ordered-shard-driver | n/a:one-sided | record |
| `psd-causal-lm` | parallel_single.py | train, stepfull | n/a:ordered-shard-driver | n/a:one-sided | record |
| `psd-causal-lm-text` | parallel_single.py | train, stepfull | n/a:ordered-shard-driver | n/a:one-sided | record |
| `psd-cd` | parallel_single.py | train, infer, model, stepfull | n/a:ordered-shard-driver | n/a:one-sided | record |
| `psd-cholesky` | parallel_single.py | train, infer, model, stepfull | n/a:ordered-shard-driver | n/a:one-sided | record |
| `psd-cross-val` | parallel_single.py | train, stepfull | n/a:ordered-shard-driver | record | record |
| `psd-dbscan` | parallel_single.py | train, infer, model, stepfull | n/a:ordered-shard-driver | n/a:one-sided | record |
| `psd-feature-freq` | parallel_single.py | train, infer, model, stepfull | n/a:ordered-shard-driver | n/a:one-sided | record |
| `psd-forecast-arima` | parallel_single.py | train, infer, model, stepfull | n/a:ordered-shard-driver | record | record |
| `psd-forecast-holtwinters` | parallel_single.py | train, infer, model, stepfull | n/a:ordered-shard-driver | record | record |
| `psd-forest-pool` | parallel_single.py | train, infer, model, stepfull | n/a:ordered-shard-driver | n/a:one-sided | record |
| `psd-gmm` | parallel_single.py | train, infer, model, stepfull | n/a:ordered-shard-driver | n/a:one-sided | record |
| `psd-gp` | parallel_single.py | train, infer, model, stepfull | n/a:ordered-shard-driver | n/a:one-sided | record |
| `psd-gpc` | parallel_single.py | train, infer, model, stepfull | n/a:ordered-shard-driver | record | record |
| `psd-gram` | parallel_single.py | train, infer, model, stepfull | n/a:ordered-shard-driver | n/a:one-sided | record |
| `psd-gram-pca` | parallel_single.py | train, infer, model, stepfull | n/a:ordered-shard-driver | n/a:one-sided | record |
| `psd-graph-agglomerative` | parallel_single.py | train, infer, model, stepfull | n/a:ordered-shard-driver | n/a:one-sided | record |
| `psd-graph-umap` | parallel_single.py | train, infer, model, stepfull | n/a:ordered-shard-driver | n/a:one-sided | record |
| `psd-hdbscan` | parallel_single.py | train, infer, model, stepfull | n/a:ordered-shard-driver | n/a:one-sided | record |
| `psd-holtwinters` | parallel_single.py | train, infer, model, stepfull | n/a:ordered-shard-driver | record | record |
| `psd-iforest` | parallel_single.py | train, infer, model, stepfull | n/a:ordered-shard-driver | n/a:one-sided | record |
| `psd-kernel-ridge` | parallel_single.py | train, infer, model, stepfull | n/a:ordered-shard-driver | n/a:one-sided | record |
| `psd-kmeans` | parallel_single.py | train, infer, model, stepfull | n/a:ordered-shard-driver | n/a:one-sided | record |
| `psd-logistic` | parallel_single.py | train, infer, model, stepfull | n/a:ordered-shard-driver | n/a:one-sided | record |
| `psd-neural-trainer-ckpt` | parallel_single.py | train, stepfull | n/a:ordered-shard-driver | record | record |
| `psd-nystroem` | parallel_single.py | train, infer, model, stepfull | n/a:ordered-shard-driver | n/a:one-sided | record |
| `psd-ordered-rmse` | parallel_single.py | train, infer, model, stepfull | n/a:ordered-shard-driver | n/a:one-sided | record |
| `psd-queries-kde` | parallel_single.py | train, stepfull | n/a:ordered-shard-driver | record | record |
| `psd-queries-knn` | parallel_single.py | train, stepfull | n/a:ordered-shard-driver | record | record |
| `psd-rbf-sampler` | parallel_single.py | train, infer, model, stepfull | n/a:ordered-shard-driver | n/a:one-sided | record |
| `psd-reference-proba` | parallel_single.py | train, stepfull | n/a:ordered-shard-driver | record | record |
| `psd-resample` | parallel_single.py | train, stepfull | n/a:ordered-shard-driver | n/a:one-sided | record |
| `psd-scaler` | parallel_single.py | train, infer, model, stepfull | n/a:ordered-shard-driver | record | record |
| `psd-scaler-minmax` | parallel_single.py | train, infer, model, stepfull | n/a:ordered-shard-driver | record | record |
| `psd-svm` | parallel_single.py | train, infer, model, stepfull | n/a:ordered-shard-driver | n/a:one-sided | record |

## F. Batch parts declared by vdv-fin (were `n/a:UNDECLARED`)

The 142 lanes that had no `_batch_decl` (the 137 vdv-cov-est lanes, `m_score.py` 50, `m_predict.py` 49,
`m_transform.py` 38, plus `samba-decode-session`, `x-prep-rfe-no-round` and `sequence-optim-adagrad` / `-lamb` /
`-rmsprop`) now declare one, each in its own file (lane vdv-fin, 2026-11-24); the tables above carry the value,
marked `(vdv-fin)`. 29 lanes declare a batch protocol (`_rows_calls` or a `_BatchRows` spec) on the held-out rows
their probe reads; the other 113 declare `n/a:<reason>` (scalar-reduction for `score` and the other scalars,
fit-endpoint for `fit_transform` / `fit_predict` / `partial_fit` / inverses of a fresh fit, forecast-fit,
whole-series, training-trajectory, fit-probe, fixed-session-batch). Every batch protocol part is OWED and is
recorded with the full CPU and NVIDIA records above.

## G. `ordered-gradient-sum` made two-sided (lane ogs-cpu, 2026-11-24)

`parallel_training.ordered_sum_gradients` takes `device=` (Andrew, 2026-11-24: a CPU side for verification), so the
lane's `_one_sided(..., missing="cpu")` declaration is gone and both columns are owed, all nine fixtures, by the
commands at the top:

| lane | file | parts | batch part | CPU column | NVIDIA column |
|---|---|---|---|---|---|
| `ordered-gradient-sum` | identity_break.py | every part the harness collects | n/a:ordered-shard-reduction | record (the CPU fold, `core/ordered_gradient_sum_host.mojo`, first on the GitHub CPU runner; the `WITH_CPU=1` CPU record on the L40S must equal it) | record (the GPU pair fold, unchanged code) |

The two columns must be equal byte for byte (same per-element step, same order; the lane also holds both to its
NumPy left-fold oracle). The CPU-only installation's `device=None` now runs the same CPU fold in place of the host
training image's `accumulate` (same arithmetic), so the CPU column is recorded, not reused.
