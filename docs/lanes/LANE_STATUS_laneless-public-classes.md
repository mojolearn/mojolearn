# lane/laneless-public-classes: twelve of twenty-one, and nine reasons

2026-09-19, Apple M4, ONE CORE, `nice -n 19`, CPU host bindings only. **No
Metal job ran.** Evidence:
`bench/results/identity_break/2026-09-19_laneless-public-classes/`.

`tools/verification_matrix.py --json` reported **35 public API entries with no
identity lane at all**. Fourteen are `mojolearn.models.*` and belong to
lane/models-namespace-lanes. This lane took the other 21.

## Covered: three lanes, twelve entries

| lane | entries | family | sabotage arm that moves it |
| --- | --- | --- | --- |
| `saved-model-host-infer` | `HostForest`, `HostGBDT`, `host_predict`, `host_predict_proba` | forest | `MOJOLEARN_FOREST_HOST_SABOTAGE` (NOT the generic one) |
| `lowbit-conversions` | `lowbit.pack_one`, `.materialize_one`, `.widen_bf16`, `.format_of`, `.is_packed`, `linalg.from_bf16` | linalg | `MOJOLEARN_LOWBIT_CONVERT_SABOTAGE` (**added by this lane**) |
| `grad-accumulation` | `training.accumulate_grads`, `training.accumulation_is_aligned` | training | `MOJOLEARN_HOST_SABOTAGE` |

All 27 cells (3 lanes x 9 fixtures, `--repeats 2`) STABLE. Both declared
batch parts STABLE and seen BATCH_MOVED under
`MOJOLEARN_IDENTITY_BATCH_SABOTAGE=1`. Each lane is declared in
`tools/identity_break.py`'s `@lane`, `host_surface.TRAINING_LANE_NAMES` and
its family's `training_lanes`, and carries a BATCH declaration.

## Three findings, each measured

### 1. The low-bit conversion seams had NO sabotage arm

`widen_bf16`, `narrow_bf16`, `quantize_rows_int8` and
`dequantize_rows_int8` in `gemm/host/gemm_lowbit_oracle.mojo` are the whole
of `gemm/IDENTICAL_LOWBIT_CONTRACT.md` clauses L-1 through L-6, and they are
what `mojolearn.lowbit` and four public `mojolearn.linalg` entries compute on
a CPU column. `MOJOLEARN_HOST_SABOTAGE` reaches `gemm_oracle`'s leaf and
`gemm_int8_oracle`'s dequantized cell and stops there. **Measured: under the
family define alone the `lowbit-conversions` cell is bit for bit the clean
cell on all nine fixtures, every part.** Same shape as `linalg-eigh` (no arm)
and `bpe-trainer` (an arm the lane never reaches).

FIXED HERE: the four seams take `MOJOLEARN_LOWBIT_CONVERT_SABOTAGE` (a low
bit flipped on a bf16 pattern and an int8 code, `gemm_oracle_sabotage_value_flip`
on a widened or dequantized float32, so no input makes any of them a no-op),
and `host_surface.GATE_SABOTAGE_OWN_DEFINES["linalg"]` names it so the CPU
identity gate's sabotage set carries it. A clean build of the edited oracle
hashes exactly what the pre-edit prebuilt binding hashed on 16 cells
(`regression-*.json`).

### 2. The forest gate refuses the groves engine it is supposed to cover

`docs/COVERAGE_AUDIT_2026-09-18.md` says HostForest and HostGBDT are covered
by "representative bundled saved models and dedicated gates". Re-run at this
commit, `tools/forest_host_gate.py check` over the eight Apple fixtures reads
IDENTICAL on every hash, so the gate is real. It does not reach two things:

* `host_predict` / `host_predict_proba`, the one-call entries the package
  exports, which the gate never calls (it calls `host_model`);
* a `parallel_groves` archive, which the gate REFUSES BY NAME
  ("the host engine is sequential", `tools/forest_host_gate.py:219`). That
  sentence stopped being true on lane/forest-groves-cpu-and-speed
  (2026-09-17). The groves HOST engine had no recorded evidence of any kind.

The lane runs all three. **The gate's groves refusal is still there and is
owed a fix** — it is a gate bug, not an engine bug, and fixing it means
`make`/`record` on a GPU box, which this lane could not do.

### 3. A generic sabotage column would have recorded both lanes as passing

`-D MOJOLEARN_HOST_SABOTAGE=1` alone leaves `saved-model-host-infer` and
`lowbit-conversions` at their clean hashes on all nine fixtures. Both need
their family's own define. This is the failure the prompt's byte_lm example
names, measured twice more.

### 4 (method). The accumulation arm would have been inert on two fixtures

`host_samba_accumulate`'s arm WRITES `0.0` into the first combined element.
`X[0, 0]` is exactly `0.0` on `denormal_ftz` and a subnormal that flushes to
`0.0` on `denormal`. Microbatches cut straight from the fixture would have
combined to `0.0` there and the arm would have changed nothing on two of nine
fixtures. The lane adds `+ 1.0` for exactly this; 9/9 move.

## Not covered, and why — nine entries

### The two decode sessions: no CPU column exists

`mamba.Mamba1DecodeSession`, `transformer.TransformerDecodeSession`.
`mamba1_session_create` and `transformer_decode_session_create` are defined
only in `bindings/_mojolearn_mamba.mojo` and
`bindings/_mojolearn_transformer.mojo`; no host binding exports them, and
both constructors refuse by name on the CPU route (measured). A lane would
read REFUSED on every CPU column, and a REFUSED cell is indistinguishable
from a pass in a column total. **These need a lane written and run on a GPU
column**, which this lane could not do (no Metal, rented GPUs belong to other
sessions).

Instead: `python/mojolearn/tests/test_decode_sessions_cpu_route.py` holds the
refusal to being BY NAME (the class, "resident decode session", and the
"step()" alternative it points at), holds it to coming from the BLOCK rather
than an import-time absence, and holds the per-call `step` path the message
names to actually running and being repeatable. It is a TEST, not a lane: it
says this box behaves, not that two boxes agree.

### The seven `parallel_*` entries: multi-device drivers that refuse on CPU

`parallel_forecasting.{predict,forecast}_arima`,
`parallel_forecasting.{predict,forecast}_exponential_smoothing`,
`parallel_gaussian_process.{fit,predict}_gaussian_process_classifier`,
`parallel_model_selection.cross_val_score`.

Measured on a CPU-only install with `devices=(0,)`:

| entry | refusal |
| --- | --- |
| all four `parallel_forecasting` | `NotImplementedError: no CPU implementation of the parallel worker operation forecast_predict yet` |
| both `parallel_gaussian_process` | `... gpc_class_fit yet` |
| `cross_val_score` | `NotImplementedError: parallel cross-validation requires CUDA or HIP GPU workers` |

`_parallel_pool.CPU_OPERATIONS` admits only the operations whose driver
splits in Python and merges byte for byte in shard order; `forecast_predict`,
`gpc_class_fit` and `gpc_class_predict` are not in it, and
`_parallel_worker.execute` refuses `cross_val_fold` a second time for any
vendor that is not CUDA or HIP. A lane for any of them reads REFUSED on every
CPU column. Note also that every `par-*` lane is excluded from a release
record (`identity_break.RECORD_EXCLUDED_PREFIXES`) and runs at
`_par_devices() == (0,)`, one device, so even on a GPU column such a lane
would not be the two-device claim these drivers make.

**CANDIDATE, NOT OPENED** (no new lanes without asking). Adding
`'forecast_predict'` to `_parallel_pool.CPU_OPERATIONS` would put the two
forecasting drivers on the CPU route: their split IS a Python series-range
split with a byte-for-byte ordered merge, which is exactly the shape that
frozenset's own docstring admits, and the ARIMA and Holt-Winters host
predict doors (`bindings/arima_host_predict.mojo`,
`bindings/holtwinters_host_predict.mojo`) already exist. A `par-forecast-predict`
lane could then hold the sharded prediction to the plain `predict`, the way
`par-arima` holds the sharded fit to the plain fit. `cross_val_score` cannot
be reached this way: its worker refuses by vendor, and the audit records the
whole feature as living on `lane/multigpu-cv` with physical qualification
owed.

## Files touched

* `gemm/host/gemm_lowbit_oracle.mojo` — the four seams' sabotage arm
* `tools/identity_break.py` — three lanes, two batch parts, one `n/a`
* `python/mojolearn/host_surface.py` — three `TRAINING_LANE_NAMES`, three
  `training_lanes`, one `GATE_SABOTAGE_OWN_DEFINES` entry
* `tools/verification_matrix.py` — the "no lane on purpose" paragraph, which
  was no longer true
* `docs/COVERAGE_AUDIT_2026-09-18.md` — the HostForest/HostGBDT claim
* `python/mojolearn/tests/test_decode_sessions_cpu_route.py` — new
* `docs/VERIFICATION_MATRIX.md` — regenerated
