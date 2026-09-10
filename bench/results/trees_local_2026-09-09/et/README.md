# Extra Trees accumulator dispatch — Apple M4, 2026-09-09

Classification split scoring now dispatches to private accumulator widths
4/8/16/32 according to the runtime class count. The default still accepts
32 classes. `MOJOLEARN_ET_MAX_ACC_32` forces the original score width for
comparison; existing fixed 4/8/16 measurement arms remain available.
Regression scoring and rows-per-thread/block-size defaults are unchanged.

The optimization changes only unused private integer array tails. Random
draws, row coverage, integer reductions, packed global accumulator strides,
score finalization, tie rules and leaf arithmetic are unchanged at <=16
classes. Both numeric modes passed 18 complete-model fingerprint comparisons
against fixed-width scoring, including all width boundaries, bootstrap and
best-first trees. The gate records the compiled numeric mode.

The public binding smoke also exposed an existing bug: split search accepted
17–32 classes but the fixed 16-wide leaf kernel returned zero probabilities.
Classification now launches leaf32 above 16 classes. Outputs above 16 classes
are intentionally corrected. The score comparison baseline shares this leaf
fix; its purpose is to isolate the scoring optimization. Regression is not
changed. The new Python classification smoke failed before the leaf fix at 17
classes (1,226 / 1,537 predicted labels differed from CPU, GPU labels all zero).

## Timing

All timed ABBA sequences held `tools/with_build_lock.sh` for their entire
execution. Each arm/process took one warmup followed by three measured fits;
each reported median has six samples. Other agents used the same lock for
compilation, preventing compiler/GPU overlap. Timed regions cover the Mojo
public device-classifier fit and synchronize; synthetic-data generation,
fingerprinting and Python input conversion are outside the timing.

| Mode | Rows | Columns/classes | Trees/depth | Width32 median ms | Dispatch median ms | Speedup |
|---|---:|---|---|---:|---:|---:|
| FAST | 262144 | 13 / 2 | 16 / 10 | 583.0995 | 450.778 | 1.294x |
| IDENTICAL | 262144 | 13 / 2 | 16 / 10 | 589.8135 | 453.958 | 1.299x |
| FAST | 65536 | 13 / 5 | 8 / 8 | 79.943 | 57.6965 | 1.386x |
| FAST | 65536 | 13 / 9 | 8 / 8 | 105.136 | 80.0565 | 1.313x |

Every timed arm produced equal complete-model fingerprints. The additional
65k/binary result is retained in `fast_timing.json`; its first two dispatch
samples were elevated, so the larger binary fixture is the preferred result.
The 17-class timing is explicitly pre-leaf-fix and is not evidence for final
code performance. All <=16-class measurements use the unchanged leaf path.
These are local M4 measurements, not a cross-vendor speed or identity claim.
NVIDIA and AMD columns are cross-vendor-pending.

## Reproduction

Run `bash extratrees/tools/check_accumulator_dispatch.sh` from the repository.
For timing, build `extratrees/checks/accumulator_dispatch_fingerprint.mojo`
with and without `-D MOJOLEARN_ET_MAX_ACC_32=1`; use
`-D MOJOLEARN_NUMERIC_IDENTICAL=1` for IDENTICAL. The standalone probe accepts
`<rows> <classes> <trees> <depth> <reps>`. Saved timing drivers show exact ABBA
invocations and expected temporary binary names; invoke drivers under the
build lock.

Rebuild the public extension with
`MOJOLEARN_NUMERIC_MODE=fast tools/with_build_lock.sh sh bindings/build_trees.sh`
and repeat for `identical`. In each mode, run
`MOJOLEARN_NUMERIC_MODE=<mode> PYTHONPATH=python tools/with_build_lock.sh pixi run python extratrees/checks/binding_dispatch_smoke.py`.
The smoke logs actual loaded ET extension paths (the ET binding does not
export a native numeric-mode readback), checks all score and leaf variants
against CPU probabilities, and checks probability normalization.
