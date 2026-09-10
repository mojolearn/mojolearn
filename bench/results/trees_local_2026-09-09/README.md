Tree performance changes, Apple M4, 2026-09-09

Implemented in the working tree based on `c59894ab`, with independent RF,
Extra Trees, and GBDT implementation/validation lanes. Timed work was
serialized with the repository build lock. No NVIDIA or AMD device was used.

The enabled changes are class-sized Extra Trees scoring and tiled host input
packing shared by RF, Extra Trees, and GBDT. Numeric-mode contracts and tuning
parameters are preserved by these optimizations. A separate existing Extra
Trees bug above 16 classes was corrected, intentionally changing those
previously incorrect outputs.

| Enabled change | Workload | Baseline ms | Updated ms | What was timed |
|---|---|---:|---:|---|
| Extra Trees, FAST | 262,144 rows, 13 features, 2 classes, 16 trees, depth 10 | 583.100 | 450.778 | Native fit |
| Extra Trees, IDENTICAL | Same | 589.814 | 453.958 | Native fit |
| Shared float32 input packing | 1,000,000 x 28, row-major | 50.635 | 23.524 | Host conversion only |
| Shared float64 input packing | Same, converted to float32 | 98.699 | 20.228 | Host conversion only |

These are synthetic local workloads, not HIGGS or end-to-end speed claims
for every estimator. The host numbers use NumPy 2.5.2, nine alternating
post-warmup samples per arm. Extra Trees uses six post-warmup samples per arm
in ABBA order. The input-conversion gains cannot be multiplied by native-fit
gains to predict an end-to-end speedup.

Extra Trees now selects 4/8/16/32 private accumulator slots according to class
count. Integer scoring, RNG, ties and row coverage are preserved. The old
16-output classification leaf kernel silently produced zero probabilities
for 17–32 classes; those fits now select a 32-output leaf kernel. Existing
leaf arithmetic at 16 or fewer classes is unchanged. Both public Python
bindings were rebuilt, and predictions/probabilities matched CPU results at
all nine tested class-count boundaries. See [ET details](et/README.md).

Input packing writes cache-sized row tiles directly into one final Fortran
allocation for large contiguous native-float row-major inputs. Small, narrow,
strided and other-dtype inputs retain the NumPy path; already-Fortran float32
inputs remain zero-copy. Bit checks cover signed zero, subnormals, infinities,
NaN payloads, read-only sources, tile tails, endian conversion and flat-view
layout. Raw timings are in [input-layout.jsonl](input-layout.jsonl).

RF kernel defaults remain unchanged. The new isolated candidate sweep checks
compiled mode/flags, complete-model fingerprints, warmups and timing canaries.
All five FAST candidates matched the baseline across 50 fits; the warmed
baseline/items4 shortlist matched across another 80 classification/regression
fits in FAST and IDENTICAL. Both timing windows were invalidated by canary
drift (1.77x and 1.88x), so they establish no kernel speedup. Raw evidence is
under [rf/](rf/); reproduction is in
[RF_CANDIDATES.md](../../../ensemble/bench/RF_CANDIDATES.md).

GBDT fusion remains opt-in. Direct fused/separate kernel comparisons passed
48 objective/layout/weight cases per mode plus million-row comparisons.
The final full-fit A/B compared splits, leaf values, predictions and the full
64 bits of losses over 96 fits at 65,537 and 1,000,003 rows. Fingerprints
matched within each mode. At 1M rows FAST aggregate medians were 488.581 vs
467.387 ms, but the winner reversed with run order. IDENTICAL medians were
568.751 vs 609.684 ms. Those results do not justify enabling fusion by default.
Use [gbdt-fit-v2-summary.log](gbdt-fit-v2-summary.log) and `gbdt-fit-v2/` as
the authoritative full-fit evidence; the earlier `gbdt-fit/` run hashed losses
after float32 conversion and is superseded. The v2 wrapper's reporting phase
failed after all fits completed because the running shell script was edited;
the summary was recovered and validated from all 16 complete logs. The final
script passes shell syntax validation.

NVIDIA IDENTICAL single-pass stable partitioning now has an explicit
`MOJOLEARN_IDENTICAL_SINGLE_PASS_PARTITION` opt-in. Default routing is
unchanged; the existing kill switch takes precedence. Default, opt-in and
opt-in-plus-kill-switch routing checks pass. Actual NVIDIA execution above
the 500,000-row-per-leaf gate remains unvalidated here, so this is not enabled
by default.

Validation completed: 59 Python tests plus 13 subtests; three RF sweep-gate
tests; 18 Extra Trees complete-model scoring comparisons per numeric mode;
the 45-cell ET batched check; FAST and IDENTICAL public ET binding smokes;
the GBDT kernel/full-fit and partition-routing checks described above.
Optimized scoring comparisons above 16 classes share the corrected leaf
implementation on both sides and therefore isolate the scoring change.

Reproduce input packing with
`PYTHONPATH=python pixi run python bench/speed/tree_input_layout.py --rounds 9`.
Run `bash extratrees/tools/check_accumulator_dispatch.sh` for ET score gates
and `tools/gbdt_fused_ab.sh <output-directory>` for the full-fit fusion A/B.
Use the build/timing locks for measurements; do not treat an invalidated RF
quiet-window result as a speed claim.
