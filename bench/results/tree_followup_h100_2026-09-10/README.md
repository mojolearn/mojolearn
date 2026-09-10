# H100 tree follow-up, 2026-09-10

Large-data training performance is the optimization target. The tiny RF fixtures
here are correctness gates only; they provide no weighted-training speed claim.

## Corrected symmetric comparison

HIGGS 1M x 28 training, fixed 500K heldout tail, 100 trees at depth 6.
MojoLearn IDENTICAL and CatBoost GPU both use random_strength=0 for this
explicitly named `matched-no-noise` benchmark profile. Product defaults are
unchanged. One warmup, six measured rounds, first arm rotates each round.
Constructor, host packing, fit and synchronization are timed; scoring is outside.

| Arm | Median fit | Max/min spread | Stable (<=1.10)? | Log loss | AUC |
| --- | ---: | ---: | --- | ---: | ---: |
| MojoLearn IDENTICAL | 524.744 ms | 1.200703 | No | 0.542067 | 0.800716 |
| CatBoost GPU | 917.5715 ms | 1.057976 | Yes | 0.541756 | 0.800891 |

No accepted speed ratio: MojoLearn still fails the predeclared stability gate.
All six MojoLearn models and prediction hashes match. CatBoost fitted settings
confirm Newton, ten leaf-estimation iterations, AnyImprovement backtracking,
Cosine, DocParallel, Plain, no bootstrap and zero split noise. Equal controls
still do not establish equivalent searchers or border grids. See the
[dispatch audit](../../../docs/lanes/SYMMETRIC_CATBOOST_COMPARISON.md).
One-second GPU telemetry is retained; it does not isolate the cause of timing
variation or replace a stage/kernel profile. Do not attribute the lower median
to identity. No random-strength quality/default-change experiment was run.

## RF class-weight correctness

The first CUDA gate failed: bootstrap=False incorrectly used unweighted bins.
`rf_class_weight_initial_failure.log` and `rf_class_weight_diagnostic.log` retain
that failure, including a positive weighted-bootstrap witness and unchanged
non-bootstrap probabilities. It was a binding defect, not an acceptable no-op.

The corrected binding dispatches non-bootstrap weighted fits to existing
WeightedClassificationBin with a data-dependent fixed-point weight scale.
`build_rf_weighted_fix.exit` and `rf_class_weight_fixed.exit` are zero.
The final gate passes six weighted cases (dict, balanced, zero-class weight,
with and without bootstrap), repeated full-model/prediction equality,
None/unit-weight equality, and a fractional-weight stump oracle [1/7, 6/7].
These are 17 tiny GPU forests, not a performance benchmark. NVIDIA IDENTICAL
is checked; this is not all-mode or cross-vendor qualification. The bounded
[weight contract](../../../docs/RF_CLASS_WEIGHT.md) documents host sampling,
Float32 weights and remaining API restrictions.

## Source, checks and lifecycle

The pod received archive `5acb3fb2`, then explicit source overlays through
`8843008c` for the relevant binding, Python guards and benchmark files.
The JSON source_commit retains the original archive marker; its per-file hashes
and `final_sources_and_binaries.sha256` identify actual overlaid source/binaries.
Unrelated integration/main changes were not compiled on the pod. All four
IDENTICAL bindings built; the RF binding was rebuilt after its dispatch fix.
No full-wheel or whole-main validation claim. Package versions, setup and
compiler logs are retained. The optional LightGBM CUDA build was skipped.

Local integration checks: 165 focused host tests passed for call-time mode,
fit-mode retention, class weights and GBDT option guards. Earlier combined checks
also passed 177 tests/13 subtests; neither count is GPU evidence. The shared
benchmark runner's fixed/rotating/failure paths passed its host control check.
Small/tall/wide reminder profiles were exercised without large allocations.

Dedicated H100 80GB pod `m3rrwgtgcng12u`, created 12:48:38 UTC, advertised
$3.49/hour. A 60-minute API-termination watchdog was armed before work.
Evidence was copied and checked locally before manual termination at 13:02:14.
DELETE returned 204 and GET returned 404 at 13:02:15: removal verified.
Approximately $0.79 compute for 13.6 minutes, excluding storage; not a final bill.
The existing Samba training pod was untouched. No rental remains from this run.
