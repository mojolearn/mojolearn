# Shared G1/G5 dispatcher: quality stage only

Base d32fdfc743e5f761409b61a3c4a2c61ded75ea81. Catalog provenance:
9ab2d3d3fb770498ef025db08f595a0149792bb7; repaired kernel imported from
M2-compiled probe 6abb76673038a3f7a3eb6ebc5be331472e32747e. No numerical
kernel changes. No default, timing, opponent run or board claim.

`MOJOLEARN_APPLE_FAST_SHARED_GEMM_G1` and `_G5` select direct/staged 64x64,
K16 full-FP32 MMA, mutually exclusively. Both default OFF, FAST Apple GPU
only, excluding the CPU column. Current SDK dispatch stays the default.
This differs from the old dropped NT scalar-tiled and pinned-vendor
experiments: it uses the catalog's Apple matrix fragment operation and
checks both NN/NT public dispatch routes before any timing.

One adapter serves `core.gemm.gemm_nt`, `gemm_nt_gram` and
`gemm.checks.gemm_identical._fast_vendor_gemm` NN/NT. It allocates nothing
and adds no wait, copy or host arithmetic. n=1 keeps original GEMV/SDK
behavior; Gram n=1 still raises; K=0 and invalid/oversized dimensions do
not enter candidate; unsupported TN still returns false. Inputs may alias
for Gram. Existing output ownership and callers' synchronization persist.
IDENTICAL, non-Apple, fused updates, AFN direct and allow_vendor=False
routes are unchanged. Statically reachable callers are not runtime proof.

## Manager commands and strict gate

M2 compile-only family `shared_gemm_probe`: A empty defines; B
`MOJOLEARN_APPLE_FAST_SHARED_GEMM_AUDIT`. The build script is
`bindings/build_shared_gemm_probe.sh`, same flags/lock as the compiled
catalog probe. A does not run; B exposes serial audit-only selection of
incumbent/G1/G5. The runtime selector and mutable counters are inaccessible
in normal estimator bindings unless this explicitly diagnostic flag is
set; never use the audit flag for production or timings.

M3 unscored command:

```
python3 tools/shared_gemm_quality.py SOURCE_SHA shared-gemm-routes-q-20261004
```

Loads verified B.so directly, requiring exact source/mode/defines/hash and
ABI1. No installed custom extension needs backup. Each fixture executes
all three arms through the actual dispatch entrances. Every call checks
all four route counters (fallback, G1, G5, total candidate launches), then
copies the full output and synchronizes. Requires G1/G5 exact equality,
finite outputs, FP64 scaled Frobenius error <=5e-6, and both scaled error
and max absolute error <= incumbent, zero extra tolerance. Fixed before
measurement; never loosen after a failure. 43 fixtures: square, ragged,
small, vector, cancellation, dynamic range, K0, representative distance,
projection, RBF, long reduction, and aliased Gram. Every call also checks
vendor TN refusal. K0 explicitly tests adapter refusal followed by GPU
zero fill: it does not claim SDK zero-extent support.

## Gaps and next gated queue expansion

This is an actual **route** harness, not fitted-estimator quality. Do not
promote or score board jobs from this gate alone. After it passes, compile
separate static G1/G5 estimator arms on current main and verify binding-level
fit state and predictions for kmeans (inertia/labels), brute KNN
(distances/neighbors/recall), SVM (decision function/predictions), KDE
(log densities), multiclass GLM (coefficients/logloss), PCA transform and
inverse (reconstruction), and resident RBF (kernel/predictions). Check
selected algorithms really reach the adapter; imports are insufficient.
If any static arm regresses, hold that variant before timings. Preserve
existing board opponent deficits. NN neural coverage is separate and not
permission to change its defaults. Strongly fused/split/batched LU,
Cholesky, decomposition and MCD paths need dedicated adapters and gates;
this two-entry integration does not claim to accelerate them.

Only after relevant quality passes should the manager authorize one
call+first-full-read measurement per arm for covered algorithm/dataset
rows. No scored jobs are included in this helper.

Compile-only repair r1: original 5a2bc66d212634e7d2a3385ddffa72309994df39
failed M2 parsing because module-level `comptime assert` is unsupported.
Move mutual-exclusion assertions into `try_shared_gemm` before its enabled
branch; keep original SHA and all quality thresholds unchanged. No GPU
evidence existed for the failed source. New tag: shared-gemm-routes-r1-q-20261004.
