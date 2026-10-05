# Scoped decomp/PCA G1/G2 adapter (OPEN, uncompiled)

Source base a78f0ab5e4f63f32029e7b212fc96cdf7784a524. Catalog fragment provenance:
fa39073607e3b19c4fbfe063a5545a63318f13c2 (direct G1 64x64x16,
G2 32x32x16). The resident screen supports *individual measured shapes*:
G1 tall NN 6.827→3.524 ms, dense NN 2.274→1.951, NT Gram 3.563→2.028;
G2 narrow NT 1.204→0.866. These compare to SDK G0, **not this probe's AFN
baseline**. The strided implementation, TN orientation and atomic split
outputs have no transferred quality or speed claim. No default changes.

The reusable selector uses only M/N/K, physical strides and input pointer
identity. It does not inspect dataset names, values, labels or run variants
at fit time. Since 2026-10-04 the routes have no shape window: the old
windows bracketed board shapes and were removed as benchmark-tuned (kept only
behind default-off MOJOLEARN_LEGACY_NARROW_SCOPED_GEMM). The window-free
routes are UNMEASURED. Selection is layout only:

| Flag suffix | Candidate | Operation (any M, N, K >= 1, Int32-safe) |
|---|---|---|
| G1_TALL | G1 | NN |
| G1_DENSE | G1 | NN (same 64x64 kernel as G1_TALL) |
| G1_GRAM | G1 | same input pointer, square, NT or TN |
| G2_NARROW | G2 | NT (not claimed by G1_GRAM) |

All flags start `MOJOLEARN_SCOPED_GEMM_`; FAST Apple is mandatory. `_SPLIT`
is an additional permission for the existing atomic operation; `_PCA`
is an additional permission for the PCA route. No flag means incumbent.
`_AUDIT` counts incumbent/G1/G2 separately for decomp non-split, decomp split,
PCA atomic, and exposes the last exact strides/splits/per. It never selects
an arm dynamically. Audit calls must be serial. Output-input alias, unsupported layouts and
non-Int32-safe extents retain incumbent behavior.
The quality probe deliberately requires positive extents and refuses K0;
product zero/empty handling remains entirely upstream and unchanged.

`x_decomp/device.mojo::_launch_gemm_mma` still computes occupancy from original
64x64 output tiles, target640, minimum512 K steps, and original AFN KB32
rounding. It still zeroes only the split output. `compute_covariance`
retains column means, centering, fused d≤128 dispatch, its independent
split count, unconditional zero and atomic semantics even at one split,
scaling, restoration and synchronization. Only the selected MMA launch
changes. Per-split padding checks the end of that split, not global K.
The kernel retains ascending 8-element MMA fragments and float32 atomic
addition; scheduling and intrinsic differences still require numerical gates.

## Quality before timing

Binding `scoped_gemm_probe`. A defines `-D MOJOLEARN_SCOPED_GEMM_AUDIT`.
B adds one profile's flags in the order tall,dense,gram,narrow,split,pca:
`tall=1`, `dense=2`, `gram=4`, `narrow=8`, `gram-split=20`, `pca=52`, `all=63`.
The helper checks manifest mode/defines/source and BOTH binary hashes before
loading either arm in a separate process. No installed custom .so required.

```
python tools/scoped_gemm_quality.py FULL_SOURCE_SHA UNIQUE_TAG PROFILE
```

The helper is **not yet in the pinned runner allowlist**; manager must review
and add it before submitting through that runner. No timings in this helper.
It captures 23 GEMM fixtures plus five PCA lifecycle fixtures: measured-shape
anchors, odd neighbors, cancellation, scale imbalance, outside-window controls,
N1, TT, alias/nonalias and K1023/1024/1025. PCA includes d128/129 and restore
on/off plus a one-split atomic control. Every call asserts exact route counts
and independently calculated split/stride metadata. The oracle requires finite
outputs, relative Frobenius error≤5e-6 and **zero allowance** for either relative
Frobenius or maximum-absolute error regression versus A. Input-after and means
are evaluated as well. Original atomic nondeterminism may cause a HOLD; do not
relax thresholds or erase failures. This is a mechanism gate, not downstream
estimator quality or scored performance evidence.

No Mojo compilation or numerical runtime was performed locally. Python syntax
and shell syntax checks only. M2 compile A/B; M3 unscored quality; review failures
per operation, then add actual PCA/RSVD/NMF caller checks and one predeclared
call+first-read measurement per eligible arm/scenario. Do not retime the previous
SDK cold or resident contracts. A new AFN comparison must be named explicitly.

Probe r1 lifecycle repair: GEMM and covariance fixtures share one process-lifetime
DeviceContext via process_ctx; buffers remain alive through download/synchronize.
The original04d source pin is preserved; no measurements or gates are changed.

M2 initial04d compile HOLD: kernel arguments used host Int, which is not
DevicePassable. r2 uses Int32 launch arguments and widens on device, matching
the established AFN kernel ABI. No numerical execution or gate changes.
