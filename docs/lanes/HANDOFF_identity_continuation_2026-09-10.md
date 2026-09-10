# IDENTICAL closure and performance continuation — September 10

The preceding pass was already pushed at `1b3f53d6`; this pass continues the
user's UMAP-first audit and performance work. Three isolated lanes handled
UMAP, IVF/kNN and Mamba/transformer; root implemented wide PCA, reviewed and
integrated the changes, serialized local builds, and owned one H100 rental.
No tree implementation was edited by this lane. Concurrent foreign changes
to `checks/fixed_point.mojo` and foreign benchmark directories were excluded.
FAST/DETERMINISTIC behavior is retained; their surfaces were not tested here.

## What the requested list now means

| Item | Result and evidence boundary |
|---|---|
| UMAP raw binary64 exp/log/log2/pow | Routed through portable IDENTICAL seams in curve, dense/sparse graph, transform, and serial dense/sparse optimizers. Apple and Linux/H100 gates pass. |
| portable_pow64 | Implemented with explicit special cases and subnormal outputs. 262,378 pairs pass; matching input/output hashes on both hosts. Tested approximation, not correctly rounded pow. |
| Other missing primitives | Reproducible inventory separates real production calls from comments/oracles. Several requested functions have no executable Mojo consumers. FP32 log2 has a real tree call and remains untouched. |
| IVF k above 256 | IDENTICAL admits k through 1024 using the existing strided rank selector; coarse selection also gates 257 probes. Existing and new gates pass on Apple and H100. |
| Fused kNN on CDNA64 | Explicit logical-32 lane IDs, votes, shuffles and broadcasts implemented. Native Apple/NVIDIA and declared-width simulations pass; strict actual gfx942 compilation passes. Physical CDNA execution remains owed. AUTO selection is unchanged. |
| Full PCA on wide matrices | IDENTICAL transposed QR plus explicit right-basis reconstruction, with public fit/transform/inverse/whitening gates. Six native fixtures match across Apple/H100, including zero/rank-deficient inputs and 129 features. |
| Embedding PLAN_SORT and log2 hashed gate | Already implemented and tested in the preceding pass. Not undone or requalified from these unrelated tests. |
| Fourth hardware column | Prior real gfx1100/gfx1201 builds remain evidence; this pass adds real CDNA compilation. No physical fourth-column execution, Intel/Qualcomm backend certification, or generic FP32 reduction is claimed. |
| Mamba-1 decode wiring | Already live: public Mamba1Block.step reaches the device mamba_step in mamba_simple.mojo. No duplicate wiring was added. |

## UMAP arithmetic and primitive audit

The primitive's observed maximum relative error is 1.2169443322931572e-13;
maximum ULP distance is 1037. Admission remains relative error <=2e-12 plus
two minimum subnormals. General powers use portable log/exp; exact identity,
reciprocal, square and integer binary-power cases have separate routes.
The local subnormal reconstruction does not alter existing exp64 semantics.

Input FNV64 is 11634649235927139599 and output FNV64 is
13353115118729037508 on Apple and Linux. The first Linux gate exposed a
signaling-NaN-to-zero-exponent libm policy difference. Production already
specified x**0 and 1**p as one; the revised gate tests that policy exactly and
reports the libm difference separately. No finite accuracy threshold changed.
The failed attempt and corrected captures are retained.

Apple/H100 match 186 small identity cells, 690 broader identity cells,
116 host-stage cells across 17 stages, and 21 public transform cells.
Both pre-change Apple identity fixtures retain their bits. These fixtures do
not prove all previously libm-dependent inputs retain their old output.

Evidence: `bench/results/umap_portable_host_math_2026-09-10/`.
Reproduction: `pixi run check-portable-pow64`,
`pixi run check-umap-portable-host-math`.
Inventory: `docs/lanes/PORTABLE_PRIMITIVE_AUDIT_2026-09-10.md` and its JSON;
`tools/audit_portable_primitive_calls.py` reproduces lexical call detection.

## Wide full PCA

For centered X with m<n, factor X^T=QR using one QR panel, SVD R^T=USV^T,
then apply the saved reflectors in reverse order to [V;0]. The right basis
is QV. This neither forms a covariance nor divides by singular values.
Saved diagonals reconstruct the original tau; the head read occurs before
the fold barrier so another warp cannot observe an overwritten head.

The new wide route uses 64 sweeps and relative tolerance 1e-6. Large-theta
rotations avoid overflowing theta squared. Columns whose pinned squared
norms flush to zero are treated as zero in the relative rotation test too;
otherwise the rank-deficient H100 fixture cycled. This was caught by the
cross-device gate and fixed before qualification. The six final native
component/singular-value hashes match. Tall fits retain their existing
rotation, 15-sweep/1e-7 budget, and pass the full existing suite on both GPUs.

The shared host tail uses min(m,n) singular values and that count for noise
variance; basis storage keeps its existing feature stride. Public shape
validation reaches the new route and rejects k>min(m,n). Nonconvergence
still raises. Square feature buffers and single-panel Q reconstruction are
remaining memory/performance limits. Tall full-SVD feature counts above 128
remain unmeasured; the 129-feature gate here is wide. No physical AMD PCA
identity claim is made.

Reproduction: `pixi run check-svd-wide-identical`; after building the
IDENTICAL estimators binding, run `tools/pca_wide_surface_check.py` with
`MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=python`.
Native and public evidence lives in
`bench/results/identity_continuation_2026-09-10/`.

## Performance and opponent reuse

Mamba-3 shares causal yintra operands while retaining each ascending FMA
chain, causal zeros and ragged terms. The new default is NVIDIA IDENTICAL
only, with capacity and occupancy guards; Apple and tiny calls retain the
previous default. The direct tile gate checks 1,128,960 cells over 100 cases.
Native continuation/decode/refusal traces and public output/report checks
remain exact. `MOJOLEARN_MAMBA3_LEGACY_YINTRA` supplies the A/B control.

Transformer's stateless forward keeps its original zero device cache, but
omits host cache allocation and the two uploads/downloads of state Python
would discard. Explicit-state, ring and backward paths retain their
behavior. The default is NVIDIA IDENTICAL only; Apple retains the old route
because its timing evidence was mixed. Apple and H100 pass all 82 full-array
hash comparisons; the dedicated fresh check covers 188,928 cells, ordered
mutable-weight refusals, dispatch and capacity guards. Both original large
16,777,216-cell output hashes are unchanged. The Torch comparison still fails
its original numerical admission, so no qualified opponent ratio is reported.

| Workload | Same-pod baseline ms | Final default ms | Less time |
|---|---:|---:|---:|
| Mamba-3 B8/L4096/D512 | 59.503218 | 55.910172 | 6.0% |
| Mamba-3 B8/L1024/D2048 | 105.693202 | 101.634823 | 3.8% |
| Transformer B8/L4096/D512 | 210.893420 | 204.793565 | 2.9% |
| Transformer B8/L1024/D2048 | 214.515634 | 188.905248 | 11.9% |

These final prices run default first, baseline second; the first order also
improves on both large shapes. Mamba-3 uses five timed rounds, transformer
seven. The unchanged tiny Mamba branch measured 1.219134 versus 1.183115 ms
in this reverse trial; no tiny-call improvement is claimed.

Final timing rows and all samples are recorded in `bench/OPPONENT_REFERENCE.md`
and the lane evidence. No opponent was retimed. This is a new physical H100
of the same model/driver, so the earlier pod's absolute own timings are not
used as the before/after denominator. Cached opponent timings retain their
original provenance and scope.

kNN's default throughput was not changed in this refusal pass. Its prior
3.47–3.79x cuML gap remains; the explicit fused arm and larger IVF k are
capability changes, not new speed claims. Dense GEMM's previous 5.2–5.7x gap
also remains. The missing general primitive families and physical fourth
column are not represented as completed work.

## Reproduction and runtime record

GPU jobs ran serially on one H100 80GB HBM3, driver 580.126.09,
pod td6w0v7xzqcxeh. The logged Python/public harness, source hashes, CPU/GPU
metadata, warmups and every sample accompany the evidence. Native Mojo is
1.0.0 (ed45d567). No opponent code or tolerance was modified.

Root reconstruction/build scripts and logs are retained in
`bench/results/identity_continuation_2026-09-10/reproduction/`.
The scripts use the original archived grid helper/generator pair; the older
main-tree speed helper is not an interchangeable fixture generator.


The pod was deleted at 06:16 EDT: DELETE 204 followed by GET 404 verified at
06:16:25. All evidence was fetched first. Up/down and detached-job logs are
included in the root evidence directory; no GPU or job remains active.

Raw lane commit IDs have equivalent source commits reachable from main:

| Change | Lane commit | Main commit |
|---|---|---|
| UMAP/pow64 |163aaf81|dd68e9aa|
| Pow special-policy gate |20061ad5|bb4e0259|
| IVF capacity |163ff740|92178f1f|
| Fused logical groups |4f201954|9a7cb5d9|
| Mamba yintra default |2c5fb5d3|28a32835|
| Transformer fresh default |09f1b539|f02511bd|
| PCA final zero-norm rule |6ce6333b|6ce6333b|

The final remote production hashes match the checkout for numerics, Mamba-3,
transformer binding and wide PCA. Ten UMAP source hashes also match. SHA256
manifests and strict comparison records accompany the captures.
