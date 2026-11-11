# KernelPCA typed native execution

KernelPCA's public `fit`, `fit_transform` and `transform` now unpack admitted
Python storage into typed Mojo requests. Private kernel matrices, row sums,
centered matrices, alphas, eigen rooms, options, dimensions and route decisions
are native values. Compiled operations exchange fixed native descriptor spans;
no Python function map or Python argument list connects these operations.

## Native execution and ownership

`core/kpca_native_api.mojo` defines `KPCAEigenRequest` and the fixed eigen ABI.
`KPCAEigenKernels[E,D,RES,X]` calls the existing `kpca_eig_run`, including
resident RBF preparation, Lanczos and dense eigendecomposition. Pointer and
output-room admission order stays in the same provider stages.

`core/kpca_native_plan.mojo` owns `KPCAKernelOptions`, `KPCAFitRoute`,
`KPCAEigenRoom`, `KPCAScratch`, `KPCAFitPrepareRequest`, `KPCADenseFit`,
`KPCAAlphaRequest`, `KPCAQueryRequest` and `KPCACenteredQuery`.
`KPCANeighborDriver` reuses the existing `XnDriver` ABI and numerical operations.
Fused query stays one `kpca_transform` operation. The old-items switch retains
kernel, row-sum, divide, center and matmul in order. The resident switch and
its existing RBF/auto/top-k selection, thresholds and defaults are unchanged.

The neighbor images export `kpca_neighbors_native_api_v1` for native metadata;
the decomp images export `kpca_decomp_native_api_v1` for the eigen driver.
`bindings/kpca_descriptor_boundary.mojo` admits both descriptor families.
Neighbor execution and metadata must belong to the same provider module;
decomp eigen and global-mean reduction likewise keep the same decomp module.
Each call copies the native driver and retains its Python image pins before
releasing the GIL. No cache-list borrow survives a GIL-free operation. The
separate native provider images remain pinned through each native call.

## Public boundary behavior

`bindings/kpca_native.mojo` retains input and fitted-array pins, public output
allocation and publication. Exact scalar admission precedes conversion,
comparison and hash operations; unsupported hooks are refused without calling
them. NMF's admitted comparison helper and IncrementalPCA's exact truth helper
were extracted unchanged for reuse in `decomp_scalar_compare.mojo` and
`decomp_truth_boundary.mojo`. Their former duplicate implementations were
removed. The shared native Array storage-admission repair is a dependency.

The fit stage order remains kernel/options, source, rounded gamma publication,
mode/providers, requested count/route, dense preparation, remove-zero option,
eigen execution and public results. Precomputed fit still ignores degree and
coef0; query still converts both. Query preserves its input/fitted-state,
alpha, precomputed-width, fitted means and scalar stages, including output
allocation before fused dispatch and after old-items centering. Public fields
are read again through exact-key admission after GIL-free stages. Public
fitted arrays may still be rebound or mutated and are admitted afresh on query.

New safety refusals reject too few eigenvalues, mismatched eigenvector row
counts, short fitted column means, an absent global-mean word, or mismatched
query/fitted feature widths before kernels can read beyond their admitted
storage. Longer eigenvalue and mean vectors retain their previous prefix use.
These add defined refusals for malformed storage; they do not change valid
array shapes or numerical operations. `fit_inverse_transform=True` remains
unsupported, as before this migration; no inverse feature is removed or added.

## Correctness repair and removals

The separately frozen dense-centering repair corrects the old global mean:
`colsum(1,n)` copied n column means, then centering read only the first divided
word. `kpca_fit_global_mean` uses the existing blocked decomp reduction on n×1,
then the existing Xn division on one word. `KPCADenseFit` and the regression
fixture call this same product helper. The resident route and tuned kernels
are unchanged. Dense outputs can change because the prior mean was wrong.
See `KPCA_DENSE_CENTERING_REPAIR_20261110.md` for preserved before evidence.

Removed: the public owner's Python operation-list construction and per-function
neighbor map; the unused private Python `_k` and `_alpha_scale` implementations;
the obsolete precomputed-kind context item; and the unused `kpca_eig_py`
Python-list wrapper, its GPU/host exports and surface inventory entry. Shared
neighbor operation exports remain because other families still use them.

## Remaining Python boundary state

This is a native execution migration, not a claim that the whole family has
zero Python objects. The actual boundary retains the public estimator fields,
public arrays, source/storage pins, descriptor/image pins, import-connected
mode/factory maps, the admitted kernel-name table and old-items metadata. The
shared public lifecycle still owns its argument/publication bookkeeping.
Native RUN does not import Python or consume those objects. Other decomposition
families retain their existing bridge entries and are not migrated by this
change. No source guard or Python fallback is added.

## Evidence and integration scope

A source-only traversal of the concrete host/device native descriptor and
provider roots found 221 local Mojo files and no Python/binding/provider-Python
imports. Ten numerical source files match base `3a4c9ac963` byte for byte:
KPCA owner, decomp owner-device, Lanczos, decomp host/device/cells and neighbor
host/device/iter/items. Hashes and full closure are in
`~/mojolearn-evidence/python-mojo-migration-20261110/kpca-native-owner/source-audit.json`.
This is source evidence, not compilation, speed or runtime evidence.

The new Python boundary suite has 13 definitions / 69 parameter-expanded cases,
covering retired-dispatch traps, descriptors and image pairing, exact scalar
forms and callback refusal, ignored options, publication order, short-buffer
refusal, and mutable/borrowed storage. Only AST parsing was done by this lane;
the suite remains uncompiled/unrun against a built product.

Root reports that the corrected shared-helper regression compiled and ran with
exit 0: mean 2.25, centered `[.25,-.25,-.25,.25]`, all canaries intact. All 58 pure
KPCA contract/state cases passed in that separate native batch. Evidence:
`~/mojolearn-evidence/python-mojo-migration-20261110/kpca-contract-centering-r2-result.json`
and its run log. The old reproducer's intentional failure remains preserved.
This does not cover the pending public boundary or all native execution routes.

Integration must compile affected GPU/host x_neighbors and x_decomp roots plus
the decomposition runtime owner (`_mojolearn_runtime_decomp_host`). The extracted
NMF/IPCA boundary helpers share that owner. Neighbor generator and host export
inventories include the native factories. Native numerical/route fixtures are
owned by the test lane; root owns all compilation/execution. No lane tests,
boxes, A/B define, performance measurement or cross-vendor identity run occurred.
