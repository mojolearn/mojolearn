# KernelPCA typed native requests

The additive `core/kpca_native_api.mojo` defines `KPCAEigenRequest`, a fixed
native eigen call, and owning-image mode/vendor/capability metadata. The
`KPCAEigenKernels[E,D,RES,X]` provider calls the unchanged `kpca_eig_run`,
including its resident RBF preparation, Lanczos convergence route and dense
eigensolver. Its old pointer/dimension/output refusal order remains.

Neighbor compute will reuse `core.neighbors_native_driver.XnDriver` and its
existing operation IDs and spans. A small `KPCANeighborInfo[DEVICE,RESIDENT]`
metadata descriptor, exported by that same neighbor image, retains its actual
resident switch, mode and vendor without changing the shared Xn ABI or adding
an extra call layer around each numerical operation. The boundary must verify
that metadata and Xn descriptors belong to the same module, and pin both;
the decomposition descriptor keeps its own separate provider image pin.

This first chunk is additive. Native fit/query plans and the public boundary
migration are still being written. No numerical body, switch/default or public
inverse-operation behavior has changed. Existing KernelPCA rejects
`fit_inverse_transform=True`; this migration preserves that refusal rather than
adding a new inverse algorithm.

No compile, execution, performance or cross-vendor identity result is claimed.
Root owns the combined build and the native test lane owns standalone fixtures.

## Additive query plans and exports

`core/kpca_native_plan.mojo` adds `KPCAKernelOptions`, `KPCAFitRoute`,
`KPCAScratch`, `KPCAEigenRoom`, `KPCAAlphaRequest`, `KPCAQueryRequest`,
`KPCANeighborDriver(XnDriver)` and `KPCACenteredQuery`. Fused queries call the
existing `kpca_transform` once; the old-items owner retains kernel/rowsum/divide/
center/matmul in order. All private buffers are native zero-filled stores, and
the existing eigen rooms retain their capacities and prefix-copy contract.
A fieldwise request consumes a local options value with `options^`.

The neighbor images export `kpca_neighbors_native_api_v1`; the decomp images
export `kpca_decomp_native_api_v1`. Host inventories and the neighbor generator
template include these exports. No unrelated kernel was regenerated. The
public boundary is not switched yet.

Dense centering has a separately identified inherited driver error; the frozen
`checks/kpca_dense_centering_repro.mojo` is assigned to root for execution before
a separate numerical repair. The completed migration will include that repair
using existing reduction kernels. Native correctness fixtures remain owned by
the test lane; no lane compilation or execution has occurred.
