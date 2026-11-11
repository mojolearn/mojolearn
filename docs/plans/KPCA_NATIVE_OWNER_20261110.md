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
