# KernelRidge, Nystroem and RBFSampler

Design note for `mojolearn.parallel_classical.fit_kernel_method`,
`apply_kernel_method` and `transform_rbf_sampler`. IDENTICAL mode only; every
multi-device result must have the exact bits of the one-device result.

## KernelRidge and Nystroem: the SVM kernel-row seam

`kernel_methods/checks/kernel_matrix.mojo::km_kernel_matrix` forms the linear
and RBF kernels through `svm/impl/distance/kernel_matrices.mojo::kernel_op`,
and the polynomial and sigmoid kernels from the same call at a linear
parameter block followed by a per-cell epilogue. `kernel_op` reads its owner
count from the calling thread's device plan (`core/device_plan.mojo`;
`MOJOLEARN_SVM_DEVICE_COUNT` without one) and moves whole output rows of `a . b^T` (plus
the RBF expansion on the original norm bytes) to owners, the path the SVC/SVR
driver qualified. Each output row keeps every contraction term of its FP32-v1
dot. The per-cell epilogues then run on the root over the gathered matrix.

The drivers are one native call each (`family_km_par_fit` /
`family_km_par_apply`, `bindings/kernel_methods_native.mojo`): in the caller's
process they bind the selected devices to the calling thread (rank r on
`devices[r]`, the root on `devices[0]`) and run the estimator's own `fit`,
`predict` and `transform` (docs/plans/NATIVE_PARALLEL_DRIVER_20261024.md):

- KernelRidge fit: kernel rows as above; the ridge on the root; the
  factorization with whole trailing-update rows and the multi-target solve
  with whole target columns (`docs/multi_gpu/cholesky.md`). The Cholesky owners
  read the same plan, bound for this operation only.
- KernelRidge predict: cross-kernel rows `K(X, X_fit_)`; the dual product on
  the root.
- Nystroem fit: basis indices from the device permutation (unchanged), the
  basis kernel's rows, the Jacobi eigendecomposition and normalization on the
  root.
- Nystroem transform: cross-kernel rows `K(X, components_)`; the
  normalization product on the root.

`kernel='laplacian'` forms an L1 distance matrix through `pairwise_distance`,
which has no distributed row path here, so the drivers refuse it by name
rather than run it on one device under a multi-device name.
`kernel='precomputed'` is already refused by the estimators.

## RBFSampler: global feature and component IDs

`RBFSampler.fit` reads only `n_features` and draws `random_weights_` and
`random_offset_` as position-mapped Philox values indexed by feature and
component. There is no data to partition and the fitted state is the same
bytes wherever it is drawn. `transform` is `scale * cos(X . W + offset)`: an
identical GEMM whose output row `i` contracts only input row `i`, then a
per-cell epilogue. `transform_rbf_sampler` splits the query into whole row
ranges (`rows_per_shard`), sends each range with the fitted estimator to a
one-device worker, and joins the outputs by copying rows in input order.

## Gates

`tools/parallel_kernel_methods_check.py`: KernelRidge dual coefficients and
predictions for four kernels across 1 to 515 rows and 1 to 3 targets;
Nystroem components, indices, normalization, eigenvalues and transforms;
RBFSampler transforms across ragged shard sizes; refusals and failed-fit
publication. The same check against a kernel-methods binding built with
`-D MOJOLEARN_CHOLESKY_PARALLEL_SABOTAGE=1` must fail. The Cholesky and SVM
native gates cover the underlying row and column partitions.

No speed or capacity claim is made. The root holds the full data, kernel
matrix and factor.
