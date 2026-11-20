# Dense KernelPCA centering reproducer

`checks/kpca_dense_centering_repro.mojo` uses the existing CPU Xn descriptor and
exact matrix `[[1,2],[2,4]]`. The independent column means are `[1.5,3]`, global
mean `2.25`, and centered result `[[0.25,-0.25],[-0.25,0.25]]`.

The current public dense-fit path instead calls `colsum(cols, 1, n)`, then
scales its n outputs by n. Since colsum reduces rows, its result equals cols;
the centering operation reads the first resulting value, predicted `0.75`.
The resident path uses the appropriate opposite-axis reduction.

No one-word overflow occurs in the current public path: despite allocating an
initial one-word `all_`, `bindings/kpca_native.mojo:141` replaces it with n words
before the second scale operation. The reproducer keeps this exact extent and
places canaries before/after every scratch/output buffer. It prints all values
before the independent numerical assertion, which is expected to fail on the
current code. This is a reproduction, not a replacement golden result.

Root owns compilation and execution. No test, timing or identity run was
performed by the lane. Preserve the failing evidence, then repair only the
erroneous driver reduction using existing parallel kernels in a separate
commit; retain the resident/Lanczos/eigen kernels and their defaults.
