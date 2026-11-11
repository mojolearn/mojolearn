# Dense KernelPCA global-mean repair

Root compiled and ran frozen reproducer `0890c974fe` (lane `6a4856f068`).
Compilation passed; execution failed the independent global-mean assertion as
intended. All canaries passed. For K=`[[1,2],[2,4]]`, the old path returned column
means `[1.5,3]`, total words `[0.75,1.5]` and centered
`[[-1.25,-1.75],[-1.75,-1.25]]`. The independent mean is `2.25` and centered
matrix is `[[0.25,-0.25],[-0.25,0.25]]`. Root preserves raw evidence in
`~/mojolearn-evidence/python-mojo-migration-20261110/kpca-centering-before-result.json`
and its corresponding run log; the original reproducer source remains intact.

`kpca_fit_global_mean` now sums the n column means as an n×1 vector through
existing `XD_COLSUM`, whose executor uses its blocked parallel reduction for
large n, and divides the single result by n with existing Xn `scale_div`.
The Float32 divisor rounding is unchanged. The neighbor and decomposition
provider images are independently pinned across both native calls.

The current public dense-fit path calls this shared helper and retains its
initial one-word `_fit_all` output. Removed: the n-word second column-sum,
n-word replacement `_fit_all`, and n-word scale. This fixes the driver shape
and reduces those intermediate transfers to one word. No tuned numerical body,
resident RBF/Lanczos path, eigensolver, default, or switch changed. Dense outputs
may change as a correctness fix; previous-version output bits are not a gate.

`checks/kpca_dense_centering_regression.mojo` calls the same helper that the
product uses, with the original independent oracle and guards surrounding the
single-word total and all other outputs. It does not merely change the expected
value in the frozen failing reproduction. Root owns its compile/run, plus the
ongoing native test lane's centering/eigen/query cases. This lane has executed
none of those checks and makes no performance or cross-vendor identity claim.
