# lane/apple-fast-lle: LocallyLinearEmbedding's null space on the sparse factor

Written without a Mojo toolchain (cloud peer, 2026-10-02); the first M3 build is the compile check.
The switch defaults OFF and is compiled under FAST + Apple only; IDENTICAL compiles main's code unchanged
(the binding registers the new entry only under the define; the Python kit takes the route only when the entry exists).

| switch | kind | site | what it changes under FAST on Apple |
|---|---|---|---|
| `-D MOJOLEARN_LLE_SPARSE_EIG=1` | build define (x_decomp binding) | `bindings/_mojolearn_x_decomp.mojo` registers `x_decomp_dev_lle_sparse_eig` (`x_decomp/lle_sparse.mojo`); `python/mojolearn/_expansion_decomp.py` `LocallyLinearEmbedding.fit` -> `_lle_sparse_eig` | method='standard', eigen_solver='auto', n > 200, n_components + 1 < 10 (main's `_lle_smallest` policy): the n_components smallest eigenpairs of M = (I - W)^T (I - W) past the constant by LOBPCG on the SPARSE factor. F x and F^T y are launches over the kNN lists (`lle_fx_kernel`) and their device CSC (`lle_resid_kernel`; the CSC from (column, entry) keys sorted by a bitonic network, one writer a cell, no atomics). Block b = n_components + 4 vectors kept orthogonal to u = 1/sqrt(n) (F's null vector, the eigenvector sklearn drops); Jacobi preconditioner T = diag(M)^-1 from the CSC; the search space S = [X, R, P] one n x 3b matrix so its Gram matrices B = S^T S and A = (F S)^T (F S) are two `launch_gemm` calls; ONE single-block kernel (`lle_rr_kernel`, 3b <= 36 cells in threadgroup memory) does the Rayleigh-Ritz: round-robin two-sided Jacobi of B, directions under 5e-5 of the largest B eigenvalue dropped, A' = Z^T A Z, its Jacobi, the b smallest Ritz pairs ascending and the next P's coefficients. The dense I - W is not built and no LU runs when it settles. Every 5 iterations one folded read (Ritz values, the wanted columns' change since the last read, the squared weight sum) applies main's stopping rules (`_LLE_SUBSPACE_TOL` 1e-5, `_LLE_STALL_TOL` 1e-2, `_LLE_NULL_FLOOR` 8 eps of F's rms row norm). Not settled in 600 iterations, or columns not unit: the fit runs main's dense route (`graph_lle_iw` + `_lle_smallest`). The reconstruction error is the sum of the Ritz values, as main's is the sum of the squared Ritz singular values. |

Cause (taxi 17.9 s vs sklearn 1.2 s, istella 17.8 s vs 2.9 s on the 0.8.34 board): after the kNN and the barycenter
weights (both already on the device), `python/mojolearn/_expansion_decomp.py` `_lle_smallest` factors the DENSE n x n
F0 with `k.lu` (10,000^2: an O(n^3) getrf), then runs subspace iteration with two dense n x n `trisolve`s, two
Householder QRs and a Jacobi SVD a step, and `graph_lle_iw` writes the dense n x n I - W first. sklearn's ARPACK works on
the sparse I - W (n (k + 1) nonzeros) with a sparse LU; this arm works on the same sparse operator without any
factorization.

Expected quality: the board checks trustworthiness k=15 of the embedding. The arm returns the same subspace to
`_LLE_SUBSPACE_TOL` when it settles, so the trustworthiness should sit within FAST's run-to-run spread of the dense route's;
`reconstruction_error_` (the sum of the smallest eigenvalues, ~1e-7) is compared only to float32's resolution of |F x|, as
main's. Risk: LOBPCG's convergence rate on this operator (eigenvalues ~1e-7 against a largest of order the squared weight
sums; the barycenter weights at reg = 1e-3 can be large) is unknown without the M3: with the diagonal preconditioner it may
need many of the 600 iterations or not settle, in which case the fit falls back to the dense route and the A/B shows the
LOBPCG time as a loss, never a quality loss. If the arm settles but slowly, the next steps are a better preconditioner
(a few steps of CG on M as the T application) and a larger block.

A/B line (`lle.txt`, light form: one alternation, 2 rounds, taxi first, no -ident line): `lle-sparse-taxi`
(afc_ab_def.sh, binding x_decomp, AFC_FAMILY=algos, "" against `-D MOJOLEARN_LLE_SPARSE_EIG=1`). istella follows a win.

Compile risks to watch (`x_decomp/lle_sparse.mojo`): the `MutPointer[UInt64, MutAnyOrigin]` key buffer built from a
pooled float buffer's address and its `//`, `%` on UInt64; `stack_allocation` arrays handed to the `@always_inline`
`_rr_jacobi(m: ShF32, ...)` helper (the ALS kernels' `_ald` idiom) with barriers inside its loops; the
`lib_smem_page_fits_for[TARGET_COLUMN, LLE_RR_BYTES]()` gate on the 21 KB page; `db.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()`
for the read buffer handed to `lle_gather_kernel`; the `comptime if LLE_SPARSE_EIG:` around `m.def_function` in the
binding's `PyInit`; `rand_cell` imported from `x_decomp.cells`.
