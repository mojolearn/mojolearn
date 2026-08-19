"""Least squares through the normal equations and an eigendecomposition.

PORT OF `raft/linalg/detail/lstsq.cuh::lstsqEig` at RAFT `9aa17e5`.
Transliterated. Do not improve.

This is cuML's OLS solver `algo = 1` (`cuml/cpp/src/glm/ols.cuh:105`). Their
six steps, copied:

    covA = A^T A                 O(rows * cols^2)
    Ab   = A^T b                 O(rows * cols)
    Q S Q* = eig(covA)           O(cols^3)
    QS   = Q invS                O(cols^2)   with DivideByNonZero
    covA = QS Q^T                O(cols^3)   == inv(A^T A)
    w    = covA Ab               O(cols^2)

**Only the first two touch rows.** That is the same shape as PCA and it is
why this section cost almost nothing: `covariance_kernel` with scale 1 is
step 1, `jacobi_eigh_kernel` is step 3, and `gemm_nt_kernel` is steps 5 and
6. The genuinely new code is `xty_kernel` and the column division.

WHY THEY DEFAULT TO SVD AND NOT TO THIS
---------------------------------------
`olsFit`'s default is `algo = 0`, `lstsqSvdJacobi`. Forming `A^T A` SQUARES
the condition number, so this route loses roughly twice the digits an SVD
route would on an ill-conditioned design. `DivideByNonZero` is the guard: a
direction the data barely constrains appears as a near-zero eigenvalue and
gets DROPPED rather than divided by, which turns the inverse into a
pseudo-inverse.

Porting their non-default solver is a deliberate choice and it is recorded
in `glm/UNPORTED.tsv`: it is the one that reuses machinery this repository
already has, and their SVD route needs a one-sided Jacobi SVD that does not
exist here yet. The accuracy difference is real and belongs in any
comparison against scikit-learn, whose `LinearRegression` uses LAPACK
`gelsd`, an SVD route.

STEP 6 IS ON THE VENDOR GEMV, AND THE SYMBOL IS `gemv_gpu` NOT `gemv`
---------------------------------------------------------------------
Step 6 is `raft::linalg::gemv` upstream and now calls MAX's gemv here too,
through `core/gemm.mojo::gemv_n`. Which symbol is not a detail. The obvious
one is wrong: `linalg.gemv.gemv` takes no `DeviceContext` and no `target` and
its own docstring opens "Computes a CPU matrix-vector product", so handing it
device pointers would be the `linalg.transpose` failure again. The GPU
counterpart in the same module, `linalg.gemv.gemv_gpu`, is the real mirror of
`raft::linalg::gemv` and is what runs.

`VENDOR_LIBRARIES.md` lists `linalg.gemv.gemv` as AVAILABLE, and AVAILABLE in
that table means only that the import compiled, which is exactly how the
wrong symbol got recorded as the finished form. **That row, and its note that
`gemv_gpu` is "Not yet wired", are now stale and need the same correction;
so does the `linalg.matmul` at `n = 1` row, whose workaround column still
points at the ported contraction.** The same sentence appears in
`bench/results/VENDOR_PATH_2026-08-19.md`.

The ported `gemm_nt_kernel` contraction stays REACHABLE rather than deleted,
the way `neighbors/` keeps its ported RAFT radix select beside
`nn.topk.top_k`: pass `use_vendor_gemv=False`. A vendor call whose answer
nothing can be checked against is a vendor call nobody should trust, and the
three OLS checks in `glm/mojo_only/ols_check.mojo` are the only instrument in
this repository that sees this step at all.

THE STREAM OVERLAP IS NOT PORTED
--------------------------------
Theirs computes `A^T A` and `A^T b` on TWO CUDA streams concurrently, with
events to join them (`lstsq.cuh`, `multAbStream`). Mojo's `DeviceContext`
gives one queue here, so ours runs them in sequence. That is a real
throughput deviation and not a correctness one, and it is exactly the kind of
control-plane concurrency `HOST_AND_DEVICE.md` says the incumbents get for
free from CUDA and we do not.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from cluster.mojo_only.reduce_by_key import copy_f32_kernel
from core.gemm import (
    GEMM_MBLK,
    GEMM_THREADS,
    gemm_nt,
    gemm_nt_kernel,
    gemm_tn,
    gemv_n,
)
from core.column_stats import (
    STATS_TPB,
    diagonal_to_vector_kernel,
    divide_columns_by_nonzero_kernel,
    xty_kernel,
)
from decomposition.mojo_only.jacobi_eigh_device import (
    JACOBI_TPB,
    jacobi_eigh_kernel,
)


def lstsq_eig(
    ctx: DeviceContext,
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    mut w: DeviceBuffer[DType.float32],
    mut cov_a: DeviceBuffer[DType.float32],
    mut q: DeviceBuffer[DType.float32],
    mut qs: DeviceBuffer[DType.float32],
    mut s_vec: DeviceBuffer[DType.float32],
    mut ab: DeviceBuffer[DType.float32],
    mut inv: DeviceBuffer[DType.float32],
    mut a_alias: DeviceBuffer[DType.float32],
    mut a_alias2: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_cols: Int,
    use_vendor_gemv: Bool = True,
) raises:
    """`w = inv(A^T A) A^T b`, their step order.

    `use_vendor_gemv` selects step 6's implementation: `True` (the default)
    calls MAX's `linalg.gemv.gemv_gpu`, `False` runs the ported RAFT
    contraction. See the comment at that step for why both exist.
    """
    # covA <- A^T A. alpha = 1, so scale 1: a Gram matrix, not a covariance.
    # covA <- A^T A. `raft::linalg::gemm(CUBLAS_OP_T, CUBLAS_OP_N, alpha=1)`,
    # so the tuned matmul with `transpose_a` and no scale. This is the ONLY
    # step here that touches rows, and it was still on the hand-written
    # contraction while every other section had moved: OLS sat at 28 ms
    # across five benchmark rounds because nothing I changed was on its path.
    gemm_tn(ctx, cov_a, a, a_alias, a_alias2, n_cols, n_cols, n_rows)

    # Ab <- A^T b. Theirs overlaps this with the line above on a second
    # stream; see the module docstring.
    ctx.enqueue_function[xty_kernel](
        ab.unsafe_ptr(),
        a.unsafe_ptr(),
        b.unsafe_ptr(),
        Int32(n_rows),
        Int32(n_cols),
        grid_dim=(n_cols, 1, 1),
        block_dim=(STATS_TPB, 1, 1),
    )
    ctx.synchronize()

    # Q S Q* <- covA. Jacobi consumes covA and leaves S on its diagonal.
    ctx.enqueue_function[jacobi_eigh_kernel](
        cov_a.unsafe_ptr(),
        q.unsafe_ptr(),
        Int32(n_cols),
        Int32(80),
        Float32(1.0e-10),
        grid_dim=(1, 1, 1),
        block_dim=(JACOBI_TPB, 1, 1),
    )
    ctx.enqueue_function[diagonal_to_vector_kernel](
        s_vec.unsafe_ptr(),
        cov_a.unsafe_ptr(),
        Int32(n_cols),
        grid_dim=((n_cols + 255) // 256, 1, 1),
        block_dim=(256, 1, 1),
    )
    ctx.synchronize()

    # QS <- Q invS, with DivideByNonZero.
    ctx.enqueue_function[divide_columns_by_nonzero_kernel](
        qs.unsafe_ptr(),
        q.unsafe_ptr(),
        s_vec.unsafe_ptr(),
        Int32(n_cols),
        Float32(1.0e-10),
        grid_dim=((n_cols * n_cols + 255) // 256, 1, 1),
        block_dim=(256, 1, 1),
    )
    # inv <- QS Q^T == Q invS Q^T == inv(A^T A)
    gemm_nt(ctx, inv, qs, q, n_cols, n_cols, n_cols)

    # w <- inv Ab.
    #
    # RAFT does not call gemm here: it calls `raft::linalg::gemv`
    # (`lstsq.cuh`, "w <- covA Ab"), because a matrix against ONE vector is a
    # different BLAS routine with a different tuning. Expressing it as a
    # matmul with `n = 1` produced zeros for some coefficients, which is what
    # sent me to read their line again.
    #
    # THE OBVIOUS SWAP IS THE WRONG SYMBOL. `linalg.gemv.gemv` is HOST-ONLY.
    # Checked, not assumed: its signature is
    # `gemv[parallelize: Bool, elementwise_lambda_fn](c_buf, a_buf, b_buf)`
    # with **no `ctx: DeviceContext` and no `target`**, and its own docstring
    # opens "Computes a CPU matrix-vector product". It is the same tell that
    # caught `nn.cumsum` in `VENDOR_LIBRARIES.md`: the GPU-capable calls in
    # this toolchain (`matmul`, `top_k`, `argsort`, `gather`) all carry a
    # context and this one does not. Recorded so nobody re-derives it.
    #
    # THE SYMBOL THAT IS RIGHT is `linalg.gemv.gemv_gpu`, same module, wrapped
    # as `core/gemm.mojo::gemv_n`. The orientation below is READ OFF
    # `max/kernels/src/linalg/gemv.mojo` at tag `max/v26.5.0` (the toolchain
    # pinned here is max 26.5.0), not inferred from the fact that it compiles:
    #
    #   * `gemv_gpu` derives its dimensions from C and A ONLY. It calls
    #     `GemmShape.get`, which returns `(c.dim[0], c.dim[1], a.dim[1])` and
    #     documents that B is skipped because B may be pre-packed. So with
    #     c = `w` shaped `(n_cols, 1)` and a = `inv` shaped
    #     `(n_cols, n_cols)`, it sees m = n_cols, n = 1, k = n_cols.
    #   * `n == 1` with a float32 A selects `GEMVAlgorithm.GEMV_KERNEL`, whose
    #     body is `accum += a[global_warp_id * k + idx] * b[idx]` over
    #     `idx < k`, then `c[global_warp_id] = accum`, one warp per output
    #     row. That IS `w[i] = sum_j inv[i][j] * ab[j]` for row-major `inv`,
    #     which is the product this step wants.
    #   * `transpose_b` stays FALSE. This is the part worth not guessing: the
    #     `transpose_b == True` arm of `gemv_gpu_dispatch` SWAPS a and b and
    #     passes `(n, m, k)`, so at n = 1 it would launch one warp and write
    #     one coefficient instead of `n_cols` of them.
    #   * `ab` is shaped `(n_cols, 1)` and not `(1, n_cols)`. `GEMV_KERNEL`
    #     reads B linearly so both give the same answer THERE, but the
    #     `MATMUL_NAIVE` fallback in the same dispatcher indexes B as
    #     `(k, n)`. `(n_cols, 1)` is correct under both arms.
    #   * `pdl_level` keeps its `PDLLevel.ON` default. PDL is Hopper-only and
    #     gated on `_SUPPORT_PDL_LAUNCH`, which is
    #     `has_nvidia_gpu_accelerator() and compute >= H100`, so on Metal the
    #     grid-dependency barriers compile out and `pdl_launch_attributes`
    #     returns an empty list. Nothing Apple-specific is being relied on.
    #
    # THE PORTED CONTRACTION STAYS REACHABLE, `use_vendor_gemv=False`. It is
    # the only thing this vendor call's answer can be checked against, and it
    # is what runs if a backend ever lacks a tuned gemv. Speed is NOT the
    # argument either way: this is `n_cols x n_cols` against `n_cols`, a
    # 32 x 32 problem in the benchmark, so the swap can only ever buy noise.
    # Faithfulness to their call is the argument.
    if use_vendor_gemv:
        gemv_n(ctx, w, inv, ab, n_cols, n_cols)
    else:
        ctx.enqueue_function[gemm_nt_kernel](
            w.unsafe_ptr(),
            inv.unsafe_ptr(),
            ab.unsafe_ptr(),
            Int32(n_cols),
            Int32(1),
            Int32(n_cols),
            grid_dim=(1, (n_cols + GEMM_MBLK - 1) // GEMM_MBLK, 1),
            block_dim=(GEMM_THREADS, 1, 1),
        )
    ctx.synchronize()
