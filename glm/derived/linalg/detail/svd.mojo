# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Derivative work: the upstream file and its pinned commit are recorded in this lane's DERIVATION_MAP.tsv and in this file's own docstring. See NOTICE.
"""`svdEig`: the SVD of a tall matrix through the eigendecomposition of its Gram.

PORT OF `raft/cpp/include/raft/linalg/detail/svd.cuh::svdEig` at RAFT
`661a3b8`. Partial: `svdEig` only (`svdQR`, `svdJacobi` and `svdReconstruction`
are not ported; see `glm/NOT_IMPLEMENTED.tsv`). Do not improve.

Their steps, copied (`svd.cuh:112-171`):

    in_cross_mult = A^T A          gemm, CUBLAS_OP_T / CUBLAS_OP_N
    eigDC(in_cross_mult) -> V, S   cuSOLVER syevd; S ASCENDING
    col_reverse(V); row_reverse(S) -> DESCENDING
    S = sqrt(S)                    seqRoot(S, S, 1, n_cols, set_neg_zero=true)
    if gen_left_vec:
        U = A V                    gemm, CUBLAS_OP_N / CUBLAS_OP_N
        U /= S (per column)        matrixVectorBinaryDivSkipZero<false,true>
                                   -- |S| < 1e-10 leaves the column AS IS

It is `cuml/cpp/src/glm/ridge.cuh::ridgeEig`'s first call, and the only
caller here. **It is the same first two steps as `lstsq_eig`** (`glm/derived/
linalg/detail/lstsq.mojo`): `core/gemm.mojo::gemm_tn` for the Gram and
`jacobi_eigh_kernel` for the eigendecomposition, so everything IDENTITY_PATHS
rows 27 and 31 pinned is reached here unchanged.

THE REVERSE IS A SORT, FOR THE SAME REASON `eig_and_truncate` GIVES
---------------------------------------------------------------------
cuSOLVER returns eigenvalues ascending, so `col_reverse`/`row_reverse` make
them descending. Cyclic Jacobi does not order anything, so a reverse here
would be an arbitrary permutation. The descending ORDER is the semantic and
is produced by a host selection sort of INDICES (no arithmetic; the same
shape `decomposition/derived/linalg/detail/pca.mojo::eig_and_truncate` uses
and for the same reason) applied on the device by `gather_columns_kernel`.
For ridge the order is not even observable in exact arithmetic -- `w = V
diag(.) V^T A^T b` is permutation-invariant -- but the final `w = V S_nnz`
sums over columns IN INDEX ORDER, so the order is a rounding order, and
mirroring theirs is how a cross-vendor card stays comparable stage for
stage.

WHAT IS NOT PORTED: `gen_left_vec == false` returns before `U`; both callers
here pass `true` and the flag is honored anyway. The two-stream overlap is
absent as in `lstsq.mojo`.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from core.column_stats import diagonal_to_vector_kernel
from core.gemm import gemm_nt, gemm_tn
from core.identity_trace import IdentityTrace
from decomposition.original.jacobi_eigh_device import (
    JACOBI_SWEEPS,
    JACOBI_TOL,
    JACOBI_TPB,
    jacobi_eigh_kernel,
)
from glm.derived.matrix.math import (
    MATRIX_ELEM_TPB,
    gather_columns_kernel,
    gather_vector_kernel,
    matrix_vector_binary_div_skip_zero_kernel,
    seq_root_kernel,
)


def _descending_order(
    ctx: DeviceContext, mut s_raw: DeviceBuffer[DType.float32], n: Int
) raises -> List[Int32]:
    """`col_reverse`/`row_reverse`'s semantic for an unordered eigensolver:
    the permutation that puts the eigenvalues DESCENDING. Indices only --
    a selection sort with a strict `>`, O(n^2) on n_cols, no float op."""
    var hs = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.enqueue_copy(dst_ptr=hs.unsafe_ptr(), src_buf=s_raw)
    ctx.synchronize()
    var order = List[Int32]()
    for i in range(n):
        order.append(Int32(i))
    for i in range(n):
        for j in range(i + 1, n):
            var vj = hs.unsafe_ptr().unsafe_load(Int(order[j]))
            var vi = hs.unsafe_ptr().unsafe_load(Int(order[i]))
            if vj > vi:
                var t = order[i]
                order[i] = order[j]
                order[j] = t
    _ = hs^
    return order^


def svd_eig_traced(
    ctx: DeviceContext,
    mut a: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_cols: Int,
    mut s: DeviceBuffer[DType.float32],
    mut u: DeviceBuffer[DType.float32],
    mut v: DeviceBuffer[DType.float32],
    gen_left_vec: Bool,
    mut trace: IdentityTrace,
    tag: String,
) raises:
    """`svdEig(handle, in, n_rows, n_cols, S, U, V, gen_left_vec, stream)`.

    `a` is `n_rows x n_cols` row-major and is NOT modified. `s` is `n_cols`,
    `u` is `n_rows x n_cols` (written only if `gen_left_vec`), `v` is
    `n_cols x n_cols` with the right singular vectors as COLUMNS, descending.
    `tag` prefixes the card's stage names (`ridge.svd.*`).
    """
    var cov = ctx.enqueue_create_buffer[DType.float32](n_cols * n_cols)
    var v_raw = ctx.enqueue_create_buffer[DType.float32](n_cols * n_cols)
    var vt = ctx.enqueue_create_buffer[DType.float32](n_cols * n_cols)
    var s_raw = ctx.enqueue_create_buffer[DType.float32](n_cols)
    var info_buf = ctx.enqueue_create_buffer[DType.float32](3)
    var xa = ctx.enqueue_create_buffer[DType.float32](n_rows * n_cols)
    var xa2 = ctx.enqueue_create_buffer[DType.float32](n_rows * n_cols)
    ctx.synchronize()

    # in_cross_mult <- A^T A. `svd.cuh:132-144`. Through `gemm_tn`'s
    # dispatch: the pinned split-K Gram kernel at the shipped shapes (row
    # 27), refused by name past its capacity under IDENTICAL.
    gemm_tn(ctx, cov, a, xa, xa2, n_cols, n_cols, n_rows)
    ctx.synchronize()
    trace.record_device[DType.float32](
        ctx, tag + ".covA", cov, n_cols * n_cols
    )

    # eigDC -> V, S. `svd.cuh:146`. The device Jacobi (row 31) consumes
    # `cov` and leaves S on its diagonal; `eigDC` ABORTS on a non-zero
    # `dev_info` (`raft/linalg/detail/eig.cuh:149-151`) and so does this.
    ctx.enqueue_function[jacobi_eigh_kernel](
        cov.unsafe_ptr(),
        v_raw.unsafe_ptr(),
        info_buf.unsafe_ptr(),
        Int32(n_cols),
        Int32(JACOBI_SWEEPS),
        Float32(JACOBI_TOL),
        grid_dim=(1, 1, 1),
        block_dim=(JACOBI_TPB, 1, 1),
    )
    ctx.enqueue_function[diagonal_to_vector_kernel](
        s_raw.unsafe_ptr(),
        cov.unsafe_ptr(),
        Int32(n_cols),
        grid_dim=((n_cols + MATRIX_ELEM_TPB - 1) // MATRIX_ELEM_TPB, 1, 1),
        block_dim=(MATRIX_ELEM_TPB, 1, 1),
    )
    var h_info = ctx.enqueue_create_host_buffer[DType.float32](3)
    ctx.enqueue_copy(dst_ptr=h_info.unsafe_ptr(), src_buf=info_buf)
    ctx.synchronize()
    if h_info.unsafe_ptr().unsafe_load(0) == Float32(0.0):
        raise Error(
            "svdEig: the device Jacobi did not converge in "
            + String(JACOBI_SWEEPS)
            + " sweeps on the "
            + String(n_cols)
            + " x "
            + String(n_cols)
            + " Gram matrix; ||offdiag||_F / ||.||_F is still "
            + String(h_info.unsafe_ptr().unsafe_load(1))
            + ". eigDC aborts here too (raft eig.cuh:149)."
        )
    _ = h_info^
    trace.record_device[DType.float32](ctx, tag + ".eigvals", s_raw, n_cols)
    trace.record_device[DType.float32](ctx, tag + ".info", info_buf, 3)

    # col_reverse(V); row_reverse(S). `svd.cuh:148-151`. A permutation of
    # indices, computed on the host, applied on the device; the module
    # docstring says why it is a sort and not a reverse.
    var order = _descending_order(ctx, s_raw, n_cols)
    var d_order = ctx.enqueue_create_buffer[DType.int32](n_cols)
    ctx.enqueue_copy(dst_buf=d_order, src_ptr=order.unsafe_ptr())
    ctx.synchronize()
    trace.record_list_i32(tag + ".order", order)
    ctx.enqueue_function[gather_columns_kernel](
        v.unsafe_ptr(),
        vt.unsafe_ptr(),
        v_raw.unsafe_ptr(),
        d_order.unsafe_ptr(),
        Int32(n_cols),
        grid_dim=((n_cols * n_cols + MATRIX_ELEM_TPB - 1) // MATRIX_ELEM_TPB, 1, 1),
        block_dim=(MATRIX_ELEM_TPB, 1, 1),
    )
    ctx.enqueue_function[gather_vector_kernel](
        s.unsafe_ptr(),
        s_raw.unsafe_ptr(),
        d_order.unsafe_ptr(),
        Int32(n_cols),
        grid_dim=((n_cols + MATRIX_ELEM_TPB - 1) // MATRIX_ELEM_TPB, 1, 1),
        block_dim=(MATRIX_ELEM_TPB, 1, 1),
    )
    # seqRoot(S, S, alpha=1, n_cols, set_neg_zero=true). `svd.cuh:153`.
    ctx.enqueue_function[seq_root_kernel](
        s.unsafe_ptr(),
        Int32(n_cols),
        Float32(1.0),
        Int32(1),
        grid_dim=((n_cols + MATRIX_ELEM_TPB - 1) // MATRIX_ELEM_TPB, 1, 1),
        block_dim=(MATRIX_ELEM_TPB, 1, 1),
    )
    ctx.synchronize()
    trace.record_device[DType.float32](ctx, tag + ".S", s, n_cols)
    trace.record_device[DType.float32](ctx, tag + ".V", v, n_cols * n_cols)

    if gen_left_vec:
        # U <- A V. `svd.cuh:155-168`, CUBLAS_OP_N / CUBLAS_OP_N. `gemm_nt`
        # computes `x . y^T`, so `y` is `V^T`, which `gather_columns_kernel`
        # wrote alongside `V`. Under IDENTICAL the pinned one-thread-per-
        # cell product (row 28); under FAST the vendor matmul.
        gemm_nt(ctx, u, a, vt, n_rows, n_cols, n_cols)
        ctx.synchronize()
        # U /= S per column, |S| < 1e-10 LEAVES THE COLUMN AS IS
        # (`return_zero` defaulted false at `svd.cuh:169`).
        ctx.enqueue_function[matrix_vector_binary_div_skip_zero_kernel](
            u.unsafe_ptr(),
            s.unsafe_ptr(),
            Int32(n_rows),
            Int32(n_cols),
            Int32(0),
            grid_dim=((n_rows * n_cols + MATRIX_ELEM_TPB - 1) // MATRIX_ELEM_TPB, 1, 1),
            block_dim=(MATRIX_ELEM_TPB, 1, 1),
        )
        ctx.synchronize()
        trace.record_device[DType.float32](ctx, tag + ".U", u, n_rows * n_cols)

    # `[[mojo-buffer-freed-at-last-use]]`: keep every scratch alive past the
    # last synchronize that waits on a kernel reading it.
    _ = cov^
    _ = v_raw^
    _ = vt^
    _ = s_raw^
    _ = info_buf^
    _ = xa^
    _ = xa2^
    _ = d_order^
