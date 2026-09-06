# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The RAFT matrix primitives `ridgeSolve` and `svdEig` call, one kernel each.

PORT OF `raft/cpp/include/raft/matrix/detail/math.cuh` at RAFT `661a3b8`
(plus `raft/linalg/detail/add.cuh::addScalar`, which lives here rather than
in a one-function file). Partial: only the entries `cuml/cpp/src/glm/
ridge.cuh` and `raft/linalg/detail/svd.cuh::svdEig` reach. Do not improve.

WHY THESE ARE PORTED AND NOT WRITTEN AS "obvious" ONE-LINERS
-------------------------------------------------------------
Every one of them carries a THRESHOLD COMPARISON or a sign rule, and two of
them are the same name with DIFFERENT semantics selected by a flag:

    matrixVectorBinaryDivSkipZero(..., return_zero=false)   |b| < 1e-10 -> a   (LEFT AS IS)
    matrixVectorBinaryDivSkipZero(..., return_zero=true)    |b| < 1e-10 -> 0
    setSmallValuesZero(thres)           a <= thres && -a <= thres -> 0
    seqRoot(set_neg_zero=true)          a < 0 -> 0, else sqrt(a * scalar)
    power(scalar)                       scalar * a * a

`svdEig` uses the first form on `U /= S` and `ridgeSolve` uses the second on
`S /= (S^2 + alpha)`; writing either from memory gets the other. The
comparisons are DISCRETE outputs of a float, IDENTITY_PATHS row 32's class:
a last-bit move in a singular value at the boundary changes WHICH arm a
column takes, not its fifth decimal. The thresholds are ABSOLUTE, on
quantities that scale with the data -- the same recorded deviation as
`OLS_NONZERO_THRESH` (`glm/NOT_IMPLEMENTED.tsv`), carried rather than corrected
because `glm/impl/` is COPY-DO-NOT-IMPROVE.

LAYOUT. RAFT is column-major. The calls `ridgeSolve` and `svdEig` make are
the `<rowMajor=false, bcastAlongRows=true>` instantiation, which means "the
vector is indexed by COLUMN". Our matrices are row-major, so the same
semantic -- column `j` of the matrix against `vec[j]` -- is spelled
`idx % n_cols`. Nothing else changes.

CORRECTED 2026-09-01. That paragraph used to open "every call here is the
`<false, true>` instantiation", and it is no longer every call: `olsFit`'s
sample-weight block (`ols.cuh:99-110`, `:129-140`) reaches the SAME two
functions with `bcastAlongRows = false`, where the vector is indexed by
ROW and is `n_rows` long. Those are the four kernels under the banner at
the bottom of this file, and the banner carries the derivation.

IDENTITY. One thread per output cell, no fold, no cross-thread combination
anywhere, so the block width is SCHEDULING (`numerics.mojo`'s distinction).
Row 10's `ftz` at every store that is a seam another kernel reads, and
row 10's `identical_sqrt` for `seqRoot` because Mojo's `std.math.sqrt`
lowers to an APPROXIMATE PTX sqrt on NVIDIA (IDENTITY_PATHS row 31's
finding). Row 9 has no consumer here: `scalar * a * b` with `scalar == 1`
is one exact multiply followed by one rounding multiply, and there is no
add to contract it into.
"""

from std.gpu import block_dim, block_idx, thread_idx

from checks.numerics import ftz, identical_pow, identical_sqrt


#: Elementwise launch width. SCHEDULING: one thread owns one cell.
comptime MATRIX_ELEM_TPB = 256


def set_small_values_zero_kernel(
    inout_v: MutPointer[Float32, MutAnyOrigin], n_in: Int32, thres: Float32
):
    """`setSmallValuesZero(inout, len, thres)`: `a <= thres && -a <= thres`
    becomes 0. Note `<=`, and note that a NaN fails both tests and is KEPT."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        var a = inout_v.unsafe_load(i)
        if a <= thres and -a <= thres:
            inout_v.unsafe_store(i, Float32(0.0))


def power_kernel(
    out_v: MutPointer[Float32, MutAnyOrigin],
    in_v: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    scalar: Float32,
):
    """`power(in, out, scalar, len)`: `scalar * a * a`, left to right."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        var a = in_v.unsafe_load(i)
        var sa = ftz(scalar * a)
        out_v.unsafe_store(i, ftz(sa * a))


def add_scalar_kernel(
    inout_v: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    scalar: Float32,
):
    """`raft::linalg::addScalar(out, in, scalar, len)`, the in-place call
    `ridgeSolve` makes (`out == in`). One pointer because Mojo refuses the
    same buffer passed twice to one launch, and because in place is what
    their call site does."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        inout_v.unsafe_store(i, ftz(inout_v.unsafe_load(i) + scalar))


def seq_root_kernel(
    inout_v: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    scalar: Float32,
    set_neg_zero: Int32,
):
    """`seqRoot(inout, scalar, len, set_neg_zero)`, the in-place overload
    `svdEig` calls (`svd.cuh:153`, `S, S`).

    `sqrt` through `identical_sqrt` (row 10's NVIDIA half): under IDENTICAL
    the portable correctly-rounded sqrt, under FAST the stdlib's device
    path verbatim."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        var a = inout_v.unsafe_load(i)
        if set_neg_zero != 0 and a < Float32(0.0):
            inout_v.unsafe_store(i, Float32(0.0))
        else:
            var p = ftz(a * scalar)
            inout_v.unsafe_store(i, ftz(identical_sqrt(p)))


def matrix_vector_binary_div_skip_zero_kernel(
    data: MutPointer[Float32, MutAnyOrigin],
    vec: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    n_cols_in: Int32,
    return_zero: Int32,
):
    """`matrixVectorBinaryDivSkipZero<false, true>(data, vec, n_row, n_col,
    stream, return_zero)`: column `j` of `data` divided by `vec[j]`, and where
    `|vec[j]| < 1e-10` the cell is left AS IS (`return_zero == 0`) or ZEROED
    (`return_zero != 0`). The two arms are the two call sites; see the
    module docstring."""
    var n_cols = Int(n_cols_in)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= Int(n_rows_in) * n_cols:
        return
    var b = vec.unsafe_load(idx % n_cols)
    if abs(b) < Float32(1.0e-10):
        if return_zero != 0:
            data.unsafe_store(idx, Float32(0.0))
        # else: `return a`, the cell is untouched
    else:
        data.unsafe_store(idx, ftz(data.unsafe_load(idx) / b))


def matrix_vector_binary_mult_kernel(
    data: MutPointer[Float32, MutAnyOrigin],
    vec: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    n_cols_in: Int32,
):
    """`matrixVectorBinaryMult<false, true>(data, vec, n_row, n_col)`:
    column `j` of `data` times `vec[j]`, in place."""
    var n_cols = Int(n_cols_in)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= Int(n_rows_in) * n_cols:
        return
    var b = vec.unsafe_load(idx % n_cols)
    data.unsafe_store(idx, ftz(data.unsafe_load(idx) * b))


def gather_columns_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    dst_t: MutPointer[Float32, MutAnyOrigin],
    src: MutPointer[Float32, MutAnyOrigin],
    order: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
):
    """`raft::matrix::col_reverse` on an `n x n` basis, generalised to a
    PERMUTATION, writing the permuted basis AND its transpose.

    `svdEig` calls `col_reverse(V)` because cuSOLVER returns eigenvalues
    ASCENDING and a reverse makes them descending. Our Jacobi leaves them
    in rotation order, so the descending order is a permutation `order`
    computed on the host (a sort of INDICES, no arithmetic) and applied
    here: `dst[:, j] = src[:, order[j]]`. `dst_t` is the same matrix
    transposed, which is the operand shape `core/gemm.mojo::gemm_nt` needs
    for `U = A V` (it computes `x . y^T`, so `y` must be `V^T`). Moving
    data only: every bit that leaves is a bit that entered, at a new
    address, so there is nothing to flush.
    """
    var n = Int(n_in)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= n * n:
        return
    var p = idx // n
    var j = idx % n
    var src_col = Int(order.unsafe_load(j))
    var v = src.unsafe_load(p * n + src_col)
    dst.unsafe_store(p * n + j, v)
    dst_t.unsafe_store(j * n + p, v)


def gather_vector_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    src: MutPointer[Float32, MutAnyOrigin],
    order: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
):
    """`raft::matrix::row_reverse` on the eigenvalue vector, as the same
    permutation `gather_columns_kernel` applies to the basis."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        dst.unsafe_store(i, src.unsafe_load(Int(order.unsafe_load(i))))


# ===========================================================================
# THE `<false, false>` INSTANTIATION -- THE VECTOR IS INDEXED BY ROW
# ===========================================================================
#
# Everything above is `<rowMajor=false, bcastAlongRows=true>`, the arm
# `ridgeSolve` and `svdEig` reach, where the vector is indexed by COLUMN.
# `olsFit`'s sample-weight block (`ols.cuh:99-110`, `:129-140`) calls the
# SAME two functions with the OTHER template argument,
# `<rowMajor=false, bcastAlongRows=false>`, and that is a different
# operation on a different-length vector. It is worth spelling the
# derivation out, because raft's own forwarder disagrees with itself about
# the vector's length and only one of the two is what runs.
#
# `matrixVectorBinaryMult<false, false>(input, sample_weight, n_rows, n_cols)`
# forwards to `matrixVectorOp<false, false>(data, data, vec, D = n_col,
# N = n_row, mul_op)` (`raft/matrix/detail/math.cuh:211-216`). There
# `apply = (rowMajor == bcastAlongRows) ? ALONG_ROWS : ALONG_COLUMNS`
# resolves to ALONG_ROWS, and the matrix view is COL-major `(N, D)`
# (`raft/linalg/detail/matrix_vector_op.cuh:41-59`). `linewise_op` then
# takes `nLines = extent(1) = n_col` and `lineLen = extent(0) = n_row` for a
# col-major layout, and with `alongLines = true` computes
# `out[i, j] = op(in[i, j], vec[i])` over col-major indexing
# `[i + lineLen * j]` -- so `i` is the ROW and the vector has `lineLen =
# n_rows` entries (`raft/matrix/linewise_op.cuh:55-59`, `:79-93`).
#
# THE FORWARDER'S OWN VIEW SAYS `n_cols` AND IS WRONG. `matrixVectorOp`
# builds the vector view with `bcastAlongRows ? N : D`, which is `D =
# n_col` here. Nothing reads that extent -- `linewise_op` is handed
# `vecs.data_handle()...`, a raw pointer, and derives the length from
# `lineLen` -- so the run-time behaviour is the `n_rows` one above, which is
# also the only reading consistent with `sample_weight` being an `n_rows`
# vector at the call site. Recorded so nobody re-derives it from the
# forwarder and ports the transpose of this kernel.
#
# In OUR row-major layout the same cell `(i, j)` is `idx = i * n_cols + j`,
# so "row `i`" is `idx // n_cols` where the column arm above writes
# `idx % n_cols`. That one operator is the whole difference and it is why
# these are separate kernels rather than a flag: a flag on the index
# arithmetic is exactly the shape a reader mis-reads.


def sqrt_elementwise_kernel(
    inout_v: MutPointer[Float32, MutAnyOrigin], n_in: Int32
):
    """`raft::linalg::sqrt(out, in, len, stream)` (`raft/linalg/sqrt.cuh:40`),
    the in-place call `olsFit` makes on `sample_weight` (`ols.cuh:100`).

    `unaryOp(out, in, len, sqrt_op{})`, one thread per element, no fold. The
    sqrt goes through `identical_sqrt` for the reason `seq_root_kernel`
    above gives and it is the same finding: Mojo's `std.math.sqrt` lowers to
    an APPROXIMATE PTX sqrt on NVIDIA (IDENTITY_PATHS row 31), and here the
    result multiplies every cell of the design matrix, so one ulp in it is
    one ulp in every row of `A` and in `b`.
    """
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        inout_v.unsafe_store(i, ftz(identical_sqrt(inout_v.unsafe_load(i))))


def power_scalar_kernel(
    inout_v: MutPointer[Float32, MutAnyOrigin], n_in: Int32, p: Float32
):
    """`raft::linalg::powerScalar(out, in, scalar, len)` (`raft/linalg/
    power.cuh:43-46`): `unaryOp(pow_const_op(scalar))`, i.e. `pow(a, p)`.

    `olsFit` calls it once, with `p = 2`, to UNDO the `sqrt` it took on the
    caller's `sample_weight` (`ols.cuh:140`); the doc comment on `olsFit`'s
    parameter says out loud that "this vector is modified during the
    computation", and this line is the restore. `identical_pow` because a
    host/vendor `powf` is not the same function on every backend and this
    value is handed back to the caller.
    """
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        inout_v.unsafe_store(i, ftz(identical_pow(inout_v.unsafe_load(i), p)))


def row_vector_binary_mult_kernel(
    data: MutPointer[Float32, MutAnyOrigin],
    vec: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    n_cols_in: Int32,
):
    """`matrixVectorBinaryMult<false, false>(data, vec, n_row, n_col)`: ROW
    `i` of `data` times `vec[i]`, in place. See the banner above."""
    var n_cols = Int(n_cols_in)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= Int(n_rows_in) * n_cols:
        return
    var b = vec.unsafe_load(idx // n_cols)
    data.unsafe_store(idx, ftz(data.unsafe_load(idx) * b))


def row_vector_binary_div_skip_zero_kernel(
    data: MutPointer[Float32, MutAnyOrigin],
    vec: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    n_cols_in: Int32,
    return_zero: Int32,
):
    """`matrixVectorBinaryDivSkipZero<false, false>(data, vec, n_row, n_col,
    stream, return_zero)`: ROW `i` of `data` divided by `vec[i]`, and where
    `|vec[i]| < 1e-10` the cell is left AS IS (`return_zero == 0`) or ZEROED.

    `olsFit` takes the `return_zero == false` arm (`ols.cuh:130`, the
    default argument), so a row whose weight was zero comes back holding
    `0 * x = 0` rather than being divided by zero -- their restore is
    deliberately NOT exact for a zero weight, and that is theirs, not a
    rounding of ours."""
    var n_cols = Int(n_cols_in)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= Int(n_rows_in) * n_cols:
        return
    var b = vec.unsafe_load(idx // n_cols)
    if abs(b) < Float32(1.0e-10):
        if return_zero != 0:
            data.unsafe_store(idx, Float32(0.0))
        # else: `return a`, the cell is untouched
    else:
        data.unsafe_store(idx, ftz(data.unsafe_load(idx) / b))


def vector_binary_mult_kernel(
    inout_v: MutPointer[Float32, MutAnyOrigin],
    vec: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """`raft::linalg::map_k(labels, n_rows, [](a, b){ return a * b; },
    stream, labels, sample_weight)` -- `ols.cuh:105-110`, the target's half
    of the same scaling. One rounding, one seam."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        inout_v.unsafe_store(i, ftz(inout_v.unsafe_load(i) * vec.unsafe_load(i)))


def vector_binary_div_kernel(
    inout_v: MutPointer[Float32, MutAnyOrigin],
    vec: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """`raft::linalg::map_k(labels, n_rows, [](a, b){ return a / b; }, ...)`
    -- `ols.cuh:135-140`, the target's restore. NOTE it is a plain divide
    with no zero guard, unlike the matrix's, so a zero weight puts an inf or
    a NaN into the caller's `labels`. Copied, not improved: that is what
    theirs does, and `glm/impl/` is COPY-DO-NOT-IMPROVE."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        inout_v.unsafe_store(i, ftz(inout_v.unsafe_load(i) / vec.unsafe_load(i)))
