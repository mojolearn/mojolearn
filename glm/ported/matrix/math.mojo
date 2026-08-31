# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Derivative work: the upstream file and its pinned commit are recorded in this lane's PORTED_MAP.tsv and in this file's own docstring. See NOTICE.
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
`OLS_NONZERO_THRESH` (`glm/UNPORTED.tsv`), carried rather than corrected
because `glm/ported/` is COPY-DO-NOT-IMPROVE.

LAYOUT. RAFT is column-major and every call here is the
`<rowMajor=false, bcastAlongRows=true>` instantiation, which means "the
vector is indexed by COLUMN". Our matrices are row-major, so the same
semantic -- column `j` of the matrix against `vec[j]` -- is spelled
`idx % n_cols`. Nothing else changes.

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

from mojo_only.numerics import ftz, identical_sqrt


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
