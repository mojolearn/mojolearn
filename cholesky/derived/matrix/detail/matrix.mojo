# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The triangular and diagonal helpers of RAFT's dense matrix utilities.

PORT of `raft/matrix/detail/matrix.cuh` at RAFT `ebf9268`
(`upstream/raft-v26.08.00`), the four kernels a Cholesky lane needs:
`getUpperTriangular` / `copyUpperTriangular` (`:196-222`),
`copyVectorToMatrixDiagonal` / `initializeDiagonalMatrix` (`:225-262`),
`copyVectorFromMatrixDiagonal` / `getDiagonalMatrix` (`:240-273`) and
`matrixDiagonalInverse` / `getDiagonalInverseMatrix` (`:277-294`).
Transliterated. **COPY, DO NOT IMPROVE**, with the two departures below.

Nothing else from that header is here: `slice`, `columnWiseSort`, `gather`,
`getL2Norm` and the print helpers belong to other sections or to nobody, and
`cholesky/NOT_IMPLEMENTED.tsv` records them as such rather than leaving them as an
invisible gap (PORTING_RULES rule 3).

# =========================================================================
# DEVIATION 1644: ROW-MAJOR, WHERE THEIRS IS COLUMN-MAJOR.
#
# Every kernel above indexes `[i + j * lda]`, which is column-major, because
# RAFT hands these matrices to cuBLAS and cuSOLVER and those are Fortran
# libraries. This tree is row-major and contiguous throughout -- `gemm/
# IDENTICAL_FP32_CONTRACT.md` section 2, and every buffer in `cholesky/` --
# so the same four kernels index `[i * n + j]`.
#
# It is a RELABELLING and not an algorithm change: a symmetric matrix's
# lower triangle in row-major storage occupies exactly the cells its upper
# triangle occupies in column-major storage, which is the same identity
# `gbdt/lapack/linear_system.mojo` already leans on when it factors a
# row-major 'L' against CatBoost's column-major 'U' request. `theirs`
# reads `getUpperTriangular` where ours reads `get_lower_triangular`, and
# the two write the same bytes for the same matrix.
#
# The `k = min(n_rows, n_cols)` generality is kept even though every caller
# here is square, because dropping it would be improving.
# =========================================================================

# =========================================================================
# DEVIATION 1645: `matrix_diagonal_inverse` IS PORTED AND IS UNREACHABLE
# FROM ANY IDENTITY PATH IN THIS LANE, ON PURPOSE.
#
# Their `getDiagonalInverseMatrix` (`:290-294`) exists so that later work can
# MULTIPLY by a reciprocal instead of DIVIDING. It is a real speed idea: a
# float divide is several times the latency of a multiply on every column,
# and a triangular solve performs one per row.
#
# It is a second rounding. `1/x` rounds, then `t * (1/x)` rounds again, and
# the composition is not the correctly-rounded quotient -- there are inputs
# where it is one ulp off, and `original/numerics.mojo`'s row-49 note
# records `identical_div` as correctly rounded on every column measured.
# `cholesky/original/trsm.mojo` therefore divides, always, and
# `CHOL_SAB_TRSM_RECIPROCAL` is the arm that swaps this shape in so the gate
# can be shown to see the difference.
#
# The kernel is ported anyway, because PORTING_RULES rule 3 says an unported
# file is visible and a mis-ported one is not, and because a FAST-mode caller
# outside this lane may legitimately want it. It has no caller HERE and
# `cholesky/DERIVATION_MAP.tsv` says so in the same words.
# =========================================================================
"""

from std.gpu import block_dim, block_idx, thread_idx

from original.numerics import ftz, identical_div


#: `dim3 block(64)` at `:218`, `:259`, `:270` and `:292`. SCHEDULING; every
#: kernel here is one thread per cell with no fold, so the width reaches
#: nothing numeric. Kept at their value because copying it costs nothing and
#: because a changed constant is a question a reader has to answer.
comptime RAFT_MATRIX_TPB = 64


def get_lower_triangular_kernel(
    src: MutPointer[Float32, MutAnyOrigin],
    dst: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    n_cols_in: Int32,
    k_in: Int32,
):
    """`getUpperTriangular` (`:204-212`), relabelled row-major.

    Theirs:

        idx_t i = idx % m, j = idx / m;
        if (i < k && j < k && j >= i) { dst[i + j * k] = src[idx]; }

    Ours walks the same linear index over an `n_rows x n_cols` ROW-major
    source, so `i = idx / n_cols` and `j = idx % n_cols`, and keeps the
    `j >= i` clause the other way up (`j <= i`) because in row-major storage
    that is the same set of cells theirs selects. `dst` is `k x k`.
    """
    var m = Int(n_rows_in)
    var n = Int(n_cols_in)
    var k = Int(k_in)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= m * n:
        return
    var i = idx // n
    var j = idx % n
    if i < k and j < k and j <= i:
        dst.unsafe_store(i * k + j, src.unsafe_load(idx))


def copy_vector_to_matrix_diagonal_kernel(
    vec: MutPointer[Float32, MutAnyOrigin],
    matrix: MutPointer[Float32, MutAnyOrigin],
    lda_in: Int32,
    k_in: Int32,
):
    """`copyVectorToMatrixDiagonal` (`:232-237`). `matrix[idx + idx * lda]`
    and `matrix[idx * lda + idx]` are the same cell, so this one is
    layout-free: the diagonal is the diagonal."""
    var lda = Int(lda_in)
    var k = Int(k_in)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx < k:
        matrix.unsafe_store(idx * lda + idx, vec.unsafe_load(idx))


def copy_vector_from_matrix_diagonal_kernel(
    vec: MutPointer[Float32, MutAnyOrigin],
    matrix: MutPointer[Float32, MutAnyOrigin],
    lda_in: Int32,
    k_in: Int32,
):
    """`copyVectorFromMatrixDiagonal` (`:247-252`). The diagonal out.

    THE CALLER IN THIS LANE is `cholesky/original/potrf.mojo::chol_logdet`,
    which extracts `diag(L)` with this kernel, records it as the card stage
    `chol.diag`, and folds the logs of it. Recording the diagonal separately
    is not decoration: a Gaussian process that disagrees across two columns
    disagrees in ONE diagonal entry long before it disagrees in the scalar
    they sum to, and a scalar hash cannot say which.
    """
    var lda = Int(lda_in)
    var k = Int(k_in)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx < k:
        vec.unsafe_store(idx, matrix.unsafe_load(idx * lda + idx))


def matrix_diagonal_inverse_kernel(
    a: MutPointer[Float32, MutAnyOrigin],
    len_in: Int32,
):
    """`matrixDiagonalInverse` (`:283-288`). `in[idx + idx*len] = 1.0 / ...`.

    **NO CALLER IN THIS LANE, BY DESIGN. DEVIATION 1645.** The divide is
    routed through `identical_div` and flushed through `ftz` so that a FAST
    caller elsewhere gets this tree's arithmetic rather than a second one,
    but nothing on an identity path here calls it, and the reason is in the
    file banner: a reciprocal is a second rounding.
    """
    var n = Int(len_in)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx < n:
        var d = ftz(a.unsafe_load(idx * n + idx))
        a.unsafe_store(idx * n + idx, ftz(identical_div(Float32(1.0), d)))
