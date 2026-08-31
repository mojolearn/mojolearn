# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Column means, centering, and the covariance product.

MOSTLY NOT PORTS OF FILES. These stand in for `raft::stats::mean`,
`raft::stats::cov` and `raft::stats::meanAdd`, which `raft/linalg/detail/
pca.cuh::pca_fit` calls in that order. RAFT is a general library this tree
does not mirror file for file, so what is reproduced is the CALL SITE and its
semantics. Same rule as `core/row_norms.mojo`.

THE GRAM PRODUCT IS NOT IN THIS FILE ANY MORE
---------------------------------------------
`raft::stats::cov` and `tsvd_fit` both ask cuBLAS for `A^T A`. This file used
to carry a hand-written contraction for that shape, a port of the
`isRowMajor == false` arm of `raft/linalg/detail/contractions.cuh` with a
split-K row partition of our own on top, on the belief that MAX's matmul could
not do it because `transpose_a` is unsupported. **`transpose_a` is still
unsupported and that belief was still wrong**: `Xt = transpose(X)` makes
`Xt . Xt^T` the N-T shape MAX does support, and `transpose_kernel` below is
the twenty lines that get there. The ported contraction and its split-K
reduction were dead code by then and are deleted. About 250 lines, none of
it reached.

`core/gemm.mojo::gemm_tn` now DISPATCHES that shape: the transpose route
above for outputs with enough tiles to fill the device, and the split-K
Gram kernel in `core/gram_splitk.mojo` for the tile-starved outputs the
shipped fits actually have, where the vendor matmul measured ~25 GFLOP/s
(one 32x32 output = one tile of parallelism). The tuned-BLAS rule stands
where the tuned kernel has parallelism; the measurement, not the rule, is
what put the small shapes on the hand-written kernel.

THE ONE SEMANTIC THAT IS EASY TO GET WRONG
------------------------------------------
THEIR `pca_fit` centers the input IN PLACE, computes the covariance from
the centered data, and then calls `raft::stats::meanAdd` to put the input
BACK (`pca.cuh:138`). The input is an in-out parameter that ends the call
unchanged. Ours meets that contract per arm (DEVIATION 42): the split-K arm
fuses the centering into the Gram kernel's read and never writes x, so
`shift_columns_kernel` does not launch there at all; the fallback arm runs
their center + restore pair exactly, and there skipping the restore leaves
the caller's matrix silently centered, which is invisible until they use it
for something else.

`cov` uses the `n_rows - 1` denominator, the sample covariance, because
`pca_fit` later multiplies the explained variances by exactly `n_rows - 1` to
recover singular values.

THE TWO FLOAT FOLDS ARE PINNED (DEVIATION 523)
-----------------------------------------------
`column_mean_kernel` and `xty_kernel` used to fold with
`max.gpu.primitives.block.sum`, whose internal cross-lane stage follows the
HARDWARE warp width -- 32 on Apple and NVIDIA, **64 on AMD's CDNA
wavefront**. Float addition is not associative, so the AMD column combined
different partials and produced a different mean and a different `A^T b`,
and PCA subtracts that mean from every row while OLS solves the normal
equations against that right-hand side. IDENTITY_PATHS row 20's defect, in
row 20's own directory, named in that file and left for this lane.

DEVIATION 510 had already added `K_LIB_COLUMN_STATS` to
`lib_block_bounds_a_float_fold`, which pins the block WIDTH to one value on
every column. That is necessary and NOT sufficient: pinning the width does
not change what the library does inside that width. The fix is REPLACE, not
PIN -- `core/pinned_reduce.pinned_block_sum`, the library call verbatim
under FAST and a halving tree with no lane primitive in it under IDENTICAL.

Also closed here: `xty_kernel`'s `acc += x * y` through
`identical_mul_add` (row 9), and row 10's `ftz` at every float seam these
kernels write -- the mean's quotient, `A^T b`, the centering store, the
pseudo-inverse's column division, and the covariance scale.

THE CENTERING STORE WAS HALF-CLOSED WHEN THIS ARRIVED, which is worth
recording because it is not the shape the audit expected. DEVIATION 522
(`core/gram_splitk.mojo`, the same day) flushed the FUSED arm's centered
read to `ftz(ftz(x) - ftz(mu))` and left this kernel -- the OTHER arm of
a pair `check_gram_centered_fused` asserts is bit-equal per cell --
unflushed. On Metal both spellings are inert and the check passes either
way; on CUDA and HIP under IDENTICAL the two arms would have staged
different tiles from the same input. `shift_columns_kernel` now mirrors
that spelling exactly.
"""

from mojo_only.kernel_matrix import (
    K_LIB_COLUMN_STATS,
    TARGET_COLUMN,
    lib_block_size_for,
)


from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.memory import stack_allocation

from core.pinned_reduce import pinned_block_sum
from mojo_only.numerics import ftz, identical_mul_add


# READ FROM THE MATRIX, not restated here. `mojo_only/kernel_matrix.mojo`
# owns every tunable in this tree; changing TARGET_COLUMN there rebuilds
# this kernel for another vendor with no edit in this file.
comptime STATS_TPB = lib_block_size_for[K_LIB_COLUMN_STATS, TARGET_COLUMN]()


def column_mean_kernel(
    mu: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    n_cols_in: Int32,
):
    """`raft::stats::mean` along columns. One block per column."""
    var n_rows = Int(n_rows_in)
    var n_cols = Int(n_cols_in)
    var col = Int(block_idx.x)
    var tid = Int(thread_idx.x)

    var acc = Float32(0.0)
    var r = tid
    while r < n_rows:
        acc += x.unsafe_load(r * n_cols + col)
        r += STATS_TPB

    # `cub::BlockReduce`'s counterpart, and NOT `block.sum` any more.
    # DEVIATION 523, IDENTITY_PATHS row 20's defect in this directory:
    # `max.gpu.primitives.block.sum` is correct and tuned, and its
    # internal cross-lane stage folds at the HARDWARE warp width -- 32
    # on Apple and NVIDIA, 64 on AMD's CDNA wavefront. Float addition is
    # not associative, so the same STATS_TPB values reduce through two
    # different association trees on two vendors, and this mean is
    # subtracted from every row before the covariance. `pinned_block_sum`
    # IS the library call under FAST (bit for bit, so the shipped build
    # is unmoved) and a halving tree with no lane primitive in it under
    # IDENTICAL. See `core/pinned_reduce.mojo` and VENDOR_LIBRARIES.md.
    #
    # The contract is met at both call sites (`pca.mojo:200`,
    # `gram_splitk_check.mojo:248`): `block_dim == STATS_TPB`, there is
    # no early return above this line, so every thread of the block
    # reaches the fold, and a thread whose stride found no row arrives
    # with `acc == 0.0`.
    var s0 = ftz(pinned_block_sum[STATS_TPB](acc))
    if tid == 0:
        # IDENTITY_PATHS row 10, and the reason this is two statements:
        # the quotient is a float SEAM another kernel reads, and a
        # division is a seam-PRODUCING operation (a centered column of
        # near-constant data drives it toward the denormal range). An
        # intermediate inside one expression cannot be reached by `ftz`
        # on a non-FTZ backend, so the fold result and the quotient each
        # get their own local. Bitwise inert on Metal, which already
        # flushes; it aligns CUDA and HIP to Metal.
        var m = ftz(s0 / Float32(n_rows))
        mu.unsafe_store(col, m)


def shift_columns_kernel(
    x: MutPointer[Float32, MutAnyOrigin],
    mu: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    n_cols_in: Int32,
    sign_in: Float32,
):
    """Center (`sign = -1`) or restore (`sign = +1`), in place.

    Both directions in one kernel because they are the same operation and
    because pairing them makes the restore hard to forget.

    AUDITED FOR DEVIATION 523. Two classes were looked for:

    - CONTRACTION (row 9): none reachable, and `identical_mul_add` is
      deliberately NOT called. `sign_in` is exactly +-1.0 at every call
      site, so `sign_in * mu` is an exact sign flip and
      `fma(sign, mu, x)` and `x + (sign * mu)` round to the same bits on
      every backend. The pin is left off rather than added-and-inert so
      that nobody later reads it as evidence that a general `sign_in` is
      safe. It is not: a caller passing anything else makes this a row-9
      site and the pin has to arrive with that caller.
    - SEAM (row 10): PRESENT, AND NOW CLOSED -- the flushes below.
      `x + sign*mu` is the centering cancellation, which is the single
      most likely denormal producer in this file, and the store is a seam
      the Gram product reads.

    **THE SPELLING IS NOT FREE: IT MIRRORS `core/gram_splitk.mojo`
    STATEMENT FOR STATEMENT.** That file's fused arm reads
    `x[t, j] - mu[j]` into its staging tile INSTEAD of reading this
    kernel's output, and `check_gram_centered_fused` asserts the two arms
    agree PER CELL, bitwise. DEVIATION 522 flushed that arm as
    `ftz(ftz(x) - ftz(mu))` (both operands and the difference). Flushing
    only one of the two halves splits the pair on CUDA and HIP while
    passing on Metal, where the flush is inert -- so the half left
    unflushed WAS the open defect, not the half being added. `x + (-1)*mu`
    is bitwise `x - mu` (exact sign flip, and IEEE `a + (-b)` IS `a - b`),
    and `ftz(-m) == -ftz(m)` because `ftz` flushes to a SIGNED zero, so
    the two spellings are the same three flushes in the same places.
    """
    var n_cols = Int(n_cols_in)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= Int(n_rows_in) * n_cols:
        return
    var col = idx % n_cols
    # One local per flush, because an intermediate INSIDE an expression
    # cannot be reached by `ftz` on a non-FTZ backend (row 10's checklist,
    # and `numerics.ftz`'s own docstring says so).
    var xv = ftz(x.unsafe_load(idx))
    var mv = ftz(mu.unsafe_load(col))
    var shifted = ftz(xv + sign_in * mv)
    x.unsafe_store(idx, shifted)


def xty_kernel(
    out_v: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    n_cols_in: Int32,
):
    """`A^T b`. Stands in for `raft::linalg::gemv(..., trans=true)`.

    One block per feature, striding rows. This and the Gram product in
    `core/gemm.mojo::gemm_tn` are the only two things in ordinary least
    squares that touch rows at all; everything after them is
    `n_cols x n_cols`.
    """
    var n_rows = Int(n_rows_in)
    var n_cols = Int(n_cols_in)
    var col = Int(block_idx.x)
    var tid = Int(thread_idx.x)

    var acc = Float32(0.0)
    var r = tid
    while r < n_rows:
        # IDENTITY_PATHS row 9, the contraction pin, and the same class
        # row 19 closed for the L2 accumulators: `acc += x * y` is ONE
        # rounding or TWO at the codegen's whim, and the whim differs per
        # backend. Pinning the fold's shape and leaving this unpinned
        # would still hand two vendors two different `A^T b`.
        # DEVIATION 523.
        acc = identical_mul_add(
            x.unsafe_load(r * n_cols + col), y.unsafe_load(r), acc
        )
        r += STATS_TPB

    # `cub::BlockReduce`'s counterpart, and NOT `block.sum` any more --
    # DEVIATION 523, the same fold defect as `column_mean_kernel` above,
    # on the right-hand side the normal equations are solved against.
    # `pinned_block_sum` is the library call verbatim under FAST and a
    # lane-width-independent halving tree under IDENTICAL. The contract
    # is met at the one call site (`lstsq.mojo:128`, `block_dim ==
    # STATS_TPB`): no early return, every thread reaches the fold, and a
    # thread with no row arrives with `acc == 0.0`.
    var s0 = ftz(pinned_block_sum[STATS_TPB](acc))
    if tid == 0:
        # IDENTITY_PATHS row 10: `out_v` is a float seam `lstsq_eig`'s
        # final `gemv` reads. `s0` is already flushed above, so the store
        # carries the flushed local rather than re-deriving it.
        out_v.unsafe_store(col, s0)


def divide_columns_by_nonzero_kernel(
    qs: MutPointer[Float32, MutAnyOrigin],
    q: MutPointer[Float32, MutAnyOrigin],
    s_vec: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    thresh_in: Float32,
):
    """`QS <- Q invS` with `DivideByNonZero`, `lstsq.cuh`'s matrixVectorOp.

    Column `k` of `Q` is divided by eigenvalue `k`, and a column whose
    eigenvalue is at or below the threshold is ZEROED rather than divided.

    That zeroing is the whole numerical story of solving least squares
    through the normal equations. `A^T A` squares the condition number, so a
    direction the data barely constrains shows up as a tiny eigenvalue, and
    dividing by it would amplify noise without bound. Dropping the direction
    instead is a pseudo-inverse, and it is why their default OLS solver is
    SVD rather than this one.
    """
    var n = Int(n_in)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= n * n:
        return
    var col = idx % n
    var lam = s_vec.unsafe_load(col)
    if lam > thresh_in or lam < -thresh_in:
        # IDENTITY_PATHS row 10, DEVIATION 523. A quotient is the
        # seam-producing operation row 10 names, `qs` is read by
        # `gemm_nt`, and `q / lam` with a barely-above-threshold `lam`
        # is exactly where a denormal appears. The compare above needs
        # no flush: `thresh_in` is 1e-10 at the one call site, twenty-
        # eight orders of magnitude above the largest denormal, so a
        # denormal `lam` takes the zero arm on every backend and can
        # never reach this division.
        qs.unsafe_store(idx, ftz(q.unsafe_load(idx) / lam))
    else:
        qs.unsafe_store(idx, Float32(0.0))


def diagonal_to_vector_kernel(
    out_v: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """Pull the eigenvalues off the diagonal Jacobi leaves behind.

    cuSOLVER hands back a separate eigenvalue array; Jacobi leaves them on
    the diagonal of the matrix it consumed. One kernel bridges the two
    conventions so the rest of the port reads like theirs.

    AUDITED FOR DEVIATION 523, NO CHANGE. There is no arithmetic here at
    all -- one strided load, one store -- so there is no fold and no
    seam-producing operation: a copy cannot make a denormal that did not
    arrive. A denormal that DID arrive is the Jacobi sweep's to flush
    (`decomposition/mojo_only/jacobi_eigh_device.mojo`, DEVIATION 511,
    another lane's file), and flushing it here instead would be inert
    for this file's only consumer anyway:
    `divide_columns_by_nonzero_kernel` zeroes anything below 1e-10, which
    is every denormal.
    """
    var n = Int(n_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < n:
        out_v.unsafe_store(i, a.unsafe_load(i * n + i))


def scale_in_place_kernel(
    a: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    scale_in: Float32,
):
    """Apply cuBLAS's `alpha` after the fact.

    `raft::linalg::gemm` takes `alpha` and folds the scale into the product.
    MAX's `matmul` has no alpha argument, so the `1 / (n_rows - 1)` that turns
    a Gram matrix into a covariance is a separate pass over `n_cols^2`
    elements. That is a deviation in launch count, not in arithmetic, and it
    is tiny: the product is O(rows * cols^2) and this is O(cols^2).
    """
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        # IDENTITY_PATHS row 10, DEVIATION 523. The covariance matrix is
        # the seam PCA's eigensolver reads and `pca_fit` later multiplies
        # back by `n_rows - 1`; the multiply by `1 / (n_rows - 1)` shrinks
        # every entry, so a small off-diagonal covariance at a large
        # `n_rows` is precisely how a denormal is produced here.
        a.unsafe_store(i, ftz(a.unsafe_load(i) * scale_in))



comptime TRANSPOSE_TILE = 32

# CUDA's grid Y and Z dimensions are capped at 65,535; X is 2^31 - 1. This is
# a HARDWARE/DRIVER limit, checked at launch, and exceeding it fails the
# launch with CUDA_ERROR_INVALID_VALUE before any thread runs -- there is no
# partial result and no wrong answer, just an unhandled exception. Named here
# rather than spelled 65535 at a call site, because DEVIATION 1883 is the
# SECOND lane this cap has taken (DEVIATION 1872 was MAX's gemv at
# n = 128256) and it will not be the last.
comptime CUDA_MAX_GRID_YZ = 65535


def transpose_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    src: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    n_cols_in: Int32,
):
    """`dst[n_cols x n_rows] = src[n_rows x n_cols]^T`, tiled through shared.

    NOT A PORT. `linalg.transpose` exists, compiles, and SIGNALS at runtime on
    device buffers — it dispatches into a HOST strided-copy path. This is the
    twenty-line kernel that `core/gemm.mojo` named as the alternative route
    and that nobody had written.

    Tiled and padded so both the read and the write are coalesced: a naive
    transpose is coalesced on exactly one side and strided on the other,
    which costs more than the whole operation is worth.

    AUDITED FOR DEVIATION 523, NO CHANGE, SAID OUT LOUD RATHER THAN
    SKIPPED. This kernel MOVES DATA: load, threadgroup store, barrier,
    threadgroup load, store. Not one arithmetic operation is performed on
    a float value anywhere in it, so it has no fold whose order could
    differ and no operation that could produce a denormal a backend might
    flush differently. Every bit that leaves it is a bit that entered it,
    at a new address. `K_LIB_TRANSPOSE` is classified the same way in
    `lib_block_bounds_a_float_fold` ("moves data").
    """
    var n_rows = Int(n_rows_in)
    var n_cols = Int(n_cols_in)
    var tile = stack_allocation[
        TRANSPOSE_TILE * (TRANSPOSE_TILE + 1),
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()

    var tx = Int(thread_idx.x)
    var ty = Int(thread_idx.y)
    var c0 = Int(block_idx.x) * TRANSPOSE_TILE

    # DEVIATION 1883 -- A GRID-STRIDE LOOP OVER THE ROW TILES, BECAUSE THE
    # ROW TILE INDEX WAS `block_idx.y` AND CUDA CAPS grid_dim.y AT 65,535.
    #
    # This kernel used to take its row tile straight from `block_idx.y`,
    # which forced the caller to launch one block per row tile in Y. On CUDA
    # the Y and Z grid dimensions are capped at 65,535 (X is 2^31 - 1), so
    # at TRANSPOSE_TILE = 32 the LARGEST MATRIX THIS COULD TRANSPOSE WAS
    # 65,535 * 32 = 2,097,120 ROWS. One row past that and the LAUNCH ITSELF
    # is rejected: CUDA_ERROR_INVALID_VALUE, before a single thread runs.
    #
    # MEASURED, H100, 2026-08-26. The classical speed leg ran `ols` and
    # `pca` at their shipped 4,000,000 x 32 and BOTH DIED HERE. Not a wrong
    # answer -- no answer at all, an unhandled exception out of
    # `gemm_tn_via_transpose`, and two lanes came home with a cuML row and
    # no row of ours. It is the same family as DEVIATION 1872 (MAX's gemv
    # over the same 65,535 cap at n = 128256) and it is the second time this
    # cap has taken a lane in one day.
    #
    # THE APPLE RUNS COULD NOT HAVE SEEN IT. Metal has no equivalent
    # threadgroup-grid cap in this range, and every Apple fixture that
    # reaches this kernel is small. A limit that only one vendor imposes,
    # only above a size no local fixture reaches, is exactly the defect a
    # cross-vendor leg exists to find.
    #
    # The fix is the standard grid-stride loop: the caller now caps Y and
    # each block walks the row tiles it owns. `n_row_tiles`, `stride` and
    # the loop bound are all block-uniform -- `block_idx.y` and `grid_dim.y`
    # are the same for every thread in a block -- so every thread in a block
    # runs the SAME number of iterations and the `barrier()` calls below are
    # reached uniformly. A divergent barrier here would be a hang, not a
    # wrong number, so this is the property that matters.
    var n_row_tiles = (n_rows + TRANSPOSE_TILE - 1) // TRANSPOSE_TILE
    var stride = Int(grid_dim.y)
    var t = Int(block_idx.y)
    while t < n_row_tiles:
        var r0 = t * TRANSPOSE_TILE

        var r = r0 + ty
        var c = c0 + tx
        if r < n_rows and c < n_cols:
            tile[ty * (TRANSPOSE_TILE + 1) + tx] = src.unsafe_load(
                r * n_cols + c
            )
        barrier()

        var r2 = c0 + ty
        var c2 = r0 + tx
        if r2 < n_cols and c2 < n_rows:
            dst.unsafe_store(
                r2 * n_rows + c2, tile[tx * (TRANSPOSE_TILE + 1) + ty]
            )
        # THE SECOND BARRIER IS NOT DECORATION. The shared tile is REUSED by
        # the next iteration, and without this a thread that has finished
        # its store can race ahead and overwrite a tile entry another thread
        # in the same block has not read yet. With one tile per launch that
        # hazard did not exist; with a loop it does.
        barrier()
        t += stride
