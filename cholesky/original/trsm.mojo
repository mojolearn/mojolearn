# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Triangular solves with multiple right-hand sides, one order everywhere.

NOT A PORT, and the reason is DEVIATION 1631: every triangular solve in
cuML, cuVS and RAFT is `cublastrsm` -- `raft/linalg/detail/
cholesky_r1_update.cuh:77` for the rank-one update, `cuml/src/solver/
lars_impl.cuh:349` and `:369` for the two LARS back-solves, and
`cusolverDnpotrs` for cuVS's ScaNN solve (`cuvs/src/neighbors/scann/detail/
scann_avq.cuh:190`). cuBLAS and cuSOLVER are CLOSED. There is no source to
transliterate, so `PORTING_RULES` 0b-i's narrow exception applies and the
question becomes what to call instead -- and under `NUMERIC_IDENTICAL` the
answer cannot be MAX's equivalent either, for the reason
`neighbors/original/pinned_distance_tile.mojo` gives about `linalg.matmul`:
a vendor library picks its own blocking, and a blocking of a triangular solve
IS a summation order.

So the shape below is chosen the way that file chooses its tile: the
SIMPLEST correct one rather than the fastest one.

THE SHAPE, and the two properties it buys
------------------------------------------
**ONE THREAD OWNS ONE RIGHT-HAND-SIDE COLUMN and performs the whole
substitution for it, in registers and in `b`'s own storage.** No float
crosses a thread boundary anywhere in this file: there is no shared-memory
staging, no block fold, no cross-block combination and no warp primitive.

1. The summation order is a pure function of `n` and of the loop written
   here, and of nothing else -- not the grid, not the block width, not the
   device, not `nrhs`.
2. Therefore launch invariance and BATCH invariance are properties of the
   kernel's shape rather than of a check that happens to pass. A column
   solved alone and the same column solved inside 4096 others execute
   character for character the same instructions on the same bytes.

The price is stated rather than hidden, exactly as the pinned distance tile
states its own: a blocked trsm turns the solve into GEMM work and reads `L`
once per tile, where this reads `i` floats of `L` per row per column. It is
O(n^2 * nrhs) global loads with no reuse. `cholesky/README.md` carries this
under WHAT IS OWED; no speed number exists for it, because nothing has run.

THE TOTAL ORDER, stated because IDENTITY_PATHS asks every tie to state one
--------------------------------------------------------------------------
There is no tie to break in a triangular solve -- no max, no min, no argmax,
no selection of any kind -- so no total order is needed and none is invented.
What IS pinned is the SUMMATION order, and it is: **`k` ASCENDING in every
loop of this file, in the forward solve and in the back solve alike.** The
back solve walks its ROWS descending (`i = n-1 ... 0`, which is what a back
substitution is) and its inner sum ASCENDING over `k = i+1 ... n-1`. Writing
that inner loop descending because the outer one is descending is the
plausible mistake, and `CHOL_SAB_PANEL_DESCENDING` is the arm that proves the
gate can see it.

THE DIVIDE IS A DIVIDE (DEVIATION 1643)
----------------------------------------
Every diagonal division is spelled `identical_div(t, l_ii)` and never
`t * (1 / l_ii)`. RAFT ships the reciprocal shape -- `matrix/detail/
matrix.cuh:283-295`'s `matrixDiagonalInverse` inverts a whole diagonal in
place so later work can multiply -- and it is a real speed idea and a real
second rounding. `identical_div` is row 49's seam
(`original/numerics.mojo`), correctly rounded on every column measured;
`1/x` followed by `x*y` is two roundings whose composition is not the
correctly-rounded quotient. `CHOL_SAB_TRSM_RECIPROCAL` is that arm.
`cholesky/derived/matrix/detail/matrix.mojo` mirrors the RAFT kernel anyway,
with its own banner saying it is unreachable from any identity path here.
"""

from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace
from cholesky.original.chol_sabotage import (
    CHOL_SAB_NONE,
    sabotage_trsm_lower_kernel,
    sabotage_trsm_upper_kernel,
)
from original.numerics import ftz, identical_div, identical_mul_add


#: SCHEDULING. Threads per block for the solve kernels. Free in both modes,
#: and `check_launch_invariance` varies it precisely to say so out loud.
comptime CHOL_SOLVE_TPB = 256


def trsm_lower_kernel(
    l: MutPointer[Float32, MutAnyOrigin],
    b: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    nrhs_in: Int32,
    ld_in: Int32,
):
    """`L X = B`, forward substitution, `X` written over `B`.

    `l` is row-major with ROW STRIDE `ld` and only the LOWER triangle of its
    leading `n x n` block, including the diagonal, is read. `b` is
    `n x nrhs` row-major, so a right-hand side is a COLUMN of `b` with
    stride `nrhs`; thread `j` owns column `j` from the first row to the last
    and reads back only values it wrote itself.

    `ld` is separate from `n` because RAFT's rank-one update solves against
    the LEADING `n-1` block of a matrix stored at the full stride
    (`cholesky_r1_update.cuh:77-88` passes `L` with `ld` and a size of
    `n - 1`), and a second spelling of forward substitution for that one
    caller would be a second thing to get wrong. Every other caller passes
    `ld == n`. It is an addressing parameter and reaches no arithmetic.
    """
    var n = Int(n_in)
    var nrhs = Int(nrhs_in)
    var ld = Int(ld_in)
    var j = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if j >= nrhs:
        return

    for i in range(n):
        var t = ftz(b.unsafe_load(i * nrhs + j))
        # k ASCENDING. See THE TOTAL ORDER in this file's header.
        for k in range(i):
            var lik = ftz(l.unsafe_load(i * ld + k))
            var bk = ftz(b.unsafe_load(k * nrhs + j))
            t = ftz(identical_mul_add(-lik, bk, t))
        var lii = ftz(l.unsafe_load(i * ld + i))
        b.unsafe_store(i * nrhs + j, ftz(identical_div(t, lii)))


def trsm_upper_kernel(
    l: MutPointer[Float32, MutAnyOrigin],
    b: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    nrhs_in: Int32,
    ld_in: Int32,
):
    """`L^T X = B`, back substitution, `X` written over `B`.

    `l` is the SAME lower-triangular factor: `L^T[i, k] = L[k, i]`, so this
    reads `l[k * ld + i]` and no transposed copy is ever materialized. Rows
    descend; the inner sum ASCENDS.
    """
    var n = Int(n_in)
    var nrhs = Int(nrhs_in)
    var ld = Int(ld_in)
    var j = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if j >= nrhs:
        return

    for ii in range(n):
        var i = n - 1 - ii
        var t = ftz(b.unsafe_load(i * nrhs + j))
        # k ASCENDING over the rows BELOW i, even though i itself descends.
        for k in range(i + 1, n):
            var lki = ftz(l.unsafe_load(k * ld + i))
            var bk = ftz(b.unsafe_load(k * nrhs + j))
            t = ftz(identical_mul_add(-lki, bk, t))
        var lii = ftz(l.unsafe_load(i * ld + i))
        b.unsafe_store(i * nrhs + j, ftz(identical_div(t, lii)))


def trsm_panel_kernel(
    a: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    j0_in: Int32,
    nb_in: Int32,
    n_trail_in: Int32,
):
    """`L21 = A21 . L11^{-T}`, the blocked factorization's panel solve.

    In place inside `a`: `A21` is the `n_trail x nb` block at rows
    `[j0+nb, n)` and columns `[j0, j0+nb)`, and `L11` is the already
    factored `nb x nb` diagonal block at `[j0, j0+nb)^2`.

    A right-side solve against `L11^T` is a LEFT-side forward solve against
    `L11` performed on each ROW independently, which is why this is one
    thread per trailing row and not one thread per column: row `r` computes
    `y` with `L11 y = A21[r, :]^T` walking `c` ascending, and writes `y` back
    over `A21[r, :]`. Nothing is shared between rows, so the row-to-thread
    mapping is scheduling and free.
    """
    var n = Int(n_in)
    var j0 = Int(j0_in)
    var nb = Int(nb_in)
    var n_trail = Int(n_trail_in)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= n_trail:
        return
    var r = j0 + nb + idx

    for c in range(nb):
        var jc = j0 + c
        var t = ftz(a.unsafe_load(r * n + jc))
        # k ASCENDING over the panel's own columns only. Columns before j0
        # were subtracted by the PREVIOUS panels' trailing updates, which is
        # what makes this right-looking rather than left-looking; summing
        # them again here would double-count and is the classic transcription
        # error in a blocked factorization.
        for k in range(j0, jc):
            var lrk = ftz(a.unsafe_load(r * n + k))
            var lck = ftz(a.unsafe_load(jc * n + k))
            t = ftz(identical_mul_add(-lrk, lck, t))
        var ljj = ftz(a.unsafe_load(jc * n + jc))
        a.unsafe_store(r * n + jc, ftz(identical_div(t, ljj)))


# ===========================================================================
# THE HOST-VISIBLE SOLVES
# ===========================================================================


def trsm_lower(
    ctx: DeviceContext,
    mut l: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    n: Int,
    nrhs: Int,
    mut trace: IdentityTrace,
    tag: StringSlice = "chol.trsm.lower",
    tpb: Int = CHOL_SOLVE_TPB,
    sabotage: Int = CHOL_SAB_NONE,
    ld: Int = 0,
) raises:
    """Solve `L X = B` in place. `l` is `n x n` row-major lower triangular
    (its strict upper triangle is never read), `b` is `n x nrhs` row-major.

    `ld` is the factor's row stride; `0` means `n`, which is every caller
    but RAFT's rank-one update.

    ASYNCHRONOUS except for the trace record, which drains by construction
    (`core/identity_trace.mojo` rule 4). The caller keeps both buffers alive
    past its own `ctx.synchronize()`.

    `tag` is the card stage this solve records. It must be UNIQUE WITHIN A
    TRACE -- `IdentityTrace._emit` raises on a repeat, deliberately, so two
    solves in one fit have to name themselves apart.
    """
    if n <= 0 or nrhs <= 0:
        raise Error(
            "trsm_lower: n and nrhs must both be positive, got n="
            + String(n)
            + " nrhs="
            + String(nrhs)
        )
    var lda = ld if ld > 0 else n
    if lda < n:
        raise Error(
            "trsm_lower: ld="
            + String(lda)
            + " is smaller than n="
            + String(n)
        )
    if len(l) < (n - 1) * lda + n:
        raise Error(
            "trsm_lower: the factor buffer holds "
            + String(len(l))
            + " floats, an n = "
            + String(n)
            + " factor at ld = "
            + String(lda)
            + " needs "
            + String((n - 1) * lda + n)
        )
    if len(b) < n * nrhs:
        raise Error(
            "trsm_lower: the right-hand-side buffer holds "
            + String(len(b))
            + " floats, "
            + String(n)
            + " x "
            + String(nrhs)
            + " needs "
            + String(n * nrhs)
        )
    var grid = (nrhs + tpb - 1) // tpb
    if sabotage != CHOL_SAB_NONE:
        ctx.enqueue_function[sabotage_trsm_lower_kernel](
            l.unsafe_ptr(),
            b.unsafe_ptr(),
            Int32(n),
            Int32(nrhs),
            Int32(lda),
            Int32(sabotage),
            grid_dim=(grid, 1, 1),
            block_dim=(tpb, 1, 1),
        )
    else:
        ctx.enqueue_function[trsm_lower_kernel](
            l.unsafe_ptr(),
            b.unsafe_ptr(),
            Int32(n),
            Int32(nrhs),
            Int32(lda),
            grid_dim=(grid, 1, 1),
            block_dim=(tpb, 1, 1),
        )
    trace.record_device(ctx, tag, b, n * nrhs)


def trsm_upper(
    ctx: DeviceContext,
    mut l: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    n: Int,
    nrhs: Int,
    mut trace: IdentityTrace,
    tag: StringSlice = "chol.trsm.upper",
    tpb: Int = CHOL_SOLVE_TPB,
    sabotage: Int = CHOL_SAB_NONE,
    ld: Int = 0,
) raises:
    """Solve `L^T X = B` in place. `l` is the same LOWER factor `trsm_lower`
    takes -- there is no separate upper operand and no transpose is
    materialized. Same contract as `trsm_lower` otherwise."""
    if n <= 0 or nrhs <= 0:
        raise Error(
            "trsm_upper: n and nrhs must both be positive, got n="
            + String(n)
            + " nrhs="
            + String(nrhs)
        )
    var lda = ld if ld > 0 else n
    if lda < n:
        raise Error(
            "trsm_upper: ld="
            + String(lda)
            + " is smaller than n="
            + String(n)
        )
    if len(l) < (n - 1) * lda + n:
        raise Error(
            "trsm_upper: the factor buffer holds "
            + String(len(l))
            + " floats, an n = "
            + String(n)
            + " factor at ld = "
            + String(lda)
            + " needs "
            + String((n - 1) * lda + n)
        )
    if len(b) < n * nrhs:
        raise Error(
            "trsm_upper: the right-hand-side buffer holds "
            + String(len(b))
            + " floats, "
            + String(n)
            + " x "
            + String(nrhs)
            + " needs "
            + String(n * nrhs)
        )
    var grid = (nrhs + tpb - 1) // tpb
    if sabotage != CHOL_SAB_NONE:
        ctx.enqueue_function[sabotage_trsm_upper_kernel](
            l.unsafe_ptr(),
            b.unsafe_ptr(),
            Int32(n),
            Int32(nrhs),
            Int32(lda),
            Int32(sabotage),
            grid_dim=(grid, 1, 1),
            block_dim=(tpb, 1, 1),
        )
    else:
        ctx.enqueue_function[trsm_upper_kernel](
            l.unsafe_ptr(),
            b.unsafe_ptr(),
            Int32(n),
            Int32(nrhs),
            Int32(lda),
            grid_dim=(grid, 1, 1),
            block_dim=(tpb, 1, 1),
        )
    trace.record_device(ctx, tag, b, n * nrhs)


def cho_solve(
    ctx: DeviceContext,
    mut l: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    n: Int,
    nrhs: Int,
    mut trace: IdentityTrace,
    tpb: Int = CHOL_SOLVE_TPB,
    sabotage: Int = CHOL_SAB_NONE,
) raises:
    """`A X = B` given `A = L L^T`, in place over `B`. cuSOLVER's `potrs`.

    Two stages, recorded as `chol.solve.forward` and `chol.solve.back`, so a
    cross-vendor diff of the solve lands on the substitution that moved
    rather than on the answer. `L y = B` then `L^T X = y`, in that order,
    which is what `potrs` with `uplo = LOWER` is.

    The factor must have come from `potrf_lower` with `info == 0`. A factor
    carrying the partial result of a FAILED factorization has zeros or
    uninitialized cells on its diagonal and this will divide by them; the
    host entry (`cholesky/estimator.mojo`) refuses that case by name, and
    this device-level form trusts its caller exactly as `potrs` does.
    """
    trsm_lower(ctx, l, b, n, nrhs, trace, "chol.solve.forward", tpb, sabotage)
    trsm_upper(ctx, l, b, n, nrhs, trace, "chol.solve.back", tpb, sabotage)
