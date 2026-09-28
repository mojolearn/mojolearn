# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Triangular solves with multiple right-hand sides, one order everywhere.

NO REFERENCE FILE, and the reason is DEVIATION 1631: every triangular solve in
cuML, cuVS and RAFT is `cublastrsm` -- `raft/linalg/detail/
cholesky_r1_update.cuh:77` for the rank-one update, `cuml/src/solver/
lars_impl.cuh:349` and `:369` for the two LARS back-solves, and
`cusolverDnpotrs` for cuVS's ScaNN solve (`cuvs/src/neighbors/scann/detail/
scann_avq.cuh:190`). cuBLAS and cuSOLVER are CLOSED. There is no source to
read, so `CONTRIBUTING.md` (Algorithms and references)'s narrow exception applies and the
question becomes what to call instead -- and under `NUMERIC_IDENTICAL` the
answer cannot be MAX's equivalent either, for the reason
`neighbors/checks/pinned_distance_tile.mojo` gives about `linalg.matmul`:
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
(`checks/numerics.mojo`), correctly rounded on every column measured;
`1/x` followed by `x*y` is two roundings whose composition is not the
correctly-rounded quotient. `CHOL_SAB_TRSM_RECIPROCAL` is that arm.
`cholesky/impl/matrix/detail/matrix.mojo` implements that kernel anyway,
with its own banner saying it is unreachable from any identity path here.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.primitives.warp import shuffle_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext
from cholesky.multi_gpu import CholSolveShard, chol_device_count

from core.identity_trace import IdentityTrace
from cholesky.checks.chol_sabotage import (
    CHOL_SAB_NONE,
    sabotage_trsm_lower_kernel,
    sabotage_trsm_upper_kernel,
)
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, NUMERIC_IDENTICAL, ftz, identical_div, identical_mul_add
from std.sys.info import has_apple_gpu_accelerator
from cholesky.checks.fast_trsm import fast_cho_solve

#: FAST on Apple: `cho_solve` takes the blocked solves of
#: `cholesky/checks/fast_trsm.mojo` (a threadgroup per diagonal block, a
#: parallel update of the rows below) instead of one thread per right-hand
#: side walking all n rows. `-D MOJOLEARN_CHOL_FAST_SERIAL_SOLVE` keeps the
#: pinned substitution.
comptime FAST_CHO_SOLVE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_CHOL_FAST_SERIAL_SOLVE"]()
)


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


# ===========================================================================
# THE SAME ORDER, PARALLEL WHERE THE ORDER ALLOWS (lane/apple-identical-neural,
# 2026-09-26; Apple IDENTICAL, `CHOL_SWEEP_SOLVES`; `-D
# MOJOLEARN_CHOL_SWEEP_SOLVES_OFF` reverts). `trsm_lower_kernel` and
# `trsm_upper_kernel` give each right-hand-side column to ONE thread, so a
# single-target solve at n = 20,000 is 4e8 dependent steps behind global
# loads (KernelRidge 20k took 102 s under IDENTICAL on the M4).
#
# Forward: row i's chain is `t = ftz(fma(-ftz(L[i][k]), x_k, t))` for k
# ascending, and x_k exists as soon as row k finishes -- so every row can
# take its step k the moment x_k is known. `trsm_lower_sweep_kernel` keeps
# every row's t in its own b cell, owned by one thread of a 1024-thread
# block (DEVIATION 6150); per 32-column block
# one simdgroup finishes the 32 diagonal rows (lane r divides, broadcasts
# x, lanes below step), then every later row takes those 32 steps in order.
# Each row's chain is the original's, term for term.
#
# Back: row i's chain runs k = i+1 .. n-1 ASCENDING while rows descend, so
# row i cannot begin before x_{i+1} exists and nothing can be overlapped.
# `trsm_upper_staged_kernel` keeps the one serial chain (thread 0) but has
# the whole block stage its L column into threadgroup memory, so the chain
# waits on threadgroup loads instead of strided global ones.
# ===========================================================================
comptime CHOL_SWEEP_SOLVES = (
    GLOBAL_NUMERIC_MODE != NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_CHOL_SWEEP_SOLVES_OFF"]()
)
comptime CHOL_SWEEP_NT = 1024
comptime CHOL_BACK_NT = 256
comptime CHOL_BACK_CH = 2048
#: Operands of this many back-substitution steps are loaded before the
#: steps run (execution only; `-D MOJOLEARN_CHOL_BACK_UNROLL_OFF` = 1).
#: On Apple the step's `ftz` is the identity: the M4's fp32 FMA never
#: returns a subnormal (`gemm/checks/apple_simdgroup_probe.mojo`, kinds 6-8:
#: 0 subnormal outputs over 4.19M cells each; the attention kernels rely on
#: the same fact), and every operand here is already flushed. Dropping it
#: takes an integer test and select off the serial chain; the stored words
#: are unchanged. `-D MOJOLEARN_CHOL_BACK_FTZ_ON` keeps it.
comptime CHOL_BACK_FMA_NO_FTZ = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_CHOL_BACK_FTZ_ON"]()
)
comptime CHOL_BACK_UNROLL = 1 if is_defined["MOJOLEARN_CHOL_BACK_UNROLL_OFF"]() else (
    64 if is_defined["MOJOLEARN_CHOL_BACK_U64"]() else (16 if is_defined["MOJOLEARN_CHOL_BACK_U16"]() else 32)
)


def trsm_lower_sweep_kernel(
    l: MutPointer[Float32, MutAnyOrigin],
    b: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    nrhs_in: Int32,
    ld_in: Int32,
):
    """`trsm_lower_kernel`'s arithmetic as a blocked column sweep; one block
    per right-hand-side column, any n.

    DEVIATION 6150 (2026-09-28): each row's running t lives in its own `b`
    cell, not in a per-thread register array. Row i belongs to thread
    `i % NT` for the whole sweep (the diagonal simdgroup's lane r IS row
    k0 + r's owner), so no other thread reads or writes that cell and no
    ordering is needed beyond program order. With a 32-float register array
    the 1024-thread pipeline's `maxTotalThreadsPerThreadgroup` fell below
    1024 on the M2 Pro (no Dynamic Caching) and the dispatch was dropped
    with no error: GP alpha, KernelRidge and the cholesky lane moved on M2
    Metal only. NO BIT MOVES: every row's chain is the same terms in the
    same order; `ftz` of an already flushed value is itself."""
    comptime NT = CHOL_SWEEP_NT
    var n = Int(n_in)
    var nrhs = Int(nrhs_in)
    var ld = Int(ld_in)
    var j = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var lane = tid % 32
    var sg = tid // 32
    var xs = stack_allocation[32, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var nb = (n + 31) // 32
    for kb in range(nb):
        var k0 = kb * 32
        var sgd = (k0 % NT) // 32
        if sg == sgd:
            var ri = k0 + lane
            var tv = Float32(0.0)
            if ri < n:
                tv = ftz(b.unsafe_load(ri * nrhs + j))
            for r in range(32):
                if k0 + r < n:
                    var x = Float32(0.0)
                    if lane == r:
                        x = ftz(identical_div(tv, ftz(l.unsafe_load((k0 + r) * ld + k0 + r))))
                        xs[r] = x
                        b.unsafe_store((k0 + r) * nrhs + j, x)
                    x = shuffle_idx(x, UInt32(r))
                    if lane > r and ri < n:
                        tv = ftz(identical_mul_add(-ftz(l.unsafe_load(ri * ld + k0 + r)), x, tv))
        barrier()
        var kc = min(32, n - k0)
        var i = tid
        while i < n:
            if i >= k0 + 32:
                var tv = ftz(b.unsafe_load(i * nrhs + j))
                for r in range(kc):
                    tv = ftz(identical_mul_add(-ftz(l.unsafe_load(i * ld + k0 + r)), xs[r], tv))
                b.unsafe_store(i * nrhs + j, tv)
            i += NT
        barrier()


#: lane/neighbors-apple (2026-09-28): `trsm_lower_sweep_kernel` with RB
#: right-hand-side columns per block. Each thread loads its row's L value
#: once per step and applies it to RB chains, so L is read nrhs / RB times
#: instead of nrhs times (GaussianProcessClassifier.predict_proba solves
#: n_train x n_star at once). Every element's chain is the sweep's, term for
#: term, in both numeric modes (the arithmetic is `trsm_lower_kernel`'s,
#: which FAST's serial column solve also runs), so no word moves. Apple;
#: `-D MOJOLEARN_CHOL_MULTI_RHS_OFF` keeps one column per block.
comptime CHOL_MULTI_RHS = (
    has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_CHOL_MULTI_RHS_OFF"]()
)
comptime CHOL_MR_NT = 1024
comptime CHOL_MR_RB = 8


def trsm_lower_multi_rhs_kernel(
    l: MutPointer[Float32, MutAnyOrigin],
    b: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    nrhs_in: Int32,
    ld_in: Int32,
):
    """`trsm_lower_sweep_kernel`'s schedule for columns [RB * block, +RB),
    any n; columns past nrhs are neither read nor written.

    DEVIATION 6150 applies here too (lane/apple-merged, 2026-09-28): each
    row's RB running values live in its own `b` cells, not in a
    SL * RB = 32-float per-thread register array, which is the shape that
    dropped the 1024-thread sweep dispatch on the M2 Pro. Row i belongs to
    thread `i % NT` for the whole sweep, so program order suffices. Every
    element's chain is the same terms in the same order; `ftz` of an
    already flushed value is itself."""
    comptime NT = CHOL_MR_NT
    comptime RB = CHOL_MR_RB
    var n = Int(n_in)
    var nrhs = Int(nrhs_in)
    var ld = Int(ld_in)
    var j0 = Int(block_idx.x) * RB
    var tid = Int(thread_idx.x)
    var lane = tid % 32
    var sg = tid // 32
    var xs = stack_allocation[32 * RB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var nb = (n + 31) // 32
    for kb in range(nb):
        var k0 = kb * 32
        var sgd = (k0 % NT) // 32
        if sg == sgd:
            var ri = k0 + lane
            var tv = SIMD[DType.float32, RB](0.0)
            if ri < n:
                comptime for c in range(RB):
                    if j0 + c < nrhs:
                        tv[c] = ftz(b.unsafe_load(ri * nrhs + j0 + c))
            for r in range(32):
                if k0 + r < n:
                    var x = SIMD[DType.float32, RB](0.0)
                    if lane == r:
                        var dg = ftz(l.unsafe_load((k0 + r) * ld + k0 + r))
                        comptime for c in range(RB):
                            x[c] = ftz(identical_div(tv[c], dg))
                            xs[r * RB + c] = x[c]
                            if j0 + c < nrhs:
                                b.unsafe_store((k0 + r) * nrhs + j0 + c, x[c])
                    comptime for c in range(RB):
                        x[c] = shuffle_idx(x[c], UInt32(r))
                    if lane > r and ri < n:
                        var lv = ftz(l.unsafe_load(ri * ld + k0 + r))
                        comptime for c in range(RB):
                            tv[c] = ftz(identical_mul_add(-lv, x[c], tv[c]))
        barrier()
        var kc = min(32, n - k0)
        var i = tid
        while i < n:
            if i >= k0 + 32:
                var tv = SIMD[DType.float32, RB](0.0)
                comptime for c in range(RB):
                    if j0 + c < nrhs:
                        tv[c] = ftz(b.unsafe_load(i * nrhs + j0 + c))
                for r in range(kc):
                    var lv = ftz(l.unsafe_load(i * ld + k0 + r))
                    comptime for c in range(RB):
                        tv[c] = ftz(identical_mul_add(-lv, xs[r * RB + c], tv[c]))
                comptime for c in range(RB):
                    if j0 + c < nrhs:
                        b.unsafe_store(i * nrhs + j0 + c, tv[c])
            i += NT
        barrier()


#: lane neighbors-apple2 (2026-09-28): `trsm_lower_multi_rhs_kernel` at 256
#: threads with each thread's later rows taken FOUR AT A TIME. DEVIATION
#: 6150 moved the running values into b cells, and the later-row phase then
#: walked a thread's rows one after another: 8 dependent chains per 32 steps
#: instead of 32, GaussianProcessClassifier.predict_proba 0.26 -> 1.05 s on
#: the M4 Pro. Here a group of four rows (i, i + NT, i + 2 NT, i + 3 NT) is
#: loaded from its b cells, stepped together (32 independent chains), and
#: stored back; the row owner is still `i % NT` (the diagonal simdgroup of
#: block k0 is `(k0 % NT) / 32`, lane r owning row k0 + r), so program order
#: suffices as before. 256 threads keeps the 32-float register tile far
#: below every Apple pipeline limit (the M2 Pro's 1024-thread drop was the
#: 1024-wide launch). Every element's chain is the same terms in the same
#: order. `-D MOJOLEARN_CHOL_MR_ROWWISE` keeps the 1024-thread row-by-row
#: kernel.
comptime CHOL_MR4 = not is_defined["MOJOLEARN_CHOL_MR_ROWWISE"]()
comptime CHOL_MR4_NT = 256
comptime CHOL_MR4_G = 4


def trsm_lower_multi_rhs4_kernel(
    l: MutPointer[Float32, MutAnyOrigin],
    b: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    nrhs_in: Int32,
    ld_in: Int32,
):
    """`trsm_lower_multi_rhs_kernel` with NT = 256 and the later rows in
    groups of CHOL_MR4_G; columns [RB * block, +RB), any n."""
    comptime NT = CHOL_MR4_NT
    comptime RB = CHOL_MR_RB
    comptime G = CHOL_MR4_G
    var n = Int(n_in)
    var nrhs = Int(nrhs_in)
    var ld = Int(ld_in)
    var j0 = Int(block_idx.x) * RB
    var tid = Int(thread_idx.x)
    var lane = tid % 32
    var sg = tid // 32
    var xs = stack_allocation[32 * RB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var nb = (n + 31) // 32
    for kb in range(nb):
        var k0 = kb * 32
        var sgd = (k0 % NT) // 32
        if sg == sgd:
            var ri = k0 + lane
            var tv = SIMD[DType.float32, RB](0.0)
            if ri < n:
                comptime for c in range(RB):
                    if j0 + c < nrhs:
                        tv[c] = ftz(b.unsafe_load(ri * nrhs + j0 + c))
            for r in range(32):
                if k0 + r < n:
                    var x = SIMD[DType.float32, RB](0.0)
                    if lane == r:
                        var dg = ftz(l.unsafe_load((k0 + r) * ld + k0 + r))
                        comptime for c in range(RB):
                            x[c] = ftz(identical_div(tv[c], dg))
                            xs[r * RB + c] = x[c]
                            if j0 + c < nrhs:
                                b.unsafe_store((k0 + r) * nrhs + j0 + c, x[c])
                    comptime for c in range(RB):
                        x[c] = shuffle_idx(x[c], UInt32(r))
                    if lane > r and ri < n:
                        var lv = ftz(l.unsafe_load(ri * ld + k0 + r))
                        comptime for c in range(RB):
                            tv[c] = ftz(identical_mul_add(-lv, x[c], tv[c]))
        barrier()
        var kc = min(32, n - k0)
        var base = tid
        while base < n:
            var tv = InlineArray[SIMD[DType.float32, RB], G](fill=SIMD[DType.float32, RB](0.0))
            var live = InlineArray[Bool, G](fill=False)
            comptime for g in range(G):
                var i = base + g * NT
                if i < n and i >= k0 + 32:
                    live[g] = True
                    comptime for c in range(RB):
                        if j0 + c < nrhs:
                            tv[g][c] = ftz(b.unsafe_load(i * nrhs + j0 + c))
            for r in range(kc):
                comptime for g in range(G):
                    if live[g]:
                        var lv = ftz(l.unsafe_load((base + g * NT) * ld + k0 + r))
                        comptime for c in range(RB):
                            tv[g][c] = ftz(identical_mul_add(-lv, xs[r * RB + c], tv[g][c]))
            comptime for g in range(G):
                if live[g]:
                    var i = base + g * NT
                    comptime for c in range(RB):
                        if j0 + c < nrhs:
                            b.unsafe_store(i * nrhs + j0 + c, tv[g][c])
            base += G * NT
        barrier()


#: lane/neighbors-apple (2026-09-28): `trsm_upper_staged_kernel` whose
#: serial chain reads x from threadgroup memory as well: the newest
#: CHOL_BACK_CH values from a ring thread 0 fills as it solves (index masked,
#: never `%`: a 64-bit modulo per step made the first attempt 3.6x slower),
#: older ones staged by the whole block beside their L column. Operands of
#: CHOL_BACK_UNROLL steps are loaded before the steps run, as in the staged
#: kernel. The same steps in the same order on the same words.
#: `-D MOJOLEARN_CHOL_BACK_RING_OFF` keeps the staged kernel.
comptime CHOL_BACK_RING = CHOL_SWEEP_SOLVES and not is_defined["MOJOLEARN_CHOL_BACK_RING_OFF"]()


def trsm_upper_ring_kernel(
    l: MutPointer[Float32, MutAnyOrigin],
    b: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    nrhs_in: Int32,
    ld_in: Int32,
):
    """`trsm_upper_kernel`'s arithmetic, one serial chain per row (thread 0);
    L column and x operands in threadgroup memory. One block per
    right-hand-side column."""
    comptime NT = CHOL_BACK_NT
    comptime CH = CHOL_BACK_CH
    comptime MASK = CH - 1
    comptime U = CHOL_BACK_UNROLL
    var n = Int(n_in)
    var nrhs = Int(nrhs_in)
    var ld = Int(ld_in)
    var j = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var lk = stack_allocation[CH, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var xs = stack_allocation[CH, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var xr = stack_allocation[CH, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    for ii in range(n):
        var i = n - 1 - ii
        var t = Float32(0.0)
        if tid == 0:
            t = ftz(b.unsafe_load(i * nrhs + j))
        var k = i + 1
        while k < n:
            var cnt = min(CH, n - k)
            var ring = k == i + 1
            for q in range(tid, cnt, NT):
                lk[q] = ftz(l.unsafe_load((k + q) * ld + i))
                if ring:
                    xs[q] = xr[(k + q) & MASK]
                else:
                    # written at least CH rows (and as many barriers) ago
                    xs[q] = b.unsafe_load((k + q) * nrhs + j)
            barrier()
            if tid == 0:
                var q = 0
                while q + U <= cnt:
                    var lv = SIMD[DType.float32, U](0.0)
                    var xv = SIMD[DType.float32, U](0.0)
                    comptime for u in range(U):
                        lv[u] = lk[q + u]
                        xv[u] = xs[q + u]
                    comptime for u in range(U):
                        comptime if CHOL_BACK_FMA_NO_FTZ:
                            t = identical_mul_add(-lv[u], ftz(xv[u]), t)
                        else:
                            t = ftz(identical_mul_add(-lv[u], ftz(xv[u]), t))
                    q += U
                while q < cnt:
                    t = ftz(identical_mul_add(-lk[q], ftz(xs[q]), t))
                    q += 1
            barrier()
            k += cnt
        if tid == 0:
            var x = ftz(identical_div(t, ftz(l.unsafe_load(i * ld + i))))
            b.unsafe_store(i * nrhs + j, x)
            xr[i & MASK] = x
        barrier()


def trsm_upper_staged_kernel(
    l: MutPointer[Float32, MutAnyOrigin],
    b: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    nrhs_in: Int32,
    ld_in: Int32,
):
    """`trsm_upper_kernel`'s arithmetic, one serial chain per row (thread 0),
    its L column staged into threadgroup memory by the whole block; one block
    per right-hand-side column. Only thread 0 reads or writes `b`."""
    comptime NT = CHOL_BACK_NT
    comptime CH = CHOL_BACK_CH
    var n = Int(n_in)
    var nrhs = Int(nrhs_in)
    var ld = Int(ld_in)
    var j = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var lk = stack_allocation[CH, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    for ii in range(n):
        var i = n - 1 - ii
        var t = Float32(0.0)
        if tid == 0:
            t = ftz(b.unsafe_load(i * nrhs + j))
        var k = i + 1
        while k < n:
            var cnt = min(CH, n - k)
            for q in range(tid, cnt, NT):
                lk[q] = ftz(l.unsafe_load((k + q) * ld + i))
            barrier()
            if tid == 0:
                comptime if CHOL_BACK_UNROLL > 1:
                    # The same steps in the same order; the operands of the
                    # next CHOL_BACK_UNROLL steps are loaded first so the
                    # chain waits on arithmetic, not on each global load.
                    comptime U = CHOL_BACK_UNROLL
                    var q = 0
                    while q + U <= cnt:
                        var lv = SIMD[DType.float32, U](0.0)
                        var xv = SIMD[DType.float32, U](0.0)
                        comptime for u in range(U):
                            lv[u] = lk[q + u]
                            xv[u] = b.unsafe_load((k + q + u) * nrhs + j)
                        comptime for u in range(U):
                            comptime if CHOL_BACK_FMA_NO_FTZ:
                                t = identical_mul_add(-lv[u], ftz(xv[u]), t)
                            else:
                                t = ftz(identical_mul_add(-lv[u], ftz(xv[u]), t))
                        q += U
                    while q < cnt:
                        t = ftz(identical_mul_add(-lk[q], ftz(b.unsafe_load((k + q) * nrhs + j)), t))
                        q += 1
                else:
                    for q in range(cnt):
                        t = ftz(identical_mul_add(-lk[q], ftz(b.unsafe_load((k + q) * nrhs + j)), t))
            barrier()
            k += cnt
        if tid == 0:
            b.unsafe_store(i * nrhs + j, ftz(identical_div(t, ftz(l.unsafe_load(i * ld + i)))))


def trsm_panel_guarded_kernel(
    a: MutPointer[Float32, MutAnyOrigin],
    stop: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
    j0_in: Int32,
    nb_in: Int32,
    n_trail_in: Int32,
):
    """`trsm_panel_kernel`, returning at once when `stop[0] != 0` (the
    factor's `info`; `potrf.mojo`'s CHOL_DEFER_INFO)."""
    if stop[0] != Int32(0):
        return
    _trsm_panel_body(a, n_in, j0_in, nb_in, n_trail_in)


def trsm_panel_kernel(
    a: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    j0_in: Int32,
    nb_in: Int32,
    n_trail_in: Int32,
):
    _trsm_panel_body(a, n_in, j0_in, nb_in, n_trail_in)


@always_inline
def _trsm_panel_body(
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
        var swept = False
        comptime if CHOL_MULTI_RHS:
            if nrhs >= CHOL_MR_RB:
                swept = True
                comptime if CHOL_MR4:
                    ctx.enqueue_function[trsm_lower_multi_rhs4_kernel](
                        l.unsafe_ptr(), b.unsafe_ptr(), Int32(n), Int32(nrhs), Int32(lda),
                        grid_dim=((nrhs + CHOL_MR_RB - 1) // CHOL_MR_RB, 1, 1),
                        block_dim=(CHOL_MR4_NT, 1, 1),
                    )
                else:
                    ctx.enqueue_function[trsm_lower_multi_rhs_kernel](
                        l.unsafe_ptr(), b.unsafe_ptr(), Int32(n), Int32(nrhs), Int32(lda),
                        grid_dim=((nrhs + CHOL_MR_RB - 1) // CHOL_MR_RB, 1, 1),
                        block_dim=(CHOL_MR_NT, 1, 1),
                    )
        comptime if CHOL_SWEEP_SOLVES:
            if not swept:
                swept = True
                ctx.enqueue_function[trsm_lower_sweep_kernel](
                    l.unsafe_ptr(), b.unsafe_ptr(), Int32(n), Int32(nrhs), Int32(lda),
                    grid_dim=(nrhs, 1, 1), block_dim=(CHOL_SWEEP_NT, 1, 1),
                )
        if not swept:
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
        var staged = False
        comptime if CHOL_BACK_RING:
            staged = True
            ctx.enqueue_function[trsm_upper_ring_kernel](
                l.unsafe_ptr(), b.unsafe_ptr(), Int32(n), Int32(nrhs), Int32(lda),
                grid_dim=(nrhs, 1, 1), block_dim=(CHOL_BACK_NT, 1, 1),
            )
        comptime if CHOL_SWEEP_SOLVES:
            if not staged:
                staged = True
                ctx.enqueue_function[trsm_upper_staged_kernel](
                    l.unsafe_ptr(), b.unsafe_ptr(), Int32(n), Int32(nrhs), Int32(lda),
                    grid_dim=(nrhs, 1, 1), block_dim=(CHOL_BACK_NT, 1, 1),
                )
        if not staged:
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
    var owners = chol_device_count()
    if owners > 1 and nrhs > 1:
        if sabotage != CHOL_SAB_NONE:
            raise Error("multi-GPU cho_solve does not execute sabotage probes")
        _cho_solve_columns(ctx, l, b, n, nrhs, trace, tpb, owners)
        return
    comptime if FAST_CHO_SOLVE:
        if sabotage == CHOL_SAB_NONE and not trace.enabled:
            fast_cho_solve(ctx, l, b, n, nrhs)
            return
    trsm_lower(ctx, l, b, n, nrhs, trace, "chol.solve.forward", tpb, sabotage)
    trsm_upper(ctx, l, b, n, nrhs, trace, "chol.solve.back", tpb, sabotage)


def _cho_solve_columns(
    ctx: DeviceContext,
    mut l: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    n: Int,
    nrhs: Int,
    mut trace: IdentityTrace,
    tpb: Int,
    count: Int,
) raises:
    """`cho_solve` with whole right-hand-side columns on owners.

    Each column's forward and back substitutions are the original kernels on
    its owner; results are copied as bytes into their original columns, and
    both card stages are recorded on the root after each gather.
    """
    if len(l) < n * n or len(b) < n * nrhs:
        raise Error("multi-GPU cho_solve: a buffer is shorter than its shape")
    if n > 2147483647 // nrhs:
        raise Error("multi-GPU cho_solve exceeds signed 32-bit indexing")
    var active = min(count, nrhs)
    # HOST STAGED. The factor and the right-hand sides are read back once;
    # every owner receives its bytes from host memory through its own
    # context, and every owner's result comes back to host memory before it
    # is scattered. No device-to-device copy and no root gather kernel is
    # involved: the device-to-device form diverged on two MI300X for every
    # factor above 1 MiB (n >= 513) in the columns owned by device 1, and
    # passed when the owner's copies were read back before the solve
    # (bench/results/multi_gpu/2026-09-14/cholesky-mi300x-diag/). The owner's
    # kernel read the previous contents of the target memory after the copy
    # and a drain of both contexts, a platform behavior of those MI300X
    # (bench/results/multi_gpu/2026-09-15/peer-copy-mi300x/; repro:
    # training/checks/peer_copy_check.mojo PEERSOLVE l_first at n=513).
    var host_l = ctx.enqueue_create_host_buffer[DType.float32](n * n)
    var host_b = ctx.enqueue_create_host_buffer[DType.float32](n * nrhs)
    var lv = l.create_sub_buffer[DType.float32](0, n * n)
    var bv = b.create_sub_buffer[DType.float32](0, n * nrhs)
    ctx.enqueue_copy(dst_ptr=host_l.unsafe_ptr(), src_buf=lv)
    ctx.enqueue_copy(dst_ptr=host_b.unsafe_ptr(), src_buf=bv)
    ctx.synchronize()
    var shards = List[CholSolveShard]()
    for rank in range(active):
        var first = nrhs * rank // active
        var width = nrhs * (rank + 1) // active - first
        var source = first
        comptime if is_defined["MOJOLEARN_CHOLESKY_PARALLEL_SABOTAGE"]():
            if rank > 0:
                source = first - 1
        var device = DeviceContext(device_id=rank)
        var packed = device.enqueue_create_host_buffer[DType.float32](n * width)
        device.synchronize()
        for i in range(n):
            for c in range(width):
                packed.unsafe_ptr()[i * width + c] = host_b.unsafe_ptr()[i * nrhs + source + c]
        var sb = device.enqueue_create_buffer[DType.float32](n * width)
        var sl = device.enqueue_create_buffer[DType.float32](n * n)
        device.enqueue_copy(dst_buf=sb, src_ptr=packed.unsafe_ptr())
        device.enqueue_copy(dst_buf=sl, src_ptr=host_l.unsafe_ptr())
        device.synchronize()
        _ = packed^
        shards.append(CholSolveShard(device^, sl^, sb^, first, width))
    var quiet = IdentityTrace.disabled()
    for stage in range(2):
        for rank in range(active):
            ref s = shards[rank]
            if stage == 0:
                trsm_lower(s.ctx, s.l, s.b, n, s.width, quiet, "chol.solve.forward", tpb)
            else:
                trsm_upper(s.ctx, s.l, s.b, n, s.width, quiet, "chol.solve.back", tpb)
        for rank in range(active):
            ref s = shards[rank]
            var result = s.ctx.enqueue_create_host_buffer[DType.float32](n * s.width)
            s.ctx.enqueue_copy(dst_ptr=result.unsafe_ptr(), src_buf=s.b)
            s.ctx.synchronize()
            for i in range(n):
                for c in range(s.width):
                    host_b.unsafe_ptr()[i * nrhs + s.first + c] = result.unsafe_ptr()[i * s.width + c]
            _ = result^
        ctx.enqueue_copy(dst_buf=bv, src_ptr=host_b.unsafe_ptr())
        ctx.synchronize()
        if stage == 0:
            trace.record_device(ctx, "chol.solve.forward", b, n * nrhs)
        else:
            trace.record_device(ctx, "chol.solve.back", b, n * nrhs)
    _ = shards^
    _ = lv^
    _ = bv^
    _ = host_l^
    _ = host_b^
    ctx.synchronize()
