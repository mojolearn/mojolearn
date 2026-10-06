# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`SmoBlockSolve`: the working-set QP by SMO, solved by the whole grid.

Reference: `cuml/cpp/src/svm/smoblocksolve.cuh` (cuML v26.08.00), the kernel
body (`:154-271`), with the same branches. The math is documented in
the reference header and is not repeated here; what follows is what changed.

    typedef cub::BlockReduce<Pair, WSIZE>    -> pinned_block_argmin/argmax
    typedef cub::BlockReduce<math_t, WSIZE>  -> pinned_block_argmax (value only
                                                used; DEVIATION 635)
    __shared__ f_u, u, l, tmp_u, tmp_l, diff, diff_end  -> threadgroup slots
    __shared__ Kd[WSIZE]                     -> threadgroup slab

THE THREE PINS (svm/README.md, identity content section 3):

# =========================================================================
# DEVIATION 633: the two arg-reductions tie-break on the TRAINING INDEX
# (`ws_idx[tid]`, smaller wins). Theirs compares `KVPair::val` only
# (`kselection.cuh:66-79`, "@todo ... consider the key when values are the
# same?"), so the winner among equal f is whatever CUB's fold keeps. Equal
# f is ordinary (duplicate rows; every alpha at 0 on the first iteration
# gives f = -y, two values over the whole set), so this is reached on every
# fit's first inner iteration. Gated by the duplicated-rows fixture against
# the host oracle, which selects the same way in serial.
# =========================================================================

# =========================================================================
# DEVIATION 635 (IDENTITY_PATHS row 39): `f_max` is the SAME key-tied
# argmax (value of the smallest training index among the maximal f), not
# a pure `max` fold. Theirs is `cub::BlockReduce<math_t>::Reduce(f_tmp,
# cuda::maximum{})`, whose survivor among EQUAL values is the fold
# topology's; the only equal values a float max can tell apart are `+0.0`
# and `-0.0`, and `f` can hold both at once (a sample exactly on the
# margin gives +0.0; a negative subnormal flushed at the f seam gives
# -0.0, row 10). Before this deviation ours was a strict-`>` halving tree
# with no key, whose survivor on that tie is a function of tree POSITION
# (the oracle's serial scan keeps the FIRST index, the tree does not), so
# `diff = f_max - f_u` could be `-0.0` on the device and `+0.0` on the
# oracle with f_u = +0.0: one recorded bit (`svm.iterNNN.diff`), decided by
# position. Now every reduction of the block solve ties on the training
# index and the oracle scans with the same rule. Bits move only for a
# working set holding both zeros as its maximal lower-set f; measured by
# `svc_check.mojo::check_block_solve_signed_zero_tie` (order A: the fixed
# spelling gives diff = -0.0/0x80000000 on device and oracle alike; the
# old spelling gives +0.0 on the device, SAB_FMAX_NOKEY).
# =========================================================================

The contraction `f += q * (Kui - Kli)` is `identical_mul_add(q, ftz(Kui -
Kli), f)` under IDENTICAL (their CUDA build contracts it to an fma; Metal
through MAX contracts too; the pin is for the backend that does not); the
quotient `(f_u - f)^2 / eta` is `ftz(ftz(d * d) / eta)`; `eta` is
`ftz(ftz(Kd_t + Kd_u) - ftz(2 * Kui))` floored at `ETA_EPS` by the
compare `if eta < ETA_EPS: eta = ETA_EPS` (theirs is `max(eta, ETA_EPS)`;
row 39: a compare, not a hardware `max`, so a `-0.0` eta gives `ETA_EPS`
on every vendor; a NaN eta needs a NaN kernel cell, which only float
overflow of a legal input can make, and the NaN it leaves in `alpha`/`f`
raises before any record, DEVIATION 637). Both helpers compile away under FAST, so
the FAST kernel is the plain expression and ONE body serves both modes
(the association is the same either way).

THE MIN-SELECTS of the alpha update (`tmp_l if tmp_l < q_l else q_l`,
`tmp_u if tmp_u < tmp_l else tmp_l`) are compare-and-select, not hardware
`min`, and never see a `-0.0` anyway: `alpha` starts at +0.0 and every
update `a +- q*y` with `0 <= q <= min(tmp_u, tmp_l)` stays in `[+0.0, C]`
(a result of exactly zero is +0.0 under round-to-nearest; a positive
subnormal flushes to +0.0), so `a`, `C - a`, and `q_l = (f - f_u)/eta`
with `f_u < f` are all `>= +0.0` with the sign bit clear (row 39).
THE GRID (cpu-gpu-cleanup c-svm, 2026-10-02). Theirs is ONE block of
`n_ws` threads (`SmoBlockSolve<<<1, n_ws>>>`), and so was ours, at six
comptime widths plus an elements-per-thread variant. Now the working set is
spread over `ceil(n_ws / TPB)` blocks of `TPB` threads, one element per
thread, and the grid solves it together in one persistent launch:

  * each of the two selections of an inner iteration (the fused f_u argmin
    / f_max argmax, then the l argmax) is a block halving tree, then a
    GRID EXCHANGE: thread 0 of every block publishes its block's winner to
    a global slot (double-buffered by epoch parity) and RELEASES its flag
    to the epoch; threads `t < n_blocks` of every block ACQUIRE flag `t`,
    load block t's partial into threadgroup memory, and every block folds
    the partials in the same fixed halving tree over block index. Every
    block therefore holds the same winner, and the control flow (the
    `diff < diff_end` and `q == 0` exits) is the same in every block;
  * the winner carries its own payload (`tmp_u` for u; `min(tmp_l, q_l)`
    for l, which each candidate forms from its own registers and Kd[u]),
    so the alpha update needs no third exchange;
  * `Kd[u]` and `Kd[l]` are read from the kernel tile's diagonal in global
    memory (the same cells the old threadgroup slab copied).

The selections are over the total order (value, then the smaller training
index), so the winner is the same element whatever the block width and the
fold shape: the bits are the one-block kernel's and the host oracle's
(`svm/host/smo_oracle.mojo::_block_solve`) unchanged. The flags are reset
by `smo_grid_reset_kernel` before every launch. The grid is at most
`SMO_GRID_WS_MAX / TPB` blocks of at least 64 threads, which every column
keeps resident at once (the spin waits need that).
"""

from std.atomic import Atomic, Ordering
from experiments.classical_identical_ideas.linear_controls import C21_EXTREMA
from std.gpu import block_idx, grid_dim, thread_idx
from std.memory import stack_allocation
from std.math import inf, max
from std.sys.compile import is_defined
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.numerics import ftz, identical_mul_add
from svm.checks.pinned_argreduce import _arg_better


#: SABOTAGES of the `f_max` fold (row 39; svc_check "signed-zero tie"):
#: NOKEY = the pre-DEVIATION-635 strict-`>` tree (position decides a +0/-0
#: tie); HWMAX = halving tree through the hardware `max(mine, other)`;
#: HWMAX_SWAP = `max(other, mine)`. The README records which of these is
#: Apple-inert and why that is exactly the hazard.
comptime SAB_FMAX_NOKEY = is_defined["MOJOLEARN_SVM_SABOTAGE_FMAX_NOKEY"]()
comptime SAB_FMAX_HWMAX = is_defined["MOJOLEARN_SVM_SABOTAGE_FMAX_HWMAX"]()
comptime SAB_FMAX_HWMAX_SWAP = is_defined["MOJOLEARN_SVM_SABOTAGE_FMAX_HWMAX_SWAP"]()
from svm.impl.smo_sets import in_lower, in_upper


#: `constexpr const int SMO_WS_SIZE = 1024` (`smosolver.cuh:116`).
comptime SMO_WS_SIZE = 1024

#: `constexpr math_t ETA_EPS = 1.0e-12` (`smoblocksolve.cuh:168`).
comptime ETA_EPS = Float32(1.0e-12)

#: The largest working set the grid solve takes (the FAST Apple 2048 set).
comptime SMO_GRID_WS_MAX = 2048

#: The narrowest block: the partial tree over `SMO_GRID_WS_MAX / TPB`
#: blocks must fit in one block's threads.
comptime SMO_GRID_TPB_MIN = 64

#: The default block width of the grid solve.
comptime SMO_GRID_TPB = 256

#: Partial slots per block per parity: 4 floats and 4 ints.
comptime SMO_GRID_SLOT = 4

#: The most blocks any width launches (`SMO_GRID_WS_MAX / SMO_GRID_TPB_MIN`).
comptime SMO_GRID_MAX_BLOCKS = SMO_GRID_WS_MAX // SMO_GRID_TPB_MIN

comptime _KEY_PAD = Int32(2147483647)


def smo_grid_reset_kernel(flags: MutPointer[Int32, MutAnyOrigin]):
    """Zero the `SMO_GRID_MAX_BLOCKS` exchange flags before a solve (one
    thread per flag)."""
    var t = Int(thread_idx.x)
    if t < SMO_GRID_MAX_BLOCKS:
        flags.unsafe_store(t, Int32(0))


@always_inline
def _fmax_better(ov: Float32, ok: Int32, mv: Float32, mk: Int32) -> Bool:
    """The f_max fold's compare: DEVIATION 635's key-tied argmax, or the
    NOKEY sabotage (value only, position decides a tie)."""
    comptime if SAB_FMAX_NOKEY:
        return ov > mv
    else:
        return _arg_better[True](ov, ok, mv, mk)


@always_inline
def _tree_dual[
    o1: MutOrigin, o2: MutOrigin, o3: MutOrigin, o4: MutOrigin,
    o5: MutOrigin, o6: MutOrigin, //, N: Int,
](
    nv: Float32,
    nk: Int32,
    npos: Int32,
    npay: Float32,
    xv: Float32,
    xk: Int32,
    s_nv: MutPointer[Float32, o1, address_space = AddressSpace.SHARED],
    s_nk: MutPointer[Int32, o2, address_space = AddressSpace.SHARED],
    s_np: MutPointer[Int32, o3, address_space = AddressSpace.SHARED],
    s_npay: MutPointer[Float32, o4, address_space = AddressSpace.SHARED],
    s_xv: MutPointer[Float32, o5, address_space = AddressSpace.SHARED],
    s_xk: MutPointer[Int32, o6, address_space = AddressSpace.SHARED],
) -> Tuple[Float32, Int32, Int32, Float32, Float32, Int32]:
    """One halving tree over the first `N` threads' entries: the (value,
    key) argmin with its position and payload, and the (value, key) argmax.
    Every thread of the block calls it (`N` a power of two, `N <=` the
    block width); the result is returned to every thread and the trailing
    barrier protects the slabs."""
    var tid = Int(thread_idx.x)
    if tid < N:
        s_nv[tid] = nv
        s_nk[tid] = nk
        s_np[tid] = npos
        s_npay[tid] = npay
        s_xv[tid] = xv
        s_xk[tid] = xk
    barrier()
    # C21 four-way tuple tree fuses two extrema and halves barrier rounds.
    # Value/key total orders are unchanged, including signed-zero ties.
    # NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
    comptime if C21_EXTREMA:
        var live = N
        while live > 1:
            var arity = 4 if live >= 4 else 2
            var stride = live // arity
            if tid < stride:
                for child in range(1, arity):
                    var other = tid + child * stride
                    var ov = s_nv[other]
                    var ok = s_nk[other]
                    if _arg_better[False](ov, ok, s_nv[tid], s_nk[tid]):
                        s_nv[tid] = ov
                        s_nk[tid] = ok
                        s_np[tid] = s_np[other]
                        s_npay[tid] = s_npay[other]
                    if _fmax_better(s_xv[other], s_xk[other], s_xv[tid], s_xk[tid]):
                        s_xv[tid] = s_xv[other]
                        s_xk[tid] = s_xk[other]
            barrier()
            live = stride
    else:
        var step = N // 2
        while step > 0:
            if tid < step:
                var ov = s_nv[tid + step]
                var ok = s_nk[tid + step]
                if _arg_better[False](ov, ok, s_nv[tid], s_nk[tid]):
                    s_nv[tid] = ov
                    s_nk[tid] = ok
                    s_np[tid] = s_np[tid + step]
                    s_npay[tid] = s_npay[tid + step]
                var mv = s_xv[tid + step]
                var mk = s_xk[tid + step]
                comptime if SAB_FMAX_HWMAX:
                    s_xv[tid] = max(s_xv[tid], mv)
                elif SAB_FMAX_HWMAX_SWAP:
                    s_xv[tid] = max(mv, s_xv[tid])
                else:
                    if _fmax_better(mv, mk, s_xv[tid], s_xk[tid]):
                        s_xv[tid] = mv
                        s_xk[tid] = mk
            barrier()
            step //= 2
    var r = (s_nv[0], s_nk[0], s_np[0], s_npay[0], s_xv[0], s_xk[0])
    barrier()
    return r


@always_inline
def _tree_max[
    o1: MutOrigin, o2: MutOrigin, o3: MutOrigin, o4: MutOrigin, //, N: Int,
](
    v: Float32,
    k: Int32,
    pos: Int32,
    pay: Float32,
    s_v: MutPointer[Float32, o1, address_space = AddressSpace.SHARED],
    s_k: MutPointer[Int32, o2, address_space = AddressSpace.SHARED],
    s_p: MutPointer[Int32, o3, address_space = AddressSpace.SHARED],
    s_pay: MutPointer[Float32, o4, address_space = AddressSpace.SHARED],
) -> Tuple[Float32, Int32, Int32, Float32]:
    """The (value, key) argmax with its position and payload, over the
    first `N` threads' entries; the contract is `_tree_dual`'s."""
    var tid = Int(thread_idx.x)
    if tid < N:
        s_v[tid] = v
        s_k[tid] = k
        s_p[tid] = pos
        s_pay[tid] = pay
    barrier()
    var step = N // 2
    while step > 0:
        if tid < step:
            var ov = s_v[tid + step]
            var ok = s_k[tid + step]
            if _arg_better[True](ov, ok, s_v[tid], s_k[tid]):
                s_v[tid] = ov
                s_k[tid] = ok
                s_p[tid] = s_p[tid + step]
                s_pay[tid] = s_pay[tid + step]
        barrier()
        step //= 2
    var r = (s_v[0], s_k[0], s_p[0], s_pay[0])
    barrier()
    return r


@always_inline
def _publish(
    xf: MutPointer[Float32, MutAnyOrigin],
    xi: MutPointer[Int32, MutAnyOrigin],
    flags: MutPointer[Int32, MutAnyOrigin],
    epoch: Int32,
    blk: Int,
    f0: Float32,
    f1: Float32,
    f2: Float32,
    i0: Int32,
    i1: Int32,
    i2: Int32,
):
    """Thread 0 of block `blk`: write the block's partial to the slot of
    this epoch's parity, then RELEASE the flag to `epoch`. The same thread
    writes the payload and releases, so the release orders the payload."""
    var base = ((Int(epoch) & 1) * SMO_GRID_MAX_BLOCKS + blk) * SMO_GRID_SLOT
    xf.unsafe_store(base, f0)
    xf.unsafe_store(base + 1, f1)
    xf.unsafe_store(base + 2, f2)
    xi.unsafe_store(base, i0)
    xi.unsafe_store(base + 1, i1)
    xi.unsafe_store(base + 2, i2)
    Atomic.store[ordering = Ordering.RELEASE](flags.unsafe_offset(blk), epoch)


@always_inline
def _await(flags: MutPointer[Int32, MutAnyOrigin], epoch: Int32, blk: Int):
    """Spin until block `blk` has published `epoch` (flags only grow, and no
    block runs more than one epoch ahead of the slowest). The ACQUIRE load's
    value is consumed by the compare, so it emits on every column
    (core/device_mutex.mojo)."""
    while Atomic.load[ordering = Ordering.ACQUIRE](flags.unsafe_offset(blk)) < epoch:
        pass


@always_inline
def _slot(epoch: Int32, blk: Int) -> Int:
    return ((Int(epoch) & 1) * SMO_GRID_MAX_BLOCKS + blk) * SMO_GRID_SLOT


def smo_grid_solve_kernel[
    TPB: Int
](
    y_array: MutPointer[Float32, MutAnyOrigin],
    n_train_in: Int32,
    alpha: MutPointer[Float32, MutAnyOrigin],
    n_ws_in: Int32,
    delta_alpha: MutPointer[Float32, MutAnyOrigin],
    f_array: MutPointer[Float32, MutAnyOrigin],
    kernel: MutPointer[Float32, MutAnyOrigin],
    ws_idx: MutPointer[Int32, MutAnyOrigin],
    C_vec: MutPointer[Float32, MutAnyOrigin],
    eps: Float32,
    return_buff: MutPointer[Float32, MutAnyOrigin],
    max_iter_in: Int32,
    xf: MutPointer[Float32, MutAnyOrigin],
    xi: MutPointer[Int32, MutAnyOrigin],
    flags: MutPointer[Int32, MutAnyOrigin],
):
    """`SmoBlockSolve<math_t, WSIZE>(...)` for `C_SVC` and `EPSILON_SVR`
    over the grid: launch `ceil(n_ws / TPB)` blocks of `TPB` threads, one
    working-set element per thread, after `smo_grid_reset_kernel`. Threads
    past `n_ws` carry the identity of every selection and do nothing else.
    `xf`, `xi` hold `2 * SMO_GRID_MAX_BLOCKS * SMO_GRID_SLOT` cells, `flags`
    `SMO_GRID_MAX_BLOCKS`."""
    comptime GP = SMO_GRID_WS_MAX // TPB
    comptime assert TPB >= SMO_GRID_TPB_MIN and GP <= TPB, "smo_grid_solve_kernel: TPB out of range"
    var n_ws = Int(n_ws_in)
    var max_iter = Int(max_iter_in)
    var tid = Int(thread_idx.x)
    var blk = Int(block_idx.x)
    var n_blocks = Int(grid_dim.x)
    var p = blk * TPB + tid
    var active = p < n_ws

    # block tree slabs
    var s_nv = stack_allocation[TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var s_nk = stack_allocation[TPB, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var s_np = stack_allocation[TPB, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var s_npay = stack_allocation[TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var s_xv = stack_allocation[TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var s_xk = stack_allocation[TPB, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    # partial tree slabs (one entry per block of the grid)
    var g_nv = stack_allocation[GP, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var g_nk = stack_allocation[GP, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var g_np = stack_allocation[GP, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var g_npay = stack_allocation[GP, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var g_xv = stack_allocation[GP, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var g_xk = stack_allocation[GP, Scalar[DType.int32], address_space = AddressSpace.SHARED]()

    var idx = 0
    var y = Float32(0.0)
    var f = Float32(0.0)
    var a = Float32(0.0)
    var C = Float32(0.0)
    var kd = Float32(0.0)
    # Padding threads carry the largest key, so an active thread always
    # wins a tie against them and `u`, `l` are real elements.
    var key = _KEY_PAD
    if active:
        idx = Int(ws_idx.unsafe_load(p))
        # store values in registers
        y = y_array.unsafe_load(idx)
        f = f_array.unsafe_load(idx)
        a = alpha.unsafe_load(idx)
        C = C_vec.unsafe_load(idx)
        kd = kernel.unsafe_load(p + p * n_ws)
        key = Int32(idx)
    var a_save = a

    var n_iter = 0
    var diff_end = Float32(0.0)
    var pos_inf = inf[DType.float32]()
    var neg_inf = -inf[DType.float32]()
    var epoch = Int32(0)

    while n_iter < max_iter:
        # mask values outside of X_upper
        var f_tmp = pos_inf
        if active and in_upper(a, y, C):
            f_tmp = f
        # the X_lower mask for f_max (nothing between the two selections
        # writes a, y, C or f)
        var f_lo = neg_inf
        if active and in_lower(a, y, C):
            f_lo = f
        # `tmp_u` of the alpha update, carried by the argmin's winner
        var pay_u = C - a if y > Float32(0.0) else a
        var r1 = _tree_dual[TPB](
            f_tmp, key, Int32(p), pay_u, f_lo, key,
            s_nv, s_nk, s_np, s_npay, s_xv, s_xk,
        )
        if n_blocks > 1:
            epoch += 1
            if tid == 0:
                _publish(xf, xi, flags, epoch, blk, r1[0], r1[3], r1[4], r1[1], r1[2], r1[5])
            var gnv = pos_inf
            var gnk = _KEY_PAD
            var gnp = Int32(0)
            var gnpay = Float32(0.0)
            var gxv = neg_inf
            var gxk = _KEY_PAD
            if tid < n_blocks:
                _await(flags, epoch, tid)
                var s = _slot(epoch, tid)
                gnv = xf.unsafe_load(s)
                gnpay = xf.unsafe_load(s + 1)
                gxv = xf.unsafe_load(s + 2)
                gnk = xi.unsafe_load(s)
                gnp = xi.unsafe_load(s + 1)
                gxk = xi.unsafe_load(s + 2)
            r1 = _tree_dual[GP](
                gnv, gnk, gnp, gnpay, gxv, gxk,
                g_nv, g_nk, g_np, g_npay, g_xv, g_xk,
            )
        var f_u = r1[0]
        var u = Int(r1[2])
        var tmp_u = r1[3]
        # DEVIATION 635: the key-tied argmax; `f_max` is the winner's own
        # bits (+0.0 or -0.0 as that sample holds it), decided by the key.
        var f_max = r1[4]
        var Kui = Float32(0.0)
        if active:
            Kui = kernel.unsafe_load(u * n_ws + p)
        var kd_u = kernel.unsafe_load(u + u * n_ws)

        # f_max - f_u is used to check stopping condition.
        var diff = ftz(f_max - f_u)
        if n_iter == 0:
            if blk == 0 and tid == 0:
                return_buff.unsafe_store(0, diff)
            var d10 = ftz(Float32(0.1) * diff)
            diff_end = eps if eps > d10 else d10
        if diff < diff_end:
            break

        if active and f_u < f and in_lower(a, y, C):
            var eta_ui = ftz(ftz(kd + kd_u) - ftz(Float32(2.0) * Kui))
            # row 39: a compare, not `max`; -0.0 < ETA_EPS is TRUE everywhere
            if eta_ui < ETA_EPS:
                eta_ui = ETA_EPS
            var d = ftz(f_u - f)
            f_tmp = ftz(ftz(d * d) / eta_ui)
        else:
            f_tmp = neg_inf
        # `min(tmp_l, q_l)` as this element would form it if it were l
        # (note: Kui == Kul for this thread)
        var tmp_l = a if y > Float32(0.0) else C - a
        var eta_ul = ftz(ftz(kd_u + kd) - ftz(Float32(2.0) * Kui))
        if eta_ul < ETA_EPS:
            eta_ul = ETA_EPS
        var q_l = ftz(ftz(f - f_u) / eta_ul)
        var pay_l = tmp_l if tmp_l < q_l else q_l
        var r2 = _tree_max[TPB](f_tmp, key, Int32(p), pay_l, s_nv, s_nk, s_np, s_npay)
        if n_blocks > 1:
            epoch += 1
            if tid == 0:
                _publish(
                    xf, xi, flags, epoch, blk, r2[0], r2[3], Float32(0.0),
                    r2[1], r2[2], Int32(0),
                )
            var gv = neg_inf
            var gk = _KEY_PAD
            var gp = Int32(0)
            var gpay = Float32(0.0)
            if tid < n_blocks:
                _await(flags, epoch, tid)
                var s = _slot(epoch, tid)
                gv = xf.unsafe_load(s)
                gpay = xf.unsafe_load(s + 1)
                gk = xi.unsafe_load(s)
                gp = xi.unsafe_load(s + 1)
            r2 = _tree_max[GP](gv, gk, gp, gpay, g_nv, g_nk, g_np, g_npay)
        var l = Int(r2[2])
        var tmp_l2 = r2[3]
        var Kli = Float32(0.0)
        if active:
            Kli = kernel.unsafe_load(l * n_ws + p)

        # Update alpha (the clipping argument is in their comment block)
        var q = tmp_u if tmp_u < tmp_l2 else tmp_l2
        if p == u:
            a = ftz(identical_mul_add(q, y, a))  # the default build's fused op (lane/pinned-mul-contract-free)
        if p == l:
            a = ftz(identical_mul_add(-q, y, a))  # the default build's fused op (lane/pinned-mul-contract-free)
        f = ftz(identical_mul_add(q, ftz(Kui - Kli), f))
        if q == Float32(0.0):
            # Probably fp underflow
            break
        n_iter += 1

    # save results to global memory before exit
    if active:
        alpha.unsafe_store(idx, a)
        # it is actually y * \Delta \alpha
        delta_alpha.unsafe_store(p, ftz(ftz(a - a_save) * y))
    # f is recalculated in f_update, therefore we do not need to save that
    if blk == 0 and tid == 0:
        return_buff.unsafe_store(1, Float32(n_iter))
