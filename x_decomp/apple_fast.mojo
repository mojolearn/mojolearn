# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The decomp lane's sparse-coding cells rescheduled for FAST on Apple
(lane/apple-fast-decomp-sparse, 2026-10-02): one threadgroup per row.

`lasso_rows_kernel` and `omp_rows_kernel` (x_decomp/device.mojo) run one
THREAD per sample. The Lasso thread sweeps its k coordinates serially and
every coordinate walks the k-wide Gram column against the row's h strip in
global memory (consecutive threads 4k bytes apart, a read-modify-write per
element). The OMP thread keeps its Cholesky, active set and solves in a
per-row scratch of k * k + 3k floats, so the launch allocates n times that
(the sparse-coder lane: 100,000 rows x 64 atoms is 1.7 GB of device scratch
for a 4-atom answer). Here a row is one threadgroup:

  * `lasso_block_kernel`: G, w, q and the projection h = G w in threadgroup
    memory. Every thread computes the same coordinate update from the same
    three values (one scalar, no broadcast), thread l applies it to h[l],
    and thread j stores w[j]. The coordinate order, the stop rule and the
    update spelling are `lasso_row`'s (x_decomp/cells.mojo).
  * `omp_block_kernel`: thread j computes atom j's residual correlation,
    the argmax (ties to the LOWER atom) is a threadgroup fold on (value,
    index), and the Cholesky extension and the two triangular solves on the
    active set (at most nnz rows) run on thread 0 in threadgroup memory.
    Same selection, same refit, no device scratch.

FAST promises quality, not bits: the spellings are the cells' own, and a
fold on (value, index) picks the atom the serial scan picks. IDENTICAL
never compiles these: x_decomp/device.mojo gates each launch on
GLOBAL_NUMERIC_MODE and TARGET_COLUMN, and each has a host-read env switch
that defaults OFF (`lasso_block_on`, `omp_block_on`, `small_eigh_j2_on`).
"""
from std.gpu import block_idx, thread_idx
from std.memory import stack_allocation
from std.os import getenv
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.numerics import ftz, identical_mul_add
from x_decomp.cells import F32Ptr, div0, mul, sqrt0, sub

comptime LB_TPB = 32
"""Threads of a Lasso block (one simdgroup); k atoms at most LB_TPB."""
comptime OMP_TPB = 128
"""Threads of an OMP block; k atoms at most OMP_TPB."""
comptime OMP_NNZ_MAX = 32
"""Active atoms at most: the threadgroup Cholesky is OMP_NNZ_MAX square."""
comptime SMALL_EIGH_MAX = 64
"""`small_eigh_j2_on`: an eigh of at most this order takes the one-launch
cyclic kernel (x_decomp/jacobi2.mojo) instead of the round-robin grid."""


def lasso_block_on(k: Int) -> Bool:
    """MOJOLEARN_DECOMP_FAST_LASSO_BLOCK=1 (host env, read at dispatch) and
    a dictionary the block holds."""
    return k >= 1 and k <= LB_TPB and String(getenv("MOJOLEARN_DECOMP_FAST_LASSO_BLOCK")) == "1"


def omp_block_on(k: Int, nnz: Int) -> Bool:
    """MOJOLEARN_DECOMP_FAST_OMP_BLOCK=1 and a dictionary and active set the
    block holds."""
    if k < 1 or k > OMP_TPB or nnz < 1 or nnz > OMP_NNZ_MAX:
        return False
    return String(getenv("MOJOLEARN_DECOMP_FAST_OMP_BLOCK")) == "1"


def small_eigh_j2_on(n: Int) -> Bool:
    """MOJOLEARN_DECOMP_FAST_SMALL_EIGH_J2=1 and n at most SMALL_EIGH_MAX.
    The round-robin eigh (x_decomp/device.mojo `_eigh_par`) launches two
    kernels per round, n - 1 rounds per sweep, and reads the off-diagonal
    norm back before every sweep: for FastICA's n_components x n_components
    decorrelation (8 x 8, every iteration) that is ~130 launches and ~16
    host syncs for 28 rotations; `_eigh2` is one launch and one readback."""
    return n >= 2 and n <= SMALL_EIGH_MAX and String(getenv("MOJOLEARN_DECOMP_FAST_SMALL_EIGH_J2")) == "1"


def lasso_block_kernel(
    g: F32Ptr, q: F32Ptr, w: F32Ptr, its: F32Ptr, n: Int32, k_in: Int32, alpha: Float32,
    max_iter: Int32, tol: Float32, positive: Int32,
):
    """Row block_idx.x of `lasso_row` (x_decomp/cells.mojo), the block
    sharing each coordinate's h update. Every thread runs the same
    coordinate loop on the same shared values, so the control flow (the
    skipped coordinates, the stop) is uniform across the block."""
    var i = Int(block_idx.x)
    if i >= Int(n):
        return
    var k = Int(k_in)
    var tid = Int(thread_idx.x)
    var base = i * k
    var sg = stack_allocation[LB_TPB * LB_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var sw = stack_allocation[LB_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var sh = stack_allocation[LB_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var sq = stack_allocation[LB_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    for t in range(tid, k * k, LB_TPB):
        sg[t] = g.unsafe_load(t)
    if tid < k:
        sw[tid] = w.unsafe_load(base + tid)
        sq[tid] = q.unsafe_load(base + tid)
    barrier()
    # h = G w, row tid: the sum over l ascending, as the cell forms it
    if tid < k:
        var acc = Float32(0)
        for l in range(k):
            acc = ftz(identical_mul_add(ftz(sg[tid * k + l]), ftz(sw[l]), acc))
        sh[tid] = acc
    barrier()
    var pos = positive != 0
    var it = 0
    for _ in range(Int(max_iter)):
        it += 1
        var w_max = Float32(0)
        var d_w_max = Float32(0)
        for j in range(k):
            var gjj = ftz(sg[j * k + j])
            if gjj == Float32(0):
                continue
            var w_j = ftz(sw[j])
            if w_j != Float32(0) and tid < k:
                sh[tid] = ftz(identical_mul_add(-w_j, ftz(sg[tid * k + j]), ftz(sh[tid])))
            barrier()
            var tmp = sub(sq[j], sh[j])
            var nw = Float32(0)
            if pos:
                if tmp > Float32(0):
                    var m = sub(tmp, alpha)
                    nw = div0(m, gjj) if m > Float32(0) else Float32(0)
            else:
                var m = sub(abs(tmp), alpha)
                if m > Float32(0):
                    nw = div0(m if tmp > Float32(0) else -m, gjj)
            barrier()
            if nw != Float32(0) and tid < k:
                sh[tid] = ftz(identical_mul_add(nw, ftz(sg[tid * k + j]), ftz(sh[tid])))
            if tid == j:
                sw[j] = nw
            barrier()
            var dw = abs(sub(nw, w_j))
            if dw > d_w_max:
                d_w_max = dw
            if abs(nw) > w_max:
                w_max = abs(nw)
        if w_max == Float32(0) or d_w_max <= mul(tol, w_max):
            break
    if tid < k:
        w.unsafe_store(base + tid, sw[tid])
    if tid == 0:
        its.unsafe_store(i, Float32(it))


def omp_block_kernel(g: F32Ptr, q: F32Ptr, w: F32Ptr, na_out: F32Ptr, n: Int32, k_in: Int32, nnz_in: Int32):
    """Row block_idx.x of `omp_row` (x_decomp/cells.mojo): the correlations
    a thread per atom, the pick a fold on (|correlation|, atom) with the
    LOWER atom on a tie, the Cholesky row and the solves on thread 0, all
    scratch in threadgroup memory (L is OMP_NNZ_MAX x OMP_NNZ_MAX)."""
    var i = Int(block_idx.x)
    if i >= Int(n):
        return
    var k = Int(k_in)
    var nnz = Int(nnz_in)
    var tid = Int(thread_idx.x)
    var base = i * k
    var sl = stack_allocation[OMP_NNZ_MAX * OMP_NNZ_MAX, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var sact = stack_allocation[OMP_NNZ_MAX, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var sgam = stack_allocation[OMP_NNZ_MAX, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var stmp = stack_allocation[OMP_NNZ_MAX, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var sv = stack_allocation[OMP_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var si = stack_allocation[OMP_TPB, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var sflag = stack_allocation[1, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    for t in range(tid, k, OMP_TPB):
        w.unsafe_store(base + t, Float32(0))
    var na = 0
    var eps = Float32(1.1920928955078125e-07)
    while na < nnz:
        # residual correlations alpha = Xy - G[:, S] gamma, atom tid
        var best_v = Float32(-1)
        var best_j = k
        if tid < k:
            var acc = ftz(q.unsafe_load(base + tid))
            for t in range(na):
                var a = Int(sact[t])
                acc = ftz(identical_mul_add(-ftz(g.unsafe_load(tid * k + a)), ftz(sgam[t]), acc))
            best_v = abs(acc)
            best_j = tid
        sv[tid] = best_v
        si[tid] = Int32(best_j)
        barrier()
        # the largest |correlation|, ties to the LOWER atom
        var h = OMP_TPB // 2
        while h > 0:
            if tid < h:
                var v2 = sv[tid + h]
                var i2 = si[tid + h]
                if v2 > sv[tid] or (v2 == sv[tid] and i2 < si[tid]):
                    sv[tid] = v2
                    si[tid] = i2
            barrier()
            h //= 2
        var lam = Int(si[0])
        var best = sv[0]
        var already = False
        for t in range(na):
            if Int(sact[t]) == lam:
                already = True
        if already or mul(best, best) < eps:
            break
        if tid == 0:
            # extend the Cholesky factor by one row
            for t in range(na):
                var a = Int(sact[t])
                var acc = ftz(g.unsafe_load(lam * k + a))
                for u in range(t):
                    acc = ftz(identical_mul_add(-ftz(sl[na * OMP_NNZ_MAX + u]), ftz(sl[t * OMP_NNZ_MAX + u]), acc))
                sl[na * OMP_NNZ_MAX + t] = div0(acc, sl[t * OMP_NNZ_MAX + t])
            var v = Float32(0)
            for t in range(na):
                var x = ftz(sl[na * OMP_NNZ_MAX + t])
                v = ftz(identical_mul_add(x, x, v))
            var lkk = sub(g.unsafe_load(lam * k + lam), v)
            if not (lkk > eps):
                sflag[0] = Float32(1)
            else:
                sflag[0] = Float32(0)
                sl[na * OMP_NNZ_MAX + na] = sqrt0(lkk)
                sact[na] = Int32(lam)
                # gamma = (L L^T)^-1 Xy[S]: forward then back substitution
                var m = na + 1
                for t in range(m):
                    var acc = ftz(q.unsafe_load(base + Int(sact[t])))
                    for u in range(t):
                        acc = ftz(identical_mul_add(-ftz(sl[t * OMP_NNZ_MAX + u]), ftz(stmp[u]), acc))
                    stmp[t] = div0(acc, sl[t * OMP_NNZ_MAX + t])
                for tt in range(m):
                    var t = m - 1 - tt
                    var acc = ftz(stmp[t])
                    for u in range(t + 1, m):
                        acc = ftz(identical_mul_add(-ftz(sl[u * OMP_NNZ_MAX + t]), ftz(sgam[u]), acc))
                    sgam[t] = div0(acc, sl[t * OMP_NNZ_MAX + t])
        barrier()
        if sflag[0] != Float32(0):
            break
        na += 1
    if tid < na:
        w.unsafe_store(base + Int(sact[tid]), sgam[tid])
    if tid == 0:
        na_out.unsafe_store(i, Float32(na))
