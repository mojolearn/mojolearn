# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Orthogonal matching pursuit with one threadgroup per row, FAST on Apple
(lane/apple-fast-decomp-sparse, 2026-10-02; recovered onto main by
lane/apple-fast-rec-decomp, 2026-10-04). Switch: `DECOMP_FAST_OMP_BLOCK`,
default OFF; taken by x_decomp/device.mojo `DevExec.omp_rows`. IDENTICAL
compiles none of this.

Cause: `omp_rows_kernel` runs one THREAD per sample with its Cholesky,
active set and solves in a per-row scratch of k * k + 3k floats, so the
launch allocates n times that (the sparse-coder lane: 100,000 rows x 64
atoms is 1.7 GB of device scratch for a 4-atom answer) and every
correlation scan is k serial dot products on one thread.

Here thread j computes atom j's residual correlation, the argmax (ties to the
LOWER atom, the serial scan's pick) is a threadgroup fold on (value, index),
and the Cholesky extension and the two triangular solves on the active set
(at most nnz rows) run on thread 0 in threadgroup memory. Same selection,
same refit (`omp_row`'s spellings, x_decomp/cells.mojo), no device scratch.
"""
from std.gpu import block_idx, thread_idx
from std.memory import stack_allocation
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, ftz, identical_mul_add
from x_decomp.cells import F32Ptr, div0, mul, sqrt0, sub

#: Recovered candidate (lane/apple-fast-rec-decomp, 2026-10-04), default OFF,
#: FAST + Apple only. Source lane/apple-fast-decomp-sparse@5fb1740cd. What it
#: does: `DevExec.omp_rows` (SparseCoder's omp transform and every kit OMP)
#: launches `omp_block_kernel`, one OMP_TPB-thread block per row, instead of
#: `omp_rows_kernel` and its n x (k * k + 3k) device scratch. Known: board
#: 0831 sparse-coder istella 6,669 ms against scikit-learn's 29 ms; queued
#: once as dsp-omp-istella (lane/apple-fast-batch prebuilt arms) with no
#: recorded result or failure. Fixed in the port: only the OMP kernel is
#: kept (the branch's DECOMP_FAST_LASSO_BLOCK is main's DECOMP_FAST_LASSO_GRP
#: and its DECOMP_FAST_DICT_UPDATE main's DECOMP_FAST_DICT_DEV, both already
#: FAST + Apple defaults), the switch is a comptime define with the FAST +
#: Apple guard, and W's row is written once per cell by its owning thread
#: (the branch zeroed it and then stored gamma from other threads, an
#: ordering Metal's barrier() does not give device memory). Bounds come
#: from the kernel: k <= OMP_TPB (a thread per atom) and nnz <= OMP_NNZ_MAX
#: (the threadgroup Cholesky's side); otherwise the old kernel runs.
comptime DECOMP_FAST_OMP_BLOCK = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and is_defined["MOJOLEARN_DECOMP_FAST_OMP_BLOCK"]()
)
comptime OMP_TPB = 128
"""Threads of an OMP block; k atoms at most OMP_TPB."""
comptime OMP_NNZ_MAX = 32
"""Active atoms at most: the threadgroup Cholesky is OMP_NNZ_MAX square."""


def omp_block_fits(k: Int, nnz: Int) -> Bool:
    """A dictionary and an active set the block holds."""
    return k >= 1 and k <= OMP_TPB and nnz >= 1 and nnz <= OMP_NNZ_MAX


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
                var ns = na + 1
                for t in range(ns):
                    var acc = ftz(q.unsafe_load(base + Int(sact[t])))
                    for u in range(t):
                        acc = ftz(identical_mul_add(-ftz(sl[t * OMP_NNZ_MAX + u]), ftz(stmp[u]), acc))
                    stmp[t] = div0(acc, sl[t * OMP_NNZ_MAX + t])
                for tt in range(ns):
                    var t = ns - 1 - tt
                    var acc = ftz(stmp[t])
                    for u in range(t + 1, ns):
                        acc = ftz(identical_mul_add(-ftz(sl[u * OMP_NNZ_MAX + t]), ftz(sgam[u]), acc))
                    sgam[t] = div0(acc, sl[t * OMP_NNZ_MAX + t])
        barrier()
        if sflag[0] != Float32(0):
            break
        na += 1
    # W's row: thread t owns atoms t, t + OMP_TPB, ... and stores gamma at an
    # active atom, 0 elsewhere (one store per cell, so no ordering between
    # threads' device stores is needed: Metal's barrier() orders threadgroup
    # memory only)
    barrier()
    for t in range(tid, k, OMP_TPB):
        var v = Float32(0)
        for u in range(na):
            if Int(sact[u]) == t:
                v = sgam[u]
        w.unsafe_store(base + t, v)
    if tid == 0:
        na_out.unsafe_store(i, Float32(na))
