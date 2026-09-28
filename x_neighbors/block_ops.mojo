# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""GPU-only threadgroup forms of the lane's SEQUENTIAL items (lane
neighbors-apple2). An item that is one whole solve (`count` "1" in
`gen.py`) runs on ONE GPU thread; here the same solve's independent scans
and updates are spread over one threadgroup, calling the SAME helpers from
`items.mojo`, so every stored value is the item's (the CPU column keeps the
item; `-D MOJOLEARN_XN_SERIAL_SMO` restores it on the GPU, an A/B arm).

ocsvm_smo_block: `ocsvm_smo_item`, libsvm's one-class solve.
  * the initial gradient and the gradient update are per sample, one
    thread per sample, the item's statements;
  * the working-set scans are argmax / argmin reductions. The item's scans
    run samples ascending with `>=` / `<=`, which keeps the LAST index of an
    equal value and never takes a NaN; that is the maximum of the pair
    (value, index) over the eligible non-NaN samples (the minimum of
    (obj, -index) for the second scan), a total order, so any reduction
    tree returns the item's index. The values are then re-read at that
    index, so gmax / gmax2 carry the item's exact bits (a -0.0 against a
    +0.0 included);
  * the two-variable step and calculate_rho run on thread 0 (the item's
    helpers).
Every value passed between threads goes through DEVICE memory or the
shared reduction slots across `_xn_barrier`, which orders device memory on
Apple too (`air.wg.barrier(3, 1)`; x_linear/team.mojo `team_barrier`).
"""
from std.gpu import thread_idx
from std.memory import bitcast, stack_allocation
from std.sys import llvm_intrinsic
from std.sys.info import is_apple_gpu
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from checks.numerics import ftz

from x_neighbors.items import (
    FP, IP, ocsvm_g0, ocsvm_obj, ocsvm_update, ocsvm_g_step, ocsvm_rho,
)

comptime OCSVM_TPB = 256


@always_inline
def _xn_barrier():
    comptime if is_apple_gpu():
        # Match the stdlib intrinsic declaration; retain both memory fences.
        llvm_intrinsic["llvm.air.wg.barrier", NoneType](Int32(3), Int32(1))
    else:
        barrier()


@always_inline
def _add(a: Float32, b: Float32) -> Float32:
    """items.mojo's `_add`."""
    return ftz(ftz(a) + ftz(b))


@always_inline
def _takes_max(v1: Float32, i1: Int32, v2: Float32, i2: Int32) -> Bool:
    """(v2, i2) beats (v1, i1) in the lexicographic maximum; -1 is empty."""
    if i2 < 0:
        return False
    if i1 < 0:
        return True
    return v2 > v1 or (v2 == v1 and i2 > i1)


@always_inline
def _takes_min(v1: Float32, i1: Int32, v2: Float32, i2: Int32) -> Bool:
    """(v2, -i2) beats (v1, -i1) in the lexicographic minimum; -1 is empty."""
    if i2 < 0:
        return False
    if i1 < 0:
        return True
    return v2 < v1 or (v2 == v1 and i2 > i1)


def ocsvm_smo_block(
    q: FP, cv: FP, alpha: FP, g: FP, info: FP, iters: IP,
    n: Int, eps: Float32, max_iter: Int,
):
    """`ocsvm_smo_item` over one threadgroup of OCSVM_TPB threads."""
    var tid = Int(thread_idx.x)
    var neg_inf = bitcast[DType.float32](UInt32(0xFF800000))
    var pos_inf = bitcast[DType.float32](UInt32(0x7F800000))
    var rv = stack_allocation[OCSVM_TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var ri = stack_allocation[OCSVM_TPB, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var rv2 = stack_allocation[OCSVM_TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var ri2 = stack_allocation[OCSVM_TPB, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    # slots: 0 gi, 1 gj, 2 stop, 3 it
    var sl = stack_allocation[4, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var sf = stack_allocation[2, Scalar[DType.float32], address_space=AddressSpace.SHARED]()

    var i = tid
    while i < n:
        g.unsafe_store(i, ocsvm_g0(q, alpha, n, i))
        i += OCSVM_TPB
    if tid == 0:
        sl[3] = Int32(0)
    _xn_barrier()

    while Int(sl[3]) < max_iter:
        # ---- gi: max of (-g_t, t) over alpha_t < C_t ---------------------
        var bv = neg_inf
        var bi = Int32(-1)
        var t = tid
        while t < n:
            if alpha.unsafe_load(t) < cv.unsafe_load(t):
                var ng = -g.unsafe_load(t)
                if ng >= bv:
                    bv = ng
                    bi = Int32(t)
            t += OCSVM_TPB
        rv[tid] = bv
        ri[tid] = bi
        _xn_barrier()
        var w = OCSVM_TPB // 2
        while w > 0:
            if tid < w:
                if _takes_max(rv[tid], ri[tid], rv[tid + w], ri[tid + w]):
                    rv[tid] = rv[tid + w]
                    ri[tid] = ri[tid + w]
            _xn_barrier()
            w //= 2
        var gi = Int(ri[0])
        # ---- gj: min of (obj_j, -j) and gmax2: max of (g_j, j), alpha_j > 0
        var ov = pos_inf
        var oi = Int32(-1)
        var mv = neg_inf
        var mi = Int32(-1)
        if gi >= 0:
            var gmax = -g.unsafe_load(gi)
            var qdi = q.unsafe_load(gi * n + gi)
            var j = tid
            while j < n:
                if alpha.unsafe_load(j) > Float32(0):
                    var gjv = g.unsafe_load(j)
                    if gjv >= mv:
                        mv = gjv
                        mi = Int32(j)
                    if _add(gmax, gjv) > Float32(0):
                        var obj = ocsvm_obj(gmax, gjv, qdi, q.unsafe_load(j * n + j), q.unsafe_load(gi * n + j))
                        if obj <= ov:
                            ov = obj
                            oi = Int32(j)
                j += OCSVM_TPB
        _xn_barrier()
        rv[tid] = ov
        ri[tid] = oi
        rv2[tid] = mv
        ri2[tid] = mi
        _xn_barrier()
        w = OCSVM_TPB // 2
        while w > 0:
            if tid < w:
                if _takes_min(rv[tid], ri[tid], rv[tid + w], ri[tid + w]):
                    rv[tid] = rv[tid + w]
                    ri[tid] = ri[tid + w]
                if _takes_max(rv2[tid], ri2[tid], rv2[tid + w], ri2[tid + w]):
                    rv2[tid] = rv2[tid + w]
                    ri2[tid] = ri2[tid + w]
            _xn_barrier()
            w //= 2
        # ---- the stopping test and the step, thread 0 --------------------
        if tid == 0:
            var gj = Int(ri[0])
            var stop = 1
            if gi >= 0 and gj >= 0:
                var gmax = -g.unsafe_load(gi)
                var gmax2 = neg_inf
                if Int(ri2[0]) >= 0:
                    gmax2 = g.unsafe_load(Int(ri2[0]))
                if not (_add(gmax, gmax2) < eps):
                    stop = 0
            sl[2] = Int32(stop)
            if stop == 0:
                sl[3] = sl[3] + 1
                var d = ocsvm_update(q, cv, alpha, g, n, gi, gj)
                sl[0] = Int32(gi)
                sl[1] = Int32(gj)
                sf[0] = d[0]
                sf[1] = d[1]
        _xn_barrier()
        if Int(sl[2]) != 0:
            break
        var si = Int(sl[0])
        var sj = Int(sl[1])
        var dai = sf[0]
        var daj = sf[1]
        var k = tid
        while k < n:
            ocsvm_g_step(q, g, n, si, sj, dai, daj, k)
            k += OCSVM_TPB
        _xn_barrier()
    _xn_barrier()
    if tid == 0:
        info.unsafe_store(0, ocsvm_rho(g, alpha, cv, n))
        iters.unsafe_store(0, sl[3])
