# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""FAST-tier device folds (lane prep-apple, 2026-09-28). FAST ONLY: the
IDENTICAL binding never compiles a call to these (x_prep/device.mojo gates
them on GLOBAL_NUMERIC_MODE), and the host always runs the units.

The units `col_stats` and `pt_fold` fold a column in row order on ONE
thread. In FAST the bits may change, so the device folds each column with a
threadgroup instead: every thread sums a strided share of the rows, then a
fixed tree combines the shares (a pairwise sum, never less accurate than the
row-order one). The quality rule is checked by bench/x_prep_quality.py
(paired against scikit-learn, 5 seeds x 2 datasets).
"""
from std.gpu import block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from x_prep.common import FP, IP, p, is_nan
from x_prep.prims import add, sub, mul, div
from x_prep.transform import PT_STATE, pt_finish, log1pf
from x_prep.prims import logf

comptime TGR = 256


def col_stats_fast_kernel(f: FP, X: Int32, n: Int32, d: Int32, O: Int32):
    """`col_stats_unit`'s outputs for column block_idx.x: count, mean, var
    (population), min, max, maxabs over the non-NaN entries (zeros when
    empty), each thread over rows tid, tid + TGR, ... then the tree."""
    var c = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var nn = Int(n)
    var dd = Int(d)
    var sh_s = stack_allocation[TGR, Float32, address_space = AddressSpace.SHARED]()
    var sh_c = stack_allocation[TGR, Float32, address_space = AddressSpace.SHARED]()
    var sh_lo = stack_allocation[TGR, Float32, address_space = AddressSpace.SHARED]()
    var sh_hi = stack_allocation[TGR, Float32, address_space = AddressSpace.SHARED]()
    var sh_ma = stack_allocation[TGR, Float32, address_space = AddressSpace.SHARED]()
    var inf = Float32(3.4028235e38)
    var cnt = Float32(0)
    var s = Float32(0)
    var lo = inf
    var hi = -inf
    var ma = Float32(0)
    for i in range(tid, nn, TGR):
        var v = f[Int(X) + i * dd + c]
        if is_nan(v):
            continue
        cnt += 1
        s = add(s, v)
        lo = min(lo, v)
        hi = max(hi, v)
        ma = max(ma, abs(v))
    sh_s[tid] = s
    sh_c[tid] = cnt
    sh_lo[tid] = lo
    sh_hi[tid] = hi
    sh_ma[tid] = ma
    barrier()
    var w = TGR // 2
    while w >= 1:
        if tid < w:
            sh_s[tid] = add(sh_s[tid], sh_s[tid + w])
            sh_c[tid] = sh_c[tid] + sh_c[tid + w]
            sh_lo[tid] = min(sh_lo[tid], sh_lo[tid + w])
            sh_hi[tid] = max(sh_hi[tid], sh_hi[tid + w])
            sh_ma[tid] = max(sh_ma[tid], sh_ma[tid + w])
        barrier()
        w //= 2
    var total = sh_c[0]
    var mean = Float32(0)
    if total > 0:
        mean = div(sh_s[0], total)
    barrier()
    var ss = Float32(0)
    if total > 0:
        for i in range(tid, nn, TGR):
            var v = f[Int(X) + i * dd + c]
            if is_nan(v):
                continue
            var e = sub(v, mean)
            ss = add(ss, mul(e, e))
    sh_s[tid] = ss
    barrier()
    var w2 = TGR // 2
    while w2 >= 1:
        if tid < w2:
            sh_s[tid] = add(sh_s[tid], sh_s[tid + w2])
        barrier()
        w2 //= 2
    if tid == 0:
        var O_ = Int(O)
        if total > 0:
            f[O_ + c] = total
            f[O_ + dd + c] = mean
            f[O_ + 2 * dd + c] = div(sh_s[0], total)
            f[O_ + 3 * dd + c] = sh_lo[0]
            f[O_ + 4 * dd + c] = sh_hi[0]
            f[O_ + 5 * dd + c] = sh_ma[0]
        else:
            for r in range(6):
                f[O_ + r * dd + c] = Float32(0)


def pt_fold_fast_kernel(f: FP, q: IP):
    """`pt_fold_unit` for column block_idx.x (q: the stage's device params):
    the first pass (count, sum T, and at K = 0 sum J) and the squared
    deviations by the tree, then `pt_finish` on thread 0."""
    var c = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var X = p(q, 0)
    var nn = p(q, 1)
    var dd = p(q, 2)
    var method = p(q, 3)
    var T = p(q, 4)
    var K = p(q, 5)
    var S = p(q, 6) + c * PT_STATE
    if f[S + 7] != Float32(0):
        return
    var first = K == 0
    var sh_s = stack_allocation[TGR, Float32, address_space = AddressSpace.SHARED]()
    var sh_c = stack_allocation[TGR, Float32, address_space = AddressSpace.SHARED]()
    var sh_j = stack_allocation[TGR, Float32, address_space = AddressSpace.SHARED]()
    var cnt = Float32(0)
    var sm = Float32(0)
    var sj = Float32(0)
    for i in range(tid, nn, TGR):
        var x = f[X + i * dd + c]
        if is_nan(x):
            continue
        cnt += 1
        sm = add(sm, f[T + i * dd + c])
        if first:
            if method == 1:
                sj = add(sj, logf(x))
            elif x >= Float32(0):
                sj = add(sj, log1pf(x))
            else:
                sj = sub(sj, log1pf(sub(Float32(0), x)))
    sh_s[tid] = sm
    sh_c[tid] = cnt
    sh_j[tid] = sj
    barrier()
    var w = TGR // 2
    while w >= 1:
        if tid < w:
            sh_s[tid] = add(sh_s[tid], sh_s[tid + w])
            sh_c[tid] = sh_c[tid] + sh_c[tid + w]
            sh_j[tid] = add(sh_j[tid], sh_j[tid + w])
        barrier()
        w //= 2
    var total = sh_c[0]
    var mean = Float32(0)
    if total > 0:
        mean = div(sh_s[0], total)
    var sjt = sh_j[0] if first else f[S + 8]
    barrier()
    var ss = Float32(0)
    if total > 0:
        for i in range(tid, nn, TGR):
            var x = f[X + i * dd + c]
            if is_nan(x):
                continue
            var e = sub(f[T + i * dd + c], mean)
            ss = add(ss, mul(e, e))
    sh_s[tid] = ss
    barrier()
    var w2 = TGR // 2
    while w2 >= 1:
        if tid < w2:
            sh_s[tid] = add(sh_s[tid], sh_s[tid + w2])
        barrier()
        w2 //= 2
    if tid == 0:
        if first:
            f[S + 8] = sjt
        pt_finish(c, f, q, Int(total), sjt, sh_s[0])
