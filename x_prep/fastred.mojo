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
from x_prep.fastpt import PT_FOLD_NOX

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
        comptime if PT_FOLD_NOX:
            # lane af-ptimpute, -D MOJOLEARN_PT_FOLD_NOX: after K = 0 fold T alone, as
            # `pt_fold_unit` does (a row is NaN exactly where its T word is, pt_map_unit). The
            # X read below is d words apart per thread, a cache line a word at Istella's d = 220,
            # 220M lines per evaluation for a NaN test (docs/apple-fast/notes/ptimpute.md).
            if not first:
                var tvx = f[T + c * nn + i]
                if is_nan(tvx):
                    continue
                cnt += 1
                sm = add(sm, tvx)
                continue
        var x = f[X + i * dd + c]
        if is_nan(x):
            continue
        cnt += 1
        sm = add(sm, f[T + c * nn + i])
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
            var tv = f[T + c * nn + i]
            if is_nan(tv):
                continue
            var e = sub(f[T + c * nn + i], mean)
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


def class_stats_fast_kernel(f: FP, q: IP):
    """`class_stats_unit` for t = block_idx.x = k*d + c: the class-k rows of
    column c summed by the tree (count, sum, then the squared deviations)."""
    var t = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var X = p(q, 0)
    var nn = p(q, 1)
    var dd = p(q, 2)
    var Y = p(q, 3)
    var k = t // dd
    var c = t % dd
    var sh_s = stack_allocation[TGR, Float32, address_space = AddressSpace.SHARED]()
    var sh_c = stack_allocation[TGR, Float32, address_space = AddressSpace.SHARED]()
    var cnt = Float32(0)
    var s = Float32(0)
    for i in range(tid, nn, TGR):
        if Int(f[Y + i]) != k:
            continue
        s = add(s, f[X + i * dd + c])
        cnt += 1
    sh_s[tid] = s
    sh_c[tid] = cnt
    barrier()
    var w = TGR // 2
    while w >= 1:
        if tid < w:
            sh_s[tid] = add(sh_s[tid], sh_s[tid + w])
            sh_c[tid] = sh_c[tid] + sh_c[tid + w]
        barrier()
        w //= 2
    var total = sh_c[0]
    var sum_ = sh_s[0]
    var mean = Float32(0)
    if total > 0:
        mean = div(sum_, total)
    barrier()
    var ss = Float32(0)
    if total > 0 and p(q, 7) >= 0:
        for i in range(tid, nn, TGR):
            if Int(f[Y + i]) != k:
                continue
            var e = sub(f[X + i * dd + c], mean)
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
        var var_ = Float32(0)
        if total > 0:
            var_ = div(sh_s[0], total)
        if c == 0 and p(q, 5) >= 0:
            f[p(q, 5) + k] = total
        if p(q, 6) >= 0:
            f[p(q, 6) + t] = mean
        if p(q, 7) >= 0:
            f[p(q, 7) + t] = var_
        if p(q, 8) >= 0:
            f[p(q, 8) + t] = sum_


def ii_mean_fast_kernel(f: FP, q: IP):
    """`ii_mean_unit` for column a = block_idx.x by the tree."""
    var a = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    if f[p(q, 7)] != Float32(0):
        return
    var X = p(q, 0)
    var nn = p(q, 1)
    var dd = p(q, 2)
    var M = p(q, 3)
    var j = p(q, 4)
    var sh_s = stack_allocation[TGR, Float32, address_space = AddressSpace.SHARED]()
    var sh_c = stack_allocation[TGR, Float32, address_space = AddressSpace.SHARED]()
    var cnt = Float32(0)
    var s = Float32(0)
    for i in range(tid, nn, TGR):
        if f[M + i * dd + j] != Float32(0):
            continue
        s = add(s, f[X + i * dd + a])
        cnt += 1
    sh_s[tid] = s
    sh_c[tid] = cnt
    barrier()
    var w = TGR // 2
    while w >= 1:
        if tid < w:
            sh_s[tid] = add(sh_s[tid], sh_s[tid + w])
            sh_c[tid] = sh_c[tid] + sh_c[tid + w]
        barrier()
        w //= 2
    if tid == 0:
        var total = sh_c[0]
        f[p(q, 5) + a] = div(sh_s[0], total) if total > 0 else Float32(0)
        if a == 0:
            f[p(q, 6)] = total


def ii_gram_fast_kernel(f: FP, q: IP):
    """`ii_gram_unit` for t = block_idx.x = a*d + b by the tree."""
    var t = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    if f[p(q, 7)] != Float32(0):
        return
    var X = p(q, 0)
    var nn = p(q, 1)
    var dd = p(q, 2)
    var M = p(q, 3)
    var j = p(q, 4)
    var a = t // dd
    var b = t % dd
    var ma = f[p(q, 5) + a]
    var mb = f[p(q, 5) + b]
    var sh_s = stack_allocation[TGR, Float32, address_space = AddressSpace.SHARED]()
    var s = Float32(0)
    for i in range(tid, nn, TGR):
        if f[M + i * dd + j] != Float32(0):
            continue
        s = add(s, mul(sub(f[X + i * dd + a], ma), sub(f[X + i * dd + b], mb)))
    sh_s[tid] = s
    barrier()
    var w = TGR // 2
    while w >= 1:
        if tid < w:
            sh_s[tid] = add(sh_s[tid], sh_s[tid + w])
        barrier()
        w //= 2
    if tid == 0:
        f[p(q, 6) + t] = sh_s[0]
