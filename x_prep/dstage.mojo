# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Staged device folds for IDENTICAL (lane prep-apple, 2026-09-28).

`col_stats` and `pt_fold` fold a column in row order on ONE thread. On the
device that thread spends its time waiting for memory: d columns are d
threads, and each keeps only RUN rows in flight. Here a threadgroup of SB
threads serves ONE column: together they copy a tile of STILE rows into
threadgroup memory (hundreds of loads in flight), then thread 0 folds the
tile in row order with the unit's own step functions (`_cs_take`,
`_ss_take`, `_pt_take1`, `_pt_take2`, `pt_finish`): the same operations on
the same values in the same order, so the same words. The other threads
only load. MOJOLEARN_XPREP_STAGED=0 keeps the units (x_prep/device.mojo).
"""
from std.gpu import block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from checks.numerics import ftz
from x_prep.common import FP, IP, p, st
from x_prep.prims import div, _cs_take, _ss_take
from x_prep.transform import PT_STATE, pt_finish, _pt_take1, _pt_take2

comptime SB = 256
#: rows per tile (two tiles of float32 = 16 KB of threadgroup memory)
comptime STILE = 2048


def col_stats_staged_kernel(f: FP, X: Int32, n: Int32, d: Int32, O: Int32):
    """`col_stats_unit` for column block_idx.x, thread 0 folding tiles."""
    var c = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var nn = Int(n)
    var dd = Int(d)
    var xb = Int(X)
    var sh = stack_allocation[STILE, Float32, address_space = AddressSpace.SHARED]()
    var cnt = 0
    var s = Float32(0)
    var lo = Float32(0)
    var hi = Float32(0)
    var ma = Float32(0)
    for base in range(0, nn, STILE):
        var m = min(STILE, nn - base)
        for u in range(tid, m, SB):
            sh[u] = f[xb + (base + u) * dd + c]
        barrier()
        if tid == 0:
            for u in range(m):
                _cs_take(ftz(sh[u]), cnt, s, lo, hi, ma)
        barrier()
    var mean = Float32(0)
    var ss = Float32(0)
    if tid == 0 and cnt > 0:
        mean = div(s, Float32(cnt))
    for base in range(0, nn, STILE):
        var m = min(STILE, nn - base)
        for u in range(tid, m, SB):
            sh[u] = f[xb + (base + u) * dd + c]
        barrier()
        if tid == 0 and cnt > 0:
            for u in range(m):
                _ss_take(ftz(sh[u]), mean, ss)
        barrier()
    if tid == 0:
        var var_ = Float32(0)
        if cnt > 0:
            var_ = div(ss, Float32(cnt))
        var o = Int(O)
        st(f, o + c, Float32(cnt))
        st(f, o + dd + c, mean)
        st(f, o + 2 * dd + c, var_)
        st(f, o + 3 * dd + c, lo)
        st(f, o + 4 * dd + c, hi)
        st(f, o + 5 * dd + c, ma)


def pt_fold_staged_kernel(f: FP, q: IP):
    """`pt_fold_unit` for column block_idx.x (q: the stage's device params),
    thread 0 folding tiles of T (column major) and, at K = 0, of X."""
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
    var Tc = T + c * nn
    var sht = stack_allocation[STILE, Float32, address_space = AddressSpace.SHARED]()
    var shx = stack_allocation[STILE, Float32, address_space = AddressSpace.SHARED]()
    var cnt = 0
    var sm = Float32(0)
    var sj = f[S + 8]
    if first:
        sj = Float32(0)
    for base in range(0, nn, STILE):
        var m = min(STILE, nn - base)
        for u in range(tid, m, SB):
            sht[u] = f[Tc + base + u]
            if first:
                shx[u] = f[X + (base + u) * dd + c]
        barrier()
        if tid == 0:
            if first:
                for u in range(m):
                    _pt_take1(ftz(shx[u]), sht[u], method, True, cnt, sm, sj)
            else:
                for u in range(m):
                    var tv = sht[u]
                    _pt_take1(tv, tv, method, False, cnt, sm, sj)
        barrier()
    var mean = Float32(0)
    var ss = Float32(0)
    if tid == 0 and cnt > 0:
        mean = div(sm, Float32(cnt))
    for base in range(0, nn, STILE):
        var m = min(STILE, nn - base)
        for u in range(tid, m, SB):
            sht[u] = f[Tc + base + u]
        barrier()
        if tid == 0 and cnt > 0:
            for u in range(m):
                var tv = sht[u]
                _pt_take2(tv, tv, mean, ss)
        barrier()
    if tid == 0:
        if first:
            f[S + 8] = sj
        pt_finish(c, f, q, cnt, sj, ss)
