# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The CPU column of `x_neighbors/iter_device.mojo`: the same loop over the
same items. HOST ONLY."""
from std.memory import bitcast
from std.sys.compile import is_defined

from x_neighbors.items import (
    FP, IP, absdiff_sum_item, matmul_item, lp_clamp_item, ls_clamp_item,
)

comptime _SABOTAGE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()


def op_lp_iterate(
    g: Int, ld: Int, ystatic: Int, unlabeled: Int, info: Int,
    n: Int, c: Int, max_iter: Int, variant: Int, tol_hi: Int, tol_lo: Int,
    alpha: Float32,
) raises:
    var tol = bitcast[DType.float64]((UInt64(tol_hi) << UInt64(32)) | UInt64(tol_lo))
    var nc = n * c
    var pg = FP(unsafe_from_address=g)
    var pys = FP(unsafe_from_address=ystatic)
    var punl = IP(unsafe_from_address=unlabeled)
    var a = List[Float32](length=nc if nc > 0 else 1, fill=Float32(0))
    var b = List[Float32](length=nc if nc > 0 else 1, fill=Float32(0))
    var nxt = List[Float32](length=nc if nc > 0 else 1, fill=Float32(0))
    var s = List[Float32](length=1, fill=Float32(0))
    var pld = FP(unsafe_from_address=ld)
    for i in range(nc):
        a[i] = pld.unsafe_load(i)
    var cur = FP(unsafe_from_address=Int(a.unsafe_ptr()))
    var prev = FP(unsafe_from_address=Int(b.unsafe_ptr()))
    var pn = FP(unsafe_from_address=Int(nxt.unsafe_ptr()))
    var ps = FP(unsafe_from_address=Int(s.unsafe_ptr()))
    var n_iter = 0
    var converged = False
    for it in range(max_iter):
        n_iter = it
        absdiff_sum_item(0, cur, prev, ps, nc)
        if Float64(ps.unsafe_load(0)) < tol:
            converged = True
            break
        for t in range(nc):
            matmul_item(t, pg, cur, pn, n, n, c)
        if variant == 0:
            for t in range(n):
                lp_clamp_item(t, pn, pys, punl, prev, n, c)
        else:
            for t in range(nc):
                ls_clamp_item(t, pn, pys, prev, nc, alpha)
        var tmp = cur
        cur = prev
        prev = tmp
    if not converged:
        n_iter += 1
    for i in range(nc):
        pld.unsafe_store(i, cur.unsafe_load(i))
    comptime if _SABOTAGE:
        if nc > 0:
            pld.unsafe_store(0, pld.unsafe_load(0) + Float32(1e-3))
    var inf = IP(unsafe_from_address=info)
    inf.unsafe_store(0, Int32(n_iter))
    inf.unsafe_store(1, Int32(1 if converged else 0))
    _ = a^
    _ = b^
    _ = nxt^
    _ = s^
