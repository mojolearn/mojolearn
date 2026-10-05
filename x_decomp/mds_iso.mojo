# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Non-metric SMACOF's disparities (MDS(metric_mds=False)) as ONE isotonic
regression of the current distances on the fixed dissimilarities (lane
cpu2-l8-decomp, 2026-10-04, re-audit L8 `MDS._nm_native`): the core shared
by the device route (x_decomp/mds_iso_dev.mojo) and the host column (the
`mds_*_host_py` entries below), so both give the same words.

The Python bookkeeping it replaces ran on the host every iteration: the
pairs gathered by host-address helpers, a host sort by (dissimilarity,
distance), x_linear's isotonic fit and predict on host buffers, and the
scatter and mirror. Here:

SETUP (once a fit): the m nonzero strict-upper-triangle dissimilarities,
ordered by (value, row-major position) (`keys`, `idx`); `gid[t]` = 1 + the
index of t's group of equal values (an inclusive count of group starts);
`gst[g]` = the first sorted position of group g, gst[G] = m.

EACH ITERATION (after the first, whose disparities are the dissimilarities
themselves):
  1. each group's sum of the current distances (its pairs in sorted order,
     float32 adds) and count: sklearn's `_make_unique` merge of equal X
     (their mean y, weight = count);
  2. the pool-adjacent-violators algorithm over the G group means, in
     chunks of ISO_CHUNK groups (each chunk on its own), then the chunks
     joined pairwise in a fixed tree (`iso_merge`: pool across the
     junction, then left and right until no violation). The isotonic fit
     is unique, so the result is the fit sklearn's single pass finds; the
     float32 sums are formed in this fixed order on every column;
  3. each group's value = its block's mean (the prediction at a threshold
     IS the fitted value; clip and interpolation never apply at the knots),
     scattered to the pair's upper position.
Python then normalizes and mirrors on the kit (ss, scale, P + P^T).

Integers (positions, counts, links) travel in pooled float buffers as int32
bits; the pointers are made on the host before launch (no int-to-pointer
cast in a kernel), and every function here is inlined into its kernel."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from x_decomp.cells import F32Ptr, I32Ptr, add, div0
from x_decomp.moves import argsort_f32

#: groups one thread pools on its own before the tree joins the chunks
comptime ISO_CHUNK = 256


@always_inline
def iso_mean(sm: F32Ptr, wt: I32Ptr, a: Int) -> Float32:
    return div0(sm.unsafe_load(a), Float32(Int(wt.unsafe_load(a))))


@always_inline
def iso_pool(sm: F32Ptr, wt: I32Ptr, endp: I32Ptr, prv: I32Ptr, hf: I32Ptr, a: Int, b: Int, bound: Int):
    """Block b (the one after a) into block a. The back link of the block
    after them is written only inside the caller's run (< bound): the next
    run's first link belongs to the thread pooling that run, and the join
    (`iso_merge`) sets it."""
    sm.unsafe_store(a, add(sm.unsafe_load(a), sm.unsafe_load(b)))
    wt.unsafe_store(a, wt.unsafe_load(a) + wt.unsafe_load(b))
    var e = Int(endp.unsafe_load(b))
    endp.unsafe_store(a, Int32(e))
    if e < bound:
        prv.unsafe_store(e, Int32(a))
    hf.unsafe_store(b, Int32(0))


@always_inline
def iso_chunk(
    sm: F32Ptr, wt: I32Ptr, endp: I32Ptr, prv: I32Ptr, last: I32Ptr, hf: I32Ptr, g0: Int, g1: Int, gtot: Int
):
    """PAVA over groups [g0, g1) (each group a block on entry: sm and wt
    hold its sum and count); last[g0] = the head of its last block."""
    for g in range(g0, g1):
        endp.unsafe_store(g, Int32(g + 1))
        prv.unsafe_store(g, Int32(g - 1))
        hf.unsafe_store(g, Int32(1))
    var x = g0
    var y = g0 + 1
    while y < g1:
        if iso_mean(sm, wt, x) > iso_mean(sm, wt, y):
            iso_pool(sm, wt, endp, prv, hf, x, y, g1)
            while x > g0:
                var px = Int(prv.unsafe_load(x))
                if iso_mean(sm, wt, px) > iso_mean(sm, wt, x):
                    iso_pool(sm, wt, endp, prv, hf, px, x, g1)
                    x = px
                else:
                    break
            y = Int(endp.unsafe_load(x))
        else:
            x = y
            y = Int(endp.unsafe_load(x))
    last.unsafe_store(g0, Int32(x))


@always_inline
def iso_merge(
    sm: F32Ptr, wt: I32Ptr, endp: I32Ptr, prv: I32Ptr, last: I32Ptr, hf: I32Ptr, l0: Int, m0: Int, r1: Int,
    gtot: Int,
):
    """Join the pooled runs [l0, m0) and [m0, r1): pool across the junction
    until no adjacent pair of blocks decreases; last[l0] = the joined run's
    last head."""
    var x = Int(last.unsafe_load(l0))
    prv.unsafe_store(m0, Int32(x))
    var y = m0
    while True:
        if iso_mean(sm, wt, x) > iso_mean(sm, wt, y):
            iso_pool(sm, wt, endp, prv, hf, x, y, r1)
            while x > l0:
                var px = Int(prv.unsafe_load(x))
                if iso_mean(sm, wt, px) > iso_mean(sm, wt, x):
                    iso_pool(sm, wt, endp, prv, hf, px, x, r1)
                    x = px
                else:
                    break
            y = Int(endp.unsafe_load(x))
            if y >= r1:
                break
        else:
            break
    if Int(endp.unsafe_load(x)) >= r1:
        last.unsafe_store(l0, Int32(x))
    else:
        last.unsafe_store(l0, last.unsafe_load(m0))


@always_inline
def iso_group_sum(d: F32Ptr, idx: I32Ptr, gst: I32Ptr, sm: F32Ptr, wt: I32Ptr, g: Int):
    """Group g's sum of the current distances (sorted order) and count."""
    var t0 = Int(gst.unsafe_load(g))
    var t1 = Int(gst.unsafe_load(g + 1))
    var s = Float32(0)
    for t in range(t0, t1):
        s = add(s, d.unsafe_load(Int(idx.unsafe_load(t))))
    sm.unsafe_store(g, s)
    wt.unsafe_store(g, Int32(t1 - t0))


def _fp(o: PythonObject) raises -> F32Ptr:
    var a = Int(py=o)
    if a == 0:
        raise Error("x_decomp: null buffer address")
    return F32Ptr(unsafe_from_address=a)


def _ip(o: PythonObject) raises -> I32Ptr:
    var a = Int(py=o)
    if a == 0:
        raise Error("x_decomp: null buffer address")
    return I32Ptr(unsafe_from_address=a)


def mds_setup_host_py(
    dis: PythonObject, keys: PythonObject, idx: PythonObject, gid: PythonObject, gst: PythonObject, p: PythonObject
) raises -> PythonObject:
    """The setup on host buffers (p = [n]; keys, idx, gid hold n (n - 1) / 2,
    gst n (n - 1) / 2 + 1). Returns (m, G)."""
    var n = Int(py=p[0])
    if n < 1 or n * n > 2147483647:
        raise Error("x_decomp: mds setup size out of range")
    var pd = _fp(dis)
    var pk = _fp(keys)
    var pi = _ip(idx)
    var pg = _ip(gid)
    var ps = _ip(gst)
    var vals = List[Float32]()
    var pos = List[Int32]()
    for i in range(n):
        for j in range(i + 1, n):
            var v = pd.unsafe_load(i * n + j)
            if v != Float32(0):
                vals.append(v)
                pos.append(Int32(i * n + j))
    var m = len(vals)
    var order = List[Int32](length=max(m, 1), fill=Int32(0))
    if m > 0:
        argsort_f32(
            F32Ptr(unsafe_from_address=Int(vals.unsafe_ptr())), m,
            I32Ptr(unsafe_from_address=Int(order.unsafe_ptr())),
        )
    var g = 0
    for t in range(m):
        var o = Int(order[t])
        var v = vals[o]
        pk.unsafe_store(t, v)
        pi.unsafe_store(t, pos[o])
        if t == 0 or v != pk.unsafe_load(t - 1):
            ps.unsafe_store(g, Int32(t))
            g += 1
        pg.unsafe_store(t, Int32(g))
    ps.unsafe_store(g, Int32(m))
    _ = len(vals)
    _ = len(order)
    return Python.tuple(m, g)


def mds_disp_host_py(d: PythonObject, dst: PythonObject, a: PythonObject, p: PythonObject) raises -> PythonObject:
    """One iteration's upper-triangle disparities into dst (n x n, zeroed
    here) on host buffers. a = [keys, idx, gid, gst, sm, wt, end, prv,
    last, hf, gv] addresses; p = [n, m, G, first]."""
    var n = Int(py=p[0])
    var m = Int(py=p[1])
    var G = Int(py=p[2])
    var first = Int(py=p[3]) != 0
    var pd = _fp(d)
    var po = _fp(dst)
    var pk = _fp(a[0])
    var pi = _ip(a[1])
    var pg = _ip(a[2])
    var ps = _ip(a[3])
    var sm = _fp(a[4])
    var wt = _ip(a[5])
    var endp = _ip(a[6])
    var prv = _ip(a[7])
    var last = _ip(a[8])
    var hf = _ip(a[9])
    var gv = _fp(a[10])
    with GILReleased(Python()):
        for q in range(n * n):
            po.unsafe_store(q, Float32(0))
        if first:
            for t in range(m):
                po.unsafe_store(Int(pi.unsafe_load(t)), pk.unsafe_load(t))
        elif G > 0:
            for g in range(G):
                iso_group_sum(pd, pi, ps, sm, wt, g)
            var c0 = 0
            while c0 < G:
                iso_chunk(sm, wt, endp, prv, last, hf, c0, min(G, c0 + ISO_CHUNK), G)
                c0 += ISO_CHUNK
            var w = ISO_CHUNK
            while w < G:
                var l0 = 0
                while l0 < G:
                    if l0 + w < G:
                        iso_merge(sm, wt, endp, prv, last, hf, l0, l0 + w, min(G, l0 + 2 * w), G)
                    l0 += 2 * w
                w *= 2
            var h = 0
            for g in range(G):
                if Int(hf.unsafe_load(g)) != 0:
                    h = g
                gv.unsafe_store(g, iso_mean(sm, wt, h))
            for t in range(m):
                po.unsafe_store(Int(pi.unsafe_load(t)), gv.unsafe_load(Int(pg.unsafe_load(t)) - 1))
    return PythonObject(m)
