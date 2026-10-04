# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Non-metric SMACOF's disparities on resident matrices (lane
cpu2-l8-decomp, 2026-10-04): the device form of x_decomp/mds_iso.mojo, the
same per-element functions, so the words are the host column's.

SETUP (`dev_mds_setup_py`, once a fit; it waits twice, for m and for G):
  * keys[t] = Dis[t] for a nonzero strict-upper entry t < n^2, NaN for every
    other t < N2 (N2 the power of two >= n^2), idx[t] = t;
  * a bitonic sort of (key, idx) over N2 (log2(N2) (log2(N2) + 1) / 2
    launches, a thread per pair, `order_less`: NaN last, ties by position),
    which is the host's stable sort of the valid entries in row-major order;
  * m = the count of valid keys; the start flags of equal-key groups summed
    by a Hillis-Steele scan (log2(m) launches, integer adds) into gid, and
    gst scattered from the starts.
EACH ITERATION (`dev_mds_disp_py`, enqueued, no wait): out zeroed; the
first iteration scatters the keys; later ones run a thread per group (sum,
count), a thread per ISO_CHUNK groups (PAVA), a thread per pair of runs per
tree level (`iso_merge`), a max scan of the block heads (log2(G) launches),
a thread per group (its block mean) and a thread per pair (the scatter)."""
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import bitcast
from std.python import Python, PythonObject

from x_decomp.cells import F32Ptr, I32Ptr
from x_decomp.device import TPB, _blocks, _down, xd_ctx
from x_decomp.mds_iso import ISO_CHUNK, iso_chunk, iso_group_sum, iso_mean, iso_merge
from x_decomp.resident import X_DECOMP_POOL, _id, _n, _ptr
from x_decomp.select_ops import order_less


def _iptr(id: Int, n: Int) raises -> I32Ptr:
    """Pooled matrix id as int32 words (made on the host, never in a kernel)."""
    _ = _ptr(id, n)
    var p = X_DECOMP_POOL.get_or_create_ptr()
    return I32Ptr(unsafe_from_address=Int(p[].bufs[id].unsafe_ptr()))


def mds_keys_kernel(dis: F32Ptr, keys: F32Ptr, idx: I32Ptr, n: Int32, total: Int32):
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < Int(total):
        var nn = Int(n)
        var v = Float32(0)
        var ok = False
        if t < nn * nn:
            var i = t // nn
            var j = t - i * nn
            if j > i:
                v = dis.unsafe_load(t)
                ok = v != Float32(0)
        var nanv = bitcast[DType.float32](UInt32(0x7FC00000))
        keys.unsafe_store(t, v if ok else nanv)
        idx.unsafe_store(t, Int32(t))


def bitonic_kernel(keys: F32Ptr, idx: I32Ptr, kk: Int32, jj: Int32, total: Int32):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(total):
        var l = i ^ Int(jj)
        if l > i:
            var a = keys.unsafe_load(i)
            var ai = Int(idx.unsafe_load(i))
            var b = keys.unsafe_load(l)
            var bi = Int(idx.unsafe_load(l))
            var asc = (i & Int(kk)) == 0
            var swap = order_less(b, bi, a, ai) if asc else order_less(a, ai, b, bi)
            if swap:
                keys.unsafe_store(i, b)
                idx.unsafe_store(i, Int32(bi))
                keys.unsafe_store(l, a)
                idx.unsafe_store(l, Int32(ai))


def mds_count_kernel(keys: F32Ptr, out: I32Ptr, total: Int32):
    """out[0] = the count of valid (non-NaN) keys, which come first."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var tot = Int(total)
    if t < tot:
        var v = keys.unsafe_load(t)
        if t == 0 and not (v == v):
            out.unsafe_store(0, Int32(0))
        if v == v:
            var nxt_bad = True
            if t + 1 < tot:
                var w = keys.unsafe_load(t + 1)
                nxt_bad = not (w == w)
            if nxt_bad:
                out.unsafe_store(0, Int32(t + 1))


def mds_start_kernel(keys: F32Ptr, flags: I32Ptr, m: Int32):
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < Int(m):
        var s = 1 if t == 0 else (1 if keys.unsafe_load(t) != keys.unsafe_load(t - 1) else 0)
        flags.unsafe_store(t, Int32(s))


def scan_add_kernel(src: I32Ptr, dst: I32Ptr, off: Int32, m: Int32):
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < Int(m):
        var v = src.unsafe_load(t)
        if t >= Int(off):
            v = v + src.unsafe_load(t - Int(off))
        dst.unsafe_store(t, v)


def scan_max_kernel(src: I32Ptr, dst: I32Ptr, off: Int32, m: Int32):
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < Int(m):
        var v = src.unsafe_load(t)
        if t >= Int(off):
            var w = src.unsafe_load(t - Int(off))
            if w > v:
                v = w
        dst.unsafe_store(t, v)


def mds_gst_kernel(keys: F32Ptr, gid: I32Ptr, gst: I32Ptr, m: Int32, out: I32Ptr):
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var mm = Int(m)
    if t < mm:
        if t == 0 or keys.unsafe_load(t) != keys.unsafe_load(t - 1):
            gst.unsafe_store(Int(gid.unsafe_load(t)) - 1, Int32(t))
        if t == mm - 1:
            var g = Int(gid.unsafe_load(t))
            gst.unsafe_store(g, Int32(mm))
            out.unsafe_store(0, Int32(g))


def zero_kernel(dst: F32Ptr, count: Int32):
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < Int(count):
        dst.unsafe_store(t, Float32(0))


def scatter_keys_kernel(keys: F32Ptr, idx: I32Ptr, out: F32Ptr, m: Int32):
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < Int(m):
        out.unsafe_store(Int(idx.unsafe_load(t)), keys.unsafe_load(t))


def group_sum_kernel(d: F32Ptr, idx: I32Ptr, gst: I32Ptr, sm: F32Ptr, wt: I32Ptr, G: Int32):
    var g = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if g < Int(G):
        iso_group_sum(d, idx, gst, sm, wt, g)


def chunk_kernel(sm: F32Ptr, wt: I32Ptr, endp: I32Ptr, prv: I32Ptr, last: I32Ptr, hf: I32Ptr, G: Int32):
    var c = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var gt = Int(G)
    var g0 = c * ISO_CHUNK
    if g0 < gt:
        var g1 = g0 + ISO_CHUNK
        if g1 > gt:
            g1 = gt
        iso_chunk(sm, wt, endp, prv, last, hf, g0, g1, gt)


def merge_kernel(sm: F32Ptr, wt: I32Ptr, endp: I32Ptr, prv: I32Ptr, last: I32Ptr, hf: I32Ptr, w: Int32, G: Int32):
    var c = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var gt = Int(G)
    var ww = Int(w)
    var l0 = c * 2 * ww
    if l0 + ww < gt:
        var r1 = l0 + 2 * ww
        if r1 > gt:
            r1 = gt
        iso_merge(sm, wt, endp, prv, last, hf, l0, l0 + ww, r1, gt)


def head_kernel(hf: I32Ptr, hd: I32Ptr, G: Int32):
    var g = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if g < Int(G):
        hd.unsafe_store(g, Int32(g) if Int(hf.unsafe_load(g)) != 0 else Int32(-1))


def value_kernel(sm: F32Ptr, wt: I32Ptr, hd: I32Ptr, gv: F32Ptr, G: Int32):
    var g = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if g < Int(G):
        gv.unsafe_store(g, iso_mean(sm, wt, Int(hd.unsafe_load(g))))


def scatter_disp_kernel(gv: F32Ptr, gid: I32Ptr, idx: I32Ptr, out: F32Ptr, m: Int32):
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < Int(m):
        out.unsafe_store(Int(idx.unsafe_load(t)), gv.unsafe_load(Int(gid.unsafe_load(t)) - 1))


def _read_i32(id: Int) raises -> Int:
    """Word 0 of pooled matrix id as an int32, after every launch before it."""
    var p = X_DECOMP_POOL.get_or_create_ptr()
    var h = List[Float32](length=1, fill=Float32(0))
    var ctx = xd_ctx()
    _down(ctx, p[].bufs[id], F32Ptr(unsafe_from_address=Int(h.unsafe_ptr())), 1)
    ctx.synchronize()
    var v = Int(bitcast[DType.int32](h[0]))
    return v


def _scan(a: I32Ptr, b: I32Ptr, m: Int, use_max: Bool) raises -> Bool:
    """Inclusive scan of a[0, m) by ping-pong with b; True when the result
    ended in b."""
    var ctx = xd_ctx()
    var src = a
    var dst = b
    var in_b = False
    var off = 1
    while off < m:
        if use_max:
            ctx.enqueue_function[scan_max_kernel](src, dst, Int32(off), Int32(m), grid_dim=_blocks(m), block_dim=TPB)
        else:
            ctx.enqueue_function[scan_add_kernel](src, dst, Int32(off), Int32(m), grid_dim=_blocks(m), block_dim=TPB)
        var t = src
        src = dst
        dst = t
        in_b = not in_b
        off *= 2
    return in_b


def dev_mds_setup_py(dis: PythonObject, ids: PythonObject, p: PythonObject) raises -> PythonObject:
    """ids = [keys, idx, gid, tmp, gst, word] device matrices (keys, idx,
    gid, tmp: N2 values; gst: N2 + 1; word: 1); p = [n, N2]. Returns (m, G).
    gid always ends in ids[2] (copied from tmp when the scan ended there)."""
    var n = _n(p, 0)
    var N2 = _n(p, 1)
    if n < 1 or n * n > N2 or (N2 & (N2 - 1)) != 0:
        raise Error("x_decomp: mds setup size out of range")
    var pd = _ptr(_id(dis), n * n)
    var kid = _id(ids[0])
    var pk = _ptr(kid, N2)
    var pi = _iptr(_id(ids[1]), N2)
    var gid_id = _id(ids[2])
    var pg = _iptr(gid_id, N2)
    var pt = _iptr(_id(ids[3]), N2)
    var ps = _iptr(_id(ids[4]), N2 + 1)
    var wid = _id(ids[5])
    var pw = _iptr(wid, 1)
    var ctx = xd_ctx()
    ctx.enqueue_function[mds_keys_kernel](pd, pk, pi, Int32(n), Int32(N2), grid_dim=_blocks(N2), block_dim=TPB)
    var kk = 2
    while kk <= N2:
        var jj = kk // 2
        while jj > 0:
            ctx.enqueue_function[bitonic_kernel](pk, pi, Int32(kk), Int32(jj), Int32(N2), grid_dim=_blocks(N2), block_dim=TPB)
            jj //= 2
        kk *= 2
    ctx.enqueue_function[mds_count_kernel](pk, pw, Int32(N2), grid_dim=_blocks(N2), block_dim=TPB)
    var m = _read_i32(wid)
    if m == 0:
        return Python.tuple(0, 0)
    ctx.enqueue_function[mds_start_kernel](pk, pg, Int32(m), grid_dim=_blocks(m), block_dim=TPB)
    if _scan(pg, pt, m, False):
        ctx.enqueue_function[scan_add_kernel](pt, pg, Int32(m + 1), Int32(m), grid_dim=_blocks(m), block_dim=TPB)
    ctx.enqueue_function[mds_gst_kernel](pk, pg, ps, Int32(m), pw, grid_dim=_blocks(m), block_dim=TPB)
    var G = _read_i32(wid)
    return Python.tuple(m, G)


def dev_mds_disp_py(d: PythonObject, out: PythonObject, ids: PythonObject, p: PythonObject) raises -> PythonObject:
    """One iteration's upper-triangle disparities into out (n x n), enqueued.
    ids = [keys, idx, gid, gst, sm, wt, end, prv, last, hf, hd0, hd1, gv]
    device matrices; p = [n, m, G, first]."""
    var n = _n(p, 0)
    var m = _n(p, 1)
    var G = _n(p, 2)
    var first = Int(py=p[3]) != 0
    var po = _ptr(_id(out), n * n)
    var ctx = xd_ctx()
    ctx.enqueue_function[zero_kernel](po, Int32(n * n), grid_dim=_blocks(n * n), block_dim=TPB)
    if m == 0:
        return PythonObject(0)
    var pk = _ptr(_id(ids[0]), m)
    var pi = _iptr(_id(ids[1]), m)
    var pg = _iptr(_id(ids[2]), m)
    if first:
        ctx.enqueue_function[scatter_keys_kernel](pk, pi, po, Int32(m), grid_dim=_blocks(m), block_dim=TPB)
        return PythonObject(m)
    var pd = _ptr(_id(d), n * n)
    var ps = _iptr(_id(ids[3]), G + 1)
    var sm = _ptr(_id(ids[4]), G)
    var wt = _iptr(_id(ids[5]), G)
    var endp = _iptr(_id(ids[6]), G)
    var prv = _iptr(_id(ids[7]), G)
    var last = _iptr(_id(ids[8]), G)
    var hf = _iptr(_id(ids[9]), G)
    var h0 = _iptr(_id(ids[10]), G)
    var h1 = _iptr(_id(ids[11]), G)
    var gv = _ptr(_id(ids[12]), G)
    ctx.enqueue_function[group_sum_kernel](pd, pi, ps, sm, wt, Int32(G), grid_dim=_blocks(G), block_dim=TPB)
    var nch = (G + ISO_CHUNK - 1) // ISO_CHUNK
    ctx.enqueue_function[chunk_kernel](sm, wt, endp, prv, last, hf, Int32(G), grid_dim=_blocks(nch), block_dim=TPB)
    var w = ISO_CHUNK
    while w < G:
        var pairs = (G + 2 * w - 1) // (2 * w)
        ctx.enqueue_function[merge_kernel](sm, wt, endp, prv, last, hf, Int32(w), Int32(G), grid_dim=_blocks(pairs), block_dim=TPB)
        w *= 2
    ctx.enqueue_function[head_kernel](hf, h0, Int32(G), grid_dim=_blocks(G), block_dim=TPB)
    var hd = h0
    if _scan(h0, h1, G, True):
        hd = h1
    ctx.enqueue_function[value_kernel](sm, wt, hd, gv, Int32(G), grid_dim=_blocks(G), block_dim=TPB)
    ctx.enqueue_function[scatter_disp_kernel](gv, pg, pi, po, Int32(m), grid_dim=_blocks(m), block_dim=TPB)
    return PythonObject(m)
