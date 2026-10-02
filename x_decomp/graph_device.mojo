# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The cells of x_decomp/graph_cells.mojo on device-resident matrices
(lane hr2-graph-embed, 2026-10-02; GPU binding only). Every launch is
parallel over rows, cells, components or component pairs; the only host
reads are the counts that size the next buffers (the arc total, the
component count, the hooking round's flag).

Connected components: hooking (each nonzero W[u, v] lowers the larger of
the two root labels to the smaller, an integer atomic min) and pointer
jumping, rounds until no edge changes a label. The labels only fall, each
to a node of the same component, so the end is the unique min-label fixed
point: the lowest node of every component, the host column's union-find
result word for word. The components are then numbered by an exclusive
scan of the root flags.

The scan: per-block Hillis-Steele scans of GS_TPB integers in threadgroup
memory, the block totals scanned the same way (recursively), then added
back. Integer sums: the same words in any order."""
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import stack_allocation
from std.atomic import Atomic
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from x_decomp.cells import F32Ptr, I32Ptr
from x_decomp.device import TPB, _blocks, xd_ctx, dijkstra_kernel, DIJKSTRA_ROWS
from x_decomp.resident import pool_alloc, pool_free, _ptr, _id, _n
from x_decomp.graph_cells import (
    knn_select_row,
    knn_dense_row,
    radius_cell,
    lle_iw_row,
    rowbest_row,
    members_comp,
    count_comp,
    join_pair,
    arc_count_row,
    arc_fill_row,
)

comptime GS_TPB = 256


@always_inline
def _t() -> Int:
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


def _ip(p: F32Ptr) -> I32Ptr:
    """A pooled float buffer's words as int32 (host side, before launch)."""
    return I32Ptr(unsafe_from_address=Int(p))


# ---- kernels ---------------------------------------------------------------


def xg_zero_kernel(w: F32Ptr, count: Int32):
    var t = _t()
    if t < Int(count):
        w.unsafe_store(t, Float32(0))


def xg_knn_kernel(
    d: F32Ptr, sv: F32Ptr, si: F32Ptr, idx: F32Ptr, dst: F32Ptr, n: Int32, m: Int32, take: Int32, excl: Int32, nn: Int32
):
    var i = _t()
    if i < Int(n):
        knn_select_row(d, i, Int(m), Int(take), Int(excl) != 0, Int(nn), sv, si, idx, dst)


def xg_knn_dense_kernel(idx: F32Ptr, w: F32Ptr, wout: F32Ptr, n: Int32, nn: Int32):
    var i = _t()
    if i < Int(n):
        knn_dense_row(i, idx, w, Int(n), Int(nn), wout)


def xg_radius_kernel(d: F32Ptr, wout: F32Ptr, n: Int32, r: Float32):
    var t = _t()
    if t < Int(n) * Int(n):
        radius_cell(t, d, Int(n), r, wout)


def xg_lle_iw_kernel(idx: F32Ptr, wb: F32Ptr, wout: F32Ptr, n: Int32, nn: Int32):
    var i = _t()
    if i < Int(n):
        lle_iw_row(i, idx, wb, Int(n), Int(nn), wout)


def xg_label_init_kernel(lab: I32Ptr, n: Int32):
    var v = _t()
    if v < Int(n):
        lab.unsafe_store(v, Int32(v))


def xg_hook_kernel(w: F32Ptr, lab: I32Ptr, n: Int32, changed: I32Ptr):
    """Row u's nonzero cells hook (either direction counts: hooking is
    symmetric in u and v)."""
    var u = _t()
    var nn = Int(n)
    if u < nn:
        for v in range(nn):
            if w.unsafe_load(u * nn + v) != Float32(0):
                var a = lab.unsafe_load(u)
                var b = lab.unsafe_load(v)
                if a != b:
                    var lo = a if a < b else b
                    var hi = b if a < b else a
                    if lab.unsafe_load(Int(hi)) > lo:
                        _ = Atomic[DType.int32].min(lab + Int(hi), lo)
                        changed.unsafe_store(0, Int32(1))


def xg_jump_kernel(lab: I32Ptr, n: Int32):
    var v = _t()
    if v < Int(n):
        var p = Int(lab.unsafe_load(v))
        while Int(lab.unsafe_load(p)) != p:
            p = Int(lab.unsafe_load(p))
        lab.unsafe_store(v, Int32(p))


def xg_root_flag_kernel(lab: I32Ptr, flag: I32Ptr, n: Int32):
    var v = _t()
    if v < Int(n):
        flag.unsafe_store(v, Int32(1) if Int(lab.unsafe_load(v)) == v else Int32(0))


def xg_comp_kernel(lab: I32Ptr, rank: I32Ptr, comp: F32Ptr, n: Int32):
    var v = _t()
    if v < Int(n):
        comp.unsafe_store(v, Float32(Int(rank.unsafe_load(Int(lab.unsafe_load(v))))))


def xg_rowbest_kernel(d: F32Ptr, comp: F32Ptr, n: Int32, C: Int32, rv: F32Ptr, rj: F32Ptr):
    var i = _t()
    if i < Int(n):
        rowbest_row(i, d, comp, Int(n), Int(C), rv, rj)


def xg_count_comp_kernel(comp: F32Ptr, n: Int32, C: Int32, cnt: I32Ptr):
    var b = _t()
    if b < Int(C):
        count_comp(b, comp, Int(n), cnt)


def xg_members_kernel(comp: F32Ptr, n: Int32, C: Int32, start: I32Ptr, mem: I32Ptr):
    var b = _t()
    if b < Int(C):
        members_comp(b, comp, Int(n), start, mem)


def xg_join_kernel(n: Int32, C: Int32, start: I32Ptr, mem: I32Ptr, rv: F32Ptr, rj: F32Ptr, w: F32Ptr):
    var t = _t()
    var c = Int(C)
    if t < c * c:
        var a = t // c
        var b = t - a * c
        if a < b:
            join_pair(a, b, Int(n), c, start, mem, rv, rj, w)


def xg_arc_count_kernel(w: F32Ptr, n: Int32, cnt: I32Ptr):
    var u = _t()
    if u < Int(n):
        arc_count_row(u, w, Int(n), cnt)


def xg_arc_fill_kernel(w: F32Ptr, n: Int32, rp: I32Ptr, adj: I32Ptr, wa: F32Ptr, wb: F32Ptr):
    var u = _t()
    if u < Int(n):
        arc_fill_row(u, w, Int(n), rp, adj, wa, wb)


def xg_scan_block_kernel(src: I32Ptr, dst: I32Ptr, sums: I32Ptr, nsrc: Int32, nout: Int32):
    """dst[g] = the exclusive prefix of src within this block (src past nsrc
    reads 0; dst written for g < nout); sums[block] = the block's total."""
    var sh = stack_allocation[GS_TPB, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    var g = Int(block_idx.x) * GS_TPB + tid
    var x = Int32(0)
    if g < Int(nsrc):
        x = src.unsafe_load(g)
    sh[tid] = x
    barrier()
    var off = 1
    while off < GS_TPB:
        var y = Int32(0)
        if tid >= off:
            y = sh[tid - off]
        barrier()
        sh[tid] = sh[tid] + y
        barrier()
        off *= 2
    var incl = sh[tid]
    if g < Int(nout):
        dst.unsafe_store(g, incl - x)
    if tid == GS_TPB - 1:
        sums.unsafe_store(Int(block_idx.x), incl)


def xg_scan_add_kernel(dst: I32Ptr, offs: I32Ptr, nout: Int32):
    var g = Int(block_idx.x) * GS_TPB + Int(thread_idx.x)
    if g < Int(nout):
        dst.unsafe_store(g, dst.unsafe_load(g) + offs.unsafe_load(Int(block_idx.x)))


# ---- host-side launch helpers ----------------------------------------------


def _xscan(ctx: DeviceContext, src: I32Ptr, dst: I32Ptr, nsrc: Int, nout: Int) raises:
    """dst[0 .. nout) = the exclusive prefix sums of src[0 .. nsrc) (zeros
    past nsrc), so dst[nsrc] is the total when nout = nsrc + 1."""
    var blocks = (nout + GS_TPB - 1) // GS_TPB
    if blocks <= 0:
        return
    var sid = pool_alloc(blocks)
    var sums = _ip(_ptr(sid, blocks))
    ctx.enqueue_function[xg_scan_block_kernel](src, dst, sums, Int32(nsrc), Int32(nout), grid_dim=blocks, block_dim=GS_TPB)
    if blocks > 1:
        var oid = pool_alloc(blocks)
        var offs = _ip(_ptr(oid, blocks))
        _xscan(ctx, sums, offs, blocks, blocks)
        ctx.enqueue_function[xg_scan_add_kernel](dst, offs, Int32(nout), grid_dim=blocks, block_dim=GS_TPB)
        pool_free(oid)
    pool_free(sid)


def _read_i32(ctx: DeviceContext, p: I32Ptr) raises -> Int:
    """One int32 of a device buffer, after every launch before it."""
    var h = ctx.enqueue_create_host_buffer[DType.int32](1)
    var dv = ctx.enqueue_create_buffer[DType.int32](1)
    ctx.enqueue_function[xg_copy1_kernel](p, dv.unsafe_ptr(), grid_dim=1, block_dim=1)
    ctx.enqueue_copy(dst_buf=h, src_buf=dv)
    ctx.synchronize()
    var v = Int(h.unsafe_ptr().unsafe_load(0))
    _ = dv^
    _ = h^
    return v


def xg_copy1_kernel(src: I32Ptr, dst: MutPointer[Int32, MutAnyOrigin]):
    if _t() == 0:
        dst.unsafe_store(0, src.unsafe_load(0))


def _zero(ctx: DeviceContext, p: F32Ptr, count: Int) raises:
    if count > 0:
        ctx.enqueue_function[xg_zero_kernel](p, Int32(count), grid_dim=_blocks(count), block_dim=TPB)


# ---- Python entries (resident ids) ------------------------------------------


def dev_graph_knn_py(d: PythonObject, idx: PythonObject, dst: PythonObject, p: PythonObject) raises -> PythonObject:
    """idx / dst (n x nn): the nn nearest columns of every row of the
    resident D (n x m), `knn_select_row`. p = [n, m, nn, excl]."""
    var n = _n(p, 0)
    var m = _n(p, 1)
    var nn = _n(p, 2)
    var excl = Int(py=p[3]) != 0
    var take = nn + (1 if excl else 0)
    if take > m or m >= 1 << 24 or n * m > 2147483647:
        raise Error("x_decomp: knn asks for more neighbors than columns")
    if n == 0 or nn == 0:
        return PythonObject(0)
    var ctx = xd_ctx()
    var sid = pool_alloc(n * take)
    var iid = pool_alloc(n * take)
    ctx.enqueue_function[xg_knn_kernel](
        _ptr(_id(d), n * m), _ptr(sid, n * take), _ptr(iid, n * take), _ptr(_id(idx), n * nn), _ptr(_id(dst), n * nn),
        Int32(n), Int32(m), Int32(take), Int32(1 if excl else 0), Int32(nn), grid_dim=_blocks(n), block_dim=TPB,
    )
    pool_free(sid)
    pool_free(iid)
    return PythonObject(n)


def dev_graph_knn_dense_py(idx: PythonObject, w: PythonObject, wout: PythonObject, p: PythonObject) raises -> PythonObject:
    var n = _n(p, 0)
    var nn = _n(p, 1)
    if n * n > 2147483647:
        raise Error("x_decomp: graph exceeds the Int32 index bound")
    var ctx = xd_ctx()
    var po = _ptr(_id(wout), n * n)
    _zero(ctx, po, n * n)
    if n > 0 and nn > 0:
        ctx.enqueue_function[xg_knn_dense_kernel](
            _ptr(_id(idx), n * nn), _ptr(_id(w), n * nn), po, Int32(n), Int32(nn), grid_dim=_blocks(n), block_dim=TPB
        )
    return PythonObject(n)


def dev_graph_radius_py(d: PythonObject, wout: PythonObject, p: PythonObject, r: PythonObject) raises -> PythonObject:
    var n = _n(p, 0)
    if n * n > 2147483647:
        raise Error("x_decomp: graph exceeds the Int32 index bound")
    if n > 0:
        xd_ctx().enqueue_function[xg_radius_kernel](
            _ptr(_id(d), n * n), _ptr(_id(wout), n * n), Int32(n), Float32(Float64(py=r)),
            grid_dim=_blocks(n * n), block_dim=TPB,
        )
    return PythonObject(n)


def dev_graph_lle_iw_py(idx: PythonObject, wb: PythonObject, wout: PythonObject, p: PythonObject) raises -> PythonObject:
    var n = _n(p, 0)
    var nn = _n(p, 1)
    if n * n > 2147483647:
        raise Error("x_decomp: graph exceeds the Int32 index bound")
    var ctx = xd_ctx()
    var po = _ptr(_id(wout), n * n)
    _zero(ctx, po, n * n)
    if n > 0:
        ctx.enqueue_function[xg_lle_iw_kernel](
            _ptr(_id(idx), n * nn), _ptr(_id(wb), n * nn), po, Int32(n), Int32(nn), grid_dim=_blocks(n), block_dim=TPB
        )
    return PythonObject(n)


def dev_graph_components_py(w: PythonObject, comp: PythonObject, p: PythonObject) raises -> PythonObject:
    """comp (n floats): every node's component, numbered by lowest node;
    returns the number of components."""
    var n = _n(p, 0)
    if n == 0:
        return PythonObject(0)
    if n * n > 2147483647:
        raise Error("x_decomp: graph exceeds the Int32 index bound")
    var ctx = xd_ctx()
    var pw = _ptr(_id(w), n * n)
    var lid = pool_alloc(n)
    var fid = pool_alloc(n)
    var rid = pool_alloc(n + 1)
    var cid = pool_alloc(1)
    var lab = _ip(_ptr(lid, n))
    var flag = _ip(_ptr(fid, n))
    var rank = _ip(_ptr(rid, n + 1))
    var chg = _ip(_ptr(cid, 1))
    var hb = ctx.enqueue_create_host_buffer[DType.int32](1)
    var dc = ctx.enqueue_create_buffer[DType.int32](1)
    ctx.enqueue_function[xg_label_init_kernel](lab, Int32(n), grid_dim=_blocks(n), block_dim=TPB)
    while True:
        _clear1(ctx, chg)
        ctx.enqueue_function[xg_hook_kernel](pw, lab, Int32(n), chg, grid_dim=_blocks(n), block_dim=TPB)
        ctx.enqueue_function[xg_jump_kernel](lab, Int32(n), grid_dim=_blocks(n), block_dim=TPB)
        ctx.enqueue_function[xg_copy1_kernel](chg, dc.unsafe_ptr(), grid_dim=1, block_dim=1)
        ctx.enqueue_copy(dst_buf=hb, src_buf=dc)
        ctx.synchronize()
        if hb.unsafe_ptr().unsafe_load(0) == 0:
            break
    ctx.enqueue_function[xg_root_flag_kernel](lab, flag, Int32(n), grid_dim=_blocks(n), block_dim=TPB)
    _xscan(ctx, flag, rank, n, n + 1)
    ctx.enqueue_function[xg_comp_kernel](lab, rank, _ptr(_id(comp), n), Int32(n), grid_dim=_blocks(n), block_dim=TPB)
    var c = _read_i32(ctx, rank + n)
    pool_free(lid)
    pool_free(fid)
    pool_free(rid)
    pool_free(cid)
    _ = dc^
    _ = hb^
    return PythonObject(c)


def xg_clear1_kernel(p: I32Ptr):
    if _t() == 0:
        p.unsafe_store(0, Int32(0))


def _clear1(ctx: DeviceContext, p: I32Ptr) raises:
    ctx.enqueue_function[xg_clear1_kernel](p, grid_dim=1, block_dim=1)


def dev_graph_join_py(w: PythonObject, d: PythonObject, comp: PythonObject, p: PythonObject) raises -> PythonObject:
    """`graph_join_py` on resident W (in place), D and comp; p = [n, C]."""
    var n = _n(p, 0)
    var C = _n(p, 1)
    if C <= 1 or n == 0:
        return PythonObject(C)
    if n * C > 2147483647 or C * C > 2147483647:
        raise Error("x_decomp: too many connected components to join")
    var ctx = xd_ctx()
    var pw = _ptr(_id(w), n * n)
    var pd = _ptr(_id(d), n * n)
    var pc = _ptr(_id(comp), n)
    var vid = pool_alloc(n * C)
    var jid = pool_alloc(n * C)
    var kid = pool_alloc(C + 1)
    var sid = pool_alloc(C + 1)
    var mid = pool_alloc(n)
    var rv = _ptr(vid, n * C)
    var rj = _ptr(jid, n * C)
    var cnt = _ip(_ptr(kid, C + 1))
    var start = _ip(_ptr(sid, C + 1))
    var mem = _ip(_ptr(mid, n))
    ctx.enqueue_function[xg_rowbest_kernel](pd, pc, Int32(n), Int32(C), rv, rj, grid_dim=_blocks(n), block_dim=TPB)
    ctx.enqueue_function[xg_count_comp_kernel](pc, Int32(n), Int32(C), cnt, grid_dim=_blocks(C), block_dim=TPB)
    _xscan(ctx, cnt, start, C, C + 1)
    ctx.enqueue_function[xg_members_kernel](pc, Int32(n), Int32(C), start, mem, grid_dim=_blocks(C), block_dim=TPB)
    ctx.enqueue_function[xg_join_kernel](
        Int32(n), Int32(C), start, mem, rv, rj, pw, grid_dim=_blocks(C * C), block_dim=TPB
    )
    pool_free(vid)
    pool_free(jid)
    pool_free(kid)
    pool_free(sid)
    pool_free(mid)
    return PythonObject(C)


def dev_graph_dijkstra_py(w: PythonObject, dist: PythonObject, p: PythonObject) raises -> PythonObject:
    """dist (n x n, resident): the shortest paths of the resident W, the
    arcs compressed on the device (count, scan, fill) and the rows of
    `dijkstra_kernel`, DIJKSTRA_ROWS a launch."""
    var n = _n(p, 0)
    if n == 0:
        return PythonObject(0)
    if n * n > 2147483647:
        raise Error("x_decomp: graph exceeds the Int32 index bound")
    var ctx = xd_ctx()
    var pw = _ptr(_id(w), n * n)
    var pd = _ptr(_id(dist), n * n)
    var kid = pool_alloc(n)
    var rid = pool_alloc(n + 1)
    var cnt = _ip(_ptr(kid, n))
    var rp = _ip(_ptr(rid, n + 1))
    ctx.enqueue_function[xg_arc_count_kernel](pw, Int32(n), cnt, grid_dim=_blocks(n), block_dim=TPB)
    _xscan(ctx, cnt, rp, n, n + 1)
    var ne = _read_i32(ctx, rp + n)
    var ea = max(ne, 1)
    var aid = pool_alloc(ea)
    var xid = pool_alloc(ea)
    var yid = pool_alloc(ea)
    var adj = _ip(_ptr(aid, ea))
    var wa = _ptr(xid, ea)
    var wb = _ptr(yid, ea)
    ctx.enqueue_function[xg_arc_fill_kernel](pw, Int32(n), rp, adj, wa, wb, grid_dim=_blocks(n), block_dim=TPB)
    var chunk = max(1, min(n, DIJKSTRA_ROWS))
    var pid = pool_alloc(n * n)
    var hid = pool_alloc(chunk * n)
    var gid = pool_alloc(n)
    var r0 = 0
    while r0 < n:
        var rows = min(chunk, n - r0)
        ctx.enqueue_function[dijkstra_kernel](
            rp, adj, wa, wb, pd, _ip(_ptr(hid, chunk * n)), _ip(_ptr(pid, n * n)), _ptr(gid, n),
            Int32(n), Int32(r0), Int32(rows), grid_dim=_blocks(rows), block_dim=TPB,
        )
        r0 += rows
    pool_free(kid)
    pool_free(rid)
    pool_free(aid)
    pool_free(xid)
    pool_free(yid)
    pool_free(pid)
    pool_free(hid)
    pool_free(gid)
    return PythonObject(n)
