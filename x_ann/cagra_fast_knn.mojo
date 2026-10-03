# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CAGRA's k-NN graph, FAST on Apple, OPT-IN A/B arms (lane/apple-fast-gap-cagra,
2026-10-03; switches in x_ann/fast_env.mojo, cost model in
docs/apple-fast/notes/gap-cagra.md).

Cause: the Istella build (400,000 x 220, intermediate degree 64) spends
nearly all of its 21.2 s in the exact k-NN graph, `knn_tiled_bigd_kernel`:
one thread per row, one threadgroup-memory load, a subtract and a fused
multiply-add per (pair, feature), 3.5e13 of them.

  * `cg_dot_knn_kernel` (DOT): a 64 x 64 tile of rows by candidates, both
    sides staged 16 features at a time, 4 x 4 dot products per thread (8
    threadgroup loads per 16 multiply-adds, no subtract), then the distance
    |a|^2 + |b|^2 - 2 a.b clamped at 0 on rows centred by a sample mean `mu`
    (the centring shrinks the norms, so less cancellation), then row i's
    owner offers the tile's candidates in ascending j through
    `ts_knn_offer`, as every k-NN kernel here does. FAST rounding moves the
    distances; the neighbor order is (distance, index), so ties stay fixed.
  * `cg_ivfg_enqueue` (IVFG): cuVS builds CAGRA's intermediate graph from an
    IVF index (IVF-PQ + refine) or NN-descent, not by brute force. Here: a
    device Lloyd k-means (n / 384 lists, a stride sample of 64 rows per list,
    stride-row seeds, sums in sample order, no atomics), every row assigned
    to its nearest list, rows sorted by list (`fast_radix_sort_pairs_u32`,
    stable), each list's probe set = itself + its PROBES - 1 nearest lists by
    centroid; then each 64-row chunk of the sorted rows runs the DOT tile
    against every row of its lists' probe sets. Every sum has a fixed order:
    the same graph on every run. When some list's probe pool holds fewer
    than intermediate_graph_degree + 1 rows, the caller builds the exact
    graph instead (one int downloaded)."""

from std.gpu import block_idx, block_dim, thread_idx
from std.memory import stack_allocation
from std.atomic import Atomic
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from x_ann.tsne_core import F32P, I32P, ts_knn_beats, ts_knn_offer
from x_ann.io import download_i32
from core.fast_radix_sort import fast_radix_sort_pairs_u32, frs_counts_len

comptime U32P = MutPointer[UInt32, MutAnyOrigin]

#: the dot tile: rows x candidates per threadgroup, features per stage,
#: threads (x, y), cells per thread
comptime DT_TI = 64
comptime DT_TJ = 64
comptime DT_KC = 16
comptime DT_TX = 16
comptime DT_TY = 16
comptime DT_RI = DT_TI // DT_TY
comptime DT_RJ = DT_TJ // DT_TX
comptime DT_TPB = DT_TX * DT_TY
comptime DT_DS = DT_TJ + 1
comptime ETPB = 256

#: IVFG: rows per list (n / IVFG_ROWS lists), sample rows per list for the
#: Lloyd iterations, Lloyd iterations, rows of the centring sample, and the
#: smallest n it takes (below, the exact graph is cheap)
comptime IVFG_ROWS = 384
comptime IVFG_SAMPLE = 64
comptime IVFG_ITERS = 10
comptime MU_ROWS = 1024
comptime IVFG_MIN_N = 65536
comptime IVFG_MAX_LISTS = 4096


def _gid() -> Int:
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


@always_inline
def _dt_tile[
    oa: MutOrigin, ob: MutOrigin, od: MutOrigin
](
    q: F32P, i0: Int, ni: Int, x: F32P, j0: Int, nj: Int, d: Int, mu: F32P, qn: F32P, xn: F32P,
    tx: Int, ty: Int, tid: Int,
    a_s: MutPointer[Float32, oa, address_space=AddressSpace.SHARED],
    b_s: MutPointer[Float32, ob, address_space=AddressSpace.SHARED],
    dist_s: MutPointer[Float32, od, address_space=AddressSpace.SHARED],
):
    """dist_s[ii * DT_DS + jj] = |q_{i0+ii} - mu|^2 + |x_{j0+jj} - mu|^2 -
    2 (q - mu).(x - mu), clamped at 0, for ii < ni, jj < nj. Ends on a
    barrier. Padded features stage as 0 (a product of zeros adds nothing)."""
    var acc = InlineArray[Float32, DT_RI * DT_RJ](fill=Float32(0.0))
    var k0 = 0
    while k0 < d:
        comptime for qq in range(DT_KC * DT_TI // DT_TPB):
            var e = tid + qq * DT_TPB
            var ii = e // DT_KC
            var kk = e - ii * DT_KC
            var c = k0 + kk
            var v = Float32(0.0)
            if c < d and ii < ni:
                v = q.unsafe_load((i0 + ii) * d + c) - mu.unsafe_load(c)
            a_s[kk * DT_TI + ii] = v
        comptime for qq in range(DT_KC * DT_TJ // DT_TPB):
            var e = tid + qq * DT_TPB
            var jj = e // DT_KC
            var kk = e - jj * DT_KC
            var c = k0 + kk
            var v = Float32(0.0)
            if c < d and jj < nj:
                v = x.unsafe_load((j0 + jj) * d + c) - mu.unsafe_load(c)
            b_s[kk * DT_TJ + jj] = v
        barrier()
        comptime for kk in range(DT_KC):
            var av = InlineArray[Float32, DT_RI](fill=Float32(0.0))
            var bv = InlineArray[Float32, DT_RJ](fill=Float32(0.0))
            comptime for r in range(DT_RI):
                av[r] = a_s[kk * DT_TI + ty + r * DT_TY]
            comptime for c in range(DT_RJ):
                bv[c] = b_s[kk * DT_TJ + tx + c * DT_TX]
            comptime for r in range(DT_RI):
                comptime for c in range(DT_RJ):
                    acc[r * DT_RJ + c] = av[r] * bv[c] + acc[r * DT_RJ + c]
        barrier()
        k0 += DT_KC
    comptime for r in range(DT_RI):
        comptime for c in range(DT_RJ):
            var ii = ty + r * DT_TY
            var jj = tx + c * DT_TX
            var dv = Float32(0.0)
            if ii < ni and jj < nj:
                dv = qn.unsafe_load(i0 + ii) + xn.unsafe_load(j0 + jj) - Float32(2.0) * acc[r * DT_RJ + c]
                if not (dv > Float32(0.0)):
                    dv = Float32(0.0)
            dist_s[ii * DT_DS + jj] = dv
    barrier()


def cg_dot_knn_kernel(
    q: F32P, qn: F32P, nq: Int32, x: F32P, xn: F32P, nx: Int32, d: Int32, mu: F32P, k: Int32,
    skip_self: Int32, out_d: F32P, out_i: I32P,
):
    """Row i = 64 * block_idx.x + t (t < 64 owns it): its k nearest rows of x
    under (distance, index), ascending; with skip_self, x is q and row i
    skips itself. Grid ceil(nq / 64), block (16, 16)."""
    var tx = Int(thread_idx.x)
    var ty = Int(thread_idx.y)
    var tid = ty * DT_TX + tx
    var nqi = Int(nq)
    var nxi = Int(nx)
    var dd = Int(d)
    var kk = Int(k)
    var i0 = Int(block_idx.x) * DT_TI
    var ni = DT_TI if nqi - i0 > DT_TI else nqi - i0
    var a_s = stack_allocation[DT_KC * DT_TI, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var b_s = stack_allocation[DT_KC * DT_TJ, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var dist_s = stack_allocation[DT_TI * DT_DS, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var i = i0 + tid
    var owner = tid < ni
    var base = i * kk
    var filled = 0
    var ld = Float32(0.0)
    var li = 0
    var j0 = 0
    while j0 < nxi:
        var nj = DT_TJ if nxi - j0 > DT_TJ else nxi - j0
        _dt_tile(q, i0, ni, x, j0, nj, dd, mu, qn, xn, tx, ty, tid, a_s, b_s, dist_s)
        if owner:
            for r in range(nj):
                var j = j0 + r
                if skip_self != 0 and j == i:
                    continue
                var dv = dist_s[tid * DT_DS + r]
                if filled == kk and not ts_knn_beats(dv, j, ld, li):
                    continue
                filled = ts_knn_offer(dv, j, base, kk, filled, out_d, out_i)
                if filled == kk:
                    ld = out_d.unsafe_load(base + kk - 1)
                    li = Int(out_i.unsafe_load(base + kk - 1))
        barrier()
        j0 += DT_TJ


def cg_norms_kernel(x: F32P, n: Int32, d: Int32, mu: F32P, dst: F32P):
    """|x_i - mu|^2, one thread per row, features ascending."""
    var i = _gid()
    if i < Int(n):
        var dd = Int(d)
        var acc = Float32(0.0)
        for c in range(dd):
            var v = x.unsafe_load(i * dd + c) - mu.unsafe_load(c)
            acc = v * v + acc
        dst.unsafe_store(i, acc)


def cg_mean_kernel(x: F32P, n: Int32, d: Int32, m: Int32, mu: F32P):
    """mu[c] = the mean of feature c over the stride sample rows (s n) / m,
    s ascending; one thread per feature."""
    var c = _gid()
    var dd = Int(d)
    if c < dd:
        var nn = Int(n)
        var mm = Int(m)
        var acc = Float32(0.0)
        for s in range(mm):
            acc += x.unsafe_load(((s * nn) // mm) * dd + c)
        mu.unsafe_store(c, acc / Float32(mm))


def cg_gather_stride_kernel(src: F32P, n_src: Int32, d: Int32, dst: F32P, n_dst: Int32):
    """dst row r = src row (r n_src) / n_dst; one thread per word."""
    var e = _gid()
    var dd = Int(d)
    var nd = Int(n_dst)
    if e < nd * dd:
        var r = e // dd
        var c = e - r * dd
        dst.unsafe_store(e, src.unsafe_load(((r * Int(n_src)) // nd) * dd + c))


def cg_gather_perm_kernel(src: F32P, d: Int32, perm: U32P, dst: F32P, n: Int32):
    """dst row r = src row perm[r]; one thread per word."""
    var e = _gid()
    var dd = Int(d)
    if e < Int(n) * dd:
        var r = e // dd
        var c = e - r * dd
        dst.unsafe_store(e, src.unsafe_load(Int(perm[r]) * dd + c))


def cg_label_keys_kernel(labels: I32P, n: Int32, keys: U32P, vals: U32P):
    var e = _gid()
    if e < Int(n):
        keys[e] = labels.unsafe_load(e).cast[DType.uint32]()
        vals[e] = UInt32(e)


@always_inline
def _lower_bound(keys: U32P, size: Int, v: Int) -> Int:
    var lo = 0
    var hi = size
    while lo < hi:
        var mid = (lo + hi) // 2
        if Int(keys[mid]) < v:
            lo = mid + 1
        else:
            hi = mid
    return lo


def cg_lloyd_update_kernel(xs: F32P, d: Int32, skeys: U32P, svals: U32P, size: Int32, cent: F32P):
    """Centroid c = block_idx.x: the mean of its sample rows, summed in
    sample order (the stable sort's order); an empty list keeps its
    centroid."""
    var c = Int(block_idx.x)
    var t = Int(thread_idx.x)
    var dd = Int(d)
    var span = stack_allocation[2, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    if t == 0:
        var lo = _lower_bound(skeys, Int(size), c)
        var hi = _lower_bound(skeys, Int(size), c + 1)
        span[0] = Int32(lo)
        span[1] = Int32(hi)
    barrier()
    var lo = Int(span[0])
    var hi = Int(span[1])
    if hi > lo:
        var inv = Float32(1.0) / Float32(hi - lo)
        var f = t
        while f < dd:
            var acc = Float32(0.0)
            for m in range(lo, hi):
                acc += xs.unsafe_load(Int(svals[m]) * dd + f)
            cent.unsafe_store(c * dd + f, acc * inv)
            f += ETPB


def cg_list_span_kernel(skeys: U32P, n: Int32, nlist: Int32, start: I32P, size: I32P):
    var c = _gid()
    if c < Int(nlist):
        var lo = _lower_bound(skeys, Int(n), c)
        var hi = _lower_bound(skeys, Int(n), c + 1)
        start.unsafe_store(c, Int32(lo))
        size.unsafe_store(c, Int32(hi - lo))


def cg_probe_kernel(cnn_i: I32P, nlist: Int32, P: Int32, probes: I32P, size: I32P, need: Int32, flag: I32P):
    """List c's probe set: c, then its P - 1 nearest lists by centroid; a
    pool under `need` rows sets the flag."""
    var c = _gid()
    var pp = Int(P)
    if c < Int(nlist):
        probes.unsafe_store(c * pp, Int32(c))
        var pool = Int(size.unsafe_load(c))
        for t in range(pp - 1):
            var o = Int(cnn_i.unsafe_load(c * (pp - 1) + t))
            probes.unsafe_store(c * pp + 1 + t, Int32(o))
            pool += Int(size.unsafe_load(o))
        if pool < Int(need):
            _ = Atomic.fetch_add(flag, Int32(1))


def cg_ivfg_kernel(
    xs: F32P, xn: F32P, perm: U32P, skeys: U32P, start: I32P, size: I32P, probes: I32P, P: Int32,
    n: Int32, d: Int32, mu: F32P, k: Int32, out_d: F32P, out_i: I32P,
):
    """Sorted rows s0 = 64 * block_idx.x .. s0 + 63 (one or more lists):
    for each list c among them, every row of c's probe lists is a candidate
    of the chunk's rows in list c, offered under its dataset id (perm), the
    row itself skipped. Grid ceil(n / 64), block (16, 16)."""
    var tx = Int(thread_idx.x)
    var ty = Int(thread_idx.y)
    var tid = ty * DT_TX + tx
    var nn = Int(n)
    var dd = Int(d)
    var kk = Int(k)
    var pp = Int(P)
    var s0 = Int(block_idx.x) * DT_TI
    var ni = DT_TI if nn - s0 > DT_TI else nn - s0
    var a_s = stack_allocation[DT_KC * DT_TI, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var b_s = stack_allocation[DT_KC * DT_TJ, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var dist_s = stack_allocation[DT_TI * DT_DS, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var live = tid < ni
    var me = 0
    var my_list = -1
    if live:
        me = Int(perm[s0 + tid])
        my_list = Int(skeys[s0 + tid])
    var base = me * kk
    var filled = 0
    var ld = Float32(0.0)
    var li = 0
    var c_first = Int(skeys[s0])
    var c_last = Int(skeys[s0 + ni - 1])
    for c in range(c_first, c_last + 1):
        if Int(size.unsafe_load(c)) == 0:
            continue
        var mine = live and my_list == c
        for pi in range(pp):
            var p = Int(probes.unsafe_load(c * pp + pi))
            var p0 = Int(start.unsafe_load(p))
            var pend = p0 + Int(size.unsafe_load(p))
            var j0 = p0
            while j0 < pend:
                var nj = DT_TJ if pend - j0 > DT_TJ else pend - j0
                _dt_tile(xs, s0, ni, xs, j0, nj, dd, mu, xn, xn, tx, ty, tid, a_s, b_s, dist_s)
                if mine:
                    for r in range(nj):
                        var j = Int(perm[j0 + r])
                        if j == me:
                            continue
                        var dv = dist_s[tid * DT_DS + r]
                        if filled == kk and not ts_knn_beats(dv, j, ld, li):
                            continue
                        filled = ts_knn_offer(dv, j, base, kk, filled, out_d, out_i)
                        if filled == kk:
                            ld = out_d.unsafe_load(base + kk - 1)
                            li = Int(out_i.unsafe_load(base + kk - 1))
                barrier()
                j0 += DT_TJ


def _f32p(mut b: DeviceBuffer[DType.float32]) -> F32P:
    return b.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()


def _i32p(mut b: DeviceBuffer[DType.int32]) -> I32P:
    return b.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()


def _u32p(mut b: DeviceBuffer[DType.uint32]) -> U32P:
    return b.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()


def _dot_knn(
    ctx: DeviceContext, q: F32P, qn: F32P, nq: Int, x: F32P, xn: F32P, nx: Int, d: Int, mu: F32P, k: Int,
    skip_self: Bool, out_d: F32P, out_i: I32P,
) raises:
    ctx.enqueue_function[cg_dot_knn_kernel](
        q, qn, Int32(nq), x, xn, Int32(nx), Int32(d), mu, Int32(k), Int32(1) if skip_self else Int32(0),
        out_d, out_i, grid_dim=(nq + DT_TI - 1) // DT_TI, block_dim=(DT_TX, DT_TY, 1),
    )


def _norms(ctx: DeviceContext, x: F32P, n: Int, d: Int, mu: F32P, dst: F32P) raises:
    ctx.enqueue_function[cg_norms_kernel](x, Int32(n), Int32(d), mu, dst, grid_dim=(n + ETPB - 1) // ETPB,
                                          block_dim=ETPB)


def _sort_labels(
    ctx: DeviceContext, labels: I32P, n: Int, mut keys: DeviceBuffer[DType.uint32],
    mut vals: DeviceBuffer[DType.uint32], mut tk: DeviceBuffer[DType.uint32], mut tv: DeviceBuffer[DType.uint32],
    mut counts: DeviceBuffer[DType.int32],
) raises:
    ctx.enqueue_function[cg_label_keys_kernel](labels, Int32(n), _u32p(keys), _u32p(vals),
                                               grid_dim=(n + ETPB - 1) // ETPB, block_dim=ETPB)
    fast_radix_sort_pairs_u32(ctx, n, keys, vals, tk, tv, counts)


def cg_mu_enqueue(ctx: DeviceContext, x: F32P, n: Int, d: Int, mut mu: DeviceBuffer[DType.float32]) raises:
    var m = MU_ROWS if n > MU_ROWS else n
    ctx.enqueue_function[cg_mean_kernel](x, Int32(n), Int32(d), Int32(m), _f32p(mu),
                                         grid_dim=(d + 63) // 64, block_dim=64)


def cg_dot_knn_enqueue(
    ctx: DeviceContext, mut dx: DeviceBuffer[DType.float32], n: Int, d: Int, k: Int,
    mut dnd: DeviceBuffer[DType.float32], mut dni: DeviceBuffer[DType.int32],
) raises:
    """DOT: the exact k-NN graph (each row's k nearest other rows), dot tile."""
    var mu = ctx.enqueue_create_buffer[DType.float32](d)
    var xn = ctx.enqueue_create_buffer[DType.float32](n)
    var x = _f32p(dx)
    cg_mu_enqueue(ctx, x, n, d, mu)
    _norms(ctx, x, n, d, _f32p(mu), _f32p(xn))
    _dot_knn(ctx, x, _f32p(xn), n, x, _f32p(xn), n, d, _f32p(mu), k, True, _f32p(dnd), _i32p(dni))
    ctx.synchronize()
    _ = xn^
    _ = mu^


def cg_ivfg_enqueue[
    PROBES: Int
](
    ctx: DeviceContext, mut dx: DeviceBuffer[DType.float32], n: Int, d: Int, k: Int,
    mut dnd: DeviceBuffer[DType.float32], mut dni: DeviceBuffer[DType.int32],
) raises -> Bool:
    """IVFG: the approximate k-NN graph into dnd / dni. False (nothing
    written) when n is under IVFG_MIN_N or a probe pool is too small: the
    caller then builds the exact graph."""
    if n < IVFG_MIN_N:
        return False
    var nlist = n // IVFG_ROWS
    if nlist > IVFG_MAX_LISTS:
        nlist = IVFG_MAX_LISTS
    if nlist < PROBES:
        return False
    var ns = nlist * IVFG_SAMPLE
    if ns > n:
        ns = n
    var x = _f32p(dx)
    var mu = ctx.enqueue_create_buffer[DType.float32](d)
    cg_mu_enqueue(ctx, x, n, d, mu)
    # the training sample and the seeds: stride rows
    var xsamp = ctx.enqueue_create_buffer[DType.float32](ns * d)
    var sn = ctx.enqueue_create_buffer[DType.float32](ns)
    var cent = ctx.enqueue_create_buffer[DType.float32](nlist * d)
    var cn = ctx.enqueue_create_buffer[DType.float32](nlist)
    ctx.enqueue_function[cg_gather_stride_kernel](x, Int32(n), Int32(d), _f32p(xsamp), Int32(ns),
                                                  grid_dim=(ns * d + ETPB - 1) // ETPB, block_dim=ETPB)
    ctx.enqueue_function[cg_gather_stride_kernel](x, Int32(n), Int32(d), _f32p(cent), Int32(nlist),
                                                  grid_dim=(nlist * d + ETPB - 1) // ETPB, block_dim=ETPB)
    _norms(ctx, _f32p(xsamp), ns, d, _f32p(mu), _f32p(sn))
    var big = n if n > ns else ns
    var lab_d = ctx.enqueue_create_buffer[DType.float32](big)
    var lab_i = ctx.enqueue_create_buffer[DType.int32](big)
    var keys = ctx.enqueue_create_buffer[DType.uint32](big)
    var vals = ctx.enqueue_create_buffer[DType.uint32](big)
    var tk = ctx.enqueue_create_buffer[DType.uint32](big)
    var tv = ctx.enqueue_create_buffer[DType.uint32](big)
    var counts = ctx.enqueue_create_buffer[DType.int32](frs_counts_len(big))
    for _ in range(IVFG_ITERS):
        _norms(ctx, _f32p(cent), nlist, d, _f32p(mu), _f32p(cn))
        _dot_knn(ctx, _f32p(xsamp), _f32p(sn), ns, _f32p(cent), _f32p(cn), nlist, d, _f32p(mu), 1, False,
                 _f32p(lab_d), _i32p(lab_i))
        _sort_labels(ctx, _i32p(lab_i), ns, keys, vals, tk, tv, counts)
        ctx.enqueue_function[cg_lloyd_update_kernel](_f32p(xsamp), Int32(d), _u32p(keys), _u32p(vals), Int32(ns),
                                                     _f32p(cent), grid_dim=nlist, block_dim=ETPB)
    # every row to its list, the rows sorted by list
    _norms(ctx, _f32p(cent), nlist, d, _f32p(mu), _f32p(cn))
    var xn = ctx.enqueue_create_buffer[DType.float32](n)
    _norms(ctx, x, n, d, _f32p(mu), _f32p(xn))
    _dot_knn(ctx, x, _f32p(xn), n, _f32p(cent), _f32p(cn), nlist, d, _f32p(mu), 1, False, _f32p(lab_d),
             _i32p(lab_i))
    _sort_labels(ctx, _i32p(lab_i), n, keys, vals, tk, tv, counts)
    var start = ctx.enqueue_create_buffer[DType.int32](nlist)
    var size = ctx.enqueue_create_buffer[DType.int32](nlist)
    ctx.enqueue_function[cg_list_span_kernel](_u32p(keys), Int32(n), Int32(nlist), _i32p(start), _i32p(size),
                                              grid_dim=(nlist + ETPB - 1) // ETPB, block_dim=ETPB)
    # probe sets: each list and its PROBES - 1 nearest lists by centroid
    var cnd = ctx.enqueue_create_buffer[DType.float32](nlist * (PROBES - 1))
    var cni = ctx.enqueue_create_buffer[DType.int32](nlist * (PROBES - 1))
    _dot_knn(ctx, _f32p(cent), _f32p(cn), nlist, _f32p(cent), _f32p(cn), nlist, d, _f32p(mu), PROBES - 1, True,
             _f32p(cnd), _i32p(cni))
    var probes = ctx.enqueue_create_buffer[DType.int32](nlist * PROBES)
    var flag = ctx.enqueue_create_buffer[DType.int32](1)
    flag.enqueue_fill(Int32(0))
    ctx.enqueue_function[cg_probe_kernel](_i32p(cni), Int32(nlist), Int32(PROBES), _i32p(probes), _i32p(size),
                                          Int32(k + 1), _i32p(flag), grid_dim=(nlist + ETPB - 1) // ETPB,
                                          block_dim=ETPB)
    var short = download_i32(ctx, flag, 1)
    var ok = short[0] == 0
    if ok:
        # the rows in list order, then the graph
        var xs = ctx.enqueue_create_buffer[DType.float32](n * d)
        var xsn = ctx.enqueue_create_buffer[DType.float32](n)
        ctx.enqueue_function[cg_gather_perm_kernel](x, Int32(d), _u32p(vals), _f32p(xs), Int32(n),
                                                    grid_dim=(n * d + ETPB - 1) // ETPB, block_dim=ETPB)
        _norms(ctx, _f32p(xs), n, d, _f32p(mu), _f32p(xsn))
        ctx.enqueue_function[cg_ivfg_kernel](
            _f32p(xs), _f32p(xsn), _u32p(vals), _u32p(keys), _i32p(start), _i32p(size), _i32p(probes),
            Int32(PROBES), Int32(n), Int32(d), _f32p(mu), Int32(k), _f32p(dnd), _i32p(dni),
            grid_dim=(n + DT_TI - 1) // DT_TI, block_dim=(DT_TX, DT_TY, 1),
        )
        ctx.synchronize()
        _ = xsn^
        _ = xs^
    _ = flag^
    _ = probes^
    _ = cni^
    _ = cnd^
    _ = size^
    _ = start^
    _ = xn^
    _ = counts^
    _ = tv^
    _ = tk^
    _ = vals^
    _ = keys^
    _ = lab_i^
    _ = lab_d^
    _ = cn^
    _ = cent^
    _ = sn^
    _ = xsamp^
    _ = mu^
    return ok
