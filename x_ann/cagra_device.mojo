# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CAGRA on the device: the exact k-NN graph and the search, one thread per
cell of `x_ann/cagra_core.mojo` / `x_ann/tsne_core.mojo`; pruning and the
reverse-edge merge are the shared host functions."""

from std.gpu import block_idx, block_dim, thread_idx
from std.memory import stack_allocation
from std.atomic import Atomic
from std.sys.compile import is_defined
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from x_ann.device_ctx import x_ann_ctx
from x_ann.stage_timer import AnnStages
from x_ann.knn_device import knn_enqueue

from x_ann.io import upload_f32, upload_i32, download_f32, download_i32
from x_ann.cagra_core import F32P, I32P, cagra_prune, cagra_reverse_merge, cg_dist, cg_search_cell
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from std.sys.info import has_apple_gpu_accelerator
from x_ann.switches import ANN3_CAGRA_TEAM

comptime TPB = 64


def _tid() -> Int:
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


def _grid(count: Int) -> Int:
    return (count + TPB - 1) // TPB


def cg_search_kernel(
    m: Int32, queries: F32P, x: F32P, n: Int32, d: Int32, graph: I32P, deg: Int32, k: Int32,
    L: Int32, width: Int32, max_iter: Int32, n_seeds: Int32, bd: F32P, bi: I32P, bx: I32P,
    visited: I32P, words: Int32, out_d: F32P, out_i: I32P,
):
    var q = _tid()
    if q < Int(m):
        cg_search_cell(q, queries, x, Int(n), Int(d), graph, Int(deg), Int(k), Int(L), Int(width),
                       Int(max_iter), Int(n_seeds), bd, bi, bx, visited, Int(words), out_d, out_i)


#: the team search (lane ann-apple3): threads per query, the longest itopk
#: list and the most candidates of one step held in threadgroup memory
comptime CG_T = 32
comptime CG_LMAX = 128
comptime CG_CMAX = 128
comptime CAGRA_TEAM = ANN3_CAGRA_TEAM and GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()


def cg_search_team_kernel(
    queries: F32P, x: F32P, n: Int32, d: Int32, graph: I32P, deg: Int32, k: Int32, L: Int32, width: Int32,
    max_iter: Int32, n_seeds: Int32, visited: I32P, words: Int32, out_d: F32P, out_i: I32P,
):
    """FAST on Apple, OPT-IN (lane ann-apple3): `cg_search_row` for query
    block_idx.x by a threadgroup of CG_T threads. THREAD 0 WALKS, as the cell
    does: the seeds in order, then per iteration the `search_width` best
    unexpanded entries front to back, every candidate through the visited
    bitset and the sorted insertion under (distance, id), in the cell's
    order. THE OTHER THREADS ONLY FORM DISTANCES: for each step (a chunk of
    seeds or one parent's graph row) thread t computes `cg_dist` of
    candidates t, t + CG_T, ... into threadgroup memory before thread 0
    inserts them. A distance is a function of the query and the row alone,
    so the list is the cell's list. The itopk list, the step's candidates
    and the step's control words live in threadgroup memory (`barrier()`
    orders it); the visited bitset is device memory that thread 0 alone
    reads and writes. Needs L <= CG_LMAX and deg <= CG_CMAX (the caller
    checks)."""
    var qi = Int(block_idx.x)
    var t = Int(thread_idx.x)
    var nn = Int(n)
    var dd = Int(d)
    var dg = Int(deg)
    var ll = Int(L)
    var ns = Int(n_seeds)
    var kk = Int(k)
    var sbd = stack_allocation[CG_LMAX, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var sbi = stack_allocation[CG_LMAX, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var sbx = stack_allocation[CG_LMAX, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var cd = stack_allocation[CG_CMAX, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var cv = stack_allocation[CG_CMAX, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var ctl = stack_allocation[2, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var q_off = qi * dd
    var vbase = qi * Int(words)
    if t == 0:
        for s in range(ll):
            sbd[s] = Float32(0.0)
            sbi[s] = Int32(-1)
            sbx[s] = Int32(0)
        for w in range(Int(words)):
            visited.unsafe_store(vbase + w, Int32(0))
    # thread 0's walk state
    var seed_at = 0
    var it = 0
    var scan = 0
    var taken = 0
    while True:
        if t == 0:
            # the next step: a chunk of seeds (-2 - its first seed), or the
            # next parent of the cell's scan, or the end
            var pick = -1
            var fin = 0
            if seed_at < ns:
                pick = -2 - seed_at
                seed_at += CG_CMAX
            else:
                while True:
                    if it >= Int(max_iter):
                        fin = 1
                        break
                    while scan < ll and taken < Int(width):
                        var cand = Int(sbi[scan])
                        if cand >= 0 and sbx[scan] == 0:
                            sbx[scan] = Int32(1)
                            taken += 1
                            pick = cand
                            scan += 1
                            break
                        scan += 1
                    if pick >= 0:
                        break
                    if taken == 0:
                        fin = 1
                        break
                    it += 1
                    scan = 0
                    taken = 0
            ctl[0] = Int32(pick)
            ctl[1] = Int32(fin)
        barrier()
        if ctl[1] != 0:
            break
        var code = Int(ctl[0])
        var cnt = dg
        var c0 = 0
        if code < 0:
            c0 = -2 - code
            cnt = ns - c0
            if cnt > CG_CMAX:
                cnt = CG_CMAX
        for e in range(t, cnt, CG_T):
            var v = 0
            if code >= 0:
                v = Int(graph.unsafe_load(code * dg + e))
            else:
                v = ((c0 + e) * nn) // ns
            cv[e] = Int32(v)
            cd[e] = cg_dist(queries, q_off, x, v, dd)
        barrier()
        if t == 0:
            for e in range(cnt):
                var v = Int(cv[e])
                var word = visited.unsafe_load(vbase + v // 32)
                var bit = Int32(1) << Int32(v % 32)
                if (word & bit) != 0:
                    continue
                visited.unsafe_store(vbase + v // 32, word | bit)
                # `cg_insert` on the threadgroup list
                var dist = cd[e]
                var id = Int32(v)
                var last_i = sbi[ll - 1]
                if last_i >= 0:
                    var last_d = sbd[ll - 1]
                    if not (dist < last_d or (dist == last_d and id < last_i)):
                        continue
                var s = ll - 1
                while s > 0:
                    var pi = sbi[s - 1]
                    var pd = sbd[s - 1]
                    if pi < 0 or dist < pd or (dist == pd and id < pi):
                        sbd[s] = pd
                        sbi[s] = pi
                        sbx[s] = sbx[s - 1]
                        s -= 1
                    else:
                        break
                sbd[s] = dist
                sbi[s] = id
                sbx[s] = Int32(0)
    if t == 0:
        for s in range(kk):
            out_d.unsafe_store(qi * kk + s, sbd[s])
            out_i.unsafe_store(qi * kk + s, sbi[s])


#: the device prune's widest k-NN row (one thread per entry)
comptime PRUNE_KMAX = 64


def prune_kernel(n: Int32, kdeg: Int32, deg: Int32, knn: I32P, pruned: I32P, bad: I32P):
    """`cagra_prune` for node a = block_idx.x, one thread per k-NN rank
    (lane ann-apple2). For a row of kdeg DISTINCT ids in [0, n) the host's
    detour count of rank kab is the number of (kad < kab, kdb) with
    knn[knn[a, kad], kdb] == knn[a, kab] (plus kdeg when that id is a
    itself), and its selection takes the deg least (count, rank) in order,
    so rank kab lands at output slot #{k : (cnt[k], k) < (cnt[kab], kab)}.
    Integer counts: the same integers, the same graph. A row that is not
    distinct is flagged (bad[a] = 1) and the host prunes the graph itself."""
    var a = Int(block_idx.x)
    var t = Int(thread_idx.x)
    var k = Int(kdeg)
    var nr = Int(n)
    var ids = stack_allocation[PRUNE_KMAX, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var cnt = stack_allocation[PRUNE_KMAX, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var flag = stack_allocation[1, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    if t == 0:
        flag[0] = 0
    if t < k:
        ids[t] = knn.unsafe_load(a * k + t)
    barrier()
    if t < k:
        var v = Int(ids[t])
        var dup = v < 0 or v >= nr
        for u in range(t):
            if Int(ids[u]) == v:
                dup = True
        if dup:
            flag[0] = 1
    barrier()
    var is_bad = flag[0] != 0
    if is_bad:
        if t == 0:
            bad.unsafe_store(a, Int32(1))
        return
    # lane ann-apple2: row a's ids sorted (distinct: position = #smaller)
    # with their ranks; thread kad walks row knn[a, kad] once, finds each
    # candidate's rank kab by binary search and adds 1 to cnt[kab] when kab >
    # kad (threadgroup integer atomics: the sum of the same ones in any
    # order is the same integer)
    var sid = stack_allocation[PRUNE_KMAX, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var srk = stack_allocation[PRUNE_KMAX, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    if t < k:
        var mine_id = ids[t]
        var pos = 0
        for u in range(k):
            if ids[u] < mine_id:
                pos += 1
        sid[pos] = mine_id
        srk[pos] = Int32(t)
        cnt[t] = Int32(k) if Int(mine_id) == a else Int32(0)
    barrier()
    if t < k - 1:
        var d = Int(ids[t])
        for kdb in range(k):
            var cand = knn.unsafe_load(d * k + kdb)
            var lo = 0
            var hi = k
            while lo < hi:
                var mid = (lo + hi) // 2
                if sid[mid] < cand:
                    lo = mid + 1
                else:
                    hi = mid
            if lo < k and sid[lo] == cand:
                var kab = Int(srk[lo])
                if kab > t:
                    _ = Atomic.fetch_add(cnt + kab, Int32(1))
    barrier()
    if t < k:
        var mine = cnt[t]
        var rank = 0
        for u in range(k):
            var o = cnt[u]
            if o < mine or (o == mine and u < t):
                rank += 1
        if rank < Int(deg):
            pruned.unsafe_store(a * Int(deg) + rank, ids[t])
    if t == 0:
        bad.unsafe_store(a, Int32(0))


def cagra_build_device(x: List[Float32], n: Int, d: Int, kdeg: Int, deg: Int) raises -> List[Int32]:
    var st = AnnStages("cagra_build")
    var ctx = x_ann_ctx()
    var dx = upload_f32(ctx, x)
    st.mark(ctx, "upload")
    var dnd = ctx.enqueue_create_buffer[DType.float32](n * kdeg)
    var dni = ctx.enqueue_create_buffer[DType.int32](n * kdeg)
    knn_enqueue(ctx, dx, n, d, kdeg, dnd, dni)
    var pruned = List[Int32]()
    var on_device = kdeg <= PRUNE_KMAX and deg <= kdeg and not is_defined["MOJOLEARN_CAGRA_HOST_PRUNE"]()
    if on_device:
        var dout = ctx.enqueue_create_buffer[DType.int32](n * deg)
        var dbad = ctx.enqueue_create_buffer[DType.int32](n)
        ctx.enqueue_function[prune_kernel](Int32(n), Int32(kdeg), Int32(deg), dni.unsafe_ptr(), dout.unsafe_ptr(),
                                           dbad.unsafe_ptr(), grid_dim=n, block_dim=PRUNE_KMAX)
        ctx.synchronize()
        st.host("knn_prune")
        var bad = download_i32(ctx, dbad, n)
        for a in range(n):
            if bad[a] != 0:
                on_device = False
                break
        if on_device:
            pruned = download_i32(ctx, dout, n * deg)
        _ = dbad^
        _ = dout^
    if not on_device:
        ctx.synchronize()
        st.host("knn")
        var knn = download_i32(ctx, dni, n * kdeg)
        st.host("download")
        pruned = cagra_prune(n, kdeg, knn, deg)
        st.host("prune")
    _ = dni^
    _ = dnd^
    _ = dx^
    _ = ctx^
    var merged = cagra_reverse_merge(n, deg, pruned)
    st.host("reverse_merge")
    return merged^


def cagra_search_device(
    x: List[Float32], n: Int, d: Int, graph: List[Int32], deg: Int, queries: List[Float32], m: Int,
    k: Int, L: Int, width: Int, max_iter: Int, n_seeds: Int,
    mut out_d: List[Float32], mut out_i: List[Int32],
) raises:
    var st = AnnStages("cagra_search")
    var ctx = x_ann_ctx()
    var dx = upload_f32(ctx, x)
    var dg = upload_i32(ctx, graph)
    st.mark(ctx, "upload")
    cagra_search_on(
        ctx, dx.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), n, d,
        dg.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), deg, queries, m, k, L, width, max_iter,
        n_seeds, out_d, out_i,
    )
    _ = dg^
    _ = dx^
    _ = ctx^


def cagra_search_on(
    ctx: DeviceContext, dx: F32P, n: Int, d: Int, dg: I32P, deg: Int, queries: List[Float32], m: Int,
    k: Int, L: Int, width: Int, max_iter: Int, n_seeds: Int,
    mut out_d: List[Float32], mut out_i: List[Int32],
) raises:
    """The search over a dataset and graph already on the device
    (`cagra_search_device` uploads them first; `x_ann/resident.mojo` holds
    them): the queries up, the walk, the two outputs down."""
    var words = (n + 31) // 32
    var st = AnnStages("cagra_search")
    var dq = upload_f32(ctx, queries)
    var bd = ctx.enqueue_create_buffer[DType.float32](m * L)
    var bi = ctx.enqueue_create_buffer[DType.int32](m * L)
    var bx = ctx.enqueue_create_buffer[DType.int32](m * L)
    var vis = ctx.enqueue_create_buffer[DType.int32](m * words)
    var od = ctx.enqueue_create_buffer[DType.float32](m * k)
    var oi = ctx.enqueue_create_buffer[DType.int32](m * k)
    # lane ann-apple3, FAST on Apple, OPT-IN: a threadgroup per query
    var team = False
    comptime if CAGRA_TEAM:
        if L <= CG_LMAX and deg <= CG_CMAX:
            team = True
            ctx.enqueue_function[cg_search_team_kernel](
                dq.unsafe_ptr(), dx, Int32(n), Int32(d), dg, Int32(deg), Int32(k), Int32(L), Int32(width),
                Int32(max_iter), Int32(n_seeds), vis.unsafe_ptr(), Int32(words), od.unsafe_ptr(),
                oi.unsafe_ptr(), grid_dim=m, block_dim=CG_T,
            )
    if not team:
        ctx.enqueue_function[cg_search_kernel](
            Int32(m), dq.unsafe_ptr(), dx, Int32(n), Int32(d), dg, Int32(deg), Int32(k),
            Int32(L), Int32(width), Int32(max_iter), Int32(n_seeds), bd.unsafe_ptr(), bi.unsafe_ptr(),
            bx.unsafe_ptr(), vis.unsafe_ptr(), Int32(words), od.unsafe_ptr(), oi.unsafe_ptr(),
            grid_dim=_grid(m), block_dim=TPB,
        )
    ctx.synchronize()
    st.host("search")
    out_d = download_f32(ctx, od, m * k)
    out_i = download_i32(ctx, oi, m * k)
    st.host("download")
    _ = oi^
    _ = od^
    _ = vis^
    _ = bx^
    _ = bi^
    _ = bd^
    _ = dq^
