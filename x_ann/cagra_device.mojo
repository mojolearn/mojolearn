# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CAGRA on the device: the exact k-NN graph, the prune, the reverse-edge
merge and the search. The search is one thread per cell of
`x_ann/cagra_core.mojo`; the prune and the merge are one block per node and
compute `cagra_prune` / `cagra_reverse_merge` (the host column's functions)
integer for integer (gap-fails2, 2026-10-02: the opt-in host prune and its
64-entry bound are gone)."""

from std.gpu import block_idx, block_dim, thread_idx
from std.memory import stack_allocation
from std.atomic import Atomic
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from x_ann.device_ctx import x_ann_ctx
from x_ann.stage_timer import AnnStages
from x_ann.knn_device import knn_enqueue

from x_ann.io import upload_f32, upload_i32, download_f32, download_i32
from x_ann.cagra_core import F32P, I32P, cg_dist, cg_search_cell, cg_seed_node
from core.fast_radix_sort import fast_radix_sort_pairs_u32, frs_counts_len
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from std.sys.info import has_apple_gpu_accelerator
from x_ann.switches import ANN3_CAGRA_TEAM
from x_ann.fast_env import FAST_CAGRA_TEAM

comptime TPB = 64


def _tid() -> Int:
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


def _grid(count: Int) -> Int:
    return (count + TPB - 1) // TPB


def cg_search_kernel(
    m: Int32, queries: F32P, x: F32P, n: Int32, d: Int32, graph: I32P, deg: Int32, k: Int32,
    L: Int32, width: Int32, max_iter: Int32, n_seeds: Int32, bd: F32P, bi: I32P, bx: I32P,
    visited: I32P, words: Int32, out_d: F32P, out_i: I32P, rs: Int32,
):
    var q = _tid()
    if q < Int(m):
        cg_search_cell(q, queries, x, Int(n), Int(d), graph, Int(deg), Int(k), Int(L), Int(width),
                       Int(max_iter), Int(n_seeds), bd, bi, bx, visited, Int(words), out_d, out_i, Int(rs))


#: the team search (lane ann-apple3): threads per query, the longest itopk
#: list and the most candidates of one step held in threadgroup memory
comptime CG_T = 32
comptime CG_LMAX = 128
comptime CG_CMAX = 128
#: the team search compiles under FAST on Apple; it runs when the ann-apple3
#: define or the env switch below asks (lane/apple-fast-ann)
comptime CAGRA_TEAM = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()


def cg_search_team_kernel(
    queries: F32P, x: F32P, n: Int32, d: Int32, graph: I32P, deg: Int32, k: Int32, L: Int32, width: Int32,
    max_iter: Int32, n_seeds: Int32, visited: I32P, words: Int32, out_d: F32P, out_i: I32P, rs: Int32,
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
                v = cg_seed_node(c0 + e, nn, ns, Int(rs))
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


#: The device prune and merge (gap-fails2, 2026-10-02): one block per node,
#: threads striding over the row, the row's composite keys sorted block-wide
#: in threadgroup memory. The widest row is the comptime capacity of the
#: launched specialization (64, 128, 256, 512 or 1024 entries, the smallest
#: that holds intermediate_graph_degree), so no degree routes to the host.
comptime PRUNE_KMAX = 1024
comptime PRUNE_TPB = 256
comptime U64_MAX = ~UInt64(0)


@always_inline
def _u32_bits(v: Int32) -> UInt64:
    """`v`'s 32 bits, zero-extended (masked: index_map.mojo::key_lo's
    sign-extension note)."""
    return v.cast[DType.uint32]().cast[DType.uint64]() & UInt64(0xFFFFFFFF)


@always_inline
def _pow2_at_least(k: Int) -> Int:
    var p = 1
    while p < k:
        p *= 2
    return p


@always_inline
def _bitonic_u64[
    o: MutOrigin
](keys: MutPointer[UInt64, o, address_space=AddressSpace.SHARED], P: Int, t: Int, tpb: Int):
    """Ascending bitonic sort of `keys[0:P]` (P a power of two) by the whole
    block. Integer compares and swaps only: one answer on every device."""
    var size = 2
    while size <= P:
        var stride = size // 2
        while stride > 0:
            var i = t
            while i < P:
                var j = i ^ stride
                if j > i:
                    var ka = keys[i]
                    var kb = keys[j]
                    var up = (i & size) == 0
                    if (ka > kb) == up:
                        keys[i] = kb
                        keys[j] = ka
                i += tpb
            barrier()
            stride //= 2
        size *= 2


def prune_kernel[KMAX: Int](n: Int32, kdeg: Int32, deg: Int32, knn: I32P, pruned: I32P, short: I32P):
    """`cagra_prune` (x_ann/cagra_core.mojo) for node a = block_idx.x, every
    row (distinct or not) on the device.

    1. The row's entries as composite keys (id as UInt32 << 32 | rank),
       sorted block-wide: equal ids are adjacent, ranks ascending.
    2. Detour counts: every (kad, kdb) pair with kad < kdeg - 1 is one
       thread step; cand = knn[knn[a, kad], kdb] lands on the FIRST rank
       kab > kad holding cand (a lower bound for (cand, kad + 1) in the
       sorted keys), which is the host's distinct-row `rank[cand] > kad` and
       its repeat-row `for kab in kad + 1 ..: if == cand: break` alike.
       Integer atomics: the same counts in any order. A walked row d outside
       [0, n) is skipped (the host skips it too).
    3. Selection: the host takes, deg times, the least eligible (cnt, rank)
       (cnt < 0xFFFF) and retires every entry with that id, so each id
       enters at its least eligible (cnt, rank) and ids come out in that
       order. Each id's representative carries (cnt << 32 | rank); a
       block-wide sort orders them; the first deg are the row. Fewer than
       deg eligible ids sets `short[0]` (the host's refusal)."""
    var a = Int(block_idx.x)
    var t = Int(thread_idx.x)
    var k = Int(kdeg)
    var nr = Int(n)
    var dg = Int(deg)
    var P = _pow2_at_least(k)
    var ids = stack_allocation[KMAX, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var cnt = stack_allocation[KMAX, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var keys = stack_allocation[KMAX, Scalar[DType.uint64], address_space=AddressSpace.SHARED]()
    var e = t
    while e < P:
        if e < k:
            var v = knn.unsafe_load(a * k + e)
            ids[e] = v
            keys[e] = (_u32_bits(v) << UInt64(32)) | UInt64(e)
            cnt[e] = Int32(k) if Int(v) == a else Int32(0)
        else:
            keys[e] = U64_MAX
        e += PRUNE_TPB
    barrier()
    _bitonic_u64(keys, P, t, PRUNE_TPB)
    var pairs = (k - 1) * k
    var pidx = t
    while pidx < pairs:
        var kad = pidx // k
        var kdb = pidx - kad * k
        var d = Int(ids[kad])
        if d >= 0 and d < nr:
            var cand = _u32_bits(knn.unsafe_load(d * k + kdb))
            var target = (cand << UInt64(32)) | UInt64(kad + 1)
            var lo = 0
            var hi = P
            while lo < hi:
                var mid = (lo + hi) // 2
                if keys[mid] < target:
                    lo = mid + 1
                else:
                    hi = mid
            if lo < k:
                var hit = keys[lo]
                if (hit >> UInt64(32)) == cand:
                    _ = Atomic.fetch_add(cnt + Int(hit & UInt64(0xFFFFFFFF)), Int32(1))
        pidx += PRUNE_TPB
    barrier()
    # representatives: sorted position p holds rank r; r represents its id
    # when eligible and least (cnt, rank) among its id's eligible entries
    comptime PER = KMAX // PRUNE_TPB if KMAX >= PRUNE_TPB else 1
    var sel = InlineArray[UInt64, PER](fill=U64_MAX)
    for j in range(PER):
        var pos = t + j * PRUNE_TPB
        if pos < k:
            var me = keys[pos]
            var r = Int(me & UInt64(0xFFFFFFFF))
            var mc = cnt[r]
            if mc < Int32(0xFFFF):
                var idb = me >> UInt64(32)
                var rep = True
                var q = pos - 1
                while q >= 0 and (keys[q] >> UInt64(32)) == idb:
                    var r2 = Int(keys[q] & UInt64(0xFFFFFFFF))
                    var c2 = cnt[r2]
                    if c2 < Int32(0xFFFF) and (c2 < mc or (c2 == mc and r2 < r)):
                        rep = False
                    q -= 1
                q = pos + 1
                while q < k and (keys[q] >> UInt64(32)) == idb:
                    var r2 = Int(keys[q] & UInt64(0xFFFFFFFF))
                    var c2 = cnt[r2]
                    if c2 < Int32(0xFFFF) and (c2 < mc or (c2 == mc and r2 < r)):
                        rep = False
                    q += 1
                if rep:
                    sel[j] = (UInt64(Int(mc)) << UInt64(32)) | UInt64(r)
    barrier()
    for j in range(PER):
        var pos = t + j * PRUNE_TPB
        if pos < P:
            keys[pos] = sel[j]
    barrier()
    _bitonic_u64(keys, P, t, PRUNE_TPB)
    var o = t
    while o < dg:
        var kk = keys[o] if o < P else U64_MAX
        if kk == U64_MAX:
            _ = Atomic.fetch_add(short, Int32(1))
        else:
            pruned.unsafe_store(a * dg + o, ids[Int(kk & UInt64(0xFFFFFFFF))])
        o += PRUNE_TPB


def rev_keys_kernel(n: Int32, deg: Int32, pruned: I32P, keys: MutPointer[UInt32, MutAnyOrigin],
                    vals: MutPointer[UInt32, MutAnyOrigin]):
    """Edge e = k * n + src (the reverse lists' (rank, source) order):
    key the destination, carry e. A STABLE sort by key then lists every
    destination's sources in (rank, source id) order, `cagra_reverse_merge`'s
    order."""
    var e = _tid()
    var nn = Int(n)
    var total = nn * Int(deg)
    if e < total:
        var k = e // nn
        var src = e - k * nn
        keys[e] = pruned.unsafe_load(src * Int(deg) + k).cast[DType.uint32]()
        vals[e] = UInt32(e)


def merge_kernel[DMAX: Int](n: Int32, deg: Int32, pruned: I32P, skeys: MutPointer[UInt32, MutAnyOrigin],
                            svals: MutPointer[UInt32, MutAnyOrigin], outg: I32P):
    """`cagra_reverse_merge`'s merge for node nid = block_idx.x, block-wide.
    The host inserts the reverse sources v_{kr-1}, ..., v_0 (kr = min(rcount,
    deg)) each at slot `protected`, moving it from its slot when present in
    the tail and dropping the last slot when absent, skipping one already in
    the protected head. Distinct sources (a source's pruned row is distinct)
    make that a move-to-front list, so the tail is
        [v_i, i ascending, v_i not in the head] ++ [the old tail minus those]
    cut at deg - protected. Each entry's slot is a count of the entries
    before it with the same flag: integer work, the host's row."""
    var nid = Int(block_idx.x)
    var t = Int(thread_idx.x)
    var nn = Int(n)
    var dg = Int(deg)
    var protected = dg // 2
    var row = stack_allocation[DMAX, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var rev = stack_allocation[DMAX, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var fa = stack_allocation[DMAX, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var fb = stack_allocation[DMAX, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var span = stack_allocation[2, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    if t == 0:
        # this node's run in the sorted keys: [lower_bound(nid), lower_bound(nid + 1))
        var total = nn * dg
        var lo = 0
        var hi = total
        while lo < hi:
            var mid = (lo + hi) // 2
            if Int(skeys[mid]) < nid:
                lo = mid + 1
            else:
                hi = mid
        var lo2 = lo
        var hi2 = total
        while lo2 < hi2:
            var mid = (lo2 + hi2) // 2
            if Int(skeys[mid]) < nid + 1:
                lo2 = mid + 1
            else:
                hi2 = mid
        span[0] = Int32(lo)
        span[1] = Int32(lo2 - lo)
    var i = t
    while i < dg:
        row[i] = pruned.unsafe_load(nid * dg + i)
        i += PRUNE_TPB
    barrier()
    var start = Int(span[0])
    var kr = Int(span[1])
    if kr > dg:
        kr = dg
    i = t
    while i < kr:
        rev[i] = Int32(Int(svals[start + i]) % nn)
        i += PRUNE_TPB
    barrier()
    # fa[i]: v_i is not in the head; fb[j]: tail entry j is no inserted v_i
    i = t
    while i < kr:
        var v = rev[i]
        var keep = Int32(1)
        for h in range(protected):
            if row[h] == v:
                keep = 0
        fa[i] = keep
        i += PRUNE_TPB
    var j = protected + t
    while j < dg:
        var w = row[j]
        var keep = Int32(1)
        for r in range(kr):
            if rev[r] == w:
                keep = 0
        fb[j] = keep
        j += PRUNE_TPB
    barrier()
    var tail = dg - protected
    var na = 0
    for r in range(kr):
        na += Int(fa[r])
    i = t
    while i < kr:
        if fa[i] != 0:
            var pos = 0
            for r in range(i):
                pos += Int(fa[r])
            if pos < tail:
                outg.unsafe_store(nid * dg + protected + pos, rev[i])
        i += PRUNE_TPB
    j = protected + t
    while j < dg:
        if fb[j] != 0:
            var pos = na
            for r in range(protected, j):
                pos += Int(fb[r])
            if pos < tail:
                outg.unsafe_store(nid * dg + protected + pos, row[j])
        j += PRUNE_TPB
    i = t
    while i < protected:
        outg.unsafe_store(nid * dg + i, row[i])
        i += PRUNE_TPB


def _launch_prune(ctx: DeviceContext, n: Int, kdeg: Int, deg: Int, dni: I32P, dout: I32P, dshort: I32P) raises:
    if kdeg <= 64:
        ctx.enqueue_function[prune_kernel[64]](Int32(n), Int32(kdeg), Int32(deg), dni, dout, dshort,
                                               grid_dim=n, block_dim=PRUNE_TPB)
    elif kdeg <= 128:
        ctx.enqueue_function[prune_kernel[128]](Int32(n), Int32(kdeg), Int32(deg), dni, dout, dshort,
                                                grid_dim=n, block_dim=PRUNE_TPB)
    elif kdeg <= 256:
        ctx.enqueue_function[prune_kernel[256]](Int32(n), Int32(kdeg), Int32(deg), dni, dout, dshort,
                                                grid_dim=n, block_dim=PRUNE_TPB)
    elif kdeg <= 512:
        ctx.enqueue_function[prune_kernel[512]](Int32(n), Int32(kdeg), Int32(deg), dni, dout, dshort,
                                                grid_dim=n, block_dim=PRUNE_TPB)
    else:
        ctx.enqueue_function[prune_kernel[1024]](Int32(n), Int32(kdeg), Int32(deg), dni, dout, dshort,
                                                 grid_dim=n, block_dim=PRUNE_TPB)


def _launch_merge(ctx: DeviceContext, n: Int, deg: Int, dpr: I32P, sk: MutPointer[UInt32, MutAnyOrigin],
                  sv: MutPointer[UInt32, MutAnyOrigin], dout: I32P) raises:
    if deg <= 64:
        ctx.enqueue_function[merge_kernel[64]](Int32(n), Int32(deg), dpr, sk, sv, dout, grid_dim=n,
                                               block_dim=PRUNE_TPB)
    elif deg <= 256:
        ctx.enqueue_function[merge_kernel[256]](Int32(n), Int32(deg), dpr, sk, sv, dout, grid_dim=n,
                                                block_dim=PRUNE_TPB)
    else:
        ctx.enqueue_function[merge_kernel[1024]](Int32(n), Int32(deg), dpr, sk, sv, dout, grid_dim=n,
                                                 block_dim=PRUNE_TPB)


def cagra_build_device(x: List[Float32], n: Int, d: Int, kdeg: Int, deg: Int) raises -> List[Int32]:
    """The exact k-NN graph, the prune and the reverse-edge merge, all on
    the device; the merged graph is the one download."""
    if kdeg > PRUNE_KMAX:
        raise Error("CAGRA: intermediate_graph_degree " + String(kdeg) + " exceeds the device prune's "
                    + String(PRUNE_KMAX) + " (one block per node holds its row)")
    if n * deg >= 2147483647:
        raise Error("CAGRA: n * graph_degree must be below 2^31 (the reverse-edge sort's index)")
    var st = AnnStages("cagra_build")
    var ctx = x_ann_ctx()
    var dx = upload_f32(ctx, x)
    st.mark(ctx, "upload")
    var dnd = ctx.enqueue_create_buffer[DType.float32](n * kdeg)
    var dni = ctx.enqueue_create_buffer[DType.int32](n * kdeg)
    knn_enqueue(ctx, dx, n, d, kdeg, dnd, dni)
    var dpr = ctx.enqueue_create_buffer[DType.int32](n * deg)
    var dshort = ctx.enqueue_create_buffer[DType.int32](1)
    dshort.enqueue_fill(Int32(0))
    _launch_prune(ctx, n, kdeg, deg, dni.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
                  dpr.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
                  dshort.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]())
    st.mark(ctx, "knn_prune")
    var short = download_i32(ctx, dshort, 1)
    if short[0] != 0:
        raise Error("CAGRA: the k-NN graph has too few distinct neighbors for graph_degree")
    var total = n * deg
    var keys = ctx.enqueue_create_buffer[DType.uint32](total)
    var vals = ctx.enqueue_create_buffer[DType.uint32](total)
    var tkeys = ctx.enqueue_create_buffer[DType.uint32](total)
    var tvals = ctx.enqueue_create_buffer[DType.uint32](total)
    var counts = ctx.enqueue_create_buffer[DType.int32](frs_counts_len(total))
    ctx.enqueue_function[rev_keys_kernel](Int32(n), Int32(deg), dpr.unsafe_ptr(), keys.unsafe_ptr(),
                                          vals.unsafe_ptr(), grid_dim=_grid(total), block_dim=TPB)
    fast_radix_sort_pairs_u32(ctx, total, keys, vals, tkeys, tvals, counts)
    var dmg = ctx.enqueue_create_buffer[DType.int32](total)
    _launch_merge(ctx, n, deg, dpr.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
                  keys.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
                  vals.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
                  dmg.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]())
    st.mark(ctx, "reverse_merge")
    var merged = download_i32(ctx, dmg, total)
    st.host("download")
    _ = dmg^
    _ = counts^
    _ = tvals^
    _ = tkeys^
    _ = vals^
    _ = keys^
    _ = dshort^
    _ = dpr^
    _ = dni^
    _ = dnd^
    _ = dx^
    _ = ctx^
    return merged^


def cagra_search_device(
    x: List[Float32], n: Int, d: Int, graph: List[Int32], deg: Int, queries: List[Float32], m: Int,
    k: Int, L: Int, width: Int, max_iter: Int, n_seeds: Int,
    mut out_d: List[Float32], mut out_i: List[Int32], rs: Int = 0,
) raises:
    var st = AnnStages("cagra_search")
    var ctx = x_ann_ctx()
    var dx = upload_f32(ctx, x)
    var dg = upload_i32(ctx, graph)
    st.mark(ctx, "upload")
    cagra_search_on(
        ctx, dx.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), n, d,
        dg.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), deg, queries, m, k, L, width, max_iter,
        n_seeds, out_d, out_i, rs,
    )
    _ = dg^
    _ = dx^
    _ = ctx^


def cagra_search_on(
    ctx: DeviceContext, dx: F32P, n: Int, d: Int, dg: I32P, deg: Int, queries: List[Float32], m: Int,
    k: Int, L: Int, width: Int, max_iter: Int, n_seeds: Int,
    mut out_d: List[Float32], mut out_i: List[Int32], rs: Int = 0,
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
    # lane ann-apple3, FAST on Apple, OPT-IN: a threadgroup per query.
    # lane/apple-fast-ann (2026-10-02): `-D MOJOLEARN_CAGRA_FAST_TEAM=1`
    # (a build define, x_ann/fast_env.mojo `FAST_CAGRA_TEAM`; no env read on
    # the search path) selects the same kernel. Cause:
    # the default `cg_search_kernel` is one thread per query, 4,000 queries
    # = 63 threadgroups of 64 threads on an 80-core GPU, each thread forming
    # every candidate distance (220 features on Istella) alone; the team
    # kernel forms a parent's 32 distances side by side. Expected to move no
    # bit (its docstring); the A/B is the measurement.
    var team = False
    comptime if CAGRA_TEAM:
        if (ANN3_CAGRA_TEAM or FAST_CAGRA_TEAM) and L <= CG_LMAX and deg <= CG_CMAX:
            team = True
            ctx.enqueue_function[cg_search_team_kernel](
                dq.unsafe_ptr(), dx, Int32(n), Int32(d), dg, Int32(deg), Int32(k), Int32(L), Int32(width),
                Int32(max_iter), Int32(n_seeds), vis.unsafe_ptr(), Int32(words), od.unsafe_ptr(),
                oi.unsafe_ptr(), Int32(rs), grid_dim=m, block_dim=CG_T,
            )
    if not team:
        ctx.enqueue_function[cg_search_kernel](
            Int32(m), dq.unsafe_ptr(), dx, Int32(n), Int32(d), dg, Int32(deg), Int32(k),
            Int32(L), Int32(width), Int32(max_iter), Int32(n_seeds), bd.unsafe_ptr(), bi.unsafe_ptr(),
            bx.unsafe_ptr(), vis.unsafe_ptr(), Int32(words), od.unsafe_ptr(), oi.unsafe_ptr(),
            Int32(rs), grid_dim=_grid(m), block_dim=TPB,
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
