# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The single-linkage tail on the device (lane hr2-mds-agglo, 2026-10-02):
the MST sort, the dendrogram and the cluster roots, with no host step.

THE SORT (`coo_sort_by_weight_device`). DEVIATION 621's total order
(`pack_edge_key(weight_order_key(w), min(u,v), max(u,v))`) as a RANK sort:
every edge counts the edges whose key is below its own (keys are distinct,
so each rank is a distinct position) and is scattered there. The same list
the host merge sort returned, by construction.

THE DENDROGRAM (`build_dendrogram_device`). `build_dendrogram_host` walks the
sorted list with a union-find: edge i's children are the clusters of its two
endpoints in the forest of the edges before it, each named by its node id
(n + the index of its last merge, or the leaf itself). Here the same names
come from a divide and conquer over the edge INDEX: for a block of edges
[lo, lo + s), every endpoint is labelled by the node id of its cluster in the
forest of the edges before lo. Splitting the block, the lower half keeps its
labels, and the upper half's labels become the components of the lower half's
edges over those labels (a forest: connected components by min-label hooking
and full pointer jumping, then the component's node id = n + its largest edge
index, and its size the sum of its labels' sizes). At block size 1 a label is
the cluster before that edge: the children, oriented (src, dst) as the
union-find's `find(src)`, `find(dst)`, the sizes summed. The same three
outputs as the host walk, bit for bit (integer work only).

WHY THE LABEL ARRAYS CAN BE SHARED BY EVERY BLOCK OF A LEVEL. A node id
names one vertex set. A label that a lower half's edge touches is merged at
that level, so no later block can still carry it, and an earlier block
cannot carry it either (an edge between that block and this one touched
it). So each label is hooked in at most one block per level, and the
upper-half lookups test a per-level stamp.
"""

from std.atomic import Atomic
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.sys.compile import is_defined
from std.sys.defines import get_defined_int

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.fast_radix_sort import fast_radix_sort_pairs_u32, frs_counts_len
from hierarchy.checks.edge_order import (
    LINK_SAB_SORT_WEIGHT_ONLY,
    edge_hi,
    edge_lo,
    pack_edge_key,
    weight_order_key,
)

comptime DD_TPB = 256
comptime I32P = MutPointer[Int32, MutAnyOrigin]
comptime F32P = MutPointer[Float32, MutAnyOrigin]
comptime U64P = MutPointer[UInt64, MutAnyOrigin]
comptime U32P = MutPointer[UInt32, MutAnyOrigin]

#: fam-cluster (2026-10-04), IDENTICAL: the edge order comes from two stable
#: radix sorts of the 64-bit key's halves (low, then high; `core/
#: fast_radix_sort.mojo`) instead of `_sort_rank_kernel`'s all-pairs count
#: (nnz^2 compares). The keys are distinct, so the sorted position IS the
#: counted rank: same permutation. `-D MOJOLEARN_IDN_DENDRO_RADIX_SORT_OFF=1`
#: restores the count.
comptime IDN_DENDRO_RADIX_SORT = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not (
        is_defined["MOJOLEARN_IDN_DENDRO_RADIX_SORT_OFF"]()
        or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
    )
)

#: fam2-cluster (2026-10-04), IDENTICAL, every vendor: the per-level hook loop
#: replaced by ONE lock-free union launch (`hdbscan/impl/cluster/detail/
#: dendrogram_union.mojo`, written for FAST on Apple by lane af-hdbscan2), so
#: no level reads a flag back: one wait for the whole dendrogram where the
#: hook loop waits two or three times per level. Integer work only; the
#: union's root is each component's smallest label, the root the converged
#: hook loop ends at, so `par`, and with it the three outputs, are the same
#: integers. Callers switch on this (`single_linkage.mojo` here and in
#: hdbscan). `-D MOJOLEARN_IDN_DENDRO_UNION_OFF=1` restores the hook loop.
comptime IDN_DENDRO_UNION = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not (
        is_defined["MOJOLEARN_IDN_DENDRO_UNION_OFF"]()
        or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
    )
)

#: fam2-cluster (2026-10-04), IDENTICAL (the route when IDN_DENDRO_UNION is
#: off): the dendrogram's hook + jump loop
#: reads its "something hooked" flag once per CHUNK of pairs instead of once
#: per pair. A pair past the fixed point changes nothing (no lower-half edge
#: joins two roots, so the hook writes nothing and the jump finds every root
#: in place), and the flag is cleared before the chunk's LAST pair only, so
#: it answers the same question the per-pair read did. Same parents, same
#: dendrogram. The chunk is `-D MOJOLEARN_IDN_DENDRO_HOOK_CHUNK=<n>`
#: (default 4; time 2 and 8 against it);
#: `-D MOJOLEARN_IDN_DENDRO_HOOK_CHUNK_OFF=1` restores the per-pair read.
comptime IDN_DENDRO_HOOK_CHUNK = (
    get_defined_int["MOJOLEARN_IDN_DENDRO_HOOK_CHUNK", 4]()
    if (
        GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
        and not (
            is_defined["MOJOLEARN_IDN_DENDRO_HOOK_CHUNK_OFF"]()
            or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
        )
    )
    else 1
)


@always_inline
def _gid() -> Int:
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


@always_inline
def _blocks(n: Int) -> Int:
    return (n + DD_TPB - 1) // DD_TPB if n > 0 else 1


# ------------------------------------------------------------------ sort ---

def _sort_key_kernel(rows: I32P, cols: I32P, data: F32P, keys: U64P, cnt: Int32, sabotage: Int32):
    var i = _gid()
    var e = Int(cnt)
    if i < e:
        var u = rows[i]
        var v = cols[i]
        var wk = weight_order_key(data[i])
        if Int(sabotage) == LINK_SAB_SORT_WEIGHT_ONLY:
            keys[i] = pack_edge_key(wk, Int32(0), Int32(0)) | UInt64(e - 1 - i)
        else:
            keys[i] = pack_edge_key(wk, edge_lo(u, v), edge_hi(u, v))


def _sort_rank_kernel(keys: U64P, rank: I32P, cnt: Int32):
    """rank[i] = the number of keys below keys[i], the keys read a tile at
    a time through shared memory."""
    var tile = stack_allocation[DD_TPB, UInt64, address_space = AddressSpace.SHARED]()
    var i = _gid()
    var e = Int(cnt)
    var mine = keys[i] if i < e else UInt64(0)
    var r = 0
    var t0 = 0
    while t0 < e:
        var j = t0 + Int(thread_idx.x)
        tile[Int(thread_idx.x)] = keys[j] if j < e else UInt64(0xFFFFFFFFFFFFFFFF)
        barrier()
        var lim = min(DD_TPB, e - t0)
        for q in range(lim):
            if tile[q] < mine:
                r += 1
        barrier()
        t0 += DD_TPB
    if i < e:
        rank[i] = Int32(r)


def _sort_half_kernel(keys: U64P, idx: U32P, half: U32P, cnt: Int32, high: Int32):
    """half[p] = the low (`high == 0`) or high 32 bits of the key position p
    holds. The low pass starts the permutation at the identity; the high
    pass reads it from `idx`."""
    var p = _gid()
    if p >= Int(cnt):
        return
    if Int(high) == 0:
        idx[p] = UInt32(p)
        half[p] = (keys[p] & UInt64(0xFFFFFFFF)).cast[DType.uint32]()
    else:
        half[p] = (keys[Int(idx[p])] >> UInt64(32)).cast[DType.uint32]()


def _sort_rank_from_perm_kernel(idx: U32P, rank: I32P, cnt: Int32):
    var p = _gid()
    if p < Int(cnt):
        rank[Int(idx[p])] = Int32(p)


def _sort_scatter_kernel(
    rows: I32P, cols: I32P, data: F32P, rank: I32P, o_rows: I32P, o_cols: I32P, o_data: F32P, cnt: Int32
):
    var i = _gid()
    if i < Int(cnt):
        var p = Int(rank[i])
        o_rows[p] = rows[i]
        o_cols[p] = cols[i]
        o_data[p] = data[i]


def _copy3_kernel(src_r: I32P, src_c: I32P, src_d: F32P, dst_r: I32P, dst_c: I32P, dst_d: F32P, cnt: Int32):
    var i = _gid()
    if i < Int(cnt):
        dst_r[i] = src_r[i]
        dst_c[i] = src_c[i]
        dst_d[i] = src_d[i]


def coo_sort_by_weight_device(
    ctx: DeviceContext,
    mut rows: DeviceBuffer[DType.int32],
    mut cols: DeviceBuffer[DType.int32],
    mut data: DeviceBuffer[DType.float32],
    nnz: Int,
    sabotage: Int32,
) raises:
    """`coo_sort_by_weight`'s order, in place, on the device."""
    if nnz <= 1:
        return
    var keys = ctx.enqueue_create_buffer[DType.uint64](nnz)
    var rank = ctx.enqueue_create_buffer[DType.int32](nnz)
    var t_rows = ctx.enqueue_create_buffer[DType.int32](nnz)
    var t_cols = ctx.enqueue_create_buffer[DType.int32](nnz)
    var t_data = ctx.enqueue_create_buffer[DType.float32](nnz)
    var g = _blocks(nnz)
    ctx.enqueue_function[_sort_key_kernel](
        rows.unsafe_ptr(), cols.unsafe_ptr(), data.unsafe_ptr(), keys.unsafe_ptr(), Int32(nnz), sabotage,
        grid_dim=(g, 1, 1), block_dim=(DD_TPB, 1, 1),
    )
    comptime if IDN_DENDRO_RADIX_SORT:
        var rs_half = ctx.enqueue_create_buffer[DType.uint32](nnz)
        var rs_idx = ctx.enqueue_create_buffer[DType.uint32](nnz)
        var rs_tk = ctx.enqueue_create_buffer[DType.uint32](nnz)
        var rs_tv = ctx.enqueue_create_buffer[DType.uint32](nnz)
        var rs_counts = ctx.enqueue_create_buffer[DType.int32](frs_counts_len(nnz))
        ctx.enqueue_function[_sort_half_kernel](
            keys.unsafe_ptr(), rs_idx.unsafe_ptr(), rs_half.unsafe_ptr(), Int32(nnz), Int32(0),
            grid_dim=(g, 1, 1), block_dim=(DD_TPB, 1, 1),
        )
        fast_radix_sort_pairs_u32(ctx, nnz, rs_half, rs_idx, rs_tk, rs_tv, rs_counts)
        ctx.enqueue_function[_sort_half_kernel](
            keys.unsafe_ptr(), rs_idx.unsafe_ptr(), rs_half.unsafe_ptr(), Int32(nnz), Int32(1),
            grid_dim=(g, 1, 1), block_dim=(DD_TPB, 1, 1),
        )
        fast_radix_sort_pairs_u32(ctx, nnz, rs_half, rs_idx, rs_tk, rs_tv, rs_counts)
        ctx.enqueue_function[_sort_rank_from_perm_kernel](
            rs_idx.unsafe_ptr(), rank.unsafe_ptr(), Int32(nnz), grid_dim=(g, 1, 1), block_dim=(DD_TPB, 1, 1),
        )
        # the launches hold raw pointers into the sort's scratch
        ctx.synchronize()
        _ = rs_half^
        _ = rs_idx^
        _ = rs_tk^
        _ = rs_tv^
        _ = rs_counts^
    else:
        ctx.enqueue_function[_sort_rank_kernel](
            keys.unsafe_ptr(), rank.unsafe_ptr(), Int32(nnz), grid_dim=(g, 1, 1), block_dim=(DD_TPB, 1, 1),
        )
    ctx.enqueue_function[_sort_scatter_kernel](
        rows.unsafe_ptr(), cols.unsafe_ptr(), data.unsafe_ptr(), rank.unsafe_ptr(),
        t_rows.unsafe_ptr(), t_cols.unsafe_ptr(), t_data.unsafe_ptr(), Int32(nnz),
        grid_dim=(g, 1, 1), block_dim=(DD_TPB, 1, 1),
    )
    ctx.enqueue_function[_copy3_kernel](
        t_rows.unsafe_ptr(), t_cols.unsafe_ptr(), t_data.unsafe_ptr(),
        rows.unsafe_ptr(), cols.unsafe_ptr(), data.unsafe_ptr(), Int32(nnz),
        grid_dim=(g, 1, 1), block_dim=(DD_TPB, 1, 1),
    )
    ctx.synchronize()
    _ = keys^
    _ = rank^
    _ = t_rows^
    _ = t_cols^
    _ = t_data^


# ------------------------------------------------------------ dendrogram ---

@always_inline
def _light(i: Int, s: Int, cnt: Int) -> Bool:
    return i < cnt and (i % s) < s // 2


def _dd_init_kernel(rows: I32P, cols: I32P, la: I32P, lb: I32P, sz: I32P, stamp: I32P, cnt: Int32, n_nodes: Int32):
    var i = _gid()
    if i < Int(n_nodes):
        stamp[i] = -1
        sz[i] = 1
    if i < Int(cnt):
        la[i] = rows[i]
        lb[i] = cols[i]


def _dd_claim_kernel(la: I32P, lb: I32P, par: I32P, top: I32P, ssum: I32P, stamp: I32P, owner: I32P, cnt: Int32, s: Int32, lvl: Int32):
    """Every label a lower-half edge touches: its own parent, no top edge
    yet, no size yet, this level's stamp (the same values from every
    thread that writes them)."""
    var i = _gid()
    if not _light(i, Int(s), Int(cnt)):
        return
    for side in range(2):
        var x = Int(la[i]) if side == 0 else Int(lb[i])
        par[x] = Int32(x)
        top[x] = -1
        ssum[x] = 0
        stamp[x] = lvl
        owner[x] = Int32(0x7FFFFFFF)


def _dd_own_kernel(la: I32P, lb: I32P, owner: I32P, cnt: Int32, s: Int32):
    """One owner slot per label (the lowest), so its size is added once."""
    var i = _gid()
    if not _light(i, Int(s), Int(cnt)):
        return
    _ = Atomic.min(owner.unsafe_offset(Int(la[i])), Int32(2 * i))
    _ = Atomic.min(owner.unsafe_offset(Int(lb[i])), Int32(2 * i + 1))


def _dd_hook_kernel(la: I32P, lb: I32P, par: I32P, flag: I32P, cnt: Int32, s: Int32):
    """The larger of the two parents points at the smaller (an integer min:
    the order the hooks land in moves nothing but the round count)."""
    var i = _gid()
    if not _light(i, Int(s), Int(cnt)):
        return
    var pa = par[Int(la[i])]
    var pb = par[Int(lb[i])]
    if pa != pb:
        flag[0] = 1
        var hi = pa if pa > pb else pb
        var lo = pb if pa > pb else pa
        _ = Atomic.min(par.unsafe_offset(Int(hi)), lo)


def _dd_jump_kernel(la: I32P, lb: I32P, par: I32P, cnt: Int32, s: Int32):
    """Each endpoint label points at its root (parents only decrease, so
    the walk ends)."""
    var i = _gid()
    if not _light(i, Int(s), Int(cnt)):
        return
    for side in range(2):
        var x = Int(la[i]) if side == 0 else Int(lb[i])
        var r = Int(par[x])
        while Int(par[r]) != r:
            r = Int(par[r])
        par[x] = Int32(r)


def _dd_top_kernel(la: I32P, lb: I32P, par: I32P, top: I32P, ssum: I32P, sz: I32P, owner: I32P, cnt: Int32, s: Int32):
    """Per component (at its root label): the largest edge index and the
    sum of its labels' sizes, each label counted by its owner slot."""
    var i = _gid()
    if not _light(i, Int(s), Int(cnt)):
        return
    var x = Int(la[i])
    var y = Int(lb[i])
    var r = Int(par[x])
    _ = Atomic.max(top.unsafe_offset(r), Int32(i))
    if Int(owner[x]) == 2 * i:
        _ = Atomic.fetch_add(ssum.unsafe_offset(r), sz[x])
    if Int(owner[y]) == 2 * i + 1:
        _ = Atomic.fetch_add(ssum.unsafe_offset(r), sz[y])


def _dd_size_kernel(la: I32P, lb: I32P, par: I32P, top: I32P, ssum: I32P, sz: I32P, owner: I32P, cnt: Int32, s: Int32, m: Int32):
    """The component's node id n + top gets the component's size."""
    var i = _gid()
    if not _light(i, Int(s), Int(cnt)):
        return
    for side in range(2):
        var x = Int(la[i]) if side == 0 else Int(lb[i])
        if Int(owner[x]) == 2 * i + side and Int(par[x]) == x:
            sz[Int(m) + Int(top[x])] = ssum[x]


def _dd_relabel_kernel(la: I32P, lb: I32P, par: I32P, top: I32P, stamp: I32P, cnt: Int32, s: Int32, lvl: Int32, m: Int32):
    """An upper-half endpoint whose label the lower half merged takes its
    component's node id."""
    var i = _gid()
    if i >= Int(cnt) or (i % Int(s)) < Int(s) // 2:
        return
    var x = Int(la[i])
    if stamp[x] == lvl:
        la[i] = m + top[Int(par[x])]
    var y = Int(lb[i])
    if stamp[y] == lvl:
        lb[i] = m + top[Int(par[y])]


def _dd_out_kernel(la: I32P, lb: I32P, data: F32P, sz: I32P, children: I32P, delta: F32P, size: I32P, cnt: Int32):
    var i = _gid()
    if i < Int(cnt):
        var a = la[i]
        var b = lb[i]
        children[2 * i] = a
        children[2 * i + 1] = b
        delta[i] = data[i]
        size[i] = sz[Int(a)] + sz[Int(b)]


def build_dendrogram_device(
    ctx: DeviceContext,
    mut rows: DeviceBuffer[DType.int32],
    mut cols: DeviceBuffer[DType.int32],
    mut data: DeviceBuffer[DType.float32],
    nnz: Int,
    mut children: DeviceBuffer[DType.int32],
    mut out_delta: DeviceBuffer[DType.float32],
    mut out_size: DeviceBuffer[DType.int32],
) raises:
    """`build_dendrogram_host`'s three outputs from the sorted edge list,
    on the device (the divide and conquer in the module docstring)."""
    var cnt = nnz
    if cnt < 1:
        return
    var m = cnt + 1
    var n_nodes = 2 * m
    var la = ctx.enqueue_create_buffer[DType.int32](cnt)
    var lb = ctx.enqueue_create_buffer[DType.int32](cnt)
    var par = ctx.enqueue_create_buffer[DType.int32](n_nodes)
    var top = ctx.enqueue_create_buffer[DType.int32](n_nodes)
    var ssum = ctx.enqueue_create_buffer[DType.int32](n_nodes)
    var sz = ctx.enqueue_create_buffer[DType.int32](n_nodes)
    var stamp = ctx.enqueue_create_buffer[DType.int32](n_nodes)
    var owner = ctx.enqueue_create_buffer[DType.int32](n_nodes)
    var flag = ctx.enqueue_create_buffer[DType.int32](1)
    var p_la = la.unsafe_ptr()
    var p_lb = lb.unsafe_ptr()
    var p_par = par.unsafe_ptr()
    var p_top = top.unsafe_ptr()
    var p_ssum = ssum.unsafe_ptr()
    var p_sz = sz.unsafe_ptr()
    var p_stamp = stamp.unsafe_ptr()
    var p_owner = owner.unsafe_ptr()
    var p_flag = flag.unsafe_ptr()
    var ge = _blocks(cnt)
    ctx.enqueue_function[_dd_init_kernel](
        rows.unsafe_ptr(), cols.unsafe_ptr(), p_la, p_lb, p_sz, p_stamp, Int32(cnt), Int32(n_nodes),
        grid_dim=(_blocks(n_nodes), 1, 1), block_dim=(DD_TPB, 1, 1),
    )
    var s = 1
    while s < cnt:
        s *= 2
    var lvl = 0
    var h_flag = ctx.enqueue_create_host_buffer[DType.int32](1)
    while s >= 2:
        var S = Int32(s)
        var L = Int32(lvl)
        ctx.enqueue_function[_dd_claim_kernel](
            p_la, p_lb, p_par, p_top, p_ssum, p_stamp, p_owner, Int32(cnt), S, L,
            grid_dim=(ge, 1, 1), block_dim=(DD_TPB, 1, 1),
        )
        ctx.enqueue_function[_dd_own_kernel](
            p_la, p_lb, p_owner, Int32(cnt), S, grid_dim=(ge, 1, 1), block_dim=(DD_TPB, 1, 1),
        )
        # hook and jump until no lower-half edge joins two roots; the flag
        # is the only value read back
        var more = True
        while more:
            # fam2-cluster, IDN_DENDRO_HOOK_CHUNK: the flag is cleared
            # before the chunk's last pair, so it reports that pair alone.
            comptime assert IDN_DENDRO_HOOK_CHUNK >= 1, "MOJOLEARN_IDN_DENDRO_HOOK_CHUNK must be >= 1"
            for q in range(IDN_DENDRO_HOOK_CHUNK):
                if q == IDN_DENDRO_HOOK_CHUNK - 1:
                    ctx.enqueue_memset(flag, Int32(0))
                ctx.enqueue_function[_dd_hook_kernel](
                    p_la, p_lb, p_par, p_flag, Int32(cnt), S, grid_dim=(ge, 1, 1), block_dim=(DD_TPB, 1, 1),
                )
                ctx.enqueue_function[_dd_jump_kernel](
                    p_la, p_lb, p_par, Int32(cnt), S, grid_dim=(ge, 1, 1), block_dim=(DD_TPB, 1, 1),
                )
            ctx.enqueue_copy(dst_ptr=h_flag.unsafe_ptr(), src_buf=flag)
            ctx.synchronize()
            more = h_flag.unsafe_ptr()[0] != 0
        ctx.enqueue_function[_dd_top_kernel](
            p_la, p_lb, p_par, p_top, p_ssum, p_sz, p_owner, Int32(cnt), S,
            grid_dim=(ge, 1, 1), block_dim=(DD_TPB, 1, 1),
        )
        ctx.enqueue_function[_dd_size_kernel](
            p_la, p_lb, p_par, p_top, p_ssum, p_sz, p_owner, Int32(cnt), S, Int32(m),
            grid_dim=(ge, 1, 1), block_dim=(DD_TPB, 1, 1),
        )
        ctx.enqueue_function[_dd_relabel_kernel](
            p_la, p_lb, p_par, p_top, p_stamp, Int32(cnt), S, L, Int32(m),
            grid_dim=(ge, 1, 1), block_dim=(DD_TPB, 1, 1),
        )
        s //= 2
        lvl += 1
    ctx.enqueue_function[_dd_out_kernel](
        p_la, p_lb, data.unsafe_ptr(), p_sz, children.unsafe_ptr(), out_delta.unsafe_ptr(), out_size.unsafe_ptr(),
        Int32(cnt), grid_dim=(ge, 1, 1), block_dim=(DD_TPB, 1, 1),
    )
    ctx.synchronize()
    _ = la^
    _ = lb^
    _ = par^
    _ = top^
    _ = ssum^
    _ = sz^
    _ = stamp^
    _ = owner^
    _ = flag^
    _ = h_flag^


# ----------------------------------------------------------------- roots ---

def _roots_kernel(children: I32P, roots: I32P, n_edges: Int32, n_clusters: Int32):
    """`extract_flattened_clusters`' roots: of the last 2 (k - 1) children
    sorted DESCENDING, the k at the tail, label j to the j-th of that tail.
    Each candidate's descending rank is the count of larger candidates
    (node ids are distinct)."""
    var t = _gid()
    var k = Int(n_clusters)
    var cs = (k - 1) * 2
    if t >= cs:
        return
    var start = Int(n_edges) - cs
    var c = children[start + t]
    var r = 0
    for q in range(cs):
        if children[start + q] > c:
            r += 1
    var j = r - (cs - k)
    if j >= 0:
        roots[j] = c
