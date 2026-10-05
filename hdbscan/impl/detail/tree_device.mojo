# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The condensed tree built on the device, and the device helpers the
selection, the labelling and the prediction data share (lane cgr2-hdbscan,
2026-10-03). Every step is a grid-wide launch; there is no host walk.

THE CONDENSE, CLOSED FORM. `condense.mojo::build_condensed_hierarchy`'s
host walk (and `hdbh_condense`, the CPU column's) visits the dendrogram
breadth first and, at every internal node it has not collapsed, takes one
of four branches on `size(left) >= min_cluster_size` and
`size(right) >= min_cluster_size`. Read as a function of the tree, the walk
says:

  live(v)     every node on the path root .. v has size >= min_cluster_size
              (a child of a live node is visited iff it is big; a small
              child is collapsed with its whole subtree)
  split(s)    s is live and both its children are big (case 1)
  start(v)    v is the root or a child of a split node: the nodes that get
              a NEW label; every other live node inherits its parent's
              label (cases 3 and 4)
  label       the root is n_leaves; the children of split node s are
              n_leaves + 1 + 2 * rank(s) (left) and that + 1 (right), where
              rank(s) is s's position among the split nodes in the order
              the walk visits them
  edges       each start node c below the root: (label of c's parent's
              start ancestor, label(c), lambda(parent(c)), size(c)); each
              leaf l: (label of l's start ancestor, l, lambda(L), 1) with
              L the nearest live ancestor of l (the node whose collapse
              emitted l)

The walk visits level by level, left before right within a level, so its
order on nodes is the order of the key (depth, preorder index): two nodes
of one level are not ancestor and descendant, and preorder puts the left
one first. `depth` and `preorder` are sums over the root path (preorder
adds `2 * size(left sibling)` at every right turn and 1 at every left
turn), computed by pointer jumping; `live` is an AND over the same path,
in the same launches. The split nodes are compacted by an exclusive scan,
stably sorted by preorder and then by depth (two radix passes, integer
keys), and a node's rank is its position. `HDB_SAB_CONDENSE_DFS` skips the
depth pass: preorder alone is the depth-first walk the sabotage names.

THE SORT. Every node has one parent, so the edges indexed by CHILD (leaf
`l` at slot `l`, cluster label `x` at slot `x - 1`) are already in child
order, and ONE stable radix sort by parent leaves them in (parent, child)
order: `CondensedHierarchy.condense()`'s order (DEVIATION 1611), the same
array element for element.

Everything here is integer work except `lambda = 1 / delta`, which goes
through `identical_div` exactly as the host walk's DEVIATION 1606 does, so
the device tree equals the host walk's tree bit for bit (`hdbh_condense`
is the CPU column's; it keeps the walk).
"""

from std.atomic import Atomic
from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from core.device_scan import device_first_nonfinite
from std.memory import bitcast
from hdbscan.impl.detail.fast_apple import HDB_LINKAGE_DEVICE
from hdbscan.impl.detail.idn_switches import IDN_HDB_CONDENSE_TWO_READS
from core.fast_radix_sort import (
    fast_radix_sort_pairs_u32,
    frs_counts_len,
    frs_exclusive_scan,
    frs_scan_blocks,
)
from hdbscan.checks.hdbscan_sabotage import (
    HDB_SAB_CONDENSE_DFS,
    HDB_SAB_LAMBDA_STD_DIV,
    HDB_SAB_NONE,
    HDB_SAB_SKIP_GUARDS,
)
from hdbscan.impl.condensed_hierarchy import CondensedHierarchy
from hierarchy.impl.cluster.detail.connectivities import FLOAT32_MAX
from checks.numerics import identical_div

comptime TD_TPB = 256
"""Threads per block of every launch here. SCHEDULING only: each launch
writes integers (or one float per slot from one thread)."""

comptime I32P = MutPointer[Int32, MutAnyOrigin]
comptime U32P = MutPointer[UInt32, MutAnyOrigin]
comptime F32P = MutPointer[Float32, MutAnyOrigin]


@always_inline
def _gid() -> Int:
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


def td_grid(n: Int) -> Int:
    return (n + TD_TPB - 1) // TD_TPB if n > 0 else 1


def td_rounds(n: Int) -> Int:
    """Pointer-jumping rounds that reach the end of any chain of `n`
    nodes: the smallest r with 2^r >= n (at least 1)."""
    var r = 1
    while (1 << r) < n:
        r += 1
    return r


# --------------------------------------------------------- shared tools ---


def td_exclusive_scan(
    ctx: DeviceContext, mut buf: DeviceBuffer[DType.int32], m: Int
) raises:
    """Exclusive scan of `buf[0:m]` in place (integer adds, exact)."""
    if m <= 0:
        return
    var bsum = ctx.enqueue_create_buffer[DType.int32](frs_scan_blocks(m))
    frs_exclusive_scan(ctx, buf, m, bsum)
    _ = bsum^


def td_sort_pairs(
    ctx: DeviceContext,
    mut keys: DeviceBuffer[DType.uint32],
    mut vals: DeviceBuffer[DType.uint32],
    size: Int,
) raises:
    """Stable ascending sort of `keys[0:size]` carrying `vals`, in place
    (`core/fast_radix_sort.mojo`; a stable sort is one permutation)."""
    if size <= 1:
        return
    var tk = ctx.enqueue_create_buffer[DType.uint32](size)
    var tv = ctx.enqueue_create_buffer[DType.uint32](size)
    var counts = ctx.enqueue_create_buffer[DType.int32](frs_counts_len(size))
    fast_radix_sort_pairs_u32(ctx, size, keys, vals, tk, tv, counts)
    _ = tk^
    _ = tv^
    _ = counts^


def td_read_i32(
    ctx: DeviceContext, buf: DeviceBuffer[DType.int32], at: Int
) raises -> Int:
    """One int32 cell back to the host (a size or a scalar test)."""
    var h = ctx.enqueue_create_host_buffer[DType.int32](1)
    var v = buf.create_sub_buffer[DType.int32](at, 1)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=v)
    ctx.synchronize()
    var out = Int(h.unsafe_ptr().unsafe_load(0))
    _ = h^
    _ = v^
    return out


def td_download_i32(
    ctx: DeviceContext, buf: DeviceBuffer[DType.int32], n: Int
) raises -> List[Int32]:
    var out = List[Int32](length=n, fill=Int32(0))
    if n <= 0:
        return out^
    var h = ctx.enqueue_create_host_buffer[DType.int32](n)
    var v = buf.create_sub_buffer[DType.int32](0, n)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=v)
    ctx.synchronize()
    for i in range(n):
        out[i] = h.unsafe_ptr().unsafe_load(i)
    _ = h^
    _ = v^
    return out^


def td_download_f32(
    ctx: DeviceContext, buf: DeviceBuffer[DType.float32], n: Int
) raises -> List[Float32]:
    var out = List[Float32](length=n, fill=Float32(0.0))
    if n <= 0:
        return out^
    var h = ctx.enqueue_create_host_buffer[DType.float32](n)
    var v = buf.create_sub_buffer[DType.float32](0, n)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=v)
    ctx.synchronize()
    for i in range(n):
        out[i] = h.unsafe_ptr().unsafe_load(i)
    _ = h^
    _ = v^
    return out^


def td_stage_i32(
    ctx: DeviceContext, buf: DeviceBuffer[DType.int32]
) raises -> HostBuffer[DType.int32]:
    """lane af-hdbscan2 (HDB_ONE_SYNC): the WHOLE device buffer copied into a
    host buffer with NO wait; the caller synchronizes once for every staged
    buffer and then takes the lists (`td_take_*`). No sub-buffer view is
    made, so nothing is freed before the copy runs."""
    var n = len(buf)
    var h = ctx.enqueue_create_host_buffer[DType.int32](max(n, 1))
    if n > 0:
        ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=buf)
    return h^


def td_stage_f32(
    ctx: DeviceContext, buf: DeviceBuffer[DType.float32]
) raises -> HostBuffer[DType.float32]:
    """`td_stage_i32` for Float32."""
    var n = len(buf)
    var h = ctx.enqueue_create_host_buffer[DType.float32](max(n, 1))
    if n > 0:
        ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=buf)
    return h^


def td_take_i32(h: HostBuffer[DType.int32], n: Int) -> List[Int32]:
    """The first `n` words of a staged host buffer, after the wait."""
    var out = List[Int32](length=max(n, 0), fill=Int32(0))
    for i in range(n):
        out[i] = h.unsafe_ptr().unsafe_load(i)
    return out^


def td_take_f32(h: HostBuffer[DType.float32], n: Int) -> List[Float32]:
    """`td_take_i32` for Float32."""
    var out = List[Float32](length=max(n, 0), fill=Float32(0.0))
    for i in range(n):
        out[i] = h.unsafe_ptr().unsafe_load(i)
    return out^


def td_upload_i32(
    ctx: DeviceContext, values: List[Int32], n: Int
) raises -> DeviceBuffer[DType.int32]:
    var d = ctx.enqueue_create_buffer[DType.int32](max(n, 1))
    if n > 0:
        var tmp = values.copy()
        ctx.enqueue_copy(
            dst_buf=d.create_sub_buffer[DType.int32](0, n),
            src_ptr=tmp.unsafe_ptr(),
        )
        ctx.synchronize()
        _ = tmp^
    return d^


def td_upload_f32(
    ctx: DeviceContext, values: List[Float32], n: Int
) raises -> DeviceBuffer[DType.float32]:
    var d = ctx.enqueue_create_buffer[DType.float32](max(n, 1))
    if n > 0:
        var tmp = values.copy()
        ctx.enqueue_copy(
            dst_buf=d.create_sub_buffer[DType.float32](0, n),
            src_ptr=tmp.unsafe_ptr(),
        )
        ctx.synchronize()
        _ = tmp^
    return d^


def td_jump_find_kernel(src: I32P, dst: I32P, n: Int32):
    """One pointer-jumping round, `dst[v] = src[src[v]]`."""
    var v = _gid()
    if v >= Int(n):
        return
    dst[v] = src[Int(src[v])]


def td_jump_or_kernel(ps: I32P, vs: I32P, pd: I32P, vd: I32P, n: Int32):
    """One round of an OR over the path: `vd[v] = vs[v] | vs[ps[v]]`,
    `pd[v] = ps[ps[v]]`."""
    var v = _gid()
    if v >= Int(n):
        return
    var p = Int(ps[v])
    vd[v] = vs[v] | vs[p]
    pd[v] = ps[p]


def td_jump_sum_kernel(ps: I32P, vs: I32P, pd: I32P, vd: I32P, n: Int32):
    """One round of a sum over the path: `vd[v] = vs[v] + vs[ps[v]]`,
    `pd[v] = ps[ps[v]]`. The root points at itself with value 0."""
    var v = _gid()
    if v >= Int(n):
        return
    var p = Int(ps[v])
    vd[v] = vs[v] + vs[p]
    pd[v] = ps[p]


def td_find(
    ctx: DeviceContext,
    mut a: DeviceBuffer[DType.int32],
    mut b: DeviceBuffer[DType.int32],
    n: Int,
) raises -> Bool:
    """Full pointer jumping on `a` (b is scratch). Returns True when the
    result ended in `b`."""
    var r = td_rounds(n)
    for k in range(r):
        if k % 2 == 0:
            ctx.enqueue_function[td_jump_find_kernel](
                a.unsafe_ptr(), b.unsafe_ptr(), Int32(n),
                grid_dim=(td_grid(n), 1, 1), block_dim=(TD_TPB, 1, 1),
            )
        else:
            ctx.enqueue_function[td_jump_find_kernel](
                b.unsafe_ptr(), a.unsafe_ptr(), Int32(n),
                grid_dim=(td_grid(n), 1, 1), block_dim=(TD_TPB, 1, 1),
            )
    return r % 2 == 1


def td_path_reduce[
    is_sum: Bool
](
    ctx: DeviceContext,
    mut pa: DeviceBuffer[DType.int32],
    mut va: DeviceBuffer[DType.int32],
    mut pb: DeviceBuffer[DType.int32],
    mut vb: DeviceBuffer[DType.int32],
    n: Int,
) raises -> Bool:
    """Path sum (or OR) by pointer jumping over (pa, va); (pb, vb) scratch.
    Returns True when the result ended in (pb, vb)."""
    var r = td_rounds(n)
    for k in range(r):
        comptime if is_sum:
            if k % 2 == 0:
                ctx.enqueue_function[td_jump_sum_kernel](
                    pa.unsafe_ptr(), va.unsafe_ptr(), pb.unsafe_ptr(),
                    vb.unsafe_ptr(), Int32(n),
                    grid_dim=(td_grid(n), 1, 1), block_dim=(TD_TPB, 1, 1),
                )
            else:
                ctx.enqueue_function[td_jump_sum_kernel](
                    pb.unsafe_ptr(), vb.unsafe_ptr(), pa.unsafe_ptr(),
                    va.unsafe_ptr(), Int32(n),
                    grid_dim=(td_grid(n), 1, 1), block_dim=(TD_TPB, 1, 1),
                )
        else:
            if k % 2 == 0:
                ctx.enqueue_function[td_jump_or_kernel](
                    pa.unsafe_ptr(), va.unsafe_ptr(), pb.unsafe_ptr(),
                    vb.unsafe_ptr(), Int32(n),
                    grid_dim=(td_grid(n), 1, 1), block_dim=(TD_TPB, 1, 1),
                )
            else:
                ctx.enqueue_function[td_jump_or_kernel](
                    pb.unsafe_ptr(), vb.unsafe_ptr(), pa.unsafe_ptr(),
                    va.unsafe_ptr(), Int32(n),
                    grid_dim=(td_grid(n), 1, 1), block_dim=(TD_TPB, 1, 1),
                )
    return r % 2 == 1


# ------------------------------------------------------- the device tree ---


@fieldwise_init
struct DeviceTree(Copyable, Movable):
    """The condensed tree on the device, sorted by (parent, child), with
    the indexes every consumer reads.

      parents, children, lambdas, sizes   n_edges, the condensed tree
      indptr      n_clusters + 1, the parent CSR (`Utils::parent_csr`)
      cpar        n_clusters, the parent of each cluster (cluster 0 points
                  at itself)
      kids        2 * n_clusters, the two cluster children of a cluster
                  (ascending), -1 when it has none (a cluster has 0 or 2)
      csize       n_clusters, the size of each cluster's edge
                  (`excess_of_mass`' cluster_sizes; cluster 0 is filled by
                  the selection, DEVIATION 1613)
      clam        n_clusters, the lambda of each cluster's own edge
      cdepth      n_clusters, the number of cluster ancestors
    """

    var n_leaves: Int
    var n_edges: Int
    var n_clusters: Int
    var max_cdepth: Int
    var parents: DeviceBuffer[DType.int32]
    var children: DeviceBuffer[DType.int32]
    var lambdas: DeviceBuffer[DType.float32]
    var sizes: DeviceBuffer[DType.int32]
    var indptr: DeviceBuffer[DType.int32]
    var cpar: DeviceBuffer[DType.int32]
    var kids: DeviceBuffer[DType.int32]
    var csize: DeviceBuffer[DType.int32]
    var clam: DeviceBuffer[DType.float32]
    var cdepth: DeviceBuffer[DType.int32]


@always_inline
def _size_of(x: Int, n: Int, sizes: I32P) -> Int:
    return 1 if x < n else Int(sizes[x - n])


def cd_init_kernel(
    par: I32P, wdep: I32P, wpre: I32P, live: I32P, side: I32P, n_nodes: Int32
):
    var v = _gid()
    if v >= Int(n_nodes):
        return
    par[v] = Int32(v)
    wdep[v] = 0
    wpre[v] = 0
    live[v] = 0
    side[v] = 0


def cd_merge_kernel(
    children: I32P,
    delta: F32P,
    sizes: I32P,
    par: I32P,
    wdep: I32P,
    wpre: I32P,
    live: I32P,
    side: I32P,
    lam: F32P,
    check: I32P,
    n_in: Int32,
    mcs: Int32,
    std_div: Int32,
):
    """Per merge i (node n + i): parent pointers, the path weights of both
    children, `live`'s own term, and `lambda = 1 / delta` (DEVIATION 1606).
    `check[0]` takes the largest child id (their `n_vertices` test),
    `check[1]` counts out-of-range children."""
    var i = _gid()
    var n = Int(n_in)
    if i >= n - 1:
        return
    var root = 2 * (n - 1)
    var l = Int(children[2 * i])
    var r = Int(children[2 * i + 1])
    _ = Atomic.max(check, Int32(max(l, r)))
    if l < 0 or r < 0 or l >= root or r >= root:
        _ = Atomic.fetch_add(check.unsafe_offset(1), Int32(1))
        return
    var node = n + i
    var sl = _size_of(l, n, sizes)
    par[l] = Int32(node)
    par[r] = Int32(node)
    wdep[l] = 1
    wdep[r] = 1
    wpre[l] = 1
    wpre[r] = Int32(2 * sl)
    side[l] = 0
    side[r] = 1
    live[node] = Int32(1) if Int(sizes[i]) >= Int(mcs) else Int32(0)
    var distance = delta[i]
    var lv = FLOAT32_MAX
    if distance > Float32(0.0):
        if std_div != 0:
            lv = Float32(1.0) / distance
        else:
            lv = identical_div(Float32(1.0), distance)
    lam[i] = lv


def cd_jump3_kernel(
    ps: I32P, ds: I32P, qs: I32P, ls: I32P,
    pd: I32P, dd: I32P, qd: I32P, ld: I32P,
    n: Int32,
):
    """One round over the root path: depth and preorder sums, `live` AND."""
    var v = _gid()
    if v >= Int(n):
        return
    var p = Int(ps[v])
    dd[v] = ds[v] + ds[p]
    qd[v] = qs[v] + qs[p]
    ld[v] = ls[v] & ls[p]
    pd[v] = ps[p]


def cd_split_flag_kernel(
    children: I32P, sizes: I32P, live: I32P, flag: I32P, n_in: Int32, mcs: Int32
):
    """flag[i] = merge i is a split (case 1 on a live node); flag[n-1] = 0."""
    var i = _gid()
    var n = Int(n_in)
    if i >= n:
        return
    if i == n - 1:
        flag[i] = 0
        return
    var l = Int(children[2 * i])
    var r = Int(children[2 * i + 1])
    var s = (
        live[n + i] != 0
        and _size_of(l, n, sizes) >= Int(mcs)
        and _size_of(r, n, sizes) >= Int(mcs)
    )
    flag[i] = Int32(1) if s else Int32(0)


def cd_split_compact_kernel(
    off: I32P, pre: I32P, keys: U32P, vals: U32P, n_in: Int32
):
    var i = _gid()
    var n = Int(n_in)
    if i >= n - 1:
        return
    var o = Int(off[i])
    if Int(off[i + 1]) == o:
        return
    keys[o] = UInt32(Int(pre[n + i]))
    vals[o] = UInt32(n + i)


def cd_gather_depth_kernel(vals: U32P, dep: I32P, keys: U32P, s: Int32):
    var j = _gid()
    if j >= Int(s):
        return
    keys[j] = UInt32(Int(dep[Int(vals[j])]))


def cd_rank_kernel(vals: U32P, rank: I32P, n_in: Int32, s: Int32):
    """rank[merge] = position of the split node in the walk's order."""
    var j = _gid()
    if j >= Int(s):
        return
    rank[Int(vals[j]) - Int(n_in)] = Int32(j)


def cd_label_kernel(
    par: I32P,
    side: I32P,
    live: I32P,
    off: I32P,
    rank: I32P,
    lab: I32P,
    ptrc: I32P,
    ptrl: I32P,
    n_in: Int32,
):
    """Start nodes get their label; every node points at itself if it is a
    start node (ptrc) / live (ptrl), else at its parent."""
    var v = _gid()
    var n = Int(n_in)
    var n_nodes = 2 * n - 1
    if v >= n_nodes:
        return
    var root = n_nodes - 1
    var p = Int(par[v])
    var label = -1
    if v == root:
        label = n
    else:
        var pi = p - n
        if pi >= 0 and Int(off[pi + 1]) - Int(off[pi]) == 1:
            label = n + 1 + 2 * Int(rank[pi]) + Int(side[v])
    lab[v] = Int32(label)
    ptrc[v] = Int32(v) if label >= 0 else Int32(p)
    ptrl[v] = Int32(v) if live[v] != 0 else Int32(p)


def cd_edges_kernel(
    par: I32P,
    side: I32P,
    sizes: I32P,
    lam: F32P,
    lab: I32P,
    ancc: I32P,
    ancl: I32P,
    ekey: U32P,
    eval: U32P,
    echild: I32P,
    elam: F32P,
    esize: I32P,
    count: I32P,
    cpar: I32P,
    kids: I32P,
    csize: I32P,
    clam: F32P,
    n_in: Int32,
):
    """One condensed edge per leaf and per start node below the root,
    written at its CHILD slot; the cluster-tree indexes beside it."""
    var v = _gid()
    var n = Int(n_in)
    var n_nodes = 2 * n - 1
    if v >= n_nodes:
        return
    var root = n_nodes - 1
    var e = -1
    var child = 0
    var plabel = 0
    var lv = Float32(0.0)
    var sz = 0
    if v < n:
        var big_l = Int(ancl[v])
        e = v
        child = v
        plabel = Int(lab[Int(ancc[v])])
        lv = lam[big_l - n]
        sz = 1
    elif v != root and Int(lab[v]) >= 0:
        var x = Int(lab[v])
        var p = Int(par[v])
        e = x - 1
        child = x
        plabel = Int(lab[Int(ancc[p])])
        lv = lam[p - n]
        sz = _size_of(v, n, sizes)
        var q = x - n
        var pq = plabel - n
        cpar[q] = Int32(pq)
        csize[q] = Int32(sz)
        clam[q] = lv
        kids[2 * pq + Int(side[v])] = Int32(q)
    if e < 0:
        return
    ekey[e] = UInt32(plabel - n)
    eval[e] = UInt32(e)
    echild[e] = Int32(child)
    elam[e] = lv
    esize[e] = Int32(sz)
    _ = Atomic.fetch_add(count.unsafe_offset(plabel - n), Int32(1))


def cd_gather_kernel(
    keys: U32P,
    vals: U32P,
    echild: I32P,
    elam: F32P,
    esize: I32P,
    parents: I32P,
    children: I32P,
    lambdas: F32P,
    sizes_out: I32P,
    n_in: Int32,
    n_edges: Int32,
):
    var j = _gid()
    if j >= Int(n_edges):
        return
    var e = Int(vals[j])
    parents[j] = Int32(Int(keys[j]) + Int(n_in))
    children[j] = echild[e]
    lambdas[j] = elam[e]
    sizes_out[j] = esize[e]


def cd_cluster_init_kernel(
    cpar: I32P, wdep: I32P, n_clusters: Int32
):
    """Cluster 0 points at itself; every cluster's depth term."""
    var q = _gid()
    if q >= Int(n_clusters):
        return
    if q == 0:
        cpar[0] = 0
        wdep[0] = 0
    else:
        wdep[q] = 1


def cd_max_kernel(vals: I32P, res: I32P, n: Int32):
    var q = _gid()
    if q >= Int(n):
        return
    _ = Atomic.max(res, vals[q])


def _refuse_nonfinite_device(
    ctx: DeviceContext,
    mut buf: DeviceBuffer[DType.float32],
    n: Int,
    what: String,
    sabotage: Int32,
) raises:
    """DEVIATION 1607 on a device array: the first NaN or infinity, found
    on the device."""
    if sabotage == HDB_SAB_SKIP_GUARDS:
        return
    var first = device_first_nonfinite(ctx, buf, n)
    if first >= 0:
        raise Error(
            "hdbscan.build_condensed_hierarchy: " + what + " hold a NaN or"
            " infinity, first at index " + String(first) + "; refused by"
            " name (DEVIATION 1607, IDENTITY_PATHS row 39). Note that"
            " lambda = FLT_MAX at delta == 0 is THEIR rule (condense.cuh"
            ":149) and is finite, so it is not what this refusal is about"
        )


# ---------------------------------------- lane af-hdbscan2, FAST Apple ---
# `-D MOJOLEARN_HDB_LINKAGE_DEVICE`: the same tree as `build_condensed_device`
# below (the same kernels, in the same order, on the same values) with the
# host's reads gathered into TWO status readbacks instead of eight waits:
#   read 1: max child id, out-of-range count, first non-finite delta, n_split
#   read 2: max cluster depth, first non-finite condensed lambda
# Main reads them as: device_first_nonfinite (1 wait), two td_read_i32 (2),
# n_split (1), a wait after the rank (1), max_cdepth (1),
# device_first_nonfinite (1), the closing synchronize (1).
# The refusals are main's, in main's order, with main's messages and the
# same first index (an integer Atomic.min over the flagged indices). Because
# read 1 comes after the depth walk and the split scan, those kernels also
# run on an invalid MST before it is refused: the merge kernel leaves every
# pointer in range, and `cdf_split_flag_kernel` is `cd_split_flag_kernel`
# with the children bounds-checked, so nothing reads out of range first.

comptime CDF_NONE = Int32(0x7FFFFFFF)


def cdf_first_nonfinite_kernel(buf: F32P, cell: I32P, n: Int32):
    """cell = min over i of the indices whose bits are NaN or infinity."""
    var i = _gid()
    if i >= Int(n):
        return
    var au = bitcast[DType.uint32](buf[i]) & UInt32(0x7FFFFFFF)
    if au >= UInt32(0x7F800000):
        _ = Atomic.min(cell, Int32(i))


def cdf_split_flag_kernel(
    children: I32P, sizes: I32P, live: I32P, flag: I32P, n_in: Int32, mcs: Int32
):
    """`cd_split_flag_kernel` with out-of-range children flagged 0 (they
    are refused by read 1 before anything consumes the flags)."""
    var i = _gid()
    var n = Int(n_in)
    if i >= n:
        return
    if i == n - 1:
        flag[i] = 0
        return
    var root = 2 * (n - 1)
    var l = Int(children[2 * i])
    var r = Int(children[2 * i + 1])
    if l < 0 or r < 0 or l >= root or r >= root:
        flag[i] = 0
        return
    var s = (
        live[n + i] != 0
        and _size_of(l, n, sizes) >= Int(mcs)
        and _size_of(r, n, sizes) >= Int(mcs)
    )
    flag[i] = Int32(1) if s else Int32(0)


def cdf_status1_kernel(
    check: I32P, bad: I32P, off_last: I32P, status: I32P
):
    """Four scalars into one status buffer (a copy, one thread; `off_last`
    points at the scan's last word, so no size reaches the launch)."""
    if _gid() != 0:
        return
    status[0] = check[0]
    status[1] = check[1]
    status[2] = bad[0]
    status[3] = off_last[0]


def cdf_status2_kernel(mx: I32P, bad: I32P, status: I32P):
    if _gid() != 0:
        return
    status[0] = mx[0]
    status[1] = bad[0]


def _condensed_two_reads(
    ctx: DeviceContext,
    mut children: DeviceBuffer[DType.int32],
    mut delta: DeviceBuffer[DType.float32],
    mut sizes: DeviceBuffer[DType.int32],
    min_cluster_size: Int,
    n_leaves: Int,
    sabotage: Int32,
) raises -> DeviceTree:
    """`build_condensed_device` with two status readbacks (block comment
    above). The caller has run the argument refusals."""
    var n = n_leaves
    var n_merges = n - 1
    var n_nodes = 2 * n - 1
    var root = n_nodes - 1

    var par = ctx.enqueue_create_buffer[DType.int32](n_nodes)
    var wdep = ctx.enqueue_create_buffer[DType.int32](n_nodes)
    var wpre = ctx.enqueue_create_buffer[DType.int32](n_nodes)
    var live = ctx.enqueue_create_buffer[DType.int32](n_nodes)
    var side = ctx.enqueue_create_buffer[DType.int32](n_nodes)
    var lam = ctx.enqueue_create_buffer[DType.float32](n_merges)
    var check = ctx.enqueue_create_buffer[DType.int32](2)
    var bad = ctx.enqueue_create_buffer[DType.int32](2)
    var status = ctx.enqueue_create_buffer[DType.int32](4)
    ctx.enqueue_memset(check, Int32(0))
    ctx.enqueue_memset(bad, CDF_NONE)
    ctx.enqueue_function[cdf_first_nonfinite_kernel](
        delta.unsafe_ptr(), bad.unsafe_ptr(), Int32(n_merges),
        grid_dim=(td_grid(n_merges), 1, 1), block_dim=(TD_TPB, 1, 1),
    )
    ctx.enqueue_function[cd_init_kernel](
        par.unsafe_ptr(), wdep.unsafe_ptr(), wpre.unsafe_ptr(),
        live.unsafe_ptr(), side.unsafe_ptr(), Int32(n_nodes),
        grid_dim=(td_grid(n_nodes), 1, 1), block_dim=(TD_TPB, 1, 1),
    )
    ctx.enqueue_function[cd_merge_kernel](
        children.unsafe_ptr(), delta.unsafe_ptr(), sizes.unsafe_ptr(),
        par.unsafe_ptr(), wdep.unsafe_ptr(), wpre.unsafe_ptr(),
        live.unsafe_ptr(), side.unsafe_ptr(), lam.unsafe_ptr(),
        check.unsafe_ptr(), Int32(n), Int32(min_cluster_size),
        Int32(1) if sabotage == HDB_SAB_LAMBDA_STD_DIV else Int32(0),
        grid_dim=(td_grid(n_merges), 1, 1), block_dim=(TD_TPB, 1, 1),
    )

    # depth, preorder and live over the root path (main's rounds).
    var par0 = ctx.enqueue_create_buffer[DType.int32](n_nodes)
    ctx.enqueue_copy(dst_buf=par0, src_buf=par)
    var pb = ctx.enqueue_create_buffer[DType.int32](n_nodes)
    var db = ctx.enqueue_create_buffer[DType.int32](n_nodes)
    var qb = ctx.enqueue_create_buffer[DType.int32](n_nodes)
    var lb = ctx.enqueue_create_buffer[DType.int32](n_nodes)
    var rounds = td_rounds(n_nodes)
    for k in range(rounds):
        if k % 2 == 0:
            ctx.enqueue_function[cd_jump3_kernel](
                par0.unsafe_ptr(), wdep.unsafe_ptr(), wpre.unsafe_ptr(),
                live.unsafe_ptr(), pb.unsafe_ptr(), db.unsafe_ptr(),
                qb.unsafe_ptr(), lb.unsafe_ptr(), Int32(n_nodes),
                grid_dim=(td_grid(n_nodes), 1, 1), block_dim=(TD_TPB, 1, 1),
            )
        else:
            ctx.enqueue_function[cd_jump3_kernel](
                pb.unsafe_ptr(), db.unsafe_ptr(), qb.unsafe_ptr(),
                lb.unsafe_ptr(), par0.unsafe_ptr(), wdep.unsafe_ptr(),
                wpre.unsafe_ptr(), live.unsafe_ptr(), Int32(n_nodes),
                grid_dim=(td_grid(n_nodes), 1, 1), block_dim=(TD_TPB, 1, 1),
            )
    if rounds % 2 == 1:
        ctx.enqueue_copy(dst_buf=wdep, src_buf=db)
        ctx.enqueue_copy(dst_buf=wpre, src_buf=qb)
        ctx.enqueue_copy(dst_buf=live, src_buf=lb)

    var off = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.enqueue_function[cdf_split_flag_kernel](
        children.unsafe_ptr(), sizes.unsafe_ptr(), live.unsafe_ptr(),
        off.unsafe_ptr(), Int32(n), Int32(min_cluster_size),
        grid_dim=(td_grid(n), 1, 1), block_dim=(TD_TPB, 1, 1),
    )
    td_exclusive_scan(ctx, off, n)

    # READ 1: the MST check, the delta refusal and n_split, one wait.
    ctx.enqueue_function[cdf_status1_kernel](
        check.unsafe_ptr(), bad.unsafe_ptr(), off.unsafe_ptr() + (n - 1),
        status.unsafe_ptr(),
        grid_dim=(1, 1, 1), block_dim=(1, 1, 1),
    )
    var h1 = ctx.enqueue_create_host_buffer[DType.int32](4)
    ctx.enqueue_copy(dst_ptr=h1.unsafe_ptr(), src_buf=status)
    ctx.synchronize()
    var first_bad_delta = Int(h1.unsafe_ptr()[2])
    if sabotage != HDB_SAB_SKIP_GUARDS and first_bad_delta != Int(CDF_NONE):
        raise Error(
            "hdbscan.build_condensed_hierarchy: dendrogram deltas hold a NaN or"
            " infinity, first at index " + String(first_bad_delta) + "; refused by"
            " name (DEVIATION 1607, IDENTITY_PATHS row 39). Note that"
            " lambda = FLT_MAX at delta == 0 is THEIR rule (condense.cuh"
            ":149) and is finite, so it is not what this refusal is about"
        )
    var n_vertices = Int(h1.unsafe_ptr()[0]) + 1
    var n_bad = Int(h1.unsafe_ptr()[1])
    if n_vertices != root or n_bad != 0:
        raise Error(
            "hdbscan.build_condensed_hierarchy: Multiple components found"
            " in MST or MST is invalid. Cannot find single-linkage"
            " solution. Found " + String(n_vertices) + " vertices total"
            " (expected " + String(root) + ")"
        )
    var n_split = Int(h1.unsafe_ptr()[3])
    _ = h1^

    var rank = ctx.enqueue_create_buffer[DType.int32](n_merges)
    ctx.enqueue_memset(rank, Int32(-1))
    if n_split > 0:
        var skeys = ctx.enqueue_create_buffer[DType.uint32](n_split)
        var svals = ctx.enqueue_create_buffer[DType.uint32](n_split)
        ctx.enqueue_function[cd_split_compact_kernel](
            off.unsafe_ptr(), wpre.unsafe_ptr(), skeys.unsafe_ptr(),
            svals.unsafe_ptr(), Int32(n),
            grid_dim=(td_grid(n), 1, 1), block_dim=(TD_TPB, 1, 1),
        )
        td_sort_pairs(ctx, skeys, svals, n_split)
        if sabotage != HDB_SAB_CONDENSE_DFS:
            ctx.enqueue_function[cd_gather_depth_kernel](
                svals.unsafe_ptr(), wdep.unsafe_ptr(), skeys.unsafe_ptr(),
                Int32(n_split),
                grid_dim=(td_grid(n_split), 1, 1), block_dim=(TD_TPB, 1, 1),
            )
            td_sort_pairs(ctx, skeys, svals, n_split)
        ctx.enqueue_function[cd_rank_kernel](
            svals.unsafe_ptr(), rank.unsafe_ptr(), Int32(n), Int32(n_split),
            grid_dim=(td_grid(n_split), 1, 1), block_dim=(TD_TPB, 1, 1),
        )
        # no wait: the scratch is released in queue order
        _ = skeys^
        _ = svals^

    var lab = ctx.enqueue_create_buffer[DType.int32](n_nodes)
    var ptrc = ctx.enqueue_create_buffer[DType.int32](n_nodes)
    var ptrl = ctx.enqueue_create_buffer[DType.int32](n_nodes)
    ctx.enqueue_function[cd_label_kernel](
        par.unsafe_ptr(), side.unsafe_ptr(), live.unsafe_ptr(),
        off.unsafe_ptr(), rank.unsafe_ptr(), lab.unsafe_ptr(),
        ptrc.unsafe_ptr(), ptrl.unsafe_ptr(), Int32(n),
        grid_dim=(td_grid(n_nodes), 1, 1), block_dim=(TD_TPB, 1, 1),
    )
    var in_b = td_find(ctx, ptrc, pb, n_nodes)
    if in_b:
        ctx.enqueue_copy(dst_buf=ptrc, src_buf=pb)
    in_b = td_find(ctx, ptrl, pb, n_nodes)
    if in_b:
        ctx.enqueue_copy(dst_buf=ptrl, src_buf=pb)

    var n_edges = n + 2 * n_split
    var n_clusters = 1 + 2 * n_split
    var ekey = ctx.enqueue_create_buffer[DType.uint32](n_edges)
    var evals = ctx.enqueue_create_buffer[DType.uint32](n_edges)
    var echild = ctx.enqueue_create_buffer[DType.int32](n_edges)
    var elam = ctx.enqueue_create_buffer[DType.float32](n_edges)
    var esize = ctx.enqueue_create_buffer[DType.int32](n_edges)
    var indptr = ctx.enqueue_create_buffer[DType.int32](n_clusters + 1)
    var cpar = ctx.enqueue_create_buffer[DType.int32](n_clusters)
    var kids = ctx.enqueue_create_buffer[DType.int32](2 * n_clusters)
    var csize = ctx.enqueue_create_buffer[DType.int32](n_clusters)
    var clam = ctx.enqueue_create_buffer[DType.float32](n_clusters)
    ctx.enqueue_memset(indptr, Int32(0))
    ctx.enqueue_memset(kids, Int32(-1))
    ctx.enqueue_memset(csize, Int32(0))
    ctx.enqueue_memset(clam, Float32(0.0))
    ctx.enqueue_function[cd_edges_kernel](
        par.unsafe_ptr(), side.unsafe_ptr(), sizes.unsafe_ptr(),
        lam.unsafe_ptr(), lab.unsafe_ptr(), ptrc.unsafe_ptr(),
        ptrl.unsafe_ptr(), ekey.unsafe_ptr(), evals.unsafe_ptr(),
        echild.unsafe_ptr(), elam.unsafe_ptr(), esize.unsafe_ptr(),
        indptr.unsafe_ptr(), cpar.unsafe_ptr(), kids.unsafe_ptr(),
        csize.unsafe_ptr(), clam.unsafe_ptr(), Int32(n),
        grid_dim=(td_grid(n_nodes), 1, 1), block_dim=(TD_TPB, 1, 1),
    )
    td_sort_pairs(ctx, ekey, evals, n_edges)
    var t_parents = ctx.enqueue_create_buffer[DType.int32](n_edges)
    var t_children = ctx.enqueue_create_buffer[DType.int32](n_edges)
    var t_lambdas = ctx.enqueue_create_buffer[DType.float32](n_edges)
    var t_sizes = ctx.enqueue_create_buffer[DType.int32](n_edges)
    ctx.enqueue_function[cd_gather_kernel](
        ekey.unsafe_ptr(), evals.unsafe_ptr(), echild.unsafe_ptr(),
        elam.unsafe_ptr(), esize.unsafe_ptr(), t_parents.unsafe_ptr(),
        t_children.unsafe_ptr(), t_lambdas.unsafe_ptr(),
        t_sizes.unsafe_ptr(), Int32(n), Int32(n_edges),
        grid_dim=(td_grid(n_edges), 1, 1), block_dim=(TD_TPB, 1, 1),
    )
    td_exclusive_scan(ctx, indptr, n_clusters + 1)

    var cdep = ctx.enqueue_create_buffer[DType.int32](n_clusters)
    ctx.enqueue_function[cd_cluster_init_kernel](
        cpar.unsafe_ptr(), cdep.unsafe_ptr(), Int32(n_clusters),
        grid_dim=(td_grid(n_clusters), 1, 1), block_dim=(TD_TPB, 1, 1),
    )
    var cp_a = ctx.enqueue_create_buffer[DType.int32](n_clusters)
    var cp_b = ctx.enqueue_create_buffer[DType.int32](n_clusters)
    var cd_b = ctx.enqueue_create_buffer[DType.int32](n_clusters)
    ctx.enqueue_copy(dst_buf=cp_a, src_buf=cpar)
    in_b = td_path_reduce[True](ctx, cp_a, cdep, cp_b, cd_b, n_clusters)
    if in_b:
        ctx.enqueue_copy(dst_buf=cdep, src_buf=cd_b)
    var mx = ctx.enqueue_create_buffer[DType.int32](1)
    ctx.enqueue_memset(mx, Int32(0))
    ctx.enqueue_function[cd_max_kernel](
        cdep.unsafe_ptr(), mx.unsafe_ptr(), Int32(n_clusters),
        grid_dim=(td_grid(n_clusters), 1, 1), block_dim=(TD_TPB, 1, 1),
    )

    # READ 2: max_cdepth and the lambda refusal, one wait.
    var bad2 = ctx.enqueue_create_buffer[DType.int32](1)
    ctx.enqueue_memset(bad2, CDF_NONE)
    ctx.enqueue_function[cdf_first_nonfinite_kernel](
        t_lambdas.unsafe_ptr(), bad2.unsafe_ptr(), Int32(n_edges),
        grid_dim=(td_grid(n_edges), 1, 1), block_dim=(TD_TPB, 1, 1),
    )
    ctx.enqueue_function[cdf_status2_kernel](
        mx.unsafe_ptr(), bad2.unsafe_ptr(), status.unsafe_ptr(),
        grid_dim=(1, 1, 1), block_dim=(1, 1, 1),
    )
    var h2 = ctx.enqueue_create_host_buffer[DType.int32](4)
    ctx.enqueue_copy(dst_ptr=h2.unsafe_ptr(), src_buf=status)
    ctx.synchronize()
    var max_cdepth = Int(h2.unsafe_ptr()[0])
    var first_bad_lam = Int(h2.unsafe_ptr()[1])
    _ = h2^
    if sabotage != HDB_SAB_SKIP_GUARDS and first_bad_lam != Int(CDF_NONE):
        raise Error(
            "hdbscan.build_condensed_hierarchy: condensed tree lambdas hold a NaN or"
            " infinity, first at index " + String(first_bad_lam) + "; refused by"
            " name (DEVIATION 1607, IDENTITY_PATHS row 39). Note that"
            " lambda = FLT_MAX at delta == 0 is THEIR rule (condense.cuh"
            ":149) and is finite, so it is not what this refusal is about"
        )
    _ = par^
    _ = wdep^
    _ = wpre^
    _ = live^
    _ = side^
    _ = lam^
    _ = check^
    _ = bad^
    _ = bad2^
    _ = status^
    _ = par0^
    _ = pb^
    _ = db^
    _ = qb^
    _ = lb^
    _ = off^
    _ = rank^
    _ = lab^
    _ = ptrc^
    _ = ptrl^
    _ = ekey^
    _ = evals^
    _ = echild^
    _ = elam^
    _ = esize^
    _ = cp_a^
    _ = cp_b^
    _ = cd_b^
    _ = mx^
    return DeviceTree(
        n, n_edges, n_clusters, max_cdepth, t_parents^, t_children^,
        t_lambdas^, t_sizes^, indptr^, cpar^, kids^, csize^, clam^, cdep^,
    )


def build_condensed_device(
    ctx: DeviceContext,
    mut children: DeviceBuffer[DType.int32],
    mut delta: DeviceBuffer[DType.float32],
    mut sizes: DeviceBuffer[DType.int32],
    min_cluster_size: Int,
    n_leaves: Int,
    sabotage: Int32 = HDB_SAB_NONE,
) raises -> DeviceTree:
    """`condense.cuh:237-286` on the device (this file's header). The
    refusals are `build_condensed_hierarchy`'s, in its order."""
    if n_leaves < 2:
        raise Error(
            "hdbscan.build_condensed_hierarchy: n_leaves=" + String(n_leaves)
            + " < 2 refused by name; their root index 2 * (n_leaves - 1)"
            " is not a node below that"
        )
    if min_cluster_size < 2:
        raise Error(
            "hdbscan.build_condensed_hierarchy: min_cluster_size="
            + String(min_cluster_size)
            + " refused by name; it must be at least 2. At 1 every leaf is"
            " its own cluster, their case-1 branch fires at every merge,"
            " and the condensed tree is the dendrogram with a different"
            " numbering -- an answer, but not HDBSCAN's"
        )
    if min_cluster_size > n_leaves:
        raise Error(
            "hdbscan.build_condensed_hierarchy: min_cluster_size="
            + String(min_cluster_size) + " > n_rows=" + String(n_leaves)
            + " refused by name; no subtree can reach that size, so every"
            " branch takes their case 2, nothing survives the size != -1"
            " filter and CondensedHierarchy.condense would read an empty"
            " minmax range"
        )
    # lane af-hdbscan2 (-D MOJOLEARN_HDB_LINKAGE_DEVICE): the same tree
    # with two status readbacks instead of eight waits.
    # fam2-cluster: the same route under IDENTICAL on every vendor
    # (IDN_HDB_CONDENSE_TWO_READS).
    comptime if HDB_LINKAGE_DEVICE or IDN_HDB_CONDENSE_TWO_READS:
        return _condensed_two_reads(
            ctx, children, delta, sizes, min_cluster_size, n_leaves, sabotage
        )
    var n = n_leaves
    var n_merges = n - 1
    var n_nodes = 2 * n - 1
    var root = n_nodes - 1

    _refuse_nonfinite_device(ctx, delta, n_merges, "dendrogram deltas", sabotage)

    var par = ctx.enqueue_create_buffer[DType.int32](n_nodes)
    var wdep = ctx.enqueue_create_buffer[DType.int32](n_nodes)
    var wpre = ctx.enqueue_create_buffer[DType.int32](n_nodes)
    var live = ctx.enqueue_create_buffer[DType.int32](n_nodes)
    var side = ctx.enqueue_create_buffer[DType.int32](n_nodes)
    var lam = ctx.enqueue_create_buffer[DType.float32](n_merges)
    var check = ctx.enqueue_create_buffer[DType.int32](2)
    ctx.enqueue_memset(check, Int32(0))
    ctx.enqueue_function[cd_init_kernel](
        par.unsafe_ptr(), wdep.unsafe_ptr(), wpre.unsafe_ptr(),
        live.unsafe_ptr(), side.unsafe_ptr(), Int32(n_nodes),
        grid_dim=(td_grid(n_nodes), 1, 1), block_dim=(TD_TPB, 1, 1),
    )
    ctx.enqueue_function[cd_merge_kernel](
        children.unsafe_ptr(), delta.unsafe_ptr(), sizes.unsafe_ptr(),
        par.unsafe_ptr(), wdep.unsafe_ptr(), wpre.unsafe_ptr(),
        live.unsafe_ptr(), side.unsafe_ptr(), lam.unsafe_ptr(),
        check.unsafe_ptr(), Int32(n), Int32(min_cluster_size),
        Int32(1) if sabotage == HDB_SAB_LAMBDA_STD_DIV else Int32(0),
        grid_dim=(td_grid(n_merges), 1, 1), block_dim=(TD_TPB, 1, 1),
    )
    # `:252-261` n_vertices = max(children) + 1 == root, their message.
    var n_vertices = td_read_i32(ctx, check, 0) + 1
    var n_bad = td_read_i32(ctx, check, 1)
    if n_vertices != root or n_bad != 0:
        raise Error(
            "hdbscan.build_condensed_hierarchy: Multiple components found"
            " in MST or MST is invalid. Cannot find single-linkage"
            " solution. Found " + String(n_vertices) + " vertices total"
            " (expected " + String(root) + ")"
        )

    # depth, preorder and live over the root path.
    var par0 = ctx.enqueue_create_buffer[DType.int32](n_nodes)
    ctx.enqueue_copy(dst_buf=par0, src_buf=par)
    var pb = ctx.enqueue_create_buffer[DType.int32](n_nodes)
    var db = ctx.enqueue_create_buffer[DType.int32](n_nodes)
    var qb = ctx.enqueue_create_buffer[DType.int32](n_nodes)
    var lb = ctx.enqueue_create_buffer[DType.int32](n_nodes)
    var rounds = td_rounds(n_nodes)
    for k in range(rounds):
        if k % 2 == 0:
            ctx.enqueue_function[cd_jump3_kernel](
                par0.unsafe_ptr(), wdep.unsafe_ptr(), wpre.unsafe_ptr(),
                live.unsafe_ptr(), pb.unsafe_ptr(), db.unsafe_ptr(),
                qb.unsafe_ptr(), lb.unsafe_ptr(), Int32(n_nodes),
                grid_dim=(td_grid(n_nodes), 1, 1), block_dim=(TD_TPB, 1, 1),
            )
        else:
            ctx.enqueue_function[cd_jump3_kernel](
                pb.unsafe_ptr(), db.unsafe_ptr(), qb.unsafe_ptr(),
                lb.unsafe_ptr(), par0.unsafe_ptr(), wdep.unsafe_ptr(),
                wpre.unsafe_ptr(), live.unsafe_ptr(), Int32(n_nodes),
                grid_dim=(td_grid(n_nodes), 1, 1), block_dim=(TD_TPB, 1, 1),
            )
    if rounds % 2 == 1:
        ctx.enqueue_copy(dst_buf=wdep, src_buf=db)
        ctx.enqueue_copy(dst_buf=wpre, src_buf=qb)
        ctx.enqueue_copy(dst_buf=live, src_buf=lb)
    # wdep = depth, wpre = preorder index, live = live, from here on.

    # the split nodes, compacted, ranked in the walk's order.
    var off = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.enqueue_function[cd_split_flag_kernel](
        children.unsafe_ptr(), sizes.unsafe_ptr(), live.unsafe_ptr(),
        off.unsafe_ptr(), Int32(n), Int32(min_cluster_size),
        grid_dim=(td_grid(n), 1, 1), block_dim=(TD_TPB, 1, 1),
    )
    td_exclusive_scan(ctx, off, n)
    var n_split = td_read_i32(ctx, off, n - 1)
    var rank = ctx.enqueue_create_buffer[DType.int32](n_merges)
    ctx.enqueue_memset(rank, Int32(-1))
    if n_split > 0:
        var skeys = ctx.enqueue_create_buffer[DType.uint32](n_split)
        var svals = ctx.enqueue_create_buffer[DType.uint32](n_split)
        ctx.enqueue_function[cd_split_compact_kernel](
            off.unsafe_ptr(), wpre.unsafe_ptr(), skeys.unsafe_ptr(),
            svals.unsafe_ptr(), Int32(n),
            grid_dim=(td_grid(n), 1, 1), block_dim=(TD_TPB, 1, 1),
        )
        td_sort_pairs(ctx, skeys, svals, n_split)
        if sabotage != HDB_SAB_CONDENSE_DFS:
            ctx.enqueue_function[cd_gather_depth_kernel](
                svals.unsafe_ptr(), wdep.unsafe_ptr(), skeys.unsafe_ptr(),
                Int32(n_split),
                grid_dim=(td_grid(n_split), 1, 1), block_dim=(TD_TPB, 1, 1),
            )
            td_sort_pairs(ctx, skeys, svals, n_split)
        ctx.enqueue_function[cd_rank_kernel](
            svals.unsafe_ptr(), rank.unsafe_ptr(), Int32(n), Int32(n_split),
            grid_dim=(td_grid(n_split), 1, 1), block_dim=(TD_TPB, 1, 1),
        )
        ctx.synchronize()
        _ = skeys^
        _ = svals^

    # labels, and the start / live ancestors of every node.
    var lab = ctx.enqueue_create_buffer[DType.int32](n_nodes)
    var ptrc = ctx.enqueue_create_buffer[DType.int32](n_nodes)
    var ptrl = ctx.enqueue_create_buffer[DType.int32](n_nodes)
    ctx.enqueue_function[cd_label_kernel](
        par.unsafe_ptr(), side.unsafe_ptr(), live.unsafe_ptr(),
        off.unsafe_ptr(), rank.unsafe_ptr(), lab.unsafe_ptr(),
        ptrc.unsafe_ptr(), ptrl.unsafe_ptr(), Int32(n),
        grid_dim=(td_grid(n_nodes), 1, 1), block_dim=(TD_TPB, 1, 1),
    )
    var in_b = td_find(ctx, ptrc, pb, n_nodes)
    if in_b:
        ctx.enqueue_copy(dst_buf=ptrc, src_buf=pb)
    in_b = td_find(ctx, ptrl, pb, n_nodes)
    if in_b:
        ctx.enqueue_copy(dst_buf=ptrl, src_buf=pb)

    # the edges at their child slots, then one stable sort by parent.
    var n_edges = n + 2 * n_split
    var n_clusters = 1 + 2 * n_split
    var ekey = ctx.enqueue_create_buffer[DType.uint32](n_edges)
    var evals = ctx.enqueue_create_buffer[DType.uint32](n_edges)
    var echild = ctx.enqueue_create_buffer[DType.int32](n_edges)
    var elam = ctx.enqueue_create_buffer[DType.float32](n_edges)
    var esize = ctx.enqueue_create_buffer[DType.int32](n_edges)
    var indptr = ctx.enqueue_create_buffer[DType.int32](n_clusters + 1)
    var cpar = ctx.enqueue_create_buffer[DType.int32](n_clusters)
    var kids = ctx.enqueue_create_buffer[DType.int32](2 * n_clusters)
    var csize = ctx.enqueue_create_buffer[DType.int32](n_clusters)
    var clam = ctx.enqueue_create_buffer[DType.float32](n_clusters)
    ctx.enqueue_memset(indptr, Int32(0))
    ctx.enqueue_memset(kids, Int32(-1))
    ctx.enqueue_memset(csize, Int32(0))
    ctx.enqueue_memset(clam, Float32(0.0))
    ctx.enqueue_function[cd_edges_kernel](
        par.unsafe_ptr(), side.unsafe_ptr(), sizes.unsafe_ptr(),
        lam.unsafe_ptr(), lab.unsafe_ptr(), ptrc.unsafe_ptr(),
        ptrl.unsafe_ptr(), ekey.unsafe_ptr(), evals.unsafe_ptr(),
        echild.unsafe_ptr(), elam.unsafe_ptr(), esize.unsafe_ptr(),
        indptr.unsafe_ptr(), cpar.unsafe_ptr(), kids.unsafe_ptr(),
        csize.unsafe_ptr(), clam.unsafe_ptr(), Int32(n),
        grid_dim=(td_grid(n_nodes), 1, 1), block_dim=(TD_TPB, 1, 1),
    )
    td_sort_pairs(ctx, ekey, evals, n_edges)
    var t_parents = ctx.enqueue_create_buffer[DType.int32](n_edges)
    var t_children = ctx.enqueue_create_buffer[DType.int32](n_edges)
    var t_lambdas = ctx.enqueue_create_buffer[DType.float32](n_edges)
    var t_sizes = ctx.enqueue_create_buffer[DType.int32](n_edges)
    ctx.enqueue_function[cd_gather_kernel](
        ekey.unsafe_ptr(), evals.unsafe_ptr(), echild.unsafe_ptr(),
        elam.unsafe_ptr(), esize.unsafe_ptr(), t_parents.unsafe_ptr(),
        t_children.unsafe_ptr(), t_lambdas.unsafe_ptr(),
        t_sizes.unsafe_ptr(), Int32(n), Int32(n_edges),
        grid_dim=(td_grid(n_edges), 1, 1), block_dim=(TD_TPB, 1, 1),
    )
    td_exclusive_scan(ctx, indptr, n_clusters + 1)

    # cluster depths over the cluster tree.
    var cdep = ctx.enqueue_create_buffer[DType.int32](n_clusters)
    ctx.enqueue_function[cd_cluster_init_kernel](
        cpar.unsafe_ptr(), cdep.unsafe_ptr(), Int32(n_clusters),
        grid_dim=(td_grid(n_clusters), 1, 1), block_dim=(TD_TPB, 1, 1),
    )
    var cp_a = ctx.enqueue_create_buffer[DType.int32](n_clusters)
    var cp_b = ctx.enqueue_create_buffer[DType.int32](n_clusters)
    var cd_b = ctx.enqueue_create_buffer[DType.int32](n_clusters)
    ctx.enqueue_copy(dst_buf=cp_a, src_buf=cpar)
    in_b = td_path_reduce[True](ctx, cp_a, cdep, cp_b, cd_b, n_clusters)
    if in_b:
        ctx.enqueue_copy(dst_buf=cdep, src_buf=cd_b)
    var mx = ctx.enqueue_create_buffer[DType.int32](1)
    ctx.enqueue_memset(mx, Int32(0))
    ctx.enqueue_function[cd_max_kernel](
        cdep.unsafe_ptr(), mx.unsafe_ptr(), Int32(n_clusters),
        grid_dim=(td_grid(n_clusters), 1, 1), block_dim=(TD_TPB, 1, 1),
    )
    var max_cdepth = td_read_i32(ctx, mx, 0)

    _refuse_nonfinite_device(
        ctx, t_lambdas, n_edges, "condensed tree lambdas", sabotage
    )
    ctx.synchronize()
    _ = par^
    _ = wdep^
    _ = wpre^
    _ = live^
    _ = side^
    _ = lam^
    _ = check^
    _ = par0^
    _ = pb^
    _ = db^
    _ = qb^
    _ = lb^
    _ = off^
    _ = rank^
    _ = lab^
    _ = ptrc^
    _ = ptrl^
    _ = ekey^
    _ = evals^
    _ = echild^
    _ = elam^
    _ = esize^
    _ = cp_a^
    _ = cp_b^
    _ = cd_b^
    _ = mx^
    return DeviceTree(
        n, n_edges, n_clusters, max_cdepth, t_parents^, t_children^,
        t_lambdas^, t_sizes^, indptr^, cpar^, kids^, csize^, clam^, cdep^,
    )
