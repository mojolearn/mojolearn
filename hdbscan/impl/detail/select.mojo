# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Cluster selection: Excess of Mass, Leaf, and the negation BFS.

Reference: `cuml-v26.08.00/cpp/src/hdbscan/detail/select.cuh`
(cuML `265b9da`): `perform_bfs` (`:57-91`), `excess_of_mass` (`:148-252`),
`leaf` (`:264-286`) and `select_clusters` (`:379-452`), plus
`detail/kernels/select.cuh::propagate_cluster_negation_kernel`
(`:24-45`) and `cluster_epsilon_search` (`:301-363`, kernel `:47-104`), all
on the device over `tree_device.mojo::DeviceTree` (lane cgr2-hdbscan,
2026-10-03). The CPU column (`hdbscan_host_oracle.mojo::hdbh_select`, with
`utils.mojo`'s cluster tree, parent CSRs and `cluster_epsilon_search_host`)
keeps the host loops; both give the same selection cell for cell.

EXCESS OF MASS IS THE DEFAULT AND IS THE ONE THIS LANE SHIPS
(`hdbscan.hpp:197`, `cluster_selection_method = EOM`). LEAF is implemented too
because it is nine lines of integer work and CONTRIBUTING.md (Non-default paths) is
explicit that a switch with an unexercised side is an unchecked path:
`check_hdbscan_selection_leaf` runs it; `cluster_selection_epsilon` is run by
the verifier lane x-cluster-hdbscan-epsilon.

======================================================================
DEVIATION BLOCK -- DEVIATION 1605. THE EXCESS-OF-MASS LOOP RUNS LEVEL BY
LEVEL ON THE DEVICE, AND ITS SUBTREE SUM IS A SERIAL ASCENDING FOLD OVER
A CLUSTER'S TWO CHILDREN.
======================================================================
WHAT THEIRS DOES (`select.cuh:205-233`). A host `for` loop from
`n_clusters - 1` down to `tree_top`, and INSIDE it, per node:
    raft::update_host(&node_stability, stability + node, 1, stream);   // :210
    subtree_stability = thrust::transform_reduce(exec_policy,
        children + indptr_h[node], children + indptr_h[node + 1],
        [=] __device__(value_idx a) { return stability[a]; },
        0.0, plus<value_t>());                                        // :215-222
    if (subtree_stability > node_stability || cluster_sizes_h[node] > max_cluster_size) {
      raft::update_device(stability + node, &subtree_stability, 1, stream);
      is_cluster_h[node] = false;                                     // :225-228
    } else frontier_h[node] = true;

WHY NOT AS-IS. `thrust::transform_reduce` is a tree reduction whose shape
is the library's, and the sum decides a BOOLEAN that decides a CLUSTER
(IDENTITY_PATHS rows 7 and 20).

WHAT OURS DOES (lane cgr2-hdbscan, 2026-10-03; until then the loop ran on
the host over one download). A node's decision reads only its children's
FINAL stabilities, and every child id is larger than its parent's, so the
descending loop is the same computation as a sweep over the cluster tree's
levels from the deepest up: one launch per level, one thread per cluster
of that level, each reading its children (written by the previous launch)
and writing only its own cells. Every cluster has zero or two cluster
children (a split makes exactly two), so the subtree sum is the two-term
serial ascending fold `ftz(identical_mul_add(1, stab[child], acc))` the
host loop runs, the same bits. The host column (`hdbh_select`) keeps the
descending loop; the two agree cell for cell.

THE WRITE-BACK IS LOAD BEARING. `stability[node]` is OVERWRITTEN with the
subtree total when a node is deselected (`:227`), and the parent's level
runs after, so it reads the updated value; `HDB_SAB_EOM_NO_UPDATE` drops
the write-back and must move the selection.

THE NEGATION (`perform_bfs`, `:239-251`) deselects every strict descendant
of a frontier cluster. Here it is an OR over each cluster's ancestor path
by pointer jumping (`tree_device.mojo::td_path_reduce`), a fixed number of
grid-wide rounds with no host count between them; every write is a
constant, so no order is chosen.
======================================================================

======================================================================
DEVIATION BLOCK -- DEVIATION 1613. `cluster_sizes[0]` IS A DATA RACE IN
THEIR KERNEL AND AN EXACT SUM IN OURS.
======================================================================
WHAT THEIRS DOES (`select.cuh:173-182`), one thread per cluster-tree
edge:
    if (get<0>(tup) == 0) cluster_sizes_ptr[0] += get<2>(tup);
    cluster_sizes_ptr[cuda::std::get<1>(tup)] = get<2>(tup);
The FIRST line is a NON-ATOMIC read-modify-write into ONE cell from every
thread whose parent is the root, so with two root children the value it
leaves is between one child's size and the sum. `cluster_sizes_h[node]` is
compared against `max_cluster_size` (`:225`), so the race can decide
whether the ROOT is deselected. We fix their bug rather than implement it.

WHAT OURS DOES. `csize[q]` is each cluster's edge size (written by the
condense, `tree_device.mojo`), and `csize[0]` is the exact integer sum of
the root's two cluster children, written by one thread in `eom_init_kernel`.
`check_hdbscan_selection_eom` asserts it against a host oracle.
======================================================================

"""

from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from hdbscan.checks.hdbscan_sabotage import (
    HDB_SAB_EOM_NO_UPDATE,
    HDB_SAB_NONE,
)
from hdbscan.impl.detail.tree_device import (
    DeviceTree,
    TD_TPB,
    td_exclusive_scan,
    td_find,
    td_grid,
    td_path_reduce,
    td_read_i32,
)
from checks.numerics import ftz, identical_div, identical_mul_add

comptime SELECT_TPB = 256
"""Their `int tpb = 256` template default (`select.cuh:57`, `:148`,
`:264`). SCHEDULING: every kernel here writes integers or one cell per
thread."""

comptime CLUSTER_SELECTION_EOM = 0
comptime CLUSTER_SELECTION_LEAF = 1
"""`hdbscan.hpp:126` `enum CLUSTER_SELECTION_METHOD { EOM = 0, LEAF = 1 }`."""

comptime I32P = MutPointer[Int32, MutAnyOrigin]
comptime F32P = MutPointer[Float32, MutAnyOrigin]


@always_inline
def _gid() -> Int:
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


def eom_init_kernel(
    isc: I32P, fr: I32P, csize: I32P, kids: I32P, allow: Int32, n: Int32
):
    """`:189-193` is_cluster all true, the root = allow_single_cluster;
    frontier all false. Cluster 0's size is its two children's sum
    (DEVIATION 1613)."""
    var q = _gid()
    if q >= Int(n):
        return
    isc[q] = allow if q == 0 else Int32(1)
    fr[q] = 0
    if q == 0:
        var k0 = Int(kids[0])
        if k0 >= 0:
            csize[0] = csize[k0] + csize[Int(kids[1])]
        else:
            csize[0] = 0


def eom_level_kernel(
    stab: F32P,
    isc: I32P,
    fr: I32P,
    kids: I32P,
    csize: I32P,
    cdepth: I32P,
    level: Int32,
    tree_top: Int32,
    max_cluster_size: Int32,
    no_update: Int32,
    n: Int32,
):
    """`:206-233` for every cluster of one level (DEVIATION 1605)."""
    var q = _gid()
    if q >= Int(n) or q < Int(tree_top) or cdepth[q] != level:
        return
    var node_stability = stab[q]
    var subtree = Float32(0.0)
    var k0 = Int(kids[2 * q])
    if k0 >= 0:
        subtree = ftz(identical_mul_add(Float32(1.0), stab[k0], subtree))
        var k1 = Int(kids[2 * q + 1])
        subtree = ftz(identical_mul_add(Float32(1.0), stab[k1], subtree))
    if subtree > node_stability or Int(csize[q]) > Int(max_cluster_size):
        # `:225-228` Deselect / merge cluster with children
        if no_update == 0:
            stab[q] = subtree
        isc[q] = 0
    else:
        # `:231` Mark children to be deselected
        fr[q] = 1


def negate_init_kernel(cpar: I32P, fr: I32P, pa: I32P, va: I32P, n: Int32):
    """The OR over the strict ancestors: va[q] covers (q, pa[q]]."""
    var q = _gid()
    if q >= Int(n):
        return
    var p = Int(cpar[q])
    pa[q] = Int32(p)
    va[q] = 0 if q == 0 else fr[p]


def negate_apply_kernel(va: I32P, isc: I32P, n: Int32):
    var q = _gid()
    if q >= Int(n):
        return
    if va[q] != 0:
        isc[q] = 0


def leaf_kernel(kids: I32P, isc: I32P, n: Int32):
    """`select.cuh:264-286`: a cluster is selected iff it is a child (not
    the root) and the parent of none."""
    var q = _gid()
    if q >= Int(n):
        return
    isc[q] = Int32(1) if (q > 0 and Int(kids[2 * q]) < 0) else Int32(0)


def eps_init_kernel(
    cpar: I32P,
    clam: F32P,
    eps: F32P,
    pa: I32P,
    pt: I32P,
    epsilon: Float32,
    n: Int32,
):
    """`select.cuh:330-334` eps = 1 / lambda (`identical_div`), and the two
    pointer arrays the upward walk becomes: `pa` stops at the root or at a
    cluster whose eps is not `<=` epsilon, `pt` at a child of the root."""
    var q = _gid()
    if q >= Int(n):
        return
    if q == 0:
        eps[0] = 0.0
        pa[0] = 0
        pt[0] = 0
        return
    var e = identical_div(Float32(1.0), clam[q])
    eps[q] = e
    var p = cpar[q]
    pa[q] = Int32(q) if not (e <= epsilon) else p
    pt[q] = Int32(q) if Int(p) == 0 else p


def eps_apply_kernel(
    sel: I32P,
    isc: I32P,
    fr: I32P,
    cpar: I32P,
    eps: F32P,
    ra: I32P,
    rt: I32P,
    epsilon: Float32,
    allow: Int32,
    n: Int32,
):
    """`kernels/select.cuh:47-104` per selected cluster (`:66` the root
    takes no part): walk up while the parent's eps is `<=` epsilon and
    select where it stops (the root only under allow_single_cluster, else
    the root's child on the path), or go on the frontier."""
    var q = _gid()
    if q >= Int(n) or q == 0 or sel[q] == 0:
        return
    if eps[q] < epsilon:
        var t = Int(ra[Int(cpar[q])])
        if t == 0 and allow == 0:
            t = Int(rt[q])
        fr[t] = 1
        isc[t] = 1
    else:
        fr[q] = 1


def flag_kernel(isc: I32P, flag: I32P, n: Int32):
    """flag[q] = isc[q] != 0 over n + 1 slots (the last one 0)."""
    var q = _gid()
    if q > Int(n):
        return
    flag[q] = 0 if q == Int(n) else (Int32(1) if isc[q] != 0 else Int32(0))


def label_map_kernel(
    isc: I32P, off: I32P, label_map: I32P, inverse: I32P, n: Int32
):
    """`extract.cuh:286-295`: ascending condensed id -> final label."""
    var q = _gid()
    if q >= Int(n):
        return
    if isc[q] != 0:
        var o = off[q]
        label_map[q] = o
        inverse[Int(o)] = Int32(q)
    else:
        label_map[q] = -1


def _negate(
    ctx: DeviceContext,
    mut tree: DeviceTree,
    mut fr: DeviceBuffer[DType.int32],
    mut isc: DeviceBuffer[DType.int32],
) raises:
    """`perform_bfs` (`select.cuh:57-91`): every strict descendant of a
    frontier cluster is deselected."""
    var n = tree.n_clusters
    var pa = ctx.enqueue_create_buffer[DType.int32](n)
    var va = ctx.enqueue_create_buffer[DType.int32](n)
    var pb = ctx.enqueue_create_buffer[DType.int32](n)
    var vb = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.enqueue_function[negate_init_kernel](
        tree.cpar.unsafe_ptr(), fr.unsafe_ptr(), pa.unsafe_ptr(),
        va.unsafe_ptr(), Int32(n),
        grid_dim=(td_grid(n), 1, 1), block_dim=(TD_TPB, 1, 1),
    )
    var in_b = td_path_reduce[False](ctx, pa, va, pb, vb, n)
    var vptr = vb.unsafe_ptr() if in_b else va.unsafe_ptr()
    ctx.enqueue_function[negate_apply_kernel](
        vptr, isc.unsafe_ptr(), Int32(n),
        grid_dim=(td_grid(n), 1, 1), block_dim=(TD_TPB, 1, 1),
    )
    ctx.synchronize()
    _ = pa^
    _ = va^
    _ = pb^
    _ = vb^


def count_selected(
    ctx: DeviceContext,
    mut isc: DeviceBuffer[DType.int32],
    mut off: DeviceBuffer[DType.int32],
    n: Int,
) raises -> Int:
    """The exclusive scan of the selection into `off` (n + 1 slots);
    returns the count (`:412`, an integer sum)."""
    ctx.enqueue_function[flag_kernel](
        isc.unsafe_ptr(), off.unsafe_ptr(), Int32(n),
        grid_dim=(td_grid(n + 1), 1, 1), block_dim=(TD_TPB, 1, 1),
    )
    td_exclusive_scan(ctx, off, n + 1)
    return td_read_i32(ctx, off, n)


def excess_of_mass(
    ctx: DeviceContext,
    mut tree: DeviceTree,
    mut stability: DeviceBuffer[DType.float32],
    mut is_cluster: DeviceBuffer[DType.int32],
    max_cluster_size: Int,
    allow_single_cluster: Bool,
    sabotage: Int32 = HDB_SAB_NONE,
) raises:
    """`select.cuh:148-252`, DEVIATIONS 1605 and 1613."""
    var n = tree.n_clusters
    var fr = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.enqueue_function[eom_init_kernel](
        is_cluster.unsafe_ptr(), fr.unsafe_ptr(), tree.csize.unsafe_ptr(),
        tree.kids.unsafe_ptr(),
        Int32(1) if allow_single_cluster else Int32(0), Int32(n),
        grid_dim=(td_grid(n), 1, 1), block_dim=(TD_TPB, 1, 1),
    )
    # `:206-233` reverse topological order, one level per launch.
    var tree_top = 0 if allow_single_cluster else 1
    var level = tree.max_cdepth
    while level >= 0:
        ctx.enqueue_function[eom_level_kernel](
            stability.unsafe_ptr(), is_cluster.unsafe_ptr(), fr.unsafe_ptr(),
            tree.kids.unsafe_ptr(), tree.csize.unsafe_ptr(),
            tree.cdepth.unsafe_ptr(), Int32(level), Int32(tree_top),
            Int32(max_cluster_size),
            Int32(1) if sabotage == HDB_SAB_EOM_NO_UPDATE else Int32(0),
            Int32(n),
            grid_dim=(td_grid(n), 1, 1), block_dim=(TD_TPB, 1, 1),
        )
        level -= 1
    # `:239-251` propagate the deselection through the subtrees.
    _negate(ctx, tree, fr, is_cluster)
    _ = fr^


def cluster_epsilon_search(
    ctx: DeviceContext,
    mut tree: DeviceTree,
    mut is_cluster: DeviceBuffer[DType.int32],
    cluster_selection_epsilon: Float32,
    allow_single_cluster: Bool,
) raises:
    """`select.cuh:301-363` + `kernels/select.cuh:47-104` on the device
    (DEVIATION 5115's host function, `utils.mojo::cluster_epsilon_search_host`,
    stays the CPU column's): the upward walk is two pointer-jumping finds,
    the descendant deselection is `_negate`. Every write is a constant."""
    var n = tree.n_clusters
    var sel = ctx.enqueue_create_buffer[DType.int32](n)
    var fr = ctx.enqueue_create_buffer[DType.int32](n)
    var eps = ctx.enqueue_create_buffer[DType.float32](n)
    var pa = ctx.enqueue_create_buffer[DType.int32](n)
    var pt = ctx.enqueue_create_buffer[DType.int32](n)
    var tmp = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.enqueue_copy(dst_buf=sel, src_buf=is_cluster)
    ctx.enqueue_memset(fr, Int32(0))
    ctx.enqueue_function[eps_init_kernel](
        tree.cpar.unsafe_ptr(), tree.clam.unsafe_ptr(), eps.unsafe_ptr(),
        pa.unsafe_ptr(), pt.unsafe_ptr(), cluster_selection_epsilon, Int32(n),
        grid_dim=(td_grid(n), 1, 1), block_dim=(TD_TPB, 1, 1),
    )
    if td_find(ctx, pa, tmp, n):
        ctx.enqueue_copy(dst_buf=pa, src_buf=tmp)
    if td_find(ctx, pt, tmp, n):
        ctx.enqueue_copy(dst_buf=pt, src_buf=tmp)
    ctx.enqueue_function[eps_apply_kernel](
        sel.unsafe_ptr(), is_cluster.unsafe_ptr(), fr.unsafe_ptr(),
        tree.cpar.unsafe_ptr(), eps.unsafe_ptr(), pa.unsafe_ptr(),
        pt.unsafe_ptr(), cluster_selection_epsilon,
        Int32(1) if allow_single_cluster else Int32(0), Int32(n),
        grid_dim=(td_grid(n), 1, 1), block_dim=(TD_TPB, 1, 1),
    )
    _negate(ctx, tree, fr, is_cluster)
    _ = sel^
    _ = fr^
    _ = eps^
    _ = pa^
    _ = pt^
    _ = tmp^


def select_clusters(
    ctx: DeviceContext,
    mut tree: DeviceTree,
    mut tree_stabilities: DeviceBuffer[DType.float32],
    mut is_cluster: DeviceBuffer[DType.int32],
    mut label_map: DeviceBuffer[DType.int32],
    mut inverse_label_map: DeviceBuffer[DType.int32],
    cluster_selection_method: Int,
    allow_single_cluster: Bool,
    max_cluster_size: Int,
    cluster_selection_epsilon: Float32,
    sabotage: Int32 = HDB_SAB_NONE,
) raises -> Int:
    """`select.cuh:379-452`, then `extract.cuh:286-295`'s label maps.
    Returns `n_selected_clusters`. `inverse_label_map` holds n_clusters
    slots; the first n_selected are written."""
    var n = tree.n_clusters
    if cluster_selection_method == CLUSTER_SELECTION_EOM:
        excess_of_mass(
            ctx, tree, tree_stabilities, is_cluster, max_cluster_size,
            allow_single_cluster, sabotage,
        )
    elif cluster_selection_method == CLUSTER_SELECTION_LEAF:
        ctx.enqueue_function[leaf_kernel](
            tree.kids.unsafe_ptr(), is_cluster.unsafe_ptr(), Int32(n),
            grid_dim=(td_grid(n), 1, 1), block_dim=(TD_TPB, 1, 1),
        )
    else:
        raise Error(
            "hdbscan.select_clusters: cluster_selection_method="
            + String(cluster_selection_method)
            + " refused by name; their enum has exactly two values, EOM=0"
            " and LEAF=1 (hdbscan.hpp:126)"
        )
    var off = ctx.enqueue_create_buffer[DType.int32](n + 1)
    var n_selected = count_selected(ctx, is_cluster, off, n)

    # `:429-451` the epsilon search. The cluster tree has edges iff n > 1.
    if cluster_selection_epsilon != Float32(0.0) and n > 1:
        var epsilon_search = n_selected != 0
        # `:435-441` this is to check when eom finds root as only cluster
        if (
            cluster_selection_method == CLUSTER_SELECTION_EOM
            and n_selected == 1
            and allow_single_cluster
        ):
            if td_read_i32(ctx, is_cluster, 0) != 0:
                epsilon_search = False
        if epsilon_search:
            cluster_epsilon_search(
                ctx, tree, is_cluster, cluster_selection_epsilon,
                allow_single_cluster,
            )
            n_selected = count_selected(ctx, is_cluster, off, n)
    ctx.enqueue_function[label_map_kernel](
        is_cluster.unsafe_ptr(), off.unsafe_ptr(), label_map.unsafe_ptr(),
        inverse_label_map.unsafe_ptr(), Int32(n),
        grid_dim=(td_grid(n), 1, 1), block_dim=(TD_TPB, 1, 1),
    )
    ctx.synchronize()
    _ = off^
    return n_selected
