# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Derivative work: the upstream file and its pinned commit are recorded in this lane's DERIVATION_MAP.tsv and in this file's own docstring. See NOTICE.
"""Cluster selection: Excess of Mass, Leaf, and the negation BFS.

PORT OF `cuml-v26.08.00/cpp/src/hdbscan/detail/select.cuh`
(cuML `265b9da`): `perform_bfs` (`:57-91`), `excess_of_mass` (`:148-252`),
`leaf` (`:264-286`) and `select_clusters` (`:379-452`), plus
`detail/kernels/select.cuh::propagate_cluster_negation_kernel`
(`:24-45`). `cluster_epsilon_search` (`:301-363`) and its kernel
(`:47-104`) are NOT PORTED and raise by name; `hdbscan/NOT_IMPLEMENTED.tsv` has
the row. `Select::parent_csr` (`:103-130`) lives in `detail/utils.mojo`
beside its twin, which is the one rename this file makes and
`DERIVATION_MAP.tsv` records it.

EXCESS OF MASS IS THE DEFAULT AND IS THE ONE THIS LANE SHIPS
(`hdbscan.hpp:197`, `cluster_selection_method = EOM`). LEAF is ported too
because it is nine lines of integer work and PORTING_RULES rule 8 is
explicit that a switch with an unexercised side is an unchecked path:
`check_hdbscan_selection_leaf` runs it, and the same rule is why
`cluster_selection_epsilon` is REFUSED rather than quietly ignored.

======================================================================
DEVIATION BLOCK -- DEVIATION 1605. THE EXCESS-OF-MASS LOOP RUNS ON THE
HOST OVER ONE DOWNLOAD, AND ITS SUBTREE SUM IS A SERIAL ASCENDING FOLD.
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

-- so the LOOP is already theirs and already on the host; what is on the
device is one scalar readback and one `transform_reduce` per node.

WHY IT CANNOT BE PORTED AS-IS. `thrust::transform_reduce` is a device
tree reduction whose shape is the library's, so a cluster with three or
more children has a summation order nobody in this repository can pin,
read or check -- IDENTITY_PATHS row 20's class. And the sum decides a
BOOLEAN (`subtree_stability > node_stability`) which decides a CLUSTER,
so a last-bit difference is not a perturbation of an output, it is a
different partition. Row 7's class: a float fold deciding a discrete
outcome.

WHAT OURS DOES. One download of `stability` before the loop, one upload
after; the loop and the subtree sum entirely on the host, the sum SERIAL
and ASCENDING over the CSR segment through `ftz`/`identical_mul_add` with
the multiplier fixed at 1.0 so the seam is spelled the same way as
DEVIATION 1603's.

THE WRITE-BACK IS LOAD BEARING AND MUST STAY IN PLACE. `stability[node]`
is OVERWRITTEN with the subtree total when a node is deselected
(`:227`), and the loop runs from the leaves TOWARD the root, so an
ancestor's sum reads the updated descendants. Doing the loop over one
host array preserves that exactly -- the same array is read and written
in the same order -- and the arm that proves it matters is
`HDB_SAB_EOM_NO_UPDATE`, which drops the write-back and must move the
selection.

WHAT IT COSTS THEM AND US. Theirs drains the queue twice per cluster;
ours drains twice per FIT. No timing was taken and none is claimed --
that is an operation count, not a measurement.
======================================================================

======================================================================
DEVIATION BLOCK -- DEVIATION 1613. `cluster_sizes[0]` IS A DATA RACE IN
THEIR KERNEL AND A SERIAL FOLD IN OURS.
======================================================================
WHAT THEIRS DOES (`select.cuh:173-182`), one thread per cluster-tree
edge:

    if (get<0>(tup) == 0) cluster_sizes_ptr[0] += get<2>(tup);
    cluster_sizes_ptr[cuda::std::get<1>(tup)] = get<2>(tup);

The second line is fine: each child is written once. The FIRST line is a
NON-ATOMIC read-modify-write into ONE cell from every thread whose parent
is the root. With more than one root child -- which is every tree that
has a root split, i.e. every interesting tree -- that is a race, and the
value it leaves is between one child's size and the sum of all of them.
`cluster_sizes_h[node]` is then compared against `max_cluster_size`
(`:225`), so on a fit that sets `max_cluster_size` the race can decide
whether the ROOT is deselected.

WHY WE DO NOT PORT IT. `[[assume-our-code-is-broken]]` says theirs is
right about DESIGN and that we fix rather than port their BUGS, numbered
and checked. The DESIGN is "cluster_sizes[0] is the total size of the
root's children"; the race is a spelling of it that does not compute it.

WHAT OURS DOES. The same two assignments in a serial host loop over the
cluster tree's edges in `(parent, child)` order, so `cluster_sizes[0]` is
the exact integer sum. An integer sum is order-free, so no order is being
chosen here -- theirs simply loses terms.

NOT MEASURED AGAINST THEIRS. We have no CUDA build to race, so the claim
above is read off `:178-179` and stated as a reading, not as an observed
divergence. What IS checked is that ours equals the exact sum:
`check_hdbscan_selection_eom` asserts `cluster_sizes[0]` against a host
oracle that sums independently.
======================================================================
"""

from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from hdbscan.original.hdbscan_sabotage import (
    HDB_SAB_EOM_NO_UPDATE,
    HDB_SAB_NONE,
)
from hdbscan.derived.hdbscan.condensed_hierarchy import CondensedHierarchy
from hdbscan.derived.hdbscan.detail.utils import make_cluster_tree, select_parent_csr
from original.numerics import ftz, identical_mul_add


comptime SELECT_TPB = 256
"""Their `int tpb = 256` template default (`select.cuh:57`, `:148`,
`:264`). SCHEDULING: the only device kernel here writes booleans."""

comptime CLUSTER_SELECTION_EOM = 0
comptime CLUSTER_SELECTION_LEAF = 1
"""`hdbscan.hpp:126` `enum CLUSTER_SELECTION_METHOD { EOM = 0, LEAF = 1 }`."""


def propagate_cluster_negation_kernel(
    indptr: MutPointer[Int32, MutAnyOrigin],
    children: MutPointer[Int32, MutAnyOrigin],
    frontier: MutPointer[Int32, MutAnyOrigin],
    next_frontier: MutPointer[Int32, MutAnyOrigin],
    is_cluster: MutPointer[Int32, MutAnyOrigin],
    n_clusters_in: Int32,
):
    """`kernels/select.cuh:24-45`, transliterated.

    NO ORDER TO PIN. Every write is a CONSTANT (`false` into `frontier`
    and `is_cluster`, `true` into `next_frontier`), so two threads that
    collide write the same byte and the result does not depend on which
    landed. That is the same argument `agglomerative.cuh`'s
    `inherit_labels` rests on, and it is why this kernel needed no
    deviation.
    """
    var cluster = Int(block_dim.x) * Int(block_idx.x) + Int(thread_idx.x)
    if cluster >= Int(n_clusters_in):
        return
    if frontier.unsafe_load(cluster) == Int32(0):
        return
    frontier.unsafe_store(cluster, Int32(0))
    var children_start = Int(indptr.unsafe_load(cluster))
    var children_stop = Int(indptr.unsafe_load(cluster + 1))
    for i in range(children_start, children_stop):
        var child = Int(children.unsafe_load(i))
        next_frontier.unsafe_store(child, Int32(1))
        is_cluster.unsafe_store(child, Int32(0))


def perform_bfs(
    ctx: DeviceContext,
    mut indptr: DeviceBuffer[DType.int32],
    mut children: DeviceBuffer[DType.int32],
    mut frontier: DeviceBuffer[DType.int32],
    mut is_cluster: DeviceBuffer[DType.int32],
    n_clusters: Int,
    tpb: Int = SELECT_TPB,
) raises:
    """`select.cuh:57-91`: while the frontier is non-empty, launch the
    kernel, swap the frontiers, and count what is left.

    THE COUNT IS INTEGER AND EXACT. Theirs is `thrust::reduce` on the
    device and read on the host (`:73`, `:88`); ours downloads the
    frontier and sums it serially. An integer sum has no order, so this
    moves where the addition happens and nothing else.

    THE LOOP HAS NO CAP AND THAT IS THEIRS. `while (n_elements_to_
    traverse > 0)` (`:81`) terminates because each round strictly
    descends one level of a finite tree and a node is removed from the
    frontier before its children are added; a cycle in `indptr`/`children`
    would hang, and a cycle is refused upstream by the dendrogram's own
    structure check. DEVIATION 519's lesson (a cap that truncates is worse
    than no cap) says not to invent one here.
    """
    var next_frontier = ctx.enqueue_create_buffer[DType.int32](n_clusters)
    ctx.enqueue_memset(next_frontier, Int32(0))
    var h_frontier = ctx.enqueue_create_host_buffer[DType.int32](n_clusters)
    ctx.synchronize()

    var grid = (n_clusters + tpb - 1) // tpb
    var n_left = _sum_frontier(ctx, frontier, h_frontier, n_clusters)
    while n_left > 0:
        ctx.enqueue_function[propagate_cluster_negation_kernel](
            indptr.unsafe_ptr(),
            children.unsafe_ptr(),
            frontier.unsafe_ptr(),
            next_frontier.unsafe_ptr(),
            is_cluster.unsafe_ptr(),
            Int32(n_clusters),
            grid_dim=(grid, 1, 1),
            block_dim=(tpb, 1, 1),
        )
        ctx.synchronize()
        # `:85-86` copy next_frontier into frontier, then clear it.
        ctx.enqueue_copy(dst_buf=frontier, src_buf=next_frontier)
        ctx.synchronize()
        ctx.enqueue_memset(next_frontier, Int32(0))
        ctx.synchronize()
        n_left = _sum_frontier(ctx, frontier, h_frontier, n_clusters)
    _ = next_frontier^
    _ = h_frontier^


def _sum_frontier(
    ctx: DeviceContext,
    mut frontier: DeviceBuffer[DType.int32],
    mut host: HostBuffer[DType.int32],
    n: Int,
) raises -> Int:
    """`thrust::reduce(frontier, frontier + n_clusters, 0)` (`:73`, `:88`),
    on the host. Integer, exact, order-free."""
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=frontier)
    ctx.synchronize()
    var total = 0
    for i in range(n):
        total += Int(host.unsafe_ptr().unsafe_load(i))
    return total


def excess_of_mass(
    ctx: DeviceContext,
    cluster_tree: CondensedHierarchy,
    mut stability: DeviceBuffer[DType.float32],
    mut is_cluster: DeviceBuffer[DType.int32],
    n_clusters: Int,
    max_cluster_size: Int,
    allow_single_cluster: Bool,
    tpb: Int = SELECT_TPB,
    sabotage: Int32 = HDB_SAB_NONE,
) raises:
    """`select.cuh:148-252`, DEVIATIONS 1605 and 1613."""
    var n_edges = cluster_tree.n_edges

    # `:166-182` cluster_sizes. DEVIATION 1613: serial, exact.
    var cluster_sizes = List[Int](capacity=n_clusters)
    for _ in range(n_clusters):
        cluster_sizes.append(0)
    for i in range(n_edges):
        var parent = Int(cluster_tree.parents[i])
        var child = Int(cluster_tree.children[i])
        var size = Int(cluster_tree.sizes[i])
        if parent == 0:
            cluster_sizes[0] += size
        if child < 0 or child >= n_clusters:
            raise Error(
                "hdbscan.excess_of_mass: cluster-tree child " + String(child)
                + " at edge " + String(i) + " is outside [0, "
                + String(n_clusters) + "); their cluster_sizes_ptr write at"
                " select.cuh:181 would land outside the array"
            )
        cluster_sizes[child] = size

    # `:189-193` is_cluster all true, root = allow_single_cluster;
    # frontier all false.
    var is_cluster_h = List[Int32](capacity=n_clusters)
    var frontier_h = List[Int32](capacity=n_clusters)
    for _ in range(n_clusters):
        is_cluster_h.append(Int32(1))
        frontier_h.append(Int32(0))
    is_cluster_h[0] = Int32(1) if allow_single_cluster else Int32(0)

    # `:195-203` the CSR over the cluster tree's parents.
    var indptr_h = select_parent_csr(cluster_tree)

    # DEVIATION 1605: one download, the whole loop on the host.
    var stab_h = _download_f32(ctx, stability, n_clusters)

    # `:206-233` reverse topological order.
    var tree_top = 0 if allow_single_cluster else 1
    var node = n_clusters - 1
    while node >= tree_top:
        var node_stability = stab_h[node]
        var lo = Int(indptr_h[node])
        var hi = Int(indptr_h[node + 1])
        var subtree_stability = Float32(0.0)
        if hi - lo > 0:
            # `:215-222` transform_reduce over this node's children,
            # serial and ascending. The multiplier is 1.0 so the seam is
            # spelled exactly as DEVIATION 1603's is.
            for i in range(lo, hi):
                var child = Int(cluster_tree.children[i])
                subtree_stability = ftz(
                    identical_mul_add(
                        Float32(1.0), stab_h[child], subtree_stability
                    )
                )
        if (
            subtree_stability > node_stability
            or cluster_sizes[node] > max_cluster_size
        ):
            # `:225-228` Deselect / merge cluster with children
            if sabotage != HDB_SAB_EOM_NO_UPDATE:
                stab_h[node] = subtree_stability
            is_cluster_h[node] = Int32(0)
        else:
            # `:231` Mark children to be deselected
            frontier_h[node] = Int32(1)
        node -= 1

    _upload_f32(ctx, stability, stab_h)

    # `:239-251` propagate the deselection through the subtrees.
    var frontier = ctx.enqueue_create_buffer[DType.int32](n_clusters)
    var d_indptr = ctx.enqueue_create_buffer[DType.int32](n_clusters + 1)
    var d_children = ctx.enqueue_create_buffer[DType.int32](
        n_edges if n_edges > 0 else 1
    )
    ctx.synchronize()
    var h_isc = is_cluster_h.copy()
    var h_fr = frontier_h.copy()
    var h_ip = indptr_h.copy()
    ctx.enqueue_copy(dst_buf=is_cluster, src_ptr=h_isc.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=frontier, src_ptr=h_fr.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_indptr, src_ptr=h_ip.unsafe_ptr())
    if n_edges > 0:
        var h_ch = cluster_tree.children.copy()
        ctx.enqueue_copy(dst_buf=d_children, src_ptr=h_ch.unsafe_ptr())
        ctx.synchronize()
        _ = h_ch^
    ctx.synchronize()
    perform_bfs(ctx, d_indptr, d_children, frontier, is_cluster, n_clusters, tpb)
    _ = frontier^
    _ = d_indptr^
    _ = d_children^
    _ = h_isc^
    _ = h_fr^
    _ = h_ip^


def leaf(
    ctx: DeviceContext,
    cluster_tree: CondensedHierarchy,
    mut is_cluster: DeviceBuffer[DType.int32],
    n_clusters: Int,
) raises:
    """`select.cuh:264-286`: "Uses the leaves of the cluster tree as final
    cluster selections". A node is selected iff it is a CHILD of some edge
    and is a PARENT of none.

    WHERE IT RUNS. Theirs is two `thrust::for_each`es writing constants;
    ours is two serial host passes over arrays that are already on the
    host, uploaded once at the end. Both writes are of the same constant
    from every source, so no order is being chosen either way.
    """
    var is_parent = List[Int32](capacity=n_clusters)
    var out = List[Int32](capacity=n_clusters)
    for _ in range(n_clusters):
        is_parent.append(Int32(0))
        out.append(Int32(0))
    # `:279-280`
    for i in range(cluster_tree.n_edges):
        var p = Int(cluster_tree.parents[i])
        if p >= 0 and p < n_clusters:
            is_parent[p] = Int32(1)
    # `:282-285`
    for i in range(cluster_tree.n_edges):
        var c = Int(cluster_tree.children[i])
        if c >= 0 and c < n_clusters:
            if is_parent[c] == Int32(0):
                out[c] = Int32(1)
    var h = out.copy()
    ctx.enqueue_copy(dst_buf=is_cluster, src_ptr=h.unsafe_ptr())
    ctx.synchronize()
    _ = h^


def cluster_epsilon_search(cluster_selection_epsilon: Float32) raises:
    """`select.cuh:301-363` and `kernels/select.cuh:47-104`. NOT PORTED;
    raises by name."""
    raise Error(
        "hdbscan.cluster_epsilon_search: cluster_selection_epsilon="
        + String(cluster_selection_epsilon)
        + " refused by name; the epsilon search is NOT PORTED (rung 2)."
        " It re-sorts the cluster tree's parents and lambdas BY CHILD in"
        " place (select.cuh:328-329), then walks each selected cluster"
        " toward the root in a do/while whose index arithmetic depends on"
        " that re-sort (kernels/select.cuh:70-100, the `child = child_idx +"
        " 1` offset), and upstream's own comment at select.cuh:418-419"
        " records a confirmed reference-implementation bug in the"
        " neighbouring LEAF branch (scikit-learn-contrib/hdbscan#476) that"
        " is commented out rather than fixed. Porting it means deciding"
        " which of two behaviors is the algorithm, which is a question for"
        " a reading of their tree, not for this rung. Use"
        " cluster_selection_epsilon=0.0, which is their default"
        " (hdbscan.hpp:136)"
    )


def select_clusters(
    ctx: DeviceContext,
    condensed_tree: CondensedHierarchy,
    mut tree_stabilities: DeviceBuffer[DType.float32],
    mut is_cluster: DeviceBuffer[DType.int32],
    cluster_selection_method: Int,
    allow_single_cluster: Bool,
    max_cluster_size: Int,
    cluster_selection_epsilon: Float32,
    tpb: Int = SELECT_TPB,
    sabotage: Int32 = HDB_SAB_NONE,
) raises -> Int:
    """`select.cuh:379-452`. Returns `n_selected_clusters` (`:412`)."""
    var n_clusters = condensed_tree.n_clusters
    # `:394`
    var cluster_tree = make_cluster_tree(condensed_tree)

    if cluster_selection_method == CLUSTER_SELECTION_EOM:
        # `:396-403`
        excess_of_mass(
            ctx, cluster_tree, tree_stabilities, is_cluster, n_clusters,
            max_cluster_size, allow_single_cluster, tpb, sabotage,
        )
    elif cluster_selection_method == CLUSTER_SELECTION_LEAF:
        # `:404-409`
        ctx.enqueue_memset(is_cluster, Int32(0))
        ctx.synchronize()
        if cluster_tree.n_edges > 0:
            leaf(ctx, cluster_tree, is_cluster, n_clusters)
    else:
        raise Error(
            "hdbscan.select_clusters: cluster_selection_method="
            + String(cluster_selection_method)
            + " refused by name; their enum has exactly two values, EOM=0"
            " and LEAF=1 (hdbscan.hpp:126)"
        )

    # `:412`
    var h_isc = _download_i32(ctx, is_cluster, n_clusters)
    var n_selected_clusters = 0
    for i in range(n_clusters):
        if h_isc[i] != Int32(0):
            n_selected_clusters += 1

    # `:429-451` the epsilon search. Refused by name rather than skipped,
    # so a caller who sets the parameter learns that it does nothing here
    # instead of getting a silently different partition.
    if cluster_selection_epsilon != Float32(0.0) and cluster_tree.n_edges > 0:
        var epsilon_search = True
        # `:431` no epsilon search if no clusters were selected
        if n_selected_clusters == 0:
            epsilon_search = False
        # `:435-441` this is to check when eom finds root as only cluster
        if cluster_selection_method == CLUSTER_SELECTION_EOM:
            if n_selected_clusters == 1:
                if h_isc[0] != Int32(0) and allow_single_cluster:
                    epsilon_search = False
        if epsilon_search:
            cluster_epsilon_search(cluster_selection_epsilon)
    _ = cluster_tree^
    return n_selected_clusters


def _download_f32(
    ctx: DeviceContext, buf: DeviceBuffer[DType.float32], n: Int
) raises -> List[Float32]:
    var h = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.synchronize()
    var v = buf.create_sub_buffer[DType.float32](0, n)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=v)
    ctx.synchronize()
    var out = List[Float32](capacity=n)
    for i in range(n):
        out.append(h.unsafe_ptr().unsafe_load(i))
    _ = h^
    _ = v^
    return out^


def _download_i32(
    ctx: DeviceContext, buf: DeviceBuffer[DType.int32], n: Int
) raises -> List[Int32]:
    var h = ctx.enqueue_create_host_buffer[DType.int32](n)
    ctx.synchronize()
    var v = buf.create_sub_buffer[DType.int32](0, n)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=v)
    ctx.synchronize()
    var out = List[Int32](capacity=n)
    for i in range(n):
        out.append(h.unsafe_ptr().unsafe_load(i))
    _ = h^
    _ = v^
    return out^


def _upload_f32(
    ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], values: List[Float32]
) raises:
    var tmp = values.copy()
    var v = buf.create_sub_buffer[DType.float32](0, len(tmp))
    ctx.enqueue_copy(dst_buf=v, src_ptr=tmp.unsafe_ptr())
    ctx.synchronize()
    _ = tmp^
    _ = v^
