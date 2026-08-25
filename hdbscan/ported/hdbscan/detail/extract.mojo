"""Cluster extraction: stabilities, selection, and the point labels.

PORT OF `cuml-v26.08.00/cpp/src/hdbscan/detail/extract.cuh`
(cuML `265b9da`): `TreeUnionFind` (`:49-86`), `do_labelling_on_host`
(`:88-167`) and `extract_clusters` (`:246-314`). `_compute_inverse_label_
map` (`:172-221`) is the CPU/GPU-interop half of `extract_clusters` and is
not reached by a fit; `hdbscan/UNPORTED.tsv` has the row.
Transliterated, their order. Do not improve.

WHAT IS DELIBERATELY NOT CALLED. Their `:311` runs
`Membership::get_probabilities`, which is `detail/membership.cuh` -- a CUB
segmented Max over the same CSR plus a per-point ratio. It is DEFERRED
(DEVIATION 1610), `probabilities` is not produced, and the surface
refuses the field by name rather than returning zeros. `hdbscan/
UNPORTED.tsv` has the row and the closure is small (the segmented Max is
DEVIATION 1604's fold with the comparison reversed).

======================================================================
DEVIATION BLOCK -- DEVIATION 1609. `TreeUnionFind::find` IS ITERATIVE.
======================================================================
WHAT THEIRS DOES (`extract.cuh:74-79`):

    value_idx find(value_idx x) {
      if (data[x * 2] != x) { data[x * 2] = find(data[x * 2]); }
      return data[x * 2];
    }

recursive full path compression: the recursion returns the root and every
frame writes it into its own slot.

WHAT OURS DOES. Two loops: walk to the root, then walk again writing the
root into every slot on the path. Identical output for every input --
same root returned, same fully compressed parent array afterwards -- and
no recursion, so a pathological chain cannot exhaust a stack. This is
`hierarchy`'s DEVIATION 622 in a different file with a different reason:
there the recursion was fine and the INDEXING was out of bounds; here the
indexing is fine and the depth is unbounded.

GATED, not assumed: `check_hdbscan_labels_vs_oracle` runs this struct
against a compression-free union-find in the oracle over the same
condensed tree and requires identical roots for every point, which is the
control DEVIATION 622 has one lane over.
======================================================================

THE CLUSTER SET IS A SORTED SET ON BOTH SIDES. Theirs is
`std::set<value_idx>` (`:281-284`), so iterating it yields ASCENDING
cluster ids and `label_map_h[cluster - n_leaves] = i` numbers the final
labels by ascending condensed id (`:291-295`). Ours builds the same
ascending list by scanning `is_cluster` from 0 upward, which is the same
order without a container. That numbering IS the labels a caller sees, so
it is part of the answer and not a formatting choice.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from hdbscan.mojo_only.hdbscan_sabotage import HDB_SAB_NONE
from hdbscan.ported.hdbscan.condensed_hierarchy import CondensedHierarchy
from hdbscan.ported.hdbscan.detail.select import (
    SELECT_TPB,
    select_clusters,
)
from hdbscan.ported.hdbscan.detail.stabilities import (
    STAB_TPB,
    compute_stabilities,
)
from hierarchy.mojo_only.edge_order import weight_order_key


struct TreeUnionFind(Movable):
    """`extract.cuh:49-86`. `data[i*2]` is the parent, `data[i*2 + 1]` the
    rank; union by rank, full path compression (DEVIATION 1609)."""

    var size: Int
    var data: List[Int32]

    def __init__(out self, size_: Int):
        """`:52-57`."""
        self.size = size_
        self.data = List[Int32](capacity=size_ * 2)
        for _ in range(size_ * 2):
            self.data.append(Int32(0))
        for i in range(size_):
            self.data[i * 2] = Int32(i)

    def find(mut self, x: Int) -> Int:
        """`:74-79`, iterative (DEVIATION 1609)."""
        var root = x
        while Int(self.data[root * 2]) != root:
            root = Int(self.data[root * 2])
        var p = x
        while p != root:
            var nxt = Int(self.data[p * 2])
            self.data[p * 2] = Int32(root)
            p = nxt
        return root

    def perform_union(mut self, x: Int, y: Int):
        """`:59-72`, union by rank, their three branches in their order."""
        var x_root = self.find(x)
        var y_root = self.find(y)
        if self.data[x_root * 2 + 1] < self.data[y_root * 2 + 1]:
            self.data[x_root * 2] = Int32(y_root)
        elif self.data[x_root * 2 + 1] > self.data[y_root * 2 + 1]:
            self.data[y_root * 2] = Int32(x_root)
        else:
            self.data[y_root * 2] = Int32(x_root)
            self.data[x_root * 2 + 1] += Int32(1)


@fieldwise_init
struct ExtractOutput(Copyable, Movable):
    """What `extract_clusters` hands back. Theirs writes through
    out-pointers and returns `clusters.size()` (`:313`); ours returns the
    same values in one struct so the driver can record each as a stage
    without a second call."""

    var n_selected: Int
    var labels: List[Int32]
    """`n_leaves`, CONDENSED cluster ids (or -1), before the label_map
    remap their `runner.h:226-233` applies."""
    var is_cluster: List[Int32]
    """`n_clusters`, the selection."""
    var tree_stabilities: List[Float32]
    """`n_clusters`, AFTER `excess_of_mass` mutated it (DEVIATION 1605)."""
    var label_map: List[Int32]
    """`n_clusters`, condensed id -> final label, -1 where unselected."""
    var inverse_label_map: List[Int32]
    """`n_selected`, final label -> condensed id."""


def do_labelling_on_host(
    tree: CondensedHierarchy,
    in_clusters: List[Int32],
    n_leaves: Int,
    allow_single_cluster: Bool,
    cluster_selection_epsilon: Float32,
) raises -> List[Int32]:
    """`extract.cuh:88-167`.

    `in_clusters` is their `std::set<value_idx>& clusters` as a MEMBERSHIP
    ARRAY indexed by node id: `in_clusters[c] != 0` iff `c` is in their
    set. Same predicate, one indexed load instead of a tree lookup; the
    set's ORDER is used at `:212` and `:291`, not here, and the callers
    that need it build it themselves.
    """
    var n_edges = tree.n_edges
    # `:112-115` size = max(parents)
    var size = Int(tree.parents[0])
    for i in range(n_edges):
        if Int(tree.parents[i]) > size:
            size = Int(tree.parents[i])

    var result = List[Int32](capacity=n_leaves)
    var parent_lambdas = List[Float32](capacity=size + 1)
    for _ in range(size + 1):
        parent_lambdas.append(Float32(0.0))

    var union_find = TreeUnionFind(size + 1)

    # `:122-129`
    for i in range(n_edges):
        var child = Int(tree.children[i])
        var parent = Int(tree.parents[i])
        if in_clusters[child] == Int32(0):
            union_find.perform_union(parent, child)
        # `:128` parent_lambdas[parent] = max(parent_lambdas[parent],
        # lambda[i]). A float max, so IDENTITY_PATHS row 39 applies and it
        # is taken on `weight_order_key`, the INTEGER order this fit's MST
        # already used -- not a hardware max, whose (+0, -0) answer splits
        # Apple from NVIDIA and AMD. The values here are lambdas
        # (non-negative or FLT_MAX by DEVIATIONS 1606 and 1607), so the
        # pin is inert on the default path and the fixture that gives it
        # teeth plants the value.
        if weight_order_key(tree.lambdas[i]) > weight_order_key(
            parent_lambdas[parent]
        ):
            parent_lambdas[parent] = tree.lambdas[i]

    # `:131-134`. Their `inverse_cluster_selection_epsilon` is left
    # UNINITIALIZED when the epsilon is zero and is then not read; ours is
    # not computed at all, because `cluster_selection_epsilon != 0` is
    # refused by name upstream in `select_clusters`. The branch below is
    # transcribed so the shape of their function survives the port.
    var n_in_clusters = 0
    for i in range(len(in_clusters)):
        if in_clusters[i] != Int32(0):
            n_in_clusters += 1

    # `:136-164`
    for i in range(n_leaves):
        var cluster = union_find.find(i)
        if cluster < n_leaves:
            result.append(Int32(-1))
        elif cluster == n_leaves:
            # `:141-160` the root. Only reachable as a LABEL when the root
            # itself was selected, which needs allow_single_cluster.
            if n_in_clusters == 1 and allow_single_cluster:
                # `:144-146` find(children_h.begin(), children_h.end(), i)
                var child_idx = -1
                for e in range(n_edges):
                    if Int(tree.children[e]) == i:
                        child_idx = e
                        break
                if child_idx < 0:
                    raise Error(
                        "hdbscan.do_labelling_on_host: point " + String(i)
                        + " does not appear as a child of any condensed"
                        " edge; their std::find at extract.cuh:144 would"
                        " return end() and the next line dereferences it"
                    )
                var child_lambda = tree.lambdas[child_idx]
                if cluster_selection_epsilon != Float32(0.0):
                    # `:148-153`. Unreachable here: a non-zero epsilon is
                    # refused by name in select_clusters. Transcribed.
                    raise Error(
                        "hdbscan.do_labelling_on_host: cluster_selection_"
                        "epsilon != 0 reached the single-cluster branch;"
                        " it is refused by name in select_clusters and"
                        " should never arrive here"
                    )
                elif child_lambda >= parent_lambdas[cluster]:
                    result.append(Int32(cluster - n_leaves))
                else:
                    result.append(Int32(-1))
            else:
                result.append(Int32(-1))
        else:
            result.append(Int32(cluster - n_leaves))
    return result^


def extract_clusters(
    ctx: DeviceContext,
    tree: CondensedHierarchy,
    n_leaves: Int,
    cluster_selection_method: Int,
    allow_single_cluster: Bool,
    max_cluster_size_in: Int,
    cluster_selection_epsilon: Float32,
    stab_tpb: Int = STAB_TPB,
    select_tpb: Int = SELECT_TPB,
    sabotage: Int32 = HDB_SAB_NONE,
) raises -> ExtractOutput:
    """`extract.cuh:246-314`."""
    var n_clusters = tree.n_clusters

    # `:263`
    var stabilities = ctx.enqueue_create_buffer[DType.float32](n_clusters)
    var is_cluster = ctx.enqueue_create_buffer[DType.int32](n_clusters)
    ctx.synchronize()
    compute_stabilities(ctx, tree, stabilities, stab_tpb, sabotage)

    # `:266` if (max_cluster_size <= 0) max_cluster_size = n_leaves
    var max_cluster_size = max_cluster_size_in
    if max_cluster_size <= 0:
        max_cluster_size = n_leaves

    # `:268-275`
    _ = select_clusters(
        ctx, tree, stabilities, is_cluster, cluster_selection_method,
        allow_single_cluster, max_cluster_size, cluster_selection_epsilon,
        select_tpb, sabotage,
    )

    # `:277-284` is_cluster back to the host; clusters = the ascending set
    # of `i + n_leaves` for every selected i.
    var h_isc = _download_i32(ctx, is_cluster, n_clusters)
    var h_stab = _download_f32(ctx, stabilities, n_clusters)

    # `:286-295` the forward and inverse maps, in ascending cluster order.
    var label_map = List[Int32](capacity=n_clusters)
    for _ in range(n_clusters):
        label_map.append(Int32(-1))
    var inverse_label_map = List[Int32]()
    var n_selected = 0
    for i in range(n_clusters):
        if h_isc[i] != Int32(0):
            label_map[i] = Int32(n_selected)
            inverse_label_map.append(Int32(i))
            n_selected += 1

    # `do_labelling_on_host`'s membership test is over NODE ids, so the
    # array is `n_leaves + n_clusters` long and a selected condensed
    # cluster `i` sits at `i + n_leaves` -- their `clusters.insert(i +
    # n_leaves)` at `:283`.
    var in_clusters = List[Int32](capacity=n_leaves + n_clusters)
    for _ in range(n_leaves + n_clusters):
        in_clusters.append(Int32(0))
    for i in range(n_clusters):
        if h_isc[i] != Int32(0):
            in_clusters[i + n_leaves] = Int32(1)

    # `:303-309`
    var labels = do_labelling_on_host(
        tree, in_clusters, n_leaves, allow_single_cluster,
        cluster_selection_epsilon,
    )

    # `:311` Membership::get_probabilities -- DEFERRED, DEVIATION 1610.

    _ = stabilities^
    _ = is_cluster^
    return ExtractOutput(
        n_selected, labels^, h_isc^, h_stab^, label_map^, inverse_label_map^
    )


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
