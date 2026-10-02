# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# SHIPS: compiled into the CPU host binding through hdbscan_host_oracle.mojo; product, not only a check.
"""`do_labelling_on_host` (`extract.cuh:88-167`) and its `TreeUnionFind`
(`:49-86`): the CPU column's labelling. A GPU fit labels on the device
(`hdbscan/impl/detail/extract.mojo::do_labelling_device`); both return the
same integers for every condensed tree (see that function's docstring:
edges are unioned in ascending-parent order, so every component's
representative is its topmost node, which is exactly the node the
device's pointer jumping reaches).

======================================================================
DEVIATION BLOCK -- DEVIATION 1609. `TreeUnionFind::find` IS ITERATIVE.
======================================================================
WHAT THEIRS DOES (`extract.cuh:74-79`): recursive full path compression.

WHAT OURS DOES. Two loops: walk to the root, then walk again writing the
root into every slot on the path. Identical output for every input, and
no recursion, so a pathological chain cannot exhaust a stack.
======================================================================
"""

from hdbscan.impl.condensed_hierarchy import CondensedHierarchy
from hierarchy.checks.edge_order import weight_order_key

from checks.numerics import identical_div


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
    # zero then, and likewise never read. `identical_div` (DEVIATION 5115).
    var inverse_cluster_selection_epsilon = Float32(0.0)
    if cluster_selection_epsilon != Float32(0.0):
        inverse_cluster_selection_epsilon = identical_div(
            Float32(1.0), cluster_selection_epsilon
        )
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
                    # `:148-153`: a point joins the root cluster when it
                    # left at or above 1 / epsilon.
                    if child_lambda >= inverse_cluster_selection_epsilon:
                        result.append(Int32(cluster - n_leaves))
                    else:
                        result.append(Int32(-1))
                elif child_lambda >= parent_lambdas[cluster]:
                    result.append(Int32(cluster - n_leaves))
                else:
                    result.append(Int32(-1))
            else:
                result.append(Int32(-1))
        else:
            result.append(Int32(cluster - n_leaves))
    return result^
