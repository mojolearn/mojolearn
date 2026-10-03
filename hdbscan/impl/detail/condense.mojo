# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The condensed tree: collapse every subtree below `min_cluster_size`.

Reference: `cuml-v26.08.00/cpp/src/hdbscan/detail/condense.cuh`
(cuML `265b9da`): `bfs_from_node` (`:37-66`), `_build_condensed_hierarchy`
(`:91-212`) and `build_condensed_hierarchy` (`:237-286`), with the
reference branches in the reference order.

ON THE DEVICE FOR OUR DEVICE FIT (lane cgr2-hdbscan, 2026-10-03):
`build_condensed_hierarchy` below hands off to
`tree_device.mojo::build_condensed_device`, which computes the walk's
result in closed form by pointer jumping and a radix sort. The walk in this
file (`bfs_from_node`, `_collapse`, `_add_edge`) is the CPU column's
(`hdbscan_host_oracle.mojo::hdbh_condense`), and the two trees are equal
element for element.

THEIR CONDENSE IS A HOST FUNCTION, and that is worth stating
because the file name says `.cuh`. cuML 26.08 rewrote condense as a
serial host walk over `std::vector`s -- their own comment at `:69` says
"This implementation is based on scikit-learn's _condense_tree
implementation" -- and copies the four output arrays to the device at
`:205-211`. So no kernel of theirs was dropped here; there is none to
drop. (The `kernels/condense.cuh` header their `:8` includes is empty of
anything this path reaches.)

======================================================================
THE TRAVERSAL IS THE NUMBERING, AND THE NUMBERING IS EVERY DOWNSTREAM
INDEX. (IDENTITY hazard 3, first half.)
======================================================================
`next_label` starts at `n_samples + 1` and increments ONCE PER SELECTED
CHILD, in the order `node_list` is visited (`:156-160`). `node_list` is
`bfs_from_node(root)`, a LEVEL-BY-LEVEL breadth-first order:
`process_queue` holds one whole level, the level is appended to `result`
in queue order, and the next level is built by walking the level's
internal nodes left child then right child (`:46-64`).

So the condensed tree's cluster ids -- and therefore `stabilities[c]`,
`is_cluster[c]`, `births[c]`, the CSR segment boundaries and the final
label numbering -- are a pure function of THAT ORDER and of nothing else.
There is no float in the traversal, no atomic, no thread. It is
reproducible on every vendor for the same reason a `for` loop is, and the
thing that could break it is a rewrite to a different traversal, which is
why `HDB_SAB_CONDENSE_DFS` exists and why the gate is `check_condensed_
tree_vs_oracle` comparing NODE FOR NODE rather than comparing a summary.

WHERE THE TRAVERSAL DOES *NOT* REACH, because a reader is owed the
narrower claim rather than the wide one. The traversal order also decides
the order in which edges are APPENDED to `out_parent`/`out_child` below,
and the order in which `_collapse` emits a subtree's leaves -- and
NEITHER of those is observable, because `CondensedHierarchy.condense()`
sorts the four arrays on `(parent, child)` (DEVIATION 1611) and a
collapsed subtree's leaf set is the same set whichever way it is walked.
The ONE channel from the walk to the output is the VALUE assigned to
`relabel` in case 1. That is why `HDB_SAB_CONDENSE_DFS` needs a fixture
whose condensed cluster tree has two case-1 nodes that are neither
ancestor nor descendant, with the left one deeper -- `HFIX_NESTED`,
derived in `hdbscan/checks/hdbscan_fixture.mojo`'s header -- and why it
sat inert on `blobs96`, whose 100 edges and 5 clusters say it has exactly
two case-1 nodes and they are nested.

THE ONE FLOAT IN THIS FILE is `lambda_value = 1 / distance`, DEVIATION
1606 below.
======================================================================

======================================================================
DEVIATION BLOCK -- DEVIATION 1606. `lambda = 1 / distance` GOES THROUGH
`identical_div`.
======================================================================
WHAT THEIRS DOES (`:149`):

    value_t lambda_value = distance > 0.0 ? 1.0 / distance
                                          : std::numeric_limits<value_t>::max();

on the HOST, in `float` (their `value_t`), with the host's own divide.

WHAT OURS DOES. The same expression with `1.0 / distance` spelled
`identical_div(1.0, distance)`, which under IDENTICAL is
`portable_divf` -- the row-10 flush model around one correctly rounded
division (IDENTITY_PATHS row 49's seam) -- and under FAST is `/`. The
guard `distance > 0.0` and the `FLT_MAX` sentinel are theirs, unchanged,
including the consequence that a negative or `-0.0` distance takes the
sentinel arm (`-0.0 > 0.0` is false on every vendor; an IEEE compare, not
a hardware max, so row 39's split does not reach it).

WHY IT MATTERS ON A HOST LINE. Every lambda is summed into a stability
and compared against `cluster_selection_epsilon`, and the sum's terms are
these bits. `IDENTITY_PATHS` row 18 is the standing warning that a HOST
libm difference is a CROSS-HOST difference and reaches the model through
a truncation or a threshold; a divide is the same class. Apple's divide
is correctly rounded, so this seam is bit-inert here and the arm that
would move it is `HDB_SAB_LAMBDA_STD_DIV` on a column whose divide is
not.
======================================================================
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from hdbscan.checks.hdbscan_sabotage import (
    HDB_SAB_CONDENSE_DFS,
    HDB_SAB_NONE,
)
from hdbscan.impl.condensed_hierarchy import CondensedHierarchy
from hdbscan.impl.detail.tree_device import (
    DeviceTree,
    build_condensed_device,
    td_download_f32,
    td_download_i32,
)


def bfs_from_node(
    bfs_root: Int,
    n_samples: Int,
    h_children: List[Int32],
    mut result: List[Int32],
    sabotage: Int32 = HDB_SAB_NONE,
) raises:
    """`condense.cuh:37-66`, "Helper function for BFS traversal from a
    given node in the hierarchy".

    Their loop, unchanged:

        process_queue = [bfs_root]
        while process_queue not empty:
            result += process_queue                 // whole level, in order
            internal = [x - n_samples for x in process_queue if x >= n_samples]
            next_queue = []
            for node in internal:
                next_queue += [children[2*node], children[2*node+1]]
            process_queue = next_queue

    `HDB_SAB_CONDENSE_DFS` replaces the level queue with a STACK, which
    visits the same nodes and produces a different ORDER -- the sabotage
    for the numbering claim above.

    THE STACK IS A PREORDER, LEFT FIRST: the right child is pushed before
    the left, so the left pops first. That matters, because it is what
    makes the two walks agree on every ancestor/descendant pair and
    disagree only where a LEFT-branch node is deeper than a RIGHT-branch
    one. On a LEFT-LEANING CATERPILLAR -- which is what
    `bfs_from_node(subtree_root, ...)` is handed inside `_collapse`, and
    what a single-linkage dendrogram is where one growing cluster absorbs
    one point at a time -- the two walks are IDENTICAL node for node, not
    merely equivalent. `HDB_SAB_CONDENSE_DFS`'s docstring in
    `hdbscan/checks/hdbscan_sabotage.mojo` carries the rest.
    """
    if sabotage == HDB_SAB_CONDENSE_DFS:
        # The sabotage arm: a depth-first stack. Same node SET, different
        # order, therefore a different `next_label` assignment. The stack
        # can hold at most one entry per node of the dendrogram.
        var stack = List[Int32](capacity=2 * n_samples + 2)
        for _ in range(2 * n_samples + 2):
            stack.append(Int32(0))
        var top = 0
        stack[top] = Int32(bfs_root)
        top += 1
        while top > 0:
            top -= 1
            var node = stack[top]
            result.append(node)
            if Int(node) >= n_samples:
                var h = Int(node) - n_samples
                stack[top] = h_children[h * 2 + 1]
                top += 1
                stack[top] = h_children[h * 2]
                top += 1
        return

    var process_queue = List[Int32]()
    process_queue.append(Int32(bfs_root))
    while len(process_queue) > 0:
        # `:48` Add all nodes in current level to result
        for i in range(len(process_queue)):
            result.append(process_queue[i])
        # `:51-54` Filter for internal nodes (>= n_samples) and convert
        # to hierarchy indices
        var internal_nodes = List[Int]()
        for i in range(len(process_queue)):
            var x = Int(process_queue[i])
            if x >= n_samples:
                internal_nodes.append(x - n_samples)
        # `:57-63` Get children of all internal nodes for next level
        var next_queue = List[Int32]()
        for i in range(len(internal_nodes)):
            var node = internal_nodes[i]
            next_queue.append(h_children[node * 2])
            next_queue.append(h_children[node * 2 + 1])
        process_queue = next_queue.copy()


def build_condensed_hierarchy(
    ctx: DeviceContext,
    mut children: DeviceBuffer[DType.int32],
    mut delta: DeviceBuffer[DType.float32],
    mut sizes: DeviceBuffer[DType.int32],
    min_cluster_size: Int,
    n_leaves: Int,
    sabotage: Int32 = HDB_SAB_NONE,
) raises -> DeviceTree:
    """`condense.cuh:237-286` for the device fit: built on the device
    (`tree_device.mojo::build_condensed_device`, the closed form of the
    walk below). The walk (`bfs_from_node`, `_collapse`, `_add_edge`) is
    the CPU column's (`hdbh_condense`)."""
    return build_condensed_device(
        ctx, children, delta, sizes, min_cluster_size, n_leaves, sabotage
    )


def download_condensed(
    ctx: DeviceContext, mut tree: DeviceTree
) raises -> CondensedHierarchy:
    """The device tree as host lists, once, for the fit's outputs."""
    return CondensedHierarchy(
        tree.n_leaves,
        tree.n_edges,
        tree.n_clusters,
        td_download_i32(ctx, tree.parents, tree.n_edges),
        td_download_i32(ctx, tree.children, tree.n_edges),
        td_download_f32(ctx, tree.lambdas, tree.n_edges),
        td_download_i32(ctx, tree.sizes, tree.n_edges),
    )


def _add_edge(
    mut out_parent: List[Int32],
    mut out_child: List[Int32],
    mut out_lambda: List[Float32],
    mut out_size: List[Int32],
    parent: Int,
    child: Int,
    lambda_value: Float32,
    size: Int,
):
    """`condense.cuh:124-129`, their `add_edge` lambda."""
    out_parent.append(Int32(parent))
    out_child.append(Int32(child))
    out_lambda.append(lambda_value)
    out_size.append(Int32(size))


def _collapse(
    subtree_root: Int,
    node: Int,
    n_samples: Int,
    h_children: List[Int32],
    relabel: List[Int],
    mut ignore: List[Int],
    mut out_parent: List[Int32],
    mut out_child: List[Int32],
    mut out_lambda: List[Float32],
    mut out_size: List[Int32],
    lambda_value: Float32,
    sabotage: Int32,
) raises:
    """`condense.cuh:163-177` (and the identical bodies at `:183-189` and
    `:195-201`): BFS the subtree, add an edge from `relabel[node]` to each
    LEAF found, and mark every visited node -- leaf or internal -- to be
    ignored.

    The reference's three copies are one function here; the body is written
    once because it is three copies of one paragraph, and

    """
    var descendants = List[Int32]()
    bfs_from_node(subtree_root, n_samples, h_children, descendants, sabotage)
    for i in range(len(descendants)):
        var sub_node = Int(descendants[i])
        if sub_node < n_samples:
            _add_edge(
                out_parent, out_child, out_lambda, out_size,
                relabel[node], sub_node, lambda_value, 1,
            )
        ignore[sub_node] = 1
