# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Derivative work: the upstream file and its pinned commit are recorded in this lane's PORTED_MAP.tsv and in this file's own docstring. See NOTICE.
"""The condensed tree: collapse every subtree below `min_cluster_size`.

PORT OF `cuml-v26.08.00/cpp/src/hdbscan/detail/condense.cuh`
(cuML `265b9da`): `bfs_from_node` (`:37-66`), `_build_condensed_hierarchy`
(`:91-212`) and `build_condensed_hierarchy` (`:237-286`). Transliterated,
their branches in their order. Do not improve.

THIS IS A HOST FUNCTION ON THEIR SIDE TOO, and that is worth stating
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

from hdbscan.mojo_only.hdbscan_sabotage import (
    HDB_SAB_CONDENSE_DFS,
    HDB_SAB_LAMBDA_STD_DIV,
    HDB_SAB_NONE,
)
from hdbscan.mojo_only.mutual_reachability_dense import refuse_nonfinite_host
from hdbscan.ported.hdbscan.condensed_hierarchy import CondensedHierarchy
from hierarchy.ported.cluster.detail.connectivities import FLOAT32_MAX
from mojo_only.numerics import identical_div


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
) raises -> CondensedHierarchy:
    """`condense.cuh:237-286`, with `_build_condensed_hierarchy`
    (`:91-212`) inlined behind it exactly as their call at `:274-283`
    puts it -- one Mojo function for their pair, recorded in
    `hdbscan/PORTED_MAP.tsv`.

    Returns the condensed hierarchy their `condensed_tree.condense(...)`
    at `:285` populates.
    """
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

    # `:250` Root is the last edge in the dendrogram
    var root = 2 * (n_leaves - 1)
    var n_edges = root
    var n_samples = n_leaves

    # `:108-116` Copy data to host
    var h_children = _copy_i32(ctx, children, n_edges)
    var h_delta = _copy_f32(ctx, delta, n_leaves - 1)
    var h_sizes = _copy_i32(ctx, sizes, n_leaves - 1)

    # `:252-261` n_vertices = max(children) + 1, then RAFT_EXPECTS
    # n_vertices == root. Their message, kept.
    var max_child = Int32(-1)
    for i in range(n_edges):
        if h_children[i] > max_child:
            max_child = h_children[i]
    var n_vertices = Int(max_child) + 1
    if n_vertices != root:
        raise Error(
            "hdbscan.build_condensed_hierarchy: Multiple components found"
            " in MST or MST is invalid. Cannot find single-linkage"
            " solution. Found " + String(n_vertices) + " vertices total"
            " (expected " + String(root) + ")"
        )

    # NOT THEIRS. DEVIATION 1607: `delta` is the MST weight column and
    # every lambda below is `1 / delta`, so a non-finite here becomes a
    # non-finite in a recorded stage.
    refuse_nonfinite_host(
        h_delta, "hdbscan.build_condensed_hierarchy", "dendrogram deltas",
        sabotage,
    )

    # `:104-106`
    var next_label = n_samples + 1

    # `:131-133` Get BFS ordering from root
    var node_list = List[Int32]()
    bfs_from_node(root, n_samples, h_children, node_list, sabotage)

    # `:135-137`. Their `ignore` is sized `node_list.size()` and indexed
    # by NODE; for a valid dendrogram those are the same number
    # (`2 * n_leaves - 1 == root + 1`), which the RAFT_EXPECTS above has
    # just established. Sized `root + 1` here so the indexing is right by
    # construction rather than by coincidence -- the same value, and the
    # note is here because a reader diffing the two lines will see the
    # difference and is owed the reason.
    var relabel = List[Int](capacity=root + 1)
    var ignore = List[Int](capacity=root + 1)
    for _ in range(root + 1):
        relabel.append(0)
        ignore.append(0)
    relabel[root] = n_samples

    var out_parent = List[Int32]()
    var out_child = List[Int32]()
    var out_lambda = List[Float32]()
    var out_size = List[Int32]()

    # `:140-203` Process nodes in BFS order
    for idx in range(len(node_list)):
        var node = Int(node_list[idx])

        # `:144` Skip if already processed or is a leaf
        if ignore[node] != 0 or node < n_samples:
            continue

        # `:146-152`
        var left = Int(h_children[(node - n_samples) * 2])
        var right = Int(h_children[(node - n_samples) * 2 + 1])
        var distance = h_delta[node - n_samples]
        var lambda_value = FLOAT32_MAX
        if distance > Float32(0.0):
            if sabotage == HDB_SAB_LAMBDA_STD_DIV:
                lambda_value = Float32(1.0) / distance
            else:
                lambda_value = identical_div(Float32(1.0), distance)

        var left_count = 1
        if left >= n_samples:
            left_count = Int(h_sizes[left - n_samples])
        var right_count = 1
        if right >= n_samples:
            right_count = Int(h_sizes[right - n_samples])

        if left_count >= min_cluster_size and right_count >= min_cluster_size:
            # `:154-160` Case 1: Both children are large enough
            relabel[left] = next_label
            next_label += 1
            _add_edge(
                out_parent, out_child, out_lambda, out_size,
                relabel[node], relabel[left], lambda_value, left_count,
            )
            relabel[right] = next_label
            next_label += 1
            _add_edge(
                out_parent, out_child, out_lambda, out_size,
                relabel[node], relabel[right], lambda_value, right_count,
            )
        elif left_count < min_cluster_size and right_count < min_cluster_size:
            # `:161-178` Case 2: Both children are too small
            _collapse(
                left, node, n_samples, h_children, relabel, ignore,
                out_parent, out_child, out_lambda, out_size, lambda_value,
                sabotage,
            )
            _collapse(
                right, node, n_samples, h_children, relabel, ignore,
                out_parent, out_child, out_lambda, out_size, lambda_value,
                sabotage,
            )
        elif left_count < min_cluster_size:
            # `:180-190` Case 3: Only left child is too small
            relabel[right] = relabel[node]
            _collapse(
                left, node, n_samples, h_children, relabel, ignore,
                out_parent, out_child, out_lambda, out_size, lambda_value,
                sabotage,
            )
        else:
            # `:192-202` Case 4: Only right child is too small
            relabel[left] = relabel[node]
            _collapse(
                right, node, n_samples, h_children, relabel, ignore,
                out_parent, out_child, out_lambda, out_size, lambda_value,
                sabotage,
            )

    # `:263-272` their output arrays are `(root + 1) * 2` slots filled
    # with -1 and then written densely from the front; `condense()`
    # filters on `size != -1`. Ours appends only the live edges, which is
    # the same set in the same order -- the -1 fill exists on their side
    # because a DEVICE array has to be sized before it is written, and
    # ours is a host `List`. `condense()` still filters, so a caller that
    # passes a padded array behaves as theirs does.
    var tree = CondensedHierarchy(n_leaves)
    tree.condense(out_parent, out_child, out_lambda, out_size)
    refuse_nonfinite_host(
        tree.lambdas, "hdbscan.build_condensed_hierarchy",
        "condensed tree lambdas", sabotage,
    )
    return tree^


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

    Their three copies are one function here; the body is transcribed
    once because it is three copies of one paragraph, and
    `hdbscan/PORTED_MAP.tsv` records the fold.
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


def _copy_i32(
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


def _copy_f32(
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
