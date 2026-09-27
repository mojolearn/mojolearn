# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`make_cluster_tree`, the two `parent_csr`s, and the CSR scan they need.

Reference: `cuml-v26.08.00/cpp/src/hdbscan/detail/utils.h` (cuML `265b9da`):
`make_cluster_tree` (`:83-140`) and `parent_csr` (`:150-170`); plus
`cuvs`/RAFT's `raft::sparse::convert::sorted_coo_to_csr` (`csr.cuh:78-90`)
as the host counting scan both of them call. Their `cub_segmented_reduce`
(`:60-76`) is NOT implemented as a function: it is a CUB dispatch wrapper, and
the two reductions that use it are replaced per-call by pinned folds
(DEVIATIONS 1603 and 1604, `stabilities.mojo`). Their `normalize`
(`:172-189`) and `softmax` (`:191-220`) are reached only by soft
clustering, which `hdbscan/NOT_IMPLEMENTED.tsv` defers.

WHERE IT RUNS. These are index plumbing over arrays of length `n_edges`
and `n_clusters`, both of which are host-resident in this lane because
their own builder is (`condense.mojo`'s header). Theirs runs the same
arithmetic as `thrust::transform` + `thrust::copy_if` + a CSR kernel.
Nothing here is a reduction over floats and nothing here has an order to
pin: a stable filter, a subtraction, and a counting scan.

THE COUNTING SCAN IS WRITTEN HERE AND IT IS A SECOND SPELLING IN THE
TREE. `spectral/impl/sparse/op/coo_ops.mojo::sorted_coo_to_csr` is the
same nine lines, over `spectral`'s own `CooGraph` struct. Importing it
would pull a graph container this lane has no other use for into the
HDBSCAN path; re-spelling it costs nine lines and is recorded here so
that a change to the counting rule is made in both places. If `core/`
ever grows a container-free CSR scan, both should call it -- the same
hand-off `hierarchy/README.md` item 5 makes for its two merge sorts.
"""

from hdbscan.impl.condensed_hierarchy import (
    CondensedHierarchy,
    pack_parent_child,
)
from hierarchy.impl.sparse.op.sort import merge_sort_u64_with_index
from checks.numerics import identical_div


def sorted_rows_to_indptr(rows: List[Int32], n_rows: Int) raises -> List[Int32]:
    """`raft::sparse::convert::sorted_coo_to_csr` (`csr.cuh:78-90`) plus
    the `n_rows + 1`-th entry holding `nnz`, which their callers get from
    `get_stop_idx`. `rows` must be sorted ascending and every entry in
    `[0, n_rows)`; an out-of-range row would silently drop an edge from
    every segment after it, so it is refused by name."""
    var counts = List[Int32](capacity=n_rows)
    for _ in range(n_rows):
        counts.append(Int32(0))
    for i in range(len(rows)):
        var r = Int(rows[i])
        if r < 0 or r >= n_rows:
            raise Error(
                "hdbscan.sorted_rows_to_indptr: row " + String(r) + " at"
                " position " + String(i) + " is outside [0, "
                + String(n_rows) + "); the CSR segment boundaries index"
                " every per-cluster array in this lane, so an out-of-range"
                " row is refused rather than dropped"
            )
        counts[r] += Int32(1)
    var indptr = List[Int32](capacity=n_rows + 1)
    var acc = Int32(0)
    for r in range(n_rows):
        indptr.append(acc)
        acc += counts[r]
    indptr.append(acc)
    return indptr^


def make_cluster_tree(
    tree: CondensedHierarchy,
) raises -> CondensedHierarchy:
    """`utils.h:83-140`: "Constructs a cluster tree from a
    CondensedHierarchy by filtering for only entries with cluster size >
    1", then subtracting `n_leaves` from both parents and children so the
    result is 0-indexed in cluster space (`:118-130`).

    `n_clusters` is CARRIED OVER unchanged (`:135`, their constructor
    argument is `condensed_tree.get_n_clusters()`), NOT recomputed from
    the filtered parents. That matters: `is_cluster`, `stability` and
    `cluster_sizes` are all `n_clusters` long and are indexed by the
    filtered tree's ids, so a recomputed (smaller) count would silently
    shorten every one of them. Transcribed, not improved.
    """
    var n_leaves = tree.n_leaves
    var parents = List[Int32]()
    var children = List[Int32]()
    var lambdas = List[Float32]()
    var sizes = List[Int32]()
    # `:92-117` transform_reduce(size > 1) then copy_if on the same
    # predicate. `thrust::copy_if` is stable and so is this loop, so the
    # cluster tree inherits the condensed tree's (parent, child) order.
    for i in range(tree.n_edges):
        if tree.sizes[i] > Int32(1):
            parents.append(tree.parents[i] - Int32(n_leaves))
            children.append(tree.children[i] - Int32(n_leaves))
            lambdas.append(tree.lambdas[i])
            sizes.append(tree.sizes[i])
    var out = CondensedHierarchy(n_leaves)
    out.n_edges = len(parents)
    out.n_clusters = tree.n_clusters
    out.parents = parents^
    out.children = children^
    out.lambdas = lambdas^
    out.sizes = sizes^
    return out^


def utils_parent_csr(tree: CondensedHierarchy) raises -> List[Int32]:
    """`utils.h:150-170` `Utils::parent_csr`, the one `compute_stabilities`
    and `get_probabilities` call: 0-index the sorted parents by
    subtracting `n_leaves` (`:165-167`), then `sorted_coo_to_csr` over
    them into `n_clusters + 1` offsets (`:169`).

    The condensed tree is ALREADY sorted by `(parent, child)`
    (`condensed_hierarchy.mojo`), which is what makes their
    `sorted_coo_to_csr` legal on it; their own `sorted_parents` copy at
    `stabilities.cuh:65-66` exists because they transform in place and
    must not disturb the tree. Ours reads without writing, so no copy is
    needed and none is made.
    """
    var rows = List[Int32](capacity=tree.n_edges)
    for i in range(tree.n_edges):
        rows.append(tree.parents[i] - Int32(tree.n_leaves))
    return sorted_rows_to_indptr(rows, tree.n_clusters)


def select_parent_csr(tree: CondensedHierarchy) raises -> List[Int32]:
    """`select.cuh:103-130` `Select::parent_csr`, the one `excess_of_mass`
    and `cluster_epsilon_search` call. Theirs `coo_sort`s the CLUSTER TREE
    in place on `(parents, children)` first (`:117-123`) and then runs the
    CSR scan; the empty case fills the offsets with zero (`:127-129`).

    The sort is a no-op on a cluster tree derived from a condensed tree
    that is already in that order, and it is run anyway -- their line is
    their line, and a caller who ever hands this an unsorted tree gets
    their behavior rather than a silently wrong CSR. Ours sorts the KEY
    ORDER and checks that the result is the identity permutation only in
    the check, never here.
    """
    if tree.n_edges == 0:
        var zeros = List[Int32](capacity=tree.n_clusters + 1)
        for _ in range(tree.n_clusters + 1):
            zeros.append(Int32(0))
        return zeros^
    var keys = List[UInt64](capacity=tree.n_edges)
    var idx = List[Int](capacity=tree.n_edges)
    for i in range(tree.n_edges):
        keys.append(pack_parent_child(tree.parents[i], tree.children[i]))
        idx.append(i)
    merge_sort_u64_with_index(keys, idx)
    var rows = List[Int32](capacity=tree.n_edges)
    for k in range(tree.n_edges):
        rows.append(tree.parents[idx[k]])
    return sorted_rows_to_indptr(rows, tree.n_clusters)


def cluster_epsilon_search_host(
    cluster_tree: CondensedHierarchy,
    mut is_cluster: List[Int32],
    n_clusters: Int,
    cluster_selection_epsilon: Float32,
    allow_single_cluster: Bool,
) raises:
    """`select.cuh:301-363` + `kernels/select.cuh:47-104`, DEVIATION 5115
    (the cluster lane): the epsilon search on the HOST, one source for the
    GPU route (`select.mojo::select_clusters`) and the CPU oracle
    (`hdbscan_host_oracle.mojo::hdbh_select`).

    Theirs sorts `(parents, lambdas)` BY CHILD in place (`:328-329`) so
    that `child_idx = child - 1` indexes them; a cluster tree has every
    non-root cluster as a child exactly once, so this builds that map
    directly (`parent_of[c]`, `lambda_of[c]`) without touching the tree.
    Each selected cluster whose `eps = 1 / lambda` (`identical_div`) is
    below the threshold walks up while the parent's eps is `<=` it and
    selects where it stops (the root only under `allow_single_cluster`,
    else itself: `kernels/select.cuh:70-94`); every other selected cluster
    goes on the frontier (`:100-102`). Then `perform_bfs` with
    `propagate_cluster_negation_kernel` deselects every descendant of the
    frontier (`:357-362`), serially: every write in both is a constant, so
    the processing order cannot move a bit (the kernel's own argument).
    Theirs agrees with scikit-learn's `epsilon_search` (`_tree.pyx`,
    `traverse_upwards`).
    """
    var parent_of = List[Int32](length=n_clusters, fill=Int32(-1))
    var eps_of = List[Float32](length=n_clusters, fill=Float32(0))
    for i in range(cluster_tree.n_edges):
        var c = Int(cluster_tree.children[i])
        if c <= 0 or c >= n_clusters:
            raise Error(
                "hdbscan.cluster_epsilon_search: cluster-tree child "
                + String(c) + " at edge " + String(i) + " is outside [1, "
                + String(n_clusters) + ")"
            )
        parent_of[c] = cluster_tree.parents[i]
        # `:330-334` eps = 1 / x
        eps_of[c] = identical_div(Float32(1.0), cluster_tree.lambdas[i])
    var selected = List[Int]()
    for c in range(n_clusters):
        if is_cluster[c] != Int32(0):
            selected.append(c)
    var frontier = List[Int32](length=n_clusters, fill=Int32(0))
    for s in range(len(selected)):
        var child = selected[s]
        # `:66` the root takes no part.
        if child == 0:
            continue
        if eps_of[child] < cluster_selection_epsilon:
            var parent = 0
            while True:
                parent = Int(parent_of[child])
                if parent == 0:
                    if not allow_single_cluster:
                        parent = child
                    break
                child = parent
                var parent_eps = eps_of[child]
                if not (parent_eps <= cluster_selection_epsilon):
                    break
            frontier[parent] = Int32(1)
            is_cluster[parent] = Int32(1)
        else:
            frontier[child] = Int32(1)
    # `perform_bfs` over `select_parent_csr` (`:354-362`).
    var indptr = select_parent_csr(cluster_tree)
    var n_left = 0
    for i in range(n_clusters):
        n_left += Int(frontier[i])
    while n_left > 0:
        var next_frontier = List[Int32](length=n_clusters, fill=Int32(0))
        for cluster in range(n_clusters):
            if frontier[cluster] == Int32(0):
                continue
            for i in range(Int(indptr[cluster]), Int(indptr[cluster + 1])):
                var ch = Int(cluster_tree.children[i])
                next_frontier[ch] = Int32(1)
                is_cluster[ch] = Int32(0)
        frontier = next_frontier^
        n_left = 0
        for i in range(n_clusters):
            n_left += Int(frontier[i])
