"""`make_cluster_tree`, the two `parent_csr`s, and the CSR scan they need.

PORT OF `cuml-v26.08.00/cpp/src/hdbscan/detail/utils.h` (cuML `265b9da`):
`make_cluster_tree` (`:83-140`) and `parent_csr` (`:150-170`); plus
`cuvs`/RAFT's `raft::sparse::convert::sorted_coo_to_csr` (`csr.cuh:78-90`)
as the host counting scan both of them call. Their `cub_segmented_reduce`
(`:60-76`) is NOT ported as a function: it is a CUB dispatch wrapper, and
the two reductions that use it are replaced per-call by pinned folds
(DEVIATIONS 1603 and 1604, `stabilities.mojo`). Their `normalize`
(`:172-189`) and `softmax` (`:191-220`) are reached only by soft
clustering, which `hdbscan/UNPORTED.tsv` defers.

WHERE IT RUNS. These are index plumbing over arrays of length `n_edges`
and `n_clusters`, both of which are host-resident in this lane because
their own builder is (`condense.mojo`'s header). Theirs runs the same
arithmetic as `thrust::transform` + `thrust::copy_if` + a CSR kernel.
Nothing here is a reduction over floats and nothing here has an order to
pin: a stable filter, a subtraction, and a counting scan.

THE COUNTING SCAN IS WRITTEN HERE AND IT IS A SECOND SPELLING IN THE
TREE. `spectral/ported/sparse/op/coo_ops.mojo::sorted_coo_to_csr` is the
same nine lines, over `spectral`'s own `CooGraph` struct. Importing it
would pull a graph container this lane has no other use for into the
HDBSCAN path; re-spelling it costs nine lines and is recorded here so
that a change to the counting rule is made in both places. If `core/`
ever grows a container-free CSR scan, both should call it -- the same
hand-off `hierarchy/README.md` item 5 makes for its two merge sorts.
"""

from hdbscan.ported.hdbscan.condensed_hierarchy import (
    CondensedHierarchy,
    pack_parent_child,
)
from hierarchy.ported.sparse.op.sort import merge_sort_u64_with_index


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
