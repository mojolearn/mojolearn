"""`CondensedHierarchy` and its `condense()`, the compaction and the sort.

PORT OF `cuml-v26.08.00/cpp/src/hdbscan/condensed_hierarchy.cu`
(cuML `265b9da`): `TupleComp` (`:34-49`), the two constructors this lane
reaches (`:51-102`) and `condense()` (`:133-187`). Transliterated, their
order. Do not improve.

WHERE IT LIVES. Theirs holds four `rmm::device_uvector`s; ours holds four
host `List`s. That is not a re-decision of the host/device split (rule 2)
but a consequence of one their 26.08 tree already made: their BUILDER,
`_build_condensed_hierarchy` (`detail/condense.cuh:92-212`), is a HOST
function over `std::vector`s that copies the finished arrays up at
`:205-211`. Every consumer that is a device kernel in this lane
(`compute_stabilities`, the selection BFS) uploads what it needs and the
upload is written at the call site, so a reader can see exactly which
stage crosses.

THE SORT IS A TOTAL ORDER AND THAT IS NOT AN ACCIDENT OF OURS. Their
`TupleComp` (`:34-49`) compares `parent`, then `child`, then `size`, and
`thrust::sort_by_key` is unstable -- but every node of a tree has exactly
ONE parent, so `child` is unique across the whole array and `(parent,
child)` already separates every pair of edges. Stability is therefore
moot on their side as well as ours, and this is the one sort in the lane
that needed no deviation. The third component (`size`) can never be
reached; it is transcribed anyway, because their comparator is the
algorithm and a reader diffing the two should see the same three clauses.

======================================================================
DEVIATION BLOCK -- DEVIATION 1611. THE CONDENSED-TREE SORT IS A HOST
STABLE MERGE SORT ON A PACKED (parent, child) KEY.
======================================================================
WHAT THEIRS DOES. `thrust::sort_by_key(..., TupleComp())` on the device
(`:185-186`), keys `(parents, children, sizes)`, payload `(lambdas)`.

WHAT OURS DOES. `hierarchy/ported/sparse/op/sort.mojo::
merge_sort_u64_with_index` -- the SAME host merge sort DEVIATION 621
already put under the MST, imported rather than rewritten -- on the key
`(UInt64(parent) << 32) | UInt64(child)`. Both fields are non-negative
and below `2 * n_leaves <= 2 * 46340`, so neither can reach the other's
half of the word and the unsigned order of the packed key IS their
lexicographic order. The payload moves through the index permutation.

WHY IT IS A DECLARED DEVIATION AT ALL, given the previous paragraph says
the result is the same: because WHERE it runs moved, because the key is
packed rather than compared field by field, and because "the result is
the same" is a claim a reader is entitled to see stated and gated rather
than inferred. `check_condensed_tree_vs_oracle` compares the sorted
arrays node for node against an oracle that sorts by an independent
route (a selection sort on the pair, written in the oracle so it shares
no line with this).

WHY NOT `pack_edge_key`. `hierarchy/mojo_only/edge_order.mojo`'s packing
is `(weight_key, lo, hi)` with SIXTEEN-bit vertex fields, sized for an
undirected edge of the distance graph. These are tree node ids, they are
not an undirected pair, they have no weight, and they need 32 bits. Using
it would mean truncating one order into another's layout. The ORDER is
the same idea and the LAYOUT is not, so this file packs its own two
fields and says so, rather than bending a neighbouring lane's function.
======================================================================
"""

from hierarchy.ported.sparse.op.sort import merge_sort_u64_with_index


def pack_parent_child(parent: Int32, child: Int32) -> UInt64:
    """Their `TupleComp`'s first two clauses as one UInt64 whose unsigned
    order is their lexicographic order. Both fields are non-negative tree
    node ids below `2 * n_leaves`; a negative or out-of-range value would
    alias, so `condense` refuses one by name before packing."""
    var p = UInt64(Int(parent)) & UInt64(0xFFFFFFFF)
    var c = UInt64(Int(child)) & UInt64(0xFFFFFFFF)
    return (p << UInt64(32)) | c


struct CondensedHierarchy(Copyable, Movable):
    """`hdbscan.hpp:80-124`'s `CondensedHierarchy`, minus the device
    vectors (see this file's header). `n_clusters` is `max(parents) -
    min(parents) + 1`, their formula at `:91` and `:179`."""

    var n_leaves: Int
    var n_edges: Int
    var n_clusters: Int
    var parents: List[Int32]
    var children: List[Int32]
    var lambdas: List[Float32]
    var sizes: List[Int32]

    def __init__(out self, n_leaves: Int):
        """`:51-61`, the empty hierarchy."""
        self.n_leaves = n_leaves
        self.n_edges = 0
        self.n_clusters = 0
        self.parents = List[Int32]()
        self.children = List[Int32]()
        self.lambdas = List[Float32]()
        self.sizes = List[Int32]()

    def condense(
        mut self,
        full_parents: List[Int32],
        full_children: List[Int32],
        full_lambdas: List[Float32],
        full_sizes: List[Int32],
    ) raises:
        """`condensed_hierarchy.cu:133-187`.

        `:144-151` count the entries with `size != -1`; `:158-170`
        `thrust::copy_if` the four arrays together on that predicate;
        `:172-179` `n_clusters = max(parents) - min(parents) + 1`;
        `:181-186` sort by `TupleComp`.
        """
        var size = len(full_sizes)
        # `:144-151` transform_reduce over (a != -1)
        var n_edges = 0
        for i in range(size):
            if full_sizes[i] != Int32(-1):
                n_edges += 1
        if n_edges == 0:
            raise Error(
                "hdbscan.CondensedHierarchy.condense: the condensed tree is"
                " EMPTY (0 of " + String(size) + " slots survived the size !="
                " -1 filter). Their minmax_element at condensed_hierarchy.cu"
                ":174 dereferences an empty range here and their n_clusters"
                " becomes whatever that read returned; refused by name"
                " instead. The cause is upstream of this call -- a"
                " min_cluster_size at or above n_rows leaves no cluster to"
                " condense"
            )
        # `:158-170` copy_if, in order (thrust::copy_if is stable and so
        # is this loop).
        var parents = List[Int32](capacity=n_edges)
        var children = List[Int32](capacity=n_edges)
        var lambdas = List[Float32](capacity=n_edges)
        var sizes = List[Int32](capacity=n_edges)
        for i in range(size):
            if full_sizes[i] != Int32(-1):
                parents.append(full_parents[i])
                children.append(full_children[i])
                lambdas.append(full_lambdas[i])
                sizes.append(full_sizes[i])

        # `:172-179` n_clusters = max_cluster - min_cluster + 1
        var min_cluster = parents[0]
        var max_cluster = parents[0]
        for i in range(n_edges):
            if parents[i] < min_cluster:
                min_cluster = parents[i]
            if parents[i] > max_cluster:
                max_cluster = parents[i]
        var n_clusters = Int(max_cluster) - Int(min_cluster) + 1

        # `:181-186` sort by (parent, child, size). DEVIATION 1611.
        var keys = List[UInt64](capacity=n_edges)
        var idx = List[Int](capacity=n_edges)
        for i in range(n_edges):
            if parents[i] < Int32(0) or children[i] < Int32(0):
                raise Error(
                    "hdbscan.CondensedHierarchy.condense: node id at edge "
                    + String(i) + " is negative (parent="
                    + String(Int(parents[i])) + ", child="
                    + String(Int(children[i]))
                    + "); the packed (parent, child) sort key of DEVIATION"
                    " 1611 is only order-preserving on non-negative ids"
                )
            keys.append(pack_parent_child(parents[i], children[i]))
            idx.append(i)
        merge_sort_u64_with_index(keys, idx)

        self.parents = List[Int32](capacity=n_edges)
        self.children = List[Int32](capacity=n_edges)
        self.lambdas = List[Float32](capacity=n_edges)
        self.sizes = List[Int32](capacity=n_edges)
        for k in range(n_edges):
            var i = idx[k]
            self.parents.append(parents[i])
            self.children.append(children[i])
            self.lambdas.append(lambdas[i])
            self.sizes.append(sizes[i])
        self.n_edges = n_edges
        self.n_clusters = n_clusters
