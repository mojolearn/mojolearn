# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The host oracle: HDBSCAN, serially, twice -- in Float32 bits and in
Float64.

NOT A PORT. cuML checks HDBSCAN against scikit-learn-contrib in Python
and accepts a tolerance; this tree checks a port against a host
computation whose every bit is predictable, which is what the identity
claim needs. Four things live here:

1. THE PINNED DISTANCES AND THE CORE DISTANCES, ON THE HOST. The same
   arithmetic the device's IDENTICAL arm performs, through the same
   helpers, IMPORTED from `hierarchy/mojo_only/linkage_oracle.mojo`
   (`host_row_norms_pinned`, `host_pinned_distance`) rather than
   re-derived: the distance step is that lane's and this lane reuses it
   whole. On top of it, the core distance is taken by FULLY SORTING each
   row on `(distance, index)` and reading slot `k - 1` -- a different
   route from the device's top-k selection, which is the point: an oracle
   that ran a top-k could agree with the device for a shared wrong
   reason, and a full sort cannot.

2. THE MUTUAL REACHABILITY MATRIX, by the same three-way total-order max
   the kernel uses, in the same operand order, on the host.

3. THE MST, THE DENDROGRAM AND THE CONDENSED TREE, serially. The MST is
   `host_kruskal` (IMPORTED: a different ALGORITHM from the device's
   Boruvka, sharing only the total order, so agreement is evidence).
   The dendrogram walks a `NaiveUnionFind` (IMPORTED: no path
   compression, DEVIATION 1609's control). The condensed tree is written
   HERE, and it is honest to say exactly what kind of control it is and
   what it is not.

   IT SHARES THE TRAVERSAL WITH THE PORT, AND THAT IS NOT A WEAKNESS TO
   HIDE. A level-by-level BFS from the root DEFINES the condensed tree's
   numbering; an oracle walking a different order would be checking a
   different algorithm. What differs is everything around it: the case
   tests are hoisted into named booleans, the collapse writes four
   parallel lists directly instead of through their `add_edge` closure,
   and the final sort is a SELECTION SORT on the `(parent, child)` pair
   rather than a merge sort on a packed key -- so a wrong packing
   (DEVIATION 1611), a swapped case or an off-by-one in `relabel` is
   caught. A MISUNDERSTANDING OF THE TRAVERSAL ITSELF IS NOT. The two
   controls that would catch one are the PLANTED labels
   (`hdbscan_fixture.mojo::hfixture_planted_label`) and the Float64
   reference below; both are in the gates and neither substitutes for
   the other.

4. THE FLOAT64 REFERENCE, for tolerance sanity rather than bits: direct-
   form distances `sqrt(sum (x_i - x_j)^2)` in double, the same core
   distances, the same mutual reachability, the same Kruskal, and the MST
   total. A Float32 answer that is bitwise self-consistent and 30% away
   from the Float64 one is a bug the bit gates cannot see.

NO FLOAT64 EVER GOES TO THE DEVICE. Metal has none. Everything in part 4
is host-only, which is what `PORTING_RULES.md` 0b-ii permits an oracle to
be.
"""

from std.math import sqrt

from hdbscan.mojo_only.hdbscan_fixture import (
    hfixture_d,
    hfixture_min_cluster_size,
    hfixture_min_samples,
    hfixture_n,
)
from hierarchy.mojo_only.edge_order import (
    pack_edge_key,
    unpack_edge_hi,
    unpack_edge_lo,
    weight_order_key,
)
from hierarchy.mojo_only.linkage_oracle import (
    NaiveUnionFind,
    host_kruskal,
    host_pinned_distance,
    host_row_norms_pinned,
    merge_sort_f64_with_index,
)
from hierarchy.ported.cluster.detail.connectivities import FLOAT32_MAX
from hierarchy.ported.sparse.op.sort import merge_sort_u64_with_index
from mojo_only.numerics import (
    ftz,
    identical_div,
    identical_mul,
    identical_mul_add,
    portable_fmaxf,
)


# ======================================================================
# 1. DISTANCES AND CORE DISTANCES
# ======================================================================


def oracle_pinned_distance_matrix(
    x: List[Float32], m: Int, d: Int
) -> List[Float32]:
    """The `m x m` matrix the device writes before the mutual
    reachability transform: `hierarchy`'s pinned arithmetic with FLT_MAX
    on the diagonal (`connectivities.cuh:162-175`)."""
    var norms = host_row_norms_pinned(x, m, d)
    var out = List[Float32](capacity=m * m)
    for i in range(m):
        for j in range(m):
            if i == j:
                out.append(FLOAT32_MAX)
            else:
                out.append(host_pinned_distance(x, norms, d, i, j, True))
    return out^


def oracle_core_dists(
    dists: List[Float32], m: Int, k: Int
) raises -> List[Float32]:
    """The core distance of every row: the `k`-th smallest entry of that
    row, WITH THE DIAGONAL COUNTED AS DISTANCE ZERO.

    THE DIAGONAL IS THE SELF-NEIGHBOUR AND IT IS WHY `k` IS
    `min_samples + 1`. The device gets its row from `knn_search`, which
    searches X against X and therefore returns the row's own point first
    at distance `+0.0`; the dense matrix has `FLT_MAX` on the diagonal
    instead, because `connectivities.cuh:162-175` puts it there so the MST
    cannot take a self-loop. So this oracle substitutes `+0.0` for the
    diagonal cell before sorting, which is what the k-NN result holds.

    THE SORT IS FULL AND ON `(distance, index)`, the same total order
    `knn_search` applies to its k slots -- so slot `k - 1` is the k-th
    smallest on both sides. Sorting the WHOLE row rather than selecting
    the top k is deliberate (see this file's header).
    """
    if k < 1 or k > m:
        raise Error(
            "hdbscan_oracle.oracle_core_dists: k=" + String(k)
            + " must be in [1, " + String(m) + "]"
        )
    var out = List[Float32](capacity=m)
    for i in range(m):
        var keys = List[UInt64](capacity=m)
        var idx = List[Int](capacity=m)
        for j in range(m):
            var v = dists[i * m + j]
            if i == j:
                v = Float32(0.0)
            # `pack_edge_key(weight_key(v), j, 0)` is the (distance,
            # index) total order in this tree's existing packing: the top
            # 32 bits order the value, the next 16 the index. `j < 46340`
            # by the same bound `hierarchy` refuses past.
            keys.append(pack_edge_key(weight_order_key(v), Int32(j), Int32(0)))
            idx.append(j)
        merge_sort_u64_with_index(keys, idx)
        var slot = idx[k - 1]
        var w = dists[i * m + slot]
        if i == slot:
            w = Float32(0.0)
        out.append(w)
    return out^


# ======================================================================
# 2. MUTUAL REACHABILITY
# ======================================================================


def oracle_mutual_reachability(
    dists: List[Float32], core: List[Float32], m: Int, alpha: Float32
) -> List[Float32]:
    """`ReachabilityPostProcess` on every off-diagonal cell, the diagonal
    at FLT_MAX. The max is `portable_fmaxf` -- the SAME total-order
    selection `identical_fmax` resolves to under IDENTICAL -- in THEIR
    operand order, so the oracle and the kernel agree by construction
    under IDENTICAL and the FAST comparison is a REPORT."""
    var inv_alpha = identical_div(Float32(1.0), alpha)
    var out = List[Float32](capacity=m * m)
    for i in range(m):
        for j in range(m):
            if i == j:
                out.append(FLOAT32_MAX)
            else:
                var scaled = identical_mul(inv_alpha, dists[i * m + j])
                out.append(
                    portable_fmaxf(core[j], portable_fmaxf(core[i], scaled))
                )
    return out^


# ======================================================================
# 3. DENDROGRAM AND CONDENSED TREE
# ======================================================================


@fieldwise_init
struct OracleDendrogram(Copyable, Movable):
    var children: List[Int32]
    var deltas: List[Float32]
    var sizes: List[Int32]


def oracle_dendrogram(
    lo: List[Int32], hi: List[Int32], w: List[Float32], m: Int
) -> OracleDendrogram:
    """`build_dendrogram_host` (`agglomerative.cuh:104-155`) over a
    compression-free union-find, returning all THREE of its outputs --
    `hierarchy/mojo_only/linkage_oracle.mojo::host_dendrogram` returns
    only `children`, and the condensed tree needs `delta` and `size` too.
    The union-find itself is IMPORTED from that file."""
    var uf = NaiveUnionFind(m)
    var children = List[Int32](capacity=(m - 1) * 2)
    var deltas = List[Float32](capacity=m - 1)
    var sizes = List[Int32](capacity=m - 1)
    for i in range(m - 1):
        var aa = uf.find(Int(lo[i]))
        var bb = uf.find(Int(hi[i]))
        children.append(Int32(aa))
        children.append(Int32(bb))
        deltas.append(w[i])
        sizes.append(Int32(uf.size[aa] + uf.size[bb]))
        uf.union(aa, bb)
    return OracleDendrogram(children^, deltas^, sizes^)


@fieldwise_init
struct OracleCondensed(Copyable, Movable):
    var n_leaves: Int
    var n_edges: Int
    var n_clusters: Int
    var parents: List[Int32]
    var children: List[Int32]
    var lambdas: List[Float32]
    var sizes: List[Int32]


def oracle_condense(
    dend: OracleDendrogram, n_leaves: Int, min_cluster_size: Int
) raises -> OracleCondensed:
    """`condense.cuh:91-212`, spelled with the BFS levels MATERIALIZED.

    The port carries one `process_queue` and rebuilds it; this walks a
    `List` of levels and concatenates them. Same order, different
    bookkeeping -- see this file's header for exactly what that does and
    does not control.
    """
    var root = 2 * (n_leaves - 1)
    var n_samples = n_leaves
    var next_label = n_samples + 1

    var node_list = _oracle_bfs(root, n_samples, dend.children)
    var relabel = List[Int](capacity=root + 1)
    var ignore = List[Int](capacity=root + 1)
    for _ in range(root + 1):
        relabel.append(0)
        ignore.append(0)
    relabel[root] = n_samples

    var parents = List[Int32]()
    var children = List[Int32]()
    var lambdas = List[Float32]()
    var sizes = List[Int32]()

    for t in range(len(node_list)):
        var node = node_list[t]
        if ignore[node] != 0 or node < n_samples:
            continue
        var left = Int(dend.children[(node - n_samples) * 2])
        var right = Int(dend.children[(node - n_samples) * 2 + 1])
        var distance = dend.deltas[node - n_samples]
        var lam = FLOAT32_MAX
        if distance > Float32(0.0):
            lam = identical_div(Float32(1.0), distance)
        var left_count = 1
        if left >= n_samples:
            left_count = Int(dend.sizes[left - n_samples])
        var right_count = 1
        if right >= n_samples:
            right_count = Int(dend.sizes[right - n_samples])

        var big_left = left_count >= min_cluster_size
        var big_right = right_count >= min_cluster_size
        if big_left and big_right:
            relabel[left] = next_label
            next_label += 1
            parents.append(Int32(relabel[node]))
            children.append(Int32(relabel[left]))
            lambdas.append(lam)
            sizes.append(Int32(left_count))
            relabel[right] = next_label
            next_label += 1
            parents.append(Int32(relabel[node]))
            children.append(Int32(relabel[right]))
            lambdas.append(lam)
            sizes.append(Int32(right_count))
        elif not big_left and not big_right:
            _oracle_collapse(
                left, node, n_samples, dend.children, relabel, ignore,
                parents, children, lambdas, sizes, lam,
            )
            _oracle_collapse(
                right, node, n_samples, dend.children, relabel, ignore,
                parents, children, lambdas, sizes, lam,
            )
        elif not big_left:
            relabel[right] = relabel[node]
            _oracle_collapse(
                left, node, n_samples, dend.children, relabel, ignore,
                parents, children, lambdas, sizes, lam,
            )
        else:
            relabel[left] = relabel[node]
            _oracle_collapse(
                right, node, n_samples, dend.children, relabel, ignore,
                parents, children, lambdas, sizes, lam,
            )

    var n_edges = len(parents)
    if n_edges == 0:
        raise Error(
            "hdbscan_oracle.oracle_condense: the condensed tree is empty at"
            " min_cluster_size=" + String(min_cluster_size)
        )
    var min_p = parents[0]
    var max_p = parents[0]
    for i in range(n_edges):
        if parents[i] < min_p:
            min_p = parents[i]
        if parents[i] > max_p:
            max_p = parents[i]
    var n_clusters = Int(max_p) - Int(min_p) + 1

    # The (parent, child) sort, by SELECTION SORT rather than the port's
    # merge sort on a packed key. A different sorting algorithm on the
    # same total order: if the packing is wrong the two disagree.
    var order = List[Int](capacity=n_edges)
    for i in range(n_edges):
        order.append(i)
    for a in range(n_edges):
        var best = a
        for b in range(a + 1, n_edges):
            var ib = order[b]
            var ibest = order[best]
            var less = False
            if parents[ib] < parents[ibest]:
                less = True
            elif parents[ib] == parents[ibest]:
                if children[ib] < children[ibest]:
                    less = True
            if less:
                best = b
        var tmp = order[a]
        order[a] = order[best]
        order[best] = tmp

    var sp = List[Int32](capacity=n_edges)
    var sc = List[Int32](capacity=n_edges)
    var sl = List[Float32](capacity=n_edges)
    var ss = List[Int32](capacity=n_edges)
    for k in range(n_edges):
        var i = order[k]
        sp.append(parents[i])
        sc.append(children[i])
        sl.append(lambdas[i])
        ss.append(sizes[i])
    return OracleCondensed(n_leaves, n_edges, n_clusters, sp^, sc^, sl^, ss^)


def _oracle_bfs(
    bfs_root: Int, n_samples: Int, children: List[Int32]
) -> List[Int]:
    """`condense.cuh:37-66`'s level-by-level order. This is the ONE part
    of the condensed-tree oracle that is not an independent spelling; the
    file header says why and what stands in for it."""
    var out = List[Int]()
    var level = List[Int]()
    level.append(bfs_root)
    while len(level) > 0:
        for t in range(len(level)):
            out.append(level[t])
        var nxt = List[Int]()
        for t in range(len(level)):
            var x = level[t]
            if x >= n_samples:
                var h = x - n_samples
                nxt.append(Int(children[h * 2]))
                nxt.append(Int(children[h * 2 + 1]))
        level = nxt.copy()
    return out^


def _oracle_collapse(
    subtree_root: Int,
    node: Int,
    n_samples: Int,
    children_in: List[Int32],
    relabel: List[Int],
    mut ignore: List[Int],
    mut parents: List[Int32],
    mut children: List[Int32],
    mut lambdas: List[Float32],
    mut sizes: List[Int32],
    lam: Float32,
):
    var desc = _oracle_bfs(subtree_root, n_samples, children_in)
    for t in range(len(desc)):
        var sub = desc[t]
        if sub < n_samples:
            parents.append(Int32(relabel[node]))
            children.append(Int32(sub))
            lambdas.append(lam)
            sizes.append(Int32(1))
        ignore[sub] = 1


# ======================================================================
# STABILITIES, SELECTION AND LABELS, SERIALLY
# ======================================================================


def oracle_indptr(tree: OracleCondensed) -> List[Int32]:
    """The CSR offsets over `parents - n_leaves`, by a running scan rather
    than a counting scan -- a second spelling of `sorted_rows_to_indptr`
    that reads the sorted array once."""
    var out = List[Int32](capacity=tree.n_clusters + 1)
    var e = 0
    for c in range(tree.n_clusters):
        out.append(Int32(e))
        while e < tree.n_edges and Int(tree.parents[e]) - tree.n_leaves == c:
            e += 1
    out.append(Int32(e))
    return out^


def oracle_stabilities(tree: OracleCondensed) -> List[Float32]:
    """DEVIATIONS 1603 and 1604 on the host, term for term: the segment
    minimum on the integer key, the `births` initialization from the
    child slot, and the ascending fold through `ftz`/`identical_mul_add`."""
    var indptr = oracle_indptr(tree)
    var births = List[Float32](capacity=tree.n_clusters)
    for _ in range(tree.n_clusters):
        births.append(Float32(0.0))
    for i in range(tree.n_edges):
        var child = Int(tree.children[i])
        if child >= tree.n_leaves:
            births[child - tree.n_leaves] = tree.lambdas[i]
    var out = List[Float32](capacity=tree.n_clusters)
    for c in range(tree.n_clusters):
        var lo = Int(indptr[c])
        var hi = Int(indptr[c + 1])
        var seg_min = FLOAT32_MAX
        for i in range(lo, hi):
            if weight_order_key(tree.lambdas[i]) < weight_order_key(seg_min):
                seg_min = tree.lambdas[i]
        var birth = births[c]
        if c > 0:
            if weight_order_key(seg_min) < weight_order_key(birth):
                birth = seg_min
            births[c] = birth
        var acc = Float32(0.0)
        for i in range(lo, hi):
            var term = ftz(tree.lambdas[i] - birth)
            acc = ftz(
                identical_mul_add(term, Float32(Int(tree.sizes[i])), acc)
            )
        out.append(acc)
    return out^


@fieldwise_init
struct OracleSelection(Copyable, Movable):
    var is_cluster: List[Int32]
    var stabilities: List[Float32]
    """AFTER the Excess-of-Mass write-back."""
    var cluster_sizes: List[Int]


def oracle_excess_of_mass(
    tree: OracleCondensed,
    stabilities_in: List[Float32],
    max_cluster_size: Int,
    allow_single_cluster: Bool,
) raises -> OracleSelection:
    """`select.cuh:148-252` serially, over the CLUSTER TREE (edges with
    size > 1, ids shifted down by `n_leaves`)."""
    var n_clusters = tree.n_clusters
    var cp = List[Int32]()
    var cc = List[Int32]()
    var cs = List[Int32]()
    for i in range(tree.n_edges):
        if tree.sizes[i] > Int32(1):
            cp.append(tree.parents[i] - Int32(tree.n_leaves))
            cc.append(tree.children[i] - Int32(tree.n_leaves))
            cs.append(tree.sizes[i])
    var n_edges = len(cp)

    var cluster_sizes = List[Int](capacity=n_clusters)
    for _ in range(n_clusters):
        cluster_sizes.append(0)
    for i in range(n_edges):
        if Int(cp[i]) == 0:
            cluster_sizes[0] += Int(cs[i])
        cluster_sizes[Int(cc[i])] = Int(cs[i])

    var indptr = List[Int32](capacity=n_clusters + 1)
    var e = 0
    for c in range(n_clusters):
        indptr.append(Int32(e))
        while e < n_edges and Int(cp[e]) == c:
            e += 1
    indptr.append(Int32(e))

    var stab = stabilities_in.copy()
    var is_cluster = List[Int32](capacity=n_clusters)
    var frontier = List[Int32](capacity=n_clusters)
    for _ in range(n_clusters):
        is_cluster.append(Int32(1))
        frontier.append(Int32(0))
    is_cluster[0] = Int32(1) if allow_single_cluster else Int32(0)

    var tree_top = 0 if allow_single_cluster else 1
    var node = n_clusters - 1
    while node >= tree_top:
        var lo = Int(indptr[node])
        var hi = Int(indptr[node + 1])
        var subtree = Float32(0.0)
        for i in range(lo, hi):
            subtree = ftz(
                identical_mul_add(Float32(1.0), stab[Int(cc[i])], subtree)
            )
        if subtree > stab[node] or cluster_sizes[node] > max_cluster_size:
            stab[node] = subtree
            is_cluster[node] = Int32(0)
        else:
            frontier[node] = Int32(1)
        node -= 1

    # The negation BFS, serially: a node on the frontier deselects its
    # children and puts them on the frontier.
    var queue = List[Int]()
    for c in range(n_clusters):
        if frontier[c] != Int32(0):
            queue.append(c)
    var qi = 0
    while qi < len(queue):
        var c = queue[qi]
        qi += 1
        var lo = Int(indptr[c])
        var hi = Int(indptr[c + 1])
        for i in range(lo, hi):
            var child = Int(cc[i])
            is_cluster[child] = Int32(0)
            queue.append(child)
    return OracleSelection(is_cluster^, stab^, cluster_sizes^)


def oracle_leaf_selection(tree: OracleCondensed) -> List[Int32]:
    """`select.cuh:264-286` serially."""
    var n_clusters = tree.n_clusters
    var is_parent = List[Int32](capacity=n_clusters)
    var out = List[Int32](capacity=n_clusters)
    for _ in range(n_clusters):
        is_parent.append(Int32(0))
        out.append(Int32(0))
    for i in range(tree.n_edges):
        if tree.sizes[i] > Int32(1):
            is_parent[Int(tree.parents[i]) - tree.n_leaves] = Int32(1)
    for i in range(tree.n_edges):
        if tree.sizes[i] > Int32(1):
            var c = Int(tree.children[i]) - tree.n_leaves
            if is_parent[c] == Int32(0):
                out[c] = Int32(1)
    return out^


def oracle_labels(
    tree: OracleCondensed, is_cluster: List[Int32], allow_single_cluster: Bool
) raises -> List[Int32]:
    """`extract.cuh:88-167` serially, over a NON-COMPRESSING union-find
    (`NaiveUnionFind`'s idea, spelled here because their `TreeUnionFind`
    unions by RANK and `NaiveUnionFind` unions by creating a new node, so
    the two are not interchangeable -- what is shared is the absence of
    path compression, which is the control DEVIATION 1609 needs)."""
    var n_leaves = tree.n_leaves
    var size = Int(tree.parents[0])
    for i in range(tree.n_edges):
        if Int(tree.parents[i]) > size:
            size = Int(tree.parents[i])

    var parent = List[Int](capacity=size + 1)
    var rank = List[Int](capacity=size + 1)
    for i in range(size + 1):
        parent.append(i)
        rank.append(0)

    var in_clusters = List[Int32](capacity=size + 1)
    for _ in range(size + 1):
        in_clusters.append(Int32(0))
    for c in range(tree.n_clusters):
        if is_cluster[c] != Int32(0):
            var node = c + n_leaves
            if node <= size:
                in_clusters[node] = Int32(1)

    var parent_lambdas = List[Float32](capacity=size + 1)
    for _ in range(size + 1):
        parent_lambdas.append(Float32(0.0))

    for i in range(tree.n_edges):
        var child = Int(tree.children[i])
        var par = Int(tree.parents[i])
        if child > size or in_clusters[child] == Int32(0):
            var rp = _naive_find(parent, par)
            var rc = _naive_find(parent, child)
            _naive_union(parent, rank, rp, rc)
        if weight_order_key(tree.lambdas[i]) > weight_order_key(
            parent_lambdas[par]
        ):
            parent_lambdas[par] = tree.lambdas[i]

    var n_in = 0
    for i in range(size + 1):
        if in_clusters[i] != Int32(0):
            n_in += 1

    var out = List[Int32](capacity=n_leaves)
    for i in range(n_leaves):
        var cluster = _naive_find(parent, i)
        if cluster < n_leaves:
            out.append(Int32(-1))
        elif cluster == n_leaves:
            if n_in == 1 and allow_single_cluster:
                var child_idx = -1
                for e in range(tree.n_edges):
                    if Int(tree.children[e]) == i:
                        child_idx = e
                        break
                if child_idx < 0:
                    raise Error(
                        "hdbscan_oracle.oracle_labels: point " + String(i)
                        + " is not a child of any condensed edge"
                    )
                if weight_order_key(tree.lambdas[child_idx]) >= weight_order_key(
                    parent_lambdas[cluster]
                ):
                    out.append(Int32(cluster - n_leaves))
                else:
                    out.append(Int32(-1))
            else:
                out.append(Int32(-1))
        else:
            out.append(Int32(cluster - n_leaves))
    return out^


def _naive_find(parent: List[Int], x: Int) -> Int:
    """No path compression, on purpose (DEVIATION 1609's control)."""
    var n = x
    while parent[n] != n:
        n = parent[n]
    return n


def _naive_union(mut parent: List[Int], mut rank: List[Int], a: Int, b: Int):
    """Union by rank, their three branches (`extract.cuh:59-72`). The RANK
    rule is part of the answer -- it decides which id becomes the root and
    therefore which label a point gets -- so it is transcribed rather than
    replaced by a simpler rule."""
    if a == b:
        return
    if rank[a] < rank[b]:
        parent[a] = b
    elif rank[a] > rank[b]:
        parent[b] = a
    else:
        parent[b] = a
        rank[a] += 1


def oracle_final_labels(
    raw_labels: List[Int32], is_cluster: List[Int32], n_clusters: Int
) -> Tuple[List[Int32], Int]:
    """`runner.h:221-233`'s remap, plus the outlier count."""
    var label_map = List[Int32](capacity=n_clusters)
    for _ in range(n_clusters):
        label_map.append(Int32(-1))
    var n_selected = 0
    for c in range(n_clusters):
        if is_cluster[c] != Int32(0):
            label_map[c] = Int32(n_selected)
            n_selected += 1
    var out = List[Int32](capacity=len(raw_labels))
    var n_out = 0
    for i in range(len(raw_labels)):
        var l = raw_labels[i]
        if l != Int32(-1):
            out.append(label_map[Int(l)])
        else:
            out.append(Int32(-1))
        if out[i] == Int32(-1):
            n_out += 1
    return (out^, n_out)


# ======================================================================
# 4. THE FLOAT64 REFERENCE
# ======================================================================


def oracle_f64_core_dists(
    x: List[Float32], m: Int, d: Int, k: Int
) -> List[Float64]:
    """Core distances from DIRECT-FORM Float64 distances,
    `sqrt(sum (x_i - x_j)^2)` -- not the expanded identity, so a
    catastrophic cancellation in the Float32 path shows up here as a
    disagreement rather than being reproduced."""
    var out = List[Float64](capacity=m)
    var row = List[Float64](capacity=m)
    var idx = List[Int](capacity=m)
    for i in range(m):
        row.clear()
        idx.clear()
        for j in range(m):
            var s = Float64(0.0)
            for f in range(d):
                var diff = Float64(x[i * d + f]) - Float64(x[j * d + f])
                s = s + diff * diff
            row.append(sqrt(s))
            idx.append(j)
        merge_sort_f64_with_index(row, idx)
        out.append(row[k - 1])
    return out^


def oracle_f64_mst_total(
    x: List[Float32], m: Int, d: Int, k: Int
) -> Float64:
    """The total weight of the MST of the Float64 mutual reachability
    graph, by Kruskal with ties in `(i, j)` order. The tolerance control
    for the whole Float32 chain up to the MST."""
    var core = oracle_f64_core_dists(x, m, d, k)
    var n_pairs = m * (m - 1) // 2
    var keys = List[Float64](capacity=n_pairs)
    var idx = List[Int](capacity=n_pairs)
    var pi = List[Int](capacity=n_pairs)
    var pj = List[Int](capacity=n_pairs)
    var t = 0
    for i in range(m):
        for j in range(i + 1, m):
            var s = Float64(0.0)
            for f in range(d):
                var diff = Float64(x[i * d + f]) - Float64(x[j * d + f])
                s = s + diff * diff
            var dij = sqrt(s)
            var mr = dij
            if core[i] > mr:
                mr = core[i]
            if core[j] > mr:
                mr = core[j]
            keys.append(mr)
            idx.append(t)
            pi.append(i)
            pj.append(j)
            t += 1
    merge_sort_f64_with_index(keys, idx)
    var uf = NaiveUnionFind(m)
    var total = Float64(0.0)
    var taken = 0
    for s in range(n_pairs):
        var u = pi[idx[s]]
        var v = pj[idx[s]]
        var ru = uf.find(u)
        var rv = uf.find(v)
        if ru != rv:
            uf.union(ru, rv)
            total = total + keys[s]
            taken += 1
            if taken == m - 1:
                break
    return total


# ======================================================================
# THE WHOLE ORACLE FOR ONE FIXTURE
# ======================================================================


@fieldwise_init
struct OracleRun(Copyable, Movable):
    var m: Int
    var k: Int
    var dists: List[Float32]
    var core: List[Float32]
    var mr: List[Float32]
    var mst_lo: List[Int32]
    var mst_hi: List[Int32]
    var mst_w: List[Float32]
    var dendrogram: OracleDendrogram
    var condensed: OracleCondensed
    var stabilities: List[Float32]
    var selection: OracleSelection
    var raw_labels: List[Int32]
    var labels: List[Int32]
    var n_outliers: Int
    var mst_total_f32_in_f64: Float64
    var mst_total_f64: Float64


def oracle_run(fix: Int, x: List[Float32]) raises -> OracleRun:
    """Every stage, for one fixture, at that fixture's parameters."""
    var m = hfixture_n(fix)
    var d = hfixture_d(fix)
    var min_samples = hfixture_min_samples(fix)
    var mcs = hfixture_min_cluster_size(fix)
    # `runner.h:68-80`'s `min_samples + 1`, and its clamp.
    var k = min_samples + 1
    if k > m:
        k = m
    var dists = oracle_pinned_distance_matrix(x, m, d)
    var core = oracle_core_dists(dists, m, k)
    var mr = oracle_mutual_reachability(dists, core, m, Float32(1.0))
    var mst = host_kruskal(mr, m)
    var dend = oracle_dendrogram(mst[0], mst[1], mst[2], m)
    var cond = oracle_condense(dend, m, mcs)
    var stab = oracle_stabilities(cond)
    var sel = oracle_excess_of_mass(cond, stab, m, False)
    var raw = oracle_labels(cond, sel.is_cluster, False)
    var final = oracle_final_labels(raw, sel.is_cluster, cond.n_clusters)
    var total32 = Float64(0.0)
    for i in range(m - 1):
        total32 = total32 + Float64(mst[2][i])
    var total64 = oracle_f64_mst_total(x, m, d, k)
    return OracleRun(
        m, k, dists^, core^, mr^, mst[0].copy(), mst[1].copy(),
        mst[2].copy(), dend^, cond^, stab^, sel^, raw.copy(),
        final[0].copy(), final[1], total32, total64,
    )
