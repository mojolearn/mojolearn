# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Agglomerative clustering, every linkage (lane/algos-cluster, option parity).
Reference: scikit-learn `sklearn/cluster/_agglomerative.py` (`ward_tree`
:196-430, `linkage_tree` :433-700, `_fix_connectivity` :37-100,
`_fix_connected_components` in `utils/graph.py`), scipy
`cluster/_hierarchy.pyx::nn_chain` and `_hierarchy_distance_update.pxi`.

The dense n x n dissimilarities are the device's (`ClusterOps.sqdist` /
`pdist`, DEVIATIONS 5100, 5101, 5111). THE MERGE LOOP IS SEQUENTIAL HOST
CODE, one source compiled into both bindings, as OPTICS's ordering loop:

- every step merges the pair (i < j) of live clusters with the LOWEST
  dissimilarity among the pairs the connectivity allows (every pair without
  one), THE LOWEST i THEN THE LOWEST j ON A TIE. This is scikit-learn's
  connectivity heap (the global minimum each step) and, for the four
  reducible linkages, the same dendrogram scipy's nn-chain returns once
  sorted; only the order of exactly tied merges can differ. DEVIATION 5118.
- the merged cluster keeps the lower slot; its row is the Lance-Williams
  update (`bodies.lance_williams`, DEVIATION 5117) against every live
  cluster; its node id is n + step.
- ward runs on SQUARED euclidean dissimilarities and reports the root of
  each merge's value, which is scikit-learn's `sqrt(2 * inertia)` and
  scipy's ward distance.
- a connectivity graph with several components is joined first, as
  scikit-learn's `_fix_connected_components`: for every pair of components
  (i, j < i, in the order components are found from vertex 0) the closest
  pair of points, the lowest (row of i, row of j) on a tie, becomes an edge.
"""
from checks.numerics import identical_sqrt
from x_cluster.bodies import LINK_SINGLE, LINK_WARD, lance_williams
from x_cluster.ops import ClusterOps


def _components(adj: List[Bool], n: Int, mut comp: List[Int]) -> Int:
    """Connected components by a breadth-first walk from the lowest unseen
    vertex; component ids in the order found (scipy's connected_components
    numbering)."""
    comp = List[Int](length=n, fill=-1)
    var c = 0
    var queue = List[Int](capacity=n)
    for s in range(n):
        if comp[s] >= 0:
            continue
        comp[s] = c
        queue.clear()
        queue.append(s)
        var h = 0
        while h < len(queue):
            var u = queue[h]
            h += 1
            for v in range(n):
                if adj[u * n + v] and comp[v] < 0:
                    comp[v] = c
                    queue.append(v)
        c += 1
    return c


def _row_min(
    i: Int, n: Int, live: List[Bool], adj: List[Bool], constrained: Bool, dm: List[Float32],
    mut nn: List[Int], mut md: List[Float32],
):
    """nn[i], md[i] = the live partner j > i (an edge when constrained) at the
    lowest dissimilarity, the lowest j on a tie; -1 and +inf when none."""
    var best = -1
    var bv = Float32.MAX * Float32(2)
    for j in range(i + 1, n):
        if not live[j]:
            continue
        if constrained and not adj[i * n + j]:
            continue
        var v = dm[i * n + j]
        if best < 0 or v < bv:
            best = j
            bv = v
    nn[i] = best
    md[i] = bv


comptime _RW = 8  # SIMD width of the unconstrained row scan


@always_inline
def _rescan(
    i: Int, n: Int, live: List[Bool], adj: List[Bool], constrained: Bool, dm: List[Float32],
    dead: List[Float32], mut nn: List[Int], mut md: List[Float32],
):
    if constrained:
        _row_min(i, n, live, adj, constrained, dm, nn, md)
    else:
        _row_min_open(i, n, live, dm, dead, nn, md)


def _row_min_open(
    i: Int, n: Int, live: List[Bool], dm: List[Float32], dead: List[Float32], mut nn: List[Int],
    mut md: List[Float32],
):
    """`_row_min` without a connectivity graph, on a matrix whose DEAD rows
    and columns hold +inf: the minimum of the row's tail by a SIMD reduction
    (a minimum is order-free and exact), then the FIRST column holding a
    value equal to it, which is the lowest j on a tie, exactly what
    `_row_min`'s strict `<` scan returns (a -0.0 and a +0.0 compare equal
    there as here; the value kept is the one AT that column). A tail whose
    minimum is +inf, or that holds a NaN, takes `_row_min` itself: there a
    dead +inf could tie a live one, and a NaN orders differently."""
    var lo = i + 1
    if lo >= n:
        nn[i] = -1
        md[i] = Float32.MAX * Float32(2)
        return
    var p = dm.unsafe_ptr() + i * n
    var dp = dead.unsafe_ptr()
    var inf = Float32.MAX * Float32(2)
    var vmin = SIMD[DType.float32, _RW](inf)
    var nan = False
    var j = lo
    # `dead[j]` is -inf for a live column (max(v, -inf) is v, bit for bit,
    # -0.0 included) and +inf for a dead one (lane/cluster-apple: a mask
    # read contiguously, where writing +inf down the dead column strode by
    # n through the matrix at every merge)
    while j + _RW <= n:
        var v = p.load[width=_RW](j)
        nan = nan or v.ne(v).reduce_or()
        vmin = min(vmin, max(v, dp.load[width=_RW](j)))
        j += _RW
    var m = vmin.reduce_min()
    while j < n:
        var v = p[j]
        if v != v:
            nan = True
        var vm = max(v, dp[j])
        if vm < m:
            m = vm
        j += 1
    if nan or not (m < inf):
        var none = List[Bool]()
        _row_min(i, n, live, none, False, dm, nn, md)
        return
    for q in range(lo, n):
        if live[q] and p[q] == m:
            nn[i] = q
            md[i] = p[q]
            return


def agglo_tree[O: ClusterOps](
    mut ops: O, x: List[Float32], n: Int, d: Int, linkage: Int, metric: Int, p: Float32,
    edges: List[Float32], n_edges: Int, n_merges: Int,
    mut children: List[Int32], mut dist: List[Float32], mut n_components: Int,
) raises:
    """metric -1: euclidean (squared for ward, rooted otherwise, through the
    squared distance); 0-4 the `bodies.pdist_cell` metrics; 5 precomputed
    (`x` is the n x n matrix; its UPPER triangle is read, as scipy's condensed
    form). `edges` holds n_edges (row, col) pairs of the connectivity graph
    (as exact floats), symmetrized here, the diagonal dropped; n_edges < 0
    means no connectivity (every pair). `children` gets n_merges (lower id,
    higher id) pairs, `dist` the merge values."""
    if n < 2:
        raise Error("AgglomerativeClustering: at least two samples are needed")
    if n_merges < 0 or n_merges > n - 1:
        raise Error("AgglomerativeClustering: n_merges must be in [0, n - 1]")
    var inf = Float32.MAX * Float32(2)
    var dm = List[Float32]()
    if metric == 5:
        dm = List[Float32](length=n * n, fill=Float32(0))
        for i in range(n):
            for j in range(i + 1, n):
                var v = x[i * n + j]
                if not (v >= Float32(0)) or v == inf:
                    raise Error("AgglomerativeClustering: a precomputed distance matrix must be finite and non-negative")
                dm[i * n + j] = v
                dm[j * n + i] = v
    else:
        var xs = ops.put(x)
        var ds = ops.zeros(n * n)
        if metric >= 0:
            ops.pdist(xs, n, xs, n, d, metric, p, ds)
        else:
            ops.sqdist(xs, n, xs, n, d, ds)
            if linkage != LINK_WARD:
                ops.sqrt(ds, n * n)
        dm = ops.get(ds, n * n)

    # -------------------------------------------------- the connectivity
    var constrained = n_edges >= 0
    var adj = List[Bool]()
    n_components = 1
    if constrained:
        adj = List[Bool](length=n * n, fill=False)
        for e in range(n_edges):
            var r = Int(edges[2 * e])
            var c = Int(edges[2 * e + 1])
            if r < 0 or r >= n or c < 0 or c >= n:
                raise Error("AgglomerativeClustering: a connectivity edge is outside [0, n)")
            if r != c:
                adj[r * n + c] = True
                adj[c * n + r] = True
        var comp = List[Int]()
        n_components = _components(adj, n, comp)
        if n_components > 1:
            var members = List[List[Int]]()
            for _c in range(n_components):
                members.append(List[Int]())
            for v in range(n):
                members[comp[v]].append(v)
            for ci in range(n_components):
                for cj in range(ci):
                    var bi = -1
                    var bj = -1
                    var bv = inf
                    for a in members[ci]:
                        for b in members[cj]:
                            var v = dm[a * n + b]
                            if linkage == LINK_WARD:
                                v = identical_sqrt(v)
                            if bi < 0 or v < bv:
                                bi = a
                                bj = b
                                bv = v
                    adj[bi * n + bj] = True
                    adj[bj * n + bi] = True

    # -------------------------------------------------- the merge loop
    var live = List[Bool](length=n, fill=True)
    var node = List[Int](capacity=n)
    var size = List[Float64](length=n, fill=Float64(1))
    for i in range(n):
        node.append(i)
    var nn = List[Int](length=n, fill=-1)
    var md = List[Float32](length=n, fill=inf)
    var dead = List[Float32](length=n, fill=-inf)

    for i in range(n):
        _rescan(i, n, live, adj, constrained, dm, dead, nn, md)
    children = List[Int32](capacity=2 * n_merges)
    dist = List[Float32](capacity=n_merges)
    for step in range(n_merges):
        var a = -1
        for i in range(n):
            if live[i] and nn[i] >= 0 and (a < 0 or md[i] < md[a]):
                a = i
        if a < 0:
            raise Error("AgglomerativeClustering: no connected pair is left to merge")
        var b = nn[a]
        var dab = dm[a * n + b]
        var lo = node[a] if node[a] < node[b] else node[b]
        var hi = node[b] if node[a] < node[b] else node[a]
        children.append(Int32(lo))
        children.append(Int32(hi))
        dist.append(identical_sqrt(dab) if linkage == LINK_WARD else dab)
        var na = size[a]
        var nb = size[b]
        for k in range(n):
            if not live[k] or k == a or k == b:
                continue
            var ha = True
            var hb = True
            if constrained:
                ha = adj[a * n + k]
                hb = adj[b * n + k]
            if linkage == LINK_WARD or ha or hb:
                var v = lance_williams(linkage, dm[a * n + k], dm[b * n + k], dab, na, nb, size[k], ha, hb)
                dm[a * n + k] = v
                dm[k * n + a] = v
            if constrained and (ha or hb):
                adj[a * n + k] = True
                adj[k * n + a] = True
        live[b] = False
        nn[b] = -1
        size[a] = na + nb
        node[a] = n + step
        # the dead cluster's column reads +inf in `_row_min_open` through
        # the mask; nothing reads a dead slot's value
        dead[b] = inf
        _rescan(a, n, live, adj, constrained, dm, dead, nn, md)
        for i in range(b):
            if not live[i] or i == a:
                continue
            if nn[i] == a or nn[i] == b:
                _rescan(i, n, live, adj, constrained, dm, dead, nn, md)
            elif i < a and (not constrained or adj[i * n + a]):
                var v = dm[i * n + a]
                if nn[i] < 0 or v < md[i] or (v == md[i] and a < nn[i]):
                    nn[i] = a
                    md[i] = v
