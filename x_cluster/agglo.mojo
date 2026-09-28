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
from std.memory import bitcast
from std.os import getenv
from std.sys.compile import is_defined
from std.time import perf_counter_ns

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, identical_sqrt
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


# FAST ONLY (lane cluster-apple3). OPT-IN while unproven:
# `-D MOJOLEARN_WARD_ROUNDS=1` takes the rounds; the default is the matrix
# loop below.
comptime WARD_ROUNDS = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and is_defined["MOJOLEARN_WARD_ROUNDS"]()
comptime WARD_MAX_ROUNDS = 256


def _ward_rounds[O: ClusterOps](
    mut ops: O, x: List[Float32], n: Int, d: Int, n_merges: Int,
    mut children: List[Int32], mut dist: List[Float32],
) raises -> Bool:
    """The ward tree by ROUNDS OF RECIPROCAL NEAREST NEIGHBOURS on the device.

    Ward's linkage is reducible: merging two clusters that are each other's
    nearest never brings the merged cluster closer to a third than the nearer
    of the two was. So every reciprocal pair of a round is a merge of the
    greedy tree, all of them can merge in the same round, and the merges
    sorted by their value are the greedy loop's sequence (exact ties aside).
    A round is one device kernel over the live clusters' centroids and sizes
    (`bodies.ward_cell`, the closed form of the Lance-Williams recurrence),
    one read of n nearest indices and values, and the centroid updates here
    in Float64; the n x n matrix, its 4 n^2 bytes read to the host and the
    loop's strided column walks are gone.

    The lowest-index cluster of the lowest pairs and its lowest partner are
    always reciprocal, so a round merges at least one pair. A round budget
    bounds the chain-shaped worst case (one merge a round): False returns the
    fit to the matrix loop, nothing written.

    A merge's value is kept at or above its children's (rounding could put
    it a last place below), so the sort by (value, merge sequence) lists
    every child before its parent; node ids are the sorted ranks, as the
    greedy loop's `n + step`."""
    var cen = List[Float64](capacity=n * d)
    for t in range(n * d):
        cen.append(Float64(x[t]))
    var sz = List[Float64](length=n, fill=Float64(1))
    var node = List[Int](capacity=n)
    for i in range(n):
        node.append(i)
    var hgt = List[Float32](length=n, fill=Float32(0))
    var ma = List[Int](capacity=n)
    var mb = List[Int](capacity=n)
    var mh = List[Float32](capacity=n)
    var cs = ops.zeros(n * d)
    var ss = ops.zeros(n)
    var nns = ops.zeros_i(n)
    var mds = ops.zeros(n)
    var nn = List[Int32]()
    var md = List[Float32]()
    var live = n
    var rounds = 0
    while live > 1:
        if rounds >= WARD_MAX_ROUNDS:
            return False
        rounds += 1
        var c32 = List[Float32](capacity=live * d)
        for t in range(live * d):
            c32.append(Float32(cen[t]))
        var s32 = List[Float32](capacity=live)
        for p in range(live):
            s32.append(Float32(sz[p]))
        ops.set(cs, c32)
        ops.set(ss, s32)
        ops.ward_nn(cs, ss, live, d, nns, mds)
        ops.get_if(nns, live, mds, live, nn, md)
        var dead = List[Bool](length=live, fill=False)
        var pairs = 0
        for p in range(live):
            var q = Int(nn[p])
            if q > p and q < live and Int(nn[q]) == p:
                var h = md[p]
                if h < hgt[p]:
                    h = hgt[p]
                if h < hgt[q]:
                    h = hgt[q]
                ma.append(node[p])
                mb.append(node[q])
                node[p] = n + len(mh)
                mh.append(h)
                var sp = sz[p]
                var sq = sz[q]
                var tot = sp + sq
                for f in range(d):
                    cen[p * d + f] = (sp * cen[p * d + f] + sq * cen[q * d + f]) / tot
                sz[p] = tot
                hgt[p] = h
                dead[q] = True
                pairs += 1
        if pairs == 0:
            return False
        var w = 0
        for p in range(live):
            if dead[p]:
                continue
            if w != p:
                for f in range(d):
                    cen[w * d + f] = cen[p * d + f]
                sz[w] = sz[p]
                node[w] = node[p]
                hgt[w] = hgt[p]
            w += 1
        live = w
    # the merges by (value, sequence): a stable radix sort on the value's bits
    var m = len(mh)
    if n_merges > m:
        return False
    var order = List[Int](capacity=m)
    for s in range(m):
        order.append(s)
    var tmp = List[Int](length=m, fill=0)
    for step in range(4):
        var shift = UInt32(8 * step)
        var count = List[Int](length=257, fill=0)
        for s in range(m):
            var b = Int((bitcast[DType.uint32](mh[order[s]]) >> shift) & UInt32(0xFF))
            count[b + 1] += 1
        for b in range(256):
            count[b + 1] += count[b]
        for s in range(m):
            var o = order[s]
            var b = Int((bitcast[DType.uint32](mh[o]) >> shift) & UInt32(0xFF))
            tmp[count[b]] = o
            count[b] += 1
        for s in range(m):
            order[s] = tmp[s]
    var rank = List[Int](length=m, fill=0)
    for r in range(m):
        rank[order[r]] = r
    children = List[Int32](capacity=2 * n_merges)
    dist = List[Float32](capacity=n_merges)
    for r in range(n_merges):
        var s = order[r]
        var a = ma[s] if ma[s] < n else n + rank[ma[s] - n]
        var b = mb[s] if mb[s] < n else n + rank[mb[s] - n]
        children.append(Int32(a if a < b else b))
        children.append(Int32(b if a < b else a))
        dist.append(identical_sqrt(mh[s]))
    if getenv("MOJOLEARN_XC_PHASES") == "1":
        print("XCPHASE agglo.ward_rounds rounds=" + String(rounds) + " merges=" + String(m))
    return True


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
    comptime if WARD_ROUNDS:
        if linkage == LINK_WARD and metric == -1 and n_edges < 0 and ops.fast_device():
            if _ward_rounds(ops, x, n, d, n_merges, children, dist):
                n_components = 1
                return
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
    # MOJOLEARN_XC_PHASES=1 (a diagnostic, lane cluster-apple3): the wall time
    # of the loop's four parts, printed once. Off, four untaken branches a step.
    var ph_on = getenv("MOJOLEARN_XC_PHASES") == "1"
    var ph_t = Int(perf_counter_ns()) if ph_on else 0
    var ph_init = 0
    var ph_argmin = 0
    var ph_lw = 0
    var ph_rescan = 0
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
    if ph_on:
        var now = Int(perf_counter_ns())
        ph_init = now - ph_t
        ph_t = now
    children = List[Int32](capacity=2 * n_merges)
    dist = List[Float32](capacity=n_merges)
    for step in range(n_merges):
        var a = -1
        for i in range(n):
            if live[i] and nn[i] >= 0 and (a < 0 or md[i] < md[a]):
                a = i
        if ph_on:
            var now = Int(perf_counter_ns())
            ph_argmin += now - ph_t
            ph_t = now
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
        if ph_on:
            var now = Int(perf_counter_ns())
            ph_lw += now - ph_t
            ph_t = now
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
        if ph_on:
            var now = Int(perf_counter_ns())
            ph_rescan += now - ph_t
            ph_t = now
    if ph_on:
        print(
            "XCPHASE agglo.loop init_ms=" + String(Float64(ph_init) / 1.0e6) + " argmin_ms="
            + String(Float64(ph_argmin) / 1.0e6) + " lance_williams_ms=" + String(Float64(ph_lw) / 1.0e6)
            + " rescan_ms=" + String(Float64(ph_rescan) / 1.0e6)
        )
