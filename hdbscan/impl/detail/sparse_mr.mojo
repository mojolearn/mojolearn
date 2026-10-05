# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The SPARSE mutual reachability MST's shared pieces: one edge's weight,
the round count, the wide edge sort. HOST SAFE (no device import), so the
device driver (`hdbscan/impl/cluster/detail/sparse_mr_mst.mojo`) and the
CPU column (`hdbscan/host/hdbscan_host_oracle.mojo`) call the SAME code.

======================================================================
DEVIATION BLOCK -- DEVIATION 1620. PAST 46,340 ROWS THE MUTUAL
REACHABILITY MST IS BUILT WITHOUT THE m x m GRAPH.
======================================================================

WHAT THE DENSE ARM DOES (DEVIATION 1600). Every cell of the m x m
mutual reachability matrix is written, then Boruvka runs over it. The
matrix's `nnz = m * m` overflows Int32 at m = 46,341 and is 40 GB at
100,000 rows, so the dense arm refuses there by name.

WHAT THIS ARM DOES. The SAME graph, never stored. Each edge's weight is
computed when it is read, by `mr_edge_weight` below, which is the dense
arm's per-cell arithmetic written once:

    acc  = ftz(fma(ftz(x_a[f]), ftz(x_b[f]), acc))  f ascending
           (`pinned_distance_tile_kernel`'s chain)
    dist = ftz(fma(-2, acc, ftz(ftz(n_a) + ftz(n_b)))), clamped at 0,
           ftz(identical_sqrt(dist))
    mr   = mr_max3(core_a, core_b, mr_scale(1/alpha, dist))
           (`mutual_reachability_dense_kernel`)

`fma(a, b, c)` is `fma(b, a, c)` bit for bit and the norm sum and the
total-order three-way max commute, so the weight of {a, b} is the same
from either end, and it is the dense matrix's cell bit for bit.

WHY THE TREE IS THE SAME TREE. The dense arm's MST is the unique minimum
spanning tree under the TOTAL order (weight_order_key(w), lo, hi)
(DEVIATION 620): with every edge distinct under that order the minimum
spanning tree is unique, so ANY algorithm that minimizes under the same
order returns the same m - 1 edges. The device arm here is Boruvka with
the weights computed on the fly; the CPU arm is Prim. Both break every
tie by (weight key, lo, hi) and nothing else.

THE ROUND COUNT. `n_boruvka_rounds_` is a published output, and it is a
function of the graph: in each Boruvka round every component takes its
cheapest outgoing edge, and under a total order that edge is also the
cheapest TREE edge leaving the component (the cut property). So the
dense solver's rounds are replayed exactly on the m - 1 tree edges by
`boruvka_rounds_on_tree`, plus the final round that finds nothing, which
`mst_solver.mojo` and `hdbh_boruvka` both count.

THE SORT. `pack_edge_key` keeps 16 bits per vertex (DEVIATION 624's
packing, valid only below 46,341 rows). `sort_edges_total_order` sorts on
the full triple with two stable passes, so it is the same order at every
m. Below 65,536 rows its result equals `coo_sort_by_weight`'s.

MEASUREMENT. `hdbscan/checks/sparse_mr_check.mojo` runs both arms on
inputs small enough for the dense one (several seeds, duplicate rows and
an integer grid for ties) and requires the MST edges, the weight bits,
the round count, the labels and the probabilities to be equal; its
sabotage patches break the tie rule and must fail it.
======================================================================
"""

from std.memory import bitcast

from checks.numerics import ftz, identical_mul_add, identical_sqrt
from hdbscan.checks.hdbscan_sabotage import mr_max3, mr_scale
from hierarchy.checks.edge_order import weight_order_key
from hierarchy.impl.sparse.op.sort import merge_sort_u64_with_index_host


@always_inline
def mr_edge_weight(
    acc: Float32,
    norm_a: Float32,
    norm_b: Float32,
    core_a: Float32,
    core_b: Float32,
    inv_alpha: Float32,
    sabotage: Int32,
) -> Float32:
    """The dense arm's cell `(a, b)` from the pinned dot product `acc`:
    `pinned_distance_tile_kernel`'s epilogue at L2SqrtExpanded, then
    `mutual_reachability_dense_kernel`'s transform."""
    var dist = ftz(
        identical_mul_add(Float32(-2.0), acc, ftz(ftz(norm_a) + ftz(norm_b)))
    )
    if dist <= Float32(0.0):
        dist = Float32(0.0)
    dist = ftz(identical_sqrt(dist))
    return mr_max3(core_a, core_b, mr_scale(inv_alpha, dist), sabotage)


@always_inline
def _find(mut parent: List[Int], a: Int) -> Int:
    var r = a
    while parent[r] != r:
        r = parent[r]
    var c = a
    while parent[c] != r:
        var nx = parent[c]
        parent[c] = r
        c = nx
    return r


@always_inline
def triple_less_i(
    wa: Int32, la: Int, ha: Int, wb: Int32, lb: Int, hb: Int
) -> Bool:
    """`edge_order.triple_less` on Int vertex ids (no 32-bit narrowing)."""
    if wa != wb:
        return wa < wb
    if la != lb:
        return la < lb
    return ha < hb


def boruvka_rounds_on_tree(
    lo: List[Int32], hi: List[Int32], w: List[Float32], m: Int
) -> Int:
    """The round count `MST_solver.solve` reports on the dense graph whose
    unique minimum spanning tree (under (weight key, lo, hi)) is these
    `m - 1` edges: the merge rounds, plus the final round that adds
    nothing. See this module's DEVIATION 1620."""
    var n_edges = len(lo)
    var parent = List[Int](capacity=m)
    for v in range(m):
        parent.append(v)
    var wk = List[Int32](capacity=n_edges)
    for e in range(n_edges):
        wk.append(weight_order_key(w[e]))
    var best = List[Int](length=m, fill=-1)
    var n_comp = m
    var rounds = 0
    while True:
        rounds += 1
        if n_comp <= 1:
            break
        for e in range(n_edges):
            var ra = _find(parent, Int(lo[e]))
            var rb = _find(parent, Int(hi[e]))
            if ra == rb:
                continue
            var l = Int(lo[e])
            var h = Int(hi[e])
            for side in range(2):
                var r = ra if side == 0 else rb
                var b = best[r]
                if b < 0 or triple_less_i(
                    wk[e], l, h, wk[b], Int(lo[b]), Int(hi[b])
                ):
                    best[r] = e
        var added = 0
        for r in range(m):
            var b = best[r]
            if b < 0:
                continue
            best[r] = -1
            var ra = _find(parent, Int(lo[b]))
            var rb = _find(parent, Int(hi[b]))
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
                n_comp -= 1
                added += 1
        if added == 0:
            # A forest cannot happen on a spanning tree; stop rather than loop.
            break
    return rounds


def sort_edges_total_order(
    lo: List[Int32], hi: List[Int32], w: List[Float32]
) -> List[Int]:
    """The permutation that sorts the edges by (weight key, lo, hi), every
    vertex id at full width: a stable pass on `hi`, then a stable pass on
    (weight key, lo). `coo_sort_by_weight`'s order (DEVIATION 621) at
    every m."""
    var n = len(lo)
    var k1 = List[UInt64](capacity=n)
    var idx = List[Int](capacity=n)
    for i in range(n):
        k1.append(UInt64(Int(hi[i])) & UInt64(0xFFFFFFFF))
        idx.append(i)
    merge_sort_u64_with_index_host(k1, idx)
    var k2 = List[UInt64](capacity=n)
    for t in range(n):
        var i = idx[t]
        # DEVIATION 624's signed-to-unsigned flip, as `pack_edge_key`.
        var wkey = UInt64(Int(bitcast[DType.uint32](weight_order_key(w[i])) ^ UInt32(0x80000000)))
        k2.append((wkey << UInt64(32)) | (UInt64(Int(lo[i])) & UInt64(0xFFFFFFFF)))
    merge_sort_u64_with_index_host(k2, idx)
    return idx^
