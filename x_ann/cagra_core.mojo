# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CAGRA: THE PER-CELL ARITHMETIC AND THE GRAPH OPTIMIZATION, ONE SOURCE FOR
THE GPU AND THE CPU (lane/algos-ann, pass 1, 2026-09-27).

Reference: cuVS `cpp/src/neighbors/detail/cagra/cagra_build.cuh` (build),
`graph_core.cuh` (`kern_prune` :220-330, `kern_make_rev_graph` :178-200,
the reverse-edge merge :430-470) and `search_single_cta_kernel.cuh` (the
greedy itopk search), pinned at ~/CascadeProjects/upstream/cuvs-v26.08.00.

THE FIXED-ORDER DESIGN vs cuVS's NONDETERMINISM
  * the intermediate k-NN graph: cuVS builds it with IVF-PQ + refine or
    NN-descent (both approximate, NN-descent refused by name). Here: EXACT
    k-NN, a per-row sorted insertion under (distance, index)
    (`tsne_core.ts_knn_cell`, the same cell).
  * pruning (`kern_prune`): detour counts are integer atomicAdds, order-free
    already; the selection keeps the `graph_degree` smallest under (count,
    rank), which is their packed (count << 16 | rank) min. Restated as
    sequential host code, the same in both drivers.
  * reverse edges (`kern_make_rev_graph`): cuVS inserts with atomicAdd, so a
    node's reverse list is in thread-arrival order. Here the order is FIXED:
    rank ascending (their one launch per rank) then source id ascending.
  * the merge keeps their shape: graph_degree / 2 protected forward edges,
    reverse edges inserted in front of the rest.
  * search: cuVS's single-CTA search uses random seeds (a hash of the query
    and a seed) and a bitonic/radix itopk with feed-order ties, and a
    probabilistic hashmap. Here one query per thread: fixed evenly spaced
    seed nodes, an exact visited bitset, a sorted itopk buffer under
    (distance, id), parents taken as the `search_width` best unexpanded
    entries, a fixed maximum of iterations.
"""

from checks.numerics import ftz, identical_mul_add


comptime F32P = MutPointer[Float32, MutAnyOrigin]
comptime I32P = MutPointer[Int32, MutAnyOrigin]


@always_inline
def cg_seed_node(t: Int, n: Int, n_seeds: Int, rs: Int) -> Int:
    """Seed node `t` of every query's walk. `rs == 0` (random_state=None):
    the evenly spaced `(t * n) // n_seeds` (DEVIATION 5824). `rs > 0`
    (random_state given, gap-fails2 2026-10-02): splitmix64 of
    `(rs, t)` reduced mod n, the role cuVS's `rand_xor_mask` plays in its
    seed hash. Integer arithmetic only, so the same node on every device and
    on the host; a repeated node is skipped by the visited set, as any
    repeat is."""
    if rs == 0:
        return (t * n) // n_seeds
    var z = UInt64(rs) * UInt64(0x9E3779B97F4A7C15) + UInt64(t + 1) * UInt64(0xD1B54A32D192ED03)
    z = (z ^ (z >> UInt64(30))) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> UInt64(27))) * UInt64(0x94D049BB133111EB)
    z = z ^ (z >> UInt64(31))
    return Int(z % UInt64(n))


@always_inline
def cg_dist(q: F32P, q_off: Int, x: F32P, v: Int, d: Int) -> Float32:
    var acc = Float32(0.0)
    for c in range(d):
        var diff = ftz(ftz(q.unsafe_load(q_off + c)) - ftz(x.unsafe_load(v * d + c)))
        acc = ftz(identical_mul_add(diff, diff, acc))
    return acc


@always_inline
def cg_insert(L: Int, base: Int, d: Float32, id: Int32, bd: F32P, bi: I32P, bx: I32P):
    """Sorted insertion under (distance, id) into a bounded buffer; the
    expanded flag travels with its entry; an empty slot (id < 0) is worst
    (DEVIATION 5822)."""
    var li = bi.unsafe_load(base + L - 1)
    if li >= 0:
        var ld = bd.unsafe_load(base + L - 1)
        if not (d < ld or (d == ld and id < li)):
            return
    var s = L - 1
    while s > 0:
        var pi = bi.unsafe_load(base + s - 1)
        var pd = bd.unsafe_load(base + s - 1)
        if pi < 0 or d < pd or (d == pd and id < pi):
            bd.unsafe_store(base + s, pd)
            bi.unsafe_store(base + s, pi)
            bx.unsafe_store(base + s, bx.unsafe_load(base + s - 1))
            s -= 1
        else:
            break
    bd.unsafe_store(base + s, d)
    bi.unsafe_store(base + s, id)
    bx.unsafe_store(base + s, Int32(0))


@always_inline
def cg_search_cell(
    qi: Int, queries: F32P, x: F32P, n: Int, d: Int, graph: I32P, deg: Int, k: Int,
    L: Int, width: Int, max_iter: Int, n_seeds: Int, bd: F32P, bi: I32P, bx: I32P,
    visited: I32P, words: Int, out_d: F32P, out_i: I32P, rs: Int = 0,
):
    cg_search_row(qi, queries, x, n, d, graph, deg, k, L, width, max_iter, n_seeds, bd, bi, bx, qi * L,
                  visited, qi * words, words, out_d, out_i, rs)


@always_inline
def cg_search_row(
    qi: Int, queries: F32P, x: F32P, n: Int, d: Int, graph: I32P, deg: Int, k: Int,
    L: Int, width: Int, max_iter: Int, n_seeds: Int, bd: F32P, bi: I32P, bx: I32P, base: Int,
    visited: I32P, vbase: Int, words: Int, out_d: F32P, out_i: I32P, rs: Int = 0,
):
    """`cg_search_cell` with the itopk buffer at `base` and the visited
    bitset at `vbase` (the device gives each query its own; a host task
    reuses one)."""
    # DEVIATION 5824: seeds (t * n) // n_seeds; DEVIATION 5823: parents are the
    # search_width best unexpanded entries, scanned front to back.
    var q_off = qi * d
    for s in range(L):
        bd.unsafe_store(base + s, Float32(0.0))
        bi.unsafe_store(base + s, Int32(-1))
        bx.unsafe_store(base + s, Int32(0))
    for w in range(words):
        visited.unsafe_store(vbase + w, Int32(0))
    for t in range(n_seeds):
        var v = cg_seed_node(t, n, n_seeds, rs)
        var word = visited.unsafe_load(vbase + v // 32)
        var bit = Int32(1) << Int32(v % 32)
        if (word & bit) != 0:
            continue
        visited.unsafe_store(vbase + v // 32, word | bit)
        cg_insert(L, base, cg_dist(queries, q_off, x, v, d), Int32(v), bd, bi, bx)
    for _ in range(max_iter):
        var taken = 0
        for s in range(L):
            if taken == width:
                break
            var p = Int(bi.unsafe_load(base + s))
            if p < 0 or bx.unsafe_load(base + s) != 0:
                continue
            bx.unsafe_store(base + s, Int32(1))
            taken += 1
            # expanding p may shift entries at or after s; the flag travels
            for e in range(deg):
                var v = Int(graph.unsafe_load(p * deg + e))
                var word = visited.unsafe_load(vbase + v // 32)
                var bit = Int32(1) << Int32(v % 32)
                if (word & bit) != 0:
                    continue
                visited.unsafe_store(vbase + v // 32, word | bit)
                cg_insert(L, base, cg_dist(queries, q_off, x, v, d), Int32(v), bd, bi, bx)
        if taken == 0:
            break
    for s in range(k):
        out_d.unsafe_store(qi * k + s, bd.unsafe_load(base + s))
        out_i.unsafe_store(qi * k + s, bi.unsafe_load(base + s))
