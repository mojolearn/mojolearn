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


def cagra_prune(n: Int, kdeg: Int, knn: List[Int32], deg: Int) raises -> List[Int32]:
    """`kern_prune`: per node, detour counts over its k-NN list, then the
    `deg` edges of smallest (count, rank) (DEVIATION 5820)."""
    var out = List[Int32](length=n * deg, fill=Int32(0))
    var cnt = List[Int](length=kdeg, fill=0)
    # rank[v] = v's position in row a, valid while stamp[v] == a (lane
    # ann-cpu, 2026-09-28): when row a names kdeg DISTINCT rows in [0, n)
    # (every exact k-NN row does), "the first kab > kad with knn[a, kab] ==
    # cand" is "rank[cand], if it is > kad", so the detour count is the
    # same integer in O(kdeg^2) per node instead of O(kdeg^3). A row with a
    # repeat or an out-of-range entry walks the original search.
    var rank = List[Int32](length=n, fill=Int32(0))
    var stamp = List[Int32](length=n, fill=Int32(-1))
    for a in range(n):
        for k in range(kdeg):
            cnt[k] = kdeg if Int(knn[a * kdeg + k]) == a else 0
        var distinct = True
        for k in range(kdeg):
            var v = Int(knn[a * kdeg + k])
            if v < 0 or v >= n or Int(stamp[v]) == a:
                distinct = False
                break
            stamp[v] = Int32(a)
            rank[v] = Int32(k)
        if distinct:
            for kad in range(kdeg - 1):
                var d = Int(knn[a * kdeg + kad])
                for kdb in range(kdeg):
                    var cand = Int(knn[d * kdeg + kdb])
                    if cand >= 0 and cand < n and Int(stamp[cand]) == a:
                        var kab = Int(rank[cand])
                        if kab > kad:
                            cnt[kab] += 1
        else:
            for k in range(kdeg):
                var v = Int(knn[a * kdeg + k])
                if v >= 0 and v < n:
                    stamp[v] = Int32(-1)
            for kad in range(kdeg - 1):
                var d = Int(knn[a * kdeg + kad])
                for kdb in range(kdeg):
                    var cand = Int(knn[d * kdeg + kdb])
                    for kab in range(kad + 1, kdeg):
                        if Int(knn[a * kdeg + kab]) == cand:
                            cnt[kab] += 1
                            break
        for i in range(deg):
            var best = -1
            for k in range(kdeg):
                if cnt[k] < 0xFFFF and (best < 0 or cnt[k] < cnt[best]):
                    best = k
            if best < 0:
                raise Error("CAGRA: the k-NN graph has too few distinct neighbors for graph_degree")
            var sel = knn[a * kdeg + best]
            for k in range(kdeg):
                if knn[a * kdeg + k] == sel:
                    cnt[k] = 0xFFFF
            out[a * deg + i] = sel
    return out^


def cagra_reverse_merge(n: Int, deg: Int, pruned: List[Int32]) -> List[Int32]:
    """`kern_make_rev_graph` + the merge, with the reverse list of each node
    in (rank, source id) order and capped at `deg`, as their count is
    (DEVIATION 5821)."""
    var rev = List[Int32](length=n * deg, fill=Int32(-1))
    var rcount = List[Int](length=n, fill=0)
    for k in range(deg):
        for src in range(n):
            var dst = Int(pruned[src * deg + k])
            if rcount[dst] < deg:
                rev[dst * deg + rcount[dst]] = Int32(src)
            rcount[dst] += 1
    var out = pruned.copy()
    var protected = deg // 2
    var row = List[Int32](length=deg, fill=Int32(0))
    for nid in range(n):
        if protected == deg:
            break
        for i in range(deg):
            row[i] = out[nid * deg + i]
        var kr = rcount[nid] if rcount[nid] < deg else deg
        while kr > 0:
            kr -= 1
            var v = rev[nid * deg + kr]
            if v < 0:
                continue
            var pos = deg
            for i in range(deg):
                if row[i] == v:
                    pos = i
                    break
            if pos < protected:
                continue
            var num_shift = pos - protected
            if pos >= deg:
                num_shift = deg - protected - 1
            var s = protected + num_shift
            while s > protected:
                row[s] = row[s - 1]
                s -= 1
            row[protected] = v
        for i in range(deg):
            out[nid * deg + i] = row[i]
    return out^


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
    visited: I32P, words: Int, out_d: F32P, out_i: I32P,
):
    cg_search_row(qi, queries, x, n, d, graph, deg, k, L, width, max_iter, n_seeds, bd, bi, bx, qi * L,
                  visited, qi * words, words, out_d, out_i)


@always_inline
def cg_search_row(
    qi: Int, queries: F32P, x: F32P, n: Int, d: Int, graph: I32P, deg: Int, k: Int,
    L: Int, width: Int, max_iter: Int, n_seeds: Int, bd: F32P, bi: I32P, bx: I32P, base: Int,
    visited: I32P, vbase: Int, words: Int, out_d: F32P, out_i: I32P,
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
        var v = (t * n) // n_seeds
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
