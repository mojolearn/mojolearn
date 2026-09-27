# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CAGRA's seams restated as plain host code, apart from
`x_ann/cagra_core.mojo` (DEVIATIONS 5820-5824; the k-NN graph is t-SNE's
cell, DEVIATION 5810):

  5820  pruning keeps the graph_degree edges of smallest (detour count, rank)
  5821  reverse edges: each node's list in (rank, source id) order, capped
  5822  the itopk buffer: the total order (distance, id), bounded
  5823  parents: the search_width best UNEXPANDED entries, scanned front to
        back once per iteration while insertions shift the buffer
  5824  seeds: node (t * n) // n_seeds for t ascending, visited-deduplicated

`high` flags compute the unpinned spelling only to show a fixture separates."""

from checks.numerics import ftz, identical_mul_add


def co_prune(n: Int, kdeg: Int, knn: List[Int], deg: Int, high: Bool = False) -> List[Int]:
    var out = List[Int](length=n * deg, fill=0)
    for a in range(n):
        var cnt = List[Int](length=kdeg, fill=0)
        for kab in range(kdeg):
            var b = knn[a * kdeg + kab]
            for kad in range(kab):
                var dd = knn[a * kdeg + kad]
                # a detour a -> dd -> b exists when b is anywhere in dd's list
                var found = False
                for kdb in range(kdeg):
                    if knn[dd * kdeg + kdb] == b:
                        found = True
                        break
                if found:
                    cnt[kab] += 1
        var taken = List[Bool](length=kdeg, fill=False)
        for i in range(deg):
            var best = -1
            for k in range(kdeg):
                if taken[k]:
                    continue
                if best < 0 or cnt[k] < cnt[best] or (high and cnt[k] == cnt[best]):
                    best = k
            taken[best] = True
            out[a * deg + i] = knn[a * kdeg + best]
    return out^


def co_reverse_merge(n: Int, deg: Int, pruned: List[Int], high: Bool = False) -> List[Int]:
    var rev = List[List[Int]]()
    for _ in range(n):
        rev.append(List[Int]())
    for k in range(deg):
        var srcs = List[Int]()
        for s in range(n):
            srcs.append(n - 1 - s if high else s)
        for t in range(n):
            var src = srcs[t]
            var dst = pruned[src * deg + k]
            if len(rev[dst]) < deg:
                rev[dst].append(src)
    var out = pruned.copy()
    var protected = deg // 2
    if protected == deg:
        return out^
    for nid in range(n):
        var row = List[Int]()
        for i in range(deg):
            row.append(out[nid * deg + i])
        var kr = len(rev[nid])
        while kr > 0:
            kr -= 1
            var v = rev[nid][kr]
            var pos = deg
            for i in range(deg):
                if row[i] == v:
                    pos = i
                    break
            if pos < protected:
                continue
            # remove v (or the last slot) from the unprotected tail, put v first in it
            var drop = pos if pos < deg else deg - 1
            var tail = List[Int]()
            tail.append(v)
            for i in range(protected, deg):
                if i != drop:
                    tail.append(row[i])
            for i in range(protected, deg):
                row[i] = tail[i - protected]
        for i in range(deg):
            out[nid * deg + i] = row[i]
    return out^


def co_dist(q: List[Float32], qo: Int, x: List[Float32], v: Int, d: Int) -> Float32:
    var acc = Float32(0.0)
    for c in range(d):
        var diff = ftz(ftz(q[qo + c]) - ftz(x[v * d + c]))
        acc = ftz(identical_mul_add(diff, diff, acc))
    return acc


def co_before(d: Float32, id: Int, sd: Float32, sid: Int, high: Bool = False) -> Bool:
    if sid < 0:
        return True
    if high:
        return d < sd or (d == sd and id > sid)
    return d < sd or (d == sd and id < sid)


def co_insert(L: Int, mut bd: List[Float32], mut bi: List[Int], mut bx: List[Bool], d: Float32, id: Int):
    """5822: find the sorted position, drop the last slot, shift the flags along."""
    var pos = L
    for s in range(L):
        if co_before(d, id, bd[s], bi[s]):
            pos = s
            break
    if pos == L:
        return
    var s = L - 1
    while s > pos:
        bd[s] = bd[s - 1]
        bi[s] = bi[s - 1]
        bx[s] = bx[s - 1]
        s -= 1
    bd[pos] = d
    bi[pos] = id
    bx[pos] = False


def co_search(x: List[Float32], n: Int, d: Int, graph: List[Int], deg: Int, q: List[Float32], m: Int, k: Int,
              L: Int, width: Int, max_iter: Int, n_seeds: Int, mut out_d: List[Float32], mut out_i: List[Int32],
              seed_offset: Int = 0, parents_from_back: Bool = False):
    out_d = List[Float32](length=m * k, fill=Float32(0.0))
    out_i = List[Int32](length=m * k, fill=Int32(-1))
    for qi in range(m):
        var bd = List[Float32](length=L, fill=Float32(0.0))
        var bi = List[Int](length=L, fill=-1)
        var bx = List[Bool](length=L, fill=False)
        var seen = List[Bool](length=n, fill=False)
        for t in range(n_seeds):
            var v = ((t * n) // n_seeds + seed_offset) % n
            if seen[v]:
                continue
            seen[v] = True
            co_insert(L, bd, bi, bx, co_dist(q, qi * d, x, v, d), v)
        for _ in range(max_iter):
            var taken = 0
            var s = 0
            while s < L and taken < width:
                var slot = L - 1 - s if parents_from_back else s
                if parents_from_back:
                    s += 1
                    if bi[slot] >= 0 and not bx[slot]:
                        bx[slot] = True
                        taken += 1
                        var pb = bi[slot]
                        for e in range(deg):
                            var v = graph[pb * deg + e]
                            if seen[v]:
                                continue
                            seen[v] = True
                            co_insert(L, bd, bi, bx, co_dist(q, qi * d, x, v, d), v)
                    continue
                if bi[s] >= 0 and not bx[s]:
                    bx[s] = True
                    taken += 1
                    var p = bi[s]
                    for e in range(deg):
                        var v = graph[p * deg + e]
                        if seen[v]:
                            continue
                        seen[v] = True
                        co_insert(L, bd, bi, bx, co_dist(q, qi * d, x, v, d), v)
                s += 1
            if taken == 0:
                break
        for s in range(k):
            out_d[qi * k + s] = bd[s]
            out_i[qi * k + s] = Int32(bi[s])
