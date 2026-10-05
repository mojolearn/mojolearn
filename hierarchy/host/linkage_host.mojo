# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# SHIPS: compiled into the CPU host binding _mojolearn_solver_host.
"""AgglomerativeClustering's CPU fit, the fast spelling of
`hierarchy/checks/linkage_oracle.mojo` (lane cluster-cpu, 2026-09-28).
HOST ONLY.

THE SAME ANSWER, A DIFFERENT WALK. The oracle's fit is
`host_pinned_distance_matrix` (every cell of the m x m matrix), then
`host_kruskal` (every one of the m(m-1)/2 edges packed and merge-sorted
under the total order `(weight_order_key(d), lo, hi)`, then walked with a
union-find), then `host_dendrogram`. Under a STRICT total order the minimum
spanning tree is unique (the cut property picks one edge per cut), so
Prim's walk under the same order returns the same m - 1 edges; sorted by
the same packed key they come out in Kruskal's acceptance order. This
module:
  * computes each edge's weight as the oracle's cell
    (`host_pinned_distance(x, norms, d, lo, hi)`): the fold of `ftz(fma(x_lo_f,
    x_hi_f, acc))` over f ascending (`fma(a, b, c)` is `fma(b, a, c)` bit
    for bit, so the row either end starts from reads the same), then the
    clamped expanded epilogue and the optional root. Vectors run across
    eight independent edges, never along a fold (`cluster/host/
    host_cells.mojo`);
  * keeps no m x m matrix and no m^2/2 key list: each vertex holds its
    best packed key to the tree, updated from the vertex added last;
  * splits each step's update over vertex chunks (`host_cells`); a chunk's
    argmin is combined in chunk order, and since every key is distinct the
    minimum is the same whatever the split;
  * sorts the m - 1 tree edges by the packed key (the oracle's
    `merge_sort_u64_with_index_host`) and builds the dendrogram with a
    path-compressing union-find (the root a find returns does not depend
    on compression, so every `children` row is the oracle's).
THE NEGATIVE CONTROL is the oracle's: `-D MOJOLEARN_HOST_SABOTAGE=1` selects
the MAXIMUM under the same order (Kruskal walking the keys descending builds
the maximum spanning tree) and emits the edges descending, as the oracle's
sabotaged walk accepts them.

Checked against the oracle, cell for cell, by
`hierarchy/checks/linkage_host_check.mojo` (planted ties, signed zero)."""
from std.sys.compile import is_defined

from checks.numerics import ftz, identical_mul_add, identical_sqrt
from cluster.host.host_cells import ftz_v, host_cells, mul_add_v
from core.host_predict_threads import host_list_ptr
from hierarchy.checks.edge_order import pack_edge_key, unpack_edge_hi, unpack_edge_lo, weight_order_key
from hierarchy.checks.linkage_oracle import host_row_norms_pinned
from hierarchy.impl.sparse.op.sort import merge_sort_u64_with_index_host

comptime LINKAGE_HOST_SABOTAGE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()

#: Vertices per task in one Prim step.
comptime PRIM_CHUNK = 2048
#: Edges per SIMD group.
comptime PRIM_W = 8


@always_inline
def _edge_dist(acc: Float32, n_a: Float32, n_b: Float32, is_sqrt: Bool) -> Float32:
    """`host_pinned_distance`'s epilogue (`ftz(n_a) + ftz(n_b)` commutes)."""
    var dist = ftz(identical_mul_add(Float32(-2.0), acc, ftz(ftz(n_a) + ftz(n_b))))
    if dist <= Float32(0.0):
        dist = Float32(0.0)
    if is_sqrt:
        dist = ftz(identical_sqrt(dist))
    return dist


@always_inline
def _select_key(key: UInt64) -> UInt64:
    """The key Prim minimizes: the packed key, or its complement under the
    sabotage (the maximum spanning tree)."""
    comptime if LINKAGE_HOST_SABOTAGE:
        return ~key
    return key


@fieldwise_init
struct LinkageHostMst(Movable):
    var lo: List[Int32]
    var hi: List[Int32]
    var w: List[Float32]


def host_prim_mst(x: List[Float32], m: Int, d: Int, is_sqrt: Bool) -> LinkageHostMst:
    """`host_kruskal(host_pinned_distance_matrix(x, m, d, is_sqrt), m)`,
    bit for bit, in O(m) memory: the m - 1 edges in the oracle's order."""
    var lo = List[Int32](capacity=m)
    var hi = List[Int32](capacity=m)
    var w = List[Float32](capacity=m)
    if m < 2:
        return LinkageHostMst(lo^, hi^, w^)
    var norms = host_row_norms_pinned(x, m, d)
    # Feature-major, flushed once (`ftz` is idempotent).
    var xt = List[Float32](length=m * d, fill=Float32(0.0))
    for v in range(m):
        for f in range(d):
            xt[f * m + v] = ftz(x[v * d + f])
    var in_tree = List[UInt8](length=m, fill=UInt8(0))
    var best = List[UInt64](length=m, fill=UInt64.MAX)
    var best_w = List[Float32](length=m, fill=Float32(0.0))
    var n_chunks = (m + PRIM_CHUNK - 1) // PRIM_CHUNK
    var chunk_key = List[UInt64](length=n_chunks, fill=UInt64.MAX)
    var chunk_arg = List[Int](length=n_chunks, fill=-1)

    var xtp = host_list_ptr(xt)
    var nrp = host_list_ptr(norms)
    var bwp = host_list_ptr(best_w)
    var itp = in_tree.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var bp = best.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var ckp = chunk_key.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var cap = chunk_arg.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()

    var cur = 0
    in_tree[0] = UInt8(1)
    var e_sel = List[UInt64](capacity=m)
    for _step in range(m - 1):
        var cur_row = List[Float32](length=d if d > 0 else 1, fill=Float32(0.0))
        for f in range(d):
            cur_row[f] = xt[f * m + cur]
        var crp = host_list_ptr(cur_row)
        var n_cur = norms[cur]

        def _chunk(c: Int) {imm xtp, imm nrp, imm bwp, imm itp, imm bp, imm ckp, imm cap, imm crp, imm cur, imm n_cur, imm m, imm d, imm is_sqrt}:
            var v0 = c * PRIM_CHUNK
            var v1 = min(v0 + PRIM_CHUNK, m)
            var v = v0
            while v < v1:
                var width = PRIM_W if v + PRIM_W <= v1 else 1
                if width == PRIM_W:
                    var acc = SIMD[DType.float32, PRIM_W](0)
                    for f in range(d):
                        var q = SIMD[DType.float32, PRIM_W](crp.unsafe_load(f))
                        acc = ftz_v[PRIM_W](mul_add_v[PRIM_W](q, (xtp + f * m + v).load[width=PRIM_W](), acc))
                    comptime for l in range(PRIM_W):
                        var u = v + l
                        if itp[u] == UInt8(0):
                            var dist = _edge_dist(acc[l], n_cur, nrp.unsafe_load(u), is_sqrt)
                            var a = Int32(cur if cur < u else u)
                            var b = Int32(u if cur < u else cur)
                            var key = _select_key(pack_edge_key(weight_order_key(dist), a, b))
                            if key < bp[u]:
                                bp[u] = key
                                bwp.unsafe_store(u, dist)
                else:
                    if itp[v] == UInt8(0):
                        var acc = Float32(0.0)
                        for f in range(d):
                            acc = ftz(identical_mul_add(crp.unsafe_load(f), xtp.unsafe_load(f * m + v), acc))
                        var dist = _edge_dist(acc, n_cur, nrp.unsafe_load(v), is_sqrt)
                        var a = Int32(cur if cur < v else v)
                        var b = Int32(v if cur < v else cur)
                        var key = _select_key(pack_edge_key(weight_order_key(dist), a, b))
                        if key < bp[v]:
                            bp[v] = key
                            bwp.unsafe_store(v, dist)
                v += width
            var kmin = UInt64.MAX
            var amin = -1
            for u in range(v0, v1):
                if itp[u] == UInt8(0) and (amin < 0 or bp[u] < kmin):
                    kmin = bp[u]
                    amin = u
            ckp[c] = kmin
            cap[c] = amin

        host_cells(_chunk, n_chunks, PRIM_CHUNK * (3 * d + 8))
        _ = cur_row^
        var nv = -1
        var nk = UInt64.MAX
        for c in range(n_chunks):
            if chunk_arg[c] >= 0 and (nv < 0 or chunk_key[c] < nk):
                nk = chunk_key[c]
                nv = chunk_arg[c]
        var key = _select_key(nk)
        lo.append(unpack_edge_lo(key))
        hi.append(unpack_edge_hi(key))
        w.append(best_w[nv])
        e_sel.append(nk)
        in_tree[nv] = UInt8(1)
        cur = nv
    # Kruskal's acceptance order: ascending selection key.
    var idx = List[Int](capacity=m - 1)
    for i in range(m - 1):
        idx.append(i)
    merge_sort_u64_with_index_host(e_sel, idx)
    var slo = List[Int32](capacity=m - 1)
    var shi = List[Int32](capacity=m - 1)
    var sw = List[Float32](capacity=m - 1)
    for i in range(m - 1):
        slo.append(lo[idx[i]])
        shi.append(hi[idx[i]])
        sw.append(w[idx[i]])
    _ = xt^
    _ = norms^
    _ = in_tree^
    _ = best^
    _ = best_w^
    _ = chunk_key^
    _ = chunk_arg^
    return LinkageHostMst(slo^, shi^, sw^)


def host_dendrogram_fast(lo: List[Int32], hi: List[Int32], m: Int) -> List[Int32]:
    """`linkage_oracle.host_dendrogram`'s rows with path compression: a
    find returns the same root either way, so every row is the oracle's."""
    var parent = List[Int](length=2 * m - 1 if m > 0 else 1, fill=-1)
    var next_label = m
    var children = List[Int32](capacity=(m - 1) * 2 if m > 1 else 0)
    for i in range(m - 1):
        var r = List[Int](capacity=2)
        r.append(Int(lo[i]))
        r.append(Int(hi[i]))
        var roots = List[Int](capacity=2)
        for s in range(2):
            var n = r[s]
            while parent[n] != -1:
                n = parent[n]
            var c = r[s]
            while parent[c] != -1:
                var nxt = parent[c]
                parent[c] = n
                c = nxt
            roots.append(n)
        children.append(Int32(roots[0]))
        children.append(Int32(roots[1]))
        parent[roots[0]] = next_label
        parent[roots[1]] = next_label
        next_label += 1
    return children^
