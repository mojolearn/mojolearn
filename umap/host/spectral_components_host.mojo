# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# SHIPS: compiled into the CPU host binding through umap/host/umap_oracle.mojo; product, not only a check.
"""HOST COLUMN of UMAP's spectral initialization with connected components
(lane sc-ann-1-spectral, 2026-11-27).

The device form is `umap/spectral_init.mojo::
spectral_initialize_device_coo_report`; both call the bodies in
`umap/spectral_components_math.mojo`, so the words are the same:

- the components: the same partition, every vertex labelled by its
  component's smallest index, components numbered in that order (the device
  reaches it by min-label propagation, this column by union-find with the
  smaller root kept; the fixed point is unique, so the labels agree);
- each component's vertices in ascending index and its edges in input order,
  relabelled to local indices (the device's stable radix sort);
- the centroids, `data_range`s, uniform draws, the bounded eigen solve
  (`oracle_embedding_capped`, the device's `SC_MAX_RESTARTS`), the
  acceptance rule, the scaling and the post-pass.

SERIAL PARTS, AND WHY: the union-find and the counting passes over vertices
and edges are O(n + nnz) single sweeps whose order is the contract (the
component numbering, the stable grouping); they are not split. The centroid
ranges, the uniform draws and the per-component eigen solves are independent
units and run on the host cores (`host_parallelize`).
"""

from spectral.host.spectral_oracle import oracle_embedding_capped
from spectral.impl.sparse.coo import CooGraph
from umap.host.spectral_post_pass_host import host_spectral_post_pass
from core.host_parallel import host_parallelize
from umap.spectral_components_math import (
    SC_MAX_RESTARTS,
    SC_STREAM_LAYOUT,
    SpectralInitReport,
    sc_accept,
    sc_component_solved,
    sc_data_range,
    sc_expansion,
    sc_fallback_message,
    sc_hash,
    sc_mag_bits,
    sc_meta_coord,
    sc_nonfinite,
    sc_random_cell,
    sc_seed_hi,
    sc_seed_lo,
    sc_spectral_cell,
    sc_sq_step,
)
from std.memory import bitcast


def _find(mut parent: List[Int], x: Int) -> Int:
    """Union-find root with path halving."""
    var v = x
    while parent[v] != v:
        parent[v] = parent[parent[v]]
        v = parent[v]
    return v


def _host_solve(
    g: CooGraph, dim: Int, seed: UInt64, mut emb: List[Float32], mut restarts: Int,
) -> Bool:
    """`_sc_solve` on the host: the bounded eigen solve of one graph into
    `emb` (`n x dim`); False when it raised, came back the wrong shape or
    was not accepted (`sc_accept`)."""
    try:
        var res = oracle_embedding_capped[DType.float32](
            g, dim + 1, True, True, Float32(1e-5), seed, SC_MAX_RESTARTS
        )
        restarts = res.restarts
        var resid = Float32(0.0)
        if len(res.restart_res) > 0:
            resid = res.restart_res[len(res.restart_res) - 1]
        if res.n_out != dim or len(res.embedding) != g.n * dim:
            return False
        if not sc_accept(res.converged, resid):
            return False
        emb = List[Float32](capacity=g.n * dim)
        for i in range(len(res.embedding)):  # small-loop(one component's embedding words, a copy)
            emb.append(res.embedding[i])
        return True
    except:
        return False


def host_spectral_initialize_report(
    coo: CooGraph, n_components: Int, seed: UInt64
) raises -> SpectralInitReport:
    """The device report's host column over the positive row-major COO
    (values already refused by the caller when non-finite or negative)."""
    var n = coo.n
    var dim = n_components
    var nnz = coo.nnz()
    if n_components < 1:
        raise Error("UMAP spectral initialization needs n_components >= 1")
    if n < 2 * n_components + 4:
        raise Error("UMAP spectral initialization has too few samples")
    var seed_lo = sc_seed_lo(seed)
    var seed_hi = sc_seed_hi(seed)
    # the components
    var parent = List[Int](capacity=n)
    for i in range(n):
        parent.append(i)
    for e in range(nnz):
        var a = _find(parent, Int(coo.rows[e]))
        var b = _find(parent, Int(coo.cols[e]))
        if a < b:
            parent[b] = a
        elif b < a:
            parent[a] = b
    var comp = List[Int32](length=n, fill=Int32(0))
    var n_comp = 0
    for i in range(n):
        var r = _find(parent, i)
        if r == i:
            comp[i] = Int32(n_comp)
            n_comp += 1
        else:
            comp[i] = comp[r]
    if n_comp <= 1:
        var emb1 = List[Float32]()
        var restarts1 = 0
        if _host_solve(coo, dim, seed, emb1, restarts1):
            try:
                host_spectral_post_pass(emb1, n, dim)
                return SpectralInitReport(emb1^, 1, 1, 0, 0, restarts1, String(""))
            except:
                pass
        var rnd = List[Float32](length=n * dim, fill=Float32(0.0))
        var rp = rnd.unsafe_ptr()
        def _single(v: Int) {imm rp, imm dim, imm seed_lo, imm seed_hi}:
            for d in range(dim):  # small-loop(dim: output coordinates)
                var h = sc_hash(seed_lo, seed_hi, SC_STREAM_LAYOUT, UInt32(v), UInt32(d))
                rp.unsafe_store(v * dim + d, sc_random_cell(h, Float32(10.0), Float32(0.0)))
        host_parallelize(_single, n)
        _ = len(rnd)
        host_spectral_post_pass(rnd, n, dim)
        return SpectralInitReport(rnd^, 1, 0, 0, 1, restarts1, sc_fallback_message(1, 1))
    # vertices and edges grouped by component (stable), local indices
    var vstart = List[Int](length=n_comp + 1, fill=0)
    var estart = List[Int](length=n_comp + 1, fill=0)
    for i in range(n):
        vstart[Int(comp[i]) + 1] += 1
    for e in range(nnz):
        estart[Int(comp[Int(coo.rows[e])]) + 1] += 1
    for c in range(n_comp):
        vstart[c + 1] += vstart[c]
        estart[c + 1] += estart[c]
    var cursor = List[Int](length=n_comp, fill=0)
    var local = List[Int32](length=n, fill=Int32(0))
    var perm = List[Int](length=n, fill=0)
    for i in range(n):
        var c = Int(comp[i])
        local[i] = Int32(cursor[c])
        perm[vstart[c] + cursor[c]] = i
        cursor[c] += 1
    var ne = nnz if nnz > 0 else 1
    var srows = List[Int32](length=ne, fill=Int32(0))
    var scols = List[Int32](length=ne, fill=Int32(0))
    var svals = List[Float32](length=ne, fill=Float32(0.0))
    for c in range(n_comp):
        cursor[c] = 0
    for e in range(nnz):
        var r = Int(coo.rows[e])
        var c = Int(comp[r])
        var q = estart[c] + cursor[c]
        cursor[c] += 1
        srows[q] = local[r]
        scols[q] = local[Int(coo.cols[e])]
        svals[q] = coo.vals[e]
    # centroids, ranges, every cell's uniform draw
    var meta = List[Float32](length=n_comp * dim, fill=Float32(0.0))
    for i in range(n_comp * dim):
        meta[i] = sc_meta_coord(i // dim, i % dim, n_comp, dim, seed_lo, seed_hi)
    var drange = List[Float32](length=n_comp, fill=Float32(0.0))
    var mp = meta.unsafe_ptr()
    var dp = drange.unsafe_ptr()
    def _range(c: Int) {imm mp, imm dp, imm n_comp, imm dim}:
        var best = Float32(-1.0)
        for j in range(n_comp):  # small-loop(n_comp: the other centroids, the device thread's order)
            var acc = Float32(0.0)
            for d in range(dim):  # small-loop(dim: output coordinates)
                acc = sc_sq_step(acc, mp[c * dim + d], mp[j * dim + d])
            if acc > Float32(0.0) and (best < Float32(0.0) or acc < best):
                best = acc
        dp.unsafe_store(c, sc_data_range(best))
    host_parallelize(_range, n_comp)
    var out = List[Float32](length=n * dim, fill=Float32(0.0))
    var op = out.unsafe_ptr()
    var cp = comp.unsafe_ptr()
    def _draw(v: Int) {imm op, imm cp, imm mp, imm dp, imm dim, imm seed_lo, imm seed_hi}:
        var c = Int(cp[v])
        for d in range(dim):  # small-loop(dim: output coordinates)
            var h = sc_hash(seed_lo, seed_hi, SC_STREAM_LAYOUT, UInt32(v), UInt32(d))
            op.unsafe_store(v * dim + d, sc_random_cell(h, dp[c], mp[c * dim + d]))
    host_parallelize(_draw, n)
    # the solved components, one task each
    var solved = List[Int]()
    var n_small = 0
    for c in range(n_comp):
        if sc_component_solved(vstart[c + 1] - vstart[c], dim):
            solved.append(c)
        else:
            n_small += 1
    var n_tasks = len(solved)
    var outcome = List[Int](length=n_tasks if n_tasks > 0 else 1, fill=0)
    var taken = List[Int](length=n_tasks if n_tasks > 0 else 1, fill=0)
    var solved_at = Int(solved.unsafe_ptr())
    var outcome_at = Int(outcome.unsafe_ptr())
    var taken_at = Int(taken.unsafe_ptr())
    var vstart_at = Int(vstart.unsafe_ptr())
    var estart_at = Int(estart.unsafe_ptr())
    var perm_at = Int(perm.unsafe_ptr())
    var sr = srows.unsafe_ptr()
    var sc = scols.unsafe_ptr()
    var sv = svals.unsafe_ptr()
    def _component(t: Int) {imm solved_at, imm outcome_at, imm taken_at, imm vstart_at, imm estart_at,
                             imm perm_at, imm sr, imm sc, imm sv, imm op, imm mp, imm dp, imm dim, imm seed}:
        var c = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=solved_at)[t]
        var vs = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=vstart_at)
        var es = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=estart_at)
        var pp = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=perm_at)
        var size = vs[c + 1] - vs[c]
        var e0 = es[c]
        var nnz_c = es[c + 1] - e0
        var rows_c = List[Int32](capacity=nnz_c)
        var cols_c = List[Int32](capacity=nnz_c)
        var vals_c = List[Float32](capacity=nnz_c)
        for q in range(nnz_c):  # small-loop(one component's edges, a copy)
            rows_c.append(sr[e0 + q])
            cols_c.append(sc[e0 + q])
            vals_c.append(sv[e0 + q])
        var g = CooGraph(size, rows_c^, cols_c^, vals_c^)
        var emb = List[Float32]()
        var restarts_c = 0
        var ok = nnz_c > 0 and _host_solve(g, dim, seed, emb, restarts_c)
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=taken_at)[t] = restarts_c
        var peak = Int32(0)
        if ok:
            for i in range(size * dim):  # small-loop(one component's embedding words)
                if sc_nonfinite(emb[i]):
                    ok = False
                    break
                var m = sc_mag_bits(emb[i])
                if m > peak:
                    peak = m
            if peak == Int32(0):
                ok = False
        if ok:
            var expansion = sc_expansion(dp[c], bitcast[DType.float32](peak))
            for p in range(size):  # small-loop(one component's rows)
                var v = pp[vs[c] + p]
                for d in range(dim):  # small-loop(dim: output coordinates)
                    op.unsafe_store(v * dim + d, sc_spectral_cell(emb[p * dim + d], expansion, mp[c * dim + d]))
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=outcome_at)[t] = 1
        else:
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=outcome_at)[t] = 2
    host_parallelize(_component, n_tasks)
    _ = len(solved)
    _ = len(vstart)
    _ = len(estart)
    _ = len(perm)
    _ = len(srows)
    _ = len(scols)
    _ = len(svals)
    _ = len(meta)
    _ = len(drange)
    _ = len(comp)
    var n_solved = 0
    var n_fallback = 0
    var max_restarts = 0
    for t in range(n_tasks):  # small-loop(one outcome per solved component)
        if outcome[t] == 1:
            n_solved += 1
        else:
            n_fallback += 1
        if taken[t] > max_restarts:
            max_restarts = taken[t]
    host_spectral_post_pass(out, n, dim)
    var msg = String("")
    if n_fallback > 0:
        msg = sc_fallback_message(n_fallback, n_comp)
    return SpectralInitReport(out^, n_comp, n_solved, n_small, n_fallback, max_restarts, msg^)
