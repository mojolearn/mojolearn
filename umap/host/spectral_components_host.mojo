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
  and this column both reach it by min-label propagation over the same
  bodies, core/cc_label_bodies.mojo; the fixed point is unique, so the
  labels agree);
- each component's vertices in ascending index and its edges in input order,
  relabelled to local indices (the device's stable radix sort);
- the centroids, `data_range`s, uniform draws, the bounded eigen solve
  (`oracle_embedding_capped`, the device's `SC_MAX_RESTARTS`), the
  acceptance rule, the scaling and the post-pass.

SPLIT ACROSS THE HOST CORES: the components (`_host_components`: hook rounds
over edge chunks, jumps and the root numbering over node chunks; lane
sc-leftovers-1, 2026-11-28, replacing a serial union-find), the centroid
ranges, the uniform draws and the per-component eigen solves
(`host_parallelize`).

SERIAL, AND WHY: the stable grouping of vertices and edges by component (the
counting passes below) is one O(n + nnz) sweep whose order is the contract.
Splitting it needs a count table per task of n_comp words (n_comp reaches n
when most vertices are alone), the shape core/sparse_coo_csr_impl.mojo caps by
memory; that split is OWED, not refused.
"""

from spectral.host.spectral_oracle import oracle_embedding_capped
from spectral.impl.sparse.coo import CooGraph
from umap.host.spectral_post_pass_host import host_spectral_post_pass
from core.host_parallel import host_parallelize
from umap.spectral_components_math import (
    SC_MAX_RESTARTS,
    SC_SABOTAGE_SOLVE,
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
from std.sys.info import num_physical_cores
from core.cc_label_bodies import CcLabelPtr, cc_hook_edge, cc_jump_node


#: The fewest items (edges or vertices) a host task takes in `_host_components`: below it a split costs more
#: than it saves (the grain of core/sparse_coo_csr_impl.mojo's tasks, a size rule, not a board shape).
comptime _SC_HOST_MIN_PER_TASK = 16384


def _sc_host_tasks(count: Int) -> Int:
    """Tasks for `count` items: one per physical core, at most one per
    `_SC_HOST_MIN_PER_TASK` items, at least one."""
    var tasks = (count + _SC_HOST_MIN_PER_TASK - 1) // _SC_HOST_MIN_PER_TASK
    var cores = num_physical_cores()
    if tasks > cores:
        tasks = cores
    return tasks if tasks > 1 else 1


def _host_components(coo: CooGraph, mut comp: List[Int32]) -> Int:
    """The connected components on the CPU column, split across the host
    cores (lane sc-leftovers-1, 2026-11-28; before, a serial union-find): the
    device's min-label rounds over the same bodies (core/cc_label_bodies.mojo;
    umap/spectral_init.mojo `sc_label_hook_kernel` / `sc_label_jump_kernel`),
    a hook over edge chunks then a jump over node chunks per round until a
    round lowers nothing, then the roots numbered in index order by a blocked
    count and scan. The fixed point is unique (every vertex labelled by its
    component's smallest index, whatever the task order), so `comp` is the
    union-find's and the device's words: component ids in order of their
    smallest vertex. Returns the component count."""
    var n = coo.n
    var nnz = coo.nnz()
    var lab = List[Int32](length=max(n, 1), fill=Int32(0))
    var lp = CcLabelPtr(unsafe_from_address=Int(lab.unsafe_ptr()))
    var rp = CcLabelPtr(unsafe_from_address=Int(coo.rows.unsafe_ptr()))
    var cq = CcLabelPtr(unsafe_from_address=Int(coo.cols.unsafe_ptr()))
    var nt = _sc_host_tasks(n)
    var nchunk = (n + nt - 1) // nt
    def _init(t: Int) {imm lp, imm nchunk, imm n}:
        var hi = min(t * nchunk + nchunk, n)
        for v in range(t * nchunk, hi):
            lp.unsafe_store(v, Int32(v))
    host_parallelize(_init, nt)
    var et = _sc_host_tasks(nnz)
    var echunk = (nnz + et - 1) // et if nnz > 0 else 1
    var lowered = List[Int32](length=et, fill=Int32(0))
    var wp = CcLabelPtr(unsafe_from_address=Int(lowered.unsafe_ptr()))
    def _hook(t: Int) {imm lp, imm rp, imm cq, imm wp, imm echunk, imm nnz}:
        var hi = min(t * echunk + echunk, nnz)
        var moved = Int32(0)
        for e in range(t * echunk, hi):
            if cc_hook_edge(lp, Int(rp.unsafe_load(e)), Int(cq.unsafe_load(e))):
                moved = Int32(1)
        wp.unsafe_store(t, moved)
    def _jump(t: Int) {imm lp, imm nchunk, imm n}:
        var hi = min(t * nchunk + nchunk, n)
        for v in range(t * nchunk, hi):
            cc_jump_node(lp, v)
    for _ in range(n + 1):  # driver-loop: over label-propagation rounds (at most n, each changing round lowers a label)
        if nnz == 0:
            break
        host_parallelize(_hook, et)
        host_parallelize(_jump, nt)
        var changed = False
        for t in range(et):  # small-loop(the hook tasks' change words, at most the core count)
            if lowered[t] != Int32(0):
                changed = True
        if not changed:
            break
    # the roots (lab[v] == v) numbered in index order: per-task counts, their
    # prefix over the tasks, each task's ranks, then every vertex its root's id
    var counts = List[Int](length=nt + 1, fill=0)
    var kp = counts.unsafe_ptr()
    def _count(t: Int) {imm lp, imm kp, imm nchunk, imm n}:
        var hi = min(t * nchunk + nchunk, n)
        var c = 0
        for v in range(t * nchunk, hi):
            if Int(lp.unsafe_load(v)) == v:
                c += 1
        kp.unsafe_store(t + 1, c)
    host_parallelize(_count, nt)
    for t in range(nt):  # small-loop(the node tasks, at most the core count)
        counts[t + 1] += counts[t]
    var n_comp = counts[nt]
    comp = List[Int32](length=max(n, 1), fill=Int32(0))
    var op = CcLabelPtr(unsafe_from_address=Int(comp.unsafe_ptr()))
    def _rank(t: Int) {imm lp, imm kp, imm op, imm nchunk, imm n}:
        var hi = min(t * nchunk + nchunk, n)
        var r = kp[t]
        for v in range(t * nchunk, hi):
            if Int(lp.unsafe_load(v)) == v:
                op.unsafe_store(v, Int32(r))
                r += 1
    host_parallelize(_rank, nt)
    def _label(t: Int) {imm lp, imm op, imm nchunk, imm n}:
        var hi = min(t * nchunk + nchunk, n)
        for v in range(t * nchunk, hi):
            var root = Int(lp.unsafe_load(v))
            if root != v:
                op.unsafe_store(v, op.unsafe_load(root))
    host_parallelize(_label, nt)
    _ = len(lab)  # the labels, change words and counts outlive every task above
    _ = len(lowered)
    _ = len(counts)
    _ = len(coo.rows)
    return n_comp

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
        comptime if SC_SABOTAGE_SOLVE:
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
    # the components (split across the host cores: `_host_components`)
    var comp = List[Int32]()
    var n_comp = _host_components(coo, comp)
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
