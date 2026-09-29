# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The mutual reachability MST without the m x m graph, on the device.

DEVIATION 1620 (`hdbscan/impl/detail/sparse_mr.mojo` has the block): past
`PAIRWISE_MAX_ROWS` the dense graph cannot exist, so Boruvka runs with
every edge weight computed when it is read. The answer is the dense arm's
tree edge for edge and bit for bit, because both minimize under the one
total order (weight key, lo, hi) and every weight is the dense cell's
arithmetic (`mr_edge_weight`).

THE ROUND. For each point `i` of a listed set, `sparse_mr_search_kernel`
finds the cheapest edge to a point of ANOTHER component: one thread per
(point, slice of the j axis), j ascending, the minimum taken on the
INTEGER weight key with a STRICT `<` so a tie keeps the lower j. For a
fixed `i` the order (key, lo, hi) among equal keys IS ascending j (j < i
reads (j, i), j > i reads (i, j), and every j < i sorts first), so the
per-point minimum is the per-point minimum of the total order. The host
folds the slices in ascending slice order under the same (key, j)
comparison, takes each component's minimum under the full triple, joins
components with a union-find and relabels. No float reduction, no atomic:
a minimum over a total order has one answer whatever the split.

THE PRUNING (the plan of `hierarchy/.../fast_boruvka.mojo`, exact here).
A point whose previous best edge still leaves its component keeps that
edge: the candidate set only shrinks and still holds it. A point whose
best joined its own component is LISTED, with its previous key as a lower
bound. A listed point whose lower bound is STRICTLY above its component's
best known key cannot supply the component's edge and sits the round out
(a tie is still searched). Phase A searches each component's listed point
with the smallest bound, phase B the rest after A's exact keys tighten the
bound. None of this changes which edge a component takes.

APPLE LAUNCH BOUND. macOS silently aborts a Metal command buffer that
holds the GPU for seconds (the output stays partly stale and
`synchronize` reports nothing). Every search launch is bounded to
`SPARSE_MR_LAUNCH_MACS` multiply-adds, its output is POISONED before the
launch and read back WHOLE after it; a poisoned cell that survives is
refused by name. The same bound runs on every column so the arithmetic
and the launches are one program.

NON-FINITE WEIGHTS. Round 1 searches every point against every other, so
every edge weight is computed at least once; a NaN or infinite weight is
reported by its point and refused by name (DEVIATION 1607's rule).
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.memory import bitcast
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.numerics import ftz, identical_mul_add
from core.row_norms import NORM_TPB, row_norm_kernel
from hdbscan.checks.hdbscan_sabotage import HDB_SAB_NONE
from hdbscan.impl.detail.sparse_mr import (
    boruvka_rounds_on_tree,
    mr_edge_weight,
    sort_edges_total_order,
    triple_less_i,
)
from hierarchy.checks.edge_order import (
    WEIGHT_KEY_SENTINEL,
    weight_order_key,
    weight_order_unkey,
)


comptime SPARSE_MR_TPB = 128
"""SCHEDULING: one thread per (point, slice); nothing folds across threads."""

comptime SPARSE_MR_LAUNCH_MACS = 1 << 30
"""The most multiply-adds one search launch may issue (the Apple bound)."""

comptime SPARSE_MR_TARGET_THREADS = 1 << 16
"""Slices per launch are chosen so a launch has about this many threads."""

comptime SMR_NONE_J: Int32 = -1
comptime SMR_BAD_J: Int32 = -2
comptime SMR_POISON_J: Int32 = -3
comptime SMR_KEY_MIN: Int32 = -0x7FFFFFFF - 1


def sparse_mr_search_kernel(
    out_key: MutPointer[Int32, MutAnyOrigin],
    out_j: MutPointer[Int32, MutAnyOrigin],
    xt: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    norms: MutPointer[Float32, MutAnyOrigin],
    core: MutPointer[Float32, MutAnyOrigin],
    comp: MutPointer[Int32, MutAnyOrigin],
    todo: MutPointer[Int32, MutAnyOrigin],
    m_in: Int32,
    d_in: Int32,
    n_todo_in: Int32,
    j0_in: Int32,
    j1_in: Int32,
    slice_in: Int32,
    inv_alpha: Float32,
    sabotage: Int32,
):
    """Cell `(s, t)`: the cheapest edge from listed point `todo[t]` to a
    point of another component among `j` in slice `s` of `[j0, j1)`.

    `xt` is X feature-major (`xt[f * m + i]`, adjacent threads read
    adjacent words), `x` row-major (every thread of a block reads the same
    `x[j * d + f]`). The chain is `pinned_distance_tile_kernel`'s cell
    (row i, col j): `fma(ftz(x_i[f]), ftz(x_j[f]), acc)`, f ascending.
    Every cell is written, a non-finite weight as `SMR_BAD_J`."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var n_todo = Int(n_todo_in)
    if t >= n_todo:
        return
    var m = Int(m_in)
    var d = Int(d_in)
    var s = Int(block_idx.y)
    var ja = Int(j0_in) + s * Int(slice_in)
    var jb = ja + Int(slice_in)
    if jb > Int(j1_in):
        jb = Int(j1_in)
    var i = Int(todo[t])
    var ci = comp[i]
    var ni = norms[i]
    var cri = core[i]
    var bk = WEIGHT_KEY_SENTINEL
    var bj = SMR_NONE_J
    var bad = False
    for j in range(ja, jb):
        if comp[j] == ci:
            continue
        var acc = Float32(0.0)
        for f in range(d):
            acc = ftz(identical_mul_add(ftz(xt[f * m + i]), ftz(x[j * d + f]), acc))
        var v = mr_edge_weight(acc, ni, norms[j], cri, core[j], inv_alpha, sabotage)
        if (bitcast[DType.uint32](v) & 0x7F800000) == 0x7F800000:
            bad = True
            continue
        var key = weight_order_key(v)
        if key < bk:
            bk = key
            bj = Int32(j)
    var cell = s * n_todo + t
    out_key[cell] = bk
    out_j[cell] = SMR_BAD_J if bad else bj


@fieldwise_init
struct SparseMst(Movable):
    """The m - 1 tree edges sorted by (weight key, lo, hi), oriented
    (lo, hi), and the dense solver's round count."""

    var lo: List[Int32]
    var hi: List[Int32]
    var w: List[Float32]
    var rounds: Int


@fieldwise_init
struct _Search(Movable):
    var key_d: DeviceBuffer[DType.int32]
    var j_d: DeviceBuffer[DType.int32]
    var todo_d: DeviceBuffer[DType.int32]
    var cap: Int


def _search(
    ctx: DeviceContext,
    mut sb: _Search,
    mut xt_d: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    mut norms_d: DeviceBuffer[DType.float32],
    mut core_d: DeviceBuffer[DType.float32],
    mut comp_d: DeviceBuffer[DType.int32],
    todo: List[Int32],
    m: Int,
    d: Int,
    inv_alpha: Float32,
    sabotage: Int32,
    mut pk: List[Int32],
    mut pj: List[Int32],
) raises -> Int:
    """Exact cheapest other-component edge of every point in `todo`, into
    `pk` / `pj`. Returns the number of launches."""
    var n_todo = len(todo)
    if n_todo == 0:
        return 0
    ctx.enqueue_copy(
        dst_buf=sb.todo_d.create_sub_buffer[DType.int32](0, n_todo),
        src_ptr=todo.unsafe_ptr(),
    )
    var bk = List[Int32](length=n_todo, fill=WEIGHT_KEY_SENTINEL)
    var bj = List[Int32](length=n_todo, fill=SMR_NONE_J)
    var dd = d if d > 0 else 1
    var span = SPARSE_MR_LAUNCH_MACS // (n_todo * dd)
    if span < 1:
        span = 1
    var want_s = (SPARSE_MR_TARGET_THREADS + n_todo - 1) // n_todo
    var h_key = ctx.enqueue_create_host_buffer[DType.int32](sb.cap)
    var h_j = ctx.enqueue_create_host_buffer[DType.int32](sb.cap)
    ctx.synchronize()
    var launches = 0
    var j0 = 0
    while j0 < m:
        var j1 = min(m, j0 + span)
        var width = j1 - j0
        var n_s = min(width, max(1, want_s))
        if n_s * n_todo > sb.cap:
            n_s = max(1, sb.cap // n_todo)
        var sl = (width + n_s - 1) // n_s
        n_s = (width + sl - 1) // sl
        var cells = n_s * n_todo
        var vj = sb.j_d.create_sub_buffer[DType.int32](0, cells)
        var vk = sb.key_d.create_sub_buffer[DType.int32](0, cells)
        # POISON, then the bounded launch, then read back whole.
        ctx.enqueue_memset(vj, SMR_POISON_J)
        ctx.enqueue_function[sparse_mr_search_kernel](
            sb.key_d.unsafe_ptr(), sb.j_d.unsafe_ptr(),
            xt_d.unsafe_ptr(), x.unsafe_ptr(), norms_d.unsafe_ptr(),
            core_d.unsafe_ptr(), comp_d.unsafe_ptr(), sb.todo_d.unsafe_ptr(),
            Int32(m), Int32(d), Int32(n_todo), Int32(j0), Int32(j1),
            Int32(sl), inv_alpha, sabotage,
            grid_dim=((n_todo + SPARSE_MR_TPB - 1) // SPARSE_MR_TPB, n_s, 1),
            block_dim=(SPARSE_MR_TPB, 1, 1),
        )
        ctx.enqueue_copy(dst_ptr=h_key.unsafe_ptr(), src_buf=vk)
        ctx.enqueue_copy(dst_ptr=h_j.unsafe_ptr(), src_buf=vj)
        ctx.synchronize()
        launches += 1
        # Fold the slices in ascending order: (key, j), strict.
        for s in range(n_s):
            for t in range(n_todo):
                var c = s * n_todo + t
                var jj = h_j.unsafe_ptr().unsafe_load(c)
                if jj == SMR_POISON_J:
                    raise Error(
                        "hdbscan.sparse_mr_mst: a search launch left cell "
                        + String(c) + " of " + String(cells)
                        + " unwritten (the poison survived); the launch was"
                        " cut short (on Apple, macOS aborts a long command"
                        " buffer without reporting it). Refused by name"
                    )
                if jj == SMR_BAD_J:
                    raise Error(
                        "hdbscan.build_mr_linkage: a mutual reachability"
                        " weight from row " + String(Int(todo[t]))
                        + " is NaN or infinite (a non-finite input row, or"
                        " rows whose squared difference overflows Float32);"
                        " refused by name (DEVIATION 623 / 1607,"
                        " IDENTITY_PATHS row 39)"
                    )
                if jj < 0:
                    continue
                var kk = h_key.unsafe_ptr().unsafe_load(c)
                if bj[t] < 0 or kk < bk[t] or (kk == bk[t] and jj < bj[t]):
                    bk[t] = kk
                    bj[t] = jj
        _ = vj^
        _ = vk^
        j0 = j1
    for t in range(n_todo):
        var i = Int(todo[t])
        pk[i] = bk[t]
        pj[i] = bj[t]
    _ = h_key^
    _ = h_j^
    return launches


def sparse_mr_mst(
    ctx: DeviceContext,
    x_host: List[Float32],
    mut x: DeviceBuffer[DType.float32],
    mut core_d: DeviceBuffer[DType.float32],
    m: Int,
    d: Int,
    inv_alpha: Float32,
    sabotage: Int32 = HDB_SAB_NONE,
) raises -> SparseMst:
    """The dense arm's `build_sorted_mst` result on the mutual reachability
    graph, with no m x m array: edges sorted by (weight key, lo, hi),
    oriented (lo, hi), weights bit for bit, and the round count."""
    # `pairwise_distances`'s norms: the same kernel, the same launch.
    var norms_d = ctx.enqueue_create_buffer[DType.float32](m)
    ctx.enqueue_function[row_norm_kernel](
        norms_d.unsafe_ptr(), x.unsafe_ptr(), Int32(d), Int32(0),
        grid_dim=(m, 1, 1), block_dim=(NORM_TPB, 1, 1),
    )
    var xt = List[Float32](length=m * d, fill=Float32(0.0))
    for i in range(m):
        for f in range(d):
            xt[f * m + i] = x_host[i * d + f]
    var xt_d = ctx.enqueue_create_buffer[DType.float32](m * d)
    ctx.enqueue_copy(dst_buf=xt_d, src_ptr=xt.unsafe_ptr())
    var comp_d = ctx.enqueue_create_buffer[DType.int32](m)
    var cap = m + SPARSE_MR_TARGET_THREADS
    var sb = _Search(
        ctx.enqueue_create_buffer[DType.int32](cap),
        ctx.enqueue_create_buffer[DType.int32](cap),
        ctx.enqueue_create_buffer[DType.int32](m),
        cap,
    )
    ctx.synchronize()

    var parent = List[Int](capacity=m)
    var comp = List[Int32](capacity=m)
    for i in range(m):
        parent.append(i)
        comp.append(Int32(i))
    var pk = List[Int32](length=m, fill=WEIGHT_KEY_SENTINEL)
    var pj = List[Int32](length=m, fill=SMR_NONE_J)
    var dropped = List[Bool](length=m, fill=False)
    var lb = List[Int32](length=m, fill=SMR_KEY_MIN)
    var ub = List[Int32](length=m, fill=WEIGHT_KEY_SENTINEL)
    var arg = List[Int](length=m, fill=-1)
    var cb = List[Int](length=m, fill=-1)
    var lo = List[Int32](capacity=m)
    var hi = List[Int32](capacity=m)
    var w = List[Float32](capacity=m)
    var n_comp = m
    var merge_rounds = 0
    while n_comp > 1:
        merge_rounds += 1
        if merge_rounds > 64:
            raise Error("hdbscan.sparse_mr_mst: Boruvka did not converge")
        ctx.enqueue_copy(dst_buf=comp_d, src_ptr=comp.unsafe_ptr())
        # The plan: exact points set the component bound; listed points
        # carry a lower bound; strictly-above-bound points sit out.
        for c in range(m):
            ub[c] = WEIGHT_KEY_SENTINEL
            arg[c] = -1
        var listed = List[Bool](length=m, fill=False)
        for i in range(m):
            dropped[i] = False
            var ci = Int(comp[i])
            var j = Int(pj[i])
            if j >= 0 and comp[j] != comp[i]:
                if pk[i] < ub[ci]:
                    ub[ci] = pk[i]
            else:
                listed[i] = True
                lb[i] = pk[i] if j >= 0 else SMR_KEY_MIN
        for i in range(m):
            if not listed[i]:
                continue
            var ci = Int(comp[i])
            if lb[i] > ub[ci]:
                dropped[i] = True
            elif arg[ci] < 0 or lb[i] < lb[arg[ci]]:
                arg[ci] = i
        var todo_a = List[Int32]()
        var deferred = List[Int32]()
        for i in range(m):
            if not listed[i] or dropped[i]:
                continue
            if arg[Int(comp[i])] == i:
                todo_a.append(Int32(i))
            else:
                deferred.append(Int32(i))
        _ = _search(
            ctx, sb, xt_d, x, norms_d, core_d, comp_d, todo_a, m, d,
            inv_alpha, sabotage, pk, pj,
        )
        if len(deferred) > 0:
            for t in range(len(todo_a)):
                var i = Int(todo_a[t])
                var ci = Int(comp[i])
                if pj[i] >= 0 and pk[i] < ub[ci]:
                    ub[ci] = pk[i]
            var todo_b = List[Int32]()
            for t in range(len(deferred)):
                var i = Int(deferred[t])
                if lb[i] > ub[Int(comp[i])]:
                    dropped[i] = True
                else:
                    todo_b.append(Int32(i))
            _ = _search(
                ctx, sb, xt_d, x, norms_d, core_d, comp_d, todo_b, m, d,
                inv_alpha, sabotage, pk, pj,
            )
        # Each component's cheapest edge under (key, lo, hi).
        for i in range(m):
            if dropped[i] or pj[i] < 0:
                continue
            var ci = Int(comp[i])
            var j = Int(pj[i])
            var b = cb[ci]
            if b < 0 or triple_less_i(
                pk[i], min(i, j), max(i, j),
                pk[b], min(b, Int(pj[b])), max(b, Int(pj[b])),
            ):
                cb[ci] = i
        var added = 0
        for c in range(m):
            var i = cb[c]
            if i < 0:
                continue
            cb[c] = -1
            var j = Int(pj[i])
            var ra = _find(parent, i)
            var rb = _find(parent, j)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
                lo.append(Int32(min(i, j)))
                hi.append(Int32(max(i, j)))
                w.append(weight_order_unkey(pk[i]))
                n_comp -= 1
                added += 1
        if added == 0:
            raise Error(
                "hdbscan.sparse_mr_mst: a Boruvka round joined nothing with "
                + String(n_comp) + " components left"
            )
        for i in range(m):
            comp[i] = Int32(_find(parent, i))

    var order = sort_edges_total_order(lo, hi, w)
    var slo = List[Int32](capacity=len(order))
    var shi = List[Int32](capacity=len(order))
    var sw = List[Float32](capacity=len(order))
    for t in range(len(order)):
        slo.append(lo[order[t]])
        shi.append(hi[order[t]])
        sw.append(w[order[t]])
    var rounds = boruvka_rounds_on_tree(slo, shi, sw, m)
    if rounds != merge_rounds + 1:
        raise Error(
            "hdbscan.sparse_mr_mst: the replayed round count " + String(rounds)
            + " is not the search's " + String(merge_rounds) + " + 1; the"
            " tree is not the minimum one under (weight key, lo, hi)"
        )
    _ = norms_d^
    _ = xt_d^
    _ = xt^
    _ = comp_d^
    _ = sb^
    return SparseMst(slo^, shi^, sw^, rounds)


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
