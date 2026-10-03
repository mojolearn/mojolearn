"""FAST-only Euclidean minimum spanning tree without the dense matrix (Apple).

The PAIRWISE route materialises `m * m` distances and `m * m` indices
(11.6 GB at m = 38,000), which is past what the M4 can allocate, so the
build failed from 38k rows up. Here the MST comes from Boruvka rounds with
the distances computed on the fly:

  * `fb_nearest_other_kernel`: for every point, the nearest point in a
    DIFFERENT component, over the whole data in shared-memory tiles
    (`O(m^2 d)` work, `O(m)` memory), ties to the lower index;
  * each component's cheapest outgoing edge under the total order (squared
    distance, min(i, j), max(i, j)) -- the per-point choice above is
    consistent with it, so no round can close a cycle -- joins the
    components and relabels the points, all on the device (the banner
    above `fast_euclidean_mst`).

Rounds at least halve the component count. The edges come back sorted by
(weight, src, dst) for the dendrogram, the weight rooted for
L2SqrtExpanded. FAST arithmetic: direct sums of squared differences.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.math import sqrt
from std.memory import bitcast
from std.atomic import Atomic
from core.fast_radix_sort import (
    fast_radix_sort_pairs_u32,
    frs_counts_len,
    frs_exclusive_scan,
    frs_scan_blocks,
)
from std.os import getenv
from std.time import perf_counter_ns
from hierarchy.impl.cluster.detail.fast_mma_boruvka import MmaBoruvka

comptime FB_TPB = 128
comptime FB_TILE = 64
comptime FB_PPT = 4
"""Listed points per thread at `n_cols <= 16` (register blocking)."""


def fb_nearest_other_kernel[DMAX: Int, MR: Bool = False, PPT: Int = 1](
    x: MutPointer[Float32, MutAnyOrigin],
    core: MutPointer[Float32, MutAnyOrigin],
    inv_alpha: Float32,
    comp: MutPointer[Int32, MutAnyOrigin],
    best_d: MutPointer[Float32, MutAnyOrigin],
    best_j: MutPointer[Int32, MutAnyOrigin],
    m_in: Int32,
    d_in: Int32,
    todo: MutPointer[Int32, MutAnyOrigin],
    n_todo_in: Int32,
):
    """Each thread serves `PPT` listed points, so every value read from the
    shared tile feeds `PPT` distance updates (register blocking)."""
    var m = Int(m_in)
    var dim = Int(d_in)
    var base_t = (Int(block_idx.x) * FB_TPB + Int(thread_idx.x)) * PPT
    var live = InlineArray[Bool, PPT](fill=False)
    var ii = InlineArray[Int, PPT](fill=0)
    var ci = InlineArray[Int32, PPT](fill=Int32(-1))
    var cri = InlineArray[Float32, PPT](fill=0.0)
    var xi = InlineArray[Float32, PPT * DMAX](fill=0.0)
    comptime for pp in range(PPT):
        if base_t + pp < Int(n_todo_in):
            live[pp] = True
            var i = Int(todo[base_t + pp])
            ii[pp] = i
            ci[pp] = comp[i]
            comptime if MR:
                cri[pp] = core[i]
            comptime for t in range(DMAX):
                if t < dim:
                    xi[pp * DMAX + t] = x[i * dim + t]
    var tile = stack_allocation[
        FB_TILE * DMAX, Float32, address_space=AddressSpace.SHARED
    ]()
    var tcomp = stack_allocation[
        FB_TILE, Int32, address_space=AddressSpace.SHARED
    ]()
    var tcore = stack_allocation[
        FB_TILE, Float32, address_space=AddressSpace.SHARED
    ]()
    var bd = InlineArray[Float32, PPT](fill=Float32.MAX)
    var bj = InlineArray[Int32, PPT](fill=Int32(-1))
    var bad = InlineArray[Bool, PPT](fill=False)
    var j0 = 0
    while j0 < m:
        var e = Int(thread_idx.x)
        while e < FB_TILE * DMAX:
            var jj = j0 + e // DMAX
            var tt = e % DMAX
            if jj < m and tt < dim:
                tile[e] = x[jj * dim + tt]
            else:
                tile[e] = 0.0
            e += FB_TPB
        if Int(thread_idx.x) < FB_TILE:
            var jj = j0 + Int(thread_idx.x)
            tcomp[Int(thread_idx.x)] = comp[jj] if jj < m else Int32(-1)
            comptime if MR:
                tcore[Int(thread_idx.x)] = core[jj] if jj < m else Float32(0)
        barrier()
        var jn = min(FB_TILE, m - j0)
        for u in range(jn):
            var cu = tcomp[u]
            var dd = SIMD[DType.float32, PPT](0)
            comptime for t in range(DMAX):
                var tv = tile[u * DMAX + t]
                comptime for pp in range(PPT):
                    var df = xi[pp * DMAX + t] - tv
                    dd[pp] += df * df
            comptime for pp in range(PPT):
                if live[pp] and cu != ci[pp]:
                    var v = dd[pp]
                    comptime if MR:
                        # Mutual reachability (reachability.cuh:222-255):
                        # max(core_j, max(core_i, (1/alpha) * d)).
                        v = max(tcore[u], max(cri[pp], inv_alpha * sqrt(v)))
                    # Exponent bits, not a float compare: FAST arithmetic
                    # may assume no inf / NaN and fold the compare away.
                    if (bitcast[DType.uint32](v) & 0x7F800000) == 0x7F800000:
                        bad[pp] = True
                    elif v < bd[pp]:
                        bd[pp] = v
                        bj[pp] = Int32(j0 + u)
        barrier()
        j0 += FB_TILE
    comptime for pp in range(PPT):
        if live[pp]:
            best_d[ii[pp]] = bd[pp]
            best_j[ii[pp]] = Int32(-2) if bad[pp] else bj[pp]


def _fb_search(
    ctx: DeviceContext,
    mut mb: MmaBoruvka,
    mut x: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    mutual_reach: Bool,
    core_ptr: MutPointer[Float32, MutAnyOrigin],
    inv_alpha: Float32,
    mut comp_d: DeviceBuffer[DType.int32],
    mut todo_d: DeviceBuffer[DType.int32],
    n_todo: Int,
    mut bd_d: DeviceBuffer[DType.float32],
    mut bj_d: DeviceBuffer[DType.int32],
) raises:
    """The nearest other-component point of the `n_todo` points listed in
    the device list `todo_d` into `bd_d` / `bj_d` (their entries only)."""
    if n_todo <= 0:
        return
    if mb.ok:
        mb.enqueue(
            ctx, x, m, n, mutual_reach, core_ptr, inv_alpha, comp_d,
            todo_d, n_todo, bd_d, bj_d,
        )
        return
    var grid = (n_todo + FB_TPB - 1) // FB_TPB
    comptime for DM in [8, 16, 32, 64]:
        # The reachability arm keeps one point per thread (its per-pair
        # sqrt, not the tile reads, dominates: HDBSCAN taxi 100k 9.6 s
        # at 1 vs 10.2 s at 4).
        comptime P = FB_PPT if DM <= 16 else 1
        var gridp = (n_todo + FB_TPB * P - 1) // (FB_TPB * P)
        if mutual_reach:
            gridp = grid
        if n <= DM and (DM == 8 or n > DM // 2):
            if mutual_reach:
                ctx.enqueue_function[fb_nearest_other_kernel[DM, True, 1]](
                    x.unsafe_ptr(), core_ptr, inv_alpha,
                    comp_d.unsafe_ptr(), bd_d.unsafe_ptr(),
                    bj_d.unsafe_ptr(), Int32(m), Int32(n),
                    todo_d.unsafe_ptr(), Int32(n_todo),
                    grid_dim=gridp, block_dim=FB_TPB,
                )
            else:
                ctx.enqueue_function[fb_nearest_other_kernel[DM, False, P]](
                    x.unsafe_ptr(), core_ptr, inv_alpha,
                    comp_d.unsafe_ptr(), bd_d.unsafe_ptr(),
                    bj_d.unsafe_ptr(), Int32(m), Int32(n),
                    todo_d.unsafe_ptr(), Int32(n_todo),
                    grid_dim=gridp, block_dim=FB_TPB,
                )


# ---------------------------------------------------------------------------
# lane/apple-fast-purity2 (2026-10-03): the Boruvka rounds on the device.
#
# The rounds were a host loop over the m points after downloading the
# search's (distance, neighbour) pairs: each component's cheapest edge, a
# union-find, the relabel and the next round's plan. Here every step is a
# kernel over the points (or the components), with 32-bit integer atomics
# only (Apple has no 64-bit atomics):
#   * a component's cheapest edge under the total order (squared distance,
#     min(i, j), max(i, j)) is three atomic-min passes: the distance bits
#     (non-negative floats order as their bits), then the low end among the
#     points at that distance, then the high end;
#   * a component keeps its edge unless the component at the other end
#     chose the SAME edge and has the lower label (the host loop's skip:
#     components ascending, the second sighting of a mutual edge found its
#     ends already joined; the chosen edges form a forest otherwise, so it
#     skipped nothing else). The kept edges take their slots by an
#     exclusive scan over the component labels: discovery order, round by
#     round, components ascending, as the host appended them;
#   * the merged component's label is its smallest old label (the host's
#     `parent[max] = min`): atomic-min label propagation over the kept
#     edges plus pointer jumping, to a fixed point (a scalar flag per pass);
#   * the plan (listed / dropped / each component's first searched point)
#     is the host's tests point by point: atomic mins for the bound and the
#     smallest lower bound (ties to the lower index), atomic appends for the
#     two search lists (their order reaches no result: each point's search
#     is its own);
#   * the edges are sorted by weight, ties by discovery order, with the
#     stable radix sort of (weight bits, slot).
# Only scalars come home: the list lengths, the kept-edge count, the
# refusal row and the fixed-point flag.
# ---------------------------------------------------------------------------

comptime FBR_TPB = 256
comptime FBR_NONE = Int32(2147483647)
"""The empty slot of an atomic-min cell (above every non-negative float's
bits and every index)."""

comptime _FI = MutPointer[Int32, MutAnyOrigin]
comptime _FF = MutPointer[Float32, MutAnyOrigin]
comptime _FU = MutPointer[UInt32, MutAnyOrigin]


@always_inline
def _fbits(v: Float32) -> Int32:
    return Int32(Int(bitcast[DType.uint32](v)))


@always_inline
def _ffrom(b: Int32) -> Float32:
    return bitcast[DType.float32](UInt32(Int(b)))


def _fbr_grid(count: Int) -> Int:
    return (count + FBR_TPB - 1) // FBR_TPB if count > 0 else 1


@always_inline
def _fbr_tid() -> Int:
    return Int(block_idx.x) * FBR_TPB + Int(thread_idx.x)


def fbr_fill_kernel(buf: _FI, n: Int32, v: Int32):
    var i = _fbr_tid()
    if i < Int(n):
        buf[i] = v


def fbr_iota_kernel(a: _FI, b: _FI, n: Int32):
    var i = _fbr_tid()
    if i < Int(n):
        a[i] = Int32(i)
        b[i] = Int32(i)


def fbr_iota1_kernel(a: _FI, n: Int32):
    var i = _fbr_tid()
    if i < Int(n):
        a[i] = Int32(i)


def fbr_phase_b_bound_kernel(todo: _FI, n_todo: Int32, comp: _FI, bd: _FF, bj: _FI, ub: _FI):
    """Phase A's exact values join their component's bound."""
    var t = _fbr_tid()
    if t < Int(n_todo):
        var i = Int(todo[t])
        if Int(bj[i]) >= 0:
            _ = Atomic.min(ub + Int(comp[i]), _fbits(bd[i]))


def fbr_phase_b_list_kernel(
    defer: _FI, n_defer: Int32, comp: _FI, bd: _FF, bj: _FI, ub: _FI,
    dropped: _FI, listb: _FI, cnt: _FI,
):
    """A deferred point sits the round out when its lower bound exceeds its
    component's bound; else it joins phase B's list. cnt[0] the list
    length, cnt[1] the dropped count."""
    var t = _fbr_tid()
    if t < Int(n_defer):
        var i = Int(defer[t])
        var ri = Int(comp[i])
        var lb = Float32(0)
        if Int(bj[i]) >= 0:
            lb = bd[i]
        if lb > _ffrom(ub[ri]):
            dropped[i] = Int32(1)
            _ = Atomic.fetch_add(cnt + 1, Int32(1))
        else:
            var p = Atomic.fetch_add(cnt, Int32(1))
            listb[Int(p)] = Int32(i)


def fbr_edge_pass_kernel(
    m_in: Int32, comp: _FI, bd: _FF, bj: _FI, dropped: _FI,
    cbd: _FI, cba: _FI, cbb: _FI, err: _FI, stage: Int32,
):
    """One of the cheapest-edge passes over the points: stage 0 the
    distance bits (and the refusal row: the lowest i whose search saw a
    NaN or an overflow), 1 the low end at that distance, 2 the high end."""
    var i = _fbr_tid()
    if i >= Int(m_in):
        return
    var j = Int(bj[i])
    if stage == Int32(0) and j == -2:
        _ = Atomic.min(err, Int32(i))
        return
    if j < 0 or dropped[i] != Int32(0):
        return
    var c = Int(comp[i])
    var db = _fbits(bd[i])
    if stage == Int32(0):
        _ = Atomic.min(cbd + c, db)
        return
    if db != cbd[c]:
        return
    var a = Int32(min(i, j))
    if stage == Int32(1):
        _ = Atomic.min(cba + c, a)
        return
    if a == cba[c]:
        _ = Atomic.min(cbb + c, Int32(max(i, j)))


def fbr_keep_kernel(m_in: Int32, comp: _FI, cba: _FI, cbb: _FI, keep: _FI, scan: _FI):
    """Component c keeps its edge unless the other end's component chose
    the same edge and has the lower label."""
    var c = _fbr_tid()
    if c >= Int(m_in):
        return
    var k = Int32(0)
    var a = cba[c]
    if a != FBR_NONE:
        var b = cbb[c]
        var ca = Int(comp[Int(a)])
        var o = ca if ca != c else Int(comp[Int(b)])
        k = Int32(1)
        if cba[o] == a and cbb[o] == b and o < c:
            k = Int32(0)
    keep[c] = k
    scan[c] = k


def fbr_write_edges_kernel(
    m_in: Int32, keep: _FI, scan: _FI, cbd: _FI, cba: _FI, cbb: _FI,
    base: Int32, cap: Int32, es: _FI, ed: _FI, ew: _FU, total: _FI,
):
    """A kept edge to slot base + its exclusive-scan rank; the last
    component also writes the round's kept count."""
    var c = _fbr_tid()
    var m = Int(m_in)
    if c >= m:
        return
    if c == m - 1:
        total[0] = scan[c] + keep[c]
    if keep[c] != Int32(0):
        var e = Int(base) + Int(scan[c])
        if e >= Int(cap):
            return  # the host refuses the round by its count
        es[e] = cba[c]
        ed[e] = cbb[c]
        ew[e] = UInt32(Int(cbd[c]))


def fbr_label_hook_kernel(m_in: Int32, keep: _FI, comp: _FI, cba: _FI, cbb: _FI, lab: _FI, flag: _FI):
    """Both ends of a kept edge take the smaller of their labels."""
    var c = _fbr_tid()
    if c >= Int(m_in) or keep[c] == Int32(0):
        return
    var c1 = Int(comp[Int(cba[c])])
    var c2 = Int(comp[Int(cbb[c])])
    var l1 = lab[c1]
    var l2 = lab[c2]
    if l1 != l2:
        var lo = min(l1, l2)
        _ = Atomic.min(lab + c1, lo)
        _ = Atomic.min(lab + c2, lo)
        flag[0] = Int32(1)


def fbr_label_jump_kernel(m_in: Int32, lab: _FI, flag: _FI):
    """Pointer jumping: a label takes its label's label."""
    var c = _fbr_tid()
    if c >= Int(m_in):
        return
    var l = lab[c]
    var ll = lab[Int(l)]
    if ll < l:
        _ = Atomic.min(lab + c, ll)
        flag[0] = Int32(1)


def fbr_relabel_kernel(m_in: Int32, comp: _FI, lab: _FI):
    var i = _fbr_tid()
    if i < Int(m_in):
        comp[i] = lab[Int(comp[i])]


def fbr_plan_bound_kernel(m_in: Int32, comp: _FI, bd: _FF, bj: _FI, dropped: _FI, listed: _FI, ub: _FI):
    """A point is listed when its nearest other-component point joined its
    component (or it has none); an unlisted point's value is exact and
    joins its component's bound."""
    var i = _fbr_tid()
    if i >= Int(m_in):
        return
    var ri = comp[i]
    var j = Int(bj[i])
    dropped[i] = Int32(0)
    if j < 0 or comp[j] == ri:
        listed[i] = Int32(1)
    else:
        listed[i] = Int32(0)
        _ = Atomic.min(ub + Int(ri), _fbits(bd[i]))


def fbr_plan_pass_kernel(
    m_in: Int32, comp: _FI, bd: _FF, bj: _FI, dropped: _FI, listed: _FI,
    ub: _FI, minlb: _FI, arg: _FI, todo: _FI, defer: _FI, cnt: _FI, stage: Int32,
):
    """The plan's passes over the listed points: stage 0 drops a point
    whose lower bound exceeds its component's bound (cnt[2] counts them)
    and takes the smallest lower bound; stage 1 the lowest index at it;
    stage 2 lists the point for phase A (that index; cnt[0]) or phase B
    (cnt[1])."""
    var i = _fbr_tid()
    if i >= Int(m_in) or listed[i] == Int32(0):
        return
    var ri = Int(comp[i])
    var lb = Float32(0)
    if Int(bj[i]) >= 0:
        lb = bd[i]
    if stage == Int32(0):
        if lb > _ffrom(ub[ri]):
            dropped[i] = Int32(1)
            _ = Atomic.fetch_add(cnt + 2, Int32(1))
        else:
            _ = Atomic.min(minlb + ri, _fbits(lb))
        return
    if dropped[i] != Int32(0):
        return
    if stage == Int32(1):
        if _fbits(lb) == minlb[ri]:
            _ = Atomic.min(arg + ri, Int32(i))
        return
    if Int(arg[ri]) == i:
        var p = Atomic.fetch_add(cnt, Int32(1))
        todo[Int(p)] = Int32(i)
    else:
        var q = Atomic.fetch_add(cnt + 1, Int32(1))
        defer[Int(q)] = Int32(i)


def fbr_gather_edges_kernel(
    ne_in: Int32, order: _FU, es: _FI, ed: _FI, ew: _FU,
    rows: _FI, cols: _FI, w_out: _FF, root: Int32,
):
    """Slot t of the sorted edges: the edge `order[t]` (its weight rooted
    for L2SqrtExpanded)."""
    var t = _fbr_tid()
    if t >= Int(ne_in):
        return
    var e = Int(order[t])
    rows[t] = es[e]
    cols[t] = ed[e]
    var w = bitcast[DType.float32](ew[e])
    if root != Int32(0):
        w = sqrt(w)
    w_out[t] = w


def _fbr_read(ctx: DeviceContext, mut buf: DeviceBuffer[DType.int32], count: Int) raises -> List[Int]:
    """`count` (at most 4) scalars home: list lengths, counts, flags."""
    var h = ctx.enqueue_create_host_buffer[DType.int32](count)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=buf.create_sub_buffer[DType.int32](0, count))
    ctx.synchronize()
    var out = List[Int](capacity=count)
    for k in range(count):
        out.append(Int(h.unsafe_ptr()[k]))
    _ = h^
    return out^


def fast_euclidean_mst(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    is_sqrt: Bool,
    mut mst_rows: DeviceBuffer[DType.int32],
    mut mst_cols: DeviceBuffer[DType.int32],
    mut mst_weights: DeviceBuffer[DType.float32],
    mutual_reach: Bool,
    core_ptr: MutPointer[Float32, MutAnyOrigin],
    inv_alpha: Float32 = 1.0,
) raises -> Int:
    """`m - 1` edges into the three buffers, ascending by (weight, src,
    dst): weight, then discovery order. Returns the Boruvka round count. `x` is `m x n` row-major,
    `n <= 64`. Every round runs on the device (see the banner above)."""
    if n > 64:
        raise Error("fast_euclidean_mst: n_cols > 64 is not taken here")
    var mg = _fbr_grid(m)
    var comp_d = ctx.enqueue_create_buffer[DType.int32](m)
    var todo_d = ctx.enqueue_create_buffer[DType.int32](m)
    var defer_d = ctx.enqueue_create_buffer[DType.int32](m)
    var listb_d = ctx.enqueue_create_buffer[DType.int32](m)
    var bd_d = ctx.enqueue_create_buffer[DType.float32](m)
    var bj_d = ctx.enqueue_create_buffer[DType.int32](m)
    var dropped_d = ctx.enqueue_create_buffer[DType.int32](m)
    var listed_d = ctx.enqueue_create_buffer[DType.int32](m)
    var ub_d = ctx.enqueue_create_buffer[DType.int32](m)
    var minlb_d = ctx.enqueue_create_buffer[DType.int32](m)
    var arg_d = ctx.enqueue_create_buffer[DType.int32](m)
    var cbd_d = ctx.enqueue_create_buffer[DType.int32](m)
    var cba_d = ctx.enqueue_create_buffer[DType.int32](m)
    var cbb_d = ctx.enqueue_create_buffer[DType.int32](m)
    var keep_d = ctx.enqueue_create_buffer[DType.int32](m)
    var scan_d = ctx.enqueue_create_buffer[DType.int32](m)
    var lab_d = ctx.enqueue_create_buffer[DType.int32](m)
    var bsum_d = ctx.enqueue_create_buffer[DType.int32](max(frs_scan_blocks(m), 1))
    var cnt_d = ctx.enqueue_create_buffer[DType.int32](4)
    var ecap = max(m - 1, 1)
    var es_d = ctx.enqueue_create_buffer[DType.int32](ecap)
    var ed_d = ctx.enqueue_create_buffer[DType.int32](ecap)
    var ew_d = ctx.enqueue_create_buffer[DType.uint32](ecap)
    var mi = Int32(m)
    ctx.enqueue_function[fbr_iota_kernel](comp_d.unsafe_ptr(), todo_d.unsafe_ptr(), mi, grid_dim=mg, block_dim=FBR_TPB)
    ctx.enqueue_function[fbr_fill_kernel](dropped_d.unsafe_ptr(), mi, Int32(0), grid_dim=mg, block_dim=FBR_TPB)
    ctx.enqueue_function[fbr_fill_kernel](ub_d.unsafe_ptr(), mi, FBR_NONE, grid_dim=mg, block_dim=FBR_TPB)
    ctx.enqueue_function[fbr_fill_kernel](minlb_d.unsafe_ptr(), mi, FBR_NONE, grid_dim=mg, block_dim=FBR_TPB)
    ctx.enqueue_function[fbr_fill_kernel](arg_d.unsafe_ptr(), mi, FBR_NONE, grid_dim=mg, block_dim=FBR_TPB)
    ctx.enqueue_function[fbr_fill_kernel](cbd_d.unsafe_ptr(), mi, FBR_NONE, grid_dim=mg, block_dim=FBR_TPB)
    ctx.enqueue_function[fbr_fill_kernel](cba_d.unsafe_ptr(), mi, FBR_NONE, grid_dim=mg, block_dim=FBR_TPB)
    ctx.enqueue_function[fbr_fill_kernel](cbb_d.unsafe_ptr(), mi, FBR_NONE, grid_dim=mg, block_dim=FBR_TPB)
    var n_comp = m
    var n_edges = 0
    var rounds = 0
    var n_todo = m
    var n_defer = 0
    # Only points whose nearest other-component point joined their own
    # component are searched again: components only merge, so a surviving
    # nearest (lowest index on a tie) is still the nearest. Two phases per
    # round. A: for each component, its listed point with the smallest lower
    # bound. B: the other listed points, after A's exact values have
    # tightened the component bound.
    # FAST, Apple, n <= 32: the matrix-unit search with the same answer
    # (`fast_mma_boruvka.mojo`); `mb.ok` is False when it declines.
    var _st_on = getenv("MOJOLEARN_STAGE_TIMES") == "1"
    var mb = MmaBoruvka(ctx, x, m, n, mutual_reach, core_ptr, inv_alpha)
    while n_comp > 1:
        rounds += 1
        var _tq0 = 0
        if _st_on:
            ctx.synchronize()
            _tq0 = Int(perf_counter_ns())
        _fb_search(
            ctx, mb, x, m, n, mutual_reach, core_ptr, inv_alpha, comp_d,
            todo_d, n_todo, bd_d, bj_d,
        )
        var n_b = 0
        var n_drop_b = 0
        if n_defer > 0:
            ctx.enqueue_function[fbr_phase_b_bound_kernel](
                todo_d.unsafe_ptr(), Int32(n_todo), comp_d.unsafe_ptr(), bd_d.unsafe_ptr(),
                bj_d.unsafe_ptr(), ub_d.unsafe_ptr(), grid_dim=_fbr_grid(n_todo), block_dim=FBR_TPB,
            )
            ctx.enqueue_function[fbr_fill_kernel](cnt_d.unsafe_ptr(), Int32(4), Int32(0), grid_dim=1, block_dim=FBR_TPB)
            ctx.enqueue_function[fbr_phase_b_list_kernel](
                defer_d.unsafe_ptr(), Int32(n_defer), comp_d.unsafe_ptr(), bd_d.unsafe_ptr(),
                bj_d.unsafe_ptr(), ub_d.unsafe_ptr(), dropped_d.unsafe_ptr(), listb_d.unsafe_ptr(),
                cnt_d.unsafe_ptr(), grid_dim=_fbr_grid(n_defer), block_dim=FBR_TPB,
            )
            var cb = _fbr_read(ctx, cnt_d, 2)
            n_b = cb[0]
            n_drop_b = cb[1]
            _fb_search(
                ctx, mb, x, m, n, mutual_reach, core_ptr, inv_alpha, comp_d,
                listb_d, n_b, bd_d, bj_d,
            )
        if _st_on:
            ctx.synchronize()
            print("BORUVKA round=" + String(rounds) + " phaseA=" + String(n_todo)
                  + " phaseB=" + String(n_b) + " droppedB=" + String(n_drop_b)
                  + " ms=" + String((Int(perf_counter_ns()) - _tq0) // 1000000))
        # Each component's cheapest edge, then the refusal row.
        ctx.enqueue_function[fbr_fill_kernel](cnt_d.unsafe_ptr(), Int32(4), FBR_NONE, grid_dim=1, block_dim=FBR_TPB)
        for stage in range(3):
            ctx.enqueue_function[fbr_edge_pass_kernel](
                mi, comp_d.unsafe_ptr(), bd_d.unsafe_ptr(), bj_d.unsafe_ptr(), dropped_d.unsafe_ptr(),
                cbd_d.unsafe_ptr(), cba_d.unsafe_ptr(), cbb_d.unsafe_ptr(), cnt_d.unsafe_ptr(),
                Int32(stage), grid_dim=mg, block_dim=FBR_TPB,
            )
        var bad = _fbr_read(ctx, cnt_d, 1)[0]
        if bad != Int(FBR_NONE):
            raise Error(
                "hierarchy.pairwise_distances: a distance from row "
                + String(bad)
                + " is NaN or overflows Float32 (a non-finite input"
                " row, or rows whose squared difference overflows);"
                " refused by name (DEVIATION 623, IDENTITY_PATHS row 39)"
            )
        # The kept edges in discovery order, then the merged labels.
        ctx.enqueue_function[fbr_keep_kernel](
            mi, comp_d.unsafe_ptr(), cba_d.unsafe_ptr(), cbb_d.unsafe_ptr(), keep_d.unsafe_ptr(),
            scan_d.unsafe_ptr(), grid_dim=mg, block_dim=FBR_TPB,
        )
        frs_exclusive_scan(ctx, scan_d, m, bsum_d)
        ctx.enqueue_function[fbr_write_edges_kernel](
            mi, keep_d.unsafe_ptr(), scan_d.unsafe_ptr(), cbd_d.unsafe_ptr(), cba_d.unsafe_ptr(),
            cbb_d.unsafe_ptr(), Int32(n_edges), Int32(ecap), es_d.unsafe_ptr(), ed_d.unsafe_ptr(), ew_d.unsafe_ptr(),
            cnt_d.unsafe_ptr(), grid_dim=mg, block_dim=FBR_TPB,
        )
        var kept = _fbr_read(ctx, cnt_d, 1)[0]
        if kept <= 0 or n_edges + kept > m - 1:
            raise Error("fast_euclidean_mst: a Boruvka round joined " + String(kept) + " components")
        var lab_p = lab_d.unsafe_ptr()
        ctx.enqueue_function[fbr_iota1_kernel](lab_p, mi, grid_dim=mg, block_dim=FBR_TPB)
        var passes = 0
        while True:
            passes += 1
            ctx.enqueue_function[fbr_fill_kernel](cnt_d.unsafe_ptr(), Int32(1), Int32(0), grid_dim=1, block_dim=FBR_TPB)
            ctx.enqueue_function[fbr_label_hook_kernel](
                mi, keep_d.unsafe_ptr(), comp_d.unsafe_ptr(), cba_d.unsafe_ptr(), cbb_d.unsafe_ptr(),
                lab_p, cnt_d.unsafe_ptr(), grid_dim=mg, block_dim=FBR_TPB,
            )
            ctx.enqueue_function[fbr_label_jump_kernel](mi, lab_p, cnt_d.unsafe_ptr(), grid_dim=mg, block_dim=FBR_TPB)
            if _fbr_read(ctx, cnt_d, 1)[0] == 0:
                break
            if passes > m + 2:
                raise Error("fast_euclidean_mst: the component labels did not settle")
        ctx.enqueue_function[fbr_relabel_kernel](mi, comp_d.unsafe_ptr(), lab_p, grid_dim=mg, block_dim=FBR_TPB)
        ctx.enqueue_function[fbr_fill_kernel](cbd_d.unsafe_ptr(), mi, FBR_NONE, grid_dim=mg, block_dim=FBR_TPB)
        ctx.enqueue_function[fbr_fill_kernel](cba_d.unsafe_ptr(), mi, FBR_NONE, grid_dim=mg, block_dim=FBR_TPB)
        ctx.enqueue_function[fbr_fill_kernel](cbb_d.unsafe_ptr(), mi, FBR_NONE, grid_dim=mg, block_dim=FBR_TPB)
        n_edges += kept
        n_comp -= kept
        if n_comp <= 1:
            break
        # Next round's plan. A listed point (its nearest other-component
        # point joined its own component, or it was never searched) has a
        # LOWER BOUND: its last value, since the other-component set only
        # shrinks (0 if never searched). A component's BOUND is its
        # unlisted members' values, which are still exact. A listed point
        # whose lower bound exceeds its component's bound cannot give the
        # component's cheapest edge (strictly: a tie that could win on the
        # index is still searched); it sits the round out, ignored by the
        # edge pass above and listed again the next round. Phase B repeats
        # the test after phase A's exact values join the bound.
        ctx.enqueue_function[fbr_fill_kernel](ub_d.unsafe_ptr(), mi, FBR_NONE, grid_dim=mg, block_dim=FBR_TPB)
        ctx.enqueue_function[fbr_plan_bound_kernel](
            mi, comp_d.unsafe_ptr(), bd_d.unsafe_ptr(), bj_d.unsafe_ptr(), dropped_d.unsafe_ptr(),
            listed_d.unsafe_ptr(), ub_d.unsafe_ptr(), grid_dim=mg, block_dim=FBR_TPB,
        )
        ctx.enqueue_function[fbr_fill_kernel](cnt_d.unsafe_ptr(), Int32(4), Int32(0), grid_dim=1, block_dim=FBR_TPB)
        for stage in range(3):
            ctx.enqueue_function[fbr_plan_pass_kernel](
                mi, comp_d.unsafe_ptr(), bd_d.unsafe_ptr(), bj_d.unsafe_ptr(), dropped_d.unsafe_ptr(),
                listed_d.unsafe_ptr(), ub_d.unsafe_ptr(), minlb_d.unsafe_ptr(), arg_d.unsafe_ptr(),
                todo_d.unsafe_ptr(), defer_d.unsafe_ptr(), cnt_d.unsafe_ptr(), Int32(stage),
                grid_dim=mg, block_dim=FBR_TPB,
            )
        ctx.enqueue_function[fbr_fill_kernel](minlb_d.unsafe_ptr(), mi, FBR_NONE, grid_dim=mg, block_dim=FBR_TPB)
        ctx.enqueue_function[fbr_fill_kernel](arg_d.unsafe_ptr(), mi, FBR_NONE, grid_dim=mg, block_dim=FBR_TPB)
        var pc = _fbr_read(ctx, cnt_d, 3)
        n_todo = pc[0]
        n_defer = pc[1]
        if _st_on:
            print("BORUVKA plan dropped=" + String(pc[2]) + " deferred="
                  + String(n_defer))
        if rounds > 64:
            raise Error("fast_euclidean_mst: Boruvka did not converge")
    # Sort the edges by weight (non-negative float bits order as the
    # floats do), ties by the order Boruvka found them (the slot): a stable
    # radix sort of (weight bits, slot).
    var ne = n_edges
    if ne > 0:
        var order_d = ctx.enqueue_create_buffer[DType.uint32](ne)
        var tk = ctx.enqueue_create_buffer[DType.uint32](ne)
        var tv = ctx.enqueue_create_buffer[DType.uint32](ne)
        var counts = ctx.enqueue_create_buffer[DType.int32](frs_counts_len(ne))
        var keys_d = ctx.enqueue_create_buffer[DType.uint32](ne)
        ctx.enqueue_copy(dst_buf=keys_d, src_buf=ew_d.create_sub_buffer[DType.uint32](0, ne))
        ctx.enqueue_function[fbr_iota_kernel](
            order_d.unsafe_ptr().bitcast[Int32](), tv.unsafe_ptr().bitcast[Int32](), Int32(ne),
            grid_dim=_fbr_grid(ne), block_dim=FBR_TPB,
        )
        fast_radix_sort_pairs_u32(ctx, ne, keys_d, order_d, tk, tv, counts)
        ctx.enqueue_function[fbr_gather_edges_kernel](
            Int32(ne), order_d.unsafe_ptr(), es_d.unsafe_ptr(), ed_d.unsafe_ptr(), ew_d.unsafe_ptr(),
            mst_rows.unsafe_ptr(), mst_cols.unsafe_ptr(), mst_weights.unsafe_ptr(),
            Int32(1) if (is_sqrt and not mutual_reach) else Int32(0),
            grid_dim=_fbr_grid(ne), block_dim=FBR_TPB,
        )
        ctx.synchronize()
        _ = order_d^
        _ = tk^
        _ = tv^
        _ = counts^
        _ = keys_d^
    ctx.synchronize()
    _ = mb^
    _ = comp_d^
    _ = todo_d^
    _ = defer_d^
    _ = listb_d^
    _ = bd_d^
    _ = bj_d^
    _ = dropped_d^
    _ = listed_d^
    _ = ub_d^
    _ = minlb_d^
    _ = arg_d^
    _ = cbd_d^
    _ = cba_d^
    _ = cbb_d^
    _ = keep_d^
    _ = scan_d^
    _ = lab_d^
    _ = bsum_d^
    _ = cnt_d^
    _ = es_d^
    _ = ed_d^
    _ = ew_d^
    return rounds
