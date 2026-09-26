"""FAST-only Euclidean minimum spanning tree without the dense matrix (Apple).

The PAIRWISE route materialises `m * m` distances and `m * m` indices
(11.6 GB at m = 38,000), which is past what the M4 can allocate, so the
build failed from 38k rows up. Here the MST comes from Boruvka rounds with
the distances computed on the fly:

  * `fb_nearest_other_kernel`: for every point, the nearest point in a
    DIFFERENT component, over the whole data in shared-memory tiles
    (`O(m^2 d)` work, `O(m)` memory), ties to the lower index;
  * the host takes each component's cheapest outgoing edge under the total
    order (squared distance, min(i, j), max(i, j)) -- the per-point choice
    above is consistent with it, so no round can close a cycle -- joins the
    components with a union-find and relabels the points.

Rounds at least halve the component count. The edges come back sorted by
(weight, src, dst) for the dendrogram, the weight rooted for
L2SqrtExpanded. FAST arithmetic: direct sums of squared differences.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.math import sqrt
from std.memory import bitcast
from std.builtin.sort import sort
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


def _find(mut parent: List[Int32], a: Int) -> Int:
    var r = a
    while Int(parent[r]) != r:
        r = Int(parent[r])
    var c = a
    while Int(parent[c]) != r:
        var nx = Int(parent[c])
        parent[c] = Int32(r)
        c = nx
    return r


@always_inline
def _edge_less(
    d1: Float32, a1: Int, b1: Int, d2: Float32, a2: Int, b2: Int
) -> Bool:
    if d1 != d2:
        return d1 < d2
    if a1 != a2:
        return a1 < a2
    return b1 < b2


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
    `n <= 64`."""
    if n > 64:
        raise Error("fast_euclidean_mst: n_cols > 64 is not taken here")
    var parent = List[Int32](capacity=m)
    var comp_h = ctx.enqueue_create_host_buffer[DType.int32](m)
    for i in range(m):
        parent.append(Int32(i))
        comp_h.unsafe_ptr().unsafe_store(i, Int32(i))
    var comp_d = ctx.enqueue_create_buffer[DType.int32](m)
    var bd_d = ctx.enqueue_create_buffer[DType.float32](m)
    var bj_d = ctx.enqueue_create_buffer[DType.int32](m)
    var bd_h = ctx.enqueue_create_host_buffer[DType.float32](m)
    var bj_h = ctx.enqueue_create_host_buffer[DType.int32](m)
    var es = List[Int32]()
    var ed = List[Int32]()
    var ew = List[Float32]()
    # Per-component best edge (indexed by root id).
    var cb_d = List[Float32](length=m, fill=Float32.MAX)
    var cb_a = List[Int32](length=m, fill=Int32(-1))
    var cb_b = List[Int32](length=m, fill=Int32(-1))
    var n_comp = m
    var rounds = 0
    # Only points whose nearest other-component point joined their own
    # component are searched again: components only merge, so a surviving
    # nearest (lowest index on a tie) is still the nearest.
    var todo_h = ctx.enqueue_create_host_buffer[DType.int32](m)
    var todo_d = ctx.enqueue_create_buffer[DType.int32](m)
    for i in range(m):
        todo_h.unsafe_ptr().unsafe_store(i, Int32(i))
    var n_todo = m
    # FAST, Apple, n <= 32: the matrix-unit search with the same answer
    # (`fast_mma_boruvka.mojo`); `mb.ok` is False when it declines.
    var _st_on = getenv("MOJOLEARN_STAGE_TIMES") == "1"
    var mb = MmaBoruvka(ctx, x, m, n, mutual_reach, core_ptr, inv_alpha)
    while n_comp > 1:
        rounds += 1
        var grid = (n_todo + FB_TPB - 1) // FB_TPB
        ctx.enqueue_copy(dst_buf=comp_d, src_ptr=comp_h.unsafe_ptr())
        if n_todo > 0:
            ctx.enqueue_copy(
                dst_buf=todo_d.create_sub_buffer[DType.int32](0, n_todo),
                src_ptr=todo_h.unsafe_ptr(),
            )
        var _tq0 = 0
        if _st_on:
            ctx.synchronize()
            _tq0 = Int(perf_counter_ns())
        if mb.ok and n_todo > 0:
            mb.enqueue(
                ctx, x, m, n, mutual_reach, core_ptr, inv_alpha, comp_d,
                todo_d, n_todo, bd_d, bj_d,
            )
        comptime for DM in [8, 16, 32, 64]:
            # The reachability arm keeps one point per thread (its per-pair
            # sqrt, not the tile reads, dominates: HDBSCAN taxi 100k 9.6 s
            # at 1 vs 10.2 s at 4).
            comptime P = FB_PPT if DM <= 16 else 1
            var gridp = (n_todo + FB_TPB * P - 1) // (FB_TPB * P)
            if mutual_reach:
                gridp = grid
            if not mb.ok and n_todo > 0 and n <= DM and (DM == 8 or n > DM // 2):
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
        if _st_on:
            ctx.synchronize()
            print("BORUVKA round=" + String(rounds) + " n_todo=" + String(n_todo)
                  + " ms=" + String((Int(perf_counter_ns()) - _tq0) // 1000000))
        ctx.enqueue_copy(dst_ptr=bd_h.unsafe_ptr(), src_buf=bd_d)
        ctx.enqueue_copy(dst_ptr=bj_h.unsafe_ptr(), src_buf=bj_d)
        ctx.synchronize()
        for i in range(m):
            var j = Int(bj_h.unsafe_ptr().unsafe_load(i))
            if j == -2:
                raise Error(
                    "hierarchy.pairwise_distances: a distance from row "
                    + String(i)
                    + " is NaN or overflows Float32 (a non-finite input"
                    " row, or rows whose squared difference overflows);"
                    " refused by name (DEVIATION 623, IDENTITY_PATHS row 39)"
                )
            if j < 0:
                continue
            var c = Int(comp_h.unsafe_ptr().unsafe_load(i))
            var dd = bd_h.unsafe_ptr().unsafe_load(i)
            var a = min(i, j)
            var b = max(i, j)
            if cb_a[c] < 0 or _edge_less(
                dd, a, b, cb_d[c], Int(cb_a[c]), Int(cb_b[c])
            ):
                cb_d[c] = dd
                cb_a[c] = Int32(a)
                cb_b[c] = Int32(b)
        for c in range(m):
            if cb_a[c] < 0:
                continue
            var a = Int(cb_a[c])
            var b = Int(cb_b[c])
            var ra = _find(parent, a)
            var rb = _find(parent, b)
            if ra != rb:
                parent[max(ra, rb)] = Int32(min(ra, rb))
                es.append(Int32(a))
                ed.append(Int32(b))
                ew.append(cb_d[c])
                n_comp -= 1
            cb_d[c] = Float32.MAX
            cb_a[c] = Int32(-1)
            cb_b[c] = Int32(-1)
        n_todo = 0
        for i in range(m):
            var ri = _find(parent, i)
            comp_h.unsafe_ptr().unsafe_store(i, Int32(ri))
            var bj = Int(bj_h.unsafe_ptr().unsafe_load(i))
            if bj < 0 or _find(parent, bj) == ri:
                todo_h.unsafe_ptr().unsafe_store(n_todo, Int32(i))
                n_todo += 1
        if rounds > 64:
            raise Error("fast_euclidean_mst: Boruvka did not converge")
    # Sort the edges by weight (non-negative float bits order as the
    # floats do), ties by the order Boruvka found them.
    var ne = len(es)
    var keys = List[UInt64](capacity=ne)
    for e in range(ne):
        var wb = UInt64(Int(bitcast[DType.uint32](ew[e])))
        keys.append((wb << 32) | UInt64(e))
    sort(keys)
    var order = List[Int](capacity=ne)
    for t in range(ne):
        order.append(Int(keys[t] & 0xFFFFFFFF))
    var hr = ctx.enqueue_create_host_buffer[DType.int32](max(ne, 1))
    var hc = ctx.enqueue_create_host_buffer[DType.int32](max(ne, 1))
    var hw = ctx.enqueue_create_host_buffer[DType.float32](max(ne, 1))
    for t in range(ne):
        var e = order[t]
        hr.unsafe_ptr().unsafe_store(t, es[e])
        hc.unsafe_ptr().unsafe_store(t, ed[e])
        var w = ew[e]
        if is_sqrt and not mutual_reach:
            w = sqrt(w)
        hw.unsafe_ptr().unsafe_store(t, w)
    if ne > 0:
        ctx.enqueue_copy(
            dst_buf=mst_rows.create_sub_buffer[DType.int32](0, ne),
            src_ptr=hr.unsafe_ptr(),
        )
        ctx.enqueue_copy(
            dst_buf=mst_cols.create_sub_buffer[DType.int32](0, ne),
            src_ptr=hc.unsafe_ptr(),
        )
        ctx.enqueue_copy(
            dst_buf=mst_weights.create_sub_buffer[DType.float32](0, ne),
            src_ptr=hw.unsafe_ptr(),
        )
    ctx.synchronize()
    _ = mb^
    _ = todo_h^
    _ = todo_d^
    _ = comp_d^
    _ = bd_d^
    _ = bj_d^
    _ = comp_h^
    _ = bd_h^
    _ = bj_h^
    _ = hr^
    _ = hc^
    _ = hw^
    return rounds
