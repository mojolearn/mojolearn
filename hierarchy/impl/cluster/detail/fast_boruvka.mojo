"""FAST-only Euclidean minimum spanning tree without the dense matrix (Apple).

The PAIRWISE route materialises `m * m` distances and `m * m` indices
(11.6 GB at m = 38,000), which is past what the M4 can allocate, so the
build failed from 38k rows up. Here the MST comes from Boruvka rounds with
the distances computed on the fly:

  * `fb_nearest_other_kernel`: for every point, the nearest point in a
    DIFFERENT component, over the whole data in shared-memory tiles
    (`O(m^2 d)` work, `O(m)` memory), ties to the lower index;
  * the device takes each component's cheapest outgoing edge under the
    total order (squared distance, min(i, j), max(i, j)) -- the per-point
    choice above is consistent with it, so no round can close a cycle --
    and joins the components by hooking and pointer jumping
    (`fast_euclidean_mst`'s banner).

Rounds at least halve the component count. The edges come back sorted by
(weight, lo, hi) for the dendrogram, the weight rooted for
L2SqrtExpanded. FAST arithmetic: direct sums of squared differences.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.math import sqrt
from std.memory import bitcast
from std.os import getenv
from std.time import perf_counter_ns
from std.atomic import Atomic
from hierarchy.impl.cluster.detail.fast_mma_boruvka import MmaBoruvka
from hierarchy.checks.edge_order import WEIGHT_KEY_SENTINEL, weight_order_key
from hdbscan.impl.cluster.detail.sparse_mr_mst import (
    SMR_DEV_TPB, SMR_ERR_BAD, SMR_LIST_A, SMR_LIST_B, SMR_ST_ADDED, SMR_ST_COUNT,
    SMR_ST_ERR, SMR_ST_ERR_ROW, SMR_ST_LEN, _compact, smr_arg_idx_kernel,
    smr_assign_kernel, smr_classify_kernel, smr_cmin_hi_kernel, smr_cmin_key_kernel,
    smr_cmin_lo_kernel, smr_drop_arg_kernel, smr_drop_b_kernel, smr_edge_rank_kernel,
    smr_edge_scatter_kernel, smr_hook_kernel, smr_init_kernel, smr_jump_kernel,
    smr_relabel_kernel, smr_round_reset_kernel, smr_ub_from_a_kernel, smr_winner_kernel,
)

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
    """The nearest other-component point of the `n_todo` listed points
    (`todo_d[0:n_todo]`, on the device) into `bd_d` / `bj_d` (their
    entries only)."""
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


def fb_keys_kernel(
    bd: MutPointer[Float32, MutAnyOrigin],
    bj: MutPointer[Int32, MutAnyOrigin],
    todo: MutPointer[Int32, MutAnyOrigin],
    pk: MutPointer[Int32, MutAnyOrigin],
    pj: MutPointer[Int32, MutAnyOrigin],
    st: MutPointer[Int32, MutAnyOrigin],
    n_todo_in: Int32,
):
    """A searched point's nearest (value, j) as the round's (key, j): the
    value's order key (`weight_order_key`, its bits for a non-negative
    value), the same j. A non-finite value (j == -2) is reported in `st`
    with its row."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= Int(n_todo_in):
        return
    var i = Int(todo[t])
    var j = bj[i]
    if j == Int32(-2):
        _ = Atomic.max(st.unsafe_offset(SMR_ST_ERR), SMR_ERR_BAD)
        _ = Atomic.min(st.unsafe_offset(SMR_ST_ERR_ROW), Int32(i))
        pk[i] = WEIGHT_KEY_SENTINEL
        pj[i] = Int32(-1)
        return
    pk[i] = weight_order_key(bd[i]) if j >= 0 else WEIGHT_KEY_SENTINEL
    pj[i] = j


def fb_sqrt_kernel(w: MutPointer[Float32, MutAnyOrigin], n_in: Int32):
    """The L2SqrtExpanded root of the sorted squared weights."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < Int(n_in):
        w[t] = sqrt(w[t])


def _fb_status(
    ctx: DeviceContext,
    mut st_d: DeviceBuffer[DType.int32],
    mut st_h: HostBuffer[DType.int32],
) raises:
    """Drain, read the status words, refuse a non-finite distance by name."""
    ctx.enqueue_copy(dst_ptr=st_h.unsafe_ptr(), src_buf=st_d)
    ctx.synchronize()
    if st_h.unsafe_ptr().unsafe_load(SMR_ST_ERR) == SMR_ERR_BAD:
        raise Error(
            "hierarchy.pairwise_distances: a distance from row "
            + String(Int(st_h.unsafe_ptr().unsafe_load(SMR_ST_ERR_ROW)))
            + " is NaN or overflows Float32 (a non-finite input"
            " row, or rows whose squared difference overflows);"
            " refused by name (DEVIATION 623, IDENTITY_PATHS row 39)"
        )


def _fb_keys(
    ctx: DeviceContext,
    mut bd_d: DeviceBuffer[DType.float32],
    mut bj_d: DeviceBuffer[DType.int32],
    mut todo_d: DeviceBuffer[DType.int32],
    mut pk_d: DeviceBuffer[DType.int32],
    mut pj_d: DeviceBuffer[DType.int32],
    mut st_d: DeviceBuffer[DType.int32],
    n_todo: Int,
) raises:
    if n_todo <= 0:
        return
    ctx.enqueue_function[fb_keys_kernel](
        bd_d.unsafe_ptr(), bj_d.unsafe_ptr(), todo_d.unsafe_ptr(),
        pk_d.unsafe_ptr(), pj_d.unsafe_ptr(), st_d.unsafe_ptr(), Int32(n_todo),
        grid_dim=((n_todo + SMR_DEV_TPB - 1) // SMR_DEV_TPB, 1, 1),
        block_dim=(SMR_DEV_TPB, 1, 1),
    )


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
    """`m - 1` edges into the three buffers, ascending by the total order
    (weight, lo, hi), oriented (lo, hi). Returns the Boruvka round count.
    `x` is `m x n` row-major, `n <= 64`.

    THE ROUND ON THE DEVICE (lane cgr5-owed, 2026-10-03; the bookkeeping was
    a host pass over the m points per round): the plan (classify, the
    phase-A / phase-B lower-bound pruning, the list compaction), each
    component's cheapest edge (integer atomic minimums over (key, lo, hi)),
    hooking and pointer jumping are `hdbscan/impl/cluster/detail/
    sparse_mr_mst.mojo`'s kernels on this file's search; a status word per
    phase (list sizes, the join count, the non-finite refusal) comes back.
    The tree is the minimum spanning tree under that total order, so the
    edges are the host plan's; the ORDER among equal weights is now (lo,
    hi) where it was discovery order (FAST: no bit promise)."""
    if n > 64:
        raise Error("fast_euclidean_mst: n_cols > 64 is not taken here")
    if m < 2:
        ctx.synchronize()
        return 0
    var g = (m + SMR_DEV_TPB - 1) // SMR_DEV_TPB
    var comp_d = ctx.enqueue_create_buffer[DType.int32](m)
    var bd_d = ctx.enqueue_create_buffer[DType.float32](m)
    var bj_d = ctx.enqueue_create_buffer[DType.int32](m)
    var todo_d = ctx.enqueue_create_buffer[DType.int32](m)
    var pk_d = ctx.enqueue_create_buffer[DType.int32](m)
    var pj_d = ctx.enqueue_create_buffer[DType.int32](m)
    var lb_d = ctx.enqueue_create_buffer[DType.int32](m)
    var state_d = ctx.enqueue_create_buffer[DType.int32](m)
    var ub_d = ctx.enqueue_create_buffer[DType.int32](m)
    var amin_d = ctx.enqueue_create_buffer[DType.int32](m)
    var aidx_d = ctx.enqueue_create_buffer[DType.int32](m)
    var ckey_d = ctx.enqueue_create_buffer[DType.int32](m)
    var clo_d = ctx.enqueue_create_buffer[DType.int32](m)
    var chi_d = ctx.enqueue_create_buffer[DType.int32](m)
    var nxt_d = ctx.enqueue_create_buffer[DType.int32](m)
    var win_d = ctx.enqueue_create_buffer[DType.int32](m)
    var par_a = ctx.enqueue_create_buffer[DType.int32](m)
    var par_b = ctx.enqueue_create_buffer[DType.int32](m)
    var e_key = ctx.enqueue_create_buffer[DType.int32](m)
    var e_lo = ctx.enqueue_create_buffer[DType.int32](m)
    var e_hi = ctx.enqueue_create_buffer[DType.int32](m)
    var rank_d = ctx.enqueue_create_buffer[DType.int32](m)
    var bcount_d = ctx.enqueue_create_buffer[DType.int32](g)
    var boff_d = ctx.enqueue_create_buffer[DType.int32](g)
    var st_d = ctx.enqueue_create_buffer[DType.int32](SMR_ST_LEN)
    var st_h = ctx.enqueue_create_host_buffer[DType.int32](SMR_ST_LEN)
    ctx.enqueue_function[smr_init_kernel](
        comp_d.unsafe_ptr(), pk_d.unsafe_ptr(), pj_d.unsafe_ptr(),
        e_key.unsafe_ptr(), e_lo.unsafe_ptr(), e_hi.unsafe_ptr(),
        st_d.unsafe_ptr(), Int32(m),
        grid_dim=(g, 1, 1), block_dim=(SMR_DEV_TPB, 1, 1),
    )
    var jumps = 1
    while (1 << jumps) < m:
        jumps += 1
    jumps += 1
    var _st_on = getenv("MOJOLEARN_STAGE_TIMES") == "1"
    # FAST, Apple, n <= 32: the matrix-unit search with the same answer
    # (`fast_mma_boruvka.mojo`); `mb.ok` is False when it declines.
    var mb = MmaBoruvka(ctx, x, m, n, mutual_reach, core_ptr, inv_alpha)
    var n_comp = m
    var rounds = 0
    while n_comp > 1:
        rounds += 1
        if rounds > 64:
            raise Error("fast_euclidean_mst: Boruvka did not converge")
        var _tq0 = 0
        if _st_on:
            ctx.synchronize()
            _tq0 = Int(perf_counter_ns())
        ctx.enqueue_function[smr_round_reset_kernel](
            ub_d.unsafe_ptr(), amin_d.unsafe_ptr(), aidx_d.unsafe_ptr(),
            ckey_d.unsafe_ptr(), clo_d.unsafe_ptr(), chi_d.unsafe_ptr(),
            nxt_d.unsafe_ptr(), win_d.unsafe_ptr(), st_d.unsafe_ptr(),
            Int32(m), grid_dim=(g, 1, 1), block_dim=(SMR_DEV_TPB, 1, 1),
        )
        ctx.enqueue_function[smr_classify_kernel](
            comp_d.unsafe_ptr(), pk_d.unsafe_ptr(), pj_d.unsafe_ptr(),
            ub_d.unsafe_ptr(), lb_d.unsafe_ptr(), state_d.unsafe_ptr(),
            Int32(m), grid_dim=(g, 1, 1), block_dim=(SMR_DEV_TPB, 1, 1),
        )
        ctx.enqueue_function[smr_drop_arg_kernel](
            comp_d.unsafe_ptr(), lb_d.unsafe_ptr(), ub_d.unsafe_ptr(),
            state_d.unsafe_ptr(), amin_d.unsafe_ptr(), Int32(m),
            grid_dim=(g, 1, 1), block_dim=(SMR_DEV_TPB, 1, 1),
        )
        ctx.enqueue_function[smr_arg_idx_kernel](
            comp_d.unsafe_ptr(), lb_d.unsafe_ptr(), state_d.unsafe_ptr(),
            amin_d.unsafe_ptr(), aidx_d.unsafe_ptr(), Int32(m),
            grid_dim=(g, 1, 1), block_dim=(SMR_DEV_TPB, 1, 1),
        )
        ctx.enqueue_function[smr_assign_kernel](
            comp_d.unsafe_ptr(), state_d.unsafe_ptr(), aidx_d.unsafe_ptr(),
            Int32(m), grid_dim=(g, 1, 1), block_dim=(SMR_DEV_TPB, 1, 1),
        )
        # phase A: each component's listed point with the smallest bound
        _compact(ctx, state_d, SMR_LIST_A, bcount_d, boff_d, todo_d, st_d, m)
        _fb_status(ctx, st_d, st_h)
        var n_a = Int(st_h.unsafe_ptr().unsafe_load(SMR_ST_COUNT))
        _fb_search(
            ctx, mb, x, m, n, mutual_reach, core_ptr, inv_alpha, comp_d,
            todo_d, n_a, bd_d, bj_d,
        )
        _fb_keys(ctx, bd_d, bj_d, todo_d, pk_d, pj_d, st_d, n_a)
        # phase B: A's exact values tighten the bounds
        ctx.enqueue_function[smr_ub_from_a_kernel](
            comp_d.unsafe_ptr(), pk_d.unsafe_ptr(), pj_d.unsafe_ptr(),
            state_d.unsafe_ptr(), ub_d.unsafe_ptr(), Int32(m),
            grid_dim=(g, 1, 1), block_dim=(SMR_DEV_TPB, 1, 1),
        )
        ctx.enqueue_function[smr_drop_b_kernel](
            comp_d.unsafe_ptr(), lb_d.unsafe_ptr(), ub_d.unsafe_ptr(),
            state_d.unsafe_ptr(), Int32(m),
            grid_dim=(g, 1, 1), block_dim=(SMR_DEV_TPB, 1, 1),
        )
        _compact(ctx, state_d, SMR_LIST_B, bcount_d, boff_d, todo_d, st_d, m)
        _fb_status(ctx, st_d, st_h)
        var n_b = Int(st_h.unsafe_ptr().unsafe_load(SMR_ST_COUNT))
        _fb_search(
            ctx, mb, x, m, n, mutual_reach, core_ptr, inv_alpha, comp_d,
            todo_d, n_b, bd_d, bj_d,
        )
        _fb_keys(ctx, bd_d, bj_d, todo_d, pk_d, pj_d, st_d, n_b)
        # each component's cheapest edge under (key, lo, hi), then the join
        ctx.enqueue_function[smr_cmin_key_kernel](
            comp_d.unsafe_ptr(), pk_d.unsafe_ptr(), pj_d.unsafe_ptr(),
            state_d.unsafe_ptr(), ckey_d.unsafe_ptr(), Int32(m),
            grid_dim=(g, 1, 1), block_dim=(SMR_DEV_TPB, 1, 1),
        )
        ctx.enqueue_function[smr_cmin_lo_kernel](
            comp_d.unsafe_ptr(), pk_d.unsafe_ptr(), pj_d.unsafe_ptr(),
            state_d.unsafe_ptr(), ckey_d.unsafe_ptr(), clo_d.unsafe_ptr(),
            Int32(m), grid_dim=(g, 1, 1), block_dim=(SMR_DEV_TPB, 1, 1),
        )
        ctx.enqueue_function[smr_cmin_hi_kernel](
            comp_d.unsafe_ptr(), pk_d.unsafe_ptr(), pj_d.unsafe_ptr(),
            state_d.unsafe_ptr(), ckey_d.unsafe_ptr(), clo_d.unsafe_ptr(),
            chi_d.unsafe_ptr(), Int32(m),
            grid_dim=(g, 1, 1), block_dim=(SMR_DEV_TPB, 1, 1),
        )
        ctx.enqueue_function[smr_winner_kernel](
            comp_d.unsafe_ptr(), pk_d.unsafe_ptr(), pj_d.unsafe_ptr(),
            state_d.unsafe_ptr(), ckey_d.unsafe_ptr(), clo_d.unsafe_ptr(),
            chi_d.unsafe_ptr(), nxt_d.unsafe_ptr(), win_d.unsafe_ptr(),
            Int32(m), grid_dim=(g, 1, 1), block_dim=(SMR_DEV_TPB, 1, 1),
        )
        ctx.enqueue_function[smr_hook_kernel](
            comp_d.unsafe_ptr(), pk_d.unsafe_ptr(), pj_d.unsafe_ptr(),
            nxt_d.unsafe_ptr(), win_d.unsafe_ptr(), par_a.unsafe_ptr(),
            e_key.unsafe_ptr(), e_lo.unsafe_ptr(), e_hi.unsafe_ptr(),
            st_d.unsafe_ptr(), Int32(m),
            grid_dim=(g, 1, 1), block_dim=(SMR_DEV_TPB, 1, 1),
        )
        for q in range(jumps):
            if q % 2 == 0:
                ctx.enqueue_function[smr_jump_kernel](
                    comp_d.unsafe_ptr(), par_a.unsafe_ptr(),
                    par_b.unsafe_ptr(), Int32(m),
                    grid_dim=(g, 1, 1), block_dim=(SMR_DEV_TPB, 1, 1),
                )
            else:
                ctx.enqueue_function[smr_jump_kernel](
                    comp_d.unsafe_ptr(), par_b.unsafe_ptr(),
                    par_a.unsafe_ptr(), Int32(m),
                    grid_dim=(g, 1, 1), block_dim=(SMR_DEV_TPB, 1, 1),
                )
        if jumps % 2 == 1:
            ctx.enqueue_function[smr_relabel_kernel](
                comp_d.unsafe_ptr(), par_b.unsafe_ptr(), Int32(m),
                grid_dim=(g, 1, 1), block_dim=(SMR_DEV_TPB, 1, 1),
            )
        else:
            ctx.enqueue_function[smr_relabel_kernel](
                comp_d.unsafe_ptr(), par_a.unsafe_ptr(), Int32(m),
                grid_dim=(g, 1, 1), block_dim=(SMR_DEV_TPB, 1, 1),
            )
        _fb_status(ctx, st_d, st_h)
        var added = Int(st_h.unsafe_ptr().unsafe_load(SMR_ST_ADDED))
        if _st_on:
            print("BORUVKA round=" + String(rounds) + " phaseA=" + String(n_a)
                  + " phaseB=" + String(n_b) + " joined=" + String(added)
                  + " ms=" + String((Int(perf_counter_ns()) - _tq0) // 1000000))
        if added == 0:
            raise Error(
                "fast_euclidean_mst: a Boruvka round joined nothing with "
                + String(n_comp) + " components left"
            )
        n_comp -= added
    # the m - 1 recorded slots (the one empty slot sorts last) by rank
    ctx.enqueue_function[smr_edge_rank_kernel](
        e_key.unsafe_ptr(), e_lo.unsafe_ptr(), e_hi.unsafe_ptr(),
        rank_d.unsafe_ptr(), Int32(m),
        grid_dim=(g, 1, 1), block_dim=(SMR_DEV_TPB, 1, 1),
    )
    ctx.enqueue_function[smr_edge_scatter_kernel](
        e_key.unsafe_ptr(), e_lo.unsafe_ptr(), e_hi.unsafe_ptr(),
        rank_d.unsafe_ptr(), mst_rows.unsafe_ptr(), mst_cols.unsafe_ptr(),
        mst_weights.unsafe_ptr(), Int32(m),
        grid_dim=(g, 1, 1), block_dim=(SMR_DEV_TPB, 1, 1),
    )
    if is_sqrt and not mutual_reach:
        ctx.enqueue_function[fb_sqrt_kernel](
            mst_weights.unsafe_ptr(), Int32(m - 1),
            grid_dim=(g, 1, 1), block_dim=(SMR_DEV_TPB, 1, 1),
        )
    ctx.synchronize()
    _ = mb^
    _ = comp_d^
    _ = bd_d^
    _ = bj_d^
    _ = todo_d^
    _ = pk_d^
    _ = pj_d^
    _ = lb_d^
    _ = state_d^
    _ = ub_d^
    _ = amin_d^
    _ = aidx_d^
    _ = ckey_d^
    _ = clo_d^
    _ = chi_d^
    _ = nxt_d^
    _ = win_d^
    _ = par_a^
    _ = par_b^
    _ = e_key^
    _ = e_lo^
    _ = e_hi^
    _ = rank_d^
    _ = bcount_d^
    _ = boff_d^
    _ = st_d^
    _ = st_h^
    return rounds
