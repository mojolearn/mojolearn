# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The d <= 64 FAST arm's mutual reachability MST with the rounds on the device
(lane af-hdbscan2, FAST on Apple only, `-D MOJOLEARN_HDB_DEV_BORUVKA`).

Main's FAST Apple route for d <= 64 is `hierarchy/.../fast_boruvka.mojo::
fast_euclidean_mst`: the search runs on the device (the matrix-unit
`MmaBoruvka`, or `fb_nearest_other_kernel`), but every round uploads the
component labels, reads back two m-word arrays (twice when points were
deferred), waits twice, and runs about eight host passes over m with a host
union-find; the edges are sorted on the host at the end and uploaded.

Here the SAME search kernels run (the same arithmetic: the direct `(x - y)^2`
sum, then `max(core_j, max(core_i, inv_alpha * sqrt(d)))`, ties to the lower
index), and everything around them is `sparse_mr_mst.mojo`'s device round:
classify / drop / phase A pick / compact (the exact pruning plan of the host
code), each component's cheapest edge under (key, lo, hi) by integer
minimums, hooking, pointer jumping, relabel; then rank + scatter emits the
m - 1 edges sorted by (weight key, lo, hi) and oriented (lo, hi). Per round
three status words come back (the two listed counts and the join count); no
m-word readback, no upload, no host loop.

THE EDGE SET is the host route's: Boruvka under one strict total order on the
edges has one answer. THE EDGE ORDER among exactly equal weights is (lo, hi)
here, the dense and sparse arms' order, where the host route emitted
discovery order; so where main's FAST taxi list differed from IDENTICAL's on
a plateau, this one agrees with IDENTICAL's. Cluster numbering on such
plateaus can therefore move; the partition cannot (hdbscan2.md).

A cut launch (macOS aborts long command buffers silently) is caught the way
the sparse arm catches it: `best_j` is poisoned before every search and a
surviving poison is refused by name.
"""

from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from hdbscan.impl.cluster.detail.sparse_mr_mst import (
    SMR_BAD_J,
    SMR_DEV_TPB,
    SMR_LIST_A,
    SMR_LIST_B,
    SMR_NONE_J,
    SMR_POISON_J,
    SMR_ST_ADDED,
    SMR_ST_COUNT,
    SMR_ST_LEN,
    _compact,
    _grid,
    _read_status,
    smr_arg_idx_kernel,
    smr_assign_kernel,
    smr_classify_kernel,
    smr_cmin_hi_kernel,
    smr_cmin_key_kernel,
    smr_cmin_lo_kernel,
    smr_drop_arg_kernel,
    smr_drop_b_kernel,
    smr_edge_rank_kernel,
    smr_edge_scatter_kernel,
    smr_hook_kernel,
    smr_init_kernel,
    smr_jump_kernel,
    smr_merge_kernel,
    smr_relabel_kernel,
    smr_round_reset_kernel,
    smr_ub_from_a_kernel,
    smr_winner_kernel,
)
from hierarchy.checks.edge_order import WEIGHT_KEY_SENTINEL, weight_order_key
from hierarchy.impl.cluster.detail.fast_boruvka import (
    FB_TPB,
    fb_nearest_other_kernel,
)
from hierarchy.impl.cluster.detail.fast_mma_boruvka import MmaBoruvka


comptime _I32P = MutPointer[Int32, MutAnyOrigin]
comptime _F32P = MutPointer[Float32, MutAnyOrigin]


def hdb_fast_fold_kernel(
    fk: _I32P, fj: _I32P, bd: _F32P, bj: _I32P, todo: _I32P, n_todo_in: Int32
):
    """Listed point t's (best weight, best j) from the search's per-point
    slots into the sparse round's (key, j) form: -1 = no other component,
    -2 = a non-finite weight, -3 = the poison (the cell was never written)."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= Int(n_todo_in):
        return
    var i = Int(todo[t])
    var j = bj[i]
    if j == Int32(-3):
        fk[t] = WEIGHT_KEY_SENTINEL
        fj[t] = SMR_POISON_J
        return
    if j == Int32(-2):
        fk[t] = WEIGHT_KEY_SENTINEL
        fj[t] = SMR_BAD_J
        return
    if j < 0:
        fk[t] = WEIGHT_KEY_SENTINEL
        fj[t] = SMR_NONE_J
        return
    fk[t] = weight_order_key(bd[i])
    fj[t] = j


def _fast_search(
    ctx: DeviceContext,
    mut mb: MmaBoruvka,
    mut x: DeviceBuffer[DType.float32],
    m: Int,
    d: Int,
    core_ptr: MutPointer[Float32, MutAnyOrigin],
    inv_alpha: Float32,
    mut comp_d: DeviceBuffer[DType.int32],
    mut todo_d: DeviceBuffer[DType.int32],
    n_todo: Int,
    mut bd_d: DeviceBuffer[DType.float32],
    mut bj_d: DeviceBuffer[DType.int32],
    mut fk_d: DeviceBuffer[DType.int32],
    mut fj_d: DeviceBuffer[DType.int32],
    mut pk_d: DeviceBuffer[DType.int32],
    mut pj_d: DeviceBuffer[DType.int32],
    mut st_d: DeviceBuffer[DType.int32],
) raises:
    """The cheapest other-component edge of the `n_todo` listed points
    (`todo_d[0:n_todo]`, on the device) merged into `pk_d` / `pj_d`."""
    if n_todo <= 0:
        return
    ctx.enqueue_memset(bj_d, Int32(-3))
    if mb.ok:
        mb.enqueue(
            ctx, x, m, d, True, core_ptr, inv_alpha, comp_d, todo_d, n_todo,
            bd_d, bj_d,
        )
    else:
        var grid = (n_todo + FB_TPB - 1) // FB_TPB
        comptime for DM in [8, 16, 32, 64]:
            if d <= DM and (DM == 8 or d > DM // 2):
                ctx.enqueue_function[fb_nearest_other_kernel[DM, True, 1]](
                    x.unsafe_ptr(), core_ptr, inv_alpha, comp_d.unsafe_ptr(),
                    bd_d.unsafe_ptr(), bj_d.unsafe_ptr(), Int32(m), Int32(d),
                    todo_d.unsafe_ptr(), Int32(n_todo),
                    grid_dim=(grid, 1, 1), block_dim=(FB_TPB, 1, 1),
                )
    var tgrid = (n_todo + SMR_DEV_TPB - 1) // SMR_DEV_TPB
    ctx.enqueue_function[hdb_fast_fold_kernel](
        fk_d.unsafe_ptr(), fj_d.unsafe_ptr(), bd_d.unsafe_ptr(),
        bj_d.unsafe_ptr(), todo_d.unsafe_ptr(), Int32(n_todo),
        grid_dim=(tgrid, 1, 1), block_dim=(SMR_DEV_TPB, 1, 1),
    )
    ctx.enqueue_function[smr_merge_kernel](
        pk_d.unsafe_ptr(), pj_d.unsafe_ptr(), fk_d.unsafe_ptr(),
        fj_d.unsafe_ptr(), todo_d.unsafe_ptr(), st_d.unsafe_ptr(),
        Int32(n_todo), Int32(1),
        grid_dim=(tgrid, 1, 1), block_dim=(SMR_DEV_TPB, 1, 1),
    )


def fast_mr_mst_device(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    mut core_d: DeviceBuffer[DType.float32],
    m: Int,
    d: Int,
    inv_alpha: Float32,
    mut mst_rows: DeviceBuffer[DType.int32],
    mut mst_cols: DeviceBuffer[DType.int32],
    mut mst_weights: DeviceBuffer[DType.float32],
) raises -> Int:
    """`fast_euclidean_mst(..., mutual_reach=True)`'s tree with the rounds
    on the device: the m - 1 edges into `mst_rows` / `mst_cols` /
    `mst_weights`, sorted by (weight key, lo, hi), oriented (lo, hi).
    Returns the round count the dense solver would report."""
    if m < 2:
        raise Error("hdbscan.fast_mr_mst_device: m=" + String(m) + " < 2")
    if d > 64:
        raise Error(
            "hdbscan.fast_mr_mst_device: n_cols=" + String(d)
            + " > 64 is not taken here (the register tile); the caller"
            " routes wider rows to the sparse arm"
        )
    var core_ptr = core_d.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var mb = MmaBoruvka(ctx, x, m, d, True, core_ptr, inv_alpha)
    var todo_d = ctx.enqueue_create_buffer[DType.int32](m)
    var fk_d = ctx.enqueue_create_buffer[DType.int32](m)
    var fj_d = ctx.enqueue_create_buffer[DType.int32](m)
    var bd_d = ctx.enqueue_create_buffer[DType.float32](m)
    var bj_d = ctx.enqueue_create_buffer[DType.int32](m)
    var comp_d = ctx.enqueue_create_buffer[DType.int32](m)
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
    var g = _grid(m)
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

    var n_comp = m
    var merge_rounds = 0
    while n_comp > 1:
        merge_rounds += 1
        if merge_rounds > 64:
            raise Error("hdbscan.fast_mr_mst_device: Boruvka did not converge")
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
        _compact(ctx, state_d, SMR_LIST_A, bcount_d, boff_d, todo_d, st_d, m)
        _read_status(ctx, st_d, st_h)
        var n_a = Int(st_h.unsafe_ptr().unsafe_load(SMR_ST_COUNT))
        _fast_search(
            ctx, mb, x, m, d, core_ptr, inv_alpha, comp_d, todo_d, n_a,
            bd_d, bj_d, fk_d, fj_d, pk_d, pj_d, st_d,
        )
        # Phase B: A's exact keys tighten the bounds.
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
        _read_status(ctx, st_d, st_h)
        var n_b = Int(st_h.unsafe_ptr().unsafe_load(SMR_ST_COUNT))
        _fast_search(
            ctx, mb, x, m, d, core_ptr, inv_alpha, comp_d, todo_d, n_b,
            bd_d, bj_d, fk_d, fj_d, pk_d, pj_d, st_d,
        )
        # Each component's cheapest edge under (key, lo, hi).
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
        _read_status(ctx, st_d, st_h)
        var added = Int(st_h.unsafe_ptr().unsafe_load(SMR_ST_ADDED))
        if added == 0:
            raise Error(
                "hdbscan.fast_mr_mst_device: a Boruvka round joined nothing"
                " with " + String(n_comp) + " components left"
            )
        n_comp -= added

    # The m - 1 recorded slots (plus the one empty slot, last) by rank.
    var rank_d = ctx.enqueue_create_buffer[DType.int32](m)
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
    ctx.synchronize()
    _ = mb^
    _ = todo_d^
    _ = fk_d^
    _ = fj_d^
    _ = bd_d^
    _ = bj_d^
    _ = comp_d^
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
    _ = bcount_d^
    _ = boff_d^
    _ = st_d^
    _ = st_h^
    _ = rank_d^
    # The dense solver's count: the merge rounds plus its final round.
    return merge_rounds + 1
