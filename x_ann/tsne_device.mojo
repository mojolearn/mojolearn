# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""t-SNE on the device: one thread per cell of `x_ann/tsne_core.mojo`; the
symmetrization runs on the device too (`x_ann/tsne_sym_device.mojo`, the
host function `tsne_symmetrize`'s CSR and words)."""

from std.gpu import block_idx, block_dim, thread_idx
from std.memory import stack_allocation
from std.time import perf_counter_ns
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from max.gpu.host import DeviceBuffer, DeviceContext
from std.sys.compile import is_defined
from std.atomic import Atomic, Ordering, fence
from std.gpu import grid_dim
from checks.kernel_matrix import TARGET_COLUMN, COLUMN_NVIDIA, COLUMN_AMD
from x_ann.device_ctx import x_ann_ctx
from core.device_fold import device_sum_f32_fixed
from x_ann.tsne_sym_device import tsne_symmetrize_device
from x_ann.stage_timer import AnnStages
from x_ann.knn_device import knn_enqueue

from metrics.checks.device_io import upload_f32, upload_i32, download_f32, download_i32
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, ftz, identical_log
from std.sys.info import has_apple_gpu_accelerator
from x_ann.switches import ANN3_TSNE_RB32, ANN3_TSNE_RB64, ANN3_TSNE_STEP_ROWS
from checks.numerics import identical_div, identical_mul
from x_ann.tsne_core import ts_q
from x_ann.tsne_core import (
    F32P, I32P, ts_kl_cell, ts_perplexity_cell, ts_repulse_fold, ts_repulse_pair, ts_repulse_terms, ts_step_cell, ts_z_add,
    ts_sum_cell, tsne_nn, tsne_symmetrize, tsne_validate,
)
from x_ann.tsne_core import TS_LANE_FOLD, TS_LANES

comptime TPB = 128


def _tid() -> Int:
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


def _grid(count: Int) -> Int:
    return (count + TPB - 1) // TPB


def perplexity_kernel(n: Int32, nn_d: F32P, nn: Int32, log_perp: Float32, p: F32P):
    var i = _tid()
    if i < Int(n):
        ts_perplexity_cell(i, nn_d, Int(nn), log_perp, p)


comptime _FAST_APPLE = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
#: rows per threadgroup and staged candidate rows per tile of the repulsion
#: (lane ann-apple3, OPT-IN trials under FAST on Apple: 32 or 64 rows per
#: threadgroup, x_ann/switches.mojo)
comptime RTB = 32 if (_FAST_APPLE and ANN3_TSNE_RB32) else (64 if (_FAST_APPLE and ANN3_TSNE_RB64) else 128)
comptime RTJ = 256


def repulse_tiled_kernel(n: Int32, y: F32P, row_z: F32P, rep: F32P):
    """`ts_repulse_cell` with the candidate rows staged in threadgroup memory
    (lane ann-apple): tiles and rows ascending, so row i folds j = 0, 1, ...,
    n - 1 in the cell's order, each j through the cell's own
    `ts_repulse_pair` on ftz(y) (the staged values are flushed once; ftz is
    idempotent). The same bits."""
    var t = Int(thread_idx.x)
    var nr = Int(n)
    var i = Int(block_idx.x) * RTB + t
    var tile = stack_allocation[2 * RTJ, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var live = i < nr
    var y0 = Float32(0.0)
    var y1 = Float32(0.0)
    if live:
        y0 = ftz(y.unsafe_load(2 * i))
        y1 = ftz(y.unsafe_load(2 * i + 1))
    var z = Float32(0.0)
    var r0 = Float32(0.0)
    var r1 = Float32(0.0)
    # TS_LANE_FOLD (x_ann/tsne_core.mojo): lane s = j mod TS_LANES; a tile
    # starts at a multiple of RTJ, itself a multiple of TS_LANES
    comptime assert RTJ % TS_LANES == 0, "the repulsion tile must hold whole lane groups"
    var lz = SIMD[DType.float32, TS_LANES](0.0)
    var l0 = SIMD[DType.float32, TS_LANES](0.0)
    var l1 = SIMD[DType.float32, TS_LANES](0.0)
    var j0 = 0
    while j0 < nr:
        for e in range(t, 2 * RTJ, RTB):
            var v = Float32(0.0)
            if 2 * j0 + e < 2 * nr:
                v = ftz(y.unsafe_load(2 * j0 + e))
            tile[e] = v
        barrier()
        if live:
            var jn = RTJ if nr - j0 > RTJ else nr - j0
            comptime if TS_LANE_FOLD:
                for g in range(RTJ // TS_LANES):
                    comptime for s in range(TS_LANES):
                        var rl = g * TS_LANES + s
                        if rl < jn and j0 + rl != i:
                            var tm = ts_repulse_terms(y0, y1, tile[2 * rl], tile[2 * rl + 1])
                            var zz = lz[s]
                            var aa = l0[s]
                            var bb = l1[s]
                            ts_repulse_fold(tm, zz, aa, bb)
                            lz[s] = zz
                            l0[s] = aa
                            l1[s] = bb
            else:
                # four j at a time: their terms are independent, so they are
                # formed first and folded after in ascending j (the same
                # statements in the same fold order; only the independent work
                # overlaps). The group holding row i itself takes the plain loop.
                var r = 0
                while r + 4 <= jn:
                    if i >= j0 + r and i < j0 + r + 4:
                        for u in range(4):
                            if j0 + r + u != i:
                                ts_repulse_pair(y0, y1, tile[2 * (r + u)], tile[2 * (r + u) + 1], z, r0, r1)
                    else:
                        var ta = ts_repulse_terms(y0, y1, tile[2 * r], tile[2 * r + 1])
                        var tb = ts_repulse_terms(y0, y1, tile[2 * r + 2], tile[2 * r + 3])
                        var tc = ts_repulse_terms(y0, y1, tile[2 * r + 4], tile[2 * r + 5])
                        var td = ts_repulse_terms(y0, y1, tile[2 * r + 6], tile[2 * r + 7])
                        ts_repulse_fold(ta, z, r0, r1)
                        ts_repulse_fold(tb, z, r0, r1)
                        ts_repulse_fold(tc, z, r0, r1)
                        ts_repulse_fold(td, z, r0, r1)
                    r += 4
                while r < jn:
                    if j0 + r != i:
                        ts_repulse_pair(y0, y1, tile[2 * r], tile[2 * r + 1], z, r0, r1)
                    r += 1
        barrier()
        j0 += RTJ
    comptime if TS_LANE_FOLD:
        # the lane partials, ascending s, each add flushed
        comptime for s in range(TS_LANES):
            z = ftz(z + lz[s])
            r0 = ftz(r0 + l0[s])
            r1 = ftz(r1 + l1[s])
    if live:
        row_z.unsafe_store(i, z)
        rep.unsafe_store(2 * i, r0)
        rep.unsafe_store(2 * i + 1, r1)


comptime TS_STEP_ROWS = _FAST_APPLE and ANN3_TSNE_STEP_ROWS


def z_level_kernel(ln: Int32, src: F32P, dst: F32P):
    """One level of Z's pinned pairwise tree (`ts_sum_cell`): node q =
    ftz(c[2q] + c[2q + 1]), an odd last node carried up unchanged."""
    var q = _tid()
    var l = Int(ln)
    if q < (l + 1) // 2:
        if 2 * q + 1 < l:
            dst.unsafe_store(q, ts_z_add(src.unsafe_load(2 * q), src.unsafe_load(2 * q + 1)))
        else:
            dst.unsafe_store(q, src.unsafe_load(2 * q))


def z_root_kernel(ln: Int32, src: F32P, z: F32P):
    """z = ftz(root) (0 for no rows)."""
    if _tid() == 0:
        z.unsafe_store(0, ftz(src.unsafe_load(0)) if Int(ln) > 0 else Float32(0.0))


def _ts_z(
    ctx: DeviceContext, n: Int, mut drz: DeviceBuffer[DType.float32], mut dz: DeviceBuffer[DType.float32],
    mut dzs: DeviceBuffer[DType.float32],
) raises:
    """Z = `ts_sum_cell`'s pinned pairwise tree over row_z, one parallel
    launch per level (ping-pong between drz and dzs; row_z is not read
    after Z), then the root. cpu-gpu-cleanup c-ann (2026-10-02): replaces
    the one-thread chain (`sum_kernel`) and Apple's opt-in one-threadgroup
    sum; the same nodes as the host column's stack form."""
    var src: F32P = drz.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var dst: F32P = dzs.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var ln = n
    while ln > 1:
        var h = (ln + 1) // 2
        ctx.enqueue_function[z_level_kernel](Int32(ln), src, dst, grid_dim=_grid(h), block_dim=TPB)
        var t = src
        src = dst
        dst = t
        ln = h
    ctx.enqueue_function[z_root_kernel](Int32(ln), src, dz.unsafe_ptr(), grid_dim=1, block_dim=1)


# lane/gap-nv-classical2: the repulsion and Z on NVIDIA and AMD. The tiled
# kernel above runs one row per thread, 20k threads for the whole GPU, each
# forming every pair's terms (a correctly rounded divide) in its own serial
# loop. `repulse_split_kernel` forms the terms of RS_ROWS rows x RS_TJ
# candidates with every thread of the block (`ts_repulse_terms`, the same
# words) into threadgroup memory, then row i's owner folds them with
#: lane idn-regress (2026-10-05): TS_LANE_FOLD on the tiled route (Apple)
#: kept all TS_LANES lanes of three sums in one thread, 192 live floats, and
#: M3 t-SNE ran 2.2-2.6x slower than the single chain (idn5 board). Here a
#: row's lanes are split over RLQ threads (thread q owns lanes q * RLS ..
#: q * RLS + RLS - 1, each folded j ascending with the same statements),
#: the lane partials meet in threadgroup memory and the row's owner adds
#: them in ascending s from +0, each add flushed: the same lanes, the same
#: order, the same bits as `repulse_tiled_kernel`. Launch shape only.
#: -D MOJOLEARN_IDN_TSNE_LANES_SPLIT_OFF restores the one-thread rows.
comptime TS_LANES_SPLIT = TS_LANE_FOLD and not is_defined["MOJOLEARN_IDN_TSNE_LANES_SPLIT_OFF"]()
comptime RLQ = 4
comptime RLS = TS_LANES // RLQ
comptime RLB = 32
comptime RL_TPB = RLB * RLQ


def repulse_lanes_kernel(n: Int32, y: F32P, row_z: F32P, rep: F32P):
    """`repulse_tiled_kernel` under TS_LANE_FOLD with RLQ threads a row
    (see TS_LANES_SPLIT): the same words."""
    comptime assert TS_LANES % RLQ == 0, "a row's lanes split evenly"
    comptime assert RTJ % TS_LANES == 0, "the repulsion tile must hold whole lane groups"
    var t = Int(thread_idx.x)
    var nr = Int(n)
    var rr = t // RLQ
    var q = t % RLQ
    var i = Int(block_idx.x) * RLB + rr
    var tile = stack_allocation[2 * RTJ, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var part = stack_allocation[3 * RLB * TS_LANES, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var live = i < nr
    var y0 = Float32(0.0)
    var y1 = Float32(0.0)
    if live:
        y0 = ftz(y.unsafe_load(2 * i))
        y1 = ftz(y.unsafe_load(2 * i + 1))
    var lz = SIMD[DType.float32, RLS](0.0)
    var l0 = SIMD[DType.float32, RLS](0.0)
    var l1 = SIMD[DType.float32, RLS](0.0)
    var j0 = 0
    while j0 < nr:
        for e in range(t, 2 * RTJ, RL_TPB):
            var v = Float32(0.0)
            if 2 * j0 + e < 2 * nr:
                v = ftz(y.unsafe_load(2 * j0 + e))
            tile[e] = v
        barrier()
        if live:
            var jn = RTJ if nr - j0 > RTJ else nr - j0
            for g in range(RTJ // TS_LANES):
                comptime for u in range(RLS):
                    var rl = g * TS_LANES + q * RLS + u
                    if rl < jn and j0 + rl != i:
                        var tm = ts_repulse_terms(y0, y1, tile[2 * rl], tile[2 * rl + 1])
                        var zz = lz[u]
                        var aa = l0[u]
                        var bb = l1[u]
                        ts_repulse_fold(tm, zz, aa, bb)
                        lz[u] = zz
                        l0[u] = aa
                        l1[u] = bb
        barrier()
        j0 += RTJ
    comptime for u in range(RLS):
        var s = q * RLS + u
        part[rr * TS_LANES + s] = lz[u]
        part[(RLB + rr) * TS_LANES + s] = l0[u]
        part[(2 * RLB + rr) * TS_LANES + s] = l1[u]
    barrier()
    if live and q == 0:
        # the lane partials, ascending s, each add flushed
        var z = Float32(0.0)
        var r0 = Float32(0.0)
        var r1 = Float32(0.0)
        for s in range(TS_LANES):
            z = ftz(z + part[rr * TS_LANES + s])
            r0 = ftz(r0 + part[(RLB + rr) * TS_LANES + s])
            r1 = ftz(r1 + part[(2 * RLB + rr) * TS_LANES + s])
        row_z.unsafe_store(i, z)
        rep.unsafe_store(2 * i, r0)
        rep.unsafe_store(2 * i + 1, r1)


def _enq_repulse(
    ctx: DeviceContext, n: Int, mut y: DeviceBuffer[DType.float32], mut drz: DeviceBuffer[DType.float32],
    mut drep: DeviceBuffer[DType.float32],
) raises:
    """The tiled route's repulsion launch (TS_LANES_SPLIT picks the shape)."""
    comptime if TS_LANES_SPLIT:
        ctx.enqueue_function[repulse_lanes_kernel](Int32(n), y.unsafe_ptr(), drz.unsafe_ptr(), drep.unsafe_ptr(),
                                                   grid_dim=(n + RLB - 1) // RLB, block_dim=RL_TPB)
    else:
        ctx.enqueue_function[repulse_tiled_kernel](Int32(n), y.unsafe_ptr(), drz.unsafe_ptr(), drep.unsafe_ptr(),
                                                   grid_dim=(n + RTB - 1) // RTB, block_dim=RTB)


# `ts_repulse_fold`'s statements in ascending j, skipping j == i: the same
# fold in the same order. Z is `ts_sum_cell`'s pinned pairwise tree, folded
# in parallel inside the same kernel (each block's rows, then the parts in
# the block that finishes last), so no one-block launch remains. Both
# coordinates' steps run in one thread (`step_rows_kernel`, the cell's
# statements in the cell's order). -D MOJOLEARN_TSNE_SPLIT_OFF=1 restores
# main's three kernels.
comptime TS_SPLIT = (
    (TARGET_COLUMN == COLUMN_NVIDIA or TARGET_COLUMN == COLUMN_AMD)
    and not is_defined["MOJOLEARN_TSNE_SPLIT_OFF"]()
)
comptime RS_ROWS = 32
comptime RS_TJ = 64
comptime RS_TPB = 256
comptime RS_DS = RS_TJ + 1
#: (row, candidate) pairs a thread of the split kernel forms per tile
comptime RS_Q = RS_ROWS * RS_TJ // RS_TPB
#: the parts folded per pass by the last block (a power of two, <= 8 * RS_TPB)
comptime ZT_CHUNK = 2048
comptime ZS_CHUNK = 2048


def repulse_split_kernel(n: Int32, y: F32P, row_z: F32P, rep: F32P, z_out: F32P, done: I32P, parts: F32P):
    var tid = Int(thread_idx.x)
    var nr = Int(n)
    var i0 = Int(block_idx.x) * RS_ROWS
    var ys = stack_allocation[2 * RS_TJ, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var yi_s = stack_allocation[2 * RS_ROWS, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var q_s = stack_allocation[RS_ROWS * RS_DS, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var a_s = stack_allocation[RS_ROWS * RS_DS, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var b_s = stack_allocation[RS_ROWS * RS_DS, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    if tid < 2 * RS_ROWS:
        var v = Float32(0.0)
        if 2 * i0 + tid < 2 * nr:
            v = ftz(y.unsafe_load(2 * i0 + tid))
        yi_s[tid] = v
    var i = i0 + tid
    var owner = tid < RS_ROWS and i < nr
    var z = Float32(0.0)
    var r0 = Float32(0.0)
    var r1 = Float32(0.0)
    # TS_LANE_FOLD (x_ann/tsne_core.mojo): this thread's RS_Q (row, lane)
    # partials stay in registers for the whole pass; thread tid owns lane
    # tid mod RS_TJ of rows tid // RS_TJ + (RS_TPB // RS_TJ) q.
    comptime assert RS_TJ == TS_LANES, "a staged candidate tile is one lane group"
    var lz = SIMD[DType.float32, RS_Q](0.0)
    var l0 = SIMD[DType.float32, RS_Q](0.0)
    var l1 = SIMD[DType.float32, RS_Q](0.0)
    var j0 = 0
    while j0 < nr:
        if tid < 2 * RS_TJ:
            var v = Float32(0.0)
            if 2 * j0 + tid < 2 * nr:
                v = ftz(y.unsafe_load(2 * j0 + tid))
            ys[tid] = v
        barrier()
        comptime if TS_LANE_FOLD:
            comptime for q in range(RS_Q):
                var e = tid + q * RS_TPB
                var rr = e // RS_TJ
                var jj = e - rr * RS_TJ
                if j0 + jj < nr and j0 + jj != i0 + rr:
                    var tm = ts_repulse_terms(yi_s[2 * rr], yi_s[2 * rr + 1], ys[2 * jj], ys[2 * jj + 1])
                    var zz = lz[q]
                    var aa = l0[q]
                    var bb = l1[q]
                    ts_repulse_fold(tm, zz, aa, bb)
                    lz[q] = zz
                    l0[q] = aa
                    l1[q] = bb
            barrier()
        else:
            comptime for q in range(RS_ROWS * RS_TJ // RS_TPB):
                var e = tid + q * RS_TPB
                var rr = e // RS_TJ
                var jj = e - rr * RS_TJ
                var tm = ts_repulse_terms(yi_s[2 * rr], yi_s[2 * rr + 1], ys[2 * jj], ys[2 * jj + 1])
                q_s[rr * RS_DS + jj] = tm[0]
                a_s[rr * RS_DS + jj] = tm[1]
                b_s[rr * RS_DS + jj] = tm[2]
            barrier()
            if owner:
                var jn = RS_TJ if nr - j0 > RS_TJ else nr - j0
                var row = tid * RS_DS
                for r in range(jn):
                    if j0 + r != i:
                        var tm = SIMD[DType.float32, 4](q_s[row + r], a_s[row + r], b_s[row + r], Float32(0.0))
                        ts_repulse_fold(tm, z, r0, r1)
            barrier()
        j0 += RS_TJ
    comptime if TS_LANE_FOLD:
        # publish the lane partials; row i's owner adds its TS_LANES lanes
        # in ascending s, each add flushed
        comptime for q in range(RS_Q):
            var e = tid + q * RS_TPB
            var rr = e // RS_TJ
            var jj = e - rr * RS_TJ
            q_s[rr * RS_DS + jj] = lz[q]
            a_s[rr * RS_DS + jj] = l0[q]
            b_s[rr * RS_DS + jj] = l1[q]
        barrier()
        if owner:
            var row = tid * RS_DS
            for s in range(RS_TJ):
                z = ftz(z + q_s[row + s])
                r0 = ftz(r0 + a_s[row + s])
                r1 = ftz(r1 + b_s[row + s])
    if owner:
        row_z.unsafe_store(i, z)
        rep.unsafe_store(2 * i, r0)
        rep.unsafe_store(2 * i + 1, r1)
        fence[ordering = Ordering.RELEASE]()
    # Z: `ts_sum_cell`'s pinned pairwise tree, in parallel. This block's
    # RS_ROWS (a power of two, so the block is an aligned subtree) fold level
    # by level into parts[b]; the block that finishes last folds the parts,
    # ZT_CHUNK (aligned) at a time, the same levels in place, until one
    # node is left. The nodes are the tree's nodes, so the bits are the
    # host's and Apple's.
    var zs = stack_allocation[ZT_CHUNK, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var rows_here = RS_ROWS if nr - i0 > RS_ROWS else nr - i0
    if tid < rows_here:
        zs[tid] = z
    var root = _tree_shared_kern(zs, rows_here, tid, RS_TPB)
    if tid == 0:
        parts.unsafe_store(Int(block_idx.x), root)
        fence[ordering = Ordering.RELEASE]()
    var last = stack_allocation[1, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    barrier()
    if tid == 0:
        var old = Atomic.fetch_add(done, Int32(1))
        last[0] = Int32(1) if Int(old) == Int(grid_dim.x) - 1 else Int32(0)
    barrier()
    if last[0] == Int32(0):
        return
    fence[ordering = Ordering.ACQUIRE]()
    var count = Int(grid_dim.x)
    while count > 1:
        var c0 = 0
        var nc = 0
        while c0 < count:
            var w = ZT_CHUNK if count - c0 > ZT_CHUNK else count - c0
            for e in range(tid, w, RS_TPB):
                zs[e] = parts.unsafe_load(c0 + e)
            var r = _tree_shared_kern(zs, w, tid, RS_TPB)
            if tid == 0:
                parts.unsafe_store(nc, r)
            barrier()
            fence[ordering = Ordering.ACQUIRE]()
            c0 += ZT_CHUNK
            nc += 1
        count = nc
    if tid == 0:
        z_out.unsafe_store(0, ftz(parts.unsafe_load(0)))
        done.unsafe_store(0, Int32(0))


#: fg-tsne-dbscan T2 (IDENTICAL, NVIDIA and AMD, DEFAULT OFF, lane
#: fg-tsne-dbscan 2026-10-09): THE REPULSION IN 256-CANDIDATE TILES, ONE ROW
#: PER THREAD. `repulse_split_kernel` stages 64 candidates per tile (one lane
#: group) and gives each thread RS_Q = 8 (row, lane) pairs of 8 DIFFERENT
#: rows, so every pair reads both points from threadgroup memory and a
#: barrier pair falls every 8 pairs per thread (n / 64 tiles x 2 barriers per
#: block per iteration). `repulse_rep_tile_kernel` stages RT2_TJ = 256
#: candidates (4 lane groups) per tile; thread (rr, q) owns row
#: i0 + rr (its y in registers, loaded once) and the RT2_U = 4 lanes
#: s = q + RT2_LQ u of that row, and for each lane folds the tile's
#: candidates j = j0 + 64 g + s for g ascending: lane s still folds its own j
#: (j mod 64 = s) ascending from +0 with `ts_repulse_fold`'s statements, j ==
#: i skipped, exactly TS_LANE_FOLD's lanes. The row's sums are its 64 lane
#: partials added in ascending s from +0, each add flushed (the split
#: kernel's owner statements), and Z is the split kernel's pinned pairwise
#: tree over the rows (RT2_ROWS a power of two: each block is an aligned
#: subtree), its parts folded by the last block. The same words in the same
#: order: bits none. Cost: 16 pairs per thread per barrier pair (8 before),
#: n / 256 tiles (4x fewer barriers per block), one y_i load per row per
#: pass; n / RT2_ROWS blocks of 256 threads (2x the blocks of the split
#: kernel: more of the GPU busy at a few tens of thousands of rows). The
#: division and the flushes stay. Expected ~2x on the iteration stage.
#: Requires TS_LANE_FOLD (the single-chain arm keeps the split kernel).
#: Default on (NVIDIA and AMD, wherever TS_SPLIT and TS_LANE_FOLD hold)
#: since 2026-10-09 (lane/postmerge-act-4; post-merge A/B, one run per arm,
#: ratios arm / default, nv L40S n0608 -> v0979..v0981, amd MI325X a1067 ->
#: a1104..a1106): alone istella NV 1.42x / AMD 0.60x, taxi NV 1.20x / AMD
#: 0.59x; with IDN_TSNE_KNN_TILE_SELECT (both promoted together) istella NV
#: 1191 ms (0.86x) / AMD 1219 (0.62x), avg 0.74x; taxi NV 1831 (1.17x) / AMD
#: 1048 (0.55x), avg 0.86x. Vendor split: NV taxi 1.17x slower with both on
#: (the average of the two vendors decides). Trustworthiness@15 equal and
#: digests identical in every arm on both vendors (istella ce69cdffd6f889b6,
#: taxi cee32a8a52defb1c): no bit change.
#: -D MOJOLEARN_IDN_TSNE_REP_TILE_OFF (or MOJOLEARN_IDN_ALL_OFF) restores the split kernel.
comptime IDN_TSNE_REP_TILE = (
    TS_SPLIT
    and TS_LANE_FOLD
    and not (
        is_defined["MOJOLEARN_IDN_TSNE_REP_TILE_OFF"]()
        or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
    )
)
#: threads per row, lanes per thread, rows per block, candidates per tile
comptime RT2_LQ = 16
comptime RT2_U = TS_LANES // RT2_LQ
comptime RT2_TPB = 256
comptime RT2_ROWS = RT2_TPB // RT2_LQ
comptime RT2_TJ = 4 * TS_LANES
#: the rows per Z part of the iteration's repulsion launch (dparts' size)
comptime TS_PART_ROWS = RT2_ROWS if IDN_TSNE_REP_TILE else RS_ROWS


def repulse_rep_tile_kernel(n: Int32, y: F32P, row_z: F32P, rep: F32P, z_out: F32P, done: I32P, parts: F32P):
    """`repulse_split_kernel` under TS_LANE_FOLD with 256-candidate tiles and
    one row per thread (IDN_TSNE_REP_TILE): the same words."""
    comptime assert RT2_U * RT2_LQ == TS_LANES, "a row's lanes split evenly over its threads"
    comptime assert RT2_TJ % TS_LANES == 0, "a tile holds whole lane groups"
    comptime assert RT2_ROWS * RT2_LQ == RT2_TPB, "the block is whole rows"
    comptime assert 2 * RT2_TJ == 2 * RT2_TPB, "one candidate staged per thread"
    var tid = Int(thread_idx.x)
    var nr = Int(n)
    var i0 = Int(block_idx.x) * RT2_ROWS
    var rr = tid // RT2_LQ
    var q = tid - rr * RT2_LQ
    var i = i0 + rr
    var live = i < nr
    var owner = live and q == 0
    var ys = stack_allocation[2 * RT2_TJ, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var part = stack_allocation[3 * RT2_ROWS * TS_LANES, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var y0 = Float32(0.0)
    var y1 = Float32(0.0)
    if live:
        y0 = ftz(y.unsafe_load(2 * i))
        y1 = ftz(y.unsafe_load(2 * i + 1))
    var lz = SIMD[DType.float32, RT2_U](0.0)
    var l0 = SIMD[DType.float32, RT2_U](0.0)
    var l1 = SIMD[DType.float32, RT2_U](0.0)
    var j0 = 0
    while j0 < nr:
        var v0 = Float32(0.0)
        var v1 = Float32(0.0)
        if j0 + tid < nr:
            v0 = ftz(y.unsafe_load(2 * (j0 + tid)))
            v1 = ftz(y.unsafe_load(2 * (j0 + tid) + 1))
        ys[2 * tid] = v0
        ys[2 * tid + 1] = v1
        barrier()
        if live:
            # lane s = q + RT2_LQ u folds j = j0 + 64 g + s, g ascending
            comptime for g in range(RT2_TJ // TS_LANES):
                comptime for u in range(RT2_U):
                    var jj = g * TS_LANES + q + RT2_LQ * u
                    if j0 + jj < nr and j0 + jj != i:
                        var tm = ts_repulse_terms(y0, y1, ys[2 * jj], ys[2 * jj + 1])
                        var zz = lz[u]
                        var aa = l0[u]
                        var bb = l1[u]
                        ts_repulse_fold(tm, zz, aa, bb)
                        lz[u] = zz
                        l0[u] = aa
                        l1[u] = bb
        barrier()
        j0 += RT2_TJ
    # publish the lane partials; row i's owner adds its TS_LANES lanes in
    # ascending s from +0, each add flushed
    comptime for u in range(RT2_U):
        var s = q + RT2_LQ * u
        part[rr * TS_LANES + s] = lz[u]
        part[(RT2_ROWS + rr) * TS_LANES + s] = l0[u]
        part[(2 * RT2_ROWS + rr) * TS_LANES + s] = l1[u]
    barrier()
    var z = Float32(0.0)
    var r0 = Float32(0.0)
    var r1 = Float32(0.0)
    if owner:
        for s in range(TS_LANES):
            z = ftz(z + part[rr * TS_LANES + s])
            r0 = ftz(r0 + part[(RT2_ROWS + rr) * TS_LANES + s])
            r1 = ftz(r1 + part[(2 * RT2_ROWS + rr) * TS_LANES + s])
        row_z.unsafe_store(i, z)
        rep.unsafe_store(2 * i, r0)
        rep.unsafe_store(2 * i + 1, r1)
        fence[ordering = Ordering.RELEASE]()
    # Z: `repulse_split_kernel`'s tail with RT2_ROWS rows per block
    var zs = stack_allocation[ZT_CHUNK, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var rows_here = RT2_ROWS if nr - i0 > RT2_ROWS else nr - i0
    if owner:
        zs[rr] = z
    var root = _tree_shared_kern(zs, rows_here, tid, RT2_TPB)
    if tid == 0:
        parts.unsafe_store(Int(block_idx.x), root)
        fence[ordering = Ordering.RELEASE]()
    var last = stack_allocation[1, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    barrier()
    if tid == 0:
        var old = Atomic.fetch_add(done, Int32(1))
        last[0] = Int32(1) if Int(old) == Int(grid_dim.x) - 1 else Int32(0)
    barrier()
    if last[0] == Int32(0):
        return
    fence[ordering = Ordering.ACQUIRE]()
    var count = Int(grid_dim.x)
    while count > 1:
        var c0 = 0
        var nc = 0
        while c0 < count:
            var w = ZT_CHUNK if count - c0 > ZT_CHUNK else count - c0
            for e in range(tid, w, RT2_TPB):
                zs[e] = parts.unsafe_load(c0 + e)
            var r = _tree_shared_kern(zs, w, tid, RT2_TPB)
            if tid == 0:
                parts.unsafe_store(nc, r)
            barrier()
            fence[ordering = Ordering.ACQUIRE]()
            c0 += ZT_CHUNK
            nc += 1
        count = nc
    if tid == 0:
        z_out.unsafe_store(0, ftz(parts.unsafe_load(0)))
        done.unsafe_store(0, Int32(0))


@always_inline
def _tree_shared_kern(
    buf: UnsafePointer[Float32, MutUntrackedOrigin, address_space=AddressSpace.SHARED], w_in: Int, tid: Int, nth: Int
) -> Float32:
    """The pinned pairwise tree over buf[0 : w_in] in place: level by level
    node q = ftz(c[2q] + c[2q + 1]), an odd last node carried; returns the
    root (unflushed when w_in == 1, as the tree carries it). Every thread of
    the block calls it."""
    barrier()
    var w = w_in
    while w > 1:
        var pairs = w // 2
        var vals = InlineArray[Float32, 8](fill=Float32(0.0))
        var k = 0
        for q in range(tid, pairs, nth):
            vals[k] = ts_z_add(buf[2 * q], buf[2 * q + 1])
            k += 1
        barrier()
        k = 0
        for q in range(tid, pairs, nth):
            buf[q] = vals[k]
            k += 1
        if w % 2 == 1 and tid == 0:
            buf[pairs] = buf[w - 1]
        barrier()
        w = pairs + w % 2
    var r = buf[0]
    barrier()
    return r


def _step_tail(
    e: Int, yi: Float32, attr: Float32, y_new: F32P, rep: F32P, z: F32P, update: F32P, gains: F32P,
    exaggeration: Float32, momentum: Float32, learning_rate: Float32,
):
    """`ts_step_cell` after its attraction sum, statement for statement."""
    var neg = ftz(identical_div(rep.unsafe_load(e), z.unsafe_load(0)))
    var grad = ftz(identical_mul(Float32(4.0), ftz(ftz(identical_mul(exaggeration, attr)) - neg)))
    var upd = update.unsafe_load(e)
    var gain = gains.unsafe_load(e)
    if ftz(identical_mul(upd, grad)) < Float32(0.0):
        gain = ftz(gain + Float32(0.2))
    else:
        gain = ftz(identical_mul(gain, Float32(0.8)))
    if gain < Float32(0.01):
        gain = Float32(0.01)
    grad = ftz(identical_mul(grad, gain))
    upd = ftz(ftz(identical_mul(momentum, upd)) - ftz(identical_mul(learning_rate, grad)))
    gains.unsafe_store(e, gain)
    update.unsafe_store(e, upd)
    y_new.unsafe_store(e, ftz(yi + upd))


def step_rows_kernel(
    n: Int32, y: F32P, y_new: F32P, indptr: I32P, indices: I32P, values: F32P, rep: F32P,
    z: F32P, update: F32P, gains: F32P, exaggeration: Float32, momentum: Float32, lr: Float32,
):
    """FAST on Apple, OPT-IN (lane ann-apple3): `ts_step_cell` for both
    coordinates of row i in one thread. The attraction walks the row's CSR
    entries once and forms each neighbor's `ts_q` once (the cell forms it once
    per coordinate); each coordinate's sum and update are the cell's
    statements in the cell's order."""
    var i = _tid()
    if i < Int(n):
        var y0 = ftz(y.unsafe_load(2 * i))
        var y1 = ftz(y.unsafe_load(2 * i + 1))
        var a0 = Float32(0.0)
        var a1 = Float32(0.0)
        for s in range(Int(indptr.unsafe_load(i)), Int(indptr.unsafe_load(i + 1))):
            var j = Int(indices.unsafe_load(s))
            var pq = ftz(identical_mul(ftz(values.unsafe_load(s)), ts_q(y, i, j)))
            a0 = ftz(a0 + ftz(identical_mul(pq, ftz(y0 - ftz(y.unsafe_load(2 * j))))))
            a1 = ftz(a1 + ftz(identical_mul(pq, ftz(y1 - ftz(y.unsafe_load(2 * j + 1))))))
        _step_tail(2 * i, y0, a0, y_new, rep, z, update, gains, exaggeration, momentum, lr)
        _step_tail(2 * i + 1, y1, a1, y_new, rep, z, update, gains, exaggeration, momentum, lr)


def step_kernel(
    count: Int32, y: F32P, y_new: F32P, indptr: I32P, indices: I32P, values: F32P, rep: F32P,
    z: F32P, update: F32P, gains: F32P, exaggeration: Float32, momentum: Float32, lr: Float32,
):
    var e = _tid()
    if e < Int(count):
        ts_step_cell(e, y, y_new, indptr, indices, values, rep, z, update, gains, exaggeration, momentum, lr)


def kl_kernel(n: Int32, y: F32P, indptr: I32P, indices: I32P, values: F32P, z: F32P, kl: F32P):
    var i = _tid()
    if i < Int(n):
        ts_kl_cell(i, y, indptr, indices, values, z, kl)


def _enq_repulse_split(
    ctx: DeviceContext, n: Int, mut ycur: DeviceBuffer[DType.float32], mut drz: DeviceBuffer[DType.float32],
    mut drep: DeviceBuffer[DType.float32], mut dz: DeviceBuffer[DType.float32], mut dcnt: DeviceBuffer[DType.int32],
    mut dparts: DeviceBuffer[DType.float32],
) raises:
    """TS_SPLIT's repulsion + Z launch (IDN_TSNE_REP_TILE picks the shape)."""
    comptime if IDN_TSNE_REP_TILE:
        ctx.enqueue_function[repulse_rep_tile_kernel](Int32(n), ycur.unsafe_ptr(), drz.unsafe_ptr(),
                                                      drep.unsafe_ptr(), dz.unsafe_ptr(), dcnt.unsafe_ptr(),
                                                      dparts.unsafe_ptr(), grid_dim=(n + RT2_ROWS - 1) // RT2_ROWS,
                                                      block_dim=RT2_TPB)
    else:
        ctx.enqueue_function[repulse_split_kernel](Int32(n), ycur.unsafe_ptr(), drz.unsafe_ptr(), drep.unsafe_ptr(),
                                                   dz.unsafe_ptr(), dcnt.unsafe_ptr(), dparts.unsafe_ptr(),
                                                   grid_dim=(n + RS_ROWS - 1) // RS_ROWS, block_dim=RS_TPB)


def _ts_iter(
    ctx: DeviceContext, mut ycur: DeviceBuffer[DType.float32], mut ynext: DeviceBuffer[DType.float32], n: Int,
    mut dptr: DeviceBuffer[DType.int32], mut dind: DeviceBuffer[DType.int32], mut dval: DeviceBuffer[DType.float32],
    mut drz: DeviceBuffer[DType.float32], mut drep: DeviceBuffer[DType.float32], mut dz: DeviceBuffer[DType.float32],
    mut dupd: DeviceBuffer[DType.float32], mut dgain: DeviceBuffer[DType.float32], ex: Float32, mom: Float32,
    lr: Float32, mut dcnt: DeviceBuffer[DType.int32], mut dparts: DeviceBuffer[DType.float32],
    mut dzs: DeviceBuffer[DType.float32],
) raises:
    comptime if TS_SPLIT:
        _enq_repulse_split(ctx, n, ycur, drz, drep, dz, dcnt, dparts)
        ctx.enqueue_function[step_rows_kernel](
            Int32(n), ycur.unsafe_ptr(), ynext.unsafe_ptr(), dptr.unsafe_ptr(), dind.unsafe_ptr(),
            dval.unsafe_ptr(), drep.unsafe_ptr(), dz.unsafe_ptr(), dupd.unsafe_ptr(), dgain.unsafe_ptr(), ex,
            mom, lr, grid_dim=_grid(n), block_dim=TPB,
        )
        return
    _enq_repulse(ctx, n, ycur, drz, drep)
    _ts_z(ctx, n, drz, dz, dzs)
    comptime if TS_STEP_ROWS:
        ctx.enqueue_function[step_rows_kernel](
            Int32(n), ycur.unsafe_ptr(), ynext.unsafe_ptr(), dptr.unsafe_ptr(), dind.unsafe_ptr(),
            dval.unsafe_ptr(), drep.unsafe_ptr(), dz.unsafe_ptr(), dupd.unsafe_ptr(), dgain.unsafe_ptr(), ex,
            mom, lr, grid_dim=_grid(n), block_dim=TPB,
        )
    else:
        ctx.enqueue_function[step_kernel](
            Int32(2 * n), ycur.unsafe_ptr(), ynext.unsafe_ptr(), dptr.unsafe_ptr(), dind.unsafe_ptr(),
            dval.unsafe_ptr(), drep.unsafe_ptr(), dz.unsafe_ptr(), dupd.unsafe_ptr(), dgain.unsafe_ptr(), ex,
            mom, lr, grid_dim=_grid(2 * n), block_dim=TPB,
        )


def _ts_iter_timed(
    ctx: DeviceContext, mut ycur: DeviceBuffer[DType.float32], mut ynext: DeviceBuffer[DType.float32], n: Int,
    mut dptr: DeviceBuffer[DType.int32], mut dind: DeviceBuffer[DType.int32], mut dval: DeviceBuffer[DType.float32],
    mut drz: DeviceBuffer[DType.float32], mut drep: DeviceBuffer[DType.float32], mut dz: DeviceBuffer[DType.float32],
    mut dupd: DeviceBuffer[DType.float32], mut dgain: DeviceBuffer[DType.float32], ex: Float32, mom: Float32,
    lr: Float32, mut t_rep: Int, mut t_sum: Int, mut t_step: Int, mut dzs: DeviceBuffer[DType.float32],
    mut dcnt: DeviceBuffer[DType.int32], mut dparts: DeviceBuffer[DType.float32],
) raises:
    """`_ts_iter` for the stage pass only (MOJOLEARN_ANN_STAGES, lane
    ann-apple3): the same three launches, drained one by one, their wall
    times added to t_rep / t_sum / t_step (ns)."""
    var t0 = Int(perf_counter_ns())
    # fg-tsne-dbscan: on TS_SPLIT builds the stage pass times the launches
    # the fit runs (repulsion and Z in one launch, reported as repulse; sum
    # 0), not the tiled route's three
    comptime if TS_SPLIT:
        _enq_repulse_split(ctx, n, ycur, drz, drep, dz, dcnt, dparts)
        ctx.synchronize()
        var ts1 = Int(perf_counter_ns())
        ctx.enqueue_function[step_rows_kernel](
            Int32(n), ycur.unsafe_ptr(), ynext.unsafe_ptr(), dptr.unsafe_ptr(), dind.unsafe_ptr(),
            dval.unsafe_ptr(), drep.unsafe_ptr(), dz.unsafe_ptr(), dupd.unsafe_ptr(), dgain.unsafe_ptr(), ex,
            mom, lr, grid_dim=_grid(n), block_dim=TPB,
        )
        ctx.synchronize()
        var ts3 = Int(perf_counter_ns())
        t_rep += ts1 - t0
        t_step += ts3 - ts1
        return
    _enq_repulse(ctx, n, ycur, drz, drep)
    ctx.synchronize()
    var t1 = Int(perf_counter_ns())
    _ts_z(ctx, n, drz, dz, dzs)
    ctx.synchronize()
    var t2 = Int(perf_counter_ns())
    comptime if TS_STEP_ROWS:
        ctx.enqueue_function[step_rows_kernel](
            Int32(n), ycur.unsafe_ptr(), ynext.unsafe_ptr(), dptr.unsafe_ptr(), dind.unsafe_ptr(),
            dval.unsafe_ptr(), drep.unsafe_ptr(), dz.unsafe_ptr(), dupd.unsafe_ptr(), dgain.unsafe_ptr(), ex,
            mom, lr, grid_dim=_grid(n), block_dim=TPB,
        )
    else:
        ctx.enqueue_function[step_kernel](
            Int32(2 * n), ycur.unsafe_ptr(), ynext.unsafe_ptr(), dptr.unsafe_ptr(), dind.unsafe_ptr(),
            dval.unsafe_ptr(), drep.unsafe_ptr(), dz.unsafe_ptr(), dupd.unsafe_ptr(), dgain.unsafe_ptr(), ex,
            mom, lr, grid_dim=_grid(2 * n), block_dim=TPB,
        )
    ctx.synchronize()
    var t3 = Int(perf_counter_ns())
    t_rep += t1 - t0
    t_sum += t2 - t1
    t_step += t3 - t2


def _ts_kl(
    ctx: DeviceContext, mut y: DeviceBuffer[DType.float32], n: Int, mut dptr: DeviceBuffer[DType.int32],
    mut dind: DeviceBuffer[DType.int32], mut dval: DeviceBuffer[DType.float32], mut drz: DeviceBuffer[DType.float32],
    mut drep: DeviceBuffer[DType.float32], mut dz: DeviceBuffer[DType.float32], mut dkl: DeviceBuffer[DType.float32],
    mut dzs: DeviceBuffer[DType.float32],
) raises:
    _enq_repulse(ctx, n, y, drz, drep)
    _ts_z(ctx, n, drz, dz, dzs)
    ctx.enqueue_function[kl_kernel](Int32(n), y.unsafe_ptr(), dptr.unsafe_ptr(), dind.unsafe_ptr(),
                                    dval.unsafe_ptr(), dz.unsafe_ptr(), dkl.unsafe_ptr(), grid_dim=_grid(n),
                                    block_dim=TPB)


def tsne_fit_device(
    x: List[Float32], n: Int, d: Int, y0: List[Float32], perplexity: Float32, exaggeration: Float32,
    learning_rate: Float32, max_iter: Int, exploration: Int, mut y_out: List[Float32], mut kl_out: Float32,
) raises:
    tsne_validate(n, d, perplexity, max_iter, exploration)
    var nn = tsne_nn(n, perplexity)
    var st = AnnStages("tsne_fit")
    var ctx = x_ann_ctx()
    var dx = upload_f32(ctx, x)
    st.mark(ctx, "upload")
    var dnd = ctx.enqueue_create_buffer[DType.float32](n * nn)
    var dni = ctx.enqueue_create_buffer[DType.int32](n * nn)
    var dp = ctx.enqueue_create_buffer[DType.float32](n * nn)
    knn_enqueue(ctx, dx, n, d, nn, dnd, dni)
    # fg-tsne-dbscan: the kNN stage alone (MOJOLEARN_ANN_STAGES only; unset,
    # `mark` neither syncs nor prints)
    st.mark(ctx, "knn")
    ctx.enqueue_function[perplexity_kernel](Int32(n), dnd.unsafe_ptr(), Int32(nn), identical_log(perplexity),
                                            dp.unsafe_ptr(), grid_dim=_grid(n), block_dim=TPB)
    ctx.synchronize()
    st.host("perplexity")
    # lane cgr4-download-loop: P symmetrized on the device
    # (x_ann/tsne_sym_device.mojo), the host function's CSR and words; the
    # graph never comes to the host
    var graph = tsne_symmetrize_device(ctx, dni, dp, n, nn)
    st.host("symmetrize")

    ref dptr = graph.indptr
    ref dind = graph.indices
    ref dval = graph.values
    var dy = upload_f32(ctx, y0)
    var dy2 = upload_f32(ctx, y0)
    var dupd = upload_f32(ctx, List[Float32](length=2 * n, fill=Float32(0.0)))
    var dgain = upload_f32(ctx, List[Float32](length=2 * n, fill=Float32(1.0)))
    var drz = ctx.enqueue_create_buffer[DType.float32](n)
    var drep = ctx.enqueue_create_buffer[DType.float32](2 * n)
    var dz = ctx.enqueue_create_buffer[DType.float32](1)
    var dcnt = ctx.enqueue_create_buffer[DType.int32](1)
    ctx.enqueue_memset(dcnt, Int32(0))
    var dparts = ctx.enqueue_create_buffer[DType.float32]((n + TS_PART_ROWS - 1) // TS_PART_ROWS + 1)
    var dzs = ctx.enqueue_create_buffer[DType.float32]((n + 1) // 2 + 1)
    var dkl = ctx.enqueue_create_buffer[DType.float32](n)
    st.mark(ctx, "upload_graph")
    var t_rep = 0
    var t_sum = 0
    var t_step = 0
    for it in range(max_iter):
        var ex = exaggeration if it < exploration else Float32(1.0)
        var mom = Float32(0.5) if it < exploration else Float32(0.8)
        if st.on:
            # the stage pass: each launch drained and timed (ann-apple3)
            if it % 2 == 0:
                _ts_iter_timed(ctx, dy, dy2, n, dptr, dind, dval, drz, drep, dz, dupd, dgain, ex, mom,
                               learning_rate, t_rep, t_sum, t_step, dzs, dcnt, dparts)
            else:
                _ts_iter_timed(ctx, dy2, dy, n, dptr, dind, dval, drz, drep, dz, dupd, dgain, ex, mom,
                               learning_rate, t_rep, t_sum, t_step, dzs, dcnt, dparts)
        elif it % 2 == 0:
            _ts_iter(ctx, dy, dy2, n, dptr, dind, dval, drz, drep, dz, dupd, dgain, ex, mom, learning_rate, dcnt, dparts, dzs)
        else:
            _ts_iter(ctx, dy2, dy, n, dptr, dind, dval, drz, drep, dz, dupd, dgain, ex, mom, learning_rate, dcnt, dparts, dzs)
    st.mark(ctx, "iterations")
    if st.on:
        print("ANN-STAGE tsne_iter repulse", Float64(t_rep) / 1.0e6)
        print("ANN-STAGE tsne_iter sum", Float64(t_sum) / 1.0e6)
        print("ANN-STAGE tsne_iter step", Float64(t_step) / 1.0e6)
    if max_iter % 2 == 0:
        _ts_kl(ctx, dy, n, dptr, dind, dval, drz, drep, dz, dkl, dzs)
    else:
        _ts_kl(ctx, dy2, n, dptr, dind, dval, drz, drep, dz, dkl, dzs)
    ctx.synchronize()
    _ = dcnt^
    _ = dparts^
    _ = dzs^
    if max_iter % 2 == 0:
        y_out = download_f32(ctx, dy, 2 * n)
    else:
        y_out = download_f32(ctx, dy2, 2 * n)
    # lane cgr4-download-loop: the KL total folds on the device in the fixed
    # order of core/device_fold.mojo (the host column folds the same order)
    kl_out = device_sum_f32_fixed(ctx, dkl, n)
    st.host("kl_download")
    _ = dkl^
    _ = dz^
    _ = drep^
    _ = drz^
    _ = dgain^
    _ = dupd^
    _ = dy2^
    _ = dy^
    _ = graph^
    _ = dp^
    _ = dni^
    _ = dnd^
    _ = dx^
    _ = ctx^
