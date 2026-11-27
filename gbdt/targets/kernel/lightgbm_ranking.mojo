# SPDX-License-Identifier: Apache-2.0
"""Device-resident grouped ranking. One workspace serves a fixed ordered training pool.

Keys are sorted by descending score, then stably by query. The existing device radix
sort carries document indices, giving equal scores their original order without atomics.
Every stage, including the final metric reduction, stays on the selected device.
"""
from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from core.stable_radix_sort import key_bits_for, stable_radix_bsum_len, stable_radix_counts_len, stable_radix_sort_pairs_u32
from gbdt.targets.lightgbm_ranking import (
    LR_PLANES, RF, RU, RankStages, RankView, lambda_derivatives, rank_ideal, rank_ndcg, rank_row, rank_sort_key,
    xe_derivatives,
)
from gbdt.targets.ranking_options import OBJECTIVE_LAMBDARANK, OBJECTIVE_RANK_XENDCG

comptime RANK_THREADS = 128


def _rank_rows_kernel[phase: Int](
    labels: RF, scores: RF, weights: RF, offsets: RU, sizes: RU, qids: RU, order: RU, second: RU,
    scratch: RF, direction: RF, hessian: RF, rows: Int32, groups: Int32, seed: UInt32, iteration: UInt32,
    weighted: Int32, count: Int32, arg: Int32, width: Int32, col: Int32,
):
    var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if row < Int(count):
        var a = RankView(labels, scores, weights, offsets, sizes, qids, order, second, scratch, direction, hessian,
                         Int(rows), Int(groups), Int(seed), Int(iteration), weighted != 0)
        rank_row[phase](a, row, Int(arg), Int(width), Int(col))


def _keys_kernel[grouped: Bool](src: RF, qids: RU, keys: RU, order: RU, count: Int32):
    var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if row < Int(count):
        comptime if grouped:
            keys[row] = qids[Int(order[row])]
        else:
            keys[row] = rank_sort_key(src[row])
            order[row] = UInt32(row)


struct LightGBMRankWorkspace(RankStages, Movable):
    var ctx: DeviceContext
    var n: Int
    var groups: Int
    var largest: Int
    var ready: Bool
    var scratch: DeviceBuffer[DType.float32]
    var order: DeviceBuffer[DType.uint32]
    var other: DeviceBuffer[DType.uint32]
    var keys: DeviceBuffer[DType.uint32]
    var temp_keys: DeviceBuffer[DType.uint32]
    var counts: DeviceBuffer[DType.int32]
    var block_sums: DeviceBuffer[DType.int32]

    def __init__(out self, ctx: DeviceContext, n_rows: Int, n_queries: Int, max_group_size: Int = 0) raises:
        if n_rows < 1 or n_rows > 2147483647 or n_queries < 1 or n_queries > n_rows:
            raise Error("LightGBM ranking needs positive query sizes covering the input rows")
        self.ctx = ctx.copy()
        self.n = n_rows
        self.groups = n_queries
        # For positive contiguous groups, the largest can hold at most n - (Q - 1).
        self.largest = max_group_size if max_group_size > 0 else n_rows - n_queries + 1
        self.ready = False
        self.scratch = ctx.enqueue_create_buffer[DType.float32](LR_PLANES * n_rows)
        self.order = ctx.enqueue_create_buffer[DType.uint32](n_rows)
        self.other = ctx.enqueue_create_buffer[DType.uint32](n_rows)
        self.keys = ctx.enqueue_create_buffer[DType.uint32](n_rows)
        self.temp_keys = ctx.enqueue_create_buffer[DType.uint32](n_rows)
        self.counts = ctx.enqueue_create_buffer[DType.int32](stable_radix_counts_len(n_rows))
        self.block_sums = ctx.enqueue_create_buffer[DType.int32](stable_radix_bsum_len(n_rows))

    def rows[phase: Int](mut self, a: RankView, count: Int, arg: Int, width: Int, col: Int) raises:
        self.ctx.enqueue_function[_rank_rows_kernel[phase]](
            a.labels, a.scores, a.weights, a.offsets, a.sizes, a.qids, a.order, a.second, a.scratch,
            a.direction, a.hessian, Int32(a.n), Int32(a.groups), UInt32(a.seed), UInt32(a.iteration),
            Int32(1 if a.weighted else 0), Int32(count), Int32(arg), Int32(width), Int32(col),
            grid_dim=(count + RANK_THREADS - 1) // RANK_THREADS, block_dim=RANK_THREADS,
        )

    def order_by[labels: Bool](mut self, a: RankView) raises:
        var src = a.labels if labels else a.scores
        var blocks = (a.n + RANK_THREADS - 1) // RANK_THREADS
        self.ctx.enqueue_function[_keys_kernel[False]](src, a.qids, self.keys.unsafe_ptr(), a.order, Int32(a.n),
                                                       grid_dim=blocks, block_dim=RANK_THREADS)
        stable_radix_sort_pairs_u32(self.ctx, a.n, 32, self.keys, self.order, self.temp_keys, self.other,
                                    self.counts, self.block_sums)
        if a.groups > 1:
            self.ctx.enqueue_function[_keys_kernel[True]](src, a.qids, self.keys.unsafe_ptr(), a.order, Int32(a.n),
                                                          grid_dim=blocks, block_dim=RANK_THREADS)
            stable_radix_sort_pairs_u32(self.ctx, a.n, key_bits_for(a.groups - 1), self.keys, self.order,
                                        self.temp_keys, self.other, self.counts, self.block_sums)


def launch_lightgbm_ranking(
    ctx: DeviceContext, mut ws: LightGBMRankWorkspace, objective: Int, mut targets: DeviceBuffer[DType.float32],
    mut scores: DeviceBuffer[DType.float32], mut weights: DeviceBuffer[DType.float32], has_weights: Bool,
    mut offsets: DeviceBuffer[DType.uint32], mut sizes: DeviceBuffer[DType.uint32], mut qids: DeviceBuffer[DType.uint32],
    n_rows: Int, n_queries: Int, seed: Int, iteration: Int,
    mut directions: DeviceBuffer[DType.float32], mut hessians: DeviceBuffer[DType.float32],
) raises:
    if n_rows != ws.n or n_queries != ws.groups:
        raise Error("LightGBM ranking workspace does not match this pool")
    var a = RankView(targets.unsafe_ptr(), scores.unsafe_ptr(), weights.unsafe_ptr(), offsets.unsafe_ptr(),
                     sizes.unsafe_ptr(), qids.unsafe_ptr(), ws.order.unsafe_ptr(), ws.other.unsafe_ptr(),
                     ws.scratch.unsafe_ptr(), directions.unsafe_ptr(), hessians.unsafe_ptr(), n_rows, n_queries,
                     seed, iteration, has_weights)
    var largest = ws.largest
    if objective == OBJECTIVE_LAMBDARANK:
        if not ws.ready:
            rank_ideal(ws, a, largest)
            ws.ready = True
        lambda_derivatives(ws, a, largest)
    elif objective == OBJECTIVE_RANK_XENDCG:
        xe_derivatives(ws, a, largest)
    else:
        raise Error("unknown grouped LightGBM objective")


def launch_lightgbm_ndcg(
    ctx: DeviceContext, mut ws: LightGBMRankWorkspace, mut targets: DeviceBuffer[DType.float32],
    mut scores: DeviceBuffer[DType.float32], mut offsets: DeviceBuffer[DType.uint32],
    mut sizes: DeviceBuffer[DType.uint32], mut qids: DeviceBuffer[DType.uint32], n_rows: Int, n_queries: Int,
    mut result: DeviceBuffer[DType.float32],
) raises:
    if n_rows != ws.n or n_queries != ws.groups:
        raise Error("LightGBM ranking workspace does not match this pool")
    var a = RankView(targets.unsafe_ptr(), scores.unsafe_ptr(), targets.unsafe_ptr(), offsets.unsafe_ptr(),
                     sizes.unsafe_ptr(), qids.unsafe_ptr(), ws.order.unsafe_ptr(), ws.other.unsafe_ptr(),
                     ws.scratch.unsafe_ptr(), result.unsafe_ptr(), result.unsafe_ptr(), n_rows, n_queries, 0, 0, False)
    var largest = ws.largest
    if not ws.ready:
        rank_ideal(ws, a, largest)
        ws.ready = True
    rank_ndcg(ws, a, largest)
