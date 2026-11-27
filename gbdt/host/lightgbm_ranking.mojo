# SPDX-License-Identifier: Apache-2.0
"""Grouped LightGBM objectives on the CPU. Scratch belongs to one ordered training pool."""
from cluster.host.host_cells import host_cells
from core.sort_ops_host import FRS_BINS, FRS_TILE, stable_radix_pairs_u32_host
from gbdt.targets.lightgbm_ranking import (
    LR_PLANES, RF, RU, RankStages, RankView, lambda_derivatives, rank_ideal, rank_ndcg, rank_row, rank_sort_key,
    xe_derivatives,
)
from gbdt.targets.ranking_options import OBJECTIVE_LAMBDARANK, OBJECTIVE_RANK_XENDCG


@always_inline
def _f(x: List[Float32]) -> RF:
    return RF(unsafe_from_address=Int(x.unsafe_ptr()))


@always_inline
def _u(x: List[UInt32]) -> RU:
    return RU(unsafe_from_address=Int(x.unsafe_ptr()))


struct HostLightGBMRank(RankStages, Movable):
    var n: Int
    var largest: Int
    var ready: Bool
    var sizes: List[UInt32]
    var offsets: List[UInt32]
    var qids: List[UInt32]
    var order: List[UInt32]
    var other: List[UInt32]
    var keys: List[UInt32]
    var temp_keys: List[UInt32]
    var counts: List[Int32]
    var scratch: List[Float32]

    def __init__(out self, group_sizes: List[Int], n_rows: Int) raises:
        if n_rows < 1 or n_rows > 2147483647 or len(group_sizes) == 0:
            raise Error("LightGBM ranking needs positive query sizes covering the input rows")
        self.n = n_rows
        self.ready = False
        self.sizes = List[UInt32](capacity=len(group_sizes))
        self.offsets = [UInt32(0)]
        var total = 0
        var largest = 0
        for size in group_sizes:  # prefix dependency: exclusive query boundaries
            if size <= 0 or size > n_rows - total:
                raise Error("LightGBM ranking query sizes must be positive and cover the input rows")
            total += size
            largest = max(largest, size)
            self.sizes.append(UInt32(size))
            self.offsets.append(UInt32(total))
        if total != n_rows:
            raise Error("LightGBM ranking query sizes must cover the input rows")
        self.largest = largest
        self.qids = List[UInt32](length=n_rows, fill=0)
        self.order = List[UInt32](length=n_rows, fill=0)
        self.other = List[UInt32](length=n_rows, fill=0)
        self.keys = List[UInt32](length=n_rows, fill=0)
        self.temp_keys = List[UInt32](length=n_rows, fill=0)
        self.counts = List[Int32](length=FRS_BINS * ((n_rows + FRS_TILE - 1) // FRS_TILE), fill=0)
        self.scratch = List[Float32](length=LR_PLANES * n_rows, fill=0)
        var qids = _u(self.qids)
        var offsets = _u(self.offsets)
        var queries = len(self.sizes)
        def set_query(row: Int) {imm qids, imm offsets, imm queries}:
            var lo = 0
            var hi = queries
            while hi - lo > 1:
                var mid = (lo + hi) // 2
                if Int(offsets[mid]) <= row:
                    lo = mid
                else:
                    hi = mid
            qids[row] = UInt32(lo)
        host_cells(set_query, n_rows, 8)

    def rows[phase: Int](mut self, a: RankView, count: Int, arg: Int, width: Int, col: Int) raises:
        def work(row: Int) {imm a, imm arg, imm width, imm col}:
            rank_row[phase](a, row, arg, width, col)
        host_cells(work, count, 32)

    def order_by[labels: Bool](mut self, a: RankView) raises:
        var keys = _u(self.keys)
        var temp = _u(self.temp_keys)
        var counts = MutPointer[Int32, MutAnyOrigin](unsafe_from_address=Int(self.counts.unsafe_ptr()))
        var src = a.labels if labels else a.scores
        def keyed(i: Int) {imm a, imm src, imm keys}:
            keys[i] = rank_sort_key(src[i])
            a.order[i] = UInt32(i)
        host_cells(keyed, a.n, 4)
        stable_radix_pairs_u32_host[4](keys, a.order, temp, a.second, counts, a.n)
        if a.groups > 1:
            def grouped(i: Int) {imm a, imm keys}:
                keys[i] = a.qids[Int(a.order[i])]
            host_cells(grouped, a.n, 2)
            if a.groups <= 65536:
                stable_radix_pairs_u32_host[2](keys, a.order, temp, a.second, counts, a.n)
            else:
                stable_radix_pairs_u32_host[4](keys, a.order, temp, a.second, counts, a.n)

    def _view(self, targets: List[Float32], scores: List[Float32], weights: List[Float32], seed: Int,
              iteration: Int, direction: RF, hessian: RF) raises -> RankView:
        if len(targets) != self.n or len(scores) != self.n or (len(weights) != 0 and len(weights) != self.n):
            raise Error("LightGBM ranking buffers must have one entry per row")
        return RankView(_f(targets), _f(scores), _f(weights) if len(weights) > 0 else _f(targets), _u(self.offsets),
                        _u(self.sizes), _u(self.qids), _u(self.order), _u(self.other), _f(self.scratch),
                        direction, hessian, self.n, len(self.sizes), seed, iteration, len(weights) != 0)

    def derivatives(mut self, objective: Int, targets: List[Float32], scores: List[Float32], weights: List[Float32],
                    seed: Int, iteration: Int, mut directions: List[Float32], mut hessians: List[Float32]) raises:
        directions.resize(self.n, Float32(0))
        hessians.resize(self.n, Float32(0))
        var a = self._view(targets, scores, weights, seed, iteration, _f(directions), _f(hessians))
        var largest = self.largest
        if objective == OBJECTIVE_LAMBDARANK:
            if not self.ready:
                rank_ideal(self, a, largest)
                self.ready = True
            lambda_derivatives(self, a, largest)
        elif objective == OBJECTIVE_RANK_XENDCG:
            xe_derivatives(self, a, largest)
        else:
            raise Error("unknown grouped LightGBM objective")

    def ndcg(mut self, targets: List[Float32], scores: List[Float32]) raises -> Float32:
        var result: List[Float32] = [0]
        var a = self._view(targets, scores, List[Float32](), 0, 0, _f(result), _f(result))
        var largest = self.largest
        if not self.ready:
            rank_ideal(self, a, largest)
            self.ready = True
        rank_ndcg(self, a, largest, 0)
        return result[0]
