"""Feed the grouped ranking derivatives into the existing symmetric tree search."""

from core.host_parallel import host_parallelize
from core.host_predict_threads import HostF32Ptr
from gbdt.host.gbdt_oracle import GBDT_MSE_BLOCK, _gbdt_units, _halving_fold
from gbdt.host.lightgbm_ranking import HostLightGBMRank
from gbdt.targets.ranking_options import check_rank_grade, OBJECTIVE_LAMBDARANK, OBJECTIVE_QUERY_SOFTMAX
from std.math import isfinite


struct HostRankingFit(Movable):
    var workspace: HostLightGBMRank
    var directions: List[Float32]
    var hessians: List[Float32]

    def __init__(out self, group_sizes: List[Int], targets: List[Float32], objective: Int = OBJECTIVE_LAMBDARANK, weights: List[Float32] = List[Float32]()) raises:
        var n = len(targets)
        var units = _gbdt_units(n, n)
        var chunk = (n + units - 1) // units
        var labels = rebind[HostF32Ptr](targets.unsafe_ptr())
        var failed = List[Int](length=units, fill=0)
        var bad = rebind[MutPointer[Int, MutUntrackedOrigin]](failed.unsafe_ptr())

        var wp = rebind[HostF32Ptr](weights.unsafe_ptr())
        var weighted = len(weights) > 0
        if weighted and len(weights) != n:
            raise Error("ranking weights must have one entry per row")
        @parameter
        def check(unit: Int) {imm n, imm chunk, imm labels, imm bad, imm objective, imm wp, imm weighted}:
            for i in range(unit * chunk, min(n, (unit + 1) * chunk)):
                var y = labels[i]
                if objective == OBJECTIVE_QUERY_SOFTMAX:
                    var w = wp[i] if weighted else Float32(1)
                    if y >= 0 and isfinite(y) and w >= 0 and isfinite(w):
                        if y > 0 and w > 0:
                            bad[unit] = -1
                        continue
                    bad[unit] = 1
                    return
                if not (y >= 0 and y <= 30) or Float32(Int(y)) != y:
                    bad[unit] = 1
                    return

        host_parallelize(check, units)
        var positive = False
        for unit in range(units):
            if objective == OBJECTIVE_QUERY_SOFTMAX:
                if failed[unit] > 0:
                    raise Error("QuerySoftMax requires finite nonnegative targets and weights")
                positive = positive or failed[unit] < 0
            else:
                check_rank_grade(failed[unit] == 0)
        if objective == OBJECTIVE_QUERY_SOFTMAX and not positive:
            raise Error("QuerySoftMax requires positive total weighted relevance")
        self.workspace = HostLightGBMRank(group_sizes, n)
        self.directions = List[Float32](length=n, fill=Float32(0))
        self.hessians = List[Float32](length=n, fill=Float32(0))

    def search(mut self, objective: Int, targets: List[Float32], cursor: List[Float32],
               weights: List[Float32], seed: Int, iteration: Int, second_order: Bool,
               mut stats: List[Float32], mut fv: List[Float32], mut mags: List[Float32]) raises:
        self.workspace.derivatives(objective, targets, cursor, weights, seed, iteration,
                                   self.directions, self.hessians)
        fv[0] = self.workspace.training_metric(objective, targets, cursor, weights)
        var n = len(targets)
        var blocks = (n + GBDT_MSE_BLOCK - 1) // GBDT_MSE_BLOCK
        var units = _gbdt_units(blocks, 8 * n)
        var has_weights = len(weights) != 0
        var d = rebind[HostF32Ptr](self.directions.unsafe_ptr())
        var h = rebind[HostF32Ptr](self.hessians.unsafe_ptr())
        var w = rebind[HostF32Ptr](weights.unsafe_ptr())
        var s = rebind[HostF32Ptr](stats.unsafe_ptr())
        var m = rebind[HostF32Ptr](mags.unsafe_ptr())

        @parameter
        def pack(unit: Int) {imm n, imm blocks, imm units, imm d, imm h, imm w,
                            imm s, imm m, imm has_weights, imm second_order}:
            var w_abs = List[Float32](length=GBDT_MSE_BLOCK, fill=Float32(0))
            var g_abs = List[Float32](length=GBDT_MSE_BLOCK, fill=Float32(0))
            var b = unit
            while b < blocks:
                for t in range(GBDT_MSE_BLOCK):
                    var row = b * GBDT_MSE_BLOCK + t
                    w_abs[t] = 0
                    g_abs[t] = 0
                    if row < n:
                        var plane0 = h[row] if second_order else (
                            w[row] if has_weights else Float32(1))
                        s[row] = plane0
                        s[n + row] = d[row]
                        w_abs[t] = abs(plane0)
                        g_abs[t] = abs(d[row])
                m[2 * b] = _halving_fold(w_abs)
                m[2 * b + 1] = _halving_fold(g_abs)
                b += units

        host_parallelize(pack, units)
