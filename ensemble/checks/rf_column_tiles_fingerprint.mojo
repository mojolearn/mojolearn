# SPDX-License-Identifier: Apache-2.0
"""Full forests + prediction bits for the optional column histogram tiles.

Compared across reference/tile2/tile4 by tools/check_rf_column_tiles.sh.
Thirteen features force both full ten-column rounds and a three-column tail.
"""
from max.gpu.host import DeviceContext
from checks.numerics import numeric_mode_name
from ensemble.checks.fingerprint_probe import _mix, _fold, _fingerprint, _rf_params
from ensemble.decisiontree.batched_levelalgo.bins import ClassificationBin, WeightedClassificationBin, RegressionBin, WeightedRegressionBin, BinScales
from ensemble.decisiontree.batched_levelalgo.objectives import ClassificationObjectiveFunction, RegressionObjectiveFunction, ObjectiveLike
from ensemble.decisiontree.decisiontree import GINI, MSE
from ensemble.randomforest import RandomForest, fit_forest
from core.launch_log import log_launch
from std.sys.compile import is_defined

comptime C = ClassificationObjectiveFunction[DType.float32, DType.int32, ClassificationBin]
comptime CW = ClassificationObjectiveFunction[DType.float32, DType.int32, WeightedClassificationBin]
comptime R = RegressionObjectiveFunction[DType.float32, DType.float32, RegressionBin]
comptime RW = RegressionObjectiveFunction[DType.float32, DType.float32, WeightedRegressionBin]

def run[O: ObjectiveLike](ctx: DeviceContext, name: String, classes: Int, bins: Int, bootstrap: Bool, weighted: Bool, max_leaves: Int = -1) raises where O.DataT == DType.float32:
    comptime N = 1031
    comptime P = 13
    log_launch("CASE_BEGIN " + name)
    var x = List[Scalar[O.DataT]]()
    var hx = ctx.enqueue_create_host_buffer[DType.float32](N * P)
    var hy = ctx.enqueue_create_host_buffer[O.LabelT](N)
    var hw = ctx.enqueue_create_host_buffer[DType.float32](N)
    var weights = List[Float32]()
    for r in range(N):
        var cls = Int(_mix(UInt64(r * 7919 + 17)) % UInt64(classes))
        var target = Float32(cls)
        comptime if O.LabelT == DType.float32:
            target += Float32(r % 7) / 8.0
        hy.unsafe_ptr()[unsafe_offset=r] = Scalar[O.LabelT](target)
        var weight = Float32(0 if r % 17 == 0 else 1 + r % 4)
        hw.unsafe_ptr()[unsafe_offset=r] = weight
        if weighted:
            weights.append(weight)
        for c in range(P):
            var h = _mix(UInt64(r * 104729 + c * 7919 + 7))
            # Ragged cardinality per column, plus informative continuous cols.
            var value = Float32(Int(h % UInt64(2 + 11 * c)))
            if c < 3:
                value = target + Float32(Int(h % 101)) / 128.0
            x.append(Scalar[O.DataT](value))
            hx.unsafe_ptr()[unsafe_offset=r * P + c] = value
    var dx = ctx.enqueue_create_buffer[DType.float32](N * P)
    var dy = ctx.enqueue_create_buffer[O.LabelT](N)
    var dw = ctx.enqueue_create_buffer[DType.float32](N)
    ctx.enqueue_copy(dx, hx)
    ctx.enqueue_copy(dy, hy)
    ctx.enqueue_copy(dw, hw)
    ctx.synchronize()
    var params = _rf_params(3, bootstrap, 1.0, 4, bins, GINI if O.LabelT == DType.int32 else MSE, 2)
    params.tree_params.max_leaves = Int32(max_leaves)
    var forest = fit_forest[O](ctx, dx, dy, dw, N, P, classes if O.LabelT == DType.int32 else 1, params, BinScales(1024.0, 1024.0), weights, row_major=True, oob_score=bootstrap)
    var hash = _fingerprint(forest)
    var estimator = RandomForest[O.DataT, O.LabelT](params, 0 if O.LabelT == DType.int32 else 1)
    var prediction = List[Scalar[O.LabelT]]()
    for _ in range(N):
        prediction.append(Scalar[O.LabelT](0))
    estimator.predict(x, N, P, prediction, forest)
    for i in range(N):
        _fold(hash, UInt64(prediction[i].to_bits()))
    for t in range(len(forest.trees)):
        ref tree = forest.trees[t]
        _fold(hash, UInt64(tree.treeid))
        _fold(hash, UInt64(tree.num_outputs))
    print("fingerprint", name, hash)
    log_launch("CASE_END " + name)
    ctx.synchronize()
    _ = dx^
    _ = dy^
    _ = dw^
    _ = hx^
    _ = hy^
    _ = hw^

def main() raises:
    print("numeric_mode", numeric_mode_name())
    print("tile2", is_defined["MOJOLEARN_RF_HIST_COLUMNS2"](), "tile4", is_defined["MOJOLEARN_RF_HIST_COLUMNS4"]())
    var ctx = DeviceContext()
    run[C](ctx, "c2_boot", 2, 128, True, False)
    run[C](ctx, "c5_no_boot", 5, 64, False, False)
    run[CW](ctx, "c5_weighted", 5, 64, False, True)
    run[CW](ctx, "c2_weighted_boot", 2, 128, True, True)
    run[C](ctx, "c17_capacity", 17, 128, False, False)
    run[C](ctx, "c2_bins256", 2, 256, False, False)
    run[C](ctx, "c5_search", 5, 257, False, False)
    run[C](ctx, "c2_leaf_cap", 2, 128, False, False, 5)
    run[R](ctx, "reg_boot", 3, 128, True, False)
    run[RW](ctx, "reg_weighted", 3, 128, False, True)
    run[RW](ctx, "reg_weighted_boot", 3, 128, True, True)
