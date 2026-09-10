# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Non-symmetric minimum selected-score improvement, full train path.

Four equally sized groups have means 0, 0.25, 4, 4.25. With L2 scoring,
unit weights and zero regularization, the root improves score by 1024 and
each child's split by exactly 2. Threshold equality must reject a split.
These exactly representable values test sign, units, child gating and the
boundary; a huge threshold alone would miss all of those.
"""
from max.gpu.host import DeviceContext
from checks.numerics import numeric_mode_name
from gbdt.models.model_text import model_text, load_model_text
from gbdt.options.catboost_options import SCORE_FUNCTION_L2
from gbdt.train import train, predict_floats
from std.memory import bitcast

comptime N = 256


def check(ctx: DeviceContext, policy: String, threshold: Float64,
          expected: Int) raises:
    var x = List[Float32]()
    var y = List[Float32]()
    var target: List[Float32] = [0, 0.25, 4, 4.25]
    for r in range(N):
        # Interleave groups so stable partition must actually reorder rows.
        x.append(Float32(r % 4))
        y.append(target[r % 4])
    var tm = train(ctx, x, y, N, 1, border_count=3,
        n_estimators=1, max_depth=2, learning_rate=Float32(1),
        l2_leaf_reg=Float32(0), score_function=SCORE_FUNCTION_L2,
        grow_policy=policy, min_split_gain=threshold,
        boost_from_average=0, bootstrap_type="No")
    if tm.model.is_oblivious() or tm.model.size() != 1:
        raise Error("wrong model kind")
    var leaves = tm.model.non_symmetric_models[0].bin_count()
    print(policy, "threshold", threshold, "leaves", leaves, "expected", expected)
    if leaves != expected:
        raise Error("min_split_gain did not gate the expected tree level")
    var p = predict_floats(ctx, tm, x, N)
    var back = load_model_text(model_text(tm))
    var p2 = predict_floats(ctx, back, x, N)
    for i in range(N):
        if bitcast[DType.uint32](p[i]) != bitcast[DType.uint32](p2[i]):
            raise Error("thresholded tree round trip changed predictions")
        var wanted = Float32(2.125)
        if expected == 4:
            wanted = target[i % 4]
        elif expected == 2:
            wanted = Float32(0.125) if i % 4 < 2 else Float32(4.125)
        if p[i] != wanted:
            raise Error("thresholded tree leaf value differs from group mean")
    if threshold == -1:
        var implicit = train(ctx, x, y, N, 1, border_count=3,
            n_estimators=1, max_depth=2, learning_rate=Float32(1),
            l2_leaf_reg=Float32(0), score_function=SCORE_FUNCTION_L2,
            grow_policy=policy, boost_from_average=0, bootstrap_type="No")
        if model_text(implicit) != model_text(tm):
            raise Error("disabled threshold changes default model bytes")


def main() raises:
    print("numeric_mode", numeric_mode_name())
    var ctx = DeviceContext()
    for policy in [String("Depthwise"), String("Lossguide")]:
        check(ctx, policy, -1, 4)
        check(ctx, policy, 0, 4)
        check(ctx, policy, 1.999, 4)
        check(ctx, policy, 2, 2)
        check(ctx, policy, 1023.999, 2)
        check(ctx, policy, 1024, 1)
    print("MIN SPLIT GAIN GREEN")
