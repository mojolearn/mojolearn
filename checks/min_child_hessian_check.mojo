# SPDX-License-Identifier: Apache-2.0
"""Analytic child Hessian bounds, candidate runner-up, and full GPU fits."""
from max.gpu.host import DeviceContext
from std.gpu import thread_idx
from std.memory import bitcast
from gbdt.methods.greedy_subsets_searcher.kernel.compute_scores import child_hessian_below
from checks.min_split_gain_check import check as check_gain
from checks.numerics import numeric_mode_name
from checks.leafwise_scores_check import make_fixture, run_device, N_BF, check_leafwise_scores
from gbdt.options.child_hessian import child_hessian_threshold
from gbdt.options.catboost_options import GROW_LOSSGUIDE, SCORE_FUNCTION_NEWTON_L2, SCORE_FUNCTION_NEWTON_COSINE
from gbdt.train import train
from gbdt.prepared import prepare_numeric_dataset
from gbdt.models.model_text import model_text

def tiny_bounds_kernel(pairs: MutPointer[UInt32, MutAnyOrigin], output: MutPointer[UInt32, MutAnyOrigin]):
    var i = Int(thread_idx.x)
    if i < 6:
        var mass = bitcast[DType.float32](pairs[unsafe_offset=2*i])
        var bound = bitcast[DType.float32](pairs[unsafe_offset=2*i+1])
        output[unsafe_offset=i] = UInt32(1 if child_hessian_below(mass, bound) else 0)

def tiny_bounds(ctx: DeviceContext) raises:
    var tiny = child_hessian_threshold(Float64(1e-300), GROW_LOSSGUIDE, SCORE_FUNCTION_NEWTON_L2)
    if bitcast[DType.uint32](tiny) != 1:
        raise Error("tiny Float64 bound did not round upward to Float32 bit1")
    var zero = child_hessian_threshold(Float64(-0.0), GROW_LOSSGUIDE, SCORE_FUNCTION_NEWTON_L2)
    if bitcast[DType.uint32](zero) != 0:
        raise Error("negative zero threshold was not canonicalized")
    var pairs: List[UInt32] = [0, 1, 1, 1, 1, 2, 0x80000000, 1, 1, 0x80000000, 0x80000000, 0x80000000]
    var expected: List[UInt32] = [1, 0, 1, 1, 0, 0]
    var h = ctx.enqueue_create_host_buffer[DType.uint32](12)
    for i in range(12):
        h.unsafe_ptr()[unsafe_offset=i] = pairs[i]
    var d = ctx.enqueue_create_buffer[DType.uint32](12)
    var out = ctx.enqueue_create_buffer[DType.uint32](6)
    var result = ctx.enqueue_create_host_buffer[DType.uint32](6)
    ctx.enqueue_copy(d, h)
    ctx.enqueue_function[tiny_bounds_kernel](d.unsafe_ptr(), out.unsafe_ptr(), grid_dim=1, block_dim=32)
    ctx.enqueue_copy(result, out)
    ctx.synchronize()
    for i in range(6):
        if result.unsafe_ptr()[unsafe_offset=i] != expected[i]:
            raise Error("device subnormal/signed-zero comparison failed")
    _ = d^
    _ = out^
    _ = h^
    print("TINY BOUNDS GREEN")


def candidates(ctx: DeviceContext) raises:
    var fx = make_fixture(1, 2, UInt64(3))
    for b in range(N_BF):
        fx.skip[b] = UInt8(1)
    fx.skip[0] = 0
    fx.skip[1] = 0
    fx.part_stats[0] = 10
    fx.part_stats[1] = 0
    fx.hist[0] = 1
    fx.hist[1] = 5
    fx.hist[N_BF] = 10
    fx.hist[N_BF + 1] = 5
    for i in range(len(fx.feature_weight)):
        fx.feature_weight[i] = 1
    for threshold in [Float64(-1), Float64(1), Float64(1.000000001), Float64(5), Float64(5.000000001)]:
        var bound = child_hessian_threshold(threshold, GROW_LOSSGUIDE, SCORE_FUNCTION_NEWTON_L2)
        var out = run_device[SCORE_FUNCTION_NEWTON_L2](ctx, fx, 0, 0, 2, bound)
        var expected = 0 if threshold <= 1 else (1 if threshold <= 5 else -1)
        if out[0][1] != expected:
            raise Error("candidate Hessian filtering failed runner-up/equality bound")
    fx.hist[0] = 9  # same invalid mass, now on the RIGHT child
    var right = run_device[SCORE_FUNCTION_NEWTON_L2](ctx, fx, 0, 0, 2, Float32(2))
    if right[0][1] != 1:
        raise Error("right-child Hessian bound was ignored")
    print("CANDIDATE BOUNDS GREEN")

def full_fit(ctx: DeviceContext, policy: String, loss: String, score: Int, threshold: Float64, expected: Int) raises:
    var x = List[Float32]()
    var y = List[Float32]()
    var weights = List[Float32]()
    var means: List[Float32] = [0, 0.25, 4, 4.25]
    var positive: List[Int] = [8, 16, 48, 56]
    for r in range(256):
        var group = r % 4
        x.append(Float32(group))
        y.append(means[group] if loss == "RMSE" else (Float32(1) if r // 4 < positive[group] else Float32(0)))
        weights.append(Float32(1) if loss == "RMSE" else Float32(2))
    var tm = train(ctx, x, y, 256, 1, border_count=3, n_estimators=1,
        max_depth=2, loss=loss, learning_rate=Float32(1), l2_leaf_reg=Float32(0),
        score_function=score, grow_policy=policy, min_child_hessian=threshold,
        boost_from_average=0, bootstrap_type="Bayesian", bagging_temperature=Float32(0), sample_weight=weights)
    print("FIT", policy, loss, score, threshold, "actual", tm.model.non_symmetric_models[0].bin_count(), "expected", expected)
    if tm.model.non_symmetric_models[0].bin_count() != expected:
        raise Error("full fit Hessian bound produced unexpected child count")
    if threshold == -1:
        var implicit = train(ctx, x, y, 256, 1, border_count=3, n_estimators=1,
            max_depth=2, loss=loss, learning_rate=Float32(1), l2_leaf_reg=Float32(0),
            score_function=score, grow_policy=policy, boost_from_average=0,
            bootstrap_type="Bayesian", bagging_temperature=Float32(0), sample_weight=weights)
        if model_text(implicit) != model_text(tm):
            raise Error("disabled Hessian bound changed model bits")
    print("FIT", policy, loss, score, threshold, expected)

def rejected_sibling(ctx: DeviceContext, policy: String) raises:
    var x = List[Float32]()
    var y = List[Float32]()
    var means: List[Float32] = [0, 0.25, 4, 4.25]
    for g in range(4):
        for _ in range(16 if g < 2 else 112):
            x.append(Float32(g))
            y.append(means[g])
    var pool = prepare_numeric_dataset(ctx, x, y, 256, 1, border_count=3)
    var tm = pool.fit(n_estimators=1, max_depth=3, grow_policy=policy,
        score_function=SCORE_FUNCTION_NEWTON_L2, min_child_hessian=17)
    if tm.model.non_symmetric_models[0].bin_count() != 3:
        raise Error("rejected sibling was revisited or eligible sibling was lost")
    print("PREPARED SIBLING GREEN", policy)

def main() raises:
    print("numeric_mode", numeric_mode_name())
    var ctx = DeviceContext()
    tiny_bounds(ctx)
    candidates(ctx)
    check_leafwise_scores(ctx)
    for policy in [String("Depthwise"), String("Lossguide")]:
        for score in [SCORE_FUNCTION_NEWTON_L2, SCORE_FUNCTION_NEWTON_COSINE]:
            for loss in [String("RMSE"), String("Logloss"), String("CrossEntropy")]:
                var h = Float64(64) if loss == "RMSE" else Float64(32)
                full_fit(ctx, policy, loss, score, -1, 4)
                full_fit(ctx, policy, loss, score, 0, 4)
                full_fit(ctx, policy, loss, score, h, 4)
                full_fit(ctx, policy, loss, score, h + 0.000000001, 2)
                full_fit(ctx, policy, loss, score, 2 * h + 0.000000001, 1)
        full_fit(ctx, policy, "RMSE", SCORE_FUNCTION_NEWTON_L2, -0.0, 4)
        rejected_sibling(ctx, policy)
        check_gain(ctx, policy, -1, 4)
        check_gain(ctx, policy, 2, 2)
        check_gain(ctx, policy, 1024, 1)
    print("MIN CHILD HESSIAN GREEN")
