# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Ordered loop gate: prefix isolation, persistent cursors, full fit/apply.

Build with -D MOJOLEARN_NUMERIC_IDENTICAL and compare ORDERED_BITS lines
between vendors. A same-vendor numerical PASS alone is not a certificate.
"""
from max.gpu.host import DeviceContext
from std.memory import bitcast
from std.math import isfinite
from core.identity_trace import IdentityTrace
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, numeric_mode_name
from gbdt.methods.dynamic_boosting import ordered_estimate_and_apply, fit_ordered_rmse
from gbdt.gpu_data.compressed_index_builder import build_layout
from gbdt.train import train_ordered_rmse, predict_floats
from gbdt.methods.doc_parallel_boosting import predict
from gbdt.models.oblivious_model import TAdditiveModel, TObliviousTreeModel, TObliviousTreeStructure, TBinarySplit


def prefix_case(ctx: DeviceContext, poison_tail: Bool, full_estimate: Bool, zero_prefix: Bool = False) raises -> List[Float32]:
    var order: List[UInt32] = [3, 0, 6, 1, 7, 2, 5, 4]
    var labels: List[Float32] = [2, 4, 20, 8, 40, 60, 6, 80]
    var hp = ctx.enqueue_create_host_buffer[DType.uint32](8)
    var hb = ctx.enqueue_create_host_buffer[DType.uint32](8)
    var hy = ctx.enqueue_create_host_buffer[DType.float32](8)
    var hw = ctx.enqueue_create_host_buffer[DType.float32](8)
    for i in range(8):
        hp.unsafe_ptr().unsafe_store(i, order[i])
        hb.unsafe_ptr().unsafe_store(i, UInt32(i % 2))
        hy.unsafe_ptr().unsafe_store(i, labels[i])
        hw.unsafe_ptr().unsafe_store(i, Float32(1))
    if zero_prefix:
        # Occupied prefix leaves have nonzero labels but zero mass. The
        # quality-only tail retains positive weights: global mass is >0.
        for i in range(4):
            hw.unsafe_ptr().unsafe_store(Int(order[i]), Float32(0))
    if poison_tail:
        for i in range(4, 8):
            hy.unsafe_ptr().unsafe_store(Int(order[i]), Float32(1000 + i * 100))
    var p = ctx.enqueue_create_buffer[DType.uint32](8)
    var b = ctx.enqueue_create_buffer[DType.uint32](8)
    var y = ctx.enqueue_create_buffer[DType.float32](8)
    var w = ctx.enqueue_create_buffer[DType.float32](8)
    var c = ctx.enqueue_create_buffer[DType.float32](8)
    ctx.enqueue_copy(dst_buf=p, src_ptr=hp.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=b, src_ptr=hb.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=y, src_ptr=hy.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=w, src_ptr=hw.unsafe_ptr())
    ctx.enqueue_memset(c, Float32(0.5))
    var trace = IdentityTrace.disabled()
    var result = List[Float32]()
    for iteration in range(2):
        var leaves = ordered_estimate_and_apply(
            ctx, 8 if full_estimate else 4, 8, 2, y, w, p, b, c,
            Float32(0.25), Float32(0), 1, trace, String("prefix"),
        )
        if zero_prefix and not full_estimate:
            for leaf in range(len(leaves)):
                if bitcast[DType.uint32](leaves[leaf]) != UInt32(0):
                    raise Error("ordered zero-mass prefix L2=0 must return positive zero leaves")
        elif not full_estimate:
            var decay = Float32(1) if iteration == 0 else Float32(0.75)
            if abs(leaves[0] - Float32(3.5) * decay) > Float32(0.00001) or abs(leaves[1] - Float32(5.5) * decay) > Float32(0.00001):
                raise Error("ordered prefix oracle includes tail labels or loses cursor history")
        for i in range(len(leaves)):
            result.append(leaves[i])
    var hc = ctx.enqueue_create_host_buffer[DType.float32](8)
    ctx.enqueue_copy(dst_ptr=hc.unsafe_ptr(), src_buf=c)
    ctx.synchronize()
    for i in range(8):
        var value = hc.unsafe_ptr().unsafe_load(i)
        if not isfinite(value):
            raise Error("ordered prefix cursor not finite")
        if zero_prefix and not full_estimate and value != Float32(0.5):
            raise Error("ordered zero-mass prefix changed its persistent cursor")
        result.append(value)
    _ = hp^
    _ = hb^
    _ = hy^
    _ = hw^
    return result^


def constant_model_check(ctx: DeviceContext) raises:
    """Known exact leaves with a nonzero bias; constants surround a split."""
    var counts: List[Int] = [1]
    var layout = build_layout(counts)
    var hx = ctx.enqueue_create_host_buffer[DType.uint32](4)
    var x = ctx.enqueue_create_buffer[DType.uint32](4)
    for i in range(4):
        hx.unsafe_ptr().unsafe_store(i, UInt32(i % 2) << layout.features[0].shift)
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    for mixed in range(2):
        var model = TAdditiveModel()
        model.bias = Float64(0.5)
        var first = TObliviousTreeModel(TObliviousTreeStructure())
        first.leaf_values.append(Float32(2))
        model.add_weak_model(first^)
        if mixed == 1:
            var structure = TObliviousTreeStructure()
            structure.splits.append(TBinarySplit(Int32(0), Int32(0), Int32(1)))
            var split_tree = TObliviousTreeModel(structure^)
            split_tree.leaf_values.append(Float32(-1))
            split_tree.leaf_values.append(Float32(3))
            model.add_weak_model(split_tree^)
            var last = TObliviousTreeModel(TObliviousTreeStructure())
            last.leaf_values.append(Float32(-0.25))
            model.add_weak_model(last^)
        var cursor = ctx.enqueue_create_buffer[DType.float32](4)
        predict(model, ctx, 4, counts, x, cursor)
        var hc = ctx.enqueue_create_host_buffer[DType.float32](4)
        ctx.enqueue_copy(dst_ptr=hc.unsafe_ptr(), src_buf=cursor)
        ctx.synchronize()
        for i in range(4):
            var expected = Float32(2.5)
            if mixed == 1:
                expected = Float32(1.25) if i % 2 == 0 else Float32(5.25)
            var actual = hc.unsafe_ptr().unsafe_load(i)
            if bitcast[DType.uint32](actual) != bitcast[DType.uint32](expected):
                raise Error("constant/mixed tree apply lost leaf values or bias")
            print("ORDERED_BITS constant_model", mixed, i, bitcast[DType.uint32](actual))
    _ = hx^


def whole_loop_cursor_check(ctx: DeviceContext) raises:
    """Replay each actual shared tree with an independent Float64 leaf oracle.

    Structure is allowed to use quality targets. Conditional on that actual
    structure, each fold must use only its own prefix and previous cursor.
    """
    comptime N = 16
    var counts: List[Int] = [1]
    var layout = build_layout(counts)
    var hx = ctx.enqueue_create_host_buffer[DType.uint32](N)
    var x = ctx.enqueue_create_buffer[DType.uint32](N)
    var y = List[Float32]()
    var w = List[Float32]()
    var p = List[UInt32]()
    for i in range(N):
        hx.unsafe_ptr().unsafe_store(i, UInt32(i % 2) << layout.features[0].shift)
        y.append(Float32(i - 4))
        w.append(Float32(1 + i % 3))
        p.append(UInt32((i * 5 + 3) % N))
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    var fit = fit_ordered_rmse(ctx, layout, x, y, w, p, 2, 1, 1, Float32(0.25), Float32(1))
    if len(fit.folds) != 3:
        raise Error("ordered small-fixture prefix fold inventory changed")
    for f in range(len(fit.folds) + 1):
        var estimate = N
        var size = N
        if f < len(fit.folds):
            estimate = fit.folds[f].estimate_samples.right
            size = fit.folds[f].quality_evaluate_samples.right
            if estimate >= size:
                raise Error("ordered fold has no quality-only tail")
        var expected = List[Float64]()
        for _ in range(size):
            expected.append(Float64(0))
        for t in range(len(fit.model.weak_models)):
            ref tree = fit.model.weak_models[t]
            var leaf_ids = List[Int]()
            var n_leaves = 1 << len(tree.structure.splits)
            for pos in range(size):
                var leaf = 0
                for level in range(len(tree.structure.splits)):
                    var split = tree.structure.splits[level]
                    if split.feature_id != 0 or split.split_type != 1:
                        raise Error("ordered fixture got unexpected split kind")
                    if Int(p[pos]) % 2 > Int(split.bin_idx):
                        leaf |= 1 << level
                leaf_ids.append(leaf)
            var values = List[Float64]()
            for leaf in range(n_leaves):
                var total = Float64(0)
                var mass = Float64(0)
                for pos in range(estimate):
                    if leaf_ids[pos] == leaf:
                        var row = Int(p[pos])
                        mass += Float64(w[row])
                        total += Float64(w[row]) * (Float64(y[row]) - expected[pos])
                values.append(Float64(0.25) * total / (mass + Float64(1)))
            for pos in range(size):
                expected[pos] += values[leaf_ids[pos]]
        var hc = ctx.enqueue_create_host_buffer[DType.float32](size)
        if f < len(fit.folds):
            ctx.enqueue_copy(dst_ptr=hc.unsafe_ptr(), src_buf=fit.fold_cursors[f])
        else:
            ctx.enqueue_copy(dst_ptr=hc.unsafe_ptr(), src_buf=fit.estimation_cursor)
        ctx.synchronize()
        for pos in range(size):
            var actual = hc.unsafe_ptr().unsafe_load(pos)
            if not isfinite(actual) or abs(Float64(actual) - expected[pos]) > Float64(0.00001):
                raise Error("ordered whole-loop cursor differs from prefix-only Float64 replay")
            print("ORDERED_BITS fold_cursor", f, pos, bitcast[DType.uint32](actual))
    _ = hx^


def main() raises:
    print("ORDERED NUMERIC MODE", numeric_mode_name())
    if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("ordered RMSE certification requires MOJOLEARN_NUMERIC_IDENTICAL")
    var ctx = DeviceContext()
    whole_loop_cursor_check(ctx)
    constant_model_check(ctx)
    var base = prefix_case(ctx, False, False)
    var poisoned = prefix_case(ctx, True, False)
    var leaked = prefix_case(ctx, True, True)
    var differs = False
    for i in range(len(base)):
        if bitcast[DType.uint32](base[i]) != bitcast[DType.uint32](poisoned[i]):
            raise Error("ordered prefix changed when only quality labels changed")
        differs = differs or base[i] != leaked[i]
        print("ORDERED_BITS prefix", i, bitcast[DType.uint32](base[i]))
    if not differs:
        raise Error("ordered leakage negative control was insensitive")
    var zero = prefix_case(ctx, False, False, True)
    var zero_full = prefix_case(ctx, False, True, True)
    var zero_control_differs = False
    for i in range(len(zero)):
        zero_control_differs = zero_control_differs or zero[i] != zero_full[i]
        print("ORDERED_BITS zero_prefix", i, bitcast[DType.uint32](zero[i]))
    if not zero_control_differs:
        raise Error("ordered zero-mass prefix control failed to see positive tail mass")

    comptime N = 32
    var x = List[Float32]()
    var y = List[Float32]()
    var w = List[Float32]()
    var p = List[UInt32]()
    for i in range(N):
        x.append(Float32(i % 8))
        y.append(Float32(2 * (i % 8) - 5))
        w.append(Float32(1 + i % 3))
        p.append(UInt32((i * 13 + 7) % N))
    var fit = train_ordered_rmse(
        ctx, x, y, N, 1, p, n_estimators=3, max_depth=2,
        border_count=7, learning_rate=Float32(0.25),
        l2_leaf_reg=Float32(1), sample_weight=w,
    )
    if len(fit.model.weak_models) != 3:
        raise Error("ordered fit did not export three trees")
    var prediction = predict_floats(ctx, fit, x, N)
    var before = Float64(0)
    var after = Float64(0)
    for i in range(N):
        if not isfinite(prediction[i]):
            raise Error("ordered full-fit prediction not finite")
        before += Float64(w[i]) * Float64(y[i]) * Float64(y[i])
        var residual = Float64(y[i]) - Float64(prediction[i])
        after += Float64(w[i]) * residual * residual
        print("ORDERED_BITS prediction", i, bitcast[DType.uint32](prediction[i]))
    if not (after < before * Float64(0.8)):
        raise Error("ordered fit did not reduce weighted squared error")
    for t in range(len(fit.model.weak_models)):
        ref tree = fit.model.weak_models[t]
        for s in range(len(tree.structure.splits)):
            var split = tree.structure.splits[s]
            print("ORDERED_BITS split", t, s, split.feature_id, split.bin_idx, split.split_type)
        for leaf in range(len(tree.leaf_values)):
            print("ORDERED_BITS leaf", t, leaf, bitcast[DType.uint32](tree.leaf_values[leaf]))

    var bad_p = p.copy()
    bad_p[0] = bad_p[1]
    var refused = False
    try:
        var invalid = train_ordered_rmse(ctx, x, y, N, 1, bad_p, n_estimators=1)
    except e:
        refused = String(e).find("bijection") >= 0
    if not refused:
        raise Error("ordered fit accepted duplicate permutation ids")
    print("ORDERED RMSE PASS: prefix isolation, leakage control, zero-mass prefix L2=0, constant/mixed model apply, two cursor updates, weighted three-tree fit/apply, invalid permutation refusal")
