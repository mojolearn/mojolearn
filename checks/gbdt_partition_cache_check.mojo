# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Full-model A/B witness for unchanged-leaf partition statistics caching.

Run tools/gbdt_partition_cache_ab.py: reference and cached builds must match
all tree fields, prediction bits, loss bits and quantization borders. The
mixed-policy direct fixture exercises binary/half-byte/one-byte histograms;
weighted boosting exercises repeated trees and workspace reuse. A negative
control omits right-child invalidation and must move a model fingerprint.
"""
from std.memory import bitcast
from std.testing import assert_equal
from checks.numerics import numeric_mode_name
from checks.depthwise_check import Fixture, default_options
from checks.lossguide_check import fit_policy, lossguide_options
from gbdt.models.non_symmetric_tree import TNonSymmetricTree
from gbdt.train import train, predict_floats
from max.gpu.host import DeviceContext


def fold(mut h: UInt64, v: UInt64):
    h = (h ^ v) * UInt64(1099511628211)


def tree_hash(mut h: UInt64, tree: TNonSymmetricTree):
    fold(h, UInt64(tree.dim))
    fold(h, UInt64(len(tree.model_structure.nodes)))
    for node in tree.model_structure.nodes:
        fold(h, UInt64(node.feature_id))
        fold(h, UInt64(node.bin))
        fold(h, UInt64(node.left_subtree))
        fold(h, UInt64(node.right_subtree))
    for value in tree.model_structure.split_types:
        fold(h, UInt64(value))
    for value in tree.leaf_values:
        fold(h, UInt64(bitcast[DType.uint32](value)))
    for value in tree.leaf_weights:
        fold(h, bitcast[DType.uint64](value))


def main() raises:
    print("numeric_mode", numeric_mode_name())
    var ctx = DeviceContext()
    var fx = Fixture(ctx.copy())
    for depth in [1, 4, 7]:
        for policy in ["Depthwise", "Lossguide"]:
            var opts = default_options(depth)
            if policy == "Lossguide":
                opts = lossguide_options(depth, 31)
            opts.random_strength = Float32(0.25)
            var model = fit_policy(fx, opts)
            var h = UInt64(14695981039346656037)
            tree_hash(h, model)
            print("fingerprint", "mixed", policy, depth, h)
    _ = fx^

    var rows = 4099
    var columns = 9
    var x = List[Float32]()
    var weights = List[Float32]()
    for f in range(columns):
        for r in range(rows):
            var bits = UInt32(r * 2654435761 + f * 40503 + 0x1234567)
            bits ^= bits << 13
            bits ^= bits >> 17
            bits ^= bits << 5
            x.append(Float32(bits % UInt32(2 if f < 3 else 64)) / 32 - 1)
    for r in range(rows):
        weights.append(Float32(0) if r % 13 == 0 else Float32(1 + r % 7) / 7)
    for policy in ["Depthwise", "Lossguide"]:
        for loss in ["RMSE", "Logloss"]:
            var y = List[Float32]()
            for r in range(rows):
                var value = x[3 * rows + r] * x[4 * rows + r] + x[6 * rows + r] * 0.3
                y.append((Float32(1) if value > 0 else Float32(0)) if loss == "Logloss" else value)
            var expected = UInt64(0)
            for repeat in range(2):
                var model = train(ctx, x, y, rows, columns,
                    border_count=32, n_estimators=5, max_depth=6,
                    grow_policy=policy, max_leaves=31 if policy == "Lossguide" else -1,
                    min_data_in_leaf=3, loss=loss, sample_weight=weights,
                    random_seed=202109, leaf_estimation_iterations=3)
                var predictions = predict_floats(ctx, model, x, rows)
                var h = UInt64(14695981039346656037)
                fold(h, bitcast[DType.uint64](model.model.bias))
                fold(h, UInt64(len(model.model.non_symmetric_models)))
                for tree in model.model.non_symmetric_models:
                    tree_hash(h, tree)
                for borders in model.borders:
                    for value in borders:
                        fold(h, UInt64(bitcast[DType.uint32](value)))
                for value in model.losses:
                    fold(h, bitcast[DType.uint64](value))
                for value in predictions:
                    fold(h, UInt64(bitcast[DType.uint32](value)))
                if repeat == 0:
                    expected = h
                else:
                    assert_equal(h, expected)
                print("fingerprint", "boosted", policy, loss, repeat, h)
    _ = ctx^
