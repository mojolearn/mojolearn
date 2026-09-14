# SPDX-License-Identifier: Apache-2.0
"""Cloud-only full IsolationForest model buffer comparison."""
from std.os import getenv, setenv
from std.memory import bitcast
from max.gpu.host import DeviceContext
from core.identity_trace import IdentityTrace
from isolation_forest.impl.isolation_forest import IsolationForest, IsolationForestModel, IF_params, read_i32, read_f32


def main() raises:
    if String(getenv("RUNPOD_POD_ID")) == "":
        raise Error("cloud host required")
    var ctx = DeviceContext()
    var x = List[Float32]()
    for i in range(257 * 7):
        x.append(Float32((i * 37) % 251) / Float32(256))
    for bootstrap in range(2):
        for features in range(3, 8, 4):
            var params = IF_params(5, 33, 5, features, bootstrap == 1, 42)
            var estimator = IsolationForest(params)
            var one = IsolationForestModel(ctx)
            var many = IsolationForestModel(ctx)
            var trace = IdentityTrace.disabled()
            if not setenv("MOJOLEARN_IFOREST_DEVICE_COUNT", "1", True):
                raise Error("setenv failed")
            estimator.fit(ctx, x, 257, 7, one, trace)
            if not setenv("MOJOLEARN_IFOREST_DEVICE_COUNT", "2", True):
                raise Error("setenv failed")
            estimator.fit(ctx, x, 257, 7, many, trace)
            var nodes = 5 * one.max_nodes_per_tree
            var a = read_f32(ctx, one.node_threshold, nodes)
            var b = read_f32(ctx, many.node_threshold, nodes)
            for i in range(nodes):
                if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
                    raise Error("threshold mismatch")
            var ai = List[List[Int32]]()
            var bi = List[List[Int32]]()
            ai.append(read_i32(ctx, one.node_feature, nodes))
            bi.append(read_i32(ctx, many.node_feature, nodes))
            ai.append(read_i32(ctx, one.node_left, nodes))
            bi.append(read_i32(ctx, many.node_left, nodes))
            ai.append(read_i32(ctx, one.node_right, nodes))
            bi.append(read_i32(ctx, many.node_right, nodes))
            ai.append(read_i32(ctx, one.global_tree_offsets, 5))
            bi.append(read_i32(ctx, many.global_tree_offsets, 5))
            ai.append(read_i32(ctx, one.global_tree_n_nodes, 5))
            bi.append(read_i32(ctx, many.global_tree_n_nodes, 5))
            ai.append(read_i32(ctx, one.global_tree_max_depth, 5))
            bi.append(read_i32(ctx, many.global_tree_max_depth, 5))
            ai.append(read_i32(ctx, one.global_feature_indices, 5 * features))
            bi.append(read_i32(ctx, many.global_feature_indices, 5 * features))
            for field in range(len(ai)):
                for i in range(len(ai[field])):
                    if ai[field][i] != bi[field][i]:
                        raise Error("model integer mismatch " + String(field) + "/" + String(i))
            print("PASS complete forest buffers", bootstrap, features)
            _ = one^
            _ = many^
    ctx.synchronize()
