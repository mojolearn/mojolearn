# SPDX-License-Identifier: Apache-2.0
"""Cloud-only resident forest buffers, root allocation and ordered path checks."""
from std.os import getenv, setenv
from std.memory import bitcast
from max.gpu.host import DeviceContext
from core.identity_trace import IdentityTrace
from isolation_forest.impl.isolation_forest import IsolationForest, IsolationForestModel, IF_params, read_i32, read_f32, path_lengths, score_samples
from metrics.checks.device_io import upload_f32, upload_i32


def check_paths(ctx: DeviceContext, one: IsolationForestModel,
    many: IsolationForestModel, x: List[Float32], rows: Int, columns: Int,
) raises:
    var a = path_lengths(ctx, one, x, rows, columns)
    var b = path_lengths(ctx, many, x, rows, columns)
    var trace = IdentityTrace.disabled()
    var sa = score_samples(ctx, one, x, rows, columns, trace)
    var sb = score_samples(ctx, many, x, rows, columns, trace)
    for i in range(rows):
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            raise Error("ordered path mismatch " + String(i))
        if bitcast[DType.uint32](sa[i]) != bitcast[DType.uint32](sb[i]):
            raise Error("score mismatch " + String(i))


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
            # Exercise release of a previously fitted full root model too.
            estimator.fit(ctx, x, 257, 7, many, trace)
            if not setenv("MOJOLEARN_IFOREST_DEVICE_COUNT", "2", True):
                raise Error("setenv failed")
            estimator.fit(ctx, x, 257, 7, many, trace)
            if len(many.shards) != 2 or len(one.shards) != 0:
                raise Error("resident owner route mismatch")
            if (len(many.node_feature) != 1 or len(many.node_threshold) != 1
                or len(many.node_left) != 1 or len(many.node_right) != 1
                or len(many.global_tree_offsets) != 1
                or len(many.global_tree_n_nodes) != 1
                or len(many.global_tree_max_depth) != 1
                or len(many.global_feature_indices) != 1):
                raise Error("full root allocation retained")
            var nodes = 5 * one.max_nodes_per_tree
            var a = read_f32(ctx, one.node_threshold, nodes)
            var ai = List[List[Int32]]()
            ai.append(read_i32(ctx, one.node_feature, nodes))
            ai.append(read_i32(ctx, one.node_left, nodes))
            ai.append(read_i32(ctx, one.node_right, nodes))
            ai.append(read_i32(ctx, one.global_tree_offsets, 5))
            ai.append(read_i32(ctx, one.global_tree_n_nodes, 5))
            ai.append(read_i32(ctx, one.global_tree_max_depth, 5))
            ai.append(read_i32(ctx, one.global_feature_indices, 5 * features))
            var total_nodes = 0
            var total_trees = 0
            for rank in range(len(many.shards)):
                ref shard = many.shards[rank]
                if shard.first != total_trees or shard.count <= 0:
                    raise Error("noncanonical tree owners")
                var local_nodes = shard.count * many.max_nodes_per_tree
                if len(shard.node_threshold) != local_nodes:
                    raise Error("owner allocation mismatch")
                var b = read_f32(shard.ctx, shard.node_threshold, local_nodes)
                for i in range(local_nodes):
                    if bitcast[DType.uint32](a[total_nodes+i]) != bitcast[DType.uint32](b[i]):
                        raise Error("threshold mismatch")
                var bi = List[List[Int32]]()
                bi.append(read_i32(shard.ctx, shard.node_feature, local_nodes))
                bi.append(read_i32(shard.ctx, shard.node_left, local_nodes))
                bi.append(read_i32(shard.ctx, shard.node_right, local_nodes))
                bi.append(read_i32(shard.ctx, shard.global_tree_offsets, shard.count))
                bi.append(read_i32(shard.ctx, shard.global_tree_n_nodes, shard.count))
                bi.append(read_i32(shard.ctx, shard.global_tree_max_depth, shard.count))
                bi.append(read_i32(shard.ctx, shard.global_feature_indices, shard.count*features))
                for field in range(len(ai)):
                    var first = total_nodes if field < 3 else total_trees
                    if field == 6:
                        first *= features
                    for i in range(len(bi[field])):
                        var value = bi[field][i]
                        if field == 3:
                            value += Int32(total_nodes)
                        if ai[field][first+i] != value:
                            raise Error("model integer mismatch " + String(field) + "/" + String(i))
                total_nodes += local_nodes
                total_trees += shard.count
            if total_nodes != nodes or total_trees != 5:
                raise Error("incomplete resident forest")
            check_paths(ctx, one, many, x, 257, 7)
            print("PASS complete resident buffers and paths", bootstrap, features)

            # Deliberate arithmetic witness: legal leaf payload storage with
            # a large first value and four ones. The original serial fold
            # loses all four ones; summing each owner's paths first does not.
            var leaf_features = List[Int32](length=nodes, fill=Int32(-1))
            var leaf_values = List[Float32](length=nodes, fill=Float32(1.0))
            leaf_values[0] = Float32(16777216.0)
            one.node_feature = upload_i32(ctx, leaf_features)
            one.node_threshold = upload_f32(ctx, leaf_values)
            for rank in range(len(many.shards)):
                ref shard = many.shards[rank]
                var local_nodes = shard.count * many.max_nodes_per_tree
                var fs = List[Int32](length=local_nodes, fill=Int32(-1))
                var vs = List[Float32](length=local_nodes, fill=Float32(1.0))
                if rank == 0:
                    vs[0] = Float32(16777216.0)
                shard.node_feature = upload_i32(shard.ctx, fs)
                shard.node_threshold = upload_f32(shard.ctx, vs)
            check_paths(ctx, one, many, x, 257, 7)
            print("PASS serial accumulator witness", bootstrap, features)
            _ = one^
            _ = many^
    ctx.synchronize()
