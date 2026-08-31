# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""DEVIATION 211's A/B: the merged-frontier forest against the per-tree loop.

    pixi run mojo run -I . extratrees/bench/batched_ab.mojo \
        <data_dir> <name> <n_rows> <n_features> <n_classes> <trees> <depth> \
        [reps] [regression] [mf_spec]

BOTH ARMS LIVE IN ONE BINARY, so this alternates INSIDE one process -- the
discipline DEVIATION 208 paid to learn (its first, wrong, number came from
two windows). Arm A is the per-tree loop the fit used to be: the dataset
resident once, then one-tree calls of the forest trainer, each with its own
workspace and row buffer, exactly the old `fit_*_device` body. Arm B is one
merged call. The trees are bit-identical between the arms
(`device_batched_check` holds that with no tolerance), so the ratio is pure
schedule: launches, readbacks and synchronize points divided by the trees in
flight, against the same kernels doing the same work.

WHAT IS TIMED: the forest build only. Upload, quantization and the digest
are outside the timed region on both arms; the DIGEST is computed on both
arms' forests and asserted equal, so a run whose arms diverged cannot
publish a ratio.
"""

from max.gpu.host import DeviceContext
from std.sys import argv
from std.time import perf_counter_ns

from extratrees.bench.bench_data import (
    all_digits,
    count_to_ratio,
    dense_class_ids,
    max_features_code,
    read_column_prefix,
    read_f32,
)
from extratrees.estimator import resolve_max_features, quantize_labels
from extratrees.derived.decisiontree.decisiontree import (
    CRITERION_MSE,
    DecisionTreeParams,
)
from extratrees.derived.decisiontree.batched_levelalgo.builder import (
    train_forest_classification_device,
    train_forest_regression_device,
    upload_dataset,
    DeviceDataset,
)
from extratrees.derived.decisiontree.flatnode import TreeMetaDataNode
from extratrees.derived.randomforest.randomforest import class_ids_for


def forest_digest(trees: List[TreeMetaDataNode[DType.float32]]) -> UInt64:
    var h = UInt64(0xCBF29CE484222325)

    def mix(mut h: UInt64, v: UInt64):
        h = (h ^ v) * UInt64(0x100000001B3)

    for t in range(len(trees)):
        mix(h, UInt64(trees[t].num_nodes()))
        for i in range(trees[t].num_nodes()):
            var n = trees[t].sparsetree[i]
            mix(h, UInt64(Int(n.ColumnId()) & 0xFFFFFFFF))
            mix(h, UInt64(Int(n.QueryValue().to_bits()) & 0xFFFFFFFF))
            mix(h, UInt64(Int(n.LeftChildId()) & 0xFFFFFFFF))
            mix(h, UInt64(Int(n.InstanceCount()) & 0xFFFFFFFF))
        for i in range(len(trees[t].vector_leaf)):
            mix(h, UInt64(Int(trees[t].vector_leaf[i].to_bits()) & 0xFFFFFFFF))
    return h


def serial_arm(
    ctx: DeviceContext,
    mut dev: DeviceDataset,
    params: DecisionTreeParams,
    n_trees: Int,
    seed: UInt64,
    regression: Bool,
    scale: Float64,
) raises -> List[TreeMetaDataNode[DType.float32]]:
    var out = List[TreeMetaDataNode[DType.float32]]()
    for t in range(n_trees):
        var one = List[Int32]()
        one.append(Int32(t))
        if regression:
            var built = train_forest_regression_device(
                ctx, dev, scale, params, one, seed
            )
            out.append(built[0].copy())
        else:
            var built = train_forest_classification_device(
                ctx, dev, params, one, seed
            )
            out.append(built[0].copy())
    return out^


def merged_arm(
    ctx: DeviceContext,
    mut dev: DeviceDataset,
    params: DecisionTreeParams,
    n_trees: Int,
    seed: UInt64,
    regression: Bool,
    scale: Float64,
) raises -> List[TreeMetaDataNode[DType.float32]]:
    var ids = List[Int32]()
    for t in range(n_trees):
        ids.append(Int32(t))
    if regression:
        return train_forest_regression_device(
            ctx, dev, scale, params, ids, seed
        )
    return train_forest_classification_device(ctx, dev, params, ids, seed)


def main() raises:
    var args = argv()
    if len(args) < 8:
        raise Error(
            "usage: mojo run -I . extratrees/bench/batched_ab.mojo"
            " <data_dir> <name> <n_rows> <n_features> <n_classes> <trees>"
            " <depth> [reps] [regression] [mf_spec]"
        )
    var data_dir = String(args[1])
    var name = String(args[2])
    var n_rows = Int(String(args[3]))
    var n_features = Int(String(args[4]))
    var n_classes = Int(String(args[5]))
    var trees = Int(String(args[6]))
    var depth = Int(String(args[7]))
    var reps = 3
    var regression = False
    var mf_spec = String("sqrt")
    for i in range(8, len(args)):
        var a = String(args[i])
        if a == String("regression"):
            regression = True
        elif all_digits(a):
            reps = Int(a)
        else:
            mf_spec = a

    var ctx = DeviceContext()
    var yfull = read_f32(data_dir + "/" + name + "_y.f32")
    var total_rows = len(yfull)
    if n_rows > total_rows:
        raise Error("label file has only " + String(total_rows) + " rows")

    var seed = UInt64(0x211AB)
    var scale = Float64(1.0)
    var dev: DeviceDataset
    var n_feat = n_features
    if regression:
        # Column 0 is the target, as in sklearn_interleaved.
        n_feat = n_features - 1
        var x = read_column_prefix(
            data_dir + "/" + name + "_Xcol.f32", total_rows, n_rows, n_features
        )
        var target = List[Float32](length=n_rows, fill=Float32(0.0))
        for r in range(n_rows):
            target[r] = x[r]
        var xcols = List[Float32](length=n_rows * n_feat, fill=Float32(0.0))
        for c in range(n_feat):
            for r in range(n_rows):
                xcols[c * n_rows + r] = x[(c + 1) * n_rows + r]
        var ql = quantize_labels(target, Int32(n_rows))
        scale = ql[1]
        dev = upload_dataset(
            ctx, xcols, ql[0], Int32(n_rows), Int32(n_feat), 1
        )
    else:
        var x = read_column_prefix(
            data_dir + "/" + name + "_Xcol.f32", total_rows, n_rows, n_features
        )
        var labels = dense_class_ids(yfull, n_rows, n_classes)
        var lf = List[Float32]()
        for r in range(n_rows):
            lf.append(Float32(Int(labels[r])))
        var ids = class_ids_for(lf, Int32(n_rows), Int32(n_classes))
        dev = upload_dataset(
            ctx, x, ids, Int32(n_rows), Int32(n_feat), Int32(n_classes)
        )

    var params = DecisionTreeParams()
    params.max_depth = Int32(depth)
    if regression:
        params.split_criterion = CRITERION_MSE
    params.max_features = count_to_ratio(
        resolve_max_features(max_features_code(mf_spec, n_feat), 0.0, n_feat),
        n_feat,
    )

    print(
        "[ab]",
        ctx.name(),
        "--",
        name,
        n_rows,
        "rows x",
        n_feat,
        "features,",
        trees,
        "trees, depth",
        depth,
        ", max_features",
        mf_spec,
        ", regression" if regression else ", classification",
    )
    print(
        "[ab] serial = one-tree forest calls (the old fit body);"
        " merged = DEVIATION 211. Alternated in ONE process,",
        reps,
        "reps. Digests asserted equal.",
    )

    var digest0 = UInt64(0)
    for rep in range(reps):
        var t0 = perf_counter_ns()
        var a = serial_arm(ctx, dev, params, trees, seed, regression, scale)
        var t1 = perf_counter_ns()
        var b = merged_arm(ctx, dev, params, trees, seed, regression, scale)
        var t2 = perf_counter_ns()
        var da = forest_digest(a)
        var db = forest_digest(b)
        if da != db:
            raise Error("the arms diverged; no ratio may be published")
        if rep == 0:
            digest0 = da
        elif da != digest0:
            raise Error("a rep diverged from rep 0; the run is not sound")
        var ms_serial = Float64(t1 - t0) / 1e6
        var ms_merged = Float64(t2 - t1) / 1e6
        print(
            "  rep",
            rep,
            ": serial",
            ms_serial,
            "ms, merged",
            ms_merged,
            "ms, ratio",
            ms_serial / ms_merged,
            "x",
        )
    print("[ab] digest", hex(digest0), "identical on every rep and both arms")
