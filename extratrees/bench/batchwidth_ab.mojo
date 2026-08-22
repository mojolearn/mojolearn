"""DEVIATION 213's A/B: the merged frontier's batch bound, swept in-process.

    pixi run mojo run -I . extratrees/bench/batchwidth_ab.mojo \
        <data_dir> <name> <n_rows> <n_features> <n_classes> <trees> <depth> \
        [reps] [regression] [mf_spec]

cuML pops at `max_batch_size = 4096` -- a bound tuned for ONE tree's
frontier on their allocator. DEVIATION 211 merged every tree's frontier into
the same queue cycle, so at sklearn's default 100 trees a deep level holds
tens of thousands of nodes and the 4096 bound splits it into dozens of
batches, each paying its own staging, launches, readback and synchronize.
The batch width is a SCHEDULING parameter contracted not to change the tree
(`NodeQueue.pop`; `device_batched_check` gates it), so widening it for the
merged forest removes fixed costs outright -- the lever class that has paid
in this lane -- at the price of a workspace that scales with the bound.

Arms: 4096 (cuML's default, the shipping value), 16384, 32768 -- alternated
inside one process, three reps, digests asserted identical across ALL arms
and reps, which also live-tests the batch-width contract at full covtype
scale. Whatever this file says decides the merged trainer's default.
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
from extratrees.ported.decisiontree.decisiontree import (
    CRITERION_MSE,
    DecisionTreeParams,
)
from extratrees.ported.decisiontree.batched_levelalgo.builder import (
    train_forest_classification_device,
    train_forest_regression_device,
    upload_dataset,
    DeviceDataset,
)
from extratrees.ported.decisiontree.flatnode import TreeMetaDataNode
from extratrees.ported.randomforest.randomforest import class_ids_for


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


def run_arm(
    ctx: DeviceContext,
    mut dev: DeviceDataset,
    base: DecisionTreeParams,
    batch: Int,
    n_trees: Int,
    seed: UInt64,
    regression: Bool,
    scale: Float64,
) raises -> List[TreeMetaDataNode[DType.float32]]:
    var params = base
    params.max_batch_size = Int32(batch)
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
            "usage: mojo run -I . extratrees/bench/batchwidth_ab.mojo"
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

    var seed = UInt64(0x213AB)
    var scale = Float64(1.0)
    var dev: DeviceDataset
    var n_feat = n_features
    if regression:
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

    var base = DecisionTreeParams()
    base.max_depth = Int32(depth)
    if regression:
        base.split_criterion = CRITERION_MSE
    base.max_features = count_to_ratio(
        resolve_max_features(max_features_code(mf_spec, n_feat), 0.0, n_feat),
        n_feat,
    )

    var widths = [4096, 16384, 32768]

    print(
        "[widthab]",
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
        "[widthab] max_batch_size arms:",
        widths[0],
        widths[1],
        widths[2],
        "-- alternated in ONE process,",
        reps,
        "reps. Digests asserted equal (the batch-width contract, live).",
    )

    var digest0 = UInt64(0)
    for rep in range(reps):
        var ms = List[Float64]()
        var d = List[UInt64]()
        for w in range(len(widths)):
            var t0 = perf_counter_ns()
            var f = run_arm(
                ctx, dev, base, widths[w], trees, seed, regression, scale
            )
            var t1 = perf_counter_ns()
            ms.append(Float64(t1 - t0) / 1e6)
            d.append(forest_digest(f))
        for w in range(1, len(widths)):
            if d[w] != d[0]:
                raise Error(
                    "batch width moved the forest -- the scheduling"
                    " contract is broken and no ratio may be published"
                )
        if rep == 0:
            digest0 = d[0]
        elif d[0] != digest0:
            raise Error("a rep diverged from rep 0; the run is not sound")
        print(
            "  rep",
            rep,
            ": 4096",
            ms[0],
            "ms, 16384",
            ms[1],
            "ms (",
            ms[0] / ms[1],
            "x ), 32768",
            ms[2],
            "ms (",
            ms[0] / ms[2],
            "x )",
        )
    print("[widthab] digest", hex(digest0), "identical across all arms")
