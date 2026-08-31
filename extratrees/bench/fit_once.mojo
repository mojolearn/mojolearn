# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""ONE merged device fit, for tracing under Apple Instruments -- or, with
`phases`, the lane's MICRO-STEP clock.

    pixi run mojo build -I . extratrees/bench/fit_once.mojo -o build/et_fit_once
    build/et_fit_once <data_dir> <name> <n_rows> <n_features> <n_classes> \
        <trees> <depth> [regression] [mf_spec] [phases]

Without `phases`: exactly one `train_forest_*_device` call -- the shipping
merged path -- printing wall time and node count; the TARGET of
`profile_et_metal.py`, whose numbers come from the GPU's own timeline.

With `phases`: TWO fits back to back -- first the shipping (unclocked) one,
then the same fit through `train_forest_*_device_timed` with an enabled
`PhaseClock`, which synchronizes at every phase boundary. The clocked fit is
a SERIALIZED version of the program (the PhaseClock docstring says why), so
the report prints both totals and their ratio: the gap is the measurement's
own distortion, stated next to the numbers it distorts. The per-phase table
is what DEVIATIONS 212/213 did not have and guessed without.
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
    train_forest_classification_device_timed,
    train_forest_regression_device,
    train_forest_regression_device_timed,
    upload_dataset,
    DeviceDataset,
    FOREST_ROW_SLOT_CAP,
    FOREST_SAB_NONE,
    N_PHASES,
    PhaseClock,
)
from extratrees.derived.randomforest.randomforest import class_ids_for


def main() raises:
    var args = argv()
    if len(args) < 8:
        raise Error(
            "usage: et_fit_once <data_dir> <name> <n_rows> <n_features>"
            " <n_classes> <trees> <depth> [regression] [mf_spec]"
        )
    var data_dir = String(args[1])
    var name = String(args[2])
    var n_rows = Int(String(args[3]))
    var n_features = Int(String(args[4]))
    var n_classes = Int(String(args[5]))
    var trees = Int(String(args[6]))
    var depth = Int(String(args[7]))
    var regression = False
    var phases = False
    var mf_spec = String("sqrt")
    for i in range(8, len(args)):
        var a = String(args[i])
        if a == String("regression"):
            regression = True
        elif a == String("phases"):
            phases = True
        elif not all_digits(a):
            mf_spec = a

    var ctx = DeviceContext()
    var yfull = read_f32(data_dir + "/" + name + "_y.f32")
    var total_rows = len(yfull)
    var seed = UInt64(0x0F17)
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

    var params = DecisionTreeParams()
    params.max_depth = Int32(depth)
    if regression:
        params.split_criterion = CRITERION_MSE
    params.max_features = count_to_ratio(
        resolve_max_features(max_features_code(mf_spec, n_feat), 0.0, n_feat),
        n_feat,
    )
    var tree_ids = List[Int32]()
    for t in range(trees):
        tree_ids.append(Int32(t))

    if phases:
        # WARMUP, unmeasured: the process's FIRST fit pays every kernel's
        # pipeline creation (measured at several hundred ms), and without
        # this fit the "distortion" ratio below compares a cold program to a
        # warm one.
        if regression:
            _ = train_forest_regression_device(
                ctx, dev, scale, params, tree_ids, seed
            )
        else:
            _ = train_forest_classification_device(
                ctx, dev, params, tree_ids, seed
            )

    var t0 = perf_counter_ns()
    var nodes = 0
    if regression:
        var f = train_forest_regression_device(
            ctx, dev, scale, params, tree_ids, seed
        )
        for t in range(len(f)):
            nodes += f[t].num_nodes()
    else:
        var f = train_forest_classification_device(
            ctx, dev, params, tree_ids, seed
        )
        for t in range(len(f)):
            nodes += f[t].num_nodes()
    var t1 = perf_counter_ns()
    var plain_ms = Float64(t1 - t0) / 1e6
    print("[fit_once]", trees, "trees,", nodes, "nodes,", plain_ms, "ms")

    if phases:
        var clock = PhaseClock(True)
        var t2 = perf_counter_ns()
        var nodes2 = 0
        if regression:
            var f = train_forest_regression_device_timed(
                ctx, dev, scale, params, tree_ids, seed, FOREST_SAB_NONE,
                FOREST_ROW_SLOT_CAP, clock,
            )
            for t in range(len(f)):
                nodes2 += f[t].num_nodes()
        else:
            var f = train_forest_classification_device_timed(
                ctx, dev, params, tree_ids, seed, FOREST_SAB_NONE,
                FOREST_ROW_SLOT_CAP, clock,
            )
            for t in range(len(f)):
                nodes2 += f[t].num_nodes()
        var t3 = perf_counter_ns()
        if nodes2 != nodes:
            raise Error("the clocked fit built a different forest")
        var timed_ms = Float64(t3 - t2) / 1e6
        var sum_ns = Int64(0)
        for p in range(N_PHASES):
            sum_ns += clock.ns[p]
        print(
            "[phases] clocked total",
            timed_ms,
            "ms vs unclocked",
            plain_ms,
            "ms -- serialization distortion",
            timed_ms / plain_ms,
            "x; phase sum",
            Float64(sum_ns) / 1e6,
            "ms",
        )
        for p in range(N_PHASES):
            var ms = Float64(clock.ns[p]) / 1e6
            print(
                "  ",
                clock.phase_name(p),
                ": ",
                ms,
                "ms (",
                100.0 * Float64(clock.ns[p]) / Float64(sum_ns),
                "% )",
            )
