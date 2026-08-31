# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Does the DEVICE forest reproduce itself at benchmark scale?

`device_forest_check` asserts determinism, and it asserts it on fixtures of a
few thousand rows in a process that fits one forest. The benchmark fits
fifteen forests of 581,012 rows in one process, and three of thirty cells came
back with a DIFFERENT NODE COUNT for a seed that had already produced another
one -- 18,020 where 29,440 was reproduced twice, 16,496 where 22,884 was, 5,802
where 5,818 was. A different node count from the same seed is not a timing
artifact.

So this refits the SAME configuration N times in ONE process and digests every
tree. The digest is over `(colid, quesval bits, left_child_id,
instance_count)` for every node, which is what `device_tree_check` compares
cell by cell -- a node count alone would miss two trees that differ in a
threshold while agreeing in shape.

Anything but N identical digests is the defect, reproduced.
"""

from max.gpu.host import DeviceContext
from std.python import Python
from std.sys import argv
from std.time import perf_counter_ns

from extratrees.bench.bench_data import (
    dense_class_ids,
    max_features_code,
    read_column_prefix,
    read_f32,
)
from extratrees.estimator import (
    ExtraTreesConfig,
    fit_extra_trees_classifier_device,

)
from extratrees.derived.randomforest.randomforest import Forest


def digest(forest: Forest) raises -> UInt64:
    """FNV-1a over every node of every tree."""
    var h = UInt64(0xCBF29CE484222325)
    for t in range(len(forest.trees)):
        ref tree = forest.trees[t]
        for i in range(tree.num_nodes()):
            var n = tree.sparsetree[i]
            var words = List[UInt64]()
            words.append(UInt64(Int(n.ColumnId()) & 0xFFFFFFFF))
            words.append(UInt64(n.QueryValue().to_bits()))
            words.append(UInt64(Int(n.LeftChildId()) & 0xFFFFFFFF))
            words.append(UInt64(Int(n.InstanceCount()) & 0xFFFFFFFF))
            for w in range(len(words)):
                var v = words[w]
                for b in range(8):
                    h = (h ^ ((v >> UInt64(b * 8)) & 0xFF)) * 0x100000001B3
    return h


def main() raises:
    var args = argv()
    if len(args) < 9:
        raise Error(
            "usage: <data_dir> <name> <n_rows> <n_features> <n_classes>"
            " <trees> <depth> <max_features> [repeats]"
        )
    var data_dir = String(args[1])
    var name = String(args[2])
    var n_rows = Int(String(args[3]))
    var n_features = Int(String(args[4]))
    var n_classes = Int(String(args[5]))
    var trees = Int(String(args[6]))
    var depth = Int(String(args[7]))
    var mf_spec = String(args[8])
    var repeats = 6
    var with_sklearn = False
    for i in range(9, len(args)):
        var a = String(args[i])
        if a == String("sklearn"):
            # THE TRIGGER, ISOLATED. Eighteen refits in a process with nothing
            # else in it were bit-identical; the three corrupted cells all
            # happened in a process where scikit-learn's ten-core fit ran
            # between ours. This runs that fit between refits and changes
            # nothing else.
            with_sklearn = True
        else:
            repeats = Int(a)

    var ctx = DeviceContext()
    var yfull = read_f32(data_dir + "/" + name + "_y.f32")
    var total_rows = len(yfull)
    var labels = dense_class_ids(yfull, n_rows, n_classes)
    var x = read_column_prefix(
        data_dir + "/" + name + "_Xcol.f32", total_rows, n_rows, n_features
    )

    var cfg = ExtraTreesConfig()
    cfg.n_estimators = Int32(trees)
    cfg.max_depth = Int32(depth)
    cfg.max_features_spec = max_features_code(mf_spec, n_features)
    cfg.random_state = 1

    print(
        "[determinism]",
        ctx.name(),
        name,
        n_rows,
        "rows,",
        trees,
        "trees, depth",
        depth,
        ", max_features",
        mf_spec,
        ",",
        repeats,
        "refits of the SAME config in ONE process",
    )
    var arm = Python.none()
    if with_sklearn:
        var sysmod = Python.import_module("sys")
        _ = sysmod.path.append("extratrees/bench")
        arm = Python.import_module("sklearn_arm")

    var first = UInt64(0)
    var differing = 0
    for i in range(repeats):
        if with_sklearn:
            var t = arm.fit_seconds_and_accuracy(
                data_dir, name, n_rows, n_features, trees, depth, 1, -1,
                mf_spec,
            )
            print("   sklearn-10core", t[0].__float__(), "s")
        var t0 = perf_counter_ns()
        var res = fit_extra_trees_classifier_device(
            ctx,
            x,
            labels,
            Int32(n_rows),
            Int32(n_features),
            Int32(n_classes),
            cfg,
        )
        var ms = Float64(perf_counter_ns() - t0) / 1e6
        var nodes = 0
        for t in range(len(res.forest.trees)):
            nodes += res.forest.trees[t].num_nodes()
        var d = digest(res.forest)
        if i == 0:
            first = d
        elif d != first:
            differing += 1
        print(
            "   refit",
            i,
            " nodes",
            nodes,
            " digest",
            hex(d),
            " same_as_first",
            d == first,
            " ",
            ms,
            "ms",
        )
    print("[determinism]", differing, "of", repeats - 1, "refits DIFFER")
    _ = x.unsafe_ptr()
    _ = labels.unsafe_ptr()
