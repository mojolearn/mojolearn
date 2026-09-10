# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Full-forest fingerprints across every accumulator dispatch boundary.

Compare this program with and without -D MOJOLEARN_ET_MAX_ACC_32=1 in BOTH
numeric modes using extratrees/tools/check_accumulator_dispatch.sh. Both arms
use corrected 32-class leaf dispatch: this isolates the score optimization,
not the pre-fix zero probabilities above 16 classes. Every node field
and leaf probability bit is folded. Bootstrap and best-first frontiers also
exercise the shared search dispatch. No timing is taken by the default run.

Optional arguments <rows> <classes> <trees> <depth> <reps> time larger fits,
with one warmup, and print fingerprints outside the timed region.
"""
from checks.numerics import numeric_mode_name
from max.gpu.host import DeviceContext
from std.sys import argv
from std.time import perf_counter_ns
from std.testing import assert_equal
from extratrees.checks.fixtures import hashed_classification
from extratrees.checks.bestfirst_fingerprint import (
    column_major, float_labels, forest_fingerprint,
)
from extratrees.estimator import (
    ExtraTreesConfig, MAX_FEATURES_ALL, fit_extra_trees_classifier_device,
)


def run_case(
    ctx: DeviceContext, rows: Int, classes: Int, trees: Int, depth: Int,
    bootstrap: Bool, bestfirst: Bool, reps: Int = 0,
) raises:
    var fixture = hashed_classification(0xACC2021, rows, 13, classes)
    var x = column_major(fixture)
    var y = float_labels(fixture)
    var config = ExtraTreesConfig()
    config.n_estimators = Int32(trees)
    config.max_depth = Int32(depth)
    config.max_features_spec = MAX_FEATURES_ALL
    config.random_state = 0xACC2021
    config.bootstrap = bootstrap
    if bestfirst:
        config.max_leaf_nodes = 13
    var fingerprint = UInt64(0)
    for rep in range(reps + 1):
        var start = perf_counter_ns() if reps > 0 else 0
        var fit = fit_extra_trees_classifier_device(
            ctx, x, y, Int32(rows), Int32(13), Int32(classes), config,
        )
        ctx.synchronize()
        var elapsed = perf_counter_ns() - start if reps > 0 else 0
        var got = forest_fingerprint(fit.forest)
        assert_equal(len(fit.forest.trees), trees)
        for t in range(trees):
            assert_equal(Int(fit.forest.trees[t].num_outputs), classes)
        if rep == 0:
            fingerprint = got
            print("fingerprint", rows, classes, trees, depth, bootstrap, bestfirst, got)
        else:
            assert_equal(got, fingerprint)
            print("fit_ms", Float64(elapsed) / 1_000_000.0)


def main() raises:
    print("numeric_mode", numeric_mode_name())
    var ctx = DeviceContext()
    var args = argv()
    if len(args) == 6:
        run_case(
            ctx, Int(String(args[1])), Int(String(args[2])),
            Int(String(args[3])), Int(String(args[4])), False, False,
            Int(String(args[5])),
        )
        return
    var class_counts: List[Int] = [1, 2, 4, 5, 8, 9, 16, 17, 32]
    for classes in class_counts:
        run_case(ctx, 1537, classes, 3, 7, False, False)
        run_case(ctx, 1537, classes, 3, 7, True, True)
