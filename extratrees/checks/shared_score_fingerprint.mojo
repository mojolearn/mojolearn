# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Whole-forest/prediction A/B for the shared integer class-count candidate.

Build with and without MOJOLEARN_ET_SHARED_CLASS_COUNTS and compare
fingerprints in FAST, DETERMINISTIC and IDENTICAL. Default cases span all
accumulator widths, Gini/entropy, bootstrap and best-first. Every model
field is folded, plus probabilities on all fixture rows (first2048 for
optional large timing cases). Predictions are outside the fit timer.

Optional arguments: rows classes trees depth reps. MOJOLEARN_ET_WARMUP_FITS
controls warmup fits (default1); correctness-only runs always fit once.
"""
from checks.numerics import numeric_mode_name
from max.gpu.host import DeviceContext
from std.sys import argv
from std.os import getenv
from extratrees.impl.decisiontree.batched_levelalgo.kernels.builder_kernels_impl import shared_class_counts_mask
from std.time import perf_counter_ns
from std.testing import assert_equal
from extratrees.checks.fixtures import hashed_classification
from extratrees.checks.bestfirst_fingerprint import (
    column_major, float_labels, forest_fingerprint, mix64,
)
from extratrees.estimator import (
    ExtraTreesConfig, MAX_FEATURES_ALL, fit_extra_trees_classifier_device,
)

from extratrees.impl.randomforest.randomforest import forest_vote
from extratrees.impl.decisiontree.decisiontree import CRITERION_ENTROPY


def run_case(
    ctx: DeviceContext, rows: Int, classes: Int, trees: Int, depth: Int,
    bootstrap: Bool, bestfirst: Bool, reps: Int = 0, entropy: Bool = False,
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
    if entropy:
        config.criterion = CRITERION_ENTROPY
    if bestfirst:
        config.max_leaf_nodes = 13
    var warmups = 1
    var warmups_env = getenv("MOJOLEARN_ET_WARMUP_FITS")
    if reps > 0 and warmups_env.byte_length() > 0:
        warmups = Int(warmups_env)
        if warmups < 1:
            raise Error("MOJOLEARN_ET_WARMUP_FITS must be positive")
    var fingerprint = UInt64(0)
    for rep in range(reps + warmups):
        var start = perf_counter_ns() if reps > 0 else 0
        var fit = fit_extra_trees_classifier_device(
            ctx, x, y, Int32(rows), Int32(13), Int32(classes), config,
        )
        ctx.synchronize()
        var elapsed = perf_counter_ns() - start if reps > 0 else 0
        var got = forest_fingerprint(fit.forest)
        var checked_rows = rows if rows < 2048 else 2048
        for r in range(checked_rows):
            var row = List[Float32]()
            for c in range(13):
                row.append(x[c * rows + r])
            var probabilities = forest_vote(fit.forest, row, 0)
            for probability in probabilities:
                got = mix64(got, UInt64(probability.to_bits[DType.uint32]()))
        assert_equal(len(fit.forest.trees), trees)
        for t in range(trees):
            assert_equal(Int(fit.forest.trees[t].num_outputs), classes)
        if rep == 0:
            fingerprint = got
            print("fingerprint", rows, classes, trees, depth, bootstrap, bestfirst, entropy, got)
        else:
            assert_equal(got, fingerprint)
            if rep >= warmups:
                print("fit_ms", Float64(elapsed) / 1_000_000.0)


def main() raises:
    print("numeric_mode", numeric_mode_name())
    print("shared_class_counts_mask", shared_class_counts_mask())
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
        run_case(ctx, 1537, classes, 3, 7, False, True, entropy=True)
