# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Our forest against scikit-learn's band — the third oracle, and its limits.

`extratrees/tools/sklearn_reference.txt` holds, per fixture, what sklearn's own
ExtraTrees does at 21 seeds: the min/median/max of holdout accuracy or MSE, and
of mean tree depth and mean leaf count. This file is the Mojo side that reads
it and puts our numbers beside theirs.

**WHAT IS A GATE HERE AND WHAT IS NOT.** Deviation 130 makes bitwise parity
impossible by construction, so sklearn's number is never a gate
(`PORTING_RULES.md` rule 4). Two kinds of row are nevertheless hard assertions,
and neither is a gate BECAUSE sklearn said so:

* the **analytic** fixtures, where the right answer is a closed form —
  `separable_gap` and `tie_pair` are perfectly separable so accuracy must be
  exactly 1.0, and `regression_step` is a step function with an empty band so
  its MSE must be 0. sklearn also reports 1.0 and ~1e-5 on those, which is a
  cross-check on the harness rather than the source of the requirement;
* the **degenerate** fixture `all_constant`, where no split exists at any node,
  so depth must be 0 and leaves exactly 1. sklearn reports 0 and 1 too. Our
  `tree_check` already asserts this independently; having both agree is what
  makes the harness itself trustworthy.

Everything else is REPORTED and not asserted. In particular
`shaped_constant_heavy` is the fixture deviations 132 and 151 predicted a loss
on: 45 of its 48 columns fail sklearn's float32 constant test, `sqrt` draws 6,
so about two thirds of nodes draw six constants. sklearn keeps drawing past
`max_features` until one is non-constant (`_splitter.pyx:573-577`) and we
cannot (deviation 151), so our trees should come out SHALLOWER. This file
prints both numbers so the prediction becomes a measurement. **It is a finding
to report and price, not a target to tune toward.**

One acknowledged mismatch, stated rather than hidden: sklearn's `max_depth` is
`None` (unlimited) and this port has no unlimited — cuML's own
`validity_check` asserts `max_depth >= 0` (`decisiontree.cu:29`). The depth
used here is set well above the mean depths sklearn reports on these fixtures
so that the cap does not bind, and the check asserts it did not bind.
"""

from std.testing import assert_equal, assert_true
from std.memory import bitcast

from extratrees.original.fixtures import (
    Dataset as FixtureDataset,
    all_shapes,
    analytic_all_constant,
    analytic_regression_step,
    analytic_separable_gap,
    analytic_tie_pair,
    hashed_classification,
    hashed_regression,
    shaped_dataset,
)
from extratrees.original.fixture_parity_check import const_heavy_shapes
from extratrees.derived.decisiontree.decisiontree import DecisionTreeParams
from extratrees.derived.randomforest.randomforest import (
    fit_classification,
    fit_regression,
    predict_class_forest,
    predict_regression_forest,
)


comptime REFERENCE_PATH = "extratrees/tools/sklearn_reference.txt"
comptime N_TREES: Int32 = 100
"""Sklearn's `n_estimators` default, and the count the reference was built at.
A different count would compare two different estimators."""

comptime MAX_DEPTH: Int32 = 40
"""Sklearn's default is `max_depth=None`. This port has no unlimited: cuML's
`validity_check` asserts `max_depth >= 0` (`decisiontree.cu:29`). 40 is far
above every mean depth in the reference (the largest is 16.74), and the check
asserts the cap never binds, so the comparison is against an effectively
unlimited tree."""


@fieldwise_init
struct Band(ImplicitlyCopyable, Movable):
    var lo: Float64
    var mid: Float64
    var hi: Float64

    def contains(self, x: Float64) -> Bool:
        return x >= self.lo and x <= self.hi


def hex_to_f64(text: StringSlice) raises -> Float64:
    """Read the HEX half of a `<decimal>/<hexbits>` pair.

    The decimal half is never parsed anywhere in this lane: `String(Float32)`
    does not round-trip in this toolchain (0.46% of values come back one ULP
    wrong), and the reference's own header says to read the hex.
    """
    var bits = UInt64(0)
    var n = 0
    for i in range(text.byte_length()):
        var c = Int(text.as_bytes()[i])
        var v: Int
        if c >= 48 and c <= 57:
            v = c - 48
        elif c >= 97 and c <= 102:
            v = c - 87
        elif c >= 65 and c <= 70:
            v = c - 55
        else:
            raise Error("bad hex digit in reference file")
        bits = (bits << 4) | UInt64(v)
        n += 1
    if n != 16:
        raise Error(
            "a float64 hex pattern is 16 digits; got " + String(n)
        )
    return bitcast[DType.float64](bits)


def read_reference() raises -> Dict[String, Float64]:
    """Flatten the reference into `"<fixture>.<key>" -> value` (hex-parsed).

    Deliberately dumb: no schema, no ordering assumption. A key that is missing
    when a caller asks for it raises, which is what should happen if the
    reference is regenerated with a different fixture set.
    """
    var out = Dict[String, Float64]()
    var text: String
    with open(REFERENCE_PATH, "r") as f:
        text = f.read()
    var fixture = String("")
    for line_slice in text.split("\n"):
        var line = String(line_slice)
        var parts = List[String]()
        for tok in line.split(" "):
            var t = String(tok)
            if t.byte_length() > 0:
                parts.append(t)
        if len(parts) == 0:
            continue
        if parts[0] == "fixture" and len(parts) >= 2:
            fixture = parts[1]
            continue
        if parts[0] == "end_fixture":
            fixture = String("")
            continue
        if fixture.byte_length() == 0 or len(parts) < 2:
            continue
        # `<key> <decimal>/<hexbits>` rows only; everything else is prose.
        var value = parts[1]
        var slash = -1
        for i in range(value.byte_length()):
            if value[byte=i] == "/":
                slash = i
        if slash < 0:
            continue
        var hexpart = String("")
        for i in range(slash + 1, value.byte_length()):
            hexpart += String(value[byte=i])
        if hexpart.byte_length() != 16:
            continue
        out[fixture + "." + parts[0]] = hex_to_f64(hexpart)
    return out^


def band_of(
    reference: Dict[String, Float64], fixture: String, key: String
) raises -> Band:
    return Band(
        reference[fixture + "." + key + "_min"],
        reference[fixture + "." + key + "_median"],
        reference[fixture + "." + key + "_max"],
    )


def is_train(row: Int) -> Bool:
    """The reference's split rule: `train iff (row_index % 4) < 2`.

    A pure function of the row index — no shuffle, no RNG, identical on both
    sides. Index parity would have been wrong on `all_constant`, whose labels
    are `r % 2`, and a prefix would have been wrong on every block-structured
    analytic fixture.
    """
    return (row % 4) < 2


@fieldwise_init
struct Slice(Movable):
    """A train or holdout split, already in column-major layout."""

    var x: List[Float32]
    var y: List[Float32]
    var rows: List[Int]
    var n_rows: Int32
    var n_cols: Int32


def split_of(
    fixture: FixtureDataset, want_train: Bool, is_classification: Bool
) -> Slice:
    var rows = List[Int]()
    for r in range(fixture.n_rows):
        if is_train(r) == want_train:
            rows.append(r)
    var n = len(rows)
    var x = List[Float32](length=n * fixture.n_cols, fill=Float32(0.0))
    var y = List[Float32]()
    for i in range(n):
        for c in range(fixture.n_cols):
            x[c * n + i] = fixture.value(rows[i], c)
        if is_classification:
            y.append(Float32(Int(fixture.label[rows[i]])))
        else:
            y.append(fixture.y[rows[i]])
    return Slice(x^, y^, rows^, Int32(n), Int32(fixture.n_cols))


def sqrt_ratio(n_cols: Int32) -> Float32:
    """Sklearn's `max_features='sqrt'` as the ratio this port takes.

    `n_sampled_cols_for` computes `Int32(max_features * n_cols)` with
    truncation (`builder.cuh:222`), so the ratio must be nudged above the exact
    quotient or an integer `sqrt(n)` truncates to `n-1` of itself. Computed as
    an integer count first, then divided, which is what sklearn does
    (`_classes.py`: `max_features = max(1, int(np.sqrt(n_features)))`).
    """
    var k = 1
    while (k + 1) * (k + 1) <= Int(n_cols):
        k += 1
    return (Float32(k) + 0.5) / Float32(Int(n_cols))


def main() raises:
    var reference = read_reference()
    print(
        "sklearn quality band, read from",
        REFERENCE_PATH,
        "-- REPORTED, not gated, except where noted",
    )
    print("")

    var cells = 0
    var out_of_band = 0
    var failures = List[String]()

    # (name, task, band key) -- built below by name so the fixture
    # constructors stay explicit rather than table-driven.
    var seed = UInt64(20260821)

    # ---- helper: fit and score one classification fixture ---------------
    for ci in range(6):
        var name: String
        var fx: FixtureDataset
        var is_cls = True
        if ci == 0:
            name = String("hashed_cls")
            fx = hashed_classification(seed, 512, 16, 3)
        elif ci == 1:
            name = String("shaped_all")
            fx = shaped_dataset(seed, 512, all_shapes())
        elif ci == 2:
            name = String("shaped_constant_heavy")
            fx = shaped_dataset(seed, 512, const_heavy_shapes())
        elif ci == 3:
            name = String("separable_gap")
            fx = analytic_separable_gap(seed).data.copy()
        elif ci == 4:
            name = String("tie_pair")
            fx = analytic_tie_pair(seed).data.copy()
        else:
            name = String("all_constant")
            fx = analytic_all_constant().data.copy()

        var train = split_of(fx, True, is_cls)
        var hold = split_of(fx, False, is_cls)

        var p = DecisionTreeParams()
        p.max_depth = MAX_DEPTH
        p.max_features = sqrt_ratio(train.n_cols)
        var forest = fit_classification(
            train.x,
            train.y,
            train.n_rows,
            train.n_cols,
            Int32(fx.n_classes),
            p,
            N_TREES,
            0xE7E7E7,
        )

        var correct = 0
        for i in range(Int(hold.n_rows)):
            var row = List[Float32]()
            for c in range(Int(hold.n_cols)):
                row.append(hold.x[c * Int(hold.n_rows) + i])
            if predict_class_forest(forest, row, 0) == Int(hold.y[i]):
                correct += 1
        var acc = Float64(correct) / Float64(Int(hold.n_rows))

        var depth_sum = 0.0
        var leaf_sum = 0.0
        var capped = 0
        for t in range(len(forest.trees)):
            depth_sum += Float64(Int(forest.trees[t].depth_counter))
            leaf_sum += Float64(Int(forest.trees[t].leaf_counter))
            if forest.trees[t].depth_counter >= MAX_DEPTH:
                capped += 1
        var mean_depth = depth_sum / Float64(len(forest.trees))
        var mean_leaves = leaf_sum / Float64(len(forest.trees))
        if capped != 0:
            failures.append(
                name
                + ": max_depth BOUND on "
                + String(capped)
                + " trees, so this is not a comparison against sklearn's"
                " unlimited depth any more"
            )
        cells += 1

        var acc_band = band_of(reference, name, "accuracy")
        var d_band = band_of(reference, name, "mean_depth")
        var l_band = band_of(reference, name, "mean_leaves")
        var mark = "  " if acc_band.contains(acc) else "**"
        if not acc_band.contains(acc):
            out_of_band += 1
        print(
            mark,
            name,
            " accuracy ",
            acc,
            " sklearn [",
            acc_band.lo,
            ",",
            acc_band.hi,
            "]",
        )
        print(
            "     depth ",
            mean_depth,
            " sklearn [",
            d_band.lo,
            ",",
            d_band.hi,
            "]    leaves ",
            mean_leaves,
            " sklearn [",
            l_band.lo,
            ",",
            l_band.hi,
            "]",
        )
        _ = train.x.unsafe_ptr()
        _ = train.y.unsafe_ptr()

        # --- the two rows that ARE assertions, and not because sklearn -----
        if name == "separable_gap" or name == "tie_pair":
            if acc != 1.0:
                failures.append(
                    name
                    + ": accuracy "
                    + String(acc)
                    + " on a PERFECTLY SEPARABLE fixture, where every"
                    " threshold the RNG can draw inside the empty band"
                    " separates the classes. sklearn gets exactly 1.0."
                )
            cells += 1
        if name == "all_constant":
            if mean_depth != 0.0:
                failures.append("all_constant: depth is not 0")
            if mean_leaves != 1.0:
                failures.append("all_constant: leaves is not 1")
            cells += 2

    # ---------------- regression fixtures ---------------------------------
    for ci in range(2):
        var name: String
        var fx: FixtureDataset
        if ci == 0:
            name = String("hashed_reg")
            fx = hashed_regression(seed, 512, 16)
        else:
            name = String("regression_step")
            fx = analytic_regression_step(seed).data.copy()

        var train = split_of(fx, True, False)
        var hold = split_of(fx, False, False)
        var p = DecisionTreeParams()
        p.max_depth = MAX_DEPTH
        p.max_features = 1.0  # sklearn's regression default
        var forest = fit_regression(
            train.x, train.y, train.n_rows, train.n_cols, p, N_TREES, 0xE7E7E7
        )

        var se = Float64(0.0)
        for i in range(Int(hold.n_rows)):
            var row = List[Float32]()
            for c in range(Int(hold.n_cols)):
                row.append(hold.x[c * Int(hold.n_rows) + i])
            var d = Float64(predict_regression_forest(forest, row, 0)) - Float64(
                hold.y[i]
            )
            se += d * d
        var mse = se / Float64(Int(hold.n_rows))

        var depth_sum = 0.0
        var leaf_sum = 0.0
        for t in range(len(forest.trees)):
            depth_sum += Float64(Int(forest.trees[t].depth_counter))
            leaf_sum += Float64(Int(forest.trees[t].leaf_counter))
        var mean_depth = depth_sum / Float64(len(forest.trees))
        var mean_leaves = leaf_sum / Float64(len(forest.trees))

        var mse_band = band_of(reference, name, "mse")
        var d_band = band_of(reference, name, "mean_depth")
        var mark = "  " if mse_band.contains(mse) else "**"
        if not mse_band.contains(mse):
            out_of_band += 1
        print(
            mark,
            name,
            " mse ",
            mse,
            " sklearn [",
            mse_band.lo,
            ",",
            mse_band.hi,
            "]",
        )
        print(
            "     depth ",
            mean_depth,
            " sklearn [",
            d_band.lo,
            ",",
            d_band.hi,
            "]    leaves ",
            mean_leaves,
        )
        if name == "regression_step":
            # THE THRESHOLD HERE WAS WRONG ONCE AND THE FIX IS RECORDED
            # RATHER THAN QUIETLY APPLIED. It used to demand `mse < 1e-6`, on
            # the reasoning that every threshold drawn inside the empty band
            # reproduces the step exactly. That is true of the SPLIT and false
            # of the HOLDOUT: a held-out row can be routed by a threshold that
            # lies inside one level's block, because a deep node's own range
            # no longer spans the gap, and land in a leaf dominated by the
            # other level. sklearn shows the same thing -- its own minimum
            # over 21 seeds is 1.95e-5, not 0 -- so `< 1e-6` was demanding
            # better than sklearn and would have been an unmeetable gate.
            #
            # The analytic statement that IS true: y takes the values +2 and
            # -3 in equal proportion, so predicting the mean gives a variance
            # of 6.25, and any learner that has found the step at all must be
            # orders below that. 0.05 is under 1% of the variance and is not
            # derived from sklearn's number.
            if not (mse < 0.05):
                failures.append(
                    "regression_step: mse "
                    + String(mse)
                    + " against a mean-predictor variance of 6.25 -- the step"
                    " has not been found at all"
                )
            cells += 1
        _ = train.x.unsafe_ptr()
        _ = train.y.unsafe_ptr()

    print("")
    print(
        "rows outside sklearn's band:",
        out_of_band,
        "-- REPORTED, not a failure. Deviations 132 and 151 predict a loss on"
        " shaped_constant_heavy in particular.",
    )
    print("quality_band: ", cells, "asserted cells")
    if len(failures) > 0:
        print("")
        print("=== ANALYTIC FAILURES -- these are NOT band misses ===")
        for i in range(len(failures)):
            print("  *", failures[i])
        print("")
        print(
            "DIAGNOSED, and it is the ported cuML sampler defect: at k = 1"
            " column 0 is NEVER drawn, for any n (measured: n=2 -> col0 0 of"
            " 64, n=8 -> col0 0 of 64). separable_gap's separable feature IS"
            " column 0, so the learner can never see it and splits on noise"
            " forever. Root cause is builder_kernels.cuh:231-232 passing"
            " mask[0] -- a FLAG -- as SubtractLeft's tile predecessor, so the"
            " block minimum is marked a duplicate whenever it equals the"
            " previous iteration's flag. Transcribed faithfully per rule 1;"
            " see DEVIATIONS.md. Whether to keep it is an OPEN item in"
            " PLAN.md and is Andrew's call, because a workaround here changes"
            " the algorithm and PORTING_RULES.md rule 4 says that is a fork,"
            " not a workaround."
        )
        raise Error(
            String(len(failures))
            + " analytic assertion(s) failed -- see the diagnosis above"
        )
    print("quality_band_check: PASS")
