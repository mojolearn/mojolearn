# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Dump the ExtraTrees fixtures CELL FOR CELL, as `<decimal>/<hexbits>`, so a
Python reader can prove it generates the IDENTICAL BYTES.

    Consumed by:  extratrees/tools/sklearn_reference.py --verify-parity
    Generator it dumps:  extratrees/mojo_only/fixtures.mojo

WHY A DUMPER EXISTS AT ALL
--------------------------
`tools/sklearn_reference.py` trains scikit-learn's `ExtraTreesClassifier` and
`ExtraTreesRegressor` on "the same fixtures our Mojo code uses". That sentence
is a CLAIM, and until something compares bytes it is an unverified one. A
quality band computed on data that merely RESEMBLES ours is not a band on our
data; it is a band on some other dataset with the same shape, and the number it
produces would be a confound rather than a reference.

So the Python side reimplements `splitmix64`, `cell_hash`, `unit_float`,
`unit_float20_open`, `signed_unit` and the shaped-column logic, and this program
is the thing it is checked AGAINST. Nothing here computes an expected value; it
prints what `fixtures.mojo` actually built.

WHY HEX AND NOT DECIMAL
-----------------------
`String(Float32)` in this toolchain does not round-trip: 0.46% of float32
values come back ONE ULP WRONG when parsed, and `String(Float32(1.4e-45))` is
the string `"0.0"`. A decimal-only dump would therefore let a one-ulp
disagreement -- exactly the kind a float32-vs-float64 mistake in the Python port
produces -- pass as agreement. Every value is printed as `<decimal>/<hexbits>`
and the READER COMPARES THE HEX. The decimal is for a human reading the diff
and is not evidence.

WHAT IS DUMPED, and why these eight
-----------------------------------
    hashed_cls              512 x 16, 3 classes   pure-noise classification
    hashed_reg              512 x 16, regression  pure-noise regression
    shaped_all              512 x 10, 2 classes   one column per adversarial SHAPE
    shaped_constant_heavy   512 x 48, 2 classes   3 hashed + 45 columns that the
                                                  1e-7 constant test calls constant
    separable_gap           256 x 2,  2 classes   analytic, perfectly separable
    regression_step         256 x 2,  regression  analytic, two-level step
    tie_pair                256 x 3,  2 classes   feature 1 is a bit-copy of 0
    all_constant            128 x 4,  2 classes   NO feature can be split

The last two are the degenerate cases DEVIATIONS 132 and 151 make a prediction
about (our trees are SHALLOWER than sklearn's when many columns are constant,
because we do not re-draw past `max_features` and we do not inherit a constant
count down the tree). `shaped_constant_heavy` is the fixture where that
prediction becomes a number: at `max_features='sqrt'` it draws 6 of 48 columns,
and 45 of the 48 are constant, so a node that samples only constants is the
COMMON case rather than a corner one.

FORMAT
------
    # extratrees-fixture-dump v1
    FIXTURE <name> seed=<n> rows=<n> cols=<n> classes=<n> task=<classification|regression>
    R <row> <dec>/<hex8> ... | <ydec>/<yhex8> <label>
    END <name>

One line per row: every feature cell in column order, then `|`, then the
regression target and the integer class label. `n_classes` is 1 on a regression
fixture, and its labels are all 0 -- `fixtures.mojo` carries both targets on
every dataset so the same feature bytes serve both tasks.

DETERMINISM
-----------
Everything printed is a pure function of `SEED` and the integer indices. There
is no RNG object, no file input and no Python. Re-running this program on any
machine this repository builds on must produce a byte-identical dump, and the
Python reader is entitled to assume so.

USAGE
    pixi run mojo run -I . extratrees/mojo_only/fixture_parity_check.mojo > dump.txt
    pixi run -e bench python extratrees/tools/sklearn_reference.py \\
        --verify-parity dump.txt
"""

from extratrees.mojo_only.fixtures import (
    N_SHAPES,
    SHAPE_CONSTANT,
    SHAPE_HASHED,
    SHAPE_NEAR_CONST_BELOW,
    SHAPE_NEAR_CONST_EQUAL,
    Dataset,
    all_shapes,
    analytic_all_constant,
    analytic_regression_step,
    analytic_separable_gap,
    analytic_tie_pair,
    hashed_classification,
    hashed_regression,
    shaped_dataset,
)

# The seed `fixtures_check.mojo` uses. Kept identical on purpose: the fixtures
# the quality band is measured on are the fixtures the property checks run on,
# not a second set that happens to share a generator.
comptime SEED: UInt64 = 20260821

comptime HASHED_ROWS = 512
comptime HASHED_COLS = 16
comptime HASHED_CLASSES = 3
comptime SHAPED_ROWS = 512

# `shaped_constant_heavy`: 3 hashed columns and 45 that the float32 1e-7
# constant test rejects. 15 of each of the three constant-family shapes rather
# than 45 of one, so the fixture cannot be passed by an implementation that
# special-cases "every value bit-identical" -- two of the three families have
# TWO distinct values whose difference is at or below the threshold.
comptime CONST_HEAVY_HASHED = 3
comptime CONST_HEAVY_PER_SHAPE = 15


def hex_digit(d: Int) -> String:
    """One lowercase hex digit. Written as a branch chain rather than an index
    into a digit table so that no allocation happens per digit; this runs about
    half a million times."""
    if d == 0:
        return "0"
    if d == 1:
        return "1"
    if d == 2:
        return "2"
    if d == 3:
        return "3"
    if d == 4:
        return "4"
    if d == 5:
        return "5"
    if d == 6:
        return "6"
    if d == 7:
        return "7"
    if d == 8:
        return "8"
    if d == 9:
        return "9"
    if d == 10:
        return "a"
    if d == 11:
        return "b"
    if d == 12:
        return "c"
    if d == 13:
        return "d"
    if d == 14:
        return "e"
    return "f"


def hex8(b: UInt32) -> String:
    """The 32 bits, most significant nibble first, always eight characters.

    Zero padding is not cosmetic: an unpadded dump makes `0x0000_0001` and
    `0x1000_0000` both print as a short string that a sloppy reader could
    misalign, and the reader compares these as strings.
    """
    var out = String("")
    for i in range(8):
        var shift = UInt32((7 - i) * 4)
        out += hex_digit(Int((b >> shift) & UInt32(0xF)))
    return out^


def cell_text(v: Float32) -> String:
    """`<decimal>/<hexbits>`. The hex is the evidence; see the module docstring."""
    return String(v) + "/" + hex8(v.to_bits[DType.uint32]())


def dump(name: String, task: String, d: Dataset) raises:
    print(
        "FIXTURE",
        name,
        "seed=" + String(SEED),
        "rows=" + String(d.n_rows),
        "cols=" + String(d.n_cols),
        "classes=" + String(d.n_classes),
        "task=" + task,
    )
    for r in range(d.n_rows):
        var line = "R " + String(r)
        for c in range(d.n_cols):
            line += " " + cell_text(d.value(r, c))
        line += " | " + cell_text(d.y[r]) + " " + String(d.label[r])
        print(line)
    print("END", name)


def const_heavy_shapes() -> List[Int]:
    var s = List[Int]()
    for _ in range(CONST_HEAVY_HASHED):
        s.append(SHAPE_HASHED)
    for _ in range(CONST_HEAVY_PER_SHAPE):
        s.append(SHAPE_CONSTANT)
    for _ in range(CONST_HEAVY_PER_SHAPE):
        s.append(SHAPE_NEAR_CONST_BELOW)
    for _ in range(CONST_HEAVY_PER_SHAPE):
        s.append(SHAPE_NEAR_CONST_EQUAL)
    return s^


def main() raises:
    print("# extratrees-fixture-dump v1")
    print("# generator extratrees/mojo_only/fixtures.mojo")
    print("# seed", SEED)
    print("# shapes", N_SHAPES)

    dump(
        "hashed_cls",
        "classification",
        hashed_classification(SEED, HASHED_ROWS, HASHED_COLS, HASHED_CLASSES),
    )
    dump(
        "hashed_reg",
        "regression",
        hashed_regression(SEED, HASHED_ROWS, HASHED_COLS),
    )
    dump(
        "shaped_all",
        "classification",
        shaped_dataset(SEED, SHAPED_ROWS, all_shapes()),
    )
    dump(
        "shaped_constant_heavy",
        "classification",
        shaped_dataset(SEED, SHAPED_ROWS, const_heavy_shapes()),
    )

    var sg = analytic_separable_gap(SEED)
    dump("separable_gap", "classification", sg.data)
    var rs = analytic_regression_step(SEED)
    dump("regression_step", "regression", rs.data)
    var tp = analytic_tie_pair(SEED)
    dump("tie_pair", "classification", tp.data)
    var ac = analytic_all_constant()
    dump("all_constant", "classification", ac.data)

    print("# end of dump")
