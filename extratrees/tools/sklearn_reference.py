"""scikit-learn ExtraTrees as a QUALITY BAND for the `extratrees/` lane.

    Writes:    extratrees/tools/sklearn_reference.txt
    Reads:     a dump produced by extratrees/mojo_only/fixture_parity_check.mojo
    Explained: extratrees/tools/README_sklearn_reference.md

THIS IS NOT A CORRECTNESS GATE, AND READING IT AS ONE IS A MISTAKE
------------------------------------------------------------------
Bitwise parity with scikit-learn is IMPOSSIBLE BY CONSTRUCTION here, and the
reason is structural rather than a matter of effort. It is `DEVIATIONS.md` 130:

  * sklearn draws every random number in `node_split_random` from ONE
    sequential 32-bit xorshift stream (`our_rand_r`, `utils/_random.pxd:20-34`)
    advancing a single state word. Draw *k* therefore depends on draws
    *0..k-1*.
  * The ORDER of those draws is the order of the Fisher-Yates feature walk
    (`tree/_splitter.pyx:592`), and that walk's path depends on WHICH FEATURES
    WERE FOUND CONSTANT, which depends on the data (`_splitter.pyx:611-621`).
  * Our builder draws counter-based keyed values off
    `(seed, tree_id, node_id, feature_id)`. It is order-independent on purpose,
    because a builder that evaluates features in parallel cannot reproduce a
    sequential stream without serializing the thing being parallelized.

Two more deviations widen the gap before a single number is compared:
DEVIATION 131 (our feature sampler is cuML's, not sklearn's Fisher-Yates) and
DEVIATION 132 (constant features are re-discovered per node, never inherited
down the tree).

So this file produces HOLDOUT ACCURACY and HOLDOUT MSE and nothing else. It is
a BAND, not a value: every quantity is reported as min/median/max over 21
sklearn seeds, because a single seed is a sample and not a range. The intended
use is "our number falls inside the range sklearn itself produces when ONLY its
seed changes". A number outside that range is a FINDING TO REPORT. It is not a
target to tune toward, and nothing in this lane may be adjusted to hit it --
that would be fitting our learner to a second learner's noise.

It is also, per `STANDING_ORDERS.md` rule 4, never run on a real dataset.
Every fixture here is constructed, adversarial, and hand-describable.

STEP 1 IS FIXTURE PARITY, AND IT IS THE LOAD-BEARING ONE
--------------------------------------------------------
A band measured on data that merely RESEMBLES ours is a band on somebody else's
dataset. So this script does not accept "same generator, same seed" as a claim:
it reimplements `fixtures.mojo`'s generator in Python and compares CELL FOR CELL
against a dump the Mojo program itself produced, and it REFUSES TO TRAIN OR
WRITE ANYTHING if a single cell disagrees.

The comparison is on FLOAT BIT PATTERNS, never on decimal text. This repository
has a standing finding that `String(Float32)` does not round-trip -- 0.46% of
float32 values come back one ULP wrong and `String(Float32(1.4e-45))` is the
string `"0.0"` -- so a decimal-only comparison would silently accept exactly the
one-ulp disagreement a float32-versus-float64 slip in this port would cause.
The dump carries `<decimal>/<hexbits>` and only the hex is evidence.

Python integers are arbitrary precision and Mojo's `UInt64` wraps, so every
addition, multiplication and shift in the ported hash is masked to 64 bits
explicitly. That masking is not decoration: an unmasked `splitmix64` agrees with
the Mojo one for small inputs and diverges for large ones, which is the shape of
bug a spot check misses and a cell-for-cell compare does not.

WHAT IS MEASURED
----------------
Eight fixtures, all from `extratrees/mojo_only/fixtures.mojo` at seed 20260821,
the same seed `fixtures_check.mojo` uses:

    hashed_cls              512 x 16, 3 classes.  Labels drawn from their OWN
                            salt, so they are a function of NOTHING in X. The
                            band here is chance-level by construction; what it
                            checks is that a learner is not somehow beating
                            chance on noise.
    hashed_reg              512 x 16, regression, target is independent noise.
    shaped_all              512 x 10, 2 classes, one column per adversarial
                            SHAPE (constant, three near-constants straddling
                            the 1e-7 threshold, two-valued, outlier, all-
                            negative, spans-zero, all-equal-but-one).
    shaped_constant_heavy   512 x 48, 2 classes, 3 hashed columns and 45 that
                            the float32 1e-7 constant test rejects. DEGENERATE.
    separable_gap           256 x 2, 2 classes, perfectly separable with an
                            empty band in [1, 9]. Accuracy 1.0 is attainable.
    regression_step         256 x 2, regression, a two-level step with an empty
                            band in [4, 6]. MSE 0.0 is attainable.
    tie_pair                256 x 3, 2 classes, feature 1 is a BIT-FOR-BIT copy
                            of feature 0.
    all_constant            128 x 4, 2 classes, every column exactly constant.
                            No split exists. DEGENERATE.

THE TWO DEGENERATE ROWS EXIST TO MAKE A PREDICTION CHECKABLE
------------------------------------------------------------
`DEVIATIONS.md` 132 and 151 together predict that OUR TREES ARE SHALLOWER THAN
SKLEARN'S on data with many constant columns, and they say why:

  * 151: sklearn's loop guard (`_splitter.pyx:573-577`) keeps drawing PAST
    `max_features`, up to all `n_features`, for as long as every feature drawn
    so far was constant. If any non-constant feature exists anywhere, sklearn
    finds one and the node splits. We evaluate the sampled `colids` exactly
    once; if all of them are constant the node becomes a leaf.
  * 132: sklearn threads `n_constant_features` down the tree, so a feature found
    constant at an ancestor is excluded from every descendant's draw. Nothing is
    inherited on our side.

`shaped_constant_heavy` is where that bites hardest: `max_features='sqrt'` of 48
draws 6 columns, 45 of the 48 are constant, so the probability that a node draws
six constants is C(45,6)/C(48,6) ~= 0.66. The common case, not a corner. This
file therefore dumps sklearn's MEAN TREE DEPTH and MEAN LEAF COUNT per fixture
so the prediction has a concrete number to be compared against rather than a
direction.

`all_constant` is the floor: no feature can be split at all, so sklearn's own
trees must be depth 0 with a single leaf, and any implementation that reports
otherwise is broken.

THE TRAIN / HOLDOUT SPLIT
-------------------------
    row r is TRAIN iff (r % 4) < 2, HOLDOUT otherwise.

Deterministic, a pure function of the row index, no shuffle and no RNG, so the
split is identical in Python and in any Mojo reader of this file. It is a 50/50
split.

Why not the obvious index parity (even train, odd holdout): `all_constant`
labels rows `r % 2`, so a parity split puts EVERY class-0 row in train and
EVERY class-1 row in holdout, and the reported accuracy would be 0.0 as an
artifact of the split rather than a fact about the learner. Why not a fixed
prefix: `separable_gap`, `tie_pair` and `regression_step` are BLOCK STRUCTURED
(rows 0..127 one class, 128..255 the other), so a prefix split trains on one
class and tests on the other. A period-4 rule breaks both structures: it takes
exactly half of every block of 4 and half of every parity class.

USAGE
    pixi run mojo run -I . extratrees/mojo_only/fixture_parity_check.mojo > /tmp/dump.txt
    pixi run -e bench python extratrees/tools/sklearn_reference.py --dump /tmp/dump.txt

    # parity only, no training, no file written:
    pixi run -e bench python extratrees/tools/sklearn_reference.py --dump /tmp/dump.txt --parity-only
"""

from __future__ import annotations

import argparse
import hashlib
import struct
import sys
from typing import Dict, List, Tuple

import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

# ---------------------------------------------------------------------------
# The seed and the shapes, mirroring `fixture_parity_check.mojo`.
# ---------------------------------------------------------------------------
SEED = 20260821

HASHED_ROWS = 512
HASHED_COLS = 16
HASHED_CLASSES = 3
SHAPED_ROWS = 512
CONST_HEAVY_HASHED = 3
CONST_HEAVY_PER_SHAPE = 15

# Salts, `fixtures.mojo:189-191`.
SALT_X = 0x0000000000000001
SALT_Y = 0x0000000000000002
SALT_LABEL = 0x0000000000000003

# Shape ids, `fixtures.mojo:175-185`.
SHAPE_HASHED = 0
SHAPE_CONSTANT = 1
SHAPE_NEAR_CONST_BELOW = 2
SHAPE_NEAR_CONST_EQUAL = 3
SHAPE_NEAR_CONST_ABOVE = 4
SHAPE_TWO_VALUED = 5
SHAPE_OUTLIER = 6
SHAPE_NEGATIVE = 7
SHAPE_SPANS_ZERO = 8
SHAPE_ONE_ODD_ROW = 9
N_SHAPES = 10

# `_partitioner.pxd:13`, and it is a float32 there.
FEATURE_THRESHOLD = np.float32(1e-7)

MASK64 = (1 << 64) - 1

# 21, an ODD count, so the median is an OBSERVED value rather than the mean of
# two neighbours. The brief's floor is 15.
N_SKLEARN_SEEDS = 21


# ---------------------------------------------------------------------------
# The generator, ported from `extratrees/mojo_only/fixtures.mojo`.
#
# Mojo's `UInt64` wraps; Python's `int` does not. Every arithmetic step below is
# masked to 64 bits EXPLICITLY. Do not remove a mask because "the value is
# small" -- splitmix64's second and third multiplies overflow on essentially
# every input.
# ---------------------------------------------------------------------------
def splitmix64(x: int) -> int:
    """`fixtures.mojo:199-215`."""
    z = (x + 0x9E3779B97F4A7C15) & MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return z ^ (z >> 31)


def cell_hash(seed: int, row: int, col: int, salt: int) -> int:
    """`fixtures.mojo:218-229`. Four sequential rounds; NOT an xor of two
    multiplied indices, which is linear over GF(2) in the low bits."""
    h = splitmix64((seed + 0x9E3779B97F4A7C15) & MASK64)
    h = splitmix64(h ^ ((row + 0x165667B19E3779F9) & MASK64))
    h = splitmix64(h ^ ((col + 0x27D4EB2F165667C5) & MASK64))
    return splitmix64(h ^ salt)


def unit_float(h: int) -> np.float32:
    """`[0, 1)` on the 2^24 grid. `fixtures.mojo:232-239`.

    The division is float64 and the narrowing to float32 is the last step,
    exactly as the Mojo does; both are exact for a 24-bit numerator over 2^24,
    which is the point of the 24.
    """
    return np.float32(np.float64(h >> 40) / np.float64(16777216.0))


def unit_float20_open(h: int) -> np.float32:
    """`(0, 1]` on the COARSER 2^20 grid. `fixtures.mojo:249-260`."""
    return np.float32(np.float64((h >> 44) + 1) / np.float64(1048576.0))


def signed_unit(h: int) -> np.float32:
    """`[-1, 1)`. `fixtures.mojo:263-271`. Float32 throughout: `2*u` is exact
    for `u` on the 2^-24 grid, so the subtraction is exact too and this stays
    bit-identical whether or not the Mojo contracts it into an FMA."""
    return np.float32(np.float32(2.0) * unit_float(h) - np.float32(1.0))


def _bits(x: np.float32) -> int:
    return int(np.float32(x).view(np.uint32))


def _from_bits(b: int) -> np.float32:
    return np.uint32(b & 0xFFFFFFFF).view(np.float32)


def next_up(x: np.float32) -> np.float32:
    """`fixtures.mojo:278-282`, positive finite `x` only."""
    return _from_bits(_bits(x) + 1)


def next_down(x: np.float32) -> np.float32:
    """`fixtures.mojo:285-288`, positive finite `x` only."""
    return _from_bits(_bits(x) - 1)


def shaped_value(seed: int, shape: int, row: int, col: int, n_rows: int) -> np.float32:
    """`fixtures.mojo:536-611`, branch for branch and in the same order."""
    h = cell_hash(seed, row, col, SALT_X)

    if shape == SHAPE_HASHED:
        return signed_unit(h)

    if shape == SHAPE_CONSTANT:
        return np.float32(3.25)

    if shape in (SHAPE_NEAR_CONST_BELOW, SHAPE_NEAR_CONST_EQUAL, SHAPE_NEAR_CONST_ABOVE):
        if shape == SHAPE_NEAR_CONST_BELOW:
            hi = next_down(FEATURE_THRESHOLD)
        elif shape == SHAPE_NEAR_CONST_EQUAL:
            hi = FEATURE_THRESHOLD
        else:
            hi = next_up(FEATURE_THRESHOLD)
        if row == 0:
            return np.float32(0.0)
        if row == 1:
            return hi
        return np.float32(0.0) if (h & 1) == 0 else hi

    if shape == SHAPE_TWO_VALUED:
        if row == 0:
            return np.float32(-2.0)
        if row == 1:
            return np.float32(5.0)
        return np.float32(-2.0) if (h & 1) == 0 else np.float32(5.0)

    if shape == SHAPE_OUTLIER:
        if row == n_rows // 2:
            return np.float32(1024.0)
        return np.float32(unit_float(h) / np.float32(1024.0))

    if shape == SHAPE_NEGATIVE:
        return np.float32(np.float32(-9.0) + np.float32(4.0) * unit_float(h))

    if shape == SHAPE_SPANS_ZERO:
        return signed_unit(h)

    if shape == SHAPE_ONE_ODD_ROW:
        if row == n_rows // 3:
            return np.float32(11.5)
        return np.float32(7.5)

    return np.float32(0.0)


class Fixture:
    """Row-major float32 features plus BOTH targets, mirroring `Dataset`.

    A fixture carries a class label and a regression target at once because
    `fixtures.mojo` does: the same feature bytes serve a classification check
    and a regression check, and `n_classes == 1` marks the regression ones.
    """

    def __init__(self, name: str, task: str, x, y, label, n_classes: int):
        self.name = name
        self.task = task
        self.x = np.asarray(x, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)
        self.label = np.asarray(label, dtype=np.int32)
        self.n_classes = n_classes

    @property
    def n_rows(self) -> int:
        return int(self.x.shape[0])

    @property
    def n_cols(self) -> int:
        return int(self.x.shape[1])


def hashed_classification(seed: int, n_rows: int, n_cols: int, n_classes: int) -> Fixture:
    """`fixtures.mojo:453-472`."""
    x = np.empty((n_rows, n_cols), dtype=np.float32)
    y = np.empty(n_rows, dtype=np.float32)
    lab = np.empty(n_rows, dtype=np.int32)
    for r in range(n_rows):
        for c in range(n_cols):
            x[r, c] = signed_unit(cell_hash(seed, r, c, SALT_X))
        k = (cell_hash(seed, r, 0, SALT_LABEL) >> 40) % n_classes
        lab[r] = k
        y[r] = np.float32(k)
    return Fixture("hashed_cls", "classification", x, y, lab, n_classes)


def hashed_regression(seed: int, n_rows: int, n_cols: int) -> Fixture:
    """`fixtures.mojo:475-487`."""
    x = np.empty((n_rows, n_cols), dtype=np.float32)
    y = np.empty(n_rows, dtype=np.float32)
    lab = np.zeros(n_rows, dtype=np.int32)
    for r in range(n_rows):
        for c in range(n_cols):
            x[r, c] = signed_unit(cell_hash(seed, r, c, SALT_X))
        y[r] = signed_unit(cell_hash(seed, r, 0, SALT_Y))
    return Fixture("hashed_reg", "regression", x, y, lab, 1)


def shaped_dataset(name: str, seed: int, n_rows: int, shapes: List[int]) -> Fixture:
    """`fixtures.mojo:614-627`. Note the hash takes the COLUMN INDEX, not the
    shape id, so two columns of the same shape are still independent draws."""
    n_cols = len(shapes)
    x = np.empty((n_rows, n_cols), dtype=np.float32)
    y = np.empty(n_rows, dtype=np.float32)
    lab = np.empty(n_rows, dtype=np.int32)
    for r in range(n_rows):
        for c in range(n_cols):
            x[r, c] = shaped_value(seed, shapes[c], r, c, n_rows)
        lab[r] = (cell_hash(seed, r, 0, SALT_LABEL) >> 40) % 2
        y[r] = signed_unit(cell_hash(seed, r, 0, SALT_Y))
    return Fixture(name, "classification", x, y, lab, 2)


def analytic_separable_gap(seed: int) -> Fixture:
    """`fixtures.mojo:667-718`. 128 class-0 rows in [0,1), 128 class-1 rows in
    (9,10], the band [1,9] empty, feature 1 pure noise."""
    n, half = 256, 128
    x = np.empty((n, 2), dtype=np.float32)
    y = np.empty(n, dtype=np.float32)
    lab = np.empty(n, dtype=np.int32)
    for r in range(n):
        h0 = cell_hash(seed, r, 0, SALT_X)
        h1 = cell_hash(seed, r, 1, SALT_X)
        if r < half:
            x[r, 0] = unit_float(h0)
            lab[r] = 0
            y[r] = np.float32(0.0)
        else:
            x[r, 0] = np.float32(np.float32(9.0) + unit_float20_open(h0))
            lab[r] = 1
            y[r] = np.float32(1.0)
        x[r, 1] = unit_float(h1)
    return Fixture("separable_gap", "classification", x, y, lab, 2)


def analytic_regression_step(seed: int) -> Fixture:
    """`fixtures.mojo:721-775`. y = 2.0 on x0 in [0,4), y = -3.0 on x0 in
    (6,10], the band [4,6] empty."""
    n, half = 256, 128
    x = np.empty((n, 2), dtype=np.float32)
    y = np.empty(n, dtype=np.float32)
    lab = np.zeros(n, dtype=np.int32)
    for r in range(n):
        h0 = cell_hash(seed, r, 0, SALT_X)
        h1 = cell_hash(seed, r, 1, SALT_X)
        if r < half:
            x[r, 0] = np.float32(np.float32(4.0) * unit_float(h0))
            y[r] = np.float32(2.0)
        else:
            x[r, 0] = np.float32(
                np.float32(6.0) + np.float32(4.0) * unit_float20_open(h0)
            )
            y[r] = np.float32(-3.0)
        x[r, 1] = unit_float(h1)
    return Fixture("regression_step", "regression", x, y, lab, 1)


def analytic_tie_pair(seed: int) -> Fixture:
    """`fixtures.mojo:778-833`. Feature 1 is a BIT-FOR-BIT copy of feature 0."""
    n, half = 256, 128
    x = np.empty((n, 3), dtype=np.float32)
    y = np.empty(n, dtype=np.float32)
    lab = np.empty(n, dtype=np.int32)
    for r in range(n):
        h0 = cell_hash(seed, r, 0, SALT_X)
        h2 = cell_hash(seed, r, 2, SALT_X)
        if r < half:
            v = unit_float(h0)
            lab[r] = 0
            y[r] = np.float32(0.0)
        else:
            v = np.float32(np.float32(9.0) + unit_float20_open(h0))
            lab[r] = 1
            y[r] = np.float32(1.0)
        x[r, 0] = v
        x[r, 1] = v
        x[r, 2] = unit_float(h2)
    return Fixture("tie_pair", "classification", x, y, lab, 2)


def analytic_all_constant() -> Fixture:
    """`fixtures.mojo:836-885`. Every column exactly constant, labels `r % 2`
    so the node is IMPURE -- "no split" cannot be right for the wrong reason."""
    n = 128
    consts = [np.float32(0.0), np.float32(-1.0), np.float32(3.25), np.float32(1000000.0)]
    x = np.empty((n, len(consts)), dtype=np.float32)
    y = np.empty(n, dtype=np.float32)
    lab = np.empty(n, dtype=np.int32)
    for r in range(n):
        for c in range(len(consts)):
            x[r, c] = consts[c]
        lab[r] = r % 2
        y[r] = np.float32(r % 2)
    return Fixture("all_constant", "classification", x, y, lab, 2)


def const_heavy_shapes() -> List[int]:
    """`fixture_parity_check.mojo:const_heavy_shapes`."""
    s = [SHAPE_HASHED] * CONST_HEAVY_HASHED
    s += [SHAPE_CONSTANT] * CONST_HEAVY_PER_SHAPE
    s += [SHAPE_NEAR_CONST_BELOW] * CONST_HEAVY_PER_SHAPE
    s += [SHAPE_NEAR_CONST_EQUAL] * CONST_HEAVY_PER_SHAPE
    return s


def build_fixtures() -> List[Fixture]:
    """The eight, IN THE ORDER `fixture_parity_check.mojo` dumps them. The
    order is part of the parity comparison: a reader that silently reorders
    could pair the wrong fixture with the wrong block and still match on the
    two that happen to share a shape."""
    return [
        hashed_classification(SEED, HASHED_ROWS, HASHED_COLS, HASHED_CLASSES),
        hashed_regression(SEED, HASHED_ROWS, HASHED_COLS),
        shaped_dataset("shaped_all", SEED, SHAPED_ROWS, list(range(N_SHAPES))),
        shaped_dataset("shaped_constant_heavy", SEED, SHAPED_ROWS, const_heavy_shapes()),
        analytic_separable_gap(SEED),
        analytic_regression_step(SEED),
        analytic_tie_pair(SEED),
        analytic_all_constant(),
    ]


# ---------------------------------------------------------------------------
# STEP 1: fixture parity, cell for cell, on BIT PATTERNS.
# ---------------------------------------------------------------------------
def hex8(x) -> str:
    return format(int(np.float32(x).view(np.uint32)), "08x")


class ParityResult:
    def __init__(self):
        self.x_cells = 0
        self.y_cells = 0
        self.label_cells = 0
        self.mismatches: List[str] = []

    @property
    def total(self) -> int:
        return self.x_cells + self.y_cells + self.label_cells

    @property
    def ok(self) -> bool:
        return not self.mismatches


def parse_dump(path: str) -> Dict[str, dict]:
    """Parse `fixture_parity_check.mojo`'s dump. Order is preserved."""
    out: Dict[str, dict] = {}
    order: List[str] = []
    cur = None
    with open(path, "r") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            if line.startswith("FIXTURE "):
                parts = line.split()
                name = parts[1]
                meta = dict(p.split("=", 1) for p in parts[2:])
                cur = {
                    "name": name,
                    "seed": int(meta["seed"]),
                    "rows": int(meta["rows"]),
                    "cols": int(meta["cols"]),
                    "classes": int(meta["classes"]),
                    "task": meta["task"],
                    "rows_data": [],
                }
                out[name] = cur
                order.append(name)
                continue
            if line.startswith("END "):
                cur = None
                continue
            if line.startswith("R "):
                if cur is None:
                    raise SystemExit(f"{path}:{lineno}: row outside a FIXTURE block")
                body = line[2:]
                head, tail = body.split(" | ", 1)
                head_parts = head.split()
                row_idx = int(head_parts[0])
                cells = head_parts[1:]
                tail_parts = tail.split()
                cur["rows_data"].append((row_idx, cells, tail_parts[0], int(tail_parts[1])))
                continue
            raise SystemExit(f"{path}:{lineno}: unrecognized line: {line!r}")
    out["__order__"] = order  # type: ignore[assignment]
    return out


def verify_parity(fixtures: List[Fixture], dump_path: str) -> ParityResult:
    """Compare every cell's HEX BITS. The decimal half of `<dec>/<hex>` is
    ignored on purpose: `String(Float32)` does not round-trip in this toolchain,
    so decimal agreement is not evidence and decimal disagreement is not a
    defect."""
    dumped = parse_dump(dump_path)
    order = dumped.pop("__order__")
    res = ParityResult()

    ours = [f.name for f in fixtures]
    if order != ours:
        res.mismatches.append(
            f"FIXTURE ORDER differs: dump {order} vs python {ours}"
        )
        return res

    for f in fixtures:
        d = dumped[f.name]
        if d["seed"] != SEED:
            res.mismatches.append(f"{f.name}: dump seed {d['seed']} != {SEED}")
        if (d["rows"], d["cols"], d["classes"], d["task"]) != (
            f.n_rows,
            f.n_cols,
            f.n_classes,
            f.task,
        ):
            res.mismatches.append(
                f"{f.name}: header {(d['rows'], d['cols'], d['classes'], d['task'])} "
                f"!= python {(f.n_rows, f.n_cols, f.n_classes, f.task)}"
            )
            continue
        if len(d["rows_data"]) != f.n_rows:
            res.mismatches.append(
                f"{f.name}: dump has {len(d['rows_data'])} rows, python has {f.n_rows}"
            )
            continue
        for row_idx, cells, ytext, label in d["rows_data"]:
            if len(cells) != f.n_cols:
                res.mismatches.append(
                    f"{f.name} row {row_idx}: {len(cells)} cells, expected {f.n_cols}"
                )
                continue
            for c, cell in enumerate(cells):
                theirs = cell.split("/", 1)[1]
                mine = hex8(f.x[row_idx, c])
                res.x_cells += 1
                if theirs != mine:
                    if len(res.mismatches) < 40:
                        res.mismatches.append(
                            f"{f.name} X[{row_idx},{c}]: mojo {theirs} python {mine}"
                        )
            res.y_cells += 1
            if ytext.split("/", 1)[1] != hex8(f.y[row_idx]):
                if len(res.mismatches) < 40:
                    res.mismatches.append(
                        f"{f.name} y[{row_idx}]: mojo {ytext.split('/', 1)[1]} "
                        f"python {hex8(f.y[row_idx])}"
                    )
            res.label_cells += 1
            if label != int(f.label[row_idx]):
                if len(res.mismatches) < 40:
                    res.mismatches.append(
                        f"{f.name} label[{row_idx}]: mojo {label} python {int(f.label[row_idx])}"
                    )
    return res


# ---------------------------------------------------------------------------
# STEP 2: the reference dump.
# ---------------------------------------------------------------------------
def split_mask(n_rows: int) -> np.ndarray:
    """TRAIN iff `(r % 4) < 2`. See the module docstring for why neither index
    parity nor a fixed prefix works on these fixtures."""
    idx = np.arange(n_rows)
    return (idx % 4) < 2


SPLIT_RULE = "train iff (row_index % 4) < 2, holdout otherwise"


def f64_hex(v: float) -> str:
    """`<hexbits>` of a float64, big-endian, so the text form of a metric is
    exact. Paired with the decimal in the dumped file as `<decimal>/<hexbits>`;
    a reader that needs the value must read the HEX."""
    return struct.pack(">d", float(v)).hex()


def f64_field(v: float) -> str:
    return f"{float(v):.17g}/{f64_hex(v)}"


def default_params(task: str, random_state: int) -> Tuple[object, dict]:
    """Construct with sklearn's OWN DEFAULTS and set nothing but the seed.

    This is deliberate. Writing the defaults out by hand and passing them
    explicitly would make "these are sklearn's defaults" a claim in a comment;
    constructing with none of them makes it a mechanical fact, and the dumped
    `get_params()` then shows a version bump that MOVES a default as a diff in
    the committed file rather than as silence.
    """
    if task == "classification":
        est = ExtraTreesClassifier(random_state=random_state)
    else:
        est = ExtraTreesRegressor(random_state=random_state)
    params = est.get_params()

    # The five the brief names, asserted rather than assumed. If a future
    # sklearn moves one of these, this raises instead of quietly changing what
    # the band means.
    expected_max_features = "sqrt" if task == "classification" else 1.0
    checks = [
        ("bootstrap", False),
        ("max_features", expected_max_features),
        ("n_estimators", 100),
        ("min_samples_split", 2),
        ("min_samples_leaf", 1),
    ]
    for key, want in checks:
        got = params[key]
        if got != want:
            raise SystemExit(
                f"sklearn {sklearn.__version__} default {key}={got!r} for {task}, "
                f"expected {want!r}. The band's meaning changed; do not regenerate "
                f"silently -- read the release notes and update this file's docstring."
            )
    return est, params


def measure(fixture: Fixture) -> dict:
    """Fit at 21 seeds, return the band plus the tree-shape statistics."""
    mask = split_mask(fixture.n_rows)
    x_tr, x_te = fixture.x[mask], fixture.x[~mask]
    if fixture.task == "classification":
        y_tr, y_te = fixture.label[mask], fixture.label[~mask]
    else:
        y_tr, y_te = fixture.y[mask], fixture.y[~mask]

    scores: List[float] = []
    depths: List[float] = []
    leaves: List[float] = []
    params = None
    for s in range(N_SKLEARN_SEEDS):
        est, params = default_params(fixture.task, s)
        est.fit(x_tr, y_tr)
        pred = est.predict(x_te)
        if fixture.task == "classification":
            scores.append(float(np.mean(pred == y_te)))
        else:
            scores.append(float(np.mean((pred.astype(np.float64) - y_te.astype(np.float64)) ** 2)))
        depths.append(float(np.mean([t.get_depth() for t in est.estimators_])))
        leaves.append(float(np.mean([t.get_n_leaves() for t in est.estimators_])))

    def band(v: List[float]) -> Tuple[float, float, float]:
        a = np.sort(np.asarray(v, dtype=np.float64))
        return float(a[0]), float(a[len(a) // 2]), float(a[-1])

    return {
        "n_train": int(mask.sum()),
        "n_holdout": int((~mask).sum()),
        "metric": "accuracy" if fixture.task == "classification" else "mse",
        "score": band(scores),
        "depth": band(depths),
        "leaves": band(leaves),
        "params": params,
    }


DEGENERATE = {
    "shaped_constant_heavy": (
        "45 of 48 columns fail the float32 1e-7 constant test; max_features="
        "'sqrt' draws 6, so P(all six constant) = C(45,6)/C(48,6) ~= 0.664. "
        "DEVIATIONS 132 and 151 predict OUR trees are SHALLOWER here."
    ),
    "all_constant": (
        "every column exactly constant, so NO split exists at any node. "
        "sklearn's own trees must be depth 0 with exactly 1 leaf. Labels are "
        "r % 2 so the node is impure and 'no split' cannot be right for the "
        "wrong reason."
    ),
}


def write_reference(
    fixtures: List[Fixture], parity: ParityResult, out_path: str, dump_path: str
) -> None:
    lines: List[str] = []
    w = lines.append

    w("# extratrees sklearn QUALITY-BAND reference, v1")
    w("#")
    w("# THIS IS NOT A CORRECTNESS GATE. Bitwise parity with scikit-learn is")
    w("# impossible by construction (DEVIATIONS.md 130: their draws come off one")
    w("# sequential xorshift stream whose ORDER depends on the data; ours are")
    w("# counter-based and order-independent). Deviations 131 and 132 widen the")
    w("# gap further. A number outside a band below is a FINDING TO REPORT, never")
    w("# a target to tune toward.")
    w("#")
    w("# Regenerate:  see extratrees/tools/README_sklearn_reference.md")
    w("#")
    w("# Every float is written as <decimal>/<hexbits>, the hexbits being the")
    w("# big-endian IEEE-754 float64 bit pattern. READ THE HEX. `String(Float32)`")
    w("# does not round-trip in this toolchain and decimal text is not evidence.")
    w("")
    w(f"version sklearn {sklearn.__version__}")
    w(f"version numpy {np.__version__}")
    w(f"version python {sys.version.split()[0]}")
    w(f"fixture_seed {SEED}")
    w(f"sklearn_seeds {N_SKLEARN_SEEDS} (random_state = 0 .. {N_SKLEARN_SEEDS - 1})")
    w(f"split_rule {SPLIT_RULE}")
    w("")

    w("# Fixture parity against extratrees/mojo_only/fixture_parity_check.mojo,")
    w("# compared on FLOAT BIT PATTERNS, cell for cell. The numbers written here")
    w("# are the actual comparison counts, not a shape calculation.")
    w(f"parity status {'EXACT' if parity.ok else 'MISMATCH'}")
    w(f"parity x_cells {parity.x_cells}")
    w(f"parity y_cells {parity.y_cells}")
    w(f"parity label_cells {parity.label_cells}")
    w(f"parity total_cells {parity.total}")
    w(f"parity mismatches {len(parity.mismatches)}")
    # The dump is NOT committed -- it is 1.5 MB of regenerable text -- so its
    # sha256 is recorded instead. A regeneration against a dump produced by a
    # CHANGED `fixtures.mojo` moves this line, which is the only way a stale or
    # substituted dump becomes visible in a diff.
    with open(dump_path, "rb") as fh:
        w(f"parity dump_sha256 {hashlib.sha256(fh.read()).hexdigest()}")
    w("")

    for task in ("classification", "regression"):
        _, params = default_params(task, 0)
        w(f"# scikit-learn's OWN defaults for {task}; only random_state is set.")
        for key in sorted(params):
            if key == "random_state":
                continue
            w(f"params {task} {key} {params[key]!r}")
        w("")

    for f in fixtures:
        m = measure(f)
        w(f"fixture {f.name}")
        w(f"  task {f.task}")
        w(f"  seed {SEED}")
        w(f"  n_rows {f.n_rows}")
        w(f"  n_cols {f.n_cols}")
        w(f"  n_classes {f.n_classes}")
        w(f"  n_train {m['n_train']}")
        w(f"  n_holdout {m['n_holdout']}")
        w(f"  split {SPLIT_RULE}")
        w(f"  metric {m['metric']}")
        lo, mid, hi = m["score"]
        w(f"  {m['metric']}_min {f64_field(lo)}")
        w(f"  {m['metric']}_median {f64_field(mid)}")
        w(f"  {m['metric']}_max {f64_field(hi)}")
        lo, mid, hi = m["depth"]
        w(f"  mean_depth_min {f64_field(lo)}")
        w(f"  mean_depth_median {f64_field(mid)}")
        w(f"  mean_depth_max {f64_field(hi)}")
        lo, mid, hi = m["leaves"]
        w(f"  mean_leaves_min {f64_field(lo)}")
        w(f"  mean_leaves_median {f64_field(mid)}")
        w(f"  mean_leaves_max {f64_field(hi)}")
        if f.name in DEGENERATE:
            w(f"  degenerate yes")
            w(f"  degenerate_note {DEGENERATE[f.name]}")
        else:
            w(f"  degenerate no")
        w("end_fixture")
        w("")

    w("# end")
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--dump",
        required=True,
        help="output of `mojo run -I . extratrees/mojo_only/fixture_parity_check.mojo`",
    )
    ap.add_argument(
        "--out",
        default="extratrees/tools/sklearn_reference.txt",
        help="where to write the reference dump",
    )
    ap.add_argument(
        "--parity-only",
        action="store_true",
        help="verify fixture parity and stop; train nothing, write nothing",
    )
    args = ap.parse_args()

    fixtures = build_fixtures()
    parity = verify_parity(fixtures, args.dump)

    print(f"fixture parity: {'EXACT' if parity.ok else 'MISMATCH'}")
    print(f"  X cells compared:      {parity.x_cells}")
    print(f"  y cells compared:      {parity.y_cells}")
    print(f"  label cells compared:  {parity.label_cells}")
    print(f"  total cells compared:  {parity.total}")
    if not parity.ok:
        print(f"  mismatches (first {min(40, len(parity.mismatches))}):")
        for m in parity.mismatches[:40]:
            print(f"    {m}")
        print()
        print(
            "REFUSING to train or write a reference. A quality band measured on"
            " data that is not our data is a confound, not a reference."
        )
        return 1

    if args.parity_only:
        return 0

    write_reference(fixtures, parity, args.out, args.dump)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
