"""CATBOOST'S OWN LOGLOSS LEAF VALUES, at ten Newton iterations and at one.

    pixi run -e bench logloss-leaf-oracle-gen   # bench/oracle_logloss_leaves.txt

WHY THIS FILE EXISTS
--------------------
CatBoost's default leaf estimator for Logloss is Newton with **TEN**
iterations and `AnyImprovement` backtracking
(`private/libs/options/catboost_options.cpp:157-164`, then `:315-329`).
RMSE gets one. This port implements the whole ten-iteration descent walker
(`gbdt/methods/leaves_estimation/{descent_helpers,step_estimator,
pointwise_oracle}.mojo`) and until this file existed NOTHING had ever
compared its output to CatBoost's.

The one gate that existed, `mojo_only/logloss_estimator_check.mojo`,
compares the device walker against a float64 host reimplementation written
in the same file. That is a good gate for the device path and it has teeth
-- truncating the simulation to one iteration fails all twelve leaves --
but it is not a gate against CatBoost. A ten-iteration Newton descent with
backtracking has many places to be subtly wrong in a way that a same-author
reimplementation reproduces faithfully.

`tools/catboost_loss_oracle.py` dumps nine objectives and Logloss is not
one of them; its leaf comparison is TREE 0 ONLY. This one is Logloss, every
tree, every leaf, at two iteration counts.

WHICH ARM OF THEIRS. Their CPU, because `task_type="GPU"` raises on Apple
silicon. So this is our GPU against their CPU, said here as it is said
beside every number in this repository. It matters more here than usual:
CatBoost's CPU and GPU leaf estimators are two DIFFERENT implementations of
the same descent (theirs at `private/libs/algo/approx_calcer/
gradient_walker.h` versus `cuda/methods/leaves_estimation/
descent_helpers.cpp`, which is the one this port mirrors), so a gap here is
first a CPU-versus-GPU question and only then a port question. What makes
the comparison legitimate anyway is that the two agree on the arithmetic
that matters at unit weights: their CPU's Newton denominator is
`-sumDer2 + l2 * (sumAllWeights / allDocCount)`
(`private/libs/algo_helpers/online_predictor.h:165-169`) and with every
weight 1.0 that scaling factor is exactly 1, which is the GPU's raw lambda.
THE FIXTURE THEREFORE USES UNIT WEIGHTS AND MUST KEEP DOING SO.

SAME EVERYTHING EXCEPT THE DEVICE
---------------------------------
The pins are `tools/catboost_arm.py:55-75`'s, one for one, with the single
deliberate exception that `leaf_estimation_iterations` is 10 on the L1 arm
rather than 1 -- ten is the whole point of the file.

  boosting_type      Plain      ordered boosting is not ported
  bootstrap_type     No         their default reweights the target BEFORE
                                the derivatives; a defaults run lines up
                                with nothing
  rsm                1.0        no feature subsampling
  has_time           True       keeps them from permuting rows
  boost_from_average False      their Logloss default is True; it starts
                                the cursor at the prior log-odds, which is
                                real behaviour this port does not
                                implement and would show up as a constant
                                offset in every leaf
  random_strength    0.0        their last source of randomness here
  model_shrink_rate  0.0        not implemented
  leaf_estimation_method        Newton, their Logloss default
  leaf_estimation_backtracking  AnyImprovement, their default

THE LEAF ORDER IS THE HARD PART, AND IT IS PROVED HERE RATHER THAN ASSUMED
--------------------------------------------------------------------------
A permutation of leaves that sums the same is exactly the failure mode a
green check hides. Two independent facts fix the mapping and BOTH are
recomputed on every generation, with the residuals written into the fixture
as the `armmap` record so the Mojo side reads them rather than trusting a
comment:

  1. THEIR APPLIER. `CalcIndexesBasic` accumulates
     `indexesVec[docId] |= (binFeature >= borderVal) << depth`
     (`libs/model/cpu/evaluator_impl.cpp:26-40`), where `depth` is the
     position of the split in the tree's OWN split list. Their trainer
     builds the same index with `splitWeight = 1 << splitParams.Depth`
     (`private/libs/algo/index_calcer.cpp:330`). So leaf index is the
     split outcomes read as a binary number with LEVEL 0 AS THE LEAST
     SIGNIFICANT BIT.
  2. NOTHING PERMUTES IT ON THE WAY INTO THE FILE.
     `TObliviousTreeBuilder::AddTree` appends `treeLeafValues` verbatim and
     `Trees.emplace_back(modelSplits)` keeps the split order it was given
     (`libs/model/model_build_helper.cpp:163-173`); `Build` re-indexes the
     split IDENTIFIERS through `BinFeatureIndexes` but never reorders a
     tree's splits or its leaves (`:176-201`).

Recomputed here as `map_residual`: every training row's leaf index is
rebuilt from the JSON splits under that convention, the model is replayed
as `sum over trees of leaf_values[leaf(row)]`, and the result is differenced
against `model.predict(x, prediction_type="RawFormulaVal")`. A wrong
convention (bit-reversed, or the splits read in the wrong order) does not
produce a small residual -- it produces a different model. A SECOND,
independent route checks the same thing: `model.calc_leaf_indexes(pool)`
returns their applier's own per-tree leaf index for every row, and
`leaf_index_mismatches` counts the rows where our reconstruction disagrees
with it. Both must be zero-ish or this generator refuses to write.

EVERY LEAF VALUE IN A TREE IS DISTINCT
--------------------------------------
`min_leaf_gap` is the smallest absolute difference between any two leaf
values inside one tree, over all trees, and the generator REFUSES to write
a fixture where it is not comfortably larger than the tolerance the Mojo
side compares at. Without that, a check that permuted the leaves would
still look green.

FLOAT PERSISTENCE
-----------------
Every float is `<decimal>/<hex bits>` and the hex is authoritative, the
same convention as `tools/catboost_loss_oracle.py` and
`gbdt/models/model_text.mojo`'s `f32_token` / `parse_f32`. String(Float32)
on this toolchain is one ULP wrong for 0.46% of float32 values.
"""
import json
import os
import struct
import sys
import tempfile

import numpy as np
import catboost


ROWS = 3000
FEATS = 8
DEPTH = 3
TREES = 12
LEARNING_RATE = 0.3
L2 = 3.0
BORDER_COUNT = 32
SEED = 0
LOGLOSS_BORDER = 0.5

MASK = (1 << 64) - 1


def splitmix(x: int) -> int:
    z = (x + 0x9E3779B97F4A7C15) & MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return z ^ (z >> 31)


def frac(i: int, salt: int) -> float:
    return (splitmix((i * 2654435761 + salt) & MASK) >> 11) * (
        1.0 / 9007199254740992.0
    )


def f32_token(v) -> str:
    f = np.float32(v)
    bits = struct.unpack("<I", struct.pack("<f", float(f)))[0]
    text = str(f)
    if np.float32(float(text)) != f:
        text = repr(float(f))
    return "%s/%08x" % (text, bits)


def f64_token(v) -> str:
    d = float(v)
    bits = struct.unpack("<Q", struct.pack("<d", d))[0]
    return "%r/%016x" % (d, bits)


def build():
    """Hashed features and a BINARY target with planted structure.

    Column 5 is quantized to seven distinct values so its fold count is
    decided by the data rather than by `border_count`; column 6 never
    enters the target at all. Both are `tools/catboost_loss_oracle.py`'s
    devices and both are kept, because a fixture whose columns are
    interchangeable cannot tell a permutation defect from a value defect.

    The label is a NOISY threshold rather than a clean one: a separable
    label drives the Newton step to a leaf's clipped extreme, where every
    iteration after the first moves almost nothing and a ten-iteration
    walker is indistinguishable from a one-iteration one. The whole point
    of the fixture is that the ten iterations MATTER, and the L3 control
    measures whether they did.

    SIX FEATURES CARRY THE TARGET AT COMPARABLE WEIGHT, and that is the one
    thing here chosen rather than inherited. With a single dominant column
    (the shape `tools/catboost_loss_oracle.py` uses) CatBoost splits the
    SAME feature twice inside one oblivious tree, which makes half the leaf
    combinations unreachable and pins those leaves at exactly 0.0 on both
    arms. Ten of sixty-four leaves came back exactly zero that way. Tied
    leaf values defeat a PLACEMENT check -- a permutation that moves two
    zeros is invisible -- so the target was spread until no two leaves in a
    tree tie. The choice was made before any comparison against our port
    was run and is not a fixture picked by what it scores.
    """
    x = np.empty((ROWS, FEATS), dtype=np.float32)
    for f in range(FEATS):
        for r in range(ROWS):
            v = (frac(r * FEATS + f, 5) - 0.5) * 2.0
            if f == 5:
                v = float(np.floor(v * 3.5)) / 3.5
            x[r, f] = np.float32(v)

    xd = x.astype(np.float64)
    noise = np.array([frac(r, 77) - 0.5 for r in range(ROWS)])
    logit = (
        1.6 * xd[:, 0]
        - 1.5 * xd[:, 1]
        + 1.4 * xd[:, 2]
        + 1.3 * xd[:, 3]
        - 1.2 * xd[:, 4]
        + 1.1 * xd[:, 7]
        + 0.9 * xd[:, 0] * xd[:, 7]
        + 0.8 * noise
    )
    y = (logit > 0.0).astype(np.float32)
    return x, y


def leaf_indexes_from_json(raw, x):
    """Every row's leaf, per tree, under THEIR convention: level 0 is the
    least significant bit and `level` is the position in the tree's own
    split list (`evaluator_impl.cpp:26-40`, `index_calcer.cpp:330`).

    The predicate is `x > border` on the raw float, which is the same
    statement as their quantized `bin > splitIdx`: a float feature's bin is
    the number of borders strictly below the value, so bin exceeds the
    split's border position exactly when the value exceeds the border.
    """
    out = []
    for tree in raw["oblivious_trees"]:
        splits = tree.get("splits", [])
        idx = np.zeros(x.shape[0], dtype=np.int64)
        for depth, s in enumerate(splits):
            fi = int(s["float_feature_index"])
            border = np.float32(s["border"])
            idx |= (x[:, fi] > border).astype(np.int64) << depth
        out.append(idx)
    return out


def fit_one(pool, x, y, iters):
    model = catboost.CatBoostClassifier(
        iterations=TREES,
        depth=DEPTH,
        learning_rate=LEARNING_RATE,
        l2_leaf_reg=L2,
        border_count=BORDER_COUNT,
        loss_function="Logloss",
        grow_policy="SymmetricTree",
        boosting_type="Plain",
        bootstrap_type="No",
        rsm=1.0,
        has_time=True,
        random_seed=SEED,
        random_strength=0.0,
        model_shrink_rate=0.0,
        boost_from_average=False,
        leaf_estimation_method="Newton",
        leaf_estimation_iterations=int(iters),
        leaf_estimation_backtracking="AnyImprovement",
        score_function="Cosine",
        thread_count=1,
        logging_level="Silent",
        allow_writing_files=False,
    )
    model.fit(pool)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "m.json")
        model.save_model(p, format="json")
        with open(p) as fh:
            raw = json.load(fh)
    preds = model.predict(x, prediction_type="RawFormulaVal")
    return model, raw, np.asarray(preds, dtype=np.float64)


def main() -> int:
    out_path = os.environ.get(
        "LOGLOSS_LEAF_ORACLE_OUT", "bench/oracle_logloss_leaves.txt"
    )
    x, y = build()
    sys.stderr.write(
        "fixture: %d rows x %d feats, positives %d\n"
        % (ROWS, FEATS, int(y.sum()))
    )

    # THE GRID COMES FROM THE POOL, not from a fitted model: a saved model
    # carries only the borders its trees used, which is a different object
    # and reads as a total mismatch against a real binarizer. Same trap
    # `tools/catboost_oracle.py` documents at length.
    pool = catboost.Pool(x, y)
    pool.quantize(
        border_count=BORDER_COUNT, feature_border_type="GreedyLogSum"
    )
    d = tempfile.mkdtemp()
    bpath = os.path.join(d, "borders.tsv")
    pool.save_quantization_borders(bpath)
    grid = {}
    with open(bpath) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            fi, bv = line.split("\t")
            grid.setdefault(int(fi), []).append(float(bv))
    for fi in range(FEATS):
        grid.setdefault(fi, [])
        grid[fi] = sorted(grid[fi])

    lines = []
    lines.append(
        "# CatBoost %s Logloss LEAF oracle. tools/catboost_logloss_leaf_oracle.py"
        % catboost.__version__
    )
    lines.append("# floats are <decimal>/<hex bits>; the hex is authoritative")
    lines.append("version %s" % catboost.__version__)
    lines.append(
        "dims %d %d %d %d %d" % (ROWS, FEATS, DEPTH, TREES, BORDER_COUNT)
    )
    lines.append("hyper %s %s" % (f32_token(LEARNING_RATE), f32_token(L2)))
    lines.append("border %s" % f32_token(LOGLOSS_BORDER))
    for fi in range(FEATS):
        lines.append(
            "borders %d %d %s"
            % (fi, len(grid[fi]), " ".join(f32_token(v) for v in grid[fi]))
        )
    for fi in range(FEATS):
        lines.append(
            "xcol %d %s" % (fi, " ".join(f32_token(v) for v in x[:, fi]))
        )
    lines.append("target %s" % " ".join(f32_token(v) for v in y))

    # L1 is the DEFAULT configuration (ten Newton iterations) and L2 is the
    # same fit at one. Their names are the Mojo side's keys.
    for arm_name, iters in (("iters10", 10), ("iters1", 1)):
        sys.stderr.write("fitting %s ...\n" % arm_name)
        model, raw, preds = fit_one(pool, x, y, iters)

        trees = raw["oblivious_trees"]
        if len(trees) != TREES:
            raise SystemExit(
                "%s produced %d trees, asked for %d"
                % (arm_name, len(trees), TREES)
            )

        sb = raw.get("scale_and_bias", [1.0, [0.0]])
        scale = float(sb[0])
        bias = sb[1]
        bias = float(bias[0]) if isinstance(bias, list) else float(bias)

        # ---- THE LEAF-ORDER PROOF, route 1: replay their own model.
        idxs = leaf_indexes_from_json(raw, x)
        replay = np.zeros(ROWS, dtype=np.float64)
        for ti, tree in enumerate(trees):
            lv = np.asarray(tree["leaf_values"], dtype=np.float64)
            replay += lv[idxs[ti]]
        replay = scale * replay + bias
        map_residual = float(np.max(np.abs(replay - preds)))

        # ---- route 2: their applier's own leaf indexes, per tree.
        try:
            theirs_idx = np.asarray(
                model.calc_leaf_indexes(pool), dtype=np.int64
            )
            mismatches = int(
                sum(
                    int(np.sum(theirs_idx[:, ti] != idxs[ti]))
                    for ti in range(TREES)
                )
            )
            have_route2 = 1
        except Exception as e:  # noqa: BLE001
            sys.stderr.write("  calc_leaf_indexes unavailable: %s\n" % e)
            mismatches = -1
            have_route2 = 0

        if map_residual > 1e-9:
            raise SystemExit(
                "leaf-order mapping is WRONG for %s: replaying their own "
                "model under `leaf = sum bit_d << d` misses their own "
                "predictions by %g. Do not write a fixture whose leaf "
                "order is not established." % (arm_name, map_residual)
            )
        if have_route2 and mismatches != 0:
            raise SystemExit(
                "leaf-order mapping disagrees with calc_leaf_indexes for "
                "%s on %d (row, tree) cells" % (arm_name, mismatches)
            )

        # ---- EVERY LEAF DISTINCT, per tree. A permutation that sums the
        # same is the failure mode this fixture exists to expose, and it is
        # only exposed if no two leaves in a tree are interchangeable.
        min_gap = float("inf")
        for tree in trees:
            lv = sorted(float(v) for v in tree["leaf_values"])
            for i in range(1, len(lv)):
                min_gap = min(min_gap, abs(lv[i] - lv[i - 1]))
        if min_gap < 1e-4:
            raise SystemExit(
                "%s has two leaf values within %g of each other in one "
                "tree; a placement check cannot tell them apart"
                % (arm_name, min_gap)
            )

        lines.append(
            "arm %s Logloss Newton %d AnyImprovement" % (arm_name, iters)
        )
        lines.append(
            "armscalebias %s %s %s"
            % (arm_name, f64_token(scale), f64_token(bias))
        )
        lines.append(
            "armmap %s %s %d %d %s"
            % (
                arm_name,
                f64_token(map_residual),
                mismatches,
                have_route2,
                f64_token(min_gap),
            )
        )
        for ti, tree in enumerate(trees):
            for lvl, s in enumerate(tree.get("splits", [])):
                lines.append(
                    "armsplit %s %d %d %d %s"
                    % (
                        arm_name,
                        ti,
                        lvl,
                        int(s["float_feature_index"]),
                        f32_token(s["border"]),
                    )
                )
            lines.append(
                "armleaves %s %d %s"
                % (
                    arm_name,
                    ti,
                    " ".join(f64_token(v) for v in tree["leaf_values"]),
                )
            )
        lines.append(
            "armpred %s %s" % (arm_name, " ".join(f64_token(v) for v in preds))
        )
        sys.stderr.write(
            "  map_residual %.3g  leaf-index mismatches %d  min leaf gap %.4g\n"
            % (map_residual, mismatches, min_gap)
        )

    with open(out_path, "w") as fh:
        fh.write("\n".join(lines))
        fh.write("\n")
    sys.stderr.write("wrote %s (%d lines)\n" % (out_path, len(lines)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
