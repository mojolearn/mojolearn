# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CatBoost's ACTUAL OUTPUT on a CATEGORICAL fixture, dumped for the port.

WHY THIS IS A SEPARATE FILE FROM `tools/catboost_oracle.py`
-----------------------------------------------------------
`tools/catboost_oracle.py` regenerates the three committed numeric fixtures
(`bench/oracle.txt`, `oracle100.txt`, `oracle254.txt`), which are this
repository's strongest evidence. Three things kept this out of it:

  * the PINS ARE DIFFERENT and two of them are new and load-bearing
    (`one_hot_max_size`, and a second reason for `has_time`);
  * this file needs a step that has no numeric counterpart -- CatBoost
    reports a one-hot split as (cat_feature_index, HASH), and the hash has
    to be resolved back to the dense category code before anything can be
    compared. That is a whole extra pass over the model, plus its
    verification;
  * a shared script with a mode flag lets a bug in the categorical arm
    change a numeric fixture. These three files are the reference the port
    is held to; they do not share a code path with an experiment.

WHAT CAN BE COMPARED, AND WHAT CANNOT -- READ THIS BEFORE CHANGING A PIN
------------------------------------------------------------------------
This port's categorical support has two arms, and only ONE of them can be
put against this oracle at all.

**THE ONE-HOT ARM CAN.** Their dispatch sends a categorical feature whose
learn-set cardinality is `1 < k <= one_hot_max_size` to equality splits and
gives it NO CTR (`greedy_tensor_search.cpp:469-471` returns from
`AddSimpleCtrs` before the projection is built; `AddOneHotFeatures` at
`:182-184` is the arm that takes it). One-hot statistics are sums over
rows, so they are permutation independent, they do not touch the target
ordering, and their candidate set is the full `bucketCount`
(`score_calcers.cpp:58-61`: `OneHotFeature` gets `bucketCount` splits where
an ordered feature gets `bucketCount - 1`, and `split.cpp:110-114` sets
`bucketCount` to the cat feature's `OnLearnOnly` unique count). That is
exactly what this port does: `build_layout(fold_counts, one_hot)` gives the
feature `k` folds and bins `0..k-1`, and the searcher emits
`BIN_SPLIT_TAKE_BIN`.

**THE CTR ARM CANNOT, AND THE BLOCKER IS NOT IN THIS PORT.** It is that
CatBoost's CPU learner cannot be configured with the CTR set this port
mirrors:

    IsSupportedCtrType(CPU, ...)   Borders Buckets BinarizedTargetMeanValue
                                   Counter
    IsSupportedCtrType(GPU, ...)   Borders Buckets FloatTargetMeanValue
                                   FeatureFreq
                                   (`private/libs/options/restrictions.h:18-48`)

`TCatFeatureParams.default()` in this port is their GPU `simple_ctr` --
`Borders` with three priors plus `CreateDefaultCounter(SimpleCtr)`, which
is `FeatureFreq` (`catboost_options.cpp:412-414, 449-452`). **FeatureFreq
does not exist on CPU.** Asking their CPU learner for it raises. So the
frequency column this port computes has no counterpart in any CPU model
that could be dumped here, and the closest CPU type (`Counter`) is a
different estimator with a different denominator, not a rename.

The `Borders` half is blocked twice over even so: their CPU default
`ctr_binarization` is `Uniform 15` (visible in any CPU model's
`features_info.ctrs`), where the GPU sets `MinEntropy 15` for a simple CTR
and `Median 15` for a tree CTR (`catboost_options.cpp:398-415`), and a CPU
`Borders` column is an ONLINE ordered statistic over their per-permutation
fold structure rather than the device pass this port runs. Two different
columns cannot produce comparable bins.

**AND FEATURE COMBINATIONS ARE NOT PORTED AT ALL.**
`gbdt/options/catboost_options.mojo`'s `TCatFeatureParams.check()` refuses
`max_ctr_complexity != 1` by name, where CatBoost's default is 4
(`cat_feature_options.cpp:231`). That is `PORTING.md` 91 and `NEXT_TWO.md`
rung 4, and it is a second, independent reason a CTR fixture is not
comparable today.

So this fixture is ONE-HOT ONLY, and that is a statement about what can be
verified rather than a fixture chosen to be passed. See DEVIATION 113.

SETTINGS PINNED, AND WHY EACH ONE
---------------------------------
Everything `tools/catboost_oracle.py` pins is pinned here for the same
reason and is not restated; what follows is what this fixture adds.

  one_hot_max_size  8       AT OR ABOVE EVERY CATEGORICAL CARDINALITY in
                            the fixture (3, 5, 8). This is the pin that
                            selects the arm: at their default of 2 the
                            k = 3, 5 and 8 columns would all become CTR
                            columns and nothing here would be comparable.
                            It is a real CatBoost option, not a fixture
                            convenience.
  max_ctr_complexity 1      Combinations are not ported and this port
                            REFUSES the option by name, so leaving
                            CatBoost at its default of 4 would be claiming
                            support this tree does not have. Inert on this
                            fixture -- no column reaches the CTR arm -- and
                            pinned anyway so the fixture states its own
                            configuration rather than depending on a
                            dispatch accident.
  has_time          True    ALREADY pinned by the numeric oracle to stop
                            row permutation, and it acquires a SECOND and
                            stronger reason here: `NeedShuffle`
                            (`private/libs/algo/preprocess.cpp:161-181`)
                            returns TRUE unconditionally when the pool has
                            any categorical feature and `has_time` is
                            false. With it false the learn pool this file
                            dumps `x` and `y` for is NOT the pool CatBoost
                            trained on.

THE FIXTURE'S SHAPE, AND ONE CARDINALITY DELIBERATELY LEFT OUT
--------------------------------------------------------------
4096 rows, 11 columns: eight numeric at a 100-border grid and three
categorical at k = 3, 5 and 8. The numeric columns are not decoration --
they put the one-hot candidates in COMPETITION with ordered candidates in
the same searcher, and at 100 borders they land in a different grid policy
(`OneByteFeatures`) from the categorical columns (`HalfByteFeatures`), so
the fixture exercises a mixed-policy compressed index that no numeric
fixture reaches.

**k = 2 IS EXCLUDED, and the reason is decided here rather than after
seeing a result.** For a two-category one-hot feature the candidates
`== code0` and `== code1` induce the SAME partition with the sides
swapped, and every score calcer in either implementation is symmetric in
the two children, so the two candidates tie EXACTLY. Which one wins is a
tie-break, not an algorithmic decision, and CatBoost's enumeration order is
its perfect-hash order while this port's is dense-code order. A fixture
built on that would report a divergence that is not one. Anything k >= 3
has no such degeneracy.

HOW A ONE-HOT SPLIT IS RESOLVED BACK TO A CATEGORY CODE
--------------------------------------------------------
CatBoost's model JSON reports a one-hot split as
`{"cat_feature_index": f, "value": H}` where H is
`(int)CityHash64(the category's string form)`. The Python package exposes
no hashing entry point, so the mapping is recovered FROM THE MODEL rather
than recomputed:

  * for each categorical column, a probe pool of `k` rows is built whose
    only varying column is that one, taking codes `0..k-1`;
  * `calc_leaf_indexes` gives each probe row's leaf per tree, and an
    oblivious tree's leaf index is the bitfield of its splits, bit `j`
    being split `j`'s outcome. So the code a split tests is the unique `c`
    whose bit is set.

That is asserted, not assumed: the resolution must be UNIQUE per split,
CONSISTENT across every tree that uses the same hash, and -- the check
that actually closes it -- the dumped structure is then walked in plain
numpy over the raw rows using the recovered codes, and its predictions
must reproduce `model.predict` to float tolerance. A wrong bit convention,
a wrong code or a wrong leaf order all break that reconstruction.

THE TEXT FORMAT IS THE NUMERIC ONE PLUS TWO RECORDS
----------------------------------------------------
`cat <flat_feature_index> <k>` declares a column categorical, and
`catsplit <tree> <flat_feature_index> <code>` is a one-hot split. `split`
keeps its exact numeric meaning. Splits of both kinds are written in DEPTH
ORDER inside a tree, interleaved, so a reader that appends in file order
gets the tree's levels in order without knowing which kind came next --
which is what `mojo_only/oracle_check.mojo` does, leaving the three
numeric fixtures byte-identical in their handling.

USAGE
  pixi run -e bench python tools/catboost_cat_oracle.py > bench/oracle_cat.json
  pixi run oracle
"""
import json
import os
import sys
import tempfile

import numpy as np
import catboost


ROWS = 4096
NUMERIC = 8
#: `(cardinality,)` per categorical column, appended after the numeric ones.
CAT_CARDINALITIES = [3, 5, 8]
DEPTH = 4
TREES = 12
LEARNING_RATE = 0.3
L2 = 3.0
BORDER_COUNT = 100
#: at or above `max(CAT_CARDINALITIES)`; see the module note.
ONE_HOT_MAX_SIZE = 8
SEED = 0


def build() -> tuple:
    """A fixture whose target needs BOTH kinds of split.

    The categorical terms are equality tests on single categories, which no
    threshold over the codes can express, and the numeric terms are linear,
    which no equality test can. A learner that ignored either kind is
    visibly worse, and a fixture that leaned entirely on one would let the
    other's candidates go unexercised.
    """
    rng = np.random.default_rng(SEED)
    xn = rng.normal(size=(ROWS, NUMERIC)).astype(np.float32)
    cols = []
    for k in CAT_CARDINALITIES:
        cols.append(rng.integers(0, k, size=ROWS))
    c = np.stack(cols, axis=1)

    y = (
        2.5 * (c[:, 0] == 1)
        + 1.7 * (c[:, 1] == 4)
        - 2.2 * (c[:, 2] == 6)
        + 1.1 * (c[:, 2] == 1)
        + 1.3 * xn[:, 0]
        - 0.9 * xn[:, 3]
        + 0.1 * rng.normal(size=ROWS)
    ).astype(np.float32)
    return xn, c, y


def as_object_matrix(xn: np.ndarray, c: np.ndarray) -> np.ndarray:
    """CatBoost refuses a float column declared categorical, so the pool is
    built as an object matrix with real ints in the categorical columns."""
    x = np.empty((ROWS, NUMERIC + len(CAT_CARDINALITIES)), dtype=object)
    for j in range(NUMERIC):
        x[:, j] = xn[:, j].astype(np.float64)
    for j in range(len(CAT_CARDINALITIES)):
        x[:, NUMERIC + j] = c[:, j].astype(np.int64)
    return x


def main() -> int:
    out_path = os.environ.get("ORACLE_CAT_OUT", "bench/oracle_cat.txt")
    feats = NUMERIC + len(CAT_CARDINALITIES)
    cat_flat = [NUMERIC + j for j in range(len(CAT_CARDINALITIES))]

    xn, c, y = build()

    # THE CARDINALITY THE FIXTURE INTENDS MUST BE THE ONE CATBOOST SEES.
    # Their arm is chosen off `GetUniqueValuesCounts().OnLearnOnly`, so a
    # code that happens to be absent from the sample would move the column
    # to a different `bucketCount` than this file claims -- and, at k = 1,
    # off the candidate list entirely (`AddOneHotFeatures`, `:182-184`).
    for j, k in enumerate(CAT_CARDINALITIES):
        seen = np.unique(c[:, j])
        if len(seen) != k or seen.min() != 0 or seen.max() != k - 1:
            raise SystemExit(
                "categorical column %d was meant to carry codes 0..%d and "
                "carries %s" % (j, k - 1, seen)
            )
        if k <= ONE_HOT_MAX_SIZE:
            continue
        raise SystemExit(
            "categorical column %d has cardinality %d above "
            "one_hot_max_size %d, which sends it to CatBoost's CTR arm; "
            "that arm is not comparable, see this file's module note"
            % (j, k, ONE_HOT_MAX_SIZE)
        )

    x_obj = as_object_matrix(xn, c)
    pool = catboost.Pool(x_obj, y, cat_features=cat_flat)

    # THE GRID COMES FROM THE POOL, NOT THE MODEL -- the same trap
    # `tools/catboost_oracle.py` documents. With categorical columns
    # present the file is keyed by FLOAT feature index, which is why the
    # numeric columns are placed FIRST: there the float index and the flat
    # index coincide and no remap is needed or guessed at.
    pool.quantize(border_count=BORDER_COUNT)
    d = tempfile.mkdtemp()
    bpath = os.path.join(d, "borders.tsv")
    pool.save_quantization_borders(bpath)
    grid: dict = {}
    with open(bpath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fi, bv = line.split("\t")
            grid.setdefault(int(fi), []).append(float(bv))
    for fi in grid:
        if fi >= NUMERIC:
            raise SystemExit(
                "save_quantization_borders emitted feature index %d, which "
                "is not a numeric column; the float-index/flat-index "
                "coincidence this file relies on does not hold" % fi
            )

    model = catboost.CatBoostRegressor(
        iterations=TREES,
        depth=DEPTH,
        learning_rate=LEARNING_RATE,
        l2_leaf_reg=L2,
        border_count=BORDER_COUNT,
        loss_function="RMSE",
        grow_policy="SymmetricTree",
        boosting_type="Plain",
        bootstrap_type="No",
        rsm=1.0,
        has_time=True,
        random_seed=SEED,
        model_shrink_rate=0.0,
        boost_from_average=False,
        leaf_estimation_iterations=1,
        random_strength=0.0,
        one_hot_max_size=ONE_HOT_MAX_SIZE,
        max_ctr_complexity=1,
        score_function=os.environ.get("ORACLE_SCORE", "Cosine"),
        logging_level="Silent",
        allow_writing_files=False,
    )
    model.fit(pool)

    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "m.json")
        model.save_model(p, format="json")
        with open(p) as f:
            raw = json.load(f)

    # --- resolve every one-hot hash back to its dense code --------------
    #
    # One probe pool per categorical column: `k` rows in which only that
    # column varies. An oblivious tree applies split `j` at depth `j` and
    # its leaf index is the bitfield of the outcomes, so bit `j` of the
    # probe row's leaf index IS split `j` evaluated on that code.
    hash_to_code: dict = {}
    for j, k in enumerate(CAT_CARDINALITIES):
        probe_xn = np.zeros((k, NUMERIC), dtype=np.float32)
        probe_c = np.zeros((k, len(CAT_CARDINALITIES)), dtype=np.int64)
        probe_c[:, j] = np.arange(k)
        px = np.empty((k, feats), dtype=object)
        for col in range(NUMERIC):
            px[:, col] = probe_xn[:, col].astype(np.float64)
        for col in range(len(CAT_CARDINALITIES)):
            px[:, NUMERIC + col] = probe_c[:, col]
        ppool = catboost.Pool(px, np.zeros(k, dtype=np.float32),
                              cat_features=cat_flat)
        leaf_idx = np.asarray(model.calc_leaf_indexes(ppool))

        for ti, tree in enumerate(raw["oblivious_trees"]):
            for depth, s in enumerate(tree.get("splits", [])):
                if s.get("split_type") != "OneHotFeature":
                    continue
                if s.get("cat_feature_index") != j:
                    continue
                fires = [
                    code for code in range(k)
                    if (int(leaf_idx[code, ti]) >> depth) & 1
                ]
                if len(fires) != 1:
                    raise SystemExit(
                        "one-hot split (tree %d, depth %d) on cat column %d "
                        "fires for %d of %d codes; the leaf-index bit "
                        "convention this file assumes does not hold"
                        % (ti, depth, j, len(fires), k)
                    )
                key = (j, int(s["value"]))
                if key in hash_to_code and hash_to_code[key] != fires[0]:
                    raise SystemExit(
                        "hash %d on cat column %d resolved to code %d here "
                        "and %d earlier"
                        % (key[1], j, fires[0], hash_to_code[key])
                    )
                hash_to_code[key] = fires[0]

    borders = [
        {"feature_index": fi, "borders": sorted(grid.get(fi, []))}
        for fi in range(NUMERIC)
    ]

    out_trees = []
    for tree in raw["oblivious_trees"]:
        splits = []
        for s in tree.get("splits", []):
            if s.get("split_type") == "OneHotFeature":
                cf = int(s["cat_feature_index"])
                splits.append({
                    "kind": "one_hot",
                    "flat_feature_index": NUMERIC + cf,
                    "cat_feature_index": cf,
                    "hash": int(s["value"]),
                    "code": hash_to_code[(cf, int(s["value"]))],
                })
            else:
                splits.append({
                    "kind": "float",
                    "flat_feature_index": int(s["float_feature_index"]),
                    "border": float(s["border"]),
                })
        out_trees.append(
            {"splits": splits, "leaf_values": tree.get("leaf_values", [])}
        )

    sb = raw.get("scale_and_bias", [1.0, [0.0]])
    scale = float(sb[0])
    bias = sb[1]
    bias = float(bias[0]) if isinstance(bias, list) else float(bias)

    preds = model.predict(x_obj)

    # --- THE CHECK THAT CLOSES THE HASH RESOLUTION ----------------------
    #
    # Walk the DUMPED structure over the RAW rows in numpy, using the
    # recovered codes and nothing from CatBoost's evaluator, and require it
    # to reproduce `model.predict`. A wrong bit convention, a wrong code,
    # a wrong leaf order or a wrong border all fail here, on the oracle's
    # own side, before the port is ever asked a question.
    recon = np.full(ROWS, bias, dtype=np.float64)
    for tree in out_trees:
        leaf = np.zeros(ROWS, dtype=np.int64)
        for depth, s in enumerate(tree["splits"]):
            if s["kind"] == "one_hot":
                bit = (
                    c[:, s["cat_feature_index"]] == s["code"]
                ).astype(np.int64)
            else:
                bit = (
                    xn[:, s["flat_feature_index"]].astype(np.float64)
                    > s["border"]
                ).astype(np.int64)
            leaf |= bit << depth
        recon += scale * np.asarray(tree["leaf_values"], dtype=np.float64)[leaf]
    worst = float(np.max(np.abs(recon - preds)))
    if not (worst < 1e-5):
        raise SystemExit(
            "the dumped structure does not reproduce CatBoost's own "
            "predictions (worst |delta| %g). The dump is wrong, not the "
            "port -- do not run the differential against it" % worst
        )

    n_one_hot = sum(
        1 for t in out_trees for s in t["splits"] if s["kind"] == "one_hot"
    )
    n_float = sum(
        1 for t in out_trees for s in t["splits"] if s["kind"] == "float"
    )
    if n_one_hot == 0:
        raise SystemExit(
            "CatBoost took no one-hot split on this fixture, so it is not a "
            "categorical fixture at all"
        )
    if n_float == 0:
        raise SystemExit(
            "CatBoost took no numeric split on this fixture, so the mixed "
            "candidate competition this fixture exists for never happened"
        )

    with open(out_path, "w") as f:
        f.write(
            "# CatBoost %s CATEGORICAL oracle (one-hot arm). Generated by"
            " tools/catboost_cat_oracle.py\n" % catboost.__version__
        )
        f.write("config %d %d %d %d %.17g %.17g %d\n" % (
            ROWS, feats, DEPTH, TREES, LEARNING_RATE, L2, BORDER_COUNT))
        f.write("train_mse %.17g\n" % float(np.mean((preds - y) ** 2)))
        f.write("baseline_mse %.17g\n" % float(np.mean((y - np.mean(y)) ** 2)))
        f.write("scale_and_bias %.17g %.17g\n" % (scale, bias))
        for j, k in enumerate(CAT_CARDINALITIES):
            f.write("cat %d %d\n" % (NUMERIC + j, k))
        for b in borders:
            f.write("borders %d %d %s\n" % (
                b["feature_index"], len(b["borders"]),
                " ".join("%.17g" % v for v in b["borders"])))
        for ti, tree in enumerate(out_trees):
            for s in tree["splits"]:
                if s["kind"] == "one_hot":
                    f.write("catsplit %d %d %d\n" % (
                        ti, s["flat_feature_index"], s["code"]))
                else:
                    f.write("split %d %d %.17g\n" % (
                        ti, s["flat_feature_index"], s["border"]))
            f.write("leaves %d %s\n" % (
                ti, " ".join("%.17g" % v for v in tree["leaf_values"])))
        for r in range(ROWS):
            row = [("%.9g" % xn[r, j]) for j in range(NUMERIC)]
            row += [("%d" % c[r, j]) for j in range(len(CAT_CARDINALITIES))]
            f.write("x %s\n" % " ".join(row))
        f.write("y %s\n" % " ".join("%.9g" % v for v in y))
        f.write("pred %s\n" % " ".join("%.9g" % v for v in preds))

    json.dump(
        {
            "catboost_version": catboost.__version__,
            "config": {
                "rows": ROWS,
                "feats": feats,
                "numeric": NUMERIC,
                "cat_cardinalities": CAT_CARDINALITIES,
                "depth": DEPTH,
                "trees": TREES,
                "learning_rate": LEARNING_RATE,
                "l2_leaf_reg": L2,
                "border_count": BORDER_COUNT,
                "one_hot_max_size": ONE_HOT_MAX_SIZE,
                "max_ctr_complexity": 1,
                "seed": SEED,
            },
            "scale_and_bias": raw.get("scale_and_bias"),
            "float_feature_borders": borders,
            "one_hot_hash_to_code": [
                {"cat_feature_index": kk[0], "hash": kk[1], "code": vv}
                for kk, vv in sorted(hash_to_code.items())
            ],
            "split_kind_counts": {"one_hot": n_one_hot, "float": n_float},
            "reconstruction_worst_abs_delta": worst,
            "trees": out_trees,
            "prediction_head": [float(v) for v in preds[:32]],
            "prediction_mean": float(np.mean(preds)),
            "train_mse": float(np.mean((preds - y) ** 2)),
            "mean_baseline_mse": float(np.mean((y - np.mean(y)) ** 2)),
        },
        sys.stdout,
        indent=1,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
