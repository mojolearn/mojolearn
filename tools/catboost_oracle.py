"""CatBoost's ACTUAL OUTPUT, dumped so the port can be compared against it.

WHY THIS EXISTS, AND WHAT IT REPLACES
-------------------------------------
`tools/catboost_reference.py` runs CatBoost and prints ONE number, ms/tree.
It compares nothing. Every correctness check in this repository compares the
device against a HOST TALLY WE WROTE OURSELVES, which is the same reading of
their source expressed twice. If we misread CatBoost, the tally encodes the
misreading and agrees with the kernel perfectly, and "0 wrong of 3168" means
we agree with us.

That is not a comparison against CatBoost. This file is.

It trains CatBoost at settings the port can actually match, then dumps what
it DECIDED, not what it cost:

  * the quantization borders per feature, so our port can be fed the SAME
    compressed index rather than its own borders. Without this the split
    indices are not comparable and nothing downstream means anything.
  * every tree's splits, as (float_feature_index, border), in depth order.
    An oblivious tree IS this list.
  * every tree's leaf values, in CatBoost's own leaf order.
  * the raw predictions on the training rows.

WHAT MATCHES AND WHAT CANNOT
----------------------------
Settings are pinned to the ones the port implements, and every one of these
is a value the port also uses rather than a convenience:

  loss_function   RMSE            the only objective ported
  grow_policy     SymmetricTree   the only policy ported
  boosting_type   Plain           ordered boosting is not ported
  bootstrap_type  No              OUR DEPARTURE, and the reason it is here.
                                  CatBoost's default is Bayesian at
                                  bagging_temperature 1.0, applied before the
                                  derivatives, so a defaults run would
                                  reweight the target and nothing would line
                                  up. `No` is the setting that makes the two
                                  comparable at all.
  rsm             1.0             no feature subsampling
  has_time        True            keeps CatBoost from permuting rows
  random_seed     0

`model_shrink_rate` and `boost_from_average` are turned off for the same
reason: both are real CatBoost behaviour we do not implement, and leaving
them on would show up as a constant offset in every leaf and hide whatever
else is wrong.

USAGE
  pixi run -e bench python tools/catboost_oracle.py > bench/oracle.json
"""
import json
import sys

import numpy as np
import catboost


def build(rows: int, feats: int, seed: int) -> tuple:
    """A dataset with real structure, not a uniform plant.

    The target is built from three features out of `feats`, so a correct
    learner has something to find and a broken one has somewhere to go
    wrong. Uniform or constant data is what let two earlier checks in this
    repository pass at exactly the parameters that were failing.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(rows, feats)).astype(np.float32)
    y = (
        3.0 * x[:, 0]
        - 2.0 * x[:, 3]
        + 1.5 * x[:, 7] * x[:, 0]
        + 0.1 * rng.normal(size=rows)
    ).astype(np.float32)
    return x, y


def main() -> int:
    rows, feats = 4096, 16
    depth, trees = 4, 12
    learning_rate, l2 = 0.3, 3.0
    border_count = 15

    x, y = build(rows, feats, seed=0)

    # THE QUANTIZATION GRID COMES FROM THE POOL, NOT FROM THE MODEL.
    #
    # The first version of this file read borders out of the saved model's
    # `features_info.float_features[].borders`, and that is a DIFFERENT
    # OBJECT. A saved model carries only the borders its trees actually use,
    # so twelve of sixteen features came back with ZERO borders and feature 0
    # came back with 14 instead of 15. Compared against a real binarizer that
    # reads as a total mismatch on every feature, which is what it did, and
    # the fault was here rather than in the port.
    #
    # `Pool.save_quantization_borders` writes the grid itself, 15 per feature
    # for all 16, which is the thing our `best_split` is supposed to
    # reproduce.
    pool = catboost.Pool(x, y)
    pool.quantize(border_count=border_count)
    import tempfile as _tf
    import os as _os
    _d = _tf.mkdtemp()
    _bpath = _os.path.join(_d, "borders.tsv")
    pool.save_quantization_borders(_bpath)
    grid: dict = {}
    with open(_bpath) as _f:
        for _line in _f:
            _line = _line.strip()
            if not _line:
                continue
            _fi, _bv = _line.split("\t")
            grid.setdefault(int(_fi), []).append(float(_bv))

    model = catboost.CatBoostRegressor(
        iterations=trees,
        depth=depth,
        learning_rate=learning_rate,
        l2_leaf_reg=l2,
        border_count=border_count,
        loss_function="RMSE",
        grow_policy="SymmetricTree",
        boosting_type="Plain",
        bootstrap_type="No",
        rsm=1.0,
        has_time=True,
        random_seed=0,
        model_shrink_rate=0.0,
        boost_from_average=False,
        leaf_estimation_iterations=1,
        logging_level="Silent",
        allow_writing_files=False,
    )
    model.fit(pool)

    # `save_model(format="json")` is the only route to the STRUCTURE. The
    # Python attributes expose predictions and feature importances, neither
    # of which says which feature a level split on.
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.json")
        model.save_model(p, format="json")
        with open(p) as f:
            raw = json.load(f)

    borders = [
        {"feature_index": fi, "borders": sorted(grid.get(fi, []))}
        for fi in range(feats)
    ]

    out_trees = []
    for t in raw["oblivious_trees"]:
        splits = [
            {
                "float_feature_index": s.get("float_feature_index"),
                "border": s.get("border"),
                "split_index": s.get("split_index"),
            }
            for s in t.get("splits", [])
        ]
        out_trees.append(
            {"splits": splits, "leaf_values": t.get("leaf_values", [])}
        )

    preds = model.predict(x)

    # A FLAT TEXT SIDECAR, because the check that reads this is in Mojo and a
    # JSON parser is not something to write in order to run a test. One
    # section per line kind, every number in full precision, so nothing is
    # lost between the oracle and the comparison.
    with open("bench/oracle.txt", "w") as f:
        f.write("# CatBoost %s oracle. Generated by tools/catboost_oracle.py\n"
                % catboost.__version__)
        f.write("config %d %d %d %d %.17g %.17g %d\n" % (
            rows, feats, depth, trees, learning_rate, l2, border_count))
        f.write("train_mse %.17g\n" % float(np.mean((preds - y) ** 2)))
        f.write("baseline_mse %.17g\n" % float(np.mean((y - np.mean(y)) ** 2)))
        for b in borders:
            f.write("borders %d %d %s\n" % (
                b["feature_index"], len(b["borders"]),
                " ".join("%.17g" % v for v in b["borders"])))
        for ti, t in enumerate(out_trees):
            for s in t["splits"]:
                f.write("split %d %d %.17g\n" % (
                    ti, s["float_feature_index"], s["border"]))
            f.write("leaves %d %s\n" % (
                ti, " ".join("%.17g" % v for v in t["leaf_values"])))
        for r in range(rows):
            f.write("x %s\n" % " ".join("%.9g" % v for v in x[r]))
        f.write("y %s\n" % " ".join("%.9g" % v for v in y))
        f.write("pred %s\n" % " ".join("%.9g" % v for v in preds))

    json.dump(
        {
            "catboost_version": catboost.__version__,
            "config": {
                "rows": rows,
                "feats": feats,
                "depth": depth,
                "trees": trees,
                "learning_rate": learning_rate,
                "l2_leaf_reg": l2,
                "border_count": border_count,
                "seed": 0,
            },
            "scale_and_bias": raw.get("scale_and_bias"),
            "float_feature_borders": borders,
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
