"""CATBOOST'S OWN OUTPUT FOR THE NINE NEWLY PORTED LOSSES.

    pixi run -e bench loss-oracle-gen        # writes bench/oracle_losses.txt

WHY THIS FILE EXISTS
--------------------
`tools/catboost_oracle.py` dumps CatBoost's own decisions for ONE objective,
RMSE. Nine more objectives were ported -- Quantile, MAE, LogLinQuantile,
MAPE, Poisson, Lq, Expectile, Tweedie, Huber -- and every gate on them
compares our device against a host tally written in this repository or
against libm. Both are worth having and neither is CatBoost. If we misread
their kernel, a libm oracle confirms we implemented our misreading
correctly.

This runs THEIR learner and dumps what it DECIDED, per loss:

  * the quantization grid, so a border disagreement is separable from a
    loss disagreement rather than being absorbed by it,
  * every tree's splits as (float_feature_index, border), in depth order,
  * every tree's leaf values in their own leaf order,
  * the raw prediction on every training row,
  * their own metric value for the objective, from `catboost.utils.eval_metric`,
    which gates OUR host implementation of the loss as well as our fit.

WHY PREDICTIONS AND NOT DERIVATIVES
-----------------------------------
Per-row derivatives are the sharper comparison and CatBoost does not expose
them. `TPointwiseTargetsImpl` is C++ with no Python binding, the custom
objective hook computes the CALLER'S derivatives rather than reporting
theirs, and no `predict`/`eval_metric`/`staged_predict` entry point returns
a gradient. What IS reachable is the tree the derivatives produced, so the
gate is one step downstream: TREE 0's splits and leaf values are a direct
algebraic function of the per-row derivatives at a zero cursor, with no
boosting drift in front of them. A derivative defect that survives tree 0's
sixteen leaf cells has to be conservative under both an argmax and a
weighted sum.

SAME EVERYTHING EXCEPT THE DEVICE
---------------------------------
Every option below is pinned on BOTH arms, and the two that are not simply
"their default" are here for a reason:

  bootstrap_type   No      their default is Bayesian/MVS depending on the
                           loss, applied BEFORE the derivatives; a defaults
                           run would reweight the target and nothing would
                           line up.
  random_strength  0       their last source of randomness in this config
                           (`greedy_search_helper.cpp:385`); absolute noise
                           on candidate scores.
  has_time         True    keeps them from permuting rows.
  boost_from_average / model_shrink_rate  off, both real CatBoost behaviour
                           this port does not implement; leaving them on
                           would show up as a constant offset in every leaf.

  leaf_estimation_method / leaf_estimation_iterations
                           SET EXPLICITLY ON BOTH ARMS rather than left to
                           either side's default, because
                           `GetEstimationMethodDefaults`
                           (`catboost_options.cpp:31-244`) is keyed on
                           TASK TYPE and Tweedie's entry differs: CPU takes
                           1 Newton iteration, GPU takes 20 (`:221-231`).
                           Our port takes the GPU number. Comparing a
                           20-iteration fit against a 1-iteration fit would
                           be a configuration difference reported as a loss
                           defect. The values written here are their CPU
                           defaults, computed by hand from their switch and
                           recorded in the fixture so the Mojo side passes
                           the same two numbers.

THE FIXTURE IS CONSTRUCTED AND HASHED, NEVER A REAL DATASET
-----------------------------------------------------------
`STANDING_ORDERS.md` rule 4. Feature values come out of splitmix64 so every
cell differs from its neighbours and a permutation defect cannot hide behind
a matching total; one column is deliberately low-cardinality (ties on the
border grid) and one is pure noise the target never sees. The signed target
carries planted outliers, because Huber's delta branch and Quantile's
robustness are invisible on clean data.

FLOAT PERSISTENCE
-----------------
Every float is written as `<decimal>/<hex bits>` and read back from the HEX,
because `String(Float32)` on this toolchain returns a one-ULP-wrong value for
0.46% of float32 values. Same convention as `gbdt/models/model_text.mojo`'s
`f32_token` / `parse_f32`, which is what the reader uses.
"""
import os
import struct
import sys

import numpy as np
import catboost
from catboost.utils import eval_metric


ROWS = 3000
FEATS = 8
DEPTH = 4
TREES = 12
LEARNING_RATE = 0.3
L2 = 3.0
BORDER_COUNT = 32
SEED = 0

MASK = (1 << 64) - 1


def splitmix(x: int) -> int:
    z = (x + 0x9E3779B97F4A7C15) & MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return z ^ (z >> 31)


def frac(i: int, salt: int) -> float:
    """Uniform in [0, 1), from the same splitmix the Mojo checks use."""
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
    """Hashed features, two targets, planted structure and planted outliers.

    Column 5 is quantized to seven distinct values on purpose: with
    `border_count` 32 it cannot fill its grid, so its fold count is decided
    by the DATA rather than by the parameter, and a border-selection
    disagreement lands there first. Column 6 never enters either target --
    a feature a correct learner should mostly ignore and a broken score
    function will happily split on.
    """
    x = np.empty((ROWS, FEATS), dtype=np.float32)
    for f in range(FEATS):
        for r in range(ROWS):
            v = (frac(r * FEATS + f, 5) - 0.5) * 2.0
            if f == 5:
                v = float(np.floor(v * 3.5)) / 3.5
            x[r, f] = np.float32(v)

    x0 = x[:, 0].astype(np.float64)
    x3 = x[:, 3].astype(np.float64)
    x7 = x[:, 7].astype(np.float64)

    noise = np.array([frac(r, 77) - 0.5 for r in range(ROWS)])
    y = 2.0 * x0 - 1.5 * x3 + 1.0 * x0 * x7 + 0.25 * noise
    # Planted outliers: 5 of them, scattered, alternating sign. Huber's
    # delta branch and Quantile's insensitivity are both invisible without
    # a tail, and a mean-shaped defect is invisible without an asymmetry.
    for k, r in enumerate((17, 613, 1229, 1847, 2459)):
        y[r] += 6.0 if (k % 2 == 0) else -7.5
    y = y.astype(np.float32)

    # Strictly positive target for the losses whose domain requires it:
    # MAPE divides by |y|, Poisson and Tweedie model a non-negative mean,
    # LogLinQuantile compares against exp(approx).
    scale = np.array([0.5 + frac(r, 99) for r in range(ROWS)])
    ypos = np.exp(0.8 * x0 - 0.6 * x3 + 0.3 * x0 * x7) * scale
    ypos = ypos.astype(np.float32)
    return x, y, ypos


# name -> (catboost loss string, target, leaf_estimation_method,
#          leaf_estimation_iterations, {param: value}, metric string)
#
# METHOD AND ITERATIONS ARE THEIR CPU DEFAULTS, transcribed from
# `GetEstimationMethodDefaults` (`catboost_options.cpp:31-244`) and the
# `useExact` block (`:289-300`), which upgrades MAE / MAPE / Quantile to
# Exact on CPU whenever `ApproxOnFullHistory` is off and there are no
# monotone constraints -- both true here. LogLinQuantile is NOT on that
# list and stays Gradient.
ARMS = [
    ("Quantile",   "Quantile",       "signed",   "Exact",    1, {"alpha": 0.5}),
    ("MAE",        "MAE",            "signed",   "Exact",    1, {}),
    ("LogLinQuantile", "LogLinQuantile", "positive", "Gradient", 1, {"alpha": 0.6}),
    ("MAPE",       "MAPE",           "positive", "Exact",    1, {}),
    ("Poisson",    "Poisson",        "positive", "Newton",  10, {}),
    # Lq TWICE, because its estimator default is keyed on the PARAMETER and
    # not on the loss (`catboost_options.cpp:82-93`): q < 2 is Gradient,
    # q >= 2 is Newton. One arm would leave one branch unchecked.
    ("Lq_1.5",     "Lq",             "signed",   "Gradient", 1, {"q": 1.5}),
    ("Lq_2.5",     "Lq",             "signed",   "Newton",   1, {"q": 2.5}),
    ("Expectile",  "Expectile",      "signed",   "Newton",   5, {"alpha": 0.3}),
    ("Tweedie",    "Tweedie",        "positive", "Newton",   1, {"variance_power": 1.5}),
    ("Huber",      "Huber",          "signed",   "Newton",   1, {"delta": 1.0}),
]


def loss_string(loss: str, params: dict) -> str:
    if not params:
        return loss
    return loss + ":" + ";".join(
        "%s=%s" % (k, repr(v)) for k, v in sorted(params.items())
    )


def main() -> int:
    out_path = os.environ.get("LOSS_ORACLE_OUT", "bench/oracle_losses.txt")
    x, y, ypos = build()
    targets = {"signed": y, "positive": ypos}

    # THE GRID COMES FROM THE POOL, not from a fitted model: a saved model
    # carries only the borders its trees used, which is a different object
    # and reads as a total mismatch against a real binarizer. Same trap
    # `tools/catboost_oracle.py` documents.
    grid_pool = catboost.Pool(x, y)
    grid_pool.quantize(
        border_count=BORDER_COUNT, feature_border_type="GreedyLogSum"
    )
    import tempfile
    d = tempfile.mkdtemp()
    bpath = os.path.join(d, "borders.tsv")
    grid_pool.save_quantization_borders(bpath)
    grid = {}
    with open(bpath) as f:
        for line in f:
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
        "# CatBoost %s loss oracle. tools/catboost_loss_oracle.py"
        % catboost.__version__
    )
    lines.append(
        "# floats are <decimal>/<hex bits>; the hex is authoritative"
    )
    lines.append("version %s" % catboost.__version__)
    lines.append(
        "dims %d %d %d %d %d" % (ROWS, FEATS, DEPTH, TREES, BORDER_COUNT)
    )
    lines.append("hyper %s %s" % (f32_token(LEARNING_RATE), f32_token(L2)))
    for fi in range(FEATS):
        lines.append(
            "borders %d %d %s"
            % (fi, len(grid[fi]), " ".join(f32_token(v) for v in grid[fi]))
        )
    for fi in range(FEATS):
        lines.append(
            "xcol %d %s" % (fi, " ".join(f32_token(v) for v in x[:, fi]))
        )
    for name, arr in targets.items():
        lines.append(
            "target %s %s" % (name, " ".join(f32_token(v) for v in arr))
        )

    for arm, loss, tname, method, iters, params in ARMS:
        t = targets[tname]
        ls = loss_string(loss, params)
        sys.stderr.write("fitting %s (%s) ...\n" % (arm, ls))
        pool = catboost.Pool(x, t)
        pool.quantize(
            border_count=BORDER_COUNT, feature_border_type="GreedyLogSum"
        )
        model = catboost.CatBoostRegressor(
            iterations=TREES,
            depth=DEPTH,
            learning_rate=LEARNING_RATE,
            l2_leaf_reg=L2,
            border_count=BORDER_COUNT,
            loss_function=ls,
            grow_policy="SymmetricTree",
            boosting_type="Plain",
            bootstrap_type="No",
            rsm=1.0,
            has_time=True,
            random_seed=SEED,
            random_strength=0.0,
            model_shrink_rate=0.0,
            boost_from_average=False,
            leaf_estimation_method=method,
            leaf_estimation_iterations=iters,
            score_function="Cosine",
            thread_count=1,
            logging_level="Silent",
            allow_writing_files=False,
        )
        model.fit(pool)

        import json
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "m.json")
            model.save_model(p, format="json")
            with open(p) as fh:
                raw = json.load(fh)

        sb = raw.get("scale_and_bias", [1.0, [0.0]])
        scale = float(sb[0])
        bias = sb[1]
        bias = float(bias[0]) if isinstance(bias, list) else float(bias)

        preds = model.predict(x, prediction_type="RawFormulaVal")

        try:
            metric = float(
                eval_metric(
                    np.asarray(t, dtype=np.float64),
                    np.asarray(preds, dtype=np.float64),
                    ls,
                )[0]
            )
            have_metric = 1
        except Exception as e:  # noqa: BLE001
            sys.stderr.write("  eval_metric unavailable for %s: %s\n" % (ls, e))
            metric = 0.0
            have_metric = 0

        pstr = " ".join(
            "%s %s" % (k, f32_token(v)) for k, v in sorted(params.items())
        )
        lines.append(
            "arm %s %s %s %s %d %d %s"
            % (arm, loss, tname, method, iters, len(params), pstr)
        )
        lines.append(
            "armscalebias %s %s %s" % (arm, f64_token(scale), f64_token(bias))
        )
        lines.append(
            "armmetric %s %d %s" % (arm, have_metric, f64_token(metric))
        )
        trees = raw["oblivious_trees"]
        if len(trees) != TREES:
            sys.stderr.write(
                "  WARNING: %s produced %d trees, asked for %d\n"
                % (arm, len(trees), TREES)
            )
        for ti, tree in enumerate(trees):
            for lvl, s in enumerate(tree.get("splits", [])):
                lines.append(
                    "armsplit %s %d %d %d %s"
                    % (
                        arm,
                        ti,
                        lvl,
                        int(s["float_feature_index"]),
                        f32_token(s["border"]),
                    )
                )
            lines.append(
                "armleaves %s %d %s"
                % (
                    arm,
                    ti,
                    " ".join(f64_token(v) for v in tree.get("leaf_values", [])),
                )
            )
        lines.append(
            "armpred %s %s" % (arm, " ".join(f64_token(v) for v in preds))
        )

    with open(out_path, "w") as fh:
        fh.write("\n".join(lines))
        fh.write("\n")
    sys.stderr.write(
        "wrote %s (%d lines, %d arms)\n" % (out_path, len(lines), len(ARMS))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
