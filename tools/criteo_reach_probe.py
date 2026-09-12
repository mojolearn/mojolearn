#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""DID THE ARM ACTUALLY RECEIVE THE CATEGORICAL COLUMNS? Reach, not speed.

    MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_STAGE_TIMES=1 \
      PYTHONPATH=python python3 tools/criteo_reach_probe.py --stage ours

WHY THIS EXISTS SEPARATELY FROM THE TIMING
-------------------------------------------
criteo is the first dataset in this repository with a categorical column, so
`cat_features` reaches code no benchmark had ever run. Two failures are
possible and they are NOT equally visible:

  * a CRASH, which is a finding and reports itself;
  * a SILENT FALLBACK, where the indices are accepted and the library splits
    26 hashed-id columns as ORDERED NUMBERS anyway. That one produces a full
    table of plausible timings against a DIFFERENT, EASIER problem, and no
    line of the output says so.

So each arm is fitted TWICE at a small size, once with the indices declared
and once with them withheld, and the two are compared. A library that used
the declaration must produce a different model: different predictions, and a
different held-out AUC. Equal answers on both sides would mean the
declaration did nothing, whatever the arm accepted.

The witnesses are per library and deliberately redundant:

  ours       the with/without A/B, plus `train_pre_quantize` from
             MOJOLEARN_STAGE_TIMES=1 -- DEVIATION 2634's CTR target prep (the
             MinEntropy target borders, the binarized target and one
             estimation order per permutation) is built inside that stage and
             ONLY when some column is declared categorical, so the stage is
             the switch's own clock. Plus the two refusals our surface owes
             by name.
  catboost   `get_cat_feature_indices()` read back off the fitted model, and
             the "Invalid type for cat_feature" raise on float columns, which
             is what makes the integer frame load-bearing rather than
             decorative.
  xgboost    the booster's own `feature_types` ('c' per declared column), and
             the raise when predict sees dtypes the fit did not.

ONE LIBRARY PER PROCESS, by `--stage`. Our binding reaches CUDA through MAX's
runtime and the opponents through their own; two runtimes contending for one
context is a plausible way to lose a rented box, and this file is not where
that risk is worth taking.

Small by construction: 200,000 rows and 10 iterations. Nothing here is a
timing and none of these numbers belongs in a speed table.
"""

import argparse
import hashlib
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
for _p in (os.path.join(_ROOT, "tools"), os.path.join(_ROOT, "python")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import speed_gbdt_arm as spec           # noqa: E402

ROWS = int(os.environ.get("CRITEO_PROBE_ROWS", "200000"))
ITERS = int(os.environ.get("CRITEO_PROBE_ITERS", "10"))


def say(*a):
    print(*a, flush=True)


def digest(vec):
    arr = np.ascontiguousarray(np.asarray(vec, dtype=np.float64).ravel())
    h = hashlib.sha256()
    h.update(str(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest()[:16]


def logloss(y, p):
    p = np.clip(np.asarray(p, dtype=np.float64).ravel(), 1e-15, 1 - 1e-15)
    y = np.asarray(y, dtype=np.float64).ravel()
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def positive_column(p):
    """The positive-class score, whatever shape the library returned."""
    a = np.asarray(p, dtype=np.float64)
    if a.ndim == 2 and a.shape[1] == 2:
        return a[:, 1]
    return a.ravel()


def load():
    d = spec.load_criteo("shipped", ROWS)
    say("PROBE-DATA rows=%d feats=%d test=%d ncat=%d cat_idx=%d..%d "
        "positives=%.4f"
        % (d.X_train.shape[0], d.X_train.shape[1], d.X_test.shape[0],
           len(d.cat_idx), min(d.cat_idx), max(d.cat_idx),
           float(d.y_train.mean())))
    card = [int(d.X_train[:, j].max()) + 1 for j in d.cat_idx]
    say("PROBE-CARD min=%d max=%d sum=%d per_col=%s"
        % (min(card), max(card), sum(card), card))
    # The codes must be DENSE integers held exactly in float32 or every arm
    # below is being handed something other than a category.
    integral = all(
        float(np.abs(d.X_train[:, j] - np.rint(d.X_train[:, j])).max()) == 0.0
        for j in d.cat_idx)
    say("PROBE-CODES integral=%s" % integral)
    return d


def report(tag, cats, ms, y, score, extra=""):
    say("PROBE-%s cats=%d ms=%.1f auc=%.6f logloss=%.6f hash=%s %s"
        % (tag, 1 if cats else 0, ms, spec.auc(y, score), logloss(y, score),
           digest(score), extra))


def compare(tag, a, b):
    """a and b are (auc, hash) for cats-declared and cats-withheld."""
    same_hash = a[1] == b[1]
    say("PROBE-%s-VERDICT auc_declared=%.6f auc_withheld=%.6f delta=%+.6f "
        "predictions_identical=%s categorical_path_reached=%s"
        % (tag, a[0], b[0], a[0] - b[0], same_hash, not same_hash))
    if same_hash:
        say("PROBE-%s-WARNING the two fits agree BIT FOR BIT, so declaring "
            "the categorical columns changed nothing: treat any timing for "
            "this arm as a numeric-split timing" % tag)


# --------------------------------------------------------------------------
# ours
# --------------------------------------------------------------------------

def stage_ours(d):
    import mojolearn

    x = np.ascontiguousarray(d.X_train, dtype=np.float32)
    y = np.ascontiguousarray(d.y_train, dtype=np.float32)
    xte = np.ascontiguousarray(d.X_test, dtype=np.float32)
    say("PROBE-OURS-BINDING mode=%s vendor=%s"
        % (mojolearn.GradientBoosting().numeric_mode_used(),
           mojolearn.GradientBoosting().vendor_used()))

    out = {}
    for cats in (True, False):
        params = dict(
            n_estimators=ITERS, max_depth=6, learning_rate=0.1,
            l2_leaf_reg=1.0, border_count=254, random_state=7,
            bootstrap_type="No", grow_policy="SymmetricTree", loss="Logloss")
        if cats:
            params["cat_features"] = list(d.cat_idx)
        say("PROBE-OURS-FIT cats=%d begin (stage table follows)" % cats)
        m = mojolearn.GradientBoosting(**params)
        t0 = time.perf_counter()
        m.fit(x, y)
        ms = (time.perf_counter() - t0) * 1000.0
        score = positive_column(m.predict_proba(xte))
        text = m.model_ or ""
        extra = ("model_bytes=%d ctr_tokens=%d"
                 % (len(text), text.lower().count("ctr")))
        report("OURS", cats, ms, d.y_test, score, extra)
        out[cats] = (spec.auc(d.y_test, score), digest(score))
    compare("OURS", out[True], out[False])

    # The refusals this surface owes BY NAME. A refusal that has quietly
    # become an acceptance is how a categorical run turns into a numeric one.
    for label, kw, want in (
        ("feature_fraction_lt_1_with_cats",
         dict(feature_fraction=0.8, cat_features=list(d.cat_idx)),
         NotImplementedError),
        ("permutation_count_without_cats",
         dict(permutation_count=2), ValueError),
        ("ctr_permutation_id_without_cats",
         dict(ctr_estimation_permutation_id=0), ValueError),
    ):
        try:
            mojolearn.GradientBoosting(n_estimators=1, **kw)
        except want as exc:
            say("PROBE-OURS-REFUSAL %s RAISED %s: %s"
                % (label, type(exc).__name__,
                   " ".join(str(exc).split())[:200]))
        except Exception as exc:                   # noqa: BLE001
            say("PROBE-OURS-REFUSAL %s WRONG-EXCEPTION %s: %s"
                % (label, type(exc).__name__,
                   " ".join(str(exc).split())[:200]))
        else:
            say("PROBE-OURS-REFUSAL %s ACCEPTED -- the named refusal is GONE"
                % label)


# --------------------------------------------------------------------------
# catboost
# --------------------------------------------------------------------------

def cat_frame(x, cat_idx):
    out = x.astype(object)
    for j in cat_idx:
        out[:, j] = x[:, j].astype(np.int64)
    return out


def stage_catboost(d):
    import catboost

    x = np.ascontiguousarray(d.X_train, dtype=np.float32)
    y = np.ascontiguousarray(d.y_train, dtype=np.float32)
    xte = np.ascontiguousarray(d.X_test, dtype=np.float32)
    say("PROBE-CATBOOST-VERSION %s" % catboost.__version__)

    def params():
        return dict(
            iterations=ITERS, depth=6, learning_rate=0.1, l2_leaf_reg=1.0,
            border_count=254, random_seed=7, bootstrap_type="No",
            boosting_type="Plain", grow_policy="SymmetricTree",
            task_type="GPU", devices="0", verbose=False,
            allow_writing_files=False)

    out = {}
    for cats in (True, False):
        p = params()
        if cats:
            p["cat_features"] = list(d.cat_idx)
        m = catboost.CatBoostClassifier(loss_function="Logloss", **p)
        xf = cat_frame(x, d.cat_idx) if cats else x
        xtf = cat_frame(xte, d.cat_idx) if cats else xte
        t0 = time.perf_counter()
        m.fit(xf, y)
        ms = (time.perf_counter() - t0) * 1000.0
        score = positive_column(m.predict_proba(xtf))
        declared = list(m.get_cat_feature_indices())
        extra = ("declared_indices=%d matches_dataset=%s"
                 % (len(declared), declared == list(d.cat_idx)))
        report("CATBOOST", cats, ms, d.y_test, score, extra)
        out[cats] = (spec.auc(d.y_test, score), digest(score))
    compare("CATBOOST", out[True], out[False])

    # SABOTAGE: the integer frame is what makes the declaration legal. Hand
    # CatBoost the raw float columns under the same declaration and it must
    # refuse, which is the proof that the frame conversion in the harness is
    # load-bearing and not decoration.
    p = params()
    p["cat_features"] = list(d.cat_idx)
    try:
        catboost.CatBoostClassifier(loss_function="Logloss", **p).fit(
            x[:20000], y[:20000])
    except Exception as exc:                       # noqa: BLE001
        say("PROBE-CATBOOST-SABOTAGE float_cat_columns RAISED %s: %s"
            % (type(exc).__name__, " ".join(str(exc).split())[:200]))
    else:
        say("PROBE-CATBOOST-SABOTAGE float_cat_columns ACCEPTED -- the "
            "integer frame is not what makes this arm categorical")


# --------------------------------------------------------------------------
# xgboost
# --------------------------------------------------------------------------

def xgb_frame(x, cat_idx):
    import pandas as pd
    df = pd.DataFrame(x)
    for j in cat_idx:
        df[j] = df[j].astype("int64").astype("category")
    return df


def stage_xgboost(d):
    import xgboost as xgb

    x = np.ascontiguousarray(d.X_train, dtype=np.float32)
    y = np.ascontiguousarray(d.y_train, dtype=np.float32)
    xte = np.ascontiguousarray(d.X_test, dtype=np.float32)
    say("PROBE-XGBOOST-VERSION %s" % xgb.__version__)

    out = {}
    for cats in (True, False):
        p = dict(
            n_estimators=ITERS, max_depth=6, learning_rate=0.1,
            reg_lambda=1.0, reg_alpha=0.0, max_bin=255, subsample=1.0,
            colsample_bytree=1.0, colsample_bylevel=1.0, colsample_bynode=1.0,
            min_child_weight=1.0, tree_method="hist", grow_policy="depthwise",
            random_state=7, device="cuda", verbosity=0)
        if cats:
            p["enable_categorical"] = True
            p["max_cat_to_onehot"] = 1     # force partition search
        m = xgb.XGBClassifier(objective="binary:logistic", **p)
        xf = xgb_frame(x, d.cat_idx) if cats else x
        xtf = xgb_frame(xte, d.cat_idx) if cats else xte
        t0 = time.perf_counter()
        m.fit(xf, y)
        ms = (time.perf_counter() - t0) * 1000.0
        score = positive_column(m.predict_proba(xtf))
        types = m.get_booster().feature_types or []
        extra = ("feature_types_c=%d of %d"
                 % (sum(1 for t in types if t == "c"), len(types)))
        report("XGBOOST", cats, ms, d.y_test, score, extra)
        out[cats] = (spec.auc(d.y_test, score), digest(score))
    compare("XGBOOST", out[True], out[False])

    # SABOTAGE: a model fitted on category dtype must refuse raw float test
    # rows. If it accepts them, the fit's categorical declaration was not
    # binding on predict and the scored numbers are not the fitted model's.
    p = dict(n_estimators=ITERS, max_depth=6, tree_method="hist",
             device="cuda", verbosity=0, enable_categorical=True,
             max_cat_to_onehot=1)
    m = xgb.XGBClassifier(objective="binary:logistic", **p)
    m.fit(xgb_frame(x[:20000], d.cat_idx), y[:20000])
    try:
        m.predict_proba(xte[:1000])
    except Exception as exc:                       # noqa: BLE001
        say("PROBE-XGBOOST-SABOTAGE raw_float_predict RAISED %s: %s"
            % (type(exc).__name__, " ".join(str(exc).split())[:200]))
    else:
        say("PROBE-XGBOOST-SABOTAGE raw_float_predict ACCEPTED -- predict did "
            "not require the dtypes the fit saw")


STAGES = {"ours": stage_ours, "catboost": stage_catboost,
          "xgboost": stage_xgboost}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="criteo_reach_probe")
    ap.add_argument("--stage", required=True, choices=sorted(STAGES))
    args = ap.parse_args(argv)
    d = load()
    if d.name != "criteo":
        raise SystemExit(
            "load_criteo fell back to %r; there is nothing categorical to "
            "probe and every line below would be about the wrong data"
            % d.name)
    STAGES[args.stage](d)
    say("PROBE-DONE stage=%s" % args.stage)
    return 0


if __name__ == "__main__":
    sys.exit(main())
