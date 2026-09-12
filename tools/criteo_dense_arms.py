#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE CTR PATH'S FIRST NUMBERS, on a criteo slice whose codes are dense.

    MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=python python3 \
      tools/criteo_dense_arms.py --policy SymmetricTree

WHY THIS FILE HAD TO EXIST (measured on an H100, pod 0zl2hrxqq26b0t,
2026-09-12, and it is a finding about the dataset, not about this script)
---------------------------------------------------------------------------
`bench/speed/forest_speed_arm.py --dataset criteo` CANNOT FIT OUR ARM AT ALL.
`_decode_criteo` assigns each category the rank of its string in sorted-unique
order over EVERY DECODED ROW, and `load_criteo` then hands `fit` a row SLICE:
the last 500,000 rows are the test block and `--rows` cuts the train block
further. A category that appears only in the rows the slice leaves out is a
HOLE, so the codes reaching `fit` are not dense, and `gbdt_fit` refuses:

    cat_features column 13 is not densely coded: category 1 is absent
    from 0..621909

That refusal is correct -- our surface documents DENSE CODES 0..k-1 -- and it
is not a `--rows` artifact: it holds for every train/test split of criteo as
decoded, including the uncapped one, because the test tail always withholds
some categories from train. The fix belongs in `_decode_criteo` (rank within
the split that will be fitted, and give test's unseen values an explicit
bucket); until then the CTR path has no numbers, which is what this file
gets.

REBASED ONTO CURRENT MAIN, 2026-09-12 (lane harness-honesty). BOTH of the
findings below have since landed in the harness itself, so this file no longer
carries either fix and MUST NOT re-apply them:

  * the per-slice re-ranking is now `_criteo_densify_slice` inside
    `load_criteo`, so `check_dense` here only VERIFIES it. Applying the old
    in-file version on top would be a second densification with different
    unknown-bucket semantics, and would map the loader's explicit unknown
    bucket onto real category 0. See `check_dense`.
  * the XGBoost frame fix is now `xgboost_arms._frame` in
    `tools/speed_gbdt_arm.py`, which pins one `CategoricalDtype(range(k))`
    across fit and predict. `xgb_frames` below does the same thing for this
    file's own arms and is kept so this driver stands alone.

WHAT THIS SCRIPT IS, AND WHAT IT REFUSES TO CHANGE
---------------------------------------------------
It checks that every categorical column is dense 0..k-1 in the rows that
reach `fit`, reports how many test rows land in the unknown bucket, and
changes no code. EVERY ARM RECEIVES THE SAME TWO MATRICES, so the unknown
bucket cannot flatter one of them. Nothing else is touched: same
hyper-parameters as `lane_config`, same 1,000,000 rows, same held-out
500,000, warm-up plus three timed rounds ALTERNATING between arms.

WHY IT STILL EXISTS NOW THAT THE HARNESS FITS criteo. `forest_speed_arm.py`
gives ours against the opponents; this gives the `ours` / `ours-nocat` pair in
ONE process, which is the only thing that prices the categorical path itself,
and it is the driver the criteo rows in `bench/OPPONENT_REFERENCE.md` were
taken with.

THE ARMS
---------
  ours          cat_features declared -- the CTR path, the point of the file
  ours-nocat    the same build with the indices WITHHELD, which is DEVIATION
                2634's skip side; the pair is the price of the categorical
                path (NOT the cost of 2634 alone, which would need
                `-D MOJOLEARN_2634_CTR_PREP_OFF=1` beside it)
  catboost-gpu  their CTR machinery, the only symmetric opponent (1831)
  xgboost-gpu   partition search, `max_cat_to_onehot=1`, depthwise/lossguide
                only
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
    a = np.asarray(p, dtype=np.float64)
    if a.ndim == 2 and a.shape[1] == 2:
        return a[:, 1]
    return a.ravel()


def check_dense(d):
    """VERIFY the loader's per-slice codes. Do not re-rank them.

    THIS FUNCTION USED TO DO THE RE-RANKING ITSELF (changed 2026-09-12, lane
    harness-honesty, when this file was rebased onto current main). At the
    time it was written `load_criteo` returned the decode's GLOBAL ranking,
    which is dense over the whole matrix and dense over no slice of it, so
    `gbdt/train.mojo:1231` refused every fit and this script re-ranked within
    the train slice to get any numbers at all.

    `_criteo_densify_slice` now does that inside the loader, and running the
    old code on top of it would be a SECOND densification with DIFFERENT
    semantics, which is worse than redundant. The loader maps a test value
    unseen in train to code `k`, one past the train maximum: a single explicit
    unknown bucket per column. A second pass re-ranks against the train
    values, does not find `k` among them, and maps it to code 0 -- silently
    turning "a category the model never saw" into "category 0, which the model
    learned a leaf value for". That is the exact substitution the loader's
    docstring warns against, arrived at by running a correct fix twice.

    So this now checks the invariant and reports the cost, and changes no
    code. Returns (X_train, X_test, cards) with the matrices as loaded."""
    xtr = np.ascontiguousarray(d.X_train, dtype=np.float32)
    xte = np.ascontiguousarray(d.X_test, dtype=np.float32)
    cards, unseen = [], []
    for j in d.cat_idx:
        present = np.unique(xtr[:, j])
        k = int(present.max()) + 1
        if present.size != k:
            raise SystemExit(
                "criteo column %d is NOT densely coded in the train slice "
                "(%d distinct codes spanning 0..%d). The loader's "
                "_criteo_densify_slice is what guarantees this; do not "
                "re-rank here, fix it there." % (j, present.size, k - 1))
        cards.append(k)
        # The loader's unknown bucket is code k itself, so counting it is a
        # lookup rather than a second searchsorted.
        unseen.append(int((xte[:, j] >= k).sum()))
    say("DENSE-CARD ncat=%d min=%d max=%d sum=%d"
        % (len(cards), min(cards), max(cards), sum(cards)))
    say("DENSE-UNSEEN test_rows=%d unknown_bucketed_total=%d per_col_max=%d"
        % (xte.shape[0], sum(unseen), max(unseen)))
    say("DENSE-CHECK every train column is dense 0..k-1 as the loader left it")
    return xtr, xte, cards


def cat_frame(x, cat_idx):
    out = x.astype(object)
    for j in cat_idx:
        out[:, j] = x[:, j].astype(np.int64)
    return out


def xgb_frames(xtr, xte, cat_idx, cards):
    """ONE CategoricalDtype per column, shared by fit and predict."""
    import pandas as pd
    from pandas.api.types import CategoricalDtype

    def build(x):
        df = pd.DataFrame(x)
        for k, j in enumerate(cat_idx):
            dtype = CategoricalDtype(categories=list(range(cards[k])))
            df[j] = df[j].astype("int64").astype(dtype)
        return df

    return build(xtr), build(xte)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="criteo_dense_arms")
    ap.add_argument("--policy", default="SymmetricTree",
                    choices=("SymmetricTree", "Depthwise", "Lossguide"))
    ap.add_argument("--rows", type=int, default=1000000)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--arms", default=None,
                    help="comma list; default is every arm the policy allows")
    args = ap.parse_args(argv)

    d = spec.load_criteo("shipped", args.rows)
    if d.name != "criteo":
        raise SystemExit("load_criteo fell back to %r" % d.name)
    say("DENSE-DATA policy=%s rows=%d feats=%d test=%d positives=%.4f"
        % (args.policy, d.X_train.shape[0], d.X_train.shape[1],
           d.X_test.shape[0], float(d.y_train.mean())))
    xtr, xte, cards = check_dense(d)
    y = np.ascontiguousarray(d.y_train, dtype=np.float32)
    yte = d.y_test
    cat = list(d.cat_idx)

    n_estimators, max_depth, borders, l2, lr, seed = 100, 6, 254, 1.0, 0.1, 7
    arms = []

    # ---- ours, and ours with the indices withheld ----
    def our_arm(name, cats):
        import mojolearn

        def make():
            p = dict(n_estimators=n_estimators, max_depth=max_depth,
                     learning_rate=lr, l2_leaf_reg=l2, border_count=borders,
                     random_state=seed, bootstrap_type="No",
                     grow_policy=args.policy, loss="Logloss")
            if args.policy == "Lossguide":
                p["max_leaves"] = 31
            if cats:
                p["cat_features"] = cat
            return mojolearn.GradientBoosting(**p)

        return (name, make,
                lambda m: m.fit(xtr, y),
                lambda m: positive_column(m.predict_proba(xte)))

    arms.append(our_arm("ours", True))
    arms.append(our_arm("ours-nocat", False))

    # ---- catboost ----
    def catboost_arm():
        import catboost
        xf = cat_frame(xtr, cat)
        xtf = cat_frame(xte, cat)

        def make():
            p = dict(iterations=n_estimators, depth=max_depth,
                     learning_rate=lr, l2_leaf_reg=l2, border_count=borders,
                     random_seed=seed, bootstrap_type="No",
                     boosting_type="Plain", grow_policy=args.policy,
                     task_type="GPU", devices="0", verbose=False,
                     allow_writing_files=False, cat_features=cat)
            if args.policy == "Lossguide":
                p["max_leaves"] = 31
            return catboost.CatBoostClassifier(loss_function="Logloss", **p)

        return ("catboost-gpu", make,
                lambda m: m.fit(xf, y),
                lambda m: positive_column(m.predict_proba(xtf)))

    arms.append(catboost_arm())

    # ---- xgboost (no symmetric grower: DEVIATION 1831) ----
    if args.policy != "SymmetricTree":
        def xgboost_arm():
            import xgboost as xgb
            dtr, dte = xgb_frames(xtr, xte, cat, cards)

            def make():
                p = dict(n_estimators=n_estimators, max_depth=max_depth,
                         learning_rate=lr, reg_lambda=l2, reg_alpha=0.0,
                         max_bin=borders + 1, subsample=1.0,
                         colsample_bytree=1.0, colsample_bylevel=1.0,
                         colsample_bynode=1.0, min_child_weight=1.0,
                         tree_method="hist", random_state=seed,
                         device="cuda", verbosity=0,
                         enable_categorical=True, max_cat_to_onehot=1,
                         grow_policy={"Depthwise": "depthwise",
                                      "Lossguide": "lossguide"}[args.policy])
                if args.policy == "Lossguide":
                    p["max_leaves"] = 31
                return xgb.XGBClassifier(objective="binary:logistic", **p)

            return ("xgboost-gpu", make,
                    lambda m: m.fit(dtr, y),
                    lambda m: positive_column(m.predict_proba(dte)))

        arms.append(xgboost_arm())

    if args.arms:
        want = [a.strip() for a in args.arms.split(",") if a.strip()]
        arms = [a for a in arms if a[0] in want]

    # ---- warm-up, then alternating timed rounds ----
    times = {a[0]: [] for a in arms}
    last = {}
    live = []
    for name, make, fit, _score in arms:
        try:
            t0 = time.perf_counter()
            m = make()
            fit(m)
            ms = (time.perf_counter() - t0) * 1000.0
            say("DENSE-WARMUP arm=%s policy=%s ms=%.1f" % (name, args.policy, ms))
            last[name] = m
            live.append((name, make, fit, _score))
        except Exception as exc:                       # noqa: BLE001
            say("DENSE-REFUSED arm=%s policy=%s during warm-up: %s %s"
                % (name, args.policy, type(exc).__name__,
                   " ".join(str(exc).split())[:220]))

    for r in range(1, args.rounds + 1):
        for name, make, fit, _score in list(live):
            try:
                t0 = time.perf_counter()
                m = make()
                fit(m)
                ms = (time.perf_counter() - t0) * 1000.0
            except Exception as exc:                   # noqa: BLE001
                say("DENSE-REFUSED arm=%s policy=%s at round %d: %s %s"
                    % (name, args.policy, r, type(exc).__name__,
                       " ".join(str(exc).split())[:220]))
                live = [a for a in live if a[0] != name]
                continue
            times[name].append(ms)
            last[name] = m
            say("DENSE arm=%s policy=%s round=%d ms=%.1f"
                % (name, args.policy, r, ms))

    # ---- medians and quality, after every timer ----
    med = {}
    for name in sorted(times):
        if times[name]:
            med[name] = float(np.median(times[name]))
            say("DENSE-MEDIAN arm=%s policy=%s median_ms=%.1f rounds=%d"
                % (name, args.policy, med[name], len(times[name])))
    for name, _make, _fit, score in arms:
        m = last.get(name)
        if m is None:
            continue
        try:
            s = score(m)
            say("DENSE-ACC arm=%s policy=%s auc=%.6f logloss=%.6f hash=%s"
                % (name, args.policy, spec.auc(yte, s), logloss(yte, s),
                   digest(s)))
        except Exception as exc:                       # noqa: BLE001
            say("DENSE-REFUSED arm=%s policy=%s while scoring: %s %s"
                % (name, args.policy, type(exc).__name__,
                   " ".join(str(exc).split())[:220]))
    for opp in ("catboost-gpu", "xgboost-gpu"):
        if "ours" in med and opp in med:
            say("DENSE-RATIO policy=%s ours_over_%s=%.4f"
                % (args.policy, opp, med["ours"] / med[opp]))
    if "ours" in med and "ours-nocat" in med:
        say("DENSE-RATIO policy=%s ours_over_ours_nocat=%.4f "
            "(the price of the categorical path, not of 2634 alone)"
            % (args.policy, med["ours"] / med["ours-nocat"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
