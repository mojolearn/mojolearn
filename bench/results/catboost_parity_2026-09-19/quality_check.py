#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Behavior check, mojolearn against CatBoost 1.2.10 CPU at identical pinned
settings (lane/catboost-parity, 2026-09-19): Plain vs Ordered boosting and
each feature_border_type. NOT a bitwise comparison and NOT a speed
comparison: two libraries, two random streams, and CatBoost's GPU learner
(the one mojolearn ports) does not run on this Mac, so their CPU learner --
whose Ordered boosting is a different implementation from their GPU one -- is
the reference arm. HIGGS is retired as a benchmark dataset for claims
(ENGINEERING_RULES section 9); it is used here only because the task named
it, to see whether the two implementations behave alike.

    python3 quality_check.py ours    --out ours.json      # pixi env with mojolearn
    python3 quality_check.py catboost --out catboost.json # env with catboost==1.2.10
    python3 quality_check.py table ours.json catboost.json

Train: the first 200,000 rows of higgs_speed.npz; test: its last 500,000.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

DATA = Path.home() / "datasets/gbm-bench/higgs/higgs_speed.npz"
N_TRAIN = 200_000
N_TEST = 500_000
BORDER_TYPES = ("GreedyLogSum", "Median", "Uniform", "UniformAndQuantiles",
                "MaxLogSum", "MinEntropy", "GreedyMinEntropy")
#: every setting both arms receive explicitly
COMMON = dict(iterations=300, depth=6, learning_rate=0.1, l2_leaf_reg=3.0,
              border_count=128, random_strength=0.0, bootstrap_type="No",
              leaf_estimation_method="Newton", leaf_estimation_iterations=10,
              boost_from_average=False, score_function="Cosine", nan_mode="Min",
              random_seed=0)
CASES = ([("Plain", "GreedyLogSum"), ("Ordered", "GreedyLogSum")]
         + [("Plain", bt) for bt in BORDER_TYPES[1:]])


def load():
    d = np.load(DATA)
    x, y = d["x"], d["y"]
    return (np.asfortranarray(x[:N_TRAIN].astype(np.float32)), y[:N_TRAIN].astype(np.float32),
            np.ascontiguousarray(x[-N_TEST:].astype(np.float32)), y[-N_TEST:].astype(np.float64))


def auc(score, y):
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), np.float64)
    s = score[order]
    i = 0
    n = len(s)
    while i < n:  # average ranks over ties
        j = i
        while j + 1 < n and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    npos = y.sum()
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * (len(y) - npos)))


def logloss(p, y):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def run_ours():
    import mojolearn as ml
    Xtr, ytr, Xte, yte = load()
    out = {}
    for boosting, bt in CASES:
        c = COMMON
        kw = dict(loss="Logloss", n_estimators=c["iterations"], max_depth=c["depth"],
                  learning_rate=c["learning_rate"], l2_leaf_reg=c["l2_leaf_reg"],
                  border_count=c["border_count"], random_strength=c["random_strength"],
                  bootstrap_type=c["bootstrap_type"], leaf_estimation_method=c["leaf_estimation_method"],
                  leaf_estimation_iterations=c["leaf_estimation_iterations"],
                  boost_from_average=c["boost_from_average"], score_function=c["score_function"],
                  nan_mode=c["nan_mode"], random_state=c["random_seed"],
                  boosting_type=boosting, feature_border_type=bt, numeric_mode="identical")
        if boosting == "Ordered":
            kw.update(permutation_count=4, fold_len_multiplier=2.0)
        t = time.time()
        m = ml.GradientBoosting(**kw).fit(Xtr, ytr)
        p = np.asarray(m.predict_proba(Xte))[:, 1]
        out[f"{boosting}/{bt}"] = dict(auc=auc(p, yte), logloss=logloss(p, yte),
                                       trees=int(m.model_.count("\ntree ")), seconds=round(time.time() - t, 1),
                                       boosting_type_=m.boosting_type_)
        print(boosting, bt, out[f"{boosting}/{bt}"], flush=True)
    return out


def run_catboost():
    import catboost
    Xtr, ytr, Xte, yte = load()
    out = {}
    for boosting, bt in CASES:
        kw = dict(COMMON, loss_function="Logloss", boosting_type=boosting, feature_border_type=bt,
                  thread_count=-1, verbose=False, allow_writing_files=False, task_type="CPU")
        if boosting == "Ordered":
            # their Python API takes permutation_count for task_type='GPU'
            # only; the CPU learner keeps its default of 4
            # (boosting_options.cpp:14), the value mojolearn is given
            kw.update(fold_len_multiplier=2.0)
        t = time.time()
        m = catboost.CatBoostClassifier(**kw).fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        got = m.get_all_params()
        out[f"{boosting}/{bt}"] = dict(auc=auc(p, yte), logloss=logloss(p, yte), trees=int(m.tree_count_),
                                       seconds=round(time.time() - t, 1),
                                       boosting_type_=got["boosting_type"],
                                       border_type_=got["feature_border_type"],
                                       catboost=catboost.__version__)
        print(boosting, bt, out[f"{boosting}/{bt}"], flush=True)
    return out


def table(ours_path, cb_path):
    ours = json.loads(Path(ours_path).read_text())
    cb = json.loads(Path(cb_path).read_text())
    print("| boosting | border type | mojolearn AUC | CatBoost CPU AUC | AUC diff | mojolearn log loss | CatBoost CPU log loss | log loss diff |")
    print("|---|---|---|---|---|---|---|---|")
    for boosting, bt in CASES:
        k = f"{boosting}/{bt}"
        a, b = ours[k], cb[k]
        print(f"| {boosting} | {bt} | {a['auc']:.5f} | {b['auc']:.5f} | {a['auc'] - b['auc']:+.5f} | "
              f"{a['logloss']:.5f} | {b['logloss']:.5f} | {a['logloss'] - b['logloss']:+.5f} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=("ours", "catboost", "table"))
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.arm == "table":
        table(*a.paths)
        return
    res = run_ours() if a.arm == "ours" else run_catboost()
    meta = dict(settings=COMMON, n_train=N_TRAIN, n_test=N_TEST, data=str(DATA), arm=a.arm)
    if a.out:
        a.out.write_text(json.dumps(dict(meta=meta, **res), indent=1))


if __name__ == "__main__":
    main()
