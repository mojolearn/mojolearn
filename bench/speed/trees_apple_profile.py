#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The trees family's Apple (Metal) speed ledger: one estimator, one dataset,
one process, a warm-up fit and `--rounds` timed fits, the prediction digest of
every fit (hashed OUTSIDE the clock) and the quality of the last one.

    pixi run -e default python bench/speed/trees_apple_profile.py \\
        --est adaboost-clf --dataset taxi --rows 1000000 --rounds 2

Estimators are the trees lane's public classes with the board settings of
tools/bench_board_algos.py (lane trees) and tools/speed_gbdt_arm.py
(`lane_config`), so a number here is the same fit the board times. The
GradientBoosting and RF/ET cells have their own harnesses
(tools/gbdt_train_probe.py, tools/forest_train_ab.py); this file is for the
rest of the family: DecisionTree, Bagging, AdaBoost, DART,
RandomTreesEmbedding, IsolationForest.

Every line begins `TAP ` so one steward stdout parses on the laptop. A before
and an after are two commits timed on the SAME Mac; their `digest` must be
equal under IDENTICAL.
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (os.path.join(_ROOT, "tools"), os.path.join(_ROOT, "python")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SEED = 7


def _make(name, task):
    import mojolearn as m
    clf = task != "regression"
    if name == "dt":
        cls = m.DecisionTreeClassifier if clf else m.DecisionTreeRegressor
        return cls(max_depth=16, random_state=SEED)
    if name == "bagging":
        if clf:
            return m.BaggingClassifier(m.DecisionTreeClassifier(max_depth=12), n_estimators=10,
                                       random_state=SEED)
        return m.BaggingRegressor(m.DecisionTreeRegressor(max_depth=12), n_estimators=10,
                                  random_state=SEED)
    if name == "adaboost":
        if clf:
            return m.AdaBoostClassifier(m.DecisionTreeClassifier(max_depth=3), n_estimators=50,
                                        random_state=SEED)
        return m.AdaBoostRegressor(m.DecisionTreeRegressor(max_depth=3), n_estimators=50,
                                   random_state=SEED)
    if name == "dart":
        cls = m.DARTClassifier if clf else m.DARTRegressor
        return cls(n_estimators=100, random_state=SEED)
    if name == "embedding":
        return m.RandomTreesEmbedding(n_estimators=100, max_depth=5, random_state=SEED)
    if name == "iforest":
        return m.IsolationForest(n_estimators=100, random_state=SEED)
    raise SystemExit("unknown --est %s" % name)


def _outputs(model, name, xt):
    if name == "embedding":
        out = model.transform(xt)
        out = out.toarray() if hasattr(out, "toarray") else np.asarray(out)
        return [np.asarray(out)]
    if name == "iforest":
        return [np.asarray(model.score_samples(xt)), np.asarray(model.predict(xt))]
    outs = [np.asarray(model.predict(xt))]
    if hasattr(model, "predict_proba"):
        try:
            outs.append(np.asarray(model.predict_proba(xt)))
        except Exception:                           # noqa: BLE001
            pass
    return outs


def _digest(arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:16]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--est", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--rows", type=int, default=1000000)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--warm-rows", type=int, default=20000)
    p.add_argument("--score-rows", type=int, default=200000)
    p.add_argument("--label", default="")
    a = p.parse_args(argv)
    import speed_gbdt_arm as spec
    import mojolearn
    data = spec.load_dataset(a.dataset, "shipped", a.rows)
    x = np.ascontiguousarray(data.X_train, dtype=np.float32)
    y = np.ascontiguousarray(data.y_train, dtype=np.float32)
    if data.task != "regression":
        y = y.astype(np.int64)
    xt = np.ascontiguousarray(data.X_test[:a.score_rows], dtype=np.float32)
    yt = np.asarray(data.y_test[:a.score_rows])
    rec = dict(est=a.est, dataset=a.dataset, rows=int(x.shape[0]), cols=int(x.shape[1]),
               task=data.task, label=a.label,
               mode=os.environ.get("MOJOLEARN_NUMERIC_MODE", "unset"), vendor=mojolearn.vendor())
    warm = _make(a.est, data.task)
    t0 = time.perf_counter()
    if a.est == "iforest" or a.est == "embedding":
        warm.fit(x[:a.warm_rows])
    else:
        warm.fit(x[:a.warm_rows], y[:a.warm_rows])
    rec["warm_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
    del warm
    ms, digests, model = [], [], None
    for _ in range(a.rounds):
        model = _make(a.est, data.task)
        t0 = time.perf_counter()
        if a.est == "iforest" or a.est == "embedding":
            model.fit(x)
        else:
            model.fit(x, y)
        ms.append(round((time.perf_counter() - t0) * 1000.0, 1))
        t1 = time.perf_counter()
        outs = _outputs(model, a.est, xt)
        rec.setdefault("infer_ms", []).append(round((time.perf_counter() - t1) * 1000.0, 1))
        digests.append(_digest(outs))
    rec["ms"], rec["digests"] = ms, digests
    pred = _outputs(model, a.est, xt)[0]
    if a.est in ("iforest", "embedding"):
        rec["quality"] = None
    elif data.task == "regression":
        rec["quality"] = dict(rmse=float(np.sqrt(np.mean(
            (pred.astype(np.float64) - yt.astype(np.float64)) ** 2))))
    else:
        rec["quality"] = dict(accuracy=float(np.mean(pred.astype(np.int64) == yt.astype(np.int64))))
    print("TAP " + json.dumps(rec), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
