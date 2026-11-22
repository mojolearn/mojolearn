#!/usr/bin/env python3
"""Output digests of the tree fits whose host bodies moved to typed core RUN
(lane bm-trees, 2026-11-22): ExtraTrees, DecisionTree(splitter='random'),
RandomTreesEmbedding, DART, RandomForest (plain and through a data session,
Bagging). A testing tool, never product execution.

One line per configuration, `<name> sha256=<digest of every output's bytes>`
or `<name> error <message>`. Run it on two builds of the CPU column (before and
after a change) and diff the files: equal lines mean equal bits.

    PYTHONPATH=<repo>/python python3 tools/bm_trees_digests.py > out.txt

Parallel plan: the configurations are independent fits, one per worker
process (at most the allocated cores, at most the configuration count).
"""
import hashlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor


def _data(kind, n=600, d=8, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    if kind == "reg":
        y = (X[:, 0] * 2 + np.sin(X[:, 1]) + 0.1 * rng.normal(size=n)).astype(np.float32)
    elif kind == "pos":
        y = np.exp(0.5 * X[:, 0]).astype(np.float32)
    else:
        k = int(kind[3:])
        y = (np.digitize(X[:, 0] + 0.5 * X[:, 1], np.linspace(-1, 1, k - 1))).astype(np.int64)
    return X, y


def _cases():
    return {
        "et_clf_gini": ("ExtraTreesClassifier", "clf3", dict(n_estimators=20, random_state=0)),
        "et_clf_entropy_leaves": ("ExtraTreesClassifier", "clf2",
                                  dict(n_estimators=10, criterion="entropy", max_leaf_nodes=12, random_state=1)),
        "et_clf_weighted": ("ExtraTreesClassifier", "clf3", dict(n_estimators=10, random_state=2)),
        "et_reg_mse": ("ExtraTreesRegressor", "reg", dict(n_estimators=20, max_depth=10, random_state=0)),
        "et_reg_poisson": ("ExtraTreesRegressor", "pos", dict(n_estimators=20, criterion="poisson", random_state=0)),
        "et_reg_gamma": ("ExtraTreesRegressor", "pos", dict(n_estimators=10, criterion="gamma", random_state=0)),
        "et_reg_bootstrap": ("ExtraTreesRegressor", "reg", dict(n_estimators=10, bootstrap=True, random_state=3)),
        "dt_random_clf": ("DecisionTreeClassifier", "clf3", dict(splitter="random", max_depth=10, random_state=0)),
        "dt_random_reg": ("DecisionTreeRegressor", "reg", dict(splitter="random", max_leaf_nodes=20, random_state=0)),
        "rte": ("RandomTreesEmbedding", "reg", dict(n_estimators=15, max_depth=5, random_state=0)),
        "dart_reg": ("DARTRegressor", "reg", dict(n_estimators=30, learning_rate=0.1, random_state=0)),
        "dart_clf2": ("DARTClassifier", "clf2", dict(n_estimators=30, random_state=0)),
        "dart_clf3_opts": ("DARTClassifier", "clf3", dict(n_estimators=30, reg_alpha=0.5, max_delta_step=0.7,
                                                          colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
                                                          random_state=0)),
        "rf_clf": ("RandomForestClassifier", "clf3", dict(n_estimators=20, random_state=0)),
        "rf_reg": ("RandomForestRegressor", "reg", dict(n_estimators=20, random_state=0)),
        "rf_reg_groves": ("RandomForestRegressor", "reg", dict(n_estimators=20, random_state=0,
                                                              inference_engine="parallel_groves")),
        "dt_best_clf": ("DecisionTreeClassifier", "clf3", dict(max_depth=8)),
        "bagging_reg": ("BaggingRegressor", "reg", dict(n_estimators=5, random_state=0)),
        "adaboost_clf": ("AdaBoostClassifier", "clf2", dict(n_estimators=5, random_state=0)),
    }


def _run(name):
    import numpy as np
    import mojolearn as ml
    cls, kind, kw = _cases()[name]
    X, y = _data(kind)
    try:
        model = getattr(ml, cls)(**kw)
        if cls == "RandomTreesEmbedding":
            model.fit(X)
            outs = [np.asarray(model.transform(X).toarray() if hasattr(model.transform(X), "toarray")
                               else model.transform(X), dtype=np.float32)]
        else:
            if name == "et_clf_weighted":
                w = (np.arange(len(y)) % 3).astype(np.float32)
                model.fit(X, y, sample_weight=w)
            else:
                model.fit(X, y)
            outs = [np.asarray(model.predict(X))]
            if hasattr(model, "predict_proba") and kind.startswith("clf"):
                outs.append(np.asarray(model.predict_proba(X)))
        h = hashlib.sha256()
        for o in outs:
            h.update(np.ascontiguousarray(o).tobytes())
        return f"{name} sha256={h.hexdigest()}"
    except Exception as e:  # noqa: BLE001 - the line records the refusal
        return f"{name} error {type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"


def main():
    names = sorted(_cases())
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:
        cores = os.cpu_count() or 1
    workers = max(1, min(int(os.environ.get("BM_TREES_WORKERS", cores)), len(names)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for line in pool.map(_run, names):
            print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
