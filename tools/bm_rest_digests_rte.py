#!/usr/bin/env python3
"""Output digests of the operations lane bm-rest-rte moved to a typed RUN (2026-11-23): RandomTreesEmbedding's fit
(xtrees/trees_boost_run.mojo `rte_fit_run`) with its transform and fit_transform, and the ExtraTrees fits whose
PACK now goes through `forest_et_store_fit` (bindings/trees_public_forest_door.mojo). A testing tool, never product
execution.

One line per configuration, `<name> sha256=<digest of every output's and fitted field's bytes>` or
`<name> error <type>: <first line of the message>` (a refusal, digested by its words). Run it on two builds of the
CPU column (the base and the head) and diff the files: equal lines mean equal bits.

    PYTHONPATH=<repo>/python python3 tools/bm_rest_digests_rte.py > out.txt

Parallel plan: the configurations are independent fits, one per worker process (at most the allocated cores, at
most the configuration count; `BM_REST_WORKERS` lowers it).
"""
import hashlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor

# The forest's fitted fields the embedding and the ExtraTrees fits store (`_store_model`, `forest_et_store_fit`).
_FOREST_FIELDS = ("_offsets", "_colid", "_quesval", "_left_child", "_leaves", "n_features_in_", "_n_trees",
                  "_num_outputs", "depth_cap_bound_", "max_depth_resolved_", "max_features_",
                  "_n_samples_bootstrap")
_RTE_FIELDS = ("_tree_base", "_node_col", "n_trees_", "n_output_features_", "n_features_in_")


def _data(kind, n=600, d=8, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    if kind == "nan":
        X[5, 2] = np.nan
        return X, None
    if kind == "f64":
        return X.astype(np.float64), None
    if kind == "fortran":
        return np.asfortranarray(X), None
    if kind == "reg":
        return X, (X[:, 0] * 2 + np.sin(X[:, 1]) + 0.1 * rng.normal(size=n)).astype(np.float32)
    if kind == "clf3":
        return X, np.digitize(X[:, 0] + 0.5 * X[:, 1], np.linspace(-1, 1, 2)).astype(np.int64)
    return X, None


def _cases():
    rte = "RandomTreesEmbedding"
    return {
        "rte_default": (rte, "x", "fit", dict(random_state=0)),
        "rte_depth5": (rte, "x", "fit", dict(n_estimators=15, max_depth=5, random_state=0)),
        "rte_leaves": (rte, "x", "fit", dict(n_estimators=12, max_depth=None, max_leaf_nodes=9, random_state=3)),
        "rte_min_leaf": (rte, "x", "fit", dict(n_estimators=10, min_samples_leaf=7, min_samples_split=11,
                                               random_state=1)),
        "rte_impurity": (rte, "x", "fit", dict(n_estimators=10, min_impurity_decrease=0.05, random_state=2)),
        "rte_seed_none": (rte, "x", "fit", dict(n_estimators=6, max_depth=3)),
        "rte_fit_transform": (rte, "x", "fit_transform", dict(n_estimators=15, max_depth=4, random_state=5)),
        "rte_float64_x": (rte, "f64", "fit", dict(n_estimators=8, max_depth=4, random_state=0)),
        "rte_fortran_x": (rte, "fortran", "fit", dict(n_estimators=8, max_depth=4, random_state=0)),
        "rte_refuse_nan": (rte, "nan", "fit", dict(n_estimators=4, random_state=0)),
        "rte_refuse_sparse": (rte, "x", "fit", dict(sparse_output=True, random_state=0)),
        "rte_refuse_split": (rte, "x", "fit", dict(min_samples_split=1, random_state=0)),
        "rte_refuse_seed": (rte, "x", "fit", dict(random_state=-1)),
        "rte_refuse_features": (rte, "x", "transform_wide", dict(n_estimators=4, random_state=0)),
        "et_reg": ("ExtraTreesRegressor", "reg", "fit", dict(n_estimators=20, max_depth=10, random_state=0)),
        "et_reg_bootstrap": ("ExtraTreesRegressor", "reg", "fit", dict(n_estimators=10, bootstrap=True,
                                                                      max_samples=0.5, random_state=3)),
        "et_clf": ("ExtraTreesClassifier", "clf3", "fit", dict(n_estimators=20, random_state=0)),
    }


def _fields(model, names):
    import numpy as np
    out = []
    for name in names:
        value = model.__dict__.get(name)
        out.append(f"{name}=None".encode() if value is None else
                   (np.ascontiguousarray(np.asarray(value)).tobytes() if hasattr(value, "__len__") or
                    hasattr(value, "__array_interface__") else repr(value).encode()))
    return out


def _run(name):
    import numpy as np
    import mojolearn as ml
    cls, kind, op, kw = _cases()[name]
    X, y = _data(kind)
    try:
        model = getattr(ml, cls)(**kw)
        if cls == "RandomTreesEmbedding":
            if op == "fit_transform":
                outs = [np.asarray(model.fit_transform(X), dtype=np.float64)]
            elif op == "transform_wide":
                model.fit(X)
                outs = [np.asarray(model.transform(np.hstack([X, X])), dtype=np.float64)]
            else:
                model.fit(X)
                outs = [np.asarray(model.transform(X), dtype=np.float64)]
            outs += _fields(model, _RTE_FIELDS) + _fields(model.forest_, _FOREST_FIELDS)
        else:
            model.fit(X, y)
            outs = [np.asarray(model.predict(X))]
            if kind.startswith("clf"):
                outs.append(np.asarray(model.predict_proba(X)))
            outs += _fields(model, _FOREST_FIELDS)
        h = hashlib.sha256()
        for o in outs:
            h.update(o if isinstance(o, bytes) else np.ascontiguousarray(o).tobytes())
        return f"{name} sha256={h.hexdigest()}"
    except Exception as e:  # noqa: BLE001 - the line records the refusal
        return f"{name} error {type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"


def main():
    names = sorted(_cases())
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:
        cores = os.cpu_count() or 1
    workers = max(1, min(int(os.environ.get("BM_REST_WORKERS", cores)), len(names)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for line in pool.map(_run, names):
            print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
