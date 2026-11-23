#!/usr/bin/env python3
"""Output digests of mutual_info_classif / mutual_info_regression and RFE, whose
programs and native-member rounds moved to the typed RUN of
bindings/prep_select_run.mojo (lane bm-rest-sel, 2026-11-23). A testing tool,
never product execution.

One line per configuration, `<name> sha256=<digest of every output's bytes>`
or `<name> error <message>` (a refusal, digested by its words). Run it on two
builds of the CPU column (before and after a change) and diff the files: equal
lines mean equal bits.

    PYTHONPATH=<repo>/python python3 tools/bm_rest_digests_sel.py > out.txt

Covered option paths: every discrete_features form (auto, True, False, a bool
mask, indices with negatives), both targets, the all-discrete and all-continuous
programs, several seeds and n_neighbors, every mutual-information refusal the
door words (n_neighbors, the mask length, a bad index, a bad entry type,
one sample per class, one sample per value); RFE over an estimator without a
usable native member (LDA: its own doors), over native members with a
coefficient block (LogisticRegression, LinearRegression, Ridge), with feature
importances (DecisionTree, RandomForest: the importance word), every
importance_getter form, fractional n_features_to_select and step, the step
refusal, fit_params (the own-door path), fit_transform, predict, score and
the fitted attributes (ranking_, support_, n_features_, the final
estimator's coef_ / feature_importances_).

Parallel plan: the configurations are independent, one per worker process
(at most the allocated cores, at most the configuration count).
"""
import hashlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor


def _mi_data(seed=0, n=400):
    import numpy as np
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    Xc = (rng.standard_normal((n, 3)) + y[:, None] * np.array([0.0, 0.5, 1.5])).astype(np.float32)
    Xd = np.stack([rng.integers(0, 4, n), (y + rng.integers(0, 2, n)) % 4], axis=1).astype(np.float32)
    X = np.concatenate([Xd[:, :1], Xc[:, :2], Xd[:, 1:], Xc[:, 2:]], axis=1)
    yr = (X[:, 2] + 0.5 * rng.standard_normal(n)).astype(np.float32)
    return X, y, yr


def _rfe_data(kind, seed=0, n=300, d=9):
    import numpy as np
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3 if kind == "clf3" else 2, n)
    X = (rng.standard_normal((n, d)) + y[:, None] * np.linspace(0, 1.4, d)).astype(np.float32)
    if kind == "reg":
        y = (X[:, 0] * 1.5 - X[:, 4] + 0.3 * rng.standard_normal(n)).astype(np.float32)
    return X, y


_MI = {
    "mi_classif_auto": ("classif", dict()),
    "mi_regression_auto": ("regression", dict()),
    "mi_classif_true": ("classif", dict(discrete_features=True)),
    "mi_regression_mask": ("regression", dict(discrete_features=[True, False, False, True, False])),
    "mi_classif_mask": ("classif", dict(discrete_features=[True, False, False, True, False])),
    "mi_classif_indices": ("classif", dict(discrete_features=[0, -2])),
    "mi_regression_indices": ("regression", dict(discrete_features=[3, 0])),
    "mi_classif_false_seed7": ("classif", dict(discrete_features=False, random_state=7)),
    "mi_regression_seed3_k5": ("regression", dict(random_state=3, n_neighbors=5)),
    "mi_classif_k1": ("classif", dict(n_neighbors=1, random_state=0)),
    "mi_classif_disc_only": ("classif_disc", dict(discrete_features=True)),
    "mi_regression_disc_only": ("regression_disc", dict(discrete_features=True)),
    "mi_refuse_k": ("classif", dict(n_neighbors=40)),
    "mi_refuse_mask_len": ("classif", dict(discrete_features=[True, False])),
    "mi_refuse_index": ("classif", dict(discrete_features=[0, 9])),
    "mi_refuse_index_type": ("classif", dict(discrete_features=[9, "a"])),
    "mi_refuse_string": ("classif", dict(discrete_features="all")),
    "mi_refuse_one_per_value": ("regression", dict(discrete_features=True)),
    "mi_refuse_one_per_class": ("classif_unique", dict()),
}

_RFE = {
    "rfe_lda": ("LinearDiscriminantAnalysis", "clf3", dict(), dict(n_features_to_select=3, step=2)),
    "rfe_lda_getter_coef": ("LinearDiscriminantAnalysis", "clf3", dict(),
                            dict(n_features_to_select=3, step=2, importance_getter="coef_")),
    "rfe_logistic": ("LogisticRegression", "clf3", dict(), dict(n_features_to_select=4, step=1)),
    "rfe_logistic_bin_frac": ("LogisticRegression", "clf2", dict(), dict(n_features_to_select=0.4, step=0.3)),
    "rfe_linear": ("LinearRegression", "reg", dict(), dict(n_features_to_select=2, step=3)),
    "rfe_ridge_none": ("Ridge", "reg", dict(alpha=0.5), dict(n_features_to_select=None, step=2)),
    "rfe_tree_imp": ("DecisionTreeClassifier", "clf3", dict(max_depth=5, random_state=0),
                     dict(n_features_to_select=3, step=2)),
    "rfe_tree_getter_imp": ("DecisionTreeClassifier", "clf3", dict(max_depth=5, random_state=0),
                            dict(n_features_to_select=3, step=2, importance_getter="feature_importances_")),
    "rfe_forest_reg": ("RandomForestRegressor", "reg", dict(n_estimators=8, random_state=0),
                       dict(n_features_to_select=3, step=2)),
    "rfe_logistic_fit_params": ("LogisticRegression", "clf3", dict(), dict(n_features_to_select=4, step=2)),
    "rfe_refuse_step": ("LogisticRegression", "clf3", dict(), dict(n_features_to_select=3, step=0)),
    "rfe_refuse_getter": ("LogisticRegression", "clf3", dict(), dict(importance_getter="feature_importances_")),
}


def _digest(outs):
    import numpy as np
    h = hashlib.sha256()
    for o in outs:
        h.update(np.ascontiguousarray(np.asarray(o)).tobytes())
    return h.hexdigest()


def _run_mi(name):
    import numpy as np
    import mojolearn as ml
    kind, kw = _MI[name]
    X, y, yr = _mi_data()
    if kind.endswith("_disc"):
        X = X[:, [0, 3]]
    if kind == "classif_unique":
        y = np.arange(len(y))
    kw = dict(kw)
    kw.setdefault("random_state", 0)
    f = ml.mutual_info_regression if kind.startswith("regression") else ml.mutual_info_classif
    return [f(X, yr if kind.startswith("regression") else y, **kw)]


def _run_rfe(name):
    import numpy as np
    import mojolearn as ml
    cls, kind, ekw, rkw = _RFE[name]
    X, y = _rfe_data(kind)
    model = ml.RFE(getattr(ml, cls)(**ekw), **rkw)
    if name.endswith("_fit_params"):
        model.fit(X, y, sample_weight=np.ones(len(y), dtype=np.float32))
    else:
        model.fit(X, y)
    outs = [model.ranking_, np.asarray(model.support_), np.asarray(model.n_features_), model.transform(X),
            model.predict(X), np.asarray(model.score(X, y)), model.fit_transform(X, y), model.ranking_]
    est = model.estimator_
    for attr in ("coef_", "intercept_", "feature_importances_"):
        if hasattr(est, attr):
            outs.append(np.asarray(getattr(est, attr)))
    return outs


def _run(name):
    try:
        outs = _run_mi(name) if name in _MI else _run_rfe(name)
        return f"{name} sha256={_digest(outs)}"
    except Exception as e:  # noqa: BLE001 - the line records the refusal
        return f"{name} error {type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"


def main():
    names = sorted(_MI) + sorted(_RFE)
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:
        cores = os.cpu_count() or 1
    workers = max(1, min(int(os.environ.get("BM_REST_SEL_WORKERS", cores)), len(names)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for line in pool.map(_run, names):
            print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
