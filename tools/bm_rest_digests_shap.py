#!/usr/bin/env python3
"""Output digests of KernelExplainer and PermutationExplainer, whose chunk loops moved from the door to the typed
RUN xtrees/agnostic_values_run.mojo (lane bm-rest-shap, 2026-11-23). A testing tool, never product execution.

One line per configuration, `<name> sha256=<digest of the values' and the expected value's bytes>` or
`<name> error <class>: <first line of the message>` (a refusal is digested by its words). Run it on two builds of
the CPU column (before and after the change) and diff the files: equal lines mean equal bits.

    PYTHONPATH=<repo>/python python3 tools/bm_rest_digests_shap.py > out.txt

The configurations cover every option path the change touched on the CPU column: the native member route
(regressor predict, classifier predict_proba, the forests, DART, GBDT), the model's own door (a family whose member
does not serve the call: KNeighborsRegressor, LinearRegression, Ridge), the identity and logit links,
nsamples 'auto' with every coalition enumerated and an int that samples (M = 8), PermutationExplainer with 1 and
many permutations and two seeds, more rows than one worker's share (the parallel chunks), and the refusals
(a Python callable, l1_reg, nsamples and npermutations types and values, a non-finite row, a feature count
mismatch, zero rows).

Parallel plan: the configurations are independent, one per worker process (at most the allocated cores, at most
the configuration count).
"""
import hashlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor


def _data(kind, n=400, d=6, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    if kind == "reg":
        y = (X[:, 0] * 2 + np.sin(X[:, 1]) - 0.5 * X[:, 2] * X[:, 3] + 0.1 * rng.normal(size=n)).astype(np.float32)
    else:
        k = int(kind[3:])
        y = (np.digitize(X[:, 0] + 0.5 * X[:, 1], np.linspace(-1, 1, k - 1))).astype(np.int64)
    return X, y


# name: (explainer, model class, data kind, features, model kwargs, explainer kwargs, values kwargs, rows)
def _cases():
    tree = dict(max_depth=5)
    return {
        "k_tree_reg": ("K", "DecisionTreeRegressor", "reg", 6, tree, {}, {}, 12),
        "k_tree_reg_many_rows": ("K", "DecisionTreeRegressor", "reg", 6, tree, {}, {}, 97),
        "k_rf_clf3": ("K", "RandomForestClassifier", "clf3", 5, dict(n_estimators=8, max_depth=4, random_state=0),
                      {}, {}, 9),
        "k_et_reg": ("K", "ExtraTreesRegressor", "reg", 6, dict(n_estimators=8, random_state=0), {}, {}, 10),
        "k_gbdt_reg": ("K", "GradientBoostingRegressor", "reg", 6, dict(n_estimators=20, random_state=0), {}, {}, 8),
        "k_dart_logit": ("K", "DARTClassifier", "clf2", 6, dict(n_estimators=20, random_state=0),
                         dict(link="logit"), {}, 6),
        "k_dart_identity": ("K", "DARTClassifier", "clf2", 6, dict(n_estimators=20, random_state=0), {}, {}, 6),
        "k_rf_sampled_m8": ("K", "RandomForestRegressor", "reg", 8, dict(n_estimators=6, random_state=0),
                            dict(random_state=3), dict(nsamples=100), 11),
        "k_knn_door": ("K", "KNeighborsRegressor", "reg", 6, dict(n_neighbors=5), {}, {}, 7),
        "k_linear_door": ("K", "LinearRegression", "reg", 6, {}, {}, {}, 7),
        "k_l1_false": ("K", "DecisionTreeRegressor", "reg", 6, tree, {}, dict(l1_reg=False), 4),
        "k_zero_rows": ("K", "DecisionTreeRegressor", "reg", 6, tree, {}, {}, 0),
        "p_tree_reg": ("P", "DecisionTreeRegressor", "reg", 6, tree, dict(random_state=0), dict(npermutations=50), 12),
        "p_tree_reg_many_rows": ("P", "DecisionTreeRegressor", "reg", 6, tree, dict(random_state=1),
                                 dict(npermutations=10), 97),
        "p_tree_reg_one": ("P", "DecisionTreeRegressor", "reg", 6, tree, dict(random_state=0),
                           dict(npermutations=1), 5),
        "p_rf_clf3": ("P", "RandomForestClassifier", "clf3", 5, dict(n_estimators=8, max_depth=4, random_state=0),
                      dict(random_state=0), dict(npermutations=20), 9),
        "p_gbdt_reg": ("P", "GradientBoostingRegressor", "reg", 6, dict(n_estimators=20, random_state=0),
                       dict(random_state=2), dict(npermutations=20), 8),
        "p_ridge_door": ("P", "Ridge", "reg", 6, {}, dict(random_state=0), dict(npermutations=20), 7),
        "p_zero_rows": ("P", "DecisionTreeRegressor", "reg", 6, tree, {}, {}, 0),
        # refusals: digested by their words
        "r_callable": ("K", None, "reg", 6, {}, {}, {}, 3),
        "r_l1": ("K", "DecisionTreeRegressor", "reg", 6, tree, {}, dict(l1_reg="num_features(3)"), 3),
        "r_nsamples_type": ("K", "DecisionTreeRegressor", "reg", 6, tree, {}, dict(nsamples=2.5), 3),
        "r_link": ("K", "DecisionTreeRegressor", "reg", 6, tree, dict(link="probit"), {}, 3),
        "r_nonfinite_k": ("K", "DecisionTreeRegressor", "reg", 6, tree, {}, {}, -3),
        "r_nonfinite_p": ("P", "DecisionTreeRegressor", "reg", 6, tree, {}, dict(npermutations=5), -3),
        "r_features": ("K", "DecisionTreeRegressor", "reg", 6, tree, {}, {}, -1),
        "r_npermutations_zero": ("P", "DecisionTreeRegressor", "reg", 6, tree, {}, dict(npermutations=0), 3),
        "r_npermutations_type": ("P", "DecisionTreeRegressor", "reg", 6, tree, {}, dict(npermutations=2.0), 3),
    }


def _run(name):
    import numpy as np
    import mojolearn as ml
    which, cls, kind, d, mkw, ekw, vkw, rows = _cases()[name]
    X, y = _data(kind, d=d)
    try:
        if cls is None:
            model = lambda Z: np.asarray(Z, dtype=np.float64).sum(axis=1)  # noqa: E731 - refused by the door
        else:
            model = getattr(ml, cls)(**mkw).fit(X, y)
        explainer = (ml.KernelExplainer if which == "K" else ml.PermutationExplainer)(model, X[:20], **ekw)
        if rows == -1:
            Q = X[300:303, : d - 1].copy()
        elif rows == -3:
            Q = X[300:303].copy()
            Q[1, 2] = np.nan
        else:
            Q = X[300:300 + rows].copy()
        phi = np.asarray(explainer.shap_values(Q, **vkw))
        ev = np.asarray(explainer.expected_value, dtype=np.float64)
        h = hashlib.sha256()
        h.update(str(phi.shape).encode())
        h.update(np.ascontiguousarray(phi).tobytes())
        h.update(np.ascontiguousarray(ev).tobytes())
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
