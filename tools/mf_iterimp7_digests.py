# SPDX-License-Identifier: Apache-2.0
"""Digests of lane mf-iterimp7 (IterativeImputer: the door rounds deleted; every supplied estimator runs through its
native member). Run on the base build (origin/lane/member-fits-7 1f029628db) and on the head build, then diff:

    python tools/mf_iterimp7_digests.py > base.txt     # base wheel / build
    python tools/mf_iterimp7_digests.py > head.txt     # head wheel / build

One line per configuration: `<name> sha256=<hash>` over fit_transform, the fitted fields, every step's public
fitted fields and its predictions, and transform twice; or `<name> error <Type>: <first line>` for a refusal.
Independent configurations run across the allocation's cores (MF_ITERIMP7_WORKERS caps the count).

EXPECTED (head vs base):
  * ridge_*, gnb_*, logcw_* (the routes the member already served): equal.
  * lp_knn, ls_knn (LabelPropagation / LabelSpreading on a target holding -1): base = the door rounds (the public
    fit per step), head = the member with the marker word: expected EQUAL values (the same kernels); a difference is
    a fault in the marker word (x_prep/iterimp_run.mojo `iterimp_marker_code`, bindings/prep_iterimp_native_door.mojo
    `_member_pack`).
  * gs_logreg, vote_reg, bag_reg (META kinds): equal when the base already took the member; where the base took the
    door rounds the values are expected equal (same kernels) and the fitted `_step_classes` field differs (None at
    base, the step labels at head).
  * kmeans: equal values expected (base door rounds: KMeans.fit/predict; head: its member).
  * nn_refused, chol_refused: base raised from the door rounds' predict call; head raises scikit-learn's words
    "The 'estimator' parameter of IterativeImputer must be None or an object implementing 'fit' and 'predict'."
  * lasso_host: base "TypeError ... requires a mojolearn estimator; got HostLasso"; head HostLasso's own fit words
    ("ImportError: mojolearn: no CPU implementation of the HostLasso fit ...").
"""
import hashlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mf_iterimp_digests import _bytes, _fitted  # noqa: E402 - the same field hashing as the iterimp rounds 1-4


def _data(seed=0, n=300, d=4, low=-1, high=3, ints=True):
    import numpy as np
    rng = np.random.default_rng(seed)
    if ints:
        base = rng.integers(low, high, n)
        X = np.stack([np.clip(base + ((j * 7 + rng.integers(0, 2, n)) % 2), low, high - 1) for j in range(d)],
                     axis=1).astype(np.float32)
    else:
        Z = rng.standard_normal((n, 2))
        X = (Z @ rng.standard_normal((2, d)) + 0.1 * rng.standard_normal((n, d))).astype(np.float32)
    X[rng.random((n, d)) < 0.2] = np.nan
    return X


def _estimator(name):
    import mojolearn as ml
    from mojolearn import _classical_host as host
    from mojolearn.model_selection import GridSearchCV
    if name == "ridge":
        return ml.Ridge(alpha=1.0)
    if name == "gnb":
        return ml.GaussianNB()
    if name == "logcw":
        return ml.LogisticRegression(class_weight={0.0: 2.0, 1.0: 1.0, 2.0: 0.5}, max_iter=200)
    if name == "lp":
        return ml.LabelPropagation(kernel="knn", n_neighbors=5, max_iter=50)
    if name == "ls":
        return ml.LabelSpreading(kernel="knn", n_neighbors=5, max_iter=50)
    if name == "gs":
        return GridSearchCV(ml.LogisticRegression(max_iter=200), {"C": [0.5, 1.0]}, cv=2, refit=True)
    if name == "vote":
        return ml.VotingRegressor([("r", ml.Ridge(alpha=1.0)), ("t", ml.DecisionTreeRegressor(max_depth=3,
                                                                                            random_state=0))])
    if name == "bag":
        return ml.BaggingRegressor(ml.Ridge(alpha=1.0), n_estimators=4, random_state=0)
    if name == "kmeans":
        return ml.KMeans(n_clusters=3, random_state=0, n_init=1)
    if name == "nn":
        return ml.NearestNeighbors(n_neighbors=3)
    if name == "chol":
        return ml.Cholesky()
    if name == "lasso_host":
        return host.HostLasso()
    raise KeyError(name)


def _cases():
    # name -> (estimator name, data kind, IterativeImputer keywords)
    return {
        "ridge_float": ("ridge", "float", dict(max_iter=3)),
        "ridge_random": ("ridge", "float", dict(max_iter=3, imputation_order="random", random_state=2,
                                                 add_indicator=True)),
        "gnb_int": ("gnb", "int0", dict(max_iter=2)),
        "logcw_int": ("logcw", "int0", dict(max_iter=2)),
        "lp_knn": ("lp", "int", dict(max_iter=2)),
        "ls_knn": ("ls", "int", dict(max_iter=2, imputation_order="descending")),
        "gs_logreg": ("gs", "int0", dict(max_iter=1)),
        "vote_reg": ("vote", "float", dict(max_iter=2)),
        "bag_reg": ("bag", "float", dict(max_iter=2)),
        "kmeans": ("kmeans", "float", dict(max_iter=1)),
        "nn_refused": ("nn", "float", dict()),
        "chol_refused": ("chol", "float", dict()),
        "lasso_host": ("lasso_host", "float", dict()),
    }


def _x(kind, seed):
    if kind == "float":
        return _data(seed, ints=False)
    if kind == "int0":
        return _data(seed, low=0, high=3)
    return _data(seed)


def _run(name):
    import numpy as np
    import mojolearn as ml
    est_name, kind, kw = _cases()[name]
    X, Xh = _x(kind, 0), _x(kind, 1)
    try:
        m = ml.IterativeImputer(estimator=_estimator(est_name), **kw)
        h = hashlib.sha256()
        h.update(_bytes(m.fit_transform(X)))
        f = m.__dict__
        for key in ("_keep", "_indicator", "_bounds_k", "n_iter_", "n_features_with_missing_", "_rng", "_post_step",
                    "n_features_in_", "_empty_kept"):
            h.update(key.encode() + _bytes(f.get(key)))
        block = np.nan_to_num(Xh)
        for step in f.get("imputation_sequence_", ()):
            h.update(_bytes([step[0], step[1]]))
            est = step[2]
            h.update(_fitted(est))
            cols = list(step[1])
            h.update(_bytes(est.predict(np.ascontiguousarray(block[:, cols]))))
        h.update(_bytes(m.transform(Xh)))
        h.update(_bytes(m.transform(Xh)))
        classes = hashlib.sha256(_bytes(f.get("_step_classes"))).hexdigest()
        return f"{name} sha256={h.hexdigest()}\n{name} step_classes sha256={classes}"
    except Exception as e:  # noqa: BLE001 - the line records the refusal
        return f"{name} error {type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"


def main():
    import mojolearn  # noqa: F401 - fail before forking when the build is missing
    names = sorted(_cases())
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:
        cores = os.cpu_count() or 1
    workers = max(1, min(int(os.environ.get("MF_ITERIMP7_WORKERS", cores)), len(names)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for line in pool.map(_run, names):
            print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
