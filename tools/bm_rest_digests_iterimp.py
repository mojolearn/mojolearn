#!/usr/bin/env python3
"""Output and fitted-state digests of IterativeImputer, whose built-in
Bayesian ridge program moved to a typed RUN (x_prep/iterimp_run.mojo; lane
bm-rest-iterimp, 2026-11-23). A testing tool, never product execution.

One line per configuration, `<name> sha256=<digest>` over the raw bytes of
fit_transform(X), every fitted attribute (initial_imputer_.statistics_ and
_fill, _keep, _indicator, _bounds_k, n_iter_, n_features_with_missing_, each
imputation step's coef / intercept or, for a supplied estimator, its feature
and neighbour list, the posterior records, _rng, _post_step), transform(Xh)
twice (the posterior step count moves) and fit(X).transform(X); or
`<name> error <class>: <first line>` for a refusal. Run it on two builds of
the CPU column (before and after a change) and diff the files: equal lines
mean equal bits.

    PYTHONPATH=<repo>/python python3 tools/bm_rest_digests_iterimp.py > out.txt

Parallel plan: the configurations are independent fits, one per worker
process (at most the allocated cores, at most the configuration count).
"""
import hashlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor


def _data(seed=0, n=400, d=5, inf=False):
    # test_x_prep_iterative.py::_data
    import numpy as np
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, 2))
    X = (Z @ rng.standard_normal((2, d)) + 0.1 * rng.standard_normal((n, d))).astype(np.float32)
    X[rng.random((n, d)) < 0.15] = np.nan
    if inf:
        X[3, 1] = np.inf
    return X


def _cases():
    import mojolearn as ml
    return {
        "default": dict(),
        "median_desc_3": dict(max_iter=3, imputation_order="descending", initial_strategy="median"),
        "bounds": dict(min_value=-1.0, max_value=1.0),
        "bounds_per_column": dict(min_value=[-1.0, -2.0, -0.5, -3.0, -1.0], max_value=[1.0, 2.0, 0.5, 3.0, 1.0]),
        "indicator": dict(add_indicator=True),
        "roman": dict(imputation_order="roman"),
        "arabic_skip": dict(imputation_order="arabic", skip_complete=True),
        "random": dict(imputation_order="random", random_state=3),
        "random_skip": dict(imputation_order="random", skip_complete=True, random_state=7),
        "nnf3": dict(n_nearest_features=3, random_state=3),
        "nnf2_random_20": dict(n_nearest_features=2, imputation_order="random", random_state=5, max_iter=20),
        "posterior": dict(sample_posterior=True, random_state=1),
        "posterior_bounds": dict(sample_posterior=True, min_value=-0.5, max_value=0.5, random_state=2),
        "posterior_nnf_indicator": dict(sample_posterior=True, n_nearest_features=2, add_indicator=True,
                                        random_state=4),
        "max_iter0": dict(max_iter=0),
        "tol0": dict(tol=0.0, max_iter=4),
        "constant_keep_empty": dict(initial_strategy="constant", fill_value=0.5, keep_empty_features=True,
                                    empty_column=True),
        "most_frequent_empty": dict(initial_strategy="most_frequent", empty_column=True),
        "missing_minus1": dict(missing_values=-1.0, numeric_missing=True),
        "ridge": dict(estimator=ml.Ridge(alpha=2.0)),
        "ridge_indicator_4": dict(estimator=ml.Ridge(alpha=2.0), add_indicator=True, max_iter=4),
        "ridge_nnf_random": dict(estimator=ml.Ridge(alpha=1.0), n_nearest_features=2, imputation_order="random",
                                 random_state=6),
        "bayes_posterior": dict(estimator=ml.BayesianRidge(), sample_posterior=True, random_state=8),
        "refuse_infinity": dict(inf=True),
        "refuse_bounds_short": dict(min_value=[-1.0, -1.0], max_value=[1.0, 1.0]),
        "refuse_strategy": dict(initial_strategy="mode"),
        "refuse_posterior_ridge": dict(estimator=ml.Ridge(), sample_posterior=True),
    }


def _bytes(value):
    import numpy as np
    if value is None:
        return b"None"
    if isinstance(value, (list, tuple)):
        return b"[" + b",".join(_bytes(v) for v in value) + b"]"
    if isinstance(value, (bool, int, float, str)):
        return repr(value).encode()
    if hasattr(value, "get_params"):  # a fitted supplied estimator: its predictions on a fixed block
        return type(value).__name__.encode()
    return np.ascontiguousarray(np.asarray(value)).tobytes()


def _run(name):
    import numpy as np
    import mojolearn as ml
    kw = dict(_cases()[name])
    inf = kw.pop("inf", False)
    empty = kw.pop("empty_column", False)
    numeric = kw.pop("numeric_missing", False)
    X = _data(0, inf=inf)
    Xh = _data(1)
    if empty:
        X[:, 2] = np.nan
    if numeric:
        X = np.where(np.isnan(X), np.float32(-1.0), X).astype(np.float32)
        Xh = np.where(np.isnan(Xh), np.float32(-1.0), Xh).astype(np.float32)
    try:
        m = ml.IterativeImputer(**kw)
        h = hashlib.sha256()
        h.update(_bytes(m.fit_transform(X)))
        f = m.__dict__
        imp = f["initial_imputer_"].__dict__
        for key in ("statistics_", "_fill", "_keep", "_indicator"):
            h.update(key.encode() + _bytes(imp.get(key)))
        for key in ("_keep", "_indicator", "_bounds_k", "n_iter_", "n_features_with_missing_", "_rng",
                    "_post_step", "n_features_in_"):
            h.update(key.encode() + _bytes(f.get(key)))
        for step in f["imputation_sequence_"]:
            h.update(_bytes(list(step)))
            if hasattr(step[-1], "predict"):
                h.update(_bytes(step[-1].predict(np.nan_to_num(Xh)[:, :len(step[1])] if len(step[1]) else
                                                 np.zeros((len(Xh), 0), np.float32))))
        h.update(_bytes(f.get("_posterior")))
        h.update(_bytes(m.transform(Xh)))
        h.update(_bytes(m.transform(Xh)))
        h.update(_bytes(f.get("_post_step")))
        h.update(_bytes(ml.IterativeImputer(**kw).fit(X).transform(X)))
        return f"{name} sha256={h.hexdigest()}"
    except Exception as e:  # noqa: BLE001 - the line records the refusal
        return f"{name} error {type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"


def main():
    import mojolearn  # noqa: F401 - fail before forking when the build is missing
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
