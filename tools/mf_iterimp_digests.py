#!/usr/bin/env python3
"""Output and fitted-state digests of IterativeImputer with a supplied
estimator, whose rounds moved from the door's per-step estimator calls to the
typed member RUN (x_prep/iterimp_run.mojo `iterimp_member_rounds`; lane
mf-iterimp, 2026-11-23). A testing tool, never product execution.

One line per configuration, `<name> sha256=<digest>` over the raw bytes of
fit_transform(X), the fitted fields (initial_imputer_.statistics_, _keep,
_indicator, _bounds_k, n_iter_, n_features_with_missing_, _rng, _post_step),
every imputation step's feat_idx, neighbor_feat_idx and its estimator's
fitted attributes (every public instance field ending in "_", by name and
bytes) and its predictions on a fixed block, transform(Xh) twice (the
posterior stream moves) and fit(X).transform(X); or
`<name> error <class>: <first line>` for a refusal. Run it on two builds of
the CPU column, the base (origin/lane/member-fits 4d76a9e16e) and the head,
and diff the files: equal lines mean equal bits.

    PYTHONPATH=<repo>/python python3 tools/mf_iterimp_digests.py > out.txt

Round 2 (lane mf-iterimp2): base origin/lane/member-fits d3a8864541. The
estimator grid adds ARDRegression; the fitted posterior attributes the
lane adds (sigma_, X_offset_, X_scale_; BayesianRidge's sigma_ is a lazy
attribute formed from the fitted factor on its first read) are left out of
the main line and hashed, read as attributes, on a second line `<name> posterior sha256=...` (`none` on the base,
which has none), so the main lines compare the old fields. The direct
BayesianRidge / ARDRegression cases (`bayesreg_*`, `ardreg_*`) hash
predict, predict(return_std=True) and every fitted field. Expected
differences base vs head: every `*_post1_*` line of bayes and ard (base: the
refusal, head: an answer), the `posterior` lines, and the
return_std line of the direct cases (base: the refusal); everything else
equal (transform with estimator= now runs over the held member states with
the same programs: equal words).

Round 3 (lane mf-iterimp3): base origin/lane/member-fits cdad3291fa. Added:
classifier estimators (GaussianNB(), LogisticRegression(),
DecisionTreeClassifier(random_state=0), KNeighborsClassifier()) on an
integer-valued matrix (`_data_int`) x order {ascending, random} x
add_indicator, fit and transform (now the member rounds and the held
states; base: the door rounds), a classifier on the continuous matrix (a
refusal in scikit-learn's continuous-target words on both), and the
degenerate shapes for the built-in estimator (`none_*`) as for Ridge and
GaussianNB: a kept empty column (keep_empty_features), one feature given,
one feature left after dropping all-NaN columns. The `_empty_kept` and
`_step_classes` fields are hashed on a third line `<name> fields sha256=...`
(`none` on the base). Expected differences base vs head: `*_keep_empty`
(the empty column is marked not missing: base, the estimator's refusal of
a fit on no rows or the built-in ridge's prediction of the whole column;
head, its fill), `*_one_feature` and `*_drop_to_one` (head: n_iter_ 0,
no step, the initial imputation, as scikit-learn), the classifier lines
(head: member rounds; the imputed values are expected equal, the steps'
fitted labels may be published in another form), and the `fields` lines.

Grid: estimator {Ridge(alpha=1), LinearRegression(), BayesianRidge(),
GaussianProcessRegressor(random_state=0), DecisionTreeRegressor(random_state=0),
KNeighborsRegressor()} x sample_posterior {False, True} x initial_strategy
{mean, median} x imputation_order {ascending, random, roman} x
n_nearest_features {None, 2} x add_indicator {False, True}, random_state 3;
plus the routes the member path leaves to the door (a kept empty column,
max_iter=0, one feature) and the refusals.

Parallel plan: the configurations are independent fits, one per worker
process (at most the allocated cores, at most the configuration count;
MF_ITERIMP_WORKERS lowers it when a measured memory limit needs that).
"""
import hashlib
import itertools
import os
import sys
from concurrent.futures import ProcessPoolExecutor

ESTIMATORS = ("ridge", "linreg", "bayes", "ard", "gpr", "tree", "knn")
CLASSIFIERS = ("gnb", "logreg", "ctree", "knnc")
POSTERIOR = ("sigma_", "X_offset_", "X_scale_")


def _data(seed=0, n=400, d=5):
    # test_x_prep_iterative.py::_data
    import numpy as np
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, 2))
    X = (Z @ rng.standard_normal((2, d)) + 0.1 * rng.standard_normal((n, d))).astype(np.float32)
    X[rng.random((n, d)) < 0.15] = np.nan
    return X


def _data_int(seed=0, n=400, d=4):
    # every column the row's label (0, 1, 2) plus a column-dependent shift, at most one NaN a row: a target a
    # classifier can impute, integer-valued in every column
    import numpy as np
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 3, n)
    X = np.stack([labels + (j % 2) for j in range(d)], axis=1).astype(np.float32)
    rows = rng.random(n) < 0.6
    cols = rng.integers(0, d, n)
    X[rows, cols[rows]] = np.nan
    return X


def _estimator(name):
    import mojolearn as ml
    if name == "none":
        return None
    if name == "gnb":
        return ml.GaussianNB()
    if name == "logreg":
        return ml.LogisticRegression()
    if name == "ctree":
        return ml.DecisionTreeClassifier(random_state=0)
    if name == "knnc":
        return ml.KNeighborsClassifier()
    if name == "ridge":
        return ml.Ridge(alpha=1.0)
    if name == "linreg":
        return ml.LinearRegression()
    if name == "bayes":
        return ml.BayesianRidge()
    if name == "ard":
        return ml.ARDRegression()
    if name == "gpr":
        return ml.GaussianProcessRegressor(random_state=0)
    if name == "tree":
        return ml.DecisionTreeRegressor(random_state=0)
    return ml.KNeighborsRegressor()


def _cases():
    cases = {}
    for est, post, strat, order, nnf, ind in itertools.product(
            ESTIMATORS, (False, True), ("mean", "median"), ("ascending", "random", "roman"), (None, 2),
            (False, True)):
        name = f"{est}_post{int(post)}_{strat}_{order}_nnf{nnf or 0}_ind{int(ind)}"
        cases[name] = dict(estimator=est, sample_posterior=post, initial_strategy=strat, imputation_order=order,
                           n_nearest_features=nnf, add_indicator=ind, random_state=3)
    # routes the member path leaves to the door rounds, and the refusals
    cases["ridge_keep_empty"] = dict(estimator="ridge", keep_empty_features=True, empty_column=True)
    cases["ridge_max_iter0"] = dict(estimator="ridge", max_iter=0)
    cases["ridge_one_feature"] = dict(estimator="ridge", one_feature=True)
    cases["ridge_bounds"] = dict(estimator="ridge", min_value=-1.0, max_value=1.0)
    cases["ridge_tol0"] = dict(estimator="ridge", tol=0.0, max_iter=4)
    cases["ridge_skip_complete"] = dict(estimator="ridge", imputation_order="arabic", skip_complete=True)
    cases["gpr_post_bounds"] = dict(estimator="gpr", sample_posterior=True, min_value=-0.5, max_value=0.5,
                                    random_state=2)
    cases["bayes_post_bounds"] = dict(estimator="bayes", sample_posterior=True, min_value=-0.5, max_value=0.5,
                                      random_state=2)
    cases["ard_post_bounds"] = dict(estimator="ard", sample_posterior=True, min_value=-0.5, max_value=0.5,
                                    random_state=2)
    # round 3: classifier members, the degenerate shapes for every route
    for est, order, ind in itertools.product(CLASSIFIERS, ("ascending", "random"), (False, True)):
        cases[f"{est}_int_{order}_ind{int(ind)}"] = dict(estimator=est, int_data=True, imputation_order=order,
                                                         add_indicator=ind, random_state=3)
    cases["gnb_continuous"] = dict(estimator="gnb")
    cases["logreg_continuous"] = dict(estimator="logreg")
    for est in ("none", "ridge", "gnb"):
        intd = est == "gnb"
        cases[f"{est}_keep_empty"] = dict(estimator=est, keep_empty_features=True, empty_column=True, int_data=intd)
        cases[f"{est}_keep_empty_skip"] = dict(estimator=est, keep_empty_features=True, empty_column=True,
                                               skip_complete=True, int_data=intd)
        cases[f"{est}_one_feature"] = dict(estimator=est, one_feature=True, int_data=intd)
        cases[f"{est}_drop_to_one"] = dict(estimator=est, drop_to_one=True, add_indicator=True, int_data=intd)
    # the direct predict(return_std=True) cases
    for est, fi, sw in itertools.product(("bayesreg", "ardreg"), (True, False), (False, True)):
        if est == "ardreg" and sw:
            continue  # ARDRegression.fit takes no sample_weight
        cases[f"{est}_fi{int(fi)}_sw{int(sw)}"] = dict(direct=est, fit_intercept=fi, weighted=sw)
    return cases


def _bytes(value):
    import numpy as np
    if value is None:
        return b"None"
    if isinstance(value, (list, tuple)):
        return b"[" + b",".join(_bytes(v) for v in value) + b"]"
    if isinstance(value, (bool, int, float, str)):
        return repr(value).encode()
    if isinstance(value, dict):
        return b"{" + b",".join(_bytes(k) + b":" + _bytes(value[k]) for k in sorted(value, key=str)) + b"}"
    if hasattr(value, "get_params"):
        return type(value).__name__.encode()
    try:
        return np.ascontiguousarray(np.asarray(value)).tobytes()
    except Exception:  # noqa: BLE001 - an opaque fitted object: its type only
        return type(value).__name__.encode()


def _fitted(est, posterior=False):
    # every public fitted instance field (name ending in "_"), read from the instance dictionary; the
    # posterior attributes of round 2 on their own (`posterior`), read as attributes (BayesianRidge.sigma_ is
    # formed on its first read, not held in the dictionary)
    fields = vars(est)
    if posterior:
        return b"".join(k.encode() + _bytes(getattr(est, k)) for k in POSTERIOR if hasattr(est, k))
    return b"".join(k.encode() + _bytes(fields[k]) for k in sorted(fields)
                    if k.endswith("_") and not k.startswith("_") and k not in POSTERIOR)


def _direct(name, kw):
    import numpy as np
    import mojolearn as ml
    rng = np.random.default_rng(7)
    X = rng.standard_normal((300, 6)).astype(np.float32)
    y = (X @ np.array([2.0, -1.0, 0.0, 0.5, 0.0, 1.5], np.float32) + 0.3).astype(np.float32)
    y += (0.2 * rng.standard_normal(300)).astype(np.float32)
    q = rng.standard_normal((64, 6)).astype(np.float32)
    cls = ml.BayesianRidge if kw["direct"] == "bayesreg" else ml.ARDRegression
    lines = []
    try:
        m = cls(fit_intercept=kw["fit_intercept"])
        if kw["weighted"]:
            m.fit(X, y, sample_weight=np.linspace(0.5, 2.0, 300).astype(np.float32))
        else:
            m.fit(X, y)
        h = hashlib.sha256()
        h.update(_fitted(m))
        h.update(_bytes(m.predict(q)))
        lines.append(f"{name} sha256={h.hexdigest()}")
        h = hashlib.sha256()
        h.update(_fitted(m, posterior=True))
        lines.append(f"{name} posterior sha256={h.hexdigest()}")
    except Exception as e:  # noqa: BLE001 - the line records the refusal
        lines.append(f"{name} error {type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}")
        return "\n".join(lines)
    try:
        mean, std = m.predict(q, return_std=True)
        h = hashlib.sha256()
        h.update(_bytes(mean))
        h.update(_bytes(std))
        lines.append(f"{name} return_std sha256={h.hexdigest()}")
    except Exception as e:  # noqa: BLE001 - the line records the refusal
        lines.append(f"{name} return_std error {type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}")
    return "\n".join(lines)


def _run(name):
    import numpy as np
    import mojolearn as ml
    kw = dict(_cases()[name])
    if "direct" in kw:
        return _direct(name, kw)
    kw["estimator"] = _estimator(kw["estimator"])
    empty = kw.pop("empty_column", False)
    one = kw.pop("one_feature", False)
    drop = kw.pop("drop_to_one", False)
    intd = kw.pop("int_data", False)
    X = _data_int(0) if intd else _data(0)
    Xh = _data_int(1) if intd else _data(1)
    if empty:
        X[:, 2] = np.nan
    if one:
        X = np.ascontiguousarray(X[:, :1])
        Xh = np.ascontiguousarray(Xh[:, :1])
    if drop:
        X[:, 1:] = np.nan
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
        block = np.nan_to_num(Xh)
        post = hashlib.sha256()
        held = False
        fields = hashlib.sha256()
        known = False
        for key in ("_empty_kept", "_step_classes"):
            if key in f:
                known = True
                fields.update(key.encode() + _bytes(f[key]))
        for step in f.get("imputation_sequence_", ()):
            h.update(_bytes([step[0], step[1]]))
            est = step[2]
            if not hasattr(est, "predict"):
                # the built-in estimator's steps are (feat_idx, coef_, intercept_)
                h.update(_bytes(est))
                continue
            h.update(_fitted(est))
            extra = _fitted(est, posterior=True)
            held = held or bool(extra)
            post.update(extra)
            cols = list(step[1]) if step[1] is not None else []
            h.update(_bytes(est.predict(np.ascontiguousarray(block[:, cols]) if cols else
                                        np.zeros((len(block), 0), np.float32))))
        h.update(_bytes(m.transform(Xh)))
        h.update(_bytes(m.transform(Xh)))
        h.update(_bytes(f.get("_post_step")))
        h.update(_bytes(f.get("_rng")))
        kw["estimator"] = _estimator(_cases()[name]["estimator"])
        h.update(_bytes(ml.IterativeImputer(**kw).fit(X).transform(X)))
        return (f"{name} sha256={h.hexdigest()}\n"
                f"{name} posterior {('sha256=' + post.hexdigest()) if held else 'none'}\n"
                f"{name} fields {('sha256=' + fields.hexdigest()) if known else 'none'}")
    except Exception as e:  # noqa: BLE001 - the line records the refusal
        return f"{name} error {type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"


def main():
    import mojolearn  # noqa: F401 - fail before forking when the build is missing
    names = sorted(_cases())
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:
        cores = os.cpu_count() or 1
    workers = max(1, min(int(os.environ.get("MF_ITERIMP_WORKERS", cores)), len(names)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for line in pool.map(_run, names):
            print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
