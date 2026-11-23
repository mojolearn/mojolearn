"""Before/after digests of the META member kind (lane mf-meta3, 2026-11-23; core/ensemble_member.mojo): every
caller that now reaches a VotingClassifier / VotingRegressor / StackingClassifier / StackingRegressor as a NATIVE
member through the member ABI, host column. Base = origin/lane/member-fits ccbef84e9b (those callers ran the
meta-estimator's own fit and predict doors: CalibratedClassifierCV's door units `_CalUnits`, the outer meta's door
loop `_fit_member`, IterativeImputer's door rounds, SHAP's `_DoorModel`), head = lane/mf-meta3.

One line per configuration and output: name and sha256 of the output bytes (dtype and shape included); a refusal
is digested by its words. EVERY line must be equal: the META kind runs the door's own RUNs in the door's order
(xtrees/trees_meta_vote_run.mojo vote_* / stack_place_run, the same member sets, the same folds of
`xt_device_folds`, the same float32 rounding of the meta features), so no bits change. A differing line is a defect
to send back, with one known exception class: a nested member's fit NOTICE (a convergence warning) is now raised at
the META's PACK, so its position among the warnings of one fit may move (warnings are not digested).

Configurations (each with predict, predict_proba where it exists, and the fitted nested members' coef_ /
intercept_ where they have them):
  * CalibratedClassifierCV(base) for base in {soft VotingClassifier(lr, gnb), StackingClassifier(lr, lda; cv 3)} x
    method {sigmoid, isotonic} x ensemble {True, False} x n_jobs {1, 4};
  * VotingClassifier([("inner", soft VotingClassifier(lr, gnb)), ("lr", lr)], voting soft) (a META member of a
    meta: the inner vote is admitted as META, its members through the leaf dispatch);
  * StackingClassifier([("vote", soft VotingClassifier(lr, gnb)), ("lda", lda)], cv 3) and its SHAP Kernel /
    Permutation values (the inner vote is served by its held META state, no `_DoorModel`);
  * IterativeImputer(estimator=VotingRegressor(ridge, linreg)) and (estimator=StackingRegressor(ridge, linreg;
    cv 3, passthrough True)), max_iter 3, transform of the fit X;
  * hard VotingClassifier as a calibration base (no predict_proba: the door units still answer: equal words);
  * GridSearchCV(VotingClassifier soft, {"weights": [[1, 1], [1, 2]]}, cv 3): best_params_, predict_proba.

The configurations are independent: one process each across the host cores.

    python tools/mf_meta3_digests.py > digests.txt
"""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import os
import sys

import numpy as np


def _digest(value):
    a = np.asarray(value)
    h = hashlib.sha256()
    h.update(f"{a.dtype.str}{a.shape}".encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:24]


def _data(seed, n, k, d=5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    s = X @ rng.normal(size=d) + 0.3 * rng.normal(size=n)
    if k == 0:
        return X, s.astype(np.float32)
    edges = np.quantile(s, np.linspace(0, 1, k + 1)[1:-1])
    return X, np.searchsorted(edges, s).astype(np.int64)


def _soft_vote(ml, weights=None):
    return ml.VotingClassifier([("lr", ml.LogisticRegression(max_iter=200)), ("gnb", ml.GaussianNB())],
                               voting="soft", weights=weights)


def _stack(ml):
    return ml.StackingClassifier([("lr", ml.LogisticRegression(max_iter=200)),
                                  ("lda", ml.LinearDiscriminantAnalysis())], cv=3)


def _members(model, tag):
    out = []
    for name in ("estimators_", "calibrated_classifiers_"):
        for i, e in enumerate(getattr(model, name, None) or ()):
            inner = getattr(e, "estimator", e)
            for attr in ("coef_", "intercept_", "theta_", "var_"):
                v = getattr(inner, attr, None)
                if v is not None:
                    out.append((f"{tag}_{name}{i}_{attr}", _digest(v)))
    return out


def _predicts(model, X, tag):
    out = [(f"{tag}_predict", _digest(model.predict(X)))]
    if hasattr(model, "predict_proba"):
        try:
            out.append((f"{tag}_proba", _digest(model.predict_proba(X))))
        except Exception as e:  # noqa: BLE001 - the words are the digest
            out.append((f"{tag}_proba_raised", f"{type(e).__name__}: {e}"))
    return out


def _configs():
    rows = []
    for base in ("vote", "stack", "hard"):
        for method in ("sigmoid", "isotonic"):
            for ensemble in (True, False):
                for jobs in (1, 4):
                    rows.append(("cal", base, method, ensemble, jobs))
    rows += [("nested_vote",), ("stack_of_vote",), ("shap", "K"), ("shap", "P"), ("iterimp", "vote"),
             ("iterimp", "stack"), ("search",)]
    return rows


CONFIGS = _configs()


def _run(index):
    import mojolearn as ml
    row = CONFIGS[index]
    tag = "_".join(str(w) for w in row)
    X, y = _data(index % 3 + 1, 150, 3)
    Xt, _ = _data(97, 40, 3)
    try:
        if row[0] == "cal":
            _, base, method, ensemble, jobs = row
            est = _soft_vote(ml) if base == "vote" else (_stack(ml) if base == "stack" else
                                                         ml.VotingClassifier([("lr", ml.LogisticRegression()),
                                                                              ("gnb", ml.GaussianNB())],
                                                                             voting="hard"))
            model = ml.CalibratedClassifierCV(est, method=method, ensemble=ensemble, cv=3, n_jobs=jobs).fit(X, y)
            return _predicts(model, Xt, tag) + _members(model, tag)
        if row[0] == "nested_vote":
            model = ml.VotingClassifier([("inner", _soft_vote(ml)), ("lr", ml.LogisticRegression())],
                                        voting="soft").fit(X, y)
            return _predicts(model, Xt, tag) + _members(model, tag)
        if row[0] == "stack_of_vote":
            model = ml.StackingClassifier([("vote", _soft_vote(ml)), ("lda", ml.LinearDiscriminantAnalysis())],
                                          cv=3).fit(X, y)
            return _predicts(model, Xt, tag) + _members(model, tag)
        if row[0] == "shap":
            model = ml.StackingClassifier([("vote", _soft_vote(ml)), ("lda", ml.LinearDiscriminantAnalysis())],
                                          cv=3).fit(X, y)
            explainer = (ml.KernelExplainer if row[1] == "K" else ml.PermutationExplainer)(model, X[:20])
            phi = np.asarray(explainer.shap_values(Xt[:5]))
            return [(tag + "_values", _digest(phi)),
                    (tag + "_expected", _digest(np.asarray(explainer.expected_value, dtype=np.float64)))]
        if row[0] == "iterimp":
            Xr, _ = _data(11, 120, 0, d=5)
            Xr = Xr.copy()
            rng = np.random.default_rng(3)
            Xr[rng.random(Xr.shape) < 0.15] = np.nan
            if row[1] == "vote":
                est = ml.VotingRegressor([("ridge", ml.Ridge(alpha=1.0)), ("lin", ml.LinearRegression())])
            else:
                est = ml.StackingRegressor([("ridge", ml.Ridge(alpha=1.0)), ("lin", ml.LinearRegression())], cv=3,
                                           passthrough=True)
            imp = ml.IterativeImputer(estimator=est, max_iter=3, random_state=0).fit(Xr)
            return [(tag + "_transform", _digest(imp.transform(Xr)))]
        if row[0] == "search":
            from mojolearn.model_selection import GridSearchCV
            model = GridSearchCV(_soft_vote(ml), {"weights": [[1, 1], [1, 2]]}, cv=3).fit(X, y)
            return [(tag + "_best", str(model.best_params_))] + _predicts(model, Xt, tag)
    except Exception as e:  # noqa: BLE001 - the words are the digest
        return [(tag + "_raised", f"{type(e).__name__}: {e}")]
    return []


def main():
    with ProcessPoolExecutor(max_workers=min(len(CONFIGS), os.cpu_count() or 1)) as pool:
        for lines in pool.map(_run, range(len(CONFIGS))):
            for name, value in lines:
                print(name, value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
