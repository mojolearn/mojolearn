"""Before/after digests of the RFE rows lane mf-rfe6 moved off the estimator's own doors (2026-11-23), host column
(and the device image when run there). One line per configuration and output: name and sha256 of the output bytes
(dtype and shape included); a refusal is digested by its exception type and words. Base: lane/member-fits-6
d50f480b2f (each configuration below took `_fit_clone`'s rounds or the member-raise fallback), head lane/mf-rfe6.
Test tool, never product runtime.

Rows (see ~/mojolearn-evidence/briefs-20261122/mf-rfe6-work/rows.tsv): a forest leaf's feature_importances_ under
a search step and a stack's final step; DeviceRows folds (cross_val_score over RFE); the weighted forest regressor
(RFE with sample_weight, and the public fit with feature_importances=True); refused targets (NaN y for Ridge and a
Gbdt regressor, short labels, a forest regressor's NaN y).

EXPECTED (stated per unit): `leaf_search_*` / `leaf_stack_*`: base raised (the door read no feature_importances_ on
the refit leaf, which fits without them: a TypeError or ValueError from the importance read), head support,
ranking and predict outputs (a behaviour change: served); `rows_cv_*`: equal scores base vs head (the same programs;
on a device image the head gathers the fold's rows inside the RUN); `rf_weighted_rfe`: base raised
RFE_NO_IMPORTANCES ("mojolearn: RFE needs an estimator with coef_ or feature_importances_"), head support and
ranking (served); `rf_weighted_public_imp`: base NotImplementedError words, head the importances (new output);
`nan_ridge` / `nan_gbdt`: equal words (base the door's re-fit raised them, head the member path raises them);
`short_labels`: base the estimator's own door's length words, head scikit-learn's "[64, 63]" words (a words change
for the families whose door words differ); `nan_forest_reg`: equal (both fit: the forest door takes a non-finite y
under squared_error, and the member path now admits y as that door does). Configurations are independent: one
process each across the host cores.

    python tools/mf_rfe6_digests.py > digests.txt
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


def _data(n=64, d=6, seed=5):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d)).astype(np.float32)
    y = (x[:, 0] + 0.6 * x[:, 2] - 0.3 * x[:, 4] > 0).astype(np.int64)
    t = (2.0 * x[:, 0] - x[:, 2] + 0.25 * x[:, 4] ** 2).astype(np.float32)
    return x, y, t


def _fitted(m, x):
    return [("support", np.asarray(m.support_)), ("ranking", np.asarray(m.ranking_)), ("predict", m.predict(x))]


def _run(name):
    """The configuration's outputs as (suffix, value) pairs."""
    import mojolearn
    from mojolearn import model_selection as ms
    from mojolearn.model_selection import GridSearchCV
    x, y, t = _data()
    every = np.arange(len(x))
    rf = lambda cls: cls(n_estimators=4, max_depth=4, random_state=0)  # noqa: E731
    if name == "leaf_search_clf":
        search = GridSearchCV(rf(mojolearn.RandomForestClassifier), {"max_depth": [4]}, cv=[(every, every)])
        return _fitted(mojolearn.RFE(search, n_features_to_select=3,
                                     importance_getter="best_estimator_.feature_importances_").fit(x, y), x)
    if name == "leaf_search_reg":
        search = GridSearchCV(rf(mojolearn.RandomForestRegressor), {"max_depth": [3, 4]}, cv=3)
        return _fitted(mojolearn.RFE(search, n_features_to_select=3,
                                     importance_getter="best_estimator_.feature_importances_").fit(x, t), x)
    if name == "leaf_stack_reg":
        stack = mojolearn.StackingRegressor([("ridge", mojolearn.Ridge(alpha=1.0))],
                                            final_estimator=rf(mojolearn.RandomForestRegressor), cv=3,
                                            passthrough=True)
        return _fitted(mojolearn.RFE(stack, n_features_to_select=3,
                                     importance_getter="final_estimator_.feature_importances_").fit(x, t), x)
    if name.startswith("rows_cv_"):
        make = (lambda: mojolearn.LogisticRegression(max_iter=50)) if name == "rows_cv_logistic" else \
            (lambda: mojolearn.LinearDiscriminantAnalysis())
        registry = ms._NATIVE_CONTEXT["family_msel_estimators"]
        plan = registry.get(mojolearn.RFE)
        before = None if plan is None else plan.get("device_rows", False)
        try:
            if plan is not None:
                plan["device_rows"] = True
            return [("scores", ms.cross_val_score(mojolearn.RFE(make(), n_features_to_select=3), x, y, cv=4))]
        finally:
            if plan is not None:
                plan["device_rows"] = before
    w = np.linspace(0.5, 2.0, len(x)).astype(np.float32)
    if name == "rf_weighted_rfe":
        return _fitted(mojolearn.RFE(rf(mojolearn.RandomForestRegressor), n_features_to_select=3).fit(
            x, t, sample_weight=w), x)
    if name == "rf_weighted_public_imp":
        est = rf(mojolearn.RandomForestRegressor).fit(x, t, sample_weight=w, feature_importances=True)
        return [("imp", np.asarray(est.feature_importances_)), ("predict", est.predict(x))]
    bad = t.copy()
    bad[3] = np.nan
    if name == "nan_ridge":
        return _fitted(mojolearn.RFE(mojolearn.Ridge(alpha=1.0), n_features_to_select=3).fit(x, bad), x)
    if name == "nan_gbdt":
        gb = mojolearn.GradientBoostingRegressor(n_estimators=3, max_depth=2, random_state=0)
        return _fitted(mojolearn.RFE(gb, n_features_to_select=3).fit(x, bad), x)
    if name == "nan_forest_reg":
        return _fitted(mojolearn.RFE(rf(mojolearn.RandomForestRegressor), n_features_to_select=3).fit(x, bad), x)
    if name == "short_labels":
        return _fitted(mojolearn.RFE(mojolearn.LogisticRegression(max_iter=50), n_features_to_select=3).fit(
            x, y[:-1]), x)
    raise ValueError(name)


def _unit(name):
    out = []
    try:
        for suffix, value in _run(name):
            out.append((f"{name}_{suffix}", _digest(value)))
    except Exception as e:  # noqa: BLE001 - the words are the digest
        out.append((name + "_raised", f"{type(e).__name__}: {e}"))
    return out


UNITS = ["leaf_search_clf", "leaf_search_reg", "leaf_stack_reg", "rows_cv_logistic", "rows_cv_lda",
         "rf_weighted_rfe", "rf_weighted_public_imp", "nan_ridge", "nan_gbdt", "nan_forest_reg", "short_labels"]


def main():
    with ProcessPoolExecutor(max_workers=min(len(UNITS), os.cpu_count() or 1)) as pool:
        for lines in pool.map(_unit, UNITS):
            for name, value in lines:
                print(name, value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
