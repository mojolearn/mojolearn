"""Before/after digests of the RFE cases lane mf-rfe7 moved off the estimator's own doors (2026-11-23; `_fit_clone`
and the door's rounds are deleted), host column (and the device image when run there). One line per configuration
and output: name and sha256 of the output bytes (dtype and shape included); a refusal is digested by its exception
type and words. Base: lane/member-fits-7 1f029628db (each configuration below took the door's rounds), head
lane/mf-rfe7. Test tool, never product runtime.

Rows: ~/mojolearn-evidence/briefs-20261122/mf-rfe7-work/rows.tsv.

EXPECTED (stated per unit):
  `svc_linear_bin`, `svc_linear_ovo`, `svr_linear`: EQUAL support, ranking and predict (the rounds read the
    kernel SVM member's coef_ block: the SVM coef_ operation's GEMV rows and sign word, which the door read through
    `prep_estimator_attribute`; the fits are the same SVM RUN).
  `nb_theta`, `mnb_log_prob`, `lda_means`: EQUAL (the named block is the array PACK writes as that attribute; the
    door admitted the attribute as float32 rows, the same words).
  `meta_kept`: EQUAL (no round; the final fit of the search: base through its fit door, head its member).
  `meta_round`, `et_auto`, `svc_rbf`, `nb_unknown`: EQUAL words where the door raised the same; the head raises them
    before the first round's fit instead of after it (no fit notice of that first round is emitted).
  `bag_step`: base the door's attribute walk's words, head "... is not one fitted estimator RFE can follow
    natively ..." (a words change).
  `rf_oob_weights`: EQUAL words ("NotImplementedError: sample_weight is not implemented together with oob_score or
    a tree-ID shard"): base the forest regressor's fit door, head RFE's UNPACK.
Configurations are independent: one process each across the host cores.

    python tools/mf_rfe7_digests.py > digests.txt
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


def _data(n=72, d=7, seed=11, classes=2):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d)).astype(np.float32)
    score = x[:, 0] + 0.7 * x[:, 2] - 0.4 * x[:, 5]
    if classes == 2:
        y = (score > 0).astype(np.int64)
    else:
        y = np.digitize(score, np.quantile(score, [1 / 3, 2 / 3])).astype(np.int64)
    t = (2.0 * x[:, 0] - x[:, 3] + 0.3 * x[:, 6]).astype(np.float32)
    return x, y, t


def _fitted(m, x):
    return [("support", np.asarray(m.support_)), ("ranking", np.asarray(m.ranking_)), ("predict", m.predict(x))]


def _run(name):
    """The configuration's outputs as (suffix, value) pairs."""
    import mojolearn
    from mojolearn.model_selection import GridSearchCV
    x, y, t = _data()
    _, y3, _ = _data(classes=3)
    every = np.arange(len(x))
    if name == "svc_linear_bin":
        return _fitted(mojolearn.RFE(mojolearn.SVC(kernel="linear"), n_features_to_select=3).fit(x, y), x)
    if name == "svc_linear_ovo":
        return _fitted(mojolearn.RFE(mojolearn.SVC(kernel="linear"), n_features_to_select=3).fit(x, y3), x)
    if name == "svr_linear":
        return _fitted(mojolearn.RFE(mojolearn.SVR(kernel="linear"), n_features_to_select=2,
                                     importance_getter="coef_").fit(x, t), x)
    if name == "nb_theta":
        return _fitted(mojolearn.RFE(mojolearn.GaussianNB(), n_features_to_select=3,
                                     importance_getter="theta_").fit(x, y3), x)
    if name == "mnb_log_prob":
        ax = np.abs(x)
        return _fitted(mojolearn.RFE(mojolearn.MultinomialNB(), n_features_to_select=3,
                                     importance_getter="feature_log_prob_").fit(ax, y3), ax)
    if name == "lda_means":
        return _fitted(mojolearn.RFE(mojolearn.LinearDiscriminantAnalysis(), n_features_to_select=3,
                                     importance_getter="means_").fit(x, y3), x)
    search = GridSearchCV(mojolearn.LinearDiscriminantAnalysis(), {"tol": [1e-4]}, cv=[(every, every)])
    if name == "meta_kept":
        return _fitted(mojolearn.RFE(search, n_features_to_select=x.shape[1]).fit(x, y), x)
    if name == "meta_round":
        return _fitted(mojolearn.RFE(search, n_features_to_select=3).fit(x, y), x)
    if name == "et_auto":
        return _fitted(mojolearn.RFE(mojolearn.ExtraTreesRegressor(n_estimators=3, random_state=0),
                                     n_features_to_select=3).fit(x, t), x)
    if name == "svc_rbf":
        return _fitted(mojolearn.RFE(mojolearn.SVC(kernel="rbf"), n_features_to_select=3).fit(x, y), x)
    if name == "nb_unknown":
        return _fitted(mojolearn.RFE(mojolearn.GaussianNB(), n_features_to_select=3,
                                     importance_getter="sigma_").fit(x, y), x)
    if name == "bag_step":
        bag = mojolearn.BaggingClassifier(mojolearn.LinearDiscriminantAnalysis(), n_estimators=2, random_state=0)
        return _fitted(mojolearn.RFE(bag, n_features_to_select=3, importance_getter="estimators_.coef_").fit(x, y), x)
    if name == "rf_oob_weights":
        rf = mojolearn.RandomForestRegressor(n_estimators=4, oob_score=True, random_state=0)
        w = np.linspace(0.5, 2.0, len(x)).astype(np.float32)
        return _fitted(mojolearn.RFE(rf, n_features_to_select=3).fit(x, t, sample_weight=w), x)
    raise ValueError(name)


def _unit(name):
    out = []
    try:
        for suffix, value in _run(name):
            out.append((f"{name}_{suffix}", _digest(value)))
    except Exception as e:  # noqa: BLE001 - the words are the digest
        out.append((name + "_raised", f"{type(e).__name__}: {e}"))
    return out


UNITS = ["svc_linear_bin", "svc_linear_ovo", "svr_linear", "nb_theta", "mnb_log_prob", "lda_means", "meta_kept",
         "meta_round", "et_auto", "svc_rbf", "nb_unknown", "bag_step", "rf_oob_weights"]


def main():
    with ProcessPoolExecutor(max_workers=min(len(UNITS), os.cpu_count() or 1)) as pool:
        for lines in pool.map(_unit, UNITS):
            for name, value in lines:
                print(name, value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
