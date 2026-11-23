"""Before/after digests for lane mf-imp6 (2026-11-23): gradient boosting `feature_importances_`, and RFE over a
gbdt reading it. One line per configuration, outputs AND fitted state as sha256 of raw bytes (a refusal by its
words). Written, not run; the orchestrator runs it on the CPU column (and on the L40S for the device column) for
base origin/lane/member-fits-6 ca3b0cfb8c and for the head of lane/mf-imp6, then diffs the two outputs.

EXPECTED (bits change: new outputs only):
  * `*_predict`, `*_model`: EQUAL before and after for every fit (the importance pass reads the fitted model and
    the fit's rows; it changes no fit and no prediction).
  * `*_imp`: before, AttributeError words ("no attribute 'feature_importances_'"); after, a float64 vector digest,
    plus `*_imp_sum` (rounded to 6 places: 1.0, or 0.0 for the constant target) and `*_imp_argmax`.
  * `not_fitted_*`: before AttributeError words; after scikit-learn's NotFittedError words.
  * `rfe_gbdt_*` (mojolearn RFE over GradientBoosting / the adapters, rounds that read importances): before, the
    refusal ("RFE needs an estimator with coef_ or feature_importances_"); after, support / ranking / predict
    digests. `rfe_gbdt_coef` (importance_getter='coef_'): the same attrgetter words both times.
    `rfe_gbdt_all` (n_features_to_select keeps every column, no round): EQUAL.
  * `sk_*` (scikit-learn's RFE, RFECV and SelectFromModel over our estimators, when scikit-learn is installed):
    before, their own AttributeError / ValueError words; after, support digests.
  * `saved_*`: GradientBoosting.load and HostGBDT.from_file of a saved fit: after, the same `*_imp` digest as the
    fit (`saved_*_same` True).
  * The device column (L40S, the GPU gbdt image) must print the same `*_imp` digests as the CPU column after
    (the importance bodies are shared; a difference is a defect).

Configurations are independent: one process each across the host cores.

    python tools/mf_imp6_digests.py > digests.txt
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


def _text(value):
    return hashlib.sha256(str(value).encode()).hexdigest()[:24]


def _data(seed=17, n=240, d=6):
    rng = np.random.default_rng(seed)
    X = rng.random((n, d), dtype=np.float32)
    s = (3.0 * X[:, 1] + X[:, 4] - 0.5 * X[:, 0]).astype(np.float32)
    t = (s + 0.05 * rng.standard_normal(n)).astype(np.float32)
    codes = np.digitize(s, np.quantile(s, [0.33, 0.66])).astype(np.float32)
    binary = (s > np.median(s)).astype(np.float32)
    w = np.array([(1.0, 2.5, 0.5, 0.75)[i % 4] for i in range(n)], dtype=np.float32)
    return X, t, codes, binary, w


def _raised(tag, error):
    return [(tag + "_raised", f"{type(error).__name__}: {error}")]


def _imp_lines(tag, model):
    try:
        imp = model.feature_importances_
    except Exception as error:  # noqa: BLE001 (the words are the digest)
        return _raised(tag + "_imp", error)
    a = np.asarray(imp)
    return [(tag + "_imp", _digest(a)), (tag + "_imp_type", type(imp).__name__ + "/" + a.dtype.str),
            (tag + "_imp_sum", f"{float(a.sum()):.6f}"), (tag + "_imp_argmax", str(int(np.argmax(a))))]


def _fit_unit(arg):
    _, loss, policy, weighted = arg
    import mojolearn as ml
    X, t, codes, binary, w = _data()
    y = {"RMSE": t, "Logloss": binary, "MultiClass": codes, "MultiRMSE": np.stack([t, X[:, 2]], 1).astype(np.float32),
         "Constant": np.full(len(t), 1.5, np.float32)}[loss]
    tag = f"gb_{loss}_{policy}_{'w' if weighted else 'u'}"
    kw = dict(n_estimators=10, max_depth=4, grow_policy=policy)
    if policy == "Lossguide":
        kw["max_leaves"] = 12
    try:
        m = ml.GradientBoosting(loss="RMSE" if loss == "Constant" else loss, **kw)
        m.fit(X, y, sample_weight=w if weighted else None)
    except Exception as error:  # noqa: BLE001
        return _raised(tag, error)
    return [(tag + "_predict", _digest(m.predict(X))), (tag + "_model", _text(m.model_))] + _imp_lines(tag, m)


def _other_unit(arg):
    which = arg[1]
    import mojolearn as ml
    X, t, codes, binary, w = _data()
    out = []
    try:
        if which == "adapter_reg":
            m = ml.GradientBoostingRegressor(n_estimators=10).fit(X, t)
            out += [(which + "_predict", _digest(m.predict(X)))] + _imp_lines(which, m)
        elif which == "adapter_clf":
            m = ml.GradientBoostingClassifier(n_estimators=10).fit(X, binary.astype(np.int64))
            out += [(which + "_proba", _digest(m.predict_proba(X)))] + _imp_lines(which, m)
        elif which == "cat":
            Xc = X.copy()
            Xc[:, 3] = np.floor(Xc[:, 3] * 5)
            m = ml.GradientBoosting(n_estimators=10, cat_features=[3]).fit(Xc, t)
            out += [(which + "_predict", _digest(m.predict(Xc)))] + _imp_lines(which, m)
        elif which == "ordered":
            perm = np.random.default_rng(5).permutation(len(t)).astype(np.int64)
            m = ml.OrderedRMSE(n_estimators=10).fit(X, t, permutation=perm)
            out += [(which + "_predict", _digest(m.predict(X)))] + _imp_lines(which, m)
        elif which == "feature_freq":
            Xf = X.copy()
            Xf[:, 0] = np.floor(Xf[:, 0] * 4)
            Xf[:, 5] = np.floor(Xf[:, 5] * 3)
            m = ml.ExperimentalTwoLevelFeatureFreq(sources=[0, 5]).fit(Xf, t)
            out += [(which + "_predict", _digest(m.predict(Xf)))] + _imp_lines(which, m)
        elif which.startswith("not_fitted_"):
            cls = {"not_fitted_gb": ml.GradientBoosting, "not_fitted_reg": ml.GradientBoostingRegressor,
                   "not_fitted_clf": ml.GradientBoostingClassifier}[which]
            out += _imp_lines(which, cls())
        elif which == "saved":
            import tempfile
            m = ml.GradientBoosting(n_estimators=10).fit(X, t)
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "gb.npz")
                m.save(path)
                own = _imp_lines("saved_fit", m)
                for name, loaded in (("saved_load", ml.GradientBoosting.load(path)),
                                     ("saved_host", ml.HostGBDT.from_file(path))):
                    lines = _imp_lines(name, loaded)
                    out += lines + [(name + "_same", str(lines[0][1] == own[0][1]))]
                out += own
    except Exception as error:  # noqa: BLE001
        out += _raised(which, error)
    return out


def _rfe_unit(arg):
    which = arg[1]
    import mojolearn as ml
    from mojolearn.feature_selection import RFE
    X, t, codes, binary, w = _data()
    tag = "rfe_gbdt_" + which
    try:
        if which == "auto":
            r = RFE(ml.GradientBoosting(n_estimators=6), n_features_to_select=2).fit(X, t)
        elif which == "named":
            r = RFE(ml.GradientBoostingRegressor(n_estimators=6), n_features_to_select=3,
                    importance_getter="feature_importances_").fit(X, t)
        elif which == "clf_step2":
            r = RFE(ml.GradientBoostingClassifier(n_estimators=6), n_features_to_select=2, step=2).fit(
                X, binary.astype(np.int64))
        elif which == "weighted":
            r = RFE(ml.GradientBoosting(n_estimators=6), n_features_to_select=2).fit(X, t, sample_weight=w)
        elif which == "coef":
            r = RFE(ml.GradientBoosting(n_estimators=6), n_features_to_select=2, importance_getter="coef_").fit(X, t)
        else:  # "all": no round
            r = RFE(ml.GradientBoosting(n_estimators=6), n_features_to_select=X.shape[1]).fit(X, t)
    except Exception as error:  # noqa: BLE001
        return _raised(tag, error)
    return [(tag + "_support", _digest(np.asarray(r.support_, dtype=bool))), (tag + "_ranking", _digest(r.ranking_)),
            (tag + "_predict", _digest(r.predict(X)))]


def _sk_unit(arg):
    which = arg[1]
    try:
        from sklearn import feature_selection as fs
    except ImportError:
        return [("sk_" + which, "scikit-learn not installed")]
    import mojolearn as ml
    X, t, codes, binary, w = _data()
    try:
        if which == "rfe":
            s = fs.RFE(ml.GradientBoostingRegressor(n_estimators=6), n_features_to_select=2).fit(X, t)
        elif which == "rfecv":
            s = fs.RFECV(ml.GradientBoostingRegressor(n_estimators=6), cv=3, min_features_to_select=2).fit(X, t)
        else:
            s = fs.SelectFromModel(ml.GradientBoostingRegressor(n_estimators=6), max_features=2,
                                   threshold=-np.inf).fit(X, t)
    except Exception as error:  # noqa: BLE001
        return _raised("sk_" + which, error)
    return [("sk_" + which + "_support", _digest(np.asarray(s.get_support(), dtype=bool)))]


def _unit(arg):
    if arg[0] == "fit":
        return _fit_unit(arg)
    if arg[0] == "rfe":
        return _rfe_unit(arg)
    if arg[0] == "sk":
        return _sk_unit(arg)
    return _other_unit(arg)


UNITS = ([("fit", loss, policy, weighted) for loss in ("RMSE", "Logloss", "MultiClass", "MultiRMSE")
          for policy in ("SymmetricTree", "Depthwise", "Lossguide") for weighted in (False, True)]
         + [("fit", "Constant", "SymmetricTree", False)]
         + [("other", c) for c in ("adapter_reg", "adapter_clf", "cat", "ordered", "feature_freq", "not_fitted_gb",
                                   "not_fitted_reg", "not_fitted_clf", "saved")]
         + [("rfe", c) for c in ("auto", "named", "clf_step2", "weighted", "coef", "all")]
         + [("sk", c) for c in ("rfe", "rfecv", "select_from_model")])


def main():
    with ProcessPoolExecutor(max_workers=min(len(UNITS), os.cpu_count() or 1)) as pool:
        for lines in pool.map(_unit, UNITS):
            for name, value in lines:
                print(name, value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
