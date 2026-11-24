# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# THE HOST-ONLY PUBLIC FUNCTIONS, one-sided (sub-lane vdv-cov-func, 2026-11-24).
#
# host_model / host_predict / host_predict_proba (the CPU-inference entry
# points), ensemble.catboost_auto_learning_rate and johnson_lindenstrauss_min_dim
# take no `device` on lane vd-functions (the first three and catboost by
# decision: host only; johnson_lindenstrauss_min_dim is a closed-form integer
# bound). Each lane is declared one-sided with the missing side named
# (`_one_sided`, registry ONE_SIDED of the harness); the missing GPU side is
# listed in docs/plans/verify_through_device/UNMAPPED.md. This fragment needs
# the harness's `_one_sided` (sub-lane vdv-harness) at load.


@lane("fn-host-predict")
def _(ml, X, yc, yr, Xh=None):
    """host_model(path).predict / predict_proba and the one-call host_predict /
    host_predict_proba on the file a 4-tree forest (fitted through ml on the
    side) saves: 512 training rows, 128 held-out rows. The one-call entries
    are held byte for byte to host_model's (the documented claim)."""
    Xr = _functions_mat(X, 512, 8, "fn-host:x")
    y = _functions_classes(X, 512, 3, "fn-host:y")
    held = _functions_mat(X[512:], 128, 8, "fn-host:h")
    rf = ml.RandomForestClassifier(n_estimators=4, max_depth=5, random_state=7).fit(Xr, y)
    with tempfile.TemporaryDirectory(prefix="ib-fn-host-") as d:
        path = os.path.join(d, "rf.npz")
        rf.save(path)
        m = ml.host_model(path)
        pred, proba = np.asarray(m.predict(held)), np.asarray(m.predict_proba(held))
        one = np.asarray(ml.host_predict(path, held))
        one_proba = np.asarray(ml.host_predict_proba(path, held))
    _same_bytes("host_predict", one, "host_model(path).predict", pred)
    _same_bytes("host_predict_proba", one_proba, "host_model(path).predict_proba", proba)
    return _fit(dict(predict=_functions_d(pred), proba=_functions_d(proba)))


@lane("fn-catboost-auto-learning-rate")
def _(ml, X, yc, yr, Xh=None):
    """ensemble.catboost_auto_learning_rate for three losses and options, row counts from the
    fixture's shape (no data read: a closed-form rate from a table)."""
    n = int(X.shape[0])
    rates = [ml.ensemble.catboost_auto_learning_rate("Logloss", n, 1000, False, False),
             ml.ensemble.catboost_auto_learning_rate("RMSE", 10 * n, 1000, False, True),
             ml.ensemble.catboost_auto_learning_rate("Logloss", n, 500, True, False),
             ml.ensemble.catboost_auto_learning_rate("MAE", n, 1000, False, True)]
    return _fit(dict(rates=_functions_d(rates)))


@lane("fn-jl-min-dim")
def _(ml, X, yc, yr, Xh=None):
    """johnson_lindenstrauss_min_dim at the fixture's row count and three eps."""
    n = int(X.shape[0])
    dims = [ml.johnson_lindenstrauss_min_dim(n, eps=eps) for eps in (0.1, 0.25, 0.5)]  # glue: three eps
    return _fit(dict(dims=_functions_d(dims)))


_one_sided("fn-host-predict", missing="gpu",
           reason="host_model / host_predict / host_predict_proba are the CPU-inference entry points (host only by decision)")
_one_sided("fn-catboost-auto-learning-rate", missing="gpu",
           reason="catboost_auto_learning_rate is a host table lookup that takes no device (host only by decision)")
_one_sided("fn-jl-min-dim", missing="gpu",
           reason="johnson_lindenstrauss_min_dim takes no device on vd-functions (closed-form integer bound)")
_batch_decl(_FUNCTIONS_BATCH_NA, "fn-host-predict", "fn-catboost-auto-learning-rate", "fn-jl-min-dim")
