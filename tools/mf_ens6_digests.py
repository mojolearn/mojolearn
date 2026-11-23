"""Before/after digests of the member routes and native refusals lane mf-ens6 added (2026-11-23) in the Select,
Sequence, Prep and Ensemble (Stacking / Calibrated / RFE kind) families, host column. One line per configuration
and output: name and sha256 of the output bytes (dtype and shape included); a refusal is digested by its words.
Base: origin/lane/member-fits-6 2f3076c639 (each setting below made the member admission answer False and the
class's own door ran it), head lane/mf-ens6 (the member runs it, or the native refusal raises). Test tool, never
product runtime.

Every configuration is fitted through GridSearchCV with one given split (train = test = every row), a one-point
grid on the estimator's first parameter by name, refit=True: the refit fits the estimator as a member. Scoring:
'r2' for a regressor, 'accuracy' for a classifier, none for a transformer. The `direct_*` units call the public
fit itself (the door; its refusal words changed where scikit-learn fails).

Rows covered (mf-cal5-work/fallback-audit.tsv, see mf-ens6-work/rows.tsv): select 87, 93; sequence 55; prep 187,
254, 347, 351; ensemble 575 (_NESTED_META_MAX), 799, 838, 868, 871, 886, 890, 894, 700, 1644, 3706; calibrated
k=2 ensemble=False (LabelPropagation, -1 marker).

EXPECTED: `*_scores`, attribute and output lines equal (the member runs the door's programs), except, stated:
`kbest_mi*` and `kbest_negk` and `rnn_steplr` may differ only in a `_raised` line becoming values on the head
side (the member now serves the setting; the base search kept the door, so most lines are equal);
`direct_quantile_sub0`, `direct_kbins_sub_float`, `direct_kbins_state_float`: base fitted (no scikit-learn
check), head one `_raised` line with scikit-learn's InvalidParameterError words (a behaviour change, stated);
`direct_kbins_rng_draw`: base `TypeError: int() argument ... not 'RandomState'`, head the same words plus "; pass
an integer random_state"; `cal_lp_binary`: base the door units' result or words, head scikit-learn's IndexError
words; `stack_kfold_one`, `stack_timeseries_unread`: the split door's words on both sides (head raises them from
the member). Configurations are independent: one process each across the host cores.

    python tools/mf_ens6_digests.py > digests.txt
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


def _data(n=48, d=4, seed=5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = (X[:, 0] - 0.5 * X[:, 1]).astype(np.float32)
    return X, y


def _pairs(n, k=3):
    rows = np.arange(n)
    return [(rows[rows % k != f], rows[rows % k == f]) for f in range(k)]


def _config(name):
    """(estimator, X, y, scoring, attribute names, output method) of configuration `name`."""
    import mojolearn
    from mojolearn import model_selection as ms
    X, y = _data()
    labels = (y > 0).astype(np.int64)
    three = np.digitize(y, [-0.5, 0.5]).astype(np.int64)
    n = len(X)
    if name == "kbest_mi_classif":
        return (mojolearn.SelectKBest(mojolearn.mutual_info_classif, k=2), X, three, None, ["scores_", "pvalues_"],
                "transform")
    if name == "kbest_mi_regression":
        return (mojolearn.SelectKBest(mojolearn.mutual_info_regression, k=2), X, y, None, ["scores_", "pvalues_"],
                "transform")
    if name == "kbest_negk":
        return (mojolearn.SelectKBest(mojolearn.f_classif, k=-1), X, labels, None, ["scores_"], "transform")
    if name == "rnn_steplr":
        x3 = np.arange(48, dtype=np.float32).reshape(8, 3, 2) / 48
        return (mojolearn.RNNRegressor(hidden_size=4, max_epochs=2, batch_size=4, random_state=7,
                                       lr_schedule=mojolearn.StepLR(0.01, step_size=2, gamma=0.5),
                                       numeric_mode="identical"), x3, x3[:, -1, 0].copy(), "r2",
                ["params_", "loss_curve_"], "predict")
    if name == "maxabs":
        return (mojolearn.MaxAbsScaler(), X, None, None, ["max_abs_", "scale_"], "transform")
    if name == "stack_listed":
        return (mojolearn.StackingRegressor([("r", mojolearn.Ridge())], cv=_pairs(n)), X, y, "r2", [], "predict")
    if name == "stack_kfold":
        return (mojolearn.StackingRegressor([("r", mojolearn.Ridge())], cv=ms.KFold(n_splits=3)), X, y, "r2", [],
                "predict")
    if name == "stack_timeseries":
        return (mojolearn.StackingRegressor([("r", mojolearn.Ridge())], cv=ms.TimeSeriesSplit(n_splits=3)), X, y,
                "r2", [], "predict")
    if name == "stack_kfold_one":
        kf = ms.KFold(n_splits=3)
        kf.n_splits = 1
        return (mojolearn.StackingRegressor([("r", mojolearn.Ridge())], cv=kf), X, y, "r2", [], "predict")
    if name == "stack_timeseries_unread":
        ts = ms.TimeSeriesSplit(n_splits=3)
        ts.max_train_size = "x"
        return (mojolearn.StackingRegressor([("r", mojolearn.Ridge())], cv=ts), X, y, "r2", [], "predict")
    if name == "stack_pair_bad":
        rows = np.arange(n)
        return (mojolearn.StackingRegressor([("r", mojolearn.Ridge())], cv=[(rows[:30], rows[30:], rows[:3])]), X,
                y, "r2", [], "predict")
    if name == "cal_stratified":
        return (mojolearn.CalibratedClassifierCV(mojolearn.LogisticRegression(), cv=ms.StratifiedKFold(n_splits=3)),
                X, labels, "accuracy", [], "predict_proba")
    if name == "cal_lp_binary":
        marked = np.where(np.arange(n) % 2 == 0, 1, -1)
        return (mojolearn.CalibratedClassifierCV(mojolearn.LabelPropagation(), ensemble=False, cv=2), X, marked,
                "accuracy", [], "predict_proba")
    if name == "rfe_callable_getter":
        return (mojolearn.RFE(mojolearn.LinearRegression(), importance_getter=lambda e: e.coef_), X, y, None,
                ["support_", "ranking_"], "transform")
    raise ValueError(name)


def _direct(name):
    import mojolearn
    X, _ = _data(n=20, d=2)
    if name == "direct_quantile_sub0":
        return mojolearn.QuantileTransformer(n_quantiles=5, subsample=0).fit(X).quantiles_
    if name == "direct_kbins_sub_float":
        return mojolearn.KBinsDiscretizer(n_bins=3, subsample=2.5).fit(X).bin_edges_[0]
    if name == "direct_kbins_state_float":
        return mojolearn.KBinsDiscretizer(n_bins=3, subsample=10, random_state=1.5).fit(X).bin_edges_[0]
    if name == "direct_kbins_rng_nodraw":
        return mojolearn.KBinsDiscretizer(n_bins=3, subsample=100,
                                          random_state=np.random.RandomState(0)).fit(X).bin_edges_[0]
    if name == "direct_kbins_rng_draw":
        return mojolearn.KBinsDiscretizer(n_bins=3, subsample=10,
                                          random_state=np.random.RandomState(0)).fit(X).bin_edges_[0]
    raise ValueError(name)


def _unit(name):
    out = []
    if name.startswith("direct_"):
        try:
            out.append((name, _digest(_direct(name))))
        except Exception as e:  # noqa: BLE001 - the words are the digest
            out.append((name + "_raised", f"{type(e).__name__}: {e}"))
        return out
    from mojolearn.model_selection import GridSearchCV
    try:
        est, X, y, scoring, attrs, method = _config(name)
        every = np.arange(len(X))
        first = sorted(est.get_params())[0]
        search = GridSearchCV(est, {first: [est.get_params()[first]]}, cv=[(every, every)], scoring=scoring,
                              refit=True)
        search.fit(X, y)
        out.append((name + "_scores", _digest(np.asarray(search.cv_results_["mean_test_score"], dtype=np.float64))))
        best = search.best_estimator_
        for attr in attrs:
            value = getattr(best, attr, None)
            out.append((f"{name}_{attr}", "None" if value is None else _digest(value)))
        try:
            out.append((f"{name}_{method}", _digest(getattr(best, method)(X[:8]))))
        except Exception as e:  # noqa: BLE001 - the words are the digest
            out.append((f"{name}_{method}_raised", f"{type(e).__name__}: {e}"))
    except Exception as e:  # noqa: BLE001 - the words are the digest
        out.append((name + "_raised", f"{type(e).__name__}: {e}"))
    return out


UNITS = ["kbest_mi_classif", "kbest_mi_regression", "kbest_negk", "rnn_steplr", "maxabs", "stack_listed",
         "stack_kfold", "stack_timeseries", "stack_kfold_one", "stack_timeseries_unread", "stack_pair_bad",
         "cal_stratified", "cal_lp_binary", "rfe_callable_getter", "direct_quantile_sub0", "direct_kbins_sub_float",
         "direct_kbins_state_float", "direct_kbins_rng_nodraw", "direct_kbins_rng_draw"]


def main():
    with ProcessPoolExecutor(max_workers=min(len(UNITS), os.cpu_count() or 1)) as pool:
        for lines in pool.map(_unit, UNITS):
            for name, value in lines:
                print(name, value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
