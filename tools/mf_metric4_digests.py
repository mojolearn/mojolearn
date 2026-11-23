"""Before/after digests of lane mf-metric4 (2026-11-23): the metric doors whose program assembly moved into typed
builders, and GridSearchCV / RandomizedSearchCV scored by every scorer name the typed search RUN now serves.
Host column. One line per configuration: name, then sha256 of the output bytes (dtype and shape included) or a
refusal's words. Test tool, never product runtime. Configurations run in parallel, one worker per allocated core.

    python tools/mf_metric4_digests.py > digests.txt      # base vs head, then diff

Expected: section (A) every line equal (the same programs). Section (B): the search lines of scorers whose old
door path called a default float32 entry (r2, neg_mean_squared_error, neg_mean_absolute_error,
neg_root_mean_squared_error, precision/recall/f1 averages, roc_auc binary) may differ in the last bits of
cv_results_ (the RUN reads the binary64 program value); every other line equal.

Lane mf-search4b extends (B) with the cases the search kind now takes (base d1ac69da3a vs head): the 'binary'
label scorers (f1, precision, recall, jaccard on a 0/1 target), our KFold / StratifiedKFold splitters and an
index list as cv, multi-metric with refit=False, and a multi-metric dict holding a binary scorer. Expected: the
binary scorers' lines may differ in the last bits (the base door read the float32 precision_recall_fscore entry;
the RUN reads the binary64 program); every other line equal, including the fitted-state lines (multimetric_,
whether best_estimator_ exists, its predictions).
"""
import hashlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np


def _digest(value):
    a = np.asarray(value)
    h = hashlib.sha256()
    h.update(f"{a.dtype.str}{a.shape}".encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:24]


def _data(seed, n=211, d=6, k=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = (X @ rng.normal(size=d) + 0.3 * rng.normal(size=n)).astype(np.float32)
    edges = np.quantile(y, np.linspace(0, 1, k + 1)[1:-1])
    return X, y, np.searchsorted(edges, y)


def _metric_cases():
    from mojolearn import metrics as m
    rng = np.random.default_rng(5)
    yt = rng.normal(size=80).astype(np.float32) + 3.0
    yp = (yt + 0.4 * rng.normal(size=80)).astype(np.float32)
    Y = np.stack([yt, yt * 0.5 + 1.0], 1).astype(np.float32)
    P = np.stack([yp, yp * 0.4 + 1.2], 1).astype(np.float32)
    w = rng.uniform(0.2, 2.0, size=80).astype(np.float32)
    ct = rng.integers(0, 4, size=80)
    cp = np.where(rng.uniform(size=80) < 0.7, ct, rng.integers(0, 4, size=80))
    pr = rng.dirichlet(np.ones(4), size=80).astype(np.float32)
    bt = (ct > 1).astype(np.int64)
    bs = rng.uniform(size=80).astype(np.float32)
    cases = []
    for name, fn in (("mse", m.mean_squared_error), ("mae", m.mean_absolute_error),
                     ("msle", m.mean_squared_log_error), ("mape", m.mean_absolute_percentage_error),
                     ("medae", m.median_absolute_error), ("ev", m.explained_variance_score), ("r2", m.r2_score),
                     ("d2abs", m.d2_absolute_error_score)):
        cases.append((f"A_{name}_plain", lambda fn=fn: fn(yt, yp)))
        cases.append((f"A_{name}_w", lambda fn=fn: fn(yt, yp, sample_weight=w)))
        cases.append((f"A_{name}_raw", lambda fn=fn: fn(Y, P, multioutput="raw_values")))
    cases.append(("A_max_error", lambda: m.max_error(yt, yp)))
    cases.append(("A_poisson", lambda: m.mean_poisson_deviance(yt, np.abs(yp) + 0.1)))
    cases.append(("A_gamma", lambda: m.mean_gamma_deviance(yt, np.abs(yp) + 0.1)))
    cases.append(("A_tweedie15", lambda: m.mean_tweedie_deviance(yt, np.abs(yp) + 0.1, power=1.5)))
    cases.append(("A_d2tweedie", lambda: m.d2_tweedie_score(yt, np.abs(yp) + 0.1, power=1.5)))
    cases.append(("A_r2_ff", lambda: m.r2_score(yt, yp, force_finite=False)))
    for avg in ("micro", "macro", "weighted", None):
        cases.append((f"A_f1_{avg}_w", lambda avg=avg: m.f1_score(ct, cp, average=avg, sample_weight=w)))
        cases.append((f"A_prfs_{avg}", lambda avg=avg: m.precision_recall_fscore_support(ct, cp, average=avg)))
        cases.append((f"A_fbeta2_{avg}", lambda avg=avg: m.fbeta_score(ct, cp, beta=2.0, average=avg)))
        cases.append((f"A_jaccard_{avg}", lambda avg=avg: m.jaccard_score(ct, cp, average=avg)))
    cases.append(("A_balanced", lambda: m.balanced_accuracy_score(ct, cp)))
    cases.append(("A_balanced_adj_w", lambda: m.balanced_accuracy_score(ct, cp, sample_weight=w, adjusted=True)))
    cases.append(("A_matthews", lambda: m.matthews_corrcoef(ct, cp)))
    cases.append(("A_matthews_w", lambda: m.matthews_corrcoef(ct, cp, sample_weight=w)))
    cases.append(("A_accuracy_count", lambda: m.accuracy_score(ct, cp, normalize=False)))
    cases.append(("A_log_loss", lambda: m.log_loss(ct, pr)))
    cases.append(("A_log_loss_w", lambda: m.log_loss(ct, pr, sample_weight=w)))
    cases.append(("A_log_loss_bin", lambda: m.log_loss(bt, bs)))
    cases.append(("A_roc_bin_w", lambda: m.roc_auc_score(bt, bs, sample_weight=w)))
    cases.append(("A_roc_bin_fpr", lambda: m.roc_auc_score(bt, bs, max_fpr=0.4)))
    for mc in ("ovr", "ovo"):
        for avg in ("macro", "weighted"):
            cases.append((f"A_roc_{mc}_{avg}", lambda mc=mc, avg=avg: m.roc_auc_score(ct, pr, multi_class=mc,
                                                                                  average=avg)))
    cases.append(("A_roc_ovr_none", lambda: m.roc_auc_score(ct, pr, multi_class="ovr", average=None)))
    cases.append(("A_roc_ovr_w", lambda: m.roc_auc_score(ct, pr, multi_class="ovr", sample_weight=w)))
    return cases


_CLF_SCORERS = ["accuracy", "balanced_accuracy", "matthews_corrcoef", "precision_micro", "precision_macro",
                "precision_weighted", "recall_micro", "recall_macro", "recall_weighted", "f1_micro", "f1_macro",
                "f1_weighted", "jaccard_micro", "jaccard_macro", "jaccard_weighted", "neg_log_loss", "roc_auc_ovr",
                "roc_auc_ovo", "roc_auc_ovr_weighted", "roc_auc_ovo_weighted"]
_BIN_SCORERS = ["roc_auc", "neg_log_loss", "f1", "average_precision"]
_REG_SCORERS = ["r2", "neg_mean_squared_error", "neg_mean_absolute_error", "neg_root_mean_squared_error",
                "neg_mean_absolute_percentage_error", "neg_median_absolute_error", "explained_variance", "max_error",
                "d2_absolute_error_score"]


def _search(kind, scoring, refit=True, train=False, cv=3):
    import mojolearn
    from mojolearn.model_selection import GridSearchCV
    X, y, c = _data(11)
    if kind == "reg":
        est, grid, target = mojolearn.Ridge(), {"alpha": [0.1, 1.0, 10.0]}, y
    elif kind == "bin":
        est, grid, target = mojolearn.LogisticRegression(max_iter=200), {"C": [0.1, 1.0]}, (c > 0).astype(np.int64)
    else:
        est, grid, target = mojolearn.LogisticRegression(max_iter=200), {"C": [0.1, 1.0, 10.0]}, c
    if cv == "kfold":
        from mojolearn.model_selection import KFold
        cv = KFold(n_splits=4, shuffle=True, random_state=3)
    elif cv == "stratified":
        from mojolearn.model_selection import StratifiedKFold
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=5)
    elif cv == "list":
        n = len(target)
        cv = [(np.arange(0, n - 50), np.arange(n - 50, n)), (np.arange(50, n), np.arange(0, 50))]
    s = GridSearchCV(est, grid, scoring=scoring, cv=cv, refit=refit, return_train_score=train).fit(X, target)
    rows = []
    for key in sorted(s.cv_results_):
        if "time" in key or key == "params" or key.startswith("param_"):
            continue
        rows.append((key, _digest(np.asarray(s.cv_results_[key], dtype=np.float64))))
    rows.append(("multimetric", repr(bool(s.multimetric_))))
    rows.append(("n_splits", repr(int(s.n_splits_))))
    if hasattr(s, "best_index_"):
        rows.append(("best_index", repr(int(s.best_index_))))
        rows.append(("best_score", repr(float(s.best_score_))))
    rows.append(("has_best_estimator", repr(hasattr(s, "best_estimator_"))))
    if hasattr(s, "best_estimator_"):
        rows.append(("best_predict", _digest(np.asarray(s.best_estimator_.predict(X)))))
    return rows


def _search_cases():
    cases = []
    for name in _CLF_SCORERS:
        cases.append((f"B_clf_{name}", lambda name=name: _search("clf", name)))
    for name in _BIN_SCORERS:
        cases.append((f"B_bin_{name}", lambda name=name: _search("bin", name)))
    for name in _REG_SCORERS:
        cases.append((f"B_reg_{name}", lambda name=name: _search("reg", name)))
    cases.append(("B_clf_f1_macro_train", lambda: _search("clf", "f1_macro", train=True)))
    cases.append(("B_clf_multi", lambda: _search("clf", ["accuracy", "balanced_accuracy", "neg_log_loss"],
                                                 refit="balanced_accuracy")))
    cases.append(("B_clf_refit_false", lambda: _search("clf", "f1_macro", refit=False)))
    cases.append(("B_reg_multi", lambda: _search("reg", {"r2": "r2", "mae": "neg_mean_absolute_error"},
                                                 refit=False)))
    # lane mf-search4b
    for name in ("f1", "precision", "recall", "jaccard"):
        cases.append((f"B_bin_{name}_binary", lambda name=name: _search("bin", name)))
    cases.append(("B_clf_kfold", lambda: _search("clf", "accuracy", cv="kfold")))
    cases.append(("B_clf_stratified", lambda: _search("clf", None, cv="stratified")))
    cases.append(("B_clf_index_list", lambda: _search("clf", "accuracy", cv="list", train=True)))
    cases.append(("B_reg_kfold", lambda: _search("reg", "r2", cv="kfold")))
    cases.append(("B_clf_multi_refit_false", lambda: _search("clf", ["accuracy", "f1_macro"], refit=False)))
    cases.append(("B_bin_multi_f1", lambda: _search("bin", {"f": "f1", "acc": "accuracy"}, refit="f", train=True)))
    cases.append(("B_clf_refit_name_single", lambda: _search("clf", "accuracy", refit="accuracy")))
    return cases


def _all_cases():
    return _metric_cases() + _search_cases()


def _run(index):
    name, thunk = _all_cases()[index]
    try:
        value = thunk()
        if isinstance(value, list):
            return [f"{name}_{k} {v}" for k, v in value]
        if isinstance(value, tuple):
            return [f"{name}_{i} {_digest(np.asarray(v, dtype=np.float64))}" for i, v in enumerate(value)
                    if v is not None]
        return [f"{name} {_digest(np.asarray(value, dtype=np.float64))}"]
    except Exception as e:  # noqa: BLE001 - the words are the digest
        return [f"{name}_raised {type(e).__name__}: {e}"]


def main():
    count = len(_all_cases())
    cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=max(1, min(cores, count))) as pool:
        for lines in pool.map(_run, range(count)):
            for line in lines:
                print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
