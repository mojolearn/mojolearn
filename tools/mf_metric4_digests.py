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

Lane mf-score5 adds section score5 (base: origin/lane/member-fits-5 80dc3165e0 vs head): (C) the doors whose
programs moved into typed builders (average_precision binary / ovr macro / weighted / None / micro, brier binary
and 3-class, top_k k=1 / k=2 / k=3 / weighted, NMI / AMI / contingency / pair confusion): every line equal.
(D) searches scored by 'average_precision', 'neg_brier_score', 'top_k_accuracy', 'normalized_mutual_info_score'
and 'adjusted_mutual_info_score' (now through the RUN, same programs: equal), with the WARNINGS each raised (the
category and words of every warning caught during the fit, in order): a precision_macro / balanced_accuracy
search whose folds leave a class unpredicted, an r2 search over 1-row test folds (index list), a binary
top_k search. Expected: (D) value lines equal; the warning lines equal in category and words (the base raised
them from the door's per-unit scorer, the head from the PACK after each unit's fit notices), the order within a
unit (test side, then train side, the scorers' order) equal.

Lane mf-search5 adds section search5 (base: origin/lane/member-fits-5 0ed9b30aa2 vs head): (E) searches over the
classes whose folds the old door handed as DeviceRows (GaussianNB, MultinomialNB, BernoulliNB,
LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis), now through the typed search RUN: scoring None (the
class's mean-accuracy `score`, now the 'accuracy' scorer), 'accuracy', 'neg_log_loss', 'balanced_accuracy',
return_train_score, an index-list cv, and the search's own `device` (None, 'cpu', and a candidate grid naming
`device`). Host column: every line equal (no member keeps X resident on the CPU; the same fits, the same
programs). On the L40S (GPU owed) the same command: every line equal to the host column's except where a line of
(B) already says the binary64 program differs; the resident route moves no bit (the same device take of the same
rows). A scoring None line that differs only in the last bits of a mean is the 'accuracy' program's float64
fraction against the class's own score program; say so if seen.

Lane mf-cluster5 adds section cluster5 (base: origin/lane/member-fits-5 9f274ad9d5 vs head): (F) the public
label-pair doors now on the cluster table's typed RUN (rand, adjusted rand, mutual information, homogeneity,
completeness, v-measure beta 1 / 2, Fowlkes-Mallows, homogeneity_completeness_v_measure, raw labels with gaps and
negatives, an empty FM, a length mismatch and a contingency= refusal): every line equal (the same image bodies,
the same union codes). (G) searches now through the typed RUN: each of the seven label-pair scorers over a
classifier; GridSearchCV(KMeans) scored by each clustering scorer and a multi-metric of them (a clusterer base);
make_scorer(mean_squared_error) (sign flipped), make_scorer(f1_score, average='macro') (the 'f1_macro' row); a
binary f1 search whose labels are {0, 2} (pos_label=1 absent: raise, and error_score=nan: its warning) and
{'a', 'b'} (the type refusal). Expected: every value line equal; a raised line equal in its words (the base's door
units raised the door's metric words per fold; the head's RUN raises the same words). The KMeans lines' base ran
the door units (the public scorer over predict); equal values are expected because the RUN reads the same labels
(float32 target words of small integers) and the same entry bodies.

Lane mf-msel5 adds section msel5 (base: origin/lane/member-fits-5 89a4f6be5d vs head): (H) the cv tools now on the
typed search RUN (cross_validate with one and several scorers, train scores, return_estimator (each fold
estimator's predict) and return_indices, error_score numeric; cross_val_score; learning_curve (with shuffle);
cross_val_predict predict / predict_proba / decision_function for a classifier and predict for a regressor;
groups with a GroupKFold); GridSearchCV.cv_results_ split<i>_train_score columns; validation_curve; an
average_precision search over labels {0, 2} (raise, and error_score=nan). Expected: every value line equal, except
the times (not digested) and these named differences: GridSearchCV with return_train_score now also has the
split<i>_train_score columns (the base had none: H_search_train_cols differs); validation_curve through the kind
(the base raised KeyError on the missing columns where the kind took it: H_validation_curve may differ from a
raised base line); H_ap_absent_* raise scikit-learn's words "pos_label=1 is not a valid label: It should be one
of [0 2]" (the base's door scorer raised its own words; compare the words).
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


def _score5_metric_cases():
    from mojolearn import metrics as m
    rng = np.random.default_rng(9)
    ct = rng.integers(0, 3, size=90)
    sc = rng.dirichlet(np.ones(3), size=90).astype(np.float32)
    bt = (rng.uniform(size=90) < 0.4).astype(np.int64)
    bs = rng.uniform(size=90).astype(np.float32)
    w = rng.uniform(0.2, 2.0, size=90).astype(np.float32)
    lt = rng.integers(0, 4, size=90)
    lp = np.where(rng.uniform(size=90) < 0.6, lt, rng.integers(0, 5, size=90))
    cases = [("C_ap_bin", lambda: m.average_precision_score(bt, bs)),
             ("C_ap_bin_w", lambda: m.average_precision_score(bt, bs, sample_weight=w)),
             ("C_ap_bin_pos0", lambda: m.average_precision_score(bt, bs, pos_label=0))]
    for avg in ("macro", "weighted", "micro", None):
        cases.append((f"C_ap_ovr_{avg}", lambda avg=avg: m.average_precision_score(ct, sc, average=avg)))
    cases.append(("C_ap_ovr_w", lambda: m.average_precision_score(ct, sc, sample_weight=w)))
    cases.append(("C_brier_bin", lambda: m.brier_score_loss(bt, bs)))
    cases.append(("C_brier_bin_w", lambda: m.brier_score_loss(bt, bs, sample_weight=w)))
    cases.append(("C_brier_3", lambda: m.brier_score_loss(ct, sc)))
    for k in (1, 2, 3):
        cases.append((f"C_topk_{k}", lambda k=k: m.top_k_accuracy_score(ct, sc, k=k)))
    cases.append(("C_topk_bin_k1", lambda: m.top_k_accuracy_score(bt, bs, k=1)))
    cases.append(("C_topk_w_count", lambda: m.top_k_accuracy_score(ct, sc, k=2, sample_weight=w, normalize=False)))
    cases.append(("C_nmi", lambda: m.normalized_mutual_info_score(lt, lp)))
    cases.append(("C_nmi_geometric", lambda: m.normalized_mutual_info_score(lt, lp, average_method="geometric")))
    cases.append(("C_ami", lambda: m.adjusted_mutual_info_score(lt, lp)))
    cases.append(("C_contingency", lambda: m.cluster.contingency_matrix(lt, lp)
                  if hasattr(m, "cluster") else m.contingency_matrix(lt, lp)))
    cases.append(("C_pair_confusion", lambda: m.cluster.pair_confusion_matrix(lt, lp)
                  if hasattr(m, "cluster") else m.pair_confusion_matrix(lt, lp)))
    return cases


def _warned(thunk):
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rows = thunk()
    for i, item in enumerate(caught):
        rows.append((f"warning{i}", f"{item.category.__name__}: {item.message}"))
    return rows


def _search_rows(est, grid, scoring, X, target, cv=3, train=False):
    from mojolearn.model_selection import GridSearchCV
    s = GridSearchCV(est, grid, scoring=scoring, cv=cv, return_train_score=train).fit(X, target)
    rows = []
    for key in sorted(s.cv_results_):
        if "time" in key or key == "params" or key.startswith("param_"):
            continue
        rows.append((key, _digest(np.asarray(s.cv_results_[key], dtype=np.float64))))
    rows.append(("best_index", repr(int(s.best_index_))))
    rows.append(("best_predict", _digest(np.asarray(s.best_estimator_.predict(X)))))
    return rows


def _score5_search_cases():
    import mojolearn
    cases = []
    for name in ("average_precision", "neg_brier_score", "top_k_accuracy", "normalized_mutual_info_score",
                 "adjusted_mutual_info_score"):
        cases.append((f"D_clf_{name}", lambda name=name: _warned(lambda: _search("clf", name))))
        cases.append((f"D_bin_{name}", lambda name=name: _warned(lambda: _search("bin", name))))
    cases.append(("D_clf_score5_multi", lambda: _warned(lambda: _search(
        "clf", ["average_precision", "neg_brier_score", "top_k_accuracy", "adjusted_mutual_info_score"],
        refit="neg_brier_score", train=True))))

    def _weak():
        # a weak model on a 4-class target: some folds leave a class unpredicted (ill-defined precision, the
        # balanced accuracy's classes in y_pred not in y_true do not arise: y_pred is a subset)
        X, y, _ = _data(13, n=120, d=2, k=4)
        c = np.searchsorted(np.quantile(y, [0.05, 0.1, 0.5]), y)
        return _search_rows(mojolearn.LogisticRegression(max_iter=50), {"C": [1e-4, 1.0]},
                            ["precision_macro", "balanced_accuracy", "recall_macro", "f1_macro"], X, c, train=True)
    cases.append(("D_clf_ill_defined", lambda: _warned(_weak)))

    def _one_row():
        X, y, _ = _data(17, n=40)
        n = len(y)
        cv = [(np.arange(1, n), np.array([0])), (np.arange(0, n - 1), np.array([n - 1]))]
        return _search_rows(mojolearn.Ridge(), {"alpha": [0.1, 1.0]}, ["r2", "d2_absolute_error_score"], X, y, cv=cv)
    cases.append(("D_reg_one_row_r2", lambda: _warned(_one_row)))
    return cases


def _search5_rows(name, scoring, cv=3, train=False, device="unset", grid_device=False, refit=True):
    import mojolearn
    from mojolearn.model_selection import GridSearchCV
    X, y, c = _data(19, n=180, d=5, k=3)
    if name in ("MultinomialNB", "BernoulliNB"):
        X = np.abs(X).astype(np.float32)
    est = getattr(mojolearn, name)()
    grid = {"GaussianNB": {"var_smoothing": [1e-9, 1e-3]}, "MultinomialNB": {"alpha": [0.5, 1.0]},
            "BernoulliNB": {"alpha": [0.5, 1.0]}, "LinearDiscriminantAnalysis": {"tol": [1e-4, 1e-2]},
            "QuadraticDiscriminantAnalysis": {"reg_param": [0.0, 0.1]}}[name]
    if grid_device:
        grid = dict(grid, device=[None])
    if cv == "list":
        n = len(c)
        cv = [(np.arange(0, n - 60), np.arange(n - 60, n)), (np.arange(60, n), np.arange(0, 60))]
    kwargs = {} if device == "unset" else {"device": device}
    s = GridSearchCV(est, grid, scoring=scoring, cv=cv, return_train_score=train, refit=refit, **kwargs).fit(X, c)
    rows = []
    for key in sorted(s.cv_results_):
        if "time" in key or key == "params" or key.startswith("param_"):
            continue
        rows.append((key, _digest(np.asarray(s.cv_results_[key], dtype=np.float64))))
    rows.append(("best_index", repr(int(s.best_index_))))
    rows.append(("best_predict", _digest(np.asarray(s.best_estimator_.predict(X)))))
    if hasattr(s.best_estimator_, "predict_proba"):
        rows.append(("best_proba", _digest(np.asarray(s.best_estimator_.predict_proba(X)))))
    return rows


def _search5_cases():
    cases = []
    for name in ("GaussianNB", "MultinomialNB", "BernoulliNB", "LinearDiscriminantAnalysis",
                 "QuadraticDiscriminantAnalysis"):
        for scoring in (None, "accuracy", "neg_log_loss", "balanced_accuracy"):
            cases.append((f"E_{name}_{scoring}", lambda name=name, scoring=scoring: _search5_rows(name, scoring)))
    cases.append(("E_GaussianNB_train", lambda: _search5_rows("GaussianNB", "accuracy", train=True)))
    cases.append(("E_GaussianNB_list", lambda: _search5_rows("GaussianNB", None, cv="list")))
    cases.append(("E_GaussianNB_device_none", lambda: _search5_rows("GaussianNB", "accuracy", device=None)))
    cases.append(("E_GaussianNB_device_cpu", lambda: _search5_rows("GaussianNB", "accuracy", device="cpu")))
    cases.append(("E_GaussianNB_grid_device", lambda: _search5_rows("GaussianNB", "accuracy", grid_device=True)))
    cases.append(("E_LDA_multi", lambda: _search5_rows("LinearDiscriminantAnalysis",
                                                       ["accuracy", "neg_log_loss"], train=True,
                                                       refit="neg_log_loss")))
    return cases


_PAIR_SCORERS = ["rand_score", "adjusted_rand_score", "mutual_info_score", "homogeneity_score", "completeness_score",
                 "v_measure_score", "fowlkes_mallows_score"]


def _cluster5_metric_cases():
    from mojolearn import metrics as m
    rng = np.random.default_rng(23)
    lt = rng.integers(0, 5, size=150).astype(np.int32)
    lp = np.where(rng.uniform(size=150) < 0.6, lt, rng.integers(0, 6, size=150)).astype(np.int32)
    gt = (lt * 3 - 4).astype(np.int32)  # gaps and negatives: the raw labels the doors see
    cases = []
    for name in ("rand_score", "adjusted_rand_score", "mutual_info_score", "homogeneity_score",
                 "completeness_score", "v_measure_score", "fowlkes_mallows_score"):
        fn = getattr(m, name)
        cases.append((f"F_{name}", lambda fn=fn: fn(lt, lp)))
        cases.append((f"F_{name}_raw", lambda fn=fn: fn(gt, lp)))
        cases.append((f"F_{name}_mismatch", lambda fn=fn: fn(lt, lp[:-1])))
    cases.append(("F_v_measure_beta2", lambda: m.v_measure_score(lt, lp, beta=2.0)))
    cases.append(("F_hcv", lambda: m.homogeneity_completeness_v_measure(lt, lp)))
    cases.append(("F_hcv_beta05", lambda: m.homogeneity_completeness_v_measure(gt, lp, beta=0.5)))
    cases.append(("F_fm_empty", lambda: m.fowlkes_mallows_score(lt[:0], lp[:0])))
    cases.append(("F_mi_contingency", lambda: m.mutual_info_score(lt, lp, contingency=np.ones((2, 2)))))
    return cases


def _rows_of_search(s, X):
    rows = []
    for key in sorted(s.cv_results_):
        if "time" in key or key == "params" or key.startswith("param_"):
            continue
        rows.append((key, _digest(np.asarray(s.cv_results_[key], dtype=np.float64))))
    if hasattr(s, "best_index_"):
        rows.append(("best_index", repr(int(s.best_index_))))
    if hasattr(s, "best_estimator_"):
        rows.append(("best_predict", _digest(np.asarray(s.best_estimator_.predict(X)))))
    return rows


def _kmeans_search(scoring, refit=True):
    import mojolearn
    from mojolearn.model_selection import GridSearchCV
    X, _, c = _data(29, n=150, d=4, k=3)
    s = GridSearchCV(mojolearn.KMeans(random_state=0, n_init=1), {"n_clusters": [2, 3, 4]}, scoring=scoring,
                     cv=3, refit=refit).fit(X, c)
    return _rows_of_search(s, X)


def _binary_labels_search(labels, error_score="raise"):
    import mojolearn
    from mojolearn.model_selection import GridSearchCV
    X, _, c = _data(31, n=120, d=4, k=2)
    target = np.asarray(labels)[(c > 0).astype(np.int64)]
    s = GridSearchCV(mojolearn.LogisticRegression(max_iter=100), {"C": [0.1, 1.0]}, scoring="f1", cv=3,
                     error_score=error_score).fit(X, target)
    return _rows_of_search(s, X)


def _cluster5_search_cases():
    from mojolearn import metrics as m
    from mojolearn.model_selection import make_scorer
    cases = []
    for name in _PAIR_SCORERS:
        cases.append((f"G_clf_{name}", lambda name=name: _warned(lambda: _search("clf", name))))
        cases.append((f"G_kmeans_{name}", lambda name=name: _warned(lambda: _kmeans_search(name))))
    for name in ("normalized_mutual_info_score", "adjusted_mutual_info_score"):
        cases.append((f"G_kmeans_{name}", lambda name=name: _warned(lambda: _kmeans_search(name))))
    cases.append(("G_kmeans_multi", lambda: _kmeans_search(["adjusted_rand_score", "v_measure_score",
                                                           "normalized_mutual_info_score"],
                                                          refit="v_measure_score")))
    cases.append(("G_kmeans_refit_false", lambda: _kmeans_search("adjusted_rand_score", refit=False)))
    cases.append(("G_make_scorer_mse_flip", lambda: _search("reg", make_scorer(m.mean_squared_error))))
    cases.append(("G_make_scorer_neg_mse", lambda: _search("reg", make_scorer(m.mean_squared_error,
                                                                            greater_is_better=False))))
    cases.append(("G_make_scorer_f1_macro", lambda: _search("clf", make_scorer(m.f1_score, average="macro"))))
    cases.append(("G_bin_absent_raise", lambda: _binary_labels_search([0, 2])))
    cases.append(("G_bin_absent_nan", lambda: _warned(lambda: _binary_labels_search([0, 2], error_score=np.nan))))
    cases.append(("G_bin_strings", lambda: _binary_labels_search(["a", "b"])))
    return cases


def _cv_validate(kind, scoring, train=False, estimator=False, indices=False, error_score=np.nan, groups=False):
    import mojolearn
    from mojolearn.model_selection import cross_validate, GroupKFold
    X, y, c = _data(37, n=180, d=5, k=3)
    est = mojolearn.LogisticRegression(max_iter=100) if kind == "clf" else mojolearn.Ridge(alpha=1.0)
    target = c if kind == "clf" else y
    cv = 3
    g = None
    if groups:
        g = np.arange(180) % 6
        cv = GroupKFold(n_splits=3)
    res = cross_validate(est, X, target, groups=g, scoring=scoring, cv=cv, return_train_score=train,
                         return_estimator=estimator, return_indices=indices, error_score=error_score)
    out = []
    for key in sorted(res):
        if key in ("fit_time", "score_time"):
            out.append((key + "_len", len(res[key])))
        elif key == "estimator":
            for i, e in enumerate(res[key]):
                out.append((f"estimator{i}_predict", _digest(np.asarray(e.predict(X), dtype=np.float64))))
        elif key == "indices":
            for side in ("train", "test"):
                for i, rows in enumerate(res[key][side]):
                    out.append((f"indices_{side}{i}", _digest(np.asarray(rows, dtype=np.int64))))
        else:
            out.append((key, _digest(np.asarray(res[key], dtype=np.float64))))
    return out


def _cv_predict(kind, method):
    import mojolearn
    from mojolearn.model_selection import cross_val_predict
    X, y, c = _data(41, n=180, d=5, k=3)
    est = mojolearn.LogisticRegression(max_iter=100) if kind == "clf" else mojolearn.Ridge(alpha=1.0)
    pred = cross_val_predict(est, X, c if kind == "clf" else y, cv=3, method=method)
    a = np.asarray(pred)
    return [("dtype", a.dtype.str), ("values", _digest(a))]


def _learning(shuffle):
    import mojolearn
    from mojolearn.model_selection import learning_curve
    X, _, c = _data(43, n=180, d=5, k=3)
    got = learning_curve(mojolearn.LogisticRegression(max_iter=100), X, c, train_sizes=[0.3, 0.6, 1.0], cv=3,
                         scoring="accuracy", shuffle=shuffle, random_state=0, return_times=True)
    return [("sizes", _digest(np.asarray(got[0], dtype=np.int64))), ("train", _digest(got[1])),
            ("test", _digest(got[2])), ("times_shape", f"{np.asarray(got[3]).shape}{np.asarray(got[4]).shape}")]


def _search_train_cols():
    import mojolearn
    from mojolearn.model_selection import GridSearchCV
    X, _, c = _data(47, n=150, d=4, k=3)
    s = GridSearchCV(mojolearn.LogisticRegression(max_iter=100), {"C": [0.1, 1.0]}, scoring="accuracy", cv=3,
                     return_train_score=True).fit(X, c)
    return [(k, _digest(np.asarray(s.cv_results_[k], dtype=np.float64))) for k in sorted(s.cv_results_)
            if k.startswith("split") or k.endswith("_test_score") or k == "mean_train_score"]


def _validation():
    import mojolearn
    from mojolearn.model_selection import validation_curve
    X, _, c = _data(53, n=150, d=4, k=3)
    tr, te = validation_curve(mojolearn.LogisticRegression(max_iter=100), X, c, param_name="C",
                              param_range=[0.1, 1.0, 10.0], cv=3, scoring="accuracy")
    return (tr, te)


def _ap_absent(error_score):
    import mojolearn
    from mojolearn.model_selection import GridSearchCV
    X, _, c = _data(59, n=120, d=4, k=2)
    target = np.asarray([0, 2])[(c > 0).astype(np.int64)]
    s = GridSearchCV(mojolearn.LogisticRegression(max_iter=100), {"C": [0.1, 1.0]}, scoring="average_precision",
                     cv=3, error_score=error_score).fit(X, target)
    return _rows_of_search(s, X)


def _msel5_cases():
    from mojolearn.model_selection import cross_val_score
    import mojolearn
    cases = [
        ("H_cv_clf_none", lambda: _cv_validate("clf", None)),
        ("H_cv_clf_acc_train", lambda: _cv_validate("clf", "accuracy", train=True)),
        ("H_cv_clf_multi", lambda: _cv_validate("clf", ["accuracy", "f1_macro", "neg_log_loss"], train=True)),
        ("H_cv_clf_estimator_indices", lambda: _cv_validate("clf", "accuracy", estimator=True, indices=True)),
        ("H_cv_reg_r2_estimator", lambda: _cv_validate("reg", "r2", estimator=True)),
        ("H_cv_reg_groups", lambda: _cv_validate("reg", "neg_mean_squared_error", groups=True, indices=True)),
        ("H_cv_score_reg", lambda: cross_val_score(mojolearn.Ridge(alpha=1.0), *_data(61)[:2], cv=4, scoring="r2")),
        ("H_cvp_clf_predict", lambda: _cv_predict("clf", "predict")),
        ("H_cvp_clf_proba", lambda: _cv_predict("clf", "predict_proba")),
        ("H_cvp_clf_decision", lambda: _cv_predict("clf", "decision_function")),
        ("H_cvp_reg_predict", lambda: _cv_predict("reg", "predict")),
        ("H_learning_plain", lambda: _learning(False)),
        ("H_learning_shuffle", lambda: _learning(True)),
        ("H_search_train_cols", _search_train_cols),
        ("H_validation_curve", _validation),
        ("H_ap_absent_raise", lambda: _ap_absent("raise")),
        ("H_ap_absent_nan", lambda: _warned(lambda: _ap_absent(np.nan))),
    ]
    return cases


def _all_cases():
    return (_metric_cases() + _search_cases() + _score5_metric_cases() + _score5_search_cases()
            + _search5_cases() + _cluster5_metric_cases() + _cluster5_search_cases() + _msel5_cases())


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
