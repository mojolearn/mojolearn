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

Lane mf-msel5b extends section msel5 (H; base: origin/lane/member-fits-5 6f0a3de6ed vs head): cross_val_predict
into the one result buffer (the H_cvp_* lines above, now with no per-fold copy), predict_log_proba, a fold that
lacks a class (KFold over sorted labels: predict_proba placed with zero columns, decision_function's refusal, a
'accuracy' search over such folds), a multi-output regressor's predict, y=None (MiniBatchKMeans cross_validate and
a KMeans cross_val_predict), a search with groups and GroupKFold, mixed string/integer labels scored by 'f1', and
cross_val_score over a default cv (the deleted slice route's case). Expected: every value line equal, except
H5b_cvp_log_proba (the log is now taken in the member's own precision: last bits may change), H5b_mixed_f1_*
(the RUN raises the metric doors' TypeError words; compare the words), and H5b_lacking_decision (scikit-learn's
"Only 2 class/es in training fold" words; compare the words).

Lane mf-kmeans5 extends section msel5 (H; base: origin/lane/member-fits-5 41899d4e82 vs head): KMeans.score
(unweighted and weighted, on the training rows and on new rows, beside -inertia_), a KMeans search and a
cross_validate with scoring None (now through the RUN: the member's score), a fold that lacks a class scored by
the member's own score and by 'neg_log_loss' / 'roc_auc_ovr' (raise and error_score=nan). Expected: the base has
no KMeans.score, so H5k_kmeans_score_* and H5k_kmeans_*_none are new lines (each score line must equal its
minus_inertia line on the training rows); H5k_lacking_own_* (the base's door units scored a lacking fold through
the fold estimator's own score, which is accuracy: values equal); H5k_lacking_log_loss_raise (scikit-learn's
"y_true and y_pred contain different number of classes" words; compare the words); the _nan lines equal.

Lane mf-msel5c extends section msel5 (H; base: origin/lane/member-fits-5 b3ca4be7a9 vs head): searches over every
splitter object the door now splits and hands to the search kind (ShuffleSplit, StratifiedShuffleSplit,
RepeatedKFold, RepeatedStratifiedKFold, TimeSeriesSplit, GroupShuffleSplit, LeaveOneGroupOut with groups, an
index list), make_scorer kwargs (fbeta_score beta=2 keeps the door; f1 pos_label=2, an explicit average='binary',
top_k_accuracy k=1 and k=3 are now typed rows), error_score='bogus' (scikit-learn's InvalidParameterError
words), a lacking fold scored by 'average_precision' and 'neg_brier_score' (raise and nan), a 2-D classifier
target (GridSearchCV over MultiOutputClassifier, scoring None, and cross_val_predict predict, integer and real
labels), permutation_test_score, and a classifier search by 'f1' / 'roc_auc' (its admission now has the labels).
Expected: value lines equal, except the scorer lines may differ in the last bits where the base scored through
the deleted member units' Python metric calls (the RUN reads the binary64 program, as mf-metric4 noted); the
error_score line changes words (was "error_score must be 'raise' or a number"); the lacking AP / brier raise
lines carry scikit-learn's words (the base: the metric door's); the multi-output lines are new values from the
kind (the base fitted them through the door: values equal, the dtype the same).

Lane mf-tail5 extends section msel5 (H; base: origin/lane/member-fits-5 2fe17e40a5 vs head): THE GIVEN y (a
GridSearchCV over RidgeCV(cv=StratifiedKFold(2)) and a StackingRegressor / VotingRegressor / BaggingRegressor
over it, float64 targets with pairs equal in float32: the member path now splits on the given values, as the
door), a multi-output classifier whose folds lack a class in one output (GridSearchCV scoring None,
cross_val_predict predict and predict_proba, the list of per-output arrays), cross_val_predict(predict) of a
lacking single-output fold (no RuntimeWarning now), permutation_test_score, learning_curve with a list of
scorers (scikit-learn's InvalidParameterError words before any fit). Expected: the given-y lines change where
the base merged float32-equal targets into one class (now the door's folds; the base raised or split
differently), the multi-output lacking lines are new values (the base fitted every output on all classes'
codes), the single-output predict line loses its warning, the learning_curve line changes words (was
"learning_curve takes one scorer" after the fits); the others equal.
Lane mf-tail5b extends section H (base: origin/lane/member-fits-5 b1109feb43 vs head): H5tb_kwargs_* (make_scorer
fbeta / labels arrays / zero_division / pos_label on AP and brier / absent, other-kind and mixed pos_label words,
now the search kind's scorer word groups), H5tb_text_multi_* (a text 2-D y), H5tb_refuse_* (learning_curve's
InvalidParameterError class, KFold n_splits=1, error_score=None, no candidates). Expected: value and words lines
equal (the base scored these through the door's Python units with the same metric programs; fbeta / labels may
move in the last bits where the base door's float32 entry differs). Expected DIFFERENT: H5tb_text_multi_predict
(base raised converting a label to float: the door bug fixed), H5tb_refuse_learning_many class (base ValueError,
now InvalidParameterError, both bases True) and its name order (the set's), H5tb_refuse_error_score_none on bases
where the door's words differed.
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
                "neg_mean_absolute_percentage_error", "neg_median_absolute_error", "explained_variance", "neg_max_error",
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


def _lacking_predict(method):
    import mojolearn
    from mojolearn.model_selection import cross_val_predict, KFold
    X, _, c = _data(67, n=150, d=4, k=3)
    order = np.argsort(c, kind="stable")
    pred = cross_val_predict(mojolearn.LogisticRegression(max_iter=100), X[order], c[order], cv=KFold(n_splits=3),
                             method=method)
    a = np.asarray(pred)
    return [("dtype", a.dtype.str), ("values", _digest(a))]


def _lacking_search():
    import mojolearn
    from mojolearn.model_selection import GridSearchCV, KFold
    X, _, c = _data(71, n=150, d=4, k=3)
    order = np.argsort(c, kind="stable")
    s = GridSearchCV(mojolearn.LogisticRegression(max_iter=100), {"C": [0.1, 1.0]}, scoring="accuracy",
                     cv=KFold(n_splits=3)).fit(X[order], c[order])
    return _rows_of_search(s, X)


def _multi_output_predict():
    import mojolearn
    from mojolearn.model_selection import cross_val_predict
    X, y, _ = _data(73, n=150, d=4, k=3)
    Y = np.stack([y, -2.0 * y + 1.0], axis=1).astype(np.float32)
    a = np.asarray(cross_val_predict(mojolearn.Ridge(alpha=1.0), X, Y, cv=3))
    return [("dtype", a.dtype.str), ("values", _digest(a))]


def _target_free(which):
    import mojolearn
    from mojolearn.model_selection import cross_validate, cross_val_predict
    X, _, _ = _data(79, n=150, d=4, k=3)
    if which == "minibatch":
        res = cross_validate(mojolearn.MiniBatchKMeans(n_clusters=3, random_state=0), X, cv=3)
        return [("test_score", _digest(np.asarray(res["test_score"], dtype=np.float64)))]
    a = np.asarray(cross_val_predict(mojolearn.KMeans(n_clusters=3), X, cv=3))
    return [("dtype", a.dtype.str), ("values", _digest(a))]


def _group_search():
    import mojolearn
    from mojolearn.model_selection import GridSearchCV, GroupKFold
    X, _, c = _data(83, n=150, d=4, k=3)
    s = GridSearchCV(mojolearn.LogisticRegression(max_iter=100), {"C": [0.1, 1.0]}, scoring="accuracy",
                     cv=GroupKFold(n_splits=3)).fit(X, c, groups=np.arange(150) % 5)
    return _rows_of_search(s, X)


def _mixed_f1(error_score):
    import mojolearn
    from mojolearn.model_selection import GridSearchCV
    X, _, c = _data(89, n=120, d=4, k=2)
    target = np.asarray([1, "a"], dtype=object)[(c > 0).astype(np.int64)]
    s = GridSearchCV(mojolearn.LogisticRegression(max_iter=100), {"C": [0.1, 1.0]}, scoring="f1", cv=3,
                     error_score=error_score).fit(X, target)
    return _rows_of_search(s, X)


def _msel5b_cases():
    from mojolearn.model_selection import cross_val_score
    import mojolearn
    return [
        ("H5b_cvp_log_proba", lambda: _cv_predict("clf", "predict_log_proba")),
        ("H5b_lacking_proba", lambda: _warned(lambda: _lacking_predict("predict_proba"))),
        ("H5b_lacking_predict", lambda: _warned(lambda: _lacking_predict("predict"))),
        ("H5b_lacking_decision", lambda: _warned(lambda: _lacking_predict("decision_function"))),
        ("H5b_lacking_search", _lacking_search),
        ("H5b_multi_output_predict", _multi_output_predict),
        ("H5b_target_free_minibatch", lambda: _target_free("minibatch")),
        ("H5b_target_free_kmeans_predict", lambda: _target_free("kmeans")),
        ("H5b_group_search", _group_search),
        ("H5b_mixed_f1_raise", lambda: _mixed_f1("raise")),
        ("H5b_mixed_f1_nan", lambda: _warned(lambda: _mixed_f1(np.nan))),
        ("H5b_cv_score_default", lambda: cross_val_score(mojolearn.Ridge(alpha=1.0), *_data(97)[:2], scoring="r2")),
    ]


def _kmeans_score(weighted):
    import mojolearn
    X, _, _ = _data(101, n=180, d=5, k=3)
    w = (1.0 + (np.arange(180) % 3)).astype(np.float32) if weighted else None
    km = mojolearn.KMeans(n_clusters=4, n_init=2, random_state=0).fit(X, sample_weight=w)
    X2, _, _ = _data(103, n=60, d=5, k=3)
    return [("score_train", repr(float(km.score(X, sample_weight=w)))), ("minus_inertia", repr(-float(km.inertia_))),
            ("score_new", repr(float(km.score(X2))))]


def _kmeans_none(which):
    import mojolearn
    from mojolearn.model_selection import GridSearchCV, cross_validate
    X, _, _ = _data(107, n=150, d=4, k=3)
    if which == "search":
        s = GridSearchCV(mojolearn.KMeans(random_state=0), {"n_clusters": [2, 3, 4]}, cv=3).fit(X)
        return _rows_of_search(s, X)
    res = cross_validate(mojolearn.KMeans(n_clusters=3, random_state=0), X, cv=3)
    return [("test_score", _digest(np.asarray(res["test_score"], dtype=np.float64)))]


def _lacking_scored(scoring, error_score="raise"):
    import mojolearn
    from mojolearn.model_selection import GridSearchCV, KFold
    X, _, c = _data(109, n=150, d=4, k=3)
    order = np.argsort(c, kind="stable")
    s = GridSearchCV(mojolearn.LogisticRegression(max_iter=100), {"C": [0.1, 1.0]}, scoring=scoring,
                     cv=KFold(n_splits=3), error_score=error_score).fit(X[order], c[order])
    return _rows_of_search(s, X)


def _kmeans5_cases():
    return [
        ("H5k_kmeans_score_plain", lambda: _kmeans_score(False)),
        ("H5k_kmeans_score_weighted", lambda: _kmeans_score(True)),
        ("H5k_kmeans_search_none", lambda: _kmeans_none("search")),
        ("H5k_kmeans_cv_none", lambda: _kmeans_none("cv")),
        ("H5k_lacking_own_raise", lambda: _warned(lambda: _lacking_scored(None))),
        ("H5k_lacking_log_loss_raise", lambda: _lacking_scored("neg_log_loss")),
        ("H5k_lacking_log_loss_nan", lambda: _warned(lambda: _lacking_scored("neg_log_loss", np.nan))),
        ("H5k_lacking_roc_ovr_nan", lambda: _warned(lambda: _lacking_scored("roc_auc_ovr", np.nan))),
    ]


def _splitter_search(which):
    import mojolearn
    from mojolearn import model_selection as ms
    X, _, c = _data(113, n=150, d=4, k=3)
    groups = np.arange(150) % 6
    cv = {
        "shuffle": lambda: ms.ShuffleSplit(n_splits=3, test_size=0.25, random_state=0),
        "strat_shuffle": lambda: ms.StratifiedShuffleSplit(n_splits=3, test_size=0.25, random_state=0),
        "repeated": lambda: ms.RepeatedKFold(n_splits=3, n_repeats=2, random_state=0),
        "repeated_strat": lambda: ms.RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=0),
        "time_series": lambda: ms.TimeSeriesSplit(n_splits=3),
        "group_shuffle": lambda: ms.GroupShuffleSplit(n_splits=3, test_size=0.3, random_state=0),
        "logo": lambda: ms.LeaveOneGroupOut(),
        "index_list": lambda: [(np.arange(50, 150), np.arange(0, 50)), (np.arange(0, 100), np.arange(100, 150))],
    }[which]()
    s = ms.GridSearchCV(mojolearn.LogisticRegression(max_iter=100), {"C": [0.1, 1.0]}, scoring="accuracy", cv=cv,
                        return_train_score=True)
    if which in ("group_shuffle", "logo"):
        s.fit(X, c, groups=groups)
    else:
        s.fit(X, c)
    return _rows_of_search(s, X)


def _kwargs_search(which):
    import mojolearn
    from mojolearn import metrics
    from mojolearn.model_selection import GridSearchCV, make_scorer
    X, _, c = _data(127, n=150, d=4, k=3)
    target = c
    if which == "fbeta":
        scoring = make_scorer(metrics.fbeta_score, beta=2.0, average="macro")
    elif which == "f1_pos2":
        target = np.where(c > 0, 2, 0)
        scoring = make_scorer(metrics.f1_score, pos_label=2)
    elif which == "f1_binary_explicit":
        target = (c > 0).astype(np.int64)
        scoring = make_scorer(metrics.f1_score, average="binary")
    else:
        scoring = make_scorer(metrics.top_k_accuracy_score, k=int(which[-1]),
                                      response_method=("decision_function", "predict_proba"))
    s = GridSearchCV(mojolearn.LogisticRegression(max_iter=100), {"C": [0.1, 1.0]}, scoring=scoring,
                     cv=3).fit(X, target)
    return _rows_of_search(s, X)


def _bogus_error_score():
    import mojolearn
    from mojolearn.model_selection import GridSearchCV
    X, _, c = _data(131, n=90, d=4, k=3)
    return _rows_of_search(GridSearchCV(mojolearn.LogisticRegression(), {"C": [1.0]}, error_score="bogus",
                                        cv=3).fit(X, c), X)


def _multi_output(which, real):
    import mojolearn
    from mojolearn.model_selection import GridSearchCV, cross_val_predict
    X, _, c = _data(137, n=150, d=4, k=3)
    Y = np.stack([c, (c + 1) % 3], axis=1)
    Y = (Y.astype(np.float64) * 0.5) if real else Y.astype(np.int64)
    base = mojolearn.MultiOutputClassifier(mojolearn.LogisticRegression(max_iter=100))
    if which == "search":
        s = GridSearchCV(base, {"estimator__C": [0.1, 1.0]}, cv=3).fit(X, Y)
        return _rows_of_search(s, X)
    a = np.asarray(cross_val_predict(base, X, Y, cv=3))
    return [("dtype", a.dtype.str), ("values", _digest(a))]


def _permutation():
    import mojolearn
    from mojolearn.model_selection import permutation_test_score
    X, _, c = _data(139, n=120, d=4, k=3)
    score, perms, pvalue = permutation_test_score(mojolearn.LogisticRegression(max_iter=100), X, c, cv=3,
                                                  n_permutations=5, random_state=0, scoring="accuracy")
    return [("score", repr(float(score))), ("perms", _digest(np.asarray(perms, dtype=np.float64))),
            ("pvalue", repr(float(pvalue)))]


def _labels_search(scoring):
    import mojolearn
    from mojolearn.model_selection import GridSearchCV
    X, _, c = _data(149, n=150, d=4, k=2)
    s = GridSearchCV(mojolearn.LogisticRegression(max_iter=100), {"C": [0.1, 1.0]}, scoring=scoring,
                     cv=3).fit(X, c)
    return _rows_of_search(s, X)


def _given_y(which):
    import mojolearn
    from mojolearn.model_selection import GridSearchCV, StratifiedKFold
    rng = np.random.default_rng(151)
    X = rng.normal(size=(80, 4)).astype(np.float32)
    # four values, each pair equal in float32 but not in float64 (2^-40 apart): 8 classes at the door
    base = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.concatenate([base, base + 2.0 ** -40] * 10).astype(np.float64)
    ridge = mojolearn.RidgeCV(alphas=[0.1, 1.0], cv=StratifiedKFold(2))
    if which == "search":
        s = GridSearchCV(ridge, {"fit_intercept": [True, False]}, cv=2).fit(X, y)
        return _rows_of_search(s, X)
    if which == "stacking":
        model = mojolearn.StackingRegressor([("r", ridge)], final_estimator=mojolearn.Ridge(), cv=3)
    elif which == "voting":
        model = mojolearn.VotingRegressor([("r", ridge)])
    else:
        model = mojolearn.BaggingRegressor(ridge, n_estimators=3, random_state=0)
    return [("predict", _digest(np.asarray(model.fit(X, y).predict(X), dtype=np.float64)))]


def _multi_lacking(which):
    import mojolearn
    from mojolearn.model_selection import GridSearchCV, cross_val_predict, KFold
    X, _, c = _data(157, n=60, d=4, k=3)
    order = np.argsort(c, kind="stable")
    X, c = X[order], c[order]
    # output 1's class 2 only in the last rows: an unshuffled KFold's last fold trains without it
    o1 = np.where(np.arange(60) >= 50, 2, np.arange(60) % 2).astype(np.int64)
    Y = np.stack([c.astype(np.int64), o1], axis=1)
    base = mojolearn.MultiOutputClassifier(mojolearn.LogisticRegression(max_iter=100))
    if which == "search":
        return _rows_of_search(GridSearchCV(base, {"estimator__C": [0.1, 1.0]}, cv=KFold(3)).fit(X, Y), X)
    out = cross_val_predict(base, X, Y, cv=KFold(3), method=which)
    if which == "predict_proba":
        return [("list", repr((type(out).__name__, len(out))))] + [
            (f"output{j}", _digest(np.asarray(p, dtype=np.float64))) for j, p in enumerate(out)]
    return [("values", _digest(np.asarray(out)))]


def _lacking_single_predict():
    import mojolearn
    from mojolearn.model_selection import cross_val_predict, KFold
    import warnings
    X, _, c = _data(163, n=60, d=4, k=3)
    order = np.argsort(c, kind="stable")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = cross_val_predict(mojolearn.LogisticRegression(max_iter=100), X[order], c[order], cv=KFold(3))
    return [("values", _digest(np.asarray(out))), ("warnings", repr(sorted(str(w.message) for w in caught)))]


def _learning_many():
    import mojolearn
    from mojolearn.model_selection import learning_curve
    X, _, c = _data(167, n=90, d=4, k=3)
    return learning_curve(mojolearn.LogisticRegression(max_iter=100), X, c, cv=3, train_sizes=[0.5, 1.0],
                          scoring=["accuracy", "f1_macro"])


def _tail5_cases():
    cases = [("H5t_given_" + w, (lambda w=w: _warned(lambda: _given_y(w))))
             for w in ("search", "stacking", "voting", "bagging")]
    cases += [("H5t_multi_lacking_" + w, (lambda w=w: _warned(lambda: _multi_lacking(w))))
              for w in ("search", "predict", "predict_proba")]
    cases += [
        ("H5t_lacking_single_predict", _lacking_single_predict),
        ("H5t_permutation", _permutation),
        ("H5t_learning_many", _learning_many),
    ]
    return cases


def _msel5c_cases():
    cases = [("H5c_split_" + w, (lambda w=w: _splitter_search(w)))
             for w in ("shuffle", "strat_shuffle", "repeated", "repeated_strat", "time_series", "group_shuffle",
                       "logo", "index_list")]
    cases += [("H5c_kwargs_" + w, (lambda w=w: _warned(lambda: _kwargs_search(w))))
              for w in ("fbeta", "f1_pos2", "f1_binary_explicit", "top_k1", "top_k3")]
    cases += [
        ("H5c_error_score_bogus", _bogus_error_score),
        ("H5c_lacking_ap_raise", lambda: _lacking_scored("average_precision")),
        ("H5c_lacking_ap_nan", lambda: _warned(lambda: _lacking_scored("average_precision", np.nan))),
        ("H5c_lacking_brier_raise", lambda: _lacking_scored("neg_brier_score")),
        ("H5c_lacking_brier_nan", lambda: _warned(lambda: _lacking_scored("neg_brier_score", np.nan))),
        ("H5c_multi_search_int", lambda: _multi_output("search", False)),
        ("H5c_multi_predict_int", lambda: _multi_output("predict", False)),
        ("H5c_multi_predict_real", lambda: _multi_output("predict", True)),
        ("H5c_permutation", _permutation),
        ("H5c_labels_f1", lambda: _warned(lambda: _labels_search("f1"))),
        ("H5c_labels_roc_auc", lambda: _labels_search("roc_auc")),
        ("H5c_labels_brier", lambda: _labels_search("neg_brier_score")),
    ]
    return cases


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


def _tail5b_search(which):
    # lane mf-tail5b item 1: make_scorer kwargs as scorer word groups (base: the door's Python units)
    import mojolearn
    from mojolearn import metrics
    from mojolearn.model_selection import GridSearchCV, make_scorer
    X, _, c = _data(151, n=150, d=4, k=3)
    target = c
    if which == "fbeta_binary":
        target = (c > 0).astype(np.int64)
        scoring = make_scorer(metrics.fbeta_score, beta=0.5)
    elif which == "fbeta_macro":
        scoring = make_scorer(metrics.fbeta_score, beta=2.0, average="macro")
    elif which == "fbeta_zero":
        scoring = make_scorer(metrics.fbeta_score, beta=0.0, average="weighted")
    elif which == "labels_array":
        scoring = make_scorer(metrics.precision_score, average="macro", labels=np.array([2, 0]))
    elif which == "labels_absent":
        scoring = make_scorer(metrics.recall_score, average="micro", labels=[0, 7])
    elif which == "zero_division_1":
        scoring = make_scorer(metrics.f1_score, average="macro", zero_division=1)
    elif which == "ap_pos0":
        target = (c > 0).astype(np.int64)
        scoring = make_scorer(metrics.average_precision_score, pos_label=0,
                              response_method=("decision_function", "predict_proba"))
    elif which == "brier_pos0":
        target = (c > 0).astype(np.int64)
        scoring = make_scorer(metrics.brier_score_loss, pos_label=0, greater_is_better=False,
                              response_method="predict_proba")
    elif which == "f1_pos_absent":
        target = np.where(c > 0, 2, 0)
        scoring = make_scorer(metrics.f1_score, pos_label=5)
    elif which == "f1_pos_kind":
        target = np.where(c > 0, "b", "a")
        scoring = make_scorer(metrics.f1_score, pos_label=1)
    else:  # f1_pos_mixed
        target = np.array([("a" if v == 0 else int(v)) for v in c], dtype=object)
        scoring = make_scorer(metrics.f1_score, pos_label="a", average="binary")
    s = GridSearchCV(mojolearn.LogisticRegression(max_iter=100), {"C": [0.1, 1.0]}, scoring=scoring, cv=3,
                     error_score="raise").fit(X, target)
    return _rows_of_search(s, X)


def _tail5b_text_multi(which):
    # lane mf-tail5b item 2: a text 2-D y (search and cross_val_predict)
    import mojolearn
    from mojolearn.model_selection import GridSearchCV, cross_val_predict
    X, _, c = _data(157, n=150, d=4, k=3)
    words = np.array(["red", "green", "blue"])
    Y = np.stack([words[c], words[(c + 1) % 3]], axis=1)
    base = mojolearn.MultiOutputClassifier(mojolearn.LogisticRegression(max_iter=100))
    if which == "search":
        return _rows_of_search(GridSearchCV(base, {"estimator__C": [0.1, 1.0]}, cv=3).fit(X, Y), X)
    a = cross_val_predict(base, X, Y, cv=3)
    return [("rows", repr([list(map(str, r)) for r in a]))]


def _tail5b_refusals(which):
    # lane mf-tail5b items 3, 4: refusals raised natively (class and words)
    import mojolearn
    from mojolearn.model_selection import GridSearchCV, learning_curve
    X, _, c = _data(163, n=90, d=4, k=3)
    try:
        if which == "learning_many":
            learning_curve(mojolearn.LogisticRegression(), X, c, cv=3, scoring=["accuracy", "f1_macro"])
        elif which == "cv_one":
            GridSearchCV(mojolearn.LogisticRegression(), {"C": [1.0]}, cv=1).fit(X, c)
        elif which == "error_score_none":
            GridSearchCV(mojolearn.LogisticRegression(), {"C": [1.0]}, cv=3, error_score=None).fit(X, c)
        else:  # no candidates
            GridSearchCV(mojolearn.LogisticRegression(), {"C": []}, cv=3).fit(X, c)
    except Exception as error:  # the class and its bases, the words
        return [("class", type(error).__name__), ("value_error", repr(isinstance(error, ValueError))),
                ("type_error", repr(isinstance(error, TypeError))), ("words", str(error))]
    return [("raised", "none")]


def _tail5b_cases():
    names = ["fbeta_binary", "fbeta_macro", "fbeta_zero", "labels_array", "labels_absent", "zero_division_1",
             "ap_pos0", "brier_pos0", "f1_pos_absent", "f1_pos_kind", "f1_pos_mixed"]
    cases = [("H5tb_kwargs_" + w, (lambda w=w: _warned(lambda: _tail5b_search(w)))) for w in names]
    cases += [("H5tb_text_multi_" + w, (lambda w=w: _tail5b_text_multi(w))) for w in ("search", "predict")]
    cases += [("H5tb_refuse_" + w, (lambda w=w: _tail5b_refusals(w)))
              for w in ("learning_many", "cv_one", "error_score_none", "no_candidates")]
    return cases


def _all_cases():
    return (_metric_cases() + _search_cases() + _score5_metric_cases() + _score5_search_cases()
            + _search5_cases() + _cluster5_metric_cases() + _cluster5_search_cases() + _msel5_cases()
            + _msel5b_cases() + _kmeans5_cases() + _msel5c_cases() + _tail5_cases() + _tail5b_cases())


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
