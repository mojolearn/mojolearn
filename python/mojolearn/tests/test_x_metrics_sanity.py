# SPDX-License-Identifier: Apache-2.0
"""The metrics lane's correctness sanity against scikit-learn 1.9 (a
tolerance check on small data, not an identity claim; the identity lanes
are tools/identity_lanes/metrics.py). Needs scikit-learn and NumPy:

    PYTHONPATH=python:/root/skl pixi run -e default python python/mojolearn/tests/test_x_metrics_sanity.py
"""
import sys
import warnings

import numpy as np

import mojolearn.metrics as mt
from sklearn import metrics as sk

FAILS = []
RTOL = 2e-4


def close(name, ours, theirs, rtol=RTOL, atol=1e-5):
    a = np.asarray(ours, dtype=np.float64)
    b = np.asarray(theirs, dtype=np.float64)
    ok = a.shape == b.shape and np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
    print(("ok  " if ok else "FAIL"), name, "" if ok else f"ours={a} sklearn={b}")
    if not ok:
        FAILS.append(name)


def call(name, fn, *args, **kw):
    try:
        return fn(*args, **kw)
    except Exception as e:  # noqa: BLE001
        print("FAIL", name, "raised", type(e).__name__, e)
        FAILS.append(name)
        return None


def both(name, fname, *args, rtol=RTOL, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ours = call(name, getattr(mt, fname), *args, **kw)
        theirs = getattr(sk, fname)(*args, **kw)
    if ours is not None:
        if isinstance(theirs, tuple):
            for i, (a, b) in enumerate(zip(ours, theirs)):
                if b is None:
                    continue
                close(f"{name}[{i}]", a, b, rtol=rtol)
        else:
            close(name, ours, theirs, rtol=rtol)


rng = np.random.default_rng(0)
n = 500
yt = rng.integers(0, 4, n).astype(np.int32)
yp = np.where(rng.random(n) < 0.7, yt, rng.integers(0, 4, n)).astype(np.int32)
bt = (rng.random(n) < 0.4).astype(np.int32)
bp = np.where(rng.random(n) < 0.8, bt, 1 - bt).astype(np.int32)
w = rng.random(n).astype(np.float32) * 2
w[::7] = 0

# classification
for sw in (None, w):
    t = "w" if sw is not None else "u"
    both(f"balanced_accuracy {t}", "balanced_accuracy_score", yt, yp, sample_weight=sw)
    both(f"balanced_accuracy adj {t}", "balanced_accuracy_score", yt, yp, sample_weight=sw, adjusted=True)
    both(f"mcc {t}", "matthews_corrcoef", yt, yp, sample_weight=sw)
    both(f"mcc binary {t}", "matthews_corrcoef", bt, bp, sample_weight=sw)
    for wt in (None, "linear", "quadratic"):
        both(f"kappa {wt} {t}", "cohen_kappa_score", yt, yp, weights=wt, sample_weight=sw)
    for avg in (None, "micro", "macro", "weighted"):
        both(f"jaccard {avg} {t}", "jaccard_score", yt, yp, average=avg, sample_weight=sw)
        both(f"prfs {avg} {t}", "precision_recall_fscore_support", yt, yp, beta=0.5, average=avg, sample_weight=sw)
        both(f"f1 {avg} {t}", "f1_score", yt, yp, average=avg, sample_weight=sw)
        both(f"precision {avg} {t}", "precision_score", yt, yp, average=avg, sample_weight=sw)
    both(f"fbeta binary {t}", "fbeta_score", bt, bp, beta=2.0, sample_weight=sw)
    both(f"jaccard binary {t}", "jaccard_score", bt, bp, sample_weight=sw)
    both(f"hamming {t}", "hamming_loss", yt, yp, sample_weight=sw)
    both(f"zero_one {t}", "zero_one_loss", yt, yp, sample_weight=sw)
    both(f"zero_one count {t}", "zero_one_loss", yt, yp, normalize=False, sample_weight=sw)
    both(f"accuracy count {t}", "accuracy_score", yt, yp, normalize=False, sample_weight=sw)
    both(f"mcm {t}", "multilabel_confusion_matrix", yt, yp, sample_weight=sw, labels=[3, 0, 1])
    both(f"lr {t}", "class_likelihood_ratios", bt, bp, sample_weight=sw)
both("confusion w", "confusion_matrix", yt, yp, sample_weight=w)
both("confusion w true", "confusion_matrix", yt, yp, sample_weight=w, normalize="true")
both("confusion w all", "confusion_matrix", yt, yp, sample_weight=w, normalize="all")
rep = mt.classification_report(yt, yp, output_dict=True)
ref = sk.classification_report(yt, yp, output_dict=True)
close("classification_report macro f1", rep["macro avg"]["f1-score"], ref["macro avg"]["f1-score"])
close("classification_report accuracy", rep["accuracy"], ref["accuracy"])

# regression
y = rng.normal(size=n).astype(np.float32)
p = (y * 0.8 + rng.normal(size=n) * 0.3).astype(np.float32)
Y2 = np.stack([y, rng.normal(size=n)], axis=1).astype(np.float32)
P2 = np.stack([p, Y2[:, 1] * 0.5], axis=1).astype(np.float32)
Y2[::5, 1] = Y2[::5, 1].round()
for sw in (None, w):
    t = "w" if sw is not None else "u"
    for mo in ("raw_values", "uniform_average", [0.3, 0.7]):
        for f in ("mean_squared_error", "mean_absolute_error", "root_mean_squared_error",
                  "mean_absolute_percentage_error", "median_absolute_error", "d2_absolute_error_score"):
            both(f"{f} {mo} {t}", f, Y2, P2, sample_weight=sw, multioutput=mo)
        both(f"pinball {mo} {t}", "mean_pinball_loss", Y2, P2, alpha=0.3, sample_weight=sw, multioutput=mo)
        both(f"d2_pinball {mo} {t}", "d2_pinball_score", Y2, P2, alpha=0.8, sample_weight=sw, multioutput=mo)
        both(f"msle {mo} {t}", "mean_squared_log_error", np.abs(Y2), np.abs(P2), sample_weight=sw, multioutput=mo)
        both(f"rmsle {mo} {t}", "root_mean_squared_log_error", np.abs(Y2), np.abs(P2), sample_weight=sw, multioutput=mo)
    for mo in ("raw_values", "uniform_average", "variance_weighted"):
        both(f"ev {mo} {t}", "explained_variance_score", Y2, P2, sample_weight=sw, multioutput=mo)
        both(f"r2 {mo} {t}", "r2_score", Y2, P2, sample_weight=sw, multioutput=mo)
    py, pp = (np.abs(y) + 0.5).astype(np.float32), (np.abs(p) + 0.25).astype(np.float32)
    for pw in (-0.5, 0, 1, 1.5, 2, 3):
        both(f"tweedie {pw} {t}", "mean_tweedie_deviance", py, pp, sample_weight=sw, power=pw)
        both(f"d2_tweedie {pw} {t}", "d2_tweedie_score", py, pp, sample_weight=sw, power=pw)
    both(f"median 1d {t}", "median_absolute_error", y, p, sample_weight=sw)
both("max_error", "max_error", y, p)
both("r2 force_finite False", "r2_score", np.ones((8, 2), np.float32), np.ones((8, 2), np.float32),
     multioutput="raw_values", force_finite=False)

# ranking
s = (rng.normal(size=n) + bt).round(1).astype(np.float32)
P = rng.random((n, 4)).astype(np.float64) + 0.05
P = (P / P.sum(axis=1, keepdims=True)).astype(np.float32)
pb = P[:, 1] / (P[:, 0] + P[:, 1])
for sw in (None, w):
    t = "w" if sw is not None else "u"
    for di in (True, False):
        both(f"roc_curve {di} {t}", "roc_curve", bt, s, sample_weight=sw, drop_intermediate=di)
        both(f"det_curve {di} {t}", "det_curve", bt, s, sample_weight=sw, drop_intermediate=di)
        both(f"pr_curve {di} {t}", "precision_recall_curve", bt, s, sample_weight=sw, drop_intermediate=di)
    both(f"auc binary {t}", "roc_auc_score", bt, s, sample_weight=sw)
    both(f"auc max_fpr {t}", "roc_auc_score", bt, s, sample_weight=sw, max_fpr=0.3)
    for avg in ("macro", "weighted", "micro", None):
        both(f"auc ovr {avg} {t}", "roc_auc_score", yt, P, multi_class="ovr", average=avg, sample_weight=sw)
        both(f"ap {avg} {t}", "average_precision_score", yt, P, average=avg, sample_weight=sw)
    both(f"ap binary {t}", "average_precision_score", bt, s, sample_weight=sw)
    for k in (1, 2, 3):
        both(f"topk {k} {t}", "top_k_accuracy_score", yt, P, k=k, sample_weight=sw)
    both(f"topk binary {t}", "top_k_accuracy_score", bt, pb, k=1, sample_weight=sw)
    both(f"brier mc {t}", "brier_score_loss", yt, P, sample_weight=sw)
    both(f"brier bin {t}", "brier_score_loss", bt, pb, sample_weight=sw)
    both(f"d2_brier {t}", "d2_brier_score", yt, P, sample_weight=sw)
    both(f"log_loss {t}", "log_loss", yt, P, sample_weight=sw)
    both(f"d2_log_loss {t}", "d2_log_loss_score", yt, P, sample_weight=sw)
    both(f"hinge bin {t}", "hinge_loss", bt, s, sample_weight=sw)
    both(f"hinge mc {t}", "hinge_loss", yt, P, sample_weight=sw)
    rel = rng.integers(0, 3, (60, 6)).astype(np.float32)
    ind = (rng.random((60, 6)) < 0.3).astype(np.float32)
    sc = rng.random((60, 6)).round(1).astype(np.float32)
    ws = None if sw is None else sw[:60]
    for k in (None, 3):
        both(f"dcg {k} {t}", "dcg_score", rel, sc, k=k, sample_weight=ws)
        both(f"ndcg {k} {t}", "ndcg_score", rel, sc, k=k, sample_weight=ws)
    both(f"coverage {t}", "coverage_error", ind, sc, sample_weight=ws)
    both(f"lrap {t}", "label_ranking_average_precision_score", ind, sc, sample_weight=ws)
    both(f"rankloss {t}", "label_ranking_loss", ind, sc, sample_weight=ws)
for avg in ("macro", "weighted"):
    both(f"auc ovo {avg}", "roc_auc_score", yt, P, multi_class="ovo", average=avg)

# clustering
eight = rng.integers(0, 8, n).astype(np.int32)
X = rng.normal(size=(n, 5)).astype(np.float32)
for m in ("min", "geometric", "arithmetic", "max"):
    both(f"nmi {m}", "normalized_mutual_info_score", yt, eight, average_method=m)
    both(f"ami {m}", "adjusted_mutual_info_score", yt, eight, average_method=m)
close("contingency", mt.contingency_matrix(yt, eight), sk.cluster.contingency_matrix(yt, eight))
close("pair_confusion", mt.pair_confusion_matrix(yt, eight), sk.cluster.pair_confusion_matrix(yt, eight))
both("calinski_harabasz", "calinski_harabasz_score", X, eight)
both("davies_bouldin", "davies_bouldin_score", X, eight)

# model_selection: the unshuffled splitters give scikit-learn's exact indices
import mojolearn.model_selection as ms
from sklearn import model_selection as skms
g = (np.arange(n) // 7) % 23
Xs = X
for name in ("KFold", "StratifiedKFold", "GroupKFold", "TimeSeriesSplit", "LeaveOneGroupOut"):
    ours = [(np.asarray(a).tolist(), np.asarray(b).tolist()) for a, b in getattr(ms, name)().split(Xs, yt, g)]
    theirs = [(a.tolist(), b.tolist()) for a, b in getattr(skms, name)().split(Xs, yt, g)]
    ok = ours == theirs
    print("ok  " if ok else "FAIL", "split", name)
    if not ok:
        FAILS.append(name)
for name, a in (("LeavePGroupsOut", 2), ("LeavePOut", 2)):
    kw = X[:9]
    ours = [(np.asarray(p).tolist(), np.asarray(q).tolist()) for p, q in getattr(ms, name)(a).split(kw, yt[:9], np.arange(9) % 4)]
    theirs = [(p.tolist(), q.tolist()) for p, q in getattr(skms, name)(a).split(kw, yt[:9], np.arange(9) % 4)]
    print("ok  " if ours == theirs else "FAIL", "split", name)
    if ours != theirs:
        FAILS.append(name)
# seeded splitters: partition invariants and determinism
for cv in (ms.KFold(5, shuffle=True, random_state=1), ms.StratifiedKFold(5, shuffle=True, random_state=1),
           ms.GroupKFold(5, shuffle=True, random_state=1), ms.RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=2)):
    s1 = [(np.asarray(a).tolist(), np.asarray(b).tolist()) for a, b in cv.split(Xs, yt, g)]
    s2 = [(np.asarray(a).tolist(), np.asarray(b).tolist()) for a, b in cv.split(Xs, yt, g)]
    ok = s1 == s2 and all(sorted(a + b) == list(range(n)) for a, b in s1)
    print("ok  " if ok else "FAIL", "seeded", type(cv).__name__)
    if not ok:
        FAILS.append(type(cv).__name__)
sss = list(ms.StratifiedShuffleSplit(3, test_size=0.2, random_state=0).split(Xs, yt))
ok = all(len(b) == 100 and len(set(a) & set(b)) == 0 for a, b in sss)
print("ok  " if ok else "FAIL", "StratifiedShuffleSplit sizes")
if not ok:
    FAILS.append("sss")
from sklearn.linear_model import Ridge
from mojolearn import GradientBoostingRegressor as GBR
yr_ = (X[:, 0] * 2 + X[:, 1]).astype(np.float32)
res = ms.cross_validate(GBR(n_estimators=5, max_depth=2), X, yr_, cv=3, scoring=["r2", "neg_mean_squared_error"])
print("ok  " if len(res["test_r2"]) == 3 else "FAIL", "cross_validate")
gs = ms.GridSearchCV(GBR(n_estimators=5), {"max_depth": [2, 3]}, cv=3, scoring="r2").fit(X, yr_)
print("ok  " if gs.best_index_ in (0, 1) else "FAIL", "GridSearchCV", gs.best_params_)

print(f"{len(FAILS)} failures")
sys.exit(1 if FAILS else 0)
