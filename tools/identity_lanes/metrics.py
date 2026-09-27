# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# THE METRICS LANE'S IDENTITY LANES (docs/lanes/ALGORITHM_EXPANSION_PLAN.md item 1a).
#
# Owned by the `metrics` expansion lane. tools/identity_break.py executes this
# file in ITS OWN namespace after every helper and registry exists
# (`_load_lane_fragments`). Rules the loader enforces: spell every lane name
# literally; add only your own lanes; rebind no existing name; prefix helpers
# with `_metrics_`. No imports are needed: np, _h, _fit, _hw are the harness's.
#
# The metrics are functions, not estimators, so every lane is a train-only
# cell (`_fit(parts)`, the pattern of the existing `metrics-*` lanes).


def _metrics_labels(X, n):
    """Four classes from the signs of two fixture columns (ties everywhere),
    and a prediction that agrees on most rows: the confusion is dense."""
    a = (X[:n, 3] > 0).astype(np.int32) + 2 * (X[:n, 4] > 0).astype(np.int32)
    b = (X[:n, 3] + np.float32(0.25) * X[:n, 5] > 0).astype(np.int32) + 2 * (X[:n, 4] > 0).astype(np.int32)
    return np.ascontiguousarray(a), np.ascontiguousarray(b)


def _metrics_weights(n, seed):
    """Non-negative float32 weights from the hashed stream, a zero every 11th
    row (a weight the folds must skip exactly)."""
    w = _hw((n,), seed, 0.0, 3.0)
    w[::11] = np.float32(0.0)
    return np.ascontiguousarray(w)


@lane("x-metrics-classification")
def _(ml, X, yc, yr, Xh=None):
    """The classification functions the metrics lane added, and the
    sample_weight / normalize options it added to the existing ones. Every
    sum crosses the device PairSum (DEVIATION 6100) or the exact integer
    group counts; the ratios are the binary64 epilogue (DEVIATION 6106)."""
    mt = ml.metrics
    n = 3000
    yt, yp = _metrics_labels(X, n)
    w = _metrics_weights(n, "x-metrics-classification:w")
    bt = np.ascontiguousarray(yc[:n]).astype(np.int32)
    bp = (X[:n, 3] > 0).astype(np.int32)
    parts = {}
    for sw in (None, w):
        tag = "w" if sw is not None else "u"
        parts[f"balanced:{tag}"] = _h(np.float64(mt.balanced_accuracy_score(yt, yp, sample_weight=sw)))
        parts[f"balanced_adj:{tag}"] = _h(np.float64(mt.balanced_accuracy_score(yt, yp, sample_weight=sw, adjusted=True)))
        parts[f"mcc:{tag}"] = _h(np.float64(mt.matthews_corrcoef(yt, yp, sample_weight=sw)))
        for kw in (None, "linear", "quadratic"):
            parts[f"kappa:{kw}:{tag}"] = _h(np.float64(mt.cohen_kappa_score(yt, yp, weights=kw, sample_weight=sw)))
        for avg in (None, "micro", "macro", "weighted"):
            parts[f"jaccard:{avg}:{tag}"] = _h(np.asarray(mt.jaccard_score(yt, yp, average=avg, sample_weight=sw), dtype=np.float64))
            p, r, f, s = mt.precision_recall_fscore_support(yt, yp, beta=0.5, average=avg, sample_weight=sw)
            parts[f"prfs:{avg}:{tag}"] = _h(*[np.asarray(v, dtype=np.float64) for v in (p, r, f) ],
                                            np.asarray(-1.0 if s is None else s, dtype=np.float64))
        parts[f"fbeta2:binary:{tag}"] = _h(np.float64(mt.fbeta_score(bt, bp, beta=2.0, sample_weight=sw)))
        parts[f"hamming:{tag}"] = _h(np.float64(mt.hamming_loss(yt, yp, sample_weight=sw)))
        parts[f"zero_one:{tag}"] = _h(np.float64(mt.zero_one_loss(yt, yp, sample_weight=sw)),
                                      np.float64(mt.zero_one_loss(yt, yp, normalize=False, sample_weight=sw)))
        parts[f"mcm:{tag}"] = _h(np.asarray(mt.multilabel_confusion_matrix(yt, yp, sample_weight=sw, labels=[3, 0, 1]), dtype=np.float64))
        parts[f"lr:{tag}"] = _h(np.asarray(mt.class_likelihood_ratios(bt, bp, sample_weight=sw), dtype=np.float64))
        parts[f"accuracy_count:{tag}"] = _h(np.float64(mt.accuracy_score(yt, yp, normalize=False, sample_weight=sw)))
    parts["confusion:w"] = _h(np.asarray(mt.confusion_matrix(yt, yp, sample_weight=w), dtype=np.float64))
    parts["confusion:w:true"] = _h(np.asarray(mt.confusion_matrix(yt, yp, sample_weight=w, normalize="true"), dtype=np.float64))
    for avg in (None, "macro", "weighted"):
        for fn in (mt.precision_score, mt.recall_score, mt.f1_score):
            parts[f"{fn.__name__}:{avg}:w"] = _h(np.asarray(fn(yt, yp, average=avg, sample_weight=w), dtype=np.float64))
    return _fit(parts)


@lane("x-metrics-regression")
def _(ml, X, yc, yr, Xh=None):
    """The regression functions the metrics lane added and the sample_weight
    / multioutput / force_finite options of the existing ones. Targets are
    fixture values (exact host arithmetic only: abs, add, a power-of-two
    scale), one and two outputs, with exact ties for the median's sort."""
    mt = ml.metrics
    n = 3000
    y = np.ascontiguousarray(yr[:n]).astype(np.float32)
    p = np.ascontiguousarray(y * np.float32(0.75) + X[:n, 3] * np.float32(0.5)).astype(np.float32)
    Y2 = np.ascontiguousarray(np.stack([y, np.round(X[:n, 4] * np.float32(4)) * np.float32(0.25)], axis=1)).astype(np.float32)
    P2 = np.ascontiguousarray(np.stack([p, np.round(X[:n, 5] * np.float32(4)) * np.float32(0.25)], axis=1)).astype(np.float32)
    pos_y = np.ascontiguousarray(np.abs(y) + np.float32(0.5)).astype(np.float32)
    pos_p = np.ascontiguousarray(np.abs(p) + np.float32(0.25)).astype(np.float32)
    w = _metrics_weights(n, "x-metrics-regression:w")
    parts = {}
    for sw in (None, w):
        tag = "w" if sw is not None else "u"
        for mo in ("raw_values", "uniform_average", [0.25, 0.75]):
            mtag = mo if isinstance(mo, str) else "array"
            for fn in (mt.mean_squared_error, mt.mean_absolute_error, mt.root_mean_squared_error,
                       mt.mean_absolute_percentage_error, mt.median_absolute_error,
                       mt.d2_absolute_error_score):
                parts[f"{fn.__name__}:{mtag}:{tag}"] = _h(np.asarray(fn(Y2, P2, sample_weight=sw, multioutput=mo), dtype=np.float64))
            parts[f"pinball:{mtag}:{tag}"] = _h(np.asarray(mt.mean_pinball_loss(Y2, P2, alpha=0.3, sample_weight=sw, multioutput=mo), dtype=np.float64))
            parts[f"d2_pinball:{mtag}:{tag}"] = _h(np.asarray(mt.d2_pinball_score(Y2, P2, alpha=0.8, sample_weight=sw, multioutput=mo), dtype=np.float64))
            parts[f"msle:{mtag}:{tag}"] = _h(np.asarray(mt.mean_squared_log_error(np.abs(Y2), np.abs(P2), sample_weight=sw, multioutput=mo), dtype=np.float64),
                                             np.asarray(mt.root_mean_squared_log_error(np.abs(Y2), np.abs(P2), sample_weight=sw, multioutput=mo), dtype=np.float64))
        for mo in ("raw_values", "uniform_average", "variance_weighted"):
            for ff in (True, False):
                parts[f"ev:{mo}:{ff}:{tag}"] = _h(np.asarray(mt.explained_variance_score(Y2, P2, sample_weight=sw, multioutput=mo, force_finite=ff), dtype=np.float64))
                parts[f"r2:{mo}:{ff}:{tag}"] = _h(np.asarray(mt.r2_score(Y2, P2, sample_weight=sw, multioutput=mo, force_finite=ff), dtype=np.float64))
        for pw in (-0.5, 0, 1, 1.5, 2, 3):
            parts[f"tweedie:{pw}:{tag}"] = _h(np.float64(mt.mean_tweedie_deviance(pos_y, pos_p, sample_weight=sw, power=pw)),
                                              np.float64(mt.d2_tweedie_score(pos_y, pos_p, sample_weight=sw, power=pw)))
        parts[f"poisson_gamma:{tag}"] = _h(np.float64(mt.mean_poisson_deviance(pos_y, pos_p, sample_weight=sw)),
                                           np.float64(mt.mean_gamma_deviance(pos_y, pos_p, sample_weight=sw)))
        parts[f"median_1d:{tag}"] = _h(np.float64(mt.median_absolute_error(y, p, sample_weight=sw)))
    parts["max_error"] = _h(np.float64(mt.max_error(y, p)))
    parts["r2_const"] = _h(np.asarray(mt.r2_score(np.ones((8, 2), np.float32), np.ones((8, 2), np.float32),
                                                  multioutput="raw_values", force_finite=False), dtype=np.float64))
    return _fit(parts)


def _metrics_proba(X, n, k):
    """A row-stochastic (n, k) Float32 matrix built with exact host steps:
    small integer weights from the fixture's signs, each row divided by its
    integer total (one correctly rounded division per cell)."""
    cols = [np.floor(np.abs(X[:n, 3 + c]) * np.float32(4)).astype(np.float32) + np.float32(1) for c in range(k)]
    M = np.stack(cols, axis=1).astype(np.float64)
    return np.ascontiguousarray((M / M.sum(axis=1, keepdims=True)).astype(np.float32))


@lane("x-metrics-ranking")
def _(ml, X, yc, yr, Xh=None):
    """The ranking and probabilistic scores the metrics lane added, and the
    weighted / partial / multiclass options of roc_auc_score,
    precision_recall_curve and log_loss. Scores are quantized fixture
    columns, so ties are plentiful and the tie handling is read."""
    mt = ml.metrics
    n = 3000
    bt = np.ascontiguousarray(yc[:n]).astype(np.int32)
    s = np.ascontiguousarray(np.round(X[:n, 3] * np.float32(8)) * np.float32(0.125)).astype(np.float32)
    yt, _ = _metrics_labels(X, n)
    P = _metrics_proba(X, n, 4)
    w = _metrics_weights(n, "x-metrics-ranking:w")
    rel = np.ascontiguousarray((np.abs(np.round(X[:200, 3:9] * np.float32(2)))).astype(np.float32))
    ind = np.ascontiguousarray((X[:200, 3:9] > np.float32(0.3)).astype(np.float32))
    sc = np.ascontiguousarray(np.round(X[:200, 9:15] * np.float32(4)) * np.float32(0.25)).astype(np.float32)
    pb = np.ascontiguousarray(P[:, 1] / (P[:, 0] + P[:, 1])).astype(np.float32)
    parts = {}
    for sw in (None, w):
        tag = "w" if sw is not None else "u"
        for di in (True, False):
            parts[f"roc_curve:{di}:{tag}"] = _h(*[np.asarray(a) for a in mt.roc_curve(bt, s, sample_weight=sw, drop_intermediate=di)])
            parts[f"pr_curve:{di}:{tag}"] = _h(*[np.asarray(a) for a in mt.precision_recall_curve(bt, s, sample_weight=sw, drop_intermediate=True)]) if di else _h(np.float64(0))
            parts[f"det_curve:{di}:{tag}"] = _h(*[np.asarray(a) for a in mt.det_curve(bt, s, sample_weight=sw, drop_intermediate=di)])
        parts[f"auc_binary:{tag}"] = _h(np.float64(mt.roc_auc_score(bt, s, sample_weight=sw)),
                                        np.float64(mt.roc_auc_score(bt, s, sample_weight=sw, max_fpr=0.3)))
        for avg in ("macro", "weighted", "micro", None):
            parts[f"auc_ovr:{avg}:{tag}"] = _h(np.asarray(mt.roc_auc_score(yt, P, multi_class="ovr", average=avg, sample_weight=sw), dtype=np.float64))
            parts[f"ap:{avg}:{tag}"] = _h(np.asarray(mt.average_precision_score(yt, P, average=avg, sample_weight=sw), dtype=np.float64))
        parts[f"ap_binary:{tag}"] = _h(np.float64(mt.average_precision_score(bt, s, sample_weight=sw)))
        for k in (1, 2, 3):
            parts[f"topk:{k}:{tag}"] = _h(np.float64(mt.top_k_accuracy_score(yt, P, k=k, sample_weight=sw)),
                                          np.float64(mt.top_k_accuracy_score(yt, P, k=k, normalize=False, sample_weight=sw)))
        parts[f"topk_binary:{tag}"] = _h(np.float64(mt.top_k_accuracy_score(bt, pb, k=1, sample_weight=sw)))
        parts[f"brier:{tag}"] = _h(np.float64(mt.brier_score_loss(yt, P, sample_weight=sw)),
                                   np.float64(mt.brier_score_loss(bt, pb, sample_weight=sw)),
                                   np.float64(mt.d2_brier_score(yt, P, sample_weight=sw)))
        parts[f"log_loss:{tag}"] = _h(np.float64(mt.log_loss(yt, P, sample_weight=sw) if sw is not None else 0.0),
                                      np.float64(mt.d2_log_loss_score(yt, P, sample_weight=sw)))
        parts[f"hinge:{tag}"] = _h(np.float64(mt.hinge_loss(bt, s, sample_weight=sw)),
                                   np.float64(mt.hinge_loss(yt, P, sample_weight=sw)))
        ws = None if sw is None else np.ascontiguousarray(sw[:200])
        for k in (None, 3):
            parts[f"dcg:{k}:{tag}"] = _h(np.float64(mt.dcg_score(rel, sc, k=k, sample_weight=ws)),
                                         np.float64(mt.dcg_score(rel, sc, k=k, sample_weight=ws, ignore_ties=True)),
                                         np.float64(mt.ndcg_score(rel, sc, k=k, sample_weight=ws)),
                                         np.float64(mt.ndcg_score(rel, sc, k=k, sample_weight=ws, ignore_ties=True)))
        parts[f"label_ranking:{tag}"] = _h(np.float64(mt.coverage_error(ind, sc, sample_weight=ws)),
                                           np.float64(mt.label_ranking_average_precision_score(ind, sc, sample_weight=ws)),
                                           np.float64(mt.label_ranking_loss(ind, sc, sample_weight=ws)))
    for avg in ("macro", "weighted"):
        parts[f"auc_ovo:{avg}"] = _h(np.float64(mt.roc_auc_score(yt[:600], P[:600], multi_class="ovo", average=avg)))
    parts["auc"] = _h(np.float64(mt.auc(*[np.asarray(a) for a in mt.roc_curve(bt, s)[:2]])))
    return _fit(parts)


@lane("x-metrics-cluster")
def _(ml, X, yc, yr, Xh=None):
    """The clustering scores the metrics lane added. The labelings are fixed
    host comparisons of fixture columns (no fit); the dispersion scores read
    fixture features through the device centroid folds."""
    mt = ml.metrics
    n = 3000
    yt, yp = _metrics_labels(X, n)
    eight = ((X[:n, 5] > 0).astype(np.int32) + 2 * (X[:n, 6] > 0).astype(np.int32)
             + 4 * (X[:n, 7] > 0).astype(np.int32)).astype(np.int32)
    feats = np.ascontiguousarray(X[:n, :6]).astype(np.float32)
    parts = {}
    for method in ("min", "geometric", "arithmetic", "max"):
        parts[f"nmi:{method}"] = _h(np.float64(mt.normalized_mutual_info_score(yt, eight, average_method=method)))
        parts[f"ami:{method}"] = _h(np.float64(mt.adjusted_mutual_info_score(yt, eight, average_method=method)))
    parts["ami_small"] = _h(np.float64(mt.adjusted_mutual_info_score(yt[:40], yp[:40])))
    parts["contingency"] = _h(np.asarray(mt.contingency_matrix(yt, eight)),
                              np.asarray(mt.contingency_matrix(yt, eight, eps=0.5)))
    parts["pair_confusion"] = _h(np.asarray(mt.pair_confusion_matrix(yt, eight)))
    parts["calinski_harabasz"] = _h(np.float64(mt.calinski_harabasz_score(feats, eight)),
                                    np.float64(mt.calinski_harabasz_score(feats, yt)))
    parts["davies_bouldin"] = _h(np.float64(mt.davies_bouldin_score(feats, eight)),
                                 np.float64(mt.davies_bouldin_score(feats, yt)))
    return _fit(parts)
