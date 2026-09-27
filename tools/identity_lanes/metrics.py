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
