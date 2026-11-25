# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# ONE LANE PER PUBLIC METRIC (mojolearn.metrics.*), sub-lane vdv-cov-func, 2026-11-24.
#
# Each lane builds the bounded metric fixture (`_functions_metrics_fx`: 2000
# rows, built afresh per lane, nothing shared), calls ONE public metric through
# `ml.metrics` with no device= (the harness's view supplies the side) and hands
# the call to `_functions_resident` (tools/identity_lanes/functions.py): the
# host-input digest is the answer; on the GPU side the same call inside a
# `resident()` scope, twice on the same buffers, must give the same bytes and
# read the store in place. The x-metrics-* and metrics-* lanes stay as they are
# (their recorded digests do not move); these lanes add the per-function cell
# and the device-resident proposition.


def _functions_metrics_fx(X, n=2000):
    """The metric fixture: labels, scores, probabilities, weights, regression
    targets and ranking matrices, every one C-contiguous in the dtype the
    metric reads (int32 labels, float32 values), built once per lane so the
    resident run hands the product the same buffers twice."""
    yt = _functions_classes(X, n, 4, "fm:yt")
    flip = (_functions_cont(X, n, 5, "fm:flip") > np.float32(0.6)).astype(np.int32)
    yp = np.ascontiguousarray(((yt + flip) % 4).astype(np.int32))
    bt = np.ascontiguousarray((yt % 2).astype(np.int32))
    bp = np.ascontiguousarray((yp % 2).astype(np.int32))
    s = np.ascontiguousarray(np.round((_functions_cont(X, n, 6, "fm:s") + np.float32(0.5) * bt) * np.float32(8))
                             * np.float32(0.125)).astype(np.float32)
    raw = np.stack([_functions_pos(X, n, 3 + c, f"fm:p{c}") for c in range(4)], axis=1).astype(np.float64)  # glue: 4 columns
    P = np.ascontiguousarray((raw / raw.sum(axis=1, keepdims=True)).astype(np.float32))
    pb = np.ascontiguousarray((P[:, 1] / (P[:, 0] + P[:, 1])).astype(np.float32))
    w = _hw((n,), "fm:w", 0.0, 3.0)
    w[::11] = np.float32(0.0)
    y = _functions_cont(X, n, 4, "fm:y")
    pr = np.ascontiguousarray((y * np.float32(0.75) + _functions_cont(X, n, 5, "fm:pr") * np.float32(0.5)).astype(np.float32))
    Y2 = np.ascontiguousarray(np.stack([y, _functions_cont(X, n, 6, "fm:y2")], axis=1).astype(np.float32))
    P2 = np.ascontiguousarray(np.stack([pr, _functions_cont(X, n, 7, "fm:p2")], axis=1).astype(np.float32))
    m = _functions_mat(X, 200, 6, "fm:rank")
    rel = np.ascontiguousarray(np.round(np.abs(m) * np.float32(4)).astype(np.float32))
    ind = (m > np.float32(0.1)).astype(np.float32)
    ind[:, 0] = np.float32(1.0)
    ind[:, 1] = np.float32(0.0)
    sc = np.ascontiguousarray(np.round(_functions_mat(X, 200, 6, "fm:sc", col0=7) * np.float32(4)) * np.float32(0.25)).astype(np.float32)
    return dict(yt=yt, yp=yp, bt=bt, bp=bp, s=s, P=P, pb=pb, w=np.ascontiguousarray(w), ws=np.ascontiguousarray(w[:200]),
                y=y, pr=pr, Y2=Y2, P2=P2, pos_y=_functions_pos(X, n, 4, "fm:py"), pos_p=_functions_pos(X, n, 5, "fm:pp"),
                rel=rel, ind=np.ascontiguousarray(ind), sc=sc,
                feats=_functions_mat(X, n, 6, "fm:feats"), eight=_functions_classes(X, n, 8, "fm:eight"))


# ---------------------------------------------------------------- classification

@lane("fn-metrics-accuracy-score")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    mt = ml.metrics
    return _functions_resident(ml, "metrics.accuracy_score", lambda p: dict(
        u=_functions_d(mt.accuracy_score(p(f["yt"]), p(f["yp"]))),
        w=_functions_d(mt.accuracy_score(p(f["yt"]), p(f["yp"]), normalize=False, sample_weight=p(f["w"])))))


@lane("fn-metrics-balanced-accuracy-score")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    mt = ml.metrics
    return _functions_resident(ml, "metrics.balanced_accuracy_score", lambda p: dict(
        u=_functions_d(mt.balanced_accuracy_score(p(f["yt"]), p(f["yp"]))),
        w=_functions_d(mt.balanced_accuracy_score(p(f["yt"]), p(f["yp"]), sample_weight=p(f["w"]), adjusted=True))))


@lane("fn-metrics-class-likelihood-ratios")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    mt = ml.metrics
    return _functions_resident(ml, "metrics.class_likelihood_ratios", lambda p: dict(
        u=_functions_d(mt.class_likelihood_ratios(p(f["bt"]), p(f["bp"]))),
        w=_functions_d(mt.class_likelihood_ratios(p(f["bt"]), p(f["bp"]), sample_weight=p(f["w"])))))


@lane("fn-metrics-classification-report")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    mt = ml.metrics
    return _functions_resident(ml, "metrics.classification_report", lambda p: dict(
        u=_functions_d(mt.classification_report(p(f["yt"]), p(f["yp"]), output_dict=True, zero_division=0)),
        w=_functions_d(mt.classification_report(p(f["yt"]), p(f["yp"]), sample_weight=p(f["w"]), output_dict=True,
                                                zero_division=0))))


@lane("fn-metrics-cohen-kappa-score")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    mt = ml.metrics
    return _functions_resident(ml, "metrics.cohen_kappa_score", lambda p: dict(
        u=_functions_d(mt.cohen_kappa_score(p(f["yt"]), p(f["yp"]))),
        quadratic=_functions_d(mt.cohen_kappa_score(p(f["yt"]), p(f["yp"]), weights="quadratic", sample_weight=p(f["w"])))))


@lane("fn-metrics-confusion-matrix")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    mt = ml.metrics
    return _functions_resident(ml, "metrics.confusion_matrix", lambda p: dict(
        u=_functions_d(mt.confusion_matrix(p(f["yt"]), p(f["yp"]))),
        w=_functions_d(mt.confusion_matrix(p(f["yt"]), p(f["yp"]), sample_weight=p(f["w"]), normalize="true"))))


@lane("fn-metrics-multilabel-confusion-matrix")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    mt = ml.metrics
    return _functions_resident(ml, "metrics.multilabel_confusion_matrix", lambda p: dict(
        u=_functions_d(mt.multilabel_confusion_matrix(p(f["yt"]), p(f["yp"]))),
        w=_functions_d(mt.multilabel_confusion_matrix(p(f["yt"]), p(f["yp"]), sample_weight=p(f["w"]), labels=[3, 0, 1]))))


@lane("fn-metrics-f1-score")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    mt = ml.metrics
    return _functions_resident(ml, "metrics.f1_score", lambda p: dict(
        binary=_functions_d(mt.f1_score(p(f["bt"]), p(f["bp"]))),
        macro=_functions_d(mt.f1_score(p(f["yt"]), p(f["yp"]), average="macro", sample_weight=p(f["w"]))),
        none=_functions_d(mt.f1_score(p(f["yt"]), p(f["yp"]), average=None))))


@lane("fn-metrics-fbeta-score")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    mt = ml.metrics
    return _functions_resident(ml, "metrics.fbeta_score", lambda p: dict(
        binary=_functions_d(mt.fbeta_score(p(f["bt"]), p(f["bp"]), beta=2.0)),
        weighted=_functions_d(mt.fbeta_score(p(f["yt"]), p(f["yp"]), beta=0.5, average="weighted", sample_weight=p(f["w"])))))


@lane("fn-metrics-precision-score")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    mt = ml.metrics
    return _functions_resident(ml, "metrics.precision_score", lambda p: dict(
        binary=_functions_d(mt.precision_score(p(f["bt"]), p(f["bp"]))),
        micro=_functions_d(mt.precision_score(p(f["yt"]), p(f["yp"]), average="micro", sample_weight=p(f["w"])))))


@lane("fn-metrics-recall-score")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    mt = ml.metrics
    return _functions_resident(ml, "metrics.recall_score", lambda p: dict(
        binary=_functions_d(mt.recall_score(p(f["bt"]), p(f["bp"]))),
        macro=_functions_d(mt.recall_score(p(f["yt"]), p(f["yp"]), average="macro", sample_weight=p(f["w"])))))


@lane("fn-metrics-precision-recall-fscore-support")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    mt = ml.metrics
    return _functions_resident(ml, "metrics.precision_recall_fscore_support", lambda p: dict(
        none=_functions_d(mt.precision_recall_fscore_support(p(f["yt"]), p(f["yp"]), beta=0.5)),
        macro=_functions_d(mt.precision_recall_fscore_support(p(f["yt"]), p(f["yp"]), average="macro",
                                                              sample_weight=p(f["w"])))))


@lane("fn-metrics-jaccard-score")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    mt = ml.metrics
    return _functions_resident(ml, "metrics.jaccard_score", lambda p: dict(
        binary=_functions_d(mt.jaccard_score(p(f["bt"]), p(f["bp"]))),
        macro=_functions_d(mt.jaccard_score(p(f["yt"]), p(f["yp"]), average="macro", sample_weight=p(f["w"])))))


@lane("fn-metrics-hamming-loss")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    mt = ml.metrics
    return _functions_resident(ml, "metrics.hamming_loss", lambda p: dict(
        u=_functions_d(mt.hamming_loss(p(f["yt"]), p(f["yp"]))),
        w=_functions_d(mt.hamming_loss(p(f["yt"]), p(f["yp"]), sample_weight=p(f["w"])))))


@lane("fn-metrics-zero-one-loss")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    mt = ml.metrics
    return _functions_resident(ml, "metrics.zero_one_loss", lambda p: dict(
        u=_functions_d(mt.zero_one_loss(p(f["yt"]), p(f["yp"]))),
        w=_functions_d(mt.zero_one_loss(p(f["yt"]), p(f["yp"]), normalize=False, sample_weight=p(f["w"])))))


@lane("fn-metrics-matthews-corrcoef")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    mt = ml.metrics
    return _functions_resident(ml, "metrics.matthews_corrcoef", lambda p: dict(
        u=_functions_d(mt.matthews_corrcoef(p(f["yt"]), p(f["yp"]))),
        w=_functions_d(mt.matthews_corrcoef(p(f["yt"]), p(f["yp"]), sample_weight=p(f["w"])))))


@lane("fn-metrics-hinge-loss")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    mt = ml.metrics
    return _functions_resident(ml, "metrics.hinge_loss", lambda p: dict(
        binary=_functions_d(mt.hinge_loss(p(f["bt"]), p(f["s"]))),
        multi=_functions_d(mt.hinge_loss(p(f["yt"]), p(f["P"]), sample_weight=p(f["w"])))))


@lane("fn-metrics-log-loss")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    mt = ml.metrics
    return _functions_resident(ml, "metrics.log_loss", lambda p: dict(
        u=_functions_d(mt.log_loss(p(f["yt"]), p(f["P"]))),
        w=_functions_d(mt.log_loss(p(f["yt"]), p(f["P"]), normalize=False, sample_weight=p(f["w"])))))


@lane("fn-metrics-brier-score-loss")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    mt = ml.metrics
    return _functions_resident(ml, "metrics.brier_score_loss", lambda p: dict(
        binary=_functions_d(mt.brier_score_loss(p(f["bt"]), p(f["pb"]))),
        multi=_functions_d(mt.brier_score_loss(p(f["yt"]), p(f["P"]), sample_weight=p(f["w"])))))


@lane("fn-metrics-d2-brier-score")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    mt = ml.metrics
    return _functions_resident(ml, "metrics.d2_brier_score", lambda p: dict(
        binary=_functions_d(mt.d2_brier_score(p(f["bt"]), p(f["pb"]))),
        multi=_functions_d(mt.d2_brier_score(p(f["yt"]), p(f["P"]), sample_weight=p(f["w"])))))


@lane("fn-metrics-d2-log-loss-score")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    mt = ml.metrics
    return _functions_resident(ml, "metrics.d2_log_loss_score", lambda p: dict(
        u=_functions_d(mt.d2_log_loss_score(p(f["yt"]), p(f["P"]))),
        w=_functions_d(mt.d2_log_loss_score(p(f["yt"]), p(f["P"]), sample_weight=p(f["w"])))))


@lane("fn-metrics-top-k-accuracy-score")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    mt = ml.metrics
    return _functions_resident(ml, "metrics.top_k_accuracy_score", lambda p: dict(
        k2=_functions_d(mt.top_k_accuracy_score(p(f["yt"]), p(f["P"]), k=2)),
        k3=_functions_d(mt.top_k_accuracy_score(p(f["yt"]), p(f["P"]), k=3, normalize=False, sample_weight=p(f["w"]))),
        binary=_functions_d(mt.top_k_accuracy_score(p(f["bt"]), p(f["pb"]), k=1))))


# ---------------------------------------------------------------- ranking and curves

@lane("fn-metrics-roc-auc-score")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    mt = ml.metrics
    return _functions_resident(ml, "metrics.roc_auc_score", lambda p: dict(
        binary=_functions_d(mt.roc_auc_score(p(f["bt"]), p(f["s"])), mt.roc_auc_score(p(f["bt"]), p(f["s"]), max_fpr=0.3)),
        ovr=_functions_d(mt.roc_auc_score(p(f["yt"]), p(f["P"]), multi_class="ovr", sample_weight=p(f["w"])))))


@lane("fn-metrics-average-precision-score")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    mt = ml.metrics
    return _functions_resident(ml, "metrics.average_precision_score", lambda p: dict(
        binary=_functions_d(mt.average_precision_score(p(f["bt"]), p(f["s"]))),
        macro=_functions_d(mt.average_precision_score(p(f["yt"]), p(f["P"]), sample_weight=p(f["w"])))))


@lane("fn-metrics-roc-curve")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    mt = ml.metrics
    return _functions_resident(ml, "metrics.roc_curve", lambda p: dict(
        u=_functions_d(mt.roc_curve(p(f["bt"]), p(f["s"]))),
        w=_functions_d(mt.roc_curve(p(f["bt"]), p(f["s"]), sample_weight=p(f["w"]), drop_intermediate=False))))


@lane("fn-metrics-precision-recall-curve")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    mt = ml.metrics
    return _functions_resident(ml, "metrics.precision_recall_curve", lambda p: dict(
        u=_functions_d(mt.precision_recall_curve(p(f["bt"]), p(f["s"]))),
        w=_functions_d(mt.precision_recall_curve(p(f["bt"]), p(f["s"]), sample_weight=p(f["w"]), drop_intermediate=True))))


@lane("fn-metrics-det-curve")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    mt = ml.metrics
    return _functions_resident(ml, "metrics.det_curve", lambda p: dict(
        u=_functions_d(mt.det_curve(p(f["bt"]), p(f["s"]))),
        w=_functions_d(mt.det_curve(p(f["bt"]), p(f["s"]), sample_weight=p(f["w"]), drop_intermediate=True))))


@lane("fn-metrics-auc")
def _(ml, X, yc, yr, Xh=None):
    """auc over a monotone x (sorted positive values) and a positive y."""
    x = np.ascontiguousarray(np.sort(_functions_pos(X, 512, 3, "fm:auc:x")))
    y = _functions_pos(X, 512, 4, "fm:auc:y")
    return _functions_resident(ml, "metrics.auc", lambda p: dict(auc=_functions_d(ml.metrics.auc(p(x), p(y)))))


@lane("fn-metrics-dcg-score")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    mt = ml.metrics
    return _functions_resident(ml, "metrics.dcg_score", lambda p: dict(
        u=_functions_d(mt.dcg_score(p(f["rel"]), p(f["sc"]))),
        k3=_functions_d(mt.dcg_score(p(f["rel"]), p(f["sc"]), k=3, sample_weight=p(f["ws"]), ignore_ties=True))))


@lane("fn-metrics-ndcg-score")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    mt = ml.metrics
    return _functions_resident(ml, "metrics.ndcg_score", lambda p: dict(
        u=_functions_d(mt.ndcg_score(p(f["rel"]), p(f["sc"]))),
        k3=_functions_d(mt.ndcg_score(p(f["rel"]), p(f["sc"]), k=3, sample_weight=p(f["ws"]), ignore_ties=True))))


@lane("fn-metrics-coverage-error")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    mt = ml.metrics
    return _functions_resident(ml, "metrics.coverage_error", lambda p: dict(
        u=_functions_d(mt.coverage_error(p(f["ind"]), p(f["sc"]))),
        w=_functions_d(mt.coverage_error(p(f["ind"]), p(f["sc"]), sample_weight=p(f["ws"])))))


@lane("fn-metrics-label-ranking-average-precision-score")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    mt = ml.metrics
    return _functions_resident(ml, "metrics.label_ranking_average_precision_score", lambda p: dict(
        u=_functions_d(mt.label_ranking_average_precision_score(p(f["ind"]), p(f["sc"]))),
        w=_functions_d(mt.label_ranking_average_precision_score(p(f["ind"]), p(f["sc"]), sample_weight=p(f["ws"])))))


@lane("fn-metrics-label-ranking-loss")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    mt = ml.metrics
    return _functions_resident(ml, "metrics.label_ranking_loss", lambda p: dict(
        u=_functions_d(mt.label_ranking_loss(p(f["ind"]), p(f["sc"]))),
        w=_functions_d(mt.label_ranking_loss(p(f["ind"]), p(f["sc"]), sample_weight=p(f["ws"])))))


# ---------------------------------------------------------------- regression

@lane("fn-metrics-mean-squared-error")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    mt = ml.metrics
    return _functions_resident(ml, "metrics.mean_squared_error", lambda p: dict(
        u=_functions_d(mt.mean_squared_error(p(f["y"]), p(f["pr"]))),
        raw=_functions_d(mt.mean_squared_error(p(f["Y2"]), p(f["P2"]), sample_weight=p(f["w"]), multioutput="raw_values"))))


@lane("fn-metrics-root-mean-squared-error")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    mt = ml.metrics
    return _functions_resident(ml, "metrics.root_mean_squared_error", lambda p: dict(
        u=_functions_d(mt.root_mean_squared_error(p(f["y"]), p(f["pr"]))),
        raw=_functions_d(mt.root_mean_squared_error(p(f["Y2"]), p(f["P2"]), sample_weight=p(f["w"]), multioutput="raw_values"))))


@lane("fn-metrics-mean-absolute-error")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    mt = ml.metrics
    return _functions_resident(ml, "metrics.mean_absolute_error", lambda p: dict(
        u=_functions_d(mt.mean_absolute_error(p(f["y"]), p(f["pr"]))),
        raw=_functions_d(mt.mean_absolute_error(p(f["Y2"]), p(f["P2"]), sample_weight=p(f["w"]), multioutput="raw_values"))))


@lane("fn-metrics-mean-absolute-percentage-error")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    mt = ml.metrics
    return _functions_resident(ml, "metrics.mean_absolute_percentage_error", lambda p: dict(
        u=_functions_d(mt.mean_absolute_percentage_error(p(f["pos_y"]), p(f["pos_p"]))),
        w=_functions_d(mt.mean_absolute_percentage_error(p(f["pos_y"]), p(f["pos_p"]), sample_weight=p(f["w"])))))


@lane("fn-metrics-median-absolute-error")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    mt = ml.metrics
    return _functions_resident(ml, "metrics.median_absolute_error", lambda p: dict(
        u=_functions_d(mt.median_absolute_error(p(f["y"]), p(f["pr"]))),
        raw=_functions_d(mt.median_absolute_error(p(f["Y2"]), p(f["P2"]), multioutput="raw_values", sample_weight=p(f["w"])))))


@lane("fn-metrics-max-error")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    return _functions_resident(ml, "metrics.max_error", lambda p: dict(
        u=_functions_d(ml.metrics.max_error(p(f["y"]), p(f["pr"])))))


@lane("fn-metrics-mean-squared-log-error")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    mt = ml.metrics
    return _functions_resident(ml, "metrics.mean_squared_log_error", lambda p: dict(
        u=_functions_d(mt.mean_squared_log_error(p(f["pos_y"]), p(f["pos_p"]))),
        w=_functions_d(mt.mean_squared_log_error(p(f["pos_y"]), p(f["pos_p"]), sample_weight=p(f["w"])))))


@lane("fn-metrics-root-mean-squared-log-error")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    mt = ml.metrics
    return _functions_resident(ml, "metrics.root_mean_squared_log_error", lambda p: dict(
        u=_functions_d(mt.root_mean_squared_log_error(p(f["pos_y"]), p(f["pos_p"]))),
        w=_functions_d(mt.root_mean_squared_log_error(p(f["pos_y"]), p(f["pos_p"]), sample_weight=p(f["w"])))))


@lane("fn-metrics-mean-pinball-loss")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    mt = ml.metrics
    return _functions_resident(ml, "metrics.mean_pinball_loss", lambda p: dict(
        u=_functions_d(mt.mean_pinball_loss(p(f["y"]), p(f["pr"]))),
        raw=_functions_d(mt.mean_pinball_loss(p(f["Y2"]), p(f["P2"]), alpha=0.3, sample_weight=p(f["w"]),
                                              multioutput="raw_values"))))


@lane("fn-metrics-mean-poisson-deviance")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    mt = ml.metrics
    return _functions_resident(ml, "metrics.mean_poisson_deviance", lambda p: dict(
        u=_functions_d(mt.mean_poisson_deviance(p(f["pos_y"]), p(f["pos_p"]))),
        w=_functions_d(mt.mean_poisson_deviance(p(f["pos_y"]), p(f["pos_p"]), sample_weight=p(f["w"])))))


@lane("fn-metrics-mean-gamma-deviance")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    mt = ml.metrics
    return _functions_resident(ml, "metrics.mean_gamma_deviance", lambda p: dict(
        u=_functions_d(mt.mean_gamma_deviance(p(f["pos_y"]), p(f["pos_p"]))),
        w=_functions_d(mt.mean_gamma_deviance(p(f["pos_y"]), p(f["pos_p"]), sample_weight=p(f["w"])))))


@lane("fn-metrics-mean-tweedie-deviance")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    mt = ml.metrics
    return _functions_resident(ml, "metrics.mean_tweedie_deviance", lambda p: dict(
        p0=_functions_d(mt.mean_tweedie_deviance(p(f["pos_y"]), p(f["pos_p"]))),
        p15=_functions_d(mt.mean_tweedie_deviance(p(f["pos_y"]), p(f["pos_p"]), power=1.5, sample_weight=p(f["w"])))))


@lane("fn-metrics-d2-tweedie-score")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    mt = ml.metrics
    return _functions_resident(ml, "metrics.d2_tweedie_score", lambda p: dict(
        p0=_functions_d(mt.d2_tweedie_score(p(f["pos_y"]), p(f["pos_p"]))),
        p15=_functions_d(mt.d2_tweedie_score(p(f["pos_y"]), p(f["pos_p"]), power=1.5, sample_weight=p(f["w"])))))


@lane("fn-metrics-d2-absolute-error-score")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    mt = ml.metrics
    return _functions_resident(ml, "metrics.d2_absolute_error_score", lambda p: dict(
        u=_functions_d(mt.d2_absolute_error_score(p(f["y"]), p(f["pr"]))),
        raw=_functions_d(mt.d2_absolute_error_score(p(f["Y2"]), p(f["P2"]), sample_weight=p(f["w"]), multioutput="raw_values"))))


@lane("fn-metrics-d2-pinball-score")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    mt = ml.metrics
    return _functions_resident(ml, "metrics.d2_pinball_score", lambda p: dict(
        u=_functions_d(mt.d2_pinball_score(p(f["y"]), p(f["pr"]))),
        raw=_functions_d(mt.d2_pinball_score(p(f["Y2"]), p(f["P2"]), alpha=0.8, sample_weight=p(f["w"]),
                                             multioutput="raw_values"))))


@lane("fn-metrics-explained-variance-score")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    mt = ml.metrics
    return _functions_resident(ml, "metrics.explained_variance_score", lambda p: dict(
        u=_functions_d(mt.explained_variance_score(p(f["y"]), p(f["pr"]))),
        vw=_functions_d(mt.explained_variance_score(p(f["Y2"]), p(f["P2"]), sample_weight=p(f["w"]),
                                                    multioutput="variance_weighted", force_finite=False))))


@lane("fn-metrics-r2-score")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    mt = ml.metrics
    return _functions_resident(ml, "metrics.r2_score", lambda p: dict(
        u=_functions_d(mt.r2_score(p(f["y"]), p(f["pr"]))),
        raw=_functions_d(mt.r2_score(p(f["Y2"]), p(f["P2"]), sample_weight=p(f["w"]), multioutput="raw_values",
                                     force_finite=False))))


# ---------------------------------------------------------------- clustering and information

@lane("fn-metrics-adjusted-mutual-info-score")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    mt = ml.metrics
    return _functions_resident(ml, "metrics.adjusted_mutual_info_score", lambda p: dict(
        arithmetic=_functions_d(mt.adjusted_mutual_info_score(p(f["yt"]), p(f["eight"]))),
        geometric=_functions_d(mt.adjusted_mutual_info_score(p(f["yt"]), p(f["eight"]), average_method="geometric"))))


@lane("fn-metrics-normalized-mutual-info-score")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    mt = ml.metrics
    return _functions_resident(ml, "metrics.normalized_mutual_info_score", lambda p: dict(
        arithmetic=_functions_d(mt.normalized_mutual_info_score(p(f["yt"]), p(f["eight"]))),
        maximum=_functions_d(mt.normalized_mutual_info_score(p(f["yt"]), p(f["eight"]), average_method="max"))))


@lane("fn-metrics-mutual-info-score")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    return _functions_resident(ml, "metrics.mutual_info_score", lambda p: dict(
        mi=_functions_d(ml.metrics.mutual_info_score(p(f["yt"]), p(f["eight"])))))


@lane("fn-metrics-adjusted-rand-score")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    return _functions_resident(ml, "metrics.adjusted_rand_score", lambda p: dict(
        ari=_functions_d(ml.metrics.adjusted_rand_score(p(f["yt"]), p(f["eight"])))))


@lane("fn-metrics-rand-score")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    return _functions_resident(ml, "metrics.rand_score", lambda p: dict(
        ri=_functions_d(ml.metrics.rand_score(p(f["yt"]), p(f["eight"])))))


@lane("fn-metrics-fowlkes-mallows-score")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    return _functions_resident(ml, "metrics.fowlkes_mallows_score", lambda p: dict(
        fm=_functions_d(ml.metrics.fowlkes_mallows_score(p(f["yt"]), p(f["eight"])))))


@lane("fn-metrics-homogeneity-score")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    return _functions_resident(ml, "metrics.homogeneity_score", lambda p: dict(
        h=_functions_d(ml.metrics.homogeneity_score(p(f["yt"]), p(f["eight"])))))


@lane("fn-metrics-completeness-score")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    return _functions_resident(ml, "metrics.completeness_score", lambda p: dict(
        c=_functions_d(ml.metrics.completeness_score(p(f["yt"]), p(f["eight"])))))


@lane("fn-metrics-v-measure-score")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    mt = ml.metrics
    return _functions_resident(ml, "metrics.v_measure_score", lambda p: dict(
        beta1=_functions_d(mt.v_measure_score(p(f["yt"]), p(f["eight"]))),
        beta2=_functions_d(mt.v_measure_score(p(f["yt"]), p(f["eight"]), beta=2.0))))


@lane("fn-metrics-homogeneity-completeness-v-measure")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    mt = ml.metrics
    return _functions_resident(ml, "metrics.homogeneity_completeness_v_measure", lambda p: dict(
        beta1=_functions_d(mt.homogeneity_completeness_v_measure(p(f["yt"]), p(f["eight"]))),
        beta05=_functions_d(mt.homogeneity_completeness_v_measure(p(f["yt"]), p(f["eight"]), beta=0.5))))


@lane("fn-metrics-contingency-matrix")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    mt = ml.metrics
    return _functions_resident(ml, "metrics.contingency_matrix", lambda p: dict(
        counts=_functions_d(mt.contingency_matrix(p(f["yt"]), p(f["eight"]))),
        eps=_functions_d(mt.contingency_matrix(p(f["yt"]), p(f["eight"]), eps=0.5))))


@lane("fn-metrics-pair-confusion-matrix")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    return _functions_resident(ml, "metrics.pair_confusion_matrix", lambda p: dict(
        pairs=_functions_d(ml.metrics.pair_confusion_matrix(p(f["yt"]), p(f["eight"])))))


@lane("fn-metrics-entropy")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    mt = ml.metrics
    return _functions_resident(ml, "metrics.entropy", lambda p: dict(
        nats=_functions_d(mt.entropy(p(f["eight"]))),
        base2=_functions_d(mt.entropy(p(f["eight"]), base=2))))


@lane("fn-metrics-kl-divergence")
def _(ml, X, yc, yr, Xh=None):
    """kl_divergence of two exact probability vectors of 64 (the harness's
    `_exact_prob`, permuted by the fixture-independent hashed stream; the
    fixture enters through a third vector built from it)."""
    P = _exact_prob(64, "fn-kl:P")
    Q = _exact_prob(64, "fn-kl:Q")
    raw = _functions_pos(X, 64, 3, "fn-kl:R").astype(np.float64)
    R = np.ascontiguousarray((raw / raw.sum()).astype(np.float32))
    return _functions_resident(ml, "metrics.kl_divergence", lambda p: dict(
        pq=_functions_d(ml.metrics.kl_divergence(p(P), p(Q))),
        rq=_functions_d(ml.metrics.kl_divergence(p(R), p(Q)))))


@lane("fn-metrics-calinski-harabasz-score")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    return _functions_resident(ml, "metrics.calinski_harabasz_score", lambda p: dict(
        ch=_functions_d(ml.metrics.calinski_harabasz_score(p(f["feats"]), p(f["eight"])))))


@lane("fn-metrics-davies-bouldin-score")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X)
    return _functions_resident(ml, "metrics.davies_bouldin_score", lambda p: dict(
        db=_functions_d(ml.metrics.davies_bouldin_score(p(f["feats"]), p(f["eight"])))))


@lane("fn-metrics-silhouette-score")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X, 1000)
    mt = ml.metrics
    # The subsample is the caller's own (the first 400 rows): silhouette_score refuses sample_size= / random_state=
    # (bindings/mcls_cluster_door.mojo MCLS_OPN_FAMILY_MCLS_SILHOUETTE_SCORE, "subsample X yourself"), so the old
    # `sampled` call refused the whole lane on both sides (lane w3-fix-9-s5, 2026-11-25).
    fs, ls = np.ascontiguousarray(f["feats"][:400]), np.ascontiguousarray(f["eight"][:400])
    return _functions_resident(ml, "metrics.silhouette_score", lambda p: dict(
        full=_functions_d(mt.silhouette_score(p(f["feats"]), p(f["eight"]))),
        sampled=_functions_d(mt.silhouette_score(p(fs), p(ls)))))


@lane("fn-metrics-silhouette-samples")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_metrics_fx(X, 1000)
    return _functions_resident(ml, "metrics.silhouette_samples", lambda p: dict(
        s=_functions_d(ml.metrics.silhouette_samples(p(f["feats"]), p(f["eight"])))))


@lane("fn-metrics-trustworthiness")
def _(ml, X, yc, yr, Xh=None):
    """trustworthiness of a 2-column view of 512 rows of 4 features."""
    hi = _functions_mat(X, 512, 4, "fn-trust:hi")
    lo = np.ascontiguousarray(hi[:, :2] + _hw((512, 2), "fn-trust:lo", -0.05, 0.05))
    return _functions_resident(ml, "metrics.trustworthiness", lambda p: dict(
        k5=_functions_d(ml.metrics.trustworthiness(p(hi), p(lo), n_neighbors=5))))


_batch_decl(_FUNCTIONS_BATCH_NA, *sorted(n for n in LANES if n.startswith("fn-metrics-")))
