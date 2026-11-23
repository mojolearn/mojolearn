# SPDX-License-Identifier: Apache-2.0
"""THE SEARCH SCORERS, TYPED (lane mf-metric4, 2026-11-23): the scorer names
`get_scorer` accepts that a search over native members scores in its RUN
(core/msel_search_run.mojo), one code each, and the one function that turns
a fold side's truth and the member's response into the scorer's value. No
Python: no Python object parameter, field, local or import.

A scorer is (the metric, its fixed options, its sign, the member response it
reads), as python/mojolearn/model_selection.py `_scorer_table` builds it. The
value comes from THE METRIC'S OWN PROGRAM BUILDER, the one its public door
calls (bindings/metrics_reg_run.mojo `reg_metric_run`,
bindings/mcls_classify_run.mojo `label_metric_run` / `log_loss_run`,
bindings/mcls_ranking_run.mojo `roc_auc_run`), run through the operation's
x_metrics route (`MetricRoute`, `metric_inputs_run`). One body per metric.

The member response (`msel_scorer_op`) is the scorer's `response_method`:
`predict` (class codes for the label metrics and accuracy, float32 targets
for the regression metrics), `predict_proba` (rows x classes), or, for
'roc_auc', `decision_function` when the estimator's class has it, else
`predict_proba` (two codes: the UNPACK reads the class's methods once, as
the scorer's `_scorer_response` does). The scorers of `_BINARY_PROBA`
('roc_auc', 'neg_log_loss') read `predict_proba[:, 1]` of a two-class
member; the RUN hands that one column.

THE BINARY LABEL SCORERS (lane mf-search4b): 'f1', 'precision', 'recall'
and 'jaccard' (average 'binary', pos_label=1) read their positive label as
a CLASS CODE the UNPACK maps from the label value (the search plan's
`positives`), through bindings/mcls_classify_run.mojo `label_binary_run`.

THE CURVE, PROBABILITY, TOP-K AND CLUSTERING SCORERS (lane mf-score5):
'average_precision' (its pos_label=1 a class code, the scorer's response
rule: decision_function when the class has it, else predict_proba; a binary
member hands the positive's column, a decision negated when the positive is
the first class, as `_get_response_values` does; more than two classes
take the one-vs-rest 'macro' curves), 'neg_brier_score' (predict_proba;
the label marks of `_pos_label(None, ...)` read at UNPACK), 'top_k_accuracy'
(k=2, the same response rule), and 'normalized_mutual_info_score' /
'adjusted_mutual_info_score' (predict labels against the true labels, a
classifier member's codes) run THE BUILDERS THEIR DOORS CALL
(bindings/metrics_exp_curve_run.mojo `average_precision_run` / `top_k_run`,
bindings/metrics_exp_rank_run.mojo `brier_run`,
bindings/metrics_exp_common_run.mojo `cluster_mi_run`).

THE METRIC WARNINGS (lane mf-score5): every builder answers the warnings its
public metric raises (ill-defined precision / recall / f-score / jaccard,
y_pred classes not in y_true, an r2 / d2 of fewer than two samples, k >=
n_classes); `msel_scorer_value` appends them as notice words
(core/member_notices.mojo MEMBER_NOTICE_METRIC_*), raised at PACK.

Not served here (the search keeps its units for them, with the reason):
'rand_score', 'adjusted_rand_score', 'mutual_info_score',
'homogeneity_score', 'completeness_score', 'v_measure_score' and
'fowlkes_mallows_score': their public metrics call the metrics binding's
own entries (bindings/_mojolearn_metrics(_host).mojo, through the Python
module); no typed entry of theirs exists for a RUN to call.
"""
from bindings.arena_prog_run import MetricRoute
from bindings.metrics_reg_run import (
    reg_metric_run, REG_R2, REG_MSE, REG_MAE, REG_RMSE, REG_MSLE, REG_RMSLE, REG_MAPE, REG_MEDAE,
    REG_EXPLAINED_VARIANCE, REG_MAX_ERROR, REG_POISSON, REG_GAMMA, REG_D2_ABSOLUTE,
)
from bindings.mcls_classify_run import (
    label_metric_run, label_binary_run, log_loss_run, LBL_JACCARD, LBL_BALANCED, LBL_PRECISION, LBL_RECALL, LBL_F1, LBL_MCC,
    LBL_MICRO, LBL_MACRO, LBL_WEIGHTED,
)
from bindings.mcls_ranking_run import roc_auc_run, ROC_BINARY, ROC_OVR, ROC_OVR_WEIGHTED, ROC_OVO, ROC_OVO_WEIGHTED
from bindings.metrics_exp_curve_run import average_precision_run, top_k_run
from bindings.metrics_exp_rank_run import brier_run
from bindings.metrics_exp_common_run import cluster_mi_run
from core.member_notices import (
    member_notice_add, MEMBER_NOTICE_METRIC_PRF, MEMBER_NOTICE_METRIC_JACCARD, MEMBER_NOTICE_METRIC_BALANCED,
    MEMBER_NOTICE_METRIC_FEW, MEMBER_NOTICE_METRIC_TOP_K,
)
from bindings.arena_prog_run import metric_inputs_run
from core.labels_accuracy_plan import label_accuracy_plan
from core.member_abi import MEMBER_OP_PREDICT, MEMBER_OP_PREDICT_PROBA, MEMBER_OP_DECISION

#: The scorer codes (`MselSearchPlan.scoring` / `.scorers`): 0 the member's
#: own score (scoring=None), 1 'accuracy', then `_TABLE` order from 2.
comptime SCORER_MEMBER = 0
comptime SCORER_ACCURACY = 1

#: The families of `_TABLE`.
comptime _F_ACCURACY = 0
comptime _F_REG = 1
comptime _F_LABEL = 2
comptime _F_LOG_LOSS = 3
comptime _F_ROC = 4
comptime _F_BINARY = 5
comptime _F_AP = 6
comptime _F_BRIER = 7
comptime _F_TOP_K = 8
comptime _F_CLUSTER = 9
#: `_F_CLUSTER` metrics.
comptime _CL_NMI = 0
comptime _CL_AMI = 1
#: The top_k scorer's k (`top_k_accuracy_score`'s default; the scorer passes none).
comptime _TOP_K = 2
#: `MselSearchPlan.positives` words a scorer reads (`msel_scorer_positive_kind`).
comptime SCORER_POSITIVE_NONE = 0
comptime SCORER_POSITIVE_LABEL_ONE = 1   # the class code of the label 1 (-1: not a label)
comptime SCORER_POSITIVE_MARKS = 2       # bindings/metrics_exp_rank_run.mojo `brier_marks_positive`'s marks


@fieldwise_init
struct MselScorer(ImplicitlyCopyable, Movable):
    """One served scorer: its name, metric family, metric and averaging
    words, sign and member response. Resolved ONCE per RUN
    (`msel_scorer_row`), never per score."""
    var name: StaticString
    var family: Int
    var metric: Int
    var avg: Int
    var sign: Float64
    var op: Int


def _table() -> List[MselScorer]:
    """`_scorer_table`'s served rows, in a fixed order (code = index + 2)."""
    var t = List[MselScorer]()
    t.append(MselScorer("r2", _F_REG, REG_R2, 0, 1.0, MEMBER_OP_PREDICT))
    t.append(MselScorer("neg_mean_squared_error", _F_REG, REG_MSE, 0, -1.0, MEMBER_OP_PREDICT))
    t.append(MselScorer("neg_mean_absolute_error", _F_REG, REG_MAE, 0, -1.0, MEMBER_OP_PREDICT))
    t.append(MselScorer("neg_root_mean_squared_error", _F_REG, REG_RMSE, 0, -1.0, MEMBER_OP_PREDICT))
    t.append(MselScorer("neg_mean_squared_log_error", _F_REG, REG_MSLE, 0, -1.0, MEMBER_OP_PREDICT))
    t.append(MselScorer("neg_root_mean_squared_log_error", _F_REG, REG_RMSLE, 0, -1.0, MEMBER_OP_PREDICT))
    t.append(MselScorer("neg_mean_absolute_percentage_error", _F_REG, REG_MAPE, 0, -1.0, MEMBER_OP_PREDICT))
    t.append(MselScorer("neg_median_absolute_error", _F_REG, REG_MEDAE, 0, -1.0, MEMBER_OP_PREDICT))
    t.append(MselScorer("explained_variance", _F_REG, REG_EXPLAINED_VARIANCE, 0, 1.0, MEMBER_OP_PREDICT))
    t.append(MselScorer("max_error", _F_REG, REG_MAX_ERROR, 0, -1.0, MEMBER_OP_PREDICT))
    t.append(MselScorer("neg_mean_poisson_deviance", _F_REG, REG_POISSON, 0, -1.0, MEMBER_OP_PREDICT))
    t.append(MselScorer("neg_mean_gamma_deviance", _F_REG, REG_GAMMA, 0, -1.0, MEMBER_OP_PREDICT))
    t.append(MselScorer("d2_absolute_error_score", _F_REG, REG_D2_ABSOLUTE, 0, 1.0, MEMBER_OP_PREDICT))
    t.append(MselScorer("balanced_accuracy", _F_LABEL, LBL_BALANCED, 0, 1.0, MEMBER_OP_PREDICT))
    t.append(MselScorer("matthews_corrcoef", _F_LABEL, LBL_MCC, 0, 1.0, MEMBER_OP_PREDICT))
    t.append(MselScorer("precision_micro", _F_LABEL, LBL_PRECISION, LBL_MICRO, 1.0, MEMBER_OP_PREDICT))
    t.append(MselScorer("precision_macro", _F_LABEL, LBL_PRECISION, LBL_MACRO, 1.0, MEMBER_OP_PREDICT))
    t.append(MselScorer("precision_weighted", _F_LABEL, LBL_PRECISION, LBL_WEIGHTED, 1.0, MEMBER_OP_PREDICT))
    t.append(MselScorer("recall_micro", _F_LABEL, LBL_RECALL, LBL_MICRO, 1.0, MEMBER_OP_PREDICT))
    t.append(MselScorer("recall_macro", _F_LABEL, LBL_RECALL, LBL_MACRO, 1.0, MEMBER_OP_PREDICT))
    t.append(MselScorer("recall_weighted", _F_LABEL, LBL_RECALL, LBL_WEIGHTED, 1.0, MEMBER_OP_PREDICT))
    t.append(MselScorer("f1_micro", _F_LABEL, LBL_F1, LBL_MICRO, 1.0, MEMBER_OP_PREDICT))
    t.append(MselScorer("f1_macro", _F_LABEL, LBL_F1, LBL_MACRO, 1.0, MEMBER_OP_PREDICT))
    t.append(MselScorer("f1_weighted", _F_LABEL, LBL_F1, LBL_WEIGHTED, 1.0, MEMBER_OP_PREDICT))
    t.append(MselScorer("jaccard_micro", _F_LABEL, LBL_JACCARD, LBL_MICRO, 1.0, MEMBER_OP_PREDICT))
    t.append(MselScorer("jaccard_macro", _F_LABEL, LBL_JACCARD, LBL_MACRO, 1.0, MEMBER_OP_PREDICT))
    t.append(MselScorer("jaccard_weighted", _F_LABEL, LBL_JACCARD, LBL_WEIGHTED, 1.0, MEMBER_OP_PREDICT))
    t.append(MselScorer("neg_log_loss", _F_LOG_LOSS, 0, 0, -1.0, MEMBER_OP_PREDICT_PROBA))
    t.append(MselScorer("roc_auc", _F_ROC, ROC_BINARY, 0, 1.0, MEMBER_OP_DECISION))
    t.append(MselScorer("roc_auc", _F_ROC, ROC_BINARY, 0, 1.0, MEMBER_OP_PREDICT_PROBA))
    t.append(MselScorer("roc_auc_ovr", _F_ROC, ROC_OVR, 0, 1.0, MEMBER_OP_PREDICT_PROBA))
    t.append(MselScorer("roc_auc_ovo", _F_ROC, ROC_OVO, 0, 1.0, MEMBER_OP_PREDICT_PROBA))
    t.append(MselScorer("roc_auc_ovr_weighted", _F_ROC, ROC_OVR_WEIGHTED, 0, 1.0, MEMBER_OP_PREDICT_PROBA))
    t.append(MselScorer("roc_auc_ovo_weighted", _F_ROC, ROC_OVO_WEIGHTED, 0, 1.0, MEMBER_OP_PREDICT_PROBA))
    # lane mf-search4b: average='binary', pos_label=1 (appended: earlier codes unchanged)
    t.append(MselScorer("precision", _F_BINARY, LBL_PRECISION, 0, 1.0, MEMBER_OP_PREDICT))
    t.append(MselScorer("recall", _F_BINARY, LBL_RECALL, 0, 1.0, MEMBER_OP_PREDICT))
    t.append(MselScorer("f1", _F_BINARY, LBL_F1, 0, 1.0, MEMBER_OP_PREDICT))
    t.append(MselScorer("jaccard", _F_BINARY, LBL_JACCARD, 0, 1.0, MEMBER_OP_PREDICT))
    # lane mf-score5 (appended: earlier codes unchanged)
    t.append(MselScorer("average_precision", _F_AP, 0, 0, 1.0, MEMBER_OP_DECISION))
    t.append(MselScorer("average_precision", _F_AP, 0, 0, 1.0, MEMBER_OP_PREDICT_PROBA))
    t.append(MselScorer("neg_brier_score", _F_BRIER, 0, 0, -1.0, MEMBER_OP_PREDICT_PROBA))
    t.append(MselScorer("top_k_accuracy", _F_TOP_K, _TOP_K, 0, 1.0, MEMBER_OP_DECISION))
    t.append(MselScorer("top_k_accuracy", _F_TOP_K, _TOP_K, 0, 1.0, MEMBER_OP_PREDICT_PROBA))
    t.append(MselScorer("normalized_mutual_info_score", _F_CLUSTER, _CL_NMI, 0, 1.0, MEMBER_OP_PREDICT))
    t.append(MselScorer("adjusted_mutual_info_score", _F_CLUSTER, _CL_AMI, 0, 1.0, MEMBER_OP_PREDICT))
    return t^


def msel_scorer_row(code: Int) raises -> MselScorer:
    """The scorer of `code` (once per RUN and scorer)."""
    if code == SCORER_ACCURACY:
        return MselScorer("accuracy", _F_ACCURACY, 0, 0, 1.0, MEMBER_OP_PREDICT)
    var t = _table()
    if code < 2 or code - 2 >= len(t):
        raise Error("RuntimeError: unknown search scorer code " + String(code))
    return t[code - 2]


def label_accuracy_value(route: MetricRoute, truth: Int, pred: Int, count: Int, classes: Int) raises -> Float64:
    """`accuracy_score(y_true, y_pred)` of one side through the native metric
    program (the plan `labels_accuracy` runs; codes in 0 .. classes-1; was
    core/msel_search_run.mojo `_accuracy`)."""
    var acc = label_accuracy_plan(count, classes, False)
    var addresses: List[Int] = [truth, pred]
    var sizes: List[Int] = [count, count]
    var arena = metric_inputs_run(route, acc.program, addresses, sizes)
    var value = arena.f64(acc.result_offset + 8)
    _ = acc^
    return value


def msel_scorer_code(name: String, has_decision: Bool) -> Int:
    """UNPACK's typed lookup: the code of the scorer `name` (`has_decision`:
    the estimator's class has `decision_function`, read once by the caller),
    or -1 when the RUN does not serve it."""
    if name == "accuracy":
        return SCORER_ACCURACY
    var t = _table()
    for i in range(len(t)):  # driver-loop: over the served scorer rows (a fixed table)
        if String(t[i].name) != name:
            continue
        if _dual(t[i]):
            if (t[i].op == MEMBER_OP_DECISION) != has_decision:
                continue
        return i + 2
    return -1


def _dual(row: MselScorer) -> Bool:
    """A scorer of response ('decision_function', 'predict_proba'): two rows,
    the UNPACK picks one by the class's methods."""
    return (row.family == _F_ROC and row.metric == ROC_BINARY) or row.family == _F_AP or row.family == _F_TOP_K


def msel_scorer_positive(row: MselScorer) -> Bool:
    """True for a scorer that reads a positive label (its pos_label=1, mapped
    to a class code at UNPACK) the RUN requires among the classes."""
    return row.family == _F_BINARY


def msel_scorer_positive_kind(row: MselScorer) -> Int:
    """The positive word the UNPACK writes for `row` (SCORER_POSITIVE_*):
    the label 1's class code for the binary label scorers and
    'average_precision', the label marks for 'neg_brier_score'."""
    if row.family == _F_BINARY or row.family == _F_AP:
        return SCORER_POSITIVE_LABEL_ONE
    if row.family == _F_BRIER:
        return SCORER_POSITIVE_MARKS
    return SCORER_POSITIVE_NONE


def msel_scorer_ap(row: MselScorer) -> Bool:
    """True for 'average_precision' (its binary column is the positive's)."""
    return row.family == _F_AP


def msel_scorer_regression(row: MselScorer) -> Bool:
    """True for a scorer over a regressor's float32 predictions."""
    return row.family == _F_REG


def msel_scorer_binary_column(row: MselScorer) -> Bool:
    """True for the `_BINARY_PROBA` scorers that read one column of a
    two-class member's `predict_proba` ('neg_log_loss', 'roc_auc',
    'neg_brier_score', 'top_k_accuracy': column 1; 'average_precision': the
    positive's, `msel_scorer_ap`)."""
    if row.op != MEMBER_OP_PREDICT_PROBA:
        return False
    return (row.family == _F_LOG_LOSS or (row.family == _F_ROC and row.metric == ROC_BINARY)
            or row.family == _F_AP or row.family == _F_BRIER or row.family == _F_TOP_K)


def _prf_notices(metric: Int, bits: Int, mut notices: List[Float64]):
    """The public metric's warning of a precision / recall / f-score / jaccard
    epilogue (`label_metric_run`'s `warn`): the scorer's one metric
    (`warn_for` = its own name), once."""
    if metric == LBL_JACCARD:
        if bits != 0:
            member_notice_add(notices, MEMBER_NOTICE_METRIC_JACCARD)
        return
    var which = 0 if metric == LBL_PRECISION else (1 if metric == LBL_RECALL else 2)
    var bit = 1 if which == 0 else (2 if which == 1 else 4)
    if (bits & bit) != 0:
        member_notice_add(notices, MEMBER_NOTICE_METRIC_PRF, Float64(which))


def msel_scorer_value(route: MetricRoute, row: MselScorer, truth: Int, response: Int, count: Int, classes: Int,
                      cols: Int, positive: Int, mut notices: List[Float64]) raises -> Float64:
    """The scorer's value on one fold side: `truth` (count int32 class codes
    in 0 .. classes-1, or count float32 targets for a regression scorer),
    `response` the member's answer as the RUN placed it (count int32 codes,
    or count x `cols` float32 words: predictions, probabilities or decision
    values; one column after the `_BINARY_PROBA` column pick), `positive`
    the scorer's positive word (`msel_scorer_positive_kind`; -1 for none).
    The public metric's warnings on this side are appended to `notices`
    (core/member_notices.mojo triples, in the metric's order). The metric's
    refusals raise its words (the caller's error rule applies)."""
    if row.family == _F_ACCURACY:
        return label_accuracy_value(route, truth, response, count, classes)
    var value: Float64
    var warn = 0
    if row.family == _F_REG:
        value = reg_metric_run(route, row.metric, truth, response, count, warn)
        if warn != 0:
            member_notice_add(notices, MEMBER_NOTICE_METRIC_FEW, 1.0 if row.metric == REG_D2_ABSOLUTE else 0.0)
    elif row.family == _F_BINARY:
        value = label_binary_run(route, row.metric, truth, response, count, classes, positive, warn)
        _prf_notices(row.metric, warn, notices)
    elif row.family == _F_LABEL:
        value = label_metric_run(route, row.metric, row.avg, truth, response, count, classes, warn)
        if row.metric == LBL_BALANCED:
            if warn != 0:
                member_notice_add(notices, MEMBER_NOTICE_METRIC_BALANCED)
        elif row.metric != LBL_MCC:
            _prf_notices(row.metric, warn, notices)
    elif row.family == _F_AP:
        value = average_precision_run(route, truth, response, count, cols, classes, positive)
    elif row.family == _F_BRIER:
        value = brier_run(route, truth, response, count, classes, cols, positive)
    elif row.family == _F_TOP_K:
        value = top_k_run(route, truth, response, count, cols, classes, row.metric, warn)
        if warn != 0:
            member_notice_add(notices, MEMBER_NOTICE_METRIC_TOP_K, Float64(row.metric), Float64(
                _present_count(truth, count, classes)))
    elif row.family == _F_CLUSTER:
        # `_average_method('arithmetic')` (the scorers pass no average_method)
        value = cluster_mi_run(route, row.metric == _CL_AMI, truth, response, count, classes, 2)
    elif row.family == _F_LOG_LOSS:
        if cols != (1 if classes == 2 else classes):
            raise Error("probability columns must match the label count (at least two)")
        value = log_loss_run(route, truth, response, count, classes)
    else:
        if row.metric == ROC_BINARY:
            if classes > 2 or cols != 1:
                raise Error("multi_class must be in ('ovo', 'ovr')")
        elif cols != classes:
            raise Error("Number of classes in y_true not equal to the number of columns in 'y_score'")
        value = roc_auc_run(route, row.metric, truth, response, count, classes)
    return row.sign * value


def _present_count(truth: Int, count: Int, classes: Int) -> Int:
    """The number of class codes y_true holds (the warning's n_classes)."""
    var t = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=truth)
    var seen = List[Bool](length=max(classes, 1), fill=False)
    var got = 0
    for i in range(count):  # driver-loop: over the side's rows (the classes y_true holds)
        var c = Int(t[i])
        if c >= 0 and c < classes and not seen[c]:
            seen[c] = True
            got += 1
    return got
