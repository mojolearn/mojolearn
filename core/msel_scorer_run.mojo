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

Not served here (the search keeps its units for them, with the reason):
'average_precision' and 'neg_brier_score' (no typed program builder: their
programs are assembled inside their doors over pins), 'top_k_accuracy' (no
typed builder; its door ranks the scores in Python-held arrays), the
clustering scores (no member response: they score labels against labels
of a clusterer, which no search member serves).
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
        if t[i].family == _F_ROC and t[i].metric == ROC_BINARY:
            if (t[i].op == MEMBER_OP_DECISION) != has_decision:
                continue
        return i + 2
    return -1


def msel_scorer_positive(row: MselScorer) -> Bool:
    """True for a scorer that reads a positive label (its pos_label=1, mapped
    to a class code at UNPACK)."""
    return row.family == _F_BINARY


def msel_scorer_regression(row: MselScorer) -> Bool:
    """True for a scorer over a regressor's float32 predictions."""
    return row.family == _F_REG


def msel_scorer_binary_column(row: MselScorer) -> Bool:
    """True for the `_BINARY_PROBA` scorers that read `predict_proba[:, 1]`
    of a two-class member ('neg_log_loss', 'roc_auc' over probabilities)."""
    return row.op == MEMBER_OP_PREDICT_PROBA and (row.family == _F_LOG_LOSS or
                                                  (row.family == _F_ROC and row.metric == ROC_BINARY))


def msel_scorer_value(route: MetricRoute, row: MselScorer, truth: Int, response: Int, count: Int, classes: Int,
                      cols: Int, positive: Int) raises -> Float64:
    """The scorer's value on one fold side: `truth` (count int32 class codes
    in 0 .. classes-1, or count float32 targets for a regression scorer),
    `response` the member's answer as the RUN placed it (count int32 codes,
    or count x `cols` float32 words: predictions, probabilities or decision
    values; one column after the `_BINARY_PROBA` column pick), `positive`
    the class code of a binary scorer's pos_label (-1 for the others). The
    metric's refusals raise its words (the caller's error rule applies)."""
    if row.family == _F_ACCURACY:
        return label_accuracy_value(route, truth, response, count, classes)
    var value: Float64
    if row.family == _F_REG:
        value = reg_metric_run(route, row.metric, truth, response, count)
    elif row.family == _F_BINARY:
        value = label_binary_run(route, row.metric, truth, response, count, classes, positive)
    elif row.family == _F_LABEL:
        value = label_metric_run(route, row.metric, row.avg, truth, response, count, classes)
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
