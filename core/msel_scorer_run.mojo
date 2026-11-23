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

THE CLUSTER LABEL-PAIR SCORERS (lane mf-cluster5): 'rand_score',
'adjusted_rand_score', 'mutual_info_score', 'homogeneity_score',
'completeness_score', 'v_measure_score' and 'fowlkes_mallows_score' run
core/metrics_cluster_table.mojo `cluster_pair_run`, the typed RUN their
public doors call, over the metrics image's entry the UNPACK resolved once
(the image's own body: device kernels on a GPU image, the oracle loops on
the CPU column). A CLUSTERER member (no class codes: a search over KMeans
and the other clusterers, `classes` 0) is scored by these and by NMI / AMI
on its predicted cluster ids against the true labels, which the search
hands as float32 words of integer labels (the regressor kind's target
words); they are read back as int32 labels, exactly.

A MAKE_SCORER SCORER (lane mf-cluster5): a native `make_scorer` whose
metric, kwargs and response method are a table row's is that row, its sign
flipped when its greater_is_better differs (`msel_scorer_flip`: the code
plus SCORER_FLIP).

A POS_LABEL THAT IS NOT ONE OF THE LABELS (lane mf-cluster5): the binary
label scorers' positive word SCORER_POSITIVE_ABSENT (integer labels, 1 not
among them) or SCORER_POSITIVE_KIND (labels of another type) makes each
side raise the door's words (bindings/mcls_classify_door.mojo
`_set_wise_labels`, with the side's present labels as `label_words` spell
them), or, with fewer than two labels present, score the absent positive
as the door does (0.0 and its ill-defined warning).
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
from core.entry_table import EntryRef, entry_ref_unresolved
from core.metrics_cluster_table import (
    cluster_pair_run, cluster_union_codes, CLUSTER_OP_RAND, CLUSTER_OP_ADJUSTED_RAND, CLUSTER_OP_MUTUAL_INFO,
    CLUSTER_OP_HOMOGENEITY, CLUSTER_OP_COMPLETENESS, CLUSTER_OP_V_MEASURE, CLUSTER_OP_FOWLKES_MALLOWS,
)
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
#: lane mf-cluster5: the label-pair metrics of core/metrics_cluster_table.mojo,
#: their word = the table's operation + _CL_PAIR.
comptime _CL_PAIR = 2
#: A make_scorer row whose sign is the table row's negated: the code plus this.
comptime SCORER_FLIP = 1048576
#: The top_k scorer's k (`top_k_accuracy_score`'s default; the scorer passes none).
comptime _TOP_K = 2
#: `MselSearchPlan.positives` words a scorer reads (`msel_scorer_positive_kind`).
comptime SCORER_POSITIVE_NONE = 0
comptime SCORER_POSITIVE_LABEL_ONE = 1   # the class code of the label 1 (-1: not a label)
comptime SCORER_POSITIVE_MARKS = 2       # bindings/metrics_exp_rank_run.mojo `brier_marks_positive`'s marks
#: Lane mf-cluster5: the positive WORD a binary label scorer gets when its
#: pos_label=1 is not one of the labels: integer labels without 1, or labels
#: of another type than pos_label's (module note).
comptime SCORER_POSITIVE_ABSENT = -2
comptime SCORER_POSITIVE_KIND = -3
#: Lane mf-msel5b: labels of string AND integer kinds. The label scorers'
#: metric doors refuse such targets before any pos_label check
#: (bindings/classical_labels_door.mojo, `SCORER_MIXED_WORDS`); the RUN
#: raises the same words per side.
comptime SCORER_POSITIVE_MIXED = -4
comptime SCORER_MIXED_WORDS = ("TypeError: y_true must contain only strings or only integers; floating labels, "
                               + "missing labels and mixed types are unsupported")


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
    # lane mf-cluster5 (appended: earlier codes unchanged)
    t.append(MselScorer("rand_score", _F_CLUSTER, _CL_PAIR + CLUSTER_OP_RAND, 0, 1.0, MEMBER_OP_PREDICT))
    t.append(MselScorer("adjusted_rand_score", _F_CLUSTER, _CL_PAIR + CLUSTER_OP_ADJUSTED_RAND, 0, 1.0,
                        MEMBER_OP_PREDICT))
    t.append(MselScorer("mutual_info_score", _F_CLUSTER, _CL_PAIR + CLUSTER_OP_MUTUAL_INFO, 0, 1.0, MEMBER_OP_PREDICT))
    t.append(MselScorer("homogeneity_score", _F_CLUSTER, _CL_PAIR + CLUSTER_OP_HOMOGENEITY, 0, 1.0, MEMBER_OP_PREDICT))
    t.append(MselScorer("completeness_score", _F_CLUSTER, _CL_PAIR + CLUSTER_OP_COMPLETENESS, 0, 1.0,
                        MEMBER_OP_PREDICT))
    t.append(MselScorer("v_measure_score", _F_CLUSTER, _CL_PAIR + CLUSTER_OP_V_MEASURE, 0, 1.0, MEMBER_OP_PREDICT))
    t.append(MselScorer("fowlkes_mallows_score", _F_CLUSTER, _CL_PAIR + CLUSTER_OP_FOWLKES_MALLOWS, 0, 1.0,
                        MEMBER_OP_PREDICT))
    return t^


def msel_scorer_row(code: Int) raises -> MselScorer:
    """The scorer of `code` (once per RUN and scorer); a code at or above
    SCORER_FLIP is the row below it with its sign negated (a make_scorer
    whose greater_is_better differs from the row's)."""
    if code >= SCORER_FLIP:
        var row = msel_scorer_row(code - SCORER_FLIP)
        row.sign = -row.sign
        return row
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


def msel_scorer_flip(code: Int) -> Int:
    """UNPACK: the code of `code`'s row with its sign negated."""
    return code - SCORER_FLIP if code >= SCORER_FLIP else code + SCORER_FLIP


def msel_scorer_cluster(row: MselScorer) -> Bool:
    """True for a clustering scorer (NMI, AMI and the label-pair metrics):
    the only scorers a clusterer member (no class codes) is scored by."""
    return row.family == _F_CLUSTER


def msel_scorer_cluster_entry(row: MselScorer) -> Bool:
    """True for a scorer that runs the metrics image's cluster entry (the
    UNPACK resolves it once)."""
    return row.family == _F_CLUSTER and row.metric >= _CL_PAIR


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


def msel_scorer_labels_from_truth(row: MselScorer) -> Bool:
    """True for the scorers whose metric takes its label set from the side's
    `y_true` (labels=None in scikit-learn: roc_auc and its ovr / ovo forms
    through `np.unique(y_true)`, log_loss through `LabelBinarizer().fit(y_true)`,
    top_k_accuracy through `np.unique(y_true)`): a fold estimator's own
    columns are read against THOSE labels (lane mf-kmeans5, the fold that
    lacks a class)."""
    return row.family == _F_ROC or row.family == _F_LOG_LOSS or row.family == _F_TOP_K


def msel_scorer_log_loss(row: MselScorer) -> Bool:
    """True for 'neg_log_loss'."""
    return row.family == _F_LOG_LOSS


def msel_scorer_ap(row: MselScorer) -> Bool:
    """True for 'average_precision' (its binary column is the positive's)."""
    return row.family == _F_AP


def msel_scorer_brier(row: MselScorer) -> Bool:
    """True for 'neg_brier_score' (lane mf-msel5c: a fold that lacks a class)."""
    return row.family == _F_BRIER


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
    """`msel_scorer_value` with no cluster entry and no label words (the
    callers whose scorers read neither)."""
    return msel_scorer_value(route, entry_ref_unresolved(), List[String](), row, truth, response, count, classes,
                             cols, positive, notices)


def msel_scorer_value(route: MetricRoute, cluster: EntryRef, label_words: List[String], row: MselScorer, truth: Int,
                      response: Int, count: Int, classes: Int, cols: Int, positive: Int,
                      mut notices: List[Float64]) raises -> Float64:
    """The scorer's value on one fold side: `truth` (count int32 class codes
    in 0 .. classes-1, or count float32 targets for a regression scorer),
    `response` the member's answer as the RUN placed it (count int32 codes,
    or count x `cols` float32 words: predictions, probabilities or decision
    values; one column after the `_BINARY_PROBA` column pick), `positive`
    the scorer's positive word (`msel_scorer_positive_kind`; -1 for none;
    SCORER_POSITIVE_ABSENT / _KIND: pos_label=1 is not a label). `cluster`:
    the metrics image's cluster entry (the label-pair scorers); `classes` 0
    with a clustering scorer: a clusterer's ids against float32 words of
    integer labels. `label_words`: the labels' reprs by class code (the
    door's refusal words).
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
        value = _binary_value(route, label_words, row, truth, response, count, classes, positive, warn)
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
        value = _cluster_value(route, cluster, row, truth, response, count, classes)
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


def _cluster_value(route: MetricRoute, cluster: EntryRef, row: MselScorer, truth: Int, response: Int, count: Int,
                   classes: Int) raises -> Float64:
    """A clustering scorer on one side (lane mf-cluster5). A classifier's
    side: class codes in 0 .. classes-1 on both (codes keep their labels'
    order, so the union codes and the Rand pair counts are the door's). A
    clusterer's side (`classes` 0): the true labels arrive as float32 words
    of integers (read back exactly, else the door's int32 refusal) and the
    member's cluster ids as int32."""
    var labels = List[Int32]()
    var t = truth
    if classes == 0:
        labels = List[Int32](unsafe_uninit_length=max(count, 1))
        var src = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=truth)
        for i in range(count):  # driver-loop: over the side's rows (the true labels as int32)
            var v = src[i]
            if not (v >= -2147483648.0 and v < 2147483648.0) or Float32(Int32(Int(v))) != v:
                raise Error("ValueError: labels_true must hold integer labels that fit in int32")
            labels[i] = Int32(Int(v))
        t = Int(labels.unsafe_ptr())
    var value: Float64
    if row.metric >= _CL_PAIR:
        value = cluster_pair_run(cluster, row.metric - _CL_PAIR, t, response, count, 1.0)
    elif classes == 0:
        # NMI / AMI read codes in 0 .. k-1: the union codes of the raw labels
        # (order-preserving, so each clustering's own classes are the door's).
        var codes = List[Int32]()
        var k = cluster_union_codes(t, response, count, codes)
        var base = Int(codes.unsafe_ptr())
        # `_average_method('arithmetic')` (the scorers pass no average_method)
        value = cluster_mi_run(route, row.metric == _CL_AMI, base, base + 4 * count, count, k, 2)
        _ = len(codes)
    else:
        # `_average_method('arithmetic')` (the scorers pass no average_method)
        value = cluster_mi_run(route, row.metric == _CL_AMI, t, response, count, classes, 2)
    _ = len(labels)  # read by the RUN above
    return value


def _present_words(label_words: List[String], seen: List[Bool]) -> String:
    """`str(present)`: the present labels' reprs as a Python list prints them."""
    var out = String("[")
    var first = True
    for c in range(len(seen)):  # small-loop(the class codes present)
        if not seen[c]:
            continue
        if not first:
            out += ", "
        first = False
        out += label_words[c] if c < len(label_words) else String(c)
    return out + "]"


def _binary_value(route: MetricRoute, label_words: List[String], row: MselScorer, truth: Int, response: Int,
                  count: Int, classes: Int, positive: Int, mut warn: Int) raises -> Float64:
    """The binary label scorers on one side, the door's `_set_wise_labels`
    order first (bindings/mcls_classify_door.mojo): more than two present
    labels refuse; pos_label of another type refuses; a pos_label not among
    two present labels refuses with them; else
    bindings/mcls_classify_run.mojo `label_binary_run` (an absent positive
    as one more class code that no row holds: scored 0.0, as the door
    scores `labels=[pos_label]`)."""
    var t = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=truth)
    var p = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=response)
    var seen = List[Bool](length=max(classes, 1), fill=False)
    var present = 0
    for i in range(count):  # driver-loop: over the side's rows (the present labels)
        var a = Int(t[i])
        var b = Int(p[i])
        if a < 0 or a >= classes or b < 0 or b >= classes:
            raise Error("ValueError: y codes outside [0, n_classes)")
        if not seen[a]:
            seen[a] = True
            present += 1
        if not seen[b]:
            seen[b] = True
            present += 1
    if present > 2:
        raise Error("Target is multiclass but average='binary'. Please choose another average setting, one of "
                    + "[None, 'micro', 'macro', 'weighted'].")
    if positive == SCORER_POSITIVE_KIND:
        raise Error("pos_label must use the same type as the targets")
    var absent = positive == SCORER_POSITIVE_ABSENT
    if (absent or (positive >= 0 and positive < classes and not seen[positive])) and present >= 2:
        raise Error("pos_label=1 is not a valid label. It should be one of " + _present_words(label_words, seen))
    if absent:
        return label_binary_run(route, row.metric, truth, response, count, classes + 1, classes, warn)
    return label_binary_run(route, row.metric, truth, response, count, classes, positive, warn)
