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
flipped when its greater_is_better differs.

THE SCORER'S OPTIONS (lane mf-tail5b, one layout for every kwarg): the
UNPACK reads a make_scorer's kwargs ONCE into `MselScorerOptions` (the sign
flip, a given pos_label, top_k's k, fbeta's beta, zero_division and a
`labels` list as class codes); the search kind carries them as one word
group per scorer (`msel_scorer_group_words` / `msel_scorer_group_read`, the
only writer and reader of that layout) and `msel_scorer_row` applies them to
the table row. 'fbeta' is the F row of its average with its beta.

A POS_LABEL THAT IS NOT ONE OF THE LABELS (lane mf-cluster5): the binary
label scorers' positive word SCORER_POSITIVE_ABSENT (integer labels, 1 not
among them) or SCORER_POSITIVE_KIND (labels of another type) makes each
side raise the door's words (bindings/mcls_classify_door.mojo
`_set_wise_labels`, with the side's present labels as `label_words` spell
them), or, with fewer than two labels present, score the absent positive
as the door does (0.0 and its ill-defined warning).

THE LAST SCORER OPTIONS (lane mf-score6), read at UNPACK into the word group:
- average=None on an averaged label row: the epilogue's per-label binary64
  values (bindings/mcls_classify_run.mojo `label_metric_options_run`, avg
  LBL_NONE). scikit-learn's `_Scorer._score` returns `sign * array`; its
  model_selection/_validation.py `_score` takes `.item()` of the answer and
  raises "scoring must return a number, got %s (%s) instead. (scorer=%s)"
  for anything that is not then a number. That check runs after the
  scorer's try, so a numeric error_score never replaces it. In a search the
  RUN does the same: one label (a `labels` list of one, or one present
  label) scores its value; more raise those words (SCORER_NUMBER_WORDS) and
  the search raises them whatever the error_score.
- average='samples': scikit-learn's own metrics refuse it on 1-D targets;
  Mojolearn's metric doors refuse it as multilabel NOT IMPLEMENTED
  (metrics/NOT_IMPLEMENTED.tsv), and so does each side here.
- sample_weight given to make_scorer: the weights' float32 bits ride the
  selection words; every row whose metric has the parameter runs its door's
  weighted program (one body); each side checks the length and values in
  the door's words, as scikit-learn passes the same kwarg to every side.
  A metric without the parameter refuses per side. Weights given to a
  search's fit reach no scorer: scikit-learn routes fit parameters to the
  estimator's fit only (without metadata routing), and Mojolearn's search
  `fit(X, y, groups)` takes none.
- a float pos_label: the binary label rows and brier refuse it per side as
  their doors' `_classification_labels` does (SCORER_FLOAT_POS_WORDS);
  average_precision's door compares it by value, and so does the UNPACK.

THE METRIC ROWS (lane mf-fitadm7): a native make_scorer whose metric has no
get_scorer name but a typed program is a row too, matched by its metric
(`msel_scorer_metric_code`): zero_one_loss (normalize True / False) and
hamming_loss over the COUNT tail (bindings/mcls_classify_run.mojo
`label_count_run`, the program bindings/metrics_exp_cls_door.mojo
`_cm_count` runs), sample_weight as the other weighted rows take it.

EVERY REMAINING SCORER, TYPED (lane mf-score8): a native make_scorer over
any connected metric that returns one number for 1-D targets and has a
typed program is a row, its kwargs read ONCE at UNPACK into the word group:
- the regression metrics (`reg_metric_options_run`, the one body of
  bindings/metrics_reg_door.mojo): mean_pinball_loss and d2_pinball_score
  (`alpha`), mean_tweedie_deviance and d2_tweedie_score (`power`, printed
  as given in the Tweedie domain words), `multioutput` on a 1-D target
  ('raw_values', 'uniform_average', r2 / explained variance's
  'variance_weighted': the door's program of that rule, one output's value),
  `force_finite` (r2, explained variance);
- accuracy_score normalize=False (the COUNT tail's correct count,
  `label_count_run` CNT_CORRECT), balanced_accuracy_score adjusted=True,
  log_loss normalize=False, top_k_accuracy_score normalize=False,
  brier_score_loss scale_by_half (the flag word: -1 'auto', 0, 1);
- cohen_kappa_score (`labels` as class codes in their order, `weights`,
  `replace_undefined_by`; bindings/mcls_classify_run.mojo `kappa_run`, the
  program its door now builds through `kappa_stages`), its two warnings as
  notice words (MEMBER_NOTICE_METRIC_KAPPA);
- a REGRESSION scorer on a CLASSIFIER: scikit-learn scores the predicted
  labels as numbers. The UNPACK reads each fit label's value once (an
  integer, a bool or a real as float32 words; a string as `float(label)`
  when that converts) into two words per label (the value's bits, its kind
  word `SCORER_LABEL_*`); each side maps its codes onto them and runs the
  regression row's program. A string that does not convert raises
  NumPy's words ("could not convert string to float: <repr>", y_true's rows
  first) and median_absolute_error / max_error, which check their targets
  with dtype='numeric', refuse string labels in scikit-learn's words
  (`SCORER_NUMERIC_WORDS`).
A kwarg value the metric's door refuses (a bad multioutput string, custom
multioutput weights on 1-D targets, an alpha outside [0, 1], a power that
is no real, a normalize that is no bool, kappa weights of another name, a
repeated or other-kind `labels` entry) is ONE MORE LABEL WORD the UNPACK
appends (`refuse_word`), raised per side in that door's words, so the
error_score rule applies as scikit-learn's scorer error would.
"""
from std.memory import bitcast
from bindings.arena_prog_run import MetricRoute
from bindings.metrics_reg_run import (
    reg_metric_run, reg_metric_options_run, reg_metric_caller, reg_metric_d2, reg_metric_takes_multioutput,
    reg_metric_takes_variance, REG_R2, REG_MSE, REG_MAE, REG_RMSE, REG_MSLE, REG_RMSLE, REG_MAPE, REG_MEDAE,
    REG_EXPLAINED_VARIANCE, REG_MAX_ERROR, REG_POISSON, REG_GAMMA, REG_D2_ABSOLUTE, REG_PINBALL, REG_TWEEDIE,
    REG_D2_TWEEDIE, REG_D2_PINBALL, REG_MO_UNIFORM,
)
from bindings.mcls_classify_run import (
    label_metric_run, label_binary_run, label_metric_options_run, label_binary_options_run, log_loss_run, LBL_JACCARD, LBL_BALANCED, LBL_PRECISION, LBL_RECALL, LBL_F1, LBL_MCC,
    LBL_MICRO, LBL_MACRO, LBL_WEIGHTED, label_count_run, CNT_LOSS, CNT_COUNT, CNT_CORRECT, log_loss_options_run,
    kappa_run, clr_run,
)
from bindings.mcls_ranking_run import (
    roc_auc_run, roc_auc_options_run, ROC_BINARY, ROC_OVR, ROC_OVR_WEIGHTED, ROC_OVO, ROC_OVO_WEIGHTED, ROC_OVR_MICRO,
    ROC_OVR_NONE,
)
from bindings.metrics_exp_curve_run import average_precision_run, top_k_run, top_k_options_run, AP_MACRO, AP_NONE
from bindings.metrics_exp_rank_run import brier_run, brier_options_run, hinge_run, d2_proba_run
from bindings.metrics_exp_common_run import cluster_mi_run
from core.entry_table import EntryRef, entry_ref_unresolved
from core.metrics_cluster_table import (
    cluster_pair_run, cluster_union_codes, CLUSTER_OP_RAND, CLUSTER_OP_ADJUSTED_RAND, CLUSTER_OP_MUTUAL_INFO,
    CLUSTER_OP_HOMOGENEITY, CLUSTER_OP_COMPLETENESS, CLUSTER_OP_V_MEASURE, CLUSTER_OP_FOWLKES_MALLOWS,
)
from core.member_notices import (
    member_notice_add, MEMBER_NOTICE_METRIC_PRF, MEMBER_NOTICE_METRIC_JACCARD, MEMBER_NOTICE_METRIC_BALANCED,
    MEMBER_NOTICE_METRIC_FEW, MEMBER_NOTICE_METRIC_TOP_K, MEMBER_NOTICE_METRIC_KAPPA, MEMBER_NOTICE_METRIC_CLR,
    MEMBER_NOTICE_METRIC_ROC_ONE,
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
#: lane mf-fitadm7: the COUNT tail's metrics (zero_one_loss, hamming_loss),
#: their metric word CNT_LOSS / CNT_COUNT or `_CNT_HAMMING`.
comptime _F_COUNT = 10
comptime _CNT_HAMMING = 2
#: lane mf-score8: accuracy_score normalize=False on the COUNT tail (its
#: metric word: bindings/mcls_classify_run.mojo CNT_CORRECT) and the
#: cohen_kappa_score family.
comptime _CNT_CORRECT = CNT_CORRECT
comptime _F_KAPPA = 11
#: lane mf-score8 round b: hinge_loss (decision values), the D^2 probability
#: scores (metric 1 d2_log_loss_score, 0 d2_brier_score: the full
#: predict_proba) and the likelihood-ratio scorers (metric 0 LR+, 1 LR-).
comptime _F_HINGE = 12
comptime _F_D2P = 13
comptime _F_CLR = 14
#: `_F_CLUSTER` metrics.
comptime _CL_NMI = 0
comptime _CL_AMI = 1
#: lane mf-cluster5: the label-pair metrics of core/metrics_cluster_table.mojo,
#: their word = the table's operation + _CL_PAIR.
comptime _CL_PAIR = 2
#: THE SCORER WORD GROUP (lane mf-tail5b; module note): per scorer
#: [code, positive word, flip (1.0), pos_label given (1.0; lane mf-score6:
#: 2.0 for a FLOAT pos_label), the given label's index among the label words
#: (-1), k (0: the row's), beta (-1: the row's), zero_division (-1 'warn',
#: else the cls_epi code 0 / 1 / 2 for nan), the number of `labels` codes,
#: (lane mf-score6) average=None's scorer-words index (-1: the row's
#: average), average='samples' (1.0), the number of sample_weight words (0:
#: none; -1: the metric takes no sample_weight), the sample_weight refusal
#: word (SCORER_WEIGHTS_OK, SCORER_WEIGHTS_EMPTY, else the label word index
#: of the door's refusal words)], then the `labels` codes (a code >= the
#: fit's classes is a label no row holds), then the sample_weight words (each
#: float32's bits as an int32). Replaces the code bits SCORER_FLIP,
#: SCORER_POS_GIVEN and SCORER_K_UNIT of lanes mf-cluster5 / mf-msel5c.
#: Lane mf-score8 appends 8 words: the multioutput rule (REG_MO_*), force_finite
#: (-1 the row's, 0, 1), the metric's alpha / power / replace_undefined_by
#: (`param`), the label word printing it when given (-1: not given), the
#: refusal word (-1: none; else the label word raised per side), the
#: normalize / adjusted flag (-1 the row's, 0, 1), cohen_kappa_score's
#: weights word, the number of fit labels whose value words follow (0: none;
#: a regression scorer on a classifier). Those value words come after the
#: sample_weight words: two per label (the float32 bits as an int32, the
#: kind word SCORER_LABEL_*). Lane mf-r9c: then the scorer's own response
#: (0 the row's; else MEMBER_OP_PROBA / _DECISION: a regression row over a
#: make_scorer's non-predict response the class has). Lane mf-r9d: then the
#: numeric_mode tier (0 the process default's route, 1 the other tier's,
#: `MselOtherTier`), the tier's refusal word (-1: none; else the label word
#: each side raises: the other tier's image is not built), and the
#: predict_log_proba flag (1: the response is the log of predict_proba).
comptime SCORER_GROUP_WORDS = 25
#: Lane mf-r9d: the shared refusal of device='cpu' with the fast tier
#: (core/device_route.mojo `device_cpu_fast`, which the native scorer door's
#: prepare raises for a host estimator's fast-mode scorer), per side.
comptime SCORER_CPU_FAST_WORDS = "ValueError: device='cpu' runs in identical mode only; numeric_mode='fast' is for Apple GPUs"
#: Lane mf-score6: the sample_weight refusal word: none, or an empty weights
#: array (each side refuses its length in the door's words).
comptime SCORER_WEIGHTS_OK = -1
comptime SCORER_WEIGHTS_EMPTY = -2
#: Lane mf-score6: a FLOAT pos_label on the binary label rows and brier. The
#: metric doors read pos_label through `_classification_labels`
#: (bindings/mcls_common_unpack.mojo `mc_label_list`), which refuses a
#: floating label with these words; average_precision's door compares it by
#: value (bindings/metrics_exp_curve_door.mojo `_ap_binary`).
comptime SCORER_FLOAT_POS_WORDS = ("TypeError: pos_label must contain only strings or only integers; floating "
                                   + "labels, missing labels and mixed types are unsupported")
#: Lane mf-score6: the words that start scikit-learn's refusal of a scorer
#: that returns no number (model_selection/_validation.py `_score`: raised
#: after the scorer's try, so a numeric error_score never replaces it).
comptime SCORER_NUMBER_WORDS = "ValueError: scoring must return a number, got "
#: Lane mf-rows8a: the native scorer door's refusal of a metric answer that
#: is no real number (bindings/msel_scorer_native_door.mojo `_scorer_metric`):
#: cohen_kappa_score's replace_undefined_by object when kappa is undefined
#: (the kappa row's flag word 1).
comptime SCORER_REAL_WORDS = "TypeError: scoring must return a real scalar"
#: Lane mf-score8: a fit label's kind word (the label value words of a
#: regression scorer on a classifier): a number (an integer, a bool or a
#: real), a string `float()` converts, a string it does not.
comptime SCORER_LABEL_NUMBER = 1
comptime SCORER_LABEL_TEXT = 2
comptime SCORER_LABEL_UNREAD = 0
#: Lane mf-score8: scikit-learn's refusal of string targets in a metric that
#: checks them with dtype='numeric' (sklearn 1.9 utils/validation.py
#: `check_array`; metrics/_regression.py median_absolute_error and max_error
#: call `_check_reg_targets(..., dtype="numeric")`).
comptime SCORER_NUMERIC_WORDS = ("ValueError: dtype='numeric' is not compatible with arrays of bytes/strings."
                                 + "Convert your data to numeric values explicitly instead.")
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


struct MselScorer(ImplicitlyCopyable, Movable):
    """One served scorer: its name, metric family, metric and averaging
    words, sign and member response, and (lane mf-tail5b) the options a
    make_scorer gave: the F-score's mode and beta^2, zero_division's code
    and whether it silences the warnings, the `labels` selection (int32
    class codes at `sel_at`, held by the plan), a given pos_label and its
    label word's index. Resolved ONCE per RUN (`msel_scorer_row`), never per
    score."""
    var name: StaticString
    var family: Int
    var metric: Int
    var avg: Int
    var sign: Float64
    var op: Int
    var fmode: Int
    var b2: Float64
    var zd: Int
    var quiet: Bool
    var sel_at: Int
    var sel_n: Int
    var pos_given: Bool
    var pos_word: Int
    # lane mf-score6: average=None (`avg_none`, the scorer's words at label
    # word `repr_word`), average='samples', the sample_weight words (float32
    # at `w_at`, `w_n` of them; -1: the metric takes no sample_weight), their
    # refusal word (`w_word`) and a float pos_label (`pos_float`).
    var avg_none: Bool
    var repr_word: Int
    var samples: Bool
    var w_at: Int
    var w_n: Int
    var w_word: Int
    var pos_float: Bool
    # lane mf-score8: the multioutput rule, force_finite, the metric's alpha
    # / power / replace_undefined_by (`param`, given when `param_given`, its
    # printed words at label word `param_word`), the refusal word, the
    # normalize / adjusted flag, kappa's weights, the fit labels' value
    # words (int32 pairs at `lv_at`, `lv_n` labels).
    var mo: Int
    var ff: Bool
    var param: Float64
    var param_given: Bool
    var param_word: Int
    var refuse_word: Int
    var flag: Int
    var wmode: Int
    var lv_at: Int
    var lv_n: Int
    # lane mf-score8 round b: the side's member answers integer ids (a
    # non-classifier's predict: a clusterer), set by the search RUN per
    # score; not a word of the group
    var ids: Bool
    # lane mf-rows8b (row :3899): the side's member answers its predict as
    # floats (a regressor) under a clustering scorer: 1 float32, 2 float64
    # (the member's element type), 0 none; set by the search RUN per score
    # like `ids`; the public cluster doors refuse it (`_cluster_value`)
    var fpred: Int
    # lane mf-rows8b (rows :3777 / :3779): a multi-output target's side, set by
    # the search RUN per score like `ids`: `outs` its outputs (0: one), and
    # for a multi-output classifier `counts_at` its outputs' class counts
    # (Int words; 0 a regressor), the row's label value words (`lv_at`,
    # `lv_n`) then every output's labels in output order
    var outs: Int
    var counts_at: Int
    # lane mf-r9c: a multi-output classifier's probability row (set by the
    # search RUN per score like `outs`): the folds' held classes per output
    # (`mheld_at`, the address of the RUN's List[List[Int32]], fold-major,
    # one list per output, empty: every class; 0 when no fold lacks one)
    # and the side's fold, which `_multi_classes` reads as the fold
    # estimator's `classes_`
    var mheld_at: Int
    var fold: Int
    # lane mf-r9d: the numeric_mode tier (`MselScorerOptions.tier`), its
    # refusal word, the predict_log_proba response (`log`), and (set by the
    # search RUN per score like `ids`) the side's candidate takes its rows on
    # the host under the fast tier (`cpu_fast`: SCORER_CPU_FAST_WORDS)
    var tier: Int
    var tier_word: Int
    var log: Bool
    var cpu_fast: Bool

    def __init__(out self, name: StaticString, family: Int, metric: Int, avg: Int, sign: Float64, op: Int):
        self.name = name
        self.family = family
        self.metric = metric
        self.avg = avg
        self.sign = sign
        self.op = op
        self.fmode = 2
        self.b2 = 1.0
        self.zd = 0
        self.quiet = False
        self.sel_at = 0
        self.sel_n = 0
        self.pos_given = False
        self.pos_word = -1
        self.avg_none = False
        self.repr_word = -1
        self.samples = False
        self.w_at = 0
        self.w_n = 0
        self.w_word = SCORER_WEIGHTS_OK
        self.pos_float = False
        self.mo = REG_MO_UNIFORM
        self.ff = True
        self.param = _nan64()
        self.param_given = False
        self.param_word = -1
        self.refuse_word = -1
        self.flag = -1
        self.wmode = 0
        self.lv_at = 0
        self.lv_n = 0
        self.ids = False
        self.fpred = 0
        self.outs = 0
        self.counts_at = 0
        self.mheld_at = 0
        self.fold = 0
        self.tier = 0
        self.tier_word = -1
        self.log = False
        self.cpu_fast = False


def _nan64() -> Float64:
    var zero = Float64(0.0)
    return zero / zero


struct MselScorerOptions(ImplicitlyCopyable, Movable):
    """A make_scorer's kwargs over a table row, read ONCE at UNPACK (lane
    mf-tail5b; module note). `sel_from` / `sel_n`: the `labels` codes'
    place among the plan's selection words. Lane mf-score6: `none_word`
    average=None's scorer-words index (-1: none), `samples`
    average='samples', `w_from` / `w_n` the sample_weight words' place among
    the selection words (`w_n` -1: the metric takes no sample_weight),
    `w_word` their refusal word (SCORER_WEIGHTS_OK / _EMPTY, else a label
    word index), `pos_float` a float pos_label (with `pos_given`)."""
    var flip: Bool
    var pos_given: Bool
    var pos_word: Int
    var k: Int
    var beta: Float64
    var zd: Int
    var sel_from: Int
    var sel_n: Int
    var none_word: Int
    var samples: Bool
    var w_from: Int
    var w_n: Int
    var w_word: Int
    var pos_float: Bool
    # lane mf-score8 (module note): `mo`, `ff` (-1 the row's), `param` with
    # `param_word` (-1: not given), `refuse_word`, `flag` (-1 the row's),
    # `wmode`, and the label value words' place (`lv_from`, `lv_n` labels).
    var mo: Int
    var ff: Int
    var param: Float64
    var param_word: Int
    var refuse_word: Int
    var flag: Int
    var wmode: Int
    var lv_from: Int
    var lv_n: Int
    # lane mf-r9c: the scorer's own response (0: the row's; module note)
    var resp: Int
    # lane mf-r9d: the numeric_mode tier (0 the default's route, 1 the other
    # tier's), its refusal word (-1: none) and the predict_log_proba response
    var tier: Int
    var tier_word: Int
    var log: Bool

    def __init__(out self):
        self.flip = False
        self.pos_given = False
        self.pos_word = -1
        self.k = 0
        self.beta = -1.0
        self.zd = -1
        self.sel_from = 0
        self.sel_n = 0
        self.none_word = -1
        self.samples = False
        self.w_from = 0
        self.w_n = 0
        self.w_word = SCORER_WEIGHTS_OK
        self.pos_float = False
        self.mo = REG_MO_UNIFORM
        self.ff = -1
        self.param = 0.0
        self.param_word = -1
        self.refuse_word = -1
        self.flag = -1
        self.wmode = 0
        self.lv_from = 0
        self.lv_n = 0
        self.resp = 0
        self.tier = 0
        self.tier_word = -1
        self.log = False


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
    # scikit-learn 1.9's name (lane mf-score8: was 'max_error'; same code)
    t.append(MselScorer("neg_max_error", _F_REG, REG_MAX_ERROR, 0, -1.0, MEMBER_OP_PREDICT))
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
    # lane mf-fitadm7 (appended: earlier codes unchanged): METRIC ROWS, no
    # get_scorer name. A native make_scorer over zero_one_loss (normalize
    # True or False) or hamming_loss, response 'predict', is matched to these
    # by its metric (`msel_scorer_metric_code`); the row's sign is the
    # metric's value (+1), the make_scorer's greater_is_better=False the flip.
    t.append(MselScorer("zero_one_loss", _F_COUNT, CNT_LOSS, 0, 1.0, MEMBER_OP_PREDICT))
    t.append(MselScorer("zero_one_loss(normalize=False)", _F_COUNT, CNT_COUNT, 0, 1.0, MEMBER_OP_PREDICT))
    t.append(MselScorer("hamming_loss", _F_COUNT, _CNT_HAMMING, 0, 1.0, MEMBER_OP_PREDICT))
    # lane mf-score8 (appended: earlier codes unchanged): METRIC ROWS of the
    # regression metrics with an alpha or a power, accuracy_score's count and
    # cohen_kappa_score (`msel_scorer_metric_code`; their sign the metric's
    # value, a make_scorer's greater_is_better=False the flip).
    t.append(MselScorer("mean_pinball_loss", _F_REG, REG_PINBALL, 0, 1.0, MEMBER_OP_PREDICT))
    t.append(MselScorer("mean_tweedie_deviance", _F_REG, REG_TWEEDIE, 0, 1.0, MEMBER_OP_PREDICT))
    t.append(MselScorer("d2_tweedie_score", _F_REG, REG_D2_TWEEDIE, 0, 1.0, MEMBER_OP_PREDICT))
    t.append(MselScorer("d2_pinball_score", _F_REG, REG_D2_PINBALL, 0, 1.0, MEMBER_OP_PREDICT))
    t.append(MselScorer("accuracy_score(normalize=False)", _F_COUNT, _CNT_CORRECT, 0, 1.0, MEMBER_OP_PREDICT))
    t.append(MselScorer("cohen_kappa_score", _F_KAPPA, 0, 0, 1.0, MEMBER_OP_PREDICT))
    # lane mf-score8 round b (appended): hinge_loss (a metric row), the
    # get_scorer rows 'd2_log_loss_score' / 'd2_brier_score' (the full
    # predict_proba) and 'positive_likelihood_ratio' /
    # 'neg_negative_likelihood_ratio' (scikit-learn 1.9 metrics/_scorer.py)
    t.append(MselScorer("hinge_loss", _F_HINGE, 0, 0, 1.0, MEMBER_OP_DECISION))
    t.append(MselScorer("d2_log_loss_score", _F_D2P, 1, 0, 1.0, MEMBER_OP_PREDICT_PROBA))
    t.append(MselScorer("d2_brier_score", _F_D2P, 0, 0, 1.0, MEMBER_OP_PREDICT_PROBA))
    t.append(MselScorer("positive_likelihood_ratio", _F_CLR, 0, 0, 1.0, MEMBER_OP_PREDICT))
    t.append(MselScorer("neg_negative_likelihood_ratio", _F_CLR, 1, 0, -1.0, MEMBER_OP_PREDICT))
    # roc_auc_score's one-vs-rest averages no get_scorer name holds (a
    # make_scorer's multi_class='ovr' with average 'micro' or None)
    t.append(MselScorer("roc_auc(multi_class='ovr', average='micro')", _F_ROC, ROC_OVR_MICRO, 0, 1.0,
                        MEMBER_OP_PREDICT_PROBA))
    t.append(MselScorer("roc_auc(multi_class='ovr', average=None)", _F_ROC, ROC_OVR_NONE, 0, 1.0,
                        MEMBER_OP_PREDICT_PROBA))
    # lane mf-r9c (door :3670, appended: earlier codes unchanged): the DUAL
    # 'roc_auc' scorer (response ('decision_function', 'predict_proba')) with
    # a make_scorer's multi_class 'ovr' / 'ovo' and its average: the
    # one-vs-rest / one-vs-one programs over the response the class has
    # (`_dual`: decision first, as 'roc_auc''s pair). A decision side that
    # does not sum to one per row raises roc_auc_score's words in the
    # program (bindings/mcls_ranking_run.mojo `_ROC_ROWS_WORDS`), as
    # `_multiclass_roc_auc_score` does; a binary member's one column takes
    # the binary arm (scikit-learn ignores multi_class there).
    t.append(MselScorer("roc_auc(multi_class='ovr')", _F_ROC, ROC_OVR, 0, 1.0, MEMBER_OP_DECISION))
    t.append(MselScorer("roc_auc(multi_class='ovr')", _F_ROC, ROC_OVR, 0, 1.0, MEMBER_OP_PREDICT_PROBA))
    t.append(MselScorer("roc_auc(multi_class='ovo')", _F_ROC, ROC_OVO, 0, 1.0, MEMBER_OP_DECISION))
    t.append(MselScorer("roc_auc(multi_class='ovo')", _F_ROC, ROC_OVO, 0, 1.0, MEMBER_OP_PREDICT_PROBA))
    t.append(MselScorer("roc_auc(multi_class='ovr', average='weighted')", _F_ROC, ROC_OVR_WEIGHTED, 0, 1.0,
                        MEMBER_OP_DECISION))
    t.append(MselScorer("roc_auc(multi_class='ovr', average='weighted')", _F_ROC, ROC_OVR_WEIGHTED, 0, 1.0,
                        MEMBER_OP_PREDICT_PROBA))
    t.append(MselScorer("roc_auc(multi_class='ovo', average='weighted')", _F_ROC, ROC_OVO_WEIGHTED, 0, 1.0,
                        MEMBER_OP_DECISION))
    t.append(MselScorer("roc_auc(multi_class='ovo', average='weighted')", _F_ROC, ROC_OVO_WEIGHTED, 0, 1.0,
                        MEMBER_OP_PREDICT_PROBA))
    # the decision halves of the micro / None rows above (their proba rows
    # serve 'roc_auc_ovr' renamed by its average too)
    t.append(MselScorer("roc_auc(multi_class='ovr', average='micro')", _F_ROC, ROC_OVR_MICRO, 0, 1.0,
                        MEMBER_OP_DECISION))
    t.append(MselScorer("roc_auc(multi_class='ovr', average=None)", _F_ROC, ROC_OVR_NONE, 0, 1.0,
                        MEMBER_OP_DECISION))
    return t^


def msel_scorer_row(code: Int, opts: MselScorerOptions, sel_base: Int) raises -> MselScorer:
    """The scorer of `code` with a make_scorer's options (once per RUN and
    scorer): the sign negated (`flip`), the caller's pos_label (a binary
    label row, 'average_precision' or 'neg_brier_score'), top_k's k, the F
    row's beta (`_fmode`: beta 0 is precision, an infinite beta recall),
    zero_division (a number silences the ill-defined warnings, as the doors'
    `_prf_warn` does) and the `labels` codes at `sel_base + 4 * sel_from`
    (an averaged label row)."""
    var row = msel_scorer_row(code)
    if opts.flip:
        row.sign = -row.sign
    if opts.pos_given:
        if row.family != _F_BINARY and row.family != _F_AP and row.family != _F_BRIER and row.family != _F_D2P:
            raise Error("RuntimeError: a given pos_label on a search scorer that reads none")
        row.pos_given = True
        row.pos_word = opts.pos_word
        row.pos_float = opts.pos_float
    if opts.k > 0:
        if row.family != _F_TOP_K:
            raise Error("RuntimeError: a given k on a search scorer other than top_k_accuracy")
        row.metric = opts.k
    if opts.beta >= 0.0:
        if (row.family != _F_LABEL and row.family != _F_BINARY) or row.metric != LBL_F1:
            raise Error("RuntimeError: a given beta on a search scorer other than an F row")
        var infinite = opts.beta - opts.beta != 0.0  # inf - inf is nan
        row.fmode = 0 if infinite else (1 if opts.beta == 0.0 else 2)
        row.b2 = opts.beta * opts.beta
    if opts.zd >= 0:
        if row.family != _F_LABEL and row.family != _F_BINARY:
            raise Error("RuntimeError: a given zero_division on a search scorer other than a label row")
        row.zd = opts.zd
        row.quiet = True
    if opts.sel_n > 0:
        # lane mf-score8: cohen_kappa_score's `labels` too (its chosen order)
        var listed = (row.family == _F_KAPPA or row.family == _F_LOG_LOSS or row.family == _F_BRIER
                      or row.family == _F_TOP_K or row.family == _F_HINGE or row.family == _F_D2P
                      or row.family == _F_ROC)
        if not listed and (row.family != _F_LABEL or row.metric == LBL_BALANCED or row.metric == LBL_MCC):
            raise Error("RuntimeError: a given labels list on a search scorer other than an averaged label row")
        row.sel_at = sel_base + 4 * opts.sel_from
        row.sel_n = opts.sel_n
    # lane mf-score6: average=None / 'samples' on an averaged label row, the
    # sample_weight words and their refusal word.
    if opts.none_word >= 0 or opts.samples:
        # lane mf-score8: average None on 'average_precision' and on roc_auc's
        # one-vs-rest row names the scorer by its words too
        var none_row = (row.family == _F_AP or row.family == _F_ROC or row.family == _F_REG) and not opts.samples
        if msel_scorer_label_row(row) != 2 and not none_row:
            raise Error("RuntimeError: average None or 'samples' on a search scorer other than an averaged label row")
        row.avg_none = opts.none_word >= 0
        row.repr_word = opts.none_word
        row.samples = opts.samples
    if opts.w_n != 0 or opts.w_word != SCORER_WEIGHTS_OK:
        if opts.w_n > 0 and not msel_scorer_weighted(row):
            raise Error("RuntimeError: sample_weight on a search scorer whose typed metric takes none")
        row.w_n = opts.w_n
        row.w_at = sel_base + 4 * opts.w_from
        row.w_word = opts.w_word
    # lane mf-score8 (module note): the regression options, the refusal word,
    # the normalize / adjusted flag, kappa's weights, the label value words.
    if opts.mo != REG_MO_UNIFORM:
        if (row.family != _F_REG or not reg_metric_takes_multioutput(row.metric)
                or (opts.mo != 1 and not reg_metric_takes_variance(row.metric))):
            raise Error("RuntimeError: a multioutput rule on a search scorer whose metric takes none")
        row.mo = opts.mo
    if opts.ff >= 0:
        if row.family != _F_REG or (row.metric != REG_R2 and row.metric != REG_EXPLAINED_VARIANCE):
            raise Error("RuntimeError: force_finite on a search scorer whose metric takes none")
        row.ff = opts.ff == 1
    if opts.param_word >= 0:
        # lane mf-rows8a: a binary label row on a non-classifier, its
        # pos_label's value (`msel_scorer_float_labels`)
        var param_row = row.family == _F_KAPPA or row.family == _F_ROC or msel_scorer_v_measure(row) or (
            row.family == _F_BINARY) or (
            row.family == _F_REG and (
            row.metric == REG_PINBALL or row.metric == REG_D2_PINBALL or row.metric == REG_TWEEDIE
            or row.metric == REG_D2_TWEEDIE))
        if not param_row:
            raise Error("RuntimeError: an alpha, power or replace_undefined_by on a search scorer that takes none")
        row.param = opts.param
        row.param_given = True
        row.param_word = opts.param_word
    row.refuse_word = opts.refuse_word
    if opts.flag >= 0:
        var flag_row = (row.family == _F_LOG_LOSS or row.family == _F_TOP_K or row.family == _F_BRIER
                        or row.family == _F_AP or msel_scorer_mi(row)
                        or (row.family == _F_LABEL and row.metric == LBL_BALANCED) or row.family == _F_KAPPA)
        if not flag_row:
            raise Error("RuntimeError: a normalize or adjusted flag on a search scorer that takes none")
        row.flag = opts.flag
    if opts.wmode != 0:
        # kappa's weights; on 'average_precision' (lane mf-score8) 1 marks a
        # given pos_label that is not 1 (the door's multiclass refusal)
        if row.family != _F_KAPPA and row.family != _F_AP:
            raise Error("RuntimeError: kappa weights on a search scorer other than cohen_kappa_score")
        row.wmode = opts.wmode
    if opts.lv_n > 0:
        # A regression row on a classifier (lane mf-score8), and every row on
        # a MULTI-OUTPUT classifier: its label and clustering rows (lane
        # mf-rows8b, `_multi_output_refusal` reads their values) and its
        # probability rows (lane mf-r9c, `_multi_classes`). Lane mf-r9c: the
        # guard that refused label value words on any row but a regression row
        # is removed (it refused rows8b's own multi-output label rows, whose
        # UNPACK hands them).
        row.lv_at = sel_base + 4 * opts.lv_from
        row.lv_n = opts.lv_n
    if opts.resp != 0:
        # lane mf-r9c (door :3670): a regression row over the make_scorer's
        # own response (`_response_regression`)
        if row.family != _F_REG or (opts.resp != MEMBER_OP_PREDICT_PROBA and opts.resp != MEMBER_OP_DECISION):
            raise Error("RuntimeError: a scorer's own response on a search scorer other than a regression row")
        row.op = opts.resp
    # lane mf-r9d (door :3789): the other numeric_mode tier's route and a
    # predict_log_proba response (a probability response's log)
    row.tier = opts.tier
    row.tier_word = opts.tier_word
    if opts.log:
        if row.op != MEMBER_OP_PREDICT_PROBA:
            raise Error("RuntimeError: predict_log_proba on a search scorer that reads no predict_proba")
        row.log = True
    return row


def msel_scorer_weighted(row: MselScorer) -> Bool:
    """Lane mf-score6: True for a row whose typed metric takes sample_weight:
    every row but max_error and the clustering scorers, whose metrics have no
    such parameter (scikit-learn's signatures and the doors')."""
    if row.family == _F_CLUSTER or row.family == _F_CLR:
        return False  # lane mf-score8: the likelihood-ratio metrics take none either
    return not (row.family == _F_REG and row.metric == REG_MAX_ERROR)


def msel_scorer_row(code: Int) raises -> MselScorer:
    """The table row of `code` with the row's own options."""
    if code == SCORER_ACCURACY:
        return MselScorer("accuracy", _F_ACCURACY, 0, 0, 1.0, MEMBER_OP_PREDICT)
    var t = _table()
    if code < 2 or code - 2 >= len(t):
        raise Error("RuntimeError: unknown search scorer code " + String(code))
    return t[code - 2]


def label_accuracy_value(route: MetricRoute, truth: Int, pred: Int, count: Int, classes: Int) raises -> Float64:
    """`label_accuracy_value` unweighted."""
    return label_accuracy_value(route, truth, pred, count, classes, 0)


def label_accuracy_value(route: MetricRoute, truth: Int, pred: Int, count: Int, classes: Int,
                         w: Int) raises -> Float64:
    """`accuracy_score(y_true, y_pred)` of one side through the native metric
    program (the plan `labels_accuracy` runs; codes in 0 .. classes-1; was
    core/msel_search_run.mojo `_accuracy`). `w` (lane mf-score6): count
    float32 sample weights (0: none), the program's third input as the
    door's weighted accuracy reads them."""
    var acc = label_accuracy_plan(count, classes, w != 0)
    var addresses: List[Int] = [truth, pred]
    var sizes: List[Int] = [count, count]
    if w != 0:
        addresses.append(w)
        sizes.append(count)
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


def msel_scorer_metric_name(metric: String, normalize: Bool) -> String:
    """Lane mf-score8: the row name of the connected metric `metric` (its
    plan name) as a make_scorer with response 'predict' wraps it: its
    get_scorer row (r2_score is 'r2', mean_squared_error
    'neg_mean_squared_error', ...) or its METRIC ROW; `normalize` picks
    zero_one_loss's and accuracy_score's row. "": no typed row."""
    if metric == "zero_one_loss":
        return String("zero_one_loss") if normalize else String("zero_one_loss(normalize=False)")
    if metric == "accuracy_score":
        return String("accuracy") if normalize else String("accuracy_score(normalize=False)")
    if (metric == "hamming_loss" or metric == "mean_pinball_loss" or metric == "mean_tweedie_deviance"
            or metric == "d2_tweedie_score" or metric == "d2_pinball_score" or metric == "cohen_kappa_score"
            or metric == "d2_absolute_error_score" or metric == "matthews_corrcoef" or metric == "hinge_loss"
            or metric == "d2_log_loss_score" or metric == "d2_brier_score" or metric == "positive_likelihood_ratio"):
        return metric.copy()
    if metric == "max_error":
        return String("neg_max_error")
    if metric == "negative_likelihood_ratio":
        return String("neg_negative_likelihood_ratio")
    if metric == "r2_score":
        return String("r2")
    if metric == "explained_variance_score":
        return String("explained_variance")
    if metric == "balanced_accuracy_score":
        return String("balanced_accuracy")
    if (metric == "mean_squared_error" or metric == "mean_absolute_error" or metric == "root_mean_squared_error"
            or metric == "mean_squared_log_error" or metric == "root_mean_squared_log_error"
            or metric == "mean_absolute_percentage_error" or metric == "median_absolute_error"
            or metric == "mean_poisson_deviance" or metric == "mean_gamma_deviance"):
        return "neg_" + metric
    return String("")


def msel_scorer_metric_code(metric: String, normalize: Bool) -> Int:
    """UNPACK's typed lookup of a METRIC ROW (lane mf-fitadm7; lane
    mf-score8: every connected metric with a typed row,
    `msel_scorer_metric_name`): the code of a native make_scorer over the
    connected metric `metric` (its plan name) with response 'predict', or
    -1 when no row serves it."""
    var name = msel_scorer_metric_name(metric, normalize)
    if name.byte_length() == 0:
        return -1
    if name == "accuracy":
        return SCORER_ACCURACY
    var t = _table()
    for i in range(len(t)):  # driver-loop: over the served scorer rows (a fixed table)
        if String(t[i].name) == name:
            return i + 2
    return -1


def msel_scorer_group_words(mut words: List[Float64], code: Int, positive: Int, opts: MselScorerOptions,
                            sels: List[Int32]):
    """UNPACK: one scorer's word group (module note), its `labels` codes
    read from `sels` at `opts.sel_from`. The one writer of the layout."""
    words.append(Float64(code))
    words.append(Float64(positive))
    words.append(1.0 if opts.flip else 0.0)
    words.append((2.0 if opts.pos_float else 1.0) if opts.pos_given else 0.0)
    words.append(Float64(opts.pos_word))
    words.append(Float64(opts.k))
    words.append(opts.beta)
    words.append(Float64(opts.zd))
    words.append(Float64(opts.sel_n))
    words.append(Float64(opts.none_word))
    words.append(1.0 if opts.samples else 0.0)
    words.append(Float64(opts.w_n))
    words.append(Float64(opts.w_word))
    # lane mf-score8
    words.append(Float64(opts.mo))
    words.append(Float64(opts.ff))
    words.append(opts.param)
    words.append(Float64(opts.param_word))
    words.append(Float64(opts.refuse_word))
    words.append(Float64(opts.flag))
    words.append(Float64(opts.wmode))
    words.append(Float64(opts.lv_n))
    words.append(Float64(opts.resp))  # lane mf-r9c
    words.append(Float64(opts.tier))  # lane mf-r9d
    words.append(Float64(opts.tier_word))
    words.append(1.0 if opts.log else 0.0)
    for j in range(opts.sel_n):  # small-loop(the scorer's labels codes)
        words.append(Float64(sels[opts.sel_from + j]))
    for j in range(max(opts.w_n, 0)):  # driver-loop: over the sample_weight words (once per admission)
        words.append(Float64(sels[opts.w_from + j]))
    for j in range(2 * opts.lv_n):  # small-loop(the fit labels' value words, two each)
        words.append(Float64(sels[opts.lv_from + j]))


def msel_scorer_group_read(words: List[Float64], at: Int, mut code: Int, mut positive: Int,
                           mut opts: MselScorerOptions, mut sels: List[Int32]) raises -> Int:
    """The one reader of a scorer's word group at `at`: its code, positive
    word and options (the `labels` codes appended to `sels`). Answers the
    index after the group; raises on a short or malformed group."""
    if at < 0 or at + SCORER_GROUP_WORDS > len(words):
        raise Error("member ABI: a search scorer's word group is cut short")
    code = Int(words[at])
    positive = Int(words[at + 1])
    opts = MselScorerOptions()
    opts.flip = words[at + 2] != 0.0
    opts.pos_given = words[at + 3] != 0.0
    opts.pos_float = words[at + 3] == 2.0
    opts.pos_word = Int(words[at + 4])
    opts.k = Int(words[at + 5])
    opts.beta = words[at + 6]
    opts.zd = Int(words[at + 7])
    var n = Int(words[at + 8])
    opts.none_word = Int(words[at + 9])
    opts.samples = words[at + 10] != 0.0
    var wn = Int(words[at + 11])
    opts.w_word = Int(words[at + 12])
    # lane mf-score8
    opts.mo = Int(words[at + 13])
    opts.ff = Int(words[at + 14])
    opts.param = words[at + 15]
    opts.param_word = Int(words[at + 16])
    opts.refuse_word = Int(words[at + 17])
    opts.flag = Int(words[at + 18])
    opts.wmode = Int(words[at + 19])
    var lv = Int(words[at + 20])
    opts.resp = Int(words[at + 21])  # lane mf-r9c
    opts.tier = Int(words[at + 22])  # lane mf-r9d
    opts.tier_word = Int(words[at + 23])
    opts.log = words[at + 24] != 0.0
    if opts.tier < 0 or opts.tier > 1:
        raise Error("member ABI: a search scorer's numeric_mode tier word is not 0 or 1")
    var wk = max(wn, 0)
    if (n < 0 or wn < -1 or lv < 0 or opts.w_word < SCORER_WEIGHTS_EMPTY
            or at + SCORER_GROUP_WORDS + n + wk + 2 * lv > len(words)):
        raise Error("member ABI: a search scorer's labels codes are cut short")
    opts.sel_from = len(sels)
    opts.sel_n = n
    for j in range(n):  # small-loop(the scorer's labels codes)
        sels.append(Int32(Int(words[at + SCORER_GROUP_WORDS + j])))
    opts.w_from = len(sels)
    opts.w_n = wn
    for j in range(wk):  # driver-loop: over the sample_weight words (once per fit)
        sels.append(Int32(Int(words[at + SCORER_GROUP_WORDS + n + j])))
    opts.lv_from = len(sels)
    opts.lv_n = lv
    for j in range(2 * lv):  # small-loop(the fit labels' value words, two each; once per fit)
        sels.append(Int32(Int(words[at + SCORER_GROUP_WORDS + n + wk + j])))
    return at + SCORER_GROUP_WORDS + n + wk + 2 * lv


def msel_scorer_top_k(row: MselScorer) -> Bool:
    """True for 'top_k_accuracy' (its k is a make_scorer kwarg)."""
    return row.family == _F_TOP_K


def msel_scorer_pos_words(row: MselScorer, label_words: List[String], positive: Int) -> String:
    """The pos_label as scikit-learn's refusal prints it (`f"pos_label={pos_label}"`):
    1 unless the caller gave it (lane mf-msel5c), then its label's word
    (lane mf-tail5b: the word the UNPACK appended for it, `pos_word`, also
    for a label that is not one of the fit's)."""
    if not row.pos_given:
        return String("1")
    var at = row.pos_word if row.pos_word >= 0 else positive
    if at < 0 or at >= len(label_words):
        return String("1")
    var word = label_words[at]
    var n = word.byte_length()
    if n >= 2 and word.startswith("'") and word.endswith("'"):
        return String(word[byte=1 : n - 1])
    return word


def msel_scorer_label_row(row: MselScorer) -> Int:
    """UNPACK (lane mf-tail5b): which label options a row takes: 2 an
    averaged precision / recall / F / jaccard row (zero_division, `labels`,
    an F row's beta), 1 a binary one (zero_division, pos_label, beta; its
    `labels` ignored as `_check_set_wise_labels` ignores them), 0 none."""
    if row.family == _F_BINARY:
        return 1
    if row.family == _F_LABEL and row.metric != LBL_BALANCED and row.metric != LBL_MCC:
        return 2
    return 0


def msel_scorer_f_row(row: MselScorer) -> Bool:
    """True for an F row (f1 and its averages): the rows 'fbeta' maps onto."""
    return (row.family == _F_LABEL or row.family == _F_BINARY) and row.metric == LBL_F1


def msel_scorer_brier(row: MselScorer) -> Bool:
    """True for 'neg_brier_score' (lane mf-msel5c: a fold that lacks a class)
    and (lane mf-score8) the D^2 probability scores, which read their
    probabilities as brier_score_loss's door does (`_proba`: the label marks
    or a given pos_label for a one-column side, the same checks)."""
    return row.family == _F_BRIER or row.family == _F_D2P


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
    the UNPACK picks one by the class's methods. Lane mf-r9c: the
    multi_class rows of the dual 'roc_auc' scorer too (their names start
    "roc_auc(multi_class=")."""
    if row.family == _F_ROC and String(row.name).startswith("roc_auc(multi_class="):
        return True
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
    if row.family == _F_BRIER or row.family == _F_D2P:
        return SCORER_POSITIVE_MARKS
    return SCORER_POSITIVE_NONE


def msel_scorer_labels_from_truth(row: MselScorer) -> Bool:
    """True for the scorers whose metric takes its label set from the side's
    `y_true` (labels=None in scikit-learn: roc_auc and its ovr / ovo forms
    through `np.unique(y_true)`, log_loss through `LabelBinarizer().fit(y_true)`,
    top_k_accuracy through `np.unique(y_true)`): a fold estimator's own
    columns are read against THOSE labels (lane mf-kmeans5, the fold that
    lacks a class)."""
    # lane mf-score8: hinge_loss and the D^2 probability scores too; a
    # given `labels` list is the label set instead (no ranking by y_true)
    if row.sel_n > 0:
        return False
    return (row.family == _F_ROC or row.family == _F_LOG_LOSS or row.family == _F_TOP_K or row.family == _F_HINGE
            or row.family == _F_D2P)


def msel_scorer_roc_problem(row: MselScorer) -> Int:
    """Lane mf-score8: a roc_auc row's problem (ROC_*), -1 for another row."""
    return row.metric if row.family == _F_ROC else -1


def msel_scorer_mi(row: MselScorer) -> Bool:
    """Lane mf-score8: NMI or AMI (their `average_method` is the flag word)."""
    return row.family == _F_CLUSTER and row.metric < _CL_PAIR


def msel_scorer_v_measure(row: MselScorer) -> Bool:
    """Lane mf-score8: v_measure_score (its `beta` is the param word)."""
    return row.family == _F_CLUSTER and row.metric == _CL_PAIR + CLUSTER_OP_V_MEASURE


def msel_scorer_d2_proba(row: MselScorer) -> Bool:
    """Lane mf-score8: d2_log_loss_score / d2_brier_score."""
    return row.family == _F_D2P


def msel_scorer_float_roc(row: MselScorer) -> Bool:
    """Lane mf-r9d (door :3960): 'roc_auc' (its binary row, multi_class
    'raise', either response half) on a NON-classifier whose class has the
    response (an outlier detector's decision_function, a mixture model's
    predict_proba): scikit-learn's scorer hands the raw answer to
    roc_auc_score with the float targets (utils/_response.py line 232),
    `_float_roc_value`."""
    return row.family == _F_ROC and row.metric == ROC_BINARY


def _float_roc_value(route: MetricRoute, cluster: EntryRef, label_words: List[String], row: MselScorer, truth: Int,
                     response: Int, count: Int, cols: Int, positive: Int,
                     mut notices: List[Float64]) raises -> Float64:
    """Lane mf-r9d (door :3960): roc_auc_score(y_true, raw answer) of a
    non-classifier's side, in scikit-learn 1.9's order (metrics/_ranking.py
    `roc_auc_score` lines 708-747, `_average_binary_score`,
    `_binary_roc_auc_score`): `type_of_target(y_true)` multiclass, or binary
    with more than two score columns, is the multiclass branch, whose
    multi_class 'raise' refuses (the words the classifier rows raise);
    continuous fails `_average_binary_score` ("continuous format is not
    supported"); binary codes y_true by its sorted distinct values
    (`label_binarize(..., classes=np.unique(y_true))[:, 0]`: the larger value
    is positive). One class present warns and answers nan
    (`_binary_roc_auc_score`, MEMBER_NOTICE_METRIC_ROC_ONE); two score
    columns then refuse in `column_or_1d`'s words; one column is the
    classifier row's program over the codes (two classes, the score as the
    positive's)."""
    var t = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=truth)
    var yt = List[Float64](capacity=max(count, 1))
    for i in range(count):  # driver-loop: over the side's rows (the targets as numbers)
        yt.append(Float64(t[i]))
    var kt = _target_type(yt)
    var width = max(cols, 1)
    if kt == 1 or (kt == 0 and width > 2):
        raise Error("multi_class must be in ('ovo', 'ovr')")
    if kt == 2:
        raise Error("ValueError: continuous format is not supported")
    # the sorted distinct values (at most two): each value's code
    var union = List[Float64]()
    for i in range(count):  # driver-loop: over the side's rows (the distinct values)
        var seen = False
        for j in range(len(union)):  # small-loop(the distinct values so far, at most two)
            seen = seen or union[j] == yt[i]
        if not seen:
            union.append(yt[i])
    if len(union) == 2 and union[0] > union[1]:
        var v = union[0]
        union[0] = union[1]
        union[1] = v
    if len(union) < 2:
        # one class present: `_binary_roc_auc_score` warns and answers nan
        # before roc_curve reads the score's shape
        member_notice_add(notices, MEMBER_NOTICE_METRIC_ROC_ONE, 1.0)
        return _nan64()
    if width == 2:
        raise Error("ValueError: y should be a 1d array, got an array of shape (" + String(count) + ", 2) instead.")
    var codes = List[Int32](length=max(count, 1), fill=Int32(0))
    for i in range(count):  # driver-loop: over the side's rows (their codes)
        if len(union) == 2 and yt[i] == union[1]:
            codes[i] = Int32(1)
    var value = msel_scorer_value(route, cluster, label_words, row, Int(codes.unsafe_ptr()), response, count, 2, 1,
                                  positive, notices)
    _ = len(codes)  # read by the program above
    return value


def msel_scorer_float_labels(row: MselScorer) -> Bool:
    """Lane mf-score8 round b: a label row a non-classifier (a regressor, or
    a clusterer's ids) is scored by, its targets and predictions read as
    labels per side as scikit-learn's `_check_targets` reads them (integral
    values: 'binary' / 'multiclass'; any other: 'continuous', refused in its
    words): accuracy, the averaged label rows and balanced accuracy /
    matthews, the COUNT rows, cohen_kappa_score, the likelihood ratios, and
    (lane mf-rows8a) the binary label rows, their pos_label a number the
    UNPACK read once (`param`, printed as given at `param_word`) found among
    the side's float labels (`_float_binary_positive`)."""
    return (row.family == _F_ACCURACY or row.family == _F_LABEL or row.family == _F_COUNT or row.family == _F_KAPPA
            or row.family == _F_CLR or row.family == _F_BINARY)


def msel_scorer_decision_only(row: MselScorer) -> Bool:
    """Lane mf-score8: a row whose response is decision_function alone
    (hinge_loss): a class without the method fails in scikit-learn's words."""
    return row.family == _F_HINGE


def msel_scorer_log_loss(row: MselScorer) -> Bool:
    """True for 'neg_log_loss'."""
    return row.family == _F_LOG_LOSS


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
    return (row.family == _F_LOG_LOSS or row.family == _F_ROC
            or row.family == _F_AP or row.family == _F_BRIER or row.family == _F_TOP_K or row.family == _F_D2P)


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
    # Lane mf-score6, in the door's order (the call's keywords, then
    # bindings/mcls_classify_door.mojo `mc_pair`'s weights, then
    # `_set_wise_labels`' average): a sample_weight the metric does not take
    # (the scorer door's words), the weights' refusal read at UNPACK, their
    # length and values per side (bindings/metrics_reg_door.mojo
    # `_reg_weights`' words), average='samples' (1-D targets; the door's
    # words: multilabel targets are NOT IMPLEMENTED, metrics/NOT_IMPLEMENTED.tsv).
    # LANE mf-r9c: a probability row on a MULTI-OUTPUT classifier's side.
    # scikit-learn's `_get_response_values` reads `type_of_target` of the fold
    # estimator's `classes_` (a list, one array per output) before the metric
    # runs; a 'binary' list fails there (`_multi_classes`).
    var mclasses = -1
    var mwidths = List[Int]()
    if row.outs > 1 and row.counts_at != 0 and row.op != MEMBER_OP_PREDICT:
        mclasses = _multi_classes(row, label_words, mwidths)
        if mclasses == _MC_BINARY:
            raise Error("AttributeError: 'list' object has no attribute " + (String("'tolist'") if row.pos_given
                        else String("'shape'")))
    if row.w_n < 0:
        raise Error("TypeError: native metric received an unexpected keyword argument")
    # LANE mf-r9d (door :3789): the native scorer door's mode binding, after
    # its kwargs check and before the metric reads anything (`prepare`): the
    # other tier's image refused (its loader's words, read once at UNPACK),
    # then a host estimator's fast tier
    if row.tier_word >= 0:
        if row.tier_word >= len(label_words):
            raise Error("member ABI: a search scorer's numeric_mode refusal word is missing")
        raise Error(label_words[row.tier_word])
    if row.cpu_fast:
        raise Error(SCORER_CPU_FAST_WORDS)
    if row.w_word == SCORER_WEIGHTS_EMPTY:
        raise Error("mojolearn " + msel_scorer_caller(row) + ": sample_weight has 0 entries for " + String(count)
                    + " samples")
    if row.w_word >= 0:
        if row.w_word >= len(label_words):
            raise Error("member ABI: a search scorer's sample_weight refusal word is missing")
        raise Error(label_words[row.w_word])
    var w = 0
    if row.w_n > 0:
        _check_weights(row, count)
        w = row.w_at
    if row.refuse_word >= 0:
        # lane mf-score8: a kwarg value the metric's door refuses (module note)
        if row.refuse_word >= len(label_words):
            raise Error("member ABI: a search scorer's refusal word is missing")
        raise Error(label_words[row.refuse_word])
    if mclasses >= 0:
        # lane mf-r9c: the metric over the multi-output response, after the
        # kwargs' refusals (the metric's parameter validation)
        return row.sign * _multi_proba_value(route, row, label_words, truth, response, count, cols, w, mclasses,
                                             mwidths, notices)
    if row.outs > 1 and row.family != _F_REG:
        # LANE mf-rows8b (rows :3777 / :3779): a label or clustering scorer on
        # a multi-output target, after the kwargs' refusals (scikit-learn's
        # parameter validation runs first), in scikit-learn's words per side
        _multi_output_refusal(row, truth, response, count)
    if row.samples:
        raise Error("NotImplementedError: mojolearn " + msel_scorer_caller(row) + ": average='samples' applies to "
                    + "multilabel targets, which are NOT IMPLEMENTED (metrics/NOT_IMPLEMENTED.tsv)")
    if classes == 0 and msel_scorer_float_roc(row):
        # lane mf-r9d (door :3960): 'roc_auc' on a non-classifier's own columns
        return _float_roc_value(route, cluster, label_words, row, truth, response, count, cols, positive, notices)
    if classes == 0 and msel_scorer_float_labels(row):
        # lane mf-score8 round b: a label row on a non-classifier's side
        return _float_label_value(route, cluster, label_words, row, truth, response, count, cols, positive, notices)
    if row.family == _F_ACCURACY:
        return label_accuracy_value(route, truth, response, count, classes, w)
    var value: Float64
    var warn = 0
    if row.family == _F_REG:
        var shown = String("")
        if row.param_word >= 0 and row.param_word < len(label_words):
            shown = label_words[row.param_word].copy()
        if row.op != MEMBER_OP_PREDICT:
            # lane mf-r9c (door :3670): over the make_scorer's own response
            value = _response_regression(route, label_words, row, truth, response, count, classes, cols, w, shown,
                                         warn)
        elif row.outs > 1 and row.counts_at != 0:
            # lane mf-rows8b (row :3779): a regression scorer on a MULTI-OUTPUT
            # classifier: each output's codes as its labels' values, then the
            # rows x outputs program a multi-output regressor's side runs
            var per_output = List[Float64]()
            value = _multi_label_regression(route, label_words, row, truth, response, count, w, shown, per_output,
                                            warn)
            if len(per_output) > 1:
                _number_error(row, label_words, per_output)
        elif row.lv_n > 0:
            # lane mf-score8: a regression scorer on a classifier (module note)
            value = _label_regression(route, label_words, row, truth, response, count, classes, w, shown, warn)
        elif row.ids:
            # lane mf-score8 round b: a clusterer's ids scored as numbers
            var p = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=response)
            var floats = List[Float32](length=max(count, 1), fill=0.0)
            for i in range(count):  # driver-loop: over the side's rows (the ids as float32)
                floats[i] = Float32(Int(p[i]))
            value = reg_metric_options_run(route, row.metric, truth, Int(floats.unsafe_ptr()), count, w, row.mo, row.ff,
                                           row.param, row.param_given, shown, warn)
            _ = len(floats)  # read by the program above
        else:
            # lane mf-score8 round b: `cols` outputs (a multi-output regressor's
            # sides, rows x outputs); 'raw_values' over several is no number
            var per_output = List[Float64]()
            value = reg_metric_options_run(route, row.metric, truth, response, count, max(cols, 1), w, row.mo, row.ff,
                                           row.param, row.param_given, shown, per_output, warn)
            if len(per_output) > 1:
                _number_error(row, label_words, per_output)
        if warn != 0:
            member_notice_add(notices, MEMBER_NOTICE_METRIC_FEW, 1.0 if reg_metric_d2(row.metric) else 0.0)
    elif row.family == _F_BINARY:
        value = _binary_value(route, label_words, row, truth, response, count, classes, positive, w, warn)
        if not row.quiet:
            _prf_notices(row.metric, warn, notices)
    elif row.family == _F_LABEL:
        var per_class = List[Float64]()
        var zd = row.zd
        if row.metric == LBL_BALANCED:
            zd = 1 if row.flag == 1 else 0  # lane mf-score8: balanced_accuracy_score's `adjusted`
        value = label_metric_options_run(route, row.metric, 0 if row.avg_none else row.avg, truth, response, count,
                                         classes, zd, row.fmode, row.b2, row.sel_at, row.sel_n, w, per_class,
                                         warn)
        if row.avg_none:
            # average=None (lane mf-score6): the per-label array. scikit-learn's
            # `_score` takes `.item()` of a one-label array as the score and
            # refuses any other (module note, SCORER_NUMBER_WORDS).
            if len(per_class) != 1:
                var shown = String(row.name)
                if row.repr_word >= 0 and row.repr_word < len(label_words):
                    shown = label_words[row.repr_word].copy()
                raise Error(SCORER_NUMBER_WORDS + _ndarray_words(per_class, row.sign)
                            + " (<class 'numpy.ndarray'>) instead. (scorer=" + shown + ")")
            value = per_class[0]
        if row.metric == LBL_BALANCED:
            if warn != 0:
                member_notice_add(notices, MEMBER_NOTICE_METRIC_BALANCED)
        elif row.metric != LBL_MCC and not row.quiet:
            _prf_notices(row.metric, warn, notices)
    elif row.family == _F_AP:
        # lane mf-score8: its multiclass `average` (the flag word) and a given
        # pos_label that is not 1 (`wmode` 1)
        var ap_per = List[Float64]()
        value = average_precision_run(route, truth, response, count, cols, classes, positive, w,
                                      row.flag if row.flag >= 0 else AP_MACRO, row.wmode == 1, ap_per)
        if row.flag == AP_NONE and len(ap_per) > 1:
            _number_error(row, label_words, ap_per)
    elif row.family == _F_BRIER:
        if row.pos_float:
            # a float pos_label (lane mf-score6): the door's `mx_pos_label` refusal
            raise Error(SCORER_FLOAT_POS_WORDS)
        # a given pos_label (lane mf-tail5b): its class code, else the label marks
        value = brier_options_run(route, truth, response, count, classes, cols, -1 if row.pos_given else positive,
                                  positive if row.pos_given else -1, w, row.flag, row.sel_at, row.sel_n)
    elif row.family == _F_TOP_K:
        value = top_k_options_run(route, truth, response, count, cols, classes, row.metric, w, row.flag != 0,
                                  row.sel_at, row.sel_n, warn)
        if warn != 0:
            var shown_classes = row.sel_n if row.sel_n > 0 else _present_count(truth, count, classes)
            member_notice_add(notices, MEMBER_NOTICE_METRIC_TOP_K, Float64(row.metric), Float64(shown_classes))
    elif row.family == _F_CLUSTER:
        value = _cluster_value(route, cluster, row, truth, response, count, classes)
    elif row.family == _F_COUNT:
        # lane mf-fitadm7: bindings/metrics_exp_cls_door.mojo `_zero_one` /
        # `_hamming` over the same program (`label_count_run`), their
        # ZeroDivisionError words on a zero total
        var zero = False
        var answer = CNT_LOSS
        if row.metric == CNT_COUNT or row.metric == _CNT_CORRECT:
            answer = row.metric  # lane mf-score8: accuracy_score normalize=False's correct count
        value = label_count_run(route, answer, truth, response, count, classes, w, zero)
        if zero:
            raise Error("ZeroDivisionError: " + (String("float division by zero") if (row.metric == _CNT_HAMMING
                        or w != 0) else String("division by zero")))
    elif row.family == _F_LOG_LOSS:
        if cols != (1 if classes == 2 else classes):
            raise Error("probability columns must match the label count (at least two)")
        # lane mf-score8: log_loss's `normalize` (flag 0: the summed loss)
        value = log_loss_options_run(route, truth, response, count, classes, w, row.flag != 0, row.sel_at, row.sel_n)
    elif row.family == _F_KAPPA:
        # lane mf-score8: bindings/metrics_exp_cls_door.mojo `_kappa`'s program
        # (`kappa_run`); its two warnings, then its replace_undefined_by
        var flags = 0
        value = kappa_run(route, truth, response, count, classes, row.sel_at, row.sel_n, row.wmode, w, flags)
        if flags != 0:
            member_notice_add(notices, MEMBER_NOTICE_METRIC_KAPPA, Float64(flags))
            if row.flag == 1:
                # lane mf-rows8a: a replace_undefined_by that is no number (the
                # flag word 1): the native scorer door refuses the object it answers
                raise Error(SCORER_REAL_WORDS)
            value = row.param if row.param_given else _nan64()
    elif row.family == _F_HINGE:
        # lane mf-score8 round b: bindings/metrics_exp_rank_door.mojo `_hinge`'s
        # program (`hinge_run`); a row outside a multiclass `labels` list
        # raises the door's KeyError of its label
        if row.sel_n > 2:
            _check_listed(label_words, row, truth, count)
        value = hinge_run(route, truth, response, count, cols, classes, row.sel_at, row.sel_n, w, String("KeyError"))
    elif row.family == _F_D2P:
        # lane mf-score8 round b: `_d2_proba`'s program (`d2_proba_run`)
        var few = False
        if row.pos_float:
            raise Error(SCORER_FLOAT_POS_WORDS)  # `_proba`'s `mx_pos_label`, as brier's
        value = d2_proba_run(route, truth, response, count, classes, cols, -1 if row.pos_given else positive,
                             positive if row.pos_given else -1, row.sel_at, row.sel_n, w, row.metric == 1, few)
        if few:
            member_notice_add(notices, MEMBER_NOTICE_METRIC_FEW, 1.0)
    elif row.family == _F_CLR:
        # lane mf-score8 round b: `_likelihood_ratios`' program (`clr_run`),
        # replace_undefined_by=1.0 as scikit-learn's two scorers pass it
        var flags = 0
        value = clr_run(route, truth, response, count, classes, w, row.metric, flags)
        if flags != 0:
            member_notice_add(notices, MEMBER_NOTICE_METRIC_CLR, Float64(flags))
    elif row.family == _F_ROC and row.metric == ROC_BINARY and not row.param_given and row.sel_n == 0 and cols == 1:
        # the default call (`_roc_auc` with its defaults)
        if classes > 2:
            raise Error("multi_class must be in ('ovo', 'ovr')")
        value = roc_auc_run(route, row.metric, truth, response, count, classes, w)
    else:
        # lane mf-score8: `_roc_options` (max_fpr, labels, the multiclass
        # averages; a binary member's one column for every multiclass row)
        var shown = String("")
        if row.param_word >= 0 and row.param_word < len(label_words):
            shown = label_words[row.param_word].copy()
        var roc_per = List[Float64]()
        var undefined = 0
        value = roc_auc_options_run(route, row.metric, truth, response, count, classes, w, cols,
                                    row.param if row.param_given else -1.0, shown, row.sel_at, row.sel_n, roc_per,
                                    undefined)
        if undefined > 0:
            member_notice_add(notices, MEMBER_NOTICE_METRIC_ROC_ONE, Float64(undefined))
        if row.metric == ROC_OVR_NONE and len(roc_per) > 1:
            _number_error(row, label_words, roc_per)
    return row.sign * value


def _number_error(row: MselScorer, label_words: List[String], per_class: List[Float64]) raises:
    """scikit-learn's `_score` refusal of a per-label array (lane mf-score6
    words, SCORER_NUMBER_WORDS), the scorer named by its repr word."""
    var shown = String(row.name)
    if row.repr_word >= 0 and row.repr_word < len(label_words):
        shown = label_words[row.repr_word].copy()
    raise Error(SCORER_NUMBER_WORDS + _ndarray_words(per_class, row.sign) + " (<class 'numpy.ndarray'>) instead. "
                + "(scorer=" + shown + ")")


def _target_type(values: List[Float64]) -> Int:
    """`type_of_target` of 1-D numbers: 0 binary (at most two distinct
    integral values), 1 multiclass (more), 2 continuous (a value that is not
    integral or not finite)."""
    var distinct = List[Float64]()
    for i in range(len(values)):  # driver-loop: over the side's values (their kind and distinct count)
        var v = values[i]
        if not (v - v == 0.0) or Float64(Int(v)) != v:
            return 2
        var seen = False
        for j in range(len(distinct)):  # small-loop(the distinct values so far, stops past two)
            seen = seen or distinct[j] == v
        if not seen:
            distinct.append(v)
            if len(distinct) > 2:
                return 1
    return 0


def _float_label_value(route: MetricRoute, cluster: EntryRef, label_words: List[String], row: MselScorer, truth: Int,
                       response: Int, count: Int, cols: Int, positive: Int,
                       mut notices: List[Float64]) raises -> Float64:
    """Lane mf-score8 round b: a label row on a regressor's or a clusterer's
    side: y_true's float32 values and the member's predictions (float32, or
    int32 ids when `row.ids`) checked as scikit-learn's `_check_targets`
    checks them, in its words; then coded by their sorted union (the labels'
    order) and scored by the row as a classifier's side."""
    var t = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=truth)
    var yt = List[Float64](capacity=max(count, 1))
    var yp = List[Float64](capacity=max(count, 1))
    for i in range(count):  # driver-loop: over the side's rows (the targets as numbers)
        yt.append(Float64(t[i]))
    if row.ids:
        var pi = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=response)
        for i in range(count):  # driver-loop: over the side's rows (the ids as numbers)
            yp.append(Float64(Int(pi[i])))
    else:
        var pf = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=response)
        for i in range(count):  # driver-loop: over the side's rows (the predictions as numbers)
            yp.append(Float64(pf[i]))
    var kt = _target_type(yt)
    var kp = _target_type(yp)
    var names: List[String] = ["binary", "multiclass", "continuous"]
    if kt != kp and not (kt <= 1 and kp <= 1):
        raise Error("ValueError: Classification metrics can't handle a mix of " + names[kt] + " and " + names[kp]
                    + " targets")
    if kt == 2:
        raise Error("ValueError: continuous is not supported")
    # the sorted union of the values: each value's code
    var union = List[Float64]()
    for i in range(count):  # driver-loop: over the side's rows (the distinct values)
        for which in range(2):  # small-loop(y_true, then y_pred)
            var v = yt[i] if which == 0 else yp[i]
            var seen = False
            for j in range(len(union)):  # small-loop(the distinct labels so far)
                seen = seen or union[j] == v
            if not seen:
                union.append(v)
    for i in range(1, len(union)):  # small-loop(the distinct labels: insertion sort, ascending)
        var v = union[i]
        var j = i - 1
        while j >= 0 and union[j] > v:  # small-loop(shift the larger labels)
            union[j + 1] = union[j]
            j -= 1
        union[j + 1] = v
    var tc = List[Int32](length=max(count, 1), fill=Int32(0))
    var pc = List[Int32](length=max(count, 1), fill=Int32(0))
    for i in range(count):  # driver-loop: over the side's rows (their codes)
        for j in range(len(union)):  # small-loop(the distinct labels)
            if union[j] == yt[i]:
                tc[i] = Int32(j)
            if union[j] == yp[i]:
                pc[i] = Int32(j)
    var side_positive = positive
    if row.family == _F_BINARY:
        # lane mf-rows8a: the binary label rows' pos_label among the side's
        # labels, `_check_set_wise_labels`' checks in its order
        side_positive = _float_binary_positive(label_words, row, union, positive)
    var value = msel_scorer_value(route, cluster, label_words, row, Int(tc.unsafe_ptr()), Int(pc.unsafe_ptr()), count,
                                  len(union), cols, side_positive, notices)
    _ = len(tc)  # read by the program above
    _ = len(pc)
    return value


def _float_binary_positive(label_words: List[String], row: MselScorer, union: List[Float64],
                           positive: Int) raises -> Int:
    """Lane mf-rows8a: a binary label row (f1 / precision / recall / jaccard,
    average 'binary') on a regressor's side, after `_check_targets` (sklearn
    1.9 metrics/_classification.py `_check_set_wise_labels`, lines 1911-1919):
    more than two labels refuse ("Target is multiclass ..."); a pos_label the
    metric door reads as no number (`row.pos_float`: None or another object,
    SCORER_FLOAT_POS_WORDS) or a string (`positive` SCORER_POSITIVE_KIND, the
    door's type words) refuse; the pos_label's value (`row.param`, printed as
    given at `param_word`) not among two present labels refuses, the labels
    printed as `f"{present_labels}"` prints NumPy's float array
    (`_numpy_float_words`); else its code, or SCORER_POSITIVE_ABSENT for one
    label present (scored as `labels=[pos_label]`: 0.0 and its warning)."""
    if len(union) > 2:
        raise Error("ValueError: Target is multiclass but average='binary'. Please choose another average setting, "
                    + "one of [None, 'micro', 'macro', 'weighted'].")
    if row.pos_float:
        raise Error(SCORER_FLOAT_POS_WORDS)
    if positive == SCORER_POSITIVE_KIND:
        raise Error("pos_label must use the same type as the targets")
    if not row.param_given or row.param_word < 0 or row.param_word >= len(label_words):
        raise Error("member ABI: a binary label scorer on a non-classifier needs its pos_label value word")
    var at = -1
    for j in range(len(union)):  # small-loop(the side's labels, at most two)
        if union[j] == row.param:
            at = j
    if at >= 0:
        return at
    if len(union) >= 2:
        raise Error("ValueError: pos_label=" + label_words[row.param_word] + " is not a valid label. It should be "
                    + "one of " + _numpy_float_words(union))
    return SCORER_POSITIVE_ABSENT


def _int_digit_list(v: Float64) -> List[Int]:
    """Lane mf-rows8a: the exact decimal digits of |v|, most significant
    first, for an integral float64 below 2^128 (a float32 target's value);
    [0] for zero."""
    var a = abs(v)
    var digits = List[Int]()
    if a < 9.0e18:
        var n = Int(a)
        if n == 0:
            digits.append(0)
            return digits^
        while n > 0:  # small-loop(the decimal digits, at most 19)
            digits.append(n % 10)
            n = n // 10
    else:
        var bits = bitcast[DType.uint64](a)
        var mant = ((bits & UInt64(0xFFFFFFFFFFFFF)) | UInt64(0x10000000000000)).cast[DType.uint128]()
        var shift = Int((bits >> UInt64(52)) & UInt64(0x7FF)) - 1075
        var n = mant << UInt128(min(max(shift, 0), 74))
        while n > UInt128(0):  # small-loop(the decimal digits, at most 39)
            digits.append(Int(n % UInt128(10)))
            n = n // UInt128(10)
    digits.reverse()
    return digits^


def _pad_left(text: String, width: Int) -> String:
    """`text.rjust(width)`."""
    var out = String("")
    for _ in range(width - text.byte_length()):  # small-loop(the padding spaces)
        out += " "
    return out + text


def _numpy_float_words(values: List[Float64]) -> String:
    """Lane mf-rows8a: `str(np.array(values))` of integral float labels
    (`_check_targets`' present labels: never continuous here), as NumPy's
    default print options print them (precision 8, floatmode 'maxprec',
    sign '-'; numpy/_core/arrayprint.py `FloatingFormat.fillFormat`):
    positional ("0.", every entry right-aligned to the widest) unless the
    largest nonzero |value| is >= 1e8 or more than 1000 times the smallest,
    then scientific (the shortest digits, at most 8 after the point, every
    mantissa padded to the longest fraction with zeros, the exponent at
    least two digits). One line: two labels at most here."""
    var max_a = 0.0
    var min_a = 0.0
    var nonzero = False
    for i in range(len(values)):  # small-loop(the labels)
        var a = abs(values[i])
        if a == 0.0:
            continue
        if not nonzero or a > max_a:
            max_a = a
        if not nonzero or a < min_a:
            min_a = a
        nonzero = True
    var sci = nonzero and (max_a >= 1.0e8 or max_a / min_a > 1000.0)
    var parts = List[String]()
    var width = 0
    if not sci:
        for i in range(len(values)):  # small-loop(the labels)
            var digits = _int_digit_list(values[i])
            var text = String("-") if values[i] < 0.0 else String("")
            for d in range(len(digits)):  # small-loop(the digits)
                text += String(digits[d])
            text += "."
            width = max(width, text.byte_length())
            parts.append(text^)
    else:
        var heads = List[String]()
        var fracs = List[List[Int]]()
        var exps = List[Int]()
        var precision = 0
        var exp_size = 2
        for i in range(len(values)):  # small-loop(the labels)
            var digits = _int_digit_list(values[i])
            var e = len(digits) - 1 if values[i] != 0.0 else 0
            if len(digits) > 9:
                # precision 8: nine significant digits, rounded
                var up = digits[9] >= 5
                var kept = List[Int]()
                for d in range(9):  # small-loop(the kept digits)
                    kept.append(digits[d])
                var at = 8
                while up and at >= 0:  # small-loop(the carry)
                    kept[at] += 1
                    up = kept[at] == 10
                    if up:
                        kept[at] = 0
                    at -= 1
                if up:
                    kept.insert(0, 1)
                    _ = kept.pop()
                    e += 1
                digits = kept^
            while len(digits) > 1 and digits[len(digits) - 1] == 0:  # small-loop(the trailing zeros)
                _ = digits.pop()
            var head = String("-") if values[i] < 0.0 else String("")
            head += String(digits[0])
            var frac = List[Int]()
            for d in range(1, len(digits)):  # small-loop(the fraction digits)
                frac.append(digits[d])
            precision = max(precision, len(frac))
            exp_size = max(exp_size, String(e).byte_length())
            width = max(width, head.byte_length())
            heads.append(head^)
            fracs.append(frac^)
            exps.append(e)
        for i in range(len(heads)):  # small-loop(the labels)
            var text = _pad_left(heads[i], width) + "."
            for d in range(precision):  # small-loop(the fraction digits, zero padded)
                text += String(fracs[i][d]) if d < len(fracs[i]) else String("0")
            var shown = String(exps[i])
            text += "e+"
            for _ in range(exp_size - shown.byte_length()):  # small-loop(the exponent's zero padding)
                text += "0"
            text += shown
            parts.append(text^)
        width = 0
    var out = String("[")
    for i in range(len(parts)):  # small-loop(the labels)
        if i > 0:
            out += " "
        out += _pad_left(parts[i], width)
    return out + "]"


def _check_listed(label_words: List[String], row: MselScorer, truth: Int, count: Int) raises:
    """Lane mf-score8: the door's strict coding of y_true among a `labels`
    list (`mx_codes(true, uniq, True)`): the first row whose label is not
    listed raises Python's KeyError of that label (its repr)."""
    var sel = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=row.sel_at)
    var t = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=truth)
    for i in range(count):  # driver-loop: over the side's rows (the first label outside the list)
        var c = Int(t[i])
        var listed = False
        for j in range(row.sel_n):  # small-loop(the caller's labels)
            listed = listed or Int(sel[j]) == c
        if not listed:
            raise Error("KeyError: " + (label_words[c] if c < len(label_words) else String(c)))


def _label_floats(label_words: List[String], row: MselScorer, codes: Int, count: Int,
                  mut dst: List[Float32]) raises:
    """Lane mf-score8: one side's class codes as their labels' float32
    values (the UNPACK's value words), as NumPy converts the labels; a
    string label `float()` does not convert raises NumPy's words."""
    var lv = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=row.lv_at)
    var c = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=codes)
    dst = List[Float32](length=max(count, 1), fill=0.0)
    for i in range(count):  # driver-loop: over the side's rows (their labels' values)
        var k = Int(c[i])
        if k < 0 or k >= row.lv_n:
            raise Error("ValueError: y codes outside [0, n_classes)")
        if Int(lv[2 * k + 1]) == SCORER_LABEL_UNREAD:
            raise Error("ValueError: could not convert string to float: "
                        + (label_words[k] if k < len(label_words) else String(k)))
        dst[i] = bitcast[DType.float32](lv[2 * k])


def _label_regression(route: MetricRoute, label_words: List[String], row: MselScorer, truth: Int, response: Int,
                      count: Int, classes: Int, w: Int, shown: String, mut warn: Int) raises -> Float64:
    """Lane mf-score8: a regression scorer on a classifier's side: the true
    and the predicted class codes as their labels' values (y_true's rows
    first, as `_check_reg_targets` converts y_true before y_pred), then the
    row's own regression program. median_absolute_error and max_error check
    their targets with dtype='numeric': string labels refuse in
    scikit-learn's words."""
    if row.lv_n != classes:
        raise Error("member ABI: a regression scorer on a classifier needs one value per class")
    if row.metric == REG_MEDAE or row.metric == REG_MAX_ERROR:
        var lv = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=row.lv_at)
        for c in range(row.lv_n):  # small-loop(the fit labels: their kinds)
            if Int(lv[2 * c + 1]) != SCORER_LABEL_NUMBER:
                raise Error(SCORER_NUMERIC_WORDS)
    var yt = List[Float32]()
    var yp = List[Float32]()
    _label_floats(label_words, row, truth, count, yt)
    _label_floats(label_words, row, response, count, yp)
    var value = reg_metric_options_run(route, row.metric, Int(yt.unsafe_ptr()), Int(yp.unsafe_ptr()), count, w, row.mo,
                                       row.ff, row.param, row.param_given, shown, warn)
    _ = len(yt)  # read by the program above
    _ = len(yp)
    return value


def _response_regression(route: MetricRoute, label_words: List[String], row: MselScorer, truth: Int, response: Int,
                         count: Int, classes: Int, cols: Int, w: Int, shown: String, mut warn: Int) raises -> Float64:
    """LANE mf-r9c (door :3670): a regression metric whose make_scorer asks a
    non-predict response the class has (r2_score over predict_proba or
    decision_function). scikit-learn 1.9: `_get_response_values` hands a
    BINARY classifier's positive column (`classes[-1]`, pos_label None: a
    regression metric takes none) or its 1-D decision, a multiclass one its
    (rows, classes) block, a non-classifier its raw answer (utils/
    _response.py lines 199-236); `_check_reg_targets` converts y_true first
    (a classifier's labels as their values, `_label_floats`; dtype='numeric'
    refuses strings for median_absolute_error / max_error), then refuses a
    block of more than one column ("y_true and y_pred have different number
    of output (1!=k)", metrics/_regression.py line 128); one column is
    scored by the row's own program. A fold estimator holding two classes
    hands its one column already (core/msel_search_run.mojo
    `_fold_view_side`)."""
    var yt = List[Float32]()
    if classes > 0:
        if row.lv_n != classes:
            raise Error("member ABI: a regression scorer on a classifier needs one value per class")
        if row.metric == REG_MEDAE or row.metric == REG_MAX_ERROR:
            var lv = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=row.lv_at)
            for c in range(row.lv_n):  # small-loop(the fit labels: their kinds)
                if Int(lv[2 * c + 1]) != SCORER_LABEL_NUMBER:
                    raise Error(SCORER_NUMERIC_WORDS)
        _label_floats(label_words, row, truth, count, yt)
    else:
        var t = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=truth)
        yt = List[Float32](length=max(count, 1), fill=0.0)
        for i in range(count):  # driver-loop: over the side's rows (its targets)
            yt[i] = t[i]
    var width = max(cols, 1)
    var column = List[Float32]()
    var pred = response
    if classes == 2 and width == 2 and row.op == MEMBER_OP_PREDICT_PROBA:
        var src = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=response)
        column = List[Float32](length=max(count, 1), fill=0.0)
        for i in range(count):  # driver-loop: over the side's rows (classes[-1]'s probability column)
            column[i] = src[2 * i + 1]
        pred = Int(column.unsafe_ptr())
        width = 1
    if width != 1:
        raise Error("ValueError: y_true and y_pred have different number of output (1!=" + String(width) + ")")
    var value = reg_metric_options_run(route, row.metric, Int(yt.unsafe_ptr()), pred, count, w, row.mo, row.ff,
                                       row.param, row.param_given, shown, warn)
    _ = len(yt)  # read by the program above
    _ = len(column)
    return value


def _multi_offset(row: MselScorer, j: Int) -> Int:
    """Lane mf-rows8b: output j's first label in the row's label value words
    (the outputs' class counts at `counts_at`, output order)."""
    var counts = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=row.counts_at)
    var at = 0
    for q in range(j):  # small-loop(the outputs before j)
        at += counts[q]
    return at


def _multi_floats(row: MselScorer, label_words: List[String], codes: Int, count: Int, convert: Bool,
                  mut dst: List[Float32], mut numeric: Bool) raises:
    """Lane mf-rows8b: one multi-output classifier side's codes (count rows x
    `row.outs`, int32, each output's own class codes) as their labels' float32
    values (the UNPACK's value words, the door's `_multi_label_values`);
    `numeric` False when a label of the side is a string (a text label that
    `float()` converts keeps its value, as NumPy's float conversion does).
    `convert` (a regression row): the side's first label, in row order,
    that `float()` does not convert raises NumPy's words (its repr is the
    label word its value word names)."""
    var outs = row.outs
    var counts = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=row.counts_at)
    var offsets = List[Int](length=outs, fill=0)
    for j in range(1, outs):  # small-loop(the outputs' first labels)
        offsets[j] = offsets[j - 1] + counts[j - 1]
    if offsets[outs - 1] + counts[outs - 1] != row.lv_n:
        raise Error("member ABI: a multi-output classifier's scorer needs one value per output label")
    var lv = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=row.lv_at)
    var c = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=codes)
    dst = List[Float32](length=max(count * outs, 1), fill=0.0)
    numeric = True
    for i in range(count * outs):  # driver-loop: over the side's rows x outputs (their labels' values)
        var j = i % outs
        var k = Int(c[i])
        if k < 0 or k >= counts[j]:
            raise Error("ValueError: y codes outside [0, n_classes)")
        var at = offsets[j] + k
        var kind = Int(lv[2 * at + 1])
        if kind != SCORER_LABEL_NUMBER:
            numeric = False
        if kind == SCORER_LABEL_UNREAD:
            if convert:
                var word = Int(lv[2 * at])
                raise Error("ValueError: could not convert string to float: "
                            + (label_words[word] if word >= 0 and word < len(label_words) else String(at)))
            continue
        dst[i] = bitcast[DType.float32](lv[2 * at])


def _multi_label_regression(route: MetricRoute, label_words: List[String], row: MselScorer, truth: Int,
                            response: Int, count: Int, w: Int, shown: String, mut per_output: List[Float64],
                            mut warn: Int) raises -> Float64:
    """Lane mf-rows8b (row :3779): a regression scorer on a multi-output
    classifier's side, as `_label_regression` per output: y_true's values
    first (`_check_reg_targets` converts y_true before y_pred), a label
    `float()` cannot convert in NumPy's words, median_absolute_error and
    max_error's dtype='numeric' refusal of string labels; then the row's
    rows x outputs program (the multi-output regressor's)."""
    var lv = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=row.lv_at)
    if row.metric == REG_MEDAE or row.metric == REG_MAX_ERROR:
        for c in range(row.lv_n):  # small-loop(the outputs' labels: their kinds)
            if Int(lv[2 * c + 1]) != SCORER_LABEL_NUMBER:
                raise Error(SCORER_NUMERIC_WORDS)
    var yt = List[Float32]()
    var yp = List[Float32]()
    var numeric = True
    _multi_floats(row, label_words, truth, count, True, yt, numeric)
    _multi_floats(row, label_words, response, count, True, yp, numeric)
    var value = reg_metric_options_run(route, row.metric, Int(yt.unsafe_ptr()), Int(yp.unsafe_ptr()), count,
                                       row.outs, w, row.mo, row.ff, row.param, row.param_given, shown, per_output,
                                       warn)
    _ = len(yt)  # read by the program above
    _ = len(yp)
    return value


def _multi_type(values: List[Float32], numeric: Bool) -> String:
    """Lane mf-rows8b: scikit-learn's `type_of_target` of a side's rows x
    outputs block (sklearn 1.9 utils/multiclass.py: `is_multilabel` first,
    fewer than three distinct values, integral; then a non-integral float
    is 'continuous'; any other 2-D block 'multiclass', with the
    '-multioutput' suffix of more than one column). A string label block is
    never multilabel (its dtype is neither integer nor float)."""
    if not numeric:
        return "multiclass-multioutput"
    var first = Float32(0.0)
    var second = Float32(0.0)
    var distinct = 0
    for i in range(len(values)):  # driver-loop: over the block's values (integral, distinct up to three)
        var v = values[i]
        if not (v >= -9.0e18 and v <= 9.0e18):
            return "continuous-multioutput"  # NaN, an infinity or beyond an integer's range
        if Float32(Float64(Int(v))) != v:
            return "continuous-multioutput"
        if distinct == 0:
            first = v
            distinct = 1
        elif distinct == 1 and v != first:
            second = v
            distinct = 2
        elif distinct == 2 and v != first and v != second:
            distinct = 3
    if distinct < 3:
        return "multilabel-indicator"
    return "multiclass-multioutput"


def msel_multi_output_type(values: Int, count: Int) -> String:
    """Lane mf-rows8b: `type_of_target` of a regressor's float32 rows x
    outputs block at `values` (`count` words; more than one output), the
    words a stratified splitter's refusal names (core/ensemble_member.mojo
    `_fit_search`)."""
    var p = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=values)
    var block = List[Float32](length=max(count, 1), fill=0.0)
    for i in range(count):  # driver-loop: over the block's words (one copy)
        block[i] = p[i]
    return _multi_type(block, True)


def _multi_output_refusal(row: MselScorer, truth: Int, response: Int, count: Int) raises:
    """LANE mf-rows8b (rows :3777 / :3779): a label or clustering scorer on
    one side of a multi-output target (`row.outs` columns). scikit-learn
    (sklearn 1.9):
      * the clustering scorers: `check_clusterings` refuses a 2-D
        labels_true (metrics/cluster/_supervised.py line 72);
      * the label scorers: `_check_targets` (metrics/_classification.py
        lines 116-145) reads both sides' `type_of_target`: different types
        refuse as a mix, 'multiclass-multioutput' and
        'continuous-multioutput' are not supported; two
        'multilabel-indicator' sides go on: confusion_matrix (balanced
        accuracy, cohen_kappa) and matthews_corrcoef refuse them (lines 555,
        1319), class_likelihood_ratios needs binary targets (line 2366);
        every other label metric scores multilabel targets, which Mojolearn
        has NOT IMPLEMENTED (metrics/NOT_IMPLEMENTED.tsv): the label doors'
        own refusal (bindings/mcls_common_unpack.mojo `mc_refuse_multilabel`).
    The side's values: a regressor's float32 targets and predictions (its
    integer ids when `row.ids`), or a multi-output classifier's codes as
    their labels' values (`_multi_floats`)."""
    var outs = row.outs
    if row.family == _F_CLUSTER:
        raise Error("ValueError: labels_true must be 1D: shape is (" + String(count) + ", " + String(outs) + ")")
    var yt = List[Float32]()
    var yp = List[Float32]()
    var true_numeric = True
    var pred_numeric = True
    if row.counts_at != 0:
        _multi_floats(row, List[String](), truth, count, False, yt, true_numeric)
        _multi_floats(row, List[String](), response, count, False, yp, pred_numeric)
    else:
        var t = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=truth)
        yt = List[Float32](length=max(count * outs, 1), fill=0.0)
        yp = List[Float32](length=max(count * outs, 1), fill=0.0)
        for i in range(count * outs):  # driver-loop: over the side's rows x outputs (its targets)
            yt[i] = t[i]
        if row.ids:
            var ip = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=response)
            for i in range(count * outs):  # driver-loop: over the side's rows x outputs (the ids as numbers)
                yp[i] = Float32(Int(ip[i]))
        else:
            var fp = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=response)
            for i in range(count * outs):  # driver-loop: over the side's rows x outputs (its predictions)
                yp[i] = fp[i]
    var type_true = _multi_type(yt, true_numeric)
    var type_pred = _multi_type(yp, pred_numeric)
    if type_true != type_pred:
        raise Error("ValueError: Classification metrics can't handle a mix of " + type_true + " and " + type_pred
                    + " targets")
    if type_true != "multilabel-indicator":
        raise Error("ValueError: " + type_true + " is not supported")
    if row.family == _F_KAPPA or (row.family == _F_LABEL and (row.metric == LBL_MCC or row.metric == LBL_BALANCED)):
        raise Error("ValueError: multilabel-indicator is not supported")
    if row.family == _F_CLR:
        raise Error("ValueError: class_likelihood_ratios only supports binary classification problems, got targets"
                    + " of type: multilabel-indicator")
    raise Error("NotImplementedError: mojolearn " + msel_scorer_caller(row) + ": y_true is 2-D; multilabel-indicator "
                + "targets are NOT IMPLEMENTED (metrics/NOT_IMPLEMENTED.tsv), only binary and multiclass 1-D labels")


#: Lane mf-r9c: `type_of_target` of a multi-output fold estimator's `classes_`
#: as `_get_response_values` acts on it: any type it hands on unchanged (the
#: per-output probability LIST reaches the metric), 'binary' (it fails on the
#: list), 'multilabel-indicator' (the positive columns, rows x outputs).
comptime _MC_LIST = 0
comptime _MC_BINARY = 1
comptime _MC_MULTILABEL = 2


def _multi_classes(row: MselScorer, label_words: List[String], mut widths: List[Int]) raises -> Int:
    """LANE mf-r9c: the fold estimator's `classes_` on a multi-output
    classifier's side (a list of one array per output: KNeighborsClassifier
    and MultiOutputClassifier hold each output's `np.unique` of its train
    labels; here the fold's held classes, `row.mheld_at`, else every class
    of the output) and `type_of_target` of that list (sklearn 1.9
    utils/multiclass.py `is_multilabel`, `type_of_target`):
      * outputs of different class counts are a ragged list, an object array
        of arrays: 'unknown' (_MC_LIST);
      * equal counts make a 2-D (outputs, count) array: numbers with at most
        two distinct values over two or more columns, all integral, are
        'multilabel-indicator' (_MC_MULTILABEL); a non-integral number is
        'continuous...' (_MC_LIST); one column with at most two distinct
        values (numbers or strings) is 'binary' (_MC_BINARY); any other is
        'multiclass...' (_MC_LIST);
      * numbers and strings together (an object array) take the list reading
        (_MC_LIST; scikit-learn's own fit refuses a mixed-kind column).
    `widths` gets each output's class count (its probability block's
    columns). Strings are told apart by their value word (a `float()`-able
    string) or their repr word. A few dozen words per side."""
    var outs = row.outs
    var counts = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=row.counts_at)
    var lv = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=row.lv_at)
    widths = List[Int](length=outs, fill=0)
    var numbers = False
    var texts = False
    var integral = True
    var distinct = 0
    var first = Float32(0.0)
    var second = Float32(0.0)
    var seen_text = List[String]()
    var offset = 0
    for j in range(outs):  # small-loop(the outputs: their fold classes)
        var held_n = 0
        var held = 0
        if row.mheld_at != 0:
            var lists = MutPointer[List[Int32], MutUntrackedOrigin](unsafe_from_address=row.mheld_at)
            held_n = len(lists[row.fold * outs + j])
            held = Int(lists[row.fold * outs + j].unsafe_ptr())
        var w = held_n if held_n > 0 else counts[j]
        widths[j] = w
        var h = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=held)
        for q in range(w):  # small-loop(output j's fold classes)
            var c = Int(h[q]) if held_n > 0 else q
            var at = offset + c
            if at < 0 or 2 * at + 1 >= 2 * row.lv_n:
                raise Error("member ABI: a multi-output classifier's scorer needs one value per output label")
            var kind = Int(lv[2 * at + 1])
            if kind == SCORER_LABEL_NUMBER:
                numbers = True
                var v = bitcast[DType.float32](lv[2 * at])
                if not (v >= -9.0e18 and v <= 9.0e18) or Float32(Float64(Int(v))) != v:
                    integral = False
                if distinct == 0:
                    first = v
                    distinct = 1
                elif distinct == 1 and v != first:
                    second = v
                    distinct = 2
                elif distinct == 2 and v != first and v != second:
                    distinct = 3
            else:
                texts = True
                var key = String("v") + String(Int(lv[2 * at]))
                if kind == SCORER_LABEL_UNREAD:
                    var word = Int(lv[2 * at])
                    key = String("r") + (label_words[word] if word >= 0 and word < len(label_words) else String(at))
                var known = False
                for s in range(len(seen_text)):  # small-loop(the distinct strings so far, at most three)
                    known = known or seen_text[s] == key
                if not known and len(seen_text) < 3:
                    seen_text.append(key)
        offset += counts[j]
    for j in range(1, outs):  # small-loop(the outputs' class counts)
        if widths[j] != widths[0]:
            return _MC_LIST
    var w0 = widths[0]
    if numbers and texts:
        return _MC_LIST
    if numbers:
        if not integral:
            return _MC_LIST
        if w0 >= 2 and distinct < 3:
            return _MC_MULTILABEL
        if w0 == 1 and distinct <= 2:
            return _MC_BINARY
        return _MC_LIST
    if w0 == 1 and len(seen_text) <= 2:
        return _MC_BINARY
    return _MC_LIST


def _multi_proba_value(route: MetricRoute, row: MselScorer, label_words: List[String], truth: Int, response: Int,
                       count: Int, cols: Int, w: Int, mclasses: Int, widths: List[Int],
                       mut notices: List[Float64]) raises -> Float64:
    """LANE mf-r9c (door :3761): a probability scorer on one side of a
    MULTI-OUTPUT classifier (KNeighborsClassifier, MultiOutputClassifier:
    `predict_proba` is a list of one (rows, classes) block per output; no
    native multi-output kind answers decision_function). After
    `_get_response_values` (`_multi_classes`), the metric as scikit-learn
    1.9 runs it on y_true (rows x outputs) and what it was handed:
      * the LIST (_MC_LIST): `check_array` of it is a 3-D array (equal
        blocks: utils/validation.py line 1083) or NumPy's inhomogeneous-shape
        refusal (ragged blocks). roc_auc_score and log_loss / brier_score_loss
        check it first; average_precision_score first reads y_true's type
        (`pos_label` fixed to 1, then `_average_binary_score`'s format
        refusal, metrics/_base.py line 73, and its length check, where the
        list's length is its outputs); the D^2 scores first check the lengths
        (metrics/_classification.py lines 3947, 4053);
      * the positive columns (_MC_MULTILABEL, rows x outputs): roc_auc_score
        and average_precision_score score a 'multilabel-indicator' y_true and
        refuse any other type in `_average_binary_score`'s words; log_loss,
        brier_score_loss and the D^2 scores binarize y_true
        (`_one_hot_encoding_multiclass_target`): LabelBinarizer refuses a
        multi-output type; with `labels`, the fit of the labels, their
        `_isin` check (a 2-D y_true's `set` is unhashable), the two-label
        check and the transform's multilabel refusal, in that order. A
        served multilabel-indicator target is Mojolearn's NOT IMPLEMENTED
        (metrics/NOT_IMPLEMENTED.tsv), the doors' words;
      * top_k_accuracy_score: `column_or_1d(y_true)` first.
    The D^2 scores of fewer than two rows over the columns: nan and their
    warning. The side's labels are read through the row's value words
    (`_multi_floats`), never its probabilities. A REGRESSION row over the
    make_scorer's predict_proba (`_response_regression`'s case on several
    outputs): y_true converted first (`_check_reg_targets`), then the list's
    `check_array` refusal, or over the positive columns the row's rows x
    outputs program (the fold estimator's blocks of `widths` columns, each
    output's last column)."""
    var m = row.outs
    if row.family == _F_REG:
        if mclasses == _MC_LIST and count != m:
            # `_check_reg_targets`' first check: the list's length is its outputs
            raise Error("ValueError: Found input variables with inconsistent numbers of samples: [" + String(count)
                        + ", " + String(m) + ((", " + String(count)) if row.w_n > 0 else String("")) + "]")
        if row.metric == REG_MEDAE or row.metric == REG_MAX_ERROR:
            var lvk = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=row.lv_at)
            for c in range(row.lv_n):  # small-loop(the outputs' labels: their kinds)
                if Int(lvk[2 * c + 1]) != SCORER_LABEL_NUMBER:
                    raise Error(SCORER_NUMERIC_WORDS)
        var yv = List[Float32]()
        var numeric_v = True
        _multi_floats(row, label_words, truth, count, True, yv, numeric_v)
        if mclasses == _MC_LIST:
            var same = True
            for j in range(1, len(widths)):  # small-loop(the outputs' block widths)
                same = same and widths[j] == widths[0]
            if same:
                raise Error("ValueError: Found array with dim 3, while dim <= 2 is required.")
            raise Error("ValueError: setting an array element with a sequence. The requested array has an "
                        + "inhomogeneous shape after 2 dimensions. The detected shape was (" + String(m) + ", "
                        + String(count) + ") + inhomogeneous part.")
        var total = 0
        for j in range(len(widths)):  # small-loop(the outputs' block widths)
            total += widths[j]
        if total != cols:
            raise Error("member ABI: a multi-output fold member answered " + String(cols) + " probability columns for "
                        + String(total) + " fold classes")
        var src = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=response)
        var yp = List[Float32](length=max(count * m, 1), fill=0.0)
        for i in range(count):  # driver-loop: over the side's rows (each output's positive column)
            var at = 0
            for j in range(m):  # small-loop(the outputs' blocks)
                at += widths[j]
                yp[i * m + j] = src[i * cols + at - 1]
        var shown = String("")
        if row.param_word >= 0 and row.param_word < len(label_words):
            shown = label_words[row.param_word].copy()
        var per_output = List[Float64]()
        var warn = 0
        var value = reg_metric_options_run(route, row.metric, Int(yv.unsafe_ptr()), Int(yp.unsafe_ptr()), count, m, w,
                                           row.mo, row.ff, row.param, row.param_given, shown, per_output, warn)
        _ = len(yv)  # read by the program above
        _ = len(yp)
        if len(per_output) > 1:
            _number_error(row, label_words, per_output)
        if warn != 0:
            member_notice_add(notices, MEMBER_NOTICE_METRIC_FEW, 1.0 if reg_metric_d2(row.metric) else 0.0)
        return value
    if row.family == _F_TOP_K:
        raise Error("ValueError: y should be a 1d array, got an array of shape (" + String(count) + ", " + String(m)
                    + ") instead.")
    var yt = List[Float32]()
    var numeric = True
    _multi_floats(row, List[String](), truth, count, False, yt, numeric)
    var target = _multi_type(yt, numeric)
    var multilabel = target == "multilabel-indicator"
    var equal = True
    for j in range(1, len(widths)):  # small-loop(the outputs' block widths)
        equal = equal and widths[j] == widths[0]
    var array_words = String("ValueError: Found array with dim 3, while dim <= 2 is required.")
    if not equal:
        array_words = ("ValueError: setting an array element with a sequence. The requested array has an "
                       + "inhomogeneous shape after 2 dimensions. The detected shape was (" + String(m) + ", "
                       + String(count) + ") + inhomogeneous part.")
    var weighted = (", " + String(count)) if row.w_n > 0 else String("")
    var not_implemented = ("NotImplementedError: mojolearn " + msel_scorer_caller(row) + ": y_true is 2-D; "
                           + "multilabel-indicator targets are NOT IMPLEMENTED (metrics/NOT_IMPLEMENTED.tsv), only "
                           + "binary and multiclass 1-D labels")
    var format_words = "ValueError: " + target + " format is not supported"
    if row.family == _F_ROC:
        if mclasses == _MC_LIST:
            raise Error(array_words)
        if multilabel:
            raise Error(not_implemented)
        raise Error(format_words)
    if row.family == _F_AP:
        if multilabel and row.wmode == 1:
            raise Error("ValueError: Parameter pos_label is fixed to 1 for multilabel-indicator y_true. Do not set "
                        + "pos_label or set pos_label to 1.")
        if not multilabel:
            raise Error(format_words)
        if mclasses == _MC_MULTILABEL:
            raise Error(not_implemented)
        if count != m:
            raise Error("ValueError: Found input variables with inconsistent numbers of samples: [" + String(count)
                        + ", " + String(m) + weighted + "]")
        raise Error(array_words)
    if row.family == _F_LOG_LOSS or row.family == _F_BRIER or row.family == _F_D2P:
        if mclasses == _MC_LIST:
            if row.family == _F_D2P and m != count:
                raise Error("ValueError: Found input variables with inconsistent numbers of samples: [" + String(m)
                            + ", " + String(count) + weighted + "]")
            raise Error(array_words)
        if row.family == _F_D2P and count < 2:
            member_notice_add(notices, MEMBER_NOTICE_METRIC_FEW, 1.0)
            return _nan64()
        if row.sel_n > 0:
            # a `labels` kwarg (the UNPACK's words, `_multi_labels_flags`):
            # one in-labels flag per output label (value-word order), then
            # the fit's empty-labels word and the one-label word (-1: none)
            if row.sel_n != row.lv_n + 2:
                raise Error("member ABI: a multi-output probability scorer's labels flags are cut short")
            var flags = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=row.sel_at)
            var empty_word = Int(flags[row.lv_n])
            var one_word = Int(flags[row.lv_n + 1])
            if empty_word >= 0:
                if empty_word >= len(label_words):
                    raise Error("member ABI: a search scorer's labels refusal word is missing")
                raise Error(label_words[empty_word])
            var counts = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=row.counts_at)
            var offsets = List[Int](length=m, fill=0)
            for j in range(1, m):  # small-loop(the outputs' first labels)
                offsets[j] = offsets[j - 1] + counts[j - 1]
            var t = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=truth)
            for i in range(count * m):  # driver-loop: over the side's rows x outputs (is each label listed)
                var at = offsets[i % m] + Int(t[i])
                if at < 0 or at >= row.lv_n:
                    raise Error("ValueError: y codes outside [0, n_classes)")
                if flags[at] == Int32(0):
                    raise Error("TypeError: unhashable type: 'numpy.ndarray'")
            if one_word >= 0:
                if one_word >= len(label_words):
                    raise Error("member ABI: a search scorer's labels refusal word is missing")
                raise Error(label_words[one_word])
            if multilabel:
                raise Error("ValueError: The object was not fitted with multilabel input.")
            raise Error("ValueError: Multioutput target data is not supported with label binarization")
        if multilabel:
            raise Error(not_implemented)
        raise Error("ValueError: Multioutput target data is not supported with label binarization")
    raise Error("RuntimeError: a multi-output classifier's probability scorer of no typed family")


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
    if classes == 0 and row.fpred != 0:
        # LANE mf-rows8b (row :3899): a regressor's float predictions under a
        # clustering scorer. The public cluster doors take integer labels
        # only (bindings/mcls_cluster_door.mojo `_pair_i32` / the MI doors,
        # bindings/mcls_common_unpack.mojo `mc_i32_1d`): labels_pred of the
        # member's float buffer refuses in their words, after labels_true's
        # check above (their order), on every side.
        raise Error("mojolearn metrics: labels_pred must be an integer label array, got buffer format "
                    + (String("'d'") if row.fpred == 2 else String("'f'"))
                    + "; the ported kernels take int32 labels")
    var value: Float64
    # lane mf-score8: v_measure's beta (the param word), NMI / AMI's
    # average_method (the flag word: 0 min, 1 geometric, 2 arithmetic, 3 max)
    var beta = row.param if row.param_given else 1.0
    var method = row.flag if row.flag >= 0 else 2
    if row.metric >= _CL_PAIR:
        value = cluster_pair_run(cluster, row.metric - _CL_PAIR, t, response, count, beta)
    elif classes == 0:
        # NMI / AMI read codes in 0 .. k-1: the union codes of the raw labels
        # (order-preserving, so each clustering's own classes are the door's).
        var codes = List[Int32]()
        var k = cluster_union_codes(t, response, count, codes)
        var base = Int(codes.unsafe_ptr())
        # `_average_method('arithmetic')` (the scorers pass no average_method)
        value = cluster_mi_run(route, row.metric == _CL_AMI, base, base + 4 * count, count, k, method)
        _ = len(codes)
    else:
        # `_average_method('arithmetic')` (the scorers pass no average_method)
        value = cluster_mi_run(route, row.metric == _CL_AMI, t, response, count, classes, method)
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
                  count: Int, classes: Int, positive: Int, w: Int, mut warn: Int) raises -> Float64:
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
    if row.pos_float:
        # a float pos_label (lane mf-score6): the door's `mc_positive` refusal
        raise Error(SCORER_FLOAT_POS_WORDS)
    if positive == SCORER_POSITIVE_KIND:
        raise Error("pos_label must use the same type as the targets")
    var absent = positive == SCORER_POSITIVE_ABSENT
    if (absent or (positive >= 0 and positive < classes and not seen[positive])) and present >= 2:
        raise Error("pos_label=" + msel_scorer_pos_words(row, label_words, positive)
                    + " is not a valid label. It should be one of " + _present_words(label_words, seen))
    if absent:
        return label_binary_options_run(route, row.metric, truth, response, count, classes + 1, classes, row.zd,
                                        row.fmode, row.b2, w, warn)
    return label_binary_options_run(route, row.metric, truth, response, count, classes, positive, row.zd, row.fmode,
                                    row.b2, w, warn)


def msel_scorer_caller(row: MselScorer) -> String:
    """Lane mf-score6: the `caller` whose words the metric door's weight and
    average refusals carry (bindings/mcls_classify_door.mojo `mc_pair`,
    bindings/metrics_exp_cls_door.mojo `mx_pair`, the curve, rank and
    ranking doors); the UNPACK spells the weights' refusals with it too."""
    if row.family == _F_REG:
        return reg_metric_caller(row.metric)
    if row.family == _F_ACCURACY:
        return String("accuracy_score")
    if row.family == _F_LOG_LOSS:
        return String("log_loss")
    if row.family == _F_AP:
        return String("average_precision_score")
    if row.family == _F_BRIER:
        return String("brier_score_loss")
    if row.family == _F_TOP_K:
        return String("top_k_accuracy_score")
    if row.family == _F_ROC:
        return String("roc_auc_score")
    if row.family == _F_CLUSTER:
        return String(row.name)
    if row.family == _F_COUNT:
        if row.metric == _CNT_CORRECT:
            return String("accuracy_score")
        return String("hamming_loss") if row.metric == _CNT_HAMMING else String("zero_one_loss")
    if row.family == _F_KAPPA:
        return String("cohen_kappa_score")
    if row.family == _F_HINGE:
        return String("hinge_loss")
    if row.family == _F_D2P:
        return String("d2_log_loss_score") if row.metric == 1 else String("d2_brier_score")
    if row.family == _F_CLR:
        return String("class_likelihood_ratios")
    if row.metric == LBL_JACCARD:
        return String("jaccard_score")
    if row.metric == LBL_BALANCED:
        return String("balanced_accuracy_score")
    if row.metric == LBL_MCC:
        return String("matthews_corrcoef")
    return String("precision_recall_fscore_support")


def _check_weights(row: MselScorer, count: Int) raises:
    """Lane mf-score6: a make_scorer's sample_weight on one side, checked as
    the door's `_sample_weight_f32` checks it (bindings/metrics_reg_door.mojo
    `_reg_weights`, in its order and words): as many entries as the side's
    samples (scikit-learn passes the same kwarg to every side, so its
    `check_consistent_length` refuses any other length too), finite and
    nonnegative, a positive largest weight."""
    var caller = msel_scorer_caller(row)
    if row.w_n != count:
        raise Error("mojolearn " + caller + ": sample_weight has " + String(row.w_n) + " entries for " + String(count)
                    + " samples")
    var w = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=row.w_at)
    var most = Float32(0.0)
    for i in range(count):  # driver-loop: over the side's weights (the door's value checks)
        var v = w[i]
        if not (v >= 0.0 and v - v == 0.0):
            raise Error("mojolearn " + caller + ": sample_weight must have finite nonnegative entries")
        most = max(most, v)
    if not most > 0.0:
        raise Error("mojolearn " + caller + ": sample_weight must have positive total weight")


def _fraction_digits(v: Float64) -> String:
    """`v` (finite, >= 0) as NumPy's positional 'maxprec' form with
    precision 8 prints it, without padding: the integer part, '.', then at
    most 8 fraction digits with the trailing zeros dropped."""
    var whole = Int(v)
    var scaled = Int((v - Float64(whole)) * 1e8 + 0.5)
    if scaled >= 100000000:
        whole += 1
        scaled -= 100000000
    var digits = String(scaled)
    while digits.byte_length() < 8:  # small-loop(left zero padding to 8 digits)
        digits = "0" + digits
    var end = digits.byte_length()
    while end > 0 and String(digits[byte=end - 1 : end]) == "0":  # small-loop(the trailing zeros)
        end -= 1
    return String(whole) + "." + String(digits[byte=0:end])


def _ndarray_words(values: List[Float64], sign: Float64) -> String:
    """`str(sign * values)` of a 1-D float64 array as NumPy prints it (the
    positional form: each value's integer part right-aligned and its
    fraction left-aligned to the widest, 'nan' right-aligned, lines wrapped
    at 75 columns with a one-space indent). NumPy switches to its
    exponential form for values >= 1e8, or a smallest nonzero below 1e-4,
    or a spread above 1000; scorer values of label metrics lie in [0, 1]
    and those forms are printed positionally here (an approximation)."""
    var parts = List[String]()
    var lead = 0
    var frac = 0
    for i in range(len(values)):  # small-loop(the labels' values)
        var v = sign * values[i]
        if v != v:
            parts.append(String("nan"))
            continue
        var neg = v < 0.0
        var text = _fraction_digits(-v if neg else v)
        if neg:
            text = "-" + text
        var dot = text.find(".")
        lead = max(lead, dot)
        frac = max(frac, text.byte_length() - dot - 1)
        parts.append(text)
    var cells = List[String]()
    for i in range(len(parts)):  # small-loop(the labels' values: padded)
        var text = parts[i].copy()
        var dot = text.find(".")
        if dot < 0:
            var width = lead + 1 + frac
            while text.byte_length() < width:  # small-loop(left padding)
                text = " " + text
        else:
            while dot < lead:  # small-loop(left padding)
                text = " " + text
                dot += 1
            while text.byte_length() - dot - 1 < frac:  # small-loop(right padding)
                text += " "
        cells.append(text)
    var out = String("")
    var line = String("[")
    for i in range(len(cells)):  # small-loop(the labels' values: NumPy's line wrapping)
        var word = cells[i].copy()
        var limit = 75
        if i + 1 < len(cells):
            word += " "
        else:
            limit = 74
        if line.byte_length() + word.byte_length() > limit and line.byte_length() > 1:
            out += String(line.rstrip()) + "\n"
            line = String(" ")
        line += word
    return out + line + "]"
