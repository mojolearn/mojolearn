# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE METRICS LANE'S DOOR.

Owned by the `metrics` expansion lane. `mojolearn/__init__.py` imports this
module and makes every name in `__all__` public as `mojolearn.<name>`. The
lane's functions belong to `mojolearn.metrics` (and, for the splitters,
`mojolearn.model_selection`), the way scikit-learn places them, so `__all__`
here is EMPTY on purpose and `_metrics_impl.py` re-exports the functions
below under `mojolearn.metrics`.

HOW THE FUNCTIONS COMPUTE. Every O(n) quantity comes out of ONE binding
entry, `x_metrics_run`, which runs a PROGRAM of units (x_metrics/common.mojo)
over one float32 arena: the GPU binding (`_mojolearn_x_metrics`) launches a
thread per unit, the host binding (`_mojolearn_x_metrics_host`, what
`_backend.binding` returns on a CPU-only install) runs the same units in a
loop. The native owners lay out the arena, list the stages and read the
per-class / per-column Float32 results back; this module only connects them.

THE EPILOGUE (DEVIATION 6106, narrowed 2026-09-28). What is left after the
O(n) folds is O(classes) arithmetic on those Float32 results: ratios,
averages, a square root. It runs here, in IEEE binary64 with ONLY + - * / and
sqrt, each correctly rounded on every host Python runs on, in a fixed
left-to-right order (and `_portable_math` for the rare logarithm), and only on
O(classes), O(outputs) or O(candidates x folds) scalars; `**`, `pow` and libm
transcendentals are not allowed. Any epilogue whose term count scales with
rows, curve points, contingency cells or k^2 * d runs in
x_metrics/epilogue.mojo under the same operation rules, the only route (lane
pyglue-sweep, 2026-10-03: the Python fallbacks and the
MOJOLEARN_METRICS_EPILOGUE=python reference arm are gone). That is bitwise the same on every machine by the IEEE standard, and it
is scikit-learn's own precision for the same step.
"""
from . import _backend

__all__ = []

_BINDING = "_mojolearn_x_metrics"

#: op name -> id; x_metrics/units.mojo `run_unit` holds the same table.
_OPS = dict(group_sort=0, group_sum=1, pair_key=2, reg_term=3, col_sort=4, wpercentile=5, col_max=6, bin_curve=7, row_metric=8, row_centroid_dist=9, permute=10,
            fold_rows=36, rows64=41, strat_codes=45, curve_fold=46, onehot=52, rep_rows=53, pair_cols=54,
            # lane fam2-prep-metrics (x_metrics/cls_epi.mojo), both numeric modes
            # since lane cpu2-l7-metrics
            cls_epi=55,
            # lane cpu2-l7-metrics: the metric tails and scans on the device
            # (x_metrics/tail.mojo, cm_epi.mojo, reg_epi.mojo, rank_epi.mojo)
            off_diff=56, flag_scan=57, proba_rows=58, cm_epi=59, reg_epi=60, rank_epi=61, cl_epi=62,
            # lane cpu4-python: the curve arrays on the device (x_metrics/curve_out.mojo)
            curve_out=63, auc_xy=67,
            # lane cpu4-python: the contingency statistics, MI and expected MI
            # on the device (x_metrics/contingency.mojo)
            cont_stats=70,
            # lane py-runtime-b: StratifiedShuffleSplit's per-class draw counts
            # (x_metrics/split.mojo approx_mode_unit)
            approx_mode=81,
            # lane ne2-b-search-workers: permutation_test_score's within-group
            # permutation as one program (x_metrics/split.mojo group_take_unit, take_rows64_unit)
            group_take=85, take_rows64=86,
            # lane ne2-b-search-workers: one StratifiedShuffleSplit split as one
            # program (x_metrics/split.mojo permute_drawn_unit, strat_take_unit)
            permute_drawn=87, strat_take=88)


def _binding(numeric_mode):
    if numeric_mode is not None and (
        not isinstance(numeric_mode, str)
        or numeric_mode.strip().lower() not in ("fast", "identical")
    ):
        raise ValueError("numeric_mode must be 'fast' or 'identical'")
    mode = (numeric_mode or _backend.default_mode()).strip().lower()
    return _backend.binding(_BINDING, mode)


# TOMBSTONE: P1_DEAD_PY_CURVE_DRIVERS (removed: the unreachable Python curve/metric drivers _FoldCurve, _curves, _fold_curves, _curve_out, _ovr, roc_auc_options, ... and the other unreferenced work functions of _expansion_metrics, _expansion_decomp, _expansion_prep, preprocessing, linear_model, _linalg_impl, _expansion_neighbors, _labels; their native owners are bindings/metrics_exp_curve.mojo and bindings/mcls_ranking.mojo) deleted 2026-11-09 by lane p1-python-callbacks-worst; code recoverable at e68654a757.
# Restore: experiments/removed/P1_DEAD_PY_CURVE_DRIVERS.patch.


class UndefinedMetricWarning(UserWarning):
    """scikit-learn's `UndefinedMetricWarning` (a UserWarning): a metric that
    is ill-defined for the input and was set to a fallback value. Defined
    here so the metrics do not import scikit-learn."""


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

#: cm_epi kinds (x_metrics/cm_epi.mojo, op 59; lane cpu2-l7-metrics)
_CM_MLCM, _CM_MCC, _CM_COUNT, _CM_NORM, _CM_KAPPA, _CM_CLR, _CM_TOTAL, _CM_WSUM_LEAF, _CM_WSUM_FOLD = range(9)
#: rows per `_wsum_leaf` unit (x_metrics/cm_epi.mojo WSUM_BLOCK)
_WSUM_BLOCK = 256
#: the NORM modes: the 'all' total, then 'all', 'true', 'pred', none
_NORM_MODES = {"all": 1, "true": 2, "pred": 3, None: 4}
#: the kappa weight modes
_KAPPA_W = {None: 0, "linear": 1, "quadratic": 2}


# Engineering notes (moved from the docstring):
# Per-label one-vs-rest 2x2 confusion matrices, `[[tn, fp], [fn, tp]]`.
#
# scikit-learn 1.9 `multilabel_confusion_matrix` for binary and multiclass
# 1-D targets (strings or integers). Unweighted counts are exact Int64;
# with sample_weight they are Float64 images of Float32 per-label PairSums
# (DEVIATION 6100). Multilabel-indicator input and samplewise=True are
# refused by name (metrics/NOT_IMPLEMENTED.tsv).
def multilabel_confusion_matrix(y_true, y_pred, *, sample_weight=None, labels=None,
                                samplewise=False, numeric_mode=None):
    """Per-label one-vs-rest 2x2 confusion matrices, ``[[tn, fp], [fn, tp]]``.

    Follows scikit-learn's `multilabel_confusion_matrix` for binary and multiclass 1-D
    targets (strings or integers). Unweighted counts are exact integers; with
    `sample_weight` they are float64. Multilabel-indicator input and ``samplewise=True``
    raise an error.
    """
    return _NATIVE(None, _OP.family_mexp_cls_mlcm,
                   (y_true, y_pred, sample_weight, labels, samplewise, numeric_mode),
                   _NATIVE_CONTEXT)


# Engineering notes (moved from the docstring):
# scikit-learn 1.9 `precision_recall_fscore_support` for binary and
# multiclass 1-D targets: per-label sums on the device (exact integers, or
# Float32 PairSums of sample_weight), the ratios and averages in the
# binary64 epilogue (DEVIATION 6106). zero_division is 'warn', 0.0, 1.0 or
# np.nan; average='samples' (multilabel) is refused by name.
def precision_recall_fscore_support(y_true, y_pred, *, beta=1.0, labels=None, pos_label=1,
                                    average=None, warn_for=("precision", "recall", "f-score"),
                                    sample_weight=None, zero_division="warn", numeric_mode=None):
    """Precision, recall, F-score and support for each label.

    Follows scikit-learn's `precision_recall_fscore_support` for binary and multiclass
    1-D targets. `zero_division` is ``'warn'``, 0.0, 1.0 or ``np.nan``.
    ``average='samples'`` (multilabel) raises an error.
    """
    return _NATIVE(None, _OP.family_mexp_cls_prfs,
                   (y_true, y_pred, beta, labels, pos_label, average, warn_for, sample_weight, zero_division, numeric_mode),
                   _NATIVE_CONTEXT)


def fbeta_score(y_true, y_pred, *, beta, labels=None, pos_label=1, average="binary",
                sample_weight=None, zero_division="warn", numeric_mode=None):
    """scikit-learn 1.9 `fbeta_score` (binary / multiclass 1-D targets); see
    precision_recall_fscore_support."""
    return _NATIVE(None, _OP.family_mexp_cls_fbeta,
                   (y_true, y_pred, beta, labels, pos_label, average, sample_weight, zero_division, numeric_mode),
                   _NATIVE_CONTEXT)


def jaccard_score(y_true, y_pred, *, labels=None, pos_label=1, average="binary",
                  sample_weight=None, zero_division="warn", numeric_mode=None):
    """scikit-learn 1.9 `jaccard_score`, `tp / (tp + fp + fn)` per label,
    for binary and multiclass 1-D targets; see precision_recall_fscore_support
    for the sums and the epilogue."""
    return _NATIVE(None, _OP.family_mexp_cls_jaccard,
                   (y_true, y_pred, labels, pos_label, average, sample_weight, zero_division, numeric_mode),
                   _NATIVE_CONTEXT)


def balanced_accuracy_score(y_true, y_pred, *, sample_weight=None, adjusted=False, numeric_mode=None):
    """scikit-learn 1.9 `balanced_accuracy_score`: the mean per-class recall
    over the classes present in y_true (a class with no true weight is
    dropped with sklearn's warning), chance-adjusted when `adjusted`."""
    return _NATIVE(None, _OP.family_mexp_cls_balanced_accuracy,
                   (y_true, y_pred, sample_weight, adjusted, numeric_mode),
                   _NATIVE_CONTEXT)


def matthews_corrcoef(y_true, y_pred, *, sample_weight=None, numeric_mode=None):
    """scikit-learn 1.9 `matthews_corrcoef` (binary and multiclass): the
    covariance form over the per-class true / predicted sums and the trace."""
    return _NATIVE(None, _OP.family_mexp_cls_matthews, (y_true, y_pred, sample_weight, numeric_mode), _NATIVE_CONTEXT)


def cohen_kappa_score(y1, y2, *, labels=None, weights=None, sample_weight=None,
                      replace_undefined_by=float("nan"), numeric_mode=None):
    """scikit-learn 1.9 `cohen_kappa_score`: the (weighted) confusion matrix
    over `labels` on the device, the kappa in the binary64 epilogue;
    weights None, 'linear' or 'quadratic'."""
    return _NATIVE(None, _OP.family_mexp_cls_kappa,
                   (y1, y2, labels, weights, sample_weight, replace_undefined_by, numeric_mode),
                   _NATIVE_CONTEXT)


def hamming_loss(y_true, y_pred, *, sample_weight=None, numeric_mode=None):
    """scikit-learn 1.9 `hamming_loss` for binary / multiclass 1-D targets:
    the (weighted) fraction of mismatched labels."""
    return _NATIVE(None, _OP.family_mexp_cls_hamming, (y_true, y_pred, sample_weight, numeric_mode), _NATIVE_CONTEXT)


def zero_one_loss(y_true, y_pred, *, normalize=True, sample_weight=None, numeric_mode=None):
    """scikit-learn 1.9 `zero_one_loss` for binary / multiclass 1-D targets."""
    return _NATIVE(None, _OP.family_mexp_cls_zero_one,
                   (y_true, y_pred, sample_weight, normalize, numeric_mode),
                   _NATIVE_CONTEXT)


def accuracy_fraction(y_true, y_pred, sample_weight=None, numeric_mode=None):
    """sklearn's ClassifierMixin.score / accuracy_score over labels of any
    kind: the (weighted) match count over the row count or the weight total,
    both from the x_metrics binding's grouped sums (lane pyglue-sweep: the
    weight total is the program's PairSum, not a host sum; lane
    box-run-2-fix: the weighted hit and total are soft-f64 sums on the
    device, `_wsum_sf64`, no longer Float32 PairSums). Lane
    cgr4-py-compute: the estimators' `score` methods called this instead of
    a per-row Python comparison."""
    return _NATIVE(None, _OP.family_mcls_accuracy_fraction, (y_true, y_pred, sample_weight, numeric_mode),
                   _NATIVE_CONTEXT)


def class_likelihood_ratios(y_true, y_pred, *, labels=None, sample_weight=None,
                            replace_undefined_by=float("nan"), numeric_mode=None):
    """scikit-learn 1.9 `class_likelihood_ratios` (binary targets): LR+ and
    LR- from the 2 x 2 (weighted) confusion matrix; `replace_undefined_by`
    np.nan, 1.0 or {'LR+': v, 'LR-': v}."""
    return _NATIVE(None, _OP.family_mexp_cls_likelihood_ratios,
                   (y_true, y_pred, labels, sample_weight, replace_undefined_by, numeric_mode),
                   _NATIVE_CONTEXT)


def classification_report(y_true, y_pred, *, labels=None, target_names=None, sample_weight=None,
                          digits=2, output_dict=False, zero_division="warn", numeric_mode=None):
    """scikit-learn 1.9 `classification_report` for binary / multiclass 1-D
    targets, from precision_recall_fscore_support."""
    return _NATIVE(None, _OP.family_mexp_cls_report,
                   (y_true, y_pred, labels, target_names, sample_weight, digits, output_dict, zero_division, numeric_mode),
                   _NATIVE_CONTEXT)


# ---------------------------------------------------------------------------
# Regression (scikit-learn 1.9 sklearn/metrics/_regression.py)
# ---------------------------------------------------------------------------

_TERM = dict(sq=0, abs=1, sqlog=2, ape=3, pinball=4, tweedie=5, diff=6)


def regression_error_options(name, y_true, y_pred, sample_weight, multioutput, numeric_mode):
    """mean_squared_error / mean_absolute_error / root_mean_squared_error
    with sample_weight, multioutput or 2-D targets (lane/metrics); the 1-D
    unweighted call keeps its original kernel and bits."""
    return _NATIVE(None, _OP.family_metric_regression_error,
                   (name, y_true, y_pred, sample_weight, multioutput, numeric_mode), _NATIVE_CONTEXT)


def mean_squared_log_error(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average",
                           numeric_mode=None):
    """scikit-learn 1.9 `mean_squared_log_error`: the mean of
    `(log1p(y) - log1p(p))^2` (portable log1p, IDENTITY_PATHS row 51);
    values <= -1 are refused as scikit-learn refuses them."""
    return _NATIVE(None, _OP.family_metric_msle, (y_true, y_pred, sample_weight, multioutput, numeric_mode),
                   _NATIVE_CONTEXT)


def root_mean_squared_log_error(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average",
                                numeric_mode=None):
    """scikit-learn 1.9 `root_mean_squared_log_error` (per-output root, then averaged)."""
    return _NATIVE(None, _OP.family_metric_rmsle, (y_true, y_pred, sample_weight, multioutput, numeric_mode),
                   _NATIVE_CONTEXT)


def mean_absolute_percentage_error(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average",
                                   numeric_mode=None):
    """scikit-learn 1.9 `mean_absolute_percentage_error`: `|p - y| / max(|y|,
    eps)` with numpy's float64 epsilon, averaged (not multiplied by 100)."""
    return _NATIVE(None, _OP.family_metric_mape, (y_true, y_pred, sample_weight, multioutput, numeric_mode),
                   _NATIVE_CONTEXT)


def mean_pinball_loss(y_true, y_pred, *, sample_weight=None, alpha=0.5, multioutput="uniform_average",
                      numeric_mode=None):
    """scikit-learn 1.9 `mean_pinball_loss` (alpha is rounded to Float32)."""
    return _NATIVE(None, _OP.family_metric_pinball, (y_true, y_pred, sample_weight, multioutput, numeric_mode, alpha),
                   _NATIVE_CONTEXT)


def median_absolute_error(y_true, y_pred, *, multioutput="uniform_average", sample_weight=None,
                          numeric_mode=None):
    """scikit-learn 1.9 `median_absolute_error`: the per-output median of
    ``|y - p|`` (the weighted percentile at 50 with averaging when weighted, the
    same answer as numpy's median when not), from a stable device sort."""
    return _NATIVE(None, _OP.family_metric_medae, (y_true, y_pred, sample_weight, multioutput, numeric_mode),
                   _NATIVE_CONTEXT)


def max_error(y_true, y_pred, *, numeric_mode=None):
    """scikit-learn 1.9 `max_error` (single output)."""
    return _NATIVE(None, _OP.family_metric_max_error, (y_true, y_pred, numeric_mode), _NATIVE_CONTEXT)


def explained_variance_score(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average",
                             force_finite=True, numeric_mode=None):
    """scikit-learn 1.9 `explained_variance_score`: `1 - Var(y - p) / Var(y)`
    with (weighted) means, one program (the means, then the centered
    squares); force_finite=False returns scikit-learn's NaN / -inf by value."""
    return _NATIVE(None, _OP.family_metric_explained_variance,
                   (y_true, y_pred, sample_weight, multioutput, force_finite, numeric_mode), _NATIVE_CONTEXT)


def r2_score_options(y_true, y_pred, sample_weight, multioutput, force_finite, numeric_mode):
    """r2_score with multioutput, 2-D targets or force_finite=False
    (lane/metrics): `sum w (y - p)^2` over `sum w (y - avg)^2` per output;
    the 1-D default call keeps its original kernel and bits."""
    return _NATIVE(None, _OP.family_metric_r2_options,
                   (y_true, y_pred, sample_weight, multioutput, force_finite, numeric_mode), _NATIVE_CONTEXT)


def mean_tweedie_deviance(y_true, y_pred, *, sample_weight=None, power=0, numeric_mode=None):
    """scikit-learn 1.9 `mean_tweedie_deviance` (single output; the power is
    rounded to Float32; pow/log are the portable row-12 functions)."""
    return _NATIVE(None, _OP.family_metric_tweedie, (y_true, y_pred, sample_weight, numeric_mode, power),
                   _NATIVE_CONTEXT)


def mean_poisson_deviance(y_true, y_pred, *, sample_weight=None, numeric_mode=None):
    """scikit-learn 1.9 `mean_poisson_deviance` (Tweedie power 1)."""
    return _NATIVE(None, _OP.family_metric_poisson, (y_true, y_pred, sample_weight, numeric_mode), _NATIVE_CONTEXT)


def mean_gamma_deviance(y_true, y_pred, *, sample_weight=None, numeric_mode=None):
    """scikit-learn 1.9 `mean_gamma_deviance` (Tweedie power 2)."""
    return _NATIVE(None, _OP.family_metric_gamma, (y_true, y_pred, sample_weight, numeric_mode), _NATIVE_CONTEXT)


def d2_tweedie_score(y_true, y_pred, *, sample_weight=None, power=0, numeric_mode=None):
    """scikit-learn 1.9 `d2_tweedie_score`: `1 - dev(y, p) / dev(y, avg y)`
    in one program; a zero null deviance answers NaN (0/0) or -inf as
    numpy's float division does."""
    return _NATIVE(None, _OP.family_metric_d2_tweedie, (y_true, y_pred, sample_weight, numeric_mode, power),
                   _NATIVE_CONTEXT)


def d2_pinball_score(y_true, y_pred, *, sample_weight=None, alpha=0.5, multioutput="uniform_average",
                     numeric_mode=None):
    """scikit-learn 1.9 `d2_pinball_score`: the pinball loss against the
    (weighted, averaged) alpha-quantile of y_true per output, one program."""
    return _NATIVE(None, _OP.family_metric_d2_pinball, (y_true, y_pred, sample_weight, multioutput, numeric_mode, alpha),
                   _NATIVE_CONTEXT)


def d2_absolute_error_score(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average",
                            numeric_mode=None):
    """scikit-learn 1.9 `d2_absolute_error_score` (d2_pinball_score at alpha 0.5)."""
    return _NATIVE(None, _OP.family_metric_d2_absolute, (y_true, y_pred, sample_weight, multioutput, numeric_mode),
                   _NATIVE_CONTEXT)


# ---------------------------------------------------------------------------
# Ranking and probabilistic scores (scikit-learn 1.9 _ranking.py, _classification.py)
# ---------------------------------------------------------------------------

_ROW = dict(topk=0, brier=1, logloss=2, hinge_bin=3, hinge_mc=4, dcg=5, coverage=6, lrap=7, rankloss=8,
            dcg_ignore_ties=9)


#: words per problem of the curve fold (x_metrics/par.mojo CF_OUT) and its
#: points per chunk (x_metrics/plan.mojo CF_CHUNK)
_CF_OUT = 10
_CF_CHUNK = 1024


#: x_metrics/curve_out.mojo kinds and statuses
_CO_ROC, _CO_PR, _CO_DET = 0, 1, 2
_CO_EMPTY, _CO_ZERO_CLASS = 1, 2


# Engineering notes (moved from the docstring):
# scikit-learn 1.9 `roc_curve` for binary targets: fpr, tpr, thresholds
# (Float64; the first threshold is +inf). The sort, the cumulative
# counts, the collinear drop and the rates run on the device (lane
# cpu4-python: x_metrics/curve_out.mojo; an empty class's NaN rates
# there too).
def roc_curve(y_true, y_score, *, pos_label=None, sample_weight=None, drop_intermediate=True,
              numeric_mode=None):
    """Receiver operating characteristic curve for binary targets.

    Follows scikit-learn's `roc_curve`. Returns ``fpr``, ``tpr`` and ``thresholds`` as
    float64 arrays; the first threshold is ``+inf``. Computed on the GPU.
    """
    return _NATIVE(None, _OP.family_mexp_curve_roc,
                   (y_true, y_score, pos_label, sample_weight, drop_intermediate, numeric_mode),
                   _NATIVE_CONTEXT)


# Engineering notes (moved from the docstring):
# scikit-learn 1.9 `det_curve`: fpr, fnr, thresholds, formed on the
# device (lane cpu4-python: x_metrics/curve_out.mojo); a zero class
# weight raises as the Python division by zero did.
def det_curve(y_true, y_score, *, pos_label=None, sample_weight=None, drop_intermediate=False,
              numeric_mode=None):
    """Detection error tradeoff curve for binary targets.

    Follows scikit-learn's `det_curve`. Returns ``fpr``, ``fnr`` and ``thresholds``,
    computed on the GPU. A zero class weight raises ``ZeroDivisionError``.
    """
    return _NATIVE(None, _OP.family_mexp_curve_det,
                   (y_true, y_score, pos_label, sample_weight, drop_intermediate, numeric_mode),
                   _NATIVE_CONTEXT)


# Engineering notes (moved from the docstring):
# scikit-learn 1.9 `auc`: the trapezoid rule over a monotonic x, on
# the device (lane cpu4-python: x_metrics/curve_out.mojo auc_xy; the host
# column runs the same units): binary64 terms (x[i] - x[i-1]) * (y[i] +
# y[i-1]) / 2, summed as a float-float in fixed chunks (the host fsum
# left the route); a non-finite term's IEEE answer. A list or an integer
# buffer is converted to Float64 first.
def auc(x, y):
    """Area under a curve by the trapezoid rule.

    Follows scikit-learn's `auc`. `x` must be monotonic. Lists and integer arrays are
    converted to float64 first. Computed on the GPU.
    """
    return _NATIVE(None, _OP.family_mexp_curve_auc, (x, y), _NATIVE_CONTEXT)


#: rank_epi kinds of the ranking averages (x_metrics/rank_epi.mojo, lane cpu2-l7-metrics S3b)
_RANK_ROW_SUM, _RANK_CURVE_SCORE, _RANK_AVERAGE, _RANK_OVO_MASK, _RANK_OVO_PAIR = 5, 6, 7, 8, 9
_ROC_ONE_CLASS = "Only one class is present in y_true. ROC AUC score is not defined in that case."


#: arena words one one-vs-one program may hold (its pairs' curve slots);
#: more pairs go to further programs
_OVO_WORDS = 1 << 26
_ROWS_MSG = ("Target scores need to be probabilities for multiclass roc_auc, i.e. they "
             "should sum up to 1.0 over classes")


def average_precision_score(y_true, y_score, *, average="macro", pos_label=1, sample_weight=None,
                            numeric_mode=None):
    """scikit-learn 1.9 `average_precision_score` (binary and multiclass
    one-vs-rest; multilabel-indicator targets are refused by name)."""
    return _NATIVE(None, _OP.family_mexp_curve_average_precision,
                   (y_true, y_score, pos_label, sample_weight, numeric_mode, average),
                   _NATIVE_CONTEXT)


def top_k_accuracy_score(y_true, y_score, *, k=2, normalize=True, sample_weight=None, labels=None,
                         numeric_mode=None):
    """scikit-learn 1.9 `top_k_accuracy_score`: a hit when the true class is
    among the k largest scores, ties ordered as numpy's stable argsort
    reversed (the larger class index first)."""
    return _NATIVE(None, _OP.family_mexp_curve_top_k,
                   (y_true, y_score, k, normalize, sample_weight, labels, numeric_mode),
                   _NATIVE_CONTEXT)


#: rank_epi kinds (x_metrics/rank_epi.mojo)
_RANK_MEAN, _RANK_NDCG_ROW, _RANK_D2_LOG, _RANK_D2_BRIER, _RANK_DCG_DISC = 0, 1, 2, 3, 4


def brier_score_loss(y_true, y_proba, *, sample_weight=None, pos_label=None, labels=None,
                     scale_by_half="auto", numeric_mode=None):
    """scikit-learn 1.9 `brier_score_loss` (binary vector or multiclass
    matrix): the mean of `sum_c (onehot - p)^2`, halved by default for the
    binary case."""
    return _NATIVE(None, _OP.family_mexp_rank_brier,
                   (y_true, y_proba, sample_weight, pos_label, labels, scale_by_half, numeric_mode),
                   _NATIVE_CONTEXT)


def d2_log_loss_score(y_true, y_proba=None, *, sample_weight=None, labels=None, numeric_mode=None):
    """scikit-learn 1.9 `d2_log_loss_score`: one minus the log loss over the
    log loss of the (weighted) class frequencies."""
    return _NATIVE(None, _OP.family_mexp_rank_d2_log_loss,
                   (y_true, y_proba, sample_weight, labels, None, numeric_mode),
                   _NATIVE_CONTEXT)


def d2_brier_score(y_true, y_proba, *, sample_weight=None, pos_label=None, labels=None, numeric_mode=None):
    """scikit-learn 1.9 `d2_brier_score`: one minus the Brier score over the
    Brier score of the (weighted) class frequencies."""
    return _NATIVE(None, _OP.family_mexp_rank_d2_brier,
                   (y_true, y_proba, sample_weight, labels, pos_label, numeric_mode),
                   _NATIVE_CONTEXT)


def hinge_loss(y_true, pred_decision, *, labels=None, sample_weight=None, numeric_mode=None):
    """scikit-learn 1.9 `hinge_loss`: binary (labels mapped to -1 / +1, the
    larger class positive) or Crammer-Singer multiclass margins."""
    return _NATIVE(None, _OP.family_mexp_rank_hinge,
                   (y_true, pred_decision, labels, sample_weight, numeric_mode),
                   _NATIVE_CONTEXT)


def dcg_score(y_true, y_score, *, k=None, log_base=2, sample_weight=None, ignore_ties=False,
              numeric_mode=None):
    """scikit-learn 1.9 `dcg_score`: tie-averaged gains by default;
    ignore_ties=True ranks tied scores by DESCENDING column index (numpy's
    `argsort(...)[::-1]` there, whose tie order its unstable sort leaves
    unspecified; this one is fixed)."""
    return _NATIVE(None, _OP.family_mexp_rank_dcg,
                   (y_true, y_score, k, log_base, sample_weight, ignore_ties, numeric_mode),
                   _NATIVE_CONTEXT)


# Engineering notes (moved from the docstring):
# scikit-learn 1.9 `ndcg_score`: each row's DCG over its ideal DCG
# (0 when the row has no relevant item), (weighted) averaged. The gains,
# the per-row ratios (`rank_epi` NDCG_ROW: the binary64 quotient narrowed
# to Float32), their (weighted) PairSum and the mean run in one device
# program (lane cpu2-l7-metrics: no host epilogue over the rows).
def ndcg_score(y_true, y_score, *, k=None, sample_weight=None, ignore_ties=False, numeric_mode=None):
    """Normalized discounted cumulative gain.

    Follows scikit-learn's `ndcg_score`: each row's DCG divided by its ideal DCG (0 when
    the row has no relevant item), averaged over rows, weighted when `sample_weight` is
    given. Computed on the GPU.
    """
    return _NATIVE(None, _OP.family_mexp_rank_ndcg,
                   (y_true, y_score, k, sample_weight, ignore_ties, numeric_mode),
                   _NATIVE_CONTEXT)


def coverage_error(y_true, y_score, *, sample_weight=None, numeric_mode=None):
    """scikit-learn 1.9 `coverage_error` (multilabel indicator y_true)."""
    return _NATIVE(None, _OP.family_mexp_rank_coverage, (y_true, y_score, sample_weight, numeric_mode), _NATIVE_CONTEXT)


def label_ranking_average_precision_score(y_true, y_score, *, sample_weight=None, numeric_mode=None):
    """scikit-learn 1.9 `label_ranking_average_precision_score`."""
    return _NATIVE(None, _OP.family_mexp_rank_lrap, (y_true, y_score, sample_weight, numeric_mode), _NATIVE_CONTEXT)


def label_ranking_loss(y_true, y_score, *, sample_weight=None, numeric_mode=None):
    """scikit-learn 1.9 `label_ranking_loss` (tied scores count as misordered)."""
    return _NATIVE(None, _OP.family_mexp_rank_loss, (y_true, y_score, sample_weight, numeric_mode), _NATIVE_CONTEXT)


# ---------------------------------------------------------------------------
# Clustering (scikit-learn 1.9 sklearn/metrics/cluster/)
# ---------------------------------------------------------------------------


def contingency_matrix(labels_true, labels_pred, *, eps=None, sparse=False, dtype="int64", numeric_mode=None):
    """scikit-learn 1.9 `contingency_matrix` (dense): exact counts from the
    device grouping, rows the sorted classes, columns the sorted clusters;
    `eps` adds a constant and makes the result Float64. The result is always
    dense; `sparse=True` raises `NotImplementedError`."""
    return _NATIVE(None, _OP.family_mexp_cluster_contingency,
                   (labels_true, labels_pred, eps, sparse, dtype, numeric_mode),
                   _NATIVE_CONTEXT)


def pair_confusion_matrix(labels_true, labels_pred, *, numeric_mode=None):
    """scikit-learn 1.9 `pair_confusion_matrix`: the 2 x 2 Int64 pair counts
    from the exact contingency matrix (exact Int64 in the binding: every
    pair count is below 2^62)."""
    return _NATIVE(None, _OP.family_mexp_cluster_pair_confusion,
                   (labels_true, labels_pred, numeric_mode),
                   _NATIVE_CONTEXT)


# Engineering notes (moved from the docstring):
# scikit-learn 1.9 `normalized_mutual_info_score`, in nats as scikit-learn
# (exact device contingency counts; the logarithms are the portable binary64
# ones of `_portable_math`, DEVIATION 6106).
def normalized_mutual_info_score(labels_true, labels_pred, *, average_method="arithmetic", numeric_mode=None):
    """Normalized mutual information between two clusterings.

    Follows scikit-learn's `normalized_mutual_info_score`, in nats.
    """
    return _NATIVE(None, _OP.family_mexp_cluster_nmi,
                   (labels_true, labels_pred, average_method, numeric_mode),
                   _NATIVE_CONTEXT)


def adjusted_mutual_info_score(labels_true, labels_pred, *, average_method="arithmetic", numeric_mode=None):
    """scikit-learn 1.9 `adjusted_mutual_info_score`: (MI - E[MI]) /
    (mean(H) - E[MI]) with scikit-learn's epsilon guards."""
    return _NATIVE(None, _OP.family_mexp_cluster_ami,
                   (labels_true, labels_pred, average_method, numeric_mode),
                   _NATIVE_CONTEXT)


def calinski_harabasz_score(X, labels, *, numeric_mode=None):
    """scikit-learn 1.9 `calinski_harabasz_score`: the between- over the
    within-cluster dispersion, scaled by (n - k) / (k - 1), in one device
    program."""
    return _NATIVE(None, _OP.family_mexp_cluster_calinski_harabasz, (X, labels, numeric_mode), _NATIVE_CONTEXT)


def davies_bouldin_score(X, labels, *, numeric_mode=None):
    """scikit-learn 1.9 `davies_bouldin_score`: the mean over clusters of the
    worst (s_i + s_j) / d(c_i, c_j), in one device program."""
    return _NATIVE(None, _OP.family_mexp_cluster_davies_bouldin, (X, labels, numeric_mode), _NATIVE_CONTEXT)


# ---------------------------------------------------------------------------
# The splitters' random draws (DEVIATION 6108; x_metrics/split.mojo)
# ---------------------------------------------------------------------------


def regression_report(y_true, y_pred, *, sample_weight=None,
                      multioutput="uniform_average", force_finite=True,
                      numeric_mode="identical"):
    """Return mean squared error, mean absolute error and R² together."""
    return _NATIVE(None, _OP.family_metric_regression_report,
                   (y_true, y_pred, sample_weight, multioutput, force_finite, numeric_mode),
                   _NATIVE_CONTEXT)


def ranking_report(y_true, y_score, *, pos_label=None, sample_weight=None,
                   include_curves=False, numeric_mode="identical"):
    """Return binary ROC AUC, average precision and optional curves together."""
    return _NATIVE(None, _OP.family_mexp_curve_ranking_report,
                   (y_true, y_score, pos_label, sample_weight, include_curves, numeric_mode),
                   _NATIVE_CONTEXT)


# Import-time connection of the shared native program runner. Every x_metrics
# program is built in Mojo (bindings/metrics_reg_native.mojo `_Prog`/`_Reg`,
# bindings/msel_split_door.mojo `_fold_rows_run`, core/metrics_program_plan.mojo);
# no Python program object exists, so the runner gets no Python program class.
# TOMBSTONE: the Python program builders `_Prog`, `_Reg`, `_group`, `_counts`, `_scan_flag`, `_proba`,
# `_targets`, `_pos_label`, `fold_rows`, `_fold_rows_run`, `_r2_sums_of` and their helpers (dead copies of the
# native owners above; lane pyout-metrics-cnn) deleted 2026-11-11; code recoverable at 6f82d0255d.
from ._native_arena_prog import connect as _connect_arena_prog
_PROG_CONTEXT = _connect_arena_prog("x_metrics", None)[1]

# The compiled family owner (bindings/metrics_reg_native.mojo, reached from
# the classical door): the regression metrics above are one native call each.
# It builds each x_metrics program in Mojo, with the fixed op and term codes
# below, and runs it through the shared program runner connected above.
from . import _native_family_bridge  # noqa: E402
_NATIVE, _NATIVE_CONTEXT = _native_family_bridge.connect()
# Lane ne1-h-labels: operations pass as integer codes from the one native
# table (bindings/native_operation_codes.mojo), read once at import.
from ._native_ops import codes as _operation_codes  # noqa: E402
_OP = _operation_codes()
_native_family_bridge.tables(family_metrics_ops=dict(_OPS), family_metrics_terms=dict(_TERM),
                             family_metrics_prog_context=_PROG_CONTEXT,
                             family_metrics_undefined=UndefinedMetricWarning)
# TOMBSTONE: family_metrics_real, family_mexp_integral (dead context entries: no native owner read them) deleted 2026-11-09 by cb99d1b18c; code recoverable at cb99d1b18c^.
# Restore: git apply experiments/removed/dead_code_p5_20261109.patch

from ._native_owner_lifetime import connect_owner_context as _connect_metric_owners
_connect_metric_owners(_NATIVE_CONTEXT)
