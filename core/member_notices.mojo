# SPDX-License-Identifier: Apache-2.0
"""FIT NOTICES of a member state: the ONE mechanism through which a member
fit's warnings reach the user (lane fx-audit, audit-trees F1 and
audit-classical-2 F1).

A family's own fit door raises some warnings at fit (RandomForest's `n_bins`
and `max_samples` notices, Nystroem's `n_components > n_samples`, the
GaussianProcessRegressor ridge retries, the PLS constant residual). On a member route the fit runs in
RUN with the GIL released, so the state keeps them as WORDS, set by FIT and
cleared by the next fit, import or free, and a family with any sets
`MEMBER_CAN_NOTICES` and exports them by name as `MEMBER_BLOCK_NOTICES`:
float64, rows = notices, `MEMBER_NOTICE_WORDS` columns
`[code, operand a, operand b]` (rows 0: none).

Every caller (model selection units, the member set behind bagging, voting,
stacking and AdaBoost, recursive feature elimination) takes the words once
per fitted unit (`member_notices_take`, no GIL, one mask test for a family
without the bit) and raises them on the public thread once its publication
can no longer fall back to the estimator's own doors
(bindings/member_fit_notices_pack.mojo `member_fit_notices_raise`), whether
or not it packs the unit. A family's PACK never raises them itself.

SERIAL BY NATURE: a handful of words per unit, read on the unit's own worker.
"""
from core.member_abi import (
    MemberCall, MemberPlan, member_call, MEMBER_OP_EXPORT, MEMBER_BLOCK_NOTICES, MEMBER_CAN_NOTICES, MEMBER_F64,
)

comptime MEMBER_NOTICE_WORDS = 3
#: RandomForest / DecisionTree (the RF layout): a = the `ensemble/rf_notices.mojo` bit mask of the fit.
comptime MEMBER_NOTICE_FOREST = 1
#: Nystroem: n_components was clamped to the rows (a = requested, b = rows).
comptime MEMBER_NOTICE_NYSTROEM_COMPONENTS = 2
#: GaussianProcessRegressor: the ridge was raised tenfold (a = the starting alpha, b = the alpha used).
comptime MEMBER_NOTICE_GP_RIDGE = 3
#: PLS: the fit stopped before n_components, the y residual constant (a = the components fitted).
comptime MEMBER_NOTICE_PLS_RESIDUAL = 4
#: A meta member's cv splitter (lane mf-split4; StratifiedKFold's plan,
#: bindings/msel_split_plan_run.mojo `msel_plan_stratified`): the least
#: populated class holds fewer rows than n_splits (a = its rows, b = n_splits).
comptime MEMBER_NOTICE_SPLIT_LEAST = 5
# THE SCORERS' METRIC WARNINGS (lane mf-score5): a search unit's scores
# (core/msel_search_run.mojo) keep, after the unit's fit notices, the
# warnings the public metric its scorer calls raises on that side, one per
# occurrence in the scorer's order, as scikit-learn's scorer raises them
# during the search. The PACK raises them in the public metric's class and
# words (bindings/member_fit_notices_pack.mojo).
#: precision / recall / f-score with a zero denominator (bindings/mcls_classify_door.mojo
#: `_prf`; a = 0 precision, 1 recall, 2 f-score).
comptime MEMBER_NOTICE_METRIC_PRF = 6
#: jaccard_score with a label of no true and no predicted samples (bindings/metrics_exp_cls_door.mojo `_jaccard`).
comptime MEMBER_NOTICE_METRIC_JACCARD = 7
#: balanced_accuracy_score: y_pred holds classes y_true does not (`_balanced_accuracy`).
comptime MEMBER_NOTICE_METRIC_BALANCED = 8
#: r2_score (a = 0) or d2_absolute_error_score (a = 1) of fewer than two samples (bindings/metrics_reg_door.mojo).
comptime MEMBER_NOTICE_METRIC_FEW = 9
#: top_k_accuracy_score with k >= n_classes (a = k, b = n_classes; bindings/metrics_exp_curve_door.mojo `_top_k`).
comptime MEMBER_NOTICE_METRIC_TOP_K = 10
#: RidgeCV's split path (lane mf-split5, core/xlinear_member.mojo): a cv
#: split with fewer than two test rows, whose R^2 scikit-learn leaves
#: undefined with an UndefinedMetricWarning (no operands). Kept apart from
#: MEMBER_NOTICE_METRIC_FEW (same words): that one is a search scorer's
#: notice raised with the search's metrics context; this one is the x_linear
#: member's own fit notice, raised with the x_linear context's category.
comptime MEMBER_NOTICE_R2_FEW = 11
#: cross_val_predict (lane mf-msel5b, core/msel_search_run.mojo): a fold
#: whose train rows hold a = m of the b = k classes, scikit-learn's
#: `_enforce_prediction_order` RuntimeWarning (`member_lacking_warning`).
comptime MEMBER_NOTICE_FOLD_LACKING = 12
#: OneHotEncoder (lane mf-enc6, core/prep_member.mojo kind 11): column a held
#: unknown categories in a transform that warns (handle_unknown='warn', or
#: 'ignore' / 'infrequent_if_exist' with a drop); b = 1: encoded as all
#: zeros ('ignore'), 0: as the infrequent category. Consecutive triples are
#: ONE warning over their columns, the encode door's `_onehot_transform`
#: words. A TRANSFORM notice: the state's notices are the last fit_transform's
#: or transform's (a caller of MEMBER_OP_TRANSFORM takes them after the call
#: as it does after FIT).
comptime MEMBER_NOTICE_ONEHOT_UNKNOWN = 13
#: cohen_kappa_score (lane mf-score8; bindings/metrics_exp_cls_door.mojo
#: `_kappa`): a = 1 no label of y2 is in both y1 and the labels, a = 2 only
#: one label in common (a search scorer's metric warning, raised at PACK).
comptime MEMBER_NOTICE_METRIC_KAPPA = 14

#: The words scikit-learn's cross_val_predict adds to its imbalanced-fold
#: warning and refusals (model_selection/_validation.py
#: `_enforce_prediction_order`, scikit-learn 1.9). The one copy: the
#: calibration's fold placement (xtrees/trees_calibrated_fit_run.mojo) and
#: the search RUN read it.
comptime MEMBER_LACKING_STRATIFY = "To fix this, use a cross-validation technique resulting in properly stratified folds"


def member_lacking_warning(m: Int, k: Int) -> String:
    """The RuntimeWarning of `_enforce_prediction_order` for a fold whose
    train rows hold `m` of the `k` classes."""
    return ("Number of classes in training fold (" + String(m) + ") does not match total number of classes ("
            + String(k) + "). Results may not be appropriate for your use case. " + MEMBER_LACKING_STRATIFY)


def member_notice_add(mut words: List[Float64], code: Int, a: Float64 = 0.0, b: Float64 = 0.0):
    """Append one notice to a state's notice words (a family's FIT)."""
    words.append(Float64(code))
    words.append(a)
    words.append(b)


def member_notices_take(plan: MemberPlan, state: Int, mut dst: List[Float64]) raises:
    """Append the notice words of the fitted state `state` to `dst` (empty
    for a family without `MEMBER_CAN_NOTICES`: nothing is called). Needs no
    GIL; the view the export hands out is copied before the state can be
    freed."""
    if state == 0 or not plan.member.can(MEMBER_CAN_NOTICES):
        return
    var call = MemberCall(plan.route, state)
    call.block = MEMBER_BLOCK_NOTICES
    call.flags = 0
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var count = call.dst.rows * call.dst.cols
    if count == 0 or call.dst.data == 0 or call.dst.dtype != MEMBER_F64:
        return
    var words = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call.dst.data)
    for i in range(count):  # small-loop(a state's notice words: a few triples)
        dst.append(words[i])
