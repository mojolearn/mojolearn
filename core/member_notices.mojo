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
