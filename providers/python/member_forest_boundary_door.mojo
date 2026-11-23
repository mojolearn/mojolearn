# SPDX-License-Identifier: Apache-2.0
"""The member boundary of the Forest family (RandomForestClassifier RandomForestRegressor): its place in
`bindings/member_boundary_door.mojo` (recipe: MEMBER-ABI-V2.md). Kind 0 is
the classifier, kind 1 the regressor (`_mojolearn_member = (MEMBER_FAMILY_FOREST, kind)`
on each class); kind 2 a saved `HostForest`, predict-only (lane mf-hostent5). The UNPACK / PACK bodies are the shared forest ones
(providers/python/forest_member_common_boundary_door.mojo), for the classes
that connect under the kind prefix "rf_"; the RUN is core/forest_member.mojo.
"""
from std.python import PythonObject
from bindings.member_owner_unpack import MemberOwner
from core.member_abi import MemberPlan
from providers.python.forest_member_common_boundary_door import (
    forest_member_admit_common, forest_member_pack_common, forest_member_state_common, forest_member_host_admit,
)


def forest_member_admit(model: PythonObject, fields: PythonObject, kind: Int, context: PythonObject,
    b: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """UNPACK: fill `owner.plan.route`, `owner.plan.params` and `owner.pins`
    from the unfitted estimator. False: this configuration keeps the
    estimator's own doors."""
    if kind == 2:
        # A saved HostForest (lane mf-hostent5): the predict-only kind; an extra-trees file's state is the
        # ExtraTrees family's (the admit names it in `owner.plan.family`).
        return forest_member_host_admit(model, context, owner)
    return forest_member_admit_common(model, kind, "rf_", context, owner)


def forest_member_pack(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan, state: Int) raises:
    """PACK: write the fitted attributes of state `state` into `model`."""
    forest_member_pack_common(model, context, b, plan, state)


def forest_member_state(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan) raises -> Int:
    """UNPACK: a new member state holding the fitted attributes of `model`."""
    return forest_member_state_common(model, context, plan)
