# SPDX-License-Identifier: Apache-2.0
"""The member boundary of the ExtraTrees family (ExtraTreesClassifier ExtraTreesRegressor): its place in
`bindings/member_boundary_door.mojo` (recipe: MEMBER-ABI-V2.md). Kind 0 is
the classifier, kind 1 the regressor (`_mojolearn_member = (MEMBER_FAMILY_EXTRA_TREES, kind)`
on each class). The UNPACK / PACK bodies are the shared forest ones
(providers/python/forest_member_common_boundary_door.mojo), for the classes
that connect under the kind prefix "et_"; the RUN is core/extra_trees_member.mojo.
"""
from std.python import PythonObject
from bindings.member_owner_unpack import MemberOwner
from core.member_abi import MemberPlan
from providers.python.forest_member_common_boundary_door import (
    forest_member_admit_common, forest_member_pack_common, forest_member_state_common,
)


def extra_trees_member_admit(model: PythonObject, fields: PythonObject, kind: Int, context: PythonObject,
    b: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """UNPACK: fill `owner.plan.route`, `owner.plan.params` and `owner.pins`
    from the unfitted estimator. False: this configuration keeps the
    estimator's own doors."""
    return forest_member_admit_common(model, kind, "et_", context, owner)


def extra_trees_member_pack(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan, state: Int) raises:
    """PACK: write the fitted attributes of state `state` into `model`."""
    forest_member_pack_common(model, context, plan, state)


def extra_trees_member_state(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan) raises -> Int:
    """UNPACK: a new member state holding the fitted attributes of `model`."""
    return forest_member_state_common(model, context, plan)
