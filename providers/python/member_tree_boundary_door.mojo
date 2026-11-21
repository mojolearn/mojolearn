# SPDX-License-Identifier: Apache-2.0
"""The member boundary of the Tree family (DecisionTreeClassifier DecisionTreeRegressor): NOT PUBLISHED YET.

This file is the family's place in `bindings/member_boundary_door.mojo`. The
family's publication replaces the three bodies below (recipe:
MEMBER-ABI-V2.md, worked form: providers/python/member_linear_boundary_door.mojo)
and sets `_mojolearn_member = (MEMBER_FAMILY_TREE, kind)` on each class
it publishes. Until then no class carries the mark and a caller keeps the
member's own doors.
"""
from std.python import PythonObject
from bindings.member_owner_unpack import MemberOwner
from core.member_abi import MemberPlan


def tree_member_admit(model: PythonObject, fields: PythonObject, kind: Int, context: PythonObject,
    b: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """UNPACK: fill `owner.plan.route`, `owner.plan.params` and `owner.pins`
    from the unfitted estimator. False: this family has no member entry."""
    return False


def tree_member_pack(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan, state: Int) raises:
    """PACK: write the fitted attributes of state `state` into `model`."""
    raise Error("RuntimeError: mojolearn: the Tree family has not published its member entry")


def tree_member_state(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan) raises -> Int:
    """UNPACK: a new member state holding the fitted attributes of `model`."""
    raise Error("RuntimeError: mojolearn: the Tree family has not published its member entry")
