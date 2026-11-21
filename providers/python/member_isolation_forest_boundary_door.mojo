# SPDX-License-Identifier: Apache-2.0
"""The member boundary of the IsolationForest family (IsolationForest),
UNPACK / PACK for core/isolation_forest_member.mojo.

ADMIT is published: the estimator's own parameter checks give the thirteen
NEW words, and the route's `entry` is the native iforest entry of the svm
image the bridge connected for the estimator's tier (core/iforest_entry.mojo).
PACK and STATE are not: the member exports no fitted blocks yet (the
forest's node arrays leave the image only through `iforest_forest_export`,
still a Python-argument binding), so there is no fitted estimator to write
and none to import. No class carries the `_mojolearn_member` mark, so no
caller reaches these until the blocks are published.
"""
from std.python import Python, PythonObject
from bindings.member_owner_unpack import MemberOwner
from bindings.trees_route_boundary_door import trees_route_context
from bindings.trees_public_iforest_door import iforest_member_words
from core.device_route import DeviceRoute, DEVICE_CPU, DEVICE_GPU
from core.iforest_entry import if_entry_ref
from core.member_abi import MemberPlan


def isolation_forest_member_admit(model: PythonObject, fields: PythonObject, kind: Int, context: PythonObject,
    b: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """UNPACK: `owner.plan.params` (thirteen words) and `owner.plan.route`
    (`entry` the iforest entry) from the unfitted estimator."""
    if kind != 0:
        return False
    var run = trees_route_context(model, context)
    var words = iforest_member_words(model, fields, run)
    var mode = Int(words[13])
    var vendor = Int(words[14])
    var route = DeviceRoute(DEVICE_CPU if vendor == 0 else DEVICE_GPU)
    route.mode = mode
    route.vendor = vendor
    route.entry = if_entry_ref(mode, vendor)
    var params = List[Float64](capacity=13)
    for i in range(13):  # small-loop(NEW words): the thirteen member parameters
        params.append(words[i])
    owner.plan.route = route^
    owner.plan.params = params^
    owner.pins = Python.list(run)
    return True


def isolation_forest_member_pack(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan, state: Int) raises:
    """PACK: not published (the member exports no fitted blocks yet)."""
    raise Error("RuntimeError: mojolearn: the IsolationForest member exports no fitted state yet; "
                + "a caller keeps the estimator's own fit")


def isolation_forest_member_state(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan) raises -> Int:
    """UNPACK of a fitted estimator: not published (no IMPORT yet)."""
    raise Error("RuntimeError: mojolearn: the IsolationForest member imports no fitted state yet; "
                + "a caller keeps the estimator's own predict")
