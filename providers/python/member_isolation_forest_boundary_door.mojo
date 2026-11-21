# SPDX-License-Identifier: Apache-2.0
"""The member boundary of the IsolationForest family (IsolationForest),
UNPACK / PACK for core/isolation_forest_member.mojo.

ADMIT: the estimator's own parameter checks on a scratch copy of its
instance dictionary (the unfitted template is never written) give the
thirteen NEW words; the route's `entry` is the native iforest entry of the
svm image the bridge connected for the estimator's tier
(core/iforest_entry.mojo). The route is the device plan: the member runs on
that one device.

PACK: EXPORT the six blocks (scalars, then the compact forest) and write the
fields a LOADED estimator holds (`_if_forest`, `offset_`, `max_samples_`,
`n_features_in_` and the resolved knobs), so its own score_samples,
decision_function, predict, save and pickle work on the result with no
training matrix. STATE: the estimator's saved forest (a loaded or unpickled
one holds it; a fitted one exports the resident forest its scores use) and
its scalars, IMPORTed into a new state; the instance dictionary is only read.
"""
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from bindings.admitted_family_door import admitted_array as family_uninit_array
from bindings.member_owner_unpack import MemberOwner
from bindings.py_balanced import py_call, py_item
from bindings.trees_keys_unpack import trees_keys
from bindings.trees_public_inputs_door import trees_address, trees_builtins
from bindings.trees_route_boundary_door import trees_route_context
from bindings.trees_public_iforest_door import (
    iforest_export_forest, iforest_forest_words, iforest_member_pack_fields, iforest_member_words,
)
from core.device_route import DeviceRoute, DEVICE_CPU, DEVICE_GPU
from core.iforest_entry import if_entry_ref
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, member_call, member_new, member_free,
    MEMBER_OP_EXPORT, MEMBER_OP_IMPORT, MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_FAMILY, MEMBER_IMPORT_DONE,
    MEMBER_F32, MEMBER_F64, MEMBER_I32,
)



def _typestr(block: Int) -> String:
    """The saved form's dtype of forest block `block` (0 offsets, 1 feature,
    2 threshold, 3 left, 4 right)."""
    return "<f4" if block == 2 else "<i4"


def isolation_forest_member_admit(model: PythonObject, fields: PythonObject, kind: Int, context: PythonObject,
    b: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """UNPACK: `owner.plan.params` (thirteen words) and `owner.plan.route`
    (`entry` the iforest entry) from the unfitted estimator, which is only
    read. `owner.pins` holds the run context, never the estimator."""
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


def _export(plan: MemberPlan, mut call: MemberCall, block: Int) raises -> MemberMatrix:
    call.block = block
    call.flags = 0
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    return call.dst


def _array(view: MemberMatrix, typestr: String, context: PythonObject) raises -> PythonObject:
    """A new flat Array holding the block's words (one copy out of the
    state's view, as every member PACK does)."""
    var count = view.rows * view.cols
    var out = family_uninit_array(Python.tuple(count), typestr, context)
    if count > 0:
        unsafe_memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=trees_address(out)),
                      src=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=view.data), count=count * 4)
    return out


def isolation_forest_member_pack(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan, state: Int) raises:
    """PACK: write the fitted attributes of state `state` into `model` (the
    caller's clone) as a loaded estimator holds them."""
    var run = trees_route_context(model, context)
    var call = MemberCall(plan.route, state)
    var scalars = _export(plan, call, MEMBER_BLOCK_SCALARS)
    if not scalars.dense_host(MEMBER_F64) or scalars.cols != 3:
        raise Error("member ABI: the IsolationForest scalars block is three float64 words")
    var w = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=scalars.data)
    var offset = w[0]
    var max_samples = Int(w[1])
    var n_features = Int(w[2])
    var parts = Python.list()
    for i in range(5):  # driver-loop: over the five fixed forest blocks
        var view = _export(plan, call, MEMBER_BLOCK_FAMILY + i)
        _ = py_call(parts.append, _array(view, _typestr(i), run))
    var forest = Python.tuple(parts[0], parts[1], parts[2], parts[3], parts[4])
    iforest_member_pack_fields(model, fields, forest, offset, max_samples, n_features, run)


def _import(plan: MemberPlan, mut call: MemberCall, block: Int, view: MemberMatrix) raises:
    call.block = block
    call.x = view
    call.flags = 0
    _ = member_call(plan.member, MEMBER_OP_IMPORT, call)


def isolation_forest_member_state(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan) raises -> Int:
    """UNPACK: a new member state holding the fitted forest of `model`. The
    caller frees it (`member_free`). Raises when the estimator is not fitted."""
    var run = trees_route_context(model, context)
    var nk = trees_keys()
    # The held saved forest, or the fitted estimator's resident forest read
    # out once (its scores' forest; a side record, not the dictionary).
    var forest = iforest_export_forest(model, run)
    var unpacked = iforest_forest_words(forest, run)
    var words = unpacked[0]
    var bi = trees_builtins()
    var scalars: List[Float64] = [
        Float64(py=py_call(bi.float, py_item(fields, nk[].k_offset_))),
        Float64(Int(py=py_call(bi.int, py_item(fields, nk[].k_max_samples_)))),
        Float64(Int(py=py_call(bi.int, py_item(fields, nk[].k_n_features_in_)))),
    ]
    var trees = Int(py=words[5])
    var total = Int(py=words[6])
    var call = member_new(plan)
    try:
        _import(plan, call, MEMBER_BLOCK_SCALARS, MemberMatrix(Int(scalars.unsafe_ptr()), 1, 3, MEMBER_F64))
        _import(plan, call, MEMBER_BLOCK_FAMILY, MemberMatrix(Int(py=words[0]), 1, trees + 1, MEMBER_I32))
        _import(plan, call, MEMBER_BLOCK_FAMILY + 1, MemberMatrix(Int(py=words[1]), 1, total, MEMBER_I32))
        _import(plan, call, MEMBER_BLOCK_FAMILY + 2, MemberMatrix(Int(py=words[2]), 1, total, MEMBER_F32))
        _import(plan, call, MEMBER_BLOCK_FAMILY + 3, MemberMatrix(Int(py=words[3]), 1, total, MEMBER_I32))
        _import(plan, call, MEMBER_BLOCK_FAMILY + 4, MemberMatrix(Int(py=words[4]), 1, total, MEMBER_I32))
        call.block = 0
        call.x = MemberMatrix()
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = unpacked  # the five forest views stay alive through the imports above
    _ = forest
    _ = len(scalars)
    return call.state
