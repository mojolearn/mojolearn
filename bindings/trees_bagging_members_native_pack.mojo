# SPDX-License-Identifier: Apache-2.0
# Lazy Bagging member materialization, reusing native clone and block copies.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_fields, py_get, py_setitem, py_contains
from bindings.trees_boost_admission_unpack import boost_tree_admit
from bindings.trees_boost_admission_pack import boost_admit
from bindings.host_inference_scalars_door import host_inference_scalar
from bindings.trees_public_boost_door import _seed, _validated_clone
from xtrees.trees_boost_run import _sub_seed
from bindings.trees_public_forest_door import forest_get_params
from bindings.trees_public_forest_door import forest_refresh, forest_capture_mode, forest_kind
from bindings.trees_public_meta_door import _copy_block
from bindings.trees_public_auxiliary_door import _model_buffers
from bindings.native_array_storage_door import native_array_from_values
from bindings.classical_arrays_door import classical_wrap_array
from bindings.gp_label_snapshot_native_unpack import gp_label_list
from bindings.trees_keys_unpack import trees_keys
from bindings.py_balanced import py_item


def bagging_members(model: PythonObject, context: PythonObject) raises -> PythonObject:
    """`estimators_` of a bag fitted as one forest: its trees as members.
    `context` is the bag's run context (its device route), handed by the
    trees door: the members are built where the bag was fitted. (It was the
    connection context read from the package module by name on every call.)"""
    var nk = trees_keys()
    var fields = boost_admit(model, context)
    var existing = py_get(fields, nk[].k__estimators)
    if existing is not None:
        return existing
    var batched = py_get(fields, nk[].k__batched)
    if batched is None:
        raise Error("AttributeError: estimators_")
    if len(batched) != 2:
        raise Error("TypeError: native Bagging batched state requires its forest/member pair")
    var forest = batched[0]
    var base = batched[1]
    boost_tree_admit(base, context, True)
    host_inference_scalar(py_item(fields, nk[].k_random_state), context)
    var ff = py_fields(forest)
    var count = Int(py=py_item(ff, nk[].k__n_trees))
    var outputs = Int(py=py_item(ff, nk[].k__num_outputs))
    if count < 0 or outputs <= 0:
        raise Error("ValueError: invalid Bagging forest dimensions")
    var storage = _model_buffers(ff, count, outputs, context)
    var offsets = gp_label_list(py_item(ff, nk[].k__offsets), context)
    var seed = _seed(py_item(fields, nk[].k_random_state), context)
    var result = Python.list()
    # The forest's arrays and scalars, read once (lane perf-trees-owed,
    # 2026-11-06: re-read from the forest's fields for every tree), and the
    # members' kind once: every member is a clone of `base`, so of its class.
    var colid = py_item(ff, nk[].k__colid)
    var quesval = py_item(ff, nk[].k__quesval)
    var left = py_item(ff, nk[].k__left_child)
    var leaves = py_item(ff, nk[].k__leaves)
    var n_features = py_item(ff, nk[].k_n_features_in_)
    var classified = py_contains(ff, nk[].k_classes_)
    var classes = py_item(ff, nk[].k_classes_) if classified else PythonObject(None)
    var n_classes = py_item(ff, nk[].k_n_classes_) if classified else PythonObject(None)
    var one = PythonObject(1)
    var width = PythonObject(outputs)
    var kind = forest_kind(base, context)
    var template = forest_get_params(base, context) if count > 0 else PythonObject(None)
    if count > 0:
        # A member read back runs where the bag ran it (trees_public_boost
        # `boost_member_device`; the fit checked the base's own device).
        py_setitem(template, nk[].k_device, py_get(fields, nk[].k_device))
    # The constructor checks run once per call (lane ne2-b-members, N049):
    # every member is a clone of the one `base` with only its random_state
    # changed, so the first member is validated (`forest_refresh`, the full
    # constructor check and field rebuild, and `forest_capture_mode`) and
    # every later member copies that validated state with its own seed
    # (trees_public_boost `_validated_clone`, shared with the DART and
    # AdaBoost fit loops). Same fields, same values: nothing a member
    # predicts changes.
    var memo = Python.list()
    for t in range(count):  # driver-loop: one fitted member per native forest tree
        var lo = Int(py=offsets[t])
        var hi = Int(py=offsets[t + 1])
        var member = _validated_clone(base, template, _sub_seed(seed, t), kind, memo, context)
        var mf = py_fields(member)
        py_setitem(mf, nk[].k__offsets, classical_wrap_array(native_array_from_values("i", Python.list(0, hi - lo)), Python.tuple(2), "<i4", "C", context))
        py_setitem(mf, nk[].k__colid, _copy_block(colid, lo, hi, "<i4", context))
        py_setitem(mf, nk[].k__quesval, _copy_block(quesval, lo, hi, "<f4", context))
        py_setitem(mf, nk[].k__left_child, _copy_block(left, lo, hi, "<i4", context))
        py_setitem(mf, nk[].k__leaves, _copy_block(leaves, lo * outputs, hi * outputs, "<f4", context))
        py_setitem(mf, nk[].k__n_trees, one)
        py_setitem(mf, nk[].k__num_outputs, width)
        py_setitem(mf, nk[].k_n_features_in_, n_features)
        if classified:
            py_setitem(mf, nk[].k_classes_, classes)
            py_setitem(mf, nk[].k_n_classes_, n_classes)
        _ = py_call(result.append, member)
    _ = storage
    py_setitem(fields, nk[].k__estimators, result)
    return result


def bagging_members_set(model: PythonObject, value: PythonObject, context: PythonObject) raises -> PythonObject:
    # Assignment preserves identity; only actual consumption admits members.
    from providers.python.classical_family_admission_pyapi_unpack import classical_family_admit
    var fields = classical_family_admit(model)
    var nk = trees_keys()
    py_setitem(fields, nk[].k__estimators, value)
    py_setitem(fields, nk[].k__batched, PythonObject(None))
    return PythonObject(None)
