# SPDX-License-Identifier: Apache-2.0
# Lazy Bagging member materialization, reusing native clone and block copies.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_fields, py_get, py_setitem, py_contains
from bindings.trees_boost_admission import boost_admit, boost_tree_admit
from bindings.host_inference_scalars import host_inference_scalar
from bindings.trees_public_boost import _clone, _seed, _sub_seed
from bindings.trees_public_forest import forest_refresh, forest_capture_mode, forest_kind
from bindings.trees_public_meta import _copy_block
from bindings.trees_public_auxiliary import _model_buffers
from bindings.native_array_storage import native_array_from_values
from bindings.classical_arrays import classical_wrap_array
from bindings.gp_label_snapshot_native import gp_label_list


def bagging_members(model: PythonObject, slice_only: Bool = False) raises -> PythonObject:
    var context = py_fields(native_loaded_module("mojolearn._expansion_trees"))["_TREES_CONTEXT"]
    var fields = boost_admit(model, context)
    var existing = py_get(fields, "_estimators")
    if existing is not None and not slice_only:
        return existing
    var batched = py_get(fields, "_batched")
    if batched is None:
        raise Error("AttributeError: estimators_")
    if len(batched) != 2:
        raise Error("TypeError: native Bagging batched state requires its forest/member pair")
    var forest = batched[0]
    var base = batched[1]
    boost_tree_admit(base, context, True)
    host_inference_scalar(fields["random_state"], context)
    var ff = py_fields(forest)
    var count = Int(py=ff["_n_trees"])
    var outputs = Int(py=ff["_num_outputs"])
    if count < 0 or outputs <= 0:
        raise Error("ValueError: invalid Bagging forest dimensions")
    var storage = _model_buffers(ff, count, outputs, context)
    var offsets = gp_label_list(ff["_offsets"], context)
    var seed = _seed(fields["random_state"], context)
    var result = Python.list()
    for t in range(count):  # driver-loop: one fitted member per native forest tree
        var lo = Int(py=offsets[t])
        var hi = Int(py=offsets[t + 1])
        var member = _clone(base, _sub_seed(seed, t), context)
        var mf = py_fields(member)
        _ = forest_refresh(member, forest_kind(member, context), context)
        _ = forest_capture_mode(mf, context)
        py_setitem(mf, "_offsets", classical_wrap_array(native_array_from_values("i", Python.list(0, hi - lo)), Python.tuple(2), "<i4", "C", context))
        py_setitem(mf, "_colid", _copy_block(ff["_colid"], lo, hi, "<i4", context))
        py_setitem(mf, "_quesval", _copy_block(ff["_quesval"], lo, hi, "<f4", context))
        py_setitem(mf, "_left_child", _copy_block(ff["_left_child"], lo, hi, "<i4", context))
        py_setitem(mf, "_leaves", _copy_block(ff["_leaves"], lo * outputs, hi * outputs, "<f4", context))
        py_setitem(mf, "_n_trees", PythonObject(1))
        py_setitem(mf, "_num_outputs", PythonObject(outputs))
        py_setitem(mf, "n_features_in_", ff["n_features_in_"])
        if py_contains(ff, "classes_"):
            py_setitem(mf, "classes_", ff["classes_"])
            py_setitem(mf, "n_classes_", ff["n_classes_"])
        _ = py_call(result.append, member)
    _ = storage
    if not slice_only:
        py_setitem(fields, "_estimators", result)
    return result


def bagging_members_set(model: PythonObject, value: PythonObject, context: PythonObject) raises -> PythonObject:
    # Assignment preserves identity; only actual consumption admits members.
    from providers.python.classical_family_admission_pyapi import classical_family_admit
    var fields = classical_family_admit(model)
    py_setitem(fields, "_estimators", value)
    py_setitem(fields, "_batched", PythonObject(None))
    return PythonObject(None)
