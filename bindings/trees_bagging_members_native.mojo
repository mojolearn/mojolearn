# SPDX-License-Identifier: Apache-2.0
# Lazy Bagging member materialization, reusing native clone and block copies.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_fields, py_get, py_setitem, py_contains
from bindings.trees_boost_admission import boost_admit, boost_tree_admit
from bindings.host_inference_scalars import host_inference_scalar
from bindings.trees_public_boost import _clone_from, _seed, _sub_seed
from bindings.trees_public_forest import forest_get_params
from bindings.trees_public_forest import forest_refresh, forest_capture_mode, forest_kind
from bindings.trees_public_meta import _copy_block
from bindings.trees_public_auxiliary import _model_buffers
from bindings.native_array_storage import native_array_from_values
from bindings.classical_arrays import classical_wrap_array
from bindings.gp_label_snapshot_native import gp_label_list


def _member_like(base: PythonObject, first: PythonObject, member_seed: Int) raises -> PythonObject:
    """A member of `base`'s class holding `first`'s validated state (the
    first member's fields after its constructor checks, `forest_refresh`, and
    `forest_capture_mode`) with its own `random_state`: the state
    `_clone_from` + `forest_refresh` + `forest_capture_mode` give it, key for
    key in the same order. Only `random_state` differs between members of
    one template, and the only derived field that reads it is `_cfg`'s
    `random_state` entry (the RF and ET derivations store the int seed; a
    member seed is always an exact nonnegative int), which gets its own
    mapping."""
    var b = native_loaded_module("builtins")
    var member = py_call(b.object.__new__, py_type(base))
    var mf = py_fields(member)
    _ = py_call(mf.update, first)
    if py_contains(mf, "random_state"):
        var own_seed = PythonObject(member_seed)
        py_setitem(mf, "random_state", own_seed)
        var cfg = py_get(mf, "_cfg")
        if cfg is not None and py_type(cfg) is b.dict and py_contains(cfg, "random_state"):
            var own = py_call(b.dict, cfg)
            py_setitem(own, "random_state", own_seed)
            py_setitem(mf, "_cfg", own)
    return member


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
    # The forest's arrays and scalars, read once (lane perf-trees-owed,
    # 2026-11-06: re-read from the forest's fields for every tree), and the
    # members' kind once: every member is a clone of `base`, so of its class.
    var colid = ff["_colid"]
    var quesval = ff["_quesval"]
    var left = ff["_left_child"]
    var leaves = ff["_leaves"]
    var n_features = ff["n_features_in_"]
    var classified = py_contains(ff, "classes_")
    var classes = ff["classes_"] if classified else PythonObject(None)
    var n_classes = ff["n_classes_"] if classified else PythonObject(None)
    var one = PythonObject(1)
    var width = PythonObject(outputs)
    var kind = forest_kind(base, context)
    var template = forest_get_params(base, context) if count > 0 else PythonObject(None)
    # The constructor checks run once per call (lane ne2-b-members, N049):
    # every member is a clone of the one `base` with only its random_state
    # changed, so the first member is validated (`forest_refresh`, the full
    # constructor check and field rebuild) and captured, and every later
    # member copies that validated state with its own seed (`_member_like`).
    # Same fields, same values: nothing a member predicts changes.
    var first = PythonObject(None)
    var b = native_loaded_module("builtins")
    for t in range(count):  # driver-loop: one fitted member per native forest tree
        var lo = Int(py=offsets[t])
        var hi = Int(py=offsets[t + 1])
        var member: PythonObject
        if first is None:
            member = _clone_from(base, template, _sub_seed(seed, t), context)
            _ = forest_refresh(member, kind, context)
            _ = forest_capture_mode(py_fields(member), context)
            first = py_call(b.dict, py_fields(member))
        else:
            member = _member_like(base, first, _sub_seed(seed, t))
        var mf = py_fields(member)
        py_setitem(mf, "_offsets", classical_wrap_array(native_array_from_values("i", Python.list(0, hi - lo)), Python.tuple(2), "<i4", "C", context))
        py_setitem(mf, "_colid", _copy_block(colid, lo, hi, "<i4", context))
        py_setitem(mf, "_quesval", _copy_block(quesval, lo, hi, "<f4", context))
        py_setitem(mf, "_left_child", _copy_block(left, lo, hi, "<i4", context))
        py_setitem(mf, "_leaves", _copy_block(leaves, lo * outputs, hi * outputs, "<f4", context))
        py_setitem(mf, "_n_trees", one)
        py_setitem(mf, "_num_outputs", width)
        py_setitem(mf, "n_features_in_", n_features)
        if classified:
            py_setitem(mf, "classes_", classes)
            py_setitem(mf, "n_classes_", n_classes)
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
