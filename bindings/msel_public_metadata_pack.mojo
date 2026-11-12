# SPDX-License-Identifier: Apache-2.0
# Completed public CV metadata without Python repr/getter or estimator hooks.
# The PACK half of the former `bindings/msel_public_metadata.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `bindings/msel_public_metadata_unpack.mojo`.
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_contains
from bindings.native_instance_item_pack import native_instance_item
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.native_array_view_pack import native_array_ndarray_view
from bindings.arena_array_metadata_pack import arena_array_metadata
from bindings.msel_common_door import msel_admit, msel_is_bool, msel_is_integral, msel_is_real
from bindings.msel_split_door import _splitter_fields, msel_flag_truth
from bindings.msel_estimator_native_door import msel_estimator_plan
from bindings.msel_scorer_native_door import msel_scorer_fields, msel_metric_plan
from bindings.py_optional import py_or_none, py_present


from bindings.msel_public_metadata_unpack import (
    _metadata_repr,
)


def msel_splitter_repr(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var fields = _splitter_fields(model, context)
    var names = py_call(b.sorted, fields)
    var text = String(py_call(b.type.__dict__["__name__"].__get__, py_type(model))) + "("
    var first = True
    for i in range(len(names)):  # driver-loop: sorted public configuration, matching get_params's public field selection
        var name = String(names[i])
        if name.startswith("_"):
            continue
        var value = native_instance_item(fields, names[i])
        msel_admit(value, context)
        if not first:
            text += ", "
        first = False
        text += name + "=" + _metadata_repr(value, context)
    return PythonObject(text + ")")


def msel_scorer_repr(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = msel_scorer_fields(model, context)
    var plan = msel_metric_plan(native_instance_item(fields, PythonObject("_score_func")), context)
    return PythonObject("make_scorer(" + String(plan["name"]) + ")")


def msel_estimator_classes(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var plan = msel_estimator_plan(model, context)
    var fields = py_call(plan["fields_get"], model, py_type(model))
    # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
    var result = Optional[PythonObject]()
    if String(py_get(plan, "classes_kind", "stored")) == "knn":
        if py_get(fields, "_y_cols") is None:
            raise Error("AttributeError: classes_ is set by fit")
        result = py_present(py_get(fields, "_label_names"))
        if not result:
            result = py_present(native_instance_item(fields, PythonObject("_classes_list")))
            msel_admit(result.value(), context)
            if py_type(result.value()) is not b.list and py_type(result.value()) is not b.tuple:
                raise Error("TypeError: native neighbor classes require stored class lists")
            if not msel_flag_truth(native_instance_item(fields, PythonObject("outputs_2d_")), context):
                result = py_present(result.value()[0])
    else:
        # Every class namespace's classes_ identity is pinned in the estimator
        # plan. A new property cannot bypass the dictionary or invoke a getter.
        var mro = plan["mro"]
        for i in range(len(mro)):  # driver-loop: respect data-descriptor precedence without executing descriptors
            var namespace = py_call(b.type.__dict__["__dict__"].__get__, mro[i])
            var member = py_get(namespace, "classes_")
            if member is not None:
                var member_mro = py_call(b.type.__dict__["__mro__"].__get__, py_type(member))
                for j in range(len(member_mro)):  # driver-loop: descriptor protocol metadata only
                    var slots = py_call(b.type.__dict__["__dict__"].__get__, member_mro[j])
                    if py_contains(slots, "__set__") or py_contains(slots, "__delete__"):
                        raise Error("TypeError: estimator class labels require a connected native property")
                break
        if not py_contains(fields, "classes_"):
            raise Error("AttributeError: fitted estimator has no classes_")
        result = py_present(native_instance_item(fields, PythonObject("classes_")))
    classical_connect_numpy(context)
    var labels = py_or_none(result)
    if py_type(labels) is context["array_class"] and py_get(context, "ndarray") is not None:
        return native_array_ndarray_view(labels, context)
    return labels
