# SPDX-License-Identifier: Apache-2.0
# Completed public CV metadata without Python repr/getter or estimator hooks.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_contains
from bindings.native_instance_item import native_instance_item
from bindings.classical_attributes import classical_connect_numpy
from bindings.native_array_view import native_array_ndarray_view
from bindings.arena_array_metadata import arena_array_metadata
from bindings.msel_common import msel_admit, msel_is_bool, msel_is_integral, msel_is_real
from bindings.msel_split import _splitter_fields, msel_flag_truth
from bindings.msel_estimator_native import msel_estimator_plan
from bindings.msel_scorer_native import msel_scorer_fields, msel_metric_plan


def _metadata_repr(value: PythonObject, context: PythonObject, depth: Int = 0) raises -> String:
    if depth > 64:
        raise Error("RecursionError: native splitter parameter nesting exceeds the supported bound")
    var b = native_loaded_module("builtins")
    var cls = py_type(value)
    if value is None or cls is b.str or cls is b.bytes or cls is b.bytearray or cls is b.bool or cls is b.int or cls is b.float or cls is b.complex or cls is b.range:
        return String(py_call(b.repr, value))
    if cls is b.list or cls is b.tuple:
        var text = String("[") if cls is b.list else String("(")
        for i in range(len(value)):  # driver-loop: passive parameter display, no element repr callbacks
            if i:
                text += ", "
            text += _metadata_repr(value[i], context, depth + 1)
        if cls is b.tuple and len(value) == 1:
            text += ","
        return text + (String("]") if cls is b.list else String(")"))
    if cls is b.dict:
        var pairs = py_call(b.list, value.items())
        var text = String("{")
        for i in range(len(pairs)):  # driver-loop: native dictionary insertion order without key comparisons
            if i:
                text += ", "
            text += _metadata_repr(pairs[i][0], context, depth + 1) + ": " + _metadata_repr(pairs[i][1], context, depth + 1)
        return text + "}"
    if cls is context["array_class"] or cls is py_get(context, "ndarray"):
        var metadata = arena_array_metadata(value, context["family_msel_resident_context"])
        var dtype = metadata["dtype"]
        if cls is not context["array_class"]:
            dtype = dtype.str  # exact immutable NumPy dtype descriptor
        return "Array(shape=" + String(py_call(b.repr, metadata["shape"])) + ", dtype=" + String(py_call(b.repr, dtype)) + ")"
    if msel_is_bool(value, context):
        return String(py_call(b.repr, PythonObject(Bool(py=value))))
    if msel_is_integral(value, context):
        return String(py_call(b.repr, py_call(b.int, value)))
    if msel_is_real(value, context):
        return String(py_call(b.repr, py_call(b.float, value)))
    # Already-admitted passive objects may have Python display implementations.
    # Keep them visible with a native type summary instead of invoking repr.
    return "<" + String(py_call(b.type.__dict__["__name__"].__get__, cls)) + ">"


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
    var result = PythonObject(None)
    if String(py_get(plan, "classes_kind", "stored")) == "knn":
        if py_get(fields, "_y_cols") is None:
            raise Error("AttributeError: classes_ is set by fit")
        result = py_get(fields, "_label_names")
        if result is None:
            result = native_instance_item(fields, PythonObject("_classes_list"))
            msel_admit(result, context)
            if py_type(result) is not b.list and py_type(result) is not b.tuple:
                raise Error("TypeError: native neighbor classes require stored class lists")
            if not msel_flag_truth(native_instance_item(fields, PythonObject("outputs_2d_")), context):
                result = result[0]
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
        result = native_instance_item(fields, PythonObject("classes_"))
    classical_connect_numpy(context)
    if py_type(result) is context["array_class"] and py_get(context, "ndarray") is not None:
        return native_array_ndarray_view(result, context)
    return result
