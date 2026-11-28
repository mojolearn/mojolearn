# SPDX-License-Identifier: Apache-2.0
# Completed public CV metadata without Python repr/getter or estimator hooks.
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_contains
from bindings.arena_array_metadata_pack import arena_array_metadata
from bindings.msel_common_door import msel_admit, msel_is_bool, msel_is_integral, msel_is_real


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
