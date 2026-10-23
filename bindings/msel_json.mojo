# SPDX-License-Identifier: Apache-2.0
# Canonical split descriptor JSON. Mojo owns traversal; the connected CPython
# C encoder supplies JSON's exact ASCII string escaping, including surrogates.
from std.math import isfinite
from bindings.native_loaded_module import native_loaded_module
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get
from bindings.msel_inputs import msel_admit


def _text(value: PythonObject, context: PythonObject, depth: Int) raises -> String:
    if depth >= 1000:
        raise Error("RecursionError: maximum recursion depth exceeded while encoding a JSON object")
    var b = native_loaded_module("builtins")
    var kind = py_type(value)
    if value is None:
        return "null"
    if kind is b.str:
        return String(py_call(context["family_msel_json_string"], value))
    if kind is b.bool:
        return "true" if Bool(py=value) else "false"
    if kind is b.int:
        return String(value)
    if kind is b.float or kind is py_get(context, "float64"):
        # json.dumps accepts np.float64 as a native float subclass, but not
        # other NumPy scalar classes. Keep that exact historical boundary.
        var number = Float64(py=value)
        if not isfinite(number):
            raise Error("Out of range float values are not JSON compliant")
        return String(py_call(b.repr, PythonObject(number)))
    var result = String()
    if kind is b.list or kind is b.tuple:
        result += "["
        for k in range(len(value)):  # driver-loop: descriptor sequence traversal, no Python iterator protocol
            if k:
                result += ","
            result += _text(value[k], context, depth + 1)
        return result + "]"
    if kind is b.dict:
        var keys = py_call(b.list, value.keys())
        for k in range(len(keys)):  # driver-loop: keys admitted before native lexical sorting
            if py_type(keys[k]) is not b.str:
                raise Error("TypeError: split descriptor JSON requires exact string keys")
        keys = py_call(b.sorted, keys)
        result += "{"
        for k in range(len(keys)):  # driver-loop: sorted descriptor fields
            if k:
                result += ","
            result += String(py_call(context["family_msel_json_string"], keys[k]))
            result += ":" + _text(value[keys[k]], context, depth + 1)
        return result + "}"
    raise Error("TypeError: split descriptor value is not JSON serializable")


def msel_canonical(value: PythonObject, context: PythonObject) raises -> PythonObject:
    msel_admit(value, context)
    var b = native_loaded_module("builtins")
    if py_type(context["family_msel_json_string"]) is not py_type(b.len):
        raise Error("TypeError: split descriptor string encoding requires the connected C entry")
    return py_call(PythonObject(_text(value, context, 0)).encode, "ascii")
