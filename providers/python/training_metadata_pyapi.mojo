# SPDX-License-Identifier: Apache-2.0
# Shared training boundary admission. Engines consume the resulting native
# scalars/descriptors; no training/batch loop calls these helpers back.
from std.math import isfinite
from std.python import PythonObject
from bindings.py_balanced import py_call, py_type
from bindings.native_loaded_module import native_loaded_module
from bindings.native_transport import _transport_json
from bindings.classical_pandas import classical_admit_value


def training_f32_scalar(value: PythonObject, name: String, context: PythonObject,
                        allow_string: Bool = False) raises -> Float64:
    classical_admit_value(value, context)
    var bi = native_loaded_module("builtins")
    var cls = py_type(value)
    if cls is bi.bool or String(py=cls.__name__) == "bool_":
        raise Error(name + " must be a finite float32 scalar")
    var number: Float64
    try:
        number = Float64(py=py_call(bi.float, value)) if allow_string else Float64(py=value.__getattr__("__float__")())
    except:
        raise Error(name + " must be a scalar")
    if not isfinite(number) or abs(number) > Float64(3.4028234663852886e38):
        raise Error(name + " must be finite float32")
    return Float64(Float32(number))


def training_json_walk(value: PythonObject, depth: Int, mut budget: Int, where: String) raises:
    var bi = native_loaded_module("builtins")
    budget -= 1
    if depth > 8 or budget < 0:
        raise Error(where + " is too complex")
    var cls = py_type(value)
    if cls is bi.dict:
        if len(value) > 128:
            raise Error(where + " object is too large")
        var keys = py_call(bi.list, value)
        for i in range(len(keys)):  # small-loop(at most128 bounded metadata keys)
            if py_type(keys[i]) is not bi.str or len(keys[i]) > 2048:
                raise Error(where + " keys must be bounded strings")
            training_json_walk(value[keys[i]], depth + 1, budget, where)
    elif cls is bi.list:
        if len(value) > 128:
            raise Error(where + " list is too large")
        for i in range(len(value)):  # small-loop(at most128 bounded metadata values)
            training_json_walk(value[i], depth + 1, budget, where)
    elif cls is bi.str:
        if len(value) > 2048:
            raise Error(where + " string is too large")
    elif cls is bi.int:
        if Int(py=py_call(value.bit_length)) > 4096:
            raise Error(where + " integer is too large")
    elif value is not None and cls is not bi.float and cls is not bi.bool:
        raise Error(where + " must contain JSON values")


def training_schedule_json(value: PythonObject, budget: Int, byte_limit: Int,
                           where: String, context: PythonObject) raises -> String:
    classical_admit_value(value, context)
    var bi = native_loaded_module("builtins")
    if py_type(value) is not bi.dict or len(value) == 0:
        raise Error(where + " must be a nonempty JSON object")
    var left = budget
    training_json_walk(value, 0, left, where)
    var text: String
    try:
        text = _transport_json(value)
    except:
        raise Error(where + " must be finite JSON")
    if text.byte_length() > byte_limit:
        raise Error(where + " exceeds " + String(byte_limit) + " bytes")
    return text
