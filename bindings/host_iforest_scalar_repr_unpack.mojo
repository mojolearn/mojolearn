# SPDX-License-Identifier: Apache-2.0
# Scalar ndarray parameter errors must not enter NumPy's Python arrayprint.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.math import isfinite, isnan
from bindings.py_balanced import py_call, py_type, py_get
from bindings.native_numpy_metadata_unpack import native_numpy_dtype


from bindings.host_iforest_scalar_repr_pack import (
    _if_repr_options,
)
from bindings.host_tree_archive_keys_unpack import host_tree_archive_keys
from bindings.py_balanced import py_item


def _if_repr_float(item: PythonObject, width: Int, options: PythonObject, context: PythonObject) raises -> String:
    var nk = host_tree_archive_keys()
    var value = Float64(py=item)
    var native_float = py_type(PythonObject(0.0))
    var formatter = py_item(native_float.__dict__, nk[].k___format__)
    if Int(py=py_item(options, nk[].k_legacy)) <= 113:
        # Legacy scalar ndarray formatting uses the unpacked scalar's repr.
        return String(py_call(py_type(PythonObject("")), PythonObject(value)))
    var sign = String(py_item(options, nk[].k_sign))
    if not isfinite(value):
        if isnan(value):
            return (sign if sign != "-" else String("")) + String(py_item(options, nk[].k_nanstr))
        return (String("-") if value < 0 else sign if sign != "-" else String("")) + String(py_item(options, nk[].k_infstr))
    var unique = String(py_item(options, nk[].k_floatmode)) != "fixed"
    var significant = 17
    if unique and value != 0:
        for digits in range(1, 18):  # driver-loop: shortest round-tripping native scalar representation
            var text = String(py_call(formatter, PythonObject(value), PythonObject("." + String(digits) + "g")))
            var candidate = Float64(text)
            var same = Float16(candidate) == Float16(value) if width == 2 else Float32(candidate) == Float32(value) if width == 4 else candidate == value
            if same:
                significant = digits
                break
    var numpy = py_get(py_item(context, nk[].k_modules), nk[].k_numpy)
    if py_type(numpy) is not py_item(context, nk[].k_host_iforest_module_type):
        raise Error("TypeError: IsolationForest parameter text requires native NumPy version metadata")
    var version = py_get(py_call(py_item(context, nk[].k_host_iforest_module_dict), numpy), nk[].k___version__)
    if py_type(version) is not py_type(PythonObject("")):
        raise Error("TypeError: IsolationForest parameter text requires native NumPy version text")
    var pieces = String(version).split(".")
    if len(pieces) < 2:
        raise Error("TypeError: IsolationForest parameter text requires major.minor NumPy version text")
    var release = Int(String(pieces[0])) * 100 + Int(String(pieces[1]))
    # These are NumPy's scalar display thresholds, not numerical dispatch.
    var cutoff = Float64(1e8)
    if release >= 203 and Int(py=py_item(options, nk[].k_legacy)) > 202:
        cutoff = 1e3 if width == 2 else 1e6 if width == 4 else 1e8
    var scientific = value != 0 and (abs(value) >= cutoff or (not Bool(py=py_item(options, nk[].k_suppress)) and abs(value) < Float64(1e-4)))
    var decimal_exponent = 0
    if value != 0 and not scientific:
        # Read the decimal exponent from the native formatter; do not introduce
        # a platform libm log10 dependency into the portable binding closure.
        var exponent_text = String(py_call(formatter, PythonObject(value), ".17e"))
        decimal_exponent = Int(String(exponent_text[byte=exponent_text.find("e") + 1:]))
    var needed = significant - 1 if scientific else max(0, significant - 1 - decimal_exponent) if value != 0 else 0
    var precision = needed
    if not unique or (String(py_item(options, nk[].k_floatmode)) != "unique" and py_item(options, nk[].k_precision) is not None):
        precision = min(Int(py=py_item(options, nk[].k_precision)), needed) if unique else Int(py=py_item(options, nk[].k_precision))
    var text = String(py_call(formatter, PythonObject(value), PythonObject(sign + "." + String(precision) + ("e" if scientific else "f"))))
    var exponent = text.find("e")
    var mantissa = String(text[byte=0:exponent]) if exponent >= 0 else text
    var tail = String(text[byte=exponent:]) if exponent >= 0 else String("")
    if unique:
        if mantissa.find(".") >= 0:
            while mantissa.endswith("0"):  # small-loop(text: trailing digits of one scalar parameter error word)
                var shorter = String(mantissa[byte=0:mantissa.byte_length() - 1])
                mantissa = shorter^
        else:
            mantissa += "."
    elif precision == 0:
        mantissa += "."
    return mantissa + tail


def host_iforest_scalar_repr(value: PythonObject, context: PythonObject) raises -> String:
    var nk = host_tree_archive_keys()
    # Caller already admitted the exact zero-dimensional numeric ndarray.
    var options = _if_repr_options(context)
    var kind = String(value.dtype.kind)
    var legacy = Int(py=py_item(options, nk[].k_legacy))
    var constructor = native_numpy_dtype(context)
    var longfloat = value.dtype.type is py_call(constructor, "g").type
    var custom = py_get(options, nk[].k_formatter)
    if custom is not None and legacy > 113:
        if py_type(custom) is not py_type(Python.dict()):
            raise Error("TypeError: IsolationForest print formatters require native configuration data")
        var keys = py_call(py_type(Python.list()), custom)
        for i in range(len(keys)):  # driver-loop: reject relevant callback formatters before any call
            if py_type(keys[i]) is not py_type(PythonObject("")):
                raise Error("TypeError: IsolationForest print formatter names must be native strings")
            var key = String(keys[i])
            if custom[keys[i]] is not None and (key == "all" or (kind == "f" and (key == "float_kind" or (key == "longfloat" if longfloat else key == "float"))) or ((kind == "i" or kind == "u") and (key == "int_kind" or key == "int")) or (kind == "b" and key == "bool")):
                raise Error("TypeError: IsolationForest parameter errors cannot invoke Python formatters; use native print options")
    var item = value.flat[0]
    var text = String(py_call(py_type(PythonObject("")), item))
    if kind == "f":
        if legacy <= 113 and longfloat:
            # ndarray.item keeps extended scalars as their native scalar type.
            ref cpy = Python().cpython()
            var result = cpy.lib.get_function[PyObjectPtr]("PyObject_Repr")(item._obj_ptr)
            if not result:
                raise cpy.unsafe_get_error()
            text = String(PythonObject(from_owned=result))
        else:
            text = _if_repr_float(item, Int(py=value.dtype.itemsize), options, context)
    elif (kind == "i" or kind == "u") and legacy > 113:
        # The integer sign option first applied to integer arrays in NumPy2.
        var numpy = py_get(py_item(context, nk[].k_modules), nk[].k_numpy)
        var version = py_get(py_call(py_item(context, nk[].k_host_iforest_module_dict), numpy), nk[].k___version__)
        if py_type(version) is not py_type(PythonObject("")):
            raise Error("TypeError: IsolationForest parameter text requires native version text")
        var parts = String(version).split(".")
        if Int(String(parts[0])) >= 2 and not text.startswith("-") and String(py_item(options, nk[].k_sign)) != "-":
            text = String(py_item(options, nk[].k_sign)) + text
    var implied = value.dtype.type is py_call(constructor, "i8").type or value.dtype.type is py_call(constructor, "f8").type or (kind == "b" and legacy > 113)
    var suffix = String("")
    if legacy > 210 and Int(py=py_item(options, nk[].k_threshold)) < 1:
        # NumPy 2.2 introduced the shape annotation for summarized arrays.
        # Older releases also use a large sentinel for the default legacy
        # option, so legacy alone does not identify this display contract.
        var numpy = py_get(py_item(context, nk[].k_modules), nk[].k_numpy)
        if py_type(numpy) is not py_item(context, nk[].k_host_iforest_module_type):
            raise Error("TypeError: IsolationForest parameter text requires native NumPy version metadata")
        var version = py_get(py_call(py_item(context, nk[].k_host_iforest_module_dict), numpy), nk[].k___version__)
        if py_type(version) is not py_type(PythonObject("")):
            raise Error("TypeError: IsolationForest parameter text requires native NumPy version text")
        var parts = String(version).split(".")
        if len(parts) < 2:
            raise Error("TypeError: IsolationForest parameter text requires major.minor NumPy version text")
        if Int(String(parts[0])) * 100 + Int(String(parts[1])) >= 202:
            suffix = "shape=()"
    if not Bool(py=value.dtype.isnative) or not implied:
        if suffix.byte_length() > 0:
            suffix += ", "
        suffix += "dtype="
        if Bool(py=value.dtype.isnative):
            # dtype.name is a native descriptor whose implementation calls
            # NumPy's Python _dtype._name_get. The admitted primitive dtype
            # already supplies its complete bit name as native metadata.
            if kind == "b":
                suffix += "bool"
            else:
                var stem = "float" if kind == "f" else "int" if kind == "i" else "uint"
                suffix += stem + String(Int(py=value.dtype.itemsize) * 8)
        else:
            # str(dtype) likewise enters NumPy's Python dtype formatter.
            # For these primitive nonnative dtypes its result is dtype.str.
            var code = value.dtype.str
            ref cpy = Python().cpython()
            var shown = cpy.lib.get_function[PyObjectPtr]("PyObject_Repr")(code._obj_ptr)
            if not shown:
                raise cpy.unsafe_get_error()
            suffix += String(PythonObject(from_owned=shown))
    if suffix.byte_length() == 0:
        return "array(" + text + ")"
    var spacer = String(" ")
    if legacy > 113 and 6 + text.byte_length() + 1 + suffix.byte_length() + 2 > Int(py=py_item(options, nk[].k_linewidth)):
        spacer = "\n      "
    return "array(" + text + "," + spacer + suffix + ")"
