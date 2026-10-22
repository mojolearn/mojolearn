# SPDX-License-Identifier: Apache-2.0
# Scalar ndarray parameter errors must not enter NumPy's Python arrayprint.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.math import isfinite, isnan
from bindings.py_balanced import py_call, py_type, py_get
from bindings.native_numpy_metadata import native_numpy_dtype


def _if_repr_options(context: PythonObject) raises -> PythonObject:
    var modules = context["modules"]
    var module = py_get(modules, "numpy._core.arrayprint")
    if module is None:
        module = py_get(modules, "numpy.core.arrayprint")
    if py_type(module) is not context["host_iforest_module_type"]:
        raise Error("TypeError: IsolationForest parameter text requires native NumPy print metadata")
    var fields = py_call(context["host_iforest_module_dict"], module)
    var options = py_get(fields, "_format_options")
    if options is None:
        var variable = py_get(fields, "format_options")
        if py_type(variable) is not context["host_iforest_context_type"]:
            raise Error("TypeError: IsolationForest parameter text requires a native print ContextVar")
        options = py_call(context["host_iforest_context_get"], variable)
    if py_type(options) is not py_type(Python.dict()):
        raise Error("TypeError: IsolationForest parameter text requires native print option data")
    var keys = py_call(py_type(Python.list()), options)
    for i in range(len(keys)):  # driver-loop: validate print metadata keys before dictionary lookups
        if py_type(keys[i]) is not py_type(PythonObject("")):
            raise Error("TypeError: IsolationForest print option names must be native strings")
    var integers = Python.tuple("legacy", "linewidth", "threshold")
    for i in range(len(integers)):  # driver-loop: exact integer print options
        if py_type(options[integers[i]]) is not py_type(PythonObject(0)):
            raise Error("TypeError: IsolationForest print options require native integers")
    var strings = Python.tuple("sign", "floatmode", "nanstr", "infstr")
    for i in range(len(strings)):  # driver-loop: exact text print options
        if py_type(options[strings[i]]) is not py_type(PythonObject("")):
            raise Error("TypeError: IsolationForest print options require native strings")
    if py_type(options["suppress"]) is not py_type(PythonObject(True)) or (options["precision"] is not None and py_type(options["precision"]) is not py_type(PythonObject(0))):
        raise Error("TypeError: IsolationForest print precision requires native scalar metadata")
    if py_get(options, "override_repr") is not None:
        raise Error("TypeError: IsolationForest parameter errors cannot invoke Python print overrides; use native print options")
    return options


def _if_repr_float(item: PythonObject, width: Int, options: PythonObject, context: PythonObject) raises -> String:
    var value = Float64(py=item)
    var native_float = py_type(PythonObject(0.0))
    var formatter = native_float.__dict__["__format__"]
    if Int(py=options["legacy"]) <= 113:
        # Legacy scalar ndarray formatting uses the unpacked scalar's repr.
        return String(py_call(py_type(PythonObject("")), PythonObject(value)))
    var sign = String(options["sign"])
    if not isfinite(value):
        if isnan(value):
            return (sign if sign != "-" else String("")) + String(options["nanstr"])
        return (String("-") if value < 0 else sign if sign != "-" else String("")) + String(options["infstr"])
    var unique = String(options["floatmode"]) != "fixed"
    var significant = 17
    if unique and value != 0:
        for digits in range(1, 18):  # driver-loop: shortest round-tripping native scalar representation
            var text = String(py_call(formatter, PythonObject(value), PythonObject("." + String(digits) + "g")))
            var candidate = Float64(text)
            var same = Float16(candidate) == Float16(value) if width == 2 else Float32(candidate) == Float32(value) if width == 4 else candidate == value
            if same:
                significant = digits
                break
    var numpy = py_get(context["modules"], "numpy")
    if py_type(numpy) is not context["host_iforest_module_type"]:
        raise Error("TypeError: IsolationForest parameter text requires native NumPy version metadata")
    var version = py_get(py_call(context["host_iforest_module_dict"], numpy), "__version__")
    if py_type(version) is not py_type(PythonObject("")):
        raise Error("TypeError: IsolationForest parameter text requires native NumPy version text")
    var pieces = String(version).split(".")
    if len(pieces) < 2:
        raise Error("TypeError: IsolationForest parameter text requires major.minor NumPy version text")
    var release = Int(String(pieces[0])) * 100 + Int(String(pieces[1]))
    # These are NumPy's scalar display thresholds, not numerical dispatch.
    var cutoff = Float64(1e8)
    if release >= 203 and Int(py=options["legacy"]) > 202:
        cutoff = 1e3 if width == 2 else 1e6 if width == 4 else 1e8
    var scientific = value != 0 and (abs(value) >= cutoff or (not Bool(py=options["suppress"]) and abs(value) < Float64(1e-4)))
    var decimal_exponent = 0
    if value != 0 and not scientific:
        # Read the decimal exponent from the native formatter; do not introduce
        # a platform libm log10 dependency into the portable binding closure.
        var exponent_text = String(py_call(formatter, PythonObject(value), ".17e"))
        decimal_exponent = Int(String(exponent_text[byte=exponent_text.find("e") + 1:]))
    var needed = significant - 1 if scientific else max(0, significant - 1 - decimal_exponent) if value != 0 else 0
    var precision = needed
    if not unique or (String(options["floatmode"]) != "unique" and options["precision"] is not None):
        precision = min(Int(py=options["precision"]), needed) if unique else Int(py=options["precision"])
    var text = String(py_call(formatter, PythonObject(value), PythonObject(sign + "." + String(precision) + ("e" if scientific else "f"))))
    var exponent = text.find("e")
    var mantissa = String(text[byte=0:exponent]) if exponent >= 0 else text
    var tail = String(text[byte=exponent:]) if exponent >= 0 else String("")
    if unique:
        if mantissa.find(".") >= 0:
            while mantissa.endswith("0"):
                var shorter = String(mantissa[byte=0:mantissa.byte_length() - 1])
                mantissa = shorter^
        else:
            mantissa += "."
    elif precision == 0:
        mantissa += "."
    return mantissa + tail


def host_iforest_scalar_repr(value: PythonObject, context: PythonObject) raises -> String:
    # Caller already admitted the exact zero-dimensional numeric ndarray.
    var options = _if_repr_options(context)
    var kind = String(value.dtype.kind)
    var legacy = Int(py=options["legacy"])
    var constructor = native_numpy_dtype(context)
    var longfloat = value.dtype.type is py_call(constructor, "g").type
    var custom = py_get(options, "formatter")
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
        var numpy = py_get(context["modules"], "numpy")
        var version = py_get(py_call(context["host_iforest_module_dict"], numpy), "__version__")
        if py_type(version) is not py_type(PythonObject("")):
            raise Error("TypeError: IsolationForest parameter text requires native version text")
        var parts = String(version).split(".")
        if Int(String(parts[0])) >= 2 and not text.startswith("-") and String(options["sign"]) != "-":
            text = String(options["sign"]) + text
    var implied = value.dtype.type is py_call(constructor, "i8").type or value.dtype.type is py_call(constructor, "f8").type or (kind == "b" and legacy > 113)
    var suffix = String("")
    if legacy > 210 and Int(py=options["threshold"]) < 1:
        suffix = "shape=()"
    if not Bool(py=value.dtype.isnative) or not implied:
        if suffix.byte_length() > 0:
            suffix += ", "
        suffix += "dtype="
        if Bool(py=value.dtype.isnative):
            suffix += String(value.dtype.name)
        else:
            var code = py_call(py_type(PythonObject("")), value.dtype)
            ref cpy = Python().cpython()
            var shown = cpy.lib.get_function[PyObjectPtr]("PyObject_Repr")(code._obj_ptr)
            if not shown:
                raise cpy.unsafe_get_error()
            suffix += String(PythonObject(from_owned=shown))
    if suffix.byte_length() == 0:
        return "array(" + text + ")"
    var spacer = String(" ")
    if legacy > 113 and 6 + text.byte_length() + 1 + suffix.byte_length() + 2 > Int(py=options["linewidth"]):
        spacer = "\n      "
    return "array(" + text + "," + spacer + suffix + ")"
