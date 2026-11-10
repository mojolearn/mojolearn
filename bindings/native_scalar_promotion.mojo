# SPDX-License-Identifier: Apache-2.0
"""Capture the exact NumPy native promotion enum at a public boundary.

The getter is a native metadata read, never a ufunc or warning callback.
https://numpy.org/neps/nep-0050-scalar-promotion.html
https://numpy.org/doc/2.3/release/2.2.0-notes.html#nep-50-promotion-state-option-removed
"""
from std.python import PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem
from bindings.native_loaded_module import native_loaded_module
from bindings.native_numpy_metadata import native_numpy_module_fields
from bindings.native_class_admission import static_builtin_type


def native_scalar_weak_promotion(admission: PythonObject, label: String) raises -> Bool:
    var module_type = py_type(native_loaded_module("sys"))
    var numpy = py_get(admission["modules"], "numpy")
    if numpy is None or py_type(numpy) is not module_type:
        raise Error("TypeError: " + label + " requires native NumPy version metadata")
    var fields = native_numpy_module_fields(numpy)
    var version = py_get(fields, "__version__")
    if py_type(version) is not py_type(PythonObject("")):
        raise Error("TypeError: " + label + " requires native NumPy version metadata")
    var text = String(version)
    var major = 0
    var minor = 0
    var component = 0
    var digits = 0
    for codepoint in text.codepoints():  # driver-loop: bounded native version metadata
        var point = Int(codepoint)
        if point == 46:
            if digits == 0:
                raise Error("TypeError: " + label + " requires native NumPy version metadata")
            if component == 1:
                break
            component = 1
            digits = 0
        elif point >= 48 and point <= 57:
            if component == 0:
                major = min(100, major * 10 + point - 48)
            else:
                minor = min(100, minor * 10 + point - 48)
            digits += 1
        else:
            break
    if component == 0 or digits == 0:
        raise Error("TypeError: " + label + " requires native NumPy version metadata")
    if major > 2 or (major == 2 and minor >= 2):
        return True
    var owner = py_get(admission["modules"], "numpy._core._multiarray_umath")
    if owner is None:
        owner = py_get(admission["modules"], "numpy.core._multiarray_umath")
    if owner is None or py_type(owner) is not module_type:
        raise Error("TypeError: " + label + " NumPy promotion requires its native state getter")
    var getter = py_get(native_numpy_module_fields(owner), "_get_promotion_state")
    if py_type(getter) is not static_builtin_type("types", "BuiltinFunctionType"):
        raise Error("TypeError: " + label + " NumPy promotion requires its native state getter")
    if getter.__self__ is not owner or String(getter.__name__) != "_get_promotion_state":
        raise Error("TypeError: " + label + " NumPy promotion getter belongs to a different native entry")
    var retained = py_get(admission, "native_scalar_promotion_getter")
    if retained is not None and retained is not getter:
        raise Error("TypeError: " + label + " NumPy promotion getter was replaced")
    py_setitem(admission, "native_scalar_promotion_getter", getter)
    var mode = py_call(getter)
    if py_type(mode) is not py_type(PythonObject("")):
        raise Error("TypeError: " + label + " NumPy promotion state requires a native string")
    var name = String(mode)
    if name == "legacy":
        return False
    if name == "weak" or name == "weak_and_warn":
        return True
    raise Error("ValueError: " + label + " NumPy promotion state is unsupported")
