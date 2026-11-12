# SPDX-License-Identifier: Apache-2.0
# Native warning delivery for sequence array-protocol casts.
# The PACK half of the former `bindings/sequence_array_warnings.mojo` (fence-flip split, 2026-11-12);
# the UNPACK half is `bindings/sequence_array_warnings_unpack.mojo`.
from bindings.native_loaded_module import native_loaded_module
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_fields
from bindings.classical_warnings_unpack import _warning_fields, _warning_class_fields, _warning_match, _warning_write
from bindings.optimizer_public_words_unpack import stash_python_error
from core.sequence_array_cast import SEQ_CAST_OVERFLOW, SEQ_CAST_UNDERFLOW, SEQ_CAST_INVALID


def sequence_array_errstate(context: PythonObject) raises -> PythonObject:
    var module = py_get(context["modules"], "numpy._core._multiarray_umath")
    if module is None:
        module = py_get(context["modules"], "numpy.core._multiarray_umath")
    if module is None or py_type(module) is not context["module_type"]:
        raise Error("TypeError: array output requires NumPy's native error-state metadata")
    var entry = py_get(py_fields(module), "_get_extobj_dict")
    var legacy = entry is None
    if legacy:
        entry = py_get(py_fields(module), "geterrobj")
    var warning = context["sequence_array_warnings"]
    var name = String("geterrobj") if legacy else String("_get_extobj_dict")
    if entry is None or py_type(entry) is not warning["warning_native_function"] or String(entry.__name__) != name or entry.__self__ is not module:
        raise Error("TypeError: NumPy error-state metadata must use its native entry")
    var stored = py_call(entry)
    if not legacy:
        return _warning_fields(stored)
    # NumPy 1.26's native getter returns [buffer_size, error_mask, callback].
    # Each base-8 digit is a documented error action, in divide/over/under/
    # invalid order. Decode the metadata here instead of calling np.geterr,
    # whose wrapper executes Python. See NumPy 1.26 geterrobj API contract.
    var b = native_loaded_module("builtins")
    if py_type(stored) is not b.list or len(stored) != 3 or py_type(stored[1]) is not b.int:
        raise Error("TypeError: NumPy error-state metadata requires a native mask")
    var mask = Int(py=stored[1])
    var result = Python.dict()
    var actions: List[String] = ["ignore", "warn", "raise", "call", "print", "log"]
    var names: List[String] = ["over", "under", "invalid"]
    for i in range(3):  # driver-loop: decode the three native cast-status mask fields
        var action = (mask >> (3 * (i + 1))) & 7
        if action >= len(actions):
            raise Error("TypeError: NumPy error-state mask has an unsupported action")
        py_setitem(result, PythonObject(names[i]), PythonObject(actions[action]))
    py_setitem(result, "call", stored[2])
    return result


def sequence_array_category(kind: String) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var category = py_get(py_fields(b), PythonObject(kind))
    var flags = b.type.__dict__["__flags__"]
    # Validate before PyErr_SetString or constructing a warning message: both
    # can instantiate the exception class. Replaced Python classes/functions
    # therefore must be refused even though the call site itself is native.
    if category is None or py_type(category) is not b.type:
        raise Error("TypeError: array warnings require native built-in exception classes")
    if (Int(py=py_call(flags.__get__, category)) & 512) != 0:
        raise Error("TypeError: array warnings require native built-in exception classes")
    if String(py_call(b.type.__getattribute__, category, "__name__")) != kind or String(py_call(b.type.__getattribute__, category, "__module__")) != "builtins":
        raise Error("TypeError: array warnings require native built-in exception identities")
    return category
