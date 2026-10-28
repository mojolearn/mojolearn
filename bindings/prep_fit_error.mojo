# SPDX-License-Identifier: Apache-2.0
# Preserve the old fit wrapper's contextual refusals and explicit cause using
# native exception objects only. Never call user exception formatting hooks.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type
from bindings.native_error_class import error_without_class
from bindings.optimizer_public_words import PYTHON_ERROR_PENDING, PENDING_PYTHON_ERROR, stash_python_error
from bindings.prep_normalizer_lifecycle import _repr_admit


def _reference(address: Int) raises -> PythonObject:
    if address == 0:
        return PythonObject(None)
    ref cpy = Python().cpython()
    return PythonObject(from_owned=cpy.lib.get_function[PyObjectPtr]("Py_NewRef")(address))


def prep_fit_error(var error: Error, name: String, rows: Int, cols: Int) raises:
    var b = native_loaded_module("builtins")
    var text = String(error)
    var pending_error = text == PYTHON_ERROR_PENDING
    var category = b.ValueError
    var cause = PythonObject(None)
    if pending_error:
        var pending = PENDING_PYTHON_ERROR.get_or_create_ptr()
        category = _reference(pending[].words[0])
        if category is not b.ValueError and category is not b.ZeroDivisionError and category is not b.Exception:
            raise error^
        var value = _reference(pending[].words[1])
        # PyErr_SetString may retain an unnormalized native string; a native
        # entry may already have created an exact builtin exception instance.
        if py_type(value) is b.str:
            text = String(value)
            cause = py_call(category, value)
        elif py_type(value) is category:
            var args = py_call(b.BaseException.__dict__["args"].__get__, value, category)
            try:
                _repr_admit(args)
            except:
                # Preserve the pending exception without invoking its args'
                # repr hooks just to add small-fit context.
                raise error^
            text = String(py_call(b.BaseException.__str__, value))
            cause = value
        else:
            raise error^
    else:
        # This is prep_stateless_c's preexisting native error mapping: these
        # named classes pass through; all other native errors become ValueError.
        if text.startswith("TypeError:") or text.startswith("AttributeError:") or text.startswith("ImportError:") or text.startswith("RuntimeError:"):
            raise error^
        text = String(error_without_class(error, "ValueError"))
    for fault in ["device_context", "CUDA", "HIP", "hipError", "Metal", "MTL"]:  # small-loop(native device-fault markers)
        if fault in text:
            raise error^
    var replacement = String()
    if category is b.Exception:
        for marker in ["NaN", "Infinit", "infinit", "nonfinite", "non-finite", "not finite", "requires finite", "must be finite"]:  # small-loop(native nonfinite refusal markers)
            if marker in text:
                replacement = "Input contains NaN or infinity (" + name + ": " + text + ")"
                break
    if replacement.byte_length() == 0:
        if rows == 1:
            replacement = name + " cannot be fit on n_samples=1: " + text
        elif cols == 1 and category is b.ValueError:
            replacement = name + " cannot be fit on n_features=1: " + text
    if replacement.byte_length() == 0:
        raise error^
    if cause is None:
        cause = py_call(category, text)
    var result = py_call(b.ValueError, replacement)
    ref cpy = Python().cpython()
    if pending_error:
        var pending = PENDING_PYTHON_ERROR.get_or_create_ptr()
        if pending[].words[2] != 0:
            var trace = _reference(pending[].words[2])
            _ = cpy.lib.get_function[Int32]("PyException_SetTraceback")(cause._obj_ptr, trace._obj_ptr)
        # The old pending triple owns these references. Drop them only after
        # cause/result have independently retained the original exception.
        var words = pending[].words.copy()
        pending[].words = List[Int](length=3, fill=0)
        for i in range(3):  # small-loop(three owned CPython exception references)
            if words[i] != 0:
                cpy.lib.get_function[NoneType]("Py_DecRef")(words[i])
    # SetCause steals the new reference. SetObject borrows result, then the
    # existing ABI pending-error facility owns the completed exception.
    cpy.lib.get_function[NoneType]("PyException_SetCause")(result._obj_ptr, cpy.Py_NewRef(cause._obj_ptr))
    cpy.lib.get_function[NoneType]("PyErr_SetObject")(b.ValueError._obj_ptr, result._obj_ptr)
    var failure = stash_python_error()
    raise failure^
