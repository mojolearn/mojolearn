# SPDX-License-Identifier: Apache-2.0
# The former _sk_validate.reraise envelope, without Python exception formatting
# hooks or an inherited Python decorator. Original kernel exceptions remain
# the explicit cause of native public shape/nonfinite refinements.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type
from bindings.optimizer_public_words import PYTHON_ERROR_PENDING, optimizer_python_error, stash_python_error


def _refinement(text: String, kind: Int, rows: Int, cols: Int, fitting: Bool,
                 name: String) -> Optional[String]:
    for marker in ["device_context", "CUDA", "HIP", "hipError", "Metal", "MTL"]:  # small-loop(device fault markers: existing public diagnostic exclusions)
        if marker in text:
            return Optional[String]()
    if kind == 0:
        for marker in ["NaN", "Infinit", "infinit", "nonfinite", "non-finite", "not finite", "requires finite", "must be finite"]:  # small-loop(nonfinite markers: existing public diagnostic spellings)
            if marker in text:
                var what = String("Input") if fitting else String("Input X")
                return Optional(what + " contains NaN or infinity (" + name + ": " + text + ")")
    if fitting and rows == 1 and kind <= 2:
        return Optional(name + " cannot be fit on n_samples=1: " + text)
    if fitting and cols == 1 and kind == 1:
        return Optional(name + " cannot be fit on n_features=1: " + text)
    return Optional[String]()


def _with_cause(message: String, original: PythonObject) raises -> Error:
    var b = native_loaded_module("builtins")
    var result = py_call(b.ValueError, PythonObject(message))
    ref cpy = Python().cpython()
    cpy.lib.get_function[NoneType]("PyException_SetCause")(result._obj_ptr, cpy.Py_NewRef(original._obj_ptr))
    cpy.lib.get_function[NoneType]("PyErr_SetObject")(b.ValueError._obj_ptr, result._obj_ptr)
    return stash_python_error()


def manifold_error(error: Error, rows: Int, cols: Int, fitting: Bool, name: String) raises -> Error:
    var b = native_loaded_module("builtins")
    var text = String(error)
    if text != PYTHON_ERROR_PENDING:
        var kind = 1  # the manifold C boundary maps an unprefixed Error to ValueError
        var cls = b.ValueError
        if text.startswith("ValueError: "):
            var stripped = String(text[byte=12:])
            text = stripped^
        elif text.startswith("ZeroDivisionError: "):
            kind = 2
            cls = b.ZeroDivisionError
            var stripped = String(text[byte=19:])
            text = stripped^
        else:
            for prefix in ["TypeError:", "RuntimeError:", "NotImplementedError:", "OverflowError:", "ImportError:", "AttributeError:", "NotFittedError:", "KeyError:", "IndexError:", "InterruptedError:"]:  # small-loop(non-validation exception categories)
                if text.startswith(prefix):
                    return Error(String(error))
        var refined = _refinement(text, kind, rows, cols, fitting, name)
        if not refined:
            return Error(String(error))
        return _with_cause(refined.value(), py_call(cls, PythonObject(text)))
    # Restore the owner-local pending triple, then fetch it into this scope.
    # Only exact builtin exception classes are formatted. Other exception
    # identities/arguments are restored unchanged, without invoking __str__.
    _ = optimizer_python_error(error)
    ref cpy = Python().cpython()
    var words = List[PyObjectPtr](length=3, fill=PyObjectPtr())
    var pointer = words.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    cpy.lib.get_function[NoneType]("PyErr_Fetch")(pointer, pointer + 1, pointer + 2)
    var kind = -1
    if words[0] == b.Exception._obj_ptr:
        kind = 0
    elif words[0] == b.ValueError._obj_ptr:
        kind = 1
    elif words[0] == b.ZeroDivisionError._obj_ptr:
        kind = 2
    if kind < 0:
        cpy.lib.get_function[NoneType]("PyErr_Restore")(words[0], words[1], words[2])
        return stash_python_error()
    var original_class = words[0]
    cpy.lib.get_function[NoneType]("PyErr_NormalizeException")(pointer, pointer + 1, pointer + 2)
    if not words[1] or words[0] != original_class:
        cpy.lib.get_function[NoneType]("PyErr_Restore")(words[0], words[1], words[2])
        return stash_python_error()
    var original = PythonObject(from_borrowed=words[1])
    var args = py_call(b.BaseException.__dict__["args"].__get__, original)
    if py_type(args) is not b.tuple or len(args) != 1 or py_type(args[0]) is not b.str:
        cpy.lib.get_function[NoneType]("PyErr_Restore")(words[0], words[1], words[2])
        return stash_python_error()
    var refined = _refinement(String(args[0]), kind, rows, cols, fitting, name)
    if not refined:
        cpy.lib.get_function[NoneType]("PyErr_Restore")(words[0], words[1], words[2])
        return stash_python_error()
    if words[2]:
        _ = cpy.lib.get_function[Int32]("PyException_SetTraceback")(original._obj_ptr, words[2])
    cpy.Py_DecRef(words[0])
    cpy.Py_DecRef(words[1])
    if words[2]:
        cpy.Py_DecRef(words[2])
    return _with_cause(refined.value(), original)
