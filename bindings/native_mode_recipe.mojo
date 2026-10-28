# SPDX-License-Identifier: Apache-2.0
# Fixed import-connected native operation recipes used by metadata owners.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_type
from bindings.optimizer_public_words import stash_python_error


def mode_recipe_call(entry: PythonObject, args: PythonObject) raises -> PythonObject:
    if py_type(entry) is not native_loaded_module("types").BuiltinFunctionType:
        raise Error("TypeError: metadata operation requires its connected native entry")
    ref cpy = Python().cpython()
    var result = cpy.PyObject_Call(entry._obj_ptr, args._obj_ptr, PyObjectPtr())
    if not result:
        var failure = stash_python_error()
        raise failure^
    return PythonObject(from_owned=result)
