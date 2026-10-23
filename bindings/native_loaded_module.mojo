# SPDX-License-Identifier: Apache-2.0
# Resolve import-time module connections without executing Python import hooks.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr


def native_loaded_module(var name: String) raises -> PythonObject:
    # PyImport_GetModuleDict and PyDict_GetItemString read the interpreter's
    # loaded-module registry. They do not invoke __import__, finders, loaders
    # or module attribute hooks. Required modules belong to package setup;
    # product execution cannot import an absent dependency as a fallback.
    ref cpy = Python().cpython()
    var modules = cpy.lib.get_function[PyObjectPtr]("PyImport_GetModuleDict")()
    if not modules:
        raise Error("ImportError: native execution has no loaded-module registry")
    var found = cpy.lib.get_function[PyObjectPtr]("PyDict_GetItemString")(modules, name.as_c_string_slice().unsafe_ptr())
    _ = name  # keep the C string storage alive through the registry lookup
    if not found:
        raise Error("ImportError: native execution requires an import-time connection for " + name)
    var module = PythonObject(from_owned=cpy.Py_NewRef(found))
    # PyModule_GetDict checks the real native module layout without running
    # an object's __dict__ descriptor; reject substituted arbitrary objects.
    var fields = cpy.lib.get_function[PyObjectPtr]("PyModule_GetDict")(module._obj_ptr)
    if not fields:
        cpy.lib.get_function[NoneType]("PyErr_Clear")()
        raise Error("TypeError: native module connection is not a native module: " + name)
    var cls = PythonObject(from_owned=cpy.PyObject_Type(module._obj_ptr))
    var flags = cpy.lib.get_function[UInt64]("PyType_GetFlags")(cls._obj_ptr)
    # ModuleType is CPython's static native type. A heap subclass may add
    # Python attribute hooks; module contents alone do not admit that class.
    if (flags & UInt64(1 << 9)) != 0:
        raise Error("TypeError: native module connection cannot use a module subclass: " + name)
    return module
