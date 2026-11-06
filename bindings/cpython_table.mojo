# SPDX-License-Identifier: Apache-2.0
# CPython C-API functions resolved ONCE per process (lane perf-cpython-table,
# 2026-11-06).
#
# `cpy.lib.get_function[R]("Name")` runs `dlsym` on every call: the dynamic
# loader walks the loaded images by name (about 50 host bindings or one merged
# file, plus NumPy, on a CPU install) for each CPython call a native door makes.
# The standard library's `CPython` handle already holds typed pointers for the
# functions it wraps (`cpy.PyDict_Next`, `cpy.PyModule_GetDict`,
# `cpy.PyErr_Clear`, `cpy.PyDict_GetItemWithError`, `cpy.PyObject_GetIter`,
# `cpy.PyIter_Next`, ...): callers use those directly. This table holds the
# stable-API functions it does not wrap, typed exactly as the C prototypes,
# loaded from the same interpreter library the standard library selected
# (`cpy.lib`), so no global symbol linkage is assumed.
#
# Every symbol below is part of CPython's stable C API since before 3.10, the
# package floor, so loading cannot miss on a supported interpreter.
from std.ffi import _Global, c_int
from std.python import Python
from std.python._cpython import ExternalFunction, PyObjectPtr

# PyObject *PyImport_GetModuleDict(void)  (borrowed)
comptime _PyImport_GetModuleDict = ExternalFunction[
    "PyImport_GetModuleDict", def() thin abi("C") -> PyObjectPtr]
# PyObject *PyDict_GetItem(PyObject *p, PyObject *key)  (borrowed, no error)
comptime _PyDict_GetItem = ExternalFunction[
    "PyDict_GetItem", def(PyObjectPtr, PyObjectPtr) thin abi("C") -> PyObjectPtr]
# Py_ssize_t PyDict_Size(PyObject *p)
comptime _PyDict_Size = ExternalFunction[
    "PyDict_Size", def(PyObjectPtr) thin abi("C") -> Int]
# unsigned long PyType_GetFlags(PyTypeObject *type)  (LP64: 64-bit)
comptime _PyType_GetFlags = ExternalFunction[
    "PyType_GetFlags", def(PyObjectPtr) thin abi("C") -> UInt64]
# PyObject *PyCFunction_GetSelf(PyObject *op)  (borrowed)
comptime _PyCFunction_GetSelf = ExternalFunction[
    "PyCFunction_GetSelf", def(PyObjectPtr) thin abi("C") -> PyObjectPtr]
# PyObject *PyObject_GenericGetDict(PyObject *o, void *context)  (new)
comptime _PyObject_GenericGetDict = ExternalFunction[
    "PyObject_GenericGetDict", def(PyObjectPtr, PyObjectPtr) thin abi("C") -> PyObjectPtr]
# PyObject *PyErr_Occurred(void)  (borrowed exception class, or NULL)
comptime _PyErr_Occurred = ExternalFunction[
    "PyErr_Occurred", def() thin abi("C") -> PyObjectPtr]
# int PyErr_ExceptionMatches(PyObject *exc)
comptime _PyErr_ExceptionMatches = ExternalFunction[
    "PyErr_ExceptionMatches", def(PyObjectPtr) thin abi("C") -> c_int]


struct CPythonTable(Movable):
    var PyImport_GetModuleDict: _PyImport_GetModuleDict.type
    var PyDict_GetItem: _PyDict_GetItem.type
    var PyDict_Size: _PyDict_Size.type
    var PyType_GetFlags: _PyType_GetFlags.type
    var PyCFunction_GetSelf: _PyCFunction_GetSelf.type
    var PyObject_GenericGetDict: _PyObject_GenericGetDict.type
    var PyErr_Occurred: _PyErr_Occurred.type
    var PyErr_ExceptionMatches: _PyErr_ExceptionMatches.type

    def __init__(out self):
        ref cpy = Python().cpython()
        var lib = cpy.lib.borrow()
        self.PyImport_GetModuleDict = _PyImport_GetModuleDict.load(lib)
        self.PyDict_GetItem = _PyDict_GetItem.load(lib)
        self.PyDict_Size = _PyDict_Size.load(lib)
        self.PyType_GetFlags = _PyType_GetFlags.load(lib)
        self.PyCFunction_GetSelf = _PyCFunction_GetSelf.load(lib)
        self.PyObject_GenericGetDict = _PyObject_GenericGetDict.load(lib)
        self.PyErr_Occurred = _PyErr_Occurred.load(lib)
        self.PyErr_ExceptionMatches = _PyErr_ExceptionMatches.load(lib)


# The named global is shared by every Mojo extension in the process (all of
# them load the same interpreter library). Bump the name's version whenever a
# field is added, removed or reordered.
comptime CPYTHON_TABLE = _Global[
    StorageType=CPythonTable, name="MojolearnCPythonTableV1",
    init_fn=CPythonTable.__init__,
]


def cpython_table() raises -> Pointer[CPythonTable, MutUntrackedOrigin]:
    """The process-wide table, filled at first use."""
    return CPYTHON_TABLE.get_or_create_ptr()
