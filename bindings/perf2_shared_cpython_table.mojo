# SPDX-License-Identifier: Apache-2.0
# CPython C-API functions of the shared per-call path, resolved ONCE per
# process (lane perf2-shared, 2026-11-06).
#
# `cpy.lib.get_function[R]("Name")` runs `dlsym` on every call. The shared
# helpers every public door goes through (the prepared registry, the identity
# snapshots, the public attribute hooks) made those lookups on each call.
# This table holds the functions those helpers use that neither the standard
# library's `CPython` handle nor `bindings/cpython_table.mojo` holds, typed
# exactly as the C prototypes and loaded from the interpreter library the
# standard library selected (`cpy.lib`). It is a separate table with its own
# global name so `bindings/cpython_table.mojo` keeps its layout.
#
# Every symbol below is exported by CPython 3.10 through 3.14 (the package
# floor and the newest supported interpreter), so loading cannot miss.
# Version-dependent symbols (PyWeakref_GetRef from 3.13, PyWeakref_GetObject
# removed in 3.15) are deliberately absent.
from std.ffi import _Global
from std.python import Python
from std.python._cpython import ExternalFunction, PyObjectPtr

# Py_ssize_t PyTuple_Size(PyObject *p)
comptime _PyTuple_Size = ExternalFunction[
    "PyTuple_Size", def(PyObjectPtr) thin abi("C") -> Int]
# Py_ssize_t PyList_Size(PyObject *list)
comptime _PyList_Size = ExternalFunction[
    "PyList_Size", def(PyObjectPtr) thin abi("C") -> Int]
# PyObject *PyList_GetItem(PyObject *list, Py_ssize_t index)  (borrowed)
comptime _PyList_GetItem = ExternalFunction[
    "PyList_GetItem", def(PyObjectPtr, Int) thin abi("C") -> PyObjectPtr]
# PyObject *PyTuple_GetItem(PyObject *p, Py_ssize_t pos)  (borrowed)
comptime _PyTuple_GetItem = ExternalFunction[
    "PyTuple_GetItem", def(PyObjectPtr, Int) thin abi("C") -> PyObjectPtr]
# long long PyLong_AsLongLong(PyObject *obj)
comptime _PyLong_AsLongLong = ExternalFunction[
    "PyLong_AsLongLong", def(PyObjectPtr) thin abi("C") -> Int64]
# PyObject *PyMethod_New(PyObject *func, PyObject *self)  (new)
comptime _PyMethod_New = ExternalFunction[
    "PyMethod_New", def(PyObjectPtr, PyObjectPtr) thin abi("C") -> PyObjectPtr]
# PyObject *PyWeakref_NewRef(PyObject *ob, PyObject *callback)  (new)
comptime _PyWeakref_NewRef = ExternalFunction[
    "PyWeakref_NewRef", def(PyObjectPtr, PyObjectPtr) thin abi("C") -> PyObjectPtr]
# PyObject *PyObject_CallNoArgs(PyObject *callable)  (new, NULL on error)
comptime _PyObject_CallNoArgs = ExternalFunction[
    "PyObject_CallNoArgs", def(PyObjectPtr) thin abi("C") -> PyObjectPtr]
# PyObject *PyObject_GenericGetAttr(PyObject *o, PyObject *name)  (new)
comptime _PyObject_GenericGetAttr = ExternalFunction[
    "PyObject_GenericGetAttr", def(PyObjectPtr, PyObjectPtr) thin abi("C") -> PyObjectPtr]
# PyFrameObject *PyEval_GetFrame(void)  (borrowed)
comptime _PyEval_GetFrame = ExternalFunction[
    "PyEval_GetFrame", def() thin abi("C") -> PyObjectPtr]
# PyCodeObject *PyFrame_GetCode(PyFrameObject *frame)  (new)
comptime _PyFrame_GetCode = ExternalFunction[
    "PyFrame_GetCode", def(PyObjectPtr) thin abi("C") -> PyObjectPtr]
# PyFrameObject *PyFrame_GetBack(PyFrameObject *frame)  (new, or NULL)
comptime _PyFrame_GetBack = ExternalFunction[
    "PyFrame_GetBack", def(PyObjectPtr) thin abi("C") -> PyObjectPtr]
# PyObject *PyFunction_GetCode(PyObject *op)  (borrowed)
comptime _PyFunction_GetCode = ExternalFunction[
    "PyFunction_GetCode", def(PyObjectPtr) thin abi("C") -> PyObjectPtr]


struct Perf2SharedCPythonTable(Movable):
    var PyTuple_Size: _PyTuple_Size.type
    var PyList_Size: _PyList_Size.type
    var PyList_GetItem: _PyList_GetItem.type
    var PyTuple_GetItem: _PyTuple_GetItem.type
    var PyLong_AsLongLong: _PyLong_AsLongLong.type
    var PyMethod_New: _PyMethod_New.type
    var PyWeakref_NewRef: _PyWeakref_NewRef.type
    var PyObject_CallNoArgs: _PyObject_CallNoArgs.type
    var PyObject_GenericGetAttr: _PyObject_GenericGetAttr.type
    var PyEval_GetFrame: _PyEval_GetFrame.type
    var PyFrame_GetCode: _PyFrame_GetCode.type
    var PyFrame_GetBack: _PyFrame_GetBack.type
    var PyFunction_GetCode: _PyFunction_GetCode.type

    def __init__(out self):
        ref cpy = Python().cpython()
        var lib = cpy.lib.borrow()
        self.PyTuple_Size = _PyTuple_Size.load(lib)
        self.PyList_Size = _PyList_Size.load(lib)
        self.PyList_GetItem = _PyList_GetItem.load(lib)
        self.PyTuple_GetItem = _PyTuple_GetItem.load(lib)
        self.PyLong_AsLongLong = _PyLong_AsLongLong.load(lib)
        self.PyMethod_New = _PyMethod_New.load(lib)
        self.PyWeakref_NewRef = _PyWeakref_NewRef.load(lib)
        self.PyObject_CallNoArgs = _PyObject_CallNoArgs.load(lib)
        self.PyObject_GenericGetAttr = _PyObject_GenericGetAttr.load(lib)
        self.PyEval_GetFrame = _PyEval_GetFrame.load(lib)
        self.PyFrame_GetCode = _PyFrame_GetCode.load(lib)
        self.PyFrame_GetBack = _PyFrame_GetBack.load(lib)
        self.PyFunction_GetCode = _PyFunction_GetCode.load(lib)


# Shared by every Mojo extension in the process (all load the same
# interpreter library). Bump the name's version whenever a field is added,
# removed or reordered.
comptime PERF2_SHARED_CPYTHON_TABLE = _Global[
    StorageType=Perf2SharedCPythonTable, name="MojolearnPerf2SharedCPythonTableV1",
    init_fn=Perf2SharedCPythonTable.__init__,
]


def perf2_cpython_table() raises -> UnsafePointer[Perf2SharedCPythonTable, MutAnyOrigin]:
    """The process-wide table, filled at first use."""
    # Same typed form as x_cnn/device.mojo `_slots` (compiles on main).
    return PERF2_SHARED_CPYTHON_TABLE.get_or_create_ptr().unsafe_origin_cast[MutAnyOrigin]()
