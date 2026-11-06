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
# void PyErr_SetObject(PyObject *type, PyObject *value)
comptime _PyErr_SetObject = ExternalFunction[
    "PyErr_SetObject", def(PyObjectPtr, PyObjectPtr) thin abi("C") -> NoneType]
# int PyObject_GetBuffer(PyObject *exporter, Py_buffer *view, int flags)
# Callers pass the Py_buffer as the first word of a native `List[Int]` of at
# least 10 words (the stable-ABI layout; `buf` is word 0, `len` word 2), cast
# with `.unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()`.
comptime _PyObject_GetBuffer = ExternalFunction[
    "PyObject_GetBuffer", def(PyObjectPtr, UnsafePointer[Int, MutUntrackedOrigin], c_int) thin abi("C") -> c_int]
# void PyBuffer_Release(Py_buffer *view)  (same Py_buffer word storage)
comptime _PyBuffer_Release = ExternalFunction[
    "PyBuffer_Release", def(UnsafePointer[Int, MutUntrackedOrigin]) thin abi("C") -> NoneType]
# int PyList_SetItem(PyObject *list, Py_ssize_t index, PyObject *item)  (steals item)
comptime _PyList_SetItem = ExternalFunction[
    "PyList_SetItem", def(PyObjectPtr, Int, PyObjectPtr) thin abi("C") -> c_int]
# PyObject *PyObject_GetItem(PyObject *o, PyObject *key)  (new, NULL on error)
comptime _PyObject_GetItem = ExternalFunction[
    "PyObject_GetItem", def(PyObjectPtr, PyObjectPtr) thin abi("C") -> PyObjectPtr]
# int PyType_IsSubtype(PyTypeObject *a, PyTypeObject *b)  (lane perf3-prep-shared:
# asked by the perf2 PCA/decomp table, pca_native.mojo, tsvd/ipca/params metadata)
comptime _PyType_IsSubtype = ExternalFunction[
    "PyType_IsSubtype", def(PyObjectPtr, PyObjectPtr) thin abi("C") -> c_int]
# int PyCallable_Check(PyObject *o)
comptime _PyCallable_Check = ExternalFunction[
    "PyCallable_Check", def(PyObjectPtr) thin abi("C") -> c_int]
# Lane p3-core-cpython (2026-11-06): entries that per-item and per-call
# sites resolved by name on every use. Pointers to C strings and raw
# buffers travel as an address `Int` (same register class as a pointer under
# the C ABI on every supported target).
# void PyErr_Clear(void)
comptime _PyErr_Clear = ExternalFunction[
    "PyErr_Clear", def() thin abi("C") -> NoneType]
# PyObject *PyList_New(Py_ssize_t len)  (new)
comptime _PyList_New = ExternalFunction[
    "PyList_New", def(Int) thin abi("C") -> PyObjectPtr]
# int PyList_Append(PyObject *list, PyObject *item)  (does not steal)
comptime _PyList_Append = ExternalFunction[
    "PyList_Append", def(PyObjectPtr, PyObjectPtr) thin abi("C") -> c_int]
# PyObject *PyFloat_FromDouble(double v)  (new)
comptime _PyFloat_FromDouble = ExternalFunction[
    "PyFloat_FromDouble", def(Float64) thin abi("C") -> PyObjectPtr]
# PyObject *PyLong_FromLongLong(long long v)  (new)
comptime _PyLong_FromLongLong = ExternalFunction[
    "PyLong_FromLongLong", def(Int64) thin abi("C") -> PyObjectPtr]
# PyObject *PyLong_FromUnsignedLongLong(unsigned long long v)  (new)
comptime _PyLong_FromUnsignedLongLong = ExternalFunction[
    "PyLong_FromUnsignedLongLong", def(UInt64) thin abi("C") -> PyObjectPtr]
# PyObject *PySlice_New(PyObject *start, PyObject *stop, PyObject *step)  (new)
comptime _PySlice_New = ExternalFunction[
    "PySlice_New", def(PyObjectPtr, PyObjectPtr, PyObjectPtr) thin abi("C") -> PyObjectPtr]
# PyObject *PyMemoryView_FromObject(PyObject *obj)  (new)
comptime _PyMemoryView_FromObject = ExternalFunction[
    "PyMemoryView_FromObject", def(PyObjectPtr) thin abi("C") -> PyObjectPtr]
# PyObject *PyType_GenericAlloc(PyTypeObject *type, Py_ssize_t nitems)  (new)
comptime _PyType_GenericAlloc = ExternalFunction[
    "PyType_GenericAlloc", def(PyObjectPtr, Int) thin abi("C") -> PyObjectPtr]
# char *PyBytes_AsString(PyObject *o)  (address; 0 on error)
comptime _PyBytes_AsString = ExternalFunction[
    "PyBytes_AsString", def(PyObjectPtr) thin abi("C") -> Int]
# Py_UCS4 PyUnicode_ReadChar(PyObject *unicode, Py_ssize_t index)
comptime _PyUnicode_ReadChar = ExternalFunction[
    "PyUnicode_ReadChar", def(PyObjectPtr, Int) thin abi("C") -> UInt32]
# int Py_GetRecursionLimit(void)
comptime _Py_GetRecursionLimit = ExternalFunction[
    "Py_GetRecursionLimit", def() thin abi("C") -> c_int]
# int Py_EnterRecursiveCall(const char *where)  (address of a C string)
comptime _Py_EnterRecursiveCall = ExternalFunction[
    "Py_EnterRecursiveCall", def(Int) thin abi("C") -> c_int]
# void Py_LeaveRecursiveCall(void)
comptime _Py_LeaveRecursiveCall = ExternalFunction[
    "Py_LeaveRecursiveCall", def() thin abi("C") -> NoneType]
# int PyBuffer_ToContiguous(void *buf, const Py_buffer *src, Py_ssize_t len, char order)
# (the Py_buffer as the same word storage as PyObject_GetBuffer)
comptime _PyBuffer_ToContiguous = ExternalFunction[
    "PyBuffer_ToContiguous", def(UnsafePointer[UInt8, MutUntrackedOrigin], UnsafePointer[Int, MutUntrackedOrigin], Int, Int8) thin abi("C") -> c_int]


struct CPythonTable(Movable):
    var PyImport_GetModuleDict: _PyImport_GetModuleDict.type
    var PyDict_GetItem: _PyDict_GetItem.type
    var PyDict_Size: _PyDict_Size.type
    var PyType_GetFlags: _PyType_GetFlags.type
    var PyCFunction_GetSelf: _PyCFunction_GetSelf.type
    var PyObject_GenericGetDict: _PyObject_GenericGetDict.type
    var PyErr_Occurred: _PyErr_Occurred.type
    var PyErr_ExceptionMatches: _PyErr_ExceptionMatches.type
    var PyErr_SetObject: _PyErr_SetObject.type
    var PyObject_GetBuffer: _PyObject_GetBuffer.type
    var PyBuffer_Release: _PyBuffer_Release.type
    var PyList_SetItem: _PyList_SetItem.type
    var PyObject_GetItem: _PyObject_GetItem.type
    var PyType_IsSubtype: _PyType_IsSubtype.type
    var PyCallable_Check: _PyCallable_Check.type
    var PyErr_Clear: _PyErr_Clear.type
    var PyList_New: _PyList_New.type
    var PyList_Append: _PyList_Append.type
    var PyFloat_FromDouble: _PyFloat_FromDouble.type
    var PyLong_FromLongLong: _PyLong_FromLongLong.type
    var PyLong_FromUnsignedLongLong: _PyLong_FromUnsignedLongLong.type
    var PySlice_New: _PySlice_New.type
    var PyMemoryView_FromObject: _PyMemoryView_FromObject.type
    var PyType_GenericAlloc: _PyType_GenericAlloc.type
    var PyBytes_AsString: _PyBytes_AsString.type
    var PyUnicode_ReadChar: _PyUnicode_ReadChar.type
    var Py_GetRecursionLimit: _Py_GetRecursionLimit.type
    var Py_EnterRecursiveCall: _Py_EnterRecursiveCall.type
    var Py_LeaveRecursiveCall: _Py_LeaveRecursiveCall.type
    var PyBuffer_ToContiguous: _PyBuffer_ToContiguous.type

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
        self.PyErr_SetObject = _PyErr_SetObject.load(lib)
        self.PyObject_GetBuffer = _PyObject_GetBuffer.load(lib)
        self.PyBuffer_Release = _PyBuffer_Release.load(lib)
        self.PyList_SetItem = _PyList_SetItem.load(lib)
        self.PyObject_GetItem = _PyObject_GetItem.load(lib)
        self.PyType_IsSubtype = _PyType_IsSubtype.load(lib)
        self.PyCallable_Check = _PyCallable_Check.load(lib)
        self.PyErr_Clear = _PyErr_Clear.load(lib)
        self.PyList_New = _PyList_New.load(lib)
        self.PyList_Append = _PyList_Append.load(lib)
        self.PyFloat_FromDouble = _PyFloat_FromDouble.load(lib)
        self.PyLong_FromLongLong = _PyLong_FromLongLong.load(lib)
        self.PyLong_FromUnsignedLongLong = _PyLong_FromUnsignedLongLong.load(lib)
        self.PySlice_New = _PySlice_New.load(lib)
        self.PyMemoryView_FromObject = _PyMemoryView_FromObject.load(lib)
        self.PyType_GenericAlloc = _PyType_GenericAlloc.load(lib)
        self.PyBytes_AsString = _PyBytes_AsString.load(lib)
        self.PyUnicode_ReadChar = _PyUnicode_ReadChar.load(lib)
        self.Py_GetRecursionLimit = _Py_GetRecursionLimit.load(lib)
        self.Py_EnterRecursiveCall = _Py_EnterRecursiveCall.load(lib)
        self.Py_LeaveRecursiveCall = _Py_LeaveRecursiveCall.load(lib)
        self.PyBuffer_ToContiguous = _PyBuffer_ToContiguous.load(lib)


# The named global is shared by every Mojo extension in the process (all of
# them load the same interpreter library). Bump the name's version whenever a
# field is added, removed or reordered.
comptime CPYTHON_TABLE = _Global[
    StorageType=CPythonTable, name="MojolearnCPythonTableV4",
    init_fn=CPythonTable.__init__,
]


def cpython_table() raises -> UnsafePointer[CPythonTable, MutAnyOrigin]:
    """The process-wide table, filled at first use."""
    # Same typed form as x_cnn/device.mojo `_slots` (compiles on main).
    return CPYTHON_TABLE.get_or_create_ptr().unsafe_origin_cast[MutAnyOrigin]()
