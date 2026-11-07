# SPDX-License-Identifier: Apache-2.0
# THE CPython function table of Mojolearn: C-API functions resolved ONCE per
# process (lane perf-cpython-table, 2026-11-06; made the only table by lane
# ne1-e-table, 2026-11-06, absorbing perf2_shared_cpython_table.mojo,
# p3_lm_tokenizer_cpython.mojo, the 18 entries of p3_labels_metrics_native.mojo,
# ClusterCPython of cluster_side_records.mojo and ManifoldCalls of
# manifold_storage_native.mojo).
#
# `cpy.lib.get_function[R]("Name")` runs `dlsym` on every call: the dynamic
# loader walks the loaded images by name (about 50 host bindings or one merged
# file, plus NumPy, on a CPU install) for each CPython call a native door makes.
# This table holds the functions typed exactly as the C prototypes, loaded
# from the same interpreter library the standard library selected
# (`cpy.lib`), so no global symbol linkage is assumed.
#
# THE ONE FORM (every caller, every lane):
#   from bindings.cpython_table import cpython_table
#   var table = cpython_table()            # once per function, outside loops
#   var fields = table[].PyModule_GetDict(module._obj_ptr)
# - A C-API function the tree needs that is not here is ADDED here (field,
#   comptime prototype, load line, global name version bump); no lane builds a
#   second table, calls `get_function[...]("Py...")` on a per-call or per-item
#   path, or registers another `_Global` of CPython symbols.
# - `PyModule_GetDict`, `PyObject_GetIter` and `PyIter_Next` (the 26 calls of
#   plan item A whose `cpy.<Name>` method form was doubtful) go through this
#   table only: the `ExternalFunction` form is the one the 0.8.66 candidate
#   compiled for all 78 entries of the former tables, so it carries no open
#   question.
# - The standard library handle (`ref cpy = Python().cpython()`) stays for
#   reference counts (`Py_NewRef`, `Py_DecRef`, `Py_IncRef`), `PyObject_Type`,
#   `PyObject_Call`, the tuple builders and `unsafe_get_error`, the calls main
#   already compiled in that form.
# - Pointers to C strings and raw buffers travel as an address `Int` (same
#   register class as a pointer under the C ABI on every supported target);
#   callers pass `Int(ptr)`.
#
# Every symbol below is exported by CPython 3.10 through 3.14 (the package
# floor and the newest supported interpreter), so loading cannot miss.
# Version-dependent symbols (PyWeakref_GetRef from 3.13, PyWeakref_GetObject
# removed in 3.15) are deliberately absent.
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
# PyObject *PyUnicode_DecodeUTF8(const char *s, Py_ssize_t size, const char *errors)
# (s and errors as addresses; errors 0 = strict)
comptime _PyUnicode_DecodeUTF8 = ExternalFunction[
    "PyUnicode_DecodeUTF8", def(Int, Int, Int) thin abi("C") -> PyObjectPtr]
# double PyOS_string_to_double(const char *s, char **endptr, PyObject *overflow_exception)
comptime _PyOS_string_to_double = ExternalFunction[
    "PyOS_string_to_double", def(Int, Int, Int) thin abi("C") -> Float64]
# PyObject *PyLong_FromString(const char *str, char **pend, int base)
comptime _PyLong_FromString = ExternalFunction[
    "PyLong_FromString", def(Int, Int, c_int) thin abi("C") -> PyObjectPtr]
# int PyBuffer_ToContiguous(void *buf, const Py_buffer *src, Py_ssize_t len, char order)
# (the Py_buffer as the same word storage as PyObject_GetBuffer)
comptime _PyBuffer_ToContiguous = ExternalFunction[
    "PyBuffer_ToContiguous", def(UnsafePointer[UInt8, MutUntrackedOrigin], UnsafePointer[Int, MutUntrackedOrigin], Int, Int8) thin abi("C") -> c_int]

# Absorbed from perf2_shared_cpython_table.mojo (lane perf2-shared).
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
# long long PyLong_AsLongLong(PyObject *obj)  (-1 with an error set on failure)
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
# Absorbed from p3_lm_tokenizer_cpython.mojo (lane p3-lm-tokenizer).
# const char *PyUnicode_AsUTF8AndSize(PyObject *unicode, Py_ssize_t *size)
# (size as the address of one Int word; result an address, 0 on error)
comptime _PyUnicode_AsUTF8AndSize = ExternalFunction[
    "PyUnicode_AsUTF8AndSize", def(PyObjectPtr, Int) thin abi("C") -> Int]
# PyObject *PyBytes_FromStringAndSize(const char *v, Py_ssize_t len)  (new)
comptime _PyBytes_FromStringAndSize = ExternalFunction[
    "PyBytes_FromStringAndSize", def(Int, Int) thin abi("C") -> PyObjectPtr]
# PyObject *PyNumber_Index(PyObject *o)  (new exact int, NULL on error)
comptime _PyNumber_Index = ExternalFunction[
    "PyNumber_Index", def(PyObjectPtr) thin abi("C") -> PyObjectPtr]
# int PyDict_Next(PyObject *p, Py_ssize_t *ppos, PyObject **pkey, PyObject **pvalue)
# (the three out-pointers as addresses; keys and values borrowed)
comptime _PyDict_Next = ExternalFunction[
    "PyDict_Next", def(PyObjectPtr, Int, Int, Int) thin abi("C") -> c_int]
# PyObject *PyCodec_LookupError(const char *name)  (new; name as an address)
comptime _PyCodec_LookupError = ExternalFunction[
    "PyCodec_LookupError", def(Int) thin abi("C") -> PyObjectPtr]
# Absorbed from p3_labels_metrics_native.mojo (lane p3-labels-metrics).
# PyObject *PyObject_CallObject(PyObject *callable, PyObject *args)  (new)
comptime _PyObject_CallObject = ExternalFunction[
    "PyObject_CallObject", def(PyObjectPtr, PyObjectPtr) thin abi("C") -> PyObjectPtr]
# int PyObject_IsTrue(PyObject *o)
comptime _PyObject_IsTrue = ExternalFunction[
    "PyObject_IsTrue", def(PyObjectPtr) thin abi("C") -> c_int]
# PyObject *PyDict_GetItemWithError(PyObject *p, PyObject *key)  (borrowed; NULL: missing or error)
comptime _PyDict_GetItemWithError = ExternalFunction[
    "PyDict_GetItemWithError", def(PyObjectPtr, PyObjectPtr) thin abi("C") -> PyObjectPtr]
# double PyFloat_AsDouble(PyObject *o)  (-1.0 with an error set on failure)
comptime _PyFloat_AsDouble = ExternalFunction[
    "PyFloat_AsDouble", def(PyObjectPtr) thin abi("C") -> Float64]
# PyObject *PyDict_Copy(PyObject *p)  (new)
comptime _PyDict_Copy = ExternalFunction[
    "PyDict_Copy", def(PyObjectPtr) thin abi("C") -> PyObjectPtr]
# Lane ne1-e-table: the 26 calls of plan item A, and per-call by-name lookups.
# PyObject *PyModule_GetDict(PyObject *module)  (borrowed)
comptime _PyModule_GetDict = ExternalFunction[
    "PyModule_GetDict", def(PyObjectPtr) thin abi("C") -> PyObjectPtr]
# PyObject *PyObject_GetIter(PyObject *o)  (new, NULL on error)
comptime _PyObject_GetIter = ExternalFunction[
    "PyObject_GetIter", def(PyObjectPtr) thin abi("C") -> PyObjectPtr]
# PyObject *PyIter_Next(PyObject *o)  (new; NULL at the end or on error)
comptime _PyIter_Next = ExternalFunction[
    "PyIter_Next", def(PyObjectPtr) thin abi("C") -> PyObjectPtr]
# PyObject *PyType_GetModule(PyTypeObject *type)  (borrowed, NULL with an error)
comptime _PyType_GetModule = ExternalFunction[
    "PyType_GetModule", def(PyObjectPtr) thin abi("C") -> PyObjectPtr]
# PyObject *PySequence_Repeat(PyObject *o, Py_ssize_t count)  (new)
comptime _PySequence_Repeat = ExternalFunction[
    "PySequence_Repeat", def(PyObjectPtr, Int) thin abi("C") -> PyObjectPtr]

# Absorbed from ManifoldCalls of manifold_storage_native.mojo and the per-call
# loads of neural_labels_native.mojo (lane ne1-e-table).
# PyObject *PyObject_CallOneArg(PyObject *callable, PyObject *arg)  (new, NULL on error)
comptime _PyObject_CallOneArg = ExternalFunction[
    "PyObject_CallOneArg", def(PyObjectPtr, PyObjectPtr) thin abi("C") -> PyObjectPtr]
# PyObject *PyNumber_Long(PyObject *o)  (new)
comptime _PyNumber_Long = ExternalFunction[
    "PyNumber_Long", def(PyObjectPtr) thin abi("C") -> PyObjectPtr]
# PyObject *PyNumber_Float(PyObject *o)  (new)
comptime _PyNumber_Float = ExternalFunction[
    "PyNumber_Float", def(PyObjectPtr) thin abi("C") -> PyObjectPtr]
# PyObject *PyUnicode_FromKindAndData(int kind, const void *buffer, Py_ssize_t size)  (new)
comptime _PyUnicode_FromKindAndData = ExternalFunction[
    "PyUnicode_FromKindAndData", def(c_int, Int, Int) thin abi("C") -> PyObjectPtr]

# Added at integration (integ/ne-20261107) for ne3-c-samba (N170) and
# ne1-a-trees (N002). PyList_GET_ITEM and PyFloat_AS_DOUBLE, also asked by
# ne1-a-trees, are C macros / static inline functions with no exported
# symbol, so they cannot be table entries: callers use PyList_GetItem and
# PyFloat_AsDouble.
# PyObject *PyByteArray_FromStringAndSize(const char *string, Py_ssize_t len)  (new; NULL string = uninitialized)
comptime _PyByteArray_FromStringAndSize = ExternalFunction[
    "PyByteArray_FromStringAndSize", def(Int, Int) thin abi("C") -> PyObjectPtr]
# char *PyByteArray_AsString(PyObject *bytearray)  (address word)
comptime _PyByteArray_AsString = ExternalFunction[
    "PyByteArray_AsString", def(PyObjectPtr) thin abi("C") -> Int]
# long long PyLong_AsLongLongAndOverflow(PyObject *obj, int *overflow)  (overflow as an address word)
comptime _PyLong_AsLongLongAndOverflow = ExternalFunction[
    "PyLong_AsLongLongAndOverflow", def(PyObjectPtr, Int) thin abi("C") -> Int64]


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
    var PyUnicode_DecodeUTF8: _PyUnicode_DecodeUTF8.type
    var PyOS_string_to_double: _PyOS_string_to_double.type
    var PyLong_FromString: _PyLong_FromString.type
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
    var PyUnicode_AsUTF8AndSize: _PyUnicode_AsUTF8AndSize.type
    var PyBytes_FromStringAndSize: _PyBytes_FromStringAndSize.type
    var PyNumber_Index: _PyNumber_Index.type
    var PyDict_Next: _PyDict_Next.type
    var PyCodec_LookupError: _PyCodec_LookupError.type
    var PyObject_CallObject: _PyObject_CallObject.type
    var PyObject_IsTrue: _PyObject_IsTrue.type
    var PyDict_GetItemWithError: _PyDict_GetItemWithError.type
    var PyFloat_AsDouble: _PyFloat_AsDouble.type
    var PyDict_Copy: _PyDict_Copy.type
    var PyModule_GetDict: _PyModule_GetDict.type
    var PyObject_GetIter: _PyObject_GetIter.type
    var PyIter_Next: _PyIter_Next.type
    var PyType_GetModule: _PyType_GetModule.type
    var PySequence_Repeat: _PySequence_Repeat.type
    var PyObject_CallOneArg: _PyObject_CallOneArg.type
    var PyNumber_Long: _PyNumber_Long.type
    var PyNumber_Float: _PyNumber_Float.type
    var PyUnicode_FromKindAndData: _PyUnicode_FromKindAndData.type
    var PyByteArray_FromStringAndSize: _PyByteArray_FromStringAndSize.type
    var PyByteArray_AsString: _PyByteArray_AsString.type
    var PyLong_AsLongLongAndOverflow: _PyLong_AsLongLongAndOverflow.type

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
        self.PyUnicode_DecodeUTF8 = _PyUnicode_DecodeUTF8.load(lib)
        self.PyOS_string_to_double = _PyOS_string_to_double.load(lib)
        self.PyLong_FromString = _PyLong_FromString.load(lib)
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
        self.PyUnicode_AsUTF8AndSize = _PyUnicode_AsUTF8AndSize.load(lib)
        self.PyBytes_FromStringAndSize = _PyBytes_FromStringAndSize.load(lib)
        self.PyNumber_Index = _PyNumber_Index.load(lib)
        self.PyDict_Next = _PyDict_Next.load(lib)
        self.PyCodec_LookupError = _PyCodec_LookupError.load(lib)
        self.PyObject_CallObject = _PyObject_CallObject.load(lib)
        self.PyObject_IsTrue = _PyObject_IsTrue.load(lib)
        self.PyDict_GetItemWithError = _PyDict_GetItemWithError.load(lib)
        self.PyFloat_AsDouble = _PyFloat_AsDouble.load(lib)
        self.PyDict_Copy = _PyDict_Copy.load(lib)
        self.PyModule_GetDict = _PyModule_GetDict.load(lib)
        self.PyObject_GetIter = _PyObject_GetIter.load(lib)
        self.PyIter_Next = _PyIter_Next.load(lib)
        self.PyType_GetModule = _PyType_GetModule.load(lib)
        self.PySequence_Repeat = _PySequence_Repeat.load(lib)
        self.PyObject_CallOneArg = _PyObject_CallOneArg.load(lib)
        self.PyNumber_Long = _PyNumber_Long.load(lib)
        self.PyNumber_Float = _PyNumber_Float.load(lib)
        self.PyUnicode_FromKindAndData = _PyUnicode_FromKindAndData.load(lib)
        self.PyByteArray_FromStringAndSize = _PyByteArray_FromStringAndSize.load(lib)
        self.PyByteArray_AsString = _PyByteArray_AsString.load(lib)
        self.PyLong_AsLongLongAndOverflow = _PyLong_AsLongLongAndOverflow.load(lib)


# The named global is shared by every Mojo extension in the process (all of
# them load the same interpreter library). Bump the name's version whenever a
# field is added, removed or reordered.
comptime CPYTHON_TABLE = _Global[
    StorageType=CPythonTable, name="MojolearnCPythonTableV6",
    init_fn=CPythonTable.__init__,
]


def cpython_table() raises -> UnsafePointer[CPythonTable, MutAnyOrigin]:
    """The process-wide table, filled at first use."""
    # Same typed form as x_cnn/device.mojo `_slots` (compiles on main).
    return CPYTHON_TABLE.get_or_create_ptr().unsafe_origin_cast[MutAnyOrigin]()
