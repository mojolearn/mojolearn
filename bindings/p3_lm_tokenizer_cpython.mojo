# SPDX-License-Identifier: Apache-2.0
"""CPython entries used per item by the tokenizer, causal LM and byte LM
bindings, resolved once per process (same form as
`bindings/perf2_shared_cpython_table.mojo`). Calls through this table replace
`cpy.lib.get_function[...]("Name")` lookups inside per-document, per-token
and per-key loops. Pointer arguments and results are carried as `Int`
addresses, the form the call sites already use."""

from std.ffi import _Global
from std.python import Python
from std.python._cpython import ExternalFunction, PyObjectPtr


comptime _PyUnicode_AsUTF8AndSize = ExternalFunction[
    "PyUnicode_AsUTF8AndSize", def(PyObjectPtr, Int) thin abi("C") -> Int]
comptime _PyUnicode_DecodeUTF8 = ExternalFunction[
    "PyUnicode_DecodeUTF8", def(Int, Int, Int) thin abi("C") -> PyObjectPtr]
comptime _PyBytes_FromStringAndSize = ExternalFunction[
    "PyBytes_FromStringAndSize", def(Int, Int) thin abi("C") -> PyObjectPtr]
comptime _PyBytes_AsString = ExternalFunction[
    "PyBytes_AsString", def(PyObjectPtr) thin abi("C") -> Int]
comptime _PyList_New = ExternalFunction[
    "PyList_New", def(Int) thin abi("C") -> PyObjectPtr]
comptime _PyList_SetItem = ExternalFunction[
    "PyList_SetItem", def(PyObjectPtr, Int, PyObjectPtr) thin abi("C") -> Int32]
comptime _PyLong_FromLongLong = ExternalFunction[
    "PyLong_FromLongLong", def(Int64) thin abi("C") -> PyObjectPtr]
comptime _PyLong_AsLongLong = ExternalFunction[
    "PyLong_AsLongLong", def(PyObjectPtr) thin abi("C") -> Int64]
comptime _PyNumber_Index = ExternalFunction[
    "PyNumber_Index", def(PyObjectPtr) thin abi("C") -> PyObjectPtr]
comptime _PyDict_Next = ExternalFunction[
    "PyDict_Next", def(PyObjectPtr, Int, Int, Int) thin abi("C") -> Int32]
comptime _PyErr_Occurred = ExternalFunction[
    "PyErr_Occurred", def() thin abi("C") -> PyObjectPtr]
comptime _PyCodec_LookupError = ExternalFunction[
    "PyCodec_LookupError", def(Int) thin abi("C") -> PyObjectPtr]


struct P3LmTokenizerCPythonTable(Movable):
    var PyUnicode_AsUTF8AndSize: _PyUnicode_AsUTF8AndSize.type
    var PyUnicode_DecodeUTF8: _PyUnicode_DecodeUTF8.type
    var PyBytes_FromStringAndSize: _PyBytes_FromStringAndSize.type
    var PyBytes_AsString: _PyBytes_AsString.type
    var PyList_New: _PyList_New.type
    var PyList_SetItem: _PyList_SetItem.type
    var PyLong_FromLongLong: _PyLong_FromLongLong.type
    var PyLong_AsLongLong: _PyLong_AsLongLong.type
    var PyNumber_Index: _PyNumber_Index.type
    var PyDict_Next: _PyDict_Next.type
    var PyErr_Occurred: _PyErr_Occurred.type
    var PyCodec_LookupError: _PyCodec_LookupError.type

    def __init__(out self):
        ref cpy = Python().cpython()
        var lib = cpy.lib.borrow()
        self.PyUnicode_AsUTF8AndSize = _PyUnicode_AsUTF8AndSize.load(lib)
        self.PyUnicode_DecodeUTF8 = _PyUnicode_DecodeUTF8.load(lib)
        self.PyBytes_FromStringAndSize = _PyBytes_FromStringAndSize.load(lib)
        self.PyBytes_AsString = _PyBytes_AsString.load(lib)
        self.PyList_New = _PyList_New.load(lib)
        self.PyList_SetItem = _PyList_SetItem.load(lib)
        self.PyLong_FromLongLong = _PyLong_FromLongLong.load(lib)
        self.PyLong_AsLongLong = _PyLong_AsLongLong.load(lib)
        self.PyNumber_Index = _PyNumber_Index.load(lib)
        self.PyDict_Next = _PyDict_Next.load(lib)
        self.PyErr_Occurred = _PyErr_Occurred.load(lib)
        self.PyCodec_LookupError = _PyCodec_LookupError.load(lib)


# Shared by every Mojo extension in the process (all load the same
# interpreter library). Bump the name's version whenever a field is added,
# removed or reordered.
comptime P3_LM_TOKENIZER_CPYTHON_TABLE = _Global[
    StorageType=P3LmTokenizerCPythonTable, name="MojolearnP3LmTokenizerCPythonTableV1",
    init_fn=P3LmTokenizerCPythonTable.__init__,
]


def lm_cpython_table() raises -> UnsafePointer[P3LmTokenizerCPythonTable, MutAnyOrigin]:
    """The process-wide table, filled at first use."""
    return P3_LM_TOKENIZER_CPYTHON_TABLE.get_or_create_ptr().unsafe_origin_cast[MutAnyOrigin]()


def lm_try_index(value: PythonObject, mut word: Int) raises -> Bool:
    """`operator.index(value)` as a 64-bit word through the table: True and
    `word` set on success. False (with no Python error pending) for the bool
    singletons and for any value that is not an index or does not fit, so
    the caller runs its original admission path and keeps its wording."""
    if value is PythonObject(True) or value is PythonObject(False):
        return False
    var table = lm_cpython_table()
    var indexed = table[].PyNumber_Index(value._obj_ptr)
    if not indexed:
        ref cpy = Python().cpython()
        cpy.lib.get_function[NoneType]("PyErr_Clear")()
        return False
    var owned = PythonObject(from_owned=indexed)
    var result = table[].PyLong_AsLongLong(owned._obj_ptr)
    if result == -1 and table[].PyErr_Occurred():
        ref cpy = Python().cpython()
        cpy.lib.get_function[NoneType]("PyErr_Clear")()
        return False
    word = Int(result)
    return True
