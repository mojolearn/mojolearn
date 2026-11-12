# SPDX-License-Identifier: Apache-2.0
# The str-label encoder: Python hands a flat list of label objects to ONE
# native call. The code points of each str are read once through the C API
# (PyUnicode_AsUCS4Copy; no Python method, comparison or hash runs), the
# distinct labels are found by hashing and ordered by code point (Python's str
# ordering), and the codes are written into a native int32 store.
#   ("encode", labels)                 -> (classes, codes) or None
#   ("lookup", classes, values, float) -> (codes, first missing, missing count) or None
# None: a label is not a str (numbers and mixed lists keep the ORDER RULE's
# own refusals and encodings in `_labels.sorted_classes`).
from std.python import Python, PythonObject
from bindings.label_sequence_source import label_source_count, label_source_raw, label_source_value
from bindings.native_loaded_module import native_loaded_module
from std.python._cpython import PyObjectPtr, GILReleased
from bindings.cpython_table import cpython_table
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.py_balanced import py_call, py_type
from bindings.str_labels_native_run import (
    StrTable, StrClasses, StrUnseen, str_classes, _str_unique_native, _str_codes_native,
)
from bindings.p3_labels_metrics_native import (
    p3_text_codes, p3_text_classes, p3_text_lookup, p3_decode_list, P3_TEXT_OK, P3_TEXT_NOT_STR)

# Py_TPFLAGS_UNICODE_SUBCLASS
comptime UNICODE_FLAG = 1 << 28
# One code point above Unicode's range (0x10FFFF): the one-point label a
# non-str value is read as in a lenient read, which no str class matches.
comptime NOT_STR_POINT = UInt32(0x110000)


def _is_str(value: PythonObject) raises -> Bool:
    ref cpy = Python().cpython()
    var cls = py_type(value)
    var flags = cpython_table()[].PyType_GetFlags(cls._obj_ptr)
    return (flags & UInt64(UNICODE_FLAG)) != 0


def _read(labels: PythonObject, mut table: StrTable) raises -> Bool:
    var pins = List[PythonObject]()
    return _read_source[False](labels, pins, table)


def _read(labels: List[PythonObject], mut table: StrTable) raises -> Bool:
    return _read_source[True](PythonObject(None), labels, table)


def _read_source[native: Bool, strict: Bool = True](labels: PythonObject, pins: List[PythonObject],
                                                    mut table: StrTable) raises -> Bool:
    # Every label's code points; False when one is not a str (`strict`), or,
    # not `strict`, a non-str value read as the one point NOT_STR_POINT
    # (no str class matches it). The C API entries are resolved once per
    # call, not once per label.
    ref cpy = Python().cpython()
    # perf-audit-3: the three C-API entries come from the process table
    # (resolved once at import), not three dlsym lookups per call.
    var cpt = cpython_table()
    for i in range(label_source_count[native](labels, pins)):  # api-bound-loop: over the caller's Python list of str labels, one read per label
        var value = label_source_value[native](labels, pins, i)
        if (cpt[].PyType_GetFlags(py_type(value)._obj_ptr) & UInt64(UNICODE_FLAG)) == 0:
            comptime if strict:
                return False
            else:
                table.starts.append(len(table.points))
                table.lengths.append(1)
                table.points.append(NOT_STR_POINT)
                continue
        var count = cpt[].PyUnicode_GetLength(value._obj_ptr)
        if count < 0:
            raise cpy.unsafe_get_error()
        var address = cpt[].PyUnicode_AsUCS4Copy(value._obj_ptr)
        if address == 0:
            raise cpy.unsafe_get_error()
        var source = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=address)
        table.starts.append(len(table.points))
        table.lengths.append(count)
        for k in range(count):  # native data loop, no CPython (not a driver loop): over one label's code points
            table.points.append(source[k])
        cpt[].PyMem_Free(address)
    _ = pins
    _ = labels
    _ = pins
    return True


def str_unique_into(table: StrTable, n: Int, codes_address: Int) raises -> List[Int]:
    """The classes of the n labels of `table` in code-point order, as the row
    of each class's FIRST label, with each label's int32 class index written
    at `codes_address` (n words, the caller's). The classes and codes the
    stable sort of all n labels gave (`str_encode`'s former fallback and
    `_text_fit`), from hashing plus a sort of the distinct classes only.
    Runs with the GIL released: the work reads only the native table and
    writes the caller's code words (round2-fixes-2; the host pool ran with
    the GIL held, perf audit 2-b)."""
    var firsts = List[Int]()
    with GILReleased(Python()):
        firsts = _str_unique_native(table, n, codes_address)
    return firsts^


def str_codes_into(rows: StrTable, known: StrTable, classes: StrClasses, address: Int,
                   as_float: Bool) raises -> StrUnseen:
    """THE str-label lookup (every caller: `str_lookup`'s code-point path and
    bindings/prep_labels_native.mojo `_text_codes`): each label of `rows`
    looked up among `classes` of `known`, its index (-1 unseen) written as
    int32 (float32 with `as_float`) at `address`, across the host pool with
    the GIL released (only native tables and the caller's words are read or
    written)."""
    var unseen = StrUnseen(List[Int](), 0)
    with GILReleased(Python()):
        unseen = _str_codes_native(rows, known, classes, address, as_float)
    return unseen^


def _store(code: String, n: Int) raises -> PythonObject:
    var array = native_loaded_module("array")
    var b = native_loaded_module("builtins")
    return py_call(array.array, code, py_call(b.bytes, 4 * max(n, 0)))


def _address(value: PythonObject) raises -> Int:
    ref cpy = Python().cpython()
    var view = List[Int](length=10, fill=0)
    var pointer = view.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    if cpython_table()[].PyObject_GetBuffer(value._obj_ptr, pointer, Int32(0)) != 0:
        raise cpy.unsafe_get_error()
    var address = view[0]
    cpython_table()[].PyBuffer_Release(pointer)
    # Keep the Py_buffer storage alive past the release.
    _ = view^
    return address


def str_encode(labels: PythonObject) raises -> PythonObject:
    """`sorted(dict.fromkeys(labels))` and each label's code, for str labels."""
    var n = len(labels)
    if n == 0:
        return PythonObject(None)
    # p3-labels-metrics: hash the labels to their k distinct values over the
    # UTF-8 bytes CPython caches in each str, sort only the k classes and
    # write each row's code from its hash slot (bindings/p3_labels_metrics_native.mojo).
    # The same classes (the first label of each class), the same code-point
    # order and codes; a str with no UTF-8 form takes the code-point path below.
    var fast = p3_text_codes(labels, True)
    if fast.status == P3_TEXT_NOT_STR:
        return PythonObject(None)
    if fast.status == P3_TEXT_OK:
        var fast_codes = _store("i", n)
        var target = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=_address(fast_codes))
        for i in range(n):  # native data loop, no CPython (not a driver loop): native copy of the n codes
            target[i] = fast.codes[i]
        _ = fast_codes  # keep the Python object alive through the native writes above
        return Python.tuple(p3_text_classes(labels, fast.first), fast_codes)
    var table = StrTable()
    if not _read(labels, table):
        return PythonObject(None)
    var codes = _store("i", n)
    # Each class is its first label (the stable order's run head), the
    # codes their ranks (`str_unique_into`: hashing, then a sort of the
    # distinct classes only).
    var firsts = str_unique_into(table, n, _address(codes))
    var classes = Python.list()
    for c in range(len(firsts)):  # driver-loop: over the distinct classes (one str each)
        _ = py_call(classes.append, labels[firsts[c]])
    _ = codes  # keep the Python object alive through the native writes above
    return Python.tuple(classes, codes)


def str_lookup(classes: PythonObject, values: PythonObject, as_float: Bool) raises -> PythonObject:
    """`{c: i for i, c in enumerate(classes)}.get(v, -1)` per value, for str
    classes: a duplicated class keeps its LAST index, as the comprehension
    does; a value that is not a str matches no str class."""
    var k = len(classes)
    if k == 0:
        return PythonObject(None)
    # p3-labels-metrics: classes hashed by their UTF-8 bytes once, each value
    # one borrowed read and one hash probe (no per-value code-point copy or
    # binary search). The former path below answers when a class is not a
    # str (None) or has no UTF-8 form.
    var n_fast = len(values)
    var fast_codes = _store("f" if as_float else "i", n_fast)
    var fast_missing = Python.list()
    var fast_count = p3_text_lookup(classes, values, _address(fast_codes), as_float, fast_missing)
    _ = fast_codes  # keep the Python object alive through the native writes above
    if fast_count >= 0:
        return Python.tuple(fast_codes, fast_missing, fast_count)
    var table = StrTable()
    if not _read(classes, table):
        return PythonObject(None)
    # The distinct sorted classes and the index each answers with.
    var lookup = str_classes(table, k)
    var n = len(values)
    # round2-fixes-2: every value's code points read ONCE (a non-str value as
    # a point no class matches), then the one shared lookup `str_codes_into`
    # (host pool, GIL released). The former loop copied each value into a
    # one-label table and searched it serially: a second lookup beside
    # prep_labels `_text_codes` (perf audit 2-b, 3).
    var rows = StrTable()
    var pins = List[PythonObject]()
    _ = _read_source[False, False](values, pins, rows)
    var codes = _store("f" if as_float else "i", n)
    var unseen = str_codes_into(rows, table, lookup, _address(codes), as_float)
    _ = codes  # keep the Python object alive through the native writes above
    var missing = Python.list()
    for u in range(len(unseen.rows)):  # driver-loop: over the first five unseen labels
        _ = py_call(missing.append, values[unseen.rows[u]])
    return Python.tuple(codes, missing, unseen.count)


def str_labels_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        var b = native_loaded_module("builtins")
        var operation = String(call[0])
        var result = PythonObject(None)
        if operation == "encode":
            if py_type(call[1]) is not b.list:
                raise Error("TypeError: str label encoding requires a native list")
            result = str_encode(call[1])
        elif operation == "decode":
            # ("decode", classes, codes address, n, code width 4 or 8) ->
            # [classes[c] for c in codes] as one presized list, or None
            # when a code is out of range (the caller's Python indexing
            # then raises its IndexError).
            var cls = py_type(call[1])
            if cls is not b.list and cls is not b.tuple:
                raise Error("TypeError: str label decoding requires a native list or tuple")
            var width = Int(py=call[4])
            if width != 4 and width != 8:
                raise Error("TypeError: str label decoding requires int32 or int64 codes")
            result = p3_decode_list(call[1], Int(py=call[2]), width, Int(py=call[3]))
        elif operation == "lookup":
            if py_type(call[1]) is not b.list or py_type(call[2]) is not b.list:
                raise Error("TypeError: str label lookup requires native lists")
            result = str_lookup(call[1], call[2], Bool(py=call[3]))
        else:
            raise Error("TypeError: unknown str label operation")
        return result.steal_data()
    except error:
        if String(error).startswith("TypeError: "):
            var message = String(error)
            return raise_python_exception(Error(String(message[byte=11:])), ExceptionType.TypeError)
        return raise_python_exception(error^, ExceptionType.ValueError)
