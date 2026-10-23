# SPDX-License-Identifier: Apache-2.0
# The str-label encoder: Python hands a flat list of label objects to ONE
# native call. The code points of each str are read once through the C API
# (PyUnicode_AsUCS4Copy; no Python method, comparison or hash runs), the
# distinct labels are ordered by code point (Python's str ordering) with a
# stable merge sort, and the codes are written into a native int32 store.
#   ("encode", labels)                 -> (classes, codes) or None
#   ("lookup", classes, values, float) -> (codes, first missing, missing count) or None
# None: a label is not a str (numbers and mixed lists keep the ORDER RULE's
# own refusals and encodings in `_labels.sorted_classes`).
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.py_balanced import py_call, py_type

# Py_TPFLAGS_UNICODE_SUBCLASS
comptime UNICODE_FLAG = 1 << 28


struct StrTable(Movable):
    """The code points of n labels, one contiguous UInt32 run each."""
    var points: List[UInt32]
    var starts: List[Int]
    var lengths: List[Int]

    def __init__(out self):
        self.points = List[UInt32]()
        self.starts = List[Int]()
        self.lengths = List[Int]()

    def compare(self, a: Int, b: Int) -> Int:
        var la = self.lengths[a]
        var lb = self.lengths[b]
        var sa = self.starts[a]
        var sb = self.starts[b]
        for k in range(min(la, lb)):  # driver-loop: over one label's code points
            var x = self.points[sa + k]
            var y = self.points[sb + k]
            if x != y:
                return -1 if x < y else 1
        if la == lb:
            return 0
        return -1 if la < lb else 1


def _is_str(value: PythonObject) raises -> Bool:
    ref cpy = Python().cpython()
    var cls = py_type(value)
    var flags = cpy.lib.get_function[UInt64]("PyType_GetFlags")(cls._obj_ptr)
    return (flags & UInt64(UNICODE_FLAG)) != 0


def _read(labels: PythonObject, mut table: StrTable) raises -> Bool:
    # Every label's code points; False when one is not a str.
    ref cpy = Python().cpython()
    for i in range(len(labels)):  # driver-loop: the explicit label input step, one read per label
        var value = labels[i]
        if not _is_str(value):
            return False
        var count = cpy.lib.get_function[Int]("PyUnicode_GetLength")(value._obj_ptr)
        if count < 0:
            raise cpy.unsafe_get_error()
        var address = cpy.lib.get_function[Int]("PyUnicode_AsUCS4Copy")(value._obj_ptr)
        if address == 0:
            raise cpy.unsafe_get_error()
        var source = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=address)
        table.starts.append(len(table.points))
        table.lengths.append(count)
        for k in range(count):  # driver-loop: over one label's code points
            table.points.append(source[k])
        cpy.lib.get_function[NoneType]("PyMem_Free")(address)
    return True


def _sorted(table: StrTable, n: Int) -> List[Int]:
    # A stable bottom-up merge sort of 0..n-1 by code-point order: equal
    # labels keep their input order, so each run starts at its first label.
    var order = List[Int](capacity=n)
    for i in range(n):  # driver-loop: over the label indices
        order.append(i)
    var buffer = List[Int](length=n, fill=0)
    var width = 1
    while width < n:  # driver-loop: over merge widths
        var lo = 0
        while lo < n:  # driver-loop: over merged runs
            var mid = min(lo + width, n)
            var hi = min(lo + 2 * width, n)
            var i = lo
            var j = mid
            var k = lo
            while i < mid and j < hi:  # driver-loop: over one merge
                if table.compare(order[j], order[i]) < 0:
                    buffer[k] = order[j]
                    j += 1
                else:
                    buffer[k] = order[i]
                    i += 1
                k += 1
            while i < mid:  # driver-loop: over one merge's tail
                buffer[k] = order[i]
                i += 1
                k += 1
            while j < hi:  # driver-loop: over one merge's tail
                buffer[k] = order[j]
                j += 1
                k += 1
            lo = hi
        for t in range(n):  # driver-loop: over the label indices
            order[t] = buffer[t]
        width *= 2
    return order^


def _store(code: String, n: Int) raises -> PythonObject:
    var array = Python.import_module("array")
    var b = Python.import_module("builtins")
    return py_call(array.array, code, py_call(b.bytes, 4 * max(n, 0)))


def _address(value: PythonObject) raises -> Int:
    ref cpy = Python().cpython()
    var view = List[Int](length=10, fill=0)
    var pointer = view.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    if cpy.lib.get_function[Int32]("PyObject_GetBuffer")(value._obj_ptr, pointer, Int32(0)) != 0:
        raise cpy.unsafe_get_error()
    var address = view[0]
    cpy.lib.get_function[NoneType]("PyBuffer_Release")(pointer)
    return address


def str_encode(labels: PythonObject) raises -> PythonObject:
    """`sorted(dict.fromkeys(labels))` and each label's code, for str labels."""
    var n = len(labels)
    if n == 0:
        return PythonObject(None)
    var table = StrTable()
    if not _read(labels, table):
        return PythonObject(None)
    var order = _sorted(table, n)
    var classes = Python.list()
    var codes = _store("i", n)
    var out = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=_address(codes))
    _ = codes  # keep the Python object alive through the native call above
    var current = -1
    for t in range(n):  # driver-loop: over the sorted labels
        var i = order[t]
        if t == 0 or table.compare(order[t - 1], i) != 0:
            current += 1
            # the first label of each run is the first seen (stable order)
            _ = py_call(classes.append, labels[i])
        out[i] = Int32(current)
        _ = codes  # keep the Python object alive through the native call above
    return Python.tuple(classes, codes)


def str_lookup(classes: PythonObject, values: PythonObject, as_float: Bool) raises -> PythonObject:
    """`{c: i for i, c in enumerate(classes)}.get(v, -1)` per value, for str
    classes: a duplicated class keeps its LAST index, as the comprehension
    does; a value that is not a str matches no str class."""
    var k = len(classes)
    var table = StrTable()
    if k == 0 or not _read(classes, table):
        return PythonObject(None)
    var order = _sorted(table, k)
    # The distinct sorted classes and the index each answers with.
    var distinct = List[Int]()
    var index = List[Int]()
    for t in range(k):  # driver-loop: over the sorted classes
        if t == 0 or table.compare(order[t - 1], order[t]) != 0:
            distinct.append(order[t])
            index.append(order[t])
        else:
            index[len(index) - 1] = max(index[len(index) - 1], order[t])
    var n = len(values)
    var codes = _store("f" if as_float else "i", n)
    var address = _address(codes)
    _ = codes  # keep the Python object alive through the native call above
    var missing = Python.list()
    var count = 0
    ref cpy = Python().cpython()
    for i in range(n):  # driver-loop: the explicit label input step, one lookup per label
        var value = values[i]
        var found = -1
        if _is_str(value):
            var single = StrTable()
            var one = Python.list(value)
            _ = _read(one, single)
            # binary search of the distinct classes, by code point
            var lo = 0
            var hi = len(distinct)
            while lo < hi:  # driver-loop: over a binary search of the classes
                var mid = (lo + hi) // 2
                var c = _cross(table, distinct[mid], single)
                if c == 0:
                    found = index[mid]
                    break
                if c < 0:
                    lo = mid + 1
                else:
                    hi = mid
        if found < 0:
            count += 1
            if len(missing) < 5:
                _ = py_call(missing.append, value)
        if as_float:
            MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=address)[i] = Float32(found)
        else:
            MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=address)[i] = Int32(found)
    return Python.tuple(codes, missing, count)


def _cross(table: StrTable, a: Int, single: StrTable) -> Int:
    # table[a] against the one label of `single`, by code point.
    var la = table.lengths[a]
    var lb = single.lengths[0]
    var sa = table.starts[a]
    for k in range(min(la, lb)):  # driver-loop: over one label's code points
        var x = table.points[sa + k]
        var y = single.points[k]
        if x != y:
            return -1 if x < y else 1
    if la == lb:
        return 0
    return -1 if la < lb else 1


def str_labels_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        var b = Python.import_module("builtins")
        var operation = String(call[0])
        var result = PythonObject(None)
        if operation == "encode":
            if py_type(call[1]) is not b.list:
                raise Error("TypeError: str label encoding requires a native list")
            result = str_encode(call[1])
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
