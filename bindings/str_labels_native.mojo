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
from std.python._cpython import PyObjectPtr
from bindings.cpython_table import cpython_table
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.py_balanced import py_call, py_type
from bindings.p3_labels_metrics_native import (
    p3_text_codes, p3_text_classes, p3_text_lookup, p3_decode_list, P3_TEXT_OK, P3_TEXT_NOT_STR)
from core.host_parallel import host_parallelize

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
        return str_cross(self, a, self, b)


def str_cross(rows: StrTable, i: Int, known: StrTable, j: Int) -> Int:
    """Code-point order (Python's str order) of label i of `rows` against
    label j of `known`: -1, 0 or 1. The one comparison every str-label path
    uses."""
    var la = rows.lengths[i]
    var lb = known.lengths[j]
    var sa = rows.starts[i]
    var sb = known.starts[j]
    for q in range(min(la, lb)):  # native data loop, no CPython (not a driver loop): over one label's code points (native)
        var x = rows.points[sa + q]
        var y = known.points[sb + q]
        if x != y:
            return -1 if x < y else 1
    if la == lb:
        return 0
    return -1 if la < lb else 1


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


def _read_source[native: Bool](labels: PythonObject, pins: List[PythonObject], mut table: StrTable) raises -> Bool:
    # Every label's code points; False when one is not a str. The C API
    # entries are resolved once per call, not once per label.
    ref cpy = Python().cpython()
    var get_length = cpy.lib.get_function[Int]("PyUnicode_GetLength")
    var as_ucs4 = cpy.lib.get_function[Int]("PyUnicode_AsUCS4Copy")
    var mem_free = cpy.lib.get_function[NoneType]("PyMem_Free")
    for i in range(label_source_count[native](labels, pins)):  # api-bound-loop: over the caller's Python list of str labels, one read per label
        var value = label_source_value[native](labels, pins, i)
        if (cpython_table()[].PyType_GetFlags(py_type(value)._obj_ptr) & UInt64(UNICODE_FLAG)) == 0:
            return False
        var count = get_length(value._obj_ptr)
        if count < 0:
            raise cpy.unsafe_get_error()
        var address = as_ucs4(value._obj_ptr)
        if address == 0:
            raise cpy.unsafe_get_error()
        var source = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=address)
        table.starts.append(len(table.points))
        table.lengths.append(count)
        for k in range(count):  # native data loop, no CPython (not a driver loop): over one label's code points
            table.points.append(source[k])
        mem_free(address)
    _ = pins
    _ = labels
    _ = pins
    return True


def _sorted(table: StrTable, n: Int) -> List[Int]:
    # A stable bottom-up merge sort of 0..n-1 by code-point order: equal
    # labels keep their input order, so each run starts at its first label.
    var rows = List[Int](capacity=n)
    for i in range(n):  # native data loop, no CPython (not a driver loop): over the label indices
        rows.append(i)
    # The identity rows: the sorted positions are the sorted label indices.
    return _sort_by_rows(table, rows)


def _sort_by_rows(table: StrTable, rows: List[Int]) -> List[Int]:
    """The positions 0..len(rows)-1 ordered by the code points of their rows
    of `table` (a permutation of positions, not rows): a stable bottom-up
    merge sort (equal labels keep their order)."""
    var n = len(rows)
    var order = List[Int](capacity=n)
    for i in range(n):  # native data loop, no CPython (not a driver loop): over the positions
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
                if table.compare(rows[order[j]], rows[order[i]]) < 0:
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
        for t in range(n):  # native data loop, no CPython (not a driver loop): over the positions
            order[t] = buffer[t]
        width *= 2
    return order^


# ---------------------------------------------------------------- distinct labels
# The ORDER RULE for str labels (the distinct values in code-point order, each
# class the FIRST label of its value, each label's code its class's rank) in
# O(n) hashing plus a sort of the k DISTINCT classes only (round2-fixes,
# 2026-11-11; perf audit 2 B): every label is hashed once across the host
# pool, each task keeps its row chunk's distinct labels (the earliest row of
# each), ONE merge of the per-task sets in chunk order keeps each class's
# global first row, the k classes are sorted, and the codes are written across
# the pool. The former route merge-sorted all n labels on one thread.

comptime _UNIQUE_CHUNK = 65536  # labels per hashing task: each task hashes and probes tens of thousands of labels, so the pool's dispatch cost is amortized whatever the label count
comptime _FNV_OFFSET = UInt64(0xCBF29CE484222325)
comptime _FNV_PRIME = UInt64(0x100000001B3)


def _str_hash(table: StrTable, i: Int) -> UInt64:
    # FNV-1a over label i's code points and its length.
    var h = _FNV_OFFSET
    var s = table.starts[i]
    for q in range(table.lengths[i]):  # native data loop, no CPython (not a driver loop): over one label's code points (native)
        h = (h ^ UInt64(table.points[s + q])) * _FNV_PRIME
    return (h ^ UInt64(table.lengths[i])) * _FNV_PRIME


def _pow2_at_least(n: Int) -> Int:
    var c = 1
    while c < n:  # driver-loop: over powers of two (a size, not data)
        c *= 2
    return c


def str_unique_into(table: StrTable, n: Int, codes_address: Int) -> List[Int]:
    """The classes of the n labels of `table` in code-point order, as the row
    of each class's FIRST label, with each label's int32 class index written
    at `codes_address` (n words, the caller's). The classes and codes the
    stable sort of all n labels gave (`str_encode`'s former fallback and
    `_text_fit`), from hashing plus a sort of the distinct classes only."""
    if n <= 0:
        return List[Int]()
    var units = (n + _UNIQUE_CHUNK - 1) // _UNIQUE_CHUNK
    var local_cap = _pow2_at_least(2 * min(n, _UNIQUE_CHUNK))  # load at most one half: a probe always ends
    var hashes = List[UInt64](length=n, fill=0)
    var local_rows = List[Int](length=units * local_cap, fill=-1)
    var tp = UnsafePointer(to=table)
    var hashes_address = Int(hashes.unsafe_ptr())
    var rows_address = Int(local_rows.unsafe_ptr())
    def dedupe(unit: Int) {imm tp, imm n, imm local_cap, imm hashes_address, imm rows_address}:
        var hp = MutPointer[UInt64, MutUntrackedOrigin](unsafe_from_address=hashes_address)
        var slots = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=rows_address)
        var base = unit * local_cap
        var mask = local_cap - 1
        var stop = min(n, (unit + 1) * _UNIQUE_CHUNK)
        for i in range(unit * _UNIQUE_CHUNK, stop):  # native data loop, no CPython (not a driver loop): labels of one hashing chunk (native, no Python)
            var h = _str_hash(tp[], i)
            hp[i] = h
            var s = Int(h & UInt64(mask))
            while True:  # driver-loop: over one open-addressing probe (native)
                var r = slots[base + s]
                if r < 0:
                    slots[base + s] = i  # the chunk's earliest row of this label
                    break
                if hp[r] == h and tp[].compare(r, i) == 0:
                    break
                s = (s + 1) & mask
    host_parallelize(dedupe, units)
    _ = hashes  # written by every task above
    # ONE merge of the per-task sets, in chunk order: the first insertion of
    # a class comes from its earliest chunk, whose slot holds its earliest
    # row there, so each class keeps its global first row.
    var total = 0
    for u in range(units * local_cap):  # native data loop, no CPython (not a driver loop): over the per-task slots (native)
        if local_rows[u] >= 0:
            total += 1
    var cap = _pow2_at_least(2 * total + 2)
    var mask = cap - 1
    var global_rows = List[Int](length=cap, fill=-1)
    var global_ids = List[Int](length=cap, fill=-1)
    var reps = List[Int](capacity=total)
    for u in range(units * local_cap):  # native data loop, no CPython (not a driver loop): the one merge of the per-task distinct sets, in chunk order (native)
        var r = local_rows[u]
        if r < 0:
            continue
        var h = hashes[r]
        var g = Int(h & UInt64(mask))
        while True:  # driver-loop: over one open-addressing probe (native)
            var q = global_rows[g]
            if q < 0:
                global_rows[g] = r
                global_ids[g] = len(reps)
                reps.append(r)
                break
            if hashes[q] == h and table.compare(q, r) == 0:
                break
            g = (g + 1) & mask
    _ = local_rows^
    # Sort only the k distinct classes (ids are positions in `reps`);
    # rank[id] is the class index.
    var order = _sort_by_rows(table, reps)
    var k = len(reps)
    var firsts = List[Int](length=k, fill=0)
    var rank = List[Int](length=k, fill=0)
    for c in range(k):  # driver-loop: over the distinct classes (native)
        firsts[c] = reps[order[c]]
        rank[order[c]] = c
    var rows_at = Int(global_rows.unsafe_ptr())
    var ids_at = Int(global_ids.unsafe_ptr())
    var rank_at = Int(rank.unsafe_ptr())
    def assign(unit: Int) {imm tp, imm n, imm mask, imm hashes_address, imm rows_at, imm ids_at, imm rank_at,
                           imm codes_address}:
        var hp = MutPointer[UInt64, MutUntrackedOrigin](unsafe_from_address=hashes_address)
        var gr = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=rows_at)
        var gi = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=ids_at)
        var rk = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=rank_at)
        var dst = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=codes_address)
        var stop = min(n, (unit + 1) * _UNIQUE_CHUNK)
        for i in range(unit * _UNIQUE_CHUNK, stop):  # native data loop, no CPython (not a driver loop): labels of one code chunk (native, no Python)
            var h = hp[i]
            var g = Int(h & UInt64(mask))
            while True:  # driver-loop: over one open-addressing probe (native)
                var q = gr[g]
                if q < 0:
                    dst[i] = Int32(-1)  # unreachable: every label's class was merged
                    break
                if hp[q] == h and tp[].compare(q, i) == 0:
                    dst[i] = Int32(rk[gi[g]])
                    break
                g = (g + 1) & mask
    host_parallelize(assign, units)
    _ = hashes  # read by every task above
    _ = global_rows
    _ = global_ids
    _ = rank
    _ = table
    return firsts^


@fieldwise_init
struct StrClasses(Movable):
    """Fitted str classes for lookup: the distinct classes in code-point
    order (`distinct`, rows of the classes table) and the index each answers
    with (`index`: a duplicated class keeps its LAST index, as
    `{c: i for i, c in enumerate(classes)}` does)."""
    var distinct: List[Int]
    var index: List[Int]


def str_classes(known: StrTable, k: Int) -> StrClasses:
    """The lookup form of the k classes of `known` (`StrClasses`)."""
    var order = _sorted(known, k)
    var distinct = List[Int]()
    var index = List[Int]()
    for t in range(k):  # driver-loop: over the sorted classes
        if t == 0 or known.compare(order[t - 1], order[t]) != 0:
            distinct.append(order[t])
            index.append(order[t])
        else:
            index[len(index) - 1] = max(index[len(index) - 1], order[t])
    return StrClasses(distinct^, index^)


def str_find(rows: StrTable, i: Int, known: StrTable, classes: StrClasses) -> Int:
    """Label i of `rows` looked up among the classes of `known`: its class
    index, or -1. A binary search of the distinct sorted classes."""
    var lo = 0
    var hi = len(classes.distinct)
    while lo < hi:  # driver-loop: over a binary search of the classes
        var mid = (lo + hi) // 2
        var c = str_cross(rows, i, known, classes.distinct[mid])
        if c == 0:
            return classes.index[mid]
        if c > 0:
            lo = mid + 1
        else:
            hi = mid
    return -1


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
    var codes = _store("f" if as_float else "i", n)
    var address = _address(codes)
    _ = codes  # keep the Python object alive through the native call above
    var missing = Python.list()
    var count = 0
    ref cpy = Python().cpython()
    # One reused one-label table and the C API entries resolved once: each
    # value used to build a Python list, a fresh table and four symbol
    # lookups.
    var get_length = cpy.lib.get_function[Int]("PyUnicode_GetLength")
    var as_ucs4 = cpy.lib.get_function[Int]("PyUnicode_AsUCS4Copy")
    var mem_free = cpy.lib.get_function[NoneType]("PyMem_Free")
    var single = StrTable()
    for i in range(n):  # api-bound-loop: over the caller's Python list of str labels, one lookup per label
        var value = values[i]
        var found = -1
        if (cpython_table()[].PyType_GetFlags(py_type(value)._obj_ptr) & UInt64(UNICODE_FLAG)) != 0:
            var length = get_length(value._obj_ptr)
            if length < 0:
                raise cpy.unsafe_get_error()
            var ucs4 = as_ucs4(value._obj_ptr)
            if ucs4 == 0:
                raise cpy.unsafe_get_error()
            var source = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=ucs4)
            single.points.clear()
            single.starts.clear()
            single.lengths.clear()
            single.starts.append(0)
            single.lengths.append(length)
            for k in range(length):  # native data loop, no CPython (not a driver loop): over one label's code points
                single.points.append(source[k])
            mem_free(ucs4)
            found = str_find(single, 0, table, lookup)
        if found < 0:
            count += 1
            if len(missing) < 5:
                _ = py_call(missing.append, value)
        if as_float:
            MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=address)[i] = Float32(found)
            _ = codes  # keep the Python object alive through the native call above
        else:
            MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=address)[i] = Int32(found)
            _ = codes  # keep the Python object alive through the native call above
    return Python.tuple(codes, missing, count)


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
