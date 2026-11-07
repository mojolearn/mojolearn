# SPDX-License-Identifier: Apache-2.0
# Native label passes shared by the label and metric doors (lane
# p3-labels-metrics, 2026-11-06).
#
# The label doors read Python label lists one item at a time through boxed
# indexing (`labels[i]`, one PyObject_GetItem and one PythonObject per item),
# appended results through a Python `list.append` call per item, and encoded
# str labels by an n log n code-point merge sort over every row. The helpers
# below keep the same classes, codes, class order, outputs and refusals and
# do the per-item work in C calls only:
#
# - items are read borrowed from the exact list / tuple storage
#   (PyList_GetItem / PyTuple_GetItem), their class compared by pointer;
# - result lists are built presized (PyList_New + PyList_SetItem);
# - str labels are hashed to their k distinct values first (FNV-1a over the
#   UTF-8 bytes CPython caches in the str object, PyUnicode_AsUTF8AndSize: no
#   per-label copy), only the k classes are sorted, and each row's code comes
#   from its hash slot. UTF-8 byte order equals code-point order, so the
#   classes come out in Python's str order; the first label of each class is
#   kept, as `dict.fromkeys` keeps it. A str holding a lone surrogate has no
#   UTF-8 form: the helper answers P3_TEXT_FALLBACK and the caller runs its
#   former code-point path unchanged.
#
# The C entries come from the one CPython table, bindings/cpython_table.mojo
# (this module's own table moved there, lane ne1-e-table, 2026-11-06).
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.native_loaded_module import native_loaded_module
from bindings.cpython_table import cpython_table

# Py_TPFLAGS_UNICODE_SUBCLASS
comptime P3_UNICODE_FLAG = 1 << 28


def p3_call_object(entry: PythonObject, args: PythonObject) raises -> PythonObject:
    """`entry(*args)` for an exact tuple `args`; NULL raises the pending
    Python error. The symbol is resolved once per process."""
    ref cpy = Python().cpython()
    var result = cpython_table()[].PyObject_CallObject(entry._obj_ptr, args._obj_ptr)
    if not result:
        raise cpy.unsafe_get_error()
    return PythonObject(from_owned=result)


def p3_type_word(raw: PyObjectPtr) raises -> Int:
    """The class pointer of a live object as a word (the reference
    PyObject_Type returns is dropped at once; the object keeps its class
    alive)."""
    ref cpy = Python().cpython()
    var cls = cpy.PyObject_Type(raw)
    var word = Int(cls)
    cpy.Py_DecRef(cls)
    return word


def p3_item(container: PythonObject, index: Int, is_tuple: Bool) raises -> PyObjectPtr:
    """A borrowed item of an exact list or tuple (the caller holds the
    container and runs no Python code while it uses the item)."""
    ref cpy = Python().cpython()
    var raw = cpython_table()[].PyTuple_GetItem(container._obj_ptr, index) if is_tuple else cpython_table()[].PyList_GetItem(container._obj_ptr, index)
    if not raw:
        raise cpy.unsafe_get_error()
    return raw


# ------------------------------------------------------------ decode


def p3_decode_list(classes: PythonObject, codes_addr: Int, width: Int, n: Int) raises -> PythonObject:
    """`[classes[c] for c in codes]` for n int32 (`width` 4) or int64
    (`width` 8) codes at `codes_addr`, `classes` an exact list or tuple: one
    presized list filled in C with the class objects themselves (new
    references). A negative code counts from the end, as Python indexing
    does. None when a code is out of range (the caller's Python indexing
    then raises its own IndexError)."""
    var b = native_loaded_module("builtins")
    var is_tuple = p3_type_word(classes._obj_ptr) == Int(b.tuple._obj_ptr)
    if not is_tuple and p3_type_word(classes._obj_ptr) != Int(b.list._obj_ptr):
        raise Error("TypeError: class labels require an exact native list or tuple")
    var k = len(classes)
    # Check every code first, so a refusal leaves nothing half built.
    for i in range(n):  # driver-loop: native range check of the code words
        var code = Int(MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=codes_addr)[i]) if width == 4 else Int(MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=codes_addr)[i])
        if code < -k or code >= k:
            return PythonObject(None)
    ref cpy = Python().cpython()
    var t = cpython_table()
    var out = t[].PyList_New(n)
    if not out:
        raise cpy.unsafe_get_error()
    var result = PythonObject(from_owned=out)
    for i in range(n):  # driver-loop: native list fill, one borrowed class read and one C store per row
        var code = Int(MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=codes_addr)[i]) if width == 4 else Int(MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=codes_addr)[i])
        if code < 0:
            code += k
        var item = p3_item(classes, code, is_tuple)
        _ = t[].PyList_SetItem(result._obj_ptr, i, cpy.Py_NewRef(item))
    _ = classes  # keep the class container alive through the borrowed reads above
    return result


# ------------------------------------------------------------ str labels

comptime P3_TEXT_OK = 1
comptime P3_TEXT_NOT_STR = 0
comptime P3_TEXT_FALLBACK = -1


struct P3TextCodes(Movable):
    """`status` P3_TEXT_*; on P3_TEXT_OK `first[c]` is the row of the first
    label of class c (classes in Python's str order) and `codes[i]` the
    dense class of row i."""
    var status: Int
    var first: List[Int]
    var codes: List[Int32]

    def __init__(out self, status: Int):
        self.status = status
        self.first = List[Int]()
        self.codes = List[Int32]()


struct P3Utf8(Movable):
    """The UTF-8 bytes (address, length) of n str labels, read in place."""
    var addr: List[Int]
    var size: List[Int]

    def __init__(out self, n: Int):
        self.addr = List[Int](capacity=n)
        self.size = List[Int](capacity=n)


def _p3_bytes_cmp(a: Int, la: Int, b: Int, lb: Int) -> Int:
    var pa = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=a)
    var pb = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=b)
    for k in range(min(la, lb)):  # driver-loop: over one label's bytes
        var x = pa[k]
        var y = pb[k]
        if x != y:
            return -1 if x < y else 1
    if la == lb:
        return 0
    return -1 if la < lb else 1


def _p3_hash(a: Int, la: Int) -> UInt64:
    var p = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=a)
    var h = UInt64(14695981039346656037)
    for k in range(la):  # driver-loop: over one label's bytes
        h = (h ^ UInt64(p[k])) * UInt64(1099511628211)
    return h


def p3_utf8_read(labels: PythonObject, n: Int, mut text: P3Utf8, any_str: Bool) raises -> Int:
    """The UTF-8 views of the n labels of an exact list. P3_TEXT_NOT_STR at
    the first label that is not a str (`any_str`: a str subclass counts, as
    the code-point reader takes it; else exactly str); P3_TEXT_FALLBACK when
    a str has no UTF-8 form (a lone surrogate; the pending error is
    cleared). The views stay valid while the list holds the labels."""
    var t = cpython_table()
    var str_word = Int(native_loaded_module("builtins").str._obj_ptr)
    var size = List[Int](length=1, fill=0)
    var size_ptr = size.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    ref cpy = Python().cpython()
    for i in range(n):  # driver-loop: one borrowed read and one C call per label, no Python call
        var raw = p3_item(labels, i, False)
        var cls = cpy.PyObject_Type(raw)
        var ok = Int(cls) == str_word
        if not ok and any_str:
            ok = (t[].PyType_GetFlags(cls) & UInt64(P3_UNICODE_FLAG)) != 0
        cpy.Py_DecRef(cls)
        if not ok:
            return P3_TEXT_NOT_STR
        var address = t[].PyUnicode_AsUTF8AndSize(raw, Int(size_ptr))
        if address == 0:
            t[].PyErr_Clear()
            return P3_TEXT_FALLBACK
        text.addr.append(address)
        text.size.append(size[0])
    _ = size^
    _ = labels  # the list owns every str whose bytes were read above
    return P3_TEXT_OK


struct P3TextSet(Movable):
    """Distinct labels by hash: `rows[d]` the first row of distinct label d,
    `slots` the open-addressing table of distinct ids (-1 empty)."""
    var rows: List[Int]
    var hashes: List[UInt64]
    var slots: List[Int]
    var mask: Int

    def __init__(out self, n: Int):
        var capacity = 16
        while capacity < 2 * n:  # driver-loop: power-of-two table size (log2 n steps)
            capacity *= 2
        self.rows = List[Int]()
        self.hashes = List[UInt64]()
        self.slots = List[Int](length=capacity, fill=-1)
        self.mask = capacity - 1

    def find_or_add(mut self, text: P3Utf8, row: Int, add: Bool) -> Int:
        """The distinct id of `text[row]`; a new id when unseen and `add`,
        else -1."""
        var a = text.addr[row]
        var la = text.size[row]
        var h = _p3_hash(a, la)
        var slot = Int(h & UInt64(self.mask))
        for _ in range(len(self.slots)):  # driver-loop: open-addressing probe (expected O(1); bounded by the table size)
            var d = self.slots[slot]
            if d < 0:
                if not add:
                    return -1
                d = len(self.rows)
                self.rows.append(row)
                self.hashes.append(h)
                self.slots[slot] = d
                return d
            if self.hashes[d] == h:
                var r = self.rows[d]
                if _p3_bytes_cmp(text.addr[r], text.size[r], a, la) == 0:
                    return d
            slot = (slot + 1) & self.mask
        return -1  # unreachable: the table is at most half full


def p3_sort_distinct(text: P3Utf8, rows: List[Int]) -> List[Int]:
    """The distinct ids 0..k-1 sorted by their bytes (bottom-up merge sort;
    the ids are distinct, so order ties cannot occur)."""
    var k = len(rows)
    var order = List[Int](capacity=k)
    for d in range(k):  # driver-loop: over the k distinct labels
        order.append(d)
    var buffer = List[Int](length=k, fill=0)
    var width = 1
    while width < k:  # driver-loop: over merge widths
        var lo = 0
        while lo < k:  # driver-loop: over merged runs
            var mid = min(lo + width, k)
            var hi = min(lo + 2 * width, k)
            var i = lo
            var j = mid
            var o = lo
            while i < mid and j < hi:  # driver-loop: over one merge
                var ri = rows[order[i]]
                var rj = rows[order[j]]
                if _p3_bytes_cmp(text.addr[rj], text.size[rj], text.addr[ri], text.size[ri]) < 0:
                    buffer[o] = order[j]
                    j += 1
                else:
                    buffer[o] = order[i]
                    i += 1
                o += 1
            while i < mid:  # driver-loop: over one merge's tail
                buffer[o] = order[i]
                i += 1
                o += 1
            while j < hi:  # driver-loop: over one merge's tail
                buffer[o] = order[j]
                j += 1
                o += 1
            lo = hi
        for s in range(k):  # driver-loop: over the k distinct labels
            order[s] = buffer[s]
        width *= 2
    return order^


def p3_text_codes(labels: PythonObject, any_str: Bool = True) raises -> P3TextCodes:
    """`sorted(dict.fromkeys(labels))` and each label's dense code for an
    exact nonempty list of str labels, by hash then a sort of the k
    distinct labels only (see the file header)."""
    var n = len(labels)
    if n == 0:
        return P3TextCodes(P3_TEXT_NOT_STR)
    var text = P3Utf8(n)
    var status = p3_utf8_read(labels, n, text, any_str)
    if status != P3_TEXT_OK:
        return P3TextCodes(status)
    var seen = P3TextSet(n)
    var ids = List[Int32](capacity=n)
    for i in range(n):  # driver-loop: one hash probe per label, native
        ids.append(Int32(seen.find_or_add(text, i, True)))
    var order = p3_sort_distinct(text, seen.rows)
    var k = len(order)
    var rank = List[Int32](length=k, fill=0)
    var out = P3TextCodes(P3_TEXT_OK)
    out.first = List[Int](capacity=k)
    for c in range(k):  # driver-loop: over the k sorted classes
        rank[order[c]] = Int32(c)
        out.first.append(seen.rows[order[c]])
    out.codes = List[Int32](capacity=n)
    for i in range(n):  # driver-loop: native code write per label
        out.codes.append(rank[Int(ids[i])])
    _ = labels  # the list owns the str bytes read above
    return out^


def p3_text_classes(labels: PythonObject, first: List[Int]) raises -> PythonObject:
    """The class list: `labels[first[c]]` for each class, built presized."""
    ref cpy = Python().cpython()
    var t = cpython_table()
    var k = len(first)
    var out = t[].PyList_New(k)
    if not out:
        raise cpy.unsafe_get_error()
    var result = PythonObject(from_owned=out)
    for c in range(k):  # driver-loop: over the k classes
        _ = t[].PyList_SetItem(result._obj_ptr, c, cpy.Py_NewRef(p3_item(labels, first[c], False)))
    _ = labels
    return result


def p3_text_lookup(classes: PythonObject, values: PythonObject, address: Int, as_float: Bool,
                   missing: PythonObject) raises -> Int:
    """`{c: i for i, c in enumerate(classes)}.get(v, -1)` per value for an
    exact list of str classes (a duplicated class keeps its LAST index): the
    codes written to `address` (float32 when `as_float`, else int32), the
    first five missing values appended to `missing`; returns the missing
    count, or -1 when the classes cannot take this path (a class
    is not a str, or has no UTF-8 form: the caller runs its former path)."""
    var k = len(classes)
    var ctext = P3Utf8(k)
    var status = p3_utf8_read(classes, k, ctext, True)
    if status != P3_TEXT_OK:
        return -1
    var seen = P3TextSet(k)
    var answer = List[Int]()
    for c in range(k):  # driver-loop: over the k classes
        var d = seen.find_or_add(ctext, c, True)
        if d == len(answer):
            answer.append(c)
        else:
            answer[d] = c  # a duplicated class keeps its last index
    var t = cpython_table()
    ref cpy = Python().cpython()
    var str_word = Int(native_loaded_module("builtins").str._obj_ptr)
    var size = List[Int](length=1, fill=0)
    var size_ptr = size.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    var n = len(values)
    var one = P3Utf8(1)
    one.addr.append(0)
    one.size.append(0)
    var count = 0
    for i in range(n):  # driver-loop: one borrowed read and one hash probe per value, no Python call
        var raw = p3_item(values, i, False)
        var cls = cpy.PyObject_Type(raw)
        var is_str = Int(cls) == str_word or (t[].PyType_GetFlags(cls) & UInt64(P3_UNICODE_FLAG)) != 0
        cpy.Py_DecRef(cls)
        var found = -1
        if is_str:
            var a = t[].PyUnicode_AsUTF8AndSize(raw, Int(size_ptr))
            if a == 0:
                # No UTF-8 form (a lone surrogate): it equals no class here,
                # since every class has one.
                t[].PyErr_Clear()
            else:
                one.addr[0] = a
                one.size[0] = size[0]
                var d = _p3_probe(seen, ctext, one)
                if d >= 0:
                    found = answer[d]
        if found < 0:
            count += 1
            if len(missing) < 5:
                _ = t[].PyList_Append(missing._obj_ptr, raw)
        if as_float:
            MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=address)[i] = Float32(found)
        else:
            MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=address)[i] = Int32(found)
    _ = size^
    _ = classes
    _ = values
    return count


def _p3_probe(seen: P3TextSet, ctext: P3Utf8, one: P3Utf8) -> Int:
    """The distinct class id of the single label in `one`, or -1."""
    var a = one.addr[0]
    var la = one.size[0]
    var h = _p3_hash(a, la)
    var slot = Int(h & UInt64(seen.mask))
    for _ in range(len(seen.slots)):  # driver-loop: open-addressing probe (expected O(1); bounded by the table size)
        var d = seen.slots[slot]
        if d < 0:
            return -1
        if seen.hashes[d] == h:
            var r = seen.rows[d]
            if _p3_bytes_cmp(ctext.addr[r], ctext.size[r], a, la) == 0:
                return d
        slot = (slot + 1) & seen.mask
    return -1


# ------------------------------------------------------------ dict codes


def p3_dict_codes(index: PythonObject, values: PythonObject, address: Int, missing: Int) raises:
    """`index.get(v, missing)` per item of an exact list or tuple `values`,
    written as int32 words to `address`; `index` is an exact dict whose
    values are Python ints within int32 (a class index table). One borrowed
    read and one C dict probe per item: the item's own hash and equality
    run in C for the admitted label classes, exactly as the Python lookup
    ran them."""
    var b = native_loaded_module("builtins")
    var is_tuple = p3_type_word(values._obj_ptr) == Int(b.tuple._obj_ptr)
    ref cpy = Python().cpython()
    var t = cpython_table()
    var out = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=address)
    for i in range(len(values)):  # driver-loop: one borrowed read and one C dict probe per label
        var raw = p3_item(values, i, is_tuple)
        var hit = t[].PyDict_GetItemWithError(index._obj_ptr, raw)
        if not hit:
            if t[].PyErr_Occurred():
                raise cpy.unsafe_get_error()
            out[i] = Int32(missing)
        else:
            var code = t[].PyLong_AsLongLong(hit)
            if code == -1:
                if t[].PyErr_Occurred():
                    raise cpy.unsafe_get_error()
            out[i] = Int32(code)
    _ = index
    _ = values


# ------------------------------------------------------------ integers


def p3_integral_list(labels: PythonObject, bool_types: PythonObject) raises -> PythonObject:
    """`[int(v) for v in labels]` for an exact list or tuple whose every
    label is an admitted integral scalar (exact int, exact bool, a NumPy
    integer or bool scalar): exact ints are kept, bools become 0 / 1
    (PyObject_IsTrue), other integer scalars their exact int value
    (PyNumber_Index, the scalar's C slot). `bool_types` is the connected
    tuple of NumPy bool classes, or None. One presized list, C calls only."""
    var b = native_loaded_module("builtins")
    var is_tuple = p3_type_word(labels._obj_ptr) == Int(b.tuple._obj_ptr)
    var n = len(labels)
    var int_word = Int(b.int._obj_ptr)
    var bool_word = Int(b.bool._obj_ptr)
    var bools = List[Int]()
    if bool_types is not None:
        for j in range(len(bool_types)):  # driver-loop: over the connected bool classes
            bools.append(Int(bool_types[j]._obj_ptr))
    ref cpy = Python().cpython()
    var t = cpython_table()
    var out = t[].PyList_New(n)
    if not out:
        raise cpy.unsafe_get_error()
    var result = PythonObject(from_owned=out)
    for i in range(n):  # driver-loop: one borrowed read and one C conversion per label, no Python call
        var raw = p3_item(labels, i, is_tuple)
        var cls = p3_type_word(raw)
        var value: PyObjectPtr
        if cls == int_word:
            value = cpy.Py_NewRef(raw)
        else:
            var is_bool = cls == bool_word
            for j in range(len(bools)):  # driver-loop: over the connected bool classes
                is_bool = is_bool or cls == bools[j]
            if is_bool:
                var truth = Int(t[].PyObject_IsTrue(raw))
                if truth < 0:
                    raise cpy.unsafe_get_error()
                value = t[].PyLong_FromLongLong(Int64(truth))
            else:
                value = t[].PyNumber_Index(raw)
            if not value:
                raise cpy.unsafe_get_error()
        _ = t[].PyList_SetItem(result._obj_ptr, i, value)
    _ = labels
    return result


# ------------------------------------------------------------ arrays


def p3_array_numbers(data: Int, sizes: List[Int], steps: List[Int], count: Int, kind: String,
                     width: Int) raises -> PythonObject:
    """The Python numbers of an integer ('i' / 'u', width 1, 2, 4 or 8;
    unsigned at most 4) or float ('f', width 4 or 8) array in logical C
    order: exact ints / floats made by PyLong_FromLongLong /
    PyFloat_FromDouble into one presized list. None for any other dtype (the
    caller keeps its general scalar decoder)."""
    var integer = kind == "i" or kind == "u"
    if integer and not (width == 1 or width == 2 or width == 4 or (width == 8 and kind == "i")):
        return PythonObject(None)
    if kind == "f" and width != 4 and width != 8:
        return PythonObject(None)
    if not integer and kind != "f":
        return PythonObject(None)
    ref cpy = Python().cpython()
    var t = cpython_table()
    var out = t[].PyList_New(count)
    if not out:
        raise cpy.unsafe_get_error()
    var result = PythonObject(from_owned=out)
    var rank = len(sizes)
    for index in range(count):  # driver-loop: one native word read and one C number per element
        var residual = index
        var offset = 0
        for reverse_axis in range(rank):  # driver-loop: over the array's axes (strided and reversed views)
            var axis = rank - 1 - reverse_axis
            offset += (residual % sizes[axis]) * steps[axis]
            residual //= sizes[axis]
        var at = data + offset
        var value: PyObjectPtr
        if kind == "f":
            var x = Float64(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=at)[0]) if width == 4 else MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=at)[0]
            value = t[].PyFloat_FromDouble(x)
        else:
            var v: Int64
            if width == 1:
                v = Int64(MutPointer[Int8, MutUntrackedOrigin](unsafe_from_address=at)[0]) if kind == "i" else Int64(MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=at)[0])
            elif width == 2:
                v = Int64(MutPointer[Int16, MutUntrackedOrigin](unsafe_from_address=at)[0]) if kind == "i" else Int64(MutPointer[UInt16, MutUntrackedOrigin](unsafe_from_address=at)[0])
            elif width == 4:
                v = Int64(MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=at)[0]) if kind == "i" else Int64(MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=at)[0])
            else:
                v = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=at)[0]
            value = t[].PyLong_FromLongLong(v)
        if not value:
            raise cpy.unsafe_get_error()
        _ = t[].PyList_SetItem(result._obj_ptr, index, value)
    return result
