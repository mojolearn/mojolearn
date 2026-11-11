# SPDX-License-Identifier: Apache-2.0
"""The boundary read kit: ONE CPython read per argument, ONE allocation per
result (lane thin-unpack-kit, 2026-11-11; spec
~/mojolearn-evidence/briefs-20261111/_thin_unpack_shared.md, design
docs/plans/PYTHON_AT_THE_BOUNDARY.md).

UNPACK of a public operation uses these and nothing per element:
  * `buffer_view(obj)`: one `PyObject_GetBuffer` per array argument (pointer,
    shape, strides, itemsize, format, readonly), released when the view is
    destroyed or `release()`d; the exporting object is kept alive by the
    view. A dtype or layout the RUN cannot take is converted ONCE, in Mojo
    (`as_f32_c` / `as_f64_c` / `as_i64_c`); a matching C-order buffer is
    never copied.
  * `read_int` / `read_index` / `read_float` / `read_float_like` /
    `read_bool` / `read_str`: one typed C-API read per scalar parameter.
  * `read_attr_*` (one `PyObject_GetAttrString` + one read) and
    `read_item_*` (the native exact-dict lookup of
    bindings/native_instance_item.mojo + one read) for an object or a
    dict of parameters.
PACK allocates one native Array per result (`pack_room`, which RUN writes
into directly, or `pack_f32` ... `pack_u8`, one copy of a Mojo-owned result).

Exact-type admission, refusal texts, GIL ownership and borrowed-buffer
lifetimes stay the caller's: these helpers run under the GIL the binding
already holds, make no Python-level call of their own (no builtins lookup,
no Python frame), and the caller still releases the GIL around RUN exactly
where it did. A buffer exporter is admitted first by the caller's existing
admission (`admit_family`, `admit_native_buffer_exporter`); use
`admitted_buffer_view` when the caller has none.

Every function makes a bounded number of CPython calls independent of the
data size; the counts are listed on each one so a door's per-operation
total can be read off its UNPACK and PACK.
"""
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.cpython_table import cpython_table
from bindings.cpython_borrowed import borrowed_type_word
from bindings.native_instance_item import native_instance_item
from bindings.native_array_storage import native_array_zeros
from bindings.native_buffer_address import native_buffer_address
from bindings.classical_arrays import classical_wrap_layout
from providers.python.borrowed_buffer_pyapi import admit_native_buffer_exporter
from core.native_array_layout import NativeArrayLayout, native_array_code, native_array_itemsize
from core.native_string import unsafe_string_from_utf8_cstr
from core.shared_slot import SharedSlot
from core.boundary_layout import (
    BufferLayout, buffer_layout_from_words, layout_is_c_contiguous, format_kind, format_matches,
    gather_f32, gather_f64, gather_i64, PY_BUFFER_WORDS, PyBUF_RECORDS_RO, PyBUF_RECORDS, PyBUF_FORMAT,
)


# ------------------------------------------------------------ scalar types

struct _ScalarTypeWords(Defaultable, Movable):
    # Addresses of CPython's int, float, bool and str types (static,
    # immortal), 0 until first read.
    var int_word: Int
    var float_word: Int
    var bool_word: Int
    var str_word: Int

    def __init__(out self):
        self.int_word = 0
        self.float_word = 0
        self.bool_word = 0
        self.str_word = 0


comptime _SCALAR_TYPES = SharedSlot[_ScalarTypeWords, "MojolearnBoundaryScalarTypesV1", 1]


def _scalar_types() raises -> UnsafePointer[_ScalarTypeWords, MutUntrackedOrigin]:
    var words = _SCALAR_TYPES.get_or_create_ptr()
    if words[].str_word == 0:
        # Once per process: the four exact builtin classes, by identity.
        var one = PythonObject(1)
        var real = PythonObject(1.5)
        var truth = PythonObject(True)
        var text = PythonObject("")
        words[].int_word = borrowed_type_word(one._obj_ptr)
        words[].float_word = borrowed_type_word(real._obj_ptr)
        words[].bool_word = borrowed_type_word(truth._obj_ptr)
        words[].str_word = borrowed_type_word(text._obj_ptr)
        _ = one
        _ = real
        _ = truth
        _ = text
    return words


def _pending_error() -> Error:
    ref cpy = Python().cpython()
    return cpy.unsafe_get_error()


# ------------------------------------------------------------ scalar reads

def read_int(obj: PythonObject, name: StaticString = "value") raises -> Int:
    """An exact Python `int` (not bool, not a subclass) as a 64-bit word.
    CPython calls: PyObject_Type + PyLong_AsLongLong."""
    if borrowed_type_word(obj._obj_ptr) != _scalar_types()[].int_word:
        raise Error("TypeError: " + String(name) + " must be an int")
    var t = cpython_table()
    var value = t[].PyLong_AsLongLong(obj._obj_ptr)
    if value == -1 and t[].PyErr_Occurred():
        raise _pending_error()
    return Int(value)


def read_index(obj: PythonObject) raises -> Int:
    """`operator.index(obj)` as a 64-bit word (the `int(...)` admission of
    an integer parameter that accepts NumPy integers and bool, exactly as
    CPython's `__index__` protocol does). CPython calls: PyNumber_Index +
    PyLong_AsLongLong (+ the release of the index object)."""
    var t = cpython_table()
    var indexed = t[].PyNumber_Index(obj._obj_ptr)
    if not indexed:
        raise _pending_error()
    var owned = PythonObject(from_owned=indexed)
    var value = t[].PyLong_AsLongLong(owned._obj_ptr)
    if value == -1 and t[].PyErr_Occurred():
        raise _pending_error()
    return Int(value)


def read_float(obj: PythonObject) raises -> Float64:
    """`PyFloat_AsDouble(obj)`: a float, an int or any object with
    `__float__`/`__index__` (a NumPy scalar), the C-API reading of a real
    parameter. One CPython call."""
    var t = cpython_table()
    var value = t[].PyFloat_AsDouble(obj._obj_ptr)
    if value == -1.0 and t[].PyErr_Occurred():
        raise _pending_error()
    return value


def read_float_like(obj: PythonObject) raises -> Float64:
    """`float(obj)` exactly (strings included), for a door that called the
    builtin `float` on a parameter. An exact float is one
    PyFloat_AsDouble; anything else one PyNumber_Float + one read."""
    var t = cpython_table()
    if borrowed_type_word(obj._obj_ptr) == _scalar_types()[].float_word:
        return t[].PyFloat_AsDouble(obj._obj_ptr)
    var converted = t[].PyNumber_Float(obj._obj_ptr)
    if not converted:
        raise _pending_error()
    var owned = PythonObject(from_owned=converted)
    return t[].PyFloat_AsDouble(owned._obj_ptr)


def read_bool(obj: PythonObject) raises -> Bool:
    """`bool(obj)` exactly (PyObject_IsTrue). One CPython call."""
    var answer = cpython_table()[].PyObject_IsTrue(obj._obj_ptr)
    if answer < 0:
        raise _pending_error()
    return answer != 0


def read_str(obj: PythonObject, name: StaticString = "value") raises -> String:
    """An exact Python `str` as an owned UTF-8 String. CPython calls:
    PyObject_Type + PyUnicode_AsUTF8AndSize, then one native copy."""
    if borrowed_type_word(obj._obj_ptr) != _scalar_types()[].str_word:
        raise Error("TypeError: " + String(name) + " must be a str")
    var size = 0
    var address = cpython_table()[].PyUnicode_AsUTF8AndSize(obj._obj_ptr, Int(MutPointer(to=size)))
    if address == 0:
        raise _pending_error()
    # CPython's UTF-8 cache is NUL-terminated and lives as long as `obj`.
    var text = unsafe_string_from_utf8_cstr(Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=address))
    _ = obj
    if text.byte_length() != size:
        raise Error("ValueError: " + String(name) + " contains an embedded NUL character")
    return text^


# ------------------------------------------------------------ attribute / item reads

def read_attr(obj: PythonObject, name: StaticString) raises -> PythonObject:
    """`getattr(obj, name)` through one PyObject_GetAttrString (descriptor
    semantics as the attribute syntax)."""
    ref cpy = Python().cpython()
    var raw = cpy.PyObject_GetAttrString(obj._obj_ptr, String(name))
    if not raw:
        raise _pending_error()
    return PythonObject(from_owned=raw)


def read_attr_int(obj: PythonObject, name: StaticString) raises -> Int:
    return read_int(read_attr(obj, name), name)


def read_attr_index(obj: PythonObject, name: StaticString) raises -> Int:
    return read_index(read_attr(obj, name))


def read_attr_float(obj: PythonObject, name: StaticString) raises -> Float64:
    return read_float(read_attr(obj, name))


def read_attr_bool(obj: PythonObject, name: StaticString) raises -> Bool:
    return read_bool(read_attr(obj, name))


def read_attr_str(obj: PythonObject, name: StaticString) raises -> String:
    return read_str(read_attr(obj, name), name)


def read_item(fields: PythonObject, key: StaticString) raises -> PythonObject:
    """`fields[key]` of an exact dict with exact-str keys, through the
    native dictionary lookup (no `__getitem__`, `__hash__` or `__eq__` hook
    runs). One PyDict_GetItemWithError after the exact-type checks."""
    return native_instance_item(fields, PythonObject(key))


def read_item_int(fields: PythonObject, key: StaticString) raises -> Int:
    return read_int(read_item(fields, key), key)


def read_item_index(fields: PythonObject, key: StaticString) raises -> Int:
    return read_index(read_item(fields, key))


def read_item_float(fields: PythonObject, key: StaticString) raises -> Float64:
    return read_float(read_item(fields, key))


def read_item_bool(fields: PythonObject, key: StaticString) raises -> Bool:
    return read_bool(read_item(fields, key))


def read_item_str(fields: PythonObject, key: StaticString) raises -> String:
    return read_str(read_item(fields, key), key)


# ------------------------------------------------------------ array arguments

struct BufferView(Movable):
    """One exported buffer of an array argument. `ptr` is the first element;
    `strides` are in bytes. The export is released by `release()` or the
    destructor; `owner` keeps the exporting object alive until then."""
    var owner: PythonObject
    var words: List[Int]
    var acquired: Bool
    var ptr: MutPointer[UInt8, MutUntrackedOrigin]
    var itemsize: Int
    var ndim: Int
    var shape: List[Int]
    var strides: List[Int]
    var format: String
    var readonly: Bool
    var layout: BufferLayout
    var kind: Int
    var converted: List[UInt8]

    def __init__(out self, obj: PythonObject, flags: Int) raises:
        var words = List[Int](length=PY_BUFFER_WORDS, fill=0)
        var words_ptr = words.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
        if cpython_table()[].PyObject_GetBuffer(obj._obj_ptr, words_ptr, Int32(flags)) != 0:
            raise _pending_error()
        var layout: BufferLayout
        var format = String("B")
        var kind: Int
        try:
            layout = buffer_layout_from_words(words)
            if words[5] != 0:
                format = unsafe_string_from_utf8_cstr(
                    Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=words[5]))
            kind = format_kind(format, layout.itemsize)
        except problem:
            cpython_table()[].PyBuffer_Release(words_ptr)
            raise problem
        # Moving the word list keeps its heap storage: the Py_buffer the
        # exporter filled stays at `words_ptr` for PyBuffer_Release.
        self.owner = obj
        self.words = words^
        self.acquired = True
        self.converted = List[UInt8]()
        self.ptr = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=layout.address)
        self.itemsize = layout.itemsize
        self.ndim = layout.ndim
        self.shape = layout.shape.copy()
        self.strides = layout.strides.copy()
        self.readonly = layout.readonly
        self.format = format^
        self.kind = kind
        self.layout = layout^

    def _release_words(mut self):
        if self.acquired:
            self.acquired = False
            try:
                cpython_table()[].PyBuffer_Release(
                    self.words.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]())
            except:
                pass

    def release(mut self):
        """Release the export now (the end of the operation). Idempotent."""
        self._release_words()

    def __deinit__(deinit self):
        if self.acquired:
            try:
                cpython_table()[].PyBuffer_Release(
                    self.words.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]())
            except:
                pass

    def address(self) -> Int:
        return self.layout.address

    def count(self) -> Int:
        return self.layout.count

    def is_c_contiguous(self) -> Bool:
        return layout_is_c_contiguous(self.layout)

    def matches(self, typestr: StaticString) -> Bool:
        """True for a C-order buffer of exactly this NumPy typestr ("<f4")."""
        return format_matches(self.kind, self.itemsize, typestr) and self.is_c_contiguous()

    def as_f32_c(mut self) raises -> Int:
        """The address of C-order float32 data: the buffer itself when it is
        already C-order float32 (no copy), else ONE native conversion copy
        owned by this view."""
        if self.matches("<f4"):
            return self.layout.address
        self.converted = List[UInt8](length=self.layout.count * 4 if self.layout.count > 0 else 4, fill=0)
        var dst = Int(self.converted.unsafe_ptr())
        gather_f32(self.layout, self.kind, dst)
        return dst

    def as_f64_c(mut self) raises -> Int:
        """`as_f32_c` for float64."""
        if self.matches("<f8"):
            return self.layout.address
        self.converted = List[UInt8](length=self.layout.count * 8 if self.layout.count > 0 else 8, fill=0)
        var dst = Int(self.converted.unsafe_ptr())
        gather_f64(self.layout, self.kind, dst)
        return dst

    def as_i64_c(mut self) raises -> Int:
        """`as_f32_c` for int64 (integer or bool storage only)."""
        if self.matches("<i8"):
            return self.layout.address
        self.converted = List[UInt8](length=self.layout.count * 8 if self.layout.count > 0 else 8, fill=0)
        var dst = Int(self.converted.unsafe_ptr())
        gather_i64(self.layout, self.kind, dst)
        return dst


def buffer_view(obj: PythonObject, flags: Int = PyBUF_RECORDS_RO) raises -> BufferView:
    """One `PyObject_GetBuffer(obj, flags)`. The caller has admitted `obj`
    as an exact native exporter (no Python `__buffer__` hook). CPython
    calls: PyObject_GetBuffer now, PyBuffer_Release at release."""
    return BufferView(obj, flags)


def admitted_buffer_view(obj: PythonObject, writable: Bool = False) raises -> BufferView:
    """`buffer_view` after the shared exact-exporter admission
    (`admit_native_buffer_exporter`: ndarray, bytes, bytearray, memoryview
    of those, native mmap, ctypes array; a bounded number of identity
    reads, never a buffer hook)."""
    admit_native_buffer_exporter(obj)
    return BufferView(obj, PyBUF_RECORDS if writable else PyBUF_RECORDS_RO)


# ------------------------------------------------------------ results

struct PackedArray(Movable):
    """A new native Array (`array`) and the address of its first element,
    which RUN writes directly (no second copy)."""
    var array: PythonObject
    var address: Int

    def __init__(out self, var array: PythonObject, address: Int):
        self.array = array^
        self.address = address


def pack_room(typestr: String, var shape: List[Int], context: PythonObject) raises -> PackedArray:
    """ONE native Array allocation of `shape` and `typestr` ("<f4", "<f8",
    "<i4", "<i8", "<u1", ...), zero-filled, C order, for RUN to write in
    place. `context` is the family context that holds `array_class` and
    `array_storage_slots` (every family door already has it). A constant
    number of CPython calls (the store, its address, the Array slots)."""
    var layout = NativeArrayLayout(shape^, native_array_itemsize(typestr), "C")
    var store = native_array_zeros(native_array_code(typestr), layout.count)
    var address = native_buffer_address(store)
    var array = classical_wrap_layout(store, layout, typestr, context)
    return PackedArray(array^, address)


def _pack_copy(typestr: String, width: Int, src_address: Int, rows: Int, cols: Int,
               context: PythonObject) raises -> PythonObject:
    var shape = List[Int](length=1, fill=rows)
    if cols >= 0:
        shape.append(cols)
    var room = pack_room(typestr, shape^, context)
    var count = rows * (cols if cols >= 0 else 1)
    if count > 0:
        unsafe_memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=room.address),
            src=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=src_address), count=count * width)
    var result = room.array
    _ = room^
    return result


def pack_f32(src_address: Int, rows: Int, cols: Int, context: PythonObject) raises -> PythonObject:
    """ONE native Array holding a copy of a Mojo-owned C-order float32
    result of `rows` x `cols` (`cols` < 0: the 1-D shape `(rows,)`). Use
    `pack_room` instead when RUN can write the Array directly."""
    return _pack_copy("<f4", 4, src_address, rows, cols, context)


def pack_f64(src_address: Int, rows: Int, cols: Int, context: PythonObject) raises -> PythonObject:
    return _pack_copy("<f8", 8, src_address, rows, cols, context)


def pack_i32(src_address: Int, rows: Int, cols: Int, context: PythonObject) raises -> PythonObject:
    return _pack_copy("<i4", 4, src_address, rows, cols, context)


def pack_i64(src_address: Int, rows: Int, cols: Int, context: PythonObject) raises -> PythonObject:
    return _pack_copy("<i8", 8, src_address, rows, cols, context)


def pack_u8(src_address: Int, rows: Int, cols: Int, context: PythonObject) raises -> PythonObject:
    return _pack_copy("<u1", 1, src_address, rows, cols, context)
