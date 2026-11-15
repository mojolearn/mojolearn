# SPDX-License-Identifier: Apache-2.0
# The npz/npy codec of `mojolearn._serialize` (lane py-archive, 2026-11-15).
#
# `_serialize.read_npz`, `peek_npz`, `write_npz`, `encode_npy`, `write_npy`,
# `decode_npy`, `_parse_header`, `scalar_str` and `exact` are one native call
# each (`classical_native_call(None, "archive_npz", (verb, ...), context)`).
# The ZIP/NPY framing, header grammar, dtype plan, payload bounds and text
# decoding are the native archive codec the public loaders already use
# (core/native_archive_{read,npy,write}.mojo, bindings/classical_archive_*).
# This door only admits the Python values a caller hands over (UNPACK), picks
# the member description `_serialize._describe` picked (dtype, shape, bytes),
# and packs the results. The refusals keep `_serialize`'s wording and
# exception types. No Python codec, `ast`, `struct` or `zipfile` code runs.
#
# Context: the save context of `_native_archive_bridge.connect()` (ndarrays
# admitted when NumPy is loaded, as the schema writer does); reads use its
# `host_archive_context`, which has no NumPy keys, so loads return Arrays.
from std.builtin.sort import sort
from std.memory import bitcast
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_cpython_table import cpython_table
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.cpython_borrowed_pack import borrowed_item
from bindings.optimizer_public_words_unpack import stash_python_error, PYTHON_ERROR_PENDING
from bindings.native_buffer_owner_pack import native_buffer_view
from bindings.classical_arrays_door import classical_is_array, classical_wrap_layout
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.classical_archive_door import (
    _archive_open, _archive_methods, _archive_file_call, _archive_pack_bytes, _archive_pack_members,
    _archive_unpack_member, _archive_unpack_unicode, _archive_array,
)
from bindings.classical_archive_read_door import (
    classical_archive_read, _archive_read_input, _archive_read_repr, _archive_dtype_name,
)
from bindings.classical_archive_output_door import archive_member_pack, archive_warning_pack, archive_events_pack
from bindings.classical_archive_read_run import _archive_missing
from providers.python.array_layout_boundary_pack import native_shape_pack
from core.native_archive_span import ArchiveSpan
from core.native_archive_npy import native_npy_layout_collect
from core.native_archive_read import NativeArchiveDecoded, native_archive_read_status, native_archive_text_cells
from core.native_archive_write import NativeArchiveMember, archive_owned, archive_uint, archive_unicode, archive_contiguous_strides
from core.native_array_layout import NativeArrayLayout, native_array_code


# ------------------------------------------------------------------ helpers


def _npz_is(value: PythonObject, cls: PythonObject) raises -> Bool:
    """isinstance(value, cls) for a builtin class: PyType_IsSubtype on the
    value's type, no __instancecheck__ hook."""
    return cpython_table()[].PyType_IsSubtype(py_type(value)._obj_ptr, cls._obj_ptr) != 0


def _npz_shape_text(shape: List[Int]) -> String:
    """repr(tuple(shape)): "()", "(3,)", "(3, 2)"."""
    var text = String("(")
    for i in range(len(shape)):  # driver-loop: over the member rank (metadata)
        if i:
            text += ", "
        text += String(shape[i])
    if len(shape) == 1:
        text += ","
    return text + ")"


def _npz_header(descr: String, fortran: Bool, shape: List[Int]) raises -> List[UInt8]:
    """The complete NPY v1.0 prefix `_serialize._header` wrote: magic,
    version, little-endian length, the dict and the 64-byte padding."""
    var text = ("{'descr': '" + descr + "', 'fortran_order': " + ("True" if fortran else "False")
                + ", 'shape': " + _npz_shape_text(shape) + ", }")
    var body = text.as_bytes()
    var hlen = len(body) + 1
    var padlen = 64 - ((10 + hlen) % 64)
    var total = hlen + padlen
    if total >= 65536:
        raise Error("ValueError: mojolearn: npy header too long for format 1.0")
    var header = List[UInt8](capacity=10 + total)
    header.append(147)
    for byte in [78, 85, 77, 80, 89]:  # "NUMPY"
        header.append(UInt8(byte))
    header.append(1)
    header.append(0)
    archive_uint(header, UInt64(total), 2)
    for byte in body:
        header.append(byte)
    for _ in range(padlen):
        header.append(32)
    header.append(10)
    return header^


def _npz_bytes(header: List[UInt8], address: Int, size: Int) raises -> PythonObject:
    """One Python bytes object holding header + payload (the payload copied once)."""
    var total = List[UInt8](capacity=len(header) + size)
    for byte in header:
        total.append(byte)
    var source = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=address)
    for k in range(size):  # unpack: typed byte copy of the native payload (no CPython call)
        total.append(source[k])
    var result = cpython_table()[].PyBytes_FromStringAndSize(Int(total.unsafe_ptr()), len(total))
    _ = total
    if not result:
        var failure = stash_python_error()
        raise failure^
    return PythonObject(from_owned=result)


def _npz_str(value: PythonObject) raises -> PythonObject:
    """str(value) for an exact builtin scalar (C tp_str, no user hook)."""
    ref cpy = Python().cpython()
    var result = cpy.lib.get_function[PyObjectPtr]("PyObject_Str")(value._obj_ptr)
    if not result:
        var failure = stash_python_error()
        raise failure^
    return PythonObject(from_owned=result)


def _npz_error(problem: Error) -> Error:
    """A native codec refusal in `_serialize`'s words ("mojolearn: ...")."""
    var text = String(problem)
    if text == PYTHON_ERROR_PENDING or text.startswith("mojolearn: ") or text.startswith("MemoryError:"):
        return Error(text)
    return Error("ValueError: mojolearn: " + text)


# ------------------------------------------------------------------ encode


struct _NpzValue(Movable):
    var member: NativeArchiveMember
    var fortran: Bool

    def __init__(out self, var member: NativeArchiveMember, fortran: Bool):
        self.member = member^
        self.fortran = fortran

    def take(deinit self) -> NativeArchiveMember:
        """The member, consuming the description (no partially moved aggregate)."""
        return self.member^


def _npz_scalar(name: String, dtype: String, var bytes: List[UInt8], c_order: Bool) -> _NpzValue:
    var shape = List[Int]()
    if c_order:
        shape.append(1)
    var width = len(bytes)
    return _NpzValue(archive_owned(name, dtype, shape^, width, bytes^), False)


def _npz_int_word(value: PythonObject) raises -> UInt64:
    var word = cpython_table()[].PyLong_AsLongLong(value._obj_ptr)
    if word == -1 and Bool(cpython_table()[].PyErr_Occurred()):
        var failure = stash_python_error()
        raise failure^
    return bitcast[DType.uint64](word)


def _npz_float_word(value: PythonObject) raises -> UInt64:
    var real = cpython_table()[].PyFloat_AsDouble(value._obj_ptr)
    if real == -1.0 and Bool(cpython_table()[].PyErr_Occurred()):
        var failure = stash_python_error()
        raise failure^
    return bitcast[DType.uint64](real)


def _npz_byte_span(value: PythonObject) raises -> ArchiveSpan:
    var address = cpython_table()[].PyBytes_AsString(value._obj_ptr)
    if address == 0:
        var failure = stash_python_error()
        raise failure^
    # PyBytes_Size: the stored length, not a bytes subclass's __len__ hook.
    var size = Python().cpython().lib.get_function[Int]("PyBytes_Size")(value._obj_ptr)
    return ArchiveSpan(address, size)


def _npz_flatten(value: PythonObject, list_type: PythonObject, tuple_type: PythonObject,
                 mut leaves: List[PythonObject]) raises -> List[Int]:
    """`_array._flatten` of a nested list: its shape, leaves appended in C
    order. A list/tuple nests; anything else is a leaf; ragged is refused
    with the general walk's words (bindings/user_objects_native_door.mojo)."""
    var is_list = _npz_is(value, list_type)
    if not is_list and not _npz_is(value, tuple_type):
        leaves.append(value)
        return List[Int]()
    # The C sizes of the stored items, not a subclass's __len__ hook.
    var n = cpython_table()[].PyList_Size(value._obj_ptr) if is_list else cpython_table()[].PyTuple_Size(value._obj_ptr)
    if n < 0:
        var failure = stash_python_error()
        raise failure^
    if n == 0:
        return List[Int](length=1, fill=0)
    var first = _npz_flatten(PythonObject(from_borrowed=borrowed_item(value, 0, not is_list)),
                             list_type, tuple_type, leaves)
    for i in range(1, n):  # list-input: a nested metadata list read once, borrowed items (recursive leaves)
        var shape = _npz_flatten(PythonObject(from_borrowed=borrowed_item(value, i, not is_list)),
                                 list_type, tuple_type, leaves)
        if shape != first:
            raise Error("ValueError: mojolearn: nested sequence is not rectangular ("
                        + _npz_shape_text(shape) + " vs " + _npz_shape_text(first) + ")")
    var result = List[Int](length=1, fill=n)
    for dimension in first:
        result.append(dimension)
    return result^


def _npz_list(name: String, value: PythonObject, context: PythonObject) raises -> _NpzValue:
    """`_serialize._describe_list`: one dtype for the whole nested list."""
    var str_type = py_type(PythonObject(""))
    var bytes_type = context["archive_bytes_type"]
    var bool_type = py_type(PythonObject(True))
    var int_type = py_type(PythonObject(0))
    var float_type = py_type(PythonObject(Float64(0)))
    var leaves = List[PythonObject]()
    var shape = _npz_flatten(value, py_type(Python.list()), py_type(Python.tuple()), leaves)
    var n = len(leaves)
    if n == 0:
        return _NpzValue(archive_owned(name, "<f8", shape^, 8, List[UInt8]()), False)
    var strs = 0
    var raw = 0
    var bools = 0
    var ints = 0
    var floats = 0
    for leaf in leaves:  # list-input: one type word per metadata leaf (PyType_IsSubtype, no hook)
        if _npz_is(leaf, str_type):
            strs += 1
        elif _npz_is(leaf, bytes_type):
            raw += 1
        elif _npz_is(leaf, bool_type):
            bools += 1
            ints += 1
        elif _npz_is(leaf, int_type):
            ints += 1
        elif _npz_is(leaf, float_type):
            floats += 1
    if strs == n:
        var strings = List[List[UInt32]](capacity=n)
        for leaf in leaves:  # list-input: one code-point copy per text leaf
            strings.append(_archive_unpack_unicode(leaf, context))
        var member = archive_unicode(name, strings)
        member.strides = archive_contiguous_strides(shape, member.width, False)
        member.shape = shape^
        return _NpzValue(member^, False)
    if raw == n:
        var width = 1
        for leaf in leaves:  # list-input: the widest bytes leaf
            width = max(width, _npz_byte_span(leaf).size)
        var bytes = List[UInt8](length=n * width, fill=0)
        for i in range(n):  # list-input: one copy per bytes leaf, zero padded
            var span = _npz_byte_span(leaves[i])
            var source = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=span.address)
            for k in range(span.size):
                bytes[i * width + k] = source[k]
        return _NpzValue(archive_owned(name, "|S" + String(width), shape^, width, bytes^), False)
    if bools == n:
        var bytes = List[UInt8](capacity=n)
        for leaf in leaves:  # list-input: one truth byte per bool leaf
            bytes.append(UInt8(1) if leaf is PythonObject(True) else UInt8(0))
        return _NpzValue(archive_owned(name, "|b1", shape^, 1, bytes^), False)
    if floats > 0 and ints + floats == n:
        var bytes = List[UInt8](capacity=n * 8)
        for leaf in leaves:  # list-input: one double per numeric leaf (PyFloat_AsDouble)
            archive_uint(bytes, _npz_float_word(leaf), 8)
        return _NpzValue(archive_owned(name, "<f8", shape^, 8, bytes^), False)
    if ints == n:
        var bytes = List[UInt8](capacity=n * 8)
        for leaf in leaves:  # list-input: one int64 word per integer leaf (PyLong_AsLongLong)
            archive_uint(bytes, _npz_int_word(leaf), 8)
        return _NpzValue(archive_owned(name, "<i8", shape^, 8, bytes^), False)
    raise Error("TypeError: mojolearn: cannot serialize a list of mixed kinds")


def _npz_value(name: String, value: PythonObject, c_order: Bool, context: PythonObject,
               mut pins: List[PythonObject]) raises -> _NpzValue:
    """`_serialize._describe`: (descr, fortran_order, shape, raw) of one
    value, as a native member. With `c_order` the bytes are linearized to C
    order and a 0-d value becomes shape (1,)."""
    if classical_is_array(value, context):
        var slots = context["array_fields"]
        var cls = context["array_class"]
        var rank = len(py_call(slots["shape"], value, cls))
        var order = String(py=py_call(slots["order"], value, cls))
        var described = _npz_buffer(name, value, c_order, rank, order == "F" and rank > 1, context, pins)
        return described^
    if py_type(value) is py_type(PythonObject(True)):
        var bytes = List[UInt8](length=1, fill=UInt8(1) if value is PythonObject(True) else UInt8(0))
        return _npz_scalar(name, "|b1", bytes^, c_order)
    if _npz_is(value, py_type(PythonObject(0))):
        var bytes = List[UInt8]()
        archive_uint(bytes, _npz_int_word(value), 8)
        return _npz_scalar(name, "<i8", bytes^, c_order)
    if _npz_is(value, py_type(PythonObject(Float64(0)))):
        var bytes = List[UInt8]()
        archive_uint(bytes, _npz_float_word(value), 8)
        return _npz_scalar(name, "<f8", bytes^, c_order)
    if _npz_is(value, py_type(PythonObject(""))):
        var strings = List[List[UInt32]]()
        strings.append(_archive_unpack_unicode(value, context))
        var member = archive_unicode(name, strings)
        if not c_order:
            member.shape = List[Int]()
            member.strides = List[Int]()
        return _NpzValue(member^, False)
    if _npz_is(value, context["archive_bytes_type"]):
        var span = _npz_byte_span(value)
        var width = max(span.size, 1)
        var bytes = List[UInt8](length=width, fill=0)
        var source = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=span.address)
        for k in range(span.size):  # unpack: typed copy of one bytes scalar
            bytes[k] = source[k]
        _ = value
        return _npz_scalar(name, "|S" + String(width), bytes^, c_order)
    if _npz_is(value, py_type(Python.list())) or _npz_is(value, py_type(Python.tuple())):
        return _npz_list(name, value, context)
    # A buffer-protocol object (a NumPy array when the caller has NumPy).
    var view = native_buffer_view(value)
    var rank = Int(py=view.ndim)
    var f_only = rank > 1 and not Bool(py=view.c_contiguous) and Bool(py=view.f_contiguous)
    pins.append(view)
    var described = _npz_buffer(name, value, c_order, rank, f_only, context, pins)
    return described^


def _npz_buffer(name: String, value: PythonObject, c_order: Bool, rank: Int, fortran: Bool,
                context: PythonObject, mut pins: List[PythonObject]) raises -> _NpzValue:
    var member = _archive_unpack_member(name, value, context, pins)
    if member.dtype == "<u1":
        member.dtype = "|u1"
    if c_order:
        return _NpzValue(member^, False)
    if rank == 0:
        member.shape = List[Int]()
        member.strides = List[Int]()
    # An F-order block keeps fortran_order True and its stored bytes, as
    # numpy.lib.format.write_array does; everything else is C order.
    return _NpzValue(member^, fortran)


def _npz_encoded(mut value: _NpzValue) raises -> List[UInt8]:
    """The npy header of a described value; its payload is ready at
    (member.source, member.size) after this call."""
    if not value.fortran:
        value.member.prepare()
    return _npz_header(value.member.dtype, value.fortran, value.member.shape)


def _npz_encode(value: PythonObject, c_order: Bool, context: PythonObject) raises -> PythonObject:
    var pins = List[PythonObject]()
    var described = _npz_value("", value, c_order, context, pins)
    var header = _npz_encoded(described)
    var result = _npz_bytes(header, described.member.source, described.member.size)
    _ = described
    _ = pins
    return result^


def _npz_write_npy(file: PythonObject, value: PythonObject, c_order: Bool, context: PythonObject) raises:
    # The header and the model-sized payload are separate writes: no joined copy.
    var methods = _archive_methods(file, context)
    var pins = List[PythonObject]()
    var described = _npz_value("", value, c_order, context, pins)
    var header = _npz_encoded(described)
    _archive_pack_bytes(file, methods, Int(header.unsafe_ptr()), len(header))
    _archive_pack_bytes(file, methods, described.member.source, described.member.size)
    _ = header
    _ = described
    _ = pins


def _npz_write(path: PythonObject, arrays: PythonObject, context: PythonObject) raises -> PythonObject:
    """`write_npz`: sorted member names, C order, zip epoch timestamps
    (the deterministic writer the schema save uses)."""
    if py_type(arrays) is not py_type(Python.dict()):
        raise Error("TypeError: mojolearn: write_npz requires a dict of name to array")
    var keys = py_call(py_type(Python.list()), arrays.keys())
    var names = List[String]()
    for i in range(len(keys)):  # driver-loop: over the archive member names (model fields)
        if py_type(keys[i]) is not py_type(PythonObject("")):
            raise Error("TypeError: mojolearn: npz member names must be str")
        names.append(String(py=keys[i]))
    sort(names)
    var pins = List[PythonObject]()
    var members = List[NativeArchiveMember]()
    for name in names:  # driver-loop: over the archive members (model fields)
        var described = _npz_value(name, arrays[PythonObject(name)], True, context, pins)
        members.append(described^.take())
    var opened = _archive_open(path, context)
    var file = opened[0]
    var methods = _archive_methods(file, context)
    var close = Bool(py=opened[1])
    try:
        _archive_pack_members(file, methods, members)
    finally:
        if close:
            _ = _archive_file_call(methods["close"], file)
        _ = pins
        _ = members
    return path


# ------------------------------------------------------------------ decode


def _npz_member(data: PythonObject, archive: PythonObject, mut view: List[Int]) raises -> NativeArchiveDecoded:
    """The decoded layout of one complete npy byte string; `view` holds the
    exported buffer the payload borrows (released by the caller)."""
    ref cpy = Python().cpython()
    var pointer = view.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    if cpython_table()[].PyObject_GetBuffer(data._obj_ptr, pointer, Int32(0)) != 0:
        raise cpy.unsafe_get_error()
    var member = NativeArchiveDecoded()
    try:
        var minor = Int(py=archive["archive_python_minor"])
        var micro = Int(py=archive["archive_python_micro"])
        member.layout = native_npy_layout_collect(ArchiveSpan(view[0], view[2]), minor, micro, member.warnings)
        native_archive_text_cells(member)
    except problem:
        cpython_table()[].PyBuffer_Release(pointer)
        raise _npz_error(problem)
    return member^


def _npz_decode(data: PythonObject, archive: PythonObject) raises -> PythonObject:
    var view = List[Int](length=10, fill=0)
    var members = List[NativeArchiveDecoded]()
    members.append(_npz_member(data, archive, view))
    var pointer = view.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    try:
        archive_events_pack(members, archive)
        var result = archive_member_pack(members[0], archive)
        return result^
    finally:
        cpython_table()[].PyBuffer_Release(pointer)
        _ = members
        _ = view
        _ = data


def _npz_parse_header(data: PythonObject, archive: PythonObject) raises -> PythonObject:
    """`(descr, fortran_order, shape, data_offset)` of an npy byte string."""
    var view = List[Int](length=10, fill=0)
    var member = _npz_member(data, archive, view)
    var pointer = view.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var descr = member.layout.dtype
    if member.layout.kind == 1:
        descr = "<U" + String(member.layout.width // 4)
    elif member.layout.kind == 2:
        descr = "|S" + String(member.layout.width)
    var offset = member.layout.payload.address - view[0]
    cpython_table()[].PyBuffer_Release(pointer)
    _ = view
    _ = data
    return Python.tuple(PythonObject(descr), PythonObject(member.layout.fortran),
                        native_shape_pack(member.layout.shape), PythonObject(offset))


def _npz_wanted(names: PythonObject) raises -> List[String]:
    var wanted = List[String]()
    for item in names:  # driver-loop: over the routing member names a loader asks for
        if py_type(item) is not py_type(PythonObject("")):
            raise Error("TypeError: mojolearn: peek_npz member names must be str")
        wanted.append(String(py=item))
    return wanted^


def _npz_peek(path: PythonObject, names: PythonObject, archive: PythonObject) raises -> PythonObject:
    """Only the named members, for loader routing (no other member is packed)."""
    var wanted = _npz_wanted(names)
    var admitted = _archive_read_input(path, archive)
    var minor = Int(py=archive["archive_python_minor"])
    var micro = Int(py=archive["archive_python_micro"])
    try:
        var members = native_archive_read_status(admitted.span, minor, micro)
        var result = Python.dict()
        for i in range(len(members)):  # driver-loop: over the archive members (model fields)
            var keep = False
            for name in wanted:  # driver-loop: over the requested member names
                keep = keep or name == members[i].name
            if not keep:
                continue
            for warning in members[i].warnings:
                archive_warning_pack(warning, archive)
            if members[i].error.byte_length():
                raise Error(members[i].error)
            py_setitem(result, PythonObject(members[i].name), archive_member_pack(members[i], archive))
        _ = members
        return result^
    finally:
        admitted.close()


# ---------------------------------------------------------- field readers


def _npz_first(value: PythonObject, name: String, archive: PythonObject) raises -> PythonObject:
    """str() of a member's first element in flat order (`_serialize._first`)."""
    var current = value
    if classical_is_array(current, archive):
        var slots = archive["array_fields"]
        var cls = archive["array_class"]
        var shape = py_call(slots["shape"], current, cls)
        var count = 1
        for i in range(len(shape)):  # driver-loop: over the member rank (metadata)
            count *= Int(py=shape[i])
        if count == 0:
            raise Error("ValueError: mojolearn: model field '" + name + "' is empty")
        var dtype = String(py=py_call(slots["dtype"], current, cls))
        var address = Int(py=py_call(slots["_addr"], current, cls))
        var element = PythonObject(None)
        if dtype == "<f8":
            element = PythonObject(MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=address)[0])
        elif dtype == "<f4":
            element = PythonObject(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=address)[0].cast[DType.float64]())
        elif dtype == "<f2":
            element = PythonObject(MutPointer[Float16, MutUntrackedOrigin](unsafe_from_address=address)[0].cast[DType.float64]())
        elif dtype == "<i8":
            element = PythonObject(Int(MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=address)[0]))
        elif dtype == "<i4":
            element = PythonObject(Int(MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=address)[0]))
        elif dtype == "<u4":
            element = PythonObject(Int(MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=address)[0]))
        elif dtype == "<u1":
            element = PythonObject(Int(MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=address)[0]))
        else:
            raise Error("TypeError: mojolearn: model field '" + name + "' has dtype " + dtype)
        _ = current
        return _npz_str(element)
    var list_type = py_type(Python.list())
    while py_type(current) is list_type:  # driver-loop: over the nesting depth of a stored field
        if len(current) == 0:
            raise Error("ValueError: mojolearn: model field '" + name + "' is empty")
        current = PythonObject(from_borrowed=borrowed_item(current, 0, False))
    if py_type(current) is py_type(PythonObject("")):
        return current
    if (py_type(current) is archive["archive_bytes_type"] or py_type(current) is py_type(PythonObject(0))
            or py_type(current) is py_type(PythonObject(Float64(0))) or py_type(current) is py_type(PythonObject(True))):
        return _npz_str(current)
    raise Error("TypeError: mojolearn: model field '" + name + "' is not a stored value")


def _npz_scalar_str(arrays: PythonObject, name: String, archive: PythonObject) raises -> PythonObject:
    if not py_contains(arrays, name):
        raise Error("ValueError: " + _archive_missing(name))
    var value = py_get(arrays, PythonObject(name))
    var result = _npz_first(value, name, archive)
    return result^


def _npz_exact(arrays: PythonObject, name: String, want: String, archive: PythonObject) raises -> PythonObject:
    """`arrays[name]` with its dtype REQUIRED to match, never cast; C order."""
    if not py_contains(arrays, name):
        raise Error("ValueError: " + _archive_missing(name))
    var value = py_get(arrays, PythonObject(name))
    var dtype = String("")
    var slots = archive["array_fields"]
    var cls = archive["array_class"]
    if classical_is_array(value, archive):
        dtype = String(py=py_call(slots["dtype"], value, cls))
    if dtype != want:
        raise Error("ValueError: mojolearn: model field '" + name + "' has dtype " + _archive_dtype_name(value, archive)
                    + ", the format stores " + want + "; refusing to cast a model file")
    var shape = py_call(slots["shape"], value, cls)
    if String(py=py_call(slots["order"], value, cls)) != "F" or len(shape) < 2:
        return value
    # An F-order block: one C-order copy (`Array._as_c`), gathered natively.
    var packed = _archive_array(value, archive)
    var dimensions = List[Int]()
    for i in range(len(shape)):  # driver-loop: over the member rank (metadata)
        dimensions.append(Int(py=shape[i]))
    var store = py_call(archive["array_type"], native_array_code(want))
    _ = py_call(store.frombytes, packed[2])
    var layout = NativeArrayLayout(dimensions^, Int(py=py_call(slots["itemsize"], value, cls)))
    var result = classical_wrap_layout(store, layout, want, archive)
    _ = packed
    return result^


def _npz_read(path: PythonObject, expected: PythonObject, archive: PythonObject) raises -> PythonObject:
    """`read_npz`: every member decoded natively, then the format check."""
    var accepted = List[String]()
    if py_type(expected) is py_type(PythonObject("")):
        accepted.append(String(py=expected))
    elif py_type(expected) is py_type(Python.tuple()):
        for i in range(len(expected)):  # driver-loop: over the formats a loader reads
            if py_type(expected[i]) is not py_type(PythonObject("")):
                raise Error("TypeError: mojolearn: expected_format must be a str or a tuple of str")
            accepted.append(String(py=expected[i]))
    else:
        raise Error("TypeError: mojolearn: expected_format must be a str or a tuple of str")
    var arrays = classical_archive_read(path, archive)
    var tag = PythonObject("")
    if py_contains(arrays, "format"):
        tag = _npz_scalar_str(arrays, "format", archive)
    var known = False
    for fmt in accepted:  # driver-loop: over the formats a loader reads
        known = known or fmt == String(py=tag)
    if not known:
        raise Error("ValueError: mojolearn: " + _archive_read_repr(path, archive) + " holds model format "
                    + _archive_read_repr(tag, archive) + ", this loader reads " + _archive_read_repr(expected, archive))
    return arrays^


# ------------------------------------------------------------------- entry


def archive_npz_call(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`classical_native_call(None, "archive_npz", (verb, ...), context)`."""
    if py_type(args) is not py_type(Python.tuple()) or len(args) == 0 or py_type(args[0]) is not py_type(PythonObject("")):
        raise Error("TypeError: archive_npz requires a (verb, ...) tuple")
    var verb = String(py=args[0])
    var archive = context["host_archive_context"]
    if verb == "read":
        var arrays = _npz_read(args[1], args[2], archive)
        return arrays^
    if verb == "peek":
        var routed = _npz_peek(args[1], args[2], archive)
        return routed^
    if verb == "decode":
        var decoded = _npz_decode(args[1], archive)
        return decoded^
    if verb == "header":
        var header = _npz_parse_header(args[1], archive)
        return header^
    if verb == "scalar_str":
        var text = _npz_scalar_str(args[1], String(py=args[2]), archive)
        return text^
    if verb == "exact":
        var exact = _npz_exact(args[1], String(py=args[2]), String(py=args[3]), archive)
        return exact^
    # Writers admit an ndarray when the caller has NumPy loaded.
    classical_connect_numpy(context)
    if verb == "write":
        var written = _npz_write(args[1], args[2], context)
        return written^
    if verb == "encode":
        var encoded = _npz_encode(args[1], Bool(py=args[2]), context)
        return encoded^
    if verb == "write_npy":
        _npz_write_npy(args[1], args[2], Bool(py=args[3]), context)
        return PythonObject(None)
    raise Error("TypeError: archive_npz has no verb " + verb)
