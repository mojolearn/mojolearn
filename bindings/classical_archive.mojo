# SPDX-License-Identifier: Apache-2.0
# Archive UNPACK/PACK: admit existing native buffers/files and retain owners.
# Format bytes, CRCs, member metadata and strided storage live in native Mojo.
from std.builtin.sort import sort
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_type, py_get, py_contains
from bindings.classical_arrays import classical_input, classical_is_array, classical_dtype_str, classical_c_bytes, classical_numpy
from bindings.classical_pandas import classical_admit_value
from bindings.sequential_admission import sequential_admit_value
from bindings.sequential_inputs import sequential_input
from bindings.native_buffer_owner import native_buffer_view
from bindings.archive_unicode_native import _archive_unicode_failure
from bindings.cpython_table import cpython_table
from core.sha256 import sha256_hex, sha256_hex_addr
from core.native_archive_span import ArchiveSpan
from core.native_archive_path import archive_filesystem_decode, archive_path_join, archive_path_normalize
from core.linear_metadata import linear_mode_name
from core.native_archive_write import (
    NativeArchiveMember, archive_contiguous_strides, archive_unicode,
    archive_linear_members, archive_prepare, archive_local, archive_central, archive_footer,
)
from bindings.native_buffer_address import native_buffer_address
from bindings.sequential_description import sequential_fields
from core.host_archive_impl import host_archive_reorder_bytes
from bindings.optimizer_public_words import stash_python_error


def _archive_file_call(entry: PythonObject, *args: PythonObject) raises -> PythonObject:
    # Fixed native IO slots may raise OSError subclasses with errno/filename
    # state. Keep their owned exception, rather than converting it to text.
    ref cpy = Python().cpython()
    var envelope = cpy.PyTuple_New(len(args))
    if not envelope:
        var failure = stash_python_error()
        raise failure^
    for i in range(len(args)):  # small-loop(args: fixed native IO entry arguments)
        _ = cpy.PyTuple_SetItem(envelope, i, cpy.Py_NewRef(args[i]._obj_ptr))
    var result = cpy.PyObject_Call(entry._obj_ptr, envelope, PyObjectPtr())
    cpy.Py_DecRef(envelope)
    if not result:
        var failure = stash_python_error()
        raise failure^
    return PythonObject(from_owned=result)


def _archive_methods(file: PythonObject, context: PythonObject) raises -> PythonObject:
    var types = context["archive_file_types"]
    var cls = py_type(file)
    for k in range(len(types)):  # driver-loop: over connected native binary file types
        if cls is types[k]:
            if k >= 2 and py_type(file.raw) is not types[1]:
                raise Error("TypeError: buffered checkpoints require a native FileIO owner, without Python raw-file callbacks")
            return context["archive_file_methods"][k]
    raise Error("TypeError: checkpoints require a native binary file or standard path; Python file protocols cannot execute")


def _archive_path_words_unpack(value: PythonObject) raises -> List[UInt32]:
    # Only exact native strings reach this boundary. Filesystem strings may
    # contain surrogateescaped bytes, so do not apply model-text admission.
    ref cpy = Python().cpython()
    var count = cpy.lib.get_function[Int]("PyUnicode_GetLength")(value._obj_ptr)
    if count < 0:
        raise cpy.unsafe_get_error()
    var result = List[UInt32](capacity=count)
    var api = cpython_table()
    for i in range(count):  # unpack: exact native filesystem Unicode storage
        result.append(api[].PyUnicode_ReadChar(value._obj_ptr, i))
    return result^


def _archive_path_words_pack(points: List[UInt32]) raises -> PythonObject:
    var result = cpython_table()[].PyUnicode_FromKindAndData(Int32(4), Int(points.unsafe_ptr()), len(points))
    if not result:
        var failure = stash_python_error()
        raise failure^
    _ = points
    return PythonObject(from_owned=result)


def _archive_path(value: PythonObject, context: PythonObject, normalize: Bool = False) raises -> PythonObject:
    # Retain the standard pathlib layouts at package connection. Mutable
    # builtins/pathlib names must not authorize a caller's Python protocols.
    var native_str = py_type(PythonObject(""))
    var cls = py_type(value)
    if cls is native_str:
        return value
    if cls is context["archive_bytes_type"]:
        var address = cpython_table()[].PyBytes_AsString(value._obj_ptr)
        if address == 0:
            raise Python().cpython().unsafe_get_error()
        var points = archive_filesystem_decode(ArchiveSpan(address, len(value)))
        var result = _archive_path_words_pack(points)
        _ = value
        return result^
    var paths = context["archive_path_types"]
    if cls is not paths[0] and cls is not paths[1]:
        raise Error("TypeError: path requires str, bytes, PosixPath or PurePosixPath; Python path callbacks are unsupported")
    var native_type = py_type(native_str)
    var mro = py_call(native_type.__dict__["__mro__"].__get__, cls)
    var bases = context["archive_path_bases"]
    var parts = PythonObject(None)
    for storage in range(2):  # small-loop(the two standard pathlib storage names)
        var field = "_raw_paths" if storage == 0 else "_parts"
        for i in range(len(mro)):  # driver-loop: connected standard pathlib ancestry
            var known = False
            for j in range(len(bases)):  # driver-loop: fixed connected native path bases
                known = known or mro[i] is bases[j]
            if not known:
                raise Error("TypeError: Python path inheritance callbacks are unsupported")
            var fields = py_call(native_type.__dict__["__dict__"].__get__, mro[i])
            if py_contains(fields, field):
                var slot = fields[field]
                if py_type(slot) is not context["archive_path_member_type"]:
                    raise Error("TypeError: Python path storage descriptors are unsupported")
                parts = py_call(slot.__get__, value, cls)
                break
        if parts is not None:
            break
    if parts is None:
        raise Error("TypeError: path requires native component storage")
    if py_type(parts) is not py_type(Python.list()) and py_type(parts) is not py_type(Python.tuple()):
        raise Error("TypeError: path components require native storage")
    var native_parts = List[List[UInt32]](capacity=len(parts))
    for i in range(len(parts)):  # driver-loop: native stored filesystem components
        var part = parts[i]
        if py_type(part) is not native_str:
            raise Error("TypeError: path components require native strings")
        native_parts.append(_archive_path_words_unpack(part))
    var points = archive_path_join(native_parts)
    if normalize:
        points = archive_path_normalize(points)
    return _archive_path_words_pack(points)


def _archive_open(path: PythonObject, context: PythonObject) raises -> PythonObject:
    var types = context["archive_file_types"]
    for k in range(len(types)):  # driver-loop: over connected native binary file types
        if py_type(path) is types[k]:
            return Python.tuple(path, False)
    var file = _archive_file_call(context["archive_open"], _archive_path(path, context), "wb")
    return Python.tuple(file, True)


def _archive_array(value: PythonObject, context: PythonObject) raises -> PythonObject:
    # Read retained native slot descriptors, not mutable Array properties.
    var slots = context["array_fields"]
    var cls = context["array_class"]
    var shape = py_call(slots["shape"], value, cls)
    var dtype = py_call(slots["dtype"], value, cls)
    var order = py_call(slots["order"], value, cls)
    var view = py_call(slots["_mv"], value, cls)
    if py_type(shape) is not py_type(Python.tuple()) or py_type(dtype) is not py_type(PythonObject("")) or py_type(order) is not py_type(PythonObject("")):
        raise Error("TypeError: checkpoint Array metadata requires native shape, dtype and order storage")
    if py_type(view) is not context["archive_memoryview_type"]:
        raise Error("TypeError: checkpoint Array storage requires a native memoryview")
    classical_admit_value(view, context)
    var descr = String(dtype)
    var width = 0
    if descr == "<f8" or descr == "<i8":
        width = 8
    elif descr == "<f4" or descr == "<i4" or descr == "<u4":
        width = 4
    elif descr == "<f2" or descr == "<u2":
        width = 2
    elif descr == "<u1" or descr == "<i1":
        width = 1
    else:
        raise Error("TypeError: checkpoint Array dtype is not native numeric storage")
    if String(order) != "C" and String(order) != "F":
        raise Error("TypeError: checkpoint Array order must be C or F")
    var count = 1
    var dimensions = List[Int]()
    var strides = List[Int]()
    var step = width
    for i in range(len(shape)):  # driver-loop: admitted Array shape and contiguous Fortran strides
        if py_type(shape[i]) is not py_type(PythonObject(0)) or Int(py=shape[i]) < 0:
            raise Error("TypeError: checkpoint Array dimensions require nonnegative native integers")
        var dimension = Int(py=shape[i])
        if dimension > 0 and step > 9223372036854775807 // dimension:
            raise Error("OverflowError: checkpoint Array shape exceeds native storage")
        count *= dimension
        dimensions.append(dimension)
        strides.append(step)
        step *= dimension
    if count > 9223372036854775807 // width or count * width != Int(py=view.nbytes):
        raise Error("ValueError: checkpoint Array storage does not match its shape")
    var raw = PythonObject(None)
    if count == 0:
        raw = py_call(context["archive_bytes_type"])
    elif String(order) == "C" or len(shape) < 2:
        raw = py_call(view.cast, "B")
    else:
        # Gather words unchanged, including signed zeros/NaN payloads, using
        # the buffer's actual address rather than mutable Array._addr metadata.
        var store = py_call(context["array_type"], "B", py_call(context["archive_bytes_type"], count * width))
        host_archive_reorder_bytes(native_buffer_address(view), dimensions, strides,
            count, width, Int(py=py_call(store.buffer_info)[0]))
        raw = native_buffer_view(store)
    _ = view^
    return Python.tuple("|u1" if descr == "<u1" else descr, shape if len(shape) else Python.tuple(1), raw)


def _archive_str_list(value: PythonObject) raises -> Bool:
    """A non-empty flat list whose every item is a str (or str subclass)."""
    if py_type(value) is not py_type(Python.list()) or len(value) == 0:
        return False
    var native_str = py_type(PythonObject(""))
    ref cpy = Python().cpython()
    var subtype = cpy.lib.get_function[Int32]("PyType_IsSubtype")
    for i in range(len(value)):  # driver-loop: over the distinct str class labels of one model
        if subtype(py_type(value[i])._obj_ptr, native_str._obj_ptr) == 0:
            return False
    return True


def _archive_class_name_unpack(cls: PythonObject) raises -> String:
    # Read the immutable type descriptor directly, without a metaclass hook
    # or a mutable builtins.type/str module attribute.
    var native_type = py_type(py_type(PythonObject("")))
    return String(py=py_call(native_type.__dict__["__name__"].__get__, cls))


def _archive_unpack_unicode(value: PythonObject, context: PythonObject) raises -> List[UInt32]:
    # Admission reads native Unicode storage; it never invokes a str subclass
    # method or creates Python byte containers for serialized metadata.
    ref cpy = Python().cpython()
    var count = cpy.lib.get_function[Int]("PyUnicode_GetLength")(value._obj_ptr)
    if count < 0:
        raise cpy.unsafe_get_error()
    var words = List[UInt32](capacity=count)
    var api = cpython_table()
    for i in range(count):  # unpack: admitted native Unicode input words
        var word = api[].PyUnicode_ReadChar(value._obj_ptr, i)
        if word >= 0xD800 and word <= 0xDFFF:
            _archive_unicode_failure(value, "utf-32-le", i, i + 1, "surrogates not allowed", context, True)
        words.append(word)
    return words^


def _archive_unpack_member(name: String, value: PythonObject, context: PythonObject,
                           mut pins: List[PythonObject]) raises -> NativeArchiveMember:
    if py_type(value) is py_type(PythonObject("")) or _archive_str_list(value):
        var strings = List[List[UInt32]]()
        if py_type(value) is py_type(PythonObject("")):
            strings.append(_archive_unpack_unicode(value, context))
        else:
            for i in range(len(value)):  # unpack: admitted flat string-label inputs
                strings.append(_archive_unpack_unicode(value[i], context))
        return archive_unicode(name, strings)
    classical_admit_value(value, context)
    var array = value if classical_is_array(value, context) or py_type(value) is py_get(context, "ndarray") else classical_input(value, context)
    pins.append(array)
    var shape = List[Int]()
    var strides = List[Int]()
    var dtype = String("")
    var width = 0
    var size = 0
    var address = 0
    if classical_is_array(array, context):
        var slots = context["array_fields"]
        var cls = context["array_class"]
        var dimensions = py_call(slots["shape"], array, cls)
        var format = py_call(slots["dtype"], array, cls)
        var order = py_call(slots["order"], array, cls)
        var view = py_call(slots["_mv"], array, cls)
        if py_type(dimensions) is not py_type(Python.tuple()) or py_type(format) is not py_type(PythonObject("")) or py_type(order) is not py_type(PythonObject("")):
            raise Error("TypeError: checkpoint Array metadata requires native shape, dtype and order storage")
        if py_type(view) is not context["archive_memoryview_type"]:
            raise Error("TypeError: checkpoint Array storage requires a native memoryview")
        classical_admit_value(view, context)
        dtype = String(format)
        if dtype == "<f8" or dtype == "<i8":
            width = 8
        elif dtype == "<f4" or dtype == "<i4" or dtype == "<u4":
            width = 4
        elif dtype == "<f2" or dtype == "<u2":
            width = 2
        elif dtype == "<u1" or dtype == "<i1":
            width = 1
        else:
            raise Error("TypeError: checkpoint Array dtype is not native numeric storage")
        if String(order) != "C" and String(order) != "F":
            raise Error("TypeError: checkpoint Array order must be C or F")
        var count = 1
        for i in range(len(dimensions)):  # unpack: checked native Array dimensions
            if py_type(dimensions[i]) is not py_type(PythonObject(0)) or Int(py=dimensions[i]) < 0:
                raise Error("TypeError: checkpoint Array dimensions require nonnegative native integers")
            var dimension = Int(py=dimensions[i])
            if dimension > 0 and count > 9223372036854775807 // dimension:
                raise Error("OverflowError: checkpoint Array shape exceeds native storage")
            count *= dimension
            shape.append(dimension)
        size = Int(py=view.nbytes)
        if count > 9223372036854775807 // width or count * width != size:
            raise Error("ValueError: checkpoint Array storage does not match its shape")
        if not len(shape):
            shape.append(1)
        strides = archive_contiguous_strides(shape, width, String(order) == "F")
        pins.append(view)
        if size:
            address = native_buffer_address(view)
        if dtype == "<u1":
            dtype = "|u1"
    else:
        # A retained memoryview pins the exact admitted ndarray's export.
        # Copy shape/strides from Py_buffer rather than creating Python
        # intermediate arrays or calling ndarray.tobytes for non-C layouts.
        var view = native_buffer_view(array)
        pins.append(view)
        dtype = classical_dtype_str(array, context)
        var exported = List[Int](length=10, fill=0)
        var raw = exported.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
        ref cpy = Python().cpython()
        if cpython_table()[].PyObject_GetBuffer(view._obj_ptr, raw, Int32(24)) != 0:
            raise cpy.unsafe_get_error()
        try:
            address = exported[0]
            size = exported[2]
            width = exported[3]
            var rank = Int(UInt64(exported[4]) >> 32)
            var dims = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=exported[6])
            var steps = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=exported[7])
            for i in range(rank):  # unpack: the admitted buffer's native layout metadata
                shape.append(dims[i])
                strides.append(steps[i])
            if rank == 0:
                shape.append(1)
                strides.append(width)
        finally:
            cpython_table()[].PyBuffer_Release(raw)
            _ = exported
        _ = view
    _ = array
    return NativeArchiveMember(name, dtype, shape^, strides^, address, size, width, List[UInt8]())


def _archive_pack_bytes(file: PythonObject, methods: PythonObject, address: Int, size: Int) raises:
    # A temporary read-only view is an output boundary object, not an
    # internal byte owner. The caller retains native storage until write
    # returns; all admitted file entries consume it synchronously in C.
    ref cpy = Python().cpython()
    var pointer = cpy.lib.get_function[PyObjectPtr]("PyMemoryView_FromMemory")(
        MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=address), size, Int32(0x100))
    if not pointer:
        raise cpy.unsafe_get_error()
    var view = PythonObject(from_owned=pointer)
    _ = _archive_file_call(methods["write"], file, view)
    _ = view


def _archive_pack_members(file: PythonObject, methods: PythonObject,
                          mut members: List[NativeArchiveMember]) raises:
    var records = archive_prepare(members)
    for i in range(len(records)):  # pack: dependent stream position and writes, in archive order
        var offset = Int(py=_archive_file_call(methods["tell"], file))
        var header = archive_local(records[i], offset)
        _archive_pack_bytes(file, methods, Int(header.unsafe_ptr()), len(header))
        _archive_pack_bytes(file, methods, members[i].source, members[i].size)
        _ = header
    var directory = Int(py=_archive_file_call(methods["tell"], file))
    for i in range(len(records)):  # pack: ordered central directory output
        var header = archive_central(records[i])
        _archive_pack_bytes(file, methods, Int(header.unsafe_ptr()), len(header))
        _ = header
    var end = Int(py=_archive_file_call(methods["tell"], file))
    var footer = archive_footer(len(records), directory, end)
    _archive_pack_bytes(file, methods, Int(footer.unsafe_ptr()), len(footer))
    _ = _archive_file_call(methods["flush"], file)
    _ = footer
    _ = records
    _ = members


def classical_archive_write(path: PythonObject, values: PythonObject, context: PythonObject) raises -> PythonObject:
    var opened = _archive_open(path, context)
    var file = opened[0]
    var methods = _archive_methods(file, context)
    var close = Bool(py=opened[1])
    var pins = List[PythonObject]()
    var members = List[NativeArchiveMember]()
    try:
        if py_type(values) is not py_type(Python.dict()):
            raise Error("TypeError: checkpoint members require a native dictionary")
        var keys = py_call(py_type(Python.list()), values.keys())
        var names = List[String]()
        for i in range(len(keys)):  # unpack: native member names from an admitted dict
            if py_type(keys[i]) is not py_type(PythonObject("")):
                raise Error("TypeError: checkpoint member names must be native strings")
            names.append(String(keys[i]))
        sort(names)
        for name in names:  # unpack: each admitted model field and its lifetime pins
            members.append(_archive_unpack_member(name, values[PythonObject(name)], context, pins))
        _archive_pack_members(file, methods, members)
    finally:
        if close:
            _ = _archive_file_call(methods["close"], file)
        _ = pins
        _ = members
    return path


def classical_archive_save(model: PythonObject, path: PythonObject, ridge: Bool,
                           context: PythonObject) raises -> PythonObject:
    var fields = sequential_fields(model)
    if not py_contains(fields, "coef_"):
        raise Error("RuntimeError: this estimator is not fitted yet")
    var mode = py_get(fields, "numeric_mode")
    if mode is None or not Bool(py=mode):
        mode = context["backend_fields"]["_DEFAULT_MODE"]
    if py_type(mode) is not py_type(PythonObject("")):
        raise Error("cannot save invalid numeric_mode")
    var source = String(mode)
    var selected = String("")
    try:
        selected = linear_mode_name(source)
    except:
        raise Error("cannot save invalid numeric_mode")
    var pins = List[PythonObject]()
    # Every save reads the current writable/rebound public coefficient field.
    var coef = _archive_unpack_member("coef", fields["coef_"], context, pins)
    var alpha = Float64(py=fields["alpha"]) if ridge else Float64(0)
    var members = archive_linear_members(coef^, _archive_class_name_unpack(py_type(model)), selected,
        Int(py=fields["n_features_in_"]), Float64(py=fields["intercept_"]), Bool(py=fields["fit_intercept"]), ridge, alpha)
    var opened = _archive_open(path, context)
    var file = opened[0]
    var methods = _archive_methods(file, context)
    try:
        _archive_pack_members(file, methods, members)
    finally:
        if Bool(py=opened[1]):
            _ = _archive_file_call(methods["close"], file)
        _ = pins
        _ = members
    return path


def classical_archive_hash(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = sequential_fields(model)
    if not py_contains(fields, "coef_"):
        raise Error("AttributeError: host linear model has no coef_")
    var value = fields["coef_"]
    if value is None:
        return PythonObject(sha256_hex(List[UInt8]()))
    sequential_admit_value(value, context)
    var pins = List[PythonObject]()
    var result = String("")
    if classical_numpy(context):
        var array = sequential_input(value, context)
        var member = _archive_unpack_member("coef", array, context, pins)
        member.prepare()
        result = sha256_hex_addr(member.source, member.size)
        _ = member
        _ = array
    else:
        # Preserve the existing NumPy-free C-order Array admission/refusal.
        var array = classical_input(value, context, "coef_")
        var raw = classical_c_bytes(array, "coef_", context)
        result = sha256_hex_addr(native_buffer_address(raw), len(raw))
        _ = raw
        _ = array
    _ = pins
    return PythonObject(result)
