# SPDX-License-Identifier: Apache-2.0
# Existing deterministic ZIP_STORED/NPY checkpoint writer. Format arithmetic,
# member order and file lifecycle are Mojo; CRC/UTF32/buffer/file entries are
# connected native C APIs. Python zipfile, numpy.savez and _serialize never run.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_contains
from bindings.classical_arrays import classical_input, classical_is_array, classical_typed, classical_dtype_str, classical_c_bytes, classical_address
from bindings.classical_pandas import classical_admit_value
from bindings.native_buffer_owner import native_buffer_view
from bindings.native_buffer_address import native_buffer_address
from bindings.sequential_description import sequential_fields
from core.host_archive_impl import host_archive_reorder_bytes
from bindings.optimizer_public_words import stash_python_error
from bindings.archive_posix_native import archive_posix_open, archive_posix_tell, archive_posix_write, archive_posix_close, archive_posix_discard


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


def _archive_uint(bytes: PythonObject, value: UInt64, width: Int) raises:
    for k in range(width):  # driver-loop: over a fixed archive header integer's bytes
        py_call(bytes.append, Int((value >> UInt64(8 * k)) & UInt64(255)))


def _archive_header(values: PythonObject, context: PythonObject) raises -> PythonObject:
    var out = py_call(context["archive_bytearray_type"])
    for i in range(len(values)):  # driver-loop: over ZIP header metadata fields
        _archive_uint(out, UInt64(py=values[i][0]), Int(py=values[i][1]))
    return out


def _archive_methods(file: PythonObject, context: PythonObject) raises -> PythonObject:
    var types = context["archive_file_types"]
    var cls = py_type(file)
    for k in range(len(types)):  # driver-loop: over connected native binary file types
        if cls is types[k]:
            if k >= 2 and py_type(file.raw) is not types[1]:
                raise Error("TypeError: buffered checkpoints require a native FileIO owner, without Python raw-file callbacks")
            return context["archive_file_methods"][k]
    raise Error("TypeError: checkpoints require a native binary file or standard path; Python file protocols cannot execute")


def _archive_path(value: PythonObject, context: PythonObject) raises -> PythonObject:
    # Retain the standard pathlib layouts at package connection. Mutable
    # builtins/pathlib names must not authorize a caller's Python protocols.
    var native_str = py_type(PythonObject(""))
    var cls = py_type(value)
    if cls is native_str:
        return value
    if cls is context["archive_bytes_type"]:
        return py_call(value.decode, "utf-8", "surrogateescape")
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
    var result = PythonObject("")
    for i in range(len(parts)):  # driver-loop: native stored filesystem components
        var part = parts[i]
        if py_type(part) is not native_str:
            raise Error("TypeError: path components require native strings")
        if Bool(py=py_call(part.startswith, "/")) or len(result) == 0:
            result = part
        elif len(part) != 0:
            result = py_call(result.rstrip, "/") + PythonObject("/") + part
    return result if len(result) != 0 else PythonObject(".")


def _archive_open(path: PythonObject, context: PythonObject) raises -> PythonObject:
    var types = context["archive_file_types"]
    for k in range(len(types)):  # driver-loop: over connected native binary file types
        if py_type(path) is types[k]:
            return Python.tuple(path, None)
    var native_path = _archive_path(path, context)
    var fd = archive_posix_open(native_path, context)
    try:
        return Python.tuple(fd, native_path)
    except:
        archive_posix_discard(fd)
        raise


def _archive_tell(file: PythonObject, methods: PythonObject, context: PythonObject) raises -> Int:
    if methods is None:
        return archive_posix_tell(Int(py=file), context)
    return Int(py=_archive_file_call(methods["tell"], file))


def _archive_write(file: PythonObject, methods: PythonObject, value: PythonObject, context: PythonObject) raises:
    if methods is None:
        archive_posix_write(Int(py=file), value, context)
    else:
        _ = _archive_file_call(methods["write"], file, value)


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


def _archive_npy(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var shape = PythonObject(None)
    var raw = PythonObject(None)
    var descr = String()
    if py_type(value) is py_type(PythonObject("")):
        var width = max(len(value), 1)
        descr = "<U" + String(width)
        shape = Python.tuple(1)
        raw = py_call(context["archive_utf32_encode"], value)[0]
        if len(value) == 0:
            raw = py_call(context["archive_bytes_type"], 4)
    else:
        classical_admit_value(value, context)
        var array = value if classical_is_array(value, context) or py_type(value) is py_get(context, "ndarray") else classical_input(value, context)
        if classical_is_array(array, context):
            var parts = _archive_array(array, context)
            descr = String(parts[0])
            shape = parts[1]
            raw = parts[2]
        else:
            descr = classical_dtype_str(array, context)
            shape = array.shape if Int(py=array.ndim) else Python.tuple(1)
            raw = py_call(context["archive_bytes_type"]) if Int(py=array.nbytes) == 0 else classical_c_bytes(array, "model field", context)
        # raw retains the actual exporter, including a borrowed contiguous
        # ndarray; no pointer outlives the owner passed to the native write.
        _ = array^
    ref cpy = Python().cpython()
    var dtype_text = PythonObject(descr)
    var repr_dtype = cpy.lib.get_function[PyObjectPtr]("PyObject_Repr")(dtype_text._obj_ptr)
    if not repr_dtype:
        raise cpy.unsafe_get_error()
    var dtype_word = PythonObject(from_owned=repr_dtype)
    var repr_shape = cpy.lib.get_function[PyObjectPtr]("PyObject_Repr")(shape._obj_ptr)
    if not repr_shape:
        raise cpy.unsafe_get_error()
    var shape_word = PythonObject(from_owned=repr_shape)
    var text = "{'descr': " + String(dtype_word) + ", 'fortran_order': False, 'shape': " + String(shape_word) + ", }"
    var body = py_call(PythonObject(text).encode, "latin1")
    var padding = 64 - ((10 + len(body) + 1) % 64)
    var length = len(body) + padding + 1
    if length >= 65536:
        raise Error("npy header too long for format 1.0")
    var header = py_call(context["archive_bytearray_type"])
    py_call(header.append, 147)
    py_call(header.extend, py_call(PythonObject("NUMPY").encode, "ascii"))
    py_call(header.append, 1)
    py_call(header.append, 0)
    _archive_uint(header, UInt64(length), 2)
    py_call(header.extend, body)
    py_call(header.extend, py_call(PythonObject(" ").encode, "ascii") * padding)
    py_call(header.append, 10)
    return Python.tuple(header, raw)


def classical_archive_write(path: PythonObject, values: PythonObject, context: PythonObject) raises -> PythonObject:
    var opened = _archive_open(path, context)
    var file = opened[0]
    # None methods identifies an owned POSIX descriptor. Exact native streams
    # use retained slots and stay open; no new CPython file is constructed.
    var methods = _archive_methods(file, context) if opened[1] is None else PythonObject(None)
    try:
        if py_type(values) is not py_type(Python.dict()):
            raise Error("TypeError: checkpoint members require a native dictionary")
        var names = py_call(py_type(Python.list()), values.keys())
        for i in range(len(names)):  # driver-loop: admit all member names before native comparisons
            if py_type(names[i]) is not py_type(PythonObject("")):
                raise Error("TypeError: checkpoint member names must be native strings")
        _ = py_call(names.sort)
        var records = Python.list()
        var crc32 = context["archive_crc32"]
        for k in range(len(names)):  # driver-loop: over sorted deterministic NPZ member names
            var name = names[k]
            var filename = py_call((name + ".npy").encode, "ascii")
            var encoded = _archive_npy(values[name], context)
            var header = encoded[0]
            var payload = encoded[1]
            var size = len(header) + len(payload)
            # The incumbent streaming writer opens members without force_zip64
            # and refuses a member above zipfile.ZIP64_LIMIT on closing it.
            if size > 2147483647:
                raise Error("RuntimeError: File size too large, try using force_zip64")
            var crc = py_call(crc32, payload, py_call(crc32, header))
            var offset = _archive_tell(file, methods, context)
            var local = _archive_header(Python.list(
                Python.tuple(0x04034B50, 4), Python.tuple(20, 2), Python.tuple(0, 2), Python.tuple(0, 2),
                Python.tuple(0, 2), Python.tuple(33, 2), Python.tuple(crc, 4), Python.tuple(size, 4),
                Python.tuple(size, 4), Python.tuple(len(filename), 2), Python.tuple(0, 2)), context)
            _archive_write(file, methods, local, context)
            _archive_write(file, methods, filename, context)
            _archive_write(file, methods, header, context)
            _archive_write(file, methods, payload, context)
            py_call(records.append, Python.tuple(filename, size, crc, offset))
            _ = encoded^
        var directory = _archive_tell(file, methods, context)
        for k in range(len(records)):  # driver-loop: over NPZ central-directory member records
            var record = records[k]
            var offset = Int(py=record[3])
            var zip64 = offset > 2147483647
            var extra = py_call(context["archive_bytearray_type"])
            if zip64:
                _archive_uint(extra, 1, 2)
                _archive_uint(extra, 8, 2)
                _archive_uint(extra, UInt64(offset), 8)
            var version = 45 if zip64 else 20
            var central = _archive_header(Python.list(
                Python.tuple(0x02014B50, 4), Python.tuple((3 << 8) | version, 2), Python.tuple(version, 2),
                Python.tuple(0, 2), Python.tuple(0, 2), Python.tuple(0, 2), Python.tuple(33, 2),
                Python.tuple(record[2], 4), Python.tuple(record[1], 4), Python.tuple(record[1], 4),
                Python.tuple(len(record[0]), 2), Python.tuple(len(extra), 2), Python.tuple(0, 2),
                Python.tuple(0, 2), Python.tuple(0, 2), Python.tuple(0o644 << 16, 4),
                Python.tuple(0xFFFFFFFF if zip64 else offset, 4)), context)
            _archive_write(file, methods, central, context)
            _archive_write(file, methods, record[0], context)
            if len(extra):
                _archive_write(file, methods, extra, context)
        var end = _archive_tell(file, methods, context)
        var directory_size = end - directory
        var count = len(records)
        if count > 65535 or directory > 2147483647 or directory_size > 2147483647:
            var wide = _archive_header(Python.list(
                Python.tuple(0x06064B50, 4), Python.tuple(44, 8), Python.tuple(45, 2), Python.tuple(45, 2),
                Python.tuple(0, 4), Python.tuple(0, 4), Python.tuple(count, 8), Python.tuple(count, 8),
                Python.tuple(directory_size, 8), Python.tuple(directory, 8)), context)
            var locator = _archive_header(Python.list(Python.tuple(0x07064B50, 4), Python.tuple(0, 4),
                                                     Python.tuple(end, 8), Python.tuple(1, 4)), context)
            _archive_write(file, methods, wide, context)
            _archive_write(file, methods, locator, context)
        var footer = _archive_header(Python.list(Python.tuple(0x06054B50, 4), Python.tuple(0, 2), Python.tuple(0, 2),
            Python.tuple(min(count, 65535), 2), Python.tuple(min(count, 65535), 2),
            Python.tuple(min(directory_size, 0xFFFFFFFF), 4), Python.tuple(min(directory, 0xFFFFFFFF), 4), Python.tuple(0, 2)), context)
        _archive_write(file, methods, footer, context)
        if methods is not None:
            _ = _archive_file_call(methods["flush"], file)
    finally:
        if methods is None:
            archive_posix_close(Int(py=file), context)
    return path


def classical_archive_save(model: PythonObject, path: PythonObject, ridge: Bool,
                           context: PythonObject) raises -> PythonObject:
    var fields = sequential_fields(model)
    if not py_contains(fields, "coef_"):
        raise Error("RuntimeError: this estimator is not fitted yet")
    var bi = Python.import_module("builtins")
    var mode = py_get(fields, "numeric_mode")
    if mode is None or not Bool(py=mode):
        mode = context["backend_fields"]["_DEFAULT_MODE"]
    if py_type(mode) is not bi.str:
        raise Error("cannot save invalid numeric_mode")
    mode = mode.strip().lower()
    if String(mode) != "fast" and String(mode) != "identical":
        raise Error("cannot save invalid numeric_mode")
    var bias = classical_typed(Python.tuple(1), "<f8", context)
    var bias_ptr = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(bias)))
    bias_ptr[0] = Float64(py=fields["intercept_"])
    var meta = classical_typed(Python.tuple(2), "<i8", context)
    var meta_ptr = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(meta)))
    meta_ptr[0] = Int64(Int(py=fields["n_features_in_"]))
    meta_ptr[1] = 1 if Bool(py=fields["fit_intercept"]) else 0
    var values = Python.dict()
    py_setitem(values, "format", "mojolearn-linear-1")
    py_setitem(values, "estimator", py_type(model).__name__)
    py_setitem(values, "numeric_mode", mode)
    py_setitem(values, "coef", fields["coef_"])
    py_setitem(values, "intercept", bias)
    py_setitem(values, "meta", meta)
    if ridge:
        var alpha = classical_typed(Python.tuple(1), "<f8", context)
        var ptr = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(alpha)))
        ptr[0] = Float64(py=fields["alpha"])
        py_setitem(values, "alpha", alpha)
    return classical_archive_write(path, values, context)
