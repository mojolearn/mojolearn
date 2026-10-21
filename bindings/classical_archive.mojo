# SPDX-License-Identifier: Apache-2.0
# Existing deterministic ZIP_STORED/NPY checkpoint writer. Format arithmetic,
# member order and file lifecycle are Mojo; CRC/UTF32/buffer/file entries are
# connected native C APIs. Python zipfile, numpy.savez and _serialize never run.
from std.python import Python, PythonObject
from bindings.native_json import native_path
from bindings.sequential_inputs import sequential_input
from bindings.sequential_arrays import sequential_array, sequential_address
from bindings.sequential_description import sequential_fields


def _archive_uint(bytes: PythonObject, value: UInt64, width: Int) raises:
    for k in range(width):  # driver-loop: over a fixed archive header integer's bytes
        bytes.append(Int((value >> UInt64(8 * k)) & UInt64(255)))


def _archive_header(values: PythonObject, context: PythonObject) raises -> PythonObject:
    var out = Python.import_module("builtins").bytearray()
    for i in range(len(values)):  # driver-loop: over ZIP header metadata fields
        _archive_uint(out, UInt64(py=values[i][0]), Int(py=values[i][1]))
    return out


def _archive_methods(file: PythonObject, context: PythonObject) raises -> PythonObject:
    var types = context["archive_file_types"]
    var cls = Python.import_module("builtins").type(file)
    for k in range(len(types)):  # driver-loop: over connected native binary file types
        if cls is types[k]:
            if k >= 2 and Python.import_module("builtins").type(file.raw) is not types[1]:
                raise Error("TypeError: buffered checkpoints require a native FileIO owner, without Python raw-file callbacks")
            return context["archive_file_methods"][k]
    raise Error("TypeError: checkpoints require a native binary file or standard path; Python file protocols cannot execute")


def _archive_open(path: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var types = context["archive_file_types"]
    for k in range(len(types)):  # driver-loop: over connected native binary file types
        if bi.type(path) is types[k]:
            return Python.tuple(path, False)
    var file = bi.open(native_path(path), "wb")
    return Python.tuple(file, True)


def _archive_npy(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var shape = PythonObject(None)
    var raw = PythonObject(None)
    var descr = String()
    if bi.type(value) is bi.str:
        var width = max(len(value), 1)
        descr = "<U" + String(width)
        shape = Python.tuple(1)
        raw = context["archive_utf32_encode"](value)[0]
        if len(value) == 0:
            raw = bi.bytes(4)
    else:
        var array = sequential_input(value, context)
        descr = String(array.dtype.str)
        shape = array.shape if Int(py=array.ndim) else Python.tuple(1)
        var view = bi.memoryview(array)
        raw = bi.bytes() if Int(py=array.nbytes) == 0 else (view.cast("B") if Bool(py=array.flags.c_contiguous) else view.tobytes("C"))
        # raw retains the actual exporter, including a borrowed contiguous
        # ndarray; no pointer outlives the owner passed to the native write.
        _ = array^
    var text = "{'descr': " + String(bi.repr(PythonObject(descr))) + ", 'fortran_order': False, 'shape': " + String(bi.repr(shape)) + ", }"
    var body = PythonObject(text).encode("latin1")
    var padding = 64 - ((10 + len(body) + 1) % 64)
    var length = len(body) + padding + 1
    if length >= 65536:
        raise Error("npy header too long for format 1.0")
    var header = bi.bytearray()
    header.append(147)
    header.extend(PythonObject("NUMPY").encode("ascii"))
    header.append(1)
    header.append(0)
    _archive_uint(header, UInt64(length), 2)
    header.extend(body)
    header.extend(PythonObject(" ").encode("ascii") * padding)
    header.append(10)
    return Python.tuple(header, raw)


def classical_archive_write(path: PythonObject, values: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var opened = _archive_open(path, context)
    var file = opened[0]
    var methods = _archive_methods(file, context)
    var close = Bool(py=opened[1])
    try:
        var names = bi.sorted(values.keys())
        var records = Python.list()
        var crc32 = context["archive_crc32"]
        for k in range(len(names)):  # driver-loop: over sorted deterministic NPZ member names
            var name = names[k]
            if bi.type(name) is not bi.str:
                raise Error("TypeError: checkpoint member names must be native strings")
            var filename = (name + ".npy").encode("ascii")
            var encoded = _archive_npy(values[name], context)
            var header = encoded[0]
            var payload = encoded[1]
            var size = len(header) + len(payload)
            # The incumbent streaming writer opens members without force_zip64
            # and refuses a member above zipfile.ZIP64_LIMIT on closing it.
            if size > 2147483647:
                raise Error("RuntimeError: File size too large, try using force_zip64")
            var crc = crc32(payload, crc32(header))
            var offset = Int(py=methods["tell"](file))
            var local = _archive_header(Python.list(
                Python.tuple(0x04034B50, 4), Python.tuple(20, 2), Python.tuple(0, 2), Python.tuple(0, 2),
                Python.tuple(0, 2), Python.tuple(33, 2), Python.tuple(crc, 4), Python.tuple(size, 4),
                Python.tuple(size, 4), Python.tuple(len(filename), 2), Python.tuple(0, 2)), context)
            _ = methods["write"](file, local)
            _ = methods["write"](file, filename)
            _ = methods["write"](file, header)
            _ = methods["write"](file, payload)
            records.append(Python.tuple(filename, size, crc, offset))
            _ = encoded^
        var directory = Int(py=methods["tell"](file))
        for k in range(len(records)):  # driver-loop: over NPZ central-directory member records
            var record = records[k]
            var offset = Int(py=record[3])
            var zip64 = offset > 2147483647
            var extra = bi.bytearray()
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
            _ = methods["write"](file, central)
            _ = methods["write"](file, record[0])
            if len(extra):
                _ = methods["write"](file, extra)
        var end = Int(py=methods["tell"](file))
        var directory_size = end - directory
        var count = len(records)
        if count > 65535 or directory > 2147483647 or directory_size > 2147483647:
            var wide = _archive_header(Python.list(
                Python.tuple(0x06064B50, 4), Python.tuple(44, 8), Python.tuple(45, 2), Python.tuple(45, 2),
                Python.tuple(0, 4), Python.tuple(0, 4), Python.tuple(count, 8), Python.tuple(count, 8),
                Python.tuple(directory_size, 8), Python.tuple(directory, 8)), context)
            var locator = _archive_header(Python.list(Python.tuple(0x07064B50, 4), Python.tuple(0, 4),
                                                     Python.tuple(end, 8), Python.tuple(1, 4)), context)
            _ = methods["write"](file, wide)
            _ = methods["write"](file, locator)
        var footer = _archive_header(Python.list(Python.tuple(0x06054B50, 4), Python.tuple(0, 2), Python.tuple(0, 2),
            Python.tuple(min(count, 65535), 2), Python.tuple(min(count, 65535), 2),
            Python.tuple(min(directory_size, 0xFFFFFFFF), 4), Python.tuple(min(directory, 0xFFFFFFFF), 4), Python.tuple(0, 2)), context)
        _ = methods["write"](file, footer)
        _ = methods["flush"](file)
    finally:
        if close:
            _ = methods["close"](file)
    return path


def classical_archive_save(model: PythonObject, path: PythonObject, ridge: Bool,
                           context: PythonObject) raises -> PythonObject:
    var fields = sequential_fields(model)
    if not Bool(py=fields.__contains__("coef_")):
        raise Error("RuntimeError: this estimator is not fitted yet")
    var bi = Python.import_module("builtins")
    var mode = fields.get("numeric_mode")
    if mode is None or not Bool(py=mode):
        mode = context["backend_fields"]["_DEFAULT_MODE"]
    if bi.type(mode) is not bi.str:
        raise Error("cannot save invalid numeric_mode")
    mode = mode.strip().lower()
    if String(mode) != "fast" and String(mode) != "identical":
        raise Error("cannot save invalid numeric_mode")
    var bias = sequential_array(Python.tuple(1), context["float64"], context)
    var bias_ptr = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(py=sequential_address(bias)))
    bias_ptr[0] = Float64(py=fields["intercept_"])
    var meta = sequential_array(Python.tuple(2), context["int64"], context)
    var meta_ptr = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(py=sequential_address(meta)))
    meta_ptr[0] = Int64(Int(py=fields["n_features_in_"]))
    meta_ptr[1] = 1 if Bool(py=fields["fit_intercept"]) else 0
    var values = Python.dict()
    values["format"] = "mojolearn-linear-1"
    values["estimator"] = "Ridge" if ridge else "LinearRegression"
    values["numeric_mode"] = mode
    values["coef"] = fields["coef_"]
    values["intercept"] = bias
    values["meta"] = meta
    if ridge:
        var alpha = sequential_array(Python.tuple(1), context["float64"], context)
        var ptr = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(py=sequential_address(alpha)))
        ptr[0] = Float64(py=fields["alpha"])
        values["alpha"] = alpha
    return classical_archive_write(path, values, context)
