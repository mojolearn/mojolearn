# SPDX-License-Identifier: Apache-2.0
"""Whole catalog operation over exported artifacts and retained native inputs."""
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from std.ffi import _Global
from std.time import perf_counter_ns
from core.borrowed_buffer import BorrowedBufferLayout
from core.sha256 import Sha256
from autograd.model_verify import verify_file_sha256
from bindings.native_loaded_module import native_loaded_module
from bindings.native_instance_item import native_instance_item
from bindings.native_json import native_json_read, native_path, native_path_exists
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_type, py_get, py_setitem
from bindings.causal_model_config import model_dict, model_plain, model_string, model_sequence, model_integer
from bindings.causal_checkpoint import checkpoint_array
from bindings.neural_block_admission import neural_array_slots
from providers.python.borrowed_buffer_pyapi import admit_native_buffer_exporter, AdmittedBorrowedBuffer
from providers.python.autograd_import_frontend_pyapi import _import
from providers.python.autograd_program_pyapi import ops_imported_run_py


struct CatalogConnection(Defaultable, Movable):
    var context: PythonObject

    def __init__(out self):
        self.context = PythonObject(None)


comptime CATALOG_CONNECTION = _Global[StorageType=CatalogConnection,
    name="MojolearnCatalogConnection", init_fn=CatalogConnection.__init__]


def _catalog_context() raises -> PythonObject:
    var result = CATALOG_CONNECTION.get_or_create_ptr()[].context
    if result is None:
        raise Error("RuntimeError: model catalog requires import-time connection")
    return result


def _catalog_connect(context: PythonObject) raises -> PythonObject:
    model_dict(context)
    var retained = Python.dict()
    for name in ["path", "version", "vendor", "default_mode"]:  # small-loop(static catalog identity metadata)
        py_setitem(retained, name, PythonObject(model_string(context[name])))
    for name in ["backend_fields", "import_context"]:  # small-loop(import-connected provider metadata owners)
        model_dict(context[name])
        py_setitem(retained, name, context[name])
    var cls = context["array_class"]
    if py_type(cls) is not native_loaded_module("builtins").type:
        raise Error("TypeError: catalog input requires its native Array class")
    py_setitem(retained, "array_class", cls)
    var state = CATALOG_CONNECTION.get_or_create_ptr()
    if state[].context is not None:
        raise Error("TypeError: catalog connection is already initialized")
    state[].context = retained
    return PythonObject(None)


def _catalog_load(context: PythonObject) raises -> PythonObject:
    var path = context["path"]
    if not native_path_exists(path):
        return Python.list()
    var doc = native_json_read(path)
    model_dict(doc)
    if model_string(py_get(doc, "schema")) != "mojolearn-model-catalog/1":
        raise Error("RuntimeError: the model catalog has an unsupported schema")
    var entries = model_plain(py_get(doc, "models", Python.list()))
    model_sequence(entries)
    return entries


def _catalog_entry(name: PythonObject, context: PythonObject) raises -> PythonObject:
    _ = model_string(name)
    var entries = _catalog_load(context)
    var names = Python.list()
    for i in range(len(entries)):  # driver-loop: inspect native-decoded catalog records
        var entry = entries[i]
        model_dict(entry)
        var current = py_get(entry, "name", PythonObject("?"))
        _ = model_string(current)
        if Bool(py=current == name):
            return entry
        _ = py_call(names.append, current)
    var known = String(py_call(PythonObject(", ").join, names)) if len(names) else "none"
    raise Error("KeyError: no model " + String(name) + " in the catalog (models: " + known + ")")


def _catalog_bytes(bytes: List[UInt8]) raises -> PythonObject:
    ref cpy = Python().cpython()
    var raw = cpy.lib.get_function[PyObjectPtr]("PyBytes_FromStringAndSize")(bytes.unsafe_ptr(), len(bytes))
    if not raw:
        raise cpy.unsafe_get_error()
    return PythonObject(from_owned=raw)


def _catalog_copy(layout: BorrowedBufferLayout, format: String, mut digest: Sha256) raises -> PythonObject:
    var code = "f" if format == "f" and layout.itemsize == 4 else (
        "i" if (format == "i" or format == "l") and layout.itemsize == 4 else (
        "q" if (format == "q" or format == "l") and layout.itemsize == 8 else ""))
    if code == "":
        raise Error("TypeError: catalog tensors require native float32, int32 or int64 buffers")
    var bytes = List[UInt8](length=layout.count * layout.itemsize, fill=0)
    layout.copy_to(Int(bytes.unsafe_ptr()))
    digest.update_list(bytes)
    var shape = Python.list()
    for i in range(len(layout.shape)):  # small-loop(captured tensor shape)
        _ = py_call(shape.append, layout.shape[i])
    return Python.tuple(_catalog_bytes(bytes), shape, code)


def _catalog_capture(value: PythonObject, context: PythonObject, mut digest: Sha256) raises -> PythonObject:
    # The public input's shape/address/storage metadata is captured exactly
    # once. The native importer receives a detached owned copy; hashing and
    # execution consume the same bytes even if caller-owned storage changes.
    if py_type(value) is context["array_class"]:
        var bi = native_loaded_module("builtins")
        var cls = context["array_class"]
        var slots = neural_array_slots(context)
        var fields = Python.dict()
        var names = py_call(bi.list, slots.keys())
        for i in range(len(names)):  # small-loop(capture Array native slots once, including int64 storage)
            py_setitem(fields, names[i], py_call(slots[names[i]].__get__, value, cls))
        var shape_value = fields["shape"]
        var stride_value = fields["strides"]
        if py_type(shape_value) is not bi.tuple or py_type(stride_value) is not bi.tuple or py_type(fields["_readonly"]) is not bi.bool:
            raise Error("TypeError: catalog Array shape/strides/readonly metadata changed")
        var shape = List[Int]()
        var strides = List[Int]()
        var count = 1
        var rank = model_integer(fields["ndim"])
        if rank != len(shape_value) or rank != len(stride_value):
            raise Error("TypeError: catalog Array rank differs from its metadata")
        for i in range(rank):  # small-loop(admit captured Array shape/stride words)
            var dimension = model_integer(shape_value[i])
            if dimension < 0 or (dimension != 0 and count > 9223372036854775807 // dimension):
                raise Error("ValueError: catalog Array shape exceeds native indexing")
            count *= dimension
            shape.append(dimension)
            strides.append(model_integer(stride_value[i]))
        var dtype = model_string(fields["dtype"])
        var code = "f" if dtype == "<f4" else ("i" if dtype == "<i4" else ("q" if dtype == "<i8" else ""))
        var itemsize = model_integer(fields["itemsize"])
        var expected = 8 if code == "q" else 4
        if code == "" or itemsize != expected or count > 9223372036854775807 // itemsize:
            raise Error("TypeError: catalog Array requires float32, int32 or int64 storage")
        if model_integer(fields["size"]) != count or model_integer(fields["nbytes"]) != count * itemsize:
            raise Error("ValueError: catalog Array byte size differs from shape")
        var order = model_string(fields["order"])
        if order != "C" and order != "F":
            raise Error("TypeError: catalog Array requires contiguous C or F layout")
        var step = itemsize
        for i in range(rank):  # small-loop(validate captured contiguous Array strides)
            var axis = rank - i - 1 if order == "C" else i
            if count > 0 and shape[axis] > 1 and strides[axis] != step:
                raise Error("ValueError: catalog Array strides differ from declared layout")
            step *= shape[axis]
        var address = model_integer(fields["_addr"])
        var store = fields["_store"]
        if store is None:
            store = fields["_mv"]
            if py_type(store) is not bi.memoryview or store.obj is None:
                raise Error("TypeError: catalog Array needs retained native storage")
        admit_native_buffer_exporter(store)
        var pin = AdmittedBorrowedBuffer(store)
        var result = PythonObject(None)
        try:
            var backing = pin.layout.value().copy()
            var compact = backing.contiguous()
            var f_step = backing.itemsize
            var f_compact = True
            for i in range(len(backing.shape)):  # small-loop(retained exporter F-contiguous layout)
                f_compact = f_compact and backing.suboffsets[i] < 0 and (backing.shape[i] <= 1 or backing.strides[i] == f_step)
                f_step *= backing.shape[i]
            var offset = address - backing.address if count else 0
            var capacity = backing.count * backing.itemsize
            if (not compact and not f_compact) or offset < 0 or offset > capacity or count * itemsize > capacity - offset:
                raise Error("ValueError: catalog Array lies outside its retained native storage")
            var indirect = List[Int](length=rank, fill=-1)
            var layout = BorrowedBufferLayout(address, itemsize, True, shape^, strides^, indirect^)
            result = _catalog_copy(layout, code, digest)
        except error:
            pin.release()
            raise error
        pin.release()
        _ = store
        _ = value
        return result
    admit_native_buffer_exporter(value)
    var pin = AdmittedBorrowedBuffer(value)
    var result = PythonObject(None)
    try:
        var format = pin.format
        if format.startswith(">") or format.startswith("!"):
            raise Error("TypeError: catalog buffers require native little-endian storage")
        if format.startswith("@") or format.startswith("=") or format.startswith("<"):
            format = String(format[byte=1:])
        result = _catalog_copy(pin.layout.value(), format, digest)
    except error:
        pin.release()
        raise error
    pin.release()
    _ = value
    return result


def _catalog_mode(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var mode = value
    if mode is None:
        var fields = context["backend_fields"]
        model_dict(fields)
        mode = native_instance_item(fields, PythonObject("_DEFAULT_MODE"))
        if mode is None:
            mode = native_instance_item(fields, PythonObject("_SELECTED"))
        if mode is None:
            mode = context["default_mode"]
    _ = model_string(mode)
    var normalized = py_call(py_call(mode.strip).lower)
    if String(normalized) != "identical" and String(normalized) != "fast":
        raise Error("ValueError: numeric_mode must be fast, identical or None")
    return normalized


def _catalog_verify(args: PythonObject, context: PythonObject) raises -> PythonObject:
    var name = args[0]
    var entry = _catalog_entry(name, context)
    if model_string(py_get(entry, "status")) != "verified":
        raise Error("ValueError: only verified catalog models can be checked")
    var source = args[1]
    var supplied = args[2]
    if source is None or supplied is None:
        raise Error("TypeError: verify_catalog_entry requires source=an already exported ONNX artifact or native GraphIR and inputs=the native catalog check arrays; export and seeded input generation belong in application/test setup")
    if args[4] is not None:
        _ = native_path(args[4])
    var bi = native_loaded_module("builtins")
    var input_values = supplied if py_type(supplied) is bi.list or py_type(supplied) is bi.tuple else Python.tuple(supplied)
    var inputs = Python.list()
    var input_hash = Sha256()
    for i in range(len(input_values)):  # driver-loop: capture and retain every explicit native model input
        var packed = _catalog_capture(input_values[i], context, input_hash)
        _ = py_call(inputs.append, checkpoint_array(packed[0], packed[1], String(packed[2]), context["array_class"]))
    var whole_input_hash = input_hash.hexdigest()
    var input_digest = PythonObject(String(whole_input_hash[byte=0:16]))
    var check = py_get(entry, "check")
    model_dict(check)
    var wanted = PythonObject(model_string(py_get(check, "input_digest")))
    if not Bool(py=input_digest == wanted):
        raise Error("ValueError: catalog check input digest differs from the recorded native input")
    var import_context = context["import_context"]
    var artifact = PythonObject("native GraphIR")
    var artifact_hash = PythonObject(None)
    if py_type(source) is not import_context["graph"]:
        source = native_path(source)
        artifact = source
        var fingerprint = verify_file_sha256(String(source))
        artifact_hash = PythonObject(fingerprint[0])
        # The catalog's ONNX entries previously checked the downloaded source
        # and then imported its specified opset conversion. Supplied artifacts
        # must match the recorded converted hash when present, else the source.
        if model_string(py_get(entry, "kind")) == "onnx":
            var declared_source = py_get(entry, "source")
            model_dict(declared_source)
            var expected_source = py_get(declared_source, "converted_sha256")
            if expected_source is None:
                expected_source = py_get(declared_source, "sha256")
            _ = model_string(expected_source)
            if not Bool(py=artifact_hash == expected_source):
                raise Error("ValueError: supplied ONNX artifact differs from the catalog's recorded artifact hash")
    var mode = _catalog_mode(args[3], context)
    var shapes = py_get(check, "import_shapes")
    var program = _import(source, PythonObject(None), shapes, mode, import_context)
    var started = perf_counter_ns()
    var outputs = ops_imported_run_py(program, inputs, Python.dict())
    var elapsed = Float64(perf_counter_ns() - started) / 1000000.0
    var values = outputs if py_type(outputs) is bi.tuple or py_type(outputs) is bi.list else Python.tuple(outputs)
    var output_hash = Sha256()
    for i in range(len(values)):  # driver-loop: hash completed native outputs in declared order
        _ = _catalog_capture(values[i], context, output_hash)
    var whole_hash = output_hash.hexdigest()
    var digest = PythonObject(String(whole_hash[byte=0:16]))
    var versions = py_get(entry, "versions", Python.dict())
    model_dict(versions)
    var version = context["version"]
    var modes = py_get(versions, version, Python.dict())
    if modes is None:
        modes = Python.dict()
    model_dict(modes)
    var cell = py_get(modes, mode, Python.dict())
    if cell is None:
        cell = Python.dict()
    model_dict(cell)
    var digests = py_get(cell, "digests", Python.dict())
    if digests is None:
        digests = Python.dict()
    model_dict(digests)
    var expected = PythonObject(None)
    if len(digests):
        expected = PythonObject(model_string(py_call(bi.list, digests.values())[0]))
    var matched = PythonObject(None) if expected is None else PythonObject(Bool(py=digest == expected))
    var report = Python.dict(name=name, version=version, numeric_mode=mode, vendor=context["vendor"],
        digest=digest, expected=expected, match=matched, input_digest=input_digest, run_ms=PythonObject(elapsed),
        source=Python.dict(artifact=artifact, sha256=artifact_hash, catalog=py_get(entry, "source")))
    _ = source
    _ = supplied
    _ = inputs
    _ = outputs
    _ = program
    return report


def catalog_public_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=args_ptr)
        var operation = model_string(args[0])
        var values = args[1]
        model_sequence(values)
        var result = PythonObject(None)
        if operation == "connect":
            result = _catalog_connect(args[2])
        elif operation == "catalog":
            result = _catalog_load(_catalog_context())
        elif operation == "verify":
            if len(values) != 5:
                raise Error("TypeError: catalog verification requires five argument words")
            result = _catalog_verify(values, _catalog_context())
        else:
            raise Error("ValueError: unknown catalog operation")
        return result.steal_data()
    except error:
        var message = String(error)
        if message.startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        if message.startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        if message.startswith("KeyError:"):
            return raise_python_exception(error_without_class(error, "KeyError"), ExceptionType("PyExc_KeyError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
