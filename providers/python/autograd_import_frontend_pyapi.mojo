# SPDX-License-Identifier: Apache-2.0
# Native artifact setup around the existing ONNX reader/lowering. Torch graph
# export is deliberately absent: a native call to that exporter still runs
# Python forward. The public source-kind/export dispatcher remains separate debt.
from std.python import Python, PythonObject
from providers.python.autograd_graph_metadata_pyapi import graph_record_bound
from bindings.byte_checkpoint_io import ByteCheckpointIOError
from providers.python.framework_metadata_pyapi import framework_metadata_operation
from providers.python.autograd_reader_frontend_pyapi import reader_frontend_operation
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_setattr, py_contains, py_fields
from bindings.native_loaded_module import native_loaded_module
from bindings.native_buffer_owner import native_storage_allocate
from bindings.native_json import native_path
from bindings.optimizer_public_words import stash_python_error, optimizer_python_error, PYTHON_ERROR_PENDING
from providers.python.autograd_python_errors_pyapi import graph_python_error, graph_int
from providers.python.autograd_session_frontend_pyapi import ops_session_initialize_py, ops_session_fields
from providers.python.autograd_ops_frontend_pyapi import _program_initialize
from providers.python.neural_descriptor_pyapi import NativeDescriptorResource, NeuralDescriptorProvider, neural_descriptor_execute, neural_descriptor_entry
from providers.python.autograd_import_lower_pyapi import _import_out
from providers.python.autograd_program_arrays_pyapi import program_admit_value
from autograd.program_import_request import NativeProgramImport
from sequence.neural_model_dispatch import PROGRAM_IMPORT


from providers.python.autograd_import_metadata_pyapi import _fields, _shape_words, _unsupported, _status


def _initialize(program: PythonObject, session: PythonObject, info: PythonObject,
                inputs: PythonObject, outputs: PythonObject, entries: PythonObject,
                context: PythonObject) raises:
    _ = _fields(program, context["layouts"][context["imported"]])
    _ = _fields(session, context["layouts"][context["session"]])
    var target = program
    _program_initialize(target, session, info)
    var b = native_loaded_module("builtins")
    py_setattr(target, "inputs", py_call(b.list, inputs))
    py_setattr(target, "outputs", py_call(b.list, outputs))
    py_setattr(target, "_run_native", entries["ops_imported_run"])
    py_setattr(target, "_run_device", entries["ops_imported_run_device"])


def _construct(program: PythonObject, session: PythonObject, info: PythonObject,
               inputs: PythonObject, outputs: PythonObject, context: PythonObject) raises -> PythonObject:
    _ = _fields(program, context["layouts"][context["imported"]])
    var fields = ops_session_fields(session)
    program_admit_value(info)
    program_admit_value(inputs)
    program_admit_value(outputs)
    var entries = Python.dict()
    var binding = fields["_b"]
    py_setitem(entries, "ops_imported_run", neural_descriptor_entry(binding, "ops_imported_run"))
    py_setitem(entries, "ops_imported_run_device", neural_descriptor_entry(binding, "ops_imported_run_device", False))
    # Build detached metadata so a failed constructor call cannot partly
    # overwrite an already constructed public program.
    var pending = native_storage_allocate(context["imported"])
    _initialize(pending, session, info, inputs, outputs, entries, context)
    _ = py_call(_fields(program, context["layouts"][context["imported"]]).update,
                _fields(pending, context["layouts"][context["imported"]]))
    return PythonObject(None)


def _import(source: PythonObject, label: PythonObject, shapes: PythonObject,
            mode: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    if mode is not None and py_type(mode) is not b.str:
        raise Error("ValueError: numeric_mode must be fast, identical or None")
    if label is not None and py_type(label) is not b.str:
        raise Error("TypeError: artifact label requires an exact string")
    var requested = context["backend"]._DEFAULT_MODE if mode is None else mode
    if requested is None:
        requested = context["backend"]._SELECTED
    if py_type(requested) is not b.str:
        raise Error("ValueError: numeric_mode must be fast, identical or None")
    var normalized = py_call(py_call(requested.strip).lower)
    if String(normalized) != "identical" and String(normalized) != "fast":
        raise Error("ValueError: numeric_mode must be fast, identical or None")
    var entries = context["modes"][normalized]
    if py_type(entries) is b.str:
        raise Error(String(entries))
    var words = _shape_words(shapes, context)
    var path = PythonObject(None)
    var reader = PythonObject(None)
    var name = source
    if py_type(source) is context["graph"]:
        var graph = _fields(source, context["layouts"][context["graph"]])
        reader = py_get(graph, "_native")
        if reader is None:
            raise Error("TypeError: import_model requires a graph returned by the native ONNX reader")
        name = graph["source"]
        if py_type(name) is not b.str:
            raise Error("TypeError: graph source requires an exact string")
    else:
        path = native_path(source)
        name = path if label is None else label
    var session = native_storage_allocate(context["session"])
    _ = _fields(session, context["layouts"][context["session"]])
    _ = ops_session_initialize_py(session, PythonObject(False), mode, context["connections"], context["backend"], context["program"])
    try:
        var fields = ops_session_fields(session)
        var lifetime = fields["_lifetime"]
        var owner = lifetime.downcast_value_ptr[NativeDescriptorResource]()
        var handle = 0
        if reader is not None:
            var stored = _fields(reader, context["layouts"][context["reader"]])
            if py_type(stored["h"]) is not b.int:
                raise Error("TypeError: native graph handle requires an exact integer")
            var graph_owner = stored["_lifetime"].downcast_value_ptr[NativeDescriptorResource]()
            var graph_provider = graph_owner[].provider.downcast_value_ptr[NeuralDescriptorProvider]()
            var session_provider = owner[].provider.downcast_value_ptr[NeuralDescriptorProvider]()
            if graph_provider[].module is not session_provider[].module:
                raise Error("ValueError: the graph was read by another binding than this session's; pass the ONNX file's path instead")
            if graph_owner[].handle == 0 or graph_owner[].handle != Int(py=stored["h"]):
                raise Error("TypeError: graph handle differs from its native registry owner")
            handle = graph_owner[].handle
        var names = List[String]()
        var dimensions = List[Int]()
        for i in range(len(words[0])):  # driver-loop: over admitted import input names
            names.append(String(words[0][i]))
        for i in range(len(words[1])):  # driver-loop: over admitted import rank and dimension metadata
            dimensions.append(Int(py=words[1][i]))
        # Stable initialized storage survives the native ABI call and owns the
        # returned graph metadata. No CPython call is used between native owners.
        var requests = List[NativeProgramImport]()
        requests.append(NativeProgramImport(reader is not None, handle,
            String("") if path is None else String(path), String(name), names^, dimensions^))
        var scope = owner[].context.copy()
        scope.append(owner[].handle)
        _ = neural_descriptor_execute(owner[].provider, PROGRAM_IMPORT,
            [Int(requests.unsafe_ptr())], scope, List[Float64](), False)
        var result = _import_out(requests[0].result)
        _ = reader
        _ = lifetime
        _status(result, name, context)
        var inputs = Python.list()
        var outputs = Python.list()
        for i in range(len(result[2])):  # driver-loop: over completed input descriptors
            var item = result[2][i]
            py_call(inputs.append, Python.tuple(item[0], py_call(b.tuple, item[1]), py_get(context["dtype_names"], item[2], item[2])))
        for i in range(len(result[3])):  # driver-loop: over completed output descriptors
            var item = result[3][i]
            py_call(outputs.append, Python.tuple(item[0], py_call(b.tuple, item[1]), "int32" if Int(py=item[2]) != 0 else "float32"))
        var program = native_storage_allocate(context["imported"])
        _initialize(program, session, result[1], inputs, outputs, entries, context)
        return program
    except error:
        var fields = ops_session_fields(session)
        fields["_lifetime"].downcast_value_ptr[NativeDescriptorResource]()[].close()
        py_setitem(fields, "_id", PythonObject(None))
        raise error^



def ops_import_model_c(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    var io_error = ByteCheckpointIOError()
    try:
        var args = PythonObject(from_borrowed=packed)
        if len(args) >= 2 and py_type(args[0]) is native_loaded_module("builtins").tuple:
            return graph_record_bound(args).steal_data()
        if len(args) == 4:
            return reader_frontend_operation(args[0], args[1], args[2], args[3]).steal_data()
        if len(args) == 3:
            return framework_metadata_operation(args[0], args[1], args[2], io_error).steal_data()
        if len(args) == 6:
            return _construct(args[0], args[1], args[2], args[3], args[4], args[5]).steal_data()
        if len(args) != 5:
            raise Error("TypeError: ops_import_model expects source, label, shapes, numeric_mode and context")
        return _import(args[0], args[1], args[2], args[3], args[4]).steal_data()
    except error:
        if io_error.pending:
            return io_error.restore()
        if String(error) == PYTHON_ERROR_PENDING:
            return optimizer_python_error(error)
        return graph_python_error(error)
