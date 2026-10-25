# SPDX-License-Identifier: Apache-2.0
# Native artifact setup around the existing ONNX reader/lowering. Torch graph
# export is deliberately absent: a native call to that exporter still runs
# Python forward. The public source-kind/export dispatcher remains separate debt.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_setattr, py_contains
from bindings.native_loaded_module import native_loaded_module
from bindings.native_buffer_owner import native_storage_allocate
from bindings.native_json import native_path
from bindings.optimizer_public_words import stash_python_error, optimizer_python_error, PYTHON_ERROR_PENDING
from autograd.python_errors_pyapi import graph_python_error, graph_int
from autograd.session_frontend_pyapi import ops_session_initialize_py
from autograd.ops_frontend_pyapi import _program_initialize


def _fields(value: PythonObject, layout: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var cls = py_type(value)
    if cls is not layout["cls"] or py_type(cls) is not b.type:
        raise Error("TypeError: artifact import requires exact native storage classes")
    var mro = py_call(b.type.__dict__["__mro__"].__get__, cls)
    if len(mro) != len(layout["mro"]):
        raise Error("TypeError: artifact import class ancestry changed")
    for i in range(len(mro)):  # driver-loop: over connected class ancestry
        if mro[i] is not layout["mro"][i]:
            raise Error("TypeError: artifact import class ancestry changed")
    var hooks = layout["hooks"]
    for i in range(len(hooks)):  # driver-loop: over fixed native storage identities
        var row = py_call(b.type.__dict__["__dict__"].__get__, hooks[i][0])
        if py_get(row, hooks[i][1]) is not hooks[i][2]:
            raise Error("TypeError: artifact import refuses changed Python storage hooks")
    var fields = py_call(layout["fields_get"], value, cls)
    var keys = py_call(b.list, fields)
    for i in range(len(keys)):  # driver-loop: over safe keys before field lookup
        if py_type(keys[i]) is not b.str:
            raise Error("TypeError: artifact state requires exact string keys")
    return fields


def _one_of(cls: PythonObject, classes: PythonObject) raises -> Bool:
    for i in range(len(classes)):  # driver-loop: over fixed native metadata types
        if cls is classes[i]:
            return True
    return False


def _scalar(value: PythonObject, classes: PythonObject) raises:
    if not _one_of(py_type(value), classes):
        raise Error("TypeError: shape metadata requires native scalar values; Python conversion hooks are unsupported")


def _shape_words(shapes: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var names = Python.list()
    var flat = Python.list()
    if shapes is None:
        return Python.tuple(names, flat)
    var cls = py_type(shapes)
    if not _one_of(cls, context["shape_mappings"]) and cls is not b.list and cls is not b.tuple:
        raise Error("TypeError: shapes requires a native mapping or key/value sequence; Python mapping callbacks are unsupported")
    # Do not traverse dimension objects yet: a standard torch.Size is a native
    # tuple subtype. It is read through tuple's C iterator below, after checking
    # that its effective iterator has not been replaced by a Python hook.
    var provided = shapes
    if cls is b.list or cls is b.tuple:
        provided = Python.dict()
        for i in range(len(shapes)):  # driver-loop: over native key/value shape pairs
            var pair = shapes[i]
            if (py_type(pair) is not b.list and py_type(pair) is not b.tuple) or len(pair) != 2:
                raise Error("TypeError: shape entries require native key/value pairs")
            var key = pair[0]
            var key_type = py_type(key)
            _scalar(key, context["shape_names"])
            py_setitem(provided, key, pair[1])
    var keys = py_call(b.list, provided)
    for i in range(len(keys)):  # driver-loop: over named input shape metadata
        var key = keys[i]
        _scalar(key, context["shape_names"])
        var name = py_call(b.str, key)
        var dims = provided[key]
        var dims_type = py_type(dims)
        var tuple_storage = False
        if py_type(dims_type) is b.type:
            var mro = py_call(b.type.__dict__["__mro__"].__get__, dims_type)
            tuple_storage = py_contains(mro, b.tuple)
            if tuple_storage:
                for j in range(len(mro)):  # driver-loop: over effective tuple iterator identity
                    var fields = py_call(b.type.__dict__["__dict__"].__get__, mro[j])
                    if py_contains(fields, "__iter__"):
                        if fields["__iter__"] is not b.tuple.__dict__["__iter__"]:
                            raise Error("TypeError: dimensions refuse Python iteration hooks")
                        break
        if tuple_storage:
            dims = py_call(b.tuple, py_call(b.tuple.__iter__, dims))
        else:
            if not _one_of(dims_type, context["shape_containers"]):
                raise Error("TypeError: dimensions require native sequence storage")
            dims = py_call(b.tuple, dims)
        py_call(names.append, name)
        py_call(flat.append, len(dims))
        for j in range(len(dims)):  # driver-loop: over native scalar axis conversion
            _scalar(dims[j], context["shape_scalars"])
            py_call(flat.append, graph_int(dims[j]))
    return Python.tuple(names, flat)


def _raise_object(category: PythonObject, exception: PythonObject) raises:
    # Public CPython exception API; the original class and metadata survive
    # the C entry without calling a Python exception constructor.
    ref cpy = Python().cpython()
    cpy.lib.get_function[NoneType]("PyErr_SetObject")(category._obj_ptr, exception._obj_ptr)
    var failure = stash_python_error()
    raise failure^


def _unsupported(source: PythonObject, problems: PythonObject, context: PythonObject) raises:
    var b = native_loaded_module("builtins")
    var output = Python.list()
    var all_nodes = True
    for i in range(len(problems)):  # driver-loop: over native lowerer's refusal records
        var item = problems[i]
        py_call(output.append, Python.tuple(item[0], item[1], item[2]))
        all_nodes = all_nodes and item[1] is not None
    var noun = String("node") if all_nodes else String("item")
    if len(problems) != 1:
        noun += "s"
    var basename = py_call(source.rsplit, "/", 1)[-1]
    if len(basename) == 0:
        basename = source
    var message = String(len(problems)) + " " + noun + " of " + String(basename) + " cannot be imported:"
    for i in range(len(output)):  # driver-loop: over complete ordered refusal text
        var item = output[i]
        message += "\n  " + String(item[0])
        if item[1] is not None:
            message += " (" + String(item[1]) + ")"
        message += ": " + String(item[2])
    var category = context["unsupported"]
    var exception = py_call(b.ValueError.__new__, category, PythonObject(message))
    var fields = py_call(b.BaseException.__dict__["__dict__"].__get__, exception, category)
    py_setitem(fields, "source", source)
    py_setitem(fields, "problems", py_call(b.tuple, output))
    _raise_object(category, exception)


def _status(result: PythonObject, source: PythonObject, context: PythonObject) raises:
    var code = Int(py=result[0])
    if code == 0:
        return
    if code == 5:
        _unsupported(source, result[1], context)
    if code == 4:
        raise Error("ValueError: the model graph: " + String(result[1]))
    if code == 6:
        var status = Int(py=result[1])
        var text = String(result[2])
        if status == 1:
            raise Error("ValueError: " + String(source) + " is not an ONNX model: " + text)
        if status == 2:
            var b = native_loaded_module("builtins")
            var exception = py_call(b.FileNotFoundError, 2, "No such file or directory", source) if Int(py=result[3]) != 0 else py_call(b.OSError, String(source) + ": " + text)
            _raise_object(py_type(exception), exception)
        if status == 3 or status == 4:
            raise Error("ValueError: " + String(source) + ": " + text)
    raise Error("RuntimeError: the model import has another result layout; rebuild the x_sequence bindings")


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
        var result = PythonObject(None)
        if reader is None:
            result = py_call(entries["onnx_import_file"], path, session._id, words[0], words[1], name)
        else:
            var stored = _fields(reader, context["layouts"][context["reader"]])
            if stored["b"] is not session._b:
                raise Error("ValueError: the graph was read by another binding than this session's; pass the ONNX file's path instead")
            if py_type(stored["h"]) is not b.int:
                raise Error("TypeError: native graph handle requires an exact integer")
            # The native reader retains the buffers. Read its own addresses;
            # no Python buffer_address, memoryview wrapper or iteration hook.
            var addresses = py_call(entries["onnx_buffers"], stored["h"])
            result = py_call(entries["onnx_import"], stored["h"], session._id, words[0], words[1], addresses, name)
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
        _ = py_call(session._session_native, session, "close", Python.tuple())
        raise error^


def ops_import_model_c(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=packed)
        if len(args) != 5:
            raise Error("TypeError: ops_import_model expects source, label, shapes, numeric_mode and context")
        return _import(args[0], args[1], args[2], args[3], args[4]).steal_data()
    except error:
        if String(error) == PYTHON_ERROR_PENDING:
            return optimizer_python_error(error)
        return graph_python_error(error)
