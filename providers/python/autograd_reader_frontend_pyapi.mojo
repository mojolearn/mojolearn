# SPDX-License-Identifier: Apache-2.0
# Public ONNX reader/resolve boundary. Metadata is decoded natively; ctypes
# native byte storage retains the common resource so views outlive GraphIR.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_fields, py_get, py_setitem, py_setattr
from bindings.native_loaded_module import native_loaded_module
from bindings.native_buffer_owner import native_storage_allocate
from bindings.native_json import native_path
from providers.python.autograd_import_metadata_pyapi import _fields, _shape_words, _unsupported, _status
from providers.python.neural_descriptor_pyapi import NeuralDescriptorProvider, NativeDescriptorResource, neural_descriptor_reuse, neural_descriptor_entry, neural_descriptor_execute, neural_descriptor_adopt
from autograd.import_reader_dispatch import NativeReaderRequest


struct _Cursor(Movable):
    var words: List[Int]
    var strings: List[String]
    var floats: List[Float64]
    var at: Int

    def __init__(out self, var words: List[Int], var strings: List[String], var floats: List[Float64]):
        self.words = words^
        self.strings = strings^
        self.floats = floats^
        self.at = 3

    def take(mut self) raises -> Int:
        if self.at >= len(self.words):
            raise Error("RuntimeError: ONNX reader metadata is truncated")
        var value = self.words[self.at]
        self.at += 1
        return value

    def text(mut self) raises -> PythonObject:
        var index = self.take()
        return PythonObject(self.strings[index])


def _header(c: _Cursor, path: PythonObject, context: PythonObject) raises -> Int:
    if len(c.words) < 3 or c.words[0] != 0x4f4e5852 or c.words[1] != 1:
        raise Error("RuntimeError: mojolearn: the x_sequence binding's ONNX reader has another word layout; rebuild the x_sequence bindings")
    var status = c.words[2]
    if status >= 1 and status <= 4:
        var missing = len(c.words) > 4 and c.words[4] == 1
        _status(Python.list(6, status, c.strings[c.words[3]], Int(missing)), path, context)
    return status


def _problems(mut c: _Cursor) raises -> PythonObject:
    var result = Python.list()
    var count = c.take()
    for i in range(count):  # driver-loop: over reader refusal records
        var name = c.text()
        var op = c.take()
        var why = c.text()
        _ = py_call(result.append, Python.tuple(name, PythonObject(None) if op < 0 else PythonObject(c.strings[op]), why))
    return result


def _tuple_words(mut c: _Cursor, strings: Bool = False) raises -> PythonObject:
    var values = Python.list()
    var count = c.take()
    for i in range(count):  # driver-loop: over graph metadata vector
        var item = c.take()
        _ = py_call(values.append, PythonObject(c.strings[item]) if strings else PythonObject(item))
    return py_call(native_loaded_module("builtins").tuple, values)


def _object(cls: PythonObject, names: List[String], values: List[PythonObject]) raises -> PythonObject:
    var result = native_storage_allocate(cls)
    var fields = py_fields(result)
    for i in range(len(names)):  # driver-loop: over fixed metadata storage fields
        py_setitem(fields, names[i], values[i])
    return result


def _tensor(mut c: _Cursor, views: PythonObject, context: PythonObject) raises -> PythonObject:
    var name = c.text()
    var dtype = context["dtypes"][c.take()]
    var shape = _tuple_words(c)
    var buffer = c.take()
    var first = c.take()
    var end = c.take()
    var slice = py_call(native_loaded_module("builtins").slice, first, end)
    return _object(context["tensor"], [String("name"), String("dtype"), String("shape"), String("data")],
        [name, dtype, shape, views[buffer][slice]])


def _value(mut c: _Cursor, context: PythonObject) raises -> PythonObject:
    var name = c.text()
    var dtype = context["dtypes"][c.take()]
    var rank = c.take()
    var dims = Python.list()
    for i in range(rank):  # driver-loop: over graph value axes
        var symbolic = c.take()
        var value = c.take()
        _ = py_call(dims.append, PythonObject(c.strings[value]) if symbolic != 0 else PythonObject(value))
    return Python.tuple(name, dtype, py_call(native_loaded_module("builtins").tuple, dims))


def _attribute(mut c: _Cursor, views: PythonObject, context: PythonObject) raises -> PythonObject:
    var kind = c.take()
    if kind == 1:
        var index = c.take()
        return PythonObject(c.floats[index])
    if kind == 2:
        return PythonObject(c.take())
    if kind == 3:
        return c.text()
    if kind == 4:
        return _tensor(c, views, context)
    if kind == 7 or kind == 8:
        return _tuple_words(c, kind == 8)
    var values = Python.list()
    if kind == 6:
        var first = c.take()
        var end = c.take()
        for i in range(first, end):  # driver-loop: over float attribute metadata
            _ = py_call(values.append, c.floats[i])
    else:
        var count = c.take()
        for i in range(count):  # driver-loop: over tensor attribute metadata
            _ = py_call(values.append, _tensor(c, views, context))
    return py_call(native_loaded_module("builtins").tuple, values)


def _views(mut c: _Cursor, buffers: List[Int], resource: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = native_loaded_module("builtins")
    var views = Python.list()
    var count = c.take()
    if count != len(buffers):
        raise Error("RuntimeError: ONNX reader buffer table differs from graph metadata")
    for i in range(count):  # driver-loop: over native reader buffer ownership records
        _ = c.take()
        _ = c.take()
        var size = c.take()
        if size == 0:
            _ = py_call(views.append, py_call(bi.memoryview, py_call(bi.bytes)))
        else:
            var base = context["byte_type"] * PythonObject(size)
            var namespace = bi.dict()
            py_setitem(namespace, "__slots__", Python.tuple("owner"))
            var cls = py_call(bi.type, "_ReaderBytes", Python.tuple(base), namespace)
            var array = py_call(cls.from_address, buffers[i])
            py_setattr(array, "owner", resource)
            _ = py_call(views.append, py_call(py_call(bi.memoryview, array).cast, "B"))
    return views


def _reader_binding(value: PythonObject, context: PythonObject) raises -> Tuple[PythonObject, PythonObject]:
    var bi = native_loaded_module("builtins")
    var binding = value
    if value is None:
        var mode = context["backend"]["_DEFAULT_MODE"]
        if mode is None:
            mode = context["backend"]["_SELECTED"]
        var entry = context["bindings"][mode]
        if py_type(entry) is bi.str:
            raise Error(String(py=entry))
        binding = entry
    var aliases = context["aliases"]
    for i in range(len(aliases)):  # driver-loop: over connected module aliases, without invoking lazy Python proxy hooks
        if binding is aliases[i][0]:
            binding = aliases[i][1]
            break
    if py_type(binding) is not py_type(bi):
        raise Error("TypeError: ONNX reader requires a connected native binding module")
    var factory = py_get(py_fields(binding), "onnx_native_api_v1")
    if factory is None or py_type(factory) is not py_type(bi.len):
        raise Error("RuntimeError: mojolearn: this install's x_sequence binding has no native ONNX reader; rebuild the x_sequence bindings")
    if factory is not neural_descriptor_entry(factory.__self__, "onnx_native_api_v1"):
        raise Error("TypeError: ONNX reader provider factory changed")
    var provider = neural_descriptor_reuse(py_get(context["providers"], factory.__self__), factory,
        "mojolearn.autograd.onnx.v1", 4)
    py_setitem(context["providers"], factory.__self__, provider)
    return (factory.__self__, provider)


def reader_read(path: PythonObject, binding: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = native_loaded_module("builtins")
    var source = native_path(path)
    var selected = _reader_binding(binding, context)
    var request = List[NativeReaderRequest]()
    request.append(NativeReaderRequest())
    request[0].path = String(py=source)
    var scope = List[Int](length=1, fill=0)
    var handle = neural_descriptor_execute(selected[1], 0, [Int(request.unsafe_ptr()), Int(scope.unsafe_ptr())],
        List[Int](), List[Float64](), False)
    var resource = neural_descriptor_adopt(selected[1], handle, 3, scope.copy())
    try:
        _ = neural_descriptor_execute(selected[1], 1, [handle, Int(request.unsafe_ptr())], scope, List[Float64](), False)
        var c = _Cursor(request[0].info.words.copy(), request[0].info.strs.copy(), request[0].info.floats.copy())
        var status = _header(c, source, context)
        if status == 5:
            _unsupported(source, _problems(c), context)
        var opset = c.take()
        var views = _views(c, request[0].buffers, resource, context)
        var inputs = Python.list()
        var input_count = c.take()
        for i in range(input_count):  # driver-loop: over graph input metadata
            _ = py_call(inputs.append, _value(c, context))
        var outputs = Python.list()
        var output_count = c.take()
        for i in range(output_count):  # driver-loop: over graph output metadata
            _ = py_call(outputs.append, _value(c, context))
        var nodes = Python.list()
        var node_count = c.take()
        for i in range(node_count):  # driver-loop: over graph node metadata
            var op = c.text()
            var domain = c.text()
            var name = c.text()
            var ins = _tuple_words(c, True)
            var outs = _tuple_words(c, True)
            var attrs = bi.dict()
            var attr_count = c.take()
            for j in range(attr_count):  # driver-loop: over one node's attributes
                var key = c.text()
                py_setitem(attrs, key, _attribute(c, views, context))
            _ = py_call(nodes.append, _object(context["node"], [String("op"), String("domain"), String("name"),
                String("inputs"), String("outputs"), String("attrs")], [op, domain, name, ins, outs, attrs]))
        var initializers = bi.dict()
        var init_count = c.take()
        for i in range(init_count):  # driver-loop: over initializer metadata, never tensor arithmetic
            var tensor = _tensor(c, views, context)
            py_setitem(initializers, py_fields(tensor)["name"], tensor)
        var native = _object(context["reader"], [String("b"), String("h"), String("views"), String("_lifetime")],
            [selected[0], PythonObject(handle), py_call(bi.tuple, views), resource])
        return _object(context["graph"], [String("opset"), String("inputs"), String("outputs"), String("nodes"),
            String("initializers"), String("source"), String("_native")], [PythonObject(opset), py_call(bi.tuple, inputs),
            py_call(bi.tuple, outputs), py_call(bi.tuple, nodes), initializers, source, native])
    except error:
        resource.downcast_value_ptr[NativeDescriptorResource]()[].close()
        raise error


def reader_resolve(value: PythonObject, shapes: PythonObject, context: PythonObject,
                   shape_only: Bool, is_native: Bool) raises -> PythonObject:
    var bi = native_loaded_module("builtins")
    var native = value
    var source = PythonObject("the model graph")
    if not is_native:
        var graph = _fields(value, context["layouts"][context["graph"]])
        native = py_get(graph, "_native")
        source = graph["source"]
    if native is None:
        raise Error("TypeError: resolve takes a graph read by mojolearn.importers.onnx.read")
    var fields = _fields(native, context["layouts"][context["reader"]])
    var lifetime = fields["_lifetime"]
    var owner = lifetime.downcast_value_ptr[NativeDescriptorResource]()
    if owner[].handle == 0 or py_type(fields["h"]) is not bi.int or owner[].handle != Int(py=fields["h"]):
        raise Error("TypeError: ONNX reader handle differs from its retained owner")
    var provider = owner[].provider.downcast_value_ptr[NeuralDescriptorProvider]()
    if provider[].factory is not py_get(py_fields(fields["b"]), "onnx_native_api_v1"):
        raise Error("TypeError: ONNX reader binding changed")
    if provider[].factory is not neural_descriptor_entry(provider[].module, "onnx_native_api_v1"):
        raise Error("TypeError: ONNX native provider factory changed")
    var packed = _shape_words(shapes, context)
    var request = List[NativeReaderRequest]()
    request.append(NativeReaderRequest())
    for i in range(len(packed[0])):  # driver-loop: over admitted input names
        request[0].names.append(String(py=packed[0][i]))
    for i in range(len(packed[1])):  # driver-loop: over admitted input dimensions
        request[0].dimensions.append(Int(py=packed[1][i]))
    _ = neural_descriptor_execute(owner[].provider, 2, [owner[].handle, Int(request.unsafe_ptr())], owner[].context, List[Float64](), False)
    var c = _Cursor(request[0].resolved.words.copy(), request[0].resolved.strs.copy(), List[Float64]())
    _ = _header(c, PythonObject("the model graph"), context)
    var output = bi.dict()
    var dtypes = bi.dict()
    var count = c.take()
    for i in range(count):  # driver-loop: over resolved graph values
        var name = c.text()
        var dtype = context["dtypes"][c.take()]
        py_setitem(output, name, _tuple_words(c))
        py_setitem(dtypes, name, dtype)
    var folded = bi.dict()
    count = c.take()
    for i in range(count):  # driver-loop: over integer shape metadata folded by the native resolver
        var name = c.text()
        py_setitem(folded, name, _tuple_words(c))
    var no_rule = _tuple_words(c, True)
    var problems = _problems(c)
    if shape_only:
        if len(problems) != 0:
            _unsupported(source, problems, context)
        return output
    return _object(context["resolved"], [String("shapes"), String("dtypes"), String("folded"), String("no_rule"), String("problems")],
        [output, dtypes, folded, no_rule, py_call(bi.tuple, problems)])


def reader_frontend_operation(operation: PythonObject, value: PythonObject, options: PythonObject,
                              context: PythonObject) raises -> PythonObject:
    if py_type(operation) is not native_loaded_module("builtins").str:
        raise Error("TypeError: ONNX operation requires native string metadata")
    var op = String(py=operation)
    if op == "reader_read":
        return reader_read(value, options, context)
    if op == "reader_resolve" or op == "resolve" or op == "resolve_shapes":
        return reader_resolve(value, options, context, op == "resolve_shapes", op == "reader_resolve")
    raise Error("ValueError: unknown ONNX reader operation")
