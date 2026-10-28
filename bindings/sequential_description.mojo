# SPDX-License-Identifier: Apache-2.0
"""Native Sequential model-description and option metadata."""
from bindings.native_loaded_module import native_loaded_module, NativeModules
from std.python import Python, PythonObject
from bindings.sequential_admission import sequential_integer, sequential_admit_value
from bindings.py_balanced import py_call, py_contains, py_fields, py_setitem, py_type
from bindings.native_identity_key import NativeIdentityKey
from bindings.native_prepared_owner import prepared_get, prepared_put


def sequential_fields(value: PythonObject) raises -> PythonObject:
    return py_fields(value)


def _integer(value: PythonObject, context: PythonObject) raises -> Bool:
    return sequential_integer(value, context)


def _positive(value: PythonObject, name: String, low: Int, context: PythonObject) raises -> Int:
    if not _integer(value, context) or Bool(py=value < PythonObject(low)):
        raise Error(name + " must be an integer >= " + String(low))
    return Int(py=value)


def _shape(mut mods: NativeModules, value: PythonObject, name: String, infer: Bool, context: PythonObject) raises -> PythonObject:
    var b = mods.module("builtins")
    var source = py_call(b.tuple, value)
    sequential_admit_value(mods, source, context)
    if len(source) < 1 or len(source) > 6:
        raise Error(name + " must list 1 to 6 dimensions")
    var out = Python.list()
    for k in range(len(source)):  # driver-loop: over at most six shape arguments
        if not _integer(source[k], context) or not (
            Bool(py=source[k] >= PythonObject(1)) or (infer and Bool(py=source[k] == PythonObject(-1)))):
            raise Error(name + " must hold positive integers" + (" or -1" if infer else ""))
        _ = py_call(out.append, py_call(b.int, source[k]))
    return out


def sequential_append(mut mods: NativeModules, model: PythonObject, code: PythonObject, a0: PythonObject,
                      a1: PythonObject, f: PythonObject, supplied: PythonObject) raises:
    var b = mods.module("builtins")
    var args = Python.list(a0, a1, 0, 0, 0, 0) if supplied is None else py_call(b.list, supplied)
    var fields = sequential_fields(model)
    _ = py_call(fields["_layers"].append, Python.tuple(code, a0, a1, f, py_call(b.tuple, args)))


def _arguments(mut mods: NativeModules, kind: String, supplied: PythonObject, keywords: PythonObject,
               names: PythonObject, defaults: PythonObject, required: Int) raises -> PythonObject:
    var b = mods.module("builtins")
    if len(supplied) > len(names):
        raise Error("TypeError: " + kind + " received too many positional arguments")
    var result = py_call(b.list, defaults)
    for k in range(len(supplied)):  # driver-loop: over the layer's bounded positional options
        py_setitem(result, k, supplied[k])
    var keys = py_call(b.tuple, keywords)
    for k in range(len(keys)):  # driver-loop: over the layer's bounded keyword options
        var found = False
        for j in range(len(names)):  # driver-loop: over the layer's option names
            if Bool(py=keys[k] == names[j]):
                if j < len(supplied):
                    raise Error("TypeError: " + kind + " received an argument twice")
                py_setitem(result, j, keywords[keys[k]])
                found = True
        if not found:
            raise Error("TypeError: " + kind + " received an unknown keyword")
    for k in range(required):  # driver-loop: over required layer options
        if k >= len(supplied) and not py_contains(keywords, names[k]):
            raise Error("TypeError: " + kind + " is missing " + String(names[k]))
    return result


def sequential_add(mut mods: NativeModules, model: PythonObject, kind_arg: PythonObject, supplied: PythonObject,
                   keywords: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = mods.module("builtins")
    if not Bool(py=py_call(b.isinstance, kind_arg, b.str)):
        raise Error("kind must name a supported Sequential layer")
    var kind = String(kind_arg.lower())
    var codes = context["layer_codes"]
    if not py_contains(codes, kind):
        raise Error("kind must name a supported Sequential layer")
    var code = codes[kind]
    var a0 = PythonObject(0)
    var a1 = PythonObject(0)
    var f = PythonObject(0.0)
    var args = PythonObject(None)
    if kind == "linear":
        var options = _arguments(mods, kind, supplied, keywords, Python.tuple("units", "bias"), Python.tuple(None, True), 1)
        if not _integer(options[0], context) or Bool(py=options[0] < PythonObject(1)):
            raise Error("units must be a positive integer")
        a0 = py_call(b.int, options[0])
        a1 = PythonObject(1 if Bool(py=options[1]) else 0)
    elif kind == "layer_norm":
        var options = _arguments(mods, kind, supplied, keywords, Python.tuple("eps"), Python.tuple(1e-5), 0)
        if not Bool(py=options[0] > PythonObject(0)):
            raise Error("eps must be positive")
        f = py_call(b.float, options[0])
    elif kind == "dropout":
        var options = _arguments(mods, kind, supplied, keywords, Python.tuple("p"), Python.tuple(None), 1)
        if not (Bool(py=options[0] >= PythonObject(0)) and Bool(py=options[0] < PythonObject(1))):
            raise Error("p must be in [0, 1)")
        f = py_call(b.float, options[0])
    elif kind == "residual":
        var options = _arguments(mods, kind, supplied, keywords, Python.tuple("source"), Python.tuple(-1), 0)
        if not _integer(options[0], context):
            raise Error("source must be -1 or the index of an earlier layer")
        a0 = py_call(b.int, options[0])
    elif kind == "reshape" or kind == "permute":
        var name = "shape" if kind == "reshape" else "axes"
        var options = _arguments(mods, kind, supplied, keywords, Python.tuple(name), Python.tuple(None), 1)
        if kind == "reshape":
            args = _shape(mods, options[0], name, True, context)
        else:
            var source = py_call(b.tuple, options[0])
            if len(source) < 1 or len(source) > 6:
                raise Error("axes must list 1 to 6 axes")
            args = Python.list()
            for k in range(len(source)):  # driver-loop: over a sample's axis permutation
                _ = py_call(args.append, _positive(source[k], "axes", 0, context))
        while len(args) < 6:  # driver-loop: over unused description argument slots
            _ = py_call(args.append, 0 if kind == "reshape" else -1)
    elif kind == "conv2d":
        var options = _arguments(mods, kind, supplied, keywords,
            Python.tuple("out_channels", "kernel_size", "stride", "padding", "bias", "groups"),
            Python.tuple(None, None, 1, 0, True, 1), 2)
        var oc = _positive(options[0], "out_channels", 1, context)
        var kernel = _positive(options[1], "kernel_size", 1, context)
        var stride = _positive(options[2], "stride", 1, context)
        var padding = _positive(options[3], "padding", 0, context)
        var groups = _positive(options[5], "groups", 1, context)
        a0 = PythonObject(oc)
        a1 = PythonObject(kernel)
        args = Python.list(oc, kernel, stride, padding, 1 if Bool(py=options[4]) else 0, groups)
    elif kind == "max_pool2d" or kind == "avg_pool2d":
        var options = _arguments(mods, kind, supplied, keywords, Python.tuple("kernel_size", "stride"), Python.tuple(None, None), 1)
        var kernel = _positive(options[0], "kernel_size", 1, context)
        var stride = kernel if options[1] is None else _positive(options[1], "stride", 1, context)
        a0 = PythonObject(kernel)
        a1 = PythonObject(stride)
    elif len(supplied) != 0 or len(keywords) != 0:
        raise Error("TypeError: " + kind + " takes no arguments")
    sequential_append(mods, model, code, a0, a1, f, args)
    return model


def sequential_compile(mut mods: NativeModules, model: PythonObject, options: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = mods.module("builtins")
    var loss = options[0]
    var optimizer = options[1]
    var schedule = options[9]
    if not py_contains(context["loss_codes"], loss):
        raise Error("loss must name a supported Sequential loss")
    if not py_contains(context["optimizer_codes"], optimizer):
        raise Error("optimizer must name a supported Sequential optimizer")
    if not py_contains(context["schedule_codes"], schedule):
        raise Error("schedule must name a supported Sequential schedule")
    var adam = String(optimizer) == "adam" or String(optimizer) == "adamw"
    var momentum = String(optimizer) == "momentum"
    var out = Python.dict()
    py_setitem(out, "loss", loss)
    py_setitem(out, "opt", context["optimizer_codes"][optimizer])
    py_setitem(out, "flags", 1 if momentum and Bool(py=options[4]) else 0)
    py_setitem(out, "lr", py_call(b.float, options[2]))
    py_setitem(out, "f1", py_call(b.float, options[5]) if adam else (py_call(b.float, options[3]) if momentum else PythonObject(0.0)))
    py_setitem(out, "f2", py_call(b.float, options[6]) if adam else PythonObject(0.0))
    py_setitem(out, "eps", py_call(b.float, options[7]))
    py_setitem(out, "weight_decay", py_call(b.float, options[8]))
    py_setitem(out, "sched", context["schedule_codes"][schedule])
    py_setitem(out, "step_size", py_call(b.int, options[10]))
    py_setitem(out, "gamma", py_call(b.float, options[11]))
    py_setitem(out, "huber_delta", py_call(b.float, options[12]))
    py_setitem(out, "tol", py_call(b.float, options[13]))
    py_setitem(out, "n_iter_no_change", py_call(b.int, options[14]))
    var fields = sequential_fields(model)
    py_setitem(fields, "_options", out)
    return model


struct SequentialDescription(Movable, Writable):
    # The execution ABI owns native words. Retain them across calls rather than
    # rebuilding Python lists and immediately copying their elements back into
    # Mojo. The public description methods keep their existing Python results.
    var identities: NativeIdentityKey
    var input_width: Int
    var cacheable: Bool
    var program: List[Int]
    var scalars: List[Float64]

    def __init__(out self, mut mods: NativeModules, layers: PythonObject,
                 width: Int, sample: PythonObject) raises:
        self.identities = NativeIdentityKey(mods)
        self.cacheable = True
        try:
            self.identities.add(layers)
            self.identities.add(sample)
        except:
            # Uncacheable extra metadata must not change ordinary execution.
            self.cacheable = False
        var b = mods.module("builtins")
        var int_type = b.int
        var float_type = b.float
        var bool_type = b.bool
        for i in range(len(self.identities.objects)):  # driver-loop: establish immutable scalar ownership once on cache miss
            var value = self.identities.objects[i]
            var cls = py_type(value)
            if value is not None and cls is not self.identities.list_type and cls is not self.identities.tuple_type and cls is not int_type and cls is not float_type and cls is not bool_type:
                # Writable native buffers can convert to scalars while retaining
                # their identity. Preserve that behavior by rebuilding for those
                # descriptions, rather than caching their converted contents.
                self.cacheable = False
                break
        self.input_width = width
        self.program = List[Int]()
        self.scalars = List[Float64]()
        var count = len(layers)
        if sample is None:
            self.program.append(1)
            self.program.append(width)
            self.program.append(count)
        else:
            var rank = len(sample)
            self.program.append(2)
            self.program.append(rank)
            for k in range(rank):  # driver-loop: capture sample dimensions in native ABI words
                self.program.append(Int(py=sample[k]))
            for k in range(rank, 8):  # driver-loop: fill unused native sample dimension slots
                self.program.append(1)
            self.program.append(count)
        for k in range(count):  # driver-loop: capture admitted layer metadata once per changed description
            var layer = layers[k]
            self.program.append(Int(py=layer[0]))
            if sample is None:
                self.program.append(Int(py=layer[1]))
                self.program.append(Int(py=layer[2]))
            else:
                var args = layer[4]
                for j in range(len(args)):  # driver-loop: capture rank-aware native layer arguments
                    self.program.append(Int(py=args[j]))
            self.program.append(0)
            self.scalars.append(Float64(py=layer[3]))

    def write_to(self, mut writer: Some[Writer]):
        writer.write("SequentialDescription(native)")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def sequential_native_description(mut mods: NativeModules, model: PythonObject,
                                  n_features: PythonObject, sample: PythonObject) raises -> PythonObject:
    var layers = sequential_fields(model)["_layers"]
    # Rank-aware programs do not consume n_features. This also preserves the
    # public description's behavior when sample is present and width is None.
    var width = 0 if sample is not None else Int(py=n_features)
    var cached = prepared_get(mods, "sequential_description_v1", model)
    if cached is not None:
        var retained = cached.downcast_value_ptr[SequentialDescription]()
        var offset = 0
        if retained[].input_width == width and retained[].identities.matches(layers, offset) and retained[].identities.matches(sample, offset) and retained[].identities.complete(offset):
            return cached
    var description = SequentialDescription(mods, layers, width, sample)
    var stable = description.cacheable and not description.identities.retains(model)
    var result = PythonObject(alloc=description^)
    if stable:
        prepared_put(mods, "sequential_description_v1", model, result)
    return result


def sequential_program(mut mods: NativeModules, model: PythonObject, n_features: PythonObject, sample: PythonObject) raises -> PythonObject:
    var b = mods.module("builtins")
    var layers = sequential_fields(model)["_layers"]
    var result = Python.list()
    if sample is None:
        result = Python.list(1, py_call(b.int, n_features), len(layers))
        var extend = result.extend
        for k in range(len(layers)):  # driver-loop: over existing model-description layers
            var layer = layers[k]
            _ = py_call(extend, Python.tuple(layer[0], layer[1], layer[2], 0))
    else:
        result = Python.list(2, len(sample))
        var extend = result.extend
        var append = result.append
        _ = py_call(extend, sample)
        for k in range(len(sample), 8):  # driver-loop: over unused sample dimension slots
            _ = py_call(append, 1)
        _ = py_call(append, len(layers))
        for k in range(len(layers)):  # driver-loop: over existing rank-aware description layers
            var layer = layers[k]
            _ = py_call(append, layer[0])
            _ = py_call(extend, layer[4])
            _ = py_call(append, 0)
    return result


def sequential_table(model: PythonObject) raises -> PythonObject:
    var layers = sequential_fields(model)["_layers"]
    var result = Python.list()
    for k in range(len(layers)):  # driver-loop: over layer scalar options
        _ = py_call(result.append, layers[k][3])
    return result


def sequential_layers(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var layers = sequential_fields(model)["_layers"]
    var result = Python.list()
    for k in range(len(layers)):  # driver-loop: over the model's public layer descriptions
        var layer = layers[k]
        var name = context["layer_names"][layer[0]]
        if Int(py=layer[0]) >= 9:
            _ = py_call(result.append, Python.tuple(name, layer[4], layer[3]))
        else:
            _ = py_call(result.append, Python.tuple(name, layer[1], layer[2], layer[3]))
    return result


def sequential_sample(mut mods: NativeModules, model: PythonObject, value: PythonObject) raises -> PythonObject:
    # The public sample-inspection operation admits current shape metadata.
    var shape = value.shape
    var dims = List[Int]()
    for k in range(Int(py=value.ndim)):  # driver-loop: over input rank metadata
        dims.append(Int(py=shape[k]))
    return sequential_sample_shape(mods, model, dims)


def sequential_sample_shape(mut mods: NativeModules, model: PythonObject, dims: List[Int]) raises -> PythonObject:
    var b = mods.module("builtins")
    var fields = sequential_fields(model)
    var shape = fields["input_shape"]
    var nd = len(dims)
    var needed = shape is not None or nd > 2
    var layers = fields["_layers"]
    for k in range(len(layers)):  # driver-loop: over shape-changing layer kinds
        needed = needed or Int(py=layers[k][0]) >= 9
    if not needed:
        return PythonObject(None)
    var actual = Python.list()
    for k in range(1, nd):  # driver-loop: over the input sample rank
        _ = py_call(actual.append, PythonObject(dims[k]))
    if shape is None:
        if nd - 1 > 7:
            raise Error("a sample has at most 7 dimensions")
        return py_call(b.tuple, actual)
    if nd > 2 and not Bool(py=py_call(b.tuple, actual) == shape):
        raise Error("X samples have a different shape from the model's input_shape")
    return shape


