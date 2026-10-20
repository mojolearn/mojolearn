# SPDX-License-Identifier: Apache-2.0
"""Native owner of the existing Sequential model-description interface."""
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception


def sequential_fields(value: PythonObject) raises -> PythonObject:
    return Python.import_module("builtins").object.__getattribute__(value, "__dict__")


def _integer(value: PythonObject, context: PythonObject) raises -> Bool:
    var b = Python.import_module("builtins")
    return not Bool(py=b.isinstance(value, b.bool)) and (
        Bool(py=b.isinstance(value, b.int)) or Bool(py=b.isinstance(value, context["integer"])))


def _positive(value: PythonObject, name: String, low: Int, context: PythonObject) raises -> Int:
    if not _integer(value, context) or Bool(py=value < PythonObject(low)):
        raise Error(name + " must be an integer >= " + String(low))
    return Int(py=value)


def _shape(value: PythonObject, name: String, infer: Bool, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var source = b.tuple(value)
    if len(source) < 1 or len(source) > 6:
        raise Error(name + " must list 1 to 6 dimensions")
    var out = Python.list()
    for k in range(len(source)):  # driver-loop: over at most six shape arguments
        if not _integer(source[k], context) or not (
            Bool(py=source[k] >= PythonObject(1)) or (infer and Bool(py=source[k] == PythonObject(-1)))):
            raise Error(name + " must hold positive integers" + (" or -1" if infer else ""))
        out.append(b.int(source[k]))
    return out


def sequential_append(model: PythonObject, code: PythonObject, a0: PythonObject,
                      a1: PythonObject, f: PythonObject, supplied: PythonObject) raises:
    var b = Python.import_module("builtins")
    var args = Python.list(a0, a1, 0, 0, 0, 0) if supplied is None else b.list(supplied)
    var fields = sequential_fields(model)
    fields["_layers"].append(Python.tuple(code, a0, a1, f, b.tuple(args)))


def _arguments(kind: String, supplied: PythonObject, keywords: PythonObject,
               names: PythonObject, defaults: PythonObject, required: Int) raises -> PythonObject:
    var b = Python.import_module("builtins")
    if len(supplied) > len(names):
        raise Error("TypeError: " + kind + " received too many positional arguments")
    var result = b.list(defaults)
    for k in range(len(supplied)):  # driver-loop: over the layer's bounded positional options
        result[k] = supplied[k]
    var keys = b.tuple(keywords)
    for k in range(len(keys)):  # driver-loop: over the layer's bounded keyword options
        var found = False
        for j in range(len(names)):  # driver-loop: over the layer's option names
            if Bool(py=keys[k] == names[j]):
                if j < len(supplied):
                    raise Error("TypeError: " + kind + " received an argument twice")
                result[j] = keywords[keys[k]]
                found = True
        if not found:
            raise Error("TypeError: " + kind + " received an unknown keyword")
    for k in range(required):  # driver-loop: over required layer options
        if k >= len(supplied) and not Bool(py=keywords.__contains__(names[k])):
            raise Error("TypeError: " + kind + " is missing " + String(names[k]))
    return result


def sequential_add(model: PythonObject, kind_arg: PythonObject, supplied: PythonObject,
                   keywords: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    if not Bool(py=b.isinstance(kind_arg, b.str)):
        raise Error("kind must name a supported Sequential layer")
    var kind = String(kind_arg.lower())
    var codes = context["layer_codes"]
    if not Bool(py=codes.__contains__(kind)):
        raise Error("kind must name a supported Sequential layer")
    var code = codes[kind]
    var a0 = PythonObject(0)
    var a1 = PythonObject(0)
    var f = PythonObject(0.0)
    var args = PythonObject(None)
    if kind == "linear":
        var options = _arguments(kind, supplied, keywords, Python.tuple("units", "bias"), Python.tuple(None, True), 1)
        if not _integer(options[0], context) or Bool(py=options[0] < PythonObject(1)):
            raise Error("units must be a positive integer")
        a0 = b.int(options[0])
        a1 = PythonObject(1 if Bool(py=options[1]) else 0)
    elif kind == "layer_norm":
        var options = _arguments(kind, supplied, keywords, Python.tuple("eps"), Python.tuple(1e-5), 0)
        if not Bool(py=options[0] > PythonObject(0)):
            raise Error("eps must be positive")
        f = b.float(options[0])
    elif kind == "dropout":
        var options = _arguments(kind, supplied, keywords, Python.tuple("p"), Python.tuple(None), 1)
        if not (Bool(py=options[0] >= PythonObject(0)) and Bool(py=options[0] < PythonObject(1))):
            raise Error("p must be in [0, 1)")
        f = b.float(options[0])
    elif kind == "residual":
        var options = _arguments(kind, supplied, keywords, Python.tuple("source"), Python.tuple(-1), 0)
        if not _integer(options[0], context):
            raise Error("source must be -1 or the index of an earlier layer")
        a0 = b.int(options[0])
    elif kind == "reshape" or kind == "permute":
        var name = "shape" if kind == "reshape" else "axes"
        var options = _arguments(kind, supplied, keywords, Python.tuple(name), Python.tuple(None), 1)
        if kind == "reshape":
            args = _shape(options[0], name, True, context)
        else:
            var source = b.tuple(options[0])
            if len(source) < 1 or len(source) > 6:
                raise Error("axes must list 1 to 6 axes")
            args = Python.list()
            for k in range(len(source)):  # driver-loop: over a sample's axis permutation
                args.append(_positive(source[k], "axes", 0, context))
        while len(args) < 6:  # driver-loop: over unused description argument slots
            args.append(0 if kind == "reshape" else -1)
    elif kind == "conv2d":
        var options = _arguments(kind, supplied, keywords,
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
        var options = _arguments(kind, supplied, keywords, Python.tuple("kernel_size", "stride"), Python.tuple(None, None), 1)
        var kernel = _positive(options[0], "kernel_size", 1, context)
        var stride = kernel if options[1] is None else _positive(options[1], "stride", 1, context)
        a0 = PythonObject(kernel)
        a1 = PythonObject(stride)
    elif len(supplied) != 0 or len(keywords) != 0:
        raise Error("TypeError: " + kind + " takes no arguments")
    sequential_append(model, code, a0, a1, f, args)
    return model


def sequential_compile(model: PythonObject, options: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var loss = options[0]
    var optimizer = options[1]
    var schedule = options[9]
    if not Bool(py=context["loss_codes"].__contains__(loss)):
        raise Error("loss must name a supported Sequential loss")
    if not Bool(py=context["optimizer_codes"].__contains__(optimizer)):
        raise Error("optimizer must name a supported Sequential optimizer")
    if not Bool(py=context["schedule_codes"].__contains__(schedule)):
        raise Error("schedule must name a supported Sequential schedule")
    var adam = String(optimizer) == "adam" or String(optimizer) == "adamw"
    var momentum = String(optimizer) == "momentum"
    var out = Python.dict()
    out["loss"] = loss
    out["opt"] = context["optimizer_codes"][optimizer]
    out["flags"] = 1 if momentum and Bool(py=options[4]) else 0
    out["lr"] = b.float(options[2])
    out["f1"] = b.float(options[5]) if adam else (b.float(options[3]) if momentum else PythonObject(0.0))
    out["f2"] = b.float(options[6]) if adam else PythonObject(0.0)
    out["eps"] = b.float(options[7])
    out["weight_decay"] = b.float(options[8])
    out["sched"] = context["schedule_codes"][schedule]
    out["step_size"] = b.int(options[10])
    out["gamma"] = b.float(options[11])
    out["huber_delta"] = b.float(options[12])
    out["tol"] = b.float(options[13])
    out["n_iter_no_change"] = b.int(options[14])
    var fields = sequential_fields(model)
    fields["_options"] = out
    return model


def sequential_program(model: PythonObject, n_features: PythonObject, sample: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var layers = sequential_fields(model)["_layers"]
    var result = Python.list()
    if sample is None:
        result = Python.list(1, b.int(n_features), len(layers))
        for k in range(len(layers)):  # driver-loop: over existing model-description layers
            var layer = layers[k]
            result.extend(Python.tuple(layer[0], layer[1], layer[2], 0))
    else:
        result = Python.list(2, len(sample))
        result.extend(sample)
        for k in range(len(sample), 8):  # driver-loop: over unused sample dimension slots
            result.append(1)
        result.append(len(layers))
        for k in range(len(layers)):  # driver-loop: over existing rank-aware description layers
            var layer = layers[k]
            result.append(layer[0])
            result.extend(layer[4])
            result.append(0)
    return result


def sequential_table(model: PythonObject) raises -> PythonObject:
    var layers = sequential_fields(model)["_layers"]
    var result = Python.list()
    for k in range(len(layers)):  # driver-loop: over layer scalar options
        result.append(layers[k][3])
    return result


def sequential_layers(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var layers = sequential_fields(model)["_layers"]
    var result = Python.list()
    for k in range(len(layers)):  # driver-loop: over the model's public layer descriptions
        var layer = layers[k]
        var name = context["layer_names"][layer[0]]
        if Int(py=layer[0]) >= 9:
            result.append(Python.tuple(name, layer[4], layer[3]))
        else:
            result.append(Python.tuple(name, layer[1], layer[2], layer[3]))
    return result


def sequential_sample(model: PythonObject, value: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var fields = sequential_fields(model)
    var shape = fields["input_shape"]
    var nd = Int(py=value.ndim)
    var needed = shape is not None or nd > 2
    var layers = fields["_layers"]
    for k in range(len(layers)):  # driver-loop: over shape-changing layer kinds
        needed = needed or Int(py=layers[k][0]) >= 9
    if not needed:
        return PythonObject(None)
    var actual = Python.list()
    for k in range(1, nd):  # driver-loop: over the input sample rank
        actual.append(b.int(value.shape[k]))
    if shape is None:
        if nd - 1 > 7:
            raise Error("a sample has at most 7 dimensions")
        return b.tuple(actual)
    if nd > 2 and not Bool(py=b.tuple(actual) == shape):
        raise Error("X samples have a different shape from the model's input_shape")
    return shape


def sequential_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        var model = call[0]
        var operation = String(call[1])
        var args = call[2]
        var context = call[3]
        var result = PythonObject(None)
        if operation == "construct":
            var b = Python.import_module("builtins")
            var fields = sequential_fields(model)
            fields["numeric_mode"] = args[0]
            fields["input_shape"] = PythonObject(None) if args[1] is None else b.tuple(_shape(args[1], "input_shape", False, context))
            fields["_layers"] = Python.list()
            fields["_options"] = PythonObject(None)
        elif operation == "add":
            result = sequential_add(model, args[0], args[1], args[2], context)
        elif operation == "append":
            sequential_append(model, args[0], args[1], args[2], args[3], args[4])
        elif operation == "compile":
            result = sequential_compile(model, args, context)
        elif operation == "layers":
            result = sequential_layers(model, context)
        elif operation == "program":
            result = sequential_program(model, args[0], args[1])
        elif operation == "table":
            result = sequential_table(model)
        elif operation == "sample":
            result = sequential_sample(model, args[0])
        else:
            raise Error("Sequential: unknown native operation")
        return result.steal_data()
    except error:
        if String(error).startswith("TypeError:"):
            return raise_python_exception(error, ExceptionType.TypeError)
        return raise_python_exception(error, ExceptionType.ValueError)
