# SPDX-License-Identifier: Apache-2.0
"""Native buffer preparation and launch ownership for existing Generator draws."""
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.generator_state_native import generator_fields, generator_next
from bindings.causal_checkpoint import checkpoint_array
from bindings.causal_lm_construct import _model_f32
from bindings.generation_buffers import generation_initialize_binding


def _generator_shape(shape: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var scalar = PythonObject(None)
    try:
        scalar = context["generator_index"](shape)
    except:
        pass
    var source = Python.tuple(scalar) if scalar is not None else bi.tuple(shape)
    var dims = bi.list()
    var count = PythonObject(1)
    for i in range(len(source)):  # driver-loop: over caller shape metadata
        var d = bi.int(source[i])
        if Bool(py=d < 1):
            raise Error("Generator: shape must be positive")
        dims.append(d)
        count = count * d
    return Python.tuple(bi.tuple(dims), count)


def _generator_binding(owner: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var mode = generator_fields(owner)["numeric_mode"]
    if not Bool(py=mode):
        mode = context["generator_backend"]._DEFAULT_MODE
    mode = mode.strip().lower()
    var connected = context["generator_bindings"].get(mode)
    if connected is None or bi.type(connected) is bi.str:
        raise Error("Generator: native RNG binding is unavailable for numeric_mode=" + String(mode))
    var vendor = String(connected["training_vendor"]())
    var compiled = Int(py=connected["training_numeric_mode"]())
    var selected = context["vendor"]
    var expected = "cpu" if selected is None else String(selected)
    if vendor != expected or compiled != (1 if String(mode) == "identical" else 0):
        raise Error("ImportError: Generator native RNG vendor or numeric mode differs from the selected route")
    var entry = connected["neural_rng"]
    if bi.type(entry) is not bi.type(bi.len):
        raise Error("Generator: RNG entry must be a compiled native function")
    return entry


def _generator_f32(value: PythonObject) raises -> Float32:
    return Float32(Float64(py=Python.import_module("builtins").float(value)))


def _generator_input(value: PythonObject, context: PythonObject) raises -> PythonObject:
    # The shared native packer preserves f32 bytes and converts arbitrary
    # buffer strides/layout into C order without Array or NumPy callbacks.
    return _model_f32(value, "Generator input", context["neural"])


def _generator_run(entry: PythonObject, shape: PythonObject, count: PythonObject,
                   kind: Int, offset: PythonObject, seed: PythonObject,
                   stream: PythonObject, a: PythonObject, b: PythonObject,
                   input: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var out = checkpoint_array(bi.bytes(count * PythonObject(4)), shape, "f", context["neural"]["array_class"])
    var source = input
    if source is None:
        source = checkpoint_array(bi.bytes(4), Python.tuple(1), "f", context["neural"]["array_class"])
    # Python integer primitives here preserve the legacy two's-complement
    # low/high split even if a caller edited the public seed field directly.
    var lo = seed.__and__(PythonObject(4294967295))
    var hi = seed.__rshift__(32).__and__(PythonObject(4294967295))
    var params = Python.list(count, bi.int(offset), bi.int(lo), bi.int(hi), bi.int(stream), kind, bi.float(a), bi.float(b))
    _ = entry(Python.list(out._addr, source._addr), params)
    _ = source^
    return out


def generator_draw(operation: String, values: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    _ = generation_initialize_binding(context["neural"]["generation_classes"])
    var owner = values[0]
    var fields = generator_fields(owner)
    if operation == "uniform":
        var shaped = _generator_shape(values[1], context)
        var lo = _generator_f32(values[2])
        var hi = _generator_f32(values[3])
        var span = Float32(Float64(hi) - Float64(lo))
        if not (span >= 0):
            raise Error("Generator.uniform: high must be >= low")
        # Preserve the established evaluation order: route refusal precedes
        # stream consumption, while allocation/kernel failures follow it.
        var entry = _generator_binding(owner, context)
        var seed = fields["seed"]
        var stream = generator_next(owner)
        return _generator_run(entry, shaped[0], shaped[1], 0, PythonObject(0), seed, stream,
                              PythonObject(Float64(lo)), PythonObject(Float64(span)), PythonObject(None), context)
    if operation == "normal":
        var shaped = _generator_shape(values[1], context)
        var sd = _generator_f32(values[3])
        if not (sd >= 0):
            raise Error("Generator.normal: std must be >= 0")
        var entry = _generator_binding(owner, context)
        var seed = fields["seed"]
        var stream = generator_next(owner)
        # Legacy normal consumes its stream before converting the mean.
        var mean = _generator_f32(values[2])
        return _generator_run(entry, shaped[0], shaped[1], 1, PythonObject(0), seed, stream,
                              PythonObject(Float64(mean)), PythonObject(Float64(sd)), PythonObject(None), context)
    if operation == "dropout":
        var probability = _generator_f32(values[2])
        if not (probability >= 0 and probability < 1):
            raise Error("Generator.dropout: p must be in [0, 1)")
        var x = _generator_input(values[1], context)
        var scale = Float32(Float64(1) / (Float64(1) - Float64(probability)))
        var stream = generator_next(owner) if values[4] is None else bi.int(values[4])
        var key = bi.dict()
        key["seed"] = fields["seed"]
        key["stream"] = stream
        key["p"] = Float64(probability)
        key["scale"] = Float64(scale)
        key["offset"] = bi.int(values[3])
        # Dropout consumed above before converting offset or selecting mode.
        var entry = _generator_binding(owner, context)
        var out = _generator_run(entry, x.shape, x.size, 2, key["offset"], key["seed"], stream,
                                 key["p"], key["scale"], x, context)
        return Python.tuple(out, key)
    if operation == "dropout_backward":
        var dy = _generator_input(values[1], context)
        var entry = _generator_binding(owner, context)
        var key = values[2]
        return _generator_run(entry, dy.shape, dy.size, 3, key["offset"], key["seed"], key["stream"],
                              key["p"], key["scale"], dy, context)
    raise Error("Generator: unknown native draw operation " + operation)


def generator_draw_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=args_ptr)
        var result = generator_draw(String(args[0]), args[1], args[2])
        return result.steal_data()
    except error:
        var message = String(error)
        if message.startswith("RuntimeError:"):
            return raise_python_exception(error, ExceptionType("PyExc_RuntimeError"))
        if message.startswith("TypeError:"):
            return raise_python_exception(error, ExceptionType.TypeError)
        if message.startswith("ImportError:"):
            return raise_python_exception(error, ExceptionType("PyExc_ImportError"))
        return raise_python_exception(error, ExceptionType.ValueError)
