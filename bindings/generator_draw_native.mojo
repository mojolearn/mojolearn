# SPDX-License-Identifier: Apache-2.0
"""Native buffer preparation and launch ownership for existing Generator draws."""
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_get, py_setitem, py_type
from bindings.generator_state_native import generator_fields, generator_next
from bindings.causal_checkpoint import checkpoint_array
from bindings.causal_lm_construct import _model_f32
from bindings.generation_buffers import generation_initialize_binding


def _generator_shape(shape: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var scalar = PythonObject(None)
    try:
        scalar = py_call(context["generator_index"], shape)
    except:
        pass
    var source = Python.tuple(scalar) if scalar is not None else py_call(bi.tuple, shape)
    var dims = bi.list()
    var count = PythonObject(1)
    for i in range(len(source)):  # driver-loop: over caller shape metadata
        var d = py_call(bi.int, source[i])
        if Bool(py=d < 1):
            raise Error("Generator: shape must be positive")
        _ = py_call(dims.append, d)
        count = count * d
    return Python.tuple(py_call(bi.tuple, dims), count)


def _generator_binding(owner: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var mode = generator_fields(owner)["numeric_mode"]
    if not Bool(py=mode):
        mode = context["generator_backend"]._DEFAULT_MODE
    mode = mode.strip().lower()
    var connected = py_get(context["generator_bindings"], mode)
    if connected is None or py_type(connected) is bi.str:
        raise Error("Generator: native RNG binding is unavailable for numeric_mode=" + String(mode))
    var vendor = String(connected["training_vendor"]())
    var compiled = Int(py=connected["training_numeric_mode"]())
    var selected = context["vendor"]
    var expected = "cpu" if selected is None else String(selected)
    if vendor != expected or compiled != (1 if String(mode) == "identical" else 0):
        raise Error("ImportError: Generator native RNG vendor or numeric mode differs from the selected route")
    var entry = connected["neural_rng"]
    if py_type(entry) is not py_type(bi.len):
        raise Error("Generator: RNG entry must be a compiled native function")
    return entry


def _generator_f32(value: PythonObject) raises -> Float32:
    return Float32(Float64(py=py_call(Python.import_module("builtins").float, value)))


def _generator_input(value: PythonObject, context: PythonObject) raises -> PythonObject:
    # The shared native packer preserves f32 bytes and converts arbitrary
    # buffer strides/layout into C order without Array or NumPy callbacks.
    var result = _model_f32(value, "Generator input", context["neural"])
    # The former _c32/as_f32_c admission rejects empty inputs before stream
    # consumption. Nonfinite input remains the existing kernel's admission.
    if Int(py=result.size) == 0:
        raise Error("Generator input is empty; at least one element is required")
    return result


def _generator_sqrt(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var x = py_call(bi.float, value)
    if Bool(py=x < 0):
        raise Error("Generator initializer: math domain error")
    var entry = context["generator_sqrt"]
    if py_type(entry) is bi.str:
        raise Error("OSError: Generator initializer requires portable math: " + String(entry))
    if py_type(entry) is not context["generator_sqrt_type"]:
        raise Error("TypeError: Generator initializer requires its connected portable-math C entry")
    # This is the original mojolearn_sqrt C ABI, connected once at import.
    # No replacement square-root approximation or Python math wrapper runs.
    return py_call(entry, x)


def _generator_div(numerator: PythonObject, denominator: PythonObject) raises -> PythonObject:
    if Bool(py=denominator == 0):
        raise Error("ZeroDivisionError: float division by zero")
    return numerator / denominator


def _generator_run(entry: PythonObject, shape: PythonObject, count: PythonObject,
                   kind: Int, offset: PythonObject, seed: PythonObject,
                   stream: PythonObject, a: PythonObject, b: PythonObject,
                   input: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var out = checkpoint_array(py_call(bi.bytes, count * PythonObject(4)), shape, "f", context["neural"]["array_class"])
    var source = input
    if source is None:
        source = checkpoint_array(py_call(bi.bytes, 4), Python.tuple(1), "f", context["neural"]["array_class"])
    # Python integer primitives here preserve the legacy two's-complement
    # low/high split even if a caller edited the public seed field directly.
    var lo = seed.__and__(PythonObject(4294967295))
    var hi = seed.__rshift__(32).__and__(PythonObject(4294967295))
    var params = Python.list(count, py_call(bi.int, offset), py_call(bi.int, lo), py_call(bi.int, hi), py_call(bi.int, stream), kind, py_call(bi.float, a), py_call(bi.float, b))
    try:
        _ = py_call(entry, Python.list(out._addr, source._addr), params)
    except error:
        # Existing def_function RNG entries report numerical admission through
        # the base Exception type; preserve that public failure contract.
        raise Error("Exception: " + String(error))
    _ = source^
    return out


def _dropout_key_field(key: PythonObject, name: String, integer: Bool) raises -> PythonObject:
    """One field of a dropout key, refused by name when it is missing or not
    the exact builtin type `dropout` stored (an int, or a float)."""
    var bi = Python.import_module("builtins")
    var value = py_get(key, PythonObject(name))
    if value is None:
        raise Error("ValueError: Generator.dropout_backward: key is missing '" + name + "'; pass the key dropout returned")
    if integer and py_type(value) is not bi.int:
        raise Error("TypeError: Generator.dropout_backward: key['" + name + "'] must be an int")
    if not integer and py_type(value) is not bi.float:
        raise Error("TypeError: Generator.dropout_backward: key['" + name + "'] must be a float")
    return value


def _dropout_key(key: PythonObject) raises -> PythonObject:
    """The key `dropout` returned, checked field by field before any draw."""
    var bi = Python.import_module("builtins")
    if py_type(key) is not bi.dict:
        raise Error("TypeError: Generator.dropout_backward: key must be the dict dropout returned")
    _ = _dropout_key_field(key, "seed", True)
    _ = _dropout_key_field(key, "stream", True)
    _ = _dropout_key_field(key, "p", False)
    _ = _dropout_key_field(key, "scale", False)
    _ = _dropout_key_field(key, "offset", True)
    return key


def generator_draw(operation: String, values: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    _ = generation_initialize_binding(context["neural"]["generation_classes"])
    var owner = values[0]
    var fields = generator_fields(owner)
    # Builtin binary64 scalar operations retain Python's separate rounding
    # boundaries and evaluation order. The loops, ABI calls, and dispatch
    # remain in this native owner, including the following draw operation.
    if operation == "kaiming_uniform":
        var slope = values[3]
        if slope is None:
            slope = _generator_sqrt(PythonObject(5.0), context)
        var square = py_call(bi.float, slope) * py_call(bi.float, slope)
        var gain = _generator_sqrt(_generator_div(PythonObject(2.0), PythonObject(1.0) + square), context)
        var bound = gain * _generator_sqrt(_generator_div(PythonObject(3.0), py_call(bi.float, py_call(bi.int, values[2]))), context)
        return generator_draw("uniform", Python.tuple(owner, values[1], -bound, bound), context)
    if operation == "xavier_uniform" or operation == "xavier_normal":
        var gain = py_call(bi.float, values[4])
        var fan = py_call(bi.float, py_call(bi.int, values[2]) + py_call(bi.int, values[3]))
        var ratio = _generator_div(PythonObject(6.0 if operation == "xavier_uniform" else 2.0), fan)
        var scale = gain * _generator_sqrt(ratio, context)
        if operation == "xavier_uniform":
            return generator_draw("uniform", Python.tuple(owner, values[1], -scale, scale), context)
        return generator_draw("normal", Python.tuple(owner, values[1], 0.0, scale), context)
    if operation == "kaiming_normal":
        var gain = values[3]
        if gain is None:
            gain = _generator_sqrt(PythonObject(2.0), context)
        var numerator = py_call(bi.float, gain)
        var denominator = _generator_sqrt(py_call(bi.float, py_call(bi.int, values[2])), context)
        var scale = _generator_div(numerator, denominator)
        return generator_draw("normal", Python.tuple(owner, values[1], 0.0, scale), context)
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
        var stream = generator_next(owner) if values[4] is None else py_call(bi.int, values[4])
        var key = bi.dict()
        py_setitem(key, "seed", fields["seed"])
        py_setitem(key, "stream", stream)
        py_setitem(key, "p", Float64(probability))
        py_setitem(key, "scale", Float64(scale))
        py_setitem(key, "offset", py_call(bi.int, values[3]))
        # Dropout consumed above before converting offset or selecting mode.
        var entry = _generator_binding(owner, context)
        var out = _generator_run(entry, x.shape, x.size, 2, key["offset"], key["seed"], stream,
                                 key["p"], key["scale"], x, context)
        return Python.tuple(out, key)
    if operation == "dropout_backward":
        var dy = _generator_input(values[1], context)
        var key = _dropout_key(values[2])
        var entry = _generator_binding(owner, context)
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
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        if message.startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        if message.startswith("ImportError:"):
            return raise_python_exception(error_without_class(error, "ImportError"), ExceptionType("PyExc_ImportError"))
        if message.startswith("OSError:"):
            return raise_python_exception(error_without_class(error, "OSError"), ExceptionType("PyExc_OSError"))
        if message.startswith("ZeroDivisionError:"):
            return raise_python_exception(error_without_class(error, "ZeroDivisionError"), ExceptionType("PyExc_ZeroDivisionError"))
        if message.startswith("Exception:"):
            return raise_python_exception(error_without_class(error, "Exception"), ExceptionType("PyExc_Exception"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
