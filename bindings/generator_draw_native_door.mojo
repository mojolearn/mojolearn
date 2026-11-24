# SPDX-License-Identifier: Apache-2.0
"""The public door of the training Generator's draws: UNPACK, RUN, PACK.

UNPACK (this file): the public arguments, the Generator's fields and the
draw route become typed words and one `GeneratorDrawRequest`.
RUN (`core/generator_draw_run.mojo`, no Python): the initializer scalars and
one call of neural primitive operation 6 through the training image's entry
of the Samba table, resolved once per numeric mode for the process.
PACK (this file): the Array the draw wrote, and the dropout key.

The former `neural_rng` Python entry (two Python lists per draw) and the
portable-math scalar entry are not called. A Generator draws on the
installation's training image, or (lane vdf-sched: constructed with an
explicit `device="cpu"` beside a GPU set, bindings/generator_device_unpack.mojo)
on the host training image, the table's vendor-0 entry.
"""
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from std.python._cpython import PyObjectPtr, GILReleased
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_get, py_setitem, py_type
from bindings.generator_state_native_unpack import generator_fields
from bindings.generator_state_native_pack import generator_next
from bindings.causal_checkpoint_door import checkpoint_array
from bindings.causal_lm_construct_door import _model_f32
from bindings.generation_buffers_door import generation_initialize_binding, generation_array, GENERATION_BOUNDARY
from core.shared_slot import SharedSlot
from bindings.py_optional import py_present
from core.entry_table import EntryRef, entry_ref_unresolved
from core.device_route import device_cpu_fast
from core.samba_entry_table import SambaEntryKey, samba_ref, samba_vendor_word, SAMBA_TABLE_TRAINING
from core.generator_draw_run import (
    GeneratorDrawRequest, generator_draw_run,
    generator_kaiming_uniform_bound, generator_xavier_scale, generator_kaiming_normal_std,
    generator_uniform_span, generator_dropout_scale,
    GENERATOR_UNIFORM, GENERATOR_NORMAL, GENERATOR_DROPOUT, GENERATOR_DROPOUT_BACKWARD,
)


def _generator_shape(shape: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = _builtins()
    # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
    var scalar = Optional[PythonObject]()
    # Row N202: exact types are decided without the index probe and its
    # swallowed exception: an exact tuple or list never has __index__, and an
    # exact int is its own index. Every other object keeps the probe.
    var kind = py_type(shape)
    if kind is bi.int:
        scalar = py_present(shape)
    elif kind is not bi.tuple and kind is not bi.list:
        try:
            scalar = py_present(py_call(context["generator_index"], shape))
        except:
            pass
    var source = Python.tuple(scalar.value()) if Bool(scalar) else py_call(bi.tuple, shape)
    var dims = bi.list()
    # Optional, not a PythonObject read and reassigned in a loop (modular/modular#7546 pattern; runtime-fix-save-forest sweep).
    var count = Optional[PythonObject](PythonObject(1))
    for i in range(len(source)):  # driver-loop: over caller shape metadata
        var d = py_call(bi.int, source[i])
        if Bool(py=d < 1):
            raise Error("Generator: shape must be positive")
        _ = py_call(dims.append, d)
        count = Optional[PythonObject](count.value() * d)
    return Python.tuple(py_call(bi.tuple, dims), count.value())


struct GeneratorDrawBoundary(Defaultable, Movable):
    # The builtins module (a process constant) and the draw route of each
    # numeric mode: the training image's primitive entry in the Samba table
    # (core/samba_entry_table.mojo), resolved at the mode's first draw and
    # held for the process. The table is keyed by (mode, vendor) and each
    # image publishes under its own compiled words, so a held entry IS the
    # vendor and mode admission the door used to repeat by calling the
    # binding's two read-back entries.
    # `host`: the host training image's identical entry (vendor 0), for a
    # Generator constructed with device="cpu" beside a GPU set (lane vdf-sched).
    var builtins: PythonObject
    var fast: EntryRef
    var identical: EntryRef
    var host: EntryRef
    var has_fast: Bool
    var has_identical: Bool
    var has_host: Bool

    def __init__(out self):
        self.builtins = PythonObject(None)
        self.fast = entry_ref_unresolved()
        self.identical = entry_ref_unresolved()
        self.host = entry_ref_unresolved()
        self.has_fast = False
        self.has_identical = False
        self.has_host = False


# V4 (lane vdf-sched): the layout gained the host entry.
comptime GENERATOR_DRAW_BOUNDARY = SharedSlot[GeneratorDrawBoundary, "MojolearnGeneratorDrawBoundaryV4"]


def _builtins() raises -> PythonObject:
    var g = GENERATOR_DRAW_BOUNDARY.get_or_create_ptr()
    if g[].builtins is None:
        g[].builtins = native_loaded_module("builtins")
    return g[].builtins


def _generator_mode_word(text: String) -> Int:
    if text == "identical":
        return 1
    if text == "fast":
        return 0
    return -1


def _generator_route(fields: PythonObject, context: PythonObject) raises -> EntryRef:
    """UNPACK: the draw route of the Generator's numeric mode. A Generator
    without a mode of its own follows the library default, which
    `set_numeric_mode` may change, so that one word is read here; nothing
    else is asked of the process once the mode's route is held."""
    var mode = fields["numeric_mode"]
    if not Bool(py=mode):
        mode = context["generator_backend"]._DEFAULT_MODE
    var text = String(mode)
    var word = _generator_mode_word(text)
    if word < 0:
        # Another spelling (case, surrounding blanks) or an unknown word.
        text = String(mode.strip().lower())
        word = _generator_mode_word(text)
        if word < 0:
            raise Error("Generator: native RNG binding is unavailable for numeric_mode=" + text)
    var g = GENERATOR_DRAW_BOUNDARY.get_or_create_ptr()
    # lane vdf-sched: the one field construction stored for device="cpu"
    # beside a GPU set (absent on every other Generator: one dictionary read).
    if py_get(fields, PythonObject("_host_device")) is not None:
        if word != 1:
            device_cpu_fast()
        if g[].has_host:
            return g[].host
        var host = entry_ref_unresolved()
        try:
            host = samba_ref[SAMBA_TABLE_TRAINING](SambaEntryKey(1, 0))
        except problem:
            raise Error("ImportError: Generator device='cpu' needs the CPU identical training image beside the GPU set: "
                        + String(problem))
        g[].host = host
        g[].has_host = True
        return host
    if word == 1 and g[].has_identical:
        return g[].identical
    if word == 0 and g[].has_fast:
        return g[].fast
    # First draw of this mode: the connection record says whether the mode's
    # training image is connected (a reason string or nothing when it is
    # not); the table gives its entry.
    var connected = py_get(context["generator_bindings"], PythonObject(text))
    if connected is None or py_type(connected) is _builtins().str:
        raise Error("Generator: native RNG binding is unavailable for numeric_mode=" + text)
    var selected = context["vendor"]
    var vendor = samba_vendor_word(String("cpu") if selected is None else String(selected))
    if vendor < 0:
        raise Error("ImportError: Generator native RNG vendor or numeric mode differs from the selected route")
    var entry = entry_ref_unresolved()
    try:
        entry = samba_ref[SAMBA_TABLE_TRAINING](SambaEntryKey(word, vendor))
    except problem:
        raise Error("ImportError: Generator native RNG vendor or numeric mode differs from the selected route: "
                    + String(problem))
    if word == 1:
        g[].identical = entry
        g[].has_identical = True
    else:
        g[].fast = entry
        g[].has_fast = True
    return entry


def _generator_f32(value: PythonObject) raises -> Float32:
    return Float32(Float64(py=py_call(_builtins().float, value)))


def _generator_input(value: PythonObject, context: PythonObject) raises -> PythonObject:
    # The shared native packer preserves f32 bytes and converts arbitrary
    # buffer strides/layout into C order without Array or NumPy callbacks.
    var result = _model_f32(value, "Generator input", context["neural"])
    # The former _c32/as_f32_c admission rejects empty inputs before stream
    # consumption. Nonfinite input remains the existing kernel's admission.
    if Int(py=result.size) == 0:
        raise Error("Generator input is empty; at least one element is required")
    return result


def _generator_f64(value: PythonObject) raises -> Float64:
    return Float64(py=py_call(_builtins().float, value))


def _generator_fan(value: PythonObject) raises -> Float64:
    """float(int(value)), by the builtin conversions the public call always
    used (an index-like or numeric fan, any size)."""
    var bi = _builtins()
    return Float64(py=py_call(bi.float, py_call(bi.int, value)))


def _generator_emit(entry: EntryRef, shape: PythonObject, count: PythonObject,
                    kind: Int, offset: PythonObject, seed: PythonObject,
                    stream: PythonObject, a: PythonObject, b: PythonObject,
                    input: PythonObject, context: PythonObject) raises -> PythonObject:
    """The Array of one draw. `offset`, `seed`, `stream`, `a` and `b` are the
    public values (the dropout key's own, or the door's): they become the
    request's words here, and a value the words cannot hold is the draw's
    refusal, as it was the entry's."""
    var bi = _builtins()
    var array_class = context["neural"]["array_class"]
    var target = PythonObject(None)
    var source = input
    var total = Int(py=count)
    if GENERATION_BOUNDARY.get_or_create_ptr()[].array_cls is array_class and total <= 2147483647:
        # One native zero-filled store wrapped directly (no zero bytes object
        # copied into a second store); same class, shape, dtype and strides.
        var dims = List[Int]()
        for i in range(len(shape)):  # small-loop(array shape dimension words)
            dims.append(Int(py=shape[i]))
        target = generation_array(dims, "f")
        if source is None:
            source = generation_array([1], "f")
    else:
        target = checkpoint_array(py_call(bi.bytes, count * PythonObject(4)), shape, "f", array_class)
        if source is None:
            source = checkpoint_array(py_call(bi.bytes, 4), Python.tuple(1), "f", array_class)
    # UNPACK. Python integer primitives preserve the two's-complement
    # low/high split even if a caller edited the public seed field.
    var low = py_call(bi.int, seed.__and__(PythonObject(4294967295)))
    var high = py_call(bi.int, seed.__rshift__(32).__and__(PythonObject(4294967295)))
    var first = py_call(bi.int, offset)
    var which = py_call(bi.int, stream)
    var pa = py_call(bi.float, a)
    var pb = py_call(bi.float, b)
    try:
        var request = GeneratorDrawRequest(
            entry, Int(py=target._addr), Int(py=source._addr), total, Int(py=first),
            Int(py=low), Int(py=high), Int(py=which), kind, Float64(py=pa), Float64(py=pb))
        # RUN, with the GIL released as the former entry ran it.
        with GILReleased(Python()):
            _ = generator_draw_run(request)
    except error:
        # The former RNG entry reported its refusals through the base
        # Exception type; preserve that public failure contract.
        raise Error("Exception: " + String(error))
    # Both Arrays are read by address above: keep them past the call.
    _ = source
    _ = target
    return target


def _generator_fresh(owner: PythonObject, fields: PythonObject, shaped: PythonObject, kind: Int,
                     a: Float64, b: Float64, context: PythonObject) raises -> PythonObject:
    """A draw of the next stream with words the door already holds: the
    route refusal precedes stream consumption, allocation and kernel
    failures follow it."""
    var entry = _generator_route(fields, context)
    var seed = fields["seed"]
    var stream = generator_next(owner)
    return _generator_emit(entry, shaped[0], shaped[1], kind, PythonObject(0), seed, stream,
                           PythonObject(a), PythonObject(b), PythonObject(None), context)


def _generator_uniform(owner: PythonObject, fields: PythonObject, shaped: PythonObject,
                       low: Float32, high: Float32, context: PythonObject) raises -> PythonObject:
    var span = generator_uniform_span(low, high)
    return _generator_fresh(owner, fields, shaped, GENERATOR_UNIFORM, Float64(low), Float64(span), context)


def _generator_normal(owner: PythonObject, fields: PythonObject, shaped: PythonObject,
                      mean: Float32, deviation: Float32, context: PythonObject) raises -> PythonObject:
    if not (deviation >= 0):
        raise Error("Generator.normal: std must be >= 0")
    return _generator_fresh(owner, fields, shaped, GENERATOR_NORMAL, Float64(mean), Float64(deviation), context)


def _dropout_key_field(key: PythonObject, name: String, integer: Bool) raises -> PythonObject:
    """One field of a dropout key, refused by name when it is missing or not
    the exact builtin type `dropout` stored (an int, or a float)."""
    var bi = _builtins()
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
    var bi = _builtins()
    if py_type(key) is not bi.dict:
        raise Error("TypeError: Generator.dropout_backward: key must be the dict dropout returned")
    _ = _dropout_key_field(key, "seed", True)
    _ = _dropout_key_field(key, "stream", True)
    _ = _dropout_key_field(key, "p", False)
    _ = _dropout_key_field(key, "scale", False)
    _ = _dropout_key_field(key, "offset", True)
    return key


def generator_draw(operation: String, values: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = _builtins()
    _ = generation_initialize_binding(context["neural"]["generation_classes"])
    var owner = values[0]
    var fields = generator_fields(owner)
    # The initializers: UNPACK converts each public scalar by the builtin
    # conversion it always had, in the same order; the bound or deviation is
    # binary64 in RUN (core/generator_draw_run.mojo); the shape is admitted
    # after it, then the draw.
    if operation == "kaiming_uniform":
        var has_slope = values[3] is not None
        var slope = _generator_f64(values[3]) if has_slope else Float64(0.0)
        var bound = generator_kaiming_uniform_bound(has_slope, slope, _generator_fan(values[2]))
        var shaped = _generator_shape(values[1], context)
        return _generator_uniform(owner, fields, shaped, Float32(-bound), Float32(bound), context)
    if operation == "xavier_uniform" or operation == "xavier_normal":
        var gain = _generator_f64(values[4])
        var fan = Float64(py=py_call(bi.float, py_call(bi.int, values[2]) + py_call(bi.int, values[3])))
        var scale = generator_xavier_scale(operation == "xavier_uniform", gain, fan)
        var shaped = _generator_shape(values[1], context)
        if operation == "xavier_uniform":
            return _generator_uniform(owner, fields, shaped, Float32(-scale), Float32(scale), context)
        return _generator_normal(owner, fields, shaped, Float32(0.0), Float32(scale), context)
    if operation == "kaiming_normal":
        var has_gain = values[3] is not None
        var gain = _generator_f64(values[3]) if has_gain else Float64(0.0)
        var scale = generator_kaiming_normal_std(has_gain, gain, _generator_fan(values[2]))
        var shaped = _generator_shape(values[1], context)
        return _generator_normal(owner, fields, shaped, Float32(0.0), Float32(scale), context)
    if operation == "uniform":
        var shaped = _generator_shape(values[1], context)
        var lo = _generator_f32(values[2])
        var hi = _generator_f32(values[3])
        return _generator_uniform(owner, fields, shaped, lo, hi, context)
    if operation == "normal":
        var shaped = _generator_shape(values[1], context)
        var sd = _generator_f32(values[3])
        if not (sd >= 0):
            raise Error("Generator.normal: std must be >= 0")
        var entry = _generator_route(fields, context)
        var seed = fields["seed"]
        var stream = generator_next(owner)
        # The public normal consumes its stream before converting the mean.
        var mean = _generator_f32(values[2])
        return _generator_emit(entry, shaped[0], shaped[1], GENERATOR_NORMAL, PythonObject(0), seed, stream,
                               PythonObject(Float64(mean)), PythonObject(Float64(sd)), PythonObject(None), context)
    if operation == "dropout":
        var probability = _generator_f32(values[2])
        var scale = generator_dropout_scale(probability)
        var x = _generator_input(values[1], context)
        var stream = generator_next(owner) if values[4] is None else py_call(bi.int, values[4])
        var key = bi.dict()
        py_setitem(key, "seed", fields["seed"])
        py_setitem(key, "stream", stream)
        py_setitem(key, "p", Float64(probability))
        py_setitem(key, "scale", Float64(scale))
        py_setitem(key, "offset", py_call(bi.int, values[3]))
        # Dropout consumed above before converting offset or selecting mode.
        var entry = _generator_route(fields, context)
        var result = _generator_emit(entry, x.shape, x.size, GENERATOR_DROPOUT, key["offset"], key["seed"], stream,
                                     key["p"], key["scale"], x, context)
        return Python.tuple(result, key)
    if operation == "dropout_backward":
        var dy = _generator_input(values[1], context)
        var key = _dropout_key(values[2])
        var entry = _generator_route(fields, context)
        return _generator_emit(entry, dy.shape, dy.size, GENERATOR_DROPOUT_BACKWARD, key["offset"], key["seed"],
                               key["stream"], key["p"], key["scale"], dy, context)
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
