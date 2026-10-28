# SPDX-License-Identifier: Apache-2.0
"""Shared pieces of the native owner of the sequence-expansion estimators
(bindings/xseq_native.mojo): the numeric-mode route, input admission and
float32 casts, zeroed outputs, the seeded initialization stream, and the
result views the public methods return.

Python connects the compiled `_mojolearn_x_sequence` entries once, at import
(`python/mojolearn/_native_xseq_bridge.py`); every public method is one call
into `xseq_native_call`, which runs the same compiled entries, in the same
order and with the same arguments, as the Python bodies it replaces.
"""
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_fields, py_contains
from bindings.py_typed_call import py_call_typed
from bindings.sequential_arrays import sequential_array, sequential_address
from bindings.sequential_inputs import sequential_input
from providers.python.neural_descriptor_pyapi import neural_descriptor_reuse, neural_descriptor_execute
from sequence.neural_model_dispatch import MODEL_OPERATION_COUNT, MODEL_UNIFORM
from autograd.model_descriptor import MODEL_ABI_NAME
from core.cast_in import cast_in_kind
from core.xseq_marshal import xseq_cast_nd, xseq_column_copy_f32, xseq_column_copy_i64

comptime XF32P = MutPointer[Float32, MutUntrackedOrigin]
comptime XI32P = MutPointer[Int32, MutUntrackedOrigin]
comptime XI64P = MutPointer[Int64, MutUntrackedOrigin]
comptime XU8P = MutPointer[UInt8, MutUntrackedOrigin]


# ---------------------------------------------------------------- the route

def xseq_mode(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    """The model's numeric mode, or the process default, as `_backend.binding`
    resolves it."""
    var b = Python.import_module("builtins")
    var mode = py_get(fields, "numeric_mode")
    if mode is None:
        mode = context["backend_fields"]["_DEFAULT_MODE"]
    if not Bool(py=py_call(b.isinstance, mode, b.str)):
        raise Error("numeric_mode must be fast, identical or None")
    mode = py_call(py_call(b.getattr, py_call(py_call(b.getattr, mode, "strip")), "lower"))
    if String(py=mode) != "identical" and String(py=mode) != "fast":
        raise Error("numeric_mode must be fast, identical or None")
    return mode


def xseq_entries(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    """The compiled `_mojolearn_x_sequence` entries of the model's mode,
    connected at import; the connection error of that mode otherwise."""
    var b = Python.import_module("builtins")
    var mode = xseq_mode(fields, context)
    var entries = context["xseq_bindings"][mode]
    if Bool(py=py_call(b.isinstance, entries, b.str)):
        raise Error(String(py=entries))
    var actual = Int(py=py_call_typed(entries["x_sequence_numeric_mode"]))
    if actual != (1 if String(py=mode) == "identical" else 0):
        raise Error("RuntimeError: mojolearn: _mojolearn_x_sequence was compiled for the other numeric mode; rebuild the "
                    + String(py=mode) + " binding")
    return entries


def xseq_neural_provider(entries: PythonObject) raises -> PythonObject:
    # Immutable engine table only. Inputs/weights are admitted afresh by callers.
    var provider = neural_descriptor_reuse(py_get(entries, "native_model_provider"),
        entries["model_native_api_v1"], String(MODEL_ABI_NAME), MODEL_OPERATION_COUNT)
    py_setitem(entries, "native_model_provider", provider)
    return provider


def xseq_output_provider(context: PythonObject) raises -> PythonObject:
    # Preserve the old base-helper route: default mode, independent of a
    # particular estimator's optional numeric_mode override.
    return xseq_neural_provider(context["xseq_bindings"][context["backend_fields"]["_DEFAULT_MODE"]])


def xseq_base(context: PythonObject, name: String) raises -> PythonObject:
    """A compiled entry of the base binding, connected at import."""
    return context["xseq_base"][PythonObject(name)]


# ---------------------------------------------------------------- inputs

def xseq_input(value: PythonObject, context: PythonObject) raises -> PythonObject:
    """`np.asarray(value)` for a numeric input, without running any Python:
    an ndarray as it is, a mojolearn Array, buffer or nested sequence read
    natively (bindings/sequential_inputs.mojo)."""
    return sequential_input(value, context)


def xseq_is_f64(value: PythonObject) raises -> Bool:
    """`value.dtype == np.float64`."""
    var dtype = value.dtype
    return Int(py=dtype.num) == 12 and Bool(py=dtype.isnative)


def xseq_ndim(value: PythonObject) raises -> Int:
    return Int(py=value.ndim)


def xseq_dim(value: PythonObject, axis: Int) raises -> Int:
    return Int(py=value.shape[axis])


def xseq_f32(value: PythonObject, name: String, copy: Bool, context: PythonObject,
             double_ints: Bool = False) raises -> PythonObject:
    """`np.array(value, dtype=np.float32, order="C", copy=copy)` of an
    admitted ndarray: the array itself when it is already C-ordered float32
    and no copy is asked for, otherwise one C-ordered float32 copy, element by
    element with the shared integer-defined conversion (`cast_f32_bits`:
    `numpy.astype(float32)`'s bits; `double_ints` takes `Array.astype`'s
    integer -> double -> float32 route, `classical_f32_bits`)."""
    var dtype = value.dtype
    var kind = String(py=dtype.kind)
    if kind != "b" and kind != "i" and kind != "u" and kind != "f":
        raise Error("TypeError: " + name + " must be a numeric array")
    if not Bool(py=dtype.isnative):
        raise Error("TypeError: " + name + " must be in native byte order")
    var rank = Int(py=value.ndim)
    var c_order = Bool(py=value.flags.c_contiguous)
    if not copy and c_order and Int(py=dtype.num) == 11:
        return value
    var shape = value.shape
    var out = sequential_array(shape, context["float32"], context)
    var item = Int(py=dtype.itemsize)
    var cast = cast_in_kind(Int(py=dtype.num), item, name)
    var dims = List[Int]()
    var steps = List[Int]()
    for k in range(rank):  # driver-loop: over array rank metadata
        dims.append(Int(py=shape[k]))
        steps.append(Int(py=value.strides[k]))
    xseq_cast_nd(Int(py=sequential_address(value)), dims, steps, item, cast, double_ints,
                 XF32P(unsafe_from_address=Int(py=sequential_address(out))))
    _ = out  # keep the Python object alive through the native call above
    return out


def xseq_refuse_f64(value: PythonObject, message: String) raises:
    """`if y.dtype == np.float64: raise TypeError(message)`."""
    if xseq_is_f64(value):
        raise Error("TypeError: " + message)


# ---------------------------------------------------------------- outputs

def xseq_zeros(shape: PythonObject, context: PythonObject) raises -> PythonObject:
    """`np.zeros(shape, dtype=np.float32)`."""
    return sequential_array(shape, context["float32"], context)


def xseq_zeros_i64(shape: PythonObject, context: PythonObject) raises -> PythonObject:
    return sequential_array(shape, context["int64"], context)


def xseq_addr(value: PythonObject) raises -> PythonObject:
    return sequential_address(value)


def xseq_f32p(value: PythonObject) raises -> XF32P:
    return XF32P(unsafe_from_address=Int(py=sequential_address(value)))


def xseq_reshape(value: PythonObject, shape: PythonObject) raises -> PythonObject:
    """`value.reshape(shape)`: a view of the same words (ndarray's own C
    method)."""
    var b = Python.import_module("builtins")
    return py_call(py_call(b.getattr, value, "reshape"), shape)


def xseq_copy(value: PythonObject) raises -> PythonObject:
    """`value.copy()` (ndarray's own C method)."""
    var b = Python.import_module("builtins")
    return py_call(py_call(b.getattr, value, "copy"))


def xseq_item(value: PythonObject, index: Int) raises -> PythonObject:
    """`value[index]` (ndarray's own C subscript): a NumPy scalar of a 1-D
    array, a row view of a 2-D one."""
    var b = Python.import_module("builtins")
    return py_call(py_call(b.getattr, value, "__getitem__"), index)


def xseq_first(value: PythonObject, one: Bool) raises -> PythonObject:
    """`value[0] if one else value`: the single series' row of a batch result."""
    return xseq_item(value, 0) if one else value


def xseq_column_f32(value: PythonObject, rows: Int, cols: Int, col: Int, context: PythonObject) raises -> PythonObject:
    """`value[:, col]` of a C-ordered (rows, cols) float32 array, as its own
    C-ordered (rows,) float32 array (the same words)."""
    var out = xseq_zeros(Python.tuple(rows), context)
    xseq_column_copy_f32(xseq_f32p(value), xseq_f32p(out), rows, cols, col)
    _ = out  # keep the Python object alive through the native call above
    return out


def xseq_column_i64(value: PythonObject, rows: Int, cols: Int, col: Int, context: PythonObject) raises -> PythonObject:
    """`value[:, col].astype(np.int64)` of a C-ordered float32 array
    (truncation toward zero, as astype)."""
    var out = xseq_zeros_i64(Python.tuple(rows), context)
    xseq_column_copy_i64(xseq_f32p(value), XI64P(unsafe_from_address=Int(py=sequential_address(out))), rows, cols, col)
    _ = out  # keep the Python object alive through the native call above
    return out


def xseq_curve_list(value: PythonObject, count: Int) raises -> PythonObject:
    """The first `count` float32 words of `value` as a list of floats
    (`curve[:count].tolist()`)."""
    var result = Python.list()
    var p = xseq_f32p(value)
    var b = Python.import_module("builtins")
    for k in range(count):  # driver-loop: over the native training engine's per-epoch scalar report
        _ = py_call(py_call(b.getattr, result, "append"), PythonObject(Float64(p[k])))
    return result


def xseq_mean_dict(forecast: PythonObject) raises -> PythonObject:
    """`{"mean": forecast}`."""
    var result = Python.dict()
    py_setitem(result, "mean", forecast)
    return result


# ---------------------------------------------------------------- scalars

def xseq_int(value: PythonObject) raises -> Int:
    """`int(value)` (the builtin conversion, its own TypeError/ValueError)."""
    var b = Python.import_module("builtins")
    return Int(py=py_call_typed(b.int, value))


def xseq_float(value: PythonObject) raises -> Float64:
    """`float(value)`."""
    var b = Python.import_module("builtins")
    return Float64(py=py_call_typed(b.float, value))


def xseq_truth(value: PythonObject) raises -> Bool:
    """`bool(value)`."""
    var b = Python.import_module("builtins")
    return Bool(py=py_call_typed(b.bool, value))


def xseq_is_int(value: PythonObject, context: PythonObject) raises -> Bool:
    """`isinstance(value, (int, np.integer))`."""
    var b = Python.import_module("builtins")
    return Bool(py=py_call(b.isinstance, value, context["int_types"]))


# ---------------------------------------------------------------- the seeded stream

struct XseqStream(Movable):
    """`_buffer.InitStream`: a seeded stream of uniform float32 draws made by
    the base binding's counter-based `uniform_init_f32`; only the counter
    lives here."""
    var seed: UInt64
    var offset: Int

    def __init__(out self, random_state: PythonObject, context: PythonObject) raises:
        var b = Python.import_module("builtins")
        var word: PythonObject
        if random_state is None:
            # `secrets.randbits(64)`: eight bytes of the OS generator
            var raw = py_call(context["os_urandom"], 8)
            word = py_call(py_call(b.getattr, b.int, "from_bytes"), raw, "little")
        else:
            word = py_call_typed(b.int, random_state)
        var low = Int(py=word.__and__(PythonObject(4294967295)))
        var high = Int(py=(word.__rshift__(PythonObject(32))).__and__(PythonObject(4294967295)))
        self.seed = (UInt64(high) << 32) | UInt64(low)
        self.offset = 0

    def fill_uniform(mut self, context: PythonObject, address: PythonObject, n: Int, low: Float64, high: Float64) raises:
        """n float32 values U(low, high) at `address`; advances the counter."""
        if n:
            _ = py_call_typed(xseq_base(context, "uniform_init_f32"), address, PythonObject(n), PythonObject(low),
                              PythonObject(high), PythonObject(Int(self.seed & 0xFFFFFFFF)),
                              PythonObject(Int(self.seed >> 32)), PythonObject(self.offset))
        self.offset += n

    def fill_uniform_native(mut self, provider: PythonObject, address: Int, n: Int, low: Float64, high: Float64) raises:
        if n:
            var addresses: List[Int] = [address]
            var options: List[Int] = [n, Int(self.seed & 0xFFFFFFFF), Int(self.seed >> 32), self.offset]
            var scalars: List[Float64] = [low, high]
            _ = neural_descriptor_execute(provider, MODEL_UNIFORM, addresses, options, scalars)
        self.offset += n

    def child_seed(mut self) -> UInt64:
        """A derived 63-bit seed (splitmix64 of the seed and the counter),
        advancing the counter by one."""
        var z = self.seed + UInt64(self.offset + 1) * UInt64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
        self.offset += 1
        return (z ^ (z >> 31)) >> 1
