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
from bindings.mode_answer_door import answer_mode_object
from core.mode_answer import MODE_FAMILY_XSEQ
from std.ffi import c_int
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.native_loaded_module_door import native_loaded_module
from bindings.sequential_admission_door import _one_of
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_fields, py_contains
from bindings.py_typed_call import py_call_typed
from std.python._cpython import GILReleased
from core.entry_table import EntryRef
from core.xseq_entry_table import XseqEntryKey, xseq_glue_ref
from sequence.xseq_door_run import xseq_run_glue
from bindings.sequential_arrays_door import sequential_array, sequential_address
from bindings.sequential_inputs_door import sequential_input
from providers.python.neural_descriptor_pyapi_door import neural_descriptor_reuse, neural_descriptor_execute
from sequence.neural_model_dispatch import MODEL_OPERATION_COUNT, MODEL_UNIFORM
from autograd.model_descriptor import MODEL_ABI_NAME
from core.xseq_marshal import xseq_column_copy_f32, xseq_column_copy_i64
from sequence.xseq_native_common_run import xseq_stream_child_seed
from bindings.xseq_keys_unpack import xseq_keys
from bindings.py_balanced import py_item
from core.device_route import device_mode_code, device_route_plan, device_route_admit
from providers.python.device_route_boundary_door import (
    DeviceRouteOwner, device_route_held, device_route_keep, device_request,
)

comptime XF32P = MutPointer[Float32, MutUntrackedOrigin]
comptime XI32P = MutPointer[Int32, MutUntrackedOrigin]
comptime XI64P = MutPointer[Int64, MutUntrackedOrigin]
comptime XU8P = MutPointer[UInt8, MutUntrackedOrigin]


# ---------------------------------------------------------------- the route

def _xseq_mode_resolve(mode_field: PythonObject, context: PythonObject) raises -> PythonObject:
    """A numeric-mode field value, or the process default for None, as
    `_backend.binding` resolves it. Runs when a route is resolved, never per
    operation."""
    var b = native_loaded_module("builtins")
    var nk = xseq_keys()
    var mode = mode_field
    if mode is None:
        mode = py_item(py_item(context, nk[].k_backend_fields), nk[].k__DEFAULT_MODE)
    # perf3-seq-neural: an exact str that already is one of the two modes
    # (every default and every documented value) is returned as it is: the
    # strip().lower() below would give an equal str, so the bindings lookup
    # and the result are the same, without four Python calls per predict.
    if py_type(mode) is b.str:
        var text = String(py=mode)
        if text == "identical" or text == "fast":
            answer_mode_object(mode, text, MODE_FAMILY_XSEQ)
            return mode
    if not Bool(py=py_call(b.isinstance, mode, b.str)):
        raise Error("numeric_mode must be fast, identical or None")
    mode = py_call(py_call(b.getattr, py_call(py_call(b.getattr, mode, "strip")), "lower"))
    var mode_word = String(py=mode)
    if mode_word != "identical" and mode_word != "fast":
        raise Error("numeric_mode must be fast, identical or None")
    answer_mode_object(mode, mode_word, MODE_FAMILY_XSEQ)
    return mode


def _xseq_key_of(mode: PythonObject, context: PythonObject) raises -> XseqEntryKey:
    """The sequence-table key of the image connected for `mode` (recorded by
    the bridge at import as `xseq_native_keys`), checked against the mode
    asked for; the connection error of that mode otherwise."""
    var nk = xseq_keys()
    var key = py_item(context, nk[].k_xseq_native_keys)[mode]
    if py_type(key) is native_loaded_module("builtins").str:
        raise Error(String(py=key))
    var code = Int(py=key[0])
    if code != (1 if String(py=mode) == "identical" else 0):
        raise Error("RuntimeError: mojolearn: _mojolearn_x_sequence was compiled for the other numeric mode; rebuild the "
                    + String(py=mode) + " binding")
    return XseqEntryKey(code, Int(py=key[1]))


# TOMBSTONE: `xseq_key` and `xseq_mode` (the forecaster doors' per-call glue selection: the mode word
# built and answered, the `xseq_native_keys` read by that word, two Int conversions and the sequence
# table's slot read and admission, at every forecaster fit and forecast; and the mode word re-resolved
# for each stored-forecast key) deleted 2026-11-21 by lane dw-neural-b; the estimator's route owner
# holds the glue entry and the stored-forecast word (`xseq_route`: `route.glue`, `route.word`). Code
# recoverable at 373a48c4d2.


def xseq_glue(entry: EntryRef, operation: Int, addresses: List[Int], ints: List[Int],
              floats: List[Float64]) raises -> Int:
    """One synchronous call of the image's glue entry
    (core/xseq_entry_table.mojo layouts) with the GIL released. The caller
    keeps every addressed buffer alive."""
    var result = 0
    var py = Python()
    with GILReleased(py):
        result = xseq_run_glue(entry, operation, addresses, ints, floats)
    return result


# TOMBSTONE: `xseq_entries` and `xseq_output_provider` (the per-call entry selection of the recurrent
# and MLP doors: the mode word built and compared, the `xseq_bindings` read by that word, the key tuple
# read and converted, at every fit, predict and score, and the default-mode read of each output helper)
# deleted 2026-11-21 by lane dw-neural-b; the estimator's route owner holds the resolved entries
# (`xseq_route`). Code recoverable at 5d94a5ad5d.


@fieldwise_init
struct XseqRoute(Movable):
    """One operation's route of a recurrent or MLP estimator: the compiled
    `_mojolearn_x_sequence` entries of its device and numeric mode, the
    entries its output helpers (weight draws, row argmax, thresholds, the
    curve minimum) run on, the metrics record of the same device, and the
    owner whose pins keep those images loaded until the operation returns."""
    var entries: PythonObject
    var output: PythonObject
    var metrics: PythonObject
    # The tier of the resolved image (1 identical, 0 fast), and whether this
    # is a CPU route beside a GPU set (an explicit `device="cpu"` there).
    var mode: Int
    var cpu_table: Bool
    var owner: PythonObject
    # The image's published glue entry (the forecaster operations), and the
    # word a stored forecast is keyed by: the resolved numeric mode (the same
    # object the key always held) on this installation's device, and that
    # mode marked `/cpu` on a CPU route beside a GPU set, so a stored answer
    # is reused only on the device that made it.
    var glue: EntryRef
    var word: PythonObject


def _xseq_route_build(requested_device: Int, mode_field: PythonObject, default_mode: PythonObject,
    context: PythonObject, b: PythonObject) raises -> DeviceRouteOwner:
    """Resolve an estimator's sequence images once: its numeric mode (None:
    the process mode), its device (None: this installation's), the entries
    and the table key of the image those select (the bridge recorded both at
    import), checked against the selected tier and vendor. Runs at an
    estimator's first operation, after a changed mode and after a new
    `__init__` or `set_params`; never per call. A metadata step with no
    independent units to split."""
    var nk = xseq_keys()
    var owner = DeviceRouteOwner(requested_device)
    var mode = _xseq_mode_resolve(mode_field, context)
    var mode_code = device_mode_code(String(py=mode))
    # The process image of that mode: its vendor word is this process's
    # device (0: a CPU-only installation), and its connection error or
    # other-mode refusal is raised here in the words it always had.
    var key = _xseq_key_of(mode, context)
    var plan = device_route_plan(requested_device, key.vendor, mode_code)
    var entries = PythonObject(None)
    var output = PythonObject(None)
    var metrics = PythonObject(None)
    if plan.cpu_table:
        # A GPU installation running `device="cpu"`: the CPU identical
        # images loaded beside the device images at import
        # (python/mojolearn/_native_xseq_bridge.py).
        var cpu = py_get(context, PythonObject("xseq_cpu_bindings"))
        var cpu_keys = py_get(context, PythonObject("xseq_cpu_native_keys"))
        if cpu is None or cpu_keys is None:
            raise Error("ImportError: device='cpu' is not connected in this installation")
        entries = py_item(cpu, mode)
        if py_type(entries) is b.str:
            raise Error(String(py=entries))
        var cpu_key = py_item(cpu_keys, mode)
        owner.route.mode = Int(py=cpu_key[0])
        owner.route.vendor = Int(py=cpu_key[1])
        device_route_admit("sequence", plan, owner.route.mode, owner.route.vendor, mode_code)
        # The output helpers of a CPU route run on the CPU image too:
        # nothing of this estimator runs on a device its caller did not name.
        output = entries
        var cpu_metrics = py_get(context, PythonObject("xseq_cpu_metrics"))
        if cpu_metrics is not None:
            metrics = py_item(cpu_metrics, mode)
    else:
        var bindings = py_item(context, nk[].k_xseq_bindings)
        entries = py_item(bindings, mode)
        if py_type(entries) is b.str:
            raise Error(String(py=entries))
        owner.route.mode = key.mode
        owner.route.vendor = key.vendor
        # The output helpers keep their image: the process default mode's,
        # whatever the estimator's own numeric_mode is.
        output = py_item(bindings, default_mode)
        metrics = py_item(py_item(context, nk[].k_xseq_metrics), mode)
    owner.route.device = plan.device
    owner.route.flags = 1 if plan.cpu_table else 0
    # The glue entry of the resolved image, read from the sequence table and
    # admitted once (every image publishes it at module init; the table's
    # refusals raise here), and the stored-forecast word of that image. The
    # `cls` pin holds the word: this family keys nothing by class.
    owner.route.entry = xseq_glue_ref(XseqEntryKey(owner.route.mode, owner.route.vendor))
    owner.cls = PythonObject(String(py=mode) + "/cpu") if plan.cpu_table else mode
    owner.mode = mode_field
    owner.default_mode = default_mode
    owner.context = context
    owner.images = entries
    owner.scope = output
    owner.table = metrics
    owner.resolved = True
    return owner^


def xseq_route_of(model: PythonObject, mode: PythonObject, fields: PythonObject,
    context: PythonObject) raises -> XseqRoute:
    """The route of `model` for this operation, for the numeric-mode field
    value `mode` the caller read (an owner that stores it under another name,
    as `VARResults._numeric_mode`). `fields` is the instance dictionary the
    device is read from on a miss. A hit is the default-mode read, one record
    read and three pointer compares; a miss (first operation, a changed mode
    object, a new `__init__` or `set_params`) resolves and stores it. No
    operation reads or compares the device."""
    var nk = xseq_keys()
    var default_mode = py_item(py_item(context, nk[].k_backend_fields), nk[].k__DEFAULT_MODE)
    var owner = device_route_held(model, mode, default_mode, context)
    if owner is None:
        var b = native_loaded_module("builtins")
        var device = device_request(fields, b)
        owner = device_route_keep(model, _xseq_route_build(device, mode, default_mode, context, b))
    var native = owner.downcast_value_ptr[DeviceRouteOwner]()
    var entries = native[].images
    var output = native[].scope
    var metrics = native[].table
    var word = native[].cls
    return XseqRoute(entries, output, metrics, native[].route.mode, native[].route.flags == 1, owner,
                     native[].route.entry, word)


def xseq_route(model: PythonObject, fields: PythonObject, context: PythonObject) raises -> XseqRoute:
    """The estimator's route for this operation, by its `numeric_mode`
    parameter (`xseq_route_of`)."""
    var nk = xseq_keys()
    return xseq_route_of(model, py_get(fields, nk[].k_numeric_mode), fields, context)


def xseq_neural_provider(entries: PythonObject) raises -> PythonObject:
    var nk = xseq_keys()
    # Immutable engine table only. Inputs/weights are admitted afresh by callers.
    var provider = neural_descriptor_reuse(py_get(entries, nk[].k_native_model_provider),
        py_item(entries, nk[].k_model_native_api_v1), String(MODEL_ABI_NAME), MODEL_OPERATION_COUNT)
    py_setitem(entries, nk[].k_native_model_provider, provider)
    return provider


def xseq_base(context: PythonObject, name: String) raises -> PythonObject:
    """A compiled entry of the base binding, connected at import."""
    var nk = xseq_keys()
    return py_item(context, nk[].k_xseq_base)[PythonObject(name)]


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


# TOMBSTONE: xseq_f32 (admission helper, merged into the one path) deleted 2026-11-08 by lane a4-admit-sequence-tsa-neural; code recoverable at f7d10d2b09.
# Restore: experiments/removed/A4_ADMISSION_HELPERS.patch. It is
# `admit_nd_f32` (bindings/admitted_sequence.mojo): the same borrow rule and
# cast bits, one parallel cast into uninitialised storage.


def xseq_refuse_f64(value: PythonObject, message: String) raises:
    """`if y.dtype == np.float64: raise TypeError(message)`."""
    if xseq_is_f64(value):
        raise Error("TypeError: " + message)


# ---------------------------------------------------------------- outputs

def xseq_zeros(shape: PythonObject, context: PythonObject) raises -> PythonObject:
    """`np.zeros(shape, dtype=np.float32)`."""
    var nk = xseq_keys()
    return sequential_array(shape, py_item(context, nk[].k_float32), context)


def xseq_zeros_i64(shape: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = xseq_keys()
    return sequential_array(shape, py_item(context, nk[].k_int64), context)


def xseq_addr(value: PythonObject) raises -> PythonObject:
    return sequential_address(value)


def xseq_f32p(value: PythonObject) raises -> XF32P:
    return XF32P(unsafe_from_address=Int(py=sequential_address(value)))


def xseq_reshape(value: PythonObject, shape: PythonObject) raises -> PythonObject:
    """`value.reshape(shape)`: a view of the same words (ndarray's own C
    method)."""
    var b = native_loaded_module("builtins")
    return py_call(py_call(b.getattr, value, "reshape"), shape)


def xseq_copy(value: PythonObject) raises -> PythonObject:
    """`value.copy()` (ndarray's own C method)."""
    var b = native_loaded_module("builtins")
    return py_call(py_call(b.getattr, value, "copy"))


def xseq_item(value: PythonObject, index: Int) raises -> PythonObject:
    """`value[index]` (ndarray's own C subscript): a NumPy scalar of a 1-D
    array, a row view of a 2-D one."""
    var b = native_loaded_module("builtins")
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
    # One PyList_New, then each completed float32 word boxed straight into
    # its slot (the bindings/py_lists.mojo pattern), instead of a getattr
    # of `append` and a call per epoch.
    ref cpy = Python().cpython()
    var raw = cpy.lib.get_function[PyObjectPtr]("PyList_New")(count)
    if not raw:
        raise cpy.unsafe_get_error()
    # Own the NULL-filled list at once: a failure releases every stored slot.
    var result = PythonObject(from_owned=raw)
    var box = cpy.lib.get_function[PyObjectPtr]("PyFloat_FromDouble")
    var set_item = cpy.lib.get_function[c_int]("PyList_SetItem")
    var p = xseq_f32p(value)
    for k in range(count):  # driver-loop: over the native training engine's per-epoch scalar report
        var item = box(Float64(p[k]))
        if not item:
            raise cpy.unsafe_get_error()
        # SetItem steals the new reference on both success and failure.
        if set_item(result._obj_ptr, k, item) != 0:
            raise cpy.unsafe_get_error()
    _ = value  # keep the source words alive through the reads above
    return result


def xseq_mean_dict(forecast: PythonObject) raises -> PythonObject:
    """`{"mean": forecast}`."""
    var result = Python.dict()
    var nk = xseq_keys()
    py_setitem(result, nk[].k_mean, forecast)
    return result


# ---------------------------------------------------------------- scalars

def xseq_int(value: PythonObject) raises -> Int:
    """`int(value)` (the builtin conversion, its own TypeError/ValueError)."""
    var b = native_loaded_module("builtins")
    return Int(py=py_call_typed(b.int, value))


def xseq_float(value: PythonObject) raises -> Float64:
    """`float(value)`."""
    var b = native_loaded_module("builtins")
    return Float64(py=py_call_typed(b.float, value))


def xseq_truth(value: PythonObject) raises -> Bool:
    """`bool(value)`."""
    var b = native_loaded_module("builtins")
    return Bool(py=py_call_typed(b.bool, value))


def xseq_is_int(value: PythonObject, context: PythonObject) raises -> Bool:
    """`isinstance(value, (int, np.integer))`."""
    var b = native_loaded_module("builtins")
    var nk = xseq_keys()
    return Bool(py=py_call(b.isinstance, value, py_item(context, nk[].k_int_types)))


# ---------------------------------------------------------------- the seeded stream

struct XseqStream(Movable):
    """`_buffer.InitStream`: a seeded stream of uniform float32 draws made by
    the base binding's counter-based `uniform_init_f32`; only the counter
    lives here."""
    var seed: UInt64
    var offset: Int

    def __init__(out self, random_state: PythonObject, context: PythonObject) raises:
        var b = native_loaded_module("builtins")
        ref cpy = Python().cpython()
        if random_state is None:
            # Native OS entropy entry, with byte ownership held through decoding.
            var raw = py_call(native_loaded_module("os").urandom, 8)
            if py_type(raw) is not b.bytes or len(raw) != 8:
                raise Error("RuntimeError: OS entropy must return eight bytes")
            var address = cpy.lib.get_function[Int]("PyBytes_AsString")(raw._obj_ptr)
            if address == 0:
                raise cpy.unsafe_get_error()
            var ptr = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=address)
            self.seed = 0
            for i in range(8):  # small-loop(one seed word from native entropy)
                self.seed |= UInt64(ptr[i]) << UInt64(8 * i)
            _ = raw
        else:
            # Admit exact built-in/NumPy scalars before the native int conversion;
            # user __int__/bitwise methods never participate in this operation.
            var cls = py_type(random_state)
            var nk = xseq_keys()
            if (cls is not b.int and cls is not b.float and cls is not b.bool
                    and cls is not b.str and cls is not b.bytes and cls is not b.bytearray
                    and not _one_of(cls, py_item(context, nk[].k_scalar_types))):
                raise Error("TypeError: random_state requires a native scalar, without conversion hooks")
            var word = py_call_typed(b.int, random_state)
            self.seed = cpy.lib.get_function[UInt64]("PyLong_AsUnsignedLongLongMask")(word._obj_ptr)
            if cpy.PyErr_Occurred():
                raise cpy.unsafe_get_error()
        self.offset = 0

    def fill_uniform(mut self, context: PythonObject, address: PythonObject, n: Int, low: Float64, high: Float64) raises:
        """n float32 values U(low, high) at `address`; advances the counter."""
        if n:
            # call-table: base helper (the base binding's `uniform_init_f32`; xcnn's caller)
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
        var child = xseq_stream_child_seed(self.seed, self.offset)
        self.offset += 1
        return child
