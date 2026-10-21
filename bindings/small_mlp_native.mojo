# SPDX-License-Identifier: Apache-2.0
"""Native public owner of SmallMLPTrainer: every public method is one call here.

The Python class keeps one integer handle (`_handle`) into the typed store in
`core/small_mlp_state_impl.mojo`. This door unpacks the caller's arguments,
selects the numeric mode and route (fused step, Apple FAST device-resident
session, or the per-operation sequence a binding without the fused entry and a
CPU-only install take), calls the compiled training entries with addresses into
the store and returns typed results. It adds no arithmetic: the forward,
backward, loss and AdamW update are the training binding's entries, called with
the same operands, in the same order, as `python/mojolearn/_mlp_impl.py` called
them before this owner existed, so every route keeps its bits.

Ownership (docs/MODULAR_ASKS.md: a PythonObject passed as a call argument or
assigned as an attribute keeps one extra reference). The trainer object never
enters this owner: the Python methods pass `self._handle`, state lives in the
native store, and the constructor assigns the returned handle itself. Caller
inputs are read through attribute and method access on the caller's object
(`__class__`, `__buffer__`, `keys()`, `__float__`, `__index__`), never passed
as a call argument, so they are not retained either (Python 3.12+; on 3.10
and 3.11, `memoryview(value)` is the only buffer door).
"""
from std.ffi import _Global
from std.math import isfinite
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class
from bindings.generation_buffers import GENERATION_BOUNDARY, generation_array, py_ints
from bindings.native_json import native_json_parse, native_json_parse_unique, native_path
from bindings.native_transport import _transport_json
from core.parallel_mlp_native_impl import mlp_targets_into
from core.small_mlp_state_impl import (
    MLP_MAX_STEP,
    MLP_STORE,
    MLP_TOTAL,
    MlpState,
    mlp_copy_f32,
    mlp_copy_tensor_out,
    mlp_copy_i32,
    mlp_offset,
    mlp_size,
    mlp_read_f32,
    mlp_require_finite,
    mlp_require_moments,
    mlp_require_result_finite,
    mlp_require_state,
    mlp_store_create,
    mlp_store_index,
    mlp_store_release,
)
from training.parallel_byte_impl import parallel_copy_words

# `mlp_train_step` / `mlp_resident_steps` modes (training/mlp_ops.mojo).
comptime _FORWARD = 0
comptime _GRADS = 1
comptime _TRAIN = 2
# The neural GEMM layouts (bindings/_mojolearn_training.mojo::neural_gemm and
# _mojolearn_linalg.mojo::gemm): 0 NN, 1 NT, 2 TN.
comptime _NN = 0
comptime _NT = 1
comptime _TN = 2
comptime _STATE_SCHEMA = "mojolearn.small-mlp-trainer.v1"
comptime _FILE_SCHEMA = "mojolearn.small-mlp-checkpoint.v1"
comptime _BASE_PROFILE = "mojolearn.neural-training.fp32.v1"
comptime _FILE_LIMIT = 32768
comptime _MODE_REFUSAL = "RuntimeError: SmallMLPTrainer requires process-selected IDENTICAL or FAST mode; select it before use"


struct SmallMlpBoundary(Defaultable, Movable):
    var bi: PythonObject
    var trainer_cls: PythonObject
    var array_cls: PythonObject
    var arrays: PythonObject
    var rlock: PythonObject
    var locks: PythonObject
    var modes: PythonObject
    var backend: PythonObject
    var environ: PythonObject
    var posix: PythonObject
    var buffer_flags: PythonObject

    def __init__(out self):
        self.bi = PythonObject(None)
        self.trainer_cls = PythonObject(None)
        self.array_cls = PythonObject(None)
        self.arrays = PythonObject(None)
        self.rlock = PythonObject(None)
        self.locks = PythonObject(None)
        self.modes = PythonObject(None)
        self.backend = PythonObject(None)
        self.environ = PythonObject(None)
        self.posix = PythonObject(None)
        self.buffer_flags = PythonObject(None)


comptime SMALL_MLP_BOUNDARY = _Global[
    StorageType=SmallMlpBoundary, name="MojoSmallMlpTrainerBoundary",
    init_fn=SmallMlpBoundary.__init__,
]


def _names() -> List[String]:
    return [String("weight1"), String("bias1"), String("weight2"), String("bias2")]


def _dims(index: Int) -> List[Int]:
    if index == 0:
        return [16, 8]
    if index == 1:
        return [16]
    if index == 2:
        return [3, 16]
    return [3]


# ------------------------------------------------------------ connection

def small_mlp_initialize(context: PythonObject) raises:
    var b = SMALL_MLP_BOUNDARY.get_or_create_ptr()
    b[].bi = Python.import_module("builtins")
    b[].trainer_cls = context["trainer"]
    b[].array_cls = context["array"]
    b[].arrays = Python.import_module("array").array
    b[].rlock = Python.import_module("_thread").RLock
    b[].locks = b[].bi.dict()
    b[].modes = context["modes"]
    b[].backend = context["backend"]
    b[].environ = context["environ"]
    b[].posix = Python.import_module("posix")
    # PyBUF_FULL_RO, the flags memoryview(value) asks with; one cached
    # object, so passing it never allocates.
    b[].buffer_flags = PythonObject(284)
    # The native Array allocator, shared without resetting any generation
    # model classes another owner already admitted.
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    g[].bi = b[].bi
    g[].arrays = b[].arrays
    g[].array_cls = b[].array_cls


def _mode_entries() raises -> PythonObject:
    """The process-selected numeric mode's training entries (identical or fast),
    read back from the compiled binding; the route choice of every call."""
    var b = SMALL_MLP_BOUNDARY.get_or_create_ptr()
    var mode = b[].backend["_DEFAULT_MODE"]
    if mode is None or b[].bi.type(mode) is not b[].bi.str or (String(py=mode) != "identical" and String(py=mode) != "fast"):
        raise Error(_MODE_REFUSAL)
    var entries = b[].modes.get(mode)
    if entries is None or b[].bi.type(entries) is b[].bi.str:
        raise Error(String(_MODE_REFUSAL) + " (" + String(entries) + ")")
    var compiled = Int(py=entries["training_numeric_mode"]())
    if compiled != (1 if String(py=mode) == "identical" else 0):
        raise Error(_MODE_REFUSAL)
    for name in [String("mlp_bias_activation"), String("mlp_relu_backward"), String("mlp_sum_rows")]:  # small-loop(the three required entry names)
        if not Bool(py=entries.__contains__(name)):
            raise Error("ImportError: SmallMLPTrainer requires updated training binding: missing " + name)
    return entries


def _mode_name(entries: PythonObject) raises -> String:
    return String(py=entries["mode"])


def _env(name: String, default: String) raises -> String:
    var b = SMALL_MLP_BOUNDARY.get_or_create_ptr()
    var raw = b[].environ.get(PythonObject(name).encode("ascii"), PythonObject(default).encode("ascii"))
    return String(py=raw.decode("latin-1"))


def _fused_enabled(entries: PythonObject) raises -> Bool:
    """ONE binding call per step (`mlp_train_step`): same kernels, operands and
    order as the per-operation sequence. MOJOLEARN_MLP_FUSED=0 restores the
    per-operation calls; a multi-device optimizer count keeps them too (that
    route lives in the per-call optimizer entry); a binding without the entry
    (a CPU-only install) takes them."""
    if _env("MOJOLEARN_MLP_FUSED", "1") == "0":
        return False
    if _env("MOJOLEARN_OPTIMIZER_DEVICE_COUNT", "1") != "1":
        return False
    return Bool(py=entries.__contains__("mlp_train_step"))


def _resident_enabled(entries: PythonObject) raises -> Bool:
    """The Apple FAST device-resident session (compiled under
    -D MOJOLEARN_AFN_MLP_RESIDENT, _MULTISTEP or _ALL only)."""
    return _fused_enabled(entries) and Bool(py=entries.__contains__("mlp_resident_open"))


# ------------------------------------------------------------ argument doors

def _is_bool(value: PythonObject) raises -> Bool:
    var bi = SMALL_MLP_BOUNDARY.get_or_create_ptr()[].bi
    var cls = value.__class__
    return cls is bi.bool or String(py=cls.__name__) == "bool_"


def _base_format(view: PythonObject) raises -> String:
    var fmt = String(py=view.format)
    if fmt.byte_length() > 1 and (fmt.startswith("@") or fmt.startswith("=") or fmt.startswith("<")):
        return String(fmt[byte=1:])
    return fmt


def _shape_text(dims: List[Int]) -> String:
    var text = String("(")
    for i in range(len(dims)):  # small-loop(shape words of a refusal message)
        if i > 0:
            text += ", "
        text += String(dims[i])
    if len(dims) == 1:
        text += ","
    return text + ")"


def _view(value: PythonObject, refusal: String) raises -> PythonObject:
    var b = SMALL_MLP_BOUNDARY.get_or_create_ptr()
    if value.__class__ is b[].array_cls:
        return value._mv
    var exporter = PythonObject(None)
    try:
        exporter = value.__buffer__
    except:
        exporter = PythonObject(None)
    if exporter is not None:
        try:
            return exporter(b[].buffer_flags)
        except:
            raise Error("TypeError: " + refusal)
    try:
        return b[].bi.memoryview(value)
    except:
        raise Error("TypeError: " + refusal)


def _shape(value: PythonObject, view: PythonObject) raises -> PythonObject:
    var b = SMALL_MLP_BOUNDARY.get_or_create_ptr()
    if value.__class__ is b[].array_cls:
        return value.shape
    return view.shape


def _words_into(value: PythonObject, dims: List[Int], label: String, dst: Int, integer: Bool = False) raises:
    """Copy a float32 (or int32) buffer of exactly `dims` into `dst`, row-major.
    The dtype is refused, never converted."""
    var b = SMALL_MLP_BOUNDARY.get_or_create_ptr()
    var kind = String("int32") if integer else String("float32")
    var refusal = "SmallMLPTrainer " + label + " must be a " + kind + " array (a NumPy array, an array.array or a mojolearn Array)"
    var view = _view(value, refusal)
    var fmt = _base_format(view)
    var itemsize = Int(py=view.itemsize)
    var ok: Bool
    if integer:
        ok = (fmt == "i" or fmt == "l") and itemsize == 4
    else:
        ok = fmt == "f" and itemsize == 4
    if not ok:
        raise Error("TypeError: " + refusal)
    var shape = _shape(value, view)
    var same = len(shape) == len(dims)
    var n = 1
    for i in range(len(dims)):  # small-loop(dims: array rank words)
        n *= dims[i]
        if same and Int(py=shape[i]) != dims[i]:
            same = False
    if not same:
        raise Error("SmallMLPTrainer " + label + " must have shape " + _shape_text(dims))
    var store = b[].arrays("i" if integer else "f")
    store.frombytes(view.tobytes("C"))
    if len(store) != n:
        raise Error("SmallMLPTrainer " + label + " must have shape " + _shape_text(dims))
    var fortran = value.__class__ is b[].array_cls and String(py=value.order) == "F" and len(dims) > 1
    parallel_copy_words(Int(py=store.buffer_info()[0]), dst, dims, n, fortran)
    _ = store^


def _owned_f32(value: PythonObject, dims: List[Int], label: String) raises -> PythonObject:
    """An owned, row-major, finite float32 copy of `value` as a mojolearn Array."""
    var out = generation_array(dims)
    _words_into(value, dims, label, Int(py=out._addr))
    mlp_require_finite(Int(py=out._addr), Int(py=out.size), label)
    return out


def _batch_in(value: PythonObject) raises -> PythonObject:
    var b = SMALL_MLP_BOUNDARY.get_or_create_ptr()
    var view = _view(value, "SmallMLPTrainer X must be a float32 array of shape (batch, 8) (a NumPy array, an array.array or a mojolearn Array), got " + String(py=value.__class__.__name__))
    var shape = _shape(value, view)
    if len(shape) != 2 or Int(py=shape[1]) != 8:
        raise Error("SmallMLPTrainer X must have shape (batch, 8)")
    var rows = Int(py=shape[0])
    if rows < 1 or rows > 256:
        raise Error("SmallMLPTrainer batch must be in [1, 256]")
    return _owned_f32(value, [rows, 8], "X")


def _targets_in(value: PythonObject, rows: Int) raises -> PythonObject:
    var b = SMALL_MLP_BOUNDARY.get_or_create_ptr()
    var view = _view(value, "SmallMLPTrainer targets must be an integer array")
    var fmt = _base_format(view)
    if fmt != "b" and fmt != "B" and fmt != "h" and fmt != "H" and fmt != "i" and fmt != "I" and fmt != "l" and fmt != "L" and fmt != "q" and fmt != "Q":
        raise Error("TypeError: SmallMLPTrainer targets must be an integer array")
    var shape = _shape(value, view)
    if len(shape) != 1 or Int(py=shape[0]) != rows:
        raise Error("SmallMLPTrainer targets must have shape (batch,) with classes 0..2")
    var packed = b[].arrays(fmt)
    packed.frombytes(view.tobytes("C"))
    var output = generation_array([rows], "i")
    var ids = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=output._addr))
    var signed = fmt == "b" or fmt == "h" or fmt == "i" or fmt == "l" or fmt == "q"
    try:
        mlp_targets_into(Int(py=packed.buffer_info()[0]), Int(py=packed.itemsize), signed, rows, ids)
    except:
        raise Error("SmallMLPTrainer targets must have shape (batch,) with classes 0..2")
    _ = packed^
    return output


def _scalar(value: PythonObject, name: String) raises -> Float64:
    if _is_bool(value):
        raise Error("SmallMLPTrainer " + name + " must be a finite float32 scalar")
    var number: Float64
    try:
        number = Float64(py=value.__getattr__("__float__")())
    except:
        raise Error("SmallMLPTrainer " + name + " must be a scalar")
    if not isfinite(number) or abs(number) > Float64(3.4028234663852886e38):
        raise Error("SmallMLPTrainer " + name + " must be finite float32")
    # One round-to-nearest-even to binary32 and back.
    return Float64(Float32(number))


def _config_into(mut state: MlpState, lr: PythonObject, betas: PythonObject,
                 eps: PythonObject, weight_decay: PythonObject) raises:
    var first = PythonObject(None)
    var second = PythonObject(None)
    var extra = True
    try:
        var items = betas.__getattr__("__iter__")()
        first = items.__getattr__("__next__")()
        second = items.__getattr__("__next__")()
        try:
            _ = items.__getattr__("__next__")()
        except:
            extra = False
    except:
        raise Error("SmallMLPTrainer betas must contain two scalars")
    if extra:
        raise Error("SmallMLPTrainer betas must contain two scalars")
    state.lr = _scalar(lr, "lr")
    state.beta1 = _scalar(first, "beta1")
    state.beta2 = _scalar(second, "beta2")
    state.eps = _scalar(eps, "eps")
    state.weight_decay = _scalar(weight_decay, "weight_decay")
    if state.lr <= 0 or state.eps <= 0 or state.weight_decay < 0:
        raise Error("SmallMLPTrainer requires lr > 0, eps > 0 and weight_decay >= 0")
    if not (state.beta1 >= 0 and state.beta1 < 1) or not (state.beta2 >= 0 and state.beta2 < 1):
        raise Error("SmallMLPTrainer betas must remain in [0, 1) in float32")


def _schedule_walk(value: PythonObject, depth: Int, mut budget: Int) raises:
    var bi = SMALL_MLP_BOUNDARY.get_or_create_ptr()[].bi
    budget -= 1
    if depth > 8 or budget < 0:
        raise Error("SmallMLPTrainer data_schedule is too complex")
    var cls = value.__class__
    if cls is bi.dict:
        var count = len(value)
        var keys = List[PythonObject]()
        var items = value.__getattr__("__iter__")()
        for _ in range(min(count, 129)):  # small-loop(keys: schedule JSON keys, at most 128)
            var key = items.__getattr__("__next__")()
            if key.__class__ is not bi.str:
                raise Error("SmallMLPTrainer data_schedule keys must be strings")
            keys.append(key)
        if count > 128:
            raise Error("SmallMLPTrainer data_schedule object is too large")
        for i in range(len(keys)):  # small-loop(keys: schedule JSON keys, at most 128)
            if len(keys[i]) > 2048:
                raise Error("SmallMLPTrainer data_schedule object is too large")
        for i in range(len(keys)):  # small-loop(keys: schedule JSON members, at most 128)
            _schedule_walk(value[keys[i]], depth + 1, budget)
    elif cls is bi.list:
        if len(value) > 128:
            raise Error("SmallMLPTrainer data_schedule list is too large")
        for i in range(len(value)):  # small-loop(value: schedule JSON list, at most 128)
            _schedule_walk(value[i], depth + 1, budget)
    elif cls is bi.str:
        if len(value) > 2048:
            raise Error("SmallMLPTrainer data_schedule string is too large")
    elif cls is bi.int:
        if Int(py=value.bit_length()) > 4096:
            raise Error("SmallMLPTrainer data_schedule integer is too large")
    elif value is not None and cls is not bi.float and cls is not bi.bool:
        raise Error("SmallMLPTrainer data_schedule must contain JSON values")


def _schedule_text(value: PythonObject) raises -> String:
    """The schedule's canonical JSON (sorted keys, compact, ASCII), bounded."""
    var bi = SMALL_MLP_BOUNDARY.get_or_create_ptr()[].bi
    if not _is_dict(value) or len(value) == 0:
        raise Error("SmallMLPTrainer data_schedule must be a nonempty JSON object")
    var budget = 128
    _schedule_walk(value, 0, budget)
    var text: String
    try:
        text = _transport_json(value)
    except:
        raise Error("SmallMLPTrainer data_schedule must be finite JSON")
    if text.byte_length() > 2048:
        raise Error("SmallMLPTrainer data_schedule exceeds 2048 bytes")
    return text


def _is_dict(value: PythonObject) raises -> Bool:
    var bi = SMALL_MLP_BOUNDARY.get_or_create_ptr()[].bi
    return Bool(py=bi.issubclass(value.__class__, bi.dict))


def _same_keys(value: PythonObject, names: String) raises -> Bool:
    """`value` is a dict whose keys are exactly the `|`-separated `names`."""
    var bi = SMALL_MLP_BOUNDARY.get_or_create_ptr()[].bi
    return _is_dict(value) and Bool(py=value.keys() == bi.set(PythonObject(names).split("|")))


def _handle_of(handle: PythonObject) raises -> Int:
    """The trainer's `_handle`, passed by the Python method itself."""
    var bi = SMALL_MLP_BOUNDARY.get_or_create_ptr()[].bi
    if handle.__class__ is not bi.int:
        raise Error("RuntimeError: SmallMLPTrainer state is closed or was never created")
    return Int(py=handle)


# ------------------------------------------------------------ state in and out

def _state_from_dict(state: PythonObject, mode: String) raises -> MlpState:
    """`state_dict` output (or a decoded checkpoint) as a validated typed state."""
    var names = _names()
    if not _same_keys(state, "schema|architecture|numeric_mode|parameter_order|weights|optimizer|config|data_schedule"):
        raise Error("SmallMLPTrainer state has missing or unknown fields")
    var order = Python.list(names[0], names[1], names[2], names[3])
    if (Bool(py=state["schema"] != PythonObject(_STATE_SCHEMA)) or Bool(py=state["architecture"] != py_ints([8, 16, 3]))
            or Bool(py=state["numeric_mode"] != PythonObject(mode)) or Bool(py=state["parameter_order"] != order)):
        raise Error("SmallMLPTrainer state schema/architecture/mode/order mismatch")
    var weights = state["weights"]
    if not _same_keys(weights, "weight1|bias1|weight2|bias2"):
        raise Error("SmallMLPTrainer state requires all four named parameters")
    var result = MlpState()
    for i in range(4):  # driver-loop: over the fixed MLP parameter registry
        _words_into(weights[names[i]], _dims(i), names[i], result.params_addr(i))
        mlp_require_finite(result.params_addr(i), mlp_size(i), names[i])
    var cfg = state["config"]
    if not _same_keys(cfg, "lr|beta1|beta2|eps|weight_decay"):
        raise Error("SmallMLPTrainer state optimizer configuration mismatch")
    _config_into(result, cfg["lr"], Python.tuple(cfg["beta1"], cfg["beta2"]), cfg["eps"], cfg["weight_decay"])
    var opt = state["optimizer"]
    if not _same_keys(opt, "kind|step|m|v|flags") or Bool(py=opt["kind"] != PythonObject("AdamW")):
        raise Error("SmallMLPTrainer state optimizer mismatch")
    if _is_bool(opt["step"]):
        raise Error("SmallMLPTrainer step must be an integer")
    var step: Int
    try:
        step = Int(py=opt["step"].__getattr__("__index__")())
    except:
        raise Error("SmallMLPTrainer step must be an integer")
    if step < 0 or step > MLP_MAX_STEP:
        raise Error("SmallMLPTrainer step is outside its supported range")
    result.step = step
    _words_into(opt["m"], [MLP_TOTAL], "m", result.m_addr())
    mlp_require_finite(result.m_addr(), MLP_TOTAL, "m")
    _words_into(opt["v"], [MLP_TOTAL], "v", result.v_addr())
    mlp_require_finite(result.v_addr(), MLP_TOTAL, "v")
    _words_into(opt["flags"], [4], "flags", result.flags_addr(), True)
    mlp_require_moments(result)
    result.schedule = _schedule_text(state["data_schedule"])
    return result^


def _f32_array_of(addr: Int, dims: List[Int]) raises -> PythonObject:
    var out = generation_array(dims)
    mlp_copy_f32(addr, Int(py=out._addr), Int(py=out.size))
    return out


def _weights_out(state: MlpState) raises -> PythonObject:
    var bi = SMALL_MLP_BOUNDARY.get_or_create_ptr()[].bi
    var names = _names()
    var weights = bi.dict()
    for i in range(4):  # driver-loop: over the fixed MLP parameter registry
        var out = generation_array(_dims(i))
        mlp_copy_tensor_out(state, i, Int(py=out._addr))
        weights[names[i]] = out
    return weights


def _state_out(state: MlpState, mode: String) raises -> PythonObject:
    """An independent snapshot, in the order and types `state_dict` documents."""
    var bi = SMALL_MLP_BOUNDARY.get_or_create_ptr()[].bi
    var names = _names()
    var result = bi.dict()
    result["schema"] = PythonObject(_STATE_SCHEMA)
    result["architecture"] = py_ints([8, 16, 3])
    result["numeric_mode"] = PythonObject(mode)
    result["parameter_order"] = Python.list(names[0], names[1], names[2], names[3])
    result["weights"] = _weights_out(state)
    var opt = bi.dict()
    opt["kind"] = PythonObject("AdamW")
    opt["step"] = PythonObject(state.step)
    opt["m"] = _f32_array_of(state.m_addr(), [MLP_TOTAL])
    opt["v"] = _f32_array_of(state.v_addr(), [MLP_TOTAL])
    var flags = generation_array([4], "i")
    var fp = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=flags._addr))
    var sp = Pointer[Int32, ImmUntrackedOrigin](unsafe_from_address=state.flags_addr())
    for i in range(4):  # small-loop(the four optimizer flag words)
        fp[i] = sp[i]
    opt["flags"] = flags
    result["optimizer"] = opt
    var cfg = bi.dict()
    cfg["lr"] = PythonObject(state.lr)
    cfg["beta1"] = PythonObject(state.beta1)
    cfg["beta2"] = PythonObject(state.beta2)
    cfg["eps"] = PythonObject(state.eps)
    cfg["weight_decay"] = PythonObject(state.weight_decay)
    result["config"] = cfg
    result["data_schedule"] = native_json_parse(PythonObject(state.schedule))
    return result


# ------------------------------------------------------------ resident session

def _resident_addresses(index: Int) raises -> PythonObject:
    var store = MLP_STORE.get_or_create_ptr()
    return Python.list(
        store[].slots[index].state.params_addr(0), store[].slots[index].state.params_addr(1),
        store[].slots[index].state.params_addr(2), store[].slots[index].state.params_addr(3),
        store[].slots[index].state.m_addr(), store[].slots[index].state.v_addr())


def _resident_sync(index: Int) raises:
    """Bring an open session's parameters and moments into the store."""
    var store = MLP_STORE.get_or_create_ptr()
    if store[].slots[index].resident < 0:
        return
    var b = SMALL_MLP_BOUNDARY.get_or_create_ptr()
    var entries = b[].modes[store[].slots[index].resident_mode]
    _ = entries["mlp_resident_download"](store[].slots[index].resident, _resident_addresses(index))


def _resident_close(index: Int) raises:
    var store = MLP_STORE.get_or_create_ptr()
    var handle = store[].slots[index].resident
    if handle < 0:
        return
    var mode = store[].slots[index].resident_mode
    store[].slots[index].resident = -1
    store[].slots[index].resident_mode = String()
    store[].slots[index].resident_cap = 0
    var b = SMALL_MLP_BOUNDARY.get_or_create_ptr()
    _ = b[].modes[mode]["mlp_resident_close"](handle)


def _settle(index: Int, close: Bool) raises:
    """Before the host state is read (or replaced): sync an open session, and
    close it when the host copy becomes the truth again."""
    _resident_sync(index)
    if close:
        _resident_close(index)


def _resident_open(index: Int, entries: PythonObject, cap_k: Int) raises -> Int:
    var store = MLP_STORE.get_or_create_ptr()
    var mode = _mode_name(entries)
    if store[].slots[index].resident >= 0:
        if store[].slots[index].resident_mode == mode and store[].slots[index].resident_cap >= cap_k:
            return store[].slots[index].resident
        _settle(index, True)
    var handle = Int(py=entries["mlp_resident_open"](_resident_addresses(index), Python.list(cap_k)))
    store[].slots[index].resident = handle
    store[].slots[index].resident_mode = mode
    store[].slots[index].resident_cap = cap_k
    return handle


def _step_error(error: Error) -> Error:
    """The native step refuses a non-finite result on the device; that
    refusal surfaces as a RuntimeError naming the step."""
    var message = String(error)
    if message.find("nonfinite") >= 0 or message.find("not finite") >= 0:
        return Error("RuntimeError: SmallMLPTrainer step returned an invalid result: " + message)
    return Error("RuntimeError: " + message)


def _native_step(entry: PythonObject, a: PythonObject, b: PythonObject) raises -> Int:
    """One fused step call."""
    try:
        return Int(py=entry(a, b))
    except error:
        raise _step_error(error)


def _native_resident_step(entry: PythonObject, a: PythonObject, b: PythonObject, c: PythonObject) raises -> Int:
    """One resident-session step call."""
    try:
        return Int(py=entry(a, b, c))
    except error:
        raise _step_error(error)


def _resident_steps(index: Int, entries: PythonObject, x: PythonObject, y: PythonObject,
                    rows: Int, k: Int, mode: Int, want_dx: Bool) raises -> PythonObject:
    """`k` steps on the session; `[losses, logits, gradients, input_grad]`
    (the logits and gradients are the last step's)."""
    var store = MLP_STORE.get_or_create_ptr()
    var handle = _resident_open(index, entries, k)
    var losses = generation_array([k])
    var logits = generation_array([rows, 3])
    var g0 = generation_array(_dims(0))
    var g1 = generation_array(_dims(1))
    var g2 = generation_array(_dims(2))
    var g3 = generation_array(_dims(3))
    var dx = generation_array([rows, 8]) if want_dx else generation_array([1])
    var t = store[].slots[index].state.step + 1
    ref s = store[].slots[index].state
    # addresses and params in this exact order (mirrored word for word in
    # `bindings/_mojolearn_training.mojo::mlp_resident_steps_binding`).
    var addresses = Python.list(x._addr, y._addr, losses._addr, logits._addr, g0._addr, g1._addr,
                                g2._addr, g3._addr, dx._addr)
    var params = Python.list(rows, k, mode, t, s.lr, s.beta1, s.beta2, s.eps, s.weight_decay, 1 if want_dx else 0)
    var entry = entries["mlp_resident_steps"] if k > 1 else entries["mlp_resident_step"]
    var written = _native_resident_step(entry, PythonObject(handle), addresses, params)
    if written != rows * 3:
        raise Error("RuntimeError: SmallMLPTrainer step returned an invalid result")
    if mode == _FORWARD:
        return Python.list(PythonObject(None), logits, PythonObject(None), PythonObject(None))
    var values = losses._mv.tolist()
    if mode == _TRAIN:
        store[].slots[index].state.step = t + k - 1
    return Python.list(values, logits, Python.list(g0, g1, g2, g3), dx if want_dx else PythonObject(None))


# ------------------------------------------------------------ fused step

def _fused(mut state: MlpState, entries: PythonObject, x: PythonObject, y: PythonObject,
           rows: Int, mode: Int, want_dx: Bool) raises -> PythonObject:
    """The forward (mode 0), the forward and backward (mode 1) or the whole
    step (mode 2) as ONE binding call. Under mode 2 the parameters, moments,
    flags and step of `state` are updated in place. Returns
    `[loss, logits, gradients, input_grad]`."""
    var train = mode == _TRAIN
    var loss = generation_array([1])
    var logits = generation_array([rows, 3])
    var g0 = generation_array(_dims(0))
    var g1 = generation_array(_dims(1))
    var g2 = generation_array(_dims(2))
    var g3 = generation_array(_dims(3))
    var dx = generation_array([rows, 8]) if want_dx else generation_array([1])
    var info = generation_array([3])
    var scratch_m = generation_array([1])
    var scratch_v = generation_array([1])
    var scratch_flags = generation_array([4], "i")
    var m = state.m_addr() if train else Int(py=scratch_m._addr)
    var v = state.v_addr() if train else Int(py=scratch_v._addr)
    var flags = state.flags_addr() if train else Int(py=scratch_flags._addr)
    var t = state.step + 1 if train else 1
    # addresses and params in this exact order (mirrored word for word in
    # `bindings/_mojolearn_training.mojo::mlp_train_step_binding`):
    #     addresses: x, y, w1, b1, w2, b2, m, v, flags, loss, logits,
    #                dw1, db1, dw2, db2, dx, info
    #     params:    rows, mode, t, lr, beta1, beta2, eps, weight_decay,
    #                want_input_grad
    var addresses = Python.list(x._addr, y._addr, state.params_addr(0), state.params_addr(1),
                                state.params_addr(2), state.params_addr(3), m, v, flags,
                                loss._addr, logits._addr, g0._addr, g1._addr, g2._addr, g3._addr,
                                dx._addr, info._addr)
    var params = Python.list(rows, mode, t, state.lr, state.beta1, state.beta2, state.eps,
                             state.weight_decay, 1 if want_dx else 0)
    var written = _native_step(entries["mlp_train_step"], addresses, params)
    _ = info^
    _ = scratch_m^
    _ = scratch_v^
    _ = scratch_flags^
    if written != rows * 3:
        raise Error("RuntimeError: SmallMLPTrainer step returned an invalid result")
    if mode == _FORWARD:
        return Python.list(0.0, logits, PythonObject(None), PythonObject(None))
    var value = mlp_read_f32(Int(py=loss._addr))
    if not isfinite(value):
        raise Error("RuntimeError: SmallMLPTrainer loss is not finite")
    if train:
        state.step = t
    return Python.list(value, logits, Python.list(g0, g1, g2, g3), dx if want_dx else PythonObject(None))


# ------------------------------------------------------------ per-operation route

def _gemm(entries: PythonObject, a: Int, b: Int, m: Int, n: Int, k: Int, layout: Int) raises -> PythonObject:
    var out = generation_array([m, n])
    if _mode_name(entries) == "identical":
        if not Bool(py=entries.__contains__("neural_gemm")):
            raise Error("ImportError: SmallMLPTrainer requires updated training binding: missing neural_gemm")
        var written = Int(py=entries["neural_gemm"](a, b, out._addr, py_ints([m, n, k, layout])))
        if written != m * n:
            raise Error("RuntimeError: SmallMLPTrainer neural GEMM returned an incomplete result")
    else:
        if not Bool(py=entries.__contains__("linalg_gemm")):
            raise Error("ImportError: SmallMLPTrainer requires the linalg binding's gemm")
        # THE OUTPUT ADDRESS COMES FIRST in the linalg entry.
        _ = entries["linalg_gemm"](out._addr, a, b, py_ints([m, n, k, layout]))
    mlp_require_result_finite(Int(py=out._addr), m * n, "GEMM")
    return out


def _bias(entries: PythonObject, values: PythonObject, bias: Int, rows: Int, cols: Int, relu: Bool) raises -> PythonObject:
    var out = generation_array([rows, cols])
    var written = Int(py=entries["mlp_bias_activation"](values._addr, bias, out._addr, py_ints([rows, cols, Int(relu)])))
    if written != rows * cols:
        raise Error("RuntimeError: SmallMLPTrainer bias/activation returned an invalid result")
    mlp_require_result_finite(Int(py=out._addr), rows * cols, "bias/activation")
    return out


def _sum(entries: PythonObject, values: PythonObject, rows: Int, cols: Int) raises -> PythonObject:
    var out = generation_array([cols])
    var written = Int(py=entries["mlp_sum_rows"](values._addr, out._addr, py_ints([rows, cols])))
    if written != cols:
        raise Error("RuntimeError: SmallMLPTrainer bias gradient returned an invalid result")
    mlp_require_result_finite(Int(py=out._addr), cols, "bias gradient")
    return out


def _forward(state: MlpState, entries: PythonObject, x: PythonObject, rows: Int) raises -> PythonObject:
    var hidden = _bias(entries, _gemm(entries, Int(py=x._addr), state.params_addr(0), rows, 16, 8, _NT),
                       state.params_addr(1), rows, 16, True)
    var logits = _bias(entries, _gemm(entries, Int(py=hidden._addr), state.params_addr(2), rows, 3, 16, _NT),
                       state.params_addr(3), rows, 3, False)
    return Python.list(hidden, logits)


def _gradient(state: MlpState, entries: PythonObject, x: PythonObject, y: PythonObject,
              rows: Int, want_dx: Bool) raises -> PythonObject:
    """The per-operation forward and backward: `[loss, logits, gradients, input_grad]`."""
    var forward = _forward(state, entries, x, rows)
    var hidden = forward[0]
    var logits = forward[1]
    var loss = generation_array([1])
    var row_loss = generation_array([rows])
    var dlogits = generation_array([rows, 3])
    # ce_loss params: rows, vocab, ignore_index, reduction (2 mean),
    # num_items, return_grad, label_smoothing
    # (bindings/_mojolearn_training.mojo::ce_loss_binding).
    _ = entries["ce_loss"](loss._addr, row_loss._addr, dlogits._addr, logits._addr, y._addr,
                           Python.list(rows, 3, -100, 2, 0, 1, 0.0))
    var value = mlp_read_f32(Int(py=loss._addr))
    if not isfinite(value):
        raise Error("RuntimeError: SmallMLPTrainer loss is not finite")
    mlp_require_finite(Int(py=dlogits._addr), rows * 3, "logit gradient")
    var dw2 = _gemm(entries, Int(py=dlogits._addr), Int(py=hidden._addr), 3, 16, rows, _TN)
    var db2 = _sum(entries, dlogits, rows, 3)
    var incoming = _gemm(entries, Int(py=dlogits._addr), state.params_addr(2), rows, 16, 3, _NN)
    var dhidden = generation_array([rows, 16])
    var written = Int(py=entries["mlp_relu_backward"](hidden._addr, incoming._addr, dhidden._addr, py_ints([rows, 16])))
    if written != rows * 16:
        raise Error("RuntimeError: SmallMLPTrainer ReLU gradient returned an invalid result")
    mlp_require_result_finite(Int(py=dhidden._addr), rows * 16, "ReLU gradient")
    var dw1 = _gemm(entries, Int(py=dhidden._addr), Int(py=x._addr), 16, 8, rows, _TN)
    var db1 = _sum(entries, dhidden, rows, 16)
    var dx = PythonObject(None)
    if want_dx:
        dx = _gemm(entries, Int(py=dhidden._addr), state.params_addr(0), rows, 8, 16, _NN)
    _ = row_loss^
    _ = incoming^
    return Python.list(value, logits, Python.list(dw1, db1, dw2, db2), dx)


def _adamw(mut state: MlpState, entries: PythonObject, grads: List[Float32]) raises:
    """One AdamW step over the packed registry (`optimizer_step`, as
    `mojolearn.AdamW.step` calls it): the parameters, moments and flags of
    `state` are updated in place."""
    var offsets: List[Int32] = [0, 128, 144, 192, 195]
    var info = List[Float32](length=3, fill=Float32(0))
    var t = state.step + 1
    # params: n_tensors, kind (2 AdamW), t, nesterov, lr, beta1, beta2, eps,
    # weight_decay, momentum, dampening, max_norm (<= 0: no clip), maximize
    # (bindings/_mojolearn_training.mojo::optimizer_step_binding).
    var plan = Python.list(4, 2, t, 0, state.lr, state.beta1, state.beta2, state.eps,
                           state.weight_decay, 0.0, 0.0, 0.0, 0)
    _ = entries["optimizer_step"](state.params_addr(0), Int(grads.unsafe_ptr()), state.m_addr(),
                                  state.v_addr(), Int(offsets.unsafe_ptr()), state.flags_addr(),
                                  Int(info.unsafe_ptr()), plan)
    _ = offsets^
    _ = info^
    state.step = t


def _items(value: PythonObject, limit: Int, refusal: String) raises -> List[PythonObject]:
    """The caller's sequence as Mojo-held items (at most `limit` when it is
    positive), read through its own iterator."""
    var result = List[PythonObject]()
    var items = PythonObject(None)
    try:
        items = value.__getattr__("__iter__")()
    except:
        raise Error("TypeError: " + refusal)
    while limit < 0 or len(result) < limit:  # driver-loop: over the caller's sequence of tensors or session batches
        try:
            result.append(items.__getattr__("__next__")())
        except:
            break
    return result^


def _pack_gradients(gradients: List[PythonObject]) raises -> List[Float32]:
    var packed = List[Float32](length=MLP_TOTAL, fill=Float32(0))
    var names = _names()
    for i in range(4):  # driver-loop: over the fixed MLP gradient registry
        var dst = Int(packed.unsafe_ptr()) + 4 * mlp_offset(i)
        _words_into(gradients[i], _dims(i), names[i] + " gradient", dst)
        mlp_require_finite(dst, mlp_size(i), names[i] + " gradient")
    return packed^


def _gradients_dict(grads: PythonObject) raises -> PythonObject:
    var bi = SMALL_MLP_BOUNDARY.get_or_create_ptr()[].bi
    var names = _names()
    var result = bi.dict()
    for i in range(4):  # driver-loop: over the fixed MLP gradient registry
        result[names[i]] = grads[i]
    return result


# ------------------------------------------------------------ public methods

def small_mlp_create(args: PythonObject) raises -> PythonObject:
    """`__init__`: weight1, bias1, weight2, bias2, data_schedule, lr, betas,
    eps, weight_decay. Returns the handle the Python constructor stores."""
    var b = SMALL_MLP_BOUNDARY.get_or_create_ptr()
    var state = MlpState()
    var names = _names()
    for i in range(4):  # driver-loop: over the fixed MLP parameter registry
        _words_into(args[i], _dims(i), names[i], state.params_addr(i))
        mlp_require_finite(state.params_addr(i), mlp_size(i), names[i])
    _config_into(state, args[5], args[6], args[7], args[8])
    state.schedule = _schedule_text(args[4])
    _ = _mode_entries()
    var handle = mlp_store_create(state^)
    b[].locks[handle] = b[].rlock()
    return PythonObject(handle)


def small_mlp_from_state(cls: PythonObject, state: PythonObject, mode: String) raises -> PythonObject:
    """A new trainer instance of `cls` owning a validated copy of `state`."""
    var b = SMALL_MLP_BOUNDARY.get_or_create_ptr()
    var typed = _state_from_dict(state, mode)
    var handle = mlp_store_create(typed^)
    b[].locks[handle] = b[].rlock()
    # `object.__new__` takes only the class; the handle goes into the new
    # instance's own dictionary, so the instance is never a call argument.
    var instance = b[].bi.object.__new__(cls)
    instance.__dict__["_handle"] = PythonObject(handle)
    return instance


def small_mlp_close(handle: PythonObject) raises:
    """Release a trainer's state (its `__del__`); a missing or foreign handle is a no-op."""
    var b = SMALL_MLP_BOUNDARY.get_or_create_ptr()
    if handle.__class__ is not b[].bi.int:
        return
    var h = Int(py=handle)
    try:
        var index = mlp_store_index(h)
        _resident_close(index)
    except:
        pass
    _ = mlp_store_release(h)
    _ = b[].locks.pop(h, PythonObject(None))


def small_mlp_state_dict(handle: Int) raises -> PythonObject:
    var entries = _mode_entries()
    var index = mlp_store_index(handle)
    _settle(index, False)
    var store = MLP_STORE.get_or_create_ptr()
    return _state_out(store[].slots[index].state, _mode_name(entries))


def small_mlp_load_state(handle: Int, state: PythonObject, mode: String) raises:
    """Validate and copy `state`, then replace the live state in one move."""
    var typed = _state_from_dict(state, mode)
    var index = mlp_store_index(handle)
    _resident_close(index)
    var store = MLP_STORE.get_or_create_ptr()
    store[].slots[index].state = typed^


def _step_count(handle: Int) raises -> PythonObject:
    var store = MLP_STORE.get_or_create_ptr()
    return PythonObject(store[].slots[mlp_store_index(handle)].state.step)


def _weights(handle: Int) raises -> PythonObject:
    _ = _mode_entries()
    var index = mlp_store_index(handle)
    _settle(index, False)
    var store = MLP_STORE.get_or_create_ptr()
    return _weights_out(store[].slots[index].state)


def _train_step(handle: Int, X: PythonObject, targets: PythonObject, want: PythonObject) raises -> PythonObject:
    var b = SMALL_MLP_BOUNDARY.get_or_create_ptr()
    var x = _batch_in(X)
    var rows = Int(py=x.shape[0])
    var y = _targets_in(targets, rows)
    if want.__class__ is not b[].bi.bool:
        raise Error("TypeError: SmallMLPTrainer return_input_grad must be a bool")
    var want_dx = Bool(py=want)
    var entries = _mode_entries()
    var index = mlp_store_index(handle)
    var store = MLP_STORE.get_or_create_ptr()
    var out: PythonObject
    if _resident_enabled(entries):
        # The state stays on the device; the binding restores the pre-step
        # state on the device when the step fails.
        if store[].slots[index].state.step == MLP_MAX_STEP:
            raise Error("SmallMLPTrainer step counter is exhausted")
        var res = _resident_steps(index, entries, x, y, rows, 1, _TRAIN, want_dx)
        out = Python.list(res[0][0], res[1], res[2], res[3])
    else:
        _settle(index, True)
        var working = store[].slots[index].state.copy()
        mlp_require_state(working)
        if working.step == MLP_MAX_STEP:
            raise Error("SmallMLPTrainer step counter is exhausted")
        var before = working.step
        if _fused_enabled(entries):
            out = _fused(working, entries, x, y, rows, _TRAIN, want_dx)
        else:
            out = _gradient(working, entries, x, y, rows, want_dx)
            _adamw(working, entries, _pack_gradients(_items(out[2], 4, "gradients")))
        # Validation precedes the single publication of the new state.
        mlp_require_state(working)
        if working.step != before + 1:
            raise Error("RuntimeError: SmallMLPTrainer optimizer did not advance exactly one step")
        index = mlp_store_index(handle)
        store[].slots[index].state = working^
    var result = b[].bi.dict()
    result["step"] = PythonObject(store[].slots[index].state.step)
    result["loss"] = out[0]
    result["logits"] = out[1]
    result["gradients"] = _gradients_dict(out[2])
    if want_dx:
        result["input_grad"] = out[3]
    return result


def _train_steps(handle: Int, X: PythonObject, targets: PythonObject, k_value: PythonObject) raises -> PythonObject:
    var b = SMALL_MLP_BOUNDARY.get_or_create_ptr()
    if k_value.__class__ is not b[].bi.int or Int(py=k_value) < 1:
        raise Error("SmallMLPTrainer train_steps k must be a positive int")
    var k = Int(py=k_value)
    var entries = _mode_entries()
    if not (_resident_enabled(entries) and Bool(py=entries.__contains__("mlp_resident_steps"))):
        raise Error("RuntimeError: SmallMLPTrainer.train_steps needs the Apple FAST training "
                    + "binding built with -D MOJOLEARN_AFN_MLP_MULTISTEP")
    var view = _view(X, "SmallMLPTrainer train_steps X must have shape (k * batch, 8)")
    var shape = _shape(X, view)
    if len(shape) != 2 or Int(py=shape[1]) != 8 or Int(py=shape[0]) % k != 0:
        raise Error("SmallMLPTrainer train_steps X must have shape (k * batch, 8)")
    var total = Int(py=shape[0])
    var batch = total // k
    if batch < 1 or batch > 256:
        raise Error("SmallMLPTrainer batch must be in [1, 256]")
    var x = _owned_f32(X, [total, 8], "X")
    var y = _targets_in(targets, total)
    var index = mlp_store_index(handle)
    var store = MLP_STORE.get_or_create_ptr()
    if store[].slots[index].state.step + k > MLP_MAX_STEP:
        raise Error("SmallMLPTrainer step counter is exhausted")
    var res = _resident_steps(index, entries, x, y, batch, k, _TRAIN, False)
    var result = b[].bi.dict()
    result["step"] = PythonObject(store[].slots[index].state.step)
    result["losses"] = res[0]
    result["loss"] = res[0][k - 1]
    result["logits"] = res[1]
    result["gradients"] = _gradients_dict(res[2])
    return result


def _predict_logits(handle: Int, X: PythonObject) raises -> PythonObject:
    var x = _batch_in(X)
    var rows = Int(py=x.shape[0])
    var entries = _mode_entries()
    var index = mlp_store_index(handle)
    var zeros = generation_array([rows], "i")
    if _resident_enabled(entries):
        return _resident_steps(index, entries, x, zeros, rows, 1, _FORWARD, False)[1]
    _settle(index, False)
    var store = MLP_STORE.get_or_create_ptr()
    var state = store[].slots[index].state.copy()
    if _fused_enabled(entries):
        return _fused(state, entries, x, zeros, rows, _FORWARD, False)[1]
    var result = _forward(state, entries, x, rows)[1]
    _ = x^
    return result


def _loss_and_grads(handle: Int, X: PythonObject, targets: PythonObject) raises -> PythonObject:
    var x = _batch_in(X)
    var rows = Int(py=x.shape[0])
    var y = _targets_in(targets, rows)
    var entries = _mode_entries()
    var index = mlp_store_index(handle)
    var out: PythonObject
    if _resident_enabled(entries):
        var res = _resident_steps(index, entries, x, y, rows, 1, _GRADS, False)
        return Python.tuple(res[0][0], res[2])
    _settle(index, False)
    var store = MLP_STORE.get_or_create_ptr()
    var state = store[].slots[index].state.copy()
    if _fused_enabled(entries):
        out = _fused(state, entries, x, y, rows, _GRADS, False)
    else:
        out = _gradient(state, entries, x, y, rows, False)
    return Python.tuple(out[0], out[2])


def _apply_gradients(handle: Int, gradients: PythonObject) raises -> PythonObject:
    var listed = _items(gradients, 5, "SmallMLPTrainer requires exactly four gradients")
    if len(listed) != 4:
        raise Error("SmallMLPTrainer requires exactly four gradients")
    var grads = _pack_gradients(listed)
    var entries = _mode_entries()
    var index = mlp_store_index(handle)
    _settle(index, True)
    var store = MLP_STORE.get_or_create_ptr()
    var working = store[].slots[index].state.copy()
    mlp_require_state(working)
    if working.step == MLP_MAX_STEP:
        raise Error("SmallMLPTrainer step counter is exhausted")
    var before = working.step
    _adamw(working, entries, grads)
    mlp_require_state(working)
    if working.step != before + 1:
        raise Error("RuntimeError: SmallMLPTrainer optimizer did not advance exactly one step")
    index = mlp_store_index(handle)
    store[].slots[index].state = working^
    return PythonObject(before + 1)


def _predict_sessions(handle: Int, inputs: PythonObject) raises -> PythonObject:
    """Logits of several independent batches with the current model,
    concatenated in the order given (`mlp_sessions`)."""
    var b = SMALL_MLP_BOUNDARY.get_or_create_ptr()
    var bi = b[].bi
    var entries = _mode_entries()
    var index = mlp_store_index(handle)
    _settle(index, False)
    var listed = _items(inputs, -1, "mojolearn.mlp_inference_sessions: inputs must be a list of arrays")
    # Mojo-side owners keep every session copy alive across the call.
    var sessions = List[PythonObject]()
    var shapes = bi.list()
    var addresses = bi.list()
    for i in range(len(listed)):  # driver-loop: over the caller's session batches
        var value = listed[i]
        var view = _view(value, "mojolearn.mlp_inference_sessions: session input must be float32")
        if _base_format(view) != "f" or Int(py=view.itemsize) != 4:
            raise Error("TypeError: mojolearn.mlp_inference_sessions: session input must be float32")
        var shape = _shape(value, view)
        var dims = List[Int]()
        for axis in range(len(shape)):  # small-loop(session shape words)
            dims.append(Int(py=shape[axis]))
        var owned = generation_array(dims)
        if Int(py=owned.size) > 0:
            _words_into(value, dims, "session input", Int(py=owned._addr))
        sessions.append(owned)
        shapes.append(owned.shape)
        addresses.append(owned._addr)
    var total: Int
    try:
        total = Int(py=entries["mlp_sessions_rows"](shapes, 8))
    except error:
        raise Error(String(error))
    var out = generation_array([total, 3])
    var store = MLP_STORE.get_or_create_ptr()
    ref s = store[].slots[index].state
    var wrote = Int(py=entries["mlp_sessions"](
        Python.list(out._addr, s.params_addr(0), s.params_addr(1), s.params_addr(2), s.params_addr(3)),
        addresses, shapes, py_ints([8, 16, 3])))
    _ = sessions^
    if wrote != Int(py=out.size):
        raise Error("RuntimeError: MLP sessions returned an invalid output count")
    return out


# ------------------------------------------------------------ checkpoint

def _encode_array(value: PythonObject, integer: Bool) raises -> PythonObject:
    """The on-disk tensor descriptor: dtype, shape, and the C-order
    little-endian bytes as lowercase hex (every supported host is
    little-endian, so the stored bytes are the buffer's own)."""
    var bi = SMALL_MLP_BOUNDARY.get_or_create_ptr()[].bi
    var result = bi.dict()
    result["dtype"] = PythonObject("<i4" if integer else "<f4")
    result["shape"] = bi.list(value.shape)
    result["hex"] = value._mv.tobytes().hex()
    return result


def _decode_array(value: PythonObject, dims: List[Int], integer: Bool) raises -> PythonObject:
    var b = SMALL_MLP_BOUNDARY.get_or_create_ptr()
    var bi = b[].bi
    var shape = py_ints(dims)
    var dtype = PythonObject("<i4" if integer else "<f4")
    if (not Bool(py=bi.isinstance(value, bi.dict)) or Bool(py=bi.set(value) != bi.set(PythonObject("dtype|shape|hex").split("|")))
            or Bool(py=value["dtype"] != dtype) or Bool(py=value["shape"] != shape)
            or bi.type(value["hex"]) is not bi.str):
        raise Error("SmallMLPTrainer checkpoint tensor descriptor mismatch")
    var cells = 1
    for i in range(len(dims)):  # small-loop(descriptor shape words)
        cells *= dims[i]
    if len(value["hex"]) != cells * 8:
        raise Error("SmallMLPTrainer checkpoint tensor length mismatch")
    var raw = PythonObject(None)
    try:
        raw = bi.bytes.fromhex(value["hex"])
    except:
        raise Error("SmallMLPTrainer checkpoint tensor is not hexadecimal")
    if len(raw) != cells * 4:
        raise Error("SmallMLPTrainer checkpoint tensor byte count mismatch")
    var store = b[].arrays("i" if integer else "f")
    store.frombytes(raw)
    var out = generation_array(dims, "i" if integer else "f")
    if integer:
        mlp_copy_i32(Int(py=store.buffer_info()[0]), Int(py=out._addr), cells)
    else:
        mlp_copy_f32(Int(py=store.buffer_info()[0]), Int(py=out._addr), cells)
    _ = store^
    return out


def _decode_state(payload: PythonObject) raises -> PythonObject:
    var bi = SMALL_MLP_BOUNDARY.get_or_create_ptr()[].bi
    if not Bool(py=bi.isinstance(payload, bi.dict)):
        raise Error("SmallMLPTrainer checkpoint payload must be an object")
    var weights = payload.get("weights")
    var opt = payload.get("optimizer")
    if weights is None or not Bool(py=bi.isinstance(opt, bi.dict)) or not Bool(py=opt.__contains__("m")) or not Bool(py=opt.__contains__("v")) or not Bool(py=opt.__contains__("flags")):
        raise Error("SmallMLPTrainer checkpoint state is incomplete")
    var names = _names()
    var order = Python.list(names[0], names[1], names[2], names[3])
    if not Bool(py=bi.isinstance(weights, bi.dict)) or Bool(py=bi.set(weights) != bi.set(order)):
        raise Error("SmallMLPTrainer checkpoint parameter registry mismatch")
    var decoded = bi.dict()
    for i in range(4):  # driver-loop: over the fixed MLP parameter registry
        decoded[names[i]] = _decode_array(weights[names[i]], _dims(i), False)
    var result = bi.dict(payload)
    result["weights"] = decoded
    var optimizer = bi.dict(opt)
    optimizer["m"] = _decode_array(opt["m"], [MLP_TOTAL], False)
    optimizer["v"] = _decode_array(opt["v"], [MLP_TOTAL], False)
    optimizer["flags"] = _decode_array(opt["flags"], [4], True)
    result["optimizer"] = optimizer
    return result


def _canonical_bytes(value: PythonObject) raises -> PythonObject:
    return PythonObject(_transport_json(value)).encode("ascii")


def _save_checkpoint(handle: Int, path: PythonObject) raises:
    """Atomically write the full state to `path`: a canonical JSON envelope
    with the payload's SHA-256, written to a private temporary file beside
    the target, flushed to disk, then renamed over it."""
    var b = SMALL_MLP_BOUNDARY.get_or_create_ptr()
    var bi = b[].bi
    var posix = b[].posix
    var state = small_mlp_state_dict(handle)
    var entries = _mode_entries()
    var names = _names()
    var weights = state["weights"]
    var encoded_weights = bi.dict()
    for i in range(4):  # driver-loop: over the fixed MLP parameter registry
        encoded_weights[names[i]] = _encode_array(weights[names[i]], False)
    state["weights"] = encoded_weights
    var opt = state["optimizer"]
    opt["m"] = _encode_array(opt["m"], False)
    opt["v"] = _encode_array(opt["v"], False)
    opt["flags"] = _encode_array(opt["flags"], True)
    var envelope = bi.dict()
    envelope["schema"] = PythonObject(_FILE_SCHEMA)
    envelope["payload"] = state
    # The digest is the Mojo SHA-256 (core/sha256.mojo), the same hex as hashlib's.
    envelope["payload_sha256"] = bi.str(entries["sha256_hex"](_canonical_bytes(state)))
    var profile = bi.str(entries["neural_arithmetic_profile"]())
    if String(py=profile) != _BASE_PROFILE:
        envelope["arithmetic_profile"] = profile
    var data = PythonObject(_transport_json(envelope) + "\n").encode("ascii")
    if len(data) > _FILE_LIMIT:
        raise Error("SmallMLPTrainer checkpoint exceeds its size bound")
    var target = native_path(path)
    var parts = target.rpartition("/")
    var directory = parts[0] if len(parts[1]) > 0 else PythonObject(".")
    if len(parts[1]) > 0 and len(parts[0]) == 0:
        directory = PythonObject("/")
    var temporary = directory + PythonObject("/.") + parts[2] + PythonObject(".") + posix.urandom(8).hex()
    var fd = -1
    try:
        fd = Int(py=posix.open(temporary, posix.O_WRONLY | posix.O_CREAT | posix.O_EXCL | posix.O_CLOEXEC, 384))
    except error:
        raise Error(_os_error(String(error)))
    var done = False
    try:
        var offset = 0
        var view = bi.memoryview(data)
        while offset < len(data):  # driver-loop: over partial OS writes of the checkpoint bytes
            offset += Int(py=posix.write(fd, view.__getitem__(bi.slice(offset, PythonObject(None)))))
        _ = posix.fsync(fd)
        _ = posix.close(fd)
        fd = -1
        _ = posix.replace(temporary, target)
        done = True
    except error:
        if fd >= 0:
            try:
                _ = posix.close(fd)
            except:
                pass
        try:
            _ = posix.unlink(temporary)
        except:
            pass
        raise Error(_os_error(String(error)))
    _ = done


def _os_error(message: String) -> String:
    if message.find("No such file") >= 0:
        return "FileNotFoundError: " + message
    if message.startswith("RuntimeError:") or message.startswith("ValueError:") or message.startswith("TypeError:"):
        return message
    return "OSError: " + message


def _from_checkpoint(cls: PythonObject, path: PythonObject) raises -> PythonObject:
    var b = SMALL_MLP_BOUNDARY.get_or_create_ptr()
    var bi = b[].bi
    var posix = b[].posix
    var source = native_path(path)
    var fd: Int
    try:
        fd = Int(py=posix.open(source, posix.O_RDONLY | posix.O_CLOEXEC))
    except error:
        raise Error(_os_error(String(error)))
    var chunks = bi.bytearray()
    try:
        while len(chunks) <= _FILE_LIMIT:  # driver-loop: over partial OS reads, at most the size bound plus one byte
            var chunk = posix.read(fd, _FILE_LIMIT + 1 - len(chunks))
            if len(chunk) == 0:
                break
            _ = chunks.extend(chunk)
    finally:
        _ = posix.close(fd)
    if len(chunks) > _FILE_LIMIT:
        raise Error("SmallMLPTrainer checkpoint exceeds its size bound")
    var envelope: PythonObject
    try:
        envelope = native_json_parse_unique(bi.bytes(chunks))
    except:
        raise Error("SmallMLPTrainer checkpoint is not valid bounded JSON")
    var three = bi.set(PythonObject("schema|payload|payload_sha256").split("|"))
    var four = bi.set(PythonObject("schema|payload|payload_sha256|arithmetic_profile").split("|"))
    if not Bool(py=bi.isinstance(envelope, bi.dict)) or (Bool(py=bi.set(envelope) != three) and Bool(py=bi.set(envelope) != four)):
        raise Error("SmallMLPTrainer checkpoint envelope mismatch")
    if Bool(py=envelope["schema"] != PythonObject(_FILE_SCHEMA)):
        raise Error("SmallMLPTrainer checkpoint schema mismatch")
    var entries = _mode_entries()
    var digest: PythonObject
    try:
        digest = bi.str(entries["sha256_hex"](_canonical_bytes(envelope["payload"])))
    except:
        raise Error("SmallMLPTrainer checkpoint integrity mismatch")
    if Bool(py=digest != envelope["payload_sha256"]):
        raise Error("SmallMLPTrainer checkpoint integrity mismatch")
    var expected = bi.str(entries["neural_arithmetic_profile"]())
    if Bool(py=envelope.get("arithmetic_profile", PythonObject(_BASE_PROFILE)) != expected):
        raise Error("SmallMLPTrainer checkpoint arithmetic profile mismatch")
    var state = _decode_state(envelope["payload"])
    return small_mlp_from_state(cls, state, _mode_name(entries))


# ------------------------------------------------------------ dispatch

def _locked(operation: String, handle_value: PythonObject, args: PythonObject) raises -> PythonObject:
    var b = SMALL_MLP_BOUNDARY.get_or_create_ptr()
    var handle = _handle_of(handle_value)
    _ = mlp_store_index(handle)
    var lock = b[].locks[handle]
    _ = lock.acquire()
    var result = PythonObject(None)
    try:
        if operation == "train_step":
            result = _train_step(handle, args[1], args[2], args[3])
        elif operation == "train_steps":
            result = _train_steps(handle, args[1], args[2], args[3])
        elif operation == "predict_logits":
            result = _predict_logits(handle, args[1])
        elif operation == "predict_logits_sessions":
            result = _predict_sessions(handle, args[1])
        elif operation == "loss_and_grads":
            result = _loss_and_grads(handle, args[1], args[2])
        elif operation == "apply_gradients":
            result = _apply_gradients(handle, args[1])
        elif operation == "state_dict":
            result = small_mlp_state_dict(handle)
        elif operation == "load_state_dict":
            small_mlp_load_state(handle, args[1], _mode_name(_mode_entries()))
        elif operation == "step":
            result = _step_count(handle)
        elif operation == "weights":
            result = _weights(handle)
        elif operation == "save_checkpoint":
            _save_checkpoint(handle, args[1])
        else:
            raise Error("SmallMLPTrainer: unknown operation " + operation)
    finally:
        _ = lock.release()
    return result


def small_mlp_call(operation: String, args: PythonObject) raises -> PythonObject:
    if operation == "initialize":
        small_mlp_initialize(args[0])
        return PythonObject(None)
    if operation == "create":
        return small_mlp_create(args)
    if operation == "close":
        small_mlp_close(args[0])
        return PythonObject(None)
    if operation == "from_checkpoint":
        return _from_checkpoint(args[0], args[1])
    return _locked(operation, args[0], args)


def small_mlp_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    var args = PythonObject(from_borrowed=args_ptr)
    try:
        var result = small_mlp_call(String(args[0]), args[1])
        return result.steal_data()
    except error:
        var message = String(error)
        if message.startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        if message.startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        if message.startswith("ImportError:"):
            return raise_python_exception(error_without_class(error, "ImportError"), ExceptionType("PyExc_ImportError"))
        if message.startswith("FileNotFoundError:"):
            return raise_python_exception(error_without_class(error, "FileNotFoundError"), ExceptionType("PyExc_FileNotFoundError"))
        if message.startswith("OSError:"):
            return raise_python_exception(error_without_class(error, "OSError"), ExceptionType("PyExc_OSError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
