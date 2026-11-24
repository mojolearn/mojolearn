# SPDX-License-Identifier: Apache-2.0
"""Native public owner of SmallMLPTrainer: every public method is one call here.

The Python class keeps one integer handle (`_handle`) into the typed store in
`core/small_mlp_state_impl.mojo`. This door unpacks the caller's arguments,
takes the trainer's route and runs the trainer's ONE path: its captured
program on the executor (autograd/small_mlp_program.mojo, the x_sequence
image's published `small_mlp_program` entry). The program opens from the host
state on first use and then holds the weights and moments as its session
parameters; the host copy comes down only when it is read (state_dict,
weights, save, inference sessions) and the program closes when the host copy
becomes the truth again (load_state_dict, close). It adds no arithmetic.

UNPACK and PACK are this file (buffer admission, typed copies, the result
objects); RUN is bindings/small_mlp_native_run.mojo (the program calls over
the typed store: no Python object). The trainer's device
(`device=None | "gpu" | "cpu"`) and the process tier are resolved ONCE into
the trainer's `DeviceRouteOwner` (`_route_build`): the program entry of the
image that pair selects and the training entries of the same device. No
operation reads or compares the device.

The public class hands the owner and arguments to this door once. Native
admission checks passive class storage and reads its handle without invoking
Python descriptors. Shared exact input admission precedes native buffer reads;
balanced call helpers retain original owners without leaking references.
"""
from std.math import isfinite
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class
from bindings.generation_buffers_door import GENERATION_BOUNDARY, generation_array, py_ints
from bindings.py_balanced import py_call, py_contains, py_get, py_setitem, py_type
from bindings.py_typed_call import py_call_typed
from bindings.mode_metadata_native_door import _fields, _class_fields
from bindings.native_loaded_module_door import NativeModules, native_interpreter_finalizing
from bindings.native_class_admission_door import class_record, class_admit
from bindings.native_buffer_owner_pack import native_storage_allocate, native_buffer_view
from providers.python.lowbit_weight_pyapi_door import low_materialize, low_formats
from providers.python.neural_descriptor_pyapi_door import neural_descriptor_entry
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.sequential_admission_door import sequential_admit_value
from bindings.small_mlp_items_door import small_mlp_items, SMALL_MLP_ITEMS_PENDING
from bindings.byte_checkpoint_io_unpack import ByteCheckpointIOError
from providers.python.training_metadata_pyapi_unpack import training_f32_scalar, training_schedule_json
from bindings.native_json_door import native_json_parse, native_json_parse_unique, native_path
from bindings.native_transport_door import _transport_json
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
    mlp_store_create,
    mlp_store_index,
    mlp_store_release,
)
from training.parallel_byte_impl import parallel_copy_words
from core.shared_slot import SharedSlot
from core.entry_table import EntryRef
from core.neural_entry_table import NeuralEntryKey, neural_entry_vendor, neural_ref, NEURAL_TABLE_MLP_PROGRAM
from core.device_route import DEVICE_INSTALL, device_mode_code, device_mode_word, device_route_plan, device_route_admit
from providers.python.device_route_boundary_door import (
    DeviceRouteOwner, device_admit, device_request, device_unpack, device_route_clear, device_route_held,
    device_route_keep, device_cpu_connect_late,
)
from bindings.small_mlp_native_run import (
    SMALL_MLP_RUN_FORWARD,
    SMALL_MLP_RUN_GRADS,
    SMALL_MLP_RUN_TRAIN,
    small_mlp_run_apply,
    small_mlp_run_close,
    small_mlp_run_settle,
    small_mlp_run_steps,
)
from bindings.sha256_door import sha256_hexdigest
from bindings.py_keys import py_op_code
from bindings.small_mlp_keys_unpack import (
    small_mlp_keys,
    SMALL_MLP_OPN_APPLY_GRADIENTS,
    SMALL_MLP_OPN_CLOSE,
    SMALL_MLP_OPN_CREATE,
    SMALL_MLP_OPN_FROM_CHECKPOINT,
    SMALL_MLP_OPN_INFERENCE_CHECKPOINT,
    SMALL_MLP_OPN_INFERENCE_CONNECT,
    SMALL_MLP_OPN_INFERENCE_INITIALIZE,
    SMALL_MLP_OPN_INFERENCE_PREDICT,
    SMALL_MLP_OPN_INFERENCE_REPR,
    SMALL_MLP_OPN_INFERENCE_WEIGHTS,
    SMALL_MLP_OPN_INITIALIZE,
    SMALL_MLP_OPN_LOAD_STATE_DICT,
    SMALL_MLP_OPN_LOSS_AND_GRADS,
    SMALL_MLP_OPN_OWNER_CLOSE,
    SMALL_MLP_OPN_OWNER_CREATE,
    SMALL_MLP_OPN_OWNER_LOAD_STATE_DICT,
    SMALL_MLP_OPN_OWNER_SAVE_CHECKPOINT,
    SMALL_MLP_OPN_OWNER_TRAIN_STEP,
    SMALL_MLP_OPN_PREDICT_LOGITS,
    SMALL_MLP_OPN_PREDICT_LOGITS_SESSIONS,
    SMALL_MLP_OPN_SAVE_CHECKPOINT,
    SMALL_MLP_OPN_STATE_DICT,
    SMALL_MLP_OPN_STEP,
    SMALL_MLP_OPN_TRAIN_STEP,
    SMALL_MLP_OPN_TRAIN_STEPS,
    SMALL_MLP_OPN_WEIGHTS,
)
from bindings.py_balanced import py_item

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
    # `device="cpu"` beside a GPU set: the mode -> entries (or connection
    # error) table of the CPU identical training image, filled in place at
    # the first device="cpu" admission (fx-host-import; it holds
    # `device_cpu_connect` until then); None on a CPU-only installation (its
    # one table is the CPU's).
    var modes_cpu: PythonObject
    var cpu_only: Bool
    var backend: PythonObject
    var environ: PythonObject
    var posix: PythonObject
    var buffer_flags: PythonObject
    var admission: PythonObject
    var inference: PythonObject
    var trainer_finalizer: PythonObject

    def __init__(out self):
        self.bi = PythonObject(None)
        self.trainer_cls = PythonObject(None)
        self.array_cls = PythonObject(None)
        self.arrays = PythonObject(None)
        self.rlock = PythonObject(None)
        self.locks = PythonObject(None)
        self.modes = PythonObject(None)
        self.modes_cpu = PythonObject(None)
        self.cpu_only = False
        self.backend = PythonObject(None)
        self.environ = PythonObject(None)
        self.posix = PythonObject(None)
        self.buffer_flags = PythonObject(None)
        self.admission = PythonObject(None)
        self.inference = PythonObject(None)
        self.trainer_finalizer = PythonObject(None)


comptime SMALL_MLP_BOUNDARY = SharedSlot[SmallMlpBoundary, "MojolearnSmallMlpTrainerBoundaryV1"]


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
    var nk = small_mlp_keys()
    var b = SMALL_MLP_BOUNDARY.get_or_create_ptr()
    b[].bi = Python.import_module("builtins")
    b[].trainer_cls = py_item(context, nk[].k_trainer)
    b[].trainer_finalizer = py_item(context, nk[].k_trainer_finalizer)
    b[].array_cls = py_item(context, nk[].k_array)
    b[].arrays = Python.import_module("array").array
    b[].rlock = Python.import_module("_thread").RLock
    b[].locks = b[].bi.dict()
    b[].modes = py_item(context, nk[].k_modes)
    b[].modes_cpu = py_get(context, "modes_cpu")
    b[].cpu_only = Bool(py=py_item(context, PythonObject("cpu_only")))
    b[].backend = py_item(context, nk[].k_backend)
    b[].environ = py_item(context, nk[].k_environ)
    b[].admission = py_item(context, nk[].k_admission)
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


# ------------------------------------------------------------ the device route
#
# A trainer's device and the process tier are resolved ONCE into the one
# `DeviceRouteOwner` (providers/python/device_route_boundary_door.mojo) with
# the rule and words of core/device_route.mojo: `route.entry` is the program
# entry of the image they select (core/neural_entry_table.mojo family
# NEURAL_TABLE_MLP_PROGRAM), `images` the training entries of the same device
# (`mlp_sessions`, the arithmetic profile). A hit is the process mode read,
# one record read and three pointer compares. Operations that hold no trainer
# (a bare state handle: the parallel owner's worker slots) take the route
# record of the connected trainer class: this installation's device.
#
# TOMBSTONE: `_mode_entries`, `MlpOperationMetadata`, `SmallMlpModeVerified` /
# `SMALL_MLP_MODE_VERIFIED` and `_program_ref` (the per-operation mode word
# built and compared, the entries table read by that word, the process-wide
# vendor word per mode and the program entry read from the neural table at
# every program call) deleted 2026-11-21 by lane dw-forecast-tail; the
# trainer's route owner holds the resolved program entry. Code recoverable at
# 9ec4ba3940.

def _route_build(requested: Int, mode: PythonObject) raises -> DeviceRouteOwner:
    """Resolve a trainer's images once (construction, a restore, the first
    operation after the process mode object changed): the tier, the device,
    the training entries of that pair checked against the selected tier, and
    the program entry of the same image set. A metadata step with no
    independent units to split."""
    var nk = small_mlp_keys()
    var b = SMALL_MLP_BOUNDARY.get_or_create_ptr()
    if mode is None or py_type(mode) is not b[].bi.str:
        raise Error(_MODE_REFUSAL)
    var mode_code = device_mode_code(String(py=mode))
    if mode_code < 0:
        raise Error(_MODE_REFUSAL)
    var plan = device_route_plan(requested, 0 if b[].cpu_only else 1, mode_code)
    var entries = PythonObject(None)
    if plan.cpu_table:
        # A GPU installation running `device="cpu"`: the CPU identical images
        # loaded beside the device images at the first device="cpu"
        # admission (python/mojolearn/_mlp_impl.py, no longer at import); a
        # trainer restored without construction connects them here, once.
        if b[].modes_cpu is None:
            raise Error("ImportError: device='cpu' is not connected in this installation")
        entries = py_get(b[].modes_cpu, mode)
        if entries is None and device_cpu_connect_late(b[].modes_cpu):
            entries = py_get(b[].modes_cpu, mode)
        if entries is None:
            raise Error("ImportError: device='cpu' is not connected in this installation")
        if py_type(entries) is b[].bi.str:
            raise Error(String(py=entries))
    else:
        entries = py_get(b[].modes, mode)
        if entries is None or py_type(entries) is b[].bi.str:
            raise Error(String(_MODE_REFUSAL) + " (" + String(entries) + ")")
    var compiled = Int(py=py_item(entries, nk[].k_training_numeric_mode)())
    if compiled != mode_code:
        raise Error(_MODE_REFUSAL)
    if not py_contains(entries, nk[].k_small_mlp_program):
        raise Error("ImportError: SmallMLPTrainer requires updated x_sequence binding: missing small_mlp_program")
    if not py_contains(entries, nk[].k_training_vendor):
        raise Error("ImportError: SmallMLPTrainer requires updated training binding: missing training_vendor")
    # The neural entry table key's vendor word (core/neural_entry_table.mojo):
    # the training and x_sequence images of one mode and device come from the
    # same selection, so they share the vendor.
    var vendor = neural_entry_vendor(String(py=py_item(entries, nk[].k_training_vendor)()))
    if vendor < 0:
        raise Error(_MODE_REFUSAL)
    if plan.cpu_table:
        device_route_admit("SmallMLPTrainer", plan, compiled, vendor, mode_code)
    var owner = DeviceRouteOwner(requested)
    owner.route.device = plan.device
    owner.route.mode = mode_code
    owner.route.vendor = vendor
    owner.route.flags = 1 if plan.cpu_table else 0
    # The program entry of that image; the table's refusals raise here.
    owner.route.entry = neural_ref[NEURAL_TABLE_MLP_PROGRAM](NeuralEntryKey(mode_code, vendor))
    owner.mode = mode
    owner.default_mode = mode
    owner.context = b[].modes
    owner.images = entries
    owner.resolved = True
    return owner^


def _owner_of(trainer: PythonObject) raises -> PythonObject:
    """The route owner of this operation: the trainer's, or (trainer None: an
    operation on a bare state handle) the connected class's, which is this
    installation's device. The device is read only on a miss."""
    var nk = small_mlp_keys()
    var b = SMALL_MLP_BOUNDARY.get_or_create_ptr()
    var mode = py_item(b[].backend, nk[].k__DEFAULT_MODE)
    var target = trainer
    if trainer is None:
        target = b[].trainer_cls
    var owner = device_route_held(target, mode, mode, b[].modes)
    if owner is None:
        var requested = DEVICE_INSTALL
        if trainer is not None:
            requested = device_request(_fields(trainer), b[].bi)
        owner = device_route_keep(target, _route_build(requested, mode))
    return owner


def _program_of(trainer: PythonObject) raises -> EntryRef:
    """The resolved program entry of this operation's route (two words, by
    value). The owner stays in the trainer's (or the class's) records."""
    var owner = _owner_of(trainer)
    var program = owner.downcast_value_ptr[DeviceRouteOwner]()[].route.entry
    _ = owner
    return program


def _mode_word(owner: PythonObject) raises -> String:
    """The numeric-mode word a state of this route carries."""
    return device_mode_word(owner.downcast_value_ptr[DeviceRouteOwner]()[].route.mode)


def _training_entries(owner: PythonObject) raises -> PythonObject:
    """The training entries of the route's device (`mlp_sessions`, the
    arithmetic profile)."""
    var entries = owner.downcast_value_ptr[DeviceRouteOwner]()[].images
    _ = owner
    return entries


# TOMBSTONE: MOJOLEARN_MLP_FUSED=0 per-operation route, the fused one-call step and the Apple FAST resident session doors (superseded by the captured program, plan row X2) deleted 2026-11-09 by lane/x2-small-mlp-on-executor; code recoverable at f59d913b0b.
# Restore: git apply experiments/removed/MOJOLEARN_X2_SMALL_MLP_HAND_ROUTES.patch


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
    try:
        return native_buffer_view(value)
    except:
        raise Error("TypeError: " + refusal)


def _shape(value: PythonObject, view: PythonObject) raises -> PythonObject:
    var b = SMALL_MLP_BOUNDARY.get_or_create_ptr()
    if value.__class__ is b[].array_cls:
        return value.shape
    return view.shape


def _source_address(value: PythonObject, view: PythonObject) raises -> Int:
    """The address of a C-contiguous caller buffer read without copying it
    through a temporary: a row-major mojolearn Array, a NumPy array (its array
    interface) or an array.array. 0 when only a byte copy can read it."""
    var nk = small_mlp_keys()
    var b = SMALL_MLP_BOUNDARY.get_or_create_ptr()
    var cls = value.__class__
    if cls is b[].array_cls:
        return Int(py=value._addr) if String(py=value.order) == "C" else 0
    if not Bool(py=view.c_contiguous):
        return 0
    if cls is b[].arrays:
        return Int(py=value.buffer_info()[0])
    try:
        var interface = value.__array_interface__
        if py_item(interface, nk[].k_strides) is None:
            return Int(py=py_item(interface, nk[].k_data)[0])
    except:
        pass
    return 0


def _words_into(value: PythonObject, dims: List[Int], label: String, dst: Int, integer: Bool = False) raises:
    """Copy a float32 (or int32) buffer of exactly `dims` into `dst`, row-major.
    The dtype is refused, never converted."""
    var b = SMALL_MLP_BOUNDARY.get_or_create_ptr()
    var kind = String("int32") if integer else String("float32")
    var refusal = "SmallMLPTrainer " + label + " must be a " + kind + " array (a NumPy array, an array.array or a mojolearn Array)"
    var view = _view(value, refusal)
    var shape = _shape(value, view)
    _words_from_view(value, view, shape, dims, label, dst, integer)


def _words_from_view(value: PythonObject, view: PythonObject, shape: PythonObject,
                     dims: List[Int], label: String, dst: Int, integer: Bool = False) raises:
    # The public operation retains the admitted buffer/shape through this copy.
    # Never acquire a second view or reread shape for an unchanged input.
    var b = SMALL_MLP_BOUNDARY.get_or_create_ptr()
    var kind = String("int32") if integer else String("float32")
    var refusal = "SmallMLPTrainer " + label + " must be a " + kind + " array (a NumPy array, an array.array or a mojolearn Array)"
    var fmt = _base_format(view)
    var itemsize = Int(py=view.itemsize)
    var ok: Bool
    if integer:
        ok = (fmt == "i" or fmt == "l") and itemsize == 4
    else:
        ok = fmt == "f" and itemsize == 4
    if not ok:
        raise Error("TypeError: " + refusal)
    var same = len(shape) == len(dims)
    var n = 1
    for i in range(len(dims)):  # small-loop(dims: array rank words)
        n *= dims[i]
        if same and Int(py=shape[i]) != dims[i]:
            same = False
    if not same or Int(py=view.nbytes) != 4 * n:
        raise Error("SmallMLPTrainer " + label + " must have shape " + _shape_text(dims))
    var source = _source_address(value, view)
    if source != 0:
        parallel_copy_words(source, dst, dims, n, False)
        _ = source  # keep the Python object alive through the native call above
        return
    var store = py_call(b[].arrays, "i" if integer else "f")
    _ = py_call(store.frombytes, py_call(view.tobytes, "C"))
    if len(store) != n:
        raise Error("SmallMLPTrainer " + label + " must have shape " + _shape_text(dims))
    var fortran = value.__class__ is b[].array_cls and String(py=value.order) == "F" and len(dims) > 1
    parallel_copy_words(Int(py=store.buffer_info()[0]), dst, dims, n, fortran)
    _ = store  # keep the Python object alive through the native call above
    _ = store^


def _batch_in(value: PythonObject) raises -> List[Float32]:
    """An owned, row-major, finite copy of the batch X (rows x 8)."""
    var view = _view(value, "SmallMLPTrainer X must be a float32 array of shape (batch, 8) (a NumPy array, an array.array or a mojolearn Array), got " + String(py=value.__class__.__name__))
    var shape = _shape(value, view)
    if len(shape) != 2 or Int(py=shape[1]) != 8:
        raise Error("SmallMLPTrainer X must have shape (batch, 8)")
    var rows = Int(py=shape[0])
    if rows < 1 or rows > 256:
        raise Error("SmallMLPTrainer batch must be in [1, 256]")
    var x = List[Float32](length=rows * 8, fill=Float32(0))
    _words_from_view(value, view, shape, [rows, 8], "X", Int(x.unsafe_ptr()))
    mlp_require_finite(Int(x.unsafe_ptr()), rows * 8, "X")
    return x^


def _targets_in(value: PythonObject, rows: Int) raises -> List[Int32]:
    var b = SMALL_MLP_BOUNDARY.get_or_create_ptr()
    var view = _view(value, "SmallMLPTrainer targets must be an integer array")
    var fmt = _base_format(view)
    if fmt != "b" and fmt != "B" and fmt != "h" and fmt != "H" and fmt != "i" and fmt != "I" and fmt != "l" and fmt != "L" and fmt != "q" and fmt != "Q":
        raise Error("TypeError: SmallMLPTrainer targets must be an integer array")
    var shape = _shape(value, view)
    if len(shape) != 1 or Int(py=shape[0]) != rows:
        raise Error("SmallMLPTrainer targets must have shape (batch,) with classes 0..2")
    var ids = List[Int32](length=rows, fill=Int32(0))
    var signed = fmt == "b" or fmt == "h" or fmt == "i" or fmt == "l" or fmt == "q"
    var itemsize = Int(py=view.itemsize)
    var source = _source_address(value, view)
    var packed = PythonObject(None)
    if source == 0:
        packed = py_call(b[].arrays, fmt)
        _ = py_call(packed.frombytes, py_call(view.tobytes, "C"))
        source = Int(py=packed.buffer_info()[0])
        _ = packed  # keep the Python object alive through the native call above
        itemsize = Int(py=packed.itemsize)
    try:
        mlp_targets_into(source, itemsize, signed, rows, MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(ids.unsafe_ptr())))
    except:
        raise Error("SmallMLPTrainer targets must have shape (batch,) with classes 0..2")
    _ = packed^
    return ids^


def _scalar(value: PythonObject, name: String) raises -> Float64:
    return training_f32_scalar(value, "SmallMLPTrainer " + name,
        SMALL_MLP_BOUNDARY.get_or_create_ptr()[].admission)


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


def _schedule_text(value: PythonObject) raises -> String:
    """The schedule's canonical JSON (sorted keys, compact, ASCII), bounded."""
    return training_schedule_json(value, 128, 2048, "SmallMLPTrainer data_schedule",
        SMALL_MLP_BOUNDARY.get_or_create_ptr()[].admission)


def _is_dict(value: PythonObject) raises -> Bool:
    var bi = SMALL_MLP_BOUNDARY.get_or_create_ptr()[].bi
    return Bool(py=py_call(bi.issubclass, value.__class__, bi.dict))


def _same_keys(value: PythonObject, names: String) raises -> Bool:
    """`value` is a dict whose keys are exactly the `|`-separated `names`."""
    var bi = SMALL_MLP_BOUNDARY.get_or_create_ptr()[].bi
    return _is_dict(value) and Bool(py=value.keys() == py_call(bi.set, py_call(PythonObject(names).split, "|")))


def _handle_of(handle: PythonObject) raises -> Int:
    """The trainer's handle, read from admitted native owner storage."""
    var bi = SMALL_MLP_BOUNDARY.get_or_create_ptr()[].bi
    if py_type(handle) is not bi.int:
        raise Error("RuntimeError: SmallMLPTrainer state is closed or was never created")
    return Int(py=handle)


# ------------------------------------------------------------ state in and out

def _state_from_dict(state: PythonObject, mode: String) raises -> MlpState:
    """`state_dict` output (or a decoded checkpoint) as a validated typed state."""
    var nk = small_mlp_keys()
    var names = _names()
    if not _same_keys(state, "schema|architecture|numeric_mode|parameter_order|weights|optimizer|config|data_schedule"):
        raise Error("SmallMLPTrainer state has missing or unknown fields")
    var order = Python.list(names[0], names[1], names[2], names[3])
    if (Bool(py=py_item(state, nk[].k_schema) != PythonObject(_STATE_SCHEMA)) or Bool(py=py_item(state, nk[].k_architecture) != py_ints([8, 16, 3]))
            or Bool(py=py_item(state, nk[].k_numeric_mode) != PythonObject(mode)) or Bool(py=py_item(state, nk[].k_parameter_order) != order)):
        raise Error("SmallMLPTrainer state schema/architecture/mode/order mismatch")
    var weights = py_item(state, nk[].k_weights)
    if not _same_keys(weights, "weight1|bias1|weight2|bias2"):
        raise Error("SmallMLPTrainer state requires all four named parameters")
    var result = MlpState()
    for i in range(4):  # driver-loop: over the fixed MLP parameter registry
        _words_into(weights[names[i]], _dims(i), names[i], result.params_addr(i))
        mlp_require_finite(result.params_addr(i), mlp_size(i), names[i])
    var cfg = py_item(state, nk[].k_config)
    if not _same_keys(cfg, "lr|beta1|beta2|eps|weight_decay"):
        raise Error("SmallMLPTrainer state optimizer configuration mismatch")
    _config_into(result, py_item(cfg, nk[].k_lr), Python.tuple(py_item(cfg, nk[].k_beta1), py_item(cfg, nk[].k_beta2)), py_item(cfg, nk[].k_eps), py_item(cfg, nk[].k_weight_decay))
    var opt = py_item(state, nk[].k_optimizer)
    if not _same_keys(opt, "kind|step|m|v|flags") or Bool(py=py_item(opt, nk[].k_kind) != PythonObject("AdamW")):
        raise Error("SmallMLPTrainer state optimizer mismatch")
    if _is_bool(py_item(opt, nk[].k_step)):
        raise Error("SmallMLPTrainer step must be an integer")
    var step: Int
    try:
        step = Int(py=py_item(opt, nk[].k_step).__getattr__("__index__")())
    except:
        raise Error("SmallMLPTrainer step must be an integer")
    if step < 0 or step > MLP_MAX_STEP:
        raise Error("SmallMLPTrainer step is outside its supported range")
    result.step = step
    _words_into(py_item(opt, nk[].k_m), [MLP_TOTAL], "m", result.m_addr())
    mlp_require_finite(result.m_addr(), MLP_TOTAL, "m")
    _words_into(py_item(opt, nk[].k_v), [MLP_TOTAL], "v", result.v_addr())
    mlp_require_finite(result.v_addr(), MLP_TOTAL, "v")
    _words_into(py_item(opt, nk[].k_flags), [4], "flags", result.flags_addr(), True)
    mlp_require_moments(result)
    result.schedule = _schedule_text(py_item(state, nk[].k_data_schedule))
    return result^


def _f32_array_of(addr: Int, dims: List[Int]) raises -> PythonObject:
    var out = generation_array(dims)
    mlp_copy_f32(addr, Int(py=out._addr), Int(py=out.size))
    _ = out  # keep the Python object alive through the native call above
    return out


def _weights_out(state: MlpState) raises -> PythonObject:
    var bi = SMALL_MLP_BOUNDARY.get_or_create_ptr()[].bi
    var names = _names()
    var weights = bi.dict()
    for i in range(4):  # driver-loop: over the fixed MLP parameter registry
        var out = generation_array(_dims(i))
        mlp_copy_tensor_out(state, i, Int(py=out._addr))
        _ = out  # keep the Python object alive through the native call above
        py_setitem(weights, names[i], out)
    return weights


def _state_out(state: MlpState, mode: String) raises -> PythonObject:
    """An independent snapshot, in the order and types `state_dict` documents."""
    var nk = small_mlp_keys()
    var bi = SMALL_MLP_BOUNDARY.get_or_create_ptr()[].bi
    var names = _names()
    var result = bi.dict()
    py_setitem(result, nk[].k_schema, PythonObject(_STATE_SCHEMA))
    py_setitem(result, nk[].k_architecture, py_ints([8, 16, 3]))
    py_setitem(result, nk[].k_numeric_mode, PythonObject(mode))
    py_setitem(result, nk[].k_parameter_order, Python.list(names[0], names[1], names[2], names[3]))
    py_setitem(result, nk[].k_weights, _weights_out(state))
    var opt = bi.dict()
    py_setitem(opt, nk[].k_kind, PythonObject("AdamW"))
    py_setitem(opt, nk[].k_step, PythonObject(state.step))
    py_setitem(opt, nk[].k_m, _f32_array_of(state.m_addr(), [MLP_TOTAL]))
    py_setitem(opt, nk[].k_v, _f32_array_of(state.v_addr(), [MLP_TOTAL]))
    var flags = generation_array([4], "i")
    var fp = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=flags._addr))
    _ = flags  # keep the Python object alive through the native call above
    var sp = Pointer[Int32, ImmUntrackedOrigin](unsafe_from_address=state.flags_addr())
    for i in range(4):  # small-loop(the four optimizer flag words)
        fp[i] = sp[i]
        _ = flags  # keep the Python object alive through the native call above
    py_setitem(opt, nk[].k_flags, flags)
    py_setitem(result, nk[].k_optimizer, opt)
    var cfg = bi.dict()
    py_setitem(cfg, nk[].k_lr, PythonObject(state.lr))
    py_setitem(cfg, nk[].k_beta1, PythonObject(state.beta1))
    py_setitem(cfg, nk[].k_beta2, PythonObject(state.beta2))
    py_setitem(cfg, nk[].k_eps, PythonObject(state.eps))
    py_setitem(cfg, nk[].k_weight_decay, PythonObject(state.weight_decay))
    py_setitem(result, nk[].k_config, cfg)
    py_setitem(result, nk[].k_data_schedule, native_json_parse(PythonObject(state.schedule)))
    return result


# ------------------------------------------------------------ the captured program
# RUN is bindings/small_mlp_native_run.mojo (open, step, download, apply,
# close over the typed store). `_steps` is the PACK side of a step: it
# allocates the result arrays, hands their addresses to RUN and builds the
# result objects.

def _steps(index: Int, program: EntryRef, x: List[Float32], y: List[Int32],
           rows: Int, k: Int, mode: Int, want_dx: Bool) raises -> PythonObject:
    """`k` runs of the program (mode 0 forward, 1 loss and gradients, 2 the
    whole step); `[losses, logits, gradients, input_grad]` (the logits and
    gradients are the last step's). The step counter advances under mode 2."""
    var bi = SMALL_MLP_BOUNDARY.get_or_create_ptr()[].bi
    var losses = List[Float32](length=k, fill=Float32(0))
    var grads = mode != SMALL_MLP_RUN_FORWARD
    var logits = generation_array([rows, 3])
    var g0 = generation_array(_dims(0)) if grads else PythonObject(None)
    var g1 = generation_array(_dims(1)) if grads else PythonObject(None)
    var g2 = generation_array(_dims(2)) if grads else PythonObject(None)
    var g3 = generation_array(_dims(3)) if grads else PythonObject(None)
    var dx = generation_array([rows, 8]) if want_dx else PythonObject(None)
    var scratch = List[Float32](length=MLP_TOTAL, fill=Float32(0))
    var sa = Int(scratch.unsafe_ptr())
    # addresses in this exact order (autograd/small_mlp_program.mojo, STEP)
    var addresses: List[Int] = [Int(x.unsafe_ptr()), Int(y.unsafe_ptr()), Int(losses.unsafe_ptr()), Int(py=logits._addr),
        Int(py=g0._addr) if grads else sa + 4 * mlp_offset(0),
        Int(py=g1._addr) if grads else sa + 4 * mlp_offset(1),
        Int(py=g2._addr) if grads else sa + 4 * mlp_offset(2),
        Int(py=g3._addr) if grads else sa + 4 * mlp_offset(3),
        Int(py=dx._addr) if want_dx else sa]
    # The GIL stays held across the program call, as it always was here.
    small_mlp_run_steps(index, program, addresses, rows, k, mode, want_dx)
    _ = dx  # keep the Python objects alive through the native call above
    _ = g3
    _ = g2
    _ = g1
    _ = g0
    _ = scratch^
    # TOMBSTONE: mlp_require_result_finite scans of the step's logits, gradients and input gradient (superseded by the
    # program's device OP_MLP_NONFINITE flags, autograd/small_mlp_program.mojo) deleted 2026-11-09 by lane/m1-followups-misc; code recoverable at 5b5490960d.
    _ = logits
    if mode == SMALL_MLP_RUN_FORWARD:
        _ = losses^
        return Python.list(PythonObject(None), logits, PythonObject(None), PythonObject(None))
    var values = bi.list()
    var append = values.append
    for i in range(k):  # driver-loop: over the k step losses the program returned
        _ = py_call(append, mlp_read_f32(Int(losses.unsafe_ptr()) + 4 * i))
    _ = losses^
    return Python.list(values, logits, Python.list(g0, g1, g2, g3), dx)


def _items(value: PythonObject, limit: Int, refusal: String) raises -> List[PythonObject]:
    var b = SMALL_MLP_BOUNDARY.get_or_create_ptr()
    return small_mlp_items(value, limit, refusal, b[].array_cls, b[].arrays, b[].bi)


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
        py_setitem(result, names[i], grads[i])
    return result


# ------------------------------------------------------------ public methods

def _create_state(args: PythonObject) raises -> MlpState:
    """UNPACK of `__init__`: weight1, bias1, weight2, bias2, data_schedule,
    lr, betas, eps, weight_decay as a validated typed state."""
    var state = MlpState()
    var names = _names()
    for i in range(4):  # driver-loop: over the fixed MLP parameter registry
        _words_into(args[i], _dims(i), names[i], state.params_addr(i))
        mlp_require_finite(state.params_addr(i), mlp_size(i), names[i])
    _config_into(state, args[5], args[6], args[7], args[8])
    state.schedule = _schedule_text(args[4])
    return state^


def _create_handle(var state: MlpState) raises -> PythonObject:
    var b = SMALL_MLP_BOUNDARY.get_or_create_ptr()
    var handle = mlp_store_create(state^)
    py_setitem(b[].locks, handle, b[].rlock())
    return PythonObject(handle)


def small_mlp_create(args: PythonObject) raises -> PythonObject:
    """A state handle with no trainer object, on this installation's device
    (the route of the connected class)."""
    var state = _create_state(args)
    _ = _owner_of(PythonObject(None))
    return _create_handle(state^)


def small_mlp_from_state(cls: PythonObject, state: PythonObject, mode: String) raises -> PythonObject:
    """A new trainer instance of `cls` owning a validated copy of `state`."""
    var nk = small_mlp_keys()
    var b = SMALL_MLP_BOUNDARY.get_or_create_ptr()
    _small_class(cls, b[].trainer_cls)
    var typed = _state_from_dict(state, mode)
    var instance = native_storage_allocate(cls)
    var handle = mlp_store_create(typed^)
    try:
        py_setitem(b[].locks, handle, b[].rlock())
        py_setitem(_fields(instance), nk[].k__handle, PythonObject(handle))
    except error:
        small_mlp_close(PythonObject(handle))
        raise error^
    return instance


def small_mlp_close(handle: PythonObject) raises:
    """Release a trainer's state (its `__del__`); a missing or foreign handle is a no-op."""
    var b = SMALL_MLP_BOUNDARY.get_or_create_ptr()
    if py_type(handle) is not b[].bi.int:
        return
    var h = Int(py=handle)
    try:
        var index = mlp_store_index(h)
        small_mlp_run_close(index)
    except:
        pass
    _ = mlp_store_release(h)
    _ = py_call(b[].locks.pop, h, PythonObject(None))


def small_mlp_state_dict(handle: Int, trainer: PythonObject) raises -> PythonObject:
    var owner = _owner_of(trainer)
    var index = mlp_store_index(handle)
    small_mlp_run_settle(index, False)
    var store = MLP_STORE.get_or_create_ptr()
    return _state_out(store[].slots[index].state, _mode_word(owner))


def small_mlp_load_state(handle: Int, state: PythonObject, mode: String) raises:
    """Validate and copy `state`, then replace the live state in one move."""
    var typed = _state_from_dict(state, mode)
    var index = mlp_store_index(handle)
    small_mlp_run_close(index)
    var store = MLP_STORE.get_or_create_ptr()
    store[].slots[index].state = typed^


def _step_count(handle: Int) raises -> PythonObject:
    var store = MLP_STORE.get_or_create_ptr()
    return PythonObject(store[].slots[mlp_store_index(handle)].state.step)


def _weights(handle: Int, trainer: PythonObject) raises -> PythonObject:
    _ = _owner_of(trainer)
    var index = mlp_store_index(handle)
    small_mlp_run_settle(index, False)
    var store = MLP_STORE.get_or_create_ptr()
    return _weights_out(store[].slots[index].state)


def _train_step(handle: Int, trainer: PythonObject, X: PythonObject, targets: PythonObject, want: PythonObject) raises -> PythonObject:
    var nk = small_mlp_keys()
    var b = SMALL_MLP_BOUNDARY.get_or_create_ptr()
    var x = _batch_in(X)
    var rows = len(x) // 8
    var y = _targets_in(targets, rows)
    if want.__class__ is not b[].bi.bool:
        raise Error("TypeError: SmallMLPTrainer return_input_grad must be a bool")
    var want_dx = Bool(py=want)
    var program = _program_of(trainer)
    var index = mlp_store_index(handle)
    var store = MLP_STORE.get_or_create_ptr()
    if store[].slots[index].state.step == MLP_MAX_STEP:
        raise Error("SmallMLPTrainer step counter is exhausted")
    # The weights stay on the program; a refused step updates nothing.
    var out = _steps(index, program, x, y, rows, 1, SMALL_MLP_RUN_TRAIN, want_dx)
    var result = b[].bi.dict()
    py_setitem(result, nk[].k_step, PythonObject(store[].slots[index].state.step))
    py_setitem(result, nk[].k_loss, out[0][0])
    py_setitem(result, nk[].k_logits, out[1])
    py_setitem(result, nk[].k_gradients, _gradients_dict(out[2]))
    if want_dx:
        py_setitem(result, nk[].k_input_grad, out[3])
    return result


def _train_steps(handle: Int, trainer: PythonObject, X: PythonObject, targets: PythonObject, k_value: PythonObject) raises -> PythonObject:
    var nk = small_mlp_keys()
    var b = SMALL_MLP_BOUNDARY.get_or_create_ptr()
    if k_value.__class__ is not b[].bi.int or Int(py=k_value) < 1:
        raise Error("SmallMLPTrainer train_steps k must be a positive int")
    var k = Int(py=k_value)
    var program = _program_of(trainer)
    var view = _view(X, "SmallMLPTrainer train_steps X must have shape (k * batch, 8)")
    var shape = _shape(X, view)
    if len(shape) != 2 or Int(py=shape[1]) != 8 or Int(py=shape[0]) % k != 0:
        raise Error("SmallMLPTrainer train_steps X must have shape (k * batch, 8)")
    var total = Int(py=shape[0])
    var batch = total // k
    if batch < 1 or batch > 256:
        raise Error("SmallMLPTrainer batch must be in [1, 256]")
    var x = List[Float32](length=total * 8, fill=Float32(0))
    _words_from_view(X, view, shape, [total, 8], "X", Int(x.unsafe_ptr()))
    mlp_require_finite(Int(x.unsafe_ptr()), total * 8, "X")
    var y = _targets_in(targets, total)
    var index = mlp_store_index(handle)
    var store = MLP_STORE.get_or_create_ptr()
    if store[].slots[index].state.step + k > MLP_MAX_STEP:
        raise Error("SmallMLPTrainer step counter is exhausted")
    var res = _steps(index, program, x, y, batch, k, SMALL_MLP_RUN_TRAIN, False)
    var result = b[].bi.dict()
    py_setitem(result, nk[].k_step, PythonObject(store[].slots[index].state.step))
    py_setitem(result, nk[].k_losses, res[0])
    py_setitem(result, nk[].k_loss, res[0][k - 1])
    py_setitem(result, nk[].k_logits, res[1])
    py_setitem(result, nk[].k_gradients, _gradients_dict(res[2]))
    return result


def _predict_logits(handle: Int, trainer: PythonObject, X: PythonObject) raises -> PythonObject:
    """The same program with recording off."""
    var x = _batch_in(X)
    var rows = len(x) // 8
    var program = _program_of(trainer)
    var index = mlp_store_index(handle)
    var zeros = List[Int32](length=rows, fill=Int32(0))
    var res = _steps(index, program, x, zeros, rows, 1, SMALL_MLP_RUN_FORWARD, False)
    var logits = py_item(res, PythonObject(1))
    return logits


def _loss_and_grads(handle: Int, trainer: PythonObject, X: PythonObject, targets: PythonObject) raises -> PythonObject:
    var x = _batch_in(X)
    var rows = len(x) // 8
    var y = _targets_in(targets, rows)
    var program = _program_of(trainer)
    var index = mlp_store_index(handle)
    var res = _steps(index, program, x, y, rows, 1, SMALL_MLP_RUN_GRADS, False)
    return Python.tuple(res[0][0], res[2])


def _apply_gradients(handle: Int, trainer: PythonObject, gradients: PythonObject) raises -> PythonObject:
    var listed = _items(gradients, 5, "SmallMLPTrainer requires exactly four gradients")
    if len(listed) != 4:
        raise Error("SmallMLPTrainer requires exactly four gradients")
    var grads = _pack_gradients(listed)
    var program = _program_of(trainer)
    var index = mlp_store_index(handle)
    var store = MLP_STORE.get_or_create_ptr()
    if store[].slots[index].state.step == MLP_MAX_STEP:
        raise Error("SmallMLPTrainer step counter is exhausted")
    # One AdamW update on the program's weights and moments (the update a
    # train_step applies); the host copy comes down only on export.
    var step = small_mlp_run_apply(index, program, Int(grads.unsafe_ptr()))
    _ = grads^
    return PythonObject(step)


def _predict_sessions(handle: Int, trainer: PythonObject, inputs: PythonObject) raises -> PythonObject:
    """Logits of several independent batches with the current model,
    concatenated in the order given (`mlp_sessions`)."""
    var nk = small_mlp_keys()
    # DEFAULT restored 2026-11-09 by integration lane i2-integrate-seven from
    # 0ce88411d3: the training binding's `mlp_sessions` (one forward per
    # session, training/neural_session_mlp.mojo); lane m2-neural-leftovers
    # had replaced it with one packed forward of the trainer's program with
    # no measurement against it.
    # TOMBSTONE: MOJOLEARN_MLP_SESSIONS_PACKED (opt-in arm, RUN OWED, no A/B recorded) deleted 2026-11-15 by lane/expdel2-02-base; code recoverable at 1c0608ab17.
    # Restore: git apply experiments/removed/MOJOLEARN_MLP_SESSIONS_PACKED.patch; record in docs/TOMBSTONES.md.
    var b = SMALL_MLP_BOUNDARY.get_or_create_ptr()
    var bi = b[].bi
    # The training entries of the trainer's device: a CPU route beside a GPU
    # set runs its sessions on the CPU image.
    var legacy = _training_entries(_owner_of(trainer))
    var index = mlp_store_index(handle)
    small_mlp_run_settle(index, False)
    var listed = _items(inputs, -1, "mojolearn.mlp_inference_sessions: inputs must be a list of arrays")
    # Mojo-owned row-major copies, alive across the call.
    var sessions = List[List[Float32]]()
    var shapes = bi.list()
    var addresses = bi.list()
    for i in range(len(listed)):  # driver-loop: over the caller's session batches
        var value = listed[i]
        var view = _view(value, "mojolearn.mlp_inference_sessions: session input must be float32")
        if _base_format(view) != "f" or Int(py=view.itemsize) != 4:
            raise Error("TypeError: mojolearn.mlp_inference_sessions: session input must be float32")
        var shape = _shape(value, view)
        var dims = List[Int]()
        var n = 1
        for axis in range(len(shape)):  # small-loop(session shape words)
            dims.append(Int(py=shape[axis]))
            n *= dims[axis]
        var owned = List[Float32](length=max(n, 1), fill=Float32(0))
        if n > 0:
            _words_from_view(value, view, shape, dims, "session input", Int(owned.unsafe_ptr()))
            _ = value  # keep the Python object alive through the native call above
        _ = py_call(addresses.append, Int(owned.unsafe_ptr()))
        _ = py_call(shapes.append, shape)
        sessions.append(owned^)
    var total: Int
    try:
        total = Int(py=py_call_typed(py_item(legacy, nk[].k_mlp_sessions_rows), shapes, 8))
    except error:
        raise Error(String(error))
    var out = generation_array([total, 3])
    var store = MLP_STORE.get_or_create_ptr()
    ref s = store[].slots[index].state
    var wrote = Int(py=py_call_typed(py_item(legacy, nk[].k_mlp_sessions),
        Python.list(out._addr, s.params_addr(0), s.params_addr(1), s.params_addr(2), s.params_addr(3)),
        addresses, shapes, py_ints([8, 16, 3])))
    _ = out  # keep the Python object alive through the native call above
    _ = sessions^
    if wrote != Int(py=out.size):
        raise Error("RuntimeError: MLP sessions returned an invalid output count")
    return out


def small_mlp_grads_open(state: PythonObject, mode: String) raises -> Int:
    """A trainer slot holding a validated copy of `state`, with no Python
    trainer object: the parallel owner's gradient workers (bindings/
    parallel_mlp_native.mojo) run their shards on its program."""
    var typed = _state_from_dict(state, mode)
    return mlp_store_create(typed^)


def small_mlp_grads_close(handle: Int) raises:
    small_mlp_close(PythonObject(handle))


def small_mlp_grads_rows(handle: Int, x_addr: Int, y_addr: Int, rows: Int) raises -> PythonObject:
    """`loss_and_grads` on rows the caller already admitted: `x_addr` is
    rows x 8 row-major float32, `y_addr` rows int32 class codes 0..2.
    Returns (loss, [dweight1, dbias1, dweight2, dbias2]); the weights and
    moments are read only."""
    if rows < 1 or rows > 256:
        raise Error("SmallMLPTrainer batch must be in [1, 256]")
    var x = List[Float32](length=rows * 8, fill=Float32(0))
    var y = List[Int32](length=rows, fill=Int32(0))
    unsafe_memcpy(dest=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(x.unsafe_ptr())),
           src=Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=x_addr), count=rows * 8)
    unsafe_memcpy(dest=MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(y.unsafe_ptr())),
           src=Pointer[Int32, ImmUntrackedOrigin](unsafe_from_address=y_addr), count=rows)
    # A worker slot holds no trainer: this installation's device.
    var program = _program_of(PythonObject(None))
    var index = mlp_store_index(handle)
    var res = _steps(index, program, x, y, rows, 1, SMALL_MLP_RUN_GRADS, False)
    return Python.tuple(res[0][0], res[2])


# ------------------------------------------------------------ checkpoint

def _encode_array(value: PythonObject, integer: Bool) raises -> PythonObject:
    """The on-disk tensor descriptor: dtype, shape, and the C-order
    little-endian bytes as lowercase hex (every supported host is
    little-endian, so the stored bytes are the buffer's own)."""
    var nk = small_mlp_keys()
    var bi = SMALL_MLP_BOUNDARY.get_or_create_ptr()[].bi
    var result = bi.dict()
    py_setitem(result, nk[].k_dtype, PythonObject("<i4" if integer else "<f4"))
    py_setitem(result, nk[].k_shape, py_call(bi.list, value.shape))
    py_setitem(result, nk[].k_hex, value._mv.tobytes().hex())
    return result


def _decode_array(value: PythonObject, dims: List[Int], integer: Bool) raises -> PythonObject:
    var nk = small_mlp_keys()
    var b = SMALL_MLP_BOUNDARY.get_or_create_ptr()
    var bi = b[].bi
    var shape = py_ints(dims)
    var dtype = PythonObject("<i4" if integer else "<f4")
    if (not Bool(py=py_call(bi.isinstance, value, bi.dict)) or Bool(py=py_call(bi.set, value) != py_call(bi.set, py_call(PythonObject("dtype|shape|hex").split, "|")))
            or Bool(py=py_item(value, nk[].k_dtype) != dtype) or Bool(py=py_item(value, nk[].k_shape) != shape)
            or py_type(py_item(value, nk[].k_hex)) is not bi.str):
        raise Error("SmallMLPTrainer checkpoint tensor descriptor mismatch")
    var cells = 1
    for i in range(len(dims)):  # small-loop(descriptor shape words)
        cells *= dims[i]
    if len(py_item(value, nk[].k_hex)) != cells * 8:
        raise Error("SmallMLPTrainer checkpoint tensor length mismatch")
    var raw = PythonObject(None)
    try:
        raw = py_call(bi.bytes.fromhex, py_item(value, nk[].k_hex))
    except:
        raise Error("SmallMLPTrainer checkpoint tensor is not hexadecimal")
    if len(raw) != cells * 4:
        raise Error("SmallMLPTrainer checkpoint tensor byte count mismatch")
    var store = py_call(b[].arrays, "i" if integer else "f")
    _ = py_call(store.frombytes, raw)
    var out = generation_array(dims, "i" if integer else "f")
    if integer:
        mlp_copy_i32(Int(py=store.buffer_info()[0]), Int(py=out._addr), cells)
        _ = out  # keep the Python object alive through the native call above
        _ = store  # keep the Python object alive through the native call above
    else:
        mlp_copy_f32(Int(py=store.buffer_info()[0]), Int(py=out._addr), cells)
        _ = out  # keep the Python object alive through the native call above
        _ = store  # keep the Python object alive through the native call above
    _ = store^
    return out


def _decode_state(payload: PythonObject) raises -> PythonObject:
    var nk = small_mlp_keys()
    var bi = SMALL_MLP_BOUNDARY.get_or_create_ptr()[].bi
    if not Bool(py=py_call(bi.isinstance, payload, bi.dict)):
        raise Error("SmallMLPTrainer checkpoint payload must be an object")
    var weights = py_get(payload, nk[].k_weights)
    var opt = py_get(payload, nk[].k_optimizer)
    if weights is None or not Bool(py=py_call(bi.isinstance, opt, bi.dict)) or not py_contains(opt, nk[].k_m) or not py_contains(opt, nk[].k_v) or not py_contains(opt, nk[].k_flags):
        raise Error("SmallMLPTrainer checkpoint state is incomplete")
    var names = _names()
    var order = Python.list(names[0], names[1], names[2], names[3])
    if not Bool(py=py_call(bi.isinstance, weights, bi.dict)) or Bool(py=py_call(bi.set, weights) != py_call(bi.set, order)):
        raise Error("SmallMLPTrainer checkpoint parameter registry mismatch")
    var decoded = bi.dict()
    for i in range(4):  # driver-loop: over the fixed MLP parameter registry
        py_setitem(decoded, names[i], _decode_array(weights[names[i]], _dims(i), False))
    var result = py_call(bi.dict, payload)
    py_setitem(result, nk[].k_weights, decoded)
    var optimizer = py_call(bi.dict, opt)
    py_setitem(optimizer, nk[].k_m, _decode_array(py_item(opt, nk[].k_m), [MLP_TOTAL], False))
    py_setitem(optimizer, nk[].k_v, _decode_array(py_item(opt, nk[].k_v), [MLP_TOTAL], False))
    py_setitem(optimizer, nk[].k_flags, _decode_array(py_item(opt, nk[].k_flags), [4], True))
    py_setitem(result, nk[].k_optimizer, optimizer)
    return result


def _canonical_bytes(value: PythonObject) raises -> PythonObject:
    return py_call(PythonObject(_transport_json(value)).encode, "ascii")


def _save_checkpoint(handle: Int, trainer: PythonObject, path: PythonObject) raises:
    """Atomically write the full state to `path`: a canonical JSON envelope
    with the payload's SHA-256, written to a private temporary file beside
    the target, flushed to disk, then renamed over it."""
    var nk = small_mlp_keys()
    var b = SMALL_MLP_BOUNDARY.get_or_create_ptr()
    var bi = b[].bi
    var posix = b[].posix
    var state = small_mlp_state_dict(handle, trainer)
    var legacy = _training_entries(_owner_of(trainer))
    var names = _names()
    var weights = py_item(state, nk[].k_weights)
    var encoded_weights = bi.dict()
    for i in range(4):  # driver-loop: over the fixed MLP parameter registry
        py_setitem(encoded_weights, names[i], _encode_array(weights[names[i]], False))
    py_setitem(state, nk[].k_weights, encoded_weights)
    var opt = py_item(state, nk[].k_optimizer)
    py_setitem(opt, nk[].k_m, _encode_array(py_item(opt, nk[].k_m), False))
    py_setitem(opt, nk[].k_v, _encode_array(py_item(opt, nk[].k_v), False))
    py_setitem(opt, nk[].k_flags, _encode_array(py_item(opt, nk[].k_flags), True))
    var envelope = bi.dict()
    py_setitem(envelope, nk[].k_schema, PythonObject(_FILE_SCHEMA))
    py_setitem(envelope, nk[].k_payload, state)
    # The digest is the Mojo SHA-256 (core/sha256.mojo), the same hex as hashlib's.
    # In-image call (bindings/sha256_door.mojo over core/sha256.mojo), the same
    # SHA-256 the training binding's `sha256_hex` entry computed: no crossing.
    py_setitem(envelope, nk[].k_payload_sha256, sha256_hexdigest(_canonical_bytes(state)))
    var profile = py_call(bi.str, py_item(legacy, nk[].k_neural_arithmetic_profile)())
    if String(py=profile) != _BASE_PROFILE:
        py_setitem(envelope, nk[].k_arithmetic_profile, profile)
    var data = py_call(PythonObject(_transport_json(envelope) + "\n").encode, "ascii")
    if len(data) > _FILE_LIMIT:
        raise Error("SmallMLPTrainer checkpoint exceeds its size bound")
    var target = native_path(path)
    var parts = py_call(target.rpartition, "/")
    var directory = parts[0] if len(parts[1]) > 0 else PythonObject(".")
    if len(parts[1]) > 0 and len(parts[0]) == 0:
        directory = PythonObject("/")
    var temporary = directory + PythonObject("/.") + parts[2] + PythonObject(".") + py_call(posix.urandom, 8).hex()
    var fd = -1
    try:
        fd = Int(py=py_call(posix.open, temporary, posix.O_WRONLY | posix.O_CREAT | posix.O_EXCL | posix.O_CLOEXEC, 384))
    except error:
        raise Error(_os_error(String(error)))
    var done = False
    try:
        var offset = 0
        var view = py_call(bi.memoryview, data)
        while offset < len(data):  # driver-loop: over partial OS writes of the checkpoint bytes
            offset += Int(py=py_call(posix.write, fd, view[py_call(bi.slice, offset, PythonObject(None))]))
        _ = py_call(posix.fsync, fd)
        _ = py_call(posix.close, fd)
        fd = -1
        _ = py_call(posix.replace, temporary, target)
        done = True
    except error:
        if fd >= 0:
            try:
                _ = py_call(posix.close, fd)
            except:
                pass
        try:
            _ = py_call(posix.unlink, temporary)
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


def _checkpoint_state(path: PythonObject, entries: PythonObject) raises -> PythonObject:
    var nk = small_mlp_keys()
    var b = SMALL_MLP_BOUNDARY.get_or_create_ptr()
    var bi = b[].bi
    var posix = b[].posix
    var source = native_path(path)
    var fd: Int
    try:
        fd = Int(py=py_call(posix.open, source, posix.O_RDONLY | posix.O_CLOEXEC))
    except error:
        raise Error(_os_error(String(error)))
    var chunks = bi.bytearray()
    try:
        while len(chunks) <= _FILE_LIMIT:  # driver-loop: over partial OS reads, at most the size bound plus one byte
            var chunk = py_call(posix.read, fd, _FILE_LIMIT + 1 - len(chunks))
            if len(chunk) == 0:
                break
            _ = py_call(chunks.extend, chunk)
    finally:
        _ = py_call(posix.close, fd)
    if len(chunks) > _FILE_LIMIT:
        raise Error("SmallMLPTrainer checkpoint exceeds its size bound")
    var envelope: PythonObject
    try:
        envelope = native_json_parse_unique(py_call(bi.bytes, chunks))
    except:
        raise Error("SmallMLPTrainer checkpoint is not valid bounded JSON")
    var three = py_call(bi.set, py_call(PythonObject("schema|payload|payload_sha256").split, "|"))
    var four = py_call(bi.set, py_call(PythonObject("schema|payload|payload_sha256|arithmetic_profile").split, "|"))
    if not Bool(py=py_call(bi.isinstance, envelope, bi.dict)) or (Bool(py=py_call(bi.set, envelope) != three) and Bool(py=py_call(bi.set, envelope) != four)):
        raise Error("SmallMLPTrainer checkpoint envelope mismatch")
    if Bool(py=py_item(envelope, nk[].k_schema) != PythonObject(_FILE_SCHEMA)):
        raise Error("SmallMLPTrainer checkpoint schema mismatch")
    var digest: PythonObject
    try:
        digest = sha256_hexdigest(_canonical_bytes(py_item(envelope, nk[].k_payload)))
    except:
        raise Error("SmallMLPTrainer checkpoint integrity mismatch")
    if Bool(py=digest != py_item(envelope, nk[].k_payload_sha256)):
        raise Error("SmallMLPTrainer checkpoint integrity mismatch")
    var expected = py_call(bi.str, py_item(entries, nk[].k_neural_arithmetic_profile)())
    if Bool(py=py_get(envelope, nk[].k_arithmetic_profile, PythonObject(_BASE_PROFILE)) != expected):
        raise Error("SmallMLPTrainer checkpoint arithmetic profile mismatch")
    var state = _decode_state(py_item(envelope, nk[].k_payload))
    return state



def _process_vendor() raises -> Int:
    """This process's vendor word for the device rule: 0 on a CPU-only
    installation."""
    return 0 if SMALL_MLP_BOUNDARY.get_or_create_ptr()[].cpu_only else 1


def _from_checkpoint(cls: PythonObject, path: PythonObject, device: PythonObject) raises -> PythonObject:
    """A trainer restored on `device` (already admitted by the caller): its
    route is resolved before the file is read, the checkpoint's arithmetic
    profile is checked against the image of that device, and the restored
    trainer keeps the route."""
    var nk = small_mlp_keys()
    var b = SMALL_MLP_BOUNDARY.get_or_create_ptr()
    var built = _route_build(device_unpack(device, b[].bi), py_item(b[].backend, nk[].k__DEFAULT_MODE))
    var legacy = built.images
    var instance = small_mlp_from_state(cls, _checkpoint_state(path, legacy), device_mode_word(built.route.mode))
    py_setitem(_fields(instance), "device", device)
    _ = device_route_keep(instance, built^)
    return instance


# ------------------------------------------------------------ passive public owners

# Class-watcher slot of an owner class that passed `_small_class` (payload:
# the address of the connected class it was checked against).
comptime _SMALL_CLASS_SLOT = "p3_small_mlp_owner_class_v1"


def _small_class(cls: PythonObject, required: PythonObject) raises:
    # The verdict depends only on the class rows (metaclass, MRO, namespaces)
    # and connected constants: keep it per class under the type watcher
    # (bindings/native_class_admission.mojo) and re-walk only after the class
    # or an ancestor changed. Python 3.10/3.11 keep the full walk per call.
    var mods = NativeModules()
    var record = class_record(mods, _SMALL_CLASS_SLOT, cls)
    if record is not None and Int(py=record) == Int(required._obj_ptr):
        return
    _small_class_walk(cls, required)
    class_admit(mods, _SMALL_CLASS_SLOT, cls, PythonObject(Int(required._obj_ptr)))


def _small_class_walk(cls: PythonObject, required: PythonObject) raises:
    var nk = small_mlp_keys()
    var boundary = SMALL_MLP_BOUNDARY.get_or_create_ptr()
    var bi = boundary[].bi
    if py_type(cls) is not bi.type:
        raise Error("TypeError: small MLP owners cannot invoke Python metaclass hooks")
    var ancestry = py_call(py_item(bi.type.__dict__, nk[].k___mro__).__get__, cls)
    var found = False
    for i in range(len(ancestry)):  # driver-loop: over passive owner class metadata
        found = found or ancestry[i] is required
        if ancestry[i] is bi.object:
            continue
        var namespace = _class_fields(ancestry[i])
        for name in ["__getattribute__", "__getattr__", "__setattr__", "_handle", "_weights", "_m", "weight_format"]:  # driver-loop: over native storage fields and hooks
            if py_contains(namespace, name):
                raise Error("TypeError: small MLP native storage cannot be replaced by Python hooks or descriptors")
        var finalizer = py_get(namespace, nk[].k___del__)
        if finalizer is not None and finalizer is not boundary[].trainer_finalizer:
            raise Error("TypeError: small MLP owners require the connected native finalizer")
    if not found:
        raise Error("TypeError: small MLP operation requires its connected native owner class")


def _small_fields(owner: PythonObject, required: PythonObject) raises -> PythonObject:
    _small_class(py_type(owner), required)
    return _fields(owner)


def _owner_operation(operation: Int, name: PythonObject, args: PythonObject) raises -> PythonObject:
    # SmallMLPTrainer.__del__ during interpreter teardown (the mlp lanes
    # and neural-trainer-unfused aborted after writing their JSON:
    # "PyImport_GetModuleDict: interpreter has no modules dictionary",
    # rc -6): the close reads loaded modules, which is fatal once the
    # modules are cleared. The process is ending; nothing to release.
    # Same rule as xcnn_native_pack and host_archive_lifetime_door. Tested
    # before the key table is fetched, so teardown reads nothing.
    if operation == SMALL_MLP_OPN_OWNER_CLOSE and native_interpreter_finalizing():
        return PythonObject(None)
    var nk = small_mlp_keys()
    var b = SMALL_MLP_BOUNDARY.get_or_create_ptr()
    if operation == SMALL_MLP_OPN_OWNER_CLOSE:
        try:
            var fields = _small_fields(args[0], b[].trainer_cls)
            var handle = py_get(fields, nk[].k__handle)
            if handle is not None:
                small_mlp_close(handle)
                py_setitem(fields, nk[].k__handle, PythonObject(None))
        except:
            pass  # Failed construction and interpreter shutdown are best-effort.
        return PythonObject(None)
    var fields = _small_fields(args[0], b[].trainer_cls)
    classical_connect_numpy(b[].admission)
    if operation == SMALL_MLP_OPN_OWNER_CREATE:
        # `device` is checked first, by the shared admission, before any
        # field is read or published; None is one identity compare.
        var values = args[1]
        var device = PythonObject(None)
        if len(values) > 9:
            device = values[9]
        device_admit(device, _process_vendor(), b[].bi)
        sequential_admit_value(values, b[].admission)
        var state = _create_state(values)
        # The route is resolved here, once (so `device="cpu"` with fast mode,
        # or without its CPU image, refuses at construction), and kept: the
        # trainer's first step is already a hit.
        var built = _route_build(device_unpack(device, b[].bi), py_item(b[].backend, nk[].k__DEFAULT_MODE))
        var handle = _create_handle(state^)
        try:
            var previous = py_get(fields, nk[].k__handle)
            if previous is not None:
                small_mlp_close(previous)
            py_setitem(fields, nk[].k__handle, handle)
            py_setitem(fields, "device", device)
        except error:
            small_mlp_close(handle)
            raise error^
        # Outside the handle's failure path: a route that could not be kept
        # is resolved again by the first operation.
        device_route_clear(args[0])
        _ = device_route_keep(args[0], built^)
        return PythonObject(None)
    var packed = Python.list(py_item(fields, nk[].k__handle))
    for i in range(1, len(args)):  # driver-loop: over native argument envelope
        _ = py_call(packed.append, args[i])
    var values = py_call(b[].bi.tuple, packed)
    # Every public trainer method now returns its final value from native code;
    # in particular load_state_dict returns the original owner here.
    # owner_<op> forwards <op>: the owner codes are the trainer codes shifted
    # by one block (bindings/small_mlp_keys_unpack.mojo order).
    var inner = operation - (SMALL_MLP_OPN_OWNER_TRAIN_STEP - SMALL_MLP_OPN_TRAIN_STEP)
    var inner_name = name
    if operation < SMALL_MLP_OPN_OWNER_TRAIN_STEP or operation > SMALL_MLP_OPN_OWNER_SAVE_CHECKPOINT:
        # Miss path only (an "owner_" name outside the table): forward the
        # rest of the name, as the former door did.
        var full = String(name)
        inner_name = PythonObject(String(full[byte=6:]))
        inner = py_op_code(nk[].ops, inner_name)
    var result = _call(inner, inner_name, values, args[0])
    return args[0] if operation == SMALL_MLP_OPN_OWNER_LOAD_STATE_DICT else result


def small_mlp_finalize_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=args_ptr)
        if len(args) == 2:
            _ = _owner_operation(SMALL_MLP_OPN_OWNER_CLOSE, PythonObject(None), Python.tuple(args[1]))
    except:
        pass
    return PythonObject(None).steal_data()


# ------------------------------------------------------------ MLPInference host owner

def _inference_entries() raises -> PythonObject:
    var nk = small_mlp_keys()
    var b = SMALL_MLP_BOUNDARY.get_or_create_ptr()
    var entries = py_item(b[].inference, nk[].k_entries)
    if py_type(entries) is b[].bi.str:
        raise Error("ImportError: MLPInference requires its native host binding: " + String(entries))
    if py_type(entries) is not b[].bi.dict:
        raise Error("TypeError: MLPInference host entries require their connected native mapping")
    for name in ["mlp_forward_logits", "sha256_hex", "neural_arithmetic_profile"]:  # driver-loop: over the three connected host entries
        if neural_descriptor_entry(py_item(entries, nk[].k_module), name) is not entries[PythonObject(name)]:
            raise Error("TypeError: MLPInference host entry changed after connection")
    return entries


def _packed_admit(value: PythonObject, context: PythonObject) raises:
    var nk = small_mlp_keys()
    var b = SMALL_MLP_BOUNDARY.get_or_create_ptr()
    if py_type(value) is not py_item(context, nk[].k_family_low_bf16) and py_type(value) is not py_item(context, nk[].k_family_low_int8):
        sequential_admit_value(value, b[].admission)
        return
    var cls = py_type(value)
    if py_type(cls) is not b[].bi.type:
        raise Error("TypeError: MLPInference packed weights require native storage classes")
    var ancestry = py_call(py_item(b[].bi.type.__dict__, nk[].k___mro__).__get__, cls)
    for i in range(len(ancestry)):  # driver-loop: over packed weight storage metadata
        if ancestry[i] is b[].bi.object:
            continue
        var namespace = _class_fields(ancestry[i])
        for name in ["__getattribute__", "__getattr__", "__setattr__", "bits", "codes", "exponents", "shape"]:  # driver-loop: over packed storage hooks and fields
            if py_contains(namespace, name):
                raise Error("TypeError: MLPInference packed storage refuses Python hooks and descriptors")
    _ = _fields(value)


def _inference_publish(fields: PythonObject, state: MlpState, format: String, entries: PythonObject) raises:
    var nk = small_mlp_keys()
    var weights = Python.list()
    for i in range(4):  # driver-loop: over the fixed inference weight registry
        _ = py_call(weights.append, _f32_array_of(state.params_addr(i), _dims(i)))
    py_setitem(fields, nk[].k__weights, weights)
    py_setitem(fields, nk[].k_weight_format, PythonObject(format))
    py_setitem(fields, nk[].k__m, py_item(entries, nk[].k_module))


def _inference_state(fields: PythonObject) raises -> MlpState:
    var nk = small_mlp_keys()
    var b = SMALL_MLP_BOUNDARY.get_or_create_ptr()
    var values = py_item(fields, nk[].k__weights)
    if (py_type(values) is not b[].bi.list and py_type(values) is not b[].bi.tuple) or len(values) != 4:
        raise Error("TypeError: MLPInference requires four native weight arrays")
    sequential_admit_value(values, b[].admission)
    var state = MlpState()
    var names = _names()
    for i in range(4):  # driver-loop: over four inference inputs; each copied from captured metadata
        _words_into(values[i], _dims(i), names[i], state.params_addr(i))
        mlp_require_finite(state.params_addr(i), mlp_size(i), names[i])
    return state^


def _inference_operation(operation: Int, args: PythonObject) raises -> PythonObject:
    var nk = small_mlp_keys()
    var b = SMALL_MLP_BOUNDARY.get_or_create_ptr()
    classical_connect_numpy(b[].admission)
    var cls = py_item(b[].inference, nk[].k_class)
    if operation == SMALL_MLP_OPN_INFERENCE_CHECKPOINT:
        _small_class(args[0], cls)
        var entries = _inference_entries()
        var payload = _checkpoint_state(args[1], entries)
        var mode = py_item(b[].backend, nk[].k__DEFAULT_MODE)
        if py_type(mode) is not b[].bi.str or (String(mode) != "identical" and String(mode) != "fast"):
            raise Error(_MODE_REFUSAL)
        var state = _state_from_dict(payload, String(mode))
        var owner = native_storage_allocate(args[0])
        _inference_publish(_fields(owner), state, "float32", entries)
        return owner
    var fields = _small_fields(args[0], cls)
    if operation == SMALL_MLP_OPN_INFERENCE_REPR:
        return PythonObject("MLPInference(architecture=[8, 16, 3])")
    if operation == SMALL_MLP_OPN_INFERENCE_INITIALIZE:
        var lowbit = py_item(b[].inference, nk[].k_lowbit)
        for i in range(1, 5):  # driver-loop: over native weights before conversion
            if i == 1 or i == 3:
                _packed_admit(args[i], lowbit)
            else:
                sequential_admit_value(args[i], b[].admission)
        var format = low_formats(Python.tuple(args[1], args[3]), lowbit)
        var low_binding = PythonObject(None)
        var w1 = low_materialize(args[1], low_binding, lowbit)
        var w2 = low_materialize(args[3], low_binding, lowbit)
        var values = Python.tuple(w1, args[2], w2, args[4])
        sequential_admit_value(values, b[].admission)
        var state = MlpState()
        var names = _names()
        for i in range(4):  # driver-loop: over fixed initialized parameter arrays
            _words_into(values[i], _dims(i), names[i], state.params_addr(i))
            mlp_require_finite(state.params_addr(i), mlp_size(i), names[i])
        _inference_publish(fields, state, format, _inference_entries())
        return PythonObject(None)
    var entries = _inference_entries()
    if py_item(fields, nk[].k__m) is not py_item(entries, nk[].k_module):
        raise Error("TypeError: MLPInference requires its connected native host module")
    var state = _inference_state(fields)
    if operation == SMALL_MLP_OPN_INFERENCE_WEIGHTS:
        return _weights_out(state)
    if operation == SMALL_MLP_OPN_INFERENCE_PREDICT:
        sequential_admit_value(args[1], b[].admission)
        var x = _batch_in(args[1])
        var rows = len(x) // 8
        var output = generation_array([rows, 3])
        var address = Int(py=output._addr)
        var wrote = Int(py=py_call_typed(py_item(entries, nk[].k_mlp_forward_logits), Python.list(Int(x.unsafe_ptr()), state.params_addr(0),
            state.params_addr(1), state.params_addr(2), state.params_addr(3), address), Python.list(rows)))
        _ = x^
        _ = state^
        if wrote != rows * 3:
            raise Error("RuntimeError: MLPInference returned an invalid result count")
        mlp_require_result_finite(address, rows * 3, "MLPInference")
        _ = output
        return output
    raise Error("ValueError: unknown MLPInference public operation")


# ------------------------------------------------------------ dispatch

def _locked(operation: Int, name: PythonObject, handle_value: PythonObject, args: PythonObject,
            trainer: PythonObject) raises -> PythonObject:
    var b = SMALL_MLP_BOUNDARY.get_or_create_ptr()
    var handle = _handle_of(handle_value)
    _ = mlp_store_index(handle)
    var lock = b[].locks[handle]
    _ = lock.acquire()
    var result = PythonObject(None)
    try:
        if operation == SMALL_MLP_OPN_TRAIN_STEP:
            result = _train_step(handle, trainer, args[1], args[2], args[3])
        elif operation == SMALL_MLP_OPN_TRAIN_STEPS:
            result = _train_steps(handle, trainer, args[1], args[2], args[3])
        elif operation == SMALL_MLP_OPN_PREDICT_LOGITS:
            result = _predict_logits(handle, trainer, args[1])
        elif operation == SMALL_MLP_OPN_PREDICT_LOGITS_SESSIONS:
            result = _predict_sessions(handle, trainer, args[1])
        elif operation == SMALL_MLP_OPN_LOSS_AND_GRADS:
            result = _loss_and_grads(handle, trainer, args[1], args[2])
        elif operation == SMALL_MLP_OPN_APPLY_GRADIENTS:
            result = _apply_gradients(handle, trainer, args[1])
        elif operation == SMALL_MLP_OPN_STATE_DICT:
            result = small_mlp_state_dict(handle, trainer)
        elif operation == SMALL_MLP_OPN_LOAD_STATE_DICT:
            small_mlp_load_state(handle, args[1], _mode_word(_owner_of(trainer)))
        elif operation == SMALL_MLP_OPN_STEP:
            result = _step_count(handle)
        elif operation == SMALL_MLP_OPN_WEIGHTS:
            result = _weights(handle, trainer)
        elif operation == SMALL_MLP_OPN_SAVE_CHECKPOINT:
            _save_checkpoint(handle, trainer, args[1])
        else:
            raise Error("SmallMLPTrainer: unknown operation " + String(name))
    finally:
        if Python().cpython().PyErr_Occurred():
            # _items preserves native iterator errors. Clear the indicator
            # while releasing the C lock, then restore owned exception refs;
            # a cleanup call must not replace the original iteration failure.
            var pending = ByteCheckpointIOError()
            pending.capture()
            try:
                _ = lock.release()
            except:
                _ = pending^.restore()
                raise Error(SMALL_MLP_ITEMS_PENDING)
            _ = pending^.restore()
        else:
            _ = lock.release()
    return result


def small_mlp_call(operation: Int, name: PythonObject, args: PythonObject) raises -> PythonObject:
    """Dispatch one trainer/inference operation by its code
    (bindings/small_mlp_keys_unpack.mojo). `name` is the caller's operation
    object, read only for an unknown name's routes and message. An operation
    on a bare state handle runs on this installation's device."""
    return _call(operation, name, args, PythonObject(None))


def _call(operation: Int, name: PythonObject, args: PythonObject, trainer: PythonObject) raises -> PythonObject:
    """`small_mlp_call` with the trainer object whose route the operation
    takes (None: a bare state handle)."""
    if operation == SMALL_MLP_OPN_INITIALIZE:
        small_mlp_initialize(args[0])
        return PythonObject(None)
    if operation == SMALL_MLP_OPN_INFERENCE_CONNECT:
        SMALL_MLP_BOUNDARY.get_or_create_ptr()[].inference = args[0]
        return PythonObject(None)
    if operation >= SMALL_MLP_OPN_INFERENCE_CHECKPOINT and operation <= SMALL_MLP_OPN_INFERENCE_PREDICT:
        return _inference_operation(operation, args)
    if operation >= SMALL_MLP_OPN_OWNER_TRAIN_STEP and operation <= SMALL_MLP_OPN_OWNER_CLOSE:
        return _owner_operation(operation, name, args)
    if operation < 0:
        # Miss path only: a name outside the table keeps the former prefix
        # routes and their refusals.
        var text = String(name)
        if text.startswith("inference_"):
            return _inference_operation(operation, args)
        if text.startswith("owner_"):
            return _owner_operation(operation, name, args)
    var boundary = SMALL_MLP_BOUNDARY.get_or_create_ptr()
    classical_connect_numpy(boundary[].admission)
    # Shared exact-native-type admission precedes every protocol/attribute read.
    # No caller-defined __buffer__, iterator, descriptor or scalar hook runs.
    var device = PythonObject(None)
    if operation == SMALL_MLP_OPN_FROM_CHECKPOINT:
        # `device` first, by the shared admission, before the class or the
        # file is read.
        if len(args) > 2:
            device = args[2]
        device_admit(device, _process_vendor(), boundary[].bi)
        _small_class(args[0], boundary[].trainer_cls)
        _ = native_path(args[1])
    elif operation == SMALL_MLP_OPN_SAVE_CHECKPOINT:
        sequential_admit_value(args[0], boundary[].admission)
        _ = native_path(args[1])
    else:
        sequential_admit_value(args, boundary[].admission)
    # TOMBSTONE: the small_mlp_api_v1 provider (the per-operation and fused routes' descriptor) deleted 2026-11-09 by lane/x2-small-mlp-on-executor; code recoverable at f59d913b0b.
    # Its leftover `native_provider_info` operation (no caller) deleted 2026-11-22 by lane neural-consolidate.
    if operation == SMALL_MLP_OPN_CREATE:
        return small_mlp_create(args)
    if operation == SMALL_MLP_OPN_CLOSE:
        small_mlp_close(args[0])
        return PythonObject(None)
    if operation == SMALL_MLP_OPN_FROM_CHECKPOINT:
        return _from_checkpoint(args[0], args[1], device)
    return _locked(operation, name, args[0], args, trainer)


def small_mlp_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    var args = PythonObject(from_borrowed=args_ptr)
    try:
        # One lookup with the caller's own str (bindings/py_keys.mojo): the
        # former door copied the name into a String, tested two prefixes and
        # compared it against up to 17 names.
        var nk = small_mlp_keys()
        var name = args[0]
        var result = small_mlp_call(py_op_code(nk[].ops, name), name, args[1])
        return result.steal_data()
    except error:
        var message = String(error)
        if message == SMALL_MLP_ITEMS_PENDING:
            return PyObjectPtr()  # Native iterator error remains set, with original class/traceback.
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
