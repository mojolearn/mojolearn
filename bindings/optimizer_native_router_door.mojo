# SPDX-License-Identifier: Apache-2.0
# Native public-operation routing, constructors, state preparation and resident IO.
from bindings.embedding_layer_native_door import emblayer_operation, emblayer_unknown_operation
from std.os import getenv
from std.python import Python, PythonObject
from std.python._cpython import GILReleased, PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.sequence_layers_native_door import sequence_layer_norm
from bindings.sequence_moe_native_door import sequence_moe_operation
from bindings.training_extra_native_door import training_chunked_lm_head, training_mlp_sessions, training_residual_dropout
from bindings.samba_head_native_unpack import training_samba_head_loss
from bindings.training_function_device_unpack import training_function_op, training_function_device, training_entries_cpu
from core.function_device import function_device_default
from providers.python.function_device_boundary_door import function_device_admit
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem, py_contains
from bindings.py_typed_call import py_call_typed
from bindings.optimizer_public_words_unpack import optimizer_python_error, stash_python_error
from bindings.optimizer_admission_door import opt_guards, opt_owner, opt_owner_mode, opt_owner_word, opt_value, opt_scalar, opt_schedule, opt_exact
from bindings.optimizer_public_door import (
    OPT_DOOR,
    door_ready, owner_kind, refuse_kwargs, OptimizerRegistryPrepared, optimizer_prepare_registry, optimizer_publish_registry, new_array, TensorView, tensor_view,
    door_step, door_io, optimizer_io, optimizer_pack_io, optimizer_load_buffers, optimizer_export_state, door_state, door_state_set, door_zero, door_clip, door_accumulate,
)
from bindings.optimizer_descriptor_boundary_door import optimizer_provider_entries, optimizer_native_mode, optimizer_driver
from core.linear_metadata import linear_mode_name
from core.shared_slot import SharedSlot
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_fields, py_get, py_setitem, py_type
from bindings.optimizer_admission_door import opt_owner, opt_value
from bindings.optimizer_owner_boundary_unpack import OptimizerOwner, OptimizerOwnerLease, optimizer_owner_retained, optimizer_owner_close, optimizer_owner_admit_replacement
from bindings.optimizer_public_door import OPT_DOOR, owner_kind
from bindings.optimizer_public_words_unpack import stash_python_error
from bindings.optimizer_scalar_boundary_unpack import optimizer_truth
from training.optimizer_owner_control import optimizer_restored_pristine
from training.optimizer_serialization_run import optimizer_serialized_field
from bindings.optimizer_owner_boundary_unpack import OptimizerOwner, OptimizerOwnerLease, optimizer_owner_value, optimizer_owner_close, optimizer_owner_idle, optimizer_owner_admit_replacement
from bindings.optimizer_owner_boundary_pack import optimizer_owner_new
from training.checks.optimizer_contract import microbatch_split_is_identical
from training.optimizer_config import OptimizerConstructorStart, OptimizerConstructorConfig, optimizer_parameter_names, optimizer_parameter_update, optimizer_profile_available, optimizer_beta_shape, optimizer_validate_accumulation, optimizer_validate_nesterov_momentum, optimizer_validate_nesterov_dampening
from bindings.py_cpython_table import cpython_table
from bindings.optimizer_container_boundary_unpack import OptimizerInputOwners
from training.optimizer_state_plan import optimizer_checkpoint_input, optimizer_state_count
from bindings.optimizer_scalar_boundary_unpack import optimizer_float, optimizer_integer, optimizer_truth
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.classical_arrays_door import classical_input, classical_cast_kind
from core.optimizer_state_impl import optimizer_state_cast
from core.classical_pandas_impl import pd_raw_at, _pd_double
from bindings.sequence_device_native_pack import sequence_device_operation, sequence_device_finalize
from bindings.sequence_optimizer_native_door import sequence_optimizer_operation
from bindings.optimizer_schedule_native_door import schedule_run
from bindings.optimizer_device_native_door import device_operation, device_new, device_entries, device_adopt
from bindings.training_linear_native_unpack import training_linear_operation
from bindings.training_rms_native_unpack import training_rms_operation
from bindings.training_embedding_native_door import training_embedding_operation
from bindings.training_cross_entropy_native_door import training_cross_entropy
from bindings.training_metadata_native_pack import training_metadata
from bindings.py_model_records import bump_model_generation
from bindings.optimizer_keys_unpack import optimizer_keys
from bindings.py_balanced import py_item
from bindings.optimizer_keys_unpack import optimizer_keys_in_finalizer
from bindings.training_grads_device_unpack import grads_device_input, clip_device, accumulate_device, mlp_sessions_device
from bindings.optimizer_keys_unpack import (
    OPTIMIZER_OPN_TRAINING_NUMERIC_MODE, OPTIMIZER_OPN_TRAINING_VENDOR, OPTIMIZER_OPN_SEQDEV_FINALIZE,
    OPTIMIZER_OPN_SEQDEV_ARRAY, OPTIMIZER_OPN_SEQDEV_WRAP, OPTIMIZER_OPN_RESIDUAL_DROPOUT,
    OPTIMIZER_OPN_RESIDUAL_DROPOUT_BACKWARD, OPTIMIZER_OPN_EMBLAYER_INIT, OPTIMIZER_OPN_EMBLAYER_LOAD,
    OPTIMIZER_OPN_SEQLN_INITIALIZE, OPTIMIZER_OPN_SEQLN_LAYER_BACKWARD, OPTIMIZER_OPN_SEQMOE_INITIALIZE, OPTIMIZER_OPN_SEQMOE_TO_DEVICE, OPTIMIZER_OPN_SEQ_CONSTRUCT,
    OPTIMIZER_OPN_SEQ_FINALIZE, OPTIMIZER_OPN_DEVICE_CONSTRUCT, OPTIMIZER_OPN_DEVICE_UPLOAD,
    OPTIMIZER_OPN_DEVICE_HOST, OPTIMIZER_OPN_SCHEDULE_CONSTRUCT, OPTIMIZER_OPN_SCHEDULE_SETATTR,
    OPTIMIZER_OPN_SCHEDULE_DELATTR, OPTIMIZER_OPN_SCHEDULE_BITS, OPTIMIZER_OPN_RESTORE_OWNER,
    OPTIMIZER_OPN_CONSTRUCT, OPTIMIZER_OPN_GET_PARAMS, OPTIMIZER_OPN_SET_PARAMS, OPTIMIZER_OPN_EXPORT_OWNER,
    OPTIMIZER_OPN_LINEAR_FORWARD, OPTIMIZER_OPN_LINEAR_BACKWARD, OPTIMIZER_OPN_RMS_NORM_FORWARD,
    OPTIMIZER_OPN_RMS_NORM_BACKWARD, OPTIMIZER_OPN_EMBEDDING_FORWARD, OPTIMIZER_OPN_EMBEDDING_BACKWARD,
    OPTIMIZER_OPN_CROSS_ENTROPY, OPTIMIZER_OPN_CHUNKED_LM_HEAD_LOSS, OPTIMIZER_OPN_SAMBA_HEAD_LOSS,
    OPTIMIZER_OPN_MLP_INFERENCE_SESSIONS, OPTIMIZER_OPN_BINDING, OPTIMIZER_OPN_RAW_IO, OPTIMIZER_OPN_PARAM_DEVICE,
    OPTIMIZER_OPN_STEP, OPTIMIZER_OPN_STATE, OPTIMIZER_OPN_STATE_SET, OPTIMIZER_OPN_LOAD, OPTIMIZER_OPN_ZERO,
    OPTIMIZER_OPN_RESIDENT, OPTIMIZER_OPN_IO, OPTIMIZER_OPN_CLIP, OPTIMIZER_OPN_ACCUMULATE, OPTIMIZER_OPN_ALIGNED,
)
from bindings.py_keys import py_op_code


# ---- from bindings/optimizer_serialization_boundary_pack.mojo (inlined: its one importer is this file) ----
# SPDX-License-Identifier: Apache-2.0
# Optimizer copy/pickle UNPACK/PACK; process-local handles never serialize.


def optimizer_restore_state(owner: PythonObject, state: PythonObject, context: PythonObject) raises -> PythonObject:
    opt_owner(owner, context)
    opt_value(state, context)
    var nk = optimizer_keys()
    if py_type(state) is not py_item(context, nk[].k_container_types)[2]:
        raise Error("TypeError: optimizer serialized state requires a native dictionary")
    var pristine_value = py_get(state, nk[].k__native_optimizer_pristine, py_get(state, nk[].k__pristine, False))
    var pristine = optimizer_restored_pristine(optimizer_truth(pristine_value, context),
        py_get(state, nk[].k__m) is not None, py_get(state, nk[].k__v) is not None)
    var fields = py_fields(owner)
    optimizer_owner_admit_replacement(owner, fields, context)
    var table = cpython_table()
    var door = OPT_DOOR.get_or_create_ptr()
    var saved_pointer = table[].PyDict_Copy(fields._obj_ptr)
    if not saved_pointer:
        var failure = stash_python_error()
        raise failure^
    var saved = PythonObject(from_owned=saved_pointer)
    var replacement = optimizer_owner_new(owner, owner_kind(owner), pristine)
    var publication = OptimizerOwnerLease(owner, replacement)
    var old = py_get(fields, nk[].k__native_optimizer_owner)
    if old is not None:
        _ = optimizer_owner_retained(old, owner)
        old.downcast_value_ptr[OptimizerOwner]()[].control.begin()
    try:
        door[].dict_clear(fields._obj_ptr)
        var position = List[Int](length=1, fill=0)
        var pair = List[PyObjectPtr](length=2, fill=PyObjectPtr())
        while table[].PyDict_Next(state._obj_ptr, Int(position.unsafe_ptr()),
                Int(pair.unsafe_ptr()), Int(pair.unsafe_ptr() + 1)):  # driver-loop: admitted external serialized fields
            var key = PythonObject(from_borrowed=pair[0])
            if optimizer_serialized_field(String(key)):
                py_setitem(fields, key, PythonObject(from_borrowed=pair[1]))
        py_setitem(fields, nk[].k__native_optimizer_owner, replacement)
        opt_owner(owner, context)
        if old is not None:
            optimizer_owner_close(old, owner)
    except error:
        door[].dict_clear(fields._obj_ptr)
        if door[].dict_update(fields._obj_ptr, saved._obj_ptr) != 0:
            if old is not None:
                old.downcast_value_ptr[OptimizerOwner]()[].control.finish()
            var failure = stash_python_error()
            raise failure^
        if old is not None:
            old.downcast_value_ptr[OptimizerOwner]()[].control.finish()
        raise error^
    if old is not None:
        old.downcast_value_ptr[OptimizerOwner]()[].control.finish()
    _ = old
    _ = publication
    return PythonObject(None)

struct OptimizerModeRoutes(Defaultable, Movable):
    # `held`: (mode object, context, entries) triples of resolved numeric
    # modes, at most OPTIMIZER_MODE_ROUTE_LIMIT of them. The mode object is an
    # exact str (immutable) and is pinned here, so its identity names its word
    # for as long as the triple is held. `identical`: the retained word of
    # the identical-only operations.
    var held: List[PythonObject]
    var identical: List[PythonObject]
    # lane vdf-seq: `held`'s form for the CPU identical training image of an
    # explicit device="cpu" beside a GPU set (`optimizer_entries_cpu`)
    var cpu_held: List[PythonObject]

    def __init__(out self):
        self.held = List[PythonObject]()
        self.identical = List[PythonObject]()
        self.cpu_held = List[PythonObject]()


# The named global may be shared by every Mojo extension in the process:
# bump the name's version whenever OptimizerModeRoutes' layout changes.
comptime OPTIMIZER_MODE_ROUTES = SharedSlot[OptimizerModeRoutes, "MojolearnOptimizerModeRoutesV2"]
comptime OPTIMIZER_MODE_ROUTE_LIMIT = 8


def optimizer_entries(mode: PythonObject, context: PythonObject) raises -> PythonObject:
    """The entry dictionary of `mode` (None: the installation's default mode).

    lane dw-autograd-seq (2026-11-21): resolved once per mode object. An
    optimizer keeps one `numeric_mode` object for its life (and None reads
    the one default-mode object), so every operation after its first is
    pointer compares against the held triples. Before, EVERY optimizer /
    training operation built the mode's String, normalised it, built a Python
    str key for `bindings[word]`, resolved the provider and compared its
    compiled mode. A different mode object, a different context or more than
    OPTIMIZER_MODE_ROUTE_LIMIT live mode objects resolve in full again; a
    refusal is never held. The compiled mode of a connected binding cannot
    change within a process."""
    var selected = mode
    var nk = optimizer_keys()
    if selected is None:
        selected = py_item(py_item(context, nk[].k_backend), nk[].k__DEFAULT_MODE)
    var routes = OPTIMIZER_MODE_ROUTES.get_or_create_ptr()
    var held_count = len(routes[].held)
    var at = 0
    while at + 2 < held_count:  # driver-loop: over at most OPTIMIZER_MODE_ROUTE_LIMIT resolved mode objects
        if routes[].held[at] is selected and routes[].held[at + 1] is context:
            var found = routes[].held[at + 2]
            return found
        at += 3
    if py_type(selected) is not py_item(context, nk[].k_scalar_types)[5]:
        raise Error("ValueError: numeric_mode must be fast, identical or None")
    var word = String()
    try:
        word = linear_mode_name(String(selected))
    except:
        raise Error("ValueError: numeric_mode must be fast, identical or None")
    var entries = py_item(context, nk[].k_bindings)[PythonObject(word)]
    if py_type(entries) is py_item(context, nk[].k_scalar_types)[5]:
        raise Error("ImportError: " + String(entries))
    var provider = optimizer_provider_entries(entries)
    if optimizer_native_mode(provider) != (1 if word == "identical" else 0):
        raise Error("ImportError: optimizer binary numeric mode disagrees with the requested mode")
    if held_count >= 3 * OPTIMIZER_MODE_ROUTE_LIMIT:
        routes[].held.clear()
    routes[].held.append(selected)
    routes[].held.append(context)
    routes[].held.append(entries)
    return entries


def optimizer_entries_cpu(mode: PythonObject, context: PythonObject) raises -> PythonObject:
    """`training_entries_cpu` held as `optimizer_entries` holds the device
    entries (lane vdf-seq, 2026-11-24): resolved once per mode object and
    context, then pointer compares. A device="cpu" SGD / Adam / AdamW keeps
    one `numeric_mode` object for its life, so every operation after its
    first reads the held triple. A refusal is never held."""
    var selected = mode
    if selected is None:
        var nk = optimizer_keys()
        selected = py_item(py_item(context, nk[].k_backend), nk[].k__DEFAULT_MODE)
    var routes = OPTIMIZER_MODE_ROUTES.get_or_create_ptr()
    var held_count = len(routes[].cpu_held)
    var at = 0
    while at + 2 < held_count:  # driver-loop: over at most OPTIMIZER_MODE_ROUTE_LIMIT resolved mode objects
        if routes[].cpu_held[at] is selected and routes[].cpu_held[at + 1] is context:
            var found = routes[].cpu_held[at + 2]
            return found
        at += 3
    var entries = training_entries_cpu(selected, context)
    if held_count >= 3 * OPTIMIZER_MODE_ROUTE_LIMIT:
        routes[].cpu_held.clear()
    routes[].cpu_held.append(selected)
    routes[].cpu_held.append(context)
    routes[].cpu_held.append(entries)
    return entries


def optimizer_entries_identical(context: PythonObject) raises -> PythonObject:
    """`optimizer_entries` of the identical mode for the operations that run
    in identical mode only; the word object is kept, so they hit the held
    triple (a new `PythonObject("identical")` per call never would)."""
    var routes = OPTIMIZER_MODE_ROUTES.get_or_create_ptr()
    if len(routes[].identical) == 0:
        routes[].identical.append(PythonObject("identical"))
    return optimizer_entries(routes[].identical[0], context)


def optimizer_betas(source: PythonObject, context: PythonObject) raises -> Tuple[Float64, Float64]:
    opt_value(source, context)
    var nk = optimizer_keys()
    if py_type(source) is py_item(context, nk[].k_container_types)[1] or py_type(source) is py_item(context, nk[].k_container_types)[0]:
        opt_scalar(source, context)
        if len(source) != 2:
            raise Error("ValueError: betas must be a pair (beta1, beta2)")
        return (optimizer_float(source[0], context), optimizer_float(source[1], context))
    var input = py_item(context, nk[].k_input_context)
    classical_connect_numpy(input)
    var value = classical_input(source, input)
    var pins = OptimizerInputOwners()
    var view = TensorView()
    if not tensor_view(value, view, pins):
        raise Error("TypeError: betas require native numeric storage")
    optimizer_beta_shape(len(view.shape), optimizer_state_count(view))
    var kind = classical_cast_kind(value, view.itemsize, "betas", input)
    var native = not view.fmt.startswith(">") and not view.fmt.startswith("!")
    var stride = view.strides[0] if view.has_strides else view.itemsize
    var first = _pd_double(pd_raw_at(view.addr, 0, view.itemsize, native), kind, view.itemsize)
    var second = _pd_double(pd_raw_at(view.addr, stride, view.itemsize, native), kind, view.itemsize)
    _ = pins
    _ = value
    return (first, second)


def optimizer_profile(context: PythonObject) raises:
    # the process default, resolved as `_numeric_profile.require_training`
    # resolves it: a recorded default, else the environment, else fp32_v1
    var nk = optimizer_keys()
    var profile = py_get(py_item(context, nk[].k_profile_defaults), nk[].k_training)
    if profile is None:
        var env = String(getenv("MOJOLEARN_NUMERIC_PROFILE", "")).strip()
        profile = PythonObject(String(env) if env.byte_length() > 0 else String("fp32_v1"))
    opt_scalar(profile, context)
    optimizer_profile_available(optimizer_truth(py_item(py_item(context, nk[].k_profiles)[profile], nk[].k_training), context), String(profile))


def optimizer_constructor_start(kind: Int, betas_value: PythonObject,
    momentum: PythonObject, dampening: PythonObject, nesterov: PythonObject,
    accumulation_value: PythonObject, maximize: PythonObject, context: PythonObject) raises -> OptimizerConstructorStart:
    var accumulation = optimizer_integer(accumulation_value, context)
    optimizer_validate_accumulation(kind, accumulation)
    var nk = optimizer_keys()
    if py_type(maximize) is not py_item(context, nk[].k_scalar_types)[1]:
        var where = String("SGD") if kind == 0 else (String("Adam") if kind == 1 else String("AdamW"))
        raise Error("TypeError: mojolearn." + where + ": maximize must be a bool")
    var use_nesterov = False
    if kind == 0:
        use_nesterov = optimizer_truth(nesterov, context)
        if use_nesterov:
            # Preserve the original admitted scalar conversion and short-circuit:
            # zero momentum refuses before even decoding dampening or lr.
            optimizer_validate_nesterov_momentum(Float64(py=momentum))
            optimizer_validate_nesterov_dampening(Float64(py=dampening))
    var betas = (Float64(0.9), Float64(0.999))
    if kind != 0:
        betas = optimizer_betas(betas_value, context)
    return OptimizerConstructorStart(kind, accumulation, betas[0], betas[1], use_nesterov,
        maximize is PythonObject(True))


def optimizer_configuration(start: OptimizerConstructorStart, lr: PythonObject, eps: PythonObject,
    weight_decay: PythonObject, momentum: PythonObject, dampening: PythonObject,
    context: PythonObject) raises -> OptimizerConstructorConfig:
    var kind = start.kind
    var decoded_lr = optimizer_float(lr, context)
    var decoded_decay = optimizer_float(weight_decay, context)
    var decoded_momentum = Float64(0)
    var decoded_dampening = Float64(0)
    var decoded_eps = Float64(1e-8)
    if kind == 0:
        decoded_momentum = optimizer_float(momentum, context)
        decoded_dampening = optimizer_float(dampening, context)
    else:
        decoded_eps = optimizer_float(eps, context)
    var config = OptimizerConstructorConfig(kind, start.accumulation, decoded_lr, start.beta1, start.beta2,
        decoded_eps, decoded_decay, decoded_momentum, decoded_dampening, start.nesterov, start.maximize)
    return config


def optimizer_initialize(owner: PythonObject, registry: OptimizerRegistryPrepared, config: OptimizerConstructorConfig,
    mode: PythonObject, schedule: PythonObject, resident: PythonObject, device: PythonObject, host: Bool,
    context: PythonObject) raises -> PythonObject:
    optimizer_owner_idle(owner)
    var fields = py_fields(owner)
    # PACK publishes public scalar mirrors only after typed configuration
    # validation. Params/schedule/mode/resident retain their boundary owners.
    # Pristine: the moments are lazy (MOJOLEARN_OPT_LAZY_MOMENTS_OFF deleted 2026-11-24, lane ab2-nseq).
    var owner_state = optimizer_owner_new(owner, config.kind, True)
    var construction = OptimizerOwnerLease(owner, owner_state)
    var nk = optimizer_keys()
    py_setitem(fields, nk[].k__native_optimizer_owner, owner_state)
    py_setitem(fields, nk[].k_numeric_mode, mode)
    py_setitem(fields, nk[].k__resident_wanted, resident)
    # lane vdf-seq: the public `device` word and its route, resolved once here
    # (True: the CPU identical training image beside a GPU set)
    py_setitem(fields, nk[].k_device, device)
    py_setitem(fields, nk[].k__host_device, PythonObject(host))
    py_setitem(fields, nk[].k_lr_schedule, schedule)
    py_setitem(fields, nk[].k_accumulation_steps, PythonObject(config.accumulation))
    optimizer_publish_registry(owner, registry)
    py_setitem(fields, nk[].k_t, PythonObject(0))
    for name in ["packed_", "total_norm_", "clip_coef_"]:  # driver-loop: over three optional result words
        py_setitem(fields, PythonObject(name), PythonObject(None))
    py_setitem(fields, nk[].k_lr, PythonObject(config.lr))
    py_setitem(fields, nk[].k_weight_decay, PythonObject(config.weight_decay))
    py_setitem(fields, nk[].k_maximize, PythonObject(config.maximize))
    if config.kind == 0:
        py_setitem(fields, nk[].k_momentum, PythonObject(config.momentum))
        py_setitem(fields, nk[].k_dampening, PythonObject(config.dampening))
        py_setitem(fields, nk[].k_nesterov, PythonObject(config.nesterov))
    else:
        py_setitem(fields, nk[].k_betas, Python.tuple(config.beta1, config.beta2))
        py_setitem(fields, nk[].k_eps, PythonObject(config.eps))
    _ = construction
    return PythonObject(None)


def optimizer_construct(owner: PythonObject, values: PythonObject, mode: PythonObject, context: PythonObject) raises -> PythonObject:
    # values: params, kind, lr, betas, eps, weight_decay, momentum,
    # dampening, nesterov, schedule, accumulation_steps, maximize, resident, kwargs,
    # and (lane vdf-seq) device.
    var kind = optimizer_integer(values[1], context)
    var where = String("SGD") if kind == 0 else (String("Adam") if kind == 1 else String("AdamW"))
    opt_schedule(values[9], context)
    for i in [2, 4, 5, 6, 7, 8, 10, 11, 12]:  # driver-loop: over scalar constructor configuration words
        opt_scalar(values[i], context)
    opt_value(values[0], context)
    opt_value(values[13], context)
    refuse_kwargs(values[13], where)
    var device = values[14] if len(values) == 15 else PythonObject(None)
    var host = optimizer_device_host(device, values[0], mode, where, context)
    optimizer_profile(context)
    var start = optimizer_constructor_start(kind, values[3], values[6], values[7], values[8], values[10], values[11], context)
    opt_scalar(mode, context)
    var registry = optimizer_prepare_registry(values[0], start.name())
    var config = optimizer_configuration(start, values[2], values[4], values[5], values[6], values[7], context)
    return optimizer_initialize(owner, registry, config, mode, values[9], values[12], device, host, context)


def optimizer_device_host(device: PythonObject, params: PythonObject, mode: PythonObject, where: String,
                          context: PythonObject) raises -> Bool:
    """UNPACK of an SGD / Adam / AdamW `device` (lane vdf-seq, 2026-11-24),
    once per construction or set_params, before any work: True for an
    explicit "cpu" beside a GPU set, whose owner then runs the CPU identical
    training image (`training_entries_cpu`, connected here once, refused here
    for a non-identical mode or a missing image). Parameters that already
    live on the device (`TrainDeviceArray`) disagree with it and are refused
    in words; nothing is copied across. None: one identity compare, False."""
    if device is None:
        return False
    var fdev = function_device_admit(device, "mojolearn." + where, True, True, context, native_loaded_module("builtins"))
    if not fdev.host_beside:
        return False
    var nk = optimizer_keys()
    var device_class = py_item(context, nk[].k_device_class)
    var b = native_loaded_module("builtins")
    if py_type(params) is b.list or py_type(params) is b.tuple:
        for i in range(len(params)):  # driver-loop: over the public parameter buffers (refusal before any work)
            if py_type(params[i]) is device_class:
                raise Error("ValueError: mojolearn." + where + " was given device='cpu' but parameter " + String(i)
                    + " is a TrainDeviceArray, which lives on the GPU; copy it to the host first, or leave device=None")
    elif py_type(params) is device_class:
        raise Error("ValueError: mojolearn." + where + " was given device='cpu' but its parameter is a "
            + "TrainDeviceArray, which lives on the GPU; copy it to the host first, or leave device=None")
    _ = optimizer_entries_cpu(mode, context)
    return True


def optimizer_state_array(source: PythonObject, integer: Bool, context: PythonObject) raises -> PythonObject:
    opt_value(source, context)
    var nk = optimizer_keys()
    var input = py_item(context, nk[].k_input_context)
    classical_connect_numpy(input)
    var value = classical_input(source, input)
    var pins = OptimizerInputOwners()
    var view = TensorView()
    if not tensor_view(value, view, pins):
        raise Error("TypeError: optimizer checkpoint requires native numeric storage")
    var count = optimizer_state_count(view)
    optimizer_checkpoint_input(len(view.shape), count)
    var kind = classical_cast_kind(value, view.itemsize, "optimizer checkpoint", input)
    var native = not view.fmt.startswith(">") and not view.fmt.startswith("!")
    var stride = view.strides[0] if view.has_strides else view.itemsize
    var result = new_array([count], "<i4" if integer else "<f4")
    var destination = TensorView()
    if not tensor_view(result, destination, pins):
        raise Error("TypeError: optimizer checkpoint output lacks native storage")
    with GILReleased(Python()):
        optimizer_state_cast(view.addr, stride, count, view.itemsize, kind, native,
            MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=destination.addr), integer)
    _ = pins
    _ = value
    return result



def optimizer_params(owner: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(owner)
    var kind = owner_kind(owner)
    var names = optimizer_parameter_names(kind)
    var result = Python.dict()
    var nk = optimizer_keys()
    for i in range(len(names)):  # driver-loop: pack fixed public constructor parameter names
        var name = PythonObject(names[i])
        if names[i] == "device":
            # an owner restored from state saved before `device` existed has none
            py_setitem(result, name, py_get(fields, nk[].k_device))
            continue
        py_setitem(result, name, py_item(fields, nk[].k__resident_wanted) if names[i] == "resident" else fields[name])
    return result


def optimizer_parameter_value(updates: PythonObject, fields: PythonObject, name: String) raises -> PythonObject:
    if py_contains(updates, PythonObject(name)):
        return updates[PythonObject(name)]
    var nk = optimizer_keys()
    return py_item(fields, nk[].k__resident_wanted) if name == "resident" else fields[PythonObject(name)]


def optimizer_set_params(owner: PythonObject, updates: PythonObject, context: PythonObject) raises -> PythonObject:
    _ = bump_model_generation(owner)
    var nk = optimizer_keys()
    if py_type(updates) is not py_item(context, nk[].k_container_types)[2]:
        raise Error("TypeError: optimizer parameter updates require native keyword arguments")
    if len(updates) == 0:
        return owner
    var fields = py_fields(owner)
    var kind = owner_kind(owner)
    var table = cpython_table()
    var door = OPT_DOOR.get_or_create_ptr()
    var position = List[Int](length=1, fill=0)
    var pair = List[PyObjectPtr](length=2, fill=PyObjectPtr())
    # Exact dictionary iteration is UNPACK; no callbacks or Python key list.
    while table[].PyDict_Next(updates._obj_ptr, Int(position.unsafe_ptr()),
            Int(pair.unsafe_ptr()), Int(pair.unsafe_ptr() + 1)):  # driver-loop: external keyword arguments
        var key = PythonObject(from_borrowed=pair[0])
        var value = PythonObject(from_borrowed=pair[1])
        if py_type(key) is not py_item(context, nk[].k_scalar_types)[5]:
            raise Error("TypeError: optimizer parameter names must be native strings")
        var name = String(key)
        if name == "lr_schedule":
            opt_schedule(value, context)
        else:
            opt_value(value, context)
        optimizer_parameter_update(kind, name)
    optimizer_owner_admit_replacement(owner, fields, context)
    var params = optimizer_parameter_value(updates, fields, "params")
    var mode = optimizer_parameter_value(updates, fields, "numeric_mode")
    var schedule = optimizer_parameter_value(updates, fields, "lr_schedule")
    var resident = optimizer_parameter_value(updates, fields, "resident")
    opt_scalar(resident, context)
    var device = updates[PythonObject("device")] if py_contains(updates, PythonObject("device")) else py_get(fields, nk[].k_device)
    var where = String("SGD") if kind == 0 else (String("Adam") if kind == 1 else String("AdamW"))
    var host = optimizer_device_host(device, params, mode, where, context)
    optimizer_profile(context)
    var momentum = optimizer_parameter_value(updates, fields, "momentum") if kind == 0 else PythonObject(None)
    var dampening = optimizer_parameter_value(updates, fields, "dampening") if kind == 0 else PythonObject(None)
    var start = optimizer_constructor_start(kind,
        PythonObject(None) if kind == 0 else optimizer_parameter_value(updates, fields, "betas"),
        momentum, dampening,
        optimizer_parameter_value(updates, fields, "nesterov") if kind == 0 else PythonObject(None),
        optimizer_parameter_value(updates, fields, "accumulation_steps"),
        optimizer_parameter_value(updates, fields, "maximize"), context)
    opt_scalar(mode, context)
    var registry = optimizer_prepare_registry(params, start.name())
    var config = optimizer_configuration(start,
        optimizer_parameter_value(updates, fields, "lr"),
        PythonObject(None) if kind == 0 else optimizer_parameter_value(updates, fields, "eps"),
        optimizer_parameter_value(updates, fields, "weight_decay"), momentum, dampening, context)
    # This retained dictionary is a PACK rollback owner, never algorithm
    # configuration. No temporary Python optimizer object/finalizer is made.
    var saved_pointer = table[].PyDict_Copy(fields._obj_ptr)
    if not saved_pointer:
        var failure = stash_python_error()
        raise failure^
    var saved = PythonObject(from_owned=saved_pointer)
    var lease = OptimizerOwnerLease(owner)
    door[].dict_clear(fields._obj_ptr)
    try:
        _ = optimizer_initialize(owner, registry, config, mode, schedule, resident, device, host, context)
        var replacement = OptimizerOwnerLease(owner)
        optimizer_owner_close(lease.value, owner)
        _ = replacement
    except error:
        door[].dict_clear(fields._obj_ptr)
        if door[].dict_update(fields._obj_ptr, saved._obj_ptr) != 0:
            var failure = stash_python_error()
            raise failure^
        raise error^
    _ = lease
    return owner


comptime OPTIMIZER_ROUTE_NONE = 0
comptime OPTIMIZER_ROUTE_SEQDEV = 1
comptime OPTIMIZER_ROUTE_EMBLAYER = 2
comptime OPTIMIZER_ROUTE_SEQLN = 3
comptime OPTIMIZER_ROUTE_SEQMOE = 4
comptime OPTIMIZER_ROUTE_SEQ = 5
comptime OPTIMIZER_ROUTE_DEVICE = 6
comptime OPTIMIZER_ROUTE_SCHEDULE = 7


def optimizer_route(operation: Int, name: PythonObject) raises -> Int:
    """The owner of an operation: a range test of its code (the codes are
    numbered by owner). A name the table lacks (-1) is copied once and routed
    by the router's old prefix tests, in their old order, so it reaches the
    owner whose unchanged unknown-operation error it raised before."""
    if operation >= 0:
        if operation >= OPTIMIZER_OPN_SEQDEV_ARRAY and operation <= OPTIMIZER_OPN_SEQDEV_WRAP:
            return OPTIMIZER_ROUTE_SEQDEV
        if operation >= OPTIMIZER_OPN_EMBLAYER_INIT and operation <= OPTIMIZER_OPN_EMBLAYER_LOAD:
            return OPTIMIZER_ROUTE_EMBLAYER
        if operation >= OPTIMIZER_OPN_SEQLN_INITIALIZE and operation <= OPTIMIZER_OPN_SEQLN_LAYER_BACKWARD:
            return OPTIMIZER_ROUTE_SEQLN
        if operation >= OPTIMIZER_OPN_SEQMOE_INITIALIZE and operation <= OPTIMIZER_OPN_SEQMOE_TO_DEVICE:
            return OPTIMIZER_ROUTE_SEQMOE
        if operation >= OPTIMIZER_OPN_SEQ_CONSTRUCT and operation <= OPTIMIZER_OPN_SEQ_FINALIZE:
            return OPTIMIZER_ROUTE_SEQ
        if operation >= OPTIMIZER_OPN_DEVICE_CONSTRUCT and operation <= OPTIMIZER_OPN_DEVICE_HOST:
            return OPTIMIZER_ROUTE_DEVICE
        if operation >= OPTIMIZER_OPN_SCHEDULE_CONSTRUCT and operation <= OPTIMIZER_OPN_SCHEDULE_BITS:
            return OPTIMIZER_ROUTE_SCHEDULE
        return OPTIMIZER_ROUTE_NONE
    var text = String(py=name)
    if text.startswith("seqdev_"):
        return OPTIMIZER_ROUTE_SEQDEV
    if text.startswith("emblayer_"):
        return OPTIMIZER_ROUTE_EMBLAYER
    if text.startswith("seqln_"):
        return OPTIMIZER_ROUTE_SEQLN
    if text.startswith("seqmoe_"):
        return OPTIMIZER_ROUTE_SEQMOE
    if text.startswith("seq_"):
        return OPTIMIZER_ROUTE_SEQ
    if text.startswith("device_"):
        return OPTIMIZER_ROUTE_DEVICE
    if text.startswith("schedule_"):
        return OPTIMIZER_ROUTE_SCHEDULE
    return OPTIMIZER_ROUTE_NONE


def optimizer_native(operation: Int, name: PythonObject, owner: PythonObject, values: PythonObject, mode: PythonObject, context: PythonObject) raises -> PythonObject:
    # bw2-optim-training (2026-11-20): `operation` is the code of
    # bindings/optimizer_keys_unpack.mojo (-1 for a name the table lacks) and
    # `name` the caller's own str: the training owners read their compiled
    # entry with it and build their refusal text from it. The old String copy
    # and its chain of prefix tests and compares are gone; `optimizer_route` is
    # a few integer compares. bw2-router-seq (2026-11-20): the sequence device,
    # Embedding, LayerNorm, MoE and sequence optimizer owners take the code too
    # (no String per call); a name the table lacks (-1) runs each owner's chain
    # to its unchanged unknown-operation error (Embedding's text names the
    # operation, so its refusal is raised here from the name).
    var route = optimizer_route(operation, name)
    if operation == OPTIMIZER_OPN_TRAINING_NUMERIC_MODE or operation == OPTIMIZER_OPN_TRAINING_VENDOR:
        # Metadata needs only its connected native entries, not optimizer
        # object admission, buffer initialization or runtime binding loaders.
        return training_metadata(operation, mode, context)
    # lane fb2-lr (2026-11-17): the per-step `lr_at` / `bits_at` no longer
    # reach this router; they are the schedule's bound native block methods
    # (bindings/lr_schedule_block_door.mojo). lane fb-optim's shortcut past
    # `opt_guards` for them (03ef3ecee3) is removed with it.
    opt_guards(context)
    if operation == OPTIMIZER_OPN_SEQDEV_FINALIZE:
        return sequence_device_finalize(owner, values, context)
    # opt_guards just ran on this context; when it is the door's own
    # (training._OPT_CONTEXT, as connected) the door does not walk it again.
    door_ready(context)
    var nk = optimizer_keys()
    if py_type(values) is not py_item(context, nk[].k_container_types)[1]:
        raise Error("TypeError: optimizer requires native argument tuples")
    # lane vdf-linalg: a public training function's `device` (the last item of
    # its tuple), read once here before any work
    # (bindings/training_function_device_unpack.mojo). None: these integer
    # compares and one identity compare; the entries below are today's.
    var fdev = function_device_default()
    if training_function_op(operation):
        var item = values[len(values) - 1]
        if item is not None:
            fdev = training_function_device(operation, item, values, context)
    if route == OPTIMIZER_ROUTE_SEQDEV:
        return sequence_device_operation(operation, owner, values, mode, context)
    if operation == OPTIMIZER_OPN_RESIDUAL_DROPOUT or operation == OPTIMIZER_OPN_RESIDUAL_DROPOUT_BACKWARD:
        # the explicit residual layer runs in identical mode only
        if fdev.host_beside:
            return training_residual_dropout(operation, values, training_entries_cpu(PythonObject("identical"), context), context)
        return training_residual_dropout(operation, values, optimizer_entries_identical(context), context)
    if route == OPTIMIZER_ROUTE_EMBLAYER:
        # mojolearn.Embedding's forward and load (bindings/embedding_layer_native.mojo)
        if operation < 0:
            emblayer_unknown_operation(String(py=name))
        return emblayer_operation(operation, owner, values, context)
    if route == OPTIMIZER_ROUTE_SEQLN:
        return sequence_layer_norm(operation, owner, values, mode, context)
    if route == OPTIMIZER_ROUTE_SEQMOE:
        return sequence_moe_operation(operation, owner, values, mode, context)
    if route == OPTIMIZER_ROUTE_SEQ:
        return sequence_optimizer_operation(operation, owner, values, mode, context)
    if route == OPTIMIZER_ROUTE_DEVICE:
        var selected = optimizer_entries(mode, context) if operation == OPTIMIZER_OPN_DEVICE_UPLOAD and py_type(values[0]) is not py_item(context, nk[].k_device_class) else PythonObject(None)
        return device_operation(operation, owner, values, context, selected)
    if route == OPTIMIZER_ROUTE_SCHEDULE:
        return schedule_run(operation, owner, values, context)
    if operation == OPTIMIZER_OPN_RESTORE_OWNER:
        return optimizer_restore_state(owner, values[0], context)
    if operation == OPTIMIZER_OPN_CONSTRUCT:
        if len(values) != 14 and len(values) != 15:
            raise Error("TypeError: invalid optimizer constructor envelope")
        opt_owner(owner, context)
        return optimizer_construct(owner, values, mode, context)
    var selected_mode = mode
    var host_owner = False
    if owner is not None:
        if operation == OPTIMIZER_OPN_STEP or operation == OPTIMIZER_OPN_STATE or operation == OPTIMIZER_OPN_ZERO:
            # These go straight to a compiled door entry whose first step is
            # the full `opt_owner` walk (the door is a public module entry,
            # so it must admit by itself; the two binaries share no trusted
            # word). Here only what this router reads is admitted: the exact
            # class, string field names and `numeric_mode`.
            selected_mode = opt_owner_mode(owner, context)
        else:
            selected_mode = opt_owner_word(owner, context, False)
        # lane vdf-seq: an SGD / Adam / AdamW constructed with device="cpu"
        # beside a GPU set runs the CPU identical training image (resolved at
        # construction). The admission above returns `numeric_mode`, which
        # this router used to read a second time; that read is now this one,
        # so a device=None owner reads no more slots than before.
        host_owner = py_get(py_fields(owner), nk[].k__host_device) is PythonObject(True)
    if operation == OPTIMIZER_OPN_GET_PARAMS:
        opt_scalar(values[0], context)
        return optimizer_params(owner, context)
    if operation == OPTIMIZER_OPN_SET_PARAMS:
        return optimizer_set_params(owner, values[0], context)
    if operation == OPTIMIZER_OPN_EXPORT_OWNER:
        return optimizer_export_state(owner)
    var entries = optimizer_entries_cpu(selected_mode, context) if (fdev.host_beside or host_owner) else optimizer_entries(selected_mode, context)
    if operation == OPTIMIZER_OPN_LINEAR_FORWARD or operation == OPTIMIZER_OPN_LINEAR_BACKWARD:
        return training_linear_operation(operation, name, values, entries, context)
    if operation == OPTIMIZER_OPN_RMS_NORM_FORWARD or operation == OPTIMIZER_OPN_RMS_NORM_BACKWARD:
        return training_rms_operation(operation, name, values, entries, context)
    if operation == OPTIMIZER_OPN_EMBEDDING_FORWARD or operation == OPTIMIZER_OPN_EMBEDDING_BACKWARD:
        return training_embedding_operation(operation, name, values, entries, context)
    if operation == OPTIMIZER_OPN_CROSS_ENTROPY:
        return training_cross_entropy(values, entries, context)
    if operation == OPTIMIZER_OPN_CHUNKED_LM_HEAD_LOSS:
        return training_chunked_lm_head(values, entries, context)
    if operation == OPTIMIZER_OPN_SAMBA_HEAD_LOSS:
        return training_samba_head_loss(values, entries, context)
    if operation == OPTIMIZER_OPN_MLP_INFERENCE_SESSIONS:
        # lane vdf-linalg3: TrainDeviceArray sessions / weights run the device
        # entry in place (bindings/training_grads_device_unpack.mojo)
        if grads_device_input(values[1], False, context) or grads_device_input(values[0], False, context):
            return mlp_sessions_device(values, entries, context)
        return training_mlp_sessions(values, entries)
    if operation == OPTIMIZER_OPN_BINDING:
        return py_item(entries, nk[].k_module)
    var provider = optimizer_provider_entries(entries)
    var module = py_item(entries, nk[].k_module)
    if operation == OPTIMIZER_OPN_RAW_IO:
        opt_value(values, context)
        return door_io(module, provider, owner, Int(py=values[0]), values[1], String(py=values[2]), Int(py=values[3]))
    if operation == OPTIMIZER_OPN_PARAM_DEVICE:
        var fields = py_fields(owner)
        var owner_state = optimizer_owner_value(owner)
        var record = owner_state.downcast_value_ptr[OptimizerOwner]()
        if not record[].control.parameters_device:
            raise Error("RuntimeError: optimizer.param_device: call params_to_device() first")
        opt_scalar(values[0], context)
        var j = optimizer_integer(values[0], context)
        var params = py_item(fields, nk[].k_params)
        if j < 0:
            j += len(params)
        if j < 0 or j >= len(params):
            raise Error("ValueError: optimizer parameter index is out of range")
        var view = TensorView()
        _ = tensor_view(params[j], view)
        var native = device_entries(py_item(entries, nk[].k_module), context)
        var result = device_new(PythonObject(None), PythonObject(None), view.shape, PythonObject(None), context)
        var device_provider = optimizer_provider_entries(native)
        var device_driver = optimizer_driver(device_provider)
        var binding = py_item(entries, nk[].k_module)
        var handle = optimizer_io(module, provider, owner, 5, PythonObject(None), "params", j)
        var published = device_adopt(result, binding, handle, device_driver)
        _ = device_provider
        _ = owner_state
        return published
    if operation == OPTIMIZER_OPN_STEP:
        opt_scalar(values[2], context)
        if host_owner:
            optimizer_refuse_device_grads(values[0], context)
        return door_step(module, provider, owner, values[0], values[1], Int(py=values[2]), values[3])
    if operation == OPTIMIZER_OPN_STATE:
        opt_scalar(values[0], context)
        return door_state(module, provider, owner, Int(py=values[0]))
    if operation == OPTIMIZER_OPN_STATE_SET:
        opt_scalar(values[0], context)
        opt_value(values[1], context)
        return door_state_set(module, provider, owner, Int(py=values[0]), values[1])
    if operation == OPTIMIZER_OPN_LOAD:
        var state = values[0]
        opt_value(state, context)
        if not opt_exact(py_type(state), py_item(context, nk[].k_mapping_types)):
            raise Error("TypeError: optimizer checkpoint must be a native dictionary")
        opt_scalar(py_item(state, nk[].k_t), context)
        var t = optimizer_integer(py_item(state, nk[].k_t), context)
        var m = optimizer_state_array(py_item(state, nk[].k_exp_avg), False, context)
        var v = optimizer_state_array(py_item(state, nk[].k_exp_avg_sq), False, context)
        var initialized = optimizer_state_array(py_item(state, nk[].k_buf_initialized), True, context)
        # Conversion released the GIL: retain the previous door's fresh owner
        # admission before reading fields another thread could have rebound.
        opt_owner(owner, context)
        optimizer_load_buffers(module, provider, owner, m, v, initialized)
        var fields = py_fields(owner)
        py_setitem(fields, nk[].k_buf_initialized, initialized)
        py_setitem(fields, nk[].k_t, PythonObject(t))
        return PythonObject(None)
    if operation == OPTIMIZER_OPN_ZERO:
        return door_zero(owner, values[0])
    if operation == OPTIMIZER_OPN_RESIDENT:
        var owner_state = optimizer_owner_value(owner)
        var active = owner_state.downcast_value_ptr[OptimizerOwner]()[].control.resident()
        _ = owner_state
        return PythonObject(active)
    if operation == OPTIMIZER_OPN_IO:
        opt_scalar(values[0], context)
        var op = Int(py=values[0])
        var fields = py_fields(owner)
        var owner_state = optimizer_owner_value(owner)
        var record = owner_state.downcast_value_ptr[OptimizerOwner]()
        var tensors = values[1]
        if op == 1 or op == 2:
            tensors = py_item(fields, nk[].k_params)
        if op == 2 and not record[].control.parameters_device:
            _ = owner_state
            return py_item(fields, nk[].k_params)
        opt_scalar(values[2], context)
        opt_scalar(values[3], context)
        var result = optimizer_io(module, provider, owner, op, tensors, String(py=values[2]), Int(py=values[3]))
        if op == 0:
            _ = owner_state
            return PythonObject(result != 0)
        if op == 1 or op == 3:
            if op == 1:
                record[].control.parameters_device = True
            else:
                record[].control.gradients_device = True
            _ = owner_state
            return owner
        if op == 2 or op == 4:
            _ = owner_state
            return tensors
        _ = owner_state
        return optimizer_pack_io(provider, op, result)
    if operation == OPTIMIZER_OPN_CLIP:
        # lane vdf-linalg3: TrainDeviceArray gradients are clipped in place on
        # the device (bindings/training_grads_device_unpack.mojo); identity
        # compares on the host path
        if grads_device_input(values[0], False, context):
            return clip_device(values, entries, context)
        return door_clip(provider, values[0], values[1], values[2], values[3], values[4])
    if operation == OPTIMIZER_OPN_ACCUMULATE:
        if grads_device_input(values[0], True, context):
            return accumulate_device(values, entries, context)
        return door_accumulate(provider, values[0], values[1])
    if operation == OPTIMIZER_OPN_ALIGNED:
        opt_scalar(values, context)
        return PythonObject(microbatch_split_is_identical(optimizer_integer(values[0], context), optimizer_integer(values[1], context)))
    raise Error("TypeError: unknown optimizer native operation")


def optimizer_refuse_device_grads(grads: PythonObject, context: PythonObject) raises:
    """A device="cpu" optimizer's step refuses a gradient that lives on the
    GPU (`TrainDeviceArray`) in words before any work (lane vdf-seq)."""
    var nk = optimizer_keys()
    var device_class = py_item(context, nk[].k_device_class)
    var b = native_loaded_module("builtins")
    if py_type(grads) is device_class:
        raise Error("ValueError: an optimizer constructed with device='cpu' was given a TrainDeviceArray gradient, "
            + "which lives on the GPU; copy it to the host first")
    if py_type(grads) is b.list or py_type(grads) is b.tuple:
        for i in range(len(grads)):  # driver-loop: over the step's gradient buffers (device='cpu' owners only)
            if py_type(grads[i]) is device_class:
                raise Error("ValueError: an optimizer constructed with device='cpu' was given a TrainDeviceArray "
                    + "gradient (" + String(i) + "), which lives on the GPU; copy it to the host first")


def optimizer_native_c(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=packed)
        if len(args) != 5:
            raise Error("TypeError: optimizer_native expects five arguments")
        # One lookup of the caller's str in the held operation table (it was
        # copied into a String per call). During interpreter teardown an
        # unfilled table is not filled (a Python `__del__` reaches this entry:
        # `seqdev_finalize`, `seq_finalize`); no call ever filled it, so
        # nothing this router made is left to release.
        var nk = optimizer_keys_in_finalizer()
        if not nk[].ready:
            return PythonObject(None).steal_data()
        var name = args[0]
        return optimizer_native(py_op_code(nk[].ops, name), name, args[1], args[2], args[3], args[4]).steal_data()
    except error:
        return _optimizer_router_error(error)


def _optimizer_router_error(error: Error) -> PyObjectPtr:
    var message = String(error)
    if message.startswith("ImportError: "):
        return raise_python_exception(Error(message[byte=13:]), ExceptionType("PyExc_ImportError"))
    if message.startswith("AttributeError: "):
        return raise_python_exception(Error(message[byte=16:]), ExceptionType("PyExc_AttributeError"))
    if message.startswith("IndexError: "):
        return raise_python_exception(Error(message[byte=12:]), ExceptionType("PyExc_IndexError"))
    return optimizer_python_error(error)


def schedule_attribute_c(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    """A built-in schedule's `__setattr__` / `__delattr__` (fb6-smoke,
    2026-11-17). Installed at import as PyInstanceMethod(MethodType(this,
    context)) on `_LrTable`, so `sched.x = v` arrives as (context, sched, "x",
    v) and `del sched.x` as (context, sched, "x"): the same router operation
    the Python methods called, with no Python frame. Construction, unpickling
    and checkpoint restore write the instance dictionary directly and never
    come here; only a caller's own edit does (one rebuild per edit)."""
    try:
        var args = PythonObject(from_borrowed=packed)
        if len(args) == 4:
            return optimizer_native(OPTIMIZER_OPN_SCHEDULE_SETATTR, PythonObject(None), args[1], Python.tuple(args[2], args[3]), PythonObject(None), args[0]).steal_data()
        if len(args) == 3:
            return optimizer_native(OPTIMIZER_OPN_SCHEDULE_DELATTR, PythonObject(None), args[1], Python.tuple(args[2]), PythonObject(None), args[0]).steal_data()
        raise Error("TypeError: schedule attribute hook expects (name, value) or (name)")
    except error:
        return _optimizer_router_error(error)
