# SPDX-License-Identifier: Apache-2.0
# Native public-operation routing, constructors, state preparation and resident IO.
from bindings.embedding_layer_native import emblayer_operation
from std.os import getenv
from std.python import Python, PythonObject
from std.python._cpython import GILReleased, PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.sequence_layers_native import sequence_layers_operation
from bindings.training_extra_native import training_chunked_lm_head, training_mlp_sessions, training_residual_dropout
from bindings.samba_head_native import training_samba_head_loss
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem, py_contains
from bindings.py_typed_call import py_call_typed
from bindings.optimizer_public_words import optimizer_python_error, stash_python_error
from bindings.optimizer_admission import opt_guards, opt_owner, opt_owner_mode, opt_value, opt_scalar, opt_schedule, opt_exact
from bindings.optimizer_public_door import (
    door_ready, owner_kind, refuse_kwargs, OptimizerRegistryPrepared, optimizer_prepare_registry, optimizer_publish_registry, new_array, TensorView, tensor_view,
    door_step, door_io, optimizer_io, optimizer_pack_io, optimizer_load_buffers, optimizer_export_state, door_state, door_state_set, door_zero, door_clip, door_accumulate,
)
from bindings.optimizer_descriptor_boundary import optimizer_provider_entries, optimizer_native_mode, optimizer_driver
from core.linear_metadata import linear_mode_name
from bindings.optimizer_serialization_boundary import optimizer_restore_state
from bindings.optimizer_owner_boundary import OptimizerOwner, OptimizerOwnerLease, optimizer_owner_new, optimizer_owner_value, optimizer_owner_close, optimizer_owner_idle, optimizer_owner_admit_replacement
from training.checks.optimizer_contract import microbatch_split_is_identical
from training.optimizer_config import OptimizerConstructorStart, OptimizerConstructorConfig, optimizer_parameter_names, optimizer_parameter_update, optimizer_profile_available, optimizer_beta_shape, optimizer_validate_accumulation, optimizer_validate_nesterov_momentum, optimizer_validate_nesterov_dampening
from bindings.cpython_table import cpython_table
from bindings.optimizer_container_boundary import OptimizerInputOwners
from training.optimizer_state_plan import optimizer_checkpoint_input, optimizer_state_count
from bindings.optimizer_scalar_boundary import optimizer_float, optimizer_integer, optimizer_truth
from bindings.classical_attributes import classical_connect_numpy
from bindings.classical_arrays import classical_input, classical_cast_kind
from core.optimizer_state_impl import optimizer_state_cast
from core.classical_pandas_impl import pd_raw_at, _pd_double
from bindings.sequence_device_native import sequence_device_operation, sequence_device_finalize
from bindings.sequence_optimizer_native import sequence_optimizer_operation
from bindings.optimizer_schedule_native import schedule_operation
from bindings.optimizer_device_native import device_operation, device_new, device_entries, device_finalize, device_adopt
from bindings.training_linear_native import training_linear_operation
from bindings.training_rms_native import training_rms_operation
from bindings.training_embedding_native import training_embedding_operation
from bindings.training_cross_entropy_native import training_cross_entropy
from bindings.training_metadata_native import training_metadata
from bindings.native_model_records import bump_model_generation


def optimizer_entries(mode: PythonObject, context: PythonObject) raises -> PythonObject:
    var selected = mode
    if selected is None:
        selected = context["backend"]["_DEFAULT_MODE"]
    if py_type(selected) is not context["scalar_types"][5]:
        raise Error("ValueError: numeric_mode must be fast, identical or None")
    var word = String()
    try:
        word = linear_mode_name(String(selected))
    except:
        raise Error("ValueError: numeric_mode must be fast, identical or None")
    var entries = context["bindings"][PythonObject(word)]
    if py_type(entries) is context["scalar_types"][5]:
        raise Error("ImportError: " + String(entries))
    var provider = optimizer_provider_entries(entries)
    if optimizer_native_mode(provider) != (1 if word == "identical" else 0):
        raise Error("ImportError: optimizer binary numeric mode disagrees with the requested mode")
    return entries


def optimizer_betas(source: PythonObject, context: PythonObject) raises -> Tuple[Float64, Float64]:
    opt_value(source, context)
    if py_type(source) is context["container_types"][1] or py_type(source) is context["container_types"][0]:
        opt_scalar(source, context)
        if len(source) != 2:
            raise Error("ValueError: betas must be a pair (beta1, beta2)")
        return (optimizer_float(source[0], context), optimizer_float(source[1], context))
    var input = context["input_context"]
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
    var profile = py_get(context["profile_defaults"], "training")
    if profile is None:
        var env = String(getenv("MOJOLEARN_NUMERIC_PROFILE", "")).strip()
        profile = PythonObject(String(env) if env.byte_length() > 0 else String("fp32_v1"))
    opt_scalar(profile, context)
    optimizer_profile_available(optimizer_truth(context["profiles"][profile]["training"], context))


def optimizer_constructor_start(kind: Int, betas_value: PythonObject,
    momentum: PythonObject, dampening: PythonObject, nesterov: PythonObject,
    accumulation_value: PythonObject, maximize: PythonObject, context: PythonObject) raises -> OptimizerConstructorStart:
    var accumulation = optimizer_integer(accumulation_value, context)
    optimizer_validate_accumulation(kind, accumulation)
    if py_type(maximize) is not context["scalar_types"][1]:
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
    mode: PythonObject, schedule: PythonObject, resident: PythonObject, context: PythonObject) raises -> PythonObject:
    optimizer_owner_idle(owner)
    var fields = py_fields(owner)
    # PACK publishes public scalar mirrors only after typed configuration
    # validation. Params/schedule/mode/resident retain their boundary owners.
    var owner_state = optimizer_owner_new(owner, config.kind, not registry.plan.eager)
    var construction = OptimizerOwnerLease(owner, owner_state)
    py_setitem(fields, "_native_optimizer_owner", owner_state)
    py_setitem(fields, "numeric_mode", mode)
    py_setitem(fields, "_resident_wanted", resident)
    py_setitem(fields, "lr_schedule", schedule)
    py_setitem(fields, "accumulation_steps", PythonObject(config.accumulation))
    optimizer_publish_registry(owner, registry)
    py_setitem(fields, "t", PythonObject(0))
    for name in ["packed_", "total_norm_", "clip_coef_"]:  # driver-loop: over three optional result words
        py_setitem(fields, PythonObject(name), PythonObject(None))
    py_setitem(fields, "lr", PythonObject(config.lr))
    py_setitem(fields, "weight_decay", PythonObject(config.weight_decay))
    py_setitem(fields, "maximize", PythonObject(config.maximize))
    if config.kind == 0:
        py_setitem(fields, "momentum", PythonObject(config.momentum))
        py_setitem(fields, "dampening", PythonObject(config.dampening))
        py_setitem(fields, "nesterov", PythonObject(config.nesterov))
    else:
        py_setitem(fields, "betas", Python.tuple(config.beta1, config.beta2))
        py_setitem(fields, "eps", PythonObject(config.eps))
    _ = construction
    return PythonObject(None)


def optimizer_construct(owner: PythonObject, values: PythonObject, mode: PythonObject, context: PythonObject) raises -> PythonObject:
    # values: params, kind, lr, betas, eps, weight_decay, momentum,
    # dampening, nesterov, schedule, accumulation_steps, maximize, resident, kwargs.
    var kind = optimizer_integer(values[1], context)
    var where = String("SGD") if kind == 0 else (String("Adam") if kind == 1 else String("AdamW"))
    opt_schedule(values[9], context)
    for i in [2, 4, 5, 6, 7, 8, 10, 11, 12]:  # driver-loop: over scalar constructor configuration words
        opt_scalar(values[i], context)
    opt_value(values[0], context)
    opt_value(values[13], context)
    refuse_kwargs(values[13], where)
    optimizer_profile(context)
    var start = optimizer_constructor_start(kind, values[3], values[6], values[7], values[8], values[10], values[11], context)
    opt_scalar(mode, context)
    var registry = optimizer_prepare_registry(values[0], start.name())
    var config = optimizer_configuration(start, values[2], values[4], values[5], values[6], values[7], context)
    return optimizer_initialize(owner, registry, config, mode, values[9], values[12], context)


def optimizer_state_array(source: PythonObject, integer: Bool, context: PythonObject) raises -> PythonObject:
    opt_value(source, context)
    var input = context["input_context"]
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
    for i in range(len(names)):  # driver-loop: pack fixed public constructor parameter names
        var name = PythonObject(names[i])
        py_setitem(result, name, fields["_resident_wanted"] if names[i] == "resident" else fields[name])
    return result


def optimizer_parameter_value(updates: PythonObject, fields: PythonObject, name: String) raises -> PythonObject:
    if py_contains(updates, PythonObject(name)):
        return updates[PythonObject(name)]
    return fields["_resident_wanted"] if name == "resident" else fields[PythonObject(name)]


def optimizer_set_params(owner: PythonObject, updates: PythonObject, context: PythonObject) raises -> PythonObject:
    _ = bump_model_generation(owner)
    if py_type(updates) is not context["container_types"][2]:
        raise Error("TypeError: optimizer parameter updates require native keyword arguments")
    if len(updates) == 0:
        return owner
    var fields = py_fields(owner)
    var kind = owner_kind(owner)
    var table = cpython_table()
    var position = List[Int](length=1, fill=0)
    var pair = List[PyObjectPtr](length=2, fill=PyObjectPtr())
    # Exact dictionary iteration is UNPACK; no callbacks or Python key list.
    while table[].PyDict_Next(updates._obj_ptr, Int(position.unsafe_ptr()),
            Int(pair.unsafe_ptr()), Int(pair.unsafe_ptr() + 1)):  # driver-loop: external keyword arguments
        var key = PythonObject(from_borrowed=pair[0])
        var value = PythonObject(from_borrowed=pair[1])
        if py_type(key) is not context["scalar_types"][5]:
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
    table[].PyDict_Clear(fields._obj_ptr)
    try:
        _ = optimizer_initialize(owner, registry, config, mode, schedule, resident, context)
        var replacement = OptimizerOwnerLease(owner)
        optimizer_owner_close(lease.value, owner)
        _ = replacement
    except error:
        table[].PyDict_Clear(fields._obj_ptr)
        if table[].PyDict_Update(fields._obj_ptr, saved._obj_ptr) != 0:
            var failure = stash_python_error()
            raise failure^
        raise error^
    _ = lease
    return owner


def optimizer_native(operation: String, owner: PythonObject, values: PythonObject, mode: PythonObject, context: PythonObject) raises -> PythonObject:
    if operation == "training_numeric_mode" or operation == "training_vendor":
        # Metadata needs only its connected native entries, not optimizer
        # object admission, buffer initialization or runtime binding loaders.
        return training_metadata(operation, mode, context)
    if operation == "device_finalize":
        return device_finalize(owner, context)
    opt_guards(context)
    if operation == "seqdev_finalize":
        return sequence_device_finalize(owner, values, context)
    # opt_guards just ran on this context; when it is the door's own
    # (training._OPT_CONTEXT, as connected) the door does not walk it again.
    door_ready(context)
    if py_type(values) is not context["container_types"][1]:
        raise Error("TypeError: optimizer requires native argument tuples")
    if operation.startswith("seqdev_"):
        return sequence_device_operation(operation, owner, values, mode, context)
    if operation == "residual_dropout" or operation == "residual_dropout_backward":
        # the explicit residual layer runs in identical mode only
        return training_residual_dropout(operation, values, optimizer_entries(PythonObject("identical"), context), context)
    if operation.startswith("emblayer_"):
        # mojolearn.Embedding's forward and load (bindings/embedding_layer_native.mojo)
        return emblayer_operation(operation, owner, values, context)
    if operation.startswith("seqln_") or operation.startswith("seqmoe_"):
        return sequence_layers_operation(operation, owner, values, mode, context)
    if operation.startswith("seq_"):
        return sequence_optimizer_operation(operation, owner, values, mode, context)
    if operation.startswith("device_"):
        var selected = optimizer_entries(mode, context) if operation == "device_upload" and py_type(values[0]) is not context["device_class"] else PythonObject(None)
        return device_operation(operation, owner, values, context, selected)
    if operation.startswith("schedule_"):
        return schedule_operation(operation, owner, values, context)
    if operation == "restore_owner":
        return optimizer_restore_state(owner, values[0], context)
    if operation == "construct":
        if len(values) != 14:
            raise Error("TypeError: invalid optimizer constructor envelope")
        opt_owner(owner, context)
        return optimizer_construct(owner, values, mode, context)
    var selected_mode = mode
    if owner is not None:
        if operation == "step" or operation == "state" or operation == "zero":
            # These go straight to a compiled door entry whose first step is
            # the full `opt_owner` walk (the door is a public module entry,
            # so it must admit by itself; the two binaries share no trusted
            # word). Here only what this router reads is admitted: the exact
            # class, string field names and `numeric_mode`.
            opt_owner_mode(owner, context)
        else:
            opt_owner(owner, context)
        selected_mode = py_get(py_fields(owner), "numeric_mode")
    if operation == "get_params":
        opt_scalar(values[0], context)
        return optimizer_params(owner, context)
    if operation == "set_params":
        return optimizer_set_params(owner, values[0], context)
    if operation == "export_owner":
        return optimizer_export_state(owner)
    var entries = optimizer_entries(selected_mode, context)
    if operation == "linear_forward" or operation == "linear_backward":
        return training_linear_operation(operation, values, entries, context)
    if operation == "rms_norm_forward" or operation == "rms_norm_backward":
        return training_rms_operation(operation, values, entries, context)
    if operation == "embedding_forward" or operation == "embedding_backward":
        return training_embedding_operation(operation, values, entries, context)
    if operation == "cross_entropy":
        return training_cross_entropy(values, entries, context)
    if operation == "chunked_lm_head_loss":
        return training_chunked_lm_head(values, entries, context)
    if operation == "samba_head_loss":
        return training_samba_head_loss(values, entries, context)
    if operation == "mlp_inference_sessions":
        return training_mlp_sessions(values, entries)
    if operation == "binding":
        return entries["module"]
    var provider = optimizer_provider_entries(entries)
    var module = entries["module"]
    if operation == "raw_io":
        opt_value(values, context)
        return door_io(module, provider, owner, Int(py=values[0]), values[1], String(py=values[2]), Int(py=values[3]))
    if operation == "param_device":
        var fields = py_fields(owner)
        var owner_state = optimizer_owner_value(owner)
        var record = owner_state.downcast_value_ptr[OptimizerOwner]()
        if not record[].control.parameters_device:
            raise Error("RuntimeError: optimizer.param_device: call params_to_device() first")
        opt_scalar(values[0], context)
        var j = optimizer_integer(values[0], context)
        var params = fields["params"]
        if j < 0:
            j += len(params)
        if j < 0 or j >= len(params):
            raise Error("ValueError: optimizer parameter index is out of range")
        var view = TensorView()
        _ = tensor_view(params[j], view)
        var native = device_entries(entries["module"], context)
        var result = device_new(PythonObject(None), PythonObject(None), view.shape, PythonObject(None), context)
        var device_provider = optimizer_provider_entries(native)
        var device_driver = optimizer_driver(device_provider)
        var binding = entries["module"]
        var handle = optimizer_io(module, provider, owner, 5, PythonObject(None), "params", j)
        var published = device_adopt(result, binding, handle, device_driver)
        _ = device_provider
        _ = owner_state
        return published
    if operation == "step":
        opt_scalar(values[2], context)
        return door_step(module, provider, owner, values[0], values[1], Int(py=values[2]), values[3])
    if operation == "state":
        opt_scalar(values[0], context)
        return door_state(module, provider, owner, Int(py=values[0]))
    if operation == "state_set":
        opt_scalar(values[0], context)
        opt_value(values[1], context)
        return door_state_set(module, provider, owner, Int(py=values[0]), values[1])
    if operation == "load":
        var state = values[0]
        opt_value(state, context)
        if not opt_exact(py_type(state), context["mapping_types"]):
            raise Error("TypeError: optimizer checkpoint must be a native dictionary")
        opt_scalar(state["t"], context)
        var t = optimizer_integer(state["t"], context)
        var m = optimizer_state_array(state["exp_avg"], False, context)
        var v = optimizer_state_array(state["exp_avg_sq"], False, context)
        var initialized = optimizer_state_array(state["buf_initialized"], True, context)
        # Conversion released the GIL: retain the previous door's fresh owner
        # admission before reading fields another thread could have rebound.
        opt_owner(owner, context)
        optimizer_load_buffers(module, provider, owner, m, v, initialized)
        var fields = py_fields(owner)
        py_setitem(fields, "buf_initialized", initialized)
        py_setitem(fields, "t", PythonObject(t))
        return PythonObject(None)
    if operation == "zero":
        return door_zero(owner, values[0])
    if operation == "resident":
        var owner_state = optimizer_owner_value(owner)
        var active = owner_state.downcast_value_ptr[OptimizerOwner]()[].control.resident()
        _ = owner_state
        return PythonObject(active)
    if operation == "io":
        opt_scalar(values[0], context)
        var op = Int(py=values[0])
        var fields = py_fields(owner)
        var owner_state = optimizer_owner_value(owner)
        var record = owner_state.downcast_value_ptr[OptimizerOwner]()
        var tensors = values[1]
        if op == 1 or op == 2:
            tensors = fields["params"]
        if op == 2 and not record[].control.parameters_device:
            _ = owner_state
            return fields["params"]
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
    if operation == "clip":
        return door_clip(provider, values[0], values[1], values[2], values[3], values[4])
    if operation == "accumulate":
        return door_accumulate(provider, values[0], values[1])
    if operation == "aligned":
        opt_scalar(values, context)
        return PythonObject(microbatch_split_is_identical(optimizer_integer(values[0], context), optimizer_integer(values[1], context)))
    raise Error("TypeError: unknown optimizer native operation")


def optimizer_native_c(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=packed)
        if len(args) != 5:
            raise Error("TypeError: optimizer_native expects five arguments")
        return optimizer_native(String(py=args[0]), args[1], args[2], args[3], args[4]).steal_data()
    except error:
        var message = String(error)
        if message.startswith("ImportError: "):
            return raise_python_exception(Error(message[byte=13:]), ExceptionType("PyExc_ImportError"))
        if message.startswith("AttributeError: "):
            return raise_python_exception(Error(message[byte=16:]), ExceptionType("PyExc_AttributeError"))
        if message.startswith("IndexError: "):
            return raise_python_exception(Error(message[byte=12:]), ExceptionType("PyExc_IndexError"))
        return optimizer_python_error(error)
