# SPDX-License-Identifier: Apache-2.0
# Native public-operation routing, constructors, state preparation and resident IO.
from bindings.embedding_layer_native import emblayer_operation
from std.os import getenv
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.sequence_layers_native import sequence_layers_operation
from bindings.training_extra_native import training_chunked_lm_head, training_mlp_sessions, training_residual_dropout
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem, py_contains
from bindings.py_typed_call import py_call_typed
from bindings.optimizer_public_words import optimizer_python_error, stash_python_error
from bindings.optimizer_admission import opt_guards, opt_owner, opt_value, opt_scalar, opt_schedule, opt_exact
from bindings.optimizer_public_door import door_ready, refuse_kwargs, new_array, TensorView, tensor_view, resident_entry
from bindings.classical_attributes import classical_connect_numpy
from bindings.classical_arrays import classical_input, classical_address, classical_is_array, classical_cast_kind
from core.optimizer_state_impl import optimizer_state_cast
from core.classical_pandas_impl import pd_raw_at, _pd_double
from bindings.sequence_device_native import sequence_device_operation, sequence_device_finalize
from bindings.sequence_optimizer_native import sequence_optimizer_operation
from bindings.optimizer_schedule_native import schedule_operation
from bindings.optimizer_device_native import device_operation, device_new, device_entries, device_finalize, device_adopt, device_release_entry
from bindings.training_linear_native import training_linear_operation
from bindings.training_rms_native import training_rms_operation
from bindings.training_embedding_native import training_embedding_operation
from bindings.training_cross_entropy_native import training_cross_entropy
from bindings.training_metadata_native import training_metadata


def optimizer_entries(mode: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var selected = mode
    if selected is None:
        selected = context["backend"]["_DEFAULT_MODE"]
    if py_type(selected) is not b.str:
        raise Error("ValueError: numeric_mode must be fast, identical or None")
    selected = py_call(py_call(selected.strip).lower)
    if String(selected) != "fast" and String(selected) != "identical":
        raise Error("ValueError: numeric_mode must be fast, identical or None")
    var entries = context["bindings"][selected]
    if py_type(entries) is b.str:
        raise Error("ImportError: " + String(entries))
    if Int(py=py_call_typed(entries["training_numeric_mode"])) != (1 if String(selected) == "identical" else 0):
        raise Error("ImportError: optimizer binary numeric mode disagrees with the requested mode")
    return entries


def optimizer_step_entry(entry: PythonObject, arguments: PythonObject, context: PythonObject) raises -> PythonObject:
    if py_type(entry) is not context["device_finalizer_types"][2]:
        raise Error("TypeError: optimizer step requires its connected compiled entry")
    # The inner compiled door has already assigned the public exception
    # category. Preserve its CPython exception triple across this second
    # native boundary instead of flattening it to an untyped Mojo Error.
    ref cpy = Python().cpython()
    var result = cpy.PyObject_Call(entry._obj_ptr, arguments._obj_ptr, PyObjectPtr())
    if not result:
        var failure = stash_python_error()
        raise failure^
    return PythonObject(from_owned=result)


def optimizer_betas(source: PythonObject, context: PythonObject) raises -> PythonObject:
    opt_value(source, context)
    if py_type(source) is context["container_types"][1] or py_type(source) is context["container_types"][0]:
        opt_scalar(source, context)
        if len(source) != 2:
            raise Error("ValueError: betas must be a pair (beta1, beta2)")
        return Python.tuple(py_call(context["scalar_types"][3], source[0]), py_call(context["scalar_types"][3], source[1]))
    var input = context["input_context"]
    classical_connect_numpy(input)
    var value = classical_input(source, input)
    if Int(py=value.ndim) != 1 or Int(py=value.size) != 2:
        raise Error("ValueError: betas must be a pair (beta1, beta2)")
    var item = Int(py=value.itemsize)
    var kind = classical_cast_kind(value, item, "betas", input)
    var native = classical_is_array(value, input) or Bool(py=value.dtype.isnative)
    var addr = Int(py=classical_address(value))
    _ = value  # keep the Python object alive through the native call above
    var stride = Int(py=value.strides[0])
    var first = _pd_double(pd_raw_at(addr, 0, item, native), kind, item)
    _ = value  # keep the Python object alive through the native call above
    var second = _pd_double(pd_raw_at(addr, stride, item, native), kind, item)
    _ = value  # keep the Python object alive through the native call above
    _ = value^
    return Python.tuple(first, second)


def optimizer_construct(owner: PythonObject, values: PythonObject, mode: PythonObject, context: PythonObject) raises -> PythonObject:
    # values: params, kind, lr, betas, eps, weight_decay, momentum,
    # dampening, nesterov, schedule, accumulation_steps, maximize, resident, kwargs.
    var fields = py_fields(owner)
    var kind = Int(py=values[1])
    var where = String("SGD") if kind == 0 else (String("Adam") if kind == 1 else String("AdamW"))
    opt_schedule(values[9], context)
    for i in [2, 4, 5, 6, 7, 8, 10, 11, 12]:  # driver-loop: over scalar constructor configuration words
        opt_scalar(values[i], context)
    opt_value(values[0], context)
    opt_value(values[13], context)
    refuse_kwargs(values[13], where)
    # the process default, resolved as `_numeric_profile.require_training`
    # resolves it: a recorded default, else the environment, else fp32_v1
    var profile = py_get(context["profile_defaults"], "training")
    if profile is None:
        var env = String(getenv("MOJOLEARN_NUMERIC_PROFILE", "")).strip()
        profile = PythonObject(String(env) if env.byte_length() > 0 else String("fp32_v1"))
    if not Bool(py=context["profiles"][profile]["training"]):
        raise Error("NotImplementedError: the selected numeric profile is unavailable for optimizer training")
    var accumulation = Int(py=py_call(context["scalar_types"][2], values[10]))
    if accumulation < 1 or (accumulation & (accumulation - 1)) != 0:
        raise Error("ValueError: mojolearn." + where + ": accumulation_steps must be a power of two >= 1 (optimizer contract clause 9.2 condition 4)")
    if py_type(values[11]) is not context["scalar_types"][1]:
        raise Error("TypeError: mojolearn." + where + ": maximize must be a bool")
    if kind == 0 and Bool(py=values[8]) and (Float64(py=values[6]) == 0.0 or Float64(py=values[7]) != 0.0):
        raise Error("ValueError: mojolearn.SGD: nesterov=True needs momentum > 0 and dampening == 0")
    var betas = values[3] if kind == 0 else optimizer_betas(values[3], context)
    py_setitem(fields, "_where", PythonObject(where))
    py_setitem(fields, "_KIND", PythonObject(kind))
    py_setitem(fields, "numeric_mode", mode)
    py_setitem(fields, "_res", PythonObject(None))
    py_setitem(fields, "_res_binding", PythonObject(None))
    py_setitem(fields, "_res_host", PythonObject(False))
    py_setitem(fields, "_host_fresh", PythonObject(True))
    py_setitem(fields, "_resident_wanted", values[12])
    py_setitem(fields, "lr_schedule", values[9])
    py_setitem(fields, "accumulation_steps", PythonObject(accumulation))
    # typed: the registry refusal keeps its class (ValueError: params is empty)
    _ = py_call_typed(context["init"], owner, values[0])
    py_setitem(fields, "t", PythonObject(0))
    for name in ["packed_", "total_norm_", "clip_coef_"]:  # driver-loop: over three optional result words
        py_setitem(fields, PythonObject(name), PythonObject(None))
    py_setitem(fields, "lr", py_call(context["scalar_types"][3], values[2]))
    py_setitem(fields, "weight_decay", py_call(context["scalar_types"][3], values[5]))
    py_setitem(fields, "maximize", values[11])
    if kind == 0:
        py_setitem(fields, "momentum", py_call(context["scalar_types"][3], values[6]))
        py_setitem(fields, "dampening", py_call(context["scalar_types"][3], values[7]))
        py_setitem(fields, "nesterov", py_call(context["scalar_types"][1], values[8]))
    else:
        py_setitem(fields, "betas", Python.tuple(py_call(context["scalar_types"][3], betas[0]), py_call(context["scalar_types"][3], betas[1])))
        py_setitem(fields, "eps", py_call(context["scalar_types"][3], values[4]))
    return PythonObject(None)


def optimizer_state_array(source: PythonObject, integer: Bool, context: PythonObject) raises -> PythonObject:
    opt_value(source, context)
    var input = context["input_context"]
    classical_connect_numpy(input)
    var value = classical_input(source, input)
    if Int(py=value.ndim) != 1 or Int(py=value.size) < 1:
        raise Error("ValueError: optimizer checkpoint arrays must be nonempty one-dimensional buffers")
    var n = Int(py=value.size)
    var result = new_array([n], "<i4" if integer else "<f4")
    var item = Int(py=value.itemsize)
    var kind = classical_cast_kind(value, item, "optimizer checkpoint", input)
    optimizer_state_cast(Int(py=classical_address(value)), Int(py=value.strides[0]), n, item, kind,
                         classical_is_array(value, input) or Bool(py=value.dtype.isnative),
                         MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=Int(py=result._addr)), integer)
    _ = value  # keep the Python object alive through the native call above
    _ = result  # keep the Python object alive through the native call above
    _ = value^
    return result



def optimizer_params(owner: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(owner)
    var kind = Int(py=fields["_KIND"])
    # Public constructor names are fixed. Preserve their established lexical
    # insertion order without calling a mutable builtins.sorted field.
    var names = Python.list("accumulation_steps", "betas", "eps", "lr", "lr_schedule", "maximize", "numeric_mode", "params", "resident", "weight_decay")
    if kind == 0:
        names = Python.list("accumulation_steps", "dampening", "lr", "lr_schedule", "maximize", "momentum", "nesterov", "numeric_mode", "params", "resident", "weight_decay")
    var result = Python.dict()
    for i in range(len(names)):  # driver-loop: over fixed public constructor parameter names
        var name = names[i]
        py_setitem(result, name, fields["_resident_wanted"] if String(name) == "resident" else fields[name])
    return result


def optimizer_set_params(owner: PythonObject, updates: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    if py_type(updates) is not b.dict:
        raise Error("TypeError: optimizer parameter updates require native keyword arguments")
    if len(updates) == 0:
        return owner
    var config = optimizer_params(owner, context)
    var keys = py_call(context["container_types"][0], updates)
    for i in range(len(keys)):  # driver-loop: over public parameter updates
        var key = keys[i]
        if py_type(key) is not b.str:
            raise Error("TypeError: optimizer parameter names must be native strings")
        if String(key) == "lr_schedule":
            opt_schedule(updates[key], context)
        else:
            opt_value(updates[key], context)
        var pieces = py_call(key.partition, "__")
        if not py_contains(config, pieces[0]):
            raise Error("ValueError: invalid optimizer parameter '" + String(pieces[0]) + "'")
        if len(pieces[1]) != 0:
            raise Error("TypeError: optimizer nested Python set_params hooks cannot execute; pass a complete native schedule or buffer configuration")
        py_setitem(config, key, updates[key])
    var fields = py_fields(owner)
    var field_names = py_call(context["container_types"][0], fields)
    for i in range(len(field_names)):  # driver-loop: over retained owner fields before replacement
        var name = String(field_names[i])
        if name == "lr_schedule":
            opt_schedule(fields[field_names[i]], context)
        elif name == "_res_binding":
            if fields[field_names[i]] is not None and py_type(fields[field_names[i]]) is not context["module_type"]:
                raise Error("TypeError: optimizer resident state requires a native binding owner")
        else:
            opt_value(fields[field_names[i]], context)
    var saved = py_call(b.dict, fields)
    var kind = Int(py=fields["_KIND"])
    var args = Python.tuple(config["params"], kind, config["lr"],
        Python.tuple(0.9, 0.999) if kind == 0 else config["betas"],
        PythonObject(1e-8) if kind == 0 else config["eps"], config["weight_decay"],
        config["momentum"] if kind == 0 else PythonObject(0.0),
        config["dampening"] if kind == 0 else PythonObject(0.0),
        config["nesterov"] if kind == 0 else PythonObject(False),
        config["lr_schedule"], config["accumulation_steps"], config["maximize"], config["resident"], Python.dict())
    # Rebuild on the existing exact owner, retaining the old state dictionary
    # until all validation succeeds. A temporary Python optimizer object would
    # invoke its Python __del__ on decref inside this native operation.
    _ = py_call(fields.clear)
    try:
        _ = optimizer_construct(owner, args, config["numeric_mode"], context)
        if py_get(saved, "_res") is not None and py_get(saved, "_res_binding") is not None:
            _ = py_call(resident_entry(saved["_res_binding"], "optimizer_resident_close"), saved["_res"])
    except error:
        _ = py_call(fields.clear)
        _ = py_call(fields.update, saved)
        raise error^
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
    door_ready()
    var b = Python.import_module("builtins")
    if py_type(values) is not b.tuple:
        raise Error("TypeError: optimizer requires native argument tuples")
    if operation.startswith("seqdev_"):
        return sequence_device_operation(operation, owner, values, mode, context)
    if operation == "residual_dropout" or operation == "residual_dropout_backward":
        # the explicit residual layer runs in identical mode only
        return training_residual_dropout(operation, values, optimizer_entries(PythonObject("identical"), context))
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
    if operation == "construct":
        if len(values) != 14:
            raise Error("TypeError: invalid optimizer constructor envelope")
        opt_owner(owner, context)
        return optimizer_construct(owner, values, mode, context)
    var selected_mode = mode
    if owner is not None:
        opt_owner(owner, context)
        selected_mode = py_get(py_fields(owner), "numeric_mode")
    if operation == "get_params":
        opt_scalar(values[0], context)
        return optimizer_params(owner, context)
    if operation == "set_params":
        return optimizer_set_params(owner, values[0], context)
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
    if operation == "mlp_inference_sessions":
        return training_mlp_sessions(values, entries)
    if operation == "binding":
        return entries["module"]
    if operation == "raw_io":
        opt_value(values, context)
        return py_call_typed(entries["optimizer_door_io"], owner, values[0], values[1], values[2], values[3])
    if operation == "param_device":
        var fields = py_fields(owner)
        if not Bool(py=py_get(fields, "_p_dev", False)):
            raise Error("RuntimeError: optimizer.param_device: call params_to_device() first")
        opt_scalar(values[0], context)
        var j = Int(py=py_call(context["scalar_types"][2], values[0]))
        var params = fields["params"]
        if j < 0:
            j += len(params)
        if j < 0 or j >= len(params):
            raise Error("ValueError: optimizer parameter index is out of range")
        var view = TensorView()
        _ = tensor_view(params[j], view)
        var native = device_entries(entries["module"], context)
        var result = device_new(PythonObject(None), PythonObject(None), view.shape, PythonObject(None), context)
        var release = device_release_entry(native, context)
        var binding = entries["module"]
        var handle = py_call_typed(entries["optimizer_door_io"], owner, 5, PythonObject(None), "params", j)
        return device_adopt(result, binding, handle, release)
    if operation == "step":
        return optimizer_step_entry(entries["optimizer_door_step"], Python.tuple(owner, values[0], values[1], values[2], values[3]), context)
    if operation == "state":
        return py_call_typed(entries["optimizer_door_state"], owner, values[0])
    if operation == "state_set":
        opt_value(values[1], context)
        return py_call_typed(entries["optimizer_door_state_set"], owner, values[0], values[1])
    if operation == "load":
        var state = values[0]
        opt_value(state, context)
        if not opt_exact(py_type(state), context["mapping_types"]):
            raise Error("TypeError: optimizer checkpoint must be a native dictionary")
        opt_scalar(state["t"], context)
        var t = py_call(context["scalar_types"][2], state["t"])
        var m = optimizer_state_array(state["exp_avg"], False, context)
        var v = optimizer_state_array(state["exp_avg_sq"], False, context)
        var initialized = optimizer_state_array(state["buf_initialized"], True, context)
        return py_call_typed(entries["optimizer_door_load_state"], owner, t, m, True, v, True, initialized, True)
    if operation == "zero":
        return py_call_typed(entries["optimizer_door_zero"], owner, values[0])
    if operation == "resident":
        var fields = py_fields(owner)
        return PythonObject(py_get(fields, "_res") is not None and not Bool(py=py_get(fields, "_res_host", False)))
    if operation == "io":
        opt_scalar(values[0], context)
        var op = Int(py=values[0])
        var fields = py_fields(owner)
        var tensors = values[1]
        if op == 1 or op == 2:
            tensors = fields["params"]
        if op == 2 and not Bool(py=py_get(fields, "_p_dev", False)):
            return fields["params"]
        var result = py_call_typed(entries["optimizer_door_io"], owner, op, tensors, values[2], values[3])
        if op == 0:
            return py_call(context["scalar_types"][1], result)
        if op == 1 or op == 3:
            py_setitem(fields, "_p_dev" if op == 1 else "_g_dev", PythonObject(True))
            return owner
        if op == 2 or op == 4:
            return tensors
        return result
    if operation == "clip":
        return py_call_typed(entries["optimizer_door_clip"], values[0], values[1], values[2], values[3], values[4])
    if operation == "accumulate":
        return py_call_typed(entries["optimizer_door_accumulate"], values[0], values[1])
    if operation == "aligned":
        opt_scalar(values, context)
        return py_call(context["scalar_types"][1], py_call_typed(entries["accumulation_is_aligned"], Python.list(py_call(context["scalar_types"][2], values[0]), py_call(context["scalar_types"][2], values[1]))))
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
