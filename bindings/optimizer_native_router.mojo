# SPDX-License-Identifier: Apache-2.0
# Native public-operation routing, constructors, state preparation and resident IO.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem
from bindings.optimizer_public_words import optimizer_python_error
from bindings.optimizer_admission import opt_guards, opt_owner, opt_value, opt_scalar, opt_schedule
from bindings.optimizer_public_door import door_ready, refuse_kwargs, new_array, TensorView, tensor_view
from bindings.classical_attributes import classical_connect_numpy
from bindings.classical_arrays import classical_input, classical_address, classical_is_array, classical_cast_kind
from core.optimizer_state_impl import optimizer_state_cast
from bindings.optimizer_schedule_native import schedule_operation


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
    if Int(py=py_call(entries["training_numeric_mode"])) != (1 if String(selected) == "identical" else 0):
        raise Error("ImportError: optimizer binary numeric mode disagrees with the requested mode")
    return entries


def optimizer_construct(owner: PythonObject, values: PythonObject, mode: PythonObject, context: PythonObject) raises -> PythonObject:
    # values: params, kind, lr, betas, eps, weight_decay, momentum,
    # dampening, nesterov, schedule, accumulation_steps, maximize, resident, kwargs.
    var b = Python.import_module("builtins")
    var fields = py_fields(owner)
    var kind = Int(py=values[1])
    var where = String("SGD") if kind == 0 else (String("Adam") if kind == 1 else String("AdamW"))
    opt_schedule(values[9], context)
    for i in [2, 3, 4, 5, 6, 7, 8, 10, 11, 12]:  # driver-loop: over scalar constructor configuration words
        opt_scalar(values[i], context)
    opt_value(values[0], context)
    opt_value(values[13], context)
    refuse_kwargs(values[13], where)
    var profile = context["profile_defaults"]["training"]
    if not Bool(py=context["profiles"][profile]["training"]):
        raise Error("NotImplementedError: the selected numeric profile is unavailable for optimizer training")
    var accumulation = Int(py=py_call(b.int, values[10]))
    if accumulation < 1 or (accumulation & (accumulation - 1)) != 0:
        raise Error("ValueError: mojolearn." + where + ": accumulation_steps must be a power of two >= 1 (optimizer contract clause 9.2 condition 4)")
    if py_type(values[11]) is not b.bool:
        raise Error("TypeError: mojolearn." + where + ": maximize must be a bool")
    if kind == 0 and Bool(py=values[8]) and (Float64(py=values[6]) == 0.0 or Float64(py=values[7]) != 0.0):
        raise Error("ValueError: mojolearn.SGD: nesterov=True needs momentum > 0 and dampening == 0")
    var betas = values[3]
    if kind != 0 and ((py_type(betas) is not b.tuple and py_type(betas) is not b.list) or len(betas) != 2):
        raise Error("ValueError: mojolearn." + where + ": betas must be a pair (beta1, beta2)")
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
    _ = py_call(context["init"], owner, values[0])
    py_setitem(fields, "t", PythonObject(0))
    for name in ["packed_", "total_norm_", "clip_coef_"]:  # driver-loop: over three optional result words
        py_setitem(fields, PythonObject(name), PythonObject(None))
    py_setitem(fields, "lr", py_call(b.float, values[2]))
    py_setitem(fields, "weight_decay", py_call(b.float, values[5]))
    py_setitem(fields, "maximize", values[11])
    if kind == 0:
        py_setitem(fields, "momentum", py_call(b.float, values[6]))
        py_setitem(fields, "dampening", py_call(b.float, values[7]))
        py_setitem(fields, "nesterov", py_call(b.bool, values[8]))
    else:
        py_setitem(fields, "betas", Python.tuple(py_call(b.float, betas[0]), py_call(b.float, betas[1])))
        py_setitem(fields, "eps", py_call(b.float, values[4]))
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
    _ = value^
    return result


def optimizer_native(operation: String, owner: PythonObject, values: PythonObject, mode: PythonObject, context: PythonObject) raises -> PythonObject:
    opt_guards(context)
    door_ready()
    var b = Python.import_module("builtins")
    if py_type(values) is not b.tuple:
        raise Error("TypeError: optimizer requires native argument tuples")
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
    var entries = optimizer_entries(selected_mode, context)
    if operation == "binding":
        return entries["module"]
    if operation == "raw_io":
        opt_value(values, context)
        return py_call(entries["optimizer_door_io"], owner, values[0], values[1], values[2], values[3])
    if operation == "param_device":
        var fields = py_fields(owner)
        if not Bool(py=py_get(fields, "_p_dev", False)):
            raise Error("RuntimeError: optimizer.param_device: call params_to_device() first")
        opt_scalar(values[0], context)
        var j = Int(py=py_call(b.int, values[0]))
        var params = fields["params"]
        if j < 0:
            j += len(params)
        if j < 0 or j >= len(params):
            raise Error("ValueError: optimizer parameter index is out of range")
        var view = TensorView()
        _ = tensor_view(params[j], view)
        var shape = Python.list()
        var size = 1
        for i in range(len(view.shape)):  # driver-loop: over native parameter shape words
            _ = py_call(shape.append, view.shape[i])
            size *= view.shape[i]
        var handle = py_call(entries["optimizer_door_io"], owner, 5, PythonObject(None), "params", j)
        var result = py_call(b.object.__new__, context["device_class"])
        var result_fields = py_fields(result)
        py_setitem(result_fields, "_b", entries["module"])
        py_setitem(result_fields, "h", py_call(b.int, handle))
        py_setitem(result_fields, "shape", py_call(b.tuple, shape))
        py_setitem(result_fields, "size", PythonObject(size))
        py_setitem(result_fields, "_base", PythonObject(None))
        return result
    if operation == "step":
        return py_call(entries["optimizer_door_step"], owner, values[0], values[1], values[2], values[3])
    if operation == "state":
        return py_call(entries["optimizer_door_state"], owner, values[0])
    if operation == "state_set":
        opt_value(values[1], context)
        return py_call(entries["optimizer_door_state_set"], owner, values[0], values[1])
    if operation == "load":
        var state = values[0]
        opt_value(state, context)
        if py_type(state) is not b.dict:
            raise Error("TypeError: optimizer checkpoint must be a native dictionary")
        opt_scalar(state["t"], context)
        var t = py_call(b.int, state["t"])
        var m = optimizer_state_array(state["exp_avg"], False, context)
        var v = optimizer_state_array(state["exp_avg_sq"], False, context)
        var initialized = optimizer_state_array(state["buf_initialized"], True, context)
        return py_call(entries["optimizer_door_load_state"], owner, t, m, True, v, True, initialized, True)
    if operation == "zero":
        return py_call(entries["optimizer_door_zero"], owner, values[0])
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
        var result = py_call(entries["optimizer_door_io"], owner, op, tensors, values[2], values[3])
        if op == 0:
            return py_call(b.bool, result)
        if op == 1 or op == 3:
            py_setitem(fields, "_p_dev" if op == 1 else "_g_dev", PythonObject(True))
            return owner
        if op == 2 or op == 4:
            return tensors
        return result
    if operation == "clip":
        return py_call(entries["optimizer_door_clip"], values[0], values[1], values[2], values[3], values[4])
    if operation == "accumulate":
        return py_call(entries["optimizer_door_accumulate"], values[0], values[1])
    if operation == "aligned":
        opt_scalar(values, context)
        return py_call(b.bool, py_call(entries["accumulation_is_aligned"], Python.list(py_call(b.int, values[0]), py_call(b.int, values[1]))))
    raise Error("TypeError: unknown optimizer native operation")


def optimizer_native_c(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=packed)
        if len(args) != 5:
            raise Error("TypeError: optimizer_native expects five arguments")
        return optimizer_native(String(py=args[0]), args[1], args[2], args[3], args[4]).steal_data()
    except error:
        return optimizer_python_error(error)
