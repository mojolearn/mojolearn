# SPDX-License-Identifier: Apache-2.0
# Whole sequence-optimizer operations. All data preparation, owner routing,
# schedule evaluation and state/output handling execute natively here.
from std.os import getenv
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_fields, py_get, py_setitem
from bindings.py_typed_call import py_call_typed
from bindings.optimizer_admission import opt_exact, opt_scalar, opt_value, opt_schedule
from bindings.optimizer_public_door import TensorView, tensor_view, base_format, same_shape, py_ints
from bindings.optimizer_device_native import device_shape, shape_count
from bindings.optimizer_schedule_native import schedule_operation
from bindings.classical_arrays import classical_input, classical_typed, classical_address, classical_is_array, classical_cast_kind
from bindings.classical_attributes import classical_connect_numpy
from core.sequence_optimizer_inputs import sequence_optimizer_cast
from core.scaler_fit_impl import scaler_fill


def seq_class(owner: PythonObject, context: PythonObject) raises -> Int:
    var classes = context["sequence_owners"]
    for i in range(len(classes)):  # driver-loop: over exact public sequence optimizer classes
        if py_type(owner) is classes[i]:
            return i
    raise Error("TypeError: sequence optimizer requires an exact native optimizer owner")


def seq_binding(mode: PythonObject, context: PythonObject, binding: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var bindings = context["sequence_bindings"]
    if binding is not None:
        for name in ["identical", "fast"]:  # driver-loop: over two compiled numeric-mode owners
            var entry = bindings[PythonObject(name)]
            if py_type(entry) is b.dict and binding is entry["module"]:
                return entry
        var extra = py_get(context, "sequence_extra_bindings", Python.tuple())
        for i in range(len(extra)):  # driver-loop: fixed connected native buffer owners
            if py_type(extra[i]) is b.dict and binding is extra[i]["module"]:
                return extra[i]
        raise Error("TypeError: sequence device/state requires its connected native binding")
    var selected = context["backend"]["_DEFAULT_MODE"] if mode is None else mode
    opt_scalar(selected, context)
    if py_type(selected) is not b.str:
        raise Error("ValueError: numeric_mode must be fast, identical or None")
    selected = py_call(py_call(selected.strip).lower)
    if String(selected) != "fast" and String(selected) != "identical":
        raise Error("ValueError: numeric_mode must be fast, identical or None")
    var entries = bindings[selected]
    if py_type(entries) is b.str:
        raise Error("ImportError: " + String(entries))
    return entries


def seq_device_get(value: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    if py_type(value) is not context["sequence_device_class"]:
        raise Error("TypeError: sequence device tensor requires its exact native owner")
    return py_call(context["sequence_device_getters"][PythonObject(name)], value)


def seq_device(value: PythonObject, context: PythonObject) raises -> Bool:
    if py_type(value) is not context["sequence_device_class"]:
        return False
    opt_scalar(seq_device_get(value, "h", context), context)
    _ = shape_count(device_shape(seq_device_get(value, "shape", context), context))
    _ = seq_binding(PythonObject(None), context, seq_device_get(value, "b", context))
    return True


def seq_list(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    if py_type(value) is b.list or py_type(value) is b.tuple:
        return py_call(b.list, value)
    if not seq_device(value, context):
        opt_value(value, context)
    return Python.list(value)


def seq_shape(value: PythonObject, context: PythonObject) raises -> List[Int]:
    if seq_device(value, context):
        return device_shape(seq_device_get(value, "shape", context), context)
    var view = TensorView()
    if not tensor_view(value, view):
        raise Error("TypeError: sequence parameter requires a native float32 buffer")
    return view.shape.copy()


def seq_addr(value: PythonObject, count: Int, context: PythonObject, wide: Bool = False) raises -> Int:
    var view = TensorView()
    if not tensor_view(value, view) or not view.c_contig or view.readonly or view.count() != count:
        raise Error("TypeError: sequence optimizer state requires writable contiguous buffers of the declared size")
    var format = base_format(view.fmt)
    if (wide and (view.itemsize != 8 or (format != "q" and format != "l"))) or (not wide and (format != "f" or view.itemsize != 4)):
        raise Error("TypeError: sequence optimizer state has the wrong native dtype")
    return view.addr


def seq_array(n: Int, context: PythonObject, fill: Float32 = 0.0, wide: Bool = False) raises -> PythonObject:
    if n < 0 or n > 9223372036854775807 // 8:
        raise Error("OverflowError: sequence optimizer buffer size is invalid")
    var input = context["input_context"]
    classical_connect_numpy(input)
    var result = classical_typed(Python.tuple(n), "<i8" if wide else "<f4", input)
    if not wide and fill != 0.0:
        scaler_fill(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(result))), n, fill)
    return result


def seq_cast(value: PythonObject, context: PythonObject, target: Int = -1) raises -> PythonObject:
    opt_value(value, context)
    var input = context["input_context"]
    classical_connect_numpy(input)
    var source = classical_input(value, input)
    var n = Int(py=source.size)
    if target >= 0 and ((Int(py=source.ndim) > 1) or (n != 1 and n != target)):
        raise Error("ValueError: optimizer checkpoint cannot broadcast to its state slot")
    var result = classical_typed(source.shape if target < 0 else Python.tuple(target), "<f4", input)
    var shape = List[Int]()
    var strides = List[Int]()
    for i in range(len(source.shape)):  # driver-loop: over native buffer shape and stride words
        shape.append(Int(py=source.shape[i]))
        strides.append(Int(py=source.strides[i]))
    var item = Int(py=source.itemsize)
    var kind = classical_cast_kind(source, item, "sequence optimizer input", input)
    var native = classical_is_array(source, input) or Bool(py=source.dtype.isnative)
    sequence_optimizer_cast(Int(py=classical_address(source)), shape, strides, item, kind, native,
                            Int(py=classical_address(result)), n if target < 0 else target, target >= 0 and n == 1)
    _ = source^
    return result


def seq_pair(value: PythonObject, context: PythonObject) raises -> PythonObject:
    opt_value(value, context)
    var b = Python.import_module("builtins")
    if py_type(value) is b.list or py_type(value) is b.tuple:
        opt_scalar(value, context)
        if len(value) != 2:
            raise Error("ValueError: optimizer configuration requires two values")
        return Python.tuple(value[0], value[1])
    var input = context["input_context"]
    classical_connect_numpy(input)
    var source = classical_input(value, input)
    if Int(py=source.ndim) != 1 or Int(py=source.size) != 2:
        raise Error("ValueError: optimizer configuration requires two values")
    # Native ndarray/Array scalar reads cannot call user conversion hooks.
    if classical_is_array(source, input):
        var packed = py_call(Python.import_module("builtins").memoryview, source._store)
        return Python.tuple(packed[0], packed[1])
    return Python.tuple(source[0], source[1])


def seq_fields(owner: PythonObject, context: PythonObject) raises -> PythonObject:
    _ = seq_class(owner, context)
    var fields = py_fields(owner)
    var b = Python.import_module("builtins")
    var keys = py_call(b.list, fields)
    for i in range(len(keys)):  # driver-loop: over stored owner field names before native dictionary lookups
        if py_type(keys[i]) is not b.str:
            raise Error("TypeError: sequence optimizer metadata requires native string keys")
    for name in ["numeric_mode", "lr", "n_total", "_kind", "_flags", "_fp", "beta2_decay", "d", "weight_decay", "eps", "betas", "max_grad_norm"]:  # driver-loop: over fixed native optimizer configuration fields
        opt_scalar(py_get(fields, PythonObject(name)), context)
    opt_schedule(py_get(fields, "lr_schedule"), context)
    for name in ["_words", "_handles", "_state", "_af_state", "_sc", "options"]:  # driver-loop: over stored native optimizer buffer containers
        opt_value(py_get(fields, PythonObject(name)), context)
    return fields


def seq_slots(fields: PythonObject, kind: Int, context: PythonObject, padded: Bool = False) raises -> PythonObject:
    var result = Python.list()
    var b = Python.import_module("builtins")
    if kind == 5:
        var states = fields["_af_state"]
        var params = fields["params"]
        if (py_type(states) is not b.list and py_type(states) is not b.tuple) or len(states) != len(params):
            raise Error("ValueError: Adafactor state must have one native dictionary per parameter")
        for i in range(len(states)):  # driver-loop: over factored state slots per parameter tensor
            var state = states[i]
            if not opt_exact(py_type(state), context["mapping_types"]):
                raise Error("TypeError: Adafactor state requires native dictionaries")
            var shape = seq_shape(params[i], context)
            var first = seq_addr(state["row_var" if len(shape) == 2 else "variance"], shape[0], context)
            var second = seq_addr(state["col_var"], shape[1], context) if len(shape) == 2 else 0
            _ = py_call(result.append, first)
            _ = py_call(result.append, second)
            if padded:
                _ = py_call(result.append, 0)
    else:
        var states = fields["_state"]
        var expected = 2 if kind == 6 else 3
        if (py_type(states) is not b.list and py_type(states) is not b.tuple) or len(states) != expected:
            raise Error("ValueError: sequence optimizer state slot count changed")
        for i in range(expected):  # driver-loop: over two or three optimizer state slots
            _ = py_call(result.append, seq_addr(states[i], Int(py=fields["n_total"]), context))
        if padded and expected == 2:
            _ = py_call(result.append, 0)
    return result


def _seq_real(value: PythonObject, label: String) raises -> PythonObject:
    # A real number for an optimizer option as a Python float; a str, None or
    # an object without __float__ is refused as TypeError naming the option
    # (the ops.Session `_float` pattern), never parsed.
    try:
        return PythonObject(Float64(py=value))
    except:
        raise Error("TypeError: " + label + " must be a real number")


def seq_construct(owner: PythonObject, values: PythonObject, mode: PythonObject, context: PythonObject) raises -> PythonObject:
    var kind = seq_class(owner, context)
    var b = Python.import_module("builtins")
    var fields = py_fields(owner)
    var params = seq_list(values[0], context)
    var options = values[2]
    opt_value(options, context)
    if py_type(options) is b.tuple:
        var names = Python.tuple(
            Python.tuple("alpha", "eps", "weight_decay", "momentum", "centered"),
            Python.tuple("lr_decay", "weight_decay", "initial_accumulator_value", "eps"),
            Python.tuple("betas", "weight_decay"), Python.tuple("betas", "eps", "weight_decay"),
            Python.tuple("betas", "eps", "weight_decay", "momentum_decay", "decoupled_weight_decay"),
            Python.tuple("beta2_decay", "eps", "d", "weight_decay", "maximize"),
            Python.tuple("bias_correction", "betas", "eps", "weight_decay", "grad_averaging", "max_grad_norm", "trust_clip", "always_adapt"))
        if len(options) != len(names[kind]):
            raise Error("TypeError: invalid sequence optimizer constructor envelope")
        var config = Python.dict()
        for i in range(len(options)):  # driver-loop: over fixed native constructor argument names
            py_setitem(config, names[kind][i], options[i])
        options = config
    if py_type(options) is not b.dict:
        raise Error("TypeError: optimizer options require a native dictionary")
    opt_scalar(values[1], context)
    opt_scalar(mode, context)
    var selected = seq_binding(mode, context, PythonObject(None))
    var device = False
    var sizes = Python.list()
    var shapes = Python.list()
    for i in range(len(params)):  # driver-loop: over public parameter buffer owners
        var dev = seq_device(params[i], context)
        if i == 0:
            device = dev
            if dev:
                selected = seq_binding(mode, context, seq_device_get(params[i], "b", context))
        if dev != device or (dev and seq_device_get(params[i], "b", context) is not selected["module"]):
            raise Error("TypeError: sequence params must all use host buffers or one native device binding")
        var shape = seq_shape(params[i], context)
        var n = shape_count(shape)
        if not dev:
            var view = TensorView()
            _ = tensor_view(params[i], view)
            if base_format(view.fmt) != "f" or view.itemsize != 4 or not view.c_contig:
                raise Error("TypeError: sequence params must be C-contiguous float32 buffers")
        if kind == 6 and n < 1:
            raise Error("TypeError: LAMB params must be nonempty")
        if kind == 5 and len(shape) != 1 and len(shape) != 2:
            raise Error("NotImplementedError: Adafactor parameters must have one or two dimensions")
        _ = py_call(sizes.append, n)
        _ = py_call(shapes.append, py_ints(shape))
    if device:
        var prefix = String("adafactor") if kind == 5 else (String("lamb") if kind == 6 else String("optimizer"))
        for suffix in ["_resident_open", "_resident_step", "_resident_step_dev"]:  # driver-loop: over required resident optimizer entries
            if py_get(selected, PythonObject(prefix + suffix)) is None:
                raise Error("RuntimeError: this binding does not support resident optimizer parameters")
        if getenv("MOJOLEARN_OPTIMIZER_RESIDENT", "1") == "0":
            raise Error("RuntimeError: device params require MOJOLEARN_OPTIMIZER_RESIDENT=1")
    py_setitem(fields, "params", params)
    py_setitem(fields, "numeric_mode", mode)
    py_setitem(fields, "_dev_b", selected["module"] if device else PythonObject(None))
    py_setitem(fields, "_res_b", PythonObject(None))
    py_setitem(fields, "lr", _seq_real(values[1], "lr"))
    var extra = Python.list()
    var fp = Python.list(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    var flags = 0
    if kind < 5:
        py_setitem(fields, "options", py_call(b.dict, options))
        opt_scalar(options["weight_decay"], context)
        py_setitem(fp, PythonObject(3), _seq_real(options["weight_decay"], "weight_decay"))
        if kind == 0 or kind == 1:
            for name in ["eps", "alpha" if kind == 0 else "lr_decay", "momentum" if kind == 0 else "initial_accumulator_value"]:  # driver-loop: over RMSprop or Adagrad scalar options
                opt_scalar(options[PythonObject(name)], context)
            py_setitem(fp, PythonObject(2), _seq_real(options["eps"], "eps"))
            py_setitem(fp, PythonObject(1 if kind == 0 else 0), _seq_real(options["alpha" if kind == 0 else "lr_decay"], String("alpha" if kind == 0 else "lr_decay")))
            py_setitem(fp, PythonObject(4 if kind == 0 else 5), _seq_real(options["momentum" if kind == 0 else "initial_accumulator_value"], String("momentum" if kind == 0 else "initial_accumulator_value")))
            if kind == 0:
                opt_scalar(options["centered"], context)
                flags = Int(Bool(py=options["centered"]))
        else:
            var betas = seq_pair(options["betas"], context)
            py_setitem(fp, PythonObject(0), _seq_real(betas[0], "betas"))
            py_setitem(fp, PythonObject(1), _seq_real(betas[1], "betas"))
            if kind == 3 and not (0.0 <= Float64(py=fp[0]) < 1.0 and 0.0 <= Float64(py=fp[1]) < 1.0):
                raise Error("ValueError: Adamax betas must lie in [0, 1)")
            if kind != 2:
                opt_scalar(options["eps"], context)
                py_setitem(fp, PythonObject(2), _seq_real(options["eps"], "eps"))
            if kind == 4:
                opt_scalar(options["momentum_decay"], context)
                opt_scalar(options["decoupled_weight_decay"], context)
                py_setitem(fp, PythonObject(4), _seq_real(options["momentum_decay"], "momentum_decay"))
                if Float64(py=fp[4]) < 0:
                    raise Error("ValueError: NAdam momentum_decay must be >= 0")
                flags = Int(Bool(py=options["decoupled_weight_decay"]))
        py_setitem(fields, "_kind", PythonObject(3 if kind == 0 else (4 if kind == 1 else kind + 5)))
        py_setitem(fields, "_flags", PythonObject(flags))
        py_setitem(fields, "_fp", fp)
    elif kind == 5:
        for name in ["beta2_decay", "d", "weight_decay", "maximize"]:  # driver-loop: over Adafactor scalar options
            opt_scalar(options[PythonObject(name)], context)
        if Bool(py=options["maximize"]):
            raise Error("NotImplementedError: Adafactor maximize is not implemented")
        var beta2_decay = Float64(py=_seq_real(options["beta2_decay"], "beta2_decay"))
        var d = Float64(py=_seq_real(options["d"], "d"))
        var decay = Float64(py=_seq_real(options["weight_decay"], "weight_decay"))
        if beta2_decay > 0 or d < 1 or Float64(py=fields["lr"]) < 0 or decay < 0:
            raise Error("ValueError: Adafactor requires beta2_decay <= 0, d >= 1 and nonnegative lr/weight_decay")
        for name in ["beta2_decay", "d", "weight_decay"]:  # driver-loop: over stored Adafactor scalar options
            py_setitem(fields, PythonObject(name), _seq_real(options[PythonObject(name)], String(name)))
        var eps = seq_pair(options["eps"], context)
        py_setitem(fields, "eps", Python.tuple(1.1920928955078125e-7 if eps[0] is None else Float64(py=_seq_real(eps[0], "eps")), _seq_real(eps[1], "eps")))
    else:
        for name in ["eps", "weight_decay", "max_grad_norm", "trust_clip", "always_adapt", "grad_averaging", "bias_correction"]:  # driver-loop: over LAMB scalar options
            opt_scalar(options[PythonObject(name)], context)
        var betas = seq_pair(options["betas"], context)
        var first = Float64(py=_seq_real(betas[0], "betas"))
        var second = Float64(py=_seq_real(betas[1], "betas"))
        if not (0.0 <= first < 1.0 and 0.0 <= second < 1.0):
            raise Error("ValueError: LAMB betas must lie in [0, 1)")
        py_setitem(fields, "betas", Python.tuple(first, second))
        for name in ["eps", "weight_decay"]:  # driver-loop: over LAMB floating-point configuration
            py_setitem(fields, PythonObject(name), _seq_real(options[PythonObject(name)], String(name)))
        py_setitem(fields, "max_grad_norm", PythonObject(None) if options["max_grad_norm"] is None else _seq_real(options["max_grad_norm"], "max_grad_norm"))
        for name in ["trust_clip", "always_adapt", "grad_averaging", "bias_correction"]:  # driver-loop: over LAMB native flag words
            _ = py_call(extra.append, Int(Bool(py=options[PythonObject(name)])))
        _ = py_call(extra.append, Int(options["max_grad_norm"] is not None))
    var words = seq_array(9, context, wide=True)
    var nh = len(params) if kind == 5 else 1
    var handles = seq_array(nh, context, wide=True)
    var hp = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(handles)))
    for i in range(nh):  # driver-loop: over per-parameter resident handle metadata
        hp[i] = -1
    py_setitem(fields, "_words", words)
    py_setitem(fields, "_handles", handles)
    var ip = Python.list(2 if kind == 5 else (1 if kind == 6 else 0), len(params))
    _ = py_call(ip.extend, sizes)
    _ = py_call(ip.extend, extra)
    var n = Int(py=py_call_typed(selected["seq_optimizer_init"], classical_address(words), ip))
    if kind < 5 and n < 1:
        raise Error("ValueError: sequence optimizer has no parameters")
    py_setitem(fields, "n_total", PythonObject(n))
    var states = Python.list()
    if kind == 5:
        for i in range(len(params)):  # driver-loop: over Adafactor moment buffer owners per parameter
            var shape = shapes[i]
            var state = Python.dict(row_var=seq_array(Int(py=shape[0]), context), col_var=seq_array(Int(py=shape[1]), context)) if len(shape) == 2 else Python.dict(variance=seq_array(Int(py=shape[0]), context))
            _ = py_call(states.append, state)
        py_setitem(fields, "_af_state", states)
    else:
        for i in range(2 if kind == 6 else 3):  # driver-loop: over optimizer moment buffer owners
            _ = py_call(states.append, seq_cast(options["initial_accumulator_value"], context, n) if kind == 1 and i == 1 else seq_array(n, context))
        py_setitem(fields, "_state", states)
        py_setitem(fields, "_sc", seq_array(2 if kind == 6 else 3, context, 1.0))
    return PythonObject(None)


def seq_step(owner: PythonObject, fields: PythonObject, gradients: PythonObject, kind: Int,
             entries: PythonObject, words: Int, handles: Int, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var params = seq_list(fields["params"], context)
    var grads = seq_list(gradients, context)
    if len(grads) != len(params):
        raise Error("ValueError: sequence optimizer needs one gradient per parameter")
    var dev = py_get(fields, "_dev_b") is not None
    var addrs = Python.list()
    var ga = Python.list()
    var kinds = Python.list()
    var sizes = Python.list()
    var rows = Python.list()
    var cols = Python.list()
    var keep = Python.list()
    var total = 0
    var input = context["input_context"]
    classical_connect_numpy(input)
    for i in range(len(params)):  # driver-loop: over parameter/gradient buffer owners
        var parameter = params[i]
        var shape = seq_shape(parameter, context)
        var n = shape_count(shape)
        if n > 9223372036854775807 - total:
            raise Error("OverflowError: sequence parameter count exceeds native indexing")
        total += n
        _ = py_call(sizes.append, n)
        if kind == 5:
            if len(shape) != 1 and len(shape) != 2:
                raise Error("ValueError: Adafactor parameter rank changed")
            _ = py_call(rows.append, shape[0])
            _ = py_call(cols.append, shape[1] if len(shape) == 2 else 0)
        if seq_device(parameter, context) != dev:
            raise Error("TypeError: sequence parameter host/device ownership changed")
        if dev:
            if seq_device_get(parameter, "b", context) is not entries["module"]:
                raise Error("TypeError: sequence parameter binding ownership changed")
            _ = py_call(addrs.append, seq_device_get(parameter, "h", context))
        else:
            _ = py_call(addrs.append, seq_addr(parameter, n, context))
        var grad = grads[i]
        if seq_device(grad, context):
            if not dev or seq_device_get(grad, "b", context) is not entries["module"] or not same_shape(seq_shape(grad, context), shape):
                raise Error("ValueError: device gradient must match its parameter shape and binding")
            _ = py_call(ga.append, seq_device_get(grad, "h", context))
            _ = py_call(kinds.append, 0)
            _ = py_call(keep.append, grad)
        else:
            opt_value(grad, context)
            var source = classical_input(grad, input)
            var view = TensorView()
            _ = tensor_view(source, view)
            if not same_shape(view.shape, shape):
                raise Error("ValueError: gradient shape must match its parameter")
            if base_format(view.fmt) == "d" or (dev and base_format(view.fmt) != "f"):
                raise Error("ValueError: Adafactor gradient must be float32 of its parameter shape" if kind == 5 and not dev else "TypeError: sequence gradients must be float32; float64 gradients are refused")
            var prepared = source if base_format(view.fmt) == "f" and view.c_contig else seq_cast(source, context)
            _ = py_call(ga.append, classical_address(prepared))
            _ = py_call(kinds.append, 1 if dev else 0)
            _ = py_call(keep.append, prepared)
    if total != Int(py=fields["n_total"]):
        raise Error("ValueError: sequence parameter size changed after construction")
    _ = py_call(addrs.extend, ga)
    var ip = Python.list(len(params), Int(dev))
    _ = py_call(ip.extend, kinds)
    _ = py_call(ip.extend, sizes)
    var slots = seq_slots(fields, kind, context)
    if kind == 5:
        for i in range(len(params)):  # driver-loop: over Adafactor first moment slot addresses
            _ = py_call(addrs.append, slots[i * 2])
        for i in range(len(params)):  # driver-loop: over Adafactor second moment slot addresses
            _ = py_call(addrs.append, slots[i * 2 + 1] if Int(py=slots[i * 2 + 1]) != 0 else slots[i * 2])
        _ = py_call(ip.extend, rows)
        _ = py_call(ip.extend, cols)
    else:
        _ = py_call(addrs.extend, slots)
        _ = py_call(addrs.append, seq_addr(fields["_sc"], 2 if kind == 6 else 3, context))
        if kind < 5:
            _ = py_call(ip.extend, Python.tuple(fields["_kind"], fields["_flags"]))
    var schedule = py_get(fields, "lr_schedule")
    if schedule is not None:
        var t = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=words)[0]
        py_setitem(fields, "lr", schedule_operation("schedule_at", schedule, Python.tuple(Int(t) + 1), context))
    var fp = Python.list(fields["lr"])
    if kind < 5:
        _ = py_call(fp.extend, fields["_fp"])
    elif kind == 5:
        _ = py_call(fp.extend, Python.tuple(fields["beta2_decay"], fields["eps"][0], fields["eps"][1], fields["d"], fields["weight_decay"]))
    else:
        _ = py_call(fp.extend, Python.tuple(fields["betas"][0], fields["betas"][1], fields["eps"], fields["weight_decay"], 1.0 if fields["max_grad_norm"] is None else fields["max_grad_norm"]))
    py_setitem(fields, "_res_b", entries["module"])
    _ = py_call_typed(entries["seq_adafactor_step" if kind == 5 else ("seq_lamb_step" if kind == 6 else "seq_optimizer_step")], words, handles, addrs, ip, fp)
    _ = keep^
    return owner


def sequence_optimizer_operation(operation: String, owner: PythonObject, values: PythonObject,
                                 mode: PythonObject, context: PythonObject) raises -> PythonObject:
    if operation == "seq_construct":
        return seq_construct(owner, values, mode, context)
    var kind = seq_class(owner, context)
    var b = Python.import_module("builtins")
    var fields = py_fields(owner) if operation == "seq_finalize" else seq_fields(owner, context)
    if operation == "seq_finalize" and py_get(fields, "_res_b") is None:
        return PythonObject(None)
    var entries = seq_binding(fields["numeric_mode"], context, py_get(fields, "_res_b") if py_get(fields, "_res_b") is not None else py_get(fields, "_dev_b"))
    if operation == "seq_binding":
        return seq_binding(fields["numeric_mode"], context, py_get(fields, "_dev_b"))["module"]
    if operation == "seq_state_binding":
        return entries["module"]
    var params = seq_list(fields["params"], context)
    var nh = len(params) if kind == 5 else 1
    var words = seq_addr(fields["_words"], 9, context, True)
    var handles = seq_addr(fields["_handles"], nh, context, True)
    var w = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=words)
    if operation == "seq_word" or operation == "seq_word_set":
        opt_scalar(values, context)
        var index = Int(py=values[0])
        if index < 0 or index >= 9:
            raise Error("ValueError: optimizer word index out of range")
        if operation == "seq_word_set":
            w[index] = Int64(Int(py=py_call(b.int, values[1])))
            return PythonObject(None)
        return PythonObject(w[index] != 0) if Bool(py=values[1]) else PythonObject(Int(w[index]))
    if operation == "seq_finalize":
        try:
            _ = py_call_typed(entries["seq_optimizer_close"], words, handles, nh)
        except:
            return PythonObject(None)
        py_setitem(fields, "_res_b", PythonObject(None))
        return PythonObject(None)
    if operation == "seq_resident":
        return PythonObject(w[2] >= 0)
    if operation == "seq_handle" or operation == "seq_handles":
        if w[2] < 0:
            return PythonObject(None)
        var hp = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=handles)
        if operation == "seq_handle":
            return PythonObject(Int(hp[0]))
        var output = Python.list()
        for i in range(nh):  # driver-loop: over returned resident handle words
            _ = py_call(output.append, Int(hp[i]))
        return output
    if operation == "seq_state_set":
        opt_value(values[0], context)
        var proposed = py_call(b.dict, fields)
        py_setitem(proposed, "_af_state" if kind == 5 else "_state", values[0])
        _ = seq_slots(proposed, kind, context)
        py_setitem(fields, "_af_state" if kind == 5 else "_state", values[0])
        w[4] = 1
        w[5] = 1
        w[6] = 0
        return PythonObject(None)
    if operation == "seq_step":
        var selected = seq_binding(fields["numeric_mode"], context, py_get(fields, "_dev_b"))
        if w[2] >= 0 and selected["module"] is not entries["module"]:
            raise Error("ValueError: resident optimizer state must keep its owning numeric-mode binding")
        return seq_step(owner, fields, values[0], kind, selected, words, handles, context)
    if operation == "seq_lr":
        var schedule = py_get(fields, "lr_schedule")
        if schedule is not None:
            py_setitem(fields, "lr", schedule_operation("schedule_at", schedule, Python.tuple(Int(w[0]) + 1), context))
        return fields["lr"]
    if operation == "seq_state" or operation == "seq_slot" or operation == "seq_slots":
        var slots = seq_slots(fields, kind, context, True)
        if operation == "seq_slots":
            return slots
        _ = py_call_typed(entries["seq_optimizer_state"], words, handles, slots)
        var state = fields["_af_state" if kind == 5 else "_state"]
        if operation == "seq_slot":
            opt_scalar(values[0], context)
            return state[values[0]]
        return state
    if operation == "seq_export" or operation == "seq_import":
        if kind == 5:
            raise Error("TypeError: Adafactor exposes state, without an optimizer checkpoint method")
        var slots = seq_slots(fields, kind, context)
        var k = len(slots)
        var n = Int(py=fields["n_total"])
        var sc_len = 2 if kind == 6 else 3
        var sc_addr = seq_addr(fields["_sc"], sc_len, context)
        var out = Python.list()
        var sc = PythonObject(None)
        var t = Int(w[0])
        var sc_t = 0
        var sc_given = False
        if operation == "seq_export":
            for i in range(k):  # driver-loop: over independent checkpoint output buffers
                _ = py_call(out.append, seq_array(n, context))
            sc = seq_array(sc_len, context)
        else:
            var state = values[0]
            opt_value(state, context)
            if not opt_exact(py_type(state), context["mapping_types"]):
                raise Error("TypeError: optimizer checkpoint requires a native dictionary")
            opt_scalar(state["t"], context)
            opt_scalar(state["lr"], context)
            var saved = state["state"]
            if py_type(saved) is not b.list and py_type(saved) is not b.tuple:
                raise Error("TypeError: checkpoint state requires a native list or tuple")
            for i in range(min(k, len(saved))):  # driver-loop: over checkpoint source buffer owners
                _ = py_call(out.append, seq_cast(saved[i], context, n))
            t = Int(py=py_call(b.int, state["t"]))
            var scalar_state = py_get(state, "scalars")
            sc_given = scalar_state is not None
            if sc_given:
                if (py_type(scalar_state) is not b.list and py_type(scalar_state) is not b.tuple) or len(scalar_state) != 2:
                    raise Error("TypeError: checkpoint scalars require a native pair")
                opt_scalar(scalar_state[0], context)
                sc_t = Int(py=py_call(b.int, scalar_state[0]))
                sc = seq_cast(scalar_state[1], context, sc_len)
            else:
                sc = fields["_sc"]
            py_setitem(fields, "lr", py_call(b.float, state["lr"]))
        var addrs = py_call(b.list, slots)
        for i in range(len(out)):  # driver-loop: over state copy buffer addresses
            _ = py_call(addrs.append, classical_address(out[i]))
        _ = py_call(addrs.extend, Python.tuple(sc_addr, classical_address(sc)))
        _ = py_call_typed(entries["seq_optimizer_export" if operation == "seq_export" else "seq_optimizer_import"], words, handles, addrs,
                    Python.list(k, n, sc_len, len(out), t, Int(sc_given), sc_t))
        if operation == "seq_import":
            return owner
        return Python.dict(t=Int(w[0]), lr=fields["lr"], state=out, options=py_call(b.dict, py_get(fields, "options", Python.dict())), scalars=Python.tuple(Int(w[1]), sc))
    raise Error("TypeError: unknown sequence optimizer native operation")
