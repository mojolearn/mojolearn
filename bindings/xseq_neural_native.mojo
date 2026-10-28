# SPDX-License-Identifier: Apache-2.0
"""Native public operations of the sequence-expansion neural estimators:
the recurrent models (`_x_sequence_rnn.py`: LSTM/GRU/RNN regressors and
classifiers) and the multi-layer perceptrons (`_x_sequence_mlp.py`:
MLPRegressor, MLPClassifier).

Each public `fit`, `predict`, `predict_proba` and `hidden_sequence` is one
call: input admission and casts, the label encoding, the seeded weight
draws, the minibatch schedule scalars, the optimizer and learning-rate
arguments, the compiled training or forward entry, and the fitted
attributes or the result, all here. The compiled entries and their argument
words are the ones the Python bodies passed, in the same order, so every
result keeps its bits. Two copies are gone: the MLP's weights are drawn
straight into the flat parameter block the entry trains (the old per-layer
arrays were packed into it, then unpacked out of it), and a classifier's
predicted codes become labels in this same call.
"""
from std.math import sqrt
from std.memory import memcpy
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_fields, py_contains
from providers.python.neural_descriptor_pyapi import neural_descriptor_execute
from providers.python.sequence_pyapi import py_ints, py_f64s
from providers.python.autograd_pyapi import cast_in_of
from sequence.ops_impl import net_of
from sequence.neural_model_dispatch import RNN_FIT, RNN_PREDICT, MLP_FIT, MLP_PREDICT, MODEL_LABELS, MODEL_ARGMAX, MODEL_MIN, MODEL_THRESHOLD
from bindings.sequential_arrays import sequential_array, sequential_address, sequential_descriptor
from bindings.optimizer_admission import opt_schedule
from bindings.optimizer_schedule_native import schedule_operation
from bindings.xseq_native_common import (
    XF32P, XI32P, xseq_entries, xseq_base, xseq_input, xseq_is_f64, xseq_f32, xseq_refuse_f64, xseq_zeros,
    xseq_addr, xseq_f32p, xseq_reshape, xseq_curve_list, xseq_int, xseq_float, xseq_truth, xseq_is_int,
    XseqStream, xseq_neural_provider, xseq_output_provider,
)
from providers.python.xseq_public_lifecycle_pyapi import xseq_public_labels
from core.xseq_marshal import xseq_fill_f32, xseq_codes_f32, xseq_take_native


def _attr(owner: PythonObject, name: String) raises -> PythonObject:
    """`getattr(owner, name)` of a class constant (`_CELL`, `_TASK`, ...)."""
    var b = Python.import_module("builtins")
    return py_call(b.getattr, owner, PythonObject(name))


def _method(owner: PythonObject, name: String) raises -> PythonObject:
    """A builtin container's or ndarray's own C method, bound to `owner`."""
    var b = Python.import_module("builtins")
    return py_call(b.getattr, owner, PythonObject(name))


def _repr(value: PythonObject) raises -> String:
    var b = Python.import_module("builtins")
    return String(py=py_call(b.repr, value))


def _eq(value: PythonObject, text: String) raises -> Bool:
    return Bool(py=value == PythonObject(text))


def _shape_text(value: PythonObject) raises -> String:
    var b = Python.import_module("builtins")
    return String(py=py_call(b.str, value.shape))


# ---------------------------------------------------------------- shared input steps

def xseq_as_f32_c(source: PythonObject, ndim: Int, name: String, context: PythonObject) raises -> PythonObject:
    """`np.asarray(_buffer.as_f32_c(source, ndim=ndim, name=name)[0])`: the
    rank and emptiness refusals in that helper's words, then one C-ordered
    float32 array (float64 narrowed round-to-nearest-even, integers through
    float64 as `Array.astype` does). `ndim` 0 accepts any rank."""
    var a = xseq_input(source, context)
    var rank = Int(py=a.ndim)
    if ndim != 0 and rank != ndim:
        var hint = String("")
        if ndim == 2 and rank < 2:
            hint = (". Reshape your data either using array.reshape(-1, 1) if your data "
                    + "has a single feature or array.reshape(1, -1) if it contains a single sample.")
        raise Error("mojolearn: " + name + " must be " + String(ndim) + "-D, got " + String(rank) + "-D shape "
                    + _shape_text(a) + hint)
    if Int(py=a.size) == 0:
        var what = String("feature(s)") if rank == 2 and Int(py=a.shape[0]) != 0 else String("sample(s)")
        raise Error("mojolearn: " + name + " is empty, shape " + _shape_text(a) + ": Found array with 0 " + what
                    + " (shape=" + _shape_text(a) + ") while a minimum of 1 is required.")
    return xseq_f32(a, name, False, context, True)


def xseq_labels(source: PythonObject, entries: PythonObject, context: PythonObject) raises -> PythonObject:
    """`(classes_, codes)` of a label vector: `unique_inverse(y)`'s sorted
    classes in y's dtype and one int32 code per row. Numeric labels take the
    compiled label sort (`model_labels`, the base binding's `unique_inverse`
    with the cast to y's dtype); str and other Python label objects take the
    label input step below."""
    var b = Python.import_module("builtins")
    var y = PythonObject(None)
    var numeric = False
    try:
        y = xseq_input(source, context)
        var kind = String(py=y.dtype.kind)
        numeric = kind == "b" or kind == "i" or kind == "u" or kind == "f"
    except:
        numeric = False
    if numeric:
        var n = Int(py=y.shape[0]) if Int(py=y.ndim) > 0 else 1
        var descriptor = sequential_descriptor(y, "y")
        var classes = sequential_array(Python.tuple(n), y.dtype, context)
        var codes = sequential_array(Python.tuple(n), context["int32"], context)
        var source = cast_in_of(descriptor, "y")
        var addresses: List[Int] = [Int(py=sequential_address(classes)), Int(py=sequential_address(codes))]
        var words: List[Int] = [source.addr, source.kind, source.itemsize, source.rows, source.cols, source.s0, source.s1]
        var count = neural_descriptor_execute(xseq_neural_provider(entries), MODEL_LABELS, addresses, words, List[Float64]())
        _ = classes  # keep the Python object alive through the native call above
        _ = codes  # keep the Python object alive through the native call above
        var selected = py_call(context["ndarray"], Python.tuple(count), dtype=y.dtype, buffer=classes)
        _ = codes  # keep the Python object alive through the native call above
        _ = y^
        return Python.tuple(selected, codes)
    # cpu-route: str and object labels have no native sort; the builtins
    # order them by the ORDER RULE (the first of equal keys kept, Python's
    # own comparison), as `_labels.sorted_classes` did
    var flat = py_call(b.list, py_call(_method(y, "tolist"))) if y is not None else py_call(b.list, source)
    if len(flat) == 0:
        raise Error("mojolearn: y is empty")
    var ordered = py_call(b.sorted, py_call(_attr(b.dict, "fromkeys"), flat))
    var index = py_call(b.dict, py_call(b.zip, ordered, py_call(b.range, len(ordered))))
    var store = py_call(context["array_type"], PythonObject("i"), py_call(b.map, _method(index, "__getitem__"), flat))
    var codes = py_call(context["ndarray"], Python.tuple(len(flat)), dtype=context["int32"], buffer=store)
    return Python.tuple(py_call(context["np_asarray"], ordered), codes)


def xseq_take(classes: PythonObject, codes: PythonObject, context: PythonObject) raises -> PythonObject:
    # Numeric measured labels are copied natively. Preserve the existing
    # string/object compatibility route explicitly; it is NOT callback-free.
    var dtype = classes.dtype
    var kind = String(py=dtype.kind)
    if kind == "b" or kind == "i" or kind == "u" or kind == "f":
        var shape = classes.shape
        var strides = classes.strides
        var dims = List[Int]()
        var steps = List[Int]()
        for axis in range(len(shape)):  # driver-loop: admit class layout once
            dims.append(Int(py=shape[axis]))
            steps.append(Int(py=strides[axis]))
        var rows = Int(py=codes.size)
        var output = sequential_array(codes.shape, dtype, context)
        xseq_take_native(Int(py=xseq_addr(classes)), dims, steps, Int(py=dtype.itemsize),
            MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(py=xseq_addr(codes))), rows,
            MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=xseq_addr(output))))
        _ = classes
        _ = codes
        return output
    return py_call(_method(classes, "take"), codes)


def xseq_argmax(scores: PythonObject, rows: Int, cols: Int, context: PythonObject) raises -> PythonObject:
    """The base binding's first-max-wins row argmax (`argmax_rows_f32`) of a
    C-ordered float32 block, as an int64 array."""
    var codes = sequential_array(Python.tuple(rows), context["int64"], context)
    if rows:
        var addresses: List[Int] = [Int(py=xseq_addr(scores)), Int(py=xseq_addr(codes))]
        var options: List[Int] = [rows, cols]
        _ = neural_descriptor_execute(xseq_output_provider(context), MODEL_ARGMAX, addresses, options, List[Float64]())
        _ = scores
        _ = codes  # keep the Python object alive through the native call above
    return codes


# ---------------------------------------------------------------- optimizer arguments

def _rnn_optimizer(name: PythonObject, options: PythonObject, context: PythonObject) raises -> PythonObject:
    """`(kind, flags, [f1, f2, eps, weight_decay, f7, initial_accumulator])`
    of the binding's optimizer slots, PyTorch's defaults under `options`."""
    var b = Python.import_module("builtins")
    var kinds = context["xseq_rnn_optimizers"]
    var known = False
    try:
        known = py_contains(kinds, name)
    except:
        known = False
    if not known:
        raise Error("optimizer must be one of " + String(py=py_call(b.str, py_call(b.sorted, kinds))) + ", got " + _repr(name))
    var o = py_call(b.dict, context["xseq_rnn_defaults"][name])
    var given = py_call(b.dict) if not xseq_truth(options) else options
    var unknown = py_call(b.set, given) - py_call(b.set, o)
    if len(unknown) > 0:
        raise Error("optimizer " + _repr(name) + " takes no option(s) " + String(py=py_call(b.str, py_call(b.sorted, unknown)))
                    + "; it takes " + String(py=py_call(b.str, py_call(b.sorted, o))))
    _ = py_call(_method(o, "update"), given)
    var kind = kinds[name]
    var label = String(py=name)
    var flags = 0
    var fp = Python.list()
    if label == "sgd":
        if xseq_truth(o["nesterov"]) and (Bool(py=o["momentum"] <= PythonObject(0)) or Bool(py=o["dampening"] != PythonObject(0))):
            raise Error("sgd: nesterov needs momentum > 0 and dampening 0 (torch.optim.SGD)")
        flags = 1 if xseq_truth(o["nesterov"]) else 0
        fp = Python.list(o["momentum"], 0.0, 0.0, o["weight_decay"], o["dampening"], 0.0)
    elif label == "adam" or label == "adamw" or label == "adamax":
        var b1 = o["betas"][0]
        var b2 = o["betas"][1]
        if not (Bool(py=PythonObject(0.0) <= b1) and Bool(py=b1 < PythonObject(1.0))
                and Bool(py=PythonObject(0.0) <= b2) and Bool(py=b2 < PythonObject(1.0))):
            raise Error(label + ": betas must lie in [0, 1)")
        fp = Python.list(b1, b2, o["eps"], o["weight_decay"], 0.0, 0.0)
    elif label == "nadam":
        if Bool(py=o["momentum_decay"] < PythonObject(0)):
            raise Error("nadam: momentum_decay must be >= 0")
        flags = 1 if xseq_truth(o["decoupled_weight_decay"]) else 0
        fp = Python.list(o["betas"][0], o["betas"][1], o["eps"], o["weight_decay"], o["momentum_decay"], 0.0)
    elif label == "lion":
        fp = Python.list(o["betas"][0], o["betas"][1], 0.0, o["weight_decay"], 0.0, 0.0)
    elif label == "rmsprop":
        flags = 1 if xseq_truth(o["centered"]) else 0
        fp = Python.list(0.0, o["alpha"], o["eps"], o["weight_decay"], o["momentum"], 0.0)
    else:
        fp = Python.list(o["lr_decay"], 0.0, o["eps"], o["weight_decay"], 0.0, o["initial_accumulator_value"])
    var floats = Python.list()
    for i in range(len(fp)):  # driver-loop: over the six fixed optimizer float slots
        _ = py_call(_method(floats, "append"), PythonObject(xseq_float(fp[i])))
    return Python.tuple(kind, flags, floats)


def _rnn_rates(fields: PythonObject, steps: Int, context: PythonObject) raises -> PythonObject:
    """The float32 learning rate of every optimizer step: the built-in
    schedule's `lr_values(steps)` (one native series), else `learning_rate`
    repeated."""
    var schedule = py_get(fields, "lr_schedule")
    if schedule is not None:
        var optimizer_context = context["optimizer_context"]
        opt_schedule(schedule, optimizer_context)
        return schedule_operation("schedule_values", schedule, Python.tuple(steps), optimizer_context)
    var rates = xseq_zeros(Python.tuple(steps), context)
    var rate = Float32(xseq_float(fields["learning_rate"]))
    xseq_fill_f32(xseq_f32p(rates), steps, rate)
    _ = rates  # keep the Python object alive through the native call above
    return rates


def _buffer_address(value: PythonObject) raises -> PythonObject:
    """The data address of an ndarray or an `array.array`."""
    var b = Python.import_module("builtins")
    if Bool(py=py_call(b.isinstance, value, Python.import_module("array").array)):
        return py_call(_method(value, "buffer_info"))[0]
    return xseq_addr(value)


# ---------------------------------------------------------------- recurrent models

def _rnn_cell(model: PythonObject, fields: PythonObject) raises -> Int:
    """`_RecurrentBase._cell()`: the binding's cell code."""
    var cell = String(py=_attr(py_type(model), "_CELL"))
    if cell == "rnn":
        var nl = py_get(fields, "nonlinearity")
        if _eq(nl, "tanh"):
            return 0
        if _eq(nl, "relu"):
            return 1
        raise Error("nonlinearity must be 'tanh' or 'relu' (nn.RNN)")
    return 2 if cell == "lstm" else 3


def _rnn_gates(cell: Int) -> Int:
    return 4 if cell == 2 else (3 if cell == 3 else 1)


def _rnn_x(source: PythonObject, fields: PythonObject, fitting: Bool, context: PythonObject) raises -> PythonObject:
    """`_RecurrentBase._check_X`: a C-ordered float32 (n, T, F) array."""
    var a = xseq_input(source, context)
    xseq_refuse_f64(a, "X: float64 is refused; pass float32 (no float64 on the device)")
    var x = xseq_f32(a, "X", False, context)
    if Int(py=x.ndim) != 3:
        raise Error("X must be 3-D: (n_samples, n_timesteps, n_features)")
    if fitting:
        py_setitem(fields, "n_features_in_", Int(py=x.shape[2]))
    elif not Bool(py=x.shape[2] == fields["n_features_in_"]):
        raise Error("X has " + String(py=py_call(Python.import_module("builtins").str, x.shape[2]))
                    + " features, the model was fitted on "
                    + String(py=py_call(Python.import_module("builtins").str, fields["n_features_in_"])))
    return x


def _rnn_ints(model: PythonObject, fields: PythonObject) raises -> PythonObject:
    """`_RecurrentBase._ints()`."""
    return Python.list(_rnn_cell(model, fields), fields["n_features_in_"], xseq_int(fields["hidden_size"]),
                       xseq_int(fields["num_layers"]), fields["n_outputs_"])


def _rnn_fit(model: PythonObject, fields: PythonObject, x_source: PythonObject, target: PythonObject,
             entries: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_RecurrentBase._fit(X, target)`."""
    var x = _rnn_x(x_source, fields, True, context)
    var n = Int(py=x.shape[0])
    var steps_t = Int(py=x.shape[1])
    if n < 1 or steps_t < 1:
        raise Error("X needs at least one sample and one time step")
    var stream = XseqStream(py_get(fields, "random_state"), context)
    # the parameters in layout order, U(-1/sqrt(H), 1/sqrt(H)), one block
    var cell = _rnn_cell(model, fields)
    var gates = _rnn_gates(cell)
    var hidden = xseq_int(fields["hidden_size"])
    var layers = xseq_int(fields["num_layers"])
    var d = Int(py=fields["n_features_in_"])
    var outputs = Int(py=fields["n_outputs_"])
    var total = 0
    for l in range(layers):  # driver-loop: over recurrent layers of the parameter layout
        var din = d if l == 0 else hidden
        total += gates * hidden * din + gates * hidden * hidden + 2 * gates * hidden
    total += outputs * hidden + outputs
    var bound = 1.0 / sqrt(xseq_float(fields["hidden_size"]))
    var params = xseq_zeros(Python.tuple(total), context)
    stream.fill_uniform_native(xseq_output_provider(context), Int(py=xseq_addr(params)), total, -bound, bound)
    _ = params  # keep the Python object alive through the native call above
    py_setitem(fields, "params_", params)
    # the minibatch schedule's scalars; each epoch's row order is built by rnn_fit
    var batch = max(1, min(xseq_int(fields["batch_size"]), n))
    var epochs = xseq_int(fields["max_epochs"])
    if epochs * n >= 2147483647:
        raise Error("max_epochs * n_samples must be below 2^31 - 1, got " + String(epochs) + " * " + String(n))
    var seed = stream.child_seed()
    var shuffle = xseq_truth(fields["shuffle"])
    var steps = epochs * ((n + batch - 1) // batch)
    var optimizer = _rnn_optimizer(py_get(fields, "optimizer"), py_get(fields, "optimizer_options"), context)
    var rates = _rnn_rates(fields, steps, context)
    var losses = xseq_zeros(Python.tuple(steps), context)
    var ip = _rnn_ints(model, fields)
    _ = py_call(_method(ip, "extend"), Python.list(Int(py=_attr(py_type(model), "_TASK")), n, steps_t, epochs, batch,
                optimizer[0], optimizer[1], 1 if shuffle else 0, Int(seed & 0xFFFFFFFF), Int(seed >> 32)))
    var addresses: List[Int] = [Int(py=xseq_addr(x)), Int(py=xseq_addr(target)), Int(py=xseq_addr(params)),
                                 Int(py=xseq_addr(losses)), Int(py=_buffer_address(rates))]
    _ = neural_descriptor_execute(xseq_neural_provider(entries), RNN_FIT, addresses, py_ints(ip), py_f64s(optimizer[2]))
    _ = target
    _ = params  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    _ = losses  # keep the Python object alive through the native call above
    _ = rates  # keep the Python object alive through the native call above
    _ = x^
    _ = rates^
    py_setitem(fields, "loss_curve_", losses)
    py_setitem(fields, "n_iter_", steps)
    py_setitem(fields, "n_timesteps_", steps_t)
    return model


def _rnn_run(model: PythonObject, fields: PythonObject, source: PythonObject, want_seq: Bool,
             context: PythonObject) raises -> PythonObject:
    """`_RecurrentBase._run(X, want_seq)`: `(out, seq)`."""
    var entries = xseq_entries(fields, context)
    var x = _rnn_x(source, fields, False, context)
    var n = Int(py=x.shape[0])
    var steps_t = Int(py=x.shape[1])
    var out = xseq_zeros(Python.tuple(n, fields["n_outputs_"]), context)
    var seq = xseq_zeros(Python.tuple(n, steps_t, xseq_int(fields["hidden_size"])) if want_seq else Python.tuple(1), context)
    var params = xseq_f32(xseq_input(fields["params_"], context), "params_", False, context)
    var ip = _rnn_ints(model, fields)
    var network = net_of(py_ints(ip))
    if Int(py=params.size) != network.n_params():
        raise Error("ValueError: recurrent params_ size differs from the fitted network layout")
    _ = py_call(_method(ip, "extend"), Python.list(Int(py=_attr(py_type(model), "_TASK")), n, steps_t,
                xseq_int(fields["predict_chunk"]), 1 if want_seq else 0))
    var addresses: List[Int] = [Int(py=xseq_addr(x)), Int(py=xseq_addr(params)), Int(py=xseq_addr(out)), Int(py=xseq_addr(seq))]
    var written = neural_descriptor_execute(xseq_neural_provider(entries), RNN_PREDICT, addresses, py_ints(ip), List[Float64]())
    if written != n * network.O:
        raise Error("RuntimeError: recurrent prediction returned an invalid result count")
    _ = params  # keep the Python object alive through the native call above
    _ = seq  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    _ = x^
    _ = params^
    return Python.tuple(out, seq)


def xseq_rnn(model: PythonObject, operation: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    if operation == "rnn_fit":
        var task = Int(py=_attr(py_type(model), "_TASK"))
        if task == 0:
            var y = xseq_input(args[1], context)
            xseq_refuse_f64(y, "y: float64 is refused; pass float32 (no float64 on the device)")
            y = xseq_f32(y, "y", False, context)
            py_setitem(fields, "_y_1d", Int(py=y.ndim) == 1)
            var rows = Int(py=y.shape[0])
            var target = xseq_reshape(y, Python.tuple(rows, -1))
            py_setitem(fields, "n_outputs_", Int(py=target.shape[1]))
            return _rnn_fit(model, fields, args[0], target, xseq_entries(fields, context), context)
        var entries = xseq_entries(fields, context)
        var encoded = xseq_labels(args[1], entries, context)
        py_setitem(fields, "classes_", encoded[0])
        var k = len(encoded[0])
        if k < 2:
            raise Error("classification needs at least two classes")
        py_setitem(fields, "n_outputs_", k)
        # the int32 codes as the float32 target the recurrent loss reads
        var codes = encoded[1]
        var count = Int(py=codes.shape[0])
        var target = xseq_zeros(Python.tuple(count), context)
        xseq_codes_f32(XI32P(unsafe_from_address=Int(py=xseq_addr(codes))), xseq_f32p(target), count)
        _ = target  # keep the Python object alive through the native call above
        _ = codes  # keep the Python object alive through the native call above
        return _rnn_fit(model, fields, args[0], target, entries, context)
    if operation == "rnn_hidden_sequence":
        return _rnn_run(model, fields, args[0], True, context)[1]
    var result = _rnn_run(model, fields, args[0], False, context)
    var out = result[0]
    if operation == "rnn_predict_proba":
        return out
    if Int(py=_attr(py_type(model), "_TASK")) == 0:
        if xseq_truth(py_get(fields, "_y_1d", PythonObject(False))):
            return xseq_reshape(out, Python.tuple(out.shape[0]))
        return out
    var rows = Int(py=out.shape[0])
    var codes = xseq_argmax(out, rows, Int(py=out.shape[1]), context)
    return xseq_take(xseq_public_labels(fields, context), codes, context)


# ---------------------------------------------------------------- multi-layer perceptrons

def _mlp_check(fields: PythonObject, context: PythonObject) raises:
    """`_BaseMLP._check()`."""
    var b = Python.import_module("builtins")
    var solver = py_get(fields, "solver")
    if _eq(solver, "lbfgs"):
        raise Error("NotImplementedError: solver='lbfgs' is not implemented (adam and sgd are)")
    if not py_contains(context["xseq_mlp_solvers"], solver):
        raise Error("solver must be one of ['adam', 'sgd']")
    if not py_contains(context["xseq_mlp_activations"], py_get(fields, "activation")):
        raise Error("activation must be one of ['identity', 'logistic', 'relu', 'tanh']")
    if not py_contains(context["xseq_mlp_rates"], py_get(fields, "learning_rate")):
        raise Error("learning_rate must be one of ['adaptive', 'constant', 'invscaling']")
    for name in ["early_stopping", "warm_start", "verbose"]:  # driver-loop: over three named flag arguments
        if xseq_truth(py_get(fields, PythonObject(name))):
            raise Error("NotImplementedError: " + String(name) + "=True is not implemented")
    if (Bool(py=fields["learning_rate_init"] <= PythonObject(0)) or Bool(py=fields["max_iter"] < PythonObject(1))
            or Bool(py=fields["alpha"] < PythonObject(0))):
        raise Error("learning_rate_init > 0, max_iter >= 1 and alpha >= 0 are required")


def _mlp_hidden(fields: PythonObject, context: PythonObject) raises -> List[Int]:
    """`_BaseMLP._hidden()`."""
    var b = Python.import_module("builtins")
    var h = fields["hidden_layer_sizes"]
    var values = Python.list(h) if xseq_is_int(h, context) else py_call(b.list, h)
    var out = List[Int]()
    for i in range(len(values)):  # driver-loop: over hidden layer size arguments
        if Bool(py=py_call(b.int, values[i]) < PythonObject(1)):
            raise Error("hidden_layer_sizes must be > 0")
    for i in range(len(values)):  # driver-loop: over hidden layer size arguments
        out.append(xseq_int(values[i]))
    return out^


def _mlp_out_name(code: Int) -> String:
    if code == 0:
        return "identity"
    if code == 1:
        return "logistic"
    if code == 2:
        return "tanh"
    if code == 3:
        return "relu"
    return "softmax"


def _mlp_layer_copy(params: PythonObject, offset: Int, shape: PythonObject, count: Int,
                    context: PythonObject) raises -> PythonObject:
    """`flat[offset:offset + count].reshape(shape).copy()`."""
    var out = xseq_zeros(shape, context)
    if count:
        memcpy(dest=xseq_f32p(out), src=xseq_f32p(params) + offset, count=count)
        _ = out  # keep the Python object alive through the native call above
    return out


def _mlp_fit(model: PythonObject, fields: PythonObject, x_source: PythonObject, target: PythonObject,
             out_act: Int, n_out: Int, context: PythonObject) raises -> PythonObject:
    """`_BaseMLP._fit(X, Y, out_act, n_out)`; `n_out` 0 is None (a float
    target of `target.shape[1]` columns)."""
    var b = Python.import_module("builtins")
    _mlp_check(fields, context)
    var x = xseq_as_f32_c(x_source, 2, "X", context)
    var rows = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    var o = Int(py=target.shape[1]) if n_out == 0 else n_out
    var hidden = _mlp_hidden(fields, context)
    var sizes = List[Int]()
    sizes.append(d)
    for i in range(len(hidden)):  # driver-loop: over hidden layer widths
        sizes.append(hidden[i])
    sizes.append(o)
    var stream = XseqStream(py_get(fields, "random_state"), context)
    var logistic = _eq(fields["activation"], "logistic")
    # Glorot-uniform per layer, drawn straight into the flat block the
    # entry trains, w then b of each layer, as the packed layout orders them
    var total = 0
    for l in range(len(sizes) - 1):  # driver-loop: over network layers
        total += sizes[l] * sizes[l + 1] + sizes[l + 1]
    var params = xseq_zeros(Python.tuple(total), context)
    var offset = 0
    var base = Int(py=xseq_addr(params))
    var output_provider = xseq_output_provider(context)
    _ = params  # keep the Python object alive through the native call above
    for l in range(len(sizes) - 1):  # driver-loop: over network layers
        var fi = sizes[l]
        var fo = sizes[l + 1]
        var factor = 2.0 if logistic else 6.0
        var bound = sqrt(factor / Float64(fi + fo))
        stream.fill_uniform_native(output_provider, base + 4 * offset, fi * fo, -bound, bound)
        _ = params  # keep the Python object alive through the native call above
        offset += fi * fo
        stream.fill_uniform_native(output_provider, base + 4 * offset, fo, -bound, bound)
        _ = params  # keep the Python object alive through the native call above
        offset += fo
    var seed = Int(stream.child_seed() % UInt64(2147483647))
    var batch: Int
    var requested = fields["batch_size"]
    if _eq(requested, "auto"):
        batch = min(200, rows)
    else:
        var clipped = py_call(b.max, PythonObject(1), py_call(b.min, requested, PythonObject(rows)))
        batch = xseq_int(clipped)
    py_setitem(fields, "n_outputs_", o)
    py_setitem(fields, "n_layers_", len(sizes))
    py_setitem(fields, "out_activation_", PythonObject(_mlp_out_name(out_act)))
    py_setitem(fields, "n_features_in_", d)
    var max_iter = xseq_int(fields["max_iter"])
    var curve = xseq_zeros(Python.tuple(max_iter), context)
    var ip = Python.list(rows, d, o, context["xseq_mlp_activations"][fields["activation"]], out_act,
                         fields["_LOSS_CODE"] if py_contains(fields, "_LOSS_CODE") else _attr(py_type(model), "_LOSS_CODE"),
                         context["xseq_mlp_solvers"][fields["solver"]], context["xseq_mlp_rates"][fields["learning_rate"]],
                         1 if xseq_truth(fields["nesterovs_momentum"]) else 0, batch, max_iter,
                         1 if xseq_truth(fields["shuffle"]) else 0, seed, xseq_int(fields["n_iter_no_change"]), len(hidden))
    for i in range(len(hidden)):  # driver-loop: over hidden layer widths
        _ = py_call(_method(ip, "append"), hidden[i])
    if n_out != 0:
        _ = py_call(_method(ip, "append"), 1 if o > 1 else 2)
    var fp = Python.list(xseq_float(fields["learning_rate_init"]), xseq_float(fields["beta_1"]),
                         xseq_float(fields["beta_2"]), xseq_float(fields["epsilon"]), xseq_float(fields["momentum"]),
                         xseq_float(fields["power_t"]), xseq_float(fields["alpha"]), xseq_float(fields["tol"]))
    var entries = xseq_entries(fields, context)
    var addresses: List[Int] = [Int(py=xseq_addr(x)), Int(py=xseq_addr(target)), Int(py=xseq_addr(params)), Int(py=xseq_addr(curve))]
    var n_iter = neural_descriptor_execute(xseq_neural_provider(entries), MLP_FIT, addresses, py_ints(ip), py_f64s(fp))
    _ = target
    _ = params  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    _ = curve  # keep the Python object alive through the native call above
    _ = x^
    # the fitted weights, one array per layer
    var coefs = Python.list()
    var intercepts = Python.list()
    offset = 0
    for l in range(len(sizes) - 1):  # driver-loop: over network layers
        var fi = sizes[l]
        var fo = sizes[l + 1]
        _ = py_call(_method(coefs, "append"), _mlp_layer_copy(params, offset, Python.tuple(fi, fo), fi * fo, context))
        offset += fi * fo
        _ = py_call(_method(intercepts, "append"), _mlp_layer_copy(params, offset, Python.tuple(fo), fo, context))
        offset += fo
    py_setitem(fields, "coefs_", coefs)
    py_setitem(fields, "intercepts_", intercepts)
    py_setitem(fields, "n_iter_", n_iter)
    _ = curve  # keep the Python object alive through the native call above
    _ = params  # keep the Python object alive through the native call above
    var losses = xseq_curve_list(curve, n_iter)
    _ = params  # keep the Python object alive through the native call above
    py_setitem(fields, "loss_curve_", losses)
    if n_iter == 0:
        raise Error("IndexError: list index out of range")
    _ = curve  # keep the Python object alive through the native calls above
    _ = params  # keep the Python object alive through the native calls above
    py_setitem(fields, "loss_", losses[n_iter - 1])
    _ = curve  # keep the Python object alive through the native call above
    _ = params  # keep the Python object alive through the native call above
    # the first-wins minimum of the curve (`reduce_stat` 0)
    var best_result = List[Float64](length=1, fill=0)
    var best_addresses: List[Int] = [Int(py=xseq_addr(curve)), Int(best_result.unsafe_ptr())]
    var best_options: List[Int] = [n_iter]
    _ = neural_descriptor_execute(xseq_output_provider(context), MODEL_MIN, best_addresses, best_options, List[Float64]())
    var best = best_result[0]
    _ = curve  # keep the Python object alive through the native call above
    _ = params  # keep the Python object alive through the native call above
    py_setitem(fields, "best_loss_", best)
    _ = curve  # keep the Python object alive through the native call above
    py_setitem(fields, "t_", n_iter * rows)
    _ = curve  # keep the Python object alive through the native call above
    _ = params  # keep the Python object alive through the native call above
    return model


def _mlp_forward(fields: PythonObject, source: PythonObject, proba2: Bool, context: PythonObject) raises -> PythonObject:
    """`_BaseMLP._forward(X, proba2)`."""
    var b = Python.import_module("builtins")
    var x = xseq_as_f32_c(source, 2, "X", context)
    var x_shape = x.shape
    var rows = Int(py=x_shape[0])
    var columns = Int(py=x_shape[1])
    if not Bool(py=PythonObject(columns) == fields["n_features_in_"]):
        raise Error("X has " + String(columns) + " features, the model was fitted on "
                    + String(py=py_call(b.str, fields["n_features_in_"])))
    var coefs = fields["coefs_"]
    var intercepts = fields["intercepts_"]
    var layers = len(coefs)
    if layers < 1 or len(intercepts) != layers:
        raise Error("ValueError: MLP fitted weights require matching nonempty layer lists")
    var hidden = Python.list()
    var weights_kept = Python.list()
    var biases_kept = Python.list()
    var weight_counts = List[Int]()
    var bias_counts = List[Int]()
    var total = 0
    var input_width = columns
    for l in range(layers):  # driver-loop: capture each admitted fitted weight and its shape once
        var weights = xseq_f32(xseq_input(coefs[l], context), "coefs_", False, context)
        var bias = xseq_f32(xseq_input(intercepts[l], context), "intercepts_", False, context)
        var weight_shape = weights.shape
        var bias_shape = bias.shape
        if len(weight_shape) != 2 or len(bias_shape) != 1:
            raise Error("ValueError: MLP weights must be matrices and biases must be vectors")
        var fi = Int(py=weight_shape[0])
        var fo = Int(py=weight_shape[1])
        if fi != input_width or fo < 1 or Int(py=bias_shape[0]) != fo:
            raise Error("ValueError: MLP fitted parameter shapes do not match the layer layout")
        input_width = fo
        var count = fi * fo
        weight_counts.append(count)
        bias_counts.append(fo)
        total += count + fo
        _ = py_call(weights_kept.append, weights)
        _ = py_call(biases_kept.append, bias)
        if l < layers - 1:
            _ = py_call(_method(hidden, "append"), fo)
    if input_width != Int(py=fields["n_outputs_"]):
        raise Error("ValueError: MLP output weights differ from n_outputs_")
    var width = 2 if proba2 else Int(py=fields["n_outputs_"])
    var out = xseq_zeros(Python.tuple(rows, width), context)
    # the packed layout of the fitted weights, w then b of each layer
    var params = xseq_zeros(Python.tuple(total), context)
    var offset = 0
    var destination = xseq_f32p(params)
    for l in range(layers):  # driver-loop: over network layers
        var w = weights_kept[l]
        var count = weight_counts[l]
        if count:
            memcpy(dest=destination + offset, src=xseq_f32p(w), count=count)
            _ = w  # keep the Python object alive through the native call above
            _ = params  # keep the Python object alive through the native call above
        offset += count
        var bias = biases_kept[l]
        count = bias_counts[l]
        if count:
            memcpy(dest=destination + offset, src=xseq_f32p(bias), count=count)
            _ = bias  # keep the Python object alive through the native call above
            _ = params  # keep the Python object alive through the native call above
        offset += count
    var out_act = Int(py=context["xseq_mlp_outputs"][fields["out_activation_"]])
    var ip = Python.list(rows, columns, fields["n_outputs_"], context["xseq_mlp_activations"][fields["activation"]],
                         out_act, 4096, len(hidden))
    _ = py_call(_method(ip, "extend"), hidden)
    if proba2:
        _ = py_call(_method(ip, "append"), 1)
    var entries = xseq_entries(fields, context)
    var addresses: List[Int] = [Int(py=xseq_addr(x)), Int(py=xseq_addr(params)), Int(py=xseq_addr(out))]
    var written = neural_descriptor_execute(xseq_neural_provider(entries), MLP_PREDICT, addresses, py_ints(ip), List[Float64]())
    _ = params  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    _ = x^
    _ = params^
    if written != rows * width:
        raise Error("RuntimeError: mlp_predict: this binding predates the two-column probability; rebuild it")
    _ = out  # keep the Python object alive through the native calls above
    return out


def xseq_mlp(model: PythonObject, operation: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    var classifier = String(py=_attr(py_type(model), "_estimator_type")) == "classifier"
    if operation == "mlp_fit":
        if not classifier:
            var y = xseq_as_f32_c(args[1], 0, "y", context)
            py_setitem(fields, "_y_1d", Int(py=y.ndim) == 1)
            var target = xseq_reshape(y, Python.tuple(Int(py=y.shape[0]), -1))
            return _mlp_fit(model, fields, args[0], target, 0, 0, context)
        var rank = 1
        try:
            rank = Int(py=xseq_input(args[1], context).ndim)
        except:
            rank = 1  # str labels: the flat list the label input step reads
        if rank != 1:
            raise Error("NotImplementedError: multilabel targets are not implemented")
        var encoded = xseq_labels(args[1], xseq_entries(fields, context), context)
        py_setitem(fields, "classes_", encoded[0])
        var k = len(encoded[0])
        if k < 2:
            raise Error("MLPClassifier needs at least two classes")
        var codes = encoded[1]
        if k == 2:
            py_setitem(fields, "_LOSS_CODE", 1)
            return _mlp_fit(model, fields, args[0], codes, 1, 1, context)
        py_setitem(fields, "_LOSS_CODE", 2)
        return _mlp_fit(model, fields, args[0], codes, 4, k, context)
    if operation == "mlp_predict_proba":
        return _mlp_forward(fields, args[0], Int(py=fields["n_outputs_"]) == 1, context)
    var out = _mlp_forward(fields, args[0], False, context)
    var rows = Int(py=out.shape[0])
    if not classifier:
        if xseq_truth(py_get(fields, "_y_1d", PythonObject(False))):
            return xseq_reshape(out, Python.tuple(rows))
        return out
    if Int(py=out.shape[1]) == 1:
        # p > 0.5 per row (`threshold_labels_i64`; NaN takes class 0)
        var codes = sequential_array(Python.tuple(rows), context["int64"], context)
        if rows:
            var addresses: List[Int] = [Int(py=xseq_addr(out)), Int(py=xseq_addr(codes))]
            var options: List[Int] = [rows]
            _ = neural_descriptor_execute(xseq_output_provider(context), MODEL_THRESHOLD, addresses, options, List[Float64]())
            _ = out  # keep the Python object alive through the native call above
            _ = codes  # keep the Python object alive through the native call above
        return xseq_take(xseq_public_labels(fields, context), codes, context)
    return xseq_take(xseq_public_labels(fields, context), xseq_argmax(out, rows, Int(py=out.shape[1]), context), context)
