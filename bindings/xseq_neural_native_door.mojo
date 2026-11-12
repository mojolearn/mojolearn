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
from bindings.neural_labels_native_unpack import neural_encode_labels, neural_decode_labels
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.native_loaded_module_door import native_loaded_module, NativeModules
from bindings.native_identity_key_door import NativeIdentityKey
from bindings.native_prepared_owner_unpack import prepared_get, prepared_put, prepared_drop
from bindings.native_model_records import records_open, model_generation
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_fields, py_contains
from providers.python.neural_descriptor_pyapi_door import NeuralDescriptorProvider
from providers.python.autograd_pyapi_door import cast_in_of
from bindings.sequential_arrays_door import sequential_array, sequential_address, sequential_descriptor
from bindings.optimizer_admission_door import opt_schedule
from bindings.optimizer_schedule_native_door import schedule_operation
from bindings.xseq_native_common_door import (
    XF32P, XI32P, xseq_entries, xseq_base, xseq_input, xseq_is_f64, xseq_refuse_f64, xseq_zeros,
    xseq_addr, xseq_f32p, xseq_reshape, xseq_curve_list, xseq_int, xseq_float, xseq_truth, xseq_is_int,
    XseqStream, xseq_neural_provider, xseq_output_provider,
)
from bindings.admitted_sequence_door import admit_nd_f32
from providers.python.xseq_public_lifecycle_pyapi_door import xseq_public_labels, xseq_private_input
from core.xseq_marshal import xseq_fill_f32, xseq_codes_f32, xseq_take_native
from sequence.perf3_seq_resident_words import seq_resident_token
from sequence.xseq_neural_run import (
    XseqNeuralEntry, XseqOptimizer, XseqRnnFitRequest, XseqRnnPredictRequest, XseqMlpFitRequest, XseqMlpPredictRequest,
    xseq_run_as_f32_check, xseq_run_uniform, xseq_run_argmax, xseq_run_threshold, xseq_run_curve_min,
    xseq_run_opt_sgd, xseq_run_opt_adam, xseq_run_opt_nadam, xseq_run_opt_lion, xseq_run_opt_rmsprop,
    xseq_run_opt_adagrad, xseq_run_rnn_cell, xseq_run_rnn_gates, xseq_run_rnn_rank_check, xseq_run_rnn_dims_check,
    xseq_run_rnn_param_count, xseq_run_rnn_bound, xseq_run_rnn_schedule, xseq_run_rnn_fit, xseq_run_rnn_layout,
    xseq_run_rnn_predict, xseq_run_mlp_scalars_check, xseq_run_mlp_hidden_check, xseq_run_mlp_out_name,
    xseq_run_mlp_sizes, xseq_run_mlp_param_count, xseq_run_mlp_glorot, xseq_run_mlp_fit, xseq_run_copy_f32,
    xseq_run_mlp_layers_check, xseq_run_mlp_layer_check, xseq_run_mlp_pack, xseq_run_mlp_output_check,
    xseq_run_mlp_predict,
)


def _attr(owner: PythonObject, name: String) raises -> PythonObject:
    """`getattr(owner, name)` of a class constant (`_CELL`, `_TASK`, ...)."""
    var b = native_loaded_module("builtins")
    return py_call(b.getattr, owner, PythonObject(name))


def _method(owner: PythonObject, name: String) raises -> PythonObject:
    """A builtin container's or ndarray's own C method, bound to `owner`."""
    var b = native_loaded_module("builtins")
    return py_call(b.getattr, owner, PythonObject(name))


def _repr(value: PythonObject) raises -> String:
    var b = native_loaded_module("builtins")
    return String(py=py_call(b.repr, value))


def _eq(value: PythonObject, text: String) raises -> Bool:
    return Bool(py=value == PythonObject(text))


def _shape_words(value: PythonObject) raises -> List[Int]:
    """UNPACK: an ndarray's shape as words."""
    var shape = value.shape
    var words = List[Int]()
    for i in range(len(shape)):  # driver-loop: over shape words
        words.append(Int(py=shape[i]))
    return words^


def _entry(provider: PythonObject) raises -> XseqNeuralEntry:
    """UNPACK: the admitted model provider's descriptor entry and the address
    of its retained call workspace, the typed words the RUN calls take (the
    provider stays referenced by the caller through every call)."""
    var owner = provider.downcast_value_ptr[NeuralDescriptorProvider]()
    return XseqNeuralEntry(owner[].record[0].entry, Int(owner[].record[0].workspace.unsafe_ptr()))


# ---------------------------------------------------------------- shared input steps

def xseq_as_f32_c(source: PythonObject, ndim: Int, name: String, context: PythonObject) raises -> PythonObject:
    """`np.asarray(_buffer.as_f32_c(source, ndim=ndim, name=name)[0])`: the
    rank and emptiness refusals in that helper's words, then one C-ordered
    float32 array (float64 narrowed round-to-nearest-even, integers through
    float64 as `Array.astype` does). `ndim` 0 accepts any rank."""
    var a = xseq_input(source, context)
    # The rank and the size decide; the shape words are read only for a
    # refusal's text (round2-fixes: the door split read every shape word on
    # every call, perf audit 2 A pattern 9; the old helper read ndim and size).
    if (ndim != 0 and Int(py=a.ndim) != ndim) or Int(py=a.size) == 0:
        xseq_run_as_f32_check(_shape_words(a), ndim, name)
    return admit_nd_f32(a, name, False, context)


def xseq_labels(source: PythonObject, entries: PythonObject, context: PythonObject) raises -> PythonObject:
    # One shared native label encoder for numeric, unicode and byte labels.
    return neural_encode_labels(source, entries, context)


def xseq_take(classes: PythonObject, codes: PythonObject, context: PythonObject) raises -> PythonObject:
    return neural_decode_labels(classes, codes, context)


def xseq_argmax(scores: PythonObject, rows: Int, cols: Int, context: PythonObject) raises -> PythonObject:
    """The base binding's first-max-wins row argmax (`argmax_rows_f32`) of a
    C-ordered float32 block, as an int64 array."""
    var codes = sequential_array(Python.tuple(rows), context["int64"], context)
    if rows:
        var scores_address = Int(py=xseq_addr(scores))
        var codes_address = Int(py=xseq_addr(codes))
        var provider = xseq_output_provider(context)
        var entry = _entry(provider)
        var py = Python()
        with GILReleased(py):
            xseq_run_argmax(entry, scores_address, codes_address, rows, cols)
        _ = provider
        _ = scores
        _ = codes  # keep the Python object alive through the native call above
    return codes


# ---------------------------------------------------------------- optimizer arguments

def _rnn_optimizer(name: PythonObject, options: PythonObject, context: PythonObject) raises -> XseqOptimizer:
    """`(kind, flags, [f1, f2, eps, weight_decay, f7, initial_accumulator])`
    of the binding's optimizer slots, PyTorch's defaults under `options`."""
    var b = native_loaded_module("builtins")
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
    # UNPACK: the merged options as Float64 and Bool words; the slot layout
    # and the range refusals are RUN (sequence/xseq_neural_run.mojo)
    var kind = Int(py=kinds[name])
    var label = String(py=name)
    if label == "sgd":
        return xseq_run_opt_sgd(kind, xseq_truth(o["nesterov"]), xseq_float(o["momentum"]),
                                xseq_float(o["weight_decay"]), xseq_float(o["dampening"]))
    elif label == "adam" or label == "adamw" or label == "adamax":
        return xseq_run_opt_adam(label, kind, xseq_float(o["betas"][0]), xseq_float(o["betas"][1]),
                                 xseq_float(o["eps"]), xseq_float(o["weight_decay"]))
    elif label == "nadam":
        return xseq_run_opt_nadam(kind, xseq_float(o["betas"][0]), xseq_float(o["betas"][1]), xseq_float(o["eps"]),
                                  xseq_float(o["weight_decay"]), xseq_float(o["momentum_decay"]),
                                  xseq_truth(o["decoupled_weight_decay"]))
    elif label == "lion":
        return xseq_run_opt_lion(kind, xseq_float(o["betas"][0]), xseq_float(o["betas"][1]),
                                 xseq_float(o["weight_decay"]))
    elif label == "rmsprop":
        return xseq_run_opt_rmsprop(kind, xseq_truth(o["centered"]), xseq_float(o["alpha"]), xseq_float(o["eps"]),
                                    xseq_float(o["weight_decay"]), xseq_float(o["momentum"]))
    return xseq_run_opt_adagrad(kind, xseq_float(o["lr_decay"]), xseq_float(o["eps"]), xseq_float(o["weight_decay"]),
                                xseq_float(o["initial_accumulator_value"]))


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
    var b = native_loaded_module("builtins")
    if Bool(py=py_call(b.isinstance, value, native_loaded_module("array").array)):
        return py_call(_method(value, "buffer_info"))[0]
    return xseq_addr(value)


# ---------------------------------------------------------------- fitted-weight records
#
# perf2-neural (2026-11-06): one native record per fitted model in the
# prepared registry (weak on the model, outside its `__dict__`, so pickling,
# cloning and scikit-learn's dict checks see nothing). It holds the identities
# of the fitted fields predict reads (`params_`, or `coefs_` / `intercepts_`
# and every array in them) plus the admitted operand built from them, so a
# predict on an unchanged model skips the per-layer admission, private views,
# dtype casts and the weight repacking. Any replaced field (refit, set
# attribute, a list item swapped, unpickling) misses the identity compare and
# takes the full path, which also writes the new record.
#
# p3-cnn-sequence-tsa (2026-11-06, replaces the pass 2 H2 per-predict
# FNV-1a64 hash of every fitted host byte): fits publish `params_`,
# `coefs_[i]` and `intercepts_[i]` as read-only views that cannot be made
# writeable again (`xseq_readonly_view`), so an in-place value edit
# (`coefs_[0][:] = W`, `params_[:] = p`) raises NumPy's read-only error
# instead of running on the stale packed / resident copy; assigning new
# arrays misses the identity compare and repacks and re-uploads. A recording
# predict marks any still-writeable fitted array read-only (an unpickled or
# user-assigned one) and records only when all are; a hit re-checks one
# flags word per array, so turning one writeable again drops the record. A
# fitted field that is not exact ndarrays is never recorded (full path every
# call). Still not noticed: header edits inside the SAME ndarray objects
# (`params_.shape = ...`), and an array made writeable, edited and made
# read-only again between two predicts.

comptime _MLP_PACKED_SLOT = "xseq_mlp_packed_v1"
comptime _RNN_PARAMS_SLOT = "xseq_rnn_params_v1"


def _record_get(mut mods: NativeModules, slot: String, model: PythonObject, first: PythonObject,
                second: PythonObject) raises -> PythonObject:
    """The model's record in `slot` when `first` (and `second`, unless None)
    are the very objects, with the same container items, it was built from;
    None otherwise (a stale record is dropped)."""
    var record = prepared_get(mods, slot, model)
    if record is None:
        return record
    var saved = record[0]
    var key = saved.downcast_value_ptr[NativeIdentityKey]()
    var offset = 0
    var same = False
    try:
        same = key[].matches(first, offset)
        if same and second is not None:
            same = key[].matches(second, offset)
        same = same and key[].complete(offset)
    except:
        same = False
    # Lane ne1-e-records (lesson 9: identity key plus generation word, no
    # content stamp): the record's last word is the model's generation when
    # it was kept (bindings/native_model_records.mojo); any later assignment,
    # deletion, set_params, __setstate__ or load moves the model's word on.
    # 0 (no dictionary watcher) leaves the identity key alone deciding.
    if same:
        var kept = Int(py=record[len(record) - 1])
        if kept == 0 or model_generation(model) == kept:
            return record
    prepared_drop(mods, slot, model)
    return PythonObject(None)


def _record_put(mut mods: NativeModules, slot: String, model: PythonObject, first: PythonObject,
                second: PythonObject, operands: PythonObject) raises:
    """Store `(identity key of first[, second], *operands)` for `model`."""
    var key = NativeIdentityKey(mods)
    key.add(first)
    if second is not None:
        key.add(second)
    if key.retains(model):
        return  # a self-referential field must not keep its weak owner alive
    var record = Python.list(PythonObject(alloc=key^))
    for i in range(len(operands)):  # small-loop(record operands)
        _ = py_call(record.append, operands[i])
    prepared_put(mods, slot, model, py_call(native_loaded_module("builtins").tuple, record))


def xseq_readonly_view(value: PythonObject) raises -> PythonObject:
    """p3-cnn-sequence-tsa (hazard A, the read-only scheme): a read-only
    view of exact ndarray `value` whose base is a read-only memoryview, so
    `arr[:] = W` raises NumPy's read-only error and the flag cannot be
    turned back on. `value` itself is marked read-only too."""
    _ = py_call(native_loaded_module("builtins").setattr, value.flags, "writeable", PythonObject(False))
    # a memoryview of a non-writeable ndarray is itself read-only
    var view = py_call(native_loaded_module("builtins").memoryview, value)
    return py_call(py_type(value), value.shape, dtype=value.dtype, buffer=view)


def _xseq_array_frozen(value: PythonObject, freeze: Bool, ndarray: PythonObject) raises -> Bool:
    """Whether `value` is an exact ndarray that is read-only (after marking
    it so when `freeze`); False for anything else. `ndarray` is the type the
    bridge connected (`context["xseq_ndarray"]`): NumPy is not an
    import-time module connection, so `native_loaded_module("numpy")`
    refuses."""
    if py_type(value) is not ndarray:
        return False
    var flags = value.flags
    if freeze and Bool(py=flags.writeable):
        _ = py_call(native_loaded_module("builtins").setattr, flags, "writeable", PythonObject(False))
    return not Bool(py=flags.writeable)


def _xseq_field_frozen(value: PythonObject, freeze: Bool, ndarray: PythonObject) raises -> Bool:
    """One fitted field: an array, or the arrays of an exact list or tuple
    (`coefs_`, `intercepts_`)."""
    var b = native_loaded_module("builtins")
    var kind = py_type(value)
    if kind is b.list or kind is b.tuple:
        for i in range(len(value)):  # driver-loop: over fitted layer arrays
            if not _xseq_array_frozen(value[i], freeze, ndarray):
                return False
        return True
    return _xseq_array_frozen(value, freeze, ndarray)


def _xseq_params_frozen(first: PythonObject, second: PythonObject, freeze: Bool, context: PythonObject) raises -> Int:
    """p3-cnn-sequence-tsa: replaces the per-predict FNV-1a64 hash of every
    fitted host byte (pass 2 H2). The record stamp is 1 when every fitted
    array of `first` (and `second`, unless None) is read-only (made so here
    when `freeze`, which a recording predict passes), else 0 (nothing is
    recorded; the full path and per-call upload run as before). Fits publish
    read-only views that cannot be made writeable again (`xseq_readonly_view`),
    so an in-place edit raises instead of running on a stale resident copy; an
    array made read-only here (an unpickled or user-assigned one) that the
    caller turns writeable again fails the check on the next predict and the
    record is rebuilt. Cost: one flags read per array, not one pass over the
    bytes."""
    var ndarray = context["xseq_ndarray"]
    if not _xseq_field_frozen(first, freeze, ndarray):
        return 0
    if second is not None and not _xseq_field_frozen(second, freeze, ndarray):
        return 0
    return 1


def _record_put_token(mut mods: NativeModules, slot: String, model: PythonObject, first: PythonObject,
                      second: PythonObject, operands: PythonObject, stamp: Int) raises -> Int:
    """`_record_put` of `(*operands, token, stamp)` with a new resident-block
    token (perf3-seq-neural, sequence/perf3_seq_resident_words.mojo) and the
    read-only stamp of `first` / `second` (`_xseq_params_frozen`), then the
    model's generation word; the token, or 0 when no record was kept (a
    self-referential field, or `stamp` 0: a fitted array that is not
    read-only): an unrecorded block is never named, so it is uploaded per
    call as before."""
    if stamp == 0:
        return 0
    var token = seq_resident_token()
    var record = Python.list()
    for i in range(len(operands)):  # small-loop(record operands)
        _ = py_call(record.append, operands[i])
    _ = py_call(record.append, PythonObject(token))
    _ = py_call(record.append, PythonObject(stamp))
    # The generation word last (read by `_record_get`); 0 when none is kept.
    _ = records_open(model)
    _ = py_call(record.append, PythonObject(model_generation(model)))
    _record_put(mods, slot, model, first, second, record)
    if prepared_get(mods, slot, model) is None:
        return 0
    return token


# ---------------------------------------------------------------- recurrent models

def _rnn_cell(model: PythonObject, fields: PythonObject) raises -> Int:
    """`_RecurrentBase._cell()`: the binding's cell code."""
    var cell = String(py=_attr(py_type(model), "_CELL"))
    var tanh = False
    var relu = False
    if cell == "rnn":
        var nl = py_get(fields, "nonlinearity")
        tanh = _eq(nl, "tanh")
        relu = not tanh and _eq(nl, "relu")
    return xseq_run_rnn_cell(cell, tanh, relu)


def _rnn_x(source: PythonObject, fields: PythonObject, fitting: Bool, context: PythonObject) raises -> PythonObject:
    """`_RecurrentBase._check_X`: a C-ordered float32 (n, T, F) array."""
    var a = xseq_input(source, context)
    xseq_refuse_f64(a, "X: float64 is refused; pass float32 (no float64 on the device)")
    var x = admit_nd_f32(a, "X", False, context)
    xseq_run_rnn_rank_check(Int(py=x.ndim))
    if fitting:
        py_setitem(fields, "n_features_in_", Int(py=x.shape[2]))
    elif not Bool(py=x.shape[2] == fields["n_features_in_"]):
        raise Error("X has " + String(py=py_call(native_loaded_module("builtins").str, x.shape[2]))
                    + " features, the model was fitted on "
                    + String(py=py_call(native_loaded_module("builtins").str, fields["n_features_in_"])))
    return x


def _rnn_ints(model: PythonObject, fields: PythonObject) raises -> List[Int]:
    """`_RecurrentBase._ints()`, as the native words the descriptor reads."""
    var cell = _rnn_cell(model, fields)
    var d = Int(py=fields["n_features_in_"])
    var hidden = xseq_int(fields["hidden_size"])
    var layers = xseq_int(fields["num_layers"])
    var ip: List[Int] = [cell, d, hidden, layers, Int(py=fields["n_outputs_"])]
    return ip^


def _rnn_fit(model: PythonObject, fields: PythonObject, x_source: PythonObject, target: PythonObject,
             entries: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_RecurrentBase._fit(X, target)`."""
    var x = _rnn_x(x_source, fields, True, context)
    var n = Int(py=x.shape[0])
    var steps_t = Int(py=x.shape[1])
    xseq_run_rnn_dims_check(n, steps_t)
    var stream = XseqStream(py_get(fields, "random_state"), context)
    var mods = NativeModules()
    prepared_drop(mods, _RNN_PARAMS_SLOT, model)  # the new params_ is a new object
    # the parameters in layout order, U(-1/sqrt(H), 1/sqrt(H)), one block
    var cell = _rnn_cell(model, fields)
    var hidden = xseq_int(fields["hidden_size"])
    var layers = xseq_int(fields["num_layers"])
    var d = Int(py=fields["n_features_in_"])
    var outputs = Int(py=fields["n_outputs_"])
    var total = xseq_run_rnn_param_count(xseq_run_rnn_gates(cell), d, hidden, layers, outputs)
    var bound = xseq_run_rnn_bound(xseq_float(fields["hidden_size"]))
    var params = xseq_zeros(Python.tuple(total), context)
    var params_address = Int(py=xseq_addr(params))
    var output_provider = xseq_output_provider(context)
    var output_entry = _entry(output_provider)
    var py = Python()
    with GILReleased(py):
        stream.offset = xseq_run_uniform(output_entry, params_address, total, stream.seed, stream.offset, -bound, bound)
    _ = output_provider
    _ = params  # keep the Python object alive through the native call above
    # p3-cnn-sequence-tsa: the fitted block is published read-only (the fit
    # entry below still writes it through its address)
    py_setitem(fields, "params_", xseq_readonly_view(params))
    # the minibatch schedule's scalars; each epoch's row order is built by rnn_fit
    var batch_size = xseq_int(fields["batch_size"])
    var epochs = xseq_int(fields["max_epochs"])
    var schedule = xseq_run_rnn_schedule(batch_size, n, epochs)
    var seed = stream.child_seed()
    var shuffle = xseq_truth(fields["shuffle"])
    var steps = schedule.steps
    var optimizer = _rnn_optimizer(py_get(fields, "optimizer"), py_get(fields, "optimizer_options"), context)
    var rates = _rnn_rates(fields, steps, context)
    var losses = xseq_zeros(Python.tuple(steps), context)
    var request = XseqRnnFitRequest(cell, d, hidden, layers, outputs, Int(py=_attr(py_type(model), "_TASK")), n,
                                    steps_t, epochs, schedule.batch, shuffle, seed)
    var addresses: List[Int] = [Int(py=xseq_addr(x)), Int(py=xseq_addr(target)), Int(py=xseq_addr(params)),
                                 Int(py=xseq_addr(losses)), Int(py=_buffer_address(rates))]
    var provider = xseq_neural_provider(entries)
    var entry = _entry(provider)
    with GILReleased(py):
        xseq_run_rnn_fit(entry, request, optimizer, addresses[0], addresses[1], addresses[2], addresses[3],
                         addresses[4])
    _ = provider
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
    # The admitted float32 view of `params_`, kept per model while `params_`
    # is the same object (see "fitted-weight records" above).
    var mods = NativeModules()
    var source_params = fields["params_"]
    var params: PythonObject
    var record = _record_get(mods, _RNN_PARAMS_SLOT, model, source_params, PythonObject(None))
    # a record is valid while params_ stays read-only (p3-cnn-sequence-tsa)
    if record is not None and _xseq_params_frozen(source_params, PythonObject(None), False, context) == 0:
        prepared_drop(mods, _RNN_PARAMS_SLOT, model)
        record = PythonObject(None)
    var token = 0
    if record is not None:
        params = record[1]
        token = Int(py=record[2])
    else:
        params = admit_nd_f32(xseq_private_input(source_params, context, "params_"), "params_", False, context)
        var stamp = _xseq_params_frozen(source_params, PythonObject(None), True, context)
        token = _record_put_token(mods, _RNN_PARAMS_SLOT, model, source_params, PythonObject(None), Python.tuple(params), stamp)
    var ints = _rnn_ints(model, fields)
    var n_params = xseq_run_rnn_layout(ints, Int(py=params.size))
    var task = Int(py=_attr(py_type(model), "_TASK"))
    var request = XseqRnnPredictRequest(ints^, n_params, task, n, steps_t, xseq_int(fields["predict_chunk"]), want_seq,
                                        Int(model._obj_ptr), token)
    var addresses: List[Int] = [Int(py=xseq_addr(x)), Int(py=xseq_addr(params)), Int(py=xseq_addr(out)), Int(py=xseq_addr(seq))]
    var provider = xseq_neural_provider(entries)
    var entry = _entry(provider)
    var py = Python()
    with GILReleased(py):
        xseq_run_rnn_predict(entry, request, addresses[0], addresses[1], addresses[2], addresses[3])
    _ = provider
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
            y = admit_nd_f32(y, "y", False, context)
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
    xseq_run_mlp_scalars_check(xseq_float(fields["learning_rate_init"]), xseq_float(fields["max_iter"]),
                               xseq_float(fields["alpha"]))


def _mlp_hidden(fields: PythonObject, context: PythonObject) raises -> List[Int]:
    """`_BaseMLP._hidden()`."""
    var b = native_loaded_module("builtins")
    var h = fields["hidden_layer_sizes"]
    var values = Python.list(h) if xseq_is_int(h, context) else py_call(b.list, h)
    var out = List[Int]()
    for i in range(len(values)):  # driver-loop: over hidden layer size arguments
        out.append(xseq_int(values[i]))
    xseq_run_mlp_hidden_check(out)
    return out^


def _mlp_layer_copy(params: PythonObject, offset: Int, shape: PythonObject, count: Int,
                    context: PythonObject) raises -> PythonObject:
    """`flat[offset:offset + count].reshape(shape).copy()`."""
    var out = xseq_zeros(shape, context)
    xseq_run_copy_f32(Int(py=xseq_addr(out)), Int(py=xseq_addr(params)) + 4 * offset, count)
    _ = params  # keep the Python object alive through the native copy above
    _ = out  # keep the Python object alive through the native copy above
    return out


def _mlp_fit(model: PythonObject, fields: PythonObject, x_source: PythonObject, target: PythonObject,
             out_act: Int, n_out: Int, context: PythonObject) raises -> PythonObject:
    """`_BaseMLP._fit(X, Y, out_act, n_out)`; `n_out` 0 is None (a float
    target of `target.shape[1]` columns)."""
    var b = native_loaded_module("builtins")
    _mlp_check(fields, context)
    var x = xseq_as_f32_c(x_source, 2, "X", context)
    var rows = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    var o = Int(py=target.shape[1]) if n_out == 0 else n_out
    var hidden = _mlp_hidden(fields, context)
    var sizes = xseq_run_mlp_sizes(d, hidden, o)
    var stream = XseqStream(py_get(fields, "random_state"), context)
    var mods = NativeModules()
    prepared_drop(mods, _MLP_PACKED_SLOT, model)  # the refit publishes new coefs_ / intercepts_
    var logistic = _eq(fields["activation"], "logistic")
    # Glorot-uniform per layer, drawn straight into the flat block the
    # entry trains, w then b of each layer, as the packed layout orders them
    var total = xseq_run_mlp_param_count(sizes)
    var params = xseq_zeros(Python.tuple(total), context)
    var base = Int(py=xseq_addr(params))
    var output_provider = xseq_output_provider(context)
    var output_entry = _entry(output_provider)
    var py = Python()
    with GILReleased(py):
        stream.offset = xseq_run_mlp_glorot(output_entry, base, sizes, logistic, stream.seed, stream.offset)
    _ = output_provider
    _ = params  # keep the Python object alive through the native calls above
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
    py_setitem(fields, "out_activation_", PythonObject(xseq_run_mlp_out_name(out_act)))
    py_setitem(fields, "n_features_in_", d)
    var max_iter = xseq_int(fields["max_iter"])
    var curve = xseq_zeros(Python.tuple(max_iter), context)
    # the descriptor's int and float words, native (no Python list built
    # and converted back word by word)
    var activation = Int(py=context["xseq_mlp_activations"][fields["activation"]])
    var loss_code = Int(py=fields["_LOSS_CODE"] if py_contains(fields, "_LOSS_CODE") else _attr(py_type(model), "_LOSS_CODE"))
    var solver = Int(py=context["xseq_mlp_solvers"][fields["solver"]])
    var rate_code = Int(py=context["xseq_mlp_rates"][fields["learning_rate"]])
    var nesterov = xseq_truth(fields["nesterovs_momentum"])
    var shuffle = xseq_truth(fields["shuffle"])
    var request = XseqMlpFitRequest(rows, d, o, activation, out_act, loss_code, solver, rate_code, nesterov, batch,
                                    max_iter, shuffle, seed, xseq_int(fields["n_iter_no_change"]), hidden.copy(), n_out)
    var fp: List[Float64] = [xseq_float(fields["learning_rate_init"]), xseq_float(fields["beta_1"]),
                             xseq_float(fields["beta_2"]), xseq_float(fields["epsilon"]), xseq_float(fields["momentum"]),
                             xseq_float(fields["power_t"]), xseq_float(fields["alpha"]), xseq_float(fields["tol"])]
    var entries = xseq_entries(fields, context)
    var addresses: List[Int] = [Int(py=xseq_addr(x)), Int(py=xseq_addr(target)), Int(py=xseq_addr(params)), Int(py=xseq_addr(curve))]
    var provider = xseq_neural_provider(entries)
    var entry = _entry(provider)
    var n_iter = 0
    with GILReleased(py):
        n_iter = xseq_run_mlp_fit(entry, request, fp, addresses[0], addresses[1], addresses[2], addresses[3])
    _ = provider
    _ = target
    _ = params  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    _ = curve  # keep the Python object alive through the native call above
    _ = x^
    # the fitted weights, one array per layer
    var coefs = Python.list()
    var intercepts = Python.list()
    var offset = 0
    for l in range(len(sizes) - 1):  # driver-loop: over network layers
        var fi = sizes[l]
        var fo = sizes[l + 1]
        # p3-cnn-sequence-tsa: published read-only (see `_xseq_params_frozen`)
        _ = py_call(_method(coefs, "append"),
                    xseq_readonly_view(_mlp_layer_copy(params, offset, Python.tuple(fi, fo), fi * fo, context)))
        offset += fi * fo
        _ = py_call(_method(intercepts, "append"),
                    xseq_readonly_view(_mlp_layer_copy(params, offset, Python.tuple(fo), fo, context)))
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
    var curve_address = Int(py=xseq_addr(curve))
    var best_provider = xseq_output_provider(context)
    var best_entry = _entry(best_provider)
    var best = 0.0
    with GILReleased(py):
        best = xseq_run_curve_min(best_entry, curve_address, n_iter)
    _ = best_provider
    _ = curve  # keep the Python object alive through the native call above
    _ = params  # keep the Python object alive through the native call above
    py_setitem(fields, "best_loss_", best)
    _ = curve  # keep the Python object alive through the native call above
    py_setitem(fields, "t_", n_iter * rows)
    _ = curve  # keep the Python object alive through the native call above
    _ = params  # keep the Python object alive through the native call above
    return model


def _mlp_pack(coefs: PythonObject, intercepts: PythonObject, columns: Int,
              context: PythonObject) raises -> PythonObject:
    """Admit the fitted layers and pack them, w then b of each layer, into one
    float32 block: `(params, hidden widths, input width, output width)`."""
    var layers = len(coefs)
    xseq_run_mlp_layers_check(layers, len(intercepts) if layers >= 1 else 0)
    var hidden = Python.list()
    var weights_kept = Python.list()
    var biases_kept = Python.list()
    var weight_addresses = List[Int]()
    var bias_addresses = List[Int]()
    var weight_counts = List[Int]()
    var bias_counts = List[Int]()
    var total = 0
    var input_width = columns
    for l in range(layers):  # driver-loop: capture each admitted fitted weight and its shape once
        var weights = admit_nd_f32(xseq_private_input(coefs[l], context, "coefs_"), "coefs_", False, context)
        var bias = admit_nd_f32(xseq_private_input(intercepts[l], context, "intercepts_"), "intercepts_", False, context)
        var weight_shape = _shape_words(weights)
        xseq_run_mlp_layer_check(weight_shape, _shape_words(bias), input_width)
        var fi = weight_shape[0]
        var fo = weight_shape[1]
        input_width = fo
        var count = fi * fo
        weight_counts.append(count)
        bias_counts.append(fo)
        total += count + fo
        weight_addresses.append(Int(py=xseq_addr(weights)))
        bias_addresses.append(Int(py=xseq_addr(bias)))
        _ = py_call(weights_kept.append, weights)
        _ = py_call(biases_kept.append, bias)
        if l < layers - 1:
            _ = py_call(hidden.append, PythonObject(fo))
    var params = xseq_zeros(Python.tuple(total), context)
    xseq_run_mlp_pack(Int(py=xseq_addr(params)), weight_addresses, weight_counts, bias_addresses, bias_counts)
    _ = weights_kept  # keep the admitted layers alive through the native copies above
    _ = biases_kept  # keep the admitted layers alive through the native copies above
    _ = params  # keep the Python object alive through the native copies above
    return Python.tuple(params, hidden, columns, input_width)


def _mlp_forward(model: PythonObject, fields: PythonObject, source: PythonObject, proba2: Bool,
                 context: PythonObject) raises -> PythonObject:
    """`_BaseMLP._forward(X, proba2)`."""
    var b = native_loaded_module("builtins")
    var x = xseq_as_f32_c(source, 2, "X", context)
    var x_shape = x.shape
    var rows = Int(py=x_shape[0])
    var columns = Int(py=x_shape[1])
    if not Bool(py=PythonObject(columns) == fields["n_features_in_"]):
        raise Error("X has " + String(columns) + " features, the model was fitted on "
                    + String(py=py_call(b.str, fields["n_features_in_"])))
    var coefs = fields["coefs_"]
    var intercepts = fields["intercepts_"]
    # The packed fitted block, kept per model while `coefs_` and
    # `intercepts_` hold the same arrays (see "fitted-weight records" above).
    # A record packed for another input width is not reused: the full path
    # then raises the layout refusal as before.
    var mods = NativeModules()
    var packed = _record_get(mods, _MLP_PACKED_SLOT, model, coefs, intercepts)
    if packed is not None and Int(py=packed[3]) != columns:
        packed = PythonObject(None)
    # a packed copy is valid while the fitted arrays stay read-only (p3-cnn-sequence-tsa)
    if packed is not None and _xseq_params_frozen(coefs, intercepts, False, context) == 0:
        prepared_drop(mods, _MLP_PACKED_SLOT, model)
        packed = PythonObject(None)
    var operands: PythonObject
    var token = 0
    if packed is not None:
        operands = Python.tuple(packed[1], packed[2], packed[3], packed[4])
        token = Int(py=packed[5])
    else:
        operands = _mlp_pack(coefs, intercepts, columns, context)
        var stamp = _xseq_params_frozen(coefs, intercepts, True, context)
        token = _record_put_token(mods, _MLP_PACKED_SLOT, model, coefs, intercepts, operands, stamp)
    var params = operands[0]
    var hidden_widths = operands[1]
    var input_width = Int(py=operands[3])
    var hidden = List[Int]()
    for i in range(len(hidden_widths)):  # driver-loop: over hidden layer widths
        hidden.append(Int(py=hidden_widths[i]))
    var n_outputs = Int(py=fields["n_outputs_"])
    xseq_run_mlp_output_check(input_width, n_outputs)
    var width = 2 if proba2 else n_outputs
    var out = xseq_zeros(Python.tuple(rows, width), context)
    var out_act = Int(py=context["xseq_mlp_outputs"][fields["out_activation_"]])
    var activation = Int(py=context["xseq_mlp_activations"][fields["activation"]])
    var entries = xseq_entries(fields, context)
    var request = XseqMlpPredictRequest(rows, columns, n_outputs, activation, out_act, hidden^, proba2, width,
                                        Int(py=params.size), Int(model._obj_ptr), token)
    var addresses: List[Int] = [Int(py=xseq_addr(x)), Int(py=xseq_addr(params)), Int(py=xseq_addr(out))]
    var provider = xseq_neural_provider(entries)
    var entry = _entry(provider)
    var py = Python()
    with GILReleased(py):
        xseq_run_mlp_predict(entry, request, addresses[0], addresses[1], addresses[2])
    _ = provider
    _ = params  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    _ = x^
    _ = params^
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
        return _mlp_forward(model, fields, args[0], Int(py=fields["n_outputs_"]) == 1, context)
    var out = _mlp_forward(model, fields, args[0], False, context)
    var rows = Int(py=out.shape[0])
    if not classifier:
        if xseq_truth(py_get(fields, "_y_1d", PythonObject(False))):
            return xseq_reshape(out, Python.tuple(rows))
        return out
    if Int(py=out.shape[1]) == 1:
        # p > 0.5 per row (`threshold_labels_i64`; NaN takes class 0)
        var codes = sequential_array(Python.tuple(rows), context["int64"], context)
        if rows:
            var scores_address = Int(py=xseq_addr(out))
            var codes_address = Int(py=xseq_addr(codes))
            var provider = xseq_output_provider(context)
            var entry = _entry(provider)
            var py = Python()
            with GILReleased(py):
                xseq_run_threshold(entry, scores_address, codes_address, rows)
            _ = provider
            _ = out  # keep the Python object alive through the native call above
            _ = codes  # keep the Python object alive through the native call above
        return xseq_take(xseq_public_labels(fields, context), codes, context)
    return xseq_take(xseq_public_labels(fields, context), xseq_argmax(out, rows, Int(py=out.shape[1]), context), context)
