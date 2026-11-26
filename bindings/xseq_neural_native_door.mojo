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
from std.sys.compile import is_defined
from std.memory import unsafe_memcpy
from std.python._cpython import GILReleased
from std.memory import unsafe_memcpy
from bindings.native_loaded_module_door import native_loaded_module, NativeModules
from bindings.optimizer_real_unpack import optimizer_real
from bindings.native_identity_key_door import NativeIdentityKey
from bindings.native_prepared_owner_unpack import prepared_get, prepared_put, prepared_drop
from bindings.py_model_records import records_open, model_generation
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_fields, py_contains
from providers.python.neural_descriptor_pyapi_door import NeuralDescriptorProvider
from providers.python.autograd_pyapi_door import cast_in_of
from bindings.sequential_arrays_door import sequential_array, sequential_address, sequential_descriptor
from bindings.optimizer_admission_door import opt_schedule
from bindings.optimizer_schedule_native_door import schedule_run
from bindings.optimizer_keys_unpack import OPTIMIZER_OPN_SCHEDULE_VALUES
from bindings.xseq_native_common_door import (
    XF32P, XI32P, xseq_route, XseqRoute, xseq_base, xseq_input, xseq_is_f64, xseq_refuse_f64, xseq_zeros,
    xseq_addr, xseq_f32p, xseq_reshape, xseq_curve_list, xseq_int, xseq_float, xseq_truth, xseq_is_int,
    XseqStream, xseq_neural_provider, xseq_zeros_i64,
)
from bindings.admitted_sequence_door import admit_nd_f32
# lane stream-fits: fit(source=) / partial_fit(source=)
from autograd.batch_source import (
    BS_FIELD_STEPS, BS_FIELD_TARGETS, SOURCE_PART_WORDS, batch_source_class_scan,
    RNN_STREAM_BALANCED, RNN_STREAM_MONITOR_WORDS, RNN_STREAM_POSITION_WORDS,
    batch_source_fit_words, batch_source_part_of, program_training_source_refuse, source_part_load,
)
from providers.python.batch_source_pyapi_door import (
    batch_source_spec, batch_source_position_out, batch_source_hold, BS_DOOR_RECURRENT,
)
from providers.python.autograd_program_arrays_pyapi_door import is_integer_input
from providers.python.xseq_public_lifecycle_pyapi_door import xseq_public_labels, xseq_private_input
from core.xseq_marshal import xseq_fill_f32, xseq_codes_f32, xseq_take_native
from sequence.perf3_seq_resident_words import seq_resident_token
from sequence.fit_loop_words import fit_loop_words
from bindings.fit_loop_pack import fit_loop_pack
from sequence.xseq_neural_run import (
    XseqNeuralEntry, XseqOptimizer, XseqRnnFitRequest, XseqRnnPredictRequest, XseqMlpFitRequest, XseqMlpPredictRequest,
    xseq_run_as_f32_check, xseq_run_uniform, xseq_run_argmax, xseq_run_threshold, xseq_run_curve_min,
    xseq_run_opt_sgd, xseq_run_opt_adam, xseq_run_opt_nadam, xseq_run_opt_lion, xseq_run_opt_rmsprop,
    xseq_run_opt_adagrad, xseq_run_rnn_cell, xseq_run_rnn_gates, xseq_run_rnn_rank_check, xseq_run_rnn_dims_check,
    xseq_run_rnn_param_count, xseq_run_rnn_bound, xseq_run_rnn_schedule, xseq_run_rnn_fit, xseq_run_rnn_layout,
    xseq_run_rnn_predict, xseq_run_mlp_scalars_check, xseq_run_mlp_hidden_check, xseq_run_mlp_out_name,
    xseq_run_mlp_sizes, xseq_run_mlp_param_count, xseq_run_mlp_glorot, xseq_run_mlp_fit, xseq_run_copy_f32,
    xseq_run_mlp_layers_check, xseq_run_mlp_layer_check, xseq_run_mlp_pack, xseq_run_mlp_output_check,
    xseq_run_mlp_predict, XseqRnnFitOptions, XseqRnnPredictOptions, xseq_run_rnn_fit_options,
    xseq_run_rnn_predict_options, xseq_run_rnn_layout_dirs, xseq_run_rnn_fit_stream, xseq_run_balanced_class_weights, xseq_run_remap_codes,
    xseq_run_step_weights, xseq_run_lengths_check, xseq_run_class_counts, RNN_REVIEW_SABOTAGE_LENGTHS_DEVICE,
    RNN_REVIEW_SABOTAGE_LENGTHS_HOST, RNN_REVIEW_SABOTAGE_PADDING_DEVICE, RNN_REVIEW_SABOTAGE_PADDING_HOST,
)
from bindings.class_weight_boundary_unpack import class_weight_dictionary_values
from core.class_weight_plan import class_weight_plan
from bindings.xseq_keys_unpack import (
    xseq_keys, XSEQ_OPN_RNN_FIT, XSEQ_OPN_RNN_HIDDEN_SEQUENCE, XSEQ_OPN_RNN_PREDICT_PROBA, XSEQ_OPN_MLP_FIT,
    XSEQ_OPN_MLP_PREDICT_PROBA,
)
from bindings.py_balanced import py_item


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


def xseq_argmax(scores: PythonObject, rows: Int, cols: Int, output: PythonObject,
                context: PythonObject) raises -> PythonObject:
    """The first-max-wins row argmax (`argmax_rows_f32`) of a C-ordered
    float32 block, as an int64 array, on the image of the estimator's route
    (`output`: the route's output entries)."""
    var nk = xseq_keys()
    var codes = sequential_array(Python.tuple(rows), py_item(context, nk[].k_int64), context)
    if rows:
        var scores_address = Int(py=xseq_addr(scores))
        var codes_address = Int(py=xseq_addr(codes))
        var provider = xseq_neural_provider(output)
        var entry = _entry(provider)
        var py = Python()
        with GILReleased(py):
            xseq_run_argmax(entry, scores_address, codes_address, rows, cols)
        _ = provider
        _ = scores
        _ = codes  # keep the Python object alive through the native call above
    return codes


# ---------------------------------------------------------------- optimizer arguments

def _opt_real(value: PythonObject) raises -> Float64:
    """`float(value)` for an optimizer hyperparameter, an integer above 2^53
    rounded once to float32 as every optimizer door does
    (bindings/optimizer_real_unpack.mojo)."""
    return optimizer_real(value, xseq_float(value))


def _rnn_optimizer(name: PythonObject, options: PythonObject, context: PythonObject) raises -> XseqOptimizer:
    """`(kind, flags, [f1, f2, eps, weight_decay, f7, initial_accumulator])`
    of the binding's optimizer slots, PyTorch's defaults under `options`."""
    var b = native_loaded_module("builtins")
    var nk = xseq_keys()
    var kinds = py_item(context, nk[].k_xseq_rnn_optimizers)
    var known = False
    try:
        known = py_contains(kinds, name)
    except:
        known = False
    if not known:
        raise Error("optimizer must be one of " + String(py=py_call(b.str, py_call(b.sorted, kinds))) + ", got " + _repr(name))
    var o = py_call(b.dict, py_item(context, nk[].k_xseq_rnn_defaults)[name])
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
        return xseq_run_opt_sgd(kind, xseq_truth(py_item(o, nk[].k_nesterov)), _opt_real(py_item(o, nk[].k_momentum)),
                                _opt_real(py_item(o, nk[].k_weight_decay)), _opt_real(py_item(o, nk[].k_dampening)))
    elif label == "adam" or label == "adamw" or label == "adamax":
        return xseq_run_opt_adam(label, kind, _opt_real(py_item(o, nk[].k_betas)[0]), _opt_real(py_item(o, nk[].k_betas)[1]),
                                 _opt_real(py_item(o, nk[].k_eps)), _opt_real(py_item(o, nk[].k_weight_decay)))
    elif label == "nadam":
        return xseq_run_opt_nadam(kind, _opt_real(py_item(o, nk[].k_betas)[0]), _opt_real(py_item(o, nk[].k_betas)[1]), _opt_real(py_item(o, nk[].k_eps)),
                                  _opt_real(py_item(o, nk[].k_weight_decay)), _opt_real(py_item(o, nk[].k_momentum_decay)),
                                  xseq_truth(py_item(o, nk[].k_decoupled_weight_decay)))
    elif label == "lion":
        return xseq_run_opt_lion(kind, _opt_real(py_item(o, nk[].k_betas)[0]), _opt_real(py_item(o, nk[].k_betas)[1]),
                                 _opt_real(py_item(o, nk[].k_weight_decay)))
    elif label == "rmsprop":
        return xseq_run_opt_rmsprop(kind, xseq_truth(py_item(o, nk[].k_centered)), _opt_real(py_item(o, nk[].k_alpha)), _opt_real(py_item(o, nk[].k_eps)),
                                    _opt_real(py_item(o, nk[].k_weight_decay)), _opt_real(py_item(o, nk[].k_momentum)))
    return xseq_run_opt_adagrad(kind, _opt_real(py_item(o, nk[].k_lr_decay)), _opt_real(py_item(o, nk[].k_eps)), _opt_real(py_item(o, nk[].k_weight_decay)),
                                _opt_real(py_item(o, nk[].k_initial_accumulator_value)))


def _rnn_rates(fields: PythonObject, steps: Int, context: PythonObject) raises -> PythonObject:
    """The float32 learning rate of every optimizer step: the built-in
    schedule's `lr_values(steps)` (one native series), else `learning_rate`
    repeated."""
    var nk = xseq_keys()
    var schedule = py_get(fields, nk[].k_lr_schedule)
    if schedule is not None:
        var optimizer_context = py_item(context, nk[].k_optimizer_context)
        opt_schedule(schedule, optimizer_context)
        return schedule_run(OPTIMIZER_OPN_SCHEDULE_VALUES, schedule, Python.tuple(steps), optimizer_context)
    var rates = xseq_zeros(Python.tuple(steps), context)
    var rate = Float32(_opt_real(py_item(fields, nk[].k_learning_rate)))
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
    # it was kept (bindings/py_model_records.mojo); any later assignment,
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
    var nk = xseq_keys()
    var ndarray = py_item(context, nk[].k_xseq_ndarray)
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
        var nk = xseq_keys()
        var nl = py_get(fields, nk[].k_nonlinearity)
        tanh = _eq(nl, "tanh")
        relu = not tanh and _eq(nl, "relu")
    return xseq_run_rnn_cell(cell, tanh, relu)


def _rnn_x(source: PythonObject, fields: PythonObject, fitting: Bool, context: PythonObject) raises -> PythonObject:
    """`_RecurrentBase._check_X`: a C-ordered float32 (n, T, F) array."""
    var a = xseq_input(source, context)
    xseq_refuse_f64(a, "X: float64 is refused; pass float32 (no float64 on the device)")
    var x = admit_nd_f32(a, "X", False, context)
    xseq_run_rnn_rank_check(Int(py=x.ndim))
    var nk = xseq_keys()
    if fitting:
        py_setitem(fields, nk[].k_n_features_in_, Int(py=x.shape[2]))
    elif not Bool(py=x.shape[2] == py_item(fields, nk[].k_n_features_in_)):
        raise Error("X has " + String(py=py_call(native_loaded_module("builtins").str, x.shape[2]))
                    + " features, the model was fitted on "
                    + String(py=py_call(native_loaded_module("builtins").str, py_item(fields, nk[].k_n_features_in_))))
    return x


def _rnn_ints(model: PythonObject, fields: PythonObject) raises -> List[Int]:
    """`_RecurrentBase._ints()`, as the native words the descriptor reads."""
    var cell = _rnn_cell(model, fields)
    var nk = xseq_keys()
    var d = Int(py=py_item(fields, nk[].k_n_features_in_))
    var hidden = xseq_int(py_item(fields, nk[].k_hidden_size))
    var layers = xseq_int(py_item(fields, nk[].k_num_layers))
    var ip: List[Int] = [cell, d, hidden, layers, Int(py=py_item(fields, nk[].k_n_outputs_))]
    return ip^


def _rnn_lengths(source: PythonObject, n: Int, context: PythonObject) raises -> PythonObject:
    """The `lengths` argument (lane seq-blocks-2): None stays None, else one
    C-ordered float32 vector of n values (integers through float64, as every
    admitted array); the values are checked natively (sequence/ops_impl.mojo
    rnn_lengths_of: whole numbers from 1 to the sequence length)."""
    if source is None:
        return source
    var lengths = xseq_as_f32_c(source, 1, "lengths", context)
    if Int(py=lengths.shape[0]) != n:
        raise Error("ValueError: lengths must hold one length per sequence (" + String(n) + ")")
    return lengths


def _rnn_lengths_address(lengths: PythonObject) raises -> Int:
    return 0 if lengths is None else Int(py=xseq_addr(lengths))


def _rnn_fit_lengths(source: PythonObject, n: Int, steps: Int, operation: String, context: PythonObject,
                     check: Bool) raises -> PythonObject:
    """fit's `lengths` (and validation_data's): `_rnn_lengths`, then every value checked ONCE here, in admission,
    before any consumer reads one (lane rnn-review-fixes, P1-a): the balanced class histogram and the valid-step
    label encoding use each length as a loop bound over its sequence's labels, so an unchecked length past its
    sequence read the next one's (past the buffer on the last row). `check` is False only on a sabotaged side
    (RNN_REVIEW_SABOTAGE_LENGTHS_*). The fit call's own admission (sequence/ops_impl.mojo rnn_lengths_of, the
    same body) still guards the native entry for its every caller."""
    var lengths = _rnn_lengths(source, n, context)
    if check and lengths is not None:
        xseq_run_lengths_check(Int(py=xseq_addr(lengths)), n, steps, operation)
    return lengths


comptime _XSEQ_COLUMN_CPU = is_defined["MOJOLEARN_COLUMN_CPU"]()


def _rnn_side(route: XseqRoute, device_fault: Bool, host_fault: Bool) -> Bool:
    """Whether a one-sided sabotage arm's fault applies to this route: the host route is an explicit
    device="cpu" beside a GPU set (route.cpu_table), the device route the rest."""
    # The CPU column (a CPU-only image, MOJOLEARN_COLUMN_CPU) is the host side too (w3-integrate-11: the host arms
    # never fired on the CPU box, whose route is not cpu_table).
    var host_side = route.cpu_table or _XSEQ_COLUMN_CPU
    return (host_fault and host_side) or (device_fault and not host_side)


# ---------------------------------------------------------------- recurrent options (lane rnn-parity)
#
# Andrew 2026-11-25. The estimators' constructor options (output, bidirectional, dropout, bias, clip_grad_norm,
# class_weight, early_stopping, validation_fraction, n_iter_no_change, tol, warm_start) and fit's extra arguments
# (sample_weight, h_0, c_0, validation_data, restore_best; partial_fit's one epoch from the fitted state) are read
# here ONCE per call (UNPACK) into typed words; with every option at its default the fit and predict calls are
# the ones of before, word for word (`_rnn_plain`).

@fieldwise_init
struct RnnFieldOptions(Copyable, Movable):
    var seq_out: Bool
    var dirs: Int
    var dropout: Float64
    var no_bias: Bool
    var clip: Float64
    var early_stopping: Bool
    var validation_fraction: Float64
    var n_iter_no_change: Int
    var tol: Float64
    var warm_start: Bool

    def plain(self) -> Bool:
        return (not self.seq_out and self.dirs == 1 and self.dropout == 0.0 and not self.no_bias
                and self.clip == 0.0 and not self.early_stopping and not self.warm_start)


def _field(fields: PythonObject, name: String, default: PythonObject) raises -> PythonObject:
    """A constructor field, its default when the field is absent (a model pickled before the option)."""
    return py_get(fields, PythonObject(name), default)


def rnn_field_options(fields: PythonObject) raises -> RnnFieldOptions:
    """UNPACK: the recurrent estimators' option fields as words, with their refusals."""
    var output = _field(fields, "output", PythonObject("final"))
    var seq_out = False
    if _eq(output, "sequence"):
        seq_out = True
    elif not _eq(output, "final"):
        raise Error("ValueError: output must be 'final' or 'sequence', got " + _repr(output))
    var dirs = 2 if xseq_truth(_field(fields, "bidirectional", PythonObject(False))) else 1
    var dropout = xseq_float(_field(fields, "dropout", PythonObject(0.0)))
    if not (dropout >= 0.0 and dropout < 1.0):
        raise Error("ValueError: dropout must be in [0, 1), got " + String(dropout))
    var no_bias = not xseq_truth(_field(fields, "bias", PythonObject(True)))
    var clip_v = _field(fields, "clip_grad_norm", PythonObject(None))
    var clip = 0.0
    if clip_v is not None:
        clip = xseq_float(clip_v)
        if not (clip > 0.0):
            raise Error("ValueError: clip_grad_norm must be a positive number or None")
    var fraction = xseq_float(_field(fields, "validation_fraction", PythonObject(0.1)))
    var early = xseq_truth(_field(fields, "early_stopping", PythonObject(False)))
    if early and not (fraction > 0.0 and fraction < 1.0):
        raise Error("ValueError: validation_fraction must be in (0, 1)")
    var patience = xseq_int(_field(fields, "n_iter_no_change", PythonObject(10)))
    var tol = xseq_float(_field(fields, "tol", PythonObject(1e-4)))
    if patience < 1 or not (tol >= 0.0):
        raise Error("ValueError: n_iter_no_change must be >= 1 and tol >= 0")
    var warm = xseq_truth(_field(fields, "warm_start", PythonObject(False)))
    return RnnFieldOptions(seq_out, dirs, dropout, no_bias, clip, early, fraction, patience, tol, warm)


def _rnn_f32(source: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    var a = xseq_input(source, context)
    xseq_refuse_f64(a, name + ": float64 is refused; pass float32 (no float64 on the device)")
    return admit_nd_f32(a, name, False, context)


def _rnn_predict_options_of(fields: PythonObject) raises -> RnnFieldOptions:
    """The prediction's options: output and bidirectional (two field reads; the fit-only options stay at their
    defaults here)."""
    var output = _field(fields, "output", PythonObject("final"))
    var seq_out = _eq(output, "sequence")
    if not seq_out and not _eq(output, "final"):
        raise Error("ValueError: output must be 'final' or 'sequence', got " + _repr(output))
    var dirs = 2 if xseq_truth(_field(fields, "bidirectional", PythonObject(False))) else 1
    return RnnFieldOptions(seq_out, dirs, 0.0, False, 0.0, False, 0.1, 10, 1e-4, False)


def _rnn_c0_check(h0: PythonObject, c0: PythonObject, cell: Int) raises:
    """c_0 is the LSTM's cell state and goes with h_0 (refused here on every route, before any call)."""
    if c0 is not None and h0 is None:
        raise Error("ValueError: c_0 needs h_0")
    if c0 is not None and cell != 2:
        raise Error("ValueError: c_0 is the LSTM's cell state; GRU and RNN take h_0 only")


def _rnn_state_arg(source: PythonObject, slots: Int, n: Int, hidden: Int, name: String,
                   context: PythonObject) raises -> PythonObject:
    """h_0 / c_0: None, or a float32 (num_layers * num_directions, n_samples, hidden_size) array (PyTorch's h_0
    with its batch axis over all the sequences)."""
    if source is None:
        return source
    var a = _rnn_f32(source, name, context)
    if Int(py=a.ndim) != 3 or Int(py=a.shape[0]) != slots or Int(py=a.shape[1]) != n or Int(py=a.shape[2]) != hidden:
        raise Error("ValueError: " + name + " must have shape (num_layers * num_directions, n_samples, hidden_size) = ("
                    + String(slots) + ", " + String(n) + ", " + String(hidden) + ")")
    return a


def _rnn_class_weights(fields: PythonObject, target: PythonObject, n: Int, steps: Int, lengths: PythonObject,
                       k: Int, context: PythonObject) raises -> PythonObject:
    """class_weight: None, 'balanced' or a dict {label: weight} (missing labels weigh 1), as k float32 words in
    classes_ order (scikit-learn's compute_class_weight); None when absent."""
    var cw = _field(fields, "class_weight", PythonObject(None))
    if cw is None:
        return cw
    var nk = xseq_keys()
    var values: List[Float64]
    var all_one = False
    if _eq(cw, "balanced"):
        values = xseq_run_balanced_class_weights(Int(py=xseq_addr(target)), n, steps,
                                                 0 if lengths is None else Int(py=xseq_addr(lengths)), k)
    else:
        var b = native_loaded_module("builtins")
        if not Bool(py=py_call(b.isinstance, cw, b.dict)):
            raise Error("ValueError: class_weight must be None, 'balanced' or a dict")
        # class_weight_dictionary_values walks an exact list or tuple (borrowed_item); a recurrent classifier's
        # classes_ is an array: its Python list once (w3-integrate-10: PyList_GetItem on the array, "bad argument
        # to internal function")
        var classes = py_item(fields, nk[].k_classes_)
        if py_type(classes) is not b.list and py_type(classes) is not b.tuple:
            classes = py_call(classes.tolist)
        var inputs = class_weight_dictionary_values(cw, classes, context)
        values = inputs.values.copy()
        all_one = inputs.all_one
    var plan = class_weight_plan(values, all_one)
    if plan.all_one:
        return PythonObject(None)
    var words = xseq_zeros(Python.tuple(k), context)
    var dst = xseq_f32p(words)
    for c in range(k):  # small-loop(classes)
        dst[c] = plan.words[c]
    _ = words
    return words


struct _RnnStart(Movable):
    """The parameters a recurrent fit starts from (lane stream-fits: one body for the array and the streamed fit):
    the fitted ones (warm_start, partial_fit) when they fit the layout, else a fresh seeded draw; the stream whose
    later draws seed the fit; the parameter count."""

    var params: PythonObject
    var stream: XseqStream
    var warm: Bool
    var total: Int

    def __init__(out self, var params: PythonObject, var stream: XseqStream, warm: Bool, total: Int):
        self.params = params^
        self.stream = stream^
        self.warm = warm
        self.total = total


def _rnn_start_params(model: PythonObject, fields: PythonObject, route: XseqRoute, fo: RnnFieldOptions, cell: Int,
                      d: Int, hidden: Int, layers: Int, outputs: Int, partial: Bool,
                      context: PythonObject) raises -> _RnnStart:
    """`_RnnStart` of a fit: writes params_ (a read-only view) and returns its buffer."""
    var nk = xseq_keys()
    var stream = XseqStream(py_get(fields, nk[].k_random_state), context)
    var mods = NativeModules()
    var total = xseq_run_rnn_param_count(xseq_run_rnn_gates(cell), d, hidden, layers, outputs)
    if fo.dirs == 2:
        total = _bidi_count(cell, d, hidden, layers, outputs)
    # warm_start / partial_fit: start from the fitted parameters when they fit this layout
    var params = PythonObject(None)
    var warm = (fo.warm_start or partial) and py_contains(fields, nk[].k_params_)
    if warm:
        var old = admit_nd_f32(xseq_private_input(py_item(fields, nk[].k_params_), context, "params_"), "params_",
                               False, context)
        if Int(py=old.size) == total:
            params = xseq_zeros(Python.tuple(total), context)
            unsafe_memcpy(dest=xseq_f32p(params), src=xseq_f32p(old), count=total)
            _ = old
        else:
            warm = False
    prepared_drop(mods, _RNN_PARAMS_SLOT, model)  # the new params_ is a new object
    if not warm:
        var bound = xseq_run_rnn_bound(xseq_float(py_item(fields, nk[].k_hidden_size)))
        params = xseq_zeros(Python.tuple(total), context)
        var params_address = Int(py=xseq_addr(params))
        var output_provider = xseq_neural_provider(route.output)
        var output_entry = _entry(output_provider)
        var py0 = Python()
        with GILReleased(py0):
            stream.offset = xseq_run_uniform(output_entry, params_address, total, stream.seed, stream.offset, -bound, bound)
        _ = output_provider
    py_setitem(fields, nk[].k_params_, xseq_readonly_view(params))
    return _RnnStart(params, stream^, warm, total)


def _rnn_fit_full(model: PythonObject, fields: PythonObject, x_source: PythonObject, target_in: PythonObject,
                  route: XseqRoute, context: PythonObject, lengths_in: PythonObject, sw_in: PythonObject,
                  h0_source: PythonObject, c0_source: PythonObject, validation: PythonObject,
                  vtarget_in: PythonObject, restore_best: Bool, partial: Bool) raises -> PythonObject:
    """`_RecurrentBase._fit(X, target)` with every option (lane rnn-parity). `target_in` is the admitted float32
    target, N rows (N T with output="sequence") of n_outputs_ values or one class code; `validation` the admitted
    (X_val, lengths_val) pair or None and `vtarget_in` its target likewise; `lengths_in` and lengths_val are
    admitted and checked by _rnn_fit_op (_rnn_fit_lengths) or None. `partial`: partial_fit, one epoch from the
    fitted parameters and optimizer state."""
    var fo = rnn_field_options(fields)
    var sw_source = sw_in
    var x_full = _rnn_x(x_source, fields, True, context)
    var target = target_in
    var lengths_source = lengths_in
    var x = x_full
    var n_all = Int(py=x_full.shape[0])
    var nk = xseq_keys()
    # early_stopping without validation_data: the last validation_fraction of the sequences, in their order
    var vx = PythonObject(None)
    var vlen = PythonObject(None)
    var vy = vtarget_in
    if validation is not None:
        vx = _rnn_x(validation[0], fields, False, context)
        vlen = validation[1]
    elif fo.early_stopping and not partial:
        var nv = max(1, Int(fo.validation_fraction * Float64(n_all)))
        if nv >= n_all:
            raise Error("ValueError: early_stopping needs more sequences than validation_fraction holds out")
        var cut = n_all - nv
        var b = native_loaded_module("builtins")
        var head_s = py_call(b.slice, PythonObject(0), PythonObject(cut))
        var tail_s = py_call(b.slice, PythonObject(cut), PythonObject(None))
        vx = x_full[tail_s]
        x = x_full[head_s]
        # the target holds one row per sequence, or with output="sequence" one per step (n T rows): cut at the
        # held-out sequences' first row
        var rows_per = Int(py=x_full.shape[1]) if fo.seq_out else 1
        var yhead = py_call(b.slice, PythonObject(0), PythonObject(cut * rows_per))
        var ytail = py_call(b.slice, PythonObject(cut * rows_per), PythonObject(None))
        vy = target[ytail]
        target = target[yhead]
        if sw_source is not None:
            # the held-out sequences' weights leave with them (validation is unweighted)
            sw_source = xseq_input(sw_source, context)[head_s]
        if lengths_source is not None:
            # admitted and checked by _rnn_fit_op (_rnn_fit_lengths): float32, C order, one per sequence
            vlen = lengths_source[tail_s]
            lengths_source = lengths_source[head_s]
    var n = Int(py=x.shape[0])
    var steps_t = Int(py=x.shape[1])
    xseq_run_rnn_dims_check(n, steps_t)
    var lengths = lengths_source  # admitted by _rnn_fit_op (_rnn_fit_lengths), cut above when held out
    var lengths_address = _rnn_lengths_address(lengths)
    var cell = _rnn_cell(model, fields)
    var hidden = xseq_int(py_item(fields, nk[].k_hidden_size))
    var layers = xseq_int(py_item(fields, nk[].k_num_layers))
    var d = Int(py=py_item(fields, nk[].k_n_features_in_))
    var outputs = Int(py=py_item(fields, nk[].k_n_outputs_))
    var task = Int(py=_attr(py_type(model), "_TASK"))
    var steps_y = steps_t if fo.seq_out else 1
    var sw = PythonObject(None)
    var sw_steps = False
    if sw_source is not None:
        sw = _rnn_f32(sw_source, "sample_weight", context)
        if Int(py=sw.ndim) == 2 and fo.seq_out and Int(py=sw.shape[0]) == n and Int(py=sw.shape[1]) == steps_t:
            sw_steps = True
        elif not (Int(py=sw.ndim) == 1 and Int(py=sw.shape[0]) == n):
            raise Error("ValueError: sample_weight must have shape (n_samples,)"
                        + String(" or (n_samples, n_timesteps)" if fo.seq_out else ""))
    var cw = PythonObject(None)
    if task == 1:
        cw = _rnn_class_weights(fields, target, n, steps_y, lengths if fo.seq_out else PythonObject(None),
                                outputs, context)
    elif _field(fields, "class_weight", PythonObject(None)) is not None:
        raise Error("ValueError: class_weight is for classifiers")
    var slots = layers * fo.dirs
    var h0 = _rnn_state_arg(h0_source, slots, n, hidden, "h_0", context)
    var c0 = _rnn_state_arg(c0_source, slots, n, hidden, "c_0", context)
    _rnn_c0_check(h0, c0, cell)
    var plain = fo.plain() and sw is None and cw is None and h0 is None and vx is None and not partial
    var start = _rnn_start_params(model, fields, route, fo, cell, d, hidden, layers, outputs, partial, context)
    var total = start.total
    var params = start.params
    var warm = start.warm
    var batch_size = xseq_int(py_item(fields, nk[].k_batch_size))
    var epochs = 1 if partial else xseq_int(py_item(fields, nk[].k_max_epochs))
    var schedule = xseq_run_rnn_schedule(batch_size, n, epochs)
    var seed = start.stream.child_seed()
    var shuffle = xseq_truth(py_item(fields, nk[].k_shuffle))
    var steps = schedule.steps
    var optimizer = _rnn_optimizer(py_get(fields, nk[].k_optimizer), py_get(fields, nk[].k_optimizer_options), context)
    # partial_fit's optimizer state (3 n_params + 4 words), kept on the model between calls
    var state = PythonObject(None)
    var t0 = 0
    if partial:
        var kept = _field(fields, "_optimizer_state_", PythonObject(None))
        state = xseq_zeros(Python.tuple(3 * total + 4), context)
        if kept is not None and warm and Int(py=kept.size) == 3 * total + 4:
            unsafe_memcpy(dest=xseq_f32p(state), src=xseq_f32p(kept), count=3 * total + 4)
            t0 = Int(xseq_f32p(state)[3 * total + 3])
        else:
            var sp = xseq_f32p(state)
            sp[3 * total] = Float32(1.0)
            sp[3 * total + 1] = Float32(1.0)
            sp[3 * total + 2] = Float32(1.0)
    var rates = _rnn_rates_from(fields, t0, steps, context)
    var losses = xseq_zeros(Python.tuple(steps), context)
    var request = XseqRnnFitRequest(cell, d, hidden, layers, outputs, task, n, steps_t, epochs, schedule.batch,
                                    shuffle, seed)
    var provider = xseq_neural_provider(route.entries)
    var entry = _entry(provider)
    var loop_words = fit_loop_words()
    var loop_address = Int(loop_words.unsafe_ptr())
    var vcurve = PythonObject(None)
    var nv = 0
    var vlengths = PythonObject(None)
    if vx is not None:
        nv = Int(py=vx.shape[0])
        if Int(py=vx.shape[1]) != steps_t:
            raise Error("ValueError: the validation sequences must have the training sequences' n_timesteps")
        vlengths = vlen  # admitted by _rnn_fit_op, or cut from the admitted training lengths
        vcurve = xseq_zeros(Python.tuple(epochs + 2), context)
    var addresses: List[Int] = [Int(py=xseq_addr(x)), Int(py=xseq_addr(target)), Int(py=xseq_addr(params)),
                                Int(py=xseq_addr(losses)), Int(py=_buffer_address(rates))]
    var py = Python()
    if plain:
        with GILReleased(py):
            xseq_run_rnn_fit(entry, request, optimizer, addresses[0], addresses[1], addresses[2], addresses[3],
                             addresses[4], lengths_address, loop_address)
    else:
        var o = XseqRnnFitOptions(
            fo.dirs, fo.seq_out, fo.no_bias, 0 if sw is None else Int(py=xseq_addr(sw)), sw_steps,
            0 if cw is None else Int(py=xseq_addr(cw)), 0 if h0 is None else Int(py=xseq_addr(h0)),
            0 if c0 is None else Int(py=xseq_addr(c0)), nv,
            fo.n_iter_no_change if (fo.early_stopping and nv > 0) else 0, restore_best and nv > 0,
            0 if vx is None else Int(py=xseq_addr(vx)), 0 if vy is None else Int(py=xseq_addr(vy)),
            _rnn_lengths_address(vlengths), 0 if vcurve is None else Int(py=xseq_addr(vcurve)),
            0 if state is None else Int(py=xseq_addr(state)), fo.dropout, fo.clip, fo.tol)
        with GILReleased(py):
            xseq_run_rnn_fit_options(entry, request, optimizer, addresses[0], addresses[1], addresses[2],
                                     addresses[3], addresses[4], lengths_address, loop_address, o)
    _ = lengths  # keep the Python objects alive through the native call above
    _ = vlengths
    _ = provider
    _ = target
    _ = params
    _ = x
    _ = vx
    _ = vy
    _ = sw
    _ = cw
    _ = h0
    _ = c0
    _ = state
    _ = losses
    _ = rates
    _ = x_full
    _ = route.owner  # the route's images stay loaded through the calls above
    var per = (n + schedule.batch - 1) // schedule.batch
    var run = epochs
    if vcurve is not None:
        var vp = xseq_f32p(vcurve)
        run = Int(vp[epochs])
        py_setitem(fields, PythonObject("validation_loss_"), vcurve[py_call(native_loaded_module("builtins").slice,
                                                                             PythonObject(0), PythonObject(run))])
        py_setitem(fields, PythonObject("best_epoch_"), Int(vp[epochs + 1]))
        _ = vcurve^  # keep-alive: an address helper's argument (LASTUSE helpers)
    var curve = losses
    if run != epochs:
        curve = losses[py_call(native_loaded_module("builtins").slice, PythonObject(0), PythonObject(run * per))]
    if partial:
        py_setitem(fields, PythonObject("_optimizer_state_"), state)
    py_setitem(fields, nk[].k_loss_curve_, curve)
    py_setitem(fields, nk[].k_n_iter_, (t0 if partial else 0) + run * per)
    py_setitem(fields, nk[].k_n_timesteps_, steps_t)
    py_setitem(fields, nk[].k_fit_loop_, fit_loop_pack(loop_words))
    return model


def _bidi_count(cell: Int, d: Int, hidden: Int, layers: Int, outputs: Int) -> Int:
    """The parameter count with two directions (sequence/recurrent.mojo Net with dirs 2)."""
    var g = xseq_run_rnn_gates(cell)
    var total = 0
    for l in range(layers):  # driver-loop: over recurrent layers of the parameter layout
        var din = d if l == 0 else 2 * hidden
        total += 2 * (g * hidden * din + g * hidden * hidden + 2 * g * hidden)
    return total + outputs * 2 * hidden + outputs


def _rnn_rates_from(fields: PythonObject, t0: Int, steps: Int, context: PythonObject) raises -> PythonObject:
    """_rnn_rates for the steps t0 .. t0 + steps - 1 (partial_fit continues a schedule where it stopped)."""
    if t0 == 0:
        return _rnn_rates(fields, steps, context)
    var whole = _rnn_rates(fields, t0 + steps, context)
    var b = native_loaded_module("builtins")
    var part = xseq_zeros(Python.tuple(steps), context)
    var src = XF32P(unsafe_from_address=Int(py=_buffer_address(whole)))
    unsafe_memcpy(dest=xseq_f32p(part), src=src + t0, count=steps)
    _ = whole
    _ = b
    return part


def _rnn_run(model: PythonObject, fields: PythonObject, source: PythonObject, want_seq: Bool,
             route: XseqRoute, context: PythonObject, lengths_source: PythonObject) raises -> PythonObject:
    """`_RecurrentBase._run(X, want_seq)`: `(out, seq)`, on the estimator's
    route (resolved by the caller, first, as the entries were), with the
    optional sequence lengths."""
    var x = _rnn_x(source, fields, False, context)
    var n = Int(py=x.shape[0])
    var steps_t = Int(py=x.shape[1])
    var lengths = _rnn_lengths(lengths_source, n, context)
    var lengths_address = _rnn_lengths_address(lengths)
    var nk = xseq_keys()
    var out = xseq_zeros(Python.tuple(n, py_item(fields, nk[].k_n_outputs_)), context)
    var seq = xseq_zeros(Python.tuple(n, steps_t, xseq_int(py_item(fields, nk[].k_hidden_size))) if want_seq else Python.tuple(1), context)
    # The admitted float32 view of `params_`, kept per model while `params_`
    # is the same object (see "fitted-weight records" above).
    var mods = NativeModules()
    var source_params = py_item(fields, nk[].k_params_)
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
    var request = XseqRnnPredictRequest(ints^, n_params, task, n, steps_t, xseq_int(py_item(fields, nk[].k_predict_chunk)), want_seq,
                                        Int(model._obj_ptr), token)
    var addresses: List[Int] = [Int(py=xseq_addr(x)), Int(py=xseq_addr(params)), Int(py=xseq_addr(out)), Int(py=xseq_addr(seq))]
    var provider = xseq_neural_provider(route.entries)
    var entry = _entry(provider)
    var py = Python()
    with GILReleased(py):
        xseq_run_rnn_predict(entry, request, addresses[0], addresses[1], addresses[2], addresses[3], lengths_address)
    _ = lengths  # keep the Python object alive through the native call above
    _ = provider
    _ = params  # keep the Python object alive through the native call above
    _ = seq  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    _ = x^
    _ = params^
    _ = route.owner  # the route's images stay loaded through the call above
    return Python.tuple(out, seq)


def _rnn_run_options(model: PythonObject, fields: PythonObject, source: PythonObject, want_seq: Bool,
                     route: XseqRoute, context: PythonObject, lengths_source: PythonObject, fo: RnnFieldOptions,
                     h0_source: PythonObject, c0_source: PythonObject, want_state: Bool) raises -> PythonObject:
    """_rnn_run with the options (lane rnn-parity): two directions, every step's head ((n, T, O)), initial states
    and, with `want_state` (streaming), the final states: `(out, seq, (h_n, c_n))` (c_n None but for the LSTM)."""
    var x = _rnn_x(source, fields, False, context)
    var n = Int(py=x.shape[0])
    var steps_t = Int(py=x.shape[1])
    var lengths = _rnn_lengths(lengths_source, n, context)
    var nk = xseq_keys()
    var hidden = xseq_int(py_item(fields, nk[].k_hidden_size))
    var layers = xseq_int(py_item(fields, nk[].k_num_layers))
    var outputs = Int(py=py_item(fields, nk[].k_n_outputs_))
    var slots = layers * fo.dirs
    var out = xseq_zeros(Python.tuple(n, steps_t, outputs) if fo.seq_out else Python.tuple(n, outputs), context)
    var seq = xseq_zeros(Python.tuple(n, steps_t, fo.dirs * hidden) if want_seq else Python.tuple(1), context)
    var h0 = _rnn_state_arg(h0_source, slots, n, hidden, "h_0", context)
    var c0 = _rnn_state_arg(c0_source, slots, n, hidden, "c_0", context)
    var cell = _rnn_cell(model, fields)
    _rnn_c0_check(h0, c0, cell)
    var hn = xseq_zeros(Python.tuple(slots, n, hidden), context) if want_state else PythonObject(None)
    var cn = xseq_zeros(Python.tuple(slots, n, hidden), context) if want_state and cell == 2 else PythonObject(None)
    var params = admit_nd_f32(xseq_private_input(py_item(fields, nk[].k_params_), context, "params_"), "params_",
                              False, context)
    var ints = _rnn_ints(model, fields)
    var n_params = xseq_run_rnn_layout_dirs(ints, fo.dirs, Int(py=params.size))
    var task = Int(py=_attr(py_type(model), "_TASK"))
    var request = XseqRnnPredictRequest(ints^, n_params, task, n, steps_t, xseq_int(py_item(fields, nk[].k_predict_chunk)),
                                        want_seq, Int(model._obj_ptr), 0)
    var o = XseqRnnPredictOptions(fo.dirs, fo.seq_out, 0 if h0 is None else Int(py=xseq_addr(h0)),
                                  0 if c0 is None else Int(py=xseq_addr(c0)), 0 if hn is None else Int(py=xseq_addr(hn)),
                                  0 if cn is None else Int(py=xseq_addr(cn)))
    var addresses: List[Int] = [Int(py=xseq_addr(x)), Int(py=xseq_addr(params)), Int(py=xseq_addr(out)), Int(py=xseq_addr(seq))]
    var provider = xseq_neural_provider(route.entries)
    var entry = _entry(provider)
    # the lengths address is read under the GIL (it touches the Python object), then the GIL is released
    var lengths_address = _rnn_lengths_address(lengths)
    var py = Python()
    with GILReleased(py):
        xseq_run_rnn_predict_options(entry, request, addresses[0], addresses[1], addresses[2], addresses[3],
                                     lengths_address, o)
    _ = lengths  # keep the Python objects alive through the native call above
    _ = provider
    _ = params
    _ = x
    _ = h0
    _ = c0
    _ = route.owner
    return Python.tuple(out, seq, Python.tuple(hn, cn))


def _arg(args: PythonObject, i: Int) raises -> PythonObject:
    """args[i], None past the end (callers of before send shorter tuples)."""
    return args[i] if len(args) > i else PythonObject(None)


def _codes_against(classes: PythonObject, source: PythonObject, route: XseqRoute, n: Int, steps: Int,
                   lengths: PythonObject, context: PythonObject) raises -> PythonObject:
    """The float32 codes of labels `source` (n sequences of `steps` labels; steps 1 for one label a sequence)
    against the FITTED `classes` (partial_fit after the first call, a validation target): the label encoder's own
    codes mapped through each of its classes' index in `classes` (equality of the few class labels: metadata); a
    label at a valid step that is not a fitted class raises. With `lengths` (admitted and checked; output=
    "sequence") only valid steps count (lane rnn-review-fixes, P1-b): a label seen only on padded steps is not
    looked up, and every padded step gets code 0, which the loss weighs 0."""
    var encoded = xseq_labels(source, route.entries, context)
    var own = encoded[0]
    var codes = encoded[1]
    var count = Int(py=codes.shape[0])
    if count != n * steps:
        raise Error("ValueError: y must hold " + String(n * steps) + " labels")
    var seen = List[Int]()
    if lengths is not None:
        seen = xseq_run_class_counts[DType.int32](Int(py=xseq_addr(codes)), n, steps, Int(py=xseq_addr(lengths)),
                                                  len(own))
    var mapping = List[Int]()
    for j in range(len(own)):  # small-loop(classes)
        if lengths is not None and seen[j] == 0:
            mapping.append(-1)  # on padded steps only: never read (xseq_run_remap_codes writes 0 there)
            continue
        var at = -1
        for i in range(len(classes)):  # small-loop(classes)
            if Bool(py=own[j] == classes[i]):
                at = i
                break
        if at < 0:
            raise Error("ValueError: y holds the label " + _repr(own[j]) + ", which is not one of the fitted classes "
                        + _repr(classes))
        mapping.append(at)
    var target = xseq_zeros(Python.tuple(count), context)
    xseq_run_remap_codes(Int(py=xseq_addr(codes)), n, steps, _rnn_lengths_address(lengths), mapping,
                         Int(py=xseq_addr(target)))
    _ = codes
    _ = lengths
    return target


def _rnn_target(model: PythonObject, fields: PythonObject, y_source: PythonObject, n: Int, steps: Int,
                seq_out: Bool, keep_classes: Bool, route: XseqRoute, context: PythonObject,
                fitting: Bool, lengths: PythonObject) raises -> PythonObject:
    """The float32 training target: regression (n,) / (n, O), or with output="sequence" (n, T) / (n, T, O), as N S
    rows of O words; classification: N S class codes (labels (n,), or (n, T) per step). `fitting` writes
    n_outputs_ / _y_1d / classes_; `keep_classes` maps against the fitted classes_ instead of encoding anew.
    `lengths`: the admitted, checked sequence lengths with output="sequence", else None. Classification then
    reads labels at VALID steps only (lane rnn-review-fixes, P1-b): classes_ (and the head's width n_outputs_)
    are the labels of valid steps, padded steps get code 0 (weighed 0 by the loss), so a label used only as
    padding is neither a class nor refused as unknown."""
    var nk = xseq_keys()
    var task = Int(py=_attr(py_type(model), "_TASK"))
    var rows = n * steps
    if task == 0:
        var y = _rnn_f32(y_source, "y", context)
        var rank = Int(py=y.ndim)
        var base = 2 if seq_out else 1
        if rank < base or rank > base + 1 or Int(py=y.shape[0]) != n or (seq_out and Int(py=y.shape[1]) != steps):
            raise Error("ValueError: y must have shape " + String("(n_samples, n_timesteps[, n_outputs])" if seq_out
                        else "(n_samples[, n_outputs])"))
        var target = xseq_reshape(y, Python.tuple(rows, -1))
        if fitting:
            py_setitem(fields, nk[].k__y_1d, rank == base)
            py_setitem(fields, nk[].k_n_outputs_, Int(py=target.shape[1]))
        elif Int(py=target.shape[1]) != Int(py=py_item(fields, nk[].k_n_outputs_)):
            raise Error("ValueError: y must have the fitted n_outputs_")
        return target
    var flat = y_source
    if seq_out:
        var y = xseq_input(y_source, context)
        if Int(py=y.ndim) != 2 or Int(py=y.shape[0]) != n or Int(py=y.shape[1]) != steps:
            raise Error("ValueError: y must have shape (n_samples, n_timesteps): one label per step")
        flat = xseq_reshape(y, Python.tuple(rows))
    var valid_only = seq_out and lengths is not None
    var lens = lengths if valid_only else PythonObject(None)
    if keep_classes:
        return _codes_against(py_item(fields, nk[].k_classes_), flat, route, n, steps, lens, context)
    var encoded = xseq_labels(flat, route.entries, context)
    var own = encoded[0]
    var codes = encoded[1]
    var count = Int(py=codes.shape[0])
    if not valid_only:
        py_setitem(fields, nk[].k_classes_, own)
        var k = len(own)
        if k < 2:
            raise Error("classification needs at least two classes")
        py_setitem(fields, nk[].k_n_outputs_, k)
        var target = xseq_zeros(Python.tuple(count), context)
        xseq_codes_f32(XI32P(unsafe_from_address=Int(py=xseq_addr(codes))), xseq_f32p(target), count)
        _ = codes
        return target
    # the classes of the valid steps, in the encoder's order; mapping[j]: own class j's index among them
    var seen = xseq_run_class_counts[DType.int32](Int(py=xseq_addr(codes)), n, steps, Int(py=xseq_addr(lens)),
                                                  len(own))
    var mapping = List[Int]()
    var kept = Python.list()
    var k = 0
    for j in range(len(own)):  # small-loop(classes)
        if seen[j] == 0:
            mapping.append(-1)  # on padded steps only: never read (xseq_run_remap_codes writes 0 there)
            continue
        mapping.append(k)
        _ = py_call(kept.append, PythonObject(j))
        k += 1
    py_setitem(fields, nk[].k_classes_, own[kept] if k != len(own) else own)
    if k < 2:
        raise Error("classification needs at least two classes")
    py_setitem(fields, nk[].k_n_outputs_, k)
    var target = xseq_zeros(Python.tuple(count), context)
    xseq_run_remap_codes(Int(py=xseq_addr(codes)), n, steps, Int(py=xseq_addr(lens)), mapping,
                         Int(py=xseq_addr(target)))
    _ = codes
    _ = lens
    return target


comptime _XI64P = MutPointer[Int64, MutUntrackedOrigin]


def _rnn_fit_source(model: PythonObject, fields: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """fit(source=) and partial_fit(source=, steps=) of the recurrent estimators (lane stream-fits, 2026-11-25).
    UNPACK: the exact BatchSource handle's words (providers/python/batch_source_pyapi_door.mojo
    batch_source_spec: opened with inputs=("steps",), target="targets"), the step count, the parameters
    (_rnn_start_params, as an array fit), the optimizer state of partial_fit, and the position to train from: the
    estimator's saved source part (`_source_state_`) when partial_fit continues, else the handle's. A classifier's
    classes are the source's class indices (classes_ = 0 .. K - 1, K scanned once from the targets on its first
    fit). RUN: one RNN_FIT call in its streamed form (sequence/xseq_neural_run.mojo xseq_run_rnn_fit_stream): the
    training image runs sequence/recurrent.mojo rnn_fit_stream in place on the source the handle's owner keeps
    (its address; opened by the first fit, reused by the later ones). PACK: params_, the per-step
    loss_curve_, the optimizer state, `_source_state_` (the consumed position and the source's binding words, so a
    pickled estimator resumes in a fresh process on the same next batch) and the consumed position on the handle.
    fit trains max_epochs epochs of the source from its position; partial_fit the rest of the current epoch, or
    exactly `steps` batches."""
    var nk = xseq_keys()
    var fo = rnn_field_options(fields)
    var route = xseq_route(model, fields, context)
    var partial = xseq_truth(_arg(args, 8))
    if args[0] is not None or args[1] is not None:
        raise Error("TypeError: fit(source=) takes no X or y: the source holds the sequences and their targets")
    if _arg(args, 2) is not None:
        raise Error("TypeError: fit(source=) takes no lengths: they are the source's record lengths")
    var source = args[9]
    var spec = batch_source_spec(source)  # the exact connected BatchSource only (no Python iterable or callback)
    if len(spec.inputs) != 1 or spec.inputs[0] != BS_FIELD_STEPS or spec.target != BS_FIELD_TARGETS:
        raise Error("ValueError: fit(source=) reads a BatchSource opened with inputs=('steps',) and target='targets'")
    # the source the handle's owner keeps (lane w3i11-source-keep; autograd/batch_source.mojo "the streamed fits"):
    # opened here once, reused by every later fit and partial_fit while it stands at the handle's position; the
    # training image uses it in place through its address. `hold` keeps the owner object alive through the call.
    var hold = batch_source_hold[False, BS_DOOR_RECURRENT](source, spec.copy())
    var max_steps = 0
    var steps_in = _arg(args, 10)
    if steps_in is not None:
        if not is_integer_input(steps_in) or Int(py=steps_in) < 1:
            raise Error("ValueError: steps must be a positive integer or None")
        max_steps = Int(py=steps_in)
    var task = Int(py=_attr(py_type(model), "_TASK"))
    var width = Int(py=source.n_features)
    var longest = Int(py=source.max_length)
    var n = Int(py=source.n_records)
    var nb = Int(py=source.n_batches)
    var tw = Int(py=source.target_width)
    var rows_per = longest if fo.seq_out else 1
    xseq_run_rnn_dims_check(n, longest)
    var fitted = partial and py_contains(fields, nk[].k_params_) and py_contains(fields, nk[].k_n_outputs_)
    if fitted and width != Int(py=py_item(fields, nk[].k_n_features_in_)):
        raise Error("ValueError: the source's records have " + String(width) + " features; the model was fitted on "
                    + String(py=py_call(native_loaded_module("builtins").str, py_item(fields, nk[].k_n_features_in_))))
    # partial_fit continues at the position its state saved, on the same sealed data, plan and fields
    if fitted:
        var kept = _field(fields, "_source_state_", PythonObject(None))
        if kept is not None:
            var part = source_part_load(_XI64P(unsafe_from_address=Int(py=xseq_addr(kept))))
            _ = kept
            if part.present:
                if not spec.sealed:
                    raise Error("ValueError: partial_fit(source=): the shard index under " + spec.path + " records no "
                                + "content digest, so a resume could not tell these shards from other data of the "
                                + "same shape; call BatchSource.seal(path) once after writing the shards")
                program_training_source_refuse(batch_source_part_of(spec), part, "partial_fit(source=)")
                spec.epoch = part.epoch
                spec.consumed = part.consumed
    # the steps of this call: from the pending batch, max_epochs epochs (fit), the rest of the epoch (partial_fit)
    # or exactly `steps`
    var first = spec.consumed if spec.consumed < nb else 0
    var total = max_steps
    if total == 0:
        total = nb - first
        if not partial:
            total += (xseq_int(py_item(fields, nk[].k_max_epochs)) - 1) * nb
    # the outputs: a regressor's target row is n_outputs_ words a step; a classifier's one class index a step
    var outputs = 0
    if task == 0:
        if tw < 1 or tw % rows_per != 0:
            raise Error("ValueError: fit(source=): each record's target row must hold n_outputs words"
                        + (String(" a step, padded to the longest record") if fo.seq_out else String(""))
                        + "; the source's targets have width " + String(tw))
        outputs = tw // rows_per
        if fitted and outputs != Int(py=py_item(fields, nk[].k_n_outputs_)):
            raise Error("ValueError: the source's targets have another n_outputs_ than the fitted model")
        py_setitem(fields, nk[].k__y_1d, outputs == 1)
    else:
        if tw != rows_per:
            raise Error("ValueError: fit(source=): a classifier reads one class index a record"
                        + (String(" a step, padded to the longest record") if fo.seq_out else String(""))
                        + "; the source's targets have width " + String(tw))
        if fitted:
            outputs = Int(py=py_item(fields, nk[].k_n_outputs_))
        else:
            # the one class admission of this opened source (the training image's check answers from it)
            outputs = batch_source_class_scan(hold.src[], 0, "fit(source=)")
            if outputs < 2:
                raise Error("classification needs at least two classes")
            var classes = xseq_zeros_i64(Python.tuple(outputs), context)
            var cp = _XI64P(unsafe_from_address=Int(py=xseq_addr(classes)))
            for k in range(outputs):  # small-loop(classes: the class labels, PACK)
                cp[k] = Int64(k)
            py_setitem(fields, nk[].k_classes_, classes)
    py_setitem(fields, nk[].k_n_outputs_, outputs)
    py_setitem(fields, nk[].k_n_features_in_, width)
    # CLASS WEIGHTS (lanes fable-finalize, stream-fits-2): a dict {class: weight}, one weight a class in classes_
    # order, applied to every record (to every valid step with output="sequence") by its own label, batch by batch
    # (sequence/recurrent.mojo rnn_fit_stream); 'balanced' is counted by the training image in its one admission
    # read of the source's targets (RNN_STREAM_BALANCED; sequence/ops_impl.mojo), valid steps only.
    var cw = PythonObject(None)
    var stream_flags = 0
    var cw_in = _field(fields, "class_weight", PythonObject(None))
    if cw_in is not None:
        if task != 1:
            raise Error("ValueError: class_weight is for classifiers")
        if _eq(cw_in, "balanced"):
            stream_flags = RNN_STREAM_BALANCED
        else:
            cw = _rnn_class_weights(fields, PythonObject(None), n, rows_per, PythonObject(None), outputs, context)
    # SAMPLE WEIGHTS AND INITIAL STATES (lane stream-fits-2): one per RECORD, the records numbered over the shards in
    # index order (BatchSource.n_records): sample_weight (n_records,) or with output="sequence" (n_records,
    # max_length); h_0 / c_0 (num_layers * num_directions, n_records, hidden_size). Each batch gathers its records'.
    var sw = PythonObject(None)
    var sw_steps = False
    if _arg(args, 3) is not None:
        sw = _rnn_f32(args[3], "sample_weight", context)
        if Int(py=sw.ndim) == 2 and fo.seq_out and Int(py=sw.shape[0]) == n and Int(py=sw.shape[1]) == longest:
            sw_steps = True
        elif not (Int(py=sw.ndim) == 1 and Int(py=sw.shape[0]) == n):
            raise Error("ValueError: fit(source=): sample_weight must have shape (n_records,) = (" + String(n) + ",)"
                        + (String(" or (n_records, max_length) = (" + String(n) + ", " + String(longest) + ")")
                           if fo.seq_out else String("")) + ": one weight a record of the source")
    # VALIDATION PER PASS (lanes fable-finalize, stream-fits-2): validation_data=(X_val, y_val, lengths_val) arrays,
    # padded to the source's max_length; their loss is taken at the end of every pass over the source this call
    # completes (validation_loss_, best_epoch_). early_stopping stops after n_iter_no_change passes without an
    # improvement of tol; restore_best (fit) keeps the best pass's parameters. The monitor carries across
    # partial_fit calls (`_stream_monitor_`), so a resumed run stops where the uninterrupted one does.
    var vx = PythonObject(None)
    var vy = PythonObject(None)
    var vlengths = PythonObject(None)
    var nv = 0
    var vd = _arg(args, 6)
    if vd is not None:
        if len(vd) != 2 and len(vd) != 3:
            raise Error("ValueError: validation_data must be (X_val, y_val) or (X_val, y_val, lengths_val)")
        if len(vd) != 3 or vd[2] is None:
            raise Error("ValueError: fit(source=): validation_data needs the validation sequences' lengths, "
                        + "(X_val, y_val, lengths_val): the source's records are masked by theirs")
        vx = _rnn_x(vd[0], fields, False, context)
        nv = Int(py=vx.shape[0])
        if Int(py=vx.shape[1]) != longest:
            raise Error("ValueError: fit(source=): the validation sequences must be padded to the source's "
                        + "max_length (" + String(longest) + " steps)")
        vlengths = _rnn_fit_lengths(vd[2] if len(vd) == 3 else PythonObject(None), nv, longest, "validation",
                                    context, True)
        vy = _rnn_target(model, fields, vd[1], nv, rows_per, fo.seq_out, True, route, context, False, vlengths)
    if fo.early_stopping and nv == 0:
        raise Error("ValueError: fit(source=) with early_stopping needs validation_data=(X_val, y_val, lengths_val): "
                    + "no part of a stream is held out (validation_fraction holds out the end of an array)")
    var restore = (not partial) and nv > 0 and (True if _arg(args, 7) is None else xseq_truth(args[7]))
    var cell = _rnn_cell(model, fields)
    var hidden = xseq_int(py_item(fields, nk[].k_hidden_size))
    var layers = xseq_int(py_item(fields, nk[].k_num_layers))
    var slots = layers * fo.dirs
    var h0 = _rnn_state_arg(_arg(args, 4), slots, n, hidden, "h_0", context)
    var c0 = _rnn_state_arg(_arg(args, 5), slots, n, hidden, "c_0", context)
    _rnn_c0_check(h0, c0, cell)
    var start = _rnn_start_params(model, fields, route, fo, cell, width, hidden, layers, outputs, partial, context)
    var n_params = start.total
    var params = start.params
    # the dropout masks' seed from the estimator's random_state alone (a fresh stream, not the one the parameter
    # draw advanced), so a partial_fit that continues from saved parameters keys its masks as the uninterrupted run
    var mask_stream = XseqStream(py_get(fields, nk[].k_random_state), context)
    var seed = mask_stream.child_seed()
    var optimizer = _rnn_optimizer(py_get(fields, nk[].k_optimizer), py_get(fields, nk[].k_optimizer_options), context)
    var state = PythonObject(None)
    var t0 = 0
    if partial:
        var kept_state = _field(fields, "_optimizer_state_", PythonObject(None))
        state = xseq_zeros(Python.tuple(3 * n_params + 4), context)
        if kept_state is not None and start.warm and Int(py=kept_state.size) == 3 * n_params + 4:
            unsafe_memcpy(dest=xseq_f32p(state), src=xseq_f32p(kept_state), count=3 * n_params + 4)
            t0 = Int(xseq_f32p(state)[3 * n_params + 3])
        else:
            var sp = xseq_f32p(state)
            sp[3 * n_params] = Float32(1.0)
            sp[3 * n_params + 1] = Float32(1.0)
            sp[3 * n_params + 2] = Float32(1.0)
    var rates = _rnn_rates_from(fields, t0, total, context)
    var losses = xseq_zeros(Python.tuple(total), context)
    # the passes over the source this call completes (each ends with the validation loss); the request's epochs is
    # the validation curve's length (at least 1: the run side sizes its words by it)
    var passes = (first + total) // nb
    var v_epochs = max(1, passes)
    var vcurve = PythonObject(None)
    if nv > 0:
        # the passes' losses, the two result words, then the monitor words (sequence/recurrent.mojo
        # RNN_STREAM_MONITOR_WORDS): partial_fit continues the monitor an earlier call left
        vcurve = xseq_zeros(Python.tuple(v_epochs + 2 + RNN_STREAM_MONITOR_WORDS), context)
        var carried = _field(fields, "_stream_monitor_", PythonObject(None)) if fitted else PythonObject(None)
        if carried is not None and Int(py=carried.size) == RNN_STREAM_MONITOR_WORDS:
            var cm = _rnn_f32(carried, "_stream_monitor_", context)
            unsafe_memcpy(dest=xseq_f32p(vcurve) + v_epochs + 2, src=xseq_f32p(cm), count=RNN_STREAM_MONITOR_WORDS)
            _ = cm
    var request = XseqRnnFitRequest(cell, width, hidden, layers, outputs, task, n, longest, v_epochs, spec.batch,
                                    spec.shuffle, seed)
    var o = XseqRnnFitOptions(
        fo.dirs, fo.seq_out, fo.no_bias, 0 if sw is None else Int(py=xseq_addr(sw)), sw_steps,
        0 if cw is None else Int(py=xseq_addr(cw)), 0 if h0 is None else Int(py=xseq_addr(h0)),
        0 if c0 is None else Int(py=xseq_addr(c0)), nv,
        fo.n_iter_no_change if (fo.early_stopping and nv > 0) else 0, restore,
        0 if vx is None else Int(py=xseq_addr(vx)), 0 if vy is None else Int(py=xseq_addr(vy)),
        _rnn_lengths_address(vlengths), 0 if vcurve is None else Int(py=xseq_addr(vcurve)),
        0 if state is None else Int(py=xseq_addr(state)), fo.dropout, fo.clip, fo.tol)
    var provider = xseq_neural_provider(route.entries)
    var entry = _entry(provider)
    var loop_words = fit_loop_words()
    # the position to train from (partial_fit continuing its saved one; else the handle's, where the owner stands)
    hold.src[].seek(spec.epoch, spec.consumed)
    var words = batch_source_fit_words(hold.src, total)
    var saved = xseq_zeros_i64(Python.tuple(SOURCE_PART_WORDS), context)
    var position = List[Int64](length=RNN_STREAM_POSITION_WORDS, fill=Int64(0))
    var params_address = Int(py=xseq_addr(params))
    var losses_address = Int(py=xseq_addr(losses))
    var rates_address = Int(py=_buffer_address(rates))
    var saved_address = Int(py=xseq_addr(saved))
    var py = Python()
    with GILReleased(py):
        xseq_run_rnn_fit_stream(entry, request, optimizer, params_address, losses_address, rates_address,
                                Int(loop_words.unsafe_ptr()), o, words, saved_address, Int(position.unsafe_ptr()),
                                stream_flags)
    _ = hold^  # keep the source's owner and every buffer alive through the native call above
    _ = sw
    _ = h0
    _ = c0
    _ = provider
    _ = params
    _ = state
    _ = losses
    _ = rates
    _ = saved
    _ = cw
    _ = vx
    _ = vy
    _ = vlengths
    _ = route.owner  # the route's images stay loaded through the call above
    if vcurve is not None:
        var vp = xseq_f32p(vcurve)
        var vran = Int(vp[v_epochs])
        py_setitem(fields, PythonObject("validation_loss_"), vcurve[py_call(native_loaded_module("builtins").slice,
                                                                             PythonObject(0), PythonObject(vran))])
        py_setitem(fields, PythonObject("best_epoch_"), Int(vp[v_epochs + 1]))
        # the monitor the next partial_fit(source=) continues
        var monitor = xseq_zeros(Python.tuple(RNN_STREAM_MONITOR_WORDS), context)
        unsafe_memcpy(dest=xseq_f32p(monitor), src=vp + v_epochs + 2, count=RNN_STREAM_MONITOR_WORDS)
        py_setitem(fields, PythonObject("_stream_monitor_"), monitor)
        _ = vcurve  # vp reads its storage: keep it alive past the last read
    else:
        py_setitem(fields, PythonObject("_stream_monitor_"), PythonObject(None))
    batch_source_position_out(source, Int(position[0]), Int(position[1]))
    var ran = Int(position[2])
    _ = position^
    var curve = losses
    if ran != total:
        # early stopping ended the call after `ran` steps
        curve = losses[py_call(native_loaded_module("builtins").slice, PythonObject(0), PythonObject(ran))]
    if partial:
        py_setitem(fields, PythonObject("_optimizer_state_"), state)
    py_setitem(fields, PythonObject("_source_state_"), saved)
    py_setitem(fields, nk[].k_loss_curve_, curve)
    py_setitem(fields, nk[].k_n_iter_, (t0 if partial else 0) + ran)
    py_setitem(fields, nk[].k_n_timesteps_, longest)
    py_setitem(fields, nk[].k_fit_loop_, fit_loop_pack(loop_words))
    return model


def _rnn_fit_op(model: PythonObject, fields: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """fit(X, y, lengths, sample_weight, h_0, c_0, validation_data, restore_best, partial) (lane rnn-parity:
    every argument after lengths optional; partial is partial_fit's one epoch). The lengths (training and
    validation) are admitted and checked here once, before the target encoder or the class histogram reads one
    (lane rnn-review-fixes)."""
    if _arg(args, 9) is not None:
        return _rnn_fit_source(model, fields, args, context)  # lane stream-fits: fit(source=) / partial_fit(source=)
    var fo = rnn_field_options(fields)
    var route = xseq_route(model, fields, context)
    var partial = xseq_truth(_arg(args, 8))
    var nk = xseq_keys()
    var x = _rnn_x(args[0], fields, not (partial and py_contains(fields, nk[].k_params_)), context)
    var n = Int(py=x.shape[0])
    var steps_t = Int(py=x.shape[1])
    var steps = steps_t if fo.seq_out else 1
    var lengths = _rnn_fit_lengths(_arg(args, 2), n, steps_t, "fit", context,
                                   not _rnn_side(route, RNN_REVIEW_SABOTAGE_LENGTHS_DEVICE,
                                                 RNN_REVIEW_SABOTAGE_LENGTHS_HOST))
    # the target's labels at valid steps only; a sabotaged side reads every step (RNN_REVIEW_SABOTAGE_PADDING_*)
    var pad_fault = _rnn_side(route, RNN_REVIEW_SABOTAGE_PADDING_DEVICE, RNN_REVIEW_SABOTAGE_PADDING_HOST)
    var keep = partial and py_contains(fields, nk[].k_classes_)
    var target = _rnn_target(model, fields, args[1], n, steps, fo.seq_out, keep, route, context, not partial or
                             not py_contains(fields, nk[].k_n_outputs_),
                             PythonObject(None) if pad_fault else lengths)
    var validation = PythonObject(None)
    var vtarget = PythonObject(None)
    var vd = _arg(args, 6)
    if vd is not None:
        if len(vd) != 2 and len(vd) != 3:
            raise Error("ValueError: validation_data must be (X_val, y_val) or (X_val, y_val, lengths_val)")
        var vx = _rnn_x(vd[0], fields, False, context)
        var nv = Int(py=vx.shape[0])
        var vsteps = Int(py=vx.shape[1])
        var vlengths = _rnn_fit_lengths(vd[2] if len(vd) == 3 else PythonObject(None), nv, vsteps, "validation",
                                        context, True)
        vtarget = _rnn_target(model, fields, vd[1], nv, vsteps if fo.seq_out else 1, fo.seq_out, True,
                              route, context, False, PythonObject(None) if pad_fault else vlengths)
        validation = Python.tuple(vd[0], vlengths)
    var restore = True if _arg(args, 7) is None else xseq_truth(args[7])
    # every option at its default runs the RNN_FIT call of before (_rnn_fit_full's `plain`)
    return _rnn_fit_full(model, fields, args[0], target, route, context, lengths, _arg(args, 3), _arg(args, 4),
                         _arg(args, 5), validation, vtarget, restore, partial)


def xseq_rnn(model: PythonObject, operation: Int, args: PythonObject, context: PythonObject) raises -> PythonObject:
    # bw2-sequence (2026-11-20): `operation` is the xseq code (bindings/xseq_keys_unpack.mojo); it was
    # the name String, compared per call. A code that is not one of these (an unknown `rnn_` name
    # is -1) predicts, as the name chain did.
    var fields = py_fields(model)
    if operation == XSEQ_OPN_RNN_FIT:
        return _rnn_fit_op(model, fields, args, context)
    # predictions: (X, lengths, h_0, c_0, want_state) (lane rnn-parity: every word after lengths optional)
    var lengths_source = _arg(args, 1)
    var route = xseq_route(model, fields, context)
    # the two words a prediction needs (output, bidirectional), not the fit's whole option set
    var fo = _rnn_predict_options_of(fields)
    var h0 = _arg(args, 2)
    var c0 = _arg(args, 3)
    var want_state = xseq_truth(_arg(args, 4))
    if want_state and h0 is not None:
        # step(X, state): the state is the (h_n, c_n) pair a previous step returned
        var b = native_loaded_module("builtins")
        if not Bool(py=py_call(b.isinstance, h0, b.tuple)) or len(h0) != 2:
            raise Error("TypeError: state must be the (h, c) pair step returned, or None")
        c0 = h0[1]
        h0 = h0[0]
    if c0 is not None and h0 is None:
        raise Error("ValueError: c_0 needs h_0")
    var options = not (fo.seq_out == False and fo.dirs == 1 and h0 is None and not want_state)
    if operation == XSEQ_OPN_RNN_HIDDEN_SEQUENCE:
        if options:
            return _rnn_run_options(model, fields, args[0], True, route, context, lengths_source, fo, h0, c0, False)[1]
        return _rnn_run(model, fields, args[0], True, route, context, lengths_source)[1]
    var nk = xseq_keys()
    var result = (_rnn_run_options(model, fields, args[0], False, route, context, lengths_source, fo, h0, c0, want_state)
                  if options else _rnn_run(model, fields, args[0], False, route, context, lengths_source))
    var out = result[0]
    var task = Int(py=_attr(py_type(model), "_TASK"))
    var n = Int(py=out.shape[0])
    var lead = Python.tuple(n, Int(py=out.shape[1])) if fo.seq_out else Python.tuple(n)
    var value: PythonObject
    if operation == XSEQ_OPN_RNN_PREDICT_PROBA or (want_state and task == 1):
        value = out
    elif task == 0:
        value = out
        if xseq_truth(py_get(fields, nk[].k__y_1d, PythonObject(False))):
            value = xseq_reshape(out, lead)
    else:
        var k = Int(py=out.shape[len(out.shape) - 1])
        var rows = Int(py=out.size) // k
        var codes = xseq_argmax(out, rows, k, route.output, context)
        value = xseq_take(xseq_public_labels(fields, context), codes, context)
        if fo.seq_out:
            value = xseq_reshape(value, lead)
    if want_state:
        return Python.tuple(value, result[2])
    return value


def xseq_rnn_sequence_score(model: PythonObject, prediction: PythonObject, args: PythonObject,
                            context: PythonObject) raises -> PythonObject:
    """score with output="sequence" (lane rnn-parity): `(prediction, args)` flattened to one row per step, and the
    steps' weights (each sequence's valid steps 1, its padded steps 0, times sample_weight per sequence or per
    step) as the score's sample_weight, so R^2 or the accuracy covers every valid step. None when the model's
    output is 'final' (the score of before)."""
    var fields = py_fields(model)
    if not _eq(_field(fields, "output", PythonObject("final")), "sequence"):
        return PythonObject(None)
    var n = Int(py=prediction.shape[0])
    var steps = Int(py=prediction.shape[1])
    var rows = n * steps
    var lead = Python.tuple(rows) if Int(py=prediction.ndim) == 2 else Python.tuple(rows, -1)
    var pred = xseq_reshape(prediction, lead)
    var y = xseq_input(args[1], context)
    var ylead = Python.tuple(rows) if Int(py=y.ndim) == 2 else Python.tuple(rows, -1)
    var yf = xseq_reshape(y, ylead)
    var lengths = _rnn_lengths(_arg(args, 3), n, context)
    var sw = PythonObject(None)
    var sw_steps = False
    if _arg(args, 2) is not None:
        sw = _rnn_f32(args[2], "sample_weight", context)
        sw_steps = Int(py=sw.ndim) == 2
    var weights = PythonObject(None)
    if lengths is not None or sw is not None:
        weights = xseq_zeros(Python.tuple(rows), context)
        xseq_run_step_weights(n, steps, _rnn_lengths_address(lengths), 0 if sw is None else Int(py=xseq_addr(sw)),
                              sw_steps, Int(py=xseq_addr(weights)))
    _ = lengths
    _ = sw
    return Python.tuple(pred, Python.tuple(args[0], yf, weights))


# ---------------------------------------------------------------- multi-layer perceptrons

def _mlp_check(fields: PythonObject, context: PythonObject) raises:
    """`_BaseMLP._check()`."""
    var nk = xseq_keys()
    var solver = py_get(fields, nk[].k_solver)
    if _eq(solver, "lbfgs"):
        raise Error("NotImplementedError: solver='lbfgs' is not implemented (adam and sgd are)")
    if not py_contains(py_item(context, nk[].k_xseq_mlp_solvers), solver):
        raise Error("solver must be one of ['adam', 'sgd']")
    if not py_contains(py_item(context, nk[].k_xseq_mlp_activations), py_get(fields, nk[].k_activation)):
        raise Error("activation must be one of ['identity', 'logistic', 'relu', 'tanh']")
    if not py_contains(py_item(context, nk[].k_xseq_mlp_rates), py_get(fields, nk[].k_learning_rate)):
        raise Error("learning_rate must be one of ['adaptive', 'constant', 'invscaling']")
    for name in ["early_stopping", "warm_start", "verbose"]:  # driver-loop: over three named flag arguments
        if xseq_truth(py_get(fields, PythonObject(name))):
            raise Error("NotImplementedError: " + String(name) + "=True is not implemented")
    xseq_run_mlp_scalars_check(xseq_float(py_item(fields, nk[].k_learning_rate_init)), xseq_float(py_item(fields, nk[].k_max_iter)),
                               xseq_float(py_item(fields, nk[].k_alpha)))


def _mlp_hidden(fields: PythonObject, context: PythonObject) raises -> List[Int]:
    """`_BaseMLP._hidden()`."""
    var b = native_loaded_module("builtins")
    var nk = xseq_keys()
    var h = py_item(fields, nk[].k_hidden_layer_sizes)
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
    var nk = xseq_keys()
    var stream = XseqStream(py_get(fields, nk[].k_random_state), context)
    var mods = NativeModules()
    prepared_drop(mods, _MLP_PACKED_SLOT, model)  # the refit publishes new coefs_ / intercepts_
    var logistic = _eq(py_item(fields, nk[].k_activation), "logistic")
    # Glorot-uniform per layer, drawn straight into the flat block the
    # entry trains, w then b of each layer, as the packed layout orders them
    var total = xseq_run_mlp_param_count(sizes)
    var params = xseq_zeros(Python.tuple(total), context)
    var base = Int(py=xseq_addr(params))
    # The estimator's route: the output image draws the weights, the fit
    # entry below is the route's own.
    var route = xseq_route(model, fields, context)
    var output_provider = xseq_neural_provider(route.output)
    var output_entry = _entry(output_provider)
    var py = Python()
    with GILReleased(py):
        stream.offset = xseq_run_mlp_glorot(output_entry, base, sizes, logistic, stream.seed, stream.offset)
    _ = output_provider
    _ = params  # keep the Python object alive through the native calls above
    var seed = Int(stream.child_seed() % UInt64(2147483647))
    var batch: Int
    var requested = py_item(fields, nk[].k_batch_size)
    if _eq(requested, "auto"):
        batch = min(200, rows)
    else:
        var clipped = py_call(b.max, PythonObject(1), py_call(b.min, requested, PythonObject(rows)))
        batch = xseq_int(clipped)
    py_setitem(fields, nk[].k_n_outputs_, o)
    py_setitem(fields, nk[].k_n_layers_, len(sizes))
    py_setitem(fields, nk[].k_out_activation_, PythonObject(xseq_run_mlp_out_name(out_act)))
    py_setitem(fields, nk[].k_n_features_in_, d)
    var max_iter = xseq_int(py_item(fields, nk[].k_max_iter))
    var curve = xseq_zeros(Python.tuple(max_iter), context)
    # the descriptor's int and float words, native (no Python list built
    # and converted back word by word)
    var activation = Int(py=py_item(context, nk[].k_xseq_mlp_activations)[py_item(fields, nk[].k_activation)])
    var loss_code = Int(py=py_item(fields, nk[].k__LOSS_CODE) if py_contains(fields, nk[].k__LOSS_CODE) else _attr(py_type(model), "_LOSS_CODE"))
    var solver = Int(py=py_item(context, nk[].k_xseq_mlp_solvers)[py_item(fields, nk[].k_solver)])
    var rate_code = Int(py=py_item(context, nk[].k_xseq_mlp_rates)[py_item(fields, nk[].k_learning_rate)])
    var nesterov = xseq_truth(py_item(fields, nk[].k_nesterovs_momentum))
    var shuffle = xseq_truth(py_item(fields, nk[].k_shuffle))
    var request = XseqMlpFitRequest(rows, d, o, activation, out_act, loss_code, solver, rate_code, nesterov, batch,
                                    max_iter, shuffle, seed, xseq_int(py_item(fields, nk[].k_n_iter_no_change)), hidden.copy(), n_out)
    var fp: List[Float64] = [_opt_real(py_item(fields, nk[].k_learning_rate_init)), _opt_real(py_item(fields, nk[].k_beta_1)),
                             _opt_real(py_item(fields, nk[].k_beta_2)), _opt_real(py_item(fields, nk[].k_epsilon)), _opt_real(py_item(fields, nk[].k_momentum)),
                             _opt_real(py_item(fields, nk[].k_power_t)), _opt_real(py_item(fields, nk[].k_alpha)), _opt_real(py_item(fields, nk[].k_tol))]
    var addresses: List[Int] = [Int(py=xseq_addr(x)), Int(py=xseq_addr(target)), Int(py=xseq_addr(params)), Int(py=xseq_addr(curve))]
    var provider = xseq_neural_provider(route.entries)
    var entry = _entry(provider)
    var n_iter = 0
    # the fit loop's record (`fit_loop_`): the trainer writes these words once, after its loop
    var loop_words = fit_loop_words()
    var loop_address = Int(loop_words.unsafe_ptr())
    with GILReleased(py):
        n_iter = xseq_run_mlp_fit(entry, request, fp, addresses[0], addresses[1], addresses[2], addresses[3],
                                  loop_address)
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
    py_setitem(fields, nk[].k_coefs_, coefs)
    py_setitem(fields, nk[].k_intercepts_, intercepts)
    py_setitem(fields, nk[].k_n_iter_, n_iter)
    py_setitem(fields, nk[].k_fit_loop_, fit_loop_pack(loop_words))
    _ = curve  # keep the Python object alive through the native call above
    _ = params  # keep the Python object alive through the native call above
    var losses = xseq_curve_list(curve, n_iter)
    _ = params  # keep the Python object alive through the native call above
    py_setitem(fields, nk[].k_loss_curve_, losses)
    if n_iter == 0:
        raise Error("IndexError: list index out of range")
    _ = curve  # keep the Python object alive through the native calls above
    _ = params  # keep the Python object alive through the native calls above
    py_setitem(fields, nk[].k_loss_, losses[n_iter - 1])
    _ = curve  # keep the Python object alive through the native call above
    _ = params  # keep the Python object alive through the native call above
    # the first-wins minimum of the curve (`reduce_stat` 0)
    var curve_address = Int(py=xseq_addr(curve))
    var best_provider = xseq_neural_provider(route.output)
    var best_entry = _entry(best_provider)
    var best = 0.0
    with GILReleased(py):
        best = xseq_run_curve_min(best_entry, curve_address, n_iter)
    _ = best_provider
    _ = curve  # keep the Python object alive through the native call above
    _ = params  # keep the Python object alive through the native call above
    py_setitem(fields, nk[].k_best_loss_, best)
    _ = route.owner  # the route's images stay loaded through the calls above
    _ = curve  # keep the Python object alive through the native call above
    py_setitem(fields, nk[].k_t_, n_iter * rows)
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
    var nk = xseq_keys()
    if not Bool(py=PythonObject(columns) == py_item(fields, nk[].k_n_features_in_)):
        raise Error("X has " + String(columns) + " features, the model was fitted on "
                    + String(py=py_call(b.str, py_item(fields, nk[].k_n_features_in_))))
    var coefs = py_item(fields, nk[].k_coefs_)
    var intercepts = py_item(fields, nk[].k_intercepts_)
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
    var n_outputs = Int(py=py_item(fields, nk[].k_n_outputs_))
    xseq_run_mlp_output_check(input_width, n_outputs)
    var width = 2 if proba2 else n_outputs
    var out = xseq_zeros(Python.tuple(rows, width), context)
    var out_act = Int(py=py_item(context, nk[].k_xseq_mlp_outputs)[py_item(fields, nk[].k_out_activation_)])
    var activation = Int(py=py_item(context, nk[].k_xseq_mlp_activations)[py_item(fields, nk[].k_activation)])
    var route = xseq_route(model, fields, context)
    var request = XseqMlpPredictRequest(rows, columns, n_outputs, activation, out_act, hidden^, proba2, width,
                                        Int(py=params.size), Int(model._obj_ptr), token)
    var addresses: List[Int] = [Int(py=xseq_addr(x)), Int(py=xseq_addr(params)), Int(py=xseq_addr(out))]
    var provider = xseq_neural_provider(route.entries)
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
    _ = route.owner  # the route's images stay loaded through the call above
    return out


def xseq_mlp(model: PythonObject, operation: Int, args: PythonObject, context: PythonObject) raises -> PythonObject:
    # bw2-sequence (2026-11-20): the xseq code, as `xseq_rnn`.
    var fields = py_fields(model)
    var classifier = String(py=_attr(py_type(model), "_estimator_type")) == "classifier"
    var nk = xseq_keys()
    if operation == XSEQ_OPN_MLP_FIT:
        if not classifier:
            var y = xseq_as_f32_c(args[1], 0, "y", context)
            py_setitem(fields, nk[].k__y_1d, Int(py=y.ndim) == 1)
            var target = xseq_reshape(y, Python.tuple(Int(py=y.shape[0]), -1))
            return _mlp_fit(model, fields, args[0], target, 0, 0, context)
        var rank = 1
        try:
            rank = Int(py=xseq_input(args[1], context).ndim)
        except:
            rank = 1  # str labels: the flat list the label input step reads
        if rank != 1:
            raise Error("NotImplementedError: multilabel targets are not implemented")
        var label_route = xseq_route(model, fields, context)
        var encoded = xseq_labels(args[1], label_route.entries, context)
        _ = label_route.owner  # the route's image stays loaded through the encoding above
        py_setitem(fields, nk[].k_classes_, encoded[0])
        var k = len(encoded[0])
        if k < 2:
            raise Error("MLPClassifier needs at least two classes")
        var codes = encoded[1]
        if k == 2:
            py_setitem(fields, nk[].k__LOSS_CODE, 1)
            return _mlp_fit(model, fields, args[0], codes, 1, 1, context)
        py_setitem(fields, nk[].k__LOSS_CODE, 2)
        return _mlp_fit(model, fields, args[0], codes, 4, k, context)
    if operation == XSEQ_OPN_MLP_PREDICT_PROBA:
        return _mlp_forward(model, fields, args[0], Int(py=py_item(fields, nk[].k_n_outputs_)) == 1, context)
    var out = _mlp_forward(model, fields, args[0], False, context)
    var rows = Int(py=out.shape[0])
    if not classifier:
        if xseq_truth(py_get(fields, nk[].k__y_1d, PythonObject(False))):
            return xseq_reshape(out, Python.tuple(rows))
        return out
    # The output helpers run on the image of the estimator's route (a hit:
    # the forward above resolved it).
    var route = xseq_route(model, fields, context)
    if Int(py=out.shape[1]) == 1:
        # p > 0.5 per row (`threshold_labels_i64`; NaN takes class 0)
        var codes = sequential_array(Python.tuple(rows), py_item(context, nk[].k_int64), context)
        if rows:
            var scores_address = Int(py=xseq_addr(out))
            var codes_address = Int(py=xseq_addr(codes))
            var provider = xseq_neural_provider(route.output)
            var entry = _entry(provider)
            var py = Python()
            with GILReleased(py):
                xseq_run_threshold(entry, scores_address, codes_address, rows)
            _ = provider
            _ = out  # keep the Python object alive through the native call above
            _ = codes  # keep the Python object alive through the native call above
        return xseq_take(xseq_public_labels(fields, context), codes, context)
    return xseq_take(xseq_public_labels(fields, context), xseq_argmax(out, rows, Int(py=out.shape[1]), route.output, context), context)
