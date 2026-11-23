# SPDX-License-Identifier: Apache-2.0
"""Native Sequential orchestration of the existing model numerical ABI."""
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.native_loaded_module_door import NativeModules
from bindings.py_balanced import py_call, py_contains, py_get, py_setitem
from core.cast_in import CastIn
from autograd.model_execution import model_counts_native
from autograd.model_descriptor import MODEL_FIT, MODEL_FIT_LABELS, MODEL_PREDICT, MODEL_PREDICT_LABELS, model_descriptor_words
from providers.python.autograd_pyapi_door import _ints, model_float64_words, model_admit_addresses
from providers.python.neural_descriptor_pyapi_door import neural_descriptor_execute
from bindings.sequential_description_unpack import sequential_fields, SequentialDescription, _integer
from bindings.sequential_description_pack import sequential_sample_shape, sequential_native_description
from bindings.sequential_arrays_door import sequential_array, sequential_address, sequential_input, sequential_descriptor, sequential_x_snapshot, sequential_curve
from bindings.py_optional import py_present
from bindings.sequential_keys_unpack import sequential_keys
from bindings.sequential_route_door import sequential_route_entries
from bindings.py_balanced import py_item
from bindings.admitted_sequence_door import admit_nd_f32


def _sequential_lengths(source: PythonObject, n: Int, context: PythonObject, name: String = "lengths") raises -> PythonObject:
    """The `lengths` argument (lane seq-blocks-3): None stays None, else one C-ordered float32 vector of n
    values (the one admission cast); the values are checked natively (autograd/model.mojo
    model_lengths_check: whole numbers from 1 to the padded sequence length)."""
    if source is None:
        return source
    var a = sequential_input(source, context)
    if Int(py=a.ndim) != 1 or Int(py=a.shape[0]) != n:
        raise Error("ValueError: " + name + " must be a vector of one length per sample (" + String(n) + ")")
    return admit_nd_f32(a, name, False, context)


def sequential_entries(model: PythonObject, context: PythonObject) raises -> PythonObject:
    # The model's route owner (bindings/sequential_route_door.mojo): resolved
    # once per model for its device and numeric mode. The former per-call
    # selection (mode word, answered mode, bindings read, vendor verdict,
    # provider reuse) runs at resolution only.
    return sequential_route_entries(model, context)


def sequential_targets(mut mods: NativeModules, model: PythonObject, source: PythonObject, n: Int, n_out: Int,
                       loss: String, entries: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = sequential_keys()
    var y = sequential_input(source, context)
    var nd = Int(py=y.ndim)
    var fields = sequential_fields(model)
    var classification = loss == "cross_entropy" or loss == "bce_with_logits" or loss == "nll"
    if nd < 1 or nd > 2 or Int(py=y.shape[0]) != n or (classification and nd != 1):
        raise Error("y must hold one target row per input sample")
    var descriptor = sequential_descriptor(y, "y")
    # lm_head (lane neural-blocks-3-lm): the (n_samples, length) int32 token ids as they are (target code 0,
    # no labels sort); the native fit checks their count and dtype against the head
    # `_y_1d` is False for a classifier (its answers are labels and probabilities, never reshaped), as a model
    # loaded from its file has it (sequential_checkpoint_door.mojo), so the two have one `digest_`; a regression
    # sets it below.
    py_setitem(fields, nk[].k__y_1d, False)
    if loss == "token_cross_entropy":
        return Python.tuple(y, descriptor, 0, PythonObject(None))
    # nn-losses-2 begin: ctc's (n_samples, max target length) label sequences, one C-ordered float32 copy (the one
    # admission cast); the native fit checks them with the target lengths (autograd/nn_ctc.mojo ctc_admit)
    if loss == "ctc":
        var labels = admit_nd_f32(y, "y", False, context)
        return Python.tuple(labels, sequential_descriptor(labels, "y"), 0, PythonObject(None))
    # nn-losses-2 end
    if classification:
        # the labels as they are and a classes buffer of their dtype: the
        # fit entry sorts them and keeps the codes on the device
        # (`model_fit_labels`); the class-count refusals run there
        var classes = sequential_array(mods, Python.tuple(n), y.dtype, context)
        return Python.tuple(y, descriptor, 2 if loss == "bce_with_logits" else 1, classes)
    py_setitem(fields, nk[].k__y_1d, nd == 1)
    var width = 1 if nd == 1 else Int(py=y.shape[1])
    if width != n_out:
        raise Error("y has " + String(width) + " outputs but the model has " + String(n_out))
    return Python.tuple(y, descriptor, 0, PythonObject(None))


def sequential_fit(mut mods: NativeModules, model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = sequential_keys()
    var b = native_loaded_module("builtins")
    var fields = sequential_fields(model)
    if len(py_item(fields, nk[].k__layers)) == 0:
        raise Error("add at least one layer before fit")
    var options = py_item(fields, nk[].k__options)
    if options is None:
        raise Error("call compile before fit")
    var validation = args[7]
    var restore_best = Bool(py=args[8])
    var lengths_source = args[9] if len(args) > 9 else PythonObject(None)
    if lengths_source is not None and validation is not None:
        raise Error("ValueError: validation_data with lengths is not supported")
    if restore_best and validation is None:
        raise Error("restore_best_weights requires validation_data")
    var seed = args[5]
    if not _integer(seed, context) or Bool(py=seed < PythonObject(0)) or Bool(py=seed > py_item(context, nk[].k_uint64_max)):
        raise Error("seed must be an integer in [0, 2**64)")
    var resume = Bool(py=args[6])
    if resume and py_get(fields, nk[].k__fit_state) is None:
        raise Error("resume=True continues an earlier fit: call fit (or from_checkpoint) first")
    var batch = args[3]
    var shuffle = args[4]
    if batch is None:
        batch = py_item(fields, nk[].k__fit_batch) if resume else PythonObject(32)
    if shuffle is None:
        shuffle = py_item(fields, nk[].k__fit_shuffle) if resume else PythonObject(True)
    var epochs = Int(py=py_call(b.int, args[2]))
    var batch_size = Int(py=py_call(b.int, batch))
    if epochs < 1 or batch_size < 1:
        raise Error("epochs and batch_size must be >= 1")
    var x_snapshot = sequential_x_snapshot(mods, args[0], "X", context)
    var x = x_snapshot.keep
    var xd = x_snapshot.abi()
    var n = x_snapshot.source.rows
    var d = x_snapshot.source.cols
    if resume and d != Int(py=py_item(fields, nk[].k_n_features_in_)):
        raise Error("X has a different feature count from the fitted model")
    var sample = sequential_sample_shape(mods, model, x_snapshot.shape)
    if resume and not Bool(py=sample == py_get(fields, nk[].k__sample_shape)):
        raise Error("resume=True takes inputs of the shape the model was fitted on")
    var warm = not resume and Bool(py=py_get(fields, nk[].k__imported, False))
    if warm and (d != Int(py=py_item(fields, nk[].k_n_features_in_)) or not Bool(py=sample == py_get(fields, nk[].k__sample_shape))):
        raise Error("X has a different feature count or sample shape from the imported weights")
    var description_owner = sequential_native_description(mods, model, PythonObject(d), sample)
    var description = description_owner.downcast_value_ptr[SequentialDescription]()
    var entries = sequential_entries(model, context)
    mods.released()
    ref native_program = description[].program
    ref native_table = description[].scalars
    # Counts depend only on this admitted native description. Keep their
    # parse result across repeated fit/resume calls; perform the first parse
    # here, after binding selection, to preserve validation/error ordering.
    if not description[].counts_ready:
        var counts = model_counts_native(native_program, native_table)
        description[].parameter_count = counts[0]
        description[].output_count = counts[1]
        description[].counts_ready = True
    var n_params = description[].parameter_count
    var n_out = description[].output_count
    var target_data = sequential_targets(mods, model, args[1], n, n_out, String(py_item(options, nk[].k_loss)), entries, context)
    var y = target_data[0]
    var yd = target_data[1]
    var target = Int(py=target_data[2])
    var classes_full = target_data[3]
    var params = PythonObject(None)
    var moment = PythonObject(None)
    var variance = PythonObject(None)
    var state = PythonObject(None)
    # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
    var previous = Optional[PythonObject]()
    var curve = PythonObject(None)
    if resume:
        params = py_item(fields, nk[].k_params_)
        moment = py_item(fields, nk[].k_optimizer_m_)
        variance = py_item(fields, nk[].k_optimizer_v_)
        state = py_item(fields, nk[].k__fit_state)
        previous = py_present(py_item(fields, nk[].k__curve))
        curve = sequential_array(mods, Python.tuple(Int(py=py_item(fields, nk[].k_n_iter_)) + epochs), py_item(context, nk[].k_float32), context)
    else:
        params = py_item(fields, nk[].k_params_) if warm else sequential_array(mods, Python.tuple(n_params), py_item(context, nk[].k_float32), context)
        moment = sequential_array(mods, Python.tuple(n_params), py_item(context, nk[].k_float32), context)
        variance = sequential_array(mods, Python.tuple(n_params), py_item(context, nk[].k_float32), context)
        state = sequential_array(mods, Python.tuple(12), py_item(context, nk[].k_int64), context)
        curve = sequential_array(mods, Python.tuple(epochs), py_item(context, nk[].k_float32), context)
    seed = py_call(b.int, seed)
    # nn-losses: the loss word, code + 256 * reduction (autograd/model.mojo LOSS_WORD_REDUCTION)
    var loss_word = Int(py=py_item(context, nk[].k_loss_codes)[py_item(options, nk[].k_loss)]) + 256 * Int(py=py_get(options, nk[].k_reduction, 0))
    var ip = Python.list(n, loss_word, target, py_item(options, nk[].k_opt), py_item(options, nk[].k_flags),
        py_item(options, nk[].k_sched), py_item(options, nk[].k_step_size), batch_size, epochs, 1 if Bool(py=shuffle) else 0,
        seed.__and__(4294967295), seed.__rshift__(32), py_item(options, nk[].k_n_iter_no_change),
        0 if resume or warm else 1, 1 if resume else 0)
    var fp = Python.list(py_item(options, nk[].k_lr), py_item(options, nk[].k_f1), py_item(options, nk[].k_f2), py_item(options, nk[].k_eps),
        py_item(options, nk[].k_weight_decay), 0.0, py_item(options, nk[].k_tol), py_item(options, nk[].k_gamma), py_item(options, nk[].k_huber_delta))
    var addresses = Python.list(xd, yd, sequential_address(params), sequential_address(curve),
        sequential_address(moment), sequential_address(variance), sequential_address(state),
        PythonObject(0) if not previous else sequential_address(previous.value()))
    var lengths = _sequential_lengths(lengths_source, n, context)
    if lengths is not None:
        # the ninth address (autograd/model_execution.mojo model_fit_native)
        _ = py_call(addresses.append, sequential_address(lengths))
    # nn-losses-2 begin: the ctc loss's target lengths, the tenth address (the ninth 0 without lengths)
    var target_lengths_source = args[10] if len(args) > 10 else PythonObject(None)
    if target_lengths_source is not None and validation is not None:
        raise Error("ValueError: validation_data with target_lengths is not supported")
    var target_lengths = _sequential_lengths(target_lengths_source, n, context, "target_lengths")
    if target_lengths is not None:
        if lengths is None:
            _ = py_call(addresses.append, 0)
        _ = py_call(addresses.append, sequential_address(target_lengths))
    # nn-losses-2 end
    _ = curve  # keep the Python object alive through the native call above
    _ = params  # keep the Python object alive through the native call above
    _ = state  # keep the Python object alive through the native call above
    _ = variance  # keep the Python object alive through the native call above
    _ = moment  # keep the Python object alive through the native call above
    _ = previous  # keep the Python object alive through the native call above
    var vx = PythonObject(None)
    var vy = PythonObject(None)
    var val_curve = PythonObject(None)
    var val_info = PythonObject(None)
    if validation is not None:
        if not (Bool(py=py_call(b.isinstance, validation, b.tuple)) or Bool(py=py_call(b.isinstance, validation, b.list))) or len(validation) != 2:
            raise Error("validation_data must be (X, y)")
        var vx_snapshot = sequential_x_snapshot(mods, validation[0], "validation X", context)
        vx = vx_snapshot.keep
        var vxd = vx_snapshot.abi()
        vy = sequential_input(validation[1], context)
        var vnd = Int(py=vy.ndim)
        if not Bool(py=sequential_sample_shape(mods, model, vx_snapshot.shape) == sample) or vx_snapshot.source.cols != d:
            raise Error("validation inputs must have the training sample shape")
        if vnd < 1 or vnd > 2 or Int(py=vy.shape[0]) != vx_snapshot.source.rows:
            raise Error("validation targets must match the validation rows")
        if target != 0 and vnd != 1:
            raise Error("validation classification targets must be a vector")
        # the classes buffer as allocated (n slots); the fit entry cuts the
        # descriptor to the classes the sort found
        var classes = PythonObject(None) if target == 0 else sequential_descriptor(classes_full, "classes")
        val_curve = sequential_array(mods, Python.tuple(epochs), py_item(context, nk[].k_float32), context)
        val_info = sequential_array(mods, Python.tuple(2), py_item(context, nk[].k_int64), context)
        _ = py_call(addresses.extend, Python.tuple(vxd, sequential_descriptor(vy, "validation y"), classes,
                    sequential_address(val_curve), sequential_address(val_info)))
        _ = curve  # keep the Python object alive through the native call above
        _ = previous  # keep the Python object alive through the native call above
        _ = state  # keep the Python object alive through the native call above
        _ = variance  # keep the Python object alive through the native call above
        _ = moment  # keep the Python object alive through the native call above
        _ = params  # keep the Python object alive through the native call above
        _ = val_info  # keep the Python object alive through the native call above
        _ = val_curve  # keep the Python object alive through the native call above
        _ = py_call(ip.append, 1 if restore_best else 0)
    var native_inputs = model_admit_addresses(addresses, target, False, x_snapshot.source)
    mods.released()
    var native_options = _ints(ip)
    var native_scalars = model_float64_words(fp)
    for scalar in native_table:  # driver-loop: reuse already admitted native layer options
        native_scalars.append(scalar)
    var provider = py_item(entries, nk[].k_native_model_provider)
    var n_iter: Int
    if target != 0:
        var class_count = List[Int](length=1, fill=0)
        var words = model_descriptor_words(native_program, native_options, native_inputs[1],
            Int(py=sequential_address(classes_full)), n_out, target, Int(class_count.unsafe_ptr()))
        n_iter = neural_descriptor_execute(provider, MODEL_FIT_LABELS, native_inputs[0], words, native_scalars)
        mods.released()
        _ = curve  # keep the Python object alive through the native call above
        _ = previous  # keep the Python object alive through the native call above
        _ = state  # keep the Python object alive through the native call above
        _ = variance  # keep the Python object alive through the native call above
        _ = moment  # keep the Python object alive through the native call above
        _ = params  # keep the Python object alive through the native call above
        _ = classes_full  # keep the Python object alive through the native call above
        var count = class_count[0]
        py_setitem(fields, nk[].k_classes_, py_call(py_item(context, nk[].k_ndarray), Python.tuple(count), dtype=classes_full.dtype, buffer=classes_full))
        mods.released()
    else:
        var words = model_descriptor_words(native_program, native_options, native_inputs[1])
        n_iter = neural_descriptor_execute(provider, MODEL_FIT, native_inputs[0], words, native_scalars)
        mods.released()
        _ = curve  # keep the Python object alive through the native call above
        _ = previous  # keep the Python object alive through the native call above
        _ = state  # keep the Python object alive through the native call above
        _ = variance  # keep the Python object alive through the native call above
        _ = moment  # keep the Python object alive through the native call above
        _ = params  # keep the Python object alive through the native call above
    _ = description_owner^
    # Raw addresses never own data. Keep every source alive through the
    # synchronous native model call, including temporary sequence buffers.
    _ = x^
    _ = y^
    _ = previous^
    _ = vx^
    _ = vy^
    _ = lengths^
    _ = target_lengths^  # nn-losses-2
    py_setitem(fields, nk[].k_params_, params)
    py_setitem(fields, nk[].k_n_iter_, n_iter)
    if validation is not None:
        var info = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(py=sequential_address(val_info)))
        _ = val_info  # keep the Python object alive through the native call above
        var ran = Int(info[0])
        _ = val_info  # keep the Python object alive through the native call above
        py_setitem(fields, nk[].k_epochs_run_, ran)
        py_setitem(fields, nk[].k_best_epoch_, PythonObject(info[1]))
        _ = val_info  # keep the Python object alive through the native call above
        py_setitem(fields, nk[].k_validation_loss_curve_, sequential_curve(val_curve, ran))
        _ = val_info^
    py_setitem(fields, nk[].k_loss_curve_, sequential_curve(curve, n_iter))
    py_setitem(fields, nk[].k_n_features_in_, d)
    py_setitem(fields, nk[].k__sample_shape, sample)
    py_setitem(fields, nk[].k_n_outputs_, n_out)
    py_setitem(fields, nk[].k__fit_loss, py_item(options, nk[].k_loss))
    py_setitem(fields, nk[].k__target, target)
    py_setitem(fields, nk[].k__fit_batch, batch_size)
    py_setitem(fields, nk[].k__fit_shuffle, Bool(py=shuffle))
    py_setitem(fields, nk[].k_optimizer_m_, moment)
    py_setitem(fields, nk[].k_optimizer_v_, variance)
    py_setitem(fields, nk[].k__fit_state, state)
    py_setitem(fields, nk[].k__curve, curve)
    py_setitem(fields, nk[].k__imported, False)
    return model


def sequential_forward(mut mods: NativeModules, model: PythonObject, source: PythonObject, transform: Int,
                       width: PythonObject, entries: PythonObject, context: PythonObject) raises -> PythonObject:
    # An overload, not a `classes = PythonObject(None)` default: a non-raising
    # caller (the abi("C") door) compile-time evaluates a default, and a
    # PythonObject cannot be built at compile time.
    return sequential_forward(mods, model, source, transform, width, entries, context, PythonObject(None))


def sequential_forward(mut mods: NativeModules, model: PythonObject, source: PythonObject, transform: Int,
                       width: PythonObject, entries: PythonObject, context: PythonObject,
                       classes: PythonObject) raises -> PythonObject:
    return sequential_forward(mods, model, source, transform, width, entries, context, classes, PythonObject(None))


def sequential_forward(mut mods: NativeModules, model: PythonObject, source: PythonObject, transform: Int,
                       width: PythonObject, entries: PythonObject, context: PythonObject,
                       classes: PythonObject, lengths_source: PythonObject) raises -> PythonObject:
    var nk = sequential_keys()
    var fields = sequential_fields(model)
    if not py_contains(fields, nk[].k_params_):
        raise Error("call fit before predict")
    var sample = py_get(fields, nk[].k__sample_shape)
    var x_snapshot = sequential_x_snapshot(mods, source, "X", context)
    var x = x_snapshot.keep
    var rank = len(x_snapshot.shape)
    if sample is None and rank != 2:
        raise Error("X must be a 2-D array of shape (n_samples, n_features)")
    if sample is not None and rank > 2:
        var actual = Python.list()
        for k in range(1, rank):  # driver-loop: over the prediction input sample rank
            _ = py_call(actual.append, PythonObject(x_snapshot.shape[k]))
        if not Bool(py=py_call(native_loaded_module("builtins").tuple, actual) == sample):
            mods.released()
            raise Error("X samples have a different shape from the fitted model")
        mods.released()
    if not Bool(py=PythonObject(x_snapshot.source.cols) == py_item(fields, nk[].k_n_features_in_)):
        raise Error("X has a different feature count from the fitted model")
    var params = py_item(fields, nk[].k_params_)
    var description_owner = sequential_native_description(mods, model, py_item(fields, nk[].k_n_features_in_), sample)
    var description = description_owner.downcast_value_ptr[SequentialDescription]()
    ref native_program = description[].program
    ref native_table = description[].scalars
    var sources: List[CastIn] = [x_snapshot.source]
    var provider = py_item(entries, nk[].k_native_model_provider)
    var lengths = _sequential_lengths(lengths_source, x_snapshot.source.rows, context)
    var out: PythonObject
    if classes is not None:
        # the labels of a classifier, in classes_' dtype
        out = sequential_array(mods, Python.tuple(x_snapshot.source.rows), classes.dtype, context)
        var addresses: List[Int] = [0, Int(py=sequential_address(params)),
                                    Int(py=sequential_address(classes)), Int(py=sequential_address(out))]
        if lengths is not None:
            addresses.append(Int(py=sequential_address(lengths)))
        var options: List[Int] = [x_snapshot.source.rows, 4096, transform, Int(py=classes.shape[0]), Int(py=classes.dtype.itemsize)]
        var words = model_descriptor_words(native_program, options, sources)
        _ = neural_descriptor_execute(provider, MODEL_PREDICT_LABELS, addresses, words, native_table)
        mods.released()
        _ = description_owner^
        _ = params  # keep the Python object alive through the native call above
        _ = x^
        _ = params^
        _ = lengths^
        return out
    var output_width = py_item(fields, nk[].k_n_outputs_) if width is None else width
    out = sequential_array(mods, Python.tuple(x_snapshot.source.rows, output_width), py_item(context, nk[].k_float32), context)
    var addresses: List[Int] = [0, Int(py=sequential_address(params)), Int(py=sequential_address(out))]
    if lengths is not None:
        addresses.append(Int(py=sequential_address(lengths)))
    var options: List[Int] = [x_snapshot.source.rows, 4096, transform]
    var words = model_descriptor_words(native_program, options, sources)
    _ = neural_descriptor_execute(provider, MODEL_PREDICT, addresses, words, native_table)
    mods.released()
    _ = description_owner^
    _ = x^
    _ = params^
    _ = lengths^
    return out


def sequential_predict(mut mods: NativeModules, model: PythonObject, source: PythonObject, probabilities: Bool,
                       context: PythonObject, lengths: PythonObject) raises -> PythonObject:
    var nk = sequential_keys()
    var fields = sequential_fields(model)
    var saved_loss = py_get(fields, nk[].k__fit_loss)
    var loss = "" if saved_loss is None else String(saved_loss)
    var entries = sequential_entries(model, context)
    mods.released()
    var classifier = loss == "cross_entropy" or loss == "bce_with_logits" or loss == "nll"
    if probabilities:
        # nll: the outputs are log-probabilities, and their softmax is the probabilities themselves
        if loss == "cross_entropy" or loss == "nll":
            return sequential_forward(mods, model, source, 1, PythonObject(None), entries, context, PythonObject(None),
                                      lengths)
        if loss == "bce_with_logits":
            return sequential_forward(mods, model, source, 3, PythonObject(2), entries, context, PythonObject(None),
                                      lengths)
        raise Error("predict_proba needs a model fitted with cross_entropy, nll or bce_with_logits")
    if classifier:
        # one entry: the argmax (threshold) codes stay on the device and are
        # gathered there into the labels (`model_predict_labels`)
        return sequential_forward(mods, model, source, 5 if loss == "bce_with_logits" else 4, PythonObject(1), entries,
                                  context, py_item(fields, nk[].k_classes_), lengths)
    var result = sequential_forward(mods, model, source, 0, PythonObject(None), entries, context, PythonObject(None),
                                    lengths)
    if Bool(py=py_get(fields, nk[].k__y_1d, False)):
        # A one-output model is already contiguous. The native ndarray
        # constructor changes only its ABI rank, preserving the returned words.
        var windowed_result14 = py_call(py_item(context, nk[].k_ndarray), Python.tuple(result.shape[0]), dtype=py_item(context, nk[].k_float32), buffer=result)
        mods.released()
        return windowed_result14^
    return result
