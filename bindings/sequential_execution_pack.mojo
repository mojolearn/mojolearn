# SPDX-License-Identifier: Apache-2.0
"""Native Sequential orchestration of the existing model numerical ABI."""
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.native_loaded_module_door import NativeModules
from bindings.py_balanced import py_call, py_contains, py_get, py_setitem
from core.cast_in import CastIn
from autograd.model_execution import model_counts_native
from autograd.model_descriptor import MODEL_FIT, MODEL_FIT_LABELS, MODEL_PREDICT, MODEL_PREDICT_LABELS, model_descriptor_words
from autograd.model_descriptor import FIT_SOURCE_MARK, FIT_STATE3_MARK  # lane stream-fits
from providers.python.autograd_pyapi_door import _ints, model_float64_words, model_admit_addresses
from providers.python.neural_descriptor_pyapi_door import neural_descriptor_execute
from bindings.sequential_description_unpack import sequential_fields, SequentialDescription, _integer
from bindings.sequential_description_pack import sequential_sample_shape, sequential_native_description
from bindings.sequential_arrays_door import sequential_array, sequential_address, sequential_input, sequential_descriptor, sequential_x_snapshot, sequential_curve
from bindings.py_optional import py_present
from bindings.sequential_keys_unpack import sequential_keys
from bindings.sequential_route_door import sequential_route_entries
from bindings.py_balanced import py_item
from autograd.model import FIT_STATE_WORDS
from autograd.nn_optim import NN_SCHED_ARGS, nn_opt_keeps_third
# lane stream-fits: Sequential.fit(source=)
from autograd.batch_source import BS_FIELD_STEPS, BS_FIELD_TARGETS, batch_source_fit_words
from providers.python.batch_source_pyapi_door import batch_source_spec, batch_source_position_out
from providers.python.autograd_program_arrays_pyapi_door import is_integer_input
from bindings.admitted_sequence_door import admit_nd_f32
from sequence.fit_loop_words import FIT_LOOP_MARK, fit_loop_words
from bindings.fit_loop_pack import fit_loop_pack
# nn-blocks-pgseq begin
from bindings.nn_pgseq_compile_pack import pgseq_door_words, pgseq_words_tuple
# nn-blocks-pgseq end


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
    if len(args) > 11 and args[11] is not None:
        return sequential_fit_source(mods, model, args, context)  # lane stream-fits: fit(source=)
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
    # lane rnn-parity-c: with lengths, the validation rows carry theirs: validation_data=(X, y, lengths)
    if validation is not None and (Bool(py=py_call(b.isinstance, validation, b.tuple)) or Bool(py=py_call(b.isinstance, validation, b.list))):
        if len(validation) == 3 and lengths_source is None:
            raise Error("ValueError: validation_data=(X, y, lengths) needs the training lengths (fit(..., lengths=))")
        if len(validation) == 2 and lengths_source is not None:
            raise Error("ValueError: with lengths, validation_data takes (X, y, lengths)")
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
    var bufs = _sequential_fit_buffers(mods, fields, options, resume, warm, n_params, epochs, context)
    var params = bufs[0]
    var moment = bufs[1]
    var variance = bufs[2]
    var state = bufs[3]
    # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
    var previous = Optional[PythonObject]()
    if resume:
        previous = py_present(py_item(fields, nk[].k__curve))
    var curve = bufs[4]
    var third = bufs[5]  # lane stream-fits: centered RMSprop's third state buffer (None for every other optimizer)
    seed = py_call(b.int, seed)
    # nn-losses: the loss word, code + 256 * reduction (autograd/model.mojo LOSS_WORD_REDUCTION)
    var loss_word = Int(py=py_item(context, nk[].k_loss_codes)[py_item(options, nk[].k_loss)]) + 256 * Int(py=py_get(options, nk[].k_reduction, 0))
    var ip = Python.list(n, loss_word, target, py_item(options, nk[].k_opt), py_item(options, nk[].k_flags),
        py_item(options, nk[].k_sched), py_item(options, nk[].k_step_size), batch_size, epochs, 1 if Bool(py=shuffle) else 0,
        seed.__and__(4294967295), seed.__rshift__(32), py_item(options, nk[].k_n_iter_no_change),
        0 if resume or warm else 1, 1 if resume else 0)
    var fp = Python.list(py_item(options, nk[].k_lr), py_item(options, nk[].k_f1), py_item(options, nk[].k_f2), py_item(options, nk[].k_eps),
        py_item(options, nk[].k_weight_decay), py_get(options, nk[].k_f7, PythonObject(0.0)), py_item(options, nk[].k_tol), py_item(options, nk[].k_gamma), py_item(options, nk[].k_huber_delta))
    var addresses = Python.list(xd, yd, sequential_address(params), sequential_address(curve),
        sequential_address(moment), sequential_address(variance), sequential_address(state),
        PythonObject(0) if not previous else sequential_address(previous.value()))
    var lengths = _sequential_lengths(lengths_source, n, context)
    if lengths is not None and validation is None:
        # the ninth address (autograd/model_execution.mojo model_fit_native); with validation data the lengths
        # follow the validation addresses (the 14th and 15th, lane rnn-parity-c)
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
    var val_lengths = PythonObject(None)
    if validation is not None:
        if not (Bool(py=py_call(b.isinstance, validation, b.tuple)) or Bool(py=py_call(b.isinstance, validation, b.list))) or (len(validation) != 2 and len(validation) != 3):
            raise Error("validation_data must be (X, y), or (X, y, lengths) with lengths")
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
        if lengths is not None:
            # lane rnn-parity-c: the training lengths (14th address) and the validation rows' (15th)
            val_lengths = _sequential_lengths(validation[2], vx_snapshot.source.rows, context, "validation lengths")
            _ = py_call(addresses.extend, Python.tuple(sequential_address(lengths), sequential_address(val_lengths)))
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
    var native_scalars = _sequential_fit_scalars(fields, options, fp, native_program, native_table)
    # lane alm-neural (2026-11-24): the fit loop's record (`fit_loop_`), written once by the trainer after its
    # epoch loop; its address ends the addresses and FIT_LOOP_MARK the options (autograd/model_descriptor.mojo)
    var loop_words = fit_loop_words()
    var native_addresses = native_inputs[0].copy()
    if third is not None:
        # lane stream-fits: the third optimizer state buffer, before the loop words (FIT_STATE3_MARK)
        native_addresses.append(Int(py=sequential_address(third)))
        native_options.append(FIT_STATE3_MARK)
    native_addresses.append(Int(loop_words.unsafe_ptr()))
    native_options.append(FIT_LOOP_MARK)
    var provider = py_item(entries, nk[].k_native_model_provider)
    var n_iter: Int
    if target != 0:
        var class_count = List[Int](length=1, fill=0)
        var words = model_descriptor_words(native_program, native_options, native_inputs[1],
            Int(py=sequential_address(classes_full)), n_out, target, Int(class_count.unsafe_ptr()))
        n_iter = neural_descriptor_execute(provider, MODEL_FIT_LABELS, native_addresses, words, native_scalars)
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
        n_iter = neural_descriptor_execute(provider, MODEL_FIT, native_addresses, words, native_scalars)
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
    _ = val_lengths^  # lane rnn-parity-c: the validation lengths' buffer, read by the native call above
    _ = target_lengths^  # nn-losses-2
    _ = third  # lane stream-fits: read and written by the native call above
    py_setitem(fields, nk[].k_params_, params)
    py_setitem(fields, nk[].k_n_iter_, n_iter)
    py_setitem(fields, nk[].k_fit_loop_, fit_loop_pack(loop_words))
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
    py_setitem(fields, PythonObject("optimizer_w_"), third)
    py_setitem(fields, nk[].k__fit_state, state)
    py_setitem(fields, nk[].k__curve, curve)
    py_setitem(fields, nk[].k__imported, False)
    return model


def _sequential_fit_buffers(mut mods: NativeModules, fields: PythonObject, options: PythonObject, resume: Bool,
                            warm: Bool, n_params: Int, epochs: Int, context: PythonObject) raises -> PythonObject:
    """(params, optimizer_m_, optimizer_v_, _fit_state, the loss curve, optimizer_w_) of a fit call: the fitted
    ones when resuming (a curve with room for `epochs` more), else new zeros. optimizer_w_ (lane stream-fits) is
    the third optimizer state buffer of centered RMSprop (autograd/nn_optim.mojo nn_opt_keeps_third), None for
    every other optimizer. One body for the array and the streamed fit."""
    var nk = sequential_keys()
    var f32 = py_item(context, nk[].k_float32)
    var third = PythonObject(None)
    var keeps = nn_opt_keeps_third(Int(py=py_item(options, nk[].k_opt)), Int(py=py_item(options, nk[].k_flags)))
    if resume:
        if keeps:
            third = py_get(fields, PythonObject("optimizer_w_"))
            if third is None:
                third = sequential_array(mods, Python.tuple(n_params), f32, context)
        return Python.tuple(py_item(fields, nk[].k_params_), py_item(fields, nk[].k_optimizer_m_),
                            py_item(fields, nk[].k_optimizer_v_), py_item(fields, nk[].k__fit_state),
                            sequential_array(mods, Python.tuple(Int(py=py_item(fields, nk[].k_n_iter_)) + epochs),
                                             f32, context), third)
    if keeps:
        third = sequential_array(mods, Python.tuple(n_params), f32, context)
    var params = py_item(fields, nk[].k_params_) if warm else sequential_array(mods, Python.tuple(n_params), f32,
                                                                              context)
    return Python.tuple(params, sequential_array(mods, Python.tuple(n_params), f32, context),
                        sequential_array(mods, Python.tuple(n_params), f32, context),
                        sequential_array(mods, Python.tuple(FIT_STATE_WORDS), py_item(context, nk[].k_int64), context),
                        sequential_array(mods, Python.tuple(epochs), f32, context), third)


def _sequential_fit_scalars(fields: PythonObject, options: PythonObject, fp: PythonObject,
                            native_program: List[Int], native_table: List[Float64]) raises -> List[Float64]:
    """The fit's float words: the nine option floats, one per layer, the schedule's argument words, then the
    parameter groups' words (autograd/model_execution.mojo model_fit_setup). Also records `_param_groups`. One body
    for the array and the streamed fit."""
    var nk = sequential_keys()
    var native_scalars = model_float64_words(fp)
    for scalar in native_table:  # driver-loop: reuse already admitted native layer options
        native_scalars.append(scalar)
    # lane nn-optim: the schedule's argument words after the layers' (autograd/model_execution.mojo)
    var sched_args = py_get(options, nk[].k_sched_args)
    if sched_args is not None:
        if len(sched_args) != NN_SCHED_ARGS:
            raise Error("the compile options' schedule arguments are damaged; call compile again")
        for k in range(NN_SCHED_ARGS):  # small-loop(8: the schedule's argument words)
            native_scalars.append(Float64(py=sched_args[k]))
    # nn-blocks-pgseq begin: compile's parameter groups, resolved against this model's tensors once per fit call,
    # after the schedule's words (autograd/nn_pgseq.mojo); a model compiled without groups adds nothing
    var param_groups = py_get(options, PythonObject("param_groups"))
    if param_groups is not None:
        if sched_args is None:
            for _ in range(NN_SCHED_ARGS):  # small-loop(8: the schedule's argument words, none given)
                native_scalars.append(0.0)
        var group_words = pgseq_door_words(param_groups, native_program, native_table)
        for k in range(len(group_words)):  # driver-loop: over the group words (once per fit call)
            native_scalars.append(group_words[k])
        # the per-tensor group table the fit trained with, for training_digest_ (python/mojolearn/_digest_fields.py)
        py_setitem(fields, PythonObject("_param_groups"), pgseq_words_tuple(group_words))
    else:
        py_setitem(fields, PythonObject("_param_groups"), PythonObject(None))
    # nn-blocks-pgseq end
    return native_scalars^


def sequential_fit_source(mut mods: NativeModules, model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """Sequential.fit(source=) (lane stream-fits, 2026-11-25). UNPACK: the exact BatchSource handle's words
    (providers/python/batch_source_pyapi_door.mojo batch_source_spec), the model's description for the source's
    record shape ((n, max_length, n_features), or (n, n_features) for records of one step), the fit buffers.
    RUN: one MODEL_FIT call carrying the source's words and path (FIT_SOURCE_MARK, autograd/model_descriptor.mojo):
    the training image opens the source and runs the streamed instance of the fit loop (autograd/model.mojo
    model_fit_state_source). PACK: the fitted fields and the consumed position on the handle; the training state
    (`_fit_state`) holds the position and the source's binding words, so save_checkpoint and a resume in a fresh
    process continue on the same next batch."""
    var nk = sequential_keys()
    var b = native_loaded_module("builtins")
    var fields = sequential_fields(model)
    if len(py_item(fields, nk[].k__layers)) == 0:
        raise Error("add at least one layer before fit")
    var options = py_item(fields, nk[].k__options)
    if options is None:
        raise Error("call compile before fit")
    if args[0] is not None or args[1] is not None:
        raise Error("TypeError: Sequential.fit(source=) takes no X or y: the source holds the records and their "
                    + "targets")
    if args[3] is not None or args[4] is not None:
        raise Error("TypeError: Sequential.fit(source=) takes no batch_size or shuffle: they are the source's "
                    + "(BatchSource(path, batch_size, shuffle=...))")
    if args[7] is not None or Bool(py=args[8]) or args[9] is not None or args[10] is not None:
        raise Error("TypeError: Sequential.fit(source=) takes no validation_data, restore_best_weights, lengths or "
                    + "target_lengths: the lengths are the source's record lengths")
    var seed = args[5]
    if not _integer(seed, context) or Bool(py=seed < PythonObject(0)) or Bool(py=seed > py_item(context, nk[].k_uint64_max)):
        raise Error("seed must be an integer in [0, 2**64)")
    var resume = Bool(py=args[6])
    if resume and py_get(fields, nk[].k__fit_state) is None:
        raise Error("resume=True continues an earlier fit: call fit (or from_checkpoint) first")
    var epochs = Int(py=py_call(b.int, args[2]))
    if epochs < 1:
        raise Error("epochs must be >= 1")
    var max_steps = 0
    var steps_in = args[12] if len(args) > 12 else PythonObject(None)
    if steps_in is not None:
        if not is_integer_input(steps_in) or Int(py=steps_in) < 1:
            raise Error("ValueError: Sequential.fit steps must be a positive integer or None")
        max_steps = Int(py=steps_in)
    var source = args[11]
    var spec = batch_source_spec(source)  # the exact connected BatchSource only (no Python iterable or callback)
    if len(spec.inputs) != 1 or spec.inputs[0] != BS_FIELD_STEPS or spec.target != BS_FIELD_TARGETS:
        raise Error("ValueError: Sequential.fit(source=) reads a BatchSource opened with inputs=('steps',) and "
                    + "target='targets'")
    var n = Int(py=source.n_records)
    var width = Int(py=source.n_features)
    var longest = Int(py=source.max_length)
    var dims = List[Int]()
    dims.append(n)
    if longest > 1:
        dims.append(longest)
    dims.append(width)
    var d = longest * width
    if resume and d != Int(py=py_item(fields, nk[].k_n_features_in_)):
        raise Error("the source's records have a different feature count from the fitted model")
    var sample = sequential_sample_shape(mods, model, dims)
    if resume and not Bool(py=sample == py_get(fields, nk[].k__sample_shape)):
        raise Error("resume=True takes records of the shape the model was fitted on")
    var warm = not resume and Bool(py=py_get(fields, nk[].k__imported, False))
    if warm and (d != Int(py=py_item(fields, nk[].k_n_features_in_)) or not Bool(py=sample == py_get(fields, nk[].k__sample_shape))):
        raise Error("the source's record shape differs from the imported weights' sample shape")
    var description_owner = sequential_native_description(mods, model, PythonObject(d), sample)
    var description = description_owner.downcast_value_ptr[SequentialDescription]()
    var entries = sequential_entries(model, context)
    mods.released()
    ref native_program = description[].program
    ref native_table = description[].scalars
    if not description[].counts_ready:
        var counts = model_counts_native(native_program, native_table)
        description[].parameter_count = counts[0]
        description[].output_count = counts[1]
        description[].counts_ready = True
    var n_params = description[].parameter_count
    var n_out = description[].output_count
    # the targets are the source's: class indices for cross_entropy, nll and bce_with_logits (classes_ is
    # arange(classes)), else n_out floats a record (the native fit checks both)
    var loss_name = String(py=py_item(options, nk[].k_loss))
    var target = 0
    var n_classes = 0
    if loss_name == "cross_entropy" or loss_name == "nll":
        target = 1
        n_classes = n_out
    elif loss_name == "bce_with_logits":
        target = 2
        n_classes = 2
    var bufs = _sequential_fit_buffers(mods, fields, options, resume, warm, n_params, epochs, context)
    var params = bufs[0]
    var moment = bufs[1]
    var variance = bufs[2]
    var state = bufs[3]
    var curve = bufs[4]
    var third = bufs[5]
    var previous = PythonObject(None)
    if resume:
        previous = py_item(fields, nk[].k__curve)
    var big_seed = py_call(b.int, seed)
    var loss_word = Int(py=py_item(context, nk[].k_loss_codes)[py_item(options, nk[].k_loss)]) + 256 * Int(py=py_get(options, nk[].k_reduction, 0))
    var ip = Python.list(n, loss_word, target, py_item(options, nk[].k_opt), py_item(options, nk[].k_flags),
        py_item(options, nk[].k_sched), py_item(options, nk[].k_step_size), spec.batch, epochs,
        1 if spec.shuffle else 0, big_seed.__and__(4294967295), big_seed.__rshift__(32),
        py_item(options, nk[].k_n_iter_no_change), 0 if resume or warm else 1, 1 if resume else 0)
    var fp = Python.list(py_item(options, nk[].k_lr), py_item(options, nk[].k_f1), py_item(options, nk[].k_f2), py_item(options, nk[].k_eps),
        py_item(options, nk[].k_weight_decay), py_get(options, nk[].k_f7, PythonObject(0.0)), py_item(options, nk[].k_tol), py_item(options, nk[].k_gamma), py_item(options, nk[].k_huber_delta))
    var native_options = _ints(ip)
    var native_scalars = _sequential_fit_scalars(fields, options, fp, native_program, native_table)
    var prev_address = 0 if previous is None else Int(py=sequential_address(previous))
    var native_addresses: List[Int] = [0, 0, Int(py=sequential_address(params)), Int(py=sequential_address(curve)),
        Int(py=sequential_address(moment)), Int(py=sequential_address(variance)), Int(py=sequential_address(state)),
        prev_address]
    if third is not None:
        native_addresses.append(Int(py=sequential_address(third)))
        native_options.append(FIT_STATE3_MARK)
    # the source's words, its path's bytes and the position-out words (autograd/model_descriptor.mojo
    # FIT_SOURCE_MARK)
    var path = spec.path
    var path_bytes = path.as_bytes()
    var words = batch_source_fit_words(spec, max_steps)
    words.append(len(path_bytes))
    for k in range(len(words)):  # small-loop(the source words)
        native_options.append(words[k])
    native_options.append(len(words))
    native_options.append(FIT_SOURCE_MARK)
    var position = List[Int64](length=2, fill=Int64(0))
    native_addresses.append(Int(path_bytes.unsafe_ptr()))
    native_addresses.append(Int(position.unsafe_ptr()))
    var loop_words = fit_loop_words()
    native_addresses.append(Int(loop_words.unsafe_ptr()))
    native_options.append(FIT_LOOP_MARK)
    var no_rows: List[CastIn] = [CastIn.f32_c(0, 1, 1), CastIn.f32_c(0, 1, 1)]
    var provider = py_item(entries, nk[].k_native_model_provider)
    var descriptor = model_descriptor_words(native_program, native_options, no_rows)
    var n_iter = neural_descriptor_execute(provider, MODEL_FIT, native_addresses, descriptor, native_scalars)
    mods.released()
    # Raw addresses never own data: every buffer the native call read stays alive through it.
    _ = path_bytes^
    _ = path
    _ = curve
    _ = previous
    _ = state
    _ = variance
    _ = moment
    _ = params
    _ = third
    _ = description_owner^
    batch_source_position_out(source, Int(position[0]), Int(position[1]))
    _ = position^
    py_setitem(fields, nk[].k_params_, params)
    py_setitem(fields, nk[].k_n_iter_, n_iter)
    py_setitem(fields, nk[].k_fit_loop_, fit_loop_pack(loop_words))
    py_setitem(fields, nk[].k_loss_curve_, sequential_curve(curve, n_iter))
    py_setitem(fields, nk[].k_n_features_in_, d)
    py_setitem(fields, nk[].k__sample_shape, sample)
    py_setitem(fields, nk[].k_n_outputs_, n_out)
    py_setitem(fields, nk[].k__fit_loss, py_item(options, nk[].k_loss))
    py_setitem(fields, nk[].k__target, target)
    py_setitem(fields, nk[].k__fit_batch, spec.batch)
    py_setitem(fields, nk[].k__fit_shuffle, spec.shuffle)
    py_setitem(fields, nk[].k__y_1d, target == 0 and n_out == 1)
    if n_classes > 0:
        # the source's class indices are the labels: classes_ = arange(classes)
        var classes = sequential_array(mods, Python.tuple(n_classes), py_item(context, nk[].k_int64), context)
        var cp = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(py=sequential_address(classes)))
        for k in range(n_classes):  # small-loop(classes: the class labels, PACK)
            cp[k] = Int64(k)
        py_setitem(fields, nk[].k_classes_, classes)
    py_setitem(fields, nk[].k_optimizer_m_, moment)
    py_setitem(fields, nk[].k_optimizer_v_, variance)
    py_setitem(fields, PythonObject("optimizer_w_"), third)
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
