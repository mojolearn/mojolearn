# SPDX-License-Identifier: Apache-2.0
"""Native Sequential orchestration of the existing model numerical ABI."""
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_contains, py_get, py_setitem
from core.cast_in import CastIn
from autograd.model_execution import model_counts_native
from autograd.model_descriptor import MODEL_ABI_NAME, MODEL_FIT, MODEL_FIT_LABELS, MODEL_PREDICT, MODEL_PREDICT_LABELS, model_descriptor_words
from providers.python.autograd_pyapi import _ints, model_float64_words, model_admit_addresses
from providers.python.neural_descriptor_pyapi import neural_descriptor_reuse, neural_descriptor_execute
from bindings.sequential_description import sequential_fields, sequential_sample_shape, sequential_program, sequential_table, _integer
from bindings.sequential_arrays import sequential_array, sequential_address, sequential_input, sequential_descriptor, sequential_x_snapshot, sequential_curve


def sequential_entries(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var mode = sequential_fields(model)["numeric_mode"]
    if mode is None:
        mode = context["backend_fields"]["_DEFAULT_MODE"]
    if not Bool(py=py_call(b.isinstance, mode, b.str)):
        raise Error("numeric_mode must be fast, identical or None")
    mode = mode.strip().lower()
    if String(mode) != "identical" and String(mode) != "fast":
        raise Error("numeric_mode must be fast, identical or None")
    var entries = context["bindings"][mode]
    if Bool(py=py_call(b.isinstance, entries, b.str)):
        raise Error(String(entries))
    var actual_mode = Int(py=entries["x_sequence_numeric_mode"]())
    var actual_vendor = String(entries["x_sequence_vendor"]())
    if actual_mode != (1 if String(mode) == "identical" else 0):
        raise Error("RuntimeError: Sequential native numerical mode differs from the requested mode")
    if actual_vendor != String(context["vendor"]):
        raise Error("RuntimeError: Sequential native vendor differs from the selected execution route")
    # Connection/metadata admission only. Exact compiled factory/module identity
    # owns the shared provider; no per-model pointer cache or Python callable is
    # used by the subsequent native fit/predict execution.
    var provider = neural_descriptor_reuse(py_get(entries, "native_model_provider"),
        entries["model_native_api_v1"], String(MODEL_ABI_NAME), 4)
    py_setitem(entries, "native_model_provider", provider)
    return entries


def sequential_targets(model: PythonObject, source: PythonObject, n: Int, n_out: Int,
                       loss: String, entries: PythonObject, context: PythonObject) raises -> PythonObject:
    var y = sequential_input(source, context)
    var nd = Int(py=y.ndim)
    var fields = sequential_fields(model)
    var classification = loss == "cross_entropy" or loss == "bce_with_logits"
    if nd < 1 or nd > 2 or Int(py=y.shape[0]) != n or (classification and nd != 1):
        raise Error("y must hold one target row per input sample")
    var descriptor = sequential_descriptor(y, "y")
    if classification:
        # the labels as they are and a classes buffer of their dtype: the
        # fit entry sorts them and keeps the codes on the device
        # (`model_fit_labels`); the class-count refusals run there
        var classes = sequential_array(Python.tuple(n), y.dtype, context)
        return Python.tuple(y, descriptor, 1 if loss == "cross_entropy" else 2, classes)
    py_setitem(fields, "_y_1d", nd == 1)
    var width = 1 if nd == 1 else Int(py=y.shape[1])
    if width != n_out:
        raise Error("y has " + String(width) + " outputs but the model has " + String(n_out))
    return Python.tuple(y, descriptor, 0, PythonObject(None))


def sequential_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var fields = sequential_fields(model)
    if len(fields["_layers"]) == 0:
        raise Error("add at least one layer before fit")
    var options = fields["_options"]
    if options is None:
        raise Error("call compile before fit")
    var validation = args[7]
    var restore_best = Bool(py=args[8])
    if restore_best and validation is None:
        raise Error("restore_best_weights requires validation_data")
    var seed = args[5]
    if not _integer(seed, context) or Bool(py=seed < PythonObject(0)) or Bool(py=seed > context["uint64_max"]):
        raise Error("seed must be an integer in [0, 2**64)")
    var resume = Bool(py=args[6])
    if resume and py_get(fields, "_fit_state") is None:
        raise Error("resume=True continues an earlier fit: call fit (or from_checkpoint) first")
    var batch = args[3]
    var shuffle = args[4]
    if batch is None:
        batch = fields["_fit_batch"] if resume else PythonObject(32)
    if shuffle is None:
        shuffle = fields["_fit_shuffle"] if resume else PythonObject(True)
    var epochs = Int(py=py_call(b.int, args[2]))
    var batch_size = Int(py=py_call(b.int, batch))
    if epochs < 1 or batch_size < 1:
        raise Error("epochs and batch_size must be >= 1")
    var x_snapshot = sequential_x_snapshot(args[0], "X", context)
    var x = x_snapshot.keep
    var xd = x_snapshot.abi()
    var n = x_snapshot.source.rows
    var d = x_snapshot.source.cols
    if resume and d != Int(py=fields["n_features_in_"]):
        raise Error("X has a different feature count from the fitted model")
    var sample = sequential_sample_shape(model, x_snapshot.shape)
    if resume and not Bool(py=sample == py_get(fields, "_sample_shape")):
        raise Error("resume=True takes inputs of the shape the model was fitted on")
    var warm = not resume and Bool(py=py_get(fields, "_imported", False))
    if warm and (d != Int(py=fields["n_features_in_"]) or not Bool(py=sample == py_get(fields, "_sample_shape"))):
        raise Error("X has a different feature count or sample shape from the imported weights")
    var program = sequential_program(model, PythonObject(d), sample)
    var table = sequential_table(model)
    var entries = sequential_entries(model, context)
    var native_program = _ints(program)
    var native_table = model_float64_words(table)
    var counts = model_counts_native(native_program, native_table)
    var n_params = counts[0]
    var n_out = counts[1]
    var target_data = sequential_targets(model, args[1], n, n_out, String(options["loss"]), entries, context)
    var y = target_data[0]
    var yd = target_data[1]
    var target = Int(py=target_data[2])
    var classes_full = target_data[3]
    var params = PythonObject(None)
    var moment = PythonObject(None)
    var variance = PythonObject(None)
    var state = PythonObject(None)
    var previous = PythonObject(None)
    var curve = PythonObject(None)
    if resume:
        params = fields["params_"]
        moment = fields["_opt_m"]
        variance = fields["_opt_v"]
        state = fields["_fit_state"]
        previous = fields["_curve"]
        curve = sequential_array(Python.tuple(Int(py=fields["n_iter_"]) + epochs), context["float32"], context)
    else:
        params = fields["params_"] if warm else sequential_array(Python.tuple(n_params), context["float32"], context)
        moment = sequential_array(Python.tuple(n_params), context["float32"], context)
        variance = sequential_array(Python.tuple(n_params), context["float32"], context)
        state = sequential_array(Python.tuple(12), context["int64"], context)
        curve = sequential_array(Python.tuple(epochs), context["float32"], context)
    seed = py_call(b.int, seed)
    var ip = Python.list(n, context["loss_codes"][options["loss"]], target, options["opt"], options["flags"],
        options["sched"], options["step_size"], batch_size, epochs, 1 if Bool(py=shuffle) else 0,
        seed.__and__(4294967295), seed.__rshift__(32), options["n_iter_no_change"],
        0 if resume or warm else 1, 1 if resume else 0)
    var fp = Python.list(options["lr"], options["f1"], options["f2"], options["eps"],
        options["weight_decay"], 0.0, options["tol"], options["gamma"], options["huber_delta"])
    var addresses = Python.list(xd, yd, sequential_address(params), sequential_address(curve),
        sequential_address(moment), sequential_address(variance), sequential_address(state),
        PythonObject(0) if previous is None else sequential_address(previous))
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
        var vx_snapshot = sequential_x_snapshot(validation[0], "validation X", context)
        vx = vx_snapshot.keep
        var vxd = vx_snapshot.abi()
        vy = sequential_input(validation[1], context)
        var vnd = Int(py=vy.ndim)
        if not Bool(py=sequential_sample_shape(model, vx_snapshot.shape) == sample) or vx_snapshot.source.cols != d:
            raise Error("validation inputs must have the training sample shape")
        if vnd < 1 or vnd > 2 or Int(py=vy.shape[0]) != vx_snapshot.source.rows:
            raise Error("validation targets must match the validation rows")
        if target != 0 and vnd != 1:
            raise Error("validation classification targets must be a vector")
        # the classes buffer as allocated (n slots); the fit entry cuts the
        # descriptor to the classes the sort found
        var classes = PythonObject(None) if target == 0 else sequential_descriptor(classes_full, "classes")
        val_curve = sequential_array(Python.tuple(epochs), context["float32"], context)
        val_info = sequential_array(Python.tuple(2), context["int64"], context)
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
    var native_options = _ints(ip)
    var native_scalars = model_float64_words(fp)
    for scalar in native_table:  # driver-loop: reuse already admitted native layer options
        native_scalars.append(scalar)
    var provider = entries["native_model_provider"]
    var n_iter: Int
    if target != 0:
        var class_count = List[Int](length=1, fill=0)
        var words = model_descriptor_words(native_program, native_options, native_inputs[1],
            Int(py=sequential_address(classes_full)), n_out, target, Int(class_count.unsafe_ptr()))
        n_iter = neural_descriptor_execute(provider, MODEL_FIT_LABELS, native_inputs[0], words, native_scalars)
        _ = curve  # keep the Python object alive through the native call above
        _ = previous  # keep the Python object alive through the native call above
        _ = state  # keep the Python object alive through the native call above
        _ = variance  # keep the Python object alive through the native call above
        _ = moment  # keep the Python object alive through the native call above
        _ = params  # keep the Python object alive through the native call above
        _ = classes_full  # keep the Python object alive through the native call above
        var count = class_count[0]
        py_setitem(fields, "classes_", py_call(context["ndarray"], Python.tuple(count), dtype=classes_full.dtype, buffer=classes_full))
    else:
        var words = model_descriptor_words(native_program, native_options, native_inputs[1])
        n_iter = neural_descriptor_execute(provider, MODEL_FIT, native_inputs[0], words, native_scalars)
        _ = curve  # keep the Python object alive through the native call above
        _ = previous  # keep the Python object alive through the native call above
        _ = state  # keep the Python object alive through the native call above
        _ = variance  # keep the Python object alive through the native call above
        _ = moment  # keep the Python object alive through the native call above
        _ = params  # keep the Python object alive through the native call above
    # Raw addresses never own data. Keep every source alive through the
    # synchronous native model call, including temporary sequence buffers.
    _ = x^
    _ = y^
    _ = previous^
    _ = vx^
    _ = vy^
    py_setitem(fields, "params_", params)
    py_setitem(fields, "n_iter_", n_iter)
    if validation is not None:
        var info = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(py=sequential_address(val_info)))
        _ = val_info  # keep the Python object alive through the native call above
        var ran = Int(info[0])
        _ = val_info  # keep the Python object alive through the native call above
        py_setitem(fields, "epochs_run_", ran)
        py_setitem(fields, "best_epoch_", PythonObject(info[1]))
        _ = val_info  # keep the Python object alive through the native call above
        py_setitem(fields, "validation_loss_curve_", sequential_curve(val_curve, ran))
        _ = val_info^
    py_setitem(fields, "loss_curve_", sequential_curve(curve, n_iter))
    py_setitem(fields, "n_features_in_", d)
    py_setitem(fields, "_sample_shape", sample)
    py_setitem(fields, "n_outputs_", n_out)
    py_setitem(fields, "_fit_loss", options["loss"])
    py_setitem(fields, "_target", target)
    py_setitem(fields, "_fit_batch", batch_size)
    py_setitem(fields, "_fit_shuffle", Bool(py=shuffle))
    py_setitem(fields, "_opt_m", moment)
    py_setitem(fields, "_opt_v", variance)
    py_setitem(fields, "_fit_state", state)
    py_setitem(fields, "_curve", curve)
    py_setitem(fields, "_imported", False)
    return model


def sequential_forward(model: PythonObject, source: PythonObject, transform: Int,
                       width: PythonObject, entries: PythonObject, context: PythonObject) raises -> PythonObject:
    # An overload, not a `classes = PythonObject(None)` default: a non-raising
    # caller (the abi("C") door) compile-time evaluates a default, and a
    # PythonObject cannot be built at compile time.
    return sequential_forward(model, source, transform, width, entries, context, PythonObject(None))


def sequential_forward(model: PythonObject, source: PythonObject, transform: Int,
                       width: PythonObject, entries: PythonObject, context: PythonObject,
                       classes: PythonObject) raises -> PythonObject:
    var fields = sequential_fields(model)
    if not py_contains(fields, "params_"):
        raise Error("call fit before predict")
    var sample = py_get(fields, "_sample_shape")
    var x_snapshot = sequential_x_snapshot(source, "X", context)
    var x = x_snapshot.keep
    var rank = len(x_snapshot.shape)
    if sample is None and rank != 2:
        raise Error("X must be a 2-D array of shape (n_samples, n_features)")
    if sample is not None and rank > 2:
        var actual = Python.list()
        for k in range(1, rank):  # driver-loop: over the prediction input sample rank
            _ = py_call(actual.append, PythonObject(x_snapshot.shape[k]))
        if not Bool(py=py_call(Python.import_module("builtins").tuple, actual) == sample):
            raise Error("X samples have a different shape from the fitted model")
    if not Bool(py=PythonObject(x_snapshot.source.cols) == fields["n_features_in_"]):
        raise Error("X has a different feature count from the fitted model")
    var params = fields["params_"]
    var native_program = _ints(sequential_program(model, fields["n_features_in_"], sample))
    var native_table = model_float64_words(sequential_table(model))
    var sources: List[CastIn] = [x_snapshot.source]
    var provider = entries["native_model_provider"]
    var out: PythonObject
    if classes is not None:
        # the labels of a classifier, in classes_' dtype
        out = sequential_array(Python.tuple(x_snapshot.source.rows), classes.dtype, context)
        var addresses: List[Int] = [0, Int(py=sequential_address(params)),
                                    Int(py=sequential_address(classes)), Int(py=sequential_address(out))]
        var options: List[Int] = [x_snapshot.source.rows, 4096, transform, Int(py=classes.shape[0]), Int(py=classes.dtype.itemsize)]
        var words = model_descriptor_words(native_program, options, sources)
        _ = neural_descriptor_execute(provider, MODEL_PREDICT_LABELS, addresses, words, native_table)
        _ = params  # keep the Python object alive through the native call above
        _ = x^
        _ = params^
        return out
    var output_width = fields["n_outputs_"] if width is None else width
    out = sequential_array(Python.tuple(x_snapshot.source.rows, output_width), context["float32"], context)
    var addresses: List[Int] = [0, Int(py=sequential_address(params)), Int(py=sequential_address(out))]
    var options: List[Int] = [x_snapshot.source.rows, 4096, transform]
    var words = model_descriptor_words(native_program, options, sources)
    _ = neural_descriptor_execute(provider, MODEL_PREDICT, addresses, words, native_table)
    _ = x^
    _ = params^
    return out


def sequential_predict(model: PythonObject, source: PythonObject, probabilities: Bool,
                       context: PythonObject) raises -> PythonObject:
    var fields = sequential_fields(model)
    var saved_loss = py_get(fields, "_fit_loss")
    var loss = "" if saved_loss is None else String(saved_loss)
    var entries = sequential_entries(model, context)
    var classifier = loss == "cross_entropy" or loss == "bce_with_logits"
    if probabilities:
        if loss == "cross_entropy":
            return sequential_forward(model, source, 1, PythonObject(None), entries, context)
        if loss == "bce_with_logits":
            return sequential_forward(model, source, 3, PythonObject(2), entries, context)
        raise Error("predict_proba needs a model fitted with cross_entropy or bce_with_logits")
    if classifier:
        # one entry: the argmax (threshold) codes stay on the device and are
        # gathered there into the labels (`model_predict_labels`)
        return sequential_forward(model, source, 4 if loss == "cross_entropy" else 5, PythonObject(1), entries,
                                  context, fields["classes_"])
    var result = sequential_forward(model, source, 0, PythonObject(None), entries, context)
    if Bool(py=py_get(fields, "_y_1d", False)):
        # A one-output model is already contiguous. The native ndarray
        # constructor changes only its ABI rank, preserving the returned words.
        return py_call(context["ndarray"], Python.tuple(result.shape[0]), dtype=context["float32"], buffer=result)
    return result
