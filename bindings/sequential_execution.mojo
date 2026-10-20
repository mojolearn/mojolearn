# SPDX-License-Identifier: Apache-2.0
"""Native Sequential orchestration of the existing model numerical ABI."""
from std.python import Python, PythonObject
from memory import UnsafePointer, MutAnyOrigin
from bindings.sequential_description import sequential_fields, sequential_sample, sequential_program, sequential_table, _integer
from bindings.sequential_arrays import sequential_array, sequential_address, sequential_input, sequential_descriptor, sequential_x, sequential_curve


def sequential_entries(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var mode = sequential_fields(model)["numeric_mode"]
    if mode is None:
        mode = context["backend_fields"]["_DEFAULT_MODE"]
    if not Bool(py=b.isinstance(mode, b.str)):
        raise Error("numeric_mode must be fast, identical or None")
    mode = mode.strip().lower()
    if String(mode) != "identical" and String(mode) != "fast":
        raise Error("numeric_mode must be fast, identical or None")
    var entries = context["bindings"][mode]
    if Bool(py=b.isinstance(entries, b.str)):
        raise Error(String(entries))
    var actual_mode = Int(py=entries["x_sequence_numeric_mode"]())
    var actual_vendor = String(entries["x_sequence_vendor"]())
    if actual_mode != (1 if String(mode) == "identical" else 0):
        raise Error("RuntimeError: Sequential native numerical mode differs from the requested mode")
    if actual_vendor != String(context["vendor"]):
        raise Error("RuntimeError: Sequential native vendor differs from the selected execution route")
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
        var classes = sequential_array(Python.tuple(n), y.dtype, context)
        var codes = sequential_array(Python.tuple(n), context["int32"], context)
        var count = Int(py=entries["model_labels"](descriptor, sequential_address(classes), sequential_address(codes)))
        var selected = context["ndarray"](Python.tuple(count), dtype=y.dtype, buffer=classes)
        fields["classes_"] = selected
        if loss == "cross_entropy":
            if count != n_out:
                raise Error("the labels have " + String(count) + " classes but the model has " + String(n_out) + " outputs")
            return Python.tuple(codes, sequential_descriptor(codes, "y"), 1)
        if count != 2 or n_out != 1:
            raise Error("bce_with_logits needs two classes and a model with one output")
        return Python.tuple(codes, sequential_descriptor(codes, "y"), 2)
    fields["_y_1d"] = nd == 1
    var width = 1 if nd == 1 else Int(py=y.shape[1])
    if width != n_out:
        raise Error("y has " + String(width) + " outputs but the model has " + String(n_out))
    return Python.tuple(y, descriptor, 0)


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
    if resume and fields.get("_fit_state") is None:
        raise Error("resume=True continues an earlier fit: call fit (or from_checkpoint) first")
    var batch = args[3]
    var shuffle = args[4]
    if batch is None:
        batch = fields["_fit_batch"] if resume else PythonObject(32)
    if shuffle is None:
        shuffle = fields["_fit_shuffle"] if resume else PythonObject(True)
    var epochs = Int(py=b.int(args[2]))
    var batch_size = Int(py=b.int(batch))
    if epochs < 1 or batch_size < 1:
        raise Error("epochs and batch_size must be >= 1")
    var x = sequential_x(args[0], "X", context)
    var xd = sequential_descriptor(x, "X", True)
    var n = Int(py=xd[3])
    var d = Int(py=xd[4])
    if resume and d != Int(py=fields["n_features_in_"]):
        raise Error("X has a different feature count from the fitted model")
    var sample = sequential_sample(model, x)
    if resume and not Bool(py=sample == fields.get("_sample_shape")):
        raise Error("resume=True takes inputs of the shape the model was fitted on")
    var warm = not resume and Bool(py=fields.get("_imported", False))
    if warm and (d != Int(py=fields["n_features_in_"]) or not Bool(py=sample == fields.get("_sample_shape"))):
        raise Error("X has a different feature count or sample shape from the imported weights")
    var program = sequential_program(model, PythonObject(d), sample)
    var table = sequential_table(model)
    var entries = sequential_entries(model, context)
    var counts = entries["model_n_params"](program, table)
    var n_params = Int(py=counts[0])
    var n_out = Int(py=counts[1])
    var target_data = sequential_targets(model, args[1], n, n_out, String(options["loss"]), entries, context)
    var y = target_data[0]
    var yd = target_data[1]
    var target = Int(py=target_data[2])
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
    seed = b.int(seed)
    var ip = Python.list(n, context["loss_codes"][options["loss"]], target, options["opt"], options["flags"],
        options["sched"], options["step_size"], batch_size, epochs, 1 if Bool(py=shuffle) else 0,
        seed.__and__(4294967295), seed.__rshift__(32), options["n_iter_no_change"],
        0 if resume or warm else 1, 1 if resume else 0)
    var fp = Python.list(options["lr"], options["f1"], options["f2"], options["eps"],
        options["weight_decay"], 0.0, options["tol"], options["gamma"], options["huber_delta"])
    fp.extend(table)
    var addresses = Python.list(xd, yd, sequential_address(params), sequential_address(curve),
        sequential_address(moment), sequential_address(variance), sequential_address(state),
        PythonObject(0) if previous is None else sequential_address(previous))
    var vx = PythonObject(None)
    var vy = PythonObject(None)
    var val_curve = PythonObject(None)
    var val_info = PythonObject(None)
    if validation is not None:
        if not (Bool(py=b.isinstance(validation, b.tuple)) or Bool(py=b.isinstance(validation, b.list))) or len(validation) != 2:
            raise Error("validation_data must be (X, y)")
        vx = sequential_x(validation[0], "validation X", context)
        var vxd = sequential_descriptor(vx, "validation X", True)
        vy = sequential_input(validation[1], context)
        var vnd = Int(py=vy.ndim)
        if not Bool(py=sequential_sample(model, vx) == sample) or Int(py=vxd[4]) != d:
            raise Error("validation inputs must have the training sample shape")
        if vnd < 1 or vnd > 2 or Int(py=vy.shape[0]) != Int(py=vxd[3]):
            raise Error("validation targets must match the validation rows")
        if target != 0 and vnd != 1:
            raise Error("validation classification targets must be a vector")
        var classes = PythonObject(None) if target == 0 else sequential_descriptor(fields["classes_"], "classes")
        val_curve = sequential_array(Python.tuple(epochs), context["float32"], context)
        val_info = sequential_array(Python.tuple(2), context["int64"], context)
        addresses.extend(Python.tuple(vxd, sequential_descriptor(vy, "validation y"), classes,
                         sequential_address(val_curve), sequential_address(val_info)))
        ip.append(1 if restore_best else 0)
    var n_iter = Int(py=entries["model_fit"](addresses, program, ip, fp))
    # Raw addresses never own data. Keep every source alive through the
    # synchronous native model call, including temporary sequence buffers.
    _ = x^
    _ = y^
    _ = previous^
    _ = vx^
    _ = vy^
    fields["params_"] = params
    fields["n_iter_"] = n_iter
    if validation is not None:
        var info = UnsafePointer[Int64, MutAnyOrigin](unsafe_from_address=Int(py=sequential_address(val_info)))
        var ran = Int(info[0])
        fields["epochs_run_"] = ran
        fields["best_epoch_"] = PythonObject(info[1])
        fields["validation_loss_curve_"] = sequential_curve(val_curve, ran)
        _ = val_info^
    fields["loss_curve_"] = sequential_curve(curve, n_iter)
    fields["n_features_in_"] = d
    fields["_sample_shape"] = sample
    fields["n_outputs_"] = n_out
    fields["_fit_loss"] = options["loss"]
    fields["_target"] = target
    fields["_fit_batch"] = batch_size
    fields["_fit_shuffle"] = Bool(py=shuffle)
    fields["_opt_m"] = moment
    fields["_opt_v"] = variance
    fields["_fit_state"] = state
    fields["_curve"] = curve
    fields["_imported"] = False
    return model


def sequential_forward(model: PythonObject, source: PythonObject, transform: Int,
                       width: PythonObject, entries: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = sequential_fields(model)
    if not Bool(py=fields.__contains__("params_")):
        raise Error("call fit before predict")
    var sample = fields.get("_sample_shape")
    var x = sequential_x(source, "X", context)
    if sample is None and Int(py=x.ndim) != 2:
        raise Error("X must be a 2-D array of shape (n_samples, n_features)")
    if sample is not None and Int(py=x.ndim) > 2:
        var actual = Python.list()
        for k in range(1, Int(py=x.ndim)):  # driver-loop: over the prediction input sample rank
            actual.append(x.shape[k])
        if not Bool(py=Python.import_module("builtins").tuple(actual) == sample):
            raise Error("X samples have a different shape from the fitted model")
    var xd = sequential_descriptor(x, "X", True)
    if not Bool(py=xd[4] == fields["n_features_in_"]):
        raise Error("X has a different feature count from the fitted model")
    var output_width = fields["n_outputs_"] if width is None else width
    var out = sequential_array(Python.tuple(xd[3], output_width), context["float32"], context)
    var params = fields["params_"]
    _ = entries["model_predict"](Python.list(xd, sequential_address(params), sequential_address(out)),
        sequential_program(model, fields["n_features_in_"], sample), sequential_table(model),
        Python.list(xd[3], 4096, transform))
    _ = x^
    _ = params^
    return out


def sequential_predict(model: PythonObject, source: PythonObject, probabilities: Bool,
                       context: PythonObject) raises -> PythonObject:
    var fields = sequential_fields(model)
    var saved_loss = fields.get("_fit_loss")
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
        var codes = sequential_forward(model, source, 4 if loss == "cross_entropy" else 5, PythonObject(1), entries, context)
        var classes = fields["classes_"]
        var labels = sequential_array(Python.tuple(codes.shape[0]), classes.dtype, context)
        _ = entries["model_gather_labels"](Python.list(sequential_address(codes), sequential_address(classes), sequential_address(labels)),
            Python.list(codes.shape[0], classes.shape[0], classes.dtype.itemsize))
        _ = codes^
        _ = classes^
        return labels
    var result = sequential_forward(model, source, 0, PythonObject(None), entries, context)
    if Bool(py=fields.get("_y_1d", False)):
        # A one-output model is already contiguous. The native ndarray
        # constructor changes only its ABI rank, preserving the returned words.
        return context["ndarray"](Python.tuple(result.shape[0]), dtype=context["float32"], buffer=result)
    return result
