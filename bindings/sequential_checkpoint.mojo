# SPDX-License-Identifier: Apache-2.0
# Complete Sequential checkpoint/export and verification-sidecar operation.
# The existing model checkpoint format and native model ABI remain unchanged.
from std.collections.string import chr
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_contains, py_get, py_setitem, py_call_typed
from bindings.native_json import native_path
from bindings.sequential_description import sequential_fields, sequential_append, sequential_program, sequential_table
from bindings.sequential_execution import sequential_entries, sequential_forward
from bindings.sequential_arrays import sequential_array, sequential_address, sequential_descriptor, sequential_curve


def _quote(value: String) -> String:
    var result = value.replace("\\", "\\\\").replace("\"", "\\\"")
    var digits = "0123456789abcdef"
    for k in range(32):  # small-loop(32: JSON control-character alphabet): report text encoding
        result = result.replace(chr(k), "\\u00" + String(digits[byte=k // 16]) + String(digits[byte=k % 16]))
    return "\"" + result + "\""


def _report_array(value: PythonObject, name: String, input: Bool, entries: PythonObject) raises -> String:
    var result = "{\"name\":" + _quote(name) + ",\"shape\":["
    for k in range(len(value.shape)):  # driver-loop: over verification array shape metadata
        if k > 0:
            result += ","
        result += String(Int(py=value.shape[k]))
    result += "]"
    if input:
        result += ",\"dtype\":\"float32\""
    var digest = py_call_typed(entries["verify_digest"], sequential_address(value), value.nbytes)
    return result + ",\"digest\":" + _quote(String(digest)) + "}"


def _sidecar(model: PythonObject, path: PythonObject, entries: PythonObject, context: PythonObject) raises:
    var fields = sequential_fields(model)
    var x = sequential_array(Python.tuple(16, fields["n_features_in_"]), context["float32"], context)
    _ = py_call_typed(entries["verify_fill"], sequential_address(x), x.size, PythonObject(0), PythonObject(0))
    var output = sequential_forward(model, x, 0, PythonObject(None), entries, context)
    var input_json = _report_array(x, "X", True, entries)
    var output_json = _report_array(output, "output", False, entries)
    var sha = py_call_typed(entries["verify_file_sha256"], path)
    var info = py_call_typed(entries["verify_model_info"], path)
    var file = String(path)
    var parts = file.split("/")
    var uname = context["uname"]()
    var vendor = String(entries["x_sequence_vendor"]())
    var mode = "identical" if Int(py=entries["x_sequence_numeric_mode"]()) == 1 else "fast"
    var report = "{\"format\":\"mojolearn-model-verify/1\",\"install\":{\"mojolearn_version\":"
    report += _quote(String(context["version"])) + ",\"numeric_mode\":" + _quote(mode)
    report += ",\"vendor\":" + _quote(vendor) + ",\"device\":" + _quote(vendor + " native executor")
    report += ",\"platform\":" + _quote(String(uname[0]) + " " + String(uname[4])) + "},\"model\":{\"file\":"
    report += _quote(String(parts[len(parts) - 1])) + ",\"kind\":" + _quote(String(info[0]))
    report += ",\"sha256\":" + _quote(String(sha[0])) + ",\"bytes\":" + String(Int(py=sha[1]))
    report += ",\"description_hash\":" + _quote(String(info[1])) + "},\"input_source\":\"seeded\",\"seed\":0,\"inputs\":["
    report += input_json + "],\"outputs\":[" + output_json + "]}\n"
    with open(file + ".verify.json", "w") as stream:
        stream.write(report)


def sequential_save(model: PythonObject, path: PythonObject, export: Bool, context: PythonObject) raises -> PythonObject:
    var fields = sequential_fields(model)
    if not py_contains(fields, "params_") or not py_contains(fields, "_target"):
        raise Error("call fit before saving the model")
    if not export and py_get(fields, "_fit_state") is None:
        raise Error("this model holds no training state (it was loaded from an export); use export")
    if py_get(fields, "_sample_shape") is not None:
        raise Error("save_checkpoint and export take models of 2-D inputs without reshape, permute, flatten, conv2d or pooling layers")
    var options = fields["_options"]
    var classes = py_get(fields, "classes_")
    var count = 0 if classes is None else Int(py=classes.shape[0])
    var ip = Python.list(1 if export else 0, context["loss_codes"][options["loss"]], fields["_target"],
        options["opt"], options["flags"], options["sched"], options["step_size"], options["n_iter_no_change"],
        fields["_fit_batch"], fields["_fit_shuffle"], fields["n_features_in_"], fields["n_outputs_"],
        py_get(fields, "_y_1d", False), PythonObject(0) if export else fields["n_iter_"], count)
    var fp = Python.list(options["lr"], options["f1"], options["f2"], options["eps"], options["weight_decay"],
        0.0, options["tol"], options["gamma"], options["huber_delta"])
    _ = py_call(fp.extend, sequential_table(model))
    var addresses = Python.list(sequential_address(fields["params_"]), 0, 0, 0, 0)
    if not export:
        py_setitem(addresses, 1, sequential_address(fields["_opt_m"]))
        py_setitem(addresses, 2, sequential_address(fields["_opt_v"]))
        py_setitem(addresses, 3, sequential_address(fields["_fit_state"]))
        py_setitem(addresses, 4, sequential_address(fields["_curve"]))
    var descriptor = PythonObject(None) if count == 0 else sequential_descriptor(classes, "classes")
    var program = sequential_program(model, fields["n_features_in_"], PythonObject(None))
    var entries = sequential_entries(model, context)
    var normalized = native_path(path)
    _ = py_call_typed(entries["model_save_checkpoint"], normalized, addresses, program, ip, fp, descriptor)
    _sidecar(model, normalized, entries, context)
    _ = fields^
    _ = classes^
    return PythonObject(None)


def sequential_load_checkpoint(cls: PythonObject, path: PythonObject, mode: PythonObject,
                               context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var model = py_call(b.object.__new__, cls)
    var fields = sequential_fields(model)
    py_setitem(fields, "numeric_mode", mode)
    py_setitem(fields, "input_shape", PythonObject(None))
    py_setitem(fields, "_layers", Python.list())
    py_setitem(fields, "_options", PythonObject(None))
    var entries = sequential_entries(model, context)
    var normalized = native_path(path)
    var description = py_call_typed(entries["model_checkpoint_info"], normalized)
    var export = Bool(py=description[0])
    var layers = description[1]
    var ip = description[2]
    var fp = description[3]
    for k in range(len(layers)):  # driver-loop: over checkpoint layer records
        var layer = layers[k]
        sequential_append(model, layer[0], layer[1], layer[2], layer[3], PythonObject(None))
    var loss_name = context["loss_names"][ip[0]]
    var options = b.dict()
    py_setitem(options, "loss", loss_name)
    py_setitem(options, "opt", ip[2])
    py_setitem(options, "flags", ip[3])
    py_setitem(options, "sched", ip[4])
    py_setitem(options, "step_size", ip[5])
    py_setitem(options, "n_iter_no_change", ip[6])
    py_setitem(options, "lr", fp[0])
    py_setitem(options, "f1", fp[1])
    py_setitem(options, "f2", fp[2])
    py_setitem(options, "eps", fp[3])
    py_setitem(options, "weight_decay", fp[4])
    py_setitem(options, "tol", fp[6])
    py_setitem(options, "gamma", fp[7])
    py_setitem(options, "huber_delta", fp[8])
    py_setitem(fields, "_options", options)
    var params = sequential_array(Python.tuple(ip[16]), context["float32"], context)
    var classes = PythonObject(None)
    if Int(py=ip[13]) > 0:
        classes = sequential_array(Python.tuple(ip[13]), context["class_dtypes"][Python.tuple(ip[14], ip[15])], context)
    var addresses = Python.list(sequential_address(params), 0, 0, 0, 0,
        PythonObject(0) if classes is None else sequential_address(classes))
    var m = PythonObject(None)
    var v = PythonObject(None)
    var state = PythonObject(None)
    var curve = PythonObject(None)
    if not export:
        m = sequential_array(Python.tuple(ip[16]), context["float32"], context)
        v = sequential_array(Python.tuple(ip[16]), context["float32"], context)
        state = sequential_array(Python.tuple(12), context["int64"], context)
        curve = sequential_array(Python.tuple(ip[12]), context["float32"], context)
        py_setitem(addresses, 1, sequential_address(m))
        py_setitem(addresses, 2, sequential_address(v))
        py_setitem(addresses, 3, sequential_address(state))
        py_setitem(addresses, 4, sequential_address(curve))
    _ = py_call_typed(entries["model_load_checkpoint"], normalized, addresses)
    py_setitem(fields, "params_", params)
    py_setitem(fields, "n_features_in_", ip[9])
    py_setitem(fields, "n_outputs_", ip[10])
    py_setitem(fields, "_fit_loss", loss_name)
    py_setitem(fields, "_target", ip[1])
    py_setitem(fields, "_fit_batch", ip[7])
    py_setitem(fields, "_fit_shuffle", py_call(b.bool, ip[8]))
    py_setitem(fields, "_y_1d", py_call(b.bool, ip[11]))
    py_setitem(fields, "_sample_shape", PythonObject(None))
    if classes is not None:
        py_setitem(fields, "classes_", classes)
    py_setitem(fields, "n_iter_", PythonObject(0) if export else ip[12])
    py_setitem(fields, "loss_curve_", Python.list() if export else sequential_curve(curve, Int(py=ip[12])))
    py_setitem(fields, "_fit_state", state)
    if not export:
        py_setitem(fields, "_opt_m", m)
        py_setitem(fields, "_opt_v", v)
        py_setitem(fields, "_curve", curve)
    return model
