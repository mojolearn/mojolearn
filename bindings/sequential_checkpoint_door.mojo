# SPDX-License-Identifier: Apache-2.0
# Complete Sequential checkpoint/export and verification-sidecar operation.
# The existing model checkpoint format and native model ABI remain unchanged.
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.native_loaded_module_door import NativeModules
from bindings.py_balanced import py_call, py_contains, py_get, py_setitem
from bindings.py_typed_call import py_call_typed
from bindings.native_json_door import native_path
from bindings.sequential_description_unpack import sequential_fields, sequential_append
from bindings.sequential_description_pack import sequential_program, sequential_table
from bindings.sequential_execution_pack import sequential_entries, sequential_forward
from bindings.sequential_arrays_door import sequential_array, sequential_address, sequential_descriptor, sequential_curve
from sequence.sequential_checkpoint_run import _quote
from bindings.sequential_keys_unpack import sequential_keys
from bindings.py_balanced import py_item


def _report_array(value: PythonObject, name: String, input: Bool, entries: PythonObject) raises -> String:
    var nk = sequential_keys()
    var result = "{\"name\":" + _quote(name) + ",\"shape\":["
    for k in range(len(value.shape)):  # driver-loop: over verification array shape metadata
        if k > 0:
            result += ","
        result += String(Int(py=value.shape[k]))
    result += "]"
    if input:
        result += ",\"dtype\":\"float32\""
    var digest = py_call_typed(py_item(entries, nk[].k_verify_digest), sequential_address(value), value.nbytes)
    return result + ",\"digest\":" + _quote(String(digest)) + "}"


def _sidecar(mut mods: NativeModules, model: PythonObject, path: PythonObject, entries: PythonObject, context: PythonObject) raises:
    var nk = sequential_keys()
    var fields = sequential_fields(model)
    var x = sequential_array(mods, Python.tuple(16, py_item(fields, nk[].k_n_features_in_)), py_item(context, nk[].k_float32), context)
    _ = py_call_typed(py_item(entries, nk[].k_verify_fill), sequential_address(x), x.size, PythonObject(0), PythonObject(0))
    mods.released()
    var output = sequential_forward(mods, model, x, 0, PythonObject(None), entries, context)
    var input_json = _report_array(x, "X", True, entries)
    mods.released()
    var output_json = _report_array(output, "output", False, entries)
    mods.released()
    var sha = py_call_typed(py_item(entries, nk[].k_verify_file_sha256), path)
    mods.released()
    var info = py_call_typed(py_item(entries, nk[].k_verify_model_info), path)
    mods.released()
    var file = String(path)
    var parts = file.split("/")
    var uname = py_item(context, nk[].k_uname)()
    mods.released()
    var vendor = String(py_item(entries, nk[].k_x_sequence_vendor)())
    mods.released()
    var mode = "identical" if Int(py=py_item(entries, nk[].k_x_sequence_numeric_mode)()) == 1 else "fast"
    mods.released()
    var report = "{\"format\":\"mojolearn-model-verify/1\",\"install\":{\"mojolearn_version\":"
    report += _quote(String(py_item(context, nk[].k_version))) + ",\"numeric_mode\":" + _quote(mode)
    report += ",\"vendor\":" + _quote(vendor) + ",\"device\":" + _quote(vendor + " native executor")
    report += ",\"platform\":" + _quote(String(uname[0]) + " " + String(uname[4])) + "},\"model\":{\"file\":"
    report += _quote(String(parts[len(parts) - 1])) + ",\"kind\":" + _quote(String(info[0]))
    report += ",\"sha256\":" + _quote(String(sha[0])) + ",\"bytes\":" + String(Int(py=sha[1]))
    report += ",\"description_hash\":" + _quote(String(info[1])) + "},\"input_source\":\"seeded\",\"seed\":0,\"inputs\":["
    report += input_json + "],\"outputs\":[" + output_json + "]}\n"
    with open(file + ".verify.json", "w") as stream:
        stream.write(report)
        mods.released()


def sequential_save(mut mods: NativeModules, model: PythonObject, path: PythonObject, export: Bool, context: PythonObject) raises -> PythonObject:
    var nk = sequential_keys()
    var fields = sequential_fields(model)
    if not py_contains(fields, nk[].k_params_) or not py_contains(fields, nk[].k__target):
        raise Error("call fit before saving the model")
    if not export and py_get(fields, nk[].k__fit_state) is None:
        raise Error("this model holds no training state (it was loaded from an export); use export")
    if py_get(fields, nk[].k__sample_shape) is not None:
        raise Error("save_checkpoint and export take models of 2-D inputs without reshape, permute, flatten, conv2d or pooling layers")
    var options = py_item(fields, nk[].k__options)
    var classes = py_get(fields, nk[].k_classes_)
    var count = 0 if classes is None else Int(py=classes.shape[0])
    var ip = Python.list(1 if export else 0, py_item(context, nk[].k_loss_codes)[py_item(options, nk[].k_loss)], py_item(fields, nk[].k__target),
        py_item(options, nk[].k_opt), py_item(options, nk[].k_flags), py_item(options, nk[].k_sched), py_item(options, nk[].k_step_size), py_item(options, nk[].k_n_iter_no_change),
        py_item(fields, nk[].k__fit_batch), py_item(fields, nk[].k__fit_shuffle), py_item(fields, nk[].k_n_features_in_), py_item(fields, nk[].k_n_outputs_),
        py_get(fields, nk[].k__y_1d, False), PythonObject(0) if export else py_item(fields, nk[].k_n_iter_), count)
    var fp = Python.list(py_item(options, nk[].k_lr), py_item(options, nk[].k_f1), py_item(options, nk[].k_f2), py_item(options, nk[].k_eps), py_item(options, nk[].k_weight_decay),
        0.0, py_item(options, nk[].k_tol), py_item(options, nk[].k_gamma), py_item(options, nk[].k_huber_delta))
    _ = py_call(fp.extend, sequential_table(model))
    var addresses = Python.list(sequential_address(py_item(fields, nk[].k_params_)), 0, 0, 0, 0)
    if not export:
        py_setitem(addresses, 1, sequential_address(py_item(fields, nk[].k_optimizer_m_)))
        py_setitem(addresses, 2, sequential_address(py_item(fields, nk[].k_optimizer_v_)))
        py_setitem(addresses, 3, sequential_address(py_item(fields, nk[].k__fit_state)))
        py_setitem(addresses, 4, sequential_address(py_item(fields, nk[].k__curve)))
    var descriptor = PythonObject(None) if count == 0 else sequential_descriptor(classes, "classes")
    var program = sequential_program(mods, model, py_item(fields, nk[].k_n_features_in_), PythonObject(None))
    var entries = sequential_entries(model, context)
    mods.released()
    var normalized = native_path(path)
    _ = py_call_typed(py_item(entries, nk[].k_model_save_checkpoint), normalized, addresses, program, ip, fp, descriptor)
    mods.released()
    _sidecar(mods, model, normalized, entries, context)
    _ = fields^
    _ = classes^
    return PythonObject(None)


def sequential_load_checkpoint(mut mods: NativeModules, cls: PythonObject, path: PythonObject, mode: PythonObject,
                               context: PythonObject) raises -> PythonObject:
    var nk = sequential_keys()
    var b = native_loaded_module("builtins")
    var model = py_call(b.object.__new__, cls)
    var fields = sequential_fields(model)
    py_setitem(fields, nk[].k_numeric_mode, mode)
    py_setitem(fields, nk[].k_input_shape, PythonObject(None))
    py_setitem(fields, nk[].k__layers, Python.list())
    py_setitem(fields, nk[].k__options, PythonObject(None))
    var entries = sequential_entries(model, context)
    mods.released()
    var normalized = native_path(path)
    var description = py_call_typed(py_item(entries, nk[].k_model_checkpoint_info), normalized)
    mods.released()
    var export = Bool(py=description[0])
    var layers = description[1]
    var ip = description[2]
    var fp = description[3]
    for k in range(len(layers)):  # driver-loop: over checkpoint layer records
        var layer = layers[k]
        sequential_append(mods, model, layer[0], layer[1], layer[2], layer[3], PythonObject(None))
    var loss_name = py_item(context, nk[].k_loss_names)[ip[0]]
    var options = b.dict()
    py_setitem(options, nk[].k_loss, loss_name)
    py_setitem(options, nk[].k_opt, ip[2])
    py_setitem(options, nk[].k_flags, ip[3])
    py_setitem(options, nk[].k_sched, ip[4])
    py_setitem(options, nk[].k_step_size, ip[5])
    py_setitem(options, nk[].k_n_iter_no_change, ip[6])
    py_setitem(options, nk[].k_lr, fp[0])
    py_setitem(options, nk[].k_f1, fp[1])
    py_setitem(options, nk[].k_f2, fp[2])
    py_setitem(options, nk[].k_eps, fp[3])
    py_setitem(options, nk[].k_weight_decay, fp[4])
    py_setitem(options, nk[].k_tol, fp[6])
    py_setitem(options, nk[].k_gamma, fp[7])
    py_setitem(options, nk[].k_huber_delta, fp[8])
    py_setitem(fields, nk[].k__options, options)
    var params = sequential_array(mods, Python.tuple(ip[16]), py_item(context, nk[].k_float32), context)
    # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
    var classes = Optional[PythonObject]()
    if Int(py=ip[13]) > 0:
        classes = sequential_array(mods, Python.tuple(ip[13]), py_item(context, nk[].k_class_dtypes)[Python.tuple(ip[14], ip[15])], context)
    var addresses = Python.list(sequential_address(params), 0, 0, 0, 0,
        PythonObject(0) if not classes else sequential_address(classes.value()))
    var m = PythonObject(None)
    var v = PythonObject(None)
    var state = PythonObject(None)
    var curve = PythonObject(None)
    if not export:
        m = sequential_array(mods, Python.tuple(ip[16]), py_item(context, nk[].k_float32), context)
        v = sequential_array(mods, Python.tuple(ip[16]), py_item(context, nk[].k_float32), context)
        state = sequential_array(mods, Python.tuple(12), py_item(context, nk[].k_int64), context)
        curve = sequential_array(mods, Python.tuple(ip[12]), py_item(context, nk[].k_float32), context)
        py_setitem(addresses, 1, sequential_address(m))
        py_setitem(addresses, 2, sequential_address(v))
        py_setitem(addresses, 3, sequential_address(state))
        py_setitem(addresses, 4, sequential_address(curve))
    _ = py_call_typed(py_item(entries, nk[].k_model_load_checkpoint), normalized, addresses)
    mods.released()
    py_setitem(fields, nk[].k_params_, params)
    py_setitem(fields, nk[].k_n_features_in_, ip[9])
    py_setitem(fields, nk[].k_n_outputs_, ip[10])
    py_setitem(fields, nk[].k__fit_loss, loss_name)
    py_setitem(fields, nk[].k__target, ip[1])
    py_setitem(fields, nk[].k__fit_batch, ip[7])
    py_setitem(fields, nk[].k__fit_shuffle, py_call(b.bool, ip[8]))
    py_setitem(fields, nk[].k__y_1d, py_call(b.bool, ip[11]))
    py_setitem(fields, nk[].k__sample_shape, PythonObject(None))
    if Bool(classes):
        py_setitem(fields, nk[].k_classes_, classes.value())
    py_setitem(fields, nk[].k_n_iter_, PythonObject(0) if export else ip[12])
    py_setitem(fields, nk[].k_loss_curve_, Python.list() if export else sequential_curve(curve, Int(py=ip[12])))
    py_setitem(fields, nk[].k__fit_state, state)
    if not export:
        py_setitem(fields, nk[].k_optimizer_m_, m)
        py_setitem(fields, nk[].k_optimizer_v_, v)
        py_setitem(fields, nk[].k__curve, curve)
    return model
