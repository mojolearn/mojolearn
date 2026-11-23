# SPDX-License-Identifier: Apache-2.0
# Complete Sequential checkpoint/export and verification-sidecar operation.
# The existing model checkpoint format and native model ABI remain unchanged.
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.native_loaded_module_door import NativeModules
from bindings.py_balanced import py_call, py_contains, py_get, py_setitem
from bindings.native_json_door import native_path
from bindings.sequential_description_unpack import sequential_fields, sequential_append, SequentialDescription
from bindings.sequential_description_pack import sequential_native_description
from bindings.sequential_execution_pack import sequential_entries, sequential_forward
from bindings.sequential_route_door import sequential_device_admit
from bindings.sequential_arrays_door import sequential_array, sequential_address, sequential_snapshot, sequential_curve
from bindings.sequential_model_file_door import (
    sequential_file_digest, sequential_file_fill, sequential_file_sha256, sequential_file_model_info,
    sequential_file_save, sequential_file_info, sequential_file_load,
)
from sequence.sequential_checkpoint_run import _quote
from bindings.sequential_keys_unpack import sequential_keys
from bindings.py_balanced import py_item
from autograd.model import FIT_STATE_WORDS, parse_model
from autograd.model_execution import model_float_words
from autograd.model_verify import model_input_id_range
from autograd.nn_optim import NN_SCHED_ARGS


def _report_array(value: PythonObject, name: String, input: Bool, entries: PythonObject, id_range: Int = 0) raises -> String:
    var nk = sequential_keys()
    var result = "{\"name\":" + _quote(name) + ",\"shape\":["
    for k in range(len(value.shape)):  # driver-loop: over verification array shape metadata
        if k > 0:
            result += ","
        result += String(Int(py=value.shape[k]))
    result += "]"
    if input:
        result += ",\"dtype\":\"float32\""
        if id_range > 0:
            # the seeded values are ids in [0, id_range) (verify-model replays them from this field)
            result += ",\"id_range\":" + String(id_range)
    var digest = sequential_file_digest(entries, Int(py=sequential_address(value)), Int(py=value.nbytes))
    _ = value  # the digested buffer stays owned through the native call
    return result + ",\"digest\":" + _quote(digest) + "}"


def _sidecar(mut mods: NativeModules, model: PythonObject, path: PythonObject, entries: PythonObject, context: PythonObject,
             id_range: Int) raises:
    var nk = sequential_keys()
    var fields = sequential_fields(model)
    var x = sequential_array(mods, Python.tuple(16, py_item(fields, nk[].k_n_features_in_)), py_item(context, nk[].k_float32), context)
    # a model whose first layer reads ids (an embedding) is probed with ids in its range, never floats
    _ = sequential_file_fill(entries, Int(py=sequential_address(x)), Int(py=x.size), 0, 0, id_range)
    mods.released()
    var output = sequential_forward(mods, model, x, 0, PythonObject(None), entries, context)
    var input_json = _report_array(x, "X", True, entries, id_range)
    mods.released()
    var output_json = _report_array(output, "output", False, entries)
    mods.released()
    var file = String(path)
    var sha = sequential_file_sha256(entries, file)
    mods.released()
    var info = sequential_file_model_info(entries, file)
    mods.released()
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
    report += _quote(String(parts[len(parts) - 1])) + ",\"kind\":" + _quote(info[0])
    report += ",\"sha256\":" + _quote(sha[0]) + ",\"bytes\":" + String(sha[1])
    report += ",\"description_hash\":" + _quote(info[1]) + "},\"input_source\":\"seeded\",\"seed\":0,\"inputs\":["
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
    # lane nn-integrate (2026-11-23): a model fitted on per-sample shapes (input_shape, flatten, convolutions,
    # pools) is saved with its version-2 description, which carries the sample shape and every layer argument
    var sample = py_get(fields, nk[].k__sample_shape)
    var options = py_item(fields, nk[].k__options)
    var classes = py_get(fields, nk[].k_classes_)
    var count = 0 if classes is None else Int(py=classes.shape[0])
    # The checkpoint's words, typed here once (UNPACK); the save itself is one
    # native descriptor call with no Python object.
    # nn-losses: the loss word, code + 256 * reduction, as the fit reads it
    var loss_word = Int(py=py_item(context, nk[].k_loss_codes)[py_item(options, nk[].k_loss)]) + 256 * Int(py=py_get(options, nk[].k_reduction, 0))
    var ip: List[Int] = [1 if export else 0, loss_word,
        Int(py=py_item(fields, nk[].k__target)), Int(py=py_item(options, nk[].k_opt)), Int(py=py_item(options, nk[].k_flags)),
        Int(py=py_item(options, nk[].k_sched)), Int(py=py_item(options, nk[].k_step_size)),
        Int(py=py_item(options, nk[].k_n_iter_no_change)), Int(py=py_item(fields, nk[].k__fit_batch)),
        Int(py=py_item(fields, nk[].k__fit_shuffle)), Int(py=py_item(fields, nk[].k_n_features_in_)),
        Int(py=py_item(fields, nk[].k_n_outputs_)), Int(py=py_get(fields, nk[].k__y_1d, False)),
        0 if export else Int(py=py_item(fields, nk[].k_n_iter_)), count]
    var fp: List[Float64] = [Float64(py=py_item(options, nk[].k_lr)), Float64(py=py_item(options, nk[].k_f1)),
        Float64(py=py_item(options, nk[].k_f2)), Float64(py=py_item(options, nk[].k_eps)),
        Float64(py=py_item(options, nk[].k_weight_decay)), Float64(py=py_get(options, nk[].k_f7, PythonObject(0.0))),
        Float64(py=py_item(options, nk[].k_tol)),
        Float64(py=py_item(options, nk[].k_gamma)), Float64(py=py_item(options, nk[].k_huber_delta))]
    var buffers: List[Int] = [Int(py=sequential_address(py_item(fields, nk[].k_params_))), 0, 0, 0, 0]
    if not export:
        buffers[1] = Int(py=sequential_address(py_item(fields, nk[].k_optimizer_m_)))
        buffers[2] = Int(py=sequential_address(py_item(fields, nk[].k_optimizer_v_)))
        buffers[3] = Int(py=sequential_address(py_item(fields, nk[].k__fit_state)))
        buffers[4] = Int(py=sequential_address(py_item(fields, nk[].k__curve)))
    # The classes_ descriptor words (the compiled model ABI's seven).
    var class_words = List[Int]()
    if count > 0:
        var described = sequential_snapshot(classes, "classes")
        class_words = [described.source.addr, described.dtype_num, described.source.itemsize,
                       described.source.rows, described.source.cols, described.source.s0, described.source.s1]
    # The model description as fit and predict hand it to the same engine:
    # the retained native program words and one scalar per layer.
    var description_owner = sequential_native_description(mods, model, py_item(fields, nk[].k_n_features_in_), sample)
    var description = description_owner.downcast_value_ptr[SequentialDescription]()
    var program = description[].program.copy()
    # the ids the model's first layer reads (0: it reads values), for the verify report's seeded probe
    var id_range = model_input_id_range(parse_model(program, model_float_words(description[].scalars)))
    for k in range(len(description[].scalars)):  # driver-loop: over layer scalar options
        fp.append(description[].scalars[k])
    # lane nn-optim: the schedule's argument words after the layers' (autograd/model_checkpoint.mojo record v2)
    var sched_args = py_get(options, nk[].k_sched_args)
    if sched_args is not None and len(sched_args) == NN_SCHED_ARGS:
        for k in range(NN_SCHED_ARGS):  # small-loop(8: the schedule's argument words)
            fp.append(Float64(py=sched_args[k]))
    _ = description_owner^
    var entries = sequential_entries(model, context)
    mods.released()
    var normalized = native_path(path)
    _ = sequential_file_save(entries, String(normalized), buffers, program, ip, fp, class_words)
    mods.released()
    _sidecar(mods, model, normalized, entries, context, id_range)
    _ = fields^
    _ = classes^
    return PythonObject(None)


def sequential_load_checkpoint(mut mods: NativeModules, cls: PythonObject, path: PythonObject, mode: PythonObject,
                               device: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = sequential_keys()
    var b = native_loaded_module("builtins")
    # The loaded model runs on `device` (None: this installation's device),
    # checked before the model exists, as the constructor checks it.
    sequential_device_admit(device, context)
    var model = py_call(b.object.__new__, cls)
    var fields = sequential_fields(model)
    py_setitem(fields, nk[].k_numeric_mode, mode)
    py_setitem(fields, nk[].k_device, device)
    py_setitem(fields, nk[].k_input_shape, PythonObject(None))
    py_setitem(fields, nk[].k__layers, Python.list())
    py_setitem(fields, nk[].k__options, PythonObject(None))
    var entries = sequential_entries(model, context)
    mods.released()
    var normalized = native_path(path)
    var file = String(normalized)
    var description = sequential_file_info(entries, file)
    mods.released()
    var export = description.is_export
    ref ip = description.ints
    ref fp = description.floats
    if len(ip) < 17 or len(fp) < 9:
        raise Error("RuntimeError: Sequential checkpoint description is incomplete")
    var n_args = len(description.layer_args) // max(1, len(description.layer_floats))
    for k in range(len(description.layer_floats)):  # driver-loop: over checkpoint layer records
        # every argument word of the layer (lane nn-integrate: three words alone lost a block layer's a2..a5)
        var supplied = Python.list()
        for j in range(n_args):  # small-loop(six layer argument words): description metadata, no data
            _ = py_call(supplied.append, PythonObject(description.layer_args[n_args * k + j]))
        sequential_append(mods, model, PythonObject(description.layer_ints[3 * k]), PythonObject(description.layer_ints[3 * k + 1]),
                          PythonObject(description.layer_ints[3 * k + 2]), PythonObject(description.layer_floats[k]),
                          supplied if n_args > 0 else PythonObject(None))
    # nn-losses: the loss word, code + 256 * reduction
    var loss_name = py_item(context, nk[].k_loss_names)[PythonObject(ip[0] % 256)]
    var options = b.dict()
    py_setitem(options, nk[].k_loss, loss_name)
    py_setitem(options, nk[].k_reduction, PythonObject(ip[0] // 256))
    py_setitem(options, nk[].k_opt, PythonObject(ip[2]))
    py_setitem(options, nk[].k_flags, PythonObject(ip[3]))
    py_setitem(options, nk[].k_sched, PythonObject(ip[4]))
    py_setitem(options, nk[].k_step_size, PythonObject(ip[5]))
    py_setitem(options, nk[].k_n_iter_no_change, PythonObject(ip[6]))
    py_setitem(options, nk[].k_lr, PythonObject(fp[0]))
    py_setitem(options, nk[].k_f1, PythonObject(fp[1]))
    py_setitem(options, nk[].k_f2, PythonObject(fp[2]))
    py_setitem(options, nk[].k_eps, PythonObject(fp[3]))
    py_setitem(options, nk[].k_weight_decay, PythonObject(fp[4]))
    py_setitem(options, nk[].k_tol, PythonObject(fp[6]))
    py_setitem(options, nk[].k_gamma, PythonObject(fp[7]))
    py_setitem(options, nk[].k_huber_delta, PythonObject(fp[8]))
    # lane nn-optim: f7 (rmsprop's momentum, lamb's max_grad_norm) and the schedule's argument words
    py_setitem(options, nk[].k_f7, PythonObject(fp[5]))
    if len(fp) >= 9 + NN_SCHED_ARGS:
        var words = Python.list()
        for k in range(NN_SCHED_ARGS):  # small-loop(8: the schedule's argument words)
            _ = py_call(words.append, PythonObject(fp[9 + k]))
        py_setitem(options, nk[].k_sched_args, py_call(b.tuple, words))
    py_setitem(fields, nk[].k__options, options)
    var params = sequential_array(mods, Python.tuple(ip[16]), py_item(context, nk[].k_float32), context)
    # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
    var classes = Optional[PythonObject]()
    if ip[13] > 0:
        classes = sequential_array(mods, Python.tuple(ip[13]), py_item(context, nk[].k_class_dtypes)[Python.tuple(ip[14], ip[15])], context)
    var buffers: List[Int] = [Int(py=sequential_address(params)), 0, 0, 0, 0,
        0 if not classes else Int(py=sequential_address(classes.value()))]
    var m = PythonObject(None)
    var v = PythonObject(None)
    var state = PythonObject(None)
    var curve = PythonObject(None)
    if not export:
        m = sequential_array(mods, Python.tuple(ip[16]), py_item(context, nk[].k_float32), context)
        v = sequential_array(mods, Python.tuple(ip[16]), py_item(context, nk[].k_float32), context)
        state = sequential_array(mods, Python.tuple(FIT_STATE_WORDS), py_item(context, nk[].k_int64), context)
        curve = sequential_array(mods, Python.tuple(ip[12]), py_item(context, nk[].k_float32), context)
        buffers[1] = Int(py=sequential_address(m))
        buffers[2] = Int(py=sequential_address(v))
        buffers[3] = Int(py=sequential_address(state))
        buffers[4] = Int(py=sequential_address(curve))
    _ = sequential_file_load(entries, file, buffers)
    mods.released()
    py_setitem(fields, nk[].k_params_, params)
    py_setitem(fields, nk[].k_n_features_in_, PythonObject(ip[9]))
    py_setitem(fields, nk[].k_n_outputs_, PythonObject(ip[10]))
    py_setitem(fields, nk[].k__fit_loss, loss_name)
    py_setitem(fields, nk[].k__target, PythonObject(ip[1]))
    py_setitem(fields, nk[].k__fit_batch, PythonObject(ip[7]))
    py_setitem(fields, nk[].k__fit_shuffle, PythonObject(ip[8] != 0))
    py_setitem(fields, nk[].k__y_1d, PythonObject(ip[11] != 0))
    var sample = PythonObject(None)
    if len(description.sample) > 0:
        var axes = Python.list()
        for k in range(len(description.sample)):  # small-loop(at most seven sample axes)
            _ = py_call(axes.append, PythonObject(description.sample[k]))
        sample = py_call(b.tuple, axes)
    py_setitem(fields, nk[].k__sample_shape, sample)
    if Bool(classes):
        py_setitem(fields, nk[].k_classes_, classes.value())
    py_setitem(fields, nk[].k_n_iter_, PythonObject(0) if export else PythonObject(ip[12]))
    py_setitem(fields, nk[].k_loss_curve_, Python.list() if export else sequential_curve(curve, ip[12]))
    py_setitem(fields, nk[].k__fit_state, state)
    if not export:
        py_setitem(fields, nk[].k_optimizer_m_, m)
        py_setitem(fields, nk[].k_optimizer_v_, v)
        py_setitem(fields, nk[].k__curve, curve)
    return model
