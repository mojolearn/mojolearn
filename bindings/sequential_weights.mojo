# SPDX-License-Identifier: Apache-2.0
# Sequential weight admission and transfer use the existing import ABI.
# Ordinary CPU torch tensors use their native dtype/device/shape metadata and
# C data_ptr/stride entries, without __array__ or a conversion helper. Custom
# mapping/tensor subclasses can define caller protocols; those are not a
# qualification of arbitrary Python callbacks as native execution.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_contains, py_get, py_setitem, py_type
from bindings.native_loaded_module import native_loaded_module, NativeModules
from bindings.py_typed_call import py_call_typed
from providers.python.autograd_weight_frontend_pyapi import _checkpoint, _record
from autograd.program_io import ProgramInput
from bindings.sequential_description import sequential_fields, sequential_program, sequential_table
from bindings.sequential_execution import sequential_entries
from bindings.sequential_arrays import sequential_array, sequential_address, sequential_input


def sequential_parameter_layers(model: PythonObject) raises -> PythonObject:
    var layers = sequential_fields(model)["_layers"]
    var result = Python.list()
    for k in range(len(layers)):  # driver-loop: over parameter layer metadata
        var layer = layers[k]
        var code = Int(py=layer[0])
        if code == 1 or code == 6 or code == 12:
            var bias = code == 6 or (Int(py=layer[2]) == 1 if code == 1 else Int(py=layer[4][4]) == 1)
            _ = py_call(result.append, Python.tuple(k, bias))
    return result


def _exact_mapping(value: PythonObject, mapping_types: PythonObject) raises -> Bool:
    var kind = py_type(value)
    return kind is mapping_types[0] or kind is mapping_types[1]


def sequential_load_weights(mut mods: NativeModules, model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var fields = sequential_fields(model)
    if len(fields["_layers"]) == 0:
        raise Error("add at least one layer before load_state_dict")
    var source = args[0]
    var source_kind = args[1]
    var names = args[2]
    var prefix = args[3]
    # Exact-type admission: no hasattr/isinstance protocol of a caller object
    # runs (a user mapping's keys/__getitem__/__fspath__ are never consulted).
    var registry = mods.module("mojolearn._native_loaded_modules")
    var path_types = registry.PATH_TYPES
    var mapping_types = registry.MAPPING_TYPES
    if py_type(prefix) is not b.str:
        raise Error("TypeError: prefix must be a string")
    if names is not None and not _exact_mapping(names, mapping_types):
        raise Error("TypeError: names must be a dict of layer index to name")
    var layers = sequential_parameter_layers(model)
    if len(layers) == 0:
        raise Error("the model has no layers with parameters")
    var requests = Python.list()
    var keys = Python.list()
    for k in range(len(layers)):  # driver-loop: over named layer weight and bias records
        var index = layers[k][0]
        var name = index if names is None else py_get(names, index, index)
        var stem = String(prefix) + String(py_call(b.str, name))
        _ = py_call(requests.append, Python.tuple(index, 0))
        _ = py_call(keys.append, stem + ".weight")
        if Bool(py=layers[k][1]):
            _ = py_call(requests.append, Python.tuple(index, 1))
            _ = py_call(keys.append, stem + ".bias")
    var source_type = py_type(source)
    var path = (source_type is b.str or source_type is b.bytes
                or source_type is path_types[0] or source_type is path_types[1])
    var kind = ("safetensors" if path else "torch") if source_kind is None else String(source_kind)
    if kind != "torch" and kind != "safetensors":
        raise Error("source must be 'torch' or 'safetensors'")
    if (kind == "safetensors") != path:
        raise Error("TypeError: safetensors takes a path; torch takes a mapping of names to tensors")
    if not path and not _exact_mapping(source, mapping_types):
        raise Error("TypeError: source='torch' takes a mapping of names to tensors (a dict or OrderedDict)")
    var owners = Python.list()
    var values = _checkpoint(source, owners) if path else source
    var held = py_call(b.list, values.keys())
    var known = py_call(b.set, held)
    var missing = Python.list()
    for k in range(len(keys)):  # driver-loop: over required tensor names
        if not py_contains(known, keys[k]):
            _ = py_call(missing.append, keys[k])
    if len(missing) > 0:
        var message = "load_state_dict: the source has no tensor " + String(py_call(b.repr, missing[0]))
        if len(missing) > 1:
            message += " (" + String(len(missing)) + " missing)"
        message += "; it holds " + String(py_call(b.repr, held[py_call(b.slice, 0, 6)]))
        if len(held) > 6:
            message += " ..."
        raise Error(message)
    if Bool(py=args[4]):
        var requested = py_call(b.set, keys)
        var unused = Python.list()
        for k in range(len(held)):  # driver-loop: over source names checked for strict admission
            if not py_contains(requested, held[k]):
                _ = py_call(unused.append, held[k])
        if len(unused) > 0:
            raise Error("load_state_dict: the model does not use " + String(py_call(b.repr, unused[py_call(b.slice, 0, 6)]))
                        + (" ..." if len(unused) > 6 else "") + "; pass strict=False to skip them")
    var records = Python.list()
    var tensor_owners = Python.list()
    var scalar_owners = List[ProgramInput]()
    for k in range(len(requests)):  # driver-loop: over typed source tensor descriptors
        var value = values[keys[k]]
        if not path and not String(py_type(value).__module__).startswith("torch"):
            value = sequential_input(value, context)
        _ = py_call(tensor_owners.append, value)
        var record = _record(keys[k], Int(py=requests[k][0]), 0, value, path, scalar_owners)
        py_setitem(record, 1, requests[k][1])
        _ = py_call(records.extend, record)
    var sample = PythonObject(None)
    var width = 0
    if py_contains(fields, "params_") and py_contains(fields, "n_features_in_"):
        sample = py_get(fields, "_sample_shape")
        width = Int(py=fields["n_features_in_"])
    else:
        var uses_shape = fields["input_shape"] is not None
        var description = fields["_layers"]
        for k in range(len(description)):  # driver-loop: over shape-bearing layer kinds
            uses_shape = uses_shape or Int(py=description[k][0]) >= 9
        if uses_shape:
            if fields["input_shape"] is None:
                raise Error("load_state_dict: set input_shape on a model with reshape, permute, flatten, conv2d or pooling layers")
            sample = fields["input_shape"]
        else:
            var first = Int(py=layers[0][0])
            var linear = Int(py=description[first][0]) == 1
            if Int(py=records[7]) != (2 if linear else 1):
                raise Error("load_state_dict: " + String(py_call(b.repr, keys[0])) + " has the wrong shape for layer " + String(first)
                            + (" (linear): expected (out_features, in_features)" if linear else " (layer_norm): expected (width,)"))
            width = Int(py=records[9 if linear else 8])
    var program = sequential_program(mods, model, PythonObject(width), sample)
    var table = sequential_table(model)
    var entries = sequential_entries(model, context)
    var counts = py_call_typed(entries["model_n_params"], program, table)
    var params = sequential_array(mods, Python.tuple(counts[0]), context["float32"], context)
    counts = py_call_typed(entries["model_import_params"], program, table, records, keys, sequential_address(params))
    # Raw addresses remain borrowed until the native transfer has completed.
    _ = owners^
    _ = tensor_owners^
    _ = scalar_owners^
    _ = values^
    py_setitem(fields, "params_", params)
    py_setitem(fields, "n_features_in_", counts[2])
    py_setitem(fields, "n_outputs_", counts[1])
    py_setitem(fields, "_sample_shape", sample)
    py_setitem(fields, "n_iter_", 0)
    py_setitem(fields, "loss_curve_", Python.list())
    py_setitem(fields, "_fit_state", PythonObject(None))
    py_setitem(fields, "_imported", True)
    return model
