# SPDX-License-Identifier: Apache-2.0
# Sequential weight admission and transfer use the existing import ABI.
# Source mappings and CPU torch tensor protocols remain caller-owned APIs;
# ordinary dict/ndarray and safetensors paths need no Python product helpers.
from std.python import Python, PythonObject
from autograd.weight_frontend import _checkpoint, _record
from autograd.program_pyapi import ProgramInput
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
            result.append(Python.tuple(k, bias))
    return result


def sequential_load_weights(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var fields = sequential_fields(model)
    if len(fields["_layers"]) == 0:
        raise Error("add at least one layer before load_state_dict")
    var source = args[0]
    var source_kind = args[1]
    var names = args[2]
    var prefix = args[3]
    if not Bool(py=b.isinstance(prefix, b.str)):
        raise Error("TypeError: prefix must be a string")
    if names is not None and not Bool(py=b.hasattr(names, "get")):
        raise Error("TypeError: names must be a dict of layer index to name")
    var layers = sequential_parameter_layers(model)
    if len(layers) == 0:
        raise Error("the model has no layers with parameters")
    var requests = Python.list()
    var keys = Python.list()
    for k in range(len(layers)):  # driver-loop: over named layer weight and bias records
        var index = layers[k][0]
        var name = index if names is None else names.get(index, index)
        var stem = String(prefix) + String(b.str(name))
        requests.append(Python.tuple(index, 0))
        keys.append(stem + ".weight")
        if Bool(py=layers[k][1]):
            requests.append(Python.tuple(index, 1))
            keys.append(stem + ".bias")
    var path = Bool(py=b.isinstance(source, b.str)) or Bool(py=b.isinstance(source, b.bytes)) or Bool(py=b.hasattr(source, "__fspath__"))
    var kind = ("safetensors" if path else "torch") if source_kind is None else String(source_kind)
    if kind != "torch" and kind != "safetensors":
        raise Error("source must be 'torch' or 'safetensors'")
    if (kind == "safetensors") != path:
        raise Error("TypeError: safetensors takes a path; torch takes a mapping of names to tensors")
    if not path and (not Bool(py=b.hasattr(source, "keys")) or not Bool(py=b.hasattr(source, "__getitem__"))):
        raise Error("TypeError: source='torch' takes a mapping of names to tensors")
    var owners = Python.list()
    var values = _checkpoint(source, owners) if path else source
    var held = b.list(values.keys())
    var known = b.set(held)
    var missing = Python.list()
    for k in range(len(keys)):  # driver-loop: over required tensor names
        if not Bool(py=known.__contains__(keys[k])):
            missing.append(keys[k])
    if len(missing) > 0:
        var message = "load_state_dict: the source has no tensor " + String(b.repr(missing[0]))
        if len(missing) > 1:
            message += " (" + String(len(missing)) + " missing)"
        message += "; it holds " + String(b.repr(held[b.slice(0, 6)]))
        if len(held) > 6:
            message += " ..."
        raise Error(message)
    if Bool(py=args[4]):
        var requested = b.set(keys)
        var unused = Python.list()
        for k in range(len(held)):  # driver-loop: over source names checked for strict admission
            if not Bool(py=requested.__contains__(held[k])):
                unused.append(held[k])
        if len(unused) > 0:
            raise Error("load_state_dict: the model does not use " + String(b.repr(unused[b.slice(0, 6)]))
                        + (" ..." if len(unused) > 6 else "") + "; pass strict=False to skip them")
    var records = Python.list()
    var tensor_owners = Python.list()
    var scalar_owners = List[ProgramInput]()
    for k in range(len(requests)):  # driver-loop: over typed source tensor descriptors
        var value = values[keys[k]]
        if not path and not String(b.type(value).__module__).startswith("torch"):
            value = sequential_input(value, context)
        tensor_owners.append(value)
        var record = _record(keys[k], Int(py=requests[k][0]), 0, value, path, scalar_owners)
        record[1] = requests[k][1]
        records.extend(record)
    var sample = PythonObject(None)
    var width = 0
    if Bool(py=fields.__contains__("params_")) and Bool(py=fields.__contains__("n_features_in_")):
        sample = fields.get("_sample_shape")
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
                raise Error("load_state_dict: " + String(b.repr(keys[0])) + " has the wrong shape for layer " + String(first)
                            + (" (linear): expected (out_features, in_features)" if linear else " (layer_norm): expected (width,)"))
            width = Int(py=records[9 if linear else 8])
    var program = sequential_program(model, PythonObject(width), sample)
    var table = sequential_table(model)
    var entries = sequential_entries(model, context)
    var counts = entries["model_n_params"](program, table)
    var params = sequential_array(Python.tuple(counts[0]), context["float32"], context)
    counts = entries["model_import_params"](program, table, records, keys, sequential_address(params))
    # Raw addresses remain borrowed until the native transfer has completed.
    _ = owners^
    _ = tensor_owners^
    _ = scalar_owners^
    _ = values^
    fields["params_"] = params
    fields["n_features_in_"] = counts[2]
    fields["n_outputs_"] = counts[1]
    fields["_sample_shape"] = sample
    fields["n_iter_"] = 0
    fields["loss_curve_"] = Python.list()
    fields["_fit_state"] = PythonObject(None)
    fields["_imported"] = True
    return model
