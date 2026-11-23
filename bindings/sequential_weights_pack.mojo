# SPDX-License-Identifier: Apache-2.0
# Sequential weight admission and transfer use the existing import ABI.
# Ordinary CPU torch tensors use their native dtype/device/shape metadata and
# C data_ptr/stride entries, without __array__ or a conversion helper. Custom
# mapping/tensor subclasses can define caller protocols; those are not a
# qualification of arbitrary Python callbacks as native execution.
# The PACK half of the former `bindings/sequential_weights.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `bindings/sequential_weights_unpack.mojo`.
from autograd.model_blocks import block_param_names
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_contains, py_get, py_setitem, py_type
from bindings.native_loaded_module_door import native_loaded_module, NativeModules
from providers.python.autograd_weight_frontend_pyapi_pack import _checkpoint, _record
from autograd.program_io import ProgramInput
from bindings.sequential_description_unpack import sequential_fields, SequentialDescription
from bindings.sequential_description_pack import sequential_native_description
from bindings.sequential_model_file_door import sequential_file_counts, sequential_file_import, sequential_file_export
from providers.python.sequence_pyapi_door import py_ints
from bindings.sequential_execution_pack import sequential_entries
from bindings.sequential_arrays_door import sequential_array, sequential_address, sequential_input
from bindings.py_model_records import bump_model_generation


from bindings.sequential_weights_unpack import (
    _exact_mapping,
)
from bindings.sequential_keys_unpack import sequential_keys
from bindings.py_balanced import py_item


def sequential_parameter_layers(model: PythonObject) raises -> PythonObject:
    var nk = sequential_keys()
    var layers = py_item(sequential_fields(model), nk[].k__layers)
    var result = Python.list()
    for k in range(len(layers)):  # driver-loop: over parameter layer metadata
        var layer = layers[k]
        var code = Int(py=layer[0])
        if code == 1 or code == 6 or code == 12 or code == 15:
            var bias = code == 6 or (Int(py=layer[2]) == 1 if code == 1 else Int(py=layer[4][4]) == 1)
            _ = py_call(result.append, Python.tuple(k, bias))
        elif code == 35 or code == 39 or code == 40 or code == 41:
            # lane neural-blocks: a block layer's tensors by name (autograd/model_blocks.mojo block_param_names)
            var suffixes = block_param_names(code)
            var names = Python.list()
            for j in range(len(suffixes)):  # small-loop(at most four tensors of a block layer)
                _ = py_call(names.append, PythonObject(suffixes[j]))
            _ = py_call(result.append, Python.tuple(k, False, py_call(native_loaded_module("builtins").tuple, names)))
        # nn-attention begin: multihead_attention's four tensors by name (autograd/nn_attention.mojo)
        elif code == 67:
            var nn_suffixes = block_param_names(code)
            var nn_names = Python.list()
            for j in range(len(nn_suffixes)):  # small-loop(four tensors of a multihead_attention layer)
                _ = py_call(nn_names.append, PythonObject(nn_suffixes[j]))
            _ = py_call(result.append, Python.tuple(k, False, py_call(native_loaded_module("builtins").tuple, nn_names)))
        # nn-attention end
    return result


def sequential_load_weights(mut mods: NativeModules, model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = sequential_keys()
    # Generation word (bindings/py_model_records.mojo, lane ne1-e-records):
    # every stamped record of this object misses from here on.
    _ = bump_model_generation(model)
    var b = native_loaded_module("builtins")
    var fields = sequential_fields(model)
    if len(py_item(fields, nk[].k__layers)) == 0:
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
        if len(layers[k]) == 3:
            for j in range(len(layers[k][2])):  # small-loop(at most four tensors of a block layer)
                _ = py_call(requests.append, Python.tuple(index, j))
                _ = py_call(keys.append, stem + String(layers[k][2][j]))
            continue
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
    mods.released()
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
        mods.released()
        py_setitem(record, 1, requests[k][1])
        _ = py_call(records.extend, record)
    var sample = PythonObject(None)
    var width = 0
    if py_contains(fields, nk[].k_params_) and py_contains(fields, nk[].k_n_features_in_):
        sample = py_get(fields, nk[].k__sample_shape)
        width = Int(py=py_item(fields, nk[].k_n_features_in_))
    else:
        var uses_shape = py_item(fields, nk[].k_input_shape) is not None
        var description = py_item(fields, nk[].k__layers)
        for k in range(len(description)):  # driver-loop: over shape-bearing layer kinds
            uses_shape = uses_shape or Int(py=description[k][0]) >= 9
        if uses_shape:
            if py_item(fields, nk[].k_input_shape) is None:
                raise Error("load_state_dict: set input_shape on a model with reshape, permute, flatten, convolution, pooling, branch or concat layers")
            sample = py_item(fields, nk[].k_input_shape)
        else:
            var first = Int(py=layers[0][0])
            var linear = Int(py=description[first][0]) == 1
            if Int(py=records[7]) != (2 if linear else 1):
                raise Error("load_state_dict: " + String(py_call(b.repr, keys[0])) + " has the wrong shape for layer " + String(first)
                            + (" (linear): expected (out_features, in_features)" if linear else " (layer_norm): expected (width,)"))
            width = Int(py=records[9 if linear else 8])
    # The model description as fit and predict hand it to the same engine
    # (the retained native program words and one scalar per layer), and the
    # source records and names typed once (UNPACK). The count and the transfer
    # are native descriptor calls with no Python object.
    var description_owner = sequential_native_description(mods, model, PythonObject(width), sample)
    var description = description_owner.downcast_value_ptr[SequentialDescription]()
    var program = description[].program.copy()
    var table = description[].scalars.copy()
    _ = description_owner^
    var record_words = py_ints(records)
    var names_native = List[String]()
    for k in range(len(keys)):  # driver-loop: over source tensor names
        names_native.append(String(py=keys[k]))
    var entries = sequential_entries(model, context)
    mods.released()
    var sizes = sequential_file_counts(entries, program, table)
    mods.released()
    var params = sequential_array(mods, Python.tuple(sizes[0]), py_item(context, nk[].k_float32), context)
    var counts = sequential_file_import(entries, program, table, record_words, names_native, Int(py=sequential_address(params)))
    mods.released()
    # Raw addresses remain borrowed until the native transfer has completed.
    _ = owners^
    _ = tensor_owners^
    _ = scalar_owners^
    _ = values^
    py_setitem(fields, nk[].k_params_, params)
    py_setitem(fields, nk[].k_n_features_in_, PythonObject(counts[2]))
    py_setitem(fields, nk[].k_n_outputs_, PythonObject(counts[1]))
    py_setitem(fields, nk[].k__sample_shape, sample)
    py_setitem(fields, nk[].k_n_iter_, 0)
    py_setitem(fields, nk[].k_loss_curve_, Python.list())
    py_setitem(fields, nk[].k__fit_state, PythonObject(None))
    py_setitem(fields, nk[].k__imported, True)
    return model


# Sequential.state_dict / parameters / named_parameters (lane api-torch round
# 2, 2026-11-21): one native transfer of params_ into PyTorch's layouts
# (MODEL_EXPORT_PARAMS, autograd/weight_import.mojo model_export_params), then
# PACK builds the result over that one buffer: per tensor a view of it shaped
# as torch shapes it, named "<layer index>.weight" / ".bias" as
# `nn.Sequential` names it, so `load_state_dict(model.state_dict())` gives
# the same params_ back.
comptime SEQUENTIAL_EXPORT_STATE_DICT = 0
comptime SEQUENTIAL_EXPORT_PARAMETERS = 1
comptime SEQUENTIAL_EXPORT_NAMED_PARAMETERS = 2
comptime SEQUENTIAL_EXPORT_SHAPE_WORDS = 8


def sequential_export_weights(mut mods: NativeModules, model: PythonObject, form: Int, context: PythonObject) raises -> PythonObject:
    var nk = sequential_keys()
    var b = native_loaded_module("builtins")
    var fields = sequential_fields(model)
    if not py_contains(fields, nk[].k_params_) or not py_contains(fields, nk[].k_n_features_in_):
        raise Error("state_dict: the model has no parameters yet; fit it or load_state_dict first")
    var params = py_item(fields, nk[].k_params_)
    var description_owner = sequential_native_description(
        mods, model, PythonObject(Int(py=py_item(fields, nk[].k_n_features_in_))), py_get(fields, nk[].k__sample_shape))
    var description = description_owner.downcast_value_ptr[SequentialDescription]()
    var program = description[].program.copy()
    var table = description[].scalars.copy()
    _ = description_owner^
    var entries = sequential_entries(model, context)
    mods.released()
    var n_params = len(params)
    var exported = sequential_array(mods, Python.tuple(n_params), py_item(context, nk[].k_float32), context)
    var model_layers = py_item(fields, nk[].k__layers)
    var shapes = List[Int](length=4 * len(model_layers) * SEQUENTIAL_EXPORT_SHAPE_WORDS, fill=0)
    var count = sequential_file_export(entries, program, table, Int(py=sequential_address(params)),
                                       Int(py=sequential_address(exported)), shapes)
    mods.released()
    _ = params^
    var named = Python.dict()
    var arrays = Python.list()
    var pairs = Python.list()
    for t in range(count):  # driver-loop: over the model's parameter tensors
        var base = t * SEQUENTIAL_EXPORT_SHAPE_WORDS
        var off = shapes[base + 2]
        var rank = shapes[base + 3]
        var size = 1
        var dims = Python.list()
        for k in range(rank):  # small-loop(rank: at most 4): shape metadata, no data
            size *= shapes[base + 4 + k]
            _ = py_call(dims.append, shapes[base + 4 + k])
        var window = exported[py_call(b.slice, off, off + size)]
        var value = py_call(window.reshape, py_call(b.tuple, dims))
        var name = String(shapes[base]) + (".bias" if shapes[base + 1] == 1 else ".weight")
        var code = Int(py=model_layers[shapes[base]][0])
        if code >= 32:
            name = String(shapes[base]) + block_param_names(code)[shapes[base + 1]]
        if form == SEQUENTIAL_EXPORT_STATE_DICT:
            py_setitem(named, PythonObject(name), value)
        elif form == SEQUENTIAL_EXPORT_PARAMETERS:
            _ = py_call(arrays.append, value)
        else:
            _ = py_call(pairs.append, Python.tuple(PythonObject(name), value))
    if form == SEQUENTIAL_EXPORT_STATE_DICT:
        return named
    if form == SEQUENTIAL_EXPORT_PARAMETERS:
        return py_call(b.iter, arrays)
    return py_call(b.iter, pairs)
