# SPDX-License-Identifier: Apache-2.0
"""Native Sequential model-description and option metadata."""
# The PACK half of the former `bindings/sequential_description.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `bindings/sequential_description_unpack.mojo`.
from bindings.native_loaded_module_door import native_loaded_module, NativeModules
from std.memory import bitcast
from std.python import Python, PythonObject
from bindings.sequential_admission_door import sequential_integer, sequential_admit_value
from bindings.py_balanced import py_call, py_contains, py_fields, py_setitem, py_type
from bindings.native_identity_key_door import NativeIdentityKey
from bindings.native_prepared_owner_unpack import prepared_get, prepared_put


from bindings.sequential_description_unpack import (
    SequentialDescription,
    _integer,
    _positive,
    sequential_append,
    sequential_fields,
)
from bindings.sequential_keys_unpack import sequential_keys
from bindings.py_balanced import py_item


def _shape(mut mods: NativeModules, value: PythonObject, name: String, infer: Bool, context: PythonObject) raises -> PythonObject:
    var b = mods.module("builtins")
    var source = py_call(b.tuple, value)
    sequential_admit_value(mods, source, context)
    if len(source) < 1 or len(source) > 6:
        raise Error(name + " must list 1 to 6 dimensions")
    var out = Python.list()
    for k in range(len(source)):  # driver-loop: over at most six shape arguments
        if not _integer(source[k], context) or not (
            Bool(py=source[k] >= PythonObject(1)) or (infer and Bool(py=source[k] == PythonObject(-1)))):
            raise Error(name + " must hold positive integers" + (" or -1" if infer else ""))
        _ = py_call(out.append, py_call(b.int, source[k]))
    return out


# nn-conv3d begin
def _triple(mut mods: NativeModules, value: PythonObject, name: String, low: Int, context: PythonObject) raises -> Int:
    """A 3-D layer's window, step, padding or dilation: an integer, or a tuple or list of three (depth,
    height, width), each >= low and at most 1023, packed as autograd/nn_conv3d.mojo dhw_pack."""
    var b = mods.module("builtins")
    var d: Int
    var h: Int
    var w: Int
    if _integer(value, context):
        d = _positive(value, name, low, context)
        h = d
        w = d
    else:
        if not (Bool(py=py_call(b.isinstance, value, b.tuple)) or Bool(py=py_call(b.isinstance, value, b.list))):
            raise Error(name + " must be an integer or a tuple of three integers")
        var source = py_call(b.tuple, value)
        sequential_admit_value(mods, source, context)
        if len(source) != 3:
            raise Error(name + " must be an integer or a tuple of three integers")
        d = _positive(source[0], name, low, context)
        h = _positive(source[1], name, low, context)
        w = _positive(source[2], name, low, context)
    if d > 1023 or h > 1023 or w > 1023:
        raise Error(name + " must be at most 1023")
    return d | (h << 10) | (w << 20)
# nn-conv3d end


def _arguments(mut mods: NativeModules, kind: String, supplied: PythonObject, keywords: PythonObject,
               names: PythonObject, defaults: PythonObject, required: Int) raises -> PythonObject:
    var b = mods.module("builtins")
    if len(supplied) > len(names):
        raise Error("TypeError: " + kind + " received too many positional arguments")
    var result = py_call(b.list, defaults)
    for k in range(len(supplied)):  # driver-loop: over the layer's bounded positional options
        py_setitem(result, k, supplied[k])
    var keys = py_call(b.tuple, keywords)
    for k in range(len(keys)):  # driver-loop: over the layer's bounded keyword options
        var found = False
        for j in range(len(names)):  # driver-loop: over the layer's option names
            if Bool(py=keys[k] == names[j]):
                if j < len(supplied):
                    raise Error("TypeError: " + kind + " received an argument twice")
                py_setitem(result, j, keywords[keys[k]])
                found = True
        if not found:
            raise Error("TypeError: " + kind + " received an unknown keyword")
    for k in range(required):  # driver-loop: over required layer options
        if k >= len(supplied) and not py_contains(keywords, names[k]):
            raise Error("TypeError: " + kind + " is missing " + String(names[k]))
    return result


def sequential_add(mut mods: NativeModules, model: PythonObject, kind_arg: PythonObject, supplied: PythonObject,
                   keywords: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = sequential_keys()
    var b = mods.module("builtins")
    if not Bool(py=py_call(b.isinstance, kind_arg, b.str)):
        raise Error("kind must name a supported Sequential layer")
    var kind = String(kind_arg.lower())
    var codes = py_item(context, nk[].k_layer_codes)
    if not py_contains(codes, kind):
        raise Error("kind must name a supported Sequential layer")
    var code = codes[kind]
    var a0 = PythonObject(0)
    var a1 = PythonObject(0)
    var f = PythonObject(0.0)
    var args = PythonObject(None)
    if kind == "linear":
        var options = _arguments(mods, kind, supplied, keywords, Python.tuple("units", "bias"), Python.tuple(None, True), 1)
        if not _integer(options[0], context) or Bool(py=options[0] < PythonObject(1)):
            raise Error("units must be a positive integer")
        a0 = py_call(b.int, options[0])
        a1 = PythonObject(1 if Bool(py=options[1]) else 0)
    elif kind == "layer_norm":
        var options = _arguments(mods, kind, supplied, keywords, Python.tuple("eps"), Python.tuple(1e-5), 0)
        if not Bool(py=options[0] > PythonObject(0)):
            raise Error("eps must be positive")
        f = py_call(b.float, options[0])
    elif kind == "dropout":
        var options = _arguments(mods, kind, supplied, keywords, Python.tuple("p"), Python.tuple(None), 1)
        if not (Bool(py=options[0] >= PythonObject(0)) and Bool(py=options[0] < PythonObject(1))):
            raise Error("p must be in [0, 1)")
        f = py_call(b.float, options[0])
    elif kind == "residual":
        var options = _arguments(mods, kind, supplied, keywords, Python.tuple("source"), Python.tuple(-1), 0)
        if not _integer(options[0], context):
            raise Error("source must be -1 or the index of an earlier layer")
        a0 = py_call(b.int, options[0])
    elif kind == "reshape" or kind == "permute":
        var name = "shape" if kind == "reshape" else "axes"
        var options = _arguments(mods, kind, supplied, keywords, Python.tuple(name), Python.tuple(None), 1)
        if kind == "reshape":
            args = _shape(mods, options[0], name, True, context)
        else:
            var source = py_call(b.tuple, options[0])
            if len(source) < 1 or len(source) > 6:
                raise Error("axes must list 1 to 6 axes")
            args = Python.list()
            for k in range(len(source)):  # driver-loop: over a sample's axis permutation
                _ = py_call(args.append, _positive(source[k], "axes", 0, context))
        while len(args) < 6:  # driver-loop: over unused description argument slots
            _ = py_call(args.append, 0 if kind == "reshape" else -1)
    elif kind == "conv2d":
        var options = _arguments(mods, kind, supplied, keywords,
            Python.tuple("out_channels", "kernel_size", "stride", "padding", "bias", "groups"),
            Python.tuple(None, None, 1, 0, True, 1), 2)
        var oc = _positive(options[0], "out_channels", 1, context)
        var kernel = _positive(options[1], "kernel_size", 1, context)
        var stride = _positive(options[2], "stride", 1, context)
        var padding = _positive(options[3], "padding", 0, context)
        var groups = _positive(options[5], "groups", 1, context)
        a0 = PythonObject(oc)
        a1 = PythonObject(kernel)
        args = Python.list(oc, kernel, stride, padding, 1 if Bool(py=options[4]) else 0, groups)
    elif kind == "max_pool2d" or kind == "avg_pool2d":
        var options = _arguments(mods, kind, supplied, keywords, Python.tuple("kernel_size", "stride"), Python.tuple(None, None), 1)
        var kernel = _positive(options[0], "kernel_size", 1, context)
        var stride = kernel if options[1] is None else _positive(options[1], "stride", 1, context)
        a0 = PythonObject(kernel)
        a1 = PythonObject(stride)
    elif kind == "conv1d":
        var options = _arguments(mods, kind, supplied, keywords,
            Python.tuple("out_channels", "kernel_size", "stride", "padding", "bias", "groups"),
            Python.tuple(None, None, 1, 0, True, 1), 2)
        var oc = _positive(options[0], "out_channels", 1, context)
        var kernel = _positive(options[1], "kernel_size", 1, context)
        var stride = _positive(options[2], "stride", 1, context)
        var padding = _positive(options[3], "padding", 0, context)
        var groups = _positive(options[5], "groups", 1, context)
        a0 = PythonObject(oc)
        a1 = PythonObject(kernel)
        args = Python.list(oc, kernel, stride, padding, 1 if Bool(py=options[4]) else 0, groups)
    elif kind == "max_pool1d" or kind == "avg_pool1d":
        var options = _arguments(mods, kind, supplied, keywords, Python.tuple("kernel_size", "stride", "padding"),
                                 Python.tuple(None, None, 0), 1)
        var kernel = _positive(options[0], "kernel_size", 1, context)
        var stride = kernel if options[1] is None else _positive(options[1], "stride", 1, context)
        var padding = _positive(options[2], "padding", 0, context)
        if 2 * padding > kernel:
            raise Error("padding must be at most half of kernel_size")
        a0 = PythonObject(kernel)
        a1 = PythonObject(stride)
        args = Python.list(kernel, stride, padding, 0, 0, 0)
    elif kind == "adaptive_avg_pool1d":
        var options = _arguments(mods, kind, supplied, keywords, Python.tuple("output_size"), Python.tuple(1), 0)
        a0 = PythonObject(_positive(options[0], "output_size", 1, context))
        args = Python.list(a0, 0, 0, 0, 0, 0)
    elif kind == "branch":
        var options = _arguments(mods, kind, supplied, keywords, Python.tuple("source"), Python.tuple(-1), 0)
        if not _integer(options[0], context):
            raise Error("source must be -1 or the index of an earlier layer")
        a0 = py_call(b.int, options[0])
        args = Python.list(a0, 0, 0, 0, 0, 0)
    elif kind == "concat":
        var options = _arguments(mods, kind, supplied, keywords, Python.tuple("sources", "axis"),
                                 Python.tuple(None, 0), 1)
        var sources = py_call(b.tuple, options[0])
        f = py_call(b.float, PythonObject(_positive(options[1], "axis", 0, context)))
        if len(sources) < 1 or len(sources) > 6:
            raise Error("sources must list 1 to 6 layers")
        args = Python.list()
        for k in range(len(sources)):  # driver-loop: over at most six concat sources
            if not _integer(sources[k], context) or Bool(py=sources[k] < PythonObject(-1)):
                raise Error("sources must hold -1 or indices of earlier layers")
            _ = py_call(args.append, py_call(b.int, sources[k]))
        while len(args) < 6:  # driver-loop: over unused description argument slots
            _ = py_call(args.append, -2)
        a0 = args[0]
        a1 = args[1]
    elif kind == "silu" or kind == "softplus":
        if len(supplied) != 0 or len(keywords) != 0:
            raise Error("TypeError: " + kind + " takes no arguments")
        args = Python.list(0, 0, 0, 0, 0, 0)
    elif kind == "gelu":
        var options = _arguments(mods, kind, supplied, keywords, Python.tuple("approximate"), Python.tuple("none"), 0)
        var form = String(py_call(b.str, options[0]))
        if form != "none" and form != "tanh":
            raise Error("approximate must be 'none' or 'tanh'")
        a0 = PythonObject(1 if form == "tanh" else 0)
        args = Python.list(a0, 0, 0, 0, 0, 0)
    elif kind == "rms_norm":
        var options = _arguments(mods, kind, supplied, keywords, Python.tuple("eps"), Python.tuple(1e-6), 0)
        if not Bool(py=options[0] > PythonObject(0)):
            raise Error("eps must be positive")
        f = py_call(b.float, options[0])
        args = Python.list(0, 0, 0, 0, 0, 0)
    elif kind == "rotary":
        var options = _arguments(mods, kind, supplied, keywords, Python.tuple("head_dim", "theta"),
                                 Python.tuple(None, 10000.0), 1)
        a0 = PythonObject(_positive(options[0], "head_dim", 2, context))
        if not Bool(py=options[1] > PythonObject(0)):
            raise Error("theta must be positive")
        f = py_call(b.float, options[1])
        args = Python.list(a0, 0, 0, 0, 0, 0)
    elif kind == "multiply":
        var options = _arguments(mods, kind, supplied, keywords, Python.tuple("source"), Python.tuple(-1), 0)
        if not _integer(options[0], context):
            raise Error("source must be -1 or the index of an earlier layer")
        a0 = py_call(b.int, options[0])
        args = Python.list(a0, 0, 0, 0, 0, 0)
    elif kind == "narrow":
        var options = _arguments(mods, kind, supplied, keywords, Python.tuple("axis", "start", "length"),
                                 Python.tuple(None, None, None), 3)
        var axis = _positive(options[0], "axis", 0, context)
        var start = _positive(options[1], "start", 0, context)
        var length = _positive(options[2], "length", 1, context)
        a0 = PythonObject(axis)
        a1 = PythonObject(start)
        args = Python.list(axis, start, length, 0, 0, 0)
    elif kind == "embedding":
        var options = _arguments(mods, kind, supplied, keywords, Python.tuple("num_embeddings", "embedding_dim"),
                                 Python.tuple(None, None), 2)
        var num = _positive(options[0], "num_embeddings", 1, context)
        var dim = _positive(options[1], "embedding_dim", 1, context)
        a0 = PythonObject(num)
        a1 = PythonObject(dim)
        args = Python.list(num, dim, 0, 0, 0, 0)
    elif kind == "attention":
        var options = _arguments(mods, kind, supplied, keywords,
            Python.tuple("heads", "kv_heads", "rotary", "theta", "window", "causal"),
            Python.tuple(None, None, True, 10000.0, 0, True), 1)
        var heads = _positive(options[0], "heads", 1, context)
        var kv = heads if options[1] is None else _positive(options[1], "kv_heads", 1, context)
        var window = _positive(options[4], "window", 0, context)
        if not Bool(py=options[3] > PythonObject(0)):
            raise Error("theta must be positive")
        f = py_call(b.float, options[3])
        a0 = PythonObject(heads)
        a1 = PythonObject(kv)
        args = Python.list(heads, kv, 1 if Bool(py=options[2]) else 0, window, 1 if Bool(py=options[5]) else 0, 0)
    elif kind == "selective_scan":
        var options = _arguments(mods, kind, supplied, keywords, Python.tuple("d_state"), Python.tuple(16), 0)
        var states = _positive(options[0], "d_state", 1, context)
        if states != 4 and states != 8 and states != 16:
            raise Error("d_state must be 4, 8 or 16")
        a0 = PythonObject(states)
        args = Python.list(states, 0, 0, 0, 0, 0)
    elif kind == "ssd_scan":
        # Mamba-2 (lane neural-blocks-3): the SSD core, one argument (the heads)
        var options = _arguments(mods, kind, supplied, keywords, Python.tuple("heads"), Python.tuple(None), 1)
        var heads = _positive(options[0], "heads", 1, context)
        a0 = PythonObject(heads)
        args = Python.list(heads, 0, 0, 0, 0, 0)
    # Mamba-3 (lane neural-blocks-3-m3)
    elif kind == "trapezoid_scan":
        var options = _arguments(mods, kind, supplied, keywords, Python.tuple("heads", "d_state"),
                                 Python.tuple(None, None), 2)
        var t_heads = _positive(options[0], "heads", 1, context)
        var t_states = _positive(options[1], "d_state", 1, context)
        a0 = PythonObject(t_heads)
        a1 = PythonObject(t_states)
        args = Python.list(t_heads, t_states, 0, 0, 0, 0)
    # lm_head (lane neural-blocks-3-lm)
    elif kind == "lm_head":
        var options = _arguments(mods, kind, supplied, keywords, Python.tuple("vocab"), Python.tuple(None), 1)
        var vocab = _positive(options[0], "vocab", 2, context)
        a0 = PythonObject(vocab)
        args = Python.list(vocab, 0, 0, 0, 0, 0)
    # end lm_head
    elif kind == "rnn" or kind == "lstm" or kind == "gru":
        var options = _arguments(mods, kind, supplied, keywords, Python.tuple("hidden_size"), Python.tuple(None), 1)
        var hidden = _positive(options[0], "hidden_size", 1, context)
        var cell = 0 if kind == "rnn" else (1 if kind == "lstm" else 2)
        a0 = PythonObject(hidden)
        a1 = PythonObject(cell)
        args = Python.list(hidden, cell, 0, 0, 0, 0)
    elif kind == "batch_norm1d" or kind == "batch_norm2d":
        # lane seq-blocks-3 (autograd/model_blocks.mojo LAYER_BATCH_NORM): a1 carries the momentum's float32 bits
        var options = _arguments(mods, kind, supplied, keywords, Python.tuple("num_features", "eps", "momentum"),
                                 Python.tuple(None, 1e-5, 0.1), 1)
        var channels = _positive(options[0], "num_features", 1, context)
        if not Bool(py=options[1] > PythonObject(0)):
            raise Error("eps must be positive")
        if not (Bool(py=options[2] >= PythonObject(0)) and Bool(py=options[2] <= PythonObject(1))):
            raise Error("momentum must be in [0, 1]")
        f = py_call(b.float, options[1])
        var bits = Int(bitcast[DType.int32](Float32(Float64(py=py_call(b.float, options[2])))))
        a0 = PythonObject(channels)
        a1 = PythonObject(bits)
        args = Python.list(channels, bits, 1 if kind == "batch_norm1d" else 2, 0, 0, 0)
    # nn-conv3d begin
    elif kind == "conv3d":
        # autograd/nn_conv3d_layers.mojo LAYER_CONV3D: a4 = bias + 2 groups, the triples packed
        var options = _arguments(mods, kind, supplied, keywords,
            Python.tuple("out_channels", "kernel_size", "stride", "padding", "dilation", "groups", "bias"),
            Python.tuple(None, None, 1, 0, 1, 1, True), 2)
        var oc = _positive(options[0], "out_channels", 1, context)
        var kernel = _triple(mods, options[1], "kernel_size", 1, context)
        var stride = _triple(mods, options[2], "stride", 1, context)
        var pad_arg = options[3]
        if Bool(py=py_call(b.isinstance, pad_arg, b.str)):
            if String(py_call(b.str, pad_arg)) != "valid":
                raise Error("padding must be an integer, a tuple of three integers or 'valid'")
            pad_arg = PythonObject(0)
        var padding = _triple(mods, pad_arg, "padding", 0, context)
        var dilation = _triple(mods, options[4], "dilation", 1, context)
        var groups = _positive(options[5], "groups", 1, context)
        var with_bias = 1 if Bool(py=options[6]) else 0
        a0 = PythonObject(oc)
        a1 = PythonObject(kernel)
        args = Python.list(oc, kernel, stride, padding, with_bias + 2 * groups, dilation)
    elif kind == "max_pool3d":
        var options = _arguments(mods, kind, supplied, keywords,
            Python.tuple("kernel_size", "stride", "padding", "dilation", "ceil_mode", "return_indices"),
            Python.tuple(None, None, 0, 1, False, False), 1)
        if Bool(py=options[5]):
            raise Error("return_indices is not supported: max_pool3d returns the pooled values only")
        var kernel = _triple(mods, options[0], "kernel_size", 1, context)
        var stride = kernel if options[1] is None else _triple(mods, options[1], "stride", 1, context)
        var padding = _triple(mods, options[2], "padding", 0, context)
        var dilation = _triple(mods, options[3], "dilation", 1, context)
        a0 = PythonObject(kernel)
        a1 = PythonObject(stride)
        args = Python.list(kernel, stride, padding, dilation, 1 if Bool(py=options[4]) else 0, 0)
    elif kind == "avg_pool3d":
        var options = _arguments(mods, kind, supplied, keywords,
            Python.tuple("kernel_size", "stride", "padding", "ceil_mode", "count_include_pad", "divisor_override"),
            Python.tuple(None, None, 0, False, True, None), 1)
        var kernel = _triple(mods, options[0], "kernel_size", 1, context)
        var stride = kernel if options[1] is None else _triple(mods, options[1], "stride", 1, context)
        var padding = _triple(mods, options[2], "padding", 0, context)
        var divisor = 0 if options[5] is None else _positive(options[5], "divisor_override", 1, context)
        a0 = PythonObject(kernel)
        a1 = PythonObject(stride)
        args = Python.list(kernel, stride, padding, 1 if Bool(py=options[4]) else 0,
                           1 if Bool(py=options[3]) else 0, divisor)
    elif kind == "adaptive_avg_pool3d":
        var options = _arguments(mods, kind, supplied, keywords, Python.tuple("output_size"), Python.tuple(None), 1)
        var size = _triple(mods, options[0], "output_size", 1, context)
        a0 = PythonObject(size)
        args = Python.list(size, 0, 0, 0, 0, 0)
    # nn-conv3d end
    elif kind == "gated":
        var options = _arguments(mods, kind, supplied, keywords, Python.tuple("gate", "up"),
                                 Python.tuple(None, None), 2)
        _gated(mods, model, options[0], options[1], context)
        return model
    elif kind == "residual_block":
        var options = _arguments(mods, kind, supplied, keywords, Python.tuple("body", "shortcut"),
                                 Python.tuple(None, None), 1)
        _residual_block(mods, model, options[0], options[1], context)
        return model
    elif kind == "parallel":
        if len(keywords) != 0:
            raise Error("TypeError: parallel takes its branches as positional arguments")
        _parallel(mods, model, supplied, context)
        return model
    elif len(supplied) != 0 or len(keywords) != 0:
        raise Error("TypeError: " + kind + " takes no arguments")
    # lane neural-blocks-3 (2026-11-22): the native model reads the float argument as a Float32 (eps, p, theta,
    # axis); the description holds that value, so a model saved and loaded (the file stores float32) has the
    # same layer list and the same digest_ (before: layer_norm(1e-5) read back 9.999999747e-06, digest differed)
    f = PythonObject(Float64(Float32(Float64(py=f))))
    sequential_append(mods, model, code, a0, a1, f, args)
    return model


# ---------------------------------------------------------------- containers (lane seq-blocks, 2026-11-22)
# A container is written into the model's flat description at add time: its sub-models' layers are
# copied in, with every layer reference renumbered (an index inside a sub-model moves by the copy's
# base; -1, the sub-model's input, becomes the block's input), and joined with branch, residual and
# concat layers (autograd/model.mojo LAYER_BRANCH, LAYER_RESIDUAL, LAYER_CONCAT). Nothing of the
# sub-models runs: they are descriptions only. Codes: 8 residual, 19 branch, 20 concat.
comptime _CODE_RESIDUAL = 8
comptime _CODE_BRANCH = 19
comptime _CODE_CONCAT = 20
# lane neural-blocks: the multiply layer (autograd/model_blocks.mojo LAYER_MULTIPLY) reads an earlier output
comptime _CODE_MULTIPLY = 37


def _model_layers(mut mods: NativeModules, model: PythonObject, part: PythonObject, what: String) raises -> PythonObject:
    if part is None or py_type(part) is not py_type(model):
        raise Error("TypeError: " + what + " must be a Sequential")
    var nk = sequential_keys()
    return py_item(sequential_fields(part), nk[].k__layers)


def _remap(r: Int, base: Int, entry: Int) -> Int:
    if r == -1:
        return entry
    return r + base


def _emit(mut mods: NativeModules, model: PythonObject, sub: PythonObject, entry: Int) raises -> Int:
    """Copy the sub-model layers `sub` onto the model, reading `entry` (-1 the model input) as their
    input. Returns the index of the last copied layer, `entry` when `sub` is empty."""
    var nk = sequential_keys()
    var b = mods.module("builtins")
    var layers = py_item(sequential_fields(model), nk[].k__layers)
    var base = len(layers)
    var count = len(sub)
    for k in range(count):  # driver-loop: over a container's sub-model layers (description only)
        var layer = sub[k]
        var code = Int(py=layer[0])
        var args = py_call(b.list, layer[4])
        var a0 = layer[1]
        var a1 = layer[2]
        if code == _CODE_RESIDUAL or code == _CODE_BRANCH or code == _CODE_MULTIPLY:
            var r = _remap(Int(py=layer[1]), base, entry)
            a0 = PythonObject(r)
            py_setitem(args, 0, r)
        elif code == _CODE_CONCAT:
            for j in range(6):  # driver-loop: over the six concat source slots
                var v = Int(py=args[j])
                if v != -2:
                    py_setitem(args, j, _remap(v, base, entry))
            a0 = args[0]
            a1 = args[1]
        _ = py_call(layers.append, Python.tuple(layer[0], a0, a1, layer[3], py_call(b.tuple, args)))
    return entry if count == 0 else base + count - 1


def _append_code(mut mods: NativeModules, model: PythonObject, code: Int, args: PythonObject) raises -> Int:
    var nk = sequential_keys()
    var b = mods.module("builtins")
    var layers = py_item(sequential_fields(model), nk[].k__layers)
    var at = len(layers)
    _ = py_call(layers.append, Python.tuple(PythonObject(code), args[0], args[1], PythonObject(0.0), py_call(b.tuple, args)))
    return at


def _block_entry(mut mods: NativeModules, model: PythonObject) raises -> Int:
    var nk = sequential_keys()
    return len(py_item(sequential_fields(model), nk[].k__layers)) - 1


def _residual_block(mut mods: NativeModules, model: PythonObject, body: PythonObject, shortcut: PythonObject,
                    context: PythonObject) raises:
    var body_layers = _model_layers(mods, model, body, "body")
    if len(body_layers) == 0:
        raise Error("body must hold at least one layer")
    var shortcut_layers = PythonObject(None)
    if shortcut is not None:
        shortcut_layers = _model_layers(mods, model, shortcut, "shortcut")
    var entry = _block_entry(mods, model)
    var body_end = _emit(mods, model, body_layers, entry)
    if shortcut is None:
        _ = _append_code(mods, model, _CODE_RESIDUAL, Python.list(entry, 0, 0, 0, 0, 0))
        return
    _ = _append_code(mods, model, _CODE_BRANCH, Python.list(entry, 0, 0, 0, 0, 0))
    _ = _emit(mods, model, shortcut_layers, entry)
    _ = _append_code(mods, model, _CODE_RESIDUAL, Python.list(body_end, 0, 0, 0, 0, 0))


def _parallel(mut mods: NativeModules, model: PythonObject, branches: PythonObject, context: PythonObject) raises:
    if len(branches) < 1 or len(branches) > 6:
        raise Error("parallel takes 1 to 6 branches")
    var parts = Python.list()
    for k in range(len(branches)):  # driver-loop: over at most six branches (admission before any copy)
        _ = py_call(parts.append, _model_layers(mods, model, branches[k], "each branch"))
    var entry = _block_entry(mods, model)
    var ends = Python.list()
    for k in range(len(parts)):  # driver-loop: over at most six branches
        if len(parts[k]) == 0:
            _ = py_call(ends.append, entry)
            continue
        if k > 0:
            _ = _append_code(mods, model, _CODE_BRANCH, Python.list(entry, 0, 0, 0, 0, 0))
        _ = py_call(ends.append, _emit(mods, model, parts[k], entry))
    while len(ends) < 6:  # driver-loop: over unused concat source slots
        _ = py_call(ends.append, -2)
    _ = _append_code(mods, model, _CODE_CONCAT, ends)


def _gated(mut mods: NativeModules, model: PythonObject, gate: PythonObject, up: PythonObject,
           context: PythonObject) raises:
    """gate(h) * up(h) (lane neural-blocks, 2026-11-22): the gate's layers from the block input, a branch
    back to it, the up path's layers, then a multiply by the gate's output (a SwiGLU MLP is
    gated(Sequential().linear(F).silu(), Sequential().linear(F)) then linear(D))."""
    var gate_layers = _model_layers(mods, model, gate, "gate")
    var up_layers = _model_layers(mods, model, up, "up")
    if len(gate_layers) == 0:
        raise Error("gate must hold at least one layer")
    var entry = _block_entry(mods, model)
    var gate_end = _emit(mods, model, gate_layers, entry)
    _ = _append_code(mods, model, _CODE_BRANCH, Python.list(entry, 0, 0, 0, 0, 0))
    _ = _emit(mods, model, up_layers, entry)
    _ = _append_code(mods, model, _CODE_MULTIPLY, Python.list(gate_end, 0, 0, 0, 0, 0))


def sequential_compile(mut mods: NativeModules, model: PythonObject, options: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = sequential_keys()
    var b = mods.module("builtins")
    var loss = options[0]
    var optimizer = options[1]
    var schedule = options[9]
    if not py_contains(py_item(context, nk[].k_loss_codes), loss):
        raise Error("loss must name a supported Sequential loss")
    if not py_contains(py_item(context, nk[].k_optimizer_codes), optimizer):
        raise Error("optimizer must name a supported Sequential optimizer")
    if not py_contains(py_item(context, nk[].k_schedule_codes), schedule):
        raise Error("schedule must name a supported Sequential schedule")
    var adam = String(optimizer) == "adam" or String(optimizer) == "adamw"
    var momentum = String(optimizer) == "momentum"
    var out = Python.dict()
    py_setitem(out, nk[].k_loss, loss)
    py_setitem(out, nk[].k_opt, py_item(context, nk[].k_optimizer_codes)[optimizer])
    py_setitem(out, nk[].k_flags, 1 if momentum and Bool(py=options[4]) else 0)
    py_setitem(out, nk[].k_lr, py_call(b.float, options[2]))
    py_setitem(out, nk[].k_f1, py_call(b.float, options[5]) if adam else (py_call(b.float, options[3]) if momentum else PythonObject(0.0)))
    py_setitem(out, nk[].k_f2, py_call(b.float, options[6]) if adam else PythonObject(0.0))
    py_setitem(out, nk[].k_eps, py_call(b.float, options[7]))
    py_setitem(out, nk[].k_weight_decay, py_call(b.float, options[8]))
    py_setitem(out, nk[].k_sched, py_item(context, nk[].k_schedule_codes)[schedule])
    py_setitem(out, nk[].k_step_size, py_call(b.int, options[10]))
    py_setitem(out, nk[].k_gamma, py_call(b.float, options[11]))
    py_setitem(out, nk[].k_huber_delta, py_call(b.float, options[12]))
    py_setitem(out, nk[].k_tol, py_call(b.float, options[13]))
    py_setitem(out, nk[].k_n_iter_no_change, py_call(b.int, options[14]))
    var fields = sequential_fields(model)
    py_setitem(fields, nk[].k__options, out)
    return model


def sequential_native_description(mut mods: NativeModules, model: PythonObject,
                                  n_features: PythonObject, sample: PythonObject) raises -> PythonObject:
    var nk = sequential_keys()
    var layers = py_item(sequential_fields(model), nk[].k__layers)
    # Rank-aware programs do not consume n_features. This also preserves the
    # public description's behavior when sample is present and width is None.
    var width = 0 if sample is not None else Int(py=n_features)
    var cached = prepared_get(mods, "sequential_description_v2", model)
    if cached is not None:
        var retained = cached.downcast_value_ptr[SequentialDescription]()
        var offset = 0
        if retained[].input_width == width and retained[].identities.matches(layers, offset) and retained[].identities.complete(offset) and retained[].matches_sample(sample):
            return cached
    var description = SequentialDescription(mods, layers, width, sample)
    var stable = description.cacheable and not description.identities.retains(model)
    var result = PythonObject(alloc=description^)
    if stable:
        prepared_put(mods, "sequential_description_v2", model, result)
    return result


def sequential_program(mut mods: NativeModules, model: PythonObject, n_features: PythonObject, sample: PythonObject) raises -> PythonObject:
    var nk = sequential_keys()
    var b = mods.module("builtins")
    var layers = py_item(sequential_fields(model), nk[].k__layers)
    var result = Python.list()
    if sample is None:
        result = Python.list(1, py_call(b.int, n_features), len(layers))
        var extend = result.extend
        for k in range(len(layers)):  # driver-loop: over existing model-description layers
            var layer = layers[k]
            _ = py_call(extend, Python.tuple(layer[0], layer[1], layer[2], 0))
    else:
        result = Python.list(2, len(sample))
        var extend = result.extend
        var append = result.append
        _ = py_call(extend, sample)
        for k in range(len(sample), 8):  # driver-loop: over unused sample dimension slots
            _ = py_call(append, 1)
        _ = py_call(append, len(layers))
        for k in range(len(layers)):  # driver-loop: over existing rank-aware description layers
            var layer = layers[k]
            _ = py_call(append, layer[0])
            _ = py_call(extend, layer[4])
            _ = py_call(append, 0)
    return result


def sequential_table(model: PythonObject) raises -> PythonObject:
    var nk = sequential_keys()
    var layers = py_item(sequential_fields(model), nk[].k__layers)
    var result = Python.list()
    for k in range(len(layers)):  # driver-loop: over layer scalar options
        _ = py_call(result.append, layers[k][3])
    return result


def sequential_layers(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = sequential_keys()
    var layers = py_item(sequential_fields(model), nk[].k__layers)
    var result = Python.list()
    for k in range(len(layers)):  # driver-loop: over the model's public layer descriptions
        var layer = layers[k]
        var name = py_item(context, nk[].k_layer_names)[layer[0]]
        if Int(py=layer[0]) >= 9:
            _ = py_call(result.append, Python.tuple(name, layer[4], layer[3]))
        else:
            _ = py_call(result.append, Python.tuple(name, layer[1], layer[2], layer[3]))
    return result

def sequential_sample_shape(mut mods: NativeModules, model: PythonObject, dims: List[Int]) raises -> PythonObject:
    var nk = sequential_keys()
    var b = mods.module("builtins")
    var fields = sequential_fields(model)
    var shape = py_item(fields, nk[].k_input_shape)
    var nd = len(dims)
    var needed = shape is not None or nd > 2
    var layers = py_item(fields, nk[].k__layers)
    var count = len(layers)
    if not needed:
        for k in range(count):  # driver-loop: over shape-changing layer kinds until the schema is known
            if Int(py=layers[k][0]) >= 9:
                needed = True
                break
    if not needed:
        return PythonObject(None)
    # An explicit shape on a matrix input is returned directly. The old loop
    # constructed an unused Python sample list in this branch. A known ranked
    # schema also needs no layer-kind walk, while len(layers) above preserves
    # the existing metadata validation before any early return.
    if shape is not None and nd <= 2:
        return shape
    var actual = Python.list()
    for k in range(1, nd):  # driver-loop: over the input sample rank
        _ = py_call(actual.append, PythonObject(dims[k]))
    if shape is None:
        if nd - 1 > 7:
            raise Error("a sample has at most 7 dimensions")
        return py_call(b.tuple, actual)
    if nd > 2 and not Bool(py=py_call(b.tuple, actual) == shape):
        raise Error("X samples have a different shape from the model's input_shape")
    return shape
