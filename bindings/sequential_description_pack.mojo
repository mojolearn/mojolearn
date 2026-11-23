# SPDX-License-Identifier: Apache-2.0
"""Native Sequential model-description and option metadata."""
# The PACK half of the former `bindings/sequential_description.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `bindings/sequential_description_unpack.mojo`.
from bindings.native_loaded_module_door import native_loaded_module, NativeModules
from std.memory import bitcast
from std.math import isfinite  # nn-activations
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
        # nn-recurrent begin: PyTorch's num_layers, dropout, bidirectional, and the output (every step's state or
        # the final state); the defaults write the one layer they always wrote (the same description words)
        var options = _arguments(mods, kind, supplied, keywords,
            Python.tuple("hidden_size", "num_layers", "bidirectional", "dropout", "output"),
            Python.tuple(None, 1, False, 0.0, "sequence"), 1)
        var hidden = _positive(options[0], "hidden_size", 1, context)
        var cell = 0 if kind == "rnn" else (1 if kind == "lstm" else 2)
        var stack = _positive(options[1], "num_layers", 1, context)
        if not Bool(py=py_call(b.isinstance, options[2], b.bool)):
            raise Error("bidirectional must be True or False")
        var both = Bool(py=options[2])
        if not (Bool(py=options[3] >= PythonObject(0)) and Bool(py=options[3] < PythonObject(1))):
            raise Error("dropout must be in [0, 1)")
        var form = String(py_call(b.str, options[4])) if Bool(py=py_call(b.isinstance, options[4], b.str)) else String("")
        if form != "sequence" and form != "final":
            raise Error("output must be 'sequence' or 'final'")
        var final = form == "final"
        if stack > 1 or both or final:
            _recurrent_stack(mods, model, code, hidden, cell, stack, both, options[3], final)
            return model
        # nn-recurrent end
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
    # nn-activations begin: leaky_relu (f negative_slope), elu (f alpha), prelu (a0 num_parameters, a1 the init's
    # float32 bits); identity, mish, hardswish and hardsigmoid take no arguments (the last branch below)
    elif kind == "leaky_relu" or kind == "elu":
        var label = "negative_slope" if kind == "leaky_relu" else "alpha"
        var options = _arguments(mods, kind, supplied, keywords, Python.tuple(label),
                                 Python.tuple(0.01 if kind == "leaky_relu" else 1.0), 0)
        var value = Float64(py=py_call(b.float, options[0]))
        if not isfinite(value) or not isfinite(Float32(value)):
            raise Error(label + " must be a finite number")
        f = PythonObject(value)
        args = Python.list(0, 0, 0, 0, 0, 0)
    elif kind == "prelu":
        var options = _arguments(mods, kind, supplied, keywords, Python.tuple("num_parameters", "init"),
                                 Python.tuple(1, 0.25), 0)
        var count = _positive(options[0], "num_parameters", 1, context)
        var init = Float32(Float64(py=py_call(b.float, options[1])))
        if not isfinite(init) or not (init > Float32(-1)):
            raise Error("init must be a finite number greater than -1")
        var init_bits = Int(bitcast[DType.int32](init))
        a0 = PythonObject(count)
        a1 = PythonObject(init_bits)
        args = Python.list(count, init_bits, 0, 0, 0, 0)
    # nn-activations end
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
    # nn-attention begin
    elif kind == "multihead_attention" or kind == "positional_encoding":
        var layer = _nn_attention_layer(mods, kind, supplied, keywords, context)
        a0 = layer[0]
        a1 = layer[1]
        f = layer[2]
        args = layer[3]
    elif (kind == "transformer_encoder_layer" or kind == "transformer_decoder_layer"
          or kind == "transformer_encoder" or kind == "transformer_decoder"):
        _nn_attention_container(mods, model, kind, supplied, keywords, context)
        return model
    # nn-attention end
    # nn-conv begin
    # lane nn-conv (2026-11-23): layer codes 56..66 (autograd/nn_conv_blocks.mojo has the argument words)
    elif kind == "conv_transpose1d" or kind == "conv_transpose2d":
        var options = _arguments(mods, kind, supplied, keywords,
            Python.tuple("out_channels", "kernel_size", "stride", "padding", "output_padding", "groups", "bias",
                         "dilation"),
            Python.tuple(None, None, 1, 0, 0, 1, True, 1), 2)
        var oc = _positive(options[0], "out_channels", 1, context)
        var kernel = _positive(options[1], "kernel_size", 1, context)
        var stride = _positive(options[2], "stride", 1, context)
        var padding = _positive(options[3], "padding", 0, context)
        var out_pad = _positive(options[4], "output_padding", 0, context)
        var groups = _positive(options[5], "groups", 1, context)
        var dilation = _positive(options[7], "dilation", 1, context)
        if out_pad >= stride:
            raise Error("output_padding must be smaller than stride")
        if oc % groups != 0:
            raise Error("groups must divide out_channels")
        a0 = PythonObject(oc)
        a1 = PythonObject(kernel)
        f = PythonObject(Float64(dilation))
        args = Python.list(oc, kernel, stride, padding, out_pad, 2 * groups + (1 if Bool(py=options[6]) else 0))
    elif kind == "adaptive_avg_pool2d" or kind == "adaptive_max_pool2d":
        var options = _arguments(mods, kind, supplied, keywords, Python.tuple("output_size"), Python.tuple(None), 1)
        var hw = _nnc_pair(mods, options[0], "output_size", True, context)
        a0 = PythonObject(hw[0])
        a1 = PythonObject(hw[1])
        args = Python.list(hw[0], hw[1], 0, 0, 0, 0)
    elif kind == "adaptive_max_pool1d":
        var options = _arguments(mods, kind, supplied, keywords, Python.tuple("output_size"), Python.tuple(None), 1)
        a0 = PythonObject(_positive(options[0], "output_size", 1, context))
        args = Python.list(a0, 0, 0, 0, 0, 0)
    elif kind == "global_max_pool":
        if len(supplied) != 0 or len(keywords) != 0:
            raise Error("TypeError: global_max_pool takes no arguments")
        args = Python.list(0, 0, 0, 0, 0, 0)
    elif (kind == "zero_pad1d" or kind == "zero_pad2d" or kind == "constant_pad1d" or kind == "constant_pad2d"
          or kind == "reflection_pad1d" or kind == "reflection_pad2d" or kind == "replication_pad1d"
          or kind == "replication_pad2d"):
        var constant = kind.startswith("constant")
        var value_default = PythonObject(None) if constant else PythonObject(0.0)
        var options = _arguments(mods, kind, supplied, keywords, Python.tuple("padding", "value"),
                                 Python.tuple(None, value_default), 2 if constant else 1)
        if not constant and (len(supplied) > 1 or py_contains(keywords, "value")):
            raise Error("TypeError: " + kind + " takes one argument, padding")
        var two = kind.endswith("2d")
        var pads = _nnc_pads(mods, options[0], two, kind.startswith("zero") or constant, context)
        if constant:
            var vb = mods.module("builtins")
            if not Bool(py=py_call(vb.isinstance, options[1], Python.tuple(vb.int, vb.float))) or Bool(
                    py=py_call(vb.isinstance, options[1], vb.bool)):
                raise Error("value must be a number")
            f = py_call(vb.float, options[1])
        a0 = PythonObject(pads[0])
        a1 = PythonObject(pads[1])
        args = Python.list(pads[0], pads[1], pads[2], pads[3], 2 if two else 1, 0)
    elif kind == "upsample":
        var options = _arguments(mods, kind, supplied, keywords,
                                 Python.tuple("size", "scale_factor", "mode", "align_corners"),
                                 Python.tuple(None, None, "nearest", None), 0)
        if (options[0] is None) == (options[1] is None):
            raise Error("upsample takes either size or scale_factor")
        var mode_name = String(py_call(b.str, options[2]))
        var mode = 0 if mode_name == "nearest" else (1 if mode_name == "linear" else (2 if mode_name == "bilinear" else -1))
        if mode < 0:
            raise Error("mode must be 'nearest', 'linear' or 'bilinear'")
        var align = 0
        if options[3] is not None:
            if mode == 0:
                raise Error("align_corners can only be set with mode 'linear' or 'bilinear'")
            align = 1 if Bool(py=options[3]) else 0
        var oh = 0
        var ow = 0
        if options[0] is not None:
            var hw = _nnc_pair(mods, options[0], "size", False, context)
            oh = hw[0]
            ow = hw[1]
        else:
            var sb = mods.module("builtins")
            if not Bool(py=py_call(sb.isinstance, options[1], Python.tuple(sb.int, sb.float))) or Bool(
                    py=py_call(sb.isinstance, options[1], sb.bool)) or not Bool(py=options[1] > PythonObject(0)):
                raise Error("scale_factor must be a positive number")
            f = py_call(sb.float, options[1])
        a0 = PythonObject(oh)
        a1 = PythonObject(ow)
        args = Python.list(oh, ow, mode, align, 0, 0)
    elif kind == "pixel_shuffle" or kind == "pixel_unshuffle":
        var name = "upscale_factor" if kind == "pixel_shuffle" else "downscale_factor"
        var options = _arguments(mods, kind, supplied, keywords, Python.tuple(name), Python.tuple(None), 1)
        a0 = PythonObject(_positive(options[0], name, 1, context))
        a1 = PythonObject(0 if kind == "pixel_shuffle" else 1)
        args = Python.list(a0, a1, 0, 0, 0, 0)
    # nn-conv end
    # nn-norms begin (autograd/model_blocks.mojo LAYER_GROUP_NORM, LAYER_INSTANCE_NORM; PyTorch's arguments and defaults)
    elif kind == "group_norm":
        var options = _arguments(mods, kind, supplied, keywords, Python.tuple("num_groups", "num_channels", "eps", "affine"),
                                 Python.tuple(None, None, 1e-5, True), 2)
        var groups = _positive(options[0], "num_groups", 1, context)
        var gchannels = _positive(options[1], "num_channels", 1, context)
        if gchannels % groups != 0:
            raise Error("num_channels must be divisible by num_groups")
        if not Bool(py=options[2] > PythonObject(0)):
            raise Error("eps must be positive")
        f = py_call(b.float, options[2])
        a0 = PythonObject(groups)
        a1 = PythonObject(gchannels)
        args = Python.list(groups, gchannels, 1 if Bool(py=options[3]) else 0, 0, 0, 0)
    elif kind == "instance_norm1d" or kind == "instance_norm2d":
        var options = _arguments(mods, kind, supplied, keywords,
                                 Python.tuple("num_features", "eps", "momentum", "affine", "track_running_stats"),
                                 Python.tuple(None, 1e-5, 0.1, False, False), 1)
        var ichannels = _positive(options[0], "num_features", 1, context)
        if not Bool(py=options[1] > PythonObject(0)):
            raise Error("eps must be positive")
        if options[2] is None:
            raise Error("momentum=None (a cumulative average) is not supported; give a momentum in [0, 1]")
        if not (Bool(py=options[2] >= PythonObject(0)) and Bool(py=options[2] <= PythonObject(1))):
            raise Error("momentum must be in [0, 1]")
        f = py_call(b.float, options[1])
        var ibits = Int(bitcast[DType.int32](Float32(Float64(py=py_call(b.float, options[2])))))
        a0 = PythonObject(ichannels)
        a1 = PythonObject(ibits)
        args = Python.list(ichannels, ibits, 1 if kind == "instance_norm1d" else 2, 1 if Bool(py=options[3]) else 0,
                           1 if Bool(py=options[4]) else 0, 0)
    # nn-norms end
    elif len(supplied) != 0 or len(keywords) != 0:
        raise Error("TypeError: " + kind + " takes no arguments")
    # lane neural-blocks-3 (2026-11-22): the native model reads the float argument as a Float32 (eps, p, theta,
    # axis); the description holds that value, so a model saved and loaded (the file stores float32) has the
    # same layer list and the same digest_ (before: layer_norm(1e-5) read back 9.999999747e-06, digest differed)
    f = PythonObject(Float64(Float32(Float64(py=f))))
    sequential_append(mods, model, code, a0, a1, f, args)
    return model


# nn-conv begin
def _nnc_pair(mut mods: NativeModules, value: PythonObject, name: String, keep: Bool,
              context: PythonObject) raises -> Tuple[Int, Int]:
    """(height, width) of an int or a pair (a 1-length tuple: the length); with `keep`, None entries are 0
    (the input's size is kept)."""
    var b = mods.module("builtins")
    if Bool(py=py_call(b.isinstance, value, Python.tuple(b.tuple, b.list))):
        var items = py_call(b.tuple, value)
        if len(items) == 1:
            return (0, _nnc_size(items[0], name, keep, context))
        if len(items) != 2:
            raise Error(name + " must be an integer or a pair of integers")
        return (_nnc_size(items[0], name, keep, context), _nnc_size(items[1], name, keep, context))
    var v = _nnc_size(value, name, keep, context)
    return (v, v)


def _nnc_size(value: PythonObject, name: String, keep: Bool, context: PythonObject) raises -> Int:
    if keep and value is None:
        return 0
    return _positive(value, name, 1, context)


def _nnc_pads(mut mods: NativeModules, value: PythonObject, two: Bool, negative: Bool,
              context: PythonObject) raises -> List[Int]:
    """[left, right, top, bottom] of a padding argument: an int (every side) or PyTorch's tuple
    (left, right) / (left, right, top, bottom). Negative pads crop (constant padding only)."""
    var b = mods.module("builtins")
    var out = List[Int](length=4, fill=0)
    var n = 4 if two else 2
    if Bool(py=py_call(b.isinstance, value, Python.tuple(b.tuple, b.list))):
        var items = py_call(b.tuple, value)
        if len(items) != n:
            raise Error("padding must be an integer or a tuple of " + String(n) + " integers")
        for k in range(n):  # driver-loop: over at most four pad sides
            out[k] = _nnc_pad(items[k], negative, context)
    else:
        var v = _nnc_pad(value, negative, context)
        for k in range(n):  # driver-loop: over at most four pad sides
            out[k] = v
    return out^


def _nnc_pad(value: PythonObject, negative: Bool, context: PythonObject) raises -> Int:
    if not _integer(value, context):
        raise Error("padding must hold integers")
    var v = Int(py=value)
    if v < 0 and not negative:
        raise Error("padding must be non-negative")
    return v
# nn-conv end


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
        # nn-attention begin: multihead_attention's key/value source (a2; -2 self and -3 an unset memory stay)
        elif code == _NN_CODE_MHA:
            var src = Int(py=args[2])
            if src >= -1:
                py_setitem(args, 2, _remap(src, base, entry))
        # nn-attention end
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


# nn-recurrent begin
comptime _CODE_DROPOUT = 7


def _append_layer(mut mods: NativeModules, model: PythonObject, code: PythonObject, args: PythonObject,
                  f: Float64) raises -> Int:
    """Append one description layer (code, args[0], args[1], f as its float32 value, args); its index."""
    var nk = sequential_keys()
    var b = mods.module("builtins")
    var layers = py_item(sequential_fields(model), nk[].k__layers)
    var at = len(layers)
    _ = py_call(layers.append, Python.tuple(code, args[0], args[1], PythonObject(Float64(Float32(f))),
                                            py_call(b.tuple, args)))
    return at


def _recurrent_stack(mut mods: NativeModules, model: PythonObject, code: PythonObject, hidden: Int, cell: Int,
                     stack: Int, both: Bool, dropout: PythonObject, final: Bool) raises:
    """PyTorch's multi-layer, bidirectional recurrent module as description layers (lane nn-recurrent): per layer
    a forward recurrent layer and, when bidirectional, a branch back to the layer's input, a reverse recurrent
    layer (a2 = 1) and a concat of the two on the features; a dropout layer after every layer but the last when
    dropout > 0. With output 'final' the last layer's recurrent layers output their final state (a3 = 1) and
    the concat joins them on the first axis: PyTorch's h_n of that layer, (directions, hidden)."""
    var b = mods.module("builtins")
    var p = Float64(py=py_call(b.float, dropout))
    var entry = _block_entry(mods, model)
    for l in range(stack):  # driver-loop: over the stacked layers (description only)
        var top = l == stack - 1
        var fin = 1 if (final and top) else 0
        var fwd = _append_layer(mods, model, code, Python.list(hidden, cell, 0, fin, 0, 0), 0.0)
        var last = fwd
        if both:
            _ = _append_code(mods, model, _CODE_BRANCH, Python.list(entry, 0, 0, 0, 0, 0))
            var rev = _append_layer(mods, model, code, Python.list(hidden, cell, 1, fin, 0, 0), 0.0)
            last = _append_layer(mods, model, PythonObject(_CODE_CONCAT), Python.list(fwd, rev, -2, -2, -2, -2),
                                 0.0 if fin == 1 else 1.0)
        if not top and p > 0.0:
            last = _append_layer(mods, model, PythonObject(_CODE_DROPOUT), Python.list(0, 0, 0, 0, 0, 0), p)
        entry = last
# nn-recurrent end


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


# nn-attention begin ---------------------------------------------------------------------------------------
# Classic attention (lane nn-attention, 2026-11-23): multihead_attention (67) and positional_encoding (72) are
# layers (autograd/nn_attention.mojo); the encoder and decoder layers and stacks are containers written here
# into the flat description from existing layers (PyTorch's TransformerEncoderLayer / TransformerDecoderLayer
# and TransformerEncoder / TransformerDecoder module for module): nothing of them runs here.
comptime _NN_CODE_MHA = 67
comptime _NN_CODE_POS_ENC = 72
comptime _NN_CODE_LINEAR = 1
comptime _NN_CODE_RELU = 2
comptime _NN_CODE_LAYER_NORM = 6
comptime _NN_CODE_DROPOUT = 7
comptime _NN_CODE_GELU = 33
comptime _NN_SELF = -2
comptime _NN_MEMORY_UNSET = -3


def _nn_float(mut mods: NativeModules, value: PythonObject, name: String) raises -> PythonObject:
    var b = mods.module("builtins")
    if value is None or py_type(value) is b.bool or py_type(value) is b.str:
        raise Error("TypeError: " + name + " must be a number")
    # one rounding to float32, as sequential_add's (the native model reads a Float32)
    return PythonObject(Float64(Float32(Float64(py=py_call(b.float, value)))))


def _nn_dropout(mut mods: NativeModules, value: PythonObject) raises -> PythonObject:
    var p = _nn_float(mods, value, "dropout")
    if not (Bool(py=p >= PythonObject(0)) and Bool(py=p < PythonObject(1))):
        raise Error("dropout must be in [0, 1)")
    return p


def _nn_flag(mut mods: NativeModules, value: PythonObject, name: String) raises -> Bool:
    var b = mods.module("builtins")
    if py_type(value) is not b.bool:
        raise Error("TypeError: " + name + " must be True or False")
    return Bool(py=value)


def _nn_source(value: PythonObject, absent: Int, name: String, context: PythonObject) raises -> Int:
    if value is None:
        return absent
    if not _integer(value, context) or Bool(py=value < PythonObject(-1)):
        raise Error(name + " must be -1 (the model input) or the index of an earlier layer")
    return Int(py=value)


def _nn_batch_first(mut mods: NativeModules, value: PythonObject) raises:
    if not _nn_flag(mods, value, "batch_first"):
        raise Error("batch_first must be True: Sequential samples are (length, features), the batch axis first")


def _nn_mha_args(heads: Int, flags: Int, source: Int, embed: Int, kdim: Int) raises -> PythonObject:
    return Python.list(heads, flags, source, embed, kdim, 0)


def _nn_attention_layer(mut mods: NativeModules, kind: String, supplied: PythonObject, keywords: PythonObject,
                        context: PythonObject) raises -> PythonObject:
    """(a0, a1, f, args) of a multihead_attention or positional_encoding layer."""
    if kind == "positional_encoding":
        var po = _arguments(mods, kind, supplied, keywords, Python.tuple("d_model", "dropout", "max_len"),
                            Python.tuple(None, 0.1, 5000), 1)
        var d = _positive(po[0], "d_model", 2, context)
        if d % 2 != 0:
            raise Error("d_model must be even")
        var p = _nn_dropout(mods, po[1])
        var ml = _positive(po[2], "max_len", 1, context)
        if ml > 8192:
            raise Error("max_len must be at most 8192")
        return Python.tuple(PythonObject(ml), PythonObject(d), p, Python.list(ml, d, 0, 0, 0, 0))
    var o = _arguments(mods, kind, supplied, keywords,
        Python.tuple("embed_dim", "num_heads", "dropout", "bias", "add_bias_kv", "add_zero_attn", "kdim", "vdim",
                     "batch_first", "source", "is_causal", "attn_mask", "key_padding_mask"),
        Python.tuple(None, None, 0.0, True, False, False, None, None, True, None, False, None, False), 2)
    var embed = _positive(o[0], "embed_dim", 1, context)
    var heads = _positive(o[1], "num_heads", 1, context)
    if embed % heads != 0:
        raise Error("embed_dim must be divisible by num_heads")
    var p = _nn_dropout(mods, o[2])
    var bias = _nn_flag(mods, o[3], "bias")
    if _nn_flag(mods, o[4], "add_bias_kv") or _nn_flag(mods, o[5], "add_zero_attn"):
        raise Error("add_bias_kv and add_zero_attn are not supported; leave them False")
    var kdim = 0 if o[6] is None else _positive(o[6], "kdim", 1, context)
    var vdim = 0 if o[7] is None else _positive(o[7], "vdim", 1, context)
    if kdim != 0 and vdim != 0 and kdim != vdim:
        raise Error("kdim and vdim must be equal: the keys and values are read from one source")
    if kdim == 0:
        kdim = vdim
    _nn_batch_first(mods, o[8])
    var source = _nn_source(o[9], _NN_SELF, "source", context)
    if source == _NN_SELF and kdim != 0 and kdim != embed:
        raise Error("self-attention takes kdim = vdim = embed_dim; give source= to attend to another layer")
    var causal = _nn_flag(mods, o[10], "is_causal")
    if o[11] is not None:
        var b = mods.module("builtins")
        if not (py_type(o[11]) is b.str and String(o[11]) == "causal"):
            raise Error("attn_mask takes None or 'causal' (a mask tensor is not supported: a Sequential layer holds "
                        + "no tensor arguments; use is_causal=True for the causal mask)")
        causal = True
    var kpm = _nn_flag(mods, o[12], "key_padding_mask")
    var flags = (1 if bias else 0) + (2 if causal else 0) + (4 if kpm else 0)
    return Python.tuple(PythonObject(heads), PythonObject(flags), p,
                        _nn_mha_args(heads, flags, source, embed, 0 if kdim == embed else kdim))


def _nn_add(mut mods: NativeModules, model: PythonObject, code: Int, args: PythonObject, f: PythonObject) raises -> Int:
    var nk = sequential_keys()
    var b = mods.module("builtins")
    var layers = py_item(sequential_fields(model), nk[].k__layers)
    var at = len(layers)
    _ = py_call(layers.append, Python.tuple(PythonObject(code), args[0], args[1], f, py_call(b.tuple, args)))
    return at


def _nn_simple(mut mods: NativeModules, model: PythonObject, code: Int, a0: Int, a1: Int, f: PythonObject) raises -> Int:
    return _nn_add(mods, model, code, Python.list(a0, a1, 0, 0, 0, 0), f)


def _nn_ffn(mut mods: NativeModules, model: PythonObject, d: Int, dff: Int, gelu: Bool, p: PythonObject) raises:
    """linear1, activation, dropout, linear2, dropout2 (PyTorch's _ff_block)."""
    _ = _nn_simple(mods, model, _NN_CODE_LINEAR, dff, 1, PythonObject(0.0))
    if gelu:
        _ = _nn_simple(mods, model, _NN_CODE_GELU, 0, 0, PythonObject(0.0))
    else:
        _ = _nn_simple(mods, model, _NN_CODE_RELU, 0, 0, PythonObject(0.0))
    _ = _nn_simple(mods, model, _NN_CODE_DROPOUT, 0, 0, p)
    _ = _nn_simple(mods, model, _NN_CODE_LINEAR, d, 1, PythonObject(0.0))
    _ = _nn_simple(mods, model, _NN_CODE_DROPOUT, 0, 0, p)


def _nn_mha(mut mods: NativeModules, model: PythonObject, d: Int, heads: Int, flags: Int, source: Int,
            p: PythonObject) raises -> Int:
    return _nn_add(mods, model, _NN_CODE_MHA, _nn_mha_args(heads, flags, source, d, 0), p)


def _nn_attention_container(mut mods: NativeModules, model: PythonObject, kind: String, supplied: PythonObject,
                            keywords: PythonObject, context: PythonObject) raises:
    var b = mods.module("builtins")
    if kind == "transformer_encoder" or kind == "transformer_decoder":
        var decoder = kind == "transformer_decoder"
        var stack_names = Python.tuple("encoder_layer", "num_layers", "norm")
        var stack_defaults = Python.tuple(None, None, None)
        if decoder:
            stack_names = Python.tuple("decoder_layer", "num_layers", "memory", "norm")
            stack_defaults = Python.tuple(None, None, None, None)
        var so = _arguments(mods, kind, supplied, keywords, stack_names, stack_defaults, 3 if decoder else 2)
        var template = _model_layers(mods, model, so[0], "decoder_layer" if decoder else "encoder_layer")
        if len(template) == 0:
            raise Error("the layer must hold at least one layer")
        var n = _positive(so[1], "num_layers", 1, context)
        var memory = 0
        var norm_at = 2
        if decoder:
            memory = _nn_source(so[2], -1, "memory", context)
            norm_at = 3
            var unset = 0
            for k in range(len(template)):  # driver-loop: over the decoder layer's description (admission)
                var layer = template[k]
                if Int(py=layer[0]) == _NN_CODE_MHA:
                    var src = Int(py=layer[4][2])
                    if src == _NN_MEMORY_UNSET:
                        unset += 1
                    elif src != _NN_SELF:
                        raise Error("decoder_layer must be built without memory (transformer_decoder sets it)")
            if unset == 0:
                raise Error("decoder_layer must hold a transformer_decoder_layer")
        var norm_layers = PythonObject(None)
        if so[norm_at] is not None:
            norm_layers = _model_layers(mods, model, so[norm_at], "norm")
        var nk = sequential_keys()
        var layers = py_item(sequential_fields(model), nk[].k__layers)
        var entry = _block_entry(mods, model)
        for _ in range(n):  # driver-loop: over num_layers copies of the layer (description only)
            var base = len(layers)
            entry = _emit(mods, model, template, entry)
            if decoder:
                for k in range(base, len(layers)):  # driver-loop: over the copy's layers (description only)
                    var layer = layers[k]
                    if Int(py=layer[0]) == _NN_CODE_MHA and Int(py=layer[4][2]) == _NN_MEMORY_UNSET:
                        var args = py_call(b.list, layer[4])
                        py_setitem(args, 2, memory)
                        py_setitem(layers, k, Python.tuple(layer[0], layer[1], layer[2], layer[3],
                                                           py_call(b.tuple, args)))
        if norm_layers is not None:
            _ = _emit(mods, model, norm_layers, entry)
        return
    var decoder = kind == "transformer_decoder_layer"
    var names = Python.tuple("d_model", "nhead", "dim_feedforward", "dropout", "activation", "layer_norm_eps",
                             "batch_first", "norm_first", "bias", "is_causal", "key_padding_mask")
    var defaults = Python.tuple(None, None, 2048, 0.1, "relu", 1e-5, True, False, True, False, False)
    if decoder:
        names = Python.tuple("d_model", "nhead", "dim_feedforward", "dropout", "activation", "layer_norm_eps",
                             "batch_first", "norm_first", "bias", "memory", "tgt_is_causal",
                             "tgt_key_padding_mask", "memory_key_padding_mask")
        defaults = Python.tuple(None, None, 2048, 0.1, "relu", 1e-5, True, False, True, None, False, False, False)
    var o = _arguments(mods, kind, supplied, keywords, names, defaults, 2)
    var d = _positive(o[0], "d_model", 1, context)
    var heads = _positive(o[1], "nhead", 1, context)
    if d % heads != 0:
        raise Error("d_model must be divisible by nhead")
    var dff = _positive(o[2], "dim_feedforward", 1, context)
    var p = _nn_dropout(mods, o[3])
    if py_type(o[4]) is not b.str or (String(o[4]) != "relu" and String(o[4]) != "gelu"):
        raise Error("activation must be 'relu' or 'gelu'")
    var gelu = String(o[4]) == "gelu"
    var eps = _nn_float(mods, o[5], "layer_norm_eps")
    if not Bool(py=eps > PythonObject(0)):
        raise Error("layer_norm_eps must be positive")
    _nn_batch_first(mods, o[6])
    var norm_first = _nn_flag(mods, o[7], "norm_first")
    if not _nn_flag(mods, o[8], "bias"):
        raise Error("bias=False is not supported: layer_norm always has a bias")
    var self_flags = 1 + (2 if _nn_flag(mods, o[10 if decoder else 9], "tgt_is_causal" if decoder else "is_causal") else 0)
    self_flags += 4 if _nn_flag(mods, o[11 if decoder else 10],
                                "tgt_key_padding_mask" if decoder else "key_padding_mask") else 0
    var entry = _block_entry(mods, model)
    var zero = PythonObject(0.0)
    if not decoder:
        if norm_first:
            # x = x + sa(norm1(x)); x = x + ff(norm2(x))
            _ = _nn_simple(mods, model, _NN_CODE_LAYER_NORM, 0, 0, eps)
            _ = _nn_mha(mods, model, d, heads, self_flags, _NN_SELF, p)
            _ = _nn_simple(mods, model, _NN_CODE_DROPOUT, 0, 0, p)
            var r1 = _nn_simple(mods, model, _CODE_RESIDUAL, entry, 0, zero)
            _ = _nn_simple(mods, model, _NN_CODE_LAYER_NORM, 0, 0, eps)
            _nn_ffn(mods, model, d, dff, gelu, p)
            _ = _nn_simple(mods, model, _CODE_RESIDUAL, r1, 0, zero)
        else:
            # x = norm1(x + sa(x)); x = norm2(x + ff(x))
            _ = _nn_mha(mods, model, d, heads, self_flags, _NN_SELF, p)
            _ = _nn_simple(mods, model, _NN_CODE_DROPOUT, 0, 0, p)
            _ = _nn_simple(mods, model, _CODE_RESIDUAL, entry, 0, zero)
            var n1 = _nn_simple(mods, model, _NN_CODE_LAYER_NORM, 0, 0, eps)
            _nn_ffn(mods, model, d, dff, gelu, p)
            _ = _nn_simple(mods, model, _CODE_RESIDUAL, n1, 0, zero)
            _ = _nn_simple(mods, model, _NN_CODE_LAYER_NORM, 0, 0, eps)
        return
    var memory = _nn_source(o[9], _NN_MEMORY_UNSET, "memory", context)
    var cross_flags = 1 + (4 if _nn_flag(mods, o[12], "memory_key_padding_mask") else 0)
    if norm_first:
        # x = x + sa(norm1(x)); x = x + mha(norm2(x), memory); x = x + ff(norm3(x))
        _ = _nn_simple(mods, model, _NN_CODE_LAYER_NORM, 0, 0, eps)
        _ = _nn_mha(mods, model, d, heads, self_flags, _NN_SELF, p)
        _ = _nn_simple(mods, model, _NN_CODE_DROPOUT, 0, 0, p)
        var r1 = _nn_simple(mods, model, _CODE_RESIDUAL, entry, 0, zero)
        _ = _nn_simple(mods, model, _NN_CODE_LAYER_NORM, 0, 0, eps)
        _ = _nn_mha(mods, model, d, heads, cross_flags, memory, p)
        _ = _nn_simple(mods, model, _NN_CODE_DROPOUT, 0, 0, p)
        var r2 = _nn_simple(mods, model, _CODE_RESIDUAL, r1, 0, zero)
        _ = _nn_simple(mods, model, _NN_CODE_LAYER_NORM, 0, 0, eps)
        _nn_ffn(mods, model, d, dff, gelu, p)
        _ = _nn_simple(mods, model, _CODE_RESIDUAL, r2, 0, zero)
    else:
        # x = norm1(x + sa(x)); x = norm2(x + mha(x, memory)); x = norm3(x + ff(x))
        _ = _nn_mha(mods, model, d, heads, self_flags, _NN_SELF, p)
        _ = _nn_simple(mods, model, _NN_CODE_DROPOUT, 0, 0, p)
        _ = _nn_simple(mods, model, _CODE_RESIDUAL, entry, 0, zero)
        var n1 = _nn_simple(mods, model, _NN_CODE_LAYER_NORM, 0, 0, eps)
        _ = _nn_mha(mods, model, d, heads, cross_flags, memory, p)
        _ = _nn_simple(mods, model, _NN_CODE_DROPOUT, 0, 0, p)
        _ = _nn_simple(mods, model, _CODE_RESIDUAL, n1, 0, zero)
        var n2 = _nn_simple(mods, model, _NN_CODE_LAYER_NORM, 0, 0, eps)
        _nn_ffn(mods, model, d, dff, gelu, p)
        _ = _nn_simple(mods, model, _CODE_RESIDUAL, n2, 0, zero)
        _ = _nn_simple(mods, model, _NN_CODE_LAYER_NORM, 0, 0, eps)
# nn-attention end
