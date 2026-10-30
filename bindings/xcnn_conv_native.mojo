# SPDX-License-Identifier: Apache-2.0
"""Whole native Conv1d/Conv2d construction, input admission, forward,
backward, weight loading, pinned state and consumed gradient conversion.
Shared tensor ownership and output allocation live in xcnn_common. Host
channel slices use native buffer copies; numerical work uses the existing
compiled convolution/padding entries. BasicBlock forward composes these
owners; its remaining public operations have a separate migration batch.
"""
from std.math import sqrt
from std.memory import memcpy
from std.python import Python, PythonObject
from bindings.mode_metadata_native import _fields
from bindings.neural_block_admission import neural_plain_value, neural_string_keys
from bindings.sequential_admission import sequential_admit_value, _one_of
from bindings.xseq_native_common import XseqStream
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_fields, py_contains
from bindings.py_typed_call import py_call_typed
from bindings.xcnn_common import (
    XCNN_F_GROUP, XCNN_F_PAD, XCNN_F_PIN, XCNN_F_LAYER_IO, xcnn_builtins, xcnn_entries, xcnn_entry, xcnn_mixed,
    xcnn_idn, xcnn_resident, xcnn_shape, xcnn_size, xcnn_tuple, xcnn_is_t, xcnn_on_dev, xcnn_new, xcnn_out_new,
    xcnn_reshape_t, xcnn_wrap, xcnn_host_of, xcnn_host_f32, xcnn_f32, xcnn_reshape, xcnn_addr, xcnn_dev, xcnn_dev_get,
    xcnn_dev_upload, xcnn_download, xcnn_slice_f32, xcnn_put_f32,
)
from bindings.xcnn_bn_native import xcnn_bn_forward_dev, xcnn_bn_forward, xcnn_bn_layers, xcnn_bn_backward, xcnn_bn_backward_dev, _bn_admit
from bindings.xseq_native_common import xseq_f32
from bindings.sequential_inputs import sequential_input


def _set(fields: PythonObject, name: String, value: PythonObject) raises:
    py_setitem(fields, PythonObject(name), value)


def _pair(fields: PythonObject, name: String) raises -> List[Int]:
    var v = fields[PythonObject(name)]
    neural_plain_value(v, 0, PythonObject(None))
    return [Int(py=v[0]), Int(py=v[1])]


def _truth(value: PythonObject) raises -> Bool:
    var cls = py_type(value)
    var b = xcnn_builtins()
    if value is not None and cls is not b.bool and cls is not b.int and cls is not b.float and cls is not b.str and cls is not b.tuple and cls is not b.list and cls is not b.dict:
        raise Error("TypeError: convolution state requires native truth metadata")
    return Bool(py=value)


# ---------------------------------------------------------------- public configuration and buffer preparation

def _snapshot(value: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    var admitted = xcnn_f32(value, name, context)
    # A fresh native view freezes the shape/strides for this operation while
    # retaining the admitted buffer's original storage owner.
    return xcnn_reshape(admitted, xcnn_shape(admitted))


def _weight_pair(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    var w = _snapshot(fields["weight_"], "weight_", context)
    var b = _snapshot(fields["bias_"], "bias_", context)
    var k = _pair(fields, "kernel_size")
    var groups = Int(py=fields["groups"])
    var inc = Int(py=fields["in_channels"])
    var outc = Int(py=fields["out_channels"])
    if not _same_shape(xcnn_shape(w), [outc, inc // groups, k[0], k[1]]) or not _same_shape(xcnn_shape(b), [outc]):
        raise Error("mojolearn: convolution weight or bias shape differs from configured channels and kernel")
    return Python.tuple(w, b)


def _conv_admit(layer: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = _fields(layer)
    # driver-loop: fixed passive convolution configuration fields
    for name in ["in_channels", "out_channels", "kernel_size", "stride", "dilation", "groups", "padding_mode",
                 "padding", "_pads", "_explicit", "_conv_pad", "bias", "random_state", "input_shape", "numeric_mode"]:
        # driver-loop: fixed passive layer configuration; no numeric/protocol hooks
        neural_plain_value(py_get(fields, PythonObject(name)), 0, Python.dict(admission_context=context))
    return fields


def _dimension(value: PythonObject, name: String, context: PythonObject) raises -> Int:
    neural_plain_value(value, 0, Python.dict(admission_context=context))
    var b = xcnn_builtins()
    var cls = py_type(value)
    if cls is not b.int and cls is not b.float and cls is not b.bool and cls is not b.str and not _one_of(cls, context["scalar_types"]):
        raise Error("TypeError: " + name + " requires a native numeric value")
    return Int(py=py_call(b.int, value))


def _size_pair(value: PythonObject, name: String, one: Bool, context: PythonObject) raises -> List[Int]:
    neural_plain_value(value, 0, Python.dict(admission_context=context))
    var b = xcnn_builtins()
    if py_type(value) is b.int or py_type(value) is b.bool:
        var n = _dimension(value, name, context)
        return [1, n] if one else [n, n]
    if py_type(value) is not b.tuple and py_type(value) is not b.list:
        raise Error("TypeError: " + name + " requires a native integer or size sequence")
    if len(value) != (1 if one else 2):
        raise Error(name + " requires " + ("one dimension" if one else "a pair"))
    return [1, _dimension(value[0], name, context)] if one else [_dimension(value[0], name, context), _dimension(value[1], name, context)]


def _same_shape(left: List[Int], right: List[Int]) -> Bool:
    if len(left) != len(right):
        return False
    for i in range(len(left)):  # small-loop(tensor rank)
        if left[i] != right[i]:
            return False
    return True


def _zeros(shape: List[Int], context: PythonObject) raises -> PythonObject:
    # Shared output storage is a freshly zero-initialized native bytearray.
    return xcnn_host_f32(shape, context)


def _channels(source: PythonObject, start: Int, count: Int, context: PythonObject) raises -> PythonObject:
    # Copy one NCHW channel group without ndarray slicing or assignment hooks.
    var shape = xcnn_shape(source)
    var out = xcnn_host_f32([shape[0], count, shape[2], shape[3]], context)
    var src = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=xcnn_addr(source)))
    var dst = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=xcnn_addr(out)))
    var spatial = shape[2] * shape[3]
    for n in range(shape[0]):  # driver-loop: one native contiguous copy per image
        memcpy(dest=dst + n * count * spatial, src=src + (n * shape[1] + start) * spatial, count=count * spatial)
    _ = source
    _ = out
    return out


def _put_channels(dest: PythonObject, start: Int, count: Int, source: PythonObject) raises:
    var shape = xcnn_shape(dest)
    var src = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=xcnn_addr(source)))
    var dst = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=xcnn_addr(dest)))
    var spatial = shape[2] * shape[3]
    for n in range(shape[0]):  # driver-loop: one native contiguous copy per image
        memcpy(dest=dst + (n * shape[1] + start) * spatial, src=src + n * count * spatial, count=count * spatial)
    _ = source
    _ = dest


def _copy_words(dest: PythonObject, offset: Int, source: PythonObject, count: Int) raises:
    if count:
        memcpy(dest=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=xcnn_addr(dest))) + offset,
               src=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=xcnn_addr(source))), count=count)
    _ = source
    _ = dest


def _conv_init(layer: PythonObject, args: PythonObject, one: Bool, context: PythonObject) raises -> PythonObject:
    neural_plain_value(args, 0, Python.dict(admission_context=context))
    var fields = _fields(layer)
    var inc = _dimension(args[0], "in_channels", context)
    var outc = _dimension(args[1], "out_channels", context)
    var k = _size_pair(args[2], "kernel_size", one, context)
    var s = _size_pair(args[3], "stride", one, context)
    var d = _size_pair(args[5], "dilation", one, context)
    var groups = _dimension(args[6], "groups", context)
    if inc <= 0 or outc <= 0 or groups <= 0 or inc % groups or outc % groups:
        raise Error("mojolearn: positive in_channels and out_channels must be divisible by groups")
    if k[0] <= 0 or k[1] <= 0 or s[0] <= 0 or s[1] <= 0 or d[0] <= 0 or d[1] <= 0:
        raise Error("mojolearn: kernel_size, stride and dilation must be positive")
    var b = xcnn_builtins()
    if py_type(args[8]) is not b.str:
        raise Error("TypeError: padding_mode requires a native string")
    var mode = String(args[8])
    if mode != "zeros" and mode != "reflect" and mode != "replicate" and mode != "circular":
        raise Error("mojolearn: padding_mode must be zeros, reflect, replicate or circular")
    var pads: List[Int] = [0, 0, 0, 0]
    var padding = args[4]
    if py_type(padding) is b.str:
        var text = String(padding)
        if text == "same":
            if s[0] != 1 or s[1] != 1:
                raise Error("mojolearn: padding='same' is not supported for strided convolutions")
            var h = d[0] * (k[0] - 1)
            var w = d[1] * (k[1] - 1)
            pads = [h // 2, h - h // 2, w // 2, w - w // 2]
        elif text != "valid":
            raise Error("mojolearn: invalid padding string; expected valid or same")
    else:
        var p = _size_pair(padding, "padding", one, context)
        if one:
            p[0] = 0
        if p[0] < 0 or p[1] < 0:
            raise Error("mojolearn: padding must be nonnegative")
        pads = [p[0], p[0], p[1], p[1]]
        padding = xcnn_tuple(p)
    var use_bias = Bool(py=py_call(b.bool, args[7]))
    var explicit = mode != "zeros" or pads[0] != pads[1] or pads[2] != pads[3]
    _set(fields, "in_channels", PythonObject(inc))
    _set(fields, "out_channels", PythonObject(outc))
    _set(fields, "kernel_size", xcnn_tuple(k))
    _set(fields, "stride", xcnn_tuple(s))
    _set(fields, "dilation", xcnn_tuple(d))
    _set(fields, "groups", PythonObject(groups))
    _set(fields, "padding_mode", args[8])
    _set(fields, "padding", padding)
    _set(fields, "_pads", xcnn_tuple(pads))
    _set(fields, "_explicit", PythonObject(explicit))
    _set(fields, "_conv_pad", Python.tuple(0 if explicit else pads[0], 0 if explicit else pads[2]))
    _set(fields, "bias", PythonObject(use_bias))
    _set(fields, "random_state", PythonObject(None) if args[9] is None else py_call(b.int, args[9]))
    _set(fields, "input_shape", args[10])
    _set(fields, "numeric_mode", args[11])
    var seed = PythonObject(None) if args[9] is None else py_call(b.int, args[9])
    var stream = XseqStream(seed, context)
    var fan = inc // groups * k[0] * k[1]
    var bound = 1.0 / sqrt(Float64(fan))
    var w = xcnn_host_f32([outc, inc // groups, k[0], k[1]], context)
    stream.fill_uniform(context, xcnn_addr(w), outc * fan, -bound, bound)
    var bias = _zeros([outc], context)
    if use_bias:
        stream.fill_uniform(context, xcnn_addr(bias), outc, -bound, bound)
    _set(fields, "weight_", w)
    _set(fields, "bias_", bias)
    return PythonObject(None)


def _conv_set_weights(layer: PythonObject, args: PythonObject, one: Bool, context: PythonObject) raises -> PythonObject:
    var fields = _conv_admit(layer, context)
    var k = _pair(fields, "kernel_size")
    var outc = Int(py=fields["out_channels"])
    var inc = Int(py=fields["in_channels"]) // Int(py=fields["groups"])
    var expected: List[Int] = [outc, inc, k[0], k[1]]
    var w = xcnn_f32(args[0], "weight", context)
    if one:
        w = xcnn_reshape(w, expected)
    if not _same_shape(xcnn_shape(w), expected):
        raise Error("mojolearn: weight shape differs from the configured convolution shape")
    var new_w = xcnn_reshape(xcnn_slice_f32(w, 0, xcnn_size(expected), context), expected)
    var bias = args[1]
    var new_b = py_get(fields, "bias_")
    if bias is not None:
        var b = xcnn_f32(bias, "bias", context)
        if xcnn_size(xcnn_shape(b)) != outc:
            raise Error("mojolearn: bias must contain out_channels values")
        new_b = xcnn_slice_f32(b, 0, outc, context)
    elif not _truth(fields["bias"]):
        new_b = _zeros([outc], context)
    _set(fields, "weight_", new_w)
    _set(fields, "bias_", new_b)
    if py_get(fields, "_pins") is not None:
        _pin_weights(layer, fields, xcnn_entries(fields, context), context)
    return layer


def _conv_gradient(layer: PythonObject, weight: Bool, context: PythonObject) raises -> PythonObject:
    var fields = _conv_admit(layer, context)
    var key = "grad_weight_" if weight else "grad_bias_"
    var pending_key = "_gw_dev" if weight else "_gb_dev"
    var pending = py_get(fields, PythonObject(pending_key))
    if pending is not None:
        var b = xcnn_builtins()
        if py_type(pending) is not b.tuple or len(pending) != 2 or py_type(pending[0]) is not context["xcnn_dev_class"] or py_type(pending[1]) is not b.int:
            raise Error("TypeError: convolution gradient storage requires native metadata")
        var shape = xcnn_shape(xcnn_f32(fields["weight_"], "weight_", context)) if weight else [Int(py=fields["out_channels"])]
        var value = xcnn_download(pending[0], pending[1], xcnn_size(shape), context)
        if weight and _truth(py_get(fields, "_gw_1d")):
            shape = [shape[0], shape[1], shape[3]]
        value = xcnn_reshape(value, shape)
        _set(fields, key, value)
        _set(fields, pending_key, PythonObject(None))
    if not py_contains(fields, PythonObject(key)):
        raise Error("AttributeError: " + key)
    return fields[PythonObject(key)]


def _save_gradients(fields: PythonObject, dw: PythonObject, db: PythonObject, context: PythonObject) raises:
    _set(fields, "grad_weight_", dw)
    _set(fields, "_gw_dev", PythonObject(None))
    _set(fields, "_gw_1d", PythonObject(False))
    _set(fields, "grad_bias_", db if _truth(fields["bias"]) else _zeros([Int(py=fields["out_channels"])], context))
    _set(fields, "_gb_dev", PythonObject(None))


# ---------------------------------------------------------------- backward, with both host and resident routes

def _conv_backward_host(layer: PythonObject, grad: PythonObject, source: PythonObject,
                        context: PythonObject) raises -> PythonObject:
    var fields = _conv_admit(layer, context)
    var entries = xcnn_entries(fields, context)
    var x = xcnn_host_of(source, context) if xcnn_is_t(source, context) else source
    var xs = xcnn_shape(x)
    _check_x(fields, xs)
    var g = xcnn_host_of(grad, context) if xcnn_is_t(grad, context) else grad
    var expected = _conv_out(fields, entries, xs)
    if not _same_shape(xcnn_shape(g), expected):
        raise Error("mojolearn: grad_out shape differs from the convolution output shape")
    var xp = x
    if _explicit(fields):
        xp = xcnn_host_f32(_padded(fields, xs), context)
        _ = py_call_typed(xcnn_entry(entries, "x_cnn_pad2d_forward"), xcnn_addr(x), xcnn_addr(xp),
                          _pad_params(layer, fields, xs, -1))
    var ps = xcnn_shape(xp)
    var prm = _conv_params(fields, ps)
    var groups = Int(py=fields["groups"])
    var cg = Int(py=fields["in_channels"]) // groups
    var og = Int(py=fields["out_channels"]) // groups
    var w = _snapshot(fields["weight_"], "weight_", context)
    var ws = xcnn_shape(w)
    var k = _pair(fields, "kernel_size")
    if not _same_shape(ws, [og * groups, cg, k[0], k[1]]):
        raise Error("mojolearn: weight_ shape differs from configured convolution")
    var dxp = xcnn_host_f32(ps, context)
    var dw = xcnn_host_f32(ws, context)
    var db = xcnn_host_f32([og * groups], context)
    var conv = xcnn_entry(entries, "x_cnn_conv2d_backward")
    var per = xcnn_size(ws) // groups
    for group in range(groups):  # driver-loop: one numerical engine call per convolution group
        var xg = xp if groups == 1 else _channels(xp, group * cg, cg, context)
        var gg = g if groups == 1 else _channels(g, group * og, og, context)
        var wg = xcnn_slice_f32(w, group * per, per, context)
        var dxg = dxp if groups == 1 else xcnn_host_f32([ps[0], cg, ps[2], ps[3]], context)
        var dwg = xcnn_host_f32([per], context)
        var dbg = xcnn_host_f32([og], context)
        _ = py_call_typed(conv, xcnn_addr(xg), xcnn_addr(wg), xcnn_addr(gg), xcnn_addr(dxg),
                          xcnn_addr(dwg), xcnn_addr(dbg), prm)
        if groups > 1:
            _put_channels(dxp, group * cg, cg, dxg)
        _copy_words(dw, group * per, dwg, per)
        _copy_words(db, group * og, dbg, og)
        _ = xg
        _ = gg
        _ = wg
        _ = dxg
    var dx = dxp
    if _explicit(fields):
        dx = xcnn_host_f32(xs, context)
        _ = py_call_typed(xcnn_entry(entries, "x_cnn_pad2d_backward"), xcnn_addr(dxp), xcnn_addr(dx),
                          _pad_params(layer, fields, xs, -1))
        # The raw address does not retain its padded-gradient buffer. Keep
        # its owner through the native upload/read before returning new dx.
        _ = dxp
    _set(fields, "_x", x)
    _set(fields, "_xp", xp)
    _save_gradients(fields, dw, db, context)
    _ = g
    _ = w
    return dx


def _conv_backward_tensor(layer: PythonObject, grad: PythonObject, source: PythonObject,
                          context: PythonObject) raises -> PythonObject:
    var fields = _conv_admit(layer, context)
    var entries = xcnn_entries(fields, context)
    if not (_res_ok(fields, entries, source, context) and xcnn_on_dev(entries, grad, context)):
        var dx = _conv_backward_host(layer, grad, source, context)
        return xcnn_wrap(entries, dx, context) if xcnn_is_t(grad, context) else dx
    var xs = xcnn_shape(source)
    _check_x(fields, xs)
    var shape = _conv_out(fields, entries, xs)
    if not _same_shape(xcnn_shape(grad), shape):
        raise Error("mojolearn: grad_out shape differs from the convolution output shape")
    var xp = _pad_t(layer, fields, entries, source, context)
    var ps = xcnn_shape(xp)
    var prm = _conv_params(fields, ps)
    var dxp = xcnn_out_new(entries, ps, context)
    var values = _weight_pair(fields, context)
    var w = values[0]
    var ws = xcnn_shape(w)
    var groups = Int(py=fields["groups"])
    var inc = Int(py=fields["in_channels"])
    var outc = Int(py=fields["out_channels"])
    var k = _pair(fields, "kernel_size")
    if not _same_shape(ws, [outc, inc // groups, k[0], k[1]]):
        raise Error("mojolearn: weight_ shape differs from configured convolution")
    var dw = xcnn_host_f32(ws, context)
    var db = xcnn_host_f32([outc], context)
    var conv = xcnn_entry(entries, "x_cnn_conv2d_backward_m")
    if groups == 1 and xcnn_resident(entries, context):
        var pair = _res_w_values(layer, fields, entries, values, context)
        var pw = pair[0]
        var gwh = xcnn_dev_get(layer, entries, "gw", xcnn_size(ws), context)
        var gbh = xcnn_dev_get(layer, entries, "gb", outc, context)
        _ = py_call_typed(conv, Python.list(xp.h, pw if pw is not None else xcnn_addr(w), grad.h, dxp.h, gwh, gbh),
                          PythonObject(63 if pw is not None else 61), prm)
        var dev = xcnn_dev(layer, entries, context)
        _set(fields, "_gw_dev", Python.tuple(dev, gwh))
        _set(fields, "_gw_1d", PythonObject(False))
        _set(fields, "_gb_dev", Python.tuple(dev, gbh) if _truth(fields["bias"]) else PythonObject(None))
        if not _truth(fields["bias"]):
            _set(fields, "grad_bias_", _zeros([outc], context))
    elif groups == 1:
        var pw = _pinned(fields, entries, "weight_", xcnn_size(ws))
        _ = py_call_typed(conv, Python.list(xp.h, pw if pw is not None else xcnn_addr(w), grad.h, dxp.h,
                                          xcnn_addr(dw), xcnn_addr(db)), PythonObject(15 if pw is not None else 13), prm)
        _save_gradients(fields, dw, db, context)
    else:
        var og = outc // groups
        var per = xcnn_size(ws) // groups
        var xg = xcnn_dev_get(layer, entries, "xg", xcnn_size(ps) // groups, context)
        var gg = xcnn_dev_get(layer, entries, "gg", xcnn_size(shape) // groups, context)
        var dxg = xcnn_dev_get(layer, entries, "dxg", xcnn_size(ps) // groups, context)
        var copy = xcnn_entry(entries, "x_cnn_chan_copy_m")
        for group in range(groups):  # driver-loop: one launch sequence per convolution group
            var wg = xcnn_slice_f32(w, group * per, per, context)
            var dwg = xcnn_host_f32([per], context)
            var dbg = xcnn_host_f32([og], context)
            _ = py_call_typed(copy, Python.list(xp.h, xg), PythonObject(3), _group_prm(ps, inc, groups, group, 0))
            _ = py_call_typed(copy, Python.list(grad.h, gg), PythonObject(3), _group_prm(shape, outc, groups, group, 0))
            _ = py_call_typed(conv, Python.list(xg, xcnn_addr(wg), gg, dxg, xcnn_addr(dwg), xcnn_addr(dbg)), PythonObject(13), prm)
            _ = py_call_typed(copy, Python.list(dxg, dxp.h), PythonObject(3), _group_prm(ps, inc, groups, group, 1))
            _copy_words(dw, group * per, dwg, per)
            _copy_words(db, group * og, dbg, og)
            _ = wg
        _save_gradients(fields, dw, db, context)
    var dx = dxp
    if _explicit(fields):
        dx = xcnn_out_new(entries, xs, context)
        _ = py_call_typed(xcnn_entry(entries, "x_cnn_pad2d_m"), Python.list(dxp.h, dx.h), PythonObject(3),
                          _pad_params(layer, fields, xs, 1))
    _set(fields, "_x", source)
    _set(fields, "_xp", xp)
    _ = w
    _ = grad
    return dx


def _to_four(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var v = value
    var s = xcnn_shape(v)
    if len(s) != 3:
        raise Error("mojolearn: expected (N, C, L)")
    var shape: List[Int] = [s[0], s[1], 1, s[2]]
    return xcnn_reshape_t(v, shape, context) if xcnn_is_t(v, context) else xcnn_reshape(v, shape)


def _conv_backward(layer: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = _conv_admit(layer, context)
    var source = args[1] if args[1] is not None else py_get(fields, "_x")
    if source is None:
        raise Error("RuntimeError: convolution backward needs a prior forward or an explicit x")
    var one = py_type(layer) is context["xcnn_conv1d_class"]
    var grad = args[0] if xcnn_is_t(args[0], context) else _snapshot(args[0], "grad_out", context)
    source = source if xcnn_is_t(source, context) else _snapshot(source, "x", context)
    if one:
        grad = _to_four(grad, context)
    if one and args[1] is not None:
        source = _to_four(source, context)
    var entries = xcnn_entries(fields, context)
    var out = PythonObject(None)
    if xcnn_is_t(grad, context) or xcnn_is_t(source, context):
        out = _conv_backward_tensor(layer, grad, source, context)
    elif _group_dev(fields, entries):
        var xs = source
        var gs = grad
        out = xcnn_host_of(_conv_backward_tensor(layer, xcnn_wrap(entries, gs, context),
                                               xcnn_wrap(entries, xs, context), context), context)
        _set(fields, "_x", xs)
        _set(fields, "_xp", PythonObject(None))
    else:
        out = _conv_backward_host(layer, grad, source, context)
    if one:
        if py_get(fields, "_gw_dev") is not None:
            _set(fields, "_gw_1d", PythonObject(True))
        else:
            var dw = fields["grad_weight_"]
            var ws = xcnn_shape(dw)
            _set(fields, "grad_weight_", xcnn_reshape(dw, [ws[0], ws[1], ws[3]]))
        var os = xcnn_shape(out)
        var shape: List[Int] = [os[0], os[1], os[3]]
        return xcnn_reshape_t(out, shape, context) if xcnn_is_t(out, context) else xcnn_reshape(out, shape)
    return out


# ---------------------------------------------------------------- conv words

def _conv_params(fields: PythonObject, shape: List[Int]) raises -> PythonObject:
    """`Conv2d._params(shape)`: one group's conv over the (padded) input."""
    var groups = Int(py=fields["groups"])
    var k = _pair(fields, "kernel_size")
    var s = _pair(fields, "stride")
    var p = _pair(fields, "_conv_pad")
    var d = _pair(fields, "dilation")
    return Python.list(shape[0], shape[1] // groups, shape[2], shape[3], Int(py=fields["out_channels"]) // groups,
                       k[0], k[1], s[0], s[1], p[0], p[1], d[0], d[1], 0, 0, 1 if _truth(fields["bias"]) else 0, 0)


def _pads(fields: PythonObject) raises -> List[Int]:
    var v = fields["_pads"]
    return [Int(py=v[0]), Int(py=v[1]), Int(py=v[2]), Int(py=v[3])]


def _pad_params(layer: PythonObject, fields: PythonObject, shape: List[Int], last: Int) raises -> PythonObject:
    """`Conv2d._pad_params(shape) + [last]` (`last` < 0: none)."""
    var b = xcnn_builtins()
    var p = _pads(fields)
    var mode = String(fields["padding_mode"])
    var code = 0 if mode == "zeros" else (1 if mode == "reflect" else (2 if mode == "replicate" else 3))
    var out = Python.list(shape[0], shape[1], shape[2], shape[3], p[0], p[1], p[2], p[3], code)
    if last >= 0:
        _ = py_call(py_call(b.getattr, out, "append"), PythonObject(last))
    return out


def _explicit(fields: PythonObject) raises -> Bool:
    return _truth(fields["_explicit"])


def _padded(fields: PythonObject, shape: List[Int]) raises -> List[Int]:
    if not _explicit(fields):
        return shape.copy()
    var p = _pads(fields)
    return [shape[0], shape[1], shape[2] + p[0] + p[1], shape[3] + p[2] + p[3]]


def _conv_out(fields: PythonObject, entries: PythonObject, shape: List[Int]) raises -> List[Int]:
    """`Conv2d._out_shape(shape)` (`x_cnn_conv_shape`)."""
    var oh_ow = py_call_typed(xcnn_entry(entries, "x_cnn_conv_shape"), _conv_params(fields, _padded(fields, shape)))
    return [shape[0], Int(py=fields["out_channels"]), Int(py=oh_ow[0]), Int(py=oh_ow[1])]


def _check_x(fields: PythonObject, shape: List[Int]) raises:
    if len(shape) != 4:
        raise Error("mojolearn: Conv2d.forward takes (N, C, H, W)")
    if shape[1] != Int(py=fields["in_channels"]):
        raise Error("mojolearn: input has " + String(shape[1]) + " channels, the layer " + String(Int(py=fields["in_channels"])))


def _group_prm(shape: List[Int], channels: Int, groups: Int, g: Int, direction: Int) raises -> PythonObject:
    var per = channels // groups
    return Python.list(shape[0], channels, shape[2] * shape[3], per, g * per, direction)


def _host_rows(source: PythonObject, start: Int, stop: Int, context: PythonObject) raises -> PythonObject:
    """`np.ascontiguousarray(source[start:stop], np.float32)`."""
    var a = source
    var shape = xcnn_shape(a)
    var width = xcnn_size(shape) // max(shape[0], 1)
    var out = xcnn_slice_f32(a, start * width, (stop - start) * width, context)
    shape[0] = stop - start
    return xcnn_reshape(out, shape)


# ---------------------------------------------------------------- pinned weights (G3)

def _pin_values(layer: PythonObject, fields: PythonObject, entries: PythonObject, values: PythonObject, context: PythonObject) raises:
    """`_pin_weights(layer, ("weight_", "bias_"))`: both uploaded once."""
    var b = xcnn_builtins()
    _set(fields, "_pins", PythonObject(None))
    if not xcnn_idn(entries, XCNN_F_PIN):
        return
    var handles = Python.dict()
    var names = Python.tuple("weight_", "bias_")
    for i in range(2):  # driver-loop: over the two named weight arrays
        var a = values[i]
        var h = xcnn_dev_upload(layer, entries, "pin_" + String(py=names[i]), a, context)
        py_setitem(handles, names[i], Python.tuple(h, xcnn_size(xcnn_shape(a))))
    _set(fields, "_pins", Python.tuple(entries["module"], handles))
    # the arrays pinned, so a reassigned `weight_` / `bias_` is uploaded again
    _set(fields, "_pin_src", Python.tuple(py_get(fields, "weight_"), py_get(fields, "bias_")))


def _pin_weights(layer: PythonObject, fields: PythonObject, entries: PythonObject, context: PythonObject) raises:
    _pin_values(layer, fields, entries, _weight_pair(fields, context), context)


def _pin_metadata(fields: PythonObject) raises:
    var b = xcnn_builtins()
    var p = py_get(fields, "_pins")
    var src = py_get(fields, "_pin_src")
    if src is not None and (py_type(src) is not b.tuple or len(src) != 2):
        raise Error("TypeError: pinned sources require a native pair")
    if p is None:
        return
    if py_type(p) is not b.tuple or len(p) != 2:
        raise Error("TypeError: pinned weights require native metadata")
    neural_string_keys(p[1])
    var names = Python.tuple("weight_", "bias_")
    for i in range(2):  # small-loop(fixed convolution pin names)
        var hit = py_get(p[1], names[i])
        if hit is not None:
            if py_type(hit) is not b.tuple or len(hit) != 2 or py_type(hit[0]) is not b.int or py_type(hit[1]) is not b.int:
                raise Error("TypeError: pinned weights require native handle and size words")


def _auto_pin(layer: PythonObject, fields: PythonObject, entries: PythonObject, values: PythonObject, context: PythonObject) raises -> Bool:
    """`_auto_pin(layer, b, ("weight_", "bias_"))`."""
    if not (xcnn_resident(entries, context) and xcnn_idn(entries, XCNN_F_PIN)):
        return False
    _pin_metadata(fields)
    var p = py_get(fields, "_pins")
    var src = py_get(fields, "_pin_src")
    if p is not None and _truth(p) and p[0] is entries["module"] and src is not None and len(src) == 2:
        if src[0] is py_get(fields, "weight_") and src[1] is py_get(fields, "bias_"):
            return True
    _pin_values(layer, fields, entries, values, context)
    return _truth(py_get(fields, "_pins"))


def _pinned(fields: PythonObject, entries: PythonObject, name: String, size: Int) raises -> PythonObject:
    """`_pinned(layer, b, name, size)`: the handle, or None."""
    _pin_metadata(fields)
    var p = py_get(fields, "_pins")
    if p is None or not _truth(p) or p[0] is not entries["module"]:
        return PythonObject(None)
    var hit = py_get(p[1], PythonObject(name))
    if hit is None or Int(py=hit[1]) != size:
        return PythonObject(None)
    return hit[0]


def _res_w_values(layer: PythonObject, fields: PythonObject, entries: PythonObject, values: PythonObject, context: PythonObject) raises -> PythonObject:
    """`Conv2d._res_w(b)`: (weight handle, bias handle) or (None, None)."""
    if Int(py=fields["groups"]) != 1 or not _auto_pin(layer, fields, entries, values, context):
        return Python.tuple(PythonObject(None), PythonObject(None))
    return Python.tuple(_pinned(fields, entries, "weight_", xcnn_size(xcnn_shape(values[0]))),
                        _pinned(fields, entries, "bias_", xcnn_size(xcnn_shape(values[1]))))


def _res_w(layer: PythonObject, fields: PythonObject, entries: PythonObject, context: PythonObject) raises -> PythonObject:
    return _res_w_values(layer, fields, entries, _weight_pair(fields, context), context)


# ---------------------------------------------------------------- Conv2d

def _res_ok(fields: PythonObject, entries: PythonObject, x: PythonObject, context: PythonObject) raises -> Bool:
    if not xcnn_on_dev(entries, x, context):
        return False
    if Int(py=fields["groups"]) != 1 and not xcnn_idn(entries, XCNN_F_GROUP):
        return False
    if _explicit(fields) and not xcnn_idn(entries, XCNN_F_PAD):
        return False
    return True


def _pad_t(layer: PythonObject, fields: PythonObject, entries: PythonObject, x: PythonObject,
           context: PythonObject) raises -> PythonObject:
    """`Conv2d._pad_t`: the explicitly padded resident copy, or x."""
    if not _explicit(fields):
        return x
    var xs = xcnn_shape(x)
    var xp = xcnn_out_new(entries, _padded(fields, xs), context)   # the pad stores every word
    _ = py_call_typed(xcnn_entry(entries, "x_cnn_pad2d_m"), Python.list(x.h, xp.h), PythonObject(3),
                      _pad_params(layer, fields, xs, 0))
    return xp


def _conv_forward_t(layer: PythonObject, x: PythonObject, context: PythonObject) raises -> PythonObject:
    """`Conv2d._forward_t(x)`: x and the output resident."""
    var fields = _conv_admit(layer, context)
    var entries = xcnn_entries(fields, context)
    var xs = xcnn_shape(x)
    _check_x(fields, xs)
    if not _res_ok(fields, entries, x, context):
        return xcnn_wrap(entries, xcnn_conv2d_forward(layer, xcnn_host_of(x, context), context), context)
    var values = _weight_pair(fields, context)
    var oshape = _conv_out(fields, entries, xs)
    var out = xcnn_out_new(entries, oshape, context)   # the conv stores every word
    var xp = _pad_t(layer, fields, entries, x, context)
    var ps = xcnn_shape(xp)
    var prm = _conv_params(fields, ps)
    var groups = Int(py=fields["groups"])
    var conv = xcnn_entry(entries, "x_cnn_conv2d_forward_m")
    if groups == 1:
        _ = _res_w_values(layer, fields, entries, values, context)   # G3: pins on first use (a no-op with the switch off)
        var pw = _pinned(fields, entries, "weight_", xcnn_size(xcnn_shape(values[0])))
        var pb = _pinned(fields, entries, "bias_", xcnn_size(xcnn_shape(values[1])))
        if pw is not None and pb is not None:
            # the pinned weights are read where they live
            _ = py_call_typed(conv, Python.list(xp.h, pw, pb, out.h), PythonObject(15), prm)
        else:
            var w = values[0]
            var bias = values[1]
            _ = py_call_typed(conv, Python.list(xp.h, xcnn_addr(w), xcnn_addr(bias), out.h), PythonObject(9), prm)
            _ = bias  # keep the Python object alive through the native call above
            _ = w  # keep the Python object alive through the native call above
    else:
        # each group's channels sliced, convolved and placed on the device
        var og = Int(py=fields["out_channels"]) // groups
        var xg = xcnn_dev_get(layer, entries, "xg", xcnn_size(ps) // groups, context)
        var yg = xcnn_dev_get(layer, entries, "yg", xcnn_size(oshape) // groups, context)
        var copy = xcnn_entry(entries, "x_cnn_chan_copy_m")
        var inc = Int(py=fields["in_channels"])
        var outc = Int(py=fields["out_channels"])
        for g in range(groups):  # driver-loop: over convolution groups
            var wg = _host_rows(values[0], g * og, (g + 1) * og, context)
            var bg = _host_rows(values[1], g * og, (g + 1) * og, context)
            _ = py_call_typed(copy, Python.list(xp.h, xg), PythonObject(3), _group_prm(ps, inc, groups, g, 0))
            _ = py_call_typed(conv, Python.list(xg, xcnn_addr(wg), xcnn_addr(bg), yg), PythonObject(9), prm)
            _ = bg  # keep the Python object alive through the native call above
            _ = wg  # keep the Python object alive through the native call above
            _ = py_call_typed(copy, Python.list(yg, out.h), PythonObject(3), _group_prm(oshape, outc, groups, g, 1))
    _set(fields, "_x", x)
    _set(fields, "_xp", xp)
    return out


def _group_dev(fields: PythonObject, entries: PythonObject) raises -> Bool:
    """`Conv2d._group_dev(b)`: a grouped conv of host arrays on the `_m` forms."""
    return (Int(py=fields["groups"]) > 1 and xcnn_mixed(entries) and xcnn_idn(entries, XCNN_F_GROUP)
            and (not _explicit(fields) or xcnn_idn(entries, XCNN_F_PAD)))


def xcnn_conv2d_forward(layer: PythonObject, x: PythonObject, context: PythonObject) raises -> PythonObject:
    """`Conv2d.forward(x)`."""
    if xcnn_is_t(x, context):
        return _conv_forward_t(layer, x, context)
    var fields = _conv_admit(layer, context)
    var a = x
    var xs = xcnn_shape(a)
    _check_x(fields, xs)
    var entries = xcnn_entries(fields, context)
    if _group_dev(fields, entries):
        var out = xcnn_host_of(_conv_forward_t(layer, xcnn_wrap(entries, a, context), context), context)
        _set(fields, "_x", a)
        _set(fields, "_xp", PythonObject(None))
        return out
    var xp = a
    if _explicit(fields):
        xp = xcnn_host_f32(_padded(fields, xs), context)
        _ = py_call_typed(xcnn_entry(entries, "x_cnn_pad2d_forward"), xcnn_addr(a), xcnn_addr(xp),
                          _pad_params(layer, fields, xs, -1))
        _ = xp  # keep the Python object alive through the native call above
        _ = a  # keep the Python object alive through the native call above
    var values = _weight_pair(fields, context)
    var ps = xcnn_shape(xp)
    var prm = _conv_params(fields, ps)
    var oshape = _conv_out(fields, xcnn_entries(fields, context), xs)
    var out = xcnn_host_f32(oshape, context)
    var groups = Int(py=fields["groups"])
    var cg = Int(py=fields["in_channels"]) // groups
    var og = Int(py=fields["out_channels"]) // groups
    var conv = xcnn_entry(entries, "x_cnn_conv2d_forward")
    for g in range(groups):  # driver-loop: over convolution groups
        var xg = xp
        if groups > 1:
            xg = _channels(xp, g * cg, cg, context)
        var wg = _host_rows(values[0], g * og, (g + 1) * og, context)
        var bg = _host_rows(values[1], g * og, (g + 1) * og, context)
        var yg = out if groups == 1 else xcnn_host_f32([xs[0], og, oshape[2], oshape[3]], context)
        _ = py_call_typed(conv, xcnn_addr(xg), xcnn_addr(wg), xcnn_addr(bg), xcnn_addr(yg), prm)
        _ = yg  # keep the Python object alive through the native call above
        _ = bg  # keep the Python object alive through the native call above
        _ = wg  # keep the Python object alive through the native call above
        _ = xg  # keep the Python object alive through the native call above
        if groups > 1:
            _put_channels(out, g * og, og, yg)
    _set(fields, "_x", a)
    _set(fields, "_xp", xp)
    return out


def xcnn_conv_forward(layer: PythonObject, x: PythonObject, context: PythonObject) raises -> PythonObject:
    """`Conv2d.forward` / `Conv1d.forward` (Conv2d over (N, C, 1, L))."""
    var b = xcnn_builtins()
    if py_type(layer) is not context["xcnn_conv1d_class"]:
        var value = x if xcnn_is_t(x, context) else _snapshot(x, "x", context)
        return xcnn_conv2d_forward(layer, value, context)
    if xcnn_is_t(x, context):
        var s = xcnn_shape(x)
        if len(s) != 3:
            raise Error("mojolearn: Conv1d.forward takes (N, C, L)")
        var out = xcnn_conv2d_forward(layer, xcnn_reshape_t(x, [s[0], s[1], 1, s[2]], context), context)
        var o = xcnn_shape(out)
        return xcnn_reshape_t(out, [o[0], o[1], o[3]], context)
    var a = _snapshot(x, "x", context)
    var s = xcnn_shape(a)
    if len(s) != 3:
        raise Error("mojolearn: Conv1d.forward takes (N, C, L)")
    var out = xcnn_conv2d_forward(layer, xcnn_reshape(a, [s[0], s[1], 1, s[2]]), context)
    var o = xcnn_shape(out)
    return xcnn_reshape(out, [o[0], o[1], o[3]])


# ---------------------------------------------------------------- ReLU and add

def xcnn_relu_forward(layer: PythonObject, x: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_ReLU.forward(x)`."""
    var fields = _fields(layer)
    if xcnn_is_t(x, context):
        var b = xcnn_entries(fields, context)
        var xs = xcnn_shape(x)
        if xcnn_on_dev(b, x, context) and xcnn_idn(b, XCNN_F_LAYER_IO) and xcnn_size(xs) > 0:
            var out = xcnn_new(b, xs, False, context)
            _ = py_call_typed(xcnn_entry(b, "x_cnn_map2_m"), Python.list(x.h, x.h, out.h), PythonObject(7),
                              Python.list(0, xcnn_size(xs)))
            _set(fields, "_x", x)
            return out
    var a = xcnn_f32(x, "x", context)
    var xs = xcnn_shape(a)
    var out = xcnn_host_f32(xs, context)
    _ = py_call_typed(xcnn_entry(xcnn_entries(fields, context), "x_cnn_relu_forward"), xcnn_addr(a), xcnn_addr(out),
                      Python.list(xcnn_size(xs)))
    _ = out  # keep the Python object alive through the native call above
    _ = a  # keep the Python object alive through the native call above
    _set(fields, "_x", a)
    return out


def _add(entries: PythonObject, left: PythonObject, right: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_add(binding, a, b)`."""
    var a = xcnn_f32(left, "a", context)
    var c = xcnn_f32(right, "b", context)
    var s = xcnn_shape(a)
    var out = xcnn_host_f32(s, context)
    _ = py_call_typed(xcnn_entry(entries, "x_cnn_add"), xcnn_addr(a), xcnn_addr(c), xcnn_addr(out),
                      Python.list(xcnn_size(s)))
    _ = out  # keep the Python object alive through the native call above
    _ = c  # keep the Python object alive through the native call above
    _ = a  # keep the Python object alive through the native call above
    return out


# ---------------------------------------------------------------- BasicBlock lifecycle and backward

def _block_new(class_key: String, context: PythonObject) raises -> PythonObject:
    var b = xcnn_builtins()
    return py_call(b.object.__new__, context[PythonObject(class_key)])


def _block_conv(inc: PythonObject, outc: PythonObject, kernel: Int, stride: PythonObject,
                padding: Int, seed: PythonObject, mode: PythonObject, context: PythonObject) raises -> PythonObject:
    var layer = _block_new("xcnn_conv2d_class", context)
    _ = _conv_init(layer, Python.tuple(inc, outc, kernel, stride, padding, 1, 1, False, "zeros", seed,
                                      PythonObject(None), mode), False, context)
    return layer


def _block_bn(channels: PythonObject, mode: PythonObject, context: PythonObject) raises -> PythonObject:
    var layer = _block_new("xcnn_bn_class", context)
    _ = xcnn_bn_layers(layer, "bn_init", Python.tuple(channels, 1e-5, 0.1, True, True, PythonObject(None), mode), context)
    return layer


def _block_init(block: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    neural_plain_value(args, 0, Python.dict(admission_context=context))
    var fields = _fields(block)
    var inc = PythonObject(_dimension(args[0], "inplanes", context))
    var outc = PythonObject(_dimension(args[1], "planes", context))
    var stride = PythonObject(_dimension(args[2], "stride", context))
    var b = xcnn_builtins()
    var seed = PythonObject(0) if args[4] is None else py_call(b.int, args[4])
    var mode = args[6]
    _set(fields, "numeric_mode", mode)
    _set(fields, "input_shape", args[5])
    _set(fields, "conv1", _block_conv(inc, outc, 3, stride, 1, seed, mode, context))
    _set(fields, "bn1", _block_bn(outc, mode, context))
    var relu1 = _block_new("xcnn_relu_class", context)
    var relu2 = _block_new("xcnn_relu_class", context)
    _set(_fields(relu1), "numeric_mode", mode)
    _set(_fields(relu2), "numeric_mode", mode)
    _set(fields, "relu1", relu1)
    _set(fields, "conv2", _block_conv(inc=outc, outc=outc, kernel=3, stride=PythonObject(1), padding=1,
                                    seed=seed + PythonObject(1), mode=mode, context=context))
    _set(fields, "bn2", _block_bn(outc, mode, context))
    _set(fields, "relu2", relu2)
    var ds = args[3]
    var use_ds = (Int(py=stride) != 1 or Int(py=inc) != Int(py=outc)) if ds is None else Bool(py=py_call(b.bool, ds))
    var downsample = PythonObject(None)
    if use_ds:
        downsample = Python.list(_block_conv(inc, outc, 1, stride, 0, seed + PythonObject(2), mode, context),
                                 _block_bn(outc, mode, context))
    _set(fields, "downsample", downsample)
    _set(fields, "training", PythonObject(True))
    return PythonObject(None)


def _block_fields(block: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = _fields(block)
    for name in ["numeric_mode", "input_shape", "training"]:  # small-loop(block configuration)
        neural_plain_value(py_get(fields, PythonObject(name)), 0, Python.dict(admission_context=context))
    for name in ["conv1", "conv2", "bn1", "bn2", "relu1", "relu2"]:  # small-loop(fixed residual block layers)
        var expected = "xcnn_conv2d_class" if name.startswith("conv") else ("xcnn_bn_class" if name.startswith("bn") else "xcnn_relu_class")
        if py_type(fields[PythonObject(name)]) is not context[PythonObject(expected)]:
            raise Error("TypeError: BasicBlock requires native convolution, batch normalization and ReLU layers")
    var ds = fields["downsample"]
    if ds is not None:
        var b = xcnn_builtins()
        if (py_type(ds) is not b.list and py_type(ds) is not b.tuple) or len(ds) != 2:
            raise Error("TypeError: BasicBlock downsample requires a native convolution/batch-normalization pair")
        if py_type(ds[0]) is not context["xcnn_conv2d_class"] or py_type(ds[1]) is not context["xcnn_bn_class"]:
            raise Error("TypeError: BasicBlock downsample cannot invoke Python layer hooks")
    return fields


def _block_train(block: PythonObject, value: PythonObject, context: PythonObject) raises -> PythonObject:
    neural_plain_value(value, 0, Python.dict(admission_context=context))
    var fields = _block_fields(block, context)
    var mode = PythonObject(Bool(py=py_call(xcnn_builtins().bool, value)))
    _set(fields, "training", mode)
    _ = xcnn_bn_layers(fields["bn1"], "bn_train", Python.tuple(mode), context)
    _ = xcnn_bn_layers(fields["bn2"], "bn_train", Python.tuple(mode), context)
    if fields["downsample"] is not None:
        _ = xcnn_bn_layers(fields["downsample"][1], "bn_train", Python.tuple(mode), context)
    return block


def _relu_backward(layer: PythonObject, gradient: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = _fields(layer)
    var stored = py_get(fields, "_x")
    if stored is None:
        raise Error("RuntimeError: ReLU backward requires a prior forward")
    var entries = xcnn_entries(fields, context)
    if xcnn_on_dev(entries, stored, context) and xcnn_on_dev(entries, gradient, context):
        var shape = xcnn_shape(stored)
        if not _same_shape(shape, xcnn_shape(gradient)):
            raise Error("ReLU gradient shape differs from its input")
        var dx = xcnn_out_new(entries, shape, context)
        _ = py_call_typed(xcnn_entry(entries, "x_cnn_map2_m"), Python.list(stored.h, gradient.h, dx.h),
                          PythonObject(7), Python.list(1, xcnn_size(shape)))
        return dx
    var x = _snapshot(stored, "x", context)
    var g = _snapshot(gradient, "grad_out", context)
    var shape = xcnn_shape(x)
    if not _same_shape(shape, xcnn_shape(g)):
        raise Error("ReLU gradient shape differs from its input")
    var dx = xcnn_host_f32(shape, context)
    _ = py_call_typed(xcnn_entry(entries, "x_cnn_relu_backward"), xcnn_addr(x), xcnn_addr(g), xcnn_addr(dx),
                      Python.list(xcnn_size(shape)))
    _set(fields, "_x", x)
    _ = g
    return dx


def _block_conv_backward(block: PythonObject, conv: PythonObject, entries: PythonObject,
                         xh: PythonObject, shape: List[Int], gh: PythonObject, name: String,
                         context: PythonObject) raises -> PythonObject:
    var fields = _conv_admit(conv, context)
    var values = _weight_pair(fields, context)
    var w = values[0]
    var ws = xcnn_shape(w)
    var outc = Int(py=fields["out_channels"])
    var dxh = xcnn_dev_get(block, entries, name, xcnn_size(shape), context)
    var entry = xcnn_entry(entries, "x_cnn_conv2d_backward_m")
    if xcnn_resident(entries, context):
        var pin = _res_w_values(conv, fields, entries, values, context)[0]
        var gwh = xcnn_dev_get(conv, entries, "gw", xcnn_size(ws), context)
        var gbh = xcnn_dev_get(conv, entries, "gb", outc, context)
        _ = py_call_typed(entry, Python.list(xh, pin if pin is not None else xcnn_addr(w), gh, dxh, gwh, gbh),
                          PythonObject(63 if pin is not None else 61), _conv_params(fields, shape))
        var dev = xcnn_dev(conv, entries, context)
        _set(fields, "_gw_dev", Python.tuple(dev, gwh))
        _set(fields, "_gw_1d", PythonObject(False))
        _set(fields, "_gb_dev", Python.tuple(dev, gbh) if _truth(fields["bias"]) else PythonObject(None))
        if not _truth(fields["bias"]):
            _set(fields, "grad_bias_", _zeros([outc], context))
    else:
        var dw = xcnn_host_f32(ws, context)
        var db = xcnn_host_f32([outc], context)
        _ = py_call_typed(entry, Python.list(xh, xcnn_addr(w), gh, dxh, xcnn_addr(dw), xcnn_addr(db)),
                          PythonObject(13), _conv_params(fields, shape))
        _save_gradients(fields, dw, db, context)
    _ = w
    return dxh


def _block_backward_chain(block: PythonObject, fields: PythonObject, entries: PythonObject,
                          gradient: PythonObject, context: PythonObject) raises -> PythonObject:
    var chain = fields["_chain"]
    neural_string_keys(chain)
    # small-loop(fixed residual chain metadata)
    for name in ["x", "s1", "s2", "xh", "c1", "b1", "r1", "c2", "s", "d1", "sd"]:
        # small-loop(fixed residual chain metadata)
        neural_plain_value(chain[PythonObject(name)], 0, PythonObject(None))
    var shape = _as_list(chain["x"])
    var s1 = _as_list(chain["s1"])
    var s2 = _as_list(chain["s2"])
    var gt = xcnn_is_t(gradient, context)
    if gt and not xcnn_on_dev(entries, gradient, context):
        return xcnn_wrap(entries, _block_backward_chain(block, fields, entries, xcnn_host_of(gradient, context), context), context)
    var g = gradient if gt else _snapshot(gradient, "grad_out", context)
    if not _same_shape(xcnn_shape(g), s2):
        raise Error("mojolearn: BasicBlock grad_out shape differs from forward output")
    var n2 = xcnn_size(s2)
    var n1 = xcnn_size(s1)
    var gh = g.h if gt else xcnn_dev_upload(block, entries, "g", g, context)
    var gs = xcnn_dev_get(block, entries, "gs", n2, context)
    var map2 = xcnn_entry(entries, "x_cnn_map2_m")
    _ = py_call_typed(map2, Python.list(chain["s"], gh, gs), PythonObject(7), Python.list(1, n2))
    var gb2 = xcnn_dev_get(block, entries, "gb2", n2, context)
    xcnn_bn_backward_dev(fields["bn2"], entries, chain["c2"], gs, gb2, True, context)
    var gr1 = _block_conv_backward(block, fields["conv2"], entries, chain["r1"], s1, gb2, "gr1", context)
    var gc1 = xcnn_dev_get(block, entries, "gc1", n1, context)
    var gb1 = xcnn_dev_get(block, entries, "gb1", n1, context)
    _ = py_call_typed(map2, Python.list(chain["b1"], gr1, gb1), PythonObject(7), Python.list(1, n1))
    xcnn_bn_backward_dev(fields["bn1"], entries, chain["c1"], gb1, gc1, True, context)
    var gx1 = _block_conv_backward(block, fields["conv1"], entries, chain["xh"], shape, gc1, "gx1", context)
    var gi = gs
    var ds = fields["downsample"]
    if ds is not None:
        var gd = xcnn_dev_get(block, entries, "gd", n2, context)
        xcnn_bn_backward_dev(ds[1], entries, chain["d1"], gs, gd, True, context)
        gi = _block_conv_backward(block, ds[0], entries, chain["xh"], shape, gd, "gdx", context)
    var dx = xcnn_out_new(entries, shape, context) if gt else xcnn_host_f32(shape, context)
    _ = py_call_typed(map2, Python.list(gx1, gi, dx.h if gt else xcnn_addr(dx)),
                      PythonObject(7 if gt else 3), Python.list(2, xcnn_size(shape)))
    _ = g
    return dx


def _block_backward(block: PythonObject, gradient: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = _block_fields(block, context)
    var entries = xcnn_entries(fields, context)
    var dev = py_get(fields, "_dev")
    if py_get(fields, "_chain") is not None and dev is not None:
        if py_type(dev) is not context["xcnn_dev_class"]:
            raise Error("TypeError: BasicBlock requires native resident storage")
        if _fields(dev)["b"] is entries["module"]:
            return _block_backward_chain(block, fields, entries, gradient, context)
    if xcnn_is_t(gradient, context):
        return xcnn_wrap(entries, _block_backward(block, xcnn_host_of(gradient, context), context), context)
    var g = _relu_backward(fields["relu2"], gradient, context)
    var gm = xcnn_bn_backward(fields["bn2"], g, context)
    gm = _conv_backward(fields["conv2"], Python.tuple(gm, PythonObject(None)), context)
    gm = _relu_backward(fields["relu1"], gm, context)
    gm = xcnn_bn_backward(fields["bn1"], gm, context)
    gm = _conv_backward(fields["conv1"], Python.tuple(gm, PythonObject(None)), context)
    var gi = g
    var ds = fields["downsample"]
    if ds is not None:
        gi = _conv_backward(ds[0], Python.tuple(xcnn_bn_backward(ds[1], g, context), PythonObject(None)), context)
    return _add(entries, gm, gi, context)


def _block_parameters(block: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = _block_fields(block, context)
    var layers = Python.list(fields["conv1"], fields["bn1"], fields["conv2"], fields["bn2"])
    var ds = fields["downsample"]
    var b = xcnn_builtins()
    if ds is not None:
        _ = py_call(layers.append, ds[0])
        _ = py_call(layers.append, ds[1])
    var out = Python.list()
    for i in range(len(layers)):  # driver-loop: block parameter ownership descriptors
        var layer = layers[i]
        _ = py_call(out.append, Python.tuple(layer, "weight_", "grad_weight_"))
        if py_type(layer) is context["xcnn_bn_class"]:
            _ = py_call(out.append, Python.tuple(layer, "bias_", "grad_bias_"))
    return out


# ---------------------------------------------------------------- BasicBlock forward

def _block_convs(fields: PythonObject) raises -> PythonObject:
    var convs = Python.list(fields["conv1"], fields["conv2"])
    var ds = fields["downsample"]
    if ds is not None and _truth(ds):
        _ = py_call(py_call(xcnn_builtins().getattr, convs, "append"), ds[0])
    return convs


def _chain_ok(fields: PythonObject, entries: PythonObject, context: PythonObject) raises -> Bool:
    if not xcnn_mixed(entries):
        return False
    var convs = _block_convs(fields)
    for i in range(len(convs)):  # driver-loop: over the block's conv layers
        var f = _conv_admit(convs[i], context)
        if Int(py=f["groups"]) != 1 or _explicit(f):
            return False
    return True


def _conv_dev(block: PythonObject, entries: PythonObject, conv: PythonObject, xh: PythonObject, xshape: List[Int],
              name: String, context: PythonObject) raises -> PythonObject:
    """`BasicBlock._conv_dev`: conv of resident `xh` into the block's
    resident array `name`; (handle, out shape)."""
    var cf = _conv_admit(conv, context)
    var oshape = _conv_out(cf, xcnn_entries(cf, context), xshape)
    var yh = xcnn_dev_get(block, entries, name, xcnn_size(oshape), context)
    var values = _weight_pair(cf, context)
    var pins = _res_w_values(conv, cf, entries, values, context)   # G3: the auto-pinned weights (None with the switch off)
    var entry = xcnn_entry(entries, "x_cnn_conv2d_forward_m")
    if pins[0] is not None and pins[1] is not None:
        _ = py_call_typed(entry, Python.list(xh, pins[0], pins[1], yh), PythonObject(15), _conv_params(cf, xshape))
    else:
        var w = values[0]
        var bias = values[1]
        _ = py_call_typed(entry, Python.list(xh, xcnn_addr(w), xcnn_addr(bias), yh), PythonObject(9), _conv_params(cf, xshape))
        _ = bias  # keep the Python object alive through the native call above
        _ = w  # keep the Python object alive through the native call above
    return Python.tuple(yh, xcnn_tuple(oshape))


def _bn_dev(block: PythonObject, entries: PythonObject, bn: PythonObject, xh: PythonObject, shape: List[Int],
            name: String, context: PythonObject) raises -> PythonObject:
    """`BasicBlock._bn_dev`: BN of resident `xh` into the block's resident
    array `name`."""
    var bf = _bn_admit(bn, context)
    var n = shape[0]
    var c = shape[1]
    var hw = 1
    for i in range(2, len(shape)):  # driver-loop: over spatial shape words
        hw *= shape[i]
    if c != Int(py=bf["num_features"]):
        raise Error("mojolearn: input has " + String(c) + " channels, the layer " + String(Int(py=bf["num_features"])))
    var yh = xcnn_dev_get(block, entries, name, n * c * hw, context)
    _ = xcnn_bn_forward_dev(bn, bf, entries, xh, yh, n, c, hw, True, context)
    _set(bf, "_xdev", xh)
    _set(bf, "_x", PythonObject(None))
    _set(bf, "_shape", xcnn_tuple(shape))
    _set(bf, "_xdev_dev", xcnn_dev(block, entries, context))
    return yh


def _as_list(shape: PythonObject) raises -> List[Int]:
    neural_plain_value(shape, 0, PythonObject(None))
    var out = List[Int]()
    for i in range(len(shape)):  # driver-loop: over shape words
        out.append(Int(py=shape[i]))
    return out^


def _forward_chain(block: PythonObject, fields: PythonObject, entries: PythonObject, x: PythonObject,
                   context: PythonObject) raises -> PythonObject:
    """`BasicBlock._forward_chain(b, x)`: the whole block on resident arrays."""
    var dev = xcnn_dev(block, entries, context)
    var xs = xcnn_shape(x)
    if len(xs) != 4:
        raise Error("mojolearn: Conv2d.forward takes (N, C, H, W)")
    var conv1 = fields["conv1"]
    var inc = Int(py=_fields(conv1)["in_channels"])
    if xs[1] != inc:
        raise Error("mojolearn: input has " + String(xs[1]) + " channels, the layer " + String(inc))
    var xt = xcnn_is_t(x, context)
    var xh = x.h if xt else xcnn_dev_upload(block, entries, "x", x, context)
    var map2 = xcnn_entry(entries, "x_cnn_map2_m")
    var c1 = _conv_dev(block, entries, conv1, xh, xs, "c1", context)
    var s1 = _as_list(c1[1])
    var n1 = xcnn_size(s1)
    var r1 = xcnn_dev_get(block, entries, "r1", n1, context)
    var b1 = _bn_dev(block, entries, fields["bn1"], c1[0], s1, "b1", context)
    _ = py_call_typed(map2, Python.list(b1, b1, r1), PythonObject(7), Python.list(0, n1))
    var c2 = _conv_dev(block, entries, fields["conv2"], r1, s1, "c2", context)
    var s2 = _as_list(c2[1])
    var n2 = xcnn_size(s2)
    var b2 = _bn_dev(block, entries, fields["bn2"], c2[0], s2, "b2", context)
    var idh = xh
    var ds = fields["downsample"]
    var downsampled = ds is not None and _truth(ds)
    var d1 = PythonObject(None)
    var sd = PythonObject(None)
    if downsampled:
        var made = _conv_dev(block, entries, ds[0], xh, xs, "d1", context)
        d1 = made[0]
        sd = made[1]
        idh = _bn_dev(block, entries, ds[1], d1, _as_list(sd), "d2", context)
    elif xcnn_size(xs) != n2:
        raise Error("mojolearn: the identity's shape is not the block's output shape")
    var sh = xcnn_dev_get(block, entries, "s", n2, context)
    _ = py_call_typed(map2, Python.list(b2, idh, sh), PythonObject(7), Python.list(2, n2))
    var y: PythonObject
    if xt:
        y = xcnn_out_new(entries, s2, context)   # the ReLU stores every word
        _ = py_call_typed(map2, Python.list(sh, sh, y.h), PythonObject(7), Python.list(0, n2))
    else:
        y = xcnn_host_f32(s2, context)
        _ = py_call_typed(map2, Python.list(sh, sh, xcnn_addr(y)), PythonObject(3), Python.list(0, n2))
    var chain = Python.dict()
    py_setitem(chain, "x", xcnn_tuple(xs))
    py_setitem(chain, "xt", x if xt else PythonObject(None))
    py_setitem(chain, "s1", c1[1])
    py_setitem(chain, "s2", c2[1])
    py_setitem(chain, "xh", xh)
    py_setitem(chain, "c1", c1[0])
    py_setitem(chain, "b1", b1)
    py_setitem(chain, "r1", r1)
    py_setitem(chain, "c2", c2[0])
    py_setitem(chain, "s", sh)
    py_setitem(chain, "d1", d1 if downsampled else ds)
    py_setitem(chain, "sd", sd if downsampled else ds)
    _set(fields, "_chain", chain)
    var c1f = _fields(conv1)
    _set(c1f, "_x", x)
    _set(c1f, "_xp", x)
    return y


def xcnn_block_forward(block: PythonObject, x: PythonObject, context: PythonObject) raises -> PythonObject:
    """`BasicBlock.forward(x)`."""
    var fields = _block_fields(block, context)
    var entries = xcnn_entries(fields, context)
    if xcnn_is_t(x, context):
        if _chain_ok(fields, entries, context) and xcnn_on_dev(entries, x, context):
            return _forward_chain(block, fields, entries, x, context)
        return xcnn_wrap(entries, xcnn_block_forward(block, xcnn_host_of(x, context), context), context)
    var a = xcnn_f32(x, "x", context)
    if _chain_ok(fields, entries, context):
        return _forward_chain(block, fields, entries, a, context)
    _set(fields, "_chain", PythonObject(None))
    var out = xcnn_relu_forward(fields["relu1"], xcnn_bn_forward(fields["bn1"], xcnn_conv2d_forward(fields["conv1"], a, context), context), context)
    out = xcnn_bn_forward(fields["bn2"], xcnn_conv2d_forward(fields["conv2"], out, context), context)
    var identity = a
    var ds = fields["downsample"]
    if ds is not None and _truth(ds):
        identity = xcnn_bn_forward(ds[1], xcnn_conv2d_forward(ds[0], a, context), context)
    return xcnn_relu_forward(fields["relu2"], _add(xcnn_entries(fields, context), out, identity, context), context)


# ---------------------------------------------------------------- dispatch

def _flat_rows(out_buf: PythonObject, context: PythonObject) raises -> PythonObject:
    var s = xcnn_shape(out_buf)
    var rest = 1
    for i in range(1, len(s)):  # driver-loop: over shape words
        rest *= s[i]
    if xcnn_is_t(out_buf, context):
        return xcnn_reshape_t(out_buf, [s[0], rest], context)
    return xcnn_reshape(out_buf, [s[0], rest])


def _rows(layer: PythonObject, source: PythonObject, required: Bool, context: PythonObject) raises -> PythonObject:
    var fields = _fields(layer)
    var a = xcnn_f32(source, "X", context)
    var s = xcnn_shape(a)
    var declared = py_get(fields, "input_shape")
    if len(s) == 2:
        if declared is None:
            if required:
                raise Error("mojolearn: row input needs input_shape=(C, H, W)")
            return a
        var dims = List[Int]()
        dims.append(s[0])
        neural_plain_value(declared, 0, PythonObject(None))
        var given = declared
        for i in range(len(given)):  # driver-loop: over the declared sample shape words
            dims.append(Int(py=given[i]))
        return xcnn_reshape(a, dims)
    return a


def xcnn_conv_layers(layer: PythonObject, operation: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    if operation == "block_init":
        return _block_init(layer, args, context)
    if operation == "block_train":
        return _block_train(layer, args[0], context)
    if operation == "block_backward":
        return _block_backward(layer, args[0], context)
    if operation == "block_parameters":
        return _block_parameters(layer, context)
    if operation == "block_fit":
        _ = _block_fields(layer, context)
        return layer
    if operation == "conv_init" or operation == "conv_init1d":
        return _conv_init(layer, args, operation == "conv_init1d", context)
    var fields = _conv_admit(layer, context)
    if operation == "conv_backward":
        return _conv_backward(layer, args, context)
    if operation == "conv_set_weights" or operation == "conv_set_weights1d":
        return _conv_set_weights(layer, args, operation == "conv_set_weights1d", context)
    if operation == "conv_pin":
        _pin_weights(layer, fields, xcnn_entries(fields, context), context)
        return layer
    if operation == "conv_unpin":
        _set(fields, "_pins", PythonObject(None))
        return layer
    if operation == "conv_grad_weight" or operation == "conv_grad_bias":
        return _conv_gradient(layer, operation == "conv_grad_weight", context)
    if operation == "conv_grad_weight_set" or operation == "conv_grad_bias_set":
        var weight = operation == "conv_grad_weight_set"
        _set(fields, "grad_weight_" if weight else "grad_bias_", args[0])
        _set(fields, "_gw_dev" if weight else "_gb_dev", PythonObject(None))
        if weight:
            _set(fields, "_gw_1d", PythonObject(False))
        return PythonObject(None)
    if operation == "conv_weight1d":
        var w = xcnn_f32(fields["weight_"], "weight_", context)
        var s = xcnn_shape(w)
        return xcnn_reshape(w, [s[0], s[1], s[3]])
    if operation == "conv_fit":
        var a = _rows(layer, args[0], True, context)
        var shape = xcnn_shape(a)
        var size = 1
        for i in range(1, len(shape)):  # driver-loop: sample shape metadata
            size *= shape[i]
        _set(fields, "n_features_in_", PythonObject(size))
        return layer
    if operation == "conv_forward":
        return xcnn_conv_forward(layer, args[0], context)
    if operation == "conv_transform":
        return _flat_rows(xcnn_conv_forward(layer, _rows(layer, args[0], True, context), context), context)
    if operation == "block_forward":
        return xcnn_block_forward(layer, args[0], context)
    if operation == "block_transform":
        return _flat_rows(xcnn_block_forward(layer, _rows(layer, args[0], False, context), context), context)
    raise Error("RuntimeError: unknown CNN layer operation " + operation)
