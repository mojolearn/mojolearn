# SPDX-License-Identifier: Apache-2.0
"""Native `forward` and `transform` of Conv2d / Conv1d and BasicBlock, and
the internal ReLU forward and add they compose
(python/mojolearn/_expansion_cnn.py).

The routes are the Python bodies': host arrays through the host entries
(an explicit pad first, one conv per channel group), grouped convs on the
resident `_m` forms when the binding has them on, resident tensors through
the `_m` entries with the G3 auto-pinned weights, and BasicBlock's resident
chain (conv, BN, ReLU, conv, BN, the identity or downsample, the add and
the ReLU on resident arrays). The same entries, operands and residency
bits, so no bit moves; the fields the Python `backward` methods read are
written under the same names.
"""
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_fields, py_contains
from bindings.py_typed_call import py_call_typed
from bindings.xcnn_common import (
    XCNN_F_GROUP, XCNN_F_PAD, XCNN_F_PIN, XCNN_F_LAYER_IO, xcnn_builtins, xcnn_entries, xcnn_entry, xcnn_mixed,
    xcnn_idn, xcnn_resident, xcnn_shape, xcnn_size, xcnn_tuple, xcnn_is_t, xcnn_on_dev, xcnn_new, xcnn_out_new,
    xcnn_reshape_t, xcnn_wrap, xcnn_host_of, xcnn_host_f32, xcnn_f32, xcnn_reshape, xcnn_addr, xcnn_dev, xcnn_dev_get,
    xcnn_dev_upload,
)
from bindings.xcnn_bn_native import xcnn_bn_forward_dev, xcnn_bn_forward
from bindings.xseq_native_common import xseq_f32
from bindings.sequential_inputs import sequential_input


def _set(fields: PythonObject, name: String, value: PythonObject) raises:
    py_setitem(fields, PythonObject(name), value)


def _pair(fields: PythonObject, name: String) raises -> List[Int]:
    var v = fields[PythonObject(name)]
    return [Int(py=v[0]), Int(py=v[1])]


def _truth(value: PythonObject) raises -> Bool:
    return Bool(py=py_call(xcnn_builtins().bool, value))


def _slice(start: Int, stop: Int) raises -> PythonObject:
    var b = xcnn_builtins()
    return py_call(b.slice, PythonObject(start), PythonObject(stop))


def _all() raises -> PythonObject:
    var b = xcnn_builtins()
    return py_call(b.slice, PythonObject(None))


def _getitem(value: PythonObject, key: PythonObject) raises -> PythonObject:
    """`value[key]` (ndarray's own C subscript)."""
    var b = xcnn_builtins()
    return py_call(py_call(b.getattr, value, "__getitem__"), key)


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
    var modes = py_call(b.getattr, py_type(layer), "_PAD_MODES")
    var out = Python.list(shape[0], shape[1], shape[2], shape[3], p[0], p[1], p[2], p[3], modes[fields["padding_mode"]])
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
    var a = sequential_input(source, context)
    return xseq_f32(_getitem(a, _slice(start, stop)), "weight_", False, context)


# ---------------------------------------------------------------- pinned weights (G3)

def _pin_weights(layer: PythonObject, fields: PythonObject, entries: PythonObject, context: PythonObject) raises:
    """`_pin_weights(layer, ("weight_", "bias_"))`: both uploaded once."""
    var b = xcnn_builtins()
    _set(fields, "_pins", PythonObject(None))
    if not xcnn_idn(entries, XCNN_F_PIN):
        return
    var handles = Python.dict()
    var names = Python.tuple("weight_", "bias_")
    for i in range(2):  # driver-loop: over the two named weight arrays
        var arr = py_get(fields, names[i])
        if arr is None:
            continue
        var a = xseq_f32(sequential_input(arr, context), String(py=names[i]), False, context)
        var h = xcnn_dev_upload(layer, entries, "pin_" + String(py=names[i]), a, context)
        py_setitem(handles, names[i], Python.tuple(h, Int(py=a.size)))
    _set(fields, "_pins", Python.tuple(entries["module"], handles))
    # the arrays pinned, so a reassigned `weight_` / `bias_` is uploaded again
    _set(fields, "_pin_src", Python.tuple(py_get(fields, "weight_"), py_get(fields, "bias_")))


def _auto_pin(layer: PythonObject, fields: PythonObject, entries: PythonObject, context: PythonObject) raises -> Bool:
    """`_auto_pin(layer, b, ("weight_", "bias_"))`."""
    if not (xcnn_resident(entries, context) and xcnn_idn(entries, XCNN_F_PIN)):
        return False
    var p = py_get(fields, "_pins")
    var src = py_get(fields, "_pin_src")
    if p is not None and _truth(p) and p[0] is entries["module"] and src is not None and len(src) == 2:
        if src[0] is py_get(fields, "weight_") and src[1] is py_get(fields, "bias_"):
            return True
    _pin_weights(layer, fields, entries, context)
    return _truth(py_get(fields, "_pins"))


def _pinned(fields: PythonObject, entries: PythonObject, name: String, size: Int) raises -> PythonObject:
    """`_pinned(layer, b, name, size)`: the handle, or None."""
    var p = py_get(fields, "_pins")
    if p is None or not _truth(p) or p[0] is not entries["module"]:
        return PythonObject(None)
    var hit = py_get(p[1], PythonObject(name))
    if hit is None or Int(py=hit[1]) != size:
        return PythonObject(None)
    return hit[0]


def _res_w(layer: PythonObject, fields: PythonObject, entries: PythonObject, context: PythonObject) raises -> PythonObject:
    """`Conv2d._res_w(b)`: (weight handle, bias handle) or (None, None)."""
    if Int(py=fields["groups"]) != 1 or not _auto_pin(layer, fields, entries, context):
        return Python.tuple(PythonObject(None), PythonObject(None))
    return Python.tuple(_pinned(fields, entries, "weight_", Int(py=fields["weight_"].size)),
                        _pinned(fields, entries, "bias_", Int(py=fields["bias_"].size)))


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
    var fields = py_fields(layer)
    var entries = xcnn_entries(fields, context)
    var xs = xcnn_shape(x)
    _check_x(fields, xs)
    if not _res_ok(fields, entries, x, context):
        return xcnn_wrap(entries, xcnn_conv2d_forward(layer, xcnn_host_of(x, context), context), context)
    var oshape = _conv_out(fields, entries, xs)
    var out = xcnn_out_new(entries, oshape, context)   # the conv stores every word
    var xp = _pad_t(layer, fields, entries, x, context)
    var ps = xcnn_shape(xp)
    var prm = _conv_params(fields, ps)
    var groups = Int(py=fields["groups"])
    var conv = xcnn_entry(entries, "x_cnn_conv2d_forward_m")
    if groups == 1:
        _ = _res_w(layer, fields, entries, context)   # G3: pins on first use (a no-op with the switch off)
        var pw = _pinned(fields, entries, "weight_", Int(py=fields["weight_"].size))
        var pb = _pinned(fields, entries, "bias_", Int(py=fields["bias_"].size))
        if pw is not None and pb is not None:
            # the pinned weights are read where they live
            _ = py_call_typed(conv, Python.list(xp.h, pw, pb, out.h), PythonObject(15), prm)
        else:
            var w = xseq_f32(sequential_input(fields["weight_"], context), "weight_", False, context)
            var bias = xseq_f32(sequential_input(fields["bias_"], context), "bias_", False, context)
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
            var wg = _host_rows(fields["weight_"], g * og, (g + 1) * og, context)
            var bg = _host_rows(fields["bias_"], g * og, (g + 1) * og, context)
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
    var fields = py_fields(layer)
    var a = xcnn_f32(x, "x", context)
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
            xg = xseq_f32(_getitem(xp, Python.tuple(_all(), _slice(g * cg, (g + 1) * cg))), "x", False, context)
        var wg = _host_rows(fields["weight_"], g * og, (g + 1) * og, context)
        var bg = _host_rows(fields["bias_"], g * og, (g + 1) * og, context)
        var yg = out if groups == 1 else xcnn_host_f32([xs[0], og, oshape[2], oshape[3]], context)
        _ = py_call_typed(conv, xcnn_addr(xg), xcnn_addr(wg), xcnn_addr(bg), xcnn_addr(yg), prm)
        _ = yg  # keep the Python object alive through the native call above
        _ = bg  # keep the Python object alive through the native call above
        _ = wg  # keep the Python object alive through the native call above
        _ = xg  # keep the Python object alive through the native call above
        if groups > 1:
            py_setitem(out, Python.tuple(_all(), _slice(g * og, (g + 1) * og)), yg)
    _set(fields, "_x", a)
    _set(fields, "_xp", xp)
    return out


def xcnn_conv_forward(layer: PythonObject, x: PythonObject, context: PythonObject) raises -> PythonObject:
    """`Conv2d.forward` / `Conv1d.forward` (Conv2d over (N, C, 1, L))."""
    var b = xcnn_builtins()
    if not Bool(py=py_call(b.isinstance, layer, context["xcnn_conv1d_class"])):
        return xcnn_conv2d_forward(layer, x, context)
    if xcnn_is_t(x, context):
        var s = xcnn_shape(x)
        if len(s) != 3:
            raise Error("mojolearn: Conv1d.forward takes (N, C, L)")
        var out = xcnn_conv2d_forward(layer, xcnn_reshape_t(x, [s[0], s[1], 1, s[2]], context), context)
        var o = xcnn_shape(out)
        return xcnn_reshape_t(out, [o[0], o[1], o[3]], context)
    var a = xcnn_f32(x, "x", context)
    var s = xcnn_shape(a)
    if len(s) != 3:
        raise Error("mojolearn: Conv1d.forward takes (N, C, L)")
    var out = xcnn_conv2d_forward(layer, xcnn_reshape(a, [s[0], s[1], 1, s[2]]), context)
    var o = xcnn_shape(out)
    return xcnn_reshape(out, [o[0], o[1], o[3]])


# ---------------------------------------------------------------- ReLU and add

def xcnn_relu_forward(layer: PythonObject, x: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_ReLU.forward(x)`."""
    var fields = py_fields(layer)
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


# ---------------------------------------------------------------- BasicBlock

def _block_convs(fields: PythonObject) raises -> PythonObject:
    var convs = Python.list(fields["conv1"], fields["conv2"])
    var ds = fields["downsample"]
    if ds is not None and _truth(ds):
        _ = py_call(py_call(xcnn_builtins().getattr, convs, "append"), ds[0])
    return convs


def _chain_ok(fields: PythonObject, entries: PythonObject) raises -> Bool:
    if not xcnn_mixed(entries):
        return False
    var convs = _block_convs(fields)
    for i in range(len(convs)):  # driver-loop: over the block's conv layers
        var f = py_fields(convs[i])
        if Int(py=f["groups"]) != 1 or _explicit(f):
            return False
    return True


def _conv_dev(block: PythonObject, entries: PythonObject, conv: PythonObject, xh: PythonObject, xshape: List[Int],
              name: String, context: PythonObject) raises -> PythonObject:
    """`BasicBlock._conv_dev`: conv of resident `xh` into the block's
    resident array `name`; (handle, out shape)."""
    var cf = py_fields(conv)
    var oshape = _conv_out(cf, xcnn_entries(cf, context), xshape)
    var yh = xcnn_dev_get(block, entries, name, xcnn_size(oshape), context)
    var pins = _res_w(conv, cf, entries, context)   # G3: the auto-pinned weights (None with the switch off)
    var entry = xcnn_entry(entries, "x_cnn_conv2d_forward_m")
    if pins[0] is not None and pins[1] is not None:
        _ = py_call_typed(entry, Python.list(xh, pins[0], pins[1], yh), PythonObject(15), _conv_params(cf, xshape))
    else:
        var w = xseq_f32(sequential_input(cf["weight_"], context), "weight_", False, context)
        var bias = xseq_f32(sequential_input(cf["bias_"], context), "bias_", False, context)
        _ = py_call_typed(entry, Python.list(xh, xcnn_addr(w), xcnn_addr(bias), yh), PythonObject(9), _conv_params(cf, xshape))
        _ = bias  # keep the Python object alive through the native call above
        _ = w  # keep the Python object alive through the native call above
    return Python.tuple(yh, xcnn_tuple(oshape))


def _bn_dev(block: PythonObject, entries: PythonObject, bn: PythonObject, xh: PythonObject, shape: List[Int],
            name: String, context: PythonObject) raises -> PythonObject:
    """`BasicBlock._bn_dev`: BN of resident `xh` into the block's resident
    array `name`."""
    var bf = py_fields(bn)
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
    var inc = Int(py=py_fields(conv1)["in_channels"])
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
    var c1f = py_fields(conv1)
    _set(c1f, "_x", x)
    _set(c1f, "_xp", x)
    return y


def xcnn_block_forward(block: PythonObject, x: PythonObject, context: PythonObject) raises -> PythonObject:
    """`BasicBlock.forward(x)`."""
    var fields = py_fields(block)
    var entries = xcnn_entries(fields, context)
    if xcnn_is_t(x, context):
        if _chain_ok(fields, entries) and xcnn_on_dev(entries, x, context):
            return _forward_chain(block, fields, entries, x, context)
        return xcnn_wrap(entries, xcnn_block_forward(block, xcnn_host_of(x, context), context), context)
    var a = xcnn_f32(x, "x", context)
    if _chain_ok(fields, entries):
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
    var fields = py_fields(layer)
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
        var given = py_call(xcnn_builtins().tuple, declared)
        for i in range(len(given)):  # driver-loop: over the declared sample shape words
            dims.append(Int(py=given[i]))
        return xcnn_reshape(a, dims)
    return a


def xcnn_conv_layers(layer: PythonObject, operation: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    if operation == "conv_forward":
        return xcnn_conv_forward(layer, args[0], context)
    if operation == "conv_transform":
        return _flat_rows(xcnn_conv_forward(layer, _rows(layer, args[0], True, context), context), context)
    if operation == "block_forward":
        return xcnn_block_forward(layer, args[0], context)
    if operation == "block_transform":
        return _flat_rows(xcnn_block_forward(layer, _rows(layer, args[0], False, context), context), context)
    raise Error("RuntimeError: unknown CNN layer operation " + operation)
