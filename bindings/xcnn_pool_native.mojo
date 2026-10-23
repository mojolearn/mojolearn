# SPDX-License-Identifier: Apache-2.0
"""Native public operations of the CNN pooling and channel-dropout layers
(python/mojolearn/_expansion_cnn.py): MaxPool2d, AvgPool2d, MaxPool1d,
AvgPool1d, AdaptiveAvgPool2d, AdaptiveMaxPool2d and Dropout2d `forward`
and `transform`, and the convolutions' `fit` (the row-input shape record).

Every route the Python bodies took is taken here, by the same rule: a host
array runs the host entry (or, on a mixed binding, the `_m` entry with the
winners or mask kept resident), a resident `DeviceTensor` runs the `_m`
entry with its residency bits, and a tensor of another binding is read back
and wrapped. The same entries, argument words and residency bits, so no bit
moves. The layer state the Python `backward` methods read (`_xshape`,
`_idx_dev`, `_idx_shape`, `_mask_dev`, `_layer`, `_dev`) is written under
the same names.
"""
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_fields
from bindings.py_typed_call import py_call_typed
from bindings.xcnn_common import (
    XCNN_F_ADAPT, XCNN_F_LAYER_IO, xcnn_builtins, xcnn_entries, xcnn_entry, xcnn_mixed, xcnn_idn, xcnn_resident, xcnn_shape,
    xcnn_size, xcnn_tuple, xcnn_is_t, xcnn_on_dev, xcnn_new, xcnn_out_new, xcnn_reshape_t, xcnn_wrap,
    xcnn_host_of, xcnn_host_f32, xcnn_host_i32, xcnn_f32, xcnn_reshape, xcnn_addr, xcnn_dev, xcnn_dev_get,
)


def _int(value: PythonObject) raises -> Int:
    return Int(py=value)


def _pair(fields: PythonObject, name: String) raises -> List[Int]:
    var v = fields[PythonObject(name)]
    return [Int(py=v[0]), Int(py=v[1])]


def _set(fields: PythonObject, name: String, value: PythonObject) raises:
    py_setitem(fields, PythonObject(name), value)


def _is(model: PythonObject, key: String, context: PythonObject) raises -> Bool:
    """`isinstance(model, <connected layer class>)`."""
    var b = xcnn_builtins()
    return Bool(py=py_call(b.isinstance, model, context[PythonObject(key)]))


def _class_name(model: PythonObject) raises -> String:
    return String(py=py_type(model).__name__)


# ---------------------------------------------------------------- 2-D pools

def _pool_params(fields: PythonObject, shape: List[Int]) raises -> PythonObject:
    """`_Pool2d._params(shape)`."""
    var k = _pair(fields, "kernel_size")
    var s = _pair(fields, "stride")
    var p = _pair(fields, "padding")
    var d = _pair(fields, "dilation")
    var divisor = py_get(fields, "divisor_override")
    var dv = 0
    if divisor is not None and Bool(py=py_call(xcnn_builtins().bool, divisor)):
        dv = Int(py=divisor)
    return Python.list(shape[0], shape[1], shape[2], shape[3], k[0], k[1], s[0], s[1], p[0], p[1], d[0], d[1], 0, 0,
                       1 if Bool(py=fields["count_include_pad"]) else 0, 0, 1 if Bool(py=fields["ceil_mode"]) else 0, dv)


def _pool_out(fields: PythonObject, entries: PythonObject, shape: List[Int]) raises -> List[Int]:
    """`_Pool2d._out_shape(shape)` (`x_cnn_pool_shape`)."""
    var oh_ow = py_call_typed(xcnn_entry(entries, "x_cnn_pool_shape"), _pool_params(fields, shape))
    return [shape[0], shape[1], Int(py=oh_ow[0]), Int(py=oh_ow[1])]


def xcnn_maxpool_forward(model: PythonObject, x: PythonObject, context: PythonObject) raises -> PythonObject:
    """`MaxPool2d.forward(x)`."""
    var fields = py_fields(model)
    if xcnn_is_t(x, context):
        # x, the output and the winners resident
        var entries = xcnn_entries(fields, context)
        var xs = xcnn_shape(x)
        if len(xs) != 4:
            raise Error("mojolearn: MaxPool2d.forward takes (N, C, H, W)")
        if not xcnn_on_dev(entries, x, context):
            return xcnn_wrap(entries, xcnn_maxpool_forward(model, xcnn_host_of(x, context), context), context)
        var shape = _pool_out(fields, entries, xs)
        var out = xcnn_out_new(entries, shape, context)   # the pool stores every word
        var h = xcnn_dev_get(model, entries, "idx", xcnn_size(shape), context)
        _ = py_call_typed(xcnn_entry(entries, "x_cnn_maxpool2d_forward_m"), Python.list(x.h, out.h, h), PythonObject(7),
                          _pool_params(fields, xs))
        _set(fields, "_xshape", xcnn_tuple(xs))
        _set(fields, "_idx_host", PythonObject(None))
        _set(fields, "_idx_dev", h)
        _set(fields, "_idx_shape", xcnn_tuple(shape))
        return out
    var a = xcnn_f32(x, "x", context)
    var xs = xcnn_shape(a)
    if len(xs) != 4:
        raise Error("mojolearn: MaxPool2d.forward takes (N, C, H, W)")
    var entries = xcnn_entries(fields, context)
    var shape = _pool_out(fields, entries, xs)
    var out = xcnn_host_f32(shape, context)
    entries = xcnn_entries(fields, context)
    if xcnn_mixed(entries):
        var h = xcnn_dev_get(model, entries, "idx", xcnn_size(shape), context)
        _ = py_call_typed(xcnn_entry(entries, "x_cnn_maxpool2d_forward_m"), Python.list(xcnn_addr(a), xcnn_addr(out), h),
                          PythonObject(4), _pool_params(fields, xs))
        _set(fields, "_xshape", xcnn_tuple(xs))
        _set(fields, "_idx_host", PythonObject(None))
        _set(fields, "_idx_dev", h)
        _set(fields, "_idx_shape", xcnn_tuple(shape))
        return out
    var idx = xcnn_host_i32(shape, context)
    _ = py_call_typed(xcnn_entry(entries, "x_cnn_maxpool2d_forward"), xcnn_addr(a), xcnn_addr(out), xcnn_addr(idx),
                      _pool_params(fields, xs))
    _set(fields, "_xshape", xcnn_tuple(xs))
    # the `indices_` setter: the host winners, no resident copy
    _set(fields, "_idx_host", idx)
    _set(fields, "_idx_dev", PythonObject(None))
    return out


def xcnn_avgpool_forward(model: PythonObject, x: PythonObject, context: PythonObject) raises -> PythonObject:
    """`AvgPool2d.forward(x)`."""
    var fields = py_fields(model)
    if xcnn_is_t(x, context):
        # x and the output resident
        var entries = xcnn_entries(fields, context)
        var xs = xcnn_shape(x)
        if len(xs) != 4:
            raise Error("mojolearn: AvgPool2d.forward takes (N, C, H, W)")
        if not xcnn_on_dev(entries, x, context):
            return xcnn_wrap(entries, xcnn_avgpool_forward(model, xcnn_host_of(x, context), context), context)
        var out = xcnn_out_new(entries, _pool_out(fields, entries, xs), context)   # the pool stores every word
        _ = py_call_typed(xcnn_entry(entries, "x_cnn_avgpool2d_forward_m"), Python.list(x.h, out.h), PythonObject(3),
                          _pool_params(fields, xs))
        _set(fields, "_xshape", xcnn_tuple(xs))
        return out
    var a = xcnn_f32(x, "x", context)
    var xs = xcnn_shape(a)
    if len(xs) != 4:
        raise Error("mojolearn: AvgPool2d.forward takes (N, C, H, W)")
    var entries = xcnn_entries(fields, context)
    var out = xcnn_host_f32(_pool_out(fields, entries, xs), context)
    entries = xcnn_entries(fields, context)
    _ = py_call_typed(xcnn_entry(entries, "x_cnn_avgpool2d_forward"), xcnn_addr(a), xcnn_addr(out), _pool_params(fields, xs))
    _set(fields, "_xshape", xcnn_tuple(xs))
    return out


def _pool2d_forward(model: PythonObject, x: PythonObject, context: PythonObject) raises -> PythonObject:
    if _is(model, "xcnn_maxpool_class", context):
        return xcnn_maxpool_forward(model, x, context)
    return xcnn_avgpool_forward(model, x, context)


def xcnn_pool_forward(model: PythonObject, x: PythonObject, context: PythonObject) raises -> PythonObject:
    """A pool's `forward`: the 1-D pools run the 2-D pool over (N, C, 1, L)."""
    if not _is(model, "xcnn_pool1d_class", context):
        return _pool2d_forward(model, x, context)
    var name = _class_name(model)
    if xcnn_is_t(x, context):
        var s = xcnn_shape(x)
        if len(s) != 3:
            raise Error("mojolearn: " + name + ".forward takes (N, C, L)")
        var out = _pool2d_forward(model, xcnn_reshape_t(x, [s[0], s[1], 1, s[2]], context), context)
        var o = xcnn_shape(out)
        if xcnn_is_t(out, context):
            return xcnn_reshape_t(out, [o[0], o[1], o[3]], context)
        return xcnn_reshape(out, [o[0], o[1], o[3]])
    var a = xcnn_f32(x, "x", context)
    var s = xcnn_shape(a)
    if len(s) != 3:
        raise Error("mojolearn: " + name + ".forward takes (N, C, L)")
    var out = _pool2d_forward(model, xcnn_reshape(a, [s[0], s[1], 1, s[2]]), context)
    var o = xcnn_shape(out)
    return xcnn_reshape(out, [o[0], o[1], o[3]])


# ---------------------------------------------------------------- row inputs

def xcnn_rows(model: PythonObject, source: PythonObject, required: Bool, context: PythonObject) raises -> PythonObject:
    """`Conv2d._rows(X)` (`required`: a row input without `input_shape` is
    refused) and `BatchNorm2d._rows(X)` (it passes through)."""
    var fields = py_fields(model)
    var a = xcnn_f32(source, "X", context)
    var s = xcnn_shape(a)
    if len(s) == 2:
        var shape = py_get(fields, "input_shape")
        if shape is None:
            if required:
                raise Error("mojolearn: row input needs input_shape=(C, H, W)")
            return a
        var dims = List[Int]()
        dims.append(s[0])
        var b = xcnn_builtins()
        var given = py_call(b.tuple, shape)
        for i in range(len(given)):  # driver-loop: over the declared sample shape words
            dims.append(Int(py=given[i]))
        return xcnn_reshape(a, dims)
    return a


def _flat_rows(out_buf: PythonObject, context: PythonObject) raises -> PythonObject:
    """`out.reshape(out.shape[0], -1)`."""
    var s = xcnn_shape(out_buf)
    var rest = 1
    for i in range(1, len(s)):  # driver-loop: over shape words
        rest *= s[i]
    if xcnn_is_t(out_buf, context):
        return xcnn_reshape_t(out_buf, [s[0], rest], context)
    return xcnn_reshape(out_buf, [s[0], rest])


def xcnn_conv_fit(model: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    """`Conv2d.fit(X)`: the row input's sample size, recorded."""
    var rows = xcnn_rows(model, source, True, context)
    var s = xcnn_shape(rows)
    var n = 1
    for i in range(1, len(s)):  # driver-loop: over shape words
        n *= s[i]
    _set(py_fields(model), "n_features_in_", PythonObject(n))
    return model


# ---------------------------------------------------------------- Dropout2d

def _dropout_words(fields: PythonObject, n: Int, c: Int, hw: Int) raises -> PythonObject:
    """The mask words of one call (the threshold of p and the seed of
    (random_state, the call count)); advances `calls_`."""
    var b = xcnn_builtins()
    var p = Float64(py=fields["p"])
    var scaled = py_call(b.round, PythonObject(p * 4294967296.0))
    var thresh = min(Int(py=scaled), 4294967296)
    var state = fields["random_state"]
    var calls = fields["calls_"]
    var seed_lo = state.__and__(PythonObject(2147483647))
    var seed_hi = ((state.__rshift__(PythonObject(31))) * PythonObject(1000003) + calls).__and__(PythonObject(2147483647))
    _set(fields, "calls_", calls + PythonObject(1))
    return Python.list(n, c, hw, seed_lo, seed_hi, thresh >> 16, thresh & 0xFFFF)


def xcnn_dropout_forward(model: PythonObject, x: PythonObject, context: PythonObject) raises -> PythonObject:
    """`Dropout2d.forward(x)`."""
    var fields = py_fields(model)
    var training = Bool(py=fields["training"])
    if xcnn_is_t(x, context):
        var bt = xcnn_entries(fields, context)
        var xs = xcnn_shape(x)
        if xcnn_on_dev(bt, x, context) and xcnn_idn(bt, XCNN_F_LAYER_IO) and xcnn_size(xs) > 0:
            # x, y and the mask resident; eval mode is the same resident words
            if not training:
                _set(fields, "_mask_host", PythonObject(None))
                _set(fields, "_mask_dev", PythonObject(None))
                return xcnn_reshape_t(x, xs, context)
            if len(xs) != 3 and len(xs) != 4:
                raise Error("mojolearn: Dropout2d.forward takes (N, C, H, W) or (C, H, W)")
            var n = 1 if len(xs) == 3 else xs[0]
            var c = xs[0] if len(xs) == 3 else xs[1]
            var hw = xs[len(xs) - 2] * xs[len(xs) - 1]
            var prm = _dropout_words(fields, n, c, hw)
            var y = xcnn_new(bt, xs, True, context)   # every path stores each mask and y word
            var h = xcnn_dev_get(model, bt, "mask", xcnn_size(xs), context)
            _ = py_call_typed(xcnn_entry(bt, "x_cnn_dropout2d_m"), Python.list(x.h, y.h, h), PythonObject(7), prm, fields["p"])
            _set(fields, "_mask_host", PythonObject(None))
            _set(fields, "_mask_dev", h)
            _set(fields, "_mask_shape", xcnn_tuple(xs))
            return y
    var a = xcnn_f32(x, "x", context)
    var b = xcnn_builtins()
    if not training:
        _set(fields, "_mask_host", PythonObject(None))
        _set(fields, "_mask_dev", PythonObject(None))
        return py_call(py_call(b.getattr, a, "copy"))
    var xs = xcnn_shape(a)
    if len(xs) != 3 and len(xs) != 4:
        raise Error("mojolearn: Dropout2d.forward takes (N, C, H, W) or (C, H, W)")
    var s4 = List[Int]()
    if len(xs) == 3:
        s4.append(1)
    for i in range(len(xs)):  # driver-loop: over shape words
        s4.append(xs[i])
    var prm = _dropout_words(fields, s4[0], s4[1], s4[2] * s4[3])
    var x4 = xcnn_reshape(a, s4)
    var y = xcnn_host_f32(s4, context)
    var entries = xcnn_entries(fields, context)
    if xcnn_mixed(entries):
        var h = xcnn_dev_get(model, entries, "mask", xcnn_size(s4), context)
        _ = py_call_typed(xcnn_entry(entries, "x_cnn_dropout2d_m"), Python.list(xcnn_addr(x4), xcnn_addr(y), h),
                          PythonObject(4), prm, fields["p"])
        _set(fields, "_mask_host", PythonObject(None))
        _set(fields, "_mask_dev", h)
        _set(fields, "_mask_shape", xcnn_tuple(xs))
        return xcnn_reshape(y, xs)
    var mask = xcnn_host_f32(s4, context)
    _ = py_call_typed(xcnn_entry(entries, "x_cnn_dropout2d"), xcnn_addr(x4), xcnn_addr(y), xcnn_addr(mask), prm, fields["p"])
    # the `mask_` setter: the host mask, no resident copy
    _set(fields, "_mask_host", xcnn_reshape(mask, xs))
    _set(fields, "_mask_dev", PythonObject(None))
    return xcnn_reshape(y, xs)


# ---------------------------------------------------------------- adaptive pools

def _pool_layer(model: PythonObject, entries: PythonObject, k: List[Int], context: PythonObject) raises -> PythonObject:
    """`_AdaptivePool._pool_for(b, k)`: the MaxPool2d / AvgPool2d of kernel =
    stride = `k` (kept across calls under G3), built with the fields its
    constructor sets."""
    var b = xcnn_builtins()
    var fields = py_fields(model)
    var maxed = Bool(py=py_call(b.getattr, py_type(model), "_max"))
    var cls = context["xcnn_maxpool_class"] if maxed else context["xcnn_avgpool_class"]
    var hit = py_get(fields, "_pool_cache")
    if xcnn_resident(entries, context):
        if hit is not None and Int(py=hit[0][0]) == k[0] and Int(py=hit[0][1]) == k[1] and py_type(hit[1]) is cls:
            return hit[1]
    var layer = py_call(py_call(b.getattr, b.object, "__new__"), cls)
    var state = py_fields(layer)
    var pair = Python.tuple(k[0], k[1])
    py_setitem(state, "ceil_mode", PythonObject(False))
    py_setitem(state, "divisor_override", PythonObject(None))
    py_setitem(state, "kernel_size", pair)
    py_setitem(state, "stride", pair)
    py_setitem(state, "padding", Python.tuple(0, 0))
    py_setitem(state, "dilation", Python.tuple(1, 1))
    py_setitem(state, "count_include_pad", PythonObject(True))
    py_setitem(state, "input_shape", PythonObject(None))
    py_setitem(state, "numeric_mode", py_get(fields, "numeric_mode"))
    _set(fields, "_pool_cache", Python.tuple(pair, layer))
    return layer


def _layer_indices(layer: PythonObject, context: PythonObject) raises -> PythonObject:
    """`MaxPool2d.indices_` of a pool this call ran: the host winners, read
    back once from the resident ones when they stayed on the device."""
    var d = py_fields(layer)
    var held = py_get(d, "_idx_host")
    if held is None and py_get(d, "_idx_dev") is not None:
        var b = xcnn_builtins()
        var shape = List[Int]()
        var given = d["_idx_shape"]
        for i in range(len(given)):  # driver-loop: over shape words
            shape.append(Int(py=given[i]))
        var idx = xcnn_host_i32(shape, context)
        var dev = d["_dev"]
        var entries = context["xcnn_bindings"]
        var owner = PythonObject(None)
        var keys = Python.tuple("identical", "fast")
        for i in range(2):  # driver-loop: over the two numeric modes
            var e = py_get(entries, keys[i])
            if e is not None and not Bool(py=py_call(b.isinstance, e, b.str)) and e["module"] is dev.b:
                owner = e
        if owner is None:
            raise Error("RuntimeError: a resident array of a binding this process did not connect")
        _ = py_call_typed(owner["symbols"]["x_cnn_res_download"], d["_idx_dev"], xcnn_addr(idx), PythonObject(xcnn_size(shape)))
        py_setitem(d, "_idx_host", idx)
        return idx
    return held


def xcnn_adaptive_forward(model: PythonObject, x: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_AdaptivePool.forward(x)`."""
    var b = xcnn_builtins()
    var fields = py_fields(model)
    var entries = xcnn_entries(fields, context)
    var m = xcnn_idn(entries, XCNN_F_ADAPT)
    var maxed = Bool(py=py_call(b.getattr, py_type(model), "_max"))
    var xt = xcnn_is_t(x, context)
    var value = x
    if xt and not m:
        value = xcnn_f32(x, "x", context)
        xt = False
    elif xt and not xcnn_on_dev(entries, x, context):
        return xcnn_wrap(entries, xcnn_adaptive_forward(model, xcnn_host_of(x, context), context), context)
    elif not xt:
        value = xcnn_f32(x, "x", context)
    var xs = xcnn_shape(value)
    if len(xs) != 4:
        raise Error("mojolearn: " + _class_name(model) + ".forward takes (N, C, H, W)")
    _set(fields, "_xshape", xcnn_tuple(xs))
    _set(fields, "_idx_host", PythonObject(None))
    _set(fields, "_idx_dev", PythonObject(None))
    var o = _pair(fields, "output_size")
    if xs[2] % o[0] == 0 and xs[3] % o[1] == 0:
        var layer = _pool_layer(model, entries, [xs[2] // o[0], xs[3] // o[1]], context)
        _set(fields, "_layer", layer)
        var out = _pool2d_forward(layer, value, context)
        if maxed and not m:
            _set(fields, "_idx_host", _layer_indices(layer, context))
        return out
    _set(fields, "_layer", PythonObject(None))
    var oshape: List[Int] = [xs[0], xs[1], o[0], o[1]]
    var prm = Python.list(xs[0], xs[1], xs[2], xs[3], o[0], o[1], 2 if maxed else 0)
    if m:
        var out = xcnn_out_new(entries, oshape, context) if xt else xcnn_host_f32(oshape, context)
        var h = xcnn_dev_get(model, entries, "idx", xcnn_size(oshape), context) if maxed else PythonObject(0)
        var bits = (3 if xt else 0) | (4 if maxed else 0)
        _ = py_call_typed(xcnn_entry(entries, "x_cnn_adaptive_pool_m"),
                          Python.list(value.h if xt else xcnn_addr(value), out.h if xt else xcnn_addr(out), h),
                          PythonObject(bits), prm)
        if maxed:
            _set(fields, "_idx_dev", h)
            _set(fields, "_idx_shape", xcnn_tuple(oshape))
        return out
    var out = xcnn_host_f32(oshape, context)
    var idx = xcnn_host_i32(oshape, context)
    _ = py_call_typed(xcnn_entry(entries, "x_cnn_adaptive_pool"), xcnn_addr(value), xcnn_addr(out), xcnn_addr(idx), prm)
    _set(fields, "_idx_host", idx if maxed else PythonObject(None))
    return out


# ---------------------------------------------------------------- dispatch

def xcnn_pool_layers(model: PythonObject, operation: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    if operation == "pool_forward":
        return xcnn_pool_forward(model, args[0], context)
    if operation == "pool_transform":
        return _flat_rows(xcnn_pool_forward(model, xcnn_rows(model, args[0], True, context), context), context)
    if operation == "dropout_forward":
        return xcnn_dropout_forward(model, args[0], context)
    if operation == "dropout_transform":
        return _flat_rows(xcnn_dropout_forward(model, xcnn_rows(model, args[0], False, context), context), context)
    if operation == "adaptive_forward":
        return xcnn_adaptive_forward(model, args[0], context)
    if operation == "adaptive_transform":
        return _flat_rows(xcnn_adaptive_forward(model, xcnn_rows(model, args[0], False, context), context), context)
    if operation == "conv_fit":
        return xcnn_conv_fit(model, args[0], context)
    raise Error("RuntimeError: unknown CNN layer operation " + operation)
