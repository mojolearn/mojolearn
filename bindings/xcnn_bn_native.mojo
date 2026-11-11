# SPDX-License-Identifier: Apache-2.0
"""Native `forward`, `transform` and `fit` of BatchNorm2d / BatchNorm1d
(python/mojolearn/_expansion_cnn.py).

The batch-statistics words (`aux`: eps, the momentum factor, gamma and
beta), the running statistics (resident under G3, read back lazily by the
layer's properties), the host or `_m` entry and the fitted fields are made
here, by the Python body's rules and with its entries and residency bits,
so no bit moves. The Python `backward` reads the same fields (`_aux`,
`_aux_dev`, `_mode`, `_x3shape`, `_xdev`, `_xdev_dev`, `_shape`, `_x`).
"""
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_fields, py_contains
from bindings.xcnn_common import (
    XCNN_F_LAYER_IO, xcnn_builtins, xcnn_entries, xcnn_word, xcnn_ptr, xcnn_mixed, xcnn_idn, xcnn_resident, xcnn_shape,
    xcnn_size, xcnn_tuple, xcnn_is_t, xcnn_on_dev, xcnn_new, xcnn_host_f32, xcnn_f32, xcnn_reshape, xcnn_addr,
    xcnn_dev, xcnn_dev_get, xcnn_dev_upload, xcnn_dev_has, xcnn_dev_handle, xcnn_download, xcnn_slice_f32, xcnn_put_f32,
    xcnn_shape_text,
)
from bindings.mode_metadata_native import _fields
from bindings.neural_block_admission import neural_plain_value
from bindings.native_loaded_module import NativeModules
from bindings.sequential_admission import sequential_admit_value
from bindings.xcnn_common import xcnn_out_new, xcnn_dev_fields, xcnn_key
from std.python._cpython import GILReleased
from x_cnn.door_run import xcnn_run_fill_f32
from x_cnn.bn_run import (
    BnSettings, bn_run_prep, bn_run_aux_head, bn_run_aux_count, bn_run_untracked_running, bn_run_check_channels,
    bn_run_host_dims, bn_run_check_saved, bn_run_check_nonneg, bn_run_forward, bn_run_forward_m, bn_run_backward,
    bn_run_flags, bn_run_dx_shape_check,
)


def _set(fields: PythonObject, name: String, value: PythonObject) raises:
    py_setitem(fields, PythonObject(name), value)


def _truth(value: PythonObject) raises -> Bool:
    return Bool(py=py_call(xcnn_builtins().bool, value))


def _bn_admit(layer: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = _fields(layer)
    var mods = NativeModules()   # one module view for the fifteen fields
    for name in ["num_features", "eps", "affine", "momentum", "track_running_stats", "num_batches_tracked_",  # driver-loop: over consumed passive settings
                 "input_shape", "numeric_mode", "training", "_run_stale", "_aux_k", "_mode", "_x3shape", "_shape", "_xdev"]:  # driver-loop: admit consumed passive settings before coercion
        var value = py_get(fields, PythonObject(name))
        neural_plain_value(mods, value, 0, context)
    for name in ["_run_dev", "_xdev_dev"]:  # driver-loop: resident owner metadata retained by forward
        var owner = py_get(fields, PythonObject(name))
        if owner is not None:
            _ = xcnn_dev_fields(owner, context)
    var b = xcnn_builtins()
    for name in ["_g_dev", "_aux_dev"]:  # driver-loop: resident auxiliary buffer state
        var pending = py_get(fields, PythonObject(name))
        if pending is not None:
            if py_type(pending) is not b.tuple or len(pending) != 2 or py_type(pending[1]) is not b.int or Int(py=pending[1]) < 1:
                raise Error("TypeError: batch normalization pending state requires native owner/handle metadata")
            _ = xcnn_dev_fields(pending[0], context)
    return fields


def _ones(count: Int, context: PythonObject) raises -> PythonObject:
    var result = xcnn_host_f32([count], context)
    xcnn_run_fill_f32(Int(py=xcnn_addr(result)), count, 1.0)
    _ = result
    return result


def _bn_init(layer: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    neural_plain_value(args, 0, context)
    var fields = _fields(layer)
    var count = Int(py=args[0])
    if count < 0:
        raise Error("ValueError: num_features must be nonnegative")
    _set(fields, "num_features", PythonObject(count))
    _set(fields, "eps", PythonObject(Float64(py=args[1])))
    _set(fields, "momentum", PythonObject(None) if args[2] is None else PythonObject(Float64(py=args[2])))
    _set(fields, "affine", PythonObject(Bool(py=args[3])))
    var track = Bool(py=args[4])
    _set(fields, "track_running_stats", PythonObject(track))
    _set(fields, "num_batches_tracked_", PythonObject(0))
    _set(fields, "input_shape", args[5])
    _set(fields, "numeric_mode", args[6])
    _set(fields, "weight_", _ones(count, context))
    _set(fields, "bias_", xcnn_host_f32([count], context))
    _set(fields, "running_mean_", xcnn_host_f32([count], context) if track else PythonObject(None))
    _set(fields, "running_var_", _ones(count, context) if track else PythonObject(None))
    _set(fields, "training", PythonObject(True))
    _set(fields, "_run_dev", PythonObject(None))
    _set(fields, "_run_stale", PythonObject(False))
    return PythonObject(None)


# ---------------------------------------------------------------- the running statistics

def _sync_running(fields: PythonObject, context: PythonObject) raises:
    """`_sync_running()`: resident running statistics read back once after a
    G3 training forward left them stale."""
    if not _truth(py_get(fields, "_run_stale", PythonObject(False))):
        return
    _set(fields, "_run_stale", PythonObject(False))
    var dev = py_get(fields, "_run_dev")
    if dev is not None and xcnn_dev_has(dev, "running"):
        var c = Int(py=fields["num_features"])
        var r = xcnn_download(dev, xcnn_dev_handle(dev, "running"), 2 * c, context)
        _set(fields, "running_mean_", xcnn_slice_f32(r, 0, c, context))
        _set(fields, "running_var_", xcnn_slice_f32(r, c, c, context))


def _running(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    """`np.concatenate([running_mean_, running_var_]).astype(np.float32)`
    through the properties' read-back."""
    _sync_running(fields, context)
    var c = Int(py=fields["num_features"])
    var r = xcnn_host_f32([2 * c], context)
    xcnn_put_f32(r, 0, fields["running_mean_"], c, "running_mean_", context)
    xcnn_put_f32(r, c, fields["running_var_"], c, "running_var_", context)
    return r


def _set_running(fields: PythonObject, running: PythonObject, context: PythonObject) raises:
    """`_finish`: the running-statistics setters (each reads back first and
    drops the resident copy)."""
    var c = Int(py=fields["num_features"])
    _sync_running(fields, context)
    _set(fields, "running_mean_", xcnn_slice_f32(running, 0, c, context))
    _set(fields, "_run_dev", PythonObject(None))
    _sync_running(fields, context)
    _set(fields, "running_var_", xcnn_slice_f32(running, c, c, context))
    _set(fields, "_run_dev", PythonObject(None))


def _grads_host(fields: PythonObject, context: PythonObject) raises:
    """`_grads_host()`: the pending resident dgamma / dbeta read back."""
    var pend = py_get(fields, "_g_dev")
    if pend is None:
        return
    _set(fields, "_g_dev", PythonObject(None))
    var c = Int(py=fields["num_features"])
    var aux = xcnn_download(pend[0], pend[1], 2 + 7 * c, context)
    var gb = xcnn_slice_f32(aux, 2 + 3 * c, c, context)
    var gw = xcnn_slice_f32(aux, 2 + 4 * c, c, context)
    if not _truth(fields["affine"]):
        gw = xcnn_host_f32([c], context)
        gb = xcnn_host_f32([c], context)
    _set(fields, "grad_weight_", gw)
    _set(fields, "grad_bias_", gb)


# ---------------------------------------------------------------- one forward

def _prep(fields: PythonObject, need_running: Bool, context: PythonObject) raises -> PythonObject:
    """`_prep(n, c, hw, need_running)`: (aux, running or None, batch_stats);
    the batch counter moves here."""
    var c = Int(py=fields["num_features"])
    var training = _truth(fields["training"])
    var track = _truth(fields["track_running_stats"])
    # UNPACK: the counter and momentum are read only when they move (as before)
    var tracked = 0
    var momentum = fields["momentum"] if training and track else PythonObject(None)
    if training and track:
        tracked = Int(py=fields["num_batches_tracked_"])
    var settings = BnSettings(c, Float64(py=fields["eps"]), training, track, tracked,
                              momentum is not None, 0.0 if momentum is None else Float64(py=momentum))
    var prep = bn_run_prep(settings)
    if prep.counted:
        _set(fields, "num_batches_tracked_", PythonObject(prep.tracked))
    var batch_stats = prep.batch_stats
    var aux = xcnn_host_f32([bn_run_aux_count(c)], context)
    bn_run_aux_head(Int(py=xcnn_addr(aux)), settings.eps, prep.factor)
    _ = aux  # keep the Python object alive through the native writes above
    xcnn_put_f32(aux, 2 + 5 * c, fields["weight_"], c, "weight_", context)
    xcnn_put_f32(aux, 2 + 6 * c, fields["bias_"], c, "bias_", context)
    var running = PythonObject(None)
    if need_running:
        if track:
            running = _running(fields, context)
        else:
            running = xcnn_host_f32([2 * c], context)
            bn_run_untracked_running(Int(py=xcnn_addr(running)), c)
            _ = running  # keep the Python object alive through the native writes above
    return Python.tuple(aux, running, batch_stats)


def _finish(fields: PythonObject, running: PythonObject, context: PythonObject) raises:
    if _truth(fields["training"]) and _truth(fields["track_running_stats"]):
        _set_running(fields, running, context)


def _forward_res(layer: PythonObject, fields: PythonObject, entries: PythonObject, xh: PythonObject, yh: PythonObject,
                 n: Int, c: Int, hw: Int, dev_y: Bool, context: PythonObject) raises -> PythonObject:
    """`_forward_res`: G3, the running statistics and aux resident."""
    var dev = xcnn_dev(layer, entries, context)
    var prepared = _prep(fields, False, context)
    var aux = prepared[0]
    var batch_stats = Bool(py=prepared[2])
    var k = 1 - Int(py=py_get(fields, "_aux_k", PythonObject(1)))
    var name = "aux" + String(k)
    var pend = py_get(fields, "_g_dev")
    var table = _fields(dev)["h"]
    var cur = py_get(table, PythonObject(name))
    if pend is not None and cur is not None and Bool(py=pend[1] == cur[0]):
        _grads_host(fields, context)   # unread gradients in the array this forward reuses
    var ah = xcnn_dev_upload(layer, entries, name, aux, context)
    var run_dev = py_get(fields, "_run_dev")
    if run_dev is None or run_dev is not dev or not xcnn_dev_has(dev, "running"):
        var r = _running(fields, context)
        _ = xcnn_dev_upload(layer, entries, "running", r, context)
        _set(fields, "_run_dev", dev)
    var rh = xcnn_dev_handle(dev, "running")
    var key = xcnn_key(entries)
    var addrs: List[Int] = [xcnn_word(xh), xcnn_word(yh), xcnn_word(rh), xcnn_word(ah)]
    var py = Python()
    with GILReleased(py):
        bn_run_forward_m(key, addrs[0], addrs[1], addrs[2], addrs[3], 15 if dev_y else 13, n, c, hw, batch_stats)
    if _truth(fields["training"]):
        _set(fields, "_run_stale", PythonObject(True))
    _set(fields, "_aux_k", PythonObject(k))
    _set(fields, "_aux", aux)
    _set(fields, "_aux_dev", Python.tuple(dev, ah))
    _set(fields, "_mode", PythonObject(batch_stats))
    _set(fields, "_x3shape", Python.tuple(n, c, hw))
    return aux


def xcnn_bn_forward_dev(layer: PythonObject, fields: PythonObject, entries: PythonObject, xh: PythonObject, yh: PythonObject,
                 n: Int, c: Int, hw: Int, dev_y: Bool, context: PythonObject) raises -> PythonObject:
    """`_forward_dev`: resident x into resident y (or a host y)."""
    if _truth(fields["track_running_stats"]) and xcnn_resident(entries, context):
        return _forward_res(layer, fields, entries, xh, yh, n, c, hw, dev_y, context)
    _set(fields, "_aux_dev", PythonObject(None))
    var prepared = _prep(fields, True, context)
    var aux = prepared[0]
    var running = prepared[1]
    var batch_stats = Bool(py=prepared[2])
    var key = xcnn_key(entries)
    var addrs: List[Int] = [xcnn_word(xh), xcnn_word(yh), xcnn_ptr(running), xcnn_ptr(aux)]
    var py = Python()
    with GILReleased(py):
        bn_run_forward_m(key, addrs[0], addrs[1], addrs[2], addrs[3], 3 if dev_y else 1, n, c, hw, batch_stats)
    _ = running  # keep the Python object alive through the native call above
    _ = aux  # keep the Python object alive through the native call above
    _finish(fields, running, context)
    _set(fields, "_aux", aux)
    _set(fields, "_mode", PythonObject(batch_stats))
    _set(fields, "_x3shape", Python.tuple(n, c, hw))
    return aux


def _host_result(y: PythonObject, shape: List[Int]) raises -> PythonObject:
    """`y.reshape(shape) if len(shape) != 2 else y[:, :, 0]`."""
    if len(shape) != 2:
        return xcnn_reshape(y, shape)
    return xcnn_reshape(y, [shape[0], shape[1]])


def xcnn_bn_forward(layer: PythonObject, x: PythonObject, context: PythonObject) raises -> PythonObject:
    """`BatchNorm2d.forward(x)`."""
    return _bn_forward_admitted(layer, _bn_admit(layer, context), x, context)


def _bn_forward_admitted(layer: PythonObject, fields: PythonObject, x: PythonObject, context: PythonObject) raises -> PythonObject:
    """`xcnn_bn_forward` on the fields this operation's `_bn_admit` returned
    (the public dispatcher admits once, not once per helper)."""
    var features = Int(py=fields["num_features"])
    if xcnn_is_t(x, context):
        var b = xcnn_entries(fields, context)
        var shape = xcnn_shape(x)
        if xcnn_on_dev(b, x, context) and xcnn_idn(b, XCNN_F_LAYER_IO) and len(shape) >= 2 and xcnn_size(shape) > 0:
            var n = shape[0]
            var c = shape[1]
            bn_run_check_channels(c, features)
            var hw = xcnn_size(shape) // (n * c)
            var y = xcnn_new(b, shape, True, context)   # bn_apply stores every word
            _ = xcnn_bn_forward_dev(layer, fields, b, x.h, y.h, n, c, hw, True, context)
            _set(fields, "_xdev", x.h)
            _set(fields, "_x", PythonObject(None))
            _set(fields, "_shape", xcnn_tuple(shape))
            _set(fields, "_xdev_dev", xcnn_dev(layer, b, context))
            _set(fields, "_xt", x)  # keeps the resident words alive for the backward
            return y
    _set(fields, "_xt", PythonObject(None))
    # `_nchw(x)`: (N, C, -1) C-ordered, the caller's shape kept
    var a = xcnn_f32(x, "x", context)
    var shape = xcnn_shape(a)
    var dims = bn_run_host_dims(shape, features)
    var n = dims[0]
    var c = dims[1]
    var hw = dims[2]
    var x3 = xcnn_reshape(a, [n, c, hw], context)
    var b = xcnn_entries(fields, context)
    var y = xcnn_host_f32([n, c, hw], context)
    if xcnn_mixed(b):
        var xh = xcnn_dev_upload(layer, b, "x", x3, context)
        _ = xcnn_bn_forward_dev(layer, fields, b, xh, xcnn_addr(y), n, c, hw, False, context)
        _ = y  # keep the Python object alive through the native call above
        _set(fields, "_xdev", xh)
        _set(fields, "_x", PythonObject(None))
        _set(fields, "_shape", xcnn_tuple(shape))
        _set(fields, "_xdev_dev", xcnn_dev(layer, b, context))
        return _host_result(y, shape)
    var prepared = _prep(fields, True, context)
    var aux = prepared[0]
    var running = prepared[1]
    var batch_stats = Bool(py=prepared[2])
    var key = xcnn_key(b)
    var addrs: List[Int] = [xcnn_ptr(x3), xcnn_ptr(y), xcnn_ptr(running), xcnn_ptr(aux)]
    var py = Python()
    with GILReleased(py):
        bn_run_forward(key, addrs[0], addrs[1], addrs[2], addrs[3], n, c, hw, batch_stats)
    _ = running  # keep the Python object alive through the native call above
    _ = aux  # keep the Python object alive through the native call above
    _ = y  # keep the Python object alive through the native call above
    _ = x3  # keep the Python object alive through the native call above
    _finish(fields, running, context)
    _set(fields, "_aux_dev", PythonObject(None))
    _set(fields, "_x", x3)
    _set(fields, "_aux", aux)
    _set(fields, "_shape", xcnn_tuple(shape))
    _set(fields, "_mode", PythonObject(batch_stats))
    _set(fields, "_x3shape", Python.tuple(n, c, hw))
    _set(fields, "_xdev", PythonObject(None))
    return _host_result(y, shape)


def _bn_rows(fields: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    """`BatchNorm2d._rows(X)` on the admitted fields."""
    var a = xcnn_f32(source, "X", context)
    var s = xcnn_shape(a)
    var declared = py_get(fields, "input_shape")
    if len(s) == 2 and declared is not None:
        var dims = List[Int]()
        dims.append(s[0])
        var given = py_call(xcnn_builtins().tuple, declared)
        for i in range(len(given)):  # driver-loop: over the declared sample shape words
            dims.append(Int(py=given[i]))
        return xcnn_reshape(a, dims, context)
    return a


def _shape_words(value: PythonObject, context: PythonObject) raises -> List[Int]:
    neural_plain_value(value, 0, context)
    var out = List[Int]()
    for i in range(len(value)):  # driver-loop: saved forward shape metadata
        var dim = Int(py=value[i])
        bn_run_check_nonneg(dim)
        out.append(dim)
    return out^


def _backward_grads(fields: PythonObject, aux: PythonObject, c: Int, context: PythonObject) raises:
    _set(fields, "_g_dev", PythonObject(None))
    var affine = _truth(fields["affine"])
    _set(fields, "grad_bias_", xcnn_slice_f32(aux, 2 + 3 * c, c, context) if affine else xcnn_host_f32([c], context))
    _set(fields, "grad_weight_", xcnn_slice_f32(aux, 2 + 4 * c, c, context) if affine else xcnn_host_f32([c], context))


def _backward_execute(layer: PythonObject, fields: PythonObject, entries: PythonObject,
                      xh: PythonObject, gh: PythonObject, dxh: PythonObject,
                      dev_x: Bool, dev_g: Bool, dev_dx: Bool, context: PythonObject) raises:
    var shape = _shape_words(fields["_x3shape"], context)
    var c = Int(py=fields["num_features"])
    bn_run_check_saved(shape, c)
    var stats = 1 if _truth(fields["_mode"]) else 0
    var ad = py_get(fields, "_aux_dev")
    if ad is not None:
        var b = xcnn_builtins()
        if py_type(ad) is not b.tuple or len(ad) != 2 or py_type(ad[1]) is not b.int:
            raise Error("TypeError: batch normalization aux state requires native owner/handle metadata")
    if (dev_x and ad is not None and _fields(ad[0])["b"] is entries["module"]
            and _truth(fields["track_running_stats"]) and xcnn_resident(entries, context)):
        _set(fields, "_g_dev", PythonObject(None))
        var flags = 8 | bn_run_flags(True, dev_g, dev_dx)
        var key = xcnn_key(entries)
        var addrs: List[Int] = [xcnn_word(xh), xcnn_word(gh), xcnn_word(dxh), xcnn_word(ad[1])]
        var py = Python()
        with GILReleased(py):
            bn_run_backward(key, addrs[0], addrs[1], addrs[2], addrs[3], flags, True, shape, c, stats)
        _set(fields, "_g_dev", ad)
        return
    var original = xcnn_f32(fields["_aux"], "saved batch normalization aux", context)
    if Int(py=original.size) != bn_run_aux_count(c):
        raise Error("ValueError: batch normalization saved aux has the wrong size")
    var aux = xcnn_slice_f32(original, 0, bn_run_aux_count(c), context)
    var key = xcnn_key(entries)
    var addrs: List[Int] = [xcnn_word(xh), xcnn_word(gh), xcnn_word(dxh), xcnn_ptr(aux)]
    var py = Python()
    with GILReleased(py):
        bn_run_backward(key, addrs[0], addrs[1], addrs[2], addrs[3], bn_run_flags(dev_x, dev_g, dev_dx),
                        dev_x or dev_g or dev_dx, shape, c, stats)
    _ = aux  # keep the Python object alive through the native call above
    _backward_grads(fields, aux, c, context)


def xcnn_bn_backward_dev(layer: PythonObject, entries: PythonObject, xh: PythonObject,
                          gh: PythonObject, dxh: PythonObject, dev_dx: Bool,
                          context: PythonObject) raises:
    var fields = _bn_admit(layer, context)
    if py_get(fields, "_x3shape") is None:
        raise Error("RuntimeError: call forward before batch normalization backward")
    _backward_execute(layer, fields, entries, xh, gh, dxh, True, True, dev_dx, context)


def xcnn_bn_backward(layer: PythonObject, gradient: PythonObject, context: PythonObject) raises -> PythonObject:
    return _bn_backward_admitted(layer, _bn_admit(layer, context), gradient, context)


def _bn_backward_admitted(layer: PythonObject, fields: PythonObject, gradient: PythonObject,
                          context: PythonObject) raises -> PythonObject:
    if py_get(fields, "_x3shape") is None or py_get(fields, "_shape") is None:
        raise Error("RuntimeError: call forward before backward")
    var shape = _shape_words(fields["_shape"], context)
    var saved = _shape_words(fields["_x3shape"], context)
    var entries = xcnn_entries(fields, context)
    var owner = py_get(fields, "_xdev_dev")
    var xh = py_get(fields, "_xdev")
    var resident_x = xh is not None and owner is not None and _fields(owner)["b"] is entries["module"] and xcnn_mixed(entries)
    if (xcnn_on_dev(entries, gradient, context) and xcnn_idn(entries, XCNN_F_LAYER_IO)
            and resident_x and xcnn_size(xcnn_shape(gradient)) == xcnn_size(saved)):
        var result = xcnn_out_new(entries, shape, context)
        _backward_execute(layer, fields, entries, xh, gradient.h, result.h, True, True, True, context)
        return result
    var g = xcnn_f32(gradient, "grad_out", context)
    bn_run_dx_shape_check(Int(py=g.size), xcnn_size(saved))
    var result = xcnn_host_f32(shape, context)
    var original = PythonObject(None)
    if not resident_x:
        original = xcnn_f32(fields["_x"], "saved batch normalization input", context)
        if Int(py=original.size) != xcnn_size(saved):
            raise Error("ValueError: batch normalization saved input has the wrong size")
        xh = xcnn_addr(original)
    _backward_execute(layer, fields, entries, xh, xcnn_addr(g), xcnn_addr(result), resident_x, False, False, context)
    _ = original
    _ = g
    return result


def xcnn_bn_layers(layer: PythonObject, operation: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    if operation == "bn_init":
        return _bn_init(layer, args, context)
    var fields = _bn_admit(layer, context)
    if operation == "bn_train":
        neural_plain_value(args[0], 0, context)
        _set(fields, "training", PythonObject(Bool(py=args[0])))
        return layer
    if operation == "bn_running_get" or operation == "bn_running_set":
        var name = String(args[0])
        if name != "running_mean_" and name != "running_var_":
            raise Error("ValueError: invalid batch normalization running field")
        _sync_running(fields, context)
        if operation == "bn_running_set":
            sequential_admit_value(args[1], context)
            _set(fields, name, args[1])
            _set(fields, "_run_dev", PythonObject(None))
            return PythonObject(None)
        return py_get(fields, PythonObject(name))
    if operation == "bn_grad_get" or operation == "bn_grad_set":
        var name = String(args[0])
        if name != "grad_weight_" and name != "grad_bias_":
            raise Error("ValueError: invalid batch normalization gradient field")
        _grads_host(fields, context)
        if operation == "bn_grad_set":
            sequential_admit_value(args[1], context)
            _set(fields, name, args[1])
            return PythonObject(None)
        if not py_contains(fields, PythonObject(name)):
            raise Error("AttributeError: " + name)
        return fields[PythonObject(name)]
    if operation == "bn_backward":
        return _bn_backward_admitted(layer, fields, args[0], context)
    if operation == "bn_forward":
        return _bn_forward_admitted(layer, fields, args[0], context)
    if operation == "bn_transform":
        var out = _bn_forward_admitted(layer, fields, _bn_rows(fields, args[0], context), context)
        var s = xcnn_shape(out)
        var rest = 1
        for i in range(1, len(s)):  # driver-loop: over shape words
            rest *= s[i]
        return xcnn_reshape(out, [s[0], rest], context)
    if operation == "bn_fit":
        var was = fields["training"]
        _set(fields, "training", PythonObject(True))
        try:
            _ = _bn_forward_admitted(layer, fields, _bn_rows(fields, args[0], context), context)
        finally:
            _set(fields, "training", was)
        return layer
    raise Error("ValueError: unknown batch normalization operation")
