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
from bindings.py_typed_call import py_call_typed
from bindings.xcnn_common import (
    XCNN_F_LAYER_IO, xcnn_builtins, xcnn_entries, xcnn_entry, xcnn_mixed, xcnn_idn, xcnn_resident, xcnn_shape,
    xcnn_size, xcnn_tuple, xcnn_is_t, xcnn_on_dev, xcnn_new, xcnn_host_f32, xcnn_f32, xcnn_reshape, xcnn_addr,
    xcnn_dev, xcnn_dev_get, xcnn_dev_upload, xcnn_dev_has, xcnn_dev_handle, xcnn_download, xcnn_slice_f32, xcnn_put_f32,
    xcnn_shape_text,
)
from bindings.xseq_native_common import xseq_f32


comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]


def _set(fields: PythonObject, name: String, value: PythonObject) raises:
    py_setitem(fields, PythonObject(name), value)


def _truth(value: PythonObject) raises -> Bool:
    return Bool(py=py_call(xcnn_builtins().bool, value))


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
    var batch_stats = training or not track
    var factor = 0.0
    if training and track:
        var tracked = Int(py=fields["num_batches_tracked_"]) + 1
        _set(fields, "num_batches_tracked_", PythonObject(tracked))
        var momentum = fields["momentum"]
        factor = 1.0 / Float64(tracked) if momentum is None else Float64(py=momentum)
    var aux = xcnn_host_f32([2 + 7 * c], context)
    var words = _F32P(unsafe_from_address=Int(py=xcnn_addr(aux)))
    _ = aux  # keep the Python object alive through the native call above
    words[0] = Float32(Float64(py=fields["eps"]))
    _ = aux  # keep the Python object alive through the native call above
    words[1] = Float32(factor)
    _ = aux  # keep the Python object alive through the native call above
    xcnn_put_f32(aux, 2 + 5 * c, fields["weight_"], c, "weight_", context)
    xcnn_put_f32(aux, 2 + 6 * c, fields["bias_"], c, "bias_", context)
    var running = PythonObject(None)
    if need_running:
        if track:
            running = _running(fields, context)
        else:
            running = xcnn_host_f32([2 * c], context)
            var r = _F32P(unsafe_from_address=Int(py=xcnn_addr(running)))
            _ = running  # keep the Python object alive through the native call above
            for i in range(c):  # driver-loop: over BatchNorm channels of the initial running variance
                r[c + i] = 1.0
                _ = running  # keep the Python object alive through the native call above
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
    var table = py_fields(dev)["h"]
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
    _ = py_call_typed(xcnn_entry(entries, "x_cnn_batchnorm_forward_m"), Python.list(xh, yh, rh, ah),
                      PythonObject(15 if dev_y else 13), Python.list(n, c, hw, 1 if batch_stats else 0))
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
    _ = py_call_typed(xcnn_entry(entries, "x_cnn_batchnorm_forward_m"), Python.list(xh, yh, xcnn_addr(running), xcnn_addr(aux)),
                      PythonObject(3 if dev_y else 1), Python.list(n, c, hw, 1 if batch_stats else 0))
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
    var fields = py_fields(layer)
    var features = Int(py=fields["num_features"])
    if xcnn_is_t(x, context):
        var b = xcnn_entries(fields, context)
        var shape = xcnn_shape(x)
        if xcnn_on_dev(b, x, context) and xcnn_idn(b, XCNN_F_LAYER_IO) and len(shape) >= 2 and xcnn_size(shape) > 0:
            var n = shape[0]
            var c = shape[1]
            if c != features:
                raise Error("mojolearn: input has " + String(c) + " channels, the layer " + String(features))
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
    if len(shape) < 2:
        raise Error("IndexError: tuple index out of range")
    var n = shape[0]
    var c = shape[1]
    if c != features:
        raise Error("mojolearn: input has " + String(c) + " channels, the layer " + String(features))
    var hw = xcnn_size(shape) // max(n * c, 1) if n * c != 0 else 0
    if len(shape) == 2:
        hw = 1
    var x3 = xcnn_reshape(a, [n, c, hw])
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
    _ = py_call_typed(xcnn_entry(b, "x_cnn_batchnorm_forward"), xcnn_addr(x3), xcnn_addr(y), xcnn_addr(running), xcnn_addr(aux),
                      Python.list(n, c, hw, 1 if batch_stats else 0))
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


def _bn_rows(layer: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    """`BatchNorm2d._rows(X)`."""
    var fields = py_fields(layer)
    var a = xcnn_f32(source, "X", context)
    var s = xcnn_shape(a)
    var declared = py_get(fields, "input_shape")
    if len(s) == 2 and declared is not None:
        var dims = List[Int]()
        dims.append(s[0])
        var given = py_call(xcnn_builtins().tuple, declared)
        for i in range(len(given)):  # driver-loop: over the declared sample shape words
            dims.append(Int(py=given[i]))
        return xcnn_reshape(a, dims)
    return a


def xcnn_bn_layers(layer: PythonObject, operation: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    if operation == "bn_forward":
        return xcnn_bn_forward(layer, args[0], context)
    if operation == "bn_transform":
        var out = xcnn_bn_forward(layer, _bn_rows(layer, args[0], context), context)
        var s = xcnn_shape(out)
        var rest = 1
        for i in range(1, len(s)):  # driver-loop: over shape words
            rest *= s[i]
        return xcnn_reshape(out, [s[0], rest])
    # `fit(X)`: one training-mode pass (updates the running statistics)
    var fields = py_fields(layer)
    var was = fields["training"]
    _set(fields, "training", PythonObject(True))
    _ = xcnn_bn_forward(layer, _bn_rows(layer, args[0], context), context)
    _set(fields, "training", was)
    return layer
