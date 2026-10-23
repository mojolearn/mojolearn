# SPDX-License-Identifier: Apache-2.0
"""Native public operations of the sequence-expansion layers, routed through
the optimizer native door (bindings/optimizer_native_router.mojo):
`layer_norm_forward`, `layer_norm_backward` and `LayerNorm.forward`
(python/mojolearn/_x_sequence_norm.py), and `MoEBlock.forward`
(python/mojolearn/_x_sequence_moe.py).

The host-array route runs the `_mojolearn_x_sequence` host entry
(`layer_norm`, `moe_forward`), a resident `SequenceDeviceTensor` the device
entry (`layer_norm_dev` with the kept statistics, `moe_forward_dev` with the
block's resident weights), with the same argument words as the Python
bodies, so no bit moves. The fields the Python `backward` methods and the
routing properties read (`_x`, `_stats`, `_routing`, `_wdev`) are written
under the same names.
"""
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_fields, py_contains
from bindings.py_typed_call import py_call_typed
from bindings.optimizer_device_native import device_shape, shape_count
from bindings.optimizer_public_door import TensorView, tensor_view, base_format
from bindings.sequence_optimizer_native import seq_binding, seq_device, seq_device_get, seq_cast
from bindings.sequence_device_native import seq_device_make, seq_device_input
from bindings.classical_arrays import classical_input, classical_typed, classical_address, classical_dtype_str
from bindings.classical_attributes import classical_connect_numpy


def _b() raises -> PythonObject:
    return Python.import_module("builtins")


def _text(value: PythonObject) raises -> String:
    return String(py=py_call(_b().str, value))


def _ints(values: List[Int]) raises -> PythonObject:
    var out = Python.list()
    for i in range(len(values)):  # driver-loop: over shape words
        _ = py_call(py_call(_b().getattr, out, "append"), PythonObject(values[i]))
    return py_call(_b().tuple, out)


# ---------------------------------------------------------------- inputs and outputs

def _f32(value: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    """`np.asarray(a)`, float64 refused by name, as C-ordered float32."""
    var input = context["input_context"]
    classical_connect_numpy(input)
    var source = classical_input(value, input, name)
    if classical_dtype_str(source, input) == "<f8":
        raise Error("TypeError: " + name + ": float64 is refused; pass float32")
    var view = TensorView()
    _ = tensor_view(source, view)
    if base_format(view.fmt) == "f" and view.itemsize == 4 and view.c_contig:
        return source
    return seq_cast(source, context)


def _host(shape: List[Int], context: PythonObject) raises -> PythonObject:
    """A new float32 host array of `shape`."""
    var input = context["input_context"]
    classical_connect_numpy(input)
    return classical_typed(_ints(shape), "<f4", input)


def _shape(value: PythonObject, context: PythonObject) raises -> List[Int]:
    if seq_device(value, context):
        return device_shape(seq_device_get(value, "shape", context), context)
    return device_shape(py_call(_b().tuple, value.shape), context)


def _new(entries: PythonObject, shape: List[Int], context: PythonObject) raises -> PythonObject:
    """`SequenceDeviceTensor._new(b, shape)`: an unwritten resident tensor."""
    var handle = py_call_typed(entries["seq_tensor_alloc"], PythonObject(max(1, shape_count(shape))), PythonObject(0))
    return seq_device_make(entries["module"], shape, handle, PythonObject(None), context)


def _wrap(entries: PythonObject, array: PythonObject, context: PythonObject) raises -> PythonObject:
    """`SequenceDeviceTensor._wrap(b, a)`: one upload."""
    var input = seq_device_input(array, False, context)
    var shape = device_shape(py_call(_b().tuple, input.shape), context)
    var n = shape_count(shape)
    var handle = py_call_typed(entries["seq_tensor_alloc"], PythonObject(max(n, 1)), PythonObject(0))
    try:
        if n > 0:
            _ = py_call_typed(entries["seq_tensor_upload"], handle, classical_address(input), PythonObject(n))
    except error:
        _ = py_call_typed(entries["seq_tensor_free"], handle)
        raise error^
    _ = input^
    return seq_device_make(entries["module"], shape, handle, PythonObject(None), context)


def _reshape(value: PythonObject, shape: List[Int]) raises -> PythonObject:
    return py_call(py_call(_b().getattr, value, "reshape"), _ints(shape))


def _addr_or_zero(value: PythonObject) raises -> PythonObject:
    return PythonObject(0) if value is None else classical_address(value)


# ---------------------------------------------------------------- LayerNorm

def _norm_d(normalized_shape: PythonObject, context: PythonObject) raises -> Int:
    """`_D(normalized_shape)[1]`."""
    var b = _b()
    var input = context["input_context"]
    classical_connect_numpy(input)
    var integer = Bool(py=py_call(b.isinstance, normalized_shape, b.int)) or Bool(py=py_call(b.isinstance, normalized_shape, input["int_types"]))
    var shape = Python.tuple(normalized_shape) if integer else py_call(b.tuple, normalized_shape)
    if len(shape) == 0:
        raise Error("ValueError: normalized_shape must be positive")
    var d = 1
    for i in range(len(shape)):  # driver-loop: over normalized shape words
        var v = Int(py=py_call_typed(b.int, shape[i]))
        if v < 1:
            raise Error("ValueError: normalized_shape must be positive")
        d *= v
    return d


def _param(value: PythonObject, name: String, d: Int, context: PythonObject) raises -> PythonObject:
    """`_f32(value, name).reshape(-1)` with its `d`-value check (None stays None)."""
    if value is None:
        return value
    var a = _f32(value, name, context)
    if Int(py=a.size) != d:
        raise Error("ValueError: layer_norm: " + name + " must hold " + String(d) + " values")
    return _reshape(a, [d])


def _ln_run_dev(x: PythonObject, d: Int, weight: PythonObject, bias: PythonObject, eps: Float64, dy: PythonObject,
                stats: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_run_dev`: (y, dx, dweight, dbias), the absent ones None."""
    var binding = seq_device_get(x, "b", context)
    var entries = seq_binding(PythonObject(None), context, binding)
    var entry = py_get(entries, "layer_norm_dev")
    if entry is None:
        raise Error("RuntimeError: layer_norm: this binding has no device LayerNorm; pass a NumPy x")
    var xs = _shape(x, context)
    var size = shape_count(xs)
    if len(xs) == 0 or xs[len(xs) - 1] == 0 or size % d != 0:
        raise Error("ValueError: layer_norm: the trailing dimensions must match normalized_shape")
    var m = size // d
    var w = _param(weight, "weight", d, context)
    var bb = _param(bias, "bias", d, context)
    var flags = Python.list(m, d, 0 if w is None else 1, 0 if bb is None else 1)
    var handles = Python.list()
    var stat_flag = Python.list()
    if stats is not None:
        var ok = (seq_device(stats[0], context) and seq_device(stats[1], context)
                  and seq_device_get(stats[0], "b", context) is binding and seq_device_get(stats[1], "b", context) is binding
                  and shape_count(_shape(stats[0], context)) == m and shape_count(_shape(stats[1], context)) == m)
        if not ok:
            raise Error("ValueError: layer_norm: stats must be two resident tensors of M floats on x's binding")
        handles = Python.list(seq_device_get(stats[0], "h", context), seq_device_get(stats[1], "h", context))
        stat_flag = Python.list(1)
    var xh = seq_device_get(x, "h", context)
    var fp = Python.list(eps)
    if dy is None:
        var y = _new(entries, xs, context)
        var hs = Python.list(xh, seq_device_get(y, "h", context), 0, 0)
        _ = py_call(py_call(_b().getattr, hs, "extend"), handles)
        var ip = py_call(_b().list, flags)
        _ = py_call(py_call(_b().getattr, ip, "append"), PythonObject(0))
        _ = py_call(py_call(_b().getattr, ip, "extend"), stat_flag)
        _ = py_call_typed(entry, hs, Python.list(_addr_or_zero(w), _addr_or_zero(bb), 0, 0), ip, fp)
        return Python.tuple(y, PythonObject(None), PythonObject(None), PythonObject(None))
    var dyt: PythonObject
    if seq_device(dy, context):
        if seq_device_get(dy, "b", context) is not binding or shape_count(_shape(dy, context)) != size or len(_shape(dy, context)) != len(xs):
            raise Error("ValueError: layer_norm_backward: dy must have x's shape (on x's binding)")
        var ds = _shape(dy, context)
        for i in range(len(xs)):  # driver-loop: over shape words
            if ds[i] != xs[i]:
                raise Error("ValueError: layer_norm_backward: dy must have x's shape (on x's binding)")
        dyt = dy
    else:
        var dyv = _f32(dy, "dy", context)
        var ds = _shape(dyv, context)
        var same = len(ds) == len(xs)
        if same:
            for i in range(len(xs)):  # driver-loop: over shape words
                if ds[i] != xs[i]:
                    same = False
        if not same:
            raise Error("ValueError: layer_norm_backward: dy must have x's shape")
        dyt = _wrap(entries, dyv, context)
    var dx = _new(entries, xs, context)
    var dw = _host([d], context)   # written by the binding when weight is given
    var db = _host([d], context)   # written by the binding when bias is given
    var hs = Python.list(xh, 0, seq_device_get(dyt, "h", context), seq_device_get(dx, "h", context))
    _ = py_call(py_call(_b().getattr, hs, "extend"), handles)
    var ip = py_call(_b().list, flags)
    _ = py_call(py_call(_b().getattr, ip, "append"), PythonObject(1))
    _ = py_call(py_call(_b().getattr, ip, "extend"), stat_flag)
    _ = py_call_typed(entry, hs, Python.list(_addr_or_zero(w), _addr_or_zero(bb), classical_address(dw), classical_address(db)), ip, fp)
    return Python.tuple(PythonObject(None), dx, dw, db)


def _ln_run(x: PythonObject, d: Int, weight: PythonObject, bias: PythonObject, eps: Float64, dy: PythonObject,
            mode: PythonObject, stats: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_run`: (y, dx, dweight, dbias)."""
    if seq_device(x, context):
        return _ln_run_dev(x, d, weight, bias, eps, dy, stats, context)
    var a = _f32(x, "x", context)
    var xs = _shape(a, context)
    var size = shape_count(xs)
    if len(xs) == 0 or xs[len(xs) - 1] == 0 or size % d != 0:
        raise Error("ValueError: layer_norm: the trailing dimensions must match normalized_shape")
    var m = size // d
    var w = _param(weight, "weight", d, context)
    var bb = _param(bias, "bias", d, context)
    var backward = dy is not None
    # backward never reads y; y and dx are written whole by the binding
    var y = PythonObject(None) if backward else _host(xs, context)
    var dyv = PythonObject(None)
    if backward:
        dyv = _f32(dy, "dy", context)
        var ds = _shape(dyv, context)
        var same = len(ds) == len(xs)
        if same:
            for i in range(len(xs)):  # driver-loop: over shape words
                if ds[i] != xs[i]:
                    same = False
        if not same:
            raise Error("ValueError: layer_norm_backward: dy must have x's shape")
    var dx = _host(xs, context) if backward else _host([1], context)
    var dw = _host([d], context)
    var db = _host([d], context)
    var entries = seq_binding(mode, context, PythonObject(None))
    var entry = py_get(entries, "layer_norm")
    if entry is None:
        raise Error("ImportError: the sequence binding lacks layer_norm; rebuild it")
    _ = py_call_typed(entry, Python.list(classical_address(a), _addr_or_zero(w), _addr_or_zero(bb), _addr_or_zero(y),
                                          _addr_or_zero(dyv), classical_address(dx), classical_address(dw), classical_address(db)),
                      Python.list(m, d, 0 if w is None else 1, 0 if bb is None else 1, 1 if backward else 0), Python.list(eps))
    _ = a^
    return Python.tuple(y, dx, dw, db)


def _ln_x(value: PythonObject, context: PythonObject) raises -> PythonObject:
    """`x if is_seq_tensor(x) else _f32(x, "x")`."""
    if seq_device(value, context):
        return value
    return _f32(value, "x", context)


def _ln_width(x: PythonObject, normalized_shape: PythonObject, context: PythonObject) raises -> Int:
    if normalized_shape is None:
        var xs = _shape(x, context)
        if len(xs) == 0:
            raise Error("IndexError: tuple index out of range")
        return _norm_d(PythonObject(xs[len(xs) - 1]), context)
    return _norm_d(normalized_shape, context)


def sequence_layer_norm(operation: String, owner: PythonObject, values: PythonObject, mode: PythonObject,
                        context: PythonObject) raises -> PythonObject:
    var b = _b()
    if operation == "seqln_forward":
        # values: (x, normalized_shape, weight, bias, eps)
        var x = _ln_x(values[0], context)
        var d = _ln_width(x, values[1], context)
        return _ln_run(x, d, values[2], values[3], Float64(py=py_call_typed(b.float, values[4])), PythonObject(None),
                       mode, PythonObject(None), context)[0]
    if operation == "seqln_backward":
        # values: (dy, x, normalized_shape, weight, bias, eps)
        var x = _ln_x(values[1], context)
        var d = _ln_width(x, values[2], context)
        var result = _ln_run(x, d, values[3], values[4], Float64(py=py_call_typed(b.float, values[5])), values[0],
                             mode, PythonObject(None), context)
        return Python.tuple(result[1], PythonObject(None) if values[3] is None else result[2],
                            PythonObject(None) if values[4] is None else result[3])
    # `LayerNorm.forward(x)`
    var fields = py_fields(owner)
    var x = _ln_x(values[0], context)
    var xs = _shape(x, context)
    var ns = fields["normalized_shape"]
    var k = len(ns)
    var same = len(xs) >= k
    if same:
        for i in range(k):  # driver-loop: over normalized shape words
            if xs[len(xs) - k + i] != Int(py=ns[i]):
                same = False
    if not same:
        raise Error("ValueError: LayerNorm: trailing shape " + _text(_ints(xs)) + " does not end in " + _text(ns))
    py_setitem(fields, "_x", x)
    py_setitem(fields, "_stats", PythonObject(None))
    var d = Int(py=fields["_D"])
    var stats = PythonObject(None)
    if seq_device(x, context) and not Bool(py=context["seqln_keep_stats_off"]):
        var entries = seq_binding(PythonObject(None), context, seq_device_get(x, "b", context))
        var m = shape_count(xs) // d
        stats = Python.tuple(_new(entries, [m], context), _new(entries, [m], context))
        py_setitem(fields, "_stats", stats)
    return _ln_run(x, d, fields["weight"], fields["bias"], Float64(py=fields["eps"]), PythonObject(None),
                   py_get(fields, "numeric_mode"), stats, context)[0]


# ---------------------------------------------------------------- MoEBlock

def _moe_check_weights(fields: PythonObject, e: Int, d: Int, f: Int) raises:
    """`_check_weights()`: the three expert weight shapes."""
    var names = Python.tuple("router", "gate_up_proj", "down_proj")
    var shapes = Python.tuple(Python.tuple(e, d), Python.tuple(e, 2 * f, d), Python.tuple(e, d, f))
    for i in range(3):  # driver-loop: over the three named expert weight arrays
        var have = py_call(_b().tuple, fields[names[i]].shape)
        if not Bool(py=have == shapes[i]):
            raise Error("ValueError: MoEBlock: " + String(py=names[i]) + " has shape " + _text(have) + ", expected " + _text(shapes[i]))


def _moe_weights(fields: PythonObject, entries: PythonObject, e: Int, d: Int, f: Int,
                 context: PythonObject) raises -> PythonObject:
    """`_weights_handle(b)`: the resident copies, put again after any weight
    assignment (the old copies freed)."""
    var module = entries["module"]
    var cur = py_get(fields, "_wdev")
    var version = py_get(fields, "_wver", PythonObject(0))
    if cur is not None and cur[0] is module and Bool(py=cur[1] == version):
        return cur[2]
    # `_free_weights()`
    py_setitem(fields, "_wdev", PythonObject(None))
    if cur is not None:
        try:
            var owner = seq_binding(PythonObject(None), context, cur[0])
            _ = py_call_typed(owner["moe_weights_free"], cur[2])
        except:
            pass
    var h = py_call_typed(entries["moe_weights_put"], Python.list(classical_address(fields["router"]),
                          classical_address(fields["gate_up_proj"]), classical_address(fields["down_proj"])),
                          Python.list(e, d, f))
    var handle = PythonObject(Int(py=h))
    py_setitem(fields, "_wdev", Python.tuple(module, version, handle))
    return handle


def sequence_moe_forward(owner: PythonObject, values: PythonObject, context: PythonObject) raises -> PythonObject:
    """`MoEBlock.forward(x)`."""
    var b = _b()
    var fields = py_fields(owner)
    var e = Int(py=fields["E"])
    var d = Int(py=fields["D"])
    var f = Int(py=fields["F"])
    var k = Int(py=fields["k"])
    var norm = 1 if Bool(py=fields["norm_topk_prob"]) else 0
    var x = values[0]
    if seq_device(x, context):
        var binding = seq_device_get(x, "b", context)
        var entries = seq_binding(PythonObject(None), context, binding)
        if py_get(entries, "moe_forward_dev") is not None and py_get(entries, "moe_weights_put") is not None:
            var xs = _shape(x, context)
            var size = shape_count(xs)
            if len(xs) < 1 or xs[len(xs) - 1] != d or size == 0:
                raise Error("ValueError: MoEBlock: x must have shape (..., " + String(d) + ")")
            var t = size // d
            _moe_check_weights(fields, e, d, f)
            var h = _moe_weights(fields, entries, e, d, f, context)
            # every cell of the four outputs is stored by the route and the combine: no fill
            var y = _new(entries, xs, context)
            var logits = _new(entries, [t, e], context)
            var sel = _new(entries, [t, k], context)
            var w = _new(entries, [t, k], context)
            _ = py_call_typed(entries["moe_forward_dev"], Python.list(seq_device_get(x, "h", context), seq_device_get(y, "h", context),
                              seq_device_get(logits, "h", context), seq_device_get(sel, "h", context), seq_device_get(w, "h", context)),
                              Python.list(t, d, f, e, k, norm, h))
            py_setitem(fields, "_routing", Python.list(logits, sel, w, PythonObject(None)))
            return y
        # no device MoE entry: x is read back and runs the host-array entry
        var input = context["input_context"]
        classical_connect_numpy(input)
        var shape = _shape(x, context)
        var host = classical_typed(_ints(shape), "<f4", input)
        var n = shape_count(shape)
        if n > 0:
            _ = py_call_typed(entries["seq_tensor_download"], seq_device_get(x, "h", context), classical_address(host), PythonObject(n))
        x = host
    var a = _f32(x, "MoEBlock", context)
    var shape = _shape(a, context)
    var size = shape_count(shape)
    if size % d != 0:
        raise Error("ValueError: cannot reshape array of size " + String(size) + " into shape (-1," + String(d) + ")")
    var t = size // d
    var y = _host([t, d], context)
    var logits = _host([t, e], context)
    var sel = _host([t, k], context)
    var w = _host([t, k], context)
    var entries = seq_binding(py_get(fields, "numeric_mode"), context, PythonObject(None))
    var ip = Python.list(t, d, f, e, k, norm)
    var xa = classical_address(a)
    if py_get(entries, "moe_weights_put") is not None:
        _moe_check_weights(fields, e, d, f)
        var h = _moe_weights(fields, entries, e, d, f, context)
        _ = py_call(py_call(b.getattr, ip, "append"), h)
        _ = py_call_typed(entries["moe_forward"], Python.list(xa, xa, xa, xa, classical_address(y), classical_address(logits),
                          classical_address(sel), classical_address(w)), ip)
    else:
        var r = _f32(fields["router"], "router", context)
        var gu = _f32(fields["gate_up_proj"], "gate_up_proj", context)
        var dn = _f32(fields["down_proj"], "down_proj", context)
        _ = py_call_typed(entries["moe_forward"], Python.list(xa, classical_address(r), classical_address(gu), classical_address(dn),
                          classical_address(y), classical_address(logits), classical_address(sel), classical_address(w)), ip)
    _ = a^
    py_setitem(fields, "_routing", Python.list(logits, sel, w, PythonObject(None)))
    return _reshape(y, shape)


def sequence_layers_operation(operation: String, owner: PythonObject, values: PythonObject, mode: PythonObject,
                              context: PythonObject) raises -> PythonObject:
    if operation == "seqmoe_forward":
        return sequence_moe_forward(owner, values, context)
    return sequence_layer_norm(operation, owner, values, mode, context)
