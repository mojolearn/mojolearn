# SPDX-License-Identifier: Apache-2.0
"""Native public training functions routed through the optimizer native door
(bindings/optimizer_native_router.mojo): `residual_dropout`,
`residual_dropout_backward` (python/mojolearn/_residual_dropout_impl.py),
`chunked_lm_head_loss` and `mlp_inference_sessions`
(python/mojolearn/_training_impl.py). Each is one call: the buffer
admission and float32 / int32 preparation, the shape refusals in the Python
bodies' words, the outputs, the compiled training entry with the same
argument words, and the result. The kernels and their order are unchanged,
so every result keeps its bits.
"""
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get
from bindings.py_typed_call import py_call_typed
from bindings.optimizer_public_door import TensorView, tensor_view, contiguous_addr, base_format, new_array, py_ints, shape_text
from bindings.optimizer_device_native import shape_count
from bindings.training_embedding_native import embedding_indices


def _entry(entries: PythonObject, name: String, missing: String) raises -> PythonObject:
    """A connected compiled training entry, refused by name when this build
    lacks it."""
    var entry = py_get(entries, PythonObject(name))
    if entry is None:
        raise Error("ImportError: " + missing + name)
    return entry


# ---------------------------------------------------------------- residual dropout

def _rd_operand(value: PythonObject, name: String, mut view: TensorView) raises -> Int:
    """`_residual_dropout_impl._operand`: a C-contiguous native float32
    buffer; its element count."""
    if not tensor_view(value, view):
        raise Error("TypeError: mojolearn: " + name + " does not support the buffer protocol")
    if base_format(view.fmt) != "f" or view.itemsize != 4 or not view.c_contig:
        raise Error("ValueError: " + name + " must be a C-contiguous native float32 buffer")
    var count = shape_count(view.shape)
    if count > 2147483647:
        raise Error("ValueError: " + name + " exceeds the native element index range")
    return count


def _rd_uint(value: PythonObject, name: String, bits: Int, b: PythonObject,
             index_fn: PythonObject) raises -> UInt64:
    """`_residual_dropout_impl._uint`, read into a native word (lane
    p3-neural-blocks: the range checks and the low/high split were about
    twelve Python integer operations per call). `b` and `index_fn`
    (builtins and operator.index) are looked up once by the caller; an exact
    int skips operator.index. Same refusals: a bool is a TypeError, a value
    outside [0, 2**bits) a ValueError."""
    if py_type(value) is b.bool:
        raise Error("TypeError: " + name + " must be an unsigned integer")
    var index = value if py_type(value) is b.int else py_call_typed(index_fn, value)
    ref cpy = Python().cpython()
    var word = cpy.lib.get_function[UInt64]("PyLong_AsUnsignedLongLong")(index._obj_ptr)
    if word == UInt64(0xFFFFFFFFFFFFFFFF) and cpy.PyErr_Occurred():
        # Negative or wider than 64 bits.
        cpy.lib.get_function[NoneType]("PyErr_Clear")()
        raise Error("ValueError: " + name + " is outside its unsigned integer range")
    if bits < 64 and word >= (UInt64(1) << UInt64(bits)):
        raise Error("ValueError: " + name + " is outside its unsigned integer range")
    _ = index
    return word


def _rd_params(count: Int, p: PythonObject, seed: PythonObject, stream: PythonObject,
               offset: PythonObject) raises -> PythonObject:
    """`_residual_dropout_impl._params`: [count, offset, seed low, seed high,
    stream, p]."""
    var b = native_loaded_module("builtins")
    var index_fn = native_loaded_module("operator").index
    var s = _rd_uint(seed, "seed", 64, b, index_fn)
    var st = _rd_uint(stream, "stream", 32, b, index_fn)
    var off = _rd_uint(offset, "offset", 63, b, index_fn)
    # offset < 2**63 here, so the comparison against 2**63 - 1 - count is exact
    # in signed words (count is a nonnegative element count).
    if Int(off) > 9223372036854775807 - count:
        raise Error("ValueError: offset plus element count exceeds the native Philox range")
    if py_type(p) is b.bool:
        raise Error("TypeError: p must be a real dropout probability")
    # p admission, FP32 rounding and scale arithmetic are the entry's
    return Python.list(count, Int(off), Int(s & UInt64(4294967295)), Int(s >> UInt64(32)), Int(st),
                       py_call_typed(b.float, p))


def training_residual_dropout(operation: String, values: PythonObject, entries: PythonObject) raises -> PythonObject:
    """values: (values, residual, p, seed, stream, offset) for the forward,
    (gradient, p, seed, stream, offset) for the backward; identical mode."""
    var missing = String("residual dropout requires updated training binding: missing ")
    if operation == "residual_dropout":
        var vv = TensorView()
        var rv = TensorView()
        var count = _rd_operand(values[0], "values", vv)
        var residual_count = _rd_operand(values[1], "residual", rv)
        if len(rv.shape) != len(vv.shape) or residual_count != count:
            raise Error("ValueError: values and residual must have identical shapes")
        for i in range(len(vv.shape)):  # driver-loop: over shape words of the two operands
            if vv.shape[i] != rv.shape[i]:
                raise Error("ValueError: values and residual must have identical shapes")
        var params = _rd_params(count, values[2], values[3], values[4], values[5])
        var result = new_array(vv.shape, "<f4")
        var written = py_call_typed(_entry(entries, "residual_dropout", missing),
                                    PythonObject(vv.addr if count else 0), PythonObject(rv.addr if count else 0),
                                    result._addr if count else PythonObject(0), params)
        _ = rv  # keep the Python object alive through the native call above
        _ = vv  # keep the Python object alive through the native call above
        _ = result  # keep the Python object alive through the native call above
        if Int(py=written) != count:
            raise Error("RuntimeError: residual_dropout returned an incomplete output")
        _ = result  # keep the Python object alive through the native calls above
        _ = rv  # keep the Python object alive through the native calls above
        _ = vv  # keep the Python object alive through the native calls above
        return result
    var gv = TensorView()
    var count = _rd_operand(values[0], "gradient", gv)
    var params = _rd_params(count, values[1], values[2], values[3], values[4])
    var d_values = new_array(gv.shape, "<f4")
    var d_residual = new_array(gv.shape, "<f4")
    var written = py_call_typed(_entry(entries, "residual_dropout_backward", missing),
                                PythonObject(gv.addr if count else 0), d_values._addr if count else PythonObject(0),
                                d_residual._addr if count else PythonObject(0), params)
    _ = d_values  # keep the Python object alive through the native call above
    _ = gv  # keep the Python object alive through the native call above
    _ = d_residual  # keep the Python object alive through the native call above
    if Int(py=written) != count:
        raise Error("RuntimeError: residual_dropout_backward returned incomplete gradients")
    _ = d_residual  # keep the Python object alive through the native calls above
    _ = d_values  # keep the Python object alive through the native calls above
    _ = gv  # keep the Python object alive through the native calls above
    return Python.tuple(d_values, d_residual)


# ---------------------------------------------------------------- float32 operands

def _c32(value: PythonObject, name: String, site: String, mut view: TensorView,
         mut keep: List[List[Float32]]) raises -> Int:
    """`_training_impl._c32`: a native float32 buffer, refused by name
    otherwise and when empty; the address of its values in C order."""
    if not tensor_view(value, view) or base_format(view.fmt) != "f" or view.itemsize != 4:
        raise Error("TypeError: mojolearn." + site + ": " + name + " must be float32")
    if len(view.shape) == 0 and not view.is_array:
        view.shape.append(1)
        view.strides.append(4)
    if shape_count(view.shape) == 0:
        var what = String("feature(s)") if len(view.shape) == 2 and view.shape[0] != 0 else String("sample(s)")
        var text = shape_text(view.shape)
        raise Error("ValueError: mojolearn: " + name + " is empty, shape " + text + ": Found array with 0 " + what
                    + " (shape=" + text + ") while a minimum of 1 is required.")
    return contiguous_addr(value, view, keep)


# ---------------------------------------------------------------- the chunked LM head

def training_chunked_lm_head(values: PythonObject, entries: PythonObject, context: PythonObject) raises -> PythonObject:
    """values: (hidden, weight, targets, return_grad)."""
    var b = native_loaded_module("builtins")
    var site = String("training.chunked_lm_head_loss")
    var keep = List[List[Float32]]()
    var hv = TensorView()
    var wv = TensorView()
    var ha = _c32(values[0], "hidden", site, hv, keep)
    var wa = _c32(values[1], "weight", site, wv, keep)
    if len(hv.shape) != 2 or len(wv.shape) != 2:
        raise Error("ValueError: mojolearn." + site + ": hidden must be (rows, width) and weight (vocab, width)")
    var rows = hv.shape[0]
    var width = hv.shape[1]
    var vocab = wv.shape[0]
    if width != wv.shape[1]:
        raise Error("ValueError: mojolearn." + site + ": hidden has width " + String(width) + ", weight has width "
                    + String(wv.shape[1]))
    if vocab < 2:
        raise Error("ValueError: mojolearn." + site + ": vocab must be at least 2, got " + String(vocab))
    var targets = embedding_indices(values[2], context)
    var count = Int(py=targets[2])
    if count != rows:
        raise Error("ValueError: mojolearn." + site + ": " + String(count) + " targets for " + String(rows) + " rows")
    var loss = new_array([1], "<f4")
    var row_max = new_array([rows], "<f4")
    var row_denom = new_array([rows], "<f4")
    var plist = Python.list(rows, vocab, width)
    var result: PythonObject
    if not Bool(py=py_call_typed(b.bool, values[3])):
        _ = py_call_typed(_entry(entries, "chunked_lm_head_v2_loss", "mojolearn: this training binding lacks "),
                          Python.list(loss._addr, row_max._addr, row_denom._addr, ha, wa, targets[1]), plist)
        _ = row_denom  # keep the Python object alive through the native call above
        _ = row_max  # keep the Python object alive through the native call above
        _ = loss  # keep the Python object alive through the native call above
        result = PythonObject(Float64(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=loss._addr))[0]))
        _ = loss  # keep the Python object alive through the native call above
    else:
        var d_hidden = new_array([rows, width], "<f4")
        var d_weight = new_array([vocab, width], "<f4")
        _ = py_call_typed(_entry(entries, "chunked_lm_head_v2_train", "mojolearn: this training binding lacks "),
                          Python.list(loss._addr, row_max._addr, row_denom._addr, d_hidden._addr, d_weight._addr,
                                      ha, wa, targets[1]), plist)
        _ = d_weight  # keep the Python object alive through the native call above
        _ = d_hidden  # keep the Python object alive through the native call above
        _ = row_denom  # keep the Python object alive through the native call above
        _ = row_max  # keep the Python object alive through the native call above
        _ = loss  # keep the Python object alive through the native call above
        result = Python.tuple(Float64(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=loss._addr))[0]),
                              d_hidden, d_weight)
        _ = loss  # keep the Python object alive through the native call above
    _ = keep^
    _ = targets^
    return result


# ---------------------------------------------------------------- MLP inference sessions

def training_mlp_sessions(values: PythonObject, entries: PythonObject) raises -> PythonObject:
    """values: (inputs, weight1, bias1, weight2, bias2)."""
    var b = native_loaded_module("builtins")
    var site = String("mlp_inference_sessions")
    var keep = List[List[Float32]]()
    var w1v = TensorView()
    var b1v = TensorView()
    var w2v = TensorView()
    var b2v = TensorView()
    var w1 = _c32(values[1], "weight1", site, w1v, keep)
    var b1 = _c32(values[2], "bias1", site, b1v, keep)
    var w2 = _c32(values[3], "weight2", site, w2v, keep)
    var b2 = _c32(values[4], "bias2", site, b2v, keep)
    if len(w1v.shape) != 2 or len(w2v.shape) != 2:
        raise Error("ValueError: MLP session weights must be matrices")
    var hidden = w1v.shape[0]
    var width = w1v.shape[1]
    var out_width = w2v.shape[0]
    if (w2v.shape[1] != hidden or len(b1v.shape) != 1 or b1v.shape[0] != hidden
            or len(b2v.shape) != 1 or b2v.shape[0] != out_width):
        raise Error("ValueError: MLP session weight/bias shapes do not agree")
    # the session inputs: shape tuples and C-order addresses, never row values
    var inputs = py_call(b.list, values[0])
    var shapes = Python.list()
    var addresses = Python.list()
    var views = List[TensorView]()
    # Bound append methods are read once, not twice per session array.
    var shapes_append = py_call(b.getattr, shapes, "append")
    var addresses_append = py_call(b.getattr, addresses, "append")
    for i in range(len(inputs)):  # driver-loop: over the caller's session arrays
        var view = TensorView()
        var address = _c32(inputs[i], "session input", site, view, keep)
        _ = py_call(shapes_append, py_ints(view.shape))
        _ = py_call(addresses_append, PythonObject(address))
        views.append(view^)
    # One native call into the training binding: the output is sized here
    # from the views already read above (the `mlp_sessions_rows` rule: at
    # least one session, every session a (rows, width) matrix), not by a
    # second call that re-reads the shape tuples. `mlp_sessions_rows` stays
    # exported for its other caller (bindings/small_mlp_native.mojo).
    var sessions_entry = _entry(entries, "mlp_sessions", "mojolearn: this training binding lacks ")
    if len(views) == 0:
        raise Error("ValueError: MLP sessions require at least one (rows, width) input")
    var total_rows = 0
    for i in range(len(views)):  # driver-loop: over the caller's session arrays (shape words, not data)
        if len(views[i].shape) != 2 or views[i].shape[1] != width or views[i].shape[0] < 0:
            raise Error("ValueError: MLP sessions require at least one (rows, width) input")
        total_rows += views[i].shape[0]
    var output = new_array([total_rows, out_width], "<f4")
    var wrote = py_call_typed(sessions_entry,
                              Python.list(output._addr, w1, b1, w2, b2), addresses, shapes,
                              Python.list(width, hidden, out_width))
    _ = output  # keep the Python object alive through the native call above
    if Int(py=wrote) != total_rows * out_width:
        raise Error("RuntimeError: MLP sessions returned an invalid output count")
    _ = output  # keep the Python object alive through the native calls above
    _ = keep^
    _ = views^
    _ = inputs^
    return output
