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
from bindings.py_balanced import py_call, py_type, py_get
from bindings.py_typed_call import py_call_typed
from bindings.optimizer_public_door import TensorView, tensor_view, contiguous_addr, base_format, new_array, py_ints, shape_text
from bindings.optimizer_device_native import shape_count
from bindings.training_embedding_native import embedding_indices
from bindings.native_error_class import error_without_class


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


def _rd_uint(value: PythonObject, name: String, bits: Int) raises -> PythonObject:
    """`_residual_dropout_impl._uint`."""
    var b = Python.import_module("builtins")
    if py_type(value) is b.bool:
        raise Error("TypeError: " + name + " must be an unsigned integer")
    var index = py_call_typed(Python.import_module("operator").index, value)
    if Bool(py=index < PythonObject(0)) or Bool(py=index >= PythonObject(1).__lshift__(PythonObject(bits))):
        raise Error("ValueError: " + name + " is outside its unsigned integer range")
    return index


def _rd_params(count: Int, p: PythonObject, seed: PythonObject, stream: PythonObject,
               offset: PythonObject) raises -> PythonObject:
    """`_residual_dropout_impl._params`: [count, offset, seed low, seed high,
    stream, p]."""
    var b = Python.import_module("builtins")
    var s = _rd_uint(seed, "seed", 64)
    var st = _rd_uint(stream, "stream", 32)
    var off = _rd_uint(offset, "offset", 63)
    var limit = PythonObject(1).__lshift__(PythonObject(63)) - PythonObject(1) - PythonObject(count)
    if Bool(py=off > limit):
        raise Error("ValueError: offset plus element count exceeds the native Philox range")
    if py_type(p) is b.bool:
        raise Error("TypeError: p must be a real dropout probability")
    # p admission, FP32 rounding and scale arithmetic are the entry's
    return Python.list(count, off, s.__and__(PythonObject(4294967295)), s.__rshift__(PythonObject(32)), st,
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
        if Int(py=written) != count:
            raise Error("RuntimeError: residual_dropout returned an incomplete output")
        return result
    var gv = TensorView()
    var count = _rd_operand(values[0], "gradient", gv)
    var params = _rd_params(count, values[1], values[2], values[3], values[4])
    var d_values = new_array(gv.shape, "<f4")
    var d_residual = new_array(gv.shape, "<f4")
    var written = py_call_typed(_entry(entries, "residual_dropout_backward", missing),
                                PythonObject(gv.addr if count else 0), d_values._addr if count else PythonObject(0),
                                d_residual._addr if count else PythonObject(0), params)
    if Int(py=written) != count:
        raise Error("RuntimeError: residual_dropout_backward returned incomplete gradients")
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
    var b = Python.import_module("builtins")
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
        result = PythonObject(Float64(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=loss._addr))[0]))
    else:
        var d_hidden = new_array([rows, width], "<f4")
        var d_weight = new_array([vocab, width], "<f4")
        _ = py_call_typed(_entry(entries, "chunked_lm_head_v2_train", "mojolearn: this training binding lacks "),
                          Python.list(loss._addr, row_max._addr, row_denom._addr, d_hidden._addr, d_weight._addr,
                                      ha, wa, targets[1]), plist)
        result = Python.tuple(Float64(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=loss._addr))[0]),
                              d_hidden, d_weight)
    _ = keep^
    _ = targets^
    return result


# ---------------------------------------------------------------- MLP inference sessions

def training_mlp_sessions(values: PythonObject, entries: PythonObject) raises -> PythonObject:
    """values: (inputs, weight1, bias1, weight2, bias2)."""
    var b = Python.import_module("builtins")
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
    for i in range(len(inputs)):  # driver-loop: over the caller's session arrays
        var view = TensorView()
        var address = _c32(inputs[i], "session input", site, view, keep)
        _ = py_call(py_call(b.getattr, shapes, "append"), py_ints(view.shape))
        _ = py_call(py_call(b.getattr, addresses, "append"), PythonObject(address))
        views.append(view^)
    var total_rows = 0
    try:
        total_rows = Int(py=py_call_typed(_entry(entries, "mlp_sessions_rows", "mojolearn: this training binding lacks "),
                                          shapes, PythonObject(width)))
    except error:
        if String(error).startswith("ImportError:"):
            raise error^
        raise Error("ValueError: " + String(error_without_class(error, "ValueError")))
    var output = new_array([total_rows, out_width], "<f4")
    var wrote = py_call_typed(_entry(entries, "mlp_sessions", "mojolearn: this training binding lacks "),
                              Python.list(output._addr, w1, b1, w2, b2), addresses, shapes,
                              Python.list(width, hidden, out_width))
    if Int(py=wrote) != total_rows * out_width:
        raise Error("RuntimeError: MLP sessions returned an invalid output count")
    _ = keep^
    _ = views^
    _ = inputs^
    return output
