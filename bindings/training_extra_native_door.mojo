# SPDX-License-Identifier: Apache-2.0
"""Native public training functions routed through the optimizer native door
(bindings/optimizer_native_router.mojo): `residual_dropout`,
`residual_dropout_backward` (python/mojolearn/_residual_dropout_impl.py),
`chunked_lm_head_loss` and `mlp_inference_sessions`
(python/mojolearn/_training_impl.py). Each is one call: the buffer
admission and float32 / int32 preparation, the shape refusals in the Python
bodies' words, the outputs, the compiled training entry with the same
argument words, and the result. `mlp_inference_sessions` runs the small-MLP
program's forward (autograd/small_mlp_program.mojo, op SESSIONS).
"""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.optimizer_container_boundary_unpack import OptimizerInputOwners
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_fields
from bindings.py_typed_call import py_call_typed
from bindings.optimizer_public_door import tensor_view, new_array
from bindings.tensor_view import TensorView, base_format, shape_text
from training.optimizer_input_plan import optimizer_contiguous_address
from bindings.optimizer_device_native_door import shape_count, device_entries, device_shape, device_allocate, device_float32
from bindings.optimizer_admission_door import opt_value
from bindings.training_embedding_native_door import embedding_indices
from bindings.training_layers_device_unpack import training_device_array, chunked_lm_head_device
from bindings.py_cpython_table import cpython_table
from bindings.optimizer_keys_unpack import optimizer_keys
from bindings.optimizer_keys_unpack import OPTIMIZER_OPN_RESIDUAL_DROPOUT
from bindings.py_balanced import py_item
from bindings.optimizer_descriptor_boundary_door import optimizer_provider_entries
from providers.python.neural_descriptor_pyapi_door import neural_descriptor_execute
from training.optimizer_native_api import (
    OPT_NATIVE_RESIDUAL_DROPOUT, OPT_NATIVE_RESIDUAL_DROPOUT_BACKWARD, OPT_NATIVE_RESIDUAL_DROPOUT_DEV,
    OPT_NATIVE_RESIDUAL_DROPOUT_BACKWARD_DEV, OPT_NATIVE_LM_HEAD_LOSS, OPT_NATIVE_LM_HEAD_TRAIN,
    OPT_NATIVE_MLP_SESSIONS,
)



def _training_call(entries: PythonObject, operation: Int, addresses: List[Int], ints: List[Int],
                   floats: List[Float64]) raises -> Int:
    """One call of the training image's native table (`optimizer_native_api_v1`;
    training/optimizer_native_extra.mojo) on typed words, the GIL released for
    the kernel as the Python entries released it. Lane dw-autograd-seq
    (2026-11-21): before, each of these was the image's Python entry of the
    same name, looked up by name in `entries` and called with Python lists."""
    return neural_descriptor_execute(optimizer_provider_entries(entries), operation, addresses, ints, floats)


# ---------------------------------------------------------------- residual dropout

def _rd_operand(value: PythonObject, name: String, mut view: TensorView, mut buffer_pins: OptimizerInputOwners) raises -> Int:
    """`_residual_dropout_impl._operand`: a C-contiguous native float32
    buffer; its element count."""
    if not tensor_view(value, view, buffer_pins):
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
        cpython_table()[].PyErr_Clear()
        raise Error("ValueError: " + name + " is outside its unsigned integer range")
    if bits < 64 and word >= (UInt64(1) << UInt64(bits)):
        raise Error("ValueError: " + name + " is outside its unsigned integer range")
    _ = index
    return word


def _rd_params(count: Int, p: PythonObject, seed: PythonObject, stream: PythonObject,
               offset: PythonObject, mut ints: List[Int], mut floats: List[Float64]) raises:
    """`_residual_dropout_impl._params`: [count, offset, seed low, seed high,
    stream] appended to `ints`, p to `floats`."""
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
    ints.append(count)
    ints.append(Int(off))
    ints.append(Int(s & UInt64(4294967295)))
    ints.append(Int(s >> UInt64(32)))
    ints.append(Int(st))
    floats.append(Float64(py=py_call_typed(b.float, p)))


def _rd_device_operand(value: PythonObject, name: String, entries: PythonObject, context: PythonObject,
                       mut shape: List[Int]) raises -> PythonObject:
    """A training device array of the identical binding: its fields; its
    shape into `shape`."""
    var nk = optimizer_keys()
    if py_type(value) is not py_item(context, nk[].k_device_class):
        raise Error("TypeError: mojolearn: " + name + " must be a device array like the first operand")
    opt_value(value, context)
    var fields = py_fields(value)
    if py_item(fields, nk[].k__b) is not py_item(entries, nk[].k_module):
        raise Error("ValueError: residual_dropout device arrays must belong to the identical training binding")
    device_float32(fields, name, "residual_dropout")
    shape = device_shape(py_item(fields, nk[].k_shape), context)
    var count = shape_count(shape)
    if count != Int(py=py_item(fields, nk[].k_size)):
        raise Error("ValueError: residual_dropout device shape does not match its stored element count")
    if count < 1:
        raise Error("ValueError: residual_dropout device arrays must be nonempty")
    if count > 2147483647:
        raise Error("ValueError: " + name + " exceeds the native element index range")
    return fields


def _rd_device(operation: Int, values: PythonObject, entries: PythonObject,
               context: PythonObject) raises -> PythonObject:
    """Lane ne3-c-blocks-embedding (row N117): residual dropout on training
    device arrays. The outputs are new device arrays of the same binding and
    the layer runs on the resident handles (`residual_dropout_dev` /
    `residual_dropout_backward_dev`): no upload, no download. The same
    parameter words and kernels as the host-array call."""
    var nk = optimizer_keys()
    var native = device_entries(py_item(entries, nk[].k_module), context)
    var shape = List[Int]()
    var first = _rd_device_operand(values[0], "values" if operation == OPTIMIZER_OPN_RESIDUAL_DROPOUT else "gradient",
                                   entries, context, shape)
    var count = shape_count(shape)
    if operation == OPTIMIZER_OPN_RESIDUAL_DROPOUT:
        var residual_shape = List[Int]()
        var second = _rd_device_operand(values[1], "residual", entries, context, residual_shape)
        if len(residual_shape) != len(shape):
            raise Error("ValueError: values and residual must have identical shapes")
        for i in range(len(shape)):  # driver-loop: over shape words of the two operands
            if residual_shape[i] != shape[i]:
                raise Error("ValueError: values and residual must have identical shapes")
        var ints = List[Int](capacity=8)
        var floats = List[Float64](capacity=1)
        _rd_params(count, values[2], values[3], values[4], values[5], ints, floats)
        var result = device_allocate(native, shape, context)
        ints.append(Int(py=py_item(py_fields(result), nk[].k_h)))
        ints.append(Int(py=py_item(first, nk[].k_h)))
        ints.append(Int(py=py_item(second, nk[].k_h)))
        var written = _training_call(native, OPT_NATIVE_RESIDUAL_DROPOUT_DEV, List[Int](), ints, floats)
        if written != count:
            raise Error("RuntimeError: residual_dropout returned an incomplete output")
        return result
    var ints = List[Int](capacity=8)
    var floats = List[Float64](capacity=1)
    _rd_params(count, values[1], values[2], values[3], values[4], ints, floats)
    var d_values = device_allocate(native, shape, context)
    var d_residual = device_allocate(native, shape, context)
    ints.append(Int(py=py_item(py_fields(d_values), nk[].k_h)))
    ints.append(Int(py=py_item(py_fields(d_residual), nk[].k_h)))
    ints.append(Int(py=py_item(first, nk[].k_h)))
    var written = _training_call(native, OPT_NATIVE_RESIDUAL_DROPOUT_BACKWARD_DEV, List[Int](), ints, floats)
    if written != count:
        raise Error("RuntimeError: residual_dropout_backward returned incomplete gradients")
    return Python.tuple(d_values, d_residual)


def training_residual_dropout(operation: Int, values: PythonObject, entries: PythonObject,
                              context: PythonObject) raises -> PythonObject:
    """values: (values, residual, p, seed, stream, offset) for the forward,
    (gradient, p, seed, stream, offset) for the backward; identical mode.
    Training device arrays run on the device (`_rd_device`)."""
    var buffer_pins = OptimizerInputOwners()
    try:
        var nk = optimizer_keys()
        if py_type(values[0]) is py_item(context, nk[].k_device_class):
            return _rd_device(operation, values, entries, context)
        if operation == OPTIMIZER_OPN_RESIDUAL_DROPOUT:
            var vv = TensorView()
            var rv = TensorView()
            var count = _rd_operand(values[0], "values", vv, buffer_pins)
            var residual_count = _rd_operand(values[1], "residual", rv, buffer_pins)
            if len(rv.shape) != len(vv.shape) or residual_count != count:
                raise Error("ValueError: values and residual must have identical shapes")
            for i in range(len(vv.shape)):  # driver-loop: over shape words of the two operands
                if vv.shape[i] != rv.shape[i]:
                    raise Error("ValueError: values and residual must have identical shapes")
            var ints = List[Int](capacity=5)
            var floats = List[Float64](capacity=1)
            _rd_params(count, values[2], values[3], values[4], values[5], ints, floats)
            var result = new_array(vv.shape, "<f4")
            var addresses: List[Int] = [vv.addr if count else 0, rv.addr if count else 0,
                                        Int(py=result._addr) if count else 0]
            var written = _training_call(entries, OPT_NATIVE_RESIDUAL_DROPOUT, addresses, ints, floats)
            _ = rv  # keep the Python object alive through the native call above
            _ = vv  # keep the Python object alive through the native call above
            _ = result  # keep the Python object alive through the native call above
            if written != count:
                raise Error("RuntimeError: residual_dropout returned an incomplete output")
            _ = result  # keep the Python object alive through the native calls above
            _ = rv  # keep the Python object alive through the native calls above
            _ = vv  # keep the Python object alive through the native calls above
            return result
        var gv = TensorView()
        var count = _rd_operand(values[0], "gradient", gv, buffer_pins)
        var ints = List[Int](capacity=5)
        var floats = List[Float64](capacity=1)
        _rd_params(count, values[1], values[2], values[3], values[4], ints, floats)
        var d_values = new_array(gv.shape, "<f4")
        var d_residual = new_array(gv.shape, "<f4")
        var addresses: List[Int] = [gv.addr if count else 0, Int(py=d_values._addr) if count else 0,
                                    Int(py=d_residual._addr) if count else 0]
        var written = _training_call(entries, OPT_NATIVE_RESIDUAL_DROPOUT_BACKWARD, addresses, ints, floats)
        _ = d_values  # keep the Python object alive through the native call above
        _ = gv  # keep the Python object alive through the native call above
        _ = d_residual  # keep the Python object alive through the native call above
        if written != count:
            raise Error("RuntimeError: residual_dropout_backward returned incomplete gradients")
        _ = d_residual  # keep the Python object alive through the native calls above
        _ = d_values  # keep the Python object alive through the native calls above
        _ = gv  # keep the Python object alive through the native calls above
        return Python.tuple(d_values, d_residual)


    finally:
        _ = buffer_pins^

# ---------------------------------------------------------------- float32 operands


def _c32(value: PythonObject, name: String, site: String, mut view: TensorView,
         mut keep: List[List[Float32]], mut buffer_pins: OptimizerInputOwners) raises -> Int:
    """`_training_impl._c32`: a native float32 buffer, refused by name
    otherwise and when empty; the address of its values in C order."""
    if not tensor_view(value, view, buffer_pins) or base_format(view.fmt) != "f" or view.itemsize != 4:
        raise Error("TypeError: mojolearn." + site + ": " + name + " must be float32")
    if len(view.shape) == 0 and not view.is_array:
        view.shape.append(1)
        view.strides.append(4)
    if shape_count(view.shape) == 0:
        var what = String("feature(s)") if len(view.shape) == 2 and view.shape[0] != 0 else String("sample(s)")
        var text = shape_text(view.shape)
        raise Error("ValueError: mojolearn: " + name + " is empty, shape " + text + ": Found array with 0 " + what
                    + " (shape=" + text + ") while a minimum of 1 is required.")
    with GILReleased(Python()):
        return optimizer_contiguous_address(view, keep)


# ---------------------------------------------------------------- the chunked LM head

def training_chunked_lm_head(values: PythonObject, entries: PythonObject, context: PythonObject) raises -> PythonObject:
    """values: (hidden, weight, targets, return_grad)."""
    var buffer_pins = OptimizerInputOwners()
    try:
        var b = native_loaded_module("builtins")
        var site = String("training.chunked_lm_head_loss")
        # lane vdf-linalg3: TrainDeviceArray hidden / weight run the device
        # entry in place (bindings/training_layers_device_unpack.mojo); two
        # identity compares on the host path
        if training_device_array(values[0], context) or training_device_array(values[1], context):
            var dev_targets = embedding_indices(values[2], context, buffer_pins)
            var dev_result = chunked_lm_head_device(values[0], values[1], Int(py=dev_targets[1]),
                                                    Int(py=dev_targets[2]),
                                                    Bool(py=py_call_typed(b.bool, values[3])), entries, context)
            _ = dev_targets^
            return dev_result
        var keep = List[List[Float32]]()
        var hv = TensorView()
        var wv = TensorView()
        var ha = _c32(values[0], "hidden", site, hv, keep, buffer_pins)
        var wa = _c32(values[1], "weight", site, wv, keep, buffer_pins)
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
        var targets = embedding_indices(values[2], context, buffer_pins)
        var count = Int(py=targets[2])
        if count != rows:
            raise Error("ValueError: mojolearn." + site + ": " + String(count) + " targets for " + String(rows) + " rows")
        var loss = new_array([1], "<f4")
        var row_max = new_array([rows], "<f4")
        var row_denom = new_array([rows], "<f4")
        var plist: List[Int] = [rows, vocab, width]
        var no_floats = List[Float64]()
        var loss_address = Int(py=loss._addr)
        var target_address = Int(py=targets[1])
        var result: PythonObject
        if not Bool(py=py_call_typed(b.bool, values[3])):
            var addresses: List[Int] = [loss_address, Int(py=row_max._addr), Int(py=row_denom._addr), ha, wa,
                                        target_address]
            _ = _training_call(entries, OPT_NATIVE_LM_HEAD_LOSS, addresses, plist, no_floats)
            _ = row_denom  # keep the Python object alive through the native call above
            _ = row_max  # keep the Python object alive through the native call above
            _ = loss  # keep the Python object alive through the native call above
            result = PythonObject(Float64(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=loss_address)[0]))
            _ = loss  # keep the Python object alive through the native call above
        else:
            var d_hidden = new_array([rows, width], "<f4")
            var d_weight = new_array([vocab, width], "<f4")
            var addresses: List[Int] = [loss_address, Int(py=row_max._addr), Int(py=row_denom._addr),
                                        Int(py=d_hidden._addr), Int(py=d_weight._addr), ha, wa, target_address]
            _ = _training_call(entries, OPT_NATIVE_LM_HEAD_TRAIN, addresses, plist, no_floats)
            _ = d_weight  # keep the Python object alive through the native call above
            _ = d_hidden  # keep the Python object alive through the native call above
            _ = row_denom  # keep the Python object alive through the native call above
            _ = row_max  # keep the Python object alive through the native call above
            _ = loss  # keep the Python object alive through the native call above
            result = Python.tuple(Float64(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=loss_address)[0]),
                                  d_hidden, d_weight)
            _ = loss  # keep the Python object alive through the native call above
        _ = keep^
        _ = targets^
        return result


    finally:
        _ = buffer_pins^

# ---------------------------------------------------------------- MLP inference sessions


def training_mlp_sessions(values: PythonObject, entries: PythonObject) raises -> PythonObject:
    """values: (inputs, weight1, bias1, weight2, bias2)."""
    var buffer_pins = OptimizerInputOwners()
    try:
        var b = native_loaded_module("builtins")
        var site = String("mlp_inference_sessions")
        var keep = List[List[Float32]]()
        var w1v = TensorView()
        var b1v = TensorView()
        var w2v = TensorView()
        var b2v = TensorView()
        var w1 = _c32(values[1], "weight1", site, w1v, keep, buffer_pins)
        var b1 = _c32(values[2], "bias1", site, b1v, keep, buffer_pins)
        var w2 = _c32(values[3], "weight2", site, w2v, keep, buffer_pins)
        var b2 = _c32(values[4], "bias2", site, b2v, keep, buffer_pins)
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
        # Typed words for the one native call (were a Python list of shape
        # tuples and a Python list of addresses, two appends per session):
        # addresses = [output, w1, b1, w2, b2, one per session];
        # ints = [width, hidden, out width, one row count per session].
        var addresses = List[Int](capacity=5 + len(inputs))
        addresses.append(0)
        addresses.append(w1)
        addresses.append(b1)
        addresses.append(w2)
        addresses.append(b2)
        var ints = List[Int](capacity=3 + len(inputs))
        ints.append(width)
        ints.append(hidden)
        ints.append(out_width)
        var views = List[TensorView]()
        for i in range(len(inputs)):  # driver-loop: over the caller's session arrays
            var view = TensorView()
            var address = _c32(inputs[i], "session input", site, view, keep, buffer_pins)
            addresses.append(address)
            views.append(view^)
        # TOMBSTONE: MOJOLEARN_MLP_SESSIONS_PACKED (opt-in arm, RUN OWED, no A/B recorded) deleted 2026-11-15 by lane/expdel2-02-base; code recoverable at 1c0608ab17.
        # Restore: git apply experiments/removed/MOJOLEARN_MLP_SESSIONS_PACKED.patch; record in docs/TOMBSTONES.md.
        # DEFAULT (restored 2026-11-09 by integration lane i2-integrate-seven
        # from 0ce88411d3): one native call into the training binding's
        # `mlp_sessions`, one forward per session (training/neural_session_mlp
        # .mojo). The output is sized here from the views already read above
        # (the `mlp_sessions_rows` rule: at least one session, every session a
        # (rows, width) matrix), not by a second call that re-reads the shape
        # tuples.
        if len(views) == 0:
            raise Error("ValueError: MLP sessions require at least one (rows, width) input")
        var total_rows = 0
        for i in range(len(views)):  # driver-loop: over the caller's session arrays (shape words, not data)
            if len(views[i].shape) != 2 or views[i].shape[1] != width or views[i].shape[0] < 0:
                raise Error("ValueError: MLP sessions require at least one (rows, width) input")
            total_rows += views[i].shape[0]
            ints.append(views[i].shape[0])
        var output = new_array([total_rows, out_width], "<f4")
        addresses[0] = Int(py=output._addr)
        var no_floats = List[Float64]()
        var wrote = _training_call(entries, OPT_NATIVE_MLP_SESSIONS, addresses, ints, no_floats)
        _ = output  # keep the Python object alive through the native call above
        if wrote != total_rows * out_width:
            raise Error("RuntimeError: MLP sessions returned an invalid output count")
        _ = output  # keep the Python object alive through the native calls above
        _ = keep^
        _ = views^
        _ = inputs^
        return output
    finally:
        _ = buffer_pins^
