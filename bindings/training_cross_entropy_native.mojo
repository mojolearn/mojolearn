# SPDX-License-Identifier: Apache-2.0
# Public loss preparation, native routing and result ownership; kernels unchanged.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_fields
from bindings.optimizer_admission import opt_value, opt_numpy
from bindings.optimizer_public_door import TensorView, tensor_view, base_format, new_array, contiguous_addr, refuse_kwargs
from bindings.optimizer_device_native import device_entries, device_shape, device_allocate, shape_count
from bindings.training_embedding_native import embedding_indices, embedding_leaves, embedding_native_call
from bindings.training_scalar_native import training_scalar_number, training_scalar_float
from bindings.training_loss_options import loss_truth, loss_reduction
from bindings.native_numpy_metadata import native_numpy_type
from bindings.sequence_dtype_native import sequence_dtype_classes, sequence_dtype_instance, sequence_dtype_create


def loss_buffer_dtype(value: PythonObject, context: PythonObject) raises:
    if not opt_numpy(py_type(value), context):
        return
    var dtype = value.dtype
    var admitted = sequence_dtype_instance(dtype, sequence_dtype_classes(context))
    for code in ["q", "Q"]:  # driver-loop: native long-long descriptor aliases
        admitted = admitted or py_type(dtype) is py_type(sequence_dtype_create(PythonObject(code), context))
    if not admitted:
        raise Error("TypeError: cross_entropy buffers require builtin native dtype metadata")


def loss_targets(value: PythonObject, device: Bool, context: PythonObject) raises -> PythonObject:
    var input = value
    loss_buffer_dtype(value, context)
    var view = TensorView()
    if not tensor_view(value, view):
        var dimensions = List[Int]()
        var depth = -1
        var leaves = Python.list()
        try:
            embedding_leaves(value, dimensions, 0, depth, leaves, context)
        except error:
            # The host wrapper translated failed Array.from_list to TypeError.
            if not device:
                raise Error("TypeError: cross_entropy targets require an integer array or a list of ints")
            raise error^
        if len(dimensions) != 1:
            if device:
                raise Error("TypeError: cross_entropy targets must be a 1-D integer array")
            raise Error("ValueError: cross_entropy targets must be 1-D class indices")
        var b = Python.import_module("builtins")
        for i in range(len(leaves)):  # driver-loop: native integer-storage leaf admission
            var cls = py_type(leaves[i])
            var builtin = cls is b.int or cls is b.bool or cls is b.float
            var numpy = opt_numpy(cls, context)
            if not builtin and not numpy:
                raise Error("TypeError: cross_entropy target lists require native numeric leaves")
            if numpy and cls is native_numpy_type("ndarray", context):
                raise Error("TypeError: cross_entropy target lists require native scalar leaves")
        # array('q', ...) preserves integer-only list construction, including
        # NumPy integer scalars, and its old TypeError for floats/overflow.
        input = embedding_native_call(context["buffer_types"][2], Python.tuple("q", leaves), True)
        _ = leaves^
        _ = tensor_view(input, view)
    var format = base_format(view.fmt)
    var integer = format == "b" or format == "B" or format == "h" or format == "H" or format == "i" or format == "I" or format == "l" or format == "L" or format == "q" or format == "Q" or format == "n" or format == "N"
    if len(view.shape) != 1:
        if device:
            raise Error("TypeError: cross_entropy targets must be a 1-D integer array")
        raise Error("ValueError: cross_entropy targets must be 1-D class indices")
    if not integer:
        raise Error("TypeError: cross_entropy targets require integer class indices, not probabilities")
    return embedding_indices(input, context)


def training_cross_entropy(values: PythonObject, entries: PythonObject, context: PythonObject) raises -> PythonObject:
    if len(values) != 8:
        raise Error("TypeError: invalid cross_entropy argument envelope")
    refuse_kwargs(values[7], "cross_entropy")
    var reduction = loss_reduction(values[3], context)
    if loss_truth(values[6], context) and reduction == 0:
        raise Error("NotImplementedError: cross_entropy reduction='none' has no backward")
    var logits = values[0]
    var device = py_type(logits) is context["device_class"]
    var native = entries
    var entry = entries["ce_loss"]
    var fields = Python.dict()
    var shape = List[Int]()
    var view = TensorView()
    if device:
        opt_value(logits, context)
        native = device_entries(entries["module"], context)
        fields = py_fields(logits)
        if fields["_b"] is not entries["module"]:
            raise Error("ValueError: cross_entropy device logits must belong to the selected native binding")
        shape = device_shape(fields["shape"], context)
        if shape_count(shape) != Int(py=fields["size"]):
            raise Error("ValueError: cross_entropy device shape does not match its stored element count")
        entry = py_get(native, "ce_loss_dev")
        if entry is None:
            raise Error("RuntimeError: this build has no native device cross_entropy entry")
    else:
        loss_buffer_dtype(logits, context)
        if not tensor_view(logits, view):
            raise Error("TypeError: cross_entropy logits require a float32 buffer")
        shape = view.shape.copy()
        if len(shape) != 2:
            raise Error("ValueError: cross_entropy logits must be 2-D (N,V)")
        var format = base_format(view.fmt)
        if format == "e":
            raise Error("NotImplementedError: cross_entropy mixed precision is not covered")
        if format != "f" or view.itemsize != 4:
            raise Error("TypeError: cross_entropy logits require native float32")
        if shape_count(shape) == 0:
            raise Error("ValueError: cross_entropy logits must be nonempty")
    if len(shape) != 2:
        raise Error("ValueError: cross_entropy logits must be 2-D (N,V)")
    if py_type(entry) is not context["device_finalizer_types"][2]:
        raise Error("TypeError: cross_entropy requires its connected compiled entry")
    var rows = shape[0]
    var vocab = shape[1]
    var targets = loss_targets(values[1], device, context)
    if Int(py=targets[2]) != rows:
        raise Error("ValueError: cross_entropy logits and targets row counts differ")
    var smoothing = PythonObject(None)
    if not device:
        smoothing = training_scalar_float(values[4], "cross_entropy", context)
    var loss = new_array([1], "<f4")
    var row_loss = new_array([rows], "<f4")
    var want_grad = loss_truth(values[6], context)
    var gradient = PythonObject(None)
    if device:
        if want_grad:
            gradient = device_allocate(native, shape, context)
    elif want_grad:
        gradient = new_array(shape, "<f4")
    else:
        gradient = new_array([1], "<f4")
    var ignored = training_scalar_number(values[2], "cross_entropy", context, True)
    var divisor = training_scalar_number(PythonObject(0) if values[5] is None else values[5], "cross_entropy", context, True)
    var params = Python.list(rows, vocab, ignored, reduction, divisor, 1 if loss_truth(values[6], context) else 0)
    if device:
        smoothing = training_scalar_float(values[4], "cross_entropy", context)
    py_call(params.append, smoothing)
    var keep = List[List[Float32]]()
    if device:
        var output_handle = fields["h"] if gradient is None else py_fields(gradient)["h"]
        _ = py_call(entry, loss._addr, row_loss._addr, targets[1], Python.list(output_handle, fields["h"]), params)
    else:
        var address = contiguous_addr(logits, view, keep)
        _ = py_call(entry, loss._addr, row_loss._addr, gradient._addr, address, targets[1], params)
    var result = PythonObject(None)
    if reduction == 0:
        result = row_loss
    else:
        var pointer = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=loss._addr))
        result = PythonObject(Float64(pointer[0]))
    if loss_truth(values[6], context):
        result = Python.tuple(result, gradient)
    _ = keep^
    _ = targets^
    _ = logits^
    _ = loss^
    _ = row_loss^
    _ = gradient^
    return result
