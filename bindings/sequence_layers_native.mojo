# SPDX-License-Identifier: Apache-2.0
# Whole public LayerNorm operations and the sequence-layer dispatcher. Input
# shape/storage/ownership are pinned once and reused for admission and execution.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_contains
from bindings.py_typed_call import py_call_typed
from bindings.mode_metadata_native import _fields, _class_fields
from bindings.optimizer_public_door import py_ints, same_shape
from bindings.optimizer_device_native import device_shape, shape_count
from bindings.optimizer_admission import opt_scalar
from bindings.sequence_optimizer_native import seq_binding
from bindings.sequence_device_native import seq_device_make
from bindings.classical_arrays import classical_address
from bindings.classical_attributes import classical_connect_numpy
from providers.python.sequence_layer_buffer_pyapi import SequenceFloatBuffer, SequenceResidentSnapshot, _array
from bindings.sequence_moe_native import sequence_moe_operation
from core.scaler_fit_impl import scaler_fill


def _b() raises -> PythonObject:
    return native_loaded_module("builtins")


def _new(entries: PythonObject, shape: List[Int], context: PythonObject) raises -> PythonObject:
    var handle = py_call_typed(entries["seq_tensor_alloc"], PythonObject(max(1, shape_count(shape))), PythonObject(0))
    return seq_device_make(entries["module"], shape, handle, PythonObject(None), context)


def _handle(value: PythonObject, context: PythonObject) raises -> PythonObject:
    # Only freshly allocated, native-owned outputs are read here.
    return py_call(context["sequence_device_getters"]["h"], value)


struct LayerInput(Movable):
    var owner: PythonObject
    var host: Optional[SequenceFloatBuffer]
    var resident: Optional[SequenceResidentSnapshot]
    var shape: List[Int]
    var count: Int
    var address: Int
    var handle: PythonObject
    var entries: PythonObject
    var device: Bool
    var absent: Bool

    def __init__(out self, value: PythonObject, context: PythonObject) raises:
        self.owner = value
        self.host = None
        self.resident = None
        self.shape = List[Int]()
        self.count = 0
        self.address = 0
        self.handle = PythonObject(None)
        self.entries = PythonObject(None)
        self.device = False
        self.absent = value is None
        if value is None:
            return
        if py_type(value) is context["sequence_device_class"]:
            self.resident = SequenceResidentSnapshot(value, context)
            self.device = True
            self.shape = self.resident.value().shape.copy()
            self.count = self.resident.value().count
            self.handle = self.resident.value().handle
            self.entries = self.resident.value().entries
        else:
            self.host = SequenceFloatBuffer(value, context)
            self.owner = self.host.value().array
            self.shape = self.host.value().shape.copy()
            self.count = self.host.value().count
            self.address = self.host.value().address


def _upload(input: LayerInput, entries: PythonObject, context: PythonObject) raises -> PythonObject:
    var output = _new(entries, input.shape, context)
    if input.count > 0:
        _ = py_call_typed(entries["seq_tensor_upload"], _handle(output, context), PythonObject(input.address), PythonObject(input.count))
    return output
def _norm_shape(normalized_shape: PythonObject, context: PythonObject) raises -> List[Int]:
    opt_scalar(normalized_shape, context)
    var b = _b()
    var input = context["input_context"]
    classical_connect_numpy(input)
    var integer = Bool(py=py_call(b.isinstance, normalized_shape, b.int)) or Bool(py=py_call(b.isinstance, normalized_shape, input["int_types"]))
    var shape = Python.tuple(normalized_shape) if integer else py_call(b.tuple, normalized_shape)
    if len(shape) == 0:
        raise Error("ValueError: normalized_shape must be positive")
    var dimensions = List[Int]()
    for i in range(len(shape)):  # small-loop(normalized shape words)
        var v = Int(py=py_call_typed(b.int, shape[i]))
        if v < 1:
            raise Error("ValueError: normalized_shape must be positive")
        dimensions.append(v)
    return dimensions^


def _norm_d(normalized_shape: PythonObject, context: PythonObject) raises -> Int:
    return shape_count(_norm_shape(normalized_shape, context))


def _ln_fields(owner: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = _b()
    var layout = context["seqln_layout"]
    var cls = py_type(owner)
    if cls is not layout[0] or py_type(cls) is not b.type:
        raise Error("TypeError: LayerNorm requires its connected native storage class")
    if py_call(b.type.__dict__["__mro__"].__get__, cls) is not layout[1]:
        raise Error("TypeError: LayerNorm class ancestry changed")
    var hooks = layout[3]
    for i in range(len(hooks)):  # small-loop(connected native storage hooks)
        var row = py_call(b.type.__dict__["__dict__"].__get__, hooks[i][0])
        if py_get(row, hooks[i][1]) is not hooks[i][2]:
            raise Error("TypeError: LayerNorm refuses replaced Python storage hooks")
    var ancestry = layout[1]
    for i in range(len(ancestry)):  # driver-loop: over connected native owner classes
        var namespace = _class_fields(ancestry[i])
        for name in ["normalized_shape", "_D", "eps", "numeric_mode", "weight", "bias", "weight_grad", "bias_grad", "_x", "_stats"]:  # driver-loop: over native layer state schema
            if py_contains(namespace, name):
                raise Error("TypeError: LayerNorm state cannot be replaced by class descriptors")
    var fields = _fields(owner)
    var keys = py_call(b.list, fields)
    for i in range(len(keys)):  # small-loop(instance metadata keys)
        if py_type(keys[i]) is not b.str:
            raise Error("TypeError: LayerNorm fields require exact string names")
    return fields


def _ln_device_entries(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    if Bool(py=context["seqln_class_device_io_off"]):
        return PythonObject(None)
    try:
        var entries = seq_binding(py_get(fields, "numeric_mode"), context, PythonObject(None))
        for name in ["seq_tensor_alloc", "seq_tensor_free", "seq_tensor_upload", "seq_tensor_download", "layer_norm_dev"]:  # small-loop(required resident native entries)
            if py_get(entries, name) is None:
                return PythonObject(None)
        return entries
    except:
        # Preserve the existing class-level host fallback on installations
        # without a resident binding. This performs no Python computation.
        return PythonObject(None)



def _width(input: LayerInput, normalized_shape: PythonObject, context: PythonObject) raises -> Int:
    if input.absent or len(input.shape) == 0:
        raise Error("ValueError: layer_norm input requires at least one dimension")
    var d = input.shape[len(input.shape) - 1] if normalized_shape is None else _norm_d(normalized_shape, context)
    if d < 1 or d > 2147483647 or input.shape[len(input.shape) - 1] == 0 or input.count % d != 0:
        raise Error("ValueError: layer_norm: trailing dimensions must match normalized_shape")
    return d


def _parameter(value: LayerInput, d: Int, name: String) raises:
    if not value.absent and (value.device or value.count != d):
        raise Error("ValueError: layer_norm: " + name + " must hold " + String(d) + " host values")


def _run(input: LayerInput, d: Int, weight: PythonObject, bias: PythonObject, eps: Float64,
         dy: PythonObject, mode: PythonObject, stats: PythonObject, grad_shape: List[Int],
         context: PythonObject) raises -> PythonObject:
    var w = LayerInput(weight, context)
    var b = LayerInput(bias, context)
    _parameter(w, d, "weight")
    _parameter(b, d, "bias")
    var gradient = LayerInput(dy, context)
    var backward = not gradient.absent
    var had_weight = not w.absent
    var had_bias = not b.absent
    if backward and not same_shape(gradient.shape, input.shape):
        raise Error("ValueError: layer_norm_backward: dy must have x's shape")
    var m = input.count // d
    var flags = Python.list(m, d, 0 if w.absent else 1, 0 if b.absent else 1, 1 if backward else 0)
    var fp = Python.list(eps)
    if input.device:
        var entry = py_get(input.entries, "layer_norm_dev")
        if entry is None:
            raise Error("RuntimeError: layer_norm: this binding has no device LayerNorm; pass a host input")
        var means = PythonObject(None)
        var inverse = PythonObject(None)
        if stats is not None:
            if (py_type(stats) is not _b().tuple and py_type(stats) is not _b().list) or len(stats) != 2:
                raise Error("TypeError: layer_norm stats require two native tensor handles")
            means = stats[0]
            inverse = stats[1]
        var mean = LayerInput(means, context)
        var rstd = LayerInput(inverse, context)
        var handles = Python.list()
        if stats is not None:
            if not mean.device or not rstd.device or mean.count != m or rstd.count != m:
                raise Error("ValueError: layer_norm: stats must be two resident tensors of M floats")
            if mean.entries["module"] is not input.entries["module"] or rstd.entries["module"] is not input.entries["module"]:
                raise Error("ValueError: layer_norm: stats must use x's native binding")
            handles = Python.list(mean.handle, rstd.handle)
            _ = py_call(flags.append, PythonObject(1))
        var output = _new(input.entries, input.shape, context)
        var dy_owner = PythonObject(None)
        var dy_handle = PythonObject(0)
        var dw = PythonObject(None)
        var db = PythonObject(None)
        if backward:
            if gradient.device:
                if gradient.entries["module"] is not input.entries["module"]:
                    raise Error("ValueError: layer_norm_backward: dy must use x's native binding")
                dy_handle = gradient.handle
            else:
                dy_owner = _upload(gradient, input.entries, context)
                dy_handle = _handle(dy_owner, context)
            dw = _array(grad_shape, context)
            db = _array(grad_shape, context)
        var hs = Python.list(input.handle, PythonObject(0) if backward else _handle(output, context),
                             dy_handle, _handle(output, context) if backward else PythonObject(0))
        _ = py_call(hs.extend, handles)
        var addrs = Python.list(w.address, b.address, classical_address(dw) if backward else PythonObject(0),
                                classical_address(db) if backward else PythonObject(0))
        _ = py_call_typed(entry, hs, addrs, flags, fp)
        _ = dy_owner^
        _ = mean^
        _ = rstd^
        _ = gradient^
        _ = w^
        _ = b^
        return Python.tuple(PythonObject(None) if backward else output, output if backward else PythonObject(None),
            dw if had_weight else PythonObject(None), db if had_bias else PythonObject(None))
    if gradient.device:
        raise Error("TypeError: layer_norm host input requires a host output gradient")
    var entries = seq_binding(mode, context, PythonObject(None))
    var entry = py_get(entries, "layer_norm")
    if entry is None:
        raise Error("ImportError: the sequence binding lacks layer_norm; rebuild it")
    var output = _array(input.shape, context)
    var scratch = _array([1], context) if not backward else PythonObject(None)
    var dw = _array(grad_shape, context)
    var db = _array(grad_shape, context)
    var addrs = Python.list(input.address, w.address, b.address,
        PythonObject(0) if backward else classical_address(output), gradient.address,
        classical_address(output) if backward else classical_address(scratch), classical_address(dw), classical_address(db))
    _ = py_call_typed(entry, addrs, flags, fp)
    _ = scratch^
    _ = gradient^
    _ = w^
    _ = b^
    return Python.tuple(PythonObject(None) if backward else output, output if backward else PythonObject(None),
            dw if had_weight else PythonObject(None), db if had_bias else PythonObject(None))


def sequence_layer_norm(operation: String, owner: PythonObject, values: PythonObject, mode: PythonObject,
                        context: PythonObject) raises -> PythonObject:
    if operation == "seqln_forward" or operation == "seqln_backward":
        var backward = operation == "seqln_backward"
        var offset = 1 if backward else 0
        opt_scalar(values[offset + 1], context)
        opt_scalar(values[offset + 4], context)
        opt_scalar(mode, context)
        var input = LayerInput(values[offset], context)
        var d = _width(input, values[offset + 1], context)
        var result = _run(input, d, values[offset + 2], values[offset + 3], Float64(py=py_call_typed(_b().float, values[offset + 4])),
            values[0] if backward else PythonObject(None), mode, PythonObject(None), [d], context)
        _ = input^
        if not backward:
            return result[0]
        return Python.tuple(result[1], PythonObject(None) if values[3] is None else result[2],
                            PythonObject(None) if values[4] is None else result[3])
    var fields = _ln_fields(owner, context)
    if operation == "seqln_initialize":
        opt_scalar(values, context)
        opt_scalar(mode, context)
        var shape = _norm_shape(values[0], context)
        var d = shape_count(shape)
        if d > 2147483647:
            raise Error("ValueError: normalized_shape exceeds native indexing")
        var affine = Bool(py=values[2])
        var with_bias = affine and Bool(py=values[3])
        var weight = PythonObject(None)
        var bias = PythonObject(None)
        if affine:
            weight = _array(shape, context)
            scaler_fill(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(weight))), d, Float32(1))
        if with_bias:
            bias = _array(shape, context)
        py_setitem(fields, "normalized_shape", py_ints(shape))
        py_setitem(fields, "_D", d)
        py_setitem(fields, "eps", Float64(py=values[1]))
        py_setitem(fields, "elementwise_affine", affine)
        py_setitem(fields, "weight", weight)
        py_setitem(fields, "bias", bias)
        py_setitem(fields, "numeric_mode", mode)
        for name in ["weight_grad", "bias_grad", "_x", "_stats"]:  # driver-loop: over initial native state fields
            py_setitem(fields, name, PythonObject(None))
        return PythonObject(None)
    if operation == "seqln_device_binding" or operation == "seqln_device_io":
        var entries = _ln_device_entries(fields, context)
        return PythonObject(entries is not None) if operation == "seqln_device_io" else (
            PythonObject(None) if entries is None else entries["module"])
    if operation == "seqln_to_device":
        var input = LayerInput(values[0], context)
        if input.absent:
            raise Error("TypeError: layer_norm input requires a native array")
        if input.device:
            return input.owner
        var entries = _ln_device_entries(fields, context)
        var result = input.owner if entries is None else _upload(input, entries, context)
        _ = input^
        return result
    for name in ["normalized_shape", "_D", "eps", "numeric_mode"]:  # driver-loop: over layer configuration metadata
        opt_scalar(fields[PythonObject(name)], context)
    var d = Int(py=fields["_D"])
    var shape = _norm_shape(fields["normalized_shape"], context)
    if d < 1 or d > 2147483647 or d != shape_count(shape):
        raise Error("ValueError: normalized_shape and native layer width must agree")
    if operation == "seqln_layer_backward":
        var x = py_get(fields, "_x")
        if x is None:
            raise Error("RuntimeError: LayerNorm.backward: call forward first")
        var input = LayerInput(x, context)
        _ = _width(input, PythonObject(d), context)
        var result = _run(input, d, fields["weight"], fields["bias"], Float64(py=fields["eps"]), values[0],
            py_get(fields, "numeric_mode"), py_get(fields, "_stats"), shape, context)
        _ = input^
        py_setitem(fields, "weight_grad", result[2])
        py_setitem(fields, "bias_grad", result[3])
        return result[1]
    if operation != "seqln_layer":
        raise Error("ValueError: unknown LayerNorm public operation")
    var input = LayerInput(values[0], context)
    _ = _width(input, PythonObject(d), context)
    var k = len(shape)
    var same = len(input.shape) >= k
    if same:
        for i in range(k):  # driver-loop: over normalized shape words
            same = same and input.shape[len(input.shape) - k + i] == shape[i]
    if not same:
        raise Error("ValueError: LayerNorm input trailing shape differs from normalized_shape")
    var stats = PythonObject(None)
    if input.device and not Bool(py=context["seqln_keep_stats_off"]):
        var m = input.count // d
        stats = Python.tuple(_new(input.entries, [m], context), _new(input.entries, [m], context))
    # Keep the same normalized input owner and resident statistics for the
    # next public backward; its metadata is refreshed at that next operation.
    py_setitem(fields, "_x", input.owner)
    py_setitem(fields, "_stats", stats)
    var result = _run(input, d, fields["weight"], fields["bias"], Float64(py=fields["eps"]), PythonObject(None),
        py_get(fields, "numeric_mode"), stats, shape, context)
    _ = input^
    return result[0]


def sequence_layers_operation(operation: String, owner: PythonObject, values: PythonObject, mode: PythonObject,
                              context: PythonObject) raises -> PythonObject:
    if operation.startswith("seqmoe_"):
        return sequence_moe_operation(operation, owner, values, mode, context)
    return sequence_layer_norm(operation, owner, values, mode, context)
