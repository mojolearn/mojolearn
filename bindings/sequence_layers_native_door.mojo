# SPDX-License-Identifier: Apache-2.0
# Whole public LayerNorm operations. Input shape/storage/ownership are pinned
# once and reused for admission and execution. The router
# (bindings/optimizer_native_router_door.mojo) calls `sequence_layer_norm` and
# `sequence_moe_operation` by the code's range (bw2-router-seq, 2026-11-20: the
# `seqmoe_` prefix dispatcher that sat here is deleted).
from bindings.native_class_admission_door import class_record, class_admit
from bindings.native_loaded_module_door import NativeModules
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_contains
from bindings.py_typed_call import py_call_typed
from bindings.mode_metadata_native_door import _fields, _class_fields
from training.optimizer_input_plan import same_shape
from providers.python.array_layout_boundary_pack import native_shape_pack
from bindings.optimizer_device_native_door import device_shape, shape_count
from bindings.optimizer_admission_door import opt_scalar
from bindings.sequence_optimizer_native_door import seq_binding, seq_device_bound, seq_device_refuse_resident
from providers.python.function_device_boundary_door import function_device_admit
from bindings.sequence_device_native_unpack import seq_device_make
from bindings.classical_arrays_door import classical_address
from bindings.classical_attributes_door import classical_connect_numpy
from providers.python.sequence_layer_buffer_pyapi_unpack import SequenceFloatBuffer, SequenceResidentSnapshot, _array, _array_unfilled
from bindings.optimizer_keys_unpack import (
    OPTIMIZER_OPN_SEQLN_BACKWARD,
    OPTIMIZER_OPN_SEQLN_DEVICE_IO,
    OPTIMIZER_OPN_SEQLN_FORWARD,
    OPTIMIZER_OPN_SEQLN_INITIALIZE,
    OPTIMIZER_OPN_SEQLN_LAYER,
    OPTIMIZER_OPN_SEQLN_LAYER_BACKWARD,
    OPTIMIZER_OPN_SEQLN_TO_DEVICE,
)
from core.scaler_fit_impl import scaler_fill
from bindings.sequence_keys_unpack import sequence_keys
from bindings.py_balanced import py_item
from bindings.sequence_ops_door import (
    seq_ops_call, seq_ops_device, seq_ops_native, seq_tensor_new, seq_tensor_release, seq_tensor_put,
)
from sequence.neural_model_dispatch import SEQ_LAYER_NORM, SEQ_LAYER_NORM_DEV, SEQ_LAYER_NORM_DEV_ALLOC


def _b() raises -> PythonObject:
    return native_loaded_module("builtins")


def _new(entries: PythonObject, shape: List[Int], context: PythonObject) raises -> PythonObject:
    var nk = sequence_keys()
    # The handle is this door's until its owner exists: released here if
    # the owner cannot be made.
    var handle = seq_tensor_new(entries, max(1, shape_count(shape)), False)
    try:
        return seq_device_make(py_item(entries, nk[].k_module), shape, PythonObject(handle), PythonObject(None), context)
    except error:
        seq_tensor_release(entries, handle)
        raise error^


def _release_made(entries: PythonObject, made: List[Int]):
    """Release the tensors a fused forward made that no owner took (a slot
    is zeroed when its owner exists)."""
    for i in range(len(made)):  # driver-loop: over one layer call's handle slots (at most six)
        if made[i] != 0:
            try:
                seq_tensor_release(entries, made[i])
            except:
                pass


def _handle(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = sequence_keys()
    # Only freshly allocated, native-owned outputs are read here.
    return py_call(py_item(py_item(context, nk[].k_sequence_device_getters), nk[].k_h), value)


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
        var nk = sequence_keys()
        if py_type(value) is py_item(context, nk[].k_sequence_device_class):
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
        seq_tensor_put(entries, Int(py=_handle(output, context)), input.address, input.count)
    return output
def _norm_shape(normalized_shape: PythonObject, context: PythonObject) raises -> List[Int]:
    opt_scalar(normalized_shape, context)
    var b = _b()
    var nk = sequence_keys()
    var input = py_item(context, nk[].k_input_context)
    classical_connect_numpy(input)
    var integer = Bool(py=py_call(b.isinstance, normalized_shape, b.int)) or Bool(py=py_call(b.isinstance, normalized_shape, py_item(input, nk[].k_int_types)))
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
    var nk = sequence_keys()
    var layout = py_item(context, nk[].k_seqln_layout)
    var cls = py_type(owner)
    if cls is not layout[0] or py_type(cls) is not b.type:
        raise Error("TypeError: LayerNorm requires its connected native storage class")
    # The ancestry, hook and descriptor checks read only the class rows and
    # the import-connected layout: kept per class (native_class_admission,
    # re-run when the class or an ancestor changes; every call on Python <
    # 3.12). Lane p3-neural-blocks.
    var mods = NativeModules()
    if class_record(mods, _LN_CLASS_SLOT, cls) is None:
        _ln_class_admit(cls, layout, b)
        class_admit(mods, _LN_CLASS_SLOT, cls, PythonObject(True))
    var fields = _fields(owner)
    var keys = py_call(b.list, fields)
    for i in range(len(keys)):  # small-loop(instance metadata keys)
        if py_type(keys[i]) is not b.str:
            raise Error("TypeError: LayerNorm fields require exact string names")
    return fields


comptime _LN_CLASS_SLOT = "seqln_class_admitted_v1"
#: LayerNorm has both sides: the device x_sequence image (`sequence/layernorm.mojo`,
#: `layernorm_dev.mojo`) and the CPU identical image's SEQ_LAYER_NORM (lane vdf-seq)
comptime LN_HAS_GPU = True
comptime LN_HAS_CPU = True


def _ln_class_admit(cls: PythonObject, layout: PythonObject, b: PythonObject) raises:
    var nk = sequence_keys()
    if py_call(py_item(b.type.__dict__, nk[].k___mro__).__get__, cls) is not layout[1]:
        raise Error("TypeError: LayerNorm class ancestry changed")
    var hooks = layout[3]
    for i in range(len(hooks)):  # small-loop(connected native storage hooks)
        var row = py_call(py_item(b.type.__dict__, nk[].k___dict__).__get__, hooks[i][0])
        if py_get(row, hooks[i][1]) is not hooks[i][2]:
            raise Error("TypeError: LayerNorm refuses replaced Python storage hooks")
    var ancestry = layout[1]
    for i in range(len(ancestry)):  # driver-loop: over connected native owner classes
        var namespace = _class_fields(ancestry[i])
        for name in ["normalized_shape", "_D", "eps", "numeric_mode", "weight", "bias", "weight_grad", "bias_grad", "_x", "_stats", "device", "_host_b"]:  # driver-loop: over native layer state schema
            if py_contains(namespace, name):
                raise Error("TypeError: LayerNorm state cannot be replaced by class descriptors")


def _ln_device_entries(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = sequence_keys()
    # TOMBSTONE: MOJOLEARN_LN_CLASS_DEVICE_IO_OFF (env; removed: decided A/B, post-merge 2026-10-10 NV 0.96 / AMD 1.03, noise = loser) deleted 2026-11-23 by lane/ab-prep-neural; code recoverable at 4d76a9e16e.
    # Restore: git apply ~/mojolearn-evidence/experiments-removed/MOJOLEARN_LN_CLASS_DEVICE_IO_OFF.patch; record in docs/TOMBSTONES.md.
    # TOMBSTONE: MOJOLEARN_SEQ_DEVICE_IO_OFF (env; removed: plumbing, the same kernels on host arrays vs resident tensors, rule 2026-11-22) deleted 2026-11-23 by lane/ab-owners; code recoverable at d125cb97c7.
    # Restore: git apply ~/mojolearn-evidence/experiments-removed/MOJOLEARN_SEQ_DEVICE_IO_OFF.patch; record in docs/TOMBSTONES.md.
    try:
        var entries = seq_binding(py_get(fields, nk[].k_numeric_mode), context, py_get(fields, nk[].k__host_b))
        # The image's route word (bindings/sequence_ops_door.mojo); was five
        # entry-name lookups.
        if not seq_ops_device(entries):
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
    return _width_known(input, d)


def _width_known(input: LayerInput, d: Int) raises -> Int:
    # `_width` for a layer width the operation already holds natively (the
    # fitted `_D`, checked against normalized_shape): the same refusals,
    # without boxing d and re-deriving it through Python int/isinstance.
    if input.absent or len(input.shape) == 0:
        raise Error("ValueError: layer_norm input requires at least one dimension")
    if d < 1 or d > 2147483647 or input.shape[len(input.shape) - 1] == 0 or input.count % d != 0:
        raise Error("ValueError: layer_norm: trailing dimensions must match normalized_shape")
    return d


def _parameter(value: LayerInput, d: Int, name: String) raises:
    if not value.absent and (value.device or value.count != d):
        raise Error("ValueError: layer_norm: " + name + " must hold " + String(d) + " host values")


def _run(input: LayerInput, d: Int, weight: PythonObject, bias: PythonObject, eps: Float64,
         dy: PythonObject, mode: PythonObject, stats: PythonObject, grad_shape: List[Int],
         bound: PythonObject, context: PythonObject, alloc_stats: Bool = False) raises -> PythonObject:
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
    var flags: List[Int] = [m, d, 0 if w.absent else 1, 0 if b.absent else 1, 1 if backward else 0]
    var floats: List[Float64] = [eps]
    var nk = sequence_keys()
    if input.device:
        # lane dw-autograd-seq (2026-11-21): the image's route word answers
        # (was a lookup of the entry name `layer_norm_dev`); the operations
        # are descriptor calls on typed words (were CPython calls on Python
        # lists of handles, addresses, flags and scalars).
        if not seq_ops_device(input.entries):
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
        var stat_handles = List[Int]()
        if stats is not None:
            if not mean.device or not rstd.device or mean.count != m or rstd.count != m:
                raise Error("ValueError: layer_norm: stats must be two resident tensors of M floats")
            if py_item(mean.entries, nk[].k_module) is not py_item(input.entries, nk[].k_module) or py_item(rstd.entries, nk[].k_module) is not py_item(input.entries, nk[].k_module):
                raise Error("ValueError: layer_norm: stats must use x's native binding")
            stat_handles.append(Int(py=mean.handle))
            stat_handles.append(Int(py=rstd.handle))
            flags.append(1)
        var module = py_item(input.entries, nk[].k_module)
        var x_handle = Int(py=input.handle)
        if not backward:
            # Lane ne3-c-blocks-embedding (row N162): a device forward
            # allocates y (and, with `alloc_stats`, the kept mean and rstd)
            # inside the one SEQ_LAYER_NORM_DEV_ALLOC call. The result then
            # carries the kept stats as a fifth item. Every device image
            # serves this form, so the forward that allocated y in a separate
            # crossing (taken only when the entry name was absent) is gone.
            var out_count = max(1, shape_count(input.shape))
            var make_stats = stats is None and alloc_stats
            var slots = 6 if (stats is not None or make_stats) else 4
            # made = the handles the layer ran on, written by the operation.
            var made = List[Int](length=slots, fill=0)
            var ints = List[Int](capacity=2 * slots + 8)
            ints.append(slots)
            ints.append(x_handle)
            ints.append(0)
            ints.append(0)
            ints.append(0)
            if stats is not None:
                ints.append(stat_handles[0])
                ints.append(stat_handles[1])
            elif make_stats:
                ints.append(0)
                ints.append(0)
                flags.append(1)
            ints.append(0)
            ints.append(out_count)
            ints.append(0)
            ints.append(0)
            if slots == 6:
                ints.append(m if make_stats else 0)
                ints.append(m if make_stats else 0)
            for i in range(len(flags)):  # driver-loop: over the layer's option words (at most six)
                ints.append(flags[i])
            var fused_addresses: List[Int] = [Int(made.unsafe_ptr()), w.address, b.address, 0, 0]
            _ = seq_ops_call(input.entries, SEQ_LAYER_NORM_DEV_ALLOC, fused_addresses, ints, floats)
            # The tensors the call made are this door's until each has its
            # owner: slot 1 (y) and, with make_stats, slots 4 and 5. Slots 0
            # and caller-given stats are not ours and are never released here.
            made[0] = 0
            if slots == 6 and not make_stats:
                made[4] = 0
                made[5] = 0
            var fused_output = PythonObject(None)
            var kept = PythonObject(None)
            try:
                fused_output = seq_device_make(module, input.shape, PythonObject(made[1]), PythonObject(None), context)
                made[1] = 0
                if make_stats:
                    var kept_mean = seq_device_make(module, [m], PythonObject(made[4]), PythonObject(None), context)
                    made[4] = 0
                    var kept_rstd = seq_device_make(module, [m], PythonObject(made[5]), PythonObject(None), context)
                    made[5] = 0
                    kept = Python.tuple(kept_mean, kept_rstd)
            except error:
                _release_made(input.entries, made)
                raise error^
            _ = mean^
            _ = rstd^
            _ = gradient^
            _ = w^
            _ = b^
            return Python.tuple(fused_output, PythonObject(None), PythonObject(None), PythonObject(None), kept)
        # backward: dx is a new resident tensor; dweight and dbias are host arrays.
        var output = _new(input.entries, input.shape, context)
        var dy_owner = PythonObject(None)
        var dy_handle = 0
        if gradient.device:
            if py_item(gradient.entries, nk[].k_module) is not py_item(input.entries, nk[].k_module):
                raise Error("ValueError: layer_norm_backward: dy must use x's native binding")
            dy_handle = Int(py=gradient.handle)
        else:
            dy_owner = _upload(gradient, input.entries, context)
            dy_handle = Int(py=_handle(dy_owner, context))
        var dw = _array(grad_shape, context)
        var db = _array(grad_shape, context)
        var ints = List[Int](capacity=14)
        ints.append(4 + len(stat_handles))
        ints.append(x_handle)
        ints.append(0)
        ints.append(dy_handle)
        ints.append(Int(py=_handle(output, context)))
        for i in range(len(stat_handles)):  # driver-loop: over the two kept statistics
            ints.append(stat_handles[i])
        for i in range(len(flags)):  # driver-loop: over the layer's option words (at most six)
            ints.append(flags[i])
        var addresses: List[Int] = [w.address, b.address, Int(py=classical_address(dw)), Int(py=classical_address(db))]
        _ = seq_ops_call(input.entries, SEQ_LAYER_NORM_DEV, addresses, ints, floats)
        _ = dy_owner^
        _ = mean^
        _ = rstd^
        _ = gradient^
        _ = w^
        _ = b^
        return Python.tuple(PythonObject(None), output,
            dw if had_weight else PythonObject(None), db if had_bias else PythonObject(None))
    if gradient.device:
        raise Error("TypeError: layer_norm host input requires a host output gradient")
    # `bound`: None (today's binding of the numeric mode) or the CPU identical
    # image's module of an explicit device="cpu" beside a GPU set, resolved
    # once per call or per construction (lane vdf-seq)
    var entries = seq_binding(mode, context, bound)
    # Every x_sequence image serves SEQ_LAYER_NORM on its own executor (was a
    # lookup of the entry name `layer_norm`).
    if not seq_ops_native(entries):
        raise Error("ImportError: the sequence binding lacks layer_norm; rebuild it")
    # y (forward) or dx (backward): the entry downloads every word, so the
    # output is unfilled NumPy storage (lane fb-neural; see _array_unfilled)
    var output = _array_unfilled(input.shape, context)
    var scratch = _array([1], context) if not backward else PythonObject(None)
    var dw = _array(grad_shape, context)
    var db = _array(grad_shape, context)
    var output_address = Int(py=classical_address(output))
    var addresses: List[Int] = [input.address, w.address, b.address,
        0 if backward else output_address, gradient.address,
        output_address if backward else Int(py=classical_address(scratch)),
        Int(py=classical_address(dw)), Int(py=classical_address(db))]
    _ = seq_ops_call(entries, SEQ_LAYER_NORM, addresses, flags, floats)
    _ = scratch^
    _ = gradient^
    _ = w^
    _ = b^
    _ = output
    _ = dw
    _ = db
    return Python.tuple(PythonObject(None) if backward else output, output if backward else PythonObject(None),
            dw if had_weight else PythonObject(None), db if had_bias else PythonObject(None))


def sequence_layer_norm(operation: Int, owner: PythonObject, values: PythonObject, mode: PythonObject,
                        context: PythonObject) raises -> PythonObject:
    if operation == OPTIMIZER_OPN_SEQLN_FORWARD or operation == OPTIMIZER_OPN_SEQLN_BACKWARD:
        var backward = operation == OPTIMIZER_OPN_SEQLN_BACKWARD
        var offset = 1 if backward else 0
        opt_scalar(values[offset + 1], context)
        opt_scalar(values[offset + 4], context)
        opt_scalar(mode, context)
        # lane vdf-seq (2026-11-24): `device` rides last; read ONCE here, before
        # any work (core/function_device.mojo). None: one identity compare and
        # today's route. A resident input with "cpu" is refused here.
        var bound = PythonObject(None)
        var device_word = values[offset + 5] if len(values) == offset + 6 else PythonObject(None)
        if device_word is not None:
            var name = String("mojolearn.layer_norm_backward") if backward else String("mojolearn.layer_norm_forward")
            var fdev = function_device_admit(device_word, name, LN_HAS_GPU, LN_HAS_CPU, context, _b())
            if fdev.host_beside:
                seq_device_refuse_resident(values[offset], name, "x", context)
                if backward:
                    seq_device_refuse_resident(values[0], name, "dy", context)
            bound = seq_device_bound(fdev, mode, context)
        var input = LayerInput(values[offset], context)
        var d = _width(input, values[offset + 1], context)
        var result = _run(input, d, values[offset + 2], values[offset + 3], Float64(py=py_call_typed(_b().float, values[offset + 4])),
            values[0] if backward else PythonObject(None), mode, PythonObject(None), [d], bound, context)
        _ = input^
        if not backward:
            return result[0]
        return Python.tuple(result[1], PythonObject(None) if values[3] is None else result[2],
                            PythonObject(None) if values[4] is None else result[3])
    var fields = _ln_fields(owner, context)
    var nk = sequence_keys()
    if operation == OPTIMIZER_OPN_SEQLN_INITIALIZE:
        opt_scalar(values, context)
        opt_scalar(mode, context)
        # lane vdf-seq: the layer's `device` (fifth item), admitted ONCE here;
        # its binding is kept in `_host_b` (None: today's route).
        var device_word = values[4] if len(values) == 5 else PythonObject(None)
        var bound = PythonObject(None)
        if device_word is not None:
            var fdev = function_device_admit(device_word, "mojolearn.LayerNorm", LN_HAS_GPU, LN_HAS_CPU, context, _b())
            bound = seq_device_bound(fdev, mode, context)
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
        py_setitem(fields, nk[].k_normalized_shape, native_shape_pack(shape))
        py_setitem(fields, nk[].k__D, d)
        py_setitem(fields, nk[].k_eps, Float64(py=values[1]))
        py_setitem(fields, nk[].k_elementwise_affine, affine)
        py_setitem(fields, nk[].k_weight, weight)
        py_setitem(fields, nk[].k_bias, bias)
        py_setitem(fields, nk[].k_numeric_mode, mode)
        py_setitem(fields, nk[].k_device, device_word)
        py_setitem(fields, nk[].k__host_b, bound)
        for name in ["weight_grad", "bias_grad", "_x", "_stats"]:  # driver-loop: over initial native state fields
            py_setitem(fields, name, PythonObject(None))
        return PythonObject(None)
    if operation == OPTIMIZER_OPN_SEQLN_DEVICE_IO:
        var entries = _ln_device_entries(fields, context)
        return PythonObject(entries is not None)
    if operation == OPTIMIZER_OPN_SEQLN_TO_DEVICE:
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
    var d = Int(py=py_item(fields, nk[].k__D))
    var shape = _norm_shape(py_item(fields, nk[].k_normalized_shape), context)
    if d < 1 or d > 2147483647 or d != shape_count(shape):
        raise Error("ValueError: normalized_shape and native layer width must agree")
    if operation == OPTIMIZER_OPN_SEQLN_LAYER_BACKWARD:
        var x = py_get(fields, nk[].k__x)
        if x is None:
            raise Error("RuntimeError: LayerNorm.backward: call forward first")
        var input = LayerInput(x, context)
        _ = _width_known(input, d)
        var result = _run(input, d, py_item(fields, nk[].k_weight), py_item(fields, nk[].k_bias), Float64(py=py_item(fields, nk[].k_eps)), values[0],
            py_get(fields, nk[].k_numeric_mode), py_get(fields, nk[].k__stats), shape, py_get(fields, nk[].k__host_b), context)
        _ = input^
        py_setitem(fields, nk[].k_weight_grad, result[2])
        py_setitem(fields, nk[].k_bias_grad, result[3])
        return result[1]
    if operation != OPTIMIZER_OPN_SEQLN_LAYER:
        raise Error("ValueError: unknown LayerNorm public operation")
    var bound = py_get(fields, nk[].k__host_b)
    if bound is not None:
        seq_device_refuse_resident(values[0], "mojolearn.LayerNorm", "x", context)
    var input = LayerInput(values[0], context)
    _ = _width_known(input, d)
    var k = len(shape)
    var same = len(input.shape) >= k
    if same:
        for i in range(k):  # driver-loop: over normalized shape words
            same = same and input.shape[len(input.shape) - k + i] == shape[i]
    if not same:
        raise Error("ValueError: LayerNorm input trailing shape differs from normalized_shape")
    var stats = PythonObject(None)
    # TOMBSTONE: MOJOLEARN_LN_KEEP_STATS_OFF (env; removed: decided A/B, post-merge 2026-10-10 NV 0.95 / AMD 1.07, neutral = loser) deleted 2026-11-23 by lane/ab-prep-neural; code recoverable at 4d76a9e16e.
    # Restore: git apply ~/mojolearn-evidence/experiments-removed/MOJOLEARN_LN_KEEP_STATS_OFF.patch; record in docs/TOMBSTONES.md.
    var keep_stats = input.device
    # Row N162: the kept stats are made inside the forward's one call
    # (`_run`, alloc_stats) and published after it. Every device image serves
    # that form (`_run` refuses an image without device LayerNorm), so the
    # two separate stat allocations, taken only when the entry name
    # `layer_norm_dev_alloc` was absent, are gone.
    var fused = keep_stats
    # Keep the same normalized input owner and resident statistics for the
    # next public backward; its metadata is refreshed at that next operation.
    py_setitem(fields, nk[].k__x, input.owner)
    py_setitem(fields, nk[].k__stats, stats)
    var result = _run(input, d, py_item(fields, nk[].k_weight), py_item(fields, nk[].k_bias), Float64(py=py_item(fields, nk[].k_eps)), PythonObject(None),
        py_get(fields, nk[].k_numeric_mode), stats, shape, bound, context, fused)
    if fused and len(result) == 5:
        py_setitem(fields, nk[].k__stats, result[4])
    _ = input^
    return result[0]

