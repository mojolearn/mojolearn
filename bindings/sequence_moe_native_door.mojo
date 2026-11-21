# SPDX-License-Identifier: Apache-2.0
# Complete MoE public operations: passive admission, stable buffer metadata,
# weight ownership, native execution and finished result publication. Numerical
# forward/backward kernels and their resident weight ABI remain unchanged.
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module, NativeModules
from bindings.native_class_admission_door import class_record, class_admit
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_setattr, py_contains
from bindings.py_typed_call import py_call_typed
from bindings.mode_metadata_native_door import _fields, _class_fields
from bindings.optimizer_admission_door import opt_scalar, opt_value
from training.optimizer_input_plan import same_shape
from bindings.optimizer_device_native_door import device_shape, shape_count
from bindings.sequence_optimizer_native_door import seq_binding
from bindings.sequence_ops_door import (
    seq_ops_call, seq_ops_device, seq_ops_native, seq_tensor_new, seq_tensor_release, seq_tensor_put,
)
from sequence.neural_model_dispatch import (
    SEQ_MOE_FORWARD, SEQ_MOE_BACKWARD, SEQ_MOE_FORWARD_DEV_ALLOC, SEQ_MOE_WEIGHTS_PUT, SEQ_MOE_WEIGHTS_FREE,
)
from bindings.sequence_device_native_unpack import seq_device_make
from bindings.classical_arrays_door import classical_input, classical_typed, classical_address
from bindings.classical_attributes_door import classical_connect_numpy
from providers.python.sequence_layer_buffer_pyapi_unpack import SequenceFloatBuffer as MoEBuffer, SequenceResidentSnapshot as MoEDevice, _array
from core.hotpath_impl_rows import normal_init_f32
from core.sequence_array_cast import sequence_cast_leaf
from bindings.sequence_array_warnings_unpack import sequence_array_report
from bindings.sequence_array_warnings_pack import sequence_array_errstate
from bindings.sequence_keys_unpack import sequence_keys
from bindings.optimizer_keys_unpack import (
    OPTIMIZER_OPN_SEQMOE_BACKWARD,
    OPTIMIZER_OPN_SEQMOE_FORWARD,
    OPTIMIZER_OPN_SEQMOE_INITIALIZE,
    OPTIMIZER_OPN_SEQMOE_LOAD,
    OPTIMIZER_OPN_SEQMOE_RELEASE,
    OPTIMIZER_OPN_SEQMOE_ROUTING,
    OPTIMIZER_OPN_SEQMOE_SELECTED,
    OPTIMIZER_OPN_SEQMOE_SETATTR,
    OPTIMIZER_OPN_SEQMOE_TO_DEVICE,
)
from bindings.py_balanced import py_item


def _b() raises -> PythonObject:
    return native_loaded_module("builtins")



# The class admission below depends only on the class rows and the
# import-connected layout: kept per class (native_class_admission, re-run when
# the class or an ancestor changes; every call on Python < 3.12). Lane
# p3-neural-blocks: it was an MRO walk with 18 name tests per row per call.
comptime _MOE_CLASS_SLOT = "seqmoe_class_admitted_v1"


def _owner(owner: PythonObject, context: PythonObject) raises -> PythonObject:
    var cls = py_type(owner)
    var mods = NativeModules()
    if class_record(mods, _MOE_CLASS_SLOT, cls) is not None:
        return _fields(mods, owner)
    var nk = sequence_keys()
    var layout = py_item(context, nk[].k_seqmoe_layout)
    var b = _b()
    if py_type(cls) is not b.type:
        raise Error("TypeError: MoEBlock refuses Python metaclass protocols")
    var ancestry = py_call(py_item(b.type.__dict__, nk[].k___mro__).__get__, cls)
    var found = False
    for i in range(len(ancestry)):  # driver-loop: over passive native owner ancestry
        if ancestry[i] is layout[0]:
            found = True
            break
        var namespace = _class_fields(ancestry[i])
        for name in ["__getattribute__", "__getattr__", "__setattr__"]:  # driver-loop: over storage-hook overrides
            if py_contains(namespace, name):
                raise Error("TypeError: MoEBlock subclasses cannot replace native storage hooks")
    if not found or py_call(py_item(b.type.__dict__, nk[].k___mro__).__get__, layout[0]) is not layout[1]:
        raise Error("TypeError: MoEBlock requires its connected native class ancestry")
    var hooks = layout[2]
    for i in range(len(hooks)):  # driver-loop: over connected native storage hooks
        if py_get(_class_fields(hooks[i][0]), hooks[i][1]) is not hooks[i][2]:
            raise Error("TypeError: MoEBlock refuses replaced Python storage hooks")
    for i in range(len(ancestry)):  # driver-loop: over passive class storage metadata
        var namespace = _class_fields(ancestry[i])
        for name in ["D", "F", "E", "k", "norm_topk_prob", "numeric_mode", "router", "gate_up_proj", "down_proj", "_wver", "_wdev", "_routing", "router_grad_", "gate_up_proj_grad_", "down_proj_grad_"]:  # driver-loop: over native state schema
            if py_contains(namespace, name):
                raise Error("TypeError: MoEBlock state cannot be replaced by class descriptors")
    class_admit(mods, _MOE_CLASS_SLOT, cls, PythonObject(True))
    return _fields(mods, owner)


def _integer(value: PythonObject, context: PythonObject) raises -> Int:
    opt_scalar(value, context)
    return Int(py=py_call(_b().int, value))


def _version(fields: PythonObject) raises -> Int:
    var nk = sequence_keys()
    var value = py_get(fields, nk[].k__wver, PythonObject(0))
    if py_type(value) is not _b().int:
        raise Error("TypeError: MoEBlock weight version requires an ordinary integer")
    return Int(py=value)


def _words(fields: PythonObject, context: PythonObject) raises -> List[Int]:
    var words = List[Int]()
    for name in ["D", "F", "E", "k"]:  # driver-loop: over capture scalar dimensions once
        var v = fields[PythonObject(name)]
        if py_type(v) is not _b().int:
            raise Error("TypeError: MoEBlock dimensions require ordinary integers")
        var n = Int(py=v)
        if n < 1 or n > 2147483647:
            raise Error("ValueError: MoEBlock dimensions must be positive native sizes")
        words.append(n)
    if words[3] > words[2] or words[1] > 2147483647 // 2:
        raise Error("ValueError: MoEBlock: 1 <= top_k <= num_experts and valid intermediate size")
    var total = 1
    for size in [words[0], 2 * words[1], words[2]]:  # driver-loop: over weight extent overflow
        if total > 2147483647 // size:
            raise Error("ValueError: MoEBlock weights exceed signed 32-bit indexing")
        total *= size
    var nk = sequence_keys()
    var norm = py_item(fields, nk[].k_norm_topk_prob)
    opt_scalar(norm, context)
    words.append(1 if Bool(py=norm) else 0)
    opt_scalar(py_get(fields, nk[].k_numeric_mode), context)
    return words^


def _shapes(words: List[Int], index: Int) -> List[Int]:
    if index == 0:
        return [words[2], words[0]]
    if index == 1:
        return [words[2], 2 * words[1], words[0]]
    return [words[2], words[0], words[1]]


def _readonly(value: PythonObject) raises:
    # `_array` allocates exact ndarrays; this is their C flags setter, never a
    # user's property or NumPy computation/conversion function.
    py_setattr(value.flags, "writeable", PythonObject(False))


def _sealed(value: PythonObject) raises -> PythonObject:
    """The stored weight: a read-only view over a read-only buffer of the
    owning ndarray, so `w.flags.writeable = True` raises instead of letting
    an in-place write diverge from the resident device copy (lane
    p3-neural-blocks; the resident key is the weight version, which only an
    assignment bumps). `value` must already be read-only (`_readonly`); its
    memoryview is then read-only and NumPy refuses to turn WRITEABLE back on
    for an array whose base buffer is read-only. Same bytes, no copy."""
    # The exact ndarray type of `value` builds the view: `numpy` itself is not
    # an import-time connection (native_loaded_module("numpy") raised
    # ImportError at MoEBlock construction), and the C type needs no module.
    var view = py_call(_b().memoryview, value)
    return py_call(py_type(value), value.shape, dtype=value.dtype, buffer=view)


def _owner_word(fields: PythonObject) -> Int:
    # The identity of this block's own instance dict: `copy.copy(block)`
    # copies `_wdev` into a new dict, and that copy must neither use nor free
    # the original block's device handle.
    return Int(fields._obj_ptr)


def _cached(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = sequence_keys()
    var cur = py_get(fields, nk[].k__wdev)
    if cur is None:
        return cur
    if (py_type(cur) is not _b().tuple or len(cur) != 4 or py_type(cur[1]) is not _b().int
            or py_type(cur[2]) is not _b().int or py_type(cur[3]) is not _b().int):
        raise Error("TypeError: MoEBlock resident weights require native owner/version/handle metadata")
    if Int(py=cur[3]) != _owner_word(fields):
        # Another block's handle, copied in by copy.copy: forget it here
        # without freeing it (its own block releases it).
        py_setitem(fields, nk[].k__wdev, PythonObject(None))
        return PythonObject(None)
    _ = seq_binding(PythonObject(None), context, cur[0])
    return cur


def _release(fields: PythonObject, context: PythonObject) raises:
    var cur = _cached(fields, context)
    var nk = sequence_keys()
    if cur is not None:
        # OWNERSHIP: the block's own `_wdev` record (its owner word is this
        # instance dictionary) holds the one resident weight handle; this is
        # its one release, in the image that made it. Only a device
        # x_sequence image makes one (`_weights`).
        var entries = seq_binding(PythonObject(None), context, cur[0])
        if not seq_ops_device(entries):
            raise Error("RuntimeError: MoEBlock: this binding holds no resident weights; pass host weights")
        var addresses = List[Int]()
        var ints: List[Int] = [Int(py=cur[2])]
        var floats = List[Float64]()
        _ = seq_ops_call(entries, SEQ_MOE_WEIGHTS_FREE, addresses, ints, floats)
    py_setitem(fields, nk[].k__wdev, PythonObject(None))


def _weights(fields: PythonObject, entries: PythonObject, words: List[Int],
             r: MoEBuffer, gu: MoEBuffer, dn: MoEBuffer, context: PythonObject) raises -> Int:
    var version = _version(fields)
    var cur = _cached(fields, context)
    var nk = sequence_keys()
    if cur is not None and cur[0] is py_item(entries, nk[].k_module) and Int(py=cur[1]) == version:
        return Int(py=cur[2])
    _release(fields, context)
    var addresses: List[Int] = [r.address, gu.address, dn.address]
    var ints: List[Int] = [words[2], words[0], words[1]]
    var floats = List[Float64]()
    var handle = seq_ops_call(entries, SEQ_MOE_WEIGHTS_PUT, addresses, ints, floats)
    # The handle is this door's until the block's `_wdev` record holds it:
    # released here if the record cannot be written.
    try:
        py_setitem(fields, nk[].k__wdev, Python.tuple(py_item(entries, nk[].k_module), version, handle, _owner_word(fields)))
    except error:
        var release: List[Int] = [handle]
        _ = seq_ops_call(entries, SEQ_MOE_WEIGHTS_FREE, List[Int](), release, floats)
        raise error^
    return handle


def _check(r: MoEBuffer, gu: MoEBuffer, dn: MoEBuffer, words: List[Int]) raises:
    if not same_shape(r.shape, _shapes(words, 0)) or not same_shape(gu.shape, _shapes(words, 1)) or not same_shape(dn.shape, _shapes(words, 2)):
        raise Error("ValueError: MoEBlock weight shape does not match the block")
    if not r.readonly or not gu.readonly or not dn.readonly:
        raise Error("TypeError: MoEBlock weight arrays must retain their read-only storage; assign replacements")



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


def _handle(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = sequence_keys()
    # Only freshly-created native outputs reach here.
    return py_call(py_item(py_item(context, nk[].k_sequence_device_getters), nk[].k_h), value)


def _host_input(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = sequence_keys()
    if py_type(value) is py_item(context, nk[].k_sequence_device_class):
        var snapshot = MoEDevice(value, context)
        return snapshot.host(context)
    return value


def _input_shape(shape: List[Int], count: Int, d: Int) raises:
    if len(shape) == 0 or shape[len(shape) - 1] != d or count == 0:
        raise Error("ValueError: MoEBlock: x must have nonempty shape (..., " + String(d) + ")")


def _initialize(fields: PythonObject, values: PythonObject, mode: PythonObject, context: PythonObject) raises -> PythonObject:
    for i in range(7):  # driver-loop: over constructor scalar admission
        opt_scalar(values[i], context)
    opt_scalar(mode, context)
    if py_type(values[5]) is not _b().str or String(values[5]) != "silu":
        raise Error("NotImplementedError: MoEBlock: hidden_act other than 'silu' is not implemented")
    var fresh = Python.dict()
    var names = Python.tuple("D", "F", "E", "k")
    for i in range(4):  # driver-loop: over constructor dimensions
        py_setitem(fresh, names[i], PythonObject(_integer(values[i], context)))
    var nk = sequence_keys()
    py_setitem(fresh, nk[].k_norm_topk_prob, PythonObject(Bool(py=values[4])))
    py_setitem(fresh, nk[].k_numeric_mode, mode)
    var words = _words(fresh, context)
    var seed = values[6]
    if seed is None:
        seed = py_call(_b().int.from_bytes, py_call_typed(py_item(context, nk[].k_seqmoe_urandom), PythonObject(8)), "little")
    else:
        seed = py_call(_b().int, seed)
    var low = py_call(py_item(_b().int.__dict__, nk[].k___and__), seed, PythonObject(4294967295))
    var high = py_call(py_item(_b().int.__dict__, nk[].k___and__), py_call(py_item(_b().int.__dict__, nk[].k___rshift__), seed, PythonObject(32)), PythonObject(4294967295))
    var word = (UInt64(Int(py=high)) << 32) | UInt64(Int(py=low))
    var offset = UInt64(0)
    var names_w = Python.tuple("router", "gate_up_proj", "down_proj")
    for i in range(3):  # driver-loop: over three fixed initialized weight tensors
        var shape = _shapes(words, i)
        var a = _array(shape, context)
        var n = shape_count(shape)
        _ = normal_init_f32(Int(py=classical_address(a)), n, 0.0, 0.02, word, offset)
        offset += UInt64(n)
        _readonly(a)
        py_setitem(fresh, names_w[i], _sealed(a))
    var version = _version(fields)
    _release(fields, context)
    py_setitem(fresh, nk[].k__wver, PythonObject(version + 3))
    _ = py_call(fields.update, fresh)
    return PythonObject(None)


def _load(owner: PythonObject, fields: PythonObject, state: PythonObject, context: PythonObject) raises -> PythonObject:
    if py_type(state) is not _b().dict:
        raise Error("TypeError: MoEBlock state requires an exact dict of native arrays")
    var keys = py_call(_b().list, state)
    for i in range(len(keys)):  # driver-loop: over state keys before keyed access
        if py_type(keys[i]) is not _b().str:
            raise Error("TypeError: MoEBlock state keys require exact strings")
    var words = _words(fields, context)
    var pending = Python.list()
    var names = Python.tuple("router", "gate_up_proj", "down_proj")
    for i in range(3):  # driver-loop: over three native parameter copies
        var buffer = MoEBuffer(state[names[i]], context, True, False)
        if not same_shape(buffer.shape, _shapes(words, i)):
            raise Error("ValueError: MoEBlock: " + String(names[i]) + " has the wrong shape")
        _readonly(buffer.array)
        _ = py_call(pending.append, _sealed(buffer.array))
    var version = _version(fields)
    for i in range(3):  # driver-loop: over publish only after complete admission
        py_setitem(fields, names[i], pending[i])
    var nk = sequence_keys()
    py_setitem(fields, nk[].k__wver, PythonObject(version + 3))
    return owner


def _run(fields: PythonObject, values: PythonObject, context: PythonObject, backward: Bool) raises -> PythonObject:
    var words = _words(fields, context)
    var d = words[0]
    var f = words[1]
    var e = words[2]
    var k = words[3]
    var nk = sequence_keys()
    var r = MoEBuffer(py_item(fields, nk[].k_router), context)
    var gu = MoEBuffer(py_item(fields, nk[].k_gate_up_proj), context)
    var dn = MoEBuffer(py_item(fields, nk[].k_down_proj), context)
    _check(r, gu, dn, words)
    var x = values[0]
    if not backward and py_type(x) is py_item(context, nk[].k_sequence_device_class):
        var snapshot = MoEDevice(x, context)
        _input_shape(snapshot.shape, snapshot.count, d)
        var entries = snapshot.entries
        # lane dw-autograd-seq (2026-11-21): the image's route word answers
        # (bindings/sequence_ops_door.mojo; was two entry-name lookups).
        if seq_ops_device(entries):
            var t = snapshot.count // d
            if t > 2147483647 // e or t > 2147483647 // k:
                raise Error("ValueError: MoEBlock routing output exceeds signed 32-bit indexing")
            var h = _weights(fields, entries, words, r, gu, dn, context)
            # Lane ne3-c-blocks-embedding (row N163): the four outputs are
            # allocated inside the forward's one call. Every device image
            # serves this form, so the forward that made each output in its
            # own crossing (taken only when the entry name
            # `moe_forward_dev_alloc` was absent) is gone.
            # ints = [H, the H handles, the H sizes, the option words].
            var made = List[Int](length=5, fill=0)
            var ints: List[Int] = [5, Int(py=snapshot.handle), 0, 0, 0, 0,
                0, max(1, shape_count(snapshot.shape)), max(1, t * e), max(1, t * k), max(1, t * k),
                t, d, f, e, k, words[4], h]
            var addresses: List[Int] = [Int(made.unsafe_ptr())]
            var floats = List[Float64]()
            _ = seq_ops_call(entries, SEQ_MOE_FORWARD_DEV_ALLOC, addresses, ints, floats)
            # Slots 1..4 are this door's until each has its owner; slot 0 is
            # the caller's x and is never released here.
            made[0] = 0
            var module = py_item(entries, nk[].k_module)
            var y = PythonObject(None)
            var logits = PythonObject(None)
            var sel = PythonObject(None)
            var w = PythonObject(None)
            try:
                y = seq_device_make(module, snapshot.shape, PythonObject(made[1]), PythonObject(None), context)
                made[1] = 0
                logits = seq_device_make(module, [t, e], PythonObject(made[2]), PythonObject(None), context)
                made[2] = 0
                sel = seq_device_make(module, [t, k], PythonObject(made[3]), PythonObject(None), context)
                made[3] = 0
                w = seq_device_make(module, [t, k], PythonObject(made[4]), PythonObject(None), context)
                made[4] = 0
            except error:
                for i in range(len(made)):  # driver-loop: over the forward's five handle slots
                    if made[i] != 0:
                        try:
                            seq_tensor_release(entries, made[i])
                        except:
                            pass
                raise error^
            py_setitem(fields, nk[].k__routing, Python.list(logits, sel, w, PythonObject(None)))
            _ = snapshot^
            _ = r^
            _ = gu^
            _ = dn^
            return y
        x = snapshot.host(context)
    elif backward:
        x = _host_input(x, context)
    var input = MoEBuffer(x, context)
    _input_shape(input.shape, input.count, d)
    var t = input.count // d
    if t > 2147483647 // e or t > 2147483647 // k:
        raise Error("ValueError: MoEBlock routing output exceeds signed 32-bit indexing")
    var entries = seq_binding(py_get(fields, nk[].k_numeric_mode), context, PythonObject(None))
    # Host arrays in, host arrays out, on the layer's own image. A device
    # image reads the block's resident weights (a seventh word); the host
    # image uploads nothing and reads the three arrays. The route word
    # answers both questions (was the entry names `moe_weights_put` and
    # `moe_backward`).
    if not seq_ops_native(entries):
        raise Error("NotImplementedError: MoEBlock: this install has no backward pass" if backward
                    else "ImportError: the sequence binding lacks moe_forward; rebuild it")
    var ip: List[Int] = [t, d, f, e, k, words[4]]
    if seq_ops_device(entries):
        ip.append(_weights(fields, entries, words, r, gu, dn, context))
    var floats = List[Float64]()
    var y = _array(input.shape, context)
    var logits = _array([t, e], context)
    var sel = _array([t, k], context)
    var w = _array([t, k], context)
    var result = y
    if backward:
        var gradient = MoEBuffer(_host_input(values[1], context), context)
        if not same_shape(gradient.shape, input.shape):
            raise Error("ValueError: MoEBlock: dy must have x's shape")
        var dx = _array(input.shape, context)
        var dr = _array(_shapes(words, 0), context)
        var dgu = _array(_shapes(words, 1), context)
        var ddn = _array(_shapes(words, 2), context)
        var addresses: List[Int] = [input.address, r.address, gu.address, dn.address,
            gradient.address, Int(py=classical_address(dx)), Int(py=classical_address(dr)),
            Int(py=classical_address(dgu)), Int(py=classical_address(ddn)),
            Int(py=classical_address(y)), Int(py=classical_address(logits)), Int(py=classical_address(sel)),
            Int(py=classical_address(w))]
        _ = seq_ops_call(entries, SEQ_MOE_BACKWARD, addresses, ip, floats)
        # The native backward call also writes its forward output. Its raw
        # address does not own the ndarray, and backward returns dx instead.
        # Keep this otherwise unused output alive through the native download.
        _ = y
        _ = gradient^
        py_setitem(fields, nk[].k_router_grad_, dr)
        py_setitem(fields, nk[].k_gate_up_proj_grad_, dgu)
        py_setitem(fields, nk[].k_down_proj_grad_, ddn)
        result = dx
    else:
        var addresses: List[Int] = [input.address, r.address, gu.address, dn.address,
            Int(py=classical_address(y)), Int(py=classical_address(logits)), Int(py=classical_address(sel)),
            Int(py=classical_address(w))]
        _ = seq_ops_call(entries, SEQ_MOE_FORWARD, addresses, ip, floats)
    _ = input^
    _ = r^
    _ = gu^
    _ = dn^
    py_setitem(fields, nk[].k__routing, Python.list(logits, sel, w, PythonObject(None)))
    return result


def _routing(fields: PythonObject, index: Int, context: PythonObject) raises -> PythonObject:
    var nk = sequence_keys()
    var routing = py_get(fields, nk[].k__routing)
    if routing is None:
        raise Error("AttributeError: MoEBlock: call forward first")
    if py_type(routing) is not _b().list or len(routing) != 4 or index < 0 or index > 3:
        raise Error("TypeError: MoEBlock routing output requires its native result slots")
    var slot = 1 if index == 3 else index
    if index == 3 and routing[3] is not None:
        # A finished native int64 output, checked before returning user state.
        opt_value(routing[3], context)
        return routing[3]
    var value = routing[slot]
    if py_type(value) is py_item(context, nk[].k_sequence_device_class):
        value = _host_input(value, context)
        py_setitem(routing, slot, value)
    if index != 3:
        opt_value(value, context)
        return value
    var source = MoEBuffer(value, context)
    var result = _array(source.shape, context, True)
    var settings = sequence_array_errstate(context)
    var status = sequence_cast_leaf(source.address, Int(py=classical_address(result)), source.count, 8, 0, "i", 8, True)
    _ = source^
    sequence_array_report(status, settings, context)
    py_setitem(routing, 3, result)
    return result


def sequence_moe_operation(operation: Int, owner: PythonObject, values: PythonObject, mode: PythonObject,
                           context: PythonObject) raises -> PythonObject:
    var fields = _owner(owner, context)
    if operation == OPTIMIZER_OPN_SEQMOE_INITIALIZE:
        return _initialize(fields, values, mode, context)
    var nk = sequence_keys()
    if operation == OPTIMIZER_OPN_SEQMOE_SETATTR:
        var name = values[0]
        if py_type(name) is not _b().str:
            raise Error("TypeError: MoEBlock attribute names require exact strings")
        var text = String(name)
        if text == "router" or text == "gate_up_proj" or text == "down_proj":
            var buffer = MoEBuffer(values[1], context, True, False)
            var version = _version(fields)
            _readonly(buffer.array)
            py_setitem(fields, name, _sealed(buffer.array))
            py_setitem(fields, nk[].k__wver, PythonObject(version + 1))
        else:
            # Ordinary instance fields retain Python assignment semantics. A
            # data descriptor is refused before its setter could invoke Python.
            var lineage = py_call(py_item(_b().type.__dict__, nk[].k___mro__).__get__, py_type(owner))
            for base in range(len(lineage)):  # driver-loop: over effective class descriptor lookup
                var namespace = _class_fields(lineage[base])
                if not py_contains(namespace, name):
                    continue
                var member = namespace[name]
                var ancestry = py_call(py_item(_b().type.__dict__, nk[].k___mro__).__get__, py_type(member))
                for i in range(len(ancestry)):  # driver-loop: over native descriptor metadata
                    if py_contains(_class_fields(ancestry[i]), nk[].k___set__) or py_contains(_class_fields(ancestry[i]), nk[].k___delete__):
                        raise Error("AttributeError: MoEBlock attribute has no native writable storage")
                break
            py_setitem(fields, name, values[1])
        return PythonObject(None)
    if operation == OPTIMIZER_OPN_SEQMOE_LOAD:
        return _load(owner, fields, values[0], context)
    if operation == OPTIMIZER_OPN_SEQMOE_RELEASE:
        try:
            _release(fields, context)
        except:
            pass  # Best-effort cleanup, as the previous public finalizer.
        return PythonObject(None)
    if operation == OPTIMIZER_OPN_SEQMOE_FORWARD or operation == OPTIMIZER_OPN_SEQMOE_BACKWARD:
        return _run(fields, values, context, operation == OPTIMIZER_OPN_SEQMOE_BACKWARD)
    if operation == OPTIMIZER_OPN_SEQMOE_ROUTING:
        return _routing(fields, _integer(values[0], context), context)
    if operation == OPTIMIZER_OPN_SEQMOE_SELECTED:
        return _routing(fields, 3, context)
    if operation == OPTIMIZER_OPN_SEQMOE_TO_DEVICE:
        opt_scalar(py_get(fields, nk[].k_numeric_mode), context)
        var entries = seq_binding(py_get(fields, nk[].k_numeric_mode), context, PythonObject(None))
        var value = values[0]
        if py_type(value) is py_item(context, nk[].k_sequence_device_class):
            var snapshot = MoEDevice(value, context)
            if seq_ops_device(entries):
                return value
            value = snapshot.host(context)
        var input = MoEBuffer(value, context, False, False, True)
        if not seq_ops_device(entries) or Bool(py=py_item(context, nk[].k_sequence_device_io_off)):
            return input.array
        var result = _new(entries, input.shape, context)
        if input.count > 0:
            seq_tensor_put(entries, Int(py=_handle(result, context)), input.address, input.count)
        _ = input^
        return result
    raise Error("ValueError: unknown MoEBlock public operation")
