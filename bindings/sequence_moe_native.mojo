# SPDX-License-Identifier: Apache-2.0
# Complete MoE public operations: passive admission, stable buffer metadata,
# weight ownership, native execution and finished result publication. Numerical
# forward/backward kernels and their resident weight ABI remain unchanged.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_setattr, py_contains
from bindings.py_typed_call import py_call_typed
from bindings.mode_metadata_native import _fields, _class_fields
from bindings.optimizer_admission import opt_scalar, opt_value
from bindings.optimizer_public_door import py_ints, same_shape
from bindings.optimizer_device_native import device_shape, shape_count
from bindings.sequence_optimizer_native import seq_binding
from bindings.sequence_device_native import seq_device_make
from bindings.classical_arrays import classical_input, classical_typed, classical_address
from bindings.classical_attributes import classical_connect_numpy
from providers.python.sequence_layer_buffer_pyapi import SequenceFloatBuffer as MoEBuffer, SequenceResidentSnapshot as MoEDevice, _array
from core.hotpath_impl_rows import normal_init_f32
from core.sequence_array_cast import sequence_cast_leaf
from bindings.sequence_array_warnings import sequence_array_errstate, sequence_array_report


def _b() raises -> PythonObject:
    return native_loaded_module("builtins")



def _owner(owner: PythonObject, context: PythonObject) raises -> PythonObject:
    var layout = context["seqmoe_layout"]
    var cls = py_type(owner)
    var b = _b()
    if py_type(cls) is not b.type:
        raise Error("TypeError: MoEBlock refuses Python metaclass protocols")
    var ancestry = py_call(b.type.__dict__["__mro__"].__get__, cls)
    var found = False
    for i in range(len(ancestry)):  # driver-loop: over passive native owner ancestry
        if ancestry[i] is layout[0]:
            found = True
            break
        var namespace = _class_fields(ancestry[i])
        for name in ["__getattribute__", "__getattr__", "__setattr__"]:  # driver-loop: over storage-hook overrides
            if py_contains(namespace, name):
                raise Error("TypeError: MoEBlock subclasses cannot replace native storage hooks")
    if not found or py_call(b.type.__dict__["__mro__"].__get__, layout[0]) is not layout[1]:
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
    return _fields(owner)


def _integer(value: PythonObject, context: PythonObject) raises -> Int:
    opt_scalar(value, context)
    return Int(py=py_call(_b().int, value))


def _version(fields: PythonObject) raises -> Int:
    var value = py_get(fields, "_wver", PythonObject(0))
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
    var norm = fields["norm_topk_prob"]
    opt_scalar(norm, context)
    words.append(1 if Bool(py=norm) else 0)
    opt_scalar(py_get(fields, "numeric_mode"), context)
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


def _cached(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    var cur = py_get(fields, "_wdev")
    if cur is None:
        return cur
    if py_type(cur) is not _b().tuple or len(cur) != 3 or py_type(cur[1]) is not _b().int or py_type(cur[2]) is not _b().int:
        raise Error("TypeError: MoEBlock resident weights require native owner/version/handle metadata")
    _ = seq_binding(PythonObject(None), context, cur[0])
    return cur


def _release(fields: PythonObject, context: PythonObject) raises:
    var cur = _cached(fields, context)
    if cur is not None:
        var entries = seq_binding(PythonObject(None), context, cur[0])
        _ = py_call_typed(entries["moe_weights_free"], cur[2])
    py_setitem(fields, "_wdev", PythonObject(None))


def _weights(fields: PythonObject, entries: PythonObject, words: List[Int],
             r: MoEBuffer, gu: MoEBuffer, dn: MoEBuffer, context: PythonObject) raises -> PythonObject:
    var version = _version(fields)
    var cur = _cached(fields, context)
    if cur is not None and cur[0] is entries["module"] and Int(py=cur[1]) == version:
        return cur[2]
    _release(fields, context)
    var handle = py_call_typed(entries["moe_weights_put"], Python.list(r.address, gu.address, dn.address),
                               Python.list(words[2], words[0], words[1]))
    py_setitem(fields, "_wdev", Python.tuple(entries["module"], version, handle))
    return handle


def _check(r: MoEBuffer, gu: MoEBuffer, dn: MoEBuffer, words: List[Int]) raises:
    if not same_shape(r.shape, _shapes(words, 0)) or not same_shape(gu.shape, _shapes(words, 1)) or not same_shape(dn.shape, _shapes(words, 2)):
        raise Error("ValueError: MoEBlock weight shape does not match the block")
    if not r.readonly or not gu.readonly or not dn.readonly:
        raise Error("TypeError: MoEBlock weight arrays must retain their read-only storage; assign replacements")



def _new(entries: PythonObject, shape: List[Int], context: PythonObject) raises -> PythonObject:
    var handle = py_call_typed(entries["seq_tensor_alloc"], PythonObject(max(1, shape_count(shape))), PythonObject(0))
    return seq_device_make(entries["module"], shape, handle, PythonObject(None), context)


def _handle(value: PythonObject, context: PythonObject) raises -> PythonObject:
    # Only freshly-created native outputs reach here.
    return py_call(context["sequence_device_getters"]["h"], value)


def _host_input(value: PythonObject, context: PythonObject) raises -> PythonObject:
    if py_type(value) is context["sequence_device_class"]:
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
    py_setitem(fresh, "norm_topk_prob", PythonObject(Bool(py=values[4])))
    py_setitem(fresh, "numeric_mode", mode)
    var words = _words(fresh, context)
    var seed = values[6]
    if seed is None:
        seed = py_call(_b().int.from_bytes, py_call_typed(context["seqmoe_urandom"], PythonObject(8)), "little")
    else:
        seed = py_call(_b().int, seed)
    var low = py_call(_b().int.__dict__["__and__"], seed, PythonObject(4294967295))
    var high = py_call(_b().int.__dict__["__and__"], py_call(_b().int.__dict__["__rshift__"], seed, PythonObject(32)), PythonObject(4294967295))
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
        py_setitem(fresh, names_w[i], a)
    var version = _version(fields)
    _release(fields, context)
    py_setitem(fresh, "_wver", PythonObject(version + 3))
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
        _ = py_call(pending.append, buffer.array)
    var version = _version(fields)
    for i in range(3):  # driver-loop: over publish only after complete admission
        py_setitem(fields, names[i], pending[i])
    py_setitem(fields, "_wver", PythonObject(version + 3))
    return owner


def _run(fields: PythonObject, values: PythonObject, context: PythonObject, backward: Bool) raises -> PythonObject:
    var words = _words(fields, context)
    var d = words[0]
    var f = words[1]
    var e = words[2]
    var k = words[3]
    var r = MoEBuffer(fields["router"], context)
    var gu = MoEBuffer(fields["gate_up_proj"], context)
    var dn = MoEBuffer(fields["down_proj"], context)
    _check(r, gu, dn, words)
    var x = values[0]
    if not backward and py_type(x) is context["sequence_device_class"]:
        var snapshot = MoEDevice(x, context)
        _input_shape(snapshot.shape, snapshot.count, d)
        var entries = snapshot.entries
        if py_get(entries, "moe_forward_dev") is not None and py_get(entries, "moe_weights_put") is not None:
            var t = snapshot.count // d
            if t > 2147483647 // e or t > 2147483647 // k:
                raise Error("ValueError: MoEBlock routing output exceeds signed 32-bit indexing")
            var h = _weights(fields, entries, words, r, gu, dn, context)
            var y = _new(entries, snapshot.shape, context)
            var logits = _new(entries, [t, e], context)
            var sel = _new(entries, [t, k], context)
            var w = _new(entries, [t, k], context)
            _ = py_call_typed(entries["moe_forward_dev"], Python.list(snapshot.handle, _handle(y, context),
                _handle(logits, context), _handle(sel, context), _handle(w, context)), Python.list(t, d, f, e, k, words[4], h))
            py_setitem(fields, "_routing", Python.list(logits, sel, w, PythonObject(None)))
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
    var entries = seq_binding(py_get(fields, "numeric_mode"), context, PythonObject(None))
    var ip = Python.list(t, d, f, e, k, words[4])
    if py_get(entries, "moe_weights_put") is not None:
        _ = py_call(ip.append, _weights(fields, entries, words, r, gu, dn, context))
    var y = _array(input.shape, context)
    var logits = _array([t, e], context)
    var sel = _array([t, k], context)
    var w = _array([t, k], context)
    var result = y
    if backward:
        if py_get(entries, "moe_backward") is None:
            raise Error("NotImplementedError: MoEBlock: this install has no backward pass")
        var gradient = MoEBuffer(_host_input(values[1], context), context)
        if not same_shape(gradient.shape, input.shape):
            raise Error("ValueError: MoEBlock: dy must have x's shape")
        var dx = _array(input.shape, context)
        var dr = _array(_shapes(words, 0), context)
        var dgu = _array(_shapes(words, 1), context)
        var ddn = _array(_shapes(words, 2), context)
        _ = py_call_typed(entries["moe_backward"], Python.list(input.address, r.address, gu.address, dn.address,
            gradient.address, classical_address(dx), classical_address(dr), classical_address(dgu), classical_address(ddn),
            classical_address(y), classical_address(logits), classical_address(sel), classical_address(w)), ip)
        # The native backward call also writes its forward output. Its raw
        # address does not own the ndarray, and backward returns dx instead.
        # Keep this otherwise unused output alive through the native download.
        _ = y
        _ = gradient^
        py_setitem(fields, "router_grad_", dr)
        py_setitem(fields, "gate_up_proj_grad_", dgu)
        py_setitem(fields, "down_proj_grad_", ddn)
        result = dx
    else:
        _ = py_call_typed(entries["moe_forward"], Python.list(input.address, r.address, gu.address, dn.address,
            classical_address(y), classical_address(logits), classical_address(sel), classical_address(w)), ip)
    _ = input^
    _ = r^
    _ = gu^
    _ = dn^
    py_setitem(fields, "_routing", Python.list(logits, sel, w, PythonObject(None)))
    return result


def _routing(fields: PythonObject, index: Int, context: PythonObject) raises -> PythonObject:
    var routing = py_get(fields, "_routing")
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
    if py_type(value) is context["sequence_device_class"]:
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


def sequence_moe_operation(operation: String, owner: PythonObject, values: PythonObject, mode: PythonObject,
                           context: PythonObject) raises -> PythonObject:
    var fields = _owner(owner, context)
    if operation == "seqmoe_initialize":
        return _initialize(fields, values, mode, context)
    if operation == "seqmoe_setattr":
        var name = values[0]
        if py_type(name) is not _b().str:
            raise Error("TypeError: MoEBlock attribute names require exact strings")
        var text = String(name)
        if text == "router" or text == "gate_up_proj" or text == "down_proj":
            var buffer = MoEBuffer(values[1], context, True, False)
            var version = _version(fields)
            _readonly(buffer.array)
            py_setitem(fields, name, buffer.array)
            py_setitem(fields, "_wver", PythonObject(version + 1))
        else:
            # Ordinary instance fields retain Python assignment semantics. A
            # data descriptor is refused before its setter could invoke Python.
            var lineage = py_call(_b().type.__dict__["__mro__"].__get__, py_type(owner))
            for base in range(len(lineage)):  # driver-loop: over effective class descriptor lookup
                var namespace = _class_fields(lineage[base])
                if not py_contains(namespace, name):
                    continue
                var member = namespace[name]
                var ancestry = py_call(_b().type.__dict__["__mro__"].__get__, py_type(member))
                for i in range(len(ancestry)):  # driver-loop: over native descriptor metadata
                    if py_contains(_class_fields(ancestry[i]), "__set__") or py_contains(_class_fields(ancestry[i]), "__delete__"):
                        raise Error("AttributeError: MoEBlock attribute has no native writable storage")
                break
            py_setitem(fields, name, values[1])
        return PythonObject(None)
    if operation == "seqmoe_load":
        return _load(owner, fields, values[0], context)
    if operation == "seqmoe_release":
        try:
            _release(fields, context)
        except:
            pass  # Best-effort cleanup, as the previous public finalizer.
        return PythonObject(None)
    if operation == "seqmoe_forward" or operation == "seqmoe_backward":
        return _run(fields, values, context, operation == "seqmoe_backward")
    if operation == "seqmoe_routing":
        return _routing(fields, _integer(values[0], context), context)
    if operation == "seqmoe_selected":
        return _routing(fields, 3, context)
    if operation == "seqmoe_to_device":
        opt_scalar(py_get(fields, "numeric_mode"), context)
        var entries = seq_binding(py_get(fields, "numeric_mode"), context, PythonObject(None))
        var value = values[0]
        if py_type(value) is context["sequence_device_class"]:
            var snapshot = MoEDevice(value, context)
            if py_get(entries, "moe_forward_dev") is not None:
                return value
            value = snapshot.host(context)
        var input = MoEBuffer(value, context, False, False, True)
        if py_get(entries, "moe_forward_dev") is None or Bool(py=context["sequence_device_io_off"]):
            return input.array
        var result = _new(entries, input.shape, context)
        if input.count > 0:
            _ = py_call_typed(entries["seq_tensor_upload"], _handle(result, context), PythonObject(input.address), PythonObject(input.count))
        _ = input^
        return result
    if operation == "seqmoe_check_weights" or operation == "seqmoe_weights":
        var words = _words(fields, context)
        var r = MoEBuffer(fields["router"], context)
        var gu = MoEBuffer(fields["gate_up_proj"], context)
        var dn = MoEBuffer(fields["down_proj"], context)
        _check(r, gu, dn, words)
        if operation == "seqmoe_check_weights":
            return PythonObject(None)
        return _weights(fields, seq_binding(PythonObject(None), context, values[0]), words, r, gu, dn, context)
    raise Error("ValueError: unknown MoEBlock public operation")
