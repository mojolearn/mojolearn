# SPDX-License-Identifier: Apache-2.0
# Packed storage front-end over the common linalg provider. Public Python bodies
# make one binding call; this owner admits buffers once, retains pins, and uses
# the existing linalg numerical entries without redispatching Python wrappers.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_fields
from bindings.native_loaded_module_door import native_loaded_module
from bindings.native_instance_item_pack import native_instance_item
from bindings.classical_pandas_door import classical_admit_value
from bindings.native_buffer_owner_pack import native_storage_allocate
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.classical_arrays_door import classical_wrap_array
from bindings.sequential_inputs_door import sequential_sequence_f32
from bindings.linalg_ops_buffers_unpack import lin_dtype_name
from bindings.linalg_ops_buffers_run import lin_is_f32
from bindings.admitted_input_unpack import AdmittedBuf, AdmittedTypedPin, admitted_typed_once, admitted_typed_c, admitted_shape_tuple, admitted_unfilled
from bindings.linalg_ops_lowbit_door import (
    lin_connected_binding, lin_lowbit, lin_call, lin_device, lin_device_binding, lin_device_context, LIN_HAS_GPU,
    LIN_HAS_CPU, lin_dev_widen_bf16, lin_dev_narrow_bf16, lin_dev_quantize_int8, lin_dev_dequantize_int8, lin_dev_copy,
)
from bindings.linalg_dev_arrays_unpack import lin_train_device, lin_is_device_array, lin_dev_device
from core.function_device import FunctionDevice, function_device_default
from core.train_dev_pool import TRAIN_DEV_F32, TRAIN_DEV_I8, TRAIN_DEV_I32, TRAIN_DEV_U16
from bindings.tensor_view import base_format, shape_text, TensorView
from bindings.lowbit_weight_run import low_empty
from bindings.lowbit_keys_unpack import lowbit_keys
from bindings.py_balanced import py_item
from bindings.linalg_ops_lowbit_run import LIN_OP_FROM_BF16, LIN_OP_DEQUANTIZE_INT8, LIN_OP_TO_BF16, LIN_OP_QUANTIZE_INT8
from bindings.operation_names_low import (
    LOW_OPN_FAMILY_LOW_IS_PACKED, LOW_OPN_FAMILY_LOW_FORMAT_OF, LOW_OPN_FAMILY_LOW_REPR, LOW_OPN_FAMILY_LOW_BF16_INIT,
    LOW_OPN_FAMILY_LOW_INT8_INIT, LOW_OPN_FAMILY_LOW_MATERIALIZE_ONE, LOW_OPN_FAMILY_LOW_WIDEN_BF16,
    LOW_OPN_FAMILY_LOW_PACK_ONE, LOW_OPN_FAMILY_LOW_UNPACK, LOW_OPN_FAMILY_LOW_PACK)



def low_fields(value: PythonObject) raises -> PythonObject:
    var fields = py_fields(value)
    var b = native_loaded_module("builtins")
    if py_type(fields) is not b.dict:
        raise Error("TypeError: packed storage requires an exact native instance dictionary")
    var keys = py_call(b.list, fields.keys())
    for i in range(len(keys)):  # small-loop(stored packed weight field names)
        if py_type(keys[i]) is not b.str:
            raise Error("TypeError: packed storage field names must be exact strings")
    return fields


def low_kind(value: PythonObject, context: PythonObject) raises -> Int:
    var nk = lowbit_keys()
    var cls = py_type(value)
    if cls is py_item(context, nk[].k_family_low_bf16):
        return 1
    if cls is py_item(context, nk[].k_family_low_int8):
        return 2
    return 0


def low_format(value: PythonObject) raises -> String:
    if py_type(value) is not py_type(PythonObject("")):
        raise Error("TypeError: mojolearn.lowbit: format must be an exact string")
    var fmt = String(py=value)
    if fmt != "float32" and fmt != "bfloat16" and fmt != "int8":
        raise Error("mojolearn.lowbit: unknown format '" + fmt + "'; one of ('float32', 'bfloat16', 'int8')")
    return fmt


def low_name(value: PythonObject) raises -> String:
    if py_type(value) is not py_type(PythonObject("")):
        raise Error("TypeError: mojolearn.lowbit: tensor names must be exact strings")
    return String(py=value)


def low_binding(mut binding: PythonObject, context: PythonObject) raises -> PythonObject:
    if binding is None:
        binding = lin_lowbit(lin_connected_binding(context), context)
    return binding


def low_wrap(kind: Int, q: AdmittedBuf, e: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = lowbit_keys()
    var result = native_storage_allocate(py_item(context, nk[].k_family_low_bf16) if kind == 1 else py_item(context, nk[].k_family_low_int8))
    var fields = py_fields(result)
    py_setitem(fields, nk[].k_bits if kind == 1 else nk[].k_codes, q.keep)
    if kind == 2:
        py_setitem(fields, nk[].k_exponents, e)
    py_setitem(fields, nk[].k_shape, admitted_shape_tuple(q.shape))
    return result


def low_admit(value: PythonObject, typestr: String, rank: Int, name: String,
              context: PythonObject, force_copy: Bool = False) raises -> AdmittedBuf:
    var input_pin = AdmittedTypedPin()
    var input = admitted_typed_once(value, context, input_pin)
    var code = base_format(input.fmt)
    var valid = (typestr == "<u2" and code == "H" and input.itemsize == 2
        or typestr == "<i1" and code == "b" and input.itemsize == 1
        or typestr == "<i4" and (code == "i" or code == "l") and input.itemsize == 4)
    if not valid or rank >= 0 and len(input.shape) != rank:
        if typestr == "<u2":
            raise Error("TypeError: mojolearn.lowbit.BF16Weight: bits must be a 2-D uint16 array")
        if typestr == "<i1":
            raise Error("TypeError: mojolearn.lowbit.Int8Weight: codes must be a 2-D int8 array")
        raise Error("TypeError: mojolearn.lowbit.Int8Weight: exponents must be int32 of shape (rows,)")
    if input.count() == 0:
        low_empty(name, input.shape)
    return admitted_typed_c(input, input_pin^, typestr, name, context, force_copy)


def low_exponents(value: PythonObject, rows: Int, context: PythonObject,
                  force_copy: Bool = False) raises -> AdmittedBuf:
    var e = low_admit(value, "<i4", 1, "exponents", context, force_copy)
    if e.dim(0) != rows:
        raise Error("TypeError: mojolearn.lowbit.Int8Weight: exponents must be int32 of shape (rows,)")
    return e^


# lane vdf-linalg5 (2026-11-24): TrainDeviceArray tensors (bindings/linalg_dev_arrays_unpack.mojo, the training ops'
# one device-array convention) are packed, widened and materialized ON THE DEVICE by the linalg device entries
# (gemm/linalg_dev_arrays.mojo): a packed weight of device tensors holds device arrays in its bits / codes /
# exponents fields, and materializes into a float32 device array. `held` is the published (class, owner context)
# pair, None before the training module's import (then no device array exists): one identity compare per tensor.


def low_is_dev(value: PythonObject, held: PythonObject) raises -> Bool:
    return held is not None and lin_is_device_array(value, held)


def low_wrap_dev(kind: Int, q: PythonObject, e: PythonObject, shape: PythonObject,
                 context: PythonObject) raises -> PythonObject:
    """A packed weight over device arrays (bits, or codes and exponents) of the float32 tensor's `shape`."""
    var nk = lowbit_keys()
    var result = native_storage_allocate(py_item(context, nk[].k_family_low_bf16) if kind == 1 else py_item(context, nk[].k_family_low_int8))
    var fields = py_fields(result)
    py_setitem(fields, nk[].k_bits if kind == 1 else nk[].k_codes, q)
    if kind == 2:
        py_setitem(fields, nk[].k_exponents, e)
    py_setitem(fields, nk[].k_shape, shape)
    return result


def low_pack_dev(value: PythonObject, fmt: String, device: FunctionDevice, mut binding: PythonObject,
                 held: PythonObject, context: PythonObject) raises -> PythonObject:
    """`pack_one` of a float32 TrainDeviceArray: 'float32' returns a device copy (lane vdf-linalg6: as the host path
    returns a copy; it was the caller's array itself, aliased); 'bfloat16' / 'int8' convert on the device into a
    packed weight of device arrays."""
    lin_dev_device(device, "mojolearn.lowbit.pack_one")
    if fmt == "float32":
        return lin_dev_copy(value, TRAIN_DEV_F32, "tensor", low_binding(binding, context), held, context)
    var b = native_loaded_module("builtins")
    var lk = lowbit_keys()
    var shape = py_call(b.tuple, py_item(py_fields(value), lk[].k_shape))
    if fmt == "bfloat16":
        return low_wrap_dev(1, lin_dev_narrow_bf16(value, low_binding(binding, context), held, context), PythonObject(None),
                            shape, context)
    var pair = lin_dev_quantize_int8(value, low_binding(binding, context), held, context)
    return low_wrap_dev(2, pair[0], pair[1], shape, context)


def low_materialize(value: PythonObject, mut binding: PythonObject, context: PythonObject) raises -> PythonObject:
    """A packed weight's float32 tensor (host storage: today's route; no device-array test)."""
    return low_materialize_on(value, binding, context, PythonObject(None), function_device_default())


def low_materialize_on(value: PythonObject, mut binding: PythonObject, context: PythonObject,
                       held: PythonObject, device: FunctionDevice) raises -> PythonObject:
    """`low_materialize` that also reads a packed weight of device arrays on the device (`held` not None)."""
    var kind = low_kind(value, context)
    if kind == 0:
        return value
    var nk = lowbit_keys()
    var fields = low_fields(value)
    var stored = native_instance_item(fields, nk[].k_bits if kind == 1 else nk[].k_codes)
    if low_is_dev(stored, held):
        # a packed weight of device arrays widens / dequantizes on the device (refused with device="cpu")
        lin_dev_device(device, "mojolearn.lowbit")
        if kind == 1:
            return lin_dev_widen_bf16(stored, low_binding(binding, context), held, context)
        return lin_dev_dequantize_int8(stored, native_instance_item(fields, nk[].k_exponents), low_binding(binding, context),
                                       held, context)
    var q = low_admit(native_instance_item(fields, nk[].k_bits if kind == 1 else nk[].k_codes), "<u2" if kind == 1 else "<i1", 2, "weight", context)
    # lane fix-output-zero-fill (2026-11-14): unfilled; from_bf16 / dequantize_int8 write every cell.
    var result = admitted_unfilled(q.shape, "<f4", context)
    if kind == 1:
        _ = low_binding(binding, context)
        _ = lin_call(context, LIN_OP_FROM_BF16, [result.addr, q.addr], [q.count()])
    else:
        var e = low_exponents(native_instance_item(fields, nk[].k_exponents), q.dim(0), context)
        _ = low_binding(binding, context)
        _ = lin_call(context, LIN_OP_DEQUANTIZE_INT8, [result.addr, q.addr, e.addr], [q.dim(0), q.dim(1)])
        _ = e^
    _ = q^
    return result.keep


def low_pack_admitted(input: TensorView, var input_pin: AdmittedTypedPin, fmt: String, name: String,
                      mut binding: PythonObject, context: PythonObject) raises -> PythonObject:
    if not lin_is_f32(input):
        raise Error("TypeError: mojolearn.lowbit: " + name + " has dtype " + lin_dtype_name(input_pin.keep, input) + "; pack takes float32")
    if len(input.shape) != 2:
        raise Error("mojolearn.lowbit: " + name + " must be 2-D to pack, got shape " + shape_text(input.shape))
    if input.count() == 0:
        low_empty(name, input.shape)
    var a = admitted_typed_c(input, input_pin^, "<f4", name, context, fmt == "float32")
    if fmt == "float32":
        return a.keep
    # lane fix-output-zero-fill (2026-11-14): unfilled; to_bf16 / quantize_int8 write every code and exponent.
    var q = admitted_unfilled(a.shape, "<u2" if fmt == "bfloat16" else "<i1", context)
    var result = PythonObject(None)
    if fmt == "bfloat16":
        _ = low_binding(binding, context)
        _ = lin_call(context, LIN_OP_TO_BF16, [q.addr, a.addr], [a.count()])
        result = low_wrap(1, q, PythonObject(None), context)
    else:
        var e = admitted_unfilled([a.dim(0)], "<i4", context)
        _ = low_binding(binding, context)
        _ = lin_call(context, LIN_OP_QUANTIZE_INT8, [q.addr, e.addr, a.addr], [a.dim(0), a.dim(1)])
        result = low_wrap(2, q, e.keep, context)
        _ = e^
    _ = a^
    _ = q^
    return result


def low_values(weights: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    if py_type(weights) is b.dict:
        return py_call(b.list, weights.values())
    if py_type(weights) is b.list or py_type(weights) is b.tuple:
        return weights
    raise Error("TypeError: mojolearn.lowbit: use a native dict, list or tuple; Python iteration hooks are not supported")


def low_formats(weights: PythonObject, context: PythonObject) raises -> String:
    var values = low_values(weights)
    var kind = 0
    for i in range(len(values)):  # driver-loop: over named packed-storage owners
        var current = low_kind(values[i], context)
        if current != 0:
            if kind != 0 and current != kind:
                raise Error("mojolearn.lowbit: weights mix formats ['bfloat16', 'int8']; store a model one way")
            kind = current
    return "float32" if kind == 0 else ("bfloat16" if kind == 1 else "int8")


def lowbit_weight_unknown_operation(operation: String) raises:
    """The error of a `family_low_*` name that is not an operation (it has no
    leaf code): the owner's unchanged text. Always raises."""
    raise Error("RuntimeError: unknown lowbit storage operation " + operation)


def _low_public_name(operation: Int) -> String:
    """The public name a `device=` refusal of `operation` leads with."""
    if operation == LOW_OPN_FAMILY_LOW_MATERIALIZE_ONE:
        return "mojolearn.lowbit.materialize_one"
    if operation == LOW_OPN_FAMILY_LOW_WIDEN_BF16:
        return "mojolearn.lowbit.widen_bf16"
    if operation == LOW_OPN_FAMILY_LOW_PACK_ONE:
        return "mojolearn.lowbit.pack_one"
    if operation == LOW_OPN_FAMILY_LOW_UNPACK:
        return "mojolearn.lowbit.unpack"
    return "mojolearn.lowbit.pack"


def lowbit_weight_operation(model: PythonObject, operation: Int, args: PythonObject,
                            var context: PythonObject) raises -> PythonObject:
    # bw2-linalg: `operation` is the leaf code of bindings/operation_names_low.mojo,
    # resolved once by the family door (it was up to eleven String compares here).
    classical_connect_numpy(context)
    var binding = PythonObject(None)
    var held = PythonObject(None)
    var device = function_device_default()
    if (operation == LOW_OPN_FAMILY_LOW_MATERIALIZE_ONE or operation == LOW_OPN_FAMILY_LOW_WIDEN_BF16
            or operation == LOW_OPN_FAMILY_LOW_PACK_ONE or operation == LOW_OPN_FAMILY_LOW_UNPACK
            or operation == LOW_OPN_FAMILY_LOW_PACK):
        # lane vdf-linalg: `device` is the last item of a computing function's
        # tuple, read once before any work (None: this one identity compare,
        # and `low_binding` resolves today's binding lazily as before). An
        # explicit device runs the linalg image of that side in its own module
        # state (bindings/linalg_ops_lowbit_door.mojo `lin_device_context`),
        # resolved here once so every later `low_binding` reuses it.
        var item = args[len(args) - 1]
        # lane vdf-linalg5: the published device-array class, read before an explicit "cpu" swaps the context
        held = lin_train_device(context)
        if item is not None:
            device = lin_device(item, _low_public_name(operation), LIN_HAS_GPU, LIN_HAS_CPU, context)
            if device.host_beside:
                context = lin_device_context(device, context)
                binding = lin_lowbit(lin_device_binding(device, context), context)
    if operation == LOW_OPN_FAMILY_LOW_IS_PACKED:
        return PythonObject(low_kind(args[0], context) != 0)
    if operation == LOW_OPN_FAMILY_LOW_FORMAT_OF:
        return PythonObject(low_formats(args[0], context))
    if operation == LOW_OPN_FAMILY_LOW_REPR:
        var kind = low_kind(model, context)
        var nk = lowbit_keys()
        var fields = low_fields(model)
        var shape = native_instance_item(fields, nk[].k_shape)
        classical_admit_value(shape, context)
        var text = String("BF16Weight" if kind == 1 else "Int8Weight")
        return PythonObject(text + "(shape=" + String(py=py_call(native_loaded_module("builtins").repr, shape)) + ")")
    if operation == LOW_OPN_FAMILY_LOW_BF16_INIT or operation == LOW_OPN_FAMILY_LOW_INT8_INIT:
        var kind = 1 if operation == LOW_OPN_FAMILY_LOW_BF16_INIT else 2
        if low_kind(model, context) != kind:
            raise Error("TypeError: lowbit storage requires its native public class without Python subclass hooks")
        var q = low_admit(args[0], "<u2" if kind == 1 else "<i1", 2, "bits" if kind == 1 else "codes", context)
        var fields = low_fields(model)
        var nk = lowbit_keys()
        if kind == 2:
            var e = low_exponents(args[1], q.dim(0), context)
            py_setitem(fields, nk[].k_codes, q.keep)
            py_setitem(fields, nk[].k_exponents, e.keep)
            _ = e^
        else:
            py_setitem(fields, nk[].k_bits, q.keep)
        py_setitem(fields, nk[].k_shape, admitted_shape_tuple(q.shape))
        _ = q^
        return PythonObject(None)
    if operation == LOW_OPN_FAMILY_LOW_MATERIALIZE_ONE:
        return low_materialize_on(args[0], binding, context, held, device)
    if operation == LOW_OPN_FAMILY_LOW_WIDEN_BF16:
        if low_is_dev(args[0], held):
            lin_dev_device(device, "mojolearn.lowbit.widen_bf16")
            return lin_dev_widen_bf16(args[0], low_binding(binding, context), held, context)
        var input_pin = AdmittedTypedPin()
        var input = admitted_typed_once(args[0], context, input_pin)
        var a = admitted_typed_c(input, input_pin^, "<u2", "bits", context)
        if a.count() == 0:
            raise Error("mojolearn.linalg.from_bf16: bits has no elements")
        var result = admitted_unfilled(a.shape, "<f4", context)
        _ = low_binding(binding, context)
        _ = lin_call(context, LIN_OP_FROM_BF16, [result.addr, a.addr], [a.count()])
        _ = a^
        return result.keep
    if operation == LOW_OPN_FAMILY_LOW_PACK_ONE:
        var fmt = low_format(args[1])
        var name = low_name(args[2])
        if low_is_dev(args[0], held):
            return low_pack_dev(args[0], fmt, device, binding, held, context)
        var input_pin = AdmittedTypedPin()
        var input = admitted_typed_once(args[0], context, input_pin)
        return low_pack_admitted(input, input_pin^, fmt, name, binding, context)
    var b = native_loaded_module("builtins")
    if operation == LOW_OPN_FAMILY_LOW_UNPACK:
        if py_type(args[0]) is not b.dict:
            return Python.tuple(args[0], "float32")
        var fmt = low_formats(args[0], context)
        if fmt == "float32":
            return Python.tuple(args[0], fmt)
        var result = Python.dict()
        var pairs = py_call(b.list, args[0].items())
        for i in range(len(pairs)):  # driver-loop: over named weights through one admitted provider
            var name = low_name(pairs[i][0])
            py_setitem(result, pairs[i][0], low_materialize_on(pairs[i][1], binding, context, held, device))
        return Python.tuple(result, fmt)
    if operation == LOW_OPN_FAMILY_LOW_PACK:
        var fmt = low_format(args[1])
        if py_type(args[0]) is not b.dict:
            raise Error("TypeError: mojolearn.lowbit.pack: weights must be a dict keyed by name; Python mapping hooks are not supported")
        var nk = lowbit_keys()
        var result = Python.dict()
        var pairs = py_call(b.list, args[0].items())
        for i in range(len(pairs)):  # driver-loop: over named weights through one admitted provider
            var key = pairs[i][0]
            var name = low_name(key)
            var value = pairs[i][1]
            var kind = low_kind(value, context)
            if kind == 0 and low_is_dev(value, held):
                # lane vdf-linalg5: a float32 device tensor packs on the device
                py_setitem(result, key, low_pack_dev(value, fmt, device, binding, held, context))
                continue
            if kind != 0 and held is not None and low_is_dev(native_instance_item(low_fields(value),
                    nk[].k_bits if kind == 1 else nk[].k_codes), held):
                # a packed weight of device arrays: the same format copies its device arrays device to device
                # (lane vdf-linalg6: as the host repack copies its packed storage; they were shared with the input);
                # another format materializes on the device, then packs there
                if kind == 1 and fmt == "bfloat16" or kind == 2 and fmt == "int8":
                    var dev_fields = low_fields(value)
                    var low_b = low_binding(binding, context)
                    var q_copy = lin_dev_copy(native_instance_item(dev_fields, nk[].k_bits if kind == 1 else nk[].k_codes),
                                              TRAIN_DEV_U16 if kind == 1 else TRAIN_DEV_I8, name, low_b, held, context)
                    var e_copy = PythonObject(None)
                    if kind == 2:
                        e_copy = lin_dev_copy(native_instance_item(dev_fields, nk[].k_exponents), TRAIN_DEV_I32, name,
                                              low_b, held, context)
                    py_setitem(result, key, low_wrap_dev(kind, q_copy, e_copy, native_instance_item(dev_fields, nk[].k_shape),
                                                         context))
                else:
                    py_setitem(result, key, low_pack_dev(low_materialize_on(value, binding, context, held, device), fmt, device,
                                                         binding, held, context))
                continue
            if kind != 0 and (kind == 1 and fmt == "bfloat16" or kind == 2 and fmt == "int8"):
                # Same-format repacking copies packed storage directly. It must
                # not widen/requantize, which changes both work and ownership.
                var fields = low_fields(value)
                var q = low_admit(native_instance_item(fields, nk[].k_bits if kind == 1 else nk[].k_codes), "<u2" if kind == 1 else "<i1", 2, name, context, True)
                if kind == 1:
                    py_setitem(result, key, low_wrap(1, q, PythonObject(None), context))
                else:
                    var e = low_exponents(native_instance_item(fields, nk[].k_exponents), q.dim(0), context, True)
                    py_setitem(result, key, low_wrap(2, q, e.keep, context))
                    _ = e^
                _ = q^
                continue
            if kind != 0:
                value = low_materialize(value, binding, context)
            if py_type(value) is b.list or py_type(value) is b.tuple:
                # Keep the existing direct float32 sequence construction and
                # its large-integer/double-rounding behavior.
                classical_admit_value(value, context)
                var sequence = sequential_sequence_f32(value, context)
                value = classical_wrap_array(sequence[1], sequence[0], "<f4", "C", context)
            var input_pin = AdmittedTypedPin()
            var input = admitted_typed_once(value, context, input_pin)
            if fmt != "float32" and len(input.shape) == 2:
                py_setitem(result, key, low_pack_admitted(input, input_pin^, fmt, name, binding, context))
            else:
                py_setitem(result, key, value)
                _ = input_pin^
        return result
    # Unreachable from the family door (every leaf is handled above).
    raise Error("RuntimeError: unknown lowbit storage operation " + String(operation))
