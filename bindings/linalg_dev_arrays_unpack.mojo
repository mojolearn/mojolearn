# SPDX-License-Identifier: Apache-2.0
"""UNPACK and PACK of `TrainDeviceArray` operands for the linalg and decomposition doors (lane vdf-linalg4,
2026-11-24): the ONE device-array convention of the training ops (bindings/training_layers_device_unpack.mojo,
bindings/optimizer_device_native_door.mojo), read by the linalg family.

The class and the training owner's context come from the family context (`family_train_device`, published by
python/mojolearn/_native_optimizer_bridge.py at the training module's import): until the training module is imported
no device array can exist, and the test is one dictionary read and one identity compare per argument. A device array
is admitted when it is exactly the class, belongs to a connected training binding with device arrays (the identical
GPU set), and its shape matches its stored element count; its handle is read in place by the linalg entry. Results
are allocated through the training owner (one pool handle each) and returned as new TrainDeviceArrays. Mixed host and
device operands are refused in words; so is a device array with device="cpu" (it lives on the GPU) and an `out`
argument (the device form returns a new device array).

DTYPE: the convention carries a storage dtype word (`dtype`; core/train_dev_pool.mojo codes; absent means float32):
float32 for every training op; int8 codes and int15 planes, int32 row exponents and uint16 bf16 bits for the low-bit
conversions.
"""
from std.python import Python, PythonObject
from bindings.py_balanced import py_type, py_fields, py_get, py_item, py_setitem
from bindings.optimizer_device_native_door import device_entries, device_shape, device_allocate, device_set, shape_count
from bindings.optimizer_keys_unpack import optimizer_keys
from bindings.linalg_keys_unpack import linalg_keys
from bindings.native_loaded_module_door import native_loaded_module
from core.function_device import FunctionDevice
from core.device_route import DEVICE_CPU
from core.train_dev_pool import TRAIN_DEV_F32, train_dev_itemsize, train_dev_words


def lin_train_device(context: PythonObject) raises -> PythonObject:
    """The published (TrainDeviceArray class, training owner context) pair, or None before the training module's
    import. One dictionary read per call."""
    var nk = linalg_keys()
    return py_get(context, nk[].k_family_train_device)


def lin_is_device_array(value: PythonObject, held: PythonObject) raises -> Bool:
    """True when `value` is exactly a TrainDeviceArray (`held` from `lin_train_device`, not None)."""
    return py_type(value) is held[0]


def lin_any_device_array(args: PythonObject, first: Int, stop: Int, held: PythonObject, tuple_len: Int = 0) raises -> Bool:
    """True when any of `args[first:stop]` is a TrainDeviceArray (identity compares only; False when `held` is None);
    `tuple_len` > 0: also the first item of a tuple argument of that length (matmul_int8's (codes, exponents) pairs,
    2; matmul_int15's (hi, lo, exponents) planes, 3)."""
    if held is None:
        return False
    var cls = held[0]
    var tuple_type = native_loaded_module("builtins").tuple if tuple_len > 0 else PythonObject(None)
    for i in range(first, stop):  # driver-loop: over the call's public arguments (at most seven)
        var kind = py_type(args[i])
        if kind is cls:
            return True
        # A quantized operand is a tuple of device arrays: its first item decides.
        if tuple_len > 0 and kind is tuple_type and len(args[i]) == tuple_len and py_type(args[i][0]) is cls:
            return True
    return False


def lin_dev_device(device: FunctionDevice, public_name: String) raises:
    """A device array with an explicit device="cpu" is refused (it lives on the GPU), before any work. `public_name`
    is the full public name (`mojolearn.linalg.qr`; vdf-linalg5: the door passed the full name and this prefixed it
    a second time)."""
    if device.request == DEVICE_CPU:
        raise Error("ValueError: " + public_name + " was given device='cpu' but an argument is a "
                    + "TrainDeviceArray, which lives on the GPU; copy it to the host with to_host(), or leave "
                    + "device=None")


@fieldwise_init
struct LinDevOperand(Copyable, Movable):
    """One admitted device-array operand: its pool handle, shape and storage dtype code."""
    var handle: Int
    var shape: List[Int]
    var dtype: Int

    def dim(self, axis: Int) -> Int:
        return self.shape[axis]

    def count(self) -> Int:
        var n = 1
        for i in range(len(self.shape)):  # driver-loop: over the operand's rank (shape words)
            n *= self.shape[i]
        return n

    def text(self) -> String:
        var t = String("(")
        for i in range(len(self.shape)):  # driver-loop: over the operand's rank (shape words)
            if i > 0:
                t += ", "
            t += String(self.shape[i])
        if len(self.shape) == 1:
            t += ","
        return t + ")"


def lin_dev_entries(value: PythonObject, held: PythonObject) raises -> PythonObject:
    """The training owner entries of the binding a device array belongs to (refused unless a connected training
    binding with device arrays)."""
    var nk = optimizer_keys()
    return device_entries(py_item(py_fields(value), nk[].k__b), held[1])


def lin_dev_dtype(fields: PythonObject) raises -> Int:
    """The storage dtype code of a device array's fields (absent: float32, every array made before the word)."""
    var nk = optimizer_keys()
    var code = py_get(fields, nk[].k_dtype)
    if code is None:
        return TRAIN_DEV_F32
    return Int(py=code)


def lin_dev_operand(value: PythonObject, name: String, who: String, entries: PythonObject, held: PythonObject,
                    dtype: Int, rank: Int) raises -> LinDevOperand:
    """A device-array operand of the binding `entries` belong to, of storage dtype `dtype` and rank `rank` (0: any),
    nonempty, its shape matching its stored element count. Refused in words otherwise."""
    return lin_dev_operand_of(value, name, "mojolearn.linalg.", who, entries, held, dtype, rank)


def lin_dev_operand_of(value: PythonObject, name: String, prefix: String, function: String, entries: PythonObject,
                       held: PythonObject, dtype: Int, rank: Int) raises -> LinDevOperand:
    """`lin_dev_operand` with the public name its refusals lead with, `prefix + function` (`mojolearn.` + `solve`,
    ...; lane vdf-linalg7, the dense linear algebra functions and sparse_encode), built on a refusal only."""
    var nk = optimizer_keys()
    if py_type(value) is not held[0]:
        raise Error("TypeError: " + prefix + function + ": device and host arrays are mixed; pass every operand "
                    + "through train_to_device, or none")
    var fields = py_fields(value)
    if py_item(fields, nk[].k__b) is not py_item(entries, nk[].k_module):
        raise Error("ValueError: " + prefix + function + ": " + name + " must belong to the selected native binding")
    var shape = device_shape(py_item(fields, nk[].k_shape), held[1])
    var count = shape_count(shape)
    if count != Int(py=py_item(fields, nk[].k_size)):
        raise Error("ValueError: " + prefix + function + ": " + name + " shape does not match its stored element "
                    + "count")
    if count == 0:
        raise Error(prefix + function + ": " + name + " has no elements")
    var code = lin_dev_dtype(fields)
    if code != dtype:
        raise Error("TypeError: " + prefix + function + ": " + name + " is a device array of " + lin_dev_dtype_name(code)
                    + "; this operand takes " + lin_dev_dtype_name(dtype) + ". Refused rather than cast.")
    if rank > 0 and len(shape) != rank:
        raise Error(prefix + function + ": " + name + " must be " + String(rank) + "-D, got "
                    + String(len(shape)) + "-D")
    return LinDevOperand(Int(py=py_item(fields, nk[].k_h)), shape^, code)


def lin_dev_dtype_name(code: Int) -> String:
    if code == TRAIN_DEV_F32:
        return "float32"
    if code == 1:
        return "int8"
    if code == 2:
        return "int32"
    if code == 3:
        return "uint16"
    return "an unknown dtype"


def lin_dev_out(out: PythonObject, who: String) raises:
    """`out` is not taken with device operands: the device form returns a new device array."""
    if out is not None:
        raise Error("TypeError: mojolearn.linalg." + who + ": out is not taken with device-array operands; the "
                    + "result is returned as a new TrainDeviceArray")


def lin_dev_new(entries: PythonObject, held: PythonObject, dims: List[Int], dtype: Int) raises -> PythonObject:
    """A new device array of `dims` elements of storage dtype `dtype` (one pool handle of the words they occupy),
    published with its dtype word. Nothing is written into it here."""
    if dtype == TRAIN_DEV_F32:
        return device_allocate(entries, dims, held[1])
    var nk = optimizer_keys()
    var n = shape_count(dims)
    # The handle holds the words the elements occupy; the owner then publishes the element shape and count
    # (`device_set`, the one publication of every device array) and the dtype word, so `to_host` downloads exactly the
    # elements' bytes.
    var words = List[Int](length=1, fill=train_dev_words(n, train_dev_itemsize(dtype)))
    var storage = device_allocate(entries, words, held[1])
    var fields = py_fields(storage)
    device_set(storage, py_item(fields, nk[].k__b), py_item(fields, nk[].k_h), dims, PythonObject(None))
    py_setitem(fields, nk[].k_dtype, PythonObject(dtype))
    return storage


def lin_dev_handle(result: PythonObject) raises -> Int:
    """The pool handle of a device array the door allocated."""
    var nk = optimizer_keys()
    return Int(py=py_item(py_fields(result), nk[].k_h))
