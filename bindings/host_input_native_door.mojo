# SPDX-License-Identifier: Apache-2.0
"""Host input and constructor admission shared by the saved-model owners of every family: the class layout check
and the saved class labels (moved byte for byte from bindings/host_archive_native_door.mojo), and the X admission,
float32 cast, scalar option check and constructor fields (moved byte for byte from
bindings/host_tree_native_door.mojo), lane thin-edges-1, 2026-11-28. Both files still export these names. A
linear, kernel or density owner that only admits its input imports this file, not the archive reader of every
family or the tree owner behind it."""
from std.python import Python, PythonObject
from std.memory import bitcast
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_item
from bindings.sequential_description_unpack import sequential_fields
from bindings.sequential_admission_door import _ordinary_torch, _one_of
from bindings.classical_arrays_door import classical_input, classical_address, classical_is_array, classical_is_f32, classical_cast_kind, classical_c_order
from bindings.classical_pandas_door import classical_admit_value, native_pandas_exact, native_pandas_f32
from bindings.native_torch_admission_door import native_torch_numpy
from bindings.host_array_native_pack import host_archive_typed
from bindings.host_tree_archive_keys_unpack import host_tree_archive_keys
from bindings.admitted_input_run import RESHAPE_HINT
from core.scaler_fit_impl import scaler_cast


def host_archive_check_layout(cls: PythonObject, context: PythonObject) raises:
    var nk = host_tree_archive_keys()
    var b = native_loaded_module("builtins")
    if py_type(cls) is not b.type:
        raise Error("TypeError: native archive classes require an ordinary native class layout")
    var mro = py_call(py_item(b.type.__dict__, nk[].k___mro__).__get__, cls)
    var found_dictionary = False
    var found_finalizer = False
    for i in range(len(mro)):  # driver-loop: connected class ancestry metadata
        var fields = py_call(py_item(b.type.__dict__, nk[].k___dict__).__get__, mro[i])
        if not found_finalizer and py_contains(fields, nk[].k___del__):
            if (py_item(fields, nk[].k___del__) is not py_get(context, nk[].k_host_archive_finalizer_descriptor)
                    and py_item(fields, nk[].k___del__) is not py_get(context, nk[].k_host_archive_peer_finalizer_descriptor)):
                raise Error("TypeError: Python finalizer hooks are not accepted by native archive loading")
            found_finalizer = True
        if not found_dictionary and py_contains(fields, nk[].k___dict__):
            if py_type(py_item(fields, nk[].k___dict__)) is not py_item(context, nk[].k_host_archive_dict_descriptor_type):
                raise Error("TypeError: Python dictionary descriptors are not accepted by native archive loading")
            found_dictionary = True
    if not found_dictionary:
        raise Error("TypeError: native archive loading requires a native instance dictionary")


def host_archive_class_values(value: PythonObject, context: PythonObject) raises -> PythonObject:
    # Archive decoding produces exact strings, lists, and Arrays. Read their
    # native storage directly; Array.tolist/__iter__ and arbitrary label
    # protocols never participate in loading.
    var b = native_loaded_module("builtins")
    if py_type(value) is b.str:
        return Python.list(value)
    var result = Python.list()
    if py_type(value) is b.list or py_type(value) is b.tuple:
        for i in range(len(value)):  # driver-loop: saved class label nesting
            var nested = host_archive_class_values(value[i], context)
            _ = py_call(result.extend, nested)
        return result
    # Fetched after the str and list branches: a label list recurses here per
    # label, and those branches read no key.
    var nk = host_tree_archive_keys()
    if py_type(value) is not py_item(context, nk[].k_array_class):
        raise Error("TypeError: saved class labels must be native archive values")
    var rank = Int(py=value.ndim)
    var count = Int(py=value.size)
    var width = Int(py=value.itemsize)
    var shape = value.shape
    var strides = value.strides
    var storage = value._mv
    var half = String(value.dtype) == "<f2"
    for i in range(count):  # driver-loop: saved distinct class label metadata
        var remaining = i
        var offset = 0
        for dimension in range(rank - 1, -1, -1):  # driver-loop: saved class label shape metadata
            var length = Int(py=shape[dimension])
            offset += (remaining % length) * Int(py=strides[dimension])
            remaining //= length
        var item = storage[offset // width]
        if half:
            item = PythonObject(Float64(bitcast[DType.float16](UInt16(Int(py=item)))))
        _ = py_call(result.append, item)
    return result


def host_tree_input(source: PythonObject, context: PythonObject) raises -> PythonObject:
    classical_admit_value(source, context)
    var prepared = source
    if native_pandas_exact(source):
        prepared = native_pandas_f32(source, context, False, True)
    elif _ordinary_torch(py_type(source), context):
        prepared = native_torch_numpy(source)
    var value = classical_input(prepared, context, "X")
    var cast = host_tree_cast(value, context)
    _ = prepared^
    return cast


def host_tree_cast(value: PythonObject, context: PythonObject) raises -> PythonObject:
    """The 2-D checks and C-order float32 cast of host_tree_input for an input
    the caller already admitted and viewed (classical_input), so a door that
    inspected the view does not admit X a second time (row N279). The caller
    keeps `value` (and whatever it views) alive through this call."""
    var nk = host_tree_archive_keys()
    var rank = Int(py=value.ndim)
    if rank != 2:
        # The product's one rank refusal (bindings/admitted_family_door.mojo:167), with the reshape hint
        # below rank 2; smoke 9685746f95 found this door's bare "X must be 2-D". Refusal path only.
        var hint = String(". ") + String(RESHAPE_HINT) if rank < 2 else String("")
        raise Error("mojolearn: X must be 2-D, got " + String(rank) + "-D shape "
                    + String(py_call(value.shape.__repr__)) + hint)
    var rows = Int(py=value.shape[0])
    var cols = Int(py=value.shape[1])
    if rows < 1 or cols < 1:
        raise Error("X is empty")
    if rows > 2147483647 // cols:
        raise Error("X exceeds the native Int32 indexing bound")
    var native = classical_is_array(value, context) or Bool(py=value.dtype.isnative)
    if classical_is_f32(value, context) and classical_c_order(value, context) and native:
        return value
    var output = host_archive_typed(Python.tuple(rows, cols), "<f4", py_item(context, nk[].k_output_context))
    var item = Int(py=value.itemsize)
    var kind = classical_cast_kind(value, item, "X", context)
    scaler_cast(Int(py=classical_address(value)), Int(py=value.strides[0]), Int(py=value.strides[1]),
        rows, cols, item, kind, native,
        MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(output))))
    return output


def host_tree_constructor_fields(model: PythonObject, context: PythonObject) raises -> PythonObject:
    host_archive_check_layout(py_type(model), context)
    var fields = sequential_fields(model)
    var b = native_loaded_module("builtins")
    var keys = py_call(py_type(Python.list()), fields.keys())
    for i in range(len(keys)):  # driver-loop: native constructor dictionary admission
        if py_type(keys[i]) is not b.str:
            raise Error("TypeError: tree state names must be native strings")
    return fields


def host_tree_scalar(value: PythonObject, context: PythonObject) raises:
    var nk = host_tree_archive_keys()
    # Conversions below call only native scalar slots. Container formatting
    # can transitively invoke element repr hooks and is not a scalar contract.
    var b = native_loaded_module("builtins")
    var cls = py_type(value)
    if not _one_of(cls, Python.tuple(b.str, b.bytes, b.bytearray, b.int, b.float, b.bool, py_type(PythonObject(None)))) and not _one_of(cls, py_item(context, nk[].k_scalar_types)):
        raise Error("TypeError: tree scalar options require native scalar values without Python conversion hooks")
