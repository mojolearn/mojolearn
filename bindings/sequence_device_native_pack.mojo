# SPDX-License-Identifier: Apache-2.0
# Native sequence tensor ownership, buffer IO and consumed float32 outputs.
# The PACK half of the former `bindings/sequence_device_native.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `bindings/sequence_device_native_unpack.mojo`.
from std.python import Python, PythonObject
from bindings.optimizer_container_boundary_unpack import OptimizerInputOwners
from std.python._cpython import PyObjectPtr
from bindings.optimizer_public_words_unpack import optimizer_python_error
from bindings.py_balanced import py_call, py_type, py_get
from bindings.py_typed_call import py_call_typed
from bindings.native_loaded_module_door import native_loaded_module
from bindings.optimizer_admission_door import opt_scalar, opt_value
from bindings.optimizer_device_native_door import device_shape, shape_count
from bindings.optimizer_public_door import tensor_view
from bindings.tensor_view import TensorView, base_format
from training.optimizer_input_plan import same_shape
from providers.python.array_layout_boundary_pack import native_shape_pack
from bindings.sequence_optimizer_native_door import seq_binding, seq_device, seq_device_get, seq_cast
from bindings.classical_arrays_door import classical_input, classical_typed, classical_address
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.sequence_array_native_unpack import sequence_array_cast
from bindings.native_numpy_metadata_unpack import native_numpy_dtype


from bindings.sequence_device_native_unpack import (
    seq_device_input,
    seq_device_make,
    seq_device_set,
)


def seq_device_entries(binding: PythonObject, context: PythonObject, required: Bool = True) raises -> PythonObject:
    var entries = seq_binding(PythonObject(None), context, binding)
    for name in ["seq_tensor_alloc", "seq_tensor_free", "seq_tensor_upload", "seq_tensor_download"]:  # driver-loop: over fixed sequence buffer ABI entries
        if py_get(entries, PythonObject(name)) is None:
            if required:
                raise Error("RuntimeError: this native sequence binding has no device tensors")
            return PythonObject(None)
    return entries


def sequence_device_finalize(owner: PythonObject, values: PythonObject,
                             context: PythonObject) raises -> PythonObject:
    # Finalizers can run after sys.modules["builtins"] has been cleared. Use
    # only the exact types, native descriptors and ABI entries retained at
    # connection time; importing or resolving normal operation metadata here
    # would prevent release during interpreter shutdown.
    var types = context["sequence_finalizer_types"]
    if py_type(values) is not types[0]:
        raise Error("TypeError: optimizer requires native argument tuples")
    if py_type(owner) is not context["sequence_device_class"]:
        raise Error("TypeError: sequence device operation requires its exact native owner")
    var binding = PythonObject(None)
    var base = PythonObject(None)
    var handle = PythonObject(None)
    try:
        binding = seq_device_get(owner, "b", context)
        base = seq_device_get(owner, "_base", context)
        handle = seq_device_get(owner, "h", context)
    except:
        # Failed constructors may have installed only some native slots.
        return PythonObject(None)
    if binding is None or base is not None:
        return PythonObject(None)
    if py_type(handle) is not types[2]:
        raise Error("TypeError: sequence tensor cleanup requires its native integer handle")
    # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
    var connected = Optional[PythonObject]()
    var bindings = context["sequence_bindings"]
    for name in ["identical", "fast"]:  # driver-loop: over two retained compiled numeric-mode owners
        var candidate = bindings[PythonObject(name)]
        if py_type(candidate) is types[1] and binding is candidate["module"]:
            connected = candidate
            break
    if not connected:
        var extra = py_get(context, "sequence_extra_bindings", Python.tuple())
        for i in range(len(extra)):  # driver-loop: fixed import-connected buffer owners
            if py_type(extra[i]) is types[1] and binding is extra[i]["module"]:
                connected = extra[i]
                break
    if not connected:
        raise Error("TypeError: sequence device/state requires its connected native binding")
    var entries = connected.value()
    var release = py_get(entries, "seq_tensor_free")
    if py_type(release) is not types[3]:
        raise Error("TypeError: sequence tensor cleanup requires its compiled release entry")
    try:
        _ = py_call(release, handle)
    except:
        # Keep ownership intact if release fails; a subsequent attempt can retry.
        return PythonObject(None)
    _ = py_call(context["sequence_device_setters"]["b"], owner, PythonObject(None))
    return PythonObject(None)



def sequence_tensor_finalize_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    # The installed native descriptor supplies (context, owner). In particular,
    # releasing temporary resident tensors inside another native operation must
    # not enter the former Python __del__ wrapper. The retained release path
    # remains usable during interpreter shutdown without module imports.
    try:
        var args = PythonObject(from_borrowed=args_ptr)
        if len(args) != 2:
            raise Error("TypeError: sequence tensor finalizer requires context and owner")
        return sequence_device_finalize(args[1], Python.tuple(), args[0]).steal_data()
    except error:
        return optimizer_python_error(error)


def sequence_device_operation(operation: String, owner: PythonObject, values: PythonObject,
                              mode: PythonObject, context: PythonObject) raises -> PythonObject:
    var buffer_pins = OptimizerInputOwners()
    try:
        var b = native_loaded_module("builtins")
        if operation == "seqdev_is":
            return PythonObject(py_type(values[0]) is context["sequence_device_class"])
        if operation == "seqdev_size_of":
            return PythonObject(shape_count(device_shape(values[0], context)))
        if operation == "seqdev_construct":
            if py_type(owner) is not context["sequence_device_class"]:
                raise Error("TypeError: sequence device constructor requires its exact owner")
            var entries = seq_device_entries(values[0], context)
            var shape = device_shape(values[1], context)
            _ = shape_count(shape)
            opt_scalar(values[2], context)
            var base = values[3]
            if base is not None:
                if not seq_device(base, context) or seq_device_get(base, "b", context) is not entries["module"]:
                    raise Error("TypeError: sequence tensor view must retain its native binding owner")
            seq_device_set(owner, entries["module"], shape, py_call(b.int, values[2]), base, context)
            return PythonObject(None)
        if operation == "seqdev_new":
            if owner is not context["sequence_device_class"]:
                raise Error("TypeError: sequence tensor allocation requires its exact native class")
            var entries = seq_device_entries(values[0], context)
            var shape = device_shape(values[1], context)
            opt_scalar(values[2], context)
            var handle = py_call_typed(entries["seq_tensor_alloc"], max(1, shape_count(shape)), Int(Bool(py=values[2])))
            return seq_device_make(entries["module"], shape, handle, PythonObject(None), context)
        if operation == "seqdev_to" or operation == "seqdev_wrap":
            var source = values[0] if operation == "seqdev_to" else values[1]
            if operation == "seqdev_to" and seq_device(source, context):
                return source
            var input = seq_device_input(source, operation == "seqdev_to", context, buffer_pins)
            if operation == "seqdev_to" and Bool(py=context["sequence_device_io_off"]):
                return input
            var selected = seq_binding(mode, context, PythonObject(None)) if operation == "seqdev_to" else seq_binding(mode, context, values[0])
            var entries = seq_device_entries(selected["module"], context, operation != "seqdev_to")
            if entries is None:
                return input
            var shape = device_shape(input.shape, context)
            var n = shape_count(shape)
            var handle = py_call_typed(entries["seq_tensor_alloc"], max(n, 1), 0)
            try:
                if n > 0:
                    _ = py_call_typed(entries["seq_tensor_upload"], handle, classical_address(input), n)
            except error:
                _ = py_call_typed(entries["seq_tensor_free"], handle)
                raise error^
            _ = input^
            return seq_device_make(entries["module"], shape, handle, PythonObject(None), context)
        if py_type(owner) is not context["sequence_device_class"]:
            raise Error("TypeError: sequence device operation requires its exact native owner")
        _ = seq_device(owner, context)
        var binding = seq_device_get(owner, "b", context)
        var entries = seq_device_entries(binding, context)
        var shape = device_shape(seq_device_get(owner, "shape", context), context)
        var n = shape_count(shape)
        var handle = seq_device_get(owner, "h", context)
        if operation == "seqdev_size":
            return PythonObject(n)
        if operation == "seqdev_ndim":
            return PythonObject(len(shape))
        if operation == "seqdev_host" or operation == "seqdev_array":
            if operation == "seqdev_array":
                # Validate before classical_connect_numpy's first-call setup can
                # touch the live numpy.dtype field, including late NumPy import.
                _ = native_numpy_dtype(context)
            var input = context["input_context"]
            classical_connect_numpy(input)
            var result = classical_typed(native_shape_pack(shape), "<f4", input)
            if n > 0:
                _ = py_call_typed(entries["seq_tensor_download"], handle, classical_address(result), n)
            return result if operation == "seqdev_host" else sequence_array_cast(result, values[0], context)
        if operation == "seqdev_upload":
            var input = seq_device_input(values[0], False, context, buffer_pins)
            if not same_shape(device_shape(input.shape, context), shape):
                raise Error("ValueError: uploaded array shape does not match the device tensor")
            if n > 0:
                _ = py_call_typed(entries["seq_tensor_upload"], handle, classical_address(input), n)
            _ = input^
            return owner
        if operation == "seqdev_copy":
            if py_get(entries, "seq_tensor_copy") is None:
                raise Error("RuntimeError: native binding has no device-to-device tensor copy")
            var result_handle = py_call_typed(entries["seq_tensor_alloc"], max(n, 1), 0)
            try:
                if n > 0:
                    _ = py_call_typed(entries["seq_tensor_copy"], result_handle, handle, n)
            except error:
                _ = py_call_typed(entries["seq_tensor_free"], result_handle)
                raise error^
            return seq_device_make(binding, shape, result_handle, PythonObject(None), context)
        if operation == "seqdev_reshape":
            var requested = values[0]
            opt_scalar(requested, context)
            if len(requested) == 1 and (py_type(requested[0]) is b.list or py_type(requested[0]) is b.tuple):
                requested = requested[0]
            var dims = List[Int]()
            var infer = -1
            for i in range(len(requested)):  # driver-loop: over requested view shape metadata
                var dim = Int(py=py_call(b.int, requested[i]))
                if dim == -1 and infer < 0:
                    infer = i
                    dims.append(1)
                elif dim < 0:
                    raise Error("ValueError: reshape accepts only one inferred dimension")
                else:
                    dims.append(dim)
            var known = shape_count(dims)
            if infer >= 0:
                dims[infer] = n // known if known != 0 else 0
            if shape_count(dims) != n:
                raise Error("ValueError: reshape dimensions do not match the device tensor size")
            return seq_device_make(binding, dims, handle, owner, context)
        raise Error("TypeError: unknown sequence device operation")
    finally:
        _ = buffer_pins^
