# SPDX-License-Identifier: Apache-2.0
# Native sequence tensor ownership, buffer IO and consumed float32 outputs.
# The PACK half of the former `bindings/sequence_device_native.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `bindings/sequence_device_native_unpack.mojo`.
from std.python import Python, PythonObject
from bindings.optimizer_container_boundary_unpack import OptimizerInputOwners
from std.python._cpython import PyObjectPtr
from bindings.optimizer_public_words_unpack import optimizer_python_error
from bindings.py_balanced import py_call, py_type, py_get
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
from bindings.sequence_keys_unpack import sequence_keys
from bindings.optimizer_keys_unpack import (
    OPTIMIZER_OPN_SEQDEV_ARRAY,
    OPTIMIZER_OPN_SEQDEV_CONSTRUCT,
    OPTIMIZER_OPN_SEQDEV_COPY,
    OPTIMIZER_OPN_SEQDEV_HOST,
    OPTIMIZER_OPN_SEQDEV_NDIM,
    OPTIMIZER_OPN_SEQDEV_NEW,
    OPTIMIZER_OPN_SEQDEV_RESHAPE,
    OPTIMIZER_OPN_SEQDEV_SIZE,
    OPTIMIZER_OPN_SEQDEV_SIZE_OF,
    OPTIMIZER_OPN_SEQDEV_TO,
    OPTIMIZER_OPN_SEQDEV_UPLOAD,
    OPTIMIZER_OPN_SEQDEV_WRAP,
)
from bindings.py_balanced import py_item
from bindings.sequence_ops_door import (
    SEQ_ROUTE_NATIVE, seq_ops_route, seq_tensor_new, seq_tensor_release, seq_tensor_put, seq_tensor_get,
    seq_tensor_clone,
)
from providers.python.neural_descriptor_pyapi_door import neural_descriptor_execute
from sequence.neural_model_dispatch import SEQ_TENSOR_FREE


def seq_device_entries(binding: PythonObject, context: PythonObject, required: Bool = True) raises -> PythonObject:
    # lane dw-autograd-seq (2026-11-21): the image's route word, held in its
    # entry dictionary (bindings/sequence_ops_door.mojo), answers whether it
    # has device tensors. Before: four entry-name lookups per operation. The
    # host x_sequence image is the only connected owner without them (the
    # Embedding owner is connected only with its tensor entries).
    var entries = seq_binding(PythonObject(None), context, binding)
    if seq_ops_route(entries) == SEQ_ROUTE_NATIVE:
        if required:
            raise Error("RuntimeError: this native sequence binding has no device tensors")
        return PythonObject(None)
    return entries


def _seq_owner(entries: PythonObject, shape: List[Int], handle: Int, context: PythonObject) raises -> PythonObject:
    """Wrap a handle this door just allocated in its Python owner. Until the
    owner exists the handle is this door's: if the owner cannot be made the
    handle is released here, so no tensor is left without an owner."""
    var nk = sequence_keys()
    try:
        return seq_device_make(py_item(entries, nk[].k_module), shape, PythonObject(handle), PythonObject(None), context)
    except error:
        seq_tensor_release(entries, handle)
        raise error^


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
        # bw2-sequence (2026-11-20): the owner's exact type was tested above,
        # so the three getters are called directly (seq_device_get repeated
        # that test per slot). This finalizer reads no key table: its names
        # stay literal, so a release during interpreter shutdown never runs a
        # table fill (fx-smoke-b59).
        var getters = context["sequence_device_getters"]
        binding = py_call(getters["b"], owner)
        base = py_call(getters["_base"], owner)
        handle = py_call(getters["h"], owner)
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
    # OWNERSHIP (lane dw-autograd-seq, 2026-11-21). The owner holds exactly
    # one pool handle while its binding slot is set and it is not a view
    # (`_base` is None, tested above): this finalizer is the only release of
    # it, and it clears the binding slot afterwards so a second run releases
    # nothing. The handle is released in the image that made it:
    #   an x_sequence image: SEQ_TENSOR_FREE on the provider that
    #     `seq_ops_route` connected and stored in this dictionary before any
    #     tensor of the image could be wrapped. The GIL stays held (the pool
    #     is not locked, and a finalizer must not yield during shutdown).
    #     Nothing is connected, imported or looked up in a key table here;
    #   another family's owner (the Embedding image; no whole-model
    #     descriptor): its retained compiled release entry, as before.
    # If the release fails the binding slot is kept, so ownership stays
    # exact and a later attempt can retry.
    var provider = py_get(entries, "native_model_provider")
    var release = PythonObject(None)
    if provider is None:
        if py_get(entries, "model_native_api_v1") is not None:
            # An x_sequence tensor whose image was never routed cannot have
            # been wrapped by a door; keep ownership intact, release nothing.
            return PythonObject(None)
        release = py_get(entries, "seq_tensor_free")
        if py_type(release) is not types[3]:
            raise Error("TypeError: sequence tensor cleanup requires its compiled release entry")
    try:
        if provider is None:
            _ = py_call(release, handle)
        else:
            var addresses = List[Int]()
            var ints: List[Int] = [Int(py=handle)]
            var floats = List[Float64]()
            _ = neural_descriptor_execute(provider, SEQ_TENSOR_FREE, addresses, ints, floats, False)
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


def sequence_device_operation(operation: Int, owner: PythonObject, values: PythonObject,
                              mode: PythonObject, context: PythonObject) raises -> PythonObject:
    var buffer_pins = OptimizerInputOwners()
    try:
        var b = native_loaded_module("builtins")
        if operation == OPTIMIZER_OPN_SEQDEV_SIZE_OF:
            return PythonObject(shape_count(device_shape(values[0], context)))
        var nk = sequence_keys()
        if operation == OPTIMIZER_OPN_SEQDEV_CONSTRUCT:
            if py_type(owner) is not py_item(context, nk[].k_sequence_device_class):
                raise Error("TypeError: sequence device constructor requires its exact owner")
            var entries = seq_device_entries(values[0], context)
            var shape = device_shape(values[1], context)
            _ = shape_count(shape)
            opt_scalar(values[2], context)
            var base = values[3]
            if base is not None:
                if not seq_device(base, context) or seq_device_get(base, "b", context) is not py_item(entries, nk[].k_module):
                    raise Error("TypeError: sequence tensor view must retain its native binding owner")
            seq_device_set(owner, py_item(entries, nk[].k_module), shape, py_call(b.int, values[2]), base, context)
            return PythonObject(None)
        if operation == OPTIMIZER_OPN_SEQDEV_NEW:
            if owner is not py_item(context, nk[].k_sequence_device_class):
                raise Error("TypeError: sequence tensor allocation requires its exact native class")
            var entries = seq_device_entries(values[0], context)
            var shape = device_shape(values[1], context)
            opt_scalar(values[2], context)
            var handle = seq_tensor_new(entries, max(1, shape_count(shape)), Bool(py=values[2]))
            return _seq_owner(entries, shape, handle, context)
        if operation == OPTIMIZER_OPN_SEQDEV_TO or operation == OPTIMIZER_OPN_SEQDEV_WRAP:
            var source = values[0] if operation == OPTIMIZER_OPN_SEQDEV_TO else values[1]
            if operation == OPTIMIZER_OPN_SEQDEV_TO and seq_device(source, context):
                return source
            var input = seq_device_input(source, operation == OPTIMIZER_OPN_SEQDEV_TO, context, buffer_pins)
            if operation == OPTIMIZER_OPN_SEQDEV_TO and Bool(py=py_item(context, nk[].k_sequence_device_io_off)):
                return input
            var selected = seq_binding(mode, context, PythonObject(None)) if operation == OPTIMIZER_OPN_SEQDEV_TO else seq_binding(mode, context, values[0])
            var entries = seq_device_entries(py_item(selected, nk[].k_module), context, operation != OPTIMIZER_OPN_SEQDEV_TO)
            if entries is None:
                return input
            var shape = device_shape(input.shape, context)
            var n = shape_count(shape)
            var handle = seq_tensor_new(entries, max(n, 1), False)
            try:
                if n > 0:
                    seq_tensor_put(entries, handle, Int(py=classical_address(input)), n)
            except error:
                seq_tensor_release(entries, handle)
                raise error^
            _ = input^
            return _seq_owner(entries, shape, handle, context)
        if py_type(owner) is not py_item(context, nk[].k_sequence_device_class):
            raise Error("TypeError: sequence device operation requires its exact native owner")
        _ = seq_device(owner, context)
        var binding = seq_device_get(owner, "b", context)
        var entries = seq_device_entries(binding, context)
        var shape = device_shape(seq_device_get(owner, "shape", context), context)
        var n = shape_count(shape)
        var handle_owner = seq_device_get(owner, "h", context)
        if operation == OPTIMIZER_OPN_SEQDEV_SIZE:
            return PythonObject(n)
        if operation == OPTIMIZER_OPN_SEQDEV_NDIM:
            return PythonObject(len(shape))
        if operation == OPTIMIZER_OPN_SEQDEV_HOST or operation == OPTIMIZER_OPN_SEQDEV_ARRAY:
            if operation == OPTIMIZER_OPN_SEQDEV_ARRAY:
                # Validate before classical_connect_numpy's first-call setup can
                # touch the live numpy.dtype field, including late NumPy import.
                _ = native_numpy_dtype(context)
            var input = py_item(context, nk[].k_input_context)
            classical_connect_numpy(input)
            var result = classical_typed(native_shape_pack(shape), "<f4", input)
            if n > 0:
                seq_tensor_get(entries, Int(py=handle_owner), Int(py=classical_address(result)), n)
            return result if operation == OPTIMIZER_OPN_SEQDEV_HOST else sequence_array_cast(result, values[0], context)
        if operation == OPTIMIZER_OPN_SEQDEV_UPLOAD:
            var input = seq_device_input(values[0], False, context, buffer_pins)
            if not same_shape(device_shape(input.shape, context), shape):
                raise Error("ValueError: uploaded array shape does not match the device tensor")
            if n > 0:
                seq_tensor_put(entries, Int(py=handle_owner), Int(py=classical_address(input)), n)
            _ = input^
            return owner
        if operation == OPTIMIZER_OPN_SEQDEV_COPY:
            # An x_sequence device image always has the copy; another
            # family's owner answers for itself inside `seq_tensor_clone`
            # (the same words when it has none).
            var result_handle = seq_tensor_new(entries, max(n, 1), False)
            try:
                if n > 0:
                    seq_tensor_clone(entries, result_handle, Int(py=handle_owner), n)
            except error:
                seq_tensor_release(entries, result_handle)
                raise error^
            return _seq_owner(entries, shape, result_handle, context)
        if operation == OPTIMIZER_OPN_SEQDEV_RESHAPE:
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
            return seq_device_make(binding, dims, handle_owner, owner, context)
        raise Error("TypeError: unknown sequence device operation")
    finally:
        _ = buffer_pins^
