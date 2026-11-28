# SPDX-License-Identifier: Apache-2.0
"""Training device UNPACK/PACK; handles and transfer requests stay native."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased, PyObjectPtr
from bindings.py_cpython_table import cpython_table
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem
from bindings.native_class_admission_door import dict_keys_admitted, dict_keys_admit
from bindings.optimizer_admission_door import opt_value, opt_scalar
from bindings.optimizer_public_door import new_array, new_zeros, TensorView, tensor_view
from bindings.optimizer_container_boundary_unpack import OptimizerInputOwners
from bindings.optimizer_scalar_boundary_unpack import optimizer_integer
from bindings.optimizer_descriptor_boundary_door import optimizer_provider_entries, optimizer_driver, optimizer_device_available
from bindings.optimizer_public_words_unpack import optimizer_python_error
from bindings.native_buffer_owner_pack import native_storage_allocate
from training.optimizer_input_plan import optimizer_contiguous_address
from training.optimizer_native_api import OptimizerNativeDriver
from training.optimizer_device_handle import OptimizerDeviceHandle
from training.optimizer_device_storage_plan import (
    optimizer_device_extent, optimizer_device_count as shape_count, optimizer_device_reshape, optimizer_device_upload,
    optimizer_device_output,
)
from bindings.optimizer_keys_unpack import optimizer_keys, optimizer_keys_in_finalizer
from bindings.native_loaded_module_door import native_interpreter_finalizing
from bindings.optimizer_keys_unpack import OPTIMIZER_OPN_DEVICE_CONSTRUCT, OPTIMIZER_OPN_DEVICE_HOST, OPTIMIZER_OPN_DEVICE_NEW, OPTIMIZER_OPN_DEVICE_RESHAPE, OPTIMIZER_OPN_DEVICE_UPLOAD
from bindings.py_balanced import py_item
from std.memory import unsafe_memcpy
from core.train_dev_pool import TRAIN_DEV_F32, TRAIN_DEV_I8, TRAIN_DEV_I32, train_dev_itemsize, train_dev_words
# The device array admission, allocation and publication helpers live in bindings/optimizer_device_array_door.mojo
# (moved byte for byte by lane thin-edges-1, 2026-11-28) and are exported from here under the same names.
from bindings.optimizer_device_array_door import (
    device_float32, device_entries, device_shape, device_set, device_new, DeviceHandleLease, device_adopt,
    DeviceAllocation, _device_allocate, device_allocate,
)


def _device_fields(owner: PythonObject, context: PythonObject, cleanup: Bool) raises -> PythonObject:
    var nk = optimizer_keys()
    var fields = py_call(py_item(context, nk[].k_device_fields_getter), owner, py_item(context, nk[].k_device_class)) if cleanup else py_fields(owner)
    if py_type(fields) is not py_item(context, nk[].k_container_types)[2]:
        raise Error("TypeError: training device cleanup requires native metadata")
    if not dict_keys_admitted(fields):
        var table = cpython_table()
        var position = List[Int](length=1, fill=0)
        var pair = List[PyObjectPtr](length=2, fill=PyObjectPtr())
        while table[].PyDict_Next(fields._obj_ptr, Int(position.unsafe_ptr()),
                Int(pair.unsafe_ptr()), Int(pair.unsafe_ptr() + 1)):  # driver-loop: exact external owner field names
            if py_type(PythonObject(from_borrowed=pair[0])) is not py_item(context, nk[].k_scalar_types)[5]:
                raise Error("TypeError: training device " + (String("cleanup requires") if cleanup else String("owner metadata requires")) + " native string keys")
        dict_keys_admit(fields)
    return fields


def device_finalize(owner: PythonObject, context: PythonObject) raises -> PythonObject:
    # A device array collected during interpreter teardown releases nothing:
    # the process is ending and its device memory goes with the context. The
    # free below connects the provider (device_entries ->
    # optimizer_provider_entries -> neural_descriptor_connect_factory_once),
    # whose loaded-module read is a fatal error once the interpreter's modules
    # are cleared: "PyImport_GetModuleDict: interpreter has no modules
    # dictionary", rc -6 while garbage-collecting at exit, in
    # training-device-cross-entropy, fn-lu-solve, nn-tr-optim-resident and
    # psd-causal-lm(-text) (L40S smoke of main 3cd641b37e; lane gs-fix). The
    # key-table guard below only covered a table never filled. Tested first,
    # as host_archive_finalize does; only finalizers ask, never a hot path.
    if native_interpreter_finalizing():
        return PythonObject(None)
    # A device array collected before any call filled the key table releases
    # nothing either (optimizer_keys_in_finalizer).
    var nk = optimizer_keys_in_finalizer()
    if not nk[].ready:
        return PythonObject(None)
    if py_type(owner) is not py_item(context, nk[].k_device_class):
        raise Error("TypeError: training device cleanup requires its exact native owner")
    var fields = _device_fields(owner, context, True)
    var binding = py_get(fields, nk[].k__b)
    if binding is None or py_get(fields, nk[].k__base) is not None:
        return PythonObject(None)
    var handle = py_get(fields, nk[].k_h)
    if py_type(handle) is not py_item(context, nk[].k_scalar_types)[2]:
        raise Error("TypeError: training device cleanup requires its native integer handle")
    var entries = device_entries(binding, context)
    var provider = optimizer_provider_entries(entries)
    var driver = optimizer_driver(provider)
    var native_handle = Int(py=handle)
    try:
        with GILReleased(Python()):
            _ = driver.free_device(native_handle)
    except:
        # Failed frees keep ownership fields for the existing explicit retry.
        return PythonObject(None)
    _ = provider
    py_setitem(fields, nk[].k__b, PythonObject(None))
    return PythonObject(None)


def device_finalize_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=args_ptr)
        # PyInstanceMethod(MethodType(entry, context)): the bound context comes
        # first, the owner second (as host_archive_finalize_c reads them). The
        # swapped read subscripted the owner ("'TrainDeviceArray' object is not
        # subscriptable") on every collection (smoke 2026-11-15).
        _ = device_finalize(args[1], args[0])
    except error:
        return optimizer_python_error(error)
    return PythonObject(None).steal_data()


def device_operation(operation: Int, owner: PythonObject, values: PythonObject, context: PythonObject,
    selected: PythonObject) raises -> PythonObject:
    var nk = optimizer_keys()
    if operation == OPTIMIZER_OPN_DEVICE_CONSTRUCT or operation == OPTIMIZER_OPN_DEVICE_NEW:
        var binding = values[0]
        var entries = device_entries(binding, context)
        var dims = device_shape(values[2] if operation == OPTIMIZER_OPN_DEVICE_CONSTRUCT else values[1], context)
        _ = shape_count(dims)
        if operation == OPTIMIZER_OPN_DEVICE_CONSTRUCT:
            if py_type(owner) is not py_item(context, nk[].k_device_class):
                raise Error("TypeError: training device array requires the exact native owner")
            opt_scalar(values[1], context)
            var base = values[3]
            if base is not None and py_type(base) is not py_item(context, nk[].k_device_class):
                raise Error("TypeError: training device array base must retain a native device-array owner")
            device_set(owner, binding, PythonObject(optimizer_integer(values[1], context)), dims, base)
            return PythonObject(None)
        return device_allocate(entries, dims, context)
    if operation == OPTIMIZER_OPN_DEVICE_UPLOAD:
        var source = values[0]
        if py_type(source) is py_item(context, nk[].k_device_class):
            opt_value(source, context)
            return source
        var entries = device_entries(py_item(selected, nk[].k_module), context)
        var view = TensorView()
        var buffer_pins = OptimizerInputOwners()
        if not tensor_view(source, view, buffer_pins):
            raise Error("TypeError: mojolearn.train_to_device: a must be float32")
        var count = optimizer_device_upload(view)
        var keep = List[List[Float32]]()
        var address = 0
        with GILReleased(Python()):
            address = optimizer_contiguous_address(view, keep)
        var allocation = _device_allocate(entries, view.shape, context)
        with GILReleased(Python()):
            _ = allocation.driver.upload_device(allocation.handle, address, count)
        _ = buffer_pins
        _ = keep^
        _ = source^
        return allocation.owner
    if py_type(owner) is not py_item(context, nk[].k_device_class):
        raise Error("TypeError: training device operation requires its exact native owner")
    var fields = _device_fields(owner, context, False)
    opt_value(owner, context)
    var entries = device_entries(py_item(fields, nk[].k__b), context)
    if operation == OPTIMIZER_OPN_DEVICE_RESHAPE:
        var dims = device_shape(values[0], context)
        optimizer_device_reshape(dims, Int(py=py_item(fields, nk[].k_size)))
        var base = py_item(fields, nk[].k__base)
        var view = device_new(py_item(fields, nk[].k__b), py_item(fields, nk[].k_h), dims, owner if base is None else base, context)
        # A typed array's view keeps its dtype word (lane vdf-linalg4).
        var code = py_get(fields, nk[].k_dtype)
        if code is not None:
            py_setitem(py_fields(view), nk[].k_dtype, code)
        return view
    if operation == OPTIMIZER_OPN_DEVICE_HOST:
        var dims = device_shape(py_item(fields, nk[].k_shape), context)
        var count = Int(py=py_item(fields, nk[].k_size))
        optimizer_device_output(dims, count)
        # lane vdf-linalg4: a typed device array (int8 codes and int15 planes, int32 row exponents, uint16 bf16 bits: the linalg low-bit
        # outputs; core/train_dev_pool.mojo) carries a dtype word. Its words are downloaded once into native storage
        # and exactly its elements' bytes are copied into the typed result (the word padding never reaches it).
        var code = py_get(fields, nk[].k_dtype)
        if code is not None and Int(py=code) != TRAIN_DEV_F32:
            var typed = Int(py=code)
            var itemsize = train_dev_itemsize(typed)
            var words = train_dev_words(count, itemsize)
            var typed_result = new_array(dims, "<i1" if typed == TRAIN_DEV_I8 else ("<i4" if typed == TRAIN_DEV_I32 else "<u2"))
            var stage = List[Float32](length=words, fill=Float32(0.0))
            var typed_provider = optimizer_provider_entries(entries)
            var typed_driver = optimizer_driver(typed_provider)
            var typed_handle = Int(py=py_item(fields, nk[].k_h))
            var typed_address = Int(py=typed_result._addr)
            var stage_address = Int(stage.unsafe_ptr())
            # lane vdf-linalg5: no download for zero elements (no zero-span transfer is asked of the driver).
            with GILReleased(Python()):
                if count > 0:
                    _ = typed_driver.download_device(typed_handle, stage_address, words)
                    unsafe_memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=typed_address),
                                  src=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=stage_address),
                                  count=count * itemsize)
            _ = stage^
            _ = typed_provider
            return typed_result
        # lane unfilled-storage-audit (2026-11-15): new_array is unfilled
        # (1bcfa0d69b). A rebound public size may download only a prefix of
        # `count` floats, so that result keeps zeros in its unwritten tail; a
        # full download (size == the shape count, what device_allocate sets)
        # stays unfilled: the download writes every element.
        var result = new_array(dims, "<f4") if count == shape_count(dims) else new_zeros(dims, "<f4")
        var provider = optimizer_provider_entries(entries)
        var driver = optimizer_driver(provider)
        var handle = Int(py=py_item(fields, nk[].k_h))
        var address = Int(py=result._addr)
        # lane vdf-linalg5: an empty array (the all-empty-sessions (0, out_width) view of mlp_inference_sessions,
        # bindings/training_grads_device_unpack.mojo) downloads nothing: the driver is never asked for a zero-span
        # transfer.
        if count > 0:
            with GILReleased(Python()):
                _ = driver.download_device(handle, address, count)
        _ = provider
        _ = result
        return result
    raise Error("TypeError: unknown training device operation")
