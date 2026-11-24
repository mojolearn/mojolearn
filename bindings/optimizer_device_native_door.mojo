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
from bindings.optimizer_keys_unpack import OPTIMIZER_OPN_DEVICE_CONSTRUCT, OPTIMIZER_OPN_DEVICE_HOST, OPTIMIZER_OPN_DEVICE_NEW, OPTIMIZER_OPN_DEVICE_RESHAPE, OPTIMIZER_OPN_DEVICE_UPLOAD
from bindings.py_balanced import py_item
from std.memory import unsafe_memcpy
from core.train_dev_pool import TRAIN_DEV_F32, TRAIN_DEV_I8, TRAIN_DEV_I32, train_dev_itemsize, train_dev_words


def device_entries(binding: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = optimizer_keys()
    var bindings = py_item(context, nk[].k_bindings)
    for mode in ["identical", "fast"]:  # driver-loop: retained native numeric-mode owners
        var entries = bindings[PythonObject(mode)]
        if py_type(entries) is py_item(context, nk[].k_container_types)[2] and binding is py_item(entries, nk[].k_module):
            var provider = optimizer_provider_entries(entries)
            if not optimizer_device_available(provider):
                raise Error("RuntimeError: this build has no training device arrays (identical GPU builds only)")
            return entries
    raise Error("TypeError: training device arrays require their connected native binding owner")


def device_shape(shape: PythonObject, context: PythonObject) raises -> List[Int]:
    opt_scalar(shape, context)
    var nk = optimizer_keys()
    if py_type(shape) is not py_item(context, nk[].k_container_types)[1] and py_type(shape) is not py_item(context, nk[].k_container_types)[0]:
        raise Error("TypeError: training device shape requires a native tuple or list")
    var dims = List[Int]()
    for i in range(len(shape)):  # driver-loop: decode external scalar shape words under the GIL
        dims.append(optimizer_device_extent(optimizer_integer(shape[i], context)))
    return dims^


def device_set(owner: PythonObject, binding: PythonObject, handle: PythonObject, dims: List[Int], base: PythonObject) raises:
    var fields = py_fields(owner)
    ref cpy = Python().cpython()
    var raw = cpy.PyTuple_New(len(dims))
    if not raw:
        raise cpy.unsafe_get_error()
    var shape = PythonObject(from_owned=raw)
    # PACK is necessarily under the GIL; no intermediate list/append/tuple call.
    for i in range(len(dims)):  # driver-loop: fixed-rank public shape publication
        var extent = PythonObject(dims[i])
        if cpy.PyTuple_SetItem(shape._obj_ptr, i, extent.steal_data()) != 0:
            raise cpy.unsafe_get_error()
    var nk = optimizer_keys()
    py_setitem(fields, nk[].k_h, handle)
    py_setitem(fields, nk[].k_shape, shape)
    py_setitem(fields, nk[].k_size, PythonObject(shape_count(dims)))
    py_setitem(fields, nk[].k__base, base)
    py_setitem(fields, nk[].k__b, binding)


def device_new(binding: PythonObject, handle: PythonObject, dims: List[Int], base: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = optimizer_keys()
    var result = native_storage_allocate(py_item(context, nk[].k_device_class))
    device_set(result, binding, handle, dims, base)
    return result


struct DeviceHandleLease(Movable):
    # Only native values own the unpublished allocation. The boundary's global
    # descriptor connection pins its extension image; no Python release call.
    var control: OptimizerDeviceHandle

    def __init__(out self, driver: OptimizerNativeDriver, handle: Int):
        self.control = OptimizerDeviceHandle(driver, handle)

    def __del__(deinit self):
        if self.control.owned:
            try:
                with GILReleased(Python()):
                    self.control.release()
            except:
                # Preserve the existing one-attempt best-effort cleanup; the
                # native field destructor must not make a second attempt.
                _ = self.control.detach()


def device_adopt(result: PythonObject, binding: PythonObject, handle: Int,
    driver: OptimizerNativeDriver) raises -> PythonObject:
    var lease = DeviceHandleLease(driver, handle)
    var fields = py_fields(result)
    # Until the final field is published only the native lease owns the handle.
    var nk = optimizer_keys()
    py_setitem(fields, nk[].k_h, PythonObject(handle))
    py_setitem(fields, nk[].k__b, binding)
    _ = lease.control.detach()
    return result


@fieldwise_init
struct DeviceAllocation(Movable):
    var owner: PythonObject  # unpublished public output owner, never RUN state
    var handle: Int
    var driver: OptimizerNativeDriver


def _device_allocate(entries: PythonObject, dims: List[Int], context: PythonObject) raises -> DeviceAllocation:
    var result = device_new(PythonObject(None), PythonObject(None), dims, PythonObject(None), context)
    var provider = optimizer_provider_entries(entries)
    var driver = optimizer_driver(provider)
    var nk = optimizer_keys()
    var binding = py_item(entries, nk[].k_module)
    var count = shape_count(dims)
    var handle = -1
    with GILReleased(Python()):
        handle = driver.allocate_device(count)
    var published = device_adopt(result, binding, handle, driver)
    _ = provider
    return DeviceAllocation(published, handle, driver.copy())


def device_allocate(entries: PythonObject, dims: List[Int], context: PythonObject) raises -> PythonObject:
    var allocation = _device_allocate(entries, dims, context)
    return allocation.owner


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
    # A device array collected during interpreter teardown before any call
    # filled the key table releases nothing (optimizer_keys_in_finalizer).
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
            with GILReleased(Python()):
                _ = typed_driver.download_device(typed_handle, stage_address, words)
                if count > 0:
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
        with GILReleased(Python()):
            _ = driver.download_device(handle, address, count)
        _ = provider
        _ = result
        return result
    raise Error("TypeError: unknown training device operation")
