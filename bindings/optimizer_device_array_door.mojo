# SPDX-License-Identifier: Apache-2.0
"""Training device arrays: admission of the owner binding and of a shape, and allocation and publication of one
array. Moved byte for byte out of bindings/optimizer_device_native_door.mojo (lane thin-edges-1, 2026-11-28), which
keeps the device operations and the finalizer and still exports every name below. The linalg family allocates its
device outputs here (bindings/linalg_dev_arrays_unpack.mojo) without importing the optimizer's public door."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.py_balanced import py_type, py_get, py_fields, py_setitem
from bindings.optimizer_admission_door import opt_scalar
from bindings.optimizer_scalar_boundary_unpack import optimizer_integer
from bindings.optimizer_descriptor_boundary_door import optimizer_provider_entries, optimizer_driver, optimizer_device_available
from bindings.native_buffer_owner_pack import native_storage_allocate
from training.optimizer_native_api import OptimizerNativeDriver
from training.optimizer_device_handle import OptimizerDeviceHandle
from training.optimizer_device_storage_plan import (
    optimizer_device_extent, optimizer_device_count as shape_count,
)
from bindings.optimizer_keys_unpack import optimizer_keys
from bindings.py_balanced import py_item
from core.train_dev_pool import TRAIN_DEV_F32


def device_float32(fields: PythonObject, name: String, where: String) raises:
    """A float32 training operand refuses a typed device array (the linalg low-bit outputs: int8 codes and int15
    planes, int32 row exponents, uint16 bf16 / float16 bits; their `dtype` word, core/train_dev_pool.mojo) in words,
    never reinterpreting its words (lane vdf-linalg5: the doors with their own class compare). One dictionary read."""
    var nk = optimizer_keys()
    var code = py_get(fields, nk[].k_dtype)
    if code is not None and Int(py=code) != TRAIN_DEV_F32:
        raise Error("TypeError: mojolearn." + where + ": " + name + " is a typed device array (int8, int32 or uint16 "
                    + "storage); this function takes float32 device arrays")


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
