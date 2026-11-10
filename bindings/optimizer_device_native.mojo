# SPDX-License-Identifier: Apache-2.0
"""Training device UNPACK/PACK; handles and transfer requests stay native."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased, PyObjectPtr
from bindings.cpython_table import cpython_table
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem
from bindings.native_class_admission import dict_keys_admitted, dict_keys_admit
from bindings.optimizer_admission import opt_value, opt_scalar
from bindings.optimizer_public_door import new_array, TensorView, tensor_view
from bindings.optimizer_container_boundary import OptimizerInputOwners
from bindings.optimizer_scalar_boundary import optimizer_integer
from bindings.optimizer_descriptor_boundary import optimizer_provider_entries, optimizer_driver, optimizer_device_available
from bindings.optimizer_public_words import optimizer_python_error
from bindings.native_buffer_owner import native_storage_allocate
from training.optimizer_input_plan import optimizer_contiguous_address
from training.optimizer_native_api import OptimizerNativeDriver
from training.optimizer_device_storage_plan import (
    optimizer_device_extent, optimizer_device_count as shape_count, optimizer_device_reshape, optimizer_device_upload,
    optimizer_device_output,
)


def device_entries(binding: PythonObject, context: PythonObject) raises -> PythonObject:
    var bindings = context["bindings"]
    for mode in ["identical", "fast"]:  # driver-loop: retained native numeric-mode owners
        var entries = bindings[PythonObject(mode)]
        if py_type(entries) is context["container_types"][2] and binding is entries["module"]:
            var provider = optimizer_provider_entries(entries)
            if not optimizer_device_available(provider):
                raise Error("RuntimeError: this build has no training device arrays (identical GPU builds only)")
            return entries
    raise Error("TypeError: training device arrays require their connected native binding owner")


def device_shape(shape: PythonObject, context: PythonObject) raises -> List[Int]:
    opt_scalar(shape, context)
    if py_type(shape) is not context["container_types"][1] and py_type(shape) is not context["container_types"][0]:
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
    py_setitem(fields, "h", handle)
    py_setitem(fields, "shape", shape)
    py_setitem(fields, "size", PythonObject(shape_count(dims)))
    py_setitem(fields, "_base", base)
    py_setitem(fields, "_b", binding)


def device_new(binding: PythonObject, handle: PythonObject, dims: List[Int], base: PythonObject, context: PythonObject) raises -> PythonObject:
    var result = native_storage_allocate(context["device_class"])
    device_set(result, binding, handle, dims, base)
    return result


struct DeviceHandleLease(Movable):
    # Only native values own the unpublished allocation. The boundary's global
    # descriptor connection pins its extension image; no Python release call.
    var driver: OptimizerNativeDriver
    var handle: Int
    var owned: Bool

    def __init__(out self, driver: OptimizerNativeDriver, handle: Int):
        self.driver = driver.copy()
        self.handle = handle
        self.owned = True

    def __del__(deinit self):
        if self.owned:
            try:
                with GILReleased(Python()):
                    _ = self.driver.free_device(self.handle)
            except:
                pass  # retain the existing best-effort failure cleanup


def device_adopt(result: PythonObject, binding: PythonObject, handle: Int,
    driver: OptimizerNativeDriver) raises -> PythonObject:
    var lease = DeviceHandleLease(driver, handle)
    var fields = py_fields(result)
    # Until the final field is published only the native lease owns the handle.
    py_setitem(fields, "h", PythonObject(handle))
    py_setitem(fields, "_b", binding)
    lease.owned = False
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
    var binding = entries["module"]
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
    var fields = py_call(context["device_fields_getter"], owner, context["device_class"]) if cleanup else py_fields(owner)
    if py_type(fields) is not context["container_types"][2]:
        raise Error("TypeError: training device cleanup requires native metadata")
    if not dict_keys_admitted(fields):
        var table = cpython_table()
        var position = List[Int](length=1, fill=0)
        var pair = List[PyObjectPtr](length=2, fill=PyObjectPtr())
        while table[].PyDict_Next(fields._obj_ptr, Int(position.unsafe_ptr()),
                Int(pair.unsafe_ptr()), Int(pair.unsafe_ptr() + 1)):  # driver-loop: exact external owner field names
            if py_type(PythonObject(from_borrowed=pair[0])) is not context["scalar_types"][5]:
                raise Error("TypeError: training device " + (String("cleanup requires") if cleanup else String("owner metadata requires")) + " native string keys")
        dict_keys_admit(fields)
    return fields


def device_finalize(owner: PythonObject, context: PythonObject) raises -> PythonObject:
    if py_type(owner) is not context["device_class"]:
        raise Error("TypeError: training device cleanup requires its exact native owner")
    var fields = _device_fields(owner, context, True)
    var binding = py_get(fields, "_b")
    if binding is None or py_get(fields, "_base") is not None:
        return PythonObject(None)
    var handle = py_get(fields, "h")
    if py_type(handle) is not context["scalar_types"][2]:
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
    py_setitem(fields, "_b", PythonObject(None))
    return PythonObject(None)


def device_finalize_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=args_ptr)
        _ = device_finalize(args[0], args[1])
    except error:
        return optimizer_python_error(error)
    return PythonObject(None).steal_data()


def device_operation(operation: String, owner: PythonObject, values: PythonObject, context: PythonObject,
    selected: PythonObject) raises -> PythonObject:
    if operation == "device_construct" or operation == "device_new":
        var binding = values[0]
        var entries = device_entries(binding, context)
        var dims = device_shape(values[2] if operation == "device_construct" else values[1], context)
        _ = shape_count(dims)
        if operation == "device_construct":
            if py_type(owner) is not context["device_class"]:
                raise Error("TypeError: training device array requires the exact native owner")
            opt_scalar(values[1], context)
            var base = values[3]
            if base is not None and py_type(base) is not context["device_class"]:
                raise Error("TypeError: training device array base must retain a native device-array owner")
            device_set(owner, binding, PythonObject(optimizer_integer(values[1], context)), dims, base)
            return PythonObject(None)
        return device_allocate(entries, dims, context)
    if operation == "device_upload":
        var source = values[0]
        if py_type(source) is context["device_class"]:
            opt_value(source, context)
            return source
        var entries = device_entries(selected["module"], context)
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
    if py_type(owner) is not context["device_class"]:
        raise Error("TypeError: training device operation requires its exact native owner")
    var fields = _device_fields(owner, context, False)
    opt_value(owner, context)
    var entries = device_entries(fields["_b"], context)
    if operation == "device_reshape":
        var dims = device_shape(values[0], context)
        optimizer_device_reshape(dims, Int(py=fields["size"]))
        var base = fields["_base"]
        return device_new(fields["_b"], fields["h"], dims, owner if base is None else base, context)
    if operation == "device_host":
        var dims = device_shape(fields["shape"], context)
        var result = new_array(dims, "<f4")
        var provider = optimizer_provider_entries(entries)
        var driver = optimizer_driver(provider)
        var handle = Int(py=fields["h"])
        var address = Int(py=result._addr)
        var count = Int(py=fields["size"])
        optimizer_device_output(dims, count)
        with GILReleased(Python()):
            _ = driver.download_device(handle, address, count)
        _ = provider
        _ = result
        return result
    raise Error("TypeError: unknown training device operation")
