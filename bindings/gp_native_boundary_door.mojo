# SPDX-License-Identifier: Apache-2.0
"""GP image pins and the native GP table entry around synchronous RUN
(core/gp_entry_table.mojo; lane gp-entry-table, 2026-11-12)."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.py_balanced import py_type, py_get, py_fields
from bindings.native_loaded_module_door import native_loaded_module
from bindings.gp_kernel_storage_boundary import GPPackedKernel
from bindings.classical_arrays_door import classical_address
from core.gp_entry_table import GPEntryKey
from std.python.bindings import lookup_py_type_object
from providers.python.device_route_boundary_door import DeviceRouteOwner, device_route_refuse
from bindings.gp_keys_unpack import gp_keys
from gaussian_process.gp_native_boundary_run import GPNative, gp_native_run
from core.gp_native_api import (
    GPKernelView,
    GPRFitRequest, GPRQueryRequest, GPRCovarianceRequest, GPRSampleRequest,
    GPRGradientRequest, GPROptimizeRequest, GPCFitRequest, GPCFitAllRequest,
    GPCQueryRequest, GPCQueryAllRequest, GPCCombineRequest, GPThetaRequest,
    GPQueryResult,
)


def gp_entry_key(binding: PythonObject) raises -> GPEntryKey:
    """UNPACK: the GP table key an image recorded as its module attribute
    `gp_native_key` = `(mode, vendor, column)` when it published
    (core/gp_entry_table.mojo). One exact read of the module's `__dict__`."""
    var b = native_loaded_module("builtins")
    if py_type(binding) is not py_type(b):
        raise Error("TypeError: GP execution requires its connected native module")
    var nk = gp_keys()
    var key = py_get(py_fields(binding), nk[].k_gp_native_key)
    if (key is None or py_type(key) is not b.tuple or len(key) != 3 or py_type(key[0]) is not b.int
            or py_type(key[1]) is not b.int or py_type(key[2]) is not b.int):
        raise Error("ImportError: mojolearn: the GP binding publishes no native gp_native_key table key; rebuild it")
    return GPEntryKey(Int(py=key[0]), Int(py=key[1]), Int(py=key[2]))


def gp_route_is_owner(binding: PythonObject) -> Bool:
    """True when `binding` is an estimator's route owner, not a module."""
    try:
        return py_type(binding) is lookup_py_type_object[DeviceRouteOwner]()
    except:
        # The owner type is not registered in this process: no owner exists.
        return False


def gp_route_cpu_metrics(binding: PythonObject) raises -> PythonObject:
    """What `score` runs its program on: None (the process's metrics image,
    for a module and for every route but one), or, for a CPU route resolved
    beside a GPU set, the program factory of the CPU metrics image. A CPU
    metrics image that is not connected is refused here, in words, by the
    `score` that needs it."""
    if not gp_route_is_owner(binding):
        return PythonObject(None)
    var held = binding.downcast_value_ptr[DeviceRouteOwner]()
    if not held[].route.has_second:
        return PythonObject(None)
    if held[].table is None:
        device_route_refuse(binding, String("ImportError: score of a device='cpu' estimator needs the CPU metrics ")
            + "binary (_mojolearn_x_metrics_host), which is not connected in this installation")
    var factory = held[].table
    return factory


struct GPConnection(Movable):
    # `image` pins the connected module (and so the published entry's code)
    # for the public call; its address is the lease identity of
    # bindings/gp_resident_boundary.mojo. RUN reads only the typed `native`
    # connection (gaussian_process/gp_native_boundary_run.mojo).
    var image: PythonObject
    var native: GPNative

    def __init__(out self, binding: PythonObject) raises:
        # A public operation hands the estimator's route owner
        # (providers/python/gp_public_lifecycle_pyapi_door.mojo `_gp_route`):
        # its entry, tier and capability word were resolved once for the
        # estimator. Archive, model-selection and worker callers hand the
        # module, whose key and table slot are read here as before.
        if gp_route_is_owner(binding):
            var held = binding.downcast_value_ptr[DeviceRouteOwner]()
            self.native = GPNative(held[].route.entry, held[].route.mode, held[].route.flags)
            self.image = held[].images
        else:
            self.native = GPNative(gp_entry_key(binding))
            self.image = binding


def gp_kernel_view(arrays: PythonObject) raises -> GPKernelView:
    var b = native_loaded_module("builtins")
    if py_type(arrays) is not b.tuple and py_type(arrays) is not b.list:
        return arrays.downcast_value_ptr[GPPackedKernel]()[].storage.view()
    # The packed storage remains pinned by the caller. This adapter also
    # serves the explicit cross-interpreter transport unpack boundary.
    return GPKernelView(Int(py=classical_address(arrays[0])), Int(py=classical_address(arrays[1])),
        Int(py=classical_address(arrays[2])), Int(py=classical_address(arrays[3])),
        Int(py=arrays[0].shape[0]), Int(py=arrays[4]))


def gp_seed_word(seed: PythonObject) raises -> UInt64:
    if py_type(seed) is not py_type(PythonObject(0)):
        raise Error("TypeError: GP seed must resolve to a native int")
    ref cpy = Python().cpython()
    var value = cpy.lib.get_function[UInt64]("PyLong_AsUnsignedLongLongMask")(seed._obj_ptr)
    _ = seed
    return value


def gp_run(connection: GPConnection, request: GPRFitRequest) raises -> Int:
    try:
        with GILReleased(Python()):
            return gp_native_run(connection.native, request)
    finally:
        _ = connection


def gp_run(connection: GPConnection, request: GPRQueryRequest) raises -> GPQueryResult:
    try:
        with GILReleased(Python()):
            return gp_native_run(connection.native, request)
    finally:
        _ = connection


def gp_run(connection: GPConnection, request: GPRCovarianceRequest) raises -> Int:
    try:
        with GILReleased(Python()):
            return gp_native_run(connection.native, request)
    finally:
        _ = connection


def gp_run(connection: GPConnection, request: GPRSampleRequest) raises -> Int:
    try:
        with GILReleased(Python()):
            return gp_native_run(connection.native, request)
    finally:
        _ = connection


def gp_run(connection: GPConnection, request: GPRGradientRequest) raises -> Int:
    try:
        with GILReleased(Python()):
            return gp_native_run(connection.native, request)
    finally:
        _ = connection


def gp_run(connection: GPConnection, request: GPROptimizeRequest) raises -> Int:
    try:
        with GILReleased(Python()):
            return gp_native_run(connection.native, request)
    finally:
        _ = connection


def gp_run(connection: GPConnection, request: GPCFitRequest) raises -> Int:
    try:
        with GILReleased(Python()):
            return gp_native_run(connection.native, request)
    finally:
        _ = connection


def gp_run(connection: GPConnection, request: GPCFitAllRequest) raises -> Int:
    try:
        with GILReleased(Python()):
            return gp_native_run(connection.native, request)
    finally:
        _ = connection


def gp_run(connection: GPConnection, request: GPCQueryRequest) raises -> Int:
    try:
        with GILReleased(Python()):
            return gp_native_run(connection.native, request)
    finally:
        _ = connection


def gp_run(connection: GPConnection, request: GPCQueryAllRequest) raises -> GPQueryResult:
    try:
        with GILReleased(Python()):
            return gp_native_run(connection.native, request)
    finally:
        _ = connection


def gp_run(connection: GPConnection, request: GPCCombineRequest) raises -> Int:
    try:
        with GILReleased(Python()):
            return gp_native_run(connection.native, request)
    finally:
        _ = connection


def gp_run(connection: GPConnection, request: GPThetaRequest) raises -> Int:
    try:
        with GILReleased(Python()):
            return gp_native_run(connection.native, request)
    finally:
        _ = connection
