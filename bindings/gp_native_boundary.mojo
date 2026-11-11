# SPDX-License-Identifier: Apache-2.0
"""GP provider/image and public storage pins around synchronous native RUN."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.py_balanced import py_type
from bindings.native_loaded_module import native_loaded_module
from bindings.gp_kernel_storage_boundary import GPPackedKernel
from bindings.classical_arrays import classical_address
from core.shared_slot import SharedSlot
from providers.python.neural_descriptor_pyapi import NeuralDescriptorProvider, neural_descriptor_connect_once
from core.gp_native_api import (
    GP_NATIVE_ABI, GP_NATIVE_COUNT, GP_MODE, GP_CAPABILITIES, GPDriver, GPKernelView,
    GPRFitRequest, GPRQueryRequest, GPRCovarianceRequest, GPRSampleRequest,
    GPRGradientRequest, GPROptimizeRequest, GPCFitRequest, GPCFitAllRequest,
    GPCQueryRequest, GPCQueryAllRequest, GPCCombineRequest, GPThetaRequest,
    GPQueryResult,
)


@fieldwise_init
struct GPDescriptorMetadata(Movable):
    # The Python reference pins the exact admitted provider image. Mode and
    # capability values themselves are native; no boxed cache payload exists.
    var provider: PythonObject
    var mode: Int
    var capabilities: Int


struct GPDescriptorMetadataTable(Defaultable, Movable):
    var rows: List[GPDescriptorMetadata]

    def __init__(out self):
        self.rows = List[GPDescriptorMetadata]()


comptime GP_DESCRIPTOR_METADATA = SharedSlot[GPDescriptorMetadataTable, "MojolearnGpDescriptorMetadataV1"]


def _gp_metadata(provider: PythonObject, driver: GPDriver) raises -> Tuple[Int, Int]:
    # UNPACK owns the GIL, as does the existing descriptor connection cache.
    # Holding the provider pins makes identity comparison safe from address
    # reuse. A replacement factory has a new admitted provider and new row.
    var table = GP_DESCRIPTOR_METADATA.get_or_create_ptr()
    for i in range(len(table[].rows)):  # driver-loop: the handful of connected GP provider images
        if table[].rows[i].provider is provider:
            return (table[].rows[i].mode, table[].rows[i].capabilities)
    var mode = driver.metadata(GP_MODE)
    var capabilities = driver.metadata(GP_CAPABILITIES)
    # Match the bounded retention policy of neural_descriptor_connect_once.
    if len(table[].rows) >= 64:
        table[].rows.clear()
    table[].rows.append(GPDescriptorMetadata(provider, mode, capabilities))
    return (mode, capabilities)


struct GPConnection(Movable):
    var provider: PythonObject
    var driver: GPDriver
    var mode: Int
    var capabilities: Int

    def __init__(out self, binding: PythonObject) raises:
        self.provider = neural_descriptor_connect_once(binding, "gp_native_api_v1", String(GP_NATIVE_ABI), GP_NATIVE_COUNT)
        self.driver = GPDriver(self.provider.downcast_value_ptr[NeuralDescriptorProvider]()[].entry)
        var metadata = _gp_metadata(self.provider, self.driver)
        self.mode = metadata[0]
        self.capabilities = metadata[1]


def gp_require(connection: GPConnection, capability: Int, entry: String) raises:
    if (connection.capabilities & capability) == 0:
        raise Error("ImportError: the connected GP provider does not support " + entry)


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
            return connection.driver.gpr_fit(request)
    finally:
        _ = connection


def gp_run(connection: GPConnection, request: GPRQueryRequest) raises -> GPQueryResult:
    try:
        with GILReleased(Python()):
            return connection.driver.gpr_query(request)
    finally:
        _ = connection


def gp_run(connection: GPConnection, request: GPRCovarianceRequest) raises -> Int:
    try:
        with GILReleased(Python()):
            return connection.driver.gpr_covariance(request)
    finally:
        _ = connection


def gp_run(connection: GPConnection, request: GPRSampleRequest) raises -> Int:
    try:
        with GILReleased(Python()):
            return connection.driver.gpr_sample(request)
    finally:
        _ = connection


def gp_run(connection: GPConnection, request: GPRGradientRequest) raises -> Int:
    try:
        with GILReleased(Python()):
            return connection.driver.gpr_gradient(request)
    finally:
        _ = connection


def gp_run(connection: GPConnection, request: GPROptimizeRequest) raises -> Int:
    try:
        with GILReleased(Python()):
            return connection.driver.gpr_optimize(request)
    finally:
        _ = connection


def gp_run(connection: GPConnection, request: GPCFitRequest) raises -> Int:
    try:
        with GILReleased(Python()):
            return connection.driver.gpc_fit(request)
    finally:
        _ = connection


def gp_run(connection: GPConnection, request: GPCFitAllRequest) raises -> Int:
    try:
        with GILReleased(Python()):
            return connection.driver.gpc_fit_all(request)
    finally:
        _ = connection


def gp_run(connection: GPConnection, request: GPCQueryRequest) raises -> Int:
    try:
        with GILReleased(Python()):
            return connection.driver.gpc_query(request)
    finally:
        _ = connection


def gp_run(connection: GPConnection, request: GPCQueryAllRequest) raises -> GPQueryResult:
    try:
        with GILReleased(Python()):
            return connection.driver.gpc_query_all(request)
    finally:
        _ = connection


def gp_run(connection: GPConnection, request: GPCCombineRequest) raises -> Int:
    try:
        with GILReleased(Python()):
            return connection.driver.gpc_combine(request)
    finally:
        _ = connection


def gp_run(connection: GPConnection, request: GPThetaRequest) raises -> Int:
    try:
        with GILReleased(Python()):
            return connection.driver.theta(request)
    finally:
        _ = connection
