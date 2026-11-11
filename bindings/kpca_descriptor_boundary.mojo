# SPDX-License-Identifier: Apache-2.0
"""KernelPCA provider admission and image pins around typed native plans."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from core.shared_slot import SharedSlot
from core.native_descriptor_abi import NativeDescriptorCall
from core.neighbors_native_driver import XnDriver
from core.kpca_native_api import (
    KPCA_NATIVE_ABI, KPCA_NATIVE_COUNT, KPCA_MODE, KPCA_VENDOR, KPCA_CAPABILITIES,
    KPCA_CAP_EIGEN, KPCA_CAP_NEIGHBORS, KPCA_CAP_RESIDENT, KPCANativeDriver, KPCAEigenRequest,
)
from core.kpca_native_plan import (
    KPCANeighborDriver, KPCAAlphaRequest, KPCAQueryRequest, KPCACenteredQuery,
    KPCAFitPrepareRequest, KPCADenseFit,
)
from core.pca_native_api import pca_vendor_name
from x_neighbors.descriptor_schema import XN_ABI_NAME, XN_OPERATION_COUNT
from x_decomp.descriptor_schema import XD_ABI_NAME, XD_OPERATION_COUNT
from bindings.py_balanced import py_call, py_type
from bindings.native_class_admission import dict_keys_admitted, dict_keys_admit
from providers.python.neural_descriptor_pyapi import (
    NeuralDescriptorProvider, neural_descriptor_connect_factory_once, neural_descriptor_connect_once,
)


@fieldwise_init
struct KPCANearConnection(Movable):
    var provider: PythonObject
    var metadata: PythonObject
    var driver: KPCANeighborDriver
    var mode: Int
    var vendor: Int
    var capabilities: Int


@fieldwise_init
struct KPCADecompConnection(Movable):
    var provider: PythonObject
    var reduction_provider: PythonObject
    var driver: KPCANativeDriver
    var reduction: NativeDescriptorCall
    var mode: Int
    var vendor: Int
    var capabilities: Int


struct KPCAConnections(Defaultable, Movable):
    var near: List[KPCANearConnection]
    var decomp: List[KPCADecompConnection]

    def __init__(out self):
        self.near = List[KPCANearConnection]()
        self.decomp = List[KPCADecompConnection]()


comptime KPCA_CONNECTIONS = SharedSlot[KPCAConnections, "MojolearnKPCAConnectionsV1"]


def _factory(context: PythonObject, table_name: String, mode: String) raises -> PythonObject:
    var table = context[PythonObject(table_name)]
    if py_type(table) is not py_type(Python.dict()):
        raise Error("TypeError: KernelPCA binding tables require native dictionaries")
    if not dict_keys_admitted(table):
        var keys = py_call(py_type(Python.list()), table)
        for i in range(len(keys)):  # driver-loop: fixed import-connected mode keys
            if py_type(keys[i]) is not py_type(PythonObject("")):
                raise Error("TypeError: KernelPCA binding tables require native string keys")
        dict_keys_admit(table)
    var factory = table[PythonObject(mode)]
    if py_type(factory) is py_type(PythonObject("")):
        raise Error(String(factory))
    if factory is None:
        raise Error("NotImplementedError: KernelPCA requires the compiled native provider for " + table_name)
    return factory


def _check(mode: String, vendor: Int, native_mode: Int, capabilities: Int,
           required: Int, context: PythonObject) raises:
    var selected_vendor = context["vendor"]
    if py_type(selected_vendor) is not py_type(PythonObject("")):
        raise Error("TypeError: KernelPCA selected vendor requires a native string")
    if pca_vendor_name(vendor) != String(selected_vendor):
        raise Error("RuntimeError: KernelPCA native vendor differs from selected execution")
    if native_mode != (1 if mode == "identical" else 0):
        raise Error("RuntimeError: KernelPCA native mode differs from selected execution")
    if (capabilities & required) == 0:
        raise Error("NotImplementedError: KernelPCA native provider lacks the required operation")


def kpca_near_connection(mode: String, context: PythonObject) raises -> Int:
    var provider = neural_descriptor_connect_factory_once(_factory(context, "kpca_neighbors_bindings", mode),
        String(XN_ABI_NAME), XN_OPERATION_COUNT)
    var metadata = neural_descriptor_connect_factory_once(_factory(context, "kpca_info_bindings", mode),
        String(KPCA_NATIVE_ABI), KPCA_NATIVE_COUNT)
    if provider.downcast_value_ptr[NeuralDescriptorProvider]()[].module is not metadata.downcast_value_ptr[NeuralDescriptorProvider]()[].module:
        raise Error("RuntimeError: KernelPCA neighbor metadata and execution must belong to the same provider image")
    var cache = KPCA_CONNECTIONS.get_or_create_ptr()
    var index = -1
    for i in range(len(cache[].near)):  # driver-loop: admitted native provider images
        if cache[].near[i].provider is provider and cache[].near[i].metadata is metadata:
            index = i
            break
    if index < 0:
        var info = KPCANativeDriver(metadata.downcast_value_ptr[NeuralDescriptorProvider]()[].entry)
        var native_mode = info.metadata(KPCA_MODE)
        var vendor = info.metadata(KPCA_VENDOR)
        var capabilities = info.metadata(KPCA_CAPABILITIES)
        var driver = KPCANeighborDriver(XnDriver(provider.downcast_value_ptr[NeuralDescriptorProvider]()[].entry))
        index = len(cache[].near)
        cache[].near.append(KPCANearConnection(provider, metadata, driver^, native_mode, vendor, capabilities))
    _check(mode, cache[].near[index].vendor, cache[].near[index].mode,
        cache[].near[index].capabilities, KPCA_CAP_NEIGHBORS, context)
    return index


def kpca_decomp_connection(mode: String, context: PythonObject) raises -> Int:
    var provider = neural_descriptor_connect_factory_once(_factory(context, "kpca_decomp_bindings", mode),
        String(KPCA_NATIVE_ABI), KPCA_NATIVE_COUNT)
    var module = provider.downcast_value_ptr[NeuralDescriptorProvider]()[].module
    var reduction = neural_descriptor_connect_once(module, "x_decomp_native_api_v1", String(XD_ABI_NAME), XD_OPERATION_COUNT)
    if reduction.downcast_value_ptr[NeuralDescriptorProvider]()[].module is not module:
        raise Error("RuntimeError: KernelPCA eigensolver and reduction must belong to the same provider image")
    var cache = KPCA_CONNECTIONS.get_or_create_ptr()
    var index = -1
    for i in range(len(cache[].decomp)):  # driver-loop: admitted native provider images
        if cache[].decomp[i].provider is provider and cache[].decomp[i].reduction_provider is reduction:
            index = i
            break
    if index < 0:
        var driver = KPCANativeDriver(provider.downcast_value_ptr[NeuralDescriptorProvider]()[].entry)
        var native_mode = driver.metadata(KPCA_MODE)
        var vendor = driver.metadata(KPCA_VENDOR)
        var capabilities = driver.metadata(KPCA_CAPABILITIES)
        var entry = reduction.downcast_value_ptr[NeuralDescriptorProvider]()[].entry
        index = len(cache[].decomp)
        cache[].decomp.append(KPCADecompConnection(provider, reduction, driver^, entry, native_mode, vendor, capabilities))
    _check(mode, cache[].decomp[index].vendor, cache[].decomp[index].mode,
        cache[].decomp[index].capabilities, KPCA_CAP_EIGEN, context)
    return index


def kpca_resident(connection: Int) raises -> Bool:
    return (KPCA_CONNECTIONS.get_or_create_ptr()[].near[connection].capabilities & KPCA_CAP_RESIDENT) != 0


@fieldwise_init
struct KPCANearCall(Movable):
    var provider: PythonObject
    var metadata: PythonObject
    var driver: KPCANeighborDriver


@fieldwise_init
struct KPCADecompCall(Movable):
    var provider: PythonObject
    var reduction_provider: PythonObject
    var driver: KPCANativeDriver
    var reduction: NativeDescriptorCall


def _near_call(connection: Int) raises -> KPCANearCall:
    var cache = KPCA_CONNECTIONS.get_or_create_ptr()
    return KPCANearCall(cache[].near[connection].provider, cache[].near[connection].metadata,
        cache[].near[connection].driver.copy())


def _decomp_call(connection: Int) raises -> KPCADecompCall:
    var cache = KPCA_CONNECTIONS.get_or_create_ptr()
    return KPCADecompCall(cache[].decomp[connection].provider, cache[].decomp[connection].reduction_provider,
        cache[].decomp[connection].driver.copy(), cache[].decomp[connection].reduction)


def kpca_run_prepare(near: Int, decomp: Int, request: KPCAFitPrepareRequest) raises -> KPCADenseFit:
    var n = _near_call(near)
    var d = _decomp_call(decomp)
    try:
        with GILReleased(Python()):
            return KPCADenseFit(n.driver, d.reduction, request)
    finally:
        _ = n
        _ = d


def kpca_run_eigen(connection: Int, request: KPCAEigenRequest) raises -> Int:
    var call = _decomp_call(connection)
    try:
        with GILReleased(Python()):
            return call.driver.eigen(request)
    finally:
        _ = call


def kpca_run_alpha(connection: Int, request: KPCAAlphaRequest) raises:
    var call = _near_call(connection)
    try:
        with GILReleased(Python()):
            call.driver.alpha(request)
    finally:
        _ = call


def kpca_run_fused(connection: Int, request: KPCAQueryRequest) raises:
    var call = _near_call(connection)
    try:
        with GILReleased(Python()):
            call.driver.fused_query(request)
    finally:
        _ = call


def kpca_run_centered(connection: Int, request: KPCAQueryRequest) raises -> KPCACenteredQuery:
    var call = _near_call(connection)
    try:
        with GILReleased(Python()):
            return KPCACenteredQuery(call.driver, request)
    finally:
        _ = call


def kpca_run_finish(connection: Int, plan: KPCACenteredQuery, alphas: Int, dst: Int, count: Int) raises:
    var call = _near_call(connection)
    try:
        with GILReleased(Python()):
            plan.finish(call.driver, alphas, dst, count)
    finally:
        _ = call
        _ = plan
