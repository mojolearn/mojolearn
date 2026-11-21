# SPDX-License-Identifier: Apache-2.0
"""KernelPCA provider admission and image pins around typed native plans (door).

The typed connection records are in `x_decomp/kpca_descriptor_boundary_run.mojo`."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from core.shared_slot import SharedSlot
from core.native_descriptor_abi import NativeDescriptorCall
from core.neighbors_native_driver import XnDriver
from core.kpca_native_api import (
    KPCA_NATIVE_ABI, KPCA_NATIVE_COUNT, KPCA_CAP_EIGEN, KPCA_CAP_NEIGHBORS, KPCANativeDriver, KPCAEigenRequest,
)
from x_decomp.kpca_descriptor_boundary_run import (
    kpca_near_index, kpca_near_add, kpca_near_facts, kpca_near_driver,
    kpca_decomp_index, kpca_decomp_add, kpca_decomp_facts, kpca_decomp_driver, kpca_decomp_reduction,
)
from bindings.py_provider_pins import provider_word
from core.kpca_native_plan import (
    KPCANeighborDriver, KPCAAlphaRequest, KPCAQueryRequest, KPCACenteredQuery,
    KPCAFitPrepareRequest, KPCADenseFit,
)
from core.pca_native_api import pca_vendor_name
from x_neighbors.descriptor_schema import XN_ABI_NAME, XN_OPERATION_COUNT
from x_decomp.descriptor_schema import XD_ABI_NAME, XD_OPERATION_COUNT
from bindings.py_balanced import py_call, py_type, py_get
from bindings.native_class_admission_door import dict_keys_admitted, dict_keys_admit
from providers.python.neural_descriptor_pyapi_door import (
    NeuralDescriptorProvider, neural_descriptor_connect_factory_once, neural_descriptor_connect_once,
)


struct KPCAProviderPins(Defaultable, Movable):
    # Door-side lifetime pins, at the index of their typed RUN record.
    var near_providers: List[PythonObject]
    var near_metadata: List[PythonObject]
    var decomp_providers: List[PythonObject]
    var decomp_reductions: List[PythonObject]

    def __init__(out self):
        self.near_providers = List[PythonObject]()
        self.near_metadata = List[PythonObject]()
        self.decomp_providers = List[PythonObject]()
        self.decomp_reductions = List[PythonObject]()


comptime KPCA_PROVIDER_PINS = SharedSlot[KPCAProviderPins, "MojolearnKPCAProviderPinsV1"]


def _factory(context: PythonObject, table_name: String, mode: String) raises -> PythonObject:
    # `table_name` is the device table's, or the CPU image's table of a
    # `device="cpu"` route beside a device set (`*_cpu_bindings`, connected
    # at import by python/mojolearn/_native_decomp_bridge.py).
    var table = py_get(context, PythonObject(table_name))
    if table is None:
        raise Error("ImportError: device='cpu' is not connected in this installation")
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
           required: Int, selected_vendor: String) raises:
    # `selected_vendor`: the vendor word the estimator's route expects (`cpu`
    # for a CPU route), read once at resolution, never per call.
    if pca_vendor_name(vendor) != selected_vendor:
        raise Error("RuntimeError: KernelPCA native vendor differs from selected execution")
    if native_mode != (1 if mode == "identical" else 0):
        raise Error("RuntimeError: KernelPCA native mode differs from selected execution")
    if (capabilities & required) == 0:
        raise Error("NotImplementedError: KernelPCA native provider lacks the required operation")


def kpca_near_connection(mode: String, cpu_image: Bool, vendor: String, context: PythonObject) raises -> Int:
    """Resolution only (bindings/kpca_native_door.mojo `_kpca_route`): the
    neighbors image pair of the device set, or of the CPU image."""
    var provider = neural_descriptor_connect_factory_once(_factory(context,
        String("kpca_neighbors_cpu_bindings") if cpu_image else String("kpca_neighbors_bindings"), mode),
        String(XN_ABI_NAME), XN_OPERATION_COUNT)
    var metadata = neural_descriptor_connect_factory_once(_factory(context,
        String("kpca_info_cpu_bindings") if cpu_image else String("kpca_info_bindings"), mode),
        String(KPCA_NATIVE_ABI), KPCA_NATIVE_COUNT)
    if provider.downcast_value_ptr[NeuralDescriptorProvider]()[].module is not metadata.downcast_value_ptr[NeuralDescriptorProvider]()[].module:
        raise Error("RuntimeError: KernelPCA neighbor metadata and execution must belong to the same provider image")
    var index = kpca_near_index(provider_word(provider), provider_word(metadata))
    if index < 0:
        # The typed record and the door pins are appended together, under the GIL.
        index = kpca_near_add(provider_word(provider), provider_word(metadata),
            KPCANativeDriver(metadata.downcast_value_ptr[NeuralDescriptorProvider]()[].record[0].entry),
            KPCANeighborDriver(XnDriver(provider.downcast_value_ptr[NeuralDescriptorProvider]()[].record[0].entry)))
        var pins = KPCA_PROVIDER_PINS.get_or_create_ptr()
        pins[].near_providers.append(provider)
        pins[].near_metadata.append(metadata)
    var facts = kpca_near_facts(index)
    _check(mode, facts.vendor, facts.mode, facts.capabilities, KPCA_CAP_NEIGHBORS, vendor)
    return index


def kpca_decomp_connection(mode: String, cpu_image: Bool, vendor: String, context: PythonObject) raises -> Int:
    """Resolution only: the eigensolver image of the device set, or the CPU image."""
    var provider = neural_descriptor_connect_factory_once(_factory(context,
        String("kpca_decomp_cpu_bindings") if cpu_image else String("kpca_decomp_bindings"), mode),
        String(KPCA_NATIVE_ABI), KPCA_NATIVE_COUNT)
    var module = provider.downcast_value_ptr[NeuralDescriptorProvider]()[].module
    var reduction = neural_descriptor_connect_once(module, "x_decomp_native_api_v1", String(XD_ABI_NAME), XD_OPERATION_COUNT)
    if reduction.downcast_value_ptr[NeuralDescriptorProvider]()[].module is not module:
        raise Error("RuntimeError: KernelPCA eigensolver and reduction must belong to the same provider image")
    var index = kpca_decomp_index(provider_word(provider), provider_word(reduction))
    if index < 0:
        # The typed record and the door pins are appended together, under the GIL.
        index = kpca_decomp_add(provider_word(provider), provider_word(reduction),
            KPCANativeDriver(provider.downcast_value_ptr[NeuralDescriptorProvider]()[].record[0].entry),
            reduction.downcast_value_ptr[NeuralDescriptorProvider]()[].record[0].entry)
        var pins = KPCA_PROVIDER_PINS.get_or_create_ptr()
        pins[].decomp_providers.append(provider)
        pins[].decomp_reductions.append(reduction)
    var facts = kpca_decomp_facts(index)
    _check(mode, facts.vendor, facts.mode, facts.capabilities, KPCA_CAP_EIGEN, vendor)
    return index


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
    var pins = KPCA_PROVIDER_PINS.get_or_create_ptr()
    return KPCANearCall(pins[].near_providers[connection], pins[].near_metadata[connection],
        kpca_near_driver(connection))


def _decomp_call(connection: Int) raises -> KPCADecompCall:
    var pins = KPCA_PROVIDER_PINS.get_or_create_ptr()
    return KPCADecompCall(pins[].decomp_providers[connection], pins[].decomp_reductions[connection],
        kpca_decomp_driver(connection), kpca_decomp_reduction(connection))


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
