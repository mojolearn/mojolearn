# SPDX-License-Identifier: Apache-2.0
"""IPCA import-connected native descriptors and compiled-image lifetime pins."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from core.shared_slot import SharedSlot
from core.pca_native_api import pca_vendor_name
from core.ipca_native_api import (
    IPCA_NATIVE_ABI, IPCA_NATIVE_COUNT, IPCA_MODE, IPCA_VENDOR, IPCA_CAPABILITIES,
    IPCANativeDriver, IPCAFitRequest, IPCAFitResult, IPCAPartialRequest, IPCAQueryRequest,
    IPCA_CAP_FIT, IPCA_CAP_PARTIAL, IPCA_CAP_QUERY, IPCA_CAP_RESIDENT,
)
from bindings.py_balanced import py_call, py_type
from bindings.decomp_owner_common import decomp_attr
from bindings.native_class_admission import dict_keys_admitted, dict_keys_admit
from providers.python.neural_descriptor_pyapi import NeuralDescriptorProvider, neural_descriptor_connect_factory_once


@fieldwise_init
struct IPCAConnection(Movable):
    var provider: PythonObject
    var driver: IPCANativeDriver
    var mode: Int
    var vendor: Int
    var capabilities: Int


struct IPCAConnections(Defaultable, Movable):
    var records: List[IPCAConnection]

    def __init__(out self):
        self.records = List[IPCAConnection]()


comptime IPCA_CONNECTIONS = SharedSlot[IPCAConnections, "MojolearnIPCAConnectionsV1"]


def ipca_connection(model: PythonObject, fields: PythonObject, context: PythonObject) raises -> Int:
    # Keep the former decomp_entries mode/availability refusal at its stage.
    var mode = decomp_attr(model, fields, "numeric_mode_")
    if py_type(mode) is not py_type(PythonObject("")):
        raise Error("numeric_mode must be fast, identical or None")
    var name = String(mode)
    if name != "identical" and name != "fast":
        name = String(py_call(py_call(mode.strip).lower))
    if name != "identical" and name != "fast":
        raise Error("numeric_mode must be fast, identical or None")
    var table = context["ipca_bindings"]
    if py_type(table) is not py_type(Python.dict()):
        raise Error("TypeError: IPCA binding tables require native dictionaries")
    if not dict_keys_admitted(table):
        var keys = py_call(py_type(Python.list()), table)
        for index in range(len(keys)):  # driver-loop: fixed connected mode names before keyed lookup
            if py_type(keys[index]) is not py_type(PythonObject("")):
                raise Error("TypeError: IPCA binding tables require native string keys")
        dict_keys_admit(table)
    var factory = table[PythonObject(name)]
    if py_type(factory) is py_type(PythonObject("")):
        raise Error(String(factory))
    if factory is None:
        raise Error("NotImplementedError: mojolearn IncrementalPCA: this build of "
                    + "_mojolearn_x_decomp does not export the x_decomp_ipca_* drivers"
                    + "; rebuild it with bindings/build_x_decomp.sh")
    var provider = neural_descriptor_connect_factory_once(factory, String(IPCA_NATIVE_ABI), IPCA_NATIVE_COUNT)
    var cache = IPCA_CONNECTIONS.get_or_create_ptr()
    var index = -1
    for candidate in range(len(cache[].records)):  # driver-loop: admitted native provider images
        if cache[].records[candidate].provider is provider:
            index = candidate
            break
    if index < 0:
        var driver = IPCANativeDriver(provider.downcast_value_ptr[NeuralDescriptorProvider]()[].entry)
        var native_mode = driver.metadata(IPCA_MODE)
        var vendor = driver.metadata(IPCA_VENDOR)
        var capabilities = driver.metadata(IPCA_CAPABILITIES)
        index = len(cache[].records)
        cache[].records.append(IPCAConnection(provider, driver^, native_mode, vendor, capabilities))
    if pca_vendor_name(cache[].records[index].vendor) != String(context["vendor"]):
        raise Error("RuntimeError: IPCA native vendor differs from selected execution")
    if cache[].records[index].mode != (1 if name == "identical" else 0):
        raise Error("RuntimeError: IPCA native mode differs from selected execution")
    if (cache[].records[index].capabilities & (IPCA_CAP_FIT | IPCA_CAP_PARTIAL | IPCA_CAP_QUERY)) != (IPCA_CAP_FIT | IPCA_CAP_PARTIAL | IPCA_CAP_QUERY):
        raise Error("NotImplementedError: mojolearn IncrementalPCA: this build of "
                    + "_mojolearn_x_decomp does not export the x_decomp_ipca_* drivers"
                    + "; rebuild it with bindings/build_x_decomp.sh")
    return index


def ipca_run_fit(connection: Int, request: IPCAFitRequest) raises -> IPCAFitResult:
    var cache = IPCA_CONNECTIONS.get_or_create_ptr()
    var pin = cache[].records[connection].provider
    var driver = cache[].records[connection].driver.copy()
    try:
        with GILReleased(Python()):
            return driver.fit(request)
    finally:
        _ = pin



def ipca_run_partial(connection: Int, request: IPCAPartialRequest) raises -> IPCAFitResult:
    var pin = ipca_provider(connection)
    var driver = ipca_driver(connection)
    try:
        with GILReleased(Python()):
            return driver.partial(request)
    finally:
        _ = pin


def ipca_run_query(connection: Int, request: IPCAQueryRequest) raises:
    var pin = ipca_provider(connection)
    var driver = ipca_driver(connection)
    try:
        with GILReleased(Python()):
            driver.query(request)
    finally:
        _ = pin


def ipca_provider(connection: Int) raises -> PythonObject:
    return IPCA_CONNECTIONS.get_or_create_ptr()[].records[connection].provider


def ipca_driver(connection: Int) raises -> IPCANativeDriver:
    return IPCA_CONNECTIONS.get_or_create_ptr()[].records[connection].driver.copy()


def ipca_resident(connection: Int) raises -> Bool:
    return (IPCA_CONNECTIONS.get_or_create_ptr()[].records[connection].capabilities & IPCA_CAP_RESIDENT) != 0


def ipca_run_free(connection: Int, handle: Int) raises:
    var pin = ipca_provider(connection)
    var driver = ipca_driver(connection)
    try:
        with GILReleased(Python()):
            driver.free(handle)
    finally:
        _ = pin
