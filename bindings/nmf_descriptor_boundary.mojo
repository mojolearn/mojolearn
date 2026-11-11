# SPDX-License-Identifier: Apache-2.0
"""NMF import-connected native descriptors and compiled-image lifetime pins."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from core.shared_slot import SharedSlot
from core.pca_native_api import pca_vendor_name
from core.nmf_native_api import (
    NMF_NATIVE_ABI, NMF_NATIVE_COUNT, NMF_MODE, NMF_VENDOR, NMF_CAPABILITIES,
    NMFNativeDriver, NMFFitRequest, NMFFitResult, NMFInverseRequest,
)
from bindings.py_balanced import py_call, py_type
from bindings.decomp_owner_common import decomp_attr
from bindings.native_class_admission import dict_keys_admitted, dict_keys_admit
from providers.python.neural_descriptor_pyapi import NeuralDescriptorProvider, neural_descriptor_connect_factory_once


@fieldwise_init
struct NMFConnection(Movable):
    var provider: PythonObject
    var driver: NMFNativeDriver
    var mode: Int
    var vendor: Int
    var capabilities: Int


struct NMFConnections(Defaultable, Movable):
    var records: List[NMFConnection]

    def __init__(out self):
        self.records = List[NMFConnection]()


comptime NMF_CONNECTIONS = SharedSlot[NMFConnections, "MojolearnNMFConnectionsV1"]


def nmf_connection(model: PythonObject, fields: PythonObject, context: PythonObject,
                   capability: Int, legacy_name: String) raises -> Int:
    # Keep the former decomp_entries mode/availability refusal at its stage.
    var mode = decomp_attr(model, fields, "numeric_mode_")
    if py_type(mode) is not py_type(PythonObject("")):
        raise Error("numeric_mode must be fast, identical or None")
    var name = String(py_call(py_call(mode.strip).lower))
    if name != "identical" and name != "fast":
        raise Error("numeric_mode must be fast, identical or None")
    var table = context["nmf_bindings"]
    if py_type(table) is not py_type(Python.dict()):
        raise Error("TypeError: NMF binding tables require native dictionaries")
    if not dict_keys_admitted(table):
        var keys = py_call(py_type(Python.list()), table)
        for index in range(len(keys)):  # driver-loop: fixed connected mode names before keyed lookup
            if py_type(keys[index]) is not py_type(PythonObject("")):
                raise Error("TypeError: NMF binding tables require native string keys")
        dict_keys_admit(table)
    var factory = table[PythonObject(name)]
    if py_type(factory) is py_type(PythonObject("")):
        raise Error(String(factory))
    if factory is None:
        raise Error("NotImplementedError: mojolearn " + String(py_type(model).__name__) + ": this build of "
                    + "_mojolearn_x_decomp does not export " + legacy_name
                    + "; rebuild it with bindings/build_x_decomp.sh")
    var provider = neural_descriptor_connect_factory_once(factory, String(NMF_NATIVE_ABI), NMF_NATIVE_COUNT)
    var cache = NMF_CONNECTIONS.get_or_create_ptr()
    var index = -1
    for candidate in range(len(cache[].records)):  # driver-loop: admitted native provider images
        if cache[].records[candidate].provider is provider:
            index = candidate
            break
    if index < 0:
        var driver = NMFNativeDriver(provider.downcast_value_ptr[NeuralDescriptorProvider]()[].entry)
        var native_mode = driver.metadata(NMF_MODE)
        var vendor = driver.metadata(NMF_VENDOR)
        var capabilities = driver.metadata(NMF_CAPABILITIES)
        index = len(cache[].records)
        cache[].records.append(NMFConnection(provider, driver^, native_mode, vendor, capabilities))
    if pca_vendor_name(cache[].records[index].vendor) != String(context["vendor"]):
        raise Error("RuntimeError: NMF native vendor differs from selected execution")
    if cache[].records[index].mode != (1 if name == "identical" else 0):
        raise Error("RuntimeError: NMF native mode differs from selected execution")
    if (cache[].records[index].capabilities & capability) == 0:
        raise Error("NotImplementedError: mojolearn " + String(py_type(model).__name__) + ": this build of "
                    + "_mojolearn_x_decomp does not export " + legacy_name
                    + "; rebuild it with bindings/build_x_decomp.sh")
    return index


def nmf_run_fit(connection: Int, request: NMFFitRequest) raises -> NMFFitResult:
    var cache = NMF_CONNECTIONS.get_or_create_ptr()
    var pin = cache[].records[connection].provider
    var driver = cache[].records[connection].driver.copy()
    try:
        with GILReleased(Python()):
            return driver.fit(request)
    finally:
        _ = pin


def nmf_run_inverse(connection: Int, request: NMFInverseRequest) raises:
    var cache = NMF_CONNECTIONS.get_or_create_ptr()
    var pin = cache[].records[connection].provider
    var driver = cache[].records[connection].driver.copy()
    try:
        with GILReleased(Python()):
            driver.inverse(request)
    finally:
        _ = pin
