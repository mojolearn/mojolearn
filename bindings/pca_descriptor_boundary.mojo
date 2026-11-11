# SPDX-License-Identifier: Apache-2.0
"""PCA provider admission and image pins surrounding typed native execution."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from core.shared_slot import SharedSlot
from core.pca_native_api import (
    PCA_NATIVE_ABI, PCA_NATIVE_COUNT, PCA_MODE, PCA_VENDOR, PCA_CAPABILITIES, PCA_CAP_WHITEN,
    PCANativeDriver, PCAFitRequest, PCAFitResult, PCAQueryRequest, PCAQueryResult, pca_vendor_name,
)
from bindings.py_balanced import py_call, py_get, py_type
from bindings.native_class_admission import dict_keys_admitted, dict_keys_admit
from providers.python.neural_descriptor_pyapi import (
    NeuralDescriptorProvider, neural_descriptor_connect_factory_once,
)


@fieldwise_init
struct PCAConnection(Movable):
    var provider: PythonObject
    var driver: PCANativeDriver
    var mode: Int
    var vendor: Int
    var capabilities: Int


struct PCAConnections(Defaultable, Movable):
    var records: List[PCAConnection]

    def __init__(out self):
        self.records = List[PCAConnection]()


comptime PCA_CONNECTIONS = SharedSlot[PCAConnections, "MojolearnPCAConnectionsV1"]


def pca_mode_name(value: PythonObject) raises -> String:
    if py_type(value) is not py_type(PythonObject("")):
        raise Error("TypeError: numeric_mode must be identical, fast or None")
    var mode = String(py_call(py_call(value.strip).lower))
    if mode != "identical" and mode != "fast":
        raise Error("numeric_mode must be identical, fast or None")
    return mode


def _pca_connect(factory: PythonObject, context: PythonObject, mode: Int) raises -> Int:
    var provider = neural_descriptor_connect_factory_once(factory, String(PCA_NATIVE_ABI), PCA_NATIVE_COUNT)
    var cache = PCA_CONNECTIONS.get_or_create_ptr()
    var index = -1
    for candidate in range(len(cache[].records)):  # driver-loop: connected compiled provider images
        if cache[].records[candidate].provider is provider:
            index = candidate
            break
    if index < 0:
        var driver = PCANativeDriver(provider.downcast_value_ptr[NeuralDescriptorProvider]()[].entry)
        var native_mode = driver.metadata(PCA_MODE)
        var vendor = driver.metadata(PCA_VENDOR)
        var capabilities = driver.metadata(PCA_CAPABILITIES)
        index = len(cache[].records)
        cache[].records.append(PCAConnection(provider, driver^, native_mode, vendor, capabilities))
    if pca_vendor_name(cache[].records[index].vendor) != String(context["vendor"]):
        raise Error("RuntimeError: PCA native vendor differs from selected execution")
    if cache[].records[index].mode != mode:
        raise Error("RuntimeError: PCA native mode differs from selected execution")
    return index


def pca_connection(mode: String, context: PythonObject) raises -> Int:
    var bindings = context["pca_bindings"]
    if py_type(bindings) is not py_type(Python.dict()):
        raise Error("TypeError: PCA binding tables require native dictionaries")
    if not dict_keys_admitted(bindings):
        var keys = py_call(py_type(Python.list()), bindings)
        for index in range(len(keys)):  # driver-loop: fixed connected mode keys, before lookup
            if py_type(keys[index]) is not py_type(PythonObject("")):
                raise Error("TypeError: PCA binding tables require native string keys")
        dict_keys_admit(bindings)
    var selected = bindings[PythonObject(mode)]
    if py_type(selected) is py_type(PythonObject("")):
        raise Error("ImportError: " + String(selected))
    return _pca_connect(selected, context, 1 if mode == "identical" else 0)


def pca_connection_matches(connection: Int, mode: String, context: PythonObject) raises -> Bool:
    # Cache validation compares identity only. A changed factory must take
    # full field admission before its provider refusal, at the original stage.
    # In particular, do not invoke a newly rebound factory on the cache path.
    var bindings = context["pca_bindings"]
    if py_type(bindings) is not py_type(Python.dict()):
        return False
    if not dict_keys_admitted(bindings):
        var keys = py_call(py_type(Python.list()), bindings)
        for index in range(len(keys)):  # driver-loop: connected mode keys before identity lookup
            if py_type(keys[index]) is not py_type(PythonObject("")):
                return False
        dict_keys_admit(bindings)
    var provider = PCA_CONNECTIONS.get_or_create_ptr()[].records[connection].provider
    var factory = provider.downcast_value_ptr[NeuralDescriptorProvider]()[].factory
    return py_get(bindings, PythonObject(mode)) is factory


def pca_fold_connection(context: PythonObject) raises -> Int:
    var factory = py_get(context, "pca_fold_factory")
    if factory is None:
        return -1
    return _pca_connect(factory, context, 1)


def pca_capabilities(connection: Int) raises -> Int:
    return PCA_CONNECTIONS.get_or_create_ptr()[].records[connection].capabilities


def pca_mode(connection: Int) raises -> Int:
    return PCA_CONNECTIONS.get_or_create_ptr()[].records[connection].mode


def pca_vendor(connection: Int) raises -> String:
    return pca_vendor_name(PCA_CONNECTIONS.get_or_create_ptr()[].records[connection].vendor)


def pca_whiten_capability(connection: Int) raises:
    if (pca_capabilities(connection) & PCA_CAP_WHITEN) == 0:
        raise Error("NotImplementedError: PCA whiten=True requires the compiled whitened transform pair")


def pca_run_fit(connection: Int, request: PCAFitRequest) raises -> PCAFitResult:
    var cache = PCA_CONNECTIONS.get_or_create_ptr()
    var pin = cache[].records[connection].provider
    var driver = cache[].records[connection].driver.copy()
    try:
        with GILReleased(Python()):
            return driver.fit(request)
    finally:
        _ = pin


def pca_run_query(connection: Int, request: PCAQueryRequest) raises -> PCAQueryResult:
    var cache = PCA_CONNECTIONS.get_or_create_ptr()
    var pin = cache[].records[connection].provider
    var driver = cache[].records[connection].driver.copy()
    try:
        with GILReleased(Python()):
            return driver.query(request)
    finally:
        _ = pin
