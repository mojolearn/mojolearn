# SPDX-License-Identifier: Apache-2.0
"""Exact provider admission and image pins around native KMeans execution."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from core.shared_slot import SharedSlot
from bindings.cpython_table import cpython_table
from bindings.family_common import family_type_name
from bindings.host_inference_scalars import host_inference_scalar
from bindings.optimizer_public_words import stash_python_error
from core.kmeans_native_api import (
    KMEANS_NATIVE_ABI, KMEANS_NATIVE_COUNT, KMEANS_MODE, KMEANS_CAPABILITIES,
    KMeansNativeDriver, KMeansFitRequest, KMeansFitResult, KMeansQueryRequest, KMeansQueryResult,
)
from providers.python.neural_descriptor_pyapi import (
    NeuralDescriptorProvider, neural_descriptor_entry, neural_descriptor_connect_factory_once,
)


@fieldwise_init
struct KMeansConnection(Movable):
    var provider: PythonObject
    var driver: KMeansNativeDriver
    var mode: Int
    var capabilities: Int


struct KMeansConnections(Defaultable, Movable):
    var records: List[KMeansConnection]

    def __init__(out self):
        self.records = List[KMeansConnection]()


comptime KMEANS_CONNECTIONS = SharedSlot[KMeansConnections, "MojolearnKMeansConnectionsV1"]


def kmeans_integer(value: PythonObject, host: Bool, context: PythonObject) raises -> Int:
    # The former GPU entry used Int; the CPU entry used operator.index and
    # refused booleans. Admit exact native scalar storage before either C slot.
    host_inference_scalar(value, context)
    if not host:
        return Int(py=value)
    var name = family_type_name(value)
    if name == "bool" or name == "bool_":
        raise Error("core host: integers expected, not booleans")
    var raw = cpython_table()[].PyNumber_Index(value._obj_ptr)
    if not raw:
        var failure = stash_python_error()
        raise failure^
    var integer = PythonObject(from_owned=raw)
    return Int(py=integer)


def kmeans_connection(binding: PythonObject) raises -> Int:
    var factory = neural_descriptor_entry(binding, "kmeans_native_api_v1")
    var provider = neural_descriptor_connect_factory_once(factory, String(KMEANS_NATIVE_ABI), KMEANS_NATIVE_COUNT)
    var cache = KMEANS_CONNECTIONS.get_or_create_ptr()
    for index in range(len(cache[].records)):  # driver-loop: connected extension images
        if cache[].records[index].provider is provider:
            return index
    var driver = KMeansNativeDriver(provider.downcast_value_ptr[NeuralDescriptorProvider]()[].entry)
    var mode = driver.metadata(KMEANS_MODE)
    var capabilities = driver.metadata(KMEANS_CAPABILITIES)
    if mode != 0 and mode != 1:
        raise Error("KMeans native provider: unsupported compiled numeric mode")
    var index = len(cache[].records)
    cache[].records.append(KMeansConnection(provider, driver^, mode, capabilities))
    return index


def kmeans_capabilities(connection: Int) raises -> Int:
    return KMEANS_CONNECTIONS.get_or_create_ptr()[].records[connection].capabilities


def kmeans_run_fit(connection: Int, request: KMeansFitRequest) raises -> KMeansFitResult:
    var cache = KMEANS_CONNECTIONS.get_or_create_ptr()
    var pin = cache[].records[connection].provider
    var driver = cache[].records[connection].driver.copy()
    try:
        with GILReleased(Python()):
            return driver.fit(request)
    finally:
        _ = pin


def kmeans_run_query(connection: Int, request: KMeansQueryRequest) raises -> KMeansQueryResult:
    var cache = KMEANS_CONNECTIONS.get_or_create_ptr()
    var pin = cache[].records[connection].provider
    var driver = cache[].records[connection].driver.copy()
    try:
        with GILReleased(Python()):
            return driver.query(request)
    finally:
        _ = pin
