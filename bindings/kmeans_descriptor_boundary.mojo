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
    KMEANS_NATIVE_ABI, KMEANS_NATIVE_COUNT,
    KMeansNativeDriver, KMeansFitRequest, KMeansFitResult, KMeansQueryRequest, KMeansQueryResult,
)
from x_cluster.kmeans_descriptor_boundary_run import kmeans_connection_index, kmeans_connection_add, kmeans_driver
from bindings.py_provider_pins import ProviderPins, provider_word
from providers.python.neural_descriptor_pyapi import (
    NeuralDescriptorProvider, neural_descriptor_entry, neural_descriptor_connect_factory_once,
)


comptime KMEANS_PROVIDER_PINS = SharedSlot[ProviderPins, "MojolearnKMeansProviderPinsV1"]


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
    var index = kmeans_connection_index(provider_word(provider))
    if index >= 0:
        return index
    # The typed record and the door pin are appended together, under the GIL.
    index = kmeans_connection_add(provider_word(provider),
        KMeansNativeDriver(provider.downcast_value_ptr[NeuralDescriptorProvider]()[].entry))
    KMEANS_PROVIDER_PINS.get_or_create_ptr()[].providers.append(provider)
    return index


def kmeans_provider(connection: Int) raises -> PythonObject:
    return KMEANS_PROVIDER_PINS.get_or_create_ptr()[].providers[connection]


def kmeans_run_fit(connection: Int, request: KMeansFitRequest) raises -> KMeansFitResult:
    var pin = kmeans_provider(connection)
    var driver = kmeans_driver(connection)
    try:
        with GILReleased(Python()):
            return driver.fit(request)
    finally:
        _ = pin


def kmeans_run_query(connection: Int, request: KMeansQueryRequest) raises -> KMeansQueryResult:
    var pin = kmeans_provider(connection)
    var driver = kmeans_driver(connection)
    try:
        with GILReleased(Python()):
            return driver.query(request)
    finally:
        _ = pin
