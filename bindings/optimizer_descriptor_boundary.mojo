# SPDX-License-Identifier: Apache-2.0
"""Optimizer descriptor UNPACK: exact factories and lifetime pins stay here."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from core.native_descriptor_call import native_descriptor_call
from core.shared_slot import SharedSlot
from providers.python.neural_descriptor_pyapi import (
    NeuralDescriptorProvider, neural_descriptor_connect_once, neural_descriptor_connect_factory_once,
)
from training.optimizer_native_api import (
    OPTIMIZER_NATIVE_ABI, OPT_NATIVE_COUNT, OPT_NATIVE_HOST, OPT_NATIVE_MODE, OPT_NATIVE_DEVICE_AVAILABLE, OptimizerNativeDriver,
)


@fieldwise_init
struct OptimizerConnection(Movable):
    var owner: PythonObject
    var driver: OptimizerNativeDriver
    var mode: Int
    var device: Bool


struct OptimizerConnections(Defaultable, Movable):
    # GIL-owned boundary cache. Providers pin extension images, capsule tables
    # and factories; algorithm calls receive only the typed driver copy.
    var records: List[OptimizerConnection]

    def __init__(out self):
        self.records = List[OptimizerConnection]()


comptime OPTIMIZER_CONNECTIONS = SharedSlot[OptimizerConnections, "MojolearnOptimizerBoundaryConnectionsV2"]


def optimizer_provider(module: PythonObject) raises -> PythonObject:
    return neural_descriptor_connect_once(module, "optimizer_native_api_v1", String(OPTIMIZER_NATIVE_ABI), OPT_NATIVE_COUNT)


def optimizer_provider_entries(entries: PythonObject) raises -> PythonObject:
    var provider = neural_descriptor_connect_factory_once(entries["optimizer_native_api_v1"], String(OPTIMIZER_NATIVE_ABI), OPT_NATIVE_COUNT)
    if provider.downcast_value_ptr[NeuralDescriptorProvider]()[].module is not entries["module"]:
        raise Error("TypeError: optimizer descriptor must belong to its connected native extension")
    return provider


def _optimizer_connection(provider: PythonObject) raises -> Int:
    var cache = OPTIMIZER_CONNECTIONS.get_or_create_ptr()
    for i in range(len(cache[].records)):  # driver-loop: over connected native extension images
        if cache[].records[i].owner is provider:
            return i
    var entry = provider.downcast_value_ptr[NeuralDescriptorProvider]()[].entry
    var host = native_descriptor_call(entry, OPT_NATIVE_HOST, [], [], []) != 0
    var mode = native_descriptor_call(entry, OPT_NATIVE_MODE, [], [], [])
    var device = native_descriptor_call(entry, OPT_NATIVE_DEVICE_AVAILABLE, [], [], []) != 0
    var index = len(cache[].records)
    # Publish one complete native record. Parallel owner/metadata vectors
    # could otherwise have different lengths after an allocation failure.
    cache[].records.append(OptimizerConnection(provider, OptimizerNativeDriver(entry, host), mode, device))
    return index


def optimizer_driver(provider: PythonObject) raises -> OptimizerNativeDriver:
    var index = _optimizer_connection(provider)
    return OPTIMIZER_CONNECTIONS.get_or_create_ptr()[].records[index].driver.copy()


def optimizer_native_mode(provider: PythonObject) raises -> Int:
    var index = _optimizer_connection(provider)
    return OPTIMIZER_CONNECTIONS.get_or_create_ptr()[].records[index].mode


def optimizer_device_available(provider: PythonObject) raises -> Bool:
    var index = _optimizer_connection(provider)
    return OPTIMIZER_CONNECTIONS.get_or_create_ptr()[].records[index].device


def optimizer_close_slot(module: PythonObject, handle: Int) raises:
    var provider = optimizer_provider(module)
    var driver = optimizer_driver(provider)
    if driver.host:
        _ = driver.close(handle)
    else:
        with GILReleased(Python()):
            _ = driver.close(handle)
    _ = provider
