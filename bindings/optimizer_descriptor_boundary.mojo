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
    OPTIMIZER_NATIVE_ABI, OPT_NATIVE_COUNT, OPT_NATIVE_HOST, OPT_NATIVE_MODE, OptimizerNativeDriver,
)


struct OptimizerConnections(Defaultable, Movable):
    # GIL-owned boundary cache. Providers pin extension images, capsule tables
    # and factories; algorithm calls receive only the typed driver copy.
    var owners: List[PythonObject]
    var drivers: List[OptimizerNativeDriver]
    var modes: List[Int]

    def __init__(out self):
        self.owners = List[PythonObject]()
        self.drivers = List[OptimizerNativeDriver]()
        self.modes = List[Int]()


comptime OPTIMIZER_CONNECTIONS = SharedSlot[OptimizerConnections, "MojolearnOptimizerBoundaryConnectionsV1"]


def optimizer_provider(module: PythonObject) raises -> PythonObject:
    return neural_descriptor_connect_once(module, "optimizer_native_api_v1", String(OPTIMIZER_NATIVE_ABI), OPT_NATIVE_COUNT)


def optimizer_provider_entries(entries: PythonObject) raises -> PythonObject:
    var provider = neural_descriptor_connect_factory_once(entries["optimizer_native_api_v1"], String(OPTIMIZER_NATIVE_ABI), OPT_NATIVE_COUNT)
    if provider.downcast_value_ptr[NeuralDescriptorProvider]()[].module is not entries["module"]:
        raise Error("TypeError: optimizer descriptor must belong to its connected native extension")
    return provider


def _optimizer_connection(provider: PythonObject) raises -> Int:
    var cache = OPTIMIZER_CONNECTIONS.get_or_create_ptr()
    for i in range(len(cache[].owners)):  # driver-loop: over connected native extension images
        if cache[].owners[i] is provider:
            return i
    var entry = provider.downcast_value_ptr[NeuralDescriptorProvider]()[].entry
    var host = native_descriptor_call(entry, OPT_NATIVE_HOST, [], [], []) != 0
    var mode = native_descriptor_call(entry, OPT_NATIVE_MODE, [], [], [])
    var index = len(cache[].owners)
    cache[].owners.append(provider)
    cache[].drivers.append(OptimizerNativeDriver(entry, host))
    cache[].modes.append(mode)
    return index


def optimizer_driver(provider: PythonObject) raises -> OptimizerNativeDriver:
    var index = _optimizer_connection(provider)
    return OPTIMIZER_CONNECTIONS.get_or_create_ptr()[].drivers[index].copy()


def optimizer_native_mode(provider: PythonObject) raises -> Int:
    var index = _optimizer_connection(provider)
    return OPTIMIZER_CONNECTIONS.get_or_create_ptr()[].modes[index]


def optimizer_close_slot(module: PythonObject, handle: Int) raises:
    var provider = optimizer_provider(module)
    var driver = optimizer_driver(provider)
    if driver.host:
        _ = driver.close(handle)
    else:
        with GILReleased(Python()):
            _ = driver.close(handle)
    _ = provider
