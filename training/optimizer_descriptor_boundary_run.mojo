# SPDX-License-Identifier: Apache-2.0
"""Typed optimizer connection records (RUN): the driver, compiled numeric mode
and device availability of each connected provider image, keyed by the
provider's address word. The door
(`bindings/optimizer_descriptor_boundary_door.mojo`) admits the provider and
keeps its pin at the same index."""
from core.shared_slot import SharedSlot
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_call import native_descriptor_call
from training.optimizer_native_api import (
    OPT_NATIVE_HOST, OPT_NATIVE_MODE, OPT_NATIVE_DEVICE_AVAILABLE, OptimizerNativeDriver,
)


@fieldwise_init
struct OptimizerConnection(Movable):
    var provider_word: Int
    var driver: OptimizerNativeDriver
    var mode: Int
    var device: Bool


struct OptimizerConnections(Defaultable, Movable):
    var records: List[OptimizerConnection]

    def __init__(out self):
        self.records = List[OptimizerConnection]()


# V3: the record names its provider by address word; the pin moved to the door.
comptime OPTIMIZER_CONNECTIONS = SharedSlot[OptimizerConnections, "MojolearnOptimizerBoundaryConnectionsV3", 3]


def optimizer_connection_index(provider_word: Int) raises -> Int:
    var cache = OPTIMIZER_CONNECTIONS.get_or_create_ptr()
    for i in range(len(cache[].records)):  # driver-loop: over connected native extension images
        if cache[].records[i].provider_word == provider_word:
            return i
    return -1


def optimizer_connection_add(provider_word: Int, entry: NativeDescriptorCall) raises -> Int:
    var host = native_descriptor_call(entry, OPT_NATIVE_HOST, [], [], []) != 0
    var mode = native_descriptor_call(entry, OPT_NATIVE_MODE, [], [], [])
    var device = native_descriptor_call(entry, OPT_NATIVE_DEVICE_AVAILABLE, [], [], []) != 0
    var cache = OPTIMIZER_CONNECTIONS.get_or_create_ptr()
    var index = len(cache[].records)
    cache[].records.append(OptimizerConnection(provider_word, OptimizerNativeDriver(entry, host), mode, device))
    return index


def optimizer_record_driver(connection: Int) raises -> OptimizerNativeDriver:
    return OPTIMIZER_CONNECTIONS.get_or_create_ptr()[].records[connection].driver.copy()


def optimizer_record_mode(connection: Int) raises -> Int:
    return OPTIMIZER_CONNECTIONS.get_or_create_ptr()[].records[connection].mode


def optimizer_record_device(connection: Int) raises -> Bool:
    return OPTIMIZER_CONNECTIONS.get_or_create_ptr()[].records[connection].device
