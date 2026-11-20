# SPDX-License-Identifier: Apache-2.0
"""Typed logistic connection records (RUN): the driver, compiled numeric mode,
host flag and multiclass availability of each connected provider image, keyed
by the provider's address word. The door
(`bindings/logistic_descriptor_boundary_door.mojo`) admits the provider and
keeps its pin at the same index."""
from core.shared_slot import SharedSlot
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_call import native_descriptor_call
from core.logistic_native_api import QN_NATIVE_MODE, QN_NATIVE_HOST, QN_NATIVE_MULTICLASS_AVAILABLE, QnNativeDriver


@fieldwise_init
struct LogisticConnection(Movable):
    var provider_word: Int
    var driver: QnNativeDriver
    var mode: Int
    var host: Bool
    var multiclass: Bool


struct LogisticConnections(Defaultable, Movable):
    var records: List[LogisticConnection]

    def __init__(out self):
        self.records = List[LogisticConnection]()


# V2: the record names its provider by address word; the pin moved to the door.
comptime LOGISTIC_CONNECTIONS = SharedSlot[LogisticConnections, "MojolearnLogisticBoundaryConnectionsV2", 2]


def logistic_connection_index(provider_word: Int) raises -> Int:
    var cache = LOGISTIC_CONNECTIONS.get_or_create_ptr()
    for i in range(len(cache[].records)):  # driver-loop: connected native extension images
        if cache[].records[i].provider_word == provider_word:
            return i
    return -1


def logistic_connection_add(provider_word: Int, entry: NativeDescriptorCall) raises -> Int:
    var mode = native_descriptor_call(entry, QN_NATIVE_MODE, [], [], [])
    var host = native_descriptor_call(entry, QN_NATIVE_HOST, [], [], []) != 0
    var multiclass = native_descriptor_call(entry, QN_NATIVE_MULTICLASS_AVAILABLE, [], [], []) != 0
    var cache = LOGISTIC_CONNECTIONS.get_or_create_ptr()
    var index = len(cache[].records)
    cache[].records.append(LogisticConnection(provider_word, QnNativeDriver(entry), mode, host, multiclass))
    return index


def logistic_mode(connection: Int) raises -> Int:
    return LOGISTIC_CONNECTIONS.get_or_create_ptr()[].records[connection].mode


def logistic_multiclass(connection: Int) raises -> Bool:
    return LOGISTIC_CONNECTIONS.get_or_create_ptr()[].records[connection].multiclass


def logistic_driver(connection: Int) raises -> QnNativeDriver:
    return LOGISTIC_CONNECTIONS.get_or_create_ptr()[].records[connection].driver.copy()
