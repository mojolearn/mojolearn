# SPDX-License-Identifier: Apache-2.0
"""Typed TruncatedSVD connection records (RUN): the driver, compiled numeric
mode, vendor, capabilities and qfix word of each connected provider image,
keyed by the provider's address word. The door
(`bindings/tsvd_descriptor_boundary_door.mojo`) admits the provider and keeps
its pin at the same index."""
from core.shared_slot import SharedSlot
from core.tsvd_native_api import TSVD_MODE, TSVD_VENDOR, TSVD_CAPABILITIES, TSVD_QFIX, TSVDNativeDriver


@fieldwise_init
struct TSVDConnection(Movable):
    var provider_word: Int
    var driver: TSVDNativeDriver
    var mode: Int
    var vendor: Int
    var capabilities: Int
    var qfix: Int


struct TSVDConnections(Defaultable, Movable):
    var records: List[TSVDConnection]

    def __init__(out self):
        self.records = List[TSVDConnection]()


# V2: the record names its provider by address word; the pin moved to the door.
comptime TSVD_CONNECTIONS = SharedSlot[TSVDConnections, "MojolearnTSVDConnectionsV2", 2]


def tsvd_connection_index(provider_word: Int) raises -> Int:
    var cache = TSVD_CONNECTIONS.get_or_create_ptr()
    for candidate in range(len(cache[].records)):  # driver-loop: connected native provider images
        if cache[].records[candidate].provider_word == provider_word:
            return candidate
    return -1


def tsvd_connection_add(provider_word: Int, var driver: TSVDNativeDriver) raises -> Int:
    var native_mode = driver.metadata(TSVD_MODE)
    var native_vendor = driver.metadata(TSVD_VENDOR)
    var capabilities = driver.metadata(TSVD_CAPABILITIES)
    var qfix = driver.metadata(TSVD_QFIX)
    var cache = TSVD_CONNECTIONS.get_or_create_ptr()
    var index = len(cache[].records)
    cache[].records.append(TSVDConnection(provider_word, driver^, native_mode, native_vendor, capabilities, qfix))
    return index


def tsvd_mode(connection: Int) raises -> Int:
    return TSVD_CONNECTIONS.get_or_create_ptr()[].records[connection].mode


def tsvd_vendor(connection: Int) raises -> Int:
    return TSVD_CONNECTIONS.get_or_create_ptr()[].records[connection].vendor


def tsvd_capabilities(connection: Int) raises -> Int:
    return TSVD_CONNECTIONS.get_or_create_ptr()[].records[connection].capabilities


def tsvd_qfix(connection: Int) raises -> Int:
    return TSVD_CONNECTIONS.get_or_create_ptr()[].records[connection].qfix


def tsvd_driver(connection: Int) raises -> TSVDNativeDriver:
    return TSVD_CONNECTIONS.get_or_create_ptr()[].records[connection].driver.copy()
