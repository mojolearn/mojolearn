# SPDX-License-Identifier: Apache-2.0
"""Typed IPCA connection records (RUN): the driver, compiled numeric mode,
vendor and capabilities of each connected provider image, keyed by the
provider's address word. The door (`bindings/ipca_descriptor_boundary_door.mojo`)
admits the provider and keeps its pin at the same index."""
from core.shared_slot import SharedSlot
from core.ipca_native_api import IPCA_MODE, IPCA_VENDOR, IPCA_CAPABILITIES, IPCA_CAP_RESIDENT, IPCANativeDriver


@fieldwise_init
struct IPCAConnection(Movable):
    var provider_word: Int
    var driver: IPCANativeDriver
    var mode: Int
    var vendor: Int
    var capabilities: Int


struct IPCAConnections(Defaultable, Movable):
    var records: List[IPCAConnection]

    def __init__(out self):
        self.records = List[IPCAConnection]()


# V2: the record names its provider by address word; the pin moved to the door.
comptime IPCA_CONNECTIONS = SharedSlot[IPCAConnections, "MojolearnIPCAConnectionsV2", 2]


def ipca_connection_index(provider_word: Int) raises -> Int:
    var cache = IPCA_CONNECTIONS.get_or_create_ptr()
    for candidate in range(len(cache[].records)):  # driver-loop: admitted native provider images
        if cache[].records[candidate].provider_word == provider_word:
            return candidate
    return -1


def ipca_connection_add(provider_word: Int, var driver: IPCANativeDriver) raises -> Int:
    var native_mode = driver.metadata(IPCA_MODE)
    var vendor = driver.metadata(IPCA_VENDOR)
    var capabilities = driver.metadata(IPCA_CAPABILITIES)
    var cache = IPCA_CONNECTIONS.get_or_create_ptr()
    var index = len(cache[].records)
    cache[].records.append(IPCAConnection(provider_word, driver^, native_mode, vendor, capabilities))
    return index


def ipca_mode(connection: Int) raises -> Int:
    return IPCA_CONNECTIONS.get_or_create_ptr()[].records[connection].mode


def ipca_vendor(connection: Int) raises -> Int:
    return IPCA_CONNECTIONS.get_or_create_ptr()[].records[connection].vendor


def ipca_capabilities(connection: Int) raises -> Int:
    return IPCA_CONNECTIONS.get_or_create_ptr()[].records[connection].capabilities


def ipca_driver(connection: Int) raises -> IPCANativeDriver:
    return IPCA_CONNECTIONS.get_or_create_ptr()[].records[connection].driver.copy()


def ipca_resident(connection: Int) raises -> Bool:
    return (ipca_capabilities(connection) & IPCA_CAP_RESIDENT) != 0
