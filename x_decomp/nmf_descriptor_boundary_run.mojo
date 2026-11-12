# SPDX-License-Identifier: Apache-2.0
"""Typed NMF connection records (RUN): the driver, compiled numeric mode,
vendor and capabilities of each connected provider image, keyed by the
provider's address word. The door (`bindings/nmf_descriptor_boundary_door.mojo`)
admits the provider and keeps its pin at the same index."""
from core.shared_slot import SharedSlot
from core.nmf_native_api import NMF_MODE, NMF_VENDOR, NMF_CAPABILITIES, NMFNativeDriver


@fieldwise_init
struct NMFConnection(Movable):
    var provider_word: Int
    var driver: NMFNativeDriver
    var mode: Int
    var vendor: Int
    var capabilities: Int


struct NMFConnections(Defaultable, Movable):
    var records: List[NMFConnection]

    def __init__(out self):
        self.records = List[NMFConnection]()


# V2: the record names its provider by address word; the pin moved to the door.
comptime NMF_CONNECTIONS = SharedSlot[NMFConnections, "MojolearnNMFConnectionsV2", 2]


def nmf_connection_index(provider_word: Int) raises -> Int:
    var cache = NMF_CONNECTIONS.get_or_create_ptr()
    for candidate in range(len(cache[].records)):  # driver-loop: admitted native provider images
        if cache[].records[candidate].provider_word == provider_word:
            return candidate
    return -1


def nmf_connection_add(provider_word: Int, var driver: NMFNativeDriver) raises -> Int:
    var native_mode = driver.metadata(NMF_MODE)
    var vendor = driver.metadata(NMF_VENDOR)
    var capabilities = driver.metadata(NMF_CAPABILITIES)
    var cache = NMF_CONNECTIONS.get_or_create_ptr()
    var index = len(cache[].records)
    cache[].records.append(NMFConnection(provider_word, driver^, native_mode, vendor, capabilities))
    return index


def nmf_mode(connection: Int) raises -> Int:
    return NMF_CONNECTIONS.get_or_create_ptr()[].records[connection].mode


def nmf_vendor(connection: Int) raises -> Int:
    return NMF_CONNECTIONS.get_or_create_ptr()[].records[connection].vendor


def nmf_capabilities(connection: Int) raises -> Int:
    return NMF_CONNECTIONS.get_or_create_ptr()[].records[connection].capabilities


def nmf_driver(connection: Int) raises -> NMFNativeDriver:
    return NMF_CONNECTIONS.get_or_create_ptr()[].records[connection].driver.copy()
