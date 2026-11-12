# SPDX-License-Identifier: Apache-2.0
"""Typed KMeans connection records (RUN): the driver, compiled numeric mode and
capabilities of each connected provider image, keyed by the provider's
address word. The door (`bindings/kmeans_descriptor_boundary_door.mojo`) admits
the provider and keeps its pin at the same index."""
from core.shared_slot import SharedSlot
from core.kmeans_native_api import KMEANS_MODE, KMEANS_CAPABILITIES, KMeansNativeDriver


@fieldwise_init
struct KMeansConnection(Movable):
    var provider_word: Int
    var driver: KMeansNativeDriver
    var mode: Int
    var capabilities: Int


struct KMeansConnections(Defaultable, Movable):
    var records: List[KMeansConnection]

    def __init__(out self):
        self.records = List[KMeansConnection]()


# V2: the record names its provider by address word; the pin moved to the door.
comptime KMEANS_CONNECTIONS = SharedSlot[KMeansConnections, "MojolearnKMeansConnectionsV2", 2]


def kmeans_connection_index(provider_word: Int) raises -> Int:
    var cache = KMEANS_CONNECTIONS.get_or_create_ptr()
    for index in range(len(cache[].records)):  # driver-loop: connected extension images
        if cache[].records[index].provider_word == provider_word:
            return index
    return -1


def kmeans_connection_add(provider_word: Int, var driver: KMeansNativeDriver) raises -> Int:
    var mode = driver.metadata(KMEANS_MODE)
    var capabilities = driver.metadata(KMEANS_CAPABILITIES)
    if mode != 0 and mode != 1:
        raise Error("KMeans native provider: unsupported compiled numeric mode")
    var cache = KMEANS_CONNECTIONS.get_or_create_ptr()
    var index = len(cache[].records)
    cache[].records.append(KMeansConnection(provider_word, driver^, mode, capabilities))
    return index


def kmeans_capabilities(connection: Int) raises -> Int:
    return KMEANS_CONNECTIONS.get_or_create_ptr()[].records[connection].capabilities


def kmeans_driver(connection: Int) raises -> KMeansNativeDriver:
    return KMEANS_CONNECTIONS.get_or_create_ptr()[].records[connection].driver.copy()
