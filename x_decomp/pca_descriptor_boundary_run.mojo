# SPDX-License-Identifier: Apache-2.0
"""Typed PCA connection records (RUN): the driver, compiled numeric mode,
vendor and capabilities of each connected provider image, keyed by the
provider's address word. The door (`bindings/pca_descriptor_boundary_door.mojo`)
admits the provider and keeps its pin at the same index."""
from core.shared_slot import SharedSlot
from core.pca_native_api import (
    PCA_MODE, PCA_VENDOR, PCA_CAPABILITIES, PCA_CAP_WHITEN, PCANativeDriver, pca_vendor_name,
)


@fieldwise_init
struct PCAConnection(Movable):
    var provider_word: Int
    var driver: PCANativeDriver
    var mode: Int
    var vendor: Int
    var capabilities: Int


struct PCAConnections(Defaultable, Movable):
    var records: List[PCAConnection]

    def __init__(out self):
        self.records = List[PCAConnection]()


# V2: the record names its provider by address word; the pin moved to the door.
comptime PCA_CONNECTIONS = SharedSlot[PCAConnections, "MojolearnPCAConnectionsV2", 2]


def pca_connection_index(provider_word: Int) raises -> Int:
    var cache = PCA_CONNECTIONS.get_or_create_ptr()
    for candidate in range(len(cache[].records)):  # driver-loop: connected compiled provider images
        if cache[].records[candidate].provider_word == provider_word:
            return candidate
    return -1


def pca_connection_add(provider_word: Int, var driver: PCANativeDriver) raises -> Int:
    var native_mode = driver.metadata(PCA_MODE)
    var vendor = driver.metadata(PCA_VENDOR)
    var capabilities = driver.metadata(PCA_CAPABILITIES)
    var cache = PCA_CONNECTIONS.get_or_create_ptr()
    var index = len(cache[].records)
    cache[].records.append(PCAConnection(provider_word, driver^, native_mode, vendor, capabilities))
    return index


def pca_driver(connection: Int) raises -> PCANativeDriver:
    return PCA_CONNECTIONS.get_or_create_ptr()[].records[connection].driver.copy()


def pca_capabilities(connection: Int) raises -> Int:
    return PCA_CONNECTIONS.get_or_create_ptr()[].records[connection].capabilities


def pca_mode(connection: Int) raises -> Int:
    return PCA_CONNECTIONS.get_or_create_ptr()[].records[connection].mode


def pca_vendor(connection: Int) raises -> String:
    return pca_vendor_name(PCA_CONNECTIONS.get_or_create_ptr()[].records[connection].vendor)


def pca_whiten_capability(connection: Int) raises:
    if (pca_capabilities(connection) & PCA_CAP_WHITEN) == 0:
        raise Error("NotImplementedError: PCA whiten=True requires the compiled whitened transform pair")
