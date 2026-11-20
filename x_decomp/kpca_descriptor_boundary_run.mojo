# SPDX-License-Identifier: Apache-2.0
"""Typed KernelPCA connection records (RUN): drivers, compiled numeric mode,
vendor and capabilities of each connected provider pair, keyed by the
providers' address words. The door
(`bindings/kpca_descriptor_boundary_door.mojo`) admits the providers and keeps
their pins at the same index."""
from core.shared_slot import SharedSlot
from core.native_descriptor_abi import NativeDescriptorCall
from core.kpca_native_api import KPCA_MODE, KPCA_VENDOR, KPCA_CAPABILITIES, KPCA_CAP_RESIDENT, KPCANativeDriver
from core.kpca_native_plan import KPCANeighborDriver


@fieldwise_init
struct KPCANearConnection(Movable):
    var provider_word: Int
    var metadata_word: Int
    var driver: KPCANeighborDriver
    var mode: Int
    var vendor: Int
    var capabilities: Int


@fieldwise_init
struct KPCADecompConnection(Movable):
    var provider_word: Int
    var reduction_word: Int
    var driver: KPCANativeDriver
    var reduction: NativeDescriptorCall
    var mode: Int
    var vendor: Int
    var capabilities: Int


struct KPCAConnections(Defaultable, Movable):
    var near: List[KPCANearConnection]
    var decomp: List[KPCADecompConnection]

    def __init__(out self):
        self.near = List[KPCANearConnection]()
        self.decomp = List[KPCADecompConnection]()


# V2: the records name their providers by address word; the pins moved to the door.
comptime KPCA_CONNECTIONS = SharedSlot[KPCAConnections, "MojolearnKPCAConnectionsV2", 2]


@fieldwise_init
struct KPCAConnectionFacts(Copyable, Movable):
    var mode: Int
    var vendor: Int
    var capabilities: Int


def kpca_near_index(provider_word: Int, metadata_word: Int) raises -> Int:
    var cache = KPCA_CONNECTIONS.get_or_create_ptr()
    for i in range(len(cache[].near)):  # driver-loop: admitted native provider images
        if cache[].near[i].provider_word == provider_word and cache[].near[i].metadata_word == metadata_word:
            return i
    return -1


def kpca_near_add(provider_word: Int, metadata_word: Int, var info: KPCANativeDriver,
                  var driver: KPCANeighborDriver) raises -> Int:
    var native_mode = info.metadata(KPCA_MODE)
    var vendor = info.metadata(KPCA_VENDOR)
    var capabilities = info.metadata(KPCA_CAPABILITIES)
    var cache = KPCA_CONNECTIONS.get_or_create_ptr()
    var index = len(cache[].near)
    cache[].near.append(KPCANearConnection(provider_word, metadata_word, driver^, native_mode, vendor, capabilities))
    return index


def kpca_near_facts(connection: Int) raises -> KPCAConnectionFacts:
    var cache = KPCA_CONNECTIONS.get_or_create_ptr()
    return KPCAConnectionFacts(cache[].near[connection].mode, cache[].near[connection].vendor,
        cache[].near[connection].capabilities)


def kpca_near_driver(connection: Int) raises -> KPCANeighborDriver:
    return KPCA_CONNECTIONS.get_or_create_ptr()[].near[connection].driver.copy()


def kpca_decomp_index(provider_word: Int, reduction_word: Int) raises -> Int:
    var cache = KPCA_CONNECTIONS.get_or_create_ptr()
    for i in range(len(cache[].decomp)):  # driver-loop: admitted native provider images
        if cache[].decomp[i].provider_word == provider_word and cache[].decomp[i].reduction_word == reduction_word:
            return i
    return -1


def kpca_decomp_add(provider_word: Int, reduction_word: Int, var driver: KPCANativeDriver,
                    reduction: NativeDescriptorCall) raises -> Int:
    var native_mode = driver.metadata(KPCA_MODE)
    var vendor = driver.metadata(KPCA_VENDOR)
    var capabilities = driver.metadata(KPCA_CAPABILITIES)
    var cache = KPCA_CONNECTIONS.get_or_create_ptr()
    var index = len(cache[].decomp)
    cache[].decomp.append(KPCADecompConnection(provider_word, reduction_word, driver^, reduction,
        native_mode, vendor, capabilities))
    return index


def kpca_decomp_facts(connection: Int) raises -> KPCAConnectionFacts:
    var cache = KPCA_CONNECTIONS.get_or_create_ptr()
    return KPCAConnectionFacts(cache[].decomp[connection].mode, cache[].decomp[connection].vendor,
        cache[].decomp[connection].capabilities)


def kpca_decomp_driver(connection: Int) raises -> KPCANativeDriver:
    return KPCA_CONNECTIONS.get_or_create_ptr()[].decomp[connection].driver.copy()


def kpca_decomp_reduction(connection: Int) raises -> NativeDescriptorCall:
    return KPCA_CONNECTIONS.get_or_create_ptr()[].decomp[connection].reduction


def kpca_resident(connection: Int) raises -> Bool:
    return (KPCA_CONNECTIONS.get_or_create_ptr()[].near[connection].capabilities & KPCA_CAP_RESIDENT) != 0
