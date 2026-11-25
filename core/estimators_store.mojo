# SPDX-License-Identifier: Apache-2.0
"""The estimators GPU image's resident store and its scope transport entry (lane vdf-scope-cluster2b,
2026-11-24).

A `DeviceStore` (core/device_store.mojo) on the estimators image's one process context
(`MojolearnEstimatorsContext*`, the context of kde/resident_fit.mojo and the DBSCAN provider), so a slot is a
buffer those kernels read directly. `estimators_store_v1` is the image's transport entry with the x_prep numbering
(core/family_store_transport.mojo): the `resident()` scope keeps a model-selection X here ONCE
(`arena_scope_store_put`, prefix `ESTIMATORS_STORE_PREFIX`) and each fold's rows are taken on the device
(`arena_store_take_rows`). The binding exports it as the capsule `estimators_store_v1`; a member route carries it
in `route.second` when the image answers the family's store capability (KernelDensity: `KDE_CAP_DEV_STORE`,
providers/python/member_density_boundary_door.mojo). GPU image only.
"""
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.device_store import DeviceStore
from core.shared_slot import SharedSlot
from core.neural_context import process_ctx
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.family_store_transport import family_store_entry
from x_prep.transport_abi import XP_TRANSPORT_ABI

#: The scope cache's family prefix of this store.
comptime ESTIMATORS_STORE_PREFIX = "estimators"

# Exactly the estimators image's owning context (kde/resident_fit.mojo, core/dbscan_native_device.mojo).
comptime _DEVCTX_SLOT = ("MojolearnEstimatorsContextIdenticalV3" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
                         else "MojolearnEstimatorsContextFastV3")

comptime ESTIMATORS_STORE = SharedSlot[DeviceStore, ("MojolearnEstimatorsStoreIdenticalV2"
                                                     if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
                                                     else "MojolearnEstimatorsStoreFastV2"), 2]


def estimators_store_ctx() raises -> DeviceContext:
    """The context every slot of the store is made on: the estimators image's kernels' own."""
    return process_ctx[_DEVCTX_SLOT]()


def estimators_store_buffer(slot: Int, words: Int) raises -> DeviceBuffer[DType.float32]:
    """A reference to slot `slot`'s buffer (at least `words` words), copied out under the store's lock: it keeps
    the device memory alive past a concurrent free while the caller's kernels read it."""
    return ESTIMATORS_STORE.get_or_create_ptr()[]._held(slot, words)


def estimators_store_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int, result: Int,
                        context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The store's transport entry (ARENA_STORE_PUT / FREE / TAKE_ROWS / LIVE)."""
    try:
        var ctx = estimators_store_ctx()
        return family_store_entry(ESTIMATORS_STORE.get_or_create_ptr()[], ctx, "estimators", operation, aa, na,
                                  pp, np, nf, result, context, sink)
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct EstimatorsStoreAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(XP_TRANSPORT_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(estimators_store_v1)


comptime ESTIMATORS_STORE_API = SharedSlot[EstimatorsStoreAPI, ("MojolearnEstimatorsStoreAPIIdenticalV1"
                                                                if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
                                                                else "MojolearnEstimatorsStoreAPIFastV1")]
