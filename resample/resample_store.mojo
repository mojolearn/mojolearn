# SPDX-License-Identifier: Apache-2.0
"""The resample GPU image's resident store and its scope transport entry (lane gd-rf-resample, 2026-11-25).

A `DeviceStore` (core/device_store.mojo) on the resample kernels' own context (`RESAMPLE_CTX_SLOT`, the slot
resample/estimator.mojo launches on), so a slot is a buffer the bootstrap, permutation and row-gather kernels read
IN PLACE. `resample_store_v1` is the image's transport entry with the x_prep numbering
(core/family_store_transport.mojo): inside `with mojolearn._arena_io.resident():` the public door
(bindings/linalg_ops_resample_door.mojo, providers/python/resample_rows_pyapi_door.mojo) keeps the CALLER'S
input buffer here once (bindings/arena_cache_run.mojo `arena_scope_store_put`, prefix `RESAMPLE_STORE_PREFIX`)
and hands the slot id to the operation, whose upload site reads `resample_store_buffer` instead of copying the
host words again. The binding exports the entry as the capsule `resample_store_v1`; the door connects it once
per process, only inside a scope, and only when the image answers `RS_OP_STORE_ROUTE` (resample_native.mojo).

GPU image only: the host image (resample/host/) has no store, answers the route operation with -1 and never
receives a slot id.
"""
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from checks.vendor import COMPILED_VENDOR
from core.device_route import device_vendor_code
from core.device_store import DeviceStore
from core.shared_slot import SharedSlot
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.neural_context import process_ctx
from core.family_store_transport import family_store_entry
from x_prep.transport_abi import XP_TRANSPORT_ABI

#: The scope cache's family prefix of this store (`resident.stats()` lists it under this name).
comptime RESAMPLE_STORE_PREFIX = "resample"

#: The resample kernels' context slot (resample/estimator.mojo `_DEVCTX_SLOT` is this value): the store's slots
#: are made on the context the kernels launch on.
comptime RESAMPLE_CTX_SLOT = ("MojolearnResampleContextIdenticalV3" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
                              else "MojolearnResampleContextFastV3")

comptime RESAMPLE_STORE = SharedSlot[DeviceStore, ("MojolearnResampleStoreIdenticalV1"
                                                   if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
                                                   else "MojolearnResampleStoreFastV1"), 2]


def resample_store_route() -> Int:
    """The scope cache's route word of this store: (tier << 8) | vendor, the `arena_cache_route` layout; -1 for
    an image without a GPU vendor (it keeps nothing resident)."""
    var vendor = device_vendor_code(String(COMPILED_VENDOR))
    if vendor <= 0:
        return -1
    return ((GLOBAL_NUMERIC_MODE & 0xFF) << 8) | (vendor & 0xFF)


def resample_store_ctx() raises -> DeviceContext:
    """The context every slot of the store is made on: the resample kernels' own."""
    return process_ctx[RESAMPLE_CTX_SLOT]()


def resample_store_buffer(slot: Int, words: Int) raises -> DeviceBuffer[DType.float32]:
    """A reference to slot `slot`'s buffer (holding at least `words` words), copied out under the store's lock: it
    keeps the device memory alive past a concurrent free while the caller's kernels read it. Read only: no
    resample kernel writes its input buffer."""
    return RESAMPLE_STORE.get_or_create_ptr()[]._held(slot, words)


def resample_store_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int, result: Int,
                      context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The store's transport entry (ARENA_STORE_PUT / FREE / TAKE_ROWS / LIVE / READ)."""
    try:
        var ctx = resample_store_ctx()
        return family_store_entry(RESAMPLE_STORE.get_or_create_ptr()[], ctx, "resample", operation, aa, na, pp, np,
                                  nf, result, context, sink)
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct ResampleStoreAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(XP_TRANSPORT_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(resample_store_v1)


comptime RESAMPLE_STORE_API = SharedSlot[ResampleStoreAPI, ("MojolearnResampleStoreAPIIdenticalV1"
                                                            if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
                                                            else "MojolearnResampleStoreAPIFastV1")]
