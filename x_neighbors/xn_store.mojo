# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The x_neighbors GPU image's resident store and its scope transport entry (lane gd-resident-2, 2026-11-25).

A `DeviceStore` (core/device_store.mojo) on the x_neighbors context (x_neighbors/device_ops.mojo `xn_ctx`), so a
slot is a buffer the x_neighbors kernels read directly. `xn_store_v1` is the image's transport entry with the
x_prep numbering (core/family_store_transport.mojo). Inside `resident()` a public function's input (the dense
adjacency of `connected_components`) is found or put ONCE in the scope's cache of the family `XN_STORE_PREFIX`
(core/scope_store_slot.mojo) and the kernels read the slot in place; the scope's close frees it through this entry.

GPU image only: the host image never reads the scope words and keeps its host path.
"""
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from checks.vendor import COMPILED_VENDOR
from core.device_store import DeviceStore
from core.shared_slot import SharedSlot
from core.entry_table import EntryRef
from core.native_descriptor_abi import NativeDescriptorError
from core.family_store_transport import family_store_entry
from core.native_descriptor_call import native_descriptor_report
from core.scope_store_slot import ScopeInput, scope_store_ref, scope_store_slot
from x_neighbors.device_ops import xn_ctx

#: The scope cache's family prefix of this store.
comptime XN_STORE_PREFIX = "x_neighbors"

comptime XN_STORE = SharedSlot[DeviceStore, ("MojolearnXNeighborsStoreIdenticalV1" if GLOBAL_NUMERIC_MODE
                                             == NUMERIC_IDENTICAL else "MojolearnXNeighborsStoreFastV1"), 2]

comptime _VENDOR_CODE = (1 if COMPILED_VENDOR == "metal" else 2 if COMPILED_VENDOR == "cuda" else 3
                         if COMPILED_VENDOR == "hip" else -1)


def xn_store_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int, result: Int,
                context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The store's transport entry (ARENA_STORE_PUT / FREE / TAKE_ROWS / LIVE / READ)."""
    try:
        var ctx = xn_ctx()
        return family_store_entry(XN_STORE.get_or_create_ptr()[], ctx, "x_neighbors", operation, aa, na, pp, np,
                                  nf, result, context, sink)
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


def xn_store_ref() -> EntryRef:
    return scope_store_ref(xn_store_v1, Int(GLOBAL_NUMERIC_MODE), _VENDOR_CODE)


def xn_scope_buffer(given: ScopeInput, address: Int, words: Int) raises -> DeviceBuffer[DType.float32]:
    """The resident slot of the caller's `words` 4-byte words at `address` inside the door's scope (found, or put
    once), as a reference to the store's buffer (it keeps the device memory alive while the caller's kernels read
    it). Only for `given.wanted()`, a host `address` and `words` >= 1."""
    var slot = scope_store_slot(xn_store_ref(), String(XN_STORE_PREFIX), given, address, words)
    return XN_STORE.get_or_create_ptr()[]._held(slot, words)
