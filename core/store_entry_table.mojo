# SPDX-License-Identifier: Apache-2.0
"""The resident device store's own entry table (lane lane-loads-3, 2026-11-28).

The ONE resident input store of a tier (x_prep/store_slot.mojo `X_PREP_STORE`
on `x_prep_ctx`: the store `resident()`, `DeviceRows`, the x_metrics programs,
the kernel metrics' slots, the member folds and the model-selection programs
keep inputs in) is a general device-buffer facility, not preprocessing. Its
store operations (put, take rows, free, live, read: the `XP_DEV_*` numbering of
`x_prep/transport_abi.mojo`) are published here as one entry by the base GPU
binding (`bindings/_mojolearn.mojo`, `core/store_transport_device.mojo`
`publish_store_transport_device`), so a caller that only keeps or reads a
resident slot resolves the store without the `_mojolearn_x_prep` image.

Before, those callers resolved the x_prep program transport
(`core/prep_entry_table.mojo` `xp_transport_ref`), which only the
`_mojolearn_x_prep` image publishes, and the Python label and metric
connections loaded that image for every classical estimator so the table was
published. The x_prep transport is unchanged: it keeps all its operations
(the store ones run the same body on the same store), and x_prep's own
program callers and the scaler cores still resolve it.

One publisher per table name (the base binding of that tier and vendor), one
store state per tier (the process slots of x_prep/store_slot.mojo). The
resolved ref carries the same layout word as an x_prep transport ref of the
same tier and vendor (`entry_ref_word(mode, vendor, XP_OP_COUNT)`), so a
scope cache's route word (bindings/arena_cache_run.mojo `arena_cache_route`)
is the same for both: two caches with one route share one device store.

A host image has no device store and publishes nothing here: vendor word 0
resolves to the table's own "provider is not published" refusal, by name.

Modeled on `core/base_entry_table.mojo`: the same `core/entry_table.mojo`
publication and lookup, separate slot names. Extending the layout requires a
new V suffix, never a new loader. Publication and lookup are small metadata
operations with no independent numerical work to split.
"""
from core.entry_table import EntryRef, entry_ref_word, entry_table_publish, entry_table_resolve
from core.native_descriptor_abi import NativeDescriptorCall
from x_prep.transport_abi import XP_OP_COUNT

comptime STORE_TRANSPORT_ABI = "mojolearn.store.transport.v1"


def _store_entry_key_check(mode: Int, vendor: Int) raises:
    # The words of core/prep_entry_table.mojo `_prep_entry_key_check`: the
    # callers' key was a PrepEntryKey before, and its refusal is unchanged.
    if mode < 0 or mode > 1 or vendor < 0 or vendor > 3:
        raise Error("native prep entries: invalid tier/vendor key")


def store_table_publish(mode: Int, vendor: Int, publisher: Int,
    schemas: List[String], operations: List[Int], entries: List[NativeDescriptorCall]) raises:
    """Publish the store entry of the GPU image (`mode`, `vendor`). Vendor
    word 0 (a host image) is refused: the host has no device store."""
    _store_entry_key_check(mode, vendor)
    if vendor == 1 and mode == 1:
        comptime NAME = "MojolearnEntriesStoreIdenticalMetalV1"
        entry_table_publish[NAME](mode, vendor, publisher, schemas, operations, entries)
        return
    if vendor == 1 and mode == 0:
        comptime NAME = "MojolearnEntriesStoreFastMetalV1"
        entry_table_publish[NAME](mode, vendor, publisher, schemas, operations, entries)
        return
    if vendor == 2 and mode == 1:
        comptime NAME = "MojolearnEntriesStoreIdenticalCUDAV1"
        entry_table_publish[NAME](mode, vendor, publisher, schemas, operations, entries)
        return
    if vendor == 2 and mode == 0:
        comptime NAME = "MojolearnEntriesStoreFastCUDAV1"
        entry_table_publish[NAME](mode, vendor, publisher, schemas, operations, entries)
        return
    if vendor == 3 and mode == 1:
        comptime NAME = "MojolearnEntriesStoreIdenticalHIPV1"
        entry_table_publish[NAME](mode, vendor, publisher, schemas, operations, entries)
        return
    if vendor == 3 and mode == 0:
        comptime NAME = "MojolearnEntriesStoreFastHIPV1"
        entry_table_publish[NAME](mode, vendor, publisher, schemas, operations, entries)
        return
    raise Error("native prep entries: unsupported execution key")


def xp_store_ref(mode: Int, vendor: Int) raises -> EntryRef:
    """The resident store entry of the image (`mode`, `vendor`), resolved
    once by its caller (per call, per run or per scope close, as the x_prep
    transport ref was); the table refusals raise here. A call through it is
    `xp_transport_call` (core/prep_entry_table.mojo) with an `XP_DEV_*`
    operation: one indirect call."""
    _store_entry_key_check(mode, vendor)
    if vendor == 1 and mode == 1:
        comptime NAME = "MojolearnEntriesStoreIdenticalMetalV1"
        return entry_table_resolve[NAME](mode, vendor, 0, String(STORE_TRANSPORT_ABI), XP_OP_COUNT)
    if vendor == 1 and mode == 0:
        comptime NAME = "MojolearnEntriesStoreFastMetalV1"
        return entry_table_resolve[NAME](mode, vendor, 0, String(STORE_TRANSPORT_ABI), XP_OP_COUNT)
    if vendor == 2 and mode == 1:
        comptime NAME = "MojolearnEntriesStoreIdenticalCUDAV1"
        return entry_table_resolve[NAME](mode, vendor, 0, String(STORE_TRANSPORT_ABI), XP_OP_COUNT)
    if vendor == 2 and mode == 0:
        comptime NAME = "MojolearnEntriesStoreFastCUDAV1"
        return entry_table_resolve[NAME](mode, vendor, 0, String(STORE_TRANSPORT_ABI), XP_OP_COUNT)
    if vendor == 3 and mode == 1:
        comptime NAME = "MojolearnEntriesStoreIdenticalHIPV1"
        return entry_table_resolve[NAME](mode, vendor, 0, String(STORE_TRANSPORT_ABI), XP_OP_COUNT)
    if vendor == 3 and mode == 0:
        comptime NAME = "MojolearnEntriesStoreFastHIPV1"
        return entry_table_resolve[NAME](mode, vendor, 0, String(STORE_TRANSPORT_ABI), XP_OP_COUNT)
    # A host key: nothing publishes these names, so the read below is the
    # table's "provider is not published", by name (never a host route here).
    if vendor == 0 and mode == 1:
        comptime NAME = "MojolearnEntriesStoreIdenticalCPUV1"
        return entry_table_resolve[NAME](mode, vendor, 0, String(STORE_TRANSPORT_ABI), XP_OP_COUNT)
    if vendor == 0 and mode == 0:
        comptime NAME = "MojolearnEntriesStoreFastCPUV1"
        return entry_table_resolve[NAME](mode, vendor, 0, String(STORE_TRANSPORT_ABI), XP_OP_COUNT)
    raise Error("native prep entries: unsupported execution key")
