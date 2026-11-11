# SPDX-License-Identifier: Apache-2.0
"""Tier/vendor-qualified native tables of the preprocessing family (lane
calltable-prep-decomp, 2026-11-11).

Modeled on `core/trees_entry_table.mojo`: the same `core/entry_table.mojo`
publication and lookup, separate slot names. Family 0 is the x_prep program
transport (`_mojolearn_x_prep` / `_mojolearn_x_prep_host`); its one entry is
the image's transport dispatch over the operations of
`x_prep/transport_abi.mojo`. A caller derives the key from the connected
image's compiled mode and vendor words (read once per binding,
`bindings/prep_entry_cache.mojo`) and calls the pointer with plain spans; no
Python object crosses.

Extending a table's layout requires a new V suffix, never a new loader.
Publication and lookup are small metadata operations with no independent
numerical work to split.
"""
from core.entry_table import entry_table_publish, entry_table_read
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_call import native_descriptor_call
from x_prep.transport_abi import XP_TRANSPORT_ABI, XP_OP_COUNT

comptime PREP_TABLE_X_PREP = 0


@fieldwise_init
struct PrepEntryKey(Copyable, Movable):
    var mode: Int
    var vendor: Int


def _prep_entry_key_check(key: PrepEntryKey) raises:
    if key.mode < 0 or key.mode > 1 or key.vendor < 0 or key.vendor > 3:
        raise Error("native prep entries: invalid tier/vendor key")


def prep_table_publish[family: Int](key: PrepEntryKey, publisher: Int,
    schemas: List[String], operations: List[Int], entries: List[NativeDescriptorCall]) raises:
    comptime assert family == PREP_TABLE_X_PREP, "unknown prep table family"
    _prep_entry_key_check(key)
    if key.vendor == 0 and key.mode == 1:
        comptime NAME = "MojolearnEntriesXPrepIdenticalCPUV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 0 and key.mode == 0:
        comptime NAME = "MojolearnEntriesXPrepFastCPUV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 1 and key.mode == 1:
        comptime NAME = "MojolearnEntriesXPrepIdenticalMetalV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 1 and key.mode == 0:
        comptime NAME = "MojolearnEntriesXPrepFastMetalV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 2 and key.mode == 1:
        comptime NAME = "MojolearnEntriesXPrepIdenticalCUDAV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 2 and key.mode == 0:
        comptime NAME = "MojolearnEntriesXPrepFastCUDAV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 3 and key.mode == 1:
        comptime NAME = "MojolearnEntriesXPrepIdenticalHIPV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 3 and key.mode == 0:
        comptime NAME = "MojolearnEntriesXPrepFastHIPV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    raise Error("native prep entries: unsupported execution key")


def prep_table_entry[family: Int](key: PrepEntryKey, index: Int,
    schema: String, operations: Int) raises -> NativeDescriptorCall:
    comptime assert family == PREP_TABLE_X_PREP, "unknown prep table family"
    _prep_entry_key_check(key)
    if key.vendor == 0 and key.mode == 1:
        comptime NAME = "MojolearnEntriesXPrepIdenticalCPUV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 0 and key.mode == 0:
        comptime NAME = "MojolearnEntriesXPrepFastCPUV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 1 and key.mode == 1:
        comptime NAME = "MojolearnEntriesXPrepIdenticalMetalV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 1 and key.mode == 0:
        comptime NAME = "MojolearnEntriesXPrepFastMetalV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 2 and key.mode == 1:
        comptime NAME = "MojolearnEntriesXPrepIdenticalCUDAV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 2 and key.mode == 0:
        comptime NAME = "MojolearnEntriesXPrepFastCUDAV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 3 and key.mode == 1:
        comptime NAME = "MojolearnEntriesXPrepIdenticalHIPV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 3 and key.mode == 0:
        comptime NAME = "MojolearnEntriesXPrepFastHIPV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    raise Error("native prep entries: unsupported execution key")


def xp_transport_call(key: PrepEntryKey, operation: Int, addresses: List[Int], ints: List[Int]) raises -> Int:
    """One x_prep transport operation in the image `key` names. The caller
    keeps every addressed buffer alive until the call returns."""
    var entry = prep_table_entry[PREP_TABLE_X_PREP](key, 0, String(XP_TRANSPORT_ABI), XP_OP_COUNT)
    return native_descriptor_call(entry, operation, addresses, ints, List[Float64]())
