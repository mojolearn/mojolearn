# SPDX-License-Identifier: Apache-2.0
"""Tier/vendor-qualified native tables of the trees family.

Modeled on `core/linear_entry_table.mojo`: the same `core/entry_table.mojo`
publication and lookup, separate slot names. Family 0 is the x_trees glue
binding (`_mojolearn_x_trees` / `_mojolearn_x_trees_host`); its one entry is
`xtrees/glue_runs.mojo::x_trees_glue_v1`, a dispatch over the glue operations
of `xtrees/glue_abi.mojo`. A caller reads the slot of the binding the trees
owner connected for its tier (the key is the bridge's `x_trees_native_key`)
and calls the pointer with plain spans; no Python object crosses.

Extending a table's layout requires a new V suffix, never a new loader.
Publication and lookup are small metadata operations with no independent
numerical work to split.
"""
from core.entry_table import entry_table_publish, entry_table_read
from core.native_descriptor_abi import NativeDescriptorCall

comptime TREES_TABLE_X_TREES = 0


@fieldwise_init
struct TreesEntryKey(Copyable, Movable):
    var mode: Int
    var vendor: Int


def _trees_entry_key_check(key: TreesEntryKey) raises:
    if key.mode < 0 or key.mode > 1 or key.vendor < 0 or key.vendor > 3:
        raise Error("native trees entries: invalid tier/vendor key")


def trees_table_publish[family: Int](key: TreesEntryKey, publisher: Int,
    schemas: List[String], operations: List[Int], entries: List[NativeDescriptorCall]) raises:
    comptime assert family == TREES_TABLE_X_TREES, "unknown trees table family"
    _trees_entry_key_check(key)
    if key.vendor == 0 and key.mode == 1:
        comptime NAME = "MojolearnEntriesXTreesIdenticalCPUV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 0 and key.mode == 0:
        comptime NAME = "MojolearnEntriesXTreesFastCPUV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 1 and key.mode == 1:
        comptime NAME = "MojolearnEntriesXTreesIdenticalMetalV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 1 and key.mode == 0:
        comptime NAME = "MojolearnEntriesXTreesFastMetalV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 2 and key.mode == 1:
        comptime NAME = "MojolearnEntriesXTreesIdenticalCUDAV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 2 and key.mode == 0:
        comptime NAME = "MojolearnEntriesXTreesFastCUDAV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 3 and key.mode == 1:
        comptime NAME = "MojolearnEntriesXTreesIdenticalHIPV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 3 and key.mode == 0:
        comptime NAME = "MojolearnEntriesXTreesFastHIPV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    raise Error("native trees entries: unsupported execution key")


def trees_table_entry[family: Int](key: TreesEntryKey, index: Int,
    schema: String, operations: Int) raises -> NativeDescriptorCall:
    comptime assert family == TREES_TABLE_X_TREES, "unknown trees table family"
    _trees_entry_key_check(key)
    if key.vendor == 0 and key.mode == 1:
        comptime NAME = "MojolearnEntriesXTreesIdenticalCPUV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 0 and key.mode == 0:
        comptime NAME = "MojolearnEntriesXTreesFastCPUV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 1 and key.mode == 1:
        comptime NAME = "MojolearnEntriesXTreesIdenticalMetalV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 1 and key.mode == 0:
        comptime NAME = "MojolearnEntriesXTreesFastMetalV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 2 and key.mode == 1:
        comptime NAME = "MojolearnEntriesXTreesIdenticalCUDAV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 2 and key.mode == 0:
        comptime NAME = "MojolearnEntriesXTreesFastCUDAV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 3 and key.mode == 1:
        comptime NAME = "MojolearnEntriesXTreesIdenticalHIPV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 3 and key.mode == 0:
        comptime NAME = "MojolearnEntriesXTreesFastHIPV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    raise Error("native trees entries: unsupported execution key")
