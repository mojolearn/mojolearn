# SPDX-License-Identifier: Apache-2.0
"""The x_trees DART table of the trees family (lane calltable-gbdt-forest-dart,
2026-11-11).

DART's round (open, step, add, close) and its predict have different bodies
in the GPU binding (`xtrees/dart_device.mojo`) and the host binding
(`xtrees/dart_host.mojo`), so each x_trees image publishes its own one-entry
DART table (`xtrees/dart_runs_device.mojo::x_trees_dart_device_v1`,
`xtrees/dart_runs_host.mojo::x_trees_dart_host_v1`) under the same tier/vendor
key as its glue table (`core/trees_entry_table.mojo`). Separate slot names, the
same `core/entry_table.mojo` publication and lookup. The ABI is
`xtrees/dart_abi.mojo`; the typed callers are `xtrees/dart_calls.mojo`.

Publication and lookup are small metadata operations with no independent
numerical work to split. A layout change takes a new V suffix.
"""
from core.entry_table import entry_table_publish, entry_table_read
from core.native_descriptor_abi import NativeDescriptorCall
from core.trees_entry_table import TreesEntryKey


def _check(key: TreesEntryKey) raises:
    if key.mode < 0 or key.mode > 1 or key.vendor < 0 or key.vendor > 3:
        raise Error("native x_trees DART entries: invalid tier/vendor key")

def trees_dart_table_publish(key: TreesEntryKey, publisher: Int,
    schemas: List[String], operations: List[Int], entries: List[NativeDescriptorCall]) raises:
    _check(key)
    if key.vendor == 0 and key.mode == 1:
        comptime NAME = "MojolearnEntriesXTreesDartIdenticalCPUV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 0 and key.mode == 0:
        comptime NAME = "MojolearnEntriesXTreesDartFastCPUV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 1 and key.mode == 1:
        comptime NAME = "MojolearnEntriesXTreesDartIdenticalMetalV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 1 and key.mode == 0:
        comptime NAME = "MojolearnEntriesXTreesDartFastMetalV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 2 and key.mode == 1:
        comptime NAME = "MojolearnEntriesXTreesDartIdenticalCUDAV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 2 and key.mode == 0:
        comptime NAME = "MojolearnEntriesXTreesDartFastCUDAV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 3 and key.mode == 1:
        comptime NAME = "MojolearnEntriesXTreesDartIdenticalHIPV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 3 and key.mode == 0:
        comptime NAME = "MojolearnEntriesXTreesDartFastHIPV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    raise Error("native x_trees DART entries: unsupported execution key")


def trees_dart_table_entry(key: TreesEntryKey, index: Int, schema: String, operations: Int) raises -> NativeDescriptorCall:
    _check(key)
    if key.vendor == 0 and key.mode == 1:
        comptime NAME = "MojolearnEntriesXTreesDartIdenticalCPUV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 0 and key.mode == 0:
        comptime NAME = "MojolearnEntriesXTreesDartFastCPUV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 1 and key.mode == 1:
        comptime NAME = "MojolearnEntriesXTreesDartIdenticalMetalV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 1 and key.mode == 0:
        comptime NAME = "MojolearnEntriesXTreesDartFastMetalV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 2 and key.mode == 1:
        comptime NAME = "MojolearnEntriesXTreesDartIdenticalCUDAV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 2 and key.mode == 0:
        comptime NAME = "MojolearnEntriesXTreesDartFastCUDAV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 3 and key.mode == 1:
        comptime NAME = "MojolearnEntriesXTreesDartIdenticalHIPV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 3 and key.mode == 0:
        comptime NAME = "MojolearnEntriesXTreesDartFastHIPV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    raise Error("native x_trees DART entries: unsupported execution key")
