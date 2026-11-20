# SPDX-License-Identifier: Apache-2.0
"""The forest binding tables of the trees family (lane calltable-forest,
2026-11-11).

Each forest image publishes one entry under its tier/vendor key in the table
of its family: `FOREST_TABLE_RF` for `_mojolearn_rf` / `_mojolearn_rf_host`
and `FOREST_TABLE_ET` for `_mojolearn_trees` / `_mojolearn_trees_host`. The
entry is the dispatch of `core/forest_entry.mojo` over the operations of
`core/forest_fit_request.mojo` (fits, export transfer, sequential and
resident predict, the host strict snapshot, capability words). The forest
door (`bindings/trees_public_forest_door.mojo`) reads the slot of the image the
trees owner connected for the estimator's tier (the bridge records the key
as `rf_native_key` / `trees_native_key`) and calls the pointer with plain
spans. Modeled on `core/gbdt_entry_table.mojo`; separate slot names.

Publication and lookup are small metadata operations with no independent
numerical work to split. A layout change takes a new V suffix.
"""
from core.entry_table import entry_table_publish, entry_table_read
from core.native_descriptor_abi import NativeDescriptorCall
from core.trees_entry_table import TreesEntryKey

comptime FOREST_TABLE_RF = 0
comptime FOREST_TABLE_ET = 1


@fieldwise_init
struct ForestEntryKey(Copyable, Movable):
    """An image's table: its family (rf or et) and its tier/vendor key."""
    var family: Int
    var key: TreesEntryKey


def _check(key: ForestEntryKey) raises:
    if key.family != FOREST_TABLE_RF and key.family != FOREST_TABLE_ET:
        raise Error("native forest entries: invalid family")
    if key.key.mode < 0 or key.key.mode > 1 or key.key.vendor < 0 or key.key.vendor > 3:
        raise Error("native forest entries: invalid tier/vendor key")


def forest_table_publish(key: ForestEntryKey, publisher: Int,
    schemas: List[String], operations: List[Int], entries: List[NativeDescriptorCall]) raises:
    _check(key)
    var mode = key.key.mode
    var vendor = key.key.vendor
    if key.family == FOREST_TABLE_RF:
        if vendor == 0 and mode == 1:
            comptime NAME = "MojolearnEntriesForestRfIdenticalCPUV1"
            entry_table_publish[NAME](mode, vendor, publisher, schemas, operations, entries)
            return
        if vendor == 0 and mode == 0:
            comptime NAME = "MojolearnEntriesForestRfFastCPUV1"
            entry_table_publish[NAME](mode, vendor, publisher, schemas, operations, entries)
            return
        if vendor == 1 and mode == 1:
            comptime NAME = "MojolearnEntriesForestRfIdenticalMetalV1"
            entry_table_publish[NAME](mode, vendor, publisher, schemas, operations, entries)
            return
        if vendor == 1 and mode == 0:
            comptime NAME = "MojolearnEntriesForestRfFastMetalV1"
            entry_table_publish[NAME](mode, vendor, publisher, schemas, operations, entries)
            return
        if vendor == 2 and mode == 1:
            comptime NAME = "MojolearnEntriesForestRfIdenticalCUDAV1"
            entry_table_publish[NAME](mode, vendor, publisher, schemas, operations, entries)
            return
        if vendor == 2 and mode == 0:
            comptime NAME = "MojolearnEntriesForestRfFastCUDAV1"
            entry_table_publish[NAME](mode, vendor, publisher, schemas, operations, entries)
            return
        if vendor == 3 and mode == 1:
            comptime NAME = "MojolearnEntriesForestRfIdenticalHIPV1"
            entry_table_publish[NAME](mode, vendor, publisher, schemas, operations, entries)
            return
        if vendor == 3 and mode == 0:
            comptime NAME = "MojolearnEntriesForestRfFastHIPV1"
            entry_table_publish[NAME](mode, vendor, publisher, schemas, operations, entries)
            return
    if key.family == FOREST_TABLE_ET:
        if vendor == 0 and mode == 1:
            comptime NAME = "MojolearnEntriesForestEtIdenticalCPUV1"
            entry_table_publish[NAME](mode, vendor, publisher, schemas, operations, entries)
            return
        if vendor == 0 and mode == 0:
            comptime NAME = "MojolearnEntriesForestEtFastCPUV1"
            entry_table_publish[NAME](mode, vendor, publisher, schemas, operations, entries)
            return
        if vendor == 1 and mode == 1:
            comptime NAME = "MojolearnEntriesForestEtIdenticalMetalV1"
            entry_table_publish[NAME](mode, vendor, publisher, schemas, operations, entries)
            return
        if vendor == 1 and mode == 0:
            comptime NAME = "MojolearnEntriesForestEtFastMetalV1"
            entry_table_publish[NAME](mode, vendor, publisher, schemas, operations, entries)
            return
        if vendor == 2 and mode == 1:
            comptime NAME = "MojolearnEntriesForestEtIdenticalCUDAV1"
            entry_table_publish[NAME](mode, vendor, publisher, schemas, operations, entries)
            return
        if vendor == 2 and mode == 0:
            comptime NAME = "MojolearnEntriesForestEtFastCUDAV1"
            entry_table_publish[NAME](mode, vendor, publisher, schemas, operations, entries)
            return
        if vendor == 3 and mode == 1:
            comptime NAME = "MojolearnEntriesForestEtIdenticalHIPV1"
            entry_table_publish[NAME](mode, vendor, publisher, schemas, operations, entries)
            return
        if vendor == 3 and mode == 0:
            comptime NAME = "MojolearnEntriesForestEtFastHIPV1"
            entry_table_publish[NAME](mode, vendor, publisher, schemas, operations, entries)
            return
    raise Error("native forest entries: unsupported execution key")


def forest_table_entry(key: ForestEntryKey, index: Int, schema: String, operations: Int) raises -> NativeDescriptorCall:
    _check(key)
    var mode = key.key.mode
    var vendor = key.key.vendor
    if key.family == FOREST_TABLE_RF:
        if vendor == 0 and mode == 1:
            comptime NAME = "MojolearnEntriesForestRfIdenticalCPUV1"
            return entry_table_read[NAME](mode, vendor, index, schema, operations)
        if vendor == 0 and mode == 0:
            comptime NAME = "MojolearnEntriesForestRfFastCPUV1"
            return entry_table_read[NAME](mode, vendor, index, schema, operations)
        if vendor == 1 and mode == 1:
            comptime NAME = "MojolearnEntriesForestRfIdenticalMetalV1"
            return entry_table_read[NAME](mode, vendor, index, schema, operations)
        if vendor == 1 and mode == 0:
            comptime NAME = "MojolearnEntriesForestRfFastMetalV1"
            return entry_table_read[NAME](mode, vendor, index, schema, operations)
        if vendor == 2 and mode == 1:
            comptime NAME = "MojolearnEntriesForestRfIdenticalCUDAV1"
            return entry_table_read[NAME](mode, vendor, index, schema, operations)
        if vendor == 2 and mode == 0:
            comptime NAME = "MojolearnEntriesForestRfFastCUDAV1"
            return entry_table_read[NAME](mode, vendor, index, schema, operations)
        if vendor == 3 and mode == 1:
            comptime NAME = "MojolearnEntriesForestRfIdenticalHIPV1"
            return entry_table_read[NAME](mode, vendor, index, schema, operations)
        if vendor == 3 and mode == 0:
            comptime NAME = "MojolearnEntriesForestRfFastHIPV1"
            return entry_table_read[NAME](mode, vendor, index, schema, operations)
    if key.family == FOREST_TABLE_ET:
        if vendor == 0 and mode == 1:
            comptime NAME = "MojolearnEntriesForestEtIdenticalCPUV1"
            return entry_table_read[NAME](mode, vendor, index, schema, operations)
        if vendor == 0 and mode == 0:
            comptime NAME = "MojolearnEntriesForestEtFastCPUV1"
            return entry_table_read[NAME](mode, vendor, index, schema, operations)
        if vendor == 1 and mode == 1:
            comptime NAME = "MojolearnEntriesForestEtIdenticalMetalV1"
            return entry_table_read[NAME](mode, vendor, index, schema, operations)
        if vendor == 1 and mode == 0:
            comptime NAME = "MojolearnEntriesForestEtFastMetalV1"
            return entry_table_read[NAME](mode, vendor, index, schema, operations)
        if vendor == 2 and mode == 1:
            comptime NAME = "MojolearnEntriesForestEtIdenticalCUDAV1"
            return entry_table_read[NAME](mode, vendor, index, schema, operations)
        if vendor == 2 and mode == 0:
            comptime NAME = "MojolearnEntriesForestEtFastCUDAV1"
            return entry_table_read[NAME](mode, vendor, index, schema, operations)
        if vendor == 3 and mode == 1:
            comptime NAME = "MojolearnEntriesForestEtIdenticalHIPV1"
            return entry_table_read[NAME](mode, vendor, index, schema, operations)
        if vendor == 3 and mode == 0:
            comptime NAME = "MojolearnEntriesForestEtFastHIPV1"
            return entry_table_read[NAME](mode, vendor, index, schema, operations)
    raise Error("native forest entries: unsupported execution key")
