# SPDX-License-Identifier: Apache-2.0
"""The gbdt binding table of the trees family (lane calltable-gbdt-forest-dart,
2026-11-11).

Each gbdt image (`_mojolearn_gbdt`, `_mojolearn_gbdt_host`) publishes one
entry (`gbdt_gpu_entry_v1` / `gbdt_host_entry_v1` in its binding file) under its tier/vendor key: the
fit, its outcome transfer, the row-major capability, the compiled tier and
the model dimension (ABI in `gbdt/fit_request.mojo`). The GBDT doors
(`bindings/trees_public_gbdt_fit.mojo`) read the slot of the image the
trees owner connected for the estimator's tier (the bridge records the key
as `gbdt_native_key`) and call the pointer with plain spans. Modeled on
`core/trees_entry_table.mojo`; separate slot names.

Publication and lookup are small metadata operations with no independent
numerical work to split. A layout change takes a new V suffix.
"""
from core.entry_table import entry_table_publish, entry_table_read
from core.native_descriptor_abi import NativeDescriptorCall
from core.trees_entry_table import TreesEntryKey


def _check(key: TreesEntryKey) raises:
    if key.mode < 0 or key.mode > 1 or key.vendor < 0 or key.vendor > 3:
        raise Error("native gbdt entries: invalid tier/vendor key")


def gbdt_vendor_word(vendor: StaticString) -> Int:
    """The key's vendor word of the vendor string a gbdt binding reports as
    `gbdt_vendor`: 0 cpu, 1 metal, 2 cuda, 3 hip, else -1 (the trees
    bridge applies the same mapping to the same string)."""
    if vendor == "cpu":
        return 0
    if vendor == "metal":
        return 1
    if vendor == "cuda":
        return 2
    if vendor == "hip":
        return 3
    return -1

def gbdt_table_publish(key: TreesEntryKey, publisher: Int,
    schemas: List[String], operations: List[Int], entries: List[NativeDescriptorCall]) raises:
    _check(key)
    if key.vendor == 0 and key.mode == 1:
        comptime NAME = "MojolearnEntriesGbdtIdenticalCPUV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 0 and key.mode == 0:
        comptime NAME = "MojolearnEntriesGbdtFastCPUV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 1 and key.mode == 1:
        comptime NAME = "MojolearnEntriesGbdtIdenticalMetalV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 1 and key.mode == 0:
        comptime NAME = "MojolearnEntriesGbdtFastMetalV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 2 and key.mode == 1:
        comptime NAME = "MojolearnEntriesGbdtIdenticalCUDAV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 2 and key.mode == 0:
        comptime NAME = "MojolearnEntriesGbdtFastCUDAV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 3 and key.mode == 1:
        comptime NAME = "MojolearnEntriesGbdtIdenticalHIPV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 3 and key.mode == 0:
        comptime NAME = "MojolearnEntriesGbdtFastHIPV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    raise Error("native gbdt entries: unsupported execution key")


def gbdt_table_entry(key: TreesEntryKey, index: Int, schema: String, operations: Int) raises -> NativeDescriptorCall:
    _check(key)
    if key.vendor == 0 and key.mode == 1:
        comptime NAME = "MojolearnEntriesGbdtIdenticalCPUV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 0 and key.mode == 0:
        comptime NAME = "MojolearnEntriesGbdtFastCPUV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 1 and key.mode == 1:
        comptime NAME = "MojolearnEntriesGbdtIdenticalMetalV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 1 and key.mode == 0:
        comptime NAME = "MojolearnEntriesGbdtFastMetalV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 2 and key.mode == 1:
        comptime NAME = "MojolearnEntriesGbdtIdenticalCUDAV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 2 and key.mode == 0:
        comptime NAME = "MojolearnEntriesGbdtFastCUDAV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 3 and key.mode == 1:
        comptime NAME = "MojolearnEntriesGbdtIdenticalHIPV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 3 and key.mode == 0:
        comptime NAME = "MojolearnEntriesGbdtFastHIPV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    raise Error("native gbdt entries: unsupported execution key")
