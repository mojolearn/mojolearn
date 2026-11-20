# SPDX-License-Identifier: Apache-2.0
"""Tier/vendor-qualified native tables of the archive writer's resident
exports (lane calltable-parallel-archive, 2026-11-11).

`bindings/archive_schema_native.mojo` saves a "device_array" step by copying
a device-resident index's array into the archive's host buffer. The resident
index lives in the image that minted its handle; that image publishes its
export here, and the archive writer calls the pointer instead of the
image's Python entry. Families:

  ARCHIVE_TABLE_IVF    `_mojolearn_ivf` (`bindings/parallel_ivf_entry_device.mojo`):
                       the IVF-Flat list data (`ivf_resident_export_list_data`).
  ARCHIVE_TABLE_X_ANN  `_mojolearn_x_ann` (`bindings/archive_x_ann_entry.mojo`):
                       the resident IVF-PQ codes (`x_ann_index_export_codes`).

One operation, AX_EXPORT: a [destination] p [handle, count] -> 0. The key of
the minting image is recorded once at load by
`python/mojolearn/_native_table_key.py`. Publication and lookup are small
metadata operations with no independent numerical work to split.
"""
from core.entry_table import EntryRef, entry_ref_call, entry_ref_word, entry_table_publish, entry_table_read
from core.native_descriptor_abi import NativeDescriptorCall
from core.parallel_entry_table import ParallelEntryKey

comptime ARCHIVE_TABLE_IVF = 0
comptime ARCHIVE_TABLE_X_ANN = 1

comptime ARCHIVE_EXPORT_ABI = "mojolearn.archive.resident_export.v1"
comptime AX_EXPORT = 0
comptime AX_OPERATION_COUNT = 1


def archive_export_family(entry_name: String) raises -> Int:
    """The family of a schema's export entry name (the step's own word)."""
    if entry_name == "ivf_flat_index_export":
        return ARCHIVE_TABLE_IVF
    if entry_name == "x_ann_index_export_codes":
        return ARCHIVE_TABLE_X_ANN
    return -1


def archive_check_spans(operation: Int, aa: Int, na: Int, pp: Int, np: Int, nf: Int, result: Int) raises:
    if not result:
        raise Error("native archive export: null result word")
    if operation != AX_EXPORT:
        raise Error("native archive export: unknown operation " + String(operation))
    if na != 1 or np != 2 or nf != 0 or not aa or not pp:
        raise Error("native archive export: request spans differ from operation 0")


def archive_table_publish(family: Int, key: ParallelEntryKey, publisher: Int,
    entry: NativeDescriptorCall) raises:
    if key.mode < 0 or key.mode > 1 or key.vendor < 0 or key.vendor > 3:
        raise Error("native archive entries: invalid tier/vendor key")
    var schemas: List[String] = [String(ARCHIVE_EXPORT_ABI)]
    var operations: List[Int] = [AX_OPERATION_COUNT]
    var entries: List[NativeDescriptorCall] = [entry]
    _archive_publish(family, key, publisher, schemas, operations, entries)
    _ = schemas
    _ = operations
    _ = entries


def _archive_publish(family: Int, key: ParallelEntryKey, publisher: Int,
    schemas: List[String], operations: List[Int], entries: List[NativeDescriptorCall]) raises:
    if family == ARCHIVE_TABLE_IVF:
        if key.vendor == 0 and key.mode == 1:
            entry_table_publish["MojolearnEntriesArchiveIvfExportIdenticalCPUV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
            return
        if key.vendor == 0 and key.mode == 0:
            entry_table_publish["MojolearnEntriesArchiveIvfExportFastCPUV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
            return
        if key.vendor == 1 and key.mode == 1:
            entry_table_publish["MojolearnEntriesArchiveIvfExportIdenticalMetalV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
            return
        if key.vendor == 1 and key.mode == 0:
            entry_table_publish["MojolearnEntriesArchiveIvfExportFastMetalV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
            return
        if key.vendor == 2 and key.mode == 1:
            entry_table_publish["MojolearnEntriesArchiveIvfExportIdenticalCUDAV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
            return
        if key.vendor == 2 and key.mode == 0:
            entry_table_publish["MojolearnEntriesArchiveIvfExportFastCUDAV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
            return
        if key.vendor == 3 and key.mode == 1:
            entry_table_publish["MojolearnEntriesArchiveIvfExportIdenticalHIPV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
            return
        if key.vendor == 3 and key.mode == 0:
            entry_table_publish["MojolearnEntriesArchiveIvfExportFastHIPV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
            return
    if family == ARCHIVE_TABLE_X_ANN:
        if key.vendor == 0 and key.mode == 1:
            entry_table_publish["MojolearnEntriesArchiveXAnnExportIdenticalCPUV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
            return
        if key.vendor == 0 and key.mode == 0:
            entry_table_publish["MojolearnEntriesArchiveXAnnExportFastCPUV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
            return
        if key.vendor == 1 and key.mode == 1:
            entry_table_publish["MojolearnEntriesArchiveXAnnExportIdenticalMetalV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
            return
        if key.vendor == 1 and key.mode == 0:
            entry_table_publish["MojolearnEntriesArchiveXAnnExportFastMetalV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
            return
        if key.vendor == 2 and key.mode == 1:
            entry_table_publish["MojolearnEntriesArchiveXAnnExportIdenticalCUDAV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
            return
        if key.vendor == 2 and key.mode == 0:
            entry_table_publish["MojolearnEntriesArchiveXAnnExportFastCUDAV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
            return
        if key.vendor == 3 and key.mode == 1:
            entry_table_publish["MojolearnEntriesArchiveXAnnExportIdenticalHIPV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
            return
        if key.vendor == 3 and key.mode == 0:
            entry_table_publish["MojolearnEntriesArchiveXAnnExportFastHIPV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
            return
    raise Error("native archive entries: unsupported execution key")


def archive_table_entry(family: Int, key: ParallelEntryKey) raises -> NativeDescriptorCall:
    if key.mode < 0 or key.mode > 1 or key.vendor < 0 or key.vendor > 3:
        raise Error("native archive entries: invalid tier/vendor key")
    var schema = String(ARCHIVE_EXPORT_ABI)
    var operations = AX_OPERATION_COUNT
    if family == ARCHIVE_TABLE_IVF:
        if key.vendor == 0 and key.mode == 1:
            return entry_table_read["MojolearnEntriesArchiveIvfExportIdenticalCPUV1"](key.mode, key.vendor, 0, schema, operations)
        if key.vendor == 0 and key.mode == 0:
            return entry_table_read["MojolearnEntriesArchiveIvfExportFastCPUV1"](key.mode, key.vendor, 0, schema, operations)
        if key.vendor == 1 and key.mode == 1:
            return entry_table_read["MojolearnEntriesArchiveIvfExportIdenticalMetalV1"](key.mode, key.vendor, 0, schema, operations)
        if key.vendor == 1 and key.mode == 0:
            return entry_table_read["MojolearnEntriesArchiveIvfExportFastMetalV1"](key.mode, key.vendor, 0, schema, operations)
        if key.vendor == 2 and key.mode == 1:
            return entry_table_read["MojolearnEntriesArchiveIvfExportIdenticalCUDAV1"](key.mode, key.vendor, 0, schema, operations)
        if key.vendor == 2 and key.mode == 0:
            return entry_table_read["MojolearnEntriesArchiveIvfExportFastCUDAV1"](key.mode, key.vendor, 0, schema, operations)
        if key.vendor == 3 and key.mode == 1:
            return entry_table_read["MojolearnEntriesArchiveIvfExportIdenticalHIPV1"](key.mode, key.vendor, 0, schema, operations)
        if key.vendor == 3 and key.mode == 0:
            return entry_table_read["MojolearnEntriesArchiveIvfExportFastHIPV1"](key.mode, key.vendor, 0, schema, operations)
    if family == ARCHIVE_TABLE_X_ANN:
        if key.vendor == 0 and key.mode == 1:
            return entry_table_read["MojolearnEntriesArchiveXAnnExportIdenticalCPUV1"](key.mode, key.vendor, 0, schema, operations)
        if key.vendor == 0 and key.mode == 0:
            return entry_table_read["MojolearnEntriesArchiveXAnnExportFastCPUV1"](key.mode, key.vendor, 0, schema, operations)
        if key.vendor == 1 and key.mode == 1:
            return entry_table_read["MojolearnEntriesArchiveXAnnExportIdenticalMetalV1"](key.mode, key.vendor, 0, schema, operations)
        if key.vendor == 1 and key.mode == 0:
            return entry_table_read["MojolearnEntriesArchiveXAnnExportFastMetalV1"](key.mode, key.vendor, 0, schema, operations)
        if key.vendor == 2 and key.mode == 1:
            return entry_table_read["MojolearnEntriesArchiveXAnnExportIdenticalCUDAV1"](key.mode, key.vendor, 0, schema, operations)
        if key.vendor == 2 and key.mode == 0:
            return entry_table_read["MojolearnEntriesArchiveXAnnExportFastCUDAV1"](key.mode, key.vendor, 0, schema, operations)
        if key.vendor == 3 and key.mode == 1:
            return entry_table_read["MojolearnEntriesArchiveXAnnExportIdenticalHIPV1"](key.mode, key.vendor, 0, schema, operations)
        if key.vendor == 3 and key.mode == 0:
            return entry_table_read["MojolearnEntriesArchiveXAnnExportFastHIPV1"](key.mode, key.vendor, 0, schema, operations)
    raise Error("native archive entries: unsupported execution key")


def archive_ref(family: Int, key: ParallelEntryKey) raises -> EntryRef:
    """The keyed image's archive export entry, resolved once (lane
    entry-table-zero-alloc); the table refusals raise here."""
    return EntryRef(archive_table_entry(family, key), entry_ref_word(key.mode, key.vendor, AX_OPERATION_COUNT))


def archive_resident_export(entry: EntryRef, handle: Int, dst: Int, count: Int) raises -> Int:
    """Copy `count` words of the resolved image's resident index `handle` into
    the host buffer at `dst` (kept alive by the caller): one indirect call."""
    var addrs: Array[Int, 1] = [dst]
    var ints: Array[Int, 2] = [handle, count]
    var floats: Array[Float64, 0] = []
    return entry_ref_call(entry, AX_EXPORT, addrs, ints, floats)
