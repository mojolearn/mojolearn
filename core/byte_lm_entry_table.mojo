# SPDX-License-Identifier: Apache-2.0
"""Tier/vendor-qualified native tables of the byte LM trainer and the
sequence optimizers (lane calltable-byte-lm, 2026-11-11).

Modeled on `core/trees_entry_table.mojo`: the same `core/entry_table.mojo`
publication and lookup, separate slot names.

Family `BYTE_LM_TABLE_TRAINER` is published by both byte LM bindings
(`_mojolearn_byte_lm`, vendor COMPILED_VENDOR; `_mojolearn_byte_lm_host`,
vendor cpu). Entry 0 is the trainer dispatch the former Python capsule
(`byte_lm_trainer_api_v1`) carried (schema `mojolearn.training.byte_lm.v1`,
13 operations); entry 1 is the glue entry of `training/byte_lm_glue_abi.mojo`
(build metadata and call-scoped native sessions).

Family `BYTE_LM_TABLE_SEQ_OPT` is published by both x_sequence bindings
(`_mojolearn_x_sequence`, `_mojolearn_x_sequence_host`): entry 0 is the
sequence optimizer entry of `sequence/opt_entry_abi.mojo`.

The vendor words are the trees table's: 0 cpu, 1 metal, 2 cuda, 3 hip.
Extending a table's layout requires a new V suffix, never a new loader.
Publication and lookup are small metadata operations with no independent
numerical work to split.
"""
from core.entry_table import entry_table_publish, entry_table_read
from core.native_descriptor_abi import NativeDescriptorCall

comptime BYTE_LM_TABLE_TRAINER = 0
comptime BYTE_LM_TABLE_SEQ_OPT = 1


@fieldwise_init
struct ByteLmEntryKey(Copyable, Movable):
    var mode: Int
    var vendor: Int


def byte_lm_vendor_word(vendor: StaticString) -> Int:
    """The vendor word of a compiled vendor string: 0 cpu, 1 metal, 2 cuda,
    3 hip, else -1 (a build with no GPU vendor publishes nothing)."""
    if vendor == "cpu":
        return 0
    if vendor == "metal":
        return 1
    if vendor == "cuda":
        return 2
    if vendor == "hip":
        return 3
    return -1


def byte_lm_vendor_text_word(vendor: String) -> Int:
    """`byte_lm_vendor_word` for a runtime vendor name (the admitted vendor
    list of the trainer's import context)."""
    if vendor == "cpu":
        return 0
    if vendor == "metal":
        return 1
    if vendor == "cuda":
        return 2
    if vendor == "hip":
        return 3
    return -1


def _byte_lm_entry_key_check(key: ByteLmEntryKey) raises:
    if key.mode < 0 or key.mode > 1 or key.vendor < 0 or key.vendor > 3:
        raise Error("native byte LM entries: invalid tier/vendor key")



def byte_lm_table_publish[family: Int](key: ByteLmEntryKey, publisher: Int,
    schemas: List[String], operations: List[Int], entries: List[NativeDescriptorCall]) raises:
    comptime assert family == BYTE_LM_TABLE_TRAINER or family == BYTE_LM_TABLE_SEQ_OPT, "unknown byte LM table family"
    _byte_lm_entry_key_check(key)
    comptime if family == BYTE_LM_TABLE_TRAINER:
        if key.vendor == 0 and key.mode == 1:
            comptime NAME = "MojolearnEntriesByteLmIdenticalCPUV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
            return
        if key.vendor == 0 and key.mode == 0:
            comptime NAME = "MojolearnEntriesByteLmFastCPUV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
            return
        if key.vendor == 1 and key.mode == 1:
            comptime NAME = "MojolearnEntriesByteLmIdenticalMetalV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
            return
        if key.vendor == 1 and key.mode == 0:
            comptime NAME = "MojolearnEntriesByteLmFastMetalV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
            return
        if key.vendor == 2 and key.mode == 1:
            comptime NAME = "MojolearnEntriesByteLmIdenticalCUDAV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
            return
        if key.vendor == 2 and key.mode == 0:
            comptime NAME = "MojolearnEntriesByteLmFastCUDAV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
            return
        if key.vendor == 3 and key.mode == 1:
            comptime NAME = "MojolearnEntriesByteLmIdenticalHIPV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
            return
        if key.vendor == 3 and key.mode == 0:
            comptime NAME = "MojolearnEntriesByteLmFastHIPV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
            return
    comptime if family == BYTE_LM_TABLE_SEQ_OPT:
        if key.vendor == 0 and key.mode == 1:
            comptime NAME = "MojolearnEntriesSeqOptIdenticalCPUV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
            return
        if key.vendor == 0 and key.mode == 0:
            comptime NAME = "MojolearnEntriesSeqOptFastCPUV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
            return
        if key.vendor == 1 and key.mode == 1:
            comptime NAME = "MojolearnEntriesSeqOptIdenticalMetalV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
            return
        if key.vendor == 1 and key.mode == 0:
            comptime NAME = "MojolearnEntriesSeqOptFastMetalV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
            return
        if key.vendor == 2 and key.mode == 1:
            comptime NAME = "MojolearnEntriesSeqOptIdenticalCUDAV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
            return
        if key.vendor == 2 and key.mode == 0:
            comptime NAME = "MojolearnEntriesSeqOptFastCUDAV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
            return
        if key.vendor == 3 and key.mode == 1:
            comptime NAME = "MojolearnEntriesSeqOptIdenticalHIPV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
            return
        if key.vendor == 3 and key.mode == 0:
            comptime NAME = "MojolearnEntriesSeqOptFastHIPV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
            return
    raise Error("native byte LM entries: unsupported execution key")


def byte_lm_table_entry[family: Int](key: ByteLmEntryKey, index: Int,
    schema: String, operations: Int) raises -> NativeDescriptorCall:
    comptime assert family == BYTE_LM_TABLE_TRAINER or family == BYTE_LM_TABLE_SEQ_OPT, "unknown byte LM table family"
    _byte_lm_entry_key_check(key)
    comptime if family == BYTE_LM_TABLE_TRAINER:
        if key.vendor == 0 and key.mode == 1:
            comptime NAME = "MojolearnEntriesByteLmIdenticalCPUV1"
            return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
        if key.vendor == 0 and key.mode == 0:
            comptime NAME = "MojolearnEntriesByteLmFastCPUV1"
            return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
        if key.vendor == 1 and key.mode == 1:
            comptime NAME = "MojolearnEntriesByteLmIdenticalMetalV1"
            return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
        if key.vendor == 1 and key.mode == 0:
            comptime NAME = "MojolearnEntriesByteLmFastMetalV1"
            return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
        if key.vendor == 2 and key.mode == 1:
            comptime NAME = "MojolearnEntriesByteLmIdenticalCUDAV1"
            return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
        if key.vendor == 2 and key.mode == 0:
            comptime NAME = "MojolearnEntriesByteLmFastCUDAV1"
            return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
        if key.vendor == 3 and key.mode == 1:
            comptime NAME = "MojolearnEntriesByteLmIdenticalHIPV1"
            return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
        if key.vendor == 3 and key.mode == 0:
            comptime NAME = "MojolearnEntriesByteLmFastHIPV1"
            return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    comptime if family == BYTE_LM_TABLE_SEQ_OPT:
        if key.vendor == 0 and key.mode == 1:
            comptime NAME = "MojolearnEntriesSeqOptIdenticalCPUV1"
            return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
        if key.vendor == 0 and key.mode == 0:
            comptime NAME = "MojolearnEntriesSeqOptFastCPUV1"
            return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
        if key.vendor == 1 and key.mode == 1:
            comptime NAME = "MojolearnEntriesSeqOptIdenticalMetalV1"
            return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
        if key.vendor == 1 and key.mode == 0:
            comptime NAME = "MojolearnEntriesSeqOptFastMetalV1"
            return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
        if key.vendor == 2 and key.mode == 1:
            comptime NAME = "MojolearnEntriesSeqOptIdenticalCUDAV1"
            return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
        if key.vendor == 2 and key.mode == 0:
            comptime NAME = "MojolearnEntriesSeqOptFastCUDAV1"
            return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
        if key.vendor == 3 and key.mode == 1:
            comptime NAME = "MojolearnEntriesSeqOptIdenticalHIPV1"
            return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
        if key.vendor == 3 and key.mode == 0:
            comptime NAME = "MojolearnEntriesSeqOptFastHIPV1"
            return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    raise Error("native byte LM entries: unsupported execution key")



def byte_lm_table_published[family: Int](key: ByteLmEntryKey, index: Int,
    schema: String, operations: Int) -> Bool:
    """True when the table of `key` is published with that entry schema: the
    trainer picks the image of its admitted vendor list by probing (one
    acquire load and a schema compare per vendor word)."""
    try:
        _ = byte_lm_table_entry[family](key, index, schema, operations)
        return True
    except:
        return False
