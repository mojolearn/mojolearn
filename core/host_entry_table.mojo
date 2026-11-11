# SPDX-License-Identifier: Apache-2.0
"""Tier-qualified native tables of the host family (lane calltable-host-misc,
2026-11-11).

Modeled on `core/trees_entry_table.mojo`: the same `core/entry_table.mojo`
publication and lookup, separate slot names. Family 0 is the CPU saved-forest
binding (`_mojolearn_forest_host`); its one entry is
`core/host_entry_table_forest.mojo::forest_host_entry_v1`, a dispatch over the
`FH_*` operations below. The host family is CPU only, so the key's vendor word
is always 0; the mode word is the tier the binding reports as
`forest_host_numeric_mode` (the host archive bridge records the key as
`host_tree_native_key`). A caller calls the pointer with plain spans; no Python
object crosses.

Request layouts (a: addresses, p: Int words, f: Float64 words, the same
length as p). Each operation is the former Python export of the same name
after its arguments were unpacked: the same checks, kernels and refusals.
  FH_RF_PREDICT_PROBA   a [offsets, colid, quesval, left, leaves, x, out]
                        p [n_rows, n_cols, n_trees, num_outputs, n_nodes] -> rows
  FH_RF_PREDICT_REG     as FH_RF_PREDICT_PROBA                           -> rows
  FH_ET_PREDICT         as FH_RF_PREDICT_PROBA                           -> rows
  FH_GROVES_PREPARE     a [offsets, colid, quesval, left, leaves]
                        p [n_trees, n_cols, num_outputs, n_nodes, family] -> handle
  FH_GROVES_PREDICT     a [x, out] p [handle, n_rows, n_cols, num_outputs, family] -> rows
  FH_GBDT_EXPAND_CTR    a [10 addresses] p [11 words]                    -> columns
  FH_GBDT_PREDICT       a [15 addresses] p [8 or 9 words], f[0] the bias -> rows
  FH_GBDT_SIGMOID_PAIR  a [raw f64, out f64] p [n]                      -> n
The caller owns every buffer until the synchronous call returns. The grove
operations run with the GIL held by their caller (a handle is never released
under a running prediction); the others may run with it released.

Extending a table's layout requires a new V suffix, never a new loader.
Publication and lookup are small metadata operations with no independent
numerical work to split.
"""
from core.entry_table import entry_table_publish, entry_table_read
from core.native_descriptor_abi import NativeDescriptorCall

comptime HOST_TABLE_FOREST_HOST = 0

comptime FOREST_HOST_ABI = "mojolearn.host.forest_host.v1"

comptime FH_RF_PREDICT_PROBA = 0
comptime FH_RF_PREDICT_REG = 1
comptime FH_ET_PREDICT = 2
comptime FH_GROVES_PREPARE = 3
comptime FH_GROVES_PREDICT = 4
comptime FH_GBDT_EXPAND_CTR = 5
comptime FH_GBDT_PREDICT = 6
comptime FH_GBDT_SIGMOID_PAIR = 7
comptime FH_OP_COUNT = 8


@fieldwise_init
struct HostEntryKey(Copyable, Movable):
    var mode: Int
    var vendor: Int


def _host_entry_key_check(key: HostEntryKey) raises:
    if key.mode < 0 or key.mode > 1 or key.vendor != 0:
        raise Error("native host entries: invalid tier/vendor key")


def host_table_publish[family: Int](key: HostEntryKey, publisher: Int,
    schemas: List[String], operations: List[Int], entries: List[NativeDescriptorCall]) raises:
    comptime assert family == HOST_TABLE_FOREST_HOST, "unknown host table family"
    _host_entry_key_check(key)
    if key.mode == 1:
        comptime NAME = "MojolearnEntriesForestHostIdenticalCPUV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    comptime NAME = "MojolearnEntriesForestHostFastCPUV1"
    entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)


def host_table_entry[family: Int](key: HostEntryKey, index: Int,
    schema: String, operations: Int) raises -> NativeDescriptorCall:
    comptime assert family == HOST_TABLE_FOREST_HOST, "unknown host table family"
    _host_entry_key_check(key)
    if key.mode == 1:
        comptime NAME = "MojolearnEntriesForestHostIdenticalCPUV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    comptime NAME = "MojolearnEntriesForestHostFastCPUV1"
    return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
