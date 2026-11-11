# SPDX-License-Identifier: Apache-2.0
"""Tier/vendor-qualified native table of the metrics binding's R^2 entries
(lane calltable-xseq, 2026-11-11).

Modeled on `core/trees_entry_table.mojo`: the same `core/entry_table.mojo`
publication and lookup under separate slot names. `_mojolearn_metrics`
(device vendor) and `_mojolearn_metrics_host` (CPU) publish one entry,
`METRICS_R2_ABI`, at module init; the regressor scores of the sequence owner
(`bindings/xseq_score_native.mojo`) call it with plain spans instead of the
Python `r2_score` / `r2_score_weighted` entries, which stay for the public
`mojolearn.metrics` functions (`python/mojolearn/_metrics_impl.py`).

Request layouts (a: addresses, p: Int words; no float words):
  MR_R2           a [y_true f32 (n), y_pred f32 (n), out f64 (1)]            p [n]
  MR_R2_WEIGHTED  a [y_true f32 (n), y_pred f32 (n), w f32 (n), out f64 (1)] p [n]
`out` receives `Float64(r2)` (the Python entries' return value); the result
word is 0. Publication and lookup are small metadata operations with no
independent numerical work to split.
"""
from core.entry_table import entry_table_publish, entry_table_read
from core.native_descriptor_abi import NativeDescriptorCall

comptime METRICS_R2_ABI = "mojolearn.metrics.r2.v1"
comptime MR_R2 = 0
comptime MR_R2_WEIGHTED = 1
comptime MR_OP_COUNT = 2


@fieldwise_init
struct MetricsEntryKey(Copyable, Movable):
    var mode: Int
    var vendor: Int


def metrics_vendor_word(vendor: StaticString) -> Int:
    """0 cpu, 1 metal, 2 cuda, 3 hip, else -1 (the bridge's mapping)."""
    if vendor == "cpu":
        return 0
    if vendor == "metal":
        return 1
    if vendor == "cuda":
        return 2
    if vendor == "hip":
        return 3
    return -1


def metrics_r2_check_spans(operation: Int, aa: Int, na: Int, pp: Int, np: Int, nf: Int, result: Int) raises -> Int:
    """Refuse a malformed request; returns n."""
    var want = 3 if operation == MR_R2 else (4 if operation == MR_R2_WEIGHTED else -1)
    if want < 0:
        raise Error("metrics r2: unknown operation " + String(operation))
    if not result or na != want or np != 1 or nf != 0 or not aa or not pp:
        raise Error("metrics r2: invalid request spans")
    var a = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)
    for i in range(want):  # small-loop(request addresses, not data)
        if not a[i]:
            raise Error("metrics r2: null buffer address")
    return MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)[0]


def _metrics_key_check(key: MetricsEntryKey) raises:
    if key.mode < 0 or key.mode > 1 or key.vendor < 0 or key.vendor > 3:
        raise Error("native metrics entries: invalid tier/vendor key")


def metrics_r2_publish(key: MetricsEntryKey, publisher: Int, entry: NativeDescriptorCall) raises:
    _metrics_key_check(key)
    var schemas: List[String] = [String(METRICS_R2_ABI)]
    var operations: List[Int] = [MR_OP_COUNT]
    var entries: List[NativeDescriptorCall] = [entry]
    if key.mode == 1 and key.vendor == 0:
        entry_table_publish["MojolearnEntriesMetricsR2IdenticalCPUV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 0 and key.vendor == 0:
        entry_table_publish["MojolearnEntriesMetricsR2FastCPUV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 1 and key.vendor == 1:
        entry_table_publish["MojolearnEntriesMetricsR2IdenticalMetalV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 0 and key.vendor == 1:
        entry_table_publish["MojolearnEntriesMetricsR2FastMetalV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 1 and key.vendor == 2:
        entry_table_publish["MojolearnEntriesMetricsR2IdenticalCUDAV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 0 and key.vendor == 2:
        entry_table_publish["MojolearnEntriesMetricsR2FastCUDAV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 1 and key.vendor == 3:
        entry_table_publish["MojolearnEntriesMetricsR2IdenticalHIPV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 0 and key.vendor == 3:
        entry_table_publish["MojolearnEntriesMetricsR2FastHIPV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    else:
        raise Error("native metrics entries: unsupported execution key")
    _ = schemas
    _ = operations
    _ = entries


def metrics_r2_entry(key: MetricsEntryKey) raises -> NativeDescriptorCall:
    _metrics_key_check(key)
    var schema = String(METRICS_R2_ABI)
    if key.mode == 1 and key.vendor == 0:
        return entry_table_read["MojolearnEntriesMetricsR2IdenticalCPUV1"](key.mode, key.vendor, 0, schema, MR_OP_COUNT)
    if key.mode == 0 and key.vendor == 0:
        return entry_table_read["MojolearnEntriesMetricsR2FastCPUV1"](key.mode, key.vendor, 0, schema, MR_OP_COUNT)
    if key.mode == 1 and key.vendor == 1:
        return entry_table_read["MojolearnEntriesMetricsR2IdenticalMetalV1"](key.mode, key.vendor, 0, schema, MR_OP_COUNT)
    if key.mode == 0 and key.vendor == 1:
        return entry_table_read["MojolearnEntriesMetricsR2FastMetalV1"](key.mode, key.vendor, 0, schema, MR_OP_COUNT)
    if key.mode == 1 and key.vendor == 2:
        return entry_table_read["MojolearnEntriesMetricsR2IdenticalCUDAV1"](key.mode, key.vendor, 0, schema, MR_OP_COUNT)
    if key.mode == 0 and key.vendor == 2:
        return entry_table_read["MojolearnEntriesMetricsR2FastCUDAV1"](key.mode, key.vendor, 0, schema, MR_OP_COUNT)
    if key.mode == 1 and key.vendor == 3:
        return entry_table_read["MojolearnEntriesMetricsR2IdenticalHIPV1"](key.mode, key.vendor, 0, schema, MR_OP_COUNT)
    if key.mode == 0 and key.vendor == 3:
        return entry_table_read["MojolearnEntriesMetricsR2FastHIPV1"](key.mode, key.vendor, 0, schema, MR_OP_COUNT)
    raise Error("native metrics entries: unsupported execution key")


struct MetricsR2Owner(Defaultable, Movable):
    """The image-lifetime publisher word of a metrics R^2 table."""

    var word: Int

    def __init__(out self):
        self.word = 1
