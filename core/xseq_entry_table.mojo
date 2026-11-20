# SPDX-License-Identifier: Apache-2.0
"""Tier/vendor-qualified native table of the sequence-expansion family
(lane calltable-xseq, 2026-11-11).

Modeled on `core/trees_entry_table.mojo`: the same `core/entry_table.mojo`
publication and lookup under separate slot names. The publisher is the
`_mojolearn_x_sequence` image (device, `sequence/xseq_glue_device.mojo`) or
`_mojolearn_x_sequence_host` (CPU, `sequence/xseq_glue_host.mojo`), at module
init. The caller is the sequence owner (`bindings/xseq_*_native.mojo`), which
reads the slot of the image its estimator's mode connected (the bridge
records that image's `(mode, vendor)` as `xseq_native_keys`) and calls the
pointer with plain spans; no Python object crosses.

One entry per table: `XSEQ_GLUE_ABI`, `XS_OP_COUNT` operations, the
forecaster bodies below. (The recurrent/MLP doors still reach the image's
whole-model entry through `model_native_api_v1`'s capsule, shared with the
xcnn label encoder in `bindings/neural_labels_native.mojo`.)

Glue request layouts (a: addresses, p: Int words, f: Float64 words). The
words are exactly the former Python entry's `addrs`, `ip`, `fp` lists, in
the same order (`sequence/ops_impl.mojo` documents each `*_impl`):
  XS_THETA                  a [y, forecast, info]         p [B, n, h, m, code, additive, mask] f [3]
  XS_THETA_FORECAST         a [y, info, forecast]         p [B, n, h, m]
  XS_CROSTON                a [y, mean]                   p [B, n, variant]
  XS_CROSTON_FORECAST       a [mean, forecast]            p [B, h]
  XS_ETS                    a [y, forecast, info, states] p [9 or 10]  f [4 or 5]
  XS_GARCH                  a [y, params, info, sigma, forecast] p [7 or 9]
  XS_PROPHET_DAYS           a [src, days, info]           p [N, kind]
  XS_PROPHET_FEATURES       a [days, periods, t, frac]    p [N, ns]    f [start, t_scale]
  XS_PROPHET_CHANGEPOINTS   a [days, user, out]           p [N, n_user, hist, n_cp] f [start, t_scale]
  XS_PROPHET_FIT            prophet_fit_impl's lists
  XS_PROPHET_PREDICT        prophet_predict_impl's lists
  XS_PROPHET_PREDICT_DAYS   prophet_predict_days_impl's lists
  XS_VAR_FIT                a [y, params, sigma, resid]   p [n, k, p, trend]
  XS_VAR_FORECAST           a [history, params, out]      p [k, p, trend, steps]
  XS_STL                    a [y, season, trend, weights, resid] p [the STL plan words]
The result word is the body's Int result.

Extending a table's layout requires a new V suffix, never a new loader.
Publication and lookup are small metadata operations with no independent
numerical work to split.
"""
from core.entry_table import EntryRef, entry_ref_word, entry_table_publish, entry_table_read
from core.native_descriptor_abi import NativeDescriptorCall

comptime XSEQ_GLUE_ABI = "mojolearn.sequence.x_sequence.glue.v1"
comptime XS_THETA = 0
comptime XS_THETA_FORECAST = 1
comptime XS_CROSTON = 2
comptime XS_CROSTON_FORECAST = 3
comptime XS_ETS = 4
comptime XS_GARCH = 5
comptime XS_PROPHET_DAYS = 6
comptime XS_PROPHET_FEATURES = 7
comptime XS_PROPHET_CHANGEPOINTS = 8
comptime XS_PROPHET_FIT = 9
comptime XS_PROPHET_PREDICT = 10
comptime XS_PROPHET_PREDICT_DAYS = 11
comptime XS_VAR_FIT = 12
comptime XS_VAR_FORECAST = 13
comptime XS_STL = 14
comptime XS_OP_COUNT = 15

comptime XSEQ_ENTRY_GLUE = 0


@fieldwise_init
struct XseqEntryKey(Copyable, Movable):
    var mode: Int
    var vendor: Int


def xseq_vendor_word(vendor: StaticString) -> Int:
    """The table's vendor word of an x_sequence image's `x_sequence_vendor`
    string: 0 cpu, 1 metal, 2 cuda, 3 hip, else -1. The bridge applies the
    same mapping (python/mojolearn/_native_xseq_bridge.py `_NATIVE_VENDORS`)."""
    if vendor == "cpu":
        return 0
    if vendor == "metal":
        return 1
    if vendor == "cuda":
        return 2
    if vendor == "hip":
        return 3
    return -1


def _xseq_key_check(key: XseqEntryKey) raises:
    if key.mode < 0 or key.mode > 1 or key.vendor < 0 or key.vendor > 3:
        raise Error("native x_sequence entries: invalid tier/vendor key")


def xseq_table_publish(key: XseqEntryKey, publisher: Int, glue: NativeDescriptorCall) raises:
    _xseq_key_check(key)
    var schemas: List[String] = [String(XSEQ_GLUE_ABI)]
    var operations: List[Int] = [XS_OP_COUNT]
    var entries: List[NativeDescriptorCall] = [glue]
    if key.mode == 1 and key.vendor == 0:
        entry_table_publish["MojolearnEntriesXSequenceIdenticalCPUV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 0 and key.vendor == 0:
        entry_table_publish["MojolearnEntriesXSequenceFastCPUV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 1 and key.vendor == 1:
        entry_table_publish["MojolearnEntriesXSequenceIdenticalMetalV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 0 and key.vendor == 1:
        entry_table_publish["MojolearnEntriesXSequenceFastMetalV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 1 and key.vendor == 2:
        entry_table_publish["MojolearnEntriesXSequenceIdenticalCUDAV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 0 and key.vendor == 2:
        entry_table_publish["MojolearnEntriesXSequenceFastCUDAV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 1 and key.vendor == 3:
        entry_table_publish["MojolearnEntriesXSequenceIdenticalHIPV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 0 and key.vendor == 3:
        entry_table_publish["MojolearnEntriesXSequenceFastHIPV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    else:
        raise Error("native x_sequence entries: unsupported execution key")
    _ = schemas
    _ = operations
    _ = entries


def xseq_table_entry(key: XseqEntryKey, index: Int, schema: String, operations: Int) raises -> NativeDescriptorCall:
    _xseq_key_check(key)
    if key.mode == 1 and key.vendor == 0:
        return entry_table_read["MojolearnEntriesXSequenceIdenticalCPUV1"](key.mode, key.vendor, index, schema, operations)
    if key.mode == 0 and key.vendor == 0:
        return entry_table_read["MojolearnEntriesXSequenceFastCPUV1"](key.mode, key.vendor, index, schema, operations)
    if key.mode == 1 and key.vendor == 1:
        return entry_table_read["MojolearnEntriesXSequenceIdenticalMetalV1"](key.mode, key.vendor, index, schema, operations)
    if key.mode == 0 and key.vendor == 1:
        return entry_table_read["MojolearnEntriesXSequenceFastMetalV1"](key.mode, key.vendor, index, schema, operations)
    if key.mode == 1 and key.vendor == 2:
        return entry_table_read["MojolearnEntriesXSequenceIdenticalCUDAV1"](key.mode, key.vendor, index, schema, operations)
    if key.mode == 0 and key.vendor == 2:
        return entry_table_read["MojolearnEntriesXSequenceFastCUDAV1"](key.mode, key.vendor, index, schema, operations)
    if key.mode == 1 and key.vendor == 3:
        return entry_table_read["MojolearnEntriesXSequenceIdenticalHIPV1"](key.mode, key.vendor, index, schema, operations)
    if key.mode == 0 and key.vendor == 3:
        return entry_table_read["MojolearnEntriesXSequenceFastHIPV1"](key.mode, key.vendor, index, schema, operations)
    raise Error("native x_sequence entries: unsupported execution key")


def xseq_glue_entry(key: XseqEntryKey) raises -> NativeDescriptorCall:
    return xseq_table_entry(key, XSEQ_ENTRY_GLUE, String(XSEQ_GLUE_ABI), XS_OP_COUNT)


def xseq_glue_ref(key: XseqEntryKey) raises -> EntryRef:
    """The glue entry resolved once (lane entry-table-zero-alloc): the
    caller holds it for the operation; every glue call is one indirect call."""
    return EntryRef(xseq_glue_entry(key), entry_ref_word(key.mode, key.vendor, XS_OP_COUNT))


def xseq_glue_check_spans(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int, result: Int) raises:
    """Refuse a malformed request before any word is read; the per-operation
    word counts are each body's own checks (the former Python entries')."""
    if operation < 0 or operation >= XS_OP_COUNT:
        raise Error("x_sequence glue: unknown operation " + String(operation))
    if not result or na < 0 or np < 0 or nf < 0:
        raise Error("x_sequence glue: invalid request spans")
    if (na > 0 and not aa) or (np > 0 and not pp) or (nf > 0 and not ff):
        raise Error("x_sequence glue: null request span")


def xseq_glue_ints(address: Int, count: Int) -> List[Int]:
    """A request span as the typed list the bodies take (the former
    `py_ints` conversion); argument words, not data."""
    var p = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=address)
    var out = List[Int](capacity=max(count, 1))
    for i in range(count):  # small-loop(entry argument words, not data)
        out.append(p[i])
    return out^


def xseq_glue_floats(address: Int, count: Int) -> List[Float64]:
    """A request span as the typed Float64 list (the former `py_f64s`)."""
    var p = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=address)
    var out = List[Float64](capacity=max(count, 1))
    for i in range(count):  # small-loop(entry argument words, not data)
        out.append(p[i])
    return out^
