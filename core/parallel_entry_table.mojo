# SPDX-License-Identifier: Apache-2.0
"""Tier/vendor-qualified native tables of the parallel owners
(lane calltable-parallel-archive, 2026-11-11).

Modeled on `core/trees_entry_table.mojo`: the same `core/entry_table.mojo`
publication and lookup, separate slot names. Two families:

  PARALLEL_TABLE_BASE  published by the base image (`_mojolearn` on a GPU
                       vendor, `_mojolearn_core_host` on the CPU) from
                       `bindings/parallel_base_entry_{device,host}.mojo`;
                       called by `bindings/cross_vendor_native.mojo` (the
                       coordinator fold) and `bindings/parallel_shard_native.mojo`
                       (the forest offset merge and the reference top-k merge).
  PARALLEL_TABLE_IVF   published by the IVF image (`_mojolearn_ivf`,
                       `_mojolearn_ivf_host`) from
                       `bindings/parallel_ivf_entry_{device,host}.mojo`; called
                       by the distributed IVF index in
                       `bindings/parallel_shard_native.mojo`.

A caller reads the slot of the image Python connected (its key is recorded
once at import by `python/mojolearn/_native_table_key.py`) and calls the
pointer with plain addresses and words; no Python object crosses.

Request layouts (a: addresses, p: Int words, no Float64 words):
  PB_FOLD_PAIR_F32     a [dst, src]                 p [n]                      -> fold status
  PB_REBASE_OFFSETS    a [raw, lens, out]           p [n_raw, parts, n_out]    -> 0
  PB_TOPK_MERGE_F32    a [table, out_dist, out_idx] p [shards, queries, k]     -> 0, 1 or 2
  PI_SHARD_PLAN        a [ind, offs, soff, local, map]  p [n, n_lists, shards] -> 0, 1 or 2
  PI_MERGE_SHARDS      a [dist, ids, counts, (d, i, c, map) * shards]
                       p [shards, m, k, rows * shards]                         -> merge status
  PI_FINALIZE          a [distances]                p [count, metric]          -> 0

Extending a layout requires a new V suffix, never a new loader. Publication
and lookup are small metadata operations with no independent numerical work
to split.
"""
from core.entry_table import entry_table_publish, entry_table_read
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_call import native_descriptor_call

comptime PARALLEL_TABLE_BASE = 0
comptime PARALLEL_TABLE_IVF = 1

comptime PARALLEL_BASE_ABI = "mojolearn.parallel.base.v1"
comptime PB_FOLD_PAIR_F32 = 0
comptime PB_REBASE_OFFSETS = 1
comptime PB_TOPK_MERGE_F32 = 2
comptime PB_OPERATION_COUNT = 3

comptime PARALLEL_IVF_ABI = "mojolearn.parallel.ivf.v1"
comptime PI_SHARD_PLAN = 0
comptime PI_MERGE_SHARDS = 1
comptime PI_FINALIZE = 2
comptime PI_OPERATION_COUNT = 3


@fieldwise_init
struct ParallelEntryKey(Copyable, Movable):
    var mode: Int
    var vendor: Int


def parallel_entry_vendor(vendor: StaticString) -> Int:
    """The key's vendor word for an image's vendor read-back: 0 cpu,
    1 metal, 2 cuda, 3 hip, else -1 (a build with no GPU target publishes
    nothing). `python/mojolearn/_native_table_key.py` applies the same map."""
    if vendor == "cpu":
        return 0
    if vendor == "metal":
        return 1
    if vendor == "cuda":
        return 2
    if vendor == "hip":
        return 3
    return -1


def _parallel_key_check(key: ParallelEntryKey) raises:
    if key.mode < 0 or key.mode > 1 or key.vendor < 0 or key.vendor > 3:
        raise Error("native parallel entries: invalid tier/vendor key")


def parallel_check_spans(family: Int, operation: Int, aa: Int, na: Int, pp: Int, np: Int,
                         nf: Int, result: Int) raises:
    """Refuse a request whose spans do not match the operation's layout
    before any word is read."""
    if not result:
        raise Error("native parallel entry: null result word")
    if na < 0 or np < 0 or nf != 0 or (na > 0 and not aa) or (np > 0 and not pp):
        raise Error("native parallel entry: null or invalid request span")
    var want_a = -1
    var want_p = -1
    if family == PARALLEL_TABLE_BASE:
        if operation == PB_FOLD_PAIR_F32:
            want_a = 2
            want_p = 1
        elif operation == PB_REBASE_OFFSETS or operation == PB_TOPK_MERGE_F32:
            want_a = 3
            want_p = 3
    elif family == PARALLEL_TABLE_IVF:
        if operation == PI_SHARD_PLAN:
            want_a = 5
            want_p = 3
        elif operation == PI_MERGE_SHARDS:
            if np < 1:
                raise Error("native parallel entry: ivf merge header is short")
            var shards = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)[0]
            if shards < 1:
                raise Error("ivf_merge_shards: need at least one shard and k >= 1")
            want_a = 3 + 4 * shards
            want_p = 3 + shards
        elif operation == PI_FINALIZE:
            want_a = 1
            want_p = 2
    if want_a < 0:
        raise Error("native parallel entry: unknown operation " + String(operation))
    if na != want_a or np != want_p:
        raise Error("native parallel entry: request spans differ from operation " + String(operation))


def parallel_table_publish(family: Int, key: ParallelEntryKey, publisher: Int,
    entry: NativeDescriptorCall) raises:
    _parallel_key_check(key)
    var schemas: List[String] = [String(PARALLEL_BASE_ABI if family == PARALLEL_TABLE_BASE else PARALLEL_IVF_ABI)]
    var operations: List[Int] = [PB_OPERATION_COUNT if family == PARALLEL_TABLE_BASE else PI_OPERATION_COUNT]
    var entries: List[NativeDescriptorCall] = [entry]
    _parallel_publish(family, key, publisher, schemas, operations, entries)
    _ = schemas
    _ = operations
    _ = entries


def _parallel_publish(family: Int, key: ParallelEntryKey, publisher: Int,
    schemas: List[String], operations: List[Int], entries: List[NativeDescriptorCall]) raises:
    if family == PARALLEL_TABLE_BASE:
        if key.vendor == 0 and key.mode == 1:
            entry_table_publish["MojolearnEntriesParallelBaseIdenticalCPUV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
            return
        if key.vendor == 0 and key.mode == 0:
            entry_table_publish["MojolearnEntriesParallelBaseFastCPUV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
            return
        if key.vendor == 1 and key.mode == 1:
            entry_table_publish["MojolearnEntriesParallelBaseIdenticalMetalV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
            return
        if key.vendor == 1 and key.mode == 0:
            entry_table_publish["MojolearnEntriesParallelBaseFastMetalV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
            return
        if key.vendor == 2 and key.mode == 1:
            entry_table_publish["MojolearnEntriesParallelBaseIdenticalCUDAV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
            return
        if key.vendor == 2 and key.mode == 0:
            entry_table_publish["MojolearnEntriesParallelBaseFastCUDAV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
            return
        if key.vendor == 3 and key.mode == 1:
            entry_table_publish["MojolearnEntriesParallelBaseIdenticalHIPV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
            return
        if key.vendor == 3 and key.mode == 0:
            entry_table_publish["MojolearnEntriesParallelBaseFastHIPV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
            return
    if family == PARALLEL_TABLE_IVF:
        if key.vendor == 0 and key.mode == 1:
            entry_table_publish["MojolearnEntriesParallelIvfIdenticalCPUV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
            return
        if key.vendor == 0 and key.mode == 0:
            entry_table_publish["MojolearnEntriesParallelIvfFastCPUV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
            return
        if key.vendor == 1 and key.mode == 1:
            entry_table_publish["MojolearnEntriesParallelIvfIdenticalMetalV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
            return
        if key.vendor == 1 and key.mode == 0:
            entry_table_publish["MojolearnEntriesParallelIvfFastMetalV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
            return
        if key.vendor == 2 and key.mode == 1:
            entry_table_publish["MojolearnEntriesParallelIvfIdenticalCUDAV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
            return
        if key.vendor == 2 and key.mode == 0:
            entry_table_publish["MojolearnEntriesParallelIvfFastCUDAV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
            return
        if key.vendor == 3 and key.mode == 1:
            entry_table_publish["MojolearnEntriesParallelIvfIdenticalHIPV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
            return
        if key.vendor == 3 and key.mode == 0:
            entry_table_publish["MojolearnEntriesParallelIvfFastHIPV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
            return
    raise Error("native parallel entries: unsupported execution key")


def parallel_table_entry(family: Int, key: ParallelEntryKey) raises -> NativeDescriptorCall:
    _parallel_key_check(key)
    var schema = String(PARALLEL_BASE_ABI if family == PARALLEL_TABLE_BASE else PARALLEL_IVF_ABI)
    var operations = PB_OPERATION_COUNT if family == PARALLEL_TABLE_BASE else PI_OPERATION_COUNT
    if family == PARALLEL_TABLE_BASE:
        if key.vendor == 0 and key.mode == 1:
            return entry_table_read["MojolearnEntriesParallelBaseIdenticalCPUV1"](key.mode, key.vendor, 0, schema, operations)
        if key.vendor == 0 and key.mode == 0:
            return entry_table_read["MojolearnEntriesParallelBaseFastCPUV1"](key.mode, key.vendor, 0, schema, operations)
        if key.vendor == 1 and key.mode == 1:
            return entry_table_read["MojolearnEntriesParallelBaseIdenticalMetalV1"](key.mode, key.vendor, 0, schema, operations)
        if key.vendor == 1 and key.mode == 0:
            return entry_table_read["MojolearnEntriesParallelBaseFastMetalV1"](key.mode, key.vendor, 0, schema, operations)
        if key.vendor == 2 and key.mode == 1:
            return entry_table_read["MojolearnEntriesParallelBaseIdenticalCUDAV1"](key.mode, key.vendor, 0, schema, operations)
        if key.vendor == 2 and key.mode == 0:
            return entry_table_read["MojolearnEntriesParallelBaseFastCUDAV1"](key.mode, key.vendor, 0, schema, operations)
        if key.vendor == 3 and key.mode == 1:
            return entry_table_read["MojolearnEntriesParallelBaseIdenticalHIPV1"](key.mode, key.vendor, 0, schema, operations)
        if key.vendor == 3 and key.mode == 0:
            return entry_table_read["MojolearnEntriesParallelBaseFastHIPV1"](key.mode, key.vendor, 0, schema, operations)
    if family == PARALLEL_TABLE_IVF:
        if key.vendor == 0 and key.mode == 1:
            return entry_table_read["MojolearnEntriesParallelIvfIdenticalCPUV1"](key.mode, key.vendor, 0, schema, operations)
        if key.vendor == 0 and key.mode == 0:
            return entry_table_read["MojolearnEntriesParallelIvfFastCPUV1"](key.mode, key.vendor, 0, schema, operations)
        if key.vendor == 1 and key.mode == 1:
            return entry_table_read["MojolearnEntriesParallelIvfIdenticalMetalV1"](key.mode, key.vendor, 0, schema, operations)
        if key.vendor == 1 and key.mode == 0:
            return entry_table_read["MojolearnEntriesParallelIvfFastMetalV1"](key.mode, key.vendor, 0, schema, operations)
        if key.vendor == 2 and key.mode == 1:
            return entry_table_read["MojolearnEntriesParallelIvfIdenticalCUDAV1"](key.mode, key.vendor, 0, schema, operations)
        if key.vendor == 2 and key.mode == 0:
            return entry_table_read["MojolearnEntriesParallelIvfFastCUDAV1"](key.mode, key.vendor, 0, schema, operations)
        if key.vendor == 3 and key.mode == 1:
            return entry_table_read["MojolearnEntriesParallelIvfIdenticalHIPV1"](key.mode, key.vendor, 0, schema, operations)
        if key.vendor == 3 and key.mode == 0:
            return entry_table_read["MojolearnEntriesParallelIvfFastHIPV1"](key.mode, key.vendor, 0, schema, operations)
    raise Error("native parallel entries: unsupported execution key")


# ---------------------------------------------------------------- typed calls
# The callers' side: plain addresses and words in the documented layout, the
# published pointer of the keyed image, the provider's refusal text raised
# unchanged. The caller keeps every addressed buffer alive until return.


def parallel_fold_pair_f32(key: ParallelEntryKey, dst: Int, src: Int, count: Int) raises -> Int:
    var entry = parallel_table_entry(PARALLEL_TABLE_BASE, key)
    var addrs: List[Int] = [dst, src]
    var ints: List[Int] = [count]
    var floats = List[Float64]()
    return native_descriptor_call(entry, PB_FOLD_PAIR_F32, addrs, ints, floats)


def parallel_rebase_offsets_i32(key: ParallelEntryKey, raw: Int, n_raw: Int, lens: Int, parts: Int,
                                n_out: Int, dst: Int) raises -> Int:
    var entry = parallel_table_entry(PARALLEL_TABLE_BASE, key)
    var addrs: List[Int] = [raw, lens, dst]
    var ints: List[Int] = [n_raw, parts, n_out]
    var floats = List[Float64]()
    return native_descriptor_call(entry, PB_REBASE_OFFSETS, addrs, ints, floats)


def parallel_topk_merge_f32(key: ParallelEntryKey, table: Int, shards: Int, queries: Int, k: Int,
                            out_dist: Int, out_idx: Int) raises -> Int:
    var entry = parallel_table_entry(PARALLEL_TABLE_BASE, key)
    var addrs: List[Int] = [table, out_dist, out_idx]
    var ints: List[Int] = [shards, queries, k]
    var floats = List[Float64]()
    return native_descriptor_call(entry, PB_TOPK_MERGE_F32, addrs, ints, floats)


def parallel_ivf_shard_plan(key: ParallelEntryKey, ind: Int, offs: Int, soff: Int, local_ids: Int, id_map: Int,
                            n: Int, n_lists: Int, shards: Int) raises -> Int:
    var entry = parallel_table_entry(PARALLEL_TABLE_IVF, key)
    var addrs: List[Int] = [ind, offs, soff, local_ids, id_map]
    var ints: List[Int] = [n, n_lists, shards]
    var floats = List[Float64]()
    return native_descriptor_call(entry, PI_SHARD_PLAN, addrs, ints, floats)


def parallel_ivf_merge_shards(key: ParallelEntryKey, addrs: List[Int], ints: List[Int]) raises -> Int:
    """`addrs` and `ints` in the PI_MERGE_SHARDS layout; the merge status."""
    var entry = parallel_table_entry(PARALLEL_TABLE_IVF, key)
    var floats = List[Float64]()
    return native_descriptor_call(entry, PI_MERGE_SHARDS, addrs, ints, floats)


def parallel_ivf_finalize(key: ParallelEntryKey, distances: Int, count: Int, metric: Int) raises -> Int:
    var entry = parallel_table_entry(PARALLEL_TABLE_IVF, key)
    var addrs: List[Int] = [distances]
    var ints: List[Int] = [count, metric]
    var floats = List[Float64]()
    return native_descriptor_call(entry, PI_FINALIZE, addrs, ints, floats)
