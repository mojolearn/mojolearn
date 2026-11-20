# SPDX-License-Identifier: Apache-2.0
"""The base helper table: the `_mojolearn` base binding's index, count,
reduction and comparison helpers (the entries `_buffer._native(name)` hands
to the Python bridges) as one native entry, so a door in another extension
module calls them with plain words instead of through CPython (lane
calltable-base-helpers, 2026-11-11).

Modeled on `core/trees_entry_table.mojo`: the same `core/entry_table.mojo`
publication and lookup, separate slot names. Two publishers share the
layout: the GPU base binding (`bindings/_mojolearn.mojo`, its device twins
in `core/base_helpers_device.mojo`) and the core host binding
(`bindings/_mojolearn_core_host.mojo`, `core/base_helpers_host.mojo`). The
base binding is built in the identical tier only, so the key's mode is the
tier the binary reports and the vendor its compiled vendor (0 cpu, 1 metal,
2 cuda, 3 hip). A caller reads the key its bridge recorded at import
(`python/mojolearn/_buffer.py::base_native_key`) and calls the pointer.

Request layouts (a: addresses, p: Int words; no float words). The caller
owns every buffer until the synchronous call returns; nothing is retained.
  BH_CHECK_INDICES_I64    a [idx]              p [n, bound]            -> 0, 1 out of range, 2 duplicate
  BH_BINCOUNT_I64         a [src, counts]      p [code, n, k, accumulate] -> 0
  BH_REDUCE_STAT          a [src, out]         p [code, n, what]       -> HpStat kind;
                          out: five Int64 words [i, f bits, hi, mid, low]
  BH_EQUAL_ELEMENTS       a [a, b, dst]        p [code, n]             -> 0
  BH_BINCOUNT2_I32        a [a, b, counts]     p [n, ka, kb]           -> 0
  BH_ROW_MEANS_F64        a [src, out]         p [rows, cols]          -> rows holding +inf and -inf
  BH_ASSIGN_FOLD_I64      a [idx, folds]       p [m, n, fold]          -> 0, 1, 2
  BH_COUNT_FOLD_HITS_I64  a [idx, folds]       p [m, n, fold]          -> hits
  BH_MARK_SPLIT_ROLES_F32 a [idx, roles]       p [m, n, split, role]   -> 0, 1, 2
Each operation is the former Python export of the same name after its
arguments were unpacked: the same checks, kernels, words and refusals.

Extending the layout requires a new V suffix, never a new loader.
Publication and lookup are small metadata operations with no independent
numerical work to split.
"""
from core.entry_table import entry_table_publish, entry_table_read
from core.native_descriptor_abi import NativeDescriptorCall

comptime BASE_HELPERS_ABI = "mojolearn.base.helpers.v1"

comptime BH_CHECK_INDICES_I64 = 0
comptime BH_BINCOUNT_I64 = 1
comptime BH_REDUCE_STAT = 2
comptime BH_EQUAL_ELEMENTS = 3
comptime BH_BINCOUNT2_I32 = 4
comptime BH_ROW_MEANS_F64 = 5
comptime BH_ASSIGN_FOLD_I64 = 6
comptime BH_COUNT_FOLD_HITS_I64 = 7
comptime BH_MARK_SPLIT_ROLES_F32 = 8
comptime BH_OP_COUNT = 9

#: Words of the BH_REDUCE_STAT out buffer.
comptime BH_STAT_WORDS = 5


@fieldwise_init
struct BaseEntryKey(Copyable, Movable):
    var mode: Int
    var vendor: Int


def base_entry_vendor(vendor: StaticString) -> Int:
    """The key's vendor word for a compiled vendor string ("cpu" for the
    host binding, `COMPILED_VENDOR` for the GPU one): 0 cpu, 1 metal, 2 cuda,
    3 hip, else -1. `python/mojolearn/_buffer.py` applies the same mapping to
    the string the binary's `mojolearn_vendor` returns."""
    if vendor == "cpu":
        return 0
    if vendor == "metal":
        return 1
    if vendor == "cuda":
        return 2
    if vendor == "hip":
        return 3
    return -1


def _base_entry_key_check(key: BaseEntryKey) raises:
    if key.mode < 0 or key.mode > 1 or key.vendor < 0 or key.vendor > 3:
        raise Error("native base helpers: invalid tier/vendor key")


def base_table_publish(key: BaseEntryKey, publisher: Int,
    schemas: List[String], operations: List[Int], entries: List[NativeDescriptorCall]) raises:
    _base_entry_key_check(key)
    if key.vendor == 0 and key.mode == 1:
        comptime NAME = "MojolearnEntriesBaseHelpersIdenticalCPUV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 0 and key.mode == 0:
        comptime NAME = "MojolearnEntriesBaseHelpersFastCPUV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 1 and key.mode == 1:
        comptime NAME = "MojolearnEntriesBaseHelpersIdenticalMetalV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 1 and key.mode == 0:
        comptime NAME = "MojolearnEntriesBaseHelpersFastMetalV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 2 and key.mode == 1:
        comptime NAME = "MojolearnEntriesBaseHelpersIdenticalCUDAV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 2 and key.mode == 0:
        comptime NAME = "MojolearnEntriesBaseHelpersFastCUDAV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 3 and key.mode == 1:
        comptime NAME = "MojolearnEntriesBaseHelpersIdenticalHIPV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 3 and key.mode == 0:
        comptime NAME = "MojolearnEntriesBaseHelpersFastHIPV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    raise Error("native base helpers: unsupported execution key")


def base_table_entry(key: BaseEntryKey, index: Int,
    schema: String, operations: Int) raises -> NativeDescriptorCall:
    _base_entry_key_check(key)
    if key.vendor == 0 and key.mode == 1:
        comptime NAME = "MojolearnEntriesBaseHelpersIdenticalCPUV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 0 and key.mode == 0:
        comptime NAME = "MojolearnEntriesBaseHelpersFastCPUV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 1 and key.mode == 1:
        comptime NAME = "MojolearnEntriesBaseHelpersIdenticalMetalV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 1 and key.mode == 0:
        comptime NAME = "MojolearnEntriesBaseHelpersFastMetalV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 2 and key.mode == 1:
        comptime NAME = "MojolearnEntriesBaseHelpersIdenticalCUDAV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 2 and key.mode == 0:
        comptime NAME = "MojolearnEntriesBaseHelpersFastCUDAV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 3 and key.mode == 1:
        comptime NAME = "MojolearnEntriesBaseHelpersIdenticalHIPV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 3 and key.mode == 0:
        comptime NAME = "MojolearnEntriesBaseHelpersFastHIPV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    raise Error("native base helpers: unsupported execution key")


def base_check_spans(operation: Int, aa: Int, na: Int, pp: Int, np: Int, nf: Int, result: Int) raises:
    """Refuse a request whose spans do not match the operation's layout
    before any word is read."""
    if operation < 0 or operation >= BH_OP_COUNT:
        raise Error("base helpers: unknown operation " + String(operation))
    if not result:
        raise Error("base helpers: null result word")
    if na < 0 or np < 0 or nf != 0 or (na > 0 and not aa) or (np > 0 and not pp):
        raise Error("base helpers: null or invalid request span")
    var want_a = 2
    var want_p = 2
    if operation == BH_CHECK_INDICES_I64:
        want_a = 1
    elif operation == BH_BINCOUNT_I64:
        want_p = 4
    elif operation == BH_REDUCE_STAT:
        want_p = 3
    elif operation == BH_EQUAL_ELEMENTS:
        want_a = 3
    elif operation == BH_BINCOUNT2_I32:
        want_a = 3
        want_p = 3
    elif operation == BH_ASSIGN_FOLD_I64 or operation == BH_COUNT_FOLD_HITS_I64:
        want_p = 3
    elif operation == BH_MARK_SPLIT_ROLES_F32:
        want_p = 4
    if na != want_a or np != want_p:
        raise Error("base helpers: request spans do not match operation " + String(operation))


def base_word(span: Int, index: Int) -> Int:
    """Word `index` of an admitted request span."""
    return MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=span)[index]
