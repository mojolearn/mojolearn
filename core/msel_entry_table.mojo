# SPDX-License-Identifier: Apache-2.0
"""The model-selection helper table: the fold, index, gather and fold-store
helpers the base bindings export for the native model-selection doors
(`bindings/msel_*.mojo`, compiled into `_mojolearn_runtime_model_selection_host`),
as one native entry per base binding, so a door calls them with plain words
instead of through CPython (lane calltable-msel, 2026-11-11).

Modeled on `core/trees_entry_table.mojo`: the same `core/entry_table.mojo`
publication and lookup, separate slot names. Two publishers share the
layout: the GPU base binding (`bindings/_mojolearn.mojo`, its device runs in
`core/msel_helpers_device.mojo`) and the core host binding
(`bindings/_mojolearn_core_host.mojo`, `core/msel_helpers_host.mojo`). The
base binding is built in the identical tier only, so the key's mode is the
tier the binary reports and the vendor its compiled vendor (0 cpu, 1 metal,
2 cuda, 3 hip). A door reads the key the model-selection bridge recorded at
import (`python/mojolearn/_native_msel_inputs.py`, `msel_native_key`) and
calls the pointer through `core/msel_helpers_calls.mojo`.

Request layouts (a: addresses, p: Int words; no float words). The caller
owns every buffer until the synchronous call returns; nothing is retained
except the fold-store blocks, which live in the publisher's process store
until MH_STORE_FREE. A zero address is legal where the former export took
one (an empty side, no codes, no permutation).
  MH_ARANGE_I64            a [dst]                        p [start, n]           -> 0
  MH_COUNT_MASK_U8         a [mask]                       p [n]                  -> nonzero bytes
  MH_SELECT_MASK_U8_I64    a [mask, test, train]          p [n]                  -> test count
  MH_MASK_FROM_INDICES_U8  a [idx, mask]                  p [k, n]               -> rows set
  MH_LEAVE_RANGE_I64       a [train, test]                p [n, lo, hi]          -> hi - lo
  MH_INDICES_OVERLAP_I64   a [a, b]                       p [na, nb, bound]      -> 0 or 1
  MH_FOLD_IDS              a [codes, counts, folds, fold_counts] p [n, k, n_splits] -> 0
  MH_SELECT_FOLD_I64       a [folds, test, train]         p [n, fold]            -> test count
  MH_SELECT_FOLDS_I64      a [folds, sizes, tests, trains] p [n, k]              -> -1 or first differing fold
  MH_GATHER_I32            a [table, codes, dst]          p [n_table, n]         -> 0
  MH_GATHER_I64            a [table, codes, dst]          p [n_table, n]         -> 0
  MH_FIRST_SEEN_I32        a [codes, enc, counts]         p [n, k]               -> classes present
  MH_STRAT_FOLD_ASSIGN_I32 a [enc, alloc, perms, counts, dst] p [n, k, n_folds]  -> 0
  MH_STORE_ALLOC           a []                           p [nbytes]             -> block id
  MH_STORE_READ            a [dst]                        p [id, nbytes]         -> 0
  MH_STORE_FREE            a []                           p [id]                 -> 0
  MH_SCATTER_ROWS          a [src]                        p [dst_id, n_dst, dst_code, src_code, n_src, width, idx_id] -> 0
  MH_PROBA_COLUMN          a [src, dst]                   p [code, n, k, col]    -> 0
  MH_SPLIT_TABLE_I32       a [perm, counts, table, sums]  p [m, n_test, n_train] -> 0
  MH_GROUP_FOLD_PERM_I32   a [counts, perm, dst, sizes]   p [m, n_folds]         -> 0
Each operation is the former Python export of the same name after its
arguments were unpacked: the same checks, kernels, words and refusals.

Extending the layout requires a new V suffix, never a new loader.
Publication and lookup are small metadata operations with no independent
numerical work to split.
"""
from core.entry_table import entry_table_publish, entry_table_read
from core.native_descriptor_abi import NativeDescriptorCall

comptime MSEL_HELPERS_ABI = "mojolearn.msel.helpers.v1"

comptime MH_ARANGE_I64 = 0
comptime MH_COUNT_MASK_U8 = 1
comptime MH_SELECT_MASK_U8_I64 = 2
comptime MH_MASK_FROM_INDICES_U8 = 3
comptime MH_LEAVE_RANGE_I64 = 4
comptime MH_INDICES_OVERLAP_I64 = 5
comptime MH_FOLD_IDS = 6
comptime MH_SELECT_FOLD_I64 = 7
comptime MH_SELECT_FOLDS_I64 = 8
comptime MH_GATHER_I32 = 9
comptime MH_GATHER_I64 = 10
comptime MH_FIRST_SEEN_I32 = 11
comptime MH_STRAT_FOLD_ASSIGN_I32 = 12
comptime MH_STORE_ALLOC = 13
comptime MH_STORE_READ = 14
comptime MH_STORE_FREE = 15
comptime MH_SCATTER_ROWS = 16
comptime MH_PROBA_COLUMN = 17
comptime MH_SPLIT_TABLE_I32 = 18
comptime MH_GROUP_FOLD_PERM_I32 = 19
comptime MH_OP_COUNT = 20


@fieldwise_init
struct MselEntryKey(Copyable, Movable):
    var mode: Int
    var vendor: Int


def msel_entry_vendor(vendor: StaticString) -> Int:
    """The key's vendor word for a compiled vendor string ("cpu" for the
    host binding, `COMPILED_VENDOR` for the GPU one): 0 cpu, 1 metal, 2 cuda,
    3 hip, else -1. `python/mojolearn/_native_msel_inputs.py` applies the
    same mapping to the string the binary's `mojolearn_vendor` returns."""
    if vendor == "cpu":
        return 0
    if vendor == "metal":
        return 1
    if vendor == "cuda":
        return 2
    if vendor == "hip":
        return 3
    return -1


def _msel_entry_key_check(key: MselEntryKey) raises:
    if key.mode < 0 or key.mode > 1 or key.vendor < 0 or key.vendor > 3:
        raise Error("native model-selection helpers: invalid tier/vendor key")


def msel_table_publish(key: MselEntryKey, publisher: Int,
    schemas: List[String], operations: List[Int], entries: List[NativeDescriptorCall]) raises:
    _msel_entry_key_check(key)
    if key.vendor == 0 and key.mode == 1:
        comptime NAME = "MojolearnEntriesMselHelpersIdenticalCPUV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 0 and key.mode == 0:
        comptime NAME = "MojolearnEntriesMselHelpersFastCPUV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 1 and key.mode == 1:
        comptime NAME = "MojolearnEntriesMselHelpersIdenticalMetalV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 1 and key.mode == 0:
        comptime NAME = "MojolearnEntriesMselHelpersFastMetalV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 2 and key.mode == 1:
        comptime NAME = "MojolearnEntriesMselHelpersIdenticalCUDAV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 2 and key.mode == 0:
        comptime NAME = "MojolearnEntriesMselHelpersFastCUDAV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 3 and key.mode == 1:
        comptime NAME = "MojolearnEntriesMselHelpersIdenticalHIPV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 3 and key.mode == 0:
        comptime NAME = "MojolearnEntriesMselHelpersFastHIPV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    raise Error("native model-selection helpers: unsupported execution key")


def msel_table_entry(key: MselEntryKey, index: Int,
    schema: String, operations: Int) raises -> NativeDescriptorCall:
    _msel_entry_key_check(key)
    if key.vendor == 0 and key.mode == 1:
        comptime NAME = "MojolearnEntriesMselHelpersIdenticalCPUV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 0 and key.mode == 0:
        comptime NAME = "MojolearnEntriesMselHelpersFastCPUV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 1 and key.mode == 1:
        comptime NAME = "MojolearnEntriesMselHelpersIdenticalMetalV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 1 and key.mode == 0:
        comptime NAME = "MojolearnEntriesMselHelpersFastMetalV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 2 and key.mode == 1:
        comptime NAME = "MojolearnEntriesMselHelpersIdenticalCUDAV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 2 and key.mode == 0:
        comptime NAME = "MojolearnEntriesMselHelpersFastCUDAV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 3 and key.mode == 1:
        comptime NAME = "MojolearnEntriesMselHelpersIdenticalHIPV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 3 and key.mode == 0:
        comptime NAME = "MojolearnEntriesMselHelpersFastHIPV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    raise Error("native model-selection helpers: unsupported execution key")


def _msel_span_shape(operation: Int) -> Tuple[Int, Int]:
    """(addresses, Int words) of an operation's request."""
    if operation == MH_ARANGE_I64:
        return (1, 2)
    if operation == MH_COUNT_MASK_U8:
        return (1, 1)
    if operation == MH_SELECT_MASK_U8_I64:
        return (3, 1)
    if operation == MH_MASK_FROM_INDICES_U8:
        return (2, 2)
    if operation == MH_LEAVE_RANGE_I64:
        return (2, 3)
    if operation == MH_INDICES_OVERLAP_I64:
        return (2, 3)
    if operation == MH_FOLD_IDS:
        return (4, 3)
    if operation == MH_SELECT_FOLD_I64:
        return (3, 2)
    if operation == MH_SELECT_FOLDS_I64:
        return (4, 2)
    if operation == MH_GATHER_I32 or operation == MH_GATHER_I64:
        return (3, 2)
    if operation == MH_FIRST_SEEN_I32:
        return (3, 2)
    if operation == MH_STRAT_FOLD_ASSIGN_I32:
        return (5, 3)
    if operation == MH_STORE_ALLOC or operation == MH_STORE_FREE:
        return (0, 1)
    if operation == MH_STORE_READ:
        return (1, 2)
    if operation == MH_SCATTER_ROWS:
        return (1, 7)
    if operation == MH_PROBA_COLUMN:
        return (2, 4)
    if operation == MH_SPLIT_TABLE_I32:
        return (4, 3)
    return (4, 2)  # MH_GROUP_FOLD_PERM_I32


def msel_check_spans(operation: Int, aa: Int, na: Int, pp: Int, np: Int, nf: Int, result: Int) raises:
    """Refuse a request whose spans do not match the operation's layout
    before any word is read."""
    if operation < 0 or operation >= MH_OP_COUNT:
        raise Error("model-selection helpers: unknown operation " + String(operation))
    if not result:
        raise Error("model-selection helpers: null result word")
    if na < 0 or np < 0 or nf != 0 or (na > 0 and not aa) or (np > 0 and not pp):
        raise Error("model-selection helpers: null or invalid request span")
    var shape = _msel_span_shape(operation)
    if na != shape[0] or np != shape[1]:
        raise Error("model-selection helpers: request spans do not match operation " + String(operation))


def msel_word(span: Int, index: Int) -> Int:
    """Word `index` of an admitted request span."""
    return MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=span)[index]
