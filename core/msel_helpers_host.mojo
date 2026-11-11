# SPDX-License-Identifier: Apache-2.0
"""The host column of the model-selection helper table
(core/msel_entry_table.mojo): the core host binding's
(`bindings/_mojolearn_core_host.mojo`) model-selection helpers as typed runs,
its one `abi("C")` entry, and its publication (lane calltable-msel,
2026-11-11).

Each run is the body the host Python export of the same name ran
(`bindings/hotpath_helpers.mojo`, `bindings/msel_host.mojo`, the label
gathers of `core/labels_native_host.mojo`) after its arguments were
unpacked: the same core function, checks and refusals. The GPU base
binding's runs (`core/msel_helpers_device.mojo`) fall back to these wherever
their device twin does not cover a call, as the device Python exports fell
back to the host exports. No Python object crosses.

The dispatch is a fixed comparison chain on the operation code, no data
work; each run's own parallelism is its core function's.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.msel_entry_table import (
    MselEntryKey, msel_table_publish, msel_check_spans, msel_word, msel_entry_vendor,
    MSEL_HELPERS_ABI, MH_OP_COUNT,
    MH_ARANGE_I64, MH_COUNT_MASK_U8, MH_SELECT_MASK_U8_I64, MH_MASK_FROM_INDICES_U8, MH_LEAVE_RANGE_I64,
    MH_INDICES_OVERLAP_I64, MH_FOLD_IDS, MH_SELECT_FOLD_I64, MH_SELECT_FOLDS_I64, MH_GATHER_I32, MH_GATHER_I64,
    MH_FIRST_SEEN_I32, MH_STRAT_FOLD_ASSIGN_I32, MH_STORE_ALLOC, MH_STORE_READ, MH_STORE_FREE, MH_SCATTER_ROWS,
    MH_PROBA_COLUMN, MH_SPLIT_TABLE_I32, MH_GROUP_FOLD_PERM_I32, MH_STORE_FREE_BYTES,
)
from core.hotpath_impl import (
    arange_i64, count_mask_u8, fold_ids, indices_overlap_i64, leave_range_i64, mask_from_indices_u8,
    select_fold_i64, select_mask_u8_i64,
)
from core.hotpath_impl_rows import first_seen_i32, strat_fold_assign_i32
from core.labels_native_api import LabelsGatherRequest
from core.labels_native_host import LabelsHostKernels
from core.msel_convert import MSEL_F32, MSEL_F64, MSEL_I64, msel_is_int, msel_valid_code
from core.msel_host import (
    MselHostStore, host_group_fold_perm_i32, host_proba_column, host_scatter_rows, host_split_table_i32,
)
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot

#: The host fold store of the core host binding: one process-wide slot, read
#: by its Python exports (`bindings/msel_host.mojo`) and by the table's runs.
comptime MSEL_HOST_STORE = SharedSlot[MselHostStore, "MojolearnMselHostStoreV1"]


def select_folds_i64_host_run(fa: Int, rows: Int, splits: Int, sa: Int, ta: Int, ra: Int) raises -> Int:
    """`select_fold_i64` of every fold 0..k-1 in one call (lane ne1-b-door,
    N051): fold f's rows into the int64 arrays whose addresses are the int64
    words `tests[f]` and `trains[f]`. Returns -1 when every fold's test
    count equals the int64 `sizes[f]`, else the first fold that differs."""
    var sizes = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=sa)
    var tests = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=ta)
    var trains = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=ra)
    var bad = -1
    for f in range(splits):  # driver-loop: over folds of one split plan
        var got = select_fold_i64(fa, rows, f, Int(tests.unsafe_load(f)), Int(trains.unsafe_load(f)))
        if bad < 0 and got != Int(sizes.unsafe_load(f)):
            bad = f
    return bad


def msel_gather_host_run(kind: Int, table: Int, classes: Int, codes: Int, count: Int, dst: Int) raises -> Int:
    """`gather_i32` (kind 0) / `gather_i64` (kind 1): dst[i] = table[codes[i]];
    a code outside [0, n_table) raises before any write."""
    LabelsHostKernels.gather(kind, LabelsGatherRequest(table, codes, dst, classes, count))
    return 0


def msel_scatter_rows_check(nd: Int, dc: Int, sa: Int, sc: Int, ns: Int, w: Int) raises:
    """The admission of `msel_scatter_rows` (both columns)."""
    if nd < 0 or ns < 0 or w < 0:
        raise Error("msel_scatter_rows: dimensions must be non-negative")
    if dc != MSEL_I64 and dc != MSEL_F64:
        raise Error("msel_scatter_rows: the block is int64 or float64")
    if not msel_valid_code(sc) or (dc == MSEL_I64 and not msel_is_int(sc)):
        raise Error("msel_scatter_rows: bad source dtype code")
    if ns > 0 and w > 0 and sa == 0:
        raise Error("msel_scatter_rows: null source")


def msel_proba_column_check(sa: Int, c: Int, nn: Int, kk: Int, cc: Int, d: Int) raises:
    """The admission of `msel_proba_column` (both columns)."""
    if c != MSEL_F32 and c != MSEL_F64:
        raise Error("msel_proba_column: float32 or float64 input")
    if nn < 0 or kk < 1 or cc < 0 or cc >= kk:
        raise Error("msel_proba_column: bad sizes")
    if nn > 0 and (sa == 0 or d == 0):
        raise Error("msel_proba_column: null buffer address")


def msel_split_table_check(pa: Int, mm: Int, te: Int, tr: Int, ca: Int, ta: Int, sa: Int) raises:
    """The admission of `msel_split_table_i32` (both columns)."""
    if mm < 1 or te < 0 or tr < 0 or te + tr > mm or mm > 2147483000:
        raise Error("split_table_i32: bad sizes")
    if (te + tr > 0 and pa == 0) or ca == 0 or ta == 0 or sa == 0:
        raise Error("split_table_i32: null buffer address")


def msel_group_fold_perm_check(ca: Int, mm: Int, K: Int, pa: Int, da: Int, sa: Int) raises:
    """The admission of `msel_group_fold_perm_i32` (both columns)."""
    if K < 1 or mm < K or mm > 2147483000:
        raise Error("group_fold_assign_i32: m >= n_folds >= 1")
    if ca == 0 or pa == 0 or da == 0 or sa == 0:
        raise Error("group_fold_assign_i32: null buffer address")


def msel_scatter_rows_host_run(di: Int, nd: Int, dc: Int, sa: Int, sc: Int, ns: Int, w: Int, ix: Int) raises -> Int:
    """Rows of a host buffer cast into rows idx of a stored int64 or float64
    block (cross_val_predict's placement)."""
    msel_scatter_rows_check(nd, dc, sa, sc, ns, w)
    if not host_scatter_rows(MSEL_HOST_STORE.get_or_create_ptr()[], di, nd, dc, sa, sc, ns, w, ix):
        raise Error("msel_scatter_rows: row index out of bounds")
    return 0


def msel_proba_column_host_run(sa: Int, c: Int, nn: Int, kk: Int, cc: Int, d: Int) raises -> Int:
    """Column `col` of an n x k float32/float64 buffer as float32."""
    msel_proba_column_check(sa, c, nn, kk, cc, d)
    host_proba_column(sa, c, nn, kk, cc, d)
    return 0


def msel_split_table_host_run(pa: Int, mm: Int, te: Int, tr: Int, ca: Int, ta: Int, sa: Int) raises -> Int:
    """GroupShuffleSplit's per-group side table: table[g] 1 test, 0 train, 2
    neither; sums [train rows, test rows]."""
    msel_split_table_check(pa, mm, te, tr, ca, ta, sa)
    if not host_split_table_i32(pa, mm, te, tr, ca, ta, sa):
        raise Error("split_table_i32: group index out of range")
    return 0


def msel_group_fold_perm_host_run(ca: Int, mm: Int, K: Int, pa: Int, da: Int, sa: Int) raises -> Int:
    """Shuffled GroupKFold (the perm branch of `group_fold_assign_i32`): the
    permuted groups in n_folds nearly equal runs."""
    msel_group_fold_perm_check(ca, mm, K, pa, da, sa)
    if not host_group_fold_perm_i32(ca, mm, K, pa, da, sa):
        raise Error("group_fold_assign_i32: a permuted group is out of range")
    return 0


def msel_helpers_host_dispatch(op: Int, aa: Int, pp: Int) raises -> Int:
    """One model-selection helper by its `MH_*` code over admitted spans."""
    if op == MH_ARANGE_I64:
        return arange_i64(msel_word(aa, 0), msel_word(pp, 0), msel_word(pp, 1))
    if op == MH_COUNT_MASK_U8:
        return count_mask_u8(msel_word(aa, 0), msel_word(pp, 0))
    if op == MH_SELECT_MASK_U8_I64:
        return select_mask_u8_i64(msel_word(aa, 0), msel_word(pp, 0), msel_word(aa, 1), msel_word(aa, 2))
    if op == MH_MASK_FROM_INDICES_U8:
        return mask_from_indices_u8(msel_word(aa, 0), msel_word(pp, 0), msel_word(pp, 1), msel_word(aa, 1))
    if op == MH_LEAVE_RANGE_I64:
        return leave_range_i64(msel_word(pp, 0), msel_word(pp, 1), msel_word(pp, 2), msel_word(aa, 0),
                               msel_word(aa, 1))
    if op == MH_INDICES_OVERLAP_I64:
        return indices_overlap_i64(msel_word(aa, 0), msel_word(pp, 0), msel_word(aa, 1), msel_word(pp, 1),
                                   msel_word(pp, 2))
    if op == MH_FOLD_IDS:
        return fold_ids(msel_word(aa, 0), msel_word(pp, 0), msel_word(pp, 1), msel_word(pp, 2), msel_word(aa, 1),
                        msel_word(aa, 2), msel_word(aa, 3))
    if op == MH_SELECT_FOLD_I64:
        return select_fold_i64(msel_word(aa, 0), msel_word(pp, 0), msel_word(pp, 1), msel_word(aa, 1),
                               msel_word(aa, 2))
    if op == MH_SELECT_FOLDS_I64:
        return select_folds_i64_host_run(msel_word(aa, 0), msel_word(pp, 0), msel_word(pp, 1), msel_word(aa, 1),
                                         msel_word(aa, 2), msel_word(aa, 3))
    if op == MH_GATHER_I32 or op == MH_GATHER_I64:
        return msel_gather_host_run(0 if op == MH_GATHER_I32 else 1, msel_word(aa, 0), msel_word(pp, 0),
                                    msel_word(aa, 1), msel_word(pp, 1), msel_word(aa, 2))
    if op == MH_FIRST_SEEN_I32:
        return first_seen_i32(msel_word(aa, 0), msel_word(pp, 0), msel_word(pp, 1), msel_word(aa, 1),
                              msel_word(aa, 2))
    if op == MH_STRAT_FOLD_ASSIGN_I32:
        return strat_fold_assign_i32(msel_word(aa, 0), msel_word(pp, 0), msel_word(pp, 1), msel_word(pp, 2),
                                     msel_word(aa, 1), msel_word(aa, 2), msel_word(aa, 3), msel_word(aa, 4))
    if op == MH_STORE_ALLOC:
        return MSEL_HOST_STORE.get_or_create_ptr()[].alloc(msel_word(pp, 0))
    if op == MH_STORE_READ:
        MSEL_HOST_STORE.get_or_create_ptr()[].read(msel_word(pp, 0), msel_word(aa, 0), msel_word(pp, 1))
        return 0
    if op == MH_STORE_FREE:
        MSEL_HOST_STORE.get_or_create_ptr()[].free(msel_word(pp, 0))
        return 0
    if op == MH_SCATTER_ROWS:
        return msel_scatter_rows_host_run(msel_word(pp, 0), msel_word(pp, 1), msel_word(pp, 2), msel_word(aa, 0),
                                          msel_word(pp, 3), msel_word(pp, 4), msel_word(pp, 5), msel_word(pp, 6))
    if op == MH_PROBA_COLUMN:
        return msel_proba_column_host_run(msel_word(aa, 0), msel_word(pp, 0), msel_word(pp, 1), msel_word(pp, 2),
                                          msel_word(pp, 3), msel_word(aa, 1))
    if op == MH_SPLIT_TABLE_I32:
        return msel_split_table_host_run(msel_word(aa, 0), msel_word(pp, 0), msel_word(pp, 1), msel_word(pp, 2),
                                         msel_word(aa, 1), msel_word(aa, 2), msel_word(aa, 3))
    if op == MH_GROUP_FOLD_PERM_I32:
        return msel_group_fold_perm_host_run(msel_word(aa, 0), msel_word(pp, 0), msel_word(pp, 1), msel_word(aa, 1),
                                             msel_word(aa, 2), msel_word(aa, 3))
    if op == MH_STORE_FREE_BYTES:
        # The host column has no device memory to bound concurrent
        # cross-validation units by (bindings/msel_cv.mojo msel_cv_units).
        return -1
    raise Error("model-selection helpers: unknown operation " + String(op))


def msel_helpers_host_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The host model-selection helper entry (`MSEL_HELPERS_ABI`): addresses
    aa[0:na], Int words pp[0:np], no float words. Writes the run's answer to
    result; a refusal goes to sink."""
    try:
        msel_check_spans(operation, aa, na, pp, np, nf, result)
        var value = msel_helpers_host_dispatch(operation, aa, pp)
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = value
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct MselHelpersHostOwner(Defaultable, Movable):
    """The image-lifetime publisher word of the host model-selection helper
    table."""

    var word: Int

    def __init__(out self):
        self.word = 1


comptime _HOST_OWNER_NAME = "MojolearnMselHelpersHostOwnerV1"
comptime MSEL_HELPERS_HOST_OWNER = SharedSlot[MselHelpersHostOwner, _HOST_OWNER_NAME]


def publish_msel_helpers_host() raises:
    """Publish the core host binding's model-selection helper entry at module
    init. A repeated init publishes the same pointer; another image with the
    same key and another pointer refuses (core/entry_table.mojo)."""
    var owner = MSEL_HELPERS_HOST_OWNER.get_or_create_ptr()
    var key = MselEntryKey(Int(GLOBAL_NUMERIC_MODE), msel_entry_vendor("cpu"))
    msel_table_publish(key, Int(owner), [String(MSEL_HELPERS_ABI)], [MH_OP_COUNT], [msel_helpers_host_v1])
