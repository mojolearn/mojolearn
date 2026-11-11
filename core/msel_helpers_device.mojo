# SPDX-License-Identifier: Apache-2.0
"""The device column of the model-selection helper table
(core/msel_entry_table.mojo): the GPU base binding's (`bindings/_mojolearn.mojo`)
model-selection helpers as typed runs, its one `abi("C")` entry, and its
publication (lane calltable-msel, 2026-11-11).

Each run is the body the device Python export of the same name ran
(`bindings/hotpath_device.mojo` and `bindings/msel_device.mojo`, which now
unpack their arguments and call these runs): the device twin of
`core/hotpath_device.mojo` / `core/msel_device.mojo` when its switch is on
and the call is one the twin covers, otherwise the host run of
`core/msel_helpers_host.mojo`, which raises its own words. Same bytes
either way. The fold store's operations run on the store's own context
(`core/msel_device_x.mojo`), as the device exports always did.

No Python object crosses; the Python exports release the GIL around the
same runs.
"""
from std.sys.compile import is_defined
from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.vendor import COMPILED_VENDOR
from core.msel_entry_table import (
    MselEntryKey, msel_table_publish, msel_check_spans, msel_word, msel_entry_vendor,
    MSEL_HELPERS_ABI, MH_OP_COUNT,
    MH_ARANGE_I64, MH_COUNT_MASK_U8, MH_SELECT_MASK_U8_I64, MH_MASK_FROM_INDICES_U8, MH_LEAVE_RANGE_I64,
    MH_INDICES_OVERLAP_I64, MH_FOLD_IDS, MH_SELECT_FOLD_I64, MH_SELECT_FOLDS_I64, MH_GATHER_I32, MH_GATHER_I64,
    MH_FIRST_SEEN_I32, MH_STRAT_FOLD_ASSIGN_I32, MH_STORE_ALLOC, MH_STORE_READ, MH_STORE_FREE, MH_SCATTER_ROWS,
    MH_PROBA_COLUMN, MH_SPLIT_TABLE_I32, MH_GROUP_FOLD_PERM_I32, MH_STORE_FREE_BYTES,
)
from core.msel_helpers_host import (
    select_folds_i64_host_run, msel_scatter_rows_check, msel_proba_column_check, msel_split_table_check,
    msel_group_fold_perm_check,
)
from core.hotpath_device import (
    HPD_MAX_N, device_arange_skip_i64, device_count_mask_u8, device_first_seen_i32, device_indices_overlap_i64,
    device_kfold_ids, device_mask_from_indices_u8, device_select_fold_i64, device_select_folds_i64,
    device_select_mask_u8_i64, device_strat_fold_assign_i32, device_stratified_fold_ids,
)
from core.hotpath_impl import (
    arange_i64, count_mask_u8, fold_ids, indices_overlap_i64, leave_range_i64, mask_from_indices_u8,
    select_fold_i64, select_mask_u8_i64,
)
from core.hotpath_impl_rows import first_seen_i32, strat_fold_assign_i32
from core.label_device_steps import LABELS_DEVICE_BASE, LABELS_DEVICE_ELEM, LABELS_DEVICE_CONTEXT
from core.labels_native_api import LabelsGatherRequest
from core.labels_native_device import LabelsDeviceKernels
from core.msel_device import (
    device_group_fold_perm_i32, device_proba_column, device_scatter_rows, device_split_table_i32,
)
from core.msel_device_x import msel_store, MSEL_CONTEXT_SLOT
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.neural_context import process_ctx
from core.shared_slot import SharedSlot

#: The IDN_HPDEV_FOLDS switch (`-D MOJOLEARN_IDN_HPDEV_FOLDS_OFF`): the fold
#: and index helpers on the device (`bindings/hotpath_device.mojo` documents
#: the switch family).
comptime MSEL_HPDEV_FOLDS = LABELS_DEVICE_BASE and not is_defined["MOJOLEARN_IDN_HPDEV_FOLDS_OFF"]()

#: Largest k x n_folds table the stratified helpers build on the device.
comptime MSEL_HPDEV_MAX_TABLE = 1 << 26


def count_mask_u8_device_run(m: Int, rows: Int) raises -> Int:
    """The number of nonzero bytes of the n-byte mask."""
    comptime if LABELS_DEVICE_ELEM:
        if rows >= 1 and rows <= HPD_MAX_N and m != 0:
            var ctx = process_ctx[LABELS_DEVICE_CONTEXT]()
            return device_count_mask_u8(ctx, m, rows)
    return count_mask_u8(m, rows)


def fold_ids_device_run(ca: Int, rows: Int, k: Int, splits: Int, na: Int, fa: Int, fca: Int) raises -> Int:
    """`fold_ids` on the device: KFold's fold of a row is a closed form of
    its index; the stratified dealing ranks each row within its class by a
    stable sort. The per-fold counts of KFold are its `n_splits` sizes,
    written here (control data)."""
    comptime if MSEL_HPDEV_FOLDS:
        if rows >= 1 and rows <= HPD_MAX_N and splits >= 2 and splits <= HPD_MAX_N and fa != 0 and fca != 0:
            if ca == 0:
                var ctx = process_ctx[LABELS_DEVICE_CONTEXT]()
                device_kfold_ids(ctx, rows, splits, fa)
                var fcp = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=fca)
                for fold in range(splits):  # small-loop(splits: KFold n_splits fold sizes): control data, not rows
                    var size = rows // splits
                    if fold < rows % splits:
                        size += 1
                    fcp.unsafe_store(fold, Int64(size))
                return 0
            if k >= 1 and na != 0 and k * splits <= MSEL_HPDEV_MAX_TABLE:
                var ctx = process_ctx[LABELS_DEVICE_CONTEXT]()
                if device_stratified_fold_ids(ctx, ca, rows, k, splits, na, fa, fca):
                    return 0
    return fold_ids(ca, rows, k, splits, na, fa, fca)


def select_fold_i64_device_run(fa: Int, rows: Int, want: Int, ta: Int, ra: Int) raises -> Int:
    """`select_fold_i64` on the device (flag, exclusive scan, scatter)."""
    comptime if MSEL_HPDEV_FOLDS:
        if (
            rows >= 1 and rows <= HPD_MAX_N and fa != 0 and ta != 0 and ra != 0
            and want >= -2147483648 and want <= 2147483647
        ):
            var ctx = process_ctx[LABELS_DEVICE_CONTEXT]()
            return device_select_fold_i64(ctx, fa, rows, want, ta, ra)
    return select_fold_i64(fa, rows, want, ta, ra)


def select_folds_i64_device_run(fa: Int, rows: Int, splits: Int, sa: Int, ta: Int, ra: Int) raises -> Int:
    """`select_folds_i64` on the device: the fold ids go up once and every
    fold's train and test rows come back behind one synchronize. A count
    that disagrees with `sizes` reruns the host run, which reports the
    fold."""
    comptime if MSEL_HPDEV_FOLDS:
        if (
            rows >= 1 and rows <= HPD_MAX_N and splits >= 1 and splits <= 2147483647
            and fa != 0 and sa != 0 and ta != 0 and ra != 0
        ):
            var ctx = process_ctx[LABELS_DEVICE_CONTEXT]()
            if device_select_folds_i64(ctx, fa, rows, splits, sa, ta, ra):
                return -1
    return select_folds_i64_host_run(fa, rows, splits, sa, ta, ra)


def select_mask_u8_i64_device_run(ma: Int, rows: Int, ta: Int, ra: Int) raises -> Int:
    """`select_mask_u8_i64` on the device."""
    comptime if MSEL_HPDEV_FOLDS:
        if rows >= 1 and rows <= HPD_MAX_N and ma != 0:
            var ctx = process_ctx[LABELS_DEVICE_CONTEXT]()
            return device_select_mask_u8_i64(ctx, ma, rows, ta, ra)
    return select_mask_u8_i64(ma, rows, ta, ra)


def mask_from_indices_u8_device_run(ia: Int, count: Int, rows: Int, ma: Int) raises -> Int:
    """`mask_from_indices_u8` on the device; an index out of range reruns
    the host run, which raises."""
    comptime if MSEL_HPDEV_FOLDS:
        if (
            rows >= 1 and rows <= HPD_MAX_N and count >= 0 and count <= HPD_MAX_N
            and ma != 0 and (count == 0 or ia != 0)
        ):
            var ctx = process_ctx[LABELS_DEVICE_CONTEXT]()
            var set = device_mask_from_indices_u8(ctx, ia, count, rows, ma)
            if set >= 0:
                return set
    return mask_from_indices_u8(ia, count, rows, ma)


def arange_i64_device_run(d: Int, lo: Int, count: Int) raises -> Int:
    """`arange_i64` on the device."""
    comptime if MSEL_HPDEV_FOLDS:
        if count >= 1 and count <= HPD_MAX_N and d != 0:
            var ctx = process_ctx[LABELS_DEVICE_CONTEXT]()
            device_arange_skip_i64(ctx, d, lo, count, 0, 0)
            return 0
    return arange_i64(d, lo, count)


def leave_range_i64_device_run(rows: Int, a: Int, b: Int, ra: Int, ta: Int) raises -> Int:
    """`leave_range_i64` on the device: test = [lo, hi), train = the other
    rows, each a range with at most one gap."""
    comptime if MSEL_HPDEV_FOLDS:
        var held = b - a
        if (
            rows >= 1 and rows <= HPD_MAX_N and a >= 0 and b >= a and b <= rows
            and (held == 0 or ta != 0) and (held == rows or ra != 0)
        ):
            var ctx = process_ctx[LABELS_DEVICE_CONTEXT]()
            if held > 0:
                device_arange_skip_i64(ctx, ta, a, held, 0, 0)
            if rows - held > 0:
                device_arange_skip_i64(ctx, ra, 0, rows - held, a, held)
            return held
    return leave_range_i64(rows, a, b, ra, ta)


def indices_overlap_i64_device_run(a: Int, na: Int, b: Int, nb: Int, limit: Int) raises -> Int:
    """`indices_overlap_i64` on the device; an index out of range reruns the
    host run."""
    comptime if MSEL_HPDEV_FOLDS:
        if (
            na >= 1 and na <= HPD_MAX_N and nb >= 1 and nb <= HPD_MAX_N
            and limit >= 1 and limit <= HPD_MAX_N and a != 0 and b != 0
        ):
            var ctx = process_ctx[LABELS_DEVICE_CONTEXT]()
            var hit = device_indices_overlap_i64(ctx, a, na, b, nb, limit)
            if hit >= 0:
                return hit
    return indices_overlap_i64(a, na, b, nb, limit)


def first_seen_i32_device_run(ca: Int, rows: Int, kk: Int, ea: Int, na: Int) raises -> Int:
    """`first_seen_i32` on the device; a code out of range reruns the host
    run, which raises."""
    comptime if MSEL_HPDEV_FOLDS:
        if (
            rows >= 1 and rows <= HPD_MAX_N and kk >= 1 and kk <= HPD_MAX_N
            and ca != 0 and ea != 0 and na != 0
        ):
            var ctx = process_ctx[LABELS_DEVICE_CONTEXT]()
            var m = device_first_seen_i32(ctx, ca, rows, kk, ea, na)
            if m >= 0:
                return m
    return first_seen_i32(ca, rows, kk, ea, na)


def strat_fold_assign_i32_device_run(ea: Int, rows: Int, kk: Int, K: Int, aa: Int, pa: Int, ca: Int,
                                     da: Int) raises -> Int:
    """`strat_fold_assign_i32` on the device; tables the device twin does
    not accept rerun the host run."""
    comptime if MSEL_HPDEV_FOLDS:
        if (
            rows >= 1 and rows <= HPD_MAX_N and kk >= 1 and K >= 1 and K <= HPD_MAX_N
            and kk * K <= MSEL_HPDEV_MAX_TABLE and ea != 0 and aa != 0 and ca != 0 and da != 0
        ):
            var ctx = process_ctx[LABELS_DEVICE_CONTEXT]()
            if device_strat_fold_assign_i32(ctx, ea, rows, kk, K, aa, pa, ca, da):
                return 0
    return strat_fold_assign_i32(ea, rows, kk, K, aa, pa, ca, da)


def msel_gather_device_run(kind: Int, table: Int, classes: Int, codes: Int, count: Int, dst: Int) raises -> Int:
    """`gather_i32` (kind 0) / `gather_i64` (kind 1) on the device, the label
    gathers of `core/labels_native_device.mojo`."""
    LabelsDeviceKernels.gather(kind, LabelsGatherRequest(table, codes, dst, classes, count))
    return 0


def msel_store_alloc_device_run(n: Int) raises -> Int:
    """A zeroed resident block of n bytes; returns its id."""
    var ctx = process_ctx[MSEL_CONTEXT_SLOT]()
    return msel_store()[].alloc(ctx, n)


def msel_store_read_device_run(id: Int, d: Int, n: Int) raises -> Int:
    """Download n bytes of a resident block."""
    var ctx = process_ctx[MSEL_CONTEXT_SLOT]()
    msel_store()[].read(ctx, id, d, n)
    return 0


def msel_store_free_device_run(id: Int) raises -> Int:
    """Release a resident block."""
    var ctx = process_ctx[MSEL_CONTEXT_SLOT]()
    msel_store()[].free(ctx, id)
    return 0


def msel_store_free_bytes_device_run() raises -> Int:
    """Free bytes of the device the fold store lives on: the measured bound
    of how many cross-validation units run at once (bindings/msel_cv.mojo
    msel_cv_units, lane p3-msel-parallel-and-process-pool)."""
    var ctx = process_ctx[MSEL_CONTEXT_SLOT]()
    var mem = ctx.get_memory_info()
    return Int(mem[0])


def msel_scatter_rows_device_run(di: Int, nd: Int, dc: Int, sa: Int, sc: Int, ns: Int, w: Int, ix: Int) raises -> Int:
    """Rows of a host buffer cast into rows idx of a resident int64 or
    float64 block (cross_val_predict's placement)."""
    msel_scatter_rows_check(nd, dc, sa, sc, ns, w)
    var ctx = process_ctx[MSEL_CONTEXT_SLOT]()
    if not device_scatter_rows(ctx, msel_store()[], di, nd, dc, sa, sc, ns, w, ix):
        raise Error("msel_scatter_rows: row index out of bounds")
    return 0


def msel_proba_column_device_run(sa: Int, c: Int, nn: Int, kk: Int, cc: Int, d: Int) raises -> Int:
    """Column `col` of an n x k float32/float64 buffer as float32."""
    msel_proba_column_check(sa, c, nn, kk, cc, d)
    var ctx = process_ctx[MSEL_CONTEXT_SLOT]()
    device_proba_column(ctx, sa, c, nn, kk, cc, d)
    return 0


def msel_split_table_device_run(pa: Int, mm: Int, te: Int, tr: Int, ca: Int, ta: Int, sa: Int) raises -> Int:
    """GroupShuffleSplit's per-group side table: table[g] 1 test, 0 train, 2
    neither; sums [train rows, test rows]."""
    msel_split_table_check(pa, mm, te, tr, ca, ta, sa)
    var ctx = process_ctx[MSEL_CONTEXT_SLOT]()
    if not device_split_table_i32(ctx, pa, mm, te, tr, ca, ta, sa):
        raise Error("split_table_i32: group index out of range")
    return 0


def msel_group_fold_perm_device_run(ca: Int, mm: Int, K: Int, pa: Int, da: Int, sa: Int) raises -> Int:
    """Shuffled GroupKFold (the perm branch of `group_fold_assign_i32`): the
    permuted groups in n_folds nearly equal runs."""
    msel_group_fold_perm_check(ca, mm, K, pa, da, sa)
    var ctx = process_ctx[MSEL_CONTEXT_SLOT]()
    if not device_group_fold_perm_i32(ctx, ca, mm, K, pa, da, sa):
        raise Error("group_fold_assign_i32: a permuted group is out of range")
    return 0


def msel_helpers_device_dispatch(op: Int, aa: Int, pp: Int) raises -> Int:
    """One model-selection helper by its `MH_*` code over admitted spans."""
    if op == MH_ARANGE_I64:
        return arange_i64_device_run(msel_word(aa, 0), msel_word(pp, 0), msel_word(pp, 1))
    if op == MH_COUNT_MASK_U8:
        return count_mask_u8_device_run(msel_word(aa, 0), msel_word(pp, 0))
    if op == MH_SELECT_MASK_U8_I64:
        return select_mask_u8_i64_device_run(msel_word(aa, 0), msel_word(pp, 0), msel_word(aa, 1), msel_word(aa, 2))
    if op == MH_MASK_FROM_INDICES_U8:
        return mask_from_indices_u8_device_run(msel_word(aa, 0), msel_word(pp, 0), msel_word(pp, 1),
                                               msel_word(aa, 1))
    if op == MH_LEAVE_RANGE_I64:
        return leave_range_i64_device_run(msel_word(pp, 0), msel_word(pp, 1), msel_word(pp, 2), msel_word(aa, 0),
                                          msel_word(aa, 1))
    if op == MH_INDICES_OVERLAP_I64:
        return indices_overlap_i64_device_run(msel_word(aa, 0), msel_word(pp, 0), msel_word(aa, 1),
                                              msel_word(pp, 1), msel_word(pp, 2))
    if op == MH_FOLD_IDS:
        return fold_ids_device_run(msel_word(aa, 0), msel_word(pp, 0), msel_word(pp, 1), msel_word(pp, 2),
                                   msel_word(aa, 1), msel_word(aa, 2), msel_word(aa, 3))
    if op == MH_SELECT_FOLD_I64:
        return select_fold_i64_device_run(msel_word(aa, 0), msel_word(pp, 0), msel_word(pp, 1), msel_word(aa, 1),
                                          msel_word(aa, 2))
    if op == MH_SELECT_FOLDS_I64:
        return select_folds_i64_device_run(msel_word(aa, 0), msel_word(pp, 0), msel_word(pp, 1), msel_word(aa, 1),
                                           msel_word(aa, 2), msel_word(aa, 3))
    if op == MH_GATHER_I32 or op == MH_GATHER_I64:
        return msel_gather_device_run(0 if op == MH_GATHER_I32 else 1, msel_word(aa, 0), msel_word(pp, 0),
                                      msel_word(aa, 1), msel_word(pp, 1), msel_word(aa, 2))
    if op == MH_FIRST_SEEN_I32:
        return first_seen_i32_device_run(msel_word(aa, 0), msel_word(pp, 0), msel_word(pp, 1), msel_word(aa, 1),
                                         msel_word(aa, 2))
    if op == MH_STRAT_FOLD_ASSIGN_I32:
        return strat_fold_assign_i32_device_run(msel_word(aa, 0), msel_word(pp, 0), msel_word(pp, 1),
                                                msel_word(pp, 2), msel_word(aa, 1), msel_word(aa, 2),
                                                msel_word(aa, 3), msel_word(aa, 4))
    if op == MH_STORE_ALLOC:
        return msel_store_alloc_device_run(msel_word(pp, 0))
    if op == MH_STORE_READ:
        return msel_store_read_device_run(msel_word(pp, 0), msel_word(aa, 0), msel_word(pp, 1))
    if op == MH_STORE_FREE:
        return msel_store_free_device_run(msel_word(pp, 0))
    if op == MH_SCATTER_ROWS:
        return msel_scatter_rows_device_run(msel_word(pp, 0), msel_word(pp, 1), msel_word(pp, 2), msel_word(aa, 0),
                                            msel_word(pp, 3), msel_word(pp, 4), msel_word(pp, 5), msel_word(pp, 6))
    if op == MH_PROBA_COLUMN:
        return msel_proba_column_device_run(msel_word(aa, 0), msel_word(pp, 0), msel_word(pp, 1), msel_word(pp, 2),
                                            msel_word(pp, 3), msel_word(aa, 1))
    if op == MH_SPLIT_TABLE_I32:
        return msel_split_table_device_run(msel_word(aa, 0), msel_word(pp, 0), msel_word(pp, 1), msel_word(pp, 2),
                                           msel_word(aa, 1), msel_word(aa, 2), msel_word(aa, 3))
    if op == MH_GROUP_FOLD_PERM_I32:
        return msel_group_fold_perm_device_run(msel_word(aa, 0), msel_word(pp, 0), msel_word(pp, 1),
                                               msel_word(aa, 1), msel_word(aa, 2), msel_word(aa, 3))
    if op == MH_STORE_FREE_BYTES:
        return msel_store_free_bytes_device_run()
    raise Error("model-selection helpers: unknown operation " + String(op))


def msel_helpers_device_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The GPU base binding's model-selection helper entry
    (`MSEL_HELPERS_ABI`): addresses aa[0:na], Int words pp[0:np], no float
    words. Writes the run's answer to result; a refusal goes to sink."""
    try:
        msel_check_spans(operation, aa, na, pp, np, nf, result)
        var value = msel_helpers_device_dispatch(operation, aa, pp)
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = value
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct MselHelpersDeviceOwner(Defaultable, Movable):
    """The image-lifetime publisher word of the device model-selection helper
    table."""

    var word: Int

    def __init__(out self):
        self.word = 1


comptime _DEVICE_OWNER_NAME = "MojolearnMselHelpersDeviceOwnerV1"
comptime MSEL_HELPERS_DEVICE_OWNER = SharedSlot[MselHelpersDeviceOwner, _DEVICE_OWNER_NAME]


def publish_msel_helpers_device() raises:
    """Publish the GPU base binding's model-selection helper entry at module
    init. A build with no GPU vendor ("none") publishes nothing: its callers
    refuse with the table's "provider is not published"."""
    var word = msel_entry_vendor(COMPILED_VENDOR)
    if word < 1:
        return
    var owner = MSEL_HELPERS_DEVICE_OWNER.get_or_create_ptr()
    var key = MselEntryKey(Int(GLOBAL_NUMERIC_MODE), word)
    msel_table_publish(key, Int(owner), [String(MSEL_HELPERS_ABI)], [MH_OP_COUNT], [msel_helpers_device_v1])
