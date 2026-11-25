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
from max.gpu.host import DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.vendor import COMPILED_VENDOR
from core.msel_entry_table import (
    MselEntryKey, msel_table_publish, msel_check_spans, msel_word, msel_entry_vendor,
    MSEL_HELPERS_ABI, MH_OP_COUNT,
    MH_ARANGE_I64, MH_COUNT_MASK_U8, MH_SELECT_MASK_U8_I64, MH_MASK_FROM_INDICES_U8, MH_LEAVE_RANGE_I64,
    MH_FOLD_IDS, MH_SELECT_FOLD_I64, MH_SELECT_FOLDS_I64, MH_GATHER_I32,
    MH_STORE_FREE,
    MH_PROBA_COLUMN, MH_SPLIT_TABLE_I32, MH_GROUP_FOLD_PERM_I32,
    MH_STORE_PUT, MH_STORE_TAKE_ROWS,
    MH_FIRST_SEEN_RES, MH_STRAT_FOLD_ASSIGN_RES, MH_SELECT_FOLDS_RES, MH_FOLD_IDS_RES,
    MH_GROUP_CLASS_COUNTS_RES,
)
from core.msel_helpers_host import (
    select_folds_i64_host_run, msel_proba_column_check, msel_split_table_check,
    msel_group_fold_perm_check, msel_gather_host_run,
)
from core.hotpath_device import (
    HPD_MAX_N, device_arange_skip_i64, device_count_mask_u8, device_first_seen_i32,
    device_kfold_ids, device_mask_from_indices_u8, device_select_fold_i64, device_select_folds_i64,
    device_select_mask_u8_i64, device_stratified_fold_ids,
    device_first_seen_on, device_strat_fold_assign_res, device_select_folds_on, device_stratified_fold_ids_on,
    device_group_class_counts_on,
)
from core.hotpath_impl import (
    arange_i64, count_mask_u8, fold_ids, leave_range_i64, mask_from_indices_u8,
    select_fold_i64, select_mask_u8_i64,
)
from core.hotpath_impl_rows import (
    bincount_i64, first_seen_i32, first_seen_rank_i32, group_class_counts_i32, strat_fold_assign_i32,
)
from core.label_device_steps import LABELS_DEVICE_BASE, LABELS_DEVICE_ELEM, LABELS_DEVICE_CONTEXT
from core.labels_native_api import LabelsGatherRequest
from core.labels_native_device import LabelsDeviceKernels
from core.msel_device import (
    device_group_fold_perm_i32, device_proba_column, device_split_table_i32,
    device_take_rows,
)
from core.msel_device_x import msel_store, MSEL_CONTEXT_SLOT
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.neural_context import process_ctx
from core.shared_slot import SharedSlot

#: The fold and index helpers on the device (`bindings/hotpath_device.mojo`
#: documents the switch family).
# TOMBSTONE: MOJOLEARN_IDN_HPDEV_FOLDS_OFF (old path, no A/B recorded beside the default) deleted 2026-11-15 by lane/expdel2-04-base; code recoverable at 8adf7b05f6.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_HPDEV_FOLDS_OFF.expdel2-04-base.patch; record in docs/TOMBSTONES.md.
comptime MSEL_HPDEV_FOLDS = LABELS_DEVICE_BASE

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


# TOMBSTONE: strat_fold_assign_i32 export (dead export: no caller since lane vdf-scope-split moved the splitters to the stored-code MH_STRAT_FOLD_ASSIGN_RES) deleted 2026-11-24 by lane vdf2-native; code recoverable at 2da8727c02.
# Restore: git apply ~/mojolearn-evidence/experiments-removed/dead_export_strat_fold_assign_i32.patch; record in docs/TOMBSTONES.md.


def msel_gather_device_run(kind: Int, table: Int, classes: Int, codes: Int, count: Int, dst: Int) raises -> Int:
    """`gather_i32` (kind 0) / `gather_i64` (kind 1) on the device, the label
    gathers of `core/labels_native_device.mojo`."""
    LabelsDeviceKernels.gather(kind, LabelsGatherRequest(table, codes, dst, classes, count))
    return 0






def msel_store_free_device_run(id: Int) raises -> Int:
    """Release a resident block."""
    var ctx = process_ctx[MSEL_CONTEXT_SLOT]()
    msel_store()[].free(ctx, id)
    return 0


def msel_store_put_device_run(src: Int, n: Int) raises -> Int:
    """A resident copy of n host bytes at src; returns its id."""
    var ctx = process_ctx[MSEL_CONTEXT_SLOT]()
    return msel_store()[].put(ctx, src, n)


def msel_store_take_rows_device_run(src_id: Int, n_src: Int, row_bytes: Int, idx_id: Int, n_idx: Int,
                                    dst: Int) raises -> Int:
    """Rows idx of the resident source into host memory (a fold's rows): one
    device gather and one download, the take-rows op of the model-selection
    helper table."""
    if n_src < 0 or row_bytes < 0 or n_idx < 0:
        raise Error("msel_take_rows: dimensions must be non-negative")
    var ctx = process_ctx[MSEL_CONTEXT_SLOT]()
    if not device_take_rows(ctx, msel_store()[], src_id, n_src, row_bytes, idx_id, n_idx, dst):
        raise Error("msel_take_rows: row index out of bounds")
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


# ---------------------------------------------------------------- layout V3: stored label codes
# (lane vdf-scope-split, 2026-11-24). The stratified splitters put their n
# int32 label codes once (MH_STORE_PUT) and these runs read them where they
# are, on the store's context: the first-seen sizes and rank table come back
# (2k words), the first-seen codes and the fold ids never cross the bus, and
# each fold's rows come down once, into the caller's arrays. A fold block is
# allocated here and returned (the caller frees it), or freed here when the
# run refuses. Where the device twin does not cover a call (a build without
# it, or a table past its bound), the block's words come down and the host
# column's core function runs on them, as the address runs rerun the host
# helper; a fold block made that way goes up once.


def _res_dims(rows: Int, kk: Int) raises:
    if rows < 1 or kk < 1:
        raise Error("model-selection helpers: a stored-code operation needs n >= 1 and k >= 1")


def _res_words(ctx: DeviceContext, id: Int, rows: Int) raises -> List[Int32]:
    """The n int32 words of a stored block, downloaded (the host fallback)."""
    var words = List[Int32](length=rows, fill=0)
    msel_store()[].read(ctx, id, Int(words.unsafe_ptr()), 4 * rows)
    return words^


def msel_first_seen_res_device_run(codes_id: Int, rows: Int, kk: Int, na: Int, ra: Int) raises -> Int:
    """MH_FIRST_SEEN_RES on the device: the class sizes in first-seen order
    and the rank table of the stored codes (2k words down); the classes
    present. A code out of range raises the host run's words."""
    _res_dims(rows, kk)
    if na == 0:
        raise Error("first_seen_i32: null counts address")
    var ctx = process_ctx[MSEL_CONTEXT_SLOT]()
    comptime if MSEL_HPDEV_FOLDS:
        if rows <= HPD_MAX_N and kk <= HPD_MAX_N:
            var codes = msel_store()[].address(codes_id, 4 * rows)
            var seen = device_first_seen_on(ctx, codes, rows, kk, 0, na, ra)
            if seen < 0:
                raise Error("first_seen_i32: code out of range")
            return seen
    var held = _res_words(ctx, codes_id, rows)  # cpu-route: no device twin for this call (build or bound)
    var m = first_seen_rank_i32(Int(held.unsafe_ptr()), rows, kk, na, ra)
    _ = held  # the address above is not an owner
    return m


def msel_strat_fold_assign_res_device_run(codes_id: Int, rows: Int, kk: Int, K: Int, width: Int, rank: Int, aa: Int,
                                          pa: Int, ca: Int, sa: Int) raises -> Int:
    """MH_STRAT_FOLD_ASSIGN_RES on the device: rank[codes] and the
    assignment on the device into a new fold block (returned, never
    downloaded), the n_folds fold sizes down."""
    _res_dims(rows, kk)
    if K < 1 or aa == 0 or ca == 0 or sa == 0 or (rank != 0 and width < 1):
        raise Error("strat_fold_assign_i32: n_folds >= 1 and non-null tables")
    var ctx = process_ctx[MSEL_CONTEXT_SLOT]()
    var store = msel_store()
    comptime if MSEL_HPDEV_FOLDS:
        if rows <= HPD_MAX_N and K <= HPD_MAX_N and width <= HPD_MAX_N and kk * K <= MSEL_HPDEV_MAX_TABLE:
            var made = store[].alloc(ctx, 4 * rows)
            var ok = False
            try:
                var codes = store[].address(codes_id, 4 * rows)
                var dev_folds = store[].address(made, 4 * rows)
                ok = device_strat_fold_assign_res(ctx, codes, rank, width, rows, kk, K, aa, pa, ca, dev_folds, sa)
            except problem:
                store[].free(ctx, made)
                raise problem^
            if ok:
                return made
            store[].free(ctx, made)
            # Inconsistent tables or a code out of range: the host run below
            # raises its own words.
    var held = _res_words(ctx, codes_id, rows)  # cpu-route: no device twin for this call (build, bound or refusal words)
    var enc = List[Int32](length=rows, fill=0)
    if rank != 0:
        _ = msel_gather_host_run(0, rank, width, Int(held.unsafe_ptr()), rows, Int(enc.unsafe_ptr()))
    else:
        enc = held.copy()
    var folds = List[Int32](length=rows, fill=0)
    _ = strat_fold_assign_i32(Int(enc.unsafe_ptr()), rows, kk, K, aa, pa, ca, Int(folds.unsafe_ptr()))
    _ = bincount_i64(Int(folds.unsafe_ptr()), 2, rows, K, sa, False)
    var folds_id = store[].put(ctx, Int(folds.unsafe_ptr()), 4 * rows)
    _ = held  # the addresses above are not owners
    _ = enc
    _ = folds
    return folds_id


def msel_select_folds_res_device_run(folds_id: Int, rows: Int, splits: Int, sa: Int, ta: Int, ra: Int) raises -> Int:
    """MH_SELECT_FOLDS_RES on the device: every fold's rows out of the
    stored fold block into the caller's arrays behind one synchronize; -1,
    or the first fold whose count differs from `sizes`."""
    if rows < 1 or splits < 1:
        raise Error("model-selection helpers: a stored-fold selection needs n >= 1 and k >= 1")
    if sa == 0 or ta == 0 or ra == 0:
        raise Error("select_folds_i64: null table address")
    var ctx = process_ctx[MSEL_CONTEXT_SLOT]()
    comptime if MSEL_HPDEV_FOLDS:
        if rows <= HPD_MAX_N and splits <= 2147483647:
            var dev_folds = msel_store()[].address(folds_id, 4 * rows)
            return device_select_folds_on(ctx, dev_folds, rows, splits, sa, ta, ra)
    var held = _res_words(ctx, folds_id, rows)  # cpu-route: no device twin for this call (build or bound)
    var bad = select_folds_i64_host_run(Int(held.unsafe_ptr()), rows, splits, sa, ta, ra)
    _ = held  # the address above is not an owner
    return bad


def msel_fold_ids_res_device_run(codes_id: Int, rows: Int, kk: Int, splits: Int, na: Int, fca: Int) raises -> Int:
    """MH_FOLD_IDS_RES on the device: the stratified `fold_ids` of the
    stored codes into a new fold block (returned, never downloaded); the
    class counts and fold sizes down."""
    _res_dims(rows, kk)
    if splits < 2 or na == 0 or fca == 0:
        raise Error("fold_ids: n_splits >= 2 and non-null counts")
    var ctx = process_ctx[MSEL_CONTEXT_SLOT]()
    var store = msel_store()
    comptime if MSEL_HPDEV_FOLDS:
        if rows <= HPD_MAX_N and splits <= HPD_MAX_N and kk * splits <= MSEL_HPDEV_MAX_TABLE:
            var made = store[].alloc(ctx, 4 * rows)
            var ok = False
            try:
                var codes = store[].address(codes_id, 4 * rows)
                var dev_folds = store[].address(made, 4 * rows)
                ok = device_stratified_fold_ids_on(ctx, codes, rows, kk, splits, na, 0, fca, dev_folds)
            except problem:
                store[].free(ctx, made)
                raise problem^
            if ok:
                return made
            store[].free(ctx, made)
            # A code out of range: the host run below raises its own words.
    var held = _res_words(ctx, codes_id, rows)  # cpu-route: no device twin for this call (build, bound or refusal words)
    var folds = List[Int32](length=rows, fill=0)
    _ = fold_ids(Int(held.unsafe_ptr()), rows, kk, splits, na, Int(folds.unsafe_ptr()), fca)
    var folds_id = store[].put(ctx, Int(folds.unsafe_ptr()), 4 * rows)
    _ = held  # the addresses above are not owners
    _ = folds
    return folds_id


def msel_group_class_counts_res_device_run(codes_id: Int, rows: Int, kk: Int, m: Int, nc: Int, ga: Int, ra: Int,
                                           da: Int) raises -> Int:
    """MH_GROUP_CLASS_COUNTS_RES on the device (layout V4, lane
    vdf-scope-split2, 2026-11-24): the group x class table counted on the
    stored codes, classes ranked as they are counted; the n group codes and
    the rank table go up, the m x n_classes table comes down."""
    _res_dims(rows, kk)
    if m < 1 or nc < 1 or ga == 0 or da == 0:
        raise Error("bincount2_i32: 0 <= n, 1 <= ka, kb and non-null tables")
    var ctx = process_ctx[MSEL_CONTEXT_SLOT]()
    comptime if MSEL_HPDEV_FOLDS:
        if rows <= HPD_MAX_N and kk <= HPD_MAX_N and nc <= HPD_MAX_N // m:
            var codes = msel_store()[].address(codes_id, 4 * rows)
            if device_group_class_counts_on(ctx, codes, rows, kk, ga, m, ra, nc, da):
                return 0
            # A code or a group out of range: the host run below raises its own words.
    var held = _res_words(ctx, codes_id, rows)  # cpu-route: no device twin for this call (build, bound or refusal words)
    _ = group_class_counts_i32(Int(held.unsafe_ptr()), rows, kk, ga, m, ra, nc, da)
    _ = held  # the address above is not an owner
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
    if op == MH_FOLD_IDS:
        return fold_ids_device_run(msel_word(aa, 0), msel_word(pp, 0), msel_word(pp, 1), msel_word(pp, 2),
                                   msel_word(aa, 1), msel_word(aa, 2), msel_word(aa, 3))
    if op == MH_SELECT_FOLD_I64:
        return select_fold_i64_device_run(msel_word(aa, 0), msel_word(pp, 0), msel_word(pp, 1), msel_word(aa, 1),
                                          msel_word(aa, 2))
    if op == MH_SELECT_FOLDS_I64:
        return select_folds_i64_device_run(msel_word(aa, 0), msel_word(pp, 0), msel_word(pp, 1), msel_word(aa, 1),
                                           msel_word(aa, 2), msel_word(aa, 3))
    if op == MH_GATHER_I32:
        return msel_gather_device_run(0, msel_word(aa, 0), msel_word(pp, 0), msel_word(aa, 1), msel_word(pp, 1),
                                      msel_word(aa, 2))
    if op == MH_STORE_FREE:
        return msel_store_free_device_run(msel_word(pp, 0))
    if op == MH_PROBA_COLUMN:
        return msel_proba_column_device_run(msel_word(aa, 0), msel_word(pp, 0), msel_word(pp, 1), msel_word(pp, 2),
                                            msel_word(pp, 3), msel_word(aa, 1))
    if op == MH_SPLIT_TABLE_I32:
        return msel_split_table_device_run(msel_word(aa, 0), msel_word(pp, 0), msel_word(pp, 1), msel_word(pp, 2),
                                           msel_word(aa, 1), msel_word(aa, 2), msel_word(aa, 3))
    if op == MH_GROUP_FOLD_PERM_I32:
        return msel_group_fold_perm_device_run(msel_word(aa, 0), msel_word(pp, 0), msel_word(pp, 1),
                                               msel_word(aa, 1), msel_word(aa, 2), msel_word(aa, 3))
    if op == MH_STORE_PUT:
        return msel_store_put_device_run(msel_word(aa, 0), msel_word(pp, 0))
    if op == MH_STORE_TAKE_ROWS:
        return msel_store_take_rows_device_run(msel_word(pp, 0), msel_word(pp, 1), msel_word(pp, 2),
                                               msel_word(pp, 3), msel_word(pp, 4), msel_word(aa, 0))
    if op == MH_FIRST_SEEN_RES:
        return msel_first_seen_res_device_run(msel_word(pp, 0), msel_word(pp, 1), msel_word(pp, 2),
                                              msel_word(aa, 0), msel_word(aa, 1))
    if op == MH_STRAT_FOLD_ASSIGN_RES:
        return msel_strat_fold_assign_res_device_run(msel_word(pp, 0), msel_word(pp, 1), msel_word(pp, 2),
                                                     msel_word(pp, 3), msel_word(pp, 4), msel_word(aa, 0),
                                                     msel_word(aa, 1), msel_word(aa, 2), msel_word(aa, 3),
                                                     msel_word(aa, 4))
    if op == MH_SELECT_FOLDS_RES:
        return msel_select_folds_res_device_run(msel_word(pp, 0), msel_word(pp, 1), msel_word(pp, 2),
                                                msel_word(aa, 0), msel_word(aa, 1), msel_word(aa, 2))
    if op == MH_FOLD_IDS_RES:
        return msel_fold_ids_res_device_run(msel_word(pp, 0), msel_word(pp, 1), msel_word(pp, 2), msel_word(pp, 3),
                                            msel_word(aa, 0), msel_word(aa, 1))
    if op == MH_GROUP_CLASS_COUNTS_RES:
        return msel_group_class_counts_res_device_run(msel_word(pp, 0), msel_word(pp, 1), msel_word(pp, 2),
                                                      msel_word(pp, 3), msel_word(pp, 4), msel_word(aa, 0),
                                                      msel_word(aa, 1), msel_word(aa, 2))
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
