# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The category map of a fit over THE FOLD X WORD, on the device (lane
vdf2-gbdtcat2, 2026-11-24).

A model-selection fold taken on the device (an x_prep store slot,
core/fold_x_device.mojo: the fold's rows, ROW-MAJOR, the member's own copy)
has no host rows. The gbdt member's `_category_fit` (core/gbdt_member.mojo)
keys each categorical column from its Float32 words, fits the ascending
distinct run (gbdt/data/category_map.mojo `category_distinct_sorted`) and
writes each row's dense code (its rank in that run) over the column. This
module does the same on the slot, per categorical column:

  1. KEY: one thread per row, `category_key_f32` of the row's cell (the host
     statement itself, `@always_inline`), into a 64-bit key buffer;
  2. SORT, FLAG, SCAN, EMIT: `device_unique_inverse_resident`
     (core/label_encode_device.mojo, `LABEL_KIND_U64`: the key unchanged):
     two stable radix sorts (low half, then high half) are the sort by the
     full 64-bit key; a boundary flag per sorted position; an exclusive scan
     of the flags; each row's code is its key's rank among the distinct
     keys, and each distinct key lands at its rank;
  3. CODE: one thread per row writes `Float32(code)` over its cell of the
     slot (the slot is the fold's own copy; the caller's X is never written).

The codes are the host's: the rank of the key in the ascending distinct run
is `category_code_of` of the same run. Only the distinct keys come home (k
words per column), as the stored map (category_map.mojo's block layout);
X never does. The refusal of a column with more than CATEGORY_MAX_VALUES
distinct keys is the host's, in its words, before that column's codes are
written. Parallel over rows in every step; the columns are a driver loop
(each column's rows run on the whole device).
"""
from std.memory import unsafe_memcpy
from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from core.fold_x_device import FOLD_X_STORE
from core.fold_x_word import fold_x_slot
from core.label_encode import LABEL_KIND_U64
from core.label_encode_device import device_unique_inverse_resident
from gbdt.data.category_map import CATEGORY_MAP_VERSION, CATEGORY_MAX_VALUES, category_key_f32

comptime _TPB = 256
comptime _FP = MutPointer[Float32, MutAnyOrigin]
comptime _U64 = MutPointer[UInt64, MutAnyOrigin]
comptime _I = MutPointer[Int32, MutAnyOrigin]


@always_inline
def _row() -> Int:
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


def _cm_key_kernel(x: _FP, keys: _U64, rows: Int32, cols: Int32, feature: Int32):
    """keys[r] = category_key_f32(x[r * cols + feature])."""
    var r = _row()
    if r >= Int(rows):
        return
    keys.unsafe_store(r, category_key_f32(x.unsafe_load(r * Int(cols) + Int(feature))))


def _cm_code_kernel(x: _FP, codes: _I, rows: Int32, cols: Int32, feature: Int32):
    """x[r * cols + feature] = Float32(codes[r]) (exact: a code is below 2^24)."""
    var r = _row()
    if r >= Int(rows):
        return
    x.unsafe_store(r * Int(cols) + Int(feature), Float32(codes.unsafe_load(r)))


def fold_x_category_fit(
    ctx: DeviceContext, x_word: Int, rows: Int, cols: Int, columns: List[Int]
) raises -> List[UInt64]:
    """Fit the category map of the ascending, distinct categorical feature
    indices `columns` over the `rows` x `cols` ROW-MAJOR rows of the fold
    slot the negative `x_word` names, write the dense codes over those
    columns of the slot, and return the stored map's words (the estimator's
    `category_map_`, gbdt/data/category_map.mojo layout). Synchronizes
    before it returns."""
    if x_word >= 0:
        raise Error("fold_x_category_fit: the X word names no fold slot")
    if rows <= 0 or cols <= 0:
        raise Error("fold_x_category_fit: the fold has no rows or no columns")
    if rows > 2147483647 or cols > 2147483647:
        raise Error("fold_x_category_fit: the fold exceeds the Int32 row bound")
    var n_cat = len(columns)
    var previous = -1
    for c in range(n_cat):  # small-loop(the categorical feature indices): request words, not row data
        if columns[c] <= previous or columns[c] >= cols:
            raise Error("category map: columns must be added in ascending feature order")
        previous = columns[c]
    var store = FOLD_X_STORE.get_or_create_ptr()
    var x = store[].ptr(fold_x_slot(x_word), rows * cols)
    var d_keys = ctx.enqueue_create_buffer[DType.uint64](rows)
    var d_codes = ctx.enqueue_create_buffer[DType.int32](rows)
    var d_classes = ctx.enqueue_create_buffer[DType.uint64](rows)
    var blocks = (rows + _TPB - 1) // _TPB
    # one host run per column: column c's distinct keys are copied home while
    # column c + 1 runs (its unique step synchronizes the context)
    var runs = List[List[UInt64]](capacity=n_cat)
    var counts = List[Int](capacity=n_cat)
    for c in range(n_cat):  # driver-loop: over the categorical columns (each column's rows run on the whole device)
        var feature = columns[c]
        ctx.enqueue_function[_cm_key_kernel](
            x, _U64(unsafe_from_address=Int(d_keys.unsafe_ptr())), Int32(rows), Int32(cols), Int32(feature),
            grid_dim=blocks, block_dim=_TPB,
        )
        # synchronizes once for its count word; `LABEL_KIND_U64` sorts the key as it is
        var k = device_unique_inverse_resident(ctx, d_keys, rows, LABEL_KIND_U64, d_codes, d_classes)
        if k > CATEGORY_MAX_VALUES:
            raise Error(
                "mojolearn: categorical feature " + String(feature) + " holds "
                + String(k) + " distinct values; at most "
                + String(CATEGORY_MAX_VALUES) + " are carried (a dense code"
                + " travels as a float32 column value)"
            )
        if k < 1:
            raise Error("fold_x_category_fit: a categorical column of " + String(rows) + " rows has no key")
        ctx.enqueue_function[_cm_code_kernel](
            x, _I(unsafe_from_address=Int(d_codes.unsafe_ptr())), Int32(rows), Int32(cols), Int32(feature),
            grid_dim=blocks, block_dim=_TPB,
        )
        var run = List[UInt64](unsafe_uninit_length=k)
        var head = d_classes.create_sub_buffer[DType.uint64](0, k)
        ctx.enqueue_copy(dst_ptr=_U64(unsafe_from_address=Int(run.unsafe_ptr())), src_buf=head)
        _ = head^
        runs.append(run^)
        counts.append(k)
    ctx.synchronize()
    _ = d_keys^
    _ = d_codes^
    _ = d_classes^
    var total = 2 + 2 * n_cat
    for c in range(n_cat):  # small-loop(the categorical columns): the map's size
        total += counts[c]
    var words = List[UInt64](unsafe_uninit_length=total)
    words[0] = UInt64(CATEGORY_MAP_VERSION)
    words[1] = UInt64(n_cat)
    var at = 2 + 2 * n_cat
    for c in range(n_cat):  # driver-loop: over the categorical columns (one copy of each column's distinct keys)
        words[2 + 2 * c] = UInt64(columns[c])
        words[3 + 2 * c] = UInt64(counts[c])
        unsafe_memcpy(
            dest=MutPointer[UInt64, MutUntrackedOrigin](unsafe_from_address=Int(words.unsafe_ptr()) + 8 * at),
            src=runs[c].unsafe_ptr(),
            count=counts[c],
        )
        at += counts[c]
    _ = runs^
    return words^


def fold_x_category_fit_into(
    ctx: DeviceContext, x_word: Int, n_rows: Int, n_cols: Int, columns: Int, n_cat: Int, dst: Int, capacity: Int
) raises -> Int:
    """XG_SLOT_CATEGORY_FIT's body (gbdt/fit_request.mojo) for an image's
    `slot_category_fit`: the `n_cat` feature indices at `columns` (Int
    words), `fold_x_category_fit` on `ctx`, the stored map's words to the
    host `dst` (at most `capacity`). Returns the words written."""
    var indices = List[Int](unsafe_uninit_length=max(n_cat, 1))
    var src = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=columns)
    for c in range(n_cat):  # small-loop(the categorical feature indices): request words, not row data
        indices[c] = src[c]
    indices.resize(n_cat, 0)
    var words = fold_x_category_fit(ctx, x_word, n_rows, n_cols, indices)
    if len(words) > capacity:
        raise Error("gbdt fold slot category map: " + String(len(words)) + " words for a destination of "
                    + String(capacity))
    unsafe_memcpy(
        dest=MutPointer[UInt64, MutUntrackedOrigin](unsafe_from_address=dst), src=words.unsafe_ptr(),
        count=len(words),
    )
    return len(words)
