# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Categorical code columns, validated, maxed and densified ON THE DEVICE
(lane/cpu3-gbdt-b, 2026-10-04).

`gbdt/train.mojo::train` walked every categorical column on the host up to
three times: the two cardinality pre-passes (`has_permutation_features`,
`ordered_ctr_column`), the per-feature code build with its density check,
and the one-hot cardinality scan in `_quantize_training_columns`. Each is now
one upload of the column and one or two kernels; the host reads back three
words (and, for the code build, the codes the host CTR calcers still take).

THE CONTRACT is `dense_category_code` (`gbdt/models/ctr_value_table.mojo`):
a finite, non-negative value below 2^32 with no fractional part. A row that
breaks it is found on the device (the lowest such row) and the host then
calls `dense_category_code` on THAT ONE VALUE, so the refusal is the same
message, same row.

THE MAXIMUM is an `Atomic.max` over the Float32 BIT PATTERN as Int32: for
non-negative floats the IEEE order and the integer order of the bits agree,
so the largest bit pattern is the largest code, exactly, up to 2^32 - 1.
Integer and compare work only: same result on every vendor and the host.

WORDS (Int32): 0 lowest invalid row (`CAT_NO_ROW` = none), 1 the largest
valid code's bit pattern (0 = code 0), 2 the lowest absent category below
the maximum (`CAT_NO_ROW` = dense).
"""

from std.atomic import Atomic
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import bitcast
from max.gpu.host import DeviceBuffer, DeviceContext
from core.device_zero import enqueue_fill
from gbdt.models.ctr_value_table import dense_category_code

comptime CAT_SCAN_TPB = 256
comptime CAT_NO_ROW = Int32(2147483647)
comptime _CAT_WORDS = 3


@fieldwise_init
struct CatColumnScan(Copyable, Movable):
    var max_code: Int
    #: the lowest category in `0..max_code` no row holds, or -1 (dense)
    var first_absent: Int


def cat_code_scan_kernel(
    x: MutPointer[Float32, MutAnyOrigin],
    codes: MutPointer[UInt32, MutAnyOrigin],
    words: MutPointer[Int32, MutAnyOrigin],
    n_rows_in: Int32,
    write_codes: Int32,
    lenient: Int32,
):
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if r >= Int(n_rows_in):
        return
    var v = x.unsafe_load(r)
    if lenient != 0:
        # the one-hot scan's `max(Int(v))` from 0: truncation is monotone,
        # so the largest non-negative value's bits give the largest code;
        # NaN and negatives never raise the floor
        if v >= Float32(0.0):
            _ = Atomic.max(words.unsafe_offset(1), bitcast[DType.int32](v))
        return
    # `dense_category_code`: NaN fails `v >= 0`, +inf fails the bound
    if not (v >= Float32(0.0) and v < Float32(4294967296.0)):
        _ = Atomic.min(words.unsafe_offset(0), Int32(r))
        return
    var u = UInt32(v)
    if Float32(u) != v:
        _ = Atomic.min(words.unsafe_offset(0), Int32(r))
        return
    _ = Atomic.max(words.unsafe_offset(1), bitcast[DType.int32](v))
    if write_codes != 0:
        codes.unsafe_store(r, u)


def cat_mark_seen_kernel(
    codes: MutPointer[UInt32, MutAnyOrigin],
    seen: MutPointer[UInt8, MutAnyOrigin],
    n_rows_in: Int32,
):
    """`seen[code] = 1`; concurrent writers store the same byte."""
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if r >= Int(n_rows_in):
        return
    seen.unsafe_store(Int(codes.unsafe_load(r)), UInt8(1))


def cat_first_absent_kernel(
    seen: MutPointer[UInt8, MutAnyOrigin],
    words: MutPointer[Int32, MutAnyOrigin],
    n_codes_in: Int32,
):
    var c = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if c >= Int(n_codes_in):
        return
    if seen.unsafe_load(c) == UInt8(0):
        _ = Atomic.min(words.unsafe_offset(2), Int32(c))


def _grid(n: Int) -> Int:
    return (n + CAT_SCAN_TPB - 1) // CAT_SCAN_TPB


def _scan_column(
    ctx: DeviceContext,
    src: MutPointer[Float32, MutUntrackedOrigin],
    n_rows: Int,
    feature: Int,
    want_codes: Bool,
    mut codes_out: List[UInt32],
    lenient: Bool = False,
) raises -> CatColumnScan:
    var d_codes = ctx.enqueue_create_buffer[DType.uint32](
        n_rows if want_codes and n_rows > 0 else 1
    )
    return _scan_column_into(
        ctx, src, n_rows, feature, want_codes, True, codes_out, d_codes,
        lenient,
    )


def _scan_column_into(
    ctx: DeviceContext,
    src: MutPointer[Float32, MutUntrackedOrigin],
    n_rows: Int,
    feature: Int,
    want_codes: Bool,
    want_host_codes: Bool,
    mut codes_out: List[UInt32],
    mut d_codes: DeviceBuffer[DType.uint32],
    lenient: Bool = False,
) raises -> CatColumnScan:
    """`_scan_column` writing the codes into the caller's `d_codes` (at
    least `n_rows` words when `want_codes`), so a caller can keep them
    resident (lane cpu4-gbdt); `want_host_codes` also reads them back into
    `codes_out`."""
    if n_rows <= 0:
        return CatColumnScan(0, -1)
    var d_x = ctx.enqueue_create_buffer[DType.float32](n_rows)
    ctx.enqueue_copy(dst_buf=d_x, src_ptr=src)
    var h = ctx.enqueue_create_host_buffer[DType.int32](_CAT_WORDS)
    h.unsafe_ptr().unsafe_store(0, CAT_NO_ROW)
    h.unsafe_ptr().unsafe_store(1, Int32(0))
    h.unsafe_ptr().unsafe_store(2, CAT_NO_ROW)
    var d_w = ctx.enqueue_create_buffer[DType.int32](_CAT_WORDS)
    ctx.enqueue_copy(dst_buf=d_w, src_ptr=h.unsafe_ptr())
    ctx.enqueue_function[cat_code_scan_kernel](
        d_x.unsafe_ptr(),
        d_codes.unsafe_ptr(),
        d_w.unsafe_ptr(),
        Int32(n_rows),
        Int32(1) if want_codes else Int32(0),
        Int32(1) if lenient else Int32(0),
        grid_dim=(_grid(n_rows), 1, 1),
        block_dim=(CAT_SCAN_TPB, 1, 1),
    )
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=d_w)
    ctx.synchronize()
    var bad = h.unsafe_ptr().unsafe_load(0)
    if bad != CAT_NO_ROW:
        # the host contract on the ONE offending value: same words, same row
        _ = dense_category_code(src.unsafe_load(Int(bad)), feature, Int(bad))
        raise Error(
            "categorical feature " + String(feature) + " row "
            + String(Int(bad)) + " is not a dense category code"
        )
    var max_f = bitcast[DType.float32](h.unsafe_ptr().unsafe_load(1))
    var max_code = 2147483647 if max_f >= Float32(2147483647.0) else Int(max_f)
    var first_absent = -1
    if want_codes:
        var n_codes = max_code + 1
        if n_codes > Int(CAT_NO_ROW):
            raise Error(
                "categorical feature " + String(feature)
                + " has a category code above 2^31 - 2"
            )
        var d_seen = ctx.enqueue_create_buffer[DType.uint8](n_codes)
        enqueue_fill(ctx, d_seen, UInt8(0))
        ctx.enqueue_function[cat_mark_seen_kernel](
            d_codes.unsafe_ptr(),
            d_seen.unsafe_ptr(),
            Int32(n_rows),
            grid_dim=(_grid(n_rows), 1, 1),
            block_dim=(CAT_SCAN_TPB, 1, 1),
        )
        ctx.enqueue_function[cat_first_absent_kernel](
            d_seen.unsafe_ptr(),
            d_w.unsafe_ptr(),
            Int32(n_codes),
            grid_dim=(_grid(n_codes), 1, 1),
            block_dim=(CAT_SCAN_TPB, 1, 1),
        )
        if want_host_codes:
            codes_out.resize(n_rows, UInt32(0))
            ctx.enqueue_copy(
                dst_ptr=codes_out.unsafe_ptr(),
                src_buf=d_codes.create_sub_buffer[DType.uint32](0, n_rows),
            )
        ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=d_w)
        ctx.synchronize()
        var absent = h.unsafe_ptr().unsafe_load(2)
        if absent != CAT_NO_ROW:
            first_absent = Int(absent)
        _ = d_seen^
    _ = d_x^
    _ = d_w^
    _ = h^
    return CatColumnScan(max_code, first_absent)


def cat_column_max_code(
    ctx: DeviceContext,
    src: MutPointer[Float32, MutUntrackedOrigin],
    n_rows: Int,
    feature: Int,
) raises -> Int:
    """The largest dense code in `src[0:n_rows]` (0 for an empty column),
    refusing an invalid code as `dense_category_code` does."""
    var unused = List[UInt32]()
    return _scan_column(ctx, src, n_rows, feature, False, unused).max_code


def cat_column_codes(
    ctx: DeviceContext,
    src: MutPointer[Float32, MutUntrackedOrigin],
    n_rows: Int,
    feature: Int,
    mut codes_out: List[UInt32],
) raises -> CatColumnScan:
    """Validate, convert to UInt32 codes (into `codes_out`, `n_rows` long),
    and find the maximum and the lowest absent category, all on the device.
    The codes come back to the host only because the CTR calcers still take
    a host list."""
    return _scan_column(ctx, src, n_rows, feature, True, codes_out)


def cat_column_codes_device(
    ctx: DeviceContext,
    src: MutPointer[Float32, MutUntrackedOrigin],
    n_rows: Int,
    feature: Int,
    mut d_codes: DeviceBuffer[DType.uint32],
) raises -> CatColumnScan:
    """`cat_column_codes` with the codes left ONLY on the device, in the
    caller's `d_codes` (`n_rows` words), nothing read back but the three
    scan words (lane cpu4-gbdt, the tensor CTR fit)."""
    var unused = List[UInt32]()
    return _scan_column_into(
        ctx, src, n_rows, feature, True, False, unused, d_codes
    )


def cat_column_codes_resident(
    ctx: DeviceContext,
    src: MutPointer[Float32, MutUntrackedOrigin],
    n_rows: Int,
    feature: Int,
    mut codes_out: List[UInt32],
    mut d_codes: DeviceBuffer[DType.uint32],
) raises -> CatColumnScan:
    """`cat_column_codes` that ALSO leaves the codes on the device, in the
    caller's `d_codes` (`n_rows` words), for the CTR calcers to read in
    place (lane cpu4-gbdt). The host copy in `codes_out` is still written:
    the apply-time `build_ctr_tables` takes it."""
    return _scan_column_into(
        ctx, src, n_rows, feature, True, True, codes_out, d_codes
    )


def onehot_column_max_code(
    ctx: DeviceContext,
    src: MutPointer[Float32, MutUntrackedOrigin],
    n_rows: Int,
    column: Int,
) raises -> Int:
    """`_quantize_training_columns`' one-hot scan, `max(0, Int(v))` over the
    column, without the dense-code refusal (that scan never refused). A
    value at or past 2^31 (or +inf) returns 2^31 - 1, which the caller's
    255-category limit refuses."""
    var unused = List[UInt32]()
    return _scan_column(ctx, src, n_rows, column, False, unused, True).max_code
