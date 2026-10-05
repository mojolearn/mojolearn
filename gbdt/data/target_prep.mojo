# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The training targets and weights, checked and folded ON THE DEVICE
(lane/cpu3-gbdt-b, 2026-10-04).

`gbdt/train.mojo::train` used to walk `y` and `sample_weight` on the host
four times: the MultiRMSE finiteness check, the multiclass label check, the
class-count scan (twice) and the per-row weight fold into a host staging
buffer. All of them are now one upload of the caller's arrays straight into
the fit's `targets` and `weights` buffers, one scan kernel whose result is
FIVE Int32 words read back, and (only with `class_weights`) one elementwise
kernel that multiplies the class weight in place.

BITS. The weight is the same single product the host loop formed,
`w * class_weights[cls]` with `w` the sample weight or 1.0, so every column
(NVIDIA, AMD, Apple and the host oracle, which keeps its own fold) holds the
same bits. The scan is integer and compare work only.

SCAN WORDS (Int32, `TARGET_NO_ROW` = none):
  0  the lowest flat index of a non-finite value over every target plane
  1  the lowest row whose multiclass label is non-finite, negative or
     `>= n_rows` (the host check's first clause)
  2  the lowest row whose multiclass label is in range but not an integer
  3  the largest valid multiclass label (-1 when none)
  4  the lowest row with a negative sample weight
The host check raised at the FIRST offending row; on one row it tested the
range before the integer, so `first_bad_label` below picks word 1 on a tie.
"""

from std.atomic import Atomic
from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from core.device_zero import enqueue_fill

comptime TARGET_PREP_TPB = 256
comptime TARGET_NO_ROW = Int32(2147483647)
comptime TARGET_SCAN_WORDS = 5
comptime _F32_MAX = Float32(3.4028234663852886e38)


@fieldwise_init
struct TargetScan(Copyable, Movable):
    """The five scan words, as Ints (-1 = none)."""

    var first_nonfinite: Int
    var first_label_range: Int
    var first_label_nonint: Int
    var max_label: Int
    var first_negative_weight: Int

    def first_bad_label(self) -> Int:
        """The row the host label check would have named first, or -1."""
        if self.first_label_range < 0:
            return self.first_label_nonint
        if self.first_label_nonint < 0:
            return self.first_label_range
        return min(self.first_label_range, self.first_label_nonint)

    def first_bad_label_is_range(self) -> Bool:
        """True when that row failed the range clause (it is tested first)."""
        var r = self.first_bad_label()
        return r >= 0 and r == self.first_label_range


def target_scan_kernel(
    y: MutPointer[Float32, MutAnyOrigin],
    sw: MutPointer[Float32, MutAnyOrigin],
    words: MutPointer[Int32, MutAnyOrigin],
    n_rows_in: Int32,
    target_dim_in: Int32,
    multiclass: Int32,
    use_sw: Int32,
):
    """One thread per flat target entry; the label and weight clauses run
    on the first plane only (one per row)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var n = Int(n_rows_in)
    if i >= n * Int(target_dim_in):
        return
    var v = y.unsafe_load(i)
    # `not isfinite(v)`: NaN fails both compares, +-inf fails one
    if not (v >= -_F32_MAX and v <= _F32_MAX):
        _ = Atomic.min(words.unsafe_offset(0), Int32(i))
    if i >= n:
        return
    if multiclass != 0:
        # the host check: `not isfinite(label) or label < 0 or
        # label >= Float32(n_rows)`; NaN and -inf fail `v >= 0`, +inf fails
        # `v < n`
        if not (v >= Float32(0.0) and v < Float32(n)):
            _ = Atomic.min(words.unsafe_offset(1), Int32(i))
        else:
            # `Float32(Int(label)) != label`; the label is in [0, n_rows),
            # so the Int32 truncation is the host's Int truncation
            var iv = Int32(v)
            if Float32(iv) != v:
                _ = Atomic.min(words.unsafe_offset(2), Int32(i))
            else:
                _ = Atomic.max(words.unsafe_offset(3), iv)
    if use_sw != 0:
        if sw.unsafe_load(i) < Float32(0.0):
            _ = Atomic.min(words.unsafe_offset(4), Int32(i))


def class_weight_fold_kernel(
    y: MutPointer[Float32, MutAnyOrigin],
    w: MutPointer[Float32, MutAnyOrigin],
    cw: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    multiclass: Int32,
):
    """`w[r] = w[r] * class_weights[cls]`, the host loop's one product.
    `cls` is the dense class code for the multiclass family and the
    binarized target (`y > 0.5`) otherwise."""
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if r >= Int(n_rows_in):
        return
    var v = y.unsafe_load(r)
    var cls: Int
    if multiclass != 0:
        cls = Int(Int32(v))
    else:
        cls = 1 if v > Float32(0.5) else 0
    w.unsafe_store(r, w.unsafe_load(r) * cw.unsafe_load(cls))


def upload_and_scan_targets(
    ctx: DeviceContext,
    y: List[Float32],
    sample_weight: List[Float32],
    n_rows: Int,
    target_dim: Int,
    multiclass: Bool,
    mut targets: DeviceBuffer[DType.float32],
    mut weights: DeviceBuffer[DType.float32],
) raises -> TargetScan:
    """Copy `y` into `targets` (`n_rows * target_dim`) and the sample
    weights (or 1.0) into `weights` (`n_rows`), then scan them. The caller
    checked both lengths. One 20 B readback."""
    var use_sw = len(sample_weight) > 0
    if n_rows * target_dim > 0:
        ctx.enqueue_copy(dst_buf=targets, src_ptr=y.unsafe_ptr())
    if use_sw:
        if n_rows > 0:
            ctx.enqueue_copy(dst_buf=weights, src_ptr=sample_weight.unsafe_ptr())
    else:
        _fill_ones(ctx, weights, n_rows)
    var h = ctx.enqueue_create_host_buffer[DType.int32](TARGET_SCAN_WORDS)
    for k in range(TARGET_SCAN_WORDS):  # small-loop(TARGET_SCAN_WORDS: scan words): five comptime words, no data
        h.unsafe_ptr().unsafe_store(k, Int32(-1) if k == 3 else TARGET_NO_ROW)
    var d = ctx.enqueue_create_buffer[DType.int32](TARGET_SCAN_WORDS)
    ctx.enqueue_copy(dst_buf=d, src_ptr=h.unsafe_ptr())
    var total = n_rows * target_dim
    if total > 0:
        ctx.enqueue_function[target_scan_kernel](
            targets.unsafe_ptr(),
            weights.unsafe_ptr(),
            d.unsafe_ptr(),
            Int32(n_rows),
            Int32(target_dim),
            Int32(1) if multiclass else Int32(0),
            Int32(1) if use_sw else Int32(0),
            grid_dim=((total + TARGET_PREP_TPB - 1) // TARGET_PREP_TPB, 1, 1),
            block_dim=(TARGET_PREP_TPB, 1, 1),
        )
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=d)
    ctx.synchronize()
    # the caller's lists stay alive past the drain (step-33 race class)
    _ = len(y)
    _ = len(sample_weight)
    var w0 = h.unsafe_ptr().unsafe_load(0)
    var w1 = h.unsafe_ptr().unsafe_load(1)
    var w2 = h.unsafe_ptr().unsafe_load(2)
    var w3 = h.unsafe_ptr().unsafe_load(3)
    var w4 = h.unsafe_ptr().unsafe_load(4)
    _ = d^
    _ = h^
    return TargetScan(
        -1 if w0 == TARGET_NO_ROW else Int(w0),
        -1 if w1 == TARGET_NO_ROW else Int(w1),
        -1 if w2 == TARGET_NO_ROW else Int(w2),
        Int(w3),
        -1 if w4 == TARGET_NO_ROW else Int(w4),
    )


def fold_class_weights(
    ctx: DeviceContext,
    mut targets: DeviceBuffer[DType.float32],
    mut weights: DeviceBuffer[DType.float32],
    class_weights: List[Float32],
    n_rows: Int,
    multiclass: Bool,
) raises:
    """Multiply `class_weights[cls(y[r])]` into `weights[r]` in place. The
    caller checked the class-weight count against the labels."""
    if n_rows <= 0 or len(class_weights) == 0:
        return
    var d_cw = ctx.enqueue_create_buffer[DType.float32](len(class_weights))
    ctx.enqueue_copy(dst_buf=d_cw, src_ptr=class_weights.unsafe_ptr())
    ctx.enqueue_function[class_weight_fold_kernel](
        targets.unsafe_ptr(),
        weights.unsafe_ptr(),
        d_cw.unsafe_ptr(),
        Int32(n_rows),
        Int32(1) if multiclass else Int32(0),
        grid_dim=((n_rows + TARGET_PREP_TPB - 1) // TARGET_PREP_TPB, 1, 1),
        block_dim=(TARGET_PREP_TPB, 1, 1),
    )
    ctx.synchronize()
    _ = len(class_weights)
    _ = d_cw^


def _fill_ones(
    ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], n: Int
) raises:
    if n > 0:
        enqueue_fill(ctx, buf, Float32(1.0))


def differs_from_first_kernel(
    v: MutPointer[Float32, MutAnyOrigin],
    flag: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
):
    """`flag[0] = 1` when some `v[r] != v[0]` (NaN compares unequal)."""
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if r < 1 or r >= Int(n_in):
        return
    if v.unsafe_load(r) != v.unsafe_load(0):
        _ = Atomic.max(flag, Int32(1))


def device_all_equal_first(
    ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], n: Int
) raises -> Bool:
    """True when every `buf[r] == buf[0]` for `r < n` (the held-out set's
    constant-target test). One 4 B readback."""
    if n <= 1:
        return True
    var h = ctx.enqueue_create_host_buffer[DType.int32](1)
    h.unsafe_ptr().unsafe_store(0, Int32(0))
    var d = ctx.enqueue_create_buffer[DType.int32](1)
    ctx.enqueue_copy(dst_buf=d, src_ptr=h.unsafe_ptr())
    ctx.enqueue_function[differs_from_first_kernel](
        buf.unsafe_ptr(),
        d.unsafe_ptr(),
        Int32(n),
        grid_dim=((n + TARGET_PREP_TPB - 1) // TARGET_PREP_TPB, 1, 1),
        block_dim=(TARGET_PREP_TPB, 1, 1),
    )
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=d)
    ctx.synchronize()
    var differs = h.unsafe_ptr().unsafe_load(0) != Int32(0)
    _ = d^
    _ = h^
    return not differs
