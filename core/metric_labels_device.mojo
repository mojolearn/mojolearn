# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The metrics' label and refusal stages over DEVICE-RESIDENT inputs (lane
vdf-xm-labels, 2026-11-24; addendum 3 of the vd-functions briefs: on the GPU
side a resident input is read in place, nothing is downloaded between
device steps, results come back once).

Every entry takes DEVICE addresses (a `core/device_store.mojo` slot's
`ptr(id, need)`, a DeviceRows slot gathered by `take_rows`, or a buffer of
the caller's own), so the x_metrics transport and the `_mojolearn_metrics`
kernel entries hand their slots straight in; nothing here owns a store or a
context.

    metric_encode_resident(ctx, src, n, kind, item, codes, classes_host)
        THE ORDER RULE (sorted distinct values, one dense int32 code per row)
        of n contiguous labels of `item` bytes at device address `src`
        (4-byte words, as a store slot holds them). The codes are written at
        device address `codes` and STAY there; the k classes (the O(k)
        result the Python label step needs) come back once to `classes_host`
        at the labels' own width. Returns k, or -2 for a NaN label.
    metric_gather_resident(ctx, table_host, nt, codes, n, dst)
        `dst[i] = table[codes[i]]` (-1 outside [0, nt)) on the device, the
        per-class table (k words, the caller's label order) uploaded once;
        returns the first row mapped to -1, or -1.
    metric_first_nonfinite_resident(ctx, addr, n)
    metric_first_negative_resident(ctx, addr, n)
        The doors' finiteness and sign refusals of a resident float32 input:
        core/device_scan.mojo's scans (one read on the device, one partial
        per block, the first index by an integer minimum).

SAME BYTES AS THE HOST COLUMN. Labels: the encode is
core/label_encode_device.mojo `device_unique_inverse_resident` (two stable
radix passes, flag, scan, emit), the host twin `score_encode_signed`
(core/labels_score_encode.mojo: the dense table, or the radix route for a
wide range). Both answer the ORDER RULE, whose outputs are integers fixed by
the input alone (the sorted distinct values; code = rank of the row's value
among them), so every route and every split gives the same classes and the
same codes: no fold order exists to change and no bit moves. Gather: integer
table reads and an integer minimum, the host twin `mexp_gather_i32`
(core/metrics_exp_impl.mojo, split over the host pool by the same lane).
Scans: bit tests and an integer minimum, the host twins in
core/host_finite_scan.mojo. The native check is
checks/native_tests/metric_labels_device.mojo.

WHY HOST INPUTS KEEP THE HOST ENCODER. A host label vector on the GPU side
is encoded on the host (the dense table, a measured winner: 3.62x geomean
over the radix route, core/labels_score_encode.mojo VERDICT 2026-11-18)
because the stage after it (x_metrics' host arena, the `_mojolearn_metrics`
entries' host pointers) reads host codes; encoding it here would add an
upload of the labels and a download of the codes per call. These entries are
for inputs that are already on the device, whose codes then stay there.
"""
from max.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.memory import stack_allocation
from sequence.ops import FP
from core.cast_in import cast_raw_bits, cast_w64_bits, narrow_w64, label_kind_of
from core.label_encode_device import device_unique_inverse_resident
from core.device_scan import (
    SCAN_TPB,
    NONFINITE_NONE,
    _scan_blocks,
    min_partials_kernel,
    device_first_nonfinite,
    device_first_negative,
)

comptime _MLD_TPB = 256
comptime _MLD_U64 = MutPointer[UInt64, MutAnyOrigin]
comptime _MLD_I32 = MutPointer[Int32, MutAnyOrigin]
comptime _MLD_F32 = MutPointer[Float32, MutAnyOrigin]
comptime _MLD_BP = MutPointer[UInt8, MutUntrackedOrigin]


@always_inline
def _mld_tid() -> Int:
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


@always_inline
def _mld_blocks(count: Int) -> Int:
    return (count + _MLD_TPB - 1) // _MLD_TPB if count > 0 else 1


def _mld_widen_kernel(raw: FP, n_: Int32, kind: Int32, isz: Int32, dst: _MLD_U64):
    """One thread per row: the row's `isz`-byte item of the packed words at
    byte offset row * isz, widened to its 64-bit label word (the host twin's
    `cast_w64_bits`)."""
    var i = _mld_tid()
    if i >= Int(n_):
        return
    var size = Int(isz)
    dst.unsafe_store(i, cast_w64_bits(cast_raw_bits(raw, i * size, size), Int(kind), size))


def _mld_gather_partial_kernel(
    part: _MLD_I32, table: _MLD_I32, nt_: Int32, codes: _MLD_I32, n_: Int32, dst: _MLD_I32,
):
    """Grid-stride over the rows: `dst[i] = table[codes[i]]`, -1 for a code
    outside [0, nt); one partial per block, the smallest row mapped to -1 or
    `NONFINITE_NONE` (device_scan's partial shape, folded by
    `min_partials_kernel`). Every row is written; the first miss does not
    stop the thread."""
    var n = Int(n_)
    var nt = Int(nt_)
    var red = stack_allocation[
        SCAN_TPB,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var tid = Int(thread_idx.x)
    var stride = Int(grid_dim.x) * SCAN_TPB
    var i = Int(block_idx.x) * SCAN_TPB + tid
    var best = NONFINITE_NONE
    while i < n:
        var c = Int(codes.unsafe_load(i))
        var v = Int32(-1)
        if c >= 0 and c < nt:
            v = table.unsafe_load(c)
        dst.unsafe_store(i, v)
        if v < Int32(0) and Int32(i) < best:
            best = Int32(i)
        i += stride
    red.unsafe_store(tid, best)
    barrier()
    var active = SCAN_TPB // 2
    while active > 0:
        if tid < active:
            var o = red.unsafe_load(tid + active)
            if o < red.unsafe_load(tid):
                red.unsafe_store(tid, o)
        barrier()
        active = active // 2
    if tid == 0:
        part.unsafe_store(Int(block_idx.x), red.unsafe_load(0))


def metric_encode_resident(
    ctx: DeviceContext, src_addr: Int, n: Int, kind: Int, item: Int, codes_addr: Int, classes_addr: Int,
) raises -> Int:
    """The ORDER RULE of n contiguous labels on the device.

    `src_addr`: device address of the labels, `item` bytes each (1, 2, 4 or
    8) of cast kind `kind` (core/cast_in.mojo CAST_KIND_*), packed in 4-byte
    words from byte 0. `codes_addr`: device address of n int32, receives one
    code per row and keeps it. `classes_addr`: HOST address of at least n
    items of `item` bytes; the first k receive the sorted distinct labels at
    their own width. Returns k, or -2 for a NaN label (floats only).
    Two waits: the encoder's one for its status words, and the one for the
    k class words."""
    if n <= 0:
        raise Error("metric labels: no labels to encode")
    if not (item == 1 or item == 2 or item == 4 or item == 8):
        raise Error("metric labels: a label item of " + String(item) + " bytes is not an integer or float width")
    if n > Int(NONFINITE_NONE):
        raise Error("metric labels: " + String(n) + " labels exceed the device encoder's int32 row range")
    var label_kind = label_kind_of(kind, item, "labels")
    var d_src = ctx.enqueue_create_buffer[DType.uint64](n)
    var d_codes = DeviceBuffer[DType.int32](ctx, _MLD_I32(unsafe_from_address=codes_addr), n, owning=False)
    var d_classes = ctx.enqueue_create_buffer[DType.uint64](n)
    ctx.enqueue_function[_mld_widen_kernel](
        FP(unsafe_from_address=src_addr), Int32(n), Int32(kind), Int32(item), d_src.unsafe_ptr(),
        grid_dim=_mld_blocks(n), block_dim=_MLD_TPB,
    )
    var k = device_unique_inverse_resident(ctx, d_src, n, label_kind, d_codes, d_classes)
    if k > 0:
        var h_classes = ctx.enqueue_create_host_buffer[DType.uint64](k)
        var head = d_classes.create_sub_buffer[DType.uint64](0, k)
        ctx.enqueue_copy(dst_ptr=h_classes.unsafe_ptr(), src_buf=head)
        ctx.synchronize()
        var dst = _MLD_BP(unsafe_from_address=classes_addr)
        var words = h_classes.unsafe_ptr()
        for c in range(k):  # driver-loop: over the k distinct classes (the O(k) result, not the rows)
            var w = narrow_w64(words[c], kind, item)
            for b in range(item):  # the item's bytes, little-endian, as the host twin stores them
                dst[c * item + b] = UInt8((w >> UInt64(8 * b)) & UInt64(0xFF))
        _ = head^
        _ = h_classes^
    _ = d_src^
    _ = d_codes^  # not owning: the caller's codes stay on the device
    _ = d_classes^
    return k


def metric_gather_resident(
    ctx: DeviceContext, table_addr: Int, nt: Int, codes_addr: Int, n: Int, dst_addr: Int,
) raises -> Int:
    """`mexp_gather_i32` on the device. `table_addr`: HOST address of the nt
    int32 per-class values (the caller's label order, -1 for a class outside
    it), uploaded once. `codes_addr` and `dst_addr`: DEVICE addresses of n
    int32 each; `dst` keeps the mapped rows. Returns the first row mapped to
    -1, or -1 when there is none. One wait, for the one answer word."""
    if n <= 0:
        return -1
    if n > Int(NONFINITE_NONE):
        raise Error("metric labels: " + String(n) + " rows exceed the device gather's int32 row range")
    var cells = nt if nt > 0 else 1
    var d_table = ctx.enqueue_create_buffer[DType.int32](cells)
    if nt > 0:
        ctx.enqueue_copy(dst_buf=d_table, src_ptr=_MLD_I32(unsafe_from_address=table_addr))
    var blocks = _scan_blocks(n)
    var d_part = ctx.enqueue_create_buffer[DType.int32](blocks)
    var d_first = ctx.enqueue_create_buffer[DType.int32](1)
    ctx.enqueue_function[_mld_gather_partial_kernel](
        d_part.unsafe_ptr(), d_table.unsafe_ptr(), Int32(nt), _MLD_I32(unsafe_from_address=codes_addr), Int32(n),
        _MLD_I32(unsafe_from_address=dst_addr), grid_dim=(blocks, 1, 1), block_dim=(SCAN_TPB, 1, 1),
    )
    ctx.enqueue_function[min_partials_kernel](
        d_first.unsafe_ptr(), d_part.unsafe_ptr(), Int32(blocks), grid_dim=(1, 1, 1), block_dim=(SCAN_TPB, 1, 1),
    )
    var h_first = ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.enqueue_copy(dst_ptr=h_first.unsafe_ptr(), src_buf=d_first)
    ctx.synchronize()
    var first = h_first.unsafe_ptr()[0]
    _ = h_first^
    _ = d_first^
    _ = d_part^
    _ = d_table^
    if first == NONFINITE_NONE:
        return -1
    return Int(first)


def metric_first_nonfinite_resident(ctx: DeviceContext, addr: Int, n: Int) raises -> Int:
    """The first index of a NaN or infinity among the n float32 words at
    device address `addr`, or -1 (`device_first_nonfinite`, by bits)."""
    if n <= 0:
        return -1
    var view = DeviceBuffer[DType.float32](ctx, _MLD_F32(unsafe_from_address=addr), n, owning=False)
    var first = device_first_nonfinite(ctx, view, n)
    _ = view^  # not owning: the caller's words stay on the device
    return first


def metric_first_negative_resident(ctx: DeviceContext, addr: Int, n: Int) raises -> Int:
    """The first index of a negative float32 word (sign set, magnitude
    nonzero) at device address `addr`, or -1 (`device_first_negative`). Run
    `metric_first_nonfinite_resident` first on words that may hold NaN."""
    if n <= 0:
        return -1
    var view = DeviceBuffer[DType.float32](ctx, _MLD_F32(unsafe_from_address=addr), n, owning=False)
    var first = device_first_negative(ctx, view, n)
    _ = view^  # not owning: the caller's words stay on the device
    return first
