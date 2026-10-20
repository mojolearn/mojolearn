# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`device_unique_inverse` over the caller's label array as it is (lane
ag-thin-glue, 2026-10-19): any accepted dtype and stride (core/cast_in.mojo),
the sorted classes written back in that same dtype. Python passes the
user's labels and an output array of their dtype; no cast happens there.

Steps, on the device: the label bytes go up once; one thread per row
widens its item to the 64-bit label word (`cast_w64_bits`: float64 bits,
or int64 bits for integers and bool); `device_unique_inverse_resident`
sorts, flags, scans and emits as before (the same codes and the same
first-seen class words); one thread per output word narrows the k class
words back to the caller's items (`narrow_w64`, exact for every word the
widening produced); the codes and the classes' bytes come down.

The host twin is core/label_cast.mojo (the same element functions around
`host_unique_inverse`), so both columns give the same bytes."""

from std.gpu import block_dim, block_idx, thread_idx
from std.memory import bitcast, memcpy
from max.gpu.host import DeviceBuffer, DeviceContext
from sequence.ops import FP
from core.cast_in import CastIn, cast_raw_bits, cast_w64_bits, narrow_w64
from core.label_encode_device import device_unique_inverse_resident

comptime _U64 = MutPointer[UInt64, MutAnyOrigin]
comptime _BP = MutPointer[UInt8, MutUntrackedOrigin]
comptime _TPB = 256


@always_inline
def _tid() -> Int:
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


@always_inline
def _blocks(count: Int) -> Int:
    return (count + _TPB - 1) // _TPB if count > 0 else 1


def _widen_kernel(raw: FP, n_: Int32, kind: Int32, isz: Int32, s0: Int32, dst: _U64):
    var i = _tid()
    if i >= Int(n_):
        return
    var size = Int(isz)
    dst.unsafe_store(i, cast_w64_bits(cast_raw_bits(raw, i * Int(s0), size), Int(kind), size))


def _narrow_kernel(cls: _U64, k_: Int32, kind: Int32, isz: Int32, dst: FP, words_: Int32):
    var t = _tid()
    if t >= Int(words_):
        return
    var size = Int(isz)
    var acc = UInt64(0)
    if size == 8:
        var v = narrow_w64(cls.unsafe_load(t >> 1), Int(kind), size)
        acc = (v >> UInt64(32 * (t & 1))) & UInt64(0xFFFFFFFF)
    else:
        var per = 4 // size
        for j in range(per):  # the 1, 2 or 4 classes of this word
            var i = t * per + j
            if i < Int(k_):
                acc |= narrow_w64(cls.unsafe_load(i), Int(kind), size) << UInt64(j * size * 8)
    dst.unsafe_store(t, bitcast[DType.float32](UInt32(acc & UInt64(0xFFFFFFFF))))


def device_unique_inverse_cast(
    ctx: DeviceContext, src: CastIn, label_kind: Int, classes_addr: Int, codes_addr: Int,
) raises -> Int:
    """Sorted distinct labels of the caller's (n, 1) label array `src` into
    host classes_addr (n slots of src's item size, the first k written) and
    one int32 code per row into host codes_addr. Returns k, or -2 for a NaN
    label. `src` must have passed CastIn.check with cols == 1; label_kind is
    core/cast_in.mojo label_kind_of's."""
    var n = src.rows
    var size = src.itemsize
    var nbytes = src.nbytes()
    var words = src.words()
    # the bytes as they are, the tail word zero-padded
    var h_raw = ctx.enqueue_create_host_buffer[DType.float32](words)
    ctx.synchronize()
    h_raw.unsafe_ptr().unsafe_store(words - 1, Float32(0))
    memcpy(dest=_BP(unsafe_from_address=Int(h_raw.unsafe_ptr())), src=_BP(unsafe_from_address=src.addr), count=nbytes)
    var d_raw = ctx.enqueue_create_buffer[DType.float32](words)
    ctx.enqueue_copy(dst_buf=d_raw, src_ptr=h_raw.unsafe_ptr())
    var d_src = ctx.enqueue_create_buffer[DType.uint64](n)
    var d_codes = ctx.enqueue_create_buffer[DType.int32](n)
    var d_classes = ctx.enqueue_create_buffer[DType.uint64](n)
    ctx.enqueue_function[_widen_kernel](
        FP(unsafe_from_address=Int(d_raw.unsafe_ptr())), Int32(n), Int32(src.kind), Int32(size), Int32(src.s0),
        d_src.unsafe_ptr(), grid_dim=_blocks(n), block_dim=_TPB,
    )
    var k = device_unique_inverse_resident(ctx, d_src, n, label_kind, d_codes, d_classes)
    if k < 0:
        _ = h_raw^
        _ = d_raw^
        _ = d_src^
        _ = d_codes^
        _ = d_classes^
        return k
    var out_words = (k * size + 3) // 4
    var d_out = ctx.enqueue_create_buffer[DType.float32](out_words)
    ctx.enqueue_function[_narrow_kernel](
        d_classes.unsafe_ptr(), Int32(k), Int32(src.kind), Int32(size),
        FP(unsafe_from_address=Int(d_out.unsafe_ptr())), Int32(out_words),
        grid_dim=_blocks(out_words), block_dim=_TPB,
    )
    var h_out = ctx.enqueue_create_host_buffer[DType.float32](out_words)
    ctx.enqueue_copy(dst_ptr=h_out.unsafe_ptr(), src_buf=d_out)
    ctx.enqueue_copy(dst_ptr=MutPointer[Int32, MutAnyOrigin](unsafe_from_address=codes_addr), src_buf=d_codes)
    ctx.synchronize()
    memcpy(dest=_BP(unsafe_from_address=classes_addr), src=_BP(unsafe_from_address=Int(h_out.unsafe_ptr())), count=k * size)
    _ = h_raw^
    _ = h_out^
    _ = d_raw^
    _ = d_out^
    _ = d_src^
    _ = d_codes^
    _ = d_classes^
    return k
