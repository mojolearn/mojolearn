# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Device twins of the shared input helpers (lane cpu2-l1-input, 2026-10-04;
cpu re-audit section 2 "Shared input, validation and layout").

The GPU base binding (`bindings/_mojolearn.mojo`, through
`bindings/hotpath_device.mojo`) runs these instead of the host loops it used
to run on every fit, transform and predict entry:

  device_all_finite        `all_finite_f32` / `all_finite_f64`
  device_transpose_to_f32  `transpose_f32` / `cast_colmajor_f64_to_f32`
  device_strided_copy      `strided_copy_bytes`
  device_check_lengths     `check_lengths_i64`
  device_ragged_rows       `ragged_rows_bytes`

Each takes the SAME host addresses its host helper takes, uploads them, does
the per-element work in kernels and downloads the result (one status word for
the predicates). Bit moves, integer compares and the float64 narrowing by
`checks/soft_f64.mojo` words only, so the bytes are the host helper's on
NVIDIA, AMD and Apple. The host helpers stay: they are the host column (core
host binding, CPU inference bindings) and the refusal path (an input a twin
does not cover, or would read out of bounds, reruns the host helper, which
raises or answers in its own words).

The predicates test BITS, never a float compare (contract section 8: Metal
flushes compare operands): non-finite is `|bits| >= 0x7F800000` (float32) or
`|bits| >= 0x7FF0000000000000` (float64), which is `not isfinite(x)` for every
value, subnormals included.

No shared memory (no page to gate); flags are plain stores of one value or
Int32 atomics (the Apple GPU has no 64-bit atomic), so every count here is
below 2^31 (`IND_MAX_N`).
"""

from std.atomic import Atomic
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.memory import bitcast
from std.sys.info import size_of
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.soft_f64 import sf64_is_nan, sf64_to_f32
from core.device_zero import enqueue_fill

comptime IND_TPB = 256
comptime IND_MAX_N = 2147483000
comptime IND_REDUCE_BLOCKS = 1024
#: The "no hit" word of `device_check_lengths`'s first-index minimum.
comptime IND_NONE: Int32 = 2147483647

comptime _U8 = MutPointer[UInt8, MutAnyOrigin]
comptime _U32 = MutPointer[UInt32, MutAnyOrigin]
comptime _U64 = MutPointer[UInt64, MutAnyOrigin]
comptime _I32 = MutPointer[Int32, MutAnyOrigin]
comptime _I64 = MutPointer[Int64, MutAnyOrigin]


@always_inline
def _tid() -> Int:
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


@always_inline
def _blocks(count: Int) -> Int:
    return (count + IND_TPB - 1) // IND_TPB if count > 0 else 1


@always_inline
def _stride_grid(n: Int) -> Int:
    return max(1, min(IND_REDUCE_BLOCKS, (n + IND_TPB - 1) // IND_TPB))


def _read_word(ctx: DeviceContext, mut d_status: DeviceBuffer[DType.int32]) raises -> Int:
    """Synchronizes and returns word 0 of the status buffer."""
    var h = ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=d_status)
    ctx.synchronize()
    var v = Int(h.unsafe_ptr()[0])
    _ = h^
    return v


# ===========================================================================
# all_finite_f32 / all_finite_f64
# ===========================================================================


def _nonfinite_u32_kernel(src: _U32, n_: Int32, status: _I32):
    """status[0] = 1 when any float32 word of src[0:n] is NaN or infinite."""
    var i = _tid()
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    var n = Int(n_)
    var hit = False
    while i < n:
        if (src.unsafe_load(i) & UInt32(0x7FFFFFFF)) >= UInt32(0x7F800000):
            hit = True
            break
        i += stride
    if hit:
        status.unsafe_store(0, Int32(1))


def _nonfinite_u64_kernel(src: _U64, n_: Int32, status: _I32):
    """status[0] = 1 when any float64 word of src[0:n] is NaN or infinite."""
    var i = _tid()
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    var n = Int(n_)
    var hit = False
    while i < n:
        if (src.unsafe_load(i) & UInt64(0x7FFFFFFFFFFFFFFF)) >= UInt64(0x7FF0000000000000):
            hit = True
            break
        i += stride
    if hit:
        status.unsafe_store(0, Int32(1))


def device_all_finite(ctx: DeviceContext, addr: Int, n: Int, is_f64: Bool) raises -> Bool:
    """`all_finite_f32` / `all_finite_f64` (`1 <= n <= IND_MAX_N`): True
    when every value is finite. One status word comes back."""
    var d_status = ctx.enqueue_create_buffer[DType.int32](1)
    enqueue_fill(ctx, d_status, Int32(0))
    if is_f64:
        var d = ctx.enqueue_create_buffer[DType.uint64](n)
        ctx.enqueue_copy(dst_buf=d, src_ptr=_U64(unsafe_from_address=addr))
        ctx.enqueue_function[_nonfinite_u64_kernel](
            d.unsafe_ptr(), Int32(n), d_status.unsafe_ptr(),
            grid_dim=_stride_grid(n), block_dim=IND_TPB,
        )
        var bad = _read_word(ctx, d_status)
        _ = d^
        _ = d_status^
        return bad == 0
    var d = ctx.enqueue_create_buffer[DType.uint32](n)
    ctx.enqueue_copy(dst_buf=d, src_ptr=_U32(unsafe_from_address=addr))
    ctx.enqueue_function[_nonfinite_u32_kernel](
        d.unsafe_ptr(), Int32(n), d_status.unsafe_ptr(),
        grid_dim=_stride_grid(n), block_dim=IND_TPB,
    )
    var bad = _read_word(ctx, d_status)
    _ = d^
    _ = d_status^
    return bad == 0


# ===========================================================================
# transpose_f32 / cast_colmajor_f64_to_f32
# ===========================================================================


@always_inline
def _narrow_word(w: UInt64) -> UInt32:
    """The float32 word a hardware double-to-float conversion gives: round to
    nearest even (`sf64_to_f32`); a NaN keeps its sign and the top 22 payload
    bits and is quieted. The same rule as `core/hotpath_device.mojo`'s
    `_narrow_f64_kernel` (the device `cast_f64_to_f32`)."""
    if sf64_is_nan(w):
        var sign = UInt32((w >> 63) << 31)
        var payload = UInt32((w >> 29) & UInt64(0x003FFFFF))
        return sign | UInt32(0x7FC00000) | payload
    return bitcast[DType.uint32](sf64_to_f32(w))


def _transpose_u32_kernel(src: _U32, nr_: Int32, nc_: Int32, dst: _U32):
    """dst[c * nr + r] = src[r * nc + c]: one thread per output word."""
    var i = _tid()
    var nr = Int(nr_)
    var nc = Int(nc_)
    if i >= nr * nc:
        return
    var c = i // nr
    var r = i - c * nr
    dst.unsafe_store(i, src.unsafe_load(r * nc + c))


def _transpose_narrow_kernel(src: _U64, nr_: Int32, nc_: Int32, dst: _U32):
    """dst[c * nr + r] = Float32(src[r * nc + c]) as words."""
    var i = _tid()
    var nr = Int(nr_)
    var nc = Int(nc_)
    if i >= nr * nc:
        return
    var c = i // nr
    var r = i - c * nr
    dst.unsafe_store(i, _narrow_word(src.unsafe_load(r * nc + c)))


def device_transpose_to_f32(
    ctx: DeviceContext, src_addr: Int, dst_addr: Int, nr: Int, nc: Int, is_f64: Bool,
) raises:
    """`transpose_f32` (float32 in) or `cast_colmajor_f64_to_f32` (float64
    in): the C-order `[nr, nc]` matrix at `src` goes up, its column-major
    float32 layout comes down to `dst` (`1 <= nr * nc <= IND_MAX_N`)."""
    var n = nr * nc
    var d_d = ctx.enqueue_create_buffer[DType.uint32](n)
    if is_f64:
        var d_s = ctx.enqueue_create_buffer[DType.uint64](n)
        ctx.enqueue_copy(dst_buf=d_s, src_ptr=_U64(unsafe_from_address=src_addr))
        ctx.enqueue_function[_transpose_narrow_kernel](
            d_s.unsafe_ptr(), Int32(nr), Int32(nc), d_d.unsafe_ptr(),
            grid_dim=_blocks(n), block_dim=IND_TPB,
        )
        ctx.enqueue_copy(dst_ptr=_U32(unsafe_from_address=dst_addr), src_buf=d_d)
        ctx.synchronize()
        _ = d_s^
        _ = d_d^
        return
    var d_s = ctx.enqueue_create_buffer[DType.uint32](n)
    ctx.enqueue_copy(dst_buf=d_s, src_ptr=_U32(unsafe_from_address=src_addr))
    ctx.enqueue_function[_transpose_u32_kernel](
        d_s.unsafe_ptr(), Int32(nr), Int32(nc), d_d.unsafe_ptr(),
        grid_dim=_blocks(n), block_dim=IND_TPB,
    )
    ctx.enqueue_copy(dst_ptr=_U32(unsafe_from_address=dst_addr), src_buf=d_d)
    ctx.synchronize()
    _ = d_s^
    _ = d_d^


# ===========================================================================
# strided_copy_bytes
# ===========================================================================


def _strided_kernel[dt: DType](
    src: MutPointer[Scalar[dt], MutAnyOrigin],
    dst: MutPointer[Scalar[dt], MutAnyOrigin],
    dims: _I64,
    nd_: Int32,
    total_: Int32,
    sbase: Int64,
    dbase: Int64,
):
    """Element t of the C-order index space of `shape` (dims[0:nd]) moves from
    src[sbase + sum i_j * s_j] to dst[dbase + sum i_j * d_j] (source strides
    dims[nd:2nd], destination strides dims[2nd:3nd]; bases relative to the
    uploaded spans). One thread per element; bits unchanged."""
    var t = _tid()
    if t >= Int(total_):
        return
    var nd = Int(nd_)
    var rem = t
    var so = Int(sbase)
    var do_ = Int(dbase)
    var k = nd - 1
    while k >= 0:
        var ext = Int(dims.unsafe_load(k))
        var ik = rem % ext
        rem = rem // ext
        so += ik * Int(dims.unsafe_load(nd + k))
        do_ += ik * Int(dims.unsafe_load(2 * nd + k))
        k -= 1
    dst.unsafe_store(do_, src.unsafe_load(so))


def _launch_strided[dt: DType](
    ctx: DeviceContext, src_addr: Int, dst_addr: Int, dims_addr: Int, nd: Int,
    total: Int, smin: Int, slen: Int, dmin: Int, dlen: Int, sbase: Int, dbase: Int,
) raises:
    comptime P = MutPointer[Scalar[dt], MutAnyOrigin]
    comptime W = size_of[Scalar[dt]]()
    var d_s = ctx.enqueue_create_buffer[dt](slen)
    var d_d = ctx.enqueue_create_buffer[dt](dlen)
    var d_dims = ctx.enqueue_create_buffer[DType.int64](3 * nd)
    ctx.enqueue_copy(dst_buf=d_s, src_ptr=P(unsafe_from_address=src_addr + smin * W))
    # The destination span goes up too: a strided destination keeps the bytes
    # between the elements this copy writes.
    ctx.enqueue_copy(dst_buf=d_d, src_ptr=P(unsafe_from_address=dst_addr + dmin * W))
    ctx.enqueue_copy(dst_buf=d_dims, src_ptr=_I64(unsafe_from_address=dims_addr))
    ctx.enqueue_function[_strided_kernel[dt]](
        d_s.unsafe_ptr(), d_d.unsafe_ptr(), d_dims.unsafe_ptr(), Int32(nd), Int32(total),
        Int64(sbase - smin), Int64(dbase - dmin),
        grid_dim=_blocks(total), block_dim=IND_TPB,
    )
    ctx.enqueue_copy(dst_ptr=P(unsafe_from_address=dst_addr + dmin * W), src_buf=d_d)
    ctx.synchronize()
    _ = d_s^
    _ = d_d^
    _ = d_dims^


def device_strided_copy(
    ctx: DeviceContext, src_addr: Int, dst_addr: Int, dims_addr: Int, nd: Int,
    itemsize: Int, total: Int,
) raises -> Bool:
    """`strided_copy_bytes` on the device. True when it wrote `dst`; False
    (nothing touched) when the source and destination spans overlap or a span
    is larger than `IND_MAX_N` elements, and the caller runs its host loop.
    The span bounds are O(ndim) scalar metadata read from `dims`."""
    var dp = _I64(unsafe_from_address=dims_addr)
    var smin = Int(dp[3 * nd])
    var smax = smin
    var dmin = Int(dp[3 * nd + 1])
    var dmax = dmin
    for k in range(nd):
        var ext = Int(dp[k]) - 1
        var s = Int(dp[nd + k]) * ext
        var d = Int(dp[2 * nd + k]) * ext
        if s < 0:
            smin += s
        else:
            smax += s
        if d < 0:
            dmin += d
        else:
            dmax += d
    if smin < 0 or dmin < 0:
        return False
    var slen = smax - smin + 1
    var dlen = dmax - dmin + 1
    if slen > IND_MAX_N or dlen > IND_MAX_N or total > IND_MAX_N:
        return False
    var s_lo = src_addr + smin * itemsize
    var s_hi = src_addr + (smax + 1) * itemsize
    var d_lo = dst_addr + dmin * itemsize
    var d_hi = dst_addr + (dmax + 1) * itemsize
    if s_lo < d_hi and d_lo < s_hi:
        return False
    var sbase = Int(dp[3 * nd])
    var dbase = Int(dp[3 * nd + 1])
    if itemsize == 1:
        _launch_strided[DType.uint8](ctx, src_addr, dst_addr, dims_addr, nd, total, smin, slen, dmin, dlen, sbase, dbase)
    elif itemsize == 2:
        _launch_strided[DType.uint16](ctx, src_addr, dst_addr, dims_addr, nd, total, smin, slen, dmin, dlen, sbase, dbase)
    elif itemsize == 4:
        _launch_strided[DType.uint32](ctx, src_addr, dst_addr, dims_addr, nd, total, smin, slen, dmin, dlen, sbase, dbase)
    elif itemsize == 8:
        _launch_strided[DType.uint64](ctx, src_addr, dst_addr, dims_addr, nd, total, smin, slen, dmin, dlen, sbase, dbase)
    else:
        return False
    return True


# ===========================================================================
# check_lengths_i64
# ===========================================================================


def _lengths_kernel(lens: _I64, n_: Int32, hi: Int64, status: _I32):
    """status[0] = min(status[0], i) for every i with lens[i] outside [1, hi]."""
    var i = _tid()
    if i >= Int(n_):
        return
    var v = lens.unsafe_load(i)
    if v < Int64(1) or v > hi:
        _ = Atomic[DType.int32].min(status, Int32(i))


def device_check_lengths(ctx: DeviceContext, addr: Int, n: Int, hi: Int) raises -> Int:
    """`check_lengths_i64` (`1 <= n <= IND_MAX_N`): the first i whose length
    is outside [1, hi], or -1. An integer minimum: order-free."""
    var d = ctx.enqueue_create_buffer[DType.int64](n)
    var d_status = ctx.enqueue_create_buffer[DType.int32](1)
    enqueue_fill(ctx, d_status, IND_NONE)
    ctx.enqueue_copy(dst_buf=d, src_ptr=_I64(unsafe_from_address=addr))
    ctx.enqueue_function[_lengths_kernel](
        d.unsafe_ptr(), Int32(n), Int64(hi), d_status.unsafe_ptr(),
        grid_dim=_blocks(n), block_dim=IND_TPB,
    )
    var first = _read_word(ctx, d_status)
    _ = d^
    _ = d_status^
    if first == Int(IND_NONE):
        return -1
    return first


# ===========================================================================
# ragged_rows_bytes
# ===========================================================================


def _ragged_check_kernel(lens: _I64, n_: Int32, row: Int64, pos: Int64, mode: Int32, status: _I32):
    """status[0] = 1 when a length would make the host loop read or write
    outside its row (the twin then declines and the host helper runs)."""
    var i = _tid()
    if i >= Int(n_):
        return
    var v = lens.unsafe_load(i)
    var bad: Bool
    if mode == Int32(2):
        bad = v < Int64(1) or v * row > pos
    else:
        bad = v < Int64(0) or v * pos > row
    if bad:
        status.unsafe_store(0, Int32(1))


def _ragged_kernel(src: _U8, dst: _U8, lens: _I64, n_: Int32, row_: Int32, pos_: Int32, mode: Int32):
    """One thread per destination byte j = i * row + k:
      mode 0: dst[j] = src[j] if k < lens[i] * pos else 0;
      mode 1: dst[j] = 0 if k >= lens[i] * pos (else unchanged);
      mode 2: dst[j] = src[i * pos + (lens[i] - 1) * row + k]."""
    var j = _tid()
    var row = Int(row_)
    if j >= Int(n_) * row:
        return
    var i = j // row
    var k = j - i * row
    var pos = Int(pos_)
    var len_i = Int(lens.unsafe_load(i))
    if mode == Int32(0):
        if k < len_i * pos:
            dst.unsafe_store(j, src.unsafe_load(j))
        else:
            dst.unsafe_store(j, UInt8(0))
    elif mode == Int32(1):
        if k >= len_i * pos:
            dst.unsafe_store(j, UInt8(0))
    elif len_i >= 1 and len_i * row <= pos:
        # (a length out of range is flagged by `_ragged_check_kernel` and the
        # result discarded; the guard only keeps the read inside `src`)
        dst.unsafe_store(j, src.unsafe_load(i * pos + (len_i - 1) * row + k))


def device_ragged_rows(
    ctx: DeviceContext, src_addr: Int, dst_addr: Int, lens_addr: Int,
    n: Int, row: Int, pos: Int, mode: Int,
) raises -> Bool:
    """`ragged_rows_bytes` on the device (`n >= 1`, `row >= 1`). True when
    it wrote `dst`; False (nothing touched) when a size is out of range or a
    length would take the host loop outside its row, and the caller runs its
    host loop. Mode 1 is in place (`src` unused), so its row bytes go up."""
    if mode < 0 or mode > 2 or pos < 1:
        return False
    var dlen = n * row
    var slen = n * pos if mode == 2 else dlen
    if dlen > IND_MAX_N or slen > IND_MAX_N:
        return False
    var d_l = ctx.enqueue_create_buffer[DType.int64](n)
    var d_status = ctx.enqueue_create_buffer[DType.int32](1)
    var d_d = ctx.enqueue_create_buffer[DType.uint8](dlen)
    var d_s = ctx.enqueue_create_buffer[DType.uint8](slen if mode != 1 else 1)
    enqueue_fill(ctx, d_status, Int32(0))
    ctx.enqueue_copy(dst_buf=d_l, src_ptr=_I64(unsafe_from_address=lens_addr))
    if mode == 1:
        ctx.enqueue_copy(dst_buf=d_d, src_ptr=_U8(unsafe_from_address=dst_addr))
    else:
        ctx.enqueue_copy(dst_buf=d_s, src_ptr=_U8(unsafe_from_address=src_addr))
    ctx.enqueue_function[_ragged_check_kernel](
        d_l.unsafe_ptr(), Int32(n), Int64(row), Int64(pos), Int32(mode), d_status.unsafe_ptr(),
        grid_dim=_blocks(n), block_dim=IND_TPB,
    )
    ctx.enqueue_function[_ragged_kernel](
        d_s.unsafe_ptr(), d_d.unsafe_ptr(), d_l.unsafe_ptr(), Int32(n), Int32(row), Int32(pos), Int32(mode),
        grid_dim=_blocks(dlen), block_dim=IND_TPB,
    )
    var bad = _read_word(ctx, d_status)
    if bad != 0:
        _ = d_l^
        _ = d_d^
        _ = d_s^
        _ = d_status^
        return False
    ctx.enqueue_copy(dst_ptr=_U8(unsafe_from_address=dst_addr), src_buf=d_d)
    ctx.synchronize()
    _ = d_l^
    _ = d_d^
    _ = d_s^
    _ = d_status^
    return True
