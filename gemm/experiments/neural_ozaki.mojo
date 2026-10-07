# SPDX-License-Identifier: Apache-2.0
"""NN-OZ: the neural IDENTICAL GEMM as an Ozaki-scheme product on int8 units.

Switch: `-D MOJOLEARN_IDN_NEURAL_GEMM_OZAKI_SLICES=<S>`, an int sweep with the
legal set {4, 5, 6}; absent (0) is OFF and compiles nothing below into any
route. IDENTICAL only (`NEURAL_EXPERIMENTS_ALLOWED`). Reached through
`gemm/neural_dispatch.mojo::identical_gemm_into`, the router every neural
model operation calls (CNN `x_cnn/device.mojo::device_gemm`, the training
MLP ops, transformer, Mamba/Samba backward).

THE CONSTRUCTION (the Ozaki scheme on integer matrix units)
-----------------------------------------------------------
1. Row scale. For every row of op(A) and every column of op(B) along `k`,
   the largest magnitude (after the repo's denormal flush) is reduced with
   an integer atomic MAX of its bit pattern. A maximum is the same under
   every grouping, so the scale is the same bits on every column. From its
   biased exponent `be`, `E = be - 126`, so every value of the row is below
   `2^E`. A non-finite value sets the row's marker to the infinity pattern.
2. Code. Each value becomes the integer `Q = RNE(x * 2^(7S-1-E))`, computed
   from the float's bits with integer shifts only (no float arithmetic, so
   no FMA, contraction or subnormal policy can move it). `|Q| <= 2^(7S-1)`.
3. Slices. `Q` is split into S balanced base-128 digits, high first:
   `Q = sum_t d_t 2^(7(S-1-t))`, `|d_t| <= 64`. Each digit plane is int8,
   stored [S][rows][kp] with `k` padded to `kp`, a multiple of 32, with zero
   codes. Every int8 product is at most 4096 in magnitude.
4. Products. Only the S diagonals `D = t + u <= S-1` (the high-order
   triangle, S(S+1)/2 int8 products per k-tile) are formed; diagonal D's
   pairs chain into ONE Int32 accumulator. Integer sums are exact, so the
   order of the k-steps, of the pairs, of the k-split blocks and of the
   atomic adds cannot move a bit: NVIDIA (IMMA m16n8k32), AMD (MFMA
   16x16x32 i8) and every column without an integer unit (the PIECES
   kernel: same diagonals on the integer ALU) store the same words by
   construction. Exactness bound: `(D+1) * k * 4096 < 2^31` for every D.
5. Epilogue. `v = sum_D acc_D 2^(7(S-1-D))` in Int64 (exact under the
   second bound on k, `ozaki_max_k`), then ONE round-to-nearest-even of
   `v * 2^(Ea + Eb - 7S - 5)` to float32 built from integer bit fields
   (results below the normal range flush to signed zero, the IDENTICAL
   denormal policy; above it, infinity). A non-finite marker on either side
   stores NaN for the cell.

ACCURACY (construction bound; measured by `neural_ozaki_check.mojo`)
-------------------------------------------------------------------
Per value the code error is at most `2^(E-7S)`, so relative to the row's
largest magnitude `2^(1-7S)`: S=4 gives 2^-27, below fp32's 2^-24 unit
roundoff. The dropped low triangle adds at most about `k * S * 2^-7S`
times `max|a_row| * max|b_col|`. Both are below the incumbent fp32 fold's
own `k * 2^-24 * sum|a||b|` worst case at S >= 4; values far below their
row's maximum keep fewer significant bits than in fp32 (absolute error is
bounded by the row maximum, not by the value). S=3 (2^-20) is NOT in the
legal set: it is lossy against fp32.

ADMISSION (exactness bounds, not shape targeting)
-------------------------------------------------
A product with `k == 0`, or `k > ozaki_max_k[S]()` (the larger `k` at
which the Int32 diagonal or the Int64 recombination could overflow), is
served by the incumbent pinned GEMM. The bound is a function of S only.

SCHEDULING (no bit depends on it)
---------------------------------
k is split across blocks so a small output with a long k (weight and bias
gradients) still fills the device: `OZAKI_TARGET_BLOCKS` blocks wanted,
each k-chunk at least `OZAKI_MIN_CHUNK` values (atomic traffic amortized
over 16 k-tiles). Partial diagonals meet in Int32 atomic adds: exact.
"""
from std.atomic import Atomic
from std.bit import count_leading_zeros
from std.gpu import WARP_SIZE, block_dim, block_idx, lane_id, thread_idx
from std.memory import bitcast
from std.sys.compile import get_defined_int
from std.sys.info import is_amd_gpu, is_nvidia_gpu
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.kernel_matrix import TARGET_COLUMN, lib_int8_matrix_unit_for
from gemm.checks.gemm_int8_mma import (
    INT8_MMA_BLOCK_TILE_M,
    INT8_MMA_BLOCK_TILE_N,
    INT8_MMA_K_TILE,
    INT8_MMA_TILE,
    INT8_MMA_TPB,
    INT8_MMA_WARPS_N,
    _imma_m16n8k32,
    _pack4,
    _pack8,
)
from gemm.checks.gemm_int15 import _mfma_i32_16x16x32_i8
from gemm.experiments.neural_profile import NEURAL_EXPERIMENTS_ALLOWED, neural_strides

#: The switch: number of int8 slices per operand; 0 (absent) is OFF.
comptime OZAKI_SLICES = get_defined_int["MOJOLEARN_IDN_NEURAL_GEMM_OZAKI_SLICES", 0]()
comptime NEURAL_OZAKI = NEURAL_EXPERIMENTS_ALLOWED and OZAKI_SLICES > 0

#: Blocks the k-split aims for: a few waves on a 142-SM L40S or a 304-CU
#: MI325X. Scheduling only.
comptime OZAKI_TARGET_BLOCKS = 1024
#: Smallest k-chunk one block owns (16 k-tiles of 32). Scheduling only.
comptime OZAKI_MIN_CHUNK = 512
#: Values of one row one absmax block reads (k-contiguous operands).
comptime OZAKI_ABS_CHUNK = 2048
#: Values of one row one absmax thread reads (row-contiguous operands).
comptime OZAKI_ABS_PCHUNK = 64
comptime OZAKI_TPB = 256
comptime _INF_BITS = Int32(0x7F800000)


def ozaki_max_k[S: Int]() -> Int:
    """The largest k the construction keeps exact: every Int32 diagonal
    `(D+1) * k * 4096 < 2^31`, and the Int64 recombination
    `k * 2^(7S+5) * 1.016 < 2^63`."""
    var by_int32 = 2147483647 // (S * 4096)
    var by_int64 = ((1 << (58 - 7 * S)) // 64) * 63
    return min(by_int32, by_int64)


def ozaki_kp(k: Int) -> Int:
    return (k + INT8_MMA_K_TILE - 1) // INT8_MMA_K_TILE * INT8_MMA_K_TILE


def _align16(x: Int) -> Int:
    return (x + 15) // 16 * 16


def ozaki_layout[S: Int](m: Int, n: Int, k: Int) -> Tuple[Int, Int, Int, Int, Int]:
    """Byte offsets in the workspace: B planes, A row markers, B column
    markers, the Int32 diagonals, and the total. A planes start at 0. The
    markers and the diagonals are contiguous so one launch zeroes them."""
    var kp = ozaki_kp(k)
    var b_planes = _align16(S * m * kp)
    var a_bits = b_planes + _align16(S * n * kp)
    var b_bits = a_bits + 4 * m
    var acc = _align16(b_bits + 4 * n)
    var total = acc + 4 * S * m * n
    return (b_planes, a_bits, b_bits, acc, total)


def ozaki_workspace_floats[S: Int](m: Int, n: Int, k: Int) -> Int:
    # +4 floats: `_ozaki_run` rounds the base up to 16 bytes, so the
    # fragment loads may state their alignment on any caller's view.
    return (ozaki_layout[S](m, n, k)[4] + 3) // 4 + 4


# ===========================================================================
# scale and slices
# ===========================================================================


@always_inline
def _mag_bits(x: Float32) -> Int32:
    """The magnitude's bit pattern, flushed: subnormals read 0, every
    non-finite value reads the infinity pattern (the row marker)."""
    var u = bitcast[DType.uint32](x) & UInt32(0x7FFFFFFF)
    if u >= UInt32(0x7F800000):
        return _INF_BITS
    if u < UInt32(0x00800000):
        return Int32(0)
    return Int32(u)


def ozaki_zero_kernel(p: MutPointer[Int32, MutAnyOrigin], count_in: Int32):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(count_in):
        p.unsafe_store(i, Int32(0))


def ozaki_absmax_kcontig_kernel(
    src: MutPointer[Float32, MutAnyOrigin],
    bits: MutPointer[Int32, MutAnyOrigin],
    k_in: Int32,
    rs_in: Int32,
):
    """Grid (rows, k-chunks). Values of a row are consecutive in memory."""
    var r = Int(block_idx.x)
    var k = Int(k_in)
    var p0 = Int(block_idx.y) * OZAKI_ABS_CHUNK
    var p1 = min(p0 + OZAKI_ABS_CHUNK, k)
    var base = r * Int(rs_in)
    var v = Int32(0)
    var p = p0 + Int(thread_idx.x)
    while p < p1:
        v = max(v, _mag_bits(src.unsafe_load(base + p)))
        p += Int(block_dim.x)
    if v > Int32(0):
        _ = Atomic.max(bits.unsafe_offset(r), v)


def ozaki_absmax_rcontig_kernel(
    src: MutPointer[Float32, MutAnyOrigin],
    bits: MutPointer[Int32, MutAnyOrigin],
    rows_in: Int32,
    k_in: Int32,
    ks_in: Int32,
):
    """Grid (row blocks, k-chunks). Consecutive rows are consecutive in
    memory, so a thread owns a row and a warp reads coalesced."""
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if r >= Int(rows_in):
        return
    var k = Int(k_in)
    var p0 = Int(block_idx.y) * OZAKI_ABS_PCHUNK
    var p1 = min(p0 + OZAKI_ABS_PCHUNK, k)
    var v = Int32(0)
    for p in range(p0, p1):
        v = max(v, _mag_bits(src.unsafe_load(r + p * Int(ks_in))))
    if v > Int32(0):
        _ = Atomic.max(bits.unsafe_offset(r), v)


@always_inline
def ozaki_code[S: Int](x: Float32, row_bits: Int32) -> Int64:
    """`RNE(x * 2^(7S-1-E))` from the bits, `E = be(row max) - 126`.
    Integer operations only."""
    var u = bitcast[DType.uint32](x)
    var ea = Int((u >> UInt32(23)) & UInt32(0xFF))
    if ea == 0 or ea == 255 or row_bits <= Int32(0) or row_bits >= _INF_BITS:
        return Int64(0)
    var e_row = Int(row_bits >> Int32(23)) - 126
    var mant = Int64((u & UInt32(0x7FFFFF)) | UInt32(0x800000))
    var shift = ea - 150 + 7 * S - 1 - e_row
    var q: Int64
    if shift >= 0:
        q = mant << Int64(shift)
    else:
        var rs = -shift
        if rs >= 26:
            q = Int64(0)
        else:
            q = mant >> Int64(rs)
            var rem = mant & ((Int64(1) << Int64(rs)) - 1)
            var half = Int64(1) << Int64(rs - 1)
            if rem > half or (rem == half and (q & 1) == 1):
                q += 1
    if (u >> UInt32(31)) != UInt32(0):
        q = -q
    return q


def ozaki_slice_kernel[S: Int](
    src: MutPointer[Float32, MutAnyOrigin],
    planes: MutPointer[Int8, MutAnyOrigin],
    bits: MutPointer[Int32, MutAnyOrigin],
    rows_in: Int32,
    k_in: Int32,
    kp_in: Int32,
    rs_in: Int32,
    ks_in: Int32,
    rows_fast_in: Int32,
):
    """One thread per padded code. With `rows_fast` the thread index runs
    over rows first, so a row-contiguous operand is read coalesced."""
    var rows = Int(rows_in)
    var kp = Int(kp_in)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= rows * kp:
        return
    var r: Int
    var p: Int
    if rows_fast_in != Int32(0):
        p = idx // rows
        r = idx - p * rows
    else:
        r = idx // kp
        p = idx - r * kp
    var q = Int64(0)
    if p < Int(k_in):
        q = ozaki_code[S](src.unsafe_load(r * Int(rs_in) + p * Int(ks_in)), bits.unsafe_load(r))
    var plane = rows * kp
    var at = r * kp + p
    # Balanced base-128 digits, low first; the low digit lands in plane S-1.
    comptime for t in range(S - 1):
        var d = q & 127
        if d >= 64:
            d -= 128
        planes.unsafe_store((S - 1 - t) * plane + at, Int8(d))
        q = (q - d) >> 7
    planes.unsafe_store(at, Int8(q))


# ===========================================================================
# products: Int32 diagonals, k split across blocks, exact atomic merge
# ===========================================================================


@always_inline
def _add_diag(acc: MutPointer[Int32, MutAnyOrigin], v: Int32, d: Int, i: Int, j: Int, m: Int, n: Int):
    if i < m and j < n and v != Int32(0):
        _ = Atomic.fetch_add(acc.unsafe_offset(d * m * n + i * n + j), v)


def ozaki_pieces_kernel[S: Int](
    pa: MutPointer[Int8, MutAnyOrigin],
    pb: MutPointer[Int8, MutAnyOrigin],
    acc: MutPointer[Int32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    kp_in: Int32,
    chunk_in: Int32,
):
    """Columns without an integer matrix unit: one thread per (cell,
    k-chunk), the same diagonals on the integer ALU."""
    var m = Int(m_in)
    var n = Int(n_in)
    var kp = Int(kp_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= m * n:
        return
    var i = cell // n
    var j = cell - i * n
    var p0 = Int(block_idx.y) * Int(chunk_in)
    var p1 = min(p0 + Int(chunk_in), kp)
    var da = InlineArray[Int32, S](fill=Int32(0))
    var db = InlineArray[Int32, S](fill=Int32(0))
    var diag = InlineArray[Int32, S](fill=Int32(0))
    for p in range(p0, p1):
        comptime for t in range(S):
            da[t] = Int32(pa.unsafe_load(t * m * kp + i * kp + p))
            db[t] = Int32(pb.unsafe_load(t * n * kp + j * kp + p))
        comptime for d in range(S):
            comptime for t in range(d + 1):
                diag[d] += da[t] * db[d - t]
    comptime for d in range(S):
        _add_diag(acc, diag[d], d, i, j, m, n)


@always_inline
def _nvidia_warp_tile_oz[S: Int](
    pa: MutPointer[Int8, MutAnyOrigin],
    pb: MutPointer[Int8, MutAnyOrigin],
    acc: MutPointer[Int32, MutAnyOrigin],
    lane: Int, row0: Int, col0: Int, m: Int, n: Int, kp: Int, k0: Int, k1: Int,
):
    """`gemm_int15.mojo::_nvidia_warp_tile15` with S planes a side and S
    diagonal accumulators per n8 half."""
    var g = lane >> 2
    var t4 = lane & 3
    var ra0 = row0 + g
    var ra1 = row0 + g + 8
    var cb0 = col0 + g
    var cb1 = col0 + 8 + g
    var acc0 = InlineArray[SIMD[DType.int32, 4], S](fill=SIMD[DType.int32, 4](0))
    var acc1 = InlineArray[SIMD[DType.int32, 4], S](fill=SIMD[DType.int32, 4](0))
    var fa = InlineArray[Int32, 4 * S](fill=Int32(0))
    var fb = InlineArray[Int32, 4 * S](fill=Int32(0))
    for kt in range(k0, k1, INT8_MMA_K_TILE):
        var ka = kt + t4 * 4
        comptime for t in range(S):
            var a_t = pa.unsafe_offset(t * m * kp)
            var b_t = pb.unsafe_offset(t * n * kp)
            fa[4 * t] = _pack4(a_t, ra0, ka, m, kp, True)
            fa[4 * t + 1] = _pack4(a_t, ra1, ka, m, kp, True)
            fa[4 * t + 2] = _pack4(a_t, ra0, ka + 16, m, kp, True)
            fa[4 * t + 3] = _pack4(a_t, ra1, ka + 16, m, kp, True)
            fb[4 * t] = _pack4(b_t, cb0, ka, n, kp, True)
            fb[4 * t + 1] = _pack4(b_t, cb0, ka + 16, n, kp, True)
            fb[4 * t + 2] = _pack4(b_t, cb1, ka, n, kp, True)
            fb[4 * t + 3] = _pack4(b_t, cb1, ka + 16, n, kp, True)
        comptime for d in range(S):
            comptime for t in range(d + 1):
                comptime u = d - t
                acc0[d] = _imma_m16n8k32(fa[4 * t], fa[4 * t + 1], fa[4 * t + 2], fa[4 * t + 3],
                                         fb[4 * u], fb[4 * u + 1], acc0[d])
                acc1[d] = _imma_m16n8k32(fa[4 * t], fa[4 * t + 1], fa[4 * t + 2], fa[4 * t + 3],
                                         fb[4 * u + 2], fb[4 * u + 3], acc1[d])
    var jc = col0 + t4 * 2
    comptime for d in range(S):
        _add_diag(acc, acc0[d][0], d, ra0, jc, m, n)
        _add_diag(acc, acc0[d][1], d, ra0, jc + 1, m, n)
        _add_diag(acc, acc0[d][2], d, ra1, jc, m, n)
        _add_diag(acc, acc0[d][3], d, ra1, jc + 1, m, n)
        _add_diag(acc, acc1[d][0], d, ra0, jc + 8, m, n)
        _add_diag(acc, acc1[d][1], d, ra0, jc + 9, m, n)
        _add_diag(acc, acc1[d][2], d, ra1, jc + 8, m, n)
        _add_diag(acc, acc1[d][3], d, ra1, jc + 9, m, n)


@always_inline
def _amd_warp_tile_oz[S: Int](
    pa: MutPointer[Int8, MutAnyOrigin],
    pb: MutPointer[Int8, MutAnyOrigin],
    acc: MutPointer[Int32, MutAnyOrigin],
    lane: Int, row0: Int, col0: Int, m: Int, n: Int, kp: Int, k0: Int, k1: Int,
):
    """`gemm_int15.mojo::_amd_warp_tile15` with S planes a side."""
    var i16 = lane & 15
    var kq = (lane >> 4) * 8
    var accv = InlineArray[SIMD[DType.int32, 4], S](fill=SIMD[DType.int32, 4](0))
    var fa = InlineArray[Int64, S](fill=Int64(0))
    var fb = InlineArray[Int64, S](fill=Int64(0))
    for kt in range(k0, k1, INT8_MMA_K_TILE):
        comptime for t in range(S):
            fa[t] = _pack8(pa.unsafe_offset(t * m * kp), row0 + i16, kt + kq, m, kp, True)
            fb[t] = _pack8(pb.unsafe_offset(t * n * kp), col0 + i16, kt + kq, n, kp, True)
        comptime for d in range(S):
            comptime for t in range(d + 1):
                accv[d] = _mfma_i32_16x16x32_i8(fa[t], fb[d - t], accv[d])
    var j = col0 + i16
    var ir = row0 + (lane >> 4) * 4
    comptime for d in range(S):
        _add_diag(acc, accv[d][0], d, ir, j, m, n)
        _add_diag(acc, accv[d][1], d, ir + 1, j, m, n)
        _add_diag(acc, accv[d][2], d, ir + 2, j, m, n)
        _add_diag(acc, accv[d][3], d, ir + 3, j, m, n)


def ozaki_mma_kernel[S: Int](
    pa: MutPointer[Int8, MutAnyOrigin],
    pb: MutPointer[Int8, MutAnyOrigin],
    acc: MutPointer[Int32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    kp_in: Int32,
    chunk_in: Int32,
    tiles_n_in: Int32,
):
    """Grid (output tiles flattened, k-chunks); a block owns a 32 x 32 tile
    (`INT8_MMA_*` geometry), a warp a 16 x 16 tile."""
    var m = Int(m_in)
    var n = Int(n_in)
    var kp = Int(kp_in)
    var tile = Int(block_idx.x)
    var tm = tile // Int(tiles_n_in)
    var tn = tile - tm * Int(tiles_n_in)
    var warp = Int(thread_idx.x) // WARP_SIZE
    var lane = Int(lane_id())
    var wm = warp // INT8_MMA_WARPS_N
    var wn = warp - wm * INT8_MMA_WARPS_N
    var row0 = tm * INT8_MMA_BLOCK_TILE_M + wm * INT8_MMA_TILE
    var col0 = tn * INT8_MMA_BLOCK_TILE_N + wn * INT8_MMA_TILE
    if row0 >= m or col0 >= n:
        return
    var k0 = Int(block_idx.y) * Int(chunk_in)
    var k1 = min(k0 + Int(chunk_in), kp)
    comptime if is_nvidia_gpu():
        _nvidia_warp_tile_oz[S](pa, pb, acc, lane, row0, col0, m, n, kp, k0, k1)
    elif is_amd_gpu():
        _amd_warp_tile_oz[S](pa, pb, acc, lane, row0, col0, m, n, kp, k0, k1)
    else:
        return


# ===========================================================================
# epilogue: one rounding, integer bit fields
# ===========================================================================


@always_inline
def ozaki_round_f32(v: Int64, x: Int) -> Float32:
    """RNE(v * 2^x) as float32; below the normal range a signed zero,
    above it infinity."""
    if v == Int64(0):
        return Float32(0.0)
    var neg = v < Int64(0)
    var mag = UInt64(-v) if neg else UInt64(v)
    var lead = 63 - Int(count_leading_zeros(mag))
    var mant: UInt64
    if lead > 23:
        var rs = lead - 23
        mant = mag >> UInt64(rs)
        var rem = mag & ((UInt64(1) << UInt64(rs)) - 1)
        var half = UInt64(1) << UInt64(rs - 1)
        if rem > half or (rem == half and (mant & 1) == 1):
            mant += 1
            if mant == (UInt64(1) << 24):
                mant >>= 1
                lead += 1
    else:
        mant = mag << UInt64(23 - lead)
    var biased = lead + x + 127
    var sign = UInt32(0x80000000) if neg else UInt32(0)
    if biased >= 255:
        return bitcast[DType.float32](sign | UInt32(0x7F800000))
    if biased <= 0:
        return bitcast[DType.float32](sign)
    return bitcast[DType.float32](sign | (UInt32(biased) << 23) | UInt32(mant & UInt64(0x7FFFFF)))


def ozaki_epilogue_kernel[S: Int](
    c: MutPointer[Float32, MutAnyOrigin],
    acc: MutPointer[Int32, MutAnyOrigin],
    a_bits: MutPointer[Int32, MutAnyOrigin],
    b_bits: MutPointer[Int32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
):
    var m = Int(m_in)
    var n = Int(n_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= m * n:
        return
    var i = cell // n
    var j = cell - i * n
    var ba = a_bits.unsafe_load(i)
    var bb = b_bits.unsafe_load(j)
    if ba >= _INF_BITS or bb >= _INF_BITS:
        c.unsafe_store(cell, bitcast[DType.float32](UInt32(0x7FC00000)))
        return
    if ba == Int32(0) or bb == Int32(0):
        c.unsafe_store(cell, Float32(0.0))
        return
    var v = Int64(0)
    comptime for d in range(S):
        v = v * 128 + Int64(acc.unsafe_load(d * m * n + cell))
    var ea = Int(ba >> Int32(23)) - 126
    var eb = Int(bb >> Int32(23)) - 126
    c.unsafe_store(cell, ozaki_round_f32(v, ea + eb - 7 * S - 5))


# ===========================================================================
# the launch
# ===========================================================================


def ozaki_admits[S: Int](m: Int, n: Int, k: Int) -> Bool:
    return m > 0 and n > 0 and k > 0 and k <= ozaki_max_k[S]()


def _k_split(kp: Int, tiles: Int) -> Tuple[Int, Int]:
    """(chunk, chunks): chunk a multiple of the k-tile."""
    var want = max(1, (OZAKI_TARGET_BLOCKS + tiles - 1) // tiles)
    var most = max(1, (kp + OZAKI_MIN_CHUNK - 1) // OZAKI_MIN_CHUNK)
    var chunks = min(want, most)
    var chunk = ozaki_kp((kp + chunks - 1) // chunks)
    return (chunk, (kp + chunk - 1) // chunk)


def _ozaki_run[S: Int](
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    base: MutPointer[Float32, MutAnyOrigin],
    m: Int, n: Int, k: Int, op: Int,
) raises:
    var lay = ozaki_layout[S](m, n, k)
    var kp = ozaki_kp(k)
    var addr = Int(base)
    var bytes = base.bitcast[Int8]().unsafe_offset((16 - (addr & 15)) & 15)
    var pa = bytes
    var pb = bytes.unsafe_offset(lay[0])
    var a_bits = bytes.unsafe_offset(lay[1]).bitcast[Int32]()
    var b_bits = bytes.unsafe_offset(lay[2]).bitcast[Int32]()
    var acc = bytes.unsafe_offset(lay[3]).bitcast[Int32]()
    var zero_count = (lay[4] - lay[1]) // 4
    ctx.enqueue_function[ozaki_zero_kernel](
        a_bits, Int32(zero_count),
        grid_dim=((zero_count + OZAKI_TPB - 1) // OZAKI_TPB, 1, 1), block_dim=(OZAKI_TPB, 1, 1),
    )
    var st = neural_strides(op, m, n, k)
    # A row i at a[i*st0 + p*st1]; B column j at b[p*st2 + j*st3].
    var a_ptr = a.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var b_ptr = b.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    _scale_and_slice[S](ctx, a_ptr, pa, a_bits, m, k, kp, st[0], st[1])
    _scale_and_slice[S](ctx, b_ptr, pb, b_bits, n, k, kp, st[3], st[2])
    comptime if lib_int8_matrix_unit_for[TARGET_COLUMN]():
        var tiles_m = (m + INT8_MMA_BLOCK_TILE_M - 1) // INT8_MMA_BLOCK_TILE_M
        var tiles_n = (n + INT8_MMA_BLOCK_TILE_N - 1) // INT8_MMA_BLOCK_TILE_N
        var split = _k_split(kp, tiles_m * tiles_n)
        ctx.enqueue_function[ozaki_mma_kernel[S]](
            pa, pb, acc, Int32(m), Int32(n), Int32(kp), Int32(split[0]), Int32(tiles_n),
            grid_dim=(tiles_m * tiles_n, split[1], 1), block_dim=(INT8_MMA_TPB, 1, 1),
        )
    else:
        var blocks = (m * n + OZAKI_TPB - 1) // OZAKI_TPB
        var split = _k_split(kp, blocks)
        ctx.enqueue_function[ozaki_pieces_kernel[S]](
            pa, pb, acc, Int32(m), Int32(n), Int32(kp), Int32(split[0]),
            grid_dim=(blocks, split[1], 1), block_dim=(OZAKI_TPB, 1, 1),
        )
    ctx.enqueue_function[ozaki_epilogue_kernel[S]](
        c.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), acc, a_bits, b_bits, Int32(m), Int32(n),
        grid_dim=((m * n + OZAKI_TPB - 1) // OZAKI_TPB, 1, 1), block_dim=(OZAKI_TPB, 1, 1),
    )


def _scale_and_slice[S: Int](
    ctx: DeviceContext,
    src: MutPointer[Float32, MutAnyOrigin],
    planes: MutPointer[Int8, MutAnyOrigin],
    bits: MutPointer[Int32, MutAnyOrigin],
    rows: Int, k: Int, kp: Int, rs: Int, ks: Int,
) raises:
    var rows_fast = Int32(1) if rs == 1 and ks != 1 else Int32(0)
    if rows_fast != Int32(0):
        ctx.enqueue_function[ozaki_absmax_rcontig_kernel](
            src, bits, Int32(rows), Int32(k), Int32(ks),
            grid_dim=((rows + OZAKI_TPB - 1) // OZAKI_TPB, (k + OZAKI_ABS_PCHUNK - 1) // OZAKI_ABS_PCHUNK, 1),
            block_dim=(OZAKI_TPB, 1, 1),
        )
    else:
        ctx.enqueue_function[ozaki_absmax_kcontig_kernel](
            src, bits, Int32(k), Int32(rs),
            grid_dim=(rows, (k + OZAKI_ABS_CHUNK - 1) // OZAKI_ABS_CHUNK, 1),
            block_dim=(OZAKI_TPB, 1, 1),
        )
    var total = rows * kp
    ctx.enqueue_function[ozaki_slice_kernel[S]](
        src, planes, bits, Int32(rows), Int32(k), Int32(kp), Int32(rs), Int32(ks), rows_fast,
        grid_dim=((total + OZAKI_TPB - 1) // OZAKI_TPB, 1, 1), block_dim=(OZAKI_TPB, 1, 1),
    )


def neural_ozaki_into[S: Int](
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    mut ws: DeviceBuffer[DType.float32],
    m: Int, n: Int, k: Int, op: Int,
) raises -> Bool:
    """`C = op(A) . op(B)` by the Ozaki construction. False (nothing
    enqueued) when the shape is outside the exactness bounds; the caller
    then runs the incumbent. Asynchronous when `ws` is large enough; a
    short `ws` is replaced by an exact-sized temporary and the call waits
    before releasing it."""
    comptime assert S >= 2 and S <= 7, "Ozaki slices must be 2..7 (Int64 code width)"
    if not ozaki_admits[S](m, n, k):
        return False
    var need = ozaki_workspace_floats[S](m, n, k)
    if len(ws) >= need:
        _ozaki_run[S](ctx, c, a, b, ws.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), m, n, k, op)
        return True
    var temporary = ctx.enqueue_create_buffer[DType.float32](need)
    try:
        _ozaki_run[S](ctx, c, a, b, temporary.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), m, n, k, op)
    except error:
        ctx.synchronize()
        raise error
    ctx.synchronize()
    _ = temporary^
    return True
