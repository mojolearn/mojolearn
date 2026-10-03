# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/apple-fast-neural-gemm (2026-10-03): the Apple FAST GEMM candidates.

Every symbol here is compiled only under `AFN_GEMM_APPLE` (the FAST tier on
an Apple GPU build) AND one of the `MOJOLEARN_AFN_GEMM_*` defines below;
IDENTICAL compiles main's code unchanged (`gemm/checks/gemm_identical.mojo`
reads `AFN_GEMM_FP32_MMA`, which is False there, so no line of this file is
instantiated under IDENTICAL).

THE KERNEL. `afn_gemm_mma_kernel` is one block per `BM x BN` output tile
(`SGM x SGN` simdgroups of `FM x FN` 8x8 fragments, each lane two cells of
each fragment: the layout `identical_gemm_apple_mma_kernel` uses), walking
`k` in `KB`-step windows staged through threadgroup memory and multiplied on
`air.simdgroup_matrix_8x8_multiply_accumulate` (fp32 in, fp32 accumulate:
exact products, f32 sums; the matrix unit's fold order is the only thing
that differs from the IDENTICAL kernel's chain). What it does NOT carry,
against the IDENTICAL matrix kernel: no leaf partition and no fold stack
(FAST's fold order is free, so the accumulator runs the whole of `k`), no
per-window admission minima (the two warp reductions and the shared
exchange per window), no flush on staging, no `rtf` seam. The next window's
global words are fetched into registers before the current one multiplies;
where two staged pages fit Apple's 32 KB (`lib_smem_pages_for`) the kernel
runs one barrier per window, else two.

Operands are generic in `AT` and `BT`: float32, or bf16 BITS in a uint16
buffer widened on the way into threadgroup memory (`bits << 16`, exact,
contract L-1), so the bf16 profile's product runs from the bits with no
widen launch and no float32 image of either operand (`AFN_GEMM_BF16_MMA`).

THE DEFINES (each default OFF; `MOJOLEARN_AFN_GEMM_ALL` turns on every one):

  MOJOLEARN_AFN_GEMM_SIMDGROUP  fp32 products on this kernel at the 64x64
      tile, ahead of the vendor route, from every fp32 entry point
      (`identical_gemm`, `identical_gemm_into`, both `allow_vendor` arms:
      the kernel is full fp32, so the TF32 objection that closes the vendor
      route does not apply to it).
  MOJOLEARN_AFN_GEMM_SPLITK     outputs with fewer than `2 x AFN_GEMM_CORES`
      tiles split `k` over `grid.y` (whole windows per split, at least
      `AFN_GEMM_SPLIT_MIN_STEPS` steps each, at most `AFN_GEMM_SPLIT_MAX`
      splits) and add their partial tiles into a zeroed `C` with global f32
      atomics (Metal has global float atomics: gbdt DEVIATION 93), one
      zero launch and one GEMM launch. The sum order across splits is
      nondeterministic, which FAST allows; f32 throughout.
  MOJOLEARN_AFN_GEMM_TILESHAPE  the tile from the shape, on the host, with
      no device readback: 128x32 when `n <= 32`, 32x128 when `m <= 32`,
      32x32 when the 64x64 grid has fewer than `2 x AFN_GEMM_CORES` tiles
      (four times the blocks), 64x64 otherwise. Composes with SPLITK (the
      smaller tile first, then a split if the grid is still short).
  MOJOLEARN_AFN_GEMM_BF16_MMA   `identical_gemm_bf16w_into` (float32 A, bf16
      B) and the binding's bf16 x bf16 call run this kernel from the bits.
  MOJOLEARN_AFN_GEMM_INT8_MMA   `identical_gemm_int8_into` runs the exact
      chunked matrix-unit kernel (`gemm_int8_apple_chunk.mojo`: every chunk
      sum below 2^24 is a float32, Int32 between chunks, the flat kernel's
      epilogue) instead of the flat one-thread-per-cell kernel's sliced
      launches with a wait between slices. Same Int32, same bits.
  MOJOLEARN_AFN_GEMM_EPILOGUE   `afn_gemm_fused_into` and the binding's
      `gemm_fused`: bias, bias + residual, bias + SiLU, bias + GELU (exact
      erf) applied in the store, for the attention, mamba and MLP lanes to
      call later (nothing calls it this round).

Shared-memory pages carry a fits gate (`lib_smem_page_fits_for`); kernel
arguments are fixed-width; pointers cross only `@always_inline` callees.
"""

from std.atomic import Atomic
from std.ffi import external_call
from std.gpu import block_dim, block_idx, thread_idx
from std.math import erf, exp
from std.memory import bitcast, stack_allocation
from std.sys.compile import is_defined
from std.sys.defines import get_defined_int
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.kernel_matrix import (
    COLUMN_APPLE,
    TARGET_COLUMN,
    lib_smem_page_fits_for,
    lib_smem_pages_for,
)
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from core.apple_air import simdgroup_load_legacy_air
from gemm.contract import OP_NN, OP_NT, OP_TN


# ===========================================================================
# The define table
# ===========================================================================

#: The tier and the vendor every candidate is behind.
comptime AFN_GEMM_APPLE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator() and not is_defined["MOJOLEARN_COLUMN_CPU"]()
    and TARGET_COLUMN == COLUMN_APPLE
)
comptime AFN_GEMM_ALL = AFN_GEMM_APPLE and is_defined["MOJOLEARN_AFN_GEMM_ALL"]()
comptime AFN_GEMM_SIMDGROUP = AFN_GEMM_ALL or (
    AFN_GEMM_APPLE and is_defined["MOJOLEARN_AFN_GEMM_SIMDGROUP"]()
)
comptime AFN_GEMM_SPLITK = AFN_GEMM_ALL or (
    AFN_GEMM_APPLE and is_defined["MOJOLEARN_AFN_GEMM_SPLITK"]()
)
comptime AFN_GEMM_TILESHAPE = AFN_GEMM_ALL or (
    AFN_GEMM_APPLE and is_defined["MOJOLEARN_AFN_GEMM_TILESHAPE"]()
)
comptime AFN_GEMM_BF16_MMA = AFN_GEMM_ALL or (
    AFN_GEMM_APPLE and is_defined["MOJOLEARN_AFN_GEMM_BF16_MMA"]()
)
comptime AFN_GEMM_INT8_MMA = AFN_GEMM_ALL or (
    AFN_GEMM_APPLE and is_defined["MOJOLEARN_AFN_GEMM_INT8_MMA"]()
)
comptime AFN_GEMM_EPILOGUE = AFN_GEMM_ALL or (
    AFN_GEMM_APPLE and is_defined["MOJOLEARN_AFN_GEMM_EPILOGUE"]()
)
#: The fp32 entry points take this kernel whenever any fp32 candidate is on
#: (SPLITK and TILESHAPE are policies of the same kernel).
comptime AFN_GEMM_FP32_MMA = AFN_GEMM_SIMDGROUP or AFN_GEMM_SPLITK or AFN_GEMM_TILESHAPE

#: The window: steps staged per barrier. 32 stages one 64x64 page of
#: 17,920 bytes (one page on Apple's 32 KB); 16 stages two pages of 9,472.
comptime AFN_GEMM_KB = get_defined_int["MOJOLEARN_AFN_GEMM_KB", 32]()
#: GPU cores the grid should cover twice over (M3 Ultra: 80).
comptime AFN_GEMM_CORES = get_defined_int["MOJOLEARN_AFN_GEMM_CORES", 80]()
comptime AFN_GEMM_SPLIT_MAX = 16
comptime AFN_GEMM_SPLIT_MIN_STEPS = 256

#: Epilogues (`afn_gemm_fused_into`).
comptime AFN_EPI_NONE = 0
comptime AFN_EPI_BIAS = 1
comptime AFN_EPI_BIAS_RESID = 2
comptime AFN_EPI_BIAS_SILU = 3
comptime AFN_EPI_BIAS_GELU = 4

#: Tiles (`afn_gemm_tile`).
comptime AFN_TILE_SQUARE = 0  # 64x64: 2x2 simdgroups, 4x4 fragments
comptime AFN_TILE_SMALL = 1  # 32x32: 2x2 simdgroups, 2x2 fragments
comptime AFN_TILE_TALL = 2  # 128x32: 4x1 simdgroups, 4x4 fragments
comptime AFN_TILE_WIDE = 3  # 32x128: 1x4 simdgroups, 4x4 fragments

comptime _M64 = SIMD[DType.float32, 64]
comptime _V2 = SIMD[DType.int64, 2]
comptime _SPtr = UnsafePointer[Float32, MutUntrackedOrigin, address_space = AddressSpace.SHARED]


# ===========================================================================
# The matrix unit
# ===========================================================================


@always_inline
def _afn_load_t(p: _SPtr, stride: Int) -> _M64:
    """Fragment M[r][c] = p[c * stride + r] (the transposed load; the AIR
    signature depends on the target, `core/apple_air.mojo`)."""
    comptime if simdgroup_load_legacy_air():
        return external_call["air.simdgroup_matrix_8x8_load.v64f32.p3f32", _M64](
            p, Int64(stride), _V2(0, 0), True
        )
    else:
        return external_call["air.simdgroup_matrix_8x8_load.v64f32.p3f32", _M64](
            p, _V2(Int64(stride), 8), _V2(Int64(stride), 1), _V2(0, 0)
        )


@always_inline
def _afn_mma(a: _M64, b: _M64, c: _M64) -> _M64:
    return external_call[
        "air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32",
        _M64,
    ](a, b, c)


@always_inline
def _afn_widen[T: DType](v: Scalar[T]) -> Float32:
    """The operand word as float32: bf16 bits are the top half of a float32
    (exact, contract L-1); float32 is itself."""
    comptime if T == DType.uint16:
        return bitcast[DType.float32](v.cast[DType.uint32]() << UInt32(16))
    else:
        return v.cast[DType.float32]()


@always_inline
def _afn_gload[
    ROWS: Int, KB: Int, NT: Int, T: DType
](
    src: MutPointer[Scalar[T], MutAnyOrigin],
    outer_stride: Int,
    k_stride: Int,
    base_outer: Int,
    outer_limit: Int,
    k0: Int,
    chunk: Int,
    tid: Int,
    ofast: Bool,
) -> SIMD[DType.float32, 4 * ((ROWS * KB) // (4 * NT))]:
    """One window's operand words for this thread, 4 per slot, widened.
    `ofast` (the outer index has stride 1): slot = (p, 4 consecutive outer);
    else slot = (outer, 4 consecutive p). Words outside `outer_limit` or
    past `chunk` read +0.0: a zero word multiplies a zero word at the same
    step, so the padding adds exactly nothing to any cell in range."""
    comptime SL = (ROWS * KB) // (4 * NT)
    comptime assert SL * 4 * NT == ROWS * KB, "_afn_gload: whole slots"
    comptime assert ROWS % 4 == 0 and KB % 4 == 0, "_afn_gload: 4-wide slots"
    var r = SIMD[DType.float32, 4 * SL](0.0)
    comptime for sl in range(SL):
        var s = sl * NT + tid
        if ofast:
            var pp = s // (ROWS // 4)
            var o4 = (s % (ROWS // 4)) * 4
            if pp < chunk:
                var go = base_outer + o4
                var off = go + (k0 + pp) * k_stride
                if go + 3 < outer_limit:
                    var v = src.unsafe_load[width=4](off)
                    comptime for q in range(4):
                        r[4 * sl + q] = _afn_widen[T](v[q])
                else:
                    comptime for q in range(4):
                        if go + q < outer_limit:
                            r[4 * sl + q] = _afn_widen[T](src.unsafe_load(off + q))
        else:
            var o = s // (KB // 4)
            var p4 = (s % (KB // 4)) * 4
            var go = base_outer + o
            if go < outer_limit:
                var off = go * outer_stride + (k0 + p4) * k_stride
                if k_stride == 1 and p4 + 3 < chunk:
                    var v = src.unsafe_load[width=4](off)
                    comptime for q in range(4):
                        r[4 * sl + q] = _afn_widen[T](v[q])
                else:
                    comptime for q in range(4):
                        if p4 + q < chunk:
                            r[4 * sl + q] = _afn_widen[T](src.unsafe_load(off + q * k_stride))
    return r


@always_inline
def _afn_stage[
    ROWS: Int, KB: Int, NT: Int, PMAJOR: Bool, ST: Int
](
    dst: _SPtr,
    r: SIMD[DType.float32, 4 * ((ROWS * KB) // (4 * NT))],
    tid: Int,
    ofast: Bool,
):
    """Store one thread's slots. `PMAJOR`: element (outer o, p) at
    `dst[p * ST + o]` (A); else at `dst[o * ST + p]` (B). No flush: FAST."""
    comptime SL = (ROWS * KB) // (4 * NT)
    comptime for sl in range(SL):
        var s = sl * NT + tid
        var v = SIMD[DType.float32, 4](0.0)
        comptime for q in range(4):
            v[q] = r[4 * sl + q]
        if ofast:
            var pp = s // (ROWS // 4)
            var o4 = (s % (ROWS // 4)) * 4
            comptime if PMAJOR:
                (dst + pp * ST + o4).store[alignment=16](v)
            else:
                comptime for q in range(4):
                    dst[(o4 + q) * ST + pp] = v[q]
        else:
            var o = s // (KB // 4)
            var p4 = (s % (KB // 4)) * 4
            comptime if PMAJOR:
                comptime for q in range(4):
                    dst[(p4 + q) * ST + o] = v[q]
            else:
                (dst + o * ST + p4).store[alignment=16](v)


@always_inline
def _afn_epilogue[
    EPI: Int
](
    v: Float32,
    bias: MutPointer[Float32, MutAnyOrigin],
    resid: MutPointer[Float32, MutAnyOrigin],
    gi: Int,
    gj: Int,
    n: Int,
) -> Float32:
    """The fused store: `bias[j]`, then the residual or the gate. f32
    throughout; GELU is the exact erf form, SiLU the exact logistic."""
    comptime if EPI == AFN_EPI_NONE:
        return v
    else:
        var x = v + bias.unsafe_load(gj)
        comptime if EPI == AFN_EPI_BIAS_RESID:
            return x + resid.unsafe_load(gi * n + gj)
        elif EPI == AFN_EPI_BIAS_SILU:
            return x / (Float32(1.0) + exp(-x))
        elif EPI == AFN_EPI_BIAS_GELU:
            return Float32(0.5) * x * (Float32(1.0) + erf(x * Float32(0.70710678118654752)))
        else:
            return x


def afn_gemm_mma_kernel[
    SGM: Int, SGN: Int, FM: Int, FN: Int, KB: Int,
    AT: DType, BT: DType, SPLIT: Bool, EPI: Int,
](
    c: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Scalar[AT], MutAnyOrigin],
    b: MutPointer[Scalar[BT], MutAnyOrigin],
    bias: MutPointer[Float32, MutAnyOrigin],
    resid: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    k_in: Int32,
    a_si_in: Int32,
    a_sp_in: Int32,
    b_sp_in: Int32,
    b_sj_in: Int32,
    k_split_in: Int32,
):
    """`C[m x n] (+)= A_eff[m x k] . B_eff[k x n]` on the matrix unit.
    `A_eff[i, p] = a[i * a_si + p * a_sp]`, `B_eff[p, j] = b[p * b_sp + j *
    b_sj]` (`gemm_operand_strides`' table). `SPLIT`: block `(tile, y)` walks
    steps `[y * k_split, min(k, (y + 1) * k_split))` and ADDS its tile into
    `c` with global f32 atomics (`c` zeroed by the launcher); else the block
    walks every step and stores through the `EPI` epilogue."""
    comptime NSG = SGM * SGN
    comptime NT = NSG * 32
    comptime BM = 8 * FM * SGM
    comptime BN = 8 * FN * SGN
    comptime AST = BM + 4
    comptime BST = KB + 4
    comptime NF = FM * FN
    comptime ASZ = KB * AST
    comptime BSZ = BN * BST
    comptime PAGE_BYTES = (ASZ + BSZ) * 4
    comptime NPG = lib_smem_pages_for[COLUMN_APPLE, PAGE_BYTES]()
    comptime assert lib_smem_page_fits_for[COLUMN_APPLE, PAGE_BYTES](), (
        "afn_gemm_mma_kernel: one staged page must fit Apple's threadgroup memory"
    )
    comptime assert KB % 8 == 0, "afn_gemm_mma_kernel: whole 8-step fragments"
    comptime assert not (SPLIT and EPI != AFN_EPI_NONE), (
        "afn_gemm_mma_kernel: a split tile adds into C, no epilogue"
    )
    var m = Int(m_in)
    var n = Int(n_in)
    var k = Int(k_in)
    var a_si = Int(a_si_in)
    var a_sp = Int(a_sp_in)
    var b_sp = Int(b_sp_in)
    var b_sj = Int(b_sj_in)
    var tid = Int(thread_idx.x)
    var sg = tid // 32
    var lane = tid % 32
    var sgm = sg // SGN
    var sgn = sg % SGN
    var nbn = (n + BN - 1) // BN
    var bid = Int(block_idx.x)
    var m0 = (bid // nbn) * BM
    var n0 = (bid % nbn) * BN
    var qd = lane // 4
    var frow = (qd & 4) + ((lane // 2) % 4)
    var fcol = (qd & 2) * 2 + (lane % 2) * 2
    var at = stack_allocation[NPG * ASZ, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var bt = stack_allocation[NPG * BSZ, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var acc = InlineArray[_M64, NF](fill=_M64(0))
    var a_ofast = a_si == 1 and a_sp != 1
    var b_ofast = b_sj == 1 and b_sp != 1
    var kb = 0
    var ke = k
    comptime if SPLIT:
        kb = Int(block_idx.y) * Int(k_split_in)
        ke = min(k, kb + Int(k_split_in))
    var windows = (ke - kb + KB - 1) // KB
    # The next window's words are in registers while this one multiplies.
    var ra = _afn_gload[BM, KB, NT, AT](a, a_si, a_sp, m0, m, kb, min(KB, ke - kb), tid, a_ofast)
    var rb = _afn_gload[BN, KB, NT, BT](b, b_sj, b_sp, n0, n, kb, min(KB, ke - kb), tid, b_ofast)
    comptime if NPG == 2:
        # Two pages: window w multiplies from one while w + 1 stages into
        # the other, one barrier per window.
        _afn_stage[BM, KB, NT, True, AST](at, ra, tid, a_ofast)
        _afn_stage[BN, KB, NT, False, BST](bt, rb, tid, b_ofast)
        barrier()
        if windows > 1:
            ra = _afn_gload[BM, KB, NT, AT](a, a_si, a_sp, m0, m, kb + KB, min(KB, ke - kb - KB), tid, a_ofast)
            rb = _afn_gload[BN, KB, NT, BT](b, b_sj, b_sp, n0, n, kb + KB, min(KB, ke - kb - KB), tid, b_ofast)
    for w in range(windows):
        var k0 = kb + w * KB
        var cur = w % NPG
        var atc = at + cur * ASZ
        var btc = bt + cur * BSZ
        comptime if NPG == 1:
            _afn_stage[BM, KB, NT, True, AST](at, ra, tid, a_ofast)
            _afn_stage[BN, KB, NT, False, BST](bt, rb, tid, b_ofast)
            barrier()
            if w + 1 < windows:
                var k1 = k0 + KB
                ra = _afn_gload[BM, KB, NT, AT](a, a_si, a_sp, m0, m, k1, min(KB, ke - k1), tid, a_ofast)
                rb = _afn_gload[BN, KB, NT, BT](b, b_sj, b_sp, n0, n, k1, min(KB, ke - k1), tid, b_ofast)
        comptime for p8 in range(KB // 8):
            var af = InlineArray[_M64, FM](fill=_M64(0))
            var bf = InlineArray[_M64, FN](fill=_M64(0))
            comptime for fm in range(FM):
                af[fm] = _afn_load_t(atc + (8 * p8) * AST + (sgm * FM + fm) * 8, AST)
            comptime for fq in range(FN):
                bf[fq] = _afn_load_t(btc + ((sgn * FN + fq) * 8) * BST + 8 * p8, BST)
            comptime for fm in range(FM):
                comptime for fq in range(FN):
                    acc[fm * FN + fq] = _afn_mma(af[fm], bf[fq], acc[fm * FN + fq])
        comptime if NPG == 2:
            if w + 1 < windows:
                var nxt = (w + 1) % NPG
                _afn_stage[BM, KB, NT, True, AST](at + nxt * ASZ, ra, tid, a_ofast)
                _afn_stage[BN, KB, NT, False, BST](bt + nxt * BSZ, rb, tid, b_ofast)
                if w + 2 < windows:
                    var k2 = k0 + 2 * KB
                    ra = _afn_gload[BM, KB, NT, AT](a, a_si, a_sp, m0, m, k2, min(KB, ke - k2), tid, a_ofast)
                    rb = _afn_gload[BN, KB, NT, BT](b, b_sj, b_sp, n0, n, k2, min(KB, ke - k2), tid, b_ofast)
        barrier()
    comptime for fm in range(FM):
        comptime for fq in range(FN):
            comptime for e in range(2):
                var gi = m0 + (sgm * FM + fm) * 8 + frow
                var gj = n0 + (sgn * FN + fq) * 8 + fcol + e
                if gi < m and gj < n:
                    var v = acc[fm * FN + fq][e]
                    comptime if SPLIT:
                        _ = Atomic.fetch_add(c.unsafe_offset(gi * n + gj), v)
                    else:
                        c.unsafe_store(gi * n + gj, _afn_epilogue[EPI](v, bias, resid, gi, gj, n))


comptime AFN_ZERO_TPB = 256


def afn_zero_kernel(c: MutPointer[Float32, MutAnyOrigin], count_in: Int32):
    """`c[0, count) = +0.0`, four words per thread (the split sum's seed;
    only the `m n` cells, not the pooled buffer around them)."""
    var count = Int(count_in)
    var i = (Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)) * 4
    if i + 3 < count:
        c.unsafe_store[width=4](i, SIMD[DType.float32, 4](0.0))
    else:
        comptime for q in range(4):
            if i + q < count:
                c.unsafe_store(i + q, Float32(0.0))


# ===========================================================================
# The host side: strides, tile, splits, launch
# ===========================================================================


@always_inline
def _afn_strides(op: Int, m: Int, n: Int, k: Int) -> Tuple[Int, Int, Int, Int]:
    """`(a_si, a_sp, b_sp, b_sj)`: `gemm_operand_strides`' table."""
    var a_si = k
    var a_sp = 1
    if op == OP_TN:
        a_si = 1
        a_sp = m
    var b_sp = n
    var b_sj = 1
    if op == OP_NT:
        b_sp = 1
        b_sj = k
    return (a_si, a_sp, b_sp, b_sj)


def afn_gemm_tile(m: Int, n: Int) -> Int:
    """The tile for `(m, n)`: a host decision from the shape alone
    (AFN_GEMM_TILESHAPE); the 64x64 tile when the define is off."""
    comptime if not AFN_GEMM_TILESHAPE:
        return AFN_TILE_SQUARE
    if n <= 32 and m > 32:
        return AFN_TILE_TALL
    if m <= 32 and n > 32:
        return AFN_TILE_WIDE
    var tiles64 = ((m + 63) // 64) * ((n + 63) // 64)
    if tiles64 < 2 * AFN_GEMM_CORES:
        return AFN_TILE_SMALL
    return AFN_TILE_SQUARE


def afn_gemm_tile_count(tile: Int, m: Int, n: Int) -> Int:
    var bm = 64
    var bn = 64
    if tile == AFN_TILE_SMALL:
        bm = 32
        bn = 32
    elif tile == AFN_TILE_TALL:
        bm = 128
        bn = 32
    elif tile == AFN_TILE_WIDE:
        bm = 32
        bn = 128
    return ((m + bm - 1) // bm) * ((n + bn - 1) // bn)


def afn_gemm_k_split(tiles: Int, k: Int) -> Int:
    """Steps per split (`grid.y` walks `ceil(k / split)` splits), or 0 when
    the product does not split: the grid already covers the cores twice,
    `k` is too short, or AFN_GEMM_SPLITK is off. Whole windows per split."""
    comptime if not AFN_GEMM_SPLITK:
        return 0
    var target = 2 * AFN_GEMM_CORES
    if tiles >= target or k < 2 * AFN_GEMM_SPLIT_MIN_STEPS:
        return 0
    var s = (target + tiles - 1) // tiles
    s = min(s, k // AFN_GEMM_SPLIT_MIN_STEPS)
    s = min(s, AFN_GEMM_SPLIT_MAX)
    if s <= 1:
        return 0
    var per = (k + s - 1) // s
    per = ((per + AFN_GEMM_KB - 1) // AFN_GEMM_KB) * AFN_GEMM_KB
    if (k + per - 1) // per <= 1:
        return 0
    return per


def _afn_launch[
    SGM: Int, SGN: Int, FM: Int, FN: Int, AT: DType, BT: DType, SPLIT: Bool, EPI: Int
](
    ctx: DeviceContext,
    c: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Scalar[AT], MutAnyOrigin],
    b: MutPointer[Scalar[BT], MutAnyOrigin],
    bias: MutPointer[Float32, MutAnyOrigin],
    resid: MutPointer[Float32, MutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
    st: Tuple[Int, Int, Int, Int],
    splits: Int,
    k_split: Int,
) raises:
    comptime kern = afn_gemm_mma_kernel[SGM, SGN, FM, FN, AFN_GEMM_KB, AT, BT, SPLIT, EPI]
    comptime BM = 8 * FM * SGM
    comptime BN = 8 * FN * SGN
    var tiles = ((m + BM - 1) // BM) * ((n + BN - 1) // BN)
    ctx.enqueue_function[kern](
        c, a, b, bias, resid,
        Int32(m), Int32(n), Int32(k),
        Int32(st[0]), Int32(st[1]), Int32(st[2]), Int32(st[3]),
        Int32(k_split),
        grid_dim=(tiles, splits, 1),
        block_dim=(SGM * SGN * 32, 1, 1),
    )


def _afn_launch_tile[
    AT: DType, BT: DType, SPLIT: Bool, EPI: Int
](
    ctx: DeviceContext,
    tile: Int,
    c: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Scalar[AT], MutAnyOrigin],
    b: MutPointer[Scalar[BT], MutAnyOrigin],
    bias: MutPointer[Float32, MutAnyOrigin],
    resid: MutPointer[Float32, MutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
    st: Tuple[Int, Int, Int, Int],
    splits: Int,
    k_split: Int,
) raises:
    """The named tile; the non-square tiles are instantiated only under
    AFN_GEMM_TILESHAPE (`afn_gemm_tile` never names them otherwise)."""
    comptime if AFN_GEMM_TILESHAPE:
        if tile == AFN_TILE_SMALL:
            _afn_launch[2, 2, 2, 2, AT, BT, SPLIT, EPI](ctx, c, a, b, bias, resid, m, n, k, st, splits, k_split)
            return
        if tile == AFN_TILE_TALL:
            _afn_launch[4, 1, 4, 4, AT, BT, SPLIT, EPI](ctx, c, a, b, bias, resid, m, n, k, st, splits, k_split)
            return
        if tile == AFN_TILE_WIDE:
            _afn_launch[1, 4, 4, 4, AT, BT, SPLIT, EPI](ctx, c, a, b, bias, resid, m, n, k, st, splits, k_split)
            return
    _afn_launch[2, 2, 4, 4, AT, BT, SPLIT, EPI](ctx, c, a, b, bias, resid, m, n, k, st, splits, k_split)


def _afn_dispatch[
    AT: DType, BT: DType, EPI: Int
](
    ctx: DeviceContext,
    c: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Scalar[AT], MutAnyOrigin],
    b: MutPointer[Scalar[BT], MutAnyOrigin],
    bias: MutPointer[Float32, MutAnyOrigin],
    resid: MutPointer[Float32, MutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
    op: Int,
) raises -> Bool:
    """Tile, split, launch. False for a shape or op this route does not
    serve (the caller falls through to its own dispatch). Asynchronous:
    every buffer is the caller's and nothing here waits."""
    if m <= 0 or n <= 0 or k <= 0:
        return False
    if op != OP_NN and op != OP_NT and op != OP_TN:
        return False
    var st = _afn_strides(op, m, n, k)
    var tile = afn_gemm_tile(m, n)
    comptime if AFN_GEMM_SPLITK and EPI == AFN_EPI_NONE:
        var k_split = afn_gemm_k_split(afn_gemm_tile_count(tile, m, n), k)
        if k_split > 0:
            var splits = (k + k_split - 1) // k_split
            ctx.enqueue_function[afn_zero_kernel](
                c, Int32(m * n),
                grid_dim=((m * n + 4 * AFN_ZERO_TPB - 1) // (4 * AFN_ZERO_TPB), 1, 1),
                block_dim=(AFN_ZERO_TPB, 1, 1),
            )
            _afn_launch_tile[AT, BT, True, AFN_EPI_NONE](
                ctx, tile, c, a, b, bias, resid, m, n, k, st, splits, k_split
            )
            return True
    _afn_launch_tile[AT, BT, False, EPI](ctx, tile, c, a, b, bias, resid, m, n, k, st, 1, k)
    return True


# ===========================================================================
# The entry points (every one asynchronous, every buffer the caller's)
# ===========================================================================


def afn_gemm_fp32_into(
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
    op: Int,
) raises -> Bool:
    """`C = op(A) . op(B)`, fp32 operands (AFN_GEMM_SIMDGROUP, _SPLITK,
    _TILESHAPE). True when served."""
    comptime if not AFN_GEMM_FP32_MMA:
        return False
    else:
        var cp = c.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        return _afn_dispatch[DType.float32, DType.float32, AFN_EPI_NONE](
            ctx, cp, a.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), b.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), cp, cp, m, n, k, op
        )


def afn_gemm_bf16_into(
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.uint16],
    m: Int,
    n: Int,
    k: Int,
    op: Int,
) raises -> Bool:
    """`C = op(A) . op(B)`, float32 A and bf16 bits B, from the bits
    (AFN_GEMM_BF16_MMA). True when served."""
    comptime if not AFN_GEMM_BF16_MMA:
        return False
    else:
        var cp = c.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        return _afn_dispatch[DType.float32, DType.uint16, AFN_EPI_NONE](
            ctx, cp, a.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), b.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), cp, cp, m, n, k, op
        )


def afn_gemm_bf16_bits_into(
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.uint16],
    mut b: DeviceBuffer[DType.uint16],
    m: Int,
    n: Int,
    k: Int,
    op: Int,
) raises -> Bool:
    """`C = op(A) . op(B)`, both operands bf16 bits, from the bits
    (AFN_GEMM_BF16_MMA; the board's gemm-bf16 call). True when served."""
    comptime if not AFN_GEMM_BF16_MMA:
        return False
    else:
        var cp = c.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        return _afn_dispatch[DType.uint16, DType.uint16, AFN_EPI_NONE](
            ctx, cp, a.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), b.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), cp, cp, m, n, k, op
        )


def afn_gemm_fused_into(
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    mut bias: DeviceBuffer[DType.float32],
    mut resid: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
    op: Int,
    epi: Int,
) raises -> Bool:
    """`C = epi(op(A) . op(B))` (AFN_GEMM_EPILOGUE): `epi` is AFN_EPI_BIAS
    (`+ bias[j]`), AFN_EPI_BIAS_RESID (`+ bias[j] + resid[i, j]`),
    AFN_EPI_BIAS_SILU or AFN_EPI_BIAS_GELU (the gate on `+ bias[j]`), or
    AFN_EPI_NONE. `bias` holds `n` floats; `resid` holds `m n` floats and is
    read only by AFN_EPI_BIAS_RESID. Never splits `k`. True when served;
    False for an unknown `epi` or an unserved shape."""
    comptime if not AFN_GEMM_EPILOGUE:
        return False
    else:
        if m <= 0 or n <= 0 or k <= 0:
            return False
        if op != OP_NN and op != OP_NT and op != OP_TN:
            return False
        var st = _afn_strides(op, m, n, k)
        var tile = afn_gemm_tile(m, n)
        var cp = c.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var ap = a.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var bp = b.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var biasp = bias.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var residp = resid.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        if epi == AFN_EPI_NONE:
            _afn_launch_tile[DType.float32, DType.float32, False, AFN_EPI_NONE](
                ctx, tile, cp, ap, bp, biasp, residp, m, n, k, st, 1, k
            )
        elif epi == AFN_EPI_BIAS:
            _afn_launch_tile[DType.float32, DType.float32, False, AFN_EPI_BIAS](
                ctx, tile, cp, ap, bp, biasp, residp, m, n, k, st, 1, k
            )
        elif epi == AFN_EPI_BIAS_RESID:
            _afn_launch_tile[DType.float32, DType.float32, False, AFN_EPI_BIAS_RESID](
                ctx, tile, cp, ap, bp, biasp, residp, m, n, k, st, 1, k
            )
        elif epi == AFN_EPI_BIAS_SILU:
            _afn_launch_tile[DType.float32, DType.float32, False, AFN_EPI_BIAS_SILU](
                ctx, tile, cp, ap, bp, biasp, residp, m, n, k, st, 1, k
            )
        elif epi == AFN_EPI_BIAS_GELU:
            _afn_launch_tile[DType.float32, DType.float32, False, AFN_EPI_BIAS_GELU](
                ctx, tile, cp, ap, bp, biasp, residp, m, n, k, st, 1, k
            )
        else:
            return False
        return True
