# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`mojolearn.identical.gemm.int8i32.v1` on the INTEGER MATRIX UNITS.

    NVIDIA   IMMA, `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`
             (sm_80 and later), reached as the NVVM intrinsic
             `llvm.nvvm.mma.m16n8k32.row.col.s8`
    AMD CDNA MFMA, `v_mfma_i32_16x16x32_i8` (gfx942, gfx950), reached as
             `llvm.amdgcn.mfma.i32.16x16x32.i8`

Lane lane/int8-mma, 2026-09-17, DEVIATION 2910. Contract clause L-9 of
`gemm/IDENTICAL_LOWBIT_CONTRACT.md`. The flat realization and the dispatch
are `gemm/checks/gemm_lowbit.mojo`; the answer is
`gemm/host/gemm_lowbit_oracle.mojo::gemm_int8_oracle`; the gates are
`gemm/checks/gemm_lowbit_check.mojo::check_int8_mma_matches_flat` and
`check_int8_device_matches_oracle`.

WHY THE UNIT CANNOT MOVE A BIT (the construction argument)
------------------------------------------------------------
Every product of two int8 codes is an integer of magnitude at most 16129
and is exact. Every partial sum of such products over `k <= INT8_MAX_K`
(131072) fits an Int32 without overflow, contract L-7, and a sum of exact
integers in a wide enough integer accumulator is the same integer whatever
the order, the grouping or the tile shape of the summation. So the unit's
internal k-tile (32 on both vendors), its per-thread fragments and the order
in which it adds the 32 products of a step are all SCHEDULING: the Int32
that leaves the last step is the Int32 `int8_dot_cell` computes with `p`
ascending. The only floating steps of the profile stay where they are, in
`dequant_int8_pinned` (L-5, L-6), which this file calls with the same
arguments the flat kernel does. No vendor rounding mode reaches an integer
MMA; there is no `.ftz`, no fused rounding, no accumulator flush to pin.

THE PADDING RULE. `k` that is not a multiple of 32, and rows beyond `m` or
columns beyond `n` inside a warp tile, are filled with the ZERO CODE. A
zero code contributes `0 * q = 0` to an integer sum, exactly, so the padded
product is the unpadded product. No float is ever padded; nothing here
widens a code before the unit does. Rows and columns beyond `m`, `n` are
masked at the store.

WHAT THIS FILE IS NOT. Modular's `max.gpu.compute.mma.mma` (the public
`mma` entry of the shipped Mojo 1.0 / MAX 26.5 stdlib, documented at
max.modular.com/api/mojo/max/gpu/compute/mma/mma) dispatches FP32, TF32,
FP16, BF16 and FP8 shapes only; its per-arch modules `arch/mma_nvidia` and
`arch/mma_amd` list no int8 or s32 form, and `layout.tensor_core.TensorCore`
lists float32, half and float8 shapes only. The int8 forms are therefore
reached the way this tree already reaches `llvm.nvvm.fma.rn.f`: by name,
through `llvm_intrinsic`. The NVVM intrinsic returns a four-register pack
(`_RegisterPackType`, the stdlib's own return type for `mma.sync`); the
AMDGPU intrinsic returns a `<4 x i32>` vector, `SIMD[DType.int32, 4]`.

RUN OWED. This file has been compiled on the Apple M4 only, where both
vendor branches are dead code, so the fragment layouts below are read
from the PTX ISA (Matrix Fragments for mma.m16n8k32, .s8) and from the
CDNA3 ISA (v_mfma_i32_16x16x32_i8) and are UNVERIFIED until the gate runs:
    RUN OWED: pixi run check-gemm-lowbit                      (H100, MI300X/MI325X)
    RUN OWED: pixi run check-gemm-lowbit-sabotage             (must FAIL, both boxes)
    RUN OWED: pixi run check-gemm-lowbit-host-sabotage        (must FAIL, both boxes)
    RUN OWED: pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_INT8_FORCE_FLAT=1 -I . gemm/checks/gemm_lowbit_check.mojo
`tools/lowbit_mma_leg.sh` runs the four on a rented box.

THE SABOTAGE ARM. `-D MOJOLEARN_LOWBIT_SABOTAGE=1` flips the value of every
cell this kernel stores, exactly as the flat kernel does (DEVIATION 2908),
so a sabotage build fails `check_int8_device_matches_oracle` through either
path and fails `check_int8_mma_matches_flat` at its oracle comparison.
"""

from max.gpu import WARP_SIZE, block_dim, block_idx, lane_id, thread_idx
from std.memory import bitcast
from std.sys import is_defined, llvm_intrinsic
from std.sys.info import is_amd_gpu, is_nvidia_gpu
from std.sys.intrinsics import _RegisterPackType
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.kernel_matrix import (
    TARGET_COLUMN,
    column_name,
    lib_int8_matrix_unit_for,
)
from checks.numerics import dequant_int8_pinned
from gemm.host.gemm_lowbit_oracle import INT8_MAX_K
from gemm.host.gemm_oracle import gemm_oracle_sabotage_value_flip

#: DEVIATION 2908, the value arm, read from the same define the flat kernel
#: reads (`gemm_lowbit.mojo::LOWBIT_SABOTAGE`); re-derived here rather than
#: imported so this file and `gemm_lowbit.mojo` do not import each other.
comptime INT8_MMA_SABOTAGE = is_defined["MOJOLEARN_LOWBIT_SABOTAGE"]()

#: The unit's k step on both vendors: m16n8k32 on NVIDIA, 16x16x32 on AMD.
#: SCHEDULING (the construction argument above); a different tile would
#: change no bit.
comptime INT8_MMA_K_TILE = 32

#: One warp owns one 16 x 16 output tile: two m16n8 halves on NVIDIA, one
#: 16x16 MFMA on AMD.
comptime INT8_MMA_TILE = 16

#: Warps per block, 2 x 2, so a block owns a 32 x 32 output tile.
comptime INT8_MMA_WARPS_M = 2
comptime INT8_MMA_WARPS_N = 2
comptime INT8_MMA_BLOCK_TILE_M = INT8_MMA_WARPS_M * INT8_MMA_TILE
comptime INT8_MMA_BLOCK_TILE_N = INT8_MMA_WARPS_N * INT8_MMA_TILE

#: Threads per block. `WARP_SIZE` is one value per build (32 on NVIDIA, 64
#: on CDNA), the same on the host launch and inside the kernel.
comptime INT8_MMA_TPB = INT8_MMA_WARPS_M * INT8_MMA_WARPS_N * WARP_SIZE


def int8_mma_admits(m: Int, n: Int, k: Int) -> Bool:
    """Whether the matrix-unit plan serves the shape. Every shape the
    profile accepts (positive extents, `k <= INT8_MAX_K`) is admitted: the
    padding rule handles every `k`, and rows and columns beyond `m`, `n`
    are masked, so no shape is refused by the plan that the profile does
    not refuse by name. Kept as a function so the dispatcher's sentence
    reads `row and shape`, and so a later scheduling threshold (a decode
    row that the flat kernel serves faster, if one is ever measured) has a
    place to live without touching the dispatcher."""
    return m > 0 and n > 0 and k > 0 and k <= INT8_MAX_K


# ===========================================================================
# fragment loads: zero codes beyond k, zero codes beyond the row count
# ===========================================================================


@always_inline
def _pack4(
    p: MutPointer[Int8, MutAnyOrigin], row: Int, k0: Int, rows: Int, k: Int
) -> Int32:
    """Four consecutive codes of `row` starting at `k0`, packed into one
    32-bit register, element `i` in byte `i` (little-endian, the order the
    PTX fragment expects). A row at or beyond `rows` and a column at or
    beyond `k` read as the ZERO CODE (the padding rule). The vector load is
    taken only when it is aligned (`k % 4 == 0` makes every row start a
    multiple of four bytes, and `k0` is always a multiple of four) and
    entirely inside the row."""
    if row >= rows:
        return Int32(0)
    var base = row * k
    if k0 + 4 <= k and (k & 3) == 0:
        return bitcast[DType.int32, 1](p.unsafe_load[width=4](base + k0))[0]
    var v = SIMD[DType.int8, 4](0)
    for i in range(4):
        if k0 + i < k:
            v[i] = p.unsafe_load(base + k0 + i)
    return bitcast[DType.int32, 1](v)[0]


@always_inline
def _pack8(
    p: MutPointer[Int8, MutAnyOrigin], row: Int, k0: Int, rows: Int, k: Int
) -> Int64:
    """Eight consecutive codes of `row` from `k0` in one 64-bit register,
    element `i` in byte `i`, the order the MFMA i8 operand expects. The
    same padding rule and the same alignment discipline as `_pack4`
    (`k % 8 == 0`; `k0` is always a multiple of eight)."""
    if row >= rows:
        return Int64(0)
    var base = row * k
    if k0 + 8 <= k and (k & 7) == 0:
        return bitcast[DType.int64, 1](p.unsafe_load[width=8](base + k0))[0]
    var v = SIMD[DType.int8, 8](0)
    for i in range(8):
        if k0 + i < k:
            v[i] = p.unsafe_load(base + k0 + i)
    return bitcast[DType.int64, 1](v)[0]


@always_inline
def _store_cell(
    c: MutPointer[Float32, MutAnyOrigin],
    ea: MutPointer[Int32, MutAnyOrigin],
    eb: MutPointer[Int32, MutAnyOrigin],
    acc: Int32,
    i: Int,
    j: Int,
    m: Int,
    n: Int,
):
    """The dequantization seam (L-5, L-6) and the store, masked to the
    output. Character for character the flat kernel's epilogue."""
    if i >= m or j >= n:
        return
    var out = dequant_int8_pinned(acc, Int(ea.unsafe_load(i)) + Int(eb.unsafe_load(j)))
    comptime if INT8_MMA_SABOTAGE:
        out = gemm_oracle_sabotage_value_flip(out)
    c.unsafe_store(i * n + j, out)


# ===========================================================================
# the two vendor realizations of one warp's 16 x 16 tile
# ===========================================================================


@always_inline
def _imma_m16n8k32(
    a0: Int32,
    a1: Int32,
    a2: Int32,
    a3: Int32,
    b0: Int32,
    b1: Int32,
    c: SIMD[DType.int32, 4],
) -> SIMD[DType.int32, 4]:
    """One `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`. The four
    result registers come back as a pack and are re-spelled as a vector."""
    var r = llvm_intrinsic[
        "llvm.nvvm.mma.m16n8k32.row.col.s8",
        _RegisterPackType[Int32, Int32, Int32, Int32],
        has_side_effect=False,
    ](a0, a1, a2, a3, b0, b1, c[0], c[1], c[2], c[3])
    return SIMD[DType.int32, 4](r[0], r[1], r[2], r[3])


@always_inline
def _nvidia_warp_tile(
    c: MutPointer[Float32, MutAnyOrigin],
    qa: MutPointer[Int8, MutAnyOrigin],
    ea: MutPointer[Int32, MutAnyOrigin],
    qb: MutPointer[Int8, MutAnyOrigin],
    eb: MutPointer[Int32, MutAnyOrigin],
    lane: Int,
    row0: Int,
    col0: Int,
    m: Int,
    n: Int,
    k: Int,
):
    """PTX ISA, Matrix Fragments for mma.m16n8k32 with .s8 operands:
    `groupID = lane >> 2`, `tig = lane & 3`.
      A (16 x 32, row): reg0 row groupID cols tig*4+0..3; reg1 row
        groupID+8 same cols; reg2 row groupID cols +16; reg3 row groupID+8
        cols +16.
      B (32 x 8, col): reg0 rows(k) tig*4+0..3 col groupID; reg1 k +16.
      C/D (16 x 8, s32): c0,c1 row groupID cols tig*2+0,1; c2,c3 row
        groupID+8.
    The 16-wide tile is two n8 halves sharing the A fragments."""
    var g = lane >> 2
    var t = lane & 3
    var ra0 = row0 + g
    var ra1 = row0 + g + 8
    var cb0 = col0 + g
    var cb1 = col0 + 8 + g
    var acc0 = SIMD[DType.int32, 4](0)
    var acc1 = SIMD[DType.int32, 4](0)
    for kt in range(0, k, INT8_MMA_K_TILE):
        var ka = kt + t * 4
        var a0 = _pack4(qa, ra0, ka, m, k)
        var a1 = _pack4(qa, ra1, ka, m, k)
        var a2 = _pack4(qa, ra0, ka + 16, m, k)
        var a3 = _pack4(qa, ra1, ka + 16, m, k)
        var b0 = _pack4(qb, cb0, ka, n, k)
        var b1 = _pack4(qb, cb0, ka + 16, n, k)
        acc0 = _imma_m16n8k32(a0, a1, a2, a3, b0, b1, acc0)
        var b2 = _pack4(qb, cb1, ka, n, k)
        var b3 = _pack4(qb, cb1, ka + 16, n, k)
        acc1 = _imma_m16n8k32(a0, a1, a2, a3, b2, b3, acc1)
    var jc = col0 + t * 2
    _store_cell(c, ea, eb, acc0[0], ra0, jc, m, n)
    _store_cell(c, ea, eb, acc0[1], ra0, jc + 1, m, n)
    _store_cell(c, ea, eb, acc0[2], ra1, jc, m, n)
    _store_cell(c, ea, eb, acc0[3], ra1, jc + 1, m, n)
    _store_cell(c, ea, eb, acc1[0], ra0, jc + 8, m, n)
    _store_cell(c, ea, eb, acc1[1], ra0, jc + 9, m, n)
    _store_cell(c, ea, eb, acc1[2], ra1, jc + 8, m, n)
    _store_cell(c, ea, eb, acc1[3], ra1, jc + 9, m, n)


@always_inline
def _amd_warp_tile(
    c: MutPointer[Float32, MutAnyOrigin],
    qa: MutPointer[Int8, MutAnyOrigin],
    ea: MutPointer[Int32, MutAnyOrigin],
    qb: MutPointer[Int8, MutAnyOrigin],
    eb: MutPointer[Int32, MutAnyOrigin],
    lane: Int,
    row0: Int,
    col0: Int,
    m: Int,
    n: Int,
    k: Int,
):
    """CDNA3 `v_mfma_i32_16x16x32_i8`, wave64, one instruction per k step
    of 32. Operand layout (AMD matrix instruction calculator, 16x16x32 i8):
      A[i][kk]: lane `i + 16 * (kk // 8)`, byte `kk % 8` of the i64;
      B[kk][j]: lane `j + 16 * (kk // 8)`, byte `kk % 8`;
      D[i][j]:  lane `j + 16 * (i // 4)`, register `i % 4`.
    `cbsz`, `abid` and `blgp` are 0: no broadcast, no permutation."""
    var i16 = lane & 15
    var kq = (lane >> 4) * 8
    var acc = SIMD[DType.int32, 4](0)
    for kt in range(0, k, INT8_MMA_K_TILE):
        var a = _pack8(qa, row0 + i16, kt + kq, m, k)
        var b = _pack8(qb, col0 + i16, kt + kq, n, k)
        acc = llvm_intrinsic[
            "llvm.amdgcn.mfma.i32.16x16x32.i8",
            SIMD[DType.int32, 4],
            has_side_effect=False,
        ](a, b, acc, Int32(0), Int32(0), Int32(0))
    var j = col0 + i16
    var ir = row0 + (lane >> 4) * 4
    _store_cell(c, ea, eb, acc[0], ir, j, m, n)
    _store_cell(c, ea, eb, acc[1], ir + 1, j, m, n)
    _store_cell(c, ea, eb, acc[2], ir + 2, j, m, n)
    _store_cell(c, ea, eb, acc[3], ir + 3, j, m, n)


def identical_gemm_int8_mma_kernel(
    c: MutPointer[Float32, MutAnyOrigin],
    qa: MutPointer[Int8, MutAnyOrigin],
    ea: MutPointer[Int32, MutAnyOrigin],
    qb: MutPointer[Int8, MutAnyOrigin],
    eb: MutPointer[Int32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    k_in: Int32,
):
    """OP_NT, `C[m x n] = Qa[m x k] . Qb[n x k]^T`, one 16 x 16 output tile
    per warp on the vendor's integer matrix unit, Int32 accumulation, then
    the dequantization seam. Grid `(ceil(n / 32), ceil(m / 32), 1)`, block
    `INT8_MMA_TPB`. On a target with no integer matrix unit (Metal, the
    CPU column) both branches are dead and the kernel stores nothing; the
    dispatcher never launches it there (`lib_int8_matrix_unit_for`)."""
    var m = Int(m_in)
    var n = Int(n_in)
    var k = Int(k_in)
    var warp = Int(thread_idx.x) // WARP_SIZE
    var lane = Int(lane_id())
    var wm = warp // INT8_MMA_WARPS_N
    var wn = warp - wm * INT8_MMA_WARPS_N
    var row0 = Int(block_idx.y) * INT8_MMA_BLOCK_TILE_M + wm * INT8_MMA_TILE
    var col0 = Int(block_idx.x) * INT8_MMA_BLOCK_TILE_N + wn * INT8_MMA_TILE
    # Uniform across the warp: every lane of a warp shares row0 and col0,
    # so no lane leaves an `mma.sync` / MFMA that another lane enters.
    if row0 >= m or col0 >= n:
        return
    comptime if is_nvidia_gpu():
        _nvidia_warp_tile(c, qa, ea, qb, eb, lane, row0, col0, m, n, k)
    elif is_amd_gpu():
        _amd_warp_tile(c, qa, ea, qb, eb, lane, row0, col0, m, n, k)
    else:
        return


def identical_gemm_int8_mma_into(
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut qa: DeviceBuffer[DType.int8],
    mut ea: DeviceBuffer[DType.int32],
    mut qb: DeviceBuffer[DType.int8],
    mut eb: DeviceBuffer[DType.int32],
    m: Int,
    n: Int,
    k: Int,
) raises:
    """The matrix-unit plan, always. The checks call this directly to
    compare it with the flat plan; production goes through
    `gemm_lowbit.mojo::identical_gemm_int8_into`. Refuses by name on a
    column whose kernel-matrix row says it has no integer matrix unit.
    Asynchronous."""
    comptime if not lib_int8_matrix_unit_for[TARGET_COLUMN]():
        raise Error(
            "identical_gemm_int8_mma: column " + column_name(TARGET_COLUMN)
            + " has no int8 matrix unit (kernel_matrix row"
            " lib_int8_matrix_unit_for); the flat kernel serves it"
        )
    else:
        if not int8_mma_admits(m, n, k):
            raise Error(
                "identical_gemm_int8_mma: m, n and k must be positive and k at"
                " most " + String(INT8_MAX_K) + " (contract L-7), got m="
                + String(m) + " n=" + String(n) + " k=" + String(k)
            )
        var grid_x = (n + INT8_MMA_BLOCK_TILE_N - 1) // INT8_MMA_BLOCK_TILE_N
        var grid_y = (m + INT8_MMA_BLOCK_TILE_M - 1) // INT8_MMA_BLOCK_TILE_M
        ctx.enqueue_function[identical_gemm_int8_mma_kernel](
            c.unsafe_ptr(),
            qa.unsafe_ptr(),
            ea.unsafe_ptr(),
            qb.unsafe_ptr(),
            eb.unsafe_ptr(),
            Int32(m),
            Int32(n),
            Int32(k),
            grid_dim=(grid_x, grid_y, 1),
            block_dim=(INT8_MMA_TPB, 1, 1),
        )
