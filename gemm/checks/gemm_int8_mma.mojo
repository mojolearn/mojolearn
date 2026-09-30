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

THE FRAGMENT LOADS STATE THEIR ALIGNMENT (lane/lowbit-int15, 2026-09-29,
DEVIATION 2975; found by lane/lowbit-mma-speed, job nvc3-0016). `_pack4`
took its word with `unsafe_load[width=4]` and stated no alignment, and on
NVIDIA the compiler emitted four byte loads for it; with `alignment=4` it
is ONE 32-bit load (that lane measured qkv.t512 on an H100 at 1.436 ms
before and 0.398 ms after, bits equal). An alignment is a PROMISE to the
compiler, so it is stated only where it is a fact:
  - the offset inside the buffer is a multiple of the word, which the
    function already tests (`k` a multiple of 4 or 8 makes every row start
    one, and `k0` always is);
  - the BASE of the buffer is a multiple of 8, which the LAUNCH reads off
    the two operand pointers it is about to pass (`mma_operands_aligned`)
    and hands the kernel as `aligned_in`. It is not assumed of an allocator.
Where either fails the load is the one this file always had, and beyond the
row the byte path with its zero codes. SCHEDULING: the same bytes reach the
unit whichever load fetched them, and the gates hold it to that.

THE SABOTAGE ARM. `-D MOJOLEARN_LOWBIT_SABOTAGE=1` flips the value of every
cell this kernel stores, exactly as the flat kernel does (DEVIATION 2908),
so a sabotage build fails `check_int8_device_matches_oracle` through either
path and fails `check_int8_mma_matches_flat` at its oracle comparison.
"""

from std.gpu import WARP_SIZE, block_dim, block_idx, lane_id, thread_idx
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

#: DEVIATION 2975: `-D MOJOLEARN_INT8_MMA_UNSTATED_LOADS=1` keeps the fragment
#: loads as they were before the alignment was stated, on every launch, so
#: one box can run both and compare bits and time. SCHEDULING.
comptime INT8_MMA_UNSTATED_LOADS = is_defined["MOJOLEARN_INT8_MMA_UNSTATED_LOADS"]()

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

#: lane/neural-net-experiment (2026-09-30): the REUSE plan. In the reference
#: plan one warp owns one 16 x 16 tile and reads every fragment of both
#: operands from device memory at every k step (eight 4-byte or two 8-byte
#: loads per unit step, one unit step per load pair). Here a warp owns a
#: 32 x 32 tile: at each k step it loads the fragments of TWO A row-tiles
#: and TWO B row-tiles and runs the four unit tiles they combine into, so
#: every fragment feeds two unit tiles instead of one, and the next step's
#: fragments are loaded before this step's units run (the loads and the
#: units are independent). The bench board's gemm-int8 lane (4096^3) took
#: 1,799 ms on an MI325X and 2,917 ms on an L40S against torch's int8 at
#: 48 ms on the L40S: the reference plan is bound by its loads, not the
#: unit. Contract L-9: an Int32 sum of exact int8 products is the same
#: integer under every order, grouping and tile, so no plan here can move
#: a bit; the epilogue stays `_store_cell` with the reference's arguments.
#: MOJOLEARN_INT8_MMA_REFERENCE=1 (a build define) keeps the reference
#: plan for the A/B and the equality gate.
comptime INT8_MMA_REUSE = not is_defined["MOJOLEARN_INT8_MMA_REFERENCE"]()
comptime INT8_MMA_REUSE_WARP_TILE = 2 * INT8_MMA_TILE
comptime INT8_MMA_REUSE_BLOCK_TILE_M = INT8_MMA_WARPS_M * INT8_MMA_REUSE_WARP_TILE
comptime INT8_MMA_REUSE_BLOCK_TILE_N = INT8_MMA_WARPS_N * INT8_MMA_REUSE_WARP_TILE


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


def mma_operands_aligned(a_address: Int, b_address: Int) -> Bool:
    """Whether the BASES of two operand buffers are both multiples of 8
    bytes, given as the addresses of the pointers the launch is about to
    pass (`Int(buffer.unsafe_ptr())`). The fragment loads state an
    alignment only when this is true (DEVIATION 2975)."""
    return ((a_address | b_address) & 7) == 0


@always_inline
def _pack4(
    p: MutPointer[Int8, MutAnyOrigin],
    row: Int,
    k0: Int,
    rows: Int,
    k: Int,
    aligned: Bool = False,
) -> Int32:
    """Four consecutive codes of `row` starting at `k0`, packed into one
    32-bit register, element `i` in byte `i` (little-endian, the order the
    PTX fragment expects). A row at or beyond `rows` and a column at or
    beyond `k` read as the ZERO CODE (the padding rule). The vector load is
    taken only when it is aligned inside the buffer (`k % 4 == 0` makes
    every row start a multiple of four bytes, and `k0` is always a multiple
    of four) and entirely inside the row; it STATES that alignment only
    when the launch found the buffer's base aligned too (`aligned`,
    DEVIATION 2975)."""
    if row >= rows:
        return Int32(0)
    var base = row * k
    if k0 + 4 <= k and (k & 3) == 0:
        if aligned:
            return bitcast[DType.int32, 1](p.unsafe_load[width=4, alignment=4](base + k0))[0]
        return bitcast[DType.int32, 1](p.unsafe_load[width=4](base + k0))[0]
    var v = SIMD[DType.int8, 4](0)
    for i in range(4):
        if k0 + i < k:
            v[i] = p.unsafe_load(base + k0 + i)
    return bitcast[DType.int32, 1](v)[0]


@always_inline
def _pack8(
    p: MutPointer[Int8, MutAnyOrigin],
    row: Int,
    k0: Int,
    rows: Int,
    k: Int,
    aligned: Bool = False,
) -> Int64:
    """Eight consecutive codes of `row` from `k0` in one 64-bit register,
    element `i` in byte `i`, the order the MFMA i8 operand expects. The
    same padding rule and the same alignment discipline as `_pack4`
    (`k % 8 == 0`; `k0` is always a multiple of eight; the alignment is
    stated only under `aligned`)."""
    if row >= rows:
        return Int64(0)
    var base = row * k
    if k0 + 8 <= k and (k & 7) == 0:
        if aligned:
            return bitcast[DType.int64, 1](p.unsafe_load[width=8, alignment=8](base + k0))[0]
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
    aligned: Bool,
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
        var a0 = _pack4(qa, ra0, ka, m, k, aligned)
        var a1 = _pack4(qa, ra1, ka, m, k, aligned)
        var a2 = _pack4(qa, ra0, ka + 16, m, k, aligned)
        var a3 = _pack4(qa, ra1, ka + 16, m, k, aligned)
        var b0 = _pack4(qb, cb0, ka, n, k, aligned)
        var b1 = _pack4(qb, cb0, ka + 16, n, k, aligned)
        acc0 = _imma_m16n8k32(a0, a1, a2, a3, b0, b1, acc0)
        var b2 = _pack4(qb, cb1, ka, n, k, aligned)
        var b3 = _pack4(qb, cb1, ka + 16, n, k, aligned)
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
    aligned: Bool,
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
        var a = _pack8(qa, row0 + i16, kt + kq, m, k, aligned)
        var b = _pack8(qb, col0 + i16, kt + kq, n, k, aligned)
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


@always_inline
def _nvidia_warp_tile_reuse(
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
    aligned: Bool,
):
    """`_nvidia_warp_tile` over a 32 x 32 warp tile: two A row-tiles (rows
    row0.. and row0+16..) and four B n8 column groups, the same fragment
    layout, each A fragment feeding four `mma.sync` and each B fragment
    two. The next k step's fragments are loaded before this step's units
    run. Same Int32 sums, same epilogue."""
    var g = lane >> 2
    var t = lane & 3
    var acc = InlineArray[SIMD[DType.int32, 4], 8](fill=SIMD[DType.int32, 4](0))
    # fragments of step kt: a[i][0..3] for A row-tile i, b[j][0..1] for B group j
    var ka = t * 4
    var a00 = _pack4(qa, row0 + g, ka, m, k, aligned)
    var a01 = _pack4(qa, row0 + g + 8, ka, m, k, aligned)
    var a02 = _pack4(qa, row0 + g, ka + 16, m, k, aligned)
    var a03 = _pack4(qa, row0 + g + 8, ka + 16, m, k, aligned)
    var a10 = _pack4(qa, row0 + 16 + g, ka, m, k, aligned)
    var a11 = _pack4(qa, row0 + 24 + g, ka, m, k, aligned)
    var a12 = _pack4(qa, row0 + 16 + g, ka + 16, m, k, aligned)
    var a13 = _pack4(qa, row0 + 24 + g, ka + 16, m, k, aligned)
    var b00 = _pack4(qb, col0 + g, ka, n, k, aligned)
    var b01 = _pack4(qb, col0 + g, ka + 16, n, k, aligned)
    var b10 = _pack4(qb, col0 + 8 + g, ka, n, k, aligned)
    var b11 = _pack4(qb, col0 + 8 + g, ka + 16, n, k, aligned)
    var b20 = _pack4(qb, col0 + 16 + g, ka, n, k, aligned)
    var b21 = _pack4(qb, col0 + 16 + g, ka + 16, n, k, aligned)
    var b30 = _pack4(qb, col0 + 24 + g, ka, n, k, aligned)
    var b31 = _pack4(qb, col0 + 24 + g, ka + 16, n, k, aligned)
    for kt in range(0, k, INT8_MMA_K_TILE):
        var kn = kt + INT8_MMA_K_TILE + t * 4
        var more = kt + INT8_MMA_K_TILE < k
        # the next step's fragments, loaded now (zero codes past k)
        var na00 = _pack4(qa, row0 + g, kn, m, k, aligned) if more else Int32(0)
        var na01 = _pack4(qa, row0 + g + 8, kn, m, k, aligned) if more else Int32(0)
        var na02 = _pack4(qa, row0 + g, kn + 16, m, k, aligned) if more else Int32(0)
        var na03 = _pack4(qa, row0 + g + 8, kn + 16, m, k, aligned) if more else Int32(0)
        var na10 = _pack4(qa, row0 + 16 + g, kn, m, k, aligned) if more else Int32(0)
        var na11 = _pack4(qa, row0 + 24 + g, kn, m, k, aligned) if more else Int32(0)
        var na12 = _pack4(qa, row0 + 16 + g, kn + 16, m, k, aligned) if more else Int32(0)
        var na13 = _pack4(qa, row0 + 24 + g, kn + 16, m, k, aligned) if more else Int32(0)
        var nb00 = _pack4(qb, col0 + g, kn, n, k, aligned) if more else Int32(0)
        var nb01 = _pack4(qb, col0 + g, kn + 16, n, k, aligned) if more else Int32(0)
        var nb10 = _pack4(qb, col0 + 8 + g, kn, n, k, aligned) if more else Int32(0)
        var nb11 = _pack4(qb, col0 + 8 + g, kn + 16, n, k, aligned) if more else Int32(0)
        var nb20 = _pack4(qb, col0 + 16 + g, kn, n, k, aligned) if more else Int32(0)
        var nb21 = _pack4(qb, col0 + 16 + g, kn + 16, n, k, aligned) if more else Int32(0)
        var nb30 = _pack4(qb, col0 + 24 + g, kn, n, k, aligned) if more else Int32(0)
        var nb31 = _pack4(qb, col0 + 24 + g, kn + 16, n, k, aligned) if more else Int32(0)
        # this step's eight unit steps: A row-tile i (0, 1) x B group j (0..3)
        acc[0] = _imma_m16n8k32(a00, a01, a02, a03, b00, b01, acc[0])
        acc[1] = _imma_m16n8k32(a00, a01, a02, a03, b10, b11, acc[1])
        acc[2] = _imma_m16n8k32(a00, a01, a02, a03, b20, b21, acc[2])
        acc[3] = _imma_m16n8k32(a00, a01, a02, a03, b30, b31, acc[3])
        acc[4] = _imma_m16n8k32(a10, a11, a12, a13, b00, b01, acc[4])
        acc[5] = _imma_m16n8k32(a10, a11, a12, a13, b10, b11, acc[5])
        acc[6] = _imma_m16n8k32(a10, a11, a12, a13, b20, b21, acc[6])
        acc[7] = _imma_m16n8k32(a10, a11, a12, a13, b30, b31, acc[7])
        a00 = na00
        a01 = na01
        a02 = na02
        a03 = na03
        a10 = na10
        a11 = na11
        a12 = na12
        a13 = na13
        b00 = nb00
        b01 = nb01
        b10 = nb10
        b11 = nb11
        b20 = nb20
        b21 = nb21
        b30 = nb30
        b31 = nb31
    # C/D (16 x 8, s32): c0, c1 row g cols t*2 + 0, 1; c2, c3 row g + 8
    comptime for i in range(2):
        comptime for j in range(4):
            var r0 = row0 + 16 * i + g
            var jc = col0 + 8 * j + t * 2
            var v = acc[i * 4 + j]
            _store_cell(c, ea, eb, v[0], r0, jc, m, n)
            _store_cell(c, ea, eb, v[1], r0, jc + 1, m, n)
            _store_cell(c, ea, eb, v[2], r0 + 8, jc, m, n)
            _store_cell(c, ea, eb, v[3], r0 + 8, jc + 1, m, n)


@always_inline
def _amd_warp_tile_reuse(
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
    aligned: Bool,
):
    """`_amd_warp_tile` over a 32 x 32 warp tile: two A operands (rows
    row0.. and row0+16..) and two B operands (columns col0.. and
    col0+16..) per k step, the same lane layout, four MFMAs, each operand
    feeding two. The next k step's operands are loaded before this step's
    MFMAs run. Same Int32 sums, same epilogue."""
    var i16 = lane & 15
    var kq = (lane >> 4) * 8
    var acc00 = SIMD[DType.int32, 4](0)
    var acc01 = SIMD[DType.int32, 4](0)
    var acc10 = SIMD[DType.int32, 4](0)
    var acc11 = SIMD[DType.int32, 4](0)
    var a0 = _pack8(qa, row0 + i16, kq, m, k, aligned)
    var a1 = _pack8(qa, row0 + 16 + i16, kq, m, k, aligned)
    var b0 = _pack8(qb, col0 + i16, kq, n, k, aligned)
    var b1 = _pack8(qb, col0 + 16 + i16, kq, n, k, aligned)
    for kt in range(0, k, INT8_MMA_K_TILE):
        var kn = kt + INT8_MMA_K_TILE + kq
        var more = kt + INT8_MMA_K_TILE < k
        var na0 = _pack8(qa, row0 + i16, kn, m, k, aligned) if more else Int64(0)
        var na1 = _pack8(qa, row0 + 16 + i16, kn, m, k, aligned) if more else Int64(0)
        var nb0 = _pack8(qb, col0 + i16, kn, n, k, aligned) if more else Int64(0)
        var nb1 = _pack8(qb, col0 + 16 + i16, kn, n, k, aligned) if more else Int64(0)
        acc00 = llvm_intrinsic[
            "llvm.amdgcn.mfma.i32.16x16x32.i8",
            SIMD[DType.int32, 4],
            has_side_effect=False,
        ](a0, b0, acc00, Int32(0), Int32(0), Int32(0))
        acc01 = llvm_intrinsic[
            "llvm.amdgcn.mfma.i32.16x16x32.i8",
            SIMD[DType.int32, 4],
            has_side_effect=False,
        ](a0, b1, acc01, Int32(0), Int32(0), Int32(0))
        acc10 = llvm_intrinsic[
            "llvm.amdgcn.mfma.i32.16x16x32.i8",
            SIMD[DType.int32, 4],
            has_side_effect=False,
        ](a1, b0, acc10, Int32(0), Int32(0), Int32(0))
        acc11 = llvm_intrinsic[
            "llvm.amdgcn.mfma.i32.16x16x32.i8",
            SIMD[DType.int32, 4],
            has_side_effect=False,
        ](a1, b1, acc11, Int32(0), Int32(0), Int32(0))
        a0 = na0
        a1 = na1
        b0 = nb0
        b1 = nb1
    # D[i][j]: lane j + 16 (i // 4), register i % 4 -- per 16 x 16 tile
    var ir = (lane >> 4) * 4
    comptime for r in range(4):
        _store_cell(c, ea, eb, acc00[r], row0 + ir + r, col0 + i16, m, n)
        _store_cell(c, ea, eb, acc01[r], row0 + ir + r, col0 + 16 + i16, m, n)
        _store_cell(c, ea, eb, acc10[r], row0 + 16 + ir + r, col0 + i16, m, n)
        _store_cell(c, ea, eb, acc11[r], row0 + 16 + ir + r, col0 + 16 + i16, m, n)


def identical_gemm_int8_mma_reuse_kernel(
    c: MutPointer[Float32, MutAnyOrigin],
    qa: MutPointer[Int8, MutAnyOrigin],
    ea: MutPointer[Int32, MutAnyOrigin],
    qb: MutPointer[Int8, MutAnyOrigin],
    eb: MutPointer[Int32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    k_in: Int32,
    aligned_in: Int32,
):
    """`identical_gemm_int8_mma_kernel` with a 32 x 32 tile per warp (the
    reuse plan, `INT8_MMA_REUSE`): grid `(ceil(n / 64), ceil(m / 64), 1)`,
    block `INT8_MMA_TPB`, the same 2 x 2 warps. A warp whose tile starts
    beyond m or n returns whole (uniform across the warp)."""
    var m = Int(m_in)
    var n = Int(n_in)
    var k = Int(k_in)
    var aligned = aligned_in != Int32(0)
    var warp = Int(thread_idx.x) // WARP_SIZE
    var lane = Int(lane_id())
    var wm = warp // INT8_MMA_WARPS_N
    var wn = warp - wm * INT8_MMA_WARPS_N
    var row0 = Int(block_idx.y) * INT8_MMA_REUSE_BLOCK_TILE_M + wm * INT8_MMA_REUSE_WARP_TILE
    var col0 = Int(block_idx.x) * INT8_MMA_REUSE_BLOCK_TILE_N + wn * INT8_MMA_REUSE_WARP_TILE
    if row0 >= m or col0 >= n:
        return
    comptime if is_nvidia_gpu():
        _nvidia_warp_tile_reuse(c, qa, ea, qb, eb, lane, row0, col0, m, n, k, aligned)
    elif is_amd_gpu():
        _amd_warp_tile_reuse(c, qa, ea, qb, eb, lane, row0, col0, m, n, k, aligned)
    else:
        return


def identical_gemm_int8_mma_kernel(
    c: MutPointer[Float32, MutAnyOrigin],
    qa: MutPointer[Int8, MutAnyOrigin],
    ea: MutPointer[Int32, MutAnyOrigin],
    qb: MutPointer[Int8, MutAnyOrigin],
    eb: MutPointer[Int32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    k_in: Int32,
    aligned_in: Int32,
):
    """OP_NT, `C[m x n] = Qa[m x k] . Qb[n x k]^T`, one 16 x 16 output tile
    per warp on the vendor's integer matrix unit, Int32 accumulation, then
    the dequantization seam. Grid `(ceil(n / 32), ceil(m / 32), 1)`, block
    `INT8_MMA_TPB`. `aligned_in` is 1 when the launch found both operand
    buffers' bases aligned (DEVIATION 2975). On a target with no integer
    matrix unit (Metal, the CPU column) both branches are dead and the
    kernel stores nothing; the dispatcher never launches it there
    (`lib_int8_matrix_unit_for`)."""
    var m = Int(m_in)
    var n = Int(n_in)
    var k = Int(k_in)
    var aligned = aligned_in != Int32(0)
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
        _nvidia_warp_tile(c, qa, ea, qb, eb, lane, row0, col0, m, n, k, aligned)
    elif is_amd_gpu():
        _amd_warp_tile(c, qa, ea, qb, eb, lane, row0, col0, m, n, k, aligned)
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
        var aligned = Int32(0)
        comptime if not INT8_MMA_UNSTATED_LOADS:
            if mma_operands_aligned(Int(qa.unsafe_ptr()), Int(qb.unsafe_ptr())):
                aligned = Int32(1)
        comptime if INT8_MMA_REUSE:
            # the reuse plan: a 64 x 64 block tile (2 x 2 warps of 32 x 32)
            var rgrid_x = (n + INT8_MMA_REUSE_BLOCK_TILE_N - 1) // INT8_MMA_REUSE_BLOCK_TILE_N
            var rgrid_y = (m + INT8_MMA_REUSE_BLOCK_TILE_M - 1) // INT8_MMA_REUSE_BLOCK_TILE_M
            ctx.enqueue_function[identical_gemm_int8_mma_reuse_kernel](
                c.unsafe_ptr(),
                qa.unsafe_ptr(),
                ea.unsafe_ptr(),
                qb.unsafe_ptr(),
                eb.unsafe_ptr(),
                Int32(m),
                Int32(n),
                Int32(k),
                aligned,
                grid_dim=(rgrid_x, rgrid_y, 1),
                block_dim=(INT8_MMA_TPB, 1, 1),
            )
            return
        ctx.enqueue_function[identical_gemm_int8_mma_kernel](
            c.unsafe_ptr(),
            qa.unsafe_ptr(),
            ea.unsafe_ptr(),
            qb.unsafe_ptr(),
            eb.unsafe_ptr(),
            Int32(m),
            Int32(n),
            Int32(k),
            aligned,
            grid_dim=(grid_x, grid_y, 1),
            block_dim=(INT8_MMA_TPB, 1, 1),
        )
