# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`mojolearn.identical.gemm.int8i32.v1` on the integer matrix units, TUNED:
the schedule of `gemm/checks/gemm_int8_mma.mojo` changed and its bits kept.

Lane lane/lowbit-mma-speed, 2026-09-29. Contract clause L-9 of
`gemm/IDENTICAL_LOWBIT_CONTRACT.md`. The reference unit plan is
`gemm/checks/gemm_int8_mma.mojo` and is not edited; the flat plan and the
dispatcher are `gemm/checks/gemm_lowbit.mojo`; the answer is
`gemm/host/gemm_lowbit_oracle.mojo::gemm_int8_oracle`; the gate is
`gemm/checks/gemm_int8_mma_tuned_check.mojo`. NOTHING HERE IS DISPATCHED:
`identical_gemm_int8_into` still takes the reference plan.

WHY IT EXISTS. lane/lowbit-units timed the reference unit plan on an H100:
1.38 to 1.66 times fp32.v1's time at the 512-token rows (qkv.t512: 6.0
T MAC/s against 8.3). In that plan one warp owns one 16 x 16 output tile
and reads every fragment of both operands from device memory at every k
step: eight four-byte loads for two unit steps.

WHY NO PLAN HERE CAN MOVE A BIT. Clause L-9's argument, unchanged: a product
of two int8 codes is an exact integer, an Int32 sum of them cannot overflow
for `k <= INT8_MAX_K`, and a sum of exact integers is the same integer under
every order, grouping and tile. What this file changes is which thread holds
which cell, where a code waits before the unit reads it (device memory or
threadgroup memory), how many bytes one load carries, and how many unit
steps run between two barriers. None of it reaches the Int32 a cell ends
with, and the only floating steps stay `dequant_int8_pinned` (L-5, L-6),
called with the reference's arguments.

THE PADDING RULE is the reference's: a step at or beyond `k`, and a row at
or beyond `m` or `n`, are the ZERO CODE, and cells beyond `m`, `n` are
masked at the store. A staged window is written WHOLE, zero codes included,
because threadgroup memory still holds the window before it.

THE LEVERS, each a comptime parameter, so every plan below is one
instantiation of one kernel and the timing harness runs them as arms of one
binary:

    FM, FN   16 x 16 unit tiles per warp, along m and along n: a warp owns
             a `16 FM x 16 FN` output tile. On NVIDIA a unit tile is two
             m16n8k32 steps that share their A fragments; on AMD it is one
             16x16x32 MFMA.
    WM, WN   warps per block: a block owns `BM x BN = 16 FM WM x 16 FN WN`
             and stages `BM + BN` operand rows per window.
    KB       k steps staged per window, a multiple of the unit's 32: `KB /
             32` unit steps between two barriers, unrolled.
    LW       bytes per staging load and store, 4 or 16.

THE STAGED ROW is `KB + 16` bytes. The sixteen bytes of padding are never
read; they move each row's bank. On NVIDIA one fragment load is eight rows
of sixteen bytes read by one warp, and with `KB` a multiple of 32 the eight
rows' first words then fall in eight different groups of four banks.

THE DIRECT KERNEL (`identical_gemm_int8_mma_direct_kernel`) is the
reference's schedule, one warp one tile, nothing staged, with ONE thing
changed per instantiation. It exists to test what is NVIDIA-specific in the
reference before anything is redesigned around a guess (the brief,
2026-09-29: the same schedule takes 0.28 of fp32.v1's time on the AMD unit
and 1.39 on the NVIDIA one). Its PROBE instantiations compute a WRONG
product on purpose, to time a part of the kernel alone; they are named
`probe`, no digest of theirs is compared with anything, and no dispatcher
may ever name one.

THE ALIGNMENT OF A LOAD IS STATED ONLY WHERE IT IS TRUE. A staging load
of `LW` bytes is one aligned load when `k` is a multiple of `LW` (every row
then starts on one), the load lies inside the row, AND the base of the
buffer is itself a multiple of `LW` bytes. The first two are tested in the
kernel; the third is read at the launch from the pointers it is about to
pass (`_bases_aligned`) and handed to the kernel as `aligned_in`. Where any
of the three fails the codes are read one byte at a time, which every
ragged `k` of the gate already does. `-D MOJOLEARN_INT8_TUNED_UNSTATED=1`
answers "not aligned" at every launch, so one box can run the byte path on
every shape and compare.

FOUR PRODUCTS, ONE STAGING (`identical_gemm_int8_pieces_tuned_kernel`). The
fifteen-bit profile (lane/lowbit-int15) splits each operand into two int8
planes, high and low, and needs the four products HH, HL, LH, LL of one
GEMM. Four launches of the kernel above stage each operand four times and
run the epilogue four times. This kernel stages the two planes of each
operand ONCE per window and runs the four unit steps of a tile on the same
fragments, into three Int32 accumulators per cell: HH, HL + LH (the two
share their power of two in the recombination, so they share a register),
LL. It stores the three sums and NOTHING ELSE: the recombination in Int64,
the pinned conversion and the scale are the fifteen-bit profile's seams and
live in its own file. THE FUSED FORM (`FUSED` True, the orchestrator's
interface with lane/lowbit-int15) is the same kernel whose last step hands
each cell's three sums to `int15_store_cell` (lane/lowbit-int15's
`gemm/checks/gemm_int15_epilogue.mojo`), which applies the mask,
the recombination, the pinned seam and the scale: one launch where the sums
form needs a second to read twelve bytes a cell back. This file states no
float rule in either form. THE TWO-PAGE FORM (`PIPE` True) stages the next
window with `cp.async` while the unit steps read this one. Every sum is an exact integer under
`INT8_PIECES_MAX_K` for the operands that bound is stated for (low planes
in [0, 127]), so every plan of it returns the same three integers.

THE SABOTAGE ARMS.
  `-D MOJOLEARN_INT8_PIECES_SABOTAGE=1` pairs the wrong fragments in the
      four-product kernel: the middle accumulator takes HL twice and never
      LH. A SCHEDULING defect that the one-product gate cannot see.
  `-D MOJOLEARN_INT8_TUNED_SABOTAGE=1` breaks the padding rule of the
      staging: a load that lies wholly beyond `k` or beyond the operand's
      rows stores NOTHING, so the ragged tail of the last window reads the
      codes the window before it left. A SCHEDULING defect, the kind this
      file can have. It reaches a shape whose last window holds a load
      wholly beyond `k` where an earlier window wrote, and it cannot reach
      a shape whose `k` is a whole number of windows, which must still pass.
  `-D MOJOLEARN_LOWBIT_SABOTAGE=1` flips the value of every cell stored, as
      in the flat kernel and the reference unit kernel (DEVIATION 2908).
"""

from std.gpu import WARP_SIZE, block_idx, lane_id, thread_idx
from std.gpu.primitives.warp import shuffle_xor
from std.memory import bitcast, stack_allocation
from std.sys import is_defined, llvm_intrinsic
from std.sys._assembly import inlined_assembly
from std.sys.info import is_amd_gpu, is_nvidia_gpu
from std.sys.intrinsics import _RegisterPackType
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.kernel_matrix import (
    COLUMN_NVIDIA,
    TARGET_COLUMN,
    column_name,
    column_shared_limit,
    lib_int8_matrix_unit_for,
)
from checks.numerics import dequant_int8_pinned, int8_row_exponent
from gemm.checks.gemm_int8_mma import _imma_m16n8k32, _pack4, int8_mma_admits
from gemm.checks.gemm_int15_epilogue import int15_store_cell
from gemm.checks.quantize_int8_par import _absmax_step, _code
from gemm.host.gemm_lowbit_oracle import INT8_MAX_K
from gemm.host.gemm_oracle import gemm_oracle_sabotage_value_flip

#: The scheduling arm: a staging load wholly outside the operand stores
#: nothing. Off in every build that does not name it.
comptime INT8_TUNED_SABOTAGE = is_defined["MOJOLEARN_INT8_TUNED_SABOTAGE"]()

#: DEVIATION 2908, the value arm, read from the define the other int8 plans
#: read.
comptime INT8_TUNED_VALUE_SABOTAGE = is_defined["MOJOLEARN_LOWBIT_SABOTAGE"]()

#: `-D MOJOLEARN_INT8_TUNED_UNSTATED=1`: no launch states an alignment, so
#: every staging load is the byte path. SCHEDULING.
comptime INT8_TUNED_UNSTATED = is_defined["MOJOLEARN_INT8_TUNED_UNSTATED"]()

#: The four-product kernel's defect arm: HL twice, LH never.
comptime INT8_PIECES_SABOTAGE = is_defined["MOJOLEARN_INT8_PIECES_SABOTAGE"]()
#: The decode kernel's quantizer arm: the first level of the warp's absmax
#: butterfly skipped. The parallel quantizer's own arm
#: (`MOJOLEARN_QUANT_PAR_SABOTAGE`) turns it on too.
comptime INT8_DECODE_QA_SABOTAGE = is_defined["MOJOLEARN_QUANT_PAR_SABOTAGE"]()

#: The largest `k` of the four-product kernel, the fifteen-bit profile's
#: own (`INT15_MAX_K`, its clause W-4), and THE OPERANDS IT IS STATED FOR:
#: HIGH planes of any int8, LOW planes in [0, 127], which is what the
#: profile's split writes (its clause W-3). Then
#:     HH        at most 128 * 128 = 16384 a step     exact to k = 131071
#:     HL + LH   at most 2 * 128 * 127 = 32512 a step  exact to k = 66052
#:     LL        at most 127 * 127 = 16129 a step
#: and 65536 is the power of two below the smallest. The kernel reads
#: codes, not ranges: a caller whose LOW planes hold negative codes has the
#: middle sum exact to k = 65535 only (`2 * 16384 * 65535 = 2147450880`),
#: and `INT8_PIECES_MAX_K_ANY_INT8` names that bound.
comptime INT8_PIECES_MAX_K = 65536
comptime INT8_PIECES_MAX_K_ANY_INT8 = 65535

#: The unit's k step on both vendors. SCHEDULING.
comptime INT8_TUNED_K_TILE = 32

#: The side of one unit tile: two m16n8 halves on NVIDIA, one MFMA on AMD.
comptime INT8_TUNED_TILE = 16

#: Bytes of padding after a staged row (see THE STAGED ROW).
comptime INT8_TUNED_ROW_PAD = 16

#: Whether this build has the unit at all.
comptime INT8_TUNED_AVAILABLE = lib_int8_matrix_unit_for[TARGET_COLUMN]()

#: Whether this build has the direct kernel's probes: NVIDIA only, since
#: what they test is NVIDIA's fragment layout.
comptime INT8_DIRECT_AVAILABLE = TARGET_COLUMN == COLUMN_NVIDIA

#: The staged plans. SCHEDULING, every one; the gate runs every one on
#: every shape. The name says what changed against the plan before it.
comptime INT8_TUNED_PLAN_STAGED = 0  #: 16x16 per warp, 2x2 warps, KB 32, LW 4
comptime INT8_TUNED_PLAN_FRAG2 = 1  #: 32x32 per warp
comptime INT8_TUNED_PLAN_FRAG4 = 2  #: 64x64 per warp
comptime INT8_TUNED_PLAN_WIDE = 3  #: 64x64 per warp, 2x4 warps
comptime INT8_TUNED_PLAN_PACK = 4  #: FRAG4 with 16-byte staging loads
comptime INT8_TUNED_PLAN_K64 = 5  #: PACK with 64 k steps per window
comptime INT8_TUNED_PLAN_K128 = 6  #: PACK with 128 k steps per window
comptime INT8_TUNED_PLAN_WIDE_K64 = 7  #: WIDE with 16-byte loads and KB 64
comptime INT8_TUNED_PLAN_ROW = 8  #: 16x64 per warp, 1x4 warps: the decode rows
#: The second round (2026-09-29, after the first timing): the SAME block
#: cut into MORE WARPS. At the 512-token rows a 128 x 128 block is about one
#: block per multiprocessor of the H100, so four warps a block leave the
#: unit idle most of the time.
comptime INT8_TUNED_PLAN_SMALL_K64 = 9  #: STAGED with 16-byte loads and KB 64
comptime INT8_TUNED_PLAN_FRAG2_K64 = 10  #: FRAG2 with 16-byte loads and KB 64
comptime INT8_TUNED_PLAN_WARPS16 = 11  #: 32x32 per warp, 4x4 warps: block 128x128
comptime INT8_TUNED_PLAN_WARPS32 = 12  #: 16x32 per warp, 8x4 warps: block 128x128
#: 16x32 per warp, 4x8 warps: block 64x256. NOT LAUNCHED ANYWHERE: the H100
#: refused every launch of it (job nvc3-0019) and of the plan it replaced,
#: 32x32 per warp in 4x8 warps (job nvc3-0018), with
#: CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES. The PTX counter (job nvc3-0020)
#: reads 85 and 108 registers a thread for the two; 1024 threads of either
#: are above the 65536 registers of one multiprocessor, and the 8x4 plan
#: that launches reads 47. The plan keeps its number and
#: `int8_tuned_plan_available` answers False for it.
comptime INT8_TUNED_PLAN_WARPS32_WIDE = 13
#: The third round (2026-09-29): TALL blocks. A block reads `BM` rows of
#: the left operand and `BN` of the right per window, so over the whole
#: product the right operand (the weights, `n k` codes, the large one) is
#: read `m / BM` times and the left `n / BN` times. At the wide rows the
#: 128 x 128 block reads the weights four times over, about as many bytes a
#: second as the box's memory gives. A taller, narrower block of the same
#: sixteen warps reads them twice, or once.
comptime INT8_TUNED_PLAN_TALL256 = 14  #: 32x32 per warp, 8x2 warps: block 256x64
comptime INT8_TUNED_PLAN_TALL512 = 15  #: 32x32 per warp, 16x1 warps: block 512x32
comptime INT8_TUNED_PLAN_TALL512_N16 = 16  #: 32x16 per warp, 16x1 warps: block 512x16
comptime INT8_TUNED_PLAN_COUNT = 17

#: The most threads a block may hold on the columns that have the unit.
comptime INT8_TUNED_MAX_TPB = 1024

#: The four-product kernel's plans. SCHEDULING, every one.
comptime INT8_PIECES_PLAN_SMALL = 0  #: 16x16 per warp, 2x2 warps: block 32x32
comptime INT8_PIECES_PLAN_WARPS16 = 1  #: 16x32 per warp, 4x4 warps: block 64x128
comptime INT8_PIECES_PLAN_FRAG2 = 2  #: 32x32 per warp, 2x4 warps: block 64x128
comptime INT8_PIECES_PLAN_SQUARE = 3  #: 16x16 per warp, 4x4 warps: block 64x64
#: TALL blocks (see the one-product plans): both planes of the weights are
#: read `m / BM` times over.
comptime INT8_PIECES_PLAN_TALL128 = 4  #: 16x32 per warp, 8x2 warps: block 128x64, KB 32
comptime INT8_PIECES_PLAN_TALL256 = 5  #: 16x32 per warp, 16x1 warps: block 256x32, KB 32
comptime INT8_PIECES_PLAN_TALL256_N16 = 6  #: 16x16 per warp, 16x1 warps: block 256x16, KB 32
#: The launcher's geometry with 32 k steps a window: the one-page control
#: of the two-page plan below (the window is what two pages cost).
comptime INT8_PIECES_PLAN_WARPS16_K32 = 7
#: TWO PAGES, `cp.async` staging of the next window (`_pieces_block`, `PIPE`).
comptime INT8_PIECES_PLAN_PIPE_WARPS16 = 8  #: 16x32 per warp, 4x4 warps, KB 32
comptime INT8_PIECES_PLAN_PIPE_FRAG2 = 9  #: 32x32 per warp, 2x4 warps, KB 32
comptime INT8_PIECES_PLAN_PIPE_SMALL = 10  #: 16x16 per warp, 2x2 warps, KB 64
comptime INT8_PIECES_PLAN_COUNT = 11

#: Outputs of at most this many rows take the launcher's small plan: a
#: 128-row block would multiply 112 rows of zero codes for them.
comptime INT8_TUNED_ROW_MAX_M = 16

#: The direct kernel's instantiations (NVIDIA only).
comptime INT8_DIRECT_REFERENCE_LOADS = 0  #: the reference, respelled here: the control
comptime INT8_DIRECT_ALIGNED_LOADS = 1  #: fragment loads with their alignment stated
comptime INT8_DIRECT_SCALAR_ACC = 2  #: ALIGNED, accumulators as four scalars
comptime INT8_DIRECT_PROBE_HOISTED = 3  #: PROBE: fragments loaded once, not per k step
comptime INT8_DIRECT_PROBE_ONE_HALF = 4  #: PROBE: one m16n8 half per k step
comptime INT8_DIRECT_PROBE_RAW_STORE = 5  #: PROBE: the epilogue without its seam
comptime INT8_DIRECT_COUNT = 6


def int8_tuned_sabotage_name() -> String:
    comptime if INT8_TUNED_SABOTAGE:
        return String("STAGING_PAD_NOT_WRITTEN")
    elif INT8_TUNED_VALUE_SABOTAGE:
        return String("LOWBIT_VALUE_FLIP")
    else:
        return String("none")


def int8_tuned_plan_name(plan: Int) -> String:
    """No spaces: the timing harness prints it as one field."""
    if plan == INT8_TUNED_PLAN_STAGED:
        return String("staged.w16x16.b32x32.k32.l4")
    if plan == INT8_TUNED_PLAN_FRAG2:
        return String("staged.w32x32.b64x64.k32.l4")
    if plan == INT8_TUNED_PLAN_FRAG4:
        return String("staged.w64x64.b128x128.k32.l4")
    if plan == INT8_TUNED_PLAN_WIDE:
        return String("staged.w64x64.b128x256.k32.l4")
    if plan == INT8_TUNED_PLAN_PACK:
        return String("staged.w64x64.b128x128.k32.l16")
    if plan == INT8_TUNED_PLAN_K64:
        return String("staged.w64x64.b128x128.k64.l16")
    if plan == INT8_TUNED_PLAN_K128:
        return String("staged.w64x64.b128x128.k128.l16")
    if plan == INT8_TUNED_PLAN_WIDE_K64:
        return String("staged.w64x64.b128x256.k64.l16")
    if plan == INT8_TUNED_PLAN_ROW:
        return String("staged.w16x64.b16x256.k64.l16")
    if plan == INT8_TUNED_PLAN_SMALL_K64:
        return String("staged.w16x16.b32x32.k64.l16")
    if plan == INT8_TUNED_PLAN_FRAG2_K64:
        return String("staged.w32x32.b64x64.k64.l16")
    if plan == INT8_TUNED_PLAN_WARPS16:
        return String("staged.w32x32.b128x128.k64.l16")
    if plan == INT8_TUNED_PLAN_WARPS32:
        return String("staged.w16x32.b128x128.k64.l16")
    if plan == INT8_TUNED_PLAN_WARPS32_WIDE:
        return String("staged.w16x32.b64x256.k64.l16")
    if plan == INT8_TUNED_PLAN_TALL256:
        return String("staged.w32x32.b256x64.k64.l16")
    if plan == INT8_TUNED_PLAN_TALL512:
        return String("staged.w32x32.b512x32.k64.l16")
    return String("staged.w32x16.b512x16.k64.l16")


def int8_tuned_plan_available(plan: Int) -> Bool:
    """Whether the column can launch the plan. The 32-warp plan is 1024
    threads a block where a warp is 32 lanes (NVIDIA) and would be 2048
    where it is 64 (AMD CDNA), above a block's limit; it is NOT RUN there,
    by name, and a plan that is not run is not a plan that agreed. The wide
    32-warp plan is refused by the one device that tried it."""
    if plan == INT8_TUNED_PLAN_WARPS32_WIDE:
        return False
    if plan == INT8_TUNED_PLAN_WARPS32:
        return 32 * WARP_SIZE <= INT8_TUNED_MAX_TPB
    return plan >= 0 and plan < INT8_TUNED_PLAN_COUNT


def int8_pieces_plan_name(plan: Int) -> String:
    """No spaces."""
    if plan == INT8_PIECES_PLAN_SMALL:
        return String("staged.w16x16.b32x32.k64.l16")
    if plan == INT8_PIECES_PLAN_WARPS16:
        return String("staged.w16x32.b64x128.k64.l16")
    if plan == INT8_PIECES_PLAN_FRAG2:
        return String("staged.w32x32.b64x128.k64.l16")
    if plan == INT8_PIECES_PLAN_SQUARE:
        return String("staged.w16x16.b64x64.k64.l16")
    if plan == INT8_PIECES_PLAN_TALL128:
        return String("staged.w16x32.b128x64.k32.l16")
    if plan == INT8_PIECES_PLAN_TALL256:
        return String("staged.w16x32.b256x32.k32.l16")
    if plan == INT8_PIECES_PLAN_TALL256_N16:
        return String("staged.w16x16.b256x16.k32.l16")
    if plan == INT8_PIECES_PLAN_WARPS16_K32:
        return String("staged.w16x32.b64x128.k32.l16")
    if plan == INT8_PIECES_PLAN_PIPE_WARPS16:
        return String("pipe2.w16x32.b64x128.k32.l16")
    if plan == INT8_PIECES_PLAN_PIPE_FRAG2:
        return String("pipe2.w32x32.b64x128.k32.l16")
    return String("pipe2.w16x16.b32x32.k64.l16")


def int8_pieces_dispatch(m: Int, n: Int, k: Int) -> Int:
    """The plan `identical_gemm_int8_pieces_tuned_into` takes. Reads the
    shape and may: every plan returns the same three integers. The H100's
    measurement of 2026-09-29 (run 8, job nvc3-0030): TWO PAGES took the
    least time of the eleven plans at all twelve rows, the 32 x 32 warps of
    the 64 x 128 block at the four 512-token rows and the 32 x 32 block at
    the eight decode rows."""
    if m <= INT8_TUNED_ROW_MAX_M:
        return INT8_PIECES_PLAN_PIPE_SMALL
    return INT8_PIECES_PLAN_PIPE_FRAG2


def int8_pieces_sabotage_name() -> String:
    comptime if INT8_PIECES_SABOTAGE:
        return String("MIDDLE_TAKES_HL_TWICE")
    elif INT8_TUNED_SABOTAGE:
        return String("STAGING_PAD_NOT_WRITTEN")
    elif INT8_TUNED_VALUE_SABOTAGE:
        return String("LOWBIT_VALUE_FLIP")
    else:
        return String("none")


def _bases_aligned(a_address: Int, b_address: Int, bytes: Int) -> Bool:
    """Whether the BASES of two operand buffers are both multiples of
    `bytes`, given as the addresses of the pointers the launch is about to
    pass. A load states its alignment only when this is true."""
    comptime if INT8_TUNED_UNSTATED:
        return False
    else:
        return ((a_address | b_address) & (bytes - 1)) == 0


def int8_direct_name(which: Int) -> String:
    """No spaces. A PROBE's product is wrong on purpose, and the caller
    that prints its name says `probe` before it."""
    if which == INT8_DIRECT_REFERENCE_LOADS:
        return String("direct.reference-loads")
    if which == INT8_DIRECT_ALIGNED_LOADS:
        return String("direct.aligned-loads")
    if which == INT8_DIRECT_SCALAR_ACC:
        return String("direct.aligned-loads.scalar-acc")
    if which == INT8_DIRECT_PROBE_HOISTED:
        return String("loads-hoisted")
    if which == INT8_DIRECT_PROBE_ONE_HALF:
        return String("one-half")
    return String("raw-store")


def int8_direct_is_probe(which: Int) -> Bool:
    """Whether the instantiation computes a wrong product on purpose."""
    return which >= INT8_DIRECT_PROBE_HOISTED


def int8_tuned_dispatch(m: Int, n: Int, k: Int) -> Int:
    """The plan `identical_gemm_int8_mma_tuned_into` takes. Reads the shape
    and may: every plan is the profile. The choice is the H100's
    measurement of 2026-09-29 (job nvc3-0020,
    `docs/lanes/progress/lowbit-mma-speed.md`): at the four 512-token rows
    the 128 x 128 block of sixteen 32 x 32 warps took the least time of the
    thirteen plans, and at the eight decode rows the 32 x 32 block of four
    16 x 16 warps did (the ROW plan, written for them, took two to four
    times as long). Between 17 and 511 rows nothing is measured."""
    if m <= INT8_TUNED_ROW_MAX_M:
        return INT8_TUNED_PLAN_SMALL_K64
    return INT8_TUNED_PLAN_WARPS16


# ===========================================================================
# the epilogue, shared by every kernel of this file
# ===========================================================================


@always_inline
def _store_cell_tuned(
    c: MutPointer[Float32, MutAnyOrigin],
    ea: MutPointer[Int32, MutAnyOrigin],
    eb: MutPointer[Int32, MutAnyOrigin],
    acc: Int32,
    i: Int,
    j: Int,
    m: Int,
    n: Int,
):
    """`gemm_int8_mma.mojo::_store_cell`: the dequantization seam (L-5,
    L-6) and the store, masked to the output."""
    if i >= m or j >= n:
        return
    var out = dequant_int8_pinned(acc, Int(ea.unsafe_load(i)) + Int(eb.unsafe_load(j)))
    comptime if INT8_TUNED_VALUE_SABOTAGE:
        out = gemm_oracle_sabotage_value_flip(out)
    c.unsafe_store(i * n + j, out)


# ===========================================================================
# THE DIRECT KERNEL: the reference's schedule, one thing changed at a time
# ===========================================================================


@always_inline
def _pack4_aligned(
    p: MutPointer[Int8, MutAnyOrigin],
    row: Int,
    k0: Int,
    rows: Int,
    k: Int,
    aligned: Bool,
) -> Int32:
    """`_pack4` with the load's alignment STATED. The reference takes its
    vector load only where the address is a multiple of four and does not
    say so to the compiler, which may then assemble the word from four
    byte loads. Same bytes, same zero codes. `aligned`: the launch found
    the buffer's base a multiple of four bytes."""
    if row >= rows:
        return Int32(0)
    var base = row * k
    if aligned and k0 + 4 <= k and (k & 3) == 0:
        return bitcast[DType.int32, 1](p.unsafe_load[width=4, alignment=4](base + k0))[0]
    var v = SIMD[DType.int8, 4](0)
    comptime for i in range(4):
        if k0 + i < k:
            v[i] = p.unsafe_load(base + k0 + i)
    return bitcast[DType.int32, 1](v)[0]


@always_inline
def _frag_word[ALIGNED: Bool](
    p: MutPointer[Int8, MutAnyOrigin],
    row: Int,
    k0: Int,
    rows: Int,
    k: Int,
    aligned: Bool,
) -> Int32:
    comptime if ALIGNED:
        return _pack4_aligned(p, row, k0, rows, k, aligned)
    else:
        return _pack4(p, row, k0, rows, k)


@always_inline
def _direct_warp_tile[WHICH: Int](
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
    """`gemm_int8_mma.mojo::_nvidia_warp_tile`, its fragment layout and its
    order, with the one change `WHICH` names."""
    comptime ALIGNED = WHICH != INT8_DIRECT_REFERENCE_LOADS
    var g = lane >> 2
    var t = lane & 3
    var ra0 = row0 + g
    var ra1 = row0 + g + 8
    var cb0 = col0 + g
    var cb1 = col0 + 8 + g
    var acc0 = SIMD[DType.int32, 4](0)
    var acc1 = SIMD[DType.int32, 4](0)
    # SCALAR_ACC: the eight accumulators as eight scalars, so nothing
    # re-spells the intrinsic's register pack as a vector.
    var s0 = Int32(0)
    var s1 = Int32(0)
    var s2 = Int32(0)
    var s3 = Int32(0)
    var s4 = Int32(0)
    var s5 = Int32(0)
    var s6 = Int32(0)
    var s7 = Int32(0)
    # PROBE_HOISTED: the first k step's fragments, read once.
    var h0 = _frag_word[ALIGNED](qa, ra0, t * 4, m, k, aligned)
    var h1 = _frag_word[ALIGNED](qa, ra1, t * 4, m, k, aligned)
    var h2 = _frag_word[ALIGNED](qa, ra0, t * 4 + 16, m, k, aligned)
    var h3 = _frag_word[ALIGNED](qa, ra1, t * 4 + 16, m, k, aligned)
    var h4 = _frag_word[ALIGNED](qb, cb0, t * 4, n, k, aligned)
    var h5 = _frag_word[ALIGNED](qb, cb0, t * 4 + 16, n, k, aligned)
    var h6 = _frag_word[ALIGNED](qb, cb1, t * 4, n, k, aligned)
    var h7 = _frag_word[ALIGNED](qb, cb1, t * 4 + 16, n, k, aligned)
    for kt in range(0, k, INT8_TUNED_K_TILE):
        var ka = kt + t * 4
        var a0 = h0
        var a1 = h1
        var a2 = h2
        var a3 = h3
        var b0 = h4
        var b1 = h5
        var b2 = h6
        var b3 = h7
        comptime if WHICH != INT8_DIRECT_PROBE_HOISTED:
            a0 = _frag_word[ALIGNED](qa, ra0, ka, m, k, aligned)
            a1 = _frag_word[ALIGNED](qa, ra1, ka, m, k, aligned)
            a2 = _frag_word[ALIGNED](qa, ra0, ka + 16, m, k, aligned)
            a3 = _frag_word[ALIGNED](qa, ra1, ka + 16, m, k, aligned)
            b0 = _frag_word[ALIGNED](qb, cb0, ka, n, k, aligned)
            b1 = _frag_word[ALIGNED](qb, cb0, ka + 16, n, k, aligned)
        comptime if WHICH == INT8_DIRECT_SCALAR_ACC:
            b2 = _frag_word[ALIGNED](qb, cb1, ka, n, k, aligned)
            b3 = _frag_word[ALIGNED](qb, cb1, ka + 16, n, k, aligned)
            var r0 = llvm_intrinsic[
                "llvm.nvvm.mma.m16n8k32.row.col.s8",
                _RegisterPackType[Int32, Int32, Int32, Int32],
                has_side_effect=False,
            ](a0, a1, a2, a3, b0, b1, s0, s1, s2, s3)
            s0 = r0[0]
            s1 = r0[1]
            s2 = r0[2]
            s3 = r0[3]
            var r1 = llvm_intrinsic[
                "llvm.nvvm.mma.m16n8k32.row.col.s8",
                _RegisterPackType[Int32, Int32, Int32, Int32],
                has_side_effect=False,
            ](a0, a1, a2, a3, b2, b3, s4, s5, s6, s7)
            s4 = r1[0]
            s5 = r1[1]
            s6 = r1[2]
            s7 = r1[3]
        elif WHICH == INT8_DIRECT_PROBE_ONE_HALF:
            acc0 = _imma_m16n8k32(a0, a1, a2, a3, b0, b1, acc0)
        else:
            acc0 = _imma_m16n8k32(a0, a1, a2, a3, b0, b1, acc0)
            comptime if WHICH != INT8_DIRECT_PROBE_HOISTED:
                b2 = _frag_word[ALIGNED](qb, cb1, ka, n, k, aligned)
                b3 = _frag_word[ALIGNED](qb, cb1, ka + 16, n, k, aligned)
            acc1 = _imma_m16n8k32(a0, a1, a2, a3, b2, b3, acc1)
    comptime if WHICH == INT8_DIRECT_SCALAR_ACC:
        acc0 = SIMD[DType.int32, 4](s0, s1, s2, s3)
        acc1 = SIMD[DType.int32, 4](s4, s5, s6, s7)
    comptime if WHICH == INT8_DIRECT_PROBE_ONE_HALF:
        # The half that did not run stores the other's cells: written, and
        # wrong on purpose.
        acc1 = acc0
    var jc = col0 + t * 2
    comptime for h in range(2):
        comptime for e in range(4):
            var acc = acc0[e]
            comptime if h == 1:
                acc = acc1[e]
            var gi = ra0 if e < 2 else ra1
            var gj = jc + h * 8 + (e & 1)
            comptime if WHICH == INT8_DIRECT_PROBE_RAW_STORE:
                # PROBE: no seam and no exponent loads; the backend's own
                # conversion of the Int32, and a half added, so that no
                # cell stored is an integer: the timing harness poisons
                # its output with one (-987654.0), and a sum of codes can
                # be that integer (nvc3-0016: a cell of mlp_down.t512 was).
                if gi < m and gj < n:
                    c.unsafe_store(gi * n + gj, Float32(acc) + Float32(0.5))
            else:
                _store_cell_tuned(c, ea, eb, acc, gi, gj, m, n)


def identical_gemm_int8_mma_direct_kernel[WHICH: Int](
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
    """`identical_gemm_int8_mma_kernel`'s grid, block and tile ownership
    (2 x 2 warps, a 32 x 32 tile per block), on NVIDIA. On any other target
    the body is dead and the launcher refuses."""
    var m = Int(m_in)
    var n = Int(n_in)
    var k = Int(k_in)
    var aligned = aligned_in != Int32(0)
    var warp = Int(thread_idx.x) // WARP_SIZE
    var lane = Int(lane_id())
    var wm = warp // 2
    var wn = warp - wm * 2
    var row0 = Int(block_idx.y) * 32 + wm * INT8_TUNED_TILE
    var col0 = Int(block_idx.x) * 32 + wn * INT8_TUNED_TILE
    # Uniform across the warp, as in the reference.
    if row0 >= m or col0 >= n:
        return
    comptime if is_nvidia_gpu():
        _direct_warp_tile[WHICH](c, qa, ea, qb, eb, lane, row0, col0, m, n, k, aligned)
    else:
        return


def _refuse_tuned_shape(m: Int, n: Int, k: Int) raises:
    if not int8_mma_admits(m, n, k):
        raise Error(
            "identical_gemm_int8_mma_tuned: m, n and k must be positive and k"
            " at most " + String(INT8_MAX_K) + " (contract L-7), got m="
            + String(m) + " n=" + String(n) + " k=" + String(k)
        )


def _launch_direct[WHICH: Int](
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
    comptime kern = identical_gemm_int8_mma_direct_kernel[WHICH]
    var aligned = Int32(0)
    if _bases_aligned(Int(qa.unsafe_ptr()), Int(qb.unsafe_ptr()), 4):
        aligned = Int32(1)
    ctx.enqueue_function[kern](
        c.unsafe_ptr(),
        qa.unsafe_ptr(),
        ea.unsafe_ptr(),
        qb.unsafe_ptr(),
        eb.unsafe_ptr(),
        Int32(m),
        Int32(n),
        Int32(k),
        aligned,
        grid_dim=((n + 31) // 32, (m + 31) // 32, 1),
        block_dim=(4 * WARP_SIZE, 1, 1),
    )


def identical_gemm_int8_mma_direct_into(
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut qa: DeviceBuffer[DType.int8],
    mut ea: DeviceBuffer[DType.int32],
    mut qb: DeviceBuffer[DType.int8],
    mut eb: DeviceBuffer[DType.int32],
    m: Int,
    n: Int,
    k: Int,
    which: Int,
) raises:
    """The direct kernel on a NAMED instantiation, for the gate (the three
    that are plans) and the timing harness (all six). Refuses by name on a
    column that is not NVIDIA. Asynchronous."""
    comptime if not INT8_DIRECT_AVAILABLE:
        raise Error(
            "identical_gemm_int8_mma_direct: column " + column_name(TARGET_COLUMN)
            + " is not NVIDIA; the direct kernel tests NVIDIA's fragment layout"
        )
    else:
        _refuse_tuned_shape(m, n, k)
        if which == INT8_DIRECT_REFERENCE_LOADS:
            _launch_direct[INT8_DIRECT_REFERENCE_LOADS](ctx, c, qa, ea, qb, eb, m, n, k)
        elif which == INT8_DIRECT_ALIGNED_LOADS:
            _launch_direct[INT8_DIRECT_ALIGNED_LOADS](ctx, c, qa, ea, qb, eb, m, n, k)
        elif which == INT8_DIRECT_SCALAR_ACC:
            _launch_direct[INT8_DIRECT_SCALAR_ACC](ctx, c, qa, ea, qb, eb, m, n, k)
        elif which == INT8_DIRECT_PROBE_HOISTED:
            _launch_direct[INT8_DIRECT_PROBE_HOISTED](ctx, c, qa, ea, qb, eb, m, n, k)
        elif which == INT8_DIRECT_PROBE_ONE_HALF:
            _launch_direct[INT8_DIRECT_PROBE_ONE_HALF](ctx, c, qa, ea, qb, eb, m, n, k)
        elif which == INT8_DIRECT_PROBE_RAW_STORE:
            _launch_direct[INT8_DIRECT_PROBE_RAW_STORE](ctx, c, qa, ea, qb, eb, m, n, k)
        else:
            raise Error(
                "identical_gemm_int8_mma_direct: no instantiation " + String(which)
            )


# ===========================================================================
# THE STAGED KERNEL
# ===========================================================================


@always_inline
def _mfma_i8(a: Int64, b: Int64, acc: SIMD[DType.int32, 4]) -> SIMD[DType.int32, 4]:
    """One `v_mfma_i32_16x16x32_i8`, the instruction the reference issues,
    with `cbsz`, `abid` and `blgp` 0."""
    return llvm_intrinsic[
        "llvm.amdgcn.mfma.i32.16x16x32.i8",
        SIMD[DType.int32, 4],
        has_side_effect=False,
    ](a, b, acc, Int32(0), Int32(0), Int32(0))


@always_inline
def _stage_window[
    ROWS: Int, KB: Int, NT: Int, LW: Int, SW: Int
](
    dst: UnsafePointer[Int32, MutUntrackedOrigin, address_space=AddressSpace.SHARED],
    dst0: Int,
    q: MutPointer[Int8, MutAnyOrigin],
    row0: Int,
    rows: Int,
    k0: Int,
    k: Int,
    tid: Int,
    aligned: Bool,
):
    """One window of one operand, device memory to threadgroup memory,
    from word `dst0` of `dst` on.

    A LOAD is `LW` consecutive codes of one row; the window has `ROWS * KB
    / LW` of them and thread `tid` owns loads `tid, tid + NT, ...`, so
    consecutive threads read consecutive bytes of a row. Code `(row r, step
    p)` lands at byte `r * 4 SW + p` of `dst`. A row at or beyond `rows` and
    a step at or beyond `k` are the ZERO CODE, and the zero codes are
    STORED: the page still holds the window before. The vector load is
    taken only when it is aligned (the buffer's base is, which `aligned`
    says, and `k` a multiple of `LW` makes every row start one) and
    entirely inside the row; its alignment is stated."""
    comptime LPR = KB // LW
    comptime LOADS = ROWS * LPR
    comptime SL = (LOADS + NT - 1) // NT
    comptime for sl in range(SL):
        var s = sl * NT + tid
        if s < LOADS:
            var r = s // LPR
            var kq = (s % LPR) * LW
            var v = SIMD[DType.int8, LW](0)
            var gr = row0 + r
            var inside = gr < rows and k0 + kq < k
            if inside:
                var base = gr * k + k0 + kq
                if aligned and k0 + kq + LW <= k and (k & (LW - 1)) == 0:
                    v = q.unsafe_load[width=LW, alignment=LW](base)
                else:
                    comptime for i in range(LW):
                        if k0 + kq + i < k:
                            v[i] = q.unsafe_load(base + i)
            comptime if INT8_TUNED_SABOTAGE:
                # SABOTAGE: the padding rule broken. What lies wholly
                # outside the operand is not written.
                if inside:
                    dst.unsafe_store[alignment=LW](
                        dst0 + r * SW + kq // 4, bitcast[DType.int32, LW // 4](v)
                    )
            else:
                dst.unsafe_store[alignment=LW](
                    dst0 + r * SW + kq // 4, bitcast[DType.int32, LW // 4](v)
                )


def identical_gemm_int8_mma_tuned_kernel[
    FM: Int, FN: Int, WM: Int, WN: Int, KB: Int, LW: Int
](
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
    """OP_NT, `C[m x n] = Qa[m x k] . Qb[n x k]^T`. One block owns a `BM x
    BN` output tile and all of its `k`; per window of `KB` steps the block
    stages `BM` rows of `Qa` and `BN` rows of `Qb`, then every warp runs
    `KB / 32` unit steps over its `16 FM x 16 FN` tile out of threadgroup
    memory. Int32 accumulation, then the reference's epilogue. Grid
    `(ceil(n / BN), ceil(m / BM), 1)`, block `WM * WN * WARP_SIZE`.

    Every thread of the block reaches every `barrier()`: the window count is
    a function of `k` and the comptime `KB`; a warp whose tile lies wholly
    outside the output still stages and still waits, and only its unit
    steps and its stores are skipped (`live`, uniform across the warp, so
    no lane leaves a unit step another lane enters).

    On a target with no integer matrix unit the unit steps are dead code
    and the launcher refuses (`lib_int8_matrix_unit_for`)."""
    comptime TM = INT8_TUNED_TILE * FM
    comptime TN = INT8_TUNED_TILE * FN
    comptime BM = WM * TM
    comptime BN = WN * TN
    comptime NT = WM * WN * WARP_SIZE
    comptime SS = KB + INT8_TUNED_ROW_PAD
    comptime SW = SS // 4
    comptime KSTEPS = KB // INT8_TUNED_K_TILE
    #: Accumulator registers of four Int32: two per unit tile on NVIDIA
    #: (the two m16n8 halves), one on AMD.
    comptime HALVES = 2 if is_nvidia_gpu() else 1
    comptime NACC = FM * FN * HALVES

    comptime assert KB % INT8_TUNED_K_TILE == 0 and KB >= INT8_TUNED_K_TILE, (
        "identical_gemm_int8_mma_tuned_kernel: a window is whole unit steps"
    )
    comptime assert LW == 4 or LW == 8 or LW == 16, (
        "identical_gemm_int8_mma_tuned_kernel: a staging load is 4, 8 or 16 bytes"
    )
    comptime assert NT <= INT8_TUNED_MAX_TPB, (
        "identical_gemm_int8_mma_tuned_kernel: a block is at most 1024 threads"
    )
    comptime assert (BM + BN) * SS <= column_shared_limit(TARGET_COLUMN), (
        "identical_gemm_int8_mma_tuned_kernel: the staged window does not fit"
        " the column's threadgroup memory"
    )

    var m = Int(m_in)
    var n = Int(n_in)
    var k = Int(k_in)
    var aligned = aligned_in != Int32(0)
    var tid = Int(thread_idx.x)
    var warp = tid // WARP_SIZE
    var lane = Int(lane_id())
    var wm = warp // WN
    var wn = warp - wm * WN
    var i0 = Int(block_idx.y) * BM
    var j0 = Int(block_idx.x) * BN
    # This warp's tile, inside the block and in the output.
    var lrow = wm * TM
    var lcol = wn * TN
    var live = i0 + lrow < m and j0 + lcol < n

    var as_ = stack_allocation[
        BM * SW,
        Scalar[DType.int32],
        alignment=16,
        address_space = AddressSpace.SHARED,
    ]()
    var bs_ = stack_allocation[
        BN * SW,
        Scalar[DType.int32],
        alignment=16,
        address_space = AddressSpace.SHARED,
    ]()

    var acc = InlineArray[SIMD[DType.int32, 4], NACC](fill=SIMD[DType.int32, 4](0))

    var windows = (k + KB - 1) // KB
    for w in range(windows):
        var k0 = w * KB
        _stage_window[BM, KB, NT, LW, SW](as_, 0, qa, i0, m, k0, k, tid, aligned)
        _stage_window[BN, KB, NT, LW, SW](bs_, 0, qb, j0, n, k0, k, tid, aligned)
        barrier()
        if live:
            comptime for ks in range(KSTEPS):
                # A unit step wholly beyond `k` adds zero codes only.
                # Uniform across the block.
                if k0 + ks * INT8_TUNED_K_TILE < k:
                    comptime if is_nvidia_gpu():
                        # PTX ISA, Matrix Fragments for mma.m16n8k32 with
                        # .s8 operands (the reference's note): group `g`,
                        # thread in group `t`. A word index is a byte index
                        # over four.
                        var g = lane >> 2
                        var t = lane & 3
                        var kw = ks * 8 + t
                        var bfr = InlineArray[Int32, 4 * FN](fill=Int32(0))
                        comptime for fq in range(FN):
                            comptime for h in range(2):
                                var br = (lcol + fq * 16 + h * 8 + g) * SW + kw
                                bfr[(fq * 2 + h) * 2] = bs_.unsafe_load(br)
                                bfr[(fq * 2 + h) * 2 + 1] = bs_.unsafe_load(br + 4)
                        comptime for fm in range(FM):
                            var ar = (lrow + fm * 16 + g) * SW + kw
                            var a0 = as_.unsafe_load(ar)
                            var a1 = as_.unsafe_load(ar + 8 * SW)
                            var a2 = as_.unsafe_load(ar + 4)
                            var a3 = as_.unsafe_load(ar + 8 * SW + 4)
                            comptime for fq in range(FN):
                                comptime for h in range(2):
                                    acc[(fm * FN + fq) * 2 + h] = _imma_m16n8k32(
                                        a0,
                                        a1,
                                        a2,
                                        a3,
                                        bfr[(fq * 2 + h) * 2],
                                        bfr[(fq * 2 + h) * 2 + 1],
                                        acc[(fm * FN + fq) * 2 + h],
                                    )
                    elif is_amd_gpu():
                        # CDNA3 `v_mfma_i32_16x16x32_i8` (the reference's
                        # note): lane `l` holds row or column `l & 15`,
                        # steps `8 (l >> 4) .. + 7` of the 32.
                        var i16 = lane & 15
                        var kw = ks * 8 + (lane >> 4) * 2
                        var bfr = InlineArray[Int64, FN](fill=Int64(0))
                        comptime for fq in range(FN):
                            bfr[fq] = bitcast[DType.int64, 1](
                                bs_.unsafe_load[width=2, alignment=8](
                                    (lcol + fq * 16 + i16) * SW + kw
                                )
                            )[0]
                        comptime for fm in range(FM):
                            var a = bitcast[DType.int64, 1](
                                as_.unsafe_load[width=2, alignment=8](
                                    (lrow + fm * 16 + i16) * SW + kw
                                )
                            )[0]
                            comptime for fq in range(FN):
                                acc[fm * FN + fq] = llvm_intrinsic[
                                    "llvm.amdgcn.mfma.i32.16x16x32.i8",
                                    SIMD[DType.int32, 4],
                                    has_side_effect=False,
                                ](a, bfr[fq], acc[fm * FN + fq], Int32(0), Int32(0), Int32(0))
        # The page is free once every thread has left the window's steps.
        barrier()

    if not live:
        return

    # ---- THE EPILOGUE. The accumulators leave their registers once, then
    # one loop stores the cells, so the seam is spelled once per kernel and
    # not once per cell.
    var outs = stack_allocation[NACC * 4, Scalar[DType.int32]]()
    comptime for a in range(NACC):
        comptime for e in range(4):
            outs.unsafe_store(a * 4 + e, acc[a][e])
    comptime if is_nvidia_gpu():
        # C/D (16 x 8, s32): c0, c1 row `g`, columns `2 t`, `2 t + 1`;
        # c2, c3 row `g + 8`.
        var g = lane >> 2
        var t = lane & 3
        for a in range(NACC):
            var tile = a >> 1
            var h = a & 1
            var fm = tile // FN
            var fq = tile - fm * FN
            var gi = i0 + lrow + fm * 16 + g
            var gj = j0 + lcol + fq * 16 + h * 8 + t * 2
            _store_cell_tuned(c, ea, eb, outs.unsafe_load(a * 4), gi, gj, m, n)
            _store_cell_tuned(c, ea, eb, outs.unsafe_load(a * 4 + 1), gi, gj + 1, m, n)
            _store_cell_tuned(c, ea, eb, outs.unsafe_load(a * 4 + 2), gi + 8, gj, m, n)
            _store_cell_tuned(c, ea, eb, outs.unsafe_load(a * 4 + 3), gi + 8, gj + 1, m, n)
    elif is_amd_gpu():
        # D[i][j]: lane `j + 16 (i // 4)`, register `i % 4`.
        var i16 = lane & 15
        var i4 = (lane >> 4) * 4
        for a in range(NACC):
            var fm = a // FN
            var fq = a - fm * FN
            var gi = i0 + lrow + fm * 16 + i4
            var gj = j0 + lcol + fq * 16 + i16
            comptime for e in range(4):
                _store_cell_tuned(c, ea, eb, outs.unsafe_load(a * 4 + e), gi + e, gj, m, n)


def _launch_tuned[
    FM: Int, FN: Int, WM: Int, WN: Int, KB: Int, LW: Int
](
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
    comptime BM = WM * FM * INT8_TUNED_TILE
    comptime BN = WN * FN * INT8_TUNED_TILE
    comptime kern = identical_gemm_int8_mma_tuned_kernel[FM, FN, WM, WN, KB, LW]
    var aligned = Int32(0)
    if _bases_aligned(Int(qa.unsafe_ptr()), Int(qb.unsafe_ptr()), LW):
        aligned = Int32(1)
    ctx.enqueue_function[kern](
        c.unsafe_ptr(),
        qa.unsafe_ptr(),
        ea.unsafe_ptr(),
        qb.unsafe_ptr(),
        eb.unsafe_ptr(),
        Int32(m),
        Int32(n),
        Int32(k),
        aligned,
        grid_dim=((n + BN - 1) // BN, (m + BM - 1) // BM, 1),
        block_dim=(WM * WN * WARP_SIZE, 1, 1),
    )


def identical_gemm_int8_mma_tuned_with_plan(
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut qa: DeviceBuffer[DType.int8],
    mut ea: DeviceBuffer[DType.int32],
    mut qb: DeviceBuffer[DType.int8],
    mut eb: DeviceBuffer[DType.int32],
    m: Int,
    n: Int,
    k: Int,
    plan: Int,
) raises:
    """The staged kernel on a NAMED plan. The gate and the timing harness
    call this; a caller that wants the product calls
    `identical_gemm_int8_mma_tuned_into`. Refuses by name on a column whose
    kernel-matrix row says it has no integer matrix unit. Asynchronous."""
    comptime if not INT8_TUNED_AVAILABLE:
        raise Error(
            "identical_gemm_int8_mma_tuned: column " + column_name(TARGET_COLUMN)
            + " has no int8 matrix unit (kernel_matrix row"
            " lib_int8_matrix_unit_for); the flat kernel serves it"
        )
    else:
        _refuse_tuned_shape(m, n, k)
        if plan == INT8_TUNED_PLAN_STAGED:
            _launch_tuned[1, 1, 2, 2, 32, 4](ctx, c, qa, ea, qb, eb, m, n, k)
        elif plan == INT8_TUNED_PLAN_FRAG2:
            _launch_tuned[2, 2, 2, 2, 32, 4](ctx, c, qa, ea, qb, eb, m, n, k)
        elif plan == INT8_TUNED_PLAN_FRAG4:
            _launch_tuned[4, 4, 2, 2, 32, 4](ctx, c, qa, ea, qb, eb, m, n, k)
        elif plan == INT8_TUNED_PLAN_WIDE:
            _launch_tuned[4, 4, 2, 4, 32, 4](ctx, c, qa, ea, qb, eb, m, n, k)
        elif plan == INT8_TUNED_PLAN_PACK:
            _launch_tuned[4, 4, 2, 2, 32, 16](ctx, c, qa, ea, qb, eb, m, n, k)
        elif plan == INT8_TUNED_PLAN_K64:
            _launch_tuned[4, 4, 2, 2, 64, 16](ctx, c, qa, ea, qb, eb, m, n, k)
        elif plan == INT8_TUNED_PLAN_K128:
            _launch_tuned[4, 4, 2, 2, 128, 16](ctx, c, qa, ea, qb, eb, m, n, k)
        elif plan == INT8_TUNED_PLAN_WIDE_K64:
            _launch_tuned[4, 4, 2, 4, 64, 16](ctx, c, qa, ea, qb, eb, m, n, k)
        elif plan == INT8_TUNED_PLAN_ROW:
            _launch_tuned[1, 4, 1, 4, 64, 16](ctx, c, qa, ea, qb, eb, m, n, k)
        elif plan == INT8_TUNED_PLAN_SMALL_K64:
            _launch_tuned[1, 1, 2, 2, 64, 16](ctx, c, qa, ea, qb, eb, m, n, k)
        elif plan == INT8_TUNED_PLAN_FRAG2_K64:
            _launch_tuned[2, 2, 2, 2, 64, 16](ctx, c, qa, ea, qb, eb, m, n, k)
        elif plan == INT8_TUNED_PLAN_WARPS16:
            _launch_tuned[2, 2, 4, 4, 64, 16](ctx, c, qa, ea, qb, eb, m, n, k)
        elif plan == INT8_TUNED_PLAN_TALL256:
            _launch_tuned[2, 2, 8, 2, 64, 16](ctx, c, qa, ea, qb, eb, m, n, k)
        elif plan == INT8_TUNED_PLAN_TALL512:
            _launch_tuned[2, 2, 16, 1, 64, 16](ctx, c, qa, ea, qb, eb, m, n, k)
        elif plan == INT8_TUNED_PLAN_TALL512_N16:
            _launch_tuned[2, 1, 16, 1, 64, 16](ctx, c, qa, ea, qb, eb, m, n, k)
        elif plan == INT8_TUNED_PLAN_WARPS32 or plan == INT8_TUNED_PLAN_WARPS32_WIDE:
            comptime if 32 * WARP_SIZE <= INT8_TUNED_MAX_TPB:
                if plan == INT8_TUNED_PLAN_WARPS32:
                    _launch_tuned[1, 2, 8, 4, 64, 16](ctx, c, qa, ea, qb, eb, m, n, k)
                else:
                    _launch_tuned[1, 2, 4, 8, 64, 16](ctx, c, qa, ea, qb, eb, m, n, k)
            else:
                raise Error(
                    "identical_gemm_int8_mma_tuned: plan " + int8_tuned_plan_name(plan)
                    + " is 32 warps a block, above a block's 1024 threads on column "
                    + column_name(TARGET_COLUMN)
                )
        else:
            raise Error("identical_gemm_int8_mma_tuned: no plan " + String(plan))


def identical_gemm_int8_mma_tuned_into(
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
    """The tuned unit plan on the plan `int8_tuned_dispatch` names.
    `identical_gemm_int8_mma_into`'s signature and its bits. Asynchronous."""
    identical_gemm_int8_mma_tuned_with_plan(
        ctx, c, qa, ea, qb, eb, m, n, k, int8_tuned_dispatch(m, n, k)
    )


# ===========================================================================
# THE DECODE KERNEL: rows 1 to 16, the weights streamed, k split over warps
# ===========================================================================
#
# At the decode rows the staged plans spend their time waiting: 32 x 32
# blocks of four warps, one window at a time and two barriers per window,
# over weights that are read once. Here a block owns SIXTEEN output columns
# and all of `m <= 16` rows, its NW warps take interleaved 64-byte chunks of
# `k`, each lane loads sixteen bytes of each of its two weight rows per
# chunk straight from device memory (no staging, no barrier in the k loop),
# and the warps' Int32 sums are added in threadgroup memory at the end.
#
# THE K ORDER. Lane `t` of a group holds the chunk's words `4t .. 4t + 3`
# of each row it reads and feeds words `4t + 2s` and `4t + 2s + 1` to the
# unit as the words the unit calls `t` and `t + 4` of step `s`. The SAME map
# is applied to the left rows and the weight rows, so every product
# `A[i][p] B[j][p]` of the chunk is formed exactly once, and the sums are
# exact Int32: the order in which the unit and the warps add them cannot
# change a bit. The contract's `INT8_MAX_K` bounds every partial sum as it
# bounds the whole.
#
# `QA` True: THE QUANTIZER IN THE LAUNCH. The left operand arrives as float32
# rows; each block reduces their absmax (warp `w` takes rows `w, w + NW,
# ...`, a warp-wide maximum, which is exact under any grouping, contract
# L-3), takes the exponent with `int8_row_exponent`, and quantizes each
# chunk as it loads it with `quantize_rows_int8_par`'s own `_code` (L-4).
# Block 0 stores the exponents for the caller. The codes are the parallel
# quantizer's, value for value; they never leave the registers.

comptime INT8_DECODE_MAX_M = 16
#: Bytes of `k` one warp takes per chunk: four lanes of sixteen bytes.
comptime INT8_DECODE_CHUNK = 64
comptime INT8_DECODE_AVAILABLE = INT8_TUNED_AVAILABLE and TARGET_COLUMN == COLUMN_NVIDIA
comptime INT8_DECODE_PLAN_W4 = 0  #: 4 warps a block
comptime INT8_DECODE_PLAN_W8 = 1  #: 8 warps a block
comptime INT8_DECODE_PLAN_W16 = 2  #: 16 warps a block
comptime INT8_DECODE_PLAN_COUNT = 3


def int8_decode_plan_name(plan: Int) -> String:
    """No spaces."""
    if plan == INT8_DECODE_PLAN_W4:
        return String("decode.n16.w4.c64")
    if plan == INT8_DECODE_PLAN_W8:
        return String("decode.n16.w8.c64")
    return String("decode.n16.w16.c64")


def int8_decode_dispatch(m: Int, n: Int, k: Int) -> Int:
    """The plan `identical_gemm_int8_mma_decode_into` takes. Every plan is
    the profile; not yet measured, so the middle one."""
    return INT8_DECODE_PLAN_W8


@always_inline
def _load16_codes(
    p: MutPointer[Int8, MutAnyOrigin], row: Int, rows: Int, kb: Int, k: Int, aligned: Bool
) -> SIMD[DType.int32, 4]:
    """Sixteen codes of row `row` from byte `kb`, as four words; a row at or
    beyond `rows` and a byte at or beyond `k` are the ZERO CODE. The vector
    load is taken only where it is whole and aligned (`aligned`: the base
    is on sixteen bytes; `k` a multiple of sixteen puts every row there),
    and it states its alignment."""
    if row >= rows:
        return SIMD[DType.int32, 4](0)
    var base = row * k
    if aligned and kb + 16 <= k and (k & 15) == 0:
        return bitcast[DType.int32, 4](p.unsafe_load[width=16, alignment=16](base + kb))
    var v = SIMD[DType.int8, 16](0)
    comptime for i in range(16):
        comptime if INT8_TUNED_SABOTAGE:
            # SABOTAGE: the padding rule broken. A byte beyond `k` is read
            # from the buffer (the next row's codes) instead of the zero code.
            if base + kb + i < rows * k:
                v[i] = p.unsafe_load(base + kb + i)
        else:
            if kb + i < k:
                v[i] = p.unsafe_load(base + kb + i)
    return bitcast[DType.int32, 4](v)


@always_inline
def _quant16_codes(
    x: MutPointer[Float32, MutAnyOrigin], row: Int, rows: Int, kb: Int, k: Int, ex: Int, aligned: Bool
) -> SIMD[DType.int32, 4]:
    """`_load16_codes` of the codes the parallel quantizer makes of row
    `row` of `x` with exponent `ex`: `_code(x, ex)` per value, the zero code
    beyond the row or beyond `k`."""
    if row >= rows:
        return SIMD[DType.int32, 4](0)
    var base = row * k
    var v = SIMD[DType.int8, 16](0)
    if aligned and kb + 16 <= k and (k & 3) == 0:
        comptime for q in range(4):
            var f = x.unsafe_load[width=4, alignment=16](base + kb + 4 * q)
            comptime for i in range(4):
                v[4 * q + i] = _code(f[i], ex)
    else:
        comptime for i in range(16):
            if kb + i < k:
                v[i] = _code(x.unsafe_load(base + kb + i), ex)
    return bitcast[DType.int32, 4](v)


@always_inline
def _store_cell_decode(
    c: MutPointer[Float32, MutAnyOrigin],
    ea_i: Int,
    eb: MutPointer[Int32, MutAnyOrigin],
    acc: Int32,
    i: Int,
    j: Int,
    m: Int,
    n: Int,
):
    """`_store_cell_tuned` with the row's exponent passed as a value."""
    if i >= m or j >= n:
        return
    var out = dequant_int8_pinned(acc, ea_i + Int(eb.unsafe_load(j)))
    comptime if INT8_TUNED_VALUE_SABOTAGE:
        out = gemm_oracle_sabotage_value_flip(out)
    c.unsafe_store(i * n + j, out)


def identical_gemm_int8_mma_decode_kernel[QA: Bool, NW: Int](
    c: MutPointer[Float32, MutAnyOrigin],
    qa: MutPointer[Int8, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    ea: MutPointer[Int32, MutAnyOrigin],
    qb: MutPointer[Int8, MutAnyOrigin],
    eb: MutPointer[Int32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    k_in: Int32,
    aligned_in: Int32,
):
    """OP_NT, `m <= 16`. `QA` False: `C = Qa . Qb^T` from the codes `qa` and
    the exponents `ea`; `x` is not read. `QA` True: the left codes and
    exponents are made here from the float32 rows `x` (the parallel
    quantizer's, value for value), block 0 stores the exponents to `ea`, and
    `qa` is not read. Grid `(ceil(n / 16), 1, 1)`, block `NW * WARP_SIZE`.

    Every thread of the block reaches both `barrier()`s: nothing before
    them returns."""
    comptime NT = NW * WARP_SIZE
    comptime assert NW >= 1 and NW * WARP_SIZE <= INT8_TUNED_MAX_TPB, (
        "identical_gemm_int8_mma_decode_kernel: a block is at most 1024 threads"
    )
    var m = Int(m_in)
    var n = Int(n_in)
    var k = Int(k_in)
    var aligned = aligned_in != Int32(0)
    var tid = Int(thread_idx.x)
    var w = tid // WARP_SIZE
    var lane = Int(lane_id())
    var g = lane >> 2
    var t = lane & 3
    var j0 = Int(block_idx.x) * 16

    var ex_s = stack_allocation[
        INT8_DECODE_MAX_M, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var red = stack_allocation[
        NW * 8 * WARP_SIZE, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()

    comptime if QA:
        # ---- THE ABSMAX of each left row, one warp a row (contract L-3).
        var r = w
        while r < m:
            var best = Float32(0.0)
            var slots = (k + 3) // 4
            var s = lane
            while s < slots:
                var c0 = s * 4
                if aligned and c0 + 4 <= k and (k & 3) == 0:
                    var v = x.unsafe_load[width=4, alignment=16](r * k + c0)
                    comptime for i in range(4):
                        best = _absmax_step(v[i], best)
                else:
                    comptime for i in range(4):
                        if c0 + i < k:
                            best = _absmax_step(x.unsafe_load(r * k + c0 + i), best)
                s += WARP_SIZE
            # A maximum of maxima over the warp: exact under any grouping.
            comptime for lvl in range(5):
                comptime if INT8_DECODE_QA_SABOTAGE and lvl == 0:
                    # SABOTAGE: the first level of the warp's butterfly is
                    # skipped; the upper half of the lanes' maxima never
                    # reach the lower half.
                    pass
                else:
                    var other = shuffle_xor(best, UInt32(16 >> lvl))
                    if other > best:
                        best = other
            if lane == 0:
                var ex = int8_row_exponent(best)
                ex_s.unsafe_store(r, Int32(ex))
                if block_idx.x == 0:
                    ea.unsafe_store(r, Int32(ex))
            r += NW
    else:
        if tid < m:
            ex_s.unsafe_store(tid, ea.unsafe_load(tid))
    barrier()

    var ex_g = 0
    var ex_g8 = 0
    comptime if QA:
        if g < m:
            ex_g = Int(ex_s.unsafe_load(g))
        if g + 8 < m:
            ex_g8 = Int(ex_s.unsafe_load(g + 8))

    var acc0 = SIMD[DType.int32, 4](0)
    var acc1 = SIMD[DType.int32, 4](0)
    var chunks = (k + INT8_DECODE_CHUNK - 1) // INT8_DECODE_CHUNK
    var ch = w
    while ch < chunks:
        var kb = ch * INT8_DECODE_CHUNK + t * 16
        var b0 = _load16_codes(qb, j0 + g, n, kb, k, aligned)
        var b1 = _load16_codes(qb, j0 + 8 + g, n, kb, k, aligned)
        var a0: SIMD[DType.int32, 4]
        var a1: SIMD[DType.int32, 4]
        comptime if QA:
            a0 = _quant16_codes(x, g, m, kb, k, ex_g, aligned)
            a1 = _quant16_codes(x, g + 8, m, kb, k, ex_g8, aligned)
        else:
            a0 = _load16_codes(qa, g, m, kb, k, aligned)
            a1 = _load16_codes(qa, g + 8, m, kb, k, aligned)
        comptime for s in range(2):
            acc0 = _imma_m16n8k32(
                a0[2 * s], a1[2 * s], a0[2 * s + 1], a1[2 * s + 1], b0[2 * s], b0[2 * s + 1], acc0
            )
            acc1 = _imma_m16n8k32(
                a0[2 * s], a1[2 * s], a0[2 * s + 1], a1[2 * s + 1], b1[2 * s], b1[2 * s + 1], acc1
            )
        ch += NW

    # ---- THE WARPS' SUMS, added in threadgroup memory (exact Int32).
    comptime for e in range(4):
        red.unsafe_store((w * 8 + e) * WARP_SIZE + lane, acc0[e])
        red.unsafe_store((w * 8 + 4 + e) * WARP_SIZE + lane, acc1[e])
    barrier()
    var q = tid
    while q < 8 * WARP_SIZE:
        var v = q // WARP_SIZE
        var ln = q - v * WARP_SIZE
        var total = Int32(0)
        for ww in range(NW):
            total += red.unsafe_load((ww * 8 + v) * WARP_SIZE + ln)
        # C/D (16 x 8, s32) of half `v // 4`: register `e = v % 4` of lane
        # `ln` is row `g + 8 (e >> 1)`, column `2 t + (e & 1)`.
        var e = v & 3
        var gi = (ln >> 2) + 8 * (e >> 1)
        var gj = j0 + (v >> 2) * 8 + (ln & 3) * 2 + (e & 1)
        if gi < m:
            _store_cell_decode(c, Int(ex_s.unsafe_load(gi)), eb, total, gi, gj, m, n)
        q += NT


def _refuse_decode_shape(m: Int, n: Int, k: Int) raises:
    if m <= 0 or n <= 0 or k <= 0 or m > INT8_DECODE_MAX_M or k > INT8_MAX_K:
        raise Error(
            "identical_gemm_int8_mma_decode: 1 <= m <= " + String(INT8_DECODE_MAX_M)
            + ", n and k positive, k at most " + String(INT8_MAX_K) + "; got m="
            + String(m) + " n=" + String(n) + " k=" + String(k)
        )


def _launch_decode[QA: Bool, NW: Int](
    ctx: DeviceContext,
    c: MutPointer[Float32, MutAnyOrigin],
    qa: MutPointer[Int8, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    ea: MutPointer[Int32, MutAnyOrigin],
    qb: MutPointer[Int8, MutAnyOrigin],
    eb: MutPointer[Int32, MutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
) raises:
    comptime kern = identical_gemm_int8_mma_decode_kernel[QA, NW]
    var aligned = Int32(0)
    var left = Int(x) if QA else Int(qa)
    if (left & 15) == 0 and (Int(qb) & 15) == 0:
        aligned = Int32(1)
    ctx.enqueue_function[kern](
        c,
        qa,
        x,
        ea,
        qb,
        eb,
        Int32(m),
        Int32(n),
        Int32(k),
        aligned,
        grid_dim=((n + 15) // 16, 1, 1),
        block_dim=(NW * WARP_SIZE, 1, 1),
    )


def _decode_with_plan[QA: Bool](
    ctx: DeviceContext,
    c: MutPointer[Float32, MutAnyOrigin],
    qa: MutPointer[Int8, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    ea: MutPointer[Int32, MutAnyOrigin],
    qb: MutPointer[Int8, MutAnyOrigin],
    eb: MutPointer[Int32, MutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
    plan: Int,
) raises:
    comptime if not INT8_DECODE_AVAILABLE:
        raise Error(
            "identical_gemm_int8_mma_decode: column " + column_name(TARGET_COLUMN)
            + " has no decode kernel (NVIDIA's m16n8k32 only); the tuned plans serve it"
        )
    else:
        _refuse_decode_shape(m, n, k)
        if plan == INT8_DECODE_PLAN_W4:
            _launch_decode[QA, 4](ctx, c, qa, x, ea, qb, eb, m, n, k)
        elif plan == INT8_DECODE_PLAN_W8:
            _launch_decode[QA, 8](ctx, c, qa, x, ea, qb, eb, m, n, k)
        elif plan == INT8_DECODE_PLAN_W16:
            _launch_decode[QA, 16](ctx, c, qa, x, ea, qb, eb, m, n, k)
        else:
            raise Error("identical_gemm_int8_mma_decode: no plan " + String(plan))


def identical_gemm_int8_mma_decode_with_plan(
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut qa: DeviceBuffer[DType.int8],
    mut ea: DeviceBuffer[DType.int32],
    mut qb: DeviceBuffer[DType.int8],
    mut eb: DeviceBuffer[DType.int32],
    m: Int,
    n: Int,
    k: Int,
    plan: Int,
) raises:
    """The decode kernel on codes, `identical_gemm_int8_mma_into`'s
    signature and its bits, `m <= 16`, NVIDIA. Asynchronous."""
    var cp = c.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    _decode_with_plan[False](
        ctx,
        cp,
        qa.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        cp,
        ea.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        qb.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        eb.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        m,
        n,
        k,
        plan,
    )


def identical_gemm_int8_mma_decode_quant_with_plan(
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    mut ea: DeviceBuffer[DType.int32],
    mut qb: DeviceBuffer[DType.int8],
    mut eb: DeviceBuffer[DType.int32],
    m: Int,
    n: Int,
    k: Int,
    plan: Int,
) raises:
    """THE QUANTIZER IN THE PRODUCT'S LAUNCH: `x` (m x k float32) in, `C`
    out, ONE launch; `ea` receives the left rows' exponents. The same
    cells as `quantize_rows_int8_par_device` then
    `identical_gemm_int8_mma_tuned_into`. `m <= 16`, NVIDIA.
    Asynchronous."""
    var cp = c.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    _decode_with_plan[True](
        ctx,
        cp,
        qb.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        x.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        ea.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        qb.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        eb.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        m,
        n,
        k,
        plan,
    )


# ===========================================================================
# FOUR PRODUCTS, ONE STAGING
# ===========================================================================


@always_inline
def _store_sums(
    s: MutPointer[Int32, MutAnyOrigin],
    hh: Int32,
    mid: Int32,
    ll: Int32,
    i: Int,
    j: Int,
    m: Int,
    n: Int,
):
    """The three sums of one cell, side by side at `3 (i n + j)`, masked to
    the output. The value arm flips the lowest bit of each."""
    if i >= m or j >= n:
        return
    var at_ = 3 * (i * n + j)
    comptime if INT8_TUNED_VALUE_SABOTAGE:
        s.unsafe_store(at_, hh ^ Int32(1))
        s.unsafe_store(at_ + 1, mid ^ Int32(1))
        s.unsafe_store(at_ + 2, ll ^ Int32(1))
    else:
        s.unsafe_store(at_, hh)
        s.unsafe_store(at_ + 1, mid)
        s.unsafe_store(at_ + 2, ll)


@always_inline
def _store_cell_of_sums[FUSED: Bool](
    s: MutPointer[Int32, MutAnyOrigin],
    c: MutPointer[Float32, MutAnyOrigin],
    ea: MutPointer[Int32, MutAnyOrigin],
    eb: MutPointer[Int32, MutAnyOrigin],
    hh: Int32,
    mid: Int32,
    ll: Int32,
    i: Int,
    j: Int,
    m: Int,
    n: Int,
):
    """One cell of either form, masked to the output: the three sums as
    they are, or what the caller's seam (`int15_store_cell`) makes of
    them. This file states no float rule."""
    comptime if FUSED:
        int15_store_cell(c, ea, eb, hh, mid, ll, i, j, m, n)
    else:
        _store_sums(s, hh, mid, ll, i, j, m, n)


@always_inline
def _async_wait_all():
    """Every `cp.async` this thread issued has landed in threadgroup memory
    (PTX `cp.async.wait_all`, which commits the open group first). Nothing
    on a column with no `cp.async`, whose staging is synchronous."""
    comptime if is_nvidia_gpu():
        inlined_assembly["cp.async.wait_all;", NoneType, constraints="~{memory}"]()


@always_inline
def _stage_window_async[
    ROWS: Int, KB: Int, NT: Int, LW: Int, SW: Int
](
    dst: UnsafePointer[Int32, MutUntrackedOrigin, address_space=AddressSpace.SHARED],
    dst0: Int,
    q: MutPointer[Int8, MutAnyOrigin],
    row0: Int,
    rows: Int,
    k0: Int,
    k: Int,
    tid: Int,
    aligned: Bool,
):
    """`_stage_window`, the same loads to the same words, with the whole
    aligned loads issued as `cp.async` on NVIDIA: the codes go from device
    memory to threadgroup memory with no register between, and the thread
    goes on. A load that is not whole and aligned (a ragged row end, a row
    or step beyond the operand) is written as `_stage_window` writes it,
    zero codes stored, synchronously. The caller waits (`_async_wait_all`)
    and then reaches a barrier before any thread reads the page."""
    comptime if not is_nvidia_gpu():
        _stage_window[ROWS, KB, NT, LW, SW](dst, dst0, q, row0, rows, k0, k, tid, aligned)
    else:
        comptime LPR = KB // LW
        comptime LOADS = ROWS * LPR
        comptime SL = (LOADS + NT - 1) // NT
        comptime CP = "cp.async.cg.shared.global [$0], [$1], 16;" if LW == 16 else (
            "cp.async.ca.shared.global [$0], [$1], 8;" if LW == 8 else "cp.async.ca.shared.global [$0], [$1], 4;"
        )
        comptime for sl in range(SL):
            var s = sl * NT + tid
            if s < LOADS:
                var r = s // LPR
                var kq = (s % LPR) * LW
                var gr = row0 + r
                var inside = gr < rows and k0 + kq < k
                var at_ = dst0 + r * SW + kq // 4
                var base = gr * k + k0 + kq
                if inside and aligned and k0 + kq + LW <= k and (k & (LW - 1)) == 0:
                    inlined_assembly[CP, NoneType, constraints="r,l,~{memory}"](
                        Int32(Int(dst) + 4 * at_), Int64(Int(q) + base)
                    )
                else:
                    var v = SIMD[DType.int8, LW](0)
                    if inside:
                        comptime for i in range(LW):
                            if k0 + kq + i < k:
                                v[i] = q.unsafe_load(base + i)
                    comptime if INT8_TUNED_SABOTAGE:
                        # SABOTAGE: the padding rule broken, as in
                        # `_stage_window`.
                        if inside:
                            dst.unsafe_store[alignment=LW](at_, bitcast[DType.int32, LW // 4](v))
                    else:
                        dst.unsafe_store[alignment=LW](at_, bitcast[DType.int32, LW // 4](v))


def identical_gemm_int8_pieces_flat_kernel(
    s: MutPointer[Int32, MutAnyOrigin],
    ah: MutPointer[Int8, MutAnyOrigin],
    al: MutPointer[Int8, MutAnyOrigin],
    bh: MutPointer[Int8, MutAnyOrigin],
    bl: MutPointer[Int8, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    k_in: Int32,
):
    """THE REFERENCE DEVICE PLAN of the four products: one thread per cell,
    `p` ascending, one product per step into each of the three Int32 sums.
    No unit, no staging, every column. Grid `ceil(m n / 256)`, block 256."""
    var m = Int(m_in)
    var n = Int(n_in)
    var k = Int(k_in)
    var cell = Int(block_idx.x) * 256 + Int(thread_idx.x)
    if cell >= m * n:
        return
    var i = cell // n
    var j = cell - i * n
    var hh = Int32(0)
    var mid = Int32(0)
    var ll = Int32(0)
    for p in range(k):
        var a_hi = Int32(ah.unsafe_load(i * k + p))
        var a_lo = Int32(al.unsafe_load(i * k + p))
        var b_hi = Int32(bh.unsafe_load(j * k + p))
        var b_lo = Int32(bl.unsafe_load(j * k + p))
        hh += a_hi * b_hi
        mid += a_hi * b_lo + a_lo * b_hi
        ll += a_lo * b_lo
    _store_sums(s, hh, mid, ll, i, j, m, n)


@always_inline
def _pieces_window_steps[
    FM: Int, FN: Int, KB: Int, SW: Int, APLANE: Int, BPLANE: Int, NACC: Int
](
    mut acc: InlineArray[SIMD[DType.int32, 4], 3 * NACC],
    as_: UnsafePointer[Int32, MutUntrackedOrigin, address_space=AddressSpace.SHARED],
    bs_: UnsafePointer[Int32, MutUntrackedOrigin, address_space=AddressSpace.SHARED],
    abase: Int,
    bbase: Int,
    k0: Int,
    k: Int,
    lane: Int,
    lrow: Int,
    lcol: Int,
    live: Bool,
):
    """The unit steps of one staged window, from the page that starts at
    word `abase` of `as_` and `bbase` of `bs_`, into the three accumulators
    of every half tile. Nothing here waits: the caller's barriers bound it."""
    comptime KSTEPS = KB // INT8_TUNED_K_TILE
    if live:
        comptime for ks in range(KSTEPS):
            if k0 + ks * INT8_TUNED_K_TILE < k:
                comptime if is_nvidia_gpu():
                    var g = lane >> 2
                    var t = lane & 3
                    var kw = ks * 8 + t
                    # Per half tile of B: the high plane's two words,
                    # then the low plane's.
                    var bfr = InlineArray[Int32, 8 * FN](fill=Int32(0))
                    comptime for fq in range(FN):
                        comptime for h in range(2):
                            var br = (lcol + fq * 16 + h * 8 + g) * SW + kw
                            bfr[(fq * 2 + h) * 4] = bs_.unsafe_load(bbase + br)
                            bfr[(fq * 2 + h) * 4 + 1] = bs_.unsafe_load(bbase + br + 4)
                            bfr[(fq * 2 + h) * 4 + 2] = bs_.unsafe_load(bbase + BPLANE + br)
                            bfr[(fq * 2 + h) * 4 + 3] = bs_.unsafe_load(bbase + BPLANE + br + 4)
                    comptime for fm in range(FM):
                        var ar = (lrow + fm * 16 + g) * SW + kw
                        var h0 = as_.unsafe_load(abase + ar)
                        var h1 = as_.unsafe_load(abase + ar + 8 * SW)
                        var h2 = as_.unsafe_load(abase + ar + 4)
                        var h3 = as_.unsafe_load(abase + ar + 8 * SW + 4)
                        var l0 = as_.unsafe_load(abase + APLANE + ar)
                        var l1 = as_.unsafe_load(abase + APLANE + ar + 8 * SW)
                        var l2 = as_.unsafe_load(abase + APLANE + ar + 4)
                        var l3 = as_.unsafe_load(abase + APLANE + ar + 8 * SW + 4)
                        comptime for fq in range(FN):
                            comptime for h in range(2):
                                var bh0 = bfr[(fq * 2 + h) * 4]
                                var bh1 = bfr[(fq * 2 + h) * 4 + 1]
                                var bl0 = bfr[(fq * 2 + h) * 4 + 2]
                                var bl1 = bfr[(fq * 2 + h) * 4 + 3]
                                acc[((fm * FN + fq) * 2 + h) * 3] = _imma_m16n8k32(
                                    h0, h1, h2, h3, bh0, bh1,
                                    acc[((fm * FN + fq) * 2 + h) * 3],
                                )
                                acc[((fm * FN + fq) * 2 + h) * 3 + 1] = _imma_m16n8k32(
                                    h0, h1, h2, h3, bl0, bl1,
                                    acc[((fm * FN + fq) * 2 + h) * 3 + 1],
                                )
                                comptime if INT8_PIECES_SABOTAGE:
                                    # SABOTAGE: HL again where LH belongs.
                                    acc[((fm * FN + fq) * 2 + h) * 3 + 1] = _imma_m16n8k32(
                                        h0, h1, h2, h3, bl0, bl1,
                                        acc[((fm * FN + fq) * 2 + h) * 3 + 1],
                                    )
                                else:
                                    acc[((fm * FN + fq) * 2 + h) * 3 + 1] = _imma_m16n8k32(
                                        l0, l1, l2, l3, bh0, bh1,
                                        acc[((fm * FN + fq) * 2 + h) * 3 + 1],
                                    )
                                acc[((fm * FN + fq) * 2 + h) * 3 + 2] = _imma_m16n8k32(
                                    l0, l1, l2, l3, bl0, bl1,
                                    acc[((fm * FN + fq) * 2 + h) * 3 + 2],
                                )
                elif is_amd_gpu():
                    var i16 = lane & 15
                    var kw = ks * 8 + (lane >> 4) * 2
                    var bfr = InlineArray[Int64, 2 * FN](fill=Int64(0))
                    comptime for fq in range(FN):
                        var br = (lcol + fq * 16 + i16) * SW + kw
                        bfr[fq * 2] = bitcast[DType.int64, 1](
                            bs_.unsafe_load[width=2, alignment=8](bbase + br)
                        )[0]
                        bfr[fq * 2 + 1] = bitcast[DType.int64, 1](
                            bs_.unsafe_load[width=2, alignment=8](bbase + BPLANE + br)
                        )[0]
                    comptime for fm in range(FM):
                        var ar = (lrow + fm * 16 + i16) * SW + kw
                        var a_hi = bitcast[DType.int64, 1](
                            as_.unsafe_load[width=2, alignment=8](abase + ar)
                        )[0]
                        var a_lo = bitcast[DType.int64, 1](
                            as_.unsafe_load[width=2, alignment=8](abase + APLANE + ar)
                        )[0]
                        comptime for fq in range(FN):
                            acc[(fm * FN + fq) * 3] = _mfma_i8(
                                a_hi, bfr[fq * 2], acc[(fm * FN + fq) * 3]
                            )
                            acc[(fm * FN + fq) * 3 + 1] = _mfma_i8(
                                a_hi, bfr[fq * 2 + 1], acc[(fm * FN + fq) * 3 + 1]
                            )
                            comptime if INT8_PIECES_SABOTAGE:
                                acc[(fm * FN + fq) * 3 + 1] = _mfma_i8(
                                    a_hi, bfr[fq * 2 + 1], acc[(fm * FN + fq) * 3 + 1]
                                )
                            else:
                                acc[(fm * FN + fq) * 3 + 1] = _mfma_i8(
                                    a_lo, bfr[fq * 2], acc[(fm * FN + fq) * 3 + 1]
                                )
                            acc[(fm * FN + fq) * 3 + 2] = _mfma_i8(
                                a_lo, bfr[fq * 2 + 1], acc[(fm * FN + fq) * 3 + 2]
                            )


@always_inline
def _pieces_block[
    FUSED: Bool, PIPE: Bool, FM: Int, FN: Int, WM: Int, WN: Int, KB: Int, LW: Int
](
    s: MutPointer[Int32, MutAnyOrigin],
    c: MutPointer[Float32, MutAnyOrigin],
    ea: MutPointer[Int32, MutAnyOrigin],
    eb: MutPointer[Int32, MutAnyOrigin],
    ah: MutPointer[Int8, MutAnyOrigin],
    al: MutPointer[Int8, MutAnyOrigin],
    bh: MutPointer[Int8, MutAnyOrigin],
    bl: MutPointer[Int8, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    k_in: Int32,
    aligned_in: Int32,
):
    """The body of both forms of the four-product kernel. OP_NT, ONE
    staging: `(Ah . Bh^T, Ah . Bl^T + Al . Bh^T, Al . Bl^T)[i, j]`.
    `identical_gemm_int8_mma_tuned_kernel`'s block, warps, windows and
    barriers; the staged page of an operand holds its two planes, the high
    one first; a tile's four unit steps read the fragments of the two
    planes of each side, loaded once. `FUSED` False: the three sums to `s`,
    and `c`, `ea`, `eb` are not read. `FUSED` True: `int15_store_cell` of
    the three sums to `c` (the caller's seam), and `s` is not written.

    `PIPE` True: TWO PAGES. The window after the one the unit steps read is
    staged into the other page while they run, with `cp.async` on NVIDIA
    (no register holds the codes on their way), and one barrier per window
    where the one-page form has two. On a column with no `cp.async` the
    same schedule stages synchronously. The same codes land at the same
    words either way; only the order of the work changes.

    Every thread of the block reaches every `barrier()`, for the reasons
    the one-product kernel gives."""
    comptime TM = INT8_TUNED_TILE * FM
    comptime TN = INT8_TUNED_TILE * FN
    comptime BM = WM * TM
    comptime BN = WN * TN
    comptime NT = WM * WN * WARP_SIZE
    comptime SS = KB + INT8_TUNED_ROW_PAD
    comptime SW = SS // 4
    comptime KSTEPS = KB // INT8_TUNED_K_TILE
    comptime HALVES = 2 if is_nvidia_gpu() else 1
    comptime NACC = FM * FN * HALVES
    #: Words of one plane of each staged operand.
    comptime APLANE = BM * SW
    comptime BPLANE = BN * SW

    comptime assert KB % INT8_TUNED_K_TILE == 0 and KB >= INT8_TUNED_K_TILE, (
        "identical_gemm_int8_pieces_tuned_kernel: a window is whole unit steps"
    )
    comptime assert LW == 4 or LW == 8 or LW == 16, (
        "identical_gemm_int8_pieces_tuned_kernel: a staging load is 4, 8 or 16 bytes"
    )
    comptime assert NT <= INT8_TUNED_MAX_TPB, (
        "identical_gemm_int8_pieces_tuned_kernel: a block is at most 1024 threads"
    )
    comptime PAGES = 2 if PIPE else 1
    #: Words of one page of each operand: its two planes.
    comptime APAGE = 2 * APLANE
    comptime BPAGE = 2 * BPLANE
    comptime assert PAGES * 2 * (BM + BN) * SS <= column_shared_limit(TARGET_COLUMN), (
        "identical_gemm_int8_pieces_tuned_kernel: the staged window does not"
        " fit the column's threadgroup memory"
    )

    var m = Int(m_in)
    var n = Int(n_in)
    var k = Int(k_in)
    var aligned = aligned_in != Int32(0)
    var tid = Int(thread_idx.x)
    var warp = tid // WARP_SIZE
    var lane = Int(lane_id())
    var wm = warp // WN
    var wn = warp - wm * WN
    var i0 = Int(block_idx.y) * BM
    var j0 = Int(block_idx.x) * BN
    var lrow = wm * TM
    var lcol = wn * TN
    var live = i0 + lrow < m and j0 + lcol < n

    var as_ = stack_allocation[
        PAGES * APAGE,
        Scalar[DType.int32],
        alignment=16,
        address_space = AddressSpace.SHARED,
    ]()
    var bs_ = stack_allocation[
        PAGES * BPAGE,
        Scalar[DType.int32],
        alignment=16,
        address_space = AddressSpace.SHARED,
    ]()

    # Three accumulators per half tile: HH, HL + LH, LL.
    var acc = InlineArray[SIMD[DType.int32, 4], 3 * NACC](fill=SIMD[DType.int32, 4](0))

    var windows = (k + KB - 1) // KB
    comptime if PIPE:
        _stage_window_async[BM, KB, NT, LW, SW](as_, 0, ah, i0, m, 0, k, tid, aligned)
        _stage_window_async[BM, KB, NT, LW, SW](as_, APLANE, al, i0, m, 0, k, tid, aligned)
        _stage_window_async[BN, KB, NT, LW, SW](bs_, 0, bh, j0, n, 0, k, tid, aligned)
        _stage_window_async[BN, KB, NT, LW, SW](bs_, BPLANE, bl, j0, n, 0, k, tid, aligned)
        _async_wait_all()
        barrier()
        for w in range(windows):
            var cur = w & 1
            if w + 1 < windows:
                # The other page was last read in window `w - 1`, whose
                # steps every thread left before the barrier that ended it.
                var nxt = 1 - cur
                var k1 = (w + 1) * KB
                _stage_window_async[BM, KB, NT, LW, SW](as_, nxt * APAGE, ah, i0, m, k1, k, tid, aligned)
                _stage_window_async[BM, KB, NT, LW, SW](
                    as_, nxt * APAGE + APLANE, al, i0, m, k1, k, tid, aligned
                )
                _stage_window_async[BN, KB, NT, LW, SW](bs_, nxt * BPAGE, bh, j0, n, k1, k, tid, aligned)
                _stage_window_async[BN, KB, NT, LW, SW](
                    bs_, nxt * BPAGE + BPLANE, bl, j0, n, k1, k, tid, aligned
                )
            _pieces_window_steps[FM, FN, KB, SW, APLANE, BPLANE, NACC](
                acc, as_, bs_, cur * APAGE, cur * BPAGE, w * KB, k, lane, lrow, lcol, live
            )
            # This thread's copies into the other page have landed; the
            # barrier makes every thread's visible and frees this page.
            _async_wait_all()
            barrier()
    else:
        for w in range(windows):
            var k0 = w * KB
            _stage_window[BM, KB, NT, LW, SW](as_, 0, ah, i0, m, k0, k, tid, aligned)
            _stage_window[BM, KB, NT, LW, SW](as_, APLANE, al, i0, m, k0, k, tid, aligned)
            _stage_window[BN, KB, NT, LW, SW](bs_, 0, bh, j0, n, k0, k, tid, aligned)
            _stage_window[BN, KB, NT, LW, SW](bs_, BPLANE, bl, j0, n, k0, k, tid, aligned)
            barrier()
            _pieces_window_steps[FM, FN, KB, SW, APLANE, BPLANE, NACC](
                acc, as_, bs_, 0, 0, k0, k, lane, lrow, lcol, live
            )
            barrier()

    if not live:
        return

    var outs = stack_allocation[NACC * 12, Scalar[DType.int32]]()
    comptime for a in range(NACC):
        comptime for p in range(3):
            comptime for e in range(4):
                outs.unsafe_store(a * 12 + p * 4 + e, acc[a * 3 + p][e])
    comptime if is_nvidia_gpu():
        var g = lane >> 2
        var t = lane & 3
        for a in range(NACC):
            var tile = a >> 1
            var h = a & 1
            var fm = tile // FN
            var fq = tile - fm * FN
            var gi = i0 + lrow + fm * 16 + g
            var gj = j0 + lcol + fq * 16 + h * 8 + t * 2
            comptime for e in range(4):
                _store_cell_of_sums[FUSED](
                    s,
                    c,
                    ea,
                    eb,
                    outs.unsafe_load(a * 12 + e),
                    outs.unsafe_load(a * 12 + 4 + e),
                    outs.unsafe_load(a * 12 + 8 + e),
                    gi + 8 * (e >> 1),
                    gj + (e & 1),
                    m,
                    n,
                )
    elif is_amd_gpu():
        var i16 = lane & 15
        var i4 = (lane >> 4) * 4
        for a in range(NACC):
            var fm = a // FN
            var fq = a - fm * FN
            var gi = i0 + lrow + fm * 16 + i4
            var gj = j0 + lcol + fq * 16 + i16
            comptime for e in range(4):
                _store_cell_of_sums[FUSED](
                    s,
                    c,
                    ea,
                    eb,
                    outs.unsafe_load(a * 12 + e),
                    outs.unsafe_load(a * 12 + 4 + e),
                    outs.unsafe_load(a * 12 + 8 + e),
                    gi + e,
                    gj,
                    m,
                    n,
                )


def identical_gemm_int8_pieces_tuned_kernel[
    FUSED: Bool, PIPE: Bool, FM: Int, FN: Int, WM: Int, WN: Int, KB: Int, LW: Int
](
    s: MutPointer[Int32, MutAnyOrigin],
    c: MutPointer[Float32, MutAnyOrigin],
    ea: MutPointer[Int32, MutAnyOrigin],
    eb: MutPointer[Int32, MutAnyOrigin],
    ah: MutPointer[Int8, MutAnyOrigin],
    al: MutPointer[Int8, MutAnyOrigin],
    bh: MutPointer[Int8, MutAnyOrigin],
    bl: MutPointer[Int8, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    k_in: Int32,
    aligned_in: Int32,
):
    """FOUR PRODUCTS, ONE STAGING. `FUSED` False, THE SUMS FORM: three Int32
    per cell at `s[3 (i n + j)]`, HH, HL + LH, LL; `c`, `ea` and `eb` are not
    read. `FUSED` True: `int15_store_cell(c, ea, eb, HH, HL + LH, LL, i, j,
    m, n)` per cell, in the launch that computed the sums; `s` is not
    written. Grid `(ceil(n / BN), ceil(m / BM), 1)`, block `WM * WN *
    WARP_SIZE`, for both. `PIPE`: see `_pieces_block`."""
    _pieces_block[FUSED, PIPE, FM, FN, WM, WN, KB, LW](
        s, c, ea, eb, ah, al, bh, bl, m_in, n_in, k_in, aligned_in
    )


def _refuse_pieces_shape(m: Int, n: Int, k: Int) raises:
    if m <= 0 or n <= 0 or k <= 0 or k > INT8_PIECES_MAX_K:
        raise Error(
            "identical_gemm_int8_pieces: m, n and k must be positive and k at"
            " most " + String(INT8_PIECES_MAX_K) + " (the middle sum takes two"
            " products per step in one Int32), got m=" + String(m)
            + " n=" + String(n) + " k=" + String(k)
        )


def identical_gemm_int8_pieces_flat_into(
    ctx: DeviceContext,
    mut s: DeviceBuffer[DType.int32],
    mut ah: DeviceBuffer[DType.int8],
    mut al: DeviceBuffer[DType.int8],
    mut bh: DeviceBuffer[DType.int8],
    mut bl: DeviceBuffer[DType.int8],
    m: Int,
    n: Int,
    k: Int,
) raises:
    """The reference device plan of the four products, always. `s` holds
    `3 m n` Int32. Every column. Asynchronous."""
    _refuse_pieces_shape(m, n, k)
    ctx.enqueue_function[identical_gemm_int8_pieces_flat_kernel](
        s.unsafe_ptr(),
        ah.unsafe_ptr(),
        al.unsafe_ptr(),
        bh.unsafe_ptr(),
        bl.unsafe_ptr(),
        Int32(m),
        Int32(n),
        Int32(k),
        grid_dim=((m * n + 255) // 256, 1, 1),
        block_dim=(256, 1, 1),
    )


def _launch_pieces[
    FUSED: Bool, PIPE: Bool, FM: Int, FN: Int, WM: Int, WN: Int, KB: Int, LW: Int
](
    ctx: DeviceContext,
    s: MutPointer[Int32, MutAnyOrigin],
    c: MutPointer[Float32, MutAnyOrigin],
    ea: MutPointer[Int32, MutAnyOrigin],
    eb: MutPointer[Int32, MutAnyOrigin],
    mut ah: DeviceBuffer[DType.int8],
    mut al: DeviceBuffer[DType.int8],
    mut bh: DeviceBuffer[DType.int8],
    mut bl: DeviceBuffer[DType.int8],
    m: Int,
    n: Int,
    k: Int,
) raises:
    """One launch of either form. The form's unused buffers are passed and
    not touched (the caller hands the ones it has)."""
    comptime BM = WM * FM * INT8_TUNED_TILE
    comptime BN = WN * FN * INT8_TUNED_TILE
    comptime kern = identical_gemm_int8_pieces_tuned_kernel[FUSED, PIPE, FM, FN, WM, WN, KB, LW]
    var aligned = Int32(0)
    if _bases_aligned(Int(ah.unsafe_ptr()), Int(al.unsafe_ptr()), LW) and _bases_aligned(
        Int(bh.unsafe_ptr()), Int(bl.unsafe_ptr()), LW
    ):
        aligned = Int32(1)
    ctx.enqueue_function[kern](
        s,
        c,
        ea,
        eb,
        ah.unsafe_ptr(),
        al.unsafe_ptr(),
        bh.unsafe_ptr(),
        bl.unsafe_ptr(),
        Int32(m),
        Int32(n),
        Int32(k),
        aligned,
        grid_dim=((n + BN - 1) // BN, (m + BM - 1) // BM, 1),
        block_dim=(WM * WN * WARP_SIZE, 1, 1),
    )


def _pieces_with_plan[FUSED: Bool](
    ctx: DeviceContext,
    s: MutPointer[Int32, MutAnyOrigin],
    c: MutPointer[Float32, MutAnyOrigin],
    ea: MutPointer[Int32, MutAnyOrigin],
    eb: MutPointer[Int32, MutAnyOrigin],
    mut ah: DeviceBuffer[DType.int8],
    mut al: DeviceBuffer[DType.int8],
    mut bh: DeviceBuffer[DType.int8],
    mut bl: DeviceBuffer[DType.int8],
    m: Int,
    n: Int,
    k: Int,
    plan: Int,
) raises:
    comptime if not INT8_TUNED_AVAILABLE:
        raise Error(
            "identical_gemm_int8_pieces_tuned: column " + column_name(TARGET_COLUMN)
            + " has no int8 matrix unit (kernel_matrix row"
            " lib_int8_matrix_unit_for); the flat kernel serves it"
        )
    else:
        _refuse_pieces_shape(m, n, k)
        if plan == INT8_PIECES_PLAN_SMALL:
            _launch_pieces[FUSED, False, 1, 1, 2, 2, 64, 16](ctx, s, c, ea, eb, ah, al, bh, bl, m, n, k)
        elif plan == INT8_PIECES_PLAN_WARPS16:
            _launch_pieces[FUSED, False, 1, 2, 4, 4, 64, 16](ctx, s, c, ea, eb, ah, al, bh, bl, m, n, k)
        elif plan == INT8_PIECES_PLAN_FRAG2:
            _launch_pieces[FUSED, False, 2, 2, 2, 4, 64, 16](ctx, s, c, ea, eb, ah, al, bh, bl, m, n, k)
        elif plan == INT8_PIECES_PLAN_SQUARE:
            _launch_pieces[FUSED, False, 1, 1, 4, 4, 64, 16](ctx, s, c, ea, eb, ah, al, bh, bl, m, n, k)
        elif plan == INT8_PIECES_PLAN_TALL128:
            _launch_pieces[FUSED, False, 1, 2, 8, 2, 32, 16](ctx, s, c, ea, eb, ah, al, bh, bl, m, n, k)
        elif plan == INT8_PIECES_PLAN_TALL256:
            _launch_pieces[FUSED, False, 1, 2, 16, 1, 32, 16](ctx, s, c, ea, eb, ah, al, bh, bl, m, n, k)
        elif plan == INT8_PIECES_PLAN_TALL256_N16:
            _launch_pieces[FUSED, False, 1, 1, 16, 1, 32, 16](ctx, s, c, ea, eb, ah, al, bh, bl, m, n, k)
        elif plan == INT8_PIECES_PLAN_WARPS16_K32:
            _launch_pieces[FUSED, False, 1, 2, 4, 4, 32, 16](ctx, s, c, ea, eb, ah, al, bh, bl, m, n, k)
        elif plan == INT8_PIECES_PLAN_PIPE_WARPS16:
            _launch_pieces[FUSED, True, 1, 2, 4, 4, 32, 16](ctx, s, c, ea, eb, ah, al, bh, bl, m, n, k)
        elif plan == INT8_PIECES_PLAN_PIPE_FRAG2:
            _launch_pieces[FUSED, True, 2, 2, 2, 4, 32, 16](ctx, s, c, ea, eb, ah, al, bh, bl, m, n, k)
        elif plan == INT8_PIECES_PLAN_PIPE_SMALL:
            _launch_pieces[FUSED, True, 1, 1, 2, 2, 64, 16](ctx, s, c, ea, eb, ah, al, bh, bl, m, n, k)
        else:
            raise Error("identical_gemm_int8_pieces_tuned: no plan " + String(plan))


def identical_gemm_int8_pieces_tuned_with_plan(
    ctx: DeviceContext,
    mut s: DeviceBuffer[DType.int32],
    mut ah: DeviceBuffer[DType.int8],
    mut al: DeviceBuffer[DType.int8],
    mut bh: DeviceBuffer[DType.int8],
    mut bl: DeviceBuffer[DType.int8],
    m: Int,
    n: Int,
    k: Int,
    plan: Int,
) raises:
    """The four-product kernel on a NAMED plan, for the gate and the timing
    harness. `s` holds `3 m n` Int32: cell `(i, j)`'s HH, HL + LH and LL at
    `3 (i n + j)`. Refuses by name on a column with no integer matrix unit.
    Asynchronous."""
    # The sums form reads no float and no exponent: `c` is `s` under the
    # float type, `ea` and `eb` are `s`, and none of the three is touched.
    var sp = s.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    _pieces_with_plan[False](ctx, sp, sp.bitcast[Float32](), sp, sp, ah, al, bh, bl, m, n, k, plan)


def identical_gemm_int8_pieces_tuned_into(
    ctx: DeviceContext,
    mut s: DeviceBuffer[DType.int32],
    mut ah: DeviceBuffer[DType.int8],
    mut al: DeviceBuffer[DType.int8],
    mut bh: DeviceBuffer[DType.int8],
    mut bl: DeviceBuffer[DType.int8],
    m: Int,
    n: Int,
    k: Int,
) raises:
    """**THE ENTRY POINT for the fifteen-bit profile's four products**, on
    the plan `int8_pieces_dispatch` names: two int8 planes of each operand
    in, three exact Int32 sums per cell out. The recombination and the
    seams are the caller's. Asynchronous."""
    identical_gemm_int8_pieces_tuned_with_plan(
        ctx, s, ah, al, bh, bl, m, n, k, int8_pieces_dispatch(m, n, k)
    )


def identical_gemm_int8_pieces_tuned_fused_with_plan(
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut ah: DeviceBuffer[DType.int8],
    mut al: DeviceBuffer[DType.int8],
    mut ea: DeviceBuffer[DType.int32],
    mut bh: DeviceBuffer[DType.int8],
    mut bl: DeviceBuffer[DType.int8],
    mut eb: DeviceBuffer[DType.int32],
    m: Int,
    n: Int,
    k: Int,
    plan: Int,
) raises:
    """THE FUSED FORM on a NAMED plan: two int8 planes and the row
    exponents of each operand in, `C[m x n]` float32 out, ONE launch, each
    cell stored by `int15_store_cell` (the caller's seam). No sums buffer.
    The sums form's refusals and plans. Asynchronous."""
    # The fused form writes no sums: `s` is `ea` and is not touched.
    _pieces_with_plan[True](
        ctx,
        ea.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        c.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        ea.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        eb.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        ah,
        al,
        bh,
        bl,
        m,
        n,
        k,
        plan,
    )


def identical_gemm_int8_pieces_tuned_fused_into(
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut ah: DeviceBuffer[DType.int8],
    mut al: DeviceBuffer[DType.int8],
    mut ea: DeviceBuffer[DType.int32],
    mut bh: DeviceBuffer[DType.int8],
    mut bl: DeviceBuffer[DType.int8],
    mut eb: DeviceBuffer[DType.int32],
    m: Int,
    n: Int,
    k: Int,
) raises:
    """**THE ENTRY POINT for a profile whose seam is `int15_store_cell`**:
    the fused form on the plan `int8_pieces_dispatch` names. Asynchronous."""
    identical_gemm_int8_pieces_tuned_fused_with_plan(
        ctx, c, ah, al, ea, bh, bl, eb, m, n, k, int8_pieces_dispatch(m, n, k)
    )
