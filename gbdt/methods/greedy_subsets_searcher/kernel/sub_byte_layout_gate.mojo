# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE GATE ON DEVIATION 1947, WITH THE ARM THAT MUST FAIL.

WHAT 1947 CLAIMS. The CatBoost histogram accumulators' 32 is a LOGICAL
partition of thread indices, not a hardware wave width, so their slice layout
is valid at any wave width once the constant feeding it is read from
`replication_lanes_for` instead of `lane_width_for`. That is what removed the
64-lane refusals for the binary, half-byte and one-byte families.

WHAT THAT CLAIM RESTS ON, and it is exactly one property per family:

  HALF-BYTE / BINARY. `Reduce()`'s stage-1 fold stride must equal the stride
  `slice_offset` writes at. Stage 1 sends cell `c` to slot `c % REDUCE_WIDTH`
  and stage 2 reads only the slots `reduce_stage2_slot` names. If the fold
  stride is a MULTIPLE of the slice stride rather than equal to it -- which
  is precisely what `REDUCE_WIDTH = hardware_lanes * 16` produces on a
  64-wide wave -- then half the replicas land in slots stage 2 never reads
  and their share of every histogram is silently dropped. Nothing crashes and
  no assert fires at runtime; the numbers are just low.

  ONE-BYTE LADDER. Two threads that share a private replica must be
  separable by the sub-copy mask. Within a replica a thread's identity enters
  the slot only through the low five bits of `threadIdx.x`, so a replica may
  hold at most 32 threads. `1024 * (tid // 32)` gives it exactly 32;
  `1024 * (tid // 64)` gives it 64, and the second half of each replica
  writes the SAME slots as the first with PLAIN, non-atomic float adds. Lost
  updates, again with nothing to see at runtime.

  HIST_2 SHARED-INT32. The shared allocation must hold as many 1024-slot
  slices as the slice keying hands out. The keying is a literal `tid // 64`;
  the sizing was `block // (2 * LANE_WIDTH)`, which agrees at 32 and
  UNDER-ALLOCATES BY HALF at 64.

WHY THIS FILE EXISTS AT ALL. Those three properties are defended in the
kernels by comptime asserts, and a comptime assert that has never been seen
to fire is a claim, not a gate. Each arm below therefore runs TWICE: once
with the shipped constants, where it must pass, and once with the constant
deliberately RE-COUPLED to the hardware lane width of the CDNA column, where
it must FAIL. If a sabotage arm passes, this file raises. That is the whole
point of it: the invariant is defended by something that can bite.

HOW THE SABOTAGE STAYS HONEST. A comptime assert cannot be sabotaged from
inside a program it would refuse to compile, so each arm carries a MODEL of
the layout arithmetic parameterized on the striping constant. A model can
drift from the code it models, so every model is first cross-checked against
the REAL function from the real kernel module at the shipped constant, cell
for cell, before it is run at the sabotaged one. A drifted model fails the
cross-check arm and this file raises on that too.

WHAT THIS GATE IS NOT. It is arithmetic about a layout; it executes no
kernel and touches no GPU. It says the layout is self-consistent at any wave
width. It does NOT say a sub-byte block has produced a correct histogram on
64-wide hardware, because none ever has -- see DEVIATION 1947's verification
clause. Both statements are needed and this file makes only the first.

RUN IT

    pixi run mojo run -I . \\
      gbdt/methods/greedy_subsets_searcher/kernel/sub_byte_layout_gate.mojo

It is host arithmetic, so it runs and gates on any box, including a laptop
with no GPU and including the Apple column that cannot reproduce the defect.
"""

from checks.kernel_matrix import (
    COLUMN_AMD,
    HIST_SMEM_SHARED2_I32,
    HIST_SMEM_WARP_PRIVATE_F32,
    PINNED_REPLICATION_LANES,
    SYNC_LANE,
    TARGET_COLUMN,
    column_lane_width,
    column_name,
    replication_lanes_for,
    sub_byte_lane_sync_for,
)
from gbdt.methods.greedy_subsets_searcher.kernel.point_hist_half_byte_template import (
    BLOCK_SIZE,
    HIST_SIZE,
    REDUCE_WIDTH,
    SLICE_LANES,
    reduce_stage2_slot,
    slice_offset,
)
from gbdt.methods.greedy_subsets_searcher.kernel.hist_one_byte import (
    LANE_WIDTH as ONE_BYTE_LANES,
    ONE_BYTE_BLOCK_SIZE,
    one_byte_slice_offset,
)
from gbdt.methods.greedy_subsets_searcher.kernel.hist_2_one_byte_base import (
    LANE_WIDTH as HIST2_LANES,
    hist2_block_size,
    hist2_smem_slots,
)


#: The stride `slice_offset` writes at, as a LITERAL, because that is how the
#: kernel writes it. The gate's whole subject is what happens when a derived
#: quantity stops agreeing with this number.
comptime HALF_BYTE_SLICE_STRIDE = 512


def sabotage_lanes() raises -> Int:
    """The hardware lane width of the CDNA column, READ FROM THE MATRIX
    rather than typed here, so every sabotage arm below is exactly "someone
    re-coupled the logical constant to `column_lane_width`" and not a made-up
    number. If the matrix's AMD row ever changes, the sabotage follows it.
    """
    return column_lane_width(COLUMN_AMD)


# ---------------------------------------------------------------------------
# ARM 1: the half-byte / binary fold stride.
# ---------------------------------------------------------------------------


def stage2_read_set() raises -> List[Bool]:
    """Which slots of the first replica stage 2 actually reads.

    `reduce_stage2_slot(tid, group)` for `tid < 128` and `group < 4`, the
    REAL function from the accumulator, not a copy of it.
    """
    var seen = List[Bool]()
    for _ in range(HALF_BYTE_SLICE_STRIDE):
        seen.append(False)
    for tid in range(128):
        for group in range(4):
            var s = reduce_stage2_slot(tid, group)
            if s < 0 or s >= HALF_BYTE_SLICE_STRIDE:
                raise Error(
                    "sub_byte_layout_gate FAIL: stage 2 reads slot "
                    + String(s)
                    + ", outside the first replica; the layout model and"
                    + " the accumulator no longer agree"
                )
            seen[s] = True
    return seen^


def half_byte_min_replica_coverage(reduce_width: Int) raises -> Int:
    """How many of the private replicas reach stage 2, minimised over the 512
    layout offsets.

    Stage 1 accumulates cell `c` into slot `c % reduce_width` and stage 2
    reads only `stage2_read_set()`. A replica's contribution to layout offset
    `o` survives when its cell lands in a slot stage 2 reads AND that slot
    carries the same within-replica offset. Every replica must survive: the
    answer must be `HIST_SIZE // 512`.
    """
    if reduce_width <= 0:
        raise Error("sub_byte_layout_gate FAIL: reduce_width must be positive")
    var seen = stage2_read_set()
    var replicas = HIST_SIZE // HALF_BYTE_SLICE_STRIDE
    var worst = replicas + 1
    for o in range(HALF_BYTE_SLICE_STRIDE):
        var live = 0
        for r in range(replicas):
            var cell = r * HALF_BYTE_SLICE_STRIDE + o
            var dest = cell % reduce_width
            if dest >= HALF_BYTE_SLICE_STRIDE:
                continue
            if dest % HALF_BYTE_SLICE_STRIDE != o:
                continue
            if not seen[dest]:
                continue
            live += 1
        if live < worst:
            worst = live
    return worst


def check_half_byte_fold_stride() raises:
    """ARM 1. The shipped fold stride must keep every replica; the stride a
    hardware-coupled constant would produce must LOSE some, and this
    function raises if it does not.
    """
    var replicas = HIST_SIZE // HALF_BYTE_SLICE_STRIDE

    # The slice layout itself, from the real function: every group of 32
    # thread indices must own a disjoint replica, and the block must use
    # exactly the scratch it allocates.
    var lo = slice_offset(0)
    var hi = slice_offset(BLOCK_SIZE - 1)
    if lo != 0:
        raise Error(
            "sub_byte_layout_gate FAIL: thread 0's replica does not start at"
            " 0 (got " + String(lo) + ")"
        )
    # THE REPLICA BASE, NOT THE SLOT. `slice_offset` returns
    # `STRIDE * (tid // LANES) + (tid & 24)`, so it ALREADY carries the
    # thread's slot inside its replica. Adding a whole stride to it
    # double-counts that slot and reports the last replica ending at 8216
    # against an 8192 allocation, which is what this gate did on its first
    # run and it was wrong about APPLE, a column this change does not touch.
    # The real kernel gate passes there with zero wrong cells.
    #
    # What must fit is the replica the thread writes into. A thread at slot
    # `o` touches `o, o+32, ..., o+15*32`, so the highest cell it reaches is
    # `base + 24 + 480 = base + 504`, inside the 512 stride.
    var hi_base = (hi // HALF_BYTE_SLICE_STRIDE) * HALF_BYTE_SLICE_STRIDE
    if hi_base + HALF_BYTE_SLICE_STRIDE > HIST_SIZE:
        raise Error(
            "sub_byte_layout_gate FAIL: the last thread's replica ends at "
            + String(hi_base + HALF_BYTE_SLICE_STRIDE)
            + " past the allocated " + String(HIST_SIZE)
        )
    if BLOCK_SIZE // PINNED_REPLICATION_LANES != replicas:
        raise Error(
            "sub_byte_layout_gate FAIL: the block cuts into "
            + String(BLOCK_SIZE // PINNED_REPLICATION_LANES)
            + " logical replica groups but the scratch holds "
            + String(replicas)
        )

    # LIVE ARM.
    var live = half_byte_min_replica_coverage(REDUCE_WIDTH)
    if live != replicas:
        raise Error(
            "sub_byte_layout_gate FAIL: at the shipped REDUCE_WIDTH "
            + String(REDUCE_WIDTH)
            + " only " + String(live) + " of " + String(replicas)
            + " replicas reach stage 2; the fold stride and the slice"
            + " stride have stopped agreeing"
        )

    # SABOTAGE ARM. This is the value the kernel computed before DEVIATION
    # 1947 on a 64-lane FAST column: the fold stride re-coupled to the
    # hardware wave.
    var bad_lanes = sabotage_lanes()
    var bad_width = bad_lanes * (HIST_SIZE // BLOCK_SIZE)
    var sabotaged = half_byte_min_replica_coverage(bad_width)
    if sabotaged == replicas:
        raise Error(
            "sub_byte_layout_gate FAIL: THE SABOTAGE ARM PASSED. A fold"
            " stride of " + String(bad_width)
            + " (the hardware lane width of the "
            + column_name(COLUMN_AMD)
            + " column times the floats per thread) kept all "
            + String(replicas)
            + " replicas, which means this gate can no longer tell a logical"
            + " striping constant from a hardware one and defends nothing"
        )
    print(
        "  half-byte fold stride: shipped "
        + String(REDUCE_WIDTH)
        + " keeps " + String(live) + "/" + String(replicas)
        + " replicas; re-coupled " + String(bad_width)
        + " keeps " + String(sabotaged) + "/" + String(replicas)
        + " (must differ)"
    )


# ---------------------------------------------------------------------------
# ARM 2: the one-byte ladder's replica occupancy.
# ---------------------------------------------------------------------------


def model_one_byte_slice(tid: Int, lanes: Int, inner_bits: Int) -> Int:
    """`TPointHistOneByte::SliceOffset()` with the striping constant made a
    PARAMETER, so the sabotaged value can be evaluated at all.

    Cross-checked against the real `one_byte_slice_offset` in the arm below
    at the shipped constant; a drift there is a failure.
    """
    var blocks = 8 >> inner_bits
    return 1024 * (tid // lanes) + (tid & ((blocks - 1) << (inner_bits + 2)))


def one_byte_replica_collisions(lanes: Int) raises -> Int:
    """How many pairs of threads share a replica AND cannot be told apart by
    the layout.

    Within a replica a thread's slot depends on `threadIdx.x` only through
    its low five bits: `tid & ((blocks - 1) << (inner_bits + 2))` for the
    sub-copy and `(tid + i) & 3` for the feature rotation. So two threads in
    one replica whose low five bits agree write the same slot for the same
    input bin, with a plain float add. Counted at 5 bits, the width with the
    most sub-copies and therefore the most room to separate threads; if it
    collides there it collides everywhere.
    """
    var collisions = 0
    for a in range(ONE_BYTE_BLOCK_SIZE):
        for b in range(a + 1, ONE_BYTE_BLOCK_SIZE):
            if (a & 31) != (b & 31):
                continue
            if model_one_byte_slice(a, lanes, 0) == model_one_byte_slice(
                b, lanes, 0
            ):
                collisions += 1
    return collisions


def check_one_byte_replica_occupancy() raises:
    """ARM 2, with its model cross-checked against the shipped kernel."""

    # CROSS-CHECK: the model must be the real function at the shipped
    # constant, thread for thread.
    for tid in range(ONE_BYTE_BLOCK_SIZE):
        var real = one_byte_slice_offset[5, HIST_SMEM_WARP_PRIVATE_F32](tid)
        var modelled = model_one_byte_slice(tid, ONE_BYTE_LANES, 0)
        if real != modelled:
            raise Error(
                "sub_byte_layout_gate FAIL: the one-byte slice MODEL has"
                " drifted from `one_byte_slice_offset` at tid "
                + String(tid) + " (" + String(modelled) + " vs "
                + String(real)
                + "); the sabotage arm below would be testing a fiction"
            )

    # LIVE ARM.
    var live = one_byte_replica_collisions(ONE_BYTE_LANES)
    if live != 0:
        raise Error(
            "sub_byte_layout_gate FAIL: the shipped one-byte striping"
            " constant " + String(ONE_BYTE_LANES) + " puts "
            + String(live)
            + " thread pairs on the same slot of the same private replica"
            + " with plain float adds"
        )

    # SABOTAGE ARM.
    var bad_lanes = sabotage_lanes()
    var sabotaged = one_byte_replica_collisions(bad_lanes)
    if sabotaged == 0:
        raise Error(
            "sub_byte_layout_gate FAIL: THE SABOTAGE ARM PASSED. Striping the"
            " one-byte replicas by the hardware lane width "
            + String(bad_lanes)
            + " put two logical 32-groups on one private replica and this"
            + " gate did not notice"
        )
    print(
        "  one-byte replica occupancy: shipped "
        + String(ONE_BYTE_LANES) + " gives " + String(live)
        + " colliding pairs; re-coupled " + String(bad_lanes)
        + " gives " + String(sabotaged) + " (must be 0 and non-zero)"
    )


# ---------------------------------------------------------------------------
# ARM 3: the hist_2 shared-Int32 allocation.
# ---------------------------------------------------------------------------


def model_hist2_slots(block: Int, lanes: Int) -> Int:
    """`hist2_smem_slots` under `HIST_SMEM_SHARED2_I32` with the striping
    constant made a parameter."""
    return (block // (2 * lanes)) * 1024


def check_hist2_shared_allocation() raises:
    """ARM 3. The allocation must cover the slices the keying hands out.

    `hist2_slice_offset` keys the shared slice by `tid // 64` -- a LITERAL,
    two logical warps to one slice through Int32 atomics -- so a block of
    `B` threads uses `ceil(B / 64)` slices of 1024 slots. The allocation was
    written as `B // (2 * LANE_WIDTH)`, which is the same number at 32 and
    HALF of it at 64.
    """
    var block = hist2_block_size[HIST_SMEM_SHARED2_I32]()
    var needed = ((block + 63) // 64) * 1024

    # CROSS-CHECK against the shipped function.
    var real = hist2_smem_slots[HIST_SMEM_SHARED2_I32]()
    var modelled = model_hist2_slots(block, HIST2_LANES)
    if real != modelled:
        raise Error(
            "sub_byte_layout_gate FAIL: the hist_2 allocation MODEL has"
            " drifted from `hist2_smem_slots` (" + String(modelled)
            + " vs " + String(real) + ")"
        )

    # LIVE ARM.
    if real < needed:
        raise Error(
            "sub_byte_layout_gate FAIL: the hist_2 shared-Int32 arm allocates "
            + String(real) + " slots for a block of " + String(block)
            + " threads whose slice keying needs " + String(needed)
        )

    # SABOTAGE ARM.
    var bad_lanes = sabotage_lanes()
    var sabotaged = model_hist2_slots(block, bad_lanes)
    if sabotaged >= needed:
        raise Error(
            "sub_byte_layout_gate FAIL: THE SABOTAGE ARM PASSED. Sizing the"
            + " hist_2 shared slices by the hardware lane width "
            + String(bad_lanes) + " allocated " + String(sabotaged)
            + " slots against the " + String(needed)
            + " its `tid // 64` keying hands out, and this gate called that"
            + " sufficient"
        )
    print(
        "  hist_2 shared allocation: needs " + String(needed)
        + " slots, shipped gives " + String(real)
        + ", re-coupled gives " + String(sabotaged)
        + " (shipped must cover, re-coupled must not)"
    )


# ---------------------------------------------------------------------------
# ARM 4: the rows themselves.
# ---------------------------------------------------------------------------


def check_matrix_rows() raises:
    """ARM 4. The three kernel families must be reading the LOGICAL row, and
    the sync row must be the one thing that still follows the hardware."""
    comptime logical = replication_lanes_for[TARGET_COLUMN, False]()
    comptime logical_identical = replication_lanes_for[TARGET_COLUMN, True]()
    if logical != PINNED_REPLICATION_LANES:
        raise Error(
            "sub_byte_layout_gate FAIL: `replication_lanes_for` returned "
            + String(logical) + " under FAST"
        )
    if logical_identical != PINNED_REPLICATION_LANES:
        raise Error(
            "sub_byte_layout_gate FAIL: `replication_lanes_for` returned "
            + String(logical_identical) + " under IDENTICAL"
        )
    if SLICE_LANES != PINNED_REPLICATION_LANES:
        raise Error(
            "sub_byte_layout_gate FAIL: the half-byte template's striping"
            " constant is " + String(SLICE_LANES) + ", not the logical "
            + String(PINNED_REPLICATION_LANES)
        )
    if ONE_BYTE_LANES != PINNED_REPLICATION_LANES:
        raise Error(
            "sub_byte_layout_gate FAIL: the one-byte kernel's striping"
            " constant is " + String(ONE_BYTE_LANES)
        )
    if HIST2_LANES != PINNED_REPLICATION_LANES:
        raise Error(
            "sub_byte_layout_gate FAIL: the hist_2 kernel's striping constant"
            " is " + String(HIST2_LANES)
        )

    # THE SYNC ROW IS THE ONE THAT STILL FOLLOWS THE HARDWARE, and this arm
    # is what keeps somebody from "simplifying" it to match the layout row.
    comptime amd_sync = sub_byte_lane_sync_for[COLUMN_AMD]()
    if amd_sync == SYNC_LANE:
        raise Error(
            "sub_byte_layout_gate FAIL: the "
            + column_name(COLUMN_AMD) + " column claims a 32-lane write-turn"
            " sync on a " + String(column_lane_width(COLUMN_AMD))
            + "-wide wave. The LAYOUT is width-independent; the SYNC is not,"
            + " and DEVIATION 1947 says so in as many words"
        )
    print(
        "  matrix rows: layout constant "
        + String(SLICE_LANES) + " on every family, "
        + column_name(TARGET_COLUMN) + " builds with sync row "
        + String(sub_byte_lane_sync_for[TARGET_COLUMN]())
        + " (0 = block barrier, 1 = lane sync)"
    )


def check_sub_byte_layout() raises:
    print(
        "sub_byte_layout_gate: column "
        + column_name(TARGET_COLUMN)
        + ", hardware lanes "
        + String(column_lane_width(TARGET_COLUMN))
        + ", logical replica lanes "
        + String(SLICE_LANES)
    )
    check_matrix_rows()
    check_half_byte_fold_stride()
    check_one_byte_replica_occupancy()
    check_hist2_shared_allocation()
    print(
        "sub_byte_layout_gate OK: four arms live, three sabotage arms failed"
        " as required (DEVIATION 1947)"
    )


def main() raises:
    check_sub_byte_layout()
