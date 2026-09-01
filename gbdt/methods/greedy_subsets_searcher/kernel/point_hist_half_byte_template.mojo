# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The conflict-free shared-memory histogram accumulator.

PORT OF `catboost/cuda/methods/greedy_subsets_searcher/kernel/
point_hist_half_byte_template.cuh` at CatBoost `54a8143a`.
Transliterated. Do not improve.

**This is the file the whole experiment is about.** It is how CatBoost
accumulates a histogram on the GPU with NO ATOMICS in the inner loop, and it
is the thing mojotrees does not do.

The mechanism, in their words rearranged:

- The threadgroup holds `BlockSize * 16` floats of scratch.
- `SliceOffset()` = `512 * (tid / 32) + (tid & 24)` gives every 32-lane warp
  its own 512-float copy, and within a warp gives each group of 8 lanes its
  own sub-copy at offset 0, 8, 16 or 24.
- `AddPoint` loops `i` over 0..7 and has lane `tid` handle feature
  `f = (tid + i) & 7`. Because the 8 lanes of a tile hold 8 distinct values
  of `tid & 7`, the rotation is a permutation: within one iteration the 8
  lanes touch 8 DISTINCT features, hence 8 distinct slots. No two lanes
  collide, so the update is a plain `Histogram[bin] += t`.
- The slot is `bin = ((ci >> (28 - 4*f)) & 15) << 5 | f`. The `<< 5` spaces
  consecutive bins 32 floats apart so that the 8 lanes' writes fall in
  different banks.

`ci` is one `UInt32` of the compressed index holding EIGHT features at 4 bits
each, so one load feeds eight histogram updates. That is the read-density
win: mojotrees issues eight one-byte loads for the same work.

DEVIATION (PORTING.md 1): `BLOCK_SIZE` is 512 rather than their 768 because
Apple caps threadgroup memory at 32 KB and this accumulator wants 16 floats
per thread, so 32,768 / (16 * 4) = 512 threads is the largest block that
fits. Their `tiled_partition<8>::sync()` becomes `turn_sync()`, which is
`syncwarp()` (32 lanes, a SUPERSET of the 8 theirs orders, and the narrowest
sync Mojo 1.0 exposes) on a column whose hardware wave is 32, and a
threadgroup `barrier()` on any other width -- see `sub_byte_lane_sync_for`
and `add_half_byte_point`'s deviation block. Warp SHUFFLES are what Mojo
lacks, not warp SYNC.

DEVIATION 1947, 2026-09-01: this accumulator's `512 * (tid // 32)` is a
LOGICAL partition of thread indices, not a statement about wave width, so it
is valid on a 64-wide wavefront and this family is no longer excluded from
64-lane FAST columns. The one constant that WAS coupled to the hardware,
`SLICE_LANES`, now reads `replication_lanes_for`. Eligible to run there; not
yet run there.
"""

from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.memory import AddressSpace
from checks.kernel_matrix import (
    replication_lanes_for,
    K_HIST_HALF_BYTE,
    TARGET_COLUMN,
    block_size_for,
    hist_floats_per_thread_for,
)
from gbdt.methods.greedy_subsets_searcher.kernel.lane_sync import turn_sync
from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_FAST,
    NUMERIC_IDENTICAL,
    ftz,
)


#: READ FROM THE MATRIX, not restated here. `checks/kernel_matrix.mojo`
#: owns every knob that is a BUDGET, and a kernel that hardcodes one makes
#: the table decoration. CatBoost uses 768; Apple's 32 KB ceiling over 16
#: floats per thread yields 512, which is exactly 32,768 bytes.
comptime BLOCK_SIZE = block_size_for[K_HIST_HALF_BYTE, TARGET_COLUMN]()


#: The mode this build compiles against. See `checks/numerics.mojo`. FAST
#: is the default, and under FAST the flush is CatBoost's own float
#: `atomicAdd` (`hist_half_byte.cu:45-51`) on every vendor Mojo builds for.
#: The fixed-point Int32 accumulator is what IDENTICAL selects, and it is
#: dead code in this build.
#:
#: "No vendor forces the deterministic row" stood here and is deleted rather
#: than annotated (2026-08-21): the `qualcomm` column and the portable
#: baseline have no CORE float atomic -- it arrives through an optional
#: extension -- so `deterministic_flush_for` returns True for them in BOTH
#: modes. That row is a vendor override now, not only a mode's choice.
#:
#: WORTH KNOWING WHERE THIS FILE SITS IN THE IDENTITY ARGUMENT. The flush is
#: fixed point under IDENTICAL, but the SHARED-MEMORY accumulation in this
#: family is float (`smem[slot] = smem[slot] + stat`, below), unlike the
#: hist_2 family's shared Int32. So for THIS family the block size is a
#: NUMERIC row -- it sets how many private slices exist, hence which floats
#: add to which -- and that is what the identity floor's 32 KB is protecting.
#: On the hist_2 path the same knob is pure scheduling, because integer
#: addition is associative. Two families, two answers, one table.
comptime BUILD_MODE = GLOBAL_NUMERIC_MODE

#: The LOGICAL width of one private replica group. READ FROM THE MATRIX, not
#: pinned here, and read from the row that means what this file needs.
#:
#: THIS USED TO BE `lane_width_for`, THE HARDWARE ROW, AND THAT WAS THE WHOLE
#: OF DEVIATIONS 1906 AND 1910 (corrected 2026-09-01, DEVIATION 1947). The
#: comment that stood here was right that a literal 32 in four files is a
#: matrix row bypassed, and wrong about which row. `lane_width_for` answers
#: "how many lanes retire together", which under FAST is 64 on CDNA; this
#: file's 512-float slices and `& 24` sub-copies ask something else entirely,
#: namely "how many consecutive thread indices share one private replica",
#: which is 32 BY OUR CHOICE on every device. Feeding the hardware answer to
#: the logical question moved `REDUCE_WIDTH` to 1024 while `slice_offset`'s
#: literal 512 stayed put, stage 1 folded cells belonging to different bins,
#: and a comptime assert turned that into a refusal for the whole family.
#:
#: The identical path had the correct spelling all along -- `lane_width_for`
#: pins 32 under IDENTICAL on every column, which is why both sub-byte
#: families already elaborate on AMD in that mode. `replication_lanes_for` is
#: that spelling made unconditional, so the layout is valid at any hardware
#: width and there is nothing left to refuse.
#:
#: NOT A LOCKSTEP CLAIM. Nothing below assumes these 32 threads execute
#: together; the write-turn sync is a separate row (`sub_byte_lane_sync_for`,
#: consumed through `lane_sync.turn_sync`) precisely because that assumption
#: does NOT travel to a 64-wide wave.
comptime SLICE_LANES = replication_lanes_for[
    TARGET_COLUMN, BUILD_MODE == NUMERIC_IDENTICAL
]()

#: The old name, kept because `probe_main.mojo` imports it and asserts on it.
#: RENAMED HERE, NOT DELETED, and the difference matters: `LANE_WIDTH` is what
#: this constant was called while it meant two things at once, and the rename
#: is the correction. `probe_main.mojo:278` still compares it against
#: `lane_width_for` -- the HARDWARE row -- so on a 64-lane column that probe
#: will raise "the half-byte kernel is NOT reading the matrix" while the
#: kernel is reading the right row and the probe is reading the wrong one.
#: That file is one line from correct (`lane_width_for` ->
#: `replication_lanes_for`) and it is OWED; it is outside this change's
#: ownership, so the alias keeps every column COMPILING and the AMD probe run
#: is where the stale comparison surfaces. Its two other lane asserts --
#: `REDUCE_WIDTH == LANE_WIDTH * 16` and `LANE_WIDTH == 32` -- now hold on
#: EVERY column for the first time, which is the point of DEVIATION 1947.
comptime LANE_WIDTH = SLICE_LANES

#: `Reduce()`'s stage-1 width (`point_hist_half_byte_template.cuh:115-134`).
#:
#: THE DERIVATION, because the old one was a coincidence. Their literal 512
#: is NOT "512 because BlockSize is 768". It is the size of ONE WARP'S
#: PRIVATE REPLICA, established by `SliceOffset()` at
#: `point_hist_half_byte_template.cuh:48-52`:
#:
#:     const int warpOffset = 512 * (threadIdx.x / 32);
#:
#: i.e. `SLICE_LANES * 16`, threads per replica group times floats per
#: thread. Folding the scratch at exactly that stride is what sums a bin
#: ACROSS the replicas'
#: private copies while leaving the 512-float layout of a single copy
#: standing for stage 2 to un-scramble. Any other width sums the wrong
#: cells, so this is a NUMERIC quantity and not a tunable one.
#:
#: It used to be read from `checks/kernel_matrix.mojo`, whose
#: `reduce_width_for` returns the BLOCK SIZE under `NUMERIC_FAST`. That is
#: 512 on Apple by accident and 768 on CatBoost's own configuration, where
#: it would be wrong. The block size does not belong in this quantity, so
#: the matrix is no longer consulted for it.
#:
#: Stage 1 strides `slot_i += BLOCK_SIZE` while `slot_i < REDUCE_WIDTH`, and
#: it carries barriers, so BLOCK_SIZE must divide or equal REDUCE_WIDTH for
#: the trip count to stay uniform. At 512 and 512 every thread makes exactly
#: one trip.
#:
#: AND IT IS 512 BY CONSTRUCTION NOW, on every column and in both modes,
#: because `SLICE_LANES` is the logical 32 the literals below are written
#: against rather than the hardware width. The assert in `slice_offset`
#: states it; `checks/kernel_matrix.mojo`'s `replication_lanes_for` owns it.
comptime REDUCE_WIDTH = SLICE_LANES * hist_floats_per_thread_for[
    K_HIST_HALF_BYTE
]()

#: `GetHistSize()`, `BlockSize * 16`, with the 16 from the matrix.
comptime HIST_SIZE = BLOCK_SIZE * hist_floats_per_thread_for[
    K_HIST_HALF_BYTE
]()


def slice_offset(tid: Int) -> Int:
    """`TPointHistHalfByteBase::SliceOffset()`, copied exactly.

        const int warpOffset = 512 * (threadIdx.x / 32);
        const int innerHistStart = threadIdx.x & 24;
        return warpOffset + innerHistStart;

    `& 24` keeps bits 3 and 4, giving 0, 8, 16 or 24: four sub-copies inside
    each warp's 512-float region. Do not "simplify" it to `& 31`; the low
    three bits are the lane's position inside the 8-lane tile and are carried
    by `f` instead.

    THE 512 AND THE 32 ARE THEIRS, TRANSCRIBED, AND THEY ARE LOGICAL
    ----------------------------------------------------------------
    512 is 16 bins of 32 slots, and the 32 is 8 features times the 4
    sub-copies a replica group is cut into. `add_point_slot` spaces bins by
    `<< 5`, so one replica has room for exactly four 8-lane tiles.

    THE PARAGRAPH THAT STOOD HERE SAID THIS GEOMETRY DOES NOT GENERALISE TO
    A 64-WIDE WAVEFRONT, AND IT IS RETRACTED (2026-09-01, DEVIATION 1947).
    It read: a 64-lane wavefront would need eight sub-copies and two of them
    would collide on every bin. That is a claim about lanes retiring
    together, and this arithmetic is not about that. `tid // 32` and
    `tid & 24` partition THREAD INDICES: every 32 consecutive indices get a
    disjoint 512-float replica, every 8 get a disjoint eighth of the 32
    slots a bin spans, and 512 threads therefore use exactly the
    `BLOCK_SIZE * 16` floats the accumulator allocates. Nothing in that
    changes when the hardware runs 64 of those threads at once; a 64-wide
    wave simply covers two of the logical groups, on two disjoint replicas.

    WHAT WAS ACTUALLY BROKEN was one constant in the wrong row.
    `REDUCE_WIDTH` was derived from the HARDWARE width, so on a FAST CDNA
    column it became 1024 while these literals stayed 512 and 32. Stage 1
    then folded replicas 0,2,4.. into the first 512 slots and 1,3,5.. into a
    second 512 that stage 2 never reads, dropping half of every histogram.
    Reading the LOGICAL row (`SLICE_LANES`) puts `REDUCE_WIDTH` back at 512
    on every column, which is what the assert below now states.

    So the family is no longer excluded anywhere: `greedy_sub_byte_excluded_
    for` returns False in every cell and the binary and half-byte launch
    sites dispatch on a 64-lane column like any other. What did NOT survive
    the retraction is the write-turn sync, which is a real lockstep question
    and is answered one row over -- see `add_half_byte_point`.

    THE ASSERT STAYS AND IT STILL BITES. It is no longer a refusal of AMD;
    it is the invariant that `SLICE_LANES` was not re-coupled to the
    hardware width, and it fires at COMPILE time in that case. The runnable
    arm that proves it can fire is
    `gbdt/methods/greedy_subsets_searcher/kernel/sub_byte_layout_gate.mojo`.
    """
    comptime assert SLICE_LANES == 32, (
        "the half-byte accumulator's slice layout is written against a"
        " LOGICAL 32-thread replica group (`point_hist_half_byte_template"
        ".cuh:48-52` plus the `<< 5` bin spacing); `SLICE_LANES` must read"
        " `replication_lanes_for`, never the hardware lane width"
    )
    comptime assert REDUCE_WIDTH == 512, (
        "stage 1 must fold at the replica stride `slice_offset` writes;"
        " a REDUCE_WIDTH above 512 folds replicas into slots stage 2 never"
        " reads and silently drops that share of every histogram"
    )
    return 512 * (tid // 32) + (tid & 24)


def add_point_slot(ci: UInt32, tid: Int, i: Int) -> Int:
    """The slot `AddPoint` writes on iteration `i`, as pure arithmetic.

        const int f = (threadIdx.x + i) & 7;
        int bin = (ci >> (28 - 4 * f)) & 15;
        bin <<= 5;
        bin += f;

    CatBoost keeps this inside `AddPoint`. Here the ARITHMETIC is factored
    out so that `AddPoint` (below) and `AddPointsImpl`'s inner batch loop,
    which is inlined at each call site, can share one copy of it. Same
    operations, same order.

    Why the slot is collision-free: the 8 lanes of a tile hold 8 distinct
    values of `tid & 7`, so `(tid + i) & 7` is a permutation of them and the
    8 lanes touch 8 distinct features on every iteration. `bin << 5` then
    spaces consecutive bins 32 floats apart so those 8 writes fall in
    different banks.
    """
    var f = (tid + i) & 7
    var bin = Int((ci >> UInt32(28 - 4 * f)) & UInt32(15))
    bin <<= 5
    bin += f
    return bin


def add_half_byte_point(
    ci: UInt32,
    stat: Float32,
    tid: Int,
    slice_base: Int,
    smem: UnsafePointer[
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
        origin=MutUntrackedOrigin,
    ],
):
    """`TPointHistHalfByteBase::AddPoint`, ONE point, as a callable
    (`point_hist_half_byte_template.cuh:65-77`):

        thread_block_tile<8> addToHistTile = tiled_partition<8>(...);
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const int f = (threadIdx.x + i) & 7;
            int bin = (ci >> (28 - 4 * f)) & 15;
            bin <<= 5;
            bin += f;
            Histogram[bin] += t;
            addToHistTile.sync();
        }

    Their `AddPoint` is a method, so `AlignMemoryAccess`
    (`compute_hist_loop_one_stat.cuh:81`, `:129`) calls the SAME code the
    main loop does. Ours used to be inlined in the loop only, which is fine
    right up until the head/tail peel needs it too; duplicating a
    sync-carrying accumulation in four places is how the four copies drift.

    DEVIATION BLOCK
    ---------------
    THEIRS: `addToHistTile.sync()` on a `tiled_partition<8>`.
    OURS:   `turn_sync()`, which is `syncwarp()` (32 lanes) on a column whose
            hardware wave IS 32 and a threadgroup `barrier()` on every other.
    WHY:    Mojo 1.0 exposes no 8-lane tile. On a 32-wide machine a warp sync
            is a SUPERSET of the 8 lanes theirs orders, so the ordering
            guarantee is strictly stronger and the result is unchanged; that
            is the swap that bought 21% and it is untouched on Apple,
            NVIDIA and RDNA.

            WIDENED FURTHER OFF 32-WIDE HARDWARE (2026-09-01, DEVIATION
            1947). What this sync must do is make lane a's store at
            iteration `i` visible to lane b before iteration `i+1` reads the
            same slot. On a 64-wide wavefront `syncwarp` still names a
            superset of the tile ARITHMETICALLY, and every call here is
            unconditional under a block-uniform trip count, so participation
            is not the hazard. The hazard is what the toolchain EMITS: if
            `syncwarp` folds to nothing on the assumption that a wavefront is
            already in lockstep, the compiler may keep a slice cell in a
            register across the iteration boundary and the tile's other
            lanes never see the store. That cannot be settled by reading
            anything in this tree, so the matrix takes the conservative side
            and `sub_byte_lane_sync_for` hands a real threadgroup barrier to
            every column that is not exactly 32 lanes wide. It reorders
            nothing: no slot receives adds from two tiles, so per slot the
            order is program order either way.

            THE PRICE, on `amd` only among buildable columns: eight
            threadgroup barriers per point instead of eight warp syncs. That
            is the row to flip first if the AMD sub-byte arm is slow, and
            flipping it needs someone to establish what `syncwarp` emits on
            CDNA, not a guess.
    """

    @parameter
    for i in range(8):
        var slot = slice_base + add_point_slot(ci, tid, i)
        # IDENTITY_PATHS ROW 10: this family accumulates in FLOAT in both
        # modes (unlike hist_2's Int32), and a running cell can pass
        # through the denormal range under cancellation -- flushed on
        # Metal's hardware, kept on CUDA's default -- so each stored
        # intermediate goes through `ftz`. Comptime no-op under FAST.
        smem[slot] = ftz(smem[slot] + stat)
        turn_sync()


def reduce_stage2_slot(tid: Int, group: Int) -> Int:
    """`Reduce()` stage 2's read offset: `32 * fold + featureId + 8 * group`
    for `fold = (tid >> 3) & 15` and `featureId = tid & 7`."""
    var fold = (tid >> 3) & 15
    var feature_id = tid & 7
    return 32 * fold + feature_id + 8 * group
