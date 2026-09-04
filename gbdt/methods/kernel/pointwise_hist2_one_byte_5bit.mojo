# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The 5-bit pointwise accumulator: `TPointHist<0, 0, BlockSize>`.

PORT OF `catboost/cuda/methods/kernel/pointwise_hist2_one_byte_5bit.cu` at
CatBoost `54a8143a`. Transliterated. Do not improve.

The first `PointHist2` implementor. It handles FOUR features at a time, each
occupying one byte of a `UInt32` compressed-index word, with up to 32 bins
each plus 32 as the "absent" marker.

WHAT IT SHARES WITH THE OTHER FAMILY, AND WHERE THEY PART
---------------------------------------------------------
`SliceOffset()` here (`:52-58`) is character-identical to the greedy-subsets
family's (`hist_2_one_byte_5bit.cu:25-31`), and so is the bin arithmetic:

    f      = ((2 * i + threadIdx.x) & 6)          i in 0..3
    bin    = (ci >> (24 - (f << 2))) & 255
    offset = f + 32 * (bin & 31)
    pass   = bin != 32

so a thread's four iterations sweep four of the eight (feature, parity)
slots, and the whole 1024-slot warp slice is 32 bins x (4 inner copies x 8
slots). This repository already ported that arithmetic once, in
`gbdt/methods/greedy_subsets_searcher/kernel/hist_2_one_byte_5bit.mojo`, and
it is written again here rather than imported because CatBoost has two files
and the port mirrors CatBoost's tree. It is three lines; it is not the
hundred-line loop `archive/reference/PORTING.md` 13 is about.

**They part at two places and both matter:**

1. THE STAT PARITY IS OPPOSITE. Theirs here is

       stat1 = flag ? t : w;   stat2 = flag ? w : t;

   and the greedy family's is `stat1 = flag ? s2 : s1`. Following the writes
   through, this family lands WEIGHT in the even slot and TARGET in the odd
   one, unconditionally. Reading the other family's convention onto this one
   swaps every gradient with its weight, which does not change any total and
   destroys every split.

2. THE REDUCE OUTPUT IS STAT-MINOR. This family finishes at
   `Buffer[2 * (maxFoldCount * f + fold) + w]` (`:243`); the greedy family
   finishes at `Histogram[maxFoldCount * 4 * isSecondStat + maxFoldCount * f
   + fold]`, stat-major. Same cells, transposed, and a transposition moves no
   total -- which is exactly the failure
   [[uniform-test-data-hides-permutation]] describes and why the gate for
   this file compares per (feature, fold, stat) and never sums.

DEVIATION (archive/reference/PORTING.md 1, same arithmetic as the other family): CatBoost runs
this at `BlockSize = 384` (`pointwise_hist2_one_byte_templ.cuh:236`), so
`384 * 32` floats is 49,152 bytes against Apple's 32,768. The matrix
resolves the block to the largest that fits the budget (256 on 32 KB) and
leaves their per-warp slice arithmetic intact: 8 warps x 1024 floats. The
BLOCK shrinks, the LAYOUT does not, and their reduce already caps
participation at `threadIdx.x < 256`. The geometry is PER ROUTE in name
(`PW_HIST2_BLOCK` for the 8-bit accumulator's launches,
`PW_HIST2_FLOAT_BLOCK` for the float accumulators', whose slice offsets do
not wrap) even though both currently resolve to the same value: doubling
the fixed route's block was measured a no-op and reverted -- the negative
is recorded on `pw_hist2_block_size_for`.

DEVIATION (archive/reference/PORTING.md 11 and 92): their `thread_block_tile<8>::sync()`
(`:79`, `:99`, `:178`) becomes a threadgroup `barrier()`, which is the only
sync Mojo exposes. That is what forces the uniform-iteration path in
`compute_point_hist2_loop.mojo`. `archive/reference/PORTING.md` 92 records that the failure
this prevents does not reproduce on this device, and that the path is kept on
the specification rather than on a measurement.

NOT PORTED FROM THIS FILE: the `TUnrollsTrait<0, FourElements>`
specialization at `:250-255` returns `Outer() == 1`, which is the value the
generic template already gives; it exists in CatBoost to satisfy an explicit
instantiation and carries no behaviour.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.kernel_matrix import (
    TARGET_COLUMN,
    lane_width_for,
    pointwise_one_byte_fixed_for,
    pw_hist2_block_size_for,
    pw_hist2_smem_floats_for,
)
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

from gbdt.methods.kernel.compute_point_hist2_loop import PointHist2


#: Whether this build routes every one-byte width through the 8-bit
#: fixed-point accumulator; the block and scratch rows key on it.
comptime PW_ONE_BYTE_FIXED = pointwise_one_byte_fixed_for[
    TARGET_COLUMN, GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
]()

#: The resolved block: CatBoost's 384-capped `block_size_for` (256 under
#: Apple's 32 KB) on BOTH routes -- doubling it under the fixed route was
#: measured a no-op; the negative is recorded on `pw_hist2_block_size_for`.
comptime PW_HIST2_BLOCK = pw_hist2_block_size_for[
    TARGET_COLUMN, PW_ONE_BYTE_FIXED
]()

#: `const int HIST_SIZE = 32 * BLOCK_SIZE;` (`:62`).
comptime PW_HIST2_SMEM_FLOATS = pw_hist2_smem_floats_for[
    TARGET_COLUMN, PW_ONE_BYTE_FIXED
]()

#: The geometry the FLOAT turn-taking accumulators (5/6/7 bit) are built
#: for: their dispatch's resolution, ALWAYS. These accumulators only launch
#: under their dispatch, where `PW_HIST2_BLOCK` equals this by definition;
#: under the fixed 8-bit route they are dead code kept compiling for the
#: columns that do launch them, and their slice offsets DO NOT WRAP (only
#: the 8-bit accumulator's does), so they must NEVER be built against a
#: block their own dispatch did not resolve -- a wider fixed-route block
#: would run their upper warps off the scratch. Their per-cell check
#: launches them at THIS block for the same reason.
comptime PW_HIST2_FLOAT_BLOCK = pw_hist2_block_size_for[
    TARGET_COLUMN, False
]()
comptime PW_HIST2_FLOAT_SMEM_FLOATS = 32 * PW_HIST2_FLOAT_BLOCK

#: `const int warpHistSize = 1024;` (`:210`). Not a tunable: it is 32 lanes
#: times the 32 floats per thread this accumulator takes, and their reduce
#: indexes it directly at `:233`.
comptime PW_WARP_HIST_SIZE = 1024

comptime PW_LANE_WIDTH = lane_width_for[
    TARGET_COLUMN, GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
]()


@always_inline
def pw_hist2_slice_offset_5(tid: Int) -> Int:
    """`TPointHist<0,0,BLOCK_SIZE>::SliceOffset()` (`:52-58`).

        const int warpId = (threadIdx.x / 32);
        const int warpOffset = 1024 * warpId;
        const int blocks = 4;
        const int innerHistStart = (threadIdx.x & ((blocks - 1) << 3));
        return warpOffset + innerHistStart;
    """
    var warp_offset = PW_WARP_HIST_SIZE * (tid // 32)
    var inner_hist_start = tid & ((4 - 1) << 3)
    return warp_offset + inner_hist_start


struct PointHist5[origin: MutOrigin](PointHist2):
    """`TPointHist<0, 0, BLOCK_SIZE>` (`:50-247`).

    Construct AFTER the caller has a zeroed shared buffer of
    `PW_HIST2_FLOAT_SMEM_FLOATS` floats; the constructor zeroes it exactly
    as theirs does (`:60-70`) and takes the trailing `__syncthreads()`.
    Built against the FLOAT geometry, always: this accumulator only
    launches under their dispatch and its slice offset does not wrap.
    """

    var base: MutPointer[
        Float32, Self.origin, address_space = AddressSpace.SHARED
    ]
    var buffer_offset: Int

    def __init__(
        out self,
        buff: MutPointer[
            Float32, Self.origin, address_space = AddressSpace.SHARED
        ],
    ):
        """Their constructor (`:60-70`), including the zero fill."""
        var tid = Int(thread_idx.x)
        var i = tid
        while i < PW_HIST2_FLOAT_SMEM_FLOATS:
            buff.unsafe_store(i, 0.0)
            i += PW_HIST2_FLOAT_BLOCK
        self.base = buff
        self.buffer_offset = pw_hist2_slice_offset_5(tid)
        barrier()

    @always_inline
    def _add(mut self, slot: Int, val: Float32):
        """Their `Add(float val, float* dst)` (`:72-74`), which is `+=`.

        It is a named method in CatBoost because the 6- and 7-bit
        accumulators override it to do the same thing at a different
        granularity; kept named here for the same reason.
        """
        var at = self.buffer_offset + slot
        self.base.unsafe_store(at, self.base.unsafe_load(at) + val)

    def add_point(
        mut self, ci: UInt32, t: Float32, w: Float32, row: UInt32
    ):
        """`AddPoint` (`:76-103`), copied.

        THE FLAG SPLITS THE BLOCK INTO TWO HALVES THAT WRITE THE SAME PAIR
        OF SLOTS IN OPPOSITE ORDER, which is what lets 8 threads share one
        inner copy without an atomic: at any instant, the even threads are
        touching the even slot and the odd threads the odd one. The sync
        between the two writes is what holds that apart, and it is theirs.
        """
        # `row` is unused here: this accumulator adds floats into private
        # or turn-taken slots and needs no atomic. See `PointHist2`.
        _ = row
        var tid = Int(thread_idx.x)
        var flag = (tid & 1) != 0

        var stat1 = t if flag else w
        var stat2 = w if flag else t

        comptime for i in range(4):
            var f = (2 * i + tid) & 6
            var bin = Int((ci >> UInt32(24 - (f << 2))) & 255)
            var passes = bin != 32
            var offset = f + 32 * (bin & 31)
            var offset1 = offset + (1 if flag else 0)
            var add1 = stat1 if passes else Float32(0.0)
            var offset2 = offset + (0 if flag else 1)
            var add2 = stat2 if passes else Float32(0.0)

            barrier()
            self._add(offset1, add1)
            barrier()
            self._add(offset2, add2)

    def add_point_2(
        mut self,
        ci: SIMD[DType.uint32, 2],
        t: SIMD[DType.float32, 2],
        w: SIMD[DType.float32, 2],
        rows: SIMD[DType.uint32, 2],
    ):
        """`AddPoint2` (`:105-143`).

        DEVIATION, and it is a deliberate simplification of THEIR code, not
        of the arithmetic. Theirs writes the two points' `stat1` between one
        pair of syncs and their `stat2` between the next, so a 2-wide load
        costs two syncs instead of four. Ours delegates to `add_point`
        twice, which costs four.

        Why: `PointHist2`'s contract says `add_point_2` must be `add_point`
        twice in (x, y) order, and the whole rung-2 identity depends on
        vector width being a LOAD choice and never a numeric one. Their
        fused form adds the same values in the same per-slot order, so it is
        numerically identical -- but proving that for every accumulator is a
        per-file obligation, and the version that cannot be wrong is the one
        that goes in first. Priced, not free: two extra threadgroup barriers
        per 2-wide point. Revisit with a measurement, not an argument.
        """
        self.add_point(ci[0], t[0], w[0], rows[0])
        self.add_point(ci[1], t[1], w[1], rows[1])

    def add_point_4(
        mut self,
        ci: SIMD[DType.uint32, 4],
        t: SIMD[DType.float32, 4],
        w: SIMD[DType.float32, 4],
        rows: SIMD[DType.uint32, 4],
    ):
        """`AddPoint4` (`:145-201`). Same deviation as `add_point_2`.

        Their comment on this one is worth carrying over verbatim, because
        it is a warning about what happens to anyone who edits it:

            "don't change anything without performance tests, nvcc is so
             awesome, that little change of code could slow everything by
             5-10%"
        """
        self.add_point(ci[0], t[0], w[0], rows[0])
        self.add_point(ci[1], t[1], w[1], rows[1])
        self.add_point(ci[2], t[2], w[2], rows[2])
        self.add_point(ci[3], t[3], w[3], rows[3])

    def reduce(mut self):
        """`Reduce` (`:205-246`), copied, in their two stages.

        STAGE 1 (`:209-221`) folds all `32 * BLOCK_SIZE` floats down to the
        1024 slots at `Buffer[1024 + start]`, summing across the per-warp
        slices. It reads a region it also writes, and that is SAFE for a
        reason worth stating: thread `t` writes `1024 + start` only for its
        own `start < 1024`, and the only reader of that address is the same
        thread on the same `start`, which reads it before writing. No other
        `start'` in `[0, 1024)` satisfies `start' + k * 1024 == 1024 +
        start`.

        STAGE 2 (`:223-245`) sums the 4 inner copies at stride 8 and lands
        the result at `Buffer[2 * (32 * f + fold) + w]` -- feature-major,
        then fold, then STAT PARITY, which is the stat-MINOR layout the
        module docstring warns about and which
        `checks/pointwise_hist2_5bit_check.mojo` gates per cell.
        """
        var tid = Int(thread_idx.x)
        # `Buffer -= SliceOffset()` (`:206`): back to the block base
        self.buffer_offset = 0
        barrier()

        var start = tid
        while start < PW_WARP_HIST_SIZE:
            var acc = Float32(0.0)
            var i = start
            while i < PW_HIST2_FLOAT_SMEM_FLOATS:
                acc += self.base.unsafe_load(i)
                i += PW_WARP_HIST_SIZE
            self.base.unsafe_store(PW_WARP_HIST_SIZE + start, acc)
            start += PW_HIST2_FLOAT_BLOCK

        barrier()

        if tid < 256:
            var w = tid & 1
            var f = tid // 64
            var fold = (tid >> 1) & 31
            comptime MAX_FOLD_COUNT = 32
            comptime INNER_HIST_COUNT = 4
            var src = PW_WARP_HIST_SIZE + 32 * fold + 2 * f + w
            var acc = Float32(0.0)
            comptime for in_warp_hist in range(INNER_HIST_COUNT):
                acc += self.base.unsafe_load(src + (in_warp_hist << 3))
            self.base.unsafe_store(
                2 * (MAX_FOLD_COUNT * f + fold) + w, acc
            )
        barrier()
