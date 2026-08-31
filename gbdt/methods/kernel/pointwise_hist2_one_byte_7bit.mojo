# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The 7-bit pointwise accumulator: `TPointHist<0, 2, BlockSize>`.

PORT OF `catboost/cuda/methods/kernel/pointwise_hist2_one_byte_7bit.cu` at
CatBoost `54a8143a`. Transliterated. Do not improve.

Up to 128 bins per feature, 128 as the absent marker, and **the whole 1024
slots of a warp slice go to ONE copy.** This is the end of the progression:

    5-bit    32 bins   4 inner copies    8 threads share one    2 syncs
    6-bit    64 bins   2 inner copies   16 threads share one    4 syncs
    7-bit   128 bins   1 inner copy     32 threads share one    8 syncs

`SliceOffset()` is therefore just the warp offset -- no `innerHistStart` term
at all -- and the collision handling goes from a single flag bit to a
four-way turn-taking loop:

    writeTime = (threadIdx.x >> 3) & 3
    for (k = 0; k < 4; ++k) { if (k == writeTime) Buffer[offset] += val;
                              syncTile.sync(); }

which is bits 3 and 4 of the thread id naming one of four groups of eight,
and eight threads never collide because their `(f, flag)` slots are all
distinct. Two of those loops per iteration, four iterations: **32 syncs per
point**, against the 5-bit file's 8.

The slot map is the simplest of the three:

    offset = f + 8 * (bin & 127)

Same two family conventions as the 5-bit file -- WEIGHT even, TARGET odd, and
a stat-MINOR reduce -- and they are still the ones that change no total.

DEVIATION (PORTING.md 11 and 92): `thread_block_tile<32>::sync()` becomes a
threadgroup `barrier()`. At 32 lanes their tile is a full warp, so this is
the case where the substitution costs most in absolute terms and least in
meaning: it widens a warp sync to a block sync 32 times per point.
"""

from std.gpu import thread_idx
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from gbdt.methods.kernel.compute_point_hist2_loop import PointHist2
from gbdt.methods.kernel.pointwise_hist2_one_byte_5bit import (
    PW_HIST2_FLOAT_BLOCK,
    PW_HIST2_FLOAT_SMEM_FLOATS,
    PW_WARP_HIST_SIZE,
)


@always_inline
def pw_hist2_slice_offset_7(tid: Int) -> Int:
    """`TPointHist<0,2,BLOCK_SIZE>::SliceOffset()`: the warp offset alone."""
    return PW_WARP_HIST_SIZE * (tid // 32)


struct PointHist7[origin: MutOrigin](PointHist2):
    """`TPointHist<0, 2, BLOCK_SIZE>`."""

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
        var tid = Int(thread_idx.x)
        var i = tid
        while i < PW_HIST2_FLOAT_SMEM_FLOATS:
            buff.unsafe_store(i, 0.0)
            i += PW_HIST2_FLOAT_BLOCK
        self.base = buff
        self.buffer_offset = pw_hist2_slice_offset_7(tid)
        barrier()

    @always_inline
    def _add(mut self, slot: Int, val: Float32):
        var at = self.buffer_offset + slot
        self.base.unsafe_store(at, self.base.unsafe_load(at) + val)

    def add_point(
        mut self, ci: UInt32, t: Float32, w: Float32, row: UInt32
    ):
        """`AddPoint`, copied, turn-taking loops included.

        THE SYNC IS AFTER THE WRITE AND INSIDE THE LOOP, so all four turns
        are separated including the last one. Hoisting it out -- which looks
        like an obvious saving of one sync -- leaves group 3's write
        unseparated from whatever the next iteration of the outer `i` loop
        does, and the next iteration writes the same slice.
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
            var passes = Float32(1.0) if bin != 128 else Float32(0.0)
            var offset = f + 8 * (bin & 127)

            var write_time = (tid >> 3) & 3

            var val1 = passes * stat1
            offset += 1 if flag else 0

            comptime for k in range(4):
                if k == write_time:
                    self._add(offset, val1)
                barrier()

            var val2 = passes * stat2
            offset = offset - 1 if flag else offset + 1

            comptime for k in range(4):
                if k == write_time:
                    self._add(offset, val2)
                barrier()

    def add_point_2(
        mut self,
        ci: SIMD[DType.uint32, 2],
        t: SIMD[DType.float32, 2],
        w: SIMD[DType.float32, 2],
        rows: SIMD[DType.uint32, 2],
    ):
        self.add_point(ci[0], t[0], w[0], rows[0])
        self.add_point(ci[1], t[1], w[1], rows[1])

    def add_point_4(
        mut self,
        ci: SIMD[DType.uint32, 4],
        t: SIMD[DType.float32, 4],
        w: SIMD[DType.float32, 4],
        rows: SIMD[DType.uint32, 4],
    ):
        self.add_point(ci[0], t[0], w[0], rows[0])
        self.add_point(ci[1], t[1], w[1], rows[1])
        self.add_point(ci[2], t[2], w[2], rows[2])
        self.add_point(ci[3], t[3], w[3], rows[3])

    def reduce(mut self):
        """`Reduce`, copied.

        Stage 1 as in the 5-bit file. Stage 2 has no inner-copy sum at all
        -- there is only one copy -- so each of the 256 threads simply
        RELOCATES four folds, `fold0 + 32k`, from `src[8 * fold]` to
        `Buffer[2 * (128 * f + fold) + w]`. 1024 output floats.
        """
        var tid = Int(thread_idx.x)
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
            var fold0 = (tid >> 1) & 31
            comptime MAX_FOLD_COUNT = 128
            var src = PW_WARP_HIST_SIZE + 2 * f + w
            comptime for k in range(4):
                var fold = fold0 + 32 * k
                self.base.unsafe_store(
                    2 * (MAX_FOLD_COUNT * f + fold) + w,
                    self.base.unsafe_load(src + 8 * fold),
                )
        barrier()
