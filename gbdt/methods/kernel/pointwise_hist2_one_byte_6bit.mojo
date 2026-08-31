# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The 6-bit pointwise accumulator: `TPointHist<0, 1, BlockSize>`.

PORT OF `catboost/cuda/methods/kernel/pointwise_hist2_one_byte_6bit.cu` at
CatBoost `54a8143a`. Transliterated. Do not improve.

Up to 64 bins per feature, 64 as the absent marker. Everything structural
about it follows from ONE fact: twice the bins in the same 1024-slot warp
slice means half as many private copies, so **two inner copies instead of
four, and sixteen threads sharing each one instead of eight.**

That is why this file has a write-ordering flag the 5-bit file does not.

    5-bit   blocks = 4   innerHistStart = tid & ((4-1) << 3)  = tid & 24
    6-bit   blocks = 2   innerHistStart = tid & ((2-1) << 4)  = tid & 16

With 8 threads per copy the (feature, parity) slots are all distinct and no
two threads can collide, which is why `AddPoint` there is two writes and two
syncs. With 16 threads per copy they CAN collide, so CatBoost splits them by
bit 3 of the thread id (`writeFirstFlag = threadIdx.x & 8`) and lets the two
halves write in sequence. Four syncs per stat pair instead of two.

THE BIN-TO-SLOT MAP IS NOT `32 * bin` ANY MORE, and this is the line to read
twice:

    offset = f + 16 * (bin & 62) + 8 * (bin & 1)

The bin's LOW bit picks which half of an 8-slot pair-group it lands in and
the remaining bits stride by 16, which packs 64 bins into the same 1024 slots
the 5-bit layout gives 32 bins. Read as `32 * (bin & 31)` -- the 5-bit
spelling -- it aliases every even bin onto its odd neighbour, and the totals
do not move.

Same two family conventions as the 5-bit file, and they are the ones that
change no total: WEIGHT in the even slot, TARGET in the odd one, and a
stat-MINOR reduce. See that file's docstring.

DEVIATION (PORTING.md 11 and 92): their `thread_block_tile<16>::sync()`
becomes a threadgroup `barrier()`, the only sync Mojo exposes. This
accumulator takes FOUR per iteration of its four-iteration loop, so sixteen
threadgroup barriers per point against the 5-bit file's eight. Priced, not
free, and it is a consequence of the missing lane sync rather than of the
algorithm.
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
def pw_hist2_slice_offset_6(tid: Int) -> Int:
    """`TPointHist<0,1,BLOCK_SIZE>::SliceOffset()`.

    `blocks = 2` and the shift is 4, against the 5-bit file's 4 and 3.
    """
    var warp_offset = PW_WARP_HIST_SIZE * (tid // 32)
    var inner_hist_start = tid & ((2 - 1) << 4)
    return warp_offset + inner_hist_start


struct PointHist6[origin: MutOrigin](PointHist2):
    """`TPointHist<0, 1, BLOCK_SIZE>`."""

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
        self.buffer_offset = pw_hist2_slice_offset_6(tid)
        barrier()

    @always_inline
    def _add(mut self, slot: Int, val: Float32):
        var at = self.buffer_offset + slot
        self.base.unsafe_store(at, self.base.unsafe_load(at) + val)

    def add_point(
        mut self, ci: UInt32, t: Float32, w: Float32, row: UInt32
    ):
        """`AddPoint`, copied including the four syncs.

        `writeFirstFlag` is theirs and it is the whole reason this is not
        the 5-bit routine with a different mask: sixteen threads share one
        inner copy here, so the halves take turns.

        Note that the pass test MULTIPLIES rather than selects
        (`pass * stat1` against the 5-bit file's `pass ? stat1 : 0.0f`).
        Transcribed as written -- at bin 64 the product is an exact zero
        either way, and changing which one runs is a change to what nvcc was
        measured against.
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
            var passes = Float32(1.0) if bin != 64 else Float32(0.0)
            var offset = f + 16 * (bin & 62) + 8 * (bin & 1)

            var write_first = (tid & 8) != 0

            var val1 = passes * stat1
            var val2 = passes * stat2

            offset += 1 if flag else 0

            barrier()
            if write_first:
                self._add(offset, val1)
            barrier()
            if not write_first:
                self._add(offset, val1)

            offset = offset - 1 if flag else offset + 1

            barrier()
            if write_first:
                self._add(offset, val2)
            barrier()
            if not write_first:
                self._add(offset, val2)

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

        Stage 1 is identical to the 5-bit file's. Stage 2 is not: 256
        threads have to land 512 output floats, so each one writes TWO
        folds, `fold0` and `fold0 + 32`, reading them 512 slots apart.
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
            comptime MAX_FOLD_COUNT = 64
            comptime INNER_HIST_COUNT = 2
            var src = (
                PW_WARP_HIST_SIZE + 8 * (fold0 & 1) + 32 * (fold0 >> 1) + w
            )
            var sum0 = Float32(0.0)
            var sum1 = Float32(0.0)
            comptime for in_warp_hist in range(INNER_HIST_COUNT):
                sum0 += self.base.unsafe_load(
                    src + 2 * f + (in_warp_hist << 4)
                )
                sum1 += self.base.unsafe_load(
                    src + 2 * f + (in_warp_hist << 4) + 512
                )
            self.base.unsafe_store(
                2 * (MAX_FOLD_COUNT * f + fold0) + w, sum0
            )
            self.base.unsafe_store(
                2 * (MAX_FOLD_COUNT * f + fold0 + 32) + w, sum1
            )
        barrier()
