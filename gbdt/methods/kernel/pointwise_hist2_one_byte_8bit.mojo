"""The 8-bit pointwise accumulator: `TPointHist<2, 1, BlockSize>`.

PORT OF `catboost/cuda/methods/kernel/pointwise_hist2_one_byte_8bit.cu` at
CatBoost `54a8143a`. Transliterated. Do not improve.

Up to 256 bins per feature. **This one is not the 5/6/7 progression continued
-- it is a different design**, and the reason is that the progression runs
out of room.

    bits   warpHistSize   copies    threads sharing one copy
      5        1024          4                 8
      6        1024          2                16
      7        1024          1                32
      8        4096          2                64

`OUTER_HIST_BITS_COUNT = 2` makes the slice four times as wide, so at
CatBoost's 384-thread block only `384 * 32 / 4096 = 3` slices exist for 12
warps: four warps share each one, and with two inner copies that is 64
threads per copy. Turn-taking would need 64 turns. So CatBoost drops the
tile syncs entirely and does two other things instead:

1. `Add` becomes `atomicAdd`.
2. `AddPoint` DEFERS. It keeps `mostRecentBin[i]` and a running
   `mostRecentStat1/2[i]` per feature slot, and only flushes to shared
   memory when the bin CHANGES. On sorted or clustered data a run of equal
   bins costs one atomic instead of one per row. `Reduce` flushes the four
   pending accumulators before it does anything else -- forget that and
   every feature loses its last run.

============================ DEVIATION 93 ============================
**METAL HAS NO THREADGROUP FLOAT ATOMICS.** Probed directly 2026-08-21:
`Atomic.fetch_add` on a `Float32` in `AddressSpace.SHARED` fails to compile
with "Unsupported local float atomic operation for given target". Global
float atomics DO work, and so do warp primitives; this is specifically the
threadgroup float case.

So this accumulator holds **Int32 fixed point** and its atomics are integer
atomics. That is the same substitution `hist2_smem_add` already makes for
the greedy-subsets family, for the same reason, and it reuses that family's
`hist2_quantize` and `hist2_dither` unchanged rather than growing a second
quantizer -- there must be exactly one dithered quantizer in this tree or
the two families will drift.

WHY THE DITHER IS KEYED ON THE ROW AND NOT THE POSITION. `hist2_quantize`
is `floor(val * scale + u)` with `u` a hash; plain truncation biases every
cell low by up to one unit per row, which `histogram_utils.hist2_quantize`
records as a measured misfit, not a theory. The hash has to be stable per
DOCUMENT, because a document's position in the index array is reordered at
every level while its id is not -- and `parent == child + sibling` has to
hold to the integer for the partial pass to compute one child and subtract
for the other. That is why `PointHist2.add_point` carries a `row`.

WHAT THIS COSTS AND WHAT IT DOES NOT. Integer addition is associative, so
the shared accumulation is order-independent and therefore reproducible run
to run -- which the float version CatBoost ships is not. The cost is the
quantization error, which the dither makes zero-mean and O(sqrt(rows))
rather than O(rows). `scale` must come from `fixed_point.choose_scale` on
the plane's sum of magnitudes; the caller supplies it and this file does not
guess.
======================================================================

THE OUTPUT IS FIXED POINT TOO, and that is a real difference from the other
three files. `reduce` leaves Int32 in the first `4 * 256 * 2` slots and the
caller divides by `scale` when it writes to global. Converting inside
`reduce` would round twice and break the exact
`parent == child + sibling`.
"""

from std.atomic import Atomic
from std.gpu import thread_idx
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from gbdt.methods.greedy_subsets_searcher.kernel.histogram_utils import (
    hist2_dither,
    hist2_quantize,
)
from gbdt.methods.kernel.compute_point_hist2_loop import PointHist2
from gbdt.methods.kernel.pointwise_hist2_one_byte_5bit import (
    PW_HIST2_BLOCK,
    PW_HIST2_SMEM_FLOATS,
)

#: `OUTER_HIST_BITS_COUNT` and `INNER_HIST_BITS_COUNT` (`:42-43`).
comptime PW8_OUTER_BITS = 2
comptime PW8_INNER_BITS = 1

#: `const int warpHistSize = 1024 << OUTER_HIST_BITS_COUNT;`
comptime PW8_WARP_HIST_SIZE = 1024 << PW8_OUTER_BITS

#: `BLOCK_SIZE * 32 / (1024 << OUTER_HIST_BITS_COUNT)` (`:53`), stated here
#: as scratch-slots over slice width so it tracks the matrix. On Apple this
#: is 2 (8192-slot scratch), so four warps share each slice -- which is
#: the whole reason this accumulator needs atomics. The `% maxBlocks` wrap
#: in `SliceOffset` tolerates ANY number of warps per slice, which is what
#: let the doubled-block experiment run correctness-free before it was
#: measured a no-op (see `pw_hist2_block_size_for`).
comptime PW8_MAX_BLOCKS = PW_HIST2_SMEM_FLOATS // PW8_WARP_HIST_SIZE

#: `1 << (5 + INNER + OUTER)` (`:176`).
comptime PW8_MAX_FOLD_COUNT = 1 << (5 + PW8_INNER_BITS + PW8_OUTER_BITS)


@always_inline
def pw_hist2_slice_offset_8(tid: Int) -> Int:
    """`TPointHist<2,1,BLOCK_SIZE>::SliceOffset()` (`:51-63`).

    Note the `% maxBlocks` on the warp id, which none of the other three
    have: there are fewer slices than warps, so warps WRAP onto them.
    """
    var warp_id = (tid // 32) % PW8_MAX_BLOCKS
    var warp_offset = PW8_WARP_HIST_SIZE * warp_id
    comptime blocks = 4 >> PW8_INNER_BITS
    var inner_hist_start = tid & ((blocks - 1) << (PW8_INNER_BITS + 3))
    return warp_offset + inner_hist_start


struct PointHist8[origin: MutOrigin](PointHist2):
    """`TPointHist<2, 1, BLOCK_SIZE>` (`:41-206`), in Int32 fixed point.

    `scale` is `fixed_point.choose_scale(sum_of_magnitudes, row_count)` for
    the plane being accumulated, computed on the HOST before the round and
    passed in. This struct never picks one: a scale chosen from data the
    device happens to see is a scale that changes between two runs of the
    same fit.
    """

    var base: MutPointer[
        Int32, Self.origin, address_space = AddressSpace.SHARED
    ]
    var buffer_offset: Int
    var scale: Float32

    #: `mostRecentBin[4]`, `mostRecentStat1[4]`, `mostRecentStat2[4]`
    #: (`:46-48`), held in QUANTIZED units so the deferred run sums exactly.
    var recent_bin: SIMD[DType.int32, 4]
    var recent_stat1: SIMD[DType.int32, 4]
    var recent_stat2: SIMD[DType.int32, 4]

    def __init__(
        out self,
        buff: MutPointer[
            Int32, Self.origin, address_space = AddressSpace.SHARED
        ],
        scale: Float32,
    ):
        """Their constructor (`:65-83`), zero fill and pending state."""
        comptime assert PW_HIST2_SMEM_FLOATS >= 2 * PW8_WARP_HIST_SIZE, (
            "PointHist8 needs two warp-hist slices: Reduce stage 1 folds"
            " slice 0 onto slice 1 IN PLACE at offset PW8_WARP_HIST_SIZE."
            " A column whose shared-memory budget resolves the scratch"
            " below 8192 slots cannot host this accumulator at any block"
            " size and must not take the fixed one-byte route; see"
            " pw_hist2_block_size_for."
        )
        var tid = Int(thread_idx.x)
        var i = tid
        while i < PW_HIST2_SMEM_FLOATS:
            buff.unsafe_store(i, Int32(0))
            i += PW_HIST2_BLOCK
        self.base = buff
        self.buffer_offset = pw_hist2_slice_offset_8(tid)
        self.scale = scale
        self.recent_bin = SIMD[DType.int32, 4](0)
        self.recent_stat1 = SIMD[DType.int32, 4](0)
        self.recent_stat2 = SIMD[DType.int32, 4](0)
        barrier()

    @always_inline
    def _add(mut self, slot: Int, qval: Int32):
        """Their `Add(float val, float* dst)` (`:85-87`), which is
        `atomicAdd(dst, val)`. Int32 here; see DEVIATION 93."""
        _ = Atomic.fetch_add(
            self.base.unsafe_offset(self.buffer_offset + slot), qval
        )

    @always_inline
    def _slot_of(self, f: Int, bin: Int32, flag: Bool) -> Int:
        """Their offset arithmetic (`:101-105`), shared by `AddPoint` and
        the pending flush in `Reduce` -- which is why it is factored, and
        why the two cannot drift apart.

            offset  = f
            offset += 8 * (bin & ((1 << INNER) - 1))
            offset += 32 * (bin >> INNER)
        """
        comptime mask = (1 << PW8_INNER_BITS) - 1
        var offset = f
        offset += 8 * Int(bin & Int32(mask))
        offset += 32 * Int(bin >> Int32(PW8_INNER_BITS))
        return offset + (1 if flag else 0)

    def add_point(
        mut self, ci: UInt32, t: Float32, w: Float32, row: UInt32
    ):
        """`AddPoint` (`:89-121`), copied, with the stats quantized on entry.

        THE FLUSH IS ON A BIN CHANGE, NOT ON EVERY POINT, and the pending
        accumulator restarts at zero AFTER the flush -- so the point that
        triggered the flush is added to the NEW bin, not the old one. Read
        it the other way round and every run of equal bins is attributed one
        row late.

        `bin != mostRecentBin[i]` is their whole predicate. Note that the
        initial `mostRecentBin` is 0, so a first point in bin 0 does not
        flush -- correct, because there is nothing pending.

        THERE IS NO ABSENT-BIN TEST HERE. The 5/6/7 accumulators drop
        `bin == bins`; at 8 bits every value 0..255 is a real bin and 256
        does not fit in the byte, so CatBoost has nothing to test. Adding a
        test would be inventing one.
        """
        var tid = Int(thread_idx.x)
        var flag = (tid & 1) != 0

        var stat1 = t if flag else w
        var stat2 = w if flag else t

        # ONE dither per (row, stat), keyed on the DOCUMENT and not on the
        # position: see the module docstring's DEVIATION 93.
        var u = hist2_dither(Int(row))
        var q1 = hist2_quantize(stat1, self.scale, u)
        var q2 = hist2_quantize(stat2, self.scale, u)

        comptime for i in range(4):
            var f = (2 * i + tid) & 6
            var bin = Int32((ci >> UInt32(24 - (f << 2))) & 255)

            if bin != self.recent_bin[i]:
                var off1 = self._slot_of(f, self.recent_bin[i], flag)
                self._add(off1, self.recent_stat1[i])
                var off2 = off1 - 1 if flag else off1 + 1
                self._add(off2, self.recent_stat2[i])

                self.recent_bin[i] = bin
                self.recent_stat1[i] = 0
                self.recent_stat2[i] = 0

            self.recent_stat1[i] += q1
            self.recent_stat2[i] += q2

    def add_point_2(
        mut self,
        ci: SIMD[DType.uint32, 2],
        t: SIMD[DType.float32, 2],
        w: SIMD[DType.float32, 2],
        rows: SIMD[DType.uint32, 2],
    ):
        """`AddPoint2` (`:123-126`), which is theirs verbatim: two
        `AddPoint` calls. This accumulator has no fused wide form to
        deviate from."""
        self.add_point(ci[0], t[0], w[0], rows[0])
        self.add_point(ci[1], t[1], w[1], rows[1])

    def add_point_4(
        mut self,
        ci: SIMD[DType.uint32, 4],
        t: SIMD[DType.float32, 4],
        w: SIMD[DType.float32, 4],
        rows: SIMD[DType.uint32, 4],
    ):
        """`AddPoint4` (`:128-133`), theirs verbatim."""
        self.add_point(ci[0], t[0], w[0], rows[0])
        self.add_point(ci[1], t[1], w[1], rows[1])
        self.add_point(ci[2], t[2], w[2], rows[2])
        self.add_point(ci[3], t[3], w[3], rows[3])

    def reduce(mut self):
        """`Reduce` (`:137-205`), copied, in THREE stages.

        STAGE 0 (`:138-152`) is the one the other three files do not have:
        flush the four pending accumulators. Every feature slot has a run
        in flight when the loop ends, and without this every feature loses
        its last run of equal bins -- silently, and by more the better
        sorted the data is.
        """
        var tid = Int(thread_idx.x)
        var flag = (tid & 1) != 0

        comptime for i in range(4):
            var f = (2 * i + tid) & 6
            var off1 = self._slot_of(f, self.recent_bin[i], flag)
            self._add(off1, self.recent_stat1[i])
            var off2 = off1 - 1 if flag else off1 + 1
            self._add(off2, self.recent_stat2[i])

        # `Buffer -= SliceOffset()` (`:154`)
        self.buffer_offset = 0
        barrier()

        # STAGE 1 (`:156-168`): fold the slices down onto the first one
        var start = tid
        while start < PW8_WARP_HIST_SIZE:
            var acc = Int32(0)
            var i = start
            while i < PW_HIST2_SMEM_FLOATS:
                acc += self.base.unsafe_load(i)
                i += PW8_WARP_HIST_SIZE
            self.base.unsafe_store(PW8_WARP_HIST_SIZE + start, acc)
            start += PW_HIST2_BLOCK

        barrier()

        # STAGE 2 (`:170-204`): sum the inner copies and lay the result out
        # stat-MINOR. 256 threads, 256 folds, so each thread does TWO folds
        # (their `fold += 128`), and all four features at once.
        if tid < 256:
            var w = tid & 1
            comptime INNER_HIST_COUNT = 4 >> PW8_INNER_BITS
            comptime LOW_BIT_MASK = (1 << PW8_INNER_BITS) - 1
            var fold = tid >> 1
            while fold < PW8_MAX_FOLD_COUNT:
                var src = (
                    PW8_WARP_HIST_SIZE
                    + 8 * (fold & LOW_BIT_MASK)
                    + 32 * (fold >> PW8_INNER_BITS)
                    + w
                )
                var acc = SIMD[DType.int32, 4](0)
                comptime for in_warp_hist in range(INNER_HIST_COUNT):
                    comptime for f in range(4):
                        acc[f] += self.base.unsafe_load(
                            src
                            + 2 * f
                            + (in_warp_hist << (3 + PW8_INNER_BITS))
                        )
                comptime for f in range(4):
                    self.base.unsafe_store(
                        2 * (PW8_MAX_FOLD_COUNT * f + fold) + w, acc[f]
                    )
                fold += 128
        barrier()
