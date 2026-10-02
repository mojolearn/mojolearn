# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`TPointHistHalfByte<BlockSize>`: ONE accumulator, TWO kernels.

Reference: `catboost/cuda/methods/kernel/pointwise_hist2_half_byte_template.cuh`
(CatBoost `54a8143a`).

This is the pointwise family's small-bin accumulator, and the thing to know
before reading either kernel that uses it is that **there is only one of it**.
`pointwise_hist2_binary.cu:39` and `pointwise_hist2_half_byte.cu:42` both say

    using THist = TPointHistHalfByte<BlockSize>;

and they differ only in how they READ the reduced block:

    half-byte   8 features x up to 16 bins.  Read `smem[fold*16 + 2*fid + w]`
                straight out -- one cell per (feature, fold).
    binary      32 features x 1 bit.  A feature is one BIT of a nibble, so
                its "left" side is the SUM over the 8 nibble values whose
                bit is 0.

Both readers are at the bottom of this file, beside the accumulator they
read, because they are the only places the reduced layout is interpreted and
a layout change has to move all three together.

THE LAYOUT, which is where the arithmetic below comes from:

    warp slice     512 floats = 16 bins x 32 slots
    32 slots       2 inner copies (`threadIdx.x & 16`) x 16
    those 16       8 features (`f`, even, 0..14) x 2 stat parities

so 16 threads share an inner copy. They do not collide, and the mechanism is
the ROTATION rather than a turn-taking flag: `shift = threadIdx.x & 14` gives
the 16 threads 8 distinct starting features, `RotateRight` pre-rotates the
packed nibbles to match, and `f = (shift + (i << 1)) & 14` walks each thread
through all 8 in a different order. On any iteration the 8 threads of a
parity pair sit on 8 distinct features.

DEVIATION: their `thread_block_tile<32>::sync()`
becomes a threadgroup `barrier()`, 16 per point (8 iterations x 2).

DEVIATION: CatBoost launches both kernels at `blockSize = 768`
(`pointwise_hist2_binary.cu:142`, `pointwise_hist2_half_byte.cu:142`), which
at 16 floats per thread is 49,152 bytes against Apple's 32,768. The matrix
row resolves it to 512 -- exactly 32,768. **Unlike every other block in this
implementation, 512 is also a FLOOR**: their `Reduce` folds the warp slices under
`if (threadIdx.x < 512)`, so a smaller block leaves the top of the first
slice unfolded and loses folds with no other symptom. Asserted, not trusted.

NOT IN CATBOOST: `AddPoint4`. This accumulator has `AddPoint` and `AddPoint2`
only, and their drivers select the scalar or the `uint2` loop and never the
`uint4` one. `PointHist2` requires all three, so `add_point_4` is four
`add_point` calls here -- reachable from the check, never from a faithful
driver.
"""

from std.gpu import thread_idx
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.kernel_matrix import (
    K_POINTWISE_HIST_2_HALF_BYTE,
    TARGET_COLUMN,
    block_size_for,
    hist_floats_per_thread_for,
)

from gbdt.methods.kernel.compute_point_hist2_loop import PointHist2
from gbdt.methods.greedy_subsets_searcher.kernel.histogram_utils import (
    hist2_dither,
    hist2_quantize,
)
from std.atomic import Atomic, Ordering
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from std.sys.info import is_apple_gpu


#: 512 on Apple against CatBoost's 768. See the module docstring.
comptime PW_HB_BLOCK = block_size_for[
    K_POINTWISE_HIST_2_HALF_BYTE, TARGET_COLUMN
]()

#: `const int HIST_SIZE = 16 * BlockSize;`
comptime PW_HB_SMEM_FLOATS = (
    hist_floats_per_thread_for[K_POINTWISE_HIST_2_HALF_BYTE]() * PW_HB_BLOCK
)

#: `const int warpOffset = 512 * (threadIdx.x / 32);`
comptime PW_HB_WARP_SLICE = 512

#: The reduced block: 16 folds x (8 features x 2 stats).
comptime PW_HB_OUT_FLOATS = 256


@always_inline
def rotate_right(bin: UInt32, bits: Int) -> UInt32:
    """`RotateRight` (`cuda_util/kernel/kernel_helpers.cuh:43-45`):

        return (bin << bits) | (bin >> (32 - bits));

    TWO THINGS ABOUT IT ARE WORTH WRITING DOWN.

    First, despite the name it rotates LEFT. Transcribed under their name so
    a reader grepping CatBoost finds this; renaming it to match its
    behaviour would break that trail.

    Second, `bits == 0` makes their expression `bin >> 32`, which is
    UNDEFINED in C and in CUDA. It works on NVIDIA because the hardware takes
    the shift count modulo 32, so `bin >> 32` yields `bin` and the `or` is
    idempotent. `bits` here is `2 * (threadIdx.x & 14)`, so zero is not a
    corner case -- every thread with `threadIdx.x & 14 == 0` hits it, an
    eighth of them. The zero case is spelled out rather than left to a shift
    whose behaviour Mojo does not owe us.
    """
    if bits == 0:
        return bin
    return (bin << UInt32(bits)) | (bin >> UInt32(32 - bits))


@always_inline
def pw_hb_slice_offset(tid: Int) -> Int:
    """`TPointHistHalfByte::SliceOffset()` (`:19-23`).

        const int warpOffset = 512 * (threadIdx.x / 32);
        const int innerHistStart = threadIdx.x & 16;
        return warpOffset + innerHistStart;
    """
    return PW_HB_WARP_SLICE * (tid // 32) + (tid & 16)


struct PointHistHalfByte[origin: MutOrigin](PointHist2):
    """`TPointHistHalfByte<BlockSize>` (`:13-110`)."""

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
        """Their constructor (`:26-34`).

        NOTE THE ORDER: the reference zeroes, then `__syncthreads()`, then sets
        `Buffer`. The one-byte accumulators set `Buffer` before the sync.
        Same effect, kept in the reference order.
        """
        comptime assert PW_HB_BLOCK >= 512, (
            "TPointHistHalfByte::Reduce folds the warp slices under"
            " `if (threadIdx.x < 512)`"
            " (pointwise_hist2_half_byte_template.cuh:78), so a block below"
            " 512 leaves the top of the first slice unfolded and loses every"
            " fold above the block size WITHOUT any other symptom. Raise the"
            " shared-memory budget or write a folding loop; do not simply"
            " lower the block."
        )
        var tid = Int(thread_idx.x)
        var i = tid
        while i < PW_HB_SMEM_FLOATS:
            buff.unsafe_store(i, 0.0)
            i += PW_HB_BLOCK
        barrier()
        self.base = buff
        self.buffer_offset = pw_hb_slice_offset(tid)

    @always_inline
    def _add(mut self, slot: Int, val: Float32):
        var at = self.buffer_offset + slot
        self.base.unsafe_store(at, self.base.unsafe_load(at) + val)

    def add_point(
        mut self, ci: UInt32, t: Float32, w: Float32, row: UInt32
    ):
        """`AddPoint` (`:36-60`), copied.

        `row` is unused: this accumulator adds floats into a slice whose
        collisions the rotation already prevents, so it needs no atomic and
        therefore no fixed point. See `PointHist2`.
        """
        _ = row
        var tid = Int(thread_idx.x)
        var flag = (tid & 1) != 0
        var add_first = t if flag else w
        var add_second = w if flag else t

        var shift = tid & 14
        var bins = rotate_right(ci, 2 * shift)

        comptime for i in range(8):
            var f = (shift + (i << 1)) & 14
            var offset = Int((bins >> UInt32(28 - 4 * i)) & 15)
            offset <<= 5
            offset += f

            barrier()
            self._add(offset + (1 if flag else 0), add_first)

            barrier()
            self._add(offset + (0 if flag else 1), add_second)

    def add_point_2(
        mut self,
        ci: SIMD[DType.uint32, 2],
        t: SIMD[DType.float32, 2],
        w: SIMD[DType.float32, 2],
        rows: SIMD[DType.uint32, 2],
    ):
        """`AddPoint2` (`:62-65`), theirs verbatim: two `AddPoint` calls."""
        self.add_point(ci[0], t[0], w[0], rows[0])
        self.add_point(ci[1], t[1], w[1], rows[1])

    def add_point_4(
        mut self,
        ci: SIMD[DType.uint32, 4],
        t: SIMD[DType.float32, 4],
        w: SIMD[DType.float32, 4],
        rows: SIMD[DType.uint32, 4],
    ):
        """NO CATBOOST COUNTERPART -- see the module docstring's last note.
        Their drivers never select the 4-wide loop for this accumulator."""
        self.add_point(ci[0], t[0], w[0], rows[0])
        self.add_point(ci[1], t[1], w[1], rows[1])
        self.add_point(ci[2], t[2], w[2], rows[2])
        self.add_point(ci[3], t[3], w[3], rows[3])

    def reduce(mut self):
        """`Reduce` (`:67-108`), copied, in two stages.

        STAGE 1 folds the `BlockSize / 32` warp slices onto the first one:
        thread `tid < 512` sums `Buffer[512 * warpId + tid]` over warps.
        Their `fold`/`sumOffset` decomposition is just `tid` split as
        `32 * fold + sumOffset`, so the address is exactly
        `512 * warpId + tid`; it is written their way because the split is
        what makes the stride obvious in the source.

        STAGE 2 collapses the TWO inner copies, which sit 16 slots apart,
        and lands the result at `Buffer[tid]` for `tid < 256`:

            Buffer[16 * fold + e] = Buffer[32 * fold + e]
                                  + Buffer[32 * fold + e + 16]

        with `e = 2 * feature + parity`. Both stages put their
        `__syncthreads()` OUTSIDE the `if`, which is what keeps them legal
        under a threadgroup barrier.
        """
        var tid = Int(thread_idx.x)
        self.buffer_offset = 0
        comptime WARP_COUNT = PW_HB_BLOCK >> 5

        var fold = (tid >> 5) & 15
        var sum_offset = tid & 31
        var acc = Float32(0.0)
        if tid < 512:
            comptime for warp_id in range(WARP_COUNT):
                acc += self.base.unsafe_load(
                    PW_HB_WARP_SLICE * warp_id + sum_offset + 32 * fold
                )
        barrier()
        if tid < 512:
            self.base.unsafe_store(tid, acc)

        barrier()

        var fold2 = (tid >> 4) & 15
        var acc2 = Float32(0.0)
        if tid < 256:
            var hist_entry_id = tid & 15
            acc2 = self.base.unsafe_load(
                32 * fold2 + hist_entry_id
            ) + self.base.unsafe_load(32 * fold2 + hist_entry_id + 16)
        barrier()
        if tid < 256:
            self.base.unsafe_store(tid, acc2)
        barrier()


# ---------------------------------------------------------------------------
# The two readers of the reduced block. Same 256 floats, two interpretations.
# ---------------------------------------------------------------------------


@always_inline
def pw_hb_half_byte_slot(fid: Int, fold: Int, w: Int) -> Int:
    """`smem[fold * 16 + 2 * fid + w]` (`pointwise_hist2_half_byte.cu:82`).

    Eight features of up to 16 bins each; one reduced cell per (feature,
    fold, stat). The caller guards `fold < feature[fid].Folds`, which is
    theirs (`:81`) and is what keeps a 4-bin feature from claiming the
    16-bin block's tail.
    """
    return fold * 16 + 2 * fid + w


@always_inline
def pw_hb_binary_sum[
    origin: MutOrigin, //
](
    smem: MutPointer[Float32, origin, address_space = AddressSpace.SHARED],
    fid: Int,
    w: Int,
) -> Float32:
    """`ComputeSplitPropertiesBImpl`'s writeback sum
    (`pointwise_hist2_binary.cu:72-82`), copied.

        const int groupId = fid / 4;
        uchar fMask = 1 << (3 - (fid & 3));
        float sum = 0.f;
        for (int i = 0; i < 16; i++) {
            if (!(i & fMask)) { sum += counters[i * 16 + 2 * groupId + w]; }
        }

    A BINARY FEATURE IS ONE BIT OF A NIBBLE, and that is the whole reason
    this is a sum rather than a lookup. Thirty-two one-bit features pack into
    eight nibbles; feature `fid` lives in nibble `fid / 4` at bit
    `3 - (fid & 3)`. The accumulator knows nothing about that -- it built a
    16-bin histogram over the nibble VALUE -- so the feature's "bit is 0"
    side is recovered by adding the eight nibble values whose bit is clear.

    The bit numbering is theirs and it is MOST-SIGNIFICANT-FIRST:
    `1 << (3 - (fid & 3))`, so feature 0 of a group is bit 3, not bit 0.
    Reading it the other way round transposes each group of four features
    onto each other and, because that is a permutation WITHIN the group,
    leaves the total over the group unchanged.
    """
    var group_id = fid // 4
    var f_mask = 1 << (3 - (fid & 3))
    var acc = Float32(0.0)
    for i in range(16):
        if (i & f_mask) == 0:
            acc += smem.unsafe_load(i * 16 + 2 * group_id + w)
    return acc


# ================= lane/neural-pass124 (2026-10-02) =================
# THE HALF-BYTE AND BINARY HISTOGRAMS IN FIXED POINT under the ordered
# tiers. `PointHistHalfByte` adds floats between two barriers per add (16
# barriers a point) so its slices never collide, and float sums tie the
# cell to that order: one block per feature group, every document in one
# block (L40S Ordered taxi 4.1M: the depth-0 half-byte and binary launches
# 7.6 and 7.8 ms a tree on 32 blocks of a 142-SM part). Here, as the 8-bit
# accumulator does (DEVIATION 93), each point's two stats are quantized on
# entry (`hist2_quantize`, the dither keyed on the row) and added as Int32
# with a shared atomic into the reduced layout itself,
# `[value * 16 + 2 * nibble + stat]`: an integer sum is the same in any
# order, so no slices, no turn-taking barriers, no warp fold, and the
# document axis may be split across blocks (the slots fold as integers).
# NEW BITS for the ordered tiers' half-byte and binary cells (the host
# oracle sums the same quantized integers); FAST keeps CatBoost's floats.
comptime PW_HB_FIXED = GLOBAL_NUMERIC_MODE != NUMERIC_FAST


comptime PW_HBI_WARPS = PW_HB_BLOCK // 32
#: Apple's threadgroup atomics cost more than the turn-taking barriers
#: (M4: 272 vs 209 ms of pw.hist at taxi 1M), so there the integer
#: accumulator keeps `PointHistHalfByte`'s slices, rotation and two barriers
#: an add, and adds Int32; elsewhere one copy per warp with shared atomics.
#: Integer sums either way: the same cells.
comptime PW_HBI_ATOMIC = not is_apple_gpu()


struct PointHistHalfByteInt[origin: MutOrigin](PointHist2):
    var base: MutPointer[Int32, Self.origin, address_space = AddressSpace.SHARED]
    var root: MutPointer[Int32, Self.origin, address_space = AddressSpace.SHARED]
    var scale: Float32

    def __init__(
        out self,
        buff: MutPointer[Int32, Self.origin, address_space = AddressSpace.SHARED],
        scale: Float32,
    ):
        comptime assert PW_HBI_WARPS * PW_HB_OUT_FLOATS <= PW_HB_SMEM_FLOATS, "the warp copies outgrow the half-byte page"
        var tid = Int(thread_idx.x)
        var i = tid
        while i < PW_HB_SMEM_FLOATS:
            buff.unsafe_store(i, Int32(0))
            i += PW_HB_BLOCK
        comptime if PW_HBI_ATOMIC:
            self.base = buff + (tid // 32) * PW_HB_OUT_FLOATS
        else:
            self.base = buff + pw_hb_slice_offset(tid)
        self.root = buff
        self.scale = scale
        barrier()

    def add_point(mut self, ci: UInt32, t: Float32, w: Float32, row: UInt32):
        var u = hist2_dither(Int(row))
        var qw = hist2_quantize(w, self.scale, u)
        var qt = hist2_quantize(t, self.scale, u)
        comptime if PW_HBI_ATOMIC:
            if qw == Int32(0) and qt == Int32(0):
                return
            comptime for i in range(8):
                var at = Int((ci >> UInt32(28 - 4 * i)) & 15) * 16 + 2 * i
                if qw != Int32(0):
                    _ = Atomic.fetch_add[ordering = Ordering.RELAXED](self.base.unsafe_offset(at), qw)
                if qt != Int32(0):
                    _ = Atomic.fetch_add[ordering = Ordering.RELAXED](self.base.unsafe_offset(at + 1), qt)
        else:
            # `PointHistHalfByte.add_point`'s windows, Int32 adds
            var tid = Int(thread_idx.x)
            var flag = (tid & 1) != 0
            var add_first = qt if flag else qw
            var add_second = qw if flag else qt
            var shift = tid & 14
            var bins = rotate_right(ci, 2 * shift)
            comptime for i in range(8):
                var f = (shift + (i << 1)) & 14
                var offset = Int((bins >> UInt32(28 - 4 * i)) & 15)
                offset <<= 5
                offset += f
                barrier()
                var a1 = offset + (1 if flag else 0)
                self.base.unsafe_store(a1, self.base.unsafe_load(a1) + add_first)
                barrier()
                var a2 = offset + (0 if flag else 1)
                self.base.unsafe_store(a2, self.base.unsafe_load(a2) + add_second)

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
        """Onto `root[0:256]` as `[value * 16 + 2 * nibble + stat]`: the
        warp copies summed (atomic form), or `PointHistHalfByte.reduce`'s
        two stages in Int32 (turn-taking form)."""
        var tid = Int(thread_idx.x)
        comptime if PW_HBI_ATOMIC:
            barrier()
            var acc = Int32(0)
            if tid < PW_HB_OUT_FLOATS:
                for wv in range(PW_HBI_WARPS):
                    acc = acc + self.root.unsafe_load(wv * PW_HB_OUT_FLOATS + tid)
            barrier()
            if tid < PW_HB_OUT_FLOATS:
                self.root.unsafe_store(tid, acc)
            barrier()
        else:
            var fold = (tid >> 5) & 15
            var sum_offset = tid & 31
            var acc = Int32(0)
            if tid < 512:
                comptime for warp_id in range(PW_HBI_WARPS):
                    acc = acc + self.root.unsafe_load(PW_HB_WARP_SLICE * warp_id + sum_offset + 32 * fold)
            barrier()
            if tid < 512:
                self.root.unsafe_store(tid, acc)
            barrier()
            var fold2 = (tid >> 4) & 15
            var acc2 = Int32(0)
            if tid < 256:
                var e = tid & 15
                acc2 = self.root.unsafe_load(32 * fold2 + e) + self.root.unsafe_load(32 * fold2 + e + 16)
            barrier()
            if tid < 256:
                self.root.unsafe_store(tid, acc2)
            barrier()


@always_inline
def pw_hb_binary_sum_int[
    origin: MutOrigin, //
](
    smem: MutPointer[Int32, origin, address_space = AddressSpace.SHARED],
    fid: Int,
    w: Int,
) -> Int32:
    """`pw_hb_binary_sum` over the Int32 cells (an integer sum)."""
    var group_id = fid // 4
    var f_mask = 1 << (3 - (fid & 3))
    var acc = Int32(0)
    for i in range(16):
        if (i & f_mask) == 0:
            acc = acc + smem.unsafe_load(i * 16 + 2 * group_id + w)
    return acc
