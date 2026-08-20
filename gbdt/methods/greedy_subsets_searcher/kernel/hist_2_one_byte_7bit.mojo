"""The 7-bit specialization of the fused two-stat one-byte accumulator.

PORT OF `hist_2_one_byte_7bit.cu` at CatBoost `54a8143a`,
`TPointHist2OneByte<7, BlockSize>`. Transliterated. Do not improve.

The family CatBoost's one-byte dispatch takes at `64 < maxBins <= 128`
(`hist_one_byte.cu:320-322`), WHICH INCLUDES THEIR OWN GPU DEFAULT BORDER
COUNT of 128. This is the hot variant of the family.

Seven bits leave no room for sub-copies at all: `SliceOffset` is the bare
warp offset and each warp holds ONE 1024-slot histogram, `offset = f +
8 * (bin & 127) + flag`. With no replication, the write conflicts are
resolved entirely in TIME: `writeTime = (threadIdx.x >> 3) & 3` splits the
warp into four 8-lane shifts that write in turn, one warp-local sync apart
(`hist_2_one_byte_7bit.cu:68-102`). Within a shift the eight lanes' slot
low bits `f + flag` are a bijection of the lane, so equal bins still land on
distinct slots.

Their sync here is already the full warp (`tiled_partition<32>`, `:36`), so
`syncwarp` is not a widening in this file; it is their sync exactly, the
same correspondence `hist_one_byte.mojo` documents.
"""

from max.gpu.memory import AddressSpace
from max.gpu.sync import syncwarp

from gbdt.methods.greedy_subsets_searcher.kernel.histogram_utils import (
    hist2_smem_add,
)


def hist2_slice_offset_7(tid: Int) -> Int:
    """`TPointHist2OneByte<7>::SliceOffset()` (`hist_2_one_byte_7bit.cu:25-29`).

        const int warpId = (threadIdx.x / 32);
        const int warpOffset = 1024 * warpId;
        return warpOffset;
    """
    return 1024 * (tid // 32)


def hist2_add_points_7[
    n: Int, dt: DType
](
    ci: InlineArray[UInt32, n],
    s1: InlineArray[Float32, n],
    s2: InlineArray[Float32, n],
    q1: InlineArray[Int32, n],
    q2: InlineArray[Int32, n],
    tid: Int,
    slice_base: Int,
    smem: UnsafePointer[
        Scalar[dt],
        address_space = AddressSpace.SHARED,
        origin=MutUntrackedOrigin,
    ],
):
    """`TPointHist2OneByte<7>::AddPointsImpl<N>`
    (`hist_2_one_byte_7bit.cu:31-104`), branch for branch, including the
    ASYMMETRY between the two phases that is easy to "fix": phase 1 syncs
    BEFORE each shift except the first (`if (t > 0) syncTile.sync()`,
    `:71-74`), phase 2 syncs AFTER each shift (`:94-102`), and the two
    phases are separated by one more sync (`:91`). The counts are uniform
    across the warp either way, and the total is theirs.

    `pass = bin != 128` is EXACTLY 128: the one encodable value past a full
    7-bit feature, the skip mark. Copied, not generalized.

    `dt`, `q1` and `q2` exist for the `HIST_SMEM_SHARED2_I32` matrix row
    (`q1`/`q2` are the batch's stats through `hist2_quantize`, computed once
    per row at load time):
    pass structure, bin decode, slot arithmetic and the four-shift write-turn
    discipline are theirs in both modes, and WHERE the add lands is decided
    once in `hist2_smem_add` (see its DEVIATION BLOCK).
    """
    var flag = tid & 1

    var stat1 = InlineArray[Float32, n](fill=Float32(0.0))
    var stat2 = InlineArray[Float32, n](fill=Float32(0.0))
    var qstat1 = InlineArray[Int32, n](fill=Int32(0))
    var qstat2 = InlineArray[Int32, n](fill=Int32(0))

    @parameter
    for k in range(n):
        if flag == 1:
            stat1[k] = s2[k]
            stat2[k] = s1[k]
            qstat1[k] = q2[k]
            qstat2[k] = q1[k]
        else:
            stat1[k] = s1[k]
            stat2[k] = s2[k]
            qstat1[k] = q1[k]
            qstat2[k] = q2[k]

    var val1 = InlineArray[Float32, n](fill=Float32(0.0))
    var val2 = InlineArray[Float32, n](fill=Float32(0.0))
    var qval1 = InlineArray[Int32, n](fill=Int32(0))
    var qval2 = InlineArray[Int32, n](fill=Int32(0))
    var offset = InlineArray[Int, n](fill=0)

    @parameter
    for i in range(4):
        var f = (2 * i + tid) & 6

        @parameter
        for k in range(n):
            var bin = Int((ci[k] >> UInt32(24 - (f << 2))) & UInt32(255))
            var keep = Float32(0.0)
            var qkeep = Int32(0)
            if bin != 128:
                keep = Float32(1.0)
                qkeep = Int32(1)
            val1[k] = keep * stat1[k]
            val2[k] = keep * stat2[k]
            qval1[k] = qkeep * qstat1[k]
            qval2[k] = qkeep * qstat2[k]
            offset[k] = slice_base + f + 8 * (bin & 127) + flag

        # const int writeTime = (threadIdx.x >> 3) & 3;
        var write_time = (tid >> 3) & 3

        @parameter
        for t in range(4):
            if t > 0:
                syncwarp()
            if t == write_time:

                @parameter
                for k in range(n):
                    hist2_smem_add[dt](smem, offset[k], val1[k], qval1[k])

        # int shift = flag ? -1 : 1;
        var shift = 1
        if flag == 1:
            shift = -1

        @parameter
        for k in range(n):
            offset[k] += shift

        syncwarp()

        @parameter
        for t in range(4):
            if t == write_time:

                @parameter
                for k in range(n):
                    hist2_smem_add[dt](smem, offset[k], val2[k], qval2[k])
            syncwarp()


def hist2_reduce_tail_7[
    dt: DType
](
    tid: Int,
    smem: UnsafePointer[
        Scalar[dt],
        address_space = AddressSpace.SHARED,
        origin=MutUntrackedOrigin,
    ],
):
    """The 7-bit tail of `Reduce()` after `ReduceToOneWarp`
    (`hist_2_one_byte_7bit.cu:113-131`). No sub-copies to sum here, so it is
    a pure PERMUTATION: each of the 256 participating threads moves four
    folds out of the combined warp histogram at `smem + 2048` into the final
    `[stat][feature][fold]` layout at `smem[0..1023]`. The read window
    (`2048+`) and the write window (`< 1024`) are disjoint, which is why
    theirs carries no barrier between them either.

    The caller supplies the trailing `__syncthreads()` (`:132`).
    """
    if tid < 256:
        var is_second_stat = tid & 1
        var f = tid // 64
        var fold0 = (tid >> 1) & 31
        comptime max_fold_count = 128

        var src = 2048 + 2 * f + is_second_stat

        @parameter
        for k in range(4):
            var fold = fold0 + 32 * k
            smem[
                max_fold_count * 4 * is_second_stat
                + max_fold_count * f
                + fold
            ] = smem[src + 8 * fold]
