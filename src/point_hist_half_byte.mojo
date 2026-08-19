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

DEVIATION, see PORTING.md 1 and 2: `BLOCK_SIZE` is 256 rather than their 768
because Apple caps threadgroup memory at 32 KB, and their 8-lane
`tiled_partition<8>::sync()` becomes a threadgroup-wide `barrier()` because
Mojo 1.0 has no warp primitives.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier


# DEVIATION (PORTING.md 1): CatBoost uses 768 here for the half-byte and
# binary kernels. `BLOCK_SIZE * 16` floats is the scratch, so 768 asks for
# 49,152 bytes and Apple gives 32,768. 512 asks for exactly 32,768, which is
# the largest block size that fits and the one that keeps `Reduce()`'s
# hardcoded 512-thread stage literal.
comptime BLOCK_SIZE = 512

#: The width `Reduce()`'s first two stages run at. CatBoost writes the
#: literal 512 and can, because its BlockSize is 768. Ours must not exceed
#: the block, or stage 1 writes only `BLOCK_SIZE` of the 512 slots that stage
#: 2 goes on to read, and the histogram silently loses everything above it.
#: At BLOCK_SIZE 512 this IS 512 and the port stays literal; the expression
#: exists so that dropping to 256 stays correct rather than silently wrong.
comptime REDUCE_WIDTH = BLOCK_SIZE if BLOCK_SIZE < 512 else 512

#: `GetHistSize()`, `BlockSize * 16`.
comptime HIST_SIZE = BLOCK_SIZE * 16


def slice_offset(tid: Int) -> Int:
    """`TPointHistHalfByteBase::SliceOffset()`, copied exactly.

        const int warpOffset = 512 * (threadIdx.x / 32);
        const int innerHistStart = threadIdx.x & 24;
        return warpOffset + innerHistStart;

    `& 24` keeps bits 3 and 4, giving 0, 8, 16 or 24: four sub-copies inside
    each warp's 512-float region. Do not "simplify" it to `& 31`; the low
    three bits are the lane's position inside the 8-lane tile and are carried
    by `f` instead.
    """
    return 512 * (tid // 32) + (tid & 24)


def hist_zero_and_slice(
    buff: UnsafePointer[Scalar[DType.float32], _, address_space = AddressSpace.SHARED],
    tid: Int,
) -> UnsafePointer[Scalar[DType.float32], _, address_space = AddressSpace.SHARED]:
    """The constructor: zero the whole scratch, sync, return this lane's slice.

        for (int i = threadIdx.x; i < histSize; i += BlockSize) buff[i] = 0;
        __syncthreads();
        Histogram = buff + SliceOffset();
    """
    var i = tid
    while i < HIST_SIZE:
        buff[i] = Scalar[DType.float32](0.0)
        i += BLOCK_SIZE
    barrier()
    return buff + slice_offset(tid)


@always_inline
def add_point(
    hist: UnsafePointer[Scalar[DType.float32], _, address_space = AddressSpace.SHARED],
    ci: UInt32,
    t: Scalar[DType.float32],
    tid: Int,
):
    """`TPointHistHalfByteBase::AddPoint`, copied exactly.

        for (int i = 0; i < 8; i++) {
            const int f = (threadIdx.x + i) & 7;
            int bin = (ci >> (28 - 4 * f)) & 15;
            bin <<= 5;
            bin += f;
            Histogram[bin] += t;
            addToHistTile.sync();
        }

    DEVIATION (PORTING.md 2): `addToHistTile.sync()` is an 8-lane barrier.
    Mojo 1.0 has only the threadgroup-wide `barrier()`, so this port syncs
    256 threads where CatBoost syncs 8. Correct, strictly more expensive, and
    the largest known deviation in the port. It is NOT safe to drop: the
    rotation only guarantees distinct slots WITHIN an iteration, so lanes
    must not run ahead into the next one.
    """

    @parameter
    for i in range(8):
        var f = (tid + i) & 7
        var bin = Int((ci >> UInt32(28 - 4 * f)) & UInt32(15))
        bin <<= 5
        bin += f
        hist[bin] += t
        # DEVIATION: tile sync widened to block barrier.
        barrier()


def hist_reduce(
    hist_sliced: UnsafePointer[
        Scalar[DType.float32], _, address_space = AddressSpace.SHARED
    ],
    tid: Int,
) -> UnsafePointer[
    Scalar[DType.float32], _, address_space = AddressSpace.SHARED
]:
    """`TPointHistHalfByteBase::Reduce()`, copied exactly.

    Collapses the per-warp, per-sub-copy replication back to one histogram of
    `8 features x 16 folds`, left at `Histogram[featureId + 8 * fold]`.

    Three stages, and the middle one is why the scratch is `16 * BlockSize`
    rather than `16 * 8`:

    1. Undo the slice, then fold the whole `16 * BlockSize` scratch down to
       512 floats by strided summation, 512 threads each walking a stride.
    2. Fold those 512 down to 128: thread `tid < 128` sums the four sub-copy
       groups `Histogram[32 * fold + featureId + 8 * group]` for
       `fold = (tid >> 3) & 15` and `featureId = tid & 7`.
    3. Write `Histogram[tid] = sum` for `tid < 128`.

    Takes the SLICED pointer and returns the BASE, because their version
    mutates `Histogram` in place with `Histogram -= SliceOffset()` and every
    later read is against the base.
    """
    var hist = hist_sliced - slice_offset(tid)

    barrier()
    # Stage 1, their `for (i = threadIdx.x; i < histSize; i += 512)`. The
    # outer `while slot` loop is the DEVIATION: it covers the 512 slots with
    # BLOCK_SIZE threads instead of assuming at least 512 of them. At
    # BLOCK_SIZE 512 it runs exactly once and is their loop.
    var slot = tid
    while slot < REDUCE_WIDTH:
        var sum = Scalar[DType.float32](0.0)
        var i = slot
        while i < HIST_SIZE:
            sum += hist[i]
            i += REDUCE_WIDTH
        barrier()
        hist[slot] = sum
        barrier()
        slot += BLOCK_SIZE

    var fold = (tid >> 3) & 15
    var sum2 = Scalar[DType.float32](0.0)
    if tid < 128:
        var feature_id = tid & 7

        @parameter
        for group in range(4):
            sum2 += hist[32 * fold + feature_id + 8 * group]
    barrier()
    if tid < 128:
        # featureId + 8 * fold
        hist[tid] = sum2
    barrier()
    return hist
