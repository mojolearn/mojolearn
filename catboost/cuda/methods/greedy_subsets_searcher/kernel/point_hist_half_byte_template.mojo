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


def add_point_slot(ci: UInt32, tid: Int, i: Int) -> Int:
    """The slot `AddPoint` writes on iteration `i`, as pure arithmetic.

        const int f = (threadIdx.x + i) & 7;
        int bin = (ci >> (28 - 4 * f)) & 15;
        bin <<= 5;
        bin += f;

    DEVIATION (PORTING.md 10): CatBoost keeps this inside `AddPoint`, which
    also owns the shared pointer and the 8-lane sync. Mojo cannot pass a
    shared-memory pointer across a function boundary without a concrete
    origin, so the ARITHMETIC lives here and the load, the add and the
    barrier are inlined at the call site. Same operations, same order; the
    split is a language constraint and not a design change.

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


def reduce_stage2_slot(tid: Int, group: Int) -> Int:
    """`Reduce()` stage 2's read offset: `32 * fold + featureId + 8 * group`
    for `fold = (tid >> 3) & 15` and `featureId = tid & 7`."""
    var fold = (tid >> 3) & 15
    var feature_id = tid & 7
    return 32 * fold + feature_id + 8 * group
