# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`TTreeUpdater`: the per-document bin array the FEATURE-PARALLEL searcher
splits on, and the compressed-bit chain that fills it.

PORT OF, in one file because they are one call chain:

    `catboost/cuda/gpu_data/oblivious_tree_bin_builder.{h,cpp}`
        `IBinarySplitProvider`, `TSplitHelper::Split`,
        `TSplitHelper::BuildMirrorSplitForDataSet` (`:89-114`),
        `TTreeUpdater::AddSplit` (`:200-210`)
    `catboost/cuda/gpu_data/splitter.h`
        `CreateCompressedSplit` (`:141-154`),
        `UpdateBinFromCompressedBits` (`:165-172`)
    `catboost/cuda/gpu_data/kernel/split.cu`
        `TBinSplitLoader` (`:9-34`), `TBinUpdater` (`:60-73`),
        `WriteCompressedSplitImpl` (`:76-99`), `UpdateBinsImpl` (`:123-136`)
    `catboost/cuda/cuda_util/kernel/compression_helper.cuh`
        `TCompressionHelper<ui64, 128>::CompressBlock` (`:73-108`) and
        `::DecompressBlock` (`:111-133`), at `BitsPerKey == 1`
    `catboost/cuda/cuda_util/compression_helpers_gpu.cpp`
        `CompressedSizeImpl<ui64>` (`:249-254`)

at CatBoost `54a8143a`. Transliterated. Do not improve.

WHY THIS FILE EXISTS AT ALL, AND WHY `PORTING.md` 91 B IS WRONG ABOUT IT
------------------------------------------------------------------------
91 B says the feature-parallel and doc-parallel searchers "share their
entire stack" and differ in "exactly three lines of `CreateSubsets`". The
first half is nearly true and the second half is not, and THIS IS THE
COUNTEREXAMPLE: the two `TSubsetsHelper::Split` specializations are
different code calling different kernels.

    Stripe (doc-parallel)   UpdateBinFromCompressedIndex(cindex, feature,
                              bin, docsForBins, depth, Bins)
                            -- reads the COMPRESSED INDEX at the gathered
                               position and ORs the predicate straight in
                            (`pointwise_optimization_subsets.cpp:35-40`)

    Mirror (feature-par.)   UpdateBins(Bins, nextLevelDocBins, docMap,
                              CurrentDepth, FoldBits)
                            -- reads a SEPARATE per-document bin array,
                               `docBins`, which `TTreeUpdater` maintains
                            (`pointwise_optimization_subsets.h:75-93`)

`docBins` does not exist on the doc-parallel path. It is one `ui32` per
document in ORIGINAL document order whose bit `d` is the document's side of
split `d`, and the whole of this file is what puts the bits there:

    TTreeUpdater::AddSplit(split)                         (`:200-210`)
      -> TSplitHelper::Split(split, LearnBins, BinarySplits.size())  (`:38-45`)
        -> GetCompressedBits(split)                       (`:22-36`)
          -> BuildMirrorSplitForDataSet                   (`:89-114`)
            -> CreateCompressedSplit -> WriteCompressedSplit  ONE BIT PER
               DOCUMENT, PACKED 64 TO A ui64
        -> UpdateBinFromCompressedBits -> UpdateBins       decompress and OR
                                                           into docBins

So the feature-parallel arm computes the same predicate the doc-parallel arm
computes, writes it into a bit-packed intermediate, decompresses it into a
document-ordered array, and only THEN gathers it into the subsets' bins --
three kernels where the doc-parallel arm uses one.

**WHY THEY DO IT THE LONG WAY, and it is not an accident.** The packed
`TMirrorBuffer<ui64>` is the thing `TScopedCacheHolder` CACHES
(`:84-87`, `:124-132`): a split's bits are keyed by the split and by the
dataset scope, so the same split reused by a later tree, by the test set, or
by a tree-CTR tensor tracker costs a lookup instead of a pass over the
compressed index. One bit per document is 1/32 of a `ui32` bin array, which
is what makes caching every split in a tree affordable. The doc-parallel
searcher caches nothing and therefore needs no intermediate.

At `FoldBits == 0` and one device the two chains produce BIT-IDENTICAL
`subsets.Bins`, which is what `mojo_only/feature_parallel_identity_check.mojo`
gates. They are not the same code and rung 2 is not free.

THE PACKED LAYOUT IS INTERLEAVED, NOT CONTIGUOUS
------------------------------------------------
`TCompressionHelper<ui64, 128>` at one bit per key holds 64 keys per word
and 128 words per block, so 8192 keys per block -- but key `k` of a block
does NOT live in word `k / 64`. `CompressBlock` (`:93`) writes

    key at local offset `BLOCK_SIZE * id + tid`  ->  word `tid`, bit `63 - id`

so consecutive documents land in CONSECUTIVE WORDS at the same bit position,
and the 64 documents sharing a word are 128 apart. That is a coalescing
layout: 128 adjacent threads write 128 adjacent words. Reading it as
`bits[k / 64] >> (k % 64)` gives a well-formed bin array made of the wrong
documents -- every count is right, every placement is wrong.

DEVIATION 122: the `TScopedCacheHolder` is not ported, so nothing is cached.
DEVIATION 123: `CompressBlock`'s four-register accumulator is one register
here.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.gpu import block_dim, block_idx, grid_dim, thread_idx

from gbdt.models.oblivious_model import TBinarySplit


# ---------------------------------------------------------------------------
# `TCompressionHelper<ui64, CompressCudaBlockSize()>` at `BitsPerKey == 1`
# ---------------------------------------------------------------------------

comptime COMPRESS_BLOCK_SIZE = 128
"""`CompressCudaBlockSize()` (`cuda_util/kernel/compression.cuh:8-10`).

    constexpr ui32 CompressCudaBlockSize() { return 128; }

DEVIATION 100 replaced CatBoost's histogram block-size literals with this
tree's kernel matrix, because those kernels size THREADGROUP MEMORY from the
block size and Metal's budget is 32 KB rather than 48. **This kernel
allocates none**, so the literal is portable exactly as written and is kept
-- and it is not merely a launch shape here: `CompressedSize` multiplies by
it, `WriteCompressedSplit` strides the destination by it, and
`CompressBlock` uses it as the stride BETWEEN the keys sharing a word.
Changing it changes the on-device layout, not the occupancy.
"""

comptime SPLIT_BITS_PER_KEY = 1
"""`CompressedSize<ui64>(docCount, 2)`
(`gpu_data/oblivious_tree_bin_builder.cpp:95`) -> `IntLog2(2) == 1`. A split
is one bit.

`NCB::IntLog2` is `ceil(log2(v))` (`libs/helpers/math_utils.h:14-16`), the
same CEIL that `PORTING.md` 107 records costing a day when it was read as
floor. At two unique values ceil and floor agree, so this constant is not
where that trap lives -- it is recorded because the SPELLING is the same.
"""

comptime KEYS_PER_STORAGE = 64
"""`KeysPerStorageType = sizeof(TStorageType) * 8 / bitsPerKey`
(`compression_helper.cuh:44`) = 64 / 1."""

comptime KEYS_PER_COMPRESS_BLOCK = KEYS_PER_STORAGE * COMPRESS_BLOCK_SIZE
"""`KeysPerBlock() = KeysPerStorageType * BLOCK_SIZE` (`:48-50`) = 8192."""


def compressed_split_size(doc_count: Int) -> Int:
    """`CompressedSize<ui64>(count, 2)`
    (`cuda_util/compression_helpers_gpu.cpp:249-254`).

        const ui32 keysPerBlock = KeysPerBlock<TStorageType>(bitsPerKey);
        return CeilDivide(count, keysPerBlock) * CompressCudaBlockSize();

    **IT IS `numBlocks * 128`, NOT `ceil(count / 64)`.** The last block is
    allocated in full even when it holds one document, because the layout is
    per-block and `WriteCompressedSplit` strides the destination by
    `BLOCK_SIZE * blockIdx.x`. Sizing this the "tight" way overruns nothing
    at 8192 documents a block and silently truncates the last block above
    that -- the documents in it read zero, land in the left child, and the
    tree is well-formed and wrong.
    """
    if doc_count <= 0:
        return 0
    var num_blocks = (
        doc_count + KEYS_PER_COMPRESS_BLOCK - 1
    ) // KEYS_PER_COMPRESS_BLOCK
    return num_blocks * COMPRESS_BLOCK_SIZE


def compressed_split_blocks(doc_count: Int) -> Int:
    """`numBlocks = CeilDivide(size, TCompressionHelper<ui64, blockSize>(1)
    .KeysPerBlock())` -- shared verbatim by `WriteCompressedSplit`
    (`split.cu:145`) and `UpdateBins` (`split.cu:171`), which is why the two
    kernels agree about where a block's words are."""
    if doc_count <= 0:
        return 0
    return (
        doc_count + KEYS_PER_COMPRESS_BLOCK - 1
    ) // KEYS_PER_COMPRESS_BLOCK


# ---------------------------------------------------------------------------
# `WriteCompressedSplitImpl<BLOCK_SIZE>` (`gpu_data/kernel/split.cu:76-99`)
# ---------------------------------------------------------------------------


def write_compressed_split_kernel(
    compressed_index: MutPointer[UInt32, MutAnyOrigin],
    indices: MutPointer[UInt32, MutAnyOrigin],
    has_indices_in: Int32,
    size_in: Int32,
    feature_offset_in: Int32,
    feature_mask: UInt32,
    feature_shift: UInt32,
    one_hot_in: Int32,
    bin_idx: UInt32,
    compressed_bits: MutPointer[UInt64, MutAnyOrigin],
):
    """`WriteCompressedSplitImpl<128>` (`:76-99`) with `TBinSplitLoader`
    (`:9-34`) inlined, at `BitsPerKey == 1`.

        if (indices) { indices += helper.KeysPerBlock() * blockIdx.x; }
        else         { compressedIndex += helper.KeysPerBlock() * blockIdx.x; }
        size -= helper.KeysPerBlock() * blockIdx.x;
        compressedBits += BLOCK_SIZE * blockIdx.x;
        compressedIndex += feature.Offset;

        const ui32 value = binIdx << feature.Shift;
        const ui32 mask  = feature.Mask << feature.Shift;
        TBinSplitLoader loader(compressedIndex, indices, value, mask,
                               feature.OneHotFeature);
        helper.CompressBlock(loader, size, compressedBits);

    **NOTE WHICH POINTER MOVES.** With `indices == nullptr` the COMPRESSED
    INDEX is advanced by the block's key count and the loader's `offset` is
    a document id LOCAL to the block; with indices it is the INDEX array
    that moves and `compressedIndex` keeps its global base. Advancing both,
    or neither, gives block 0 the right answer and every other block a
    shifted one -- which a fixture under 8193 rows cannot see, because there
    is only block 0.

    `feature.Offset` is added AFTER the block stride, and it is an ELEMENT
    offset into the compressed index. This tree's layout stores a COLUMN
    index and strides by `n_rows`, so the caller multiplies -- the same
    conversion the doc-parallel searcher's comment records costing a debug
    session.

    THE PREDICATE IS `TBinSplitLoader::operator()` (`:29-33`) and it is the
    SAME ONE `UpdateBinsFromCompressedIndexImpl` uses (`:191-197`):

        featureVal = CompressedIndex[idx] & Mask;
        return TakeEqual ? (featureVal == Value) : featureVal > Value;

    `TakeEqual` is `feature.OneHotFeature`. That the two kernels share a
    predicate is what makes this port's identity gate an identity at all.
    """
    var block = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var has_indices = has_indices_in != 0
    var one_hot = one_hot_in != 0

    var cindex_base = Int(feature_offset_in)
    var indices_base = 0
    if has_indices:
        indices_base = KEYS_PER_COMPRESS_BLOCK * block
    else:
        cindex_base += KEYS_PER_COMPRESS_BLOCK * block

    var src_size = Int(size_in) - KEYS_PER_COMPRESS_BLOCK * block
    var dst_base = COMPRESS_BLOCK_SIZE * block

    var value = bin_idx << feature_shift
    var mask = feature_mask << feature_shift

    # `CompressBlock(loader, srcSize, dst)` (`compression_helper.cuh:73-108`).
    # DEVIATION 123: their four `compressedEntries[N]` registers OR-folded at
    # `:100-103` are one accumulator here; the value is the same OR of the
    # same 64 terms and the register split is scheduling.
    var acc = UInt64(0)
    for id in range(KEYS_PER_STORAGE):
        var offset = COMPRESS_BLOCK_SIZE * id + tid
        var key = UInt64(0)
        if offset < src_size:
            # TBinSplitLoader::operator()(offset)
            var idx = offset
            if has_indices:
                idx = Int(indices.unsafe_load(indices_base + offset))
            var feature_val = (
                compressed_index.unsafe_load(cindex_base + idx) & mask
            )
            var bit = UInt32(0)
            if one_hot:
                if feature_val == value:
                    bit = UInt32(1)
            else:
                if feature_val > value:
                    bit = UInt32(1)
            key = UInt64(bit) & UInt64(1)
        # `(key << ((KeysPerStorageType - id - 1) * BitsPerKey))` (`:95`)
        acc |= key << UInt64(KEYS_PER_STORAGE - id - 1)

    # `if (tid < srcSize) { dst[tid] = compressedEntries[0]; }` (`:105-107`).
    # The guard is in KEYS, not words, and it is theirs: at a full block
    # `srcSize` is 8192 so all 128 threads write, and at a partial one the
    # keys a thread owns are exactly the ones it just packed.
    if tid < src_size:
        compressed_bits.unsafe_store(dst_base + tid, acc)


def create_compressed_split(
    ctx: DeviceContext,
    mut compressed_index: DeviceBuffer[DType.uint32],
    mut indices: DeviceBuffer[DType.uint32],
    has_indices: Bool,
    doc_count: Int,
    feature_offset: UInt32,
    feature_mask: UInt32,
    feature_shift: UInt32,
    feature_one_hot: Bool,
    bin_idx: UInt32,
    mut compressed_bits: DeviceBuffer[DType.uint64],
) raises:
    """`CreateCompressedSplit` (`gpu_data/splitter.h:141-154`) and
    `NKernel::WriteCompressedSplit` (`split.cu:138-151`).

        constexpr int blockSize = CompressCudaBlockSize();
        const int numBlocks = CeilDivide(size,
            TCompressionHelper<ui64, blockSize>(1).KeysPerBlock());
        if (numBlocks) { WriteCompressedSplitImpl<blockSize><<<...>>>(...); }

    Their `if (numBlocks)` is the whole of their empty-grid guard on this
    call and it is kept.

    `has_indices` is `readIndices != nullptr` upstream. `TSplitHelper::
    GetCompressedBits` passes `nullptr` for an ordinary feature (`:25`) and
    `&DataSet.GetInverseIndices()` for a PERMUTATION-DEPENDENT one (`:27-29`)
    -- a simple-CTR column, whose compressed index is stored in the
    permutation's order while `docBins` is in the original one. **The True
    side is not reachable from this rung** (there are no CTR columns in the
    feature-parallel searcher until rung 4), so it is gated separately in
    `mojo_only/feature_parallel_identity_check.mojo` against the identity
    permutation rather than left as an unrun branch -- `PORTING_RULES.md` 8.
    """
    var num_blocks = compressed_split_blocks(doc_count)
    if num_blocks == 0:
        return
    ctx.enqueue_function[write_compressed_split_kernel](
        compressed_index.unsafe_ptr(),
        indices.unsafe_ptr(),
        Int32(1) if has_indices else Int32(0),
        Int32(doc_count),
        Int32(feature_offset),
        feature_mask,
        feature_shift,
        Int32(1) if feature_one_hot else Int32(0),
        bin_idx,
        compressed_bits.unsafe_ptr(),
        grid_dim=(num_blocks, 1, 1),
        block_dim=(COMPRESS_BLOCK_SIZE, 1, 1),
    )


# ---------------------------------------------------------------------------
# `UpdateBinsImpl<BLOCK_SIZE>` (`gpu_data/kernel/split.cu:123-136`)
# ---------------------------------------------------------------------------


def update_bins_from_compressed_bits_kernel(
    compressed_bits: MutPointer[UInt64, MutAnyOrigin],
    depth_in: Int32,
    bins: MutPointer[UInt32, MutAnyOrigin],
    size_in: Int32,
):
    """`UpdateBinsImpl<128>` (`:123-136`) with `TBinUpdater` (`:60-73`)
    inlined, at `BitsPerKey == 1`.

        bins += helper.KeysPerBlock() * blockIdx.x;
        size -= helper.KeysPerBlock() * blockIdx.x;
        compressedBits += BLOCK_SIZE * blockIdx.x;
        TBinUpdater writer(bins, depth);
        helper.DecompressBlock(writer, compressedBits, size);

    and `TBinUpdater::operator()(offset, bin)` is `Bins[offset] |= bin
    << Depth`.

    **IT IS AN OR, NOT A STORE**, and `Depth` is `BinarySplits.size()` at the
    moment `AddSplit` was called (`oblivious_tree_bin_builder.cpp:202`) --
    the number of splits ALREADY in the tree, which is the current depth.
    So `docBins` accumulates the path: bit `d` is the document's side of
    split `d`, and after the last level it IS the leaf id. That is why
    `Fit` hands it to `CacheBinsForModel` unchanged
    (`oblivious_tree_structure_searcher.cpp:300-304`).

    A store instead of an OR gives every document a leaf id of 0 or
    `1 << d` -- two occupied leaves at every depth, a monotone model that
    still beats the mean, and no crash.
    """
    var block = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var depth = UInt32(depth_in)

    var bins_base = KEYS_PER_COMPRESS_BLOCK * block
    var dst_size = Int(size_in) - KEYS_PER_COMPRESS_BLOCK * block
    var src_base = COMPRESS_BLOCK_SIZE * block

    # `const TStorageType compressedKeys = tid < dstSize ? src[tid] : 0;`
    # (`compression_helper.cuh:118`). The guard is in KEYS, matching the one
    # `CompressBlock` wrote under.
    var compressed = UInt64(0)
    if tid < dst_size:
        compressed = compressed_bits.unsafe_load(src_base + tid)

    for id in range(KEYS_PER_STORAGE):
        var dst_offset = COMPRESS_BLOCK_SIZE * id + tid
        if dst_offset < dst_size:
            # `(compressedKeys >> ((KeysPerStorageType - id - 1) *
            #   BitsPerKey)) & mask` (`:127`)
            var entry = UInt32(
                (compressed >> UInt64(KEYS_PER_STORAGE - id - 1)) & UInt64(1)
            )
            var slot = bins_base + dst_offset
            bins.unsafe_store(slot, bins.unsafe_load(slot) | (entry << depth))


def update_bin_from_compressed_bits(
    ctx: DeviceContext,
    mut compressed_bits: DeviceBuffer[DType.uint64],
    mut bins: DeviceBuffer[DType.uint32],
    doc_count: Int,
    depth: Int,
) raises:
    """`UpdateBinFromCompressedBits` (`gpu_data/splitter.h:165-172`) and
    `NKernel::UpdateBins` (`split.cu:165-176`).

    Same `numBlocks` and same `blockSize` as `WriteCompressedSplit`, which
    is what makes the two agree about a block's words -- see
    `compressed_split_blocks`.
    """
    var num_blocks = compressed_split_blocks(doc_count)
    if num_blocks == 0:
        return
    ctx.enqueue_function[update_bins_from_compressed_bits_kernel](
        compressed_bits.unsafe_ptr(),
        Int32(depth),
        bins.unsafe_ptr(),
        Int32(doc_count),
        grid_dim=(num_blocks, 1, 1),
        block_dim=(COMPRESS_BLOCK_SIZE, 1, 1),
    )


# ---------------------------------------------------------------------------
# `TTreeUpdater` (`gpu_data/oblivious_tree_bin_builder.{h,cpp}`)
# ---------------------------------------------------------------------------


struct TTreeUpdater(Movable):
    """`TTreeUpdater` (`oblivious_tree_bin_builder.h:93-...`), reduced to the
    learn set with no test set, no tensor tracker and no CTR splits.

    Theirs holds `LearnBins`, `TestBins`, `BinarySplits`, a `TSplitHelper`
    per dataset, a `TFeatureTensorTracker` and the `TScopedCacheHolder`. At
    this rung there is one dataset, no CTRs and no cache, so what is left is
    `LearnBins` and `BinarySplits`.

    DEVIATION 121: NO TEST SET. `LinkedTest`/`TestBins` (`:103-104`, and the
    second `Split` at `:204-206`) mirror every split into a second bin array
    for the held-out pool so the leaves estimator can score it. This
    repository's boosting loop owns the test pool elsewhere and hands the
    searcher one set of documents; a test-bin array with no reader is a
    field that is never checked. It is not arithmetic and it changes no
    split.

    DEVIATION 122: NO `TScopedCacheHolder`. Their `GetCompressedBits` caches
    the packed bits keyed by `(dataset scope, split)`
    (`oblivious_tree_bin_builder.cpp:84-87`), so a split proposed again by a
    later tree, or needed again by the test set or a tensor tracker, is a
    lookup. Ours recomputes. **Within one tree the cache CANNOT hit** --
    `Fit` breaks on `result.HasSplit(bestSplit)` (`:266`) before calling
    `AddSplit`, so no split is added twice -- so the cost is one
    `WriteCompressedSplit` per repeated split ACROSS trees, and the
    identity this rung gates is unaffected. Implementing a cache that never
    hits inside the only caller there is would be a path with no reachable
    branch, which this repository has been bitten by four times
    ([[reached-but-inert]]).
    """

    var learn_bins: DeviceBuffer[DType.uint32]
    """`LearnBins` -- their `docBins`, one `ui32` per document in ORIGINAL
    document order. `Fit` creates it as
    `TMirrorBuffer<ui32>::CopyMapping(DataSet.GetIndices())`
    (`oblivious_tree_structure_searcher.cpp:49`) and hands it to
    `CacheBinsForModel` at the end (`:300-304`)."""

    var compressed_bits: DeviceBuffer[DType.uint64]
    """The packed intermediate. Theirs is a fresh
    `TMirrorBuffer<ui64>::Create(CompressedSize<ui64>(docCount, 2))` per
    split inside `BuildMirrorSplitForDataSet` (`:96`); ours is allocated
    once and reused, because with no cache holder there is nothing to hold
    the old ones and a per-level allocation on a Metal queue is a stall.
    Every word is fully overwritten by `CompressBlock`, which writes an
    assignment rather than an OR, so reuse carries nothing forward -- and
    `mojo_only/feature_parallel_identity_check.mojo`'s poison sub-gate is
    what says so rather than this sentence."""

    var binary_splits: List[TBinarySplit]
    """`BinarySplits`. Its SIZE is the argument `AddSplit` passes as the
    depth (`:202`), so this list is not bookkeeping -- it is the bit index."""

    var doc_count: Int

    def __init__(
        out self,
        ctx: DeviceContext,
        doc_count: Int,
    ) raises:
        """`TTreeUpdater(cacheHolder, featuresManager, ctrTargets, learnSet,
        learnBins)` plus `Fit`'s `docBins` creation (`:49`).

        Theirs does NOT zero `docBins`. `TMirrorBuffer<ui32>::CopyMapping`
        allocates and `TSplitHelper::Split` ORs into it, so their first
        `AddSplit` ORs into whatever the allocator handed back -- except
        that `Fit` is the only creator and CUDA's allocator on this path
        hands back a zeroed slab. Ours memsets, because relying on that is
        the difference between a tree and a tree-shaped random number.
        """
        self.doc_count = doc_count
        self.learn_bins = ctx.enqueue_create_buffer[DType.uint32](doc_count)
        ctx.enqueue_memset(self.learn_bins, UInt32(0))
        var packed = compressed_split_size(doc_count)
        if packed < 1:
            packed = 1
        self.compressed_bits = ctx.enqueue_create_buffer[DType.uint64](packed)
        self.binary_splits = List[TBinarySplit]()

    def add_split(
        mut self,
        ctx: DeviceContext,
        mut compressed_index: DeviceBuffer[DType.uint32],
        mut read_indices: DeviceBuffer[DType.uint32],
        has_read_indices: Bool,
        split: TBinarySplit,
        feature_offset: UInt32,
        feature_mask: UInt32,
        feature_shift: UInt32,
        feature_one_hot: Bool,
    ) raises:
        """`TTreeUpdater::AddSplit(split)`
        (`oblivious_tree_bin_builder.cpp:200-210`), call for call.

            SplitHelper->Split(split, LearnBins, (ui32)BinarySplits.size());
            if (LinkedTest) { TestSplitHelper->Split(...); }   // DEVIATION 121
            BinarySplits.push_back(split);

        and `TSplitHelper::Split` (`:38-45`) is

            const auto& compressedBits = GetCompressedBits(split);
            UpdateBinFromCompressedBits(compressedBits, bins, depth);

        **THE DEPTH IS `BinarySplits.size()` BEFORE THE PUSH.** Reading it
        after would write bit `d + 1` and leave bit `d` clear: the leaf ids
        would be a strictly increasing set of powers of two, every level's
        partition array would come back half empty, and the searcher would
        still return a tree of the right depth.
        """
        var depth = len(self.binary_splits)
        create_compressed_split(
            ctx,
            compressed_index,
            read_indices,
            has_read_indices,
            self.doc_count,
            feature_offset,
            feature_mask,
            feature_shift,
            feature_one_hot,
            UInt32(split.bin_idx),
            self.compressed_bits,
        )
        update_bin_from_compressed_bits(
            ctx,
            self.compressed_bits,
            self.learn_bins,
            self.doc_count,
            depth,
        )
        self.binary_splits.append(split)
