# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Write one feature's bins into the packed compressed index.

Reference: `WriteCompressedIndexImpl`,
`catboost/cuda/gpu_data/kernel/binarize.cu` (CatBoost `54a8143a`).

**This is the kernel that creates the read-density advantage.** Everything
downstream reads `cindex[feature.Offset + row]` and extracts its feature by
shift and mask, so one 4-byte load serves 32 binary features, 8 half-byte
features or 4 one-byte features. Without this the packing is arithmetic in
`grid_policy` that nothing acts on.

The reference kernel:

    cindex += feature.Offset;
    ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < docCount) {
        const ui32 bin = (((ui32)bins[i]) & feature.Mask) << feature.Shift;
        cindex[i] = cindex[i] | bin;
        i += blockDim.x * gridDim.x;
    }

Note the OR, not a store. Features sharing a `UInt32` are written one at a
time by separate launches, each contributing its own bit field, so the
destination must already be ZERO before the first feature of a group writes.
That is a precondition on the caller and it is easy to get wrong: a reused
buffer that is not cleared produces bins that are the OR of two datasets,
silently, with no bounds error to catch it.

**Their multi-feature variant uses `atomicOr`** (`binarize.cu:93`) when
several features are written concurrently into the same word. Metal HAS
integer atomics, so unlike the float `atomicAdd` in the histogram flush, that
one implements directly if we ever need it. Recorded because it is the one place
CatBoost's atomics are portable to us and it would be easy to assume
otherwise.
"""

from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.memory import bitcast, stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from gbdt.apple_fast_classical import AFCL_T03
from gbdt.gpu_data.apple_fast_trees_experiments import AFT_G05, AFT_G06


#: `binarize.cu:26`.
comptime WRITE_BLOCK_SIZE = 256


def write_compressed_index_kernel(
    feature_offset: Int32,
    feature_mask: UInt32,
    feature_shift: UInt32,
    bins: MutPointer[UInt8, MutAnyOrigin],
    doc_count: Int32,
    cindex: MutPointer[UInt32, MutAnyOrigin],
):
    """`WriteCompressedIndexImpl`, copied.

    One feature, every row, OR-ed into the word this feature shares with its
    group. `feature_offset` is in UInt32 units and selects the group's column
    of the compressed index.

    DEVIATION: their `TCFeature` is a struct passed by value; Mojo kernel
    parameters are scalars and pointers, so the three fields it reads are
    passed separately. Same values, same arithmetic.
    """
    var n = Int(doc_count)
    var base = Int(feature_offset)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    while i < n:
        var bin = (UInt32(bins.unsafe_load(i)) & feature_mask) << feature_shift
        cindex.unsafe_store(base + i, cindex.unsafe_load(base + i) | bin)
        i += stride


#: `BinarizeFloatFeature`'s launch shape (`binarize.cu:245-246`).
# G05: a 512-thread group stages the same at-most-256 borders while
# halving group occupancy pressure. Eight rows/lane and comparisons stay.
# Uncompiled/unverified/unmeasured; A/B uses the same quantization route.
comptime BINARIZE_BLOCK_SIZE = 512 if AFT_G05 or AFCL_T03 else 1024
comptime BINARIZE_DOCS_PER_THREAD = 8


@always_inline
def exact_f32_gt(a: Float32, b: Float32) -> Bool:
    """IEEE `a > b` on float32, decided on the BIT PATTERNS, so a device
    that flushes subnormal operands of a float compare (Metal) answers what
    the host answers. gbdt-depthwise / denormal on an Apple M4 (0.8.35,
    `verify --cross-check all`): a subnormal value against a subnormal
    border compared as `0 > 0` on Metal and as the true order on the host,
    so GPU and host predicts of the same saved model put rows in different
    bins. The fit quantizes through this same kernel, so Metal fits on
    subnormal inputs now see the order NVIDIA, AMD and the host already
    saw.

    Exactly the float `>`: NaN on either side is false; -0 and +0 are
    equal (both map to key 0x80000000); every other pair compares by the
    sign-magnitude total order of its bits."""
    var ua = bitcast[DType.uint32](a)
    var ub = bitcast[DType.uint32](b)
    var ma = ua & UInt32(0x7FFFFFFF)
    var mb = ub & UInt32(0x7FFFFFFF)
    if ma > UInt32(0x7F800000) or mb > UInt32(0x7F800000):
        return False
    # a positive value's key is its bits with the sign set; a negative
    # value's key is the bitwise NOT. A signed zero is canonicalised to +0.
    var ka = (ua | UInt32(0x80000000)) if (ua >> 31) == 0 else ~ua
    var kb = (ub | UInt32(0x80000000)) if (ub >> 31) == 0 else ~ub
    if ma == 0:
        ka = UInt32(0x80000000)
    if mb == 0:
        kb = UInt32(0x80000000)
    return ka > kb


def binarize_float_feature_kernel(
    feature_offset: Int32,
    feature_mask: UInt32,
    feature_shift: UInt32,
    values: MutPointer[Float32, MutAnyOrigin],
    doc_count: Int32,
    borders: MutPointer[Float32, MutAnyOrigin],
    dst: MutPointer[UInt32, MutAnyOrigin],
):
    """`BinarizeFloatFeatureImpl<false, 1024, 8>`, copied (`binarize.cu:37`).

    Raw float values -> this feature's bin, OR-ed into the packed
    compressed index, which is the quantization CatBoost's own predict
    performs internally on raw input. The BORDERS BUFFER LAYOUT IS THEIRS:
    `borders[0]` holds the border COUNT as a float, `borders[1..count]`
    hold the border values, sorted ascending. The bin is the count of
    borders the value EXCEEDS (`featureValues[j] > borderValue`,
    `binarize.cu:77`), which is exactly `numpy.searchsorted(side='left')`
    -- the rule `tools/interleaved_prep.py` binned with, so a CPU-binned
    fixture and this kernel agree bin for bin.

    The `<false>` (non-atomic) arm: features are written one launch at a
    time, as `write_compressed_index_kernel` writes them, so the plain
    read-OR-store cannot race. Their `<true>` arm exists for concurrent
    features and Metal's integer `atomicOr` could carry it if ever needed.
    The `gatherIndex` argument of theirs is null on this path and is not
    carried. DEVIATION (same as `write_compressed_index_kernel`): their
    by-value `TCFeature` arrives as three scalar parameters.
    """
    var n = Int(doc_count)
    var base = Int(feature_offset)
    var tid = Int(thread_idx.x)
    var i = Int(block_idx.x) * BINARIZE_BLOCK_SIZE * BINARIZE_DOCS_PER_THREAD + tid

    # `__shared__ float sharedBorders[256]`: slot 0 broadcasts the count,
    # then the borders themselves replace it, exactly their two-step load.
    var shared_borders = stack_allocation[
        256,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    shared_borders[0] = borders.unsafe_load(0)
    barrier()
    var borders_count = Int(shared_borders[0])
    barrier()
    if tid < borders_count:
        shared_borders[tid] = borders.unsafe_load(tid + 1)
    barrier()

    var index = InlineArray[UInt32, BINARIZE_DOCS_PER_THREAD](fill=0)
    var feature_values = InlineArray[Float32, BINARIZE_DOCS_PER_THREAD](
        fill=Float32(0.0)
    )

    @parameter
    for j in range(BINARIZE_DOCS_PER_THREAD):
        var idx = i + j * BINARIZE_BLOCK_SIZE
        if idx < n:
            feature_values[j] = values.unsafe_load(idx)

    for border in range(borders_count):
        var border_value = shared_borders[border]

        @parameter
        for j in range(BINARIZE_DOCS_PER_THREAD):
            # bit-pattern compare, not the float `>`: see `exact_f32_gt`
            if exact_f32_gt(feature_values[j], border_value):
                index[j] += 1

    @parameter
    for j in range(BINARIZE_DOCS_PER_THREAD):
        var idx = i + j * BINARIZE_BLOCK_SIZE
        if idx < n:
            var bin = dst.unsafe_load(base + idx)
            bin |= (index[j] & feature_mask) << feature_shift
            dst.unsafe_store(base + idx, bin)


# ---------------------------------------------------------------------------
# lane/apple-fast-sym-feat (2026-10-03): kernels referenced ONLY under the
# FAST + Apple guards `GBDT_QUANT_DEVICE` / `GBDT_INDEX_PACK_DEVICE`
# (`gbdt/train.mojo`) and `GBDT_PREDICT_PACKED` (`gbdt/resident_model.mojo`).
# IDENTICAL never instantiates them.
# ---------------------------------------------------------------------------

#: `pack_cindex_words_kernel`'s block and rows per thread (2048 rows a block).
# G06: retain 2048 rows per block but halve threads and double independent
# rows per worker, amortizing the shared word/border table over more work
# per lane. This deliberately trades registers for occupancy; no evidence.
comptime PACK_BLOCK = 128 if AFT_G06 else 256
comptime PACK_DOCS = 16 if AFT_G06 else 8
#: the most features one compressed-index word holds (the binary policy's
#: `features_per_int`), and the shared slab for the word's border values:
#: 32 x 1 (binary), 8 x 15 (half-byte) or 4 x 255 (one-byte) at most, so
#: 1024 floats covers every policy. The host builder checks both bounds
#: (`pack_word_table`) and takes the per-feature launches when exceeded.
comptime PACK_MAX_ENTRIES = 32
comptime PACK_BORDER_CAP = 1024


def pack_cindex_words_kernel(
    x_cols: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    word_start: MutPointer[UInt32, MutAnyOrigin],
    entry_col: MutPointer[UInt32, MutAnyOrigin],
    entry_shift: MutPointer[UInt32, MutAnyOrigin],
    entry_mask: MutPointer[UInt32, MutAnyOrigin],
    entry_slab: MutPointer[UInt32, MutAnyOrigin],
    entry_sub: MutPointer[Float32, MutAnyOrigin],
    borders: MutPointer[Float32, MutAnyOrigin],
    cindex: MutPointer[UInt32, MutAnyOrigin],
):
    """Every bordered feature of every compressed-index word in ONE launch.

    `block_idx.y` is the word (the `offset` column of `build_layout`); its
    features are entries `word_start[w] .. word_start[w + 1])` of the
    per-entry tables: the column of `x_cols` (column-major, `col * n_rows +
    row`), the feature's shift and mask, the offset of its border slab in
    `borders` (`borders[slab]` the count as a float, the values after it,
    the layout `binarize_float_feature_kernel` reads) and the NaN substitute
    (`nan_substitution`; a NaN substitute means leave NaN alone, and a NaN
    then compares false against every border and lands in bin 0, exactly as
    the per-feature kernel bins an AS_IS column). Per row the word is
    assembled in a register from `(bin & mask) << shift` over its entries
    and STORED once: no OR into a zeroed buffer, no atomics, because this
    kernel owns every bit of the word. The bin is the same count of
    `value > border` the per-feature kernel takes, over the same borders,
    so the words are bit for bit the 220-launch build's.
    """
    var n = Int(n_rows_in)
    var w = Int(block_idx.y)
    var e0 = Int(word_start.unsafe_load(w))
    var ne = Int(word_start.unsafe_load(w + 1)) - e0
    var sh_meta = stack_allocation[
        5 * PACK_MAX_ENTRIES,
        Scalar[DType.uint32],
        address_space = AddressSpace.SHARED,
    ]()
    var sh_sub = stack_allocation[
        PACK_MAX_ENTRIES,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var sh_vals = stack_allocation[
        PACK_BORDER_CAP,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var tid = Int(thread_idx.x)
    if tid == 0:
        var acc = 0
        for e in range(ne):
            var slab = Int(entry_slab.unsafe_load(e0 + e))
            var cnt = Int(borders.unsafe_load(slab))
            sh_meta[e] = UInt32(cnt)
            sh_meta[PACK_MAX_ENTRIES + e] = UInt32(acc)
            sh_meta[2 * PACK_MAX_ENTRIES + e] = entry_col.unsafe_load(e0 + e)
            sh_meta[3 * PACK_MAX_ENTRIES + e] = entry_shift.unsafe_load(e0 + e)
            sh_meta[4 * PACK_MAX_ENTRIES + e] = entry_mask.unsafe_load(e0 + e)
            sh_sub[e] = entry_sub.unsafe_load(e0 + e)
            acc += cnt
    barrier()
    for e in range(ne):
        var cnt = Int(sh_meta[e])
        var off = Int(sh_meta[PACK_MAX_ENTRIES + e])
        var slab = Int(entry_slab.unsafe_load(e0 + e))
        var j = tid
        while j < cnt:
            sh_vals[off + j] = borders.unsafe_load(slab + 1 + j)
            j += PACK_BLOCK
    barrier()
    var i = Int(block_idx.x) * PACK_BLOCK * PACK_DOCS + tid
    var word = InlineArray[UInt32, PACK_DOCS](fill=0)
    for e in range(ne):
        var cnt = Int(sh_meta[e])
        var off = Int(sh_meta[PACK_MAX_ENTRIES + e])
        var col = Int(sh_meta[2 * PACK_MAX_ENTRIES + e])
        var shift = sh_meta[3 * PACK_MAX_ENTRIES + e]
        var mask = sh_meta[4 * PACK_MAX_ENTRIES + e]
        var sub = sh_sub[e]
        var vals = InlineArray[Float32, PACK_DOCS](fill=Float32(0.0))
        var bins = InlineArray[UInt32, PACK_DOCS](fill=0)

        @parameter
        for j in range(PACK_DOCS):
            var idx = i + j * PACK_BLOCK
            if idx < n:
                var v = x_cols.unsafe_load(col * n + idx)
                if v != v and sub == sub:
                    v = sub
                vals[j] = v
        for b in range(cnt):
            var bv = sh_vals[off + b]

            @parameter
            for j in range(PACK_DOCS):
                # bit-pattern compare, as `binarize_float_feature_kernel`
                # (`exact_f32_gt`: Metal flushes subnormal compare operands)
                if exact_f32_gt(vals[j], bv):
                    bins[j] += 1

        @parameter
        for j in range(PACK_DOCS):
            word[j] |= (bins[j] & mask) << shift

    @parameter
    for j in range(PACK_DOCS):
        var idx = i + j * PACK_BLOCK
        if idx < n:
            cindex.unsafe_store(w * n + idx, word[j])


#: `transpose_rows_to_columns_kernel`: 32 x 32 tiles, 32 x 8 threads.
comptime TR_TILE = 32
comptime TR_ROWS_PER_PASS = 8
comptime TR_MAX_GRID_Y = 4096


def transpose_rows_to_columns_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    src: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    n_cols_in: Int32,
):
    """Row-major `src[row * n_cols + col]` to column-major `dst[col * n_rows
    + row]`, tiled through threadgroup memory so both sides are coalesced.
    Moves bits, computes nothing. `block_idx.x` is the column tile, the row
    tiles grid-stride over `block_idx.y` (the grid's y is capped at
    `TR_MAX_GRID_Y`)."""
    var n_rows = Int(n_rows_in)
    var n_cols = Int(n_cols_in)
    var tile = stack_allocation[
        TR_TILE * (TR_TILE + 1),
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var tx = Int(thread_idx.x)
    var ty = Int(thread_idx.y)
    var c0 = Int(block_idx.x) * TR_TILE
    var n_row_tiles = (n_rows + TR_TILE - 1) // TR_TILE
    var rt = Int(block_idx.y)
    while rt < n_row_tiles:
        var r0 = rt * TR_TILE

        @parameter
        for k in range(TR_TILE // TR_ROWS_PER_PASS):
            var lr = ty + k * TR_ROWS_PER_PASS
            var r = r0 + lr
            var c = c0 + tx
            if r < n_rows and c < n_cols:
                tile[lr * (TR_TILE + 1) + tx] = src.unsafe_load(r * n_cols + c)
        barrier()

        @parameter
        for k in range(TR_TILE // TR_ROWS_PER_PASS):
            var lc = ty + k * TR_ROWS_PER_PASS
            var c = c0 + lc
            var r = r0 + tx
            if r < n_rows and c < n_cols:
                dst.unsafe_store(c * n_rows + r, tile[tx * (TR_TILE + 1) + lc])
        barrier()
        rt += Int(grid_dim.y)
