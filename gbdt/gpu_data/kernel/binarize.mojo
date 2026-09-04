# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Write one feature's bins into the packed compressed index.

PORT OF `WriteCompressedIndexImpl` in
`catboost/cuda/gpu_data/kernel/binarize.cu` at CatBoost `54a8143a`.
Transliterated. Do not improve.

**This is the kernel that creates the read-density advantage.** Everything
downstream reads `cindex[feature.Offset + row]` and extracts its feature by
shift and mask, so one 4-byte load serves 32 binary features, 8 half-byte
features or 4 one-byte features. Without this the packing is arithmetic in
`grid_policy` that nothing acts on.

Their whole kernel:

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
one ports directly if we ever need it. Recorded because it is the one place
CatBoost's atomics are portable to us and it would be easy to assume
otherwise after archive/reference/PORTING.md item 7.
"""

from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier


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
comptime BINARIZE_BLOCK_SIZE = 1024
comptime BINARIZE_DOCS_PER_THREAD = 8


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
            if feature_values[j] > border_value:
                index[j] += 1

    @parameter
    for j in range(BINARIZE_DOCS_PER_THREAD):
        var idx = i + j * BINARIZE_BLOCK_SIZE
        if idx < n:
            var bin = dst.unsafe_load(base + idx)
            bin |= (index[j] & feature_mask) << feature_shift
            dst.unsafe_store(base + idx, bin)
