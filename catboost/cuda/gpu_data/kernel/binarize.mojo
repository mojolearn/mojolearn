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
otherwise after PORTING.md item 7.
"""

from std.gpu import block_dim, block_idx, grid_dim, thread_idx


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
