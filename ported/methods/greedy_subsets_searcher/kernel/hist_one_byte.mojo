"""The one-byte histogram kernel: 4 features per 4-byte load, 32 to 256 bins.

PORT OF `hist_one_byte.cu` and the base it derives from,
`hist_2_one_byte_base.cuh`, at CatBoost `54a8143a`. Transliterated. Do not
improve.

This is the odd one of the three and it does not derive from the other two.
Where binary and half-byte share `TPointHistHalfByteBase` at 16 floats per
thread, this one uses 32 and trades replication for serialization to make
wide features fit.

THE INNER-BITS TRICK, which is the whole file
---------------------------------------------
A one-byte feature has up to 256 bins, far more than a private per-lane slot
can hold. So the bin is split:

    InnerHistBitsCount = Bits - 5
    higherBin = (bin >> 5) & ((1 << InnerHistBitsCount) - 1)
    offset    = 4 * higherBin + f + ((bin & 31) << 5)

The low 5 bits of the bin index the slot directly. The HIGH bits are handled
by running `1 << InnerHistBitsCount` PASSES, where on pass `p` only the lanes
whose `higherBin == p` write:

    for (k = 0; k < (1 << InnerHistBitsCount); ++k) {
        const int pass = ((threadIdx.x >> 2) + k) & mask;
        syncTile.sync();
        if (pass == higherBin) Histogram[offset] += statToAdd;
    }

So a 5-bit feature costs one pass and a 8-bit feature costs eight. **Wide
features are paid for in TIME rather than in shared memory**, which is the
opposite trade from replicating the histogram, and it is why this kernel can
exist at all inside a 32 KB budget.

Their `SliceOffset` differs to match: `1024 * (threadIdx.x / 32)` gives each
warp 1024 floats (32 per lane), and the inner offset is masked by the number
of blocks the inner bits leave.

DEVIATION (PORTING.md 1): CatBoost runs this at `BlockSize = 384`, so
`384 * 32` floats is 49,152 bytes and Apple gives 32,768. `BLOCK_SIZE = 256`
asks for exactly 32,768 and keeps their per-warp slice arithmetic intact,
since 8 warps times 1024 floats is 8192 floats.
"""

from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from mojo_only.kernel_matrix import (
    K_HIST_ONE_BYTE,
    TARGET_COLUMN,
    block_size_for,
    hist_floats_per_thread_for,
)


#: READ FROM THE MATRIX. Theirs is 384; Apple's 32 KB over 32 floats per
#: thread yields 256, which is exactly 32,768 bytes and keeps their per-warp
#: slice arithmetic intact at 8 warps of 1024 floats.
comptime ONE_BYTE_BLOCK_SIZE = block_size_for[K_HIST_ONE_BYTE, TARGET_COLUMN]()

#: `GetHistSize()` = `BlockSize * 32`, the 32 from the matrix.
comptime ONE_BYTE_HIST_SIZE = ONE_BYTE_BLOCK_SIZE * hist_floats_per_thread_for[
    K_HIST_ONE_BYTE
]()

#: `Unroll` for `__CUDA_ARCH__ >= 700`.
comptime ONE_BYTE_UNROLL = 2

comptime LANE_WIDTH = 32


def one_byte_slice_offset[bits: Int](tid: Int) -> Int:
    """`TPointHistOneByte::SliceOffset()`, copied.

        const int warpOffset = 1024 * (threadIdx.x / 32);
        const int blocks = 8 >> InnerHistBitsCount;
        const int innerHistStart =
            threadIdx.x & ((blocks - 1) << (InnerHistBitsCount + 2));
        return warpOffset + innerHistStart;

    `blocks` shrinks as the feature widens: a 5-bit feature gets 8 private
    sub-copies per warp, an 8-bit feature gets 1. That is the same trade as
    the pass loop, seen from the memory side.
    """
    comptime inner_bits = bits - 5
    var warp_offset = 1024 * (tid // LANE_WIDTH)
    comptime blocks = 8 >> inner_bits
    var inner_hist_start = tid & ((blocks - 1) << (inner_bits + 2))
    return warp_offset + inner_hist_start


def one_byte_bin_offset[bits: Int](ci: UInt32, tid: Int, i: Int) -> Int:
    """The slot for iteration `i`, as pure arithmetic (PORTING.md 10).

        int f = (threadIdx.x + i) & 3;
        int bin = (ci >> (24 - 8 * f)) & 255;
        const int higherBin = (bin >> 5) & mask;
        int offset = 4 * higherBin + f + ((bin & 31) << 5);

    Four features per word here, so the rotation is `& 3` rather than `& 7`.
    """
    comptime inner_bits = bits - 5
    comptime mask = (1 << inner_bits) - 1
    var f = (tid + i) & 3
    var bin = Int((ci >> UInt32(24 - 8 * f)) & UInt32(255))
    var higher_bin = (bin >> 5) & mask
    return 4 * higher_bin + f + ((bin & 31) << 5)


def one_byte_higher_bin[bits: Int](ci: UInt32, tid: Int, i: Int) -> Int:
    """`higherBin` alone: which PASS of the inner-bits loop may write."""
    comptime inner_bits = bits - 5
    comptime mask = (1 << inner_bits) - 1
    var f = (tid + i) & 3
    var bin = Int((ci >> UInt32(24 - 8 * f)) & UInt32(255))
    return (bin >> 5) & mask
