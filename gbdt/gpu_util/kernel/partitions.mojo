"""`UpdatePartitionOffsets`: sorted bins to per-bin start offsets.

PORT OF `catboost/cuda/cuda_util/kernel/partitions.cu` at CatBoost
`54a8143a` -- the `TVecOffsetWriter` arm of `UpdatePartitionOffsets`
(`:81-107`, kernel) and the `ui32*` entry that dispatches it (`:155-176`).
The `TPartitionOffsetWriter` arm (writing `TDataPartition` records for the
tree learner) has no caller in this port and is NOT PORTED YET; the tree
path derives its partitions elsewhere.

Who calls this: `TWeightedBinFreqCalcer::VisitEqualUpToPriorFreqCtrs`
(`ctrs/ctr_calcers.h:318`), handing it the scanned segment ids
(`Bins`, ascending) and a `SegmentStarts` buffer of `partCount` entries --
the segment count plus one fake bin whose entry receives the total size so
every segment can read `[starts[s], starts[s+1])`.

The kernel's two walks are the part worth reading before assuming it is a
one-liner, and both are copied exactly:

* AT A BOUNDARY (`bin0 != bin1`) it walks DOWN from `bin0` writing offset
  `i` until it meets the previous position's bin -- which backfills every
  EMPTY bin between two present ones with the next present bin's start.
  At `i == 0` the sentinel `bin1 = 0xFFFFFFFF` ends the walk through
  unsigned wraparound after bin 0 is written; the wrap is the loop's exit
  condition in their code and in this one.
* AT THE LAST POSITION it walks UP from `bin0 + 1` to
  `min(lastBin, partCount)` writing `size` -- the empty-suffix fill that
  gives the fake last bin its end. Under `dont_write_empty_suffix` (their
  `DONT_WRITE_EMPTY_SUFFIX = true` template arm) `lastBin` is the last
  sorted bin, so the walk writes nothing; the dispatcher pre-fills the
  buffer instead.

The dispatcher's `partCount == size` arm (`:165-172`) pre-fills every
entry with `size` and takes the no-suffix kernel; `_fill_u32_kernel` below
is their `FillBuffer` (`cuda_util/fill.cu`) for exactly that call, local
because nothing else here fills a buffer with a non-zero value.

Grid-stride, blockSize 256, no shared memory, no warp intrinsics, no
vendor row.
"""
from max.gpu.host import DeviceBuffer, DeviceContext
from std.gpu import block_dim, block_idx, grid_dim, thread_idx

comptime PARTITIONS_BLOCK_SIZE = 256


def _fill_u32_kernel(
    dst: MutPointer[UInt32, MutAnyOrigin],
    value: UInt32,
    size_in: Int32,
):
    var size = Int(size_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    while i < size:
        dst.unsafe_store(i, value)
        i += stride


def update_partition_offsets_kernel(
    part_offsets: MutPointer[UInt32, MutAnyOrigin],
    part_count_in: Int32,
    sorted_bins: MutPointer[UInt32, MutAnyOrigin],
    size_in: Int32,
    dont_write_empty_suffix: Int32,
):
    # partitions.cu:81-107, TVecOffsetWriter: Write(bin, offset) is
    # part_offsets[bin] = offset.
    var size = Int(size_in)
    var part_count = UInt32(Int(part_count_in))
    var last_bin: UInt32
    if dont_write_empty_suffix != Int32(0):
        last_bin = sorted_bins.unsafe_load(size - 1)
    else:
        last_bin = UInt32(0xFFFFFFFF)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    while i < size:
        var bin0 = sorted_bins.unsafe_load(i)
        var bin1: UInt32
        if i > 0:
            bin1 = sorted_bins.unsafe_load(i - 1)
        else:
            bin1 = UInt32(0xFFFFFFFF)
        if bin0 != bin1:
            var b = bin0
            while b != bin1:
                part_offsets.unsafe_store(Int(b), UInt32(i))
                b -= 1  # wraps past 0 to 0xFFFFFFFF, ending the i==0 walk
        if i + 1 == size:
            var limit = last_bin
            if part_count < limit:
                limit = part_count
            var b = bin0 + 1
            while b < limit:
                part_offsets.unsafe_store(Int(b), UInt32(size))
                b += 1
        i += stride


def launch_update_partition_offsets(
    ctx: DeviceContext,
    mut part_offsets: DeviceBuffer[DType.uint32],
    part_count: Int,
    mut sorted_bins: DeviceBuffer[DType.uint32],
    size: Int,
) raises:
    """The `ui32*` entry (`partitions.cu:155-176`), both arms."""
    var num_blocks = (size + PARTITIONS_BLOCK_SIZE - 1) // (
        PARTITIONS_BLOCK_SIZE
    )
    if num_blocks == 0:
        # ========================= CORRECTED 2026-08-21 =================
        # This used to `return`, with a comment claiming the `ui32` entry
        # "has no work to do on an empty input". That was wrong. THEIR else
        # arm is not empty (`partitions.cu:176`):
        #
        #     } else {
        #         FillBuffer(partOffsets, static_cast<ui32>(0), partCount,
        #                    stream);
        #     }
        #
        # It ZEROES all `partCount` offsets. Returning early leaves them
        # holding whatever was there before, so a caller that reads
        # `[starts[s], starts[s+1])` on an empty input reads garbage
        # instead of a run of empty segments.
        #
        # Unreached today -- the CTR path never passes size 0 -- which is
        # exactly why it survived: a divergence in a branch nothing takes
        # is invisible until something takes it. Found by the lane porting
        # `pointwise_optimization_subsets`, which needs this entry.
        # ================================================================
        ctx.enqueue_function[_fill_u32_kernel](
            part_offsets.unsafe_ptr(),
            UInt32(0),
            Int32(part_count),
            grid_dim=(
                (part_count + PARTITIONS_BLOCK_SIZE - 1)
                // PARTITIONS_BLOCK_SIZE,
                1,
                1,
            ),
            block_dim=(PARTITIONS_BLOCK_SIZE, 1, 1),
        )
        return
    var skip_suffix = part_count == size
    if skip_suffix:
        # their FillBuffer(partOffsets, size, size, stream) (:166)
        ctx.enqueue_function[_fill_u32_kernel](
            part_offsets.unsafe_ptr(), UInt32(size), Int32(part_count),
            grid_dim=(num_blocks, 1, 1),
            block_dim=(PARTITIONS_BLOCK_SIZE, 1, 1),
        )
    ctx.enqueue_function[update_partition_offsets_kernel](
        part_offsets.unsafe_ptr(),
        Int32(part_count),
        sorted_bins.unsafe_ptr(),
        Int32(size),
        Int32(1) if skip_suffix else Int32(0),
        grid_dim=(num_blocks, 1, 1),
        block_dim=(PARTITIONS_BLOCK_SIZE, 1, 1),
    )
