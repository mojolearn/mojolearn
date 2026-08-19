"""A stable 1-bit partition per leaf range, producing a gather map.

NO CATBOOST COUNTERPART, in the same sense `fixed_point.mojo` has none:
CatBoost gets this from a third-party library we do not have. They call
`cub::DeviceRadixSort::SortPairs` once per leaf in a host loop
(`split_points.cu:658-689`), 255 launches for a depth-8 tree, and their own
comments call it wrong:

    //TODO(noxoomo): cub sucks for this, write proper segmented version
    //TODO(noxoomo): for oblivious trees we have overhead for launching
      kernel per leaf

There is no CUB in Mojo, and a full radix sort is not what the algorithm
needs anyway: the key is ONE BIT, the split flag, so a stable partition is
sufficient and is what a "proper segmented version" would be. So this is not
a shortcut around their design, it is the thing their TODO asks for.

WHAT IT COMPUTES
----------------
For each leaf's range, `map[i]` is the SOURCE position of the row that
belongs at destination position `i`, with every flag-0 row before every
flag-1 row and original order preserved within each side. Stability is not
optional: `update_partitions_after_split_kernel` finds the boundary by
scanning for the single transition in the sorted flags, so a non-stable
partition still has one transition but permutes rows within a side, which
would silently change which rows a leaf holds without changing any count.

ONE BLOCK PER LEAF, chunked
---------------------------
Grid y is the leaf, exactly as every other kernel here, so all leaves are one
launch and CatBoost's per-leaf host loop disappears. A leaf larger than the
block is processed in chunks with a running count of zeros carried between
them, which is what keeps the result stable across chunk boundaries.
"""

from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.primitives.block import broadcast as block_broadcast
from max.gpu.primitives.block import prefix_sum as block_prefix_sum
from max.gpu.sync import barrier


comptime PARTITION_BLOCK = 256


def stable_partition_kernel(
    leaves: MutPointer[UInt32, MutAnyOrigin],
    part_offset: MutPointer[UInt32, MutAnyOrigin],
    part_size: MutPointer[UInt32, MutAnyOrigin],
    flags: MutPointer[UInt8, MutAnyOrigin],
    gather_map: MutPointer[UInt32, MutAnyOrigin],
    sorted_flags: MutPointer[UInt8, MutAnyOrigin],
):
    """Stable partition of one leaf's range by its 1-bit split flag.

    Writes `gather_map` (destination -> source, leaf-relative) and
    `sorted_flags` (the flags in destination order), the second because
    `update_partitions_after_split_kernel` scans it for the boundary and must
    not have to re-derive it.

    The two-pass shape is forced by needing the TOTAL zero count before any
    flag-1 row knows its destination: pass one counts zeros over the whole
    leaf, pass two places every row. A single-pass version would need the
    total up front, which is the thing being computed.
    """
    var leaf_id = Int(leaves.unsafe_load(Int(block_idx.y)))
    var offset = Int(part_offset.unsafe_load(leaf_id))
    var size = Int(part_size.unsafe_load(leaf_id))
    var tid = Int(thread_idx.x)

    var shared_total = stack_allocation[
        1, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()

    # --- pass one: total zeros in this leaf ------------------------------
    var zeros = 0
    var chunk = 0
    while chunk < size:
        var v = Int32(0)
        if chunk + tid < size:
            if flags.unsafe_load(offset + chunk + tid) == UInt8(0):
                v = Int32(1)
        var inc = block_prefix_sum[
            block_size=PARTITION_BLOCK, exclusive=False
        ](v)
        var total = block_broadcast[block_size=PARTITION_BLOCK](
            inc, src_thread = PARTITION_BLOCK - 1
        )
        zeros += Int(total)
        chunk += PARTITION_BLOCK
    if tid == 0:
        shared_total[0] = Int32(zeros)
    barrier()
    var n_zeros = Int(shared_total[0])

    # --- pass two: place every row ---------------------------------------
    # `left_seen` and `right_seen` carry across chunks, which is what makes
    # the partition stable rather than merely correct in aggregate.
    var left_seen = 0
    var right_seen = 0
    chunk = 0
    while chunk < size:
        var i = chunk + tid
        var is_zero = Int32(0)
        var in_range = i < size
        if in_range:
            if flags.unsafe_load(offset + i) == UInt8(0):
                is_zero = Int32(1)

        var inc_zero = block_prefix_sum[
            block_size=PARTITION_BLOCK, exclusive=False
        ](is_zero)
        var chunk_zeros = block_broadcast[block_size=PARTITION_BLOCK](
            inc_zero, src_thread = PARTITION_BLOCK - 1
        )
        # Exclusive rank within the chunk.
        var rank_zero = Int(inc_zero) - Int(is_zero)

        var is_one = Int32(0)
        if in_range and is_zero == Int32(0):
            is_one = Int32(1)
        var inc_one = block_prefix_sum[
            block_size=PARTITION_BLOCK, exclusive=False
        ](is_one)
        var rank_one = Int(inc_one) - Int(is_one)

        if in_range:
            var dst = 0
            if is_zero == Int32(1):
                dst = left_seen + rank_zero
            else:
                dst = n_zeros + right_seen + rank_one
            gather_map.unsafe_store(offset + dst, UInt32(i))
            sorted_flags.unsafe_store(
                offset + dst, UInt8(0) if is_zero == Int32(1) else UInt8(1)
            )

        left_seen += Int(chunk_zeros)
        var chunk_ones = block_broadcast[block_size=PARTITION_BLOCK](
            inc_one, src_thread = PARTITION_BLOCK - 1
        )
        right_seen += Int(chunk_ones)
        barrier()
        chunk += PARTITION_BLOCK
