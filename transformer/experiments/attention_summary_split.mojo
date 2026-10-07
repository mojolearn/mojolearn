# SPDX-License-Identifier: Apache-2.0
"""Split-KV forward for the NN20 stable-summary attention profile (lane/neural-fusions, L13).

`-D MOJOLEARN_IDN_NN20_SPLIT_KV` (IDENTICAL only, default OFF, requires
`MOJOLEARN_NN20_BALANCED_SUMMARY_TREE`). Integer parameter
`MOJOLEARN_IDN_NN20_SPLIT_KV_LEAVES`, legal 2|4|8|16 (default 4): the number of
32-key NN20 leaves one split owns (a power of two).

WHY. NN20's model forward (`summary_model_forward_kernel`) is one thread per
query row walking every key leaf of the row serially. At batch 1 the row count
is `n_heads * L` (12288 at the board's 6 x 2048), about 86 threads per SM on a
142-SM L40S and 40 per CU on a 304-CU MI325X, and the causal rows near the end
walk 64 leaves each. Splitting the key axis multiplies the independent work by
the number of splits per row and evens the causal tail (flash-decoding's idea).

BITS. Unchanged against NN20 (the switch moves no bit RELATIVE TO NN20; NN20
itself is a profile change against attention v1). NN20 folds the row's leaf
summaries with a binary counter anchored at absolute key 0: leaf `t` is pushed
and merged with the stack while the low bits of `t` are ones, and the unmatched
right fringe is folded low level -> high level at the end. The split is an
ALIGNED block of `G = 2^s` leaves, `[jG, (j+1)G)`:

  * a full block's counter fold is exactly the complete subtree node(s, j) the
    row's counter builds (the low `s` bits of `t` and of `t - jG` agree);
  * the last, partial block of `r < G` leaves folds its fringe low -> high,
    which is what the row's final fringe fold does to those low levels;
  * folding the block summaries with the same counter (eagerly merging the
    trailing ones of the last block's index, then the fringe of the block
    count) yields the same right-nested sequence of merges over the 1-bits of
    `q = tiles // G` with the partial block at the bottom as the row's lazy
    fringe fold does.

So every `merge_summaries` call meets the same two operands in the same order;
leaf summaries and merges are the contract's own functions, imported. Empty
leaves and blocks are the empty summary, a bit-copy identity at every merge,
exactly as in NN20. Host and device columns are untouched (NN20's host path is
the reference). No rule reads a shape: `G` is a fixed parameter and the grid
covers any (rows, keys).
"""
from std.gpu import block_dim, block_idx, thread_idx
from std.sys.compile import is_defined, get_defined_int
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_div

from transformer.experiments.attention_summary_contract import (
    NN20_BALANCED_SUMMARY_TREE, NN20_KEY_LEAF, leaf_summary, merge_summaries,
    _copy_summary, _empty,
)

comptime IDN_NN20_SPLIT_KV = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and NN20_BALANCED_SUMMARY_TREE
    and is_defined["MOJOLEARN_IDN_NN20_SPLIT_KV"]()
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
comptime NN20_SPLIT_LEAVES = get_defined_int["MOJOLEARN_IDN_NN20_SPLIT_KV_LEAVES", 4]()
comptime _SPLIT_TPB = 64


def _levels_for(count: Int) -> Int:
    """`summary_levels`' rule over `count` items: the smallest L with 2^(L-1) >= count."""
    var levels = 1
    var capacity = 1
    while capacity < count:
        capacity *= 2
        levels += 1
    return levels


def _split_levels() -> Int:
    return _levels_for(NN20_SPLIT_LEAVES)


def nn20_split_scratch_floats(rows: Int, tiles: Int, width: Int) -> Int:
    """Per (row, split): the block summary, plus the split's counter stack
    (levels for G leaves) and two temporaries."""
    var blocks = (tiles + NN20_SPLIT_LEAVES - 1) // NN20_SPLIT_LEAVES
    return rows * blocks * (1 + _split_levels() + 2) * (width + 2)


def nn20_merge_scratch_floats(rows: Int, tiles: Int, width: Int) -> Int:
    var blocks = (tiles + NN20_SPLIT_LEAVES - 1) // NN20_SPLIT_LEAVES
    return rows * (_levels_for(blocks) + 2) * (width + 2)


def _fold_fringe(stack: MutPointer[Float32, MutAnyOrigin],
                 mut current: MutPointer[Float32, MutAnyOrigin],
                 mut work: MutPointer[Float32, MutAnyOrigin],
                 count: Int, levels: Int, width: Int):
    """`summary_attention_forward_row`'s final loop: low -> high over the
    1-bits of `count`, merge(stack[level], current)."""
    var fields = width + 2
    var have = False
    for level in range(levels):
        if ((count >> level) & 1) != 0:
            if not have:
                _copy_summary(current, stack + level * fields, width)
                have = True
            else:
                merge_summaries(work, stack + level * fields, current, width)
                var swap = current
                current = work
                work = swap


def _push(stack: MutPointer[Float32, MutAnyOrigin],
          mut current: MutPointer[Float32, MutAnyOrigin],
          mut work: MutPointer[Float32, MutAnyOrigin],
          index: Int, width: Int):
    """`summary_attention_forward_row`'s push of item `index` (in `current`)."""
    var fields = width + 2
    var carry = index
    var level = 0
    while (carry & 1) != 0:
        merge_summaries(work, stack + level * fields, current, width)
        var swap = current
        current = work
        work = swap
        carry >>= 1
        level += 1
    _copy_summary(stack + level * fields, current, width)


def nn20_split_block_kernel(
    q: MutPointer[Float32, MutAnyOrigin], k: MutPointer[Float32, MutAnyOrigin],
    v: MutPointer[Float32, MutAnyOrigin], lo: MutPointer[Int32, MutAnyOrigin],
    hi: MutPointer[Int32, MutAnyOrigin], blocks_out: MutPointer[Float32, MutAnyOrigin],
    scratch: MutPointer[Float32, MutAnyOrigin],
    rows_in: Int32, keys_in: Int32, hd_in: Int32, width_in: Int32,
    qpg_in: Int32, key_origin_in: Int32, scale: Float32,
):
    """One thread per (row, split j): the summary of leaves [jG, (j+1)G)."""
    var keys = Int(keys_in)
    var width = Int(width_in)
    var key_origin = Int(key_origin_in)
    var fields = width + 2
    var tiles = (key_origin + keys + NN20_KEY_LEAF - 1) // NN20_KEY_LEAF
    var blocks = (tiles + NN20_SPLIT_LEAVES - 1) // NN20_SPLIT_LEAVES
    var gid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if gid >= Int(rows_in) * blocks:
        return
    var row = gid // blocks
    var j = gid - row * blocks
    var dst = blocks_out + gid * fields
    var begin = Int(lo.unsafe_load(row))
    var end = Int(hi.unsafe_load(row))
    _empty(dst, width)
    if begin < 0 or begin > end or end > keys:
        return  # the merge kernel records the refusal, as NN20 does
    var t_lo = j * NN20_SPLIT_LEAVES
    var t_hi = min(tiles, t_lo + NN20_SPLIT_LEAVES)
    # A split whose keys lie wholly outside [begin, end) is the empty summary:
    # every leaf in it is empty and every merge of empties is a bit copy.
    var key_first = t_lo * NN20_KEY_LEAF - key_origin
    var key_last = t_hi * NN20_KEY_LEAF - key_origin
    if key_last <= begin or key_first >= end:
        return
    var levels = _split_levels()
    var stack = scratch + gid * (levels + 2) * fields
    var current = stack + levels * fields
    var work = current + fields
    var group = row // Int(qpg_in)
    var hd = Int(hd_in)
    for tile in range(t_lo, t_hi):
        leaf_summary(current, q, k, v, row, group,
            max(begin, tile * NN20_KEY_LEAF - key_origin),
            min(end, min(keys, (tile + 1) * NN20_KEY_LEAF - key_origin)),
            keys, hd, width, scale)
        _push(stack, current, work, tile - t_lo, width)
    _fold_fringe(stack, current, work, t_hi - t_lo, levels, width)
    _copy_summary(dst, current, width)


def nn20_split_merge_kernel(
    lo: MutPointer[Int32, MutAnyOrigin], hi: MutPointer[Int32, MutAnyOrigin],
    blocks_in: MutPointer[Float32, MutAnyOrigin], output: MutPointer[Float32, MutAnyOrigin],
    maxes: MutPointer[Float32, MutAnyOrigin], denoms: MutPointer[Float32, MutAnyOrigin],
    scratch: MutPointer[Float32, MutAnyOrigin], status: MutPointer[Int32, MutAnyOrigin],
    rows_in: Int32, keys_in: Int32, width_in: Int32, key_origin_in: Int32,
):
    """One thread per row: the counter fold over the row's split summaries,
    then `summary_attention_forward_row`'s outputs, line for line."""
    var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if row >= Int(rows_in):
        return
    var keys = Int(keys_in)
    var width = Int(width_in)
    var fields = width + 2
    var begin = Int(lo.unsafe_load(row))
    var end = Int(hi.unsafe_load(row))
    if begin < 0 or begin > end or end > keys:
        status.unsafe_store(row, Int32(1))
        return
    status.unsafe_store(row, Int32(0))
    var tiles = (Int(key_origin_in) + keys + NN20_KEY_LEAF - 1) // NN20_KEY_LEAF
    var blocks = (tiles + NN20_SPLIT_LEAVES - 1) // NN20_SPLIT_LEAVES
    var levels = _levels_for(blocks)
    var stack = scratch + row * (levels + 2) * fields
    var current = stack + levels * fields
    var work = current + fields
    _empty(current, width)
    for j in range(blocks):
        _copy_summary(current, blocks_in + (row * blocks + j) * fields, width)
        _push(stack, current, work, j, width)
    _fold_fringe(stack, current, work, blocks, levels, width)
    var z = current.unsafe_load(1)
    maxes.unsafe_store(row, current.unsafe_load(0))
    denoms.unsafe_store(row, z)
    for d in range(width):
        var value = Float32(0.0)
        if z != Float32(0.0):
            value = ftz(identical_div(current.unsafe_load(2 + d), z))
        output.unsafe_store(row * width + d, value)


def enqueue_nn20_split_forward(
    ctx: DeviceContext,
    mut q: DeviceBuffer[DType.float32], mut k: DeviceBuffer[DType.float32],
    mut v: DeviceBuffer[DType.float32], mut lo: DeviceBuffer[DType.int32],
    mut hi: DeviceBuffer[DType.int32], mut output: DeviceBuffer[DType.float32],
    mut maxes: DeviceBuffer[DType.float32], mut denoms: DeviceBuffer[DType.float32],
    mut status: DeviceBuffer[DType.int32],
    mut block_scratch: DeviceBuffer[DType.float32],
    mut merge_scratch: DeviceBuffer[DType.float32],
    rows: Int, keys: Int, hd: Int, qpg: Int, key_origin: Int, scale: Float32,
) raises:
    """ASYNCHRONOUS; the caller owns both scratch buffers past its wait.
    `block_scratch` holds `nn20_split_scratch_floats(rows, tiles, hd)` floats,
    `merge_scratch` `nn20_merge_scratch_floats(rows, tiles, hd)`."""
    comptime assert (
        NN20_SPLIT_LEAVES == 2 or NN20_SPLIT_LEAVES == 4
        or NN20_SPLIT_LEAVES == 8 or NN20_SPLIT_LEAVES == 16
    ), "MOJOLEARN_IDN_NN20_SPLIT_KV_LEAVES: legal set 2|4|8|16 (a power of two)"
    var tiles = (key_origin + keys + NN20_KEY_LEAF - 1) // NN20_KEY_LEAF
    var blocks = (tiles + NN20_SPLIT_LEAVES - 1) // NN20_SPLIT_LEAVES
    if len(block_scratch) < nn20_split_scratch_floats(rows, tiles, hd):
        raise Error("NN20 split-KV: block scratch too small")
    if len(merge_scratch) < nn20_merge_scratch_floats(rows, tiles, hd):
        raise Error("NN20 split-KV: merge scratch too small")
    if rows * blocks * (hd + 2) > 2147483647:
        raise Error("NN20 split-KV: split index exceeds Int32 kernel ABI")
    # The block summaries sit at the head of `block_scratch`, the per-split
    # counter stacks after them.
    var summaries = rows * blocks * (hd + 2)
    var stacks = block_scratch.unsafe_ptr() + summaries
    var cells = rows * blocks
    ctx.enqueue_function[nn20_split_block_kernel](
        q.unsafe_ptr(), k.unsafe_ptr(), v.unsafe_ptr(), lo.unsafe_ptr(), hi.unsafe_ptr(),
        block_scratch.unsafe_ptr(), stacks,
        Int32(rows), Int32(keys), Int32(hd), Int32(hd), Int32(qpg), Int32(key_origin), scale,
        grid_dim=((cells + _SPLIT_TPB - 1) // _SPLIT_TPB, 1, 1), block_dim=(_SPLIT_TPB, 1, 1),
    )
    ctx.enqueue_function[nn20_split_merge_kernel](
        lo.unsafe_ptr(), hi.unsafe_ptr(), block_scratch.unsafe_ptr(), output.unsafe_ptr(),
        maxes.unsafe_ptr(), denoms.unsafe_ptr(), merge_scratch.unsafe_ptr(), status.unsafe_ptr(),
        Int32(rows), Int32(keys), Int32(hd), Int32(key_origin),
        grid_dim=((rows + _SPLIT_TPB - 1) // _SPLIT_TPB, 1, 1), block_dim=(_SPLIT_TPB, 1, 1),
    )
