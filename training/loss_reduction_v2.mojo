# SPDX-License-Identifier: Apache-2.0
"""NI35 CE token-total numerical version, shared by native host and GPUs.

Only L12 changes. Leaves are consecutive groups of 256 token rows, each
folded from positive zero in ascending order. Levels combine adjacent left
and right leaves, carrying an odd final child without an extra add. Logical
leaves never depend on block/warp width, vendor, dataset or benchmark shape.

256 bounds each owner's dependency chain and input span to 1 KiB. It is a
fixed arithmetic contract, not a shape dispatch threshold. The unchanged L13
then divides the final total by the existing ce_divisor result. Ignored rows
remain explicit positive-zero inputs. No approximation or token is omitted.

Source only: this revision has no compilation, identity, quality or timing
evidence. Default OFF; enabling it deliberately changes the loss profile.
"""
from checks.numerics import ftz

comptime LOSS_TOKEN_TREE_V2_LEAF = 256
comptime LOSS_TOKEN_TREE_V2_PROFILE = "mojolearn.identical.loss.ce.token-tree256.fp32.v2"


@always_inline
def loss_token_tree_v2_add(left: Float32, right: Float32) -> Float32:
    """One rounded, flushed addition; no adjacent product can contract."""
    return ftz(ftz(left) + ftz(right))


@always_inline
def loss_token_tree_v2_leaf_count(count: Int) -> Int:
    return (count + LOSS_TOKEN_TREE_V2_LEAF - 1) // LOSS_TOKEN_TREE_V2_LEAF


def loss_token_tree_v2_leaf(
    values: MutPointer[Float32, MutAnyOrigin], begin: Int, end: Int,
) -> Float32:
    var acc = Float32(0.0)
    for i in range(begin, end):
        acc = loss_token_tree_v2_add(acc, values.unsafe_load(i))
    return acc


def loss_token_tree_v2_host(values: List[Float32], base: Int, count: Int) -> Float32:
    """Native Mojo host total with precisely the device's logical graph.

    Leaves/parents may run in any physical schedule; each output has one
    writer. This scalar owner spells that graph independently of host SIMD.
    """
    if count <= 0:
        return Float32(0.0)
    var leaves = loss_token_tree_v2_leaf_count(count)
    var level = List[Float32](length=leaves, fill=Float32(0.0))
    for leaf in range(leaves):
        var lo = leaf * LOSS_TOKEN_TREE_V2_LEAF
        var hi = min(lo + LOSS_TOKEN_TREE_V2_LEAF, count)
        var acc = Float32(0.0)
        for i in range(lo, hi):
            acc = loss_token_tree_v2_add(acc, values[base + i])
        level[leaf] = acc
    var active = leaves
    while active > 1:
        var parents = (active + 1) // 2
        for parent in range(parents):
            var left = parent * 2
            var value = level[left]
            if left + 1 < active:
                value = loss_token_tree_v2_add(value, level[left + 1])
            # Ascending parent order is safe in place: each source index
            # is >= its destination, and no later parent reads that slot.
            level[parent] = value
        active = parents
    return level[0]
