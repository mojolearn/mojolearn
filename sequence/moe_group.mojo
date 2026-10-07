# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The MoE pairs grouped by expert ON THE DEVICE, for any expert count
(lane cgr5-owed, 2026-10-03). It replaced `moe_forward_run`'s host round
trip (synchronize, download the picks, counting sort over T * k pairs on the
host, four uploads) on every column the tiled products run on.

`moe_group_count_kernel` counts the pairs per expert (integer atomic adds,
exact), `moe_group_offsets_kernel` gives each expert its first pair `poff[e]`
and its first block of the hidden and the out products (`boff_h[e]`,
`boff_o[e]`, ceil(count / TILE_P) x tiles per expert), one thread per expert
over the E counts below it, and `moe_group_scatter_kernel` gives every pair a
slot of its expert's range by an atomic cursor. The slot order inside an
expert varies run to run; no output cell's chain reads another pair (the
products write `h[pair]`, `s[pair]`; `moe_combine_kernel` sums the picks in
pick order), so the words do not. The products' grids are the upper bound
(T k / TILE_P + E) x tiles; a block past its expert's pairs writes nothing.
"""
from std.gpu import block_dim, block_idx, thread_idx
from std.atomic import Atomic
from std.sys.compile import is_defined
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

# NN44: deterministic ascending-pair grouping; OFF and unmeasured. Each
# expert scans the pairs once, replacing contended atomic scatter cursors.
# O(E*P) comparisons is a real cost; measure skew and full MoE operations.
# L11 (2026-10-07): NN44's grouping and NI53 launched the same stable kernel;
# one switch, MOJOLEARN_IDN_MOE_STABLE_PACK (the NN44_STABLE_GROUP define is
# retired). NN44_EXPERT_BISECT (sequence/moe_tiled.mojo) stays independent.
comptime NN44_STABLE_GROUP = False

from sequence.ops import FP
from sequence.moe_tiled import TILE_P

comptime MOE_GROUP_TPB = 256

# NI53: independent stable-packing A/B, default OFF. One expert owns its pair
# list; it scans pair IDs ascending, writes every routed pair once, and never
# changes capacity or drops a token. Counts/offsets retain exact integer sums.
# This removes scatter atomics and makes packing reproducible; it trades O(E P)
# cheap expert comparisons for atomics, so whole-workload benefit is unproven.
# Host item execution is unaffected because it consumes original pair IDs.
# This S subarm does not claim a MoE backward/training path that does not exist.
comptime MOE_STABLE_PACK = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and is_defined["MOJOLEARN_IDN_MOE_STABLE_PACK"]()
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)


def moe_group_blocks(n_pairs: Int, n_experts: Int, n_tiles: Int) -> Int:
    """The products' grid upper bound: sum over e of ceil(c_e / TILE_P) x
    tiles <= (n_pairs / TILE_P + E) x tiles."""
    return (n_pairs // TILE_P + n_experts) * n_tiles


def moe_group_zero_all_kernel(cnt: FP, n_words: Int32):
    """counts[0..E) and cursors[E..2E) to zero (int32 words)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_words):
        cnt.bitcast[Int32]().unsafe_store(i, Int32(0))


def moe_group_count_all_kernel(sel: FP, cnt: FP, n_pairs: Int32):
    """counts[e] += 1 for each pair's expert, one thread per pair."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_pairs):
        var e = Int(sel.unsafe_load(i))
        _ = Atomic.fetch_add(cnt.bitcast[Int32]() + e, Int32(1))


def moe_group_offsets_all_kernel(
    cnt: FP, poff: FP, boff_h: FP, boff_o: FP, n_experts: Int32, n_ftiles: Int32, n_dtiles: Int32,
):
    """Thread e in [0, E]: poff[e] = the pairs of the experts below e;
    boff_h[e] / boff_o[e] = their blocks (ceil(c / TILE_P) x tiles each)."""
    var e = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if e > Int(n_experts):
        return
    var ci = cnt.bitcast[Int32]()
    var s = 0
    var tiles = 0
    for j in range(e):
        var c = Int(ci.unsafe_load(j))
        s += c
        tiles += (c + TILE_P - 1) // TILE_P
    poff.unsafe_store(e, Float32(s))
    boff_h.unsafe_store(e, Float32(tiles * Int(n_ftiles)))
    boff_o.unsafe_store(e, Float32(tiles * Int(n_dtiles)))


def moe_group_scatter_all_kernel(sel: FP, cnt: FP, poff: FP, order: FP, n_pairs: Int32, n_experts: Int32):
    """order[poff[e] + slot] = pair, the slot by atomic add on e's cursor."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_pairs):
        var e = Int(sel.unsafe_load(i))
        var slot = Atomic.fetch_add(cnt.bitcast[Int32]() + Int(n_experts) + e, Int32(1))
        order.unsafe_store(Int(poff.unsafe_load(e)) + Int(slot), Float32(i))


def moe_group_stable_all_kernel(sel: FP, poff: FP, order: FP, n_pairs: Int32, n_experts: Int32):
    """One expert per task. Integer membership only; pair outputs and
    weighted combine remain in their original token/pick positions."""
    var e = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if e >= Int(n_experts):
        return
    var dst = Int(poff.unsafe_load(e))
    for pair in range(Int(n_pairs)):
        if Int(sel.unsafe_load(pair)) == e:
            order.unsafe_store(dst, Float32(pair))
            dst += 1
