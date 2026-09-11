# SPDX-License-Identifier: Apache-2.0
"""Small-k composite-key selector for the IDENTICAL tiled k-NN arm.

PRODUCTION DISPATCH SINCE 2026-09-09 through the kernel-matrix row
`knn_smallk_select_for` (NVIDIA and AMD by default), for every 1 <= k <= 64
via `smallk_bucket_kernel` / `smallk_select_launch` at the bottom of this
file; the k in {8, 10, 16} specializations and the runtime baseline above
them are kept for the A/B drivers. `partial_topk_merge_kernel`, also at the
bottom, is the index-axis tiling's merge.

For 1 <= k <= 16, each of 256 threads retains its k smallest composite
(distance,index) keys, then the block merges those sorted lists. An element
outside a thread's local top-k cannot belong to the block top-k. Every merge
uses UInt64 minima; there are no floating reductions or atomic tie choices.
The key is the EXISTING selector's key, including its select-max ordering,
signed-zero distinction and NaN payload order. Output values are copied from
the original selected index, preserving their bits. The sentinel cannot be a
real key because accepted row lengths are <= INT32_MAX, below UINT32_MAX.

Caller owns nonoverlapping buffers and retains them through synchronization.
The optional specialized arm fixes K=8/10/16 at compile time, unrolling
insertion and shifts so local SIMD indexing cannot spill via runtime indices.
Other K values use the retained runtime baseline.

`smallk_bucket_kernel` carries three gated candidates behind the selection
trial hook (`-D MOJOLEARN_KNN_SELECT_TRIAL=1`, arms chosen per request from
MOJOLEARN_KNN_SELECT): the block-uniform trip count (DEVIATION 2497, C4),
the block head-bound rejection (DEVIATION 2498, C1, measured NEGATIVE on
the H100 2026-09-11 and kept as that record) and the warp-scope group
bound (DEVIATION 2515, C2). See the hook comment above the kernel; without
the define the shipped kernel is the 2026-09-09 one plus C4.
"""
from std.gpu import block_idx, thread_idx
from std.os import getenv
from std.sys.compile import is_defined
from neighbors.checks.lane_minimum import shuffle_min_u64
from std.gpu.primitives.warp import shuffle_xor
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from checks.kernel_matrix import (
    TARGET_COLUMN,
    knn_selector_shuffle_for,
    knn_selector_specialize_common_for,
    lib_lane_width_for,
)
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from neighbors.checks.select_radix_identical import composite_key

comptime SMALLK_BLOCK = 256
comptime SMALLK_LIMIT = 16


def smallk_identical_kernel(
    values: MutPointer[Float32, MutAnyOrigin],
    out_values: MutPointer[Float32, MutAnyOrigin],
    out_indices: MutPointer[UInt32, MutAnyOrigin],
    length_in: Int32, k_in: Int32, select_min_in: Int32,
):
    var length = Int(length_in)
    var k = Int(k_in)
    var tid = Int(thread_idx.x)
    var row = Int(block_idx.x)
    var sentinel = UInt64(18446744073709551615)
    var local_keys = SIMD[DType.uint64, SMALLK_LIMIT](sentinel)
    var heads = stack_allocation[
        SMALLK_BLOCK, Scalar[DType.uint64], address_space=AddressSpace.SHARED,
    ]()
    var col = tid
    while col < length:
        var pending = composite_key(values.unsafe_load(row * length + col), UInt32(col), select_min_in != 0)
        if pending < local_keys[k - 1]:
            # Carry insertion; a rejected item does no list maintenance.
            for slot in range(k):
                if pending < local_keys[slot]:
                    var previous = local_keys[slot]
                    local_keys[slot] = pending
                    pending = previous
        col += SMALLK_BLOCK
    for rank in range(k):
        var mine = local_keys[0]
        heads[tid] = mine
        barrier()
        var stride = SMALLK_BLOCK // 2
        while stride > 0:
            if tid < stride:
                var other = heads[tid + stride]
                if other < heads[tid]:
                    heads[tid] = other
            barrier()
            stride //= 2
        var winner = heads[0]
        # No thread may overwrite shared heads before every thread reads it.
        barrier()
        if tid == 0:
            var selected = UInt32(winner & UInt64(4294967295))
            out_indices.unsafe_store(row * k + rank, selected)
            out_values.unsafe_store(row * k + rank, values.unsafe_load(row * length + Int(selected)))
        if mine == winner:
            for slot in range(k - 1):
                local_keys[slot] = local_keys[slot + 1]
            local_keys[k - 1] = sentinel
        barrier()


# Specialized counterpart preserves the baseline kernel above for A/B gates.
def smallk_specialized_kernel[K: Int](
    values: MutPointer[Float32, MutAnyOrigin],
    out_values: MutPointer[Float32, MutAnyOrigin],
    out_indices: MutPointer[UInt32, MutAnyOrigin],
    length_in: Int32, k_in: Int32, select_min_in: Int32,
):
    var length = Int(length_in)
    # K is fixed at compile time; every SIMD lane access is constant.
    var k = K
    var tid = Int(thread_idx.x)
    var row = Int(block_idx.x)
    var sentinel = UInt64(18446744073709551615)
    var local_keys = SIMD[DType.uint64, SMALLK_LIMIT](sentinel)
    var heads = stack_allocation[
        SMALLK_BLOCK, Scalar[DType.uint64], address_space=AddressSpace.SHARED,
    ]()
    var col = tid
    while col < length:
        var pending = composite_key(values.unsafe_load(row * length + col), UInt32(col), select_min_in != 0)
        if pending < local_keys[K - 1]:
            # Carry insertion; a rejected item does no list maintenance.
            comptime for slot in range(K):
                if pending < local_keys[slot]:
                    var previous = local_keys[slot]
                    local_keys[slot] = pending
                    pending = previous
        col += SMALLK_BLOCK
    for rank in range(k):
        var mine = local_keys[0]
        heads[tid] = mine
        barrier()
        var stride = SMALLK_BLOCK // 2
        while stride > 0:
            if tid < stride:
                var other = heads[tid + stride]
                if other < heads[tid]:
                    heads[tid] = other
            barrier()
            stride //= 2
        var winner = heads[0]
        # No thread may overwrite shared heads before every thread reads it.
        barrier()
        if tid == 0:
            var selected = UInt32(winner & UInt64(4294967295))
            out_indices.unsafe_store(row * k + rank, selected)
            out_values.unsafe_store(row * k + rank, values.unsafe_load(row * length + Int(selected)))
        if mine == winner:
            comptime for slot in range(K - 1):
                local_keys[slot] = local_keys[slot + 1]
            local_keys[K - 1] = sentinel
        barrier()


def smallk_identical_into(
    ctx: DeviceContext, mut values: DeviceBuffer[DType.float32],
    mut out_values: DeviceBuffer[DType.float32], mut out_indices: DeviceBuffer[DType.uint32],
    rows: Int, length: Int, k: Int, select_min: Bool = True, specialized: Bool = False,
) raises:
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("small-k candidate requires IDENTICAL")
    if rows <= 0 or rows > 2147483647 or length <= 0 or length > 2147483647:
        raise Error("small-k requires positive Int32 dimensions")
    if k < 1 or k > SMALLK_LIMIT or k > length:
        raise Error("small-k candidate supports only 1 <= k <= min(16, length)")
    if len(values) < rows * length or len(out_values) < rows * k or len(out_indices) < rows * k:
        raise Error("small-k buffer capacity is insufficient")
    if specialized:
        if k == 8:
            ctx.enqueue_function[smallk_specialized_kernel[8]](
                values.unsafe_ptr(), out_values.unsafe_ptr(), out_indices.unsafe_ptr(),
                Int32(length), Int32(k), Int32(select_min),
                grid_dim=(rows, 1, 1), block_dim=(SMALLK_BLOCK, 1, 1),
            )
            return
        elif k == 10:
            ctx.enqueue_function[smallk_specialized_kernel[10]](
                values.unsafe_ptr(), out_values.unsafe_ptr(), out_indices.unsafe_ptr(),
                Int32(length), Int32(k), Int32(select_min),
                grid_dim=(rows, 1, 1), block_dim=(SMALLK_BLOCK, 1, 1),
            )
            return
        elif k == 16:
            ctx.enqueue_function[smallk_specialized_kernel[16]](
                values.unsafe_ptr(), out_values.unsafe_ptr(), out_indices.unsafe_ptr(),
                Int32(length), Int32(k), Int32(select_min),
                grid_dim=(rows, 1, 1), block_dim=(SMALLK_BLOCK, 1, 1),
            )
            return
    # Other k values retain the baseline runtime implementation.
    ctx.enqueue_function[smallk_identical_kernel](
        values.unsafe_ptr(), out_values.unsafe_ptr(), out_indices.unsafe_ptr(),
        Int32(length), Int32(k), Int32(select_min),
        grid_dim=(rows, 1, 1), block_dim=(SMALLK_BLOCK, 1, 1),
    )


# ---------------------------------------------------------------------------
# The generalized selector, 2026-09-09: every 1 <= k <= SMALLK_MAX_K.
#
# Same algorithm as `smallk_specialized_kernel` with the per-thread capacity a
# comptime bucket (16 / 32 / 64) and k a runtime argument, so three
# instantiations cover the whole band instead of one per k. Every list index
# is a comptime constant, so the local list stays in registers. The block
# merge pops one winner per rank through a shared-memory min tree; both the
# min and the pop are exact integer operations on unique keys, so the output
# is the k smallest composite keys ascending, which is what the radix
# selector's rank pass writes too.
# ---------------------------------------------------------------------------

comptime SMALLK_MAX_K = 64

# THE BLOCK MINIMUM, 2026-09-09 (lane/knn-selector; kernel-matrix row
# `knn_selector_shuffle_for`). The rank loop needs the block's smallest head
# key k times. The tree form pops it through log2(256) = 8 shared-memory
# levels with a barrier at each, eleven barriers a rank. The butterfly form
# folds each lane group with `shuffle_xor` (no barrier, every lane ends
# holding the group minimum), parks one key per group in shared memory,
# takes ONE barrier, and every thread reads the SMALLK_WARPS group minima
# itself. The slots are double-buffered by rank parity so that one barrier
# per rank is also enough to keep a fast group's next write off a slow
# thread's current read. A UInt64 minimum is associative, commutative and
# idempotent, so the winner is the same key whichever tree finds it.
#
# The key is 64 bits and the shuffle primitive is taken at 32, so a key
# crosses lanes as two halves. Every lane reaches every shuffle: a lane that
# skipped one would hang its group.
comptime SMALLK_SHUFFLE = knn_selector_shuffle_for[
    TARGET_COLUMN, GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
]()
comptime SMALLK_LANES = lib_lane_width_for[TARGET_COLUMN]()
comptime SMALLK_WARPS = SMALLK_BLOCK // SMALLK_LANES


comptime SMALLK_SCAN_UNROLL = 8
comptime SMALLK_SCAN_SPAN = SMALLK_SCAN_UNROLL * SMALLK_BLOCK


# ---------------------------------------------------------------------------
# THE SELECTION TRIAL HOOK (DEVIATION 2497 and 2498, 2026-09-11; brief
# `docs/lanes/BRIEF_knn_selection_2026-09-10.md` section 4).
#
# `-D MOJOLEARN_KNN_SELECT_TRIAL=1` (never on a shipped build) compiles every
# arm of `smallk_bucket_kernel` and lets the HOST pick one per request from
# the environment. `smallk_select_arm_from_env` is read ONCE per request by
# `_tiled_brute_force_knn_impl` (`neighbors/impl/detail/knn_brute_force.mojo`)
# and the result is passed down to `smallk_select_launch` as `arm`, the
# `MOJOLEARN_KNN_VECTOR_TRIAL` pattern:
#
#   MOJOLEARN_KNN_SELECT=baseline    today's kernel: per-thread trip count,
#                                    no bound
#   MOJOLEARN_KNN_SELECT=uniform     C4 only: block-uniform trip count
#                                    (DEVIATION 2497), no bound
#   MOJOLEARN_KNN_SELECT=headbound   C4 + C1 block head-bound rejection
#                                    (DEVIATION 2498; NEGATIVE on the H100,
#                                    kept as the measured record)
#   MOJOLEARN_KNN_SELECT=warpbound   C4 + C2 warp-scope group bound
#                                    (DEVIATION 2515): shuffles only, no
#                                    barrier, no shared memory
#   unset or empty                   the build default, SMALLK_ARM_DEFAULT
#   anything else                    RAISES; the gate harness relies on it
#   MOJOLEARN_KNN_SELECT_SABOTAGE=1  the chosen arm's SABOTAGE instantiation
#                                    (reach proof; see the kernel)
#
# WITHOUT THE DEFINE NONE OF THIS EXISTS: `smallk_select_arm_from_env`
# returns SMALLK_ARM_DEFAULT without touching the environment, the launch
# refuses any other arm, and the only instantiations in the binary are
# `smallk_bucket_kernel[CAP, K, SMALLK_UNIFORM_TRIP_DEFAULT,
# SMALLK_HEAD_BOUND_DEFAULT, False, SMALLK_WARPBOUND_DEFAULT]`. With the two
# bound defaults False (the state until a bound gate passes) that is the
# [CAP, K] kernel of 2026-09-09 under C4's trip count: the `comptime if`
# arms below fold away and the non-trial code path is the one that shipped.
#
# THE DEFAULTS. Three comptime switches here rather than kernel-matrix rows,
# because this lane may not edit `checks/kernel_matrix.mojo`; the flip that
# promotes an arm moves them into a SCHEDULING row
# (`knn_selector_head_bound_for[column, identical]`, brief section 4) in
# the same session as the measured win. Order of flips: UNIFORM first
# (gated alone, arms baseline,uniform, equality on every fixture including
# `divergent_tail`; DONE 2026-09-11), then ONE bound arm (arms
# baseline,<bound arm>, equality plus the request-level price). Every bound
# arm requires UNIFORM; HEAD_BOUND and WARPBOUND exclude each other.
# ---------------------------------------------------------------------------
comptime SMALLK_SELECT_TRIAL = is_defined["MOJOLEARN_KNN_SELECT_TRIAL"]()
comptime SMALLK_UNIFORM_TRIP_DEFAULT = True  # DEVIATION 2497: flipped 2026-09-11 on the H100 and M4 gates
comptime SMALLK_HEAD_BOUND_DEFAULT = False  # DEVIATION 2498: NEGATIVE on the H100 2026-09-11, stays off
comptime SMALLK_WARPBOUND_DEFAULT = False  # DEVIATION 2515: RUN OWED (brief, "Implementation pass, C2")

comptime SMALLK_ARM_BASELINE = 0
comptime SMALLK_ARM_UNIFORM = 1
comptime SMALLK_ARM_HEADBOUND = 2
comptime SMALLK_ARM_WARPBOUND = 3
# OR'd into the arm value; the launch strips it.
comptime SMALLK_ARM_SABOTAGE = 16
comptime SMALLK_ARM_DEFAULT = SMALLK_ARM_HEADBOUND if SMALLK_HEAD_BOUND_DEFAULT else (
    SMALLK_ARM_WARPBOUND if SMALLK_WARPBOUND_DEFAULT else (
        SMALLK_ARM_UNIFORM if SMALLK_UNIFORM_TRIP_DEFAULT else SMALLK_ARM_BASELINE
    )
)


def smallk_select_arm_from_env() raises -> Int:
    """The selector arm for THIS request, read once on the host.

    Trial builds read `MOJOLEARN_KNN_SELECT` (baseline / uniform / headbound /
    warpbound, unset = the build default, anything else raises) and
    `MOJOLEARN_KNN_SELECT_SABOTAGE` (exactly "1" sets the SMALLK_ARM_SABOTAGE
    bit). Every other build returns SMALLK_ARM_DEFAULT without reading the
    environment at all.
    """
    comptime if not SMALLK_SELECT_TRIAL:
        return SMALLK_ARM_DEFAULT
    var name = String(getenv("MOJOLEARN_KNN_SELECT"))
    var arm: Int
    if name == "":
        arm = SMALLK_ARM_DEFAULT
    elif name == "baseline":
        arm = SMALLK_ARM_BASELINE
    elif name == "uniform":
        arm = SMALLK_ARM_UNIFORM
    elif name == "headbound":
        arm = SMALLK_ARM_HEADBOUND
    elif name == "warpbound":
        arm = SMALLK_ARM_WARPBOUND
    else:
        raise Error(
            "MOJOLEARN_KNN_SELECT='" + name
            + "' is not a selector arm (baseline, uniform, headbound, warpbound, or unset)"
        )
    if String(getenv("MOJOLEARN_KNN_SELECT_SABOTAGE")) == "1":
        arm = arm | SMALLK_ARM_SABOTAGE
    return arm


@always_inline
def _smallk_insert[CAP: Int](
    mut local_keys: SIMD[DType.uint64, CAP], mut threshold: UInt64,
    pending_in: UInt64, k: Int,
):
    """Carry-insert one key below `threshold` into the ascending local list
    (slots `0 .. k-1` of CAP) and refresh `threshold = local_keys[k - 1]`.
    Every list index is a comptime constant, so the list stays in
    registers."""
    var pending = pending_in
    comptime for slot in range(CAP):
        if slot < k:
            if pending < local_keys[slot]:
                var previous = local_keys[slot]
                local_keys[slot] = pending
                pending = previous
    comptime for slot in range(CAP):
        if slot == k - 1:
            threshold = local_keys[slot]


# C1's refresh cadence, in COMPLETED batches of SMALLK_SCAN_SPAN columns:
# after batch 1 (2,048 columns seen, every lane holds min(k, 8) real keys,
# so every published head is real), then after 4, 12 and 28 (8,192, 24,576
# and 57,344 columns). A 65,536-column partition has 32 batches and takes
# all four; the 6,784-column last partition of 400k has 3 and takes the
# first; the carved k-wide tail partition has none and is the baseline
# kernel. The cadence is a scheduling choice, not an order-sensitive one:
# any bound that is the k-th smallest of a subset of the union's keys at
# any moment is admissible (the argument at the refresh below).
@always_inline
def _smallk_bound_refresh_due(done: Int) -> Bool:
    return done == 1 or done == 4 or done == 12 or done == 28


# ---------------------------------------------------------------------------
# C2 (DEVIATION 2515): the warp-scope group bound.
#
# THE BOUND. Every lane publishes its `depth`-th smallest key (depth =
# ceil(k / LANES), so its head for k <= LANES). The warp is cut into
# LANES / group aligned groups of `group` lanes, `group` the largest power
# of two with (LANES / group) * depth >= k; each group takes the MINIMUM of
# its published keys, and the bound is the MAXIMUM over the groups. Both
# folds are xor butterflies on the 64-bit key taken as two 32-bit halves
# (the rank phase's `shuffle_min_u64` shape), five steps on 32 lanes, no
# barrier, no shared memory. For k = 10 and 15 on 32 lanes: depth 1, group
# 2, sixteen pairs.
#
# WHY IT IS AN UPPER BOUND ON THE WARP UNION'S k-TH SMALLEST. A group's
# minimum m_q is the published key of some lane L_q of that group, and L_q
# holds `depth` keys <= m_q (its `depth` smallest). The bound B is >= every
# m_q, so every L_q holds `depth` keys <= B; the L_q are distinct lanes
# (groups are disjoint) and no key lives in two lanes (each column is
# scanned by one lane), so at least (LANES / group) * depth >= k distinct
# keys of the warp's union are <= B. The warp's union is a subset of the
# block's, so the block union has at least k keys <= B as well: a pending
# key at or above B is not among the row's k smallest. The rest of C1's
# argument (the count never drops, equality is impossible because keys carry
# their column, the rank phase pops exact minima of a union that still
# holds the true top-k) is unchanged; see the refresh in the kernel.
#
# WHY THIS BOUND AND NOT THE WARP MINIMUM OF THE LANES' k-TH KEYS. That one
# is valid too (the lane attaining the minimum holds k keys at or below it)
# and costs the same five steps, but a lane's k-th key sits near the k/i
# quantile after i elements (Gamma(k)/i, spread sqrt(k)/i) and the minimum
# over 32 such is still near 4.7/i at k10 and 8.2/i at k15, which leaves
# the warp-level "some lane inserts" probability near 1 for most of a
# 256-element lane sequence; the event model in the brief gives it 0.88x
# (k10) to 0.94x (k15) of the scan's instructions. The group bound sits
# near 1.7/i at k <= 16 (the maximum over 16 pairs of the pair's minimum
# head, independent of k) and models at 0.62x / 0.55x with the
# every-second-batch cadence. The k-th smallest of the 32 heads (k removal
# rounds) would sit near 0.4/i (k10) to 0.6/i (k15) but costs k times the
# shuffles; C1's measured loss says the refresh price, not the bound's
# tightness, is what decides.
#
# THE CADENCE. `SMALLK_WARPBOUND_EVERY` completed batches, starting at the
# first batch count after which every lane holds k keys (ceil(k / 8): 2 for
# k in 9..16), so every published key is real and the lane's own threshold
# is real too. Before that, and in every partition with fewer full batches
# than that (fewer than 4,096 columns for k in 9..16: the 3,940-column
# `divergent_tail` remainder and the carved k-wide tail), `bound` stays the
# sentinel and min(threshold, sentinel) = threshold: the uniform arm by
# construction. Two is the default because one refresh is about 45
# instructions, under one insertion chain at k10 (about 60) and half of one
# at k15 (about 90): the model's saving from refreshing every batch instead
# of every second is three events (about 180 instructions) against sixteen
# more refreshes (about 700), and every fourth batch is flat at k10 and one
# percent worse at k15. The cadence is a scheduling choice: any bound that
# is an upper bound on the union's k-th smallest at any moment is
# admissible, so the parameter is free to move without an order argument.
# ---------------------------------------------------------------------------
comptime SMALLK_WARPBOUND_EVERY = 2


@always_inline
def _smallk_shuffle_xor_u64(value: UInt64, offset: Int) -> UInt64:
    var hi = shuffle_xor(UInt32(value >> UInt64(32)), UInt32(offset))
    var lo = shuffle_xor(UInt32(value & UInt64(0xFFFFFFFF)), UInt32(offset))
    return (UInt64(hi) << UInt64(32)) | UInt64(lo)


@always_inline
def _smallk_warp_group_bound[LANES: Int](published: UInt64, group: Int) -> UInt64:
    """Maximum over the LANES / group aligned lane groups of each group's
    minimum published key. Every lane of the warp must call this
    convergently (`group` and the trip counts are warp-uniform: they derive
    from k and the partition length alone) and every lane returns the same
    value. Integer min and max are associative, commutative and idempotent,
    so the result does not depend on the butterfly's step order."""
    var v = published
    var offset = 1
    while offset < group:
        var other = _smallk_shuffle_xor_u64(v, offset)
        if other < v:
            v = other
        offset *= 2
    while offset < LANES:
        var other = _smallk_shuffle_xor_u64(v, offset)
        if other > v:
            v = other
        offset *= 2
    return v


@always_inline
def _smallk_warpbound_refresh_due(done: Int, fill: Int) -> Bool:
    return done >= fill and (done - fill) % SMALLK_WARPBOUND_EVERY == 0


def smallk_bucket_kernel[
    CAP: Int, K: Int = 0, UNIFORM: Bool = False, BOUND: Bool = False, SABOTAGE: Bool = False,
    WARPBOUND: Bool = False,
](
    values: MutPointer[Float32, MutAnyOrigin],
    out_values: MutPointer[Float32, MutAnyOrigin],
    out_indices: MutPointer[UInt32, MutAnyOrigin],
    length_in: Int32, k_in: Int32, select_min_in: Int32,
):
    """The small-k selector, one block of SMALLK_BLOCK threads per row.

    UNIFORM (DEVIATION 2497, C4): the unrolled scan's batch trip count is
    `length // SMALLK_SCAN_SPAN`, derived from the partition length alone
    and therefore identical on every thread of the block, so a block
    collective inside the batch loop is legal. False is the 2026-09-09
    per-thread condition `col + 7 * 256 < length`.
    BOUND (DEVIATION 2498, C1): at `_smallk_bound_refresh_due` points the
    block computes the k-th smallest of its 256 list heads and every lane
    rejects pending keys at or above min(own k-th, that bound). Requires
    UNIFORM (the refresh has a barrier in the loop). Measured NEGATIVE on
    the H100 (2026-09-11); kept as the record.
    WARPBOUND (DEVIATION 2515, C2): every SMALLK_WARPBOUND_EVERY batches
    once the lists are full, each warp folds its lanes' heads into the
    group bound described above `_smallk_warp_group_bound` with shuffles
    only, and every lane of that warp rejects pending keys at or above
    min(own k-th, that bound). Requires UNIFORM (the shuffles must be
    convergent) and a fixed-lane-width column (SMALLK_SHUFFLE); excludes
    BOUND.
    SABOTAGE: reach proof for the gate, per arm. Never on by default.
    """
    comptime assert UNIFORM or not BOUND, "C1's in-loop barrier needs C4's block-uniform trip count"
    comptime assert UNIFORM or not WARPBOUND, "C2's in-loop shuffles need C4's block-uniform trip count"
    comptime assert not (BOUND and WARPBOUND), "one bound arm at most"
    var length = Int(length_in)
    var k = K if K > 0 else Int(k_in)
    var tid = Int(thread_idx.x)
    var row = Int(block_idx.x)
    var sentinel = UInt64(18446744073709551615)
    var local_keys = SIMD[DType.uint64, CAP](sentinel)
    # `local_keys[k - 1]`, kept in its own register so the reject test never
    # indexes the list at a runtime position.
    var threshold = sentinel
    var heads = stack_allocation[
        SMALLK_BLOCK, Scalar[DType.uint64], address_space=AddressSpace.SHARED,
    ]()
    var select_min = select_min_in != 0
    var base = row * length
    var col = tid
    # C1 state. `bound` is the k-th smallest of the 256 heads published at
    # the last refresh (the sentinel before the first one and when fewer
    # than k real heads exist); `gate = min(threshold, bound)` is the reject
    # test. `rounds` counts every block-minimum round taken so far
    # (refreshes and ranks) so the butterfly's parity double-buffering
    # carries across the refresh / rank boundary: two consecutive rounds
    # never write the same page, and a page is rewritten only after the
    # barrier of the round that followed its last read. Under BOUND=False
    # the three are constants and fold away.
    var bound = sentinel
    var gate = sentinel
    var rounds = 0
    # C2 state (constants under comptime K; folded away unless WARPBOUND).
    # `wb_fill`: batches after which every lane holds k keys; `wb_depth`:
    # the published slot + 1; `wb_group`: lanes per group. See the comment
    # above `_smallk_warp_group_bound`.
    var wb_fill = (k + SMALLK_SCAN_UNROLL - 1) // SMALLK_SCAN_UNROLL
    var wb_depth = (k + SMALLK_LANES - 1) // SMALLK_LANES
    var wb_group = 1
    while (SMALLK_LANES // (wb_group * 2)) * wb_depth >= k:
        wb_group *= 2
    # THE SCAN, unrolled SMALLK_SCAN_UNROLL loads deep (2026-09-09). The
    # loop body is a load, a key, a compare and a rarely taken insertion;
    # written one element at a time, each iteration waits for its own
    # global load before the next is issued. The unrolled form issues
    # SMALLK_SCAN_UNROLL independent loads first and then keys and inserts
    # them in ascending column order, the same order and the same insertion
    # as the scalar tail below, so every thread's local list is the list it
    # always was. (The "latency-bound" reading of the 2026-09-09 comment is
    # superseded by the H100 profile of 2026-09-11: 15.4 us per launch per
    # unit of k, so the scan's insertion chain is the cost; brief section 2
    # and "Run 1 results".)
    comptime if UNIFORM:
        # C4. Batch b covers columns [b * SPAN, (b + 1) * SPAN); thread tid
        # reads b * SPAN + tid + u * 256 for u in 0..7. The block takes batch
        # b iff (b + 1) * SPAN <= length, a condition with no `tid` in it.
        # WHY NO THREAD'S VISITED SET CHANGES: (i) a batch the block takes
        # satisfies tid + 7 * 256 < SPAN <= length - b * SPAN for every
        # tid, so the 2026-09-09 per-thread condition took it too, with the
        # same eight columns in the same order; (ii) a batch the per-thread
        # form took and this form does not is the batch b = length // SPAN
        # for the tids with tid + 7 * 256 < length - b * SPAN, that is when
        # (length - 1792) mod 2048 is in 1..255; those tids now reach the
        # tail loop at col = b * SPAN + tid and visit b * SPAN + tid + u *
        # 256, u = 0..7, all < length, ascending, through the same
        # `_smallk_insert`, and stop there because the ninth column is
        # >= b * SPAN + 2048 > length; the per-thread form's tail for those
        # tids started at b * SPAN + 2048 + tid >= length and was empty. So
        # both forms apply the same inserts in the same order to every lane.
        # There are no extra iterations for lanes past the end: every lane
        # of a taken batch has all eight columns inside the row.
        var batch_base = 0
        var done = 0
        while batch_base + SMALLK_SCAN_SPAN <= length:
            var batch = SIMD[DType.float32, SMALLK_SCAN_UNROLL](0.0)
            comptime for u in range(SMALLK_SCAN_UNROLL):
                batch[u] = values.unsafe_load(base + batch_base + tid + u * SMALLK_BLOCK)
            comptime for u in range(SMALLK_SCAN_UNROLL):
                var pending = composite_key(
                    batch[u], UInt32(batch_base + tid + u * SMALLK_BLOCK), select_min
                )
                comptime if SABOTAGE and (not BOUND) and (not WARPBOUND) and u == 0:
                    # `uniform` arm reach: bit 0 of the index half of the
                    # first element of every batch is flipped, so one
                    # candidate column in eight carries its neighbor's
                    # index and the gathered value moves with it. Only
                    # this loop form carries it: a flip proves this loop.
                    pending = pending ^ UInt64(1)
                comptime if BOUND or WARPBOUND:
                    if pending < gate:
                        _smallk_insert[CAP](local_keys, threshold, pending, k)
                        gate = threshold if threshold < bound else bound
                else:
                    if pending < threshold:
                        _smallk_insert[CAP](local_keys, threshold, pending, k)
            batch_base += SMALLK_SCAN_SPAN
            done += 1
            comptime if WARPBOUND and SMALLK_SHUFFLE:
                if _smallk_warpbound_refresh_due(done, wb_fill):
                    # C2 REFRESH. `done` is block-uniform (C4), so every
                    # lane of every warp is here, and the butterfly is
                    # convergent. Every lane publishes its wb_depth-th
                    # smallest key (a real key: the lists are full), the
                    # warp folds them into the group bound, and the gate
                    # becomes min(threshold, bound). No barrier, no shared
                    # memory, `rounds` untouched, so the rank phase below
                    # starts from the same parity as the uniform arm.
                    #
                    # WHY THE OUTPUT BITS ARE UNCHANGED. At least k keys of
                    # the warp's union, hence of the block's, are <= bound
                    # (the argument above `_smallk_warp_group_bound`). That
                    # count never drops afterwards: a key leaves a list only
                    # when a smaller key from the same lane pushes it off
                    # the end, and the smaller key is <= bound too. A pending
                    # key p >= bound therefore has at least k union keys
                    # below it and cannot be among the row's k smallest
                    # (p == bound is impossible: p's column is in no list,
                    # and keys carry their column). Dropping it is the same
                    # act as the baseline's dropping of a key at or above
                    # the lane's own k-th. So the union still contains the
                    # true top-k after the scan, the rank phase pops the
                    # union's exact minima k times, the k-th and the ties
                    # are decided by the same UInt64 key compare, and the
                    # value is still read from the original tile cell. Warps
                    # hold different bounds; each is a bound on its own
                    # union, which is a subset of the block's, so the
                    # argument holds per warp.
                    var published = sentinel
                    comptime for slot in range(CAP):
                        if slot == wb_depth - 1:
                            published = local_keys[slot]
                    var found = _smallk_warp_group_bound[SMALLK_LANES](published, wb_group)
                    comptime if SABOTAGE:
                        # `warpbound` arm reach: bit 63 of the reduced bound
                        # is cleared. `twiddle_in` sets bit 31 of every
                        # non-negative float's bits, so every composite key
                        # with a non-negative distance has bit 63 set and
                        # the sabotaged bound sits below all of them: from
                        # the first refresh on, every lane rejects every
                        # non-negative-distance key, and the output is the
                        # top-k of the first wb_fill batches (4,096 columns
                        # at k in 9..16). That differs from the true top-k
                        # whenever one true neighbor lies beyond those
                        # columns: certain on the arms check (a planted +0.0
                        # at length / 2 and length - 1) and with probability
                        # 1 - (4096 / 65536)^k on a hashed row. Only the
                        # reduction's own result is sabotaged, and only when
                        # it is a real key: a refresh that never produced
                        # one cannot prove reach.
                        if found != sentinel:
                            found = found & UInt64(0x7FFFFFFFFFFFFFFF)
                    bound = found
                    gate = threshold if threshold < bound else bound
            comptime if BOUND:
                if _smallk_bound_refresh_due(done):
                    # C1 REFRESH. Every thread publishes its head
                    # (`local_keys[0]`, the sentinel if its list is empty);
                    # the block pops the k smallest of those 256 heads with
                    # the rank phase's own machinery (same butterfly or tree,
                    # same UInt64 compares, the popped lane's copy replaced
                    # by the sentinel, one barrier per round); the k-th pop
                    # is `bound`. `done` is block-uniform (C4), so every
                    # thread reaches every barrier.
                    #
                    # WHY THE OUTPUT BITS ARE UNCHANGED. The 256 heads are
                    # keys of the union of the lanes' lists, so `bound` is
                    # the k-th smallest of a SUBSET of the union and at
                    # least k keys of the union are <= bound at this
                    # moment. That count never drops: a key leaves a list
                    # only when a smaller key from the same lane pushes it
                    # off the end, and the smaller key is <= bound too. A
                    # pending key p >= bound therefore has at least k union
                    # keys below it and cannot be among the row's k smallest
                    # (p == bound is impossible: p's column is not in any
                    # list, and keys carry their column). Dropping it is the
                    # same act as the baseline's dropping of a key at or
                    # above the lane's own k-th. So the union still contains
                    # the true top-k after the scan, the rank phase pops the
                    # union's exact minima k times, the k-th and the ties
                    # are decided by the same UInt64 key compare, and the
                    # value is still read from the original tile cell.
                    var mine = local_keys[0]
                    var kth = sentinel
                    comptime if SMALLK_SHUFFLE:
                        var warp = tid // SMALLK_LANES
                        var lane = tid % SMALLK_LANES
                        for r in range(k):
                            var group_min = shuffle_min_u64[SMALLK_LANES](mine)
                            var page = (rounds & 1) * SMALLK_WARPS
                            if lane == 0:
                                heads[page + warp] = group_min
                            barrier()
                            var winner = heads[page]
                            comptime for w in range(1, SMALLK_WARPS):
                                var other = heads[page + w]
                                if other < winner:
                                    winner = other
                            if mine == winner:
                                mine = sentinel
                            comptime if SABOTAGE:
                                # `headbound` arm reach: the bound is the
                                # FIRST pop (the block minimum head), not
                                # the k-th, so from here on a lane admits
                                # only new record minima and the union
                                # loses true neighbors on nearly every row.
                                # Only the refresh carries it: a flip proves
                                # the bound path ran. The round count and
                                # barriers are unchanged.
                                if r == 0:
                                    kth = winner
                            else:
                                kth = winner
                            rounds += 1
                    else:
                        for r in range(k):
                            heads[tid] = mine
                            barrier()
                            var stride = SMALLK_BLOCK // 2
                            while stride > 0:
                                if tid < stride:
                                    var other = heads[tid + stride]
                                    if other < heads[tid]:
                                        heads[tid] = other
                                barrier()
                                stride //= 2
                            var winner = heads[0]
                            barrier()
                            if mine == winner:
                                mine = sentinel
                            comptime if SABOTAGE:
                                if r == 0:
                                    kth = winner
                            else:
                                kth = winner
                            rounds += 1
                    bound = kth
                    gate = threshold if threshold < bound else bound
        col = batch_base + tid
    else:
        # The 2026-09-09 form: the batch condition is per thread.
        while col + (SMALLK_SCAN_UNROLL - 1) * SMALLK_BLOCK < length:
            var batch = SIMD[DType.float32, SMALLK_SCAN_UNROLL](0.0)
            comptime for u in range(SMALLK_SCAN_UNROLL):
                batch[u] = values.unsafe_load(base + col + u * SMALLK_BLOCK)
            comptime for u in range(SMALLK_SCAN_UNROLL):
                var pending = composite_key(batch[u], UInt32(col + u * SMALLK_BLOCK), select_min)
                comptime if SABOTAGE and u == 0:
                    # `baseline` arm reach: same flip as the uniform arm's,
                    # carried by this loop form only.
                    pending = pending ^ UInt64(1)
                if pending < threshold:
                    _smallk_insert[CAP](local_keys, threshold, pending, k)
            col += SMALLK_SCAN_SPAN
    while col < length:
        var pending = composite_key(values.unsafe_load(base + col), UInt32(col), select_min)
        comptime if BOUND or WARPBOUND:
            if pending < gate:
                _smallk_insert[CAP](local_keys, threshold, pending, k)
                gate = threshold if threshold < bound else bound
        else:
            if pending < threshold:
                _smallk_insert[CAP](local_keys, threshold, pending, k)
        col += SMALLK_BLOCK
    comptime if SMALLK_SHUFFLE:
        var warp = tid // SMALLK_LANES
        var lane = tid % SMALLK_LANES
        for rank in range(k):
            var mine = local_keys[0]
            var group_min = shuffle_min_u64[SMALLK_LANES](mine)
            # `rounds` is 0 unless a C1 refresh ran; see its comment.
            var page = ((rank + rounds) & 1) * SMALLK_WARPS
            if lane == 0:
                heads[page + warp] = group_min
            barrier()
            var winner = heads[page]
            comptime for w in range(1, SMALLK_WARPS):
                var other = heads[page + w]
                if other < winner:
                    winner = other
            if tid == 0:
                var selected = UInt32(winner & UInt64(4294967295))
                out_indices.unsafe_store(row * k + rank, selected)
                out_values.unsafe_store(row * k + rank, values.unsafe_load(row * length + Int(selected)))
            if mine == winner:
                comptime for slot in range(CAP - 1):
                    local_keys[slot] = local_keys[slot + 1]
                local_keys[CAP - 1] = sentinel
    else:
        for rank in range(k):
            var mine = local_keys[0]
            heads[tid] = mine
            barrier()
            var stride = SMALLK_BLOCK // 2
            while stride > 0:
                if tid < stride:
                    var other = heads[tid + stride]
                    if other < heads[tid]:
                        heads[tid] = other
                barrier()
                stride //= 2
            var winner = heads[0]
            barrier()
            if tid == 0:
                var selected = UInt32(winner & UInt64(4294967295))
                out_indices.unsafe_store(row * k + rank, selected)
                out_values.unsafe_store(row * k + rank, values.unsafe_load(row * length + Int(selected)))
            if mine == winner:
                comptime for slot in range(CAP - 1):
                    local_keys[slot] = local_keys[slot + 1]
                local_keys[CAP - 1] = sentinel
            barrier()


@always_inline
def _smallk_enqueue[
    CAP: Int, K: Int, UNIFORM: Bool, BOUND: Bool, SABOTAGE: Bool, WARPBOUND: Bool = False
](
    ctx: DeviceContext,
    values: MutPointer[Float32, MutAnyOrigin],
    out_values: MutPointer[Float32, MutAnyOrigin],
    out_indices: MutPointer[UInt32, MutAnyOrigin],
    rows: Int, length: Int, k: Int, select_min: Bool,
) raises:
    ctx.enqueue_function[smallk_bucket_kernel[CAP, K, UNIFORM, BOUND, SABOTAGE, WARPBOUND]](
        values, out_values, out_indices,
        Int32(length), Int32(k), Int32(select_min),
        grid_dim=(rows, 1, 1), block_dim=(SMALLK_BLOCK, 1, 1),
    )


@always_inline
def _smallk_launch_bucket[CAP: Int, K: Int](
    ctx: DeviceContext,
    values: MutPointer[Float32, MutAnyOrigin],
    out_values: MutPointer[Float32, MutAnyOrigin],
    out_indices: MutPointer[UInt32, MutAnyOrigin],
    rows: Int, length: Int, k: Int, select_min: Bool, arm: Int,
) raises:
    """One capacity bucket: the arm's instantiation on a trial build, the
    build default otherwise (any other arm is refused there, so a check
    that asks for an arm cannot pass silently on a non-trial build)."""
    comptime assert not (SMALLK_HEAD_BOUND_DEFAULT and SMALLK_WARPBOUND_DEFAULT), "one bound arm at most"
    comptime if SMALLK_SELECT_TRIAL:
        var sabotage = (arm & SMALLK_ARM_SABOTAGE) != 0
        var which = arm & (SMALLK_ARM_SABOTAGE - 1)
        if which == SMALLK_ARM_BASELINE:
            if sabotage:
                _smallk_enqueue[CAP, K, False, False, True](ctx, values, out_values, out_indices, rows, length, k, select_min)
            else:
                _smallk_enqueue[CAP, K, False, False, False](ctx, values, out_values, out_indices, rows, length, k, select_min)
        elif which == SMALLK_ARM_UNIFORM:
            if sabotage:
                _smallk_enqueue[CAP, K, True, False, True](ctx, values, out_values, out_indices, rows, length, k, select_min)
            else:
                _smallk_enqueue[CAP, K, True, False, False](ctx, values, out_values, out_indices, rows, length, k, select_min)
        elif which == SMALLK_ARM_HEADBOUND:
            if sabotage:
                _smallk_enqueue[CAP, K, True, True, True](ctx, values, out_values, out_indices, rows, length, k, select_min)
            else:
                _smallk_enqueue[CAP, K, True, True, False](ctx, values, out_values, out_indices, rows, length, k, select_min)
        elif which == SMALLK_ARM_WARPBOUND:
            comptime if not SMALLK_SHUFFLE:
                # The refresh is shuffles; a column whose lane width the
                # vendor's compiler chooses per kernel has no convergent
                # warp to fold. Refuse rather than run the uniform arm under
                # this name.
                raise Error("small-k selector: the warpbound arm needs a fixed-lane-width column")
            if sabotage:
                _smallk_enqueue[CAP, K, True, False, True, True](ctx, values, out_values, out_indices, rows, length, k, select_min)
            else:
                _smallk_enqueue[CAP, K, True, False, False, True](ctx, values, out_values, out_indices, rows, length, k, select_min)
        else:
            raise Error("small-k selector: unknown arm " + String(arm))
    else:
        if arm != SMALLK_ARM_DEFAULT:
            raise Error(
                "small-k selector: arm " + String(arm)
                + " needs a build with -D MOJOLEARN_KNN_SELECT_TRIAL=1"
            )
        _smallk_enqueue[
            CAP, K, SMALLK_UNIFORM_TRIP_DEFAULT, SMALLK_HEAD_BOUND_DEFAULT, False, SMALLK_WARPBOUND_DEFAULT
        ](ctx, values, out_values, out_indices, rows, length, k, select_min)


def smallk_select_launch(
    ctx: DeviceContext,
    values: MutPointer[Float32, MutAnyOrigin],
    out_values: MutPointer[Float32, MutAnyOrigin],
    out_indices: MutPointer[UInt32, MutAnyOrigin],
    rows: Int, length: Int, k: Int, select_min: Bool = True,
    arm: Int = SMALLK_ARM_DEFAULT,
) raises:
    """One query tile's top-k for 1 <= k <= SMALLK_MAX_K, bucketed by capacity.

    Pointer form, so the caller can offset into the outer output buffer the
    way the radix launch does. `length >= k` is the caller's to guarantee:
    with fewer real keys than k the sentinel would win a rank. `arm` is
    `smallk_select_arm_from_env()`'s value (read once per request by the
    caller); it is meaningful on trial builds only.
    """
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("small-k selector requires IDENTICAL")
    if rows <= 0 or rows > 2147483647 or length <= 0 or length > 2147483647:
        raise Error("small-k selector requires positive Int32 dimensions")
    if k < 1 or k > SMALLK_MAX_K or k > length:
        raise Error("small-k selector supports only 1 <= k <= min(64, length)")
    comptime if knn_selector_specialize_common_for[TARGET_COLUMN, GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL]():
        if k == 10:
            _smallk_launch_bucket[16, 10](ctx, values, out_values, out_indices, rows, length, k, select_min, arm)
            return
        elif k == 15:
            _smallk_launch_bucket[16, 15](ctx, values, out_values, out_indices, rows, length, k, select_min, arm)
            return
    if k <= 16:
        _smallk_launch_bucket[16, 0](ctx, values, out_values, out_indices, rows, length, k, select_min, arm)
    elif k <= 32:
        _smallk_launch_bucket[32, 0](ctx, values, out_values, out_indices, rows, length, k, select_min, arm)
    else:
        _smallk_launch_bucket[64, 0](ctx, values, out_values, out_indices, rows, length, k, select_min, arm)


# ---------------------------------------------------------------------------
# The partial top-k merge, 2026-09-09: index-axis tiling's second half.
#
# `running` holds a row's k best (distance, index) pairs so far, ascending by
# composite key with GLOBAL indices; `partial` holds the k best of the next
# column tile, ascending, with indices LOCAL to that tile (add `base`). Keys
# are unique, so the rank of an element in the union is its own position
# plus the number of elements of the other list below it, and the k smallest
# ranks are written back in place. No arithmetic on the distances, no
# arrival order anywhere: the result is a pure function of the two lists.
# ---------------------------------------------------------------------------

comptime MERGE_BLOCK = 256
comptime MERGE_MAX_K = 1024


def partial_topk_merge_kernel(
    running_values: MutPointer[Float32, MutAnyOrigin],
    running_indices: MutPointer[UInt32, MutAnyOrigin],
    partial_values: MutPointer[Float32, MutAnyOrigin],
    partial_indices: MutPointer[UInt32, MutAnyOrigin],
    k_in: Int32, base_in: Int32, select_min_in: Int32,
):
    var k = Int(k_in)
    var base = UInt32(Int(base_in))
    var select_min = select_min_in != 0
    var tid = Int(thread_idx.x)
    var row = Int(block_idx.x)
    var keys = stack_allocation[
        2 * MERGE_MAX_K, Scalar[DType.uint64], address_space=AddressSpace.SHARED,
    ]()
    var vals = stack_allocation[
        2 * MERGE_MAX_K, Scalar[DType.float32], address_space=AddressSpace.SHARED,
    ]()
    var slot = tid
    while slot < 2 * k:
        if slot < k:
            var v = running_values.unsafe_load(row * k + slot)
            keys[slot] = composite_key(v, running_indices.unsafe_load(row * k + slot), select_min)
            vals[slot] = v
        else:
            var v = partial_values.unsafe_load(row * k + slot - k)
            keys[slot] = composite_key(v, partial_indices.unsafe_load(row * k + slot - k) + base, select_min)
            vals[slot] = v
        slot += MERGE_BLOCK
    barrier()
    slot = tid
    while slot < 2 * k:
        var key = keys[slot]
        var rank: Int
        var other_start: Int
        if slot < k:
            rank = slot
            other_start = k
        else:
            rank = slot - k
            other_start = 0
        for j in range(other_start, other_start + k):
            if keys[j] < key:
                rank += 1
        if rank < k:
            running_values.unsafe_store(row * k + rank, vals[slot])
            running_indices.unsafe_store(row * k + rank, UInt32(key & UInt64(4294967295)))
        slot += MERGE_BLOCK
    barrier()


def partial_topk_merge_launch(
    ctx: DeviceContext,
    running_values: MutPointer[Float32, MutAnyOrigin],
    running_indices: MutPointer[UInt32, MutAnyOrigin],
    partial_values: MutPointer[Float32, MutAnyOrigin],
    partial_indices: MutPointer[UInt32, MutAnyOrigin],
    rows: Int, k: Int, base: Int, select_min: Bool = True,
) raises:
    """Merge one column tile's partial top-k into the running top-k, per row."""
    if rows <= 0 or rows > 2147483647 or base < 0 or base > 2147483647:
        raise Error("partial top-k merge requires positive Int32 dimensions")
    if k < 1 or k > MERGE_MAX_K:
        raise Error("partial top-k merge supports only 1 <= k <= 1024")
    ctx.enqueue_function[partial_topk_merge_kernel](
        running_values, running_indices, partial_values, partial_indices,
        Int32(k), Int32(base), Int32(select_min),
        grid_dim=(rows, 1, 1), block_dim=(MERGE_BLOCK, 1, 1),
    )
