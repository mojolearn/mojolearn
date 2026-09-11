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

`smallk_bucket_kernel` carries the gated candidates behind the selection
trial hook (`-D MOJOLEARN_KNN_SELECT_TRIAL=1`, arms chosen per request from
MOJOLEARN_KNN_SELECT): the block-uniform trip count (DEVIATION 2497, C4,
the shipped default since 2026-09-11), the block head-bound rejection
(DEVIATION 2498, C1) and the warp-scope group bound (DEVIATION 2515, C2),
both measured NEGATIVE on the H100 2026-09-11 and kept as that record, and
deferred insertion (DEVIATION 2517). See the hook comment above the kernel; without
the define the shipped kernel is the 2026-09-09 one plus C4. The same hook
carries three TIMING-ONLY arms (DEVIATION 2516) whose output is invalid by
construction: `skiprank`, `skipscan` and `scanonly1` measure the scan phase
and the rank phase of the shipped kernel separately; the gate runs them in
its timing block alone, never in a correctness or reach section. Their
verdict (the per-lane K-deep list is the k-proportional cost, and it is paid
per element step whether or not a lane inserts) is what the fourth gated
candidate answers: `deferred` (DEVIATION 2517) makes the per-element work
K-independent by queueing admitted keys and running the K-chain only at
warp-uniform drains. Deferred measured NEGATIVE too, and the kernel stats
leg (DEVIATION 2519) found the list register-resident at 54 to 56 registers
and four blocks per SM; the fifth and sixth candidates act on that:
`capk` (DEVIATION 2521) instantiates the K-specialized kernels with CAP = K
(the list holds exactly the k keys the block's top-k can draw on) and
`capk_selp` adds a branch-free min/max carry chain. See the comment above
`_smallk_insert`. Both measured NEUTRAL at 75 percent occupancy, so the
seventh pass (DEVIATION 2522) measures the chain itself: `noshift`
(timing-only: the admission compare and the list without the K-chain),
`voteguard` (a real warp-uniform branch around the chain, output valid) and
`votecount` (the admission rate per warp-step, read back through a device
counter). See the comment above `_smallk_overwrite_last`. That measurement
priced the chain at 6.7 ms (k10) and 10.8 ms (k15) of the launch, issued on
90 to 96 percent of warp-steps; the eighth pass (DEVIATION 2523) composes
the two levers, C2's warp bound (fewer admitting lanes per step) inside
`voteguard`'s ballot branch (a step with no admitting lane skips the
chain): `warpbound_guard`, `warpbound_guard1` (refresh every batch) and
`warpbound_count` (the admit rate under the bound). See the comment above
`_smallk_warpbound_refresh_due`.
"""
from std.atomic import Atomic
from std.gpu import block_idx, thread_idx
from std.os import getenv
from std.sys.compile import is_defined
from neighbors.checks.lane_minimum import shuffle_min_u64
from std.gpu.primitives.warp import shuffle_xor, vote
from std.memory import bitcast, stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from checks.kernel_matrix import (
    TARGET_COLUMN,
    knn_selector_shuffle_for,
    knn_selector_specialize_common_for,
    knn_selector_warpbound_guard_for,
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
#                                    (DEVIATION 2515; NEGATIVE on the H100
#                                    2026-09-11, kept as the record)
#   MOJOLEARN_KNN_SELECT=deferred    C4 + deferred insertion (DEVIATION
#                                    2517): per element only key, compare
#                                    and a SMALLK_DEFER_Q-slot register
#                                    queue; the K-chain runs at warp-uniform
#                                    drains (every batch, and as soon as a
#                                    `vote` says any lane's queue is full);
#                                    NEGATIVE on the H100 2026-09-11
#   MOJOLEARN_KNN_SELECT=capk        C4 with CAP = K (DEVIATION 2521): the
#                                    K-specialized buckets run
#                                    `smallk_bucket_kernel[K, K, ...]`, a
#                                    list of exactly k slots instead of 16;
#                                    the generic bucket has no such
#                                    instantiation and RAISES under this
#                                    name (see `_smallk_launch_bucket`)
#   MOJOLEARN_KNN_SELECT=capk_selp   `capk` plus the branch-free carry
#                                    chain (SELP: every step of the insert
#                                    is one unsigned min and one unsigned
#                                    max, no per-step branch); same
#                                    restriction to the K-specialized
#                                    buckets
#   MOJOLEARN_KNN_SELECT=voteguard   C4 with the per-element admission
#                                    wrapped in a warp-uniform `vote`
#                                    guard (DEVIATION 2522): the K-chain
#                                    is entered only on element steps
#                                    where some lane of the warp admits;
#                                    bit-identical (the same lanes insert
#                                    the same keys); output VALID, a
#                                    normal trial arm
#   MOJOLEARN_KNN_SELECT=votecount   `voteguard` plus a per-block count
#                                    of warp-steps and of warp-steps with
#                                    any admission, accumulated into a
#                                    device counter the launcher reads
#                                    back and prints as `KNN_ADMIT_RATE
#                                    warp_steps N any_admit M` under the
#                                    phase-timer build; output VALID but
#                                    its time is not a price (the launcher
#                                    synchronizes per launch); the gate
#                                    lists it as timing-only
#   MOJOLEARN_KNN_SELECT=warpbound_guard
#                                    C4 + C2 + the vote guard (DEVIATION
#                                    2523): the warp bound refresh exactly
#                                    as `warpbound` does it (every
#                                    SMALLK_WARPBOUND_EVERY batches once
#                                    the lists are full), and the
#                                    admission test `pending < min(
#                                    threshold, bound)` inside the ballot
#                                    branch, so a warp-step where no lane
#                                    admits skips the chain; bit-identical
#                                    (C2's union argument plus voteguard's
#                                    same-lanes-same-keys); output VALID,
#                                    a normal trial arm
#   MOJOLEARN_KNN_SELECT=warpbound_guard1
#                                    `warpbound_guard` with the bound
#                                    refreshed EVERY batch (WB_EVERY = 1)
#   MOJOLEARN_KNN_SELECT=warpbound_count
#                                    `warpbound_guard` plus `votecount`'s
#                                    admission counter: the admit rate
#                                    UNDER THE BOUND, against votecount's
#                                    0.90 / 0.96 (C2's event model says
#                                    0.45 / 0.54); output VALID, time not
#                                    a price; the gate lists it as
#                                    timing-only
#   unset or empty                   the build default, SMALLK_ARM_DEFAULT
#   anything else                    RAISES; the gate harness relies on it
#   MOJOLEARN_KNN_SELECT_SABOTAGE=1  the chosen arm's SABOTAGE instantiation
#                                    (reach proof; see the kernel)
#
# TIMING-ONLY ARMS (DEVIATION 2516). OUTPUT INVALID BY CONSTRUCTION; the
# gate runs them in its timing block only, never in a correctness, oracle or
# reach section, and reports them under a separate table that says so. They
# measure the two phases of the shipped kernel (the uniform default: C4 trip
# count, no bound) one at a time, on the H100, instead of modeling the
# split from the k-slope:
#
#   MOJOLEARN_KNN_SELECT=skiprank    the scan exactly as the default runs
#                                    it, then NO rank phase: every lane's
#                                    list is folded into a block digest
#                                    that thread 0 uses as a gather address
#                                    and writes (so no scan work is dead),
#                                    the other k - 1 slots get the sentinel
#   MOJOLEARN_KNN_SELECT=skipscan    NO scan: every lane's k slots are
#                                    filled with a synthetic ascending
#                                    pattern of block-distinct keys whose
#                                    index halves are in range, then the
#                                    rank phase exactly as the default runs
#                                    it (k rounds, one winner per round,
#                                    the winner's value gathered from the
#                                    tile)
#   MOJOLEARN_KNN_SELECT=scanonly1   `skiprank` with the register list one
#                                    key deep (CAP = 1, K = 1): the scan's
#                                    cost with no list to maintain beyond a
#                                    running minimum
#   MOJOLEARN_KNN_SELECT=noshift     (DEVIATION 2522) the uniform scan
#                                    with the K-deep list and the
#                                    admission compare, but an admitted
#                                    key OVERWRITES slot K - 1 (the
#                                    threshold slot) instead of running
#                                    the carry chain; the threshold is
#                                    refreshed from that slot and the rank
#                                    phase runs on whatever the list
#                                    holds. select_ms(uniform) minus
#                                    select_ms(noshift) is the chain's
#                                    own cost.
#
# A timing-only arm refuses the sabotage bit (there is no reach to prove
# on an arm whose output is wrong by design). They exist on trial builds
# only, like every other arm.
#
# WITHOUT THE DEFINE NONE OF THIS EXISTS: `smallk_select_arm_from_env`
# returns SMALLK_ARM_DEFAULT without touching the environment, the launch
# refuses any other arm, and the only instantiations in the binary are
# `smallk_bucket_kernel[CAP, K, SMALLK_UNIFORM_TRIP_DEFAULT,
# SMALLK_HEAD_BOUND_DEFAULT, False, SMALLK_WARPBOUND_DEFAULT or
# SMALLK_WARPBOUND_GUARD_DEFAULT, SMALLK_PHASE_FULL, SMALLK_DEFERRED_DEFAULT,
# SMALLK_SELP_DEFAULT, SMALLK_CHAIN_VOTEGUARD if
# SMALLK_WARPBOUND_GUARD_DEFAULT else SMALLK_CHAIN_INSERT]` with
# CAP the bucket capacity (16 / 32 / 64) unless SMALLK_CAPK_DEFAULT is on,
# in which case the K-specialized buckets take CAP = K. With the two bound
# defaults, the deferred default, the capk default, the selp default and
# the warpbound-guard default False (the state until a gate passes) that is
# the [CAP, K] kernel of 2026-09-09 under C4's trip count: the `comptime if`
# arms below fold away and the non-trial code path is the one that shipped.
#
# THE DEFAULTS. Seven comptime switches here rather than kernel-matrix rows,
# because this lane may not edit `checks/kernel_matrix.mojo`; the flip that
# promotes an arm moves them into a SCHEDULING row
# (`knn_selector_head_bound_for[column, identical]`, brief section 4) in
# the same session as the measured win. Order of flips: UNIFORM first
# (gated alone, arms baseline,uniform, equality on every fixture including
# `divergent_tail`; DONE 2026-09-11), then ONE candidate arm (arms
# uniform,<arm>, equality plus the request-level price). Every candidate
# arm requires UNIFORM; HEAD_BOUND, WARPBOUND and DEFERRED exclude each
# other; CAPK and SELP (DEVIATION 2521) are the uniform arm's scan with a
# shorter list and exclude the three of them; SELP requires CAPK;
# WARPBOUND_GUARD (DEVIATION 2523) is WARPBOUND composed with the VOTEGUARD
# chain form and excludes every other candidate default.
# ---------------------------------------------------------------------------
comptime SMALLK_SELECT_TRIAL = is_defined["MOJOLEARN_KNN_SELECT_TRIAL"]()
comptime SMALLK_UNIFORM_TRIP_DEFAULT = True  # DEVIATION 2497: flipped 2026-09-11 on the H100 and M4 gates
comptime SMALLK_HEAD_BOUND_DEFAULT = False  # DEVIATION 2498: NEGATIVE on the H100 2026-09-11, stays off
comptime SMALLK_WARPBOUND_DEFAULT = False  # DEVIATION 2515: NEGATIVE on the H100 2026-09-11, stays off
comptime SMALLK_DEFERRED_DEFAULT = False  # DEVIATION 2517: NEGATIVE on the H100 2026-09-11, stays off
comptime SMALLK_CAPK_DEFAULT = False  # DEVIATION 2521: NEUTRAL on the H100 2026-09-11, stays off
comptime SMALLK_SELP_DEFAULT = False  # DEVIATION 2521: NEUTRAL on the H100 2026-09-11; requires SMALLK_CAPK_DEFAULT
comptime SMALLK_WARPBOUND_GUARD_DEFAULT = knn_selector_warpbound_guard_for[TARGET_COLUMN, GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL]() and SMALLK_UNIFORM_TRIP_DEFAULT  # DEVIATION 2523: FLIPPED 2026-09-11 on the H100 promotion run (all eight cells, bit-equal); NVIDIA row in checks/kernel_matrix.mojo

comptime SMALLK_ARM_BASELINE = 0
comptime SMALLK_ARM_UNIFORM = 1
comptime SMALLK_ARM_HEADBOUND = 2
comptime SMALLK_ARM_WARPBOUND = 3
# Timing-only arms (DEVIATION 2516); see the hook comment. Never a default.
comptime SMALLK_ARM_SKIPRANK = 4
comptime SMALLK_ARM_SKIPSCAN = 5
comptime SMALLK_ARM_SCANONLY1 = 6
# Deferred insertion (DEVIATION 2517); see the comment above `_smallk_drain`.
comptime SMALLK_ARM_DEFERRED = 7
# CAP = K and the branch-free chain (DEVIATION 2521); see `_smallk_insert`.
comptime SMALLK_ARM_CAPK = 8
comptime SMALLK_ARM_CAPK_SELP = 9
# The chain measurement (DEVIATION 2522); see `_smallk_overwrite_last`.
# `noshift` is timing-only (output invalid); `voteguard` is a normal arm
# (output valid); `votecount` is `voteguard` plus the admission counter
# (output valid, time not a price). Never a default.
comptime SMALLK_ARM_NOSHIFT = 10
comptime SMALLK_ARM_VOTEGUARD = 11
comptime SMALLK_ARM_VOTECOUNT = 12
# The warp bound composed with the vote guard (DEVIATION 2523); see the
# comment above `_smallk_warpbound_refresh_due`. `warpbound_guard` and
# `warpbound_guard1` are normal arms (output valid); `warpbound_count` is
# the counter form (output valid, time not a price). Never a default until
# SMALLK_WARPBOUND_GUARD_DEFAULT flips.
comptime SMALLK_ARM_WARPBOUND_GUARD = 13
comptime SMALLK_ARM_WARPBOUND_COUNT = 14
comptime SMALLK_ARM_WARPBOUND_GUARD1 = 15
# OR'd into the arm value; the launch strips it. The arm space below it is
# now FULL (0 .. 15): the next arm moves this bit to 32 and the mask in
# `_smallk_launch_bucket` with it.
comptime SMALLK_ARM_SABOTAGE = 16

# The kernel's PHASE parameter: which phases of `smallk_bucket_kernel` run.
# FULL is every shipped instantiation; the other two exist on trial builds
# only and produce invalid output on purpose.
comptime SMALLK_PHASE_FULL = 0
comptime SMALLK_PHASE_SKIPRANK = 1
comptime SMALLK_PHASE_SKIPSCAN = 2
# The kernel's CHAIN parameter (DEVIATION 2522): what an admitted key does
# to the list in the uniform scan form. INSERT is every shipped
# instantiation (the carry chain of `_smallk_insert`); the other three
# exist on trial builds only.
comptime SMALLK_CHAIN_INSERT = 0
comptime SMALLK_CHAIN_NOSHIFT = 1
comptime SMALLK_CHAIN_VOTEGUARD = 2
comptime SMALLK_CHAIN_VOTECOUNT = 3
# The profile phase's build define (`-D MOJOLEARN_KNN_PHASE_TIMERS=1`, read
# by `knn_brute_force.mojo` as KNN_PHASE_TIMERS): the `votecount` launcher
# prints its counter line under it only, so the line lands in the same
# fd 1 capture the gate harness parses the phase line from.
comptime SMALLK_PHASE_TIMERS = is_defined["MOJOLEARN_KNN_PHASE_TIMERS"]()
comptime SMALLK_ARM_DEFAULT = SMALLK_ARM_HEADBOUND if SMALLK_HEAD_BOUND_DEFAULT else (
    SMALLK_ARM_WARPBOUND if SMALLK_WARPBOUND_DEFAULT else (
        SMALLK_ARM_WARPBOUND_GUARD if SMALLK_WARPBOUND_GUARD_DEFAULT else (
            SMALLK_ARM_DEFERRED if SMALLK_DEFERRED_DEFAULT else (
                SMALLK_ARM_CAPK_SELP if (SMALLK_CAPK_DEFAULT and SMALLK_SELP_DEFAULT) else (
                    SMALLK_ARM_CAPK if SMALLK_CAPK_DEFAULT else (
                        SMALLK_ARM_UNIFORM if SMALLK_UNIFORM_TRIP_DEFAULT else SMALLK_ARM_BASELINE
                    )
                )
            )
        )
    )
)


def smallk_select_arm_from_env() raises -> Int:
    """The selector arm for THIS request, read once on the host.

    Trial builds read `MOJOLEARN_KNN_SELECT` (baseline / uniform / headbound /
    warpbound / deferred / capk / capk_selp / voteguard / votecount /
    warpbound_guard / warpbound_guard1 / warpbound_count, the timing-only
    skiprank / skipscan / scanonly1 / noshift, unset = the build default,
    anything else raises) and
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
    elif name == "skiprank":
        arm = SMALLK_ARM_SKIPRANK
    elif name == "skipscan":
        arm = SMALLK_ARM_SKIPSCAN
    elif name == "scanonly1":
        arm = SMALLK_ARM_SCANONLY1
    elif name == "deferred":
        arm = SMALLK_ARM_DEFERRED
    elif name == "capk":
        arm = SMALLK_ARM_CAPK
    elif name == "capk_selp":
        arm = SMALLK_ARM_CAPK_SELP
    elif name == "noshift":
        arm = SMALLK_ARM_NOSHIFT
    elif name == "voteguard":
        arm = SMALLK_ARM_VOTEGUARD
    elif name == "votecount":
        arm = SMALLK_ARM_VOTECOUNT
    elif name == "warpbound_guard":
        arm = SMALLK_ARM_WARPBOUND_GUARD
    elif name == "warpbound_guard1":
        arm = SMALLK_ARM_WARPBOUND_GUARD1
    elif name == "warpbound_count":
        arm = SMALLK_ARM_WARPBOUND_COUNT
    else:
        raise Error(
            "MOJOLEARN_KNN_SELECT='" + name
            + "' is not a selector arm (baseline, uniform, headbound, warpbound,"
            + " deferred, capk, capk_selp, voteguard, votecount, warpbound_guard,"
            + " warpbound_guard1, warpbound_count, the timing-only"
            + " skiprank, skipscan, scanonly1, noshift, or unset)"
        )
    if String(getenv("MOJOLEARN_KNN_SELECT_SABOTAGE")) == "1":
        arm = arm | SMALLK_ARM_SABOTAGE
    return arm


# ---------------------------------------------------------------------------
# CAP = K (DEVIATION 2521, arm `capk`) and the branch-free chain (`capk_selp`).
#
# WHAT THE STATS LEG SAID (brief, "Step 6 result"). The CAP = 16 list is
# register-resident with no spills: 54 registers at K = 10 and 56 at K = 15,
# four 256-thread blocks per SM (50 percent occupancy), against 31 registers
# and eight blocks for the CAP = 1 control, on a scan whose 3.2 ms floor is
# tile reads. So the K cost is instruction count plus occupancy, and the
# sixteen-slot list carries dead weight at K = 10: slots 10 .. 15 are never
# written by the insert (its `slot < k` guard folds them out) but the rank
# phase's shift READS them (`local_keys[slot + 1]` up to slot 15), so they
# stay live across the whole scan as twelve registers of the sentinel.
#
# THE MECHANISM. The K-specialized buckets are instantiated with CAP = K:
# `smallk_bucket_kernel[10, 10, ...]` and `[15, 15, ...]`. Every touch of
# the list is bounded by CAP (the insert chain, the threshold pick, the
# skipscan fill, the digest, the rank phase's shift and its sentinel write),
# so a CAP = K instantiation reads and writes exactly K slots. Mojo's SIMD
# width must be a power of two (the repo pads every such queue, for example
# `warp_topk.mojo`'s TQP), so the STORAGE is `SIMD[DType.uint64, STORE]`
# with STORE the next power of two at or above CAP (16 for 10 and 15) and
# lanes CAP .. STORE - 1 are never read or written after the sentinel fill:
# they have no use in the compiled kernel and are dead (no phi, no copy, no
# register), which is what CAP = K means at the register level. For every
# existing instantiation STORE == CAP and the storage is what it was.
# The helpers below take the storage width as an inferred parameter (`W`)
# and the list depth as CAP, so their bodies are unchanged for W == CAP.
#
# WHY THE OUTPUT BITS ARE UNCHANGED. A lane's list holds its CAP smallest
# keys of everything it scanned. Only a lane's k smallest can ever be in
# the block's top-k: any other key of that lane has k same-lane keys below
# it. With CAP = K the list holds exactly those k, so the union of the 256
# lists still contains the row's true top-k after the scan; the rank phase
# is textually the same loop (k exact UInt64 minima of the union, ties by
# the index half of the same composite key, the winner's value gathered
# from the same tile cell), and its shift moves slots 1 .. K - 1 down and
# writes the sentinel at K - 1, which is the content slots 0 .. K - 1 had
# under CAP = 16 after the same shift (slots K .. 15 were the sentinel).
# The admission test is the same `pending < threshold` with the same
# threshold (the lane's k-th smallest), so every lane admits the same keys
# in the same order. Nothing depends on a spare slot: the insert's carry
# runs off the end of a full list the same way at K as at 16.
#
# SELP (`capk_selp`, lever 2 of the stats leg). The carry chain's per-step
# `if pending < local_keys[slot]: swap` is replaced by one unsigned minimum
# into the slot and one unsigned maximum into the carry: the smaller of
# (carry, slot) stays, the larger moves on, every step, no branch. That is
# the same sorted list whichever way it is computed: at each step the
# multiset {slot, carry} is preserved and the slot takes its minimum, so
# after CAP steps the slots hold the CAP smallest of the old list plus the
# new key, ascending, and the carry holds what fell off; with unique keys
# the compare never ties on real keys, and a tie of two sentinels yields
# the sentinel either way. SELP requires CAP == K (no `slot < k` guard in
# the chain) and is the uniform scan form only; the admission branch
# `pending < threshold` around the chain is kept, so a warp with no
# admitting lane still skips it exactly as the uniform arm does.
#
# SABOTAGE (reach): both arms are the uniform scan form, so they carry the
# uniform arm's flip (bit 0 of the index half on `u == 0` of every batch,
# inside the uniform loop). A flip proves the arm's own launch branch and
# its loop ran; the instantiation it ran is distinguished from the CAP = 16
# uniform kernel by the stats leg's register count and by the gate's timing,
# not by reach, since a depth-specific perturbation (dropping slot K - 1)
# flips only when a row's whole top-k sits in one lane and is not reliable.
# ---------------------------------------------------------------------------
@always_inline
def _smallk_insert[W: Int, //, CAP: Int, SELP: Bool = False](
    mut local_keys: SIMD[DType.uint64, W], mut threshold: UInt64,
    pending_in: UInt64, k: Int,
):
    """Carry-insert one key below `threshold` into the ascending local list
    (slots `0 .. k-1` of CAP, stored in the first CAP lanes of a W-wide
    register vector, W >= CAP) and refresh `threshold = local_keys[k - 1]`.
    Every list index is a comptime constant, so the list stays in
    registers. SELP (DEVIATION 2521) is the branch-free min/max chain; it
    requires CAP == k, which the kernel asserts at compile time."""
    comptime assert W >= CAP, "the list's storage must hold its depth"
    var pending = pending_in
    comptime if SELP:
        comptime for slot in range(CAP):
            var current = local_keys[slot]
            local_keys[slot] = min(pending, current)
            pending = max(pending, current)
    else:
        comptime for slot in range(CAP):
            if slot < k:
                if pending < local_keys[slot]:
                    var previous = local_keys[slot]
                    local_keys[slot] = pending
                    pending = previous
    comptime for slot in range(CAP):
        if slot == k - 1:
            threshold = local_keys[slot]


# ---------------------------------------------------------------------------
# THE CHAIN MEASUREMENT (DEVIATION 2522): `noshift`, `voteguard`, `votecount`.
#
# WHAT THE SEVEN PASSES BEFORE IT SAID (brief, steps 2 to 7). Every arm that
# changed how OFTEN the K-chain runs (headbound, warpbound: half the
# insertion events; deferred: the chain once per several elements) or how
# many REGISTERS it takes (capk: 54 to 40 registers, 4 to 6 blocks per SM)
# measured neutral against the 0.80 ms per unit of k. That is consistent
# with one reading only: the chain's instructions are issued on every
# element step for every lane whether or not the lane admits, because the
# compiler if-converts the admission branch `if pending < threshold` and
# the whole predicated chain executes each step. Two measurements decide
# it; this block is those two.
#
# `noshift` (TIMING ONLY, OUTPUT INVALID): the uniform scan form with the
# same K-deep register list, the same per-element key and admission
# compare, but an admitted key overwrites slot K - 1 (the threshold slot)
# in place of the carry chain, and the threshold is refreshed from that
# slot (`_smallk_overwrite_last`). The list is still CAP registers, the
# compare still runs every element, the rank phase still consumes the list
# (slot K - 1 flows down the pops into `mine`, the winner and the output
# index, so the write cannot be dropped; the threshold feeds the compare
# that feeds the write, so the compare cannot be dropped either). What is
# gone is exactly the chain's instructions. select_ms(uniform) minus
# select_ms(noshift) at the same k is therefore the chain's own cost;
# if it is the 0.80 ms per k slope, the chain is the whole K cost.
# Two things about it are NOT the uniform arm's and are known: (1) its
# threshold is a running minimum (each admitted key becomes the
# threshold), so it admits LESS often than the uniform arm; since the
# arm has no chain, its own time is admission-independent up to the
# predicated 8-byte write, and the difference is read against the uniform
# arm's real admissions. (2) Its rank phase: slots 0 .. K - 2 hold the
# sentinel, so the first K - 1 ranks are won by the sentinel on every lane
# and every lane pops (the shipped kernel pops one lane per rank); that is
# K - 1 rounds of a CAP - 1 register shift on 8 warps instead of 1, about
# a microsecond per launch, and the gathered index is masked into range
# (`% length`, under `comptime if` on this chain form only) so thread 0's
# gather of a sentinel index cannot fault.
#
# `voteguard` (OUTPUT VALID): the uniform arm with the admission wrapped in
# a warp-uniform guard: `if vote.any(admit): if admit: chain`. Every lane
# of the warp takes the vote (the uniform batch loop's trip count is
# block-uniform, C4, so the ballot is convergent; the tail loop's count is
# per lane, so the tail keeps the plain form and holds at most eight
# elements per lane). Bit-identical by construction: a lane inserts the
# same key at the same step in both forms (admit is unchanged; a lane
# that does not admit does nothing in both), and the list's content is a
# function of the inserted keys and their order. The measurement is
# whether a branch on a ballot, which the compiler cannot if-convert as it
# can a per-lane predicate around a short region, changes the chain's
# cost. If it does not, the reason is the admission rate per warp-step,
# which `votecount` reads: a per-lane `warp_steps` and `admit_steps`
# (identical across a warp's lanes, so lane 0's is the warp's) folded per
# block through shared memory into ONE pair of atomics on a two-slot
# device counter the launcher zeroes before and reads back after every
# launch, printed as `KNN_ADMIT_RATE warp_steps N any_admit M` under the
# phase-timer build. M / N is the fraction of warp-steps on which the
# guarded chain runs; 1 - M / N is the most `voteguard` could save of the
# chain's cost.
#
# SABOTAGE (reach): `voteguard` carries the uniform flip (bit 0 of the
# index half of the first element of every batch) INSIDE the guarded and
# admitted path, after the compare, so a flip proves the vote guard's body
# ran on an admitted key; the plain uniform flip before the compare is
# compiled out on this chain form. `noshift` and `votecount` refuse the
# sabotage bit like every timing-only arm.
# ---------------------------------------------------------------------------
@always_inline
def _smallk_overwrite_last[W: Int, //, CAP: Int](
    mut local_keys: SIMD[DType.uint64, W], mut threshold: UInt64,
    pending: UInt64, k: Int,
):
    """`noshift` (DEVIATION 2522, timing-only): the admitted key overwrites
    slot k - 1 of CAP and becomes the threshold; no carry chain. Every list
    index is a comptime constant, so the list stays in registers."""
    comptime assert W >= CAP, "the list's storage must hold its depth"
    comptime for slot in range(CAP):
        if slot == k - 1:
            local_keys[slot] = pending
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


# ---------------------------------------------------------------------------
# THE WARP BOUND INSIDE THE VOTE GUARD (DEVIATION 2523): `warpbound_guard`,
# `warpbound_guard1`, `warpbound_count`.
#
# WHAT STEP 8 SAID (brief, "Step 8 result"). The chain is 6.7 ms of the
# k10 launch and 10.8 ms of the k15 launch, the whole K cost; it issues on
# every warp-step where any of the 32 lanes admits; and under the shipped
# per-lane threshold that is 90 to 96 percent of warp-steps (votecount), so
# the ballot branch alone (`voteguard`) could skip only 4 to 10 percent of
# chains and saved 3 percent. C2's warp bound halves the admitting steps
# (its event model: 231 to 116 per 256 at k10, 246 to 138 at k15) but was
# measured with the chain if-converted, so the skipped admissions saved no
# chain and the arm paid its refreshes for nothing. The two levers compose:
#
#   the bound lowers the admission predicate  (pending < min(threshold, bound))
#   the ballot turns a step with no admitting lane into a skipped chain
#
# THE MECHANISM, and why the two features do not interfere. The refresh is
# C2's, textually: after `done += 1` at the bottom of a batch, when
# `_smallk_warpbound_refresh_due[WB_EVERY](done, wb_fill)`, every lane
# publishes its head, the warp folds the group bound with five xor shuffles
# and sets `gate = min(threshold, bound)`. That point is OUTSIDE the ballot
# branch (the unrolled `u` loop is closed) and block-uniform under C4, so
# every lane of every warp reaches the shuffles whatever the ballots inside
# the batch did. Inside the batch, the per-element step is `voteguard`'s
# with one change: `admit = pending < gate` instead of `pending < threshold`;
# then `any = ballot(admit) != 0; if any: if admit: _smallk_insert(...);
# gate = min(threshold, bound)`. The ballot is convergent for the same reason
# as in `voteguard` (nothing above it diverges; the trip count has no `tid`
# in it). The bound only LOWERS the predicate; it never adds an insertion
# path, and the chain inside the branch is the unchanged `_smallk_insert`.
# The tail loop (per-lane trip count, no ballot) keeps C2's plain
# `pending < gate` form on at most eight elements per lane. WB_EVERY is the
# refresh cadence: SMALLK_WARPBOUND_EVERY (2, C2's) for `warpbound_guard`,
# 1 for `warpbound_guard1`, since with the branch the bound's benefit is
# now realized per skipped step and a tighter bound may pay.
#
# WHY THE OUTPUT BITS ARE UNCHANGED. C2's argument (above
# `_smallk_warp_group_bound`): at least k keys of the warp's union, hence
# of the block's, are at or below the bound, that count never drops, so a
# pending key at or above min(threshold, bound) is not among the row's k
# smallest and dropping it is the baseline's own act; the union still holds
# the true top-k after the scan. Plus `voteguard`'s (above
# `_smallk_overwrite_last`): `admit` is the same test on the same gate as
# C2's, a lane that admits inserts the same key at the same step through
# the same `_smallk_insert`, a lane that does not admit does nothing in
# both forms, and the list is a function of the inserted keys and their
# order. The rank phase is untouched (no shared memory, no barrier,
# `rounds` stays 0). Neither argument depends on the other: the bound
# decides WHAT is admitted, the ballot decides WHEN the warp executes the
# chain, and the composition is both statements at once.
#
# SABOTAGE (reach): C2's, unchanged: bit 63 of the reduced bound is cleared
# inside the refresh, so from the first refresh on the gate sits below
# every non-negative-distance key and the output is the top-k of the first
# wb_fill batches (4,096 columns at k in 9..16), which flips every row with
# a true neighbor beyond them (certain on the arms check: the planted +0.0
# at length / 2 and length - 1). WHY IT STILL PROVES THE GUARDED PATH RAN:
# on this chain form the sabotaged bound has exactly one consumer, the
# `admit = pending < gate` that the ballot reads inside the guard, so the
# flip exists only if the bound reached the guard's predicate (on plain
# `voteguard` the same sabotage would flip nothing: its predicate reads
# `threshold`, and the refresh is not compiled in); and the cells that
# survive the flip (the top-k of the first 4,096 columns) were inserted by
# the guarded chain, the only insertion path in this loop form. The uniform
# and voteguard index flips are compiled out under WARPBOUND so a flip is
# never attributable to them. `warpbound_count` refuses the sabotage bit
# like `votecount`.
#
# `warpbound_count`: `warpbound_guard` with `votecount`'s counters, so the
# printed `KNN_ADMIT_RATE` line is the fraction of warp-steps on which some
# lane admits UNDER THE BOUND. Against votecount's 0.903 / 0.960 that is the
# check of C2's event model (116 / 256 = 0.45 at k10, 138 / 256 = 0.54 at
# k15); the same launcher, with `warpbound 1` on the line.
# ---------------------------------------------------------------------------
@always_inline
def _smallk_warpbound_refresh_due[EVERY: Int = SMALLK_WARPBOUND_EVERY](done: Int, fill: Int) -> Bool:
    """C2's cadence: every EVERY completed batches from `fill` on. EVERY is
    SMALLK_WARPBOUND_EVERY on every instantiation but `warpbound_guard1`
    (DEVIATION 2523), which refreshes every batch."""
    comptime assert EVERY >= 1, "the warp bound refresh cadence is at least one batch"
    return done >= fill and (done - fill) % EVERY == 0


# ---------------------------------------------------------------------------
# DEFERRED INSERTION (DEVIATION 2517): the K-chain leaves the element step.
#
# WHAT THE PHASE SPLIT SAID (brief, "Step 4 result"). Of the 8.8 (k10) to
# 12.8 ms (k15) scan, 3.2 ms is k-independent and 5.6 to 9.6 ms is the
# per-lane K-deep list, 0.80 ms per unit of k; and both bound arms halved
# the lanes' insertion events and LOST, so the K-chain's cost is paid on
# every element step whether or not the lane inserts (the warp executes the
# chain for its whole unrolled batch whenever any lane's predicate is on,
# and the predicate is on somewhere in the warp nearly always). The lever is
# therefore how often the chain EXECUTES, not how often a lane admits.
#
# THE MECHANISM. Per lane, an element step does only: the key, one compare
# against the lane's threshold, and, when admitted, an append into a
# SMALLK_DEFER_Q-slot queue held in registers (newest at slot 0, a shift of
# Q - 1 comptime-indexed moves, plus a counter). No K-chain per element.
# The queue is drained at WARP-UNIFORM points: at the end of every unrolled
# batch, and immediately after any element step at which a `vote` over the
# warp says some lane's queue is full. A drain runs `_smallk_insert`
# (unchanged) once per queued key, predicated per slot on the lane's count,
# so the warp executes the chain max(count over its lanes) times per drain
# instead of once per element step, then clears the counts. When queues
# rarely fill (the steady state: a lane admits its i-th element with
# probability about k / i), a batch costs the warp about max over 32 lanes
# of a small binomial, two to three chains instead of eight; in the first
# batches, where every lane admits everything, the queues fill every Q
# elements and the cost equals today's. Event model (pure Python, iid keys,
# 256 elements per lane, 32 lanes, 400 trials): chain executions per warp
# per launch 231 -> 116 at k10 (0.50x) and 246 -> 138 at k15 (0.56x) with
# Q = 4; Q = 8 gives 0.47x / 0.52x for twice the queue registers; Q = 2
# gives 0.66x / 0.74x. The floor is the early batches plus the warp maximum
# per batch, not Q.
#
# WHY Q = 4. Half an unrolled batch: past the fourth batch at k in 9..16 a
# lane admits under 2.4 elements per batch in expectation, so four slots
# rarely fill and the drain cadence is the batch end; four UInt64 are eight
# 32-bit registers on top of the list's 32 (CAP = 16), where eight slots
# would be sixteen for a six percent smaller chain count in the model.
# SMALLK_DEFER_Q is the comptime knob; if the gate shows the arm
# register-bound (occupancy: 256 threads a block, and every extra register
# per thread past a 64-register boundary costs a resident block per SM),
# Q = 2 is the first thing to try, Q = 8 if it is not.
#
# WHY THE OUTPUT BITS ARE UNCHANGED. Let S be the set of keys a lane scans
# (unique: each carries its column). The eager path keeps L_e = the k
# smallest of the prefix seen so far and admits p iff p < threshold_e, the
# k-th smallest of that prefix (sentinel while fewer than k). The deferred
# path keeps L_d = the k smallest of the set I of keys DRAINED so far and
# admits p iff p < threshold_d, the k-th smallest of I. I is a subset of
# the prefix, so threshold_d >= threshold_e at every step: whatever the
# eager path admits, the deferred path admits (a SUPERSET, never a subset).
# Take any x among the k smallest of S. When x is scanned, at most k - 1
# keys of S are below x, so at most k - 1 keys of I are below x, and x is
# not in I (not drained yet), so the k-th smallest of I is above x (or the
# sentinel): x is admitted, queued, and inserted at the next drain; it then
# never leaves L_d, because a key leaves only when k smaller keys of the
# same lane have been inserted and only k - 1 exist. So after the last
# drain L_d holds every one of the k smallest of S, holds exactly min(k,
# |S|) keys, and holds only keys of S: L_d is the k smallest of S, sorted,
# which is L_e. Every extra key the stale threshold admitted was inserted
# by the same `_smallk_insert`, which keeps "the list is the k smallest of
# everything inserted so far" under ANY insertion order and leaves the
# list unchanged for a key at or above its k-th (the carry runs off the
# end), so the extras cost chain executions and change nothing. Equality
# of sets of unique keys is equality of the sorted lists slot for slot;
# the rank phase reads only those lists (never the threshold), pops the
# union's exact minima with the same UInt64 compare, decides ties by the
# same index half, and gathers the winner's value from the same tile cell.
# CORNERS: the queue must be empty before the rank phase (the batch loop
# drains at every batch end and once more after the loop, a guard for any
# future cadence), and the tail loop (C4's remainder under 2,048 columns,
# a per-lane trip count where a `vote` would not be convergent) inserts
# eagerly through the same chain, starting from an empty queue; at most
# eight elements per lane, so nothing to defer there. Partitions too short
# for one batch (the carved k-wide tail) never enter the batch loop and are
# the uniform arm by construction.
#
# THE `vote`. One ballot per element step over the warp, `count == Q`, on
# the mask width of the column's lane count (SMALLK_MASK_DT, the
# ball-cover kernel's rule: a 64-lane wavefront needs a 64-bit ballot).
# Every lane reaches it (the append is closed before it, the batch trip
# count is block-uniform under C4), so it is convergent. The drain itself
# has no collective: the chain is per lane, and warp uniformity is a
# scheduling choice that makes the lanes' chains coincide.
#
# SABOTAGE (reach): at every drain the newest queued key (slot 0) is
# skipped, never inserted, on every lane whose count is nonzero. Late in
# the scan a lane's queue at the batch-end drain usually holds one key, so
# a true neighbor admitted there is the newest and is dropped with high
# probability; a row has k of them, so on the hashed fixtures thousands
# of cells move per request. On the arms check it is certain: the planted
# +0.0 at column length - 1 (the last element of the last batch, so the
# newest in lane 255's queue at that drain) is a top-k key of rows 0 and 1
# and vanishes.
# ---------------------------------------------------------------------------
comptime SMALLK_DEFER_Q = 4
comptime SMALLK_MASK_DT = DType.uint64 if SMALLK_LANES == 64 else DType.uint32


@always_inline
def _smallk_warp_any(predicate: Bool) -> Bool:
    """Warp-uniform `any` over the column's lane width, the ballot form the
    deferred arm uses; every lane must reach it."""
    return vote[SMALLK_MASK_DT](predicate) != Scalar[SMALLK_MASK_DT](0)


@always_inline
def _smallk_drain[W: Int, //, CAP: Int, SABOTAGE: Bool](
    mut local_keys: SIMD[DType.uint64, W], mut threshold: UInt64,
    queue: SIMD[DType.uint64, SMALLK_DEFER_Q], mut qcount: Int, k: Int,
):
    """Insert every queued key (slots `0 .. qcount - 1`, newest first) through
    the unchanged K-chain and clear the count. Predicated per slot, so a
    warp executes the chain max(qcount over its lanes) times. Under SABOTAGE
    slot 0 (the newest key) is never inserted."""
    comptime for s in range(SMALLK_DEFER_Q):
        comptime if not (SABOTAGE and s == 0):
            if s < qcount:
                _smallk_insert[CAP=CAP](local_keys, threshold, queue[s], k)
    qcount = 0


@always_inline
def _smallk_append(mut queue: SIMD[DType.uint64, SMALLK_DEFER_Q], mut qcount: Int, pending: UInt64):
    """Push `pending` at slot 0, shifting the older keys up one slot. Every
    index is a comptime constant, so the queue stays in registers. The
    caller drains before the count can exceed SMALLK_DEFER_Q."""
    comptime for i in range(SMALLK_DEFER_Q - 1):
        comptime s = SMALLK_DEFER_Q - 1 - i
        queue[s] = queue[s - 1]
    queue[0] = pending
    qcount += 1


def smallk_bucket_kernel[
    CAP: Int, K: Int = 0, UNIFORM: Bool = False, BOUND: Bool = False, SABOTAGE: Bool = False,
    WARPBOUND: Bool = False, PHASE: Int = SMALLK_PHASE_FULL, DEFERRED: Bool = False,
    SELP: Bool = False, CHAIN: Int = SMALLK_CHAIN_INSERT, WB_EVERY: Int = SMALLK_WARPBOUND_EVERY,
](
    values: MutPointer[Float32, MutAnyOrigin],
    out_values: MutPointer[Float32, MutAnyOrigin],
    out_indices: MutPointer[UInt32, MutAnyOrigin],
    length_in: Int32, k_in: Int32, select_min_in: Int32,
    counters: MutPointer[UInt32, MutAnyOrigin],
):
    """The small-k selector, one block of SMALLK_BLOCK threads per row.

    `counters` (DEVIATION 2522) is read or written by the VOTECOUNT chain
    form only: two UInt64 slots (warp-steps, warp-steps with any admission)
    the block adds its counts to. Every other instantiation never touches
    it (the launcher passes the output-index address as a placeholder); an
    unused kernel parameter sits in the constant bank and is never loaded,
    so the shipped kernel's code is unchanged; its `uniform` select_ms
    against steps 4 to 7 is the control for that claim.

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
    PHASE (DEVIATION 2516, TIMING ONLY, OUTPUT INVALID): SMALLK_PHASE_SKIPRANK
    runs the scan and then a digest epilogue instead of the rank phase;
    SMALLK_PHASE_SKIPSCAN skips the scan, fills every lane's k slots with a
    synthetic pattern and runs the rank phase. Trial builds only, the
    uniform scan form only, no bound, no sabotage. See the hook comment.
    DEFERRED (DEVIATION 2517): admitted keys go to a SMALLK_DEFER_Q-slot
    register queue and the K-chain runs only at warp-uniform drains (every
    batch end, and right after any element step at which a `vote` finds a
    full queue in the warp). Requires UNIFORM (the vote must be convergent)
    and a fixed-lane-width column; excludes both bounds. See the comment
    above `_smallk_drain` for the argument and the sabotage.
    CAP = K (DEVIATION 2521, arm `capk`): not a parameter of its own but the
    instantiation `[K, K, ...]`; the list is exactly K deep and its storage
    is the next power-of-two SIMD width (see the comment above
    `_smallk_insert`). SELP (arm `capk_selp`): the branch-free min/max carry
    chain in place of the per-step branch; requires CAP == K > 0 and the
    uniform scan form, excludes both bounds, deferred and the timing-only
    phases.
    CHAIN (DEVIATION 2522): what an admitted key does in the uniform scan
    form. INSERT is the shipped carry chain; NOSHIFT (timing-only, output
    invalid) overwrites the threshold slot instead; VOTEGUARD wraps the
    chain in a warp-uniform `vote` guard (output valid); VOTECOUNT is
    VOTEGUARD plus the per-block admission counter in `counters`. All
    three are the uniform scan form only, trial builds only, no C1 bound,
    no deferral, no timing-only phase. See the comment above
    `_smallk_overwrite_last`.
    WARPBOUND with CHAIN = VOTEGUARD or VOTECOUNT (DEVIATION 2523, arms
    `warpbound_guard`, `warpbound_guard1`, `warpbound_count`): C2's refresh
    at the batch boundary and the admission `pending < min(threshold,
    bound)` inside the ballot branch. WB_EVERY is the refresh cadence in
    completed batches (SMALLK_WARPBOUND_EVERY on every instantiation but
    `warpbound_guard1`, which takes 1); it is read only under WARPBOUND.
    See the comment above `_smallk_warpbound_refresh_due`.
    """
    comptime assert CAP >= 1 and CAP <= SMALLK_MAX_K, "the list depth is 1 .. SMALLK_MAX_K"
    comptime assert K == 0 or K <= CAP, "a K-specialized list must hold K keys"
    comptime assert UNIFORM or not BOUND, "C1's in-loop barrier needs C4's block-uniform trip count"
    comptime assert UNIFORM or not WARPBOUND, "C2's in-loop shuffles need C4's block-uniform trip count"
    comptime assert not (BOUND and WARPBOUND), "one bound arm at most"
    comptime assert UNIFORM or not DEFERRED, "the deferred arm's in-loop vote needs C4's block-uniform trip count"
    comptime assert not (DEFERRED and (BOUND or WARPBOUND)), "the deferred arm carries no bound"
    comptime assert PHASE == SMALLK_PHASE_FULL or not DEFERRED, "the deferred arm is a full kernel, not a timing-only phase"
    comptime assert PHASE == SMALLK_PHASE_FULL or SMALLK_SELECT_TRIAL, "timing-only phases exist on trial builds only"
    comptime assert PHASE == SMALLK_PHASE_FULL or (UNIFORM and not BOUND and not WARPBOUND and not SABOTAGE), "a timing-only phase measures the uniform default: no bound, no sabotage"
    comptime assert not SELP or (K > 0 and CAP == K), "the branch-free chain has no slot guard: it needs CAP == K"
    comptime assert not SELP or (UNIFORM and not BOUND and not WARPBOUND and not DEFERRED and PHASE == SMALLK_PHASE_FULL), "the branch-free chain is the uniform scan form only"
    comptime assert CHAIN == SMALLK_CHAIN_INSERT or SMALLK_SELECT_TRIAL or (CHAIN == SMALLK_CHAIN_VOTEGUARD and WARPBOUND and SMALLK_WARPBOUND_GUARD_DEFAULT), "the chain measurement arms exist on trial builds only (or as the flipped warpbound_guard default)"
    comptime assert CHAIN == SMALLK_CHAIN_INSERT or (UNIFORM and not BOUND and not DEFERRED and PHASE == SMALLK_PHASE_FULL), "the chain measurement arms are the uniform scan form only"
    # DEVIATION 2523: the warp bound composes with the ballot forms only.
    comptime assert not WARPBOUND or CHAIN == SMALLK_CHAIN_INSERT or CHAIN == SMALLK_CHAIN_VOTEGUARD or CHAIN == SMALLK_CHAIN_VOTECOUNT, "the warp bound composes with the vote guard (noshift has no chain to guard)"
    comptime assert WB_EVERY >= 1, "the warp bound refresh cadence is at least one batch"
    comptime assert WB_EVERY == SMALLK_WARPBOUND_EVERY or (WARPBOUND and SMALLK_SELECT_TRIAL), "a non-default refresh cadence is a warpbound trial arm"
    comptime assert not (CHAIN == SMALLK_CHAIN_NOSHIFT and SABOTAGE), "noshift is timing-only and carries no sabotage"
    comptime assert not (CHAIN == SMALLK_CHAIN_VOTECOUNT and SABOTAGE), "votecount is a counter arm and carries no sabotage"
    comptime assert not (CHAIN == SMALLK_CHAIN_NOSHIFT and SELP), "noshift has no chain to make branch-free"
    # The list's storage width: Mojo's SIMD width must be a power of two, so
    # a CAP of 10 or 15 (CAP = K, DEVIATION 2521) is stored in 16 lanes of
    # which only the first CAP are ever touched after the fill below; the
    # rest have no use and are dead in the compiled kernel. Written as a
    # comptime conditional chain, not a bit trick, so it folds with
    # certainty (`warp_topk.mojo`'s TQP). STORE == CAP for 1, 16, 32, 64.
    comptime STORE = 1 if CAP <= 1 else (
        2 if CAP <= 2 else (
            4 if CAP <= 4 else (
                8 if CAP <= 8 else (
                    16 if CAP <= 16 else (32 if CAP <= 32 else 64)
                )
            )
        )
    )
    var length = Int(length_in)
    var k = K if K > 0 else Int(k_in)
    var tid = Int(thread_idx.x)
    var row = Int(block_idx.x)
    var sentinel = UInt64(18446744073709551615)
    var local_keys = SIMD[DType.uint64, STORE](sentinel)
    # `local_keys[k - 1]`, kept in its own register so the reject test never
    # indexes the list at a runtime position.
    var threshold = sentinel
    var heads = stack_allocation[
        SMALLK_BLOCK, Scalar[DType.uint64], address_space=AddressSpace.SHARED,
    ]()
    var select_min = select_min_in != 0
    var base = row * length
    var col = tid
    # TIMING-ONLY `skipscan` (DEVIATION 2516). The three scan loops below
    # run over `scan_length`, a plain copy of `length` on every instantiation
    # but SKIPSCAN, where it is 0 and no scan loop is entered; instead every
    # lane's k slots are filled here with a synthetic pattern so the rank
    # phase does the same work it does after a real scan:
    #   ordinal(slot, tid) = slot * 256 + tid + 1      (1 .. k * 256)
    #   key = ordinal << 32 | column,  column = (ordinal * 2654435761) mod length
    # Every key is distinct in the block (the ordinal is), ascending within
    # a lane (the ordinal grows with the slot, so slot 0 is the lane's
    # minimum exactly as after a real scan), the index half is a real column
    # of the row (the winner's gather stays in range and lands on a hashed,
    # not a leading, column), and the k block minima are the k smallest
    # ordinals, one lane each, so every round has exactly one winning lane
    # that shifts its list, as in the real kernel. Slots k .. CAP - 1 stay
    # the sentinel, as after a real scan. The scan's own state (threshold,
    # gate) is left at the sentinel because nothing reads it afterwards.
    var scan_length = length
    comptime if PHASE == SMALLK_PHASE_SKIPSCAN:
        scan_length = 0
        comptime for slot in range(CAP):
            if slot < k:
                var ordinal = UInt64(slot * SMALLK_BLOCK + tid + 1)
                var column = (ordinal * UInt64(2654435761)) % UInt64(length)
                local_keys[slot] = (ordinal << UInt64(32)) | column
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
    # DEFERRED state (DEVIATION 2517): the register queue of admitted keys
    # (newest at slot 0) and its count. Constants unless DEFERRED, so they
    # fold away on every other instantiation.
    var queue = SIMD[DType.uint64, SMALLK_DEFER_Q](sentinel)
    var qcount = 0
    # VOTECOUNT state (DEVIATION 2522): element steps this lane's warp took
    # in the uniform batch loop, and those on which some lane admitted.
    # Identical across a warp's lanes (the loop and the ballot are
    # warp-uniform). Constants unless VOTECOUNT, so they fold away.
    var warp_steps = 0
    var admit_steps = 0
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
        while batch_base + SMALLK_SCAN_SPAN <= scan_length:
            var batch = SIMD[DType.float32, SMALLK_SCAN_UNROLL](0.0)
            comptime for u in range(SMALLK_SCAN_UNROLL):
                batch[u] = values.unsafe_load(base + batch_base + tid + u * SMALLK_BLOCK)
            comptime for u in range(SMALLK_SCAN_UNROLL):
                var pending = composite_key(
                    batch[u], UInt32(batch_base + tid + u * SMALLK_BLOCK), select_min
                )
                comptime if SABOTAGE and (not BOUND) and (not WARPBOUND) and (not DEFERRED) and CHAIN != SMALLK_CHAIN_VOTEGUARD and u == 0:
                    # `uniform` arm reach: bit 0 of the index half of the
                    # first element of every batch is flipped, so one
                    # candidate column in eight carries its neighbor's
                    # index and the gathered value moves with it. Only
                    # this loop form carries it: a flip proves this loop.
                    # (`voteguard` carries the same flip inside its guard
                    # below, so its reach proves the guarded body.)
                    pending = pending ^ UInt64(1)
                comptime if BOUND or (WARPBOUND and CHAIN == SMALLK_CHAIN_INSERT):
                    # C1 and C2 (`headbound`, `warpbound`): the gate as the
                    # predicate, the chain as the uniform arm runs it. The
                    # C2 + vote guard composition (DEVIATION 2523) takes
                    # the ballot branch below instead.
                    if pending < gate:
                        _smallk_insert[CAP=CAP](local_keys, threshold, pending, k)
                        gate = threshold if threshold < bound else bound
                elif DEFERRED:
                    # DEFERRED element step: key, compare, append. The
                    # threshold may be stale (last drain), which admits a
                    # superset of the eager path's keys; see `_smallk_drain`.
                    if pending < threshold:
                        _smallk_append(queue, qcount, pending)
                    # Warp-uniform full test: every lane votes, every lane
                    # drains, so the lanes' chains coincide. Convergent: the
                    # append above is closed and the batch trip count is
                    # block-uniform (C4).
                    if vote[SMALLK_MASK_DT](qcount == SMALLK_DEFER_Q) != Scalar[SMALLK_MASK_DT](0):
                        _smallk_drain[CAP=CAP, SABOTAGE=SABOTAGE](local_keys, threshold, queue, qcount, k)
                elif CHAIN == SMALLK_CHAIN_NOSHIFT:
                    # TIMING-ONLY `noshift` (DEVIATION 2522): the compare,
                    # then the admitted key overwrites the threshold slot;
                    # no carry chain. See `_smallk_overwrite_last`.
                    if pending < threshold:
                        _smallk_overwrite_last[CAP=CAP](local_keys, threshold, pending, k)
                elif CHAIN == SMALLK_CHAIN_VOTEGUARD or CHAIN == SMALLK_CHAIN_VOTECOUNT:
                    # `voteguard` (DEVIATION 2522): the same admission, but
                    # the chain sits behind a warp-uniform ballot. Every
                    # lane votes (convergent: the batch trip count is
                    # block-uniform, C4, and nothing above diverges), so a
                    # warp with no admitting lane skips the chain by a
                    # real branch; a lane that admits inserts the same
                    # key at the same step as the uniform arm.
                    #
                    # Under WARPBOUND (DEVIATION 2523, `warpbound_guard`):
                    # the predicate is C2's gate, min(threshold, bound),
                    # refreshed at the batch boundary below, outside this
                    # branch; the bound only lowers the predicate, the
                    # ballot and the chain are the voteguard form's. See
                    # the comment above `_smallk_warpbound_refresh_due`.
                    var admit = Bool(pending < threshold)
                    comptime if WARPBOUND:
                        # The compare above is dead on this instantiation
                        # (the gate is at most the threshold) and folds out.
                        admit = Bool(pending < gate)
                    var any_admit = _smallk_warp_any(admit)
                    comptime if CHAIN == SMALLK_CHAIN_VOTECOUNT:
                        warp_steps += 1
                        if any_admit:
                            admit_steps += 1
                    if any_admit:
                        if admit:
                            comptime if SABOTAGE and u == 0 and not WARPBOUND:
                                # `voteguard` reach: the uniform flip, but
                                # inside the guarded and admitted path, so
                                # a flip proves this body ran. Under
                                # WARPBOUND the reach is the bound's own
                                # sabotage in the refresh (bit 63), so this
                                # flip is compiled out there.
                                pending = pending ^ UInt64(1)
                            _smallk_insert[CAP=CAP, SELP=SELP](local_keys, threshold, pending, k)
                            comptime if WARPBOUND:
                                gate = threshold if threshold < bound else bound
                else:
                    if pending < threshold:
                        _smallk_insert[CAP=CAP, SELP=SELP](local_keys, threshold, pending, k)
            comptime if DEFERRED:
                # Batch-end drain, block-uniform under C4: after it every
                # lane's threshold is its true k-th of everything scanned.
                _smallk_drain[CAP=CAP, SABOTAGE=SABOTAGE](local_keys, threshold, queue, qcount, k)
            batch_base += SMALLK_SCAN_SPAN
            done += 1
            comptime if WARPBOUND and SMALLK_SHUFFLE:
                if _smallk_warpbound_refresh_due[WB_EVERY](done, wb_fill):
                    # C2 REFRESH. `done` is block-uniform (C4), so every
                    # lane of every warp is here, and the butterfly is
                    # convergent; under the vote guard forms (DEVIATION
                    # 2523) this point is outside the ballot branch (the
                    # `u` loop above is closed), so the ballots inside the
                    # batch cannot keep a lane from it. Every lane publishes its wb_depth-th
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
                        # `warpbound` arm reach (and `warpbound_guard` /
                        # `warpbound_guard1`, DEVIATION 2523, whose only
                        # consumer of the bound is the guard's predicate):
                        # bit 63 of the reduced bound
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
        comptime if DEFERRED:
            # The queue MUST be empty before the tail loop and the rank
            # phase. It already is (every batch drained at its end); this
            # drain is the guard that keeps the invariant if the cadence
            # ever moves, and it costs SMALLK_DEFER_Q compares once. The
            # tail loop below inserts eagerly through the same chain from
            # this empty queue; a `vote` there would not be convergent (the
            # tail's trip count is per lane), and it holds at most eight
            # elements per lane.
            _smallk_drain[CAP=CAP, SABOTAGE=SABOTAGE](local_keys, threshold, queue, qcount, k)
        col = batch_base + tid
    else:
        # The 2026-09-09 form: the batch condition is per thread.
        while col + (SMALLK_SCAN_UNROLL - 1) * SMALLK_BLOCK < scan_length:
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
                    _smallk_insert[CAP=CAP](local_keys, threshold, pending, k)
            col += SMALLK_SCAN_SPAN
    while col < scan_length:
        var pending = composite_key(values.unsafe_load(base + col), UInt32(col), select_min)
        comptime if BOUND or WARPBOUND:
            # The bound arms' tail, the vote guard compositions included
            # (DEVIATION 2523): the gate as the predicate, no ballot (the
            # trip count is per lane), at most eight elements per lane.
            if pending < gate:
                _smallk_insert[CAP=CAP](local_keys, threshold, pending, k)
                gate = threshold if threshold < bound else bound
        elif CHAIN == SMALLK_CHAIN_NOSHIFT:
            if pending < threshold:
                _smallk_overwrite_last[CAP=CAP](local_keys, threshold, pending, k)
        else:
            # The tail's trip count is per lane, so no ballot here: the
            # voteguard forms take the plain admission on these at most
            # SMALLK_SCAN_UNROLL elements per lane (DEVIATION 2522).
            if pending < threshold:
                _smallk_insert[CAP=CAP, SELP=SELP](local_keys, threshold, pending, k)
        col += SMALLK_BLOCK
    comptime if CHAIN == SMALLK_CHAIN_VOTECOUNT:
        # VOTECOUNT epilogue (DEVIATION 2522): lane 0 of every warp parks
        # its warp's two counts in shared memory (slots 0 .. 2 * WARPS - 1
        # of `heads`, free until the rank phase's first write, which the
        # second barrier orders after thread 0's read), thread 0 folds them
        # and adds the block's pair to the device counter with two atomics.
        # Arrival order cannot matter: integer addition commutes.
        if tid % SMALLK_LANES == 0:
            heads[tid // SMALLK_LANES] = UInt64(warp_steps)
            heads[SMALLK_WARPS + tid // SMALLK_LANES] = UInt64(admit_steps)
        barrier()
        if tid == 0:
            var block_steps = UInt64(0)
            var block_admits = UInt64(0)
            comptime for w in range(SMALLK_WARPS):
                block_steps += heads[w]
                block_admits += heads[SMALLK_WARPS + w]
            # UInt32 atomics: every column has them (Metal has no 64-bit
            # atomic add); one launch's totals fit (under 2^22 warp-steps).
            _ = Atomic.fetch_add(counters, UInt32(block_steps))
            _ = Atomic.fetch_add(counters.unsafe_offset(1), UInt32(block_admits))
        barrier()
    comptime if PHASE == SMALLK_PHASE_SKIPRANK:
        # TIMING-ONLY `skiprank` (DEVIATION 2516): no rank phase. The scan's
        # result must be CONSUMED or the compiler may drop the scan (its
        # loads have no other side effect): every lane folds its whole list
        # and its threshold into one XOR digest, the block folds the 256
        # digests (the rank phase's own butterfly shape where the column has
        # fixed-width lanes, the shared tree elsewhere), and thread 0 uses
        # the block digest as a GATHER ADDRESS into the row and writes the
        # gathered cell and the column to slot 0. A load address that
        # depends on every lane's list keeps every insert live; the shuffle
        # or shared fold is a collective every lane takes, so no lane's scan
        # can be sunk under thread 0's branch. Slots 1 .. k - 1 get the
        # sentinel key's two halves (index 0xFFFFFFFF, value bits
        # 0xFFFFFFFF). Cost of this epilogue: about ONE rank round (ten
        # shuffles, one barrier, eight shared loads, one gather) and it does
        # not depend on k, so select_ms(skiprank) overstates the scan by
        # about one rank round and its k-slope is the scan's k-slope alone.
        var digest = threshold
        comptime for slot in range(CAP):
            digest = digest ^ local_keys[slot]
        var block_digest = digest
        comptime if SMALLK_SHUFFLE:
            var offset = 1
            while offset < SMALLK_LANES:
                digest = digest ^ _smallk_shuffle_xor_u64(digest, offset)
                offset *= 2
            if tid % SMALLK_LANES == 0:
                heads[tid // SMALLK_LANES] = digest
            barrier()
            block_digest = heads[0]
            comptime for w in range(1, SMALLK_WARPS):
                block_digest = block_digest ^ heads[w]
        else:
            heads[tid] = digest
            barrier()
            block_digest = heads[0]
            for t in range(1, SMALLK_BLOCK):
                block_digest = block_digest ^ heads[t]
        if tid == 0:
            var probe = Int(block_digest % UInt64(length))
            out_indices.unsafe_store(row * k, UInt32(probe))
            out_values.unsafe_store(row * k, values.unsafe_load(base + probe))
            for r in range(1, k):
                out_indices.unsafe_store(row * k + r, UInt32(4294967295))
                out_values.unsafe_store(row * k + r, bitcast[DType.float32](UInt32(4294967295)))
    elif SMALLK_SHUFFLE:
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
                comptime if CHAIN == SMALLK_CHAIN_NOSHIFT:
                    # TIMING-ONLY `noshift`: the first K - 1 winners are
                    # the sentinel; keep thread 0's gather in the row.
                    selected = selected % UInt32(length)
                out_indices.unsafe_store(row * k + rank, selected)
                out_values.unsafe_store(row * k + rank, values.unsafe_load(row * length + Int(selected)))
            if mine == winner:
                # The pop: slots 1 .. CAP - 1 move down, the last becomes
                # the sentinel. This is the one place that reads slots at
                # or above K on a CAP = 16, K < 16 kernel (they are the
                # sentinel there, so slots 0 .. K - 1 end up the same as
                # under CAP = K, DEVIATION 2521); it is also what keeps
                # those lanes live across the scan on the shipped kernels.
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
                comptime if CHAIN == SMALLK_CHAIN_NOSHIFT:
                    # TIMING-ONLY `noshift`: the first K - 1 winners are
                    # the sentinel; keep thread 0's gather in the row.
                    selected = selected % UInt32(length)
                out_indices.unsafe_store(row * k + rank, selected)
                out_values.unsafe_store(row * k + rank, values.unsafe_load(row * length + Int(selected)))
            if mine == winner:
                comptime for slot in range(CAP - 1):
                    local_keys[slot] = local_keys[slot + 1]
                local_keys[CAP - 1] = sentinel
            barrier()


@always_inline
def _smallk_enqueue[
    CAP: Int, K: Int, UNIFORM: Bool, BOUND: Bool, SABOTAGE: Bool, WARPBOUND: Bool = False,
    PHASE: Int = SMALLK_PHASE_FULL, DEFERRED: Bool = False, SELP: Bool = False,
    CHAIN: Int = SMALLK_CHAIN_INSERT, WB_EVERY: Int = SMALLK_WARPBOUND_EVERY,
](
    ctx: DeviceContext,
    values: MutPointer[Float32, MutAnyOrigin],
    out_values: MutPointer[Float32, MutAnyOrigin],
    out_indices: MutPointer[UInt32, MutAnyOrigin],
    rows: Int, length: Int, k: Int, select_min: Bool,
) raises:
    """Every instantiation but VOTECOUNT: the kernel's `counters` argument
    is a placeholder (the output-index address, never dereferenced; the
    VOTECOUNT code that reads it is folded out). VOTECOUNT launches go
    through `_smallk_launch_votecount` instead, which owns the counter."""
    comptime assert CHAIN != SMALLK_CHAIN_VOTECOUNT, "votecount launches carry a real counter: use _smallk_launch_votecount"
    var placeholder = MutPointer[UInt32, MutAnyOrigin](unsafe_from_address=Int(out_indices))
    ctx.enqueue_function[smallk_bucket_kernel[CAP, K, UNIFORM, BOUND, SABOTAGE, WARPBOUND, PHASE, DEFERRED, SELP, CHAIN, WB_EVERY]](
        values, out_values, out_indices,
        Int32(length), Int32(k), Int32(select_min), placeholder,
        grid_dim=(rows, 1, 1), block_dim=(SMALLK_BLOCK, 1, 1),
    )


@always_inline
def _smallk_launch_votecount[CAP: Int, K: Int, WARPBOUND: Bool = False](
    ctx: DeviceContext,
    values: MutPointer[Float32, MutAnyOrigin],
    out_values: MutPointer[Float32, MutAnyOrigin],
    out_indices: MutPointer[UInt32, MutAnyOrigin],
    rows: Int, length: Int, k: Int, select_min: Bool,
) raises:
    """`votecount` (DEVIATION 2522): the VOTEGUARD kernel with the
    per-block admission counter. A two-slot device counter is zeroed from
    a host buffer before the launch, read back after it, and the launch is
    SYNCHRONIZED here (the readback needs it), so this arm's select_ms is
    not a price: only the printed counts are the measurement. Under the
    phase-timer build the line `KNN_ADMIT_RATE warp_steps N any_admit M`
    goes to fd 1, one per launch; the gate harness sums the lines of a
    request. Without that define the counts are gathered and dropped.
    WARPBOUND (DEVIATION 2523, `warpbound_count`): the same counter on the
    WARPBOUND + VOTECOUNT instantiation at C2's cadence, so the line is the
    admit rate UNDER THE BOUND; the line carries `warpbound 1` then."""
    var counters = ctx.enqueue_create_buffer[DType.uint32](2)
    var host = ctx.enqueue_create_host_buffer[DType.uint32](2)
    ctx.synchronize()
    host.unsafe_ptr().unsafe_store(0, UInt32(0))
    host.unsafe_ptr().unsafe_store(1, UInt32(0))
    ctx.enqueue_copy(dst_buf=counters, src_ptr=host.unsafe_ptr())
    ctx.enqueue_function[smallk_bucket_kernel[
        CAP, K, True, False, False, WARPBOUND, SMALLK_PHASE_FULL, False, False, SMALLK_CHAIN_VOTECOUNT
    ]](
        values, out_values, out_indices,
        Int32(length), Int32(k), Int32(select_min), counters.unsafe_ptr(),
        grid_dim=(rows, 1, 1), block_dim=(SMALLK_BLOCK, 1, 1),
    )
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=counters)
    ctx.synchronize()
    var warp_steps = host.unsafe_ptr().unsafe_load(0)
    var any_admit = host.unsafe_ptr().unsafe_load(1)
    comptime if SMALLK_PHASE_TIMERS:
        print(
            "KNN_ADMIT_RATE", "warp_steps", warp_steps, "any_admit", any_admit,
            "rows", rows, "length", length, "k", k, "warpbound", Int(WARPBOUND),
        )
    # Both buffers outlive the synchronization above (buffer-freed-at-last-use).
    _ = counters^
    _ = host^


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
    that asks for an arm cannot pass silently on a non-trial build).

    CAP = K (DEVIATION 2521): the `capk` arms instantiate `[K, K, ...]` on
    the K-specialized buckets (K > 0). The generic bucket (K == 0) has no
    CAP = K instantiation (k is a runtime value there) and the arms RAISE
    on it rather than run the CAP = 16 uniform kernel under their name;
    a column without the specialization row takes them only with
    `-D MOJOLEARN_KNN_IDENTICAL_SPECIALIZE_COMMON=1`.
    """
    comptime assert not (SMALLK_HEAD_BOUND_DEFAULT and SMALLK_WARPBOUND_DEFAULT), "one bound arm at most"
    comptime assert not (SMALLK_DEFERRED_DEFAULT and (SMALLK_HEAD_BOUND_DEFAULT or SMALLK_WARPBOUND_DEFAULT)), "the deferred default excludes both bound defaults"
    comptime assert not (SMALLK_CAPK_DEFAULT and (SMALLK_HEAD_BOUND_DEFAULT or SMALLK_WARPBOUND_DEFAULT or SMALLK_DEFERRED_DEFAULT)), "the CAP = K default is the uniform arm's scan: no bound, no deferral"
    comptime assert not SMALLK_SELP_DEFAULT or SMALLK_CAPK_DEFAULT, "the branch-free chain default requires the CAP = K default"
    comptime assert not SMALLK_CAPK_DEFAULT or SMALLK_UNIFORM_TRIP_DEFAULT, "the CAP = K default is the uniform scan form"
    comptime assert not SMALLK_WARPBOUND_GUARD_DEFAULT or (SMALLK_UNIFORM_TRIP_DEFAULT and SMALLK_SHUFFLE), "the warpbound_guard default needs C4 and a fixed-lane-width column (DEVIATION 2523)"
    comptime assert not (SMALLK_WARPBOUND_GUARD_DEFAULT and (SMALLK_HEAD_BOUND_DEFAULT or SMALLK_WARPBOUND_DEFAULT or SMALLK_DEFERRED_DEFAULT or SMALLK_CAPK_DEFAULT or SMALLK_SELP_DEFAULT)), "the warpbound_guard default excludes every other candidate default"
    # The default path's list depth: CAP = K on the K-specialized buckets
    # once the capk gate flips SMALLK_CAPK_DEFAULT, the bucket capacity
    # otherwise (today). K == 0 (the generic bucket) always keeps CAP.
    comptime DEFAULT_CAP = K if (SMALLK_CAPK_DEFAULT and K > 0) else CAP
    # The default path's bound and chain form (DEVIATION 2523): C2's bound
    # with the vote guard once SMALLK_WARPBOUND_GUARD_DEFAULT flips; today
    # both fold to the shipped values (no bound, the INSERT chain), so the
    # non-trial instantiation below is the one that shipped.
    comptime DEFAULT_WARPBOUND = SMALLK_WARPBOUND_DEFAULT or SMALLK_WARPBOUND_GUARD_DEFAULT
    comptime DEFAULT_CHAIN = SMALLK_CHAIN_VOTEGUARD if SMALLK_WARPBOUND_GUARD_DEFAULT else SMALLK_CHAIN_INSERT
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
        elif which == SMALLK_ARM_DEFERRED:
            comptime if not SMALLK_SHUFFLE:
                # The full test is a warp ballot on the column's lane
                # width; a column whose lane width the vendor's compiler
                # chooses per kernel has no convergent warp to ballot.
                # Refuse rather than run the uniform arm under this name.
                raise Error("small-k selector: the deferred arm needs a fixed-lane-width column")
            if sabotage:
                _smallk_enqueue[CAP, K, True, False, True, False, SMALLK_PHASE_FULL, True](
                    ctx, values, out_values, out_indices, rows, length, k, select_min
                )
            else:
                _smallk_enqueue[CAP, K, True, False, False, False, SMALLK_PHASE_FULL, True](
                    ctx, values, out_values, out_indices, rows, length, k, select_min
                )
        elif which == SMALLK_ARM_CAPK or which == SMALLK_ARM_CAPK_SELP:
            # CAP = K (DEVIATION 2521): the uniform arm's kernel with the
            # list exactly K deep; `capk_selp` adds the branch-free chain.
            # Only a K-specialized bucket has a comptime K to fold into CAP.
            comptime if K == 0:
                raise Error(
                    "small-k selector: the capk arms need a K-specialized bucket"
                    + " (k 10 or 15 on a column with knn_selector_specialize_common_for,"
                    + " or -D MOJOLEARN_KNN_IDENTICAL_SPECIALIZE_COMMON=1)"
                )
            else:
                if which == SMALLK_ARM_CAPK_SELP:
                    if sabotage:
                        _smallk_enqueue[K, K, True, False, True, False, SMALLK_PHASE_FULL, False, True](
                            ctx, values, out_values, out_indices, rows, length, k, select_min
                        )
                    else:
                        _smallk_enqueue[K, K, True, False, False, False, SMALLK_PHASE_FULL, False, True](
                            ctx, values, out_values, out_indices, rows, length, k, select_min
                        )
                else:
                    if sabotage:
                        _smallk_enqueue[K, K, True, False, True](ctx, values, out_values, out_indices, rows, length, k, select_min)
                    else:
                        _smallk_enqueue[K, K, True, False, False](ctx, values, out_values, out_indices, rows, length, k, select_min)
        elif which == SMALLK_ARM_VOTEGUARD:
            # The vote guard (DEVIATION 2522): the uniform arm's kernel
            # with the K-chain behind a warp-uniform ballot. Output valid;
            # reach through the uniform flip inside the guarded body.
            comptime if not SMALLK_SHUFFLE:
                # The guard is a warp ballot on the column's lane width; a
                # column whose lane width the vendor's compiler chooses
                # per kernel has no convergent warp to ballot. Refuse
                # rather than run the uniform arm under this name.
                raise Error("small-k selector: the voteguard arm needs a fixed-lane-width column")
            else:
                if sabotage:
                    _smallk_enqueue[CAP, K, True, False, True, False, SMALLK_PHASE_FULL, False, False, SMALLK_CHAIN_VOTEGUARD](
                        ctx, values, out_values, out_indices, rows, length, k, select_min
                    )
                else:
                    _smallk_enqueue[CAP, K, True, False, False, False, SMALLK_PHASE_FULL, False, False, SMALLK_CHAIN_VOTEGUARD](
                        ctx, values, out_values, out_indices, rows, length, k, select_min
                    )
        elif which == SMALLK_ARM_VOTECOUNT:
            # The admission counter (DEVIATION 2522): `voteguard` plus the
            # per-block count, read back and printed by its own launcher.
            # Output valid, time not a price, no sabotage.
            comptime if not SMALLK_SHUFFLE:
                raise Error("small-k selector: the votecount arm needs a fixed-lane-width column")
            else:
                if sabotage:
                    raise Error("small-k selector: the votecount arm carries no sabotage (a counter arm; the gate lists it as timing-only)")
                _smallk_launch_votecount[CAP, K](ctx, values, out_values, out_indices, rows, length, k, select_min)
        elif which == SMALLK_ARM_WARPBOUND_GUARD or which == SMALLK_ARM_WARPBOUND_GUARD1:
            # C2's warp bound inside the vote guard (DEVIATION 2523): the
            # WARPBOUND + VOTEGUARD instantiation, at C2's cadence
            # (`warpbound_guard`) or refreshed every batch
            # (`warpbound_guard1`). Output valid; reach through C2's
            # bit-63 sabotage in the refresh (see the comment above
            # `_smallk_warpbound_refresh_due`).
            comptime if not SMALLK_SHUFFLE:
                # Both the refresh (shuffles) and the guard (a ballot)
                # need a convergent warp of fixed width. Refuse rather
                # than run the uniform arm under this name.
                raise Error("small-k selector: the warpbound_guard arms need a fixed-lane-width column")
            else:
                if which == SMALLK_ARM_WARPBOUND_GUARD1:
                    if sabotage:
                        _smallk_enqueue[CAP, K, True, False, True, True, SMALLK_PHASE_FULL, False, False, SMALLK_CHAIN_VOTEGUARD, 1](
                            ctx, values, out_values, out_indices, rows, length, k, select_min
                        )
                    else:
                        _smallk_enqueue[CAP, K, True, False, False, True, SMALLK_PHASE_FULL, False, False, SMALLK_CHAIN_VOTEGUARD, 1](
                            ctx, values, out_values, out_indices, rows, length, k, select_min
                        )
                else:
                    if sabotage:
                        _smallk_enqueue[CAP, K, True, False, True, True, SMALLK_PHASE_FULL, False, False, SMALLK_CHAIN_VOTEGUARD](
                            ctx, values, out_values, out_indices, rows, length, k, select_min
                        )
                    else:
                        _smallk_enqueue[CAP, K, True, False, False, True, SMALLK_PHASE_FULL, False, False, SMALLK_CHAIN_VOTEGUARD](
                            ctx, values, out_values, out_indices, rows, length, k, select_min
                        )
        elif which == SMALLK_ARM_WARPBOUND_COUNT:
            # The admission counter under the bound (DEVIATION 2523):
            # `warpbound_guard` plus `votecount`'s counter, C2's cadence.
            # Output valid, time not a price, no sabotage.
            comptime if not SMALLK_SHUFFLE:
                raise Error("small-k selector: the warpbound_count arm needs a fixed-lane-width column")
            else:
                if sabotage:
                    raise Error("small-k selector: the warpbound_count arm carries no sabotage (a counter arm; the gate lists it as timing-only)")
                _smallk_launch_votecount[CAP, K, True](ctx, values, out_values, out_indices, rows, length, k, select_min)
        elif which == SMALLK_ARM_SKIPRANK or which == SMALLK_ARM_SKIPSCAN or which == SMALLK_ARM_SCANONLY1 or which == SMALLK_ARM_NOSHIFT:
            # TIMING-ONLY arms (DEVIATION 2516, and `noshift` of DEVIATION
            # 2522): the uniform default's scan form, no bound, and never a
            # sabotage instantiation (their output is invalid by design;
            # there is no reach to prove).
            if sabotage:
                raise Error("small-k selector: timing-only arms carry no sabotage")
            if which == SMALLK_ARM_NOSHIFT:
                _smallk_enqueue[CAP, K, True, False, False, False, SMALLK_PHASE_FULL, False, False, SMALLK_CHAIN_NOSHIFT](
                    ctx, values, out_values, out_indices, rows, length, k, select_min
                )
            elif which == SMALLK_ARM_SKIPRANK:
                _smallk_enqueue[CAP, K, True, False, False, False, SMALLK_PHASE_SKIPRANK](
                    ctx, values, out_values, out_indices, rows, length, k, select_min
                )
            elif which == SMALLK_ARM_SKIPSCAN:
                _smallk_enqueue[CAP, K, True, False, False, False, SMALLK_PHASE_SKIPSCAN](
                    ctx, values, out_values, out_indices, rows, length, k, select_min
                )
            else:
                # `scanonly1`: the same body with a one-key list (CAP = 1,
                # K = 1 folded), so the scan keeps a running minimum and
                # nothing else; the runtime k is ignored by the kernel and
                # thread 0 writes one slot per row.
                _smallk_enqueue[1, 1, True, False, False, False, SMALLK_PHASE_SKIPRANK](
                    ctx, values, out_values, out_indices, rows, length, k, select_min
                )
        else:
            raise Error("small-k selector: unknown arm " + String(arm))
    else:
        if arm != SMALLK_ARM_DEFAULT:
            raise Error(
                "small-k selector: arm " + String(arm)
                + " needs a build with -D MOJOLEARN_KNN_SELECT_TRIAL=1"
            )
        _smallk_enqueue[
            DEFAULT_CAP, K, SMALLK_UNIFORM_TRIP_DEFAULT, SMALLK_HEAD_BOUND_DEFAULT, False, DEFAULT_WARPBOUND,
            SMALLK_PHASE_FULL, SMALLK_DEFERRED_DEFAULT, SMALLK_SELP_DEFAULT, DEFAULT_CHAIN,
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
