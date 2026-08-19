"""One table: every kernel, every tunable it takes, per GPU vendor.

NOT a port. CatBoost tunes against `TArchProps` scattered through the kernel
files and ships one vendor. **We ship Metal, CUDA and HIP from one source and
the product property is that a fit gives the SAME MODEL on all three.** That
cannot be maintained by constants sprinkled across kernels, so every knob any
kernel takes is declared here, once, with its vendor column and its class.

There is no CPU column. This tree has no CPU path.

CLASSIFICATION, the one question
--------------------------------
Each row is NUMERIC or SCHEDULING by the single test:

    Does it change the SEQUENCE or the PRECISION of the arithmetic?

SCHEDULING rows vary per vendor in every mode: they decide which thread does
which work, never what is added to what. NUMERIC rows change the answer, so
under the default they are pinned to a column every vendor can meet.

**A row is not scheduling because it looks like geometry.** `block_size` is
scheduling; `replication_lanes` is numeric, and they are usually chosen
together. Getting this backwards is how a bit-identity claim becomes false.

THE CROSS-VENDOR HAZARD THIS TABLE EXISTS FOR
---------------------------------------------
CatBoost's `SliceOffset()` is `512 * (threadIdx.x / 32) + (threadIdx.x & 24)`
and its inner loop syncs a `tiled_partition<8>`. **Both hardcode a 32-lane
warp.** That is true on NVIDIA and on Apple, and FALSE on AMD, whose
wavefront is 64. Ported literally, the same source would replicate the
histogram into a different number of private copies on AMD and reduce a
different number of partial sums in a different order, so a CUDA fit and a
HIP fit would disagree in the last bits with nothing in the code admitting
it.

So `replication_lanes` is a declared NUMERIC row pinned to 32, rather than
`warp_size` read from the device. AMD pays a wider sync for it. That is the
cost of the product property, and it is a number we can measure rather than
an argument we have to have.
"""

from mojo_only.numerics import NumericMode, NUMERIC_FAST, NUMERIC_IDENTICAL


# --- the four columns ----------------------------------------------------
#
# BIT_IDENTICAL is a real column, not a mode flag. It holds the value every
# vendor can meet, so it is the INTERSECTION of the other three: where they
# disagree it takes the most restrictive. That is what makes it safe, and it
# is why it can be read, printed and diffed against a vendor column to show
# exactly what identity costs on that device.
comptime COLUMN_BIT_IDENTICAL = 0
comptime COLUMN_APPLE = 1
comptime COLUMN_NVIDIA = 2
comptime COLUMN_AMD = 3

#: Kept as aliases so kernel code can name the API it targets rather than the
#: company. Metal is Apple's, CUDA is NVIDIA's, HIP is AMD's.
comptime COLUMN_METAL = COLUMN_APPLE
comptime COLUMN_CUDA = COLUMN_NVIDIA
comptime COLUMN_HIP = COLUMN_AMD


def column_name(column: Int) -> String:
    if column == COLUMN_BIT_IDENTICAL:
        return String("bit-identical")
    if column == COLUMN_APPLE:
        return String("apple")
    if column == COLUMN_NVIDIA:
        return String("nvidia")
    return String("amd")

# --- the kernels ---------------------------------------------------------
comptime K_HIST_BINARY = 0
comptime K_HIST_HALF_BYTE = 1
comptime K_HIST_ONE_BYTE = 2
comptime K_SCAN = 3
comptime K_SUBTRACT = 4
comptime K_SCORES = 5
comptime K_SPLIT_POINTS = 6

#: NUMERIC. Lanes one private histogram copy is shared by. CatBoost's 32,
#: pinned for every vendor because AMD's wavefront is 64 and letting it
#: follow the hardware changes the reduction tree. See the module docstring.
comptime PINNED_REPLICATION_LANES = 32

#: NUMERIC. Width of `Reduce()`'s first stage. CatBoost writes the literal
#: 512, which is only correct because its block is 768. Pinned so the
#: reduction tree has one shape everywhere.
comptime PINNED_REDUCE_WIDTH = 512


@fieldwise_init
struct KernelSpec(Copyable, Movable):
    """Every knob one kernel takes. Resolved once, never re-derived."""

    var block_size: Int
    """SCHEDULING. Threads per threadgroup."""

    var hist_floats_per_thread: Int
    """NUMERIC, and it reads as a memory-budget row. Shared scratch is
    `block_size * this`, and the product decides how many private copies
    exist, hence how many partial sums are reduced."""

    var features_per_int: Int
    """NUMERIC in effect: it is the packing, so it decides which features
    share a load and therefore which sums are formed. Fixed by the grid
    policy, never by the vendor."""

    var replication_lanes: Int
    """NUMERIC. See PINNED_REPLICATION_LANES."""

    var reduce_width: Int
    """NUMERIC. See PINNED_REDUCE_WIDTH."""

    var deterministic_flush: Bool
    """NUMERIC. Fixed-point integer flush instead of float `atomicAdd`.

    Forced true on Apple whatever the mode asks for: Metal has no float
    atomic add, so the alternative does not exist. See
    `column_has_float_atomics`."""

    var flush_forced_by_vendor: Bool
    """Whether `deterministic_flush` is the mode's choice or the vendor's
    constraint, so a report can say "Apple is deterministic because Metal
    cannot do otherwise" rather than implying the user asked for it."""

    def shared_bytes(self) -> Int:
        """What this spec asks of threadgroup memory."""
        return self.block_size * self.hist_floats_per_thread * 4


def column_shared_limit(column: Int) -> Int:
    """Threadgroup / shared / LDS bytes a single block may claim.

    `BIT_IDENTICAL` takes the MINIMUM across the three, which is Apple's 32
    KB, because a shared-memory budget decides the replication factor and the
    replication factor is numeric. A safe column that let NVIDIA claim 48 KB
    would not be safe.
    """
    if column == COLUMN_APPLE:
        return 32 * 1024
    if column == COLUMN_NVIDIA:
        return 48 * 1024
    if column == COLUMN_AMD:
        return 64 * 1024
    return 32 * 1024  # BIT_IDENTICAL: the intersection, Apple-bound


def column_has_float_atomics(column: Int) -> Bool:
    """Whether this vendor can do `atomicAdd` on a `float` at all.

    **APPLE CANNOT. Metal has no floating-point atomic add.** That is not a
    tuning preference, it is an absent instruction, and it makes CatBoost's
    flush (`atomicAdd(dst + fold, val)` in every `AddToGlobalMemory`)
    unportable rather than merely non-deterministic.

    The consequence for the table is that `deterministic_flush` is a numeric
    row the apple column CANNOT negotiate: `FAST` does not turn it off there,
    because there is nothing to turn on. `spec_for` forces it rather than
    letting a mode request produce a kernel that cannot be compiled.

    The pleasant corollary: on Apple the flush is fixed-point integer in both
    modes, integer addition is associative, and Apple's lane width already
    equals the pinned 32, so **Apple is reproducible by default and identity
    costs it nothing.** The price of the bit-identical column is paid by
    NVIDIA and AMD, which is the opposite of what one would guess.
    """
    return column != COLUMN_APPLE and column != COLUMN_BIT_IDENTICAL


def column_lane_width(column: Int) -> Int:
    """Hardware lanes that move in lockstep: warp on NVIDIA, SIMD group on
    Apple, WAVEFRONT on AMD.

    **This is the cross-vendor hazard in one number.** 32, 32, 64. CatBoost
    hardcodes 32 throughout, which is right on two of the three. Reading it
    from the device would change the replication geometry on AMD and silently
    move the last bits, so `BIT_IDENTICAL` pins 32 and AMD pays a wider sync
    for it.
    """
    if column == COLUMN_AMD:
        return 64
    return 32


def spec_for(kernel: Int, device: Int, mode: NumericMode) raises -> KernelSpec:
    """The resolved knobs for one kernel, substituting column by column.

    This is the "sub in and out" step, and it is per ROW rather than per
    spec: a SCHEDULING row is taken from the `device` column always, and a
    NUMERIC row is taken from `COLUMN_BIT_IDENTICAL` under the default and
    from the `device` column only when the caller opted into `FAST`.

    So a fit on Apple runs Apple's block sizes and grid shapes at full speed
    while its reduction geometry is the safe column's, and the same fit on
    NVIDIA does the same with NVIDIA's scheduling. Both produce one model.
    """
    var identical = mode.mode == NUMERIC_IDENTICAL
    #: The column every NUMERIC row reads from.
    var numeric_column = COLUMN_BIT_IDENTICAL if identical else device

    var floats_per_thread = 16
    var per_int = 8
    if kernel == K_HIST_BINARY:
        per_int = 32
    elif kernel == K_HIST_ONE_BYTE:
        floats_per_thread = 32
        per_int = 4
    elif kernel == K_HIST_HALF_BYTE:
        per_int = 8
    else:
        # The non-histogram kernels claim no replicated scratch.
        floats_per_thread = 0
        per_int = 0

    # SCHEDULING: the largest block whose scratch fits this vendor, capped at
    # CatBoost's own choice because more threads than they use buys nothing
    # and costs a wider barrier.
    var catboost_block = 384 if kernel == K_HIST_ONE_BYTE else 768
    var block = catboost_block
    if floats_per_thread > 0:
        # SCHEDULING for the block count, but the LIMIT that bounds it is
        # numeric, because the scratch budget decides the replication factor.
        # So the budget comes from `numeric_column` and the block size chosen
        # under it comes from the device.
        var limit = column_shared_limit(numeric_column) // (
            floats_per_thread * 4
        )
        if limit < block:
            block = limit
    elif kernel == K_SCORES:
        block = 128  # compute_scores.cu:167
    elif kernel == K_SPLIT_POINTS:
        block = 256  # compute_scores.cu:493
    else:
        block = 512

    if block < 32:
        raise Error(
            "kernel "
            + String(kernel)
            + " cannot fit a block in column "
            + column_name(numeric_column)
            + ": "
            + String(floats_per_thread)
            + " floats per thread leaves room for "
            + String(block)
            + " threads, and the replication geometry needs at least one"
            " full lane group"
        )

    # The flush row, and the one place a VENDOR overrides a MODE. Asking for
    # FAST on Apple cannot produce a float atomic, so the row is forced and
    # the fact that it was forced is recorded beside it.
    var vendor_forces_flush = not column_has_float_atomics(device)
    var flush = identical or vendor_forces_flush

    return KernelSpec(
        block,
        floats_per_thread,
        per_int,
        PINNED_REPLICATION_LANES if identical else column_lane_width(device),
        PINNED_REDUCE_WIDTH if identical else block,
        flush,
        vendor_forces_flush,
    )


# ---------------------------------------------------------------------------
# COMPTIME ACCESSORS, so a kernel READS this table instead of restating it.
#
# `spec_for` above resolves a whole spec at runtime, which is right for a
# report and useless to a kernel: a threadgroup allocation size must be known
# at compile time. These are the same rows as pure comptime functions so that
#
#     comptime BLOCK_SIZE = block_size_for[K_HIST_BINARY, TARGET_COLUMN]()
#
# is legal at the top of a kernel file. Without them the table is decoration:
# every kernel restates its own constants and the matrix silently describes a
# build nobody runs. That failure has a name in this project's history and it
# is the reason these exist.
# ---------------------------------------------------------------------------

#: The column kernels compile against. One value for the whole build, because
#: a threadgroup size is fixed at compile time and cannot follow a runtime
#: device query. Cross-compiling for another vendor means rebuilding with
#: this changed, which is the honest shape of the constraint.
comptime TARGET_COLUMN = COLUMN_APPLE


def hist_floats_per_thread_for[kernel: Int]() -> Int:
    """Shared floats per thread. `GetHistSize()` is this times the block."""
    if kernel == K_HIST_ONE_BYTE:
        return 32
    return 16


def catboost_block_for[kernel: Int]() -> Int:
    """What CatBoost uses, before our shared-memory budget bites."""
    if kernel == K_HIST_ONE_BYTE:
        return 384
    return 768


def block_size_for[kernel: Int, column: Int]() -> Int:
    """SCHEDULING row, bounded by a NUMERIC one.

    The block itself is scheduling: it decides which thread does which work.
    The BUDGET that bounds it is numeric, because the shared-memory ceiling
    decides the replication factor and the replication factor decides how
    many partial sums combine. So the ceiling comes from the column and the
    block is the largest that fits under it, capped at CatBoost's own choice
    because more threads than they use buys nothing and costs a wider
    barrier.
    """
    comptime floats = hist_floats_per_thread_for[kernel]()
    comptime cap = catboost_block_for[kernel]()
    comptime limit = column_shared_limit(column) // (floats * 4)
    return limit if limit < cap else cap


def lane_width_for[column: Int, identical: Bool]() -> Int:
    """NUMERIC. Pinned to 32 under identity so AMD's 64-wide wavefront cannot
    change the reduction geometry. See `column_lane_width`."""
    if identical:
        return PINNED_REPLICATION_LANES
    return column_lane_width(column)


def reduce_width_for[kernel: Int, column: Int, identical: Bool]() -> Int:
    """NUMERIC. `Reduce()`'s stage width. CatBoost writes a literal 512 that
    is only correct at its own block size; pinned here so the reduction tree
    has one shape, and never above the block or stage 1 leaves slots unwritten.
    """
    comptime block = block_size_for[kernel, column]()
    comptime pinned = PINNED_REDUCE_WIDTH if identical else block
    return block if block < pinned else pinned


# ---------------------------------------------------------------------------
# SYNC GRANULARITY, and the correctness bug it exists to prevent.
# ---------------------------------------------------------------------------

#: Only threadgroup-wide `barrier()` is available.
comptime SYNC_BLOCK = 0

#: A lane group can sync independently (CUDA `tiled_partition`, AMD wave ops).
#: NOT REACHABLE FROM MOJO TODAY on any vendor.
comptime SYNC_LANE = 1


def sync_granularity_for[column: Int]() -> Int:
    """The finest sync a kernel may rely on. `SYNC_BLOCK` for EVERY column.

    **This is a Mojo limitation and not an Apple one, which is worth stating
    because the natural assumption is the opposite.** `max.gpu.primitives`
    exposes `block` only: `block.prefix_sum`, `block.max`, `block.min`. There
    is no warp, simdgroup or shuffle primitive for any vendor. NVIDIA and AMD
    hardware both have them and CatBoost uses them heavily
    (`tiled_partition<8>` in the half-byte accumulator, `tiled_partition<32>`
    in the one-byte one, `cub::WarpScan` in the bin scan); we cannot emit any
    of it from here.

    So the row is `SYNC_BLOCK` across the board today. When Mojo exposes lane
    primitives this becomes a per-column value, the kernels follow it, and
    nothing else in the tree changes.

    SCHEDULING row: it decides which threads wait for which, never what is
    added to what.
    """
    return SYNC_BLOCK


def requires_uniform_iteration_for[column: Int]() -> Bool:
    """Whether every thread of a block must run the SAME iteration count.

    **A correctness requirement, not a tuning knob, and it cost a wrong
    answer to learn.** CatBoost syncs a `tiled_partition<8>` inside the
    histogram inner loop, which is lane-local, so warps with different
    iteration counts never wait on each other. Widened to a threadgroup
    `barrier()`, a warp that finishes early walks past a barrier its
    neighbours are still waiting on, and a divergent threadgroup barrier is
    undefined behavior.

    It is not a rare edge case. A 64-row partition over a 512-thread block
    gives warp 0 one iteration and warps 1 through 15 zero.

    So under `SYNC_BLOCK` the kernels derive one iteration count for the
    whole block and let threads with no rows contribute a 0.0 stat, which
    keeps every lane inside every barrier. Adding 0.0 changes no sum, so this
    is a scheduling change and the histogram is unaffected.
    """
    return sync_granularity_for[column]() == SYNC_BLOCK
