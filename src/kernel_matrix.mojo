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

from .numerics import NumericMode, NUMERIC_FAST, NUMERIC_IDENTICAL


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
    """NUMERIC. Fixed-point integer flush instead of float atomicAdd."""

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

    return KernelSpec(
        block,
        floats_per_thread,
        per_int,
        PINNED_REPLICATION_LANES if identical else column_lane_width(device),
        PINNED_REDUCE_WIDTH if identical else block,
        identical,
    )
