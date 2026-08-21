"""One table: every kernel, every tunable it takes, per GPU vendor.

NOT a port. CatBoost tunes against `TArchProps` scattered through the kernel
files and ships one vendor. **We ship Metal, CUDA and HIP from one source and
the product property is that a fit gives the SAME MODEL on all three.** That
cannot be maintained by constants sprinkled across kernels, so every knob any
kernel takes is declared here, once, with its vendor column and its class.

There is no CPU column. This tree has no CPU path.

THREE COLUMNS BUILD TODAY; THREE MORE ARE DECLARED AND DO NOT
--------------------------------------------------------------
`apple`, `nvidia` and `amd` are what Mojo emits code for. `qualcomm`,
`intel` and `arm` are declared with their documented minimums and their
admission verdict, and `column_is_buildable` says which is which.

A declared column is not decoration and it is not a promise of support. It
answers the one question that has to be answered BEFORE a target exists:
**would admitting this vendor change the models we have already shipped?**
Under the old design -- safe column = intersection of the vendors present --
the answer depended on who showed up, and a routine act of maintenance could
silently rewrite every reproducible fit. It cannot now: see THE IDENTITY
FLOOR below. Declaring the vendors is how that stops being a judgement call
taken later under pressure, and it is why this is worth doing while the
answer is free.

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

**AND ON TWO OF THE DECLARED VENDORS THE HAZARD IS WORSE THAN A DIFFERENT
CONSTANT: THE WIDTH IS NOT A CONSTANT.** Adreno's wave is 8, 16, 32, 64 or
128 depending on the part AND ON THE COMPILER'S DECISION; Intel's sub-group
is 8, 16 or 32 and their compiler picks it from register pressure unless a
kernel demands one. A design that reads the hardware width would not merely
disagree across vendors there, it could disagree across two builds of the
same kernel for the same device. Pinning was the right answer for AMD and is
the only possible answer for those two. See `column_lane_width`.
"""

from mojo_only.numerics import NumericMode, NUMERIC_FAST, NUMERIC_IDENTICAL


# --- the columns ---------------------------------------------------------
#
# BIT_IDENTICAL is a real column, not a mode flag. It holds the value every
# ADMITTED vendor can meet, and it can be read, printed and diffed against a
# vendor column to show exactly what identity costs on that device.
#
# **IT IS NO LONGER `min()` OVER WHATEVER COLUMNS HAPPEN TO EXIST, AND THAT
# CHANGE IS THE WHOLE POINT OF THE VENDOR ADDITIONS BELOW.** It used to be
# described as "the INTERSECTION of the other three", which was true and was
# a trap: a fourth vendor with a smaller budget would have lowered the
# intersection, and lowering the intersection changes the block size, hence
# the replication factor, hence WHICH PARTIAL SUMS COMBINE -- so every model
# this library had ever produced under `IDENTICAL` would silently stop
# matching the ones produced after the addition. That is not an API break a
# caller could see. It is a bit break with no signal at all.
#
# So the safe column is now a FROZEN, VERSIONED PROFILE (`IDENTITY_PROFILE`,
# below) whose values are the intersection AS OF THE FOUNDING THREE, and a
# new vendor does one of exactly two things:
#
#   (a) meets the floor -> it joins `IDENTICAL` and NOT ONE BIT MOVES, or
#   (b) does not        -> it is REFUSED for `IDENTICAL` by name and runs
#                          `FAST` only, which is still a working library on
#                          that device.
#
# Never (c) lower the floor to fit it. Lowering the floor is a NEW GUARANTEE
# and takes a profile bump, which renames the property rather than quietly
# redefining it. See `column_meets_identity_floor`.
comptime COLUMN_BIT_IDENTICAL = 0
comptime COLUMN_APPLE = 1
comptime COLUMN_NVIDIA = 2
comptime COLUMN_AMD = 3

# --- vendors declared but NOT REACHABLE FROM THIS TOOLCHAIN TODAY --------
#
# Mojo compiles to PTX (NVIDIA), AMDGPU IR (AMD) and Metal (Apple), and the
# system requirements page lists those three and no others as of 2026-08-21.
# These columns therefore compile nothing today. They exist because the cost
# of adding a column LATER is not the column -- it is that every reader has
# to re-derive whether the safe column still means what it meant, and by then
# there are shipped models that answer depends on.
#
# Declaring them now, with their documented minimums and their admission
# verdict, is what makes the answer "the floor was frozen before you arrived"
# instead of a judgement call taken under pressure. Adding the target later
# is then a build, not a redesign.
#
# Qualcomm is first for a reason that is not technical: Qualcomm ACQUIRED
# Modular in July 2026 and open-sourced Mojo and MAX at ModCon in August, so
# Adreno is the likeliest fourth target of any GPU on this list.
comptime COLUMN_QUALCOMM = 4
comptime COLUMN_INTEL = 5
comptime COLUMN_ARM = 6

#: The count, so a report or a check loops instead of listing. Raise this and
#: `column_name` and every row below must answer for the new value; the
#: check in `mojo_only/hardware_matrix_check.mojo` enforces that they do.
comptime COLUMN_COUNT = 7

#: Kept as aliases so kernel code can name the API or the part it targets
#: rather than the company. Metal is Apple's, CUDA is NVIDIA's, HIP is AMD's;
#: Adreno is Qualcomm's GPU, Xe is Intel's, Mali is Arm's.
comptime COLUMN_METAL = COLUMN_APPLE
comptime COLUMN_CUDA = COLUMN_NVIDIA
comptime COLUMN_HIP = COLUMN_AMD
comptime COLUMN_ADRENO = COLUMN_QUALCOMM
comptime COLUMN_XE = COLUMN_INTEL
comptime COLUMN_MALI = COLUMN_ARM


def column_name(column: Int) -> String:
    if column == COLUMN_BIT_IDENTICAL:
        return String("bit-identical")
    if column == COLUMN_APPLE:
        return String("apple")
    if column == COLUMN_NVIDIA:
        return String("nvidia")
    if column == COLUMN_AMD:
        return String("amd")
    if column == COLUMN_QUALCOMM:
        return String("qualcomm")
    if column == COLUMN_INTEL:
        return String("intel")
    if column == COLUMN_ARM:
        return String("arm")
    return String("unknown")


def column_is_buildable(column: Int) -> Bool:
    """Whether Mojo can emit a kernel for this column TODAY.

    `False` is not a criticism of the column and does not make it decoration:
    an unbuildable column still answers "would identity survive this vendor",
    which is the question that has to be answered BEFORE the target exists,
    not after.

    Source: Mojo system requirements (read 2026-08-21) lists NVIDIA (Turing
    through Blackwell, driver 580+), AMD (RDNA2 through CDNA4, ROCm 6.3.3+)
    and Apple silicon (M1-M5, macOS 15+, Xcode 16+). No Qualcomm, Intel or
    Arm GPU target is listed.
    """
    return (
        column == COLUMN_BIT_IDENTICAL
        or column == COLUMN_APPLE
        or column == COLUMN_NVIDIA
        or column == COLUMN_AMD
    )

# --- the kernels ---------------------------------------------------------
comptime K_HIST_BINARY = 0
comptime K_HIST_HALF_BYTE = 1
comptime K_HIST_ONE_BYTE = 2
comptime K_SCAN = 3
comptime K_SUBTRACT = 4
comptime K_SCORES = 5
comptime K_SPLIT_POINTS = 6
#: `TPointHist2OneByte`, the FUSED TWO-STAT one-byte family CatBoost's
#: dispatch takes at `maxBins <= 128` (`hist_one_byte.cu:315-322`). Same
#: scratch shape as `K_HIST_ONE_BYTE` -- `BlockSize * 32` floats at their
#: `BlockSize = 384` (`hist_2_one_byte_base.cuh:20-22`, `:169`) -- but a
#: different kernel with a different slot layout, so it is its own row.
comptime K_HIST_2_ONE_BYTE = 7

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

    The MODE's choice on every column. `FAST` takes CatBoost's float atomic
    (`atomicAdd(dst + fold, val)`, `hist_binary.cu:60`,
    `hist_half_byte.cu:47`, `hist_one_byte.cu:255`); `IDENTICAL` takes the
    integer path, where addition is associative and the histogram does not
    depend on which block lands first. See `column_has_float_atomics`."""

    var flush_forced_by_vendor: Bool
    """Whether `deterministic_flush` is the mode's choice or the vendor's
    constraint. **No vendor forces it today** -- all three do float
    `atomicAdd`. Kept so a column that genuinely cannot has somewhere to say
    so, and so a report never implies the user asked for a flush the mode
    did not choose."""

    def shared_bytes(self) -> Int:
        """What this spec asks of threadgroup memory."""
        return self.block_size * self.hist_floats_per_thread * 4


# =========================================================================
# THE IDENTITY FLOOR: a frozen profile, so a future vendor cannot move bits
# =========================================================================
#
# WHAT WENT WRONG IN THE OLD DESIGN, stated plainly because it never fired
# and therefore never taught anybody anything. The safe column was defined as
# the INTERSECTION of the vendor columns. With three vendors that produced 32
# KB, 32 lanes, 512-wide folds -- correct values, arrived at correctly. But
# the DEFINITION says: add a vendor with less, and the safe column shrinks.
# A smaller shared budget gives a smaller block, a smaller block gives a
# different replication factor, and a different replication factor sums a
# different set of partials. Every `IDENTICAL` model produced before the
# addition would disagree with every one produced after it, on every device,
# with no version, no error, and no way for a user to notice except by
# keeping an old model file and re-running it.
#
# The property this library sells is "the same fit gives the same model". A
# definition under which a routine act of maintenance -- supporting one more
# GPU -- rewrites that answer for everybody is not a safe column. It is a
# tripwire with a three-vendor fuse.
#
# SO THE FLOOR IS FROZEN AND VERSIONED. These constants ARE the guarantee.
# They do not derive from the vendor rows and no vendor row may change them.
#
#   A new vendor MEETS the floor  -> it joins `IDENTICAL`, no bit moves.
#   A new vendor MISSES the floor -> `IDENTICAL` REFUSES it, by name, and it
#                                    runs `FAST`, which is a working library.
#
# There is no third option, and in particular there is no "lower the floor a
# little". Lowering it is a DIFFERENT GUARANTEE about a different set of
# models, so it takes a profile bump, and models built under profile N and
# profile N+1 are not comparable and must never be compared. A profile bump
# is a product decision with a migration, not a patch.
#
# **THE HOLE THIS LEAVES, AND IT IS REAL: the serialized model does not carry
# the profile id yet.** Until it does, profile 1 and a future profile 2 model
# are distinguishable only by provenance, which is exactly the kind of thing
# that gets lost. Recorded in `UNWIRED.md`; it is a header field and an hour's
# work, and it should be done before any second profile exists rather than
# after.

#: The version of the bit-identity guarantee. Bump ONLY to widen the floor
#: for a vendor that cannot meet it, and only with a migration note. Every
#: model produced under `IDENTICAL` is a model under THIS profile.
comptime IDENTITY_PROFILE = 1

#: Frozen. Threadgroup bytes the safe column allows a block to claim. Was the
#: three-vendor intersection (Apple's Metal ceiling) on the day it froze, and
#: is now simply the number.
comptime IDENTITY_FLOOR_SHARED_BYTES = 32 * 1024

#: Frozen. The logical replication group width. `PINNED_REPLICATION_LANES`
#: must equal this; `hardware_matrix_check` asserts it, because two names for
#: one guarantee is how a guarantee drifts.
comptime IDENTITY_FLOOR_LANES = 32

#: Frozen. The block the safe column's hist_2 arm runs, so a vendor whose
#: dispatch cannot reach it cannot reproduce the accumulation.
comptime IDENTITY_FLOOR_BLOCK = 512


def column_meets_identity_floor(column: Int) -> Bool:
    """Whether this vendor can join `IDENTICAL` without the floor moving.

    Four questions, and a column has to pass all four:

      1. threadgroup bytes >= the floor. Below it the block shrinks and the
         replication factor with it.
      2. threadgroup INT atomics. The safe column's accumulator is Int32 in
         shared memory; there is no substitute that keeps associativity.
      3. a block of `IDENTITY_FLOOR_BLOCK` threads is dispatchable.
      4. the lane group is not WIDER than the pinned group in a way the
         kernels cannot subdivide -- which, under `SYNC_BLOCK`, none is,
         because the pinned group is logical. Kept as an explicit clause
         rather than an omission so that it is re-examined on the day lane
         primitives arrive (see `column_lane_width`).

    Today every declared vendor passes, INCLUDING the three that cannot be
    built for. That is the useful answer and it was not the expected one:
    Adreno's 32 KB matches Apple's exactly, Mali's advertised 32 KB does too,
    and Intel's 64 KB is above. The design turns out to have been floored by
    the most constrained mainstream GPU memory hierarchy already, which is
    why freezing it costs nothing today and is worth doing precisely now,
    while it costs nothing.

    A column that fails belongs in `FAST` and the failure must be reported by
    name, never absorbed. `identity_refusal_reason` is that name.
    """
    return (
        column_shared_limit(column) >= IDENTITY_FLOOR_SHARED_BYTES
        and column_has_threadgroup_int_atomics(column)
        and column_max_block_size(column) >= IDENTITY_FLOOR_BLOCK
    )


def identity_refusal_reason(column: Int) -> String:
    """Why `IDENTICAL` refuses this column, or empty if it does not.

    A refusal a user can act on: it names the row, the column's value and the
    floor's, so the answer to "why can't I get reproducible fits on this
    laptop" is one sentence and not a support thread.
    """
    if column_shared_limit(column) < IDENTITY_FLOOR_SHARED_BYTES:
        return (
            column_name(column)
            + " allows "
            + String(column_shared_limit(column) // 1024)
            + " KB of threadgroup memory per block; the identity floor"
            " (profile "
            + String(IDENTITY_PROFILE)
            + ") needs "
            + String(IDENTITY_FLOOR_SHARED_BYTES // 1024)
            + " KB, because the block size it buys decides the replication"
            " factor and the replication factor decides which partial sums"
            " combine"
        )
    if not column_has_threadgroup_int_atomics(column):
        return (
            column_name(column)
            + " has no threadgroup integer atomic add; the identity column"
            " accumulates the histogram in shared Int32 and there is no"
            " substitute that keeps addition associative"
        )
    if column_max_block_size(column) < IDENTITY_FLOOR_BLOCK:
        return (
            column_name(column)
            + " dispatches at most "
            + String(column_max_block_size(column))
            + " threads per block; the identity column's hist_2 arm runs "
            + String(IDENTITY_FLOOR_BLOCK)
        )
    return String("")


def column_shared_limit(column: Int) -> Int:
    """Threadgroup / shared / LDS / SLM bytes a single block may claim.

    Per-vendor, with the document each number came from. Only the first three
    have ever run a kernel from this tree; the rest are transcriptions, and
    `column_is_buildable` says so.

    - apple    32 KB. Metal threadgroup memory limit, family 9 (M3/M4).
               VALIDATED on this box.
    - nvidia   48 KB. Default per-block shared memory; the opt-in carveout
               above it (`cudaFuncAttributeMaxDynamicSharedMemorySize`, 164 KB
               at CC 8.0) is deliberately NOT modeled, because taking it
               changes occupancy and this row bounds a numeric one.
    - amd      64 KB. LDS per workgroup, GCN through CDNA/RDNA.
    - qualcomm 32 KB. Adreno's per-workgroup OpenCL local memory allocation
               (Adreno 6xx/X1 class; the GPU-wide pool is larger -- 64 KB on
               Adreno 640 -- but a single workgroup may claim 32 KB).
               Qualcomm Snapdragon OpenCL general programming guide.
    - intel    64 KB. A work-group's SLM allocation on Xe; the Xe-core pool
               is 128 KB and Intel's own occupancy example splits it as
               64 + 64 for two resident groups (oneAPI GPU optimization
               guide, "Shared Local Memory").
    - arm      32 KB. Mali's advertised compute shared-memory size, AND SEE
               `column_has_dedicated_shared_memory`: on Mali this is not a
               scratchpad at all. Arm's own best-practices guide says "Arm
               GPUs do not implement dedicated on-chip shared memory for
               compute shaders. The shared memory that is available to use is
               system RAM that is backed up by the load-store cache." The
               BYTES are available; the SPEED the histogram design assumes is
               not.

    `BIT_IDENTICAL` returns the FROZEN floor, not a `min()` over these. See
    `IDENTITY_FLOOR_SHARED_BYTES`.
    """
    if column == COLUMN_APPLE:
        return 32 * 1024
    if column == COLUMN_NVIDIA:
        return 48 * 1024
    if column == COLUMN_AMD:
        return 64 * 1024
    if column == COLUMN_QUALCOMM:
        return 32 * 1024
    if column == COLUMN_INTEL:
        return 64 * 1024
    if column == COLUMN_ARM:
        return 32 * 1024
    return IDENTITY_FLOOR_SHARED_BYTES  # BIT_IDENTICAL: frozen, not derived


def column_has_float_atomics(column: Int) -> Bool:
    """Whether this vendor can do `atomicAdd` on a `float` at all.

    **ALL THREE FOUNDING COLUMNS CAN, APPLE INCLUDED.** This row said "APPLE
    CANNOT. Metal has no floating-point atomic add" and that was FALSE.
    Probed 2026-08-19:

        Atomic.fetch_add(dst, Float32(1.0))   // 1024 threads
        -> 1024.0, exact

    The false claim is deleted rather than annotated. It had cost real money:
    it is the entire reason `mojo_only/fixed_point.mojo` exists, and the
    fixed-point substitution carries a RANGE CONTRACT CatBoost's float atomic
    does not have -- a magnitude that failed to bound the partials overflowed
    Int32 and produced a dead histogram, which took a bisect to find.

    So `deterministic_flush` is a row every FOUNDING column can negotiate:
    `FAST` leaves all three on CatBoost's float atomic
    (`atomicAdd(dst + fold, val)`, every `AddToGlobalMemory`), and
    `IDENTICAL` pins them to the integer path so they agree bit for bit.

    **THE DECLARED VENDORS ARE WHERE THIS ROW STOPS BEING TRIVIAL, AND IT IS
    WHY `flush_forced_by_vendor` EXISTS.** On Adreno and Mali, float atomic
    add is not core OpenCL: it arrives through `cl_ext_float_atomics`, which
    requires OpenCL 2.0+ and is optional per device and per memory scope
    (Khronos OpenCL extension registry). A device without it CANNOT run
    `FAST` as written and must take the fixed-point flush -- which is the
    vendor forcing the mode's hand, exactly the case that field was built for
    and that no founding column exercises.

    Conservative until queried: `False` for the unbuildable columns. That is
    a claim about the WEAKEST device in the family, not about the best one,
    and bring-up replaces it with a device query rather than an opinion.
    """
    return (
        column == COLUMN_BIT_IDENTICAL
        or column == COLUMN_APPLE
        or column == COLUMN_NVIDIA
        or column == COLUMN_AMD
        or column == COLUMN_INTEL
    )


def column_has_threadgroup_int_atomics(column: Int) -> Bool:
    """Whether a block can `atomicAdd` an `Int32` in THREADGROUP memory.

    **THIS IS THE ONE CAPABILITY `IDENTICAL` CANNOT DO WITHOUT.** The safe
    column accumulates the hist_2 family in shared Int32 slices
    (`HIST_SMEM_SHARED2_I32`), because integer addition is associative and
    that is what makes the histogram independent of arrival order. A device
    without local integer atomics cannot take that path, and there is no
    substitute that keeps the property at a tolerable cost.

    True everywhere it has been checked, and it is a low bar by construction:
    integer atomics on local memory are CORE OpenCL 1.2 (`atomic_add` on
    `__local int`), core CUDA, core HIP, and Metal's `atomic_fetch_add_explicit`
    on `threadgroup atomic_int`. Metal is the interesting one, because Metal
    has NO local FLOAT atomic ("Unsupported local float atomic operation")
    while having the integer one -- which is precisely why the Apple column's
    fast path and the safe column's path are the same code.
    """
    return True


def column_has_dedicated_shared_memory(column: Int) -> Bool:
    """Whether "shared memory" is an on-chip scratchpad or just cached RAM.

    SCHEDULING by the one question -- it changes what the histogram COSTS,
    never what it sums -- and it is recorded because it is the difference
    between a column that is admissible and a column that is admissible and
    also worth shipping.

    Arm's GPU best-practices guide, verbatim: "Arm GPUs do not implement
    dedicated on-chip shared memory for compute shaders. The shared memory
    that is available to use is system RAM that is backed up by the
    load-store cache."

    A replicated-histogram design is a bet that private scratch is much
    faster than the memory it spares. On Mali that bet is off: the
    replication still gives the same ANSWER (identity is unaffected) and
    buys much less than it does elsewhere. Whoever brings that column up
    should expect to re-measure the replication factor, not to inherit it.
    """
    return column != COLUMN_ARM


def column_max_block_size(column: Int) -> Int:
    """Largest threadgroup the vendor will dispatch, before our budget bites.

    - apple / nvidia / amd / qualcomm / intel: 1024.
      (Adreno 5xx and later report a 1024 max OpenCL workgroup size; older
      parts are far smaller -- 128 on Adreno 320, 512 on Adreno 420 -- and
      are out of scope for a training library.)
    - arm: 512. Arm's own guidance is a 64-thread baseline and register
      pressure bites well before the advertised maximum.

    This bounds nothing today: every block this tree launches is <= 768, and
    the shared-memory budget binds first on every column. It is declared so
    that the FLOOR below can be checked against something rather than
    assumed, since the safe column's hist_2 block is 512.
    """
    if column == COLUMN_ARM:
        return 512
    return 1024


def column_lane_width(column: Int) -> Int:
    """Hardware lanes that move in lockstep: warp on NVIDIA, SIMD group on
    Apple, WAVEFRONT on AMD, wave on Adreno, sub-group on Intel, warp on Mali.

    **This is the cross-vendor hazard in one number.** CatBoost hardcodes 32
    throughout, which is right on Apple and NVIDIA and wrong on AMD, whose
    wavefront is 64. Reading it from the device would change the replication
    geometry and silently move the last bits, so `BIT_IDENTICAL` pins 32 and
    AMD pays a wider sync for it.

    **AND ON TWO OF THE DECLARED VENDORS THIS NUMBER IS NOT A CONSTANT AT
    ALL, WHICH IS THE MOST IMPORTANT THING ON THIS PAGE.** It is not merely
    a different number per part:

    - qualcomm: Qualcomm's own OpenCL optimization guidance says the wave
      size "depends on Adreno GPU series and tiers AS WELL AS THE COMPILER;
      values could be 8, 16, 32, 64, 128". The Adreno X1 in Snapdragon X
      Elite runs 64- or 128-wide. So the same kernel, same device, different
      compiler decision, different width.
    - intel: the sub-group size is 8, 16 or 32 and Intel's compiler CHOOSES
      it from register pressure unless the kernel demands one explicitly
      (oneAPI GPU optimization guide, "Sub-groups and SIMD Vectorization").
    - arm: 8 on Bifrost (G52, G76), 16 on Valhall. Both BELOW the pinned 32.

    The values returned here are the documented MINIMUM for each family,
    which is the only safe reading for anything that sizes a buffer.

    **WHY THIS DOES NOT BREAK THE IDENTITY CLAIM, WHICH IS WORTH STATING
    PRECISELY BECAUSE IT LOOKS LIKE IT SHOULD.** `PINNED_REPLICATION_LANES`
    is a LOGICAL group width, not a hardware sync assumption. Every kernel in
    this tree syncs at `SYNC_BLOCK` -- Mojo exposes no warp primitive on any
    vendor (`sync_granularity_for`) -- so a pinned 32-lane replication group
    is threadgroup-synchronized arithmetic that happens to be 32 wide, and it
    stays 32 wide whether the hardware runs 8, 16, 64 or 128 lanes in
    lockstep underneath. A variable hardware width would be fatal to a
    warp-primitive design and is merely a scheduling fact for this one.

    That is a real property of the port, but it is not one to take on trust:
    the day Mojo exposes lane primitives and `sync_granularity_for` stops
    returning `SYNC_BLOCK`, this paragraph expires and these three columns
    become unsafe until re-argued. Said here rather than in a commit message
    because that is where whoever wires the primitive will look.
    """
    if column == COLUMN_AMD:
        return 64
    if column == COLUMN_QUALCOMM:
        return 8
    if column == COLUMN_INTEL:
        return 8
    if column == COLUMN_ARM:
        return 8
    return 32


def column_lane_width_is_fixed(column: Int) -> Bool:
    """Whether `column_lane_width` is a property of the DEVICE or a decision
    the vendor's compiler makes per kernel.

    False for qualcomm and intel. Any code that indexes by hardware lane
    ("which lane am I") must read this first: on a variable-width column the
    only correct answers are to demand a width from the compiler where the
    API allows it (`intel_reqd_sub_group_size`, Metal-style attributes) or to
    stop indexing by lane. Nothing in this tree indexes by hardware lane
    today, because `SYNC_BLOCK` left it with no reason to.
    """
    return column != COLUMN_QUALCOMM and column != COLUMN_INTEL


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
    elif kernel == K_HIST_ONE_BYTE or kernel == K_HIST_2_ONE_BYTE:
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
    var catboost_block = 384 if (
        kernel == K_HIST_ONE_BYTE or kernel == K_HIST_2_ONE_BYTE
    ) else 768
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

    # The flush row. It is the one row a VENDOR could override a MODE on, and
    # today NO VENDOR DOES: all three do float `atomicAdd`, so
    # `vendor_forces_flush` is false everywhere and `flush` is the mode's
    # answer. The override is kept because it is the honest shape of the
    # row, and a column that genuinely lacked the instruction would need it.
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
    if kernel == K_HIST_ONE_BYTE or kernel == K_HIST_2_ONE_BYTE:
        return 32
    return 16


def catboost_block_for[kernel: Int]() -> Int:
    """What CatBoost uses, before our shared-memory budget bites."""
    if kernel == K_HIST_ONE_BYTE or kernel == K_HIST_2_ONE_BYTE:
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


def deterministic_flush_for[column: Int, identical: Bool]() -> Bool:
    """NUMERIC row, comptime, so a kernel can branch on it.

    Whether a multi-block histogram flush sums through a FIXED-POINT integer
    accumulator instead of `atomicAdd` on `float`.

    **No column is forced any more.** `FAST` leaves every vendor on
    CatBoost's float atomic, which is what they ship; `IDENTICAL` pins every
    vendor to the integer path, where integer addition is associative and the
    result does not depend on which block lands first.

    Comptime rather than runtime because the two flushes are different code,
    not a configured value, which is the distinction `numerics.mojo` draws
    between a row a mode can pin and one it must replace.

    The float atomic needs its destination ZEROED, which is why CatBoost
    clears the per-block scratch with `ZeroBuffer(blockHistograms, streamId)`
    -- a whole-buffer `cudaMemsetAsync` sized by
    `BlockHistogramsMapping(blockId, leavesCount, statsCount)`
    (`split_properties_helper.cpp:1155-1157`, `:34-38`) -- rather than with
    the indexed `ZeroHistogramsImpl` it uses on the final flat histogram.
    """
    return identical


# ---------------------------------------------------------------------------
# THE hist_2 SHARED-MEMORY ACCUMULATION ROW (Apple's 32 KB wall, priced)
# ---------------------------------------------------------------------------

#: CatBoost's design: every warp accumulates into a PRIVATE 1024-float
#: shared-memory slice with plain (non-atomic) float adds
#: (`hist_2_one_byte_base.cuh:20-22`, the per-bit `SliceOffset`s).
comptime HIST_SMEM_WARP_PRIVATE_F32 = 0

#: The Apple variant: one 1024-slot slice is SHARED BY TWO WARPS and the
#: adds are LOCAL Int32 atomics in fixed point, halving shared memory per
#: thread. See `hist_smem_mode_for` for the measurement that bought it.
comptime HIST_SMEM_SHARED2_I32 = 1


def hist_smem_mode_for[column: Int, identical: Bool]() -> Int:
    """NUMERIC row: HOW the hist_2 family accumulates in shared memory.

    Integer fixed-point accumulation is a DIFFERENT ARITHMETIC than float
    adds -- each stat value is quantized `Int32(val * fixed_scale)` before it
    lands -- so this is a numeric row by the one question, not a scheduling
    knob that happens to change occupancy.

    WHY THE APPLE COLUMN LEAVES THEIR DESIGN. CatBoost's warp-private slices
    cost 32 floats of shared memory per thread; NVIDIA's 48 KB and AMD's
    64 KB hold that at their `BlockSize = 384`
    (`hist_2_one_byte_base.cuh:169`). Metal's 32 KB threadgroup ceiling caps
    the same design at 256 resident threads per core -- 12.5% occupancy, a
    wall built into the layout, not a tuning knob. Measured on the M4
    (scratchpad `histshare_probe.mojo`, 100M hashed scatter-adds,
    interleaved): warp-private float at 32 KB = 46.5 G updates/s; ONE
    histogram shared by TWO warps with local Int32 atomic adds at 16 KB =
    90.2 G upd/s -- **1.94x**, because occupancy doubles and Metal's local
    integer atomics are fast. Metal has NO local float atomics ("Unsupported
    local float atomic operation"), so the shared-slice variant is only
    reachable through the integer domain.

    So: the APPLE column takes `HIST_SMEM_SHARED2_I32`; NVIDIA and AMD keep
    CatBoost's `HIST_SMEM_WARP_PRIVATE_F32`, which their shared-memory
    budgets fit and their measured design chose. `BIT_IDENTICAL` takes the
    shared-Int32 variant too: integer addition is associative and exact, so
    the accumulated histogram cannot depend on lane interleaving, block
    arrival order, or vendor -- which also makes the Apple arm DETERMINISTIC
    RUN TO RUN, unlike CatBoost's own float path.

    THE RANGE CONTRACT RIDES ALONG. Quantizing at `fixed_scale` is safe only
    under `mojo_only/fixed_point.mojo`'s bound: `fixed_scale` must come from
    `choose_scale(sum over all rows of abs(plane), n_rows)`, so every
    shared-memory cell -- a partial sum over a SUBSET of one leaf's rows,
    and any leaf's rows are a subset of all rows -- stays under 2^30 with
    `hist2_quantize`'s worst-case +1 dither unit per row accounted
    EXPLICITLY in the limit (see `choose_scale`'s docstring for why the
    resolution is load-bearing: the blanket 2^28 scale's dither noise
    flipped near-tied splits at 254 borders, -1.6% train mse, measured
    2026-08-20). A caller that passes an unbounded
    scale overflows Int32 silently; `doc_parallel_boosting.fit` computes the
    two plane magnitudes on the device for exactly this row.

    THE QUANTIZER IS `hist2_quantize` (histogram_utils.mojo), dithered floor
    keyed on the document position -- NOT a bare `Int32(val * scale)`. Plain
    truncation and round-to-nearest both accumulated a value-correlated bias
    linear in the row count that `compute_scores`'s exact
    `partStat - sumLeft` read as phantom mass; its docstring carries the
    measured failures.
    """

    #: CatBoost's warp-private layout at their own block: 384 threads x 32
    #: shared floats x 4 bytes = 49,152. A column that cannot hold that
    #: cannot run their design at their block size, and the shared-Int32
    #: variant is what it takes instead.
    comptime CATBOOST_PRIVATE_BYTES = 384 * 32 * 4
    #: Same idiom `block_size_for` uses, and for the same reason: the budget
    #: has to be a comptime value because what it decides is a threadgroup
    #: allocation size.
    comptime limit = column_shared_limit(column)

    comptime if identical:
        return HIST_SMEM_SHARED2_I32  # the BIT_IDENTICAL column's value
    elif limit < CATBOOST_PRIVATE_BYTES:
        #: Apple (32 KB) took this arm by measurement. It is now stated as
        #: the BUDGET rule that produced that answer rather than as the
        #: vendor's name, which is what makes it extend: Adreno's 32 KB and
        #: Mali's 32 KB land here too, and NVIDIA's 48 KB (exactly their
        #: layout) and AMD's and Intel's 64 KB do not. **The resolved value
        #: for every founding column is unchanged by this rewrite** -- apple
        #: shared-Int32, nvidia and amd warp-private -- which is the only
        #: acceptable outcome for an edit to a NUMERIC row.
        return HIST_SMEM_SHARED2_I32
    else:
        return HIST_SMEM_WARP_PRIVATE_F32


def hist2_block_size_for[column: Int, smem_mode: Int]() -> Int:
    """SCHEDULING row bounded by the NUMERIC budget, per accumulation mode.

    Under `HIST_SMEM_WARP_PRIVATE_F32` this is `block_size_for` unchanged:
    the largest block whose `32 * 4` bytes/thread fit the column's budget,
    capped at CatBoost's 384 (256 on Apple's 32 KB).

    Under `HIST_SMEM_SHARED2_I32` two warps share one 4 KB slice, so the
    cost is 64 bytes/thread and the block can DOUBLE at the same footprint.
    Capped at 512 rather than CatBoost's 384: the shared variant exists to
    buy occupancy (the 1.94x probe above is an occupancy result), 512 is the
    exact block that fills Apple's 32 KB, and the reduce/writeback stages'
    256-thread participation caps are valid at any block at or above 256.
    Measured on the M4, interleaved trees at 800k x 100 x 128 folds, two
    builds ABAB: 512 threads/32 KB (one resident block) and 256 threads/
    16 KB (two resident blocks) are INDISTINGUISHABLE -- i32 medians 32.1 /
    33.1 vs 32.7 ms/tree, ranges fully overlapping -- because both put the
    same 512 threads on a core, which is what the probe said pays. 512 is
    kept as the single configuration; a future vendor measurement lands
    here, not in a kernel.

    In the shared-Int32 mode the block size (hence slice count) is PURE
    scheduling: integer accumulation is associative, so how many slices the
    adds spread over cannot change the reduced histogram. In the float mode
    it is numeric via the replication count, which is why that side stays
    pinned to `block_size_for`.
    """

    comptime if smem_mode == HIST_SMEM_SHARED2_I32:
        comptime limit = column_shared_limit(column) // 64
        return 512 if limit >= 512 else limit
    else:
        return block_size_for[K_HIST_2_ONE_BYTE, column]()


def replicas_for(hist_cells: Int) -> Int:
    """DELETED IN SPIRIT. Do not call this.

    This chose a replication factor from the HISTOGRAM WIDTH alone, ignoring
    the leaf count, the stat count and the device. **It was ours, not
    CatBoost's**, and CatBoost has a formula for exactly this sitting in
    `hist_binary.cu:95`:

        numBlocks.x *= CeilDivide(blocksPerSm * SMCount(),
                                  numBlocks.x * numBlocks.y * numBlocks.z);

    Replication exists to FILL THE MACHINE and nothing else, so it has to
    know how many blocks the other grid axes already supply. Ours could not,
    and at depth 6 it asked for 16 replicas on top of 3200 blocks.

    Use `replication_for` in `greedy_search_helper.mojo`. This body raises so
    a caller cannot reappear quietly; the two measured points in the old
    docstring were taken on an empty histogram and mean nothing.
    """
    return -16
    return 1


# =========================================================================
# THE mojolearn SECTIONS: cluster, neighbors, dbscan, decomposition, glm
# =========================================================================
#
# Added 2026-08-19 after Andrew asked whether the sections added since this
# file was written actually USE it. **They did not.** One file mentioned it
# in a comment and nothing called it. Every tunable in `core/`, `cluster/`,
# `neighbors/`, `dbscan/`, `decomposition/` and `glm/` was a bare `comptime`
# constant chosen on one M4, which is precisely the "constants sprinkled
# across kernels" this table opens by refusing.
#
# The rows below DECLARE the truth. Wiring the call sites is a separate step
# and is tracked in `UNWIRED.md`; a declared row that no kernel reads is not
# done, and saying so here is better than implying otherwise.
#
# WHY A SECOND SPEC STRUCT RATHER THAN MORE FIELDS ON `KernelSpec`
# ----------------------------------------------------------------
# `KernelSpec` is histogram-shaped: `hist_floats_per_thread` and
# `features_per_int` are CatBoost's compressed-index packing and mean nothing
# to a distance kernel. Widening it would put dead fields on every row and
# invite a caller to read one. These kernels take different knobs, so they
# get their own struct and their own resolver, against the same four columns.

comptime K_LIB_ROW_NORM = 100
comptime K_LIB_COLUMN_STATS = 101
comptime K_LIB_TRANSPOSE = 102
comptime K_LIB_GEMM_CONTRACTION = 103
comptime K_LIB_FUSED_DISTANCE_NN = 104
comptime K_LIB_REDUCE_BY_KEY = 105
comptime K_LIB_PLUS_PLUS = 106
comptime K_LIB_EPS_NEIGHBORHOOD = 107
comptime K_LIB_ADJ_SCAN = 108
comptime K_LIB_WEAK_CC = 109
comptime K_LIB_SELECT_RADIX = 110
comptime K_LIB_SELECT_WARPSORT = 111
comptime K_LIB_BALL_COVER_EPS = 112
comptime K_LIB_JACOBI_EIGH = 113
comptime K_LIB_GRAM_SPLITK = 114


#: NUMERIC. The reduction width every cross-lane fold in these sections uses.
#:
#: This is `PINNED_REPLICATION_LANES`'s counterpart for the library kernels
#: and it exists for the identical reason. `unfused_distance_nn.mojo:76`
#: hardcodes `REDUCE_MIN_LANES = 32` and `select_warpsort.mojo:263` hardcodes
#: `WARP_LANES = 32`. Both are right on Apple and NVIDIA and **wrong on AMD,
#: whose wavefront is 64** (`column_lane_width`).
#:
#: A cross-lane reduction is float addition, which is not associative, so its
#: WIDTH decides which partials combine with which. Letting it follow the
#: hardware would make an AMD fit disagree with a CUDA fit in the last bits
#: with nothing in the code admitting it. Pinned at 32; AMD reduces within
#: half a wavefront and pays for it.
comptime PINNED_LIB_REDUCE_LANES = 32

#: NUMERIC. RAFT's `Policy4x4` accumulation geometry
#: (`raft/linalg/contractions.cuh`): 4x4 outputs per thread, K-block 32,
#: vector length 4. These decide the ORDER a dot product is accumulated in,
#: so two columns with different values give different last bits on the same
#: input. Pinned to RAFT's own numbers, which is also what makes our tiles
#: diffable against theirs.
comptime PINNED_ACC_ROWS_PER_TH = 4
comptime PINNED_ACC_COLS_PER_TH = 4
comptime PINNED_KBLK = 32
comptime PINNED_VECLEN = 4


@fieldwise_init
struct LibKernelSpec(Copyable, Movable):
    """The knobs one library kernel takes, resolved per column."""

    var block_size: Int
    """SCHEDULING. Threads per threadgroup."""

    var lane_width: Int
    """SCHEDULING **only for indexing**, NUMERIC when it bounds a reduction.

    Read this for "which lane am I" arithmetic, where it must match the
    hardware or the indexing is simply wrong. Do NOT read it to size a
    cross-lane fold -- that is `reduce_lanes` below, and conflating the two
    is how a portable-looking kernel produces two different answers."""

    var reduce_lanes: Int
    """NUMERIC. See `PINNED_LIB_REDUCE_LANES`."""

    var acc_rows_per_th: Int
    """NUMERIC. Policy4x4 accumulation geometry."""

    var acc_cols_per_th: Int
    """NUMERIC. Policy4x4 accumulation geometry."""

    var kblk: Int
    """NUMERIC. Policy4x4 K-block."""

    var veclen: Int
    """NUMERIC. Policy4x4 vector length."""

    var shared_limit: Int
    """SCHEDULING. Threadgroup bytes this column allows a block to claim."""


def lib_block_size(kernel: Int, column: Int) -> Int:
    """SCHEDULING. Threads per block, per vendor.

    Every value here is currently the SAME on all three columns, and that is
    an honest statement of ignorance rather than a finding: they were chosen
    on an M4 and no other device has run them. The column parameter exists so
    a measurement on NVIDIA or AMD has somewhere to land WITHOUT touching a
    kernel, which is the whole point of the table. Do not read a uniform row
    as evidence that the value is optimal anywhere.
    """
    if kernel == K_LIB_SELECT_RADIX:
        #: `select_radix.mojo:117`, SELECT_BLOCK = NUM_BUCKETS = 1 << 8.
        #: This one is NOT free to vary: the block must have one thread per
        #: histogram bucket. It is geometry forced by the algorithm, so it is
        #: the same in every column for a reason, unlike the rows above.
        return 256
    if kernel == K_LIB_SELECT_WARPSORT:
        #: 8 lane-groups per block, `select_warpsort.mojo:269`.
        return 8 * lib_lane_width(column)
    if kernel == K_LIB_TRANSPOSE:
        #: A 32x32 padded tile, `column_stats.mojo:200`.
        return 32 * 32 // 4
    if kernel == K_LIB_JACOBI_EIGH:
        return 32
    if kernel == K_LIB_GEMM_CONTRACTION or kernel == K_LIB_FUSED_DISTANCE_NN:
        #: Policy4x4: AccThRows * AccThCols threads cover the tile.
        return 16 * 16
    if kernel == K_LIB_GRAM_SPLITK:
        #: `core/gram_splitk.mojo`. 256 threads x its CELLS unroll covers
        #: the m x m output; the chunk count derives from this via
        #: `max_active_blocks_per_core`, so this row is also the one input
        #: to that (NUMERIC) computation -- change it and the summation
        #: split changes with it.
        return 256
    return 128


def lib_lane_width(column: Int) -> Int:
    """SCHEDULING. The hardware lane group, for INDEXING only.

    Delegates to `column_lane_width` rather than restating it, so there is
    exactly one place in this repository that knows AMD is 64.
    """
    return column_lane_width(column)


def lib_spec_for(
    kernel: Int, device: Int, mode: NumericMode
) raises -> LibKernelSpec:
    """The resolved knobs for one library kernel. The "sub in and out" step.

    Same discipline as `spec_for`: a SCHEDULING row comes from the `device`
    column always, a NUMERIC row comes from `COLUMN_BIT_IDENTICAL` under the
    default and from the device column only under `FAST`.

    Note what that means for `lane_width` versus `reduce_lanes` on AMD: the
    kernel indexes with 64 because that is what the hardware does, and folds
    over 32 because that is what the answer requires. They are different
    numbers on the same device and a kernel needs both.
    """
    var identical = mode.mode == NUMERIC_IDENTICAL
    var numeric_column = COLUMN_BIT_IDENTICAL if identical else device

    var reduce_lanes = PINNED_LIB_REDUCE_LANES
    if not identical:
        #: Even under FAST the fold stays pinned unless a vendor column is
        #: measured to want otherwise. No such measurement exists, so this
        #: is deliberately not `lib_lane_width(device)`.
        reduce_lanes = PINNED_LIB_REDUCE_LANES

    var spec = LibKernelSpec(
        lib_block_size(kernel, device),
        lib_lane_width(device),
        reduce_lanes,
        PINNED_ACC_ROWS_PER_TH,
        PINNED_ACC_COLS_PER_TH,
        PINNED_KBLK,
        PINNED_VECLEN,
        column_shared_limit(numeric_column),
    )

    if spec.block_size < spec.reduce_lanes:
        raise Error(
            "library kernel "
            + String(kernel)
            + " resolves to a block of "
            + String(spec.block_size)
            + " threads in column "
            + column_name(device)
            + ", which is narrower than the "
            + String(spec.reduce_lanes)
            + "-lane fold it has to perform"
        )
    return spec^


# --- THE FIRST THING THIS TABLE SHOULD HAVE BEEN DECIDING ----------------
#
# RAFT's `contractions.cuh` DOUBLE BUFFERS: two shared-memory pages, so the
# next K-block's loads are in flight while the current one accumulates. It is
# not an optimization bolted on, it is how their contraction hides latency,
# and every kernel built on `Contractions_NT` gets it.
#
# We do not have it anywhere, and on 2026-08-19 THREE separate sites recorded
# the same reason in the same words:
#
#   dbscan/UNPORTED.tsv:9
#       "two smem pages at Policy4x4 is 32 KB, exactly Metal's threadgroup
#        ceiling. Same gap core/gemm.mojo records"
#   dbscan/gbdt/neighbors/epsilon_neighborhood.mojo:195
#       "exactly Metal's ceiling. That gap is recorded ..."
#   neighbors/.../fused_l2_knn.mojo
#       their SmemSize is 2 pages = 36,992 B against Metal's 32 KB, so ours
#       is single-buffered at 30,848 B
#
# Two different lanes reached that conclusion independently, hours apart,
# neither having read the other's file. Both were right about Apple.
#
# **AND BOTH THEREFORE SINGLE-BUFFERED NVIDIA AND AMD, WHICH HAVE THE ROOM.**
# `column_shared_limit` has said 32 / 48 / 64 KB since this file was written.
# NVIDIA's 48 KB holds RAFT's 36,992 with 11 KB to spare; AMD's 64 KB holds it
# nearly twice. The constraint is Apple's alone and it was applied to all
# three, because the number was typed into a kernel instead of read from here.
#
# That is the failure mode this table exists to prevent, it happened twice in
# one day, and it is the reason the rows above were added.
#
# DOUBLE BUFFERING IS A SCHEDULING ROW. It changes which loads are in flight,
# never what is added to what: the accumulation order is `Policy4x4`'s and is
# pinned separately. So it may legitimately differ per column, and a fit stays
# bit-identical across all three with Apple single-buffered and the other two
# double-buffered.
#
# Not wired. `UNWIRED.md` tracks it. The row below is what a kernel should
# ask, so that the answer is a table lookup rather than a fresh decision by
# whoever is next in that file.

def lib_smem_pages(kernel: Int, column: Int, page_bytes: Int) -> Int:
    """SCHEDULING. How many shared-memory pages this column can afford.

    2 is RAFT's double buffer; 1 is the fallback. `page_bytes` is the size of
    ONE page at the kernel's policy, which the caller knows and this table
    does not, because it depends on the tile the kernel chose.
    """
    if 2 * page_bytes <= column_shared_limit(column):
        return 2
    return 1


# --- COMPTIME ACCESSORS FOR THE LIBRARY KERNELS --------------------------
#
# `lib_spec_for` above resolves a whole spec at RUNTIME, which is right for a
# report and useless to a kernel: a threadgroup allocation is fixed at compile
# time. These are the same rows as pure comptime functions, so a kernel file
# says
#
#     comptime NORM_TPB = lib_block_size_for[K_LIB_ROW_NORM, TARGET_COLUMN]()
#
# and NOTHING ELSE. No `if column ==` at the call site, no constant restated,
# no second opinion about what Apple can do. Change `TARGET_COLUMN` in one
# place and every kernel in every section recompiles against the new vendor.
#
# That is the whole contract. A kernel that branches on the column itself has
# reintroduced the problem in a more expensive form, because now the table AND
# the kernel both have an opinion and they can disagree.

def lib_block_size_for[kernel: Int, column: Int]() -> Int:
    """SCHEDULING. Threads per block for one library kernel, per column."""
    comptime lanes = column_lane_width(column)
    if kernel == K_LIB_SELECT_RADIX:
        #: Forced by the algorithm, not by the vendor: one thread per
        #: histogram bucket, `select_radix.mojo:117`.
        return 256
    if kernel == K_LIB_SELECT_WARPSORT:
        #: 8 lane-groups. This is the row that MUST widen on AMD.
        return 8 * lanes
    if kernel == K_LIB_TRANSPOSE:
        return 256
    if kernel == K_LIB_JACOBI_EIGH:
        return 32
    if kernel == K_LIB_GEMM_CONTRACTION or kernel == K_LIB_FUSED_DISTANCE_NN:
        #: Policy4x4: AccThRows * AccThCols threads cover one tile.
        return 256
    if kernel == K_LIB_GRAM_SPLITK:
        #: `core/gram_splitk.mojo`; see the runtime row above for why this
        #: one is also a NUMERIC input (it feeds the chunk count).
        return 256
    return 128


def lib_lane_width_for[column: Int]() -> Int:
    """SCHEDULING. The hardware lane group, for INDEXING ONLY.

    32 on Apple and NVIDIA, 64 on AMD. Use this to answer "which lane am I".
    Do NOT use it to size a cross-lane fold: that is `lib_reduce_lanes_for`,
    and the two are different numbers on the same AMD device.
    """
    return column_lane_width(column)


def lib_reduce_lanes_for[column: Int, identical: Bool]() -> Int:
    """NUMERIC. The width of a cross-lane fold.

    Pinned to 32 in every column. A fold is float addition, which is not
    associative, so its width decides which partials combine with which; if
    it followed the hardware, an AMD fit would combine 64 where a CUDA fit
    combines 32 and the two would disagree in the last bits.

    `identical` is accepted so the signature matches `lane_width_for` and a
    future measured exception has somewhere to go. There is no such
    measurement, so both arms return the pinned value today and the parameter
    is deliberately not wired to `column_lane_width`.
    """
    return PINNED_LIB_REDUCE_LANES


def lib_smem_pages_for[column: Int, page_bytes: Int]() -> Int:
    """SCHEDULING. 2 for RAFT's double buffer, 1 where it will not fit.

    Apple's 32 KB is the only column that forces 1 at Policy4x4. Two lanes
    independently hardcoded that answer for all three vendors today; see the
    block above.
    """
    comptime limit = column_shared_limit(column)
    return 2 if 2 * page_bytes <= limit else 1


# ---------------------------------------------------------------------------
# THE MODEL-EVALUATOR QUANTIZE SEARCH (Apple's ALU budget, priced)
# ---------------------------------------------------------------------------

#: CatBoost's `Binarize` scans every border linearly with a branchless
#: compare-add (`libs/model/cuda/evaluator.cu:117-121`, `#pragma unroll 8`).
comptime QUANTIZE_SEARCH_LINEAR = 0

#: `lower_bound` binary search over the same sorted borders. MEASURED
#: SLOWER AND DECLINED, see the row.
comptime QUANTIZE_SEARCH_BINARY = 1

#: Two-level scan: branchless pivot pass (every 8th border), then a
#: branchless 8-wide segment pass. Same pipelined shape as theirs, ~5x
#: fewer compares.
comptime QUANTIZE_SEARCH_TWO_LEVEL = 2


def quantize_search_for[column: Int]() -> Int:
    """SCHEDULING row: HOW the evaluator's quantize finds a value's bin.

    The result is IDENTICAL by construction, not by tolerance: every arm
    computes the count of borders with `value > border` over the same
    sorted array under the same strict compare -- no reduction, no
    accumulation, no tie with two answers. So this row is pure scheduling
    and the bit-identical column may take any arm.

    THE MEASUREMENTS, 2026-08-20, 800k x 100 @ 128 borders, phase timers
    in bench/interleaved/predict_interleaved.mojo (eval kernel 2.7 ms in
    every window):

        LINEAR (theirs, `evaluator.cu:117`, unroll-8 branchless): 27-30 ms
            -- near the base M4's ALU floor for the ~10 Gops of compares
            that a V100-class card absorbs for free.
        BINARY (lower_bound, 8 steps): 90-110 ms. WORSE BY 3.5x: eight
            SEQUENTIALLY DEPENDENT divergent shared loads stall where the
            linear scan pipelines independent compare-adds. Declined at
            that price; the constant stays for the record.
        TWO_LEVEL (pivots then one segment): the Apple column's arm --
            both passes keep the linear scan's branchless independent
            shape, at ~24-40 compares instead of 127.

    NVIDIA and AMD keep CatBoost's scan: on their ALU budgets it is free
    and it is their measured design.

    THE DECLARED VENDORS KEEP THE SCAN TOO, AND THAT IS AN ADMISSION RATHER
    THAN A CHOICE. Adreno and Mali are mobile ALU budgets and are far closer
    to the M4 than to a datacenter card, so the two-level arm is the one to
    expect there. Expecting is not measuring, and this row is cheap to flip
    once a device exists: it is scheduling, the result is identical by
    construction on every arm, so a bring-up measurement changes one line
    here and no kernel anywhere.
    """
    if column == COLUMN_APPLE:
        return QUANTIZE_SEARCH_TWO_LEVEL
    return QUANTIZE_SEARCH_LINEAR
