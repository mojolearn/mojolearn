# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""One table: every kernel, every tunable it takes, per GPU vendor.

NOT a port. CatBoost tunes against `TArchProps` scattered through the kernel
files and ships one vendor. **We ship Metal, CUDA and HIP from one source and
the product property is that a fit gives the SAME MODEL on all three.** That
cannot be maintained by constants sprinkled across kernels, so every knob any
kernel takes is declared here, once, with its vendor column and its class.

There is no CPU column. This tree has no CPU path.

THREE COLUMNS BUILD TODAY; THREE MORE ARE DECLARED AND DO NOT
--------------------------------------------------------------
`apple`, `nvidia`, `amd` (CDNA) and `amd-rdna` are what Mojo emits code for.
`qualcomm`, `intel` and the portable `spec-baseline` are declared with their
documented minimums and their admission verdict, and `column_is_buildable`
says which is which.

**THE COLUMNS ARE THE DEVICES PEOPLE TRAIN ON, plus the floor beneath them.**
An `arm` column for Mali was added and struck the same day: nobody trains a
gradient-boosted model on a phone GPU, and a column implies an intent this
project does not have. The Mali FACTS that are worth keeping -- it advertises
32 KB of compute shared memory that is not a scratchpad at all -- are kept in
`archive/reference/VENDOR_COLUMNS.md`, where a warning belongs, rather than as a build target
nobody will build.

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

from std.sys.compile import is_defined
from std.sys.info import (
    has_amd_gpu_accelerator,
    has_nvidia_gpu_accelerator,
)

from checks.numerics import (
    NumericMode,
    NUMERIC_FAST,
    NUMERIC_IDENTICAL,
    GLOBAL_NUMERIC_MODE,
)


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
#: AMD SHIPS TWO WAVEFRONT WIDTHS AND MOJO SUPPORTS BOTH FAMILIES, so one
#: `amd` column was wrong for half of them. `COLUMN_AMD` is CDNA -- MI300X
#: and MI355X, the parts Mojo tests continuously -- and is wave64.
#: `COLUMN_AMD_RDNA` is the consumer training card, RX 9070 / 7900 / 6900,
#: which runs **wave32**: RDNA's native compute mode, and the only one HIP
#: reaches (the `mwavefrontsize64` compiler option is experimental and the
#: HIP runtime does not support it).
#:
#: This is not a hypothetical distinction. `lib_block_size_for` sizes the
#: warpsort block as `8 * lane_width`, so the single `amd` column resolved
#: 512 threads for a part whose lane group is 32 and whose correct answer is
#: 256. A scheduling row, so no bit moved -- but wrong on hardware people
#: actually train on, which is the case this table exists to prevent.
comptime COLUMN_AMD_RDNA = 4
comptime COLUMN_QUALCOMM = 5
comptime COLUMN_INTEL = 6

#: NOT A VENDOR. The PORTABLE FLOOR: what the graphics specifications
#: GUARANTEE any conformant GPU provides, as opposed to what a particular
#: vendor's parts happen to provide.
#:
#: **It exists to be REFUSED, and that is its job.** Two things follow from
#: having it, and neither is available from a table of real vendors:
#:
#: 1. **It answers "what about some GPU we have not thought of" with a
#:    number instead of a shrug.** Any future conformant device clears this
#:    bar; a device that also clears the identity floor joins; one that only
#:    clears this does not. The open-ended question becomes a comparison.
#: 2. **It gives the admission gate its first refused member.** Until this
#:    column existed, `column_meets_identity_floor` returned True for every
#:    column in the table, so the refusal path had never once executed. An
#:    unreached branch is not a working guard, it is an untested one, and
#:    this repository's rule is that reach is proved per branch. This column
#:    is that proof, permanently, and `hardware_matrix_check` asserts the
#:    refusal is still happening.
#:
#: Values are the intersection of the two portable specifications:
#:   - Vulkan required limits: `maxComputeSharedMemorySize` 16,384 bytes,
#:     `maxComputeWorkGroupInvocations` 128.
#:   - WebGPU defaults: `maxComputeWorkgroupStorageSize` 16,384 bytes,
#:     `maxComputeInvocationsPerWorkgroup` 256.
#: The most restrictive of each: 16 KB and 128 threads.
comptime COLUMN_SPEC_BASELINE = 7

#: The count, so a report or a check loops instead of listing. Raise this and
#: `column_name` and every row below must answer for the new value; the
#: check in `checks/hardware_matrix_check.mojo` enforces that they do.
comptime COLUMN_COUNT = 8

#: Kept as aliases so kernel code can name the API or the part it targets
#: rather than the company. Metal is Apple's, CUDA is NVIDIA's, HIP is AMD's;
#: Adreno is Qualcomm's GPU and Xe is Intel's; CDNA and RDNA are AMD's two
#: architectures, which differ in the one number this table exists for.
comptime COLUMN_METAL = COLUMN_APPLE
comptime COLUMN_CUDA = COLUMN_NVIDIA
comptime COLUMN_HIP = COLUMN_AMD
comptime COLUMN_CDNA = COLUMN_AMD
comptime COLUMN_RDNA = COLUMN_AMD_RDNA
comptime COLUMN_ADRENO = COLUMN_QUALCOMM
comptime COLUMN_XE = COLUMN_INTEL


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
    if column == COLUMN_AMD_RDNA:
        return String("amd-rdna")
    if column == COLUMN_SPEC_BASELINE:
        return String("spec-baseline")
    return String("unknown")


def column_is_buildable(column: Int) -> Bool:
    """Whether Mojo can emit a kernel for this column TODAY.

    `False` is not a criticism of the column and does not make it decoration:
    an unbuildable column still answers "would identity survive this vendor",
    which is the question that has to be answered BEFORE the target exists,
    not after.

    Source: Mojo system requirements (read 2026-08-21) lists NVIDIA (Turing
    through Blackwell, driver 580+), AMD (RDNA2 through CDNA4, ROCm 6.3.3+ --
    BOTH families, which is why there are two AMD columns) and Apple silicon
    (M1-M5, macOS 15+, Xcode 16+). No Qualcomm or Intel GPU target is
    listed.
    """
    return (
        column == COLUMN_BIT_IDENTICAL
        or column == COLUMN_APPLE
        or column == COLUMN_NVIDIA
        or column == COLUMN_AMD
        or column == COLUMN_AMD_RDNA
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

#: `TPointHist<0,0,BlockSize>` and its 6/7/8-bit siblings -- CatBoost's OTHER
#: histogram family, the one BOTH of their oblivious searchers share and the
#: one this repository runs for single-target symmetric trees once rung 1 of
#: `archive/plans/NEXT_TWO.md` lands (`archive/reference/PORTING.md` 91 B).
#:
#: SAME SCRATCH SHAPE as `K_HIST_2_ONE_BYTE` and NOT the same kernel. Both
#: take `BlockSize * 32` floats at CatBoost's `BlockSize = 384`, and their
#: `SliceOffset()` is character-identical
#: (`pointwise_hist2_one_byte_5bit.cu:52-58` against
#: `hist_2_one_byte_5bit.cu:25-31`). They diverge at the REDUCE: this family
#: writes `Buffer[2 * (32 * f + fold) + w]`, stat-MINOR, and the other writes
#: `Histogram[32 * 4 * isSecondStat + 32 * f + fold]`, stat-MAJOR. Its own
#: row because a shared row would invite sharing the writeback.
comptime K_POINTWISE_HIST_2 = 8

#: `TPointHistHalfByte<BlockSize>`, the pointwise family's OTHER accumulator.
#: One struct serves BOTH of their small-bin kernels -- binary (32 one-bit
#: features per block) and half-byte (8 features of up to 16 bins) -- which
#: differ only in how they READ the reduced result, never in how they build
#: it (`pointwise_hist2_binary.cu:39` and `pointwise_hist2_half_byte.cu:42`
#: both instantiate `THist = TPointHistHalfByte<BlockSize>`).
#:
#: 16 floats per thread, not 32: half the bins, half the scratch. CatBoost
#: launches both at 768; the 32 KB ceiling over 16 floats gives 512, which
#: is exactly 32,768 bytes.
#:
#: 512 IS A FLOOR HERE AND NOT ONLY A BUDGET, which no other row can say.
#: Their `Reduce` folds the warp slices with `if (threadIdx.x < 512)`
#: (`pointwise_hist2_half_byte_template.cuh:78`), so a block below 512
#: leaves the top of the first slice unfolded and silently loses every fold
#: above the block size. The accumulator asserts it.
comptime K_POINTWISE_HIST_2_HALF_BYTE = 9

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
    constraint.

    **This field had no true case until 2026-08-21 and now it has three.**
    The sentence that stood here -- "No vendor forces it today, all three do
    float `atomicAdd`" -- was true of the founding columns and is deleted
    rather than annotated: on `qualcomm` and the portable baseline,
    float atomic add is not core (it arrives through
    `cl_ext_float_atomics` / `VK_EXT_shader_atomic_float`, both optional), so
    those columns take the fixed-point flush in BOTH modes. A report must
    never imply the user asked for a flush the mode did not choose."""

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
# that gets lost. Recorded in `archive/plans/UNWIRED.md`; it is a header field and an hour's
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
    Adreno's 32 KB matches Apple's exactly and Intel's 64 KB is above. The design turns out to have been floored by
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
    - amd-rdna 64 KB. Same LDS per workgroup as CDNA; the two AMD columns
               differ in wavefront width, not in memory.

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
    if column == COLUMN_AMD_RDNA:
        return 64 * 1024
    if column == COLUMN_SPEC_BASELINE:
        #: 16 KB. Vulkan's required `maxComputeSharedMemorySize` and WebGPU's
        #: default `maxComputeWorkgroupStorageSize` agree on this number, and
        #: it is HALF the identity floor. That is the whole finding: the
        #: guarantee we ship is not a portable-by-specification guarantee, it
        #: is a guarantee about devices whose real limits exceed the spec's,
        #: which every mainstream GPU's do.
        return 16 * 1024
    return IDENTITY_FLOOR_SHARED_BYTES  # BIT_IDENTICAL: frozen, not derived


def column_has_float_atomics(column: Int) -> Bool:
    """Whether this vendor can do `atomicAdd` on a `float` at all.

    **ALL THREE FOUNDING COLUMNS CAN, APPLE INCLUDED.** This row said "APPLE
    CANNOT. Metal has no floating-point atomic add" and that was FALSE.
    Probed 2026-08-19:

        Atomic.fetch_add(dst, Float32(1.0))   // 1024 threads
        -> 1024.0, exact

    The false claim is deleted rather than annotated. It had cost real money:
    it is the entire reason `checks/fixed_point.mojo` exists, and the
    fixed-point substitution carries a RANGE CONTRACT CatBoost's float atomic
    does not have -- a magnitude that failed to bound the partials overflowed
    Int32 and produced a dead histogram, which took a bisect to find.

    So `deterministic_flush` is a row every FOUNDING column can negotiate:
    `FAST` leaves all three on CatBoost's float atomic
    (`atomicAdd(dst + fold, val)`, every `AddToGlobalMemory`), and
    `IDENTICAL` pins them to the integer path so they agree bit for bit.

    **THE DECLARED VENDORS ARE WHERE THIS ROW STOPS BEING TRIVIAL, AND IT IS
    WHY `flush_forced_by_vendor` EXISTS.** On Adreno, float atomic add is not
    core OpenCL: it arrives through `cl_ext_float_atomics`, which requires
    OpenCL 2.0+ and is optional per device and per memory scope (Khronos
    OpenCL extension registry). A device without it CANNOT run `FAST` as
    written and must take the fixed-point flush -- which is the
    vendor forcing the mode's hand, exactly the case that field was built for
    and that no founding column exercises.

    Conservative until queried: `False` for the mobile columns and for the
    portable baseline (neither Vulkan core nor WebGPU core has a float atomic
    add). That is a claim about the WEAKEST device in the family, not about
    the best one. A PARTICULAR Adreno that advertises the extension can run
    the float path -- but the flush is comptime, two different bodies of
    code, so a build has to choose in advance and the safe choice is the one
    that runs everywhere in the family. Bring-up may add a second build, not
    a runtime branch.

    Intel is `True` because float atomics are broadly available on Xe; it is
    the one entry here that should be re-checked against a device rather than
    a document, and it is marked so in `archive/reference/VENDOR_COLUMNS.md`.
    """
    return (
        column == COLUMN_BIT_IDENTICAL
        or column == COLUMN_APPLE
        or column == COLUMN_NVIDIA
        or column == COLUMN_AMD
        or column == COLUMN_AMD_RDNA
        or column == COLUMN_INTEL
    )


def column_compares_flush_subnormals(column: Int) -> Bool:
    """CAPABILITY. Whether the vendor's float COMPARE flushes subnormals.

    Not a knob this project chooses -- a vendor property nobody chose --
    which is why it is a CAPABILITY row rather than a NUMERIC one. Moved
    here from `ensemble/.../quantiles.mojo`'s DEVIATION 403 block at that
    lane's request (the lane charter forbids it editing this file).

    MEASURED on Apple, 2026-08-22 (quantiles_check): a float32 subnormal
    survives host -> device -> host bit-exact, but `0x006CE3EE ==
    0x00000001` compares TRUE, subnormals compare equal to both zeros, and
    `subnormal + (-0.0)` returns `+0.0` -- arithmetic and comparison flush
    while storage does not. CUDA honors subnormals in both. Consequence:
    cuML's `unique` compare (`quantiles.cuh:104`) collapses subnormal
    classes on Apple that CUDA keeps, so `n_bins_array` differs per vendor
    under FAST (measured spread 5v6, 4v6, 5v7 on the boundary column).
    Copying cuML exactly on a device that cannot compare exactly is NOT
    AVAILABLE; `IDENTICAL` aligns every vendor to ONE behavior through
    `numerics.ftz` (DEVIATION 403, IDENTITY_PATHS row 10).

    Conservative until queried: only Apple is measured True. The declared
    columns are transcriptions -- re-check on first device contact, with
    `check-ieee-arith` per row 10's rule.
    """
    return column == COLUMN_APPLE


#: The per-operand truncation of a TF32 tensor-core product, as a relative
#: bound on ONE fp32 multiply: TF32 keeps 10 explicit mantissa bits, so each
#: operand is rounded to within 2^-11 of itself and the product of two is
#: within ~2^-10 = 9.77e-4 of the exact fp32 product. Accumulation is fp32.
#: A cell of `A . B^T` is therefore within `2^-10 * sum |a_t b_t|` of exact,
#: PLUS the ordinary fp32 accumulation budget. Rounded up to one decimal
#: digit so a check can print it. Measured on an H100, 2026-08-23, through
#: MAX 26.5.0's `linalg.matmul`: 2.4e-5 of the magnitude at 33x33x257
#: (`checks/gram_splitk_check.mojo`), 4e-5 at 512x64x32 (the k-means
#: unfused arm) -- both well inside this bound and ~100x outside the fp32
#: budget the checks held every column to before DEVIATION 529.
comptime VENDOR_TF32_PRODUCT_REL_BOUND = Float64(1.0e-3)


def column_vendor_fp32_matmul_is_tf32(column: Int) -> Bool:
    """CAPABILITY. Whether MAX's `linalg.matmul` on fp32 inputs runs TF32
    tensor cores on this vendor -- i.e. whether a FAST build's vendor
    matrix products (`core/gemm.mojo::gemm_nt`, `gemm_tn`'s transpose arm)
    are TF32-accuracy rather than fp32-accuracy.

    NOT a knob this project chooses; the library's default. Read from the
    pinned toolchain's source (`max/kernels/src/linalg/matmul`, tag
    `max/v26.5.0`), and it is the root cause of BOTH NVIDIA-only FAST
    failures of the 2026-08-23 H100 leg (DEVIATIONS 529 and 540, the
    k-means arm comparison and the Gram vendor arm):

    - `matmul/__init__.mojo:59`: `use_tf32: Bool = True` is the default and
      `:90` documents it as "allow TF32 tensor-core truncation for fp32
      inputs".
    - `matmul/gpu/__init__.mojo:493-498`: `use_tf32=False` is a
      COMPILE-TIME ASSERT on every NVIDIA part before SM100 (Blackwell):
      "use_tf32=False is only implemented for the SM100 matmul dispatch".
      So on an H100 there is no flag to ask for fp32; the opt-out does not
      compile.
    - `matmul/vendor/matmul.mojo:73,116`: the cuBLAS fallback (the arm a
      dynamic-shape fp32 product lands on) hard-codes
      `vendor_matmul[use_tf32=True]`, which at `vendor/blas.mojo:751-758`
      is `COMPUTE_32F_FAST_TF32` plus `CUBLAS_TF32_TENSOR_OP_MATH`; the
      source's own comment: "the result is not bit-wise identical to the
      result of FP32". The SM90 warp-specialized fp32 path is a tf32 `mma`
      as well (`_multistage_gemm_gpu.mojo:738`).

    WHY THE OTHER FOUNDING COLUMNS ANSWER FALSE, from the same source:
    Apple M1-M4 take `gemm_kernel_apple_8x8` (`gpu/__init__.mojo:657-688`),
    an fp32 `simdgroup_matrix` kernel -- the M4 this row was measured on
    returns the fused kernel's bits exactly. AMD CDNA's fp32 MFMA is
    `16x16x4` f32 (`layout/tensor_core.mojo:1446`), a full-precision
    instruction, and its rocBLAS fallback ignores `use_tf32`
    (`vendor/blas.mojo:929`, `compute_type = float32`). RDNA has no fp32
    WMMA at all in MAX and the others are undeclared, so they stay False --
    held to the fp32 budget until a device says otherwise, which is the
    direction a tolerance row should err.

    **THE APPLE COLUMN HAS A GENERATION INSIDE IT, AND THIS ROW CANNOT SEE
    IT AT COMPILE TIME.** On an M5 (`ctx.compute_capability() == 5`) MAX
    26.5.0 routes fp32 products with `m > 1, n >= 64, k >= 16` to
    `enqueue_apple_matmul`, whose simdgroup MMA "truncates them to fp19"
    (`utils_gpu.mojo:564-569`), and that path is ON BY DEFAULT --
    `MODULAR_APPLE_M5_ALLOW_LOSSY_F32_MATMUL` defaults to "1" and only "0"
    selects the precise kernel. fp19 is 10 explicit mantissa bits, the
    same width as TF32, so on an M5 the vendor arm is TF32-class unless the
    caller sets that variable. The column is one build for every Apple
    part, so the generation is a RUNTIME fact: checks ask
    `vendor_fp32_matmul_is_lossy(column, compute_capability)` below, which
    folds it in, rather than this predicate alone. Untested on an M5 --
    transcribed from the dispatch, flagged for first device contact.

    WHO READS THIS. Two checks, so the answer lives here rather than in
    either: `checks/gram_splitk_check.mojo` (the Gram vendor arm) and
    `cluster/checks/kmeans_check.mojo` (the unfused assignment arm),
    each of which holds the vendor arm to `VENDOR_TF32_PRODUCT_REL_BOUND`
    on a lossy column and to the fp32 budget on an exact one, and SAYS SO
    in the line it prints. IDENTICAL never consults this row: under
    IDENTICAL `gemm_nt` is `pinned_gemm_nt_kernel` and `gemm_tn` is the
    split-K kernel or a refusal (DEVIATIONS 521, 526), and neither calls
    `linalg.matmul`.
    """
    return column == COLUMN_NVIDIA


def vendor_fp32_matmul_is_lossy(column: Int, compute_capability: Int) -> Bool:
    """The runtime form of `column_vendor_fp32_matmul_is_tf32`: the column
    predicate OR'd with the one generation fact the column cannot carry --
    an Apple part reporting `compute_capability == 5` (M5) runs MAX
    26.5.0's fp19 simdgroup path by default. Pass
    `DeviceContext.compute_capability()`; it is the same query MAX's own
    dispatch makes (`matmul/gpu/__init__.mojo:619`). On every non-Apple
    column the second argument is ignored."""
    if column_vendor_fp32_matmul_is_tf32(column):
        return True
    return column == COLUMN_APPLE and compute_capability == 5


def vendor_fp32_matmul_precision_name(
    column: Int, compute_capability: Int
) -> String:
    """What a check prints beside its tolerance: the precision class of the
    vendor fp32 product on this build and device."""
    if column_vendor_fp32_matmul_is_tf32(column):
        return String("TF32 (10-bit mantissa tensor-core product)")
    if column == COLUMN_APPLE and compute_capability == 5:
        return String("fp19 (Apple M5 simdgroup MMA, 10-bit mantissa)")
    return String("fp32")


def column_has_threadgroup_int_atomics(column: Int) -> Bool:
    """Whether a block can `atomicAdd` an `Int32` in THREADGROUP memory.

    **THIS IS THE ONE CAPABILITY `IDENTICAL` CANNOT DO WITHOUT.** The safe
    column accumulates the hist_2 family in shared Int32 slices
    (`HIST_SMEM_SHARED2_I32`), because integer addition is associative and
    that is what makes the histogram independent of arrival order. A device
    without local integer atomics cannot take that path, and there is no
    substitute that keeps the property at a tolerable cost.

    **True in every column INCLUDING the portable baseline, which is the
    single luckiest fact in this design.** The one capability the identity
    guarantee cannot do without is the one that is core in every compute API
    there is: Vulkan core atomics operate on the Workgroup storage class, and
    WGSL has `atomicAdd` on `var<workgroup> atomic<i32>`. So the reason the
    baseline column is refused is a SIZE, never a missing instruction -- and
    a size is something a future profile could accommodate, where a missing
    instruction would not be.

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

    True in every column here, and the row exists because that is not a
    property of GPUs in general. Arm's GPU best-practices guide, verbatim:
    "Arm GPUs do not implement dedicated on-chip shared memory for compute
    shaders. The shared memory that is available to use is system RAM that is
    backed up by the load-store cache." Mali is not a column (nobody trains
    on a phone), but it is the existence proof that a conformant GPU can
    advertise the capacity and provide none of the speed.

    A replicated-histogram design is a bet that private scratch is much
    faster than the memory it spares. Any future column that answers False
    here loses that bet: the replication still gives the same ANSWER
    (identity is unaffected) and buys much less. Re-measure the replication
    factor there; do not inherit ours.
    """
    return True


def column_spec_guarantees_onchip_shared(column: Int) -> Bool:
    """Whether anything PROMISES the shared memory is on chip.

    Separate from `column_has_dedicated_shared_memory`, which records what a
    named vendor actually built. Neither Vulkan nor WebGPU says where
    Workgroup storage lives: Arm's Mali is a shipping, conformant
    implementation that backs it with cached system RAM, and a future device
    may do the same.

    Nothing reads this yet. It is declared because "the spec guarantees the
    capacity and says nothing about the speed" is the kind of distinction
    that gets discovered twice, and the second discovery is expensive.
    """
    return column != COLUMN_SPEC_BASELINE


def column_max_block_size(column: Int) -> Int:
    """Largest threadgroup the vendor will dispatch, before our budget bites.

    - every vendor column: 1024.
      (Adreno 5xx and later report a 1024 max OpenCL workgroup size; older
      parts are far smaller -- 128 on Adreno 320, 512 on Adreno 420 -- and
      are out of scope for a training library. Mali's 512, when it was a
      column, was likewise slack: 32 KB of shared memory already resolves
      every kernel here to 512 or less, so the cap never bound.)

    This bounds nothing today: every block this tree launches is <= 768, and
    the shared-memory budget binds first on every column. It is declared so
    that the FLOOR below can be checked against something rather than
    assumed, since the safe column's hist_2 block is 512.
    """
    if column == COLUMN_SPEC_BASELINE:
        #: 128, the most restrictive of Vulkan's required
        #: `maxComputeWorkGroupInvocations` (128) and WebGPU's default
        #: `maxComputeInvocationsPerWorkgroup` (256). A QUARTER of the
        #: identity floor's block, so this column fails two rows and not
        #: one; `identity_refusal_reason` reports the memory row because it
        #: is checked first, which is the more fundamental of the two.
        return 128
    return 1024


def column_lane_width(column: Int) -> Int:
    """Hardware lanes that move in lockstep: warp on NVIDIA, SIMD group on
    Apple, WAVEFRONT on AMD, wave on Adreno, sub-group on Intel.

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
    - amd: 64 on CDNA and 32 on RDNA, which is why they are two columns and
      not one. Same vendor, same API, different reduction geometry if
      anything ever read it instead of the pinned value.

    The values returned here are the documented MINIMUM for each family,
    which is the only safe reading for anything that sizes a buffer.

    **WHY THIS DOES NOT BREAK THE IDENTITY CLAIM, WHICH IS WORTH STATING
    PRECISELY BECAUSE IT LOOKS LIKE IT SHOULD.** `PINNED_REPLICATION_LANES`
    is a LOGICAL group width, not a hardware sync assumption. Every kernel in
    this tree syncs at `SYNC_BLOCK` (`sync_granularity_for`), so a pinned
    32-lane replication group
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
    if column == COLUMN_AMD_RDNA:
        #: 32. RDNA's native compute wavefront, and the only width HIP
        #: reaches: `mwavefrontsize64` is an experimental compiler option the
        #: HIP runtime does not support. So an RDNA build indexes by 32 where
        #: a CDNA build indexes by 64, on the same vendor.
        return 32
    if column == COLUMN_SPEC_BASELINE:
        #: 1. Neither specification guarantees a subgroup at all -- Vulkan
        #: subgroups are a 1.1 feature with a device-reported size, and core
        #: WGSL has no subgroup concept. The only width portable code may
        #: ASSUME is one, which is the strongest possible argument for a
        #: pinned LOGICAL group and against reading the hardware.
        return 1
    return 32


def column_lane_width_is_fixed(column: Int) -> Bool:
    """Whether `column_lane_width` is a property of the DEVICE or a decision
    the vendor's compiler makes per kernel.

    False for qualcomm and intel. Any code that indexes by hardware lane
    ("which lane am I") must read this first: on a variable-width column the
    only correct answers are to demand a width from the compiler where the
    API allows it (`intel_reqd_sub_group_size`, Metal-style attributes) or to
    stop indexing by lane.

    CORRECTED 2026-09-01, AND THE CORRECTION NARROWS THIS FUNCTION'S
    JUSTIFICATION. Two sentences stood here and above: that Mojo exposes no
    warp primitive on any vendor, and that nothing in this tree indexes by
    hardware lane. BOTH ARE FALSE. Thirteen files import `shuffle_xor` or
    `lane_id` (see archive/reference/VENDOR_COLUMNS.md, which now names them), `split_warp_reduce`
    sizes its butterfly by `WARP_SIZE`, and DEVIATION BLOCK 30 in
    `dbscan/impl/neighbors/epsilon_neighborhood.mojo` says in its own
    source that its width-16 reduction is correct only at a lane width of 32.

    WHAT SURVIVES, and it is what this function actually rests on: no
    LANE-GROUP BARRIER exists, `sync_granularity_for` is `SYNC_BLOCK` on every
    column, and `PINNED_REPLICATION_LANES` is a logical width. The identity
    floor is therefore still width-independent, which is the claim that
    matters here.

    WHAT NO LONGER FOLLOWS: the refusal below cannot be justified by "nothing
    indexes by lane", because things do. For `amd` the admission is carried by
    MEASUREMENT instead, three-vendor DBSCAN, k-NN and forest cards, plus the
    arithmetic fact that 16 divides 64. For `qualcomm` and `intel` the
    argument is OWED: a width-16 group does not fit an 8-wide wave, and
    neither column can be built for today, so nothing can settle it by
    running. That is a debt, not a measured defect, and it is recorded as one.
    """
    return (
        column != COLUMN_QUALCOMM
        and column != COLUMN_INTEL
        and column != COLUMN_SPEC_BASELINE
    )


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
    #: The DEVICE's dispatch cap bounds every kernel, histogram or not, and
    #: it comes from the device column even under `IDENTICAL`: it is a
    #: launch-validity wall, not an arithmetic choice. A column that cannot
    #: reach the identity floor's block is refused by
    #: `column_meets_identity_floor` before it ever gets here, so clamping
    #: cannot silently produce a different reduction on an admitted column.
    var hard_cap = column_max_block_size(device)
    if hard_cap < block:
        block = hard_cap
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
    # DELEGATE, do not restate. This line read `identical or
    # vendor_forces_flush` until 2026-08-29 -- a SECOND expression of the rule
    # `NumericMode.deterministic_flush()` already owns, and this file's own
    # `deterministic_flush_for` docstring says why that is a defect: "Two
    # expressions of one rule is how a rule drifts". It duly drifted. When the
    # middle tier landed, `deterministic_flush()` was re-keyed to
    # `PIN_DETERMINISM` and this copy was not, so `spec_for` REPORTED a float
    # atomic for a DETERMINISTIC build whose kernels were compiling the
    # fixed-point flush -- the report and the kernel disagreeing, which is
    # exactly the shape of the Adreno bug recorded in `deterministic_flush_for`.
    var flush = mode.deterministic_flush() or vendor_forces_flush

    return KernelSpec(
        block,
        floats_per_thread,
        per_int,
        # THE REPORT MUST SAY WHAT THE KERNELS COMPILE. This read
        # `column_lane_width(device)` under FAST until 2026-09-01, which
        # reported 64 for the CDNA column while every histogram kernel in
        # the tree now stripes its private replicas by the LOGICAL
        # `replication_lanes_for` in both modes (DEVIATION 1947). A report
        # that disagrees with the kernel is the exact defect the flush row
        # three lines up records; the hardware width is still available by
        # name through `column_lane_width` and `lane_width_for`.
        PINNED_REPLICATION_LANES,
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
#: device query. Cross-compiling for another vendor means rebuilding on (or
#: for) that vendor's box, which is the honest shape of the constraint.
#:
#: RESOLVED IN TWO LAYERS (2026-08-22, E1 findings):
#:
#: 1. AN EXPLICIT BUILD DEFINE WINS: `mojo build -D MOJOLEARN_COLUMN_<NAME>`
#:    (the build scripts spell it `MOJOLEARN_TARGET_COLUMN=<name>` in the
#:    environment and pass the define through). This is how a cross-compile
#:    or an RDNA part names its column -- `has_amd_gpu_accelerator` cannot
#:    tell a wavefront-64 CDNA part from a wavefront-32 RDNA one.
#: 2. OTHERWISE, THE BUILD MACHINE'S OWN ACCELERATOR. This was a hardcoded
#:    `COLUMN_APPLE`, so the first AMD build compiled every kernel against
#:    Apple's scheduling constants -- it RAN (Apple's numbers fit inside
#:    AMD's limits, and IDENTICAL pins the numeric rows to the floor
#:    anyway), but the matrix was describing a build nobody was running,
#:    the exact failure this file's own header names. Bare AMD resolves to
#:    the CDNA column (the datacenter parts E1 runs on).
comptime TARGET_COLUMN = (
    COLUMN_APPLE if is_defined["MOJOLEARN_COLUMN_APPLE"]() else
    COLUMN_NVIDIA if is_defined["MOJOLEARN_COLUMN_NVIDIA"]() else
    COLUMN_AMD if is_defined["MOJOLEARN_COLUMN_AMD"]() else
    COLUMN_AMD_RDNA if is_defined["MOJOLEARN_COLUMN_AMD_RDNA"]() else
    COLUMN_AMD if has_amd_gpu_accelerator() else
    COLUMN_NVIDIA if has_nvidia_gpu_accelerator() else
    COLUMN_APPLE
)


#: What the column would have been with NO `-D` override: the vendor of the
#: device this build will actually run on.
comptime DETECTED_COLUMN = (
    COLUMN_AMD if has_amd_gpu_accelerator() else
    COLUMN_NVIDIA if has_nvidia_gpu_accelerator() else
    COLUMN_APPLE
)


def column_is_simulated() -> Bool:
    """True when `-D MOJOLEARN_COLUMN_*` names a vendor this device is not.

    ADDED 2026-08-23, and it exists to bound a technique rather than to
    enable one. Building another vendor's column locally
    (`-D MOJOLEARN_COLUMN_AMD=1` on an M4) is genuinely useful --
    `tools/check_column_invariance.sh` is built on it and it found DEVIATIONS
    509/512/513 before any AMD hardware existed -- because most of what a
    column decides is a CONSTANT that the source responds to: a block size, a
    shared budget, a core count, a grid.

    IT IS NOT USEFUL, AND IS ACTIVELY MISLEADING, FOR A KERNEL WHOSE
    CORRECTNESS IS COUPLED TO THE HARDWARE'S LANE WIDTH. `vote` returns one
    bit per lane of the REAL wavefront. A 64-lane ballot compiled for
    `COLUMN_AMD` and executed on a 32-lane Metal warp does not approximate
    AMD's answer, it is wrong: the top 32 bits are never set and the
    `pop_count(mask & lid_mask)` positions are garbage.

    THE PREVIOUS STATE WAS WORSE THAN A FAILING CHECK, WHICH IS WHY THIS
    EXISTS. Before DEVIATION 515, `RBC_LANES` was the literal 32, so an
    AMD-column build silently compiled a THIRTY-TWO-lane ball cover and
    `check_dbscan_arms_agree_on_the_border` passed on it -- a green check
    that was testing the Apple kernel while its header said AMD. Making
    `RBC_LANES` follow the column turned that false pass into a real
    failure, and this predicate is how a check says "not answerable on this
    build" instead of either lying or failing.
    """
    return TARGET_COLUMN != DETECTED_COLUMN


def hist_floats_per_thread_for[kernel: Int]() -> Int:
    """Shared floats per thread. `GetHistSize()` is this times the block."""
    if (
        kernel == K_HIST_ONE_BYTE
        or kernel == K_HIST_2_ONE_BYTE
        or kernel == K_POINTWISE_HIST_2
    ):
        return 32
    return 16


def catboost_block_for[kernel: Int]() -> Int:
    """What CatBoost uses, before our shared-memory budget bites."""
    if (
        kernel == K_HIST_ONE_BYTE
        or kernel == K_HIST_2_ONE_BYTE
        or kernel == K_POINTWISE_HIST_2
    ):
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

    IDENTITY GATE (2026-08-22, closes the audit finding that IDENTITY_PATHS
    row 3 was closed only at `spec_for`, the runtime REPORT, while this
    comptime accessor -- the one the kernels actually compile against --
    read the device column unconditionally): under a `NUMERIC_IDENTICAL`
    build the budget is `IDENTITY_FLOOR_SHARED_BYTES` and the block is
    additionally capped at `IDENTITY_FLOOR_BLOCK`, mirroring `spec_for`'s
    resolution exactly. The mode is read HERE rather than taken as a
    parameter so every caller -- the hist kernels and the hist2/pointwise
    wrappers that derive from this -- gates at once with no call-site
    wiring, the same one-line-toggle argument that created
    `GLOBAL_NUMERIC_MODE`. On Apple the floor and the column coincide
    (32 KB, block 512), so both modes give the same numbers there; the
    gate moves the NVIDIA column's 768 to 512 (and the one-byte family's
    384 to 256) under IDENTICAL, one geometry on every vendor. The device
    dispatch cap below still applies: a column that cannot reach the floor
    is refused by `column_meets_identity_floor` before any launch.
    """
    comptime floats = hist_floats_per_thread_for[kernel]()
    comptime identical = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    comptime cb_cap = catboost_block_for[kernel]()
    comptime cap = (
        IDENTITY_FLOOR_BLOCK if identical
        and IDENTITY_FLOOR_BLOCK < cb_cap else cb_cap
    )
    comptime budget = (
        IDENTITY_FLOOR_SHARED_BYTES if identical
        else column_shared_limit(column)
    )
    comptime limit = budget // (floats * 4)
    comptime by_smem = limit if limit < cap else cap
    #: AND THE VENDOR'S DISPATCH CAP, which this row did not consult until
    #: 2026-08-21. On every column that could be built it was slack -- all
    #: three founding caps are 1024 and the largest block here is 768, so
    #: the memory budget always bound first and the omission was invisible.
    #: The portable-baseline column exposed it on its first run: 16 KB of
    #: shared memory permits 256 threads, the specifications guarantee only
    #: 128 invocations per workgroup, and the resolver was handing back a
    #: block the device is not required to be able to launch at all. A
    #: declared column earning its place on day one.
    comptime hard = column_max_block_size(column)
    return by_smem if by_smem < hard else hard


def lane_width_for[column: Int, identical: Bool]() -> Int:
    """NUMERIC. Pinned to 32 under identity so AMD's 64-wide wavefront cannot
    change the reduction geometry. See `column_lane_width`.

    NOT THE ROW A HISTOGRAM SLICE LAYOUT WANTS, and that is why this
    docstring now carries a pointer. Under `FAST` this returns the HARDWARE
    width, so a CDNA column got 64 and every constant derived from it moved
    while the layout literals beside it stayed at 32 and 512. That is what
    made the sub-byte and one-byte ladders refuse on a 64-wide wave. A
    kernel whose slice geometry is a LOGICAL striping decision reads
    `replication_lanes_for` instead; this row survives for anything that
    genuinely needs to know how wide the hardware executes.
    """
    if identical:
        return PINNED_REPLICATION_LANES
    return column_lane_width(column)


def replication_lanes_for[column: Int, identical: Bool]() -> Int:
    """NUMERIC, and it is the row the whole CatBoost histogram family's
    layout actually rests on: the LOGICAL width of one private-replica
    group. `PINNED_REPLICATION_LANES` in every cell of the table, in BOTH
    modes, on every column, whatever the hardware executes underneath.

    WHY THIS IS A DIFFERENT ROW FROM `lane_width_for`, and why splitting
    them ADDS a capability rather than papering over one. CatBoost's
    accumulators cut the block into groups of 32 consecutive `threadIdx.x`
    and give each group a private slice: `512 * (tid / 32)` in the sub-byte
    families, `1024 * (tid / 32)` in the one-byte ladder. Nothing in that
    arithmetic asks how many lanes retire per cycle. It asks only that the
    groups be DISJOINT and that the fold stride at the end match the slice
    stride. Both hold for any block size that is a multiple of 32, on
    hardware of any width.

    What the 32 was doing in `lane_width_for` was two jobs at once: naming
    the private-replica group AND naming the hardware wave. On Apple and
    NVIDIA those are the same number and the conflation is invisible. On
    CDNA they are 32 and 64, the derived `REDUCE_WIDTH` moved to 1024 while
    the literal 512 in `slice_offset` did not, and stage 1 folded cells
    belonging to different bins. The refusals (DEVIATIONS 1906 and 1910)
    were placed against that inconsistency, not against the layout.

    THE IDENTICAL PATH ALREADY HELD THE ANSWER. `lane_width_for` returns the
    pinned 32 under `IDENTICAL` on every column, the asserts pass, and the
    whole ladder including both sub-byte families elaborates on AMD today.
    So the spelling that makes a 32-slot layout valid on a 64-wide wave was
    already in the tree and was reachable from one mode only. This row is
    that spelling, made unconditional. See DEVIATION 1947.

    TWO PRECEDENTS, one of which has already RUN on a 64-wide wave:
    `gbdt/methods/kernel/pointwise_hist2_half_byte_template.mojo` hardcodes
    the same 32 as a logical constant, never consults this table, and
    therefore compiles at any width; and
    `gbdt/methods/greedy_subsets_searcher/kernel/hist_2_one_byte_8bit.mojo`
    stripes its loads by a logical `H8_LANE = 32` and is the kernel the
    MI325X has been running one-byte blocks through since 2026-08-27.

    THE PARAMETERS ARE DELIBERATELY IGNORED. They are kept so a call site
    reads like every other matrix accessor and so the answer's INDEPENDENCE
    from column and mode is stated in the signature rather than assumed.
    A build that wants the hardware number must say so by name.

    WHAT THIS ROW DOES NOT BUY, stated here because it is the thing a reader
    will over-read. A width-independent LAYOUT is not a width-independent
    SYNC. The sub-byte accumulator's turn-taking sync is a separate row,
    `sub_byte_lane_sync_for`, and it is what actually changes on a 64-wide
    wave.
    """
    comptime assert PINNED_REPLICATION_LANES == IDENTITY_FLOOR_LANES, (
        "the logical replication width and the identity floor's lane count"
        " are one guarantee with two names; keep them equal"
    )
    return PINNED_REPLICATION_LANES


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
#: REACHABLE: `max.gpu.sync.syncwarp` is imported and called by four gbdt
#: kernels. The row below stays SYNC_BLOCK because no kernel outside that
#: family RELIES on lane-group sync, not because none is available.
comptime SYNC_LANE = 1


def sync_granularity_for[column: Int]() -> Int:
    """The finest sync a kernel may rely on. `SYNC_BLOCK` for EVERY column.

    CORRECTED 2026-09-01. THIS DOCSTRING SAID MOJO EXPOSES NO WARP PRIMITIVE
    ON ANY VENDOR. THAT IS FALSE, and it is the third copy of a claim already
    retracted elsewhere in this file and in archive/reference/VENDOR_COLUMNS.md; commit 858f9811
    fixed two and missed this one. Twenty-four files import `shuffle_xor` from
    `std.gpu.primitives.warp`, and four gbdt kernels import and call
    `syncwarp` from `max.gpu.sync`, which is precisely a lane-group barrier.
    CatBoost's `tiled_partition<8>` and `tiled_partition<32>` DO have a
    counterpart we can emit.

    WHAT IS ACTUALLY TRUE, and it is what this row rests on: no kernel in this
    tree outside the gbdt sub-byte family RELIES on lane-group sync for
    correctness, so `SYNC_BLOCK` is the finest granularity a kernel may
    ASSUME. That is a statement about what our kernels depend on, not about
    what the language offers, and the difference matters because three vendor
    admissions were resting on the stronger, false version.

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


def sub_byte_lane_sync_for[column: Int]() -> Int:
    """SCHEDULING row (DEVIATION 1947): which barrier the CatBoost
    histogram accumulators use for their TURN-TAKING sync, the one that
    stands where theirs writes `tiled_partition<8>::sync()` or
    `tiled_partition<32>::sync()` between two writes to the same private
    slice.

    `SYNC_LANE` (`syncwarp`, 32 lanes) where the hardware wave is exactly
    the logical replication group; `SYNC_BLOCK` (`barrier()`) everywhere
    else.

    WHAT THE SYNC IS FOR, because the answer follows from it. Inside one
    8-lane tile the eight lanes hold eight distinct values of `tid & 7`, so
    on a single iteration `i` they touch eight distinct slots and cannot
    collide. ACROSS iterations they can: lane a at `i` and lane b at `i+1`
    can land on the same slot of the same private slice. The sync makes the
    store from iteration `i` visible to the tile before iteration `i+1`
    reads it. It must therefore (a) order execution over AT LEAST the eight
    lanes of the tile and (b) act as a memory barrier for the shared slice.

    WHY 32 IS RIGHT ONLY WHEN THE WAVE IS 32. `syncwarp` orders one warp.
    On a 32-wide machine that is a strict SUPERSET of the 8 lanes theirs
    orders, which is why the port has always been correct on Apple and
    NVIDIA and why swapping it in bought 21% (archive/reference/VENDOR_LIBS.md section 4).
    Two directions break that:

    - NARROWER THAN 32 (`qualcomm` 8, `intel` 8, the spec baseline 1, and
      on the first two the width is a COMPILER decision, not a device
      property -- `column_lane_width_is_fixed`). A logical 32-lane group
      then spans several hardware waves and a wave-local sync does not
      order the group at all. Nothing has ever built on those columns, so
      this was latent, not observed.
    - WIDER THAN 32 (`amd`, CDNA, 64). Here the SUPERSET argument still
      holds arithmetically -- 64 lanes contain the 8 -- and every call in
      this family is unconditional with a block-uniform trip count
      (`requires_uniform_iteration_for`), so no lane can skip a sync its
      neighbours reach. What CANNOT be established by reading is what
      `max.gpu.sync.syncwarp` LOWERS TO on a 64-wide wave: whether it emits
      a wave barrier with the fence, or is folded to nothing on the
      assumption that a wavefront is already in lockstep, in which case the
      compiler is free to keep a slice cell in a register across the
      iteration boundary and the tile's other lanes never see the store.
      That is a memory-model question about a toolchain, and the honest
      answer is that the source is not in this tree to read.

    So the row takes the conservative side on every column that is not
    exactly 32 wide: `barrier()`, whose participation requirement is
    already met and whose semantics are the same on every backend.

    THIS IS A SCHEDULING ROW AND IT MOVES NO BITS, which is the reason it
    may differ per column at all. Widening a sync changes WHICH threads
    wait, never what is added to what: two lanes of one tile never write
    one slot within an iteration, two tiles never share a slot (`tid & 24`
    gives each its own eighth of the 32 slots a bin spans), and two logical
    warps never share a slice. Per slot the add order is program order
    inside one tile, whatever the barrier's width. The IDENTICAL column and
    every 32-lane column keep `syncwarp` and are byte-for-byte untouched.

    THE PRICE, on `amd` only among buildable columns: up to eight
    threadgroup barriers per point in the sub-byte families instead of
    eight warp syncs. That is the cost `pointwise_one_byte_fixed_for`
    measured at 4-7x on APPLE for the pointwise ladder, and it is the
    reason that row exists; a 64-lane column pays it here to get a
    low-cardinality histogram AT ALL rather than a refusal. When someone
    establishes what `syncwarp` emits on CDNA, or Mojo grows a documented
    wave-width barrier, this row is the single place that flips, and the
    A/B is a one-line change with the sub-byte gate already standing.
    """
    if not column_lane_width_is_fixed(column):
        return SYNC_BLOCK
    if column_lane_width(column) != PINNED_REPLICATION_LANES:
        return SYNC_BLOCK
    return SYNC_LANE


def deterministic_flush_for[column: Int, identical: Bool]() -> Bool:
    """NUMERIC row, comptime, so a kernel can branch on it.

    Whether a multi-block histogram flush sums through a FIXED-POINT integer
    accumulator instead of `atomicAdd` on `float`.

    `IDENTICAL` pins every vendor to the integer path, where integer addition
    is associative and the result does not depend on which block lands first.
    `FAST` leaves a vendor on CatBoost's float atomic **if it has one**.

    **THIS ROW USED TO RETURN `identical` AND IGNORE THE COLUMN, AND THAT WAS
    A LATENT BUG THE MOMENT A COLUMN WITHOUT FLOAT ATOMICS WAS DECLARED.**
    The runtime resolver (`spec_for`) has always computed
    `identical or not column_has_float_atomics(device)`; this comptime
    accessor -- the one KERNELS actually branch on -- computed only the first
    half. While all three founding columns had float atomics the two agreed
    and nothing could go wrong. On Adreno under `FAST` they would have
    disagreed: the report would say fixed point and the kernel would
    emit a float `atomicAdd` the device does not have. Found 2026-08-21 by
    Andrew asking whether the float-atomic row should be `no` for those two.
    Two expressions of one rule is how a rule drifts; this one now delegates.

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
    return identical or not column_has_float_atomics(column)


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


def pointwise_one_byte_fixed_for[column: Int, identical: Bool]() -> Bool:
    """NUMERIC row: whether the POINTWISE one-byte family routes EVERY
    width through the 8-bit fixed-point accumulator.

    CatBoost's dispatch hands each feature to the narrowest accumulator
    that fits it (5/6/7/8 bits), and the narrow ones accumulate floats
    with warp-synchronous turn-taking -- free on hardware where a warp
    executes in lockstep. Mojo has no warp barrier, so the port raises
    each turn's sync to a FULL THREADGROUP `barrier()` inside
    `add_point` -- up to eight per point -- and the narrow accumulators
    arrive 4-7x off pace on Apple (measured 2026-08-21: the pointwise
    searcher's histogram phase at 105 ms/tree on eps500, 95% of its
    3.5x arm gap, every feature on the 7-bit float path). The 8-bit
    accumulator already holds Int32 fixed point with LOCAL INTEGER
    ATOMICS and no per-point barrier (DEVIATION 93), which is the same
    substitution `HIST_SMEM_SHARED2_I32` measured at 1.94x for the
    greedy family.

    So the APPLE column routes all one-byte widths to the 8-bit
    fixed-point kernel: `pw_bounds[8]` widens to (15, 256] and the
    5/6/7 launches are skipped. `IDENTICAL` takes the same route
    everywhere -- integer accumulation is the cross-device arithmetic.
    NVIDIA and AMD under FAST keep CatBoost's dispatch, because on THEIR
    hardware the turn-taking is what CatBoost designed it to be.

    ONE SENTENCE THERE IS CORRECTED (2026-09-01). It read "their warps ARE
    in lockstep and their turn-taking is genuinely free", which is true of
    CatBoost's kernel and NOT of this port of it: the pointwise ladder
    raises every turn to a threadgroup `barrier()`
    (`pointwise_hist2_one_byte_templ.mojo`, `..._5bit.mojo`), as the
    paragraph above says in as many words, so those columns pay the wide
    sync too and simply have not been measured paying it. The greedy
    family's equivalent row is `sub_byte_lane_sync_for`, which keeps
    `syncwarp` where the hardware wave is 32 and takes `barrier()`
    elsewhere; this row's A/B for NVIDIA and AMD is OWED and unrelated to
    DEVIATION 1947.

    A NUMERIC row, not a scheduling one: float adds become quantized
    Int32 adds, the same trade DEVIATION 93 already made for 8-bit
    features -- and on Apple it makes the pointwise arm's histogram
    arithmetic IDENTICAL to the greedy arm's, which the two-searcher
    bit-identity gate then checks for free.
    """
    comptime if identical:
        return True
    return column == COLUMN_APPLE


def greedy_one_byte_fixed_for[column: Int, identical: Bool]() -> Bool:
    """SCHEDULING row (DEVIATION 1906, NARROWED by DEVIATION 1947): whether
    the GREEDY one-byte family routes EVERY width through the fused 8-bit
    fixed-point kernel (`hist_2_one_byte_8bit.mojo`) instead of CatBoost's
    maxBins ladder.

    THIS ROW USED TO BE A REFUSAL DRESSED AS A ROUTE, AND THAT PART IS
    RETRACTED. What stood here was that the ladder's kernels are "32-LANE BY
    CONSTRUCTION" and that a CDNA column under FAST therefore had NO one-byte
    kernel at all, so this route was the only one available. The first half
    was a conflation and the second no longer follows. The ladder's slice
    arithmetic -- `1024 * (tid / 32)` and the sub-copy masks -- cuts the
    block into DISJOINT groups of 32 consecutive threads, which is a LOGICAL
    striping decision that holds at any hardware width; `Reduce`'s stage 1
    folds at a LITERAL 1024 that never moved. The only thing that was truly
    coupled to the wave was the constant those files read for the striping,
    `lane_width_for`, which hands back the hardware 64 under FAST and left
    the derived quantities disagreeing with the literals beside them. They
    read `replication_lanes_for` now, the comptime asserts hold by
    construction, and the ladder elaborates on a 64-wide wave exactly as it
    already did under IDENTICAL.

    So the ladder is AVAILABLE on a 64-lane column and this row is now what
    it always should have been: a PERFORMANCE PREFERENCE with a price, not a
    statement about what can be built.

    WHAT THE PREFERENCE BUYS, and it is unchanged. The fused 8-bit kernel
    accumulates Int32 in ONE shared histogram per warp pair with threadgroup
    integer atomics, syncs only at block barriers, and walks the compressed
    index ONCE for both stats where the PASS family walks it per stat. The
    ladder's narrow kernels instead take CatBoost's turn-taking float adds,
    and on any column that is not exactly 32 lanes wide those turns now cost
    a THREADGROUP barrier each (`sub_byte_lane_sync_for`), which is the
    4-7x that `pointwise_one_byte_fixed_for` measured on Apple. A 64-lane
    column keeping the fused route is the same judgement Apple already
    makes, on the same evidence.

    WHAT IT COSTS: the fused kernel gives a 64-lane column the general
    256-bin layout for a 32-fold feature, where the ladder would buy more
    private sub-copies and less shared traffic. That A/B is now RUNNABLE on
    a 64-lane column for the first time, and it was not before; this row is
    the one line that flips if the number disagrees. It is OWED, and the
    orchestrator owns it.

    `IDENTICAL` is UNTOUCHED and returns False: identity has always compiled
    the ladder and the identical route must not move -- identity reaches the
    fused kernel through `hist_smem_mode_for`'s shared-Int32 arm, not through
    this row. 32-lane columns under FAST returned False and kept CatBoost's
    dispatch byte for byte until 2026-09-03, when the measurement below
    flipped them; the block under the signature has the numbers. Variable-width columns
    (qualcomm, intel, spec-baseline) are still NOT claimed here -- their
    problem was never the slice layout, it is that a logical 32-lane group
    does not fit an 8-wide wave, which `sub_byte_lane_sync_for` answers and
    no column has yet built to confirm.
    """
    comptime if identical:
        return False
    # DEVIATION 2043, NOW ON BY DEFAULT ON EVERY FAST COLUMN. It began as a
    # diagnostic for one question -- why does IDENTICAL beat FAST on the gbdt
    # lane on an H100 (0.938) when the AMD cause, the decoupled-lookback
    # partition of DEVIATION 2042, is measured INERT there (1.003)? -- and
    # the answer turned out to be a shipping decision.
    #
    # The line below is the asymmetry. It returns `column_lane_width == 64`,
    # so under FAST it is TRUE on AMD's 64-wide CDNA and FALSE on NVIDIA's
    # 32-wide SM. At 254 borders every feature is one-byte with more than
    # 128 bins, and the dispatch in `greedy_search_helper` keys on this row,
    # so of the four cells (fast/identical x nvidia/amd) FAST-ON-NVIDIA IS
    # THE ONLY ONE that falls through to CatBoost's PASS(8) ladder --
    # `gridDim.z = stat_count`, TWO FULL WALKS over the compressed index per
    # level. The other three run the fused two-stat kernel: one walk, one
    # launch, half the cindex traffic. Same shape as 2042: a route only the
    # fast arm compiles, and it is the slower one.
    #
    # THE HASHES DIFFER AND THAT IS NOT A DEFECT: the fused arm accumulates
    # dithered fixed-point Int32 and the PASS arm accumulates warp-private
    # f32. What stood here was that a moved hash therefore made this a
    # TRADEOFF REPORT on which "NO DEFAULT MAY FLIP". **THAT IS RETRACTED**,
    # because it applied an IDENTICAL-tier gate to a FAST-tier decision.
    # FAST promises no bits by construction ([[fast-is-not-identical]]), so a
    # moved hash there is evidence of nothing; the gate a FAST default turns
    # on is FASTER AND NO WORSE ON HELD-OUT ACCURACY. Nothing had measured
    # the second half, so the flip was withheld on the wrong criterion.
    #
    # MEASURED 2026-09-03, H100, higgs 1,000,000 x 28, gbdt-symmetric,
    # 3 rounds, held-out AUC on the fixed 500,000-row tail
    # (`tools/gbdt_accuracy_ab.sh`, bench/results/e1/2026-09-03_190949):
    #
    #     FAST as shipped (PASS ladder, two walks)  0.9728 s   AUC 0.800447
    #     FAST fused one-byte (this row True)       0.9349 s   AUC 0.800538
    #     IDENTICAL, same box, same fixture         0.9195 s   AUC 0.800716
    #
    # ratio 0.961, d_AUC +0.000091, d_logloss -0.000001, and base's own AUC
    # band across the three rounds was ZERO, so the deltas are not spread.
    # FASTER AND SLIGHTLY MORE ACCURATE, so the row flips.
    #
    # THE QUANTIZATION IS NOT A PRECISION-FOR-SPEED TRADE, which is the
    # thing everyone expects it to be. `choose_scale` picks a POWER OF TWO,
    # so `val * fixed_scale` is exact; integer addition never rounds, where
    # an f32 bin absorbs small gradients once its running sum is large; and
    # `hist2_quantize`'s dither makes the single load-time rounding
    # zero-mean. IDENTICAL, the most quantized arm of the three, scores the
    # BEST AUC here.
    #
    # WHAT THIS DOES NOT CLOSE: IDENTICAL is STILL faster than the fused
    # FAST arm (0.9195 vs 0.9349). This row narrows NVIDIA's FAST deficit
    # from 5.8% to 1.7% and does not eliminate it, so a second cause remains
    # unlocated and FAST-on-NVIDIA is still the slow cell.
    #
    # SCOPE: one lane, one dataset, one rung, one column. gbdt-lossguide and
    # gbdt-depthwise are UNMEASURED on this row, and so is every fixture
    # that is not higgs at a million rows.
    comptime if is_defined["MOJOLEARN_2043_FAST_FUSED_ONE_BYTE"]():
        return True
    return True


def greedy_sub_byte_excluded_for[column: Int, identical: Bool]() -> Bool:
    """ROUTING row, RETRACTED 2026-09-01 (DEVIATION 1947 supersedes
    DEVIATION 1910): whether the GREEDY sub-byte histogram families --
    BINARY (32 features per word) and HALF-BYTE (8 per word) -- are
    comptime-EXCLUDED from the build, their launch sites refusing at runtime
    BY NAME.

    **False in every cell, and the row is kept rather than deleted because a
    retraction that leaves no trace is how the same wrong conclusion gets
    re-derived.** DEVIATION 1910 read like a fact about CatBoost's design
    and it was a fact about ONE CONSTANT WE CHOSE.

    WHAT 1910 SAID, in its own terms. Both families accumulate through
    `point_hist_half_byte_template.slice_offset`, whose 512-float slices and
    4 sub-copies are "32-lane by construction", so a 64-wide wavefront would
    need eight sub-copies and two of them would collide on every bin;
    CatBoost never wrote that layout; therefore a CDNA column under FAST
    must build without the kernels and refuse at the dispatch.

    WHAT IS ACTUALLY TRUE. `slice_offset` is `512 * (tid // 32) + (tid & 24)`.
    Every group of 32 CONSECUTIVE THREAD INDICES gets its own 512-float
    replica and every group of 8 its own eighth of the 32 slots a bin spans.
    That partition is disjoint and exhaustive for any block that is a
    multiple of 32 -- 512 threads give 16 replicas of 512 floats, which is
    exactly the `BlockSize * 16` of scratch the accumulator allocates -- and
    it never asks how many lanes the hardware retires together. A 64-wide
    wavefront does not need eight sub-copies; it runs two of the logical
    32-groups at once, on two disjoint replicas, which is what a wide wave
    is for. The thing that actually broke was `REDUCE_WIDTH`, derived from
    `lane_width_for` and therefore 1024 on a FAST CDNA column while
    `slice_offset`'s literal stayed 512: stage 1 then folded slices 0,2,4..
    into the first half and 1,3,5.. into a second half that stage 2 never
    reads, dropping half the histogram. One constant, in the wrong row.

    AND THE IDENTICAL PATH ALREADY HELD THE FIX. `lane_width_for` returns
    the pinned 32 under IDENTICAL on every column, so both families compile
    and their asserts pass on AMD today. The layout was never mode-dependent;
    only the constant feeding it was. `replication_lanes_for` is that
    constant made unconditional, and with it the two families build and
    dispatch on a 64-lane FAST column with no refusal anywhere.

    THE GOVERNING RULE THIS ROW GOT WRONG: that CatBoost never wrote a
    wide-wavefront layout is not a reason for us to refuse one. We did not
    have to write one either -- theirs is width-independent once the
    striping constant is read as logical, which is what the pointwise
    family (`pointwise_hist2_half_byte_template.mojo`, a hardcoded logical
    32 that has always compiled at any width) and the fused 8-bit kernel
    (`H8_LANE`, running on the MI325X since 2026-08-27) already did.

    WHAT DID NOT SURVIVE THE RETRACTION, and it is real: the turn-taking
    sync. `syncwarp` orders one hardware wave, and what it lowers to on a
    64-wide one cannot be established from this tree. That is
    `sub_byte_lane_sync_for`, a SEPARATE row, which takes `barrier()` on
    every column that is not exactly 32 lanes wide. The exclusion is gone;
    the caution moved to the row that owns it.

    WHAT IS NOT YET EVIDENCE. Nothing in this tree has ever EXECUTED a
    sub-byte block on 64-wide hardware in either mode: IDENTICAL compiles
    them on AMD, and every AMD fixture to date is 255-border float data
    whose grid contains one-byte blocks only. This row makes the code
    ELIGIBLE to run there. The verification that would settle it is named in
    DEVIATION 1947 and is OWED.
    """
    return False


def greedy_quantized_hist_for[column: Int, identical: Bool]() -> Bool:
    """NUMERIC row (DEVIATIONS 1911/1912): whether the NON-SYMMETRIC
    drivers' one-byte histogram build routes through the QUANTIZED
    SHARED-HISTOGRAM family (`kernel/hist_quantized_shared.mojo`) --
    per-round fixed-point gradient pairs packed one 64-bit word per row,
    ONE shared-memory Int32 histogram per thread block accumulated with
    threadgroup integer atomics, one global flush per block, dequantize
    at the flat-histogram bridge.

    THE DESIGN IS THE RECONS' BORROW 2/5, NOT OURS: XGBoost quantizes
    every (grad, hess) to fixed point once per round and keeps ONE
    histogram copy per block in shared memory with integer atomics
    (`quantiser.cuh`, `histogram.cu:70-233` -- recon_xgboost_gpu.md a);
    LightGBM's quantized path packs the discretized pair into one word
    per row and lands it with one block atomic
    (`cuda_histogram_constructor.cu:291-294` -- recon_lightgbm_cuda.md
    b2). CatBoost's warp-private float slices cost 128 B of shared
    memory PER THREAD regardless of bin count; the shared-copy design
    costs bytes PER BIN, amortized over the block.

    WHY THIS IS PORTABLE WHERE THE FLOAT DESIGN IS NOT: Metal has no
    threadgroup FLOAT atomics but full threadgroup INTEGER atomics
    (`column_has_threadgroup_int_atomics` -- true in every column,
    portable baseline included), which is exactly why the quantized
    integer histogram is the one shared-memory design every vendor can
    run from one source. The kernels carry no lane-width assumption --
    Int32 atomics, block barriers, uniform trip counts -- so 32-lane and
    64-lane columns compile the same file (the DEVIATION 1906/1910
    refusals do not apply to this family).

    A NUMERIC row: on the NVIDIA/AMD FAST columns the histogram
    arithmetic changes from CatBoost's order-nondeterministic float
    atomics to the SAME dithered fixed-point Int32 accumulation the
    Apple FAST arm (`HIST_SMEM_SHARED2_I32`) and the IDENTICAL column
    already run (`hist2_quantize` under `choose_scale`'s bound) -- which
    also makes the FAST histogram deterministic run to run, deleting
    the leaf-choice-cascade variance candidate recon_lightgbm_cuda.md
    names first. `IDENTICAL` returns False: the identical route must not
    move (DEVIATION 1906's precedent), and its byte-compare against main
    passes by construction. The variable-width columns (qualcomm, intel,
    spec-baseline) are NOT claimed.

    THE PRICE: a second one-byte code path, alive only under FAST on the
    named columns, and a per-level quantize pass over the rows being
    built (the stat planes are re-permuted every level, so the
    once-per-round quantize XGBoost enjoys is per-LEVEL here until
    DEVIATION 1902 stops permuting them). The A/B against the standing
    arms is the orchestrator's gate, and this row is where the routing
    flips if the number disagrees.
    """
    comptime if identical:
        return False
    # DEVIATION 2045, DIAGNOSTIC, OFF BY DEFAULT. The second candidate for
    # the question 2044 asks. ON under FAST, OFF under IDENTICAL, and the
    # docstring above names its price without ever measuring it: "a per-level
    # quantize pass over the rows being built ... the once-per-round quantize
    # XGBoost enjoys is per-LEVEL here". A pass per level is a real cost the
    # identical arm does not pay.
    #
    # Scoped to the NON-SYMMETRIC drivers, so it is EXPECTED INERT on
    # gbdt-symmetric and is the row to reach for on depthwise and lossguide.
    # Recording that here so an inert symmetric result reads as confirmation
    # of the scope rather than as refutation of the candidate.
    comptime if is_defined["MOJOLEARN_2045_FAST_NO_QUANT_HIST"]():
        return False
    return (
        column == COLUMN_APPLE
        or column == COLUMN_NVIDIA
        or column == COLUMN_AMD
        or column == COLUMN_AMD_RDNA
    )


def quantized_hist_group_features_for[column: Int]() -> Int:
    """SCHEDULING row (DEVIATION 1913): how many one-byte features one
    thread block's shared histogram covers in the quantized family.

    XGBoost packs features into groups whose total bins FILL the
    per-block shared-memory budget (`feature_groups.cu:29-45`,
    `histogram.cu:344-451` -- ~227 KB opt-in on H100, ~55 features per
    group at 255 bins); LightGBM does the same at
    `shared_hist_size_ / 2` (`cuda_row_data.cpp:179-291`). The point of
    the group is that every block covering G features reads the stat
    plane once for G features, so the number of full passes over the
    gradients is `ceil(F / G)` instead of `ceil(F / 4)`.

    THE BUDGET IS THE COLUMN'S, NOT NVIDIA'S: `column_shared_limit` --
    Apple/Adreno 32 KB, NVIDIA 48 KB (the opt-in carveout above it is
    deliberately not modeled, same stance as that row), AMD 64 KB. A
    group of G one-byte features costs `G * 256 bins * 2 stats * 4 B =
    G * 2 KB`, so:

        apple 32 KB -> 16 features    nvidia 48 KB -> 24
        amd / amd-rdna 64 KB -> 32    spec-baseline 16 KB -> 8

    Rounded DOWN to a multiple of 4 because the compressed index packs
    four one-byte features per UInt32 word and a group must own whole
    words; capped at 32 (the largest any declared column affords);
    floored at 4 (one word). Comptime, because what it sizes is a
    threadgroup allocation.
    """
    comptime limit = column_shared_limit(column)
    var g = limit // (256 * 2 * 4)
    g = (g // 4) * 4
    if g < 4:
        g = 4
    if g > 32:
        g = 32
    return g


def reorder_single_pass_for[column: Int, identical: Bool]() -> Bool:
    """SCHEDULING row (DEVIATION 1907): whether the leaf reorder's stable
    one-bit partition may take the SINGLE-PASS decoupled-lookback path
    (`gbdt/gpu_util/kernel/reorder_single_pass.mojo`) for a level whose
    leaf bound is above CatBoost's `FastSortSize()` == 500,000 rows.

    CatBoost's own dispatch (`split_points.cu:737-741`) hands leaves above
    that threshold to `cub::DeviceRadixSort` instead of its scan-plus-
    reorder path; our port had only the scan path, whose serial per-leaf
    phase grows linearly with the largest leaf and is the recorded cause
    of the FAST speed lane's remaining 5M-rung deficit. The single-pass
    formulation is CUB's own `DispatchScan` machinery (decoupled lookback,
    `single_pass_scan_operators.cuh`), the same machinery XGBoost's row
    partitioner runs (`row_partitioner.cuh:98-201`).

    WHY THIS IS A ROW AND NOT A DEFAULT: decoupled lookback spins on
    another block's published status word, which requires co-resident
    thread blocks to make INDEPENDENT FORWARD PROGRESS. CUDA and HIP
    provide that guarantee -- CUB and rocPRIM shipping the protocol is the
    vendors' own attestation -- so the NVIDIA and AMD columns take the
    path under FAST. Metal promises nothing of the shape (and a Metal
    kernel that spins is not interrupted, `random_gen.mojo:105`), so the
    APPLE column keeps the 3-launch block-scan partition byte for byte.
    `IDENTICAL` returns False on every column: the identical route must
    not move, exactly as DEVIATION 1906's row holds it. The variable-width
    columns (amd-rdna, qualcomm, intel, spec-baseline) are NOT claimed --
    nobody has stated their forward-progress terms here.

    A SCHEDULING row, not a numeric one, stated deliberately: the
    single-pass path produces the SAME PERMUTATION as the 3-launch path
    by construction (its scatter is `partition_place_kernel`'s arithmetic
    over the same exact integers -- see the file banner), so the row moves
    launches and passes, never a value. `DEVIATION 115a`
    (`core/scan_by_key.mojo`) declined the lookback across all columns for
    want of a stated guarantee; this row narrows that decline to the
    columns whose vendors decline it themselves.

    THE PRICE: a second code path for one kernel family, alive only on
    FAST NVIDIA/AMD above the threshold. The A/B is the orchestrator's 5M
    rung intra-leg, and this row is where the routing flips if the number
    disagrees.
    """
    comptime if identical:
        return False
    # DEVIATION 2042, OFF BY DEFAULT. The third candidate for why IDENTICAL
    # BEATS FAST on the gbdt lane, and the one this file already owed an A/B
    # for -- the paragraph above says "the A/B is the orchestrator's 5M rung
    # intra-leg" and it has never been run on AMD.
    #
    # Only the FAST build compiles this route. Its lookback walk is SERIAL
    # ON THREAD 0 (reorder_single_pass.mojo:120-124 says so and names the
    # warp-windowed version as the recorded next step) and it takes a global
    # atomic ticket for tile ordering, while IDENTICAL takes the three-launch
    # stable partition. A spin-wait kernel that only the fast arm runs is a
    # plausible reason the fast arm loses.
    #
    # Setting this makes FAST take IDENTICAL's route, so the arm is a pure
    # routing swap. It is BIT-NEUTRAL BY CONSTRUCTION -- the two routes are
    # gated against each other as an identity elsewhere in this tree -- and
    # the A/B harness will say so or refuse.
    comptime if is_defined["MOJOLEARN_2042_FAST_NO_LOOKBACK"]():
        return False
    # DEVIATION 2042: **AMD FLIPPED OFF 2026-09-03 ON A MEASUREMENT.** This
    # row's own paragraph above has owed an A/B since the route was added
    # ("the A/B is the orchestrator's 5M rung intra-leg, and this row is
    # where the routing flips if the number disagrees"). It was finally run
    # and the number disagrees.
    #
    # MI325X, 1,000,000 rows, two FAST builds alternated, 3 rounds, gbdt:
    #
    #     routed here    0.602159 s
    #     stable partition 0.471577 s      ratio 0.783, BITS EQUAL
    #
    # Both arms hash 45bfcba4070bf931, so this is a bit-identical win and
    # the default moves in the session that measured it. rf 0.998 and et
    # 1.001, both bits equal, so the flip costs the other two tree lanes
    # nothing.
    #
    # WHAT IT EXPLAINS. IDENTICAL was BEATING FAST on the gbdt lane on both
    # devices, which is not supposed to happen -- identity is not free, and
    # IDENTICAL there also pays a software Cephes expf and a
    # barrier-per-step reduction. The gap was this route: FAST off it runs
    # at 0.4716 s against IDENTICAL's 0.471-0.475 s. The tax was never
    # negative; the fast arm was carrying a spin-wait kernel only it
    # compiles, whose lookback walk is SERIAL ON THREAD 0
    # (reorder_single_pass.mojo:120-124 names the warp-windowed version as
    # the fix) plus a global atomic ticket for tile ordering.
    #
    # NVIDIA IS MEASURED AND STAYS ON THE ROUTE. The same A/B on an H100 at
    # the same shape: gbdt 1.003, rf 0.998, et 1.001, all bits equal. The
    # route costs nothing there, so there is nothing to flip, and the
    # default is unchanged rather than changed to match AMD.
    #
    # WHICH LEAVES NVIDIA'S OWN INVERSION UNEXPLAINED. IDENTICAL still beats
    # FAST on gbdt there, 0.938, and this route is not why. That is a
    # smaller effect than AMD's 0.731 and it is a SEPARATE open question;
    # the AMD answer must not be quoted as covering it.
    comptime if column == COLUMN_AMD:
        return False
    return column == COLUMN_NVIDIA


def ridx_only_splits_for[column: Int, identical: Bool]() -> Bool:
    """SCHEDULING row (DEVIATION 1902): whether the NON-SYMMETRIC driver's
    split moves only the row index, leaving the stat planes stationary for
    the life of the fit, with every stat reader gathering
    `stats[row_index[pos]]` instead of reading a permuted plane.

    XGBoost's `RowPartitioner` (`row_partitioner.cuh:112-201`, gather at
    `histogram.cu:186,212`) and LightGBM's CUDA partition
    (`cuda_data_partition.cu:288-334`, gather at
    `cuda_histogram_constructor.cu:53-55`) are both this design; CatBoost
    moves the stat columns (`split_points.cpp:64-136`), which is why this
    is a numbered deviation. The routed launcher is
    `launch_reorder_index_only`
    (`gbdt/methods/greedy_subsets_searcher/kernel/split_points_ridx.mojo`),
    the gathered readers are the hist gather kernels' `ridx_stats` arm and
    `compute_partition_stats_gather`
    (`gbdt/gpu_util/kernel/partition_stats_gather.mojo`).

    A SCHEDULING row, not a numeric one: the readers gather THE SAME BITS
    the old permuted plane held at the same loop position (the invariant on
    `split_points_ridx.mojo`'s banner -- a permutation of floats moves
    bytes and never re-rounds, and the two schedules permute with the same
    `gather_map`), in the same accumulation order under the same
    position-keyed dither. Same histograms, same partition stats, same
    model, byte for byte; what moves is `2 * ceil(stat_count / 8)` launches
    and `stat_count * 8` bytes/row of traffic per over-1024-row split,
    times the `max_leaves - 1` sequential splits a lossguide tree runs.

    EVERY GPU COLUMN TAKES IT UNDER FAST: the mechanism is plain global
    loads through an index the kernels ALREADY load for the compressed-
    index gather -- no spin, no lane-width assumption, no forward-progress
    term -- so unlike DEVIATION 1907's row there is no vendor guarantee to
    decline. What is NOT yet priced is the coalescing of the hist kernels'
    stat read turning indirect; the orchestrator's A/B (byte-compare of the
    FAST model pre/post routing, plus the 1M/2M lossguide rungs) is where
    the default dies if the gather costs more than the reorder saved.

    `IDENTICAL` returns False on every column: the identical route keeps
    the stat-moving path BYTE FOR BYTE, exactly as DEVIATIONS 1901/1903
    hold it, so the merge gate's compare against main passes by
    construction. The unnamed columns (qualcomm, intel, spec-baseline)
    return False only because nobody has run the A/B there; flipping them
    is one line here once a number exists.
    """
    comptime if identical:
        return False
    # DEVIATION 2044, DIAGNOSTIC, OFF BY DEFAULT. Why is IDENTICAL still
    # FASTER than FAST on gbdt-symmetric on an H100 after 2043 flipped
    # (0.9195 vs 0.9349, measured 2026-09-03)? This row is a candidate: it is
    # ON under FAST on every GPU column and OFF under IDENTICAL, and the
    # docstring above prices only one side of it. What it SAVES is launches
    # and stat traffic "times the `max_leaves - 1` sequential splits a
    # LOSSGUIDE tree runs" -- and a SYMMETRIC tree runs none of those, so on
    # this lane the saving may be near zero while the cost the docstring
    # itself calls UNPRICED, "the coalescing of the hist kernels\' stat read
    # turning indirect", is paid in full. Uncoalesced global loads are not
    # cheap on an SM.
    #
    # BIT-NEUTRAL BY CONSTRUCTION, which is what lets the plain A/B harness
    # rule on it: DEVIATION 1902\'s invariant is "Same histograms, same
    # partition stats, same model, byte for byte". If the hashes MOVE under
    # this define, that is a defect in 1902, not a speed result.
    comptime if is_defined["MOJOLEARN_2044_FAST_NO_RIDX_ONLY"]():
        return False
    return (
        column == COLUMN_APPLE
        or column == COLUMN_NVIDIA
        or column == COLUMN_AMD
        or column == COLUMN_AMD_RDNA
    )


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
    under `checks/fixed_point.mojo`'s bound: `fixed_scale` must come from
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

    # DEVIATION 2046, DIAGNOSTIC, OFF BY DEFAULT. The third candidate. On
    # NVIDIA, FAST keeps CatBoost's warp-private f32 slices while IDENTICAL
    # takes the shared-Int32 arm, where FOUR warps share one 8 KB slice
    # (`hist_2_one_byte_8bit.mojo`) instead of each holding its own. That is
    # an OCCUPANCY difference and not only a numeric one, and it runs the
    # same direction as the 2043 result: the integer accumulator keeps
    # buying cheaper schedules rather than costing them.
    #
    # THE HASHES WILL MOVE -- f32 accumulation becomes dithered fixed point --
    # so this is not a question for the bit-neutral A/B harness. It needs the
    # accuracy harness, on the gate 2043 established: faster AND no worse on
    # held-out AUC.
    comptime if is_defined["MOJOLEARN_2046_FAST_SHARED_I32"]():
        return HIST_SMEM_SHARED2_I32
    comptime if identical:
        return HIST_SMEM_SHARED2_I32  # the BIT_IDENTICAL column's value
    elif limit < CATBOOST_PRIVATE_BYTES:
        #: Apple (32 KB) took this arm by measurement. It is now stated as
        #: the BUDGET rule that produced that answer rather than as the
        #: vendor's name, which is what makes it extend: Adreno's 32 KB
        #: land here too, and NVIDIA's 48 KB (exactly their
        #: layout) and AMD's and Intel's 64 KB do not. **The resolved value
        #: for every founding column is unchanged by this rewrite** -- apple
        #: shared-Int32, nvidia and amd warp-private -- which is the only
        #: acceptable outcome for an edit to a NUMERIC row.
        return HIST_SMEM_SHARED2_I32
    else:
        return HIST_SMEM_WARP_PRIVATE_F32


# ---------------------------------------------------------------------------
# THE PARTITION-STATS CHUNK COUNT, and the misclassification it corrects
# ---------------------------------------------------------------------------

#: NUMERIC, and it spent this project's whole life classified as SCHEDULING.
#:
#: `partition_stats_chunks(sm_count, n_stats)` is CatBoost's
#: `CeilDivide(2 * TArchProps::SMCount(), statCount)`
#: (`update_part_props.cu:215`): the number of chunks a leaf's rows are split
#: into before `block.sum` reduces each one in FLOAT. **So the machine's core
#: count decides how a float sum is partitioned, and the partition of a float
#: sum decides its last bits.** Apple's 10 cores and an A100's 108 SMs give
#: different chunk counts, different partials, and therefore different
#: per-leaf stats -- which become the LEAF VALUES, which are the model.
#:
#: `hardware_matrix.gpu_cores_for` declares itself SCHEDULING: "the device
#: column always answers". True of every other reader it has and FALSE of
#: this one. `numerics.mojo` opens by warning about exactly this mistake --
#: "a block count is a summation order" -- and the mistake was in the tree
#: anyway, one indirection away, which is how it survived a matrix, a check
#: and two audits.
#:
#: Found 2026-08-21 chasing "will the identical column actually be identical
#: across GPUs". It is the largest reason the answer was no, and it is the
#: one hole that is pure pinning rather than a new kernel: the reduction is
#: already deterministic GIVEN a chunk count, so pinning the count closes it.
comptime PINNED_PARTITION_CHUNKS_SM = 32


def partition_chunks_sm_for[identical: Bool](device_sm: Int) -> Int:
    """The `sm_count` the partition-stats chunk formula is fed.

    Under `FAST`, the device's own core count: CatBoost's behaviour, and what
    fills the machine. Under `IDENTICAL`, a PINNED value, so the float
    partition has the same shape on every vendor.

    32 rather than any real device's count, deliberately: it is above Apple's
    10 and below a datacenter part's 108, so no column runs the pinned arm at
    its own number and nobody can mistake the pin for a measurement. The cost
    is a grid that under-fills a large GPU and over-fills a small one, on the
    opt-in arm only -- a chunk count, not a kernel, so measuring it is a
    parameter sweep and it has not been run.

    READERS (2026-08-22): no longer just the partition-stats chunk formula.
    `std_dev_blocks` (random_score_helper.mojo, DEVIATION 252), the greedy
    arm's `target_variance_blocks` (compute_scores.mojo, DEVIATION 353) and
    the histogram replication factor (DEVIATION 354) all pin their
    machine-derived counts through this one function, so the row-7 class
    has ONE pin to audit.
    """
    comptime if identical:
        return PINNED_PARTITION_CHUNKS_SM
    else:
        return device_sm


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

    comptime hard = column_max_block_size(column)
    comptime if smem_mode == HIST_SMEM_SHARED2_I32:
        comptime limit = column_shared_limit(column) // 64
        comptime by_smem = 512 if limit >= 512 else limit
        #: The vendor dispatch cap, for the reason in `block_size_for`.
        return by_smem if by_smem < hard else hard
    else:
        return block_size_for[K_HIST_2_ONE_BYTE, column]()


def pw_hist2_block_size_for[column: Int, fixed: Bool]() -> Int:
    """SCHEDULING row for the POINTWISE one-byte family's block, per route.

    BOTH ARMS RESOLVE TO CATBOOST'S OWN GEOMETRY (`block_size_for`: 32
    shared floats per thread, capped at their 384, 256 under Apple's
    32 KB), and the `fixed` parameter exists because the ROUTES ARE
    DIFFERENT PROGRAMS and their geometries must be free to differ -- the
    8-bit accumulator's atomic Int32 slices tolerate any number of warps
    wrapping onto them, the float accumulators' non-wrapping slice offsets
    do not -- so every consumer must say which program it is sizing.

    ============== MEASURED NEGATIVE (2026-08-21, M4) ==============
    The obvious occupancy purchase WAS TRIED AND DOES NOT PAY. Under the
    fixed route the block was doubled to 512 at 16 slots/thread -- same
    32 KB, same two 4096-slot slices, twice the resident threads, the
    exact shape of `hist2_block_size_for`'s measured 1.94x for the greedy
    family, and correctness-free here because the adds are associative
    Int32 atomics (every gate stayed green, fit bit-identical). ABAB in
    one window, eps500 400k x 500 @128, interleaved arms, settled half:
    baseline pointwise fit 1.181/1.193 s, doubled-block 1.183/1.247 s --
    INDISTINGUISHABLE. This kernel is not occupancy-bound at that shape:
    its deferred flush (one atomic per RUN of equal bins, not per point)
    keeps shared-atomic work low, so the wall the greedy probe measured
    is not the wall here, and doubling warps-per-slice doubles slice
    contention in exchange. The same lesson as the fused-kernel traffic
    model: a bound TRANSFERRED from another kernel is a hypothesis, not
    a diagnosis, until this kernel measures it. Do not re-raise the
    block without an ABAB showing a win at a shipped shape.
    ================================================================

    THE 8-BIT ACCUMULATOR NEEDS TWO 4096-SLOT SLICES: its `Reduce`
    stage 1 folds slice 0 onto slice 1 in place, so
    `pw_hist2_smem_floats_for` must return at least 8192. Every founding
    column provides that (Apple 256 x 32 = 8192; NVIDIA/AMD 384 x 32 =
    12288). A column whose budget cannot hold 32 KB cannot host this
    accumulator at any block size -- that column must not take the fixed
    route, and `PointHist8`'s constructor refuses it at compile time.
    """
    return block_size_for[K_POINTWISE_HIST_2, column]()


def pw_hist2_smem_floats_for[column: Int, fixed: Bool]() -> Int:
    """Companion to `pw_hist2_block_size_for`: the shared scratch, in
    4-byte slots. Their `HIST_SIZE = 32 * BLOCK_SIZE`
    (`pointwise_hist2_one_byte_templ.cuh:62`), both routes -- see the
    measured negative on the block row for why the fixed route's slots-
    per-thread did not stay halved."""
    return 32 * pw_hist2_block_size_for[column, fixed]()


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
# and is tracked in `archive/plans/UNWIRED.md`; a declared row that no kernel reads is not
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
comptime K_LIB_WEIGHTED_VERTEX_DEG = 115


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
#   dbscan/NOT_IMPLEMENTED.tsv:9
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
# Not wired. `archive/plans/UNWIRED.md` tracks it. The row below is what a kernel should
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

def lib_block_bounds_a_float_fold[kernel: Int]() -> Bool:
    """NUMERIC CLASSIFIER, and the correction of a label this file got wrong.

    `lib_block_size` and `lib_block_size_for` both open by calling
    themselves SCHEDULING. For most of these kernels that is right. For
    THREE of them it is exactly the mistake `numerics.mojo` opens by
    warning about -- *a block count is a summation order* -- because the
    block size is the WIDTH of a `block.sum` over floats:

    - `K_LIB_ROW_NORM`      `core/row_norms.mojo`, the sum of squares that
                            becomes every expanded-L2 distance.
    - `K_LIB_REDUCE_BY_KEY` `cluster/checks/reduce_by_key.mojo`'s
                            `sum_partials_kernel` / `finish_sum_kernel`,
                            which carry inertia AND the centroid SHIFT the
                            Lloyd loop converges on -- so a different width
                            can end the fit an iteration earlier and change
                            the MODEL, not just a reported cost.
    - `K_LIB_PLUS_PLUS`     `cluster/checks/plus_plus.mojo`: the
                            candidate-cost `block.sum` and the three-stage
                            float SCAN whose chunk count is
                            `ceildiv(n, PLUS_PLUS_TPB)`. The scan feeds
                            `binary_search_kernel`, so its rounding decides
                            WHICH SAMPLE k-means++ draws.
    - `K_LIB_COLUMN_STATS`  `core/column_stats.mojo`, ADDED 2026-08-23
                            (DEVIATION 510). Both `column_mean_kernel` and
                            `xty_kernel` fold Float32 at this width, and
                            both feed PCA and OLS: the mean is subtracted
                            from every row before the covariance, and
                            `A^T b` is the right-hand side the normal
                            equations are solved against. Missed by
                            DEVIATION 508, which enumerated the three rows
                            the unsupervised sections reach and stopped at
                            the section boundary -- the row was one
                            directory over in `core/`, reached by
                            `decomposition/`, and a row's LANE is not a
                            reason for it to be classified differently.

    - `K_LIB_WEIGHTED_VERTEX_DEG` `dbscan/impl/dbscan/vertexdeg/algo.mojo`,
                            ADDED 2026-09-01 with `sample_weight`. Both
                            `weighted_vertex_deg_dense_kernel` and
                            `weighted_vertex_deg_csr_kernel` stride their
                            per-thread partials by this number and then fold
                            them with `core/pinned_reduce.pinned_block_sum`
                            at this width, so it is simultaneously the fold
                            width AND the stride -- DEVIATION 524's shape
                            exactly, and the reason the pinned fold alone is
                            not enough. The value it produces is thresholded
                            against `min_pts` by `CorePoints::compute`
                            (`runner.cuh:301`), so one bit here is one
                            point's core status and from there one cluster
                            against two. Bit-inert today (128 in every
                            column), listed so that a landed vendor number
                            is safe rather than silent.

    Every other library row here is either integer (`K_LIB_EPS_NEIGHBORHOOD`
    and `K_LIB_ADJ_SCAN` fold `Int32`, which is associative; so do
    `K_LIB_SELECT_RADIX`'s histogram scan and `K_LIB_BALL_COVER_EPS`'s
    counter), carries no fold at all (`K_LIB_WEAK_CC`), folds with an
    ORDER-INDEPENDENT operator (`K_LIB_TRANSPOSE` moves data; the ball-cover
    scan's `block.max` is a max, which is associative AND commutative on
    floats away from NaN), or has its numeric geometry pinned separately
    (`PINNED_ACC_*`, `PINNED_KBLK`, `PINNED_VECLEN` for the two contraction
    rows; `K_LIB_GRAM_SPLITK` says so in its own row).

    `K_LIB_JACOBI_EIGH` IS LISTED (2026-08-23): `decomposition/checks/
    jacobi_eigh_device.mojo` folds two Float32 sums at that width and
    strides its per-thread partials by it, so under IDENTICAL the row
    resolves to the identity floor (32) on every vendor, and under FAST it
    may carry a vendor number -- AMD's is its 64-wide wavefront, because
    the FAST library fold cannot compile narrower there. See DEVIATION
    511 and the note in that file.

    **THE SENTENCE THAT USED TO END THAT PARAGRAPH -- "so the fix there is
    the fold, not the row" -- IS INCOMPLETE AND IS DELETED** (DEVIATION
    524, IDENTITY_PATHS row 31). Replacing the fold with
    `core/pinned_reduce.pinned_block_sum` removes the LANE-width
    dependence, which is what it was built to do. It does not remove the
    BLOCK-SIZE dependence: this number is simultaneously the fold's width
    AND the stride the per-thread partials are strided by, so two columns
    carrying two values here would compute two different multisets of
    partials and then fold each of them correctly. That is why the row is
    NOW IN THIS LIST: under IDENTICAL it is the floor's 32 on every vendor
    (gated from the consumer's side by `check_jacobi_fold_width_is_pinned`),
    and the one vendor number that landed (AMD FAST = 64, the wavefront)
    came with the listing, as this paragraph used to demand.

    BIT-INERT TODAY, AND THAT IS THE POINT. `lib_block_size` returns 128
    for all four in every column, so gating them moves nothing on any
    machine that exists. What the gate buys is the NEXT measurement: this
    table's own docstring invites a vendor number to land here "WITHOUT
    touching a kernel", and landing one on any of these three rows would
    have silently made an AMD fit disagree with a CUDA fit in the last
    bits of every distance. The gate makes that landing safe instead of
    silent.
    """
    return (
        kernel == K_LIB_ROW_NORM
        or kernel == K_LIB_REDUCE_BY_KEY
        or kernel == K_LIB_PLUS_PLUS
        or kernel == K_LIB_COLUMN_STATS
        or kernel == K_LIB_JACOBI_EIGH
        or kernel == K_LIB_WEIGHTED_VERTEX_DEG
    )


def lib_block_size_for[kernel: Int, column: Int]() -> Int:
    """SCHEDULING for most rows, NUMERIC for three of them.

    IDENTITY GATE (2026-08-23, DEVIATION 508), the same shape row 3's
    re-close put on `block_size_for` and for the same reason: under a
    `NUMERIC_IDENTICAL` build the column is replaced by
    `COLUMN_BIT_IDENTICAL` for every kernel
    `lib_block_bounds_a_float_fold` names, so the fold has ONE width on
    every vendor. Read `GLOBAL_NUMERIC_MODE` HERE rather than at the call
    sites: these are comptime accessors compiled into the kernels, and a
    gate at the report level is the defect row 3 was re-opened for.
    """
    comptime identical = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    comptime numeric_row = lib_block_bounds_a_float_fold[kernel]()
    comptime resolved = (
        COLUMN_BIT_IDENTICAL if identical and numeric_row else column
    )
    comptime lanes = column_lane_width(resolved)
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
        #: NUMERIC (listed above, 2026-08-23): under IDENTICAL `resolved`
        #: is the identity floor on every vendor and this is 32 everywhere,
        #: which is what the E1U/E2U cards certified. Under FAST it is the
        #: column's own number, and on AMD that is the WAVEFRONT: the
        #: FAST fold is the library `block_sum`, whose lane primitive
        #: asserts at compile time on a 32-thread block of a 64-wide
        #: wavefront -- which is why `bindings/build_estimators.sh` did not
        #: build on the MI325X at all under FAST (every leg through
        #: 2026-08-23 printed "expected on AMD"). FAST claims no
        #: cross-vendor bit, so AMD's FAST Jacobi may stride its partials
        #: by 64; IDENTICAL's may not, and does not.
        return 32 if lanes <= 32 else lanes
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


def knn_warpsort_select_for[column: Int, identical: Bool]() -> Bool:
    """SCHEDULING row (DEVIATION 1922): whether the k-NN TILED path's
    selector is the ported RAFT WARPSORT (`select_warpsort.mojo`,
    `warpsort_topk_block_kernel`) instead of the ported RAFT radix
    (`select_radix.mojo`) for `2 < k <= 256`.

    THEIR OWN DISPATCH IS THE ARGUMENT, NOT A PREFERENCE OF OURS. RAFT's
    `select_k` (`raft/matrix/detail/select_k-inl.cuh:38`) sends
    `2 < k <= 256` to the warpsort family and only `k > 256` to radix, so
    every k a user actually asks for takes warpsort on their stack.
    `neighbors/README.md` recorded that our tree ran radix ALONE across
    that whole range with the two never measured against each other; the
    2026-08-25/26 H100 legs put the knn lane 21.6-23.9x behind `cuml-gpu`
    at k = 10, which is squarely inside their warpsort band.

    WHY A ROW AND NOT A DEFAULT FLIP: `select_warpsort.mojo` pins
    `WARP_LANES = 32` because RAFT pins `WarpSize = 32` -- its subwarp
    sizes, array lengths (`Capacity / min(Capacity, 32)`) and shuffles are
    32-lane by construction, so on a 64-wide wavefront the bitonic
    subwarps would be mis-sized and the shuffles would cross them (the
    file's own docstring, item 9; the same geometry that makes
    `fused_l2_knn` refuse there). So:

      - NVIDIA under FAST: True. The column the deficit is measured on,
        the lane width RAFT wrote the kernel for, and their dispatch's
        own choice for this k band.
      - APPLE under FAST: False FOR NOW -- radix vs warpsort has never
        been measured on Metal (`neighbors/README.md`), the Mac board's
        knn rows were all taken on radix, and leaving the column
        byte-for-byte untouched keeps those numbers meaningful. This row
        is where Apple flips if a Mac window says warpsort wins there.
      - AMD (CDNA, 64-lane): False, EXCLUDED like DEVIATION 1910's
        sub-byte families: the kernel's geometry does not exist at
        64 lanes until a width-parameterized port is written. Radix has
        no warp primitive at all and stays correct there.
      - IDENTICAL: False on every column. The identical selector is
        `select_radix_identical` with the composite `(distance, index)`
        key (DEVIATIONS 500/501) and MUST NOT move; the FAISS queue
        compares distance only and has no tie class.

    A SCHEDULING row by the one question with a stated seam: both
    selectors return the k smallest DISTANCES of the same materialized
    tile, so the selected value multiset is the same; which EQUIDISTANT
    index survives and in which slot ties land may differ (radix's
    arrival-order atomics vs the bitonic network's positional merge).
    FAST already declares the tie set unpinned on this path (row 11), so
    the seam is inside FAST's existing declaration, exactly like a
    GEMM-order change.
    """
    comptime if identical:
        return False
    return column == COLUMN_NVIDIA


def knn_auto_follows_their_dispatch_for[column: Int, identical: Bool]() -> Bool:
    """SCHEDULING row (DEVIATION 1923): whether the k-NN AUTO arm follows
    cuVS's dispatch UNCONDITIONALLY -- `k <= 64` + row-major + L2 goes to
    `fusedL2Knn`, x-split included (`knn_brute_force.cuh:443`) -- instead
    of DEVIATION 36's shape test (fused only when `launchConfigGenerator`
    picks `grid_x == 1`, tiled when it would engage the x-split).

    DEVIATION 36's rule is sound ON THE COLUMN THAT MEASURED IT. Every
    number behind it is an Apple millisecond: the x-split's mutex merge
    was CATASTROPHIC on Metal (0.19x at 500-1,000 queries) because the
    acquire-load spin serializes against Metal's scheduler, and the
    grid_x == 1 gate is what keeps AUTO off that cliff. Nothing about
    that measurement transfers to CUDA: the mutex handoff is cuVS's OWN
    protocol on their OWN scheduler with the forward-progress guarantee
    they wrote it against, and cuVS ships it as the unconditional default
    for exactly this shape band. Meanwhile the 2026-08-25/26 H100 legs
    measured our knn lane 21.6-23.9x behind `cuml-gpu` at
    400,000 x 32, 4,000 queries, k = 10 -- a shape where the NVIDIA
    column's occupancy inputs make `launchConfigGenerator` return
    `grid_x > 1` (y_chunks = ceildiv(4000, Mblk) < numSMs *
    blocksPerSM), so AUTO was taking the TILED arm (materialize + radix)
    on the one vendor whose own library never takes it there.

      - NVIDIA under FAST: True -- their dispatch, restored exactly,
        x-split and mutex merge included. `run_knn`'s `ours-fused` /
        `ours-tiled` arms are the A/B that confirms or flips this row.
      - APPLE under FAST: False -- DEVIATION 36's measured rule stands
        byte for byte; the Metal x-split cliff is real and measured.
      - AMD: False and moot -- the fused arm refuses at 64 lanes
        (DEVIATION 512) and AUTO already pins tiled there.
      - IDENTICAL: False on every column -- DEVIATION 509 pins AUTO to
        the tiled arm everywhere, and this row must not reach it.

    SCHEDULING, not numeric, in the same sense as DEVIATION 36 itself:
    both arms match the host Float64 oracle slot for slot on the gated
    fixtures; what moves is which kernel runs and FAST's unpinned tie
    set (row 11), never a gated bit.
    """
    comptime if identical:
        return False
    return column == COLUMN_NVIDIA


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
