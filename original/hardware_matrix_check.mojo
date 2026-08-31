# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The hardware matrix: columns exist, Apple is bit-for-bit the old
constants, and every reader agrees with the table.

Three duties, in the order the refactor could have broken them:

1. **The Apple column reproduces the previously hardcoded values EXACTLY.**
   `APPLE_M4_GPU_CORES = 10`, `APPLE_M4_MAX_THREADS_PER_CORE = 3072` and
   `METAL_MAX_THREADGROUP_MEM = 32768` were constants in
   `pairwise_distance_base.mojo` until 2026-08-19; the k-NN AUTO default
   and DEVIATION 36's measured tables depend on them bit-for-bit. (The
   grids themselves are additionally pinned by `check_launch_config_values`
   passing UNCHANGED, which is the other half of the same proof.)

2. **All three vendor columns EXIST and RESOLVE.** Every row here is host
   arithmetic, so the nvidia and amd columns are evaluated on this box --
   structural validation. It is NOT hardware validation: nothing has run on
   an A100 or an MI250X, the pins merely stop the documented values
   drifting silently.

3. **READERS AGREE WITH THE TABLE** (the `UNWIRED.md` rule: a row nothing
   reads is indistinguishable from one something reads, so every reader is
   diffed against the row it claims to read, and a disagreement RAISES):
   `max_active_blocks_per_core` (the launch computation's occupancy reader),
   `TARGET_GPU_CORES` (its core-count reader), `gram_splitk_chunk_count`
   (the summation split), `gram_splitk_applies` (the dispatch), and
   `fused_l2_knn_grid` (the number the k-NN AUTO default flips on).
"""

from core.gram_splitk import (
    GRAM_OVERSUBSCRIBE,
    GRAM_STAGE_FLOATS,
    GRAM_TPB,
    PINNED_GRAM_SPLITK_CHUNKS,
    gram_splitk_applies,
    gram_splitk_chunk_count,
)
from original.hardware_matrix import (
    gpu_cores_for,
    gram_splitk_is_target_arm,
    max_active_blocks_for,
    max_threads_per_core_for,
    smem_per_core_for,
    smem_statically_partitioned_for,
    threadgroup_limit_for,
)
from original.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from original.kernel_matrix import (
    COLUMN_AMD,
    COLUMN_APPLE,
    COLUMN_AMD_RDNA,
    COLUMN_BIT_IDENTICAL,
    COLUMN_COUNT,
    COLUMN_INTEL,
    COLUMN_NVIDIA,
    COLUMN_QUALCOMM,
    COLUMN_SPEC_BASELINE,
    HIST_SMEM_SHARED2_I32,
    HIST_SMEM_WARP_PRIVATE_F32,
    IDENTITY_FLOOR_BLOCK,
    IDENTITY_FLOOR_LANES,
    IDENTITY_FLOOR_SHARED_BYTES,
    K_HIST_2_ONE_BYTE,
    K_LIB_SELECT_WARPSORT,
    PINNED_REPLICATION_LANES,
    TARGET_COLUMN,
    column_max_block_size,
    column_meets_identity_floor,
    column_name,
    column_shared_limit,
    deterministic_flush_for,
    hist2_block_size_for,
    hist_smem_mode_for,
    identity_refusal_reason,
    lib_block_size_for,
    lib_lane_width_for,
)
from neighbors.derived.distance.detail.pairwise_distance_base import (
    TARGET_GPU_CORES,
    max_active_blocks_per_core,
)
from neighbors.derived.neighbors.detail.fused_l2_knn import fused_l2_knn_grid


def _pin(name: String, got: Int, want: Int) raises:
    if got != want:
        raise Error(
            "check_hardware_matrix FAIL: "
            + name
            + " = "
            + String(got)
            + ", want "
            + String(want)
        )


def check_hardware_matrix() raises:
    # ---- 1. the Apple column, bit-for-bit the pre-keying constants ------
    _pin("apple gpu_cores", gpu_cores_for[COLUMN_APPLE](), 10)
    _pin(
        "apple max_threads_per_core",
        max_threads_per_core_for[COLUMN_APPLE](),
        3072,
    )
    _pin(
        "apple threadgroup_limit", threadgroup_limit_for[COLUMN_APPLE](), 32768
    )
    if smem_statically_partitioned_for[COLUMN_APPLE]():
        raise Error(
            "check_hardware_matrix FAIL: the Apple column claims a static"
            " shared-memory partition; family 9 dynamically caches"
            " threadgroup memory (SCALING_2026-08-19.md) and the old"
            " computation never divided by it"
        )
    # The occupancy the 2026-08-19 grids ran on: 3072 // 256 = 12 at the
    # fused kernel's 18,432-byte footprint, and minGridSize = 120.
    _pin(
        "apple max_active_blocks(256, 18432)",
        max_active_blocks_for[COLUMN_APPLE](256, 18432),
        12,
    )

    # ---- 2. the nvidia and amd columns exist and resolve (host-only; ----
    # ---- structural, NOT hardware validation) ---------------------------
    _pin("nvidia gpu_cores (A100 SMs)", gpu_cores_for[COLUMN_NVIDIA](), 108)
    _pin(
        "nvidia max_threads_per_core",
        max_threads_per_core_for[COLUMN_NVIDIA](),
        2048,
    )
    _pin(
        "nvidia threadgroup_limit",
        threadgroup_limit_for[COLUMN_NVIDIA](),
        48 * 1024,
    )
    _pin(
        "nvidia smem_per_core", smem_per_core_for[COLUMN_NVIDIA](), 164 * 1024
    )
    _pin("amd gpu_cores (MI250X GCD CUs)", gpu_cores_for[COLUMN_AMD](), 110)
    _pin(
        "amd max_threads_per_core", max_threads_per_core_for[COLUMN_AMD](), 2048
    )
    _pin("amd threadgroup_limit", threadgroup_limit_for[COLUMN_AMD](), 64 * 1024)
    _pin("amd smem_per_core", smem_per_core_for[COLUMN_AMD](), 64 * 1024)
    if not smem_statically_partitioned_for[COLUMN_NVIDIA]():
        raise Error(
            "check_hardware_matrix FAIL: nvidia must divide occupancy by its"
            " static shared-memory partition and does not"
        )
    if not smem_statically_partitioned_for[COLUMN_AMD]():
        raise Error(
            "check_hardware_matrix FAIL: amd must divide occupancy by its"
            " static LDS partition and does not"
        )
    # The divisor DIFFERS by column at the fused kernel's footprint: the
    # thread term binds on nvidia (2048//256 = 8 < 164K//18432 = 9), the
    # LDS term binds on amd (64K//18432 = 3 < 8). If either pin moves, the
    # occupancy model changed and every downstream grid changed with it.
    _pin(
        "nvidia max_active_blocks(256, 18432)",
        max_active_blocks_for[COLUMN_NVIDIA](256, 18432),
        8,
    )
    _pin(
        "amd max_active_blocks(256, 18432)",
        max_active_blocks_for[COLUMN_AMD](256, 18432),
        3,
    )
    # The per-block wall reads the COLUMN's cap, not Metal's: 33 KB fits
    # nvidia and amd and must raise only on apple.
    var walled_apple = False
    try:
        _ = max_active_blocks_for[COLUMN_APPLE](256, 33 * 1024)
    except:
        walled_apple = True
    if not walled_apple:
        raise Error(
            "check_hardware_matrix FAIL: a 33 KB request did not raise on"
            " the apple column"
        )
    if max_active_blocks_for[COLUMN_NVIDIA](256, 33 * 1024) != 4:
        raise Error(
            "check_hardware_matrix FAIL: 33 KB on nvidia must pass the 48 KB"
            " wall and resolve 164K//33K = 4 blocks"
        )
    var walled_amd = False
    try:
        _ = max_active_blocks_for[COLUMN_AMD](256, 65 * 1024)
    except:
        walled_amd = True
    if not walled_amd:
        raise Error(
            "check_hardware_matrix FAIL: a 65 KB request did not raise on"
            " the amd column"
        )

    # ---- 3. readers agree with the table --------------------------------
    _pin(
        "TARGET_GPU_CORES reader vs table",
        TARGET_GPU_CORES,
        gpu_cores_for[TARGET_COLUMN](),
    )
    _pin(
        "max_active_blocks_per_core reader vs table",
        max_active_blocks_per_core(256, 18432),
        max_active_blocks_for[TARGET_COLUMN](256, 18432),
    )
    # THE CHUNK COUNT IS A NUMERIC ROW UNDER IDENTICAL, NOT A TABLE READER
    # (DEVIATION 520, IDENTITY_PATHS row 27). Under FAST it is still the
    # machine's own number and this file still pins it against the table.
    # Under IDENTICAL it is `PINNED_GRAM_SPLITK_CHUNKS` on every column,
    # because the count IS the k-axis summation split and a summation split
    # read from the hardware is a different model per vendor. Asserting the
    # table value in both modes would demand the identity build read the
    # machine, which is the defect.
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        _pin(
            "gram_splitk_chunk_count pinned (IDENTICAL)",
            gram_splitk_chunk_count(),
            PINNED_GRAM_SPLITK_CHUNKS,
        )
    else:
        _pin(
            "gram_splitk_chunk_count reader vs table",
            gram_splitk_chunk_count(),
            gpu_cores_for[TARGET_COLUMN]()
            * max_active_blocks_for[TARGET_COLUMN](
                GRAM_TPB, GRAM_STAGE_FLOATS * 4
            )
            * GRAM_OVERSUBSCRIBE,
        )
    # The dispatch row: split-K is the apple arm and ONLY the apple arm.
    if not gram_splitk_is_target_arm[COLUMN_APPLE]():
        raise Error(
            "check_hardware_matrix FAIL: split-K must be the apple column's"
            " Gram arm (MAX has no split-K on Apple, LANE_gram-splitk)"
        )
    if gram_splitk_is_target_arm[COLUMN_NVIDIA]():
        raise Error(
            "check_hardware_matrix FAIL: nvidia must hand the Gram shape"
            " back to MAX's matmul (its split-K gate is open there)"
        )
    if gram_splitk_is_target_arm[COLUMN_AMD]():
        raise Error(
            "check_hardware_matrix FAIL: amd must hand the Gram shape back"
            " to MAX's matmul (its split-K gate is open there)"
        )

    # Apple-build pins: the exact numbers the 2026-08-19 measured tables
    # and the shipped defaults stand on.
    comptime if TARGET_COLUMN == COLUMN_APPLE:
        # 240 is the APPLE MACHINE's number and belongs to the FAST arm.
        # Under IDENTICAL the same call returns the pinned constant on every
        # column including this one -- see DEVIATION 520 above.
        comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
            _pin("apple-build gram chunk grid", gram_splitk_chunk_count(), 240)
        var g = fused_l2_knn_grid(2000, 200000)
        if g[0] != 1 or g[1] != 120:
            raise Error(
                "check_hardware_matrix FAIL: fused_l2_knn_grid(2000,"
                " 200000) = ("
                + String(g[0])
                + ", "
                + String(g[1])
                + "), want (1, 120) -- the k-NN AUTO default reads this"
            )
        if not gram_splitk_applies(32, 32, 4000000):
            raise Error(
                "check_hardware_matrix FAIL: the apple build must route the"
                " 32x32x4M Gram to split-K and does not"
            )
    else:
        # A non-apple FAST build must NOT run the hand-written Gram kernel.
        # Under IDENTICAL the answer is the OPPOSITE on every column:
        # DEVIATION 521 routes the Gram shape to split-K everywhere, because
        # the vendor matmul is a closed k-split (and TF32 on NVIDIA, row 33).
        # Until 2026-08-23 this branch asserted the FAST answer in both
        # modes and phase 0 of every NVIDIA/AMD E1 leg printed a FINDING
        # for what was DEVIATION 521 doing its job.
        comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
            if gram_splitk_applies(32, 32, 4000000):
                raise Error(
                    "check_hardware_matrix FAIL: a non-apple build routed"
                    " the Gram shape to the split-K kernel; MAX's own"
                    " matmul owns that regime off Apple"
                )
        else:
            if not gram_splitk_applies(32, 32, 4000000):
                raise Error(
                    "check_hardware_matrix FAIL: an IDENTICAL build on a"
                    " non-apple column handed the 32x32x4M Gram to the"
                    " vendor matmul; DEVIATION 521 pins split-K on every"
                    " column"
                )

    # ---- 5. THE IDENTITY FLOOR, and the regression it guards -----------
    #
    # Three vendor columns were added on 2026-08-21 without a target to
    # build for, and `hist_smem_mode_for` was rewritten from "is this the
    # apple column" to "is this column's budget under CatBoost's layout" so
    # that it would extend to them. **That is an edit to a NUMERIC row, so
    # the only acceptable outcome is that every founding column resolves to
    # exactly what it resolved to before.** These pins are that proof, and
    # they are the reason the rewrite is allowed to stand.
    _pin(
        "apple hist2 smem mode (shared Int32)",
        hist_smem_mode_for[COLUMN_APPLE, False](),
        HIST_SMEM_SHARED2_I32,
    )
    _pin(
        "nvidia hist2 smem mode (CatBoost warp-private f32)",
        hist_smem_mode_for[COLUMN_NVIDIA, False](),
        HIST_SMEM_WARP_PRIVATE_F32,
    )
    _pin(
        "amd hist2 smem mode (CatBoost warp-private f32)",
        hist_smem_mode_for[COLUMN_AMD, False](),
        HIST_SMEM_WARP_PRIVATE_F32,
    )
    _pin(
        "identical hist2 smem mode (shared Int32)",
        hist_smem_mode_for[COLUMN_BIT_IDENTICAL, True](),
        HIST_SMEM_SHARED2_I32,
    )
    _pin(
        "apple hist2 block (512 fills Metal's 32 KB)",
        hist2_block_size_for[COLUMN_APPLE, HIST_SMEM_SHARED2_I32](),
        512,
    )
    # MODE-AWARE since 2026-08-22 (found by the E1 Mac bootstrap): under an
    # IDENTICAL build `block_size_for`'s identity gate caps the one-byte
    # family's budget at the 32 KB floor, so NVIDIA's warp-private hist2
    # block is 256 there BY DESIGN (one geometry on every vendor), and 384
    # -- CatBoost's own -- under FAST. Asserting 384 unconditionally made
    # this check fail on exactly the build the gate exists for.
    comptime _nvidia_hist2_want = 256 if (
        GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    ) else 384
    _pin(
        "nvidia hist2 block (CatBoost's 384 FAST / floor 256 IDENTICAL)",
        hist2_block_size_for[COLUMN_NVIDIA, HIST_SMEM_WARP_PRIVATE_F32](),
        _nvidia_hist2_want,
    )

    # THE SPLIT THAT CAUSED THIS COLUMN. One `amd` column resolved a
    # 512-thread warpsort block for RDNA parts whose lane group is 32; the
    # correct answer is 256. Scheduling, so no bit moved, but wrong on
    # hardware people train on.
    _pin("amd (CDNA) lane width", lib_lane_width_for[COLUMN_AMD](), 64)
    _pin("amd-rdna lane width", lib_lane_width_for[COLUMN_AMD_RDNA](), 32)
    _pin(
        "amd (CDNA) warpsort block",
        lib_block_size_for[K_LIB_SELECT_WARPSORT, COLUMN_AMD](),
        512,
    )
    _pin(
        "amd-rdna warpsort block (8 lane-groups of 32)",
        lib_block_size_for[K_LIB_SELECT_WARPSORT, COLUMN_AMD_RDNA](),
        256,
    )

    # The floor is FROZEN. These are not "the current values", they are the
    # guarantee, and an edit to one of them is a profile bump with a
    # migration -- not a patch that happens to fail a test.
    _pin("identity floor shared bytes", IDENTITY_FLOOR_SHARED_BYTES, 32768)
    _pin("identity floor lanes", IDENTITY_FLOOR_LANES, 32)
    _pin("identity floor block", IDENTITY_FLOOR_BLOCK, 512)
    _pin(
        "the safe column reads the FROZEN floor, not a min() over vendors",
        column_shared_limit(COLUMN_BIT_IDENTICAL),
        IDENTITY_FLOOR_SHARED_BYTES,
    )
    # Two names for one guarantee is how a guarantee drifts.
    _pin(
        "PINNED_REPLICATION_LANES == IDENTITY_FLOOR_LANES",
        PINNED_REPLICATION_LANES,
        IDENTITY_FLOOR_LANES,
    )

    # Every column answers every row: a new column must not fall through to
    # another vendor's value by accident, which is exactly how the machine
    # rows silently returned Apple's 10 cores for NVIDIA until 2026-08-19.
    for c in range(COLUMN_COUNT):
        if column_name(c) == String("unknown"):
            raise Error(
                "check_hardware_matrix FAIL: column "
                + String(c)
                + " is inside COLUMN_COUNT and has no name; every row in"
                " kernel_matrix.mojo must answer for it"
            )
        if column_shared_limit(c) <= 0 or column_max_block_size(c) <= 0:
            raise Error(
                "check_hardware_matrix FAIL: column "
                + column_name(c)
                + " has no declared shared-memory or block-size minimum"
            )
        # The verdict and the reason must agree. A column refused with no
        # reason is a support ticket; a column admitted WITH a reason is a
        # bug in the floor.
        var refused = identity_refusal_reason(c).byte_length() > 0
        if refused == column_meets_identity_floor(c):
            raise Error(
                "check_hardware_matrix FAIL: column "
                + column_name(c)
                + " disagrees with itself about identity admission"
            )

    # Today's finding, pinned so that a change to it is deliberate: every
    # declared VENDOR meets the floor. Adreno and Mali advertise the same
    # 32 KB Apple does, and Intel more, so the design was already floored by
    # the most constrained mainstream GPU memory hierarchy.
    #
    # The PORTABLE BASELINE does not, and must not: it is the specifications'
    # guaranteed minimum (16 KB, 128 invocations), which is half our floor's
    # memory and a quarter of its block. A day when it passes is a day
    # somebody lowered the floor.
    for c in range(COLUMN_COUNT):
        if c == COLUMN_SPEC_BASELINE:
            continue
        if not column_meets_identity_floor(c):
            raise Error(
                "check_hardware_matrix FAIL: "
                + identity_refusal_reason(c)
                + " -- if this is a NEW vendor that genuinely cannot meet"
                " the floor, it belongs in FAST and this loop needs an"
                " explicit exception naming it, NOT a lower floor"
            )
    if column_meets_identity_floor(COLUMN_SPEC_BASELINE):
        raise Error(
            "check_hardware_matrix FAIL: the portable baseline (16 KB, 128"
            " invocations) now MEETS the identity floor, which can only mean"
            " the floor was lowered. That is a profile bump: it changes"
            " every model produced under the old profile, and it does not"
            " happen as a side effect of a table edit"
        )
    # REACH, not output: the refusal branch has to have executed. Before the
    # baseline column existed every member passed, so this guard had never
    # once run the code that refuses -- an unreached branch is an untested
    # one, and this tree's rule is that reach is proved per branch.
    if identity_refusal_reason(COLUMN_SPEC_BASELINE).byte_length() == 0:
        raise Error(
            "check_hardware_matrix FAIL: the baseline column is refused and"
            " gives no reason; a refusal a user cannot act on is a support"
            " thread, not a guard"
        )

    # The vendor-forced flush: `qualcomm` and the baseline have no core
    # float atomic, so they take fixed point in BOTH modes. This is the
    # row that was computed in two places and agreed only by luck until
    # 2026-08-21; pinned here so the two expressions cannot drift again.
    if not deterministic_flush_for[COLUMN_QUALCOMM, False]():
        raise Error(
            "check_hardware_matrix FAIL: a qualcomm FAST build must take the"
            " fixed-point flush -- float atomic add is an optional extension"
            " there and the kernel would emit an instruction the device may"
            " not have"
        )
    if deterministic_flush_for[COLUMN_AMD_RDNA, False]():
        raise Error(
            "check_hardware_matrix FAIL: an amd-rdna FAST build must stay on"
            " CatBoost's float atomic; RDNA has them"
        )
    if deterministic_flush_for[COLUMN_APPLE, False]():
        raise Error(
            "check_hardware_matrix FAIL: an apple FAST build must stay on"
            " CatBoost's float atomic; that is the shipped default and the"
            " arm every measured number was taken on"
        )
    if deterministic_flush_for[COLUMN_NVIDIA, False]():
        raise Error(
            "check_hardware_matrix FAIL: an nvidia FAST build must stay on"
            " CatBoost's float atomic"
        )

    print(
        "check_hardware_matrix OK: apple column = the old constants"
        " bit-for-bit (10 cores, 3072 threads/core, 32 KB wall, occupancy"
        " 12); nvidia resolves 108/2048/48KB wall/164K partition ->"
        " 8 blocks; amd resolves 110/2048/64K partition -> 3 blocks (LDS"
        " term binding); readers (TARGET_GPU_CORES,"
        " max_active_blocks_per_core, gram chunk count = "
        + String(gram_splitk_chunk_count())
        + ", gram dispatch, fused_l2_knn_grid) all agree with the table;"
        " build column "
        + column_name(TARGET_COLUMN)
        + "; identity floor frozen at profile 1 (32 KB / 32 lanes / block"
        " 512): all "
        + String(COLUMN_COUNT - 2)
        + " vendor columns meet it, the portable baseline is REFUSED, apple/nvidia/amd smem modes unchanged"
        " by the budget rewrite"
    )


def main() raises:
    # STANDALONE DRIVER. `decomposition/pca_main.mojo` and
    # `neighbors/knn_main.mojo` both call this first; neither is a
    # registered task, so until now the only way to run it was to run a
    # whole other section's suite.
    check_hardware_matrix()
