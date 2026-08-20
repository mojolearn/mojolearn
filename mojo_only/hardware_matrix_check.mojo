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
    gram_splitk_applies,
    gram_splitk_chunk_count,
)
from mojo_only.hardware_matrix import (
    gpu_cores_for,
    gram_splitk_is_target_arm,
    max_active_blocks_for,
    max_threads_per_core_for,
    smem_per_core_for,
    smem_statically_partitioned_for,
    threadgroup_limit_for,
)
from mojo_only.kernel_matrix import (
    COLUMN_AMD,
    COLUMN_APPLE,
    COLUMN_NVIDIA,
    TARGET_COLUMN,
    column_name,
)
from neighbors.ported.distance.detail.pairwise_distance_base import (
    TARGET_GPU_CORES,
    max_active_blocks_per_core,
)
from neighbors.ported.neighbors.detail.fused_l2_knn import fused_l2_knn_grid


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
    _pin(
        "gram_splitk_chunk_count reader vs table",
        gram_splitk_chunk_count(),
        gpu_cores_for[TARGET_COLUMN]()
        * max_active_blocks_for[TARGET_COLUMN](GRAM_TPB, GRAM_STAGE_FLOATS * 4)
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
        # A non-apple build must NOT run the hand-written Gram kernel.
        if gram_splitk_applies(32, 32, 4000000):
            raise Error(
                "check_hardware_matrix FAIL: a non-apple build routed the"
                " Gram shape to the split-K kernel; MAX's own matmul owns"
                " that regime off Apple"
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
    )
