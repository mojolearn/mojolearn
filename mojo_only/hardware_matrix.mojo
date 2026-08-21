"""Per-target HARDWARE rows: what the machine is, keyed like the kernel matrix.

`mojo_only/kernel_matrix.mojo` declares every KERNEL knob per vendor column.
These are the rows underneath those: what the device itself is -- core count,
thread slots per core, shared-memory partitioning -- the numbers CUDA
programs never write down because `cudaDevAttrMultiProcessorCount` and
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` answer them at RUNTIME
(cuVS `pairwise_distance_base.cuh:300-307`). Metal exposes neither query
through Mojo, so this repo pins them at comptime.

UNTIL 2026-08-19 IT PINNED APPLE'S NUMBERS UNCONDITIONALLY. Three scheduling
decisions read `APPLE_M4_GPU_CORES = 10` and friends on every target: the
pairwise-distance launch computation, the split-K Gram chunk count and
dispatch predicate, and (through the first) the k-NN AUTO default. On an
A100 that is 10 cores where there are 108 SMs -- the same shape of bug as
MAX's Apple matmul arm starving our Gram product, pointed the other way.
Correctness was never affected (every row here is SCHEDULING: it decides how
much of the machine a launch asks for, never what is added to what), but a
supported target would have been starved or mis-dispatched.

THE COLUMNS ARE `kernel_matrix`'s, imported, not restated. This file is a
sibling rather than a section of that file only because the kernel matrix is
kernel-shaped (block sizes, replication, reduction widths) and these rows are
machine-shaped; the discipline is identical: a reader says
`row_for[TARGET_COLUMN]()` and NOTHING ELSE, and `mojo_only/
hardware_matrix_check.mojo` RAISES if any reader disagrees with the table.

VALIDATION HONESTY, which is this repo's stated posture
--------------------------------------------------------
- APPLE is the VALIDATED column: one M4, and the values below are bit-for-bit
  the constants the 2026-08-19 measurement rounds ran on. The k-NN AUTO
  default and DEVIATION 36's tables depend on them; they must not drift.
- NVIDIA and AMD are SUPPORTED-NOT-VALIDATED: honest values transcribed from
  the vendors' architecture documents (cited per row), never measured by this
  repo, and pinned to ONE representative device each -- NVIDIA's A100
  (GA100, compute capability 8.0), the device cuVS's own tuning targets, and
  AMD's MI250X (CDNA2, one GCD), ROCm's flagship of the same generation.
  Upstream queries these at runtime and is right on every device; a pinned
  column is right on the pinned device and merely reasonable elsewhere.
  When Mojo exposes the device queries, these rows become the fallback and
  the queries become the truth, in that order.
- QUALCOMM, INTEL and ARM are DECLARED-NOT-BUILDABLE: Mojo emits no code for
  them today (`kernel_matrix.column_is_buildable`). Their KERNEL rows in
  `kernel_matrix.mojo` are documented vendor minimums with a citation each,
  because those decide an admission question that has to be settled in
  advance. Their MACHINE rows here are mostly conservative PLACEHOLDERS and
  are labelled individually, because "how many cores does a Qualcomm GPU
  have" has no single answer worth pinning and no bit depends on it.

WHAT IS DELIBERATELY NOT MODELED (all three columns)
-----------------------------------------------------
CUDA's occupancy is a min over MORE terms than thread slots and shared
memory: registers per SM and a max-resident-blocks-per-SM cap (32 at CC
8.0). Neither has a Metal-side query, neither is modeled here, and at the
block sizes this repo launches (256 threads) the thread-slot term binds
first on every column (8 or 12 blocks, under any documented cap), so the
omission cannot change a grid today. Stated so that a future 64-thread
launch does not trust this table past what it knows.
"""

from mojo_only.kernel_matrix import (
    COLUMN_AMD,
    COLUMN_APPLE,
    COLUMN_ARM,
    COLUMN_BIT_IDENTICAL,
    COLUMN_INTEL,
    COLUMN_NVIDIA,
    COLUMN_QUALCOMM,
    column_name,
    column_shared_limit,
)


def gpu_cores_for[column: Int]() -> Int:
    """SCHEDULING. Their `numSMs` (`cudaDevAttrMultiProcessorCount`,
    `pairwise_distance_base.cuh:300-301`): GPU cores on Apple, SMs on
    NVIDIA, CUs on AMD -- the unit a resident block occupies one of.

    - apple: 10, the base M4's GPU core count. VALIDATED (this box).
    - nvidia: 108 SMs, A100 (NVIDIA A100 Tensor Core GPU Architecture
      whitepaper, GA100 at 108 of 128 SMs enabled). UNVALIDATED.
    - amd: 110 CUs per MI250X GCD (AMD CDNA2 whitepaper). UNVALIDATED.
    - qualcomm, intel, arm: **CONSERVATIVE PLACEHOLDERS, NOT TRANSCRIPTIONS,
      and they are marked as such at the call site.** These families span an
      enormous range -- an Adreno in a phone and an Adreno X1 in a laptop are
      both "qualcomm" -- and picking a representative part the way the NVIDIA
      and AMD rows do would state a precision this table does not have. Every
      one of them is deliberately LOW, so a grid computed from it under-asks
      for the machine (slow, correct) rather than over-asks (a launch that
      fails on a device nobody here can test). They are SCHEDULING rows, so
      no answer they give can move a bit; replace them with the device query
      at bring-up rather than with a better guess.

    SCHEDULING rows have no bit-identical value to take -- the device column
    always answers -- but the column resolves to the intersection (the
    minimum) anyway, mirroring `column_shared_limit`, so a misdirected read
    is conservative rather than wrong.
    """
    if column == COLUMN_NVIDIA:
        return 108
    if column == COLUMN_AMD:
        return 110
    if column == COLUMN_QUALCOMM:
        return 6  # PLACEHOLDER, see below
    if column == COLUMN_INTEL:
        return 8  # PLACEHOLDER, see below
    if column == COLUMN_ARM:
        return 4  # PLACEHOLDER, see below
    return 10  # apple, and the bit-identical intersection


def max_threads_per_core_for[column: Int]() -> Int:
    """SCHEDULING. Resident thread slots per core: the one occupancy divisor
    every column can express.

    - apple: 3072 (96 resident SIMD-groups of 32 lanes on M-series cores).
      VALIDATED: the 2026-08-19 grids were computed from it.
    - nvidia: 2048 max resident threads per SM at compute capability 8.0
      (CUDA C++ Programming Guide, table "Technical Specifications per
      Compute Capability"). UNVALIDATED.
    - amd: 2048 per CU on CDNA2 (8 waves per SIMD x 4 SIMDs x 64 lanes;
      AMD CDNA2 ISA reference). UNVALIDATED.
    """
    if column == COLUMN_APPLE:
        return 3072
    if column == COLUMN_QUALCOMM or column == COLUMN_INTEL:
        return 1024  # conservative placeholder, see `gpu_cores_for`
    if column == COLUMN_ARM:
        return 512  # conservative placeholder, see `gpu_cores_for`
    return 2048  # nvidia, amd, and the bit-identical intersection


def threadgroup_limit_for[column: Int]() -> Int:
    """SCHEDULING as used here (a launch-validity wall). Per-BLOCK shared /
    threadgroup / LDS cap. Delegates to `kernel_matrix.column_shared_limit`
    -- 32 / 48 / 64 KB -- so exactly one place in this repository knows the
    per-block budgets. NVIDIA's opt-in carveout above 48 KB
    (`cudaFuncAttributeMaxDynamicSharedMemorySize`) is not modeled.
    """
    return column_shared_limit(column)


def smem_statically_partitioned_for[column: Int]() -> Bool:
    """SCHEDULING. Whether shared memory is a static per-core partition that
    DIVIDES occupancy, or a dynamically cached pool that only walls a launch.

    - apple: False. On Apple GPU family 9 (M3/M4) threadgroup memory is
      dynamically cached rather than statically partitioned per core; the
      measured query sweep in `bench/results/SCALING_2026-08-19.md`
      (deficit shrinking monotonically from 0.66x at ~32 blocks to 0.93x at
      ~2,000 blocks with 18.5 KB of threadgroup memory per block throughout)
      is only possible if many such blocks share a core -- a static 32 KB
      partition would cap them at one. So on Apple, `smem_bytes` is a
      validity wall and never a divisor.
    - nvidia, amd: True. Shared memory / LDS is a per-SM (per-CU) resource
      that resident blocks split, and `cudaOccupancyMaxActiveBlocksPer-
      Multiprocessor` includes exactly this term. UNVALIDATED here.

    - qualcomm, intel: True. Both partition a per-core local-memory pool
      between resident work-groups; Intel's guide computes occupancy from
      exactly that division. UNVALIDATED.
    - arm: **False, and for a different reason than Apple's.** Mali has no
      dedicated compute scratchpad at all -- Arm's best-practices guide says
      the shared memory "is system RAM that is backed up by the load-store
      cache" -- so there is no per-core pool for resident groups to divide.
      It is not a divisor because it is not a partition, which is the same
      answer Apple gives from the opposite premise. See
      `kernel_matrix.column_has_dedicated_shared_memory`.

    The bit-identical intersection is True (the restrictive reading).
    """
    return column != COLUMN_APPLE and column != COLUMN_ARM


def smem_per_core_for[column: Int]() -> Int:
    """SCHEDULING. The per-CORE shared-memory pool the occupancy divisor
    uses where `smem_statically_partitioned_for` is True.

    - apple: 32768. Not a partition (see above); recorded as the
      per-threadgroup cap so a misdirected read is conservative.
    - nvidia: 167936 (164 KB configurable shared memory per A100 SM, of the
      192 KB unified L1/shared array; A100 whitepaper). UNVALIDATED.
    - amd: 65536 (64 KB LDS per CDNA2 CU; AMD CDNA2 whitepaper).
      UNVALIDATED.
    """
    if column == COLUMN_NVIDIA:
        return 164 * 1024
    if column == COLUMN_AMD:
        return 64 * 1024
    if column == COLUMN_INTEL:
        #: 128 KB of SLM per Xe-core, which Intel's own occupancy example
        #: splits as 64 + 64 between two resident work-groups (oneAPI GPU
        #: optimization guide, "Shared Local Memory"). The one row on the
        #: declared vendors that is a real transcription rather than a
        #: placeholder.
        return 128 * 1024
    if column == COLUMN_QUALCOMM or column == COLUMN_ARM:
        return 32 * 1024  # conservative; see `gpu_cores_for`
    return 32 * 1024  # apple, and the bit-identical intersection


def max_active_blocks_for[
    column: Int
](n_threads: Int, smem_bytes: Int) raises -> Int:
    """`cudaOccupancyMaxActiveBlocksPerMultiprocessor(func, Nthreads,
    sMemSize)` (`pairwise_distance_base.cuh:306-307`), resolved per column.

    The expressible terms only: thread slots always; the shared-memory
    divisor where the column statically partitions (nvidia, amd); the
    per-block cap as a validity WALL everywhere (a request over it cannot
    launch at all, so raising beats returning a grid that will fail on the
    device). Registers and the max-blocks-per-SM cap are not modeled -- see
    the module docstring for why that cannot change a grid today.

    The APPLE arm is bit-for-bit the pre-keying computation: wall at 32 KB,
    `3072 // n_threads`, floor 1. `check_launch_config_values`' eight pinned
    grids passing unchanged is the proof.
    """
    if smem_bytes > threadgroup_limit_for[column]():
        raise Error(
            "max_active_blocks_for: threadgroup memory request of "
            + String(smem_bytes)
            + " bytes exceeds the "
            + column_name(column)
            + " column's per-block cap of "
            + String(threadgroup_limit_for[column]())
            + "; the kernel cannot launch"
        )
    var per_core = max_threads_per_core_for[column]() // n_threads
    if smem_statically_partitioned_for[column]() and smem_bytes > 0:
        var by_smem = smem_per_core_for[column]() // smem_bytes
        if by_smem < per_core:
            per_core = by_smem
    if per_core < 1:
        per_core = 1
    return per_core


def gram_splitk_is_target_arm[column: Int]() -> Bool:
    """SCHEDULING (dispatch). Whether `core/gram_splitk.mojo` is this
    column's arm for the tile-starved Gram shape, or MAX's own matmul is.

    The split-K kernel exists because MAX's matmul has NO split-K reachable
    ON APPLE: `num_k_partitions` lives only in the internal `MatmulConfig`,
    and every arm that reads it -- `multistage_gemm_split_k_kernel`,
    `SplitKTileScheduler`, and the AMD `amd_4wave_split_k_matmul` family --
    is comptime-gated `not has_apple_gpu_accelerator()`
    (`max/kernels/src/linalg/matmul/gpu/__init__.mojo:725` and `:1368` at
    tag `max/v26.5.0`; read in source and recorded in
    `bench/results/LANE_gram-splitk_2026-08-19.md`, finding 1). The Apple
    arm is `gemm_kernel_apple_8x8` with per-output-tile parallelism only,
    measured ~13x off the bandwidth floor at 32 x 32 x 4M.

    On NVIDIA and AMD that gate is OPEN: their matmul dispatch owns split-K
    machinery for exactly this regime, and the vendor rule (call MAX where
    they call cuBLAS) resumes. So the hand-written kernel is the APPLE arm,
    and non-Apple columns answer False here, sending `gemm_tn` to
    `gemm_tn_via_transpose` -> `linalg.matmul`. UNVALIDATED on nvidia/amd
    (whether their split-K actually engages at this shape is theirs to
    decide at their dispatch; ours is only to hand the shape back).

    The bit-identical column answers True: the split-K kernel is the arm
    with the determinism guarantee (fixed chunk grid, serial ascending
    fold), which is what that column exists to buy.
    """
    return column == COLUMN_APPLE or column == COLUMN_BIT_IDENTICAL
