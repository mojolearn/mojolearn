# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""One table: every kernel, every tunable it takes, per GPU vendor."""

from std.sys.compile import is_defined
from std.sys.info import (
    has_amd_gpu_accelerator,
    has_amd_rdna_gpu_accelerator,
    has_nvidia_gpu_accelerator,
)

from checks.numerics import (
    NumericMode,
    NUMERIC_FAST,
    NUMERIC_IDENTICAL,
    GLOBAL_NUMERIC_MODE,
)


comptime COLUMN_BIT_IDENTICAL = 0
comptime COLUMN_APPLE = 1
comptime COLUMN_NVIDIA = 2
comptime COLUMN_AMD = 3

comptime COLUMN_AMD_RDNA = 4
comptime COLUMN_QUALCOMM = 5
comptime COLUMN_INTEL = 6

comptime COLUMN_SPEC_BASELINE = 7

comptime COLUMN_COUNT = 8

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
    """Whether Mojo can emit a kernel for this column TODAY."""
    return (
        column == COLUMN_BIT_IDENTICAL
        or column == COLUMN_APPLE
        or column == COLUMN_NVIDIA
        or column == COLUMN_AMD
        or column == COLUMN_AMD_RDNA
    )

comptime K_HIST_BINARY = 0
comptime K_HIST_HALF_BYTE = 1
comptime K_HIST_ONE_BYTE = 2
comptime K_SCAN = 3
comptime K_SUBTRACT = 4
comptime K_SCORES = 5
comptime K_SPLIT_POINTS = 6
comptime K_HIST_2_ONE_BYTE = 7

comptime K_POINTWISE_HIST_2 = 8

comptime K_POINTWISE_HIST_2_HALF_BYTE = 9

comptime PINNED_REPLICATION_LANES = 32

comptime PINNED_REDUCE_WIDTH = 512


@fieldwise_init
struct KernelSpec(Copyable, Movable):
    """Every knob one kernel takes. Resolved once, never re-derived."""

    var block_size: Int
    """SCHEDULING. Threads per threadgroup."""

    var hist_floats_per_thread: Int
    """NUMERIC, and it reads as a memory-budget row."""

    var features_per_int: Int
    """NUMERIC in effect: it is the packing, so it decides which features share a load and therefore which sums are formed."""

    var replication_lanes: Int
    """NUMERIC. See PINNED_REPLICATION_LANES."""

    var reduce_width: Int
    """NUMERIC. See PINNED_REDUCE_WIDTH."""

    var deterministic_flush: Bool
    """NUMERIC."""

    var flush_forced_by_vendor: Bool
    """Whether `deterministic_flush` is the mode's choice or the vendor's constraint."""

    def shared_bytes(self) -> Int:
        """What this spec asks of threadgroup memory."""
        return self.block_size * self.hist_floats_per_thread * 4



comptime IDENTITY_PROFILE = 1

comptime IDENTITY_FLOOR_SHARED_BYTES = 32 * 1024

comptime IDENTITY_FLOOR_LANES = 32

comptime IDENTITY_FLOOR_BLOCK = 512


def column_meets_identity_floor(column: Int) -> Bool:
    """Whether this vendor can join `IDENTICAL` without the floor moving."""
    return (
        column_shared_limit(column) >= IDENTITY_FLOOR_SHARED_BYTES
        and column_has_threadgroup_int_atomics(column)
        and column_max_block_size(column) >= IDENTITY_FLOOR_BLOCK
    )


def identity_refusal_reason(column: Int) -> String:
    """Why `IDENTICAL` refuses this column, or empty if it does not."""
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
    """Threadgroup / shared / LDS / SLM bytes a single block may claim."""
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
        return 16 * 1024
    return IDENTITY_FLOOR_SHARED_BYTES  # BIT_IDENTICAL: frozen, not derived


def column_has_float_atomics(column: Int) -> Bool:
    """Whether this vendor can do `atomicAdd` on a `float` at all."""
    return (
        column == COLUMN_BIT_IDENTICAL
        or column == COLUMN_APPLE
        or column == COLUMN_NVIDIA
        or column == COLUMN_AMD
        or column == COLUMN_AMD_RDNA
        or column == COLUMN_INTEL
    )


def column_compares_flush_subnormals(column: Int) -> Bool:
    """CAPABILITY."""
    return column == COLUMN_APPLE


comptime VENDOR_TF32_PRODUCT_REL_BOUND = Float64(1.0e-3)


def column_vendor_fp32_matmul_is_tf32(column: Int) -> Bool:
    """CAPABILITY."""
    return column == COLUMN_NVIDIA


def vendor_fp32_matmul_is_lossy(column: Int, compute_capability: Int) -> Bool:
    """The runtime form of `column_vendor_fp32_matmul_is_tf32`: the column predicate OR'd with the one generation fact the column cannot carry -- an Apple part reporting `compute_capability == 5` (M5) runs MAX 26.5.0's fp19 simdgroup path by default."""
    if column_vendor_fp32_matmul_is_tf32(column):
        return True
    return column == COLUMN_APPLE and compute_capability == 5


def vendor_fp32_matmul_precision_name(
    column: Int, compute_capability: Int
) -> String:
    """What a check prints beside its tolerance: the precision class of the vendor fp32 product on this build and device."""
    if column_vendor_fp32_matmul_is_tf32(column):
        return String("TF32 (10-bit mantissa tensor-core product)")
    if column == COLUMN_APPLE and compute_capability == 5:
        return String("fp19 (Apple M5 simdgroup MMA, 10-bit mantissa)")
    return String("fp32")


def column_has_threadgroup_int_atomics(column: Int) -> Bool:
    """Whether a block can `atomicAdd` an `Int32` in THREADGROUP memory."""
    return True


def column_has_dedicated_shared_memory(column: Int) -> Bool:
    """Whether "shared memory" is an on-chip scratchpad or just cached RAM."""
    return True


def column_spec_guarantees_onchip_shared(column: Int) -> Bool:
    """Whether anything PROMISES the shared memory is on chip."""
    return column != COLUMN_SPEC_BASELINE


def column_max_block_size(column: Int) -> Int:
    """Largest threadgroup the vendor will dispatch, before our budget bites."""
    if column == COLUMN_SPEC_BASELINE:
        return 128
    return 1024


def column_lane_width(column: Int) -> Int:
    """Hardware lanes that move in lockstep: warp on NVIDIA, SIMD group on Apple, WAVEFRONT on AMD, wave on Adreno, sub-group on Intel."""
    if column == COLUMN_AMD:
        return 64
    if column == COLUMN_QUALCOMM:
        return 8
    if column == COLUMN_INTEL:
        return 8
    if column == COLUMN_AMD_RDNA:
        return 32
    if column == COLUMN_SPEC_BASELINE:
        return 1
    return 32


def column_lane_width_is_fixed(column: Int) -> Bool:
    """Whether `column_lane_width` is a property of the DEVICE or a decision the vendor's compiler makes per kernel."""
    return (
        column != COLUMN_QUALCOMM
        and column != COLUMN_INTEL
        and column != COLUMN_SPEC_BASELINE
    )


def spec_for(kernel: Int, device: Int, mode: NumericMode) raises -> KernelSpec:
    """The resolved knobs for one kernel, substituting column by column."""
    var identical = mode.mode == NUMERIC_IDENTICAL
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
        floats_per_thread = 0
        per_int = 0

    var catboost_block = 384 if (
        kernel == K_HIST_ONE_BYTE or kernel == K_HIST_2_ONE_BYTE
    ) else 768
    var block = catboost_block
    if floats_per_thread > 0:
        var limit = column_shared_limit(numeric_column) // (
            floats_per_thread * 4
        )
        if limit < block:
            block = limit
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

    var vendor_forces_flush = not column_has_float_atomics(device)
    var flush = mode.deterministic_flush() or vendor_forces_flush

    return KernelSpec(
        block,
        floats_per_thread,
        per_int,
        PINNED_REPLICATION_LANES,
        PINNED_REDUCE_WIDTH if identical else block,
        flush,
        vendor_forces_flush,
    )



comptime TARGET_COLUMN = (
    COLUMN_APPLE if is_defined["MOJOLEARN_COLUMN_APPLE"]() else
    COLUMN_NVIDIA if is_defined["MOJOLEARN_COLUMN_NVIDIA"]() else
    COLUMN_AMD if is_defined["MOJOLEARN_COLUMN_AMD"]() else
    COLUMN_AMD_RDNA if is_defined["MOJOLEARN_COLUMN_AMD_RDNA"]() else
    # Explicit declaration simulation only; these do not enable a backend.
    COLUMN_QUALCOMM if is_defined["MOJOLEARN_COLUMN_QUALCOMM"]() else
    COLUMN_INTEL if is_defined["MOJOLEARN_COLUMN_INTEL"]() else
    COLUMN_SPEC_BASELINE if is_defined["MOJOLEARN_COLUMN_SPEC_BASELINE"]() else
    COLUMN_AMD_RDNA if has_amd_rdna_gpu_accelerator() else
    COLUMN_AMD if has_amd_gpu_accelerator() else
    COLUMN_NVIDIA if has_nvidia_gpu_accelerator() else
    COLUMN_APPLE
)


comptime DETECTED_COLUMN = (
    COLUMN_AMD_RDNA if has_amd_rdna_gpu_accelerator() else
    COLUMN_AMD if has_amd_gpu_accelerator() else
    COLUMN_NVIDIA if has_nvidia_gpu_accelerator() else
    COLUMN_APPLE
)


def column_is_simulated() -> Bool:
    """True when `-D MOJOLEARN_COLUMN_*` names a vendor this device is not."""
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
    """SCHEDULING row, bounded by a NUMERIC one."""
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
    comptime hard = column_max_block_size(column)
    return by_smem if by_smem < hard else hard


def lane_width_for[column: Int, identical: Bool]() -> Int:
    """NUMERIC."""
    if identical:
        return PINNED_REPLICATION_LANES
    return column_lane_width(column)


def replication_lanes_for[column: Int, identical: Bool]() -> Int:
    """NUMERIC, and it is the row the whole CatBoost histogram family's layout actually rests on: the LOGICAL width of one private-replica group."""
    comptime assert PINNED_REPLICATION_LANES == IDENTITY_FLOOR_LANES, (
        "the logical replication width and the identity floor's lane count"
        " are one guarantee with two names; keep them equal"
    )
    return PINNED_REPLICATION_LANES


def reduce_width_for[kernel: Int, column: Int, identical: Bool]() -> Int:
    """NUMERIC."""
    comptime block = block_size_for[kernel, column]()
    comptime pinned = PINNED_REDUCE_WIDTH if identical else block
    return block if block < pinned else pinned



comptime SYNC_BLOCK = 0

comptime SYNC_LANE = 1


def sync_granularity_for[column: Int]() -> Int:
    """The finest sync a kernel may rely on."""
    return SYNC_BLOCK


def requires_uniform_iteration_for[column: Int]() -> Bool:
    """Whether every thread of a block must run the SAME iteration count."""
    return sync_granularity_for[column]() == SYNC_BLOCK


def sub_byte_lane_sync_for[column: Int]() -> Int:
    """SCHEDULING row (DEVIATION 1947): which barrier the CatBoost histogram accumulators use for their TURN-TAKING sync, the one that stands where theirs writes `tiled_partition<8>::sync()` or `tiled_partition<32>::sync()` between two writes to the same private slice."""
    if not column_lane_width_is_fixed(column):
        return SYNC_BLOCK
    if column_lane_width(column) != PINNED_REPLICATION_LANES:
        return SYNC_BLOCK
    return SYNC_LANE


def deterministic_flush_for[column: Int, identical: Bool]() -> Bool:
    """NUMERIC row, comptime, so a kernel can branch on it."""
    return identical or not column_has_float_atomics(column)



comptime HIST_SMEM_WARP_PRIVATE_F32 = 0

comptime HIST_SMEM_SHARED2_I32 = 1


def pointwise_one_byte_fixed_for[column: Int, identical: Bool]() -> Bool:
    """NUMERIC row: whether the POINTWISE one-byte family routes EVERY width through the 8-bit fixed-point accumulator."""
    comptime if identical:
        return True
    return column == COLUMN_APPLE


def pointwise_doc_split_for[column: Int, ordered: Bool]() -> Bool:
    """NUMERIC row (DEVIATION 2624): whether the POINTWISE histogram launchers split the DOCUMENT axis `EstimateBlockPerFeatureMultiplier` ways. Above one, every document block of a feature float-`atomicAdd`s its partial into the same `binSums` cell in whatever order the device finishes them, and the multiplier itself follows `sm_count`, so the ordered tiers (deterministic and identical) keep one block per feature group per part; see `pw_block_multiplier` in `gbdt/methods/pointwise_kernels.mojo`."""
    comptime if ordered:
        return False
    return True


def greedy_one_byte_fixed_for[column: Int, identical: Bool]() -> Bool:
    """SCHEDULING row (DEVIATION 1906, NARROWED by DEVIATION 1947): whether the GREEDY one-byte family routes EVERY width through the fused 8-bit fixed-point kernel (`hist_2_one_byte_8bit.mojo`) instead of CatBoost's maxBins ladder."""
    comptime if identical:
        return False
    comptime if is_defined["MOJOLEARN_2043_FAST_FUSED_ONE_BYTE"]():
        return True
    return True


def greedy_sub_byte_excluded_for[column: Int, identical: Bool]() -> Bool:
    """ROUTING row, RETRACTED 2026-09-01 (DEVIATION 1947 supersedes DEVIATION 1910): whether the GREEDY sub-byte histogram families -- BINARY (32 features per word) and HALF-BYTE (8 per word) -- are comptime-EXCLUDED from the build, their launch sites refusing at runtime BY NAME."""
    return False


def greedy_quantized_hist_for[column: Int, identical: Bool]() -> Bool:
    """NUMERIC row (DEVIATIONS 1911/1912): whether the NON-SYMMETRIC drivers' one-byte histogram build routes through the QUANTIZED SHARED-HISTOGRAM family (`kernel/hist_quantized_shared.mojo`) -- per-round fixed-point gradient pairs packed one 64-bit word per row, ONE shared-memory Int32 histogram per thread block accumulated with threadgroup integer..."""
    comptime if identical:
        return False
    comptime if is_defined["MOJOLEARN_2045_FAST_NO_QUANT_HIST"]():
        return False
    return (
        column == COLUMN_APPLE
        or column == COLUMN_NVIDIA
        or column == COLUMN_AMD
        or column == COLUMN_AMD_RDNA
    )


def quantized_hist_group_features_for[column: Int]() -> Int:
    """SCHEDULING row (DEVIATION 1913): how many one-byte features one thread block's shared histogram covers in the quantized family."""
    comptime limit = column_shared_limit(column)
    var g = limit // (256 * 2 * 4)
    g = (g // 4) * 4
    if g < 4:
        g = 4
    if g > 32:
        g = 32
    return g


def reorder_single_pass_for[column: Int, identical: Bool]() -> Bool:
    """SCHEDULING row (DEVIATION 1907): stable partition above 500,000 rows.

    IDENTICAL's NVIDIA candidate requires an explicit build define until
    a large-input identity and timing run exercises the routed kernel.
    Small identity fixtures cannot reach this branch. The kill switch wins
    over the opt-in, and other vendors retain the established partition.
    """
    comptime if is_defined["MOJOLEARN_2042_FAST_NO_LOOKBACK"]():
        return False
    comptime if identical:
        return (
            column == COLUMN_NVIDIA
            and is_defined["MOJOLEARN_IDENTICAL_SINGLE_PASS_PARTITION"]()
        )
    return column == COLUMN_NVIDIA


def ridx_only_splits_for[column: Int, identical: Bool]() -> Bool:
    """SCHEDULING row (DEVIATION 1902): whether the NON-SYMMETRIC driver's split moves only the row index, leaving the stat planes stationary for the life of the fit, with every stat reader gathering `stats[row_index[pos]]` instead of reading a permuted plane."""
    comptime if identical:
        return False
    comptime if is_defined["MOJOLEARN_2044_FAST_NO_RIDX_ONLY"]():
        return False
    return (
        column == COLUMN_APPLE
        or column == COLUMN_NVIDIA
        or column == COLUMN_AMD
        or column == COLUMN_AMD_RDNA
    )


def hist_smem_mode_for[column: Int, identical: Bool]() -> Int:
    """NUMERIC row: HOW the hist_2 family accumulates in shared memory."""

    comptime CATBOOST_PRIVATE_BYTES = 384 * 32 * 4
    comptime limit = column_shared_limit(column)

    comptime if is_defined["MOJOLEARN_2046_FAST_SHARED_I32"]():
        return HIST_SMEM_SHARED2_I32
    comptime if identical:
        return HIST_SMEM_SHARED2_I32  # the BIT_IDENTICAL column's value
    elif limit < CATBOOST_PRIVATE_BYTES:
        return HIST_SMEM_SHARED2_I32
    else:
        return HIST_SMEM_WARP_PRIVATE_F32



comptime PINNED_PARTITION_CHUNKS_SM = 32


def partition_chunks_sm_for[identical: Bool](device_sm: Int) -> Int:
    """The `sm_count` the partition-stats chunk formula is fed."""
    comptime if identical:
        return PINNED_PARTITION_CHUNKS_SM
    else:
        return device_sm


def hist2_block_size_for[column: Int, smem_mode: Int]() -> Int:
    """SCHEDULING row bounded by the NUMERIC budget, per accumulation mode."""

    comptime hard = column_max_block_size(column)
    comptime if smem_mode == HIST_SMEM_SHARED2_I32:
        comptime limit = column_shared_limit(column) // 64
        comptime by_smem = 512 if limit >= 512 else limit
        return by_smem if by_smem < hard else hard
    else:
        return block_size_for[K_HIST_2_ONE_BYTE, column]()


def pw_hist2_block_size_for[column: Int, fixed: Bool]() -> Int:
    """SCHEDULING row for the POINTWISE one-byte family's block, per route."""
    return block_size_for[K_POINTWISE_HIST_2, column]()


def pw_hist2_smem_floats_for[column: Int, fixed: Bool]() -> Int:
    """Companion to `pw_hist2_block_size_for`: the shared scratch, in 4-byte slots."""
    return 32 * pw_hist2_block_size_for[column, fixed]()


def replicas_for(hist_cells: Int) -> Int:
    """DELETED IN SPIRIT."""
    return -16
    return 1



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


comptime PINNED_LIB_REDUCE_LANES = 32

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
    """SCHEDULING **only for indexing**, NUMERIC when it bounds a reduction."""

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
    """SCHEDULING."""
    if kernel == K_LIB_SELECT_RADIX:
        return 256
    if kernel == K_LIB_SELECT_WARPSORT:
        return 8 * lib_lane_width(column)
    if kernel == K_LIB_TRANSPOSE:
        return 32 * 32 // 4
    if kernel == K_LIB_JACOBI_EIGH:
        return 32
    if kernel == K_LIB_GEMM_CONTRACTION or kernel == K_LIB_FUSED_DISTANCE_NN:
        return 16 * 16
    if kernel == K_LIB_GRAM_SPLITK:
        return 256
    return 128


def lib_lane_width(column: Int) -> Int:
    """SCHEDULING."""
    return column_lane_width(column)


def lib_spec_for(
    kernel: Int, device: Int, mode: NumericMode
) raises -> LibKernelSpec:
    """The resolved knobs for one library kernel."""
    var identical = mode.mode == NUMERIC_IDENTICAL
    var numeric_column = COLUMN_BIT_IDENTICAL if identical else device

    var reduce_lanes = PINNED_LIB_REDUCE_LANES
    if not identical:
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



def lib_smem_pages(kernel: Int, column: Int, page_bytes: Int) -> Int:
    """SCHEDULING."""
    if 2 * page_bytes <= column_shared_limit(column):
        return 2
    return 1



def lib_block_bounds_a_float_fold[kernel: Int]() -> Bool:
    """NUMERIC CLASSIFIER, and the correction of a label this file got wrong."""
    return (
        kernel == K_LIB_ROW_NORM
        or kernel == K_LIB_REDUCE_BY_KEY
        or kernel == K_LIB_PLUS_PLUS
        or kernel == K_LIB_COLUMN_STATS
        or kernel == K_LIB_JACOBI_EIGH
        or kernel == K_LIB_WEIGHTED_VERTEX_DEG
    )


def lib_block_size_for[kernel: Int, column: Int]() -> Int:
    """SCHEDULING for most rows, NUMERIC for three of them."""
    comptime identical = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    comptime numeric_row = lib_block_bounds_a_float_fold[kernel]()
    comptime resolved = (
        COLUMN_BIT_IDENTICAL if identical and numeric_row else column
    )
    comptime lanes = column_lane_width(resolved)
    if kernel == K_LIB_SELECT_RADIX:
        return 256
    if kernel == K_LIB_SELECT_WARPSORT:
        return 8 * lanes
    if kernel == K_LIB_TRANSPOSE:
        return 256
    if kernel == K_LIB_JACOBI_EIGH:
        return 32 if lanes <= 32 else lanes
    if kernel == K_LIB_GEMM_CONTRACTION or kernel == K_LIB_FUSED_DISTANCE_NN:
        return 256
    if kernel == K_LIB_GRAM_SPLITK:
        return 256
    return 128


def lib_lane_width_for[column: Int]() -> Int:
    """SCHEDULING."""
    return column_lane_width(column)


def lib_reduce_lanes_for[column: Int, identical: Bool]() -> Int:
    """NUMERIC."""
    return PINNED_LIB_REDUCE_LANES


def lib_smem_pages_for[column: Int, page_bytes: Int]() -> Int:
    """SCHEDULING."""
    comptime limit = column_shared_limit(column)
    return 2 if 2 * page_bytes <= limit else 1


def lib_smem_page_fits_for[column: Int, page_bytes: Int]() -> Bool:
    """SCHEDULING row (2026-09-09, orchestrator, Apple RUN OWED of the split-K lane): whether ONE shared page of `page_bytes` fits under the column's shared limit at all. `lib_smem_pages_for` answers "one page or two"; it cannot say "not even one". The 128x128 tuned pair at K step 32 is 36,864 bytes a page, which no NVIDIA leg noticed (48 KB) and which Metal refuses at pipeline creation (32 KB: "Threadgroup memory size (36864) exceeds the maximum threadgroup memory allowed (32768)", gemm_device_check and gemm_backward_check on the M4). A plan whose page does not fit resolves its K step down through this row instead of naming a vendor. The fused attention kernels ask the same question per head dim: the head-dim-128 backward path claims 35,600 bytes and takes the eager path on a 32 KB column; register-blocked forward needs 18,624 bytes and fits."""
    return page_bytes <= column_shared_limit(column)


def lib_hardware_ftz_fma_for[column: Int]() -> Bool:
    """Capability row for NVIDIA's explicit round-to-nearest FMA intrinsics.

    This is not permission to replace round-then-flush with a bare .ftz
    instruction. At a smallest-normal rounding boundary, fma.rn.ftz can
    return zero where round-then-flush returns 0x00800000. Callers must
    preserve explicit round-then-flush semantics or correct that case;
    see the adversarial seam evidence from 2026-09-09.
    Other columns retain their existing software spelling.
    """
    return column == COLUMN_NVIDIA


def attn_zdot_rows_per_block_for[column: Int]() -> Int:
    """SCHEDULING row (DEVIATION 2528, 2026-09-11, trial arm only; brief docs/lanes/BRIEF_attention_step_2026-09-11.md section 12): query rows per 256-thread block of the fused attention's register-blocked y/dy kernel (`fused_bwd_ydy_tiled_kernel`), 64 or 32. The kernel's shared page is `(2 * rows + 128) * 20` floats (20,480 B at 64, 15,360 B at 32), and on a column whose shared memory is partitioned per compute unit the page bounds the resident blocks. The rows are a schedule, never a numeric term: every chain keeps its terms and order at either value. UNMEASURED on every column. AMD reads 32 as the variant section 11.3 named to price; the page-only count (3 blocks x 64 rows vs 4 x 32 rows per CU) does not favor it, so the AMD leg prices both through the `_r32` / `_r64` arm names and this row follows that measurement. The shipped build reads it nowhere."""
    if column == COLUMN_AMD:
        return 32
    return 64


def lib_gemm_block_parallelism_for[column: Int]() -> Int:
    """SCHEDULING row, SHIPPED since DEVIATION 2595 (2026-09-11; brief docs/lanes/BRIEF_gemm_long_k_2026-09-11.md sections 3, 4 and 10; first added by DEVIATION 2591 as a trial-arm row): how many 256-thread GEMM blocks the column runs side by side. A value above 0 TURNS ON the `ksplit` default in `gemm/checks/gemm_identical.mojo::identical_gemm_shipped_into`: every call the long-k group rule takes (section 4, rules 1, 2 and 4, at `S` = this value) runs the 128x128 group kernel over power-of-two leaf groups plus one fold launch, and every other call runs the plan `choose_gemm_plan` picks, as before. 0 turns it off: the dispatch compiles to the old line and the TUNED 128x128 plan runs exactly as it did. NVIDIA 132, MEASURED: the H100's SM count (docs/lanes/BRIEF_attention_step_2026-09-11.md section 3.1), and the value the `ksplit` arm ran at on the H100 leg that flipped it (bench/results/e1g/2026-09-11_152822-nvidia-h100-80gb-hbm3-gemm-longk, lean step geomean 0.895 on enwik8 and Pile GitHub, every step witness equal). AMD 0, OFF UNTIL THE MI300X LEG DECIDES THE AMD VALUE: nothing on an AMD board has measured the arm, and this row takes the value that leg reads (its CONTROL pair and its `ksplit` against `shipped` lean step verdict). The trial arm still runs on AMD at the column's reading through `lib_gemm_block_parallelism_trial_for`. Every other column 0 (Apple included, so the Apple identity card compiles the old line). A wrong value costs time and can never move a bit, because the group size reaches no leaf boundary and no tree level (brief section 5.5)."""
    if column == COLUMN_NVIDIA:
        return 132
    if column == COLUMN_AMD:
        # Measured 2026-09-11 on the Hot Aisle MI300X with the trial arm at S=110
        # (e1g/2026-09-11_164818-amd-mi300x-hotaisle-gemm-longk): lean step
        # 1.953 -> 1.198 s on both corpora, geomean 0.614, witnesses equal.
        return 110
    return 0


def lib_gemm_block_parallelism_trial_for[column: Int]() -> Int:
    """SCHEDULING row (DEVIATION 2595, 2026-09-11, trial arm only; brief docs/lanes/BRIEF_gemm_long_k_2026-09-11.md section 10): the `S` the `ksplit` TRIAL arm reads, so a leg can still force the arm on a column whose shipped row is 0. The shipped row wherever it is above 0 (NVIDIA 132, so the arm and the default split identically there). AMD 110, from a READING, not a measurement (DEVIATION 2591): the attention brief section 11.1 transcribes 110 CUs (pinned to the MI250X; the MI325X and MI300X counts are not in the repository) and resident blocks per CU as `min(2048 // 256, 65536 // page bytes)`; the shipped 128x128 GEMM block holds two 20,480 B pages (40,960 B), so one block per CU and 110 side by side. The MI300X leg's CONTROL pair `ctl_nt_1536x1408x768` / `ctl_nt_1664x1408x768` (`bench/gemm_step_price_main.mojo`) reads the real value. Every other column 0, meaning no reading: the arm then takes the finest split the workspace cap allows. The shipped build reads it nowhere."""
    comptime shipped = lib_gemm_block_parallelism_for[column]()
    if shipped > 0:
        return shipped
    if column == COLUMN_AMD:
        return 110
    return 0


def attn_fwd_rows_per_block_for[column: Int]() -> Int:
    """SCHEDULING row (DEVIATION 2531, 2026-09-11; brief docs/lanes/BRIEF_attention_step_2026-09-11.md sections 14 and 15): query rows per 256-thread block of the fused attention's second-round forward kernel (`fused_attn_forward_r2_kernel`), 64 (the shipped hd-64 sstash geometry) or 32, read by the bare `_fgrid` arm token (`_fgrid_r32` / `_fgrid_r64` force it). The kernel's shared page is `(32 * 64 + rows * 35) * 4` bytes (17,152 B at 64 rows, 12,672 B at 32), and on a column whose shared memory is partitioned per compute unit the page bounds the resident blocks. The rows are a schedule, never a numeric term: the score, denominator and context chains keep their terms and order at either value, and the row maximum is an `identical_fmax` fold whose grouping is free. NVIDIA 32, MEASURED (DEVIATION 2534, H100 leg bench/results/e1g/2026-09-11_154257-nvidia-h100-80gb-hbm3-attention-round3, commit 5bcfa71d): the lean LM step under `stash_tiled_fgrid_r32` was 0.3718 / 0.3716 s against `stash_tiled` 0.3845 / 0.3819 s (enwik8 / Pile GitHub), every step witness equal, and `fgrid_r64` priced at 1.00x of stash_tiled on real activations while `fgrid_r32` priced 1.07x. AMD 32 is still the variant brief section 11.4 named to price (the page-only count, 3 blocks x 64 rows against 5 x 32 per CU, does not settle it); THE MI300X LEG DECIDES IT through the `_fgrid_r32` / `_fgrid_r64` arm names. Every other column 64, unmeasured. The shipped default arm (`attn_default_arm_for`) forces its rows with `_fgrid_r32`, so this row never moves a shipped path."""
    if column == COLUMN_NVIDIA:
        return 32
    if column == COLUMN_AMD:
        return 32
    return 64


def attn_dkdv_keys_per_block_for[column: Int]() -> Int:
    """SCHEDULING row (DEVIATION 2597, 2026-09-11; brief docs/lanes/BRIEF_attention_step_2026-09-11.md sections 16 and 18): keys per 256-thread block of the fused attention's trial dk/dv folds over the stash (`fused_bwd_dkdv_r2_kernel`, and `fused_bwd_kvfold_r2_kernel` under the `_kvsplit` token), 64 (the shipped `fused_bwd_dkdv_tiled_pf_kernel` geometry) or 32, read by the bare `_kvgrid` arm token (`_kvgrid_r32` / `_kvgrid_r64` force it). At 32 keys a thread holds 8 dk and 8 dv accumulators instead of 16 and 16, and the joint page is `(2 * 16 * 64 + 2 * 16 * keys) * 4` bytes (16,384 B at 64, 12,288 B at 32; a `_kvsplit` fold page is half that). The keys per block are a schedule, never a numeric term: every dk and dv chain keeps its terms and its order (heads of the kv group ascending, queries ascending over the key's visible range) at either value. AMD 32, MEASURED: DigitalOcean MI325X leg bench/results/e1g/2026-09-11_180903-amd-mi325x-do-attention-dkdv (commit 5cc3b8df), `stash_tiled_fgrid_r32_qres_pf_kvgrid_r32` against `baseline` lean step 1.623 / 1.633 -> 1.376 / 1.370 s (enwik8 / Pile GitHub, FLIP geomean 0.8436, every step witness equal), in-step dk/dv 169.8 ms (baseline) -> 21.2 ms; section 16 had read 32 from the tiled dk/dv thread state (32 accumulators and 10 operand registers per thread, twice the tiled dq fold's). The AMD shipped default (`attn_default_arm_for`) forces the same 32 with `_kvgrid_r32` (brief section 18), so this row and the default agree on AMD and a bare `_kvgrid` resolves to the default's instantiation there. Every other column 64, unmeasured. A shipped build reads this row only for a default carrying bare `_kvgrid`, which no column's default does."""
    if column == COLUMN_AMD:
        return 32
    return 64


comptime ATTN_DEFAULT_WORD_BASELINE = 0
comptime ATTN_DEFAULT_WORD_STASH_TILED = 7
"""The attention arm word `stash_tiled`: bits 1 (fwd_sstash), 2 (bwd_stash) and 4 (bwd_tiled) of transformer/impl/llama/fused_attention.mojo (DEVIATIONS 2525 to 2527). The matrix cannot import that file (it imports this one), so the word is a literal here and fused_attention.mojo asserts at build time that it equals its own composition."""

comptime ATTN_DEFAULT_WORD_STASH_TILED_FGRID_R32_QRES_PF = 3175
"""The attention arm word `stash_tiled_fgrid_r32_qres_pf`: stash_tiled (7) | 64 (fwd_grid, DEVIATION 2531) | 2048 (forward rows 32) | 32 (fwd_qres, DEVIATION 2530) | 1024 (preflush, DEVIATION 2533) = 3175. fused_attention.mojo asserts at build time that it equals its own composition."""

comptime ATTN_DEFAULT_WORD_STASH_TILED_FGRID_R32_QRES_PF_KVGRID_R32 = 52327
"""The attention arm word `stash_tiled_fgrid_r32_qres_pf_kvgrid_r32`: stash_tiled_fgrid_r32_qres_pf (3175) | 16384 (bwd_kvgrid, DEVIATION 2597) | 32768 (dk/dv keys per block 32) = 52327. fused_attention.mojo asserts at build time that it equals its own composition (`ATTN_ARM_R3_KVGRID_R32_DEFAULT`)."""


def attn_default_arm_for[column: Int]() -> Int:
    """ROUTING row (DEVIATION 2534, 2026-09-11; brief docs/lanes/BRIEF_attention_step_2026-09-11.md sections 15 and 18): the attention arm word the SHIPPED build runs on this column (`ATTN_ARM_DEFAULT` in transformer/impl/llama/fused_attention.mojo; a `-D MOJOLEARN_ATTN_ARM_TRIAL=1` build runs it when MOJOLEARN_ATTN_ARM is unset and keeps every other arm selectable by name). Every arm is bit-equal to the eager oracle by the identity arguments of brief sections 4, 12, 14 and 16, so this row picks a schedule and never a result. NVIDIA `stash_tiled_fgrid_r32_qres_pf`, MEASURED: H100 leg bench/results/e1g/2026-09-11_154257-nvidia-h100-80gb-hbm3-attention-round3 (commit 5bcfa71d), lean LM step 0.3845 / 0.3819 s under stash_tiled against 0.3346 / 0.3340 s (enwik8 / Pile GitHub), every step witness equal, fwd+bwd on real activations 1.41x of stash_tiled; ENGINEERING_RULES 9 flips it. AMD `stash_tiled_fgrid_r32_qres_pf_kvgrid_r32` (ATTN_DEFAULT_WORD_STASH_TILED_FGRID_R32_QRES_PF_KVGRID_R32), MEASURED on the DigitalOcean MI325X against the previous AMD default `baseline`, every step witness equal (the comment in the body names the evidence and the verdict); a shipped build compiles its DEVIATION 2597 dk/dv kernel because the default carries it (brief section 18). Apple and every other column `stash_tiled` (unmeasured for the round 3 and 2597 arms as a price). `-D MOJOLEARN_ATTN_DEFAULT_R3_EVERY_COLUMN=1` returns the NVIDIA word on every column, so a no-trial build on a Mac reaches the shipped round 3 branch; `-D MOJOLEARN_ATTN_DEFAULT_KVGRID_EVERY_COLUMN=1` returns the AMD word on every column, so the same build reaches the shipped DEVIATION 2597 dk/dv branch (check knobs, the `MOJOLEARN_EXPERIMENTAL_SMALLK_IDENTICAL` pattern; never a shipped build; at most one of the two)."""
    comptime assert not (is_defined["MOJOLEARN_ATTN_DEFAULT_R3_EVERY_COLUMN"]() and is_defined["MOJOLEARN_ATTN_DEFAULT_KVGRID_EVERY_COLUMN"]()), (
        "MOJOLEARN_ATTN_DEFAULT_R3_EVERY_COLUMN and"
        " MOJOLEARN_ATTN_DEFAULT_KVGRID_EVERY_COLUMN each name a different"
        " default for every column; define at most one"
    )
    comptime if is_defined["MOJOLEARN_ATTN_DEFAULT_KVGRID_EVERY_COLUMN"]():
        return ATTN_DEFAULT_WORD_STASH_TILED_FGRID_R32_QRES_PF_KVGRID_R32
    comptime if is_defined["MOJOLEARN_ATTN_DEFAULT_R3_EVERY_COLUMN"]():
        return ATTN_DEFAULT_WORD_STASH_TILED_FGRID_R32_QRES_PF
    if column == COLUMN_NVIDIA:
        # Measured 2026-09-11 on a RunPod H100 80GB HBM3 (1980 MHz) against the
        # previous NVIDIA default stash_tiled_fgrid_r32_qres_pf, commit 6d4bd867,
        # every step witness equal, card IDENTICAL
        # (bench/results/e1g/2026-09-11_185833-nvidia-h100-80gb-hbm3-attention-zdot):
        #   verdict stash_tiled_fgrid_r32_qres_pf_kvgrid_r32 FLIP geomean=0.9908
        #   enwik8=0.9901 pilegithub=0.9915 (vs the previous default, same pod)
        # Lean step 0.2929 / 0.2926 -> 0.2900 / 0.2901 s (enwik8 / Pile GitHub).
        # Same leg: _zlag_kvgrid_r32 0.9902 (its DEVIATION 2598 zdot schedule is
        # compiled on trial builds only), _zlag 0.9968, _kvsplit NO FLIP 1.0131.
        # Before it: stash_tiled_fgrid_r32_qres_pf (round 3, e1g/...154257).
        return ATTN_DEFAULT_WORD_STASH_TILED_FGRID_R32_QRES_PF_KVGRID_R32
    if column == COLUMN_AMD:
        # Measured 2026-09-11 on the DigitalOcean MI325X against the previous
        # AMD default baseline, commit 5cc3b8df, every step witness equal
        # (bench/results/e1g/2026-09-11_180903-amd-mi325x-do-attention-dkdv):
        #   verdict stash_tiled_fgrid_r32_qres_pf_kvgrid_r32 FLIP geomean=0.8436
        #   enwik8=0.8478 pilegithub=0.8394 (vs baseline, DO MI325X, witnesses equal)
        # Lean step 1.623 / 1.633 -> 1.376 / 1.370 s (enwik8 / Pile GitHub); in-step
        # dk/dv 169.8 ms (baseline) -> 21.2 ms. The runners-up on the same leg:
        # _kvsplit FLIP 0.8552, _kvrecompute FLIP 0.8652, and the round 3 arm
        # without a dk/dv token NO FLIP 1.1400. ENGINEERING_RULES 9 flips the
        # winner. Before it: baseline, from one RunPod MI300X pod
        # (e1g/2026-09-11_171959-amd-mi300x-runpod-attention-three).
        return ATTN_DEFAULT_WORD_STASH_TILED_FGRID_R32_QRES_PF_KVGRID_R32
    return ATTN_DEFAULT_WORD_STASH_TILED


def knn_warpsort_select_for[column: Int, identical: Bool]() -> Bool:
    """SCHEDULING row (DEVIATION 1922): whether the k-NN TILED path's selector is the implemented RAFT WARPSORT (`select_warpsort.mojo`, `warpsort_topk_block_kernel`) instead of the implemented RAFT radix (`select_radix.mojo`) for `2 < k <= 256`."""
    comptime if identical:
        return False
    return column == COLUMN_NVIDIA


def knn_auto_follows_their_dispatch_for[column: Int, identical: Bool]() -> Bool:
    """SCHEDULING row (DEVIATION 1923): whether the k-NN AUTO arm follows cuVS's dispatch UNCONDITIONALLY -- `k <= 64` + row-major + L2 goes to `fusedL2Knn`, x-split included (`knn_brute_force.cuh:443`) -- instead of DEVIATION 36's shape test (fused only when `launchConfigGenerator` picks `grid_x == 1`, tiled when it would engage the x-split)."""
    comptime if identical:
        return False
    return column == COLUMN_NVIDIA



comptime QUANTIZE_SEARCH_LINEAR = 0

comptime QUANTIZE_SEARCH_BINARY = 1

comptime QUANTIZE_SEARCH_TWO_LEVEL = 2


def quantize_search_for[column: Int]() -> Int:
    """SCHEDULING row: HOW the evaluator's quantize finds a value's bin."""
    if column == COLUMN_APPLE:
        return QUANTIZE_SEARCH_TWO_LEVEL
    return QUANTIZE_SEARCH_LINEAR


def _knn_identical_round_column(column: Int) -> Bool:
    """The columns whose IDENTICAL k-NN defaults were flipped 2026-09-09: small-k selector, transposed index layout with the register tile, and index-axis tiling. NVIDIA and AMD flipped on the H100 evidence; Apple flipped the same afternoon on the M4 four-arm price (100k x 32, k 10, 9 rounds, PRICE_MS medians baseline -> both: 20.5 -> 15.1 ms at 32 queries, 26.9 -> 14.9 at 128, 182.2 -> 66.6 at 1000; all four arms byte-equal to the NVIDIA baseline across 143,628 cells). On Apple the transpose-only arm was slightly faster still at 128 and 1000 queries (12.1, 60.2 ms) and slower at 32 (16.3 ms); the both column is the shipped one."""
    return (
        column == COLUMN_NVIDIA
        or column == COLUMN_AMD
        or column == COLUMN_AMD_RDNA
        or column == COLUMN_APPLE
    )


def knn_smallk_select_for[column: Int, identical: Bool]() -> Bool:
    """ROUTING row: whether the IDENTICAL tiled k-NN arm selects k <= KNN_SMALLK_MAX_K with the per-thread composite-key selector (`neighbors/checks/select_smallk_identical_candidate.mojo`) instead of the 64-bit radix. Both return the k smallest (distance, index) keys ascending, so the bits are equal by construction and the gate is the four-arm dispatch check. `-D MOJOLEARN_KNN_IDENTICAL_LEGACY_SELECT=1` forces the radix on every column; `-D MOJOLEARN_EXPERIMENTAL_SMALLK_IDENTICAL=1` forces the selector on every column."""
    comptime if not identical:
        return False
    comptime if is_defined["MOJOLEARN_KNN_IDENTICAL_LEGACY_SELECT"]():
        return False
    comptime if is_defined["MOJOLEARN_EXPERIMENTAL_SMALLK_IDENTICAL"]():
        return True
    return _knn_identical_round_column(column)


def knn_transposed_index_for[column: Int, identical: Bool]() -> Bool:
    """ROUTING row: whether the IDENTICAL tiled k-NN arm transposes the index once per request so the pinned distance tile reads it coalesced. Same per-cell fma chain, so the bits are equal. `-D MOJOLEARN_KNN_IDENTICAL_LEGACY_LAYOUT=1` forces the row-major layout; `-D MOJOLEARN_EXPERIMENTAL_KNN_TRANSPOSE_IDENTICAL=1` forces the transpose on every column."""
    comptime if not identical:
        return False
    comptime if is_defined["MOJOLEARN_KNN_IDENTICAL_LEGACY_LAYOUT"]():
        return False
    comptime if is_defined["MOJOLEARN_EXPERIMENTAL_KNN_TRANSPOSE_IDENTICAL"]():
        return True
    return _knn_identical_round_column(column)


def knn_distance_register_tile_for[column: Int, identical: Bool]() -> Bool:
    """SCHEDULING row: whether the transposed IDENTICAL distance tile computes a column-selected register tile per thread (`pinned_distance_tile.mojo::pinned_distance_register_tile_kernel`) instead of one cell per thread. Every cell's chain is still one ascending serial fma chain over the feature axis (IDENTITY_PATHS row 24), so the bits are equal. Only reachable when `knn_transposed_index_for` is true. `-D MOJOLEARN_KNN_IDENTICAL_SCALAR_TILE=1` keeps one cell per thread."""
    comptime if not identical:
        return False
    comptime if is_defined["MOJOLEARN_KNN_IDENTICAL_SCALAR_TILE"]():
        return False
    return knn_transposed_index_for[column, identical]()


def knn_selector_specialize_common_for[column: Int, identical: Bool]() -> Bool:
    """Compile-time k=10/15 removes dynamic insertion guards and threshold
    selection. NVIDIA 400k/4000q/k10: selector 20.3 -> 9.1 ms on the L40S
    (2026-09-09 selector resume). The same integer composite-key scan and
    block minimum are retained. Other columns can force the specialization
    for qualification; the generic capacity buckets remain the A/B arm.
    """
    comptime if not identical:
        return False
    comptime if is_defined["MOJOLEARN_KNN_IDENTICAL_GENERIC_K"]():
        return False
    comptime if is_defined["MOJOLEARN_KNN_IDENTICAL_SPECIALIZE_COMMON"]():
        return True
    return column == COLUMN_NVIDIA


def knn_selector_shuffle_for[column: Int, identical: Bool]() -> Bool:
    """SCHEDULING row (2026-09-09, lane/knn-selector): whether the IDENTICAL small-k selector (`select_smallk_identical_candidate.mojo::smallk_bucket_kernel`) takes each rank's block minimum through a lane-group butterfly (`shuffle_xor` over `column_lane_width` lanes, one shared slot per lane group, ONE barrier per rank, double-buffered slots) instead of the eight-level shared-memory tree (eleven barriers per rank). The reduced value is a UInt64 composite key and the fold is an integer minimum, which is associative, commutative and idempotent, so the winner is the same key under any tree and the bits are equal by construction; the gate is the four-arm dispatch check. Every column with a fixed lane width (`column_lane_width_is_fixed`) takes the butterfly; the Qualcomm and Intel columns, whose lane width the vendor's compiler chooses per kernel, keep the tree. `-D MOJOLEARN_KNN_IDENTICAL_TREE_SELECT=1` forces the tree on every column."""
    comptime if not identical:
        return False
    comptime if is_defined["MOJOLEARN_KNN_IDENTICAL_TREE_SELECT"]():
        return False
    return column_lane_width_is_fixed(column)


def knn_selector_warpbound_guard_for[column: Int, identical: Bool]() -> Bool:
    """SCHEDULING row (2026-09-11, DEVIATION 2523): whether the IDENTICAL small-k selector (`select_smallk_identical_candidate.mojo::smallk_bucket_kernel`) runs its per-lane insertion chain behind a warp-uniform ballot with a warp-scope admission bound (every second batch, each lane publishes its list head, aligned lane groups take their minimum through five xor shuffles, the bound is the maximum over groups, and a lane admits against min(own k-th smallest, bound)) instead of the always-issued predicated chain. The bound is at or above the k-th smallest of a subset of the union of the warp's lists, so at least k union keys are at or below it and no key above it can be in the row's top-k; keys carry their column so no equality; the union still holds the true top-k, the rank phase pops the same UInt64 minima with the same index tie rule, and the bits are equal by construction (the nine-arm dispatch check, 18 planted cases, M4 and H100). Measured on the H100 2026-09-11: the chain issued on 90 to 96 percent of warp-steps under the per-lane threshold and on 46 percent under the bound; full requests at 400k x 4k x d32 went 30.8 to 27.8 ms (k10) and 36.0 to 31.0 ms (k15), every pair in both orders (bench/results/e1g/2026-09-11_023138-nvidia). NVIDIA only until the Apple and AMD columns are timed (RUN OWED; both are fixed-lane-width columns and pass the identity check, so timing is the only gate). Requires the block-uniform trip count (DEVIATION 2497) and a fixed-lane-width column. `-D MOJOLEARN_KNN_IDENTICAL_INSERT_CHAIN=1` restores the predicated chain on every column."""
    comptime if not identical:
        return False
    comptime if is_defined["MOJOLEARN_KNN_IDENTICAL_INSERT_CHAIN"]():
        return False
    comptime if not column_lane_width_is_fixed(column):
        return False
    return column == COLUMN_NVIDIA


def svm_block_solve_warp_folds_for[column: Int, width: Int]() -> Bool:
    """SCHEDULING row (2026-09-11, DEVIATION 2623): whether `svm/impl/smoblocksolve.mojo::smo_block_solve_kernel[width]` folds its three arg-reductions with `block_argext` warp butterflies (DEVIATION 2491) instead of the halving trees with a one-slot thread ballot that preceded it. Both select the same (value, key) element under a total order with unique keys, so no column's bits depend on this row. NVIDIA refuses the warp kernel at width 1024 (H100 80GB HBM3, driver 580.126.09, CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES, so every SVC fit above 512 training rows failed from 2491 through 0.8.2) and launches it at 512; the tree kernel launches at 1024 there. Fusing two of the warp folds or dropping the WSIZE threadgroup diagonal did not make the warp kernel launch. The Apple M4 and the AMD MI300X launch the warp kernel at 1024. `-D MOJOLEARN_SVM_TREE_FOLDS` takes the tree schedule on every column for an A/B."""
    comptime if is_defined["MOJOLEARN_SVM_TREE_FOLDS"]():
        return False
    return not (column == COLUMN_NVIDIA and width > 512)


comptime SVM_SCHED_TREE = 0
comptime SVM_SCHED_WARP = 1
comptime SVM_SCHED_WARP_LANE0 = 2
comptime SVM_SCHED_FUSED_TREE = 3
comptime SVM_SCHED_RARY_TREE = 4


def svm_block_solve_tree_arity_for[column: Int, width: Int]() -> Int:
    """SCHEDULING row (2026-09-11, DEVIATION 2628): the arity R of SVM_SCHED_RARY_TREE's threadgroup tree (a power of two; levels = ceil(log_R(width))). Selection under a total order, so no bits depend on it. `-D MOJOLEARN_SVM_ARITY_16` / `_64` for an A/B; default 32."""
    comptime if is_defined["MOJOLEARN_SVM_ARITY_16"]():
        return 16
    comptime if is_defined["MOJOLEARN_SVM_ARITY_64"]():
        return 64
    return 32


def svm_block_solve_schedule_for[column: Int, width: Int]() -> Int:
    """SCHEDULING row (2026-09-11, DEVIATIONS 2627 and 2628): which schedule folds the three arg-reductions of `svm/impl/smoblocksolve.mojo::smo_block_solve_kernel[width]`. SVM_SCHED_TREE (0) is the pre-2491 halving trees with a thread ballot for `u` and `l` (42 barriers per inner iteration at width 1024); SVM_SCHED_WARP (1) is DEVIATION 2491's `block_argext` warp butterflies (8); SVM_SCHED_WARP_LANE0 (2) is `block_argext_lane0`, the same butterflies with the cross-warp fold on lane 0 in a runtime loop and a warp broadcast (DEVIATION 2627, 5); SVM_SCHED_FUSED_TREE (3) is one halving tree carrying the argmin, its thread and the argmax together plus a thread-carrying tree for `l`, no ballot (DEVIATION 2628, 26); SVM_SCHED_RARY_TREE (4) carries the same selections on a threadgroup tree of arity `svm_block_solve_tree_arity_for` (DEVIATION 2628's second shape, about 10). All five select the same (value, key) element under a total order with unique keys, so no column's bits depend on this row. Default: `svm_block_solve_warp_folds_for` (DEVIATION 2623) picks WARP or TREE. `-D MOJOLEARN_SVM_SCHED_TREE`, `_WARP`, `_WARP_LANE0`, `_FUSED_TREE` or `_RARY_TREE` forces one schedule on every column for an A/B (the `_RARY_TREE` define was missing from this row at 48f92b19, so that commit's R-ary kernel was unreachable)."""
    comptime if is_defined["MOJOLEARN_SVM_SCHED_TREE"]():
        return SVM_SCHED_TREE
    comptime if is_defined["MOJOLEARN_SVM_SCHED_WARP"]():
        return SVM_SCHED_WARP
    comptime if is_defined["MOJOLEARN_SVM_SCHED_WARP_LANE0"]():
        return SVM_SCHED_WARP_LANE0
    comptime if is_defined["MOJOLEARN_SVM_SCHED_FUSED_TREE"]():
        return SVM_SCHED_FUSED_TREE
    comptime if is_defined["MOJOLEARN_SVM_SCHED_RARY_TREE"]():
        return SVM_SCHED_RARY_TREE
    if svm_block_solve_warp_folds_for[column, width]():
        return SVM_SCHED_WARP
    # DEVIATION 2666 (2026-09-11): the column DEVIATION 2623 sends to the
    # halving trees -- NVIDIA above width 512, where CUDA refuses the warp
    # kernel -- takes the FUSED_TREE schedule instead. Measured on an NVIDIA
    # H200 (RunPod 4oih8bhjepzlmm, driver 570.211.01, ptxas 12.9.86), taxi
    # 10,000 x 11, five fits each: FUSED_TREE 771.1 ms, TREE 866.9 ms,
    # RARY_TREE at arity 16 1,324.4 ms and at 32 1,945.6 ms (1,937.3 ms
    # without its trailing and second update barriers), every arm giving the
    # same fits (n=400/600/2000 457e29b82bca9df9, 733a383c5699f427,
    # 2b66bc991a9c9ed0; taxi b0f91a7958162936) from five different binaries.
    # NVIDIA ONLY: Metal refuses the fused kernel's width-1024 pipeline
    # (threadgroup memory 36872 > 32768, Apple M4 gate 2026-09-11), so this
    # stays a row and never a global default. `-D MOJOLEARN_SVM_TREE_FOLDS`
    # still takes the pre-2491 trees on every column for an A/B.
    comptime if not is_defined["MOJOLEARN_SVM_TREE_FOLDS"]():
        comptime if column == COLUMN_NVIDIA:
            return SVM_SCHED_FUSED_TREE
    return SVM_SCHED_TREE


def umap_device_optimizer_for[column: Int, identical: Bool]() -> Bool:
    """ROUTING row (2026-09-09, lane/umap-optimizer): whether the IDENTICAL UMAP layout optimizer runs on the device (`umap/optimizer_identical_device.mojo`: one thread per vertex, one epoch snapshot, each vertex's update a fixed-order fold over its CSR row, negatives from Philox keyed by (seed, epoch, edge, slot), no atomics, no launch-geometry dependence) instead of the serial host loop (`umap/optimizer.mojo::optimize_layout_identical`, `umap/sparse_optimizer.mojo::optimize_sparse_layout_identical`). The two produce DIFFERENT bits (Jacobi versus Gauss-Seidel order); the device path is the IDENTICAL contract on every column and is gated against itself across launch widths and GPUs, not against the host loop. `-D MOJOLEARN_UMAP_IDENTICAL_HOST_OPTIMIZER=1` restores the host loop on every column (the pre-2026-09-09 cards). FAST and DETERMINISTIC never enter this row."""
    comptime if not identical:
        return False
    comptime if is_defined["MOJOLEARN_UMAP_IDENTICAL_HOST_OPTIMIZER"]():
        return False
    return True


comptime KNN_IDENTICAL_INDEX_TILE = 65536


def knn_index_tile_columns_for[column: Int, identical: Bool]() -> Int:
    """SCHEDULING row: the widest index-axis column tile the IDENTICAL tiled k-NN arm computes per query tile before merging partial top-k lists under the composite total order (`select_smallk_identical_candidate.mojo::partial_topk_merge_kernel`). 0 means the index axis is never split. Merging sorted (distance, index) lists is order-independent, so the bits are equal to the untiled path. `-D MOJOLEARN_KNN_IDENTICAL_NO_INDEX_TILE=1` restores the untiled path."""
    comptime if not identical:
        return 0
    comptime if is_defined["MOJOLEARN_KNN_IDENTICAL_NO_INDEX_TILE"]():
        return 0
    if _knn_identical_round_column(column):
        return KNN_IDENTICAL_INDEX_TILE
    return 0


def gemm_wide_split_for[column: Int]() -> Bool:
    """Execution-only wide split-K tiles on the measured NVIDIA column.

    The 128x128/KS16 tile reduces operand reloads for complete output tiles.
    Other columns keep their previous dispatcher pending local timings;
    every column's all-plan correctness gate still exercises the new tile.
    """
    return column == COLUMN_NVIDIA


def knn_distance_zero_fma_repair_for[column: Int, identical: Bool]() -> Bool:
    """Repair Apple's pre-round FMA underflow only at the kNN register seam.

    The integer slow path runs only for a zero result and restores a rounded
    smallest-normal result when required. NVIDIA already uses round-then-FTZ.
    The exact integer oracle checks 396584 actual/simulated-underflow triples.
    The disable flag retains an explicit Apple before/after performance arm.
    """
    comptime if is_defined["MOJOLEARN_KNN_IDENTICAL_NO_ZERO_FMA_REPAIR"]():
        return False
    return identical and column == COLUMN_APPLE


@always_inline
def knn_distance_preflight_for[column: Int, identical: Bool]() -> Bool:
    """Exact whole-chain exponent admission avoids unnecessary Apple repairs."""
    comptime if is_defined["MOJOLEARN_KNN_IDENTICAL_NO_PREFLIGHT"]():
        return False
    return identical and column == COLUMN_APPLE


@always_inline
def knn_distance_metadata_for[column: Int, identical: Bool]() -> Bool:
    """Force request-local exponent minima outside the measured default scope."""
    comptime if is_defined["MOJOLEARN_EXPERIMENTAL_KNN_PREFLIGHT_METADATA"]():
        return knn_distance_preflight_for[column, identical]()
    return False


@always_inline
def knn_distance_metadata_default_for[column: Int, identical: Bool]() -> Bool:
    """Apple capability; runtime dispatch additionally requires measured shapes."""
    comptime if is_defined["MOJOLEARN_KNN_IDENTICAL_NO_METADATA"]():
        return False
    return knn_distance_preflight_for[column, identical]()


@always_inline
def knn_distance_hardware_flush_for[column: Int, identical: Bool]() -> Bool:
    """Fully rounded NVIDIA FMA followed by exact hardware FTZ multiplication."""
    comptime if is_defined["MOJOLEARN_KNN_IDENTICAL_SOFTWARE_FLUSH"]():
        return False
    return identical and column == COLUMN_NVIDIA


@always_inline
def knn_distance_rows_for[column: Int, identical: Bool]() -> Int:
    """Measured NVIDIA eight-query register tile; each cell keeps its FMA chain.

    The 32/128-feature cases improve; the 8-feature coverage is flat to 0.9%
    slower. Other columns retain four query rows per thread.
    """
    comptime if is_defined["MOJOLEARN_KNN_IDENTICAL_ROWS4"]():
        return 4
    return 8 if identical and column == COLUMN_NVIDIA else 4
