# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""One table: every kernel, every tunable it takes, per GPU vendor."""

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
    COLUMN_AMD if has_amd_gpu_accelerator() else
    COLUMN_NVIDIA if has_nvidia_gpu_accelerator() else
    COLUMN_APPLE
)


comptime DETECTED_COLUMN = (
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
    """SCHEDULING row (DEVIATION 1907): whether the leaf reorder's stable one-bit partition may take the SINGLE-PASS decoupled-lookback path (`gbdt/gpu_util/kernel/reorder_single_pass.mojo`) for a level whose leaf bound is above CatBoost's `FastSortSize()` == 500,000 rows."""
    comptime if identical:
        return False
    comptime if is_defined["MOJOLEARN_2042_FAST_NO_LOOKBACK"]():
        return False
    comptime if column == COLUMN_AMD:
        return False
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


def knn_warpsort_select_for[column: Int, identical: Bool]() -> Bool:
    """SCHEDULING row (DEVIATION 1922): whether the k-NN TILED path's selector is the ported RAFT WARPSORT (`select_warpsort.mojo`, `warpsort_topk_block_kernel`) instead of the ported RAFT radix (`select_radix.mojo`) for `2 < k <= 256`."""
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
