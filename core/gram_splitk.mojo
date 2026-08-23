"""Split-K Gram kernel: `z[m x m] = x[k x m]^T . x[k x m]` with k in the millions.

HAND-WRITTEN, UNDER THE CLOSED-LIBRARY EXCEPTION, WITH THE MEASUREMENT
----------------------------------------------------------------------
Their route is a CLOSED library call: `raft/stats/detail/cov.cuh:65-66` asks
cuBLAS for `CUBLAS_OP_T, CUBLAS_OP_N`, and `raft/linalg/detail/lstsq.cuh:
293-309` asks the same for `covA <- A^T A`. There is no kernel of theirs to
port, so `PORTING_RULES.md 0b-i` says call the MAX equivalent -- and the MAX
equivalent is MEASURED unusable at this shape:

- `bench/results/LANE_covariance-unblock_2026-08-19.md`, orchestrator
  postscript: `linalg.matmul[transpose_b=True]` delivers ~25 GFLOP/s on the
  32 x 32 x 4,000,000 Gram product (322.9 ms, `PHASE pca.gemm_nt_core`)
  against ~248 GFLOP/s on square shapes. The product reads 512 MB once, so
  its bandwidth floor is ~10-15 ms; the vendor route is ~13x off it.
- The cause is in MAX's source (modular checkout, tag `max/v26.5.0`,
  matching the installed max-26.5.0): the Apple fp32 arm is
  `gemm_kernel_apple_8x8` launched with `grid_dim = (ceildiv(n, 64),
  ceildiv(m, 64))` (`max/kernels/src/linalg/matmul/gpu/__init__.mojo:
  663-688`). A 32 x 32 output is ONE threadgroup on a 10-core GPU, and the
  4M-deep reduction runs inside it, serialized.
- MAX ships split-K machinery (`multistage_gemm_split_k_kernel`,
  `SplitKTileScheduler`, `amd_4wave_split_k_matmul`) and ALL of it is
  comptime-gated `not has_apple_gpu_accelerator()`
  (`matmul/gpu/__init__.mojo:725, :1368`); the public `matmul` entry point
  exposes no k-partitioning parameter at all (`matmul/__init__.mojo`). No
  syrk / rank-k update exists anywhere in `linalg`, and
  `linalg.bmm.batched_matmul`'s only Apple GPU arm is
  `naive_batched_matmul_kernel` (`bmm.mojo:899-925`) -- a scalar per-thread
  k-loop with no shared-memory tiling whose B accesses under `transpose_b`
  stride by k floats, uncoalesced -- and expressing the Gram as k-chunked
  batches would ADD a materialized chunk-major copy of X (twice, PORTING.md
  24) plus its write-and-re-read traffic. The MAX routes are exhausted.

WHAT THIS KERNEL DOES
---------------------
Grid over k-chunks: `gram_splitk_chunk_count()` blocks, each owning one
contiguous slice of X's rows. A block streams its slice through a shared
staging tile (`GRAM_ROWS_TILE` rows at a time; the load is the linear copy
of a row-major span, in `GRAM_STAGE_W`-float vectors whenever
`m % GRAM_STAGE_W == 0` -- scalar global loads are measured ~3x slower on
this device -- and scalar at ragged widths) and accumulates the full
m x m partial Gram in registers, `CELLS` cells per thread, fp32. Cell
ownership is a SQUARE REGISTER TILE whenever the tile side
`T = sqrt(CELLS)` divides m (`gram_splitk_reg_tiled`): each thread owns a
T x T rectangle of the output and feeds T*T FMAs from 2T shared reads per
staged row -- the remap `bench/results/GRAM_PROFILE_2026-08-20.md` funded
after eliminating fp32 throughput and DRAM as the limiter at the bench
shape. Ragged widths keep the strided-singles ownership whole. Either way
the accumulation ORDER per cell is k-ascending and identical. Partials
land in a workspace
at `partials[chunk * m*m + cell]`; a second small kernel folds them. X is
read from DRAM EXACTLY ONCE, with no transposes -- the materialized route's
two `transpose_kernel` passes (~22 ms at the bench shape) disappear as well.

DETERMINISM, WHICH THE PAPER CLAIMS RIDE ON
-------------------------------------------
- Within a chunk each cell is one serial fp32 chain in row order.
- The fold is `gram_splitk_reduce_kernel`: one thread per output cell,
  chunk 0 to chunk N-1, ascending, serial. NO device-wide float atomics
  anywhere. Run-to-run bit-identical.
- The chunk count is a fixed function of the hardware constants (no runtime
  query, no atomics-based scheduling), so the same (m, k) always produces
  the same partition ON ONE COLUMN. **That is run-to-run determinism and it
  is NOT cross-vendor identity**, which is the distinction this section used
  to blur: the constants are the TARGET COLUMN's core count and occupancy,
  so two columns compile two different partitions of the k axis, and a
  partition of a float sum is a summation order (IDENTITY_PATHS row 7's
  class). Under `NUMERIC_IDENTICAL` the count is pinned
  (`PINNED_GRAM_SPLITK_CHUNKS`, DEVIATION 520) and the partition is the
  same on every column; under `NUMERIC_FAST` it still fills the machine.
- The per-cell products and the fold are pinned to ONE rounding and ONE
  denormal policy under IDENTICAL (`identical_mul_add`, `ftz`; DEVIATION
  522). Unpinned, `acc + a*b` is one rounding or two AT THE CODEGEN'S WHIM
  and the whim is per backend.
- This kernel is the arm on EVERY column under IDENTICAL, not just Apple's
  (DEVIATION 521). Two columns running two different kernels cannot be bit
  identical however well each is pinned, and the vendor matmul is a closed
  library whose tile shape and k-split are its own business.

BITWISE SYMMETRY, WHICH `check_covariance_is_symmetric` ENFORCES
----------------------------------------------------------------
Cell (i, j) accumulates `tile[r*m + i] * tile[r*m + j]` and cell (j, i)
accumulates the same two loads multiplied in the other order, over the same
rows in the same sequence -- in EVERY ownership arm: the register-tile arm
forms the identical two loads, merely staged through registers, with the
i-operand first exactly as the strided arms multiply. IEEE float multiply
is exactly commutative, so
the two partials are bit-identical, and the reduce folds both in the same
chunk order, so the output is bit-identical across the diagonal by
construction.

THIS KERNEL IS THE APPLE COLUMN'S ARM. The closed-library exception above
is an APPLE fact: MAX's split-K arms are comptime-gated
`not has_apple_gpu_accelerator()`, so on NVIDIA and AMD the gate is open
and the vendor rule resumes. `gram_splitk_applies` consults
`hardware_matrix.gram_splitk_is_target_arm` and answers False for non-Apple
target columns, sending the shape back to `linalg.matmul` through
`gemm_tn_via_transpose`.

ACCURACY. The split sum (chunk partials of ~k/chunks terms, then a fold
over the chunks -- 240 of them on the Apple column) is a two-level
pairwise-style summation: its error grows like
O(k/chunks + chunks) rounding steps instead of the O(k) of one serial fp32
chain, so it is BETTER conditioned than the route it replaces, not worse.
`mojo_only/gram_splitk_check.mojo` proves it against a Float64 host oracle
at shapes that force every chunk live.

THE CENTERED READ: RAFT'S `stable=false` TODO, IMPLEMENTED
----------------------------------------------------------
`raft/stats/detail/cov.cuh:67-69` is the arm they DECLARED and never
shipped: `///@todo: implement this using cutlass + customized epilogue!`
over `ASSERT(false, "cov: Implement stable=false case!")`. cuBLAS exposes
no epilogue hook, so their only shipped arm (`cov.cuh:58-66`,
`stable=true`) mean-centers the input IN PLACE before the GEMM and cuML
adds the mean back after (`pca.cuh:138`). This kernel is hand-written, so
the epilogue exists here: `gram_splitk_partial_centered_kernel` reads
every element as `x[t, j] - mu[j]` into the staging tile -- the SAME fp32
subtraction `shift_columns_kernel` performs and stores, so the fused Gram
is BIT-IDENTICAL to center-then-gemm (`check_gram_centered_fused`, per
cell, `!=`) -- and X is NEVER WRITTEN. Bitwise symmetry is unchanged: the
tile simply holds centered values, and cells (i, j) and (j, i) still
multiply the same two tile loads in commuted order. OPT-IN ONLY:
`gemm_tn`'s dispatch never takes it; `compute_covariance`'s split-K arm is
the one caller, so OLS and tSVD are bit-for-bit untouched. DEVIATION 42.
"""

from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.memory import stack_allocation

from mojo_only.hardware_matrix import gram_splitk_is_target_arm
from mojo_only.kernel_matrix import (
    COLUMN_BIT_IDENTICAL,
    K_LIB_GRAM_SPLITK,
    TARGET_COLUMN,
    lib_block_size_for,
)
from mojo_only.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    ftz_simd,
    identical_mul_add,
)
from neighbors.ported.distance.detail.pairwise_distance_base import (
    TARGET_GPU_CORES,
    max_active_blocks_per_core,
)


comptime GRAM_TPB = lib_block_size_for[K_LIB_GRAM_SPLITK, TARGET_COLUMN]()

#: Rows of X staged in shared memory per barrier. 32 rows keeps the staging
#: tile at `32 * GRAM_MAX_COLS * 4 = 16 KB`, half of Metal's 32 KB
#: per-threadgroup cap, and one barrier per 32 rows instead of one per row.
comptime GRAM_ROWS_TILE = 32

#: Widest Gram this kernel serves. Set by the staging tile: one full row of
#: X must fit a tile line, and `GRAM_ROWS_TILE * GRAM_MAX_COLS` floats is
#: the 16 KB above. Wider outputs fall back to the transpose + matmul route,
#: whose starvation shrinks as the output grows.
comptime GRAM_MAX_COLS = 128

comptime GRAM_STAGE_FLOATS = GRAM_ROWS_TILE * GRAM_MAX_COLS

#: Floats per global load/store in the staging copy's vector arm: a
#: 16-byte `SIMD[float32, 4]`. Scalar global loads cost ~3x on this device
#: (LANE_kmeans-kernel, assignment kernel 63 -> 21 ms from vectorizing
#: reads; upstream's scalar reads lean on NVIDIA warp-coalescing Apple
#: does not replicate). DATA MOVEMENT ONLY: the arm split never touches
#: accumulation arithmetic or order.
comptime GRAM_STAGE_W = 4

#: Chunks per resident-block slot. 2 gives the tail somewhere to hide: with
#: exactly one chunk per slot the slowest block IS the runtime; with two,
#: a finished slot picks up the next chunk. Larger factors shrink chunks
#: (more reduce work) without adding occupancy. SCHEDULING, not numeric --
#: but it feeds the chunk count, which IS numeric (it fixes the summation
#: split), which is why the count is a function below and not a per-call
#: choice: same machine constants, same partition, every run.
comptime GRAM_OVERSUBSCRIBE = 2

#: THE PINNED CHUNK COUNT (IDENTITY_PATHS row 27, DEVIATION 520). Under
#: `NUMERIC_IDENTICAL` the k partition is THIS number on every column
#: instead of the machine-derived one below.
#:
#: The formula under it is `cores x resident-blocks x 2`, which is 240 on
#: the Apple column and a different number on every other one -- so the
#: file's own "fixed means deterministic" claim was true WITHIN a build and
#: false ACROSS vendors, which is precisely IDENTITY_PATHS row 7's class:
#: *a block count is a summation order*, and this one splits the k axis of
#: every Gram product PCA, truncated SVD and OLS compute.
#:
#: 128 rather than any real column's number, deliberately and for the same
#: reason `PINNED_PARTITION_CHUNKS_SM` is 32: no column runs the pinned arm
#: at its own count, so nobody can mistake the pin for a measurement. It is
#: also a power of two, which the reduce does not require and the two-level
#: error bound likes. The cost is a partition that under-fills a large GPU
#: and over-fills a small one, on the opt-in arm only.
#:
#: SAFE FOR THE WORKSPACE BY DIRECTION, not by luck: the partials buffer is
#: `n_chunks * m * m` floats and `gram_splitk_scratch_covers` tests
#: `k >= n_chunks * m`, so pinning DOWN from 240 shrinks the workspace and
#: widens the reuse window. A pin ABOVE any column's count would have to
#: re-argue both.
comptime PINNED_GRAM_SPLITK_CHUNKS = 128

#: THE COLUMN THE ARM DECISION IS READ FROM (DEVIATION 521). ONE definition,
#: two readers -- `gram_splitk_applies` compiles against it and
#: `core/gemm_identity_check.mojo` asserts it -- so the check cannot come to
#: a different conclusion than the kernel it is checking. Under IDENTICAL it
#: is `COLUMN_BIT_IDENTICAL`, whose `gram_splitk_is_target_arm` row already
#: answers True and had simply never been asked; under FAST it is the
#: device's own column, unchanged.
comptime GRAM_SPLITK_RESOLVED_COLUMN = (
    COLUMN_BIT_IDENTICAL if GLOBAL_NUMERIC_MODE
    == NUMERIC_IDENTICAL else TARGET_COLUMN
)

#: Register accumulators per thread in the widest instantiation.
#: `GRAM_TPB * this` caps m*m at 16,384 = 128 x 128, matching GRAM_MAX_COLS.
comptime GRAM_MAX_CELLS_PER_THREAD = 64

#: `gemm_kernel_apple_8x8`'s BLOCK_M = BLOCK_N = 64: the output-tile side of
#: the arm MAX's matmul dispatch takes for fp32 on Apple M1-M4
#: (`max/kernels/src/linalg/matmul/gpu/__init__.mojo:663-688` at
#: `max/v26.5.0`, `grid_dim=(ceildiv(n, 64), ceildiv(m, 64))`). The dispatch
#: predicate below counts the vendor kernel's tiles with it.
comptime VENDOR_MATMUL_TILE = 64


def gram_splitk_chunk_count() raises -> Int:
    """The FIXED number of k-chunks. NUMERIC: this IS the summation split.

    Under `NUMERIC_FAST`, `blocks-that-fit x GRAM_OVERSUBSCRIBE`, computed
    from the SAME target-keyed readers `pairwise_distance_base.mojo`
    exposes (on the Apple column: 10 cores x 3072-thread slots / 256
    threads = 120 resident blocks, x2 = 240, and `hardware_matrix_check`
    pins that 240).

    Under `NUMERIC_IDENTICAL`, `PINNED_GRAM_SPLITK_CHUNKS` -- see that
    constant for the argument and the cost.

    **THE SENTENCE THAT USED TO STAND HERE WAS TRUE OF ONE BUILD AND FALSE
    OF THE CLAIM IT WAS READ AS SUPPORTING.** It said "Fixed means
    deterministic: the summation split never depends on anything measured
    at runtime, and the same build always produces the same partition." Every
    clause of that is correct and none of it is cross-vendor identity: the
    count is fixed at COMPILE time from the TARGET COLUMN's core count and
    occupancy, so two columns compile two different partitions of the k
    axis, and a partition of a float sum is a summation order. The module
    header's DETERMINISM section makes the same move ("the chunk count is a
    fixed function of the hardware constants ... so the same (m, k) always
    produces the same partition") and is corrected there too. Run-to-run
    determinism on one device was never the property in question.

    THE PIN IS AT THIS FUNCTION, not at its callers, because the launch
    geometry AND the `partials` workspace sizing AND
    `gram_splitk_scratch_covers` all read it: pinning one caller would size
    a buffer from one count and index it with another. Same reasoning, same
    shape, as IDENTITY_PATHS finding 3 and `partition_chunks_sm_for`.
    """
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return PINNED_GRAM_SPLITK_CHUNKS
    return (
        TARGET_GPU_CORES
        * max_active_blocks_per_core(GRAM_TPB, GRAM_STAGE_FLOATS * 4)
        * GRAM_OVERSUBSCRIBE
    )


@always_inline
def gram_splitk_stage_vectorized(m: Int) -> Bool:
    """True when the staging copy takes its `GRAM_STAGE_W`-float vector arm.

    `m % GRAM_STAGE_W == 0` makes every chunk's span start (`t * m`), every
    vector offset within it, and the span length multiples of GRAM_STAGE_W,
    so no vector wraps a row remainder and every access is 16-byte aligned
    off the buffer base; every other width takes the scalar arm whole. ONE
    predicate, both readers: the kernel body branches on this and
    `gram_splitk_check` asserts what it decides at the shipped and ragged
    widths, so the check and the kernel cannot drift.
    """
    return (m % GRAM_STAGE_W) == 0


@always_inline
def gram_splitk_cells_for(m: Int) -> Int:
    """The CELLS width the launch dispatch instantiates for an m x m output.

    Smallest of {4, 16, 64} whose `GRAM_TPB * CELLS` covers m*m, so a
    32 x 32 Gram does not pay a 64-cell unroll's dead guards. ONE function,
    three readers (both `_splitk_launch` bodies and `check_gram_dispatch`),
    so the register-tile predicate below can never drift from the width the
    dispatch actually picks.
    """
    if m * m <= GRAM_TPB * 4:
        return 4
    if m * m <= GRAM_TPB * 16:
        return 16
    return GRAM_MAX_CELLS_PER_THREAD


@always_inline
def gram_splitk_reg_tile_side(cells: Int) -> Int:
    """Side of the square register tile that factors one CELLS width.

    2x2, 4x4, 8x8 are exactly the square factorizations of the existing
    {4, 16, 64} cell counts: the accumulator register count per thread is
    UNCHANGED by the register-tile arm; only the shared-read pattern moves.
    """
    if cells == 4:
        return 2
    if cells == 16:
        return 4
    return 8


@always_inline
def gram_splitk_reg_tiled[CELLS: Int](m: Int) -> Bool:
    """True when the accumulation loop takes the register-tile arm.

    The arm needs the tile side to divide m so every thread's rectangle is
    full; every other width keeps the strided-singles arms whole, exactly
    the staging copy's vector/scalar split style. ONE predicate, both
    readers: the kernel body branches on this and `check_gram_dispatch`
    asserts what it decides at the shipped and ragged widths.
    """
    return (m % gram_splitk_reg_tile_side(CELLS)) == 0


def gram_splitk_applies(m: Int, n: Int, k: Int) raises -> Bool:
    """True when `gemm_tn` should take the split-K path.

    THE TARGET DECIDES FIRST, IN ONE PLACE: this kernel is the APPLE
    column's arm only. MAX's matmul has no split-K reachable on Apple (every
    such arm is comptime-gated `not has_apple_gpu_accelerator()`,
    `matmul/gpu/__init__.mojo:725`/`:1368` at `max/v26.5.0`, read in source
    by LANE_gram-splitk), which is why this kernel exists; on NVIDIA and AMD
    that gate is open and the vendor rule resumes -- hand the Gram shape
    back to `linalg.matmul` via `gemm_tn_via_transpose` and let THEIR
    dispatch split k. `hardware_matrix.gram_splitk_is_target_arm` is that
    row; no other site may branch on the vendor.

    THE THRESHOLD IS COMPUTED, NOT A CONSTANT: the vendor matmul parallelizes
    over output tiles of `VENDOR_MATMUL_TILE^2` only, so when the output has
    fewer tiles than the device has resident-block slots
    (`TARGET_GPU_CORES * max_active_blocks_per_core`), the vendor kernel
    cannot fill the machine at ANY k and the k-axis is the only parallelism
    left. That is exactly the split-K kernel's regime. `k` is accepted and
    deliberately unused: starvation is a property of the output shape alone,
    and at small k both routes are microseconds.

    The two capacity bounds are the kernel's own: one staged row per tile
    line (`m <= GRAM_MAX_COLS`) and `m*n` register cells across the block.

    **UNDER `NUMERIC_IDENTICAL` THE COLUMN IS RESOLVED TO
    `COLUMN_BIT_IDENTICAL` AND THE STARVATION TEST IS NOT CONSULTED**
    (IDENTITY_PATHS row 27, DEVIATION 521). Two reasons, and the second is
    the one that is easy to miss:

    1. `gram_splitk_is_target_arm[TARGET_COLUMN]()` reads the DEVICE's
       column unconditionally, so an IDENTICAL build on NVIDIA or AMD
       answered False here and ran `linalg.matmul` -- a closed vendor
       library whose tile shape and k-split are per-vendor, and a k-split
       IS a summation order. Every pin inside this kernel is worthless if
       the other vendor never enters the kernel. That row already answers
       True for `COLUMN_BIT_IDENTICAL` ("the split-K kernel is the arm with
       the determinism guarantee"); the accessor simply never asked it.
       This is IDENTITY_PATHS row 3's defect exactly -- a gate that lived
       at the report while the compiled accessor read the device column --
       and it is fixed the same way, at the accessor, so every caller
       inherits it.
    2. The starvation predicate `vendor_tiles < block_slots` is computed
       from `TARGET_GPU_CORES` and `max_active_blocks_per_core`. Under
       IDENTICAL it would decide the ARM from the machine's size, so a
       10-core box and a 108-SM box would take different kernels at the
       same shape. It is a PERFORMANCE dispatch and it has no business
       deciding identity, so under IDENTICAL the arm is this kernel
       wherever the kernel's own capacity allows.

    Where capacity does NOT allow (`m > GRAM_MAX_COLS`, or `m*n` over the
    register budget), this returns False and `gemm_tn` REFUSES rather than
    quietly handing the shape to the vendor matmul. REFUSE is the third and
    only other legitimate move; a silent fall-through to a closed library
    would be a toggle that returns a non-identical model, which
    IDENTITY_PATHS' opening rule calls worse than no toggle.
    """
    comptime identical = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    comptime if not gram_splitk_is_target_arm[
        GRAM_SPLITK_RESOLVED_COLUMN
    ]():
        # Non-Apple target under FAST: MAX's own split-K machinery exists
        # there, so the vendor matmul is preferred at every shape.
        return False
    if m != n:
        # Two DIFFERENT operand views cannot be a Gram; only the Gram shape
        # ships through gemm_tn, but refuse rather than assume.
        return False
    if m > GRAM_MAX_COLS:
        return False
    if m * n > GRAM_TPB * GRAM_MAX_CELLS_PER_THREAD:
        return False
    comptime if identical:
        # Capacity holds and the column is pinned: this kernel IS the arm.
        # The starvation test below is a performance dispatch keyed to the
        # machine's size and must not decide which kernel an identity build
        # runs. See the docstring's reason 2.
        return True
    var vendor_tiles = (
        (m + VENDOR_MATMUL_TILE - 1) // VENDOR_MATMUL_TILE
    ) * ((n + VENDOR_MATMUL_TILE - 1) // VENDOR_MATMUL_TILE)
    var block_slots = TARGET_GPU_CORES * max_active_blocks_per_core(
        GRAM_TPB, GRAM_STAGE_FLOATS * 4
    )
    return vendor_tiles < block_slots


@always_inline
def _gram_splitk_partial_body[CELLS: Int, CENTERED: Bool](
    partials: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    mu: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    k_in: Int32,
    chunk_rows_in: Int32,
):
    """One block = one k-chunk's full m x m partial Gram, fp32 in registers.

    A chunk whose slice starts at or past k writes an all-zero partial, so a
    fixed grid of `gram_splitk_chunk_count()` blocks is correct at every k,
    including k smaller than the chunk count (then `chunk_rows = 1` and the
    trailing chunks are empty) and k not a multiple of the chunk size (the
    last live chunk is short: `t_end` is clamped to k).

    `CENTERED` is the fused-epilogue arm (module header): the staging-tile
    load subtracts `mu[column]` in registers, so BOTH operand reads of every
    product see centered values and X is never written. `mu` is read on that
    arm only; the plain entry passes a dead pointer the `comptime if`
    eliminates.
    """
    var m = Int(m_in)
    var k = Int(k_in)
    var kc = Int(chunk_rows_in)
    var mn = m * m
    var chunk = Int(block_idx.x)
    var tid = Int(thread_idx.x)

    var tile = stack_allocation[
        GRAM_STAGE_FLOATS,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()

    # REGISTER-TILE cell ownership (GRAM_PROFILE_2026-08-20.md: at the bench
    # shape neither fp32 throughput nor DRAM binds; the limiter by
    # elimination is the accumulation loop's shared-read rate). When the
    # tile side T divides m, each thread owns a T x T RECTANGLE of the
    # output instead of CELLS strided singles: per staged row it loads the
    # rectangle's T column values and T row values from shared into
    # registers and does T*T FMAs -- T/2 FMA per shared read (2x2 = 1.0,
    # 4x4 = 2.0, 8x8 = 4.0) against the strided singles' ~0.5. T*T == CELLS
    # exactly, so the accumulator count per thread is unchanged. The remap
    # moves WHICH thread owns a cell, never the k-ascending order that
    # cell's products are added in, and each partial still lands at the
    # same `chunk * mn + i*m + j` slot, so writer and reader agree and the
    # output is bit-identical (before/after FNV dump,
    # bench/results/LANE_gram-tile_2026-08-20.md, not argued). Ragged m
    # keeps the strided arms below, whole -- the staging copy's
    # vector/scalar split style. Block-uniform in m: no divergent barrier.
    comptime T = gram_splitk_reg_tile_side(CELLS)
    var tiled = gram_splitk_reg_tiled[CELLS](m)
    var mt = 1
    var bi0 = 0
    var bj0 = 0
    var own_tile = False
    if tiled:
        mt = m // T  # rectangles per side; tiled implies m >= T, so mt >= 1
        # `mt * mt = m*m / CELLS <= GRAM_TPB` under the launch dispatch, so
        # every rectangle has an owning thread and the m x m output is
        # covered exactly once.
        own_tile = tid < mt * mt
        if own_tile:
            bi0 = (tid // mt) * T
            bj0 = (tid % mt) * T

    # Cell coordinates are invariant over the whole row loop; hoist them.
    # Strided arms only: the tiled arm indexes off (bi0, bj0) and must not
    # carry 2 * CELLS dead int32 lanes of register pressure.
    var ii = SIMD[DType.int32, CELLS](0)
    var jj = SIMD[DType.int32, CELLS](0)
    if not tiled:
        comptime for c in range(CELLS):
            var cell = tid + c * GRAM_TPB
            if cell < mn:
                ii[c] = Int32(cell // m)
                jj[c] = Int32(cell % m)

    var acc = SIMD[DType.float32, CELLS](0.0)

    # FLOOR KNOB (SCOREBOARD_2026-08-19 item 5, first knob). When the block's
    # thread count is an exact multiple of m AND the register-tile arm
    # declined the shape, `jj[c] = (tid + c * GRAM_TPB) % m` is the SAME
    # index for every c, so the second tile read of the inner unroll is ONE
    # load per row, hoisted out of the comptime loop, instead of CELLS loads
    # of the same address. Since the register-tile arm took every width
    # whose tile side divides m, this arm now serves only m = 1 among the
    # 256-dividing widths; it stays because it is still the better read
    # pattern wherever it applies. Same address between the same barriers,
    # same value, same multiply in the same order: bit-identical, proven by
    # the before/after bit dump in
    # bench/results/LANE_pca-centering_2026-08-20.md, not argued. The branch
    # is block-uniform, so no barrier is divergent.
    var uniform_jj = (GRAM_TPB % m) == 0
    var jm = tid % m

    # The staging copy's width, decided once: block-uniform in m, so
    # neither barrier below is divergent.
    var stage_vec = gram_splitk_stage_vectorized(m)

    var t = chunk * kc
    var t_end = t + kc
    if t_end > k:
        t_end = k
    while t < t_end:
        var rows = t_end - t
        if rows > GRAM_ROWS_TILE:
            rows = GRAM_ROWS_TILE
        # Rows t..t+rows of row-major (k x m) are one contiguous span, so
        # the cooperative load is a linear, coalesced copy. It is a PURE
        # data movement: whichever arm runs, the SAME values land in the
        # SAME tile slots between the SAME barriers, so the accumulation
        # below sees identical inputs in an identical order and the output
        # is bit-identical across arms (FNV bit dump,
        # bench/results/LANE_splitk-interior_2026-08-20.md, plus the
        # destructive reach probe recorded there).
        var span = rows * m
        if stage_vec:
            # Vector arm (`gram_splitk_stage_vectorized`): GRAM_STAGE_W
            # floats per load/store. `t * m`, `e`, and `span` are all
            # multiples of GRAM_STAGE_W here, so every access is 16-byte
            # aligned off the buffer base and no vector splits a row.
            var nvec = span // GRAM_STAGE_W
            var vi = tid
            while vi < nvec:
                var e = vi * GRAM_STAGE_W
                var v = x.unsafe_load[width=GRAM_STAGE_W](t * m + e)
                comptime if CENTERED:
                    # DEVIATION 522 / IDENTITY_PATHS row 10: the centered
                    # value is a SEAM -- it is written into the staging tile
                    # for every other thread of the block to read -- so both
                    # operands and the difference are flushed under
                    # IDENTICAL. Metal flushes these in hardware and CUDA's
                    # default does not, so without this a denormal column
                    # mean or a near-cancelling row makes the two columns
                    # stage different tiles from the same input. `ftz_simd`
                    # is bitwise inert on an FTZ backend and compiles away
                    # entirely under FAST.
                    v = ftz_simd[GRAM_STAGE_W](v)
                    # Element e sits in column `e % m` (t * m is a multiple
                    # of m), and columns `e % m .. e % m + GRAM_STAGE_W - 1`
                    # never wrap: e % m and m are both multiples of
                    # GRAM_STAGE_W. The per-lane fp32 subtract is the same
                    # IEEE op the scalar arm performs, lane by lane.
                    v = ftz_simd[GRAM_STAGE_W](
                        v - ftz_simd[GRAM_STAGE_W](
                            mu.unsafe_load[width=GRAM_STAGE_W](e % m)
                        )
                    )
                tile.unsafe_store(e, v)
                vi += GRAM_TPB
        else:
            var i = tid
            while i < span:
                comptime if CENTERED:
                    # `t * m` is a multiple of m, so element i of the span
                    # sits in column `i % m`. This is the SAME fp32
                    # subtraction `shift_columns_kernel` performs and
                    # stores (its runtime `x + (-1.0) * mu` is bitwise
                    # `x - mu`: the multiply by -1.0 is an exact sign flip
                    # and IEEE `a + (-b)` IS `a - b`), landed in the shared
                    # tile instead of back in DRAM. Bit-identity is proven
                    # per cell by `check_gram_centered_fused`, not argued.
                    # DEVIATION 522 / row 10, the scalar twin of the vector
                    # arm above: operands and difference flushed, so the
                    # two staging arms stay the bit-identical pair the
                    # module header claims they are on EVERY column and not
                    # only on one that flushes in hardware.
                    tile[i] = ftz(
                        ftz(x.unsafe_load(t * m + i))
                        - ftz(mu.unsafe_load(i % m))
                    )
                else:
                    tile[i] = x.unsafe_load(t * m + i)
                i += GRAM_TPB
        barrier()
        if tiled:
            # Register-tile arm: T column values into `cv`, then per owned
            # row one scalar `xi`; every product is the SAME
            # `tile[base + i] * tile[base + j]` (i-operand first) the
            # strided arms form, read through registers -- a load cannot
            # change a value, and each cell's chain still runs r ascending
            # within the tile and t ascending across tiles.
            if own_tile:
                for r in range(rows):
                    var base = r * m
                    var cv = SIMD[DType.float32, T](0.0)
                    comptime for b in range(T):
                        cv[b] = tile[base + bj0 + b]
                    comptime for a in range(T):
                        var xi = tile[base + bi0 + a]
                        comptime for b in range(T):
                            # DEVIATION 522 / IDENTITY_PATHS row 9. `acc +
                            # xi*cv` is ONE rounding or TWO at the codegen's
                            # whim and the whim is per backend, so the pin
                            # is at the source: `fma` under IDENTICAL, the
                            # naive chain under FAST. Bit-inert on a
                            # backend that already contracts (Metal does --
                            # row 9's 2026-08-23 correction), which is
                            # exactly why "Apple's bits did not move" is
                            # not evidence the pin is unreached.
                            acc[a * T + b] = identical_mul_add(
                                xi, cv[b], acc[a * T + b]
                            )
        elif uniform_jj:
            for r in range(rows):
                var base = r * m
                var xj = tile[base + jm]
                comptime for c in range(CELLS):
                    if tid + c * GRAM_TPB < mn:
                        # DEVIATION 522 / row 9, same seam, uniform-jj arm.
                        acc[c] = identical_mul_add(
                            tile[base + Int(ii[c])], xj, acc[c]
                        )
        else:
            for r in range(rows):
                var base = r * m
                comptime for c in range(CELLS):
                    if tid + c * GRAM_TPB < mn:
                        # DEVIATION 522 / row 9, same seam, strided arm.
                        # The i-operand stays FIRST in every arm: the
                        # module's BITWISE SYMMETRY argument rests on cells
                        # (i,j) and (j,i) forming the same two loads in
                        # commuted order, and `fma(a,b,c)` is exactly as
                        # commutative in a and b as `a*b` was.
                        acc[c] = identical_mul_add(
                            tile[base + Int(ii[c])],
                            tile[base + Int(jj[c])],
                            acc[c],
                        )
        barrier()
        t += rows

    # The writeback mirrors the ownership arm, and BOTH arms address a cell
    # as `chunk * mn + i*m + j`: the reduce reads the identical slot per
    # cell whichever arm produced it, in the same ascending-chunk order.
    # DEVIATION 522 / IDENTITY_PATHS row 10: `partials` is the seam between
    # this kernel and the reduce, so every stored value is flushed under
    # IDENTICAL. A denormal partial is not exotic here -- it is one chunk's
    # slice of a centered column, which is where cancellation lives.
    if tiled:
        if own_tile:
            comptime for a in range(T):
                comptime for b in range(T):
                    var cell = (bi0 + a) * m + (bj0 + b)
                    partials.unsafe_store(
                        chunk * mn + cell, ftz(acc[a * T + b])
                    )
    else:
        comptime for c in range(CELLS):
            var cell = tid + c * GRAM_TPB
            if cell < mn:
                partials.unsafe_store(chunk * mn + cell, ftz(acc[c]))


def gram_splitk_partial_kernel[CELLS: Int](
    partials: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    k_in: Int32,
    chunk_rows_in: Int32,
):
    """The plain entry: the Gram of X as stored. `x` fills the body's dead
    `mu` slot; the `CENTERED=False` instantiation never reads it."""
    _gram_splitk_partial_body[CELLS, False](
        partials, x, x, m_in, k_in, chunk_rows_in
    )


def gram_splitk_partial_centered_kernel[CELLS: Int](
    partials: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    mu: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    k_in: Int32,
    chunk_rows_in: Int32,
):
    """The fused-epilogue entry: the Gram of `X - 1 mu^T`, X read-only.

    RAFT's `stable=false` TODO (`raft/stats/detail/cov.cuh:67-69`),
    implemented -- see the module header. DEVIATION 42.
    """
    _gram_splitk_partial_body[CELLS, True](
        partials, x, mu, m_in, k_in, chunk_rows_in
    )


def gram_splitk_reduce_kernel(
    z: MutPointer[Float32, MutAnyOrigin],
    partials: MutPointer[Float32, MutAnyOrigin],
    mn_in: Int32,
    n_chunks_in: Int32,
):
    """One thread per output cell folds the partials chunk 0..N-1, serial.

    Ascending fixed order, no atomics: the determinism of the whole path
    rests on this loop's order, so do not "improve" it into a tree that
    depends on launch geometry, and never into `Atomic.fetch_add`.

    **This loop is the reason the whole path is batch-invariant.** The fold
    is over `n_chunks`, which is a pure function of the pinned constant
    under IDENTICAL, and the per-cell order is the SAME sequence whatever
    grid the launch chose, because a cell's owner reads every chunk itself
    rather than combining whatever partials happened to arrive. Nothing
    here consults the block index, the block size, the lane width or the
    device.

    DEVIATION 522 / IDENTITY_PATHS row 10: each loaded partial and each
    running sum is flushed under IDENTICAL. Unlike the accumulation loop
    inside the partial kernel, this fold is only `n_chunks` steps long, so
    flushing every intermediate is affordable and there is no reason to
    accept a named residue where the exact model is cheap.
    """
    var mn = Int(mn_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= mn:
        return
    var acc = Float32(0.0)
    for c in range(Int(n_chunks_in)):
        acc = ftz(acc + ftz(partials.unsafe_load(c * mn + cell)))
    z.unsafe_store(cell, ftz(acc))


def _enqueue_partial[CELLS: Int](
    ctx: DeviceContext,
    mut partials: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    m: Int,
    k: Int,
    kc: Int,
    n_chunks: Int,
) raises:
    """One width instantiation of the PLAIN partial kernel."""
    comptime kern = gram_splitk_partial_kernel[CELLS]
    ctx.enqueue_function[kern](
        partials.unsafe_ptr(),
        x.unsafe_ptr(),
        Int32(m),
        Int32(k),
        Int32(kc),
        grid_dim=(n_chunks, 1, 1),
        block_dim=(GRAM_TPB, 1, 1),
    )


def _enqueue_partial_centered[CELLS: Int](
    ctx: DeviceContext,
    mut partials: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    mut mu: DeviceBuffer[DType.float32],
    m: Int,
    k: Int,
    kc: Int,
    n_chunks: Int,
) raises:
    """One width instantiation of the CENTERED partial kernel."""
    comptime kern = gram_splitk_partial_centered_kernel[CELLS]
    ctx.enqueue_function[kern](
        partials.unsafe_ptr(),
        x.unsafe_ptr(),
        mu.unsafe_ptr(),
        Int32(m),
        Int32(k),
        Int32(kc),
        grid_dim=(n_chunks, 1, 1),
        block_dim=(GRAM_TPB, 1, 1),
    )


def _splitk_chunk_rows(k: Int, n_chunks: Int) -> Int:
    var kc = (k + n_chunks - 1) // n_chunks
    if kc < 1:
        kc = 1
    return kc


def _enqueue_reduce(
    ctx: DeviceContext,
    mut z: DeviceBuffer[DType.float32],
    mut partials: DeviceBuffer[DType.float32],
    mn: Int,
    n_chunks: Int,
) raises:
    ctx.enqueue_function[gram_splitk_reduce_kernel](
        z.unsafe_ptr(),
        partials.unsafe_ptr(),
        Int32(mn),
        Int32(n_chunks),
        grid_dim=((mn + GRAM_TPB - 1) // GRAM_TPB, 1, 1),
        block_dim=(GRAM_TPB, 1, 1),
    )


def _splitk_launch(
    ctx: DeviceContext,
    mut z: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    mut partials: DeviceBuffer[DType.float32],
    m: Int,
    k: Int,
) raises:
    """Enqueue the plain partial + reduce pair into a CALLER-OWNED workspace.

    `partials` must hold at least `gram_splitk_chunk_count() * m * m`
    floats. The caller synchronizes; nothing here waits. Which buffer the
    partials land in is SCHEDULING, not numerics: the kernels, the chunk
    partition and the ascending fold are identical whoever owns the memory.
    """
    var n_chunks = gram_splitk_chunk_count()
    var kc = _splitk_chunk_rows(k, n_chunks)
    var mn = m * m

    # Three register widths, smallest that fits (`gram_splitk_cells_for`,
    # one function with the checks), so a 32 x 32 Gram does not pay a
    # 64-cell unroll's dead guards. The instantiation changes WHICH thread
    # owns which cell, never the order any cell is accumulated in, so it is
    # scheduling, not numerics.
    var cells = gram_splitk_cells_for(m)
    if cells == 4:
        _enqueue_partial[4](ctx, partials, x, m, k, kc, n_chunks)
    elif cells == 16:
        _enqueue_partial[16](ctx, partials, x, m, k, kc, n_chunks)
    else:
        _enqueue_partial[GRAM_MAX_CELLS_PER_THREAD](
            ctx, partials, x, m, k, kc, n_chunks
        )
    _enqueue_reduce(ctx, z, partials, mn, n_chunks)


def _splitk_launch_centered(
    ctx: DeviceContext,
    mut z: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    mut mu: DeviceBuffer[DType.float32],
    mut partials: DeviceBuffer[DType.float32],
    m: Int,
    k: Int,
) raises:
    """`_splitk_launch` with the centered tile read (module header;
    DEVIATION 42). Same width dispatch, same reduce, X read-only."""
    var n_chunks = gram_splitk_chunk_count()
    var kc = _splitk_chunk_rows(k, n_chunks)
    var mn = m * m
    var cells = gram_splitk_cells_for(m)
    if cells == 4:
        _enqueue_partial_centered[4](
            ctx, partials, x, mu, m, k, kc, n_chunks
        )
    elif cells == 16:
        _enqueue_partial_centered[16](
            ctx, partials, x, mu, m, k, kc, n_chunks
        )
    else:
        _enqueue_partial_centered[GRAM_MAX_CELLS_PER_THREAD](
            ctx, partials, x, mu, m, k, kc, n_chunks
        )
    _enqueue_reduce(ctx, z, partials, mn, n_chunks)


def gram_splitk_scratch_covers(m: Int, k: Int) raises -> Bool:
    """Whether a `k * m`-float scratch buffer (the `xt` contract every
    `gemm_tn` caller already meets) covers the `n_chunks * m * m` partials
    workspace. `k * m >= n_chunks * m * m  <=>  k >= n_chunks * m`; at the
    bench's m = 32 that is k >= 7,680, so every shipped tall-skinny fit
    reuses the scratch and never allocates on the fit path. A pure function
    of (m, k): the same shape always lands the partials in the same place.
    """
    return k >= gram_splitk_chunk_count() * m


def gemm_tn_splitk(
    ctx: DeviceContext,
    mut z: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    m: Int,
    k: Int,
) raises:
    """`z[m x m] = x[k x m]^T . x[k x m]` on the split-K kernel pair,
    allocating its own partials workspace (`n_chunks * m * m` floats, at
    most 15.7 MB at m = 128 and 240 KB at the bench's m = 32), freed at
    scope exit after the synchronize. The direct-call entry the checks
    exercise by name; `gemm_tn` goes through `gemm_tn_splitk_into` so the
    shipped fits reuse the `xt` scratch instead (FLOOR KNOB,
    SCOREBOARD_2026-08-19 item 5, second knob).
    """
    var n_chunks = gram_splitk_chunk_count()
    var partials = ctx.enqueue_create_buffer[DType.float32](n_chunks * m * m)
    _splitk_launch(ctx, z, x, partials, m, k)
    ctx.synchronize()


def gemm_tn_splitk_into(
    ctx: DeviceContext,
    mut z: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    mut scratch: DeviceBuffer[DType.float32],
    m: Int,
    k: Int,
) raises:
    """`gemm_tn_splitk`, but the partials land in `scratch` when it covers
    them (`gram_splitk_scratch_covers`), skipping the per-call
    `enqueue_create_buffer`. `scratch` is `gemm_tn`'s `xt` alias buffer:
    at least `k * m` floats, pure scratch on this arm (the transpose arm
    overwrites it wholesale too, so no caller may rely on its contents).
    RAFT/cuML hand workspace buffers down the call chain the same way
    rather than allocating inside a GEMM; small k, where `k * m` does not
    cover `n_chunks * m * m` (PCA's own 4-column checks), still allocates.
    """
    if gram_splitk_scratch_covers(m, k):
        _splitk_launch(ctx, z, x, scratch, m, k)
        ctx.synchronize()
        return
    gemm_tn_splitk(ctx, z, x, m, k)


def gram_centered_splitk(
    ctx: DeviceContext,
    mut z: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    mut mu: DeviceBuffer[DType.float32],
    m: Int,
    k: Int,
) raises:
    """`z[m x m] = (X - 1 mu^T)^T (X - 1 mu^T)` with the centering FUSED
    into the split-K read. X IS NEVER WRITTEN.

    RAFT's own declared design for `cov`'s `stable=false` arm, which they
    never shipped (`raft/stats/detail/cov.cuh:67-69`: the cutlass-epilogue
    @todo over `ASSERT(false, ...)`; module header; DEVIATION 42).
    Bit-identical to center-then-`gemm_tn_splitk` because the tile load
    performs the exact fp32 subtraction `shift_columns_kernel` would have
    stored -- proven per cell by `check_gram_centered_fused`. OPT-IN:
    `gemm_tn` never dispatches here; `compute_covariance`'s split-K arm is
    the one caller, so OLS/tSVD stay bit-for-bit on the plain kernel.
    """
    var n_chunks = gram_splitk_chunk_count()
    var partials = ctx.enqueue_create_buffer[DType.float32](n_chunks * m * m)
    _splitk_launch_centered(ctx, z, x, mu, partials, m, k)
    ctx.synchronize()


def gram_centered_splitk_into(
    ctx: DeviceContext,
    mut z: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    mut mu: DeviceBuffer[DType.float32],
    mut scratch: DeviceBuffer[DType.float32],
    m: Int,
    k: Int,
) raises:
    """`gram_centered_splitk`, partials landing in `scratch` (>= k * m
    floats, pure scratch) when `gram_splitk_scratch_covers`, exactly as
    `gemm_tn_splitk_into` does for the plain arm; small k allocates."""
    if gram_splitk_scratch_covers(m, k):
        _splitk_launch_centered(ctx, z, x, mu, scratch, m, k)
        ctx.synchronize()
        return
    gram_centered_splitk(ctx, z, x, mu, m, k)
