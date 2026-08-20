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
of a row-major span, coalesced) and accumulates the full m x m partial Gram
in registers, `CELLS` cells per thread, fp32. Partials land in a workspace
at `partials[chunk * m*m + cell]`; a second small kernel folds them. X is
read from DRAM EXACTLY ONCE, with no transposes -- the materialized route's
two `transpose_kernel` passes (~22 ms at the bench shape) disappear as well.

DETERMINISM, WHICH THE PAPER CLAIMS RIDE ON
-------------------------------------------
- The chunk count is a fixed function of the hardware constants (no runtime
  query, no atomics-based scheduling), so the same (m, k) always produces
  the same partition.
- Within a chunk each cell is one serial fp32 chain in row order.
- The fold is `gram_splitk_reduce_kernel`: one thread per output cell,
  chunk 0 to chunk N-1, ascending, serial. NO device-wide float atomics
  anywhere. Run-to-run bit-identical.

BITWISE SYMMETRY, WHICH `check_covariance_is_symmetric` ENFORCES
----------------------------------------------------------------
Cell (i, j) accumulates `tile[r*m + i] * tile[r*m + j]` and cell (j, i)
accumulates the same two loads multiplied in the other order, over the same
rows in the same sequence. IEEE float multiply is exactly commutative, so
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
"""

from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.memory import stack_allocation

from mojo_only.hardware_matrix import gram_splitk_is_target_arm
from mojo_only.kernel_matrix import (
    K_LIB_GRAM_SPLITK,
    TARGET_COLUMN,
    lib_block_size_for,
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

#: Chunks per resident-block slot. 2 gives the tail somewhere to hide: with
#: exactly one chunk per slot the slowest block IS the runtime; with two,
#: a finished slot picks up the next chunk. Larger factors shrink chunks
#: (more reduce work) without adding occupancy. SCHEDULING, not numeric --
#: but it feeds the chunk count, which IS numeric (it fixes the summation
#: split), which is why the count is a function below and not a per-call
#: choice: same machine constants, same partition, every run.
comptime GRAM_OVERSUBSCRIBE = 2

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
    """The FIXED number of k-chunks, from the machine constants.

    `blocks-that-fit x GRAM_OVERSUBSCRIBE`, computed from the SAME
    target-keyed readers `pairwise_distance_base.mojo` exposes (on the Apple
    column: 10 cores x 3072-thread slots / 256 threads = 120 resident
    blocks, x2 = 240, and `hardware_matrix_check` pins that 240). Fixed
    means deterministic: the summation split never depends on anything
    measured at runtime, and the same build always produces the same
    partition.
    """
    return (
        TARGET_GPU_CORES
        * max_active_blocks_per_core(GRAM_TPB, GRAM_STAGE_FLOATS * 4)
        * GRAM_OVERSUBSCRIBE
    )


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
    """
    comptime if not gram_splitk_is_target_arm[TARGET_COLUMN]():
        # Non-Apple target: MAX's own split-K machinery exists there, so
        # the vendor matmul is preferred at every shape.
        return False
    if m != n:
        # Two DIFFERENT operand views cannot be a Gram; only the Gram shape
        # ships through gemm_tn, but refuse rather than assume.
        return False
    if m > GRAM_MAX_COLS:
        return False
    if m * n > GRAM_TPB * GRAM_MAX_CELLS_PER_THREAD:
        return False
    var vendor_tiles = (
        (m + VENDOR_MATMUL_TILE - 1) // VENDOR_MATMUL_TILE
    ) * ((n + VENDOR_MATMUL_TILE - 1) // VENDOR_MATMUL_TILE)
    var block_slots = TARGET_GPU_CORES * max_active_blocks_per_core(
        GRAM_TPB, GRAM_STAGE_FLOATS * 4
    )
    return vendor_tiles < block_slots


def gram_splitk_partial_kernel[CELLS: Int](
    partials: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
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

    # Cell coordinates are invariant over the whole row loop; hoist them.
    var ii = SIMD[DType.int32, CELLS](0)
    var jj = SIMD[DType.int32, CELLS](0)
    comptime for c in range(CELLS):
        var cell = tid + c * GRAM_TPB
        if cell < mn:
            ii[c] = Int32(cell // m)
            jj[c] = Int32(cell % m)

    var acc = SIMD[DType.float32, CELLS](0.0)

    # FLOOR KNOB (SCOREBOARD_2026-08-19 item 5, first knob). When the block's
    # thread count is an exact multiple of m -- true at every power-of-two
    # width up to 128, including the shipped m = 32 -- then
    # `jj[c] = (tid + c * GRAM_TPB) % m` is the SAME index for every c, so
    # the second tile read of the inner unroll is ONE load per row, hoisted
    # out of the comptime loop, instead of CELLS loads of the same address.
    # Same address between the same barriers, same value, same multiply in
    # the same order: bit-identical, proven by the before/after bit dump in
    # bench/results/LANE_pca-centering_2026-08-20.md, not argued. The branch
    # is block-uniform, so no barrier is divergent.
    var uniform_jj = (GRAM_TPB % m) == 0
    var jm = tid % m

    var t = chunk * kc
    var t_end = t + kc
    if t_end > k:
        t_end = k
    while t < t_end:
        var rows = t_end - t
        if rows > GRAM_ROWS_TILE:
            rows = GRAM_ROWS_TILE
        # Rows t..t+rows of row-major (k x m) are one contiguous span, so
        # the cooperative load is a linear, coalesced copy.
        var span = rows * m
        var i = tid
        while i < span:
            tile[i] = x.unsafe_load(t * m + i)
            i += GRAM_TPB
        barrier()
        if uniform_jj:
            for r in range(rows):
                var base = r * m
                var xj = tile[base + jm]
                comptime for c in range(CELLS):
                    if tid + c * GRAM_TPB < mn:
                        acc[c] = acc[c] + tile[base + Int(ii[c])] * xj
        else:
            for r in range(rows):
                var base = r * m
                comptime for c in range(CELLS):
                    if tid + c * GRAM_TPB < mn:
                        acc[c] = (
                            acc[c]
                            + tile[base + Int(ii[c])] * tile[base + Int(jj[c])]
                        )
        barrier()
        t += rows

    comptime for c in range(CELLS):
        var cell = tid + c * GRAM_TPB
        if cell < mn:
            partials.unsafe_store(chunk * mn + cell, acc[c])


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
    """
    var mn = Int(mn_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= mn:
        return
    var acc = Float32(0.0)
    for c in range(Int(n_chunks_in)):
        acc += partials.unsafe_load(c * mn + cell)
    z.unsafe_store(cell, acc)


def _splitk_launch(
    ctx: DeviceContext,
    mut z: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    mut partials: DeviceBuffer[DType.float32],
    m: Int,
    k: Int,
) raises:
    """Enqueue the partial + reduce pair into a CALLER-OWNED workspace.

    `partials` must hold at least `gram_splitk_chunk_count() * m * m`
    floats. The caller synchronizes; nothing here waits. Which buffer the
    partials land in is SCHEDULING, not numerics: the kernels, the chunk
    partition and the ascending fold are identical whoever owns the memory.
    """
    var n_chunks = gram_splitk_chunk_count()
    var kc = (k + n_chunks - 1) // n_chunks
    if kc < 1:
        kc = 1
    var mn = m * m

    # Three register widths, smallest that fits, so a 32 x 32 Gram does not
    # pay a 64-cell unroll's dead guards. The instantiation changes WHICH
    # thread owns which cell, never the order any cell is accumulated in,
    # so it is scheduling, not numerics.
    if mn <= GRAM_TPB * 4:
        comptime kernel4 = gram_splitk_partial_kernel[4]
        ctx.enqueue_function[kernel4](
            partials.unsafe_ptr(),
            x.unsafe_ptr(),
            Int32(m),
            Int32(k),
            Int32(kc),
            grid_dim=(n_chunks, 1, 1),
            block_dim=(GRAM_TPB, 1, 1),
        )
    elif mn <= GRAM_TPB * 16:
        comptime kernel16 = gram_splitk_partial_kernel[16]
        ctx.enqueue_function[kernel16](
            partials.unsafe_ptr(),
            x.unsafe_ptr(),
            Int32(m),
            Int32(k),
            Int32(kc),
            grid_dim=(n_chunks, 1, 1),
            block_dim=(GRAM_TPB, 1, 1),
        )
    else:
        comptime kernel64 = gram_splitk_partial_kernel[
            GRAM_MAX_CELLS_PER_THREAD
        ]
        ctx.enqueue_function[kernel64](
            partials.unsafe_ptr(),
            x.unsafe_ptr(),
            Int32(m),
            Int32(k),
            Int32(kc),
            grid_dim=(n_chunks, 1, 1),
            block_dim=(GRAM_TPB, 1, 1),
        )

    ctx.enqueue_function[gram_splitk_reduce_kernel](
        z.unsafe_ptr(),
        partials.unsafe_ptr(),
        Int32(mn),
        Int32(n_chunks),
        grid_dim=((mn + GRAM_TPB - 1) // GRAM_TPB, 1, 1),
        block_dim=(GRAM_TPB, 1, 1),
    )


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
