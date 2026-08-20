"""`launchConfigGenerator`: the grid shape is COMPUTED, never a constant.

PORT OF `cuvs/cpp/src/distance/detail/pairwise_distance_base.cuh:295-322`
at cuVS `94c2819`. Partial (the `PairwiseDistances` struct itself is inlined
into `fused_l2_knn.mojo`; this file is its launch computation). Do not
improve.

Their computation, transcribed branch for branch below:

    numSMs        = cudaDevAttrMultiProcessorCount          (:300-301)
    numBlocksPerSm= cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                        func, Nthreads, sMemSize)            (:306-307)
    minGridSize   = numSMs * numBlocksPerSm                  (:308)
    yChunks       = ceildiv(m, Mblk); xChunks = ceildiv(n, Nblk)
    grid.y        = yChunks > minGridSize ? minGridSize : yChunks
    grid.x        = (minGridSize - grid.y) <= 0 ? 1 : xChunks
    if (grid.x != 1) { i = 1; while (grid.y * i < minGridSize) i++;
                       grid.x = i >= xChunks ? xChunks : i; }

The point of the shape: fill the device. `grid.y` takes as many row tiles
as fit; only when the row tiles alone cannot occupy the device does the
grid split the COLUMN axis (`grid.x > 1`), and then only by the smallest
factor that reaches `minGridSize`. Every block being resident is also what
makes the `gridDim.x > 1` mutex handoff in `fusedL2kNN` deadlock-free: a
spinning producer's consumer is always running. So this computation is not
a tuning knob; it is the merge protocol's progress guarantee.

THE HARDWARE INPUTS ARE THE M4'S, THE COMPUTATION IS THEIRS
------------------------------------------------------------
Their two hardware numbers are runtime queries of the device. Metal exposes
neither query through Mojo, so the M4's values are pinned HERE, in one
place, and nothing downstream may hard-code a block count derived from
them.

- `numSMs` -> `APPLE_M4_GPU_CORES = 10`. The base M4 has a 10-core GPU;
  a GPU core is the SM's counterpart in this computation.
- the occupancy call -> `max_active_blocks_per_core`. CUDA's occupancy
  computation is min over per-SM limits (thread slots, registers, static
  shared-memory partition). On the M4 the expressible term is the THREAD
  SLOTS one: `APPLE_M4_MAX_THREADS_PER_CORE // n_threads`. The register
  term has no Metal-side query at all. The static shared-memory term is
  deliberately NOT a divisor here: on Apple GPU family 9 (M3/M4)
  threadgroup memory is dynamically cached rather than statically
  partitioned per core, and the measured query sweep in
  `bench/results/SCALING_2026-08-19.md` (deficit shrinking monotonically
  from 0.66x at ~32 blocks through 0.93x at ~2,000 blocks, 18.5 KB of
  threadgroup memory per block throughout) is only possible if many such
  blocks share a core -- a static 32 KB partition would cap them at one.
  `smem_bytes` is still taken and still CHECKED against the 32 KB
  per-threadgroup allocation cap, which is a validity wall, not an
  occupancy divisor.
"""

#: Their `numSMs`, `pairwise_distance_base.cuh:300-301`, fed the M4 value.
#: The ONE place the core count lives.
comptime APPLE_M4_GPU_CORES = 10

#: Thread slots per GPU core: 96 resident SIMD-groups of 32 lanes. The
#: scheduler-slot limit of Apple's GPU cores, and the one occupancy input
#: Metal leaves expressible (see module docstring).
comptime APPLE_M4_MAX_THREADS_PER_CORE = 3072

#: Metal's per-threadgroup allocation cap. `PORTING.md 1` records the wall.
comptime METAL_MAX_THREADGROUP_MEM = 32768


def max_active_blocks_per_core(n_threads: Int, smem_bytes: Int) raises -> Int:
    """`cudaOccupancyMaxActiveBlocksPerMultiprocessor(func, Nthreads,
    sMemSize)` at `pairwise_distance_base.cuh:306-307`, with M4 inputs.

    `smem_bytes` is a validity check only on family 9; the divisor is the
    thread-slot term. See the module docstring for why.
    """
    if smem_bytes > METAL_MAX_THREADGROUP_MEM:
        raise Error(
            "max_active_blocks_per_core: threadgroup memory request exceeds"
            " Metal's 32 KB per-threadgroup cap; the kernel cannot launch"
        )
    var per_core = APPLE_M4_MAX_THREADS_PER_CORE // n_threads
    if per_core < 1:
        per_core = 1
    return per_core


def launch_config_generator(
    m: Int, n: Int, mblk: Int, nblk: Int, n_threads: Int, smem_bytes: Int
) raises -> Tuple[Int, Int]:
    """`launchConfigGenerator<P>(m, n, sMemSize, func)`,
    `pairwise_distance_base.cuh:295-322`, returning `(grid_x, grid_y)`.

    `mblk`/`nblk`/`n_threads` stand in for the `P` template parameter's
    `Mblk`/`Nblk`/`Nthreads`; the occupancy of `func` is
    `max_active_blocks_per_core` above.
    """
    var num_blocks_per_sm = max_active_blocks_per_core(n_threads, smem_bytes)
    # `std::size_t minGridSize = numSMs * numBlocksPerSm;` `:308`
    var min_grid_size = APPLE_M4_GPU_CORES * num_blocks_per_sm
    # `std::size_t yChunks = raft::ceildiv<int>(m, P::Mblk);` `:309`
    var y_chunks = (m + mblk - 1) // mblk
    # `std::size_t xChunks = raft::ceildiv<int>(n, P::Nblk);` `:310`
    var x_chunks = (n + nblk - 1) // nblk
    # `grid.y = yChunks > minGridSize ? minGridSize : yChunks;` `:311`
    var grid_y = min_grid_size if y_chunks > min_grid_size else y_chunks
    # `grid.x = (minGridSize - grid.y) <= 0 ? 1 : xChunks;` `:312`
    # (unsigned upstream; `grid.y <= minGridSize` always, so `<= 0` is `== 0`)
    var grid_x = 1 if (min_grid_size - grid_y) <= 0 else x_chunks
    # `:313-319`
    if grid_x != 1:
        var i = 1
        while grid_y * i < min_grid_size:
            i += 1
        grid_x = x_chunks if i >= x_chunks else i
    return (grid_x, grid_y)
