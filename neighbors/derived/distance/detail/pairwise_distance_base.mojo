# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
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

THE HARDWARE INPUTS ARE THE TARGET COLUMN'S, THE COMPUTATION IS THEIRS
-----------------------------------------------------------------------
Their two hardware numbers are runtime queries of the device. Metal exposes
neither query through Mojo, so the values are pinned per vendor column in
`original/hardware_matrix.mojo` and read here through
`kernel_matrix.TARGET_COLUMN` -- the target decides in ONE place, and
nothing downstream may hard-code a block count derived from these.

- `numSMs` -> `TARGET_GPU_CORES = gpu_cores_for[TARGET_COLUMN]()`. The
  Apple column is the base M4's 10 GPU cores (a GPU core is the SM's
  counterpart in this computation); nvidia and amd hold the A100's 108 SMs
  and the MI250X GCD's 110 CUs, UNVALIDATED, sources in the matrix.
- the occupancy call -> `max_active_blocks_per_core`, which resolves
  `hardware_matrix.max_active_blocks_for` at the target column. On the
  Apple column the divisor is the thread-slot term alone and `smem_bytes`
  is only a validity wall (family-9 dynamic caching; the measurement and
  the argument live on `smem_statically_partitioned_for`); on nvidia/amd
  the static shared-memory partition divides too, as CUDA's own occupancy
  computation does.
"""

from original.hardware_matrix import gpu_cores_for, max_active_blocks_for
from original.kernel_matrix import TARGET_COLUMN

#: Their `numSMs`, `pairwise_distance_base.cuh:300-301`, read from the
#: hardware matrix's target column. The ONE place a launch reads the core
#: count. Apple column = 10, the M4 value the 2026-08-19 grids ran on.
comptime TARGET_GPU_CORES = gpu_cores_for[TARGET_COLUMN]()


def max_active_blocks_per_core(n_threads: Int, smem_bytes: Int) raises -> Int:
    """`cudaOccupancyMaxActiveBlocksPerMultiprocessor(func, Nthreads,
    sMemSize)` at `pairwise_distance_base.cuh:306-307`, resolved against
    the target column.

    A thin reader of `hardware_matrix.max_active_blocks_for` so every
    consumer (this file's `launch_config_generator`, `core/gram_splitk`'s
    chunk count, and through the first the k-NN AUTO default) keys off the
    same table row. `original/hardware_matrix_check.mojo` raises if this
    reader and the table ever disagree.
    """
    return max_active_blocks_for[TARGET_COLUMN](n_threads, smem_bytes)


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
    var min_grid_size = TARGET_GPU_CORES * num_blocks_per_sm
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
