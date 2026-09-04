# The GEMM round. My diagnosis was wrong, and the numbers say so.

## What I predicted

`bench/results/FIRST_RUN_2026-08-19.md` blamed `core/gemm.mojo` for k-means,
k-NN, PCA and DBSCAN all losing to scikit-learn, and said one kernel was
worth more than any number of ports.

## What happened

Ported RAFT's `linalg/contractions.cuh` register-tiling policy
(`Policy4x4<float>`: 4x4 outputs per thread, 64x64 output tile, Kblk 32,
padded shared stride). **First attempt was SLOWER than the naive kernel it
replaced**, across the board.

Cause: `stack_allocation` with no address space is thread-local MEMORY, not
registers, so every accumulator access became a load and a store. Their
`acc[][]` and `regx[]` are plain C arrays that nvcc keeps in registers.
Moving the accumulators to `SIMD` values fixed it. `archive/reference/PORTING.md 26`.

Three measurements, same shapes, same machine, alternating processes:

    arm       naive 16x16   tiled+smem-acc   tiled+SIMD-acc   sklearn
    kmeans        450.83          542.92           491.05       55.96
    knn            21.67           30.88            22.26       31.38
    pca            16.94           16.92            17.00        6.12
    dbscan         15.52           20.38            17.08        8.76
    ols            28.18           27.67            28.14      102.36

## The finding

**Only k-NN was GEMM-bound.** It goes from a tie to a clean 1.41x win, and
that is the whole effect of this round.

k-means did not move materially, PCA did not move AT ALL, and DBSCAN did not
recover. So the thing dominating those three is somewhere else, and the
diagnosis in the first results file was wrong. Two candidates, neither yet
measured:

- **`reduce_min_kernel` launches one block of 128 threads per row.** At
  k-means' 200k rows and 16 clusters that is 200,000 blocks in which 112 of
  128 threads have nothing to do. The block shape is sized for the row count
  and the work is sized for the cluster count.
- **`core/column_stats.mojo::covariance_kernel` was STILL the naive 16x16
  tile.** (Fixed later the same day: it is now a port of RAFT's COLUMN-major
  contraction policy plus a split-K over rows. See archive/reference/VENDOR_LIBRARIES.md.) Only `gemm_nt_kernel` was rewritten. PCA's whole row-scaling cost
  is that kernel, which is exactly why PCA did not move by one millisecond.

That second one is the more embarrassing and the more instructive: there are
TWO contraction-shaped kernels in `core/` and this round fixed one of them.

## What that argues for, and it is Andrew's suggestion

One contraction layer, written once, used by everything. `core/` is already
half that and it should be the whole of it.

**And MAX already ships the layer.** `linalg.matmul` imports and compiles in
this toolchain, alongside `linalg`, `nn`, `algorithm` and `layout`. Those are
Modular's own tuned kernels and they are the direct analog of cuBLAS, except
that they run on Metal. Hand-porting RAFT's contraction was the right instinct
about WHERE the leverage is and the wrong choice of WHAT to port: the
incumbents call a vendor BLAS, and the faithful mirror of that is to call
ours, not to write one.

Next round, in order:
1. Put `linalg.matmul` behind `core/gemm.mojo`'s signature and measure it
   against the hand-ported kernel. One change, five algorithms.
2. Profile k-means per phase instead of guessing again.
3. Only then decide whether `covariance_kernel` needs its own treatment or
   folds into the same layer.

## Correctness

All thirteen checks across the five sections still pass, unchanged, on both
GEMM versions.
