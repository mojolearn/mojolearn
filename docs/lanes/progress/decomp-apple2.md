# decomp-apple2: progress (Apple speed round 2, IDENTICAL and FAST)

Brief: ~/mojolearn-evidence/apple2_speed_brief.md. Branch lane/decomp-apple2
off lane/apple-merged 037daa353. Round 1: docs/lanes/progress/decomp-apple.md.
Evidence: ~/mojolearn-evidence/decomp-apple2/.

## Changes (all x_decomp local unless noted; shared code untouched)

1. **jacobi2** (x_decomp/jacobi2.mojo, 5c144678d): the kit's two Jacobi
   solvers rescheduled, same bits by construction.
   - eigh (Isomap, ClassicalMDS, every kit eigh): one barrier per rotation
     (the pick read from a double-buffered threadgroup stash of the three
     cells by every thread, instead of thread 0 plus a barrier), all of a
     lane's loads issued before its first store, the basis kept transposed.
   - svd (LocallyLinearEmbedding, lstsq, NMF init, FactorAnalysis, CCA...):
     R and V transposed in scratch; the 32 fold lanes own rows t, t+32, ...
     for the whole solve and fold the NEXT pair's Gram in the rotation pass
     (three folds, one barrier); V rotated by the other 224 threads. No
     device word crosses threads.
   - Switch: MOJOLEARN_XD_JACOBI=1 selects the old kernels (A/B only).
2. **ALS team kernel + small LU in one launch** (cf15378a3): als_rows runs
   a block per row, thread t owning cells t, t+256, ... of the f*f+f
   accumulators in threadgroup memory with als_row's exact per-cell
   sequence, then als_row_solve (als_row's tail, split out; host unchanged).
   LU with n <= 16 runs lu_serial as one launch (MinCovDet: 9k LUs of 8x8
   took 20 s as 5n launches each). Switches: MOJOLEARN_XD_ALS_TEAM=0,
   MOJOLEARN_XD_LU_SERIAL=0.
3. **FAST Lanczos** (3d4f61168, python only): Isomap / ClassicalMDS in FAST
   mode with eigen_solver 'auto', n > 200 and nc < 10 (sklearn KernelPCA's
   own ARPACK rule) take a full-reorthogonalization Lanczos on the kit;
   falls back to the exact solve if not converged by 600 vectors.
   IDENTICAL unchanged. Quality: bench/decomp_fast_quality.py.

## A/B (same Mac, same job, arm 1 = old switches, arm 2 = new)

(pending)
