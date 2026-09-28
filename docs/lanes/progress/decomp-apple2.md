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

## Consolidated Metal failure, 2026-09-28

The frozen main `308878e80679` base check failed in the common `x_decomp_eigh`
path for `x-decomp-factor-analysis`, `x-decomp-fastica`, `x-decomp-ipca`,
`x-decomp-manifold`, and `x-decomp-robust-cov`. The M4 compiler diagnostic
reports show SIGABRT with METAL reason `cannot select: 113 7, 1` in `agc.main`;
16 reports at 19:00–19:01 UTC have that same reason. This is compiler failure,
not a numerical comparison or evidence that retrying these lanes will help.

Metal now defaults eigh to the established `device_eigh` implementation.
CUDA/HIP retain jacobi2 eigh; the new SVD default is unchanged everywhere.
`MOJOLEARN_XD_JACOBI=2` remains an explicit experimental Metal opt-in, with
this compiler limitation unresolved. Removing its fence would lose required
device-memory ordering and is not the fix. Qualification of the restored
Metal default requires a targeted follow-up of the five lanes above; no pass
is claimed by this source change.

## NMF batch contract correction

The repair check at `9d64cb98b284` exposed `BATCH_MOVED` for NMF.transform
on both Apple and AMD. This method solves coefficients with fitted components
held fixed, but its finite iterative solve uses a global stopping criterion:
CD sums the row violations before comparing with the initial violation; MU
uses global reconstruction error and also initializes from the input matrix's
mean. A row alone can therefore stop at a different iteration. The batch
probe had incorrectly promised independent-row semantics for that solve.

The NMF batch probe now checks `inverse_transform`: independent coefficient
rows multiplied by the fitted components. It compares full, individual and
split outputs and remains sensitive to injected output corruption. The batch
revision is `nmf-inverse-batch-2026-09-28-v2`; previous transform probe records
cannot qualify this method. Train/inference hashes still cover transform's
actual CPU/GPU and cross-vendor identity. No solver arithmetic or convergence
rule was changed to make the new probe pass; native qualification is pending.
