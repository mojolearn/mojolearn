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

### Job 1790606245923 (m4pro-b, IDENTICAL, commit 89abb0595)

Output: ~/mojolearn-evidence/decomp-apple2/job2_m4pro-b.txt.

| call | old s | new s | digests |
|---|---|---|---|
| svd 200000x28 | 0.065 | 0.144 | equal (svd2 now gated to n >= 256) |
| svd 1000x256 | 1.526 | 1.430 | equal |
| svd 800x800 | 49.109 | 40.507 | equal |
| orth 200000x15 | 0.024 | 0.026 | equal |
| randomized_svd(5) 200000x28 | 0.149 | 0.131 | equal |
| lstsq 200000x27 | 0.076 | 0.074 | equal |
| eigh 800 / 1500 (old only) | 9.98 / 98.4 | FAILED | |

- jacobi2 eigh FAILED on Metal: "Failed to create compute pipeline state
  (GPU machine code generation)". Suspect: the atomic fence in its barrier
  (compiles to AIR, refused at pipeline creation). Fixed by dropping the
  fence (the old kernel's plain barrier); an unroll 1 instantiation is
  timed beside unroll 4 in the next job in case the fence was not it.
- bench/decomp_out_digest.py (27 algorithms): old Metal == new Metal ==
  CPU for every algorithm the new eigh does not reach (ALS team kernel,
  svd2 via LLE / FA / lstsq / CCA / rsvd, orth device guard, all equal).
  IPCA, FastICA, Isomap, CMDS and MinCovDet failed on the new arm (eigh2).
- The fits in the A/B script failed on a hashing bug in the bench (fixed).

### Job 1790619265077 (m4pro-b, IDENTICAL, commit eba5d65de)

Output: ~/mojolearn-evidence/decomp-apple2/job3_m4pro-b.txt. Arm 1 = every
switch old, arm 2 = new with eigh2 unroll 4, arm 3 = new with eigh2 unroll 1.
Every row's output hash is EQUAL across the three arms.

| call | old s | new, unroll 4 | new, unroll 1 (DEFAULT) | old/default |
|---|---|---|---|---|
| eigh 64 | 0.043 | 0.023 | 0.017 | 2.5x |
| eigh 256 | 0.370 | 0.446 | 0.322 | 1.15x |
| eigh 800 | 10.148 | 10.878 | 6.585 | 1.54x |
| eigh 1500 | 98.572 | 82.692 | 46.289 | 2.13x |
| svd 1000x256 | 1.548 | 1.436 | 1.439 | 1.08x |
| svd 800x800 | 51.171 | 42.726 | 42.824 | 1.19x |
| PCA(randomized,5) 200k x 28 | 0.143 | 0.133 | 0.131 | 1.09x |
| randomized_svd(5) 200k x 28 | 0.136 | 0.131 | 0.130 | 1.05x |
| FactorAnalysis(5) 200k x 28 | 0.185 | 0.185 | 0.184 | 1.00x |
| lstsq 200k x 27 | 0.074 | 0.073 | 0.072 | 1.03x |
| ALS(32f,5it) 20000 x 2000 | 3.688 | 1.993 | 1.986 | 1.86x |
| MinCovDet 20000 x 8 | 46.209 | 22.379 | 21.927 | 2.11x |
| Isomap(10nn) 1500 | 89.040 | 74.625 | 42.075 | 2.12x |
| ClassicalMDS 1500 | 59.181 | 49.585 | 27.760 | 2.13x |
| LocallyLinearEmbedding(10nn) 1500 | 359.990 | 303.816 | 303.655 | 1.19x |

(eigh 8 is left out: its first new-arm call paid the new kernel's pipeline
creation. MinCovDet's 8x8 eighs and 8x8 LUs are the real small-n
measurement.)

Digests: bench/decomp_out_digest.py, all 27 algorithms, old Metal == new
Metal (this job, unroll 4 build default at the time) == CPU (job
1790606245923's CPU column). The unroll 1 instantiation's hashes are the A/B
rows above (EQUAL on every row, including Isomap, CMDS, LLE, MinCovDet, ALS).

## FINAL (2026-09-28 ~19:00Z, freeze): branch lane/decomp-apple2

Default ON (A/B gain, equal digests, same Mac, same job):
- jacobi2 eigh at unroll 1 (every x_decomp kit eigh): Isomap / ClassicalMDS
  about 2.1x at 1500 rows, MinCovDet with the LU change 2.1x.
- jacobi2 one-sided SVD from n = 256 (LLE 1.19x, svd 800 1.19x); below 256
  the old kernel (the new one measured 0.45x at n = 28).
- ALS team kernel (1.86x).
- Small LU (n <= 16) in one launch (in MinCovDet's 2.1x).
- orth rank guard on the device (flat: 0.024 -> 0.026 s alone, 1.05-1.09x
  in the randomized fits; kept because the digests match, its switch is
  MOJOLEARN_XD_ORTH_DEV=0).

OPT-IN (unproven): FAST Lanczos top eigenpairs for Isomap / ClassicalMDS
(MOJOLEARN_XD_LANCZOS=1). The quality script
bench/decomp_fast_quality.py could not run: FAST mode wants a FAST-built
_mojolearn_x_decomp.so and the job built only IDENTICAL. Host-kit check at
n = 300 / 500 only (eigen errors 5e-7 / 8e-7 against the dense Jacobi's
2e-6 / 4e-6, float64 reference).

Shared code: none changed (decomposition/, core/ untouched). jacobi2 lives
in x_decomp and imports the shipped kernels' helpers; the old kernels stay
behind MOJOLEARN_XD_JACOBI=1. The same rescheduling would reach PCA full,
lstsq_eig and Nystroem if ported to decomposition/checks/jacobi_eigh_device.

UNPROVEN (for the combined verification run):
- Every change above has Apple (M4 Pro) digests only; none has run on the
  M2 Pro, NVIDIA or AMD. Both jacobi2 kernels and the ALS team kernel use
  256 threads (under the M2 dispatch limit).
- The final default switch to unroll 1 (commit after eba5d65de) was not
  rebuilt as a default on a Mac; its bits are the arm 3 rows above.
- jacobi2 eigh still hands device words between lanes behind a plain
  barrier(), exactly as jacobi_eigh_kernel does (Metal's barrier orders
  threadgroup memory only; an atomic fence made Metal refuse the pipeline).
  The svd2 and ALS kernels have no cross-thread device traffic.
- FAST Lanczos (opt-in) has no GPU quality check.

Commits: 5c144678d (jacobi2), cf15378a3 (ALS team, small LU), 3d4f61168
(FAST Lanczos), 5a2f15039 (eigh2 per-lane basis), c844561f7 (orth device
guard), 89abb0595 (merge apple-merged), eba5d65de (fence removed, svd2
from 256), and the FINAL commit.
