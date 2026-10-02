# lane/apple-fast-decomp-linalg: the linalg doors and the decomp kit's gemm (FAST on Apple)

Written without a Mojo toolchain (cloud peer); the first M3 build is the compile check (bindings/build_x_decomp.sh FAST,
each define through MOJOLEARN_MOJO_BUILD_FLAGS, as tools/afc_ab_def.sh does). Every switch is a `-D` build define, default
off, compiled only under `XD_FAST_APPLE` (x_decomp/device.mojo: `GLOBAL_NUMERIC_MODE == NUMERIC_FAST and
has_apple_gpu_accelerator()`). The Python doors learn the build's defines from the binding (`x_decomp_fast_defines` in
x_decomp/api.mojo, `_kit_fast_define` in python/mojolearn/_expansion_decomp.py, asked once per kit), so nothing reads an
env variable. IDENTICAL compiles and runs main's code unchanged. Merged with origin/main (fa667c10e): main's blocked TSQR
(`qr` reduced, `svd` thin, `lstsq`), its one-launch-per-panel LU and its device `lu_solve` are the defaults the arms race.

| define | site | lanes | what it changes under FAST on Apple |
|---|---|---|---|
| `MOJOLEARN_QR_FAST_DEV` | `_linalg_impl.py _qr_q` (FAST kit ahead of `_qr_tsqr`); `x_decomp/device.mojo DevExec.geqrf / orgqr` -> `_geqrf_fast / _orgqr_fast`; kernels in `x_decomp/fast_qr.mojo` | qr | geqrf + orgqr with every fold a grid reduction: the column norm as dlassq (scale, ssq) pairs over `fq_head_blocks` blocks and a one-block fold (`fq_head_finish_kernel`, tau and beta on its thread 0), w = v^T a_j as row-chunk partials folded by `fold_kernel`; 5 launches a column. Races main's TSQR (`_qr_tsqr`, LAPACK signs against TSQR's non-negative R diagonal: the board's quality column is sign-free). |
| `MOJOLEARN_SVD_FAST_CHOLQR` | `_linalg_impl.py svd` (FAST kit on the whole-matrix `_svd_tall`, ahead of `_svd_tsqr`); `device.mojo orth_on_device_diag` | svd | U = orth(A V / s) by CholeskyQR2 (two passes: G = Q^T Q by the kit gemm, its Cholesky, Q <- Q R^-1) instead of two sliced Householder passes; a failed factorization or a factor diagonal spanning more than 2^8 falls back to the sliced passes, read once per pass on the host where the pass waits. |
| `MOJOLEARN_EIGH_FAST_RR` | `_linalg_impl.py eigh` (FAST kit `eigh`) | eigh | the kit's round-robin Jacobi (`x_decomp/jacobi_par.mojo`: n/2 blocks a round, the off-norm test once a sweep) instead of `decomposition/linalg_public_device.mojo device_eigh`, `jacobi_eigh_kernel` on ONE block of 256 threads for the whole n = 4096 solve. Same ascending order and sign pin; bits differ (FAST). |
| `MOJOLEARN_DECOMP_FAST_GEMM_TILED` | `device.mojo launch_gemm` -> `x_decomp/fast_gemm.mojo fg_gemm_tiled_kernel` | lstsq, randomized-svd, nmf, als (every kit gemm; also pls, cca, factor-analysis, svd's CholeskyQR2) | 32 x 32 output tiles with the k axis staged 16 deep in threadgroup memory, the same FOLD_BLOCK partials (grid z) and `fold_kernel`; replaces `gemm_kernel` / `gemm_part_kernel`, one thread per output cell reading the whole k axis from device memory. lstsq on main is the TSQR of [a b] plus the small products and the residual gemm A X: the arm covers those. |
| `MOJOLEARN_FA_FAST_QRR` | `device.mojo DevExec.qr_r` | factor-analysis | the matrix uploaded straight from the caller's floats and `qr_factor` run on it, R downloaded once; the route below copies the m x n values into a host List one append at a time (1,000,000 x 220: 220 million appends) before `device_qr_r` uploads that copy. |
| `MOJOLEARN_PLS_FAST_DEADCOLS` | `_expansion_decomp.py _PLS.fit` | pls, pls-canonical, cca | the dead-column scan of Yk per component on the device (abs, gts, colsum; q floats read back) instead of `Yk.s` (the whole n x q downloaded, evicted, re-uploaded) and n x q Python comparisons per component. |

Dropped at the merge, main's replacement does what they did: MOJOLEARN_LU_FACTOR_FAST_FUSED (main's `launch_lu` runs one
launch per panel; the lane's one-block per-column step is the shape main refused from neural-pass115) and
MOJOLEARN_LU_SOLVE_FAST_TRISOLVE (main's `launch_lu_solve` runs the permutation and `launch_trisolve` on the device; the
host walk it bypassed is gone).

Risky compile sites (no toolchain here):
- x_decomp/api.mojo `fast_defines_py`: nested `comptime if` on `is_defined[...]()`, `String +=` of literals; registered
  unparametrized in bindings/_mojolearn_x_decomp.mojo as `numeric_mode_py` is.
- x_decomp/device.mojo `comptime XD_FAST_APPLE` (`has_apple_gpu_accelerator` from std.sys.info, the tier branch's idiom).
- x_decomp/fast_qr.mojo `fq_head_finish_kernel`: one block, SHARED `stack_allocation` pairs, `_ssq_merge` on `mut` locals;
  `_geqrf_fast` / `_orgqr_fast` pass `buf.unsafe_ptr()` to `enqueue_function` as the file does everywhere.
- x_decomp/fast_gemm.mojo: SHARED tiles of FG_TPB, `fg_tiles`, the grid z fold.
- `orth_on_device_diag`'s CholeskyQR2 pass: `enqueue_copy` of info and the factor into host buffers, read after the sync.

Request lines: docs/apple-fast/ab/decomp-linalg.txt (light form: one dataset per change, afc_ab_def.sh, binding x_decomp).
