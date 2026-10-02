# lane/apple-fast-decomp-linalg: the linalg doors and the decomp kit's gemm (FAST on Apple)

Written without a Mojo toolchain (cloud peer); the first M3 build is the compile check (bindings/build_x_decomp.sh FAST,
each define through MOJOLEARN_MOJO_BUILD_FLAGS, as tools/afc_ab_def.sh does). Every switch is a `-D` build define, default
off, compiled only under `XD_FAST_APPLE` (x_decomp/device.mojo: `GLOBAL_NUMERIC_MODE == NUMERIC_FAST and
has_apple_gpu_accelerator()`). The Python doors learn the build's defines from the binding (`x_decomp_fast_defines` in
x_decomp/api.mojo, `_kit_fast_define` in python/mojolearn/_expansion_decomp.py, asked once per kit), so nothing reads an
env variable. IDENTICAL compiles and runs main's code unchanged. Merged with origin/main (fa667c10e, then 829c3fb4a): main's blocked TSQR
(`qr` reduced, `svd` thin, `lstsq`), its one-launch-per-panel LU and its device `lu_solve` are the defaults the arms race.

| define | site | lanes | what it changes under FAST on Apple |
|---|---|---|---|
| `MOJOLEARN_QR_FAST_DEV` | `_linalg_impl.py _qr_q` (FAST kit ahead of `_qr_tsqr`); `x_decomp/device.mojo DevExec.geqrf / orgqr` -> `_geqrf_fast / _orgqr_fast`; kernels in `x_decomp/fast_qr.mojo` | qr | geqrf + orgqr with every fold a grid reduction: the column norm as dlassq (scale, ssq) pairs over `fq_head_blocks` blocks and a one-block fold (`fq_head_finish_kernel`, tau and beta on its thread 0), w = v^T a_j as row-chunk partials folded by `fold_kernel`; 5 launches a column. Races main's TSQR (`_qr_tsqr`, LAPACK signs against TSQR's non-negative R diagonal: the board's quality column is sign-free). |
| `MOJOLEARN_SVD_FAST_CHOLQR` | `_linalg_impl.py svd` (FAST kit on the whole-matrix `_svd_tall`, ahead of `_svd_tsqr`); `device.mojo orth_on_device_diag` | svd | U = orth(A V / s) by CholeskyQR2 (two passes: G = Q^T Q by the kit gemm, its Cholesky, Q <- Q R^-1) instead of two sliced Householder passes; a failed factorization or a factor diagonal spanning more than 2^8 falls back to the sliced passes, read once per pass on the host where the pass waits. |
| `MOJOLEARN_DECOMP_FAST_GEMM_TILED` | `device.mojo launch_gemm` -> `x_decomp/fast_gemm.mojo fg_gemm_tiled_kernel` | lstsq, randomized-svd, nmf, als (every kit gemm; also pls, cca, factor-analysis, svd's CholeskyQR2) | 32 x 32 output tiles with the k axis staged 16 deep in threadgroup memory, the same FOLD_BLOCK partials (grid z) and `fold_kernel`; replaces `gemm_kernel` / `gemm_part_kernel`, one thread per output cell reading the whole k axis from device memory. lstsq on main is the TSQR of [a b] plus the small products and the residual gemm A X: the arm covers those. |
| `MOJOLEARN_FA_FAST_QRR` | `device.mojo DevExec.qr_r` | factor-analysis | the matrix uploaded straight from the caller's floats and `qr_factor` run on it, R downloaded once; the route below copies the m x n values into a host List one append at a time (1,000,000 x 220: 220 million appends) before `device_qr_r` uploads that copy. |
| `MOJOLEARN_PLS_FAST_DEADCOLS` | `_expansion_decomp.py _PLS.fit` | pls, pls-canonical, cca | the dead-column scan of Yk per component on the device (abs, gts, colsum; q floats read back) instead of `Yk.s` (the whole n x q downloaded, evicted, re-uploaded) and n x q Python comparisons per component. |
| `MOJOLEARN_CHOL_FAST_BLOCKED` (pass 2) | `device.mojo DevExec.chol` -> `launch_chol_blocked`; also the Gram Cholesky inside `orth_on_device_diag`'s CholeskyQR2 pass | cholesky (the board's 8192 x 8192 synthetic), svd with the CholQR arm | blocked right-looking Cholesky: per panel of CH_NB = 32 columns one fixed-size block factors the diagonal block in threadgroup memory (`chol_panel_kernel`), one thread per row below solves against it with the mirror zeroed (`chol_trsm_kernel`), and the trailing symmetric update is the LU trail's register-blocked 64 x 64 tile kernel with the U slab the panel rows' transpose, tiles above the diagonal skipped (`chol_trail_rb_kernel`). 3 n / 32 launches (8192: 768) in place of the column driver's 2 n (16,384), whose column-j cells are each a j-long serial chain per thread. Gated by the 16 KB shared page (`lib_smem_page_fits_for`). |
| `MOJOLEARN_LU_FAST_PANEL4` (pass 2) | `device.mojo launch_lu` panel loop -> `lu_l_diag_kernel` | lu-factor, lu-solve (its clock is the factor: the board's lu-solve row is `lu_factor + lu_solve`) | main's `launch_lu` is already blocked (32-column panels, `lu_swaps_trsm_kernel`, the register-blocked `lu_trail_rb_kernel` trailing update), so no blocked structure is added; its panel is six launches a column, two of them one thread. The fused grid kernel reads the pivot directly for the multipliers and lets thread 0 write scal, act and info: four launches a column, 16,384 fewer at 8192. The remaining floor is the per-column pivot search (one block, a cross-row reduction) and the swap: a one-launch panel over n rows is the shape main refused (neural-pass115/135), so the next step past this is a two-launch grid pivot (partials + finish), not taken here. |

Dropped at the merges, main's replacement does what they did: MOJOLEARN_EIGH_FAST_RR (main 829c3fb4a, lane fix-eigh-main:
`linalg.eigh` runs x_decomp's round-robin Jacobi on the device, the route the switch selected), MOJOLEARN_LU_FACTOR_FAST_FUSED (main's `launch_lu` runs one
launch per panel; the lane's one-block per-column step is the shape main refused from neural-pass115) and
MOJOLEARN_LU_SOLVE_FAST_TRISOLVE (main's `launch_lu_solve` runs the permutation and `launch_trisolve` on the device; the
host walk it bypassed is gone).

svd istella (17.8 s vs torch-gpu 2.5 s on the user's Oct 2 board): the clock is passes over the 1,000,000 x 220 matrix,
not flops (one pass is 880 MB; at the M3 Ultra's bandwidth a few ms). `_svd_tall` makes three such passes (the sliced
QR inside `k.svd`, A V, and `orth_diag`'s two sliced Householder passes = four with the gemm), each sliced pass 64
slices x 32 threads walking columns at stride d; main's TSQR makes two (R, then Q U_R). The arm expected to close most
of the 7x is MOJOLEARN_SVD_FAST_CHOLQR: it turns `orth_diag`'s two sliced Householder passes into two gemm-shaped
passes (G = Q^T Q, Q R^-1), which the tiled gemm arm (MOJOLEARN_DECOMP_FAST_GEMM_TILED) then makes bandwidth-shaped;
the sliced QR inside `k.svd` stays the one non-gemm pass. No randomized or blocked bidiagonalization is needed at
d = 220: the only SVD of a square matrix is of the 220 x 220 R, already small; what matters is the number and the shape
of the passes over the rows. The cheap second step added in pass 2 is the Gram Cholesky inside the CholQR2 pass
through the blocked launch (MOJOLEARN_CHOL_FAST_BLOCKED, 880 one-thread/column launches an svd -> ~42); the guard
readback of G's diagonal (one sync a pass) stays, it is the pass's own wait. Request line dlin-svd-cholqr-chol-istella
races both defines against old FAST. If after these the row is still several x, the remaining lever is the sliced QR
pass itself (TSQR-shaped, main's `tsqr_device.mojo`, not this lane's).

Risky compile sites (no toolchain here):
- x_decomp/api.mojo `fast_defines_py`: nested `comptime if` on `is_defined[...]()`, `String +=` of literals; registered
  unparametrized in bindings/_mojolearn_x_decomp.mojo as `numeric_mode_py` is.
- x_decomp/device.mojo `comptime XD_FAST_APPLE` (`has_apple_gpu_accelerator` from std.sys.info, the tier branch's idiom).
- x_decomp/fast_qr.mojo `fq_head_finish_kernel`: one block, SHARED `stack_allocation` pairs, `_ssq_merge` on `mut` locals;
  `_geqrf_fast` / `_orgqr_fast` pass `buf.unsafe_ptr()` to `enqueue_function` as the file does everywhere.
- x_decomp/fast_gemm.mojo: SHARED tiles of FG_TPB, `fg_tiles`, the grid z fold.
- pass 2: `comptime CHOL_FAST_BLOCKED` (an `and` of the alias, `is_defined` and `lib_smem_page_fits_for`),
  `chol_panel_kernel`'s `while` loops over `ww * ww` with a conditional expression in the store, `chol_trail_rb_kernel`'s
  early `return` before `barrier()` (whole tiles only, so no thread of a block that reaches the barrier is missing),
  `lu_l_diag_kernel`'s `div0` on a register; `comptime if LU_FAST_PANEL4:` / `else:` around launches in `launch_lu`.
- `orth_on_device_diag`'s CholeskyQR2 pass: `enqueue_copy` of info and the factor into host buffers, read after the sync.

Request lines: docs/apple-fast/ab/decomp-linalg.txt (light form: one dataset per change, afc_ab_def.sh, binding x_decomp).
