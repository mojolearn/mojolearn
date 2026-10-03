# lane/apple-fast-fa: FactorAnalysis on the Apple GPU (FAST)

Binding x_decomp (bindings/build_x_decomp.sh). Board lane `factor-analysis` (AFC_FAMILY=algos), taxi (deciding, the gaps
file's row: FAST 144 / IDENTICAL 157 ms on the M3 Ultra) and istella (1M x 220; no board row yet, the heavier EM).
Profile of main's fit: docs/apple-fast/notes/fa.md. Code: x_decomp/fa_fast.mojo (every kernel and entry behind
`FA_FAST_APPLE`), entries registered in bindings/_mojolearn_x_decomp.mojo only under that guard, Python routes in
python/mojolearn/_expansion_decomp.py `FactorAnalysis` taken only when the binding reports the define
(`x_decomp_fa_defines`). Every define defaults OFF; IDENTICAL compiles main's code. Limits: d <= 256 features (fit),
nc <= 16 and d (nc + 1) <= 4096 (transform); past them main's route runs. Request lines: docs/apple-fast/ab/fa.txt
(14: seven per dataset; EIG_SMALL, LIVEBUF and LL_DEVICE act only inside ITER_DEVICE, so their A arm is ITER_DEVICE).

**MOJOLEARN_FA_GRAM_ONCE.** FA's EM depends on the data only through the sample covariance. One tiled pass over the
resident X (64 x 64 output tiles, 16-row slabs centered at load in threadgroup memory, a partial per 8192 rows, one fold)
gives G = Xc^T Xc (d x d) and var = diag(G) / n. It replaces main's setup: the Xc and Xc^2 buffers (n x d each), the
880 MB (istella) download of Xc, the host copy of every value into a List and the re-upload for `qr_r`, and the QR of
the n x d data. Each iteration then takes the eigh of D G D / n (same spectrum and right vectors as main's SVD of
R D / sqrt(n)), the route main already takes when n < d; the loop stays Python. Expected: a large setup cut on istella
(hundreds of ms), small on taxi. Risk: G squares the condition number of Xc; float32 eigenvalues of D G D / n near the
noise floor lose relative precision (the SVD of R does not); quality check = loglike_ end value and score vs IDENTICAL.

**MOJOLEARN_FA_ITER_DEVICE** (implies GRAM_ONCE's pass). The whole EM loop as ONE binding call on the resident G: per
iteration one scale launch (D G D / n and sqrt(psi) + 1e-12), main's round-robin eigh kernels on the loop's own buffers
(no sign-flip, ordering or download launches), one finish launch (orders the eigenvalues, signs the nc columns as main's
`sign_flip_kernel`, W, the psi update, the 2 d log terms) and ONE readback of 2 d + 4 floats; the log-likelihood is
summed in float64 on the host in main's order and main's tol test stops the loop. Removes ~20 small launches and ~12
syncs per iteration plus all Python per-iteration work. Expected: the main win on taxi (launch-and-wait bound). Risk: low.

**MOJOLEARN_FA_EIG_SMALL** (inside ITER_DEVICE). The eigh of the d x d as ONE launch of one 256-thread threadgroup: the
same round-robin rounds, rotation cells (x_decomp/rr.mojo) and per-sweep convergence test as main's grid eigh, every round
behind a device-memory barrier, instead of 2 (d - 1) launches and one sync per sweep. Expected: large at d = 220 (219
rounds a sweep), moderate at d = 11. Risk: one threadgroup does all the work at d = 220 (36k cell updates a round); if a
launch nears the macOS command-buffer limit it would show as an error, not wrong output (the status words are checked).

**MOJOLEARN_FA_LIVEBUF** (inside ITER_DEVICE). The loop's ~14 scratch buffers as one arena buffer (one live Metal buffer
instead of ~14; Apple launch cost grows with live buffers), and W plus psi read back in one copy. Expected: small (a few
percent of the per-launch cost). Risk: none on quality.

**MOJOLEARN_FA_LL_DEVICE** (inside ITER_DEVICE; A/B against ITER_DEVICE + EIG_SMALL, since the grid eigh still syncs
every sweep). The convergence test on the device: the finish launch sums the 2 d terms in double-float float32 (TwoSum;
Metal has no float64), applies -(n / 2)(S - S_prev) < tol and sets a flag that turns every later launch into a no-op; the
host reads the flag every 4 iterations and the per-iteration ll pairs once at the end. Removes the per-iteration wait
(the GPU queue runs ahead by up to 4 iterations). Risk: the double-float difference may cross tol one iteration earlier or
later than main's float64 test (n_iter_ +/- 1); W and psi are those of the stopping iteration either way.

**MOJOLEARN_FA_TRANSFORM_FUSED.** transform as one launch over rows: P = (W / psi)^T cov_z (d x nc) and the mean in
threadgroup memory, one row per thread, (x - mean) P accumulated in registers, one readback. Replaces the n x d centered
copy, two GEMM launches and the n x nc intermediate. Counts in infer_ms, not the fit time. Risk: reassociation
((X - mean) Wpsi^T) cov_z -> (X - mean)(Wpsi^T cov_z): float32 rounding only.

**MOJOLEARN_FA_ALL.** Every define above (they compose: GRAM_ONCE + ITER_DEVICE + EIG_SMALL + LIVEBUF + LL_DEVICE +
TRANSFORM_FUSED). The first request line per dataset.
