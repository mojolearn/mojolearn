# LANE covariance-unblock, 2026-08-19: the zero-copy T-N route is UNWIREABLE, the materialized route stands, and the scoreboard's diagnosis is falsified

## The finding that matters most

**The premise of this lane — that the T-N (covariance) matmul shape is
blocked and is the root cause of PCA 0.25x / OLS 0.38x — was already false
at HEAD.** `core/gemm.mojo::gemm_tn` has served exactly that shape on MAX's
tuned matmul since before the scoreboard run: two `transpose_kernel` passes
then the proven `gemm_nt`, wired into PCA (`pca.mojo:202`), tSVD
(`tsvd.mojo:89`) and OLS (`lstsq.mojo:120`). The SCOREBOARD numbers (PCA
520.0 ms = 0.25x, OLS 397.8 ms = 0.38x vs `ols_normal_eq`, commit b2d438c)
were measured WITH that route in place: `gemm_tn` landed by 048f3da <
b2d438c. So SCOREBOARD_2026-08-19.md's "what to do next" item 1 — "the
covariance shape needs `linalg.transpose` + N-T matmul" — is falsified twice
over (that route exists AND `linalg.transpose` aborts). This lane was told
not to touch the scoreboard file; **the orchestrator should delete that
sentence** and re-diagnose PCA/OLS by phase, because the covariance product
is not where the 4x lives. Arithmetic bound: at 4M x 32 the Gram product is
8.2 GFLOP (~33 ms at the measured 248 GFLOP/s) and the two transposes are
~2 GB of traffic (~20 ms); the whole covariance step cannot account for
520 ms.

## Routes tried, in the ordered given, each with poisoned-output probes

### (a) LAYOUT ROUTE, zero-copy col-major view — WORKS, THEN LIES: UNWIREABLE

`layout.tile_layout` DOES ship `col_major`, `TileTensor` accepts it, and
`matmul[transpose_b=False]` accepts the view as operand A. First battery
(per-cell Float64 oracle, hashed splitmix64 fixtures, output poisoned with
-987654):

    8x8x8 ok | 33x17x255 ok | 255x257x65 ok | 129x127x513 ok
    32x32x100003 ok | 1x8x33 ok | 1x1x7 ok
    8x1x33 WRONG: cell (0,0) device -2.6746 host -3.1073, all poison gone

Aliasing: one buffer as both operands is a COMPILE error ("aliasing values
passed mutably"), so the route still needs the existing alias copy.
Dev-to-dev `enqueue_copy(dst_buf=, src_buf=)` verified element-exact at
320,224 elements. At 32x32x10007 the view's output is BIT-IDENTICAL to the
materialized route (0 of 1024 cells differ; both routes' worst
accumulation-order error 4.09e-6 relative, same number).

**Wired into `gemm_tn` it failed the suite immediately**: pca_main
"6 of 6 off-diagonal pairs are not bitwise symmetric"
(`check_covariance_is_symmetric`), ols_main "device Jacobi did not converge
... ||offdiag||/||A||_F still 0.51755863" — a garbled contraction surfacing
exactly as PORTING.md 23 predicts. The failing regime, mapped by sweep
(m=n x k, per-cell oracle):

    k=64   k=512  k=2048  k=10007  k=100003
    m=4    WRONG  WRONG   WRONG    ok      ok
    m=8    WRONG  WRONG   WRONG    ok      ok
    m=16   WRONG  WRONG   WRONG    ok      ok
    m=24   WRONG  WRONG   WRONG    ok      ok
    m=31   WRONG  WRONG   WRONG    ok      ok
    m=32   WRONG  WRONG   WRONG    ok      ok
    m=33   WRONG  WRONG   WRONG    ok      ok
    m=64   WRONG  WRONG   WRONG    ok      ok

WRONG means EVERY cell wrong (poison overwritten with plausible numbers),
against the first battery's small-shape passes (33x17x255 ok, 8x8x8 ok).
The dispatcher honors the view's strides on some arms and indexes raw
row-major memory on others, and the boundary zigzags with shape — no
predicate a guard could encode. `gemv_gpu` with the view is wrong at 8 of 8
outputs (k=4001), killing the zero-copy X^T y as well. **Route (a) is
recorded as UNWIREABLE**, with four sentinel shapes kept in the vendor table
(`check_matmul_colmajor`) so a toolchain that fixes it announces itself.

### (b) MATERIALIZED ROUTE — already shipped at HEAD; left exactly as it was

`gemm_tn` = `transpose_kernel` x2 + `gemm_nt` (the n==1 -> `gemv_n` guard
untouched). This lane's wiring experiment was fully reverted after (a)'s
verdict; the shipped fit paths are byte-for-byte HEAD. New evidence added
for it: a WIRED vendor row through the real wrapper (32x32x10007, output and
both alias buffers pre-poisoned) — the first direct vendor-table coverage
`gemm_tn` has had.

### (c) `linalg.transpose` re-verification (required once)

Still a hard abort, not a catchable raise: exit 133 inside
`linalg::transpose::_copy_with_strides rank=2 dtype=f32`,
"enqueue_cpu_range is only supported on CPU DeviceContexts"
(`vendor_main --transpose`, 2026-08-19 evening). It stays banned; the ban is
re-dated in VENDOR_LIBRARIES.md.

## What RAFT does (file:line, read from the checkout at 661a3b8)

- `raft/stats/detail/cov.cuh:65-66` — cov is ONE cublas gemm,
  `transA = !rowMajor` (OP_T for row-major data), with `alpha = 1/(N-1)`
  FUSED into the gemm. MAX's matmul has no alpha, so our separate
  `scale_in_place_kernel` pass stands (documented deviation, O(cols^2)).
- `raft/linalg/detail/lstsq.cuh:293-305` — `covA <- A^T A` via gemm
  `CUBLAS_OP_T, CUBLAS_OP_N`; `:309` — `Ab <- A^T b` via
  `raft::linalg::gemv(..., trans=true)` on a second stream; `:338` —
  `w <- covA Ab` via gemv, trans=false.
- `raft/linalg/detail/transpose.cuh:158, 207, 249` — their transpose is
  `cublasgeam`, closed, so there is nothing to port; our `transpose_kernel`
  stands on the documented hand-write exception.

cuBLAS serves OP_T by reading the same bytes through swapped strides — the
col-major view IS the faithful mirror of their call, which is why it was
worth three probes. The mirror is defeated by MAX's dispatcher, not by the
idea.

## X^T y

`xty_kernel` (hand-written, `column_stats.mojo`) stays. The gemv correctness
row exists (`gemv_gpu[transpose_b=False]`, CORRECT, m 1..100003) but covers
the ROW-MAJOR orientation only; the transposed orientation RAFT's `:309`
wants has NO vendor route: `gemv_gpu` on a col-major view is wrong at every
output (new vendor row), and matmul-at-n=1 with the view is the 8x1x33
failure above. Materializing a transpose just for the gemv would cost more
traffic than `xty_kernel`'s single pass.

## What changed in this commit

- `mojo_only/vendor_correctness_check.mojo`: `check_matmul_colmajor` — the
  arm-dependent col-major row (WRONG, unwired), the gemv-view row (WRONG,
  unwired), and the WIRED `gemm_tn` row (CORRECT, aliases pre-poisoned,
  mag-relative 1e-5 tolerance justified by the same-sign diagonal; measured
  spread 4.09e-6 on BOTH routes).
- `VENDOR_LIBRARIES.md`: three new table rows; `gemm_nt` n=1 wired row
  corrected to CORRECT (the guard exists and passes — the old WRONG row and
  the "exits non-zero today" header sentence were stale); transpose ban
  re-dated; the stale `ColKernelPolicy` RESOLVED paragraph and C8's stale
  complaint rewritten to the current truth.
- `PORTING.md`: new hazard section — a col_major view is honored by some
  matmul arms and silently ignored by others, with the sweep and the
  two lessons (a passing battery is evidence about its shapes only;
  arm-dependent correctness is disqualifying by itself).
- NO shipped algorithm code changed. The fit paths, `core/gemm.mojo`, and
  `core/column_stats.mojo` are byte-identical to HEAD (wiring experiment
  reverted in-session after the checks failed it).

## Checks, all green at the final tree

- vendor_main: exit 0, "no WIRED primitive tested WRONG"; new rows print
  `matmul colmajor-A 32x32x100003: ok / 8x8x512: WRONG, as recorded /
  32x32x2048: WRONG, as recorded / 8x1x33: WRONG, as recorded`,
  `gemv_gpu colmajor-A WRONG at 8 of 8 outputs, as recorded`,
  `core/gemm.mojo::gemm_tn (transpose x2 + gemm_nt) ok`.
- pca_main: check_covariance_is_symmetric OK (all 6 pairs bitwise),
  check_pca_fit OK, check_pca_invariants OK (x3 scaling moved variances by
  exactly 9, +1000 shift moved nothing), check_input_restored OK (worst
  element 9.5e-07), check_tsvd_against_pca OK.
- pca_wide_main: OK at 64 and 128 features; truncation OK.
- ols_main: exact, scale-invariant, beats-truth-on-noise, dispatch guard —
  all OK.
- jacobi_main: n = 16..256 all OK.
- Reach: the new vendor rows are their own sabotage (expected-WRONG probes
  that print "as recorded"); the wired-route reach evidence is the in-session
  failure itself — wiring a wrong covariance through `gemm_tn` broke
  pca_main and ols_main on the first run, so the checks demonstrably see
  this path.

## What the orchestrator should time

Nothing from this lane — no shipped code moved, so the PCA/OLS rows will not
move. What this lane buys is the diagnosis: **stop attributing PCA 0.25x /
OLS 0.38x to the covariance matmul.** The next measurement worth taking is a
per-phase timer inside `pca_fit` and `lstsq_eig` at the bench shapes
(4M x 32), the way the DBSCAN lane did it: mean/center passes, `gemm_tn`
(transposes vs matmul), `xty_kernel`, Jacobi + truncation, restore pass, and
the buffer allocations in the harness. The covariance step's arithmetic
ceiling (~50-60 ms of the 520) says at least 4x of the gap is elsewhere.
Candidates visible from reading, unmeasured: `xty_kernel` runs n_cols blocks
(32 blocks x 256 threads for a 512 MB read at the OLS shape); Jacobi runs
grid=(1,1,1); the fixed per-fit `ctx.synchronize()` count.
