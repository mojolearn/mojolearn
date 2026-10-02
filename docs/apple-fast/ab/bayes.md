# lane/apple-fast-bayes: BayesianRidge / ARD Gram sse under FAST on Apple (NEXT_PASS item 5)

Written without a Mojo toolchain (cloud peer); the first M3 build of `x_linear` (`bindings/build_x_linear.sh`, FAST) is
the compile check. Every change is compiled under FAST + Apple only and defaults OFF; IDENTICAL compiles main's code
unchanged.

| switch | kind | site | what it changes under FAST on Apple |
|---|---|---|---|
| `-D MOJOLEARN_BAYES_GRID_GUARD=1` | define, `BAYES_GRID_GUARD` comptime in `x_linear/device.mojo` | `bayes_step_guard_kernel` and the guarded iteration loop of the BayesianRidge grid driver (`x_linear/device.mojo`, the `if bayes_grid:` block) | main's `bayes_step_gram_kernel` takes the iteration's sse as yy - sum_k (2 z_k vty_k - ev_k z_k^2) on one thread: on istella (220 features, near-null Gram directions, f32 noise eigenvalues) it cancels below zero, sse clamps to 0, alpha goes inf, coef NaN. The guard takes the sse relative to a REFERENCE row pass: s0 + sum_k dz_k (ev_k (z_k + z0_k) - 2 vty_k) with a bound (every eigenvalue off by 2^-12; trusted when it moves sse by at most 2^-8 of itself, `GRAM_SSE_TRUST`). The step kernel computes the next iteration's candidate for the coefficients it writes and leaves the verdict in state[6], the value in state[7]; the driver's existing per-iteration read of the stop word brings it along (no extra device-to-host read). An untrusted candidate makes the next iteration run main's `bayes_resid_kernel` + `bayes_part_kernel` (a thread per row, a thread per block), and that pass becomes the reference (s0 in state[4], z0 in the unused sse scratch fw[3dd + 5d, +d), so n >= d). No yy pass under the guard. |
| `MOJOLEARN_X_LINEAR_GRAM_SSE=0` (main's env arm, not new here) | env, read at dispatch in `x_linear/device.mojo` (hip[5]) | both paths | the row-pass arm: every iteration's sse from the rows. This is the reference the bar compares against (finite r2, no worse than the row pass; sklearn itself is r2 -426,163 on istella). |

Not covered by the define: ARD. ARD still fits on the one-block `fit_kernel` -> `ard_fit` (`x_linear/bayes.mojo`), whose
Gram sse already carries the same delta form and bound (`_t_sse_delta`, commit 79f7ace5f, FAST + Apple, on by default
with the row pass as its `MOJOLEARN_X_LINEAR_GRAM_SSE=0` arm). The `bayes-ard-guard-istella` line therefore races the
same code in both arms: it is the quality/spread check of the merged branch's ARD guard against main's ARD numbers
(main: ARD istella 0.86 s); the row-pass comparison for ARD is `tools/afc_ab.sh ... 1 2 - MOJOLEARN_X_LINEAR_GRAM_SSE=0`
if the orchestrator wants it under a second tag.

The one-block BayesianRidge guard (`bayes_ridge_fit` in `x_linear/bayes.mojo`) is kept: it is reached under main's
`MOJOLEARN_X_LINEAR_BAYES_GRID=0` / `MOJOLEARN_X_LINEAR_BAYES_GRID_GRAM=0` arms.

## Risky compile sites

- `x_linear/device.mojo` `bayes_step_guard_kernel`: `var sse: Float32` assigned in both branches (the idiom of
  `ard_fit`); `GRAM_SSE_TRUST` imported from `x_linear/bayes.mojo` (a module-level comptime Float32); `ratio` declared in
  two sibling scopes.
- `x_linear/device.mojo`, the guarded loop under `comptime if BAYES_GRID_GUARD:` inside `if bayes_grid:`: `max_iter = 0`
  after it so main's loop below does not run; `hst[6]` read from the 8-word state list main already downloads.
- Pointers: the kernel takes `FP` and is launched with `buf.unsafe_ptr()` exactly as the sibling `bayes_*_kernel` launches
  in the same block; no `MutAnyOrigin` parameters were added.

## Request lines (`bayes.txt`)

- `bayes-br-guard-istella`: bayesian-ridge istella, arm A main's grid Gram sse, arm B the guard. Bar: finite r2 and no
  worse than the row-pass arm.
- `bayes-ard-guard-istella`: ard istella, both arms the merged branch (see above).
