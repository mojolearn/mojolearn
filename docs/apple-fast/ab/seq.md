# lane/apple-fast-seq: the classical sequence forecasters' per-step state in registers

Written without a Mojo toolchain (cloud peer); the first M3 build is the compile check.
Every switch is a build define compiled under FAST only (`GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL and
is_defined[...]()`) and defaults OFF; IDENTICAL compiles the old code. The worked example is
`sequence/theta.mojo` theta_run_reg on lane/apple-fast-tier (`-D MOJOLEARN_SEQ_THETA_REG=1`).

| switch | kind | site | what it changes under FAST |
|---|---|---|---|
| `-D MOJOLEARN_SEQ_CROSTON_REG=1` | build define | `sequence/croston.mojo` `_croston_reg` (op_croston, classic and SBA) | the two exponential smoothings folded into the scan over y: no compacted demand / interval arrays written to and read back from device scratch; same operations, same order. The optimized variant (golden section, repeated passes) keeps the stored path |
| `-D MOJOLEARN_SEQ_GARCH_REG=1` | build define | `sequence/garch.mojo` `garch_nll_reg`, `GarchObj.eval` | the Nelder-Mead evaluations' residuals, sigma2 recursion and log-likelihood in one register pass (p, o, q <= 1; larger orders keep the stored path); the final evaluation still stores sigma2 for the sigma output and the forecast |
| `-D MOJOLEARN_SEQ_GARCH_GRID=1` | build define (+ a launch in `sequence/pyapi.mojo` garch_py) | `sequence/garch.mojo` `_garch_grid_cell`, op_garch (i8 1 / 2) | the 4 x 4 x 4 starting-value grid as one element per (series, candidate), a launch of B x 64 scratch-free cells before the fit; the fit's thread takes the argmin over the 64 values in candidate order with the serial loop's strict `<` (first lowest wins), then builds that candidate with the serial loop's statements |

Causes:
- croston: `sequence/croston.mojo` op_croston's first loop writes the positive demands and the intervals
  to the thread's scratch row (2 n floats) and `ses_forecast` reads each back once, one thread per
  series (64 on taxi-hourly): a device-memory round trip per element with nothing to hide it behind.
- garch: `GarchObj.eval` writes n residuals, `garch_sigma2` writes sigma2_t and reads sigma2_{t-1},
  r_{t-1}, and `garch_nll` reads both rows again, per evaluation of two 2000-iteration Nelder-Mead
  runs, one thread per series. A GARCH(1, 1) step reads only the previous residual and variance and
  the likelihood is a running sum.
- garch grid: op_garch's "starting_values" loop runs 64 serial `garch_nll` passes per thread before the
  optimiser starts (64 threads x 64 passes on taxi-hourly); a GARCH(1, 1) cell needs no scratch, so
  the grid is 4096 independent elements.

Not changed, and why:
- ets (`sequence/ets.mojo` ets_lik): the level, trend and error sum are already registers; the only
  per-step scratch is the seasonal ring (one load and one store per step, m <= 58, dynamically
  indexed), and the board's damped-ets has no season, so there is nothing to move. No auto model
  selection exists (the 'Z' letters are not carried), so the second lever has no candidate loop.
- theta: lane/apple-fast-tier's file; its 4-model loop is the other candidate for a (series, model)
  launch, noted for that lane.
- autoarima: runs over `arima/` and the `_mojolearn_arima` binding (python/mojolearn/_x_sequence_autoarima.py),
  outside this lane's files; its order search is already batched over the series group per order.
- stl (loess windows over the whole series), var (design + GEMM + Cholesky): the state is the whole
  history, not a recurrence.
- prophet (`sequence/prophet.mojo` _fg_data): the per-point running sum is the whole gradient vector
  (P = 3 + S + K, about 42 on the board) updated through runtime-bounded loops, a dynamically indexed
  thread array rather than registers; and the board's 64 x 1,440 fit runs on the host executor
  (bindings/_mojolearn_x_sequence.mojo prophet_host_max), as garch does (garch_host_max 4096).

Queue notes (docs/apple-fast/ab/seq.txt, 11 lines): the garch define arms run with
`MOJOLEARN_SEQ_GARCH_HOST_MAX=0` so both arms take the device path (the shipped binding routes
batches <= 4096 series to the host executor); `seq-garch-dev` is the device FAST baseline at head,
`seq-garch-ident` / `seq-garch-fast` are the board's own route. The defines compile the same bodies
into the host executor, so the host route also takes them when its env is unset.

Keep rule: a switch becomes the FAST default when its arm is faster on the M3 and held-out quality
stays within FAST's run-to-run spread (the bits are the stored path's by construction, so the
quality should be identical); then the define goes and the arm is the code.
`tools/afc_ab_def.sh` is copied from lane/apple-fast-tier.
