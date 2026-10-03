# gap-arima: autoarima FAST on Apple vs statsforecast-cpu

Rows (BOARD_M3_FAST.md, 18dc09a7a): taxi-hourly 39,298 ms vs 2,844 (13.8x); synthetic 28,504 ms vs 2,880 (9.9x).
Quality column: forecast_rmse (74.66 taxi, 2.624 synthetic); every candidate below is meant to keep the bits.

## The timed region (tools/bench_board_algos.py:2845-2851, :2913-2916)
`AutoARIMA(Y32 (64, 1392))`, `search(s=1, d 0..1, p 0..3, q 0..3, ic=aicc, fit_intercept=auto)`, `fit()`, `forecast(48)`.
- `python/mojolearn/_x_sequence_autoarima.py:110-118`: `select_d` (KPSS) once for the batch.
- `:120-133`: per d group, 15-16 orders, each a separate `ARIMA(maxiter=20).fit(sub)` -> one `batched_fit_x`
  (estimate_x0, Jones, then `batched_min_lbfgs`). Up to ~31 sequential solves.
- `:156-161`: `fit()` refits EVERY distinct chosen order on its sub-batch, sequentially, `maxiter=1000`, from estimate_x0
  again. The number of solver passes is the number of distinct winning orders (often 10-25 of 64 series); each pass
  costs (iterations of its slowest series) x (line-search steps) x (launch chain + waits), independent of how few
  series it holds (one 128-thread group).

## Per evaluation (one point for every series; arima/impl/fast_eval_ws.mojo:150-207, default since 6c54b7e87)
stack, unpack, Jones, init matrices, `kalman_init_state_kernel`, the loop kernel, mark, grad, copy, finish:
~12 launches. The loop kernel (arima/impl/batched_kalman.mojo `batched_kalman_loop_kernel`, ~:590-700) is one thread
per (series, member) walking 1,392 steps serially; at 64 x (N+1) <= 512 threads the GPU is nearly idle and each step
is a dependent rd^3 Riccati update (TP = T P, L = T - K Z, P = TP L' + RQR, symmetrize) plus the state update.

## Per L-BFGS iteration (arima/impl/batched_fit.mojo `batched_min_lbfgs`, ~:570-650)
three host waits: `_read_flag(any_active)` at the top, `_read_flag(any_searching)` right after the prelude (:594),
and one more per line-search step. The line search is lock-step: every series waits for the batch's longest search.

## Hypothesis (ranked)
1. Host waits + launch chain per evaluation x the number of evaluations, the evaluations multiplied by the sequential
   per-order solves of search and fit (maxiter 1000) - fixed cost per round, independent of batch size.
2. The serial Riccati recursion inside the loop kernel: P converges to a fixed point in a few dozen steps for a
   stationary/invertible model, yet every one of the 1,392 steps recomputes it.
3. Lock-step line search: rounds = sum over iterations of the batch's max LS steps.

## Candidates (all FAST + Apple, default OFF, same bits by construction)
| define | site | what |
|---|---|---|
| `MOJOLEARN_ARIMA_FAST_P_FIX=1` | batched_kalman.mojo `KALMAN_FAST_P_FIX`, loop kernel steps 2/3/5/6 | once a step leaves P bitwise unchanged (no NaN), keep F, log F, TP, K and skip the Riccati update: only pred / v / alpha / ll sums per step |
| `MOJOLEARN_ARIMA_FAST_LS_NOREAD=1` | fast_lbfgs_async.mojo `ARIMA_FAST_LS_NOREAD`; batched_fit.mojo LS loop | skip the redundant post-prelude `any_searching` read: one host wait fewer per iteration |
| `MOJOLEARN_ARIMA_FAST_ASYNC=1` | fast_lbfgs_async.mojo `async_start_kernel`, `async_step_kernel`, `async_min_lbfgs`; batched_fit.mojo branch | per-series device state machine: accept, verdict, history, next prelude and next candidate in one kernel after each shared evaluation; one flag read per 4 evaluations; no lock-step line search |
| P_FIX + ASYNC | both | composed |

Not taken: batching all orders of a search step into one launch (needs a per-member-order filter; a much larger change),
warm-starting `fit()` from the search's x (changes the iterate sequence, a quality risk), the time-parallel scan
(KALMAN_TIME_SCAN, already measured as quality-negative).
