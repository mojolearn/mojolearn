# lane/apple-fast-tsa: the ARIMA fit's evaluation (FAST on Apple, `-D` switches, default OFF)

Written without a Mojo toolchain (cloud peer); the first M3 build is the compile check (binding
`arima`, FAST). IDENTICAL compiles main's code unchanged: every switch is a comptime alias of
`GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator() and is_defined[...]()`.

The board lane is `autoarima` (`tools/bench_board_algos.py`, AFC_FAMILY=algos, taxi-hourly): every
candidate order is an `ARIMA.fit` through `_mojolearn_arima` -> `batched_fit_x` -> `batched_min_lbfgs`,
so each switch is exercised hundreds of times per lane run. (`arima` in `tools/bench_board_more.py`
is classical2 on synthetic only and already 3x ahead of statsmodels; not requested.)

| switch | site | what it changes under FAST on Apple |
|---|---|---|
| `MOJOLEARN_ARIMA_FAST_LLONLY=1` | `arima/impl/batched_kalman.mojo` `KALMAN_LL_ONLY`, `batched_kalman_loop_kernel[RD_C, LL_ONLY]`, `_launch_loop_ll_only`; `batched_arima.mojo::batched_loglike_grad_host` passes `ll_only=True` | the fit's stacked evaluations run the loop kernel instantiated with `LL_ONLY = True`: the per-step `pred` / `vs` / `Fs` stores (three device stores per step per thread, read by nobody inside the fit) are gone; the recurrence and the log-likelihood sum are the same statements. predict / forecast / the card keep the storing kernel. |
| `MOJOLEARN_ARIMA_FAST_EVAL_WS=1` | `arima/impl/fast_eval_ws.mojo` (`FastEvalWS`, `ew_stack_kernel`, `ew_grad_kernel`); `batched_kalman.mojo::fast_kalman_into`; `batched_fit.mojo` (`_eval_ws`, `_eval_ws_packed`, the `Optional[FastEvalWS]` in `batched_min_lbfgs`) | the evaluation's ~45 device buffers, the (N + 1) series copies, four host buffers and 2N small launches per candidate point become ONE workspace per solve: one candidate upload, one stacking kernel over (N + 1) x batch members (`perturb_kernel`'s statement), unpack, Jones, the filter's three launches into the held workspace, one gradient kernel (`grad_kernel`'s statement), the same four readbacks and one synchronize. FIT_COMPACT's repack rebuilds the workspace at the packed size from `d_y_c`. Infeasible members mark -inf through `_mark_infeasible` as main's path does. |

Both compose: `tsa-both-autoarima` is the two defines together (`fast_kalman_into` takes the LL_ONLY
instantiation when both are on).

## Causes (what was slow)
- `batched_loglike_grad_host` (`arima/impl/batched_arima.mojo`) is host and allocator time, not
  filter time: at the board's 64 ARMA series the Kalman pass is ~320 threads x 2,000 steps, while
  each evaluation creates y_ext, x_ext, d_grad, two ARIMAParams (7 buffers each), a
  KalmanWorkspace (26 buffers), four host buffers, and issues 2 (N + 1) copies plus 2N launches.
- `batched_kalman_loop_kernel` (`batched_kalman.mojo` ~:596-616) stores pred, vs and Fs every step
  from a one-thread-per-member serial recurrence with nothing to hide the stores behind.

## Keep rule
Faster on the M3 at `autoarima` taxi-hourly with quality (the board's forecast score) within FAST
run-to-run spread -> the define becomes the FAST default (the `is_defined` term dropped) -> main.

## Risky compile sites (first M3 build)
- `batched_fit.mojo`: `Optional[FastEvalWS]` over a `Movable`-only struct, `ews.value()` passed to a
  `mut FastEvalWS` parameter, `ews = None` then `ews = FastEvalWS(...)` on repack (the
  `Optional[ByteLogitsScratch]` idiom of `training/byte_lm_logits.mojo:440-460`).
- `fast_eval_ws.mojo`: `comptime if not KALMAN_FAST_EVAL_WS: raise ... else: <body>` inside a method;
  the struct holds `HostBuffer` fields (as `SplitStaging` in the trees builder does).
- `batched_kalman.mojo`: the two-parameter instantiation `batched_kalman_loop_kernel[1, True]`;
  `fast_kalman_into` / `_launch_loop_ll_only` take `mut` DeviceBuffer / struct arguments and pass bare
  `buf.unsafe_ptr()` only into `enqueue_function` (the file's idiom), never into a pointer-typed helper.
- `fast_eval_ws.mojo` imports `_mark_infeasible` from `batched_arima` (which imports `batched_kalman`,
  never `fast_eval_ws`): no cycle.
