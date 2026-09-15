# lane/arima-exog: exogenous regressors on ARIMA

Branch `lane/arima-exog`, worktree
`/private/tmp/claude-501/-Users-andrewhendel-CascadeProjects-mojolearn/e52730fc-0ee7-4bf3-8741-5fa0e0f1876f/scratchpad/wt-arima-exog`.
Resumable: every step below says what is done and what the next command is.

## What the lane adds

`ARIMA.fit(y, exog)`, `forecast(steps, exog)` and `predict(start, end, exog)`, regression with
ARIMA errors, following cuML (`python/cuml/cuml/tsa/arima.pyx:342-362`, `:688-723`;
`cpp/src/arima/batched_arima.cu:117-157`, `:854-931`; `cpp/src/arima/batched_kalman.cu:921-972`,
`:181`, `:286`; `cpp/src_prims/timeSeries/arima_helpers.cuh:73-119`, `:262-292`).

- `beta` is packed after `mu` (`arima_common.cu:24-36`) and passes through the Jones transform
  untouched, copied as `mu` is.
- The regressors are differenced beside `y` under simple differencing; their future values are
  differenced against their past (`prepare_future_data`).
- `estimate_x0` regresses the differenced `y` on the differenced regressors before the ARMA
  start values, zeroes `beta` when the solve refuses, and subtracts the fitted component.
- The Kalman filter adds `x_t beta` as the observation intercept to every prediction and
  forecast.

DEVIATIONS: 994 `EXOG_MAX = 17`; 995 the closed cuBLAS gemms' fold is ours (serial ascending
fma from 0), used by both the observation intercept and the start-value regression; 996 the
exog layout `(batch_size, n_obs, n_exog)`; 997 a non-finite regressor is refused by name; 998
the saved-model format `mojolearn-arima-2`, with `mojolearn-arima-1` unchanged byte for byte
for a fit without regressors.

## Files

Device: `arima/impl/tsa/arima_common.mojo` (EXOG_MAX, `beta`, packing), `arima/impl/batched_kalman.mojo`
(`obs_intercept_kernel`, the loop's `has_exog` arms, `batched_kalman_filter_x`),
`arima/impl/estimate_x0.mojo` (`exog_regression_kernel`, `estimate_x0_x`),
`arima/impl/batched_arima.mojo` (`_x` entries), `arima/impl/batched_fit.mojo` (`batched_fit_x`),
`arima/impl/timeSeries/arima_helpers.mojo` (`prepare_future_data`, the beta copy),
`arima/estimator.mojo`. Host: `arima/host/arima_oracle.mojo`. Boundary:
`bindings/arima_exog_layout.mojo` (new, shared), `bindings/_mojolearn_arima.mojo`,
`bindings/_mojolearn_arima_host.mojo`, `bindings/arima_host_predict.mojo`. Python:
`python/mojolearn/_arima_impl.py`. Harness: `tools/identity_break.py`,
`tools/classical_host_gate.py`, `python/mojolearn/host_surface.py`.

Every entry point without `_x` is the `n_exog = 0` door the checks and the card call; it hands
the `_x` entry one-float placeholders nothing reads, so no existing caller changed.

## Steps

1. DONE. Code, tests (`python/mojolearn/tests/test_arima_exog.py`), lanes `arima-exog` and
   `arima-exog-seasonal`, manifest, CHANGELOG, SUPPORT_MATRIX, NOT_IMPLEMENTED row removed.
2. DONE. The reference arima host binding compiles
   (`bindings/build_arima_host.sh`, scratchpad `arima-exog/host1`).
3. IN PROGRESS. The Metal `_mojolearn_arima.so` build (`bindings/build_arima.sh`), which runs
   its own smoke (fit, forecast and predict with regressors, the refusals).
4. OWED. Metal evidence: the two new lanes on the base fixture, the existing arima lanes
   unchanged on the base fixture, `classical_host_gate.py record` of the new lanes.
5. OWED. The CPU column and sabotage column on one RunPod CPU pod
   (`tools/runpod_cpu_leg.sh --build arima,forecast --sabotage-build arima,forecast`), the owed
   check, and the saved-model gate from the installed test wheel.
6. OWED. One small RunPod NVIDIA pod for the H100 column of the two new lanes.
7. OWED. statsmodels `SARIMAX` agreement on coefficients and forecasts, reported not tuned to.
8. OWED. Merge to main (docs_facts --check, wheel_ci pins, inventory), then remove the worktree.

## Commands

    # Metal (through the Mac GPU lock, one at a time)
    bash <scratchpad>/mac_slot.sh metal python3 tools/identity_break.py \
        --lanes arima-exog,arima-exog-seasonal --fixtures base --repeats 1 --json metal.json

    # CPU pod
    bash tools/runpod_cpu_leg.sh --lane arima-exog --build arima,forecast \
        --sabotage-build arima,forecast --cmd-file <cmd>.sh --rent
