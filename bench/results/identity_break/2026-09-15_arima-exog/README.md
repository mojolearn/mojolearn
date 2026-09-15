# Exogenous regressors on ARIMA (2026-09-15)

Branch `lane/arima-exog`. Lanes `arima-exog` and `arima-exog-seasonal` join the `arima` host
family (CPU training) and the shipped `forecast` host family (public CPU inference from a saved
model). The existing `arima`, `arima-011` and `arima-seasonal-c` lanes are unchanged and are
spot checked on the base fixture.

## Public API

- `ARIMA.fit(y, exog)`, `ARIMA.forecast(steps, exog)`, `ARIMA.predict(start, end, exog)`.
  `exog` is `(batch_size, n_obs, n_exog)` (DEVIATION 996); a 1-D `y` also takes `(n_obs,)` and
  `(n_obs, n_exog)`, and a 2-D `y` takes `(batch_size, n_obs)` for one regressor.
- New attributes `beta_` `(batch_size, n_exog)` and `n_exog_`. `params_` packs `beta` after
  `mu`, as `arima_common.cu:24-36` packs it, so `N = p + q + P + Q + k + n_exog + 1`.
- Future values are required exactly when the model has a regression and the prediction passes
  `n_obs`, and refused otherwise, which is cuML's rule and its words (`arima.pyx:688-696`).
- Saved models: a fit WITHOUT regressors is still `mojolearn-arima-1`, byte for byte what it
  was; a fit WITH them is `mojolearn-arima-2` and carries the regressors and `n_exog`
  (DEVIATION 998). `mojolearn.host_model(path)` predicts from either on a CPU-only install.

## Method, and where it differs from the reference

Regression with ARIMA errors: the regressors are differenced beside `y`, `beta` is started by a
least-squares regression of the differenced `y` on the differenced regressors before the ARMA
start values (`batched_arima.cu:854-931`) and then fitted jointly with them by the L-BFGS, and
`x_t beta` is the observation intercept the Kalman filter adds to every prediction and forecast
(`batched_kalman.cu:921-972`, `:181`, `:286`). Future regressors are differenced against their
past (`arima_helpers.cuh:262-292`).

| DEVIATION | what |
|---|---|
| 994 | `EXOG_MAX = 17`, the Householder QR's column bound; cuML bounds `n_exog` by nothing |
| 995 | the two closed cuBLAS gemms (observation intercept, start-value regression) are ours to spell: serial ascending fma from zero |
| 996 | `exog` is `(batch_size, n_obs, n_exog)`, permuted to cuML's strided-batched layout by one file |
| 997 | a non-finite regressor is refused by name with its series, row and regressor; theirs reads `exog` with `ensure_all_finite=False` and has no missing-value arm for it |
| 998 | the saved-model format is versioned so no existing ARIMA model file changes |

## Evidence

OWED, filled as each step lands. Every claim below is a file in this directory.

| file | verdict |
|---|---|
| `metal_test.txt` | `test_arima_exog.py` on the M4's Metal set: the fit, the forecast pair, the refusals, save and load |
| `apple-m4-metal.json`, `diff.*` | the two new lanes on the base fixture, and the three existing ARIMA lanes on the base fixture against the 166-lane apple-m4 column |
| `classical_host_record_metal.txt` | `classical_host_gate.py record` of the two lanes into `bench/results/classical_host/2026-09-15-apple-m4-arima-exog` |
| `x86-runpod/` | the CPU column, the sabotage column, the exog-only sabotage column, the owed check, the saved-model gate, the installed test wheel |
| `statsmodels_agreement.{txt,json}` | agreement with statsmodels `SARIMAX` on the coefficients and the forecasts, REPORTED, not tuned to (`statsmodels_agreement.py`) |
| `nvidia/` | the H100 column of the two new lanes |

## Owed

- The AMD column of the two new lanes (no AMD box this lane).
- The NVIDIA and AMD `model` cells of `mojolearn-arima-2`.
