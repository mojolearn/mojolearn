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
| `metal/metal.new-base.json` | the two new lanes on the base fixture, Metal, at 20e786447: `cells=2 stable=2 moved=0 refused=0`, and infer, model and batch stable on both. `reload` equals `infer`, so a saved model reloads and predicts the same bytes |
| `metal/diff.old-base.txt` | THE EXISTING LANES ARE UNCHANGED: arima, arima-011 and arima-seasonal-c on the base fixture against the three GPU columns of the 166-lane record read `IDENTICAL x4` on train, infer and batch (`summary: IDENTICAL=27`, `(batch): IDENTICAL=27`). The three `model` rows read ONE-COLUMN because the record's columns predate `ARIMA.save` (`n/a:no-save` there, a hash here), which is the saved-model lane's shape and not this lane's doing |
| `metal/diff.new-base.txt` | the new lanes against the same three columns: ONE-COLUMN on every cell, because the record predates the lanes and the GPU columns read `(not run)`. `--require-columns 4` FAILS here by design and the tool says why: OWED needs a CPU column to hash the cell ("not OWED: no CPU column hashes it"). The OWED verdict is taken on the pod, where the CPU column exists |
| `../../classical_host/2026-09-15-apple-m4-arima-exog/` | `classical_host_gate.py record` on the M4's Metal set. PARTIAL AND SAID TO BE: `arima-exog` is nine of nine fixtures, `arima-exog-seasonal` is what finished before the run was stopped for a Mac restart (the GPU was degraded by a command-queue leak, which is why it crawled). Every committed fixture has an `expected.json` that parses; the rest is OWED and `docs/lanes/LANE_STATUS_lane-arima-exog.md` carries the exact command |
| `x86-runpod/` | the CPU column, the sabotage column, the exog-only sabotage column, the owed check, the saved-model gate, the installed test wheel |
| `statsmodels_agreement.{txt,json}` | agreement with statsmodels `SARIMAX` on the coefficients and the forecasts, REPORTED, not tuned to (`statsmodels_agreement.py`) |
| `nvidia/` | the H100 column of the two new lanes |

## What has run so far (2026-09-15, at 20e786447)

- Both host bindings and the Metal binding compile with the exogenous arms. The shipped
  forecast host binding grows from 298,536 to 315,256 bytes.
- `test_arima_exog.py`: 7 source checks and the runtime check pass on the M4's Metal set
  (8 passed), and the runtime check passes again on a staged CPU-only package against the
  reference arima host binding (the fit) and the shipped forecast host binding (the
  prediction), which is where `mojolearn.host_model` is exercised.
- TWO DEFECTS THE RUNTIME CHECK FOUND, both fixed and both invisible to a source read: the
  saved-model format registry `_FORMATS` did not carry `mojolearn-arima-2`, so `host_model`
  refused a saved exog model on the one install that needs it; and `HostARIMA._HOST_ARRAYS`
  did not name `_exog`, so `model_sha256` did not cover the regressors.

## Owed

- The rest of the `arima-exog-seasonal` recording (see the table above).
- The AMD column of the two new lanes (no AMD box this lane).
- The NVIDIA and AMD `model` cells of `mojolearn-arima-2`.
