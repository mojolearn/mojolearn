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

Every claim is a file in this directory. The CPU pod is `x86-runpod/` (pod
`exrdkn3qlq35bj`, AMD EPYC 9655P, 8 vCPU, 335 s billed, $0.0223, DELETE
verified in `x86-runpod/teardown.txt`).

| file | verdict |
|---|---|
| `metal/metal.new-base.json` | the two new lanes on the base fixture, Metal, at 20e786447: `cells=2 stable=2 moved=0 refused=0`, infer, model and batch stable on both, and `reload` equal to `infer` (a saved model reloads and predicts the same bytes) |
| `metal/diff.old-base.txt` | THE EXISTING LANES ARE UNCHANGED: arima, arima-011 and arima-seasonal-c on the base fixture against the three GPU columns read `IDENTICAL x4` on train, infer and batch (`summary: IDENTICAL=27`). The three `model` rows read ONE-COLUMN because those columns predate `ARIMA.save` |
| `../../classical_host/2026-09-15-apple-m4-arima-exog/` | `classical_host_gate.py record` on the M4's Metal set, BOTH lanes, nine fixtures each (18 fixture directories, every `expected.json` parses), 9m07s. Listed in `FORECAST_RECORDED` |
| `gate_cpu_local.{txt,json}` | the same recording through THIS MAC's host bindings: `gate verdict IDENTICAL (18 fixtures, 0 GPU columns, exit 0)`, every surface EQUAL including `beta`, `predict_in_sample` and `predict_straddle` |
| `x86-runpod/diff.record-vs-cpu.txt` | THE CPU COLUMN: every cell of both new lanes on all nine fixtures reads `OWED x1` against the three GPU columns, which predate the lanes. OWED is the right verdict and needs a CPU column to exist, which this run supplies |
| `x86-runpod/owed_sabotage_check.txt` | THE OWED CHECK: `owed verdict OK (72 of 72 owed cell part(s) moved, 0 failure(s))` -- the sabotage moves every owed part of both lanes |
| `x86-runpod/classical_gate_cpu.{txt,json}` | the Metal recordings (18 new fixtures + the three existing ARIMA base fixtures) through the x86 host bindings against three GPU columns: `gate verdict IDENTICAL (21 fixtures, 3 GPU columns, exit 0)` |
| `metal/diff.cpu-vs-sabotage.txt` | THE VALUE SABOTAGE, CPU production column against CPU whole-host sabotage column: `summary: DIVERGENT=18`, `(infer/model): DIVERGENT=36`, `(batch): DIVERGENT=18`. Every fixture of both lanes moves |
| `x86-runpod/diff.cpu-vs-exog-sabotage.txt` | THE EXOG ARITHMETIC'S OWN CONTROL (`MOJOLEARN_ARIMA_EXOG_SABOTAGE`, the lowest bit of every finite observation intercept and nothing else): DIVERGENT on both lanes. On `arima-exog` the parts that differ are `beta, mu, sigma2, forecast` and **`ar` still AGREES**, which is what pins the divergence to the exogenous arithmetic rather than to the fit at large |
| `x86-runpod/classical_gate_sab.txt`, `classical_gate_forecast_only_sab.txt` | `EXPECTED MISMATCH SEEN (18 fixtures)` for the whole-host sabotage set and for the forecast binding alone |
| `x86-runpod/diff.record-vs-cpu-arima-base*.txt` | the existing ARIMA lanes on the base fixture with the CPU column: `IDENTICAL x4`, and DIVERGENT under sabotage |
| `x86-runpod/fit_symbols.txt` | `forecast 0`, `arima 2`: no fit symbol in the shipped forecast binding, with the reference binding as the witness that a zero means something |
| `statsmodels_agreement.{txt,json}` | AGREEMENT WITH statsmodels SARIMAX, REPORTED, NOT TUNED TO (see below) |

## Agreement with statsmodels SARIMAX

`statsmodels_agreement.py`, three cases on one synthetic series with a real
regression component (`beta_true = [2.0, -0.5]`). Our float32 fit against
their float64 one, different optimizers and different start values.

| case | beta max_abs | ar max_abs | forecast max_abs | sigma2 |
|---|---|---|---|---|
| `(1,0,0) trend=n` | 6.0e-04 | 9.8e-04 | 1.4e-03 | 0.255855 against 0.256203 |
| `(1,0,0) trend=c` | 4.4e-04 | 9.3e-04 | 2.0e-03 | 0.255925 against 0.256202 |
| `(1,1,0)(1,0,0,4)` | 5.7e-04 | 4.6e-03 | 1.7e-03 | 0.303159 against 0.303598 |

THE INTERCEPT CHECK IS THE INTERESTING ONE. cuML's `mu`, which this package
spells `trend='c'`, is a STATE intercept, so the implied mean is
`mu / (1 - phi)`. Measured: ours `0.00073671`, statsmodels' `intercept`
`0.0007397`. They agree to 4e-06, which confirms DEVIATION 993's reading of
what `mu` is rather than assuming it.

A CORRECTION, RECORDED RATHER THAN QUIETLY FIXED. The first run of this
script read `forecast max_abs 4.38` on the seasonal case while every
coefficient agreed to 1e-03. That shape is a scale mismatch, not a bad fit:
statsmodels with `simple_differencing=True` fits AND PREDICTS on the
DIFFERENCED series, while this package undifferences its forecast back to the
level. With `simple_differencing=False` their forecast is on the level and
the gap falls to 1.7e-03, in line with the other two cases. The script now
says so at the call.

## What has run so far (2026-09-15)

- Both host bindings and the Metal binding compile with the exogenous arms.
  The shipped forecast host binding grows from 298,536 to 315,256 bytes.
- `test_arima_exog.py`: 7 source checks and the runtime check pass on the M4's
  Metal set (8 passed), and the runtime check passes again on a staged
  CPU-only package against the reference arima host binding (the fit) and the
  shipped forecast host binding (the prediction).
- TWO DEFECTS THE RUNTIME CHECK FOUND, both fixed and both invisible to a
  source read: `_FORMATS` did not carry `mojolearn-arima-2`, so
  `mojolearn.host_model` refused a saved exog model on the one install that
  needs it; and `HostARIMA._HOST_ARRAYS` did not name `_exog`, so
  `model_sha256` did not cover the regressors.

## Owed, and why

- THE NVIDIA AND AMD COLUMNS of both new lanes, and their `model` cells for
  `mojolearn-arima-2`. Owed to the next release record; the NVIDIA slot that
  night belonged to the 0.8.6 record and no NVIDIA pod was rented here.
- THE POD'S pytest, statsmodels AND WHEEL STEPS did not run, each for an
  environment reason recorded in `x86-runpod/step_exit_codes.txt`: the leg
  installed the `default` pixi env only, so `.pixi/envs/test/bin/python` was
  absent (`pytest 127`); that env carries no `pip`, so statsmodels could not
  be installed (`No module named pip`) and no `setuptools`, so the wheel
  build failed. The tests were run on the Mac instead (156 source + the two
  runtime checks) and the agreement was run on the Mac against the Metal set.
  The INSTALLED-WHEEL gate is genuinely owed: what is measured here is the
  binding size delta, not a pip install.
- `x86-runpod/diff.record-vs-cpu-sabotage.txt` reads `ONE-COLUMN=18`, not
  DIVERGENT, because it compares the SABOTAGE CPU column against three GPU
  columns that hold no cells for these lanes. It is kept for completeness;
  the instruments that carry the sabotage claim are the owed check and
  `metal/diff.cpu-vs-sabotage.txt`.
- The host-infer diff (`MOJOLEARN_IDENTITY_HOST_INFER`) exited 1 saying the
  CPU column "is not STABLE over two or more repeats (STABLE, 1 repeat(s))".
  That is a flag mistake in the pod command, `--repeats 1`; the column itself
  is stable and the primary column ran with `--repeats 2`.
