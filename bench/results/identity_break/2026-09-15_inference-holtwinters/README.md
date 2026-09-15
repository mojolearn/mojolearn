# Public CPU inference for saved Holt-Winters models (2026-09-15)

Branch `lane/inference-holtwinters`, final code at 6238827b9 (every pod JSON records its commit).
Lanes `holtwinters` and `holtwinters-multiplicative` join the shipped `forecast` host family
(`_mojolearn_forecast_host`). The line search change this waited for (e31994918, DEVIATION 2717)
moved no cell of these fixtures (`../2026-09-15_holtwinters-linesearch-fix/README.md`).

## Public API

- `ExponentialSmoothing.save(path)` and `ExponentialSmoothing.load(path)`, format
  `mojolearn-holtwinters-1`: the packed level, trend and season buffer, `sse`, `alpha`, `beta`,
  `gamma`, `n_iter`, `criterion`, `meta` [n, ts_num, seasonal_periods, start_periods], `eps`,
  `seasonal`, `numeric_mode`. `endog` is not saved; nothing a loaded model answers reads it.
- `ExponentialSmoothing.predict(start=0, end=None, index=None)`: the in-sample one-step
  predictions at times `[start, min(end, n))` and the forecast after `n`, so `predict(n, n + h)`
  is `forecast(h)` byte for byte. Return shapes are `forecast`'s. Times `t < 2 *
  seasonal_periods` are the canonical quiet NaN, by name: their prediction reads the
  decomposition's start state, which neither the GPU fit nor the file keeps.
- `mojolearn.host_model(path)` returns `HostExponentialSmoothing` bound to the forecast host
  binding on any machine; on a CPU-only install the plain class routes `_mojolearn_tsa` there
  when the reference tsa binding is not built. `fit` refuses; `kpss_test` refuses by name.

The arithmetic is `holtwinters/host/hw_predict.mojo` (imports only `checks/numerics.mojo`).
`bindings/holtwinters_host_predict.mojo` registers `holtwinters_forecast` and
`holtwinters_predict` for the forecast host and the reference tsa host, and `holtwinters_predict`
for the GPU tsa binding. `hw_oracle.mojo::oracle_forecast` forecasts through the same body.

## Metal recording (Apple M4, through the Mac GPU lock)

| file | verdict |
|---|---|
| `metal_smoke.txt` | fit, save, load, `predict(0, n)` (NaN before 2f, finite after), straddle = in-sample tail + forecast head, loaded bytes equal, both seasonal modes |
| `apple-m4-metal.base.json`, `diff.record-apple-vs-metal-base.txt` | the rebuilt identical `_mojolearn_tsa.so` on the base fixture against the 166-lane apple-m4 column: IDENTICAL=2 on train, infer and batch |
| `classical_host_record_metal.txt` | `classical_host_gate.py record` of both lanes on all nine fixtures into `bench/results/classical_host/2026-09-15-apple-m4-holtwinters` (18 fixtures): the forecast pair, `predict_in_sample`, `predict_straddle`, level, trend, season, alpha, beta, gamma, sse |

## The CPU columns: `x86-runpod-4/` (pod r67b1clhums7vd, AMD EPYC, 8 vCPU)

`tools/runpod_cpu_leg.sh --build core,tsa,arima,forecast --sabotage-build core,tsa,arima,forecast`,
command `x86-runpod-4/user_cmd.sh`, 86 s billed, $0.0057, DELETE verified (`teardown.txt`).
Files under `x86-runpod-4/remote/leg_out/`; every step's exit code is in `step_exit_codes.txt`.
Columns against the 166-lane record's apple-m4, nvidia-h100-sm_90a and amd-mi325x-gfx942.

| file | verdict |
|---|---|
| `cpu-x86.json`, `diff.record-vs-cpu.txt` | 9 fixtures x 2 lanes, 2 repeats, `--require-columns 4 --owed-json`: `summary: IDENTICAL=18`, `(infer/model): IDENTICAL=18, OWED=18`, `(batch): IDENTICAL=18`, `require-columns 4 ... OK (18 OWED)` |
| `owed_cells.json` | the 18 OWED parts: the `model` cells of both lanes on every fixture, whose save format is new |
| `cpu-x86.sabotage.json`, `diff.record-vs-cpu-sabotage.txt` | `-D MOJOLEARN_HOST_SABOTAGE=1` host set: `summary: DIVERGENT=18`, `(infer/model): DIVERGENT=18`, `(batch): DIVERGENT=18`; the diff exits 1 |
| `owed_sabotage_check.txt` | `owed verdict OK (18 of 18 owed cell part(s) moved, 0 failure(s))` |
| `cpu-x86.host-infer.json`, `diff.record-vs-cpu-host-infer.txt` | `MOJOLEARN_IDENTITY_HOST_INFER`: the infer cell asked of `host_model(<saved file>)`, the forecast host binding: `summary: IDENTICAL=18`, `(infer/model): IDENTICAL=18`, `(batch): IDENTICAL=18` |
| `classical_gate_cpu.{txt,json}` | the Metal recordings (18 Holt-Winters fixtures and the three ARIMA lanes' base fixtures) through the source tree's host bindings, 3 GPU columns: `gate verdict IDENTICAL (21 fixtures, 3 GPU columns, exit 0)` |
| `classical_gate_sab.{txt,json}` | the sabotage host set, `--every-lane`: `EXPECTED MISMATCH SEEN (21 fixtures)`, 84 surfaces DIFFER |
| `classical_gate_forecast_only_sab.{txt,json}` | THE VALUE SABOTAGE ON THE FORECAST PATH ALONE: the production host set with only `_mojolearn_forecast_host.so` swapped for its sabotage build (`HW_PREDICT_SABOTAGE`, the lowest bit of every finite forecast and prediction flipped): `EXPECTED MISMATCH SEEN (18 fixtures)`, 72 surfaces DIFFER (the forecast, both predictions and the identity hash on every fixture; the fitted state read from the file stays EQUAL) |
| `cpu-x86.arima-base.json`, `diff.record-vs-cpu-arima-base.txt` | spot check of the existing ARIMA forecast lanes on the base fixture: the 3 CPU cells IDENTICAL x4 with the three columns (the summary's other 24 are fixtures this column did not run) |
| `cpu-x86.arima-base.sabotage.json`, `diff.record-vs-cpu-arima-base-sabotage.txt` | the same under sabotage: the 3 base cells DIVERGENT |
| `pytest.txt` | `test_holtwinters_inference`, `test_host_surface`, `test_cpu_inference_boundary`, `test_cpu_training_arima` with the host bindings built: `165 passed, 1 deselected` (the deselected check needs recording directories the pod does not ship; it passes in the tree) |
| `fit_symbols.txt` | `nm -C` count of fit, decomposition and line search names: forecast 0, tsa 3, arima 2 (the reference bindings are the witness that a zero means something) |
| `wheel.txt`, `classical_gate_installed_wheel.{txt,json}`, `installed_smoke.txt` | an isolated venv install of a Linux test wheel holding the shipped core and forecast host bindings: `gate verdict IDENTICAL (21 fixtures, 3 GPU columns, exit 0)` through the installed binding; the plain class and `host_model` answer the same bytes, `fit` and `kpss_test` refuse by name |

## Wheel

The Linux x86-64 test wheel, the same stage with this branch's and origin/main's
`_mojolearn_forecast_host.so` (`x86-runpod-4/remote/leg_out/wheel.txt`):

| | forecast host .so | compressed in the wheel | wheel |
|---|---|---|---|
| origin/main | 267,200 | 69,213 | 1,036,273 |
| this branch | 298,536 | 78,756 | 1,045,819 |
| change | +31,336 | +9,543 | +9,546 |

No reference tsa binding ships.

## Earlier pods (kept; not the final columns)

- `x86-runpod/` (upsxf62678fbvz, 244 s, $0.0163) at 225106ae4: the owed check read 12 of 18,
  because the fit sabotage's split SSE multiply-add leaves the fitted bytes unchanged on
  denormal, denormal_ftz and wide, so those files could not move. The wheel steps had no pip.
- `x86-runpod-2/` (99pu33xitq3txa, 131 s, $0.0087) at 204e1561f: that commit did not compile the
  tsa host (`build_tsa.log`), every Holt-Winters fit REFUSED; its identity verdicts are void.
- `x86-runpod-3/` (ovb9le1qdopo4q, 206 s, $0.0137) at 27b6a8505: owed 18 of 18, but the sabotage
  column read DIVERGENT=16: a flip of the fitted components followed by the forecast binding's
  output flip restored holtwinters' forecast bytes on denormal and denormal_ftz. 6238827b9 flips
  only the per-series floats.

Every pod's DELETE was verified (`teardown.txt` in each directory). Four pods, 667 s, $0.0444.

## Owed

- NVIDIA and AMD model cells for the new format (the 18 OWED parts in `owed_cells.json`).
- NVIDIA and AMD recordings of `predict_in_sample` and `predict_straddle` (Apple only).
- The CPU identity gate workflow checks `FORECAST_RECORDED` through `saved_model_recorded()`; the
  new recording rides that list, and the workflow builds every manifest family.
