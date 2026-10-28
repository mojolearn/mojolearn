# peer6-forecast-pandas HANDOFF (2026-10-28)

## 1. Prophet pair: lane/peer6-prophet-20261027 @ d950eac50c (pushed, contains origin/main 4771b55daa)
- b00f153308 + companion 2a63c7c545 = ONE unit (never b00 alone). Companion: leaf-depth/product-capacity admission
  (providers/python/prophet_buffers_pyapi.mojo) + classical_connect_numpy on the lifecycle context (prophet_lifecycle_pyapi.mojo).
- Compile: full xseq dispatcher exit0 233.30s -j10, experiments/peer6_prophet_admission_20261028/compile.json
  (attempt7; the 7 changed source files' hashes match HEAD). d950eac50c only adds the delivery record.

## 2. Forecast pandas admission: lane/peer6-timeseries-20261027 @ 65d659a023 (pushed)
- 12f691b2b1 merge origin/main (lane side kept for cherry-picked ARIMA/xseq files; main's Holt-Winters bridge lines kept).
- ac2484736f: bindings/classical_pandas.mojo:172 native_pandas_f32 gains `refuse_double` (raised before conversion when pandas
  promotes beyond float32); providers/python/arima_buffer_pyapi.mojo arima_buffer converts pandas (+ `converted` flag);
  forecast_matrix_pyapi.mojo _matrix (VAR/STL) passes its float64 refusal; autoarima_lifecycle_pyapi.mojo _series_buffer (endog);
  xseq_forecast_native.mojo _series (Croston/Theta/ETS/GARCH); tsa_arima_native.mojo _series_major input_copied_ True for pandas.
- Compiles (CPU/O0/identical/-j10): xseq dispatcher exit0 42s, TSA dispatcher exit0 27s; experiments/peer6_forecast_pandas_20261028/compile.json.
- Order: after e456212ca7 + 4b9999bee4 (same lane).

## 3. Metrics: lane/peer6-metrics-20261027 @ 20089632e6 (pushed)
- e26b84da4a: arena_prog_native._metrics_run_fields(fields,mode,context) (bindings/arena_prog_native.mojo:471); _metrics_run keeps
  Prog API; mcls_common.mojo:703, metrics_exp_common.mojo:319, metrics_reg_native.mojo:268 pass a native dict (no Python Prog).
- Includes de7fea9506, merges origin/main (1c311ac20d) and linalg helper a3a8150111 (6592b962e2).
- Compile: checks/metrics_program_fields_compile.mojo (metrics_boundaries closure + Prog wrapper) exit0 42s;
  experiments/peer6_metrics_program_fields_20261028/compile.json.

## 4. Resample pandas dtypes: lane/peer6-resample-pandas @ a1cb276bf9 (pushed)
- 4d6f90f829: finished draft resample_pandas_pyapi.mojo (categorical dictionaries, SparseArray, IntervalArray, extended precision);
  bf93046c4b merge origin/main. Compile peer6_resample_pandas_compile exit0 at bf93046c4b;
  experiments/peer6_resample_pandas_dtypes_20261028/compile.json.

## Not covered / pending
- No runtime, identity or GPU compile anywhere. Other importers of classical_pandas.mojo not recompiled (default-compatible param).
- Other _metrics_run callers (trees_public_score, classical_labels) not recompiled (API unchanged).
- Resample pandas reader still not wired into the public resample row op; Arrow temporal/decimal/nested still refuse.
- Metric pandas input gaps remain.
- Rented boxes: none. Evidence: ~/mojolearn-evidence/peer6-forecast-pandas-20261028/ (compile.sh, per-root logs/binaries/source.tgz).
