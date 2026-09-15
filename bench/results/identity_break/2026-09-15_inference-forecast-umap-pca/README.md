# Public CPU inference: ARIMA, UMAP transform, whitened full-SVD PCA (2026-09-15)

Branch `lane/inference-forecast-umap-pca`, code at ae97a3e0f (every JSON and recording
records it). Lanes: `arima`, `arima-011`, `arima-seasonal-c`, `umap`, `pca-full-whiten`.
Holt-Winters is not in this group; it waits for the change to its fit's line search.

Where it ran: the Apple M4, one core (nice 19, one-thread knobs, `-j 1`), shared machine,
one process at a time. No box rented.

| file | verdict |
|---|---|
| `cpu-apple-m4.json` | CPU column, host set of reference bindings, 9 fixtures, 2 repeats: cells=45 stable, infer stable=45, model stable=45 (the new ARIMA and UMAP save and load round trip, RELOAD checked), batch stable=36, n/a=9 (umap, batch-dependent by contract) |
| `diff.record-vs-cpu.txt` | against the 166-lane record's apple-m4, nvidia-h100-sm_90a and amd-mi325x-gfx942 columns, `--require-columns 4 --owed-json`: `summary: IDENTICAL=45`, `summary (infer/model): IDENTICAL=54, OWED=36`, `summary (batch): IDENTICAL=36, N/A=9`, `require-columns 4 ... OK (36 OWED)` |
| `owed_cells.json` | the 36 OWED parts: the `model` cells of arima, arima-011, arima-seasonal-c and umap, whose save format is new, so no committed GPU column hashes them (pca-full-whiten's model cells are IDENTICAL x4) |
| `cpu-apple-m4.sabotage.json`, `diff.record-vs-cpu-sabotage.txt` | `-D MOJOLEARN_HOST_SABOTAGE=1` estimators, metrics, arima and forecast bindings: `summary: DIVERGENT=45`, `summary (infer/model): DIVERGENT=54`, `summary (batch): DIVERGENT=36`, diff exits 1 |
| `owed_sabotage_check.log` | `tools/cpu_identity_gate_check.py owed`: 36 of 36 OWED parts moved under sabotage |
| `classical_gate_cpu.{log,json}` | `tools/classical_host_gate.py check` of the Metal recordings (below) from the source tree's host bindings, with the three GPU columns: `gate verdict IDENTICAL (45 fixtures, 3 GPU columns)`, every identity hash EQUAL on all three columns |
| `classical_gate_sabotage.{log,json}` | the same check on the sabotage set: `EXPECTED MISMATCH SEEN`; every forecast, in-sample and straddling prediction, transform and inverse transform DIFFER on all 45 fixtures, while the parameters read from the saved files stay EQUAL |
| `classical_gate_installed_wheel.{log,json}` | an isolated `pip install --target` of a locally built test wheel holding exactly the nine wheel host families (no GPU set, so the CPU-only route): `gate verdict IDENTICAL (45 fixtures, 3 GPU columns)`; the saved ARIMA models were served by the installed `_mojolearn_forecast_host.so`, UMAP by `_mojolearn_metrics_host.so`, PCA by `_mojolearn_estimators_host.so` |

Recordings: `bench/results/classical_host/2026-09-15-apple-m4-umap-pca` (pca-full-whiten, umap)
and `bench/results/classical_host/2026-09-15-apple-m4-arima` (the three ARIMA lanes), recorded
through `classical_host_gate.py record` on the M4's Metal identical set (vendor `metal`). They were
one directory, `2026-09-15-apple-m4-forecast-umap-pca`, when the reports above were written, and
were split by lane afterwards with no file changed. A Metal identity run on the base fixture read
the committed apple-m4 infer and model cells IDENTICAL for all five lanes first, so the Metal
binaries that recorded were the committed column's.

What each ARIMA recording holds beside the identity pair (forecast(H) and predict(n_obs,
n_obs + H)): `predict(0, n_obs)` (in sample, with the NaN prefix at d + s*D), `predict(n_obs - 16,
n_obs + 16)`, `params_`, `sigma2_` and the lane's `ar_`, `ma_`, `sar_` and `mu_`. The in-sample and
straddling predictions are Apple-recorded only; no committed NVIDIA or AMD column hashes them.

UMAP transform depends on the query batch by contract (umap/transform.mojo). Every UMAP cell
here is the 64-row batch identity_break asks; the claim is the GPU's bytes for that same batch.

Wheel: the nine shipped host bindings total 3,821,224 bytes on macOS arm64; the new
`_mojolearn_forecast_host.so` is 268,696 of them and has no fit symbols (no L-BFGS, estimate_x0
or least-squares symbol), and the test wheel was 1,593,235 bytes compressed. The reference
`_mojolearn_arima_host.so` (360,840 bytes, the fit) is not in the wheel.

Owed to the release record: NVIDIA and AMD model cells for the four new save formats, and NVIDIA
and AMD recordings of the in-sample and straddling ARIMA predictions. Owed to the workflow's
owner: building the forecast family in the CPU identity gate and checking `FORECAST_RECORDED`.
