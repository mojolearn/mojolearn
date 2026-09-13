# Forest host inference fixtures

Read by `.github/workflows/forest-host-gate.yml` and `tools/forest_host_gate.py`
(the forest host lane, 2026-09-13, docs/lanes/BRIEF_forest_host_inference_2026-09-13.md).

One directory per recording. Each holds

| file | written by | holds |
| --- | --- | --- |
| `model.npz` | `RandomForest*.save`, `ExtraTrees*.save` or `GradientBoosting.save` on the GPU box | the five forest model arrays (a sequential archive) or the GBDT model text, under 5 MB |
| `fixture.json` | the recording script | `rows`, `features`, `seed`, `generator`, `x_sha256`; the rows are regenerated from the seed on both sides |
| `expected.json` | `tools/forest_host_gate.py record` on the GPU box | `status`, the vendor and numeric mode, `model_sha256`, `x_sha256`, and the SHA-256, dtype and shape of `predict` and `predict_proba` |

`status` is `RECORDED` only when `record` wrote the file on a box whose
`mojolearn.vendor()` is a GPU API. A directory whose `expected.json` says
`OWED` has no recording yet; the gate exits 2 on it and the workflow fails.
Nothing in this directory is ever written by hand except the OWED
placeholders and this README.

To record on a new box, from a checkout with that box's GPU build:

```
python3 tools/forest_host_gate.py record bench/results/forest_host/<directory>
```

The model must already be in the directory; fit it there with
`inference_engine="sequential"` under `MOJOLEARN_NUMERIC_MODE=identical`, and
keep it small. The 2026-09-13 Apple forest recordings were made by the smoke
in the brief (200 rows, 8 features, 8 trees, depth 6); `make` reproduces that
fit for every kind, forests and GBDT alike:

```
python3 tools/forest_host_gate.py make bench/results/forest_host/<directory> --kind gbdt_depthwise
python3 tools/forest_host_gate.py record bench/results/forest_host/<directory>
```

The kinds are `rf_classifier`, `rf_regressor`, `et_classifier`,
`et_regressor`, `gbdt_symmetric`, `gbdt_depthwise`, `gbdt_lossguide` and
`gbdt_rmse` (the four GBDT lanes of `tools/identity_break.py`; 200 rows,
8 features, 8 trees, depth 4, `max_leaves=16` for Lossguide).

## The three GPU recordings (2026-09-13)

Eight kinds on each of three GPUs, fitted on the box by `make` and recorded by `record`:
`2026-09-13-apple-m4-*` (this Mac, Metal), `2026-09-13-nvidia-*` (RunPod H100, CUDA sm_90a,
`bench/results/e1g/2026-09-13_205108-nvidia-h100-forest-host-record`) and `2026-09-13-amd-*`
(DigitalOcean MI325X, HIP gfx942, `bench/results/e1g/2026-09-13_205113-amd-mi325x-do-forest-host-record`).
For every kind the saved `model.npz` bytes and every prediction digest are the same on all three
GPUs, and this Mac's CPU-only host binding reproduces all 24 (`gate verdict IDENTICAL (24 fixtures,
exit 0)`). The NVIDIA leg's directories came home named `gpu-*` because the RunPod runner passes no
environment to the body; they were renamed by vendor here, and each `expected.json` records
`vendor` itself.
