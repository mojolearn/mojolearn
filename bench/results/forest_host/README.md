# Forest host inference fixtures

Read by `.github/workflows/forest-host-gate.yml` and `tools/forest_host_gate.py`
(the forest host lane, 2026-09-13, docs/lanes/BRIEF_forest_host_inference_2026-09-13.md).

One directory per recording. Each holds

| file | written by | holds |
| --- | --- | --- |
| `model.npz` | `RandomForest*.save` or `ExtraTrees*.save` on the GPU box | the five model arrays, a sequential archive, under 5 MB |
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
keep it small. The 2026-09-13 Apple recordings were made by the smoke in the
brief (200 rows, 8 features, 8 trees, depth 6).
