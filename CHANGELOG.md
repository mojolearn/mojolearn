# Changelog

## 0.1.0 (2026-08-23)

First public release. One wheel, `pip install mojolearn`, macOS arm64,
Python 3.10 through 3.14, GPU only.

### Estimators

- `GradientBoosting` (CatBoost GPU oblivious trees; 12 losses plus
  MultiClass; depthwise and lossguide growth; CTRs; eval sets; overfitting
  detector; save and load)
- `RandomForestClassifier`, `RandomForestRegressor` (cuML design)
- `ExtraTreesClassifier`, `ExtraTreesRegressor` (cuML design)
- `KMeans`, `NearestNeighbors` (cuVS, RAFT, FAISS designs)
- `DBSCAN`, `PCA`, `TruncatedSVD`, `LinearRegression` (cuML and RAFT designs)

### Numeric modes

- `FAST` (default) and `IDENTICAL` ship in the same wheel as two binary sets;
  `MOJOLEARN_NUMERIC_MODE=identical` at import selects the identical set and
  `mojolearn.numeric_mode()` reads the loaded mode back from the binary.
- Cross-vendor certificates at this release's source: E2 sub-feature matrix
  99 cells, 0 divergent, Apple M4 (Metal) against NVIDIA H100 (CUDA) and
  against AMD MI325X (HIP); see `E2_RESULTS.md`, `E1_RESULTS.md` and the
  ledger `IDENTITY_PATHS.md`.

### Packaging

- The release build compiles all five extensions in both numeric modes from
  a clean checkout, stages the MAX runtime dylibs once for all ten, and gates
  the finished wheel by installing it into a clean venv under every claimed
  interpreter and fitting every estimator family in both modes.
- Built extensions are no longer tracked in git.
- One release workflow, `release-provenance.yml`, on an ephemeral self-hosted
  runner started by `tools/release_runner.sh`; the runbook is
  `docs/PYPI_RELEASE.md`.
- `.zenodo.json` and `CITATION.cff` carry the author ORCID; every GitHub
  release is archived on Zenodo with a DOI.

### Known limits

- No CPU path. No Linux or NVIDIA wheel yet (source builds only).
- Performance measured on one M4; see the README benchmark section and its
  stated accuracy gap on higgs GBDT.
- `MultiClassOneVsAll` is not on the Python surface.
