# mojolearn

[![PyPI](https://img.shields.io/pypi/v/mojolearn.svg)](https://pypi.org/project/mojolearn/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22068632.svg)](https://doi.org/10.5281/zenodo.22068632)

**GPU machine learning in Mojo, with an explicit reproducibility contract.**

mojolearn provides Python APIs with familiar scikit-learn shapes over one
Mojo source tree targeting Apple Metal, NVIDIA CUDA, and AMD HIP. Its defining
feature is a choice of numerical contract on every supported estimator:

| mode | contract |
|---|---|
| `fast` | Optimize for throughput; repeated fits need not return identical bits. |
| `deterministic` | The same build, input, and device return the same bits on repeated runs. |
| `identical` | Certified configurations return the same bits across Metal, CUDA, and HIP. |

`IDENTICAL` is supported by stage-level identity cards and separating
sabotage tests. It is not inferred from a final-output hash. Claims apply only
to configurations recorded in [the support matrix](SUPPORT_MATRIX.md).

## Install

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install mojolearn
```

Published wheels target Apple silicon on macOS and selected NVIDIA/AMD
architectures on Linux x86-64. There is no CPU fallback. Run the diagnostic
command before depending on a new machine:

```sh
mojolearn doctor
```

The exact wheel, architecture, Python, and evidence boundaries live in
[SUPPORT_MATRIX.md](SUPPORT_MATRIX.md). Source builds may support hardware
outside the architectures packaged in a released wheel; that is not the same
as released-wheel support.

## Quick start

```python
import numpy as np
import mojolearn

rng = np.random.default_rng(0)
X = rng.random((100_000, 20), dtype=np.float32)
y = (X[:, 0] + X[:, 1] > 1.0).astype(np.float32)

model = mojolearn.GradientBoosting(
    loss="Logloss", n_estimators=200, max_depth=6,
    numeric_mode="deterministic",
)
model.fit(X, y)
print(model.predict_proba(X[:5]))
print(model.numeric_mode_used(), mojolearn.vendor())
```

Choose a process default with `mojolearn.set_numeric_mode("identical")`, or
set the starting default before import:

```sh
MOJOLEARN_NUMERIC_MODE=identical python train.py
```

More than one tier may be loaded in one process through per-estimator
`numeric_mode=` arguments.

## Public API

Classical estimators include:

- Gradient boosting, random forests, and Extra Trees
- K-means, nearest-neighbor estimators, DBSCAN, hierarchical and spectral clustering
- PCA, truncated SVD, linear and logistic regression, ridge, lasso, and elastic net
- SVC, SVR, kernel density, isolation forest, and Gaussian-process regression
- Exponential smoothing and batched ARIMA

Additional modules provide scoring metrics, FP32 matrix multiplication,
optimizer/training primitives, and reference-pinned Mamba and transformer
blocks. These surfaces do not all have the same validation depth; consult the
support matrix before treating an experimental surface as release-qualified.

The APIs intentionally resemble scikit-learn, but mojolearn is not a drop-in
replacement. Defaults follow the upstream GPU implementation mirrored by an
algorithm where applicable. Unsupported parameters raise explicitly rather
than being silently ignored.

## What the identity claim means

Cross-vendor identity is a profile, not a statement that every GPU operation
is universally identical. A profile fixes relevant reduction order,
partitioning, FMA policy, flush-to-zero seams, transcendental spellings, and
tie rules. A numerical change must either prove bit-inertness against the
profile or introduce a new profile version.

The project distinguishes four artifact classes:

```text
source check -> Python binding -> built native artifact -> installed wheel
```

Evidence for one class does not automatically validate the next. Current
certificates, configurations, and outstanding vendor legs are listed in
[SUPPORT_MATRIX.md](SUPPORT_MATRIX.md). Historical cards and investigations
under `bench/results/` and `archive/` are evidence, not current guidance.

## Limitations

- GPU hardware is required.
- Released-wheel support is narrower than source-build support.
- `fast` deliberately makes no repeatability promise.
- `deterministic` does not promise agreement between different devices.
- `identical` covers certified profiles and fixtures, not arbitrary untested shapes or future toolchains.
- Some recent Python and neural-operator surfaces still have vendor legs or independent-reference checks pending.
- Parameter coverage is intentionally smaller than scikit-learn, CatBoost, or cuML.

mojolearn is beta software. Pin the package version and numerical profile for
production or archival work.

## Development

Start with [docs/START_HERE.md](docs/START_HERE.md). The shortest full local
check is `pixi run probe`.

A numerical test counts as evidence only after a separating arm demonstrates
that it fails when the relevant rule is broken. Contributors need one
supported GPU; maintainers close cross-vendor certification columns.

Current priorities are in [ROADMAP.md](ROADMAP.md). See also
[verification](docs/VERIFY.md), [release](docs/PYPI_RELEASE.md),
[porting rules](PORTING_RULES.md), [contributing](CONTRIBUTING.md), and
[attribution](NOTICE).

## Provenance and citation

The shipped implementation is Mojo. Algorithmic designs derive in part from
CatBoost, cuML, cuVS, RAFT, and FAISS; exact provenance and licenses are in
`NOTICE`, `DERIVATION_MAP.tsv`, source headers, and the archived derivation
ledger.

To cite mojolearn, use [CITATION.cff](CITATION.cff). The concept DOI is
[10.5281/zenodo.22068632](https://doi.org/10.5281/zenodo.22068632).
