# mojolearn

[![PyPI](https://img.shields.io/pypi/v/mojolearn.svg)](https://pypi.org/project/mojolearn/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22068632.svg)](https://doi.org/10.5281/zenodo.22068632)

**Machine learning that trains and predicts bitwise identically across Apple,
NVIDIA and AMD GPUs.**

Give mojolearn the same code, data, hyperparameters and seed on an Apple M4,
an NVIDIA H100 and an AMD MI325X, and you get the same bits on all three. Not
close, not within a tolerance. The same bits. A model trained on AMD and the
same model trained on NVIDIA are byte for byte the same model, and either one
makes exactly the same predictions. This is `identical` mode, and **it is the
default**. The claim is proven by stage-level identity cards and separating
sabotage tests, never inferred from a final-output hash, and it holds only for
the configurations recorded in [the support matrix](SUPPORT_MATRIX.md).

RF/ET offer `inference_engine="sequential"` (existing host prediction) and
experimental `inference_engine="parallel_groves"` (shared GPU prediction).
Both retain GPU training; see the [inference algorithms and numerical contract](docs/FOREST_INFERENCE_ENGINES.md).

The unreleased 0.8.0 source removes the NumPy runtime dependency and returns
`mojolearn.Array` objects. Existing NumPy inputs remain supported; callers can
use `numpy.asarray(result)` for a zero-copy view. See the
[NumPy-free contract](python/mojolearn/NUMPY_FREE_CONTRACT.md) and
[qualification roadmap](docs/lanes/NUMPY_FREE_RESIDUAL_2026-09-10.md).
The published version below retains its existing API.

Random forest and ExtraTrees fit now export model bytes directly into owned
Array buffers, avoiding per-node Python objects; see the
[forest ownership contract](python/mojolearn/NUMPY_FREE_CONTRACT.md#forest-fit-ownership-deviation-2482).

## Training performance priority

Optimize GPU training for **large real datasets**. Training-speed claims and
performance-driven default changes require measurements on representative large
workloads, such as HIGGS with 1 million rows, with held-out quality and memory
pressure recorded. Size is not a universal row cutoff: feature count, classes,
bins, tree depth and device memory also determine the workload. Small synthetic
fixtures remain useful for correctness, smoke tests and isolated diagnostics;
they do not establish a large-data speed gain or justify a speed default.
See the [tree roadmap](docs/lanes/DECISION_TREE_ROADMAP.md) and
[GPU measurement plan](docs/lanes/TREE_GPU_MEASUREMENT_NEXT.md).

## What bitwise identity means, and why it is not the default

Floating-point addition is not associative, so the order in which a GPU sums
numbers changes the answer. Vendors choose that order differently, and they
differ again in FMA contraction, denormal handling, tie-breaking, and how
`exp`, `log` and the other elementary functions are spelled. Two GPUs given
the same job return two slightly different answers, and the difference does
not stay small. One rounding can flip a tree learner's winning split and
every node beneath it. It can redraw UMAP's neighbor graph and the embedding
built from it. Inside a training loop it perturbs a gradient, then the
optimizer state, then every step after that, so two machines running the same
job walk away with two different models.

Mojo and MAX compile one source for Metal, CUDA and HIP, which is what makes
the code portable. Portability is inherited. Identity is not, and none of the
above is fixed by recompiling. mojolearn supplies the part that does not come
for free.

- An inventory, at the algorithm level, of every operation that can move
  model bits.
- A frozen numerical profile covering reduction order, partitioning, FMA
  policy, flush-to-zero seams, transcendental spellings and tie rules.
- Portable replacements for order-dependent reductions and for closed vendor
  libraries whose internals cannot be pinned.
- A per-estimator choice of `fast`, same-device `deterministic`, or
  cross-device `identical`, with an explicit refusal when the promise cannot
  be met.
- Optional stage hashing and three-vendor certificates that test the promise
  instead of asserting it.

The kernels are mojolearn's own, in Mojo. Identical mode does not delegate to
PyTorch, to MAX's matrix-multiplication kernels, or to vendor BLAS and solver
libraries, because owning the arithmetic and the reduction order is the whole
mechanism.

## What it is for

Exact model bytes make a computation auditable. Replay a certified workload on
different supported hardware, compare the recorded stage traces, and you can
say where two runs first diverged with no tolerance to argue about. That is
the basis for audits, regression tests and model change control, and it
matters most in finance, healthcare, legal services and government, where a
review can require a computation to be reproduced and its changes accounted
for.

It also lets a job move. Train on rented NVIDIA capacity, continue on AMD from
the checkpoint, and the run stays on the same trajectory rather than a nearby
one. Hardware stops being a confounding variable in a mixed fleet.

There is a second reason to be here, independent of the contract. CatBoost,
XGBoost, LightGBM and cuML have no Metal backend, so GPU tree training and GPU
classical learning have not run on Apple silicon at all. One Mojo source
builds for Metal, CUDA and HIP, which puts them on the laptop as well as the
datacenter.

The reference has to be created and replayed under the same numerical profile.
Identity does not certify a run performed in `fast` mode, in another
framework, or on a device that has not passed the same checks.

## The evidence behind the claim

- **Neural inference and training.** Mamba and transformer forward
  computations agree bit for bit across the three vendors on their recorded
  fixtures, as do gradients, optimizer updates and checkpoint bytes in
  fixed-shape transformer training. A two-block, 34,944-parameter byte-level
  language model trained on real text ran 128 steps with byte-identical
  parameters, gradients, optimizer state and loss on Apple Metal, NVIDIA CUDA
  and AMD HIP; held-out loss fell from 5.5413 to 2.8436 on all three
  ([three-vendor record](bench/results/resume/2026-09-07-root-byte-lm-three-vendor/README.md)).
  Checkpoint continuation between NVIDIA and AMD, in both directions,
  preserves the uninterrupted training trajectory
  ([cross-vendor record](bench/results/resume/2026-09-06-root-byte-lm-cross-vendor/README.md)).
  Metal checkpoint resume remains open.
- **Trees and classical learning.** Gradient boosting, random forests, Extra
  Trees, k-means, DBSCAN, k-NN, PCA, truncated SVD, OLS, ridge, logistic
  regression, FP32 matrix multiplication, isolation forest and ARIMA filtering
  carry three-vendor cards for recorded configurations. Model state and
  recorded training stages match, not only predictions.
- **UMAP.** Neighbor selection and iterative updates match across the three
  vendors on named fixtures.

Two other modes sit beside `identical`, selectable at runtime on the three
tree estimators only (see "Which families offer which tiers" below):

| mode | contract |
|---|---|
| `fast` | Optimize for throughput; repeated fits need not return identical bits. |
| `deterministic` | The same build, input, and device return the same bits on repeated runs. It makes no cross-vendor promise. |
| `identical` | Certified configurations return the same bits across Metal, CUDA, and HIP. |

Bitwise identity carries implementation and execution costs. Measured against
cuML, cuBLAS and PyTorch, `identical` mode is competitive on some measured
tree workloads and substantially slower on many classical, matrix and neural
workloads. Those measurements reflect both the numerical constraints and
optimization gaps in the current kernels. The numbers are in the accompanying
paper; the raw records behind them live under `bench/results/`.

**`identical` is the default**, in the published <!--fact:published_version-->0.8.1<!--/fact--> wheels and in this
source. For the tree estimators you opt out of it, not into it, by
setting `MOJOLEARN_NUMERIC_MODE=fast` or `deterministic` in the environment
before import, or by calling `mojolearn.set_numeric_mode(...)` in code.

### Which families offer which tiers

One rule: **the tree lanes ship three tiers, everything else ships `identical`
only** (DEVIATION 2490, 0.8.0).

| family | bindings | tiers |
|---|---|---|
| Trees: gradient boosting, random forest, extra trees | `gbdt`, `rf`, `trees` | `fast`, `deterministic`, `identical` |
| Everything else: k-means, k-NN, PCA, truncated SVD, linear models, SVC, SVR, isolation forest, kernel density, clustering, UMAP, GP, ARIMA, preprocessing, and the whole neural surface | all others | `identical` only |

Asking an `identical`-only family for a lower tier raises a named error rather
than resolving to something weaker.

Cross-vendor bitwise identity is the product, and it is the default. A `fast`
tier only earns its place where it has a measured win over the opponent's own
CPU, and that is trees on Apple silicon: tree fitting calls no BLAS, so the
opponent gets nothing from Accelerate's AMX coprocessor, and extra trees
measured 1.25-1.61x scikit-learn on all ten cores at covtype 581k. The
classical families have a BLAS call in the inner loop, and on an M4 Accelerate
reaches 1438 GFLOP/s of fp32 GEMM on four performance cores against roughly
4000 for the ten-core GPU, with one CPU thread already taking 88 of the 120
GB/s the two share. A `fast` kernel there wins about 2.5x at best over a CPU
scikit-learn gets for free, for the price of the reproducibility guarantee.
`SVC` and `SVR` could beat libsvm's single thread, but two families with a
fast tier that are not "trees" is a rule you would have to look up, and one
rule beats two wins. The neural lanes gate every fused kernel on the identical
contract, so their lower tiers were slower than the default anyway.

What every other family offers instead is the part no vendor sells: cuML is
CUDA and Linux only and does not run on Apple silicon at all, and
cross-vendor bitwise identity is available nowhere else.

## Who this is for

- People who need a reproducibility contract, same bits on repeated runs or
  across vendors, and will pay for it in time. The cost is small on some
  measured tree workloads and large elsewhere; see the paper before deciding.
- People on Apple silicon who want GPU gradient boosting, random forests,
  Extra Trees, clustering, nearest neighbors, decompositions and linear
  models without leaving the machine.
- Not yet people training real neural networks. The certified trainers are
  fixed small shapes, an MLP and the two-block byte LM above. Larger models,
  other shapes and other optimizers are outside the evidence, and the byte-LM
  native trainer is not in any published wheel.

## Install

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install mojolearn
```

Version **<!--fact:published_version-->0.8.1<!--/fact--> is published on PyPI** as an alpha API release, a macOS arm64
wheel and one Linux x86-64 wheel that now carries **CUDA sm_89, CUDA sm_90 and
HIP gfx942** together, the tree bindings in all three numeric modes and every other binding in identical only, plus the
identical-mode byte-LM trainer extension per architecture. NVIDIA Linux is no
longer source-build-only. Installed per-architecture qualification was not run
for the Linux wheel. The wheels expose public `linalg`, `umap`, `training`,
Mamba and Transformer APIs, including UMAP transform and CSR support. Newer
Python API exposure does not inherit every numerical certificate. See
[CHANGELOG.md](CHANGELOG.md) and the
[support matrix](SUPPORT_MATRIX.md) for exact artifacts and limits.
There is no CPU fallback.
Run the diagnostic command before depending on a new machine:

```sh
mojolearn doctor
```

The exact wheel, architecture, Python, and evidence boundaries live in
[SUPPORT_MATRIX.md](SUPPORT_MATRIX.md). Source builds may support hardware
outside the architectures packaged in a released wheel; that is not the same
as released-wheel support.

## Project status

### Stability and release cadence

mojolearn went from 0.1.0 on 2026-08-23 to <!--fact:published_version-->0.8.1<!--/fact--> on <!--fact:published_date-->2026-09-11<!--/fact-->, eight PyPI
releases in under three weeks (0.1.0, 0.2.0, 0.3.0, 0.3.1, 0.5.0, 0.6.0,
0.7.0, 0.8.0; 0.3.2, 0.4.0 and 0.6.1 are recorded in [CHANGELOG.md](CHANGELOG.md) but
were not published to PyPI). One release was yanked. 0.3.0, published
2026-08-30 as the first release with a Linux wheel, had been compiled for the
build machine's CPU and
carried unconditional AVX-512 instructions in its host code, so every numeric
mode died with SIGILL on any x86-64 host without AVX-512. It is yanked on PyPI
with the reason "SIGILL on x86-64 without AVX-512; use 0.3.1". 0.3.1 pinned
the Linux baseline to x86-64-v3 and added a gate on the shipped binary; the
defect and both gates are documented in
`packaging/linux/isa_baseline_linux.py` and `packaging/wheel_ci.py`.

The Python API is beta and will change between minor versions. The stable
surface is the set of numerical profiles (`fast`, `deterministic`,
`identical`) and the certified configurations recorded in
[SUPPORT_MATRIX.md](SUPPORT_MATRIX.md): a profile version changes only
through an explicit decision, and a numerical change must either prove itself
bit-inert or introduce a new profile version. For production or archival work
pin both the package version and the numeric profile, in code or through
`MOJOLEARN_NUMERIC_MODE`. A certificate names a commit, a configuration (the
fixture, the numeric profile, the parameters) and the devices it ran on, and
never more. A newer version, a different shape or an unrun vendor column is
not covered by it.

### Maintenance and bus factor

The project has one maintainer today. Three things limit what that means for
a reader.

Every claim in this repository is backed by a recorded artifact under
`bench/results/` that names its commit, device, toolchain, mode and
limitations, and each is reproducible from the commands in the docs
([verification](docs/VERIFY.md), [conformance bundles](docs/CONFORMANCE.md),
[release runbook](docs/PYPI_RELEASE.md)). Historical cards and investigations
under `bench/results/` and `archive/` are evidence, not current guidance;
[SUPPORT_MATRIX.md](SUPPORT_MATRIX.md) is updated only from recorded evidence.

Contributions are governed by [CONTRIBUTING.md](CONTRIBUTING.md) and
[GOVERNANCE.md](GOVERNANCE.md). A contributor needs one GPU of any vendor and
marks the vendor columns they did not run `cross-vendor-pending`; closing a
cross-vendor claim is a maintainer job. Any change that can move `identical`
bits must show that it is bit-inert, supply a separating fixture and a
profile-version decision, or add a named refusal. External pull requests get
an admission report and a hosted CPU report; there is no GPU automation and
no automatic merge. Governance uses lazy consensus with a seven-day objection
window, maintainership is explicitly transferable, a sole maintainer records
nominations in a public issue, and the succession steps for a sole maintainer
(nominate two successors, transfer access, document release and certification
steps, rotate credentials, publish open blockers) are written down. The code
is Apache-2.0.

You can verify a certificate without trusting the maintainer. On any
supported GPU, `MOJOLEARN_NUMERIC_MODE=identical python -m mojolearn verify`
runs a pinned fixture, captures its stage-level identity card and compares it
with the reference card shipped in the installation; `python -m mojolearn
check-fixture` checks the fixture's input hashes without a GPU. Recorded
cards carry stage tags, dtypes, element counts and raw-bit hashes and are
compared with `tools/identity_trace_diff.py`, the one comparator the
repository uses. `python -m mojolearn conformance` exports and validates
bundles so another implementation can compare itself without running Mojo,
and `tools/verify_umap_qualification.py` rechecks retained release evidence
against a wheel without GPU work. One local run establishes one build on one
device; a cross-vendor claim needs every named leg, and the cards for each
leg are in the tree.

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
- UMAP embeddings with dense Euclidean input and 2D/3D spectral initialization;
  version 0.6.0 adds unseen-sample transformation and CSR graph storage

Additional modules provide scoring metrics, FP32 matrix multiplication,
optimizer/training primitives, and reference-pinned Mamba and transformer
blocks. These surfaces do not all have the same validation depth; consult the
support matrix before treating an experimental surface as release-qualified.

UMAP in the 0.5.0 API supports fitting and embedding the supplied samples:

```python
X = np.array([0, 1, 2.2, 4, 6.5, 10, 14.5, 20], dtype=np.float32)[:, None]
embedding = mojolearn.UMAP(
    n_neighbors=3, n_components=2, n_epochs=4, random_state=19,
    numeric_mode="identical",
).fit_transform(X)
```

The 0.5.0 implementation stores a dense graph and does not support
`transform`. In **0.6.0**, public fitting stores
the graph in CSR form, using O(n_samples × n_neighbors) graph space, and
`transform(X_new)` embeds unseen samples against a frozen fitted model. Input
remains a dense Euclidean array; CSR describes internal graph storage.
Exact neighbor search still performs quadratic pair comparisons.

Source checks for the integrated fit/transform API passed all three numeric
modes on Apple, NVIDIA and AMD. The named IDENTICAL held-out embeddings
match across all three vendors. The macOS 0.6.0 candidate also passed clean
installed fit/transform and quality checks. See the [version-specific evidence](SUPPORT_MATRIX.md#umap-060-release-candidate).

Transformation retains private training data and embedding copies. Changing
parameters or numeric mode requires refitting, and changing query batching
can change results. Supervised targets, alternate metrics and alternate
initialization remain unsupported.

The APIs intentionally resemble scikit-learn, but mojolearn is not a drop-in
replacement. Where an algorithm has a settled convention for a default, that
convention is followed. Unsupported parameters raise explicitly rather than
being silently ignored.

## The exact scope of the claim

Fix a source commit, a supported configuration, a seed and byte-identical
input. On any two certified machines, every recorded training stage has the
same bits, and either model produces exactly the same predictions. This is a
claim about the trained model, not byte-for-byte equality of archive
metadata. If a configuration cannot meet the contract, the library raises a
named error instead of silently returning a possibly different model; a
refusal is reported as a refusal, never counted as a pass.

Cross-vendor identity is a profile, not a statement that every GPU operation
is universally identical. A profile fixes relevant reduction order,
partitioning, FMA policy, flush-to-zero seams, transcendental spellings, and
tie rules. A numerical change must either prove bit-inertness against the
profile or introduce a new profile version. Additional devices must pass the
same identity checks; the guarantee covers only devices and configurations
that have.

The project distinguishes four artifact classes:

```text
source check -> Python binding -> built native artifact -> installed wheel
```

Evidence for one class does not automatically validate the next. Current
certificates, configurations, and outstanding vendor legs are listed in
[SUPPORT_MATRIX.md](SUPPORT_MATRIX.md). Historical cards and investigations
under `bench/results/` and `archive/` are evidence, not current guidance.

## Limitations

What will get in your way first:

- GPU hardware is required. There is no CPU fallback, and the library refuses
  rather than silently running elsewhere.
- mojolearn is not a drop-in replacement for scikit-learn, CatBoost or cuML.
  Parameter coverage is intentionally smaller than any of them, and
  unsupported parameters raise.
- Source builds need the Mojo toolchain through [pixi](https://pixi.sh), and
  one build targets one GPU architecture. NVIDIA Linux is source-build-only
  today.
- The support matrix is honest about gaps. Several public surfaces still have
  vendor legs or independent-reference checks pending, and an unrun column is
  pending, never inferred.

And the standing limits of the contract itself:

- Released-wheel support is narrower than source-build support.
- `fast` deliberately makes no repeatability promise, and is built only for
  the three tree families (DEVIATION 2490).
- `deterministic` does not promise agreement between different devices.
- `identical` covers certified profiles and fixtures, not arbitrary untested shapes or future toolchains.
- Some recent Python and neural-operator surfaces still have vendor legs or independent-reference checks pending.
- Parameter coverage is intentionally smaller than scikit-learn, CatBoost, or cuML.
- The experimental k-NN selector remains behind an explicit build flag; normal
  wheel builds retain the existing dispatch.

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
[engineering rules](ENGINEERING_RULES.md), [contributing](CONTRIBUTING.md),
[governance](GOVERNANCE.md), and [notices](NOTICE).

## Citation

Every line of Mojo in this repository was written for it. The library
implements published machine-learning algorithms, and where a specific
published formulation is followed closely enough that a reader would want the
reference, the source file names it. The numerical contract that is the
project's distinguishing result has no counterpart anywhere.

To cite mojolearn, use [CITATION.cff](CITATION.cff). The concept DOI is
[10.5281/zenodo.22068632](https://doi.org/10.5281/zenodo.22068632).
