# mojolearn

[![PyPI](https://img.shields.io/pypi/v/mojolearn.svg)](https://pypi.org/project/mojolearn/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22068632.svg)](https://doi.org/10.5281/zenodo.22068632)

**Bitwise-identical machine learning across Apple, NVIDIA and AMD GPUs.**

The same machine-learning workload can produce different bits on different
GPUs, changing predictions, learned models and subsequent training updates.
mojolearn is a GPU machine-learning library whose default **identical** mode
produces bitwise-identical results across verified Apple, NVIDIA and AMD GPUs
and x86-64 and Arm CPUs. Results agree bit for bit, not merely within a
numerical tolerance.

- **57 machine-learning algorithms in 12 families**, from decision trees to
  neural networks, with training and inference for every one of them.
- **Serve models trained elsewhere.** Neural models trained in other
  frameworks can be served with identical outputs across vendors.
- **Move training between vendors.** A training run can be handed off from
  one GPU vendor to another mid-run, or shared by Apple, NVIDIA and AMD GPUs
  working on one model at once, and the training state after every step is
  bitwise identical to a single GPU's.
- **Full FP32.** Neural training and inference run in full FP32
  floating-point arithmetic, without reducing the computation to integers.
- **Every algorithm runs on the Mac's own GPU**, where XGBoost, LightGBM,
  CatBoost and cuML have no supported GPU path.

To the author's knowledge, mojolearn is the first system to enable these
capabilities.

One Mojo codebase implements these algorithms across Metal, CUDA and HIP
under a shared numerical contract. Mojo compiles code for different GPUs,
but maintaining that contract and implementing the algorithms is original
work.

## Why it matters

A model may serve predictions on NVIDIA GPUs, run locally on an Apple laptop
and later be evaluated on an AMD server. With identical weights and inputs,
different GPUs normally return different output bits, so hardware becomes
another variable in serving, regression testing and auditing. In a tree
learner one rounding difference can change the winning split and every node
beneath it. In a training loop it perturbs a gradient, then the optimizer
state, then every step after that.

mojolearn removes hardware from that list. A team can test a deployed model
on different hardware, replay a recorded decision during an audit, replace
serving hardware without numerical change, and resume a checkpoint on
another vendor's GPU on the same trajectory as uninterrupted training.

## How it works

Floating-point addition is not associative, and GPU vendors differ in
reduction order, fused multiply-add contraction, denormal handling, square
root, tie-breaking and elementary functions. Recompiling does not fix any of
that. mojolearn's numerical contract fixes the operations that determine
model bits, including reduction order, rounding and elementary functions,
and leaves tiling, thread layout, memory placement and kernel scheduling
free, so each backend can be tuned for its own hardware without changing a
bit. The kernels are mojolearn's own, written in Mojo, and replace vendor
math routines wherever their results would depend on the backend.

## Algorithms

| family | algorithms |
|---|---|
| gradient boosting | symmetric-tree, depth-wise and loss-guided boosting, ordered boosting |
| forests | random forest, extremely randomized trees, isolation forest |
| clustering | k-means, DBSCAN, HDBSCAN, agglomerative, spectral, Gaussian mixture |
| neighbors, density and search | nearest neighbors, k-NN prediction, radius neighbors, random ball cover, IVF-Flat, kernel density |
| linear models | least squares, ridge, lasso, elastic net, logistic regression |
| kernel methods and Gaussian processes | support vector machines, kernel ridge, Gaussian process regression and classification, Nystroem, random Fourier features |
| decomposition and manifold | PCA, truncated SVD, UMAP |
| time series | ARIMA, Holt-Winters exponential smoothing, KPSS test |
| preprocessing and resampling | standard and min-max scalers, bootstrap, permutation test, Monte Carlo integration |
| neural blocks and optimizers | transformer, Mamba-1, Mamba-2, Mamba-3, Samba hybrid stack, MLP, Adam, AdamW, SGD with momentum |
| language model and tokenizer | decoder language model, byte-level BPE tokenizer |
| linear algebra | matrix product, Cholesky, QR, symmetric eigendecomposition, singular values |

Also provided are evaluation metrics, cross-validation, CPU training and
inference, Hugging Face checkpoint loading and decoding, BF16 and INT8
weight storage, tokenized corpora, multi-GPU execution plans and a built-in
verifier.

## Numeric modes

`identical` is the default and, outside the tree learners, the only mode.
Gradient boosting, random forests and Extra Trees also offer two opt-in
modes.

| mode | contract |
|---|---|
| `identical` | The same bits across Apple, NVIDIA and AMD GPUs and CPUs. |
| `deterministic` | The same bits on repeated runs on one device. |
| `fast` | Throughput only, with no repeatability promise. |

Select a mode per estimator with `numeric_mode=`, per process with
`mojolearn.set_numeric_mode(...)`, or before import with
`MOJOLEARN_NUMERIC_MODE`. A configuration that cannot meet its mode's
contract raises a named error rather than silently returning something
weaker.

## Install

```sh
pip install mojolearn
```

Version <!--fact:published_version-->0.8.17<!--/fact--> is on PyPI as a
macOS arm64 wheel (Apple Metal) and a Linux x86-64 wheel carrying NVIDIA
CUDA and AMD HIP together. Both wheels train and predict on the CPU as well.

```sh
mojolearn doctor
```

reports what the installed wheel supports on this machine.

## Quick start

```python
import numpy as np
import mojolearn

rng = np.random.default_rng(0)
X = rng.random((100_000, 20), dtype=np.float32)
y = (X[:, 0] + X[:, 1] > 1.0).astype(np.float32)

model = mojolearn.GradientBoosting(loss="Logloss", n_estimators=200, max_depth=6)
model.fit(X, y)
print(model.predict_proba(X[:5]))
print(model.numeric_mode_used(), mojolearn.vendor())
```

Run the same script on an Apple, NVIDIA or AMD GPU and the model and its
predictions are the same bits.

## Verify it yourself

The wheel ships the reference results recorded on Apple, NVIDIA, AMD and
CPU, and a verifier that checks your machine against them.

```sh
python -m mojolearn verify --quick   # one lane per family
python -m mojolearn verify --all     # every lane
```

Each part reads IDENTICAL or DIVERGENT against the recorded references.
[docs/VERIFY.md](docs/VERIFY.md) describes the verifier and
[docs/VERIFY_EXTERNALLY.md](docs/VERIFY_EXTERNALLY.md) shows how to check
the claims from outside the project.

## Scope

The guarantee holds for the same code revision, numerical profile, input
bytes, hyperparameters and seed, in identical mode, on configurations the
verification record covers. It is an FP32 guarantee. Weights trained in
another framework reproduce under mojolearn's arithmetic, not that
framework's. [SUPPORT_MATRIX.md](SUPPORT_MATRIX.md) lists the verified
devices and configurations.

## Documentation

- [Support matrix](SUPPORT_MATRIX.md)
- [Verification](docs/VERIFY.md)
- [Changelog](CHANGELOG.md)
- [Roadmap](ROADMAP.md)
- [Getting started as a contributor](docs/START_HERE.md)
- [Contributing](CONTRIBUTING.md) and [governance](GOVERNANCE.md)

## Trademarks and affiliation

mojolearn is an independent project by Andrew Hendel, licensed under
Apache-2.0. It is not affiliated with, sponsored by, or endorsed by Modular,
Inc. MAX® and Mojo® are trademarks of Modular, Inc. Binary wheels include
unmodified Modular runtime components redistributed under Modular's own
license; see [NOTICE](NOTICE).

## Citation

To cite mojolearn, use [CITATION.cff](CITATION.cff). The concept DOI is
[10.5281/zenodo.22068632](https://doi.org/10.5281/zenodo.22068632).
