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

- **Training and inference for transformers and hybrids.** Decoder language
  models, Mamba-1/2/3 state-space models and Samba attention/state-space
  hybrids train and serve under the same contract, alongside CNNs, recurrent
  and graph networks.
- **100+ machine-learning algorithms and model components across 13 families**,
  from neural networks to tree ensembles and classical models.
- **Serve models trained elsewhere.** Neural models trained in other
  frameworks can be served with identical outputs across vendors.
- **Move training between vendors.** A training run can be handed off from
  one GPU vendor to another mid-run, or shared by Apple, NVIDIA and AMD GPUs
  working on one model at once, and the training state after every step is
  bitwise identical to a single GPU's.
- **GPT-3 Small, trained twice, the same bits.** Two 5,000-step runs of a
  162-million-parameter GPT-3 Small-shaped decoder, each on 2.62 billion
  tokens, took different hardware routes across Apple, NVIDIA and AMD GPUs,
  with checkpoints handed between vendors and one segment trained jointly on
  an NVIDIA H100 and an AMD MI325X. The runs agree at every step and write the
  same final checkpoint.
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

The table shows representative methods, with related estimator variants grouped together.

| family | algorithms and components |
|---|---|
| neural blocks and optimizers | transformers, Mamba-1/2/3, Samba, MLPs, RNNs, LSTMs, GRUs, mixture of experts, Adam, AdamW, SGD, RMSprop, Adagrad, Lion, Adafactor, LAMB |
| language models and tokenization | decoder language models, byte-level BPE tokenization |
| convolutional and graph networks | 1-D and 2-D convolutions, pooling, batch normalization, residual blocks, CNN classifiers, GCN and GraphSAGE convolutions |
| boosting | symmetric-tree, depth-wise, loss-guided and ordered boosting; AdaBoost, DART |
| trees and ensembles | decision trees, random forests, Extra Trees, isolation forests, bagging, voting, stacking |
| clustering | k-means, mini-batch and bisecting k-means, DBSCAN, HDBSCAN, agglomerative, spectral, mean shift, OPTICS, affinity propagation, Gaussian and Bayesian Gaussian mixtures |
| neighbors, density and search | nearest neighbors, k-NN, radius neighbors, kernel density, local outlier factor, label propagation, IVF-Flat, IVF-PQ, CAGRA |
| linear models | least squares, ridge, lasso, elastic net, logistic regression, SGD, Huber, Poisson, Gamma, Tweedie, Bayesian ridge, ARD, Lars, quantile and isotonic regression |
| kernel methods and Gaussian processes | SVMs, kernel ridge, Gaussian process regression and classification, sparse variational GP, Nystroem, random Fourier and chi-square features |
| decomposition and manifold | PCA, incremental and kernel PCA, truncated SVD, NMF, ICA, factor analysis, PLS, CCA, dictionary learning, sparse PCA, LDA, UMAP, t-SNE, Isomap, MDS, LLE |
| time series | ARIMA, AutoARIMA, Holt-Winters, ETS, STL, VAR, Theta, Croston, GARCH, Prophet-style forecasting, KPSS |
| preprocessing, probabilistic models and resampling | scalers, encoders, imputers, discretization, feature selection, naive Bayes, discriminant analysis, bootstrap, permutation tests, Monte Carlo integration |
| linear algebra | matrix products, Cholesky, QR, LU, eigendecomposition, SVD, least squares |

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

The `identical` contract is across hardware within one release. mojolearn is
alpha: a release may change `identical` output bits (a faster or more accurate
route), provided the new bits agree on every supported vendor and the CPU.
Compare runs made with the same release.

Select a mode per estimator with `numeric_mode=`, per process with
`mojolearn.set_numeric_mode(...)`, or before import with
`MOJOLEARN_NUMERIC_MODE`. A configuration that cannot meet its mode's
contract raises a named error rather than silently returning something
weaker.

## Install

```sh
pip install mojolearn   # Mac: Apple GPU; Linux: NVIDIA or AMD GPU, or CPU
```

macOS arm64 wheels include Apple Metal support. In 0.8.25 on Linux x86-64,
`pip install mojolearn` automatically installs the `mojolearn-nvidia` and
`mojolearn-amd` GPU packages and selects the backend for the GPU it finds.
Both platforms also support CPU training and inference.

On NVIDIA, native kernels are used first; this release ships them for the
sm_89 and sm_90a architectures. Any other NVIDIA GPU with compute capability
8.0 or newer uses the bundled PTX build, which the driver compiles on your
machine:

- `fast` and `deterministic` modes use it automatically.
- `identical` mode uses it only on a device and driver that the release
  admitted or that you qualified yourself. Otherwise it refuses and names the
  command to run once:

  ```sh
  python -m mojolearn verify --qualify-gpu
  ```

  The command runs every identity check through the PTX build on your GPU and
  compares each result with the reference results in the wheel. It records the
  qualification only if all of them match, and only for that exact wheel,
  device and driver. A short pinned list of results the reference cannot
  judge is left out of the comparison and named in the record. See
  [NVIDIA PTX fallback](docs/NVIDIA_PTX_IDENTITY.md).

mojolearn never substitutes the CPU for a GPU it cannot serve.

```sh
mojolearn doctor
```

reports what the installed wheel supports on this machine.

Verification combines inference and classical training in one routine scope.
Neural training is a separate opt-in:

```sh
python -m mojolearn verify                    # every applicable routine lane, base fixture
python -m mojolearn verify --quick            # representative routine checks
python -m mojolearn verify --all              # entire routine scope, all nine fixtures
python -m mojolearn verify --neural-training  # neural training/backward/optimizer checks
```

Routine checks include bundled reference hashes, saved-model inference, and,
on a GPU installation, direct GPU/CPU inference comparisons. Neural training
is excluded from default, quick and all; combine `--neural-training --all`
to exercise its complete fixture set. Reports name excluded and missing checks.

The advanced `--cross-check quick|default|all` option runs only direct GPU/CPU
comparisons. Its `default` scope retains the 24-lane diagnostic sample;
`all` has no lane cap and runs every eligible lane on every fixture using
sequential child processes. Bare `verify` uses all eligible cross-check lanes
on the base fixture, without that diagnostic sample cap.

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

In 0.8.25, install verification support with
`python -m pip install "mojolearn[verify]"`: the same library plus its optional
verification dependency. On 0.8.24, use `python -m pip install numpy`.
Optional ANN, CNN and sequence APIs that use NumPy are supported by
`mojolearn[numpy]`; the verification extra includes that same dependency.
The base install does not require NumPy. See [verification support](docs/VERIFY.md).

```sh
python -m mojolearn verify          # inference and classical training, base fixture
python -m mojolearn verify --quick  # representative checks from the same scope
python -m mojolearn verify --all    # the same scope on all nine fixtures
```

In 0.8.25, `--all` also includes applicable gradient, batch-size,
ragged-batch and sampler/replay checks. On 0.8.24, add `--batch-checks` for
those probes. `--quick` keeps its smaller scope.

Each part reports its comparison with the recorded references; missing
references read OWED and inapplicable properties are named explicitly.
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

## Runtime rule

No Python compute in the runtime, ever. The Python package is only the API surface (argument checks, routing, calls into compiled Mojo); every computed value comes from Mojo, on the GPU for GPU paths.

## Benchmark boards

These are the canonical pages. Update them in place; don't start new ones.

- [Apple M3 Ultra board](bench/results/bench_board/m3ultra-0834/BOARD.md): every lane, with our FAST and IDENTICAL arms next to the opponents. The IDENTICAL cells come from the 2026-10-04 sweep (`tools/af_board_ident_update.py`); FAST and opponent cells come from the 0.8.34 board.
- [Apple FAST refresh](docs/apple-fast/BOARD_M3_FAST.md): FAST before and after for each lane against the best opponent (`tools/af_board_merge.py`).
- [Apple IDENTICAL refresh](docs/apple-fast/BOARD_M3_IDENTICAL.md): the same view for IDENTICAL (`tools/af_board_merge.py --mode identical`).
- [Board protocol](docs/BENCH_BOARD.md) and [older boards](bench/results/board-archive/README.md).

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
