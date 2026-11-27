# Mojolearn

[![PyPI](https://img.shields.io/pypi/v/mojolearn.svg)](https://pypi.org/project/mojolearn/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23829035.svg)](https://doi.org/10.5281/zenodo.23829035)

**Bitwise-Identical Machine Learning Across GPUs and CPUs.**

The same machine-learning workload can produce different bits on different
GPUs, changing predictions, learned models and subsequent training updates.
Mojolearn is a machine-learning library whose default mode is **identical**.
In `identical` mode, Mojolearn returns the same bits on Apple, NVIDIA and AMD GPUs.
Its x86-64 Linux and Apple silicon CPU bindings give the same bits too. Results agree
bit for bit, not merely within a numerical tolerance.

- **Training and inference for transformers and hybrids.** Decoder language
  models, Mamba-1/2/3 state-space models and Samba attention/state-space
  hybrids train and serve under the same contract, alongside recurrent
  networks.
- **100+ machine-learning algorithms and model components across 13 families**,
  from neural networks to tree ensembles and classical models.
- **Serve models trained elsewhere.** Neural models trained in other
  frameworks can be served with identical outputs across vendors.
- **Move training between vendors.** A training run can be handed off from
  one GPU vendor to another mid-run, and the training state after every step
  is bitwise identical to an uninterrupted run's.
- **A language model trained twice, the same bits.** The
  [training experiment](https://mojolearn.org/#experiment) on mojolearn.org.
- **Full FP32 training.** Neural training runs in full FP32 floating-point
  arithmetic. Transformer inference uses a 15-bit integer profile with exact
  sums by default, and full FP32 is one setting away
  ([numeric profiles](https://docs.mojolearn.org/user_guide/numeric_modes.html#numeric-profiles)).
- **Every algorithm runs on the Mac's own GPU**, where XGBoost, LightGBM,
  CatBoost and cuML have no supported GPU path.

To the author's knowledge, Mojolearn is the first system to enable these
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

Mojolearn removes hardware from that list. A team can test a deployed model
on different hardware, replay a recorded decision during an audit, replace
serving hardware without numerical change, and resume a checkpoint on
another vendor's GPU on the same trajectory as uninterrupted training.

## How it works

Floating-point addition is not associative, and GPU vendors differ in
reduction order, fused multiply-add contraction, denormal handling, square
root, tie-breaking and elementary functions. Recompiling does not fix any of
that. Mojolearn's numerical contract fixes the operations that determine
model bits, including reduction order, rounding and elementary functions,
and leaves tiling, thread layout, memory placement and kernel scheduling
free, so each backend can be tuned for its own hardware without changing a
bit. The kernels are Mojolearn's own, written in Mojo, and replace vendor
math routines wherever their results would depend on the backend.

## Algorithms

The table shows representative methods, with related estimator variants grouped together.

| family | algorithms and components |
|---|---|
| neural blocks and optimizers | transformers, Mamba-1/2/3, Samba, MLPs, RNNs, LSTMs, GRUs, mixture of experts, Adam, AdamW, SGD, RMSprop, Adagrad, Lion, Adafactor, LAMB |
| convolutional and graph networks | convolutions, pooling, residual blocks, CNN classifiers, GCN and GraphSAGE convolutions |
| language models and tokenization | decoder language models, byte-level BPE tokenization |
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
weight storage, tokenized corpora and a built-in verifier.

## Numeric modes

`identical` is the default. `fast` is available on Apple GPUs only, as an
opt-in for most families. [SUPPORT_MATRIX.md](SUPPORT_MATRIX.md#numeric-modes) has
the table by family.

| mode | contract |
|---|---|
| `identical` | The default. In `identical` mode, Mojolearn returns the same bits on Apple, NVIDIA and AMD GPUs. |
| `fast` | Apple GPUs only: throughput, with no repeatability promise. |

The `identical` contract is across hardware within one release. Mojolearn is
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

macOS arm64 wheels include Apple Metal support. On Linux x86-64,
`pip install mojolearn` installs `mojolearn-core` with both GPU packages,
`mojolearn-nvidia` and `mojolearn-amd`, and selects the backend for the GPU it
finds. To install one vendor only:

```sh
pip install mojolearn-nvidia   # Linux: mojolearn-core + NVIDIA
pip install mojolearn-amd      # Linux: mojolearn-core + AMD
```

Both platforms also support CPU training and inference.

The NVIDIA wheel includes native `sm_89` kernels for Ada GPUs and `sm_80` PTX
for NVIDIA GPUs with compute capability 8.0 or later, including A100 and H100.
The AMD wheel includes `gfx942` kernels for MI300X and MI325X. See [NVIDIA PTX](docs/NVIDIA_PTX_IDENTITY.md).

Mojolearn never substitutes the CPU for a GPU it cannot serve.

```sh
mojolearn doctor
```

reports what the installed wheel supports on this machine.

Verification has two profiles, classic (classical training and inference,
saved-model inference) and neural (neural training, backward and optimizer
checks), and `--all` runs both:

```sh
python -m mojolearn verify             # classic lanes on the base fixture
python -m mojolearn verify --quick     # one representative classic lane per family
python -m mojolearn verify --classic   # every classic lane, all nine fixtures
python -m mojolearn verify --neural    # every neural training lane, all nine fixtures
python -m mojolearn verify --all       # both profiles in one run and one report
```

Classic checks include bundled reference hashes, saved-model inference, and,
on a GPU installation, direct GPU/CPU inference comparisons. Reports name the
selected profile and every lane it leaves out. `--all` also runs the
single-device commands, each with its own verdict: captured-program training
and partitioned execution on one device, captured graphs, generation, loaded
language models, one composed-model recipe and one saved program checked with
`verify-model`. These checks run on a CPU-only installation or use one GPU for the GPU checks.

The run ends with a short sample of lanes where the CPU and the GPU took
different paths (kernel launches and copies on one side, host thread splits on
the other) to the same bits. `python -m mojolearn verify --show-paths [LANE]`
shows what each side did for every lane, or one lane in full, from the saved
record, running nothing. `python -m mojolearn verify --tamper-test` moves one
input by one ULP and shows that verify catches it.

The endpoint checks automatically change selected model settings or state and
check that the output changes and returns to its original value on each applicable
device. Run `python -m mojolearn verify --coverage` to see which APIs the
fixtures exercise and which native faults are assigned to each lane. Release
testing rebuilds Mojo kernels with those faults and records which fixtures detect them.

Your own composed model: `verify --recipe r.json` fits it from scratch on the
CPU and the GPU and compares the results the recipe selects (fitted
parameters, `digest_`, predictions, score); `--against other.record.json`
compares with a record from another machine. A saved program carries its
report (`Program.save` writes `<file>.verify.json`), and
`python -m mojolearn verify-model <file>` checks each output against it on
another device.

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
print(model.digest_)
```

Run the example on two supported devices and compare the printed digests.
For your own data, use the same input bytes in the same row order, seed,
settings and Mojolearn release on both machines.

## Rank items within queries

Pass one contiguous group of rows per query and supply one query ID per row.
For integer relevance grades from 0 to 30, use
`GradientBoosting(loss="lambdarank")` or `GradientBoosting(loss="rank_xendcg")`.
Both objectives support symmetric, depth-wise and leaf-wise trees. Set
`grow_policy="Lossguide"` for leaf-wise growth and `cat_features` for categorical
columns. Pass `eval_set=(X_valid, relevance_valid, query_ids_valid)` and
`eval_at=k` to select the best iteration by mean held-out NDCG@k. The default
cutoff is 30. `test_loss_curve_` reports one minus this score. `predict` returns
a score per item and `loss_curve_` reports one minus training NDCG@30.

Use `loss="QuerySoftMax"` for nonnegative, real-valued relevance. Pass query IDs
and optional per-row weights. The training curve reports the weighted mean
negative log probability at CatBoost's default scale and curvature regularization.
It accepts numeric features on symmetric trees and takes its query groups from the training set.

For `QueryRMSE`, `PairLogit`, and `YetiRank`, specify categorical columns on the
estimator and supply the validation set's own query IDs when fitting.

```python
ranker = mojolearn.GradientBoosting(loss="QueryRMSE", cat_features=[0])
ranker.fit(X_train, relevance, group_id=query_ids,
           eval_set=(X_valid, valid_relevance, valid_query_ids))
```

## Build your own model

The autograd engine behind Mojolearn's neural trainers is public. `mojolearn.Sequential` builds and
trains a feed-forward model layer by layer, and `mojolearn.ops.Session` gives you the operations,
losses, metrics and data steps themselves, with gradients, so you can compose your own algorithm on
the GPU. What you build keeps the same bits on Apple, NVIDIA and AMD in `identical` mode. See
[Build your own model](https://docs.mojolearn.org/user_guide/autograd.html).

## Check it yourself

You do not have to take the claim on trust. The wheel ships the results
recorded on Apple, NVIDIA, AMD and the CPU, and a verifier that checks your
machine against them:

```sh
python -m pip install "mojolearn[verify]"
python -m mojolearn verify --all            # every lane on all nine fixtures, plus the single-device commands
python -m mojolearn verify --tamper-test    # moves one input by one ULP and shows that verify catches it
python -m mojolearn verify --show-paths     # what the CPU and the GPU each did to reach the same bits
```

Each check reports its comparison with the recorded references; a missing
reference reads OWED and an inapplicable property is named. The base install
does not need NumPy; the `verify` extra adds it. [docs/VERIFY.md](docs/VERIFY.md)
describes the verifier and [docs/VERIFY_EXTERNALLY.md](docs/VERIFY_EXTERNALLY.md)
shows how to check the claims from outside the project.

## Scope

The guarantee holds for the same code revision, numerical profile, input
bytes, hyperparameters and seed, in identical mode, on configurations the
verification record covers. Weights trained in
another framework reproduce under Mojolearn's arithmetic, not that
framework's. [SUPPORT_MATRIX.md](SUPPORT_MATRIX.md) lists the verified
devices and configurations.

## Runtime rule

Mojolearn's numerical kernels are implemented in Mojo, with GPU execution on GPU paths. The Python package provides the public API and PyPI distribution.

## Benchmark boards

These are the canonical pages. Update them in place; don't start new ones.

- [Full-dataset A/B experiment board](experiments/six_lane_integration/measurements/20261006/BOARD.md): NVIDIA/AMD IDENTICAL and Apple FAST attempts, recorded toggles, A/B times, quality metrics and decisions. Failed attempts and unrun experiments remain visible; combined measurements do not qualify individual switches.
- [Apple M3 Ultra board](bench/results/bench_board/m3ultra-0834/BOARD.md): every lane, with our FAST and IDENTICAL arms next to the opponents. The IDENTICAL cells come from the 2026-10-04 sweep (`tools/af_board_ident_update.py`); FAST and opponent cells come from the 0.8.34 board.
- [Apple FAST refresh](docs/apple-fast/BOARD_M3_FAST.md): FAST before and after for each lane against the best opponent (`tools/af_board_merge.py`).
- [Apple IDENTICAL refresh](docs/apple-fast/BOARD_M3_IDENTICAL.md): the same view for IDENTICAL (`tools/af_board_merge.py --mode identical`).
- [Board protocol](docs/BENCH_BOARD.md) and [older boards](bench/results/board-archive/README.md).

## Documentation

- **[docs.mojolearn.org](https://docs.mojolearn.org)**: install, user guide and API reference
  (source in [docs-site/](docs-site/))
- [Install picker](https://mojolearn.org/get-started) and project site: [mojolearn.org](https://mojolearn.org)
- [Support matrix](SUPPORT_MATRIX.md)
- [Verification](docs/VERIFY.md)
- [Changelog](CHANGELOG.md)
- [Roadmap](ROADMAP.md)
- [Getting started as a contributor](docs/START_HERE.md)
- [Contributing](CONTRIBUTING.md) and [governance](GOVERNANCE.md)

## Trademarks and affiliation

Mojolearn is an independent project by Andrew Hendel, licensed under
Apache-2.0. It is not affiliated with, sponsored by, or endorsed by Modular,
Inc. MAX® and Mojo® are trademarks of Modular, Inc. Binary wheels include
unmodified Modular runtime components redistributed under Modular's own
license; see [NOTICE](NOTICE).

## Citation

To cite Mojolearn, cite the paper: Andrew Hendel, *Bitwise-Identical Machine Learning
Across GPUs and CPUs*, 2026, [10.5281/zenodo.23829035](https://doi.org/10.5281/zenodo.23829035)
([CITATION.cff](CITATION.cff)). To refer to the software, use the concept DOI
[10.5281/zenodo.22068632](https://doi.org/10.5281/zenodo.22068632), which resolves to the newest release.
