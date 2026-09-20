# mojolearn

[![PyPI](https://img.shields.io/pypi/v/mojolearn.svg)](https://pypi.org/project/mojolearn/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22068632.svg)](https://doi.org/10.5281/zenodo.22068632)

**Machine learning that trains and predicts bitwise identically across Apple,
NVIDIA and AMD GPUs, for certified configurations.**

Give mojolearn the same code, data, hyperparameters and seed on two certified
machines and you get the same bits on both. Not close, not within a tolerance.
The same bits. The byte-level language model record holds on an NVIDIA RTX
4090 (sm_89), an AMD MI325X (gfx942) and an Apple M4
([record](bench/results/resume/2026-09-07-root-byte-lm-three-vendor/README.md));
the forest record of September 12 holds on an NVIDIA H100 (sm_90a), an AMD
MI300X (gfx942) and an Apple M4
([brief](docs/lanes/BRIEF_amd_confirmations_2026-09-12.md)). A model trained
on AMD and the same model trained on NVIDIA are byte for byte the same model,
and either one makes exactly the same predictions. This is `identical` mode, and **it is the
default**. The claim is proven by stage-level identity cards and separating
sabotage tests, never inferred from a final-output hash, and it holds only for
the configurations recorded in [the support matrix](SUPPORT_MATRIX.md).

**It also trains on the Mac's own GPU.** GPU tree training and GPU classical
learning have not had an Apple silicon backend. One Mojo source builds for
Metal, CUDA and HIP, so gradient boosting, random forests, Extra Trees, the
isolation forest, clustering, nearest neighbors, decompositions and linear
models fit on an Apple M-series GPU as well as on a datacenter card, and Apple
Metal is one of the three vendor columns in both records named above.
That is a capability claim about where the code runs, not a speed claim.
Separately, and only for gradient boosting, random forests and Extra Trees,
there is an optional `fast` tier meant for this machine. `fast` is a different
tier from the bitwise-identical default, and it is the opposite promise. It
offers throughput and nothing else, repeated fits on the same device need not
return the same bits, and no `fast` result is certified. No speed claim is
published for it. Why not is under
[Which families offer which tiers](#which-families-offer-which-tiers).

RF/ET offer `inference_engine="sequential"` (existing host prediction) and
`inference_engine="parallel_groves"` (shared GPU prediction). The default
`"auto"` selects parallel groves only in the `fast` tier and retains the
sequential engine in `deterministic` and `identical`; either explicit spelling
overrides it. Both retain GPU training; see the
[inference algorithms and numerical contract](docs/FOREST_INFERENCE_ENGINES.md).

Since 0.8.0 the library has no NumPy runtime dependency and returns
`mojolearn.Array` objects. Existing NumPy inputs remain supported; callers can
use `numpy.asarray(result)` for a zero-copy view. See the
[NumPy-free contract](python/mojolearn/NUMPY_FREE_CONTRACT.md) and
[qualification roadmap](docs/lanes/NUMPY_FREE_RESIDUAL_2026-09-10.md).

Random forest and ExtraTrees fit now export model bytes directly into owned
Array buffers, avoiding per-node Python objects; see the
[forest ownership contract](python/mojolearn/NUMPY_FREE_CONTRACT.md#forest-fit-ownership-deviation-2482).

## Training performance priority

Optimize GPU training for **large real datasets**. Training-speed claims and
performance-driven default changes require measurements on representative large
workloads, the two real datasets of ENGINEERING_RULES.md section 9 (NYC taxi
and Istella-S LETOR) at 1 million rows or more, with held-out quality and memory
pressure recorded. Size is not a universal row cutoff: feature count, classes,
bins, tree depth and device memory also determine the workload. Small synthetic
fixtures remain useful for correctness, smoke tests and isolated diagnostics;
they do not establish a large-data speed gain or justify a speed default.
There is no fixed minimum percentage speedup: small reproducible improvements
may ship when they generalize and justify their complexity. See the
[performance acceptance policy](docs/PERFORMANCE_ACCEPTANCE.md).
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

The second reason to be here is independent of the contract and is stated at
the top of this file. The Apple silicon backend puts GPU tree training and GPU
classical learning on the laptop as well as in the datacenter, from the same
Mojo source that builds for CUDA and HIP.

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
- **Trees and classical learning.** Three layers of evidence, each scoped to
  its commit. First, stage-level three-vendor identity cards for gradient
  boosting, random forests, Extra Trees, k-means, DBSCAN, k-NN, PCA, truncated
  SVD, OLS, ridge, logistic regression, FP32 matrix multiplication, isolation
  forest and ARIMA filtering, recorded at August 2026 commits
  ([identity-path ledger](IDENTITY_PATHS.md)); model state and recorded
  training stages match there, not only predictions. Second, a three-vendor
  prediction diff at the current default, September 12, for random forest,
  Extra Trees and isolation forest (45 of 45 cells equal on an Apple M4, an
  NVIDIA H100 and an AMD MI300X), SVC (three fit hashes match) and GBDT (36 of
  36 cells on the 0.8.2 line)
  ([AMD confirmations brief](docs/lanes/BRIEF_amd_confirmations_2026-09-12.md),
  [CHANGELOG](CHANGELOG.md)). Third, September 13, every public lane at the
  0.8.4 default: 28 estimators on nine hostile fixtures, 252 cells, identical
  on an Apple M4, an NVIDIA H100 and an AMD MI325X, plus 189 cells of
  predictions on rows the model never saw and 72 cells of saved model bytes
  for the forest and GBDT lanes, all identical across the three
  ([record](bench/results/identity_break/2026-09-13_three-columns/README.md)); the
  next day, at the 0.8.5 default, all 46 public lanes with every column filled, 414 training
  cells and 459 inference and saved-model cells identical on an Apple M4, an NVIDIA H100 and an
  AMD MI300X ([record](bench/results/identity_break/2026-09-14_46-lanes/README.md)).
  Fourth, the same day, CPUs with no GPU at all: random forests, Extra Trees
  and the four gradient boosting variants trained on each of the three GPUs
  save the same model bytes, and a CPU-only binding reproduces every
  prediction digest of all 24 recordings on seven CPUs (Intel Xeon, AMD
  EPYC, Azure Cobalt Neoverse-N2, Apple M1), with a sabotage build refused on
  each ([fixtures](bench/results/forest_host/README.md)).
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
optimization gaps in the current kernels. A paper collecting the numbers is
in preparation outside this repository; the raw records behind them live
under `bench/results/`.

**`identical` is the default**, in the published <!--fact:published_version-->0.8.8<!--/fact--> wheels and in this
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
tier is a different kind of thing, because a tier sold on speed is a claim, and
the argument for putting one on trees and nowhere else is structural rather
than a published number. Tree fitting calls no BLAS, so an opponent on an M4
gets nothing from Accelerate's AMX coprocessor. The
classical families have a BLAS call in the inner loop, and on an M4 Accelerate
reaches 1438 GFLOP/s of fp32 GEMM on four performance cores against roughly
4000 for the ten-core GPU, with one CPU thread already taking 88 of the 120
GB/s the two share. A `fast` kernel there wins about 2.5x at best over a CPU
scikit-learn gets for free, for the price of the reproducibility guarantee.
`SVC` and `SVR` could beat libsvm's single thread, but two families with a
fast tier that are not "trees" is a rule you would have to look up, and one
rule beats two wins. The neural lanes gate every fused kernel on the identical
contract, so their lower tiers were slower than the default anyway.

No tree speed ratio is published here, on Apple or anywhere else.
`bench/OPPONENT_REFERENCE.md` keeps a "Rows never to quote" list, and its
entries include our own `fast` and `deterministic` arms on every vendor and
every Apple number under `bench/results/fast_speed/mac-*`. On the two datasets
a training-speed claim requires, NYC taxi and Istella-S at a million rows, the
extra trees lane does not beat scikit-learn in either tier. The Apple tree
numbers that do exist are August 2026 runs on covtype and HIGGS, below that
row floor and on datasets since retired, and they straddle parity with
scikit-learn across four windows in one week that were never reconciled. The
Apple silicon backend is therefore offered here as a capability and nothing
more, and a qualifying Apple measurement is owed.

What every other family offers instead is cross-vendor bitwise identity:
the same bits on Apple, NVIDIA and AMD GPUs and on the CPU.

## Who this is for

- People who need a reproducibility contract, same bits on repeated runs or
  across vendors, and will pay for it in time. The cost is small on some
  measured tree workloads and large elsewhere; read the records under
  `bench/results/` before deciding.
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

Version **<!--fact:published_version-->0.8.8<!--/fact--> is published on PyPI** as a macOS arm64
wheel and one Linux x86-64 wheel carrying CUDA and HIP together. The Linux
wheel targets **CUDA sm_89, CUDA sm_90a and HIP gfx942**, with the tree bindings
in all three numeric modes, every other GPU binding in identical mode, and
all 32 CPU host bindings. Its release gates require installed-wheel checks on
all three Linux architectures, including the tree bindings' three numeric modes.
The wheels expose public `linalg`, `umap`, `training`,
Mamba and Transformer APIs, including UMAP transform and CSR support. Newer
Python API exposure does not inherit every numerical certificate. See
[CHANGELOG.md](CHANGELOG.md) and the
[support matrix](SUPPORT_MATRIX.md) for exact artifacts and limits.
There is no silent CPU fallback: a box with a supported GPU trains on it.
On a CPU-only install, `fit` trains on the CPU for every estimator that has
a CPU binding, in the same arithmetic as the GPU builds, so the model is
bit-identical to the one a GPU would have produced. Each CPU binding is held to the same
bit-identity gate against the Apple, NVIDIA and AMD columns as the GPU builds,
with a sabotage build required to fail it. Inference on a CPU from a saved
model: <!--fact:host_inference_surfaces-->random forests, Extra Trees and eight gradient boosting variants; nearest neighbors on every metric and the ball cover, k-NN classification and k-NN regression with either weighting, radius neighbors and k-means assignment and distances; linear regression, ridge, truncated SVD, logistic regression, PCA with and without whitening (either solver), kernel density on every kernel, metric and weighting, the standard and min-max scalers, lasso, elasticnet, kernel ridge, the Nystroem approximation and random Fourier features, linear SVC and SVR and quasi-Newton regression on the squared and absolute losses; UMAP transform of a saved embedding (the GPU's bytes for a row, whatever else is asked in the same batch); SVC and the isolation forest; the Gaussian mixture's scores, probabilities, labels and samples; the Gaussian process regressor's predictive mean and std, normalized targets included, and the Gaussian process classifier's labels and probabilities; HDBSCAN's approximate_predict, membership_vector and all_points_membership_vectors; Embedding lookup in a saved table; IVF-Flat search over a saved index and extending it; batched ARIMA prediction, in sample and out of sample, and forecasts, with or without exogenous regressors, and Holt-Winters forecasts and in-sample one-step predictions, additive and multiplicative<!--/fact-->
(the forests: 24 three-GPU recordings reproduced on seven CPUs,
[fixtures](bench/results/forest_host/README.md); the classical
estimators: Apple M4, NVIDIA H100 and AMD MI300X recordings of the first
five, 45 fixtures each, and Apple M4 recordings of kernel density, SVC,
whitened PCA and the three k-NN classes, 54 fixtures each, reproduced on
the CPU path, [fixtures](bench/results/classical_host/)). Training on a CPU:
<!--fact:host_training_lanes-->kernel ridge poly kernel variant, kernel ridge sigmoid kernel variant, kernel ridge laplacian kernel variant, nystroem poly kernel variant, nystroem sigmoid kernel variant, nystroem laplacian kernel variant, the row-sharded random Fourier feature transform, ordinary differencing order selection, pinned GEMM, kernel density, Holt-Winters, lasso, elasticnet, SVC, agglomerative clustering, the Extra Trees classifier, the Extra Trees regressor, the isolation forest, nearest neighbors, the k-NN classifier, the k-NN regressor, PCA, whitened PCA, truncated SVD, linear regression, ridge, DBSCAN, k-means, the metrics, spectral clustering, the standard scaler, the min-max scaler, logistic regression, the random forest classifier, the random forest regressor, k-means with a random start, k-means from given centroids, weighted k-means, the standard scaler without centering, the standard scaler without scaling, the clipped min-max scaler, spectral clustering on a precomputed affinity, spectral embedding (Laplacian eigenmaps), the Gaussian process with an RBF kernel, the Gaussian process with a Matern kernel at nu 0.5, the Gaussian process with a Matern kernel at nu 1.5, the Gaussian process with an ARD Matern kernel at nu 2.5, the Gaussian process with normalized targets, Gaussian process hyperparameter optimization, Gaussian process hyperparameter optimization with restarts, the binary Gaussian process classifier, the one-vs-rest Gaussian process classifier, the class-sharded Gaussian process classifier fit, the class-sharded Gaussian process classifier prediction, the resident Mamba-1 decode session, the resident Transformer decode session, the layer-owned causal language model, the fold-dispatched cross-validation, nearest neighbors under squared euclidean distance, the distance-weighted k-NN classifier, the distance-weighted k-NN regressor, the transposed GEMM ops, the Householder QR's R factor, both slice arms, the symmetric Jacobi eigendecomposition, ascending, the singular values, descending, brute-force DBSCAN under manhattan distance, kernel density with the tophat kernel under squared euclidean distance, kernel density with the Epanechnikov kernel under manhattan distance, kernel density with the exponential kernel under chebyshev distance, kernel density with the linear kernel under cosine distance, kernel density with the cosine kernel under minkowski distance, weighted kernel density, linear regression without an intercept, weighted linear regression, ridge without an intercept, unpenalized logistic regression without an intercept, elasticnet at the l2 end without an intercept, multiplicative Holt-Winters, the linear SVC, the polynomial SVC, the tuned isolation forest, nearest neighbors under manhattan distance, nearest neighbors under chebyshev distance, nearest neighbors under cosine distance, nearest neighbors under minkowski distance at p 3, nearest neighbors over the random ball cover, radius neighbors, radius neighbors under manhattan distance, radius neighbors under chebyshev distance, radius neighbors under minkowski distance at p 3, weighted DBSCAN, l1-penalized logistic regression, elasticnet-penalized logistic regression, multiclass logistic regression, linear SVC on the hinge loss, linear SVC on the squared hinge loss, linear SVR on the epsilon-insensitive loss, linear SVR on the squared epsilon-insensitive loss, quasi-Newton regression on the squared loss, quasi-Newton regression on the absolute loss, the KPSS stationarity test, SVR, the linear SVR, whitened PCA through the full SVD, the classification, ranking and regression metrics, the Fowlkes-Mallows index, the combined homogeneity, completeness and V-measure scores, the weighted scores of the gradient boosting classifier and regressor, the weighted scores of the random forest classifier and regressor, the small MLP, gradient boosting on symmetric trees with the Logloss loss, gradient boosting on symmetric trees with the RMSE loss, gradient boosting on depthwise trees with the Logloss loss, gradient boosting on lossguide trees with the Logloss loss, gradient boosting with the Min and Max NaN modes, the gradient boosting classifier, the gradient boosting regressor, gradient boosting with the Quantile, MAE, LogLinQuantile, MAPE, Poisson, Lq, Expectile, Tweedie, Huber and CrossEntropy losses, gradient boosting with Exact leaves and the Poisson bootstrap, gradient boosting on lossguide trees with the NewtonCosine score and the searcher options, multiclass gradient boosting, one-vs-all gradient boosting, ordered boosting with the RMSE loss (OrderedRMSE), gradient boosting with the six non-default feature border types, ordered boosting (boosting_type='Ordered') with the Logloss and RMSE losses, ordered boosting with the Bayesian bootstrap and score noise, boost from average on the MAE, Quantile and MAPE losses, gradient boosting at CatBoost's GPU defaults (auto learning rate, Bayesian bootstrap, score noise), gradient boosting with the bootstraps and the score noise on the RMSE loss and on Depthwise and Lossguide trees, gradient boosting with an eval set, the overfitting detector and best-model truncation, the two-level FeatureFreq estimator, gradient boosting with the pointwise searcher, L2 scores, the Bayesian bootstrap and an eval set, gradient boosting with one-hot categorical columns, gradient boosting with the QueryRMSE ranking loss on query groups, gradient boosting with the PairLogit ranking loss on generated and explicit pairs, gradient boosting with the YetiRank ranking loss on query groups, ARIMA, differenced ARIMA, seasonal ARIMA, ARIMA with exogenous regressors, differenced seasonal ARIMA with exogenous regressors, UMAP, k-means under the rooted euclidean metric, k-means from the classic k-means++ start, cross-validation of gradient boosting, the bootstrap, the permutation test, Monte Carlo integration, SGD with momentum, Nesterov and dampening, Adam and AdamW with the gradient clip and accumulation, the cross-entropy loss arms, the embedding, RMSNorm and linear training primitives, the ordered shard gradient reduction, the Cholesky factorization and solve, random Fourier features, kernel ridge, the Nystroem kernel approximation, the Gaussian mixture, the Gaussian mixture with a random start, HDBSCAN, HDBSCAN with leaf selection, the random forest classifier with entropy splits, log2 features and no bootstrap, the class-weighted random forest classifier with the parallel groves engine, the random forest regressor with the Poisson criterion, the random forest regressor with the gamma and inverse Gaussian criteria, the best-first Extra Trees classifier with entropy splits, the bootstrapped Extra Trees regressor with the parallel groves engine, the Mamba-2 block, the Mamba-2 block with an active dt clamp, the Mamba-1 block, the Mamba-3 block, the Transformer block, the sliding-window Transformer block, the Samba stack, the Samba stack with untied embeddings, dropout, accumulation, clipping and a cosine schedule, the byte LM forward pass on its reference path (inference), the byte LM forward pass on its threaded path (inference), the published byte LM host training step, the byte LM shape object at two non-default shapes, the column-sharded standard scaler, the column-sharded min-max scaler, series-sharded ARIMA, series-sharded Holt-Winters, the series-sharded ARIMA prediction and forecast drivers, the series-sharded Holt-Winters prediction and forecast drivers, query-sharded k-NN classification, query-sharded nearest-neighbor distances and indices, query-sharded radius neighbors, query-sharded kernel density, reference-sharded k-NN classification, reference-sharded k-NN regression, the tree-range-sharded random forest classifier, the tree-range-sharded Extra Trees regressor, the tree-range-sharded Extra Trees classifier, the tree-range-sharded random forest regressor, the small MLP trained over ordered logical gradient shards, the Samba stack trained over ordered logical gradient shards, the Samba stack trained over ordered logical gradient shards under a global norm clip, the Embedding layer, the Embedding layer on its sorted execution plan, the IVF-Flat index, the IVF-Flat index under euclidean distance, the shard-distributed IVF-Flat index, extending a built IVF-Flat index, the byte LM trainer, the byte LM trainer on its resident session, samples from the Gaussian mixture, samples from the Gaussian mixture with a random start, posterior draws from the Gaussian process, posterior draws from the Gaussian process with normalized targets, the byte-level BPE tokenizer (inference, host integers), byte-level BPE vocabulary training, a trained BPE vocabulary written, loaded back and used, a corpus tokenized once, cached and read back as batches, the Hugging Face checkpoint reader and the option matrix, the Hugging Face byte-level BPE tokenizer (three pre-tokenization patterns), a Hugging Face causal language model loaded and run, predictions of Metal-saved gradient boosting models with CTR tables (inference), predictions of Metal-saved gradient boosting models with tensor CTRs (inference), the bf16-storage GEMM profile, the int8 GEMM profile with power-of-two scales, the Transformer block with bf16-stored weights, the Transformer block with int8-stored weights, the Mamba-1 block with bf16-stored weights, the Mamba-1 block with int8-stored weights, the Mamba-2 block with bf16-stored weights, the Mamba-2 block with int8-stored weights, the Mamba-3 block with bf16-stored weights, the Mamba-3 block with int8-stored weights, the small MLP with bf16-stored weights, the small MLP with int8-stored weights, the Samba stack with bf16-stored weights, the Samba stack with int8-stored weights, saved forest and gradient boosting models predicted on the CPU (inference), the bf16 and int8 weight-storage conversions and gradient accumulation across microbatches<!--/fact-->,
the first six identical to the three GPU columns on seven CPUs
([gate](.github/workflows/cpu-identity-gate.yml)) and, since 2026-09-14,
agglomerative clustering, Extra Trees (classifier and regressor, the saved
model bytes included) and the isolation forest identical to the three GPU
columns on the Apple M4 host path with the seven-runner run of the same
gate pending
([columns](bench/results/identity_break/2026-09-14_cpu-phase1b/README.md)).
The whole CPU surface is declared once, in
[python/mojolearn/host_surface.py](python/mojolearn/host_surface.py), and the
[support matrix](SUPPORT_MATRIX.md) carries it as a table. The byte LM
has two CPU surfaces of its own. `LanguageModelInference` runs the forward
pass; see [docs/BYTE_LM_CPU_INFERENCE.md](docs/BYTE_LM_CPU_INFERENCE.md) for
the CPUs it is certified on. `LanguageModelHostTrainer` runs one training
step, forward, backward and the AdamW update, and reproduces the recorded GPU
bytes of the retained three-vendor capture for all 128 of its steps, the
gradient and the loss and the post-step parameters and both Adam moments
alike; see [docs/BYTE_LM_CPU_TRAINING.md](docs/BYTE_LM_CPU_TRAINING.md), which
also states what it does not claim. Both are one model profile at one batch
shape, and identity is claimed per shape because the weight gradients contract
over the token count. From 0.8.7 every one of these host bindings ships in both
wheels (0.8.5 and earlier carry only the byte LM's; 0.8.6 was folded into 0.8.7
and never published); each also builds from source with
`bindings/build_*_host.sh` (a shim over
`bindings/build_host_family.sh <family>`). Every lane not named here has
no CPU path at all: <!--fact:no_cpu_path-->(1) gradient boosting training with sample weights on any arm (`gbdt_fit` refuses `sample_weight`, and `class_weights` outside MultiClass and MultiClassOneVsAll, which reach the device through the same per-row weight column): the device's weighted target, histogram and partition-reduce kernels are a second launch arm (`has_weights`) and the gbdt/host oracles restate the unit-weight arm only; (2) gradient boosting training on a CTR categorical column, a `cat_features` column with more than `one_hot_max_size` categories: the CTR calcers build ordered target statistics over several permutations, with their online counters, grids and tables joined back into the compressed index, and the host path is pinned to one permutation with no calcer. One-hot categorical columns DO train, `ExperimentalTwoLevelFeatureFreq` has its own CPU route (gbdt/host/gbdt_oracle_feature_freq.mojo) except on a tree whose level winner is the tensor column itself, and CTR INFERENCE from a saved model is closed; it is the calcer tables' training that is not; (3) gradient boosting training with a categorical or one-hot column outside SymmetricTree with Logloss and Plain boosting: the one-hot grid and the `take_bin` equality split are restated in the symmetric searcher alone; (4) gradient boosting training with an eval set or the overfitting detector outside SymmetricTree with Logloss, Ordered boosting and the pointwise searcher's own lane: the held-out curve runs THAT arm's loss kernel (the multilogit and one-vs-all launches, `launch_approximate` at each pointwise objective) and the non-symmetric shapes put a tree on the cursor through a different apply, and neither is restated; (5) gradient boosting training with the pointwise searcher outside the gbdt-pointwise-l2-bayesian-eval configuration (L2 scores, the Bayesian bootstrap, Newton leaves, sample weights, an eval set with the Iter detector, boost_from_average on, GreedyLogSum borders, numeric columns): every other option selects a different launch shape of the pointwise kernels and one shape is restated; (6) gradient boosting training at a (loss, grow_policy, score_function, leaf_estimation_method, bootstrap_type) combination outside the ones the gbdt/host oracles restate, each refused by name: the pointwise losses under Depthwise and Lossguide, score functions and leaf estimators outside each policy's covered pair, any bootstrap and random_strength under MultiClass and MultiClassOneVsAll (so their DEFAULT fit refuses; pass bootstrap_type='No' and random_strength=0), the Poisson bootstrap under Depthwise and Lossguide, random_strength under the ranking losses, Depthwise's min_split_gain, min_child_hessian and min_data_in_leaf, feature_fraction outside Lossguide with Logloss, boost_from_average outside RMSE and the quantile family, and a NaN in X outside SymmetricTree with Logloss -- each one its own device kernel or its own searcher gate order. The bootstraps and the score noise DO train on Logloss, RMSE and the ten pointwise losses under SymmetricTree and on Logloss and RMSE under Depthwise and Lossguide (the gbdt-stochastic-arms lane), which is every default fit of GradientBoostingRegressor and GradientBoostingClassifier<!--/fact-->.
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

The current published release is <!--fact:published_version-->0.8.8<!--/fact-->, dated
<!--fact:published_date-->2026-09-19<!--/fact-->. [CHANGELOG.md](CHANGELOG.md)
records published releases and versions that were prepared but never published.

Version 0.3.0, published
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

The identity cards and legs cited in this README are recorded under
`bench/results/`, each naming its commit, device, toolchain, mode and
limitations. The per-release install qualification logs are retained outside
the tree and summarized per release in [CHANGELOG.md](CHANGELOG.md). Each
recorded card is reproducible from the commands in the docs
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

### Check the claims on your machine

The commands below describe the 0.8.7 verifier; earlier wheels do not contain
this full verifier. Inspect `verify --coverage` for the installed
package's 246 appendix entries, additional lanes, missing references and batch
contracts. Mappings are not certification. `verify --batch-checks` additionally
runs gradient, batch-size, ragged-batch and sampler/replay probes; see [the verification guide](docs/VERIFY.md).

```sh
python -m mojolearn verify --coverage
python -m mojolearn verify --all        # or --quick, one lane per family
```

This runs the identity lanes the committed records were written with, on
fixtures generated inside the package, and compares every train, infer,
saved-model and batch part with the reference hashes the Apple, NVIDIA, AMD
and CPU records carry, shipped in the wheel. On a CPU-only install it runs
the CPU reference lanes and loads small GPU-trained models, which must
answer with the recorded GPU bits. Each part reads IDENTICAL, DIVERGENT,
OWED (no record yet) or REFUSED; `--json` writes a report to share.
[docs/VERIFY.md](docs/VERIFY.md) says what a local run proves and what it
does not.

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
device; a cross-vendor claim needs every named leg, and the identity cards
for each leg are under `bench/results/`. The per-release install
qualification logs are kept outside the tree and summarized in
[CHANGELOG.md](CHANGELOG.md).

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
parameters or numeric mode requires refitting. Changing query batching does
NOT change results: a batch of N returns the same bytes as N calls of one
row. Supervised targets, alternate metrics and alternate initialization
remain unsupported.

The APIs follow familiar estimator conventions, but mojolearn is not a drop-in
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
refusal is reported as a refusal, never counted as a pass. The claim is
float32 only. Float64 input is converted with a copy for most estimators
(`python/mojolearn/_buffer.py`) and refused by name on the linalg
(`python/mojolearn/_linalg_impl.py`) and Mamba (`python/mojolearn/_mamba_impl.py`)
surfaces. The cross-vendor identity cells behind the September 12 diff are
recorded at fixtures of up to 20,000 rows by 16 columns
(`tools/identity_break.py`), not at the 1M-row speed workloads.

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

- GPU hardware is required to train every estimator, except the six lanes
  and the byte LM named under "There is no silent CPU fallback" above. The
  library refuses rather than silently running elsewhere. Inference without a
  GPU exists only for the forests, gradient boosting, five classical
  estimators and the byte LM, each through an explicitly named host class or
  binding that needs its own build.
- mojolearn is not a drop-in replacement for scikit-learn, CatBoost or cuML.
  Parameter coverage is intentionally smaller than any of them, and
  unsupported parameters raise.
- Source builds need the Mojo toolchain through pixi, and
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
[verification](docs/VERIFY.md), [verifying the identity claims yourself, from
outside, at three costs](docs/VERIFY_EXTERNALLY.md), [release](docs/PYPI_RELEASE.md),
[engineering rules](ENGINEERING_RULES.md), [contributing](CONTRIBUTING.md),
[governance](GOVERNANCE.md), and [notices](NOTICE).

## Trademarks and affiliation

mojolearn is an independent project by Andrew Hendel. It is not affiliated
with, sponsored by, or endorsed by Modular, Inc. MAX® and Mojo® are trademarks
of Modular, Inc. Binary wheels include unmodified Modular runtime components
redistributed under Modular's own license; see [NOTICE](NOTICE).

## Citation

Every line of Mojo in this repository was written for it. The library
implements published machine-learning algorithms, and where a specific
published formulation is followed closely enough that a reader would want the
reference, the source file names it. The numerical contract that is the
project's distinguishing result has no counterpart anywhere.

To cite mojolearn, use [CITATION.cff](CITATION.cff). The concept DOI is
[10.5281/zenodo.22068632](https://doi.org/10.5281/zenodo.22068632).

### CPU inference and verification cadence

The public CPU surface is saved-model inference: byte-LM, forests, and the
classical models listed in [the support matrix](SUPPORT_MATRIX.md). Ordinary
CPU estimator `fit` calls refuse; GPU training remains available. The
`LanguageModelHostTrainer` already published in 0.8.5 remains supported.
The broader CPU training implementations are internal numerical references,
available from source rather than added to the public CPU training API.

Routine pushes run inference checks and small reference probes. Full CPU
reference verification runs weekly, manually, and before a PyPI publication.
Release GPU certification uses one Apple, one AMD and one NVIDIA device.
`python -m mojolearn identity` on CPU defaults to seven small reference lanes;
the full source verifier retains every covered lane and fixture. An identity
result certifies only the lanes, fixtures and numerical profile it reports.
