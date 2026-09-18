# Alpha Python API

The alpha surface exposes the implemented Python operations below. Availability
also requires a matching native extension for the wheel's GPU architecture and
numeric tier. A Python export alone does not qualify an installed wheel or
establish cross-vendor bitwise identity. There is no CPU fallback.

Tiers (DEVIATION 2490, 0.8.0): `_mojolearn_gbdt`, `_mojolearn_rf` and
`_mojolearn_trees` ship `fast`, `deterministic` and `identical`. Every other
extension in this table ships `identical` only; passing `numeric_mode="fast"`
or `"deterministic"` to those components, or importing under
`MOJOLEARN_NUMERIC_MODE=fast`, raises by name.

| Component | Public import | Native extension | Implemented scope |
| --- | --- | --- | --- |
| FP32 GEMM | `mojolearn.linalg.matmul` or `mojolearn.matmul` | `_mojolearn_linalg` | NN, NT and TN matrix products; optional output buffer |
| UMAP | `mojolearn.umap.UMAP` or `mojolearn.UMAP` | `_mojolearn_metrics` | Dense Euclidean fit/fit_transform, CSR graph storage, held-out transform, 2D/3D spectral initialization |
| Transformer | `mojolearn.transformer.TransformerBlock`, `TransformerState` | `_mojolearn_transformer` | Supplied-weight FP32 decoder-block forward, prefill and decode with explicit KV state |
| Mamba | `mojolearn.mamba.Mamba1Block`, `Mamba2Block`, `Mamba3Block` and their state classes | `_mojolearn_mamba` | Explicit-weight block operations and recurrent states; each class documents its supported configuration |
| Optimizers and losses | `mojolearn.training` | `_mojolearn_training` | SGD, Adam, AdamW, L2 gradient clipping, class-index cross entropy with optional logits gradient |
| CPU neural inference | `mojolearn.MLPInference`, `mojolearn.TransformerBlockInference` (`mojolearn.neural_inference`) | `_mojolearn_neural_host` (a CPU binding under `mojolearn/host/`, loaded by path, `bindings/build_neural_host.sh`) | Forward only, from weights trained on a GPU: `MLPInference.from_checkpoint(path).predict_logits(X)` for a `SmallMLPTrainer` checkpoint, and `TransformerBlockInference(weights, n_heads=...).forward(x, lengths=None)` from a zero state. `Mamba1BlockInference(weights)`, `Mamba2BlockInference(weights, dt_limit=...)` and `Mamba3BlockInference(weights).forward(x, lengths=None)` from a zero state; `SambaInference.from_checkpoint(path).forward(ids, lengths=None)` for `SambaStack.save_checkpoint` files (`SambaInference(config, weights)` too). The byte LM's is `LanguageModelInference.from_checkpoint` (`byte_lm` family). No state, step or backward; no training |
| Small MLP training | `mojolearn.neural_network.SmallMLPTrainer` | `_mojolearn_linalg`, `_mojolearn_training` | Fixed 8→16→3 FP32 ReLU, mean cross entropy and AdamW; batch 1–256 |
| Small byte-LM training | `mojolearn.language_model.SmallByteLanguageModelTrainer` | `_mojolearn_byte_lm` | Fixed two-block FP32 next-byte model, full gradients and AdamW; CUDA/HIP IDENTICAL only |
| Byte-level BPE tokenizer | `mojolearn.tokenizer.BpeTokenizer` or `mojolearn.BpeTokenizer` | `_mojolearn_tokenizer_host` (a CPU binding under `mojolearn/host/`, built from source with `bindings/build_tokenizer_host.sh`; the same binary on every box, there is no GPU path) | byte-level BPE in the GPT-2 format over a vocabulary the user supplies; mojolearn ships none. `BpeTokenizer.from_files(encoder_json, vocab_bpe)`, `from_ranks_file(path)` or `from_token_bytes(tokens)`; `BpeTokenizer()` refuses by name. `encode`/`encode_bytes` and `decode`/`decode_bytes`, ids in [0, `n_vocab`), `<|endoftext|>` = `n_vocab - 1` only with `allow_endoftext=True`; held id for id to a second Python encoder over a synthetic vocabulary mojolearn trains itself. `encode_batch` encodes a list of documents in one binding call, each document's ids those of encoding it alone; `decode_batch`/`decode_bytes_batch` loop over `decode_bytes`. No other pre-tokenizer pattern |

| Cholesky | `mojolearn.Cholesky` or `mojolearn.linalg.Cholesky` | `_mojolearn_gp` (the GP build already links `cholesky/`) | `fit(A)` factors `A + jitter I` (jitter `0.0` or the profile's pinned ridge, read from the binding), `solve(B)`, `L_`, `info_`, `nb_`, `logdet_`, `jitter_`; a failed factor is a result and its solve is refused by name. Written and compile-checked 2026-09-14 (workstream D); no run through this door on any box yet |
| Kernel methods | `mojolearn.kernel_methods.KernelRidge`, `Nystroem`, `RBFSampler` (all three also from `mojolearn`) | `_mojolearn_kernel_methods` (`bindings/build_kernel_methods.sh`) | linear, poly, rbf, sigmoid and laplacian kernels; `KernelRidge.fit/predict`, `Nystroem.fit/transform` (eigenvalues, eigenvectors and the Jacobi sweep count carried), `RBFSampler.fit/transform`; `precomputed` and `gamma='scale'` refused by name. Written and compile-checked 2026-09-14; no run through this door on any box yet |
| Gaussian mixture | `mojolearn.mixture.GaussianMixture` or `mojolearn.GaussianMixture` | `_mojolearn_mixture` (`bindings/build_mixture.sh`) | `covariance_type='full'`, `init_params` 'kmeans' or 'random', `n_init=1`; `fit`, `predict`, `predict_proba`, `score_samples`, `score`, `bic`, `aic`; the other covariance types and inits refused by name on the Mojo host, `n_init > 1`, `warm_start` and the `*_init` arrays refused by name here. Written and compile-checked 2026-09-14; no run through this door on any box yet |
| HDBSCAN | `mojolearn.hdbscan.HDBSCAN` or `mojolearn.HDBSCAN` | `_mojolearn_hdbscan` (`bindings/build_hdbscan.sh`) | brute-force k-NN, Boruvka MST, condensed tree, euclidean only, `cluster_selection_method` 'eom' or 'leaf', `cluster_selection_epsilon=0.0` only; `labels_`, `core_distances_`, `n_clusters_`, `n_outliers_`; `probabilities_` refused by name (DEVIATION 1610). Written and compile-checked 2026-09-14; no run through this door on any box yet |
| Resampling | `mojolearn.resample.bootstrap`, `permutation_test`, `monte_carlo_integrate` | `_mojolearn_resample` (`bindings/build_resample.sh`) | six statistics, percentile and basic intervals (`bca` refused by name), three alternatives, the `r_first` and `i_first` batch-invariance handles, three compiled integrands with their closed forms. Written and compile-checked 2026-09-14; no run through this door on any box yet |
| IVF-FLAT | `mojolearn.IVFIndex` | `_mojolearn_ivf` (`bindings/build_ivf.sh`) | IVF-Flat build plus search under one card: `fit(X)` records the rows, `search(queries)` returns float32 distances and int32 original row ids, `n_candidates_` per query; `n_probes` required, `n_probes > n_lists` refused on the Mojo host, metric 'sqeuclidean' (squared distances, the default) or 'euclidean' (the same squared selection with the k kept distances rooted; refused from the exposure until fix/ivf-l2sqrt on 2026-09-14, when rooted norms made its search return all-zero distances, now searched by `ivf_check.mojo::check_l2_sqrt_is_the_root_of_l2` and the `ivf-euclidean` identity lane). The index does not persist between calls. Exposed 2026-09-14 |
| Embedding | `mojolearn.Embedding` or `mojolearn.embedding.Embedding` | `_mojolearn_embedding` (`bindings/build_embedding.sh`) | profile `mojolearn.identical.embedding.fp32.v1`: `forward(ids)` gathers, `backward(ids, dy, grad=None)` returns the dense gradient by the ascending fold, `padding_idx` stores +0.0 in its row, `grad=` carries a microbatch accumulator bit for bit; `plan="sort"` selects PLAN_SORT for the backward's run structure (default `"scan"`; both give the same bits, lane `embedding-sort`, 2026-09-15); `from_pretrained`; `max_norm`, `scale_grad_by_freq`, `sparse` and a missing `weight` refused by name. Exposed 2026-09-14 |
| Training primitives | `mojolearn.training.embedding_forward`, `embedding_backward`, `rms_norm_forward`, `rms_norm_backward`, `linear_forward`, `linear_backward` | `_mojolearn_training` | the six primitives the shipped binding already exported, now named by the module (2026-09-14); float32 buffers only |
| k-means arms | `mojolearn.KMeans(metric=..., oversampling_factor=...)` | `_mojolearn` | `metric` ('euclidean' and 'l2_expanded' are the recorded default, 'l2_sqrt_expanded' takes the root; any other value is refused by name, and 'cosine' was deleted on 2026-09-18 because cuVS refuses cosine k-means too and the mean update does not minimize cosine distance) and `oversampling_factor` (`0.0` selects the classic sequential k-means++ seeding). Routed 2026-09-14; the default's bits are unchanged |

The existing classical estimators and metrics remain exported from `mojolearn`.
All optimizer classes, both small trainers, `cross_entropy` and
`clip_grad_norm_` are also available directly from `mojolearn`. Native extension
lookup remains lazy; a missing component raises when used.

## Arrays: NumPy is optional

NumPy is not a dependency of this package (0.7). Every input is read through
the buffer protocol, so a NumPy array, an `array.array`, a `mojolearn.Array`
or any other object exporting a buffer is accepted; the surfaces that certify
bits (GEMM, Mamba, Transformer, the training primitives) refuse a non-float32
buffer by name rather than casting it. Every array an operation returns is a
`mojolearn.Array`: it owns its memory, exposes `shape`, `dtype`, `tobytes()`,
`tolist()`, indexing and the buffer protocol, and exports
`__array_interface__`, so `numpy.asarray(result)` is a zero-copy view for a
caller who has NumPy. Buffers a surface updates in place (optimizer
parameters, recurrent states, `matmul(out=...)`) must be writable and
C-contiguous. The examples below use NumPy for the caller's inputs; install
it with `pip install mojolearn[test]`, which is also what the test suite and
`mojolearn verify` need.

## Numeric mode and GEMM

Select IDENTICAL before importing the package for these examples:

```sh
MOJOLEARN_NUMERIC_MODE=identical python example.py
```

```python
import numpy as np
from mojolearn.linalg import matmul, profile

a = np.arange(32, dtype=np.float32).reshape(8, 4)
b = np.arange(12, dtype=np.float32).reshape(4, 3)
c = matmul(a, b)  # (8, 3), FP32; IDENTICAL required by default
print(profile())
```

`matmul(..., identical=False)` makes no identity claim about the result but
runs the same binary: `_mojolearn_linalg` ships in the `identical` tier alone
(DEVIATION 2490, 0.8.0), so there is no FAST linalg to select and asking for
one raises by name. The product's fixed FP32 reduction
contract does not promise NumPy, cuBLAS, or PyTorch's output bits. The GEMM
contract documents its measured shape sweep; accepting other shapes does not
mean those shapes were individually measured.

## UMAP

```python
from mojolearn.umap import UMAP

model = UMAP(n_neighbors=15, n_components=2, n_epochs=100,
             random_state=19, learning_rate=1.0, repulsion_strength=1.0,
             negative_sample_rate=5, numeric_mode="identical")
embedding = model.fit_transform(X_train)
held_out = model.transform(X_test)
```

Input is dense, converted to FP32. CSR describes the stored neighbor graph.
Exact neighbor search still performs quadratic pair comparisons. Supervision,
alternate metrics and alternate initialization are unsupported. Transform
uses frozen private training data/embedding copies; changing query batching
does not change results, a batch of N being the concatenation of N batches of
one. Neither API similarity nor neighborhood-quality agreement means bitwise
agreement with cuML or umap-learn.

## Transformer and training primitives

`TransformerBlock(weights, n_heads=..., n_kv_heads=..., numeric_mode=...)`
accepts the explicit FP32 weight registry documented on the class. Use
`forward(x)` for a full uncached call, or allocate a `TransformerState` with
`allocate_state(batch_size, max_tokens)` and pass it to `forward`/`step`.
The state contains KV buffers and a token cursor. This block API exposes
forward operations; it is not a generic transformer trainer.

```python
from mojolearn.training import AdamW, cross_entropy, clip_grad_norm_

loss, dlogits = cross_entropy(logits, targets, return_grad=True,
                             numeric_mode="identical")
optimizer = AdamW(parameters, lr=0.001, numeric_mode="identical")
# The caller computes parameter gradients in the same tensor order.
optimizer.step(parameter_gradients)
```

`logits` is FP32 `(N, classes)` and `targets` contains integer class indices.
The optimizer mutates supplied parameter arrays in place, which is why they
must be writable, C-contiguous float32 buffers. `clip_grad_norm_` mutates
gradient arrays the same way and returns the pre-clip L2 norm. These primitives do not
provide automatic differentiation or a failure-rollback contract. Their
docstrings describe reductions, ignore indices, smoothing, optimizer settings
and refusals.

## Fixed small trainers

Both trainers copy caller weights, serialize state updates with a lock, and
publish a completed update only after native work succeeds. Their state
snapshots and separate canonical JSON checkpoints retain all parameters,
moments, flags, optimizer configuration, step and data-schedule descriptor.
The caller supplies the next batch after resume; neither trainer fetches data
or silently changes the process-selected numeric mode.

`SmallMLPTrainer(weight1, bias1, weight2, bias2, data_schedule=...)` requires
FP32 shapes `(16, 8)`, `(16,)`, `(3, 16)` and `(3,)`. `train_step(X, targets,
return_input_grad=True)` returns pre-update loss/logits and all parameter
gradients, with an optional input gradient. `predict_logits(X)`, `state_dict`,
`load_state_dict`, `save_checkpoint` and `from_checkpoint` are available.
The process must already select IDENTICAL.

`SmallByteLanguageModelTrainer(parameters, data_schedule=...)` accepts copied
FP32 flat weights of length 34,944 or its exact 20-name parameter registry,
available through `parameter_registry()`. Its profile fixes batch 2, context
32, width 32, two decoder blocks, four query heads, two KV heads, feed-forward
width 64 and byte vocabulary 256. `train_step(ids)` takes actual int32 shape
`(2, 33)`, predicts the next byte at 32 positions per row, and returns mean
loss, pre-update gradients and the completed-step cursor. `evaluate(ids)`
computes loss while requiring the complete training state to remain unchanged.
It also exposes complete state/checkpoint methods and `run_metadata()`.
With `resident=True` the trainer keeps parameters, moments and the gradient
on the device between calls and `train_step` returns only loss, step and
flags (`step_result='lean'`, the resident default since 2026-09-11);
`export_state()`, `export_gradients()` and `export_checkpoint()` copy them
out on demand with the same validation, `step_result='full'` restores the
complete dict, and a failed step rolls the device state back to the last
committed step. Every exported array is a copy; mutating it cannot reach
the session.
CUDA/HIP and process-selected IDENTICAL are required. This is a bounded
training demonstration, with no useful-generation or reasoning-quality claim.

## Evidence boundary

New public modules are export and packaging changes. Their examples and
installed-wheel imports must be checked in the actual alpha artifact. Existing
kernel cards apply only to their recorded profiles and inputs; they do not
automatically qualify new Python wrappers, new optimizer paths, or a whole
training trajectory. Source-build results do not certify a published wheel.

For new training work, independent-reference tolerance correctness, repeated
raw-byte agreement, cross-vendor state agreement, successful checkpoint resume,
and held-out learning are separate checks. A successful guard exit, the exact
source/binding/input witnesses, and effective failure controls are required
before admitting the corresponding retained run. No universal bitwise claim
or reference feature-parity claim is implied by the alpha export list.
