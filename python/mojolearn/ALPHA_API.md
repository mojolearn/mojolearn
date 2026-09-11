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
| Small MLP training | `mojolearn.neural_network.SmallMLPTrainer` | `_mojolearn_linalg`, `_mojolearn_training` | Fixed 8→16→3 FP32 ReLU, mean cross entropy and AdamW; batch 1–256 |
| Small byte-LM training | `mojolearn.language_model.SmallByteLanguageModelTrainer` | `_mojolearn_byte_lm` | Fixed two-block FP32 next-byte model, full gradients and AdamW; CUDA/HIP IDENTICAL only |

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
can change results. Neither API similarity nor neighborhood-quality agreement
means bitwise agreement with cuML or umap-learn.

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
or upstream feature-parity claim is implied by the alpha export list.
