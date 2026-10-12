# Neural networks

mojolearn ships neural building blocks, forward-only inference classes, small trainers, and the optimizer
and loss primitives they are built from. In `identical` mode (the default) all of them compute in FP32.
The full list of names is in the [neural API reference](../api/neural.rst).

These are bounded building blocks, not a general autograd framework or a drop-in PyTorch replacement.
Each class documents the settings it supports, and anything else is refused by name.

## What is available

| Group | Names |
|---|---|
| Decoder blocks | `TransformerBlock`, `Mamba1Block`, `Mamba2Block`, `Mamba3Block`, `MoEBlock` |
| Stacks | `SambaConfig`, `SambaStack` (Mamba-3 and attention layers in any order) |
| Forward-only (CPU) | `TransformerBlockInference`, `Mamba1BlockInference`, `Mamba2BlockInference`, `Mamba3BlockInference`, `SambaInference`, `MLPInference` |
| Decode state | `TransformerState`, `Mamba1State`, `Mamba2State`, `Mamba3State` |
| Layers | `Embedding`, `LayerNorm`, `Conv1d`, `Conv2d`, `BatchNorm1d`, `BatchNorm2d`, `MaxPool1d`, `MaxPool2d`, `AvgPool1d`, `AvgPool2d`, `AdaptiveAvgPool2d`, `AdaptiveMaxPool2d`, `Dropout2d`, `BasicBlock`, `GCNConv`, `SAGEConv` |
| Estimators | `MLPClassifier`, `MLPRegressor`, `CNNClassifier`, `RNNClassifier`, `RNNRegressor`, `LSTMClassifier`, `LSTMRegressor`, `GRUClassifier`, `GRURegressor` |
| Trainers | `SmallMLPTrainer`, `SambaStack.train_step` |
| Optimizers | `SGD`, `Adam`, `AdamW`, `Adagrad`, `Adamax`, `Adafactor`, `NAdam`, `RMSprop`, `LAMB`, `Lion` |
| Schedules | `StepLR`, `ExponentialLR`, `OneCycleLR` |
| Functions | `cross_entropy`, `clip_grad_norm_`, `layer_norm_forward`, `layer_norm_backward` |

The GPU classes run on the GPU and have no CPU fallback. The `*Inference` classes run the same forward
pass on the CPU, from the same weights. The CNN and sequence estimators take NumPy arrays and need
the `mojolearn[numpy]` extra (see [Install](../install.md)).

## Weights come from you

The decoder blocks have no initializer. You pass a dict of float32 arrays keyed by the reference
parameter names (Hugging Face's names for Llama and Mamba). The block reads its shapes from the arrays and
from a few shape arguments. A `TransformerBlock` takes nine weights:

| Name | Shape |
|---|---|
| `input_layernorm.weight`, `post_attention_layernorm.weight` | `(d_model,)` |
| `q_proj.weight` | `(n_heads * head_dim, d_model)` |
| `k_proj.weight`, `v_proj.weight` | `(n_kv_heads * head_dim, d_model)` |
| `o_proj.weight` | `(d_model, n_heads * head_dim)` |
| `gate_proj.weight`, `up_proj.weight` | `(intermediate, d_model)` |
| `down_proj.weight` | `(d_model, intermediate)` |

Linear weights are `[out_features, in_features]`, as in PyTorch. `head_dim` must be even, `d_model` must
equal `n_heads * head_dim`, and `n_heads` must be a multiple of `n_kv_heads`.

## A block forward pass

```python
import numpy as np
import mojolearn

rng = np.random.default_rng(0)
d_model, n_heads, n_kv_heads, head_dim, inter = 64, 4, 2, 16, 128

def w(*shape):
    return (0.02 * rng.standard_normal(shape)).astype(np.float32)

weights = {
    "input_layernorm.weight": np.ones(d_model, np.float32),
    "post_attention_layernorm.weight": np.ones(d_model, np.float32),
    "q_proj.weight": w(n_heads * head_dim, d_model),
    "k_proj.weight": w(n_kv_heads * head_dim, d_model),
    "v_proj.weight": w(n_kv_heads * head_dim, d_model),
    "o_proj.weight": w(d_model, n_heads * head_dim),
    "gate_proj.weight": w(inter, d_model),
    "up_proj.weight": w(inter, d_model),
    "down_proj.weight": w(d_model, inter),
}

block = mojolearn.TransformerBlock(weights, n_heads=n_heads, n_kv_heads=n_kv_heads)
x = rng.standard_normal((2, 16, d_model)).astype(np.float32)   # (B, L, d_model)
y = block.forward(x)                                           # (B, L, d_model)
```

With no state, `forward` runs a self-contained prefill and discards the cache. To decode, allocate a state
and carry it:

```python
state = block.allocate_state(2, 64)        # batch_size, max_tokens
y = block.forward(x, state)                # prefill 16 positions
x_next = rng.standard_normal((2, 1, d_model)).astype(np.float32)
y_next = block.step(x_next, state)         # one more position per row
```

In `identical` mode, a prefill followed by `step` calls gives the same bits as one prefill over the whole
sequence. `forward(x, lengths=[...])` runs a ragged batch: each row's real positions match that row run
alone, and padding positions return `+0.0`. The Mamba blocks follow the same pattern, except that
`allocate_state` takes only `batch_size`.

`TransformerBlock` also takes options for common decoder families: `rope_theta`, `rope_scaling`,
`window` (sliding window), `qkv_bias`, `o_bias`, `norm`, `mlp`, `qk_norm` and `attn_softcap`.

## Inference on the CPU

Each block has a forward-only twin with the same constructor that runs on the CPU, so a model trained on a
GPU can serve on a machine with no GPU:

```python
cpu_block = mojolearn.TransformerBlockInference(weights, n_heads=n_heads, n_kv_heads=n_kv_heads)
y_cpu = cpu_block.forward(x)
```

`backward` is refused on the inference classes. `SambaInference.from_checkpoint(path)` loads a file
written by `SambaStack.save_checkpoint`, and `MLPInference.from_checkpoint(path)` loads a
`SmallMLPTrainer` checkpoint.

## Low-bit weight storage

`mojolearn.lowbit.pack(weights, fmt)` converts every 2-D float32 tensor in a weight dict to `"bfloat16"`
or `"int8"` (one power-of-two scale per output row). Each inference class accepts the packed dict and
widens it back to float32 before computing, so the arithmetic is the FP32 path:

```python
from mojolearn import lowbit

packed = lowbit.pack(weights, "bfloat16")
small = mojolearn.TransformerBlockInference(packed, n_heads=n_heads, n_kv_heads=n_kv_heads)
```

The result is the same bits as the FP32 block run on the widened weights. It is not the same as the
original FP32 model, because a bf16 or int8 weight is a rounded weight. This saves memory and disk; it is
not a lower-precision matrix unit.

## A Samba stack

`SambaStack` holds a token embedding, a sequence of `"mamba3"` and `"attention"` layers, a final norm and
the output head, with its own AdamW optimizer:

```python
from mojolearn.training import Generator

cfg = mojolearn.SambaConfig(vocab=256, d_model=64, layers=("mamba3", "attention"),
                            n_heads=4, head_dim=16)
stack = mojolearn.SambaStack(cfg, generator=Generator(7), lr=1e-3, max_norm=1.0)
inputs = rng.integers(0, 256, size=(2, 32)).astype(np.int32)
targets = rng.integers(0, 256, size=(2, 32)).astype(np.int32)
out = stack.train_step(inputs, targets)
stack.save_checkpoint("run.json")
```

Supply exactly one of `weights` (a dict keyed by `cfg.registry()` names) or `generator` (a seeded
`mojolearn.training.Generator`). `d_model` must
be a multiple of 32 for a Mamba-3 layer and equal `n_heads * head_dim` for an attention layer.

## Training primitives

The optimizers update caller-owned float32 arrays in place. Pass gradients in the same order as the
parameters; that order is part of the result.

```python
w1 = (0.1 * rng.standard_normal((16, 8))).astype(np.float32)
b1 = np.zeros(16, np.float32)
opt = mojolearn.AdamW([w1, b1], lr=1e-3, weight_decay=0.01)

logits = rng.standard_normal((32, 10)).astype(np.float32)
targets = rng.integers(0, 10, size=32).astype(np.int32)
loss, dlogits = mojolearn.cross_entropy(logits, targets, return_grad=True)

# ... compute g_w1 and g_b1 from dlogits ...
total_norm = mojolearn.clip_grad_norm_([g_w1, g_b1], max_norm=1.0)   # scales in place
opt.step([g_w1, g_b1])
```

`clip_grad_norm_` returns the norm before clipping, as PyTorch does. `opt.step(grads, max_norm=...)` clips
and updates in one call. `cross_entropy` takes `(N, V)` float32 logits and `(N,)` integer class indices.

`SmallMLPTrainer` is a complete trainer for a fixed 8 -> 16 -> 3 ReLU network with mean cross-entropy
and AdamW. It saves and restores its whole state, and `data_schedule` (a JSON object you write) is stored
with the step counter so you can resume on the right batch.

## CNN and sequence estimators

These follow the scikit-learn interface:

```python
X_img = rng.random((512, 1 * 8 * 8), dtype=np.float32)
y_img = rng.integers(0, 3, size=512)
cnn = mojolearn.CNNClassifier(input_shape=(1, 8, 8), conv_channels=(8,), max_iter=5).fit(X_img, y_img)

X_seq = rng.random((256, 20, 4), dtype=np.float32)   # (n_samples, n_timesteps, n_features)
y_seq = rng.integers(0, 2, size=256)
lstm = mojolearn.LSTMClassifier(hidden_size=16, max_epochs=5).fit(X_seq, y_seq)
print(lstm.predict(X_seq[:5]))
```

`CNNClassifier` accepts `(n, C*H*W)` rows with `input_shape=(C, H, W)`, or `(n, C, H, W)`.

## Related pages

- [Language models](language_models.md): decoder language models, tokenizers and checkpoints.
- [Multi-GPU training](multi_gpu.md): ordered gradients across several GPUs.
- [Numeric modes](numeric_modes.md): what `identical` and `fast` promise.
