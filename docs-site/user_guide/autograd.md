# Build your own model

mojolearn trains its neural models with a native autograd engine: every forward operation is recorded,
and the backward pass runs the matching gradient kernels on the device, with each sum taken in a fixed
order. Two public entry points give you the same engine. `Sequential` assembles a feed-forward model
layer by layer and trains it with one call. `mojolearn.ops.Session` hands you the operations
themselves, so you can compose your own inference or training algorithm without writing a kernel.

In `identical` mode, mojolearn returns the same bits on Apple, NVIDIA and AMD GPUs. Anything you build
from these operations inherits that: for the same inputs and seeds, your model's parameters and
predictions are the same on every vendor.

The full list of names is in the [API reference](../api/autograd.rst). Both entry points take NumPy
arrays and need the `mojolearn[numpy]` extra (see [Install](../install.md)).

## Sequential quickstart

```python
import numpy as np
import mojolearn

rng = np.random.default_rng(0)
X = rng.standard_normal((200, 4), dtype=np.float32)
y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)

clf = (mojolearn.Sequential()
       .linear(16).relu()
       .linear(2)
       .compile(loss="cross_entropy", optimizer="adam", lr=0.01))
clf.fit(X, y, epochs=20, batch_size=32, seed=0)

proba = clf.predict_proba(X)   # (200, 2), columns in clf.classes_ order
labels = clf.predict(X)        # (200,)
print(clf.loss_curve_[-1])
```

The parameters are initialized from `seed`, the rows are shuffled from `seed` every epoch, and
`clf.params_` holds every trained parameter as one flat float32 vector. For regression, use
`loss="mse"` or `loss="huber"` with one output per target column.

## Operation session quickstart

A session keeps tensors on the device and runs one native operation per call. Each call returns an
integer handle; values come back to Python only when you ask for them.

```python
import numpy as np
from mojolearn import ops

rng = np.random.default_rng(0)
X = rng.standard_normal((64, 4), dtype=np.float32)
W = rng.standard_normal((4, 3), dtype=np.float32)

with ops.Session() as s:
    x = s.tensor(X)
    h = s.linear(x, s.tensor(W), activation="relu")   # relu(X @ W)
    p = s.softmax(h)
    total = s.scalar(s.sum(h))      # one value back, for a branch in your code
    probs = s.numpy(p)              # (64, 3) float32
```

Use `scalar` when your Python code has to branch on a computed value; keep everything else on the
device.

With `Session(record=True)` the same composition is differentiable. Mark the trainable inputs with
`requires_grad=True`, compute a `(1, 1)` loss, call `backward`, and read each gradient with `grad`:

```python
Y = rng.standard_normal((64, 3), dtype=np.float32)

with ops.Session(record=True) as s:
    w = s.tensor(W, requires_grad=True)
    loss = s.mse(s.matmul(s.tensor(X), w), s.tensor(Y))
    s.backward(loss)
    grad_w = s.grad(w)               # (4, 3), the gradient of the loss with respect to W
```

## What is available

| Group | `Sequential` | `ops.Session` |
|---|---|---|
| Layers and operations | `linear`, `relu`, `sigmoid`, `tanh`, `softmax`, `layer_norm`, `residual` | `matmul`, `add`, `sub`, `mul`, `div` (with row, column and scalar broadcasting), `relu`, `sigmoid`, `tanh`, `exp`, `log`, `sqrt`, `scale`, `transpose`, `reshape`, `sum`, `mean`, `softmax`, `log_softmax`, `linear`, `layer_norm`, `dropout`, `full`, `copy`, `detach` |
| Losses | `mse`, `huber`, `cross_entropy`, `bce_with_logits` | `mse`, `mae`, `huber`, `bce_with_logits`, `cross_entropy` (mean or sum) |
| Metrics | `loss_curve_` | `accuracy`, `log_loss`, `roc_auc` |
| Optimizers | `sgd`, `momentum` (optionally Nesterov), `adam`, `adamw`, with weight decay | gradients from `backward` and `grad`; apply your own update with the session's operations |
| Schedules and stopping | `constant`, `step`, `exponential`; stop after `n_iter_no_change` epochs without improvement | your own loop |
| Data | shuffling and batching from `seed` | `permutation` (a seeded row order), `gather_rows` (a batch of rows), `dropout` (a seeded mask) |

## Example: a two-expert mixture classifier

This example builds a model mojolearn does not ship: two linear experts and a softmax gate that weights
them per row. It is trained with a recording session, then used two ways: the soft mixture, and a hard
route that sends the whole batch to the expert the gate prefers on average.

```python
import hashlib
import numpy as np
from mojolearn import ops

rng = np.random.default_rng(0)
X = rng.standard_normal((128, 4), dtype=np.float32)
y = ((X[:, 0] > 0) == (X[:, 1] > 0)).astype(np.float32).reshape(-1, 1)   # class codes 0 and 1

params = {
    "W0": 0.1 * rng.standard_normal((4, 2), dtype=np.float32), "b0": np.zeros((1, 2), np.float32),
    "W1": 0.1 * rng.standard_normal((4, 2), dtype=np.float32), "b1": np.zeros((1, 2), np.float32),
    "Wg": 0.1 * rng.standard_normal((4, 2), dtype=np.float32), "bg": np.zeros((1, 2), np.float32),
}
FIRST = np.array([[1.0], [0.0]], np.float32)    # picks gate column 0
SECOND = np.array([[0.0], [1.0]], np.float32)   # picks gate column 1


def mixture(s, x, p, first, second):
    """Per-row gate-weighted sum of the two experts' logits, and the gate."""
    z0 = s.linear(x, p["W0"], p["b0"])
    z1 = s.linear(x, p["W1"], p["b1"])
    gate = s.linear(x, p["Wg"], p["bg"], activation="softmax")
    logits = s.add(s.mul(s.matmul(gate, first), z0), s.mul(s.matmul(gate, second), z1))
    return logits, gate


# Train: full-batch gradient descent, every step on the device.
lr = 0.5
with ops.Session(record=True) as s:
    x, codes = s.tensor(X), s.tensor(y)
    first, second = s.tensor(FIRST), s.tensor(SECOND)
    for step in range(60):
        h = {k: s.tensor(v, requires_grad=True) for k, v in params.items()}
        logits, _ = mixture(s, x, h, first, second)
        s.backward(s.cross_entropy(logits, codes))
        for k, hk in h.items():
            g = s.tensor(s.grad(hk))
            params[k] = s.numpy(s.sub(hk, s.scale(g, lr)))   # p - lr * grad
            s.free(g)
            s.free(hk)
        s.reset()

# Predict: the soft mixture, and a hard route chosen from one value read back.
with ops.Session() as s:
    x = s.tensor(X)
    p = {k: s.tensor(v) for k, v in params.items()}
    logits, gate = mixture(s, x, p, s.tensor(FIRST), s.tensor(SECOND))
    proba = s.numpy(s.softmax(logits))
    accuracy = s.accuracy(s.tensor(y), logits)

    margin = s.matmul(s.mean(gate, axis=0), s.tensor(np.array([[1.0], [-1.0]], np.float32)))
    expert = "0" if s.scalar(margin) > 0 else "1"
    hard = s.numpy(s.softmax(s.linear(x, p["W" + expert], p["b" + expert])))

print("accuracy", accuracy)
print("digest", hashlib.sha256(proba.tobytes()).hexdigest()[:16])
```

The last line prints a digest of the predicted probabilities. Run the script on an Apple, an NVIDIA and
an AMD GPU: in `identical` mode the three digests are the same.

## Numeric modes

Both entry points take `numeric_mode`: `Sequential(numeric_mode=...)` and
`ops.Session(numeric_mode=...)`. The default is `identical`. `fast` is available on Apple GPUs only.
See [Numeric modes](numeric_modes.md).
