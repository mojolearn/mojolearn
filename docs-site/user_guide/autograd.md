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

<!-- block: seq-quickstart -->
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

<!-- block: sess-setup -->
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

<!-- block: sess-loss -->
```python
Y = rng.standard_normal((64, 3), dtype=np.float32)

with ops.Session(record=True) as s:
    w = s.tensor(W, requires_grad=True)
    loss = s.mse(s.matmul(s.tensor(X), w), s.tensor(Y))
    s.backward(loss)
    grad_w = s.grad(w)               # (4, 3), the gradient of the loss with respect to W
```

To train, make an optimizer with `optimizer` and call `step` after each `backward`. The parameters are
updated in place on the device, and the optimizer's state (momentum, moment estimates, step count) stays
there too, so the loop reads nothing back:

<!-- block: sess-train -->
```python
with ops.Session(record=True) as s:
    x, y = s.tensor(X), s.tensor(Y)
    w = s.tensor(W, requires_grad=True)
    opt = s.optimizer("adam", lr=0.01, schedule="step", step_size=10, gamma=0.5)
    for epoch in range(30):
        s.backward(s.mse(s.matmul(x, w), y))
        s.step(opt, [w])             # w updated on the device
        s.scheduler_step(opt)        # one epoch of the learning-rate schedule
        s.reset()                    # drop this pass's graph; w and its optimizer state stay
    trained = s.numpy(w)
```

The optimizers and schedules are the ones `Sequential` uses: the same model, data, starting weights and
options give the same weights, bit for bit. `optimizer_state` returns a parameter's state for a
checkpoint and `load_optimizer_state` restores it. To sum the gradients of several losses before one
step, call `zero_grad`, then `backward(loss, accumulate=True)` for each loss.

## What is available

| Group | `Sequential` | `ops.Session` |
|---|---|---|
| Layers and operations | `linear`, `relu`, `sigmoid`, `tanh`, `softmax`, `layer_norm`, `residual` | `matmul`, `add`, `sub`, `mul`, `div` (with row, column and scalar broadcasting), `relu`, `sigmoid`, `tanh`, `exp`, `log`, `sqrt`, `scale`, `transpose`, `reshape`, `sum`, `mean`, `softmax`, `log_softmax`, `linear`, `layer_norm`, `dropout`, `full`, `copy`, `detach` |
| Losses | `mse`, `huber`, `cross_entropy`, `bce_with_logits` | `mse`, `mae`, `huber`, `bce_with_logits`, `cross_entropy` (mean or sum) |
| Metrics | `loss_curve_` | `accuracy`, `log_loss`, `roc_auc` |
| Optimizers | `sgd`, `momentum` (optionally Nesterov), `adam`, `adamw`, with weight decay | `optimizer` (`sgd` with optional momentum and Nesterov, `adam`, `adamw`, with weight decay) and `step`, in place on the device; `zero_grad`; `optimizer_state` and `load_optimizer_state` for checkpoints |
| Schedules and stopping | `constant`, `step`, `exponential`; stop after `n_iter_no_change` epochs without improvement | `constant`, `step`, `exponential`, advanced by `scheduler_step`; your own stopping rule |
| Data | shuffling and batching from `seed` | `permutation` (a seeded row order), `gather_rows` (a batch of rows), `dropout` (a seeded mask) |

## Example: a two-expert mixture classifier

This example builds a model mojolearn does not ship: two linear experts and a softmax gate that weights
them per row. It is trained with a recording session, then used two ways: the soft mixture, and a hard
route that sends the whole batch to the expert the gate prefers on average.

<!-- block: mixture -->
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
with ops.Session(record=True) as s:
    x, codes = s.tensor(X), s.tensor(y)
    first, second = s.tensor(FIRST), s.tensor(SECOND)
    h = {k: s.tensor(v, requires_grad=True) for k, v in params.items()}
    opt = s.optimizer("sgd", lr=0.5)
    for step in range(60):
        logits, _ = mixture(s, x, h, first, second)
        s.backward(s.cross_entropy(logits, codes))
        s.step(opt, list(h.values()))   # p - lr * grad, in place on the device
        s.reset()
    params = {k: s.numpy(hk) for k, hk in h.items()}

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

## Second derivatives

`backward(loss, create_graph=True)` records the gradient computation itself. Each gradient is then a
tensor: `grad_tensor(h)` returns its handle, you can keep computing with it, and a second `backward` from
a scalar built on it gives second derivatives. `grad(h)` still returns the last backward's gradient as an
array.

A Hessian-vector product by hand, then the same in one call:

<!-- block: ho-setup -->
```python
rng = np.random.default_rng(0)
X = rng.standard_normal((64, 4), dtype=np.float32)
Y = rng.standard_normal((64, 1), dtype=np.float32)
W0 = rng.standard_normal((4, 1), dtype=np.float32)
V = rng.standard_normal((4, 1), dtype=np.float32)

with ops.Session(record=True) as s:
    x, y = s.tensor(X), s.tensor(Y)
    w = s.tensor(W0, requires_grad=True)
    loss = s.mse(s.linear(x, w, activation="tanh"), y)

    s.backward(loss, create_graph=True)   # the gradients are tensors now
    g = s.grad_tensor(w)                  # dL/dw, differentiable
    v = s.tensor(V)                       # the direction
    s.backward(s.sum(s.mul(g, v)))        # a second backward
    hv = s.grad(w)                        # H v, (4, 1)

    (hv_handle,) = s.hvp(loss, [w], [v])  # the same product in one call
    hv2 = s.numpy(hv_handle)
```

`hvp(loss, params, v)` takes one direction per parameter and returns one handle per parameter: the
Hessian of the loss with respect to all the parameters together, times the stacked directions. Where
the loss has no curvature (a parameter it depends on linearly) the result is zeros.

`vjp(output, inputs, v)` gives `J^T v` for an output of any shape and a seed `v` of its shape, one handle
per input; `create_graph=True` makes the results differentiable. `jvp(output, inputs, u)` gives `J u`,
the derivative of `output` along one direction per input, as a handle of `output`'s shape.

<!-- block: ho-hvp -->
```python
with ops.Session(record=True) as s:
    x = s.tensor(X, requires_grad=True)
    w = s.tensor(W0)
    out = s.tanh(s.matmul(x, w))                       # (64, 1)
    (jt_v,) = s.vjp(out, [x], s.full((64, 1), 1.0))   # (64, 4)
    j_u = s.jvp(out, [x], [s.tensor(np.ones((64, 4), dtype=np.float32))])  # (64, 1)
```

### A physics-informed loss

Derivatives of a network's output with respect to its input, differentiated again, train a network to
satisfy a differential equation. Here `u(x)` is a small network of one input, fitted so that
`u''(x) = -sin(x)`. The rows are independent, so the gradient of `sum(u)` with respect to `x` holds
`du/dx` row by row, and the gradient of `sum(du/dx)` holds `d2u/dx2`:

<!-- block: ho-physics -->
```python
Xc = np.linspace(-3, 3, 128, dtype=np.float32).reshape(-1, 1)   # collocation points
F = np.sin(Xc)

with ops.Session(record=True) as s:
    w1 = s.tensor(0.5 * rng.standard_normal((1, 16), dtype=np.float32), requires_grad=True)
    b1 = s.tensor(np.zeros((1, 16), dtype=np.float32), requires_grad=True)
    w2 = s.tensor(0.5 * rng.standard_normal((16, 1), dtype=np.float32), requires_grad=True)
    x, f = s.tensor(Xc, requires_grad=True), s.tensor(F)
    opt = s.optimizer("adam", lr=0.01)
    for epoch in range(200):
        u = s.matmul(s.linear(x, w1, b1, activation="tanh"), w2)   # (128, 1)
        s.backward(s.sum(u), create_graph=True)
        du = s.grad_tensor(x)                                        # du/dx per row
        s.backward(s.sum(du), create_graph=True)
        d2u = s.grad_tensor(x)                                       # d2u/dx2 per row
        r = s.add(d2u, f)                                            # the residual u'' + sin(x)
        s.backward(s.mean(s.mul(r, r)))                              # a third backward
        s.step(opt, [w1, b1, w2])
        s.reset()
```

Each recorded backward keeps its intermediate tensors until `reset`, and takes a few operations per
recorded operation, so use `create_graph=True` only where you differentiate again. `layer_norm` has a
first-order gradient only: a backward with `create_graph=True` through it raises an error that says so.

## Save a trained model for inference

`model.export("model.ckpt")` writes the layers and the trained parameters (and a classifier's classes),
without the optimizer state. `mojolearn.Sequential.load("model.ckpt")` reads it back on any install, a
GPU install or the CPU-only install, and `predict` and `predict_proba` return the same values there as
on the machine that trained it. To keep training later, save with `save_checkpoint` instead and see
[Moving training between vendors](cross_vendor_training.md).

<!-- block: export -->
```python
clf.export("model.ckpt")
served = mojolearn.Sequential.load("model.ckpt")
labels_served = served.predict(X)
```

## Numeric modes

Both entry points take `numeric_mode`: `Sequential(numeric_mode=...)` and
`ops.Session(numeric_mode=...)`. The default is `identical`. `fast` is available on Apple GPUs only.
See [Numeric modes](numeric_modes.md).
