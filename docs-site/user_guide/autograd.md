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
| Shapes | `Sequential(input_shape=...)`, `conv2d`, `max_pool2d`, `avg_pool2d`, `reshape`, `permute`, `flatten` | tensors of up to 8 dimensions; `shape`; batched `matmul`; `reshape`, `flatten`, `squeeze`, `unsqueeze`, `permute`, `transpose` of any two axes; `sum`, `mean`, `max` over axes; `softmax`, `log_softmax` along an axis; `concat`, `split`, `take`; `conv2d`, `max_pool2d`, `avg_pool2d` |
| Data | shuffling and batching from `seed` | `permutation` (a seeded row order), `gather_rows` (a batch of rows), `dropout` (a seeded mask) |
| Import weights | `load_state_dict` (a PyTorch `state_dict` or a safetensors checkpoint) | `load_weights` |

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

## Shapes

Tensors are not limited to matrices. `tensor` keeps the shape of an array of up to 8 dimensions (an
array of 3 or more dimensions must be C-contiguous: pass `numpy.ascontiguousarray(x)`), `shape`
returns every dimension, and `numpy` returns an array of that shape. The operations follow NumPy's
conventions:

- `matmul` multiplies the last two axes; the leading axes are batch axes and broadcast, so a
  `(batch, heads, m, k)` tensor times a `(batch, heads, k, n)` or a `(k, n)` tensor gives
  `(batch, heads, m, n)`.
- `sum`, `mean` and `max` take `axis` (an integer or a tuple, negative axes counting from the end)
  and `keepdims`. `softmax` and `log_softmax` take `axis`, the last axis by default.
- `reshape` (one dimension may be -1), `flatten`, `squeeze`, `unsqueeze`, `permute` and `transpose`
  (any two axes; the last two by default) rearrange a tensor. `concat` joins tensors along an axis,
  `split` cuts one into pieces and `take` picks slices at integer indices.
- `linear` acts on the last axis, and `cross_entropy` takes `(..., classes)` logits with one class
  code per position of the leading axes.
- `conv2d`, `max_pool2d` and `avg_pool2d` work on `(N, C, H, W)` images.

Every one of these records its gradient in a recording session. A small image classifier and an
attention-style product, each with one backward pass:

<!-- block: shapes -->
```python
import hashlib
import numpy as np
from mojolearn import ops

rng = np.random.default_rng(0)
images = rng.standard_normal((8, 1, 8, 8), dtype=np.float32)        # (batch, channels, height, width)
left, right = images[:, 0, :4].mean(axis=(1, 2)), images[:, 0, 4:].mean(axis=(1, 2))
codes = (left > right).astype(np.float32).reshape(-1, 1)             # one class code per image
K = 0.3 * rng.standard_normal((4, 1, 3, 3), dtype=np.float32)        # 4 filters of 3 x 3
W = 0.1 * rng.standard_normal((64, 2), dtype=np.float32)             # 4 x 4 x 4 features -> 2 classes

with ops.Session(record=True) as s:
    k = s.tensor(K, requires_grad=True)
    kb = s.tensor(np.zeros((1, 4), np.float32), requires_grad=True)
    w = s.tensor(W, requires_grad=True)
    h = s.relu(s.conv2d(s.tensor(images), k, kb, padding=1))        # (8, 4, 8, 8)
    h = s.max_pool2d(h, 2)                                           # (8, 4, 4, 4)
    logits = s.linear(s.flatten(h), w)                               # (8, 64) -> (8, 2)
    s.backward(s.cross_entropy(logits, s.tensor(codes)))
    grad_k, grad_kb, grad_w = s.grad(k), s.grad(kb), s.grad(w)       # (4, 1, 3, 3), (1, 4), (64, 2)

Q = rng.standard_normal((2, 4, 6, 4), dtype=np.float32)              # (batch, heads, queries, features)
Kv = rng.standard_normal((2, 4, 6, 4), dtype=np.float32)             # keys, also used as values

with ops.Session(record=True) as s:
    q, kv = s.tensor(Q, requires_grad=True), s.tensor(Kv)
    scores = s.scale(s.matmul(q, s.transpose(kv)), 0.5)              # (2, 4, 6, 6), 1 / sqrt(4)
    out = s.matmul(s.softmax(scores), kv)                            # (2, 4, 6, 4)
    merged = s.reshape(s.permute(out, (0, 2, 1, 3)), (2, 6, 16))     # the heads side by side
    s.backward(s.mean(merged))
    attention = s.numpy(merged)
    grad_q = s.grad(q)                                               # (2, 4, 6, 4)

digest = hashlib.sha256()
for a in (grad_k, grad_kb, grad_w, attention, grad_q):
    digest.update(a.tobytes())
print("digest", digest.hexdigest()[:16])
```

In `identical` mode, mojolearn returns the same bits on Apple, NVIDIA and AMD GPUs: run this script on
each and the printed digests are the same.

`Sequential` takes the same shapes. Give it the shape of one sample with `input_shape`, or let `fit`
take it from `X`, and build with `conv2d`, `max_pool2d`, `avg_pool2d`, `reshape`, `permute` and
`flatten`. A `linear` layer acts on the last axis of each sample; the model must end with one vector
per sample, so a model on images ends with `flatten` and a `linear` layer:

<!-- block: shapes-seq -->
```python
import hashlib
import numpy as np
import mojolearn

rng = np.random.default_rng(0)
images = rng.standard_normal((64, 1, 8, 8), dtype=np.float32)
labels = (images[:, 0, :4].mean(axis=(1, 2)) > images[:, 0, 4:].mean(axis=(1, 2))).astype(np.int64)

cnn = (mojolearn.Sequential(input_shape=(1, 8, 8))
       .conv2d(4, 3, padding=1).relu()      # 4 filters of 3 x 3: (4, 8, 8) per image
       .max_pool2d(2)                       # (4, 4, 4)
       .flatten()                           # 64 values
       .linear(2)
       .compile(loss="cross_entropy", optimizer="adam", lr=0.01))
cnn.fit(images, labels, epochs=10, batch_size=16, seed=0)

proba = cnn.predict_proba(images)                   # (64, 2)
proba_flat = cnn.predict_proba(images.reshape(64, -1))   # rows of 64 values, read as (1, 8, 8)
print("digest", hashlib.sha256(cnn.params_.tobytes() + proba.tobytes()).hexdigest()[:16])
```

With an `input_shape`, `fit` and `predict` also take a 2-D `X` whose rows hold each sample's values
in row-major order, and give the same results. `params_` holds a conv2d layer's weight
`(out_channels, in_channels, kernel_size, kernel_size)` row-major, then its bias.

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

## Run a captured model

Each operation of a session is one call from Python. Once a composition works, `capture` turns it into a
`Program` that runs the whole model in one call: `run` uploads new input arrays, runs every captured
operation on the device and returns the outputs, the same values to the bit as calling the operations
one by one. Capture from an inference session (`Session()`, without `record=True`). The tensors the
outputs read that are not inputs, such as the weights, are the program's parameters: `run` reads them
where they are, so they are not copied per call. `save` writes the program and its parameters, and
`load_program` reads the file back on any install, a GPU install or the CPU-only install.

<!-- block: program -->
```python
import numpy as np
from mojolearn import ops

rng = np.random.default_rng(0)
K = 0.3 * rng.standard_normal((4, 1, 3, 3), dtype=np.float32)       # 4 filters of 3 x 3
W = 0.1 * rng.standard_normal((64, 2), dtype=np.float32)            # 4 x 4 x 4 features -> 2 classes
batch1 = rng.standard_normal((8, 1, 8, 8), dtype=np.float32)
batch2 = rng.standard_normal((8, 1, 8, 8), dtype=np.float32)

with ops.Session() as s:
    images = s.tensor(batch1)
    h = s.max_pool2d(s.relu(s.conv2d(images, s.tensor(K), padding=1)), 2)
    proba = s.softmax(s.linear(s.flatten(h), s.tensor(W)))         # (8, 2)
    eager = s.numpy(proba)
    model = s.capture(inputs=[images], outputs=[proba])
    s.reset()                         # the program keeps what it needs
    first = model.run(batch1)         # equal to eager, bit for bit
    second = model.run(batch2)        # one call per batch
    model.save("cnn.program")

with ops.Session() as s:
    served = s.load_program("cnn.program")
    served_second = served.run(batch2)   # equal to second
```

Each input array must have the shape the captured input had. `embedding` and `take` read an index array
from Python, so a composition that uses them cannot be captured.

## Import weights

`load_state_dict` loads weights trained in PyTorch or published as a Hugging Face safetensors
checkpoint into a `Sequential` model. Pass a PyTorch `state_dict()` (tensors on the CPU, or NumPy
arrays), or the path of a `.safetensors` file or of a checkpoint directory. Each layer with parameters
takes `<index>.weight` and `<index>.bias`, the names a `torch.nn.Sequential` of the same layers uses
(activations count as layers). Use `names={0: "fc1", 2: "fc2"}` and `prefix="model."` for tensors named
another way. Weights keep PyTorch's layouts: a `linear` weight is `(out_features, in_features)`, a
`conv2d` weight `(out_channels, in_channels, kernel_size, kernel_size)`. float16, bfloat16, float32 and
float64 tensors are converted to float32 on the device. A missing tensor, a tensor of the wrong shape,
and (unless `strict=False`) a tensor the model does not use raise an error that names it. After
loading, `predict` uses the weights, and the next `fit` starts from them.

In a session, `load_weights` writes named tensors into tensors you made with `tensor`. Add
`"transpose"` to load a PyTorch linear weight into the `(inputs, outputs)` weight `linear` takes.

<!-- block: import-weights -->
```python
import numpy as np
import mojolearn
from mojolearn import ops

# the state_dict of torch.nn.Sequential(Linear(4, 16), ReLU(), Linear(16, 2))
rng = np.random.default_rng(0)
state = {"0.weight": rng.standard_normal((16, 4), dtype=np.float32),
         "0.bias": np.zeros(16, dtype=np.float32),
         "2.weight": rng.standard_normal((2, 16), dtype=np.float32),
         "2.bias": np.zeros(2, dtype=np.float32)}
X_new = rng.standard_normal((32, 4), dtype=np.float32)

imported = mojolearn.Sequential().linear(16).relu().linear(2).load_state_dict(state)
logits = imported.predict(X_new)

# the same weights in a session
with ops.Session() as s:
    w1 = s.tensor(np.zeros((4, 16), dtype=np.float32))
    b1 = s.tensor(np.zeros((1, 16), dtype=np.float32))
    w2 = s.tensor(np.zeros((16, 2), dtype=np.float32))
    b2 = s.tensor(np.zeros((1, 2), dtype=np.float32))
    s.load_weights(state, [(w1, "0.weight", "transpose"), (b1, "0.bias"),
                           (w2, "2.weight", "transpose"), (b2, "2.bias")])
    hidden = s.linear(s.tensor(X_new), w1, b1, activation="relu")
    logits_session = s.numpy(s.linear(hidden, w2, b2))
```

## Import a model

`mojolearn.import_model` loads a whole trained model for inference: its operations and its weights.
Pass the path of an ONNX file (opset 13 or later), such as one written by `torch.onnx.export`, or a
`torch.export.ExportedProgram`. The import returns a `Program`: `run` takes new input arrays, runs every
operation of the model on the device in one call and returns the outputs, with the same results on
Apple, NVIDIA and AMD GPUs and on the CPU. The weights are copied to the device once, when the model is
imported. `save` writes the program and its weights, and `load_program` reads the file back on any
install, a GPU install or the CPU-only install.

A program runs arrays of one fixed shape. `shapes` sets every input axis the model leaves open, such as
the batch, by input name; `inputs` and `outputs` list each input and output as its name, shape and
dtype. float16, bfloat16 and float32 weights are used exactly; float64 weights are rounded to float32
once.

<!-- block: import -->
```python
import numpy as np
import mojolearn
from mojolearn import ops

# cnn.onnx: Conv(3 -> 4, 3 x 3, padding 1), Relu, MaxPool(2), Flatten, Gemm(64 -> 10),
# with the input named "x" and an open batch axis
model = mojolearn.import_model("cnn.onnx", shapes={"x": (8, 3, 8, 8)})
print(model.inputs)    # [('x', (8, 3, 8, 8), 'float32')]
print(model.outputs)   # [('y', (8, 10), 'float32')]

rng = np.random.default_rng(0)
images = rng.standard_normal((8, 3, 8, 8), dtype=np.float32)
logits = model.run(images)            # (8, 10)
model.save("cnn.program")

with ops.Session() as s:
    served = s.load_program("cnn.program")
    logits_served = served.run(images)  # equal to logits, bit for bit
```

An import supports the operations `mojolearn.importers.supported_ops()` lists, each with the attribute
values it accepts. A model that uses anything else raises `mojolearn.importers.UnsupportedModel`, which
names every node the import cannot run, so one attempt shows every change the model needs.
Imported models run inference only; to train, build the model with `Sequential` or a session and load
its weights as in [Import weights](#import-weights).

## Import a pretrained model

A pretrained PyTorch model imports the same way: put it in evaluation mode, export it with
`torch.export.export` and an example input of the shape you will run, and pass the exported program to
`mojolearn.import_model`. The example below imports torchvision's ResNet-18 with its published ImageNet
weights and classifies a batch of four images. PyTorch is only needed to export the model; the saved
program runs without it.

<!-- block: import-pretrained -->
```python
import numpy as np
import torch
import torchvision
import mojolearn

weights = torchvision.models.ResNet18_Weights.DEFAULT
net = torchvision.models.resnet18(weights=weights).eval()
example = torch.zeros(4, 3, 224, 224)
exported = torch.export.export(net, (example,))

model = mojolearn.import_model(exported)
print(model.inputs)    # [('x', (4, 3, 224, 224), 'float32')]

# four normalized 224 x 224 RGB images, as weights.transforms() prepares them
rng = np.random.default_rng(0)
images = rng.standard_normal((4, 3, 224, 224), dtype=np.float32)
logits = model.run(images)            # (4, 1000)
classes = logits.argmax(axis=1)
labels = [weights.meta["categories"][c] for c in classes]
model.save("resnet18.program")
```

The program takes inputs of the example's shape. To run another batch size, export with that size, or
mark the batch axis as dynamic with `torch.export.Dim` and set it with `shapes` at import.

## Numeric modes

Both entry points take `numeric_mode`: `Sequential(numeric_mode=...)` and
`ops.Session(numeric_mode=...)`. The default is `identical`. `fast` is available on Apple GPUs only.
See [Numeric modes](numeric_modes.md).
