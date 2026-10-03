# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Recurrent sequence models of the sequence expansion lane: RNN, LSTM and GRU
layers under a linear head, trained by backpropagation through time on the
GPU (`_mojolearn_x_sequence`) or on the CPU host binding, the same arithmetic
(`sequence/recurrent.mojo`).

The layers are PyTorch's (`nn.RNN`, `nn.LSTM`, `nn.GRU`, batch_first=True):
the same parameter names, shapes, gate orders and initialisation rule
(U(-1/sqrt(H), 1/sqrt(H)) for every weight and bias, the head as
`nn.Linear`), so `state_dict()` loads into a torch module and back. The head
reads the top layer's last hidden state. Regression minimises the mean squared
error, classification the softmax cross-entropy.

Refused by name (sequence/NOT_IMPLEMENTED.tsv): bidirectional, dropout
between layers, proj_size, initial hidden state, packed / variable-length
sequences, float64.
"""
from ._optional_numpy import require_numpy
np = require_numpy('_x_sequence_rnn')

from . import _backend
from ._labels import argmax_rows, unique_inverse

_BINDING = "_mojolearn_x_sequence"

CELLS = {"rnn_tanh": 0, "rnn_relu": 1, "lstm": 2, "gru": 3}
GATES = {0: 1, 1: 1, 2: 4, 3: 3}
OPTIMIZERS = {"sgd": 0, "adam": 1, "adamw": 2, "rmsprop": 3, "adagrad": 4, "lion": 7, "adamax": 8, "nadam": 9}
_OPT_DEFAULTS = {
    "sgd": dict(momentum=0.0, dampening=0.0, nesterov=False, weight_decay=0.0),
    "adam": dict(betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0),
    "adamw": dict(betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2),
    "rmsprop": dict(alpha=0.99, eps=1e-8, weight_decay=0.0, momentum=0.0, centered=False),
    "adagrad": dict(lr_decay=0.0, eps=1e-10, weight_decay=0.0, initial_accumulator_value=0.0),
    "lion": dict(betas=(0.9, 0.99), weight_decay=0.0),
    "adamax": dict(betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0),
    "nadam": dict(betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, momentum_decay=4e-3,
                  decoupled_weight_decay=False),
}


def binding(numeric_mode=None):
    return _backend.binding("_mojolearn_x_sequence", numeric_mode)


def optimizer_arguments(name, options):
    """(kind, flags, [f1, f2, eps, weight_decay, f7, initial_accumulator])
    for the binding's optimizer slots, PyTorch's defaults under `options`."""
    if name not in OPTIMIZERS:
        raise ValueError(f"optimizer must be one of {sorted(OPTIMIZERS)}, got {name!r}")
    o = dict(_OPT_DEFAULTS[name])
    unknown = set(options or {}) - set(o)
    if unknown:
        raise ValueError(f"optimizer {name!r} takes no option(s) {sorted(unknown)}; it takes {sorted(o)}")
    o.update(options or {})
    kind = OPTIMIZERS[name]
    if name == "sgd":
        if o["nesterov"] and (o["momentum"] <= 0 or o["dampening"] != 0):
            raise ValueError("sgd: nesterov needs momentum > 0 and dampening 0 (torch.optim.SGD)")
        return kind, int(bool(o["nesterov"])), [o["momentum"], 0.0, 0.0, o["weight_decay"], o["dampening"], 0.0]
    if name in ("adam", "adamw", "adamax"):
        b1, b2 = o["betas"]
        if not (0.0 <= b1 < 1.0 and 0.0 <= b2 < 1.0):
            raise ValueError(f"{name}: betas must lie in [0, 1)")
        return kind, 0, [b1, b2, o["eps"], o["weight_decay"], 0.0, 0.0]
    if name == "nadam":
        b1, b2 = o["betas"]
        if o["momentum_decay"] < 0:
            raise ValueError("nadam: momentum_decay must be >= 0")
        return kind, int(bool(o["decoupled_weight_decay"])), [b1, b2, o["eps"], o["weight_decay"],
                                                               o["momentum_decay"], 0.0]
    if name == "lion":
        b1, b2 = o["betas"]
        return kind, 0, [b1, b2, 0.0, o["weight_decay"], 0.0, 0.0]
    if name == "rmsprop":
        return kind, int(bool(o["centered"])), [0.0, o["alpha"], o["eps"], o["weight_decay"], o["momentum"], 0.0]
    return kind, 0, [o["lr_decay"], 0.0, o["eps"], o["weight_decay"], 0.0, o["initial_accumulator_value"]]


def _f32(a, name):
    a = np.asarray(a)
    if a.dtype == np.float64:
        raise TypeError(f"{name}: float64 is refused; pass float32 (no float64 on the device)")
    return np.ascontiguousarray(a, dtype=np.float32)


class _RecurrentBase:
    _CELL = "lstm"
    _TASK = 0

    def __init__(self, hidden_size=16, num_layers=1, optimizer="adam", learning_rate=1e-3,
                 batch_size=32, max_epochs=10, shuffle=True, random_state=0,
                 optimizer_options=None, nonlinearity="tanh", numeric_mode=None,
                 predict_chunk=4096, lr_schedule=None):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.shuffle = shuffle
        self.random_state = random_state
        self.optimizer_options = optimizer_options
        self.nonlinearity = nonlinearity
        self.numeric_mode = numeric_mode
        self.predict_chunk = predict_chunk
        self.lr_schedule = lr_schedule

    # ------------------------------------------------------------ shape
    def _cell(self):
        if self._CELL == "rnn":
            if self.nonlinearity not in ("tanh", "relu"):
                raise ValueError("nonlinearity must be 'tanh' or 'relu' (nn.RNN)")
            return CELLS["rnn_" + self.nonlinearity]
        return CELLS[self._CELL]

    def _ints(self):
        return [self._cell(), self.n_features_in_, int(self.hidden_size), int(self.num_layers), self.n_outputs_]

    def _layout(self):
        """[(torch name, shape)] in the binding's flat order."""
        G, H = GATES[self._cell()], int(self.hidden_size)
        out = []
        for l in range(int(self.num_layers)):  # glue: one entry per recurrent layer
            din = self.n_features_in_ if l == 0 else H
            out += [(f"weight_ih_l{l}", (G * H, din)), (f"weight_hh_l{l}", (G * H, H)),
                    (f"bias_ih_l{l}", (G * H,)), (f"bias_hh_l{l}", (G * H,))]
        return out + [("head.weight", (self.n_outputs_, H)), ("head.bias", (self.n_outputs_,))]

    def _init_params(self, rng):
        """Every parameter U(-1/sqrt(H), 1/sqrt(H)) in layout order, one flat
        float32 block drawn in Mojo from the seeded stream (`_buffer.InitStream`;
        lane cgr4-py-compute: it was numpy's Generator)."""
        k = 1.0 / float(np.sqrt(float(self.hidden_size)))  # glue: scalar init bound from hidden_size
        total = sum(int(np.prod(s)) for _, s in self._layout())  # glue: parameter tensor sizes of the layout
        out = np.empty(total, dtype=np.float32)
        rng.fill_uniform(out.ctypes.data, total, -k, k)
        return out

    def state_dict(self):
        """{torch name: array}: `nn.LSTM`/`nn.GRU`/`nn.RNN` names for the
        recurrent layers, `head.weight` and `head.bias` for the `nn.Linear`."""
        out, off = {}, 0
        for name, shape in self._layout():  # glue: one view per parameter tensor
            n = int(np.prod(shape))  # glue: product of the tensor shape dims
            out[name] = self.params_[off:off + n].reshape(shape).copy()
            off += n
        return out

    def load_state_dict(self, state):
        parts = []
        for name, shape in self._layout():  # glue: one check per parameter tensor
            a = _f32(state[name], name)
            if a.shape != tuple(shape):
                raise ValueError(f"{name}: shape {a.shape}, expected {tuple(shape)}")
            parts.append(a.ravel())
        self.params_ = np.ascontiguousarray(np.concatenate(parts), dtype=np.float32)  # glue: packs user weight tensors into the flat buffer
        return self

    # ------------------------------------------------------------ data
    def _check_X(self, X, fitting):
        X = _f32(X, "X")
        if X.ndim != 3:
            raise ValueError("X must be 3-D: (n_samples, n_timesteps, n_features)")
        if fitting:
            self.n_features_in_ = int(X.shape[2])
        elif X.shape[2] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[2]} features, the model was fitted on {self.n_features_in_}")
        return X

    def _schedule(self, n, rng):
        """(order, steps): every epoch's row order (int32, one permutation
        per epoch drawn in epoch order) and each step's (offset, count) pair
        into it (int32). The order was float32, which refused a schedule of
        2^24 rows or more (17 epochs at 1M rows); the step offsets bound it
        now, below 2^31 - 1."""
        bs = max(1, min(int(self.batch_size), n))
        epochs = int(self.max_epochs)
        if epochs * n >= 2 ** 31 - 1:
            raise ValueError(f"max_epochs * n_samples must be below 2^31 - 1, got {epochs} * {n}")
        # one 64-bit seed from the estimator's stream; every epoch's order and
        # every step's (offset, count) built in Mojo (`epoch_schedule`,
        # sequence/schedule.mojo; lane cgr4-py-compute)
        seed = rng.child_seed() if hasattr(rng, "child_seed") else int(rng.integers(0, 2 ** 63, dtype=np.int64))
        per = -(-n // bs)
        order = np.empty(max(epochs * n, 1), dtype=np.int32)
        steps = np.empty(max(2 * epochs * per, 2), dtype=np.int32)
        got = binding(self.numeric_mode).epoch_schedule(
            [order.ctypes.data, steps.ctypes.data],
            [n, epochs, bs, int(bool(self.shuffle)), seed & 0xFFFFFFFF, seed >> 32])
        return order[:epochs * n], steps[:2 * got]

    def _lrs(self, n_steps):
        """The learning rate of every optimizer step: `lr_schedule.lr_at(t)`
        (t one-based) when a schedule is set, else `learning_rate`."""
        if self.lr_schedule is not None:
            return np.asarray([self.lr_schedule.lr_at(t) for t in range(1, n_steps + 1)], dtype=np.float32)
        return np.full(n_steps, self.learning_rate, dtype=np.float32)

    def _fit(self, X, target):
        X = self._check_X(X, True)
        n, T = int(X.shape[0]), int(X.shape[1])
        if n < 1 or T < 1:
            raise ValueError("X needs at least one sample and one time step")
        from ._buffer import InitStream
        rng = InitStream(None if self.random_state is None else int(self.random_state))
        self.params_ = self._init_params(rng)
        order, steps = self._schedule(n, rng)
        n_steps = len(steps) // 2
        kind, flags, fp = optimizer_arguments(self.optimizer, self.optimizer_options)
        lrs = self._lrs(n_steps)
        losses = np.zeros(n_steps, dtype=np.float32)
        ip = self._ints() + [self._TASK, n, T, len(order), n_steps, kind, flags]
        b = binding(self.numeric_mode)
        b.rnn_fit([X.ctypes.data, target.ctypes.data, order.ctypes.data, steps.ctypes.data,
                   self.params_.ctypes.data, losses.ctypes.data, lrs.ctypes.data],
                  ip, [float(v) for v in fp])  # glue: the optimizer float arguments
        self.loss_curve_ = losses
        self.n_iter_ = n_steps
        self.n_timesteps_ = T
        return self

    def _run(self, X, want_seq=False):
        X = self._check_X(X, False)
        n, T = int(X.shape[0]), int(X.shape[1])
        out = np.zeros((n, self.n_outputs_), dtype=np.float32)
        seq = np.zeros((n, T, int(self.hidden_size)) if want_seq else (1,), dtype=np.float32)
        ip = self._ints() + [self._TASK, n, T, int(self.predict_chunk), int(want_seq)]
        binding(self.numeric_mode).rnn_predict(
            [X.ctypes.data, self.params_.ctypes.data, out.ctypes.data, seq.ctypes.data], ip)
        return out, seq

    def hidden_sequence(self, X):
        """The top layer's hidden states h_1..h_T, (n_samples, n_timesteps,
        hidden_size): `nn.LSTM(..., batch_first=True)(X)[0]`."""
        return self._run(X, want_seq=True)[1]


class _RecurrentRegressor(_RecurrentBase):
    _TASK = 0

    def fit(self, X, y):
        y = _f32(y, "y")
        self._y_1d = y.ndim == 1
        Y = y.reshape(len(y), -1)
        self.n_outputs_ = int(Y.shape[1])
        return self._fit(X, np.ascontiguousarray(Y))

    def predict(self, X):
        out = self._run(X)[0]
        return out[:, 0].copy() if self._y_1d else out


class _RecurrentClassifier(_RecurrentBase):
    _TASK = 1

    def fit(self, X, y):
        y = np.asarray(y)
        # sorted classes and inverse codes on the device (_labels.unique_inverse)
        cls, inv = unique_inverse(y)
        self.classes_ = np.asarray(cls).astype(y.dtype, copy=False) if y.dtype.kind in "biuf" else np.asarray(cls)
        inv = np.asarray(inv)
        if len(self.classes_) < 2:
            raise ValueError("classification needs at least two classes")
        self.n_outputs_ = int(len(self.classes_))
        return self._fit(X, np.ascontiguousarray(inv.astype(np.float32)))

    def predict_proba(self, X):
        return self._run(X)[0]

    def predict(self, X):
        # the first-max-wins row argmax in Mojo (`argmax_rows`)
        return self.classes_[np.asarray(argmax_rows(np.ascontiguousarray(self.predict_proba(X))))]


class LSTMRegressor(_RecurrentRegressor):
    """LSTM layers (`nn.LSTM`, batch_first) under a linear head, mean squared
    error, trained by BPTT on the GPU. X is (n_samples, n_timesteps,
    n_features)."""
    _CELL = "lstm"


class LSTMClassifier(_RecurrentClassifier):
    """LSTM layers (`nn.LSTM`, batch_first) under a linear head, softmax
    cross-entropy, trained by BPTT on the GPU."""
    _CELL = "lstm"


class GRURegressor(_RecurrentRegressor):
    """GRU layers (`nn.GRU`, batch_first; gates r, z, n and
    h' = n + z (h - n)) under a linear head, mean squared error."""
    _CELL = "gru"


class GRUClassifier(_RecurrentClassifier):
    """GRU layers (`nn.GRU`, batch_first) under a linear head, softmax
    cross-entropy."""
    _CELL = "gru"


class RNNRegressor(_RecurrentRegressor):
    """Elman RNN layers (`nn.RNN`, batch_first; nonlinearity 'tanh' or
    'relu') under a linear head, mean squared error."""
    _CELL = "rnn"


class RNNClassifier(_RecurrentClassifier):
    """Elman RNN layers (`nn.RNN`, batch_first) under a linear head, softmax
    cross-entropy."""
    _CELL = "rnn"
