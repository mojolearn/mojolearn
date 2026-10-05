# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""MLPClassifier and MLPRegressor, scikit-learn's
(`sklearn/neural_network/_multilayer_perceptron.py`), trained on the GPU by
the sequence lane's binding (`sequence/mlp.mojo`, `sequence/mlp_fit.mojo`):
the same parameters, attributes, initialisation (Glorot uniform from
`random_state`, as the reference draws it), losses, Adam and SGD updates,
learning-rate schedules and stopping rule. Shuffling uses the lane's own
host generator seeded from `random_state`, so a shuffled fit is not the
reference's sample order.

Refused by name (sequence/NOT_IMPLEMENTED.tsv): solver='lbfgs',
early_stopping=True, warm_start, partial_fit, multilabel targets, verbose,
float64 input."""
from ._optional_numpy import require_numpy
np = require_numpy('_x_sequence_mlp')

from . import _backend
from ._buffer import _native
from ._labels import argmax_rows, threshold_codes, unique_inverse

_ACT = {"identity": 0, "logistic": 1, "tanh": 2, "relu": 3}
_SOLVER = {"adam": 0, "sgd": 1}
_LR = {"constant": 0, "invscaling": 1, "adaptive": 2}


class _BaseMLP:
    _LOSS = None

    def __init__(self, hidden_layer_sizes=(100,), activation="relu", *, solver="adam", alpha=0.0001,
                 batch_size="auto", learning_rate="constant", learning_rate_init=0.001, power_t=0.5,
                 max_iter=200, shuffle=True, random_state=None, tol=1e-4, verbose=False, warm_start=False,
                 momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1,
                 beta_1=0.9, beta_2=0.999, epsilon=1e-8, n_iter_no_change=10, max_fun=15000,
                 numeric_mode=None):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.power_t = power_t
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.tol = tol
        self.verbose = verbose
        self.warm_start = warm_start
        self.momentum = momentum
        self.nesterovs_momentum = nesterovs_momentum
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.n_iter_no_change = n_iter_no_change
        self.max_fun = max_fun
        self.numeric_mode = numeric_mode

    def _binding(self):
        return _backend.binding("_mojolearn_x_sequence", self.numeric_mode)

    def _hidden(self):
        h = self.hidden_layer_sizes
        h = [h] if isinstance(h, (int, np.integer)) else list(h)
        if any(int(v) < 1 for v in h):  # glue: hidden layer size arguments
            raise ValueError("hidden_layer_sizes must be > 0")
        return [int(v) for v in h]  # glue: hidden layer size arguments

    def _check(self):
        if self.solver == "lbfgs":
            raise NotImplementedError("solver='lbfgs' is not implemented (adam and sgd are)")
        if self.solver not in _SOLVER:
            raise ValueError(f"solver must be one of {sorted(_SOLVER)}")
        if self.activation not in _ACT:
            raise ValueError(f"activation must be one of {sorted(_ACT)}")
        if self.learning_rate not in _LR:
            raise ValueError(f"learning_rate must be one of {sorted(_LR)}")
        for name in ("early_stopping", "warm_start", "verbose"):  # glue: three named flag arguments
            if getattr(self, name):
                raise NotImplementedError(f"{name}=True is not implemented")
        if self.learning_rate_init <= 0 or self.max_iter < 1 or self.alpha < 0:
            raise ValueError("learning_rate_init > 0, max_iter >= 1 and alpha >= 0 are required")

    def _X(self, X):
        X = np.asarray(X)
        if X.dtype == np.float64:
            raise TypeError("float64 is refused; pass float32 (no float64 on the device)")
        X = np.ascontiguousarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("X must be 2-D")
        return X

    def _init(self, sizes, rng):
        coefs, intercepts = [], []
        for fi, fo in zip(sizes[:-1], sizes[1:]):  # glue: one pair per layer
            factor = 2.0 if self.activation == "logistic" else 6.0
            bound = np.sqrt(factor / (fi + fo))  # glue: one scalar init bound per layer
            # drawn in Mojo from the seeded stream (`_buffer.InitStream`)
            w = np.empty((fi, fo), dtype=np.float32)
            rng.fill_uniform(w.ctypes.data, w.size, -float(bound), float(bound))
            b = np.empty(fo, dtype=np.float32)
            rng.fill_uniform(b.ctypes.data, b.size, -float(bound), float(bound))
            coefs.append(w)
            intercepts.append(b)
        return coefs, intercepts

    def _pack(self):
        return np.ascontiguousarray(np.concatenate(  # glue: packs the per-layer weights into the flat buffer
            [np.concatenate([w.ravel(), b]) for w, b in zip(self.coefs_, self.intercepts_)]), dtype=np.float32)  # glue: packs the per-layer weights into the flat buffer

    def _unpack(self, flat):
        off = 0
        for i, (w, b) in enumerate(zip(self.coefs_, self.intercepts_)):  # glue: unpacks the flat buffer per layer
            self.coefs_[i] = flat[off:off + w.size].reshape(w.shape).copy()
            off += w.size
            self.intercepts_[i] = flat[off:off + b.size].copy()
            off += b.size

    def _fit(self, X, Y, out_act, n_out=None):
        """`n_out` (cpu2-l11-neural): `Y` is the (N,) int32 class codes and
        the binding builds the target on the device: the (N, n_out) one-hot
        for n_out > 1, the codes as the (N, 1) float target for n_out 1."""
        self._check()
        X = self._X(X)
        N, D = X.shape
        O = Y.shape[1] if n_out is None else int(n_out)
        hidden = self._hidden()
        sizes = [D] + hidden + [O]
        from ._buffer import InitStream
        rng = InitStream(None if self.random_state is None else int(self.random_state))
        self.coefs_, self.intercepts_ = self._init(sizes, rng)
        seed = rng.child_seed() % (2 ** 31 - 1)
        bs = min(200, N) if self.batch_size == "auto" else int(np.clip(self.batch_size, 1, N))  # glue: scalar batch size clip
        self.n_outputs_ = O
        self.n_layers_ = len(sizes)
        self.out_activation_ = ["identity", "logistic", "tanh", "relu", "softmax"][out_act]
        self.n_features_in_ = D
        params = self._pack()
        curve = np.zeros(int(self.max_iter), dtype=np.float32)
        ip = [N, D, O, _ACT[self.activation], out_act, self._LOSS_CODE, _SOLVER[self.solver],
              _LR[self.learning_rate], int(bool(self.nesterovs_momentum)), bs, int(self.max_iter),
              int(bool(self.shuffle)), seed, int(self.n_iter_no_change), len(hidden)] + hidden
        if n_out is not None:
            ip = ip + [1 if O > 1 else 2]
        fp = [float(self.learning_rate_init), float(self.beta_1), float(self.beta_2), float(self.epsilon),
              float(self.momentum), float(self.power_t), float(self.alpha), float(self.tol)]
        Y = np.ascontiguousarray(Y, dtype=np.float32 if n_out is None else np.int32)
        n_iter = int(self._binding().mlp_fit([X.ctypes.data, Y.ctypes.data, params.ctypes.data,
                                              curve.ctypes.data], ip, fp))
        self._unpack(params)
        self.n_iter_ = n_iter
        self.loss_curve_ = curve[:n_iter].tolist()
        self.loss_ = self.loss_curve_[-1]
        # the first-wins minimum of the curve in Mojo (`reduce_stat` 0)
        self.best_loss_ = float(_native("reduce_stat")(curve.ctypes.data, 0, n_iter, 0))
        self.t_ = n_iter * N
        return self

    def _forward(self, X, proba2=False):
        X = self._X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, the model was fitted on {self.n_features_in_}")
        hidden = [w.shape[1] for w in self.coefs_[:-1]]  # glue: hidden width per layer
        width = 2 if proba2 else self.n_outputs_
        out = np.zeros((X.shape[0], width), dtype=np.float32)
        params = self._pack()
        out_act = ["identity", "logistic", "tanh", "relu", "softmax"].index(self.out_activation_)
        ip = [X.shape[0], X.shape[1], self.n_outputs_, _ACT[self.activation], out_act, 4096, len(hidden)] + hidden
        if proba2:
            ip = ip + [1]
        written = int(self._binding().mlp_predict([X.ctypes.data, params.ctypes.data, out.ctypes.data], ip))
        if written != out.size:
            raise RuntimeError("mlp_predict: this binding predates the two-column probability; rebuild it")
        return out


class MLPRegressor(_BaseMLP):
    """sklearn MLPRegressor: squared loss (mean / 2) plus L2, identity
    output."""
    _LOSS_CODE = 0

    def fit(self, X, y):
        y = np.asarray(y)
        if y.dtype == np.float64:
            raise TypeError("float64 is refused; pass float32")
        self._y_1d = y.ndim == 1
        return self._fit(X, y.reshape(len(y), -1).astype(np.float32), 0)

    def predict(self, X):
        out = self._forward(X)
        return out[:, 0].copy() if self._y_1d else out


class MLPClassifier(_BaseMLP):
    """sklearn MLPClassifier: log loss plus L2; a logistic output for two
    classes, softmax beyond."""

    def fit(self, X, y):
        y = np.asarray(y)
        if y.ndim != 1:
            raise NotImplementedError("multilabel targets are not implemented")
        # sorted classes and inverse codes on the device (_labels.unique_inverse)
        cls, inv = unique_inverse(y)
        self.classes_ = np.asarray(cls).astype(y.dtype, copy=False) if y.dtype.kind in "biuf" else np.asarray(cls)
        inv = np.asarray(inv)
        k = len(self.classes_)
        if k < 2:
            raise ValueError("MLPClassifier needs at least two classes")
        # cpu2-l11-neural: the int32 codes go down as they are; the binding
        # builds the target on the device (`OP_ONE_HOT`).
        codes = np.ascontiguousarray(inv, dtype=np.int32)
        if k == 2:
            self._LOSS_CODE = 1
            return self._fit(X, codes, 1, n_out=1)
        self._LOSS_CODE = 2
        return self._fit(X, codes, 4, n_out=k)

    def predict_proba(self, X):
        if self.n_outputs_ == 1:
            # cpu2-l11-neural: [1 - p, p] written by the binding (`OP_PROBA2`)
            return self._forward(X, proba2=True)
        return self._forward(X)

    def predict(self, X):
        p = self._forward(X)
        if p.shape[1] == 1:
            # p > 0.5 per row in Mojo (`threshold_codes`; NaN takes class 0)
            return self.classes_[np.asarray(threshold_codes(np.ascontiguousarray(p[:, 0]), 0.5))]
        # the first-max-wins row argmax in Mojo (`argmax_rows`)
        return self.classes_[np.asarray(argmax_rows(p))]
