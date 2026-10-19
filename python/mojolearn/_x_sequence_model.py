# SPDX-License-Identifier: Apache-2.0
"""`Sequential`: a model the user assembles layer by layer, trained by the
native autograd tape (`autograd/model.mojo`).

Python is the API shell only: it records the layers and options as a flat
integer program and a float table, checks the arguments, and makes one call
into Mojo per fit or prediction. Initialisation, shuffling, batching, the
forward and backward passes, the optimizer and the loss curve all run in
Mojo on the executor.
"""
from ._optional_numpy import require_numpy
np = require_numpy('_x_sequence_model')

from . import _backend
from ._labels import argmax_rows, threshold_codes, unique_inverse

# Description codes (autograd/model.mojo LAYER_*, LOSS_*, ...).
_SPEC_VERSION = 1
_LAYER = {"linear": 1, "relu": 2, "sigmoid": 3, "tanh": 4, "softmax": 5,
          "layer_norm": 6, "dropout": 7, "residual": 8}
_LOSS = {"mse": 0, "cross_entropy": 1, "bce_with_logits": 2, "huber": 3}
_TARGET_DENSE, _TARGET_CODES, _TARGET_CODE_VALUES = 0, 1, 2
_SCHED = {"constant": 0, "step": 1, "exponential": 2}
_OUT_RAW, _OUT_SOFTMAX, _OUT_SIGMOID, _OUT_PROBA2 = 0, 1, 2, 3
# sequence/ops.mojo OPT_*: "momentum" is SGD with a momentum buffer.
_OPT = {"sgd": 0, "momentum": 0, "adam": 1, "adamw": 2}
_PREDICT_CHUNK = 4096


def _f32_2d(X, name):
    # float64 input is cast to float32 once, natively, as every other
    # estimator's input is; the view keeps the converted Array alive
    from ._buffer import as_f32_c
    return np.asarray(as_f32_c(X, ndim=2, name=name)[0])


class Sequential:
    """A feed-forward model built layer by layer and trained on the GPU.

    Add layers with :meth:`add` (or the shortcuts :meth:`linear`,
    :meth:`relu`, ...), choose the loss and optimizer with :meth:`compile`,
    then call :meth:`fit`. Every computation of training and prediction runs
    natively; in ``identical`` mode the trained parameters and predictions
    are the same bits on Apple, NVIDIA and AMD GPUs for the same seed.

    Parameters
    ----------
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode. None uses the library default. ``'fast'`` is available
        on Apple GPUs only.

    Attributes
    ----------
    params_ : ndarray of float32
        All trained parameters as one flat vector, layers in order (a linear
        layer's weight ``(in, out)`` row-major, then its bias; a layer norm's
        weight, then its bias).
    loss_curve_ : list of float
        Mean training loss of every epoch.
    n_iter_ : int
        Number of epochs run.
    n_features_in_ : int
        Number of input features.
    classes_ : ndarray
        Sorted class labels (``cross_entropy`` and ``bce_with_logits`` only).

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((256, 4), dtype=np.float32)
    >>> y = X @ np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float32)
    >>> model = mojolearn.Sequential().linear(16).relu().linear(1)
    >>> model = model.compile(loss="mse", optimizer="adam", lr=0.01)
    >>> model = model.fit(X, y, epochs=20, batch_size=32, seed=0)
    >>> pred = model.predict(X)
    """

    def __init__(self, numeric_mode=None):
        self.numeric_mode = numeric_mode
        self._layers = []
        self._options = None

    # ------------------------------------------------------------ building
    def add(self, kind, *args, **kwargs):
        """Append a layer and return the model.

        ``kind`` is one of ``'linear'`` (``units``, ``bias=True``),
        ``'relu'``, ``'sigmoid'``, ``'tanh'``, ``'softmax'``,
        ``'layer_norm'`` (``eps=1e-5``), ``'dropout'`` (``p``) or
        ``'residual'`` (``source=-1``: add the output of layer ``source``,
        or the model input for -1, to the current output).
        """
        if not isinstance(kind, str) or kind.lower() not in _LAYER:
            raise ValueError(f"kind must be one of {sorted(_LAYER)}")
        kind = kind.lower()
        if kind == "linear":
            return self._linear(*args, **kwargs)
        if kind == "layer_norm":
            return self._layer_norm(*args, **kwargs)
        if kind == "dropout":
            return self._dropout(*args, **kwargs)
        if kind == "residual":
            return self._residual(*args, **kwargs)
        if args or kwargs:
            raise TypeError(f"{kind} takes no arguments")
        self._layers.append((_LAYER[kind], 0, 0, 0.0))
        return self

    def _linear(self, units, bias=True):
        if isinstance(units, bool) or not isinstance(units, (int, np.integer)) or units < 1:
            raise ValueError("units must be a positive integer")
        self._layers.append((_LAYER["linear"], int(units), 1 if bias else 0, 0.0))
        return self

    def _layer_norm(self, eps=1e-5):
        if not eps > 0:
            raise ValueError("eps must be positive")
        self._layers.append((_LAYER["layer_norm"], 0, 0, float(eps)))
        return self

    def _dropout(self, p):
        if not 0 <= p < 1:
            raise ValueError("p must be in [0, 1)")
        self._layers.append((_LAYER["dropout"], 0, 0, float(p)))
        return self

    def _residual(self, source=-1):
        if isinstance(source, bool) or not isinstance(source, (int, np.integer)):
            raise ValueError("source must be -1 or the index of an earlier layer")
        self._layers.append((_LAYER["residual"], int(source), 0, 0.0))
        return self

    def linear(self, units, bias=True):
        """Append a fully connected layer with ``units`` outputs."""
        return self._linear(units, bias)

    def relu(self):
        """Append a ReLU activation."""
        return self.add("relu")

    def sigmoid(self):
        """Append a logistic sigmoid activation."""
        return self.add("sigmoid")

    def tanh(self):
        """Append a hyperbolic tangent activation."""
        return self.add("tanh")

    def softmax(self):
        """Append a row-wise softmax."""
        return self.add("softmax")

    def layer_norm(self, eps=1e-5):
        """Append a layer normalization with a learned weight and bias."""
        return self._layer_norm(eps)

    def dropout(self, p):
        """Append dropout with drop probability ``p`` (active in fit only)."""
        return self._dropout(p)

    def residual(self, source=-1):
        """Add the output of layer ``source`` (-1: the model input) to the current output."""
        return self._residual(source)

    @property
    def layers(self):
        """The layers as ``(kind, arg0, arg1, float_arg)`` tuples, in order."""
        names = {v: k for k, v in _LAYER.items()}  # glue: inverts the layer name table
        return [(names[c], a0, a1, f) for c, a0, a1, f in self._layers]  # glue: describes the recorded layers

    def compile(self, loss="mse", optimizer="adam", lr=1e-3, *, momentum=0.9, nesterov=False,
                beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0, schedule="constant",
                step_size=1, gamma=0.1, huber_delta=1.0, tol=1e-4, n_iter_no_change=0):
        """Choose the loss, optimizer and learning-rate schedule; returns the model.

        Parameters
        ----------
        loss : {'mse', 'cross_entropy', 'bce_with_logits', 'huber'}, default 'mse'
            ``cross_entropy`` takes logits and integer class labels;
            ``bce_with_logits`` takes one logit per row and binary labels.
        optimizer : {'sgd', 'momentum', 'adam', 'adamw'}, default 'adam'
            ``'momentum'`` is SGD with ``momentum``.
        lr : float, default 1e-3
            Learning rate (> 0).
        momentum : float, default 0.9
            Momentum of ``'momentum'``.
        nesterov : bool, default False
            Nesterov momentum for ``'momentum'``.
        beta1, beta2, eps : float
            Adam and AdamW moment decays and denominator term.
        weight_decay : float, default 0.0
            L2 penalty (sgd, momentum, adam) or decoupled decay (adamw).
        schedule : {'constant', 'step', 'exponential'}, default 'constant'
            ``'step'`` multiplies the rate by ``gamma`` every ``step_size``
            epochs; ``'exponential'`` by ``gamma`` every epoch.
        huber_delta : float, default 1.0
            Threshold of the Huber loss.
        tol, n_iter_no_change : float, int
            Stop when the epoch loss has not improved by more than ``tol``
            for ``n_iter_no_change`` epochs; 0 trains every epoch.
        """
        if loss not in _LOSS:
            raise ValueError(f"loss must be one of {sorted(_LOSS)}")
        if optimizer not in _OPT:
            raise ValueError(f"optimizer must be one of {sorted(_OPT)}")
        if schedule not in _SCHED:
            raise ValueError(f"schedule must be one of {sorted(_SCHED)}")
        mu = float(momentum) if optimizer == "momentum" else 0.0
        if optimizer == "adam" or optimizer == "adamw":
            f1, f2 = float(beta1), float(beta2)
        else:
            f1, f2 = mu, 0.0
        flags = 1 if (optimizer == "momentum" and nesterov) else 0
        self._options = dict(loss=loss, opt=_OPT[optimizer], flags=flags, lr=float(lr), f1=f1, f2=f2,
                             eps=float(eps), weight_decay=float(weight_decay), sched=_SCHED[schedule],
                             step_size=int(step_size), gamma=float(gamma), huber_delta=float(huber_delta),
                             tol=float(tol), n_iter_no_change=int(n_iter_no_change))
        return self

    # ------------------------------------------------------------ description
    def _binding(self):
        return _backend.binding("_mojolearn_x_sequence", self.numeric_mode)

    def _program(self, n_features):
        prog = [_SPEC_VERSION, int(n_features), len(self._layers)]
        for code, a0, a1, _ in self._layers:  # glue: one description record per layer
            prog += [code, a0, a1, 0]
        return prog

    def _table(self):
        return [f for _, _, _, f in self._layers]  # glue: one float argument per layer

    # ------------------------------------------------------------ training
    def fit(self, X, y, *, epochs=10, batch_size=32, shuffle=True, seed=0):
        """Train on ``X`` (n_samples, n_features) and targets ``y``; returns the model.

        ``y`` is (n_samples,) or (n_samples, n_outputs) for ``mse`` and
        ``huber``, and 1-D labels for ``cross_entropy`` and
        ``bce_with_logits``. Parameters are initialized from ``seed``.
        """
        if not self._layers:
            raise ValueError("add at least one layer before fit")
        if self._options is None:
            raise ValueError("call compile before fit")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2 ** 64:
            raise ValueError("seed must be an integer in [0, 2**64)")
        if int(epochs) < 1 or int(batch_size) < 1:
            raise ValueError("epochs and batch_size must be >= 1")
        o = self._options
        X = _f32_2d(X, "X")
        n, d = X.shape
        prog = self._program(d)
        table = self._table()
        b = self._binding()
        n_params, n_out = (int(v) for v in b.model_n_params(prog, table))  # glue: two counts returned by Mojo
        Y, target = self._targets(y, n, n_out, o["loss"])
        params = np.zeros(n_params, dtype=np.float32)
        curve = np.zeros(int(epochs), dtype=np.float32)
        seed = int(seed)
        ip = [n, _LOSS[o["loss"]], target, o["opt"], o["flags"], o["sched"], o["step_size"],
              int(batch_size), int(epochs), int(bool(shuffle)), seed & 0xFFFFFFFF, seed >> 32,
              o["n_iter_no_change"], 1]
        fp = [o["lr"], o["f1"], o["f2"], o["eps"], o["weight_decay"], 0.0, o["tol"], o["gamma"],
              o["huber_delta"]] + table
        n_iter = int(b.model_fit([X.ctypes.data, Y.ctypes.data, params.ctypes.data, curve.ctypes.data],
                                 prog, ip, fp))
        self.params_ = params
        self.n_iter_ = n_iter
        self.loss_curve_ = curve[:n_iter].tolist()
        self.n_features_in_ = d
        self.n_outputs_ = n_out
        self._fit_loss = o["loss"]
        return self

    def _targets(self, y, n, n_out, loss):
        if loss in ("cross_entropy", "bce_with_logits"):
            y = np.asarray(y)
            if y.ndim != 1 or len(y) != n:
                raise ValueError(f"{loss} takes one label per row of X")
            # sorted classes and int32 codes on the device (_labels.unique_inverse)
            cls, inv = unique_inverse(y)
            self.classes_ = np.asarray(cls).astype(y.dtype, copy=False) if y.dtype.kind in "biuf" else np.asarray(cls)
            codes = np.ascontiguousarray(np.asarray(inv), dtype=np.int32)
            k = len(self.classes_)
            if loss == "cross_entropy":
                if k != n_out:
                    raise ValueError(f"the labels have {k} classes but the model has {n_out} outputs")
                return codes, _TARGET_CODES
            if k != 2 or n_out != 1:
                raise ValueError("bce_with_logits needs two classes and a model with one output")
            return codes, _TARGET_CODE_VALUES
        from ._buffer import as_f32_c
        y = np.asarray(as_f32_c(y, ndim=None, name="y")[0])
        if y.ndim not in (1, 2) or len(y) != n:
            raise ValueError("y must be (n_samples,) or (n_samples, n_outputs)")
        self._y_1d = y.ndim == 1
        Y = np.ascontiguousarray(y.reshape(n, -1))
        if Y.shape[1] != n_out:
            raise ValueError(f"y has {Y.shape[1]} outputs but the model has {n_out}")
        return Y, _TARGET_DENSE

    # ------------------------------------------------------------ prediction
    def _forward(self, X, transform, width=None):
        if not hasattr(self, "params_"):
            raise ValueError("call fit before predict")
        X = _f32_2d(X, "X")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, the model was fitted on {self.n_features_in_}")
        width = self.n_outputs_ if width is None else width
        out = np.zeros((X.shape[0], width), dtype=np.float32)
        self._binding().model_predict([X.ctypes.data, self.params_.ctypes.data, out.ctypes.data],
                                      self._program(self.n_features_in_), self._table(),
                                      [X.shape[0], _PREDICT_CHUNK, transform])
        return out

    def predict(self, X):
        """Model outputs; class labels for ``cross_entropy`` and ``bce_with_logits``."""
        loss = getattr(self, "_fit_loss", None)
        if loss == "cross_entropy":
            # the first-max-wins row argmax in Mojo (`argmax_rows`)
            return self.classes_[np.asarray(argmax_rows(self._forward(X, _OUT_RAW)))]
        if loss == "bce_with_logits":
            p = self._forward(X, _OUT_SIGMOID)
            # p > 0.5 per row in Mojo (`threshold_codes`; NaN takes class 0)
            return self.classes_[np.asarray(threshold_codes(np.ascontiguousarray(p[:, 0]), 0.5))]
        out = self._forward(X, _OUT_RAW)
        return out[:, 0].copy() if getattr(self, "_y_1d", False) else out

    def predict_proba(self, X):
        """Class probabilities (n_samples, n_classes), columns in ``classes_`` order."""
        loss = getattr(self, "_fit_loss", None)
        if loss == "cross_entropy":
            return self._forward(X, _OUT_SOFTMAX)
        if loss == "bce_with_logits":
            # [1 - p, p] written by the binding (`OP_PROBA2`)
            return self._forward(X, _OUT_PROBA2, width=2)
        raise ValueError("predict_proba needs a model fitted with cross_entropy or bce_with_logits")
