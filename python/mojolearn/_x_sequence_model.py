# SPDX-License-Identifier: Apache-2.0
"""A feed-forward model you assemble layer by layer and train on the GPU."""
# Engineering note: `Sequential` is trained by the native autograd tape
# (autograd/model.mojo). Python is the API shell only: it records the layers
# and options as a flat integer program and a float table, checks the
# arguments, and makes one call into Mojo per fit or prediction.
# Initialisation, shuffling, batching, the forward and backward passes, the
# optimizer and the loss curve all run in Mojo on the executor.
import os

from ._optional_numpy import require_numpy
np = require_numpy('_x_sequence_model')
from . import _backend
from ._cast_in import as_input, describe, matrix_view

# Description codes (autograd/model.mojo LAYER_*, LOSS_*, ...).
_SPEC_VERSION = 1
# Version 2 (autograd/model.mojo SPEC2_*): a per-sample input shape of any
# rank and the shape layers; [2, ndim, d0 .. d7, layer count] then per layer
# (code, six arguments, 0).
_SPEC2_VERSION = 2
_SPEC2_DIMS = 8
_SPEC2_ARGS = 6
_LAYER = {"linear": 1, "relu": 2, "sigmoid": 3, "tanh": 4, "softmax": 5,
          "layer_norm": 6, "dropout": 7, "residual": 8,
          "reshape": 9, "permute": 10, "flatten": 11, "conv2d": 12, "max_pool2d": 13, "avg_pool2d": 14}
_SHAPE_LAYERS = frozenset((9, 10, 11, 12, 13, 14))
_LOSS = {"mse": 0, "cross_entropy": 1, "bce_with_logits": 2, "huber": 3}
_TARGET_DENSE, _TARGET_CODES, _TARGET_CODE_VALUES = 0, 1, 2
_SCHED = {"constant": 0, "step": 1, "exponential": 2}
_OUT_RAW, _OUT_SOFTMAX, _OUT_SIGMOID, _OUT_PROBA2, _OUT_ARGMAX, _OUT_THRESHOLD = 0, 1, 2, 3, 4, 5
# sequence/ops.mojo OPT_*: "momentum" is SGD with a momentum buffer.
_OPT = {"sgd": 0, "momentum": 0, "adam": 1, "adamw": 2}
_PREDICT_CHUNK = 4096
# autograd/model.mojo FIT_STATE_WORDS: the fit's resumable counters
_FIT_STATE_WORDS = 12
# core/cast_in.mojo CAST_KIND_* and item size -> the classes_ dtype of a
# loaded checkpoint
_CLASS_DTYPE = {(0, 1): "u1", (0, 2): "u2", (0, 4): "u4", (0, 8): "u8",
                (1, 1): "i1", (1, 2): "i2", (1, 4): "i4", (1, 8): "i8",
                (2, 2): "f2", (2, 4): "f4", (2, 8): "f8", (3, 1): "?"}
def _x_2d(X, name):
    # the user's array as it is and its descriptor: Mojo casts it to float32
    # on the executor (_cast_in.py); the caller keeps the array alive
    X = as_input(X)
    if X.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array of shape (n_samples, n_features)")
    return X, describe(X, name)


def _x_nd(X, name):
    # X of any rank >= 2: a 2-D X as it is; a higher-rank X (C order) as its
    # (n_samples, values per sample) view of the same memory. Returns the
    # array, its matrix view and the view's descriptor.
    X = as_input(X)
    if X.ndim < 2 or X.ndim > _SPEC2_DIMS:
        raise ValueError(f"{name} must have shape (n_samples, ...) with 2 to {_SPEC2_DIMS} dimensions")
    M = X if X.ndim == 2 else matrix_view(X, name, samples=True)
    return X, M, describe(M, name)


def _shape_arg(values, name, allow_infer=False):
    values = tuple(values)
    if not 1 <= len(values) <= _SPEC2_ARGS:
        raise ValueError(f"{name} must list 1 to {_SPEC2_ARGS} dimensions")
    out = []
    for v in values:  # glue: at most six shape arguments
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or not (v >= 1 or (allow_infer and v == -1)):
            raise ValueError(f"{name} must hold positive integers" + (" or -1" if allow_infer else ""))
        out.append(int(v))
    return out


def _positive(value, name, low=1):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < low:
        raise ValueError(f"{name} must be an integer >= {low}")
    return int(value)


class Sequential:
    """A feed-forward model built layer by layer and trained on the GPU.

    Add layers with :meth:`add` (or the shortcuts :meth:`linear`,
    :meth:`relu`, ...), choose the loss and optimizer with :meth:`compile`,
    then call :meth:`fit`. Parameter initialization, shuffling, batching, the
    forward and backward passes and the optimizer all run natively on the
    device.

    In `identical` mode, mojolearn returns the same bits on Apple, NVIDIA and AMD GPUs.
    For the same data and seed, the trained parameters and the predictions
    of a ``Sequential`` model are the same on every vendor.

    Parameters
    ----------
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode. None uses the library default. ``'fast'`` is available
        on Apple GPUs only.
    input_shape : tuple of int or None, default None
        The shape of one sample, for example ``(channels, height, width)``.
        None takes it from ``X`` at fit. With an input shape, ``fit`` and
        ``predict`` also accept a 2-D ``X`` whose rows hold each sample's
        values in row-major order.

    A ``linear`` layer acts on the last axis of each sample. The model's
    output must be one vector per sample: end a model on images or
    sequences with :meth:`flatten` and a ``linear`` layer.

    Attributes
    ----------
    params_ : ndarray of float32
        All trained parameters as one flat vector, layers in order (a linear
        layer's weight ``(in, out)`` row-major, then its bias; a layer norm's
        weight, then its bias; a conv2d layer's weight
        ``(out_channels, in_channels, kernel_size, kernel_size)`` row-major,
        then its bias).
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
    >>> y = (X[:, 0] - 2 * X[:, 1]) + (0.5 * X[:, 2] + 3 * X[:, 3])
    >>> model = mojolearn.Sequential().linear(16).relu().linear(1)
    >>> model = model.compile(loss="mse", optimizer="adam", lr=0.01)
    >>> model = model.fit(X, y, epochs=20, batch_size=32, seed=0)
    >>> pred = model.predict(X)

    A classifier with class probabilities:

    >>> labels = (X[:, 0] > X[:, 1]).astype(np.int64)
    >>> clf = mojolearn.Sequential().linear(16).relu().linear(2)
    >>> clf = clf.compile(loss="cross_entropy", optimizer="adam", lr=0.01)
    >>> clf = clf.fit(X, labels, epochs=20, batch_size=32, seed=0)
    >>> proba = clf.predict_proba(X)
    """

    def __init__(self, numeric_mode=None, *, input_shape=None):
        self.numeric_mode = numeric_mode
        self.input_shape = None if input_shape is None else tuple(_shape_arg(input_shape, "input_shape"))
        self._layers = []
        self._options = None

    # ------------------------------------------------------------ building
    def add(self, kind, *args, **kwargs):
        """Append a layer and return the model.

        Parameters
        ----------
        kind : str
            One of ``'linear'`` (``units``, ``bias=True``), ``'relu'``,
            ``'sigmoid'``, ``'tanh'``, ``'softmax'``, ``'layer_norm'``
            (``eps=1e-5``), ``'residual'`` (``source=-1``: add the output
            of layer ``source``, or the model input for -1, to the current
            output), ``'reshape'`` (``shape``), ``'permute'`` (``axes``),
            ``'flatten'``, ``'conv2d'`` (``out_channels``, ``kernel_size``,
            ``stride=1``, ``padding=0``, ``bias=True``), ``'max_pool2d'``
            or ``'avg_pool2d'`` (``kernel_size``, ``stride=None``).
        *args, **kwargs
            The layer's arguments, as listed for ``kind``.

        Returns
        -------
        Sequential
            The model, so calls chain.

        Examples
        --------
        >>> import mojolearn
        >>> model = mojolearn.Sequential().add("linear", 8).add("relu").add("linear", 1)
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
        if kind == "reshape":
            return self.reshape(*args, **kwargs)
        if kind == "permute":
            return self.permute(*args, **kwargs)
        if kind == "conv2d":
            return self.conv2d(*args, **kwargs)
        if kind == "max_pool2d":
            return self.max_pool2d(*args, **kwargs)
        if kind == "avg_pool2d":
            return self.avg_pool2d(*args, **kwargs)
        if args or kwargs:
            raise TypeError(f"{kind} takes no arguments")
        self._append(_LAYER[kind])
        return self

    def _append(self, code, a0=0, a1=0, f=0.0, args=None):
        # one layer: (code, arg0, arg1, float argument, the six integer
        # arguments of a version-2 description record)
        args = [a0, a1, 0, 0, 0, 0] if args is None else list(args)
        self._layers.append((code, a0, a1, f, tuple(args)))

    def _linear(self, units, bias=True):
        if isinstance(units, bool) or not isinstance(units, (int, np.integer)) or units < 1:
            raise ValueError("units must be a positive integer")
        self._append(_LAYER["linear"], int(units), 1 if bias else 0)
        return self

    def _layer_norm(self, eps=1e-5):
        if not eps > 0:
            raise ValueError("eps must be positive")
        self._append(_LAYER["layer_norm"], f=float(eps))
        return self

    def _dropout(self, p):
        if not 0 <= p < 1:
            raise ValueError("p must be in [0, 1)")
        self._append(_LAYER["dropout"], f=float(p))
        return self

    def _residual(self, source=-1):
        if isinstance(source, bool) or not isinstance(source, (int, np.integer)):
            raise ValueError("source must be -1 or the index of an earlier layer")
        self._append(_LAYER["residual"], int(source))
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

    def reshape(self, shape):
        """Append a reshape of each sample to ``shape`` (up to 6
        dimensions, one of them -1 to infer it)."""
        dims = _shape_arg(shape, "shape", allow_infer=True)
        self._append(_LAYER["reshape"], args=dims + [0] * (_SPEC2_ARGS - len(dims)))
        return self

    def permute(self, axes):
        """Append a reordering of each sample's axes: output axis ``k`` is
        input axis ``axes[k]`` (axes count from 0 within a sample)."""
        values = tuple(axes)
        if not 1 <= len(values) <= _SPEC2_ARGS:
            raise ValueError(f"axes must list 1 to {_SPEC2_ARGS} axes")
        order = [_positive(a, "axes", 0) for a in values]  # glue: at most six axes
        self._append(_LAYER["permute"], args=order + [-1] * (_SPEC2_ARGS - len(order)))
        return self

    def flatten(self):
        """Append a flattening of each sample into one vector."""
        self._append(_LAYER["flatten"])
        return self

    def conv2d(self, out_channels, kernel_size, stride=1, padding=0, bias=True):
        """Append a 2-D convolution of ``(channels, height, width)``
        samples with ``out_channels`` square ``kernel_size`` kernels,
        ``stride`` and zero ``padding`` on every side."""
        oc = _positive(out_channels, "out_channels")
        k = _positive(kernel_size, "kernel_size")
        st = _positive(stride, "stride")
        pad = _positive(padding, "padding", 0)
        self._append(_LAYER["conv2d"], oc, k, args=[oc, k, st, pad, 1 if bias else 0, 0])
        return self

    def max_pool2d(self, kernel_size, stride=None):
        """Append a max pooling of ``(channels, height, width)`` samples
        over square ``kernel_size`` windows, ``stride`` apart (by default
        ``kernel_size``)."""
        k = _positive(kernel_size, "kernel_size")
        st = k if stride is None else _positive(stride, "stride")
        self._append(_LAYER["max_pool2d"], k, st)
        return self

    def avg_pool2d(self, kernel_size, stride=None):
        """Append an average pooling of ``(channels, height, width)``
        samples over square ``kernel_size`` windows, ``stride`` apart (by
        default ``kernel_size``)."""
        k = _positive(kernel_size, "kernel_size")
        st = k if stride is None else _positive(stride, "stride")
        self._append(_LAYER["avg_pool2d"], k, st)
        return self

    @property
    def layers(self):
        """The layers in order: ``(kind, arg0, arg1, float_arg)`` tuples,
        and ``(kind, arguments, float_arg)`` for ``reshape``, ``permute``,
        ``flatten``, ``conv2d`` and the pools, with ``arguments`` the
        layer's six integer arguments."""
        names = {v: k for k, v in _LAYER.items()}  # glue: inverts the layer name table
        return [(names[c], args, f) if c in _SHAPE_LAYERS else (names[c], a0, a1, f)
                for c, a0, a1, f, args in self._layers]  # glue: describes the recorded layers

    def compile(self, loss="mse", optimizer="adam", lr=1e-3, *, momentum=0.9, nesterov=False,
                beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0, schedule="constant",
                step_size=1, gamma=0.1, huber_delta=1.0, tol=1e-4, n_iter_no_change=0):
        """Choose the loss, optimizer and learning-rate schedule.

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

        Returns
        -------
        Sequential
            The model.

        Examples
        --------
        >>> import mojolearn
        >>> model = mojolearn.Sequential().linear(8).relu().linear(1)
        >>> model = model.compile(loss="huber", optimizer="momentum", lr=0.01, nesterov=True)
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

    def _program(self, n_features, sample=None):
        if sample is None:
            prog = [_SPEC_VERSION, int(n_features), len(self._layers)]
            for code, a0, a1, _, _ in self._layers:  # glue: one description record per layer
                prog += [code, a0, a1, 0]
            return prog
        # version 2: the per-sample shape, padded with 1, then 8 words a layer
        prog = [_SPEC2_VERSION, len(sample)] + list(sample) + [1] * (_SPEC2_DIMS - len(sample))
        prog.append(len(self._layers))
        for code, _, _, _, args in self._layers:  # glue: one description record per layer
            prog += [code] + list(args) + [0]
        return prog

    def _table(self):
        return [f for _, _, _, f, _ in self._layers]  # glue: one float argument per layer

    def _sample_of(self, X):
        # the per-sample shape of a version-2 model (None: version 1)
        uses_shape = self.input_shape is not None or X.ndim > 2 or any(
            c in _SHAPE_LAYERS for c, _, _, _, _ in self._layers)  # glue: a layer-kind check
        if not uses_shape:
            return None
        if self.input_shape is None:
            if X.ndim - 1 > _SPEC2_DIMS - 1:
                raise ValueError(f"a sample has at most {_SPEC2_DIMS - 1} dimensions")
            return tuple(int(d) for d in X.shape[1:])  # glue: at most seven dimensions
        if X.ndim > 2 and tuple(X.shape[1:]) != self.input_shape:
            raise ValueError(f"X samples have shape {tuple(X.shape[1:])}, the model's input_shape is "
                             f"{self.input_shape}")
        return self.input_shape

    # ------------------------------------------------------------ training
    def fit(self, X, y, *, epochs=10, batch_size=None, shuffle=None, seed=0, resume=False):
        """Train the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples, ...)
            Training inputs: a bool, integer, float16, float32 or float64
            array, used as float32. A 2-D array may have any memory layout;
            an array of 3 or more dimensions (one sample per index of the
            first axis) must be C-contiguous.
        y : array-like
            Targets: (n_samples,) or (n_samples, n_outputs) for ``mse`` and
            ``huber``; one label per row for ``cross_entropy`` (any number
            of classes, one model output per class) and ``bce_with_logits``
            (two classes, one model output).
        epochs : int, default 10
            Passes over the data (with ``resume=True``, passes added to
            the ones already run).
        batch_size : int or None, default None
            Rows per optimizer step. None uses 32, or the previous fit's
            batch size when resuming.
        shuffle : bool or None, default None
            Visit the rows in a new random order every epoch. None uses
            True, or the previous fit's choice when resuming.
        seed : int, default 0
            Seed of the parameter initialization and the row order. A
            resumed fit keeps the seed of the fit it continues.
        resume : bool, default False
            Continue the previous fit (or the training state loaded with
            :meth:`from_checkpoint`) instead of starting over: the
            parameters, the optimizer state, the epoch and step counters
            and the row order carry on, so ``fit(epochs=2)`` followed by
            ``fit(epochs=2, resume=True)`` gives the same parameters as
            ``fit(epochs=4)``. Pass the same training data.

        Returns
        -------
        Sequential
            The fitted model, with ``params_``, ``loss_curve_``, ``n_iter_``
            and ``n_features_in_`` set (and ``classes_`` for classifiers).
        """
        if not self._layers:
            raise ValueError("add at least one layer before fit")
        if self._options is None:
            raise ValueError("call compile before fit")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2 ** 64:
            raise ValueError("seed must be an integer in [0, 2**64)")
        resume = bool(resume)
        if resume and getattr(self, "_fit_state", None) is None:
            raise ValueError("resume=True continues an earlier fit: call fit (or from_checkpoint) first")
        if batch_size is None:
            batch_size = self._fit_batch if resume else 32
        if shuffle is None:
            shuffle = self._fit_shuffle if resume else True
        if int(epochs) < 1 or int(batch_size) < 1:
            raise ValueError("epochs and batch_size must be >= 1")
        o = self._options
        X, M, xd = _x_nd(X, "X")
        n, d = M.shape
        if resume and d != self.n_features_in_:
            raise ValueError(f"X has {d} features, the model was fitted on {self.n_features_in_}")
        sample = self._sample_of(X)
        if resume and sample != getattr(self, "_sample_shape", None):
            raise ValueError("resume=True takes inputs of the shape the model was fitted on")
        # after load_state_dict the fit starts from the imported parameters
        warm = not resume and getattr(self, "_imported", False)
        if warm and (d != self.n_features_in_ or sample != getattr(self, "_sample_shape", None)):
            raise ValueError(f"X has {d} values per sample and shape {sample}; the imported weights take "
                             f"{self.n_features_in_} values per sample and shape {self._sample_shape}")
        prog = self._program(d, sample)
        table = self._table()
        b = self._binding()
        n_params, n_out = (int(v) for v in b.model_n_params(prog, table))  # glue: two counts returned by Mojo
        Y, yd, target = self._targets(y, n, n_out, o["loss"])
        if resume:
            # the training state continues in place; Mojo checks it belongs
            # to this model and optimizer
            params, m, v, state, prev = self.params_, self._opt_m, self._opt_v, self._fit_state, self._curve
            curve = np.zeros(self.n_iter_ + int(epochs), dtype=np.float32)
        else:
            params = self.params_ if warm else np.zeros(n_params, dtype=np.float32)
            m = np.zeros(n_params, dtype=np.float32)
            v = np.zeros(n_params, dtype=np.float32)
            state = np.zeros(_FIT_STATE_WORDS, dtype=np.int64)
            prev = None
            curve = np.zeros(int(epochs), dtype=np.float32)
        seed = int(seed)
        ip = [n, _LOSS[o["loss"]], target, o["opt"], o["flags"], o["sched"], o["step_size"],
              int(batch_size), int(epochs), int(bool(shuffle)), seed & 0xFFFFFFFF, seed >> 32,
              o["n_iter_no_change"], 0 if (resume or warm) else 1, 1 if resume else 0]
        fp = [o["lr"], o["f1"], o["f2"], o["eps"], o["weight_decay"], 0.0, o["tol"], o["gamma"],
              o["huber_delta"]] + table
        addrs = [xd, yd, params.ctypes.data, curve.ctypes.data, m.ctypes.data, v.ctypes.data,
                 state.ctypes.data, 0 if prev is None else prev.ctypes.data]
        # X, Y and the state arrays stay referenced until the call returns
        # (the descriptors and addresses point into them)
        n_iter = int(b.model_fit(addrs, prog, ip, fp))
        self.params_ = params
        self.n_iter_ = n_iter
        self.loss_curve_ = curve[:n_iter].tolist()
        self.n_features_in_ = d
        self._sample_shape = sample
        self.n_outputs_ = n_out
        self._fit_loss = o["loss"]
        self._target = target
        self._fit_batch = int(batch_size)
        self._fit_shuffle = bool(shuffle)
        self._opt_m, self._opt_v, self._fit_state, self._curve = m, v, state, curve
        self._imported = False
        return self

    # ------------------------------------------------------------ weight import
    def _param_layers(self):
        # (layer index, has bias) of every layer that holds parameters
        out = []
        for i, (code, a0, a1, _, args) in enumerate(self._layers):  # glue: one entry per parameter layer
            if code == _LAYER["linear"]:
                out.append((i, a1 == 1))
            elif code == _LAYER["layer_norm"]:
                out.append((i, True))
            elif code == _LAYER["conv2d"]:
                out.append((i, args[4] == 1))
        return out

    def load_state_dict(self, state, *, source=None, names=None, prefix="", strict=True):
        """Load trained weights from PyTorch or a safetensors checkpoint.

        Each layer with parameters takes the tensors ``<name>.weight`` and
        ``<name>.bias``, where ``<name>`` is the layer's index in the model
        (as in a ``torch.nn.Sequential`` of the same layers, activations
        included) with ``prefix`` in front. Weights use PyTorch's layouts:
        a ``linear`` weight is ``(out_features, in_features)``, a
        ``conv2d`` weight ``(out_channels, in_channels, kernel_size,
        kernel_size)``, a ``layer_norm`` weight and bias ``(width,)``.
        float16, bfloat16, float32 and float64 tensors are accepted and
        converted to float32 on the device (float16 and bfloat16 exactly,
        float64 rounded to the nearest float32).

        After loading, :meth:`predict` uses the weights, and the next
        :meth:`fit` (without ``resume``) starts from them.

        Parameters
        ----------
        state : mapping or str or path-like
            A mapping of names to tensors, such as ``module.state_dict()``
            (tensors on the CPU, or NumPy arrays), or the path of a
            ``.safetensors`` file or of a checkpoint directory
            (``model.safetensors``, or ``model.safetensors.index.json`` and
            its shards).
        source : {'torch', 'safetensors'} or None, default None
            ``'torch'`` for a mapping, ``'safetensors'`` for a path. None
            takes ``'safetensors'`` for a path and ``'torch'`` otherwise.
        names : dict or None, default None
            ``{layer index: name}`` for layers whose tensors are named
            differently, for example ``{0: "fc1", 2: "fc2"}``.
        prefix : str, default ''
            Put in front of every name, for example ``"model."``.
        strict : bool, default True
            Refuse a source that holds tensors the model does not use.

        Returns
        -------
        Sequential
            The model, with ``params_``, ``n_features_in_`` and
            ``n_outputs_`` set.

        Raises
        ------
        ValueError
            A tensor is missing, unused (with ``strict``) or of the wrong
            shape, or the model needs ``input_shape`` to size its layers.

        Examples
        --------
        >>> import numpy as np, mojolearn
        >>> rng = np.random.default_rng(0)
        >>> state = {"0.weight": rng.standard_normal((8, 4), dtype=np.float32),
        ...          "0.bias": np.zeros(8, dtype=np.float32),
        ...          "2.weight": rng.standard_normal((1, 8), dtype=np.float32),
        ...          "2.bias": np.zeros(1, dtype=np.float32)}
        >>> model = mojolearn.Sequential().linear(8).relu().linear(1)
        >>> model = model.load_state_dict(state)
        >>> pred = model.predict(rng.random((16, 4), dtype=np.float32))
        """
        from ._weight_import import LAYOUT_TORCH, SLOT_BIAS, SLOT_WEIGHT, WeightSource
        if not self._layers:
            raise ValueError("add at least one layer before load_state_dict")
        if not isinstance(prefix, str):
            raise TypeError("prefix must be a string")
        if names is not None and not hasattr(names, "get"):
            raise TypeError("names must be a dict of layer index to name")
        layers = self._param_layers()
        if not layers:
            raise ValueError("the model has no layers with parameters")
        entries = []  # (key, layer, slot)
        for i, bias in layers:  # glue: one name per layer parameter
            stem = prefix + str(i if names is None else names.get(i, i))
            entries.append((stem + ".weight", i, SLOT_WEIGHT))
            if bias:
                entries.append((stem + ".bias", i, SLOT_BIAS))
        keys = [k for k, _, _ in entries]  # glue: the requested tensor names
        with WeightSource(state, source) as src:
            missing = src.missing(keys)
            if missing:
                held = src.names()
                raise ValueError(f"load_state_dict: the source has no tensor {missing[0]!r}"
                                 + (f" ({len(missing)} missing)" if len(missing) > 1 else "")
                                 + f"; it holds {held[:6]}" + (" ..." if len(held) > 6 else ""))
            if strict:
                unused = src.unused(keys)
                if unused:
                    raise ValueError(f"load_state_dict: the model does not use {unused[:6]}"
                                     + (" ..." if len(unused) > 6 else "")
                                     + "; pass strict=False to skip them")
            sample, d = self._import_shape(src, layers, entries)
            prog = self._program(d, sample)
            table = self._table()
            b = self._binding()
            n_params, _ = (int(v) for v in b.model_n_params(prog, table))  # glue: two counts returned by Mojo
            recs = []
            for key, i, slot in entries:  # glue: one record per source tensor
                recs += src.record(key, i, slot, LAYOUT_TORCH)
            params = np.empty(n_params, dtype=np.float32)
            # Mojo checks every dtype and shape, then writes every parameter
            counts = b.model_import_params(prog, table, recs, keys, params.ctypes.data)
            n_params, n_out, n_in = (int(v) for v in counts)  # glue: three counts returned by Mojo
        self.params_ = params
        self.n_features_in_ = n_in
        self.n_outputs_ = n_out
        self._sample_shape = sample
        self.n_iter_ = 0
        self.loss_curve_ = []
        self._fit_state = None
        self._imported = True
        return self

    def _import_shape(self, src, layers, entries):
        # the per-sample shape (None: a version-1 description) and the input
        # width of the description load_state_dict passes to Mojo
        if hasattr(self, "params_") and hasattr(self, "n_features_in_"):
            return getattr(self, "_sample_shape", None), self.n_features_in_
        uses_shape = self.input_shape is not None or any(
            c in _SHAPE_LAYERS for c, _, _, _, _ in self._layers)  # glue: a layer-kind check
        if uses_shape:
            if self.input_shape is None:
                raise ValueError("load_state_dict: set input_shape on a model with reshape, permute, flatten, "
                                 "conv2d or pooling layers")
            return self.input_shape, 0
        # a 2-D model: the input width is the first parameter layer's
        # (the layers before it keep the width)
        first = layers[0][0]
        key = entries[0][0]
        shape = src.shape(key)
        if self._layers[first][0] == _LAYER["linear"]:
            if len(shape) != 2:
                raise ValueError(f"load_state_dict: {key!r} has shape {shape}; layer {first} (linear) takes a "
                                 "weight of shape (out_features, in_features)")
            return None, int(shape[1])
        if len(shape) != 1:
            raise ValueError(f"load_state_dict: {key!r} has shape {shape}; layer {first} (layer_norm) takes a "
                             "weight of shape (width,)")
        return None, int(shape[0])

    # ------------------------------------------------------------ checkpoints
    def _save(self, path, export):
        if not hasattr(self, "params_"):
            raise ValueError("call fit before saving the model")
        if not hasattr(self, "_target"):
            # weights from load_state_dict without a fit: no loss or training options to save
            raise ValueError("call fit before saving the model")
        if not export and getattr(self, "_fit_state", None) is None:
            raise ValueError("this model holds no training state (it was loaded from an export); use export")
        if getattr(self, "_sample_shape", None) is not None:
            # the checkpoint format holds version-1 descriptions only
            # (training/checkpoint.mojo); a shape-layer model is refused by name
            raise ValueError("save_checkpoint and export take models of 2-D inputs without reshape, "
                             "permute, flatten, conv2d or pooling layers")
        o = self._options
        classes = getattr(self, "classes_", None)
        k = 0 if classes is None else int(classes.shape[0])
        ip = [1 if export else 0, _LOSS[o["loss"]], self._target, o["opt"], o["flags"], o["sched"],
              o["step_size"], o["n_iter_no_change"], self._fit_batch, int(self._fit_shuffle),
              self.n_features_in_, self.n_outputs_, int(getattr(self, "_y_1d", False)),
              0 if export else self.n_iter_, k]
        fp = [o["lr"], o["f1"], o["f2"], o["eps"], o["weight_decay"], 0.0, o["tol"], o["gamma"],
              o["huber_delta"]] + self._table()
        if export:
            addrs = [self.params_.ctypes.data, 0, 0, 0, 0]
        else:
            addrs = [self.params_.ctypes.data, self._opt_m.ctypes.data, self._opt_v.ctypes.data,
                     self._fit_state.ctypes.data, self._curve.ctypes.data]
        desc = None
        if k:
            # classes_ as it is (Mojo refuses a non-contiguous one)
            desc = describe(classes, "classes")
        prog = self._program(self.n_features_in_)
        self._binding().model_save_checkpoint(os.fspath(path), addrs, prog, ip, fp, desc)  # cpu-route: checkpoint file written on the host

    def save_checkpoint(self, path):
        """Save the model and its full training state to ``path``.

        The file holds the layers, the loss, optimizer and schedule chosen
        with :meth:`compile`, the trained parameters, the optimizer state,
        the loss curve and the epoch and step counters of the row order and
        the optimizer. :meth:`from_checkpoint` restores all of it, and
        ``fit(..., resume=True)`` then continues training exactly where it
        stopped, on any install: a checkpoint written on one GPU vendor (or
        on the CPU) resumes to the same parameters on another.

        Parameters
        ----------
        path : str or path-like
            Destination file.

        Examples
        --------
        >>> import numpy as np, mojolearn
        >>> rng = np.random.default_rng(0)
        >>> X = rng.random((256, 4), dtype=np.float32)
        >>> y = X @ np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float32)
        >>> model = mojolearn.Sequential().linear(16).relu().linear(1)
        >>> model = model.compile(loss="mse", optimizer="adam", lr=0.01)
        >>> model = model.fit(X, y, epochs=2, batch_size=32, seed=0)
        >>> model.save_checkpoint("model.ckpt")  # doctest: +SKIP
        >>> again = mojolearn.Sequential.from_checkpoint("model.ckpt")  # doctest: +SKIP
        >>> again = again.fit(X, y, epochs=2, resume=True)  # doctest: +SKIP
        """
        self._save(path, export=False)

    def export(self, path):
        """Save the trained model for inference only.

        The file holds the layers and the trained parameters (and the
        classes of a classifier), without the optimizer state.
        :meth:`load` reads it back on any install, GPU or CPU, and
        :meth:`predict` and :meth:`predict_proba` give the same values
        there as here.

        Parameters
        ----------
        path : str or path-like
            Destination file.

        Examples
        --------
        >>> import numpy as np, mojolearn
        >>> rng = np.random.default_rng(0)
        >>> X = rng.random((256, 4), dtype=np.float32)
        >>> labels = (X[:, 0] > X[:, 1]).astype(np.int64)
        >>> clf = mojolearn.Sequential().linear(16).relu().linear(2)
        >>> clf = clf.compile(loss="cross_entropy", optimizer="adam", lr=0.01)
        >>> clf = clf.fit(X, labels, epochs=5, batch_size=32, seed=0)
        >>> clf.export("classifier.ckpt")  # doctest: +SKIP
        >>> served = mojolearn.Sequential.load("classifier.ckpt")  # doctest: +SKIP
        >>> proba = served.predict_proba(X)  # doctest: +SKIP
        """
        self._save(path, export=True)

    @classmethod
    def from_checkpoint(cls, path, numeric_mode=None):
        """Load a model saved with :meth:`save_checkpoint` or :meth:`export`.

        Parameters
        ----------
        path : str or path-like
            A file written by :meth:`save_checkpoint` or :meth:`export`.
        numeric_mode : {'identical', 'fast'} or None, default None
            Numeric mode of the loaded model. None uses the library default.
            ``'fast'`` is available on Apple GPUs only.

        Returns
        -------
        Sequential
            The model, ready for :meth:`predict`; from a
            :meth:`save_checkpoint` file also ready for
            ``fit(..., resume=True)``.

        Raises
        ------
        ValueError
            The file is not a complete model checkpoint (wrong format,
            truncated or damaged).
        """
        model = cls(numeric_mode=numeric_mode)
        b = model._binding()
        path = os.fspath(path)
        export, layers, ints, floats = b.model_checkpoint_info(path)  # cpu-route: checkpoint file read on the host
        for c, a0, a1, f in layers:  # glue: one description record per layer
            model._append(int(c), int(a0), int(a1), float(f))
        model._sample_shape = None
        (loss, target, opt, flags, sched, step_size, patience, batch, shuffle, d, n_out, y_1d,
         n_run, k, class_kind, class_size, n_params) = (int(v) for v in ints)  # glue: counts returned by Mojo
        lr, f1, f2, eps, wd, _, tol, gamma, huber = (float(v) for v in floats)  # glue: options returned by Mojo
        loss_name = {v: name for name, v in _LOSS.items()}[loss]  # glue: inverts the loss name table
        model._options = dict(loss=loss_name, opt=opt, flags=flags, lr=lr, f1=f1, f2=f2, eps=eps,
                              weight_decay=wd, sched=sched, step_size=step_size, gamma=gamma,
                              huber_delta=huber, tol=tol, n_iter_no_change=patience)
        params = np.zeros(n_params, dtype=np.float32)
        classes = None
        if k:
            classes = np.empty(k, dtype=np.dtype(_CLASS_DTYPE[(class_kind, class_size)]))
        addrs = [params.ctypes.data, 0, 0, 0, 0, 0 if classes is None else classes.ctypes.data]
        if not int(export):
            m = np.zeros(n_params, dtype=np.float32)
            v = np.zeros(n_params, dtype=np.float32)
            state = np.zeros(_FIT_STATE_WORDS, dtype=np.int64)
            curve = np.zeros(n_run, dtype=np.float32)
            addrs[1:5] = [m.ctypes.data, v.ctypes.data, state.ctypes.data, curve.ctypes.data]
        b.model_load_checkpoint(path, addrs)  # cpu-route: checkpoint file read on the host
        model.params_ = params
        model.n_features_in_ = d
        model.n_outputs_ = n_out
        model._fit_loss = loss_name
        model._target = target
        model._fit_batch = batch
        model._fit_shuffle = bool(shuffle)
        model._y_1d = bool(y_1d)
        if classes is not None:
            model.classes_ = classes
        if int(export):
            model.n_iter_ = 0
            model.loss_curve_ = []
            model._fit_state = None
        else:
            model.n_iter_ = n_run
            model.loss_curve_ = curve[:n_run].tolist()
            model._opt_m, model._opt_v, model._fit_state, model._curve = m, v, state, curve
        return model

    @classmethod
    def load(cls, path, numeric_mode=None):
        """Load a model saved with :meth:`export` (or :meth:`save_checkpoint`).

        The same as :meth:`from_checkpoint`.

        Examples
        --------
        >>> import mojolearn
        >>> model = mojolearn.Sequential.load("classifier.ckpt")  # doctest: +SKIP
        """
        return cls.from_checkpoint(path, numeric_mode=numeric_mode)

    def _targets(self, y, n, n_out, loss):
        if loss in ("cross_entropy", "bce_with_logits"):
            y = as_input(y)
            if y.ndim != 1 or len(y) != n:
                raise ValueError(f"{loss} takes one label per row of X")
            if y.dtype.kind not in "biuf":
                raise ValueError(f"{loss} takes numeric labels")
            # sorted classes (in y's own dtype) and int32 codes in Mojo, by
            # the sequence binding itself (`model_labels`: the device sort on
            # a GPU, its host twin on the CPU column), from y as it is
            classes = np.empty(n, dtype=y.dtype)
            codes = np.empty(n, dtype=np.int32)
            k = int(self._binding().model_labels(describe(y, "y"), classes.ctypes.data,
                                                 codes.ctypes.data))
            self.classes_ = classes[:k]
            if loss == "cross_entropy":
                if k != n_out:
                    raise ValueError(f"the labels have {k} classes but the model has {n_out} outputs")
                return codes, describe(codes, "y"), _TARGET_CODES
            if k != 2 or n_out != 1:
                raise ValueError("bce_with_logits needs two classes and a model with one output")
            return codes, describe(codes, "y"), _TARGET_CODE_VALUES
        y = as_input(y)
        if y.ndim not in (1, 2) or len(y) != n:
            raise ValueError("y must be (n_samples,) or (n_samples, n_outputs)")
        self._y_1d = y.ndim == 1
        width = 1 if y.ndim == 1 else y.shape[1]
        if width != n_out:
            raise ValueError(f"y has {width} outputs but the model has {n_out}")
        # y as it is; Mojo casts it to float32 on the executor
        return y, describe(y, "y"), _TARGET_DENSE

    # ------------------------------------------------------------ prediction
    def _forward(self, X, transform, width=None):
        if not hasattr(self, "params_"):
            raise ValueError("call fit before predict")
        sample = getattr(self, "_sample_shape", None)
        if sample is None:
            X, xd = _x_2d(X, "X")
            M = X
        else:
            X, M, xd = _x_nd(X, "X")
            if X.ndim > 2 and tuple(X.shape[1:]) != sample:
                raise ValueError(f"X samples have shape {tuple(X.shape[1:])}, the model was fitted on {sample}")
        if M.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {M.shape[1]} features, the model was fitted on {self.n_features_in_}")
        width = self.n_outputs_ if width is None else width
        out = np.zeros((M.shape[0], width), dtype=np.float32)
        self._binding().model_predict([xd, self.params_.ctypes.data, out.ctypes.data],
                                      self._program(self.n_features_in_, sample), self._table(),
                                      [M.shape[0], _PREDICT_CHUNK, transform])
        return out

    def predict(self, X):
        """Predict with the fitted model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Inputs: a bool, integer, float16, float32 or float64 array in
            any memory layout, used as float32.

        Returns
        -------
        ndarray
            The model outputs, (n_samples,) or (n_samples, n_outputs) in the
            shape ``y`` had at fit; for ``cross_entropy`` and
            ``bce_with_logits`` models, the predicted class labels
            (n_samples,).
        """
        loss = getattr(self, "_fit_loss", None)
        if loss == "cross_entropy" or loss == "bce_with_logits":
            # class codes from the binding: the first-max-wins row argmax of
            # the logits, or sigmoid p > 0.5 (a NaN takes class 0)
            mode = _OUT_ARGMAX if loss == "cross_entropy" else _OUT_THRESHOLD
            codes = self._forward(X, mode, width=1)
            # the labels by one gather over classes_ in Mojo (`model_gather_labels`;
            # classes_ is numeric: fit refuses non-numeric labels)
            classes = self.classes_
            labels = np.empty(codes.shape[0], dtype=classes.dtype)
            self._binding().model_gather_labels(
                [codes.ctypes.data, classes.ctypes.data, labels.ctypes.data],
                [codes.shape[0], classes.shape[0], classes.dtype.itemsize])
            return labels
        out = self._forward(X, _OUT_RAW)
        return out[:, 0].copy() if getattr(self, "_y_1d", False) else out

    def predict_proba(self, X):
        """Class probabilities of a ``cross_entropy`` or ``bce_with_logits`` model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Inputs: a bool, integer, float16, float32 or float64 array in
            any memory layout, used as float32.

        Returns
        -------
        ndarray of float32, shape (n_samples, n_classes)
            Each row sums to one; columns follow ``classes_``.
        """
        loss = getattr(self, "_fit_loss", None)
        if loss == "cross_entropy":
            return self._forward(X, _OUT_SOFTMAX)
        if loss == "bce_with_logits":
            # [1 - p, p] written by the binding (`OP_PROBA2`)
            return self._forward(X, _OUT_PROBA2, width=2)
        raise ValueError("predict_proba needs a model fitted with cross_entropy or bce_with_logits")
