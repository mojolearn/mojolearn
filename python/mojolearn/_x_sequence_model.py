# SPDX-License-Identifier: Apache-2.0
"""A feed-forward model you assemble layer by layer and train on the GPU."""
# Native model descriptions and numerical execution retain the existing
# Sequential ABI. Each public operation delegates one native call.
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
        return _SEQUENTIAL_ENTRY(self, "construct", (numeric_mode, input_shape), _SEQUENTIAL_CONTEXT)

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
        return _SEQUENTIAL_ENTRY(self, "add", (kind, args, kwargs), _SEQUENTIAL_CONTEXT)

    def _append(self, code, a0=0, a1=0, f=0.0, args=None):
        # one layer: (code, arg0, arg1, float argument, the six integer
        # arguments of a version-2 description record)
        return _SEQUENTIAL_ENTRY(self, "append", (code, a0, a1, f, args), _SEQUENTIAL_CONTEXT)

    def _linear(self, units, bias=True):
        return _SEQUENTIAL_ENTRY(self, "add", ("linear", (units, bias), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def _layer_norm(self, eps=1e-5):
        return _SEQUENTIAL_ENTRY(self, "add", ("layer_norm", (eps,), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def _dropout(self, p):
        return _SEQUENTIAL_ENTRY(self, "add", ("dropout", (p,), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def _residual(self, source=-1):
        return _SEQUENTIAL_ENTRY(self, "add", ("residual", (source,), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def linear(self, units, bias=True):
        """Append a fully connected layer with ``units`` outputs."""
        return _SEQUENTIAL_ENTRY(self, "add", ("linear", (units, bias), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def relu(self):
        """Append a ReLU activation."""
        return _SEQUENTIAL_ENTRY(self, "add", ("relu", (), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def sigmoid(self):
        """Append a logistic sigmoid activation."""
        return _SEQUENTIAL_ENTRY(self, "add", ("sigmoid", (), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def tanh(self):
        """Append a hyperbolic tangent activation."""
        return _SEQUENTIAL_ENTRY(self, "add", ("tanh", (), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def softmax(self):
        """Append a row-wise softmax."""
        return _SEQUENTIAL_ENTRY(self, "add", ("softmax", (), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def layer_norm(self, eps=1e-5):
        """Append a layer normalization with a learned weight and bias."""
        return _SEQUENTIAL_ENTRY(self, "add", ("layer_norm", (eps,), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def dropout(self, p):
        """Append dropout with drop probability ``p`` (active in fit only)."""
        return _SEQUENTIAL_ENTRY(self, "add", ("dropout", (p,), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def residual(self, source=-1):
        """Add the output of layer ``source`` (-1: the model input) to the current output."""
        return _SEQUENTIAL_ENTRY(self, "add", ("residual", (source,), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def reshape(self, shape):
        """Append a reshape of each sample to ``shape`` (up to 6
        dimensions, one of them -1 to infer it)."""
        return _SEQUENTIAL_ENTRY(self, "add", ("reshape", (shape,), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def permute(self, axes):
        """Append a reordering of each sample's axes: output axis ``k`` is
        input axis ``axes[k]`` (axes count from 0 within a sample)."""
        return _SEQUENTIAL_ENTRY(self, "add", ("permute", (axes,), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def flatten(self):
        """Append a flattening of each sample into one vector."""
        return _SEQUENTIAL_ENTRY(self, "add", ("flatten", (), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def conv2d(self, out_channels, kernel_size, stride=1, padding=0, bias=True, groups=1):
        """Append a 2-D convolution of ``(channels, height, width)``
        samples with ``out_channels`` square ``kernel_size`` kernels,
        ``stride`` and zero ``padding`` on every side. With ``groups``
        greater than 1 the channels split into ``groups`` equal blocks and
        each block of outputs sees only its block of inputs (``groups``
        equal to the input channels is a depthwise convolution); the weight
        is ``(out_channels, channels // groups, kernel_size, kernel_size)``.

        Example::

            model.conv2d(32, 3, padding=1, groups=32)
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("conv2d", (out_channels, kernel_size, stride, padding, bias, groups), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def max_pool2d(self, kernel_size, stride=None):
        """Append a max pooling of ``(channels, height, width)`` samples
        over square ``kernel_size`` windows, ``stride`` apart (by default
        ``kernel_size``)."""
        return _SEQUENTIAL_ENTRY(self, "add", ("max_pool2d", (kernel_size, stride), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def avg_pool2d(self, kernel_size, stride=None):
        """Append an average pooling of ``(channels, height, width)``
        samples over square ``kernel_size`` windows, ``stride`` apart (by
        default ``kernel_size``)."""
        return _SEQUENTIAL_ENTRY(self, "add", ("avg_pool2d", (kernel_size, stride), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    @property
    def layers(self):
        """The layers in order: ``(kind, arg0, arg1, float_arg)`` tuples,
        and ``(kind, arguments, float_arg)`` for ``reshape``, ``permute``,
        ``flatten``, ``conv2d`` and the pools, with ``arguments`` the
        layer's six integer arguments."""
        return _SEQUENTIAL_ENTRY(self, "layers", (), _SEQUENTIAL_CONTEXT)

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
        return _SEQUENTIAL_ENTRY(self, "compile", (loss, optimizer, lr, momentum, nesterov, beta1, beta2, eps, weight_decay, schedule, step_size, gamma, huber_delta, tol, n_iter_no_change), _SEQUENTIAL_CONTEXT)

    # ------------------------------------------------------------ description
    def _binding(self):
        return _backend.binding("_mojolearn_x_sequence", self.numeric_mode)

    def _program(self, n_features, sample=None):
        return _SEQUENTIAL_ENTRY(self, "program", (n_features, sample), _SEQUENTIAL_CONTEXT)

    def _table(self):
        return _SEQUENTIAL_ENTRY(self, "table", (), _SEQUENTIAL_CONTEXT)

    def _sample_of(self, X):
        # the per-sample shape of a version-2 model (None: version 1)
        return _SEQUENTIAL_ENTRY(self, "sample", (X,), _SEQUENTIAL_CONTEXT)

    # ------------------------------------------------------------ training
    def fit(self, X, y, *, epochs=10, batch_size=None, shuffle=None, seed=0, resume=False,
            validation_data=None, restore_best_weights=False):
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
        validation_data : tuple of (X, y) or None, default None
            Held-out data evaluated after each epoch with dropout disabled.
            When provided, early stopping monitors its loss. Monitoring
            starts afresh for each fit call.
        restore_best_weights : bool, default False
            With validation data, restore the best epoch's parameters,
            optimizer and counters together. ``n_iter_`` is the retained
            epoch; ``epochs_run_`` counts epochs executed in this call.
            ``best_epoch_`` names the best absolute epoch and
            ``validation_loss_curve_`` keeps every validation loss from
            this call. Resuming continues from the retained epoch.

        Returns
        -------
        Sequential
            The fitted model, with ``params_``, ``loss_curve_``, ``n_iter_``
            and ``n_features_in_`` set (and ``classes_`` for classifiers).
        """
        return _SEQUENTIAL_ENTRY(self, "fit", (X, y, epochs, batch_size, shuffle, seed, resume, validation_data, restore_best_weights), _SEQUENTIAL_CONTEXT)

    # ------------------------------------------------------------ weight import
    def _param_layers(self):
        # (layer index, has bias) of every layer that holds parameters
        return _SEQUENTIAL_ENTRY(self, "parameter_layers", (), _SEQUENTIAL_CONTEXT)

    def load_state_dict(self, state, *, source=None, names=None, prefix="", strict=True):
        """Load trained weights from PyTorch or a safetensors checkpoint.

        Each layer with parameters takes the tensors ``<name>.weight`` and
        ``<name>.bias``, where ``<name>`` is the layer's index in the model
        (as in a ``torch.nn.Sequential`` of the same layers, activations
        included) with ``prefix`` in front. Weights use PyTorch's layouts:
        a ``linear`` weight is ``(out_features, in_features)``, a
        ``conv2d`` weight ``(out_channels, in_channels // groups,
        kernel_size, kernel_size)``, a ``layer_norm`` weight and bias ``(width,)``.
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
        return _SEQUENTIAL_ENTRY(self, "load_weights", (state, source, names, prefix, strict), _SEQUENTIAL_CONTEXT)

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
        return _SEQUENTIAL_ENTRY(self, "save", (path, export), _SEQUENTIAL_CONTEXT)

    def save_checkpoint(self, path):
        """Save the model and its full training state to ``path``.

        The file holds the layers, the loss, optimizer and schedule chosen
        with :meth:`compile`, the trained parameters, the optimizer state,
        the loss curve and the epoch and step counters of the row order and
        the optimizer. :meth:`from_checkpoint` restores all of it, and
        ``fit(..., resume=True)`` then continues training exactly where it
        stopped, on any install: a checkpoint written on one GPU vendor (or
        on the CPU) resumes to the same parameters on another.

        Beside it, ``<path>.verify.json`` records this install's report (the
        model's outputs on seeded inputs, by digest), for
        ``python -m mojolearn verify-model <path>`` on another install.

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
        return _SEQUENTIAL_ENTRY(self, "save", (path, False), _SEQUENTIAL_CONTEXT)

    def export(self, path):
        """Save the trained model for inference only.

        The file holds the layers and the trained parameters (and the
        classes of a classifier), without the optimizer state.
        :meth:`load` reads it back on any install, GPU or CPU, and
        :meth:`predict` and :meth:`predict_proba` give the same values
        there as here. Beside it, ``<path>.verify.json`` records this
        install's report, for ``python -m mojolearn verify-model <path>`` on
        another install.

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
        return _SEQUENTIAL_ENTRY(self, "save", (path, True), _SEQUENTIAL_CONTEXT)

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
        return _SEQUENTIAL_ENTRY(cls, "load_checkpoint", (path, numeric_mode), _SEQUENTIAL_CONTEXT)

    @classmethod
    def load(cls, path, numeric_mode=None):
        """Load a model saved with :meth:`export` (or :meth:`save_checkpoint`).

        The same as :meth:`from_checkpoint`.

        Examples
        --------
        >>> import mojolearn
        >>> model = mojolearn.Sequential.load("classifier.ckpt")  # doctest: +SKIP
        """
        return _SEQUENTIAL_ENTRY(cls, "load_checkpoint", (path, numeric_mode), _SEQUENTIAL_CONTEXT)

    def _targets(self, y, n, n_out, loss):
        return _SEQUENTIAL_ENTRY(self, "targets", (y, n, n_out, loss), _SEQUENTIAL_CONTEXT)

    # ------------------------------------------------------------ prediction
    def _forward(self, X, transform, width=None):
        return _SEQUENTIAL_ENTRY(self, "forward", (X, transform, width), _SEQUENTIAL_CONTEXT)

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
        return _SEQUENTIAL_ENTRY(self, "predict", (X,), _SEQUENTIAL_CONTEXT)

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
        return _SEQUENTIAL_ENTRY(self, "predict_proba", (X,), _SEQUENTIAL_CONTEXT)


# Package-import ABI connection. No model operation calls this setup again.
from .tokenizer import _binding as _sequential_tokenizer_binding
_SEQUENTIAL_ENTRY = _sequential_tokenizer_binding()._module.sequential_native_call
_SEQUENTIAL_NO_KEYWORDS = {}
_SEQUENTIAL_CONTEXT = {
    "integer": np.integer, "layer_codes": _LAYER, "loss_codes": _LOSS,
    "optimizer_codes": _OPT, "schedule_codes": _SCHED,
    "layer_names": ("", "linear", "relu", "sigmoid", "tanh", "softmax",
                    "layer_norm", "dropout", "residual", "reshape", "permute",
                    "flatten", "conv2d", "max_pool2d", "avg_pool2d"),
}


def _connect_sequential_bindings():
    # ABI connection at module import only; no Sequential operation calls it.
    from types import BuiltinFunctionType, ModuleType
    from collections import OrderedDict, defaultdict
    from pathlib import PosixPath, PurePosixPath
    import mmap
    import sys
    from array import array
    from ._array import Array
    from ._version import __version__
    selected = _backend.default_mode()
    entries = {}
    for mode in (selected, "fast" if selected == "identical" else "identical"):  # glue: connect native modes at package import
        try:
            module = _backend.binding("_mojolearn_x_sequence", mode)
            symbols = {}
            for name in ("x_sequence_vendor", "x_sequence_numeric_mode", "model_n_params", "model_fit", "model_predict", "model_labels", "model_gather_labels", "model_import_params", "model_save_checkpoint", "model_load_checkpoint", "model_checkpoint_info", "verify_digest", "verify_file_sha256", "verify_model_info", "verify_fill"):  # glue: resolve existing compiled ABI symbols at import
                entry = getattr(module, name)
                if not isinstance(entry, BuiltinFunctionType):
                    raise TypeError(f"Sequential requires a compiled {name}")
                symbols[name] = entry
            entries[mode] = symbols
        except (ImportError, AttributeError, ValueError, RuntimeError) as error:
            entries[mode] = type(error).__name__ + ": " + str(error)
    class_dtypes = {key: np.dtype(value) for key, value in _CLASS_DTYPE.items()}  # glue: connect checkpoint dtype classes once at package import
    numeric_dtypes = {("uifb"[kind], size): dtype for (kind, size), dtype in class_dtypes.items()}  # glue: connect native numeric dtype metadata once at package import
    buffer_codes = {("b", 1): "b", ("i", 1): "b", ("i", 2): "h", ("i", 4): "i", ("i", 8): "q", ("u", 1): "B", ("u", 2): "H", ("u", 4): "I", ("u", 8): "Q", ("f", 2): "H", ("f", 4): "f", ("f", 8): "d"}
    scalar_types = tuple(dtype.type for dtype in class_dtypes.values())  # glue: connect exact native scalar types once at package import
    integer_types = (int,) + tuple(dtype.type for (kind, _), dtype in class_dtypes.items() if kind in (0, 1))  # glue: connect exact native integer types once at package import
    array_fields = {name: vars(Array)[name].__get__ for name in ("_mv", "shape", "dtype", "order")}  # glue: retain native slot descriptors once at package import
    return {
        "sequential_class": Sequential, "module_type": ModuleType, "modules": sys.modules,
        "mapping_types": (dict, OrderedDict, defaultdict), "scalar_types": scalar_types, "integer_types": integer_types,
        "native_value_types": (type(None), str, bytes, bytearray, bool, int, float, range, np.ndarray, Array, array, PosixPath, PurePosixPath, type(iter([])), type(iter(())), type(iter(range(0))), type(reversed([]))),
        "buffer_owner_types": (type(None), bytes, bytearray, array, np.ndarray, mmap.mmap), "array_fields": array_fields,
        "model_fields": ("numeric_mode", "input_shape", "_layers", "_options", "params_", "n_features_in_", "n_outputs_", "_sample_shape", "n_iter_", "loss_curve_", "_fit_state", "_imported", "_fit_loss", "_target", "_fit_batch", "_fit_shuffle", "_y_1d", "classes_", "_opt_m", "_opt_v", "_curve", "validation_loss_curve_", "epochs_run_", "best_epoch_"),
        "numeric_dtypes": numeric_dtypes, "buffer_codes": buffer_codes, "scalar_type": np.generic,
        "version": __version__, "uname": os.uname, "class_dtypes": class_dtypes,
        "loss_names": ("mse", "cross_entropy", "bce_with_logits", "huber"),
        "bindings": entries, "backend_fields": vars(_backend), "vendor": _backend.vendor(),
        "ndarray": np.ndarray, "array_type": array, "array_class": Array,
        "float_types": (float, np.floating), "int_types": (int, np.integer),
        "bool_types": (bool, np.bool_), "int64_min": -(2 ** 63),
        "int64_max": 2 ** 63 - 1, "uint64_max": 2 ** 64 - 1,
        "float32": np.dtype("f4"), "float64": np.dtype("f8"),
        "int32": np.dtype("i4"), "int64": np.dtype("i8"),
        "uint64": np.dtype("u8"), "bool": np.dtype("?"),
    }


_SEQUENTIAL_CONTEXT.update(_connect_sequential_bindings())
_SEQUENTIAL_ENTRY(None, "initialize", (), _SEQUENTIAL_CONTEXT)
