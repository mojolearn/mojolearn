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
from ._mode import ClassifierScoreMixin, ParamsMixin, RegressorScoreMixin

from collections import OrderedDict
from . import _native_xseq_bridge
from ._native_full_reset_params import _INIT_ENTRY as _FULL_RESET_INIT, _CONTEXT as _FULL_RESET_CONTEXT

# Package-import connection: every public operation below is ONE call into
# the compiled owner (bindings/xseq_neural_native.mojo), which admits the
# inputs, encodes the labels, draws the weights, builds the schedule and
# optimizer arguments and runs `rnn_fit` / `rnn_predict`.
_XSEQ, _XSEQ_CONTEXT = _native_xseq_bridge.connect()

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



class _RecurrentBase(ParamsMixin):
    # The xseq public native lifecycle owns the former fit/validation/output wrappers.
    _mojolearn_native_public = True
    _CELL = "lstm"
    _TASK = 0

    def __init__(self, hidden_size=16, num_layers=1, optimizer="adam", learning_rate=1e-3,
                 batch_size=32, max_epochs=10, shuffle=True, random_state=0,
                 optimizer_options=None, nonlinearity="tanh", numeric_mode=None,
                 predict_chunk=4096, lr_schedule=None):
        # One native call: the full-reset owner replays this constructor's field
        # writes (bindings/full_reset_constructor_native.mojo). connect_module
        # installs the same entry as each concrete class's __init__.
        return _FULL_RESET_INIT(self, (), {"hidden_size": hidden_size, "num_layers": num_layers, "optimizer": optimizer, "learning_rate": learning_rate, "batch_size": batch_size, "max_epochs": max_epochs, "shuffle": shuffle, "random_state": random_state, "optimizer_options": optimizer_options, "nonlinearity": nonlinearity, "numeric_mode": numeric_mode, "predict_chunk": predict_chunk, "lr_schedule": lr_schedule}, _FULL_RESET_CONTEXT, type(self))

    # ------------------------------------------------------------ shape
    def _cell(self):
        if self._CELL == "rnn":
            if self.nonlinearity not in ("tanh", "relu"):
                raise ValueError("nonlinearity must be 'tanh' or 'relu' (nn.RNN)")
            return CELLS["rnn_" + self.nonlinearity]
        return CELLS[self._CELL]

    def state_dict(self):
        """{torch name: array}: `nn.LSTM`/`nn.GRU`/`nn.RNN` names for the
        recurrent layers, `head.weight` and `head.bias` for the `nn.Linear`."""
        return _XSEQ(self, "rnn_state_dict", (), _XSEQ_CONTEXT)

    def load_state_dict(self, state):
        """Load the named recurrent and head parameter arrays; return this model."""
        return _XSEQ(self, "rnn_load_state_dict", (state,), _XSEQ_CONTEXT)

    def hidden_sequence(self, X):
        """The top layer's hidden states h_1..h_T, (n_samples, n_timesteps,
        hidden_size): `nn.LSTM(..., batch_first=True)(X)[0]`."""
        return _XSEQ(self, "rnn_hidden_sequence", (X,), _XSEQ_CONTEXT)


class _RecurrentRegressor(RegressorScoreMixin, _RecurrentBase):
    # scikit-learn tags, read by `_mode.sklearn_tags`
    _estimator_type = "regressor"
    _TASK = 0

    def fit(self, X, y):
        return _XSEQ(self, "rnn_fit", (X, y), _XSEQ_CONTEXT)

    def predict(self, X):
        return _XSEQ(self, "rnn_predict", (X,), _XSEQ_CONTEXT)

    def score(self, X, y, sample_weight=None):
        """Coefficient of determination R^2 of ``predict(X)`` against ``y``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_timesteps, n_features)
            Test sequences.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True targets.
        sample_weight : array-like of shape (n_samples,), default None
            Sample weights.

        Returns
        -------
        float
            R^2, averaged uniformly over outputs; 1.0 is a perfect fit.
        """
        return _XSEQ(self, "rnn_score", (X, y, sample_weight), _XSEQ_CONTEXT)


class _RecurrentClassifier(ClassifierScoreMixin, _RecurrentBase):
    # scikit-learn tags, read by `_mode.sklearn_tags`
    _estimator_type = "classifier"
    _TASK = 1

    def fit(self, X, y):
        return _XSEQ(self, "rnn_fit", (X, y), _XSEQ_CONTEXT)

    def predict_proba(self, X):
        return _XSEQ(self, "rnn_predict_proba", (X,), _XSEQ_CONTEXT)

    def predict(self, X):
        return _XSEQ(self, "rnn_predict", (X,), _XSEQ_CONTEXT)

    def score(self, X, y, sample_weight=None):
        """Mean accuracy of ``predict(X)`` against ``y``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_timesteps, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels.
        sample_weight : array-like of shape (n_samples,), default None
            Sample weights.

        Returns
        -------
        float
            Fraction of samples predicted correctly (weighted when
            ``sample_weight`` is given).
        """
        return _XSEQ(self, "rnn_classifier_score", (X, y, sample_weight), _XSEQ_CONTEXT)


class LSTMRegressor(_RecurrentRegressor):
    """LSTM sequence regressor: recurrent layers under a linear head.

    The recurrent layers are PyTorch's `nn.LSTM` with ``batch_first=True``: the same
    parameter names, shapes, gate order and initialization (every weight and bias drawn
    from U(-1/sqrt(hidden_size), 1/sqrt(hidden_size)), the head as `nn.Linear`), so
    `state_dict()` loads into a torch module and back. A linear head reads the top
    layer's last hidden state, and training minimizes the mean squared error by backpropagation
    through time.

    `X` has shape (n_samples, n_timesteps, n_features) and must be float32 (float64
    raises `TypeError`). `y` is (n_samples,) or (n_samples, n_outputs).

    Not supported: bidirectional layers, dropout between layers, `proj_size`, an initial
    hidden state, and packed or variable-length sequences.

    Parameters
    ----------
    hidden_size : int, default 16
        Size of the hidden state of each recurrent layer.
    num_layers : int, default 1
        Number of stacked recurrent layers.
    optimizer : {'adam', 'adamw', 'sgd', 'rmsprop', 'adagrad', 'lion', 'adamax', 'nadam'}, default 'adam'
        Optimizer, with PyTorch's update rule and defaults.
    learning_rate : float, default 1e-3
        Learning rate, used when `lr_schedule` is None.
    batch_size : int, default 32
        Minibatch size (capped at n_samples).
    max_epochs : int, default 10
        Number of passes over the data. ``max_epochs * n_samples`` must be below 2**31 - 1.
    shuffle : bool, default True
        Reshuffle the rows every epoch.
    random_state : int or None, default 0
        Seed of the weight initialization and the shuffle.
    optimizer_options : dict or None, default None
        Extra optimizer arguments under their PyTorch names (for example
        ``{'betas': (0.9, 0.99), 'weight_decay': 0.01}``). Unknown names raise `ValueError`.
    nonlinearity : str, default 'tanh'
        Unused by this model; accepted for a common signature.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode. None uses the library default. ``'fast'`` is available on Apple GPUs only.
    predict_chunk : int, default 4096
        Rows processed per chunk at prediction time.
    lr_schedule : object or None, default None
        A built-in schedule (`ConstantLR`, `WarmupLinearLR`, `WarmupCosineLR`, `StepLR`,
        `ExponentialLR` or `OneCycleLR`); it sets the learning rate of every optimizer step.
        Any other object raises `TypeError`.

    Attributes
    ----------
    n_features_in_ : int
        Number of input features.
    n_outputs_ : int
        Number of head outputs.
    n_timesteps_ : int
        Sequence length seen in `fit`.
    n_iter_ : int
        Number of optimizer steps taken.
    loss_curve_ : ndarray
        Training loss at every optimizer step.
    params_ : ndarray
        All parameters as one flat float32 array.

    Methods
    -------
    fit(X, y)
        Train the model; returns self.
    predict(X)
        Predicted values.
    hidden_sequence(X)
        Top layer's hidden states, (n_samples, n_timesteps, hidden_size), as
        ``nn.LSTM(..., batch_first=True)(X)[0]``.
    state_dict()
        Parameters as a dict of arrays under PyTorch names.
    load_state_dict(state)
        Load parameters from such a dict; returns self.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((256, 20, 4), dtype=np.float32)
    >>> y = X[:, :, 0].sum(axis=1)
    >>> model = mojolearn.LSTMRegressor(hidden_size=32, max_epochs=5).fit(X, y)
    >>> pred = model.predict(X)
    """
    _CELL = "lstm"


class LSTMClassifier(_RecurrentClassifier):
    """LSTM sequence classifier: recurrent layers under a linear head.

    The recurrent layers are PyTorch's `nn.LSTM` with ``batch_first=True``: the same
    parameter names, shapes, gate order and initialization (every weight and bias drawn
    from U(-1/sqrt(hidden_size), 1/sqrt(hidden_size)), the head as `nn.Linear`), so
    `state_dict()` loads into a torch module and back. A linear head reads the top
    layer's last hidden state, and training minimizes the softmax cross-entropy by backpropagation
    through time.

    `X` has shape (n_samples, n_timesteps, n_features) and must be float32 (float64
    raises `TypeError`). `y` holds class labels (at least two classes).

    Not supported: bidirectional layers, dropout between layers, `proj_size`, an initial
    hidden state, and packed or variable-length sequences.

    Parameters
    ----------
    hidden_size : int, default 16
        Size of the hidden state of each recurrent layer.
    num_layers : int, default 1
        Number of stacked recurrent layers.
    optimizer : {'adam', 'adamw', 'sgd', 'rmsprop', 'adagrad', 'lion', 'adamax', 'nadam'}, default 'adam'
        Optimizer, with PyTorch's update rule and defaults.
    learning_rate : float, default 1e-3
        Learning rate, used when `lr_schedule` is None.
    batch_size : int, default 32
        Minibatch size (capped at n_samples).
    max_epochs : int, default 10
        Number of passes over the data. ``max_epochs * n_samples`` must be below 2**31 - 1.
    shuffle : bool, default True
        Reshuffle the rows every epoch.
    random_state : int or None, default 0
        Seed of the weight initialization and the shuffle.
    optimizer_options : dict or None, default None
        Extra optimizer arguments under their PyTorch names (for example
        ``{'betas': (0.9, 0.99), 'weight_decay': 0.01}``). Unknown names raise `ValueError`.
    nonlinearity : str, default 'tanh'
        Unused by this model; accepted for a common signature.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode. None uses the library default. ``'fast'`` is available on Apple GPUs only.
    predict_chunk : int, default 4096
        Rows processed per chunk at prediction time.
    lr_schedule : object or None, default None
        A built-in schedule (`ConstantLR`, `WarmupLinearLR`, `WarmupCosineLR`, `StepLR`,
        `ExponentialLR` or `OneCycleLR`); it sets the learning rate of every optimizer step.
        Any other object raises `TypeError`.

    Attributes
    ----------
    classes_ : ndarray
        Sorted class labels.
    n_features_in_ : int
        Number of input features.
    n_outputs_ : int
        Number of head outputs.
    n_timesteps_ : int
        Sequence length seen in `fit`.
    n_iter_ : int
        Number of optimizer steps taken.
    loss_curve_ : ndarray
        Training loss at every optimizer step.
    params_ : ndarray
        All parameters as one flat float32 array.

    Methods
    -------
    fit(X, y)
        Train the model; returns self.
    predict(X)
        Predicted class labels.
    predict_proba(X)
        Class probabilities, (n_samples, n_classes).
    hidden_sequence(X)
        Top layer's hidden states, (n_samples, n_timesteps, hidden_size), as
        ``nn.LSTM(..., batch_first=True)(X)[0]``.
    state_dict()
        Parameters as a dict of arrays under PyTorch names.
    load_state_dict(state)
        Load parameters from such a dict; returns self.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((256, 20, 4), dtype=np.float32)
    >>> y = (X[:, -1, 0] > 0.5).astype(np.int64)
    >>> model = mojolearn.LSTMClassifier(hidden_size=32, max_epochs=5).fit(X, y)
    >>> pred = model.predict(X)
    """
    _CELL = "lstm"


class GRURegressor(_RecurrentRegressor):
    """GRU sequence regressor: recurrent layers under a linear head.

    The GRU gates are r, z, n with ``h' = n + z * (h - n)``.
    The recurrent layers are PyTorch's `nn.GRU` with ``batch_first=True``: the same
    parameter names, shapes, gate order and initialization (every weight and bias drawn
    from U(-1/sqrt(hidden_size), 1/sqrt(hidden_size)), the head as `nn.Linear`), so
    `state_dict()` loads into a torch module and back. A linear head reads the top
    layer's last hidden state, and training minimizes the mean squared error by backpropagation
    through time.

    `X` has shape (n_samples, n_timesteps, n_features) and must be float32 (float64
    raises `TypeError`). `y` is (n_samples,) or (n_samples, n_outputs).

    Not supported: bidirectional layers, dropout between layers, `proj_size`, an initial
    hidden state, and packed or variable-length sequences.

    Parameters
    ----------
    hidden_size : int, default 16
        Size of the hidden state of each recurrent layer.
    num_layers : int, default 1
        Number of stacked recurrent layers.
    optimizer : {'adam', 'adamw', 'sgd', 'rmsprop', 'adagrad', 'lion', 'adamax', 'nadam'}, default 'adam'
        Optimizer, with PyTorch's update rule and defaults.
    learning_rate : float, default 1e-3
        Learning rate, used when `lr_schedule` is None.
    batch_size : int, default 32
        Minibatch size (capped at n_samples).
    max_epochs : int, default 10
        Number of passes over the data. ``max_epochs * n_samples`` must be below 2**31 - 1.
    shuffle : bool, default True
        Reshuffle the rows every epoch.
    random_state : int or None, default 0
        Seed of the weight initialization and the shuffle.
    optimizer_options : dict or None, default None
        Extra optimizer arguments under their PyTorch names (for example
        ``{'betas': (0.9, 0.99), 'weight_decay': 0.01}``). Unknown names raise `ValueError`.
    nonlinearity : str, default 'tanh'
        Unused by this model; accepted for a common signature.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode. None uses the library default. ``'fast'`` is available on Apple GPUs only.
    predict_chunk : int, default 4096
        Rows processed per chunk at prediction time.
    lr_schedule : object or None, default None
        A built-in schedule (`ConstantLR`, `WarmupLinearLR`, `WarmupCosineLR`, `StepLR`,
        `ExponentialLR` or `OneCycleLR`); it sets the learning rate of every optimizer step.
        Any other object raises `TypeError`.

    Attributes
    ----------
    n_features_in_ : int
        Number of input features.
    n_outputs_ : int
        Number of head outputs.
    n_timesteps_ : int
        Sequence length seen in `fit`.
    n_iter_ : int
        Number of optimizer steps taken.
    loss_curve_ : ndarray
        Training loss at every optimizer step.
    params_ : ndarray
        All parameters as one flat float32 array.

    Methods
    -------
    fit(X, y)
        Train the model; returns self.
    predict(X)
        Predicted values.
    hidden_sequence(X)
        Top layer's hidden states, (n_samples, n_timesteps, hidden_size), as
        ``nn.GRU(..., batch_first=True)(X)[0]``.
    state_dict()
        Parameters as a dict of arrays under PyTorch names.
    load_state_dict(state)
        Load parameters from such a dict; returns self.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((256, 20, 4), dtype=np.float32)
    >>> y = X[:, :, 0].sum(axis=1)
    >>> model = mojolearn.GRURegressor(hidden_size=32, max_epochs=5).fit(X, y)
    >>> pred = model.predict(X)
    """
    _CELL = "gru"


class GRUClassifier(_RecurrentClassifier):
    """GRU sequence classifier: recurrent layers under a linear head.

    The recurrent layers are PyTorch's `nn.GRU` with ``batch_first=True``: the same
    parameter names, shapes, gate order and initialization (every weight and bias drawn
    from U(-1/sqrt(hidden_size), 1/sqrt(hidden_size)), the head as `nn.Linear`), so
    `state_dict()` loads into a torch module and back. A linear head reads the top
    layer's last hidden state, and training minimizes the softmax cross-entropy by backpropagation
    through time.

    `X` has shape (n_samples, n_timesteps, n_features) and must be float32 (float64
    raises `TypeError`). `y` holds class labels (at least two classes).

    Not supported: bidirectional layers, dropout between layers, `proj_size`, an initial
    hidden state, and packed or variable-length sequences.

    Parameters
    ----------
    hidden_size : int, default 16
        Size of the hidden state of each recurrent layer.
    num_layers : int, default 1
        Number of stacked recurrent layers.
    optimizer : {'adam', 'adamw', 'sgd', 'rmsprop', 'adagrad', 'lion', 'adamax', 'nadam'}, default 'adam'
        Optimizer, with PyTorch's update rule and defaults.
    learning_rate : float, default 1e-3
        Learning rate, used when `lr_schedule` is None.
    batch_size : int, default 32
        Minibatch size (capped at n_samples).
    max_epochs : int, default 10
        Number of passes over the data. ``max_epochs * n_samples`` must be below 2**31 - 1.
    shuffle : bool, default True
        Reshuffle the rows every epoch.
    random_state : int or None, default 0
        Seed of the weight initialization and the shuffle.
    optimizer_options : dict or None, default None
        Extra optimizer arguments under their PyTorch names (for example
        ``{'betas': (0.9, 0.99), 'weight_decay': 0.01}``). Unknown names raise `ValueError`.
    nonlinearity : str, default 'tanh'
        Unused by this model; accepted for a common signature.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode. None uses the library default. ``'fast'`` is available on Apple GPUs only.
    predict_chunk : int, default 4096
        Rows processed per chunk at prediction time.
    lr_schedule : object or None, default None
        A built-in schedule (`ConstantLR`, `WarmupLinearLR`, `WarmupCosineLR`, `StepLR`,
        `ExponentialLR` or `OneCycleLR`); it sets the learning rate of every optimizer step.
        Any other object raises `TypeError`.

    Attributes
    ----------
    classes_ : ndarray
        Sorted class labels.
    n_features_in_ : int
        Number of input features.
    n_outputs_ : int
        Number of head outputs.
    n_timesteps_ : int
        Sequence length seen in `fit`.
    n_iter_ : int
        Number of optimizer steps taken.
    loss_curve_ : ndarray
        Training loss at every optimizer step.
    params_ : ndarray
        All parameters as one flat float32 array.

    Methods
    -------
    fit(X, y)
        Train the model; returns self.
    predict(X)
        Predicted class labels.
    predict_proba(X)
        Class probabilities, (n_samples, n_classes).
    hidden_sequence(X)
        Top layer's hidden states, (n_samples, n_timesteps, hidden_size), as
        ``nn.GRU(..., batch_first=True)(X)[0]``.
    state_dict()
        Parameters as a dict of arrays under PyTorch names.
    load_state_dict(state)
        Load parameters from such a dict; returns self.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((256, 20, 4), dtype=np.float32)
    >>> y = (X[:, -1, 0] > 0.5).astype(np.int64)
    >>> model = mojolearn.GRUClassifier(hidden_size=32, max_epochs=5).fit(X, y)
    >>> pred = model.predict(X)
    """
    _CELL = "gru"


class RNNRegressor(_RecurrentRegressor):
    """Elman RNN sequence regressor: recurrent layers under a linear head.

    The nonlinearity is 'tanh' or 'relu'.
    The recurrent layers are PyTorch's `nn.RNN` with ``batch_first=True``: the same
    parameter names, shapes, gate order and initialization (every weight and bias drawn
    from U(-1/sqrt(hidden_size), 1/sqrt(hidden_size)), the head as `nn.Linear`), so
    `state_dict()` loads into a torch module and back. A linear head reads the top
    layer's last hidden state, and training minimizes the mean squared error by backpropagation
    through time.

    `X` has shape (n_samples, n_timesteps, n_features) and must be float32 (float64
    raises `TypeError`). `y` is (n_samples,) or (n_samples, n_outputs).

    Not supported: bidirectional layers, dropout between layers, `proj_size`, an initial
    hidden state, and packed or variable-length sequences.

    Parameters
    ----------
    hidden_size : int, default 16
        Size of the hidden state of each recurrent layer.
    num_layers : int, default 1
        Number of stacked recurrent layers.
    optimizer : {'adam', 'adamw', 'sgd', 'rmsprop', 'adagrad', 'lion', 'adamax', 'nadam'}, default 'adam'
        Optimizer, with PyTorch's update rule and defaults.
    learning_rate : float, default 1e-3
        Learning rate, used when `lr_schedule` is None.
    batch_size : int, default 32
        Minibatch size (capped at n_samples).
    max_epochs : int, default 10
        Number of passes over the data. ``max_epochs * n_samples`` must be below 2**31 - 1.
    shuffle : bool, default True
        Reshuffle the rows every epoch.
    random_state : int or None, default 0
        Seed of the weight initialization and the shuffle.
    optimizer_options : dict or None, default None
        Extra optimizer arguments under their PyTorch names (for example
        ``{'betas': (0.9, 0.99), 'weight_decay': 0.01}``). Unknown names raise `ValueError`.
    nonlinearity : {'tanh', 'relu'}, default 'tanh'
        Activation of the recurrence, as `nn.RNN`.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode. None uses the library default. ``'fast'`` is available on Apple GPUs only.
    predict_chunk : int, default 4096
        Rows processed per chunk at prediction time.
    lr_schedule : object or None, default None
        A built-in schedule (`ConstantLR`, `WarmupLinearLR`, `WarmupCosineLR`, `StepLR`,
        `ExponentialLR` or `OneCycleLR`); it sets the learning rate of every optimizer step.
        Any other object raises `TypeError`.

    Attributes
    ----------
    n_features_in_ : int
        Number of input features.
    n_outputs_ : int
        Number of head outputs.
    n_timesteps_ : int
        Sequence length seen in `fit`.
    n_iter_ : int
        Number of optimizer steps taken.
    loss_curve_ : ndarray
        Training loss at every optimizer step.
    params_ : ndarray
        All parameters as one flat float32 array.

    Methods
    -------
    fit(X, y)
        Train the model; returns self.
    predict(X)
        Predicted values.
    hidden_sequence(X)
        Top layer's hidden states, (n_samples, n_timesteps, hidden_size), as
        ``nn.RNN(..., batch_first=True)(X)[0]``.
    state_dict()
        Parameters as a dict of arrays under PyTorch names.
    load_state_dict(state)
        Load parameters from such a dict; returns self.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((256, 20, 4), dtype=np.float32)
    >>> y = X[:, :, 0].sum(axis=1)
    >>> model = mojolearn.RNNRegressor(hidden_size=32, max_epochs=5).fit(X, y)
    >>> pred = model.predict(X)
    """
    _CELL = "rnn"


class RNNClassifier(_RecurrentClassifier):
    """Elman RNN sequence classifier: recurrent layers under a linear head.

    The nonlinearity is 'tanh' or 'relu'.
    The recurrent layers are PyTorch's `nn.RNN` with ``batch_first=True``: the same
    parameter names, shapes, gate order and initialization (every weight and bias drawn
    from U(-1/sqrt(hidden_size), 1/sqrt(hidden_size)), the head as `nn.Linear`), so
    `state_dict()` loads into a torch module and back. A linear head reads the top
    layer's last hidden state, and training minimizes the softmax cross-entropy by backpropagation
    through time.

    `X` has shape (n_samples, n_timesteps, n_features) and must be float32 (float64
    raises `TypeError`). `y` holds class labels (at least two classes).

    Not supported: bidirectional layers, dropout between layers, `proj_size`, an initial
    hidden state, and packed or variable-length sequences.

    Parameters
    ----------
    hidden_size : int, default 16
        Size of the hidden state of each recurrent layer.
    num_layers : int, default 1
        Number of stacked recurrent layers.
    optimizer : {'adam', 'adamw', 'sgd', 'rmsprop', 'adagrad', 'lion', 'adamax', 'nadam'}, default 'adam'
        Optimizer, with PyTorch's update rule and defaults.
    learning_rate : float, default 1e-3
        Learning rate, used when `lr_schedule` is None.
    batch_size : int, default 32
        Minibatch size (capped at n_samples).
    max_epochs : int, default 10
        Number of passes over the data. ``max_epochs * n_samples`` must be below 2**31 - 1.
    shuffle : bool, default True
        Reshuffle the rows every epoch.
    random_state : int or None, default 0
        Seed of the weight initialization and the shuffle.
    optimizer_options : dict or None, default None
        Extra optimizer arguments under their PyTorch names (for example
        ``{'betas': (0.9, 0.99), 'weight_decay': 0.01}``). Unknown names raise `ValueError`.
    nonlinearity : {'tanh', 'relu'}, default 'tanh'
        Activation of the recurrence, as `nn.RNN`.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode. None uses the library default. ``'fast'`` is available on Apple GPUs only.
    predict_chunk : int, default 4096
        Rows processed per chunk at prediction time.
    lr_schedule : object or None, default None
        A built-in schedule (`ConstantLR`, `WarmupLinearLR`, `WarmupCosineLR`, `StepLR`,
        `ExponentialLR` or `OneCycleLR`); it sets the learning rate of every optimizer step.
        Any other object raises `TypeError`.

    Attributes
    ----------
    classes_ : ndarray
        Sorted class labels.
    n_features_in_ : int
        Number of input features.
    n_outputs_ : int
        Number of head outputs.
    n_timesteps_ : int
        Sequence length seen in `fit`.
    n_iter_ : int
        Number of optimizer steps taken.
    loss_curve_ : ndarray
        Training loss at every optimizer step.
    params_ : ndarray
        All parameters as one flat float32 array.

    Methods
    -------
    fit(X, y)
        Train the model; returns self.
    predict(X)
        Predicted class labels.
    predict_proba(X)
        Class probabilities, (n_samples, n_classes).
    hidden_sequence(X)
        Top layer's hidden states, (n_samples, n_timesteps, hidden_size), as
        ``nn.RNN(..., batch_first=True)(X)[0]``.
    state_dict()
        Parameters as a dict of arrays under PyTorch names.
    load_state_dict(state)
        Load parameters from such a dict; returns self.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((256, 20, 4), dtype=np.float32)
    >>> y = (X[:, -1, 0] > 0.5).astype(np.int64)
    >>> model = mojolearn.RNNClassifier(hidden_size=32, max_epochs=5).fit(X, y)
    >>> pred = model.predict(X)
    """
    _CELL = "rnn"


def _state_layout(cls):
    # glue: connect fixed native storage identities at module import.
    mro = cls.__mro__
    rows = tuple(vars(base) for base in mro)  # glue: imported class metadata
    descriptor = next(row["__dict__"] for row in rows if "__dict__" in row)  # glue: native storage descriptor
    names = ("__getattribute__", "__getattr__", "__setattr__", "__dict__")
    hooks = tuple((base, name, row.get(name)) for base, row in zip(mro, rows) for name in names)  # glue: fixed native storage identities
    return cls, mro, descriptor.__get__, hooks, cls._CELL


_native_xseq_bridge.register(
    xseq_rnn_optimizers=OPTIMIZERS,
    xseq_rnn_defaults=_OPT_DEFAULTS,
    xseq_rnn_state_mappings=(dict, OrderedDict),
)

# Last statement: connect this module's classes to the native full-reset
# set_params now that they exist. The package imports without NumPy, so it
# cannot connect them at `import mojolearn`.
from ._native_full_reset_params import connect_module as _connect_full_reset
_connect_full_reset(globals())
del _connect_full_reset
# glue: pin the storage hooks only after connect_full_reset has installed the
# native __getattribute__ slot on these classes; a snapshot taken before it
# records None and refuses every connected instance in state_dict and
# load_state_dict, while a later replacement is still refused.
_native_xseq_bridge.register(
    xseq_rnn_state_layouts=tuple(_state_layout(cls) for cls in (RNNRegressor, RNNClassifier, LSTMRegressor, LSTMClassifier, GRURegressor, GRUClassifier)),  # glue: fixed recurrent classes
)
# glue: reuse the final native owner storage contract for public operations.
from ._native_full_reset_params import _CONTEXT as _XSEQ_OWNER_CONTEXT
_native_xseq_bridge.register(xseq_owner_context=_XSEQ_OWNER_CONTEXT)
