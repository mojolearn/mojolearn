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

Options (lane rnn-parity): `output='final'|'sequence'` (a score at every
step, trained with a loss at every step), `bidirectional`, `dropout`
between layers, `bias`, `clip_grad_norm`, `class_weight`, early stopping on
validation data, `warm_start`, `partial_fit`, an initial state `h_0` / `c_0`
and streaming `step`. Variable lengths (`lengths=`) in fit, predict, score
and hidden_sequence. Refused by name (sequence/NOT_IMPLEMENTED.tsv):
proj_size, float64, float16 / bfloat16.
"""
from ._mode import ParamsMixin

from collections import OrderedDict
from . import _native_xseq_bridge
from ._native_full_reset_params import _INIT_ENTRY as _FULL_RESET_INIT, _CONTEXT as _FULL_RESET_CONTEXT

# Package-import connection: every public operation below is ONE call into
# the compiled owner (bindings/xseq_neural_native_door.mojo), which admits the
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
    #: `device=None | "gpu" | "cpu"`: every operation runs on the estimator's
    #: one native route (bindings/xseq_native_common_door.mojo `xseq_route`);
    #: its check runs the same estimator on the other device, in this process.
    _mojolearn_device = True
    _CELL = "lstm"
    _TASK = 0

    def __init__(self, hidden_size=16, num_layers=1, optimizer="adam", learning_rate=1e-3,
                 batch_size=32, max_epochs=10, shuffle=True, random_state=0,
                 optimizer_options=None, nonlinearity="tanh", numeric_mode=None,
                 predict_chunk=4096, lr_schedule=None, output="final", bidirectional=False,
                 dropout=0.0, bias=True, clip_grad_norm=None, class_weight=None,
                 early_stopping=False, validation_fraction=0.1, n_iter_no_change=10, tol=1e-4,
                 warm_start=False):
        # One native call: the full-reset owner replays this constructor's field
        # writes (bindings/full_reset_constructor_native_door.mojo). connect_module
        # installs the same entry as each concrete class's __init__.
        return _FULL_RESET_INIT(self, (), {"hidden_size": hidden_size, "num_layers": num_layers, "optimizer": optimizer, "learning_rate": learning_rate, "batch_size": batch_size, "max_epochs": max_epochs, "shuffle": shuffle, "random_state": random_state, "optimizer_options": optimizer_options, "nonlinearity": nonlinearity, "numeric_mode": numeric_mode, "predict_chunk": predict_chunk, "lr_schedule": lr_schedule, "output": output, "bidirectional": bidirectional, "dropout": dropout, "bias": bias, "clip_grad_norm": clip_grad_norm, "class_weight": class_weight, "early_stopping": early_stopping, "validation_fraction": validation_fraction, "n_iter_no_change": n_iter_no_change, "tol": tol, "warm_start": warm_start}, _FULL_RESET_CONTEXT, type(self))

    # ------------------------------------------------------------ shape
    def state_dict(self):
        """{torch name: array}: `nn.LSTM`/`nn.GRU`/`nn.RNN` names for the
        recurrent layers, `head.weight` and `head.bias` for the `nn.Linear`."""
        return _XSEQ(self, "rnn_state_dict", (), _XSEQ_CONTEXT)

    def load_state_dict(self, state):
        """Load the named recurrent and head parameter arrays; return this model."""
        return _XSEQ(self, "rnn_load_state_dict", (state,), _XSEQ_CONTEXT)

    def hidden_sequence(self, X, lengths=None, h_0=None, c_0=None):
        """The top layer's hidden states h_1..h_T, (n_samples, n_timesteps,
        hidden_size), or (..., 2 * hidden_size) when bidirectional (the
        forward direction first): `nn.LSTM(..., batch_first=True)(X)[0]`.
        With ``lengths``, a sequence's steps past its length repeat its last
        valid state. ``h_0`` / ``c_0`` as in :meth:`fit`."""
        return _XSEQ(self, "rnn_hidden_sequence", (X, lengths, h_0, c_0), _XSEQ_CONTEXT)

    def step(self, X, state=None, lengths=None):
        """Run the next steps of live sequences, carrying their state.

        Parameters
        ----------
        X : array-like of shape (n_sequences, n_steps, n_features)
            The next steps of each sequence (often one step).
        state : tuple or None, default None
            The ``state`` returned by the previous call; None starts every
            sequence from zeros.
        lengths : array-like of shape (n_sequences,), default None
            Valid steps of each sequence in ``X``.

        Returns
        -------
        scores : ndarray
            What :meth:`predict` returns for regressors, or the class
            probabilities for classifiers, for these steps (every step with
            ``output='sequence'``).
        state : tuple
            ``(h, c)``: each layer and direction's final hidden state,
            shape (num_layers * num_directions, n_sequences, hidden_size),
            and the LSTM's cell state (None for GRU and RNN). Pass it to the
            next call.

        Running a sequence through ``step`` in pieces gives the same scores
        and states as running it whole, bit for bit, at any number of
        features. A bidirectional model cannot stream (its reverse direction
        reads the steps that come later) and raises ``ValueError``; use
        :meth:`predict` on the whole sequence.

        Examples
        --------
        >>> import numpy as np
        >>> from mojolearn import GRUClassifier
        >>> X = np.random.rand(32, 10, 3).astype(np.float32)
        >>> y = (X[:, :, 0] > 0.5).astype(np.int64)
        >>> model = GRUClassifier(output="sequence", max_epochs=2).fit(X, y)
        >>> state = None
        >>> for t in range(10):
        ...     scores, state = model.step(X[:, t:t + 1], state)
        """
        name = "rnn_predict_proba" if self._TASK == 1 else "rnn_predict"
        return _XSEQ(self, name, (X, lengths, state, None, True), _XSEQ_CONTEXT)

    def vendor_used(self):
        """The backend this estimator runs on.

        Returns
        -------
        str
            ``'metal'``, ``'cuda'``, ``'hip'`` or ``'cpu'``, read from the
            compiled binary the estimator uses.
        """
        return _XSEQ(self, "xseq_vendor_used", (), _XSEQ_CONTEXT)


class _RecurrentRegressor(_RecurrentBase):
    # scikit-learn tags, read by `_mode.sklearn_tags`
    _estimator_type = "regressor"
    _TASK = 0

    def fit(self, X, y, lengths=None, sample_weight=None, h_0=None, c_0=None, validation_data=None,
            restore_best=True):
        """Train the network on sequences ``X`` and targets ``y``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_timesteps, n_features)
            Training sequences; shorter sequences are padded to n_timesteps.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Targets; with ``output='sequence'``, (n_samples, n_timesteps)
            or (n_samples, n_timesteps, n_outputs), a target at every step.
        lengths : array-like of shape (n_samples,), default None
            Each sequence's number of valid steps, from 1 to n_timesteps.
            Steps past a sequence's length are padding: they never enter
            its state, and the output is read at its own last valid step.
        sample_weight : array-like of shape (n_samples,) or (n_samples, n_timesteps), default None
            Weight of each sequence, or with ``output='sequence'`` of each
            step. The loss is the weighted mean (sum of weight times loss
            over the sum of the weights); padded steps weigh 0.
        h_0, c_0 : array-like of shape (num_layers * num_directions, n_samples, hidden_size), default None
            The initial hidden state (and the LSTM's cell state) of every
            sequence, as PyTorch's ``h_0`` / ``c_0``; None is zeros.
        validation_data : tuple, default None
            ``(X_val, y_val)`` or ``(X_val, y_val, lengths_val)``: the loss
            on it is recorded after every epoch (``validation_loss_``), and
            with ``early_stopping=True`` training stops after
            ``n_iter_no_change`` epochs without an improvement of ``tol``.
        restore_best : bool, default True
            With validation, keep the parameters of the epoch with the lowest
            validation loss (``best_epoch_``).

        Returns
        -------
        self

        Examples
        --------
        >>> import numpy as np
        >>> from mojolearn import GRURegressor
        >>> X = np.random.rand(64, 12, 2).astype(np.float32)
        >>> lengths = np.random.randint(4, 13, size=64)
        >>> y = X[np.arange(64), lengths - 1, 0]
        >>> model = GRURegressor(hidden_size=16).fit(X, y, lengths=lengths)
        >>> model.predict(X, lengths=lengths).shape
        (64,)
        """
        return _XSEQ(self, "rnn_fit", (X, y, lengths, sample_weight, h_0, c_0, validation_data, restore_best),
                     _XSEQ_CONTEXT)

    def partial_fit(self, X, y, lengths=None, sample_weight=None, h_0=None, c_0=None):
        """Train one more epoch from the current parameters and optimizer
        state (the first call starts as :meth:`fit` does). Run your own
        loop, one epoch per call; arguments as in :meth:`fit`. For a
        classifier the first call's labels fix ``classes_``; later calls may
        hold any subset of them.

        Returns
        -------
        self
        """
        return _XSEQ(self, "rnn_fit", (X, y, lengths, sample_weight, h_0, c_0, None, True, True), _XSEQ_CONTEXT)

    def predict(self, X, lengths=None, h_0=None, c_0=None):
        """Predicted targets for sequences ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_timesteps, n_features)
            Sequences.
        lengths : array-like of shape (n_samples,), default None
            Each sequence's number of valid steps, from 1 to n_timesteps.
            Steps past a sequence's length are padding: they never enter
            its state, and the output is read at its own last valid step.

        Returns
        -------
        ndarray of shape (n_samples,) or (n_samples, n_outputs)
            With ``output='sequence'``, (n_samples, n_timesteps[, n_outputs]):
            a prediction at every step (a padded step repeats its sequence's
            last valid one).
        """
        return _XSEQ(self, "rnn_predict", (X, lengths, h_0, c_0), _XSEQ_CONTEXT)

    def score(self, X, y, sample_weight=None, lengths=None):
        """Coefficient of determination R^2 of ``predict(X)`` against ``y``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_timesteps, n_features)
            Test sequences.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True targets.
        sample_weight : array-like of shape (n_samples,), default None
            Sample weights.
        lengths : array-like of shape (n_samples,), default None
            Each sequence's number of valid steps, from 1 to n_timesteps.
            Steps past a sequence's length are padding: they never enter
            its state, and the output is read at its own last valid step.

        Returns
        -------
        float
            R^2, averaged uniformly over outputs; 1.0 is a perfect fit.
        """
        return _XSEQ(self, "rnn_score", (X, y, sample_weight, lengths), _XSEQ_CONTEXT)


class _RecurrentClassifier(_RecurrentBase):
    # scikit-learn tags, read by `_mode.sklearn_tags`
    _estimator_type = "classifier"
    _TASK = 1

    def fit(self, X, y, lengths=None, sample_weight=None, h_0=None, c_0=None, validation_data=None,
            restore_best=True):
        """Train the network on sequences ``X`` and labels ``y``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_timesteps, n_features)
            Training sequences; shorter sequences are padded to n_timesteps.
        y : array-like of shape (n_samples,)
            Class labels; with ``output='sequence'``, (n_samples,
            n_timesteps), a label at every step. With ``lengths``, the
            labels of padded steps are not read: ``classes_`` holds the
            labels of valid steps.
        lengths : array-like of shape (n_samples,), default None
            Each sequence's number of valid steps, from 1 to n_timesteps.
            Steps past a sequence's length are padding: they never enter
            its state, and the output is read at its own last valid step.
        sample_weight : array-like of shape (n_samples,) or (n_samples, n_timesteps), default None
            Weight of each sequence, or with ``output='sequence'`` of each
            step. The loss is the weighted mean (sum of weight times loss
            over the sum of the weights); padded steps weigh 0.
        h_0, c_0 : array-like of shape (num_layers * num_directions, n_samples, hidden_size), default None
            The initial hidden state (and the LSTM's cell state) of every
            sequence, as PyTorch's ``h_0`` / ``c_0``; None is zeros.
        validation_data : tuple, default None
            ``(X_val, y_val)`` or ``(X_val, y_val, lengths_val)``: the loss
            on it is recorded after every epoch (``validation_loss_``), and
            with ``early_stopping=True`` training stops after
            ``n_iter_no_change`` epochs without an improvement of ``tol``.
        restore_best : bool, default True
            With validation, keep the parameters of the epoch with the lowest
            validation loss (``best_epoch_``).

        Returns
        -------
        self

        Examples
        --------
        >>> import numpy as np
        >>> from mojolearn import LSTMClassifier
        >>> X = np.random.rand(64, 12, 2).astype(np.float32)
        >>> lengths = np.random.randint(4, 13, size=64)
        >>> y = (X[np.arange(64), lengths - 1, 0] > 0.5).astype(np.int64)
        >>> model = LSTMClassifier(hidden_size=16).fit(X, y, lengths=lengths)
        >>> model.predict(X, lengths=lengths).shape
        (64,)
        """
        return _XSEQ(self, "rnn_fit", (X, y, lengths, sample_weight, h_0, c_0, validation_data, restore_best),
                     _XSEQ_CONTEXT)

    def partial_fit(self, X, y, lengths=None, sample_weight=None, h_0=None, c_0=None):
        """Train one more epoch from the current parameters and optimizer
        state (the first call starts as :meth:`fit` does). Run your own
        loop, one epoch per call; arguments as in :meth:`fit`. For a
        classifier the first call's labels fix ``classes_``; later calls may
        hold any subset of them.

        Returns
        -------
        self
        """
        return _XSEQ(self, "rnn_fit", (X, y, lengths, sample_weight, h_0, c_0, None, True, True), _XSEQ_CONTEXT)

    def predict_proba(self, X, lengths=None, h_0=None, c_0=None):
        """Class probabilities for sequences ``X``, shape (n_samples, n_classes),
        or (n_samples, n_timesteps, n_classes) with ``output='sequence'``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_timesteps, n_features)
            Sequences.
        lengths : array-like of shape (n_samples,), default None
            Each sequence's number of valid steps, from 1 to n_timesteps.
            Steps past a sequence's length are padding: they never enter
            its state, and the output is read at its own last valid step.
        """
        return _XSEQ(self, "rnn_predict_proba", (X, lengths, h_0, c_0), _XSEQ_CONTEXT)

    def predict(self, X, lengths=None, h_0=None, c_0=None):
        """Predicted class labels for sequences ``X``, shape (n_samples,), or
        (n_samples, n_timesteps) with ``output='sequence'``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_timesteps, n_features)
            Sequences.
        lengths : array-like of shape (n_samples,), default None
            Each sequence's number of valid steps, from 1 to n_timesteps.
            Steps past a sequence's length are padding: they never enter
            its state, and the output is read at its own last valid step.
        """
        return _XSEQ(self, "rnn_predict", (X, lengths, h_0, c_0), _XSEQ_CONTEXT)

    def score(self, X, y, sample_weight=None, lengths=None):
        """Mean accuracy of ``predict(X)`` against ``y``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_timesteps, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels.
        sample_weight : array-like of shape (n_samples,), default None
            Sample weights.
        lengths : array-like of shape (n_samples,), default None
            Each sequence's number of valid steps, from 1 to n_timesteps.
            Steps past a sequence's length are padding: they never enter
            its state, and the output is read at its own last valid step.

        Returns
        -------
        float
            Fraction of samples predicted correctly (weighted when
            ``sample_weight`` is given).
        """
        return _XSEQ(self, "rnn_classifier_score", (X, y, sample_weight, lengths), _XSEQ_CONTEXT)


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

    Variable-length sequences take ``lengths=`` in every method. Not supported: `proj_size`
    and float64, float16 or bfloat16 data.

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
    device : {None, 'gpu', 'cpu'}, default None
        Where the model is fitted and predicts. None uses this installation's device, 'cpu'
        runs on the CPU and 'gpu' requires a GPU installation. Other values raise `ValueError`.
    predict_chunk : int, default 4096
        Rows processed per chunk at prediction time.
    lr_schedule : object or None, default None
        A built-in schedule (`ConstantLR`, `WarmupLinearLR`, `WarmupCosineLR`, `StepLR`,
        `ExponentialLR` or `OneCycleLR`); it sets the learning rate of every optimizer step.
        Any other object raises `TypeError`.
    output : {'final', 'sequence'}, default 'final'
        ``'final'``: one prediction per sequence, from the final state of each direction
        (PyTorch's ``h_n``: a sequence's last valid step). ``'sequence'``: a prediction at
        every step from that step's state, trained with a loss at every valid step (the
        word Mojolearn's `Sequential` recurrent layers use).
    bidirectional : bool, default False
        Add a reverse direction to every layer, as ``nn.GRU(bidirectional=True)``;
        `state_dict` names its parameters with PyTorch's ``_reverse`` suffix.
    dropout : float, default 0.0
        Dropout on the output of every layer but the last, during fit only.
    bias : bool, default True
        False: the recurrent layers have no biases (the head keeps its own).
    clip_grad_norm : float or None, default None
        Scale the gradients so their total 2-norm is at most this before every update,
        as ``torch.nn.utils.clip_grad_norm_``.
    class_weight : dict, 'balanced' or None, default None
        Classifiers: weight of each class in the loss, ``{label: weight}`` (missing labels
        weigh 1) or ``'balanced'`` (``n / (n_classes * count)``).
    early_stopping : bool, default False
        Stop when the validation loss has not improved by ``tol`` for ``n_iter_no_change``
        epochs and keep the best epoch. Without ``validation_data`` in `fit`, the last
        ``validation_fraction`` of the sequences is held out.
    validation_fraction : float, default 0.1
        Share of the sequences held out by ``early_stopping``.
    n_iter_no_change : int, default 10
        Epochs without improvement before ``early_stopping`` stops.
    tol : float, default 1e-4
        The improvement of the validation loss that counts for ``early_stopping``.
    warm_start : bool, default False
        Start `fit` from the fitted parameters when the shapes allow.

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
    fit_loop_ : tuple of (int, float)
        The number of epochs the last fit ran (each epoch is
        ``ceil(n_samples / batch_size)`` optimizer steps) and the time in
        milliseconds its training loop took, from the start of the first
        epoch to the end of the last.
    loss_curve_ : ndarray
        Training loss at every optimizer step.
    validation_loss_ : ndarray
        With validation, the validation loss after every epoch.
    best_epoch_ : int
        With validation, the epoch of the lowest validation loss.
    params_ : ndarray
        All parameters as one flat float32 array.

    Methods
    -------
    fit(X, y)
        Train the model; returns self.
    partial_fit(X, y)
        Train one more epoch; returns self.
    step(X, state)
        Predict the next steps of live sequences, carrying their state.
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
    # The member ABI (core/member_abi.mojo): family 20 (Sequence), kind 0.
    _mojolearn_member = (20, 0)


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

    Variable-length sequences take ``lengths=`` in every method. Not supported: `proj_size`
    and float64, float16 or bfloat16 data.

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
    device : {None, 'gpu', 'cpu'}, default None
        Where the model is fitted and predicts. None uses this installation's device, 'cpu'
        runs on the CPU and 'gpu' requires a GPU installation. Other values raise `ValueError`.
    predict_chunk : int, default 4096
        Rows processed per chunk at prediction time.
    lr_schedule : object or None, default None
        A built-in schedule (`ConstantLR`, `WarmupLinearLR`, `WarmupCosineLR`, `StepLR`,
        `ExponentialLR` or `OneCycleLR`); it sets the learning rate of every optimizer step.
        Any other object raises `TypeError`.
    output : {'final', 'sequence'}, default 'final'
        ``'final'``: one prediction per sequence, from the final state of each direction
        (PyTorch's ``h_n``: a sequence's last valid step). ``'sequence'``: a prediction at
        every step from that step's state, trained with a loss at every valid step (the
        word Mojolearn's `Sequential` recurrent layers use).
    bidirectional : bool, default False
        Add a reverse direction to every layer, as ``nn.GRU(bidirectional=True)``;
        `state_dict` names its parameters with PyTorch's ``_reverse`` suffix.
    dropout : float, default 0.0
        Dropout on the output of every layer but the last, during fit only.
    bias : bool, default True
        False: the recurrent layers have no biases (the head keeps its own).
    clip_grad_norm : float or None, default None
        Scale the gradients so their total 2-norm is at most this before every update,
        as ``torch.nn.utils.clip_grad_norm_``.
    class_weight : dict, 'balanced' or None, default None
        Classifiers: weight of each class in the loss, ``{label: weight}`` (missing labels
        weigh 1) or ``'balanced'`` (``n / (n_classes * count)``).
    early_stopping : bool, default False
        Stop when the validation loss has not improved by ``tol`` for ``n_iter_no_change``
        epochs and keep the best epoch. Without ``validation_data`` in `fit`, the last
        ``validation_fraction`` of the sequences is held out.
    validation_fraction : float, default 0.1
        Share of the sequences held out by ``early_stopping``.
    n_iter_no_change : int, default 10
        Epochs without improvement before ``early_stopping`` stops.
    tol : float, default 1e-4
        The improvement of the validation loss that counts for ``early_stopping``.
    warm_start : bool, default False
        Start `fit` from the fitted parameters when the shapes allow.

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
    fit_loop_ : tuple of (int, float)
        The number of epochs the last fit ran (each epoch is
        ``ceil(n_samples / batch_size)`` optimizer steps) and the time in
        milliseconds its training loop took, from the start of the first
        epoch to the end of the last.
    loss_curve_ : ndarray
        Training loss at every optimizer step.
    validation_loss_ : ndarray
        With validation, the validation loss after every epoch.
    best_epoch_ : int
        With validation, the epoch of the lowest validation loss.
    params_ : ndarray
        All parameters as one flat float32 array.

    Methods
    -------
    fit(X, y)
        Train the model; returns self.
    partial_fit(X, y)
        Train one more epoch; returns self.
    step(X, state)
        Predict the next steps of live sequences, carrying their state.
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
    # The member ABI (core/member_abi.mojo): family 20 (Sequence), kind 1.
    _mojolearn_member = (20, 1)


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

    Variable-length sequences take ``lengths=`` in every method. Not supported: `proj_size`
    and float64, float16 or bfloat16 data.

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
    device : {None, 'gpu', 'cpu'}, default None
        Where the model is fitted and predicts. None uses this installation's device, 'cpu'
        runs on the CPU and 'gpu' requires a GPU installation. Other values raise `ValueError`.
    predict_chunk : int, default 4096
        Rows processed per chunk at prediction time.
    lr_schedule : object or None, default None
        A built-in schedule (`ConstantLR`, `WarmupLinearLR`, `WarmupCosineLR`, `StepLR`,
        `ExponentialLR` or `OneCycleLR`); it sets the learning rate of every optimizer step.
        Any other object raises `TypeError`.
    output : {'final', 'sequence'}, default 'final'
        ``'final'``: one prediction per sequence, from the final state of each direction
        (PyTorch's ``h_n``: a sequence's last valid step). ``'sequence'``: a prediction at
        every step from that step's state, trained with a loss at every valid step (the
        word Mojolearn's `Sequential` recurrent layers use).
    bidirectional : bool, default False
        Add a reverse direction to every layer, as ``nn.GRU(bidirectional=True)``;
        `state_dict` names its parameters with PyTorch's ``_reverse`` suffix.
    dropout : float, default 0.0
        Dropout on the output of every layer but the last, during fit only.
    bias : bool, default True
        False: the recurrent layers have no biases (the head keeps its own).
    clip_grad_norm : float or None, default None
        Scale the gradients so their total 2-norm is at most this before every update,
        as ``torch.nn.utils.clip_grad_norm_``.
    class_weight : dict, 'balanced' or None, default None
        Classifiers: weight of each class in the loss, ``{label: weight}`` (missing labels
        weigh 1) or ``'balanced'`` (``n / (n_classes * count)``).
    early_stopping : bool, default False
        Stop when the validation loss has not improved by ``tol`` for ``n_iter_no_change``
        epochs and keep the best epoch. Without ``validation_data`` in `fit`, the last
        ``validation_fraction`` of the sequences is held out.
    validation_fraction : float, default 0.1
        Share of the sequences held out by ``early_stopping``.
    n_iter_no_change : int, default 10
        Epochs without improvement before ``early_stopping`` stops.
    tol : float, default 1e-4
        The improvement of the validation loss that counts for ``early_stopping``.
    warm_start : bool, default False
        Start `fit` from the fitted parameters when the shapes allow.

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
    fit_loop_ : tuple of (int, float)
        The number of epochs the last fit ran (each epoch is
        ``ceil(n_samples / batch_size)`` optimizer steps) and the time in
        milliseconds its training loop took, from the start of the first
        epoch to the end of the last.
    loss_curve_ : ndarray
        Training loss at every optimizer step.
    validation_loss_ : ndarray
        With validation, the validation loss after every epoch.
    best_epoch_ : int
        With validation, the epoch of the lowest validation loss.
    params_ : ndarray
        All parameters as one flat float32 array.

    Methods
    -------
    fit(X, y)
        Train the model; returns self.
    partial_fit(X, y)
        Train one more epoch; returns self.
    step(X, state)
        Predict the next steps of live sequences, carrying their state.
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
    # The member ABI (core/member_abi.mojo): family 20 (Sequence), kind 0.
    _mojolearn_member = (20, 0)


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

    Variable-length sequences take ``lengths=`` in every method. Not supported: `proj_size`
    and float64, float16 or bfloat16 data.

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
    device : {None, 'gpu', 'cpu'}, default None
        Where the model is fitted and predicts. None uses this installation's device, 'cpu'
        runs on the CPU and 'gpu' requires a GPU installation. Other values raise `ValueError`.
    predict_chunk : int, default 4096
        Rows processed per chunk at prediction time.
    lr_schedule : object or None, default None
        A built-in schedule (`ConstantLR`, `WarmupLinearLR`, `WarmupCosineLR`, `StepLR`,
        `ExponentialLR` or `OneCycleLR`); it sets the learning rate of every optimizer step.
        Any other object raises `TypeError`.
    output : {'final', 'sequence'}, default 'final'
        ``'final'``: one prediction per sequence, from the final state of each direction
        (PyTorch's ``h_n``: a sequence's last valid step). ``'sequence'``: a prediction at
        every step from that step's state, trained with a loss at every valid step (the
        word Mojolearn's `Sequential` recurrent layers use).
    bidirectional : bool, default False
        Add a reverse direction to every layer, as ``nn.GRU(bidirectional=True)``;
        `state_dict` names its parameters with PyTorch's ``_reverse`` suffix.
    dropout : float, default 0.0
        Dropout on the output of every layer but the last, during fit only.
    bias : bool, default True
        False: the recurrent layers have no biases (the head keeps its own).
    clip_grad_norm : float or None, default None
        Scale the gradients so their total 2-norm is at most this before every update,
        as ``torch.nn.utils.clip_grad_norm_``.
    class_weight : dict, 'balanced' or None, default None
        Classifiers: weight of each class in the loss, ``{label: weight}`` (missing labels
        weigh 1) or ``'balanced'`` (``n / (n_classes * count)``).
    early_stopping : bool, default False
        Stop when the validation loss has not improved by ``tol`` for ``n_iter_no_change``
        epochs and keep the best epoch. Without ``validation_data`` in `fit`, the last
        ``validation_fraction`` of the sequences is held out.
    validation_fraction : float, default 0.1
        Share of the sequences held out by ``early_stopping``.
    n_iter_no_change : int, default 10
        Epochs without improvement before ``early_stopping`` stops.
    tol : float, default 1e-4
        The improvement of the validation loss that counts for ``early_stopping``.
    warm_start : bool, default False
        Start `fit` from the fitted parameters when the shapes allow.

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
    fit_loop_ : tuple of (int, float)
        The number of epochs the last fit ran (each epoch is
        ``ceil(n_samples / batch_size)`` optimizer steps) and the time in
        milliseconds its training loop took, from the start of the first
        epoch to the end of the last.
    loss_curve_ : ndarray
        Training loss at every optimizer step.
    validation_loss_ : ndarray
        With validation, the validation loss after every epoch.
    best_epoch_ : int
        With validation, the epoch of the lowest validation loss.
    params_ : ndarray
        All parameters as one flat float32 array.

    Methods
    -------
    fit(X, y)
        Train the model; returns self.
    partial_fit(X, y)
        Train one more epoch; returns self.
    step(X, state)
        Predict the next steps of live sequences, carrying their state.
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
    # The member ABI (core/member_abi.mojo): family 20 (Sequence), kind 1.
    _mojolearn_member = (20, 1)


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

    Variable-length sequences take ``lengths=`` in every method. Not supported: `proj_size`
    and float64, float16 or bfloat16 data.

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
    device : {None, 'gpu', 'cpu'}, default None
        Where the model is fitted and predicts. None uses this installation's device, 'cpu'
        runs on the CPU and 'gpu' requires a GPU installation. Other values raise `ValueError`.
    predict_chunk : int, default 4096
        Rows processed per chunk at prediction time.
    lr_schedule : object or None, default None
        A built-in schedule (`ConstantLR`, `WarmupLinearLR`, `WarmupCosineLR`, `StepLR`,
        `ExponentialLR` or `OneCycleLR`); it sets the learning rate of every optimizer step.
        Any other object raises `TypeError`.
    output : {'final', 'sequence'}, default 'final'
        ``'final'``: one prediction per sequence, from the final state of each direction
        (PyTorch's ``h_n``: a sequence's last valid step). ``'sequence'``: a prediction at
        every step from that step's state, trained with a loss at every valid step (the
        word Mojolearn's `Sequential` recurrent layers use).
    bidirectional : bool, default False
        Add a reverse direction to every layer, as ``nn.GRU(bidirectional=True)``;
        `state_dict` names its parameters with PyTorch's ``_reverse`` suffix.
    dropout : float, default 0.0
        Dropout on the output of every layer but the last, during fit only.
    bias : bool, default True
        False: the recurrent layers have no biases (the head keeps its own).
    clip_grad_norm : float or None, default None
        Scale the gradients so their total 2-norm is at most this before every update,
        as ``torch.nn.utils.clip_grad_norm_``.
    class_weight : dict, 'balanced' or None, default None
        Classifiers: weight of each class in the loss, ``{label: weight}`` (missing labels
        weigh 1) or ``'balanced'`` (``n / (n_classes * count)``).
    early_stopping : bool, default False
        Stop when the validation loss has not improved by ``tol`` for ``n_iter_no_change``
        epochs and keep the best epoch. Without ``validation_data`` in `fit`, the last
        ``validation_fraction`` of the sequences is held out.
    validation_fraction : float, default 0.1
        Share of the sequences held out by ``early_stopping``.
    n_iter_no_change : int, default 10
        Epochs without improvement before ``early_stopping`` stops.
    tol : float, default 1e-4
        The improvement of the validation loss that counts for ``early_stopping``.
    warm_start : bool, default False
        Start `fit` from the fitted parameters when the shapes allow.

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
    fit_loop_ : tuple of (int, float)
        The number of epochs the last fit ran (each epoch is
        ``ceil(n_samples / batch_size)`` optimizer steps) and the time in
        milliseconds its training loop took, from the start of the first
        epoch to the end of the last.
    loss_curve_ : ndarray
        Training loss at every optimizer step.
    validation_loss_ : ndarray
        With validation, the validation loss after every epoch.
    best_epoch_ : int
        With validation, the epoch of the lowest validation loss.
    params_ : ndarray
        All parameters as one flat float32 array.

    Methods
    -------
    fit(X, y)
        Train the model; returns self.
    partial_fit(X, y)
        Train one more epoch; returns self.
    step(X, state)
        Predict the next steps of live sequences, carrying their state.
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
    # The member ABI (core/member_abi.mojo): family 20 (Sequence), kind 0.
    _mojolearn_member = (20, 0)


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

    Variable-length sequences take ``lengths=`` in every method. Not supported: `proj_size`
    and float64, float16 or bfloat16 data.

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
    device : {None, 'gpu', 'cpu'}, default None
        Where the model is fitted and predicts. None uses this installation's device, 'cpu'
        runs on the CPU and 'gpu' requires a GPU installation. Other values raise `ValueError`.
    predict_chunk : int, default 4096
        Rows processed per chunk at prediction time.
    lr_schedule : object or None, default None
        A built-in schedule (`ConstantLR`, `WarmupLinearLR`, `WarmupCosineLR`, `StepLR`,
        `ExponentialLR` or `OneCycleLR`); it sets the learning rate of every optimizer step.
        Any other object raises `TypeError`.
    output : {'final', 'sequence'}, default 'final'
        ``'final'``: one prediction per sequence, from the final state of each direction
        (PyTorch's ``h_n``: a sequence's last valid step). ``'sequence'``: a prediction at
        every step from that step's state, trained with a loss at every valid step (the
        word Mojolearn's `Sequential` recurrent layers use).
    bidirectional : bool, default False
        Add a reverse direction to every layer, as ``nn.GRU(bidirectional=True)``;
        `state_dict` names its parameters with PyTorch's ``_reverse`` suffix.
    dropout : float, default 0.0
        Dropout on the output of every layer but the last, during fit only.
    bias : bool, default True
        False: the recurrent layers have no biases (the head keeps its own).
    clip_grad_norm : float or None, default None
        Scale the gradients so their total 2-norm is at most this before every update,
        as ``torch.nn.utils.clip_grad_norm_``.
    class_weight : dict, 'balanced' or None, default None
        Classifiers: weight of each class in the loss, ``{label: weight}`` (missing labels
        weigh 1) or ``'balanced'`` (``n / (n_classes * count)``).
    early_stopping : bool, default False
        Stop when the validation loss has not improved by ``tol`` for ``n_iter_no_change``
        epochs and keep the best epoch. Without ``validation_data`` in `fit`, the last
        ``validation_fraction`` of the sequences is held out.
    validation_fraction : float, default 0.1
        Share of the sequences held out by ``early_stopping``.
    n_iter_no_change : int, default 10
        Epochs without improvement before ``early_stopping`` stops.
    tol : float, default 1e-4
        The improvement of the validation loss that counts for ``early_stopping``.
    warm_start : bool, default False
        Start `fit` from the fitted parameters when the shapes allow.

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
    fit_loop_ : tuple of (int, float)
        The number of epochs the last fit ran (each epoch is
        ``ceil(n_samples / batch_size)`` optimizer steps) and the time in
        milliseconds its training loop took, from the start of the first
        epoch to the end of the last.
    loss_curve_ : ndarray
        Training loss at every optimizer step.
    validation_loss_ : ndarray
        With validation, the validation loss after every epoch.
    best_epoch_ : int
        With validation, the epoch of the lowest validation loss.
    params_ : ndarray
        All parameters as one flat float32 array.

    Methods
    -------
    fit(X, y)
        Train the model; returns self.
    partial_fit(X, y)
        Train one more epoch; returns self.
    step(X, state)
        Predict the next steps of live sequences, carrying their state.
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
    # The member ABI (core/member_abi.mojo): family 20 (Sequence), kind 1.
    _mojolearn_member = (20, 1)


def _state_layout(cls):
    # glue: the exact recurrent class and its cell name. Storage, ancestry and
    # hook admission is the connected owner contract (xseq_public_owner),
    # pinned after the native attribute slots are installed.
    return cls, cls._CELL


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
_native_xseq_bridge.register(
    xseq_rnn_state_layouts=tuple(_state_layout(cls) for cls in (RNNRegressor, RNNClassifier, LSTMRegressor, LSTMClassifier, GRURegressor, GRUClassifier)),  # glue: fixed recurrent classes
)
# glue: reuse the final native owner storage contract for public operations.
from ._native_full_reset_params import _CONTEXT as _XSEQ_OWNER_CONTEXT
_native_xseq_bridge.register(xseq_owner_context=_XSEQ_OWNER_CONTEXT)
