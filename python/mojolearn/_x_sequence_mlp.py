# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""MLPClassifier and MLPRegressor, scikit-learn's
(`sklearn/neural_network/_multilayer_perceptron.py`), trained on the GPU by
the sequence lane's binding (`sequence/mlp.mojo`, `sequence/mlp_fit.mojo`):
the same parameters, attributes, initialisation (Glorot uniform with the
reference's bounds, drawn from `random_state` by the seeded Mojo stream
`_buffer.InitStream`, so not the reference's RandomState values), losses,
Adam and SGD updates,
learning-rate schedules and stopping rule. Shuffling uses the lane's own
host generator seeded from `random_state`, so a shuffled fit is not the
reference's sample order.

Refused by name (sequence/NOT_IMPLEMENTED.tsv): solver='lbfgs',
early_stopping=True, warm_start, partial_fit, multilabel targets, verbose,
float64 input."""
from ._optional_numpy import require_numpy
from ._mode import ClassifierScoreMixin, ParamsMixin, RegressorScoreMixin
np = require_numpy('_x_sequence_mlp')

from . import _native_xseq_bridge

# Package-import connection: fit, predict and predict_proba are ONE call each
# into the compiled owner (bindings/xseq_neural_native.mojo): input casts,
# label encoding, the Glorot draws (straight into the flat parameter block
# `mlp_fit` trains), the stopping report and the predicted labels.
_XSEQ, _XSEQ_CONTEXT = _native_xseq_bridge.connect()

_ACT = {"identity": 0, "logistic": 1, "tanh": 2, "relu": 3}
_SOLVER = {"adam": 0, "sgd": 1}
_LR = {"constant": 0, "invscaling": 1, "adaptive": 2}


class _BaseMLP(ParamsMixin):
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


class MLPRegressor(RegressorScoreMixin, _BaseMLP):
    """Multi-layer perceptron regressor, as scikit-learn's `MLPRegressor`.

    Minimizes half the mean squared error plus an L2 penalty, with an identity output
    layer, using Adam or SGD. Parameters, attributes, losses, update rules,
    learning-rate schedules and the stopping rule follow scikit-learn. Weights use
    scikit-learn's Glorot-uniform bounds but are drawn from mojolearn's own seeded
    generator, and the shuffle order is mojolearn's own, so results for a given
    `random_state` differ from scikit-learn's.

    `X` must be 2-D float32 (float64 raises `TypeError`); `y` is (n_samples,) or
    (n_samples, n_outputs). Not supported: ``solver='lbfgs'``, ``early_stopping=True``,
    `warm_start`, `partial_fit` and `verbose`.

    Parameters
    ----------
    hidden_layer_sizes : int or tuple of int, default (100,)
        Width of each hidden layer; every width must be > 0.
    activation : {'identity', 'logistic', 'tanh', 'relu'}, default 'relu'
        Hidden-layer activation.
    solver : {'adam', 'sgd'}, default 'adam'
        Optimizer. 'lbfgs' raises `NotImplementedError`.
    alpha : float, default 0.0001
        L2 penalty strength (>= 0).
    batch_size : int or 'auto', default 'auto'
        Minibatch size; 'auto' is ``min(200, n_samples)``.
    learning_rate : {'constant', 'invscaling', 'adaptive'}, default 'constant'
        Learning-rate schedule of the 'sgd' solver, as scikit-learn.
    learning_rate_init : float, default 0.001
        Initial learning rate (> 0).
    power_t : float, default 0.5
        Exponent of the 'invscaling' schedule.
    max_iter : int, default 200
        Maximum number of epochs (>= 1).
    shuffle : bool, default True
        Shuffle the rows every epoch.
    random_state : int or None, default None
        Seed of the weight initialization and the shuffle.
    tol : float, default 1e-4
        Tolerance of the stopping rule.
    verbose : bool, default False
        Must be False; True raises `NotImplementedError`.
    warm_start : bool, default False
        Must be False; True raises `NotImplementedError`.
    momentum : float, default 0.9
        Momentum of the 'sgd' solver.
    nesterovs_momentum : bool, default True
        Use Nesterov momentum with the 'sgd' solver.
    early_stopping : bool, default False
        Must be False; True raises `NotImplementedError`.
    validation_fraction : float, default 0.1
        Accepted for compatibility; unused because early stopping is not supported.
    beta_1 : float, default 0.9
        Adam's first-moment decay.
    beta_2 : float, default 0.999
        Adam's second-moment decay.
    epsilon : float, default 1e-8
        Adam's denominator term.
    n_iter_no_change : int, default 10
        Epochs without `tol` improvement before stopping.
    max_fun : int, default 15000
        Accepted for compatibility (lbfgs only); unused.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode. None uses the library default. ``'fast'`` is available on Apple GPUs only.

    Attributes
    ----------
    coefs_ : list of ndarray
        Weight matrix of each layer, (n_in, n_out).
    intercepts_ : list of ndarray
        Bias vector of each layer.
    loss_ : float
        Loss of the last epoch.
    best_loss_ : float
        Lowest loss seen.
    loss_curve_ : list of float
        Loss of every epoch.
    n_iter_ : int
        Number of epochs run.
    t_ : int
        Number of training samples seen.
    n_features_in_ : int
        Number of input features.
    n_layers_ : int
        Number of layers, input and output included.
    n_outputs_ : int
        Number of outputs.
    out_activation_ : str
        Output activation.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> y = X.sum(axis=1)
    >>> model = mojolearn.MLPRegressor(hidden_layer_sizes=(64,), max_iter=50).fit(X, y)
    >>> pred = model.predict(X)
    """
    # scikit-learn tags, read by `_mode.sklearn_tags`
    _estimator_type = "regressor"
    _LOSS_CODE = 0

    def fit(self, X, y):
        """Train on `X` (n_samples, n_features) float32 and targets `y`; returns self."""
        return _XSEQ(self, "mlp_fit", (X, y), _XSEQ_CONTEXT)

    def predict(self, X):
        """Predicted values, (n_samples,) for a 1-D training target, else (n_samples, n_outputs)."""
        return _XSEQ(self, "mlp_predict", (X,), _XSEQ_CONTEXT)

    def score(self, X, y, sample_weight=None):
        """Coefficient of determination R^2 of ``predict(X)`` against ``y``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True targets.
        sample_weight : array-like of shape (n_samples,), default None
            Sample weights.

        Returns
        -------
        float
            R^2, averaged uniformly over outputs; 1.0 is a perfect fit.
        """
        return _XSEQ(self, "mlp_score", (X, y, sample_weight), _XSEQ_CONTEXT)


class MLPClassifier(ClassifierScoreMixin, _BaseMLP):
    """Multi-layer perceptron classifier, as scikit-learn's `MLPClassifier`.

    Minimizes the log loss plus an L2 penalty, with a logistic output for two classes
    and softmax for more, using Adam or SGD. Parameters, attributes, losses, update
    rules, learning-rate schedules and the stopping rule follow scikit-learn. Weights
    use scikit-learn's Glorot-uniform bounds but are drawn from mojolearn's own seeded
    generator, and the shuffle order is mojolearn's own, so results for a given
    `random_state` differ from scikit-learn's.

    `X` must be 2-D float32 (float64 raises `TypeError`); `y` is a 1-D array of labels
    with at least two classes. Not supported: multilabel targets, ``solver='lbfgs'``,
    ``early_stopping=True``, `warm_start`, `partial_fit` and `verbose`.

    Parameters
    ----------
    hidden_layer_sizes : int or tuple of int, default (100,)
        Width of each hidden layer; every width must be > 0.
    activation : {'identity', 'logistic', 'tanh', 'relu'}, default 'relu'
        Hidden-layer activation.
    solver : {'adam', 'sgd'}, default 'adam'
        Optimizer. 'lbfgs' raises `NotImplementedError`.
    alpha : float, default 0.0001
        L2 penalty strength (>= 0).
    batch_size : int or 'auto', default 'auto'
        Minibatch size; 'auto' is ``min(200, n_samples)``.
    learning_rate : {'constant', 'invscaling', 'adaptive'}, default 'constant'
        Learning-rate schedule of the 'sgd' solver, as scikit-learn.
    learning_rate_init : float, default 0.001
        Initial learning rate (> 0).
    power_t : float, default 0.5
        Exponent of the 'invscaling' schedule.
    max_iter : int, default 200
        Maximum number of epochs (>= 1).
    shuffle : bool, default True
        Shuffle the rows every epoch.
    random_state : int or None, default None
        Seed of the weight initialization and the shuffle.
    tol : float, default 1e-4
        Tolerance of the stopping rule.
    verbose : bool, default False
        Must be False; True raises `NotImplementedError`.
    warm_start : bool, default False
        Must be False; True raises `NotImplementedError`.
    momentum : float, default 0.9
        Momentum of the 'sgd' solver.
    nesterovs_momentum : bool, default True
        Use Nesterov momentum with the 'sgd' solver.
    early_stopping : bool, default False
        Must be False; True raises `NotImplementedError`.
    validation_fraction : float, default 0.1
        Accepted for compatibility; unused because early stopping is not supported.
    beta_1 : float, default 0.9
        Adam's first-moment decay.
    beta_2 : float, default 0.999
        Adam's second-moment decay.
    epsilon : float, default 1e-8
        Adam's denominator term.
    n_iter_no_change : int, default 10
        Epochs without `tol` improvement before stopping.
    max_fun : int, default 15000
        Accepted for compatibility (lbfgs only); unused.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode. None uses the library default. ``'fast'`` is available on Apple GPUs only.

    Attributes
    ----------
    classes_ : ndarray
        Sorted class labels.
    coefs_ : list of ndarray
        Weight matrix of each layer, (n_in, n_out).
    intercepts_ : list of ndarray
        Bias vector of each layer.
    loss_ : float
        Loss of the last epoch.
    best_loss_ : float
        Lowest loss seen.
    loss_curve_ : list of float
        Loss of every epoch.
    n_iter_ : int
        Number of epochs run.
    t_ : int
        Number of training samples seen.
    n_features_in_ : int
        Number of input features.
    n_layers_ : int
        Number of layers, input and output included.
    n_outputs_ : int
        Number of outputs.
    out_activation_ : str
        Output activation.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> y = (X[:, 0] > 0.5).astype(np.int64)
    >>> model = mojolearn.MLPClassifier(hidden_layer_sizes=(32,), max_iter=50).fit(X, y)
    >>> proba = model.predict_proba(X)
    """
    # scikit-learn tags, read by `_mode.sklearn_tags`
    _estimator_type = "classifier"

    def fit(self, X, y):
        """Train on `X` (n_samples, n_features) float32 and 1-D labels `y`; returns self."""
        return _XSEQ(self, "mlp_fit", (X, y), _XSEQ_CONTEXT)

    def predict_proba(self, X):
        """Class probabilities, (n_samples, n_classes), columns in `classes_` order."""
        return _XSEQ(self, "mlp_predict_proba", (X,), _XSEQ_CONTEXT)

    def predict(self, X):
        """Predicted class labels."""
        return _XSEQ(self, "mlp_predict", (X,), _XSEQ_CONTEXT)

    def score(self, X, y, sample_weight=None):
        """Mean accuracy of ``predict(X)`` against ``y``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
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
        return _XSEQ(self, "mlp_classifier_score", (X, y, sample_weight), _XSEQ_CONTEXT)


_native_xseq_bridge.register(
    xseq_mlp_activations=_ACT,
    xseq_mlp_solvers=_SOLVER,
    xseq_mlp_rates=_LR,
    xseq_mlp_outputs={"identity": 0, "logistic": 1, "tanh": 2, "relu": 3, "softmax": 4},
)

# Last statement: connect this module's classes to the native full-reset
# set_params now that they exist. The package imports without NumPy, so it
# cannot connect them at `import mojolearn`.
from ._native_full_reset_params import connect_module as _connect_full_reset
_connect_full_reset(globals())
del _connect_full_reset
