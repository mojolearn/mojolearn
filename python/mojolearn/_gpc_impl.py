# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Gaussian process classification: the Laplace approximation.

PRIVATE MODULE. `GaussianProcessClassifier` is re-exported from
`mojolearn/__init__.py`; `HostGaussianProcessClassifier` is what
`mojolearn.host_model` returns for a saved classifier.

THE REFERENCE is scikit-learn 1.9.0 `sklearn/gaussian_process/_gpc.py`:
`_BinaryGaussianProcessClassifierLaplace` (`_gpc.py:36-513`, Rasmussen and
Williams Algorithms 3.1 and 3.2 with the logistic likelihood) and
`GaussianProcessClassifier` (`_gpc.py:516` on), which wraps the binary
estimator in `OneVsRestClassifier` past two classes. Every name here means
what scikit-learn means by it.

WHERE THE ARITHMETIC IS. One binary Laplace fit per binding call,
`gpc_fit` and `gpc_predict` on `_mojolearn_gp` (the GPU binding,
`gaussian_process/classifier.mojo`) or on `_mojolearn_gp_host` (the CPU host
binding, `gaussian_process/host/gpc_oracle.mojo`); the per-row steps of
both are `gaussian_process/host/gpc_steps.mojo`, which carries DEVIATIONS
2830 (the stop rule), 2831 (the pinned float32 orders) and 2832 (the
float64 probability and its erf). This file holds the class encoding, the
one-vs-rest composition (DEVIATION 2833) and the saved model.

DEVIATION 2833, THE ONE-VS-REST COMPOSITION, in plain Python float64 (every
value it reads is a float64 or an exact widening of a float32, and CPython
float arithmetic is IEEE double with no contraction, so it is the same on
every host):
  - column k of a K-class fit is the binary target `y == classes_[k]`, the
    `LabelBinarizer` column `OneVsRestClassifier` fits (`multiclass.py`);
  - `predict_proba` divides each row by its sum where the sum is not zero
    (`multiclass.py:559-560`), the sum taken over k ASCENDING from +0.0
    (NumPy's reduction order may differ in the last bits);
  - `predict` is the first k whose unnormalized class-k probability is
    strictly largest, which is NumPy's argmax tie rule that
    `OneVsRestClassifier.predict` reproduces (`multiclass.py:499-508`); a
    binary model predicts `classes_[1]` where the float32 latent mean is
    strictly positive (`_gpc.py:287-290`);
  - `log_marginal_likelihood_value_` is the mean of the K binary values
    (`_gpc.py:216-222`), summed ascending from +0.0 then divided by K.

WHAT IS HONORED AND WHAT IS REFUSED, one line per parameter:

    kernel                honored   RBF, Matern (nu in {0.5, 1.5, 2.5}),
                                    ConstantKernel and WhiteKernel via + and *;
                                    None is scikit-learn's default
                                    ConstantKernel(1.0) * RBF(1.0)
    optimizer             refused   anything but None (DEVIATION 1761). THE
                                    DEFAULT IS None, NOT 'fmin_l_bfgs_b', so
                                    the fitted kernel_ is the kernel passed
    n_restarts_optimizer  refused   anything but 0; it restarts the optimizer
    max_iter_predict      honored   at least 1; the Newton loop's cap
                                    (DEVIATION 2830 says how it stops early)
    warm_start            refused   True: it starts the Newton loop from the
                                    previous fit's latent values, so a second
                                    fit would depend on the first
    copy_X_train          refused   False: X always crosses to float32 memory
    random_state          accepted  unused: only the refused optimizer
                                    restarts draw random numbers
    multi_class           honored   'one_vs_rest'; 'one_vs_one' is refused BY
                                    NAME (not implemented, and it has no
                                    predict_proba in the reference either)
    n_jobs                refused   anything but None; the binary fits run
                                    one after another on one device

THE INFERENCE BOUNDARY.
`fit` on a CPU-only install refuses unless the internal verifier's
`reference_training()` is active. `predict`, `predict_proba`,
`latent_mean_and_variance` and `load` are public on the CPU:
`GaussianProcessClassifier.load(path)` routes `_mojolearn_gp` to
`_mojolearn_gp_host` on a CPU-only install, and `mojolearn.host_model(path)`
returns `HostGaussianProcessClassifier`, bound to the host binding on any
box.
"""

from . import _backend, _native_archive_bridge, _native_family_bridge
import numbers

from ._gp_impl import _KERNEL_NODES, _MODE_CODE, Kernel
from ._mode import NumericModeMixin

#: The saved-model format tag.
_GPC_FORMAT = "mojolearn.gpc.v1"
_ESTIMATOR = "GaussianProcessClassifier"
_NAME = "mojolearn GaussianProcessClassifier"

# The compiled saved-model owner, connected once at import: `save` and
# `load` are one native call each (bindings/archive_schema_native.mojo
# writes the schema registered below the class; host_archive_native.mojo
# builds the loaded model).
_ARCHIVE, _ARCHIVE_CONTEXT = _native_archive_bridge.connect()
# The compiled owner of the public methods, connected once at import: fit,
# predict, predict_proba and latent_mean_and_variance are one native call
# each (bindings/gp_classifier_native.mojo). fit and predict also hand over
# the base binding, whose label entries encode and decode the classes.
_NATIVE, _NATIVE_CONTEXT = _native_family_bridge.connect()
_native_family_bridge.connect_bindings("_mojolearn_gp", "_mojolearn")






class _SavedKernel(Kernel):
    """A kernel read back from a model file: the postfix nodes exactly as
    they were saved (float32 values round-trip through Python floats
    exactly), so a loaded model sends the binding the bytes the fit sent."""

    def __init__(self, nodes):
        self._node_list = [(int(k), float(p), [float(v) for v in ls]) for k, p, ls in nodes]  # glue: copies the kernel's postfix nodes

    def _nodes(self):
        return [(k, p, list(ls)) for k, p, ls in self._node_list]  # glue: copies the kernel's postfix nodes

    def _name(self):
        return f"SavedKernel({len(self._node_list)} postfix nodes)"


class _BinaryLaplace:
    """One binary Laplace fit, the state `_BinaryGaussianProcessClassifierLaplace`
    keeps (`_gpc.py:263-265`): the 0/1 targets `y_train_`, `pi_`, `W_sr_`,
    `L_` (the lower factor of `B = I + W_sr K W_sr`), the likelihood, and the
    Newton iteration count and panel width that ran."""

    def __init__(self, y_train, L, pi, W_sr, lml, n_iter, nb):
        self.y_train_ = y_train
        self.L_ = L
        self.pi_ = pi
        self.W_sr_ = W_sr
        self.log_marginal_likelihood_value_ = lml
        self.n_iter_ = n_iter
        self.nb_ = nb


# Maintainer notes: reference is scikit-learn 1.9.0 `_gpc.py` (binary
# Laplace `_gpc.py:36-513`, R&W Algorithms 3.1/3.2, logistic likelihood;
# OneVsRest wrapper `_gpc.py:516` on). Stop rule DEVIATION 2830, pinned
# float32 orders 2831, float64 probability and erf 2832, one-vs-rest
# composition 2833 (see the module docstring). optimizer refused per
# DEVIATION 1761: an optimizer's iteration count is data dependent and its
# line search, gradient fold and tolerance are not pinned for this estimator.
class GaussianProcessClassifier(NumericModeMixin):
    """Gaussian process classification by the Laplace approximation.

    Follows scikit-learn's ``GaussianProcessClassifier``: a latent Gaussian
    process with a logistic likelihood, fitted by Newton iteration (Rasmussen
    and Williams, Algorithms 3.1 and 3.2). Two classes give one binary model;
    more classes give one binary model per class against the rest, with
    probabilities normalized per row as in scikit-learn's one-vs-rest
    wrapper. X is dense float32 in C order (other inputs are converted).

    Differences from scikit-learn that change results:

    - There is no hyperparameter optimizer. ``optimizer`` must be None (the
      default here; scikit-learn's is ``"fmin_l_bfgs_b"``), so ``kernel_``
      is the kernel you pass.
    - Every fit starts from zero latent values; ``warm_start=True`` is
      refused.

    Parameters
    ----------
    kernel : Kernel or None, default None
        A combination of `RBF`, `Matern`, `ConstantKernel` and `WhiteKernel`
        with ``+`` and ``*``. None means ``ConstantKernel(1.0) * RBF(1.0)``.
        Other kernel types raise ``TypeError``.
    optimizer : None, default None
        Only None (or ``"none"``) is accepted; anything else raises
        ``NotImplementedError``.
    n_restarts_optimizer : int, default 0
        Only 0 is accepted (``NotImplementedError`` otherwise).
    max_iter_predict : int, default 100
        Maximum number of Newton iterations per binary fit; the iteration
        can stop earlier on convergence. Must be an int (``TypeError``) of at
        least 1 (``ValueError``).
    warm_start : bool, default False
        Only False is accepted (``NotImplementedError`` otherwise).
    copy_X_train : bool, default True
        Only True is accepted: X is always copied to float32 memory.
    random_state : int, RandomState instance or None, default None
        Accepted for compatibility; nothing in the fit is random.
    multi_class : "one_vs_rest", default "one_vs_rest"
        ``"one_vs_one"`` raises ``NotImplementedError``; any other value
        raises ``ValueError``.
    n_jobs : None, default None
        Only None is accepted; the binary fits run on one device.
    numeric_mode : str or None, default None
        The numeric mode for this estimator; None uses the process default.
        See the numeric modes guide.

    Attributes
    ----------
    classes_ : list
        The class labels, in sorted order.
    n_classes_ : int
        Number of classes.
    X_train_ : Array of shape (n_train, n_features), float32
        The training inputs.
    estimators_ : list
        The binary fits: one for two classes, one per class past two. Each
        has ``y_train_``, ``L_``, ``pi_``, ``W_sr_``,
        ``log_marginal_likelihood_value_`` and ``n_iter_``.
    y_train_, L_, pi_, W_sr_ : Array
        Two classes only: the 0/1 targets, the lower Cholesky factor of
        ``I + W_sr K W_sr``, the training probabilities and the square root
        of the likelihood Hessian of the single binary fit.
    log_marginal_likelihood_value_ : float
        The binary fit's log marginal likelihood, or the mean over the
        binary fits past two classes.
    n_iter_ : int or list of int
        Newton iterations run: an int for two classes, one int per class
        past two.
    kernel_ : Kernel
        The kernel passed in.
    n_features_in_ : int
        Number of features seen during fit.
    input_copied_ : bool
        Whether fit had to copy X to reach float32 C order.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((200, 4), dtype=np.float32)
    >>> y = (X[:, 0] + X[:, 1] > 1.0).astype(np.int32)
    >>> model = mojolearn.GaussianProcessClassifier(mojolearn.RBF(0.5)).fit(X, y)
    >>> proba = model.predict_proba(X[:5])
    """

    #: scikit-learn's estimator kind: `cross_val_score` stratifies a
    #: classifier's default folds, as scikit-learn's does.
    _estimator_type = "classifier"

    _BINDING = "_mojolearn_gp"

    # The native public owner replaces _mode lifecycle/output decorators.
    # The shared full-reset installer still owns the field-only constructor.
    _mojolearn_native_public = True

    def __init__(
        self,
        kernel=None,
        *,
        optimizer=None,
        n_restarts_optimizer=0,
        max_iter_predict=100,
        warm_start=False,
        copy_X_train=True,
        random_state=None,
        multi_class="one_vs_rest",
        n_jobs=None,
        numeric_mode=None,
    ):
        # Stored verbatim, as scikit-learn requires; `_validate_params`
        # checks them when `fit` (or `load`) runs.
        self.kernel = kernel
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.max_iter_predict = max_iter_predict
        self.warm_start = warm_start
        self.copy_X_train = copy_X_train
        self.random_state = random_state
        self.multi_class = multi_class
        self.n_jobs = n_jobs
        self.numeric_mode = numeric_mode


    # -- the binding ----------------------------------------------------------

    def _extension(self):
        """`_mojolearn_gp` (or the host binding) with its compile-time tier
        cross-checked against the requested one, `_gp_impl.py`'s rule."""
        mod = self._bind()
        want = getattr(self, "numeric_mode", None) or _backend.default_mode()
        fn = getattr(mod, "gp_numeric_mode", None)
        if fn is not None:
            got = int(fn())
            if got != _MODE_CODE.get(want):
                raise RuntimeError(
                    f"{_NAME}: numeric_mode={want!r} was requested but "
                    f"{mod.__name__} reports compile-time mode code {got}; "
                    "rebuild it with bash bindings/build_gp.sh"
                )
        return mod

    # -- fit ------------------------------------------------------------------

    # Reference: sklearn `_gpc.py:164-226`.
    def fit(self, X, y):
        """Fit the classifier.

        Encodes the classes, then runs one binary Laplace fit for two classes
        or one per class against the rest.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training inputs, converted to float32.
        y : array-like of shape (n_samples,)
            Class labels; at least two distinct classes (``ValueError``
            otherwise).

        Returns
        -------
        self : GaussianProcessClassifier
        """
        return _NATIVE(self, "family_gp_gpc_fit", (None, None, X, y), _NATIVE_CONTEXT)



    # -- prediction -------------------------------------------------------------



    # Reference: sklearn `_gpc.py:410-441`, binary only as `_gpc.py:873-878`.
    def latent_mean_and_variance(self, X):
        """Mean and variance of the latent function at ``X``.

        Binary models only, as in scikit-learn; a model with more than two
        classes raises ``ValueError``.

        Parameters
        ----------
        X : array-like of shape (n_query, n_features)
            Query points, converted to float32.

        Returns
        -------
        mean : Array of shape (n_query,), float32
        var : Array of shape (n_query,), float32
        """
        return _NATIVE(self, "family_gp_gpc_latent", (None, X), _NATIVE_CONTEXT)

    # Reference: sklearn `_gpc.py:292-329` and `multiclass.py:523-562`
    # (DEVIATIONS 2832 and 2833).
    def predict_proba(self, X):
        """Class probabilities for ``X``.

        Past two classes, each row of the per-class probabilities is divided
        by its sum, as in scikit-learn's one-vs-rest wrapper.

        Parameters
        ----------
        X : array-like of shape (n_query, n_features)
            Query points, converted to float32.

        Returns
        -------
        Array of shape (n_query, n_classes), float64
            Columns follow ``classes_``.
        """
        return _NATIVE(self, "family_gp_gpc_predict_proba", (None, X), _NATIVE_CONTEXT)

    # Reference: sklearn `_gpc.py:287-290`; multiclass argmax per DEVIATION 2833.
    def predict(self, X):
        """Predict class labels for ``X``.

        With two classes, ``classes_[1]`` where the latent mean is positive
        and ``classes_[0]`` elsewhere. Past two, the class with the largest
        one-vs-rest probability; a tie goes to the first such class.

        Parameters
        ----------
        X : array-like of shape (n_query, n_features)
            Query points, converted to float32.

        Returns
        -------
        labels : Array or list of shape (n_query,)
            Labels drawn from ``classes_``: an int64 or float64 Array for
            numeric classes, a list for string or bool classes.
        """
        return _NATIVE(self, "family_gp_gpc_predict", (None, None, X), _NATIVE_CONTEXT)

    # Reference: sklearn `_gpc.py:279-353`, the `theta is None` arm only.
    def log_marginal_likelihood(self, theta=None, eval_gradient=False, clone_kernel=True):
        """Log marginal likelihood of the fitted model.

        Only ``theta=None`` is supported: it returns
        ``log_marginal_likelihood_value_``. A theta raises
        ``NotImplementedError`` and ``eval_gradient=True`` raises
        ``ValueError``.

        Parameters
        ----------
        theta : None, default None
        eval_gradient : bool, default False
        clone_kernel : bool, default True
            Accepted for scikit-learn compatibility; ignored.

        Returns
        -------
        float
        """
        return _NATIVE(self, "family_gp_gpc_lml", (theta, eval_gradient, clone_kernel), _NATIVE_CONTEXT)

    # A host summary outside the identity claim.
    def score(self, X, y):
        """Mean accuracy of `predict` on ``X`` against ``y``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test inputs.
        y : array-like of shape (n_samples,)
            True labels.

        Returns
        -------
        float
        """
        return _NATIVE(self, "family_gp_gpc_score", (None, None, X, y),
                       _NATIVE_CONTEXT)

    # -- save and load ----------------------------------------------------------

    def save(self, path):
        """Save the fitted model to ``path`` as an ``.npz`` file.

        The file holds everything prediction reads, byte for byte: the
        kernel, the float32 training inputs, the classes, the numeric mode
        and each binary fit's state. Raises ``RuntimeError`` if the model is
        not fitted.

        Parameters
        ----------
        path : str or path-like
            Destination file.
        """
        return _ARCHIVE(self, "archive_save", (path,), _ARCHIVE_CONTEXT)

    @classmethod
    def load(cls, path):
        """Load a model written by `save`.

        The loaded model supports `predict`, `predict_proba` and
        `latent_mean_and_variance`, including on a CPU-only install. Every
        array is read at its saved dtype and never cast; a file saved by a
        different estimator or with inconsistent contents raises
        ``ValueError``.

        Parameters
        ----------
        path : str or path-like
            File written by `save`.

        Returns
        -------
        GaussianProcessClassifier
        """
        return _ARCHIVE(cls, "host_archive_load", (path,), _ARCHIVE_CONTEXT)


# The file: the kernel as `_kernel_arrays` packs it, the float32 training
# inputs, the classes as `_labels.classes_member` wrote them, `meta` <i8
# [n_train, n_features, n_estimators, max_iter_predict, n_ls, n_classes] and
# each binary fit's state stacked in class order.
_GPC_KERNEL_NODES = dict(_KERNEL_NODES)
_GPC_KERNEL_NODES[_SavedKernel] = ("nodes", "_node_list")
_native_archive_bridge.register(GaussianProcessClassifier, _GPC_FORMAT, (
    _native_archive_bridge.fitted("estimators_", f"{_NAME}: call fit before save"),
    ("format", "format", _GPC_FORMAT),
    ("format", "estimator", _ESTIMATOR),
    ("mode", "numeric_mode", f"{_NAME}: "),
    ("kernel", "_kernel", _GPC_KERNEL_NODES, "<f4", "kinds", "kparams", "ls_len", "ls"),
    ("class_table", "classes", "classes_", numbers.Integral, numbers.Real),
    ("array", "x", "X_train_"),
    ("i8", "meta", (("dim", "X_train_", 0), ("dim", "X_train_", 1), ("len", "estimators_"),
                    ("int", "_max_iter_predict"), ("kernel_ls", "_kernel", _GPC_KERNEL_NODES),
                    ("int", "n_classes_"))),
    ("stack", "y", "estimators_", "y_train_", "<f4", ("dim", "X_train_", 0)),
    ("stack", "L", "estimators_", "L_", "<f4", ("mul", ("dim", "X_train_", 0), ("dim", "X_train_", 0))),
    ("stack", "pi", "estimators_", "pi_", "<f4", ("dim", "X_train_", 0)),
    ("stack", "wsr", "estimators_", "W_sr_", "<f4", ("dim", "X_train_", 0)),
    ("each", "lml", "estimators_", "log_marginal_likelihood_value_", "<f8"),
    ("each", "n_iter", "estimators_", "n_iter_", "<i8"),
    ("each", "nb", "estimators_", "nb_", "<i8"),
), host_archive_gpc_saved_kernel=_SavedKernel, host_archive_binary_laplace=_BinaryLaplace)

# The native owner's tables: every kernel node layout (the regressor's and
# this file's saved kernel), the binary fit class and the default bounds.
_native_family_bridge.tables(family_gp_kernel_nodes=dict(_GPC_KERNEL_NODES),
                             family_gp_binary_laplace=_BinaryLaplace,
                             family_gp_binary_namespace=dict(vars(_BinaryLaplace)),
                             family_gp_default_bounds=(1e-5, 1e5))


# Import-time connections for native completed-fit labels and archive reads.
# Family and archive owners have separate contexts; neither inherits the tree
# or logistic owner's ClassLabels class/namespace metadata.
from ._labels import ClassLabels as _ClassLabels
_GP_LABELS_METADATA = dict(
    classifier_labels_class=_ClassLabels,
    classifier_labels_namespace=dict(vars(_ClassLabels)),
    classifier_labels_mro=type.__dict__["__mro__"].__get__(_ClassLabels),
)
_native_family_bridge.tables(**_GP_LABELS_METADATA)
_ARCHIVE_CONTEXT.update(_GP_LABELS_METADATA)
del _ClassLabels, _GP_LABELS_METADATA


__all__ = ["GaussianProcessClassifier"]
