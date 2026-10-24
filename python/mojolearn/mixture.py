# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Maintainer notes (moved from the module docstring; workstream D, 2026-09-14):
# The Python door of `mixture/estimator.mojo` through
# `bindings/_mojolearn_mixture.mojo`. There is no GPU reference: cuML, cuVS and
# RAFT ship no Gaussian mixture at the pinned commits, so scikit-learn's
# `_gaussian_mixture.py` and `_base.py` are the semantics and the oracle, never
# the design source. `mixture/README.md` carries the parameter mapping.
# The default binding implements covariance_type='full', init_params 'kmeans'
# (through the identity-certified k-means) and 'random' (position-mapped Philox
# draws, DEVIATION 1733), n_init=1, warm_start=False; every other option is
# routed to the cluster lane's mixture driver (bindings/mixture_native.mojo `_gmm_ext_fit`).
# A collapsed component raises (DEVIATION 1723). No speed claim.
"""Gaussian mixture models.

`GaussianMixture` fits a mixture of Gaussians by expectation-maximization,
with scikit-learn's parameter names and meanings.
"""
from . import _native_archive_bridge, _native_family_bridge
from ._mode import NumericModeMixin

_MODE_CODE = {"fast": 0, "identical": 1}

# The compiled family owners, connected once at import: fit, fit_predict,
# predict, predict_proba, score_samples, score, bic and aic of
# GaussianMixture and BayesianGaussianMixture (bindings/mixture_native.mojo)
# and `sample` (bindings/svm_mixture_native.mojo) are one native call each,
# handed the binding `_bind` resolves. The covariance-type and init tables
# are the option-parity solver's codes (x_cluster/entries.mojo::bgmm_entry).
_NATIVE, _NATIVE_CONTEXT = _native_family_bridge.connect()
_native_family_bridge.tables(
    family_gmm_mode_codes=dict(_MODE_CODE),
    family_mix_ctypes={"full": 0, "tied": 1, "diag": 2, "spherical": 3},
    family_mix_inits={"kmeans": 0, "random": 1, "k-means++": 2, "random_from_data": 3},
)

#: The model file (the neighbors and density inference lane, 2026-09-15):
#: `weights`, `means`, `covariances`, `precisions_cholesky` and
#: `log_det_chol` as flat `<f4` in the order the scoring entries address
#: them, `ints` `<i8` [n_components, n_features, n_iter_, converged_,
#: max_iter, random_state], `reals` `<f8` [lower_bound_, tol, reg_covar],
#: `covariance_type` and `init_params` as text.
_GMM_FORMAT = "mojolearn-gmm-1"

# The compiled saved-model owner, connected once at import: `save` and
# `load` are one native call each (bindings/archive_schema_native.mojo
# writes the schema registered below the class; host_archive_native.mojo
# builds the loaded model).
_ARCHIVE, _ARCHIVE_CONTEXT = _native_archive_bridge.connect()

# log_det_chol_ is `_compute_log_det_cholesky`'s answer from the device fold
# (DEVIATION 1726), never re-derived from the diagonal in Python.
# converged_=False: scikit-learn warns, this reports (DEVIATION 1746).
class GaussianMixture(NumericModeMixin):
    """Gaussian mixture model fitted by expectation-maximization.

    Follows scikit-learn's `GaussianMixture`: every parameter has the same
    name, meaning and default. Iterates until the change in the lower bound
    is below `tol` or `max_iter` is reached. `X` is read as float32 and the
    fitted arrays are float32.

    The default configuration (`covariance_type='full'`,
    `init_params='kmeans'` or `'random'`, `n_init=1`, no warm start and no
    initial arrays) runs on mojolearn's main mixture solver. Any other
    option runs on a second solver; a model fitted that way can score and
    predict, but `save` and `sample` raise `NotImplementedError`.

    Parameters
    ----------
    n_components : int, default 1
        Number of mixture components.
    covariance_type : {'full', 'tied', 'diag', 'spherical'}, default 'full'
        Form of each component's covariance, as in scikit-learn.
    tol : float, default 1e-3
        Convergence threshold on the change in the lower bound.
    reg_covar : float, default 1e-6
        Added to the diagonal of each covariance.
    max_iter : int, default 100
        Maximum number of EM iterations.
    init_params : {'kmeans', 'random', 'k-means++', 'random_from_data'}, default 'kmeans'
        How the responsibilities are initialized.
    random_state : int, default 0
        Seed for initialization and for `sample`.
    n_init : int, default 1
        Number of initializations; the best one is kept.
    warm_start : bool, default False
        Start from the previous fit's solution when the shape and covariance
        type match.
    means_init : array-like of shape (n_components, n_features), default None
        Initial means.
    weights_init : array-like of shape (n_components,), default None
        Initial weights.
    precisions_init : array-like, default None
        Initial precisions, shaped by `covariance_type` as in scikit-learn.

    Attributes
    ----------
    weights_ : Array of shape (n_components,), float32
        Weight of each component.
    means_ : Array of shape (n_components, n_features), float32
        Mean of each component.
    covariances_ : Array, float32
        Covariance of each component, shaped by `covariance_type`; for 'full',
        `(n_components, n_features, n_features)`. Includes `reg_covar` on
        the diagonal, as scikit-learn's does.
    precisions_cholesky_ : Array, float32
        Cholesky factors of the precision matrices, shaped like
        `covariances_`.
    log_det_chol_ : Array of shape (n_components,), float32
        Log-determinant of each precision Cholesky factor.
    n_iter_ : int
        Number of EM iterations run.
    converged_ : bool
        False when the fit stopped at `max_iter` before reaching `tol`.
        scikit-learn also warns in that case; mojolearn only sets this flag.
    lower_bound_ : float
        Lower bound on the log likelihood at the end of the fit.
    n_features_in_ : int
        Number of features seen during `fit`.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> gmm = mojolearn.GaussianMixture(n_components=3).fit(X)
    >>> labels = gmm.predict(X)
    >>> proba = gmm.predict_proba(X)
    >>> X_new, y_new = gmm.sample(100)
    """
    # scikit-learn tags, read by `_mode.sklearn_tags`
    _estimator_type = "density_estimator"

    _BINDING = "_mojolearn_mixture"

    def __init__(
        self,
        n_components=1,
        covariance_type="full",
        tol=1e-3,
        reg_covar=1e-6,
        max_iter=100,
        init_params="kmeans",
        random_state=0,
        n_init=1,
        warm_start=False,
        means_init=None,
        weights_init=None,
        precisions_init=None,
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.init_params = init_params
        self.random_state = random_state
        self.n_init = n_init
        self.warm_start = warm_start
        self.means_init = means_init
        self.weights_init = weights_init
        self.precisions_init = precisions_init

    # EM runs one shot on the device (`gaussian_mixture_fit`). Option parity
    # (2026-09-27, lane/algos-cluster): any option this binding does not carry
    # runs the same EM on the cluster lane's mixture driver
    # (`x_cluster/bgmm.mojo`, plain mode; bindings/mixture_native.mojo `_gmm_ext_fit`).
    # The default fit is untouched and keeps its recorded bits.
    def _extension(self):
        """This object's tier of the mixture binding, its compile-time mode
        read back (glue: binding choice). The parallel driver's workers
        (the multi-GPU driver's native `family_par_run`) reach the binding through it."""
        from . import _backend
        mod = self._bind()
        want = getattr(self, "numeric_mode", None) or _backend.default_mode()
        fn = getattr(mod, "mixture_numeric_mode", None)
        if fn is not None and int(fn()) != _MODE_CODE.get(want):
            raise RuntimeError(
                f"mojolearn GaussianMixture: numeric_mode={want!r} was requested "
                f"but {mod.__name__} reports another compile-time mode; "
                "rebuild it with bash bindings/build_mixture.sh")
        return mod

    def fit(self, X, y=None):
        """Fit the mixture to `X` by expectation-maximization.

        Runs until `abs(lower_bound - previous) < tol` or `max_iter`
        iterations.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, read as float32.
        y : ignored
            Present for scikit-learn API compatibility.

        Returns
        -------
        self : GaussianMixture
            The fitted estimator.

        An unknown `covariance_type` or `init_params`, or a component that
        collapses during the fit, raises an error naming the problem; the fit
        never returns a model with a reset component.

        Raises
        ------
        ValueError
            For a non-positive `n_components` or initial arrays of the wrong
            shape.
        TypeError
            If `n_components` is not an int or `tol`/`reg_covar` is not a
            number.
        """
        return _NATIVE(self, "family_mix_gmm_fit", (self._bind(), self._bind("_mojolearn_x_cluster"), X), _NATIVE_CONTEXT)


    def score_samples(self, X):
        """Return the log likelihood of each row of `X` under the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        log_likelihood : Array of shape (n_samples,), float32
        """
        return _NATIVE(self, "family_mix_gmm_score_samples", (self._bind(), X), _NATIVE_CONTEXT)

    def predict_proba(self, X):
        """Return the posterior probability of each component for each row.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        proba : Array of shape (n_samples, n_components), float32
        """
        return _NATIVE(self, "family_mix_gmm_predict_proba", (self._bind(), X), _NATIVE_CONTEXT)

    def predict(self, X):
        """Return the most likely component for each row of `X`.

        A tie goes to the lowest component index.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        labels : Array of shape (n_samples,), int32
        """
        return _NATIVE(self, "family_mix_gmm_predict", (self._bind(), X), _NATIVE_CONTEXT)

    def fit_predict(self, X, y=None):
        """Fit the model, then return `predict(X)`.

        Returns
        -------
        labels : Array of shape (n_samples,), int32
        """
        return _NATIVE(self, "family_mix_gmm_fit_predict", (self._bind(), self._bind("_mojolearn_x_cluster"), X), _NATIVE_CONTEXT)

    def save(self, path):
        """Save the fitted model to `path` as an `.npz` file.

        The file holds every fitted array plus `n_iter_`, `converged_` and
        `lower_bound_`. Load it with `load`, or with
        `mojolearn.host_model(path)` to score on a machine with no GPU.

        Parameters
        ----------
        path : str or path-like
            Destination file.

        Raises
        ------
        RuntimeError
            If the model is not fitted.
        NotImplementedError
            If the model was fitted with options outside the default
            configuration (see the class description).
        """
        return _ARCHIVE(self, "archive_save", (path,), _ARCHIVE_CONTEXT)

    @classmethod
    def load(cls, path):
        """Load a model written by `save`.

        Arrays keep their saved dtypes. The result supports `score_samples`,
        `predict_proba`, `predict`, `score`, `bic` and `aic`.

        Parameters
        ----------
        path : str or path-like
            File written by `GaussianMixture.save`.

        Returns
        -------
        model : GaussianMixture

        Raises
        ------
        ValueError
            If the file is of another format or estimator, or is truncated or
            inconsistent.
        """
        return _ARCHIVE(cls, "host_archive_load", (path,), _ARCHIVE_CONTEXT)


    def score(self, X, y=None):
        """Return the mean log likelihood of the rows of `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : ignored
            Present for scikit-learn API compatibility.

        Returns
        -------
        score : float
        """
        return _NATIVE(self, "family_mix_gmm_score", (self._bind(), X), _NATIVE_CONTEXT)

    def bic(self, X):
        """Return the Bayesian information criterion of the model on `X`.

        Lower is better.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        bic : float
        """
        return _NATIVE(self, "family_mix_gmm_bic", (self._bind(), X), _NATIVE_CONTEXT)

    def aic(self, X):
        """Return the Akaike information criterion of the model on `X`.

        Lower is better.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        aic : float
        """
        return _NATIVE(self, "family_mix_gmm_aic", (self._bind(), X), _NATIVE_CONTEXT)

    # Reference: scikit-learn `BaseMixture.sample`. Draws are position-mapped
    # Philox keyed by `random_state` (DEVIATION 2791, `mixture/checks/sample.mojo`);
    # the normals go through the fitted `precisions_cholesky_` (DEVIATION 2792).
    def sample(self, n_samples=1):
        """Draw random rows from the fitted mixture.

        As in scikit-learn, the number of rows per component is a multinomial
        draw over `weights_`, and the rows come out grouped by component in
        ascending order. The draws are keyed by `random_state`, so the same
        model and `random_state` give the same rows on every call and every
        GPU vendor; they differ from scikit-learn's draws. `X` is float32 and
        `y` int32, where scikit-learn returns float64 and int64.

        Parameters
        ----------
        n_samples : int, default 1
            Number of rows to draw. Must be at least 1.

        Returns
        -------
        X : Array of shape (n_samples, n_features), float32
            The drawn rows.
        y : Array of shape (n_samples,), int32
            The component each row was drawn from.

        Raises
        ------
        ValueError
            If the model is not fitted or `random_state` is not an int in
            `[0, 2**64)`.
        TypeError
            If `n_samples` is not an int.
        NotImplementedError
            If the model was fitted with options outside the default
            configuration.
        """
        return _NATIVE(self, "family_svm_gmm_sample", (self._bind(), n_samples), _NATIVE_CONTEXT)


# The saved model, written natively from this schema (bindings/
# archive_schema_native.mojo) with the bytes `_serialize.write_npz` wrote;
# the refusals keep the wording and order of the former Python save.
_native_archive_bridge.register(GaussianMixture, _GMM_FORMAT, (
    _native_archive_bridge.fitted("weights_"),
    ("refuse_set", "_ext", "NotImplementedError",
     "mojolearn GaussianMixture: save() carries the default fit's model format only; a fit with the "
     "options routed to x_cluster is not saved (x_cluster/NOT_IMPLEMENTED.tsv)"),
    ("need", "means_", "AttributeError", "'{cls}' object has no attribute 'means_'"),
    *_native_archive_bridge.head(_GMM_FORMAT),
    ("str", "covariance_type", "covariance_type"),
    ("str", "init_params", "init_params"),
    ("array", "weights", "weights_"),
    ("flat", "means", "means_"),
    ("flat", "covariances", "covariances_"),
    ("flat", "precisions_cholesky", "precisions_cholesky_"),
    ("array", "log_det_chol", "log_det_chol_"),
    ("i8", "ints", (("dim", "means_", 0), ("dim", "means_", 1), ("int", "n_iter_"), ("bool", "converged_"),
                    ("int", "max_iter"), ("int", "random_state"))),
    ("f8", "reals", (("float", "lower_bound_"), ("float", "tol"), ("float", "reg_covar"))),
))


__all__ = ["GaussianMixture"]


# scikit-learn keeps more estimators in this module than it defines; they resolve from
# the package at first use (`_sklearn_modules`), so `from mojolearn.mixture import ...`
# takes the same names as scikit-learn's module.
from ._sklearn_modules import LAYOUT as _SKLEARN_LAYOUT, lazy_exports as _lazy_exports  # noqa: E402
__getattr__ = _lazy_exports("mixture")
__all__ += _SKLEARN_LAYOUT["mixture"]
