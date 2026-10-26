# SPDX-License-Identifier: Apache-2.0
"""CPU inference for saved LinearRegression, Ridge, TruncatedSVD,
LogisticRegression and PCA models (the classical host inference lane,
2026-09-13), since the kde svc host lane (2026-09-14) saved
KernelDensity and SVC models, since the knn host inference lane
(2026-09-14) NearestNeighbors, KNeighborsClassifier and KNeighborsRegressor,
and since lane/inference-linear-svm (2026-09-15) StandardScaler,
MinMaxScaler, ElasticNet, Lasso, KernelRidge, Nystroem and RBFSampler, whose
entries the estimators host binding serves, and since lane/inference-svm
(2026-09-15) SVR (rbf and linear) beside SVC's linear and polynomial
kernels, through the svm host binding's `svr_predict` and `svc_predict`,
and since the neighbors and density inference lane (2026-09-15)
RadiusNeighbors and every k-NN metric, the ball cover and the KDE kernel and
metric pairs, IsolationForest, and GaussianMixture and HDBSCAN through their
inference-only bindings.

`host_model(path)` loads a file written by one of those classes' `save` and
returns an instance of a HOST SUBCLASS of the same class: the same Python
`predict`, `predict_proba`, `decision_function`, `transform` or
`kneighbors` as the GPU class, character for character, with ONE
difference, `_bind` answers the CPU binding of the class's family
(`mojolearn/host/_mojolearn_estimators_host.so` for the five classical
estimators and KernelDensity, `mojolearn/host/_mojolearn_svm_host.so` for
SVC, `mojolearn/host/_mojolearn_core_host.so` for the three k-NN classes) instead of the GPU set. Each binding exports the GPU binding's
names under the GPU binding's address contracts
(`bindings/_mojolearn_estimators_host.mojo`, `bindings/_mojolearn_svm_host.mojo`,
`bindings/_mojolearn_core_host.mojo`), and their arithmetic is
`core/classical_host_predict.mojo` (the pinned gemm/gemv kernel, the
intercept and bias epilogues, the centering kernel and the host sigmoid)
and `core/knn_host_predict.mojo` (the pinned distance tile, the halving
tree row norm, the composite-key selection, the vote and the mean),
`kde/host/kde_oracle.mojo` (`oracle_score_samples`) and
`svm/host/smo_oracle.mojo` (`smo_oracle_decision`).

On a CPU-only install none of this is needed: `_backend._HOST_MODULES`
routes each family to its host binding and the plain classes' `load` and
`predict` run through it. These subclasses exist so that a box WITH a GPU
(the Mac that records the GPU answer) can run the host path in the same
process, which is how tools/classical_host_gate.py compares the two bit
for bit. The binding is loaded through `_backend.load_host_module`, which
honors MOJOLEARN_HOST_DIR (the gate's sabotage set) and refuses a sabotage
build unless MOJOLEARN_HOST_ALLOW_SABOTAGE=1.

This module holds no arithmetic. What it promises is what the gate
measured; the brief records on which CPUs that has passed.

Since lane/inference-forecast-umap-pca (2026-09-15) also saved ARIMA models
(`predict`, in sample and out of sample, `forecast` and the fitted
attributes) through `mojolearn/host/_mojolearn_forecast_host.so`, the
inference binding that carries no fit, and saved UMAP embeddings
(`transform`, whose answer depends on the query batch by the transform's
contract) through `mojolearn/host/_mojolearn_metrics_host.so`. Since
lane/inference-holtwinters (2026-09-15) also saved Holt-Winters models
(`forecast`, `predict` and the fitted state) through the same forecast
binding.
"""
import hashlib
import os

from . import _backend, _native_classical_bridge, _serialize
from ._iforest_impl import IsolationForest, _IFOREST_FORMAT
from ._arima_impl import ARIMA, _ARIMA_FORMAT, _ARIMA_FORMAT_EXOG
from ._tsa_impl import ExponentialSmoothing, _HW_FORMAT
from ._cholesky_impl import _CHOLESKY_FORMAT, Cholesky
from ._ivf_impl import IVFIndex, _IVF_FORMAT
from .embedding import Embedding, _EMBEDDING_FORMAT
from ._gpc_impl import _GPC_FORMAT, GaussianProcessClassifier
from ._solver_impl import ElasticNet, Lasso, _CD_FORMAT
from ._svm_impl import SVC, SVR, _SVC_FORMAT, _SVR_FORMAT
from ._umap_impl import UMAP, _UMAP_FORMAT
from ._spectral_impl import SpectralClustering, _SPECTRAL_FORMAT
from .decomposition import PCA, TruncatedSVD, _PCA_FORMAT, _TSVD_FORMAT
from ._hierarchy_impl import AgglomerativeClustering, _AGGLOMERATIVE_FORMAT
from .cluster import KMeans, _KMEANS_FORMAT
from .density import DBSCAN, KernelDensity, _DBSCAN_FORMAT, _KDE_FORMAT
from ._gp_impl import GaussianProcessRegressor, _GP_FORMAT
from .hdbscan import HDBSCAN, _HDBSCAN_FORMAT
from .mixture import GaussianMixture, _GMM_FORMAT
from .kernel_methods import (
    KernelRidge, Nystroem, RBFSampler, _KERNEL_RIDGE_FORMAT, _NYSTROEM_FORMAT,
    _RBF_SAMPLER_FORMAT,
)
from .linear_model import (
    _CLASSICAL, LinearRegression, LogisticRegression, Ridge, _LINEAR_FORMAT,
    _LOGISTIC_FORMAT, _native_classical,
)
from .neighbors import (
    KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors, RadiusNeighbors,
    _KNN_FORMAT, _RADIUS_FORMAT,
)
from .preprocessing import MinMaxScaler, StandardScaler, _SCALER_FORMAT

_HOST_BASENAME = "_mojolearn_estimators_host"
#: GPU family -> the host binding a host subclass of that family binds. The
#: scalers, coordinate descent and the kernel methods have reference-only
#: training bindings that do not ship in a wheel; their saved-model entries
#: (`standard_transform`, `minmax_transform`, `cd_predict`,
#: `kernel_ridge_predict`, `nystroem_transform`, `rbf_sampler_transform`)
#: are served by the shipped estimators host binding
#: (lane/inference-linear-svm, 2026-09-15).
_HOST_BASENAMES = {
    "_mojolearn_estimators": _HOST_BASENAME,
    "_mojolearn_svm": "_mojolearn_svm_host",
    "_mojolearn": "_mojolearn_core_host",
    "_mojolearn_x_neighbors": "_mojolearn_x_neighbors_host",
    # The inference-only binding a wheel ships, never the reference one, so
    # a GPU box checks the binary a CPU-only install runs.
    "_mojolearn_mixture": "_mojolearn_mixture_infer_host",
    "_mojolearn_hdbscan": "_mojolearn_hdbscan_infer_host",
    "_mojolearn_gp": "_mojolearn_gp_infer_host",
    "_mojolearn_arima": "_mojolearn_forecast_host",
    "_mojolearn_tsa": "_mojolearn_forecast_host",
    "_mojolearn_ivf": "_mojolearn_ivf_search_host",
    "_mojolearn_embedding": "_mojolearn_embedding_infer_host",
    "_mojolearn_metrics": "_mojolearn_metrics_host",
    "_mojolearn_preprocessing": _HOST_BASENAME,
    "_mojolearn_solver": _HOST_BASENAME,
    "_mojolearn_kernel_methods": _HOST_BASENAME,
}


def binary_path():
    """The binary `host_model` loads for the five classical estimators, or
    would load."""
    return _backend.host_module_path(_HOST_BASENAME)


def binary_paths():
    """Every binary `host_model` may load, by GPU family."""
    return {family: _backend.host_module_path(b) for family, b in _HOST_BASENAMES.items()}


class _HostBound:
    """Bind the class's family and declared helpers to CPU modules only.

    Undeclared families are refused, so a host subclass cannot reach a GPU
    binding by accident.
    """

    _HOST_INFERENCE_ONLY = True
    _HOST_AUX_BINDINGS = ()

    def _bind(self, name=None):
        name = name or self._BINDING
        if ((name != self._BINDING and name not in self._HOST_AUX_BINDINGS)
                or name not in _HOST_BASENAMES):
            raise ImportError(
                f"mojolearn: the host {type(self).__name__} serves "
                f"{self._BINDING} only, not {name}"
            )
        mode = getattr(self, "numeric_mode", None)
        if mode is not None and mode != "identical":
            raise ValueError(
                f"mojolearn: {type(self).__name__} runs IDENTICAL only on the "
                f"host; this model was saved {mode!r}"
            )
        return _backend.load_host_module(_HOST_BASENAMES[name])

    def _host_refusals(self):
        """Raised by `host_model` after `load`, for a saved parameter the
        host binding has no entry for; a subclass overrides."""

    def vendor_used(self):
        return "cpu"

    def model_sha256(self):
        """SHA-256 over the model file's bytes as `save` would write them
        again, for a report to name the model it predicted with."""
        h = hashlib.sha256()
        for name in sorted(self._HOST_ARRAYS):
            value = getattr(self, name)
            if value is not None:
                h.update(value.tobytes())
        return h.hexdigest()


class HostLinearRegression(_HostBound, LinearRegression):
    _HOST_ARRAYS = ("coef_",)

    def vendor_used(self):
        """The backend this model runs on: always ``'cpu'``."""
        return _native_classical(self, "vendor_used", (), _CLASSICAL)

    def model_sha256(self):
        """SHA-256 over the saved coefficient bytes, for a report to name the
        model it predicted with."""
        return _native_classical(self, "model_sha256", (), _CLASSICAL)


class HostRidge(_HostBound, Ridge):
    _HOST_ARRAYS = ("coef_",)

    def vendor_used(self):
        """The backend this model runs on: always ``'cpu'``."""
        return _native_classical(self, "vendor_used", (), _CLASSICAL)

    def model_sha256(self):
        """SHA-256 over the saved coefficient bytes, for a report to name the
        model it predicted with."""
        return _native_classical(self, "model_sha256", (), _CLASSICAL)


class _NativeArchiveLoad:
    @classmethod
    def load(cls, path):
        """Load saved model state from ``path`` into this CPU model class."""
        return _native_classical(cls, "host_archive_load", (path,), _CLASSICAL)


def _register_host_inference_class(cls):
    # glue: ordinary subclass layout metadata is connected at class definition.
    context = _CLASSICAL.get("host_inference_context")
    if context is not None:
        context["classical_public_layouts"][cls] = _native_classical_bridge._layout(
            cls, context["classical_attribute_descriptor"])


_NATIVE_HOST_FITS = {}
_NATIVE_HOST_INITIALIZERS = {}


def _native_host_initializer(method):
    # glue: restore the complete native constructor after mixin class setup.
    _NATIVE_HOST_INITIALIZERS[method.__qualname__] = method
    return method


def _native_host_fit_method(method):
    # glue: the complete native fit owns validation and fitted-state publication.
    method._mojolearn_tracks_fit = True
    _NATIVE_HOST_FITS[method.__qualname__] = method
    return method


def _native_host_inference_method(method):
    # glue: only these complete native methods own fitted/input validation
    # and result publication. Keep ordinary mixin setup for every other
    # inherited method and for overrides on an application subclass.
    method._mojolearn_public_out = True
    method._mojolearn_needs_fit = True
    method._mojolearn_validates = True
    return method


class HostLogisticRegression(_NativeArchiveLoad, _HostBound, LogisticRegression):
    _HOST_ARRAYS = ("_w",)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _register_host_inference_class(cls)

    @_native_host_inference_method
    def decision_function(self, X):
        """Return fitted linear class scores on the CPU.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples with the fitted feature count.

        Returns
        -------
        scores : float32 array of shape (n_samples,) or (n_samples, n_classes)
            One score per binary sample or one score per class otherwise.

        Examples
        --------
        >>> scores = model.decision_function(X)
        """
        return _native_classical(self, "host_infer_decision_function", (X,), _CLASSICAL)

    @_native_host_inference_method
    def predict(self, X):
        """Return fitted class labels for samples on the CPU.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples with the fitted feature count.

        Returns
        -------
        labels : array or list of length n_samples
            Integer labels use int64, real labels float64; other labels use a
            list. Binary positive scores select the second class; multiclass
            score ties select the first matching class.

        Examples
        --------
        >>> labels = model.predict(X)
        """
        return _native_classical(self, "host_infer_predict", (X,), _CLASSICAL)

    @_native_host_inference_method
    def predict_proba(self, X):
        """Return class probabilities in fitted class order on the CPU.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples with the fitted feature count.

        Returns
        -------
        probabilities : float64 array of shape (n_samples, n_classes)
            The binary sigmoid or multiclass softmax of the fitted scores.

        Examples
        --------
        >>> probabilities = model.predict_proba(X)
        """
        return _native_classical(self, "host_infer_predict_proba", (X,), _CLASSICAL)

    @_native_host_inference_method
    def predict_log_proba(self, X):
        """Return natural logarithms of class probabilities on the CPU.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples with the fitted feature count.

        Returns
        -------
        log_probabilities : float64 array of shape (n_samples, n_classes)
            Natural log probabilities, with negative infinity for exact zero.

        Examples
        --------
        >>> log_probabilities = model.predict_log_proba(X)
        """
        return _native_classical(self, "host_infer_predict_log_proba", (X,), _CLASSICAL)

    def _host_refusals(self):
        """A model with more than two classes needs the softmax link
        (`qn_softmax`, lane/logistic-multiclass, 2026-09-14); a host build
        without it is refused by name at load, not at the first predict."""
        if len(self.classes_) > 2:
            binding = self._bind("_mojolearn_estimators")
            if not callable(getattr(binding, "qn_softmax", None)):
                raise ImportError(
                    "mojolearn: this build of _mojolearn_estimators_host does "
                    "not export qn_softmax, so a LogisticRegression with "
                    f"{len(self.classes_)} classes cannot predict on the host; "
                    "rebuild it with bindings/build_estimators_host.sh"
                )


class _HostProjection(_HostBound):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _register_host_inference_class(cls)

    @_native_host_inference_method
    def transform(self, X):
        """Project samples onto the fitted components on the CPU.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples in the original feature space.

        Returns
        -------
        Z : array of shape (n_samples, n_components), float32
            Projected samples, whitened when the fitted PCA requests it.

        Examples
        --------
        >>> reduced = model.transform(X)
        """
        return _native_classical(self, "host_infer_transform", (X,), _CLASSICAL)

    @_native_host_inference_method
    def inverse_transform(self, X):
        """Reconstruct projected samples in the original feature space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)
            Samples in the fitted component space.

        Returns
        -------
        X_original : array of shape (n_samples, n_features), float32
            Reconstructed samples, undoing PCA whitening when requested.

        Examples
        --------
        >>> reconstructed = model.inverse_transform(reduced)
        """
        return _native_classical(self, "host_infer_inverse_transform", (X,), _CLASSICAL)


class HostTruncatedSVD(_NativeArchiveLoad, _HostProjection, TruncatedSVD):
    _HOST_ARRAYS = ("components_", "singular_values_")


class HostPCA(_NativeArchiveLoad, _HostProjection, PCA):
    _HOST_ARRAYS = ("components_", "mean_", "singular_values_")


class HostKernelDensity(_NativeArchiveLoad, _HostBound, KernelDensity):
    """Evaluate a saved kernel density model on the CPU."""
    _HOST_ARRAYS = ("_x", "_w")

    def model_sha256(self):
        """Hash the stored training rows and optional weights in field-name order.

        Arrays contribute physical storage bytes; NumPy arrays contribute
        logical C-order bytes. Model options are not included.

        Returns
        -------
        str
            The 64-character hexadecimal SHA-256 digest.

        Examples
        --------
        >>> digest = model.model_sha256()
        """
        return _native_classical(self, "host_density_hash", ("kde",), _CLASSICAL)

    @_native_host_initializer
    def __init__(self, *, bandwidth=1.0, algorithm="auto", kernel="gaussian", metric="euclidean",
                 atol=0, rtol=0, breadth_first=True, leaf_size=40, metric_params=None, numeric_mode=None):
        """Create a CPU kernel density estimator with the supplied options.

        Parameters
        ----------
        bandwidth : float, default=1.0
            Kernel bandwidth.
        algorithm : str, default="auto"
            Density evaluation algorithm.
        kernel : str, default="gaussian"
            Density kernel.
        metric : str, default="euclidean"
            Distance metric.
        atol, rtol : float, default=0
            Absolute and relative approximation tolerances.
        breadth_first : bool, default=True
            Tree traversal option.
        leaf_size : int, default=40
            Tree leaf size option.
        metric_params : dict or None, default=None
            Additional distance metric options.
        numeric_mode : str or None, default=None
            Numeric mode recorded with the estimator.

        Returns
        -------
        HostKernelDensity
            The estimator, ready to receive training rows or saved state.

        Examples
        --------
        >>> model = HostKernelDensity(bandwidth=0.5)
        """
        return _native_classical(self, "host_density_initialize", ("kde", bandwidth, kernel, metric, algorithm,
                                  atol, rtol, breadth_first, leaf_size, metric_params, numeric_mode), _CLASSICAL)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _register_host_inference_class(cls)

    @_native_host_fit_method
    def fit(self, X, y=None, sample_weight=None):
        """Store finite training rows and optional nonnegative weights on the CPU.

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            Training rows, converted to float32 when necessary.
        y : array-like, default=None
            Optional target validated for compatibility and otherwise ignored.
        sample_weight : scalar or array-like of shape (n_samples,), default=None
            Finite nonnegative row weights containing at least one positive value.

        Returns
        -------
        self
            The fitted density estimator.

        Examples
        --------
        >>> model = model.fit(X, sample_weight=weights)
        """
        return _native_classical(self, "host_kde_fit", (X, y, sample_weight), _CLASSICAL)

    def save(self, path):
        """Write the fitted model and its options to a deterministic NPZ archive.

        Parameters
        ----------
        path : str, path-like or binary file
            Destination for the fitted training rows and model metadata.

        Returns
        -------
        path
            The supplied destination, ready for ``host_model`` or ``load``.

        Examples
        --------
        >>> model.save("model.npz")
        """
        return _native_classical(self, "host_density_save", ("kde", path), _CLASSICAL)

    @_native_host_inference_method
    def score_samples(self, X):
        """Return the log density at each query sample.

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_queries, n_features)
            Query samples with the fitted feature count.

        Returns
        -------
        array-like of float32
            One log density per query; compact-kernel misses return -3.4028235e38.

        Examples
        --------
        >>> densities = model.score_samples(X)
        """
        return _native_classical(self, "host_infer_score_samples", (X,), _CLASSICAL)

    @_native_host_inference_method
    def score(self, X, y=None):
        """Return the total log density using the fixed native summation order.

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_queries, n_features)
            Query samples with the fitted feature count.
        y : ignored
            Accepted for compatibility with estimator scoring interfaces.

        Returns
        -------
        float
            Total log density of the queries.

        Examples
        --------
        >>> total_log_density = model.score(X)
        """
        return _native_classical(self, "host_infer_score", (X, y), _CLASSICAL)


class HostSVC(_NativeArchiveLoad, _HostBound, SVC):
    """Evaluate a saved support vector classifier on the CPU."""
    _HOST_ARRAYS = ("dual_coef_", "support_vectors_", "intercept_")

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # glue: ordinary subclass layout is recorded only at class definition.
        context = _CLASSICAL.get("host_inference_context")
        if context is not None:
            context["classical_public_layouts"][cls] = _native_classical_bridge._layout(
                cls, context["host_svc_attribute_descriptor"])

    @_native_host_inference_method
    def decision_function(self, X):
        """Return binary margins or multiclass one-vs-one/one-vs-rest scores.

        Parameters
        ----------
        X : array-like or sparse matrix
            Query samples; precomputed kernels use training rows as columns.

        Returns
        -------
        array-like
            One margin per binary query, or the configured multiclass scores.

        Examples
        --------
        >>> margins = model.decision_function(X)
        """
        return _native_classical(self, "host_infer_decision_function", (X,), _CLASSICAL)

    @_native_host_inference_method
    def predict(self, X):
        """Return fitted class labels for each query sample.

        Parameters
        ----------
        X : array-like or sparse matrix
            Query samples; precomputed kernels use training rows as columns.

        Returns
        -------
        array-like or list
            Predicted labels, using the configured multiclass tie rule.

        Examples
        --------
        >>> labels = model.predict(X)
        """
        return _native_classical(self, "host_infer_predict", (X,), _CLASSICAL)

    @_native_host_inference_method
    def predict_proba(self, X):
        """Return calibrated class probabilities when probability=True.

        Parameters
        ----------
        X : array-like or sparse matrix
            Query samples; precomputed kernels use training rows as columns.

        Returns
        -------
        array-like of shape (n_samples, n_classes)
            Float64 probabilities in fitted class order.

        Examples
        --------
        >>> probabilities = model.predict_proba(X)
        """
        return _native_classical(self, "host_infer_predict_proba", (X,), _CLASSICAL)

    @_native_host_inference_method
    def predict_log_proba(self, X):
        """Return natural logarithms of calibrated class probabilities.

        Parameters
        ----------
        X : array-like or sparse matrix
            Query samples; precomputed kernels use training rows as columns.

        Returns
        -------
        array-like of shape (n_samples, n_classes)
            Float64 log probabilities; available when probability=True.

        Examples
        --------
        >>> log_probabilities = model.predict_log_proba(X)
        """
        return _native_classical(self, "host_infer_predict_log_proba", (X,), _CLASSICAL)


class HostIsolationForest(_NativeArchiveLoad, _HostBound, IsolationForest):
    """Score samples with a saved isolation forest on the CPU."""
    _HOST_ARRAYS = ("_x",)

    def model_sha256(self):
        """Hash the stored training rows without model options or archive metadata.

        Arrays contribute physical storage bytes; NumPy arrays contribute
        logical C-order bytes.

        Returns
        -------
        str
            The 64-character hexadecimal SHA-256 digest.

        Examples
        --------
        >>> digest = model.model_sha256()
        """
        return _native_classical(self, "host_density_hash", ("iforest",), _CLASSICAL)

    @_native_host_initializer
    def __init__(self, *, n_estimators=100, max_samples="auto", max_depth=None, max_features=1.0,
                 bootstrap=False, contamination="auto", random_state=None, warm_start=False,
                 n_jobs=None, verbose=False, output_type=None, numeric_mode=None):
        """Create a CPU isolation forest with the supplied options.

        Parameters
        ----------
        n_estimators : int, default=100
            Number of isolation trees.
        max_samples : int, float or str, default="auto"
            Training sample count or fraction per tree.
        max_depth : int or None, default=None
            Maximum tree depth.
        max_features : int or float, default=1.0
            Feature count or fraction per tree.
        bootstrap : bool, default=False
            Whether trees sample rows with replacement.
        contamination : float or str, default="auto"
            Expected outlier fraction or automatic threshold selection.
        random_state : int or None, default=None
            Random seed.
        warm_start : bool, default=False
            Whether subsequent training extends an existing forest.
        n_jobs : int or None, default=None
            Worker count option.
        verbose : bool, default=False
            Verbosity option.
        output_type : str or None, default=None
            Preferred output container.
        numeric_mode : str or None, default=None
            Numeric mode recorded with the estimator.

        Returns
        -------
        HostIsolationForest
            The estimator, ready to receive saved state.

        Examples
        --------
        >>> model = HostIsolationForest(n_estimators=20)
        """
        return _native_classical(self, "host_density_initialize", ("iforest", n_estimators, max_samples, max_depth,
                                  max_features, bootstrap, contamination, random_state, warm_start,
                                  n_jobs, verbose, output_type, numeric_mode), _CLASSICAL)

    @_native_host_fit_method
    def fit(self, X, y=None):
        """Fit a seeded isolation forest to finite dense training samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Dense training samples, converted to float32 when necessary.
        y : array-like, default=None
            Optional target validated for compatibility and otherwise ignored.

        Returns
        -------
        self
            The fitted isolation forest with its learned anomaly threshold.

        Examples
        --------
        >>> model = model.fit(X)
        """
        return _native_classical(self, "host_iforest_fit", (X, y), _CLASSICAL)

    def fit_predict(self, X, y=None):
        """Fit on `X` and return `predict(X)`, both on the CPU.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Dense training samples.
        y : array-like, default=None
            Ignored.

        Returns
        -------
        array-like of shape (n_samples,)
            -1 for an anomaly, 1 for an inlier.

        Examples
        --------
        >>> labels = model.fit_predict(X)
        """
        return self.fit(X, y=y).predict(X)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _register_host_inference_class(cls)

    def save(self, path):
        """Write the fitted model and its options to a deterministic NPZ archive.

        Parameters
        ----------
        path : str, path-like or binary file
            Destination for the fitted training rows and model metadata.

        Returns
        -------
        path
            The supplied destination, ready for ``host_model`` or ``load``.

        Examples
        --------
        >>> model.save("model.npz")
        """
        return _native_classical(self, "host_density_save", ("iforest", path), _CLASSICAL)

    @_native_host_inference_method
    def score_samples(self, X):
        """Return anomaly scores, with lower scores indicating more anomalous samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Dense query samples with the fitted feature count.

        Returns
        -------
        array-like of shape (n_samples,)
            Float64 scores, one per input row.

        Examples
        --------
        >>> scores = model.score_samples(X)
        """
        return _native_classical(self, "host_infer_score_samples", (X,), _CLASSICAL)

    @_native_host_inference_method
    def decision_function(self, X):
        """Return scores relative to the fitted anomaly threshold.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Dense query samples with the fitted feature count.

        Returns
        -------
        array-like of shape (n_samples,)
            Float64 decisions; negative values indicate anomalies.

        Examples
        --------
        >>> decisions = model.decision_function(X)
        """
        return _native_classical(self, "host_infer_decision_function", (X,), _CLASSICAL)

    @_native_host_inference_method
    def predict(self, X):
        """Label each sample as an anomaly or inlier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Dense query samples with the fitted feature count.

        Returns
        -------
        array-like of shape (n_samples,)
            Int32 labels: -1 for anomalies and 1 for inliers.

        Examples
        --------
        >>> labels = model.predict(X)
        """
        return _native_classical(self, "host_infer_predict", (X,), _CLASSICAL)


class HostSVR(_NativeArchiveLoad, _HostBound, SVR):
    """Predict regression targets from a saved support vector model on the CPU."""
    _HOST_ARRAYS = ("dual_coef_", "support_vectors_", "intercept_")

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # glue: ordinary subclass layout is recorded only at class definition.
        context = _CLASSICAL.get("host_inference_context")
        if context is not None:
            context["classical_public_layouts"][cls] = _native_classical_bridge._layout(
                cls, context["host_svr_attribute_descriptor"])

    @_native_host_inference_method
    def predict(self, X):
        """Return support vector regression predictions on the CPU.

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            Query samples. A precomputed kernel uses one column per fitted
            training sample, in the original training order.

        Returns
        -------
        predictions : array of shape (n_samples,)
            Weighted support-vector kernel values plus the fitted intercept.

        Examples
        --------
        >>> predictions = model.predict(X)
        """
        return _native_classical(self, "host_infer_predict", (X,), _CLASSICAL)


class HostDBSCAN(_NativeArchiveLoad, _HostBound, DBSCAN):
    """`DBSCAN.predict` from a saved `prediction_data=True` model through
    `_mojolearn_estimators_host.labeled_reference_predict`
    (lane/inference-transductive-predict, 2026-09-15, DEVIATION 2740)."""
    _HOST_ARRAYS = ("components_", "core_sample_indices_", "_core_labels")


class HostAgglomerativeClustering(_NativeArchiveLoad, _HostBound, AgglomerativeClustering):
    """`AgglomerativeClustering.predict` from a saved model through the
    estimators host binding; the fit's solver family does not ship."""
    _HOST_ARRAYS = ("_fit_X", "labels_")


class HostKMeans(_NativeArchiveLoad, _HostBound, KMeans):
    """`KMeans.predict` and `KMeans.transform` from a saved model through
    `_mojolearn_core_host`'s `kmeans_predict` and `kmeans_transform`
    (lane/kmeans-save, 2026-09-16). The arithmetic is
    `cluster/host/kmeans_oracle.mojo`, the fit's own final assignment, so a
    model fitted on a GPU labels a row on a CPU with the GPU's bits. The
    cosine metric is refused by name inside `host_validate_params`, as it is
    at fit, so the refusal is the same one on both sides.

    `labels_` is in the file and in the hash: it is the fit's assignment of
    the training rows, and it is what a caller checks `predict` against.
    """
    _HOST_ARRAYS = ("cluster_centers_", "labels_")


    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _register_host_inference_class(cls)


    @_native_host_inference_method
    def predict(self, X):
        """Return the nearest fitted center index for each input row."""
        return _native_classical(self, "host_infer_predict", (X,), _CLASSICAL)

    @_native_host_inference_method
    def transform(self, X):
        """Return distances from each input row to the fitted centers."""
        return _native_classical(self, "host_infer_transform", (X,), _CLASSICAL)


class HostGaussianMixture(_NativeArchiveLoad, _HostBound, GaussianMixture):
    """score_samples, predict_proba, predict, score, bic and aic of a saved
    GaussianMixture through the inference-only mixture binding."""
    _HOST_ARRAYS = ("weights_", "means_", "covariances_", "precisions_cholesky_", "log_det_chol_")


class HostGaussianProcessRegressor(_NativeArchiveLoad, _HostBound, GaussianProcessRegressor):
    """The predictive mean and std of a saved GaussianProcessRegressor through
    the inference-only gp binding; normalize_y's scale-back is the GPU
    class's own `predict`."""
    _HOST_ARRAYS = ("X_train_", "L_", "alpha_")


class HostHDBSCAN(_NativeArchiveLoad, _HostBound, HDBSCAN):
    """A saved HDBSCAN that `mojolearn.hdbscan.approximate_predict` accepts,
    predicting through the inference-only hdbscan binding."""
    _HOST_ARRAYS = ("_raw_data", "core_distances_", "labels_")


#: GaussianProcessClassifier's host bindings (moved from `_gpc_impl` by
#: lane/cgr-kernel so the GPU class never names host code).
_GPC_HOST_BASENAME = "_mojolearn_gp_host"
#: The inference-only gp binding a wheel ships (the neighbors and density
#: inference lane, 2026-09-15): `gpc_predict` with no Laplace fit. A saved
#: classifier predicts through it when it is built (every wheel), and
#: through the reference binding otherwise (a source build of the routed
#: families, as the CPU identity gate makes).
_GPC_HOST_INFERENCE_BASENAME = "_mojolearn_gp_infer_host"

class HostGaussianProcessClassifier(_NativeArchiveLoad, GaussianProcessClassifier):
    """`GaussianProcessClassifier` bound to `_mojolearn_gp_infer_host` (or,
    when only the reference set is built, `_mojolearn_gp_host`) on any box,
    a GPU box included, so a GPU fit and a CPU prediction compare in one
    process (`mojolearn.host_model` returns this for a saved classifier).
    IDENTICAL only; fit refuses outside `reference_training()`."""

    _HOST_INFERENCE_ONLY = True

    def _bind(self, name=None):
        name = name or self._BINDING
        if name != self._BINDING:
            raise ImportError(
                f"mojolearn: the host {type(self).__name__} serves {self._BINDING} only, not {name}"
            )
        mode = getattr(self, "numeric_mode", None)
        if mode is not None and mode != "identical":
            raise ValueError(
                f"mojolearn: {type(self).__name__} runs IDENTICAL only on the host; "
                f"this model was saved {mode!r}"
            )
        if os.path.exists(_backend.host_module_path(_GPC_HOST_INFERENCE_BASENAME)):
            return _backend.load_host_module(_GPC_HOST_INFERENCE_BASENAME)
        return _backend.load_host_module(_GPC_HOST_BASENAME)

    def _host_refusals(self):
        """Nothing beyond `load`'s checks: the host binding exports both
        classification entries."""

    def vendor_used(self):
        return "cpu"


_LINALG_HOST_BASENAME = "_mojolearn_linalg_host"


class HostCholesky(_NativeArchiveLoad, Cholesky):
    """`Cholesky` bound to `_mojolearn_linalg_host` on any box, a GPU box
    included, so a GPU factor and a CPU solve can be compared in one
    process (`mojolearn.host_model` returns this for a saved factor).
    IDENTICAL only. (It lived in `_cholesky_impl` until cpu-gpu-cleanup
    c-linear, 2026-10-02: a GPU-path module never loads a host binding.)"""

    _HOST_INFERENCE_ONLY = True

    def _door(self):
        mode = getattr(self, "numeric_mode", None)
        if mode is not None and mode != "identical":
            raise ValueError(
                f"mojolearn: HostCholesky runs IDENTICAL only on the host; this "
                f"factor was saved {mode!r}"
            )
        return _backend.load_host_module(_LINALG_HOST_BASENAME), "linalg_numeric_mode"

    def _host_refusals(self):
        """Nothing beyond `load`'s own checks: the host binding carries all
        three door names."""

    def vendor_used(self):
        return "cpu"


class _HostKNN(_HostBound):
    """The three k-NN host classes. The random ball cover arm is served
    since the neighbors and density inference lane (2026-09-15): the core
    host binding exports `rbc_knn_search` (the knn-rbc training lane's
    exhaustive restatement, which the cover prunes exactly), so a saved
    `NearestNeighbors(algorithm='rbc')` answers `kneighbors` on a CPU. The
    classifier and the regressor refuse 'rbc' in their own `load`."""


class HostNearestNeighbors(_NativeArchiveLoad, _HostKNN, NearestNeighbors):
    _HOST_ARRAYS = ("_index",)


class HostKNeighborsClassifier(_NativeArchiveLoad, _HostKNN, KNeighborsClassifier):
    _HOST_ARRAYS = ("_index", "_y_cols")


class HostKNeighborsRegressor(_NativeArchiveLoad, _HostKNN, KNeighborsRegressor):
    _HOST_ARRAYS = ("_index", "_y_cols")


class HostRadiusNeighbors(_NativeArchiveLoad, _HostBound, RadiusNeighbors):
    """`radius_neighbors` of a saved RadiusNeighbors through the core host
    binding's radius_neighbors_count and radius_neighbors_fill (the radius
    training lanes' ball cover restated as an exhaustive scan)."""
    _HOST_ARRAYS = ("_index",)
    # Sorted ragged results use the expansion binding's compiled segmented
    # sort. Route that helper to its host binary as well as the core radius
    # search; accepting an arbitrary family here could reach the GPU.
    _HOST_AUX_BINDINGS = ("_mojolearn_x_neighbors",)


class HostARIMA(_NativeArchiveLoad, _HostBound, ARIMA):
    """A saved ARIMA model on the forecast inference binding, which exports
    `arima_predict` and `arima_forecast` and no `arima_fit`.

    `_exog` is the fit's exogenous regressors, which a `mojolearn-arima-2`
    file carries and prediction reads (lane/arima-exog, 2026-09-15). It is
    `None` for a model fit without them, and `model_sha256` skips a `None`,
    so every `mojolearn-arima-1` model hashes exactly as it did."""
    _HOST_ARRAYS = ("_y", "params_", "_exog")


class HostExponentialSmoothing(_NativeArchiveLoad, _HostBound, ExponentialSmoothing):
    """A saved Holt-Winters model on the forecast inference binding, which
    exports `holtwinters_forecast` and `holtwinters_predict` and no
    `holtwinters_fit` (lane/inference-holtwinters, 2026-09-15)."""
    _HOST_ARRAYS = ("_comps",)


class HostUMAP(_NativeArchiveLoad, _HostBound, UMAP):
    """A saved UMAP embedding on the metrics host binding. `transform`
    answers the GPU's bytes for a row whatever else is asked in the same
    batch: since lane/umap-batch-fix (2026-09-16) a batch of N is the
    concatenation of N batches of one (umap/transform.mojo)."""
    _HOST_ARRAYS = ("_transform_training", "_transform_embedding")


class HostSpectralClustering(_NativeArchiveLoad, _HostBound, SpectralClustering):
    """`SpectralClustering.predict` from a saved `prediction_data=True`
    model through `_mojolearn_metrics_host.spectral_predict`, which runs no
    fit (lane/spectral-predict, 2026-09-15, DEVIATION 2860)."""
    _BINDING = "_mojolearn_metrics"
    _HOST_ARRAYS = ("_pd_eigenvalues", "_pd_eigenvectors", "_pd_diag", "_pd_centroids", "_fit_X", "labels_")


class _HostScaler(_HostBound):
    """The scalers ask for their binding through `_binding(mode)` with the
    fitted mode; the host answers the estimators host binding for an
    IDENTICAL model and refuses any other mode by name."""

    _BINDING = "_mojolearn_preprocessing"

    def _binding(self, mode):
        if mode != "identical":
            raise ValueError(
                f"mojolearn: {type(self).__name__} runs IDENTICAL only on the "
                f"host; this model was saved {mode!r}"
            )
        return self._bind(self._BINDING)


class HostStandardScaler(_HostScaler, StandardScaler):
    _HOST_ARRAYS = ("mean_", "var_", "scale_")


class HostMinMaxScaler(_NativeArchiveLoad, _HostScaler, MinMaxScaler):
    _HOST_ARRAYS = ("data_min_", "data_max_", "data_range_", "scale_", "min_")


class _HostCD(_HostBound):
    _BINDING = "_mojolearn_solver"
    _HOST_ARRAYS = ("coef_",)


    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _register_host_inference_class(cls)

    def _solver(self):
        return self._bind(self._BINDING)


    @_native_host_inference_method
    def predict(self, X):
        """Predict responses with the fitted coefficients and intercept.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input rows with the fitted number of features.

        Returns
        -------
        predictions : ndarray of shape (n_samples,)
            Predicted responses, published as float64 values.

        Examples
        --------
        >>> predictions = model.predict([[1.0, 2.0]])
        """
        return _native_classical(self, "host_infer_predict", (X,), _CLASSICAL)


class HostElasticNet(_NativeArchiveLoad, _HostCD, ElasticNet):
    pass


class HostLasso(_NativeArchiveLoad, _HostCD, Lasso):
    pass


class _HostKernelMethod(_HostBound):
    """Saved kernel models on the host, with native parameter validation."""

    _BINDING = "_mojolearn_kernel_methods"

    def __init_subclass__(cls, **kwargs):
        # glue: register ordinary subclass layouts only at class definition
        super().__init_subclass__(**kwargs)
        _register_host_inference_class(cls)

    def _host_refusals(self):
        from .kernel_methods import (KERNEL_LINEAR, KERNEL_RBF, KERNEL_POLYNOMIAL,
                                     KERNEL_SIGMOID, KERNEL_LAPLACIAN)
        kernel = self._kernel_params[0]
        if kernel not in (KERNEL_LINEAR, KERNEL_RBF, KERNEL_POLYNOMIAL,
                          KERNEL_SIGMOID, KERNEL_LAPLACIAN):
            raise ImportError(f"mojolearn: unsupported saved kernel code {kernel}")


class HostKernelRidge(_NativeArchiveLoad, _HostKernelMethod, KernelRidge):
    _HOST_ARRAYS = ("X_fit_", "dual_coef_")

    @_native_host_inference_method
    def predict(self, X):
        """Predict responses for dense or sparse rows using the fitted kernel.

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_queries, n_features)
            Query rows with the fitted number of features.

        Returns
        -------
        predictions : ndarray
            Float64 responses, one-dimensional for a single fitted target.

        Examples
        --------
        >>> predictions = model.predict([[1.0, 2.0]])
        """
        return _native_classical(self, "host_infer_predict", (X,), _CLASSICAL)


class HostNystroem(_NativeArchiveLoad, _HostKernelMethod, Nystroem):
    _HOST_ARRAYS = ("components_", "component_indices_", "normalization_",
                    "eigenvalues_", "eigenvectors_")

    @_native_host_inference_method
    def transform(self, X):
        """Map dense or sparse rows to the fitted approximate kernel features.

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_rows, n_features)
            Rows with the fitted number of features.

        Returns
        -------
        features : ndarray of shape (n_rows, n_components)
            Float32 approximate kernel features.

        Examples
        --------
        >>> features = model.transform([[1.0, 2.0]])
        """
        return _native_classical(self, "host_infer_transform", (X,), _CLASSICAL)


class HostRBFSampler(_NativeArchiveLoad, _HostBound, RBFSampler):
    _BINDING = "_mojolearn_kernel_methods"
    _HOST_ARRAYS = ("random_weights_", "random_offset_")

    def __init_subclass__(cls, **kwargs):
        # glue: register ordinary subclass layouts only at class definition
        super().__init_subclass__(**kwargs)
        _register_host_inference_class(cls)

    @_native_host_inference_method
    def transform(self, X):
        """Map dense or sparse rows to the fitted random Fourier features.

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_rows, n_features)
            Rows with the fitted number of features.

        Returns
        -------
        features : ndarray of shape (n_rows, n_components)
            Float32 random Fourier features.

        Examples
        --------
        >>> features = model.transform([[1.0, 2.0]])
        """
        return _native_classical(self, "host_infer_transform", (X,), _CLASSICAL)


class HostIVFIndex(_NativeArchiveLoad, _HostBound, IVFIndex):
    """A saved IVF-Flat index on the search inference binding, which exports
    `ivf_flat_search` and no build (lane/inference-embedding-ivf-cholesky,
    2026-09-15)."""
    _HOST_ARRAYS = ("centers_", "center_norms_", "list_offsets_", "list_indices_", "list_data_")


class HostEmbedding(_NativeArchiveLoad, _HostBound, Embedding):
    """A saved embedding table on the lookup inference binding, which
    exports `embedding_forward` and no backward."""
    _HOST_ARRAYS = ("weight",)


#: format tag -> (estimator name, host class). A file whose `estimator`
#: member names another class is refused by that class's own `load`.
_FORMATS = {
    _ARIMA_FORMAT: {"ARIMA": HostARIMA},
    # lane/arima-exog (2026-09-15): the same host class serves a model fitted
    # WITH regressors; `ARIMA.load` reads either tag and the file says which.
    _ARIMA_FORMAT_EXOG: {"ARIMA": HostARIMA},
    _HW_FORMAT: {"ExponentialSmoothing": HostExponentialSmoothing},
    _UMAP_FORMAT: {"UMAP": HostUMAP},
    _SPECTRAL_FORMAT: {"SpectralClustering": HostSpectralClustering},
    _SCALER_FORMAT: {"StandardScaler": HostStandardScaler, "MinMaxScaler": HostMinMaxScaler},
    _CD_FORMAT: {"ElasticNet": HostElasticNet, "Lasso": HostLasso},
    _KERNEL_RIDGE_FORMAT: {"KernelRidge": HostKernelRidge},
    _NYSTROEM_FORMAT: {"Nystroem": HostNystroem},
    _RBF_SAMPLER_FORMAT: {"RBFSampler": HostRBFSampler},
    _LINEAR_FORMAT: {"LinearRegression": HostLinearRegression, "Ridge": HostRidge},
    _LOGISTIC_FORMAT: {"LogisticRegression": HostLogisticRegression},
    _TSVD_FORMAT: {"TruncatedSVD": HostTruncatedSVD},
    _PCA_FORMAT: {"PCA": HostPCA},
    _KDE_FORMAT: {"KernelDensity": HostKernelDensity},
    _SVC_FORMAT: {"SVC": HostSVC},
    _IFOREST_FORMAT: {"IsolationForest": HostIsolationForest},
    _GMM_FORMAT: {"GaussianMixture": HostGaussianMixture},
    _HDBSCAN_FORMAT: {"HDBSCAN": HostHDBSCAN},
    _GP_FORMAT: {"GaussianProcessRegressor": HostGaussianProcessRegressor},
    _SVR_FORMAT: {"SVR": HostSVR},
    _DBSCAN_FORMAT: {"DBSCAN": HostDBSCAN},
    _AGGLOMERATIVE_FORMAT: {"AgglomerativeClustering": HostAgglomerativeClustering},
    # lane/kmeans-save (2026-09-16): a saved k-means model, every metric and
    # every start in ONE format, predicted on the core host binding.
    _KMEANS_FORMAT: {"KMeans": HostKMeans},
    _KNN_FORMAT: {
        "NearestNeighbors": HostNearestNeighbors,
        "KNeighborsClassifier": HostKNeighborsClassifier,
        "KNeighborsRegressor": HostKNeighborsRegressor,
    },
    _RADIUS_FORMAT: {"RadiusNeighbors": HostRadiusNeighbors},
    # A saved Cholesky factor (lane/inference-embedding-ivf-cholesky,
    # 2026-09-15): `HostCholesky` solves on `_mojolearn_linalg_host`.
    _CHOLESKY_FORMAT: {"Cholesky": HostCholesky},
    _IVF_FORMAT: {"IVFIndex": HostIVFIndex},
    _EMBEDDING_FORMAT: {"Embedding": HostEmbedding},
    # A saved GaussianProcessClassifier (lane/gaussian-process-classifier,
    # 2026-09-15): predicts on `_mojolearn_gp_infer_host` when it is built,
    # else on `_mojolearn_gp_host`.
    _GPC_FORMAT: {"GaussianProcessClassifier": HostGaussianProcessClassifier},
}


def _merge_expansion_doors():
    """THE EXPANSION LANES' SAVED-MODEL ROUTES (lane/algos-prep, 2026-09-27).
    Each lane's door,
    `_expansion_<lane>.py`, may carry `CLASSICAL_HOST_BASENAMES` (its GPU
    binding -> its host binding) and `classical_host_formats()` (format ->
    {estimator: host class}), called here once this module is complete so
    its host classes can subclass `_HostBound`. A format or family already
    routed is refused by name. This module is imported lazily (by
    `host_model`), after the package, so every door is complete by now."""
    # By name, not by a static import: tools/lane_select.py follows static
    # imports, and a static edge here would hand every lane that reaches a
    # saved-model route every expansion lane's door and bindings.
    import importlib
    from . import host_surface
    for lane in host_surface.EXPANSION_LANES:
        door = importlib.import_module(f"{__package__}._expansion_{lane}")
        for family, basename in getattr(door, "CLASSICAL_HOST_BASENAMES", {}).items():
            if family in _HOST_BASENAMES:
                raise RuntimeError(f"mojolearn: {door.__name__} reroutes {family}")
            _HOST_BASENAMES[family] = basename
        formats = getattr(door, "classical_host_formats", None)
        for fmt, estimators in (formats() if formats else {}).items():
            if fmt in _FORMATS:
                raise RuntimeError(f"mojolearn: {door.__name__} redeclares the saved format {fmt!r}")
            _FORMATS[fmt] = dict(estimators)


_merge_expansion_doors()
CLASSICAL_FORMATS = tuple(_FORMATS)

def _host_fit_helpers():
    """The CPU decomposition and x_linear binaries the explicit host
    LinearRegression and Ridge fit with (TSQR, weighted ridge); an
    unbuilt one is passed as its ImportError, refused when a fit needs it."""
    loaded = []
    for basename in ("_mojolearn_x_decomp_host", "_mojolearn_x_linear_host"):  # glue: the two helper binaries
        try:
            loaded.append(_backend.load_host_module(basename))
        except ImportError as error:
            loaded.append(error)
    return tuple(loaded)


# The explicit host classes run through the same compiled owner as
# LinearRegression and Ridge, with their CPU identical binding.
_native_classical_bridge.connect_host_public(HostRidge, HostLinearRegression, _CLASSICAL, _host_fit_helpers())

# Fixed archive metadata only; the compiled owner reads and routes saved
# state for every supported classical, forest and GBDT format.
from . import _native_host_archive_bridge


def _host_archive_cleanup_entries():
    # glue: only IVF's CPU inference binding owns a resident registry entry.
    # The other CPU host bindings keep ordinary array references. Device
    # finalizers stay on their original GPU classes.
    entries = {}
    for basename, key, name in (("_mojolearn_ivf_search_host", "IVFIndex", "ivf_flat_index_release"),  # glue: fixed explicit CPU archive entries
                                 ("_mojolearn_svm_host", "svc_ovo_layout", "svc_ovo_layout"),
                                 ("_mojolearn_estimators_host", "qn_softmax", "qn_softmax")):
        try:
            module = _backend.load_host_module(basename)
        except ImportError:
            continue
        entries[key] = getattr(module, name, None)
    return entries


def _host_archive_forest_metadata():
    # glue: fixed CPU saved-forest classes and binding metadata at import.
    from types import BuiltinFunctionType
    from . import _forest_host, _gbdt_host
    module, error = None, ""
    try:
        module = _forest_host._load()
    except ImportError as unavailable:
        error = str(unavailable)
    formats = {}
    for fmt, family in _forest_host._FORMATS.items():  # glue: two saved forest families
        classes = {name: _forest_host.HostForest for name, info in _forest_host._ESTIMATORS.items()  # glue: fixed forest class registry
                   if info[0] == family}
        formats[fmt] = classes
        formats[fmt + _forest_host._GROVES_SUFFIX] = dict(classes)
    forest_formats = tuple(formats)
    formats[_gbdt_host.GBDT_FORMAT] = {name: _gbdt_host.HostGBDT for name in _gbdt_host.GBDT_ESTIMATORS}  # glue: fixed GBDT archive names
    return dict(host_archive_extra_formats=formats, host_archive_forest_formats=forest_formats,
                host_tree_classes=(_forest_host.HostForest, _gbdt_host.HostGBDT),
                host_archive_forest_estimators=dict(_forest_host._ESTIMATORS),
                host_archive_forest_binding=module, host_archive_forest_binding_error=error,
                host_archive_forest_has_groves=isinstance(getattr(module, "forest_host_groves_prepare", None), BuiltinFunctionType),
                host_archive_gbdt_estimators=dict(_gbdt_host.GBDT_ESTIMATORS),
                host_archive_gbdt_classification_losses=tuple(_gbdt_host._CLASSIFICATION_LOSSES),
                host_archive_gbdt_proba_losses=tuple(_gbdt_host._PROBA_LOSSES))


_native_host_archive_bridge.connect(_CLASSICAL, _FORMATS, _host_archive_cleanup_entries(), _host_archive_forest_metadata())
_native_host_archive_bridge.connect_peer(_CLASSICAL)


def host_model(path):
    """The host model for a saved classical file, by its `format` and
    `estimator` members. Any other format is refused with the tag it
    carries."""
    return _native_classical(None, "host_archive_classical_model", (path,), _CLASSICAL)


# HostStandardScaler owns fixed CPU entries even on a GPU installation. Connect
# them only in this explicit host surface; ordinary scalers retain GPU entries.
from .preprocessing import _STANDARD_CONTEXT as _SCALER_CONTEXT
from ._native_scaler_bridge import connect_host as _connect_scaler_host, connect_host_unavailable as _scaler_host_unavailable
try:
    _scaler_training = _backend.load_host_module("_mojolearn_preprocessing_host")
except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as _scaler_training_error:
    _scaler_training = type(_scaler_training_error).__name__ + ": " + str(_scaler_training_error)
try:
    _connect_scaler_host(HostStandardScaler, _SCALER_CONTEXT,
                         _backend.load_host_module("_mojolearn_estimators_host"),
                         _backend.load_host_module("_mojolearn_x_prep_host"), _scaler_training)
except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as _scaler_error:
    _scaler_host_unavailable(HostStandardScaler, _SCALER_CONTEXT, _scaler_error)

# The MinMaxScaler host API likewise retains fixed CPU entries at setup.
from .preprocessing import _MINMAX_CONTEXT
from ._native_minmax_bridge import connect_host as _connect_minmax_host
try:
    _connect_minmax_host(HostMinMaxScaler, _MINMAX_CONTEXT,
                         _backend.load_host_module("_mojolearn_estimators_host"),
                         _backend.load_host_module("_mojolearn_x_prep_host"), _scaler_training)
except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as _minmax_error:
    _scaler_host_unavailable(HostMinMaxScaler, _MINMAX_CONTEXT, _minmax_error)


# HostLogisticRegression fits and scores through LogisticRegression's native
# owner with fixed CPU entries; its score predicts through the classical host
# inference owner, as its predict does.
from .linear_model import _LR_CONTEXT
from ._native_logistic_bridge import connect_host as _connect_logistic_host
try:
    _logistic_host = _backend.load_host_module(_HOST_BASENAME)
except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as _logistic_host_error:
    _logistic_host = _logistic_host_error
_connect_logistic_host(HostLogisticRegression, _LR_CONTEXT, _logistic_host, _native_classical, _CLASSICAL)


# Import-time connection of complete explicit CPU prediction owners.
def _connect_host_inference():
    from types import BuiltinFunctionType, MethodType, FunctionType, MemberDescriptorType, MethodDescriptorType, ModuleType
    from _contextvars import ContextVar
    from ._native_owner_lifetime import connect_owner_context
    from ._labels import ClassLabels
    from .tokenizer import _binding
    entries = {}
    for basename, names in (("_mojolearn_estimators_host", ("cd_predict", "kde_score_samples", "kernel_ridge_predict", "nystroem_transform", "rbf_sampler_transform",
                                                          "tsvd_transform", "pca_transform", "inverse_transform", "pca_whiten_transform", "pca_whiten_inverse_transform",
                                                          "qn_decision_function", "qn_predict_binary", "qn_sigmoid", "qn_softmax", "py2mojo_rows", "py2mojo_linear_flags")),  # glue: fixed CPU prediction ABI
                            ("_mojolearn_svm_host", ("svr_predict", "svc_predict", "svc_pair_epilogue", "svc_dual_gemv", "iforest_run")),
                            ("_mojolearn_core_host", ("kmeans_predict", "kmeans_transform"))):
        try:
            module = _backend.load_host_module(basename)
        except ImportError:
            module = None
        for name in names:  # glue: record native entry identities once at import
            entry = getattr(module, name, None)
            if entry is not None and not isinstance(entry, BuiltinFunctionType):
                raise TypeError("explicit CPU prediction requires a native binding entry")
            entries[name] = entry
    # CPU training is public. Replace only this complete native fit after
    # mixin class setup; inherited/overridden operations keep their guards.
    HostKernelDensity.fit = _NATIVE_HOST_FITS["HostKernelDensity.fit"]
    HostIsolationForest.fit = _NATIVE_HOST_FITS["HostIsolationForest.fit"]
    HostKernelDensity.__init__ = _NATIVE_HOST_INITIALIZERS["HostKernelDensity.__init__"]
    HostIsolationForest.__init__ = _NATIVE_HOST_INITIALIZERS["HostIsolationForest.__init__"]
    if type(list.sort) is not MethodDescriptorType:
        raise TypeError("explicit CPU model hashes require a native sorting descriptor")
    context = dict(_CLASSICAL["host_tree_context"])
    context.update(host_inference_entries=entries,
                   host_inference_class_labels=ClassLabels,
                   host_kde_bytearray_type=bytearray,
                   host_kde_bytes_type=bytes,
                   host_iforest_memoryview_type=memoryview,
                   host_iforest_module_type=ModuleType, host_iforest_module_dict=vars(ModuleType)["__dict__"].__get__,
                   host_iforest_context_type=ContextVar, host_iforest_context_get=ContextVar.get,
                   host_density_object_setattr=vars(object)["__setattr__"],
                   host_hash_list_sort=list.sort,
                   host_hash_field_collection_types=(list, tuple, set, frozenset, dict, str, bytes, bytearray, range),
                   host_hash_noniterable_types=(int, float, bool, complex, type(None)),
                   host_hash_name_sort_types=(str, bytes, int, float, bool, complex, type(None)),
                   host_hash_missing_tobytes_types=(str, bytes, bytearray, list, tuple, dict, set, frozenset, range, int, float, bool, complex),
                   host_density_member_type=MemberDescriptorType,
                   host_density_function_type=FunctionType, host_density_attribute_error=AttributeError,
                   host_kde_sequence_types=(list, tuple, range),
                   host_inference_kinds={HostElasticNet: "cd", HostLasso: "cd", HostKMeans: "kmeans",
                                         HostKernelRidge: "kernel_ridge", HostNystroem: "nystroem", HostRBFSampler: "rbf_sampler",
                                         HostPCA: "pca", HostTruncatedSVD: "tsvd", HostLogisticRegression: "logistic", HostSVR: "svr", HostSVC: "svc", HostKernelDensity: "kde", HostIsolationForest: "iforest"})
    connect_owner_context(context)
    from ._native_iforest_depth import connect as _connect_iforest_depth
    _connect_iforest_depth(context)
    # The established native attribute entry is schema-neutral: it admits
    # the supplied class layout and preserves internal Array/public ndarray
    # publication. It does not select scaler bindings or execute scaler work.
    module = _binding()._module
    attribute = module.standard_scaler_attribute_call
    if not isinstance(attribute, BuiltinFunctionType):
        raise TypeError("explicit CPU model attributes require a native binding entry")
    descriptor = module.classical_attribute_descriptor(MethodType(attribute, context))
    context.update(classical_attribute_descriptor=descriptor, classical_public_layouts={},
                   classical_public_installed=True)
    for cls in (HostElasticNet, HostLasso, HostKMeans, HostKernelRidge, HostNystroem, HostRBFSampler, HostPCA, HostTruncatedSVD, HostLogisticRegression, HostKernelDensity, HostIsolationForest):  # glue: fixed native prediction class layouts
        cls.__getattribute__ = descriptor
        context["classical_public_layouts"][cls] = _native_classical_bridge._layout(cls, descriptor)
    # SVR's existing coef_ property is computed natively rather than invoking
    # its Python descriptor. Every other attribute keeps the ordinary schema.
    svr_attribute = module.host_svr_attribute_call
    if not isinstance(svr_attribute, BuiltinFunctionType):
        raise TypeError("explicit CPU SVR attributes require a native binding entry")
    svr_descriptor = module.classical_attribute_descriptor(MethodType(svr_attribute, context))
    context.update(host_svr_attribute_descriptor=svr_descriptor,
                   host_svr_coefficient_property=vars(SVR)["coef_"])
    HostSVR.__getattribute__ = svr_descriptor
    context["classical_public_layouts"][HostSVR] = _native_classical_bridge._layout(HostSVR, svr_descriptor)
    svc_attribute = module.host_svc_attribute_call
    if not isinstance(svc_attribute, BuiltinFunctionType):
        raise TypeError("explicit CPU SVC attributes require a native binding entry")
    svc_descriptor = module.classical_attribute_descriptor(MethodType(svc_attribute, context))
    context.update(host_svc_attribute_descriptor=svc_descriptor,
                   host_svc_coefficient_property=vars(SVC)["coef_"],
                   host_svc_probability_methods={name: vars(HostSVC)[name]
                       for name in ("predict_proba", "predict_log_proba")})
    HostSVC.__getattribute__ = svc_descriptor
    context["classical_public_layouts"][HostSVC] = _native_classical_bridge._layout(HostSVC, svc_descriptor)
    _CLASSICAL["host_inference_context"] = context


_connect_host_inference()

# Connect after the host inference descriptor is installed, preserving the
# explicit CPU owner's attribute layout even in a package with GPU bindings.
from ._native_logistic_bridge import connect_public as _connect_logistic_public
_connect_logistic_public(HostLogisticRegression, _LR_CONTEXT, install=False)

# Complete fixed recipes are connected only after every host owner and class
# descriptor above has finished setup. Public prediction performs no imports.
def _host_archive_prediction_bindings():
    """The host bindings the saved classes' public predict methods bound
    through `_bind`, loaded once at setup. An unbuilt binding or a failed
    read-back is kept as its refusal text, raised when a prediction needs it
    (the old lazy `_bind` raised the same ImportError at that point)."""
    def load(basename):
        try:
            return _backend.load_host_module(basename)
        except (ImportError, AttributeError, TypeError, ValueError, RuntimeError, OSError) as error:
            return "ImportError: " + str(error)
    gpc = (_GPC_HOST_INFERENCE_BASENAME
           if os.path.exists(_backend.host_module_path(_GPC_HOST_INFERENCE_BASENAME))
           else _GPC_HOST_BASENAME)
    return {
        "_mojolearn_core_host": load(_HOST_BASENAMES["_mojolearn"]),
        "_mojolearn_estimators_host": load(_HOST_BASENAMES["_mojolearn_estimators"]),
        "mixture": load(_HOST_BASENAMES[HostGaussianMixture._BINDING]),
        "gp": load(_HOST_BASENAMES[HostGaussianProcessRegressor._BINDING]),
        "spectral": load(_HOST_BASENAMES[HostSpectralClustering._BINDING]),
        "forecast": load(_HOST_BASENAMES["_mojolearn_arima"]),
        "gpc": load(gpc),
        # The GP classifier's label gather: the core host binding exports the
        # exact gather_i64/gather_f64 the GPU base binding does, so a saved
        # classifier decodes on the CPU without a binding proxy.
        "gpc_core": load(_HOST_BASENAMES["_mojolearn"]),
    }


from ._native_host_archive_predict import connect as _connect_archive_prediction
_connect_archive_prediction(_CLASSICAL, globals(), _host_archive_prediction_bindings())

# Complete saved-model query ownership uses fixed CPU entries connected here.
from ._native_neighbor_queries import connect_host as _connect_neighbor_host
_connect_neighbor_host({HostNearestNeighbors: NearestNeighbors,
                        HostKNeighborsClassifier: KNeighborsClassifier,
                        HostKNeighborsRegressor: KNeighborsRegressor,
                        HostRadiusNeighbors: RadiusNeighbors, HostIVFIndex: IVFIndex}, _HOST_BASENAMES,
                       _CLASSICAL["host_archive_context"])
