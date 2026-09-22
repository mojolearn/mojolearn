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

from . import _backend, _serialize
from ._iforest_impl import IsolationForest, _IFOREST_FORMAT
from ._arima_impl import ARIMA, _ARIMA_FORMAT, _ARIMA_FORMAT_EXOG
from ._tsa_impl import ExponentialSmoothing, _HW_FORMAT
from ._cholesky_impl import _CHOLESKY_FORMAT, HostCholesky
from ._ivf_impl import IVFIndex, _IVF_FORMAT
from .embedding import Embedding, _EMBEDDING_FORMAT
from ._gpc_impl import _GPC_FORMAT, HostGaussianProcessClassifier
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
    LinearRegression, LogisticRegression, Ridge, _LINEAR_FORMAT,
    _LOGISTIC_FORMAT,
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
    """`_bind` answers the CPU binding of the class's own family
    (`_BINDING`) and refuses every other family by name, so a host subclass
    can never reach a GPU binding by accident."""

    _HOST_INFERENCE_ONLY = True

    def _bind(self, name=None):
        name = name or self._BINDING
        if name != self._BINDING or name not in _HOST_BASENAMES:
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


class HostRidge(_HostBound, Ridge):
    _HOST_ARRAYS = ("coef_",)


class HostLogisticRegression(_HostBound, LogisticRegression):
    _HOST_ARRAYS = ("_w",)

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


class HostTruncatedSVD(_HostBound, TruncatedSVD):
    _HOST_ARRAYS = ("components_", "singular_values_")


class HostPCA(_HostBound, PCA):
    _HOST_ARRAYS = ("components_", "mean_", "singular_values_")

    def _whiten_binding(self):
        """The host binding carries the whitened pair since the kde svc host
        lane (2026-09-14); an older host build without it is refused by
        name, as the GPU class refuses an older GPU build."""
        binding = self._bind("_mojolearn_estimators")
        if not all(callable(getattr(binding, name, None)) for name in
                   ("pca_whiten_transform", "pca_whiten_inverse_transform")):
            raise ImportError(
                "mojolearn: this build of _mojolearn_estimators_host does not "
                "export pca_whiten_transform and pca_whiten_inverse_transform; "
                "rebuild it with bindings/build_estimators_host.sh"
            )
        return binding

    def _dense_binding(self):
        raise ImportError(
            "mojolearn: no CPU implementation of the dense PCA fit; the host "
            "PCA transforms a saved model only"
        )


class HostKernelDensity(_HostBound, KernelDensity):
    _HOST_ARRAYS = ("_x", "_w")


class HostSVC(_HostBound, SVC):
    _HOST_ARRAYS = ("dual_coef_", "support_vectors_", "intercept_")


class HostIsolationForest(_HostBound, IsolationForest):
    """`score_samples`, `decision_function` and `predict` of a saved
    IsolationForest through the svm host binding's iforest_run: the forest
    rebuilt from the saved training matrix and knobs (DEVIATION 874), as
    every GPU scoring call rebuilds it, then scored."""
    _HOST_ARRAYS = ("_x",)


class HostSVR(_HostBound, SVR):
    """`SVR.predict` from a saved model through
    `_mojolearn_svm_host.svr_predict` (lane/inference-svm, 2026-09-15)."""
    _HOST_ARRAYS = ("dual_coef_", "support_vectors_", "intercept_")


class HostDBSCAN(_HostBound, DBSCAN):
    """`DBSCAN.predict` from a saved `prediction_data=True` model through
    `_mojolearn_estimators_host.labeled_reference_predict`
    (lane/inference-transductive-predict, 2026-09-15, DEVIATION 2740)."""
    _HOST_ARRAYS = ("components_", "core_sample_indices_", "_core_labels")


class HostAgglomerativeClustering(_HostBound, AgglomerativeClustering):
    """`AgglomerativeClustering.predict` from a saved model through the
    estimators host binding; the fit's solver family does not ship."""
    _HOST_ARRAYS = ("_fit_X", "labels_")


class HostKMeans(_HostBound, KMeans):
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


class HostGaussianMixture(_HostBound, GaussianMixture):
    """score_samples, predict_proba, predict, score, bic and aic of a saved
    GaussianMixture through the inference-only mixture binding."""
    _HOST_ARRAYS = ("weights_", "means_", "covariances_", "precisions_cholesky_", "log_det_chol_")


class HostGaussianProcessRegressor(_HostBound, GaussianProcessRegressor):
    """The predictive mean and std of a saved GaussianProcessRegressor through
    the inference-only gp binding; normalize_y's scale-back is the GPU
    class's own `predict`."""
    _HOST_ARRAYS = ("X_train_", "L_", "alpha_")


class HostHDBSCAN(_HostBound, HDBSCAN):
    """A saved HDBSCAN that `mojolearn.hdbscan.approximate_predict` accepts,
    predicting through the inference-only hdbscan binding."""
    _HOST_ARRAYS = ("_raw_data", "core_distances_", "labels_")


class _HostKNN(_HostBound):
    """The three k-NN host classes. The random ball cover arm is served
    since the neighbors and density inference lane (2026-09-15): the core
    host binding exports `rbc_knn_search` (the knn-rbc training lane's
    exhaustive restatement, which the cover prunes exactly), so a saved
    `NearestNeighbors(algorithm='rbc')` answers `kneighbors` on a CPU. The
    classifier and the regressor refuse 'rbc' in their own `load`."""


class HostNearestNeighbors(_HostKNN, NearestNeighbors):
    _HOST_ARRAYS = ("_index",)


class HostKNeighborsClassifier(_HostKNN, KNeighborsClassifier):
    _HOST_ARRAYS = ("_index", "_y_cols")


class HostKNeighborsRegressor(_HostKNN, KNeighborsRegressor):
    _HOST_ARRAYS = ("_index", "_y_cols")


class HostRadiusNeighbors(_HostBound, RadiusNeighbors):
    """`radius_neighbors` of a saved RadiusNeighbors through the core host
    binding's radius_neighbors_count and radius_neighbors_fill (the radius
    training lanes' ball cover restated as an exhaustive scan)."""
    _HOST_ARRAYS = ("_index",)


class HostARIMA(_HostBound, ARIMA):
    """A saved ARIMA model on the forecast inference binding, which exports
    `arima_predict` and `arima_forecast` and no `arima_fit`.

    `_exog` is the fit's exogenous regressors, which a `mojolearn-arima-2`
    file carries and prediction reads (lane/arima-exog, 2026-09-15). It is
    `None` for a model fit without them, and `model_sha256` skips a `None`,
    so every `mojolearn-arima-1` model hashes exactly as it did."""
    _HOST_ARRAYS = ("_y", "params_", "_exog")


class HostExponentialSmoothing(_HostBound, ExponentialSmoothing):
    """A saved Holt-Winters model on the forecast inference binding, which
    exports `holtwinters_forecast` and `holtwinters_predict` and no
    `holtwinters_fit` (lane/inference-holtwinters, 2026-09-15)."""
    _HOST_ARRAYS = ("_comps",)


class HostUMAP(_HostBound, UMAP):
    """A saved UMAP embedding on the metrics host binding. `transform`
    answers the GPU's bytes for a row whatever else is asked in the same
    batch: since lane/umap-batch-fix (2026-09-16) a batch of N is the
    concatenation of N batches of one (umap/transform.mojo)."""
    _HOST_ARRAYS = ("_transform_training", "_transform_embedding")


class HostSpectralClustering(_HostBound, SpectralClustering):
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


class HostMinMaxScaler(_HostScaler, MinMaxScaler):
    _HOST_ARRAYS = ("data_min_", "data_max_", "data_range_", "scale_", "min_")


class _HostCD(_HostBound):
    _BINDING = "_mojolearn_solver"
    _HOST_ARRAYS = ("coef_",)

    def _solver(self):
        return self._bind(self._BINDING)


class HostElasticNet(_HostCD, ElasticNet):
    pass


class HostLasso(_HostCD, Lasso):
    pass


class _HostKernelMethod(_HostBound):
    """Saved kernel models on the host, with native parameter validation."""

    _BINDING = "_mojolearn_kernel_methods"

    def _host_refusals(self):
        from .kernel_methods import (KERNEL_LINEAR, KERNEL_RBF, KERNEL_POLYNOMIAL,
                                     KERNEL_SIGMOID, KERNEL_LAPLACIAN)
        kernel = self._kernel_params[0]
        if kernel not in (KERNEL_LINEAR, KERNEL_RBF, KERNEL_POLYNOMIAL,
                          KERNEL_SIGMOID, KERNEL_LAPLACIAN):
            raise ImportError(f"mojolearn: unsupported saved kernel code {kernel}")


class HostKernelRidge(_HostKernelMethod, KernelRidge):
    _HOST_ARRAYS = ("X_fit_", "dual_coef_")


class HostNystroem(_HostKernelMethod, Nystroem):
    _HOST_ARRAYS = ("components_", "component_indices_", "normalization_",
                    "eigenvalues_", "eigenvectors_")


class HostRBFSampler(_HostBound, RBFSampler):
    _BINDING = "_mojolearn_kernel_methods"
    _HOST_ARRAYS = ("random_weights_", "random_offset_")


class HostIVFIndex(_HostBound, IVFIndex):
    """A saved IVF-Flat index on the search inference binding, which exports
    `ivf_flat_search` and no build (lane/inference-embedding-ivf-cholesky,
    2026-09-15)."""
    _HOST_ARRAYS = ("centers_", "center_norms_", "list_offsets_", "list_indices_", "list_data_")


class HostEmbedding(_HostBound, Embedding):
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
CLASSICAL_FORMATS = tuple(_FORMATS)


def host_model(path):
    """The host model for a saved classical file, by its `format` and
    `estimator` members. Any other format is refused with the tag it
    carries."""
    arrays = _serialize.peek_npz(path, ("format", "estimator"))
    fmt = _serialize.scalar_str(arrays, "format") if "format" in arrays else ""
    if fmt not in CLASSICAL_FORMATS:
        # Keep the former full-decode error precedence for malformed files;
        # only accepted archives take the routing fast path.
        _serialize.read_npz(path, CLASSICAL_FORMATS)
        raise AssertionError("read_npz accepted a format rejected by the router")
    estimator = _serialize.scalar_str(arrays, "estimator")
    cls = _FORMATS[fmt].get(estimator)
    if cls is None:
        raise ValueError(
            f"mojolearn: {path!r} was saved by {estimator}, which {fmt} does not hold"
        )
    model = cls.load(path)
    model._host_refusals()
    model.estimator = estimator
    return model
