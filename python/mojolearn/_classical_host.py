# SPDX-License-Identifier: Apache-2.0
"""CPU inference for saved LinearRegression, Ridge, TruncatedSVD,
LogisticRegression and PCA models (the classical host inference lane,
2026-09-13; brief docs/lanes/BRIEF_forest_host_inference_2026-09-13.md,
"Classical lanes"), since the kde svc host lane (2026-09-14) saved
KernelDensity and SVC models, since the knn host inference lane
(2026-09-14) NearestNeighbors, KNeighborsClassifier and KNeighborsRegressor,
and since lane/inference-linear-svm (2026-09-15) StandardScaler,
MinMaxScaler, ElasticNet, Lasso, KernelRidge, Nystroem and RBFSampler, whose
entries the estimators host binding serves, and since lane/inference-svm
(2026-09-15) SVR (rbf and linear) beside SVC's linear and polynomial
kernels, through the svm host binding's `svr_predict` and `svc_predict`.

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
contract) through `mojolearn/host/_mojolearn_metrics_host.so`.
"""
import hashlib

from . import _backend, _serialize
from ._arima_impl import ARIMA, _ARIMA_FORMAT
from ._cholesky_impl import _CHOLESKY_FORMAT, HostCholesky
from ._gpc_impl import _GPC_FORMAT, HostGaussianProcessClassifier
from ._solver_impl import ElasticNet, Lasso, _CD_FORMAT
from ._svm_impl import SVC, SVR, _SVC_FORMAT, _SVR_FORMAT
from ._umap_impl import UMAP, _UMAP_FORMAT
from .decomposition import PCA, TruncatedSVD, _PCA_FORMAT, _TSVD_FORMAT
from ._hierarchy_impl import AgglomerativeClustering, _AGGLOMERATIVE_FORMAT
from .density import DBSCAN, KernelDensity, _DBSCAN_FORMAT, _KDE_FORMAT
from .kernel_methods import (
    KernelRidge, Nystroem, RBFSampler, _KERNEL_RIDGE_FORMAT, _NYSTROEM_FORMAT,
    _RBF_SAMPLER_FORMAT,
)
from .linear_model import (
    LinearRegression, LogisticRegression, Ridge, _LINEAR_FORMAT,
    _LOGISTIC_FORMAT,
)
from .neighbors import (
    KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors, _KNN_FORMAT,
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
    "_mojolearn_arima": "_mojolearn_forecast_host",
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


class _HostKNN(_HostBound):
    """The three k-NN host classes' shared refusal: the random ball cover
    arm has no host entry (`rbc_knn_search` is absent from the core host
    binding and would refuse by name at the first query; this names it at
    load instead)."""

    def _host_refusals(self):
        if self.algorithm == "rbc":
            raise ImportError(
                "mojolearn: no CPU implementation of _mojolearn.rbc_knn_search "
                f"yet; the host {type(self).__name__} runs the brute arm only "
                "(docs/lanes/BRIEF_forest_host_inference_2026-09-13.md)"
            )


class HostNearestNeighbors(_HostKNN, NearestNeighbors):
    _HOST_ARRAYS = ("_index",)


class HostKNeighborsClassifier(_HostKNN, KNeighborsClassifier):
    _HOST_ARRAYS = ("_index", "_y_cols")


class HostKNeighborsRegressor(_HostKNN, KNeighborsRegressor):
    _HOST_ARRAYS = ("_index", "_y_cols")


class HostARIMA(_HostBound, ARIMA):
    """A saved ARIMA model on the forecast inference binding, which exports
    `arima_predict` and `arima_forecast` and no `arima_fit`."""
    _HOST_ARRAYS = ("_y", "params_")


class HostUMAP(_HostBound, UMAP):
    """A saved UMAP embedding on the metrics host binding. `transform`
    answers the GPU's bytes for the same query batch; the answer for a row
    depends on the batch it is asked in (umap/transform.mojo)."""
    _HOST_ARRAYS = ("_transform_training", "_transform_embedding")


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
    """The kernel methods' host classes. The host restatement serves the
    linear and rbf kernels; a model saved with another kernel is refused by
    name at load, not at the first predict."""

    _BINDING = "_mojolearn_kernel_methods"

    def _host_refusals(self):
        from .kernel_methods import KERNEL_LINEAR, KERNEL_RBF
        kernel = self._kernel_params[0]
        if kernel not in (KERNEL_LINEAR, KERNEL_RBF):
            raise ImportError(
                f"mojolearn: no CPU implementation of {type(self).__name__} with "
                f"kernel code {kernel}; the host serves the linear and rbf kernels only"
            )


class HostKernelRidge(_HostKernelMethod, KernelRidge):
    _HOST_ARRAYS = ("X_fit_", "dual_coef_")


class HostNystroem(_HostKernelMethod, Nystroem):
    _HOST_ARRAYS = ("components_", "component_indices_", "normalization_",
                    "eigenvalues_", "eigenvectors_")


class HostRBFSampler(_HostBound, RBFSampler):
    _BINDING = "_mojolearn_kernel_methods"
    _HOST_ARRAYS = ("random_weights_", "random_offset_")


#: format tag -> (estimator name, host class). A file whose `estimator`
#: member names another class is refused by that class's own `load`.
_FORMATS = {
    _ARIMA_FORMAT: {"ARIMA": HostARIMA},
    _UMAP_FORMAT: {"UMAP": HostUMAP},
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
    _SVR_FORMAT: {"SVR": HostSVR},
    _DBSCAN_FORMAT: {"DBSCAN": HostDBSCAN},
    _AGGLOMERATIVE_FORMAT: {"AgglomerativeClustering": HostAgglomerativeClustering},
    _KNN_FORMAT: {
        "NearestNeighbors": HostNearestNeighbors,
        "KNeighborsClassifier": HostKNeighborsClassifier,
        "KNeighborsRegressor": HostKNeighborsRegressor,
    },
    # A saved Cholesky factor (lane/inference-embedding-ivf-cholesky,
    # 2026-09-15): `HostCholesky` solves on `_mojolearn_linalg_host`.
    _CHOLESKY_FORMAT: {"Cholesky": HostCholesky},
    # A saved GaussianProcessClassifier (lane/gaussian-process-classifier,
    # 2026-09-15): predicts on `_mojolearn_gp_host`.
    _GPC_FORMAT: {"GaussianProcessClassifier": HostGaussianProcessClassifier},
}
CLASSICAL_FORMATS = tuple(_FORMATS)


def host_model(path):
    """The host model for a saved classical file, by its `format` and
    `estimator` members. Any other format is refused with the tag it
    carries."""
    arrays = _serialize.read_npz(path, CLASSICAL_FORMATS)
    fmt = _serialize.scalar_str(arrays, "format")
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
