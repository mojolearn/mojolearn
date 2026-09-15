# SPDX-License-Identifier: Apache-2.0
"""CPU inference for saved LinearRegression, Ridge, TruncatedSVD,
LogisticRegression and PCA models (the classical host inference lane,
2026-09-13; brief docs/lanes/BRIEF_forest_host_inference_2026-09-13.md,
"Classical lanes"), since the kde svc host lane (2026-09-14) saved
KernelDensity and SVC models, and since the knn host inference lane
(2026-09-14) NearestNeighbors, KNeighborsClassifier and KNeighborsRegressor.

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
"""
import hashlib

from . import _backend, _serialize
from ._cholesky_impl import _CHOLESKY_FORMAT, HostCholesky
from ._gpc_impl import _GPC_FORMAT, HostGaussianProcessClassifier
from ._svm_impl import SVC, _SVC_FORMAT
from .decomposition import PCA, TruncatedSVD, _PCA_FORMAT, _TSVD_FORMAT
from .density import KernelDensity, _KDE_FORMAT
from .linear_model import (
    LinearRegression, LogisticRegression, Ridge, _LINEAR_FORMAT,
    _LOGISTIC_FORMAT,
)
from .neighbors import (
    KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors, _KNN_FORMAT,
)

_HOST_BASENAME = "_mojolearn_estimators_host"
#: GPU family -> the host binding a host subclass of that family binds.
_HOST_BASENAMES = {
    "_mojolearn_estimators": _HOST_BASENAME,
    "_mojolearn_svm": "_mojolearn_svm_host",
    "_mojolearn": "_mojolearn_core_host",
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


#: format tag -> (estimator name, host class). A file whose `estimator`
#: member names another class is refused by that class's own `load`.
_FORMATS = {
    _LINEAR_FORMAT: {"LinearRegression": HostLinearRegression, "Ridge": HostRidge},
    _LOGISTIC_FORMAT: {"LogisticRegression": HostLogisticRegression},
    _TSVD_FORMAT: {"TruncatedSVD": HostTruncatedSVD},
    _PCA_FORMAT: {"PCA": HostPCA},
    _KDE_FORMAT: {"KernelDensity": HostKernelDensity},
    _SVC_FORMAT: {"SVC": HostSVC},
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
