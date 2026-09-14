# SPDX-License-Identifier: Apache-2.0
"""CPU inference for saved LinearRegression, Ridge, TruncatedSVD,
LogisticRegression and PCA models (the classical host inference lane,
2026-09-13; brief docs/lanes/BRIEF_forest_host_inference_2026-09-13.md,
"Classical lanes").

`host_model(path)` loads a file written by one of those classes' `save` and
returns an instance of a HOST SUBCLASS of the same class: the same Python
`predict`, `predict_proba`, `decision_function` or `transform` as the GPU
class, character for character, with ONE difference, `_bind` answers the
CPU binding `mojolearn/host/_mojolearn_estimators_host.so` instead of the
GPU set. That binding exports the GPU binding's names under the GPU
binding's address contracts (`bindings/_mojolearn_estimators_host.mojo`),
and its arithmetic is `core/classical_host_predict.mojo`, the restatement
of the pinned gemm/gemv kernel, the intercept and bias epilogues, the
centering kernel and the host sigmoid.

On a CPU-only install none of this is needed: `_backend._HOST_MODULES`
routes `_mojolearn_estimators` to the host binding and the plain classes'
`load` and `predict` run through it. These subclasses exist so that a box
WITH a GPU (the Mac that records the GPU answer) can run the host path in
the same process, which is how tools/classical_host_gate.py compares the
two bit for bit. The binding is loaded through `_backend.load_host_module`,
which honors MOJOLEARN_HOST_DIR (the gate's sabotage set) and refuses a
sabotage build unless MOJOLEARN_HOST_ALLOW_SABOTAGE=1.

This module holds no arithmetic. What it promises is what the gate
measured; the brief records on which CPUs that has passed.
"""
import hashlib

from . import _backend, _serialize
from .decomposition import PCA, TruncatedSVD, _PCA_FORMAT, _TSVD_FORMAT
from .linear_model import (
    LinearRegression, LogisticRegression, Ridge, _LINEAR_FORMAT,
    _LOGISTIC_FORMAT,
)

_HOST_BASENAME = "_mojolearn_estimators_host"


def binary_path():
    """The binary `host_model` loads, or would load."""
    return _backend.host_module_path(_HOST_BASENAME)


class _HostBound:
    """`_bind` answers the CPU binding for `_mojolearn_estimators` and
    refuses every other family by name, so a host subclass can never reach
    a GPU binding by accident."""

    def _bind(self, name=None):
        name = name or self._BINDING
        if name != "_mojolearn_estimators":
            raise ImportError(
                f"mojolearn: the host {type(self).__name__} serves "
                f"_mojolearn_estimators only, not {name}"
            )
        mode = getattr(self, "numeric_mode", None)
        if mode is not None and mode != "identical":
            raise ValueError(
                f"mojolearn: {type(self).__name__} runs IDENTICAL only on the "
                f"host; this model was saved {mode!r}"
            )
        return _backend.load_host_module(_HOST_BASENAME)

    def vendor_used(self):
        return "cpu"

    def model_sha256(self):
        """SHA-256 over the model file's bytes as `save` would write them
        again, for a report to name the model it predicted with."""
        h = hashlib.sha256()
        for name in sorted(self._HOST_ARRAYS):
            h.update(getattr(self, name).tobytes())
        return h.hexdigest()


class HostLinearRegression(_HostBound, LinearRegression):
    _HOST_ARRAYS = ("coef_",)


class HostRidge(_HostBound, Ridge):
    _HOST_ARRAYS = ("coef_",)


class HostLogisticRegression(_HostBound, LogisticRegression):
    _HOST_ARRAYS = ("_w",)


class HostTruncatedSVD(_HostBound, TruncatedSVD):
    _HOST_ARRAYS = ("components_", "singular_values_")


class HostPCA(_HostBound, PCA):
    _HOST_ARRAYS = ("components_", "mean_", "singular_values_")

    def _whiten_binding(self):
        raise ImportError(
            "mojolearn: no CPU implementation of _mojolearn_estimators."
            "pca_whiten_transform yet; the host PCA transforms whiten=False "
            "models only (docs/lanes/BRIEF_forest_host_inference_2026-09-13.md)"
        )

    def _dense_binding(self):
        raise ImportError(
            "mojolearn: no CPU implementation of the dense PCA fit; the host "
            "PCA transforms a saved model only"
        )


#: format tag -> (estimator name, host class). A file whose `estimator`
#: member names another class is refused by that class's own `load`.
_FORMATS = {
    _LINEAR_FORMAT: {"LinearRegression": HostLinearRegression, "Ridge": HostRidge},
    _LOGISTIC_FORMAT: {"LogisticRegression": HostLogisticRegression},
    _TSVD_FORMAT: {"TruncatedSVD": HostTruncatedSVD},
    _PCA_FORMAT: {"PCA": HostPCA},
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
    model.estimator = estimator
    return model
