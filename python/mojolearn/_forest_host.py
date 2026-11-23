# SPDX-License-Identifier: Apache-2.0
"""CPU inference for saved RandomForest and ExtraTrees models (the forest host
lane, 2026-09-13).

`HostForest.from_file(path)` loads a model written by `RandomForest*.save` or
`ExtraTrees*.save` and predicts on the CPU through `_mojolearn_forest_host`,
compiled with no accelerator target, running whichever of the two inference
engines the archive names (docs/FOREST_INFERENCE_ENGINES.md): the sequential
walk for a `mojolearn-randomforest-1` or `mojolearn-extratrees-1` archive,
and, since lane/forest-groves-cpu-and-speed (2026-09-17), the
`parallel_groves` engine for a `-parallel-groves-1` archive (32 fixed tree
groups, the 16/8/4/2/1 fold, `core/forest_host_groves.mojo`), so a groves
model trained on a GPU infers the GPU groves engine's bits on a CPU. A groves
archive saved under a numeric mode other than `identical` is refused by name,
because this binding computes IDENTICAL bits only. `predict` and
`predict_proba` return exactly what the GPU classes return for the same file.
RF probabilities float32, RF and ET regression float32, ET probabilities
float64, class labels through `_labels.decode_labels`.

This module holds no arithmetic. What it promises is what the gate measured:
tools/forest_host_gate.py compares the host predictions of a recorded model
and fixture against the SHA-256 a GPU run recorded. A CPU on which that gate
has not passed is not certified, whatever this code returns on it.

The binary is loaded by path and not through `_backend.load_set`, because
that selector refuses a binary whose vendor read-back is not a GPU API, and
this one reads back `cpu` by design. WHICH DIRECTORY is still
`_backend.host_module_path`'s answer, so `MOJOLEARN_HOST_DIR` moves this
door along with every other host binding. It did not until
lane/host-path-resolution (2026-09-16), and a process told to use another
set loaded the package's own forest binding here instead, which is why
tools/identity_break.py guards every host prediction against the binary it
asked for. `MOJOLEARN_FOREST_HOST_BINARY` names a different file and still
wins over the directory, which is how the gate loads its sabotage build; a
sabotage build is refused unless `MOJOLEARN_FOREST_HOST_ALLOW_SABOTAGE=1`.
"""
import hashlib
import importlib.machinery
import importlib.util
import os
import sys
import weakref

from . import _backend
from ._array import Array
from ._buffer import addr, addr_ro, empty
from . import _digest_fields
from ._mode import CheckMixin as _CheckMixin

_EXTENSION = '_mojolearn_forest_host'
_MODULE_NAME = 'mojolearn._host.' + _EXTENSION


# THE LOADER LIVES IN `_backend` (lane/lazy-imports, 2026-10-10):
# `_buffer._host_native` needs only this binary, and importing this whole
# door from the buffer helpers put its imports in every lane's closure.
def binary_path():
    """The binary this process loads, or would load: the file
    `MOJOLEARN_FOREST_HOST_BINARY` names, else this install's host binding
    of that name (`_backend.door_host_binary_path`)."""
    return _backend.door_host_binary_path(_EXTENSION)


def _load():
    """`_mojolearn_forest_host`, loaded once and read back (`_backend.load_door_host`)."""
    return _backend.load_door_host(_EXTENSION)


#: The sequential archive formats and the family each belongs to; a
#: `-parallel-groves-1` archive of either is the same family under the
#: other engine.
_FORMATS = {
    'mojolearn-randomforest-1': 'rf',
    'mojolearn-extratrees-1': 'et',
}
_GROVES_SUFFIX = '-parallel-groves-1'
#: The binding's family codes for the grove entries.
_ESTIMATORS = {
    'RandomForestClassifier': ('rf', True),
    'RandomForestRegressor': ('rf', False),
    'ExtraTreesClassifier': ('et', True),
    'ExtraTreesRegressor': ('et', False),
}


_MODEL_ARRAYS = (('offsets', '<i4'), ('colid', '<i4'), ('quesval', '<f4'),
                 ('left_child', '<i4'), ('leaves', '<f4'))


class HostForest:
    """A saved random forest or extra-trees model that predicts on the CPU.

    Build one with `HostForest.from_file(path)` from a file written by
    `RandomForestClassifier.save`, `RandomForestRegressor.save`,
    `ExtraTreesClassifier.save` or `ExtraTreesRegressor.save`. Prediction
    runs the inference engine the file names ('sequential' or
    'parallel_groves') and returns what the GPU estimator returns for the
    same file: random forest probabilities as float32, extra-trees
    probabilities as float64, every regression value as float32. The model
    arrays are used exactly as saved, never cast; a file whose tree arrays
    are inconsistent raises `ValueError`.

    Attributes
    ----------
    estimator : str
        Name of the estimator class that saved the model.
    classes_ : list
        Class labels (classifiers only).
    n_classes_ : int
        Number of classes (classifiers only).
    n_features_in_ : int
        Number of features the model expects.
    device : str or None
        The device recorded in the file.
    digest_ : str
        The hash of the model: the `digest_` of the estimator that saved
        the file.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> y = X[:, 0] * 2.0
    >>> mojolearn.RandomForestRegressor(n_estimators=20).fit(X, y).save("rf.npz")
    >>> pred = mojolearn.HostForest.from_file("rf.npz").predict(X)
    """

    #: A predict-only member of a meta-estimator or an explainer (core/forest_member_run.mojo: family 14, kind 2,
    #: the saved model imported from its own arrays; an extra-trees file is the ExtraTrees family's).
    _mojolearn_member = (14, 2)
    _mojolearn_member_predict_only = True

    def __init__(self, *, estimator, offsets, colid, quesval, left_child, leaves,
                 n_features, n_trees, num_outputs, classes=None, device=None,
                 inference_engine='sequential'):
        return _native_host_archive(self, _OP.host_tree_init_forest,
            (estimator, offsets, colid, quesval, left_child, leaves,
             n_features, n_trees, num_outputs, classes, device, inference_engine), _ARCHIVE_CONTEXT)

    @classmethod
    def from_file(cls, path):
        """Load a forest from a file written by a random forest or
        extra-trees `save`.

        The model predicts through the inference engine the file names. A
        model saved with `inference_engine='parallel_groves'` must have been
        saved under `numeric_mode='identical'`; otherwise this raises
        `ValueError`, because the CPU engine computes `identical` results
        only.

        Parameters
        ----------
        path : str or path-like
            The saved model file.

        Returns
        -------
        HostForest
        """
        return _native_host_archive(cls, "host_archive_load", (path,), _ARCHIVE_CONTEXT)

    # `digest_`: the shared estimator property (python/mojolearn/_mode.py
    # `CheckMixin.digest_`, one native call), hashed under the public class
    # the file was written by and from the steps that class is hashed by, so
    # a forest and this CPU copy of its file have one digest (fx-audit 32).
    digest_ = _CheckMixin.__dict__["digest_"]

    @property
    def n_trees(self):
        """Number of trees in the model."""
        return _native_host_archive(self, _OP.host_tree_count, (), _ARCHIVE_CONTEXT)

    @property
    def inference_engine(self):
        """The inference engine the file named: 'sequential' or
        'parallel_groves'. Read-only."""
        return _native_host_archive(self, _OP.host_tree_engine, (), _ARCHIVE_CONTEXT)


    def __getstate__(self):
        return _native_host_archive(self, _OP.host_tree_getstate, (), _ARCHIVE_CONTEXT)

    def __setstate__(self, state):
        return _native_host_archive(self, _OP.host_tree_setstate, (state,), _ARCHIVE_CONTEXT)

    def model_sha256(self):
        """SHA-256 hex digest of the model arrays' bytes, in save order.

        Identifies the exact model used for a prediction.
        """
        return _native_host_archive(self, _OP.host_tree_hash, (), _ARCHIVE_CONTEXT)

    def _vote(self, X):
        """The divided vote, `(n_rows, num_outputs)` float32, what the GPU
        bindings' sequential entries return before the Python layer's argmax
        or widening."""
        return _native_host_archive(self, _OP.host_tree_vote, (X,), _ARCHIVE_CONTEXT)

    def predict_proba(self, X):
        """Class probabilities, for classifier models only.

        Returns the same values as the estimator's own `predict_proba`:
        float32 for a random forest, float64 for extra trees, shape
        (n_samples, n_classes). A regressor model raises `AttributeError`.
        """
        return _native_host_archive(self, _OP.host_tree_proba, (X,), _ARCHIVE_CONTEXT)

    def predict(self, X):
        """Predict with the saved model.

        For a classifier, the class with the highest probability (first
        maximum winning) mapped through `classes_`. For a regressor, the mean
        of the trees' predictions, as float32 (a NumPy array when NumPy is
        installed), as the estimator's own `predict` returns it.
        """
        return _native_host_archive(self, _OP.host_tree_predict, (X,), _ARCHIVE_CONTEXT)


# Maintainer note (moved from the docstring): HostGBDT lives in
# `_gbdt_host.py`; the classical and knn host models in `_classical_host.py`.
def host_model(path):
    """Load a saved model for prediction on the CPU.

    Chooses the CPU model from the file's format: a `HostForest` for a random
    forest or extra-trees file, a `HostGBDT` for a `GradientBoosting.save`
    file, or the CPU model for a saved LinearRegression, Ridge, TruncatedSVD,
    LogisticRegression, PCA, NearestNeighbors, KNeighborsClassifier,
    KNeighborsRegressor, StandardScaler, MinMaxScaler, ElasticNet, Lasso,
    KernelRidge, Nystroem or RBFSampler. Any other format raises an error
    naming the format.

    Parameters
    ----------
    path : str or path-like
        The saved model file.

    Returns
    -------
    model
        An object with `predict` (and `predict_proba` for classifiers).

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> mojolearn.RandomForestRegressor(n_estimators=20).fit(X, X[:, 0]).save("rf.npz")
    >>> pred = mojolearn.host_model("rf.npz").predict(X)
    """
    return _native_host_archive(None, "host_archive_model", (path,), _ARCHIVE_CONTEXT)


def host_predict(path, X):
    """Predict on the CPU with a saved model: `host_model(path).predict(X)`.

    Parameters
    ----------
    path : str or path-like
        The saved model file.
    X : array-like of shape (n_samples, n_features)
        Input data.

    Returns
    -------
    Array or list of shape (n_samples,)
        Predictions, as the saved model's `predict` returns them.
    """
    return _native_host_archive(None, "host_archive_predict", (path, X), _ARCHIVE_CONTEXT)


def host_predict_proba(path, X):
    """Class probabilities on the CPU from a saved classifier:
    `host_model(path).predict_proba(X)`.

    Parameters
    ----------
    path : str or path-like
        The saved model file.
    X : array-like of shape (n_samples, n_features)
        Input data.

    Returns
    -------
    Array of shape (n_samples, n_classes)
        Class probabilities.
    """
    return _native_host_archive(None, "host_archive_predict_proba", (path, X), _ARCHIVE_CONTEXT)


# Fixed public archive owner connected at import, after host classes exist.
from .linear_model import _native_classical as _native_host_archive, _CLASSICAL as _ARCHIVE_CONTEXT
# Lane ne1-h-labels: operations pass as integer codes from the one native
# table (bindings/native_operation_codes.mojo), read once at import.
from ._native_ops import codes as _operation_codes  # noqa: E402
_OP = _operation_codes()
from . import _classical_host as _archive_registry

# TreeExplainer's native owner (bindings/trees_explainer_native_door.mojo) reads a
# HostForest's flat arrays; connect the exact class once, at import.
from . import _native_trees_bridge as _explainer_bridge
_explainer_bridge.connect()[1].setdefault("trees_explainer_host_forest", HostForest)
del _explainer_bridge

# glue: once at import, class metadata only. `digest_` of a HostForest is the
# digest of the forest that wrote its file: hashed under the `estimator` the
# file names, through the forest steps of that family, with the engine the
# file named (_digest_fields.forest_steps; fx-audit 32).
_digest_fields.twin_field(HostForest, "estimator")
_digest_fields.schema(HostForest, tuple(
    ("when", "family", family, _digest_fields.forest_steps(fmt, _digest_fields.host_forest_engine(fmt)))
    for fmt, family in _FORMATS.items()))  # glue: the two saved forest families
# The digest reads the instance through the shared family admission, which
# holds the classes connected when their module was loaded: this module loads
# on first touch of HostForest, after the package's connection.
from ._native_family_layouts import connect as _connect_family_layouts  # noqa: E402
_connect_family_layouts(sys.modules[__package__], only=(HostForest,))
del _connect_family_layouts
# glue: once at import. This module loads lazily, possibly after the model selection hub connected its marked
# classes: HostForest's predict-only member plan is registered here (lane mf-cal5).
from ._native_msel_more import connect_host_tree_member as _connect_host_tree_member  # noqa: E402
_connect_host_tree_member(HostForest)
del _connect_host_tree_member
