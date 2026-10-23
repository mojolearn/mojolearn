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
RF probabilities float32, RF regression float32, ET probabilities and
regression float64, class labels through `_labels.decode_labels`.

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
from . import _serialize
from ._array import Array
from ._buffer import addr, addr_ro, as_f32_c, empty
from ._labels import argmax_rows, classes_from_member, decode_labels

_EXTENSION = '_mojolearn_forest_host'
_MODULE_NAME = 'mojolearn._host.' + _EXTENSION
_IDENTICAL_CODE = 1


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
_ENGINES = ('sequential', 'parallel_groves')
#: The binding's family codes for the grove entries.
_GROVES_FAMILY = {'rf': 0, 'et': 1}
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
    same file: random forest probabilities and regression values as float32,
    extra-trees probabilities and regression values as float64. The model
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

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> y = X[:, 0] * 2.0
    >>> mojolearn.RandomForestRegressor(n_estimators=20).fit(X, y).save("rf.npz")
    >>> pred = mojolearn.HostForest.from_file("rf.npz").predict(X)
    """

    def __init__(self, *, estimator, offsets, colid, quesval, left_child, leaves,
                 n_features, n_trees, num_outputs, classes=None, device=None,
                 inference_engine='sequential'):
        return _native_host_archive(self, "host_tree_init_forest",
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

    @property
    def n_trees(self):
        """Number of trees in the model."""
        return _native_host_archive(self, "host_tree_count", (), _ARCHIVE_CONTEXT)

    @property
    def inference_engine(self):
        """The inference engine the file named: 'sequential' or
        'parallel_groves'. Read-only."""
        return _native_host_archive(self, "host_tree_engine", (), _ARCHIVE_CONTEXT)

    def _groves_handle(self):
        """The host grove snapshot of this model (validated once, released
        with the model), prepared on the first groves prediction."""
        return _native_host_archive(self, "host_tree_groves", (), _ARCHIVE_CONTEXT)

    def __getstate__(self):
        return _native_host_archive(self, "host_tree_getstate", (), _ARCHIVE_CONTEXT)

    def __setstate__(self, state):
        return _native_host_archive(self, "host_tree_setstate", (state,), _ARCHIVE_CONTEXT)

    def model_sha256(self):
        """SHA-256 hex digest of the model arrays' bytes, in save order.

        Identifies the exact model used for a prediction.
        """
        return _native_host_archive(self, "host_tree_hash", (), _ARCHIVE_CONTEXT)

    def _vote(self, X):
        """The divided vote, `(n_rows, num_outputs)` float32, what the GPU
        bindings' sequential entries return before the Python layer's argmax
        or widening."""
        return _native_host_archive(self, "host_tree_vote", (X,), _ARCHIVE_CONTEXT)

    def predict_proba(self, X):
        """Class probabilities, for classifier models only.

        Returns the same values as the estimator's own `predict_proba`:
        float32 for a random forest, float64 for extra trees, shape
        (n_samples, n_classes). A regressor model raises `AttributeError`.
        """
        return _native_host_archive(self, "host_tree_proba", (X,), _ARCHIVE_CONTEXT)

    def _predict_with_proba(self, X):
        """The two classifier views from one host traversal.

        Kept in lockstep with the RF/ET estimator helper used by the identity
        harness, including Extra Trees' exact float32-to-float64 widening.
        """
        return _native_host_archive(self, "host_tree_predict_pair", (X,), _ARCHIVE_CONTEXT)

    def predict(self, X):
        """Predict with the saved model.

        For a classifier, the class with the highest probability (first
        maximum winning) mapped through `classes_`. For a regressor, the mean
        of the trees' predictions: float32 for a random forest, float64 for
        extra trees.
        """
        return _native_host_archive(self, "host_tree_predict", (X,), _ARCHIVE_CONTEXT)


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
from . import _classical_host as _archive_registry
