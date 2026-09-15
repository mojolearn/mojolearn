# SPDX-License-Identifier: Apache-2.0
"""CPU inference for saved RandomForest and ExtraTrees models (the forest host
lane, 2026-09-13).

`HostForest.from_file(path)` loads a model written by `RandomForest*.save` or
`ExtraTrees*.save` and predicts on the CPU through `_mojolearn_forest_host`,
the sequential inference algorithm (docs/FOREST_INFERENCE_ENGINES.md) compiled
with no accelerator target. `predict` and `predict_proba` return exactly what
the GPU classes return for the same file. RF probabilities float32, RF
regression float32, ET probabilities and regression float64, class labels
through `_labels.decode_labels`.

This module holds no arithmetic. What it promises is what the gate measured:
tools/forest_host_gate.py compares the host predictions of a recorded model
and fixture against the SHA-256 a GPU run recorded, and the brief in
docs/lanes/BRIEF_forest_host_inference_2026-09-13.md records on which CPUs
that has passed. A CPU not listed there is not certified, whatever this code
returns on it.

The binary is loaded from `mojolearn/host/` by path and not through
`_backend.load_set`, because that selector refuses a binary whose vendor
read-back is not a GPU API, and this one reads back `cpu` by design.
`MOJOLEARN_FOREST_HOST_BINARY` names a different file, which is how the gate
loads its sabotage build; a sabotage build is refused unless
`MOJOLEARN_FOREST_HOST_ALLOW_SABOTAGE=1`.
"""
import hashlib
import importlib.machinery
import importlib.util
import os
import sys

from . import _serialize
from ._array import Array
from ._buffer import addr, addr_ro, as_f32_c, empty
from ._labels import argmax_rows, classes_from_member, decode_labels

_EXTENSION = '_mojolearn_forest_host'
_MODULE_NAME = 'mojolearn._host.' + _EXTENSION
_IDENTICAL_CODE = 1
_MODULE = None

#: The sequential archive formats and the family each belongs to. A
#: `-parallel-groves-1` archive is read so it can be refused by name.
_FORMATS = {
    'mojolearn-randomforest-1': 'rf',
    'mojolearn-extratrees-1': 'et',
}
_GROVES_SUFFIX = '-parallel-groves-1'
_ESTIMATORS = {
    'RandomForestClassifier': ('rf', True),
    'RandomForestRegressor': ('rf', False),
    'ExtraTreesClassifier': ('et', True),
    'ExtraTreesRegressor': ('et', False),
}


def binary_path():
    """The binary this process loads, or would load."""
    override = os.environ.get('MOJOLEARN_FOREST_HOST_BINARY', '').strip()
    if override:
        return os.path.abspath(override)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'host', _EXTENSION + '.so')


def _load():
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    path = binary_path()
    if not os.path.exists(path):
        raise ImportError(
            f"mojolearn: {path} is not built. Build it with "
            "bindings/build_forest_host.sh")
    module = sys.modules.get(_MODULE_NAME)
    if module is None:
        loader = importlib.machinery.ExtensionFileLoader(_MODULE_NAME, path)
        spec = importlib.util.spec_from_loader(_MODULE_NAME, loader, origin=path)
        module = importlib.util.module_from_spec(spec)
        loader.exec_module(module)
        sys.modules[_MODULE_NAME] = module
    if int(module.forest_host_numeric_mode()) != _IDENTICAL_CODE:
        raise RuntimeError(f"mojolearn: {path} was not compiled IDENTICAL; rebuild it")
    if str(module.forest_host_vendor()) != 'cpu':
        raise RuntimeError(f"mojolearn: {path} does not read back as the CPU binding")
    if bool(module.forest_host_sabotage()) and os.environ.get('MOJOLEARN_FOREST_HOST_ALLOW_SABOTAGE') != '1':
        raise RuntimeError(
            f"mojolearn: {path} is the gate's SABOTAGE build and computes "
            "wrong answers on purpose; it is refused outside the gate")
    _MODULE = module
    return module


_MODEL_ARRAYS = (('offsets', '<i4'), ('colid', '<i4'), ('quesval', '<f4'),
                 ('left_child', '<i4'), ('leaves', '<f4'))


class HostForest:
    """A saved forest that predicts on the CPU. The model arrays are the
    file's bytes, exact dtypes, never cast; the binding refuses a file whose
    tree spans, child indices or column ids do not fit its own arrays."""

    def __init__(self, *, estimator, offsets, colid, quesval, left_child, leaves,
                 n_features, n_trees, num_outputs, classes=None, device=None):
        if estimator not in _ESTIMATORS:
            raise ValueError(f"mojolearn: {estimator!r} is not a forest estimator this loader reads")
        self.estimator = estimator
        self.family, self.is_classifier = _ESTIMATORS[estimator]
        self.device = device
        arrays = dict(offsets=offsets, colid=colid, quesval=quesval,
                      left_child=left_child, leaves=leaves)
        for name, dtype in _MODEL_ARRAYS:
            a = arrays[name]
            if not isinstance(a, Array) or a.dtype != dtype or a.ndim != 1 or a.order != 'C':
                raise ValueError(f"mojolearn: model array {name!r} must be a 1-D C-order {dtype} Array")
        for name, value in (('n_features', n_features), ('n_trees', n_trees), ('num_outputs', num_outputs)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"mojolearn: {name} must be a positive int")
        nodes = colid.size
        if (offsets.size != n_trees + 1 or nodes < n_trees or quesval.size != nodes
                or left_child.size != nodes or leaves.size != nodes * num_outputs
                or int(offsets[0]) != 0 or int(offsets[-1]) != nodes):
            raise ValueError("mojolearn: forest model array shapes do not match metadata")
        if self.is_classifier:
            if classes is None or len(classes) != num_outputs:
                raise ValueError("mojolearn: a classifier archive must carry num_outputs classes")
            self.classes_ = list(classes)
            self.n_classes_ = len(self.classes_)
        elif num_outputs != 1:
            raise ValueError("mojolearn: a regressor archive must carry one output")
        self._offsets = offsets
        self._colid = colid
        self._quesval = quesval
        self._left_child = left_child
        self._leaves = leaves
        self.n_features_in_ = int(n_features)
        self._n_trees = int(n_trees)
        self._num_outputs = int(num_outputs)
        self._binding = _load()

    @classmethod
    def from_file(cls, path):
        """A forest from a file written by `RandomForest*.save` or
        `ExtraTrees*.save`. A `parallel_groves` archive is refused by name,
        because the predictions it was saved beside are the other inference
        algorithm's bits and this loader runs the sequential one."""
        accepted = tuple(_FORMATS) + tuple(f + _GROVES_SUFFIX for f in _FORMATS)
        arrays = _serialize.read_npz(path, accepted)
        fmt = _serialize.scalar_str(arrays, 'format')
        if fmt.endswith(_GROVES_SUFFIX):
            raise ValueError(
                f"mojolearn: {path!r} is a {fmt} archive; the host engine runs the "
                "sequential algorithm and does not reproduce parallel_groves bits. "
                "Save the model with inference_engine='sequential'")
        estimator = _serialize.scalar_str(arrays, 'estimator')
        if estimator not in _ESTIMATORS or _ESTIMATORS[estimator][0] != _FORMATS[fmt]:
            raise ValueError(f"mojolearn: {path!r} was saved by {estimator}, which {fmt} does not hold")
        fields = {name: _serialize.exact(arrays, name, dtype) for name, dtype in _MODEL_ARRAYS}
        meta = _serialize.exact(arrays, 'meta', '<i8')
        if meta.size < 3:
            raise ValueError(f"mojolearn: {path!r} meta holds {meta.size} fields, at least 3 are needed")
        classes = classes_from_member(arrays['classes']) if 'classes' in arrays else None
        return cls(estimator=estimator, n_features=int(meta[0]), n_trees=int(meta[1]),
                   num_outputs=int(meta[2]), classes=classes,
                   device=_serialize.scalar_str(arrays, 'device') if 'device' in arrays else None,
                   **fields)

    @property
    def n_trees(self):
        return self._n_trees

    @property
    def inference_engine(self):
        return 'sequential'

    def model_sha256(self):
        """SHA-256 over the five model arrays' bytes in save order, for a
        report to name the model it predicted with."""
        h = hashlib.sha256()
        for a in (self._offsets, self._colid, self._quesval, self._left_child, self._leaves):
            h.update(a.tobytes())
        return h.hexdigest()

    def _vote(self, X):
        """The divided vote, `(n_rows, num_outputs)` float32, what the GPU
        bindings' sequential entries return before the Python layer's argmax
        or widening."""
        Xa, _ = as_f32_c(X, ndim=2, name='X')
        n_rows, n_features = Xa.shape
        if n_features != self.n_features_in_:
            raise ValueError(f"X has {n_features} features, fit saw {self.n_features_in_}")
        out = empty((n_rows * self._num_outputs,), '<f4')
        if self.family == 'rf':
            entry = 'forest_host_rf_predict_proba' if self.is_classifier else 'forest_host_rf_predict_reg'
        else:
            entry = 'forest_host_et_predict'
        wrote = getattr(self._binding, entry)(
            [addr_ro(self._offsets, name='offsets'), addr_ro(self._colid, name='colid'),
             addr_ro(self._quesval, name='quesval'), addr_ro(self._left_child, name='left_child'),
             addr_ro(self._leaves, name='leaves'), addr_ro(Xa, name='X'), addr(out, name='out')],
            [int(n_rows), int(n_features), self._n_trees, self._num_outputs, int(self._colid.size)])
        if int(wrote) != n_rows:
            raise RuntimeError(f"{entry} wrote {wrote} of {n_rows} rows")
        return out.reshape((n_rows, self._num_outputs))

    def predict_proba(self, X):
        """Classifiers only. RF gives the float32 vote fractions
        (`RandomForestClassifier.predict_proba`), ET the same vote widened
        exactly to float64 (`ExtraTreesClassifier.predict_proba`)."""
        if not self.is_classifier:
            raise AttributeError(f"{self.estimator} has no predict_proba")
        vote = self._vote(X)
        return vote if self.family == 'rf' else vote.astype('<f8')

    def predict(self, X):
        """Classifiers, the argmax of the vote (first max wins) mapped
        through `classes_`. Regressors, the forest mean per row, float32 for
        RF and float64 for ET, as the GPU classes return it."""
        vote = self._vote(X)
        if self.is_classifier:
            return decode_labels(self.classes_, argmax_rows(vote))
        flat = vote.reshape((vote.shape[0],))
        return flat if self.family == 'rf' else flat.astype('<f8')


def host_model(path):
    """The host model for a saved file, by its `format` member: a
    `HostForest` for a forest archive, a `HostGBDT` (`_gbdt_host.py`) for a
    `GradientBoosting.save` archive, a host subclass of LinearRegression,
    Ridge, TruncatedSVD, LogisticRegression or PCA (`_classical_host.py`,
    the classical host inference lane, 2026-09-13), of NearestNeighbors,
    KNeighborsClassifier or KNeighborsRegressor (the knn host inference
    lane, 2026-09-14), or of StandardScaler, MinMaxScaler, ElasticNet,
    Lasso, KernelRidge, Nystroem or RBFSampler (lane/inference-linear-svm,
    2026-09-15) for one of their archives. Any other format is refused with
    the tag it carries."""
    from ._classical_host import CLASSICAL_FORMATS
    from ._classical_host import host_model as classical_host_model
    from ._gbdt_host import GBDT_FORMAT, HostGBDT
    accepted = (tuple(_FORMATS) + tuple(f + _GROVES_SUFFIX for f in _FORMATS)
                + (GBDT_FORMAT,) + CLASSICAL_FORMATS)
    fmt = _serialize.scalar_str(_serialize.read_npz(path, accepted), 'format')
    if fmt == GBDT_FORMAT:
        return HostGBDT.from_file(path)
    if fmt in CLASSICAL_FORMATS:
        return classical_host_model(path)
    return HostForest.from_file(path)


def host_predict(path, X):
    """`host_model(path).predict(X)`."""
    return host_model(path).predict(X)


def host_predict_proba(path, X):
    """`host_model(path).predict_proba(X)`."""
    return host_model(path).predict_proba(X)
