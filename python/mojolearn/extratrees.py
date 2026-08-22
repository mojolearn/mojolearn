"""Extremely Randomized Trees on the GPU, mirroring sklearn's surface.

The learner is `extratrees/`: cuML's batched-levelalgo tree builder and
random forest with sklearn's `RandomSplitter` semantics on top -- the
histogram-free formulation (Geurts, Ernst & Wehenkel 2006) that neither cuML
nor LightGBM ships on a GPU. The defaults are scikit-learn's
`ExtraTreesClassifier` / `ExtraTreesRegressor` defaults, name for name.

EVERY sklearn PARAMETER IS EITHER HONOURED OR REFUSED BY NAME. None is
accepted and ignored -- a silently dropped option is indistinguishable, from
the caller's side, from one that works. Almost all refusals fire in the Mojo
layer (`extratrees/estimator.mojo`), which is the single place both the host
and device arms resolve their configuration; this wrapper refuses only what
never crosses the boundary: the criteria this port has not transcribed
(UNPORTED.tsv rows 7 and 11-14) and the two forest-level knobs that do not
exist here (`n_jobs`, `verbose`).

TWO DEVIATIONS FROM sklearn'S CONTRACT, STATED RATHER THAN HIDDEN:

* `random_state=None` is DETERMINISTIC here (seed 0). sklearn's None draws
  from the global RNG; this library's whole claim is bit-reproducibility, so
  an entry point that is nondeterministic by default would be the wrong
  default. Pass an int to choose a different seed.
* `device` is a constructor parameter sklearn does not have: "gpu" (default)
  runs the split search on the GPU, "cpu" runs the host transcription. The
  two produce the SAME forest for classification (bit-identical, the lane's
  device_forest_check) and structure-identical trees for regression whose
  leaf values differ by at most one fixed-point quantization step
  (deviation 135).

X IS COPIED TWICE PER FIT: once by NumPy to column-major float32 (the
builder's layout, cuML's own) and once across the boundary into Mojo. On
large matrices that is the dominant cost of the CALL and is named here so
nobody times it as the fit.
"""

import numpy as np

from . import _mojolearn_trees
from ._arrays import _addr, _addr_ro, as_f32_c

# estimator.mojo's max_features sentinels, same values in the same words.
_MF_SQRT = -1
_MF_LOG2 = -2
_MF_ALL = -3

_UNPORTED_CRITERIA = {
    # sklearn name -> (which estimator, the recorded reason)
    "entropy": ("classifier", "UNPORTED.tsv row 11: cuML objectives.cuh"
                ":110-193, marked 'not yet'"),
    "log_loss": ("classifier", "sklearn's alias of entropy; UNPORTED.tsv"
                 " row 11"),
    "friedman_mse": ("regressor", "no cuML counterpart; the exhaustive"
                     " splitter it serves is the ensemble/ lane"),
    "absolute_error": ("regressor", "UNPORTED.tsv row 7: MAE needs an order"
                       " statistic per candidate, a different kernel shape"),
    "poisson": ("regressor", "UNPORTED.tsv row 12: cuML objectives.cuh"
                ":267-346, marked 'not yet'"),
}


def _refuse_forest_knobs(n_jobs, verbose):
    if n_jobs is not None:
        raise NotImplementedError(
            "n_jobs is not ported: there is no CPU thread pool here -- the"
            " fit runs one host thread driving the GPU (device='gpu') or the"
            " host transcription (device='cpu'). Refused by name rather than"
            " accepted and ignored."
        )
    if verbose:
        raise NotImplementedError(
            "verbose is not ported: nothing in the Mojo layer logs per-tree"
            " progress. Refused by name rather than accepted and ignored."
        )


def _max_features_slots(max_features):
    """sklearn's max_features forms into (spec, fraction)."""
    if max_features is None:
        return _MF_ALL, 0.0
    if isinstance(max_features, str):
        if max_features == "sqrt":
            return _MF_SQRT, 0.0
        if max_features == "log2":
            return _MF_LOG2, 0.0
        raise ValueError(
            f"max_features={max_features!r} is not a recognised form; sklearn"
            " accepts 'sqrt', 'log2', None, a float fraction or an int count"
        )
    if isinstance(max_features, (int, np.integer)) and not isinstance(
        max_features, bool
    ):
        return int(max_features), 0.0
    return 0, float(max_features)


def _fit_params(n_rows, n_features, n_classes, cfg, device):
    """The 21-slot params list, in _mojolearn_trees.mojo's exact order:
    n_rows, n_features, n_classes, n_estimators, max_depth,
    min_samples_split, min_samples_leaf, min_weight_fraction_leaf,
    max_features_spec, max_features_fraction, min_impurity_decrease,
    bootstrap, oob_score, random_state, warm_start, ccp_alpha,
    has_class_weight, has_monotonic_cst, max_samples_set, max_leaf_nodes,
    device."""
    spec, fraction = _max_features_slots(cfg["max_features"])
    return [
        int(n_rows),
        int(n_features),
        int(n_classes),
        int(cfg["n_estimators"]),
        -1 if cfg["max_depth"] is None else int(cfg["max_depth"]),
        int(cfg["min_samples_split"]),
        int(cfg["min_samples_leaf"]),
        float(cfg["min_weight_fraction_leaf"]),
        spec,
        fraction,
        float(cfg["min_impurity_decrease"]),
        1 if cfg["bootstrap"] else 0,
        1 if cfg["oob_score"] else 0,
        0 if cfg["random_state"] is None else int(cfg["random_state"]),
        1 if cfg["warm_start"] else 0,
        float(cfg["ccp_alpha"]),
        0 if cfg["class_weight"] is None else 1,
        0 if cfg["monotonic_cst"] is None else 1,
        0 if cfg["max_samples"] is None else 1,
        -1 if cfg["max_leaf_nodes"] is None else int(cfg["max_leaf_nodes"]),
        1 if device == "gpu" else 0,
    ]


class _ExtraTreesBase:
    def __init__(self, device):
        if device not in ("gpu", "cpu"):
            raise ValueError(
                f"device must be 'gpu' or 'cpu', got {device!r}"
            )
        self.device = device

    def _fit_arrays(self, X, y, n_classes, fit_fn):
        Xa = np.asarray(X)
        if Xa.ndim != 2:
            raise ValueError(
                f"X must be 2-D, got {Xa.ndim}-D shape {Xa.shape}"
            )
        n_rows, n_features = Xa.shape
        if len(y) != n_rows:
            raise ValueError(
                f"y has {len(y)} rows, X has {n_rows}"
            )
        # Column-major is the builder's layout (cuML's `data` is
        # column-major); asfortranarray is that copy, named in the module
        # docstring.
        Xf = np.asfortranarray(Xa, dtype=np.float32)
        ya = np.ascontiguousarray(y, dtype=np.float32)
        params = _fit_params(
            n_rows, n_features, n_classes, self._cfg, self.device
        )
        out = fit_fn(_addr_ro(Xf), _addr_ro(ya), params)
        del Xf, ya  # the borrow ends with the call
        offsets, colid, quesval, left_child, leaves, meta = out
        self._offsets = np.asarray(offsets, dtype=np.int32)
        self._colid = np.asarray(colid, dtype=np.int32)
        self._quesval = np.asarray(quesval, dtype=np.float32)
        self._left_child = np.asarray(left_child, dtype=np.int32)
        self._leaves = np.asarray(leaves, dtype=np.float32)
        self.n_features_in_ = int(n_features)
        self._n_trees = int(meta[0])
        self._num_outputs = int(meta[1])
        self.depth_cap_bound_ = bool(meta[2])
        self.max_depth_resolved_ = int(meta[3])
        self.max_features_ = int(meta[4])
        return self

    def _vote(self, X):
        if not hasattr(self, "_offsets"):
            raise RuntimeError("this estimator is not fitted yet")
        Xa, _ = as_f32_c(X, "X")
        n_rows, n_features = Xa.shape
        if n_features != self.n_features_in_:
            raise ValueError(
                f"X has {n_features} features, fit saw {self.n_features_in_}"
            )
        out = np.empty(n_rows * self._num_outputs, dtype=np.float32)
        wrote = _mojolearn_trees.et_predict(
            _addr_ro(self._offsets),
            _addr_ro(self._colid),
            _addr_ro(self._quesval),
            _addr_ro(self._left_child),
            _addr_ro(self._leaves),
            _addr_ro(Xa),
            _addr(out),
            [int(n_rows), int(n_features), self._n_trees,
             self._num_outputs],
        )
        if wrote != n_rows:
            raise RuntimeError(
                f"et_predict wrote {wrote} of {n_rows} rows"
            )
        return out.reshape(n_rows, self._num_outputs)


class ExtraTreesClassifier(_ExtraTreesBase):
    """sklearn's `ExtraTreesClassifier`, honoured or refused by name.

    `predict_proba` is `forest_vote`'s average of per-tree leaf
    distributions (cuML `randomforest.cuh:229-242`); `predict` is its
    argmax mapped back through `classes_`.
    """

    def __init__(
        self,
        n_estimators=100,
        *,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=False,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
        monotonic_cst=None,
        device="gpu",
    ):
        super().__init__(device)
        _refuse_forest_knobs(n_jobs, verbose)
        if criterion != "gini":
            reason = _UNPORTED_CRITERIA.get(criterion)
            raise NotImplementedError(
                f"criterion={criterion!r} is not ported"
                + (f" ({reason[1]})" if reason else "")
                + "; only 'gini' is."
            )
        self._cfg = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            random_state=random_state,
            warm_start=warm_start,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
            monotonic_cst=monotonic_cst,
        )

    def fit(self, X, y):
        ya = np.asarray(y)
        self.classes_, codes = np.unique(ya, return_inverse=True)
        self.n_classes_ = int(len(self.classes_))
        return self._fit_arrays(
            X,
            codes.astype(np.float32),
            self.n_classes_,
            _mojolearn_trees.et_classifier_fit,
        )

    def predict_proba(self, X):
        return self._vote(X).astype(np.float64)

    def predict(self, X):
        return self.classes_[np.argmax(self._vote(X), axis=1)]


class ExtraTreesRegressor(_ExtraTreesBase):
    """sklearn's `ExtraTreesRegressor`, honoured or refused by name.

    With `device='gpu'` the leaf values are means of fixed-point quantized
    labels (deviation 135) and differ from the CPU arm's by at most one
    quantization step; the tree structure is identical.
    """

    def __init__(
        self,
        n_estimators=100,
        *,
        criterion="squared_error",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=False,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None,
        monotonic_cst=None,
        device="gpu",
    ):
        super().__init__(device)
        _refuse_forest_knobs(n_jobs, verbose)
        if criterion != "squared_error":
            reason = _UNPORTED_CRITERIA.get(criterion)
            raise NotImplementedError(
                f"criterion={criterion!r} is not ported"
                + (f" ({reason[1]})" if reason else "")
                + "; only 'squared_error' is."
            )
        self._cfg = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            random_state=random_state,
            warm_start=warm_start,
            class_weight=None,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
            monotonic_cst=monotonic_cst,
        )
        # sklearn's regressor default: all features. A float 1.0 means the
        # fraction form; estimator.mojo nudges the ratio to the middle of
        # its truncation bucket either way.
        if max_features == 1.0 and not isinstance(max_features, int):
            self._cfg["max_features"] = None

    def fit(self, X, y):
        return self._fit_arrays(
            X,
            np.ascontiguousarray(y, dtype=np.float32),
            0,
            _mojolearn_trees.et_regressor_fit,
        )

    def predict(self, X):
        return self._vote(X)[:, 0].astype(np.float64)
