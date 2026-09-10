# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
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
never crosses the boundary: the criteria this implementation has not transcribed
(NOT_IMPLEMENTED.tsv rows 7 and 12-14; row 11, entropy, is IMPLEMENTED -- DEVIATION
459) and the two forest-level knobs that do not exist here (`n_jobs`,
`verbose`). The one time this sentence was false --
the regressor's `max_features` rode across and was overwritten on the far
side (DEVIATION 458) -- `extratrees/tools/wrapper_reach_check.py` now gates
from this side of the boundary.

TWO DEVIATIONS FROM sklearn'S CONTRACT, STATED RATHER THAN HIDDEN:

* `random_state=None` is DETERMINISTIC here (seed 0). sklearn's None draws
  from the global RNG; this library's whole claim is bit-reproducibility, so
  an entry point that is nondeterministic by default would be the wrong
  default. Pass an int to choose a different seed.
* `device="gpu"` is retained for constructor compatibility. CPU training is
  retired from the public API; host reference trainers exist only for checks.
  GPU regression uses fixed-point quantized labels (deviation 135).

BOOTSTRAP (DEVIATION 460): `bootstrap=True` draws each tree's rows with
replacement through cuML's own sampler (the fnv1a32 `(seed, tree)` chain
feeding RAFT's Philox `uniformInt`, reused from the RF lane), and
`max_samples` is resolved exactly as sklearn's `_get_n_samples_bootstrap`
resolves it (None = n_rows, int = that count, float f = max(int(f *
n_rows), 1)). `oob_score` stays refused: the out-of-bag mask is not implemented.

`max_leaf_nodes` (2026-09-01): HONOURED, and it was refused until that day.
sklearn's meaning is best-first growth (`_tree.pyx:374-508`), which is a
different expansion ORDER and therefore a second growth mode in the Mojo
layer rather than a parameter on the existing one -- `extratrees/`
DEVIATION BLOCKS 466 to 469 carry the design, the frontier key, the tie rule
and what the mode costs in launches. Passing an int selects it and yields
exactly that many leaves per tree; `None`, the default, keeps the
depth-wise growth this library has always done, bit for bit. cuML's own leaf
budget, `max_leaves`, is a WEAKER and DIFFERENT guarantee -- a cap on the
breadth-first frontier that reorders nothing -- and the two are deliberately
not spelled alike and are never aliased onto each other. `max_leaves` is
honoured in the Mojo layer under its own name and is still NOT a keyword on
these classes, because the 22-slot params list below does not carry it and a
keyword that rode across and did nothing is precisely what this module
refuses to do.

TWO THINGS ABOUT BEST-FIRST THAT ARE NOT sklearn'S, stated here because a
caller comparing forests will see them. (1) The frontier's TIE RULE is ours:
on an equal improvement the smaller node id is expanded first. sklearn has
no rule -- its heap surfaces whichever equal record its layout puts on top
-- so there was nothing to inherit. (2) The improvement itself is computed
from cuML's gain as `(node_rows / tree_rows) * gain`, which is sklearn's
`impurity_improvement` rearranged, in pinned float32 rather than float64.
Both are gated by `extratrees/checks/bestfirst_check.mojo`.

X IS COPIED TWICE PER FIT: once on the host to column-major float32 (the
builder's layout, cuML's own; `_buffer.as_f32_colmajor`, zero-copy for a
float32 F-order input) and once across the boundary into Mojo. On large
matrices that is the dominant cost of the CALL and is named here so nobody
times it as the fit.

INPUTS AND OUTPUTS ARE NOT NumPy (DEVIATION 2343). `X` is anything the
buffer protocol exposes -- an ndarray, an `array.array`, a
`mojolearn.Array` -- or a nested list; `y` for the regressor likewise, and
for the classifier any sequence of label objects. `predict_proba` and the
regressor's `predict` return a float64 `mojolearn.Array`
(`numpy.asarray(result)` is a zero-copy view for a caller who has NumPy).
`classes_` is a PYTHON LIST of the caller's label objects in the order
`_labels.py` defines (DEVIATION 2340: numbers by value, strings by code
point, NaN refused), and the classifier's `predict` returns an int64 or
float64 `Array` for numeric labels and a Python list for str labels.
Nothing in this module imports NumPy.
"""

import numbers

from . import _mojolearn_trees, _serialize
from ._array import Array
from ._buffer import addr, addr_ro, as_f32_c, as_f32_colmajor, empty
from ._labels import (
    argmax_rows, classes_from_member, classes_member, decode_labels,
    flatten_labels, is_bool, sorted_classes,
)
from ._mode import NumericModeMixin
from ._forest_protocol import ForestProtocol, forest_estimator

#: The npz model-file format tag `save` writes and `load` requires.
_MODEL_FORMAT = "mojolearn-extratrees-1"

# estimator.mojo's max_features sentinels, same values in the same words.
_MF_SQRT = -1
_MF_LOG2 = -2
_MF_ALL = -3

# decisiontree.mojo's CRITERION_* codes, slot 21 of the params list.
_CRITERION_GINI = 0
_CRITERION_ENTROPY = 1
_CRITERION_MSE = 2

# sklearn's classifier criteria -> the code; 'log_loss' is sklearn's alias
# of 'entropy' (the same Entropy criterion object, _classes.py).
_CLASSIFIER_CRITERIA = {
    "gini": _CRITERION_GINI,
    "entropy": _CRITERION_ENTROPY,
    "log_loss": _CRITERION_ENTROPY,
}

_UNSUPPORTED_CRITERIA = {
    # sklearn name -> (which estimator, the recorded reason)
    "friedman_mse": ("regressor", "no cuML counterpart; the exhaustive"
                     " splitter it serves is the ensemble/ lane"),
    "absolute_error": ("regressor", "NOT_IMPLEMENTED.tsv row 7: MAE needs an order"
                       " statistic per candidate, a different kernel shape"),
    "poisson": ("regressor", "NOT_IMPLEMENTED.tsv row 12: cuML objectives.cuh"
                ":267-346, marked 'not yet'"),
}


def _refuse_forest_knobs(n_jobs, verbose):
    if n_jobs is not None:
        raise NotImplementedError(
            "n_jobs is not implemented: there is no CPU thread pool here -- the"
            " fit runs one host thread driving the GPU (device='gpu')."
            " Refused by name rather than"
            " accepted and ignored."
        )
    if verbose:
        raise NotImplementedError(
            "verbose is not implemented: nothing in the Mojo layer logs per-tree"
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
    if isinstance(max_features, numbers.Integral) and not is_bool(
        max_features
    ):
        return int(max_features), 0.0
    return 0, float(max_features)


def _n_samples_bootstrap(n_rows, max_samples):
    """sklearn's `_get_n_samples_bootstrap` (`ensemble/_bootstrap.py:10-62`,
    unweighted branch), as the COUNT slot 18 carries: None -> 0 (meaning
    n_rows), int -> itself, float f -> max(int(f * n_rows), 1). The value
    check is sklearn's `_parameter_constraints` (`_forest.py:198-202`)."""
    if max_samples is None:
        return 0
    if is_bool(max_samples):
        raise ValueError("max_samples must be None, an int >= 1 or a float")
    if isinstance(max_samples, numbers.Integral):
        if max_samples < 1:
            raise ValueError(
                f"max_samples={max_samples} must be >= 1 when an int"
            )
        return int(max_samples)
    f = float(max_samples)
    if not f > 0.0:
        raise ValueError(f"max_samples={max_samples} must be > 0 when a float")
    return max(int(f * n_rows), 1)


def _validate_device(device):
    if not isinstance(device, str) or device != "gpu":
        raise ValueError(
            f"Extra Trees training is GPU-only; device must be 'gpu', got {device!r}"
        )


def _fit_params(n_rows, n_features, n_classes, cfg, device, criterion):
    """The 22-slot params list, in _mojolearn_trees.mojo's exact order:
    n_rows, n_features, n_classes, n_estimators, max_depth,
    min_samples_split, min_samples_leaf, min_weight_fraction_leaf,
    max_features_spec, max_features_fraction, min_impurity_decrease,
    bootstrap, oob_score, random_state, warm_start, ccp_alpha,
    has_class_weight, has_monotonic_cst, max_samples (resolved count),
    max_leaf_nodes, device, criterion."""
    _validate_device(device)
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
        _n_samples_bootstrap(n_rows, cfg["max_samples"]),
        -1 if cfg["max_leaf_nodes"] is None else int(cfg["max_leaf_nodes"]),
        1,  # GPU-only ABI; constructor validation rejects other devices.
        int(criterion),
    ]


class _ExtraTreesBase(ForestProtocol, NumericModeMixin):
    _BINDING = "_mojolearn_trees"
    def __init__(self, device):
        _validate_device(device)
        self.device = device

    def _fit_arrays(self, X, ya, n_classes, fit_fn):
        # Column-major is the builder's layout (cuML's `data` is
        # column-major); `as_f32_colmajor` is that copy, named in the
        # module docstring, and zero for a float32 F-order input
        # (DEVIATION 2343). Its 2-D refusal reads "mojolearn: X must be
        # 2-D, got ...", the wording `_arrays.py` used. `ya` arrives as
        # a float32 `Array` from the caller.
        Xf, _ = as_f32_colmajor(X, name="X")
        n_rows, n_features = Xf.shape
        if len(ya) != n_rows:
            raise ValueError(
                f"y has {len(ya)} rows, X has {n_rows}"
            )
        params = _fit_params(
            n_rows, n_features, n_classes, self._cfg, self.device,
            self._criterion_code,
        )
        out = fit_fn(addr_ro(Xf, name="X"), addr_ro(ya, name="y"), params)
        del Xf, ya  # the borrow ends with the call
        offsets, colid, quesval, left_child, leaves, meta = out
        # The binding returns Python lists; packing them is the same
        # O(nodes) conversion `np.asarray(list)` was.
        self._offsets = Array.from_list([int(v) for v in offsets], "<i4")
        self._colid = Array.from_list([int(v) for v in colid], "<i4")
        self._quesval = Array.from_list([float(v) for v in quesval], "<f4")
        self._left_child = Array.from_list([int(v) for v in left_child], "<i4")
        self._leaves = Array.from_list([float(v) for v in leaves], "<f4")
        self.n_features_in_ = int(n_features)
        self._n_trees = int(meta[0])
        self._num_outputs = int(meta[1])
        self.depth_cap_bound_ = bool(meta[2])
        self.max_depth_resolved_ = int(meta[3])
        self.max_features_ = int(meta[4])
        # sklearn's private `_n_samples_bootstrap` (None without bootstrap);
        # the count the fit USED, reported from the far side of the boundary
        # so a check can see the knob reach it (DEVIATION 460).
        self._n_samples_bootstrap = int(meta[5]) or None
        return self

    def _vote(self, X):
        if not hasattr(self, "_offsets"):
            raise RuntimeError("this estimator is not fitted yet")
        Xa, _ = as_f32_c(X, ndim=2, name="X")
        n_rows, n_features = Xa.shape
        if n_features != self.n_features_in_:
            raise ValueError(
                f"X has {n_features} features, fit saw {self.n_features_in_}"
            )
        out = empty((n_rows * self._num_outputs,), "<f4")
        wrote = self._predict_forest("et_predict", Xa, out)
        if wrote != n_rows:
            raise RuntimeError(
                f"et_predict wrote {wrote} of {n_rows} rows"
            )
        return out.reshape((n_rows, self._num_outputs))

    def save(self, path):
        """Write the fitted forest to `path` as an npz.

        The file holds the five prediction arrays exactly as fitted, raw
        bytes and exact dtypes. GPU-parallel archives also retain their
        inference engine and numeric mode in a separately versioned format.
        Floats never pass through decimal text. The bytes of the file itself are a pure function of
        the model (see `_serialize.write_npz`), so equal models give equal
        file hashes across machines.
        """
        if not hasattr(self, "_offsets"):
            raise RuntimeError("this estimator is not fitted yet")
        # DEVIATION 2343: members are `Array`s and plain str (which
        # `_serialize.write_npz` encodes as the `<U` scalar member
        # `np.asarray(str)` gave); `classes` is `_labels.classes_member`
        # (int64 / float64 `Array`, or the list of str for str labels).
        arrays = {
            "format": _MODEL_FORMAT,
            "estimator": type(self).__name__,
            "device": self.device,
            "offsets": self._offsets,
            "colid": self._colid,
            "quesval": self._quesval,
            "left_child": self._left_child,
            "leaves": self._leaves,
            "meta": Array.from_list(
                [
                    int(self.n_features_in_),
                    int(self._n_trees),
                    int(self._num_outputs),
                    1 if self.depth_cap_bound_ else 0,
                    int(self.max_depth_resolved_),
                    int(self.max_features_),
                ],
                "<i8",
            ),
        }
        if hasattr(self, "classes_"):
            arrays["classes"] = classes_member(self.classes_)
        self._archive_inference_metadata(arrays, _MODEL_FORMAT)
        return _serialize.write_npz(path, arrays)

    @classmethod
    def load(cls, path):
        """Load a forest saved by `save`. The result predicts; it does not
        refit. Loading a file saved by the other estimator class raises
        rather than reinterpreting its leaves."""
        arrays = _serialize.read_npz(path, (_MODEL_FORMAT, _MODEL_FORMAT + "-parallel-groves-1"))
        saved_as = _serialize.scalar_str(arrays, "estimator")
        if saved_as != cls.__name__:
            raise ValueError(
                f"mojolearn: {path!r} was saved by {saved_as}, not "
                f"{cls.__name__}"
            )
        obj = cls.__new__(cls)
        obj.device = _serialize.scalar_str(arrays, "device")
        obj._restore_inference_metadata(arrays, _MODEL_FORMAT)
        obj._offsets = _serialize.exact(arrays, "offsets", "<i4")
        obj._colid = _serialize.exact(arrays, "colid", "<i4")
        obj._quesval = _serialize.exact(arrays, "quesval", "<f4")
        obj._left_child = _serialize.exact(arrays, "left_child", "<i4")
        obj._leaves = _serialize.exact(arrays, "leaves", "<f4")
        meta = _serialize.exact(arrays, "meta", "<i8")
        obj.n_features_in_ = int(meta[0])
        obj._n_trees = int(meta[1])
        obj._num_outputs = int(meta[2])
        obj.depth_cap_bound_ = bool(int(meta[3]))
        obj.max_depth_resolved_ = int(meta[4])
        obj.max_features_ = int(meta[5])
        if "classes" in arrays:
            # a 0.6.x file's `classes` member loads too (int, float, bool
            # or `<U` from `np.unique`); bool labels come back as ints
            obj.classes_ = classes_from_member(arrays["classes"])
            obj.n_classes_ = int(len(obj.classes_))
        return obj


@forest_estimator("classifier")
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
        inference_engine="sequential",
    ):
        super().__init__(device)
        _refuse_forest_knobs(n_jobs, verbose)
        if criterion not in _CLASSIFIER_CRITERIA:
            reason = _UNSUPPORTED_CRITERIA.get(criterion)
            raise NotImplementedError(
                f"criterion={criterion!r} is not implemented"
                + (f" ({reason[1]})" if reason else "")
                + "; 'gini', 'entropy' and its alias 'log_loss' are."
            )
        self.criterion = criterion
        self._criterion_code = _CLASSIFIER_CRITERIA[criterion]
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
        self._refresh_config()
        self._capture_fit_mode()
        # DEVIATION 2340: `classes_` is a Python list of the caller's
        # label objects under `_labels.sorted_classes`'s order rule, and
        # the codes are one dict lookup per row (the permitted O(rows)
        # label-encoding loop). The codes cross as float32, as before.
        self.classes_, codes = sorted_classes(flatten_labels(y))
        self.n_classes_ = int(len(self.classes_))
        return self._fit_arrays(
            X,
            Array.from_list([float(c) for c in codes], "<f4"),
            self.n_classes_,
            self._bind("_mojolearn_trees").et_classifier_fit,
        )

    def predict_proba(self, X):
        """Averaged per-tree leaf distributions, `(n_samples,
        n_classes)` float64 `Array` (exact widening of the float32 vote),
        columns in `classes_` order."""
        return self._vote(X).astype("<f8")

    def predict(self, X):
        """The argmax of the vote (first max wins) mapped through
        `classes_`: an int64 or float64 `Array` for numeric labels, a
        Python list for str labels (DEVIATION 2340)."""
        return decode_labels(self.classes_, argmax_rows(self._vote(X)))


@forest_estimator("regressor")
class ExtraTreesRegressor(_ExtraTreesBase):
    """sklearn's `ExtraTreesRegressor`, honoured or refused by name.

    GPU training produces means of fixed-point quantized labels
    (deviation 135). Prediction supports the sequential and parallel_groves
    inference algorithms; there is no CPU training implementation.
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
        inference_engine="sequential",
    ):
        super().__init__(device)
        _refuse_forest_knobs(n_jobs, verbose)
        if criterion != "squared_error":
            reason = _UNSUPPORTED_CRITERIA.get(criterion)
            raise NotImplementedError(
                f"criterion={criterion!r} is not implemented"
                + (f" ({reason[1]})" if reason else "")
                + "; only 'squared_error' is."
            )
        self.criterion = criterion
        self._criterion_code = _CRITERION_MSE
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
        self._refresh_config()
        self._capture_fit_mode()
        ya, _ = as_f32_c(y, ndim=1, name="y")
        return self._fit_arrays(
            X,
            ya,
            0,
            self._bind("_mojolearn_trees").et_regressor_fit,
        )

    def predict(self, X):
        """The forest mean per row, a float64 `Array` of `(n_samples,)`
        (exact widening of the float32 vote)."""
        vote = self._vote(X)  # (n_rows, 1): one output for the regressor
        return vote.reshape((vote.shape[0],)).astype("<f8")
