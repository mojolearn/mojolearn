# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Extremely Randomized Trees on the GPU, mirroring sklearn's surface.

The learner is `extratrees/`: a batched-levelalgo tree builder and
random forest (reference: cuML) with sklearn's `RandomSplitter` semantics on top -- the
histogram-free formulation (Geurts, Ernst & Wehenkel 2006) that neither cuML
nor LightGBM ships on a GPU. The defaults are scikit-learn's
`ExtraTreesClassifier` / `ExtraTreesRegressor` defaults, name for name.

EVERY sklearn PARAMETER IS EITHER HONOURED OR REFUSED BY NAME. None is
accepted and ignored -- a silently dropped option is indistinguishable, from
the caller's side, from one that works. Almost all refusals fire in the Mojo
layer (`extratrees/estimator.mojo`), which is the single place both the host
and device arms resolve their configuration; this wrapper refuses only what
never crosses the boundary: the criteria this implementation does not have
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
  GPU regression uses fixed-point quantized labels (deviation 135). On a
  CPU-ONLY INSTALL (no GPU set, `mojolearn/host/_mojolearn_trees_host.so`
  built; the CPU training lane, 2026-09-14) the same calls route to
  `bindings/_mojolearn_trees_host.mojo`, the device trainer restated on the
  host, whose forests are bit identical to the GPU's on
  `tools/identity_break.py`'s fixtures; `device` still reads `"gpu"` there
  because the fit is the GPU's fit, computed elsewhere.

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
builder's layout, the same as cuML's; `_buffer.as_f32_colmajor`, zero-copy for a
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

from . import _mojolearn_trees, _native_archive_bridge, _native_trees_bridge
from ._labels import argmax_rows, decode_labels, is_bool
from ._mode import NumericModeMixin
from ._forest_protocol import ForestProtocol, forest_estimator

#: The npz model-file format tag `save` writes and `load` requires.
_MODEL_FORMAT = "mojolearn-extratrees-1"

# The compiled saved-model owner, connected once at import: `save` and
# `load` are one native call each (bindings/archive_schema_native.mojo
# writes the schema registered after `_ExtraTreesBase`;
# bindings/host_archive_native.mojo `host_archive_public_forest` builds the
# loaded model).
_ARCHIVE, _ARCHIVE_CONTEXT = _native_archive_bridge.connect()
# The compiled tree-ensemble owner: `fit`, `predict` and `predict_proba` are
# one native call each (bindings/trees_public_forest.mojo).
_TREES, _TREES_CONTEXT = _native_trees_bridge.connect()

# estimator.mojo's max_features sentinels, same values in the same words.
_MF_SQRT = -1
_MF_LOG2 = -2
_MF_ALL = -3

# decisiontree.mojo's CRITERION_* codes, slot 21 of the params list.
_CRITERION_GINI = 0
_CRITERION_ENTROPY = 1
_CRITERION_MSE = 2
_CRITERION_POISSON = 4
_CRITERION_GAMMA = 5
_CRITERION_INVERSE_GAUSSIAN = 6

# The regressor's criteria -> the code. sklearn names `poisson`; `gamma` and
# `inverse_gaussian` are cuML's (RandomForestRegressor spells them the same
# way here). DEVIATION 5610: the three deviances rank on the scaled
# fixed-point label sums (objectives.mojo::regression_deviance_gain).
_REGRESSOR_CRITERIA = {
    "squared_error": _CRITERION_MSE,
    "poisson": _CRITERION_POISSON,
    "gamma": _CRITERION_GAMMA,
    "inverse_gaussian": _CRITERION_INVERSE_GAUSSIAN,
}

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
}


_NONFINITE_REFUSAL = "X contains NaN or infinity; the forest has no missing-value arm"


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


class _ExtraTreesBase(ForestProtocol, NumericModeMixin):
    _BINDING = "_mojolearn_trees"
    def __init__(self, device):
        _validate_device(device)
        self.device = device

    def save(self, path):
        """Write the fitted forest to `path` as an npz file.

        The file holds the prediction arrays exactly as fitted, with exact
        dtypes; floats never pass through decimal text. A model saved with the
        'parallel_groves' inference engine also records its engine and
        numeric mode. The file bytes depend only on the model, so equal models
        give equal file hashes on any machine.
        """
        return _ARCHIVE(self, "archive_save", (path,), _ARCHIVE_CONTEXT)

    @classmethod
    def load(cls, path):
        """Load a forest saved by `save`.

        The loaded model predicts; it does not refit. Loading a file saved by
        a different estimator class raises `ValueError`.
        """
        return _ARCHIVE(cls, "host_archive_load", (path,), _ARCHIVE_CONTEXT)


#: The saved formats `load` reads, and the save schema of both extra-trees
#: classes (bindings/archive_schema_native.mojo): the members, dtypes and
#: refusals the 0.8.55 Python `save` wrote.
_MODEL_FORMATS = (_MODEL_FORMAT, _MODEL_FORMAT + "-parallel-groves-1")
_native_archive_bridge.register(_ExtraTreesBase, _MODEL_FORMATS, (
    _native_archive_bridge.fitted("_offsets"),
    ("format", "format", _MODEL_FORMAT),
    ("estimator", "estimator"),
    ("array", "device", "device"),
    ("array", "offsets", "_offsets"),
    ("array", "colid", "_colid"),
    ("array", "quesval", "_quesval"),
    ("array", "left_child", "_left_child"),
    ("array", "leaves", "_leaves"),
    ("i8", "meta", (("int", "n_features_in_"), ("int", "_n_trees"), ("int", "_num_outputs"),
                    ("bool", "depth_cap_bound_"), ("int", "max_depth_resolved_"),
                    ("int", "max_features_"))),
    ("classes", "classes", "classes_"),
    ("forest_engine", _MODEL_FORMAT),
))


@forest_estimator("classifier")
# Maintainer note (moved from the docstring): `predict_proba` is
# `forest_vote`'s average of per-tree leaf distributions (cuML
# `randomforest.cuh:229-242`); `predict` is its argmax through `classes_`.
class ExtraTreesClassifier(_ExtraTreesBase):
    """Extra-trees classifier with scikit-learn's defaults.

    Extremely randomized trees: each split draws one random threshold per
    candidate feature (scikit-learn's random splitter, Geurts, Ernst and
    Wehenkel 2006) instead of searching for the best one. Defaults are
    scikit-learn's, name for name. Every scikit-learn parameter is either
    honoured or refused by name; none is accepted and ignored.

    `X` is converted to float32; NaN or infinity in `X` raises `ValueError`.
    `y` may hold numbers or strings; NaN labels are refused. Supports the `fast` (Apple GPUs only) and
    `identical` numeric modes through `numeric_mode`.

    Parameters
    ----------
    n_estimators : int, default 100
        Number of trees.
    criterion : {'gini', 'entropy', 'log_loss'}, default 'gini'
        Split criterion; 'log_loss' is an alias of 'entropy'. Other names
        raise `NotImplementedError`.
    max_depth : int or None, default None
        Maximum tree depth; None means unlimited.
    min_samples_split : int, default 2
        Minimum rows needed to split a node.
    min_samples_leaf : int, default 1
        Minimum rows in a leaf.
    min_weight_fraction_leaf : float, default 0.0
        Minimum fraction of the total sample weight (of all rows when no
        ``sample_weight`` is given) required in each leaf, in [0.0, 0.5].
    max_features : {'sqrt', 'log2'}, int, float or None, default 'sqrt'
        Features drawn per split: a name, a count, a fraction, or None for
        all features.
    max_leaf_nodes : int or None, default None
        scikit-learn's best-first growth: an int grows each tree to exactly
        that many leaves, expanding the node with the largest impurity
        improvement first (ties go to the smaller node id). None keeps
        depth-wise growth. The improvement is computed in float32.
    min_impurity_decrease : float, default 0.0
        Minimum impurity decrease required to split.
    bootstrap : bool, default False
        Sample rows with replacement for each tree.
    oob_score : bool, default False
        Only False is accepted; out-of-bag scoring is not implemented.
    n_jobs : None, default None
        Only None is accepted; any other value raises `NotImplementedError`.
    random_state : int or None, default None
        Seed. None means seed 0, so the default fit is deterministic, unlike
        scikit-learn.
    verbose : int, default 0
        Only 0 is accepted; any other value raises `NotImplementedError`.
    warm_start : bool, default False
        Only False is accepted; there is no incremental fit.
    class_weight : dict, 'balanced' or None, default None
        Per-class weights, multiplied into ``sample_weight``.
    ccp_alpha : float, default 0.0
        Only 0.0 is accepted; cost-complexity pruning is not implemented.
    max_samples : int, float or None, default None
        Rows drawn per tree when `bootstrap=True`, resolved as in
        scikit-learn: None means all rows, an int is a count, a float f
        means max(int(f * n_samples), 1). Setting it without bootstrap
        raises an error, as in scikit-learn.
    monotonic_cst : None, default None
        Only None is accepted.
    device : {'gpu'}, default 'gpu'
        Only 'gpu' is accepted; other values raise `ValueError`. On a
        CPU-only install the same calls run the host build of the trainer.
    inference_engine : {'auto', 'sequential', 'parallel_groves'}, default 'auto'
        How `predict` reduces over trees. 'auto' uses 'parallel_groves' in
        `fast` mode and 'sequential' otherwise.
    numeric_mode : {'fast', 'identical'} or None, default None
        Numeric mode for this estimator; None uses the package default.
        ``'fast'`` is available on Apple GPUs only.

    Attributes
    ----------
    classes_ : list
        Class labels in sorted order (numbers by value, strings by code
        point).
    inference_engine_ : str
        The inference engine `predict` runs: 'sequential' or
        'parallel_groves' ('auto' resolved for the fitted numeric mode).
    n_features_in_ : int
        Number of features seen in `fit`.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = (X[:, 0] > 0.5).astype(np.int64)
    >>> model = mojolearn.ExtraTreesClassifier(n_estimators=50).fit(X, y)
    >>> proba = model.predict_proba(X)
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
        inference_engine="auto",
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

    def fit(self, X, y, sample_weight=None):
        """Fit the forest.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, converted to float32.
        y : array-like of shape (n_samples,)
            Targets.
        sample_weight : array-like of shape (n_samples,), default None
            Per-row weights, multiplied into the ``class_weight`` row
            weights.

        Returns
        -------
        self
        """
        return _TREES(self, "trees_forest_fit", (X, y, sample_weight, False, None), _TREES_CONTEXT)

    def _fit_with_tree_start(self, X, y, tree_start=None, sample_weight=None):
        """`fit` of the trees from global tree ID `tree_start` on (a shard of
        a multi-device forest)."""
        return _TREES(self, "trees_forest_fit", (X, y, sample_weight, False, tree_start), _TREES_CONTEXT)

    def predict_proba(self, X):
        """Class probabilities: the average of the trees' leaf class
        distributions.

        Returns
        -------
        Array of shape (n_samples, n_classes), float64
            Columns in `classes_` order.
        """
        return _TREES(self, "trees_forest_predict_proba", (X,), _TREES_CONTEXT)

    def _predict_with_proba(self, X):
        """Return labels and probabilities from one exact vote matrix.

        ``predict`` consumes the float32 vote while ``predict_proba`` exactly
        widens it to float64.  Keeping both views avoids a second complete
        forest traversal in identity/evidence tooling without changing
        either public result.
        """
        proba = self.predict_proba(X)
        return decode_labels(self.classes_, argmax_rows(proba)), proba

    def predict(self, X):
        """Predicted class labels: the argmax of `predict_proba`, first
        maximum winning, mapped through `classes_`.

        Returns
        -------
        Array or list of shape (n_samples,)
            An int64 or float64 `Array` for numeric labels, a Python list for
            string labels.
        """
        return _TREES(self, "trees_forest_predict", (X,), _TREES_CONTEXT)

@forest_estimator("regressor")
# Maintainer note (moved from the docstring): GPU training produces means of
# fixed-point quantized labels (deviation 135).
class ExtraTreesRegressor(_ExtraTreesBase):
    """Extra-trees regressor with scikit-learn's defaults.

    Extremely randomized trees: each split draws one random threshold per
    candidate feature (scikit-learn's random splitter, Geurts, Ernst and
    Wehenkel 2006) instead of searching for the best one. Defaults are
    scikit-learn's, name for name. Every scikit-learn parameter is either
    honoured or refused by name; none is accepted and ignored.

    `X` is converted to float32; NaN or infinity in `X` raises `ValueError`.
    `y` is converted to float32. Leaf values are means of fixed-point
    quantized labels. Supports the `fast` (Apple GPUs only) and
    `identical` numeric modes through `numeric_mode`.

    Parameters
    ----------
    n_estimators : int, default 100
        Number of trees.
    criterion : {'squared_error', 'poisson', 'gamma', 'inverse_gaussian'}, default 'squared_error'
        Split criterion. 'poisson' requires finite `y >= 0` with at least one
        positive value; 'gamma' and 'inverse_gaussian' require finite
        `y > 0`; otherwise `fit` raises `ValueError`. 'absolute_error' and
        'friedman_mse' raise `NotImplementedError`.
    max_depth : int or None, default None
        Maximum tree depth; None means unlimited.
    min_samples_split : int, default 2
        Minimum rows needed to split a node.
    min_samples_leaf : int, default 1
        Minimum rows in a leaf.
    min_weight_fraction_leaf : float, default 0.0
        Minimum fraction of the training rows required in each leaf, in
        [0.0, 0.5].
    max_features : {'sqrt', 'log2'}, int, float or None, default 1.0
        Features drawn per split: a name, a count, a fraction, or None for
        all features.
    max_leaf_nodes : int or None, default None
        scikit-learn's best-first growth: an int grows each tree to exactly
        that many leaves, expanding the node with the largest impurity
        improvement first (ties go to the smaller node id). None keeps
        depth-wise growth. The improvement is computed in float32.
    min_impurity_decrease : float, default 0.0
        Minimum impurity decrease required to split.
    bootstrap : bool, default False
        Sample rows with replacement for each tree.
    oob_score : bool, default False
        Only False is accepted; out-of-bag scoring is not implemented.
    n_jobs : None, default None
        Only None is accepted; any other value raises `NotImplementedError`.
    random_state : int or None, default None
        Seed. None means seed 0, so the default fit is deterministic, unlike
        scikit-learn.
    verbose : int, default 0
        Only 0 is accepted; any other value raises `NotImplementedError`.
    warm_start : bool, default False
        Only False is accepted; there is no incremental fit.
    ccp_alpha : float, default 0.0
        Only 0.0 is accepted; cost-complexity pruning is not implemented.
    max_samples : int, float or None, default None
        Rows drawn per tree when `bootstrap=True`, resolved as in
        scikit-learn: None means all rows, an int is a count, a float f
        means max(int(f * n_samples), 1). Setting it without bootstrap
        raises an error, as in scikit-learn.
    monotonic_cst : None, default None
        Only None is accepted.
    device : {'gpu'}, default 'gpu'
        Only 'gpu' is accepted; other values raise `ValueError`. On a
        CPU-only install the same calls run the host build of the trainer.
    inference_engine : {'auto', 'sequential', 'parallel_groves'}, default 'auto'
        How `predict` reduces over trees. 'auto' uses 'parallel_groves' in
        `fast` mode and 'sequential' otherwise.
    numeric_mode : {'fast', 'identical'} or None, default None
        Numeric mode for this estimator; None uses the package default.
        ``'fast'`` is available on Apple GPUs only.

    Attributes
    ----------
    inference_engine_ : str
        The inference engine `predict` runs: 'sequential' or
        'parallel_groves' ('auto' resolved for the fitted numeric mode).
    n_features_in_ : int
        Number of features seen in `fit`.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = X[:, 0] * 2.0 + X[:, 1]
    >>> model = mojolearn.ExtraTreesRegressor(n_estimators=50).fit(X, y)
    >>> pred = model.predict(X)
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
        inference_engine="auto",
    ):
        super().__init__(device)
        _refuse_forest_knobs(n_jobs, verbose)
        if criterion not in _REGRESSOR_CRITERIA:
            reason = _UNSUPPORTED_CRITERIA.get(criterion)
            raise NotImplementedError(
                f"criterion={criterion!r} is not implemented"
                + (f" ({reason[1]})" if reason else "")
                + "; 'squared_error', 'poisson', 'gamma' and 'inverse_gaussian' are."
            )
        self.criterion = criterion
        self._criterion_code = _REGRESSOR_CRITERIA[criterion]
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

    def fit(self, X, y, sample_weight=None):
        """Fit the forest.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, converted to float32.
        y : array-like of shape (n_samples,)
            Targets.
        sample_weight : array-like of shape (n_samples,), default None
            Per-row weights. Splits and leaf values use the weighted sums,
            as in scikit-learn.

        Returns
        -------
        self
        """
        return _TREES(self, "trees_forest_fit", (X, y, sample_weight, False, None), _TREES_CONTEXT)

    def _fit_with_tree_start(self, X, y, tree_start=None, sample_weight=None):
        """`fit` of the trees from global tree ID `tree_start` on (a shard of
        a multi-device forest)."""
        return _TREES(self, "trees_forest_fit", (X, y, sample_weight, False, tree_start), _TREES_CONTEXT)

    def predict(self, X):
        """Predicted values: the mean of the trees' predictions.

        Returns
        -------
        Array of shape (n_samples,), float64
        """
        return _TREES(self, "trees_forest_predict", (X,), _TREES_CONTEXT)


# The forests' native owner reads these fixed tables (criteria, refusal
# texts, max_features slot codes, deviance criterion codes) and assembles a
# splitter='random' DecisionTree*'s inner forest from these classes.
_ET_TABLES = dict(
    cls_criteria=_CLASSIFIER_CRITERIA, reg_criteria=_REGRESSOR_CRITERIA, unsupported=_UNSUPPORTED_CRITERIA,
    mf_all=_MF_ALL, mf_sqrt=_MF_SQRT, mf_log2=_MF_LOG2, poisson=_CRITERION_POISSON,
    gamma=_CRITERION_GAMMA, inverse_gaussian=_CRITERION_INVERSE_GAUSSIAN,
)
_native_trees_bridge.register(ExtraTreesClassifier, "et_classifier", trees_et_tables=_ET_TABLES,
                              trees_et_classifier=ExtraTreesClassifier, trees_et_regressor=ExtraTreesRegressor)
_native_trees_bridge.register(ExtraTreesRegressor, "et_regressor", trees_et_tables=_ET_TABLES)
